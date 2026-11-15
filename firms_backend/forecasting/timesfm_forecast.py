"""Zero-shot detection and FRP forecasts on the original 1 km grid."""
from datetime import timedelta

import duckdb
import numpy as np

from ..api import _spatial
from . import grid_1km, monthly_cells, year_1km

MODEL_ID = "google/timesfm-3.0-pytorch"
MODEL_REVISION = year_1km.MODEL_REVISION
METHOD = "timesfm_3"


class TimesFM:
    def __init__(self, checkpoint, threads, batch_size):
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision("high")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=str(checkpoint), per_core_batch_size=batch_size, device=self.device,
        ))
        self.levels = np.r_[0, self.model.config.quantiles, 1]

    def score(self, history, horizon):
        has_frp = np.isfinite(history[:, 1, :]).any(axis=1)
        forecasts = list(self.model.predict_batch(
            contexts=list(history), horizon=horizon, return_quantiles=True,
            use_symmetric_averaging=False, sort_quantiles=True,
        ))
        # The mean of the binary detection distribution estimates its event probability.
        quantiles = np.clip(np.stack([item.quantiles[0] for item in forecasts]), 0, 1)
        padded = np.pad(quantiles, ((0, 0), (0, 0), (1, 1)), mode="edge")
        probability = np.trapezoid(padded, self.levels, axis=-1).astype(np.float32)
        log_frp = np.stack([item.forecast[1] for item in forecasts])
        strength = np.where(has_frp[:, None], np.expm1(np.maximum(0, log_frp)), np.nan)
        return probability, strength


def _history(cells, periods):
    contexts = np.zeros((len(cells), 2, len(periods)), dtype=np.float32)
    contexts[:, 1, :] = np.nan
    for t, (keys, frp) in enumerate(periods):
        if not len(keys):
            continue
        index = np.minimum(np.searchsorted(keys, cells), len(keys) - 1)
        found = keys[index] == cells
        contexts[found, 0, t] = 1
        contexts[found, 1, t] = np.log1p(np.maximum(0, frp[index[found]]))
    return contexts


def _coordinates(cells, bbox):
    x = cells % monthly_cells.WIDTH - monthly_cells.X_ORIGIN
    y = cells // monthly_cells.WIDTH - monthly_cells.Y_ORIGIN
    longitude = (np.maximum(-180, np.degrees(x / monthly_cells.X_SCALE)) +
                 np.minimum(180, np.degrees((x + 1) / monthly_cells.X_SCALE))) / 2
    latitude = np.degrees(np.arcsin((np.maximum(-1, y / monthly_cells.Y_SCALE) +
                                      np.minimum(1, (y + 1) / monthly_cells.Y_SCALE)) / 2))
    if bbox is not None:
        _, (west, east, south, north) = _spatial(bbox)
        if west < east:
            within = (longitude >= west) & (longitude <= east)
        else:
            within = (longitude >= west) | (longitude <= east)
        keep = within & (latitude >= south) & (latitude <= north)
        cells, x, y, longitude, latitude = (field[keep] for field in (cells, x, y, longitude, latitude))
    return cells, {"grid_x": x, "grid_y": y, "longitude": longitude, "latitude": latitude}


def forecast(kind, cache, cutoff, output, *, model, bbox, threads, batch_size):
    daily = kind == "daily"
    if daily:
        read = grid_1km.archive(cache)
        candidates = grid_1km.candidate_cells(read, cutoff)
        periods = [read(cutoff - timedelta(days=i)) for i in reversed(range(grid_1km.CONTEXT))]
        targets = [str(cutoff + timedelta(days=h)) for h in range(1, grid_1km.HORIZON + 1)]
        skip, partition, target, lead, p_column = 0, "day", "date_utc", "lead_days", "p_detected"
    else:
        origin = np.datetime64(cutoff, "M")
        cutoff = cutoff.replace(day=14)
        read = monthly_cells.reader(cache)
        candidates = year_1km.candidates(read, origin)
        periods = [(keys, frp) for keys, _, _, frp in (
            read(origin - i * monthly_cells.MONTH) for i in reversed(range(1, year_1km.CONTEXT + 1))
        )]
        targets = [str(origin + h * monthly_cells.MONTH) for h in year_1km.LEADS]
        # The first forecast step is the incomplete origin month.
        skip, partition, target, lead, p_column = 1, "month", "target_month", "lead_months", "p_detected_any"

    cells, coords = _coordinates(candidates, bbox)
    probability = np.empty((len(cells), len(targets)), dtype=np.float32)
    strength = np.empty_like(probability)
    recent = np.empty(len(cells), dtype=np.uint8) if daily else None
    for i in range(0, len(cells), batch_size):
        batch = slice(i, i + batch_size)
        history = _history(cells[batch], periods)
        p, frp = model.score(history, len(targets) + skip)
        probability[batch] = p[:, skip:]
        strength[batch] = frp[:, skip:]
        if daily:
            recent[batch] = history[:, 0, -14:].sum(axis=1, dtype=np.uint8)

    with duckdb.connect(config={"threads": str(threads)}) as db:
        for h, time in enumerate(targets):
            path = output / f"{partition}={time}" / "data.parquet"
            path.parent.mkdir()
            columns = {**coords, p_column: probability[:, h],
                       "frp": np.ma.masked_invalid(strength[:, h])}
            if daily:
                columns.update(observed_in_previous_14d=recent > 0, days_detected_previous_14d=recent)
            db.register("cells", columns)
            flags = ", observed_in_previous_14d, days_detected_previous_14d" if daily else ""
            db.execute(f"""
                COPY (
                    SELECT ?::DATE AS {target}, ?::DATE AS forecast_through_utc, ?::TINYINT AS {lead},
                           concat(grid_x, ':', grid_y) AS grid_id, grid_x, grid_y, longitude, latitude,
                           {p_column}, frp AS frp_noaa20_mw_if_detected_p50 {flags}
                    FROM cells ORDER BY grid_y, grid_x
                ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)
            """, [time if skip == 0 else time + "-01", cutoff, h + 1, str(path)])
            db.unregister("cells")

    return len(cells)
