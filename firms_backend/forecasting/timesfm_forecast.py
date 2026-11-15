"""Zero-shot detection and FRP forecasts on the original 1 km grid."""
from datetime import timedelta

import duckdb
import numpy as np

from ..api import _spatial
from . import daily_cells, monthly_cells

MODEL_ID = "google/timesfm-3.0-pytorch"
MODEL_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
METHOD = "timesfm_3"


class TimesFM:
    def __init__(self, checkpoint, threads, batch_size):
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision("high")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        evaluator = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=str(checkpoint), per_core_batch_size=batch_size, device=self.device,
        ))
        self.model = evaluator.model
        self.median = evaluator.config.median_quantile_index
        self.levels = np.r_[0, evaluator.config.quantiles, 1]
        self.horizon_patch = evaluator.config.output_patch_length
        self.mixed_precision = self.device == "cuda" and torch.cuda.is_bf16_supported()

    def _decode(self, contexts, horizon):
        torch = self.torch
        # Cells with identical model inputs share the exact same forward pass.
        contexts = np.ascontiguousarray(contexts, dtype=np.float32)
        keys = contexts.view(np.dtype((np.void, contexts.shape[1] * contexts.itemsize))).ravel()
        _, rows, inverse = np.unique(keys, return_index=True, return_inverse=True)
        decoded = np.empty((len(rows), horizon, len(self.levels) - 2), dtype=np.float32)
        padded_horizon = ((horizon + self.horizon_patch - 1) // self.horizon_patch) * self.horizon_patch
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16,
                                                   enabled=self.mixed_precision):
            for start in range(0, len(rows), self.batch_size):
                batch = slice(start, start + self.batch_size)
                values = torch.from_numpy(contexts[rows[batch], None, :]).to(self.device)
                result = self.model.decode(values, horizon=padded_horizon)
                decoded[batch] = result[:, 0, :horizon].float().cpu().numpy()
        return np.sort(decoded, axis=-1)[inverse]

    def score(self, history, horizon):
        # The mean of the binary detection distribution estimates its event probability.
        quantiles = np.clip(self._decode(history[:, 0, :], horizon), 0, 1)
        padded = np.pad(quantiles, ((0, 0), (0, 0), (1, 1)), mode="edge")
        probability = np.trapezoid(padded, self.levels, axis=-1).astype(np.float32)
        frp = history[:, 1, :]
        has_frp = np.isfinite(frp).any(axis=1)
        strength = np.full_like(probability, np.nan)
        if has_frp.any():
            values, low, high, scale = _normalize_frp(frp[has_frp])
            log_frp = self._decode(values, horizon)[..., self.median] * scale + low
            # Bound long-horizon extrapolation before reversing the logarithm.
            strength[has_frp] = np.expm1(np.clip(log_frp, 0, high))
        return probability, strength


def _normalize_frp(values):
    """Scale each log-FRP history and linearly interpolate its unobserved periods."""
    known = np.isfinite(values)
    low = np.min(values, axis=1, where=known, initial=np.inf, keepdims=True)
    high = np.max(values, axis=1, where=known, initial=-np.inf, keepdims=True)
    scale = np.where(high > low, high - low, 1)
    width = values.shape[1]
    periods = np.arange(width, dtype=np.int16)
    left = np.maximum.accumulate(np.where(known, periods, -1), axis=1)
    right = np.minimum.accumulate(np.where(known, periods, width)[:, ::-1], axis=1)[:, ::-1]
    left = np.where(left < 0, right, left)
    right = np.where(right == width, left, right)
    normalized = (values - low) / scale
    start = np.take_along_axis(normalized, left, axis=1)
    end = np.take_along_axis(normalized, right, axis=1)
    fraction = (periods - left) / np.maximum(right - left, 1)
    return (start + fraction * (end - start)).astype(np.float32), low, high, scale


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
        read = daily_cells.archive(cache)
        candidates = daily_cells.candidate_cells(read, cutoff)
        periods = [read(cutoff - timedelta(days=i)) for i in reversed(range(daily_cells.CONTEXT))]
        targets = [str(cutoff + timedelta(days=h)) for h in range(1, daily_cells.HORIZON + 1)]
        skip, partition, target, lead, p_column = 0, "day", "date_utc", "lead_days", "p_detected"
    else:
        origin = np.datetime64(cutoff, "M")
        cutoff = cutoff.replace(day=14)
        read = monthly_cells.reader(cache)
        candidates = monthly_cells.candidates(read, origin)
        periods = [(keys, frp) for keys, _, _, frp in (
            read(origin - i * monthly_cells.MONTH) for i in reversed(range(1, monthly_cells.CONTEXT + 1))
        )]
        targets = [str(origin + h * monthly_cells.MONTH) for h in range(1, monthly_cells.HORIZON + 1)]
        # The first forecast step is the incomplete origin month.
        skip, partition, target, lead, p_column = 1, "month", "target_month", "lead_months", "p_detected_any"

    cells, coords = _coordinates(candidates, bbox)
    print(f"TimesFM 3.0: scoring {len(cells):,} {kind} candidate cells", flush=True)
    probability = np.empty((len(cells), len(targets)), dtype=np.float32)
    strength = np.empty_like(probability)
    recent = np.empty(len(cells), dtype=np.uint8) if daily else None
    chunk_size = max(32_768, batch_size)
    for i in range(0, len(cells), chunk_size):
        batch = slice(i, i + chunk_size)
        history = _history(cells[batch], periods)
        p, frp = model.score(history, len(targets) + skip)
        probability[batch] = p[:, skip:]
        strength[batch] = frp[:, skip:]
        if daily:
            recent[batch] = history[:, 0, -14:].sum(axis=1, dtype=np.uint8)
        if i % (chunk_size * 8) == 0 or i + chunk_size >= len(cells):
            print(f"Scored {min(i + chunk_size, len(cells)):,}/{len(cells):,} cells", flush=True)

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
