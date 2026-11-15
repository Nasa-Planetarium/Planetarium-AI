"""Forecast daily detection and conditional FRP on the harmonized archive's exact 1 km grid."""
import argparse
from collections import defaultdict
from datetime import date, timedelta
from functools import lru_cache
import json
from pathlib import Path

import duckdb
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

WIDTH = 34668
X_ORIGIN = 17334
Y_ORIGIN = 7357
HEIGHT = 14714
X_SCALE = 6371.0088 * np.cos(np.pi / 6)
Y_SCALE = 6371.0088 / np.cos(np.pi / 6)
HORIZON = 7
CONTEXT = 64
CANDIDATE_DAYS = 14
NEIGHBOR_DAYS = 7
TRAIN_ORIGINS = (
    "2024-02-14", "2024-04-14", "2024-06-14", "2024-08-14", "2024-10-14", "2024-12-14",
    "2025-02-14", "2025-04-14", "2025-06-14", "2025-08-14", "2025-10-14", "2025-12-14",
)
VALIDATION_ORIGINS = ("2026-02-14", "2026-04-14", "2026-06-14", "2026-08-14")
TEST_ORIGINS = ("2026-09-14", "2026-09-28", "2026-10-14", "2026-10-28", "2026-11-07")
OFFSETS = tuple((x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if (x, y) != (0, 0))


def extract_days(source, target, last_year, threads):
    target.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={"threads": str(threads), "memory_limit": "11GB",
                                "preserve_insertion_order": "false"}) as db:
        for year in range(2022, last_year + 1):
            rows = db.execute("""
                SELECT date_utc, ((grid_y + 7357) * 34668 + grid_x + 17334)::INT AS cell,
                       frp_noaa20_mw::FLOAT AS frp
                FROM read_parquet(?) WHERE detected ORDER BY date_utc, cell
            """, [str(source / "grid_daily" / f"year={year}" / "month=*/*.parquet")]).fetchnumpy()
            days = rows["date_utc"].astype("datetime64[D]")
            dates, starts = np.unique(days, return_index=True)
            frp = np.ma.filled(rows["frp"], np.nan)
            for day, i, j in zip(dates, starts, np.r_[starts[1:], len(days)], strict=True):
                np.savez(target / f"{day}.npz", ids=rows["cell"][i:j], frp=frp[i:j])
            print(f"Extracted {year}: {len(days):,} exact grid-days", flush=True)


def archive(cache):
    @lru_cache(maxsize=130)
    def read(day):
        with np.load(cache / f"{day}.npz") as file:
            return file["ids"], file["frp"]
    return read


def match(sorted_cells, points):
    indices = np.searchsorted(sorted_cells, points)
    found = (indices < len(sorted_cells)) & (sorted_cells[np.minimum(indices, len(sorted_cells) - 1)] == points)
    return indices[found], found


def shift(cells, dx, dy):
    x = (cells % WIDTH + dx) % WIDTH
    y = cells // WIDTH + dy
    valid = (y >= 0) & (y < HEIGHT)
    return (y[valid] * WIDTH + x[valid]).astype(np.int32), valid


def candidate_cells(read, origin):
    daily = [read(origin - timedelta(days=i))[0] for i in range(CANDIDATE_DAYS)]
    neighborhood = np.unique(np.concatenate(daily[:NEIGHBOR_DAYS]))
    return np.unique(np.concatenate([*daily, *(shift(neighborhood, dx, dy)[0] for dx, dy in OFFSETS)]))


def feature_matrix(read, origin, cells):
    n = len(cells)
    history = np.zeros((n, CONTEXT), dtype=np.uint8)
    last_hit = np.full(n, CONTEXT, dtype=np.uint8)
    last_frp = np.full(n, np.nan, dtype=np.float32)
    frp_sum = np.zeros(n, dtype=np.float32)
    frp_n = np.zeros(n, dtype=np.uint8)
    nearby = np.zeros((n, 3), dtype=np.float32)
    historical = np.zeros((n, 2), dtype=np.float32)

    for i in range(CONTEXT):
        ids, frp = read(origin - timedelta(days=i))
        idx, found = match(cells, ids)
        history[idx, CONTEXT - i - 1] = 1
        last_hit[idx] = np.minimum(last_hit[idx], i)
        known = np.isfinite(frp[found])
        idx_mw = idx[known]
        mw = np.log1p(np.maximum(frp[found][known], 0))
        if i < 30:
            missing = np.isnan(last_frp[idx_mw])
            last_frp[idx_mw[missing]] = mw[missing]
            frp_sum[idx_mw] += mw
            frp_n[idx_mw] += 1
        if i < NEIGHBOR_DAYS:
            for dx, dy in OFFSETS:
                shifted, _ = shift(ids, dx, dy)
                nearby_idx, _ = match(cells, shifted)
                nearby[nearby_idx, 0] += 1
                if i < 3:
                    nearby[nearby_idx, 1] += 1
                if i == 0:
                    nearby[nearby_idx, 2] += 1

    for year in (1, 2):
        for offset in range(-7, 8):
            ids, _ = read(origin - timedelta(days=365 * year + offset))
            idx, _ = match(cells, ids)
            historical[idx, year - 1] += 1

    columns = [history[:, -window:].sum(axis=1, dtype=np.uint8) for window in (1, 3, 7, 14, 30, 64)]
    frp_avg = np.divide(frp_sum, frp_n, out=np.full_like(frp_sum, np.nan), where=frp_n > 0)
    angle = 2 * np.pi * origin.timetuple().tm_yday / 365.25
    longitude = (cells % WIDTH - X_ORIGIN + .5) / X_SCALE
    matrix = np.column_stack([
        *columns, last_hit, nearby, historical, last_frp, frp_avg, frp_n,
        np.sin(longitude), np.cos(longitude), (cells // WIDTH - Y_ORIGIN + .5) / Y_SCALE,
        np.full(n, np.sin(angle)), np.full(n, np.cos(angle)),
    ]).astype(np.float32)
    return matrix, history


def labels(read, origin, cells):
    hit = np.zeros((len(cells), HORIZON), dtype=np.uint8)
    frp = np.full(hit.shape, np.nan, dtype=np.float32)
    for h in range(HORIZON):
        obs, mw = read(origin + timedelta(days=h + 1))
        idx, found = match(cells, obs)
        hit[idx, h] = 1
        frp[idx, h] = mw[found]
    return hit, frp


def with_lead(features, lead):
    return np.column_stack((features, np.full(len(features), lead, dtype=np.float32)))


def sample_observations(read, origins, cap):
    for value in origins:
        origin = date.fromisoformat(value)
        universe = candidate_cells(read, origin)
        random = np.random.default_rng(origin.toordinal())
        cells = np.sort(random.choice(universe, cap, replace=False)) if cap < len(universe) else universe
        features, histories = feature_matrix(read, origin, cells)
        hit, frp = labels(read, origin, cells)
        totals = []
        for h in range(HORIZON):
            actual, _ = read(origin + timedelta(days=h + 1))
            _, found = match(universe, actual)
            totals.append((len(actual), int(found.sum())))
        print(value, f"candidates {len(universe):,}, sampled {len(cells):,}, positive rows {hit.sum():,}", flush=True)
        yield origin, features, histories, hit, frp, np.array(totals)


def fit(read, cap, threads):
    x_detection, y_detection, x_strength, y_strength = [], [], [], []
    for _, features, _, hit, frp, _ in sample_observations(read, TRAIN_ORIGINS, cap):
        for lead in range(1, HORIZON + 1):
            matrix = with_lead(features, lead)
            known = np.isfinite(frp[:, lead - 1])
            x_detection.append(matrix)
            y_detection.append(hit[:, lead - 1])
            x_strength.append(matrix[known])
            y_strength.append(np.log1p(np.maximum(frp[known, lead - 1], 0)))

    settings = dict(n_estimators=145, num_leaves=31, learning_rate=.075, min_child_samples=110,
                    colsample_bytree=.92, max_bin=80, verbosity=-1, random_state=4517, n_jobs=threads)
    detector = LGBMClassifier(objective="binary", **settings)
    intensity = LGBMRegressor(objective="regression_l1", **settings)
    print("Training exact-cell detection and FRP models", flush=True)
    detector.fit(np.concatenate(x_detection), np.concatenate(y_detection))
    intensity.fit(np.concatenate(x_strength), np.concatenate(y_strength))
    return detector.booster_, intensity.booster_


class Foundation:
    def __init__(self, checkpoint, threads):
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        torch.set_float32_matmul_precision("high")
        torch.set_num_threads(threads)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(checkpoint),
                                                  per_core_batch_size=128, device=self.device))

    def predict(self, history, eligible):
        result = np.zeros((len(history), HORIZON), dtype=np.float32)
        indices = np.flatnonzero(eligible)
        for start in range(0, len(indices), 4096):
            chunk = indices[start:start + 4096]
            outputs = self.model.predict_batch(
                contexts=list(history[chunk].astype(np.float32)), horizon=HORIZON,
                return_quantiles=True, use_symmetric_averaging=False,
            )
            for i, response in zip(chunk, outputs, strict=True):
                # The average of forecast quantiles is a score, calibrated against later detections.
                result[i] = np.clip(response.quantiles, 0, 1).mean(axis=1)
        return result


def predict(features, histories, detector, intensity, foundation):
    eligible = features[:, 3] >= 3
    p = np.column_stack([detector.predict(with_lead(features, h)) for h in range(1, HORIZON + 1)])
    mw = np.column_stack([np.maximum(0, np.expm1(intensity.predict(with_lead(features, h))))
                          for h in range(1, HORIZON + 1)])
    ts = foundation.predict(histories, eligible) if foundation else np.zeros_like(p)
    return p, mw, ts, eligible


def logits(p):
    safe = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(safe / (1 - safe))


def calibration_features(p, ts, eligible, use_timesfm):
    lead = np.broadcast_to(np.arange(1, HORIZON + 1), p.shape)
    columns = [logits(p).ravel(), lead.ravel()]
    if use_timesfm:
        flag = np.broadcast_to(eligible[:, None], p.shape)
        columns.extend([flag.ravel(), (flag * ts).ravel(), (flag * logits(p)).ravel()])
    return np.column_stack(columns)


def evaluate_data(read, origins, cap, detector, intensity, foundation):
    values = defaultdict(list)
    for _, features, histories, actual, observed_mw, totals in sample_observations(read, origins, cap):
        p, mw, ts, eligible = predict(features, histories, detector, intensity, foundation)
        keys = ["base", "strength", "timesfm", "eligible", "actual", "observed_mw", "baseline"]
        for name, row in zip(keys, (p, mw, ts, eligible, actual, observed_mw,
                                  np.broadcast_to(features[:, 3:4] / 14, p.shape)), strict=True):
            values[name].append(row)
        values["coverage"].append(totals)
    return {name: np.concatenate(rows) for name, rows in values.items()}


def metrics(actual, p):
    return {
        "brier": float(brier_score_loss(actual.ravel(), p.ravel())),
        "average_precision": float(average_precision_score(actual.ravel(), p.ravel())),
        "log_loss": float(log_loss(actual.ravel(), p.ravel(), labels=[0, 1])),
    }


def apply_calibrators(values, calibrators):
    predictions = {"lightgbm": values["base"]}
    for name, model in calibrators.items():
        x = calibration_features(values["base"], values["timesfm"], values["eligible"], name == "lightgbm_timesfm_3")
        predictions[name] = model.predict_proba(x)[:, 1].reshape(values["base"].shape)
    return predictions


def fit_calibrators(validation, foundation):
    models = {}
    for name in ("lightgbm_calibrated", "lightgbm_timesfm_3"):
        with_timesfm = name == "lightgbm_timesfm_3"
        if with_timesfm and not foundation:
            continue
        x = calibration_features(validation["base"], validation["timesfm"], validation["eligible"], with_timesfm)
        model = LogisticRegression(C=1.0, max_iter=150)
        model.fit(x, validation["actual"].ravel())
        models[name] = model
    return models


def evaluate(values, calibrators, selected):
    predictions = apply_calibrators(values, calibrators)
    predictions["recent_frequency"] = values["baseline"]
    actual = values["actual"]
    observed_mw = values["observed_mw"]
    coverage = values["coverage"]
    known = np.isfinite(observed_mw)
    eligible = values["eligible"]
    return {
        "metrics": {name: metrics(actual, p) for name, p in predictions.items()},
        "selected_by_lead": {str(i + 1): metrics(actual[:, i], predictions[selected][:, i])
                             for i in range(HORIZON)},
        "forecast_cell_days_evaluated": int(actual.size),
        "eligible_cell_days_for_timesfm": int(eligible.sum() * HORIZON),
        "eligible_detection_metrics": {name: metrics(actual[eligible], p[eligible])
                                       for name, p in predictions.items()},
        "observed_grid_days": int(coverage[:, 0].sum()),
        "observed_grid_days_covered": int(coverage[:, 1].sum()),
        "candidate_coverage": float(coverage[:, 1].sum() / coverage[:, 0].sum()),
        "frp_test_rows": int(known.sum()),
        "frp_mw_mae_given_detection": float(np.mean(np.abs(values["strength"][known] - observed_mw[known]))),
        "detection_rate_within_evaluation_sample": float(actual.mean()),
    }


def output_forecast(read, origin, detector, intensity, foundation, calibrators, selected, output, threads):
    cells = candidate_cells(read, origin)
    print(f"Forecast origin {origin}: {len(cells):,} exact 1 km cells", flush=True)
    features, histories = feature_matrix(read, origin, cells)
    detection = np.empty((len(cells), HORIZON), dtype=np.float32)
    strength = np.empty_like(detection)
    use_foundation = foundation if selected == "lightgbm_timesfm_3" else None
    for start in range(0, len(cells), 140_000):
        indices = slice(start, start + 140_000)
        p, mw, ts, eligible = predict(features[indices], histories[indices], detector, intensity, use_foundation)
        values = {"base": p, "timesfm": ts, "eligible": eligible}
        models = {selected: calibrators[selected]} if selected in calibrators else {}
        detection[indices] = apply_calibrators(values, models)[selected]
        strength[indices] = mw
        if start % 560_000 == 0:
            print(f"  scored {min(start + 140_000, len(cells)):,} cells", flush=True)

    x = cells % WIDTH - X_ORIGIN
    y = cells // WIDTH - Y_ORIGIN
    with duckdb.connect(config={"threads": str(threads), "memory_limit": "7GB"}) as db:
        for h in range(HORIZON):
            day = origin + timedelta(days=h + 1)
            path = output / f"day={day}" / "data.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            db.register("cells", {
                "grid_x": x, "grid_y": y,
                "p_detected": detection[:, h], "frp_noaa20_mw_if_detected_p50": strength[:, h],
                "observed_in_previous_14d": features[:, 3] > 0,
                "days_detected_previous_14d": features[:, 3].astype(np.uint8),
            })
            db.execute(f"""
                COPY (
                    SELECT DATE '{day}' AS date_utc, DATE '{origin}' AS forecast_through_utc,
                           {h + 1}::TINYINT AS lead_days,
                           concat(grid_x, ':', grid_y) AS grid_id,
                           grid_x, grid_y,
                           (greatest(-180, degrees(grid_x / {X_SCALE}))
                               + least(180, degrees((grid_x + 1) / {X_SCALE}))) / 2 AS longitude,
                           degrees(asin((greatest(-1, grid_y / {Y_SCALE})
                               + least(1, (grid_y + 1) / {Y_SCALE})) / 2)) AS latitude,
                           p_detected, frp_noaa20_mw_if_detected_p50,
                           observed_in_previous_14d, days_detected_previous_14d
                    FROM cells
                    ORDER BY grid_y,grid_x
                ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)
            """, [str(path)])
            db.unregister("cells")
            print("Saved", day, flush=True)
    return len(cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("final-delivery/dataset"))
    parser.add_argument("--cache", type=Path, required=True, help="Working directory for day-level extracts")
    parser.add_argument("--reuse-cache", action="store_true", help="Use existing day-level extracts for the same archive")
    parser.add_argument("--output", type=Path, default=Path("forecast/results_1km"))
    parser.add_argument("--checkpoint-3", type=Path)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--train-sample", type=int, default=48000)
    parser.add_argument("--evaluation-sample", type=int, default=130000)
    args = parser.parse_args()
    manifest = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    if manifest["grid_km"] != 1 or manifest["daily_key"] != ["date_utc", "grid_id"]:
        raise ValueError("Expected the harmonized 1 km grid-day archive")
    origin = date.fromisoformat(manifest["date_range_utc_inclusive"][1])
    if not args.reuse_cache:
        extract_days(args.data, args.cache, origin.year, args.threads)
    read = archive(args.cache)
    detector, intensity = fit(read, args.train_sample, args.threads)
    foundation = Foundation(args.checkpoint_3, args.threads) if args.checkpoint_3 else None
    calibration_data = evaluate_data(
        read, VALIDATION_ORIGINS, args.evaluation_sample, detector, intensity, foundation)
    calibrators = fit_calibrators(calibration_data, foundation)
    validation_predictions = apply_calibrators(calibration_data, calibrators)
    selected = min(validation_predictions, key=lambda name: metrics(
        calibration_data["actual"], validation_predictions[name])["brier"])
    validation = evaluate(calibration_data, calibrators, selected)
    print("Selected on validation Brier score:", selected, validation["metrics"], flush=True)
    testing = evaluate_data(read, TEST_ORIGINS, args.evaluation_sample, detector, intensity, foundation)
    test = evaluate(testing, calibrators, selected)
    print("Test:", test, flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    n = output_forecast(read, origin, detector, intensity, foundation, calibrators, selected, args.output, args.threads)
    metadata = {
        "input": str(args.data),
        "grid_km": 1, "key": ["date_utc", "grid_id"],
        "forecast_origin_utc": str(origin),
        "date_range_utc_inclusive": [str(origin + timedelta(days=1)), str(origin + timedelta(days=HORIZON))],
        "candidate_cells": n, "forecast_rows": n * HORIZON,
        "candidate_definition": "Last 14 days of occupied archive cells plus the eight direct neighbors of cells occupied in the last seven days",
        "target": "A harmonized hotspot detection is recorded in the cell on the UTC date",
        "strength_target": "Median NOAA-20-equivalent FRP in MW conditional on a detectable record with available FRP",
        "selected_detector": selected,
        "timesfm_comparison": "google/timesfm-3.0-pytorch; 64 daily occupancy values; at least three occupied days in the last 14"
        if foundation else None,
        "train_origins": TRAIN_ORIGINS, "validation_origins": VALIDATION_ORIGINS, "test_origins": TEST_ORIGINS,
        "validation": validation, "test": test,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
