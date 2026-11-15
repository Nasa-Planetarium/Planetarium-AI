"""Predict a year of monthly detections on the original harmonized 1 km cells."""
import argparse
from collections import defaultdict
from datetime import date
import json
from pathlib import Path
import tempfile

import duckdb
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from .monthly_cells import MONTH, WIDTH, X_ORIGIN, Y_ORIGIN, X_SCALE, Y_SCALE, date_label, reader

CONTEXT = 36
LEADS = np.arange(1, 13)
RECURRENT_MIN = 18
TRAIN = ("2022-01", "2022-05", "2022-09", "2023-01", "2023-04", "2023-07",
         "2023-10", "2024-01", "2024-04", "2024-07", "2024-10")
VALID = ("2024-11", "2024-12", "2025-03", "2025-06", "2025-09")
TEST = ("2025-10", "2025-12", "2026-03", "2026-06", "2026-09")
MODEL_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


def match(cells, keys):
    idx = np.searchsorted(cells, keys)
    targets = cells[np.minimum(idx, len(cells) - 1)]
    valid = (idx < len(cells)) & (targets == keys)
    return idx[valid], valid


def candidates(read, origin):
    older = [read(origin - i * MONTH)[0] for i in range(1, CONTEXT + 1)]
    cells, counts = np.unique(np.concatenate(older), return_counts=True)
    now_cells, _, first14, _ = read(origin)
    return np.unique(np.concatenate([cells[counts >= 2], older[0], now_cells[first14 > 0]]))


def history_for(read, origin, cells):
    days = np.zeros((len(cells), CONTEXT), dtype=np.uint8)
    frp_last = np.full(len(cells), np.nan, dtype=np.float32)
    frp_total = np.zeros(len(cells), dtype=np.float32)
    frp_counts = np.zeros(len(cells), dtype=np.uint8)
    since = np.full(len(cells), CONTEXT, dtype=np.uint8)
    for lag in range(1, CONTEXT + 1):
        keys, counts, _, frp = read(origin - lag * MONTH)
        idx, found = match(cells, keys)
        days[idx, -lag] = counts[found]
        since[idx] = np.minimum(since[idx], lag - 1)
        if lag <= 12:
            valid = np.isfinite(frp[found])
            target = idx[valid]
            value = np.log1p(np.maximum(0, frp[found][valid]))
            unfilled = np.isnan(frp_last[target])
            frp_last[target[unfilled]] = value[unfilled]
            frp_counts[target] += 1
            frp_total[target] += value
    keys, _, first14, _ = read(origin)
    idx, found = match(cells, keys)
    partial = np.zeros(len(cells), dtype=np.uint8)
    partial[idx] = first14[found]
    recorded = days > 0
    active = [recorded[:, -window:].sum(axis=1, dtype=np.uint8) for window in (3, 6, 12, 24, 36)]
    totals = [days[:, -window:].sum(axis=1, dtype=np.int16) for window in (3, 6, 12, 36)]
    longitude = ((cells % WIDTH) - X_ORIGIN + .5) / X_SCALE
    mean_frp = np.divide(frp_total, frp_counts, out=np.full(len(cells), np.nan, dtype=np.float32), where=frp_counts > 0)
    shared = np.column_stack([
        partial, since, *active, *totals, days[:, -1], frp_last, mean_frp, frp_counts,
        np.sin(longitude), np.cos(longitude), ((cells // WIDTH) - Y_ORIGIN + .5) / Y_SCALE,
    ]).astype(np.float32)
    return days, shared, active[-1] >= RECURRENT_MIN


def monthly_features(days, shared, origin, lead):
    target = origin + lead * MONTH
    angle = 2 * np.pi * (target.astype(int) % 12) / 12
    n = len(days)
    same_month = [days[:, lead - 12 * year].astype(np.float32) if lead < 12 * year
                  else np.full(n, np.nan, dtype=np.float32) for year in (1, 2, 3)]
    prior_rate = sum(value > 0 for value in same_month) / (2 if lead == 12 else 3)
    matrix = np.empty((n, shared.shape[1] + 7), dtype=np.float32)
    matrix[:, :shared.shape[1]] = shared
    matrix[:, -7:] = np.column_stack([
        *same_month, prior_rate, np.full(n, lead), np.full(n, np.sin(angle)), np.full(n, np.cos(angle)),
    ])
    return matrix, prior_rate


def openings(read, names, first, last, cap):
    for name in names:
        origin = np.datetime64(name, "M")
        leads = [lead for lead in LEADS if first <= origin + lead * MONTH <= last]
        if not leads:
            continue
        pool = candidates(read, origin)
        seed = np.random.default_rng(int(origin.astype(int)) + 412845)
        cells = np.sort(seed.choice(pool, cap, replace=False)) if cap < len(pool) else pool
        history, common, eligible = history_for(read, origin, cells)
        print(name, f"{len(pool):,} candidate cells, {len(cells):,} sampled, leads {leads[0]}–{leads[-1]}", flush=True)
        yield origin, leads, pool, cells, history, common, eligible


def actual_at(read, cells, target):
    ids, _, _, frp = read(target)
    index, found = match(cells, ids)
    present = np.zeros(len(cells), dtype=np.uint8)
    mw = np.full(len(cells), np.nan, dtype=np.float32)
    present[index] = 1
    mw[index] = frp[found]
    return present, mw, ids


def train(read, cap, threads):
    features, outcomes, strength_features, strength_targets = [], [], [], []
    for origin, leads, _, cells, history, shared, _ in openings(
            read, TRAIN, np.datetime64("2022-02", "M"), np.datetime64("2024-12", "M"), cap):
        for lead in leads:
            x, _ = monthly_features(history, shared, origin, lead)
            hit, frp, _ = actual_at(read, cells, origin + lead * MONTH)
            known = np.isfinite(frp)
            features.append(x)
            outcomes.append(hit)
            strength_features.append(x[known])
            strength_targets.append(np.log1p(np.maximum(frp[known], 0)))
    settings = dict(n_estimators=145, num_leaves=35, max_bin=90, learning_rate=.07,
                    min_child_samples=160, colsample_bytree=.92, reg_lambda=.5,
                    n_jobs=threads, random_state=4218, verbosity=-1)
    det = LGBMClassifier(objective="binary", **settings)
    mw = LGBMRegressor(objective="regression_l1", **settings)
    print("Training 12-lead detection and FRP models", flush=True)
    det.fit(np.concatenate(features), np.concatenate(outcomes))
    mw.fit(np.concatenate(strength_features), np.concatenate(strength_targets))
    return det.booster_, mw.booster_


class Foundation:
    def __init__(self, path, threads):
        import torch
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision("high")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=str(path), per_core_batch_size=128, device=self.device))
        print("TimesFM 3.0 loaded", self.device, flush=True)

    def score(self, history, eligible):
        active = np.flatnonzero(eligible)
        output = np.zeros((len(history), len(LEADS)), dtype=np.float32)
        for start in range(0, len(active), 4096):
            batch = active[start:start + 4096]
            contexts = (history[batch] > 0).astype(np.float32)
            rows = self.model.predict_batch(
                contexts=list(contexts), horizon=len(LEADS) + 1,
                return_quantiles=True, use_symmetric_averaging=False,
            )
            for index, row in zip(batch, rows, strict=True):
                # Step one lands in the unfinished origin month, which is not forecast here.
                output[index] = np.clip(row.quantiles[1:], 0, 1).mean(axis=1)
        return output


def det_features(p, lead, score=None, eligible=None):
    safe = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(safe / (1 - safe))
    return np.column_stack((logit, lead)) if score is None else np.column_stack(
        (logit, lead, eligible, score * eligible, logit * eligible)
    )


def evaluate_data(read, names, first, last, cap, detector, intensity, foundation):
    output = defaultdict(list)
    for origin, leads, pool, cells, days, shared, eligible in openings(read, names, first, last, cap):
        ts = foundation.score(days, eligible) if foundation else np.zeros((len(cells), len(LEADS)), dtype=np.float32)
        for lead in leads:
            x, prior = monthly_features(days, shared, origin, lead)
            hit, frp, target_ids = actual_at(read, cells, origin + lead * MONTH)
            p = detector.predict(x)
            mw = np.maximum(0, np.expm1(intensity.predict(x)))
            for key, values in (
                ("p", p), ("mw", mw), ("ts", ts[:, lead - 1]), ("eligible", eligible),
                ("hit", hit), ("frp", frp), ("seasonal", prior), ("lead", np.full(len(cells), lead, dtype=np.uint8)),
            ):
                output[key].append(values)
            _, included = match(pool, target_ids)
            output["coverage"].append((len(target_ids), int(included.sum())))
    return {key: np.concatenate(parts) if key != "coverage" else np.array(parts) for key, parts in output.items()}


def fit_calibration(panel, foundation):
    calibrators = {}
    for name in ("lightgbm_calibrated", "lightgbm_timesfm_3"):
        coupled = name == "lightgbm_timesfm_3"
        if coupled and not foundation:
            continue
        features = det_features(panel["p"], panel["lead"], panel["ts"], panel["eligible"]) if coupled else det_features(panel["p"], panel["lead"])
        calibrators[name] = LogisticRegression(max_iter=160).fit(features, panel["hit"])
    return calibrators


def metrics(actual, p):
    return {"brier": float(brier_score_loss(actual, p)),
            "average_precision": float(average_precision_score(actual, p)),
            "log_loss": float(log_loss(actual, p, labels=[0, 1]))}


def predict_panel(panel, calibrators):
    predictions = {"lightgbm": panel["p"], "same_month_last_3_years": panel["seasonal"]}
    for name, calibration in calibrators.items():
        features = (det_features(panel["p"], panel["lead"], panel["ts"], panel["eligible"])
                    if name == "lightgbm_timesfm_3" else det_features(panel["p"], panel["lead"]))
        predictions[name] = calibration.predict_proba(features)[:, 1]
    return predictions


def summarize(panel, predictions, chosen):
    known = np.isfinite(panel["frp"])
    observed, covered = panel["coverage"].sum(axis=0)
    hit = panel["hit"]
    reports = {name: metrics(hit, p) for name, p in predictions.items()}
    by_horizon = {}
    for lead in LEADS:
        mask = panel["lead"] == lead
        by_horizon[str(lead)] = {**metrics(hit[mask], predictions[chosen][mask]), "rows": int(mask.sum())}
    return {
        "models": reports, "selected_by_lead": by_horizon, "candidate_coverage": float(covered / observed),
        "observed_cell_months": int(observed), "observed_cell_months_in_candidates": int(covered),
        "scored_sample_rows": len(hit), "sample_prevalence": float(hit.mean()),
        "timesfm_eligible_sample_rows": int(panel["eligible"].sum()),
        "conditional_frp_mae_mw": float(np.abs(panel["frp"][known] - panel["mw"][known]).mean()),
        "conditional_frp_sample_rows": int(known.sum()),
    }


def forecast(read, origin, cutoff, detector, intensity, foundation, calibrators, chosen, directory, threads):
    cells = candidates(read, origin)
    n = len(cells)
    print(f"Forecasting {n:,} original 1 km cells", flush=True)
    days, common, eligible = history_for(read, origin, cells)
    combined = chosen == "lightgbm_timesfm_3"
    ts = foundation.score(days, eligible) if combined else None
    x = cells % WIDTH - X_ORIGIN
    y = cells // WIDTH - Y_ORIGIN
    longitude = (np.maximum(-180, np.degrees(x / X_SCALE)) +
                 np.minimum(180, np.degrees((x + 1) / X_SCALE))) / 2
    latitude = np.degrees(np.arcsin((np.maximum(-1, y / Y_SCALE) +
                                      np.minimum(1, (y + 1) / Y_SCALE)) / 2))
    p = np.empty(n, dtype=np.float32)
    frp = np.empty_like(p)
    with duckdb.connect(config={"threads": str(threads), "memory_limit": "8GB",
                                "preserve_insertion_order": "false"}) as db:
        for lead in LEADS:
            for offset in range(0, n, 300000):
                batch = slice(offset, offset + 300000)
                matrix, _ = monthly_features(days[batch], common[batch], origin, lead)
                probability = detector.predict(matrix)
                if chosen in calibrators:
                    h = np.full(len(probability), lead, dtype=np.float32)
                    features = (det_features(probability,h,ts[batch,lead - 1],eligible[batch])
                                if combined else det_features(probability,h))
                    probability = calibrators[chosen].predict_proba(features)[:, 1]
                p[batch] = probability
                frp[batch] = np.maximum(0, np.expm1(intensity.predict(matrix)))
            month = origin + lead * MONTH
            path = directory / f"month={month}" / "data.parquet"
            path.parent.mkdir(parents=True)
            db.register("incoming", {
                "grid_x": x, "grid_y": y, "longitude": longitude, "latitude": latitude,
                "p_detected_any": p, "frp_noaa20_mw_if_detected_p50": frp,
            })
            db.execute(f"""
                COPY (
                    SELECT DATE '{date_label(month)}' AS target_month,
                           DATE '{cutoff}' AS forecast_through_utc,
                           {lead}::TINYINT AS lead_months, concat(grid_x, ':', grid_y) AS grid_id, *
                    FROM incoming ORDER BY grid_y, grid_x
                ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)
            """, [str(path)])
            db.unregister("incoming")
            print(month, "saved", int((p >= .2).sum()), "cells at or above 20%", flush=True)
    return n, int(eligible.sum()) if combined else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint-3", type=Path)
    parser.add_argument("--output", type=Path, default=Path("forecast/results_1km_year"))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--train-sample", type=int, default=42000)
    parser.add_argument("--evaluation-sample", type=int, default=65000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.cache / "metadata.json").read_text())
    last = date.fromisoformat(manifest["input_through_utc"])
    origin = np.datetime64(last, "M")
    cutoff = last.replace(day=14)
    if last < cutoff:
        raise ValueError("At least the first 14 days of the origin month are needed")
    read = reader(args.cache)
    detector, intensity = train(read, args.train_sample, args.threads)
    foundation = Foundation(args.checkpoint_3, args.threads) if args.checkpoint_3 else None

    validation = evaluate_data(read, VALID, np.datetime64("2025-01", "M"), np.datetime64("2025-12", "M"),
                               args.evaluation_sample, detector, intensity, foundation)
    calibrators = fit_calibration(validation, foundation)
    comparisons = predict_panel(validation, calibrators)
    chosen = min((name for name in comparisons if name != "same_month_last_3_years"),
                 key=lambda name: brier_score_loss(validation["hit"], comparisons[name]))
    validation_report = summarize(validation, comparisons, chosen)
    print("Selected on validation Brier", chosen, validation_report["models"], flush=True)
    test = evaluate_data(read, TEST, np.datetime64("2026-01", "M"), np.datetime64("2026-10", "M"),
                         args.evaluation_sample, detector, intensity, foundation)
    test_report = summarize(test, predict_panel(test, calibrators), chosen)
    print("Holdout:", test_report["models"], "coverage", test_report["candidate_coverage"], flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        cells, recurrent = forecast(read, origin, cutoff, detector, intensity, foundation, calibrators,
                                    chosen, stage, args.threads)
        metadata = {
            "grid_km": 1, "key": ["target_month", "grid_id"],
            "forecast_origin_utc": str(cutoff),
            "target_months_inclusive": [str(origin + MONTH), str(origin + len(LEADS) * MONTH)],
            "candidate_cells_per_month": cells, "forecast_rows": cells * len(LEADS),
            "candidate_definition": "Observed in >=2 of the last 36 complete months, in the last complete month, or during days 1–14 of the origin month",
            "detection_target": "At least one recorded detection in the original 1 km grid cell during the UTC calendar month",
            "strength_target": "Conditional forecast of the median daily NOAA-20-equivalent FRP in MW during the target month",
            "selected_detector": chosen,
            "timesfm_checkpoint": {"id": "google/timesfm-3.0-pytorch", "revision": MODEL_REVISION, "device": foundation.device}
            if foundation else None,
            "timesfm_eligibility": f"At least {RECURRENT_MIN} active months in the last {CONTEXT} complete months",
            "timesfm_cells_scored": recurrent,
            "train_origin_months": TRAIN, "validation_origin_months": VALID, "test_origin_months": TEST,
            "label_periods": {"training_through": "2024-12", "validation": "2025", "holdout": ["2026-01", "2026-10"]},
            "validation": validation_report, "test": test_report,
            "partition_glob": "month=*/data.parquet",
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        stage.rename(args.output)


if __name__ == "__main__":
    main()
