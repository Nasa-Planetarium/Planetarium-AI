"""Compare one value per region and target calendar month on the v1.2 release."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from joblib import load

from planetarium.config import PROCESSED_DATA_DIR
from planetarium.modeling.value_benchmark import (
    EVAL_START,
    ONE_DAY,
    ONE_MONTH,
    SEED,
    errors,
)
from planetarium.modeling.value_data import (
    LAYOUTS,
    SCALES,
    WORK,
    grid_value_totals,
    load_panel,
)
from planetarium.modeling.value_features import FEATURES, make_batch
from planetarium.modeling.value_models import ZERO_KEEP, fit_model

FIT_START = np.datetime64("2018-01", "M")
LAST_MONTH = "2026-10"
MONTHLY_MODELS = (
    "lightgbm_l1",
    "lightgbm_l1_zero50",
    "lightgbm_l1_zero20",
    "lightgbm_tweedie",
    "xgboost",
    "xgboost_l1",
)
DAILY_COMPARATORS = ("lightgbm_l1_zero20", "xgboost_l1_zero20", "xgboost")


def monthly_labels(root: Path, regions: pl.DataFrame, months: np.ndarray) -> np.ndarray:
    year, month = pl.col("year").cast(pl.Int32), pl.col("month").cast(pl.Int32)
    periods = year * 12 + month - 1 - (int(months[0].astype(int)) + 1970 * 12)
    source = (
        pl.scan_parquet(root / "harmonized_cell_month.parquet")
        .filter(
            pl.col("year").is_between(int(str(months[0])[:4]), int(str(months[-1])[:4]))
        )
        .select(pl.col("cell_id").cast(pl.Int32), "status", "value", period=periods)
        .filter(pl.col("period").is_between(0, len(months) - 1))
    )
    complete = source.filter(
        pl.col("status").is_in(["ok", "nrt"]), pl.col("value").is_not_null()
    ).select("cell_id", "period", "value")
    available = (
        complete.select(pl.col("period").unique().sort()).collect()["period"].to_numpy()
    )
    if not np.array_equal(available, np.arange(len(months))):
        missing = np.setdiff1d(np.arange(len(months)), available)
        raise ValueError(
            f"Full-month labels are unavailable for {', '.join(map(str, months[missing]))}."
        )
    values = np.zeros((len(months), regions.height), dtype=np.float64)
    indexed = regions.with_row_index("ix")
    for scale, layout in zip(SCALES, LAYOUTS, strict=True):
        boxes = indexed.filter(pl.col("scale") == scale)
        totals = grid_value_totals(complete, boxes, layout, "period")
        values[totals["period"].to_numpy(), totals["ix"].to_numpy()] = totals[
            "value"
        ].to_numpy()
    return values


def calendar_window(
    observed_days: np.ndarray, month: np.datetime64, issue_day: int
) -> tuple[int, slice]:
    """Find the observation index and the exact target-month offsets in a forecast."""
    target = month.astype("datetime64[D]")
    previous = (month - ONE_MONTH).astype("datetime64[D]")
    cutoff = min(previous + (issue_day - 1) * ONE_DAY, target - ONE_DAY)
    last = np.searchsorted(observed_days, cutoff, side="right") - 1
    if last < 0:
        raise ValueError(f"No observations by {cutoff} for target month {month}.")
    origin = observed_days[last]
    next_month = (month + ONE_MONTH).astype("datetime64[D]")
    return last, slice(
        int((target - origin) / ONE_DAY) - 1, int((next_month - origin) / ONE_DAY) - 1
    )


def make_month_features(
    target_dates_x: np.ndarray, monthly: np.ndarray, i: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Monthly features use finalized labels through two months before the target.
    recent = monthly[max(0, i - 13) : i - 1]
    scale = 1 + np.maximum(recent.mean(axis=0), 0)
    analog_indices = np.arange(i - 12, max(0, i - 60) - 1, -12)
    analog = np.maximum(monthly[analog_indices], 0)
    median = np.median(analog, axis=0)
    x = np.column_stack(
        (
            target_dates_x.mean(axis=1),
            np.log1p(median / scale),
            np.log1p(analog.mean(axis=0) / scale),
            np.log1p(np.maximum(monthly[i - 2], 0) / scale),
            np.log1p(np.maximum(monthly[i - 4 : i - 1].mean(axis=0), 0) / scale),
            np.log(scale),
        )
    ).astype(np.float32)
    return x, scale.astype(np.float32), median


def score_models(predictions, actual, months, scales, issue_day):
    years = months.astype("datetime64[Y]").astype(int) + 1970
    for model, prediction in predictions.items():
        for year in np.unique(years):
            for scale in SCALES:
                keep = np.ix_(years == year, scales == scale)
                result = errors(actual[keep], prediction[keep])
                n, total = result["n"], result["actual_sum"]
                yield {
                    "as_of_day": issue_day,
                    "model": model,
                    "year": year,
                    "scale": scale,
                    "wape": 100 * result["abs_error"] / total if total > 0 else None,
                    "mae": result["abs_error"] / n if n else None,
                    "bias": 100 * (result["predicted_sum"] / total - 1)
                    if total > 0
                    else None,
                    "n": n,
                }


def compare(panel, months, monthly, issue_day, args, daily_models):
    good = np.flatnonzero(panel.coverage == 3)
    observed_days = panel.dates[good]
    fit = np.flatnonzero((months >= FIT_START) & (months < EVAL_START))
    test = np.flatnonzero(months >= EVAL_START)
    x, scale, seasonal = [], [], []
    daily = {model: [] for model in daily_models}
    for i in np.r_[fit, test]:
        origin_idx, window = calendar_window(observed_days, months[i], issue_day)
        batch = make_batch(panel, int(good[origin_idx]), horizon=window.stop)
        target_x = batch.x[:, window]
        features, region_scales, baseline = make_month_features(target_x, monthly, i)
        x.append(features)
        scale.append(region_scales)
        if months[i] >= EVAL_START:
            seasonal.append(baseline)
            for model, estimator in daily_models.items():
                daily[model].append(
                    estimator.predict(target_x, batch.scale).sum(axis=1)
                )

    x, scale = np.stack(x), np.stack(scale)
    train_x = x[: len(fit)].reshape(-1, x.shape[-1])
    train_scale = scale[: len(fit)].ravel()
    y = np.maximum(monthly[fit], 0).astype(np.float32).ravel() / train_scale
    weight = np.minimum(np.sqrt(train_scale), 8)
    random_rank = np.random.default_rng(SEED).random(len(y))
    forecast_x = x[len(fit) :].reshape(-1, 1, x.shape[-1])
    forecast_scale = scale[len(fit) :].ravel()
    print(
        f"issue day {issue_day}: {len(y):,} training labels, {(y == 0).mean():.2%} zeros",
        flush=True,
    )

    def fit_predict(model):
        keep_zero = ZERO_KEEP.get(model, 1.0)
        keep = (y > 0) | (random_rank < keep_zero)
        w = weight[keep] ** 2 if "_l1" in model else weight[keep]
        fitted = fit_model(model, train_x[keep], y[keep], w, args.threads)
        prediction = fitted.predict(forecast_x, forecast_scale).reshape(len(test), -1)
        return model, prediction

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        predictions = dict(executor.map(fit_predict, MONTHLY_MODELS))
    predictions["seasonal_median"] = np.stack(seasonal)
    predictions.update(
        {f"reuse_{name}": np.stack(values) for name, values in daily.items()}
    )
    return list(
        score_models(
            predictions,
            monthly[test],
            months[test],
            panel.regions["scale"].to_numpy(),
            issue_day,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--out", type=Path, default=WORK / "calendar_value_metrics.csv")
    parser.add_argument("--last-month", default=LAST_MONTH)
    parser.add_argument("--daily-model-dir", type=Path)
    parser.add_argument(
        "--issue-day", type=int, nargs="+", choices=(12, 31), default=(12, 31)
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    last = np.datetime64(args.last_month, "M")
    if last < EVAL_START:
        parser.error("--last-month must be January 2025 or later.")
    months = np.arange(
        FIT_START - 12 * ONE_MONTH, last + ONE_MONTH, dtype="datetime64[M]"
    )
    panel = load_panel(args.root, args.work)
    monthly = monthly_labels(args.root, panel.regions, months)
    daily_models = {}
    if args.daily_model_dir is not None:
        for name in DAILY_COMPARATORS:
            checkpoint = load(args.daily_model_dir / f"{name}.joblib")
            if checkpoint["model"] != name or checkpoint["features"] != list(FEATURES):
                raise ValueError(f"Incompatible daily checkpoint for {name}.")
            if np.datetime64(checkpoint["fit_end"], "M") >= EVAL_START:
                raise ValueError(
                    f"Training of daily checkpoint {name} overlaps the test months."
                )
            daily_models[name] = checkpoint["regressor"]
    rows = []
    for issue_day in dict.fromkeys(args.issue_day):
        rows.extend(compare(panel, months, monthly, issue_day, args, daily_models))
    table = pl.DataFrame(rows).sort("as_of_day", "year", "scale", "model")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.out)
    print(
        table.filter(pl.col("scale") != "0.1deg")
        .group_by("as_of_day", "year", "model")
        .agg(pl.col("wape", "mae", "bias").mean().round(2))
        .sort("as_of_day", "year", "wape")
    )


if __name__ == "__main__":
    main()
