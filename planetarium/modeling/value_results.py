"""Summarize the monthly forecast scores without averaging monthly ratios."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from planetarium.config import REPORTS_DIR
from planetarium.modeling.value_data import WORK

PARTS = (
    "n",
    "active",
    "actual_sum",
    "predicted_sum",
    "abs_error",
    "sq_error",
    "positive_abs_error",
    "zero_predicted_sum",
)
WINDOW_KEY = ("month", "origin", "window_start", "window_end", "scale", "band")


def with_scores(frame: pl.DataFrame) -> pl.DataFrame:
    n = pl.when(pl.col("n") > 0).then(pl.col("n"))
    actual = pl.when(pl.col("actual_sum") > 0).then(pl.col("actual_sum"))
    active = pl.when(pl.col("active") > 0).then(pl.col("active"))
    zeros = pl.col("n") - pl.col("active")
    zero_count = pl.when(zeros > 0).then(zeros)
    return frame.with_columns(
        wape=pl.col("abs_error") / actual * 100,
        mae=pl.col("abs_error") / n,
        rmse=(pl.col("sq_error") / n).sqrt(),
        bias=(pl.col("predicted_sum") - pl.col("actual_sum")) / actual * 100,
        mae_on_positive=pl.col("positive_abs_error") / active,
        mean_prediction_on_zero=pl.col("zero_predicted_sum") / zero_count,
    )


def summarize(results: Path, out: Path):
    files = sorted(results.glob("metrics_*.csv"))
    if not files:
        raise ValueError(f"No model score files in {results}.")
    frames = [pl.read_csv(file).sort(WINDOW_KEY) for file in files]
    common = (*WINDOW_KEY, "regions", "n", "active", "actual_sum")
    truth = frames[0].select(common)
    if any(not frame.select(common).equals(truth) for frame in frames):
        raise ValueError(
            "The evaluated months, regions or actual values differ across model files."
        )
    parts = pl.concat(frames).with_columns(
        pl.col("month").str.slice(0, 4).alias("year")
    )
    out.mkdir(parents=True, exist_ok=True)
    ranks = with_scores(
        parts.group_by("model", "year", "scale", "band").agg(pl.col(*PARTS).sum())
    ).sort("band", "year", "scale", "wape")
    ranks.write_csv(out / "forecast_value_yearly.csv")
    with_scores(parts).filter(pl.col("band").is_in(["daily", "total30"])).sort(
        "month", "band", "scale", "model"
    ).write_csv(out / "forecast_value_monthly.csv")
    return (
        ranks.group_by("year", "model", "band")
        .agg(
            macro_wape=pl.col("wape").mean(),
            macro_bias=pl.col("bias").mean(),
        )
        .sort("year", "band", "macro_wape")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=WORK / "results")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    board = summarize(args.results, args.out).filter(
        pl.col("band").is_in(["daily", "total30"])
    )
    print(
        board.pivot(on="band", index=["year", "model"], values="macro_wape").sort(
            "year", "daily"
        )
    )


if __name__ == "__main__":
    main()
