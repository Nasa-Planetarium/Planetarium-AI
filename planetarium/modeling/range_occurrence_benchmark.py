"""Fit recursive occurrence models and evaluate or forecast an exact daily interval."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from filelock import FileLock

from planetarium.modeling.daily_occurrence import (
    ONE_DAY,
    block_series,
    calendar_bin,
    grid_series,
    history_stats,
    initial_state,
    model_predictor,
    observed_events,
    observed_origin,
    recursive_probabilities,
    region_masks,
    region_series,
    region_sides,
)
from planetarium.modeling.daily_occurrence_data import WORK, load_panel
from planetarium.modeling.forecast_benchmark import MODELS, _atomic, _measure, fit

LEARNED = ("logistic", "xgboost")


def _fit_window(dates, origin, first, last, calibration_days) -> tuple[int, int, int]:
    if last is not None and last > dates[origin]:
        raise ValueError(
            "Training and calibration must end at or before the observation origin."
        )
    stop = origin + 1 if last is None else int(dates.searchsorted(last, side="right"))
    if first is None:
        first = min(
            dates[0] + 365 * ONE_DAY, dates[max(2, stop - calibration_days - 1)]
        )
    start = int(dates.searchsorted(first))
    calibration = stop - calibration_days
    if not 2 <= start < calibration < stop:
        raise ValueError(
            "Choose dates with distinct training and calibration periods and at least two context days."
        )
    return start, calibration, stop


def _predictors(panel, base, origin, args):
    used = [model for model in args.models if model in LEARNED]
    if not used:
        return {}, None
    first = np.datetime64(args.train_start, "D") if args.train_start else None
    last = np.datetime64(args.train_end, "D") if args.train_end else None
    window = _fit_window(panel.dates, origin, first, last, args.calibration_days)
    lo, cal, stop = window
    tag = f"{panel.dates[lo]}_{panel.dates[cal]}_{panel.dates[stop - 1] + ONE_DAY}"
    directory = base / "range_models" / tag
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / "fit.lock"):
        names = ("logistic.json", "xgboost.json", "xgboost.ubj")
        if not all((directory / name).is_file() for name in names):
            fit(panel, directory, args.threads, args.device, window=window)
        available = {name: model_predictor(name, directory) for name in used}
    if any(moment > panel.dates[origin] for _, moment in available.values()):
        raise ValueError("A fitted model is newer than the observation origin.")
    return {name: prediction for name, (prediction, _) in available.items()}, directory


def _scores(actual: np.ndarray, probability: np.ndarray) -> dict:
    good = actual >= 0
    if not good.any():
        return {
            "rows": 0,
            "positive_rate": None,
            "brier": None,
            "log_loss": None,
            "pr_auc": None,
        }
    return _measure([(actual[good], probability[good])])


def _daily_rows(panel, origin, dates, region_id, model, probabilities, actual):
    frame = pl.DataFrame(
        {
            "region_id": region_id,
            "model": model,
            "as_of_date": str(panel.dates[origin]),
            "date_local": dates,
            "step": (dates - panel.dates[origin]).astype(int),
            "p_any": probabilities,
        },
        schema_overrides={"as_of_date": pl.String},
    ).with_columns(pl.col("as_of_date").str.to_date())
    if actual is not None:
        frame = frame.with_columns(
            pl.Series("actual", actual).replace(-1, None),
        )
    return frame


def run(args) -> None:
    start = np.datetime64(args.start, "D")
    end = np.datetime64(args.end, "D")
    anchor = np.datetime64(args.as_of, "D") if args.as_of else start - ONE_DAY
    if not anchor < start <= end:
        raise ValueError("Expected as-of < start <= end.")
    if args.forecast_only and args.regions is None:
        raise ValueError("Forecast-only mode needs a regions JSON file.")
    panel = load_panel(args.root, args.work)
    if not args.forecast_only and end > panel.dates[-1]:
        raise ValueError(
            "Evaluation ends after the observation calendar. Use --forecast-only for future dates."
        )
    if args.regions:
        ids, masks = region_masks(
            json.loads(args.regions.read_text(encoding="utf-8")), panel.cells
        )
        sides = region_sides(panel, masks)
    else:
        sides = panel.sides
    origin = observed_origin(panel, anchor, sides)
    lo, hi = (int(i) for i in panel.dates.searchsorted([start, end + ONE_DAY]))
    dates = np.arange(panel.dates[origin] + ONE_DAY, end + ONE_DAY)
    keep = dates >= start
    bins = calendar_bin(dates)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "run.lock"):
        if (output / "run.json").exists():
            raise ValueError(
                "The output already contains a run. Provide a new output directory."
            )
        predictors, directory = _predictors(
            panel, args.work / panel.fingerprint, origin, args
        )
        if args.regions:
            groups = [
                (
                    "region",
                    region_series(
                        panel, masks, sides, origin + 1 if args.forecast_only else hi
                    ),
                )
            ]
        else:
            groups = [("cell", grid_series(panel)), ("box", block_series(panel, 4))]
        scores, daily = [], []
        for grain, series in groups:
            stats = history_stats(series, panel.dates, panel.coverage, origin + 1)
            state = initial_state(series, stats, panel.dates, panel.coverage, origin)
            actual = (
                None if args.forecast_only else observed_events(panel, series, lo, hi)
            )
            for model in args.models:
                predicted = recursive_probabilities(
                    series, stats, state, bins, model, predictors.get(model)
                )[keep]
                if grain == "region":
                    for i, region_id in enumerate(ids):
                        observed = actual[:, i] if actual is not None else None
                        daily.append(
                            _daily_rows(
                                panel,
                                origin,
                                dates[keep],
                                region_id,
                                model,
                                predicted[:, i],
                                observed,
                            )
                        )
                        if observed is not None:
                            scores.append(
                                {
                                    "grain": grain,
                                    "region_id": region_id,
                                    "model": model,
                                    **_scores(observed, predicted[:, i]),
                                }
                            )
                else:
                    scores.append(
                        {
                            "grain": grain,
                            "region_id": None,
                            "model": model,
                            **_scores(actual, predicted),
                        }
                    )
            print(
                f"{grain}: {len(args.models)} models, {int(keep.sum())} dates",
                flush=True,
            )
        if scores:
            _atomic(output / "metrics.csv", pl.DataFrame(scores).write_csv())
        if daily:
            _atomic(output / "predictions.csv", pl.concat(daily).write_csv())
        _atomic(
            output / "run.json",
            json.dumps(
                {
                    "dataset": panel.fingerprint,
                    "as_of": str(panel.dates[origin]),
                    "start": str(start),
                    "end": str(end),
                    "models": args.models,
                    "fitted_models": str(directory) if directory is not None else None,
                    "forecast_only": args.forecast_only,
                },
                indent=2,
            ),
        )
    print(f"Saved {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument(
        "--regions",
        type=Path,
        help="Point, box and union definitions. Omit for worldwide evaluation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--train-start", type=date.fromisoformat, help="First labeled training date."
    )
    parser.add_argument(
        "--train-end",
        type=date.fromisoformat,
        help="Last calibration date, at or before as-of.",
    )
    parser.add_argument("--calibration-days", type=int, default=90)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--forecast-only", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if min(args.threads, args.calibration_days) < 1:
        parser.error("Threads and calibration days must be positive.")
    if len(set(args.models)) != len(args.models):
        parser.error("Each model can be requested once.")
    try:
        run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
