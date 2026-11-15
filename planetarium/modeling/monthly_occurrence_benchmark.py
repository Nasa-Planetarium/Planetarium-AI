"""Score every daily forecast pooled by target month, with monthly observation updates."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
from filelock import FileLock

from planetarium.modeling.daily_occurrence import (
    SEASONS,
    STATE_COUNT,
    add_transition_counts,
    block_series,
    calendar_bin,
    grid_series,
    initial_state,
    model_predictor,
    roll,
    smooth_transitions,
    transition_table,
    usable,
)
from planetarium.modeling.daily_occurrence_data import WORK, load_panel
from planetarium.modeling.forecast_benchmark import (
    CAL,
    MODELS,
    _atomic,
    _measure,
    fit,
)
from planetarium.modeling.foundation_occurrence_benchmark import (
    CONTEXT,
    fit_probability_head,
    history_features,
)
from planetarium.modeling.foundation_occurrence_models import (
    CHECKPOINTS,
    infer_quantiles,
)

MONTH = np.timedelta64(1, "M")
DAY = np.timedelta64(1, "D")
CANDIDATES = ("history", *CHECKPOINTS)


@dataclass(frozen=True)
class Fold:
    month: str
    origin: int
    start: int
    stop: int

    @property
    def horizon(self) -> int:
        return self.stop - self.origin - 1


def folds_for_months(dates, coverage, first, last) -> list[Fold]:
    good = np.flatnonzero(coverage == 3)
    result = []
    for month in np.arange(first, last + MONTH):
        start, stop = np.searchsorted(
            dates,
            [month.astype("datetime64[D]"), (month + MONTH).astype("datetime64[D]")],
        )
        idx = np.searchsorted(good, start) - 1
        if (
            idx < 0
            or good[idx] < CONTEXT - 1
            or stop - start
            != int(
                (
                    (month + MONTH).astype("datetime64[D]")
                    - month.astype("datetime64[D]")
                )
                / DAY
            )
        ):
            raise ValueError(f"{month}: missing history or calendar dates")
        result.append(Fold(str(month), int(good[idx]), int(start), int(stop)))
    return result


def spatial_series(panel):
    return (("cell", grid_series(panel)), ("box", block_series(panel, 4)))


def targets(panel, series, start, stop) -> np.ndarray:
    actual = series.events[start:stop].astype(np.int8)
    actual[~usable(panel.coverage[start:stop], series.sides)] = -1
    return actual


def pack_history(panel, series, origin, last_known):
    days = np.arange(origin - CONTEXT + 1, origin + 1)
    history = np.empty((len(series.sides), CONTEXT), dtype=np.uint8)
    for side, known in last_known.items():
        columns = np.flatnonzero(series.sides == side)
        indices = known[days]
        history[columns] = series.events[np.ix_(indices.clip(min=0), columns)].T * (
            indices >= 0
        )
    return np.packbits(history, axis=1).view("<u8").ravel()


def prepare_foundation(panel, evaluation: list[Fold], path: Path) -> None:
    if path.exists():
        return
    horizon = max(fold.horizon for fold in evaluation)
    calibration = folds_for_months(
        panel.dates,
        panel.coverage,
        np.datetime64(f"{CAL}-01"),
        np.datetime64(f"{CAL}-12"),
    )
    cutoff = calibration[-1].stop
    histories = {}
    arrays = {}
    for grain, series in spatial_series(panel):
        last_known = {
            side: np.maximum.accumulate(
                np.where(
                    (panel.coverage & side) == side, np.arange(len(panel.dates)), -1
                )
            )
            for side in np.unique(series.sides)
        }
        for split, folds in (("eval", evaluation), ("cal", calibration)):
            histories[f"{grain}_{split}"] = np.stack(
                [pack_history(panel, series, f.origin, last_known) for f in folds]
            )
        truth = np.full(
            (len(calibration), horizon, len(series.sides)), -1, dtype=np.int8
        )
        for position, fold in enumerate(calibration):
            first = fold.origin + 1
            actual = targets(panel, series, first, min(first + horizon, cutoff))
            truth[position, : len(actual)] = actual
        arrays[f"{grain}_truth"] = truth.transpose(1, 0, 2)
    keys, inverse = np.unique(
        np.concatenate([h.ravel() for h in histories.values()]), return_inverse=True
    )
    arrays["keys"] = keys
    offset = 0
    for name, history in histories.items():
        arrays[name] = (
            inverse[offset : offset + history.size]
            .reshape(history.shape)
            .astype(np.int32)
        )
        offset += history.size
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    print(
        f"{len(keys):,} distinct histories, {len(evaluation)} test months", flush=True
    )


def measure_month(panel, fold, grain, model, actual, probability) -> dict:
    if actual.shape != probability.shape or len(actual) != fold.stop - fold.start:
        raise ValueError("Probabilities and actuals must include every target date.")
    valid = actual >= 0
    if not valid.any():
        raise ValueError(f"No usable observations in {fold.month}")
    return {
        "month": fold.month,
        "grain": grain,
        "model": model,
        "as_of": str(panel.dates[fold.origin]),
        "first_date": str(panel.dates[fold.start]),
        "last_date": str(panel.dates[fold.stop - 1]),
        "forecast_days": len(actual),
        "days_scored": int(valid.any(axis=1).sum()),
        "rollout_days": fold.horizon,
        "possible_rows": actual.size,
        **_measure([(actual[valid], probability[valid])]),
    }


def _score_transition(
    model, *, panel, fold, grain, series, stats, state, bins, labels, predictors
):
    table = {
        b: transition_table(model, stats, series, int(b), predictors.get(model))
        for b in np.unique(bins)
    }
    offset = fold.start - fold.origin - 1
    probability = roll(state, bins, table.__getitem__)[offset:]
    return measure_month(panel, fold, grain, model, labels, probability)


def run_local(panel, folds, folder, fitted, threads, device) -> None:
    models = ("logistic", "xgboost")
    with FileLock(fitted.parent / "benchmark.lock"):
        if any(
            not (fitted / artifact).is_file()
            for artifact in ("logistic.json", "xgboost.json", "xgboost.ubj")
        ):
            fit(panel, fitted, threads, device)
        checked = {model: model_predictor(model, fitted) for model in models}
    for _, cutoff in checked.values():
        if panel.dates[folds[0].origin] < cutoff:
            raise ValueError(
                "Supervised model calibration extends past the first origin."
            )
    predictors = {model: values[0] for model, values in checked.items()}
    rows = []
    with ThreadPoolExecutor(min(threads, len(MODELS))) as pool:
        for grain, series in spatial_series(panel):
            count = np.zeros(
                (SEASONS, len(series.sides), STATE_COUNT), dtype=np.float32
            )
            hits = np.zeros_like(count)
            cursor = 2
            for fold in folds:
                end_history = fold.origin + 1
                add_transition_counts(
                    series,
                    panel.dates,
                    panel.coverage,
                    cursor,
                    end_history,
                    count,
                    hits,
                )
                cursor = end_history
                stats = smooth_transitions(count, hits)
                state = initial_state(
                    series, stats, panel.dates, panel.coverage, fold.origin
                )
                bins = calendar_bin(panel.dates[end_history : fold.stop])
                labels = targets(panel, series, fold.start, fold.stop)
                score = partial(
                    _score_transition,
                    panel=panel,
                    fold=fold,
                    grain=grain,
                    series=series,
                    stats=stats,
                    state=state,
                    bins=bins,
                    labels=labels,
                    predictors=predictors,
                )
                rows.extend(pool.map(score, MODELS))
                print(
                    f"{grain} {fold.month}: completed {len(MODELS)} models", flush=True
                )
    _atomic(folder / "local_metrics.csv", pl.DataFrame(rows).write_csv())


def run_foundation(panel, folds, folder, model, args) -> None:
    horizon = max(fold.horizon for fold in folds)
    with np.load(folder / "inputs.npz") as arrays:
        keys = arrays["keys"]
        features = history_features(keys)
        quantiles = None
        metadata = {"fit_year": CAL, "horizon": horizon, "context": CONTEXT}
        if model != "history":
            quantiles, runtime = infer_quantiles(
                model,
                keys,
                CONTEXT,
                horizon,
                np.arange(horizon),
                args.device,
                args.batch_size,
                args.threads,
            )
            repo, revision = CHECKPOINTS[model]
            metadata.update(checkpoint=repo, revision=revision, **runtime)
        rows = []
        heads = {}
        for grain, series in spatial_series(panel):
            cal = arrays[f"{grain}_cal"].ravel()
            truth = arrays[f"{grain}_truth"]
            indices = arrays[f"{grain}_eval"]
            probabilities = np.empty((horizon, len(keys)), dtype=np.float32)
            for lead in range(horizon):
                x = (
                    features
                    if quantiles is None
                    else np.column_stack([features, quantiles[:, lead].clip(-1, 2)])
                )
                p, heads[f"{grain}_{lead + 1}"] = fit_probability_head(
                    x, cal, truth[lead].ravel()
                )
                probabilities[lead] = p
            for fold, index in zip(folds, indices, strict=True):
                offset = fold.start - fold.origin - 1
                actual = targets(panel, series, fold.start, fold.stop)
                prediction = probabilities[offset : fold.horizon, index]
                rows.append(
                    measure_month(panel, fold, grain, model, actual, prediction)
                )
            print(f"{model} {grain}: completed {len(folds)} months", flush=True)
        _atomic(
            folder / f"{model}_heads.json", json.dumps({**metadata, "heads": heads})
        )
        _atomic(folder / f"{model}_metrics.csv", pl.DataFrame(rows).write_csv())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("prepare", "local", "report", *CANDIDATES))
    parser.add_argument("--start", default="2025-01")
    parser.add_argument("--end", help="Last calendar month to evaluate, inclusive")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--force", action="store_true", help="Recompute the chosen model"
    )
    args = parser.parse_args()
    panel = load_panel(args.root, args.work)
    last_good = panel.dates[panel.coverage == 3][-1]
    latest = last_good.astype("datetime64[M]")
    if (latest + MONTH).astype("datetime64[D]") - DAY > last_good:
        latest -= MONTH
    first = np.datetime64(args.start, "M")
    last = np.datetime64(args.end, "M") if args.end else latest
    if min(args.batch_size, args.threads) < 1:
        parser.error("batch size and threads must be positive")
    if not np.datetime64(f"{CAL + 1}-01") <= first <= last <= latest:
        parser.error(f"Choose ordered months from {CAL + 1}-01 through {latest}")
    folds = folds_for_months(panel.dates, panel.coverage, first, last)
    base = args.work / panel.fingerprint
    folder = base / f"monthly-v1_{first}_{last}"
    folder.mkdir(exist_ok=True)
    if args.model == "report":
        from planetarium.modeling.monthly_occurrence_report import record

        with FileLock(base / "monthly-report.lock"):
            record(folder)
        return
    if args.model in {"prepare", *CANDIDATES}:
        with FileLock(folder / "prepare.lock"):
            prepare_foundation(panel, folds, folder / "inputs.npz")
    if args.model == "prepare":
        return
    with FileLock(folder / f"{args.model}.lock"):
        if (folder / f"{args.model}_metrics.csv").is_file() and not args.force:
            print(f"{args.model} already completed in {folder}; use --force to rerun")
        elif args.model == "local":
            run_local(panel, folds, folder, base / "models", args.threads, args.device)
        else:
            run_foundation(panel, folds, folder, args.model, args)


if __name__ == "__main__":
    main()
