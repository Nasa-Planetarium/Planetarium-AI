"""Benchmark frozen Hugging Face models with fitted daily occurrence heads."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from filelock import FileLock
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from planetarium.modeling.daily_occurrence import block_series, grid_series, usable
from planetarium.modeling.daily_occurrence_data import WORK, load_panel
from planetarium.modeling.forecast_benchmark import (
    LEADS,
    PERIODS,
    _atomic,
    _measure,
    evaluation_origins,
)
from planetarium.modeling.foundation_occurrence_models import (
    CHECKPOINTS,
    infer_quantiles,
)

CONTEXT = 64
LEAD_INDEX = np.asarray(LEADS) - 1
GRAINS = ("cell", "box")


def prepare_inputs(panel, path: Path) -> None:
    """Map identical 64-day inputs to a shared forecast without sampling rows."""
    if path.exists():
        return
    windows = []
    groups = {}
    for grain, series in (
        ("cell", grid_series(panel)),
        ("box", block_series(panel, 4)),
    ):
        complete = usable(panel.coverage, series.sides)
        last_known = {
            side: np.maximum.accumulate(
                np.where(
                    (panel.coverage & side) == side, np.arange(len(panel.dates)), -1
                )
            )
            for side in np.unique(series.sides)
        }
        for period, years in PERIODS.items():
            observations = []
            for year in range(years[0], years[1] + 1):
                for origin in evaluation_origins(panel.dates, period, year):
                    cells = np.flatnonzero(complete[origin])
                    history = np.zeros((len(cells), CONTEXT), dtype=np.uint8)
                    window = np.arange(origin + 1 - CONTEXT, origin + 1)
                    for side, indices in last_known.items():
                        columns = series.sides[cells] == side
                        dates = indices[window]
                        history[columns] = series.events[
                            np.ix_(dates.clip(min=0), cells[columns])
                        ].T * (dates >= 0)
                    windows.append(np.packbits(history, axis=1).view("<u8").ravel())
                    targets = origin + np.asarray(LEADS)
                    future = series.events[np.ix_(targets, cells)].T.astype(np.int8)
                    future[~complete[np.ix_(targets, cells)].T] = -1
                    observations.append(future)
            if not observations:
                raise ValueError(f"No {period} dates available for {grain} evaluation")
            groups[f"{grain}_{period}"] = np.concatenate(observations)
    keys, inverse = np.unique(np.concatenate(windows), return_inverse=True)
    arrays = {"keys": keys}
    offset = 0
    for name, truth in groups.items():
        arrays[f"{name}_key"] = inverse[offset : offset + len(truth)].astype(np.int32)
        arrays[f"{name}_truth"] = truth
        offset += len(truth)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    print(f"{offset:,} region-origins, {len(keys):,} distinct contexts", flush=True)


def history_features(keys: np.ndarray) -> np.ndarray:
    context = np.unpackbits(keys.view(np.uint8).reshape(-1, CONTEXT // 8), axis=1)
    return np.column_stack([context[:, -n:].mean(axis=1) for n in (1, 2, 7, CONTEXT)])


def fit_probability_head(features, indices, truth):
    counts = np.vstack(
        [
            np.bincount(indices[truth == label], minlength=len(features))
            for label in (0, 1)
        ]
    )
    labels, rows = np.nonzero(counts)
    weights = counts[labels, rows]
    scaler = StandardScaler().fit(features[rows], sample_weight=weights)
    model = LogisticRegression(max_iter=200).fit(
        scaler.transform(features[rows]), labels, sample_weight=weights
    )
    p = model.predict_proba(scaler.transform(features))[:, 1]
    return p, {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "fit_rows": int(weights.sum()),
    }


def score(
    path: Path, quantiles: np.ndarray | None, name: str
) -> tuple[pl.DataFrame, dict]:
    """Fit each probability head with the 2023 labels only."""
    results = []
    heads = {}
    with np.load(path) as inputs:
        history_rates = history_features(inputs["keys"])
        for grain in GRAINS:
            tables = {
                period: (
                    inputs[f"{grain}_{period}_key"],
                    inputs[f"{grain}_{period}_truth"],
                )
                for period in PERIODS
            }
            indices, truth = tables["validation"]
            for column, lead in enumerate(LEADS):
                features = history_rates
                if quantiles is not None:
                    features = np.column_stack(
                        [features, quantiles[:, column].clip(-1, 2)]
                    )
                p, heads[f"{grain}_{lead}"] = fit_probability_head(
                    features, indices, truth[:, column]
                )
                for period, (keys, labels_by_lead) in tables.items():
                    if period == "validation":
                        continue
                    y = labels_by_lead[:, column]
                    valid = y >= 0
                    results.append(
                        {
                            "period": period,
                            "grain": grain,
                            "model": name,
                            "step": lead,
                            **_measure([(y[valid], p[keys[valid]])]),
                        }
                    )
    return pl.DataFrame(results), heads


def run(
    model: str,
    folder: Path,
    input_path: Path,
    device: str,
    batch_size: int,
    threads: int,
) -> None:
    metrics_path = folder / f"{model}_metrics.csv"
    if metrics_path.is_file():
        print(f"{model} already scored in {folder}", flush=True)
        return
    start = perf_counter()
    quantiles = None
    metadata = {"context": CONTEXT, "fit_year": 2023, "steps": LEADS}
    if model != "history":
        with np.load(input_path) as arrays:
            keys = arrays["keys"]
        quantiles, runtime = infer_quantiles(
            model, keys, CONTEXT, max(LEADS), LEAD_INDEX, device, batch_size, threads
        )
        repo, revision = CHECKPOINTS[model]
        metadata.update(
            checkpoint=repo,
            revision=revision,
            device=device,
            inference_seconds=round(perf_counter() - start, 2),
            distinct_contexts=len(keys),
            **runtime,
        )
    frame, heads = score(input_path, quantiles, model)
    _atomic(
        folder / f"{model}_heads.json",
        json.dumps(
            {**metadata, "scikit_learn": version("scikit-learn"), "heads": heads},
            indent=2,
        ),
    )
    _atomic(metrics_path, frame.write_csv())
    print(f"{model}: complete in {perf_counter() - start:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("prepare", "history", *CHECKPOINTS))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error("batch size and threads must be positive")
    panel = load_panel(args.root, args.work)
    folder = args.work / panel.fingerprint / "foundation-64-v1"
    folder.mkdir(exist_ok=True)
    input_path = folder / "inputs.npz"
    with FileLock(folder / "prepare.lock"):
        prepare_inputs(panel, input_path)
    if args.model != "prepare":
        with FileLock(folder / f"{args.model}.lock"):
            run(
                args.model,
                folder,
                input_path,
                args.device,
                args.batch_size,
                args.threads,
            )


if __name__ == "__main__":
    main()
