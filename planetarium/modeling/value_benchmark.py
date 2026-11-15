"""Train and compare fixed 30-day forecasts at successive monthly origins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from filelock import FileLock

from planetarium.config import PROCESSED_DATA_DIR
from planetarium.modeling.value_data import SCALES, WORK, ValuePanel, load_panel
from planetarium.modeling.value_features import (
    FEATURES,
    HORIZON,
    ForecastBatch,
    make_batch,
)
from planetarium.modeling.value_models import (
    BASELINES,
    FOUNDATION,
    MODELS,
    ZERO_KEEP,
    fit_model,
)

FIT_END = np.datetime64("2024-12-31")
MIN_ORIGIN = np.datetime64("2018-01-01")
CAL_START = np.datetime64("2024-01")
EVAL_START = np.datetime64("2025-01")
SEED = 1126
ONE_DAY = np.timedelta64(1, "D")
ONE_MONTH = np.timedelta64(1, "M")


def labels(panel: ValuePanel, origin: int) -> np.ndarray:
    return np.asarray(panel.values[origin + 1 : origin + HORIZON + 1]).T


def rolling_origins(panel: ValuePanel, test_start=EVAL_START, step: int = 1):
    """Use the latest globally available observation on or before each month end."""
    good = np.flatnonzero(panel.coverage == 3)
    for month in np.arange(
        test_start,
        panel.dates[-1].astype("datetime64[M]") + ONE_MONTH,
        step * ONE_MONTH,
        dtype="datetime64[M]",
    ):
        anchor = month.astype("datetime64[D]") - ONE_DAY
        candidates = good[panel.dates[good] <= anchor]
        if not len(candidates):
            continue
        origin = int(candidates[-1])
        month_end_of_horizon = month.astype("datetime64[D]") + (HORIZON - 1) * ONE_DAY
        if (
            origin + HORIZON < len(panel.dates)
            and month_end_of_horizon <= panel.dates[good[-1]]
            and (panel.coverage[origin + 1 : origin + HORIZON + 1] == 3).any()
        ):
            yield str(month), origin


def training_rows(panel: ValuePanel, fit_end=FIT_END, min_origin=MIN_ORIGIN):
    """Spread training examples over regions, seasons and leads without using future state."""
    rng = np.random.default_rng(SEED)
    eligible = np.flatnonzero(
        (panel.coverage == 3)
        & (panel.dates >= min_origin)
        & (panel.dates + HORIZON * ONE_DAY <= fit_end)
    )
    x_rows, y_rows, weights = [], [], []
    for origin in eligible[::21]:
        batch = make_batch(panel, int(origin))
        actual = labels(panel, origin)
        # Five uniformly spread leads per region rather than 30 correlated rows.
        region = rng.choice(
            len(batch.scale), min(4500, len(batch.scale)), replace=False
        )[:, None]
        leads = rng.integers(6, size=region.shape) + np.arange(0, HORIZON, 6)
        target = actual[region, leads]
        known = np.isfinite(target)
        x_rows.append(batch.x[region, leads][known])
        y_rows.append((target / batch.scale[region])[known])
        weight = np.minimum(np.sqrt(batch.scale), 8)
        weights.append(np.broadcast_to(weight[region], target.shape)[known])
    if not x_rows:
        raise ValueError("No usable training rows before the specified fit end.")
    return np.concatenate(x_rows), np.concatenate(y_rows), np.concatenate(weights)


def prepare_training(panel: ValuePanel, path: Path):
    with FileLock(path / "training.lock"):
        if not (path / "training.ready").exists():
            x, y, w = training_rows(panel)
            if (y < 0).any():
                raise ValueError("The selected daily values contain a negative label.")
            np.save(path / "x.npy", x)
            np.save(path / "y.npy", y)
            np.save(path / "w.npy", w)
            (path / "training.ready").touch()
    return tuple(np.load(path / f"{key}.npy", mmap_mode="r") for key in ("x", "y", "w"))


def save_batch(panel: ValuePanel, origin: int, path: Path):
    dest = path / f"window_{origin}"
    dest.mkdir(exist_ok=True)
    with FileLock(dest / "prepare.lock"):
        if not (dest / "ready").exists():
            batch = make_batch(panel, origin)
            for key in ("x", "scale", "seasonal", "recent", "context"):
                np.save(dest / f"{key}.npy", getattr(batch, key))
            (dest / "ready").touch()
    return ForecastBatch(
        *(
            np.load(dest / f"{key}.npy", mmap_mode="r")
            for key in ("x", "scale", "seasonal", "recent", "context")
        )
    )


def baseline(name: str, batch: ForecastBatch):
    if name == "seasonal":
        return batch.seasonal
    history = batch.recent if name == "mean30" else np.zeros_like(batch.recent)
    return np.broadcast_to(history[:, None], batch.seasonal.shape)


def foundation_features(quantiles: np.ndarray, batch: ForecastBatch) -> np.ndarray:
    # Project forecast quantiles to a locally fitted mean and let calibration
    # blend in climatology and recent activity without using future labels.
    return np.concatenate(
        (
            np.maximum(quantiles, 0) / batch.scale[:, None, None],
            (batch.seasonal / batch.scale[:, None])[..., None],
            np.broadcast_to(
                (batch.recent / batch.scale)[:, None, None], (*batch.seasonal.shape, 1)
            ),
        ),
        axis=-1,
    )


def fit_foundation(infer, panel: ValuePanel, path: Path):
    from scipy.optimize import nnls

    xtx = np.zeros((len(SCALES), 5, 5), dtype=np.float64)
    xty = np.zeros((len(SCALES), 5), dtype=np.float64)
    keys = panel.regions["scale"].to_numpy()
    for month, origin in rolling_origins(panel, CAL_START, step=3):
        if panel.dates[origin] + HORIZON * ONE_DAY > FIT_END:
            break
        batch = save_batch(panel, origin, path)
        x = foundation_features(infer(batch.context), batch)
        actual = labels(panel, origin)
        for k, scale in enumerate(SCALES):
            known = np.isfinite(actual) & (keys[:, None] == scale)
            weight = np.broadcast_to(
                np.minimum(np.sqrt(batch.scale), 8)[:, None], actual.shape
            )
            y = (actual / batch.scale[:, None])[known]
            z = x[known].astype(np.float64)
            weighted = z * weight[known, None]
            xtx[k] += z.T @ weighted
            xty[k] += weighted.T @ y
        print(f"calibration {month}", flush=True)
    coefficients = []
    for a, b in zip(xtx, xty, strict=True):
        if not a.any():
            raise ValueError("Missing calibration history.")
        r = np.linalg.cholesky(a + np.eye(len(a)) * 10)
        coefficients.append(nnls(r.T, np.linalg.solve(r, b))[0])
    return np.stack(coefficients)


def errors(actual, prediction):
    truth = actual[np.isfinite(actual)].astype(np.float64)
    forecast = prediction[np.isfinite(actual)].astype(np.float64)
    difference = forecast - truth
    active = truth > 0
    return {
        "n": len(truth),
        "active": int(active.sum()),
        "actual_sum": truth.sum(),
        "predicted_sum": forecast.sum(),
        "abs_error": np.abs(difference).sum(),
        "sq_error": np.square(difference).sum(),
        "positive_abs_error": np.abs(difference[active]).sum(),
        "zero_predicted_sum": forecast[~active].sum(),
    }


def window_metrics(panel, month, origin, name, actual, prediction):
    start = str(panel.dates[origin] + ONE_DAY)
    end = str(panel.dates[origin] + HORIZON * ONE_DAY)
    if prediction.shape != actual.shape or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid forecast shape or values for {name} in {month}.")
    for scale in SCALES:
        selected = panel.regions["scale"].to_numpy() == scale
        a, p = actual[selected], prediction[selected]
        complete = np.isfinite(a).all(axis=1)
        common = {
            "model": name,
            "month": month,
            "origin": str(panel.dates[origin]),
            "window_start": start,
            "window_end": end,
            "scale": scale,
            "regions": int(selected.sum()),
        }
        for band, sl in (
            ("daily", slice(None)),
            ("d01-07", slice(0, 7)),
            ("d08-14", slice(7, 14)),
            ("d15-30", slice(14, 30)),
        ):
            yield dict(**common, band=band, **errors(a[:, sl], p[:, sl]))
        yield dict(
            **common,
            band="total30",
            **errors(
                a[complete].astype(np.float64).sum(axis=1),
                p[complete].astype(np.float64).sum(axis=1),
            ),
        )


def run_model(name: str, args, panel: ValuePanel, windows, run: Path, cache: Path):
    from joblib import dump

    print(f"starting {name}", flush=True)
    started = perf_counter()
    config = {}
    saved = []
    if name in MODELS:
        x, y, weight = prepare_training(panel, cache)
        if name in {"extra_trees", "random_forest"} and len(y) > 400_000:
            ix = np.sort(
                np.random.default_rng(SEED).choice(len(y), 400_000, replace=False)
            )
            x, y, weight = x[ix], y[ix], weight[ix]
        if "_l1" in name:
            # Match absolute error in original value units up to the existing weight cap.
            weight = weight**2
        zero_keep = ZERO_KEEP.get(name, 1.0)
        if zero_keep < 1:
            is_positive = y > 0
            selected = is_positive | (
                np.random.default_rng(SEED).random(len(y)) < zero_keep
            )
            x, y, weight = x[selected], y[selected], weight[selected]
            if name.endswith("_ipw"):
                weight[y == 0] /= zero_keep
        estimator = fit_model(name, x, y, weight, args.threads, args.device)
        if name.startswith("xgboost") and args.device == "cuda":
            estimator.estimator.set_params(device="cpu")
        config = {
            "train_rows": len(y),
            "train_zeros": int((y == 0).sum()),
            "zero_keep": zero_keep,
            "fit_end": str(FIT_END),
            "features": list(FEATURES),
        }
        temporary_model = run / f".{name}.joblib.tmp"
        dump(dict(model=name, regressor=estimator, **config), temporary_model)
        temporary_model.replace(run / f"{name}.joblib")
        predict = lambda batch: estimator.predict(batch.x, batch.scale)
    elif name in FOUNDATION:
        from planetarium.modeling.value_foundation import load_inference

        infer, config = load_inference(
            name, HORIZON, args.device, args.batch_size, args.threads
        )
        weights = fit_foundation(infer, panel, cache)
        by_size = dict(zip(SCALES, weights.tolist(), strict=True))
        config.update({"calibration_end": str(FIT_END), "weights": by_size})
        key = panel.regions["scale"].to_list()
        coefficients = np.asarray([by_size[scale] for scale in key], dtype=np.float32)
        predict = lambda batch: (
            (
                foundation_features(infer(batch.context), batch)
                * coefficients[:, None, :]
            ).sum(axis=-1)
            * batch.scale[:, None]
        )
    else:
        predict = lambda batch: baseline(name, batch)

    rows = []
    for month, origin in windows:
        batch = save_batch(panel, origin, cache)
        prediction = predict(batch)
        actual = labels(panel, origin)
        rows.extend(window_metrics(panel, month, origin, name, actual, prediction))
        if args.keep_predictions:
            saved.append(prediction)
        print(f"{name}: {month} as of {panel.dates[origin]}", flush=True)
    if args.keep_predictions:
        np.save(run / f"predictions_{name}.npy", np.stack(saved))
    config.update({"model": name, "seconds": round(perf_counter() - started, 2)})
    (run / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
    temporary_metrics = run / f".metrics_{name}.csv.tmp"
    pl.DataFrame(rows).write_csv(temporary_metrics)
    temporary_metrics.replace(run / f"metrics_{name}.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--out", type=Path, default=WORK / "results")
    parser.add_argument("--models", nargs="+", choices=BASELINES + MODELS + FOUNDATION)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--keep-predictions", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    panel = load_panel(args.root, args.work)
    cache = args.work / "benchmark-v1" / panel.fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    windows = list(rolling_origins(panel))
    if not windows:
        raise ValueError(
            "The release does not contain any complete evaluation periods."
        )
    if args.prepare:
        x, _, _ = prepare_training(panel, cache)
        for key, origin in windows:
            save_batch(panel, origin, cache)
            print(f"prepared {key}", flush=True)
        print(f"{len(x):,} training rows")
        return
    for name in args.models or BASELINES + MODELS:
        with FileLock(args.out / f"{name}.lock"):
            run_model(name, args, panel, windows, args.out, cache)


if __name__ == "__main__":
    main()
