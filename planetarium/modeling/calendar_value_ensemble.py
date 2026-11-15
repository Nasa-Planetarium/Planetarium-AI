"""Fit a blend of monthly LightGBM and FireOutlook Planner counts on 2025."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse import csr_matrix

from planetarium.config import PROCESSED_DATA_DIR
from planetarium.modeling.calendar_value_benchmark import (
    FIT_START,
    LAST_MONTH,
    monthly_labels,
    predict_months,
    score_models,
)
from planetarium.modeling.value_benchmark import ONE_MONTH
from planetarium.modeling.value_data import LAYOUTS, SCALES, WORK, load_panel

AREAS = SCALES[1:]
BIAS_LIMIT = 0.10


def planner_counts(path: Path, regions: pl.DataFrame, months: np.ndarray) -> np.ndarray:
    """Aggregate out-of-sample Planner grid means using its bbox area weights."""
    rows = pl.read_parquet(path, columns=["year", "month", "g", "mu"]).sort(
        "year", "month", "g"
    )
    groups = rows.partition_by("year", "month", maintain_order=True)
    keys = [f"{part['year'][0]}-{part['month'][0]:02d}" for part in groups]
    if keys != months.astype(str).tolist():
        raise ValueError(f"{path} does not contain the required ordered target months.")
    grid = groups[0]["g"].to_numpy()
    if len(grid) != len(np.unique(grid)) or any(
        not np.array_equal(part["g"].to_numpy(), grid) for part in groups
    ):
        raise ValueError(f"{path} has missing, duplicate or inconsistent grid cells.")
    means = np.stack([part["mu"].to_numpy() for part in groups])
    if not np.isfinite(means).all() or (means < 0).any():
        raise ValueError(f"{path} contains invalid count predictions.")

    layout = {name: sizes[:2] for name, sizes in zip(SCALES, LAYOUTS, strict=True)}
    by_grid = {int(g): i for i, g in enumerate(grid)}
    row_ix, grid_ix, area = [], [], []
    for i, (scale, x, y) in enumerate(
        regions.select("scale", "west", "south").iter_rows()
    ):
        width, height = layout[scale]
        east, north = x + width, y + height
        for row in range(y // 5, (north + 4) // 5):
            for col in range(x // 5, (east + 4) // 5):
                pos = by_grid.get(row * 720 + col)
                if pos is None:
                    continue
                overlap_x = min(east, 5 * col + 5) - max(x, 5 * col)
                overlap_y = min(north, 5 * row + 5) - max(y, 5 * row)
                row_ix.append(i)
                grid_ix.append(pos)
                area.append(overlap_x * overlap_y / 25)
    weights = csr_matrix((area, (row_ix, grid_ix)), shape=(regions.height, len(grid)))
    return (weights @ means.T).T


def macro_weights(actual: np.ndarray, area_index: np.ndarray) -> np.ndarray:
    weights = np.empty(len(area_index), dtype=float)
    for i in range(len(AREAS)):
        keep = area_index == i
        total = actual[:, keep].sum()
        if total <= 0:
            raise ValueError(
                f"Cannot calibrate {AREAS[i]} without positive total activity."
            )
        weights[keep] = 1 / (len(AREAS) * total)
    return weights


def fit_blend(
    actual, lightgbm, planner, weight, groups, limit=BIAS_LIMIT
) -> np.ndarray:
    """Minimize macro WAPE for nonnegative weights with a bound on macro bias."""
    groups = groups.ravel()
    diff = (lightgbm - actual).ravel()
    step = (planner - lightgbm).ravel()
    w = np.broadcast_to(weight, actual.shape).ravel()
    n = int(groups.max()) + 1
    base_bias = float(w @ diff)
    delta = np.bincount(groups, weights=w * step, minlength=n)
    attainable_min = base_bias + np.minimum(delta, 0).sum()
    attainable_max = base_bias + np.maximum(delta, 0).sum()
    lower, upper = np.clip([-limit, limit], attainable_min, attainable_max)
    target = float(np.clip(base_bias, lower, upper))
    initial = np.zeros(n)
    if target != base_bias:
        extreme = (delta * (target - base_bias) > 0).astype(float)
        initial = extreme * ((target - base_bias) / (delta @ extreme))
    constraints = (
        [LinearConstraint(delta, lower - base_bias, upper - base_bias)]
        if np.any(delta)
        else []
    )

    def loss(alpha):
        residual = diff + step * alpha[groups]
        grad = np.bincount(groups, weights=w * step * np.sign(residual), minlength=n)
        return float(w @ np.abs(residual)), grad

    result = minimize(
        loss,
        initial,
        jac=True,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(f"Ensemble fit failed: {result.message}")
    return np.clip(result.x, 0, 1)


def fit_and_score(lightgbm, planner, actual, months, scales, issue):
    fit = months.astype("datetime64[Y]") == np.datetime64("2025", "Y")
    which = np.array([AREAS.index(s) for s in scales])
    weight = macro_weights(actual[fit], which)
    old_bias = float(((lightgbm[fit] - actual[fit]) * weight).sum())
    adjusted = 1 + np.clip(old_bias, -BIAS_LIMIT, BIAS_LIMIT)
    gain = adjusted / (1 + old_bias) if old_bias > -1 else 1.0
    fixed = fit_blend(
        actual[fit],
        lightgbm[fit],
        planner[fit],
        weight,
        np.zeros(actual[fit].shape, dtype=int),
    ).item()
    groups = 2 * which[None, :] + (planner > lightgbm)
    seen = np.unique(groups[fit])
    adaptive = np.full(2 * len(AREAS), fixed)
    adaptive[seen] = fit_blend(
        actual[fit],
        lightgbm[fit],
        planner[fit],
        weight,
        np.searchsorted(seen, groups[fit]),
    )
    delta = planner - lightgbm
    predictions = {
        "lightgbm_l1": lightgbm,
        "main_planner_count": planner,
        "lightgbm_bias_adjusted": lightgbm * gain,
        "fixed_blend": lightgbm + fixed * delta,
        "adaptive_blend": lightgbm + adaptive[groups] * delta,
    }
    params = {
        "lightgbm_gain": float(gain),
        "fixed_planner_weight": fixed,
        "adaptive_planner_weight": {
            scale: {
                "planner_at_most_lightgbm": float(adaptive[2 * i]),
                "planner_above_lightgbm": float(adaptive[2 * i + 1]),
            }
            for i, scale in enumerate(AREAS)
        },
    }
    rows = [
        r
        for r in score_models(predictions, actual, months, scales, issue)
        if r["scale"] in AREAS
    ]
    return rows, params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--planner-dir", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=WORK / "calendar_ensemble_metrics.csv"
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    panel = load_panel(args.root, args.work)
    months = np.arange(
        FIT_START - 12 * ONE_MONTH,
        np.datetime64(LAST_MONTH, "M") + ONE_MONTH,
        dtype="datetime64[M]",
    )
    actual = monthly_labels(args.root, panel.regions, months)
    eligible = panel.regions["scale"].is_in(AREAS).to_numpy()
    regions = panel.regions.filter(eligible)
    all_rows = []
    params = {
        "training_year": 2025,
        "holdout_year": 2026,
        "bias_limit": BIAS_LIMIT,
        "issue_days": {},
    }
    for issue in (12, 31):
        test, pred = predict_months(
            panel,
            months,
            actual,
            issue,
            threads=args.threads,
            workers=1,
            daily_models={},
            models=("lightgbm_l1",),
        )
        planner = planner_counts(
            args.planner_dir / f"pred_main_{issue}.parquet", regions, months[test]
        )
        rows, weights = fit_and_score(
            pred["lightgbm_l1"][:, eligible],
            planner,
            actual[test][:, eligible],
            months[test],
            regions["scale"].to_numpy(),
            issue,
        )
        all_rows.extend(rows)
        params["issue_days"][str(issue)] = weights
    table = pl.DataFrame(all_rows).sort("as_of_day", "year", "scale", "model")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(params, indent=2) + "\n")
    print(
        table.group_by("as_of_day", "year", "model")
        .agg(pl.col("wape", "bias").mean())
        .sort("as_of_day", "year", "wape")
    )


if __name__ == "__main__":
    main()
