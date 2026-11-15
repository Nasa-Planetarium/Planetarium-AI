"""Prepare, fit, evaluate and query daily hotspot occurrence forecasts."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from filelock import FileLock
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from planetarium.config import REPORTS_DIR, SEED
from planetarium.modeling.daily_occurrence import (
    Series,
    block_series,
    calendar_bin,
    feature_rows,
    forecast,
    grid_series,
    history_stats,
    initial_state,
    model_predictor,
    roll,
    transition_table,
    usable,
)
from planetarium.modeling.daily_occurrence_data import WORK, load_panel

REPORT = REPORTS_DIR / "forecast_benchmark.md"
MODELS = ("seasonal", "markov1", "markov2", "logistic", "xgboost")
LEADS = (1, 2, 3, 7, 14)
TRAIN = range(2016, 2023)
CAL = 2023
PERIODS = {"validation": (2023, 2023), "test": (2024, 2025), "recent": (2026, 2026)}
LABELS = {
    "seasonal": "계절 발생률",
    "markov1": "1차 계절 Markov",
    "markov2": "2차 계절 Markov",
    "logistic": "로지스틱 전이 모델",
    "xgboost": "XGBoost 전이 모델",
}
START_MARKER = "<!-- daily-occurrence-results:start -->"
END_MARKER = "<!-- daily-occurrence-results:end -->"


def _atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _combined(parts):
    return Series(
        np.column_stack([s.events for s in parts]),
        np.concatenate([s.sides for s in parts]),
        np.concatenate([s.lat for s in parts]),
        np.concatenate([s.size for s in parts]),
    )


def _training_rows(
    panel, series: Series, periods: list[tuple[int, int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    groups_x, groups_y = [], []
    rng = np.random.default_rng(SEED)
    bins = calendar_bin(panel.dates)
    for start, stop, samples in periods:
        stats = history_stats(series, panel.dates, panel.coverage, int(start))
        t = rng.integers(start, stop, size=samples)
        g = rng.integers(len(series.size), size=samples)
        side = series.sides[g]
        known = (
            panel.coverage[t] & panel.coverage[t - 1] & panel.coverage[t - 2] & side
        ) == side
        t, g = t[known], g[known]
        states = 2 * series.events[t - 2, g] + series.events[t - 1, g]
        groups_x.append(feature_rows(stats, series, bins[t], g, states))
        groups_y.append(series.events[t, g])
    return np.concatenate(groups_x), np.concatenate(groups_y)


def _periods(dates, first, stop, samples_per_year) -> list[tuple[int, int, int]]:
    boundaries = (
        np.flatnonzero(
            dates[first + 1 : stop].astype("datetime64[Y]")
            != dates[first : stop - 1].astype("datetime64[Y]")
        )
        + first
        + 1
    )
    return [
        (int(lo), int(hi), max(1, round(samples_per_year * (hi - lo) / 365.25)))
        for lo, hi in zip([first, *boundaries], [*boundaries, stop], strict=True)
    ]


def fit(
    panel,
    path: Path,
    threads: int,
    device: str,
    *,
    window: tuple[int, int, int] | None = None,
) -> None:
    from xgboost import XGBClassifier

    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    grids = grid_series(panel)
    subset = rng.choice(len(grids.size), min(5500, len(grids.size)), replace=False)
    areas = [
        block_series(panel, span, sample_count=1100, rng=rng) for span in (2, 4, 8, 12)
    ]
    series = _combined([grids.select(subset), *areas])
    if window is None:
        starts = panel.dates.searchsorted(
            np.array(
                [f"{year}-01-01" for year in (*TRAIN, CAL, CAL + 1)],
                dtype="datetime64[D]",
            )
        ).tolist()
        fit_periods = [(lo, hi, 92_000) for lo, hi in pairwise(starts[:-1])]
        cal_periods = [(starts[-2], starts[-1], 180_000)]
    else:
        first, cal_first, stop = window
        if not (2 <= first < cal_first < stop <= len(panel.dates)):
            raise ValueError(
                "Training and calibration both need earlier daily observations."
            )
        fit_periods = _periods(panel.dates, first, cal_first, 92_000)
        cal_periods = [(cal_first, stop, 180_000)]
    xx, yy = _training_rows(panel, series, fit_periods)
    cx, cy = _training_rows(panel, series, cal_periods)
    if len(np.unique(yy)) != 2 or len(np.unique(cy)) != 2:
        raise ValueError(
            "Training and calibration each need observed detections and non-detections."
        )
    print(f"Fitting {len(yy):,} one-step pairs, calibration {len(cy):,}", flush=True)

    def train_logistic():
        scaler = StandardScaler().fit(xx)
        estimator = LogisticRegression(max_iter=120).fit(scaler.transform(xx), yy)
        return estimator.decision_function(scaler.transform(cx)), {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": estimator.coef_[0].tolist(),
            "bias": float(estimator.intercept_[0]),
        }

    def train_xgb():
        estimator = XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.07,
            objective="binary:logistic",
            tree_method="hist",
            max_bin=64,
            device=device,
            n_jobs=threads,
            random_state=SEED,
        ).fit(xx, yy)
        booster = estimator.get_booster()
        booster.set_param({"device": "cpu", "nthread": threads})
        margin = booster.inplace_predict(cx, predict_type="margin")
        booster.save_model(path / "xgboost.tmp.ubj")
        (path / "xgboost.tmp.ubj").replace(path / "xgboost.ubj")
        return margin, {}

    with ThreadPoolExecutor(2) as pool:
        jobs = {
            "logistic": pool.submit(train_logistic),
            "xgboost": pool.submit(train_xgb),
        }
        for name, job in jobs.items():
            margin, config = job.result()
            cal = LogisticRegression(max_iter=100).fit(margin.reshape(-1, 1), cy)
            _atomic(
                path / f"{name}.json",
                json.dumps(
                    {
                        **config,
                        "cal_slope": float(cal.coef_[0, 0]),
                        "cal_bias": float(cal.intercept_[0]),
                        "train_from": str(panel.dates[fit_periods[0][0]]),
                        "train_until": str(panel.dates[fit_periods[-1][1] - 1]),
                        "cal_from": str(panel.dates[cal_periods[0][0]]),
                        "cal_until": str(panel.dates[cal_periods[-1][1] - 1]),
                        "fit_rows": len(yy),
                        "cal_rows": len(cy),
                        "device": device if name == "xgboost" else "cpu",
                    }
                ),
            )
            print(f"{name} fit and calibration complete", flush=True)


def _measure(fragments) -> dict:
    truth = np.concatenate([item[0] for item in fragments])
    p = np.concatenate([item[1] for item in fragments])
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("Non-probability output in the rollout.")
    return {
        "rows": len(truth),
        "positive_rate": float(truth.mean()),
        "brier": brier_score_loss(truth, p),
        "log_loss": log_loss(truth, p.clip(1e-7, 1 - 1e-7), labels=[0, 1]),
        "pr_auc": average_precision_score(truth, p)
        if truth.min() < truth.max()
        else None,
    }


def evaluation_origins(dates: np.ndarray, period: str, year: int) -> range:
    end = f"{year}-07-31" if period == "recent" else f"{year}-12-31"
    lo = int(np.searchsorted(dates, np.datetime64(f"{year}-01-01")))
    hi = int(np.searchsorted(dates, np.datetime64(end), side="right"))
    return range(lo, hi - max(LEADS), 14)


def evaluate(panel, path: Path) -> pl.DataFrame:
    predictors = {
        name: model_predictor(name, path)[0] for name in ("logistic", "xgboost")
    }
    test_blocks = block_series(panel, 4)
    results = []
    for grain, series in (("cell", grid_series(panel)), ("box", test_blocks)):
        complete = usable(panel.coverage, series.sides)
        for period, years in PERIODS.items():
            store = {(model, lead): [] for model in MODELS for lead in LEADS}
            for year in range(years[0], years[1] + 1):
                origins = evaluation_origins(panel.dates, period, year)
                if not origins:
                    continue
                stats = history_stats(
                    series, panel.dates, panel.coverage, origins.start
                )
                tables = {model: {} for model in MODELS}
                for origin in origins:
                    state = initial_state(
                        series, stats, panel.dates, panel.coverage, origin
                    )
                    bins = calendar_bin(
                        panel.dates[origin + 1 : origin + 1 + max(LEADS)]
                    )
                    for model in MODELS:
                        known = tables[model]
                        for b in set(bins) - known.keys():
                            known[b] = transition_table(
                                model, stats, series, int(b), predictors.get(model)
                            )
                        predicted = roll(state, bins, known.__getitem__)
                        for lead in LEADS:
                            target = origin + lead
                            valid = complete[origin] & complete[target]
                            store[model, lead].append(
                                (
                                    series.events[target, valid],
                                    predicted[lead - 1, valid],
                                )
                            )
                print(f"{grain} {year}: {len(origins)} forecast origins", flush=True)
            for (model, lead), values in store.items():
                if values:
                    results.append(
                        {
                            "period": period,
                            "grain": grain,
                            "model": model,
                            "step": lead,
                            **_measure(values),
                        }
                    )
    return pl.DataFrame(results)


def record(frame: pl.DataFrame, base: Path, path: Path) -> None:
    # Select using validation only, giving the two spatial scales equal weight.
    selected = (
        frame.filter((pl.col("period") == "validation") & pl.col("step").is_in([1, 7]))
        .group_by("model")
        .agg(pl.col("brier").mean())
        .sort("brier", "model")["model"][0]
    )
    _atomic(
        path / "selected.json",
        json.dumps({"model": selected, "available_from": f"{CAL}-12-31"}),
    )
    frame.write_csv(base / "metrics.csv")
    x = frame.filter(pl.col("period") == "test")
    by_key = {(r["model"], r["grain"], r["step"]): r for r in x.to_dicts()}
    lines = [
        START_MARKER,
        "",
        f"검증 구간에서 선택한 기본 모델은 **{LABELS[selected]}**입니다.",
        "",
        "### 2024–2025 전 세계 홀드아웃",
        "",
        "| 모델 | 격자 1스텝 Brier | 3스텝 | 7스텝 | 14스텝 | 격자 1스텝 PR-AUC | 범위 1스텝 Brier | 범위 7스텝 Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        nums = [by_key[model, "cell", step]["brier"] for step in (1, 3, 7, 14)]
        nums += [
            by_key[model, "cell", 1]["pr_auc"],
            *[by_key[model, "box", step]["brier"] for step in (1, 7)],
        ]
        lines.append(
            "| " + " | ".join([LABELS[model], *[f"{v:.5f}" for v in nums]]) + " |"
        )
    lines += ["", END_MARKER]
    text = REPORT.read_text(encoding="utf-8")
    if START_MARKER in text and END_MARKER in text:
        text = (
            text.split(START_MARKER, 1)[0]
            + "\n".join(lines)
            + text.split(END_MARKER, 1)[1]
        )
    else:
        text += "\n" + "\n".join(lines) + "\n"
    _atomic(REPORT, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory containing the release Parquet and GeoJSON files",
    )
    parser.add_argument("--work", type=Path, default=WORK)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("report")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--threads", type=int, default=8)
    bench.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    request = sub.add_parser("forecast")
    request.add_argument("--regions", type=Path, required=True)
    request.add_argument("--start", required=True)
    request.add_argument("--end", required=True)
    request.add_argument("--as-of")
    request.add_argument("--model", choices=("auto", *MODELS), default="auto")
    request.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "forecast":
        regions = json.loads(args.regions.read_text(encoding="utf-8"))
        table = forecast(
            regions,
            args.start,
            args.end,
            as_of=args.as_of,
            model=args.model,
            root=args.root,
            work=args.work,
        )
        if args.output:
            if args.output.suffix == ".parquet":
                table.write_parquet(args.output)
            else:
                table.write_csv(args.output)
        else:
            print(table.write_csv(), end="")
        return

    panel = load_panel(args.root, args.work)
    print(
        f"{panel.cells.height:,} grid cells, {len(panel.dates):,} daily positions, {panel.fingerprint}"
    )
    if args.command in {"benchmark", "report"}:
        base = args.work / panel.fingerprint
        path = base / "models"
        with FileLock(base / "benchmark.lock"):
            if args.command == "benchmark":
                if args.threads < 1:
                    parser.error("threads must be positive")
                fit(panel, path, args.threads, args.device)
                frame = evaluate(panel, path)
            else:
                frame = pl.read_csv(base / "metrics.csv")
            record(frame, base, path)


if __name__ == "__main__":
    main()
