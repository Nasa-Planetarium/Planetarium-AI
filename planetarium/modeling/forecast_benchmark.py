"""Rolling verification of next-month hotspot occurrence probabilities."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from filelock import FileLock
from scipy.ndimage import uniform_filter
from sklearn.metrics import average_precision_score

from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.grid_occurrence import id05, tcol

WORK = INTERIM_DATA_DIR / "occurrence_benchmark"
REPORT = REPORTS_DIR / "forecast_benchmark.md"
FIRST = 2012 * 12 + 1
LAST = 2026 * 12 + 9
YEARS = range(2020, 2027)
LOOKBACK = 36
FIT_SAMPLE = 600_000
CAL_SAMPLE = 200_000
MODELS = {
    "timesfm3": "TimesFM 3.0 + Platt",
    "seasonal": "과거 같은 달 발생률",
    "markov": "계절 Markov",
    "logistic": "로지스틱 회귀",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "patchtsmixer": "PatchTSMixer 이진 분류",
    "patchtst": "PatchTST 이진 분류",
}
BASELINES = {"seasonal", "markov"}
NEURAL = {"patchtsmixer", "patchtst"}
DATES = np.arange(FIRST, LAST + 2)
EVALUATION = np.flatnonzero((DATES >= 2020 * 12) & (DATES <= 2026 * 12 + 6))
FUTURE = len(DATES) - 1
SPLITS = {
    "validation": (2020 * 12, 2024 * 12),
    "test": (2024 * 12, 2026 * 12),
    "recent": (2026 * 12, 2026 * 12 + 7),
}


def _write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def aggregate() -> pl.DataFrame:
    """Reuse the release's grid IDs and monthly point-count label definition."""
    return (
        pl.scan_parquet(PROCESSED_DATA_DIR / "harmonized_cell_day.parquet")
        .with_columns(t=tcol())
        .filter(pl.col("t").is_between(FIRST, LAST), pl.col("n_points") > 0)
        .group_by(g=id05(pl.col("cell_id")), t="t")
        .agg(n=pl.col("n_points").cast(pl.UInt64).sum())
        .collect(engine="streaming")
    )


def land_cells() -> pl.DataFrame:
    import shapely
    from shapely.geometry import shape

    data = json.loads(
        (PROCESSED_DATA_DIR / "aoi_boundaries.geojson").read_text(encoding="utf-8")
    )
    geometries, countries, continents = [], [], []
    for item in data["features"]:
        props = item["properties"]
        if (
            props["level"] not in {"country", "kr"}
            or props["parent_aoi"] == "CONT_Antarctica"
        ):
            continue
        for part in shapely.get_parts(shapely.make_valid(shape(item["geometry"]))):
            if part.area > 0:
                geometries.append(part)
                countries.append(props["aoi_id"])
                continents.append(props["parent_aoi"])

    grid = np.arange(360 * 720, dtype=np.int32)
    lon, lat = grid % 720 * 0.5 - 180, grid // 720 * 0.5 - 90
    boxes = shapely.box(lon, lat, lon + 0.5, lat + 0.5)
    grid_ix, part_ix = shapely.STRtree(geometries).query(boxes, predicate="intersects")
    overlap = shapely.area(
        shapely.intersection(boxes[grid_ix], np.asarray(geometries)[part_ix])
    )
    return (
        pl.DataFrame(
            {
                "g": grid_ix.astype(np.int32),
                "country": np.asarray(countries)[part_ix],
                "continent": np.asarray(continents)[part_ix],
                "area": overlap,
            }
        )
        .filter(pl.col("area") > 0)
        .group_by("g", "country", "continent")
        .agg(pl.col("area").sum())
        .sort(["g", "area", "country"], descending=[False, True, False])
        .unique("g", keep="first")
        .sort("g")
        .with_columns(
            lon=pl.col("g") % 720 * 0.5 - 179.75, lat=pl.col("g") // 720 * 0.5 - 89.75
        )
        .drop("area")
    )


def features(counts: np.ndarray, cells: pl.DataFrame, target: int) -> np.ndarray:
    """Only observations at positions smaller than target can enter this row."""
    prior = counts[:target]
    active = prior > 0
    same_index = np.arange(target - 12, -1, -12)
    same = active[same_index]
    previous = active[-1]
    with_prior = same_index[same_index > 0]
    conditional = active[with_prior - 1] == previous
    base = (same.sum(axis=0) + 0.5) / (len(same) + 1)
    markov = ((active[with_prior] & conditional).sum(axis=0) + 4 * base) / (
        conditional.sum(axis=0) + 4
    )
    _, country = np.unique(cells["country"].to_numpy(), return_inverse=True)
    country_base = (np.bincount(country, weights=base) / np.bincount(country))[country]
    g = cells["g"].to_numpy()
    row, col = g // 720, g % 720
    dense = np.zeros((360, 720), dtype=np.float32)
    dense[row, col] = base
    near = uniform_filter(dense, size=3, mode=("constant", "wrap"))[row, col]
    angle = 2 * np.pi * (DATES[target] % 12) / 12
    return np.column_stack(
        (
            base,
            markov,
            (same[:5].sum(axis=0) + 0.5) / (min(5, len(same)) + 1),
            (same.sum(axis=0) + 2 * country_base) / (len(same) + 2),
            country_base,
            near,
            *(np.log1p(prior[-lag]) for lag in (1, 2, 3, 6, 12)),
            *(active[-length:].mean(axis=0) for length in (3, 6, 12, 36)),
            cells["lat"].to_numpy() / 90,
            cells["lon"].to_numpy() / 180,
            np.full(len(cells), np.sin(angle)),
            np.full(len(cells), np.cos(angle)),
        )
    ).astype(np.float32)


def prepare() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    marker = WORK / "panel.json"
    if marker.exists():
        print(marker, flush=True)
        return
    status = (
        pl.read_parquet(PROCESSED_DATA_DIR / "aoi_month.parquet")
        .filter(pl.col("aoi_id") == "WORLD")
        .with_columns(t=tcol())
        .filter(pl.col("t").is_between(FIRST, LAST))
        .sort("t")
    )
    if (
        status["t"].to_list() != list(range(FIRST, LAST + 1))
        or not status["status"].is_in(["ok", "nrt"]).all()
        or not (status["value_source"] == "snpp_observed").all()
        or not (
            status.filter(pl.col("t") <= 2026 * 12 + 6)["data_source"] == "SP"
        ).all()
    ):
        raise ValueError("Unexpected observation coverage for the v1.2 S-NPP panel.")
    cells = land_cells()
    det = (
        aggregate()
        .join(cells.with_row_index("ix").select("g", "ix"), on="g", how="inner")
        .select("t", "ix", "n")
    )
    if det["n"].max() > np.iinfo(np.uint32).max:
        raise ValueError("Monthly detection count exceeds the stored range.")
    counts = np.zeros((len(DATES) - 1, cells.height), dtype=np.uint32)
    counts[det["t"].to_numpy() - FIRST, det["ix"].to_numpy()] = det["n"].to_numpy()
    cells.write_parquet(WORK / "cells.parquet")
    np.save(WORK / "counts.npy", counts)
    shape = (
        len(DATES) - LOOKBACK,
        cells.height,
        features(counts, cells, LOOKBACK).shape[1],
    )
    xx = np.lib.format.open_memmap(
        WORK / "features.npy", mode="w+", dtype=np.float32, shape=shape
    )
    for target in range(LOOKBACK, len(DATES)):
        xx[target - LOOKBACK] = features(counts, cells, target)
    xx.flush()
    _write(
        marker,
        json.dumps(
            {
                "grids": cells.height,
                "months": len(counts),
                "positive_rows": int(np.count_nonzero(counts)),
            },
            indent=2,
        ),
    )
    print(f"{cells.height:,} cells, {len(counts)} observed months", flush=True)


@dataclass
class Panel:
    cells: pl.DataFrame
    counts: np.ndarray
    x: np.ndarray

    @property
    def width(self) -> int:
        return self.cells.height

    def lookup(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target, grid = np.divmod(ids, self.width)
        return self.x[target - LOOKBACK, grid], (self.counts[target, grid] > 0).astype(
            np.uint8
        )


def sample(months: np.ndarray, width: int, limit: int, seed: int) -> np.ndarray:
    size = len(months) * width
    if size <= limit:
        rows = np.arange(size)
    else:
        rows = np.random.default_rng(seed).choice(size, size=limit, replace=False)
        rows.sort()
    return months[rows // width] * width + rows % width


def month_text(value: int) -> str:
    return f"{value // 12:04d}-{value % 12 + 1:02d}"


def score(p: np.ndarray, panel: Panel) -> pl.DataFrame:
    truth = panel.counts[EVALUATION] > 0
    rows = []
    groups = {"WORLD": np.ones(panel.width, dtype=bool)}
    continent = panel.cells["continent"].to_numpy()
    groups.update({str(name): continent == name for name in np.unique(continent)})
    for split, (first, stop) in SPLITS.items():
        use = (DATES[EVALUATION] >= first) & (DATES[EVALUATION] < stop)
        for group, cols in groups.items():
            y = truth[use][:, cols].ravel()
            s = p[use][:, cols].ravel()
            clipped = s.astype(np.float64).clip(1e-7, 1 - 1e-7)
            bins = np.minimum((s * 10).astype(int), 9)
            ece = np.abs(np.bincount(bins, weights=s - y, minlength=10)).sum() / len(y)
            rows.append(
                {
                    "split": split,
                    "group": group,
                    "rows": len(y),
                    "positive_rate": y.mean(),
                    "brier": np.square(s - y).mean(),
                    "pr_auc": average_precision_score(y, s) if y.any() else None,
                    "log_loss": -np.where(
                        y, np.log(clipped), np.log1p(-clipped)
                    ).mean(),
                    "ece": ece,
                    "mean_probability": s.mean(),
                }
            )
    return pl.DataFrame(rows)


def _render() -> None:
    files = sorted((WORK / "metrics").glob("*.csv"))
    lines = ["<!-- occurrence-results:start -->", "", "## 전 세계 비교", ""]
    if files:
        frame = pl.concat([pl.read_csv(f) for f in files], how="diagonal_relaxed")
        _write(WORK / "metrics.csv", frame.write_csv())
        rows = {
            (r["model"], r["split"]): r
            for r in frame.filter(pl.col("group") == "WORLD").to_dicts()
        }
        lines += [
            "| 모델 | 2020–2023 Brier | 2024–2025 Brier | 2024–2025 PR-AUC | 2024–2025 log loss | 2026년 1–7월 Brier |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in sorted(
            {r[0] for r in rows}, key=lambda m: rows[m, "validation"]["brier"]
        ):
            val, test, recent = (rows[name, s] for s in SPLITS)
            lines.append(
                f"| {MODELS[name]} | {val['brier']:.5f} | {test['brier']:.5f} | "
                f"{test['pr_auc']:.5f} | {test['log_loss']:.5f} | {recent['brier']:.5f} |"
            )
    lines += ["", "<!-- occurrence-results:end -->"]
    text = REPORT.read_text(encoding="utf-8")
    start, end = "<!-- occurrence-results:start -->", "<!-- occurrence-results:end -->"
    if start in text and end in text:
        text = text.split(start, 1)[0] + "\n".join(lines) + text.split(end, 1)[1]
    else:
        text += "\n" + "\n".join(lines) + "\n"
    _write(REPORT, text)


def run(model: str, threads: int, device: str, force: bool) -> None:
    from planetarium.modeling.occurrence_models import make_predictor, timesfm_signal

    if not (WORK / "panel.json").exists():
        raise ValueError("Run prepare before running a model.")
    panel = Panel(
        pl.read_parquet(WORK / "cells.parquet"),
        np.load(WORK / "counts.npy", mmap_mode="r"),
        np.load(WORK / "features.npy", mmap_mode="r"),
    )
    for name in ("runs", "metrics"):
        (WORK / name).mkdir(exist_ok=True)
    summary = WORK / "metrics" / f"{model}.csv"
    with FileLock(WORK / "runs" / f"{model}.lock", timeout=0):
        if summary.exists() and not force:
            print(f"{model} already complete")
            return
        start = time.perf_counter()
        predictions = np.empty((len(EVALUATION), panel.width), dtype=np.float32)
        signal = (
            timesfm_signal(panel, WORK, device, threads)
            if model == "timesfm3"
            else None
        )
        for year in YEARS:
            which = np.flatnonzero(DATES[EVALUATION] // 12 == year)
            train_months = np.flatnonzero(
                (np.arange(len(DATES)) >= LOOKBACK) & (DATES < (year - 1) * 12)
            )
            calibration_months = np.flatnonzero(DATES // 12 == year - 1)
            assert DATES[train_months[-1]] < DATES[calibration_months[0]] < year * 12
            predictor = make_predictor(
                model, panel, train_months, calibration_months, threads, device, signal
            )
            for i in which:
                predictions[i] = predictor(int(EVALUATION[i]))
            if year == YEARS[-1]:
                future = predictor(FUTURE)
            print(f"{model} {year}: {len(which)} months scored", flush=True)
        for p in (predictions, future):
            if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
                raise ValueError("Model returned invalid probabilities.")
        stem = WORK / "runs" / model
        np.savez_compressed(
            stem.with_suffix(".npz"),
            p_any=predictions,
            g=panel.cells["g"].to_numpy(),
            target_month=DATES[EVALUATION],
        )
        (
            panel.cells.with_columns(
                origin_month=pl.lit(month_text(LAST)),
                target_month=pl.lit(month_text(LAST + 1)),
                p_any=pl.Series(future.astype(np.float32)),
                p_none=pl.Series((1 - future).astype(np.float32)),
            )
            .rename({"g": "cell_id"})
            .write_parquet(stem.with_suffix(".parquet"))
        )
        results = score(predictions, panel).with_columns(pl.lit(model).alias("model"))
        with FileLock(WORK / "report.lock"):
            _write(summary, results.write_csv())
            _render()
        print(f"{model} complete ({time.perf_counter() - start:.1f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "report"])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.command == "prepare":
        WORK.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(INTERIM_DATA_DIR / "occurrence_benchmark_prepare.lock"):
            prepare()
    elif args.command == "report":
        with FileLock(WORK / "report.lock"):
            _render()
    elif args.model is not None:
        run(args.model, args.threads, args.device, args.force)
    else:
        parser.error("run requires --model")


if __name__ == "__main__":
    main()
