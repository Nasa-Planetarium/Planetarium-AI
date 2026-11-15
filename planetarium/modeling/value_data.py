"""Daily S-NPP-equivalent value and the satellite-day availability mask."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from filelock import FileLock

from planetarium.config import INTERIM_DATA_DIR, NOAA20_BRIDGE_START, PROCESSED_DATA_DIR

WORK = INTERIM_DATA_DIR / "value_forecast"
FIRST = date(2017, 1, 1)
DAY = timedelta(days=1)
WIDTH = 3600
HEIGHT = 1800
TABLE = "harmonized_cell_day.parquet"
FILES = (TABLE, "satellite_day_status.parquet", "aoi_boundaries.geojson")
SCALES = ("0.1deg", "0.5deg", "2deg", "rectangles")
# Samples are selected by geography alone, without checking detection history.
LAYOUTS = (
    (1, 1, 0, 0, 3000),
    (5, 5, 0, 0, 4000),
    (20, 20, 0, 0, None),
    (13, 7, 4, 2, 1800),
)
SOURCES = ("snpp_observed", "n20_bridge")


@dataclass(frozen=True)
class ValuePanel:
    dates: np.ndarray
    coverage: np.ndarray
    regions: pl.DataFrame
    values: np.ndarray
    fingerprint: str = ""


def normal_days(root: Path) -> tuple[np.ndarray, np.ndarray]:
    status = pl.read_parquet(root / "satellite_day_status.parquet").filter(
        pl.col("satellite").is_in(["S-NPP", "NOAA-20"])
    )
    good = {
        (row["satellite"], row["date_utc"])
        for row in status.select(
            "satellite", "date_utc", "coverage", "status"
        ).iter_rows(named=True)
        if row["status"] == "ok" and row["coverage"] == "global"
    }
    last = status["date_utc"].max()
    dates = [FIRST + i * DAY for i in range((last - FIRST).days + 1)]
    coverage = []
    for d in dates:
        sensor = "NOAA-20" if d >= NOAA20_BRIDGE_START else "S-NPP"
        today = (sensor, d) in good
        east = today and (sensor, d - DAY) in good
        west = today and (sensor, d + DAY) in good
        coverage.append(int(east) + 2 * int(west))
    return np.asarray(dates, dtype="datetime64[D]"), np.asarray(
        coverage, dtype=np.uint8
    )


def land_geometries(root: Path):
    import shapely
    from shapely.geometry import shape

    data = json.loads((root / "aoi_boundaries.geojson").read_text(encoding="utf-8"))
    geometries, continents = [], []
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
                continents.append(props["parent_aoi"])

    return np.asarray(geometries), np.asarray(continents)


def sides_for_bounds(west: np.ndarray, width: int) -> np.ndarray:
    return (west + width > 1800).astype(np.uint8) + 2 * (west < 1800).astype(np.uint8)


def choose_boxes(root: Path) -> pl.DataFrame:
    import shapely

    geometries, continents = land_geometries(root)
    tree = shapely.STRtree(geometries)
    rng = np.random.default_rng(1126)
    frames = []
    for scale, (width, height, ox, oy, limit) in zip(SCALES, LAYOUTS, strict=True):
        # Sample native cells from a geographic lattice and check their centroids.
        stride = 2 if scale == "0.1deg" else 1
        x, y = np.meshgrid(
            np.arange(ox, WIDTH - width + 1, width * stride),
            np.arange(oy, HEIGHT - height + 1, height * stride),
        )
        x, y = x.ravel(), y.ravel()
        if limit is not None:
            order = rng.permutation(len(x))
            x, y = x[order], y[order]
        matched = []
        for start in range(0, len(x), 60_000):
            xx, yy = x[start : start + 60_000], y[start : start + 60_000]
            lon, lat = xx / 10 - 180, yy / 10 - 90
            shapes = (
                shapely.points(lon + 0.05, lat + 0.05)
                if scale == "0.1deg"
                else shapely.box(lon, lat, lon + width / 10, lat + height / 10)
            )
            i, j = tree.query(shapes, predicate="intersects")
            if scale != "0.1deg":
                area = shapely.area(shapely.intersection(shapes[i], geometries[j]))
                keep = area > 0
                i, j, area = i[keep], j[keep], area[keep]
                order = np.lexsort((-area, i))
                i, j = i[order], j[order]
            _, first = np.unique(i, return_index=True)
            matched.append(np.column_stack((i[first] + start, j[first])))
            if limit is not None and sum(map(len, matched)) >= limit:
                break
        items = np.concatenate(matched)[:limit]
        xx, yy = x[items[:, 0]], y[items[:, 0]]
        frames.append(
            pl.DataFrame(
                {
                    "scale": scale,
                    "continent": continents[items[:, 1]],
                    "west": xx,
                    "south": yy,
                    "lon": (xx + width / 2) / 10 - 180,
                    "lat": (yy + height / 2) / 10 - 90,
                    "sides": sides_for_bounds(xx, width),
                    "n_cells": width * height,
                }
            )
        )
    return pl.concat(frames)


def scan_values(root: Path, first: date, last: date) -> pl.LazyFrame:
    return (
        pl.scan_parquet(root / TABLE)
        .filter(
            pl.col("year").is_between(first.year, last.year),
            pl.col("date_local").is_between(first, last),
            pl.col("value_source").is_in(SOURCES),
            pl.col("value").is_not_null(),
        )
        .select(pl.col("cell_id").cast(pl.Int32), "date_local", "value")
    )


def fill_matrix(
    totals: pl.DataFrame, dates: np.ndarray, sides: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(dates), len(sides)), dtype=np.float32)
    day = (totals["date_local"].to_numpy() - dates[0]).astype(int)
    result[day, totals["ix"].to_numpy()] = totals["value"].to_numpy()
    return result


def mask_unobserved(
    values: np.ndarray, coverage: np.ndarray, sides: np.ndarray
) -> None:
    for i in range(1, 4):
        values[np.ix_((coverage & i) != i, sides == i)] = np.nan


def _build(root: Path, dest: Path) -> None:
    dates, coverage = normal_days(root)
    regions = choose_boxes(root)
    observations = scan_values(root, FIRST, dates[-1].astype(date))
    matrix = np.lib.format.open_memmap(
        dest / "values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(dates), regions.height),
    )
    offset = 0
    for scale, (width, height, ox, oy, _) in zip(SCALES, LAYOUTS, strict=True):
        boxes = regions.filter(pl.col("scale") == scale)
        ncols = (WIDTH - ox) // width
        keys = boxes.with_row_index("ix").select(
            "ix",
            key=(
                ((pl.col("south") - oy) // height) * ncols
                + (pl.col("west") - ox) // width
            ).cast(pl.Int32),
        )
        col = pl.col("cell_id") % WIDTH - ox
        row = pl.col("cell_id") // WIDTH - oy
        sums = (
            observations.filter(
                col.is_between(0, ncols * width - 1),
                row.is_between(0, ((HEIGHT - oy) // height) * height - 1),
            )
            .with_columns(key=row // height * ncols + col // width)
            .join(keys.lazy(), on="key", how="inner")
            .group_by("date_local", "ix")
            .agg(pl.col("value").cast(pl.Float64).sum())
            .collect(engine="streaming")
        )
        values = fill_matrix(sums, dates, boxes["sides"].to_numpy())
        mask_unobserved(values, coverage, boxes["sides"].to_numpy())
        end = offset + boxes.height
        matrix[:, offset:end] = values
        offset = end
        print(f"{scale}: {boxes.height:,} regions", flush=True)
    matrix.flush()
    regions.write_parquet(dest / "regions.parquet")
    np.savez(dest / "calendar.npz", dates=dates, coverage=coverage)
    (dest / "ready").touch()


def load_panel(root: Path = PROCESSED_DATA_DIR, work: Path = WORK) -> ValuePanel:
    root = root.resolve()
    signature = [
        (name, (root / name).stat().st_size, (root / name).stat().st_mtime_ns)
        for name in FILES
    ]
    fingerprint = hashlib.sha256(
        repr(("value-v1", root, signature)).encode()
    ).hexdigest()[:16]
    directory = work / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / "prepare.lock"):
        if not (directory / "ready").exists():
            _build(root, directory)
    with np.load(directory / "calendar.npz") as calendar:
        return ValuePanel(
            calendar["dates"],
            calendar["coverage"],
            pl.read_parquet(directory / "regions.parquet"),
            np.load(directory / "values.npy", mmap_mode="r"),
            fingerprint,
        )
