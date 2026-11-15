"""Daily reference-detection labels and observation coverage."""

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
from planetarium.grid_occurrence import id05

WORK = INTERIM_DATA_DIR / "daily_occurrence"
FIRST = date(2012, 2, 1)
DAY = timedelta(days=1)
FILES = (
    "aoi_boundaries.geojson",
    "harmonized_cell_day.parquet",
    "satellite_day_status.parquet",
)


@dataclass(frozen=True)
class DailyPanel:
    cells: pl.DataFrame
    dates: np.ndarray
    events: np.ndarray
    coverage: np.ndarray
    fingerprint: str

    @property
    def sides(self) -> np.ndarray:
        return np.where(self.cells["lon"].to_numpy() >= 0, 1, 2).astype(np.uint8)


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


def land_cells(root: Path) -> pl.DataFrame:
    import shapely
    from shapely.geometry import shape

    data = json.loads((root / "aoi_boundaries.geojson").read_text(encoding="utf-8"))
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


def _prepare(root: Path, directory: Path) -> None:
    dates, coverage = normal_days(root)
    cells = land_cells(root)
    det = (
        pl.scan_parquet(root / "harmonized_cell_day.parquet")
        .filter(
            pl.col("date_local").is_between(FIRST, dates[-1].astype(date)),
            pl.col("n_points") > 0,
            pl.col("value_source").is_in(["snpp_observed", "n20_bridge"]),
        )
        .select(g=id05(pl.col("cell_id")), day=pl.col("date_local"))
        .unique(subset=["day", "g"])
        .join(cells.with_row_index("ix").select("g", "ix").lazy(), on="g", how="inner")
        .select("day", "ix")
        .collect(engine="streaming")
    )
    matrix = np.lib.format.open_memmap(
        directory / "events.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(dates), cells.height),
    )
    matrix[:] = 0
    by_date = (det["day"].to_numpy() - dates[0]).astype(int)
    matrix[by_date, det["ix"].to_numpy()] = 1
    matrix.flush()
    del matrix
    cells.write_parquet(directory / "cells.parquet")
    np.savez(directory / "calendar.npz", dates=dates, coverage=coverage)
    (directory / "ready").touch()


def load_panel(root: Path | None = None, work: Path | None = None) -> DailyPanel:
    """Prepare or memory-map the daily panel for the current release files."""
    root = Path(root or PROCESSED_DATA_DIR).resolve()
    work = Path(work or WORK)
    inputs = [(name, (root / name).stat()) for name in FILES]
    stamp = str(root) + repr([(name, s.st_size, s.st_mtime_ns) for name, s in inputs])
    fingerprint = hashlib.sha256(("daily-v1" + stamp).encode()).hexdigest()[:16]
    directory = work / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(directory / "prepare.lock"):
        if not (directory / "ready").exists():
            _prepare(root, directory)
    with np.load(directory / "calendar.npz") as calendar:
        return DailyPanel(
            pl.read_parquet(directory / "cells.parquet"),
            calendar["dates"],
            np.load(directory / "events.npy", mmap_mode="r"),
            calendar["coverage"],
            fingerprint,
        )
