"""Read only the covered time window and grid extent needed for a forecast."""
from datetime import timedelta
from functools import lru_cache
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np

from ..api import ROOT, X_SCALE, Y_SCALE, _bbox, _day
from ..converter.convert import sql_string
from ..converter.world_calendar import PRODUCTION_ORDER
from ..converter.world_features import PLATFORMS

WIDTH, HEIGHT = 34668, 14714
X_ORIGIN, Y_ORIGIN = 17334, 7357
MONTH = np.timedelta64(1, "M")


def _longitude_ranges(west, east):
    return [(west, east)] if west < east else [(west, 180), (-180, east)]


def _covers(outer, inner):
    west, south, east, north = outer
    iw, i_south, ie, i_north = inner
    return south <= i_south and north >= i_north and all(
        any(a <= left and right <= b for a, b in _longitude_ranges(west, east))
        for left, right in _longitude_ranges(iw, ie)
    )


def _bounds(bbox, border):
    west, south, east, north = bbox
    x = []
    for lo, hi in _longitude_ranges(west, east):
        lo = int(np.floor(np.radians(lo) * X_SCALE)) - border
        hi = int(np.floor(np.radians(hi) * X_SCALE)) + border
        x.append((lo, hi))
        if lo < -X_ORIGIN:
            x.append((lo + WIDTH, X_ORIGIN - 1))
        if hi >= X_ORIGIN:
            x.append((-X_ORIGIN, hi - WIDTH))
    y = np.floor(np.sin(np.radians([south, north])) * Y_SCALE).astype(int)
    longitude = " OR ".join("grid_x BETWEEN ? AND ?" for _ in x)
    ranges = [v for pair in x for v in pair] + [int(y[0] - border), int(y[1] + border)]
    return f"({longitude}) AND grid_y BETWEEN ? AND ?", ranges


@lru_cache(maxsize=1)
def _frp_sql():
    bundle = joblib.load(ROOT / "converter/global/world_daily_model.joblib")
    ratios = bundle["frp_models"]["ratios"]
    cases = " ".join(f"WHEN {sql_string(PLATFORMS[key])} THEN {float(ratio)}" for key, ratio in ratios.items())
    return f"native_grid_frp_mw * CASE native_frp_source_platform WHEN 'NOAA-20' THEN 1 {cases} END"


_PLATFORM_RANKS = "CASE native_frp_source_platform " + " ".join(
    f"WHEN {sql_string(PLATFORMS[key])} THEN {rank}" for rank, key in enumerate(PRODUCTION_ORDER)
) + " ELSE 9 END"


class History:
    def __init__(self, archive, updates, bbox, threads):
        self.sources = []
        self.covered = []
        self.bbox = bbox
        self.threads = threads
        for index, folder in enumerate((archive, *updates)):
            folder = Path(folder).resolve()
            info = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            if info.get("grid_km") != 1 or info.get("daily_key") != ["date_utc", "grid_id"]:
                raise ValueError(f"Expected a harmonized FIRMS 1 km dataset in {folder}")
            key = "coverage_utc_inclusive" if index else "date_range_utc_inclusive"
            if key not in info:
                raise ValueError("Use convert_api_csv to create forecast updates with known coverage")
            first, last = map(_day, info[key])
            self.sources.append((folder, first, last))
            if index == 0 or _covers(_bbox(info["bbox_wsen"]), bbox):
                self.covered.append((first, last))

    @property
    def last_day(self):
        return max(last for _, last in self.covered)

    def _check_coverage(self, first, last):
        day = first
        for begin, end in sorted(self.covered):
            if begin <= day <= end:
                day = end + timedelta(days=1)
                if day > last:
                    return
        raise ValueError(f"Missing FIRMS history for {day} UTC in the requested region")

    def periods(self, first, last, *, monthly=False):
        self._check_coverage(first, last)
        unit = "M" if monthly else "D"
        periods = np.arange(np.datetime64(first, unit), np.datetime64(last, unit) + np.timedelta64(1, unit),
                            dtype=f"datetime64[{unit}]")
        paths = []
        for path, begin, end in self.sources:
            lo, hi = max(first, begin), min(last, end)
            if lo > hi:
                continue
            months = np.arange(np.datetime64(lo, "M"), np.datetime64(hi, "M") + MONTH, dtype="datetime64[M]")
            for month in months:
                year, number = map(int, str(month).split("-"))
                partition = path / "grid_daily" / f"year={year}" / f"month={number}" / "data.parquet"
                if partition.is_file():
                    paths.append(str(partition))
                elif path == self.sources[0][0]:
                    raise FileNotFoundError(f"FIRMS archive partition is unavailable: {partition}")
        empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
        by_period = dict.fromkeys(periods, empty)
        if not paths:
            return list(by_period.values())
        bounds, parameters = _bounds(self.bbox, border=0 if monthly else 1)
        daily = f"""
            SELECT date_utc AS time, ((grid_y + {Y_ORIGIN}) * {WIDTH} + grid_x + {X_ORIGIN})::INT AS cell,
                   arg_min(({_frp_sql()}), ({_PLATFORM_RANKS}, -list_position(?, filename)))
                    FILTER (WHERE isfinite(native_grid_frp_mw) AND native_grid_frp_mw >= 0) AS frp
            FROM read_parquet(?, filename=true)
            WHERE detected AND date_utc BETWEEN ? AND ? AND {bounds}
            GROUP BY time, cell
        """
        query = f"SELECT date_trunc('month', time)::DATE AS time, cell, median(frp)::FLOAT AS frp FROM ({daily}) GROUP BY 1, 2" if monthly else daily
        with duckdb.connect(config={"threads": str(self.threads)}) as db:
            data = db.execute(f"SELECT * FROM ({query}) ORDER BY time, cell",
                              [paths, paths, first, last, *parameters]).fetchnumpy()
        stamps = data["time"].astype(f"datetime64[{unit}]")
        values = np.ma.filled(data["frp"], np.nan)
        times, starts = np.unique(stamps, return_index=True)
        for i, time in enumerate(times):
            start = starts[i]
            end = starts[i + 1] if i + 1 < len(starts) else len(stamps)
            by_period[time] = data["cell"][start:end], values[start:end]
        return list(by_period.values())
