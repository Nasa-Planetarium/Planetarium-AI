"""Direct Python and DuckDB interfaces for the Planetarium backend."""
from datetime import date, datetime
import json
import math
import os
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parent
X_SCALE = 6371.0088 * math.cos(math.radians(30))
Y_SCALE = 6371.0088 / math.cos(math.radians(30))
COORDS = f"""
    (greatest(-180, degrees(grid_x / {X_SCALE}))
        + least(180, degrees((grid_x + 1) / {X_SCALE}))) / 2 AS longitude,
    degrees(asin((greatest(-1, grid_y / {Y_SCALE})
        + least(1, (grid_y + 1) / {Y_SCALE})) / 2)) AS latitude
"""


def convert_firms(source, output, *, work=None, workers=3):
    """Convert one FIRMS MODIS or VIIRS CSV to the common archive schema."""
    from .converter.harmonize import convert_csv

    return convert_csv(source, output, work=work, workers=workers)


def _day(value):
    if isinstance(value, str):
        return date.fromisoformat(value)
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError("Expected a UTC date in YYYY-MM-DD format")
    return value


def _spatial(bbox):
    if bbox is None:
        return "TRUE", []
    west, south, east, north = map(float, bbox)
    if not (-180 <= west <= 180 and -180 <= east <= 180 and west != east
            and -90 <= south < north <= 90):
        raise ValueError("Expected (west, south, east, north) in WGS84 degrees")
    x = "(longitude BETWEEN ? AND ?)" if west < east else "(longitude >= ? OR longitude <= ?)"
    return f"{x} AND latitude BETWEEN ? AND ?", [west, east, south, north]


class FirmsStore:
    """Bounded map queries using published Parquet partitions and their native IDs."""

    def __init__(self, asset_root=None, *, threads=4):
        data_root = asset_root if asset_root is not None else os.environ.get("FIRMS_BACKEND_DATA_HOME", ROOT)
        self.root = Path(data_root).resolve()
        self.threads = threads

    def metadata(self):
        available = {
            "archive": self.root / "dataset/metadata.json",
            "daily_forecast": self.root / "predictions/daily/metadata.json",
            "annual_forecast": self.root / "predictions/annual/metadata.json",
        }
        return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in available.items() if path.is_file()}

    def detected(self, day, *, bbox=None, limit=5000):
        """Return recorded 1 km grid-day hotspots, with coordinates and raw provenance."""
        day = self._archive_day(day)
        relative = f"dataset/grid_daily/year={day.year}/month={day.month}/data.parquet"
        return self._rows(
            relative, f"SELECT * EXCLUDE(year,month), {COORDS} FROM read_parquet(?) WHERE date_utc = ?",
            [day], bbox, None, None, "source_detection_count", limit,
        )

    def observations(self, day, *, bbox=None, limit=5000):
        """Return normalized FIRMS source observations for the selected UTC date."""
        day = self._archive_day(day)
        relative = f"dataset/observations/year={day.year}/month={day.month}/*.parquet"
        return self._rows(
            relative, "SELECT * EXCLUDE(year,month) FROM read_parquet(?) WHERE date_utc = ?",
            [day], bbox, None, None, "observed_at_utc", limit,
        )

    def forecast_daily(self, day, *, bbox=None, minimum=0.0, limit=5000):
        """Return the saved UTC-day detection probabilities and conditional FRP."""
        day = _day(day)
        return self._rows(
            f"predictions/daily/day={day}/data.parquet",
            "SELECT * EXCLUDE(day) FROM read_parquet(?)",
            [], bbox, "p_detected", minimum, "p_detected", limit,
        )

    def forecast_monthly(self, month, *, bbox=None, minimum=0.0, limit=5000):
        """Return the saved monthly probability of one or more detections and conditional FRP."""
        if isinstance(month, str) and len(month) == 7:
            month = month + "-01"
        day = _day(month)
        if day.day != 1:
            raise ValueError("Expected a UTC month in YYYY-MM format or its first day")
        return self._rows(
            f"predictions/annual/month={day:%Y-%m}/data.parquet",
            "SELECT * EXCLUDE(month) FROM read_parquet(?)",
            [], bbox, "p_detected_any", minimum, "p_detected_any", limit,
        )

    def _archive_day(self, value):
        day = _day(value)
        manifest = json.loads((self.root / "dataset/metadata.json").read_text(encoding="utf-8"))
        first, last = map(date.fromisoformat, manifest["date_range_utc_inclusive"])
        if not first <= day <= last:
            raise FileNotFoundError(f"UTC day {day} is outside the archive [{first}, {last}]")
        return day

    def _rows(self, relative, query, params, bbox, probability, minimum, order, limit):
        if type(limit) is not int or not 1 <= limit <= 50000:
            raise ValueError("limit must be an integer between 1 and 50000")
        path = self.root / relative
        if not any(path.parent.glob(path.name)):
            raise FileNotFoundError(f"FIRMS partition is unavailable: {path}")
        predicate, location = _spatial(bbox)
        if minimum is not None:
            if not 0 <= minimum <= 1:
                raise ValueError("minimum must be between 0 and 1")
            predicate = f"{predicate} AND {probability} >= ?"
            location.append(minimum)
        with duckdb.connect(config={"threads": str(self.threads)}) as db:
            result = db.execute(f"""
                SELECT * FROM ({query}) WHERE {predicate}
                ORDER BY {order} DESC, grid_id LIMIT ?
            """, [str(path), *params, *location, limit])
            columns = [column[0] for column in result.description]
            return [
                dict(zip(columns, (v.isoformat() if isinstance(v, date) else v for v in row), strict=True))
                for row in result.fetchall()
            ]
