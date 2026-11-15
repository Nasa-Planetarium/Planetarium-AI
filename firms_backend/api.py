"""Direct Python and DuckDB interfaces for the Planetarium backend."""
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import tempfile

import duckdb


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data" / "processed"
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


def convert_api_csv(payload, output, *, bbox, date_from, date_to, workers=3):
    """Convert a FIRMS Area API response and record the queried region and UTC days."""
    from .converter.harmonize import detect_sensor

    if not isinstance(payload, bytes):
        raise TypeError("Provide the FIRMS CSV response body as bytes")
    bounds = _bbox(bbox)
    first, last = _day(date_from), _day(date_to)
    if first > last:
        raise ValueError("date_from must be on or before date_to")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".firms-api-", dir=output.parent) as temp:
        folder = Path(temp)
        source = folder / "firms-api.csv"
        source.write_bytes(payload)
        stage = folder / "update"
        if detect_sensor(source, allow_empty=True):
            metadata = convert_firms(source, stage, work=folder, workers=workers)
            observed_first, observed_last = map(_day, metadata["date_range_utc_inclusive"])
            if not first <= observed_first <= observed_last <= last:
                raise ValueError("FIRMS acquisitions fall outside the stated UTC date range")
        else:
            stage.mkdir()
            metadata = {"schema_version": "harmonized-1", "grid_km": 1, "daily_key": ["date_utc", "grid_id"],
                        "source_sensors": [], "source_detection_count": 0, "grid_day_rows": 0}
        metadata.update(bbox_wsen=list(bounds), coverage_utc_inclusive=[str(first), str(last)])
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return metadata


def _day(value):
    if isinstance(value, str):
        return date.fromisoformat(value)
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError("Expected a UTC date in YYYY-MM-DD format")
    return value


def _bbox(bbox):
    west, south, east, north = map(float, bbox)
    if not (-180 <= west <= 180 and -180 <= east <= 180 and west != east
            and -90 <= south < north <= 90):
        raise ValueError("Expected (west, south, east, north) in WGS84 degrees")
    return west, south, east, north


def _spatial(bbox):
    if bbox is None:
        return "TRUE", []
    west, south, east, north = _bbox(bbox)
    x = "(longitude BETWEEN ? AND ?)" if west < east else "(longitude >= ? OR longitude <= ?)"
    return f"{x} AND latitude BETWEEN ? AND ?", [west, east, south, north]


class FirmsStore:
    """Bounded map queries using published Parquet partitions and their native IDs."""

    def __init__(self, asset_root=None, *, threads=4):
        data_root = asset_root if asset_root is not None else os.environ.get("FIRMS_BACKEND_DATA_HOME", DATA_ROOT)
        self.root = Path(data_root).resolve()
        self.threads = threads

    def metadata(self):
        available = {
            "archive": self.root / "dataset/metadata.json",
        }
        return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in available.items() if path.is_file()}

    def detected(self, day, *, bbox=None, limit=5000):
        """Return recorded 1 km grid-day hotspots, with coordinates and raw provenance."""
        day = self._archive_day(day)
        relative = f"dataset/grid_daily/year={day.year}/month={day.month}/data.parquet"
        return self._rows(
            relative, f"SELECT * EXCLUDE(year,month), {COORDS} FROM read_parquet(?) WHERE date_utc = ?",
            [day], bbox, "source_detection_count", limit,
        )

    def observations(self, day, *, bbox=None, limit=5000):
        """Return normalized FIRMS source observations for the selected UTC date."""
        day = self._archive_day(day)
        relative = f"dataset/observations/year={day.year}/month={day.month}/*.parquet"
        return self._rows(
            relative, "SELECT * EXCLUDE(year,month) FROM read_parquet(?) WHERE date_utc = ?",
            [day], bbox, "observed_at_utc", limit,
        )

    def _archive_day(self, value):
        day = _day(value)
        manifest = json.loads((self.root / "dataset/metadata.json").read_text(encoding="utf-8"))
        first, last = map(date.fromisoformat, manifest["date_range_utc_inclusive"])
        if not first <= day <= last:
            raise FileNotFoundError(f"UTC day {day} is outside the archive [{first}, {last}]")
        return day

    def _rows(self, relative, query, params, bbox, order, limit):
        if type(limit) is not int or not 1 <= limit <= 50000:
            raise ValueError("limit must be an integer between 1 and 50000")
        path = self.root / relative
        if not any(path.parent.glob(path.name)):
            raise FileNotFoundError(f"FIRMS partition is unavailable: {path}")
        predicate, location = _spatial(bbox)
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
