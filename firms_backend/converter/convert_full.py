"""Convert one complete worldwide FIRMS CSV into partitioned Parquet and monthly CSV."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import shutil
import tempfile
import warnings

import duckdb

from .convert import (
    GRID_DAILY_COLUMNS, GRID_X_SCALE, GRID_Y_SCALE, add_world_grid_outputs,
    build_grid_activity, define_grid, hotspot_query, load_grid_model, sql_string,
)

HOTSPOT_COLUMNS = """
    observed_at AS observed_at_utc, observation_date AS date_utc,
    lon AS longitude, lat AS latitude, sensor, platform, 'detected' AS detection_status,
    hotspot_type, type_raw, confidence_raw, scan_km, track_km, frp_mw, daynight,
    bt_mir_k, mir_band, bt_tir_k, tir_band, version_raw, satellite_raw, instrument_raw,
    source_file, cell_id(cell_x(lon),cell_y(lat)) AS grid_id,
    cell_x(lon) AS grid_x, cell_y(lat) AS grid_y,
    year(observation_date) AS year, month(observation_date) AS month
"""
READER_OPTIONS = {"preserve_insertion_order": "false"}
MODEL_ROOT = Path(__file__).resolve().parent / "global"


def connect(work, threads, memory):
    return duckdb.connect(config={
        **READER_OPTIONS, "threads": str(threads), "memory_limit": memory,
        "temp_directory": str(work / "spill")
    })


def parquet(path):
    return f"read_parquet({sql_string(path)})"


def normalize(source, sensor, work):
    target = work / "hotspots"
    with connect(work, 10, "10GB") as db:
        define_grid(db, 1)
        query = hotspot_query(source, sensor, -180, 180)
        # Full input range, including future FIRMS acquisitions.
        filters = ("0001-01-01", "9999-12-31", -90, 90, -180, 180)
        db.execute(f"""
            COPY (SELECT {HOTSPOT_COLUMNS} FROM ({query}))
            TO {sql_string(target)}
            (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (year, month),
             ROW_GROUP_SIZE 131072)
        """, filters)


def convert_month(partition, output, work, fitted, calendar):
    relative = partition.relative_to(work / "hotspots")
    points = parquet(partition / "*.parquet")
    daily = output / "grid_daily" / relative / "data.parquet"
    monthly = work / "monthly" / relative / "data.parquet"
    daily.parent.mkdir(parents=True, exist_ok=True)
    monthly.parent.mkdir(parents=True, exist_ok=True)
    temp_daily = daily.with_suffix(".parquet.tmp")
    with connect(work / "monthly" / relative, 6, "5GB") as db:
        define_grid(db, 1)
        db.execute(f"""
            CREATE TEMP VIEW points AS
            SELECT *, date_utc AS observation_date, observed_at_utc AS observed_at,
                   longitude AS lon, latitude AS lat
            FROM {points};
            CREATE TEMP VIEW map_points AS SELECT * FROM points
        """)
        build_grid_activity(db)
        add_world_grid_outputs(db, fitted)
        db.execute(f"""
            COPY (SELECT {GRID_DAILY_COLUMNS}, grid_x, grid_y
                  FROM grid_daily ORDER BY date_utc, grid_y, grid_x, platform)
            TO {sql_string(temp_daily)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)
        """)
        source_rows, first, last = db.execute(
            "SELECT count(*), min(date_utc), max(date_utc) FROM points"
        ).fetchone()
        grid_rows, detections = db.execute(
            "SELECT count(*), sum(detected_pixels) FROM grid_daily"
        ).fetchone()
        if source_rows != detections:
            raise ValueError(f"{relative}: hotspots {source_rows}, grid detections {detections}")
        from .world_calendar import apply_calendar_model

        apply_calendar_model(db, calendar, first, last)
        db.execute(f"""
            COPY (SELECT * FROM platform_calendar)
            TO {sql_string(monthly)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    shutil.copytree(partition, output / "hotspots" / relative, dirs_exist_ok=True)
    temp_daily.replace(daily)
    return source_rows, grid_rows, first, last


def summarize(sensor, source, output, work, counts):
    from .world_calendar import PRODUCTION_ORDER, combined_calendar_query
    from .world_features import PLATFORMS

    first = min(row[2] for row in counts)
    last = max(row[3] for row in counts)
    platform_ids = (0, 1) if sensor == "MODIS" else (2, 3, 4)
    order = "CASE platform " + " ".join(f"WHEN {p} THEN {i}" for i, p in enumerate(PRODUCTION_ORDER)) + " END"
    names = ",".join(f"({i},{sql_string(PLATFORMS[i])})" for i in platform_ids)
    daily = parquet(output / "grid_daily/year=*/month=*/*.parquet")
    hotspots = parquet(output / "hotspots/year=*/month=*/*.parquet")
    with connect(work, 10, "8GB") as db:
        define_grid(db, 1)
        db.execute(f"""
            CREATE TEMP VIEW platform_calendar AS
            SELECT * FROM read_parquet({sql_string(work / "monthly/year=*/month=*/*.parquet")},
                                       hive_partitioning=false);
            COPY ({combined_calendar_query(platform_ids)})
            TO {sql_string(output / "calendar.csv")} (FORMAT CSV, HEADER true);
            COPY (
                SELECT p.*, n.platform_name,
                       row_number() OVER(PARTITION BY month,bx,grid_by ORDER BY {order}) = 1 AS selected
                FROM platform_calendar p JOIN (VALUES {names}) n(id,platform_name) ON p.platform=n.id
                ORDER BY month,bx,grid_by,platform
            ) TO {sql_string(output / "calendar_blocks.parquet")} (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        db.execute(f"""
            COPY (
                WITH cells AS (SELECT DISTINCT grid_x, grid_y FROM {daily})
                SELECT cell_id(grid_x, grid_y) AS grid_id, grid_x, grid_y,
                       greatest(-180, degrees(grid_x / {GRID_X_SCALE})) AS west,
                       degrees(asin(greatest(-1, grid_y / {GRID_Y_SCALE}))) AS south,
                       least(180, degrees((grid_x + 1) / {GRID_X_SCALE})) AS east,
                       degrees(asin(least(1, (grid_y + 1) / {GRID_Y_SCALE}))) AS north
                FROM cells ORDER BY grid_y, grid_x
            ) TO {sql_string(output / "grid_cells.parquet")} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        cells = db.execute(
            f"SELECT count(*) FROM {parquet(output / 'grid_cells.parquet')}"
        ).fetchone()[0]
        observed_count = db.execute(
            f"SELECT count(*) FROM {hotspots}"
        ).fetchone()[0]
        grid_count, pixel_count = db.execute(
            f"SELECT count(*), sum(detected_pixels) FROM {daily}"
        ).fetchone()
        if observed_count != pixel_count or observed_count != sum(row[0] for row in counts):
            raise ValueError("Saved products differ from monthly detection counts")
        db.execute(f"""
            COPY (
                SELECT date_trunc('hour', observed_at_utc) AS hour_utc, count(*) AS detected_pixels
                FROM {hotspots} GROUP BY hour_utc ORDER BY hour_utc
            ) TO {sql_string(output / "hourly.csv")} (FORMAT CSV, HEADER true)
        """)

    metadata = {
        "schema_version": 2,
        "input_sensor": sensor,
        "input_file": source.name,
        "bbox_wsen": [-180, -90, 180, 90],
        "dates_utc_inclusive": [str(first), str(last)],
        "hotspot_rows": observed_count,
        "grid_daily_rows": grid_count,
        "grid_cells": cells,
        "observed_months": len(counts),
        "grid_km": 1,
        "grid_projection": "spherical cylindrical equal-area, standard parallel 30°",
        "hotspot_timestamp_time_zone": "UTC",
        "grid_daily_grain": "UTC day, 1 km cell, platform; observed detections only",
        "grid_frp_mw": "median of per-minute sums of valid pixel FRP",
        "other_platform_record_probability": "probability of a qualifying detection by the named reference platform in the same cell and UTC day, conditional on a source detection",
        "noaa20_equivalent_frp_mw": "observed on NOAA-20; estimated from qualifying source detections on other platforms",
        "calendar_metric": "NOAA-20-equivalent active 1 km cell-days, observed/estimated within 250 km monthly blocks",
        "calendar_selection": "one platform per block and month: NOAA-20, S-NPP, NOAA-21, Aqua, Terra, restricted to the input sensor",
        "hourly_metric": "raw source detections by UTC hour; missing rows have zero detections",
        "models": {
            "grid_daily": "world_daily_model.joblib",
            "calendar": "world_calendar_model.joblib",
        },
        "paths": {
            "hotspots": "hotspots/year=*/month=*/*.parquet",
            "grid_daily": "grid_daily/year=*/month=*/*.parquet",
            "grid_cells": "grid_cells.parquet",
            "calendar": "calendar.csv",
            "calendar_blocks": "calendar_blocks.parquet",
            "hourly": "hourly.csv",
        },
    }
    temp_metadata = output / "metadata.json.tmp"
    temp_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_metadata.replace(output / "metadata.json")
    print(json.dumps(metadata), flush=True)


def from_normalized(sensor, source, work, output, workers, previous_counts=None):
    import joblib

    fitted = load_grid_model(MODEL_ROOT / "world_daily_model.joblib")
    calendar = joblib.load(MODEL_ROOT / "world_calendar_model.joblib")
    partitions = sorted((work / "hotspots").glob("year=*/month=*"),
                        key=lambda path: tuple(int(part.split("=")[1]) for part in path.parts[-2:]))
    if not partitions:
        raise ValueError(f"No observation rows in {source}")
    previous_counts = previous_counts or {}
    output.mkdir(parents=True, exist_ok=bool(previous_counts))
    (work / "monthly").mkdir(exist_ok=bool(previous_counts))
    counts = list(previous_counts.values())
    remaining = (month for month in partitions if month not in previous_counts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        for _ in range(workers):
            month = next(remaining, None)
            if month is not None:
                pending.add(pool.submit(convert_month, month, output, work, fitted, calendar))
        while pending:
            results, pending = wait(pending, return_when=FIRST_COMPLETED)
            for result in results:
                counts.append(result.result())
                next_month = next(remaining, None)
                if next_month is not None:
                    pending.add(pool.submit(convert_month, next_month, output, work, fitted, calendar))
            if len(counts) % 12 < len(results) or len(counts) == len(partitions):
                print(f"{sensor}: {len(counts)}/{len(partitions)} months saved", flush=True)
    summarize(sensor, source, output, work, counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--modis", type=Path)
    source.add_argument("--viirs", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", type=Path, help="Parent directory for temporary working files")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    sensor = "MODIS" if args.modis else "VIIRS"
    path = (args.modis or args.viirs).resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    with tempfile.TemporaryDirectory(prefix="firms-world-", dir=args.work) as tmp:
        work = Path(tmp)
        normalize(path, sensor, work)
        from_normalized(sensor, path, work, args.output, args.workers)


if __name__ == "__main__":
    main()
