"""Build one NOAA-20-scale hotspot dataset and calendar from FIRMS observations."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

from .convert import GRID_X_SCALE, GRID_Y_SCALE, define_grid, sql_string
from .convert_full import connect, from_normalized, normalize, parquet
from .world_calendar import PRODUCTION_ORDER
from .world_features import BLOCK_KM, PLATFORMS

PARTITIONS = "year=*/month=*"
PLATFORM_IDS = ", ".join(f"({sql_string(name)}, {index})" for index, name in enumerate(PLATFORMS))
PRIORITY = "CASE platform_id " + " ".join(
    f"WHEN {platform} THEN {index}" for index, platform in enumerate(PRODUCTION_ORDER)
) + " END"


def merged_days(inputs):
    return f"""
        WITH ranked AS (
            SELECT d.*, p.platform_id, {PRIORITY} AS platform_priority
            FROM {inputs} d
            JOIN (VALUES {PLATFORM_IDS}) p(name, platform_id) ON d.platform = p.name
        ),
        cell_days AS (
            SELECT date_utc, grid_x, grid_y,
                   bit_or(1 << platform_id) AS platform_mask,
                   sum(detected_pixels)::BIGINT AS source_detection_count,
                   min(first_detection_utc)::TIMESTAMP AS first_detection_utc,
                   max(last_detection_utc)::TIMESTAMP AS last_detection_utc,
                   arg_min(struct_pack(
                       value := noaa20_equivalent_frp_mw,
                       method := frp_reference_method,
                       platform := platform
                   ), platform_priority)
                     FILTER (WHERE noaa20_equivalent_frp_mw IS NOT NULL) AS reference,
                   arg_min(struct_pack(
                       value := grid_frp_mw,
                       platform := platform
                   ), platform_priority)
                     FILTER (WHERE grid_frp_mw IS NOT NULL) AS native,
                   arg_min(struct_pack(
                       value := other_platform_record_probability,
                       target := probability_target_platform,
                       source := platform
                   ), (CASE probability_target_platform WHEN 'NOAA-20' THEN 0 ELSE 1 END,
                       platform_priority))
                     FILTER (WHERE other_platform_record_probability IS NOT NULL) AS agreement
            FROM ranked
            GROUP BY date_utc, grid_x, grid_y
        )
        SELECT date_utc, cell_id(grid_x, grid_y) AS grid_id, grid_x, grid_y,
               true AS detected, source_detection_count,
               (platform_mask & 3) <> 0 AS seen_modis,
               (platform_mask & 28) <> 0 AS seen_viirs,
               list_filter({list(PLATFORMS)}, lambda _, i: (platform_mask & (1 << (i - 1))) <> 0)
                 AS source_platforms,
               first_detection_utc, last_detection_utc,
               reference.value AS frp_noaa20_mw,
               reference.method AS frp_reference_method,
               reference.platform AS frp_source_platform,
               native.value AS native_grid_frp_mw,
               native.platform AS native_frp_source_platform,
               agreement.value AS cross_platform_record_probability,
               agreement.target AS probability_target_platform,
               agreement.source AS probability_source_platform
        FROM cell_days
    """


def parquet_union(paths):
    return "read_parquet([" + ", ".join(sql_string(path) for path in paths) + "])"


def write_calendar(db, folders, output):
    queries = []
    for folder in folders:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        first, last = map(sql_string, metadata["dates_utc_inclusive"])
        queries.append(f"""
            SELECT *, month < {first}::DATE OR last_day(month) > {last}::DATE AS partial_input_month
            FROM {parquet(folder / "calendar_blocks.parquet")}
        """)
    source = " UNION ALL ".join(queries)
    db.execute(f"""
        COPY (
            WITH ranked AS (
                SELECT month, bx, grid_by, platform_name, platform AS platform_id,
                       observed_cell_days, activity_noaa20_equivalent_cell_days, partial_input_month,
                       {PRIORITY} AS platform_priority,
                       count(*) OVER (PARTITION BY month,bx,grid_by) AS reporting_platforms
            FROM ({source})
                WHERE activity_noaa20_equivalent_cell_days IS NOT NULL
            )
            SELECT month, bx, grid_by, platform_name AS source_platform, reporting_platforms,
                   observed_cell_days, activity_noaa20_equivalent_cell_days, partial_input_month,
                   CASE WHEN platform_id = 3 THEN 'noaa20_observed'
                        ELSE 'model_estimated' END AS activity_source,
                   greatest(-180, degrees(bx * {BLOCK_KM} / {GRID_X_SCALE})) AS west,
                   degrees(asin(greatest(-1, grid_by * {BLOCK_KM} / {GRID_Y_SCALE}))) AS south,
                   least(180, degrees((bx + 1) * {BLOCK_KM} / {GRID_X_SCALE})) AS east,
                   degrees(asin(least(1, (grid_by + 1) * {BLOCK_KM} / {GRID_Y_SCALE}))) AS north
            FROM ranked
            QUALIFY row_number() OVER (
                PARTITION BY month,bx,grid_by ORDER BY partial_input_month,platform_priority
            ) = 1
            ORDER BY month,grid_by,bx
        ) TO {sql_string(output / "calendar_blocks.parquet")}
          (FORMAT PARQUET, COMPRESSION ZSTD);
        COPY (
            SELECT month, sum(activity_noaa20_equivalent_cell_days)
                          AS activity_noaa20_equivalent_cell_days,
                   count(*) AS blocks_with_records,
                   countif(activity_source = 'noaa20_observed') AS observed_blocks,
                   countif(activity_source = 'model_estimated') AS estimated_blocks,
                   countif(partial_input_month) AS partial_input_blocks,
                   countif(source_platform IN ('Aqua','Terra')) AS modis_blocks,
                   countif(source_platform IN ('S-NPP','NOAA-20','NOAA-21')) AS viirs_blocks
            FROM {parquet(output / "calendar_blocks.parquet")}
            GROUP BY month ORDER BY month
        ) TO {sql_string(output / "calendar.csv")} (FORMAT CSV, HEADER true);
    """)


def write_cells(db, paths, output):
    db.execute(f"""
        COPY (
            WITH cells AS (SELECT DISTINCT grid_x, grid_y FROM {parquet_union(paths)})
            SELECT cell_id(grid_x, grid_y) AS grid_id, grid_x, grid_y,
                   greatest(-180, degrees(grid_x / {GRID_X_SCALE})) AS west,
                   degrees(asin(greatest(-1, grid_y / {GRID_Y_SCALE}))) AS south,
                   least(180, degrees((grid_x + 1) / {GRID_X_SCALE})) AS east,
                   degrees(asin(least(1, (grid_y + 1) / {GRID_Y_SCALE}))) AS north
            FROM cells ORDER BY grid_y, grid_x
        ) TO {sql_string(output / "grid_cells.parquet")}
          (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


def write_observations(folders, output, *, link=False):
    """Keep the original sensor details in one partitioned observation table."""
    for index, folder in enumerate(folders):
        observation_root = folder / "hotspots"
        for file in observation_root.glob(f"{PARTITIONS}/*.parquet"):
            relative = file.relative_to(observation_root)
            destination = output / "observations" / relative.parent / f"{index}-{file.name}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if link:
                os.link(file, destination)
            else:
                shutil.copyfile(file, destination)


def write_hourly(db, output):
    daily = parquet(output / "grid_daily" / PARTITIONS / "data.parquet")
    db.execute(f"""
        COPY (
            SELECT date_trunc('hour', first_detection_utc) AS hour_utc,
                   count(*) AS first_detected_cells
            FROM {daily}
            GROUP BY hour_utc ORDER BY hour_utc
        ) TO {sql_string(output / "hourly.csv")} (FORMAT CSV, HEADER true)
    """)


def assemble(folders, output, work, *, link_observations=False):
    """Assemble one or both already converted sensor archives into the common schema."""
    folders = tuple(Path(folder).resolve() for folder in folders)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifests = [json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
                 for folder in folders]
    sensors = {manifest["input_sensor"] for manifest in manifests}
    if not sensors or not sensors <= {"MODIS", "VIIRS"} or len(sensors) != len(folders):
        raise ValueError("Provide one completed archive per sensor")
    for manifest in manifests:
        if manifest["grid_km"] != 1 or manifest["schema_version"] != 2:
            raise ValueError("Expected version 2 converted inputs on the 1 km grid")
    available = {}
    for folder in folders:
        for file in (folder / "grid_daily").glob(f"{PARTITIONS}/data.parquet"):
            key = file.parent.relative_to(folder / "grid_daily")
            available.setdefault(key, []).append(file)
    if not available:
        raise ValueError("No converted grid partitions found")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        rows = detections = overlap = 0
        dates = []
        with connect(Path(work), 10, "10GB") as db:
            define_grid(db, 1)
            for number, (partition, paths) in enumerate(sorted(available.items()), start=1):
                destination = stage / "grid_daily" / partition / "data.parquet"
                destination.parent.mkdir(parents=True)
                db.execute(f"""
                    COPY ({merged_days(parquet_union(paths))} ORDER BY date_utc,grid_y,grid_x)
                    TO {sql_string(destination)}
                    (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)
                """)
                count, raw, both, first, last = db.execute(f"""
                    SELECT count(*), sum(source_detection_count), countif(seen_modis AND seen_viirs),
                           min(date_utc), max(date_utc)
                    FROM {parquet(destination)}
                """).fetchone()
                rows += count
                detections += raw
                overlap += both
                dates.extend((first, last))
                if number % 24 == 0 or number == len(available):
                    print(f"{number}/{len(available)} months harmonized", flush=True)
            expected = sum(manifest["hotspot_rows"] for manifest in manifests)
            if detections != expected:
                raise ValueError(f"Input detections {expected:,}, harmonized detections {detections:,}")
            write_calendar(db, folders, stage)
            write_hourly(db, stage)
            write_cells(db, [folder / "grid_cells.parquet" for folder in folders], stage)
            cells = db.execute(f"SELECT count(*) FROM {parquet(stage / 'grid_cells.parquet')}").fetchone()[0]
            write_observations(folders, stage, link=link_observations)
            observations = parquet(stage / "observations" / PARTITIONS / "*.parquet")
            if db.execute(f"SELECT count(*) FROM {observations}").fetchone()[0] != expected:
                raise ValueError("The observation table does not account for every source detection")

        metadata = {
            "schema_version": "harmonized-1",
            "date_range_utc_inclusive": [str(min(dates)), str(max(dates))],
            "source_sensors": sorted(sensors),
            "source_detection_count": detections,
            "grid_day_rows": rows,
            "grid_days_seen_by_both_sensors": overlap,
            "grid_cells": cells,
            "grid_km": 1,
            "daily_key": ["date_utc", "grid_id"],
            "daily_frp_reference": "NOAA-20, MW, per-platform reference observed or estimated",
            "priority": [PLATFORMS[i] for i in PRODUCTION_ORDER],
            "monthly_key": ["month", "bx", "grid_by"],
            "monthly_block_km": BLOCK_KM,
            "monthly_activity_unit": "NOAA-20-equivalent active 1 km cell-days",
            "monthly_selection": "Prefer complete input months, then apply the stated platform priority.",
            "observation_storage": "local hard-linked Parquet" if link_observations else "copied Parquet",
            "hourly_metric": "UTC hour of the first detection in each 1 km cell-day",
            "paths": {
                "daily": f"grid_daily/{PARTITIONS}/data.parquet",
                "observations": f"observations/{PARTITIONS}/*.parquet",
                "cells": "grid_cells.parquet",
                "monthly_blocks": "calendar_blocks.parquet",
                "monthly_global": "calendar.csv",
                "hourly_global": "hourly.csv",
            },
        }
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return metadata


def detect_sensor(source):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        records = csv.DictReader(stream)
        if records.fieldnames is None:
            raise ValueError("The input CSV is empty")
        fields = {field.strip().lower(): field for field in records.fieldnames}
        if "satellite" not in fields:
            raise ValueError("FIRMS CSV requires a satellite column")
        for record in records:
            satellite = (record[fields["satellite"]] or "").strip().upper()
            if satellite in {"T", "A", "TERRA", "AQUA"}:
                return "MODIS"
            if satellite in {"N", "SNPP", "S-NPP", "N20", "J1", "NOAA-20", "N21", "J2", "NOAA-21"}:
                return "VIIRS"
            raise ValueError(f"Unknown FIRMS satellite: {satellite!r}")
        raise ValueError("The input CSV has a header and no detections")


def convert_csv(source, output, *, work=None, workers=3):
    """Convert one raw MODIS or VIIRS CSV to the same product as the worldwide archive."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    sensor = detect_sensor(source)
    with tempfile.TemporaryDirectory(prefix="firms-harmonize-", dir=work) as temp:
        processing = Path(temp)
        normalize(source, sensor, processing)
        intermediate = processing / "converted"
        from_normalized(sensor, source, processing, intermediate, workers)
        return assemble((intermediate,), output, processing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="One raw FIRMS MODIS or VIIRS CSV")
    source.add_argument("--from-converted", nargs="+", type=Path,
                        help="Assemble completed per-sensor export directories")
    parser.add_argument("--link-observations", action="store_true",
                        help="Hard-link immutable observation files from same-volume converted archives")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", type=Path, help="Parent for temporary processing data")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    if args.input:
        if args.link_observations:
            parser.error("--link-observations requires --from-converted")
        result = convert_csv(args.input, args.output, work=args.work, workers=args.workers)
    else:
        with tempfile.TemporaryDirectory(prefix="firms-assembly-", dir=args.work) as temp:
            result = assemble(args.from_converted, args.output, temp,
                              link_observations=args.link_observations)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
