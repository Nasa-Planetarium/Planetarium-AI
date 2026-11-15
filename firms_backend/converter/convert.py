"""Convert FIRMS MODIS/VIIRS CSVs into observation, grid, and calendar outputs."""
import argparse
import csv
from datetime import date
import json
import math
from pathlib import Path
import tempfile

import duckdb


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def column(headers, *names, optional=False):
    for name in names:
        if name in headers:
            return '"' + name + '"'
    if optional:
        return "NULL::VARCHAR"
    raise ValueError(f"Missing FIRMS column: {' / '.join(names)}")


def hotspot_query(path, sensor, west, east):
    lon_join = "AND" if west <= east else "OR"
    with path.open(newline="", encoding="utf-8-sig") as file:
        headers = {name.strip().lower() for name in next(csv.reader(file))}
    cols = {name: column(headers, name) for name in (
        "latitude", "longitude", "scan", "track", "acq_date", "acq_time", "satellite",
        "confidence", "version", "frp", "daynight",
    )}
    lat, lon = cols["latitude"], cols["longitude"]
    instrument = column(headers, "instrument", optional=True)
    hotspot_type = column(headers, "type", optional=True)
    if sensor == "MODIS":
        mir, tir = column(headers, "brightness"), column(headers, "bright_t31")
        mir_band, tir_band = "MODIS 21/22", "MODIS 31"
        platform = """
            CASE upper(trim(satellite)) WHEN 'TERRA' THEN 'Terra' WHEN 'T' THEN 'Terra'
                 WHEN 'AQUA' THEN 'Aqua' WHEN 'A' THEN 'Aqua'
                 ELSE error('Unknown MODIS satellite: ' || coalesce(satellite, '<missing>')) END
        """
    else:
        mir = column(headers, "bright_ti4", "brightness")
        tir = column(headers, "bright_ti5", "bright_t31", "brightness_2")
        mir_band, tir_band = "VIIRS I4", "VIIRS I5"
        platform = """
            CASE upper(trim(satellite))
                 WHEN 'SNPP' THEN 'S-NPP' WHEN 'S-NPP' THEN 'S-NPP' WHEN 'N' THEN 'S-NPP'
                 WHEN 'N20' THEN 'NOAA-20' WHEN 'NOAA-20' THEN 'NOAA-20' WHEN 'J1' THEN 'NOAA-20'
                 WHEN 'N21' THEN 'NOAA-21' WHEN 'NOAA-21' THEN 'NOAA-21' WHEN 'J2' THEN 'NOAA-21'
                 ELSE error('Unknown VIIRS satellite: ' || coalesce(satellite, '<missing>')) END
        """
    return f"""
        SELECT acq_date::DATE AS observation_date,
            CASE WHEN regexp_full_match(trim(acq_time), '[0-9]{{1,4}}')
                 THEN strptime(acq_date || ' ' || lpad(trim(acq_time), 4, '0'), '%Y-%m-%d %H%M')
                 ELSE error('Invalid FIRMS acquisition time: ' || coalesce(acq_time, '<missing>'))
            END AS observed_at,
            {lat}::DOUBLE AS lat, {lon}::DOUBLE AS lon,
            {sql_string(sensor)} AS sensor, {platform} AS platform,
            satellite AS satellite_raw, {instrument} AS instrument_raw, version AS version_raw,
            confidence AS confidence_raw, upper(daynight) AS daynight,
            {hotspot_type}::INTEGER AS type_raw,
            CASE {hotspot_type}::INTEGER
                WHEN 0 THEN 'presumed_vegetation_fire'
                WHEN 1 THEN 'active_volcano'
                WHEN 2 THEN 'other_static_land_source'
                WHEN 3 THEN 'offshore_detection'
                ELSE 'unknown'
            END AS hotspot_type,
            scan::DOUBLE AS scan_km, track::DOUBLE AS track_km,
            {mir}::DOUBLE AS bt_mir_k, {sql_string(mir_band)} AS mir_band,
            {tir}::DOUBLE AS bt_tir_k, {sql_string(tir_band)} AS tir_band,
            frp::DOUBLE AS frp_mw, {sql_string(path.name)} AS source_file
        FROM read_csv({sql_string(path)}, header=true, all_varchar=true, delim=',',
                      strict_mode=true,
                      auto_detect=true, sample_size=10000)
        WHERE acq_date BETWEEN ? AND ? AND {lat}::DOUBLE BETWEEN ? AND ?
              AND ({lon}::DOUBLE >= ? {lon_join} {lon}::DOUBLE <= ?)
    """


EARTH_RADIUS_KM = 6371.0088
STANDARD_PARALLEL_COS = math.cos(math.radians(30))
GRID_X_SCALE = EARTH_RADIUS_KM * STANDARD_PARALLEL_COS
GRID_Y_SCALE = EARTH_RADIUS_KM / STANDARD_PARALLEL_COS


def define_grid(db, grid_km):
    db.execute(f"""
        CREATE MACRO cell_x(lon) AS floor(
            {GRID_X_SCALE} * radians(CASE WHEN lon = 180 THEN -180 ELSE lon END) / {grid_km}
        )::INTEGER;
        CREATE MACRO cell_y(lat) AS floor(
            {GRID_Y_SCALE} * sin(radians(lat)) / {grid_km}
        )::INTEGER;
        CREATE MACRO cell_id(x, y) AS concat(x, ':', y);
    """)


def copy_query(db, query, path):
    db.execute(f"COPY ({query}) TO {sql_string(path)} (FORMAT CSV, HEADER true)")


def write_grid_cells(db, path, grid_km):
    cells = db.execute("""
        SELECT DISTINCT grid_x, grid_y, cell_id(grid_x, grid_y)
        FROM map_points ORDER BY grid_x, grid_y
    """)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        output.write('{"type":"FeatureCollection","features":[')
        while batch := cells.fetchmany(4096):
            for x, y, identifier in batch:
                west = max(-180, math.degrees(x * grid_km / GRID_X_SCALE))
                east = min(180, math.degrees((x + 1) * grid_km / GRID_X_SCALE))
                south = math.degrees(math.asin(max(-1, y * grid_km / GRID_Y_SCALE)))
                north = math.degrees(math.asin(min(1, (y + 1) * grid_km / GRID_Y_SCALE)))
                ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
                if count:
                    output.write(",")
                json.dump({
                    "type": "Feature", "id": identifier,
                    "properties": {"grid_id": identifier},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                }, output, separators=(",", ":"))
                count += 1
        output.write("]}\n")
    return count


GRID_DAILY_COLUMNS = """
    date_utc, cell_id(grid_x, grid_y) AS grid_id,
    sensor, platform, detected_pixels, frp_value_count, peak_pixel_frp_mw,
    first_detection_utc, last_detection_utc, grid_frp_mw, frp_observation_minutes,
    other_platform_record_probability, probability_target_platform,
    noaa20_equivalent_frp_mw, frp_reference_method
"""


def build_grid_activity(db):
    db.execute("""
        CREATE TEMP TABLE grid_daily AS
        WITH minute_activity AS (
            SELECT observation_date, grid_x, grid_y, sensor, platform, observed_at,
                   count(*) AS n,
                   count(*) FILTER (WHERE isfinite(frp_mw) AND frp_mw >= 0) AS valid_frp_count,
                   max(frp_mw) FILTER (WHERE isfinite(frp_mw) AND frp_mw >= 0) AS pixel_peak,
                   sum(frp_mw) FILTER (WHERE isfinite(frp_mw) AND frp_mw >= 0) AS minute_frp
            FROM map_points GROUP BY ALL
        )
        SELECT observation_date AS date_utc, grid_x, grid_y, sensor, platform,
               sum(n)::BIGINT AS detected_pixels,
               sum(valid_frp_count)::BIGINT AS frp_value_count,
               max(pixel_peak) AS peak_pixel_frp_mw,
               median(minute_frp) AS grid_frp_mw,
               count(minute_frp) AS frp_observation_minutes,
               strftime(min(observed_at), '%Y-%m-%dT%H:%M:%SZ') AS first_detection_utc,
               strftime(max(observed_at), '%Y-%m-%dT%H:%M:%SZ') AS last_detection_utc,
               NULL::DOUBLE AS other_platform_record_probability,
               NULL::VARCHAR AS probability_target_platform,
               NULL::DOUBLE AS noaa20_equivalent_frp_mw,
               NULL::VARCHAR AS frp_reference_method
        FROM minute_activity GROUP BY observation_date, grid_x, grid_y, sensor, platform
    """)


def write_grid_activity(db, output, agreement_model):
    build_grid_activity(db)
    add_grid_probabilities(db, agreement_model)
    copy_query(db, f"""
        SELECT {GRID_DAILY_COLUMNS} FROM grid_daily
        ORDER BY date_utc, grid_id, sensor, platform
    """, output / "grid_daily.csv")


def load_grid_model(model_path):
    import joblib
    fitted = joblib.load(model_path)
    if fitted.get("version") != 2 or "frp_models" not in fitted:
        raise ValueError("Use the bundled world_daily_model.joblib")
    return fitted


def add_grid_probabilities(db, model_path):
    add_world_grid_outputs(db, load_grid_model(model_path))


def add_world_grid_outputs(db, fitted):
    import numpy as np
    from .world_daily import route_probabilities, standardized_frp

    platforms = fitted["platforms"]
    source_values = ",".join(f"({sql_string(name)},{i})" for i, name in enumerate(platforms))
    target_values = ",".join(
        f"({source},{target},DATE '{fitted['available'][target]}')"
        for source, target in fitted["routes"]
    )
    values = db.execute(f"""
        WITH source(code,platform_id) AS (VALUES {source_values}),
        route(platform_id,target_platform,available) AS (VALUES {target_values}),
        minutes AS (
            SELECT observation_date, grid_x AS x, grid_y AS y, p.platform, platform_id,
                   observed_at, count(*) AS n,
                   sum(frp_mw::FLOAT) FILTER (WHERE isfinite(frp_mw) AND frp_mw >= 0) AS minute_frp,
                   avg((scan_km * track_km)::FLOAT) AS area,
                   avg((daynight = 'D')::INTEGER) AS day_share
            FROM map_points p JOIN source ON p.platform = source.code
            WHERE (type_raw = 0 OR type_raw IS NULL)
              AND CASE WHEN sensor = 'MODIS' THEN try_cast(confidence_raw AS DOUBLE) >= 30
                       ELSE lower(confidence_raw) IN ('n','h') END
            GROUP BY ALL
        ),
        by_day AS (
            SELECT observation_date,x,y,platform AS platform_name,platform_id,
                   month(observation_date) AS mo, sum(n)::BIGINT AS pixels,
                   avg(area) AS pixel_area, median(minute_frp) AS grid_frp_mw,
                   sum(n * day_share) / sum(n) AS day_share, count(*) AS minutes
            FROM minutes GROUP BY observation_date,x,y,platform,platform_id
        )
        SELECT by_day.*, by_day.platform_id AS platform, coalesce(target_platform, -1) AS target_platform
        FROM by_day LEFT JOIN route ON by_day.platform_id = route.platform_id
                                   AND observation_date >= available
        QUALIFY row_number() OVER (PARTITION BY observation_date,x,y,by_day.platform_id
                                   ORDER BY CASE WHEN target_platform = 3 THEN 0 ELSE 1 END) = 1
    """).fetchnumpy()
    if not len(values["pixels"]):
        return
    probability = route_probabilities(fitted, values)
    frp, method = standardized_frp(fitted, values)
    db.register("world_grid_predictions", {
        "day": values["observation_date"], "x": values["x"], "y": values["y"],
        "platform_name": values["platform_name"], "probability": probability,
        "target_name": np.append(np.array(platforms), "")[values["target_platform"]],
        "frp": frp, "method": method,
    })
    db.execute("""
        UPDATE grid_daily AS d
        SET other_platform_record_probability = CASE WHEN isfinite(p.probability) THEN p.probability END,
            probability_target_platform = CASE WHEN isfinite(p.probability) THEN p.target_name END,
            noaa20_equivalent_frp_mw = CASE WHEN isfinite(p.frp) THEN p.frp END,
            frp_reference_method = nullif(p.method,'')
        FROM world_grid_predictions p
        WHERE d.date_utc = p.day AND d.grid_x = p.x AND d.grid_y = p.y AND d.platform = p.platform_name
    """)


def convert(*, sensor, source, output, bbox, start, end, map_start, map_end,
            threads=8, calendar_model=None, agreement_model=None):
    """Convert a single sensor's FIRMS CSV using pretrained models."""
    sensor = sensor.upper()
    if sensor not in ("MODIS", "VIIRS"):
        raise ValueError("sensor must be MODIS or VIIRS")
    validate_options(bbox, start, end, map_start, map_end, threads, 1)
    model_root = Path(__file__).resolve().parent / "global"
    source = Path(source).resolve()
    output = Path(output)
    calendar_model = Path(calendar_model) if calendar_model else default_calendar_model(model_root, bbox)
    agreement_model = Path(agreement_model or model_root / "world_daily_model.joblib")
    for model in (calendar_model, agreement_model):
        if not model.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model}")
    return run(argparse.Namespace(
        modis=source if sensor == "MODIS" else None,
        viirs=source if sensor == "VIIRS" else None,
        output=output, bbox=bbox, start=start, end=end, map_start=map_start, map_end=map_end,
        threads=threads, calendar_model=calendar_model, agreement_model=agreement_model, grid_km=1,
    ))


def default_calendar_model(model_root, bbox):
    west, south, east, north = bbox
    width = math.radians((east - west) % 360 if west > east else east - west) * GRID_X_SCALE
    height = GRID_Y_SCALE * (math.sin(math.radians(north)) - math.sin(math.radians(south)))
    extent = max(width, height)
    for limit, name in ((100, "world_calendar_10km_model.joblib"),
                        (700, "world_calendar_50km_model.joblib")):
        regional = model_root / name
        if extent <= limit and regional.is_file():
            return regional
    return model_root / "world_calendar_model.joblib"


def validate_options(bbox, start, end, map_start, map_end, threads, grid_km):
    west, south, east, north = bbox
    if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south < north <= 90):
        raise ValueError("invalid longitude/latitude bounds")
    if not (start <= end and map_start <= map_end and threads > 0):
        raise ValueError("invalid date range or thread count")
    if grid_km != 1:
        raise ValueError("trained models require --grid-km 1")


def run(args):
    west, south, east, north = args.bbox
    source = (args.modis or args.viirs).resolve()
    sensor = "MODIS" if args.modis else "VIIRS"
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="firms-duckdb-") as tmp:
        with duckdb.connect(config={"threads": str(args.threads), "memory_limit": "6GB",
                                    "temp_directory": tmp, "preserve_insertion_order": "false"}) as db:
            query = hotspot_query(source, sensor, west, east)
            filtered = [str(min(args.start, args.map_start)), str(max(args.end, args.map_end)),
                        south, north, west, east]
            db.execute("CREATE TABLE points AS " + query, filtered)
            print("Loaded AOI observations", flush=True)
            define_grid(db, args.grid_km)
            from .world_calendar import apply_to_points, combined_calendar_query

            model_info = apply_to_points(db, args.calendar_model, args.start, args.end)
            db.execute(f"CREATE TEMP TABLE world_months AS {combined_calendar_query()}")
            start, end = sql_string(args.start), sql_string(args.end)
            copy_query(db, f"""
                SELECT CAST(series.month AS DATE) AS month,
                       world_months.* EXCLUDE (month, activity_noaa20_equivalent_cell_days, activity_source),
                       CASE WHEN series.month >= {start}::DATE AND last_day(series.month) <= {end}::DATE
                            THEN activity_noaa20_equivalent_cell_days
                       END AS activity_noaa20_equivalent_cell_days,
                       CASE WHEN series.month < {start}::DATE
                              OR last_day(series.month) > {end}::DATE
                            THEN 'partial_requested_month'
                            WHEN blocks_with_records IS NULL THEN 'unavailable'
                            ELSE world_months.activity_source END AS activity_source
                FROM generate_series(date_trunc('month', {start}::DATE),
                     date_trunc('month', {end}::DATE), INTERVAL 1 MONTH) AS series(month)
                LEFT JOIN world_months ON series.month = world_months.month
                ORDER BY series.month
            """, args.output / "calendar.csv")
            calendar_rows = (args.end.year - args.start.year) * 12 + args.end.month - args.start.month + 1
            map_start, map_end = sql_string(args.map_start), sql_string(args.map_end)
            db.execute(f"""
                CREATE TEMP VIEW map_points AS
                SELECT *, cell_x(lon) AS grid_x, cell_y(lat) AS grid_y
                FROM points WHERE observation_date BETWEEN {map_start}::DATE AND {map_end}::DATE
            """)
            copy_query(db, """
                SELECT strftime(observed_at, '%Y-%m-%dT%H:%M:%SZ') AS observed_at_utc,
                       lon AS longitude, lat AS latitude, sensor, platform, 'detected' AS detection_status,
                       hotspot_type, type_raw, confidence_raw, scan_km, track_km, frp_mw, daynight,
                       bt_mir_k, mir_band, bt_tir_k, tir_band, version_raw, satellite_raw,
                       instrument_raw, source_file, cell_id(grid_x, grid_y) AS grid_id
                FROM map_points
                ORDER BY observed_at, sensor, platform, lat, lon
            """, args.output / "hotspots.csv")
            map_counts = dict(db.execute(
                "SELECT sensor, count(*) FROM map_points GROUP BY sensor"
            ).fetchall())
            write_grid_activity(db, args.output, args.agreement_model)
            grid_cells = write_grid_cells(db, args.output / "grid_cells.geojson", args.grid_km)
    metadata = {
        "schema_version": 2, "bbox_wsen": args.bbox,
        "input_sensor": sensor, "input_file": source.name,
        "calendar_dates_utc_inclusive": [str(args.start), str(args.end)],
        "map_dates_utc_inclusive": [str(args.map_start), str(args.map_end)],
        "map_observations_by_sensor": map_counts, "calendar_month_rows": calendar_rows,
        "grid_km": args.grid_km, "grid_projection": "spherical equal-area cylindrical, standard parallel 30 degrees",
        "map_grid_cells": grid_cells,
        "map_grid_daily_grain": "UTC day, grid cell, sensor and platform; all map detections",
        "map_grid_strength": "median of minute-level sums of valid pixel FRP in MW per cell, day and platform",
        "map_grid_probability": "probability of a qualifying record from the other reference platform in this cell and UTC day, conditional on a source detection" ,
        "map_grid_frp_reference": "NOAA-20 MW; observed if platform is NOAA-20, otherwise a paired-detection estimate",
        "metric": "monthly sum of distinct UTC date and grid cell pairs per platform",
        "calendar_model": model_info,
        "cohort": "all five platforms; day and night; MODIS confidence >=30, VIIRS n/h; presumed vegetation or unknown",
        "reference": "NOAA-20 recorded activity",
        "file_names": {"hotspots": "hotspots.csv", "calendar": "calendar.csv",
                       "grid_daily": "grid_daily.csv", "grid_cells": "grid_cells.geojson"},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--modis", type=Path)
    source.add_argument("--viirs", type=Path)
    p.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--start", required=True, type=date.fromisoformat)
    p.add_argument("--end", required=True, type=date.fromisoformat)
    p.add_argument("--map-start", required=True, type=date.fromisoformat)
    p.add_argument("--map-end", required=True, type=date.fromisoformat)
    p.add_argument("--grid-km", type=float, default=1)
    model_root = Path(__file__).resolve().parent / "global"
    p.add_argument("--agreement-model", type=Path, default=model_root / "world_daily_model.joblib",
                   help="Trusted local world_daily_model.joblib trained at 1 km")
    p.add_argument("--calendar-model", type=Path,
                   help="Trusted local calendar model. Defaults to a packaged model chosen by region size")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        validate_options(args.bbox, args.start, args.end, args.map_start, args.map_end, args.threads, args.grid_km)
        metadata = convert(
            sensor="MODIS" if args.modis else "VIIRS", source=args.modis or args.viirs,
            output=args.output, bbox=args.bbox, start=args.start, end=args.end,
            map_start=args.map_start, map_end=args.map_end, threads=args.threads,
            calendar_model=args.calendar_model, agreement_model=args.agreement_model,
        )
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
