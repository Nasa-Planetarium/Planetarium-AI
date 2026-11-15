"""FIRMS CSV normalization and grid SQL."""
import csv
import math


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


def load_grid_model(model_path):
    import joblib
    fitted = joblib.load(model_path)
    if fitted.get("version") != 2 or "frp_models" not in fitted:
        raise ValueError("Use the bundled world_daily_model.joblib")
    return fitted


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
