"""Convert monthly instrument summaries to NOAA-20 equivalents."""

import numpy as np

from .convert import sql_string
from .world_features import BLOCK_KM, PLATFORMS, monthly_features_query

REFERENCE = 3
PRODUCTION_ORDER = (3, 2, 4, 1, 0)


def dated_rows(db):
    return db.execute("""
        SELECT *, year(month)::INT AS yr, month(month)::INT AS mo,
               extract(day FROM last_day(month))::INT AS days_in_month,
               date_diff('month', DATE '2000-01-01',month)::INT AS month_index
        FROM monthly ORDER BY month,bx,grid_by,platform
    """).fetchnumpy()


def matrix(rows, block_km=BLOCK_KM):
    a = rows["cell_days"]
    month_angle = (rows["mo"] - 1) * (2 * np.pi / 12)
    lon_angle = (rows["bx"] + .5) * block_km * (np.pi / 17333.5)
    features = [
        rows["platform"], np.log1p(a), np.log1p(rows["pixels"] / a),
        rows["active_days"] / rows["days_in_month"], np.log1p(rows["pixel_area"]),
        rows["nrt_share"], rows["day_share"], np.log1p(rows["mean_cell_day_peak"]),
        np.log1p(rows["frp_sum"] / a), np.sin(month_angle), np.cos(month_angle),
        np.sin(lon_angle), np.cos(lon_angle), (rows["grid_by"] + .5) * block_km / 7356.0,
    ]
    return np.column_stack([np.ma.filled(np.ma.asarray(f, dtype=float), np.nan) for f in features])


def predict(bundle, rows):
    features = matrix(rows, bundle["block_km"])
    result = np.full(len(features), np.nan)
    for platform in range(len(PLATFORMS)):
        subset = rows["platform"] == platform
        if not subset.any():
            continue
        if platform == REFERENCE:
            result[subset] = rows["cell_days"][subset]
            continue
        selection, factor = bundle["selected"][platform], bundle["factors"][platform]
        if selection == "unadjusted":
            result[subset] = rows["cell_days"][subset]
        elif selection == "constant_ratio":
            result[subset] = rows["cell_days"][subset] * factor
        else:
            result[subset] = np.maximum(0, bundle["models"][selection].predict(features[subset]))
    return result


def combined_calendar_query(options=PRODUCTION_ORDER):
    order = "CASE platform " + " ".join(f"WHEN {p} THEN {i}" for i, p in enumerate(PRODUCTION_ORDER)) + " END"
    codes = ",".join(map(str, options))
    return f"""
        WITH chosen AS (
            SELECT * FROM platform_calendar WHERE platform IN ({codes})
            QUALIFY row_number() OVER(PARTITION BY month,bx,grid_by ORDER BY {order}) = 1
        )
        SELECT CAST(month AS DATE) AS month, count(*) AS blocks_with_records,
               sum(observed_cell_days)::BIGINT AS selected_platform_cell_days,
               sum(activity_noaa20_equivalent_cell_days) AS activity_noaa20_equivalent_cell_days,
               count(*) FILTER (WHERE platform=3) AS observed_blocks,
               count(*) FILTER (WHERE platform<>3) AS estimated_blocks,
               count(*) FILTER (WHERE platform=0) AS terra_blocks,
               count(*) FILTER (WHERE platform=1) AS aqua_blocks,
               count(*) FILTER (WHERE platform=2) AS snpp_blocks,
               count(*) FILTER (WHERE platform=4) AS noaa21_blocks,
               CASE WHEN estimated_blocks = 0 THEN 'noaa20_observed'
                    WHEN observed_blocks = 0 THEN 'estimated_from_platform'
                    ELSE 'observed_and_estimated' END AS activity_source
        FROM chosen GROUP BY month ORDER BY month
    """


def apply_calendar_model(db, model, start, end):
    platforms = ",".join(f"({sql_string(name)},{i})" for i, name in enumerate(PLATFORMS))
    db.execute(f"""
        CREATE TEMP VIEW source_rows AS
        SELECT observation_date AS day, p.platform_id AS platform,
               cell_x(lon) AS x, cell_y(lat) AS y,
               CASE WHEN isfinite(frp_mw) AND frp_mw >= 0 THEN frp_mw::FLOAT END AS frp,
               (scan_km * track_km)::FLOAT AS pixel_area,
               ends_with(version_raw,'NRT')::UTINYINT AS nrt,
               (daynight = 'D')::UTINYINT AS daytime
        FROM points JOIN (VALUES {platforms}) p(name,platform_id) ON points.platform=p.name
        WHERE observation_date BETWEEN {sql_string(start)}::DATE AND {sql_string(end)}::DATE
          AND (type_raw = 0 OR type_raw IS NULL)
          AND CASE WHEN sensor = 'MODIS' THEN try_cast(confidence_raw AS DOUBLE) >= 30
                   ELSE lower(confidence_raw) IN ('n','h') END;
        CREATE TEMP TABLE monthly AS {monthly_features_query('source_rows', model['block_km'])};
    """)
    rows = dated_rows(db)
    activity = predict(model, rows) if len(rows["platform"]) else np.empty(0, dtype=float)
    db.register("world_activity", {"month": rows["month"], "bx": rows["bx"],
                                   "grid_by": rows["grid_by"], "platform": rows["platform"],
                                   "observed_cell_days": rows["cell_days"],
                                   "activity_noaa20_equivalent_cell_days": activity})
    db.execute("""
        CREATE TEMP TABLE platform_calendar AS
        SELECT * FROM world_activity
    """)
    return {
        "block_km": model["block_km"],
        "selection": model.get("selection", "platform_mae_2024"),
        "by_platform": {PLATFORMS[k]: v for k, v in model["selected"].items()},
    }
