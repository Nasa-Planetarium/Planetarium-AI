"""Worldwide single-platform to NOAA-20 calendar conversion and model comparison."""
import argparse
import json
from pathlib import Path

import duckdb
import joblib
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .convert import sql_string
from .world_features import BLOCK_KM, PLATFORMS

REFERENCE = 3
PRODUCTION_ORDER = (3, 2, 4, 1, 0)
MODEL_FEATURES = (
    "platform", "log_cell_days", "log_pixels_per_cell_day", "active_days_fraction",
    "log_pixel_area", "nrt_share", "day_share", "log_mean_cell_day_peak",
    "log_frp_per_cell_day", "month_sin", "month_cos", "longitude_sin", "longitude_cos",
    "latitude_equal_area",
)


def load_features(db, path):
    db.execute(f"""
        CREATE OR REPLACE TABLE monthly AS
        SELECT * FROM read_parquet({sql_string(path / '*_monthly_*.parquet')})
    """)
    return dated_rows(db)


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
        elif selection == "weighted_logratio":
            result[subset] = rows["cell_days"][subset] * np.exp(
                bundle["models"][selection].predict(features[subset])
            )
        elif selection == "tweedie_ratio_blend":
            counts = rows["cell_days"][subset]
            design = features[subset]
            tweedie = bundle["models"]["shared_tweedie"].predict(design)
            ratio = counts * np.exp(bundle["models"]["weighted_logratio"].predict(design))
            weight = bundle["blend_weights"][platform]
            result[subset] = weight * tweedie + (1 - weight) * ratio
        else:
            model = bundle["individual"].get(platform) if selection == "individual_tweedie" else bundle["models"][selection]
            result[subset] = np.maximum(0, model.predict(features[subset]))
    return result


def select_metric(y, p, mask):
    return float(np.abs(y[mask] - p[mask]).mean())


def scores(target, candidates, mask):
    if not mask.any():
        return None
    actual = target[mask]
    values = {}
    for key, output in candidates.items():
        errors = output[mask] - actual
        values[key] = {
            "mae": float(np.mean(np.abs(errors))),
            "wape": float(np.sum(np.abs(errors)) / actual.sum()) if actual.sum() else None,
            "relative_bias": float(errors.sum() / actual.sum()) if actual.sum() else None,
        }
    return {"n": int(mask.sum()), "results": values}


def fit(db, directory, block_km):
    all_rows = load_features(db, directory)
    paired = db.execute("""
        WITH reference_rows AS (
            SELECT month,bx,grid_by,cell_days FROM monthly WHERE platform = 3
        )
        SELECT source.*,year(source.month)::INT AS yr, month(source.month)::INT AS mo,
               extract(day FROM last_day(source.month))::INT AS days_in_month,
               coalesce(ref.cell_days,0)::DOUBLE AS target,
               ref.cell_days IS NOT NULL AS paired_positive
        FROM monthly source LEFT JOIN reference_rows ref USING (month,bx,grid_by)
        WHERE source.platform <> 3 AND source.month >= DATE '2018-04-01'
        ORDER BY source.month, source.bx, source.grid_by, source.platform
    """).fetchnumpy()
    x = matrix(paired, block_km)
    y = paired["target"]
    p = paired["platform"]
    year = paired["yr"]
    parent_x = paired["bx"] * block_km // BLOCK_KM
    parent_y = paired["grid_by"] * block_km // BLOCK_KM
    partition = (parent_x * 271 + parent_y * 509) % 10 == 0
    is_training = (year <= 2023) & ~partition
    is_selection = (year == 2024) & ~partition
    is_testing = year == 2025
    is_recent = year == 2026
    count = paired["cell_days"].astype(float)
    factors = {int(i): float(y[is_training & (p == i)].sum() / count[is_training & (p == i)].sum())
               for i in np.unique(p)}
    trained = {}
    estimators = {
        "shared_hist_poisson": HistGradientBoostingRegressor(
            loss="poisson", max_iter=200, max_leaf_nodes=31, l2_regularization=2,
            early_stopping=False, random_state=42),
        "shared_tweedie": LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.4, n_estimators=400,
            learning_rate=.055, num_leaves=127, min_child_samples=100,
            reg_lambda=4, verbosity=-1, n_jobs=8, random_state=42),
        "shared_absolute_error": HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=175, max_leaf_nodes=31,
            l2_regularization=2, early_stopping=False, random_state=42),
    }
    predictions = {"unadjusted": count,
                   "constant_ratio": count * np.array([factors[i] for i in p])}
    ridge = Ridge(alpha=10).fit(np.nan_to_num(x[is_training], nan=-1),
                                np.log1p(y[is_training]) - np.log1p(count[is_training]))
    predictions["rich_ridge_baseline"] = count * np.exp(ridge.predict(np.nan_to_num(x, nan=-1)))
    for name, model in estimators.items():
        if isinstance(model, LGBMRegressor):
            model.fit(x[is_training], y[is_training], categorical_feature=[0])
        else:
            model.fit(x[is_training], y[is_training])
        trained[name] = model
        predictor = model.booster_ if isinstance(model, LGBMRegressor) else model
        predictions[name] = np.maximum(0, predictor.predict(x))
        print("finished", name, flush=True)

    individual = {}
    individual_prediction = np.empty(len(y), dtype=float)
    for platform in factors:
        subset = p == platform
        mask = is_training & subset
        model = LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.4, n_estimators=240,
            learning_rate=.055, num_leaves=31, min_child_samples=90,
            reg_lambda=3, verbosity=-1, n_jobs=8, random_state=42,
        ).fit(x[mask], y[mask])
        individual[platform] = model.booster_
        individual_prediction[subset] = np.maximum(0, individual[platform].predict(x[subset]))
    predictions["individual_tweedie"] = individual_prediction
    ratio_model = LGBMRegressor(
        objective="regression_l1", num_leaves=47, n_estimators=350, learning_rate=.06,
        min_child_samples=80, reg_lambda=3, verbosity=-1, n_jobs=8, random_state=42,
    ).fit(
        x[is_training], np.log((y[is_training] + 1) / (count[is_training] + 1)),
        sample_weight=np.sqrt(count[is_training]), categorical_feature=[0],
    )
    trained["weighted_logratio"] = ratio_model
    predictions["weighted_logratio"] = count * np.exp(ratio_model.booster_.predict(x))
    eligible_methods = [k for k in predictions if k != "rich_ridge_baseline"]
    selected = {platform: min(eligible_methods, key=lambda k: select_metric(
        y, predictions[k], is_selection & (p == platform))) for platform in factors}
    blend_weights = {}
    blend = predictions["shared_tweedie"].copy()
    for platform in (0, 1):
        subset = p == platform
        validation = is_selection & subset
        a = predictions["shared_tweedie"]
        b = predictions["weighted_logratio"]
        best_weight = min((.2, .4, .6, .8), key=lambda weight: select_metric(
            y, weight * a + (1 - weight) * b, validation))
        fused = best_weight * a + (1 - best_weight) * b
        blend[subset] = fused[subset]
        incumbent = predictions[selected[platform]]
        # A blend must improve selection MAE by at least 1% to justify the extra model.
        if select_metric(y, fused, validation) < .99 * select_metric(y, incumbent, validation):
            selected[platform] = "tweedie_ratio_blend"
            blend_weights[platform] = best_weight
    predictions["tweedie_ratio_blend"] = blend
    output = np.empty(len(y), dtype=float)
    for platform, selected_name in selected.items():
        output[p == platform] = predictions[selected_name][p == platform]
    predictions["selected_by_platform"] = output
    models = {k: model.booster_ if isinstance(model, LGBMRegressor) else model
              for k, model in trained.items()}
    bundle = {
        "version": 2, "platforms": PLATFORMS, "reference": REFERENCE, "grid_km": 1,
        "block_km": block_km,
        "features": MODEL_FEATURES, "factors": factors, "selected": selected,
        "blend_weights": blend_weights,
        "models": models, "individual": individual,
    }
    report = {
        "reference": f"NOAA-20 recorded monthly 1 km grid-cell-days in a {block_km} km block",
        "block_km": block_km,
        "overlap": "April 2018 onward, starting as a platform becomes available",
        "splits": "train 2018-04–2023; select 2024; temporal test 2025; stress test 2026 through October",
        "spatial_holdout": "fixed 10% of global blocks excluded from fitting and selection",
        "zero_target": "no matching qualifying NOAA-20 record in that block-month; coverage unknown",
        "selected_methods": {PLATFORMS[k]: v for k, v in selected.items()},
        "tweedie_blend_weights": {PLATFORMS[k]: v for k, v in blend_weights.items()},
        "comparison": {},
    }
    for label, mask in {"train": is_training, "selection_2024": is_selection, "test_2025": is_testing,
                        "test_2025_spatial": is_testing & partition,
                        "test_2025_joint_records": is_testing & paired["paired_positive"],
                        "stress_2026": is_recent,
                        "stress_2026_nrt": is_recent & (paired["mo"] >= 8)}.items():
        report["comparison"][label] = {
            name: scores(y, predictions, mask & (p == i))
            for i, name in enumerate(PLATFORMS) if i != REFERENCE
        }
        report["comparison"][label]["all_platforms"] = scores(y, predictions, mask)
    print(json.dumps(report["selected_methods"]), flush=True)
    if not np.allclose(predict(bundle, paired), output, rtol=1e-8, atol=1e-6):
        raise ValueError("Select and inference differ")
    return bundle, report, all_rows


def export(db, bundle, rows, output):
    activity = predict(bundle, rows)
    db.register("predictions", {
        "month": rows["month"], "bx": rows["bx"], "grid_by": rows["grid_by"],
        "platform": rows["platform"], "observed_cell_days": rows["cell_days"],
        "activity_noaa20_equivalent_cell_days": activity,
    })
    db.execute("""
        CREATE TABLE platform_calendar AS
        SELECT month, bx, grid_by, platform, observed_cell_days,
               activity_noaa20_equivalent_cell_days,
               CASE WHEN platform=3 THEN 'noaa20_observed' ELSE 'estimated_from_platform' END AS activity_source
        FROM predictions
    """)
    data = sql_string(output / "platform_calendar.parquet")
    db.execute(f"""
        COPY (SELECT * FROM platform_calendar ORDER BY month,bx,grid_by,platform)
        TO {data} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    for scope, options in {"modis": (0, 1), "viirs": (2, 3, 4)}.items():
        db.execute(f"""
            COPY ({combined_calendar_query(options)})
            TO {sql_string(output / f'calendar_{scope}.csv')} (HEADER true)
        """)


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


def apply_to_points(db, model_path, start, end):
    return apply_calendar_model(db, joblib.load(model_path), start, end)


def apply_calendar_model(db, model, start, end):
    from .world_features import monthly_features_query

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-km", type=int, choices=(50, 125, 250), default=250)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={"threads": "8"}) as db:
        model, report, rows = fit(db, args.features, args.block_km)
        joblib.dump(model, args.output / "world_calendar_model.joblib", compress=3)
        export(db, model, rows, args.output)
    (args.output / "calendar_validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
