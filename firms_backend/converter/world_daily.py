"""Model cross-platform hotspot agreement and paired FRP on a worldwide 1 km grid."""
import argparse
from datetime import date
import json
from pathlib import Path

import duckdb
import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .convert import sql_string
from .world_features import BLOCK_KM, DAILY_SAMPLE_MODULUS, PLATFORMS

# Different instrument families and one reference per direction at inference.
AGREEMENT_ROUTES = ((0, 2), (0, 3), (1, 2), (1, 3), (2, 1), (3, 1), (4, 1))
FRP_ROUTES = ((0, 3), (1, 3), (2, 3), (4, 3))
ROUTES = tuple(dict.fromkeys(AGREEMENT_ROUTES + FRP_ROUTES))
AVAILABLE = (date(2000, 11, 1), date(2002, 7, 3), date(2012, 1, 20),
             date(2018, 4, 1), date(2023, 3, 1))
FEATURES = ("source_platform", "target_platform", "month_sin", "month_cos",
            "longitude_sin", "longitude_cos", "latitude_equal_area", "log_pixels",
            "log_frp", "log_pixel_area", "day_share", "log_minutes")


def feature_matrix(rows):
    angle = (rows["mo"] - 1) * (2 * np.pi / 12)
    longitude = rows["x"] * (np.pi / 17333.5)
    values = (rows["platform"], rows["target_platform"], np.sin(angle), np.cos(angle),
              np.sin(longitude), np.cos(longitude), rows["y"] / 7356.0,
              np.log1p(rows["pixels"]), np.log1p(rows["grid_frp_mw"]),
              np.log1p(rows["pixel_area"]), rows["day_share"], np.log1p(rows["minutes"]))
    return np.column_stack([np.ma.filled(np.ma.asarray(v, dtype=np.float32), np.nan)
                            for v in values])


def route_probabilities(bundle, rows):
    x = feature_matrix(rows)
    routes = rows["platform"] * len(PLATFORMS) + rows["target_platform"]
    probabilities = np.full(len(x), np.nan)
    for route, method in bundle["selected"].items():
        mask = routes == route
        if mask.any():
            probabilities[mask] = predict_probability(bundle, method, route, x[mask])
    return probabilities


def standardized_frp(bundle, rows):
    native_frp = np.ma.filled(rows["grid_frp_mw"], np.nan)
    eligible = np.isfinite(native_frp)
    features = feature_matrix({**rows, "target_platform": np.full(len(native_frp), 3)})
    values = np.full(len(native_frp), np.nan)
    methods = np.full(len(native_frp), "", dtype="U18")
    reference = eligible & (rows["platform"] == 3)
    for source, target in FRP_ROUTES:
        mask = eligible & (rows["platform"] == source)
        if mask.any():
            chosen = bundle["selected_frp"][source * len(PLATFORMS) + target]
            values[mask] = frp_prediction(bundle["frp_models"], chosen, source, features[mask], native_frp[mask])
            methods[mask] = "model_estimated" if chosen not in ("native", "ratio") else chosen
    values[reference] = native_frp[reference]
    methods[reference] = "noaa20_observed"
    return values, methods


def predict_probability(bundle, name, route, x):
    if name == "route_prevalence":
        return np.full(len(x), bundle["prevalence"][route])
    model_name, _, calibration = name.partition("/")
    model = bundle["classifiers"][model_name]
    matrix = np.nan_to_num(x, nan=-1.0) if model_name == "logistic" else x
    raw = model.predict_proba(matrix)[:, 1]
    if not calibration:
        return raw
    calibrator = bundle["calibrators"][model_name, route, calibration]
    if calibration == "isotonic":
        return np.clip(calibrator.predict(raw), 1e-6, 1 - 1e-6)
    return calibrator.predict_proba(logit(raw))[:, 1]


def logit(probabilities):
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def load_examples(db, directory):
    routes = ",".join(
        f"({s},{t},DATE '{max(AVAILABLE[s], AVAILABLE[t])}')" for s, t in ROUTES
    )
    return db.execute(f"""
        WITH samples AS (SELECT * FROM read_parquet({sql_string(directory / '*_daily_*.parquet')})),
        route(platform,target_platform,first_day) AS (VALUES {routes})
        SELECT year(s.day)::SMALLINT AS yr,month(s.day)::UTINYINT AS mo,s.x,s.y,
               s.platform,r.target_platform,s.pixels,s.grid_frp_mw,s.pixel_area,s.day_share,s.minutes,
               (t.day IS NOT NULL)::UTINYINT AS matched,t.grid_frp_mw AS target_frp
        FROM samples s JOIN route r ON s.platform = r.platform AND s.day >= r.first_day
        LEFT JOIN samples t ON t.day = s.day AND t.x = s.x AND t.y = s.y
                           AND t.platform = r.target_platform
        ORDER BY s.day,s.x,s.y,s.platform,r.target_platform
    """).fetchnumpy()


def metrics(y, p):
    return {"n": len(y), "observed_rate": float(y.mean()), "predicted_rate": float(p.mean()),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "average_precision": float(average_precision_score(y, p))}


def train_agreement(rows, x, train, cal, selection, test_masks):
    y = rows["matched"]
    route_id = rows["platform"] * len(PLATFORMS) + rows["target_platform"]
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=180)),
        "lightgbm": LGBMClassifier(n_estimators=220, learning_rate=.08, num_leaves=48,
                                   min_child_samples=125, reg_lambda=3, verbosity=-1,
                                   n_jobs=8, random_state=42),
        "hist_boosting": HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=31,
            l2_regularization=2, early_stopping=False, random_state=42),
    }
    for name, model in models.items():
        x_train = np.nan_to_num(x[train], nan=-1.0) if name == "logistic" else x[train]
        if name == "lightgbm":
            model.fit(x_train, y[train], categorical_feature=[0, 1])
        else:
            model.fit(x_train, y[train])
        print(f"agreement model fitted: {name}", flush=True)
    bundle = {
        "version": 2, "features": FEATURES, "grid_km": 1,
        "classifiers": models, "calibrators": {}, "prevalence": {}, "selected": {},
        "routes": AGREEMENT_ROUTES, "platforms": PLATFORMS, "available": AVAILABLE,
    }
    comparison = {}
    for source, target in AGREEMENT_ROUTES:
        route = source * len(PLATFORMS) + target
        name = f"{PLATFORMS[source]} to {PLATFORMS[target]}"
        is_route = route_id == route
        bundle["prevalence"][route] = float(y[train & is_route].mean())
        to_calibrate = cal & is_route
        to_select = selection & is_route
        methods = ["route_prevalence"]
        for model_name, model in models.items():
            matrix = x[to_calibrate]
            if model_name == "logistic":
                matrix = np.nan_to_num(matrix, nan=-1.0)
            raw = model.predict_proba(matrix)[:, 1]
            for calibrator_name, calibrator, feature in (
                ("sigmoid", LogisticRegression(C=100), logit(raw)),
                ("isotonic", IsotonicRegression(out_of_bounds="clip"), raw),
            ):
                calibrator.fit(feature, y[to_calibrate])
                bundle["calibrators"][model_name, route, calibrator_name] = calibrator
                methods.append(f"{model_name}/{calibrator_name}")
            methods.append(model_name)
        results = {}
        for label, base in {"selection_2024_h2": selection, **test_masks}.items():
            mask = base & is_route
            if not mask.any():
                continue
            results[label] = {
                method: metrics(y[mask], predict_probability(bundle, method, route, x[mask]))
                for method in methods
            }
        winner = min(methods, key=lambda m: results["selection_2024_h2"][m]["brier"])
        bundle["selected"][route] = winner
        comparison[name] = {"method": winner, "training_n": int((train & is_route).sum()),
                            "calibration_n": int(to_calibrate.sum()), "metrics": results}
        print(name, winner, results["test_2025"][winner], flush=True)
    return bundle, comparison


def frp_prediction(models, method, source, x, native):
    if method == "native":
        return native
    if method == "ratio":
        return native * models["ratios"][source]
    estimate = models[method].booster_.predict(x)
    return np.maximum(0, np.expm1(estimate) if method == "log_l1" else estimate)


def train_frp(rows, x, train, selection, test_masks):
    source_frp = np.ma.filled(rows["grid_frp_mw"], np.nan)
    target_frp = np.ma.filled(rows["target_frp"], np.nan)
    eligible = np.isfinite(source_frp) & np.isfinite(target_frp) & (rows["target_platform"] == 3)
    active_train = train & eligible
    models = {"ratios": {}}
    for method, objective in (("l1", "regression_l1"), ("log_l1", "regression_l1"),
                              ("tweedie", "tweedie")):
        model = LGBMRegressor(objective=objective, tweedie_variance_power=1.5,
                             n_estimators=220, learning_rate=.065, num_leaves=48,
                             min_child_samples=100, reg_lambda=3, verbosity=-1,
                             n_jobs=8, random_state=42)
        labels = target_frp[active_train]
        if method == "log_l1":
            labels = np.log1p(labels)
        model.fit(x[active_train], labels, categorical_feature=[0, 1])
        models[method] = model
    summary = {}
    selected = {}
    for source, target in FRP_ROUTES:
        route = source * len(PLATFORMS) + target
        name = f"{PLATFORMS[source]} to {PLATFORMS[target]}"
        mask = (rows["platform"] == source) & eligible
        training = train & mask
        models["ratios"][source] = float(target_frp[training].sum() / source_frp[training].sum())
        report = {}
        for period, scope in {"selection_2024_h2": selection, **test_masks}.items():
            items = scope & mask
            if not items.any():
                continue
            truth = target_frp[items]
            report[period] = {"n": int(items.sum()), "models": {}}
            for method in ("native", "ratio", "l1", "log_l1", "tweedie"):
                estimates = frp_prediction(models, method, source, x[items], source_frp[items])
                report[period]["models"][method] = {
                    "mae_mw": float(np.abs(estimates - truth).mean()),
                    "mae_log_mw": float(np.abs(np.log1p(estimates) - np.log1p(truth)).mean()),
                }
        validation = report["selection_2024_h2"]["models"]
        winner = min(validation, key=lambda method: validation[method]["mae_mw"])
        selected[route] = winner
        summary[name] = {"method": winner, "training_n": int(training.sum()), "metrics": report}
        print("paired FRP", name, winner, report.get("test_2025"), flush=True)
    return models, selected, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={"threads": "8"}) as db:
        rows = load_examples(db, args.features)
    x = feature_matrix(rows)
    year, month = rows["yr"], rows["mo"]
    spatial = ((rows["x"] // BLOCK_KM) * 271 + (rows["y"] // BLOCK_KM) * 509) % 10 == 0
    train = (year <= 2023) & ~spatial
    cal = (year == 2024) & (month <= 6) & ~spatial
    selection = (year == 2024) & (month > 6) & ~spatial
    test_masks = {"test_2025": year == 2025, "test_2025_spatial": (year == 2025) & spatial,
                  "stress_2026": year == 2026}
    print("daily directional examples", len(year), "training", int(train.sum()), flush=True)
    bundle, comparison = train_agreement(rows, x, train, cal, selection, test_masks)
    model, selected_frp, frp_report = train_frp(rows, x, train, selection, test_masks)
    bundle.update({"frp_models": model, "selected_frp": selected_frp})
    joblib.dump(bundle, args.output / "world_daily_model.joblib", compress=3)
    report = {
        "sample": f"deterministic 1-in-{DAILY_SAMPLE_MODULUS} of global 1 km cell/day keys",
        "splits": "overlap through 2023 train; Jan–Jun 2024 calibrate; Jul–Dec 2024 select; 2025 test; 2026 stress",
        "spatial_holdout": "same 250 km blocks reserved from all fitting and selection",
        "probability_target": "record from the named target platform in the same UTC day and grid cell, conditional on a source record",
        "frp_target": "median per-minute pixel FRP sums (MW), conditional on valid FRP in both platforms",
        "example_count": len(year),
        "agreement": comparison,
        "paired_frp": frp_report,
    }
    (args.output / "daily_validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
