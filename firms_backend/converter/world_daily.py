"""Calibrated cross-platform detection agreement and FRP conversion."""
import numpy as np

from .world_features import PLATFORMS

FRP_ROUTES = ((0, 3), (1, 3), (2, 3), (4, 3))


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
    values = np.full(len(native_frp), np.nan)
    methods = np.full(len(native_frp), "", dtype="U18")
    reference = eligible & (rows["platform"] == 3)
    for source, target in FRP_ROUTES:
        mask = eligible & (rows["platform"] == source)
        if mask.any():
            chosen = bundle["selected_frp"][source * len(PLATFORMS) + target]
            values[mask] = frp_prediction(bundle["frp_models"], chosen, source, native_frp[mask])
            methods[mask] = chosen
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


def frp_prediction(models, method, source, native):
    if method == "native":
        return native
    if method == "ratio":
        return native * models["ratios"][source]
    raise ValueError(f"Unsupported FRP conversion method: {method}")
