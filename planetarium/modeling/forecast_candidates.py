"""Baselines and dataset-trained models for monthly hotspot forecasts."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib.metadata import version

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SEASON = 12
LAGS = (1, 2, 3, 4, 6, 9, 12, 13, 18, 24, 25, 36)
ROLLING = (3, 6, 12, 24)


def _baseline(model, values, origins, horizon):
    seasonal = origins[:, None] + np.arange(1, horizon + 1) - SEASON
    if model == "last_value":
        return np.repeat(values[:, origins].T[:, :, None], horizon, axis=2)
    if model == "seasonal_naive":
        return values[:, seasonal].transpose(1, 0, 2).copy()

    indices = seasonal[..., None] - SEASON * np.arange(5)
    reducer = np.median if model == "seasonal_median" else np.mean
    return reducer(values[:, indices], axis=-1).transpose(1, 0, 2).copy()


def _seasonal_smoothing(values, origins, horizon):
    alpha = 0.25
    state = values.copy()
    for t in range(SEASON, state.shape[1]):
        state[:, t] = alpha * values[:, t] + (1 - alpha) * state[:, t - SEASON]
    forecast = _baseline("seasonal_naive", state, origins, horizon)
    return forecast, {"season": SEASON, "alpha": alpha}


def _seasonal_ar(panel, horizon):
    lags = (1, 2, 3, 12, 13)
    alpha = 4.0
    logged = np.log1p(panel.values).astype(np.float64)
    n_series, n_months = logged.shape
    angles = (
        2
        * np.pi
        * ((panel.dates[0].astype(int) + np.arange(n_months + horizon)) % SEASON)
        / SEASON
    )
    calendar = np.column_stack([np.ones_like(angles), np.sin(angles), np.cos(angles)])
    n_features = calendar.shape[1] + len(lags)
    penalty = np.eye(n_features) * alpha
    penalty[0, 0] = 0
    gram = np.broadcast_to(penalty, (n_series, n_features, n_features)).copy()
    response = np.zeros((n_series, n_features), dtype=np.float64)
    point = np.empty((len(panel.origins), n_series, horizon), dtype=np.float32)
    at_origin = {origin: row for row, origin in enumerate(panel.origins)}

    for month in range(max(lags), int(panel.origins[-1]) + 1):
        known_calendar = np.broadcast_to(calendar[month], (n_series, calendar.shape[1]))
        features = np.column_stack(
            [known_calendar, *(logged[:, month - lag] for lag in lags)]
        )
        gram += features[:, :, None] * features[:, None, :]
        response += features * logged[:, month, None]
        if month not in at_origin:
            continue

        weights = np.linalg.solve(gram, response[..., None])[..., 0]
        predictions = []
        for lead in range(1, horizon + 1):
            target = month + lead
            known_calendar = np.broadcast_to(
                calendar[target], (n_series, calendar.shape[1])
            )
            history = [
                logged[:, target - lag] if lead <= lag else predictions[lead - lag - 1]
                for lag in lags
            ]
            x = np.column_stack([known_calendar, *history])
            predictions.append(np.maximum(0, np.einsum("ij,ij->i", x, weights)))
        point[at_origin[month]] = np.expm1(np.column_stack(predictions))
    return point, {"lags": lags, "alpha": alpha, "season": SEASON, "target": "log1p"}


def _tsb(values, origins, horizon):
    alpha = 0.1
    # Teunter–Syntetos–Babai separately smooths the chance of an event and its
    # size, and updates the occurrence probability even in zero months.
    probability = np.zeros(values.shape[0], dtype=np.float32)
    size = np.zeros_like(probability)
    expected = np.empty_like(values)
    for t, amounts in enumerate(values.T):
        has_event = amounts > 0
        first_event = has_event & (size == 0)
        size[has_event] += alpha * (amounts[has_event] - size[has_event])
        size[first_event] = amounts[first_event]
        probability += alpha * (has_event - probability)
        expected[:, t] = size * probability
    forecast = _baseline("last_value", expected, origins, horizon)
    return forecast, {"alpha_occurrence": alpha, "alpha_size": alpha}


def _croston_sba(values, origins, horizon):
    alpha = 0.1
    size = np.zeros(values.shape[0], dtype=np.float32)
    interval = np.zeros_like(size)
    previous = np.full(len(size), -1, dtype=int)
    expected = np.zeros_like(values)
    for month, amounts in enumerate(values.T):
        event = amounts > 0
        first = event & (previous == -1)
        gap = month - previous[event]
        size[event] += alpha * (amounts[event] - size[event])
        interval[event] += alpha * (gap - interval[event])
        size[first] = amounts[first]
        interval[first] = month + 1
        previous[event] = month
        np.divide(
            (1 - alpha / 2) * size, interval, out=expected[:, month], where=interval > 0
        )
    return _baseline("last_value", expected, origins, horizon), {"alpha": alpha}


def _sarima(panel, horizon):
    """Conditional least squares for SARIMA(1,0,0)(0,1,0)[12]."""
    log = np.log1p(panel.values.astype(np.float64))
    diff = log[:, SEASON:] - log[:, :-SEASON]
    x, y = diff[:, :-1], diff[:, 1:]
    end = panel.origins - SEASON - 1
    numerator = np.cumsum(x * y, axis=1)[:, end]
    denominator = np.cumsum(x**2, axis=1)[:, end]
    ar = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    ar = ar.clip(-0.99, 0.99)
    leads = np.arange(1, horizon + 1)
    seasonal = log[:, panel.origins[:, None] + leads - SEASON]
    projected = seasonal + diff[:, end + 1, None] * ar[:, :, None] ** leads
    point = np.expm1(projected).transpose(1, 0, 2).astype(np.float32)
    return point, {
        "order": [1, 0, 0],
        "seasonal_order": [0, 1, 0, SEASON],
        "target": "log1p",
        "estimation": "conditional least squares",
    }


def _theta_block(block, origins, horizon):
    from statsforecast.models import AutoTheta

    estimator = AutoTheta(season_length=SEASON, decomposition_type="additive")
    fitted = np.empty((len(origins), len(block), horizon), dtype=np.float32)
    for col, values in enumerate(block):
        different = np.flatnonzero(values[1:] != values[0])
        first_change = int(different[0] + 1) if different.size else len(values)
        for row, origin in enumerate(origins):
            if origin < first_change:
                fitted[row, col] = values[0]
            else:
                fitted[row, col] = estimator.forecast(values[: origin + 1], h=horizon)[
                    "mean"
                ]
    return fitted


def _parallel_rows(fit, blocks, workers, model):
    if workers == 1:
        return [fit(block) for block in blocks]
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for i, result in enumerate(pool.map(fit, blocks), start=1):
            results.append(result)
            if i == 1 or i % 8 == 0 or i == len(blocks):
                print(f"{model} groups {i}/{len(blocks)}", flush=True)
    return results


def _theta(panel, horizon, workers):
    logged = np.log1p(panel.values)
    blocks = np.array_split(logged, min(len(logged), workers * 3))
    fit = partial(_theta_block, origins=panel.origins, horizon=horizon)
    forecasts = _parallel_rows(fit, blocks, workers, "theta")
    return np.expm1(np.concatenate(forecasts, axis=1)), {
        "package": f"statsforecast=={version('statsforecast')}",
        "season": SEASON,
        "target": "log1p",
        "decomposition": "additive",
    }


def _autoets_block(block, origins, folds, horizon):
    from statsforecast.models import AutoETS

    fitted = np.empty((len(origins), len(block), horizon), dtype=np.float32)
    fallbacks = 0
    for col, values in enumerate(block):
        for cutoff, rows in folds:
            estimator = AutoETS(season_length=SEASON, model="AZZ")
            try:
                estimator.fit(values[: cutoff + 1])
            except Exception as exc:
                if str(exc) != "no model able to be fitted":
                    raise
                estimator = None
            for row in rows:
                origin = origins[row]
                if estimator is not None:
                    point = estimator.forward(values[: origin + 1], h=horizon)["mean"]
                    if np.isfinite(point).all():
                        fitted[row, col] = point
                        continue
                fitted[row, col] = values[origin + np.arange(1, horizon + 1) - SEASON]
                fallbacks += 1
    return fitted, fallbacks


def _autoets(panel, horizon, workers):
    logged = np.log1p(panel.values)
    blocks = np.array_split(logged, min(len(logged), workers * 3))
    fit = partial(
        _autoets_block,
        origins=panel.origins,
        folds=list(training_folds(panel)),
        horizon=horizon,
    )
    results = _parallel_rows(fit, blocks, workers, "autoets")
    forecasts, fallbacks = zip(*results, strict=True)
    return np.expm1(np.concatenate(forecasts, axis=1)), {
        "package": f"statsforecast=={version('statsforecast')}",
        "season": SEASON,
        "model": "AZZ",
        "target": "log1p",
        "fit": "annual, monthly state update",
        "seasonal_naive_fallback_origins": sum(fallbacks),
    }


def training_folds(panel):
    """Fit once per year and share that fit across eligible forecast origins."""
    # NumPy month codes are zero-based within a calendar year.
    months = panel.dates[panel.origins].astype(np.int64)
    cutoffs = panel.origins - (months + 1) % SEASON
    for cutoff in np.unique(cutoffs):
        yield int(cutoff), np.flatnonzero(cutoffs == cutoff)


class PanelFeatures:
    def __init__(self, panel):
        self.log = np.log1p(panel.values)
        self.dates = panel.dates
        self.size = np.log1p(panel.sizes).astype(np.float32)
        labels = np.unique(panel.levels)
        self.levels = (panel.levels[:, None] == labels).astype(np.float32)
        self.rolling = []
        for window in ROLLING:
            values = sliding_window_view(self.log, window, axis=1)
            padding = ((0, 0), (window - 1, 0))
            self.rolling.append(np.pad(values.mean(axis=-1), padding))
            self.rolling.append(np.pad(values.max(axis=-1), padding))
            self.rolling.append(
                np.pad((values > 0).mean(axis=-1, dtype=np.float32), padding)
            )

    def examples(self, origins, horizon, include_target):
        n_series = self.log.shape[0]
        count = len(origins) * n_series * horizon
        features = np.empty(
            (count, len(LAGS) + len(self.rolling) + 10 + self.levels.shape[1]),
            dtype=np.float32,
        )
        labels = np.empty(count, dtype=np.float32) if include_target else None
        for lead in range(1, horizon + 1):
            # The last available observation is at origins. Calendar and
            # matching months from earlier years are known at that time.
            targets = origins + lead
            month = self.dates[0].astype(np.int64) + targets
            angle = 2 * np.pi * (month % SEASON) / SEASON
            seasonal = np.stack([self.log[:, targets - SEASON * y] for y in (1, 2, 3)])
            features_by_lead = features[lead - 1 :: horizon].reshape(
                len(origins), n_series, -1
            )
            past = [self.log[:, origins + 1 - lag] for lag in LAGS]
            summary = [series[:, origins] for series in self.rolling]
            for i, column in enumerate(
                [*past, *summary, *seasonal, np.mean(seasonal, axis=0)]
            ):
                features_by_lead[:, :, i] = column.T
            fixed = (np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle))
            offset = len(past) + len(summary) + 4
            for i, column in enumerate(fixed):
                features_by_lead[:, :, offset + i] = column[:, None]
            features_by_lead[:, :, offset + 4] = lead
            features_by_lead[:, :, offset + 5] = self.size[None, :]
            features_by_lead[:, :, offset + 6 :] = self.levels[None, :, :]
            if labels is not None:
                labels[lead - 1 :: horizon] = self.log[:, targets].T.ravel()
        return features, labels


def _estimator(model, threads):
    if model == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=10, solver="cholesky"), "scikit-learn"
    if model == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression_l1",
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            max_bin=63,
            n_jobs=threads,
            force_col_wise=True,
            verbosity=-1,
            random_state=42,
        ), "lightgbm"
    if model == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            loss_function="MAE",
            iterations=120,
            depth=6,
            learning_rate=0.08,
            border_count=63,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
            random_seed=42,
        ), "catboost"
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=120,
        learning_rate=0.08,
        tree_method="hist",
        max_depth=6,
        max_bin=63,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=threads,
        random_state=42,
    ), "xgboost"


def _tree_or_ridge(model, panel, horizon, threads):
    features = PanelFeatures(panel)
    forecast = np.empty((len(panel.origins), len(panel.ids), horizon), dtype=np.float32)
    for cutoff, rows in training_folds(panel):
        train_origins = np.arange(max(LAGS) - 1, cutoff - horizon + 1)
        assert train_origins[-1] + horizon == cutoff <= panel.origins[rows].min()
        train, target = features.examples(train_origins, horizon, include_target=True)
        estimator, package = _estimator(model, threads)
        print(
            f"{model} fit through {panel.dates[cutoff]} ({len(train):,} rows)",
            flush=True,
        )
        if model == "ridge":
            estimator.fit(train.astype(np.float64), target.astype(np.float64))
        else:
            estimator.fit(train, target)
        del train, target
        for row in rows:
            x, _ = features.examples(
                panel.origins[row : row + 1], horizon, include_target=False
            )
            result = (
                estimator.booster_.predict(x, num_threads=threads)
                if model == "lightgbm"
                else estimator.predict(x)
            )
            forecast[row] = np.expm1(result.reshape(len(panel.ids), horizon))
    parameters = estimator.get_params()
    if model == "xgboost":
        parameters["missing"] = "NaN"
    return forecast, {
        "package": f"{package}=={version(package)}",
        "fit": "annual",
        "target": "log1p",
        "n_features": x.shape[1],
        "estimator": parameters,
        "training_dtype": "float64" if model == "ridge" else "float32",
    }


def predict(model, panel, horizon, device, threads, batch_size):
    if model == "seasonal_ar":
        point, config = _seasonal_ar(panel, horizon)
    elif model.startswith("seasonal_") or model == "last_value":
        if model == "seasonal_ets":
            point, config = _seasonal_smoothing(panel.values, panel.origins, horizon)
        else:
            point = _baseline(model, panel.values, panel.origins, horizon)
            config = (
                {"season": SEASON, "same_month_years": 5}
                if model in ("seasonal_median", "seasonal_mean")
                else {}
            )
    elif model == "tsb":
        point, config = _tsb(panel.values, panel.origins, horizon)
    elif model == "croston_sba":
        point, config = _croston_sba(panel.values, panel.origins, horizon)
    elif model == "sarima":
        point, config = _sarima(panel, horizon)
    elif model == "theta":
        point, config = _theta(panel, horizon, threads)
    elif model == "autoets":
        point, config = _autoets(panel, horizon, threads)
    elif model in ("dlinear", "nhits", "tide", "patchtst"):
        from planetarium.modeling.forecast_neural import train

        point, config = train(model, panel, horizon, device, threads, batch_size)
    else:
        point, config = _tree_or_ridge(model, panel, horizon, threads)
    return point, None, None, config
