"""Regression models for normalized daily satellite detection amounts."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

BASELINES = ("zero", "mean30", "seasonal")
ZERO_KEEP = {
    "lightgbm_l1_zero50": 0.5,
    "lightgbm_l1_zero20": 0.2,
    "lightgbm_l1_zero20_ipw": 0.2,
    "xgboost_l1_zero20": 0.2,
    "hurdle_zero20": 0.2,
}
MODELS = (
    "ridge",
    "poisson",
    "extra_trees",
    "hist_boost",
    "lightgbm_tweedie",
    "xgboost",
    "mlp",
    "hurdle",
    "random_forest",
    "elastic_net",
    "lightgbm_l1",
    "xgboost_l1",
    *ZERO_KEEP,
)
FOUNDATION = ("timesfm3", "chronos2", "chronos_bolt", "tirex2")


@dataclass
class Fitted:
    estimator: object
    probability: object | None = None
    zero_keep: float = 1.0

    def predict(self, x: np.ndarray, scale: np.ndarray) -> np.ndarray:
        x_flat = np.ascontiguousarray(x.reshape(-1, x.shape[-1]))
        prediction = self.estimator.predict(x_flat)
        if self.probability is not None:
            probability = self.probability.predict_proba(x_flat)[:, 1]
            prediction *= restore_probability(probability, self.zero_keep)
        return np.maximum(prediction.reshape(x.shape[:2]) * scale[:, None], 0)


def restore_probability(probability: np.ndarray, zero_keep: float) -> np.ndarray:
    """Undo uniform subsampling of the zero class in the training posterior."""
    return probability * zero_keep / (1 - probability + probability * zero_keep)


def fit_model(model: str, x, y, weight, threads: int, device: str = "cpu") -> Fitted:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, PoissonRegressor, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    trees = {
        "n_estimators": 64,
        "max_depth": 14,
        "min_samples_leaf": 30,
        "n_jobs": threads,
        "random_state": 1126,
    }
    lgb = {
        "n_estimators": 230,
        "learning_rate": 0.055,
        "num_leaves": 63,
        "min_child_samples": 100,
        "n_jobs": threads,
        "verbosity": -1,
        "random_state": 1126,
    }
    xgb = {
        "n_estimators": 260,
        "learning_rate": 0.055,
        "max_depth": 7,
        "min_child_weight": 40,
        "tree_method": "hist",
        "device": device,
        "n_jobs": threads,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "random_state": 1126,
    }
    builders = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=20)),
        "elastic_net": lambda: make_pipeline(
            StandardScaler(), ElasticNet(alpha=0.002, l1_ratio=0.55)
        ),
        "poisson": lambda: make_pipeline(
            StandardScaler(), PoissonRegressor(alpha=0.02, max_iter=200)
        ),
        "extra_trees": lambda: ExtraTreesRegressor(**trees, max_features=0.75),
        "random_forest": lambda: RandomForestRegressor(
            **trees, max_features=0.6, max_samples=0.45
        ),
        "hist_boost": lambda: HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=170,
            max_leaf_nodes=47,
            l2_regularization=2,
            min_samples_leaf=90,
            early_stopping=False,
            random_state=1126,
        ),
        "lightgbm_tweedie": lambda: LGBMRegressor(
            **lgb, objective="tweedie", tweedie_variance_power=1.5
        ),
        "lightgbm_l1": lambda: LGBMRegressor(**lgb, objective="regression_l1"),
        "xgboost": lambda: XGBRegressor(
            **xgb, objective="reg:tweedie", tweedie_variance_power=1.5
        ),
        "xgboost_l1": lambda: XGBRegressor(**xgb, objective="reg:absoluteerror"),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                alpha=0.01,
                loss="poisson",
                batch_size=1024,
                learning_rate_init=0.001,
                max_iter=42,
                random_state=1126,
            ),
        ),
        "hurdle": lambda: LGBMRegressor(**lgb, objective="gamma"),
    }
    print(f"fitting {model}: {len(y):,} rows", flush=True)
    start = perf_counter()
    family = model.split("_zero", 1)[0]
    estimator = builders[family]()
    probability = None
    if family == "hurdle":
        positive = y > 0
        probability = LGBMClassifier(**lgb).fit(x, positive, sample_weight=weight)
        estimator.fit(x[positive], y[positive], sample_weight=weight[positive])
    else:
        # Pipelines route the weights to their final regression estimator.
        options = (
            {"sample_weight": weight}
            if not hasattr(estimator, "steps")
            else {f"{estimator.steps[-1][0]}__sample_weight": weight}
        )
        estimator.fit(x, y, **options)
    print(f"{model}: trained in {perf_counter() - start:.1f}s", flush=True)
    return Fitted(estimator, probability, ZERO_KEEP.get(model, 1.0))
