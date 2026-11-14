"""검증 지표와 평가 지역 정의 (M1 공통).

지역 (전체 칸 기준 합계):  world, 대륙(continent), korea(is_korea), argentina(adm0_a3 == ARG)
칸·월 지표:  MAE, RMSE(log1p), Poisson deviance — 학습 표(표본) 기준
지역·월 합계 지표: MAE(월 합계), bias = Σ예측/Σ실제, Pearson, Spearman (연도 안 월 순위 / 전체 기간)
"""

from __future__ import annotations

import numpy as np
import polars as pl

REGIONS_FIXED = ["world", "korea", "argentina"]


def region_masks(df: pl.DataFrame | pl.LazyFrame, continents: list[str]) -> dict[str, pl.Expr]:
    m = {"world": pl.lit(True), "korea": pl.col("is_korea"), "argentina": pl.col("adm0_a3") == "ARG"}
    for c in continents:
        if c not in ("none",):
            m[f"continent:{c}"] = pl.col("continent") == c
    return m


def poisson_deviance(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / p), 0.0)
    return float(np.mean(2 * (term - (y - p))))


def cell_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "cell_MAE": float(np.mean(np.abs(y - p))),
        "cell_RMSE_log1p": float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.clip(p, 0, None))) ** 2))),
        "cell_poisson_dev": poisson_deviance(y, p),
        "cell_n": float(len(y)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    return pl.Series(a).rank("average").to_numpy()


def total_metrics(true: np.ndarray, pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    st = float(np.sum(true))
    return {
        f"{prefix}MAE": float(np.mean(np.abs(true - pred))),
        f"{prefix}bias": float(np.sum(pred) / st) if st > 0 else float("nan"),
        f"{prefix}pearson": _corr(true, pred),
        f"{prefix}spearman": _corr(_rank(true), _rank(pred)),
        f"{prefix}n_months": float(len(true)),
    }
