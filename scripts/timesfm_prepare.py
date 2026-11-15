"""TimesFM 비교 실험 입력 만들기 (프로젝트 환경에서 실행). 출력: <out>/series_{grid,sgg}.npz
행렬 V[t, i] = log1p(월 value), t = 2012-02 ~ 2025-12 (진행 중인 달 없음). 값이 없는 달은 0 (감지 없음).
  grid: data/interim/grid_occurrence/agg.parquet (0.5° 격자 월 value 합, 고정 열원 제외 v1.2), 대상 격자 52,855
  sgg : data/processed/aoi_month.parquet (한국 시군구 252)
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
T0, T1 = 2012 * 12 + 1, 2025 * 12 + 11          # 2012-02 .. 2025-12 (t = year*12 + month-1)
T = np.arange(T0, T1 + 1)

cells = pl.read_parquet(INTERIM_DATA_DIR / "grid_occurrence" / "cells.parquet").sort("g")
a = pl.read_parquet(INTERIM_DATA_DIR / "grid_occurrence" / "agg.parquet").filter(pl.col("t").is_between(T0, T1)).join(
    cells.select("g"), on="g", how="semi")
gi = {g: i for i, g in enumerate(cells["g"].to_list())}
V = np.zeros((len(T), cells.height), np.float32)
V[a["t"].to_numpy() - T0, np.array([gi[g] for g in a["g"].to_list()])] = np.log1p(np.maximum(a["v"].to_numpy(), 0))
np.savez_compressed(OUT / "series_grid.npz", V=V, t=T, ids=cells["g"].to_numpy())
print("grid", V.shape, "nonzero share", float((V > 0).mean()))

aoi = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_list.parquet").filter(pl.col("level") == "sgg", ~pl.col("hidden")).sort("aoi_id")
am = (pl.read_parquet(PROCESSED_DATA_DIR / "aoi_month.parquet", columns=["aoi_id", "year", "month", "value"])
      .join(aoi.select("aoi_id"), on="aoi_id", how="semi")
      .with_columns(t=pl.col("year").cast(pl.Int32) * 12 + pl.col("month").cast(pl.Int32) - 1).filter(pl.col("t").is_between(T0, T1)))
ai = {x: i for i, x in enumerate(aoi["aoi_id"].to_list())}
S = np.zeros((len(T), aoi.height), np.float32)
S[am["t"].to_numpy() - T0, np.array([ai[x] for x in am["aoi_id"].to_list()])] = np.log1p(np.maximum(am["value"].to_numpy(), 0))
np.savez_compressed(OUT / "series_sgg.npz", V=S, t=T, ids=np.array(aoi["aoi_id"].to_list()))
print("sgg", S.shape, "nonzero share", float((S > 0).mean()))
