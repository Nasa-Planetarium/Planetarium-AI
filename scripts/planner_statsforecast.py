"""FireOutlook Planner — 지역 월 건수 후보 AutoETS·AutoTheta (statsforecast, Apache-2.0).

프로젝트 환경이 아니라 별도 가상환경에서 돈다 (statsforecast를 프로젝트 의존성에 넣지 않음):
    uv venv sf-venv --python 3.12
    uv pip install --python sf-venv/Scripts/python.exe statsforecast polars pyarrow pandas
    sf-venv/Scripts/python.exe scripts/planner_statsforecast.py --series data/interim/planner/region_series.parquet \
        --out data/interim/planner/region_sf_backtest.parquet [--service data/interim/planner/region_sf_service.parquet]

입력  region_series.parquet: aoi_id, t(= 연 × 12 + 월 − 1), y(S-NPP 기준 월 건수, 완결된 달만)
백테스트  기준 달 c (데이터 ≤ c만 사용)마다 12달 앞까지 예측 → 대상 t = c + h. c = 2018-12 … 마지막 − 1.
서비스  마지막 완결 달까지로 학습해 13달 앞까지 (대상 = 2026-11 … 2027-11; 2026-11은 진행 중인 달이라 h 계산에서 제외).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import polars as pl


def run(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, AutoTheta
    sf = StatsForecast(models=[AutoETS(season_length=12), AutoTheta(season_length=12)], freq="MS", n_jobs=1)
    return sf.forecast(df=df, h=horizon)


def to_pd(s: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"unique_id": s["aoi_id"].to_list(),
                         "ds": pd.to_datetime([f"{t // 12}-{t % 12 + 1:02d}-01" for t in s["t"].to_list()]),
                         "y": np.log1p(np.clip(s["y"].to_numpy(), 0, None))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--service")
    ap.add_argument("--first-cutoff", type=int, default=2018 * 12 + 11)
    a = ap.parse_args()
    s = pl.read_parquet(a.series)
    last = int(s["t"].max())
    parts = []
    t0 = time.perf_counter()
    cutoffs = list(range(a.first_cutoff, last))
    for i, c in enumerate(cutoffs):
        f = run(to_pd(s.filter(pl.col("t") <= c)), 12)
        f["cutoff"] = c
        parts.append(f)
        if i % 10 == 0:
            print(f"cutoff {c // 12}-{c % 12 + 1:02d} ({i + 1}/{len(cutoffs)}), {time.perf_counter() - t0:,.0f}s", flush=True)
    bt = pd.concat(parts, ignore_index=True)
    out = pl.from_pandas(bt.reset_index() if "unique_id" not in bt.columns else bt)
    out = out.with_columns(t=(pl.col("ds").dt.year().cast(pl.Int32) * 12 + pl.col("ds").dt.month().cast(pl.Int32) - 1)).with_columns(
        h=pl.col("t") - pl.col("cutoff"),
        AutoETS=pl.col("AutoETS").exp() - 1, AutoTheta=pl.col("AutoTheta").exp() - 1).rename({"unique_id": "aoi_id"})
    out.select("aoi_id", "cutoff", "t", "h", "AutoETS", "AutoTheta").write_parquet(a.out)
    print(f"[완료] backtest {out.height:,}행, {time.perf_counter() - t0:,.0f}s")
    if a.service:
        f = run(to_pd(s), 13)
        sv = pl.from_pandas(f.reset_index() if "unique_id" not in f.columns else f).rename({"unique_id": "aoi_id"}).with_columns(
            t=(pl.col("ds").dt.year().cast(pl.Int32) * 12 + pl.col("ds").dt.month().cast(pl.Int32) - 1),
            AutoETS=pl.col("AutoETS").exp() - 1, AutoTheta=pl.col("AutoTheta").exp() - 1).with_columns(cutoff=pl.lit(last))
        sv.select("aoi_id", "cutoff", "t", "AutoETS", "AutoTheta").write_parquet(a.service)
        print(f"[완료] service {sv.height:,}행")


if __name__ == "__main__":
    main()
