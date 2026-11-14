"""M2V Uncertainty Band — AOI × 월 합계 구간 보정.

칸별 q10/q90을 그냥 더하면 합계의 구간이 지나치게 넓어진다(칸 오차가 서로 상쇄되는 것을 무시). 대신
leave-year-out OOF 예측을 전체 칸(패널 + 감지 없는 칸)에 만들어 AOI × 월로 합치고,
'실제 / 예측' 비율의 10·90 분위수를 AOI 유형별·예측 크기 구간별로 구해 AOI 합계 value에 곱한다.
  value_lo = value × q10(비율),  value_hi = value × q90(비율)

절차
  1) fold 모델: 연도 Y를 통째로 빼고(조기 종료는 안쪽 연도) m2v_xgb와 같은 설정·목적함수로 학습 (튜닝 없음).
     학습한 fold 모델은 models/folds/m2v_xgb_fold_<Y>.json에 저장해 다시 쓰지 않는다.
  2) 그 해의 모든 칸 × 정상 월에 예측 (zero rule 끔, T6과 같은 방식) → AOI 면적 가중 합 → AOI × 월 실제·예측.
  3) AOI 유형(world, continent, country(KR 포함), adm1, sido, sgg) × 예측 크기 구간별로 비율 분위수.
  4) 적중률: 분위수를 그 해를 뺀 나머지 해로 구해 그 해에 적용(leave-year-out) → 유형별 실제가 [lo, hi] 안에 든 비율.

출력  data/processed/oof_aoi_month_m2v.parquet   (aoi_id, level, year, month, true, pred)
      models/m2v_aoi_interval.parquet             (level, bin, n, q10, q50, q90, pred_min, pred_max)
      reports/m2v_aoi_interval_coverage.csv

실행  uv run python -m planetarium.modeling.m2v_interval      (터미널에서 진행바)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.modeling.m1_xgb import (
    FOLD_YEARS, M1_FEATURES, OBJECTIVES, category_maps, dense_zero_rows, inner_year, xgb_predict, xgb_train,
)
from planetarium.runlog import start_run_log
from planetarium.train_table import GRID_PATH, TRAIN_PATH, load_panel

FOLD_DIR = MODELS_DIR / "folds"
OOF_AOI_PATH = PROCESSED_DATA_DIR / "oof_aoi_month_m2v.parquet"
AOI_INTERVAL_PATH = MODELS_DIR / "m2v_aoi_interval.parquet"
OBJECTIVE = "tweedie_1.1"
BINS = [1.0, 10.0, 100.0, 1000.0, 10000.0]   # 예측 크기 구간 경계 (감지 수)
LEVEL_GROUP = {"kr": "country"}             # KR은 국가 유형으로 함께 보정
MIN_N = 200                                 # 구간 표본이 이보다 적으면 그 유형 전체 값을 씀


def level_expr() -> pl.Expr:
    return pl.col("level").replace(LEVEL_GROUP)


def bin_expr(col: str = "pred") -> pl.Expr:
    return pl.col(col).cut(BINS, left_closed=True).cast(pl.String)


def fold_model(y: int, tt: pl.DataFrame, cats: dict):
    import xgboost as xgb
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = FOLD_DIR / f"m2v_xgb_fold_{y}.json"
    if path.exists():
        b = xgb.Booster()
        b.load_model(path)
        return b
    v = inner_year(y)
    b = xgb_train(tt.filter(~pl.col("year").is_in([y, v])), tt.filter(pl.col("year") == v), M1_FEATURES, cats,
                  OBJECTIVES[OBJECTIVE])
    b.save_model(path)
    return b


def oof_aoi(membership: pl.DataFrame, levels: pl.DataFrame) -> pl.DataFrame:
    tt = pl.read_parquet(TRAIN_PATH).with_columns(w=pl.col("sample_weight") / pl.col("sample_weight").mean())
    cats = category_maps()
    grid = pl.read_parquet(GRID_PATH).with_columns(
        pl.col("adm0_a3").fill_null("none"), pl.col("subregion").fill_null("none"), pl.col("continent").fill_null("none"))
    out = []
    bar = tqdm(FOLD_YEARS, desc="fold 모델 + AOI OOF", unit="fold", dynamic_ncols=True)
    for y in bar:
        t0 = time.perf_counter()
        bar.set_postfix_str(f"{y}년 학습")
        b = fold_model(y, tt, cats)
        bar.set_postfix_str(f"{y}년 예측")
        panel = load_panel("train", [y]).collect().with_columns(pl.col("adm0_a3").fill_null("none"))
        dz = dense_zero_rows(y, grid, panel.select("cell_id", "year", "month"))
        rows = pl.concat([
            panel.select("cell_id", "year", "month", "y").with_columns(pred=pl.Series(xgb_predict(b, panel, M1_FEATURES, cats), dtype=pl.Float64)),
            dz.select("cell_id", "year", "month", "y").with_columns(pred=pl.Series(xgb_predict(b, dz, M1_FEATURES, cats), dtype=pl.Float64)),
        ])
        agg = (rows.lazy().join(membership.lazy(), on="cell_id")
               .group_by("aoi_id", "year", "month")
               .agg(true=(pl.col("y") * pl.col("weight")).sum(), pred=(pl.col("pred") * pl.col("weight")).sum())
               .collect())
        out.append(agg)
        tqdm.write(f"[fold {y}] 반복 {b.num_boosted_rounds()}, 칸·월 {rows.height:,}, AOI·월 {agg.height:,}, {time.perf_counter() - t0:.0f}초")
    df = pl.concat(out).join(levels, on="aoi_id", how="left")
    df.write_parquet(OOF_AOI_PATH)
    return df


def interval_table(d: pl.DataFrame) -> pl.DataFrame:
    """유형 × 예측 구간별 '실제/예측' 분위수. 표본이 적은 구간은 유형 전체 값으로."""
    d = d.filter(pl.col("pred") > 0).with_columns(lv=level_expr(), bin=bin_expr(), r=pl.col("true") / pl.col("pred"))
    by_bin = d.group_by("lv", "bin").agg(n=pl.len(), q10=pl.col("r").quantile(0.1), q50=pl.col("r").quantile(0.5),
                                         q90=pl.col("r").quantile(0.9), pred_min=pl.col("pred").min(), pred_max=pl.col("pred").max())
    by_lv = d.group_by("lv").agg(q10_all=pl.col("r").quantile(0.1), q50_all=pl.col("r").quantile(0.5), q90_all=pl.col("r").quantile(0.9))
    t = by_bin.join(by_lv, on="lv").with_columns(
        fallback=pl.col("n") < MIN_N,
        q10=pl.when(pl.col("n") < MIN_N).then("q10_all").otherwise("q10"),
        q50=pl.when(pl.col("n") < MIN_N).then("q50_all").otherwise("q50"),
        q90=pl.when(pl.col("n") < MIN_N).then("q90_all").otherwise("q90"),
    ).select(pl.col("lv").alias("level"), "bin", "n", "q10", "q50", "q90", "pred_min", "pred_max", "fallback")
    return t.sort("level", "pred_min")


def apply_interval(df: pl.DataFrame, table: pl.DataFrame, value: str = "value") -> pl.DataFrame:
    """df(level, value)에 value_lo/hi 추가. 구간 표에 없는 유형·구간은 그 유형의 전체 분위수로."""
    lvl_all = table.group_by("level").agg(q10_l=pl.col("q10").median(), q90_l=pl.col("q90").median())
    return (df.with_columns(_lv=level_expr(), _bin=bin_expr(value))
            .join(table.select(pl.col("level").alias("_lv"), pl.col("bin").alias("_bin"), "q10", "q90"), on=["_lv", "_bin"], how="left")
            .join(lvl_all.rename({"level": "_lv"}), on="_lv", how="left")
            .with_columns(value_lo=pl.col(value) * pl.coalesce("q10", "q10_l", pl.lit(1.0)),
                          value_hi=pl.col(value) * pl.coalesce("q90", "q90_l", pl.lit(1.0)))
            .drop("_lv", "_bin", "q10", "q90", "q10_l", "q90_l"))


def coverage(d: pl.DataFrame) -> pl.DataFrame:
    """분위수를 그 해를 뺀 나머지 해로 구해 그 해에 적용한 적중률 (유형별)."""
    parts = []
    for y in FOLD_YEARS:
        t = interval_table(d.filter(pl.col("year") != y))
        test = d.filter(pl.col("year") == y, pl.col("pred") > 0).rename({"pred": "value"})
        parts.append(apply_interval(test, t).with_columns(hit=(pl.col("true") >= pl.col("value_lo")) & (pl.col("true") <= pl.col("value_hi"))))
    a = pl.concat(parts).with_columns(lv=level_expr())
    rep = pl.concat([
        a.group_by("lv").agg(n=pl.len(), coverage=pl.col("hit").mean(), median_width_ratio=((pl.col("value_hi") - pl.col("value_lo")) / pl.col("value")).median()),
        a.with_columns(lv=pl.concat_str("lv", pl.lit(" / "), bin_expr("value"))).group_by("lv").agg(
            n=pl.len(), coverage=pl.col("hit").mean(), median_width_ratio=((pl.col("value_hi") - pl.col("value_lo")) / pl.col("value")).median()),
    ]).rename({"lv": "level_bin"}).sort("level_bin")
    rep.write_csv(REPORTS_DIR / "m2v_aoi_interval_coverage.csv")
    return rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="M2V Uncertainty Band: AOI × 월 구간 보정")
    ap.add_argument("--reuse-oof", action="store_true", help="이미 만든 oof_aoi_month_m2v.parquet로 분위수·적중률만")
    args = ap.parse_args(argv)
    start_run_log("m2v_interval")
    from planetarium.harmonize import aoi_membership
    membership, aoi_list = aoi_membership()
    levels = aoi_list.select("aoi_id", "level")
    d = pl.read_parquet(OOF_AOI_PATH) if args.reuse_oof and OOF_AOI_PATH.exists() else oof_aoi(membership, levels)
    t = interval_table(d)
    t.write_parquet(AOI_INTERVAL_PATH)
    cov = coverage(d)
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=3, tbl_width_chars=160):
        print(t)
        print(cov)


if __name__ == "__main__":
    main()
