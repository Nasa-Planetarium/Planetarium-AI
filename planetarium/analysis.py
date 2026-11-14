"""T7: 같은 달 과거 대비 비교 ('이번 달 감지 수가 예년 같은 달보다 많은가?').

compare_month(aoi_id, year, month) -> dict
  기준 분포: 2001~2025 같은 달 값 (대상 연도 제외, status == ok인 달만)
  반환: value, baseline_median, baseline_p25, baseline_p75, percentile, ratio_to_median, n_baseline_years,
        verdict ∈ {above, similar, below, uncertain}, status, low_sample
  verdict 규칙
    - n_baseline_years < 10, 또는 대상 월 status가 missing/partial이면 uncertain
    - low_sample(기준 분포 75백분위 < 1: 예년 같은 달 감지가 거의 없는 지역): 값이 3 이상이면 above, 아니면 uncertain
      (작은 지역에서 0과 1의 차이로 백분위가 크게 흔들리는 것을 막기 위함)
    - percentile ≥ 80 → above, ≤ 20 → below, 그 외 similar
  percentile = (기준값 < 값 개수 + 0.5 × 같은 값 개수) / 기준 연도 수 × 100
  감지 수는 산불 건수가 아니라 관측 활동이다.

출력  data/processed/aoi_month_compare.parquet  (모든 AOI × 월, 백엔드가 그대로 읽음)

실행  uv run python -m planetarium.analysis
"""

from __future__ import annotations

import argparse
import time

import polars as pl

from planetarium.config import PROCESSED_DATA_DIR
from planetarium.harmonize import AOI_MONTH_PATH
from planetarium.runlog import start_run_log

COMPARE_PATH = PROCESSED_DATA_DIR / "aoi_month_compare.parquet"
BASE_YEARS = (2001, 2025)
MIN_YEARS = 10
ABOVE, BELOW = 80.0, 20.0
LOW_SAMPLE_P75 = 1.0
LOW_SAMPLE_ABOVE_VALUE = 3.0


def _compare(target: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """target(aoi_id, year, month, value, status) × base(aoi_id, year, month, value) -> 비교표."""
    b = base.select("aoi_id", "month", pl.col("year").alias("b_year"), pl.col("value").alias("b"))
    pairs = target.join(b, on=["aoi_id", "month"], how="left").filter(
        pl.col("b_year").is_null() | (pl.col("b_year") != pl.col("year")))
    stats = pairs.group_by("aoi_id", "year", "month").agg(
        value=pl.col("value").first(), status=pl.col("status").first(),
        baseline_median=pl.col("b").median(), baseline_p25=pl.col("b").quantile(0.25, "linear"),
        baseline_p75=pl.col("b").quantile(0.75, "linear"),
        n_baseline_years=pl.col("b").count().cast(pl.Int16),
        n_less=(pl.col("b") < pl.col("value")).sum(), n_equal=(pl.col("b") == pl.col("value")).sum(),
    )
    low = pl.col("baseline_p75") < LOW_SAMPLE_P75
    pct = (pl.col("n_less") + 0.5 * pl.col("n_equal")) / pl.col("n_baseline_years") * 100
    return stats.with_columns(
        percentile=pl.when(pl.col("n_baseline_years") > 0).then(pct),
        ratio_to_median=pl.when(pl.col("baseline_median") > 0).then(pl.col("value") / pl.col("baseline_median")),
        low_sample=low.fill_null(False),
    ).with_columns(
        verdict=pl.when((pl.col("n_baseline_years") < MIN_YEARS) | pl.col("status").is_in(["missing", "partial"])
                        | pl.col("value").is_null()).then(pl.lit("uncertain"))
        .when(pl.col("low_sample")).then(
            pl.when(pl.col("value") >= LOW_SAMPLE_ABOVE_VALUE).then(pl.lit("above")).otherwise(pl.lit("uncertain")))
        .when(pl.col("percentile") >= ABOVE).then(pl.lit("above"))
        .when(pl.col("percentile") <= BELOW).then(pl.lit("below"))
        .otherwise(pl.lit("similar")),
    ).select("aoi_id", "year", "month", "value", "baseline_median", "baseline_p25", "baseline_p75", "percentile",
             "ratio_to_median", "n_baseline_years", "verdict", "status", "low_sample")


def _base(am: pl.LazyFrame) -> pl.LazyFrame:
    return am.filter(pl.col("year").is_between(*BASE_YEARS), pl.col("status") == "ok", pl.col("value").is_not_null())


def compare_month(aoi_id: str, year: int, month: int) -> dict:
    am = pl.scan_parquet(AOI_MONTH_PATH).filter(pl.col("aoi_id") == aoi_id)
    target = am.filter(pl.col("year") == year, pl.col("month") == month).select("aoi_id", "year", "month", "value", "status")
    res = _compare(target, _base(am)).collect()
    if res.height == 0:
        raise KeyError(f"{aoi_id} {year}-{month:02d} 자료가 없습니다")
    return res.to_dicts()[0]


def build_compare() -> pl.DataFrame:
    am = pl.scan_parquet(AOI_MONTH_PATH)
    df = _compare(am.select("aoi_id", "year", "month", "value", "status"), _base(am)).sort("aoi_id", "year", "month").collect()
    df.write_parquet(COMPARE_PATH, compression="zstd")
    return df


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="같은 달 과거 대비 비교표").parse_args(argv)
    start_run_log("analysis")
    t0 = time.perf_counter()
    df = build_compare()
    print(f"[완료] {COMPARE_PATH.name}: {df.height:,}행 ({time.perf_counter() - t0:,.0f}초)")
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=2, thousands_separator=","):
        print(df.group_by("verdict").len().sort("verdict"))
        print(df.filter(pl.col("aoi_id").is_in(["KR", "ADM0_ARG", "WORLD"]), pl.col("year") >= 2025)
              .select("aoi_id", "year", "month", "value", "baseline_median", "percentile", "ratio_to_median",
                      "n_baseline_years", "verdict", "status").sort("aoi_id", "year", "month"))
        print(compare_month("KR", 2025, 3))


if __name__ == "__main__":
    main()
