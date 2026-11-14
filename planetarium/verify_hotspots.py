"""통합 핫스팟 4단계 검증.

  (a) 칸·월: 기준 점 가중치 합 + residual(기준 점이 없는 칸·월 + 배경 예측) = harmonized_cell_month.value (+ 배경)
  (b) AOI·월: (a)의 칸 합을 면적 가중으로 묶은 값 = aoi_month.value
  (c) harmonized_cell_day: AOI 일 합계를 월로 묶은 값(value + residual_month) = aoi_month.value
  예시: 한국(KR) 2005-03, 2025-03

출력  reports/hotspots_verify_cell_month.csv, reports/hotspots_verify_aoi_month.csv (연도별 최대 차이)
실행  uv run python -m planetarium.verify_hotspots
"""

from __future__ import annotations

import time

import polars as pl
from tqdm import tqdm

from planetarium.config import REPORTS_DIR
from planetarium.harmonize import AOI_MONTH_PATH, HCM_BACKGROUND_DIR, HCM_PATH, aoi_membership
from planetarium.hotspots import CELL_DAY_PATH, HOTSPOTS_DIR, RESIDUAL_PATH
from planetarium.runlog import start_run_log

KEYS = ["cell_id", "year", "month"]
TOL = 1e-3


def year_cells(y: int) -> pl.DataFrame:
    """그 해 칸·월: 점 가중치 합(w), residual(r: 기준 점 없는 칸·월), 배경(bg), 칸 값(value)."""
    ref = pl.scan_parquet(HOTSPOTS_DIR / "reference" / f"year={y}" / "*" / "*.parquet")
    w = ref.group_by("cell_id", year=pl.col("date_local").dt.year().cast(pl.Int16),
                     month=pl.col("date_local").dt.month().cast(pl.Int8)).agg(
        w=pl.col("harmonized_weight").cast(pl.Float64).sum()).collect()
    r = pl.scan_parquet(RESIDUAL_PATH).filter(pl.col("year") == y).group_by(KEYS).agg(r=pl.col("value").sum()).collect()
    bgf = HCM_BACKGROUND_DIR / f"{y}.parquet"
    bg = (pl.scan_parquet(bgf).group_by(KEYS).agg(bg=pl.col("value").cast(pl.Float64).sum()).collect()
          if bgf.exists() else pl.DataFrame(schema={"cell_id": pl.Int64, "year": pl.Int16, "month": pl.Int8, "bg": pl.Float64}))
    cv = pl.scan_parquet(HCM_PATH).filter(pl.col("year") == y).select(*KEYS, "value").collect()
    out = cv
    for t in (w, r, bg):
        out = out.join(t, on=KEYS, how="full", coalesce=True)
    return out.with_columns(pl.col("value", "w", "r", "bg").fill_null(0.0))


def main(argv: list[str] | None = None) -> None:
    start_run_log("verify_hotspots")
    t0 = time.perf_counter()
    membership, _ = aoi_membership()
    am = pl.read_parquet(AOI_MONTH_PATH, columns=["aoi_id", "year", "month", "value", "status"])
    years = sorted(am["year"].unique().to_list())
    cell_rows, aoi_rows, aoi_parts = [], [], []
    for y in tqdm(years, desc="검증 (a)(b)", unit="year", dynamic_ncols=True):
        c = year_cells(y)
        d = (c["w"] + c["r"] - c["value"]).abs()
        cell_rows.append({"year": y, "cells": c.height, "max_abs_diff": float(d.max() or 0), "n_over_tol": int((d > TOL).sum()),
                          "points_weight": float(c["w"].sum()), "residual": float(c["r"].sum()), "background": float(c["bg"].sum()),
                          "cell_value": float(c["value"].sum())})
        a = (c.lazy().join(membership.lazy(), on="cell_id")
             .group_by("aoi_id", "year", "month")
             .agg(pts=((pl.col("w") + pl.col("r") + pl.col("bg")) * pl.col("weight")).sum()).collect())
        aoi_parts.append(a)
    A = pl.concat(aoi_parts).join(am, on=["aoi_id", "year", "month"], how="right").filter(pl.col("value").is_not_null()) \
        .with_columns(pl.col("pts").fill_null(0.0), d=(pl.col("pts").fill_null(0.0) - pl.col("value")).abs(),
                      rel=(pl.col("pts").fill_null(0.0) - pl.col("value")).abs() / pl.col("value").abs().clip(lower_bound=1.0))

    # (c) harmonized_cell_day → 월 → AOI
    cd = pl.scan_parquet(CELL_DAY_PATH)
    cdm = cd.group_by(KEYS).agg(v=(pl.col("value") + pl.col("residual_month").fill_null(0)).sum()).collect()
    C = (cdm.lazy().join(membership.lazy(), on="cell_id").group_by("aoi_id", "year", "month")
         .agg(cd=(pl.col("v") * pl.col("weight")).sum()).collect()
         .join(am, on=["aoi_id", "year", "month"], how="right").filter(pl.col("value").is_not_null())
         .with_columns(pl.col("cd").fill_null(0.0))
         .with_columns(d=(pl.col("cd") - pl.col("value")).abs(),
                       rel=(pl.col("cd") - pl.col("value")).abs() / pl.col("value").abs().clip(lower_bound=1.0)))

    cr = pl.DataFrame(cell_rows)
    cr.write_csv(REPORTS_DIR / "hotspots_verify_cell_month.csv")
    ar = (A.group_by("year").agg(aoi_months=pl.len(), b_max_abs=pl.col("d").max(), b_max_rel=pl.col("rel").max(),
                                 b_over_tol=(pl.col("rel") > TOL).sum())
          .join(C.group_by("year").agg(c_max_abs=pl.col("d").max(), c_max_rel=pl.col("rel").max(),
                                       c_over_tol=(pl.col("rel") > TOL).sum()), on="year").sort("year"))
    ar.write_csv(REPORTS_DIR / "hotspots_verify_aoi_month.csv")
    cfg = dict(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_formatting="ASCII_MARKDOWN",
               tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, float_precision=6, thousands_separator=",")
    with pl.Config(**cfg):
        print("(a) 칸·월: 점 가중치 + residual = 칸 value");  print(cr)
        print("(b) AOI·월 (점+residual+배경) / (c) harmonized_cell_day 월 합 vs aoi_month.value"); print(ar)
        for y, m in ((2005, 3), (2025, 3)):
            ex = A.filter(pl.col("aoi_id") == "KR", pl.col("year") == y, pl.col("month") == m)
            ec = C.filter(pl.col("aoi_id") == "KR", pl.col("year") == y, pl.col("month") == m)
            print(f"KR {y}-{m:02d}: aoi_month {ex['value'][0]:.4f} | 점+residual+배경 {ex['pts'][0]:.4f} | cell_day 월 합 {ec['cd'][0]:.4f}")
    print(f"검증 끝 ({time.perf_counter() - t0:,.0f}초)")


if __name__ == "__main__":
    main()
