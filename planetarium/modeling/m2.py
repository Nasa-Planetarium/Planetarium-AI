"""M2: NOAA-20 → S-NPP 연결 계수 (S-NPP 종료 후 2026-11-03부터 NOAA-20을 S-NPP 기준으로 환산).

방법
  - 두 위성이 모두 status=ok인 날(공통일)만으로 칸 × 월 감지 수를 집계 (M1과 같은 공통일 규칙).
    VIIRS 세 위성은 같은 센서이고 관측 시각이 1시간 안이라 (NOAA-21 → S-NPP → NOAA-20 순) 단순 비율로 충분하다고 본다.
  - 비율 = Σ S-NPP / Σ NOAA-20 를 (지역 × 달력 월)과 (지역 전체)로 계산.
    지역: world, 대륙, korea, argentina
  - 대체 규칙: 그 (지역, 월)의 두 위성 모두 감지가 있는 칸·월이 MIN_CELLS 미만이거나,
    칸·월 log1p 상관이 MIN_CORR 미만이면 그 지역 전체 비율을 쓴다. 지역 전체도 같은 기준에 못 미치면 world 값.
  - 기간: 정식 2018-05 ~ 2025-12 (계수 산출), 2026-01 ~ 2026-11-01 NRT (확인용, 따로 보고;
    2026-01~09는 지금 동아시아 자료만 있음).
  - NOAA-21(2023-03~)도 같은 방식으로 계산만 해서 보고 (적용하지 않음; 2023~2025 정식 자료는 지금 동아시아만).

출력
  models/m2_ratios.parquet        pair, region, month(0=전체), ratio, n_cells, corr, used_ratio, fallback
  reports/validation_m2.csv       pair, period, region, year, ratio, dev_from_1, n_cells, corr

실행  uv run python -m planetarium.modeling.m2
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import polars as pl

from planetarium.config import EAST_ASIA_BBOX, MODELS_DIR, REPORTS_DIR
from planetarium.monthly import STATUS_PATH, daily_coverage
from planetarium.runlog import start_run_log
from planetarium.train_table import aggregate_common, cell_attrs, to_wide, with_attrs

M2_RATIOS_PATH = MODELS_DIR / "m2_ratios.parquet"
MIN_CELLS = 200
MIN_CORR = 0.5
PAIRS = {
    "NOAA-20": {"other": "noaa20", "sp": (date(2018, 5, 1), date(2025, 12, 31)), "nrt": (date(2026, 1, 1), date(2026, 11, 1))},
    "NOAA-21": {"other": "noaa21", "sp": (date(2023, 3, 1), date(2025, 12, 31)), "nrt": (date(2026, 1, 1), date(2026, 11, 1))},
}


def common_dates(other: str, start: date, end: date) -> pl.DataFrame:
    st = pl.read_parquet(STATUS_PATH).filter(pl.col("satellite").is_in(["S-NPP", other]),
                                             pl.col("date_utc").is_between(start, end))
    w = st.pivot(on="satellite", index="date_utc", values="status")
    w = w.filter((pl.col("S-NPP") == "ok") & (pl.col(other) == "ok"))
    return w.select(pl.col("date_utc").alias("date_local"),
                    year=pl.col("date_utc").dt.year().cast(pl.Int16), month=pl.col("date_utc").dt.month().cast(pl.Int8))


def pair_table(other: str, prefix: str, start: date, end: date) -> pl.DataFrame:
    """두 위성이 모두 덮는 지역만 비교: 날짜마다 한쪽이라도 동아시아만 있으면 동아시아 칸만 쓴다
    (예: NOAA-21 정식 2023~2025, 2026-01~09 다운로드는 지금 동아시아만)."""
    dates = common_dates(other, start, end)
    cov = daily_coverage()
    region = [
        "global" if cov.get(("S-NPP", d)) == "global" and cov.get((other, d)) == "global" else "east_asia"
        for d in dates["date_local"].to_list()
    ]
    dates = dates.with_columns(cov=pl.Series(region))
    sats = {"S-NPP": "snpp", other: prefix}
    parts = []
    for c in ("global", "east_asia"):
        dd = dates.filter(pl.col("cov") == c)
        if dd.height == 0:
            continue
        agg = aggregate_common(dd, sats)
        if c == "east_asia":
            x0, y0, x1, y1 = EAST_ASIA_BBOX
            lat = (pl.col("cell_id") // 3600 + 0.5) * 0.1 - 90
            lon = (pl.col("cell_id") % 3600 + 0.5) * 0.1 - 180
            agg = agg.filter(lat.is_between(y0, y1), lon.is_between(x0, x1))
        parts.append(agg)
    agg = pl.concat(parts).group_by("cell_id", "year", "month", "satellite").agg(
        pl.col("n_det", "n_det_conf", "n_night", "frp_sum").sum(), pl.col("frp_max").max(), pl.col("active_days").sum())
    wide = to_wide(agg, sats)
    return with_attrs(wide, cell_attrs()).with_columns(pl.col("snpp_n_det", f"{prefix}_n_det").fill_null(0))


def masks() -> dict[str, pl.Expr]:
    conts = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
    m = {"world": pl.lit(True), "korea": pl.col("is_korea"), "argentina": pl.col("adm0_a3") == "ARG"}
    m.update({f"continent:{c}": pl.col("continent") == c for c in conts})
    return m


def ratio_stats(d: pl.DataFrame, prefix: str, by: list[str]) -> pl.DataFrame:
    o = f"{prefix}_n_det"
    both = (pl.col("snpp_n_det") > 0) & (pl.col(o) > 0)
    return d.group_by(by).agg(
        snpp=pl.col("snpp_n_det").sum(), other=pl.col(o).sum(),
        n_cells=both.sum(),
        corr=pl.corr(pl.col("snpp_n_det").log1p(), pl.col(o).log1p()),
    ).with_columns(ratio=pl.col("snpp") / pl.col("other"))


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="NOAA-20/21 → S-NPP 연결 계수").parse_args(argv)
    start_run_log("m2")
    ratios, val = [], []
    for sat, cfg in PAIRS.items():
        prefix = cfg["other"]
        for period in ("sp", "nrt"):
            d = pair_table(sat, prefix, *cfg[period])
            if d.height == 0:
                continue
            for region, m in masks().items():
                sub = d.filter(m)
                if sub.height == 0:
                    continue
                yr = ratio_stats(sub, prefix, ["year"]).with_columns(
                    pair=pl.lit(f"S-NPP/{sat}"), period=pl.lit(period.upper()), region=pl.lit(region))
                val.append(yr)
                allp = ratio_stats(sub, prefix, []).with_columns(
                    pair=pl.lit(f"S-NPP/{sat}"), period=pl.lit(period.upper()), region=pl.lit(region),
                    year=pl.lit(None, dtype=pl.Int16))
                val.append(allp.select(yr.columns))
                if period == "sp":
                    mon = ratio_stats(sub, prefix, ["month"]).with_columns(month=pl.col("month").cast(pl.Int8))
                    ov = allp.with_columns(month=pl.lit(0, dtype=pl.Int8)).select(mon.columns)
                    ratios.append(pl.concat([ov, mon]).with_columns(pair=pl.lit(f"S-NPP/{sat}"), region=pl.lit(region)))

    v = pl.concat(val, how="diagonal_relaxed").with_columns(dev_from_1=pl.col("ratio") - 1).select(
        "pair", "period", "region", "year", "ratio", "dev_from_1", "n_cells", "corr", "snpp", "other").sort(
        "pair", "period", "region", "year", nulls_last=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    v.write_csv(REPORTS_DIR / "validation_m2.csv")

    r = pl.concat(ratios)
    ok = (pl.col("n_cells") >= MIN_CELLS) & (pl.col("corr") >= MIN_CORR)
    overall = r.filter(pl.col("month") == 0).select("pair", "region", ratio_all="ratio", ok_all=ok)
    world_m = r.filter(pl.col("region") == "world").select("pair", "month", ratio_world="ratio")
    world_all = r.filter(pl.col("region") == "world", pl.col("month") == 0).select("pair", ratio_world_all="ratio")
    r = (
        r.join(overall, on=["pair", "region"], how="left")
        .join(world_m, on=["pair", "month"], how="left")
        .join(world_all, on="pair", how="left")
        .with_columns(
            fallback=pl.when(ok).then(pl.lit("none"))
            .when(pl.col("ok_all")).then(pl.lit("region_overall"))
            .otherwise(pl.lit("world")),
        )
        .with_columns(used_ratio=pl.when(pl.col("fallback") == "none").then(pl.col("ratio"))
                      .when(pl.col("fallback") == "region_overall").then(pl.col("ratio_all"))
                      .otherwise(pl.coalesce("ratio_world", "ratio_world_all")))
        .select("pair", "region", "month", "ratio", "n_cells", "corr", "used_ratio", "fallback", "snpp", "other")
        .sort("pair", "region", "month")
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    r.write_parquet(M2_RATIOS_PATH)

    cfgp = dict(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_formatting="ASCII_MARKDOWN",
                tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, float_precision=3)
    with pl.Config(**cfgp):
        print("[M2 전체 비율 (정식/NRT)]")
        print(v.filter(pl.col("year").is_null()).select("pair", "period", "region", "ratio", "n_cells", "corr"))
        print("[연도별 S-NPP/NOAA-20 비율 (정식)]")
        print(v.filter(pl.col("pair") == "S-NPP/NOAA-20", pl.col("period") == "SP", pl.col("year").is_not_null(),
                       pl.col("region").is_in(["world", "korea", "argentina"]))
              .pivot(on="region", index="year", values="ratio").sort("year"))
        print("[적용 비율 대체 현황]")
        print(r.filter(pl.col("pair") == "S-NPP/NOAA-20").group_by("region").agg(
            months_none=(pl.col("fallback") == "none").sum(), months_fallback=(pl.col("fallback") != "none").sum(),
            used_min=pl.col("used_ratio").min(), used_max=pl.col("used_ratio").max()).sort("region"))


if __name__ == "__main__":
    main()
