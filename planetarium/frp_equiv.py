"""FRP를 S-NPP 기준으로 환산 (frp_snpp_equiv) — 별도 파일, 기존 산출물은 그대로.

비율 (ratios):  같은 0.1° 칸·같은 현지 날짜에 S-NPP와 다른 위성이 모두 감지한 칸·일에서
                r = Σ FRP(위성) / Σ FRP(S-NPP) 의 중앙값 (낮·밤 따로). 자료: FIRMS 표준 처리(SP) 통합 파일, type 0, 고정 열원 칸 제외.
                위성별 비교 기간: Aqua·Terra 2012-02~2025, NOAA-20 2018~2025, NOAA-21 2024~2025 (SP가 있는 해).
환산:           frp_snpp_equiv = frp / r(위성, 낮·밤). S-NPP는 r = 1.
                MODIS 2000~2011년 점에도 2012년 이후에 구한 비율을 그대로 쓴다 (가정; 리포트에 기록).
산출:           reports/frp_snpp_ratios.csv (위성 × 낮·밤 × 연도별 비율과 표본 수, 전체 기간 비율)
                data/processed/harmonized_hotspots_frp/year=YYYY.parquet  점 키 + layer + frp + frp_snpp_equiv (기준·기타 점, 고정 열원 제외)
                data/processed/harmonized_cell_day_frp.parquet  칸 × 현지 날짜: 기준 점 frp 합, frp_snpp_equiv 합, 최대, 점 수
설계 참고: 팀원 Jonghun Shin(ozingmw)의 firms-1km-backend FRP 위성 간 환산(ratio) 경로.

실행  uv run python -m planetarium.frp_equiv ratios|points|all
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import date

import polars as pl
from tqdm import tqdm

from planetarium.config import FIRMS_MERGED_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log
from planetarium.static_exclusion import is_static_expr, mask_ids

RATIO_PATH = REPORTS_DIR / "frp_snpp_ratios.csv"
PTS_DIR = PROCESSED_DATA_DIR / "harmonized_hotspots_frp"
CD_PATH = PROCESSED_DATA_DIR / "harmonized_cell_day_frp.parquet"
HOT = PROCESSED_DATA_DIR / "harmonized_hotspots"
PT_KEY = ["satellite", "latitude", "longitude", "acq_datetime_utc"]
SATS = {"Aqua": range(2012, 2026), "Terra": range(2012, 2026), "N20": range(2018, 2026), "N21": range(2024, 2026)}
TERRA_LAST_YEAR = 2019


def local_date() -> pl.Expr:
    return (pl.col("acq_datetime_utc").dt.replace_time_zone(None)
            + pl.duration(minutes=(pl.col("longitude") * 4).round().cast(pl.Int64))).dt.date()


def ratios() -> pl.DataFrame:
    mask = mask_ids()
    rows = []
    for y in tqdm(range(2012, 2026), desc="FRP 겹침 칸·일", unit="year", dynamic_ncols=True):
        lf = (pl.scan_parquet(str(FIRMS_MERGED_DIR / "*.parquet"))
              .filter(pl.col("source") == "SP", pl.col("type") == 0, pl.col("acq_date").is_between(date(y, 1, 1), date(y, 12, 31)),
                      pl.col("frp") >= 0, ~is_static_expr(mask))
              .with_columns(d=local_date(), cell=((pl.col("latitude") + 90) / 0.1).floor().cast(pl.Int64) * 3600
                            + ((pl.col("longitude") + 180) / 0.1).floor().cast(pl.Int64))
              .filter(pl.col("d") >= date(2012, 2, 1))
              .group_by("cell", "d", "satellite", "daynight").agg(frp=pl.col("frp").cast(pl.Float64).sum(), n=pl.len()))
        cd = lf.collect(engine="streaming")
        sn = cd.filter(pl.col("satellite") == "N").select("cell", "d", "daynight", frp_sn="frp", n_sn="n")
        for sat, yrs in SATS.items():
            if y not in yrs:
                continue
            j = cd.filter(pl.col("satellite") == sat).join(sn, on=["cell", "d", "daynight"]).filter(pl.col("frp_sn") > 0, pl.col("frp") > 0)
            for (dn,), g in j.group_by("daynight"):
                r = (g["frp"] / g["frp_sn"]).log()
                rows.append({"satellite": sat, "daynight": dn, "year": y, "cell_days": g.height, "median_ratio": math.exp(r.median()),
                             "sum_ratio": float(g["frp"].sum() / g["frp_sn"].sum()),
                             "log_ratio_sum": float(r.sum()), "pixel_frp_median_sat": float((g["frp"] / g["n"]).median()),
                             "pixel_frp_median_snpp": float((g["frp_sn"] / g["n_sn"]).median())})
        del cd, sn
        tqdm.write(f"[{y}] 최대 메모리 {peak_memory_gb() or 0:.1f} GB")
    by_year = pl.DataFrame(rows).sort("satellite", "daynight", "year")
    # 전체 기간: 칸·일 로그 비율의 가중 평균은 연도별 중앙값으로 근사하지 않고, 연도별 중앙값의 표본 수 가중 중앙값을 쓴다
    # Terra는 2020년 무렵부터 궤도가 밀려 관측 시각이 이르게 바뀌므로(낮 비율 0.98 → 0.75), 2000~2011년 점에 쓸 비율은 2012~2019년만으로
    use = by_year.filter((pl.col("satellite") != "Terra") | (pl.col("year") <= TERRA_LAST_YEAR))
    tot = (use.group_by("satellite", "daynight").agg(
        cell_days=pl.col("cell_days").sum(), years=pl.len(),
        ratio=(pl.col("median_ratio").log() * pl.col("cell_days")).sum() / pl.col("cell_days").sum(),
        ratio_year_min=pl.col("median_ratio").min(), ratio_year_max=pl.col("median_ratio").max(),
        sum_ratio=pl.col("sum_ratio").median())
        .with_columns(ratio=pl.col("ratio").exp(), year=pl.lit("all")).sort("satellite", "daynight"))
    out = pl.concat([by_year.with_columns(pl.col("year").cast(pl.String)).select("satellite", "daynight", "year", "cell_days", ratio="median_ratio",
                                                                                  sum_ratio="sum_ratio", pixel_frp_median_sat="pixel_frp_median_sat",
                                                                                  pixel_frp_median_snpp="pixel_frp_median_snpp"),
                     tot.select("satellite", "daynight", "year", "cell_days", "ratio", "sum_ratio", "ratio_year_min", "ratio_year_max")],
                    how="diagonal_relaxed")
    out.write_csv(RATIO_PATH)
    with pl.Config(tbl_rows=-1, float_precision=3, tbl_width_chars=200):
        print(tot)
    return tot


def factor_table() -> pl.DataFrame:
    t = pl.read_csv(RATIO_PATH).filter(pl.col("year") == "all").select("satellite", "daynight", "ratio")
    return pl.concat([t, pl.DataFrame({"satellite": ["N", "N"], "daynight": ["D", "N"], "ratio": [1.0, 1.0]})])


def points() -> None:
    mask, fac = mask_ids(), factor_table()
    PTS_DIR.mkdir(parents=True, exist_ok=True)
    cds, log = [], []
    years = sorted(int(p.name.split("=")[1]) for p in (HOT / "reference").glob("year=*"))
    for y in tqdm(years, desc="점 FRP 환산", unit="year", dynamic_ncols=True):
        parts = []
        for layer in ("reference", "other"):
            d = HOT / layer / f"year={y}"
            if not d.exists():
                continue
            parts.append(pl.scan_parquet(str(d / "*" / "*.parquet")).filter(~is_static_expr(mask))
                         .select(*PT_KEY, "date_local", "cell_id", "daynight", "frp", "is_reference", layer=pl.lit(layer)))
        df = (pl.concat(parts, how="diagonal_relaxed").join(fac.lazy(), on=["satellite", "daynight"], how="left")
              .with_columns(frp_snpp_equiv=(pl.col("frp") / pl.col("ratio")).cast(pl.Float32)).drop("ratio").collect(engine="streaming"))
        df.select(*PT_KEY, "layer", "frp", "frp_snpp_equiv").write_parquet(PTS_DIR / f"year={y}.parquet", compression="zstd")
        cds.append(df.filter("is_reference").group_by("cell_id", "date_local").agg(
            frp_sum=pl.col("frp").cast(pl.Float64).sum(), frp_snpp_equiv_sum=pl.col("frp_snpp_equiv").cast(pl.Float64).sum(),
            frp_snpp_equiv_max=pl.col("frp_snpp_equiv").max(), n_points=pl.len().cast(pl.Int32)))
        log.append({"year": y, "points": df.height, "no_factor": int(df["frp_snpp_equiv"].is_null().sum() - df["frp"].is_null().sum()),
                    "peak_mem_gb": round(peak_memory_gb() or 0, 2)})
        del df
    pl.concat(cds).sort("cell_id", "date_local").write_parquet(CD_PATH, compression="zstd")
    lg = pl.DataFrame(log)
    lg.write_csv(REPORTS_DIR / "frp_snpp_equiv_build.csv")
    print(f"[완료] {PTS_DIR.name}/ {int(lg['points'].sum()):,}점, {CD_PATH.name}, 비율 없는 점 {int(lg['no_factor'].sum()):,}, "
          f"최대 메모리 {peak_memory_gb() or 0:.2f} GB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["ratios", "points", "all"])
    a = ap.parse_args(argv)
    start_run_log("frp_equiv")
    t0 = time.perf_counter()
    if a.step in ("ratios", "all"):
        ratios()
    if a.step in ("points", "all"):
        points()
    print(f"[완료] {a.step} {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
