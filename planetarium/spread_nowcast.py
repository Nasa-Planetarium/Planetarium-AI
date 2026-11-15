"""다음 날 확산 예측(spread nowcast) — 위성 감지 기반 단기 확산 가능성 (물리 시뮬레이션 아님).

질문: 오늘(현지 날짜 d) 감지가 있는 곳 주변의 작은 칸(0.025°, 약 2.5 km)에 내일(d+1) 감지가 생길 확률.

단계 (단계마다 보고·확인)
  samples  후보 칸 표본 + 감지 기반 특성 + 라벨  → data/interim/spread_nowcast/samples_<region>_<year>.parquet
  (weather, train, case, latest 는 다음 단계에서 추가)

범위
  KR : 상자 124.5~132E, 33~38.7N (상자 안 모든 감지. 일본 규슈·쓰시마, 북한 일부 포함 → cand_adm0로 구분)
  ARG: 아르헨티나 국가 경계 (활성 칸 = adm0_a3 == ARG 점, 라벨은 국경 밖 감지도 포함하도록 상자 + 0.1°에서 읽음)
  기간: 2012-02-01 ~ 2025-12-30 (라벨 날짜 ≤ 2025-12-31). 기준 점(harmonized_hotspots/reference) 중 VIIRS(S-NPP)만.

표본
  후보 칸 = 오늘 활성 칸과 그 주변 반경 3칸(7×7) 안의 칸. 라벨 = 다음 현지 날짜에 그 칸에 감지가 있는가 (0/1).
  자료 상태: S-NPP satellite_day_status가 d와 d+1 모두 ok인 날만 (결측·저하일 라벨은 '감지 없음'이 아니라 '모름'이므로 제외).
  d-1이 ok가 아니면 어제 특성은 0이 되므로 prev_ok 플래그를 둔다.

특성 (누수 금지: d+1 정보 없음)
  self/3×3/7×7 감지 수·FRP 합·최대 (오늘, 어제), 연속 연소 일수(자기 칸, 7×7 최대), 가장 가까운 다른 활성 칸 거리(km),
  8방향(후보 칸 → 활성 칸 방향, 북에서 시계 방향 45° 구간) 활성 칸 수,
  확산 벡터 sx, sy = Σ (활성 칸 → 후보 칸 단위 벡터 × 감지 수 / 거리)  (기상 단계에서 풍향과의 cos = 바람 정렬 지수)
  월(sin, cos), 위도·경도.

메모리: 연 단위로 처리, 연도 파일이 있으면 건너뜀(--resume 기본). 최대 메모리를 로그에 남긴다.

실행  uv run python -m planetarium.spread_nowcast samples [--regions KR ARG] [--years 2012 2013] [--force]
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from planetarium.monthly import STATUS_PATH
from planetarium.runlog import peak_memory_gb, start_run_log
from planetarium.world import WORLD_CELLS_PATH

RES = 0.025
N_LON = round(360 / RES)            # 14,400
RADIUS = 3                          # 후보: 활성 칸 주변 7×7
KM_PER_CELL = RES * 111.32          # 위도 방향 칸 크기(km), 경도 방향은 cos(위도) 곱
START, END = date(2012, 2, 1), date(2025, 12, 31)
HOTSPOT_REF = PROCESSED_DATA_DIR / "harmonized_hotspots" / "reference"
OUT_DIR = INTERIM_DATA_DIR / "spread_nowcast"
REGIONS = {
    # name: (상자 lon0, lat0, lon1, lat1), 활성 점 조건
    "KR": ((124.5, 33.0, 132.0, 38.7), None),
    "ARG": ((-73.7, -55.2, -53.5, -21.7), "ARG"),
}
PAD = 0.1  # 라벨·주변 감지는 상자보다 조금 넓게 읽는다 (경계 칸)
DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def sample_path(region: str, year: int):
    return OUT_DIR / f"samples_{region}_{year}.parquet"


# ── 입력 ─────────────────────────────────────────────────────────────────

def load_points(region: str, first: date, last: date, sat: str = "N") -> pl.DataFrame:
    """기준 점 중 S-NPP(VIIRS), 현지 날짜 first~last, 상자+PAD 안. is_src = 활성 칸 조건을 만족하는 점. 고정 열원 점 제외(v1.2)."""
    from planetarium.static_exclusion import is_static_expr
    (x0, y0, x1, y1), adm0 = REGIONS[region]
    lf = pl.scan_parquet(HOTSPOT_REF / "**" / "*.parquet", hive_partitioning=True).filter(
        pl.col("year").is_between(first.year, last.year),
        pl.col("date_local").is_between(first, last),
        pl.col("satellite") == sat,
        pl.col("longitude").is_between(x0 - PAD, x1 + PAD), pl.col("latitude").is_between(y0 - PAD, y1 + PAD),
        ~is_static_expr(),  # v1.2: 고정 열원 점 제외 (학습 표본은 제외 전 자료로 만들었음)
    )
    inside = pl.col("longitude").is_between(x0, x1) & pl.col("latitude").is_between(y0, y1)
    src = inside if adm0 is None else (pl.col("adm0_a3") == adm0)
    return lf.select(
        "date_local", "frp",
        iy=((pl.col("latitude") + 90) / RES).floor().cast(pl.Int32),
        ix=((pl.col("longitude") + 180) / RES).floor().cast(pl.Int32),
        is_src=src,
    ).collect()


def cell_days(points: pl.DataFrame) -> pl.DataFrame:
    """칸 × 날짜: 감지 수, FRP 합·최대, 활성 칸 조건(src)."""
    return points.group_by("date_local", "iy", "ix").agg(
        n=pl.len().cast(pl.Int32), frp_sum=pl.col("frp").sum().cast(pl.Float32), frp_max=pl.col("frp").max().cast(pl.Float32),
        src=pl.col("is_src").any())


def add_streak(cd: pl.DataFrame) -> pl.DataFrame:
    """연속 연소 일수: 그 칸이 오늘까지 며칠 연속 활성이었나 (오늘 포함)."""
    s = cd.sort("iy", "ix", "date_local").with_columns(
        _brk=((pl.col("date_local").diff().dt.total_days() != 1) | (pl.col("iy").diff() != 0) | (pl.col("ix").diff() != 0))
        .fill_null(True))
    s = s.with_columns(_run=pl.col("_brk").cum_sum())
    return s.with_columns(streak=(pl.int_range(pl.len()).over("_run") + 1).cast(pl.Int16)).drop("_brk", "_run")


def ok_days() -> set[date]:
    st = pl.read_parquet(STATUS_PATH).filter(pl.col("satellite") == "S-NPP", pl.col("status") == "ok")
    return set(st["date_utc"].to_list())


def cell_adm0() -> pl.DataFrame:
    """0.1° 세계 칸 → adm0 (0.025° 칸 중심을 0.1° 칸으로 찾아 붙인다)."""
    return pl.read_parquet(WORLD_CELLS_PATH, columns=["cell_id", "adm0_a3"])


# ── 후보 표본 ─────────────────────────────────────────────────────────────

def offsets() -> pl.DataFrame:
    rows = []
    for dy in range(-RADIUS, RADIUS + 1):
        for dx in range(-RADIUS, RADIUS + 1):
            ang = (math.degrees(math.atan2(-dx, -dy)) + 360) % 360  # 후보 → 활성 칸 방향 (북=0, 시계 방향); 활성 = 후보 - offset
            rows.append({"dy": dy, "dx": dx, "cheb": max(abs(dx), abs(dy)), "sector": int(((ang + 22.5) % 360) // 45)})
    return pl.DataFrame(rows, schema={"dy": pl.Int32, "dx": pl.Int32, "cheb": pl.Int32, "sector": pl.Int8})


def neighborhood(active: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """활성 칸 목록 → 후보 칸별 주변 집계. active: date_local, iy, ix, n, frp_sum, frp_max, streak."""
    off = offsets()
    e = active.join(off, how="cross").with_columns(iy=pl.col("iy") + pl.col("dy"), ix=pl.col("ix") + pl.col("dx"))
    # 거리(km): 위도에 따른 경도 방향 축소
    coslat = ((pl.col("iy").cast(pl.Float64) + 0.5) * RES - 90).radians().cos()
    ex, ey = pl.col("dx").cast(pl.Float64) * coslat * KM_PER_CELL, pl.col("dy").cast(pl.Float64) * KM_PER_CELL
    e = e.with_columns(dist=(ex ** 2 + ey ** 2).sqrt(), ux=ex, uy=ey)
    is_self = pl.col("cheb") == 0
    in3 = pl.col("cheb") <= 1
    other = ~is_self
    aggs = [
        pl.col("n").filter(is_self).sum().alias(f"{prefix}self_n"),
        pl.col("frp_sum").filter(is_self).sum().alias(f"{prefix}self_frp"),
        pl.col("n").filter(in3).sum().alias(f"{prefix}n3"),
        pl.col("frp_sum").filter(in3).sum().alias(f"{prefix}frp3"),
        pl.col("n").sum().alias(f"{prefix}n7"),
        pl.col("frp_sum").sum().alias(f"{prefix}frp7"),
        pl.col("frp_max").max().alias(f"{prefix}frpmax7"),
        pl.len().alias(f"{prefix}cells7"),
    ]
    if prefix == "":
        aggs += [
            pl.col("streak").filter(is_self).max().fill_null(0).alias("self_streak"),
            pl.col("streak").max().alias("streak7_max"),
            pl.col("dist").filter(other).min().alias("near_km"),
            # 확산 벡터: 활성 칸 → 후보 칸 방향(= offset 방향) 단위 벡터 × 감지 수 / 거리
            (pl.col("ux") / pl.col("dist") ** 2 * pl.col("n")).filter(other).sum().alias("sx"),
            (pl.col("uy") / pl.col("dist") ** 2 * pl.col("n")).filter(other).sum().alias("sy"),
            *[(pl.col("sector") == k).filter(other).sum().cast(pl.Int16).alias(f"dir_{d}") for k, d in enumerate(DIRS)],
        ]
    return e.group_by("date_local", "iy", "ix").agg(aggs)


def build_year(region: str, year: int, ok: set[date], adm: pl.DataFrame) -> dict:
    t0 = time.perf_counter()
    d0, d1 = max(date(year, 1, 1), START), min(date(year, 12, 31), END - timedelta(days=1))
    pts = load_points(region, d0 - timedelta(days=1), d1 + timedelta(days=1))
    cd = add_streak(cell_days(pts))
    act = cd.filter(pl.col("src"), pl.col("date_local").is_between(d0, d1))
    today = neighborhood(act, "")
    prev = neighborhood(cd.filter(pl.col("src")).with_columns(date_local=pl.col("date_local") + pl.duration(days=1))
                        .filter(pl.col("date_local").is_between(d0, d1)), "prev_")
    lab = cd.select(pl.col("date_local") - pl.duration(days=1), "iy", "ix", next_n="n")
    s = (today.join(prev, on=["date_local", "iy", "ix"], how="left")
         .join(lab, on=["date_local", "iy", "ix"], how="left")
         .with_columns(label=(pl.col("next_n").fill_null(0) > 0).cast(pl.Int8)))
    okd = pl.DataFrame({"d": sorted(ok)}, schema={"d": pl.Date})
    n_all = s.height
    # 후보 칸이 덮는 '내일 활성 칸' 비율 (반경 밖에서 새로 생긴 감지는 이 방법으로 예측할 수 없음 = 재현율 상한)
    nxt = (cd.filter(pl.col("src"), (pl.col("date_local") - pl.duration(days=1)).is_in(act["date_local"].unique().implode()))
           .select(pl.col("date_local") - pl.duration(days=1), "iy", "ix")
           .join(okd.rename({"d": "date_local"}), on="date_local", how="semi")
           .join(okd.select(date_local=pl.col("d") - pl.duration(days=1)), on="date_local", how="semi"))
    covered = nxt.join(today.select("date_local", "iy", "ix"), on=["date_local", "iy", "ix"], how="semi").height
    s = (s.join(okd.rename({"d": "date_local"}), on="date_local", how="semi")
         .join(okd.with_columns(date_local=pl.col("d") - pl.duration(days=1)).select("date_local"), on="date_local", how="semi")
         .join(okd.with_columns(date_local=pl.col("d") + pl.duration(days=1), prev_ok=pl.lit(True)).select("date_local", "prev_ok"),
               on="date_local", how="left")
         .with_columns(pl.col("prev_ok").fill_null(False)))
    lat = (pl.col("iy").cast(pl.Float64) + 0.5) * RES - 90
    lon = (pl.col("ix").cast(pl.Float64) + 0.5) * RES - 180
    m = pl.col("date_local").dt.month().cast(pl.Float64)
    s = s.with_columns(
        cell_lat=lat.cast(pl.Float32), cell_lon=lon.cast(pl.Float32),
        cell_id01=(((lat + 90) / 0.1).floor() * 3600 + ((lon + 180) / 0.1).floor()).cast(pl.Int64),
        month_sin=(2 * np.pi * m / 12).sin().cast(pl.Float32), month_cos=(2 * np.pi * m / 12).cos().cast(pl.Float32),
        region=pl.lit(region),
    ).join(adm.rename({"cell_id": "cell_id01", "adm0_a3": "cand_adm0"}), on="cell_id01", how="left").drop("cell_id01", "next_n")
    fill0 = [c for c in s.columns if c.startswith(("prev_",)) and c != "prev_ok"]
    s = s.with_columns(pl.col(fill0).fill_null(0)).sort("date_local", "iy", "ix")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s.write_parquet(sample_path(region, year), compression="zstd")
    return {"region": region, "year": year, "points": pts.height, "active_cell_days": act.height,
            "candidates_all": n_all, "candidates_ok": s.height, "positives": int(s["label"].sum()),
            "pos_rate": float(s["label"].mean()) if s.height else None,
            "active_today_pos_rate": float(s.filter(pl.col("self_n") > 0)["label"].mean()) if s.height else None,
            "next_active_cells": nxt.height, "next_active_covered": covered,
            "sec": round(time.perf_counter() - t0, 1), "peak_mem_gb": round(peak_memory_gb() or 0, 2)}


def samples(regions: list[str], years: list[int], force: bool) -> pl.DataFrame:
    ok, adm = ok_days(), cell_adm0()
    log = []
    jobs = [(r, y) for r in regions for y in years]
    for r, y in tqdm(jobs, desc="후보 표본", unit="year", dynamic_ncols=True):
        if sample_path(r, y).exists() and not force:
            tqdm.write(f"[있음] {sample_path(r, y).name} (다시 만들려면 --force)")
            continue
        info = build_year(r, y, ok, adm)
        log.append(info)
        tqdm.write(f"[{r} {y}] 후보 {info['candidates_ok']:,} (자료 상태 제외 전 {info['candidates_all']:,}), 양성 {info['positives']:,} "
                   f"({(info['pos_rate'] or 0) * 100:.2f}%), 내일 활성 칸 중 후보 안 {info['next_active_covered']:,}/{info['next_active_cells']:,}, "
                   f"{info['sec']}초, 최대 메모리 {info['peak_mem_gb']} GB")
    return pl.DataFrame(log)


def summary(regions: list[str]) -> pl.DataFrame:
    rows = []
    for r in regions:
        lf = pl.scan_parquet(OUT_DIR / f"samples_{r}_*.parquet")
        rows.append(lf.group_by("region", pl.col("date_local").dt.year().alias("year")).agg(
            candidates=pl.len(), positives=pl.col("label").sum(), active_days=pl.col("date_local").n_unique(),
            pos_rate=pl.col("label").mean(),
            pos_rate_active_self=pl.col("label").filter(pl.col("self_n") > 0).mean(),
            pos_rate_new=pl.col("label").filter(pl.col("self_n") == 0).mean(),
            share_self_active=(pl.col("self_n") > 0).mean(),
            share_home_country=(pl.col("cand_adm0") == ("KOR" if r == "KR" else "ARG")).mean(),
        ).collect())
    return pl.concat(rows).sort("region", "year")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="다음 날 확산 예측 (spread nowcast)")
    ap.add_argument("step", choices=["samples"])
    ap.add_argument("--regions", nargs="+", default=list(REGIONS))
    ap.add_argument("--years", nargs="+", type=int, default=list(range(START.year, END.year + 1)))
    ap.add_argument("--force", action="store_true", help="이미 있는 연도 파일도 다시 (기본: 있으면 건너뜀 = --resume)")
    ap.add_argument("--resume", action="store_true", help="기본 동작 (있는 연도 파일 건너뜀)")
    args = ap.parse_args(argv)
    start_run_log("spread_nowcast")
    t0 = time.perf_counter()
    if args.step == "samples":
        log = samples(args.regions, args.years, args.force)
        if log.height:
            log.write_csv(OUT_DIR / f"samples_build_log_{int(time.time())}.csv")
        summ = summary(args.regions)
        summ.write_csv(OUT_DIR / "samples_summary.csv")
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_formatting="ASCII_MARKDOWN",
                       tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, float_precision=4, thousands_separator=","):
            print(summ)
            tot = summ.group_by("region").agg(pl.col("candidates", "positives").sum()).with_columns(
                pos_rate=pl.col("positives") / pl.col("candidates"), neg_per_pos=(pl.col("candidates") - pl.col("positives")) / pl.col("positives"))
            print(tot)
    print(f"[완료] {args.step} {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
