"""위성별 '자료가 있는 날' 판정과 칸 × 월 집계.

1) satellite_day_status  (data/processed/satellite_day_status.parquet)
   위성 × UTC 날짜마다 자료가 정상인지 판정한다. 0과 결측을 구분하기 위한 표.
   - 판정 기준은 config.STATUS_BBOX(지금은 동아시아 상자 lon 100~145, lat 20~50) 안 감지 수 n_det_ea.
     2026-01~09 Archive Download(846xxx)는 동아시아만 받았고 나머지(연간 SP, drive zip NRT, 7일 NRT)는 전 세계라,
     전 세계 감지 수로 판정하면 커버리지가 바뀌는 날(2026-01-01, 2026-10-01 전후)에 가짜 low/missing이 생긴다.
     동아시아 상자는 모든 기간에 똑같이 덮이므로 이것으로 판정한다. 전 세계 감지 수(n_det_global)는 참고용이며
     그날 커버리지가 전 세계가 아니면 null.
   - 감지 수는 종류(type)와 상관없이 중복 제거 후 모든 감지 (자료 유무를 보는 값이므로 고정 열원도 포함).
   - status
       missing : n_det_ea == 0, 그날 덮는 파일이 없음, 또는 알려진 결측일(강제)
       low     : n_det_ea < 직전 30일(결측일 제외) 중앙값의 50%
       ok      : 그 외
     직전 30일에 유효한 날이 10일 미만이면(위성 운용 시작 직후) 중앙값 비교를 하지 않는다.
   - note: partial_known_outage(알려진 결측일인데 감지가 일부 있음), known_outage, partial_last_day(S-NPP 2026-11-02),
           coverage_change(파일 커버리지가 바뀐 날)
   - n_sats_low_same_day: 같은 날 기준선 대비 50% 미만인 다른 위성 수. 여러 위성이 함께 낮으면
     수신 장애보다 날씨(구름·비)나 실제 화재 활동 감소일 가능성이 크다 (판정에는 쓰지 않고 참고용).

2) cell_month  (data/processed/cell_month_0.1deg.parquet) — 분석 범위(config.ANALYSIS_BBOX, 기본 전 세계) 칸
   키: cell_id, cell_lat, cell_lon, year, month, satellite  (month는 현지 태양시 날짜 기준)
   값: n_det, n_det_conf, active_days, n_day, n_night, frp_sum, frp_max, source(NRT가 섞이면 NRT),
       obs_days(그 위성-월의 status=ok 일수), low_days, missing_days, days_in_month,
       month_status(complete / partial: 자료가 월말 전에 끝남 — 진행 중인 달 또는 S-NPP 운용 종료 달)
   감지가 없는 칸·월은 행이 없다. 그 칸·월이 '0'인지 '결측'인지는 obs_days로 판단한다
   (위성-월 단위 표 satellite_month_status.parquet도 같이 저장).

실행
    uv run python -m planetarium.monthly             # 상태표 + 칸×월
    uv run python -m planetarium.monthly --status-only
"""

from __future__ import annotations

import argparse
import calendar
import statistics
import time
from collections import deque
from datetime import date, timedelta

import polars as pl
from tqdm import tqdm

from planetarium.config import (ANALYSIS_BBOX, EAST_ASIA_BBOX, KNOWN_MISSING_DAYS, KNOWN_REFILLED_DAYS, PROCESSED_DATA_DIR,
                                STATUS_BBOX,
                                WORLD_BBOX, cell_day_dir)
from planetarium.dataset import SAT_DISPLAY, SATELLITES, sat_display, dedupe_plans, load_coverage, merged_files, _scan_one
from planetarium.runlog import start_run_log

STATUS_PATH = PROCESSED_DATA_DIR / "satellite_day_status.parquet"
SAT_MONTH_PATH = PROCESSED_DATA_DIR / "satellite_month_status.parquet"
CELL_MONTH_PATH = PROCESSED_DATA_DIR / "cell_month_0.1deg.parquet"

LOW_RATIO = 0.5
BASELINE_DAYS = 30
BASELINE_MIN_VALID = 10

# 위성 자료 시작일 (FIRMS 기준)
SAT_START = {
    "Terra": date(2000, 11, 1),
    "Aqua": date(2002, 7, 4),
    "S-NPP": date(2012, 1, 20),
    "NOAA-20": date(2018, 4, 1),
    "NOAA-21": date(2023, 3, 1),
}
# S-NPP 마지막 감지: 2026-11-02 15:16 UTC (데이터 전달 종료). 그날은 부분 일.
SNPP_LAST_DAY = date(2026, 11, 2)

LON0, LAT0, LON1, LAT1 = EAST_ASIA_BBOX  # 동아시아 상자 (참고 컬럼 n_det_ea·커버리지 구분용)
STATUS_BASIS = "global" if tuple(STATUS_BBOX) == tuple(WORLD_BBOX) else "bbox"


def in_bbox(bbox: tuple[float, float, float, float], lat: str = "latitude", lon: str = "longitude") -> pl.Expr:
    x0, y0, x1, y1 = bbox
    if (x0, y0, x1, y1) == (-180.0, -90.0, 180.0, 90.0):
        return pl.col(lat).is_not_null()  # 전 세계: 행마다 True (리터럴이면 집계에서 1번만 셈)
    return pl.col(lat).is_between(y0, y1) & pl.col(lon).is_between(x0, x1)


def in_east_asia(lat: str = "latitude", lon: str = "longitude") -> pl.Expr:
    """동아시아 상자 안."""
    return in_bbox(EAST_ASIA_BBOX, lat, lon)


def in_status_bbox(lat: str = "latitude", lon: str = "longitude") -> pl.Expr:
    """상태 판정 상자(config.STATUS_BBOX) 안."""
    return in_bbox(STATUS_BBOX, lat, lon)


def in_analysis(lat: str = "cell_lat", lon: str = "cell_lon") -> pl.Expr:
    """분석 범위(ANALYSIS_BBOX, 기본 전 세계) 안."""
    return in_bbox(ANALYSIS_BBOX, lat, lon)


# ── 1) 위성 × 날짜 상태 ───────────────────────────────────────────────────

def daily_counts() -> pl.DataFrame:
    """파일별(중복 제거 적용) 위성 × UTC 날짜 감지 수. 같은 날이 여러 파일에 걸치면 합친다."""
    plans = dedupe_plans(load_coverage())
    parts = []
    for p in tqdm(merged_files(), desc="일별 감지 수", unit="file", leave=False):
        parts.append(
            _scan_one(p, plans.get(p.name))
            .with_columns(satellite=sat_display())
            .group_by("satellite", pl.col("acq_date").alias("date_utc"))
            .agg(n_det_global=pl.len(), n_det_ea=in_east_asia().sum(), n_det_status=in_status_bbox().sum())
            .collect()
        )
    return (
        pl.concat(parts)
        .group_by("satellite", "date_utc")
        .agg(pl.col("n_det_global").sum(), pl.col("n_det_ea").sum(), pl.col("n_det_status").sum())
    )


def daily_coverage() -> dict[tuple[str, date], str]:
    """(위성, 날짜) -> 'global' / 'east_asia'. 그날을 덮는 파일 중 하나라도 전 세계면 global."""
    cov = load_coverage().with_columns(
        coverage=pl.when((pl.col("lon_min") <= -180) & (pl.col("lon_max") >= 180)).then(pl.lit("global"))
        .when((pl.col("lon_min") <= LON0) & (pl.col("lon_max") >= LON1)
              & (pl.col("lat_min") <= LAT0) & (pl.col("lat_max") >= LAT1)).then(pl.lit("east_asia"))
        .otherwise(pl.lit("partial_region")),
    )
    out: dict[tuple[str, date], str] = {}
    rank = {"global": 2, "east_asia": 1, "partial_region": 0}
    for r in cov.iter_rows(named=True):
        d, end = r["t_start"].date(), r["t_end"].date()
        while d <= end:
            k = (SAT_DISPLAY.get(r["satellite"], r["satellite"]), d)
            if k not in out or rank[r["coverage"]] > rank[out[k]]:
                out[k] = r["coverage"]
            d += timedelta(days=1)
    return out


SEASONAL_HALF_WINDOW = 15
SEASONAL_MIN_YEARS = 2


def seasonal_baseline(st: pl.DataFrame) -> pl.DataFrame:
    """위성 × 날짜마다 다른 연도 같은 날짜 ±15일(해당 연도 제외, 결측일 제외)의 판정 감지 수 중앙값."""
    v = st.filter(~pl.col("status").is_in(["missing", "partial"])).select(
        "satellite", y2=pl.col("date_utc").dt.year(), doy2=pl.col("date_utc").dt.ordinal_day(), e2=pl.col("n_det_status"))
    t = st.select("satellite", "date_utc", y=pl.col("date_utc").dt.year(), doy=pl.col("date_utc").dt.ordinal_day())
    j = t.join(v, on="satellite").filter(pl.col("y2") != pl.col("y"))
    dd = (pl.col("doy") - pl.col("doy2")).abs()
    j = j.filter(pl.min_horizontal(dd, 366 - dd) <= SEASONAL_HALF_WINDOW)  # 연말·연초는 이어서
    return j.group_by("satellite", "date_utc").agg(
        seasonal_baseline=pl.col("e2").median(), seasonal_years=pl.col("y2").n_unique()).with_columns(
        seasonal_baseline=pl.when(pl.col("seasonal_years") >= SEASONAL_MIN_YEARS).then(pl.col("seasonal_baseline")))


def build_satellite_day_status() -> pl.DataFrame:
    counts = daily_counts()
    cov = daily_coverage()
    last_day = counts.group_by("satellite").agg(last=pl.col("date_utc").max())
    last = dict(last_day.iter_rows())
    last["S-NPP"] = min(last.get("S-NPP", SNPP_LAST_DAY), SNPP_LAST_DAY)
    # 받는 중인 날: 운용 중인 위성(자료가 전체 최신일 전날 이후까지 있음)의 마지막 관측일.
    # 감지가 아직 다 들어오지 않았으므로 low/missing 판정에서 빼고 status=partial, note=partial_current_day.
    # 다음 갱신 때 그날 뒤로 자료가 더 들어오면 일반 규칙으로 다시 판정된다.
    latest = max(last.values())
    current_day = {s: d for s, d in last.items() if s != "S-NPP" and d >= latest - timedelta(days=1)}
    cnt = {(s, d): (g, e, j) for s, d, g, e, j in
           counts.select("satellite", "date_utc", "n_det_global", "n_det_ea", "n_det_status").iter_rows()}
    known = {(s, d) for s, days in KNOWN_MISSING_DAYS.items() for d in days}
    refilled = {(s, d) for s, days in KNOWN_REFILLED_DAYS.items() for d in days}

    rows = []
    for sat in SATELLITES:
        d, end = SAT_START[sat], last[sat]
        window: deque[tuple[date, int]] = deque()  # 직전 30일의 유효(결측 아님) 감지 수
        prev_cov = None
        while d <= end:
            g, ea, e = cnt.get((sat, d), (0, 0, 0))  # e: 판정에 쓰는 감지 수 (STATUS_BBOX 안)
            coverage = cov.get((sat, d), "none")
            while window and window[0][0] < d - timedelta(days=BASELINE_DAYS):
                window.popleft()
            baseline = statistics.median(v for _, v in window) if len(window) >= BASELINE_MIN_VALID else None
            note = None
            if coverage == "none" or e == 0:
                status = "missing"
            elif baseline is not None and e < LOW_RATIO * baseline:
                status = "low"
            else:
                status = "ok"
            if current_day.get(sat) == d:
                status, note = "partial", "partial_current_day"
            elif (sat, d) in known:
                note = "partial_known_outage" if e > 0 else "known_outage"
                status = "missing"
            elif (sat, d) in refilled:
                note = "earlier_nrt_gap_refilled"  # 강제 missing 해제, 일반 규칙으로 판정
            elif sat == "S-NPP" and d == SNPP_LAST_DAY:
                note = "partial_last_day"
            elif prev_cov is not None and coverage != prev_cov:
                note = "coverage_change"
            if status not in ("missing", "partial"):
                window.append((d, e))
            rows.append({
                "satellite": sat, "date_utc": d, "n_det_ea": ea, "n_det_status": e,
                "n_det_global": g if coverage == "global" else None,
                "coverage": coverage, "status_basis": STATUS_BASIS, "baseline": baseline,
                "ratio_to_baseline": (e / baseline) if baseline else None,
                "status": status, "note": note,
            })
            prev_cov = coverage
            d += timedelta(days=1)

    st = pl.DataFrame(rows, schema={
        "satellite": pl.String, "date_utc": pl.Date, "n_det_ea": pl.Int64, "n_det_status": pl.Int64, "n_det_global": pl.Int64,
        "coverage": pl.String, "status_basis": pl.String, "baseline": pl.Float64, "ratio_to_baseline": pl.Float64,
        "status": pl.String, "note": pl.String,
    })
    # 같은 날 다른 위성도 기준선의 50% 미만이었는지
    # (결측인 위성은 '함께 낮음'으로 세지 않는다 — 다른 위성의 수신 장애가 날씨처럼 보이지 않도록)
    valid = ~pl.col("status").is_in(["missing", "partial"]) & pl.col("ratio_to_baseline").is_not_null()
    lowish = (valid & (pl.col("ratio_to_baseline") < LOW_RATIO)).cast(pl.Int8)
    st = st.with_columns(_low=lowish, _valid=valid.cast(pl.Int8)).with_columns(
        n_sats_low_same_day=(pl.col("_low").sum().over("date_utc") - pl.col("_low")).cast(pl.Int8),
        n_other_sats=(pl.col("_valid").sum().over("date_utc") - pl.col("_valid")).cast(pl.Int8),
    ).drop("_low", "_valid")
    # 최종 status: 동아시아 감지 수는 날씨(구름·비)와 실제 화재 활동에 따라 하루에도 크게 변한다.
    # 검증 결과 VIIRS low일의 91~99%, MODIS low일의 62%가 다른 위성도 같은 날 함께 낮았다(= 자료 문제가 아님).
    # 그래서 '그 위성만 떨어진 날'만 low로 두고, 다른 위성도 함께 낮은 날은 ok + note=low_shared_with_other_sats.
    # 다른 위성이 없는 날(2000-11~2002-07 Terra 단독)은 원래 규칙 그대로. 원래 규칙 결과는 status_raw에 남긴다.
    # 비교할 다른 위성이 없는 날(2000-11~2002-07 Terra 단독 등)은 감소가 자료 문제인지 확인할 수 없다.
    # 예: 2001-05 Terra는 봄 화재 철(3~4월) 직후 감소로 24일이 low가 되어 월 전체가 결측처럼 보였다.
    # 이런 날은 ok + note=low_unverified_single_sat로 두고, 감지 0인 날만 missing으로 본다.
    #
    # 전 세계 기준(STATUS_BASIS == "global")에서는 이 예외를 끈다: 전 세계 감지 수는 날씨로 50%씩 떨어지지 않으므로
    # 여러 VIIRS 위성이 함께 떨어진 날(예: 2026-10-12~13)은 수신 장애로 본다 → 위성과 무관하게 50% 미만이면 low.
    # 비교용으로 예외를 적용했을 때의 결과(r1 규칙)를 status_r1에 남긴다.
    is_low = pl.col("status") == "low"
    shared = is_low & (pl.col("n_other_sats") > 0) & (pl.col("n_sats_low_same_day") > 0)
    single = is_low & (pl.col("n_other_sats") == 0)
    refilled_expr = pl.struct("satellite", "date_utc").is_in(
        [{"satellite": s, "date_utc": d} for s, d in refilled]) if refilled else pl.lit(False)
    st = st.with_columns(
        status_raw=pl.col("status"),
        status_r1=pl.when(refilled_expr).then(pl.lit("missing"))
        .when(shared | single).then(pl.lit("ok")).otherwise(pl.col("status")),
    )
    if STATUS_BASIS != "global":
        st = st.with_columns(
            status=pl.when(shared | single).then(pl.lit("ok")).otherwise(pl.col("status")),
            note=pl.when(shared).then(pl.lit("low_shared_with_other_sats"))
            .when(single).then(pl.lit("low_unverified_single_sat")).otherwise(pl.col("note")),
        )
    else:
        st = st.with_columns(note=pl.when(pl.col("note").is_null() & shared).then(pl.lit("low_multi_sat_reception_outage"))
                             .otherwise(pl.col("note")))
    # 계절 기준(r3): 모든 위성이 함께 떨어진 날(예: 2019-09-26~10-07)은 남반구 화재 철이 끝나는 등 계절 감소일 수 있다.
    # 그 위성의 '다른 연도 같은 날짜 ±15일' 감지 수 중앙값(해당 연도 제외, 결측일 제외)과도 비교해
    # 직전 30일 기준과 계절 기준 둘 다 50% 미만일 때만 low. 계절 기준으로 정상이면 ok + note=seasonal_decline.
    # 다른 연도가 SEASONAL_MIN_YEARS 미만이면(운용 초기) 계절 기준을 쓰지 않는다.
    st = st.join(seasonal_baseline(st), on=["satellite", "date_utc"], how="left").with_columns(
        ratio_to_seasonal=pl.when(pl.col("seasonal_baseline") > 0).then(pl.col("n_det_status") / pl.col("seasonal_baseline")))
    seasonal_ok = (pl.col("status") == "low") & (pl.col("ratio_to_seasonal") >= LOW_RATIO)
    st = st.with_columns(
        status_r2=pl.col("status"),
        note=pl.when(seasonal_ok).then(pl.lit("seasonal_decline")).otherwise(pl.col("note")),
        status=pl.when(seasonal_ok).then(pl.lit("ok")).otherwise(pl.col("status")),
    )
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    st.write_parquet(STATUS_PATH)
    return st


# ── 2) 칸 × 월 ────────────────────────────────────────────────────────────

def build_satellite_month_status(st: pl.DataFrame) -> pl.DataFrame:
    sm = (
        st.with_columns(year=pl.col("date_utc").dt.year().cast(pl.Int16), month=pl.col("date_utc").dt.month().cast(pl.Int8))
        .group_by("satellite", "year", "month")
        .agg(
            obs_days=(pl.col("status") == "ok").sum().cast(pl.Int16),
            low_days=(pl.col("status") == "low").sum().cast(pl.Int16),
            missing_days=(pl.col("status") == "missing").sum().cast(pl.Int16),
            first_day=pl.col("date_utc").min(),
            last_day=pl.col("date_utc").max(),
            n_det_ea=pl.col("n_det_ea").sum(),
        )
    )
    days_in = [calendar.monthrange(y, m)[1] for y, m in sm.select("year", "month").iter_rows()]
    sm = sm.with_columns(days_in_month=pl.Series(days_in, dtype=pl.Int16))
    month_end = pl.date(pl.col("year"), pl.col("month"), pl.col("days_in_month"))
    month_start = pl.date(pl.col("year"), pl.col("month"), 1)
    snpp_end = (pl.col("satellite") == "S-NPP") & (pl.col("last_day") == SNPP_LAST_DAY)
    sm = sm.with_columns(
        # partial: 자료가 월말 전에 끝남(진행 중인 달·S-NPP 종료) 또는 운용 시작 달이 월초가 아님
        # S-NPP 마지막 날(11-02)은 부분 일이므로 그 달도 partial
        month_status=pl.when((pl.col("last_day") < month_end) | (pl.col("first_day") > month_start) | snpp_end)
        .then(pl.lit("partial")).otherwise(pl.lit("complete")),
    ).sort("satellite", "year", "month")
    sm.write_parquet(SAT_MONTH_PATH)
    return sm


def build_cell_month(sm: pl.DataFrame, res: float = 0.1) -> pl.DataFrame:
    """분석 범위(ANALYSIS_BBOX) 칸 × 월 × 위성. 전 세계는 크므로 현지 날짜 연도별로 나눠 집계한다."""
    lf = pl.scan_parquet(cell_day_dir(res) / "*.parquet").filter(in_analysis("cell_lat", "cell_lon"))
    years = lf.select(pl.col("year").unique()).collect()["year"].sort().to_list()
    smj = sm.select("satellite", "year", "month", "obs_days", "low_days", "missing_days", "days_in_month", "month_status")
    tmp_dir = CELL_MONTH_PATH.with_suffix(".parts")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for y in tqdm(years, desc="칸×월", unit="year", leave=False):
        # 같은 칸·날짜가 여러 파일에 걸칠 수 있다 (예: UTC 2025-12-31 저녁 → 현지 2026-01-01) → 날짜 단위로 먼저 합침
        part = (
            lf.filter(pl.col("year") == y)
            .group_by("cell_id", "cell_lat", "cell_lon", "date_local", "year", "month", "satellite")
            .agg(pl.col("n_det", "n_det_conf", "n_day", "n_night", "frp_sum").sum(), pl.col("frp_max").max(),
                 any_nrt=(pl.col("source") == "NRT").any())
            .group_by("cell_id", "cell_lat", "cell_lon", "year", "month", "satellite")
            .agg(
                pl.col("n_det", "n_det_conf").sum(),
                active_days=(pl.col("n_det") > 0).sum().cast(pl.Int16),
                n_day=pl.col("n_day").sum(), n_night=pl.col("n_night").sum(),
                frp_sum=pl.col("frp_sum").sum(), frp_max=pl.col("frp_max").max(),
                source=pl.when(pl.col("any_nrt").any()).then(pl.lit("NRT")).otherwise(pl.lit("SP")),
            )
            .collect()
            .join(smj, on=["satellite", "year", "month"], how="left")
            .sort("cell_id", "year", "month", "satellite")
        )
        part.write_parquet(tmp_dir / f"{y}.parquet", compression="zstd")
        total += part.height
    cm = pl.scan_parquet(sorted(tmp_dir.glob("*.parquet")))
    cm.sink_parquet(CELL_MONTH_PATH, compression="zstd")
    for f in tmp_dir.glob("*.parquet"):
        f.unlink()
    tmp_dir.rmdir()
    return pl.scan_parquet(CELL_MONTH_PATH).select(pl.len(), pl.col("cell_id").n_unique()).collect()


def load_cell_month() -> pl.LazyFrame:
    return pl.scan_parquet(CELL_MONTH_PATH)


def print_table(df: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_formatting="ASCII_MARKDOWN",
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, thousands_separator=","):
        print(df)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="위성별 자료 상태와 칸 × 월 집계 (동아시아)")
    ap.add_argument("--status-only", action="store_true", help="satellite_day_status만 다시 만듦")
    args = ap.parse_args(argv)
    start_run_log("monthly")

    t0 = time.perf_counter()
    st = build_satellite_day_status()
    print(f"[완료] {STATUS_PATH.name}: {st.height:,}행, {time.perf_counter() - t0:,.0f}초")
    print_table(
        st.group_by("satellite").agg(
            first=pl.col("date_utc").min(), last=pl.col("date_utc").max(), days=pl.len(),
            ok=(pl.col("status") == "ok").sum(), low=(pl.col("status") == "low").sum(),
            missing=(pl.col("status") == "missing").sum(),
        ).sort(pl.col("satellite").replace_strict({k: i for i, k in enumerate(SATELLITES)}))
    )
    print("[메모가 있는 날]")
    print_table(st.filter(pl.col("note").is_not_null()).select(
        "satellite", "date_utc", "n_det_ea", "n_det_global", "coverage", "ratio_to_baseline", "status", "note"))
    sm = build_satellite_month_status(st)
    print(f"[완료] {SAT_MONTH_PATH.name}: {sm.height:,}행")
    if args.status_only:
        return
    t0 = time.perf_counter()
    n = build_cell_month(sm)
    print(f"[완료] {CELL_MONTH_PATH.name}: {n.item(0, 0):,}행 (칸 {n.item(0, 1):,}개), 범위 {ANALYSIS_BBOX}, "
          f"{time.perf_counter() - t0:,.0f}초")
    print_table(pl.scan_parquet(CELL_MONTH_PATH).group_by("satellite").agg(
        rows=pl.len(), n_det=pl.col("n_det").sum(), first=pl.col("year").min(), last=pl.col("year").max()).collect())


if __name__ == "__main__":
    main()
