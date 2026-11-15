"""캐나다 산불 기상 지수(FWI System) 일별 계산 — NASA POWER(MERRA-2, 0.5°) 기상, 한국 상자·아르헨티나.

계산식: Van Wagner (1987) "Development and structure of the Canadian Forest Fire Weather Index System",
        Forestry Technical Report 35 (cffdrs R 패키지와 같은 식, 위도별 일장 보정 포함).
        FFMC(미세 연료 수분), DMC(부식층 수분), DC(가뭄), ISI(초기 확산), BUI(연소 가능 연료), FWI(종합).

근사 (원래 입력은 현지 정오 관측; POWER는 일 통계만 준다)
  기온   = T2M_MAX (일 최고)          — 정오(13시 LST) 기온 대신. 정오보다 약간 높아 FFMC·DMC·DC가 조금 높게 나온다.
  습도   = RH2M_MIN_EST (일 최저 근사) — 정오 습도 대신. 이슬점·최고 기온으로 Magnus 식 근사(weather_power.py).
  풍속   = WS10M (일평균, m/s × 3.6 = km/h) — 정오 10 m 풍속 대신. 정오보다 낮을 수 있어 ISI·FWI가 조금 낮게 나온다.
  강수   = PRECTOTCORR (그날 0~24시 LST 합) — 원래는 전날 정오~오늘 정오 24시간.
  시작값 = FFMC 85, DMC 6, DC 15 (2012-01-01), 이후 겨울에도 멈추지 않고 이어서 계산 (눈 덮임·월동 보정 없음).
           DC는 수개월 이상 걸려 안정되므로 2012년 상반기 값은 참고용.
위도별 일장 보정: DMC Le — 33°N 이상 표준표, 10~30°N, -10~-30°, -30° 이하, ±10° 안 = 9 (cffdrs와 같음);
                  DC Lf — 20°N 이상 / -20° 이하 / 그 사이 1.4.
위험 등급(캐나다 전국 기준, 지역 기준 아님): low < 5.2 ≤ moderate < 11.2 ≤ high < 21.3 ≤ very_high < 38 ≤ extreme.

산출물
  data/processed/fwi_cell_day.parquet  region, glat, glon, date, FFMC, DMC, DC, ISI, BUI, FWI, fwi_class, fwi_pct, fwi_rel_class
  data/processed/fwi_aoi_day.parquet   AOI × 날짜 (KR·시도·시군구, ADM0_ARG·1급): fwi_mean/max, ffmc_mean, dc_mean, fwi_class(평균 기준),
                                       fwi_pct, fwi_rel_class
  data/processed/fwi_clim_{cell,aoi}_month.parquet  칸(AOI) × 달 경계값 q25·q50·q75·q90 (2012~2025) — 새 날짜의 상대 등급 판정용
  reports/fwi_validation.csv           FWI 등급별 '다음 날 새 감지' 발생률 (0.5° 칸, S-NPP, d·d+1 자료 정상인 날)
  reports/fwi_validation_relative.csv  상대 등급별 같은 표, AUC는 백분위로

상대 등급(fwi_rel_class): 같은 칸(AOI)·같은 달의 2012~2025 일 FWI 분포에서 그날 값의 백분위(fwi_pct, 중간 순위)
  below_normal < 25 ≤ normal < 75 ≤ above_normal < 90 ≤ well_above_normal.
  2012년 초 DC 안정화 기간도 기후값에 들어간다. 기후 기간 밖의 날짜는 fwi_clim_* 경계값으로 등급만 판정.

실행  uv run python -m planetarium.fwi compute|aoi|validate|all
"""

from __future__ import annotations

import argparse
import time
from datetime import timedelta

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium.config import PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log
from planetarium.weather_power import DAILY_PATH, GRID

FWI_PATH = PROCESSED_DATA_DIR / "fwi_cell_day.parquet"
FWI_AOI_PATH = PROCESSED_DATA_DIR / "fwi_aoi_day.parquet"
CLASSES = [(5.2, "low"), (11.2, "moderate"), (21.3, "high"), (38.0, "very_high"), (np.inf, "extreme")]
ELL = {1: [6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6], 2: [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],
       3: [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2], 4: [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10, 11.2, 11.8]}
FL01 = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5, 2.4, 0.4, -1.6, -1.6]
FL02 = [6.4, 5, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8]


def day_length(lat: np.ndarray, month: int) -> tuple[np.ndarray, np.ndarray]:
    m = month - 1
    le = np.select([lat > 30, (lat <= 30) & (lat > 10), (lat <= 10) & (lat > -10), (lat <= -10) & (lat > -30)],
                   [ELL[1][m], ELL[2][m], 9.0, ELL[3][m]], ELL[4][m])
    lf = np.select([lat > 20, lat <= -20], [FL01[m], FL02[m]], 1.4)
    return le, lf


def ffmc_step(f0, T, H, W, ro):
    mo = 147.2 * (101 - f0) / (59.5 + f0)
    rf = np.maximum(ro - 0.5, 1e-9)
    wet = ro > 0.5
    add = 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf))
    mo = np.where(wet, np.where(mo > 150, mo + add + 0.0015 * (mo - 150) ** 2 * np.sqrt(rf), mo + add), mo)
    mo = np.minimum(mo, 250)
    ed = 0.942 * H ** 0.679 + 11 * np.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H))
    ew = 0.618 * H ** 0.753 + 10 * np.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H))
    ko = 0.424 * (1 - (H / 100) ** 1.7) + 0.0694 * np.sqrt(W) * (1 - (H / 100) ** 8)
    kd = ko * 0.581 * np.exp(0.0365 * T)
    k1 = 0.424 * (1 - ((100 - H) / 100) ** 1.7) + 0.0694 * np.sqrt(W) * (1 - ((100 - H) / 100) ** 8)
    kw = k1 * 0.581 * np.exp(0.0365 * T)
    m = np.where(mo > ed, ed + (mo - ed) * 10 ** (-kd), np.where(mo < ew, ew - (ew - mo) * 10 ** (-kw), mo))
    return np.clip(59.5 * (250 - m) / (147.2 + m), 0, 101)


def dmc_step(p0, T, H, ro, le):
    T = np.maximum(T, -1.1)
    rk = 1.894 * (T + 1.1) * (100 - H) * le * 1e-4
    rw = 0.92 * ro - 1.27
    wmi = 20 + 280 / np.exp(0.023 * p0)
    lp = np.log(np.maximum(p0, 1e-9))
    b = np.where(p0 <= 33, 100 / (0.5 + 0.3 * p0), np.where(p0 <= 65, 14 - 1.3 * lp, 6.2 * lp - 17.2))
    wmr = wmi + 1000 * rw / (48.77 + b * rw)
    pr = np.maximum(43.43 * (5.6348 - np.log(np.maximum(wmr - 20, 1e-9))), 0)
    p = np.where(ro > 1.5, pr, p0)
    return np.maximum(p + rk, 0)


def dc_step(d0, T, ro, lf):
    T = np.maximum(T, -2.8)
    pe = np.maximum((0.36 * (T + 2.8) + lf) / 2, 0)
    rw = 0.83 * ro - 1.27
    smi = 800 * np.exp(-d0 / 400)
    dr = np.maximum(d0 - 400 * np.log(1 + 3.937 * np.maximum(rw, 0) / smi), 0)
    return np.where(ro > 2.8, dr, d0) + pe


def isi_bui_fwi(ffmc, dmc, dc, W):
    fm = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    isi = 0.208 * np.exp(0.05039 * W) * 91.9 * np.exp(-0.1386 * fm) * (1 + fm ** 5.31 / 4.93e7)
    s = dmc + 0.4 * dc
    with np.errstate(divide="ignore", invalid="ignore"):
        bui = np.where(dmc <= 0.4 * dc, 0.8 * dmc * dc / s, dmc - (1 - 0.8 * dc / s) * (0.92 + (0.0114 * dmc) ** 1.7))
    bui = np.where(s > 0, np.maximum(bui, 0), 0)
    fd = np.where(bui <= 80, 0.626 * bui ** 0.809 + 2, 1000 / (25 + 108.64 * np.exp(-0.023 * bui)))
    B = 0.1 * isi * fd
    with np.errstate(divide="ignore", invalid="ignore"):
        fwi = np.where(B > 1, np.exp(2.72 * (0.434 * np.log(np.maximum(B, 1e-9))) ** 0.647), B)
    return isi, bui, fwi


def fwi_class(col: str = "FWI") -> pl.Expr:
    e = pl.when(pl.col(col) < CLASSES[0][0]).then(pl.lit(CLASSES[0][1]))
    for th, name in CLASSES[1:-1]:
        e = e.when(pl.col(col) < th).then(pl.lit(name))
    return e.otherwise(pl.lit(CLASSES[-1][1]))


REL_CLASSES = [(25.0, "below_normal"), (75.0, "normal"), (90.0, "above_normal"), (np.inf, "well_above_normal")]
CLIM_YEARS = (2012, 2025)
CLIM_CELL_PATH = PROCESSED_DATA_DIR / "fwi_clim_cell_month.parquet"
CLIM_AOI_PATH = PROCESSED_DATA_DIR / "fwi_clim_aoi_month.parquet"


def fwi_rel_class(col: str = "fwi_pct") -> pl.Expr:
    """상대 등급: 같은 칸(또는 AOI)·같은 달의 2012~2025 FWI 분포에서의 백분위."""
    e = pl.when(pl.col(col).is_null()).then(pl.lit(None, dtype=pl.String))
    for th, name in REL_CLASSES[:-1]:
        e = e.when(pl.col(col) < th).then(pl.lit(name))
    return e.otherwise(pl.lit(REL_CLASSES[-1][1]))


def add_relative(df: pl.DataFrame, keys: list[str], col: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """keys·달별로 col의 백분위(중간 순위, 0~100)를 붙이고 fwi_rel_class를 단다.
    기후값 = CLIM_YEARS 안의 날만 (그 밖의 날은 아래 경계표로 판정). 경계표(q25·q75·q90)는 새 날짜용으로 따로 돌려준다."""
    g = keys + ["month"]
    df = df.with_columns(month=pl.col("date").dt.month().cast(pl.Int8))
    clim = df.filter(pl.col("date").dt.year().is_between(*CLIM_YEARS), pl.col(col).is_not_null())
    n = pl.len().over(g)
    ranked = clim.select(*keys, "date", fwi_pct=((pl.col(col).rank("average").over(g) - 0.5) / n * 100).cast(pl.Float32))
    bounds = clim.group_by(g).agg(n_days=pl.len().cast(pl.Int32),
                                  **{f"q{q}": pl.col(col).quantile(q / 100, "linear").cast(pl.Float32) for q in (25, 50, 75, 90)})
    out = df.join(ranked, on=keys + ["date"], how="left")
    # 기후 기간 밖(예: 최신 날짜)은 경계표로 등급만 판정 (백분위는 없음)
    out = out.join(bounds.select(*g, "q25", "q75", "q90"), on=g, how="left")
    out = out.with_columns(fwi_rel_class=pl.when(pl.col("fwi_pct").is_not_null()).then(fwi_rel_class())
                           .when(pl.col(col).is_null() | pl.col("q25").is_null()).then(pl.lit(None, dtype=pl.String))
                           .when(pl.col(col) < pl.col("q25")).then(pl.lit("below_normal"))
                           .when(pl.col(col) < pl.col("q75")).then(pl.lit("normal"))
                           .when(pl.col(col) < pl.col("q90")).then(pl.lit("above_normal"))
                           .otherwise(pl.lit("well_above_normal"))).drop("month", "q25", "q75", "q90")
    return out, bounds.sort(g)


def compute(w: pl.DataFrame | None = None, start: tuple | None = None) -> pl.DataFrame:
    """w: region, glat, glon, date, T2M_MAX, RH2M_MIN_EST, WS10M, PRECTOTCORR (지점마다 연속 날짜). start: (ffmc, dmc, dc) 배열 또는 None."""
    w = (w if w is not None else pl.read_parquet(DAILY_PATH)).sort("region", "glat", "glon", "date")
    pts = w.select("region", "glat", "glon").unique(maintain_order=True)
    days = w["date"].unique().sort()
    key = pts.with_row_index("pi")
    w = w.join(key, on=["region", "glat", "glon"])
    n, nd = pts.height, days.len()
    di = {d: i for i, d in enumerate(days.to_list())}
    idx_d = np.fromiter((di[d] for d in w["date"].to_list()), dtype=np.int64, count=w.height)
    idx_p = w["pi"].to_numpy()

    def grid(col):
        a = np.full((nd, n), np.nan, dtype=np.float64)
        a[idx_d, idx_p] = w[col].to_numpy().astype(np.float64)
        return a
    T, H, W, R = grid("T2M_MAX"), np.clip(grid("RH2M_MIN_EST"), 0, 100), grid("WS10M") * 3.6, np.maximum(grid("PRECTOTCORR"), 0)
    lat = pts["glat"].to_numpy()
    f, p, d = (np.full(n, 85.0), np.full(n, 6.0), np.full(n, 15.0)) if start is None else start
    out = {k: np.full((nd, n), np.nan) for k in ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI")}
    months = [x.month for x in days.to_list()]
    for i in tqdm(range(nd), desc="FWI 일별", unit="day", dynamic_ncols=True):
        ok = ~(np.isnan(T[i]) | np.isnan(H[i]) | np.isnan(W[i]) | np.isnan(R[i]))
        le, lf = day_length(lat, months[i])
        f2 = ffmc_step(f, T[i], H[i], W[i], R[i])
        p2 = dmc_step(p, T[i], H[i], R[i], le)
        d2 = dc_step(d, T[i], R[i], lf)
        f, p, d = np.where(ok, f2, f), np.where(ok, p2, p), np.where(ok, d2, d)  # 결측일은 전날 값 유지
        isi, bui, fwi = isi_bui_fwi(f, p, d, np.where(ok, W[i], 0))
        for k, v in (("FFMC", f), ("DMC", p), ("DC", d), ("ISI", isi), ("BUI", bui), ("FWI", fwi)):
            out[k][i] = np.where(ok, v, np.nan)
    res = pl.DataFrame({
        "pi": np.tile(np.arange(n), nd), "date": np.repeat(days.to_numpy(), n),
        **{k: v.reshape(-1).astype(np.float32) for k, v in out.items()},
    }).join(key, on="pi").drop("pi")
    return res.select("region", "glat", "glon", "date", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI").with_columns(fwi_class=fwi_class()).sort(
        "region", "glat", "glon", "date")


def apply_clim(df: pl.DataFrame, clim: pl.DataFrame, keys: list[str], col: str) -> pl.DataFrame:
    """기후 기간 밖 날짜: 경계값(q25·q75·q90)으로 fwi_rel_class만 정하고 fwi_pct는 비운다."""
    g = keys + ["month"]
    out = df.with_columns(month=pl.col("date").dt.month().cast(pl.Int8)).join(clim.select(*g, "q25", "q75", "q90"), on=g, how="left")
    return out.with_columns(
        fwi_pct=pl.lit(None, dtype=pl.Float32),
        fwi_rel_class=pl.when(pl.col(col).is_null() | pl.col("q25").is_null()).then(pl.lit(None, dtype=pl.String))
        .when(pl.col(col) < pl.col("q25")).then(pl.lit("below_normal"))
        .when(pl.col(col) < pl.col("q75")).then(pl.lit("normal"))
        .when(pl.col(col) < pl.col("q90")).then(pl.lit("above_normal"))
        .otherwise(pl.lit("well_above_normal"))).drop("month", "q25", "q75", "q90")


def extend(fetch_date: str, skip_fetch: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """2026-01-01 ~ fetch_date를 NASA POWER에서 받아 2025-12-31 상태(FFMC·DMC·DC)에서 이어 계산.
    기존 파일의 2026년 이후 행은 바꿔 끼운다 (2012~2025 행·경계값은 그대로)."""
    from planetarium.weather_power import combine, fetch, recent_dir
    cache = recent_dir(fetch_date)
    w_path = cache / "power_daily_2026.parquet"
    if not skip_fetch:
        fetch(cache, "20260101", fetch_date)
    w = combine(cache, w_path)
    old = pl.read_parquet(FWI_PATH)
    hist = old.filter(pl.col("date").dt.year() < 2026)
    state = (hist.filter(pl.col("date") == pl.date(2025, 12, 31)).select("region", "glat", "glon", "FFMC", "DMC", "DC"))
    pts = w.select("region", "glat", "glon").unique().sort("region", "glat", "glon")
    st = pts.join(state, on=["region", "glat", "glon"], how="left")
    miss = st.filter(pl.col("FFMC").is_null()).height
    if miss:
        print(f"[주의] 2025-12-31 상태가 없는 지점 {miss}개 → 시작값 85/6/15")
    st = st.with_columns(pl.col("FFMC").fill_null(85.0), pl.col("DMC").fill_null(6.0), pl.col("DC").fill_null(15.0))
    start = tuple(st[c].to_numpy().astype(np.float64) for c in ("FFMC", "DMC", "DC"))
    new = compute(w, start=start).filter(pl.col("date") >= pl.date(2026, 1, 1))
    # 끝부분 결측(POWER 근실시간 지연)은 행을 두지 않음
    new = new.filter(pl.col("FWI").is_not_null() & pl.col("FWI").is_not_nan())
    new = apply_clim(new, pl.read_parquet(CLIM_CELL_PATH), ["region", "glat", "glon"], "FWI")
    cell = pl.concat([hist, new.select(hist.columns)]).sort("region", "glat", "glon", "date")
    a_old = pl.read_parquet(FWI_AOI_PATH)
    a_hist = a_old.filter(pl.col("date").dt.year() < 2026)
    a_new = apply_clim(aoi_day(new), pl.read_parquet(CLIM_AOI_PATH), ["aoi_id"], "fwi_mean")
    aoi = pl.concat([a_hist, a_new.select(a_hist.columns)]).sort("aoi_id", "date")
    return cell, aoi


def aoi_day(fwi: pl.DataFrame) -> pl.DataFrame:
    """AOI × 날짜: 0.1° 칸 × AOI 면적 가중치를 0.5° 칸으로 묶어 가중 평균 (작은 시군구도 값이 생김)."""
    from planetarium.harmonize import aoi_membership
    mem, aoi = aoi_membership()
    keep = aoi.filter(pl.col("level").is_in(["kr", "sido", "sgg"]) | (pl.col("aoi_id") == "ADM0_ARG")
                      | ((pl.col("level") == "adm1") & (pl.col("parent_aoi") == "ADM0_ARG")))
    m = mem.join(keep.select("aoi_id"), on="aoi_id", how="semi").with_columns(
        lat=(pl.col("cell_id") // 3600 + 0.5) * 0.1 - 90, lon=(pl.col("cell_id") % 3600 + 0.5) * 0.1 - 180)
    m = m.with_columns(glat=((pl.col("lat") / GRID).floor() * GRID + GRID / 2).round(3),
                       glon=((pl.col("lon") / GRID).floor() * GRID + GRID / 2).round(3),
                       region=pl.when(pl.col("aoi_id").str.contains("ARG")).then(pl.lit("ARG")).otherwise(pl.lit("KR")))
    w = m.group_by("aoi_id", "region", "glat", "glon").agg(w=pl.col("weight").sum())
    parts = []
    for (region,), g in tqdm(w.group_by("region"), desc="AOI 일 요약", unit="region"):
        f = fwi.filter(pl.col("region") == region)
        j = g.join(f, on=["region", "glat", "glon"])
        parts.append(j.group_by("aoi_id", "date").agg(
            fwi_mean=((pl.col("FWI") * pl.col("w")).sum() / pl.col("w").filter(pl.col("FWI").is_not_null()).sum()).cast(pl.Float32),
            fwi_max=pl.col("FWI").max(),
            isi_mean=((pl.col("ISI") * pl.col("w")).sum() / pl.col("w").filter(pl.col("ISI").is_not_null()).sum()).cast(pl.Float32),
            bui_mean=((pl.col("BUI") * pl.col("w")).sum() / pl.col("w").filter(pl.col("BUI").is_not_null()).sum()).cast(pl.Float32),
            ffmc_mean=((pl.col("FFMC") * pl.col("w")).sum() / pl.col("w").filter(pl.col("FFMC").is_not_null()).sum()).cast(pl.Float32),
            dc_mean=((pl.col("DC") * pl.col("w")).sum() / pl.col("w").filter(pl.col("DC").is_not_null()).sum()).cast(pl.Float32),
            n_cells=pl.len().cast(pl.Int16)))
    out = pl.concat(parts).with_columns(fwi_class=fwi_class("fwi_mean"))
    return out.join(keep.select("aoi_id", "name", "level"), on="aoi_id").sort("aoi_id", "date")


def validate(fwi: pl.DataFrame) -> pl.DataFrame:
    """0.5° 칸 × 날짜: FWI 등급별 '다음 날 새 감지'(그날 감지 없던 0.5° 칸에 다음 날 감지) 발생률과 '다음 날 감지' 발생률."""
    from planetarium.monthly import STATUS_PATH
    from planetarium.spread_nowcast import load_points
    from datetime import date
    ok = set(pl.read_parquet(STATUS_PATH).filter(pl.col("satellite") == "S-NPP", pl.col("status") == "ok")["date_utc"].to_list())
    parts = []
    for region in ("KR", "ARG"):
        for y in tqdm(range(2012, 2026), desc=f"검증 점 {region}", unit="year", leave=False):
            p = load_points(region, date(y, 1, 1), date(y, 12, 31))
            p = p.filter(pl.col("is_src")).with_columns(
                lat=(pl.col("iy").cast(pl.Float64) + 0.5) * 0.025 - 90, lon=(pl.col("ix").cast(pl.Float64) + 0.5) * 0.025 - 180)
            parts.append(p.select("date_local", region=pl.lit(region),
                                  glat=((pl.col("lat") / GRID).floor() * GRID + GRID / 2).round(3),
                                  glon=((pl.col("lon") / GRID).floor() * GRID + GRID / 2).round(3)).unique())
    act = pl.concat(parts).rename({"date_local": "date"}).with_columns(active=pl.lit(True))
    okd = pl.DataFrame({"date": sorted(ok)}, schema={"date": pl.Date})
    f = (fwi.filter(pl.col("date").dt.year().is_between(2012, 2025), pl.col("FWI").is_not_null())
         .join(okd, on="date", how="semi").join(okd.select(date=pl.col("date") - pl.duration(days=1)), on="date", how="semi")
         .join(act, on=["region", "glat", "glon", "date"], how="left")
         .join(act.select("region", "glat", "glon", date=pl.col("date") - pl.duration(days=1), next_active="active"),
               on=["region", "glat", "glon", "date"], how="left")
         .with_columns(pl.col("active", "next_active").fill_null(False)))
    # 그 지역에 감지가 한 번이라도 있던 0.5° 칸만 (바다·불이 안 나는 칸 제외)
    fire_cells = act.select("region", "glat", "glon").unique()
    f = f.join(fire_cells, on=["region", "glat", "glon"], how="semi")
    rep = summarize(f, "fwi_class", "FWI", [n for _, n in CLASSES])
    rep.write_csv(REPORTS_DIR / "fwi_validation.csv")
    rel = summarize(f.filter(pl.col("fwi_pct").is_not_null()), "fwi_rel_class", "fwi_pct", [n for _, n in REL_CLASSES])
    rel.write_csv(REPORTS_DIR / "fwi_validation_relative.csv")
    return rep, rel


def summarize(f: pl.DataFrame, cls: str, score: str, names: list[str]) -> pl.DataFrame:
    """등급별 다음 날 감지·새 감지 발생률, 지역 평균 대비 배율, score로 가른 AUC(그날 감지 없던 칸)."""
    from sklearn.metrics import roc_auc_score
    order = {name: i for i, name in enumerate(names)}
    rep = f.group_by("region", cls).agg(
        cell_days=pl.len(),
        p_next_detection=pl.col("next_active").mean(),
        p_new_detection=pl.col("next_active").filter(~pl.col("active")).mean(),
        fwi_median=pl.col("FWI").median(),
    ).with_columns(o=pl.col(cls).replace_strict(order)).sort("region", "o").drop("o")
    base = f.group_by("region").agg(base_new=pl.col("next_active").filter(~pl.col("active")).mean())
    rep = rep.join(base, on="region").with_columns(lift_new=pl.col("p_new_detection") / pl.col("base_new"))
    aucs = {}
    for region in ("KR", "ARG"):
        g = f.filter(pl.col("region") == region, ~pl.col("active"))
        aucs[region] = roc_auc_score(g["next_active"].to_numpy(), g[score].to_numpy())
    return rep.with_columns(**{f"auc_new_by_{score.lower()}": pl.col("region").replace_strict(aucs, return_dtype=pl.Float64)})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="캐나다 FWI 시스템 일별 (NASA POWER)")
    ap.add_argument("step", choices=["compute", "aoi", "validate", "all", "extend"])
    ap.add_argument("--fetch-date", default=None, help="extend: POWER 끝 날짜 YYYYMMDD (기본 오늘)")
    ap.add_argument("--skip-fetch", action="store_true", help="extend: 이미 받은 캐시만 합침")
    args = ap.parse_args(argv)
    start_run_log("fwi")
    t0 = time.perf_counter()
    steps = ["compute", "aoi", "validate"] if args.step == "all" else [args.step]
    if args.step == "extend":
        from datetime import date
        fd = args.fetch_date or date.today().strftime("%Y%m%d")
        cell, aoi = extend(fd, args.skip_fetch)
        cell.write_parquet(FWI_PATH, compression="zstd")
        aoi.write_parquet(FWI_AOI_PATH, compression="zstd")
        n26 = cell.filter(pl.col("date").dt.year() >= 2026)
        print(f"[완료] {FWI_PATH.name}: {cell.height:,}행 (2026년 {n26.height:,}행, 마지막 날 {n26['date'].max()}), "
              f"{FWI_AOI_PATH.name}: {aoi.height:,}행")
        last = aoi.filter(pl.col("date").dt.year() >= 2026).group_by("aoi_id").agg(pl.col("date").max()).join(aoi, on=["aoi_id", "date"])
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_hide_column_data_types=True, float_precision=2):
            print(last.filter(pl.col("aoi_id").is_in(["KR", "SGG_47730", "ADM0_ARG"]))
                  .select("aoi_id", "name", "date", "fwi_mean", "fwi_max", "fwi_class", "fwi_rel_class"))
        print(f"[완료] extend {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")
        return
    if "compute" in steps:
        f, clim = add_relative(compute(), ["region", "glat", "glon"], "FWI")
        f.write_parquet(FWI_PATH, compression="zstd")
        clim.write_parquet(CLIM_CELL_PATH, compression="zstd")
        print(f"[완료] {CLIM_CELL_PATH.name}: {clim.height:,}행 (칸 × 달 경계값 q25·q50·q75·q90)")
        print(f"[완료] {FWI_PATH.name}: {f.height:,}행, 결측 {f['FWI'].null_count():,}, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")
        print(f.group_by("region").agg(pl.col("FWI").quantile(q).alias(f"q{int(q * 100)}") for q in (0.5, 0.9, 0.99)))
    fwi = pl.read_parquet(FWI_PATH)
    if "aoi" in steps:
        a, clim = add_relative(aoi_day(fwi), ["aoi_id"], "fwi_mean")
        a.write_parquet(FWI_AOI_PATH, compression="zstd")
        clim.write_parquet(CLIM_AOI_PATH, compression="zstd")
        print(f"[완료] {CLIM_AOI_PATH.name}: {clim.height:,}행 (AOI × 달 경계값)")
        print(f"[완료] {FWI_AOI_PATH.name}: {a.height:,}행, AOI {a['aoi_id'].n_unique()}개")
        print(a.filter(pl.col("aoi_id").is_in(["KR", "SGG_47730"]), pl.col("date").is_between(pl.date(2025, 3, 20), pl.date(2025, 3, 28)))
              .select("aoi_id", "name", "date", "fwi_mean", "fwi_class", "fwi_pct", "fwi_rel_class").sort("aoi_id", "date"))
    if "validate" in steps:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                       tbl_hide_dataframe_shape=True, float_precision=4, thousands_separator=","):
            for t in validate(fwi):
                print(t)
    print(f"[완료] {args.step} {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
