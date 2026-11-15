"""FireOutlook Planner — "그 지역에서 과거 같은 날짜(±7일)에 탐지가 1건 이상 있던 날의 비율" (일별 탐지일 기후값).

쓰는 곳  월 조각의 확률이 0.98 이상이면(큰 지역) 1 − (1 − p)^f 로 날짜에 나누면 하루 확률도 ≈ 1이 되어 예상 탐지일이 부풀려진다.
         그런 조각(모델·평년 각각)은 날짜별 p_day / base_p_day를 이 기후값으로 쓴다 (최근 상태를 반영할 일 단위 정보가 없으므로 기후값 그대로).
지역     aoi_id: 0.1° 칸 소속 가중치 ≥ 0.5인 칸 중 하나라도 기준 점(고정 열원 제외)이 있으면 그날 "탐지 있음".
         2012–2023 같은 달력 날짜의 비율 → ±7일 이동평균 (연말·연초 이어 붙임). 366칸 (2/29 포함).
bbox·폴리곤  0.5° 격자 칸별 같은 비율(2012–2023, ±7일)을 독립 가정으로 합침: 1 − Π(1 − r_g)^{w_g} (큰 영역은 조금 높게 나올 수 있음).
파일     data/processed/planner_day_any.npz (region, region_ids, cell, g) — 서비스 파일
         data/interim/planner/aoi_day_any.parquet (aoi_id, date: 2012–2025 탐지 있던 날) — 확인용
실행     uv run python -m planetarium.planner_dayany build | check
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium import grid_occurrence as go
from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.planner_dayweights import HALF, SLOT
from planetarium.runlog import peak_memory_gb, start_run_log

OUT = PROCESSED_DATA_DIR / "planner_day_any.npz"
AOI_DAYS = INTERIM_DATA_DIR / "planner" / "aoi_day_any.parquet"
CLIM_YEARS = range(2012, 2024)


def _smooth(M: np.ndarray) -> np.ndarray:
    K = np.concatenate([M[:, -HALF:], M, M[:, :HALF]], axis=1)
    cs = np.cumsum(np.pad(K, ((0, 0), (1, 0))), axis=1)
    return (cs[:, 2 * HALF + 1:] - cs[:, :-(2 * HALF + 1)]) / (2 * HALF + 1)


def _slot_years(years) -> np.ndarray:
    """달력 칸마다 그 칸이 있는 해의 수 (2/29는 윤년만)."""
    n = np.zeros(366)
    for y in years:
        d = date(y, 1, 1)
        while d.year == y:
            n[SLOT[(d.month, d.day)]] += 1
            d += timedelta(1)
    return n


def aoi_days() -> pl.DataFrame:
    """aoi × 날짜: 탐지 있던 날 (2012-02 ~ 2025). 해마다 읽는다."""
    from planetarium.harmonize import aoi_membership
    mem, _ = aoi_membership()
    mem = mem.filter(pl.col("weight") >= 0.5).select("cell_id", "aoi_id")
    parts = []
    for y in tqdm(range(2012, 2026), desc="aoi 탐지일", unit="year"):
        cd = (pl.scan_parquet(PROCESSED_DATA_DIR / "harmonized_cell_day.parquet").filter(pl.col("year") == y, pl.col("n_points") > 0)
              .select("cell_id", "date_local").unique().collect())
        parts.append(cd.join(mem, on="cell_id").select("aoi_id", date="date_local").unique())
    out = pl.concat(parts)
    AOI_DAYS.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(AOI_DAYS)
    return out


def build() -> None:
    t0 = time.perf_counter()
    ad = aoi_days() if not AOI_DAYS.exists() else pl.read_parquet(AOI_DAYS)
    ny = _slot_years(CLIM_YEARS)
    ids = sorted(pl.read_parquet(PROCESSED_DATA_DIR / "aoi_list.parquet")["aoi_id"].to_list())
    ix = {a: i for i, a in enumerate(ids)}
    # 2012-01은 VIIRS 기준 점 이전(MODIS 시기)이라 빼고, 1월 칸은 11해로 나눈다 (격자와 같은 기간)
    ny = ny.copy()
    for s_, (m_, _) in enumerate(sorted(SLOT, key=SLOT.get)):
        if m_ == 1:
            ny[s_] -= 1
    c = (ad.filter(pl.col("date").dt.year().is_in(list(CLIM_YEARS)), pl.col("date") >= date(2012, 2, 1))
         .with_columns(slot=pl.struct(m=pl.col("date").dt.month(), d=pl.col("date").dt.day())
                       .map_elements(lambda s: SLOT[(s["m"], s["d"])], return_dtype=pl.Int32))
         .group_by("aoi_id", "slot").agg(n=pl.len()))
    R = np.zeros((len(ids), 366))
    R[[ix[a] for a in c["aoi_id"].to_list()], c["slot"].to_numpy()] = c["n"].to_numpy()
    R = _smooth(R / np.maximum(ny, 1))
    # 0.5° 칸 (planner_daily 일 배열 P: 2012-02-01부터)
    from planetarium.planner_daily import D0, WORK
    P = np.load(WORK / "P.npy", mmap_mode="r")
    G, D = P.shape
    S = np.zeros((G, 366))
    for i in range(D):
        d = D0 + timedelta(i)
        if d.year in CLIM_YEARS:
            S[:, SLOT[(d.month, d.day)]] += np.asarray(P[:, i]) > 0
    Cc = _smooth(S / np.maximum(ny, 1))
    g = pl.read_parquet(go.CELLS)["g"].to_numpy()
    np.savez_compressed(OUT, region=R.astype(np.float32), region_ids=np.array(ids, dtype=str), cell=Cc.astype(np.float32), g=g)
    print(f"[완료] {OUT.name}: 지역 {R.shape}, 격자 {Cc.shape}, {time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB")


def actual_fire_days(aoi_id: str, start_md: tuple[int, int], n_days: int, years=range(2016, 2026)) -> float:
    """과거 같은 기간(그해 start_md부터 n_days일)의 실제 탐지일 수 평균."""
    ad = pl.read_parquet(AOI_DAYS).filter(pl.col("aoi_id") == aoi_id)
    have = set(ad["date"].to_list())
    out = []
    for y in years:
        d0 = date(y, *start_md)
        out.append(sum((d0 + timedelta(k)) in have for k in range(n_days)))
    return float(np.mean(out))


def check(before_csv: str | None = None) -> pl.DataFrame:
    from planetarium import forecast as F
    rows = []
    for reg in ("ADM1_ARG-1299", "ADM0_ARG", "WORLD", "SIDO_51"):
        r = F.forecast(reg, horizon_days=30, today="2026-11-15")
        rows.append({"region": reg, "p_any": r["p_any"], "value_source": r["value_source"],
                     "after_model": r["expected_fire_days"], "after_base": r["baseline"]["expected_fire_days"],
                     "actual_2016_2025_mean": actual_fire_days(reg, (11, 15), 30)})
    m = pl.DataFrame(rows)
    if before_csv:
        m = pl.read_csv(before_csv).select("region", "before_model", "before_base").join(m, on="region")
    m.write_csv(REPORTS_DIR / "planner_30d_fire_days_check.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(m)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "check"])
    ap.add_argument("--before")
    a = ap.parse_args()
    start_run_log("planner_dayany")
    build() if a.step == "build" else check(a.before)
