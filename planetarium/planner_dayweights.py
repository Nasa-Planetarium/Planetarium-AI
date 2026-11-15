"""FireOutlook Planner — 달 일부 기간의 비율 f를 "날짜 수 비율" 대신 "일별 기후값 가중치"로.

기후값  0.5° 격자 × 달력 날짜(2/29 포함 366칸)마다 연도 평균 탐지 수(S-NPP 기준 value), 앞뒤 ±7일 이동평균(연말·연초 이어 붙임).
        한 달 안에서 요청 날짜들의 기후값 합 / 그 달 전체 기후값 합 = f.
대체    그 칸·그 달의 과거 탐지일 수가 MIN_DAYS 미만이면 같은 나라(adm0) 모든 칸의 합친 모양 → 나라도 부족하면 대륙 → 그래도 부족하면 균등.
지역    지역 f = Σ w_g · E_g · f_g / Σ w_g · E_g  (E_g = 그 칸 예년 같은 달 중앙값 건수; 합이 0이면 균등)
검증    홀드아웃 2024–2025, 기준일을 7일 간격으로 잡고 "k = 8 … 90일 뒤부터 7일" 구간을 월 예측 조각으로 계산 → 균등 vs 기후 가중.
        기후값은 2012–2023만 사용(누설 없음). 서비스용은 2012–2025.
실행    uv run python -m planetarium.planner_dayweights validate | build
"""

from __future__ import annotations

import argparse
import calendar
import json
import time
from datetime import date, timedelta

import numpy as np
import polars as pl

from planetarium import grid_occurrence as go
from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

DAILY = INTERIM_DATA_DIR / "planner" / "daily"
MONTHLY = INTERIM_DATA_DIR / "planner" / "monthly"
OUT = PROCESSED_DATA_DIR / "planner_day_weights.npz"   # 서비스 파일 (forecast.py가 읽음)
D0 = date(2012, 2, 1)
HALF = 7
MIN_DAYS = 15          # 그 칸·그 달 과거 탐지일 수 (전체 해 합) 기준
SLOTS = [(m, d) for m in range(1, 13) for d in range(1, calendar.monthrange(2000, m)[1] + 1)]   # 366칸
SLOT = {md: i for i, md in enumerate(SLOTS)}


def _arrays():
    return np.load(DAILY / "V.npy", mmap_mode="r"), np.load(DAILY / "P.npy", mmap_mode="r")


def climatology(years: range, kind: str = "count") -> tuple[np.ndarray, np.ndarray]:
    """→ (C[g, 366] 칸별 날짜별 평균 탐지 수(±7일 평균, 대체 적용; 각 달 안의 모양만 의미), level[g, 12] 0 칸 1 나라 2 대륙 3 균등)."""
    V, P = _arrays()
    G, D = V.shape
    days = [D0 + timedelta(i) for i in range(D)]
    S = np.zeros((G, 366), np.float64)
    N = np.zeros(366)
    ND = np.zeros((G, 12))
    idx = {}
    for i, d in enumerate(days):
        if d.year in years:
            idx.setdefault(SLOT[(d.month, d.day)], []).append(i)
    for s, ii in idx.items():
        anyd = (np.asarray(P[:, ii]) > 0).sum(1)
        # count: 평균 탐지 수 (건수 조각용) / any: 탐지가 있던 날의 비율 (확률 조각용 — hazard ∝ 그날 탐지 확률)
        S[:, s] = np.asarray(V[:, ii], dtype=np.float64).sum(1) if kind == "count" else anyd
        N[s] = len(ii)
        ND[:, SLOTS[s][0] - 1] += anyd
    M = S / np.maximum(N, 1)
    K = np.concatenate([M[:, -HALF:], M, M[:, :HALF]], axis=1)
    cs = np.cumsum(np.pad(K, ((0, 0), (1, 0))), axis=1)
    C = (cs[:, 2 * HALF + 1:] - cs[:, :-(2 * HALF + 1)]) / (2 * HALF + 1)
    cells = pl.read_parquet(go.CELLS).select("adm0", "continent")
    month_slots = [np.array([s for s, (mm, _) in enumerate(SLOTS) if mm == m]) for m in range(1, 13)]
    out = C.copy()
    level = np.zeros((G, 12), np.int8)          # 0 칸, 1 나라, 2 대륙, 3 균등
    need = ND < MIN_DAYS
    level[need] = 3
    for li, key in ((2, "continent"), (1, "adm0")):     # 대륙 먼저 채우고 나라로 덮어씀 (더 가까운 쪽 우선)
        codes = cells[key].to_numpy()
        for u in np.unique(codes):
            k = np.flatnonzero(codes == u)
            agg_nd = ND[k].sum(0)
            for m in range(12):
                kk = k[need[k, m]]
                if len(kk) == 0 or agg_nd[m] < MIN_DAYS:
                    continue
                sl = month_slots[m]
                shape = C[k][:, sl].sum(0)
                if shape.sum() > 0:
                    out[np.ix_(kk, sl)] = shape / shape.sum()
                    level[kk, m] = li
    for m in range(12):                           # 균등
        sl = month_slots[m]
        kk = np.flatnonzero(level[:, m] == 3)
        out[np.ix_(kk, sl)] = 1.0
    # 칸 자체 값이 0인 달(탐지는 있지만 이동평균 합 0)은 없음; 안전하게 합 0이면 균등
    for m in range(12):
        sl = month_slots[m]
        z = out[:, sl].sum(1) <= 0
        out[np.ix_(np.flatnonzero(z), sl)] = 1.0
    return out.astype(np.float32), level


def frac(C: np.ndarray, days: list[date]) -> np.ndarray:
    """같은 달 안의 날짜들 → 칸별 f (그 해 그 달의 실제 날짜들로 정규화; 2/29는 윤년에만)."""
    y, m = days[0].year, days[0].month
    all_s = [SLOT[(m, d)] for d in range(1, calendar.monthrange(y, m)[1] + 1)]
    sel = [SLOT[(d.month, d.day)] for d in days]
    return C[:, sel].sum(1) / C[:, all_s].sum(1)


def build() -> None:
    t0 = time.perf_counter()
    C, level = climatology(range(2012, 2026))
    CA, level_a = climatology(range(2012, 2026), "any")
    g = pl.read_parquet(go.CELLS)["g"].to_numpy()
    np.savez_compressed(OUT, C=C, CA=CA, level=level, level_any=level_a, g=g)
    print(f"[완료] {OUT.name}: {C.shape}, 대체 비율 칸 {np.mean(level == 0):.3f} / 나라 {np.mean(level == 1):.3f} / 대륙 {np.mean(level == 2):.3f} / 균등 {np.mean(level == 3):.3f}, "
          f"{time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB")


def validate() -> pl.DataFrame:
    t0 = time.perf_counter()
    C, level = climatology(range(2012, 2024))
    CA, _ = climatology(range(2012, 2024), "any")
    V, P = _arrays()
    cells = pl.read_parquet(go.CELLS).select("g", "persistent_source").with_row_index("gi")
    keep = ~cells["persistent_source"].to_numpy()
    G = cells.height
    gi_of = dict(zip(cells["g"].to_list(), cells["gi"].to_list()))
    sel = json.loads((MONTHLY / "selection.json").read_text())
    # 월 예측(OOF) 밀집 배열: (h, 대상 달 t) → p, mu, 기준선 p, 중앙값
    pred = {}
    for h in range(1, 5):
        o = pl.read_parquet(MONTHLY / f"h{h}" / "oof.parquet").filter(pl.col("year").is_between(2024, 2025))
        s = sel[str(h)]
        pe = sum(o[c] * w for c, w in s["ens_weights"].items()) if s["model"] == "ens" else o[f"p_{s['model']}"]
        o = o.with_columns(p=pe, gi=pl.col("g").replace_strict(gi_of, return_dtype=pl.Int64))
        for (yy, mm), d in o.group_by("year", "month"):
            arr = {k: np.zeros(G, np.float64) for k in ("p", "mu", "pb", "med")}
            gi = d["gi"].to_numpy()
            arr["p"][gi], arr["mu"][gi] = d["p"].to_numpy(), d["mu"].to_numpy()
            arr["pb"][gi], arr["med"][gi] = d["p_base_rate"].to_numpy(), d["med_value"].fill_null(0).to_numpy()
            pred[(h, yy * 12 + mm - 1)] = arr
    origins = [date(2024, 1, 1) + timedelta(7 * i) for i in range(200)]
    origins = [d for d in origins if d + timedelta(96) <= date(2025, 12, 31)]
    ks = [8, 15, 22, 30, 45, 60, 75, 90]
    acc = {}
    for d0 in origins:
        t0m = d0.year * 12 + d0.month - 1
        for k in ks:
            win = [d0 + timedelta(k + j) for j in range(7)]
            di = [(w - D0).days for w in win]
            act_c = np.asarray(V[:, di], dtype=np.float64).sum(1)
            act_y = (np.asarray(P[:, di]) > 0).any(1).astype(float)
            res = {}
            for meth in ("uniform", "clim", "clim_count+any"):
                ec, logq = np.zeros(G), np.zeros(G)
                for (yy, mm) in sorted({(w.year, w.month) for w in win}):
                    days = [w for w in win if (w.year, w.month) == (yy, mm)]
                    t = yy * 12 + mm - 1
                    h = t - t0m
                    if h == 0:                              # 진행 중인 달: 기준선(예년 같은 달)
                        a = pred.get((1, t)) or pred.get((2, t))
                        p_m, e_m = a["pb"], a["med"]
                    else:
                        a = pred[(h, t)]
                        p_m, e_m = a["p"], a["mu"]
                    f = (np.full(G, len(days) / calendar.monthrange(yy, mm)[1]) if meth == "uniform" else frac(C, days))
                    fp = frac(CA, days) if meth == "clim_count+any" else f
                    ec += e_m * f
                    logq += fp * np.log1p(-np.clip(p_m, 0, 0.999999))
                res[meth] = (ec, 1 - np.exp(logq))
            kb = "8-14" if k < 15 else "15-30" if k <= 30 else "31-60" if k <= 60 else "61-90"
            for meth, (ec, pp) in res.items():
                a_ = acc.setdefault((kb, meth), np.zeros(4))
                a_ += [np.abs(ec - act_c)[keep].sum(), act_c[keep].sum(), ((pp - act_y) ** 2)[keep].sum(), keep.sum()]
                b_ = acc.setdefault(("all", meth), np.zeros(4))
                b_ += [np.abs(ec - act_c)[keep].sum(), act_c[keep].sum(), ((pp - act_y) ** 2)[keep].sum(), keep.sum()]
    rows = [{"k_days": kb, "method": meth, "wape_count": v[0] / v[1], "brier": v[2] / v[3], "cell_windows": int(v[3])} for (kb, meth), v in acc.items()]
    m = pl.DataFrame(rows).sort("k_days", "method")
    m.write_csv(REPORTS_DIR / "planner_dayweights_validation.csv")
    with pl.Config(tbl_rows=-1, float_precision=5):
        print(m)
    print(f"기준일 {len(origins)}개 × k {len(ks)}개, 대체: 칸 {np.mean(level == 0):.3f} / 나라 {np.mean(level == 1):.3f} / 대륙 {np.mean(level == 2):.3f} / 균등 {np.mean(level == 3):.3f}, "
          f"{time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["validate", "build"])
    a = ap.parse_args()
    start_run_log("planner_dayweights")
    {"validate": validate, "build": build}[a.step]()
