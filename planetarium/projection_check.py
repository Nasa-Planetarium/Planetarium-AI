"""진행 중인 달 월말 예상(value_projected) 방식 비교: 관측일 비율 vs 일별 기후 가중치.

기존   projected = 지금까지 합계 × (그 달 일수 / 지난 일수)
새 방식 projected = 지금까지 합계 ÷ (그 달 기후 탐지 수 중 지난 날짜들이 차지하는 비중)   — planner_day_weights와 같은 계산
       (0.5° 칸별 ±7일 평균, 자료 적은 칸은 나라·대륙 모양). 비중 < 0.15면 기존 방식으로 대체.
혼합   기후 비중과 날짜 비율의 평균으로 나누기 (기후 모양의 잡음을 반으로 줄인 변형)
검증   2018–2025 각 달을 5·10·15·20일째 끝에서 잘라 월말 예상 vs 실제. 기후 가중치는 그 해를 뺀 2012–2025로 (누설 없음).
       0.5° 칸 단위로 계산해 지역(시군구·시도·adm1·나라·대륙·전 세계)은 면적 가중치로 합침.
       (실제 서비스는 관측일 수(status ok)를 쓰지만 이 검증에서는 지난 일수로 둠 — 대부분의 달에서 같다)
결과   reports/projection_method_validation.csv
실행   uv run python -m planetarium.projection_check
"""

from __future__ import annotations

import calendar
import time
from datetime import date

import numpy as np
import polars as pl
import scipy.sparse as sp

from planetarium.config import PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.planner_daily import D0, WORK
from planetarium.planner_dayweights import SLOT, climatology
from planetarium.runlog import peak_memory_gb, start_run_log

CUTS = (5, 10, 15, 20)
YEARS = range(2018, 2026)
MIN_SHARE = 0.15
UNITS = {"Korea sgg": "sgg", "Korea sido": "sido", "adm1": "adm1", "countries": "country", "continents": "continent", "world": "world"}


def main() -> pl.DataFrame:
    t0 = time.perf_counter()
    V = np.load(WORK / "V.npy", mmap_mode="r")
    z = np.load(PROCESSED_DATA_DIR / "planner_aoi_weights.npz")
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    lv = pl.DataFrame({"aoi_id": list(z["ids"])}).join(pl.read_parquet(PROCESSED_DATA_DIR / "aoi_list.parquet", columns=["aoi_id", "level"]),
                                                     on="aoi_id", how="left")
    masks = {u: (lv["level"] == l_).to_numpy() for u, l_ in UNITS.items()}
    acc: dict = {}
    for Y in YEARS:
        C, _ = climatology([y for y in range(2012, 2026) if y != Y], "count")
        for m in range(1, 13):
            dim = calendar.monthrange(Y, m)[1]
            i0 = (date(Y, m, 1) - D0).days
            Vm = np.asarray(V[:, i0:i0 + dim], dtype=np.float64)
            actual = Vm.sum(1)
            all_s = [SLOT[(m, d)] for d in range(1, dim + 1)]
            tot = C[:, all_s].sum(1)
            for k in CUTS:
                mtd = Vm[:, :k].sum(1)
                uni = mtd * dim / k
                share = C[:, all_s[:k]].sum(1) / np.where(tot > 0, tot, 1)
                clim = np.where(share >= MIN_SHARE, mtd / np.maximum(share, 1e-9), uni)
                sb = 0.5 * share + 0.5 * k / dim          # 혼합: 기후 비중과 날짜 비율의 평균
                blend = np.where(sb >= MIN_SHARE, mtd / np.maximum(sb, 1e-9), uni)
                frames = [("0.5° grid cells", uni, clim, blend, actual)] + [
                    (u, (W @ uni)[mk], (W @ clim)[mk], (W @ blend)[mk], (W @ actual)[mk]) for u, mk in masks.items()]
                for u, pu, pc, pb, ac in frames:
                    a_ = acc.setdefault((u, k), np.zeros(7))
                    a_ += [np.abs(pu - ac).sum(), np.abs(pc - ac).sum(), (pu - ac).sum(), (pc - ac).sum(), ac.sum(),
                           np.abs(pb - ac).sum(), (pb - ac).sum()]
        print(f"[{Y}] {time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB", flush=True)
    rows = [{"unit": u, "cut_day": k, "wape_uniform": v[0] / v[4], "wape_climatology": v[1] / v[4], "bias_uniform": v[2] / v[4],
             "bias_climatology": v[3] / v[4], "wape_improvement": 1 - v[1] / v[0],
             "wape_blend": v[5] / v[4], "bias_blend": v[6] / v[4], "wape_improvement_blend": 1 - v[5] / v[0]} for (u, k), v in acc.items()]
    m = pl.DataFrame(rows).sort("unit", "cut_day")
    m.write_csv(REPORTS_DIR / "projection_method_validation.csv")
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=3):
        print(m)
    return m


if __name__ == "__main__":
    start_run_log("projection_check")
    main()
