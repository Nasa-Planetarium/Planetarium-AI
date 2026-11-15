"""/occurrence 와 /forecast(FireOutlook Planner)가 같은 대상·시점에서 같은 값을 내는지 확인.

    uv run python -m tests.test_forecast_consistency        (pytest가 있으면 pytest tests/ 로도 돈다)

- 한국 시군구 다음 달(2026-12 한 달 전체) → aoi_occurrence_nextmonth_latest.p_any 와 같아야 함
- 0.5° 격자 한 칸 다음 달 → grid_occurrence_nextmonth_latest.p_any 와 같아야 함
- 응답 시간: 미리 읽은 뒤 aoi 한 건 < 1초
- 30일 질의: skill none이면 평년 값(p_any = baseline.p_any, p_day = base_p_day), 아니면 보정 모델이고 날짜별 합성 = p_any
"""

from __future__ import annotations

import time

import polars as pl

from planetarium.config import PROCESSED_DATA_DIR
from planetarium.forecast import forecast

NEXT = ("2026-12-01", "2026-12-31")


def test_sgg_next_month_matches_occurrence():
    occ = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_occurrence_nextmonth_latest.parquet").sort("aoi_id")
    assert occ["target_month"].unique().to_list() == ["2026-12"]
    sample = occ.gather_every(25)       # 252개 중 11개
    for a, p in sample.select("aoi_id", "p_any").iter_rows():
        r = forecast(a, *NEXT)
        assert abs(r["p_any"] - round(p, 4)) < 1e-6, (a, r["p_any"], p)


def test_grid_cell_next_month_matches_occurrence_grid():
    g = pl.read_parquet(PROCESSED_DATA_DIR / "grid_occurrence_nextmonth_latest.parquet")
    sample = pl.concat([g.filter(pl.col("p_any") > 0.5).head(5), g.filter(pl.col("p_any") < 0.2).head(5),
                        g.filter(pl.col("adm0") == "KOR").head(3)])
    for cid, lat, lon, p in sample.select("cell_id", "lat", "lon", "p_any").iter_rows():
        r = forecast((lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25), *NEXT)
        assert abs(r["p_any"] - round(p, 4)) < 1e-6, (cid, r["p_any"], p)


def test_latency_under_1s():
    forecast("SIDO_51", "2026-11-15", "2026-11-21")        # 파일 읽기(첫 호출)
    t0 = time.perf_counter()
    forecast("SIDO_51", "2026-11-15", "2027-01-31")
    assert time.perf_counter() - t0 < 1.0


def test_30d_value_source():
    """30일 질의: skill none이면 p_any = 평년(baseline.p_any)이고 날짜별 p_day = base_p_day; 아니면 보정 모델이고 날짜별 합성 = p_any."""
    import numpy as np
    for reg in ("SGG_47730", "ADM0_ARG", "SIDO_51", "ADM1_ARG-1299", (127.0, 37.5, 127.5, 38.0), "WORLD"):
        r = forecast(reg, horizon_days=30, today="2026-11-15")
        assert len(r["daily"]) == 30 and r["daily"][0]["lead_days_from_as_of"] == 1
        comp = 1 - float(np.prod([1 - d["p_day"] for d in r["daily"]]))
        assert abs(comp - r["p_any"]) < 2e-3, (reg, comp, r["p_any"])
        if r["skill"] == "none":
            assert r["value_source"] == "baseline" and r["p_any"] == r["baseline"]["p_any"], reg
            assert all(d["p_day"] == d["base_p_day"] for d in r["daily"]), reg
        else:
            assert r["value_source"] == "model_calibrated", reg


if __name__ == "__main__":
    for f in (test_sgg_next_month_matches_occurrence, test_grid_cell_next_month_matches_occurrence_grid, test_latency_under_1s,
              test_30d_value_source):
        f()
        print(f"[통과] {f.__name__}")
