"""FireOutlook Planner — 서비스 파일 만들기(publish)와 갱신(refresh).

publish  학습·검증 작업 결과(data/interim/planner, reports)에서 forecast()가 읽는 서비스 파일을 data/processed/에 만든다.
         planner_aoi_weights.npz        aoi_list 지역 × 0.5° 격자 면적 비율 (피클 없음)
         planner_region_calibration.json 지역 크기 구간별 isotonic 보정점 + 구간 비율
         planner_region_skill.json      (시점, 지역 등급) → high | low | none  (검증·확인 BSS 중 작은 값)
         planner_count_choice.json      (시점, 지역 묶음) → 건수 원천·count_skill·구간 비율
         planner_region_counts_sf.parquet AutoETS·AutoTheta 서비스 예측 (대륙·전 세계·한국·아르헨티나)
         planner_sgg_choice.json        한국 시군구 2·3개월 직접 모델 선택
         planner_30d_skill.json         30일 질의 요약 skill (지역 등급별, planner_30d 검증)
         planner_meta.json              이름·버전·기준일·검증 요약·라이선스
         (planner_monthly_grid / planner_daily_grid / planner_sgg_monthly .parquet, planner_day_weights.npz 는 각 단계가 직접 만든다)
refresh  일 단위(1~7일) 갱신: 일 배열 → 표본(기준일 행) → 최종 학습·예측 → publish. 약 9분, 최대 메모리 약 17 GB.
         월 단위(1~12개월)는 달마다: planner_monthly final --origin YYYY-MM --as-of YYYY-MM-DD (grid_occurrence agg가 먼저 갱신돼 있어야 함).
실행     uv run python -m planetarium.planner_publish publish
         uv run python -m planetarium.planner_publish refresh [--last auto]
         fetch_nrt --fetch --append --with-planner  (= refresh --last auto)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import polars as pl

from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

PL = INTERIM_DATA_DIR / "planner"
META = PROCESSED_DATA_DIR / "planner_meta.json"


def _skill(b: float) -> str:
    return "high" if b >= 0.05 else "low" if b >= 0.01 else "none"


def publish() -> None:
    from planetarium.forecast import SVC, save_aoi_weights
    if not SVC["aoi_weights"].exists():      # 경계가 바뀌지 않으면 다시 쓰지 않는다 (npz는 저장 시각 때문에 sha256이 바뀜)
        save_aoi_weights(SVC["aoi_weights"])
    shutil.copyfile(PL / "monthly" / "region_calibration.json", SVC["region_cal"])
    shutil.copyfile(PL / "count_choice.json", SVC["count_choice"])
    shutil.copyfile(PL / "region_sf_service_small.parquet", SVC["region_counts_sf"])
    shutil.copyfile(PL / "sgg" / "choice.json", SVC["sgg_choice"])
    m = pl.read_csv(REPORTS_DIR / "planner_region_metrics.csv")
    m = (m.filter(pl.col("period").is_in(["validation 2020-2023 (LOYO)", "holdout 2024-2025"]))
         .group_by("horizon_months", "group").agg(b=pl.col("bss_calibrated_vs_base").min()))
    sk: dict = {}
    for h, g, b in m.iter_rows():
        sk.setdefault(str(h), {})[g] = _skill(b)
    SVC["region_skill"].write_text(json.dumps(sk, indent=1, sort_keys=True), encoding="utf-8")
    # 30일 질의 요약 skill (planner_30d: 홀드아웃 2024–2025 "기준일 다음 날부터 30일", raw·교차 보정 BSS 중 작은 값)
    v30 = REPORTS_DIR / "planner_30d_validation.csv"
    if v30.exists():
        t = pl.read_csv(v30)
        iso_f = PL / "d30" / "iso30.json"
        iso = json.loads(iso_f.read_text(encoding="utf-8")) if iso_f.exists() else {}
        SVC["skill_30d"].write_text(json.dumps({u: {"iso": iso.get(u), "skill": s_, "count_skill": c_, "bss_raw": round(b1, 4), "bss_crossfit": round(b2, 4),
                                                    "count_improvement": round(ci, 4), "mean_pred": round(mp, 4), "obs_rate": round(ob, 4)}
                                                for u, s_, c_, b1, b2, ci, mp, ob in t.select(
                                                    "unit", "skill_30d", "count_skill_30d", "bss_raw", "bss_crossfit_iso", "count_improvement",
                                                    "mean_p", "obs_rate").iter_rows()}, ensure_ascii=False, indent=1), encoding="utf-8")
    mo = pl.read_parquet(PROCESSED_DATA_DIR / "planner_monthly_grid.parquet", columns=["origin_month", "as_of_date"])
    dy = PROCESSED_DATA_DIR / "planner_daily_grid.parquet"
    meta = {
        "display_name": "FireOutlook Planner", "internal_id": "planner", "version": "v1.2-planner-2026-11-15",
        "function": "planetarium.forecast.forecast(region, start, end, as_of=None)", "api": "POST /forecast (docs/api_contract.md)",
        "monthly": {"file": "planner_monthly_grid.parquet", "origin_month": mo["origin_month"][0], "as_of_date": str(mo["as_of_date"][0]),
                    "horizons_months": list(range(1, 13)), "model": "ens (XGBoost + LightGBM, isotonic); h=1 grid = FireOutlook Monthly – Grid"},
        "daily": {"file": dy.name, "as_of_date": str(pl.read_parquet(dy, columns=["as_of_date"])["as_of_date"][0]) if dy.exists() else None,
                  "horizons_days": list(range(1, 8)), "model": "ens (XGBoost + LightGBM, weighted isotonic)"},
        "korea_sgg": {"h1": "FireOutlook Monthly – District (aoi_occurrence)", "h2_h3": "planner_sgg direct XGBoost + Platt"},
        "period_rule": "1–7 days: daily model; later: monthly model × daily climatology weights (count shape for counts, detection-day shape for probability)",
        "validation": {"selection": "2020-2023", "holdout": "2024-2025 (no re-selection)", "extra": "2026-01..10",
                       "reports": ["planner_monthly_metrics.csv", "planner_daily_metrics.csv", "planner_region_metrics.csv",
                                   "planner_count_routing.csv", "planner_sgg_compare.csv", "planner_dayweights_validation.csv",
                                   "planner_region_count_compare.csv", "planner_daily_count_baseline.csv", "planner_grid_h1_vs_grid_occurrence.csv"]},
        "notice": "위성 탐지 기반 확률이며 산불 발생을 보장하지 않습니다.",
        "license_note": "Service models: XGBoost (Apache-2.0), LightGBM (MIT), scikit-learn (BSD-3), statsforecast AutoETS/AutoTheta (Apache-2.0). "
                        "TimesFM (non-commercial license) was used only in comparison experiments and is not part of the service.",
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[완료] publish: {', '.join(p.name for p in SVC.values())}, {META.name}")


def refresh(last: str | None = "auto") -> None:
    """일 단위 갱신 (fetch_nrt --with-planner). 월 단위·지역 보정·AutoETS는 달마다 따로."""
    from planetarium import planner_daily as pdl
    t0 = time.perf_counter()
    pdl.set_last(last)
    print(f"[planner] 일 단위 기준일 {pdl.LAST}", flush=True)
    pdl.build()
    pdl.samples()
    pdl.final()
    publish()
    print(f"[완료] planner refresh {time.perf_counter() - t0:,.0f}s, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["publish", "refresh"])
    ap.add_argument("--last", default="auto", help="일 단위 기준일 YYYY-MM-DD 또는 auto")
    a = ap.parse_args(argv)
    start_run_log("planner_publish")
    publish() if a.step == "publish" else refresh(a.last)


if __name__ == "__main__":
    main()
