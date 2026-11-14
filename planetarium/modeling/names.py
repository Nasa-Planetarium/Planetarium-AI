"""모델·파일·값 이름 (2026-11-14 확정, 의미가 드러나는 이름으로 변경).

모델 (학습 결과·수치는 그대로, 이름만 바뀜)
  m2v_xgb          "M2V Harmonizer"       MODIS → VIIRS(S-NPP 기준) 환산 XGBoost (구 M1)
  m2v_xgb_terra    Terra 단독 변형         2000-11 ~ 2002-06 적용 (구 M1_terra)
  m2v_q10_q90      "M2V Uncertainty Band" 10·90 분위수 구간 (구 M1 분위수 모델)
  ratio_lat        위도 5° 띠 × 월 비율 기준선 (구 B1)
  ratio_country    국가 × 월 비율 (하위지역 → 위도띠 대체) (구 B1c)
  loglinear_ridge  log1p 선형 Ridge (구 B2)
  n20_bridge       NOAA-20 → S-NPP 연결 계수 (구 M2)
  '+zr' 접미사: 'MODIS·주변 3×3 감지가 모두 0이면 예측 0' 규칙을 켠 변형.

value_source (통합 시계열)
  modis_m2v_terra  2000-11 ~ 2002-06 (구 modis_harmonized_terra_only)
  modis_m2v        2002-07 ~ 2012-01 (구 modis_harmonized)
  snpp_observed    2012-02 ~ 2026-10 (그대로)
  n20_bridge       2026-11 ~        (구 noaa20_bridged)
"""

from __future__ import annotations

from planetarium.config import MODELS_DIR

METHOD_VERSION = "v1-m2v-2026-11-14"

# 검증 리포트·그림의 모델 이름 (구 → 신)
MODEL_LABELS: dict[str, str] = {
    "B1": "ratio_lat",
    "B1_terra": "ratio_lat_terra",
    "B1c": "ratio_country",
    "B1c_terra": "ratio_country_terra",
    "B2": "loglinear_ridge",
    "B2+zr": "loglinear_ridge+zr",
    "M1": "m2v_xgb",
    "M1+zr": "m2v_xgb+zr",
    "M1_terra": "m2v_xgb_terra",
    "M1_terra+zr": "m2v_xgb_terra+zr",
    "M1_q": "m2v_q10_q90",
    "M2": "n20_bridge",
}

VALUE_SOURCES: dict[str, str] = {
    "modis_harmonized_terra_only": "modis_m2v_terra",
    "modis_harmonized": "modis_m2v",
    "snpp_observed": "snpp_observed",
    "noaa20_bridged": "n20_bridge",
}

# models/ 파일 (구 이름은 지우지 않고 남겨 둠)
MODEL_FILES: dict[str, str] = {
    "m1_xgb.json": "m2v_xgb.json",
    "m1_terra_xgb.json": "m2v_xgb_terra.json",
    "m1_q10_q90_xgb.json": "m2v_q10_q90.json",
    "b1_ratios.parquet": "ratio_lat.parquet",
    "b1_interval.parquet": "ratio_lat_interval.parquet",
    "b1c_ratios.parquet": "ratio_country.parquet",
    "b2_ridge.json": "loglinear_ridge.json",
    "m2_ratios.parquet": "n20_bridge.parquet",
    "m1_meta.json": "m2v_meta.json",
}

M2V_PATH = MODELS_DIR / "m2v_xgb.json"
M2V_TERRA_PATH = MODELS_DIR / "m2v_xgb_terra.json"
M2V_Q_PATH = MODELS_DIR / "m2v_q10_q90.json"
RATIO_LAT_PATH = MODELS_DIR / "ratio_lat.parquet"
RATIO_LAT_INTERVAL_PATH = MODELS_DIR / "ratio_lat_interval.parquet"
RATIO_COUNTRY_PATH = MODELS_DIR / "ratio_country.parquet"
LOGLINEAR_PATH = MODELS_DIR / "loglinear_ridge.json"
N20_BRIDGE_PATH = MODELS_DIR / "n20_bridge.parquet"
META_PATH = MODELS_DIR / "m2v_meta.json"


def label(old: str) -> str:
    return MODEL_LABELS.get(old, old)
