"""프로젝트 경로와 공통 상수."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── FIRMS 원본 ────────────────────────────────────────────────────────────
FIRMS_RAW_DIR = RAW_DATA_DIR / "firms"
DOWNLOADS_DIR = FIRMS_RAW_DIR / "downloads"          # FIRMS Archive Download zip/csv
ACTIVE_FIRE_7D_DIR = FIRMS_RAW_DIR / "active_fire_7d"  # <받은 날짜>/*_Global_7d.csv

# ── 통합 결과 ─────────────────────────────────────────────────────────────
# 감지 점 Parquet: <sensor>_<year>.parquet, dl_<kind>_<slug>_<id>.parquet, nrt7d_<slug>_<날짜>.parquet
FIRMS_MERGED_DIR = INTERIM_DATA_DIR / "firms_hotspots"
# 파일별 시간·공간 범위 (중복 제거 우선순위 계산용). FIRMS_MERGED_DIR 밖에 두어 glob에 섞이지 않게 한다.
FIRMS_COVERAGE_PATH = INTERIM_DATA_DIR / "firms_coverage.parquet"


def cell_day_dir(res_deg: float = 0.1) -> Path:
    """격자 칸 × 날짜 집계 폴더: data/processed/cell_day_<res>deg/"""
    return PROCESSED_DATA_DIR / f"cell_day_{res_deg:g}deg"


# 아카이브 폴더 이름 -> sensor 값
SENSORS: dict[str, str] = {
    "modis": "MODIS",
    "viirs-snpp": "VIIRS",
    "viirs-noaa20": "VIIRS",
}
# 아카이브 zip 접두어가 폴더 이름과 다른 경우 (FIRMS는 NOAA-20을 JPSS-1로 부른다)
ARCHIVE_ZIP_PREFIX: dict[str, str] = {
    "viirs-noaa20": "viirs-jpss1",
}

# ── 지역 ──────────────────────────────────────────────────────────────────
# 상자 형식: (lon_min, lat_min, lon_max, lat_max) — FIRMS 지역 지정 순서와 같음
EAST_ASIA_BBOX = (100.0, 20.0, 145.0, 50.0)
WORLD_BBOX = (-180.0, -90.0, 180.0, 90.0)


def _bbox_from_env(name: str, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """환경변수로 지역을 바꿔 다시 돌릴 수 있게: 예) PLANETARIUM_ANALYSIS_BBOX="100,20,145,50" """
    import os
    v = os.environ.get(name)
    if not v:
        return default
    parts = tuple(float(x) for x in v.split(","))
    if len(parts) != 4:
        raise ValueError(f"{name}는 'lon_min,lat_min,lon_max,lat_max' 형식이어야 합니다: {v}")
    return parts  # type: ignore[return-value]


# 분석(칸×월 집계·학습·통합 시계열) 범위. 기본 전 세계.
ANALYSIS_BBOX = _bbox_from_env("PLANETARIUM_ANALYSIS_BBOX", WORLD_BBOX)
# 위성 자료 상태(satellite_day_status) 판정 범위.
# 2026-01~09와 NOAA-21 정식 자료도 전 세계로 받았으므로(846238~846244) 모든 기간을 전 세계 감지 수로 판정한다.
# (이전 v0는 동아시아 상자 기준이었음)
STATUS_BBOX = _bbox_from_env("PLANETARIUM_STATUS_BBOX", WORLD_BBOX)
# 자료가 일부 지역만 있는 기간 (이 밖의 칸은 해당 월 '자료 없음'으로 처리). 지금은 없음.
# 예) [(date(2026, 1, 1), date(2026, 9, 30), EAST_ASIA_BBOX)]
REGIONAL_ONLY_PERIODS: list[tuple[date, date, tuple[float, float, float, float]]] = []

# ── 위성 일정 ─────────────────────────────────────────────────────────────
# S-NPP 데이터 전달 종료 (이후는 NOAA-20을 S-NPP 기준으로 연결)
SNPP_END_UTC = datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc)
# 실제 마지막 S-NPP 감지는 2026-11-02 15:16 UTC → 그때까지는 실측으로 쓰고, 2026-11월은 S-NPP 기준 partial.
SNPP_LAST_DETECTION_UTC = datetime(2026, 11, 2, 15, 16, tzinfo=timezone.utc)
# M2(NOAA-20 → S-NPP 환산) 연결 시작일 (T6)
NOAA20_BRIDGE_START = date(2026, 11, 3)
# 알려진 결측일 (UTC 날짜). T2에서 status=missing으로 강제한다.
KNOWN_MISSING_DAYS: dict[str, list[date]] = {}
# 예전에 알려진 결측일로 강제 missing하던 날. 전 세계 감지 수가 정상(직전 30일 중앙값의 0.93~1.14배)이라
# 예전 NRT 파일에서 비어 있던 것이 이후 파일에서 채워진 것으로 보고 강제를 해제했다 (일반 규칙으로 판정, note만 남김).
KNOWN_REFILLED_DAYS: dict[str, list[date]] = {
    "NOAA-20": [date(2026, 11, 5), date(2026, 11, 6), date(2026, 11, 7)],
    "NOAA-21": [date(2026, 11, 5), date(2026, 11, 6)],
}
# 수신 장애로 VIIRS 감지가 급감한 날 (참고용; T2의 low 판정으로 잡히는지 확인)
KNOWN_LOW_DAYS_VIIRS = [date(2026, 10, 12), date(2026, 10, 13)]

SEED = 42
