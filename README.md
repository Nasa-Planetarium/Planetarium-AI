# Planetarium-AI

Team Planetarium · NASA Space Apps 2026 Seoul
챌린지: Harmonization of MODIS and VIIRS Hot Spots

폴더 구조는 [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/) 관례를 따릅니다.

```
Planetarium-AI/
├── data/
│   ├── raw/firms/            원본 그대로 (수정 금지)
│   │   ├── modis/            modis_2000~2025_all_countries.zip (26개)
│   │   ├── viirs-snpp/       viirs-snpp_2012~2025_all_countries.zip (14개)
│   │   ├── viirs-noaa20/     viirs-jpss1_2018~2025_all_countries.zip (8개, NOAA-20)
│   │   ├── downloads/        FIRMS Archive Download zip (M-C61·SV-C2·J1V-C2·J2V-C2 2026-01~09 / J2V-C2_846116 = NOAA-21 정식 2023-03~2025-12)
│   │   └── active_fire_7d/   FIRMS 최근 7일 NRT csv (날짜별 폴더)
│   ├── external/             외부 출처 (자세한 내용은 data/external/README.md)
│   │   ├── boundaries/       행정구역 경계 (admdongkor 2026-10·2025-12, geoBoundaries, KOSTAT 2013, Natural Earth)
│   │   └── kfs_wildfire_stats/ 산림청 산불통계 2022–2025
│   ├── interim/
│   │   ├── firms_hotspots/   공통 스키마 감지 점: <sensor>_<year> / dl_<kind>_<slug>_<id> / nrt7d_<slug>_<날짜>.parquet
│   │   └── firms_coverage.parquet  파일×위성별 시간·공간 범위 (중복 제거용)
│   └── processed/            격자·월 집계, 보정 결과처럼 모델/앱에 바로 쓰는 최종본
├── docs/                     문서
├── models/                   학습한 보정 모델
├── notebooks/                탐색용 노트북 (예: 1.0-gh-eda-korea.ipynb)
├── references/               FIRMS 속성 설명 등 참고 자료
├── reports/
│   ├── figures/              발표용 그림
│   ├── firms_merge_summary.csv   위성 × source × 연도별 행 수·기간
│   ├── firms_dedupe_summary.csv  중복 제거로 빠진 행 수
│   └── *_run.log                 실행 로그
├── planetarium/              파이썬 패키지 (재사용 코드)
│   ├── config.py             경로·상수 (동아시아 상자, 알려진 결측일, SEED)
│   ├── runlog.py             실행 로그를 reports/*_run.log에도 기록
│   ├── dataset.py            원본 zip·csv -> 통합 Parquet, load_hotspots() (중복 제거 포함)
│   ├── features.py           통합 Parquet -> 0.1° 칸 × 현지 날짜 집계 (다섯 위성), 고정 열원 마스크
│   ├── monthly.py            위성별 자료 상태(satellite_day_status), 칸 × 월 집계 (동아시아)
│   ├── korea.py              한국 0.1° 칸 목록 (육지 비율, 시·도/시군구)
│   └── train_table.py        M1 학습 표 (MODIS 특성 → S-NPP 감지 수)
├── pyproject.toml            uv 의존성
└── README.md
```

## 실행

```powershell
# 1) uv 설치 (이미 있으면 건너뜀)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2) 의존성 설치 (.venv 생성)
uv sync

# 3) 전체 통합 (아카이브 MODIS·S-NPP·NOAA-20 + downloads + 최근 7일)
uv run python -m planetarium.dataset
```

옵션:

| 명령 | 설명 |
|---|---|
| `--sensor modis` / `viirs-snpp` / `viirs-noaa20` | 아카이브 한 센서만 (downloads·7일은 건너뜀) |
| `--years 2012-2014 2020` | 아카이브 일부 연도만 (downloads·7일은 건너뜀) |
| `--downloads-only` | `downloads/`와 `active_fire_7d/`만 처리 |
| `--force` | 이미 만든 Parquet도 다시 생성 |
| `--summary-only` | 변환은 건너뛰고 범위·요약·중복 제거 리포트만 |

NOAA-20 아카이브는 폴더가 `viirs-noaa20/`, zip 이름은 `viirs-jpss1_<year>_all_countries.zip`(FIRMS가 JPSS-1으로 부름)이고
결과는 `viirs-noaa20_<year>.parquet`, `satellite = NOAA-20`.

### FIRMS Archive Download (`downloads/`)

zip이나 csv를 그대로 넣으면 됩니다. 안의 `fire_<archive|nrt>_<product>_<id>.csv` 이름으로 위성을 구분합니다.
`fire_archive` = SP(정식), `fire_nrt` = NRT. 결과 `dl_<archive|nrt>_<slug>_<id>.parquet`.

| product | 위성 | slug |
|---|---|---|
| `M-C61` | MODIS Terra·Aqua | `modis` |
| `SV-C2` | S-NPP | `viirs-snpp` |
| `J1V-C2` | NOAA-20 | `viirs-noaa20` |
| `J2V-C2` | NOAA-21 | `viirs-noaa21` |
| `LS` | Landsat | 건너뜀 (컬럼 뜻이 다름) |

### 최근 7일 NRT (`active_fire_7d/<받은 날짜>/`)

`MODIS_C6_1_Global_7d.csv`(Terra·Aqua), `J1_VIIRS_C2_Global_7d.csv`(NOAA-20), `J2_VIIRS_C2_Global_7d.csv`(NOAA-21).
결과 `nrt7d_<slug>_<받은 날짜>.parquet`. 전 세계 파일이지만 지역 필터는 하지 않습니다(작고, 위성별 자료 유무 판정에 전 세계 감지 수가 필요).

### source 구분

`version`이 숫자만(`6.1`, `61.03`, `2`)이면 `SP`, `NRT`·`RT`·`URT` 접미사가 붙으면 `NRT` (FIRMS 속성 문서 기준).

### 중복 제거 (`load_hotspots()` 기본 적용)

같은 관측이 여러 파일에 들어 있어 위성별로 우선순위를 둡니다: **SP > Archive Download NRT > 7일 NRT**.

- 낮은 순위 파일의 감지는, 높은 순위 파일이 같은 위성·같은 시간 범위(첫 날 00:00 ~ 마지막 감지 시각)·같은 지역(감지 위경도 범위를 5° 단위로 바깥 반올림)을 덮으면 버립니다.
  날짜 단위가 아니라 실제 시각 기준이라, 예를 들어 drive zip NRT가 끝난 2026-11-11 10시(UTC) 이후의 7일 파일 감지는 남습니다.
- 같은 순위끼리는 `(satellite, latitude, longitude, acq_datetime_utc)`가 같은 행을 하나만 남깁니다.
- 결과: `reports/firms_dedupe_summary.csv`(파일별 제거 행 수), `reports/firms_merge_summary.csv`(`rows`, `rows_after_dedupe`).

```python
import polars as pl
from planetarium.dataset import load_hotspots

korea = (
    load_hotspots()                       # 중복 제거 적용. 원본 그대로는 load_hotspots(dedupe=False)
    .filter(pl.col("country") == "Republic of Korea", pl.col("type") == 0)
    .collect()
)
```

## 격자 칸 × 날짜 집계 (`features`)

통합 Parquet(중복 제거 적용)을 0.1° 칸 × 현지 태양시 날짜 × 위성 × source로 집계합니다. 다섯 위성(Terra·Aqua·S-NPP·NOAA-20·NOAA-21) 모두.

- SP 자료: 식생 화재(type 0)만.
- NRT(type 없음): SP 자료에서 type 1·2가 2개 연도 이상 나온 0.01° 칸(`static_source_cells_0.01deg.parquet`)의 감지를 뺍니다.

```powershell
uv run python -m planetarium.features            # 기본 0.1°
uv run python -m planetarium.features --force    # 마스크와 집계 모두 다시
```

결과: `data/processed/cell_day_0.1deg/<입력과 같은 이름>.parquet`, 읽기는 `planetarium.features.load_cell_day()`.

## 위성별 자료 상태 · 칸 × 월 (`monthly`)

```powershell
uv run python -m planetarium.monthly                # 상태표 + 칸×월
uv run python -m planetarium.monthly --status-only
```

- `satellite_day_status.parquet` (위성 × UTC 날짜): `n_det_ea`(동아시아 상자 감지 수, 판정 기준), `n_det_global`(참고, 그날 전 세계 커버리지일 때만),
  `coverage`(global / east_asia), `baseline_ea`(직전 30일 중앙값), `status`, `status_raw`, `note`, `n_sats_low_same_day`.
  - 판정을 동아시아 상자로 하는 이유: 2026-01~09 Archive Download는 동아시아만 있어 전 세계 감지 수로는 커버리지 경계에서 가짜 결측이 생김.
  - `missing`: 감지 0, 자료 없음, 알려진 결측일(강제; 감지가 일부 있으면 note=`partial_known_outage`).
  - `low`: 기준선의 50% 미만 **이면서 같은 날 다른 위성은 정상**인 날. 다른 위성도 함께 낮으면 날씨·화재 활동 변화로 보고 ok(note=`low_shared_with_other_sats`),
    비교할 위성이 없는 날은 ok(note=`low_unverified_single_sat`). 원래 규칙 결과는 `status_raw`.
- `satellite_month_status.parquet` (위성 × 월): `obs_days`(ok 일수), `low_days`, `missing_days`, `month_status`(complete / partial).
  S-NPP 2026-11은 마지막 감지(11-02 15:16 UTC) 때문에 partial. NOAA-20 연결(M2)은 2026-11-03부터.
- `cell_month_0.1deg.parquet` (동아시아 칸 × 월 × 위성, 현지 날짜 기준): `n_det, n_det_conf, active_days, n_day, n_night, frp_sum, frp_max, source,
  obs_days, low_days, missing_days, days_in_month, month_status`. 감지 없는 칸·월은 행이 없으니 0/결측은 `obs_days`로 판단.

## 한국 칸 목록 (`korea`)

```powershell
uv run python -m planetarium.korea
```

`data/processed/korea_cells_0.1deg.parquet`: geoBoundaries ADM0와 겹치는 0.1° 칸마다 `land_frac`(칸 면적 중 한국 육지 비율, EPSG:5179 면적),
`is_korea`(land_frac ≥ 0.1), 겹침 면적이 가장 큰 시·도/시군구(admdongkor `*_20251231`), `island`(본토와 안 겹치는 칸), `center_in`(중심점 방식 비교용).
행정구역 밖인 칸(비무장지대 부근)은 가장 가까운 시·도/시군구로 배정.

## M1 학습 표 (`train_table`)

```powershell
uv run python -m planetarium.train_table
```

- 기간 2012-02 ~ 2025-12 정식 자료, 동아시아 상자. Aqua·Terra·S-NPP 중 하나라도 `obs_days` < 50%인 달은 제외 (`reports/train_excluded_months.csv`).
- 행: 세 위성 중 하나라도 감지가 있는 칸·월 + 셋 다 0인 칸·월 같은 수만큼 무작위 표본(seed 42; 후보는 기간 중 감지가 한 번이라도 있던 칸 × 정상 월).
- 타깃 `y` = S-NPP 감지 수. 특성은 MODIS에서만 (`planetarium.train_table.MODIS_FEATURES`), S-NPP 정보·MODIS 처리 버전은 넣지 않음.
- 결과 `data/processed/train_modis_snpp.parquet`, 요약 `reports/train_table_summary.csv`.

컬럼 설명과 처리 방식은 `firms-merge-guide.html` 문서를 참고하세요.
