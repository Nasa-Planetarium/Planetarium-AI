# Planetarium-AI · FIRMS 1km 백엔드

Team Planetarium · NASA Space Apps 2026 Seoul

MODIS와 VIIRS를 통합한 전 세계 과거 관측과, FIRMS API 응답을 같은 1km 격자로 변환하는 Python 패키지입니다. 백엔드는 원하는 지역과 UTC 기간을 지정해 TimesFM 3.0 예측을 실행할 수 있습니다.

```text
Planetarium-AI/
├── data/processed/
│   └── dataset/                 Release에서 설치하는 과거 통합 Parquet
├── models/
│   └── timesfm-3.0/             Google에서 내려받는 공식 체크포인트
├── firms_backend/
│   ├── api.py                  과거 관측 조회 및 FIRMS CSV 응답 변환
│   ├── converter/              공통 스키마 변환과 센서 보정 모델
│   ├── forecasting/            지역·기간별 TimesFM 추론
│   ├── release_assets.py       통합 데이터 다운로드 및 SHA-256 검증
│   └── release-manifest.json   아카이브 분할 파일 목록
├── pyproject.toml
└── README.md
```

## 1. 설치와 과거 통합 데이터

Python 3.12 환경에서 `handoff/firms-1km-backend` 브랜치를 체크아웃한 뒤 저장소 루트에서 실행합니다.

```sh
python -m pip install -e .
python -m firms_backend.release_assets install --destination data/processed
```

[데이터 Release](https://github.com/Nasa-Planetarium/Planetarium-AI/releases/tag/firms-1km-backend-2026-11-14)는 **2000-11-01~2026-11-14 UTC**의 전 세계 통합 관측을 담습니다. 처리된 관측 660,831,145건과 1km 격자·일 357,814,169행이 있습니다. 설치기는 분할 tar 14개를 순차 다운로드해 검증하며, 설치에 약 25 GiB의 여유 공간이 필요합니다. 이미 tar를 받은 경우 `--local /path/to/tars`를 붙입니다.

| 경로 (`data/processed/dataset/` 아래) | 내용 |
|---|---|
| `grid_daily/year=YYYY/month=M/data.parquet` | `date_utc, grid_id`별 검출 여부, 원본 센서·위성, FRP |
| `observations/year=YYYY/month=M/*.parquet` | 정규화된 개별 위성 관측과 획득 시각 |
| `grid_cells.parquet` | WGS84 좌표로 표현한 1km 격자 경계 |
| `metadata.json` | 관측 기간, 공통 스키마, 데이터 행 수 |

```python
from firms_backend.api import FirmsStore

area = (124.0, 33.0, 132.0, 39.0)  # west, south, east, north (WGS84)
store = FirmsStore("data/processed")
points = store.observations("2026-11-04", bbox=area, limit=1000)
cells = store.detected("2026-11-04", bbox=area, limit=1000)
```

백엔드에서 별도 볼륨을 쓰면 `FIRMS_BACKEND_DATA_HOME`을 `dataset/`의 상위 경로로 지정합니다.

## 2. FIRMS API 응답 변환

FIRMS Area API는 지역·제품·기간을 지정해 CSV를 반환합니다. 아래 예시는 백엔드가 이미 받은 CSV 응답 본문을 사용합니다. 한 호출에 하나의 MODIS 또는 VIIRS 응답을 전달합니다. `date_from`, `date_to`와 `bbox`에는 실제 API 요청 범위를 넣습니다. [NASA FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area/)

```python
from firms_backend.api import convert_api_csv

# response.content는 백엔드가 받은 FIRMS Area API의 CSV 본문(bytes)입니다.
metadata = convert_api_csv(
    response.content,
    "data/processed/updates/viirs-noaa20-2026-11-15",
    bbox=area,
    date_from="2026-11-15",
    date_to="2026-11-15",
)
```

결과는 통합본과 같은 1km 스키마의 Parquet와 요청 범위를 기록한 메타데이터입니다. 검출점 없이 헤더만 반환된 날도 요청 기간에 포함됩니다. 여러 제품이나 기간을 받으면 각각 별도 출력 폴더에 변환한 뒤 예측 시 `updates`에 전달합니다. CSV 파일을 직접 변환할 때는 `convert_firms("input.csv", "converted")`를 사용합니다.

## 3. 지역·기간별 TimesFM 3.0 예측

```sh
python -m pip install -e '.[forecast]'
hf download google/timesfm-3.0-pytorch config.json model.safetensors LICENSE \
  --revision 43046b85ec22d584a13f8098c2ed39c889e129c2 \
  --local-dir models/timesfm-3.0
```

공식 체크포인트는 Google에서 직접 설치하며 비상업적·비운영 연구 용도로 사용합니다. 설치 위치가 다르면 `FIRMS_BACKEND_TIMESFM_CHECKPOINT` 또는 `checkpoint=`를 지정합니다. [TimesFM 3.0 라이선스](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)

```python
from firms_backend.forecasting.predict import FirmsForecaster

# 백엔드 워커에서 인스턴스를 재사용합니다.
forecast = FirmsForecaster("data/processed", checkpoint="models/timesfm-3.0")

two_weeks = forecast.forecast_daily(
    "2026-11-15", end_date="2026-11-28", bbox=area, as_of="2026-11-14", limit=1000,
)
quarter = forecast.forecast_monthly(
    "2026-12", end_month="2027-02", bbox=area, as_of="2026-11-14", limit=1000,
)

# 새 관측을 기준으로 같은 지역을 예측할 때
updated = forecast.forecast_daily(
    "2026-11-18", bbox=area, as_of="2026-11-15",
    updates=["data/processed/updates/viirs-noaa20-2026-11-15"],
)
```

- 시작과 종료 시점을 모두 포함합니다. `end_date` 또는 `end_month`를 생략하면 시작 시점 하나만 반환합니다. 결과는 선택한 단위의 날짜 또는 월 순서로 정렬되며 `limit`은 각 시점에 적용됩니다.
- **일별**은 기준일을 포함한 과거 64일, **월별**은 완결된 36개월의 관측을 사용합니다. 시작일은 기준일 이후, 시작 월은 기준 월 이후여야 합니다. 월별 예측에는 기준 월의 첫 14일까지 관측 범위가 필요하며 해당 기간은 후보 격자를 고르는 데 사용합니다.
- 모델은 기준시점부터 요청한 종료 시점까지 예측하고, `rows`에는 요청 구간의 결과를 반환합니다. 긴 기간을 요청할 수 있으며, 장기 예측의 정확도는 별도로 검증해야 합니다.
- `as_of`의 기본값은 요청 지역을 덮는 데이터의 마지막 UTC 날짜입니다. 필요한 기간에 빈 날짜가 있으면 그 날짜를 오류로 반환합니다. 실제 검출이 0건인 날짜는 요청 범위로 기록된 경우 정상 처리합니다.
- 응답의 `rows`는 `grid_id`, `longitude`, `latitude`, 검출 확률, NOAA-20 기준 조건부 FRP 중앙값(MW)을 담습니다. 확률은 일별 `p_detected`, 월별 `p_detected_any`입니다. FRP 이력이 없으면 FRP가 `null`이고, 후보 격자가 없으면 `rows`가 빈 목록입니다. 예측된 FRP의 상한은 해당 격자의 과거 관측 최댓값입니다. 응답에는 기준일, 모델 버전과 후보 격자 수도 함께 들어갑니다.

```sh
python -m firms_backend.forecasting.predict --date 2026-11-15 --end-date 2026-11-28 \
  --as-of 2026-11-14 --bbox 124 33 132 39 --limit 20
```
