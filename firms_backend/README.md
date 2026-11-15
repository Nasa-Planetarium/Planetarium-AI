# FIRMS 1km 백엔드 전달 패키지

| 경로 | 내용 |
|---|---|
| `dataset/` | 2000-11-01~2026-11-14 UTC 전 세계 통합 데이터. 개별 관측 660,831,145건, 1km 격자·일 357,814,169행, 격자 경계와 월·시간 요약. |
| `converter/` | FIRMS MODIS 또는 VIIRS CSV를 동일한 스키마로 바꾸는 코드와 학습 모델. |
| `forecasting/` | TimesFM 3.0으로 1km 격자의 일별 7일 및 월별 12개월 예측을 생성하는 코드입니다. |
| `predictions/` | 기존 전 세계 TimesFM 3.0+LightGBM 결합 실험 결과입니다. `daily/`는 2026-11-15~21일, `annual/`은 2026-12~2027-11월 예측입니다. |

이 패키지의 기본 키는 관측·일별 예측의 `date_utc, grid_id`, 월별 예측의 `target_month, grid_id`입니다. `grid_id`는 1km 등면적 격자의 `grid_x:grid_y`입니다. 조회 API의 좌표는 WGS84 경도·위도입니다.

## 백엔드에서 읽기

저장소 루트에서 Python 3.12 가상환경에 설치합니다.

```sh
python -m pip install -e ./firms_backend
```

로컬 전달본에는 데이터가 이미 있습니다. Git 체크아웃에 데이터를 설치할 때는 Release의 검증된 tar를 사용합니다.

```sh
python -m firms_backend.release_assets install --destination ./firms_backend
```

`--group archive`, `--group daily`, `--group annual`로 필요한 자산만 받을 수 있습니다. tar를 이미 받았다면 `--local /path/to/tars`를 추가합니다. 백엔드 서비스에 `FIRMS_BACKEND_DATA_HOME`을 지정할 때는 `dataset/`과 `predictions/`의 상위 경로를 설정합니다. 코드에서 `FirmsStore(asset_root=...)`로도 지정할 수 있습니다.

```python
from firms_backend.api import FirmsStore, convert_firms

store = FirmsStore()
area = (124.0, 33.0, 132.0, 39.0)  # west, south, east, north
archive = store.detected("2026-11-04", bbox=area, limit=2000)
points = store.observations("2026-11-04", bbox=area, limit=2000)
next_day = store.forecast_daily("2026-11-15", bbox=area, minimum=0.2, limit=2000)
next_month = store.forecast_monthly("2027-05", bbox=area, minimum=0.2, limit=2000)
coverage = store.metadata()

# 별도 FIRMS CSV를 변환할 때 새 출력 경로를 지정합니다.
# convert_firms("input.csv", "converted/input")
```

함수는 HTTP 응답으로 넘길 수 있는 딕셔너리 목록을 반환합니다. 해당 날짜의 Parquet 파티션만 열고 범위와 확률 필터를 DB에서 적용합니다. 날짜별 데이터 파일이 없거나 전체 보존 기간에 없는 날짜는 `FileNotFoundError`, 유효하지 않은 입력은 `ValueError`입니다. 예측 파일에 없는 격자는 미예측입니다. `p_detected`는 해당 UTC 날짜의 기록 확률, `p_detected_any`는 해당 UTC 월에 한 번 이상 기록될 확률입니다. FRP는 검출이 기록될 때의 NOAA-20 기준 조건부 중앙값(MW)입니다.

## 변환 및 TimesFM 3.0 예측

```sh
python -m firms_backend.converter.harmonize --input input.csv --output converted/input
python -m pip install -r firms_backend/requirements-timesfm.txt
hf download google/timesfm-3.0-pytorch \
  --revision 43046b85ec22d584a13f8098c2ed39c889e129c2 \
  --local-dir firms_backend/timesfm-checkpoint
python -m firms_backend.forecasting.predict daily --cache work/daily \
  --bbox 124 33 132 39 --output next/predictions/daily
python -m firms_backend.forecasting.predict annual --cache work/monthly \
  --bbox 124 33 132 39 --output next/predictions/annual
```

`predict`는 기본으로 공식 TimesFM 3.0 체크포인트를 사용합니다. 다른 위치에 체크포인트를 설치했다면 `--checkpoint-3` 또는 `FIRMS_BACKEND_TIMESFM_CHECKPOINT`로 경로를 지정합니다. 기존 LightGBM 비교 모델은 `--method lightgbm_calibrated`로 직접 선택할 수 있습니다. 공식 체크포인트와 연구 출력은 비상업적·비운영 용도입니다. [TimesFM 3.0 라이선스](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)

TimesFM은 격자의 검출 여부와 로그 FRP를 두 시계열로 받아 직접 예측합니다. 입력은 일별 64일 또는 완결된 월별 36개월입니다. 검출 확률은 0~1로 제한한 분위수의 적분으로 추정하며 별도 확률 보정을 적용하지 않습니다. 유효 FRP 이력이 없는 격자의 FRP는 `null`입니다. 월별 실행은 기준 달 첫 14일의 자료까지 확보한 뒤 다음 달부터 12개월을 출력합니다.

`--bbox`를 생략하면 전 세계 후보 격자를 계산합니다. 범위를 지정하면 그 경계 안에 중심이 있는 원래 1km 격자를 예측합니다. `--data`로 갱신한 통합 아카이브를 선택하고, 같은 아카이브의 기존 작업 캐시는 `--reuse-cache`로 읽습니다. 위 예제의 결과를 백엔드에서 읽으려면 `FirmsStore(asset_root="next")`로 열면 됩니다.

```python
from firms_backend.forecasting.predict import predict_from_cache

predict_from_cache("annual", "work/monthly", "next/predictions/annual",
                   bbox=(124, 33, 132, 39))
```

### 과거 데이터 검증

각 기준일에 전 세계 후보 격자에서 5,000개를 무작위로 뽑고, 기준일 이후의 관측과 대조했습니다. 비교 기준은 각 격자의 입력 기간 중 검출 빈도입니다.

| 예측 | 검증 행 | Brier (낮을수록 좋음), TimesFM / 검출 빈도 | 평균 정밀도 (높을수록 좋음), TimesFM / 검출 빈도 |
|---|---:|---:|---:|
| 일별 7일 | 105,000 | 0.02121 / 0.01489 | 0.05695 / 0.08775 |
| 월별 12개월 | 120,000 | 0.08204 / 0.07253 | 0.14323 / 0.14051 |

일별 검증 기준일은 2026-09-14, 2026-10-14, 2026-11-07 UTC이고 월별 기준월은 2024-09, 2025-09입니다. 검출과 FRP 실측 및 예측이 함께 있는 행에서 FRP 평균 절대 오차는 일별 5.80 MW, 월별 11.26 MW였습니다.

## GitHub 데이터 자산 구성

전체 아카이브와 예측 파일은 약 24 GiB입니다. `release-manifest.json`에는 파일 크기와 SHA-256이 있습니다. 작성된 데이터 자산은 로컬 전달본의 `.release-assets/`에 있습니다. 같은 태그의 GitHub Release에 tar를 업로드한 뒤에 원격 `install`을 사용할 수 있습니다.

```sh
git push origin HEAD:refs/heads/handoff/firms-1km-backend
gh release create firms-1km-backend-2026-11-14 firms_backend/.release-assets/*.tar \
  --repo Nasa-Planetarium/Planetarium-AI --target handoff/firms-1km-backend \
  --title "FIRMS 1km backend data" --notes "Harmonized dataset and research forecast snapshots" \
  --latest=false
```
