# FIRMS 1km 백엔드 전달 패키지

| 경로 | 내용 |
|---|---|
| `dataset/` | 2000-11-01~2026-11-14 UTC 전 세계 통합 데이터. 개별 관측 660,831,145건, 1km 격자·일 357,814,169행, 격자 경계와 월·시간 요약. |
| `converter/` | FIRMS MODIS 또는 VIIRS CSV를 동일한 스키마로 바꾸는 코드와 학습 모델. |
| `forecasting/` | 1km 일별 7일 및 월별 12개월 예측 코드. `forecasting/models/`에 LightGBM 검출, FRP, 확률 보정 모델이 들어 있습니다. |
| `predictions/` | TimesFM 3.0을 결합한 연구 실행 결과. `daily/`는 2026-11-15~21일, `annual/`은 2026-12~2027-11월 예측입니다. |

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

## 변환과 예측 다시 실행

```sh
python -m firms_backend.converter.harmonize --input input.csv --output converted/input
python -m firms_backend.forecasting.predict daily --cache work/daily --output next/daily
python -m firms_backend.forecasting.predict annual --cache work/monthly --output next/annual
```

`predict`는 포함된 LightGBM 보정 모델로 실행됩니다. Hugging Face, GPU 또는 PyTorch 의존성이 없습니다. 예측에는 배포본처럼 다년간의 1km 통합 시계열이 필요합니다. 확장한 관측 아카이브의 경로를 `--data`로 지정할 수 있습니다. 같은 아카이브에서 이미 만든 작업 캐시는 `--reuse-cache`로 읽습니다. 월별 입력은 기준 달 첫 14일이 확보되어야 하며, 그 다음 달부터 12개월을 예측합니다. 재생성 결과를 조회할 때는 생성 폴더를 서비스의 `predictions/daily` 또는 `predictions/annual`에 배치합니다. `forecasting.train_models`로 동일 모델을 학습할 수 있습니다.

`predictions/`의 기존 예측은 실험에서 사용한 LightGBM+TimesFM 3.0 결합 방식입니다. 연구 실행을 재현하려면 별도 가상환경을 사용합니다. 공식 체크포인트와 연구 출력은 비상업적·비운영 용도입니다. [TimesFM 3.0 라이선스](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)

```sh
python -m pip install -r firms_backend/requirements-timesfm.txt
hf download google/timesfm-3.0-pytorch \
  --revision 43046b85ec22d584a13f8098c2ed39c889e129c2 \
  --local-dir firms_backend/timesfm-checkpoint
python -m firms_backend.forecasting.grid_1km --data firms_backend/dataset \
  --cache work/daily --output research/daily --checkpoint-3 firms_backend/timesfm-checkpoint
python -m firms_backend.forecasting.monthly_cells --source firms_backend/dataset --cache work/monthly
python -m firms_backend.forecasting.year_1km --cache work/monthly \
  --output research/annual --checkpoint-3 firms_backend/timesfm-checkpoint
```

## GitHub 데이터 자산 구성

전체 아카이브와 예측 파일은 약 24 GiB입니다. `release-manifest.json`에는 파일 크기와 SHA-256이 있습니다. 작성된 데이터 자산은 로컬 전달본의 `.release-assets/`에 있습니다. 같은 태그의 GitHub Release에 tar를 업로드한 뒤에 원격 `install`을 사용할 수 있습니다.

```sh
git push origin HEAD:refs/heads/handoff/firms-1km-backend
gh release create firms-1km-backend-2026-11-14 firms_backend/.release-assets/*.tar \
  --repo Nasa-Planetarium/Planetarium-AI --target handoff/firms-1km-backend \
  --title "FIRMS 1km backend data" --notes "Harmonized dataset and research forecast snapshots" \
  --latest=false
```
