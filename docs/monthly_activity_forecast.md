# 과학자 탭의 다음 달 연소 활동 예측

**권장 모델은 관측 0을 유지하고 달력 월 합계를 학습한 LightGBM L1입니다.** 예측 분석 탭에 사용자 지정 지역의 다음 달 S-NPP 기준 환산 탐지량을 하나의 숫자로 전달합니다. 예를 들어 관측 기준일이 2026-11-12이면 대상은 2026-12-01부터 2026-12-31까지입니다. [Emberline 예측 분석](https://emberline-9sd.pages.dev/ko/lab/)

월간 모델은 백테스트에서 검증했습니다. 아래 응답은 백엔드 연동안입니다. 현재 공개된 Python 조회 함수 `store.value_forecast`는 관측 기준일 다음 날부터 일별 값 30개를 반환합니다. [기존 30일 예측](../reports/forecast_benchmark.md)

## 예측 단위와 입력

- `value`는 대상 월 전체의 S-NPP 기준 환산 탐지량입니다. 학습 정답은 v1.2의 `harmonized_cell_month.value`를 지역 안에서 합산한 값입니다. 월 보정분은 이 컬럼에 이미 반영되어 있습니다. [릴리스 v1.2](release_v1.2-2026-11-15.md) · [월 집계 계약](api_contract.md#summaryaoibboxpolygonstartendfreqdaymonth)
- 지역은 하나의 사각 범위 `bbox` 또는 여러 사각 범위의 합집합 `bboxes`로 정의할 수 있습니다. 0.1° 격자 중심이 범위 안에 있는 칸을 합산하고 중복 칸은 한 번만 셉니다. 검증은 전 세계 격자 및 사각 범위에서 수행했습니다. 기존 화면의 행정구역 경계를 쓰는 경우 과거 실측과 예측 양쪽에 동일한 지역 집계 규칙을 적용해야 합니다.
- 일별 관측에서 기준일까지의 최근 3·7·14·30·90일 평균, 탐지 빈도, 연중 시기, 지역 크기를 사용합니다. 확정 월 데이터에서는 대상 월의 직전 달을 제외하고 과거 같은 월의 평균·중앙값과 최근 완결 월 평균을 사용합니다. 직전 달의 진행 상황은 기준일까지 확인된 일별 관측으로 반영합니다.
- 위성 상태상 정상으로 확인된 일별 관측을 사용합니다. 달력 월 학습에는 `status=ok` 또는 `nrt`인 월별 자료를 사용합니다. 모델 학습에서 소수의 음수 라벨은 0으로 하한 처리했고, 평가에는 릴리스 원래 `value`를 사용했습니다.

## 권장 근거

2018~2024년의 지역·월 라벨 1,188,348건으로 학습했습니다. 이 중 66.37%가 0이었습니다. 평가 기간은 2025년 전체와 2026년 1~10월입니다. 매 회차 직전 달 12일 또는 직전 달 말일을 상한으로 가장 최근에 동·서경 자료가 모두 확인된 날을 기준일로 삼았습니다. 비교에 사용한 지역은 0.5° 사각형 4,000개, 2° 사각형 5,347개, 1.3°×0.7° 사각형 1,800개입니다. 0.1° 단일 격자 3,000개 결과도 CSV에 따로 있습니다.

WAPE는 절대 오차 합을 실제 탐지량 합으로 나눈 값이며 낮을수록 좋습니다. 각 연도·지역 크기별 WAPE를 계산한 뒤 세 지역 크기에 같은 비중을 주었습니다.

| 예측 방식 | 월중 2025 | 월중 2026년 1~10월 | 월말 2025 | 월말 2026년 1~10월 |
|---|---:|---:|---:|---:|
| **월간 LightGBM L1 · 관측 0 유지** | **49.12%** | **51.96%** | 46.79% | 49.43% |
| 월간 LightGBM L1 · 학습 0의 50% 제거 | 49.13% | 52.17% | **46.51%** | 49.57% |
| 월간 LightGBM L1 · 학습 0의 80% 제거 | 49.35% | 52.54% | 46.63% | **49.35%** |
| 과거 최대 5년 같은 달 중앙값 | 52.71% | 55.49% | 52.71% | 55.49% |
| 기존 일별 LightGBM L1 · 대상 월까지 연장 후 합산 | 55.04% | 56.02% | 49.08% | 50.30% |

월중 발표에서는 관측 0을 유지한 버전이 두 기간 모두 가장 낮았습니다. 월말 발표에서는 LightGBM L1 세 버전 간 차이가 0.3%p 미만이었습니다. 권장 모델의 월중 합계 편향은 2025년 −18.11%, 2026년 1~10월 −19.75%여서 탐지량을 낮게 예측하는 경향이 있습니다. 현재 비교는 같은 지역의 미래 월을 평가했습니다. 지역 크기나 경계가 크게 달라지는 경우의 성능과 80% 예측 구간은 별도 검증이 필요합니다. [모델·기간·공간 규모별 WAPE, MAE, 편향](../reports/forecast_calendar_month_metrics.csv)

## 백엔드 전달 형식

지역과 대상 월 조합마다 결과 한 건을 반환합니다. 아래 수치는 형식을 보여 주기 위한 예시입니다.

```json
{
  "region_id": "selected-area",
  "target_month": "2026-12",
  "as_of_date": "2026-11-12",
  "value": 123.4
}
```

`target_month`는 `YYYY-MM`이고, `as_of_date`는 모델이 읽은 최신 현지 태양일입니다. 프론트엔드는 `value`를 월별 추이의 중앙값 위치에 표시하고, 같은 지역의 실측 월 합계와 비교합니다. 월간 80% 예측 구간을 표시하려면 월 합계에 대해 별도로 보정한 구간이 필요합니다.

## 검증 재현

[v1.2 릴리스](https://github.com/Nasa-Planetarium/Planetarium-AI/releases/tag/data-v1.2-2026-11-15)의 `harmonized_cell_day.parquet`, `harmonized_cell_month.parquet`, `satellite_day_status.parquet`, `aoi_boundaries.geojson`를 `data/processed/`에 설치합니다.

```bash
uv sync --locked --no-dev
uv run --locked --no-dev python -m planetarium.modeling.calendar_value_benchmark
```

기본 출력은 `data/interim/value_forecast/calendar_value_metrics.csv`입니다. 두 발행 시점의 월간 모델 여섯 설정과 평년 중앙값을 평가합니다. 시점 하나만 실행하려면 `--issue-day 12` 또는 `--issue-day 31`을 넘깁니다. 기존 일별 모델을 연장한 비교 행은 해당 체크포인트 폴더를 지정하면 추가됩니다.

```bash
uv run --locked --no-dev python -m planetarium.modeling.value_benchmark \
  --models lightgbm_l1_zero20 xgboost_l1_zero20 xgboost
uv run --locked --no-dev python -m planetarium.modeling.calendar_value_benchmark \
  --daily-model-dir data/interim/value_forecast/results
```

기록한 비교 지표에는 기존 실험에서 학습한 일별 체크포인트를 사용했습니다. XGBoost 일별 모델은 RTX 4090에서 학습했습니다. 월간 벤치마크는 CPU에서 실행합니다. 일별 모델을 연장한 비교는 30일을 넘는 시차도 사용한 실험입니다.
