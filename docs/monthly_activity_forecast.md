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

## `main`의 FireOutlook Planner와 비교

`main`의 FireOutlook Planner는 `POST /forecast`에 AOI, bbox 또는 GeoJSON을 받고 기간 합계 `expected_count`를 제공합니다. 다음 달 전체를 조회할 때 bbox의 건수 경로는 0.5° 격자별 XGBoost Tweedie 예측을 면적 비율로 합산합니다. 등록된 AOI는 별도 건수 경로를 쓸 수 있습니다. 실제 v1.2 산출물에서 `[128, 36, 130, 38]`의 2026년 12월 조회는 기준일 2026-11-14, `expected_count=7.73`, `count_source=model_mu`였습니다. [main API 계약](https://github.com/Nasa-Planetarium/Planetarium-AI/blob/c43368c62bd8253879c0547527bdb9441b36aa1a/docs/api_contract.md#L314)

기존 벤치마크와 같은 사각 범위 11,147개 및 같은 확정 월 `value`에서 비교했습니다. 그중 2,010개에는 Planner의 서비스 격자가 겹치지 않아 Planner의 빈 지역 처리와 같이 0을 합산했습니다. 수치는 지역 규모 세 종류의 평균 WAPE입니다.

| 방식 | 월중 2025 | 월중 2026년 1~10월 | 월말 2025 | 월말 2026년 1~10월 |
|---|---:|---:|---:|---:|
| 작업 브랜치 월간 LightGBM L1 · 관측 0 유지 | **49.12%** | **51.96%** | **46.79%** | **49.43%** |
| main Planner의 월간 건수 모델 | 58.08% | 61.95% | 58.37% | 61.01% |
| main의 격자별 예년 같은 달 중앙값을 면적 합산 | 51.50% | 54.49% | 51.50% | 54.49% |

- `main`의 `c43368c` 코드와 하이퍼파라미터로 XGBoost Tweedie 건수 모델을 재학습했습니다. 릴리스와 같은 52,855개 서비스 격자 및 `grid_occurrence` 특성을 썼습니다. 2012년 2월~2023년을 학습에, 2024년을 조기 종료에 사용해 두 시험 연도의 건수 모델을 고정했습니다. 작업 브랜치 모델도 2024년까지의 라벨로 학습했습니다.
- 월중이나 관측 공백이 있는 월말의 직전 달 값은 기준일까지 관측된 월 누계를 관측일 수로 월말까지 환산해 재현했습니다. 이 값은 당시 서비스의 역사적 스냅샷 대신 사용한 추정입니다. 월말 표는 Planner의 **월간 건수 성분**을 비교합니다. 실제 API의 일별 파일이 대상 월 첫 7일 일부를 덮으면 Planner는 그 날짜에 일별 성분을 사용할 수 있습니다.
- 월중 기준에서 규모 평균 합계 편향은 LightGBM L1이 −18.11% / −19.75%, Planner 건수 모델이 −1.63% / −5.27%였습니다. 개별 지역의 월 합계 절대 오차를 중시하는 화면에는 LightGBM L1이 유리했고, 전체량 편향은 Planner가 작았습니다. 결과 범위는 검증한 사각 범위까지입니다. 등록 AOI, 국가, 대륙은 Planner의 별도 경로와 같은 조건에서 평가해야 합니다. [동일 조건 비교 CSV](../reports/forecast_calendar_main_comparison.csv)

## 월간 LightGBM과 Planner의 앙상블

같은 지역·대상 월의 LightGBM L1 예측과 Planner 월간 건수 예측을 두 입력으로 사용했습니다. 예측은 `(1 − w) × LightGBM + w × Planner`입니다. 두 기본 모델은 2024년까지의 라벨을 사용했습니다. 혼합 가중치는 지역 범위 11,147개의 2025년 예측과 실측값으로 학습하고, 2026년 1~10월은 가중치를 고정해 평가했습니다. 월중·월말은 각각 계수를 학습했습니다.

- **고정 혼합**은 전 지역에 Planner 가중치 하나를 씁니다. **적응 혼합**은 지역 크기 세 종류와 Planner가 LightGBM보다 큰지 여부에 따라 가중치를 학습합니다.
- 가중치는 0~1 범위에서 지정하고, 2025년 규모 평균 합계 편향의 절댓값을 10% 이내로 맞추면서 WAPE를 최소화했습니다. 10%는 실험에서 선택한 학습 기준입니다. 비교용으로 LightGBM만 일정 비율로 확대해 편향을 줄이는 방식도 학습했습니다.
- 월중 적응 혼합에서 Planner가 LightGBM보다 높을 때의 Planner 비중은 0.5° 범위 37.83%, 2° 범위 40.18%, 1.3°×0.7° 범위 30.09%였습니다. 낮거나 같으면 각각 약 0%, 22.11%, 약 0%였습니다. [학습한 계수](../reports/forecast_calendar_ensemble_weights.json)

| 모델 | 월중 WAPE | 월중 합계 편향 | 월말 WAPE | 월말 합계 편향 |
|---|---:|---:|---:|---:|
| LightGBM L1 | **51.96%** | −19.75% | **49.43%** | −20.72% |
| 적응 혼합 | 54.02% | −12.29% | 50.08% | −15.84% |
| 고정 혼합 | 55.03% | −12.63% | 50.05% | −15.88% |
| LightGBM 단독 비율 보정 | 54.49% | −11.80% | 50.38% | −16.54% |
| Planner 월간 건수 | 61.95% | −5.27% | 61.01% | +3.51% |

표는 2026년 1~10월 홀드아웃입니다. 월중 적응 혼합은 LightGBM 대비 WAPE가 2.05%p 높았고, 과소예측은 7.47%p 줄었습니다. 비율 보정과 비교하면 월중 WAPE는 0.47%p 낮고 과소예측은 0.48%p 컸습니다. 개별 지역의 월 합계 절대 오차를 최소화하는 목적에는 LightGBM L1이 가장 좋았고, 과소예측도 줄이는 절충안에는 월중 적응 혼합이 알맞습니다. 적응 혼합의 2026년 편향 절댓값은 두 시점 모두 10%를 넘었습니다. [모델·기준일·지역 크기별 지표](../reports/forecast_calendar_ensemble_metrics.csv)

재현에는 앞의 Planner 비교에 사용한 과거 격자 예측 `pred_main_12.parquet`, `pred_main_31.parquet`가 필요합니다. 파일은 2025년 1월~2026년 10월의 `year`, `month`, 0.5° 격자 ID `g`, 월별 건수 추정 `mu`를 담으며, 현재 작업 폴더에서는 `data/interim/calendar_ensemble/`에 있습니다. 명령은 LightGBM을 다시 학습하고 사각 범위의 Planner 예측을 합산한 뒤 지표 CSV와 혼합 가중치 JSON을 생성합니다. 결과는 백테스트용입니다.

```bash
uv run --locked --no-dev python -m planetarium.modeling.calendar_value_ensemble \
  --planner-dir data/interim/calendar_ensemble
```

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
