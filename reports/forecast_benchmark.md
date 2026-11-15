# 지역·기간별 열원 탐지 확률

월말마다 실제 관측을 반영해 다음 달의 모든 날짜를 평가한 최신 비교는 [월별 순차 검증 보고서](forecast_monthly_backtest.md)에 정리했습니다.

요청한 각 지역에서 기준 열원이 한 번 이상 탐지될 확률을 날짜마다 출력합니다. 한 스텝은 받은 `harmonized_cell_day.parquet`의 관측 간격인 **현지 태양시 1일**입니다. 기준일 다음 날이 `step=1`입니다. 지정 기간이 더 늦게 시작하면 중간 스텝을 계산한 뒤 요청한 날짜의 행만 반환합니다.

## 조회 예시

점 좌표는 `[경도, 위도]`, 사각 범위는 `[서, 남, 동, 북]`입니다. `bbox`는 한 범위, `bboxes`는 같은 ID에 속한 여러 범위의 합집합입니다. 여러 지역을 따로 보려면 ID를 따로 지정합니다.

```python
from planetarium import store

probabilities = store.occurrence_forecast(
    [
        {"id": "uiseong", "point": [128.70, 36.35]},
        {"id": "two_zones", "bboxes": [[128.4, 36, 129.6, 36.8], [-63, -34, -61, -32]]},
    ],
    start="2026-11-15",
    end="2026-11-21",
)
```

`region_id`, `as_of_date`, `date_local`, `step`, `p_any`, `p_none`, `n_cells`, `model`을 반환합니다. 시작일과 종료일을 모두 포함합니다. `as_of`의 기본 상한은 요청 시작일의 전날이고, 지정한 지역들의 위성 관측이 모두 정상인 마지막 날을 공통 기준일로 사용합니다. 이 배포판에서 동반구 단독으로는 2026년 11월 13일, 서반구가 포함되면 11월 12일이 마지막 완전 관측일입니다.

점에는 그 점이 속한 0.5° 격자를 적용합니다. 범위에는 육지 격자 중 양의 면적으로 겹치는 칸을 적용합니다. 범위를 겹쳐 요청해도 합집합의 칸은 한 번씩 포함합니다. 합집합에서 실제 한 건 이상 탐지된 날짜를 `1`로 만들어 해당 범위의 과거 발생률과 상태 전이율을 계산합니다. 경계에 걸치는 칸은 그 칸 전체를 사용합니다.

같은 형식의 배열을 `regions.json`으로 저장하면 CLI에서도 실행할 수 있습니다.

```bash
uv run python -m planetarium.modeling.forecast_benchmark prepare
uv run python -m planetarium.modeling.forecast_benchmark benchmark --threads 12 --device cuda
uv run python -m planetarium.modeling.forecast_benchmark forecast \
  --regions regions.json --start 2026-11-15 --end 2026-11-21 --output probabilities.parquet
```

`forecast`는 필요한 일별 패널을 자동으로 준비합니다. `benchmark`는 학습 모델·비교 지표를 별도로 저장합니다. 선택 모델 파일이 없는 경우에는 지역별 이력으로 2차 계절 Markov 확률을 계산합니다. CPU에서 학습하려면 `benchmark --device cpu`를 사용합니다. 입력 경로와 캐시 경로는 서브커맨드 앞에 `--root`, `--work`로 지정할 수 있습니다.

## 학습과 검증

- 2012년 2월 이후 기준 감지점이 하루에 한 번 이상 있으면 `1`입니다. v1.2 고정 열원 제외가 적용된 `n_points`에서 S-NPP와 전환 이후 NOAA-20의 기준 감지를 사용합니다. 위성 상태표의 완전 관측일을 사용하며 현지 날짜가 걸치는 UTC 날짜를 함께 확인합니다.
- 계절 발생률과 1·2차 Markov를 기준선으로 사용합니다. 로지스틱 회귀와 XGBoost는 격자와 여러 크기의 영역에서 다음 일별 상태의 확률을 학습합니다. 모델 학습 기간은 2016–2022년이고 2023년 자료에서 Platt 확률 보정을 적합합니다. 기준일이 속한 해 이전의 관측으로 계절 통계를 계산하며, 최근 이진 상태는 기준일까지의 관측을 사용합니다.
- 예측은 직전 두 스텝의 상태 확률을 다음 스텝으로 전달합니다. 예를 들어 내일 열원이 탐지되는 경우와 탐지되지 않는 경우를 각 확률로 가중해 그다음 날을 계산합니다. 예측 도중 실제 미래 관측값을 입력하지 않습니다.
- 전 세계 육지 격자 및 2° 사각 범위의 정답을 비교합니다. 기준일을 14일 간격으로 이동하며 1·2·3·7·14스텝의 Brier score, log loss, PR-AUC를 기록합니다. 2023년은 기본 모델 선택, 2024–2025년은 홀드아웃, 2026년 1–7월은 추가 검증입니다.
- v1.2 전 세계 0.5° 육지 격자 69,215개를 평가했습니다. 배포 자료에서 날짜별 정답을 대조했고, 위 예시의 다중 지역 요청과 전체 연속 예측에서 같은 날짜의 확률이 일치하는 것을 확인했습니다.
- 데이터 준비 결과, 학습 모델, `metrics.csv`는 `data/interim/daily_occurrence/`에서 배포 파일 묶음별 하위 폴더에 저장됩니다.

<!-- daily-occurrence-results:start -->

검증 구간에서 선택한 기본 모델은 **XGBoost 전이 모델**입니다.

### 2024–2025 전 세계 홀드아웃

| 모델 | 격자 1스텝 Brier | 3스텝 | 7스텝 | 14스텝 | 격자 1스텝 PR-AUC | 범위 1스텝 Brier | 범위 7스텝 Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| 계절 발생률 | 0.03545 | 0.03440 | 0.03444 | 0.03543 | 0.64214 | 0.08196 | 0.08160 |
| 1차 계절 Markov | 0.03248 | 0.03387 | 0.03438 | 0.03538 | 0.68953 | 0.07360 | 0.08143 |
| 2차 계절 Markov | 0.03220 | 0.03337 | 0.03422 | 0.03532 | 0.69422 | 0.07285 | 0.08090 |
| 로지스틱 전이 모델 | 0.03288 | 0.03472 | 0.03587 | 0.03726 | 0.68982 | 0.07469 | 0.08478 |
| XGBoost 전이 모델 | 0.03190 | 0.03329 | 0.03425 | 0.03533 | 0.69890 | 0.07243 | 0.08113 |

<!-- daily-occurrence-results:end -->

## Hugging Face 사전학습 모델 비교

Hugging Face의 네 체크포인트를 같은 일별 발생 문제에서 비교했습니다.

- TimesFM 3.0 [모델 카드](https://huggingface.co/google/timesfm-3.0-pytorch)
- Chronos-2 [모델 카드](https://huggingface.co/amazon/chronos-2)
- Chronos-Bolt Base [모델 카드](https://huggingface.co/amazon/chronos-bolt-base)
- TiRex-2 [모델 카드](https://huggingface.co/NX-AI/TiRex-2)

각 모델에는 기준일까지의 64일 이진 감지 이력을 입력했습니다. 이력의 관측 공백은 해당 시점보다 앞선 마지막 정상 관측 상태로 채웠습니다. 기준일에서 14개 스텝의 예측을 한 번에 만들고 기존 실험과 같은 1·2·3·7·14스텝을 채점했습니다. 테스트 구간의 미래 관측은 채점에만 사용했습니다.

사전학습 가중치를 고정하고, 2023년 우리 데이터로 **격자·범위 및 리드별 탐지 확률 보정기**를 적합했습니다. 보정기는 0.1·0.5·0.9 분위수와 최근 1·2·7·64일 탐지율을 입력받는 로지스틱 회귀입니다. 분위수 없이 같은 탐지율만 입력한 보정기도 기준선으로 계산했습니다.

전 세계 0.5° 격자 69,215개와 2° 범위 5,347개를 대상으로 2023년 26개 기준일에서 보정기를 학습하고, 2024–2025년 52개 기준일과 2026년 1–7월 15개 기준일에서 평가했습니다. 기준일과 정답 날짜가 모두 완전 관측인 경우만 채점했습니다. 동일한 64일 관측 패턴 1,363,899개를 각각 한 번 추론한 후 6,486,894개 지역·기준일 원본 행에 매핑했습니다. 이 방식은 동일 입력의 반복 계산만 줄입니다.

2024–2025년 홀드아웃의 1스텝 격자 평가 행 중 65.66%는 최근 64일이 모두 미탐지 상태였습니다. 각 체크포인트에서 이런 행은 같은 입력으로 같은 확률을 받습니다. 계절과 누적 지역 이력을 사용하는 기존 전이 모델과 입력 정보의 범위가 다릅니다.

### 2024–2025 전 세계 홀드아웃

| 모델 | 격자 1스텝 Brier | 7스텝 | 14스텝 | 격자 1스텝 PR-AUC | 범위 1스텝 Brier | 7스텝 | 14스텝 |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost 전이 모델 | 0.03190 | 0.03425 | 0.03533 | 0.69890 | 0.07243 | 0.08113 | 0.08063 |
| 2차 계절 Markov | 0.03220 | 0.03422 | 0.03532 | 0.69422 | 0.07285 | 0.08090 | 0.08058 |
| 최근 탐지율 보정 | 0.03413 | 0.03807 | 0.04114 | 0.67253 | 0.07655 | 0.08789 | 0.09278 |
| TimesFM 3.0 | 0.03339 | 0.03688 | 0.03974 | 0.67362 | 0.07496 | 0.08541 | 0.09006 |
| Chronos-2 | 0.03378 | 0.03734 | 0.04028 | 0.66978 | 0.07557 | 0.08632 | 0.09107 |
| Chronos-Bolt Base | 0.03362 | 0.03728 | 0.04022 | 0.67031 | 0.07536 | 0.08627 | 0.09094 |
| TiRex-2 | 0.03376 | 0.03702 | 0.03992 | 0.67127 | 0.07546 | 0.08562 | 0.09041 |

1스텝은 격자 3,295,553행과 범위 254,533행을 같은 정답으로 채점했습니다. TimesFM 3.0은 사전학습 모델 가운데 두 공간 단위의 평가 리드 모두에서 Brier가 가장 낮았습니다. 전체 모델을 비교하면 XGBoost가 두 공간 단위 모두 1스텝에서 앞섰고, 2차 계절 Markov가 7·14스텝에서 근소하게 앞섰습니다. 2026년 추가 검증 및 전체 리드의 Brier, log loss, PR-AUC도 CSV에 기록했습니다. [전체 지표](forecast_foundation_occurrence_metrics.csv)

### 재현

CUDA 12.8용 PyTorch를 설치한 별도 실험 환경에서 실행했습니다. 체크포인트 리비전은 실행 코드에 고정되어 있습니다. TimesFM 3.0의 CUDA 추론에는 BF16 혼합 정밀도를 적용했습니다.

```bash
forecast_env="${XDG_CACHE_HOME:-$HOME/.cache}/planetarium-forecast-hf"
UV_PROJECT_ENVIRONMENT="$forecast_env" uv sync --locked --no-dev --python 3.12
forecast_python="$forecast_env/bin/python"
uv pip install --python "$forecast_python" torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$forecast_python" timesfm==3.0.2 chronos-forecasting==2.3.2 tirex-2==0.5.0 transformers==5.23.0

"$forecast_python" -m planetarium.modeling.foundation_occurrence_benchmark prepare
"$forecast_python" -m planetarium.modeling.foundation_occurrence_benchmark history --device cpu
"$forecast_python" -m planetarium.modeling.foundation_occurrence_benchmark timesfm3 --batch-size 1536
"$forecast_python" -m planetarium.modeling.foundation_occurrence_benchmark chronos2 --batch-size 384
"$forecast_python" -m planetarium.modeling.foundation_occurrence_benchmark chronos_bolt --batch-size 512
"$forecast_python" -m planetarium.modeling.foundation_occurrence_benchmark tirex2 --batch-size 512
```

보정기 계수와 모델별 지표는 `data/interim/daily_occurrence/<fingerprint>/foundation-64-v1/`에 생성됩니다. 이 실험의 보정기는 지역·기간 조회의 예측 후보를 평가하는 용도로 사용했습니다.
