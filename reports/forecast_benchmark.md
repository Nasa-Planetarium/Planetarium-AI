# 월간 열점 탐지량 예측 실험

NASA Planetarium v1.2 배포 파일의 월간 S-NPP 기준 탐지 환산량을 예측합니다. 전 세계 합계 `WORLD`는 배포 파일에 있는 전 세계 격자의 집계값입니다. [v1.2 배포 설명](../docs/release_v1.2-2026-11-15.md)

## 2차 비교 결과 — 2026년 11월 15일

- **전 세계 합계 홀드아웃 1위는 AutoTheta입니다.** 10개를 추가해 총 27개 모델을 비교했습니다. AutoTheta의 2024–2025년 WAPE는 **10.89%**였고, 기준선 중 가장 낮았던 최근 5년 같은 달 평균(13.97%)보다 22.0% 낮았습니다. 1개월과 2개월 앞에서는 AutoTheta가 각각 9.54%, 10.49%였고, 3개월 앞에서는 새 후보인 AutoETS가 12.50%로 가장 낮았습니다.
- 2020–2023년에는 Chronos-Bolt Base가 10.37%로 가장 낮았고, 2024–2025년에는 15.47%였습니다. 2026년 1–7월 정식 처리본에서 가장 낮은 WAPE는 TimesFM 3.0의 **5.79%**였습니다.
- 2024–2025년 대륙·해역 8곳의 절대오차를 합친 WAPE는 AutoETS가 **22.50%**, 국가 단위 249곳에서는 Timestral 209M이 **29.95%**로 가장 낮았습니다. 5,051개 지역에 동일 가중치를 준 홀드아웃 log MAE는 N-HiTS가 **0.4511**로 가장 낮았고, 전 세계 합계 WAPE는 **38.48%**였습니다.
- 저장된 예측과 원본 Parquet에서 27개 모델의 평가 예측 32,321,349건을 대조했습니다. 전 세계 합계, 대륙·해역, 국가 단위 지표와 전체 지역 log MAE를 다시 계산했습니다.

### 전 세계 합계 전망

2026년 10월 말까지의 관측을 사용했습니다. 단위는 S-NPP 기준 탐지 환산량이며 반올림했습니다.

| 모델 | 2026년 11월 | 2026년 12월 | 2027년 1월 |
|---|---:|---:|---:|
| AutoTheta | 1,134,807 | 1,291,413 | 1,496,569 |
| AutoETS | 1,124,411 | 1,276,249 | 1,477,978 |
| TimesFM 3.0 | 1,102,787 | 1,241,181 | 1,479,123 |

## 2차 비교 조건

- `aoi_list.parquet`의 5,051개 지역 전체를 평가합니다. 이전 비교에서 빠졌던 숨김 지역 63곳도 포함했습니다. 각 지역에 2012년 2월–2026년 10월 177개월의 연속 이력이 있습니다.
- 전 세계 합계 `WORLD`의 WAPE로 후보를 비교합니다. 8개 대륙·해역과 국가 단위 249곳에는 집단 WAPE를, 전체 5,051개 지역에는 동일 가중치의 log MAE를 계산합니다. 여러 행정 수준의 지역은 공간적으로 중첩됩니다.
- 이전과 같은 예측 대상월로 비교합니다. 2020–2023년 검증, 2024–2025년 홀드아웃, 2026년 1–7월 추가 정식 처리본을 1·2·3개월 리드에서 각각 채점합니다. 모델마다 지역별 1,197,087건과 그 안에 포함된 `WORLD` 237건을 평가합니다. 2026년 8–10월의 NRT는 10월 말 기준 전망의 마지막 입력으로 사용합니다.
- 전 세계 합계 WAPE는 예측과 실적의 절대차 합을 실적 합으로 나눈 값입니다. 지역별 log MAE는 `abs(log1p(예측) - log1p(실적))`의 평균입니다. 배포 자료에 남은 음수 143건과 음수 예측은 0으로 처리했습니다.
- 기존 모델과 새 후보를 모두 같은 패널에서 실행했습니다. 학습에는 예측 기준월까지 확보된 관측을 사용했습니다. AutoETS는 연말에 매개변수를 적합하고 월별 관측으로 상태를 갱신했습니다. SARIMA는 로그 환산량에 대해 `(1,0,0)×(0,1,0)[12]` 구조를 사용했습니다.
- 추가 후보는 AutoETS, SARIMA, Croston-SBA, CatBoost, TiDE, PatchTST, TimesFM 2.5, Chronos-Bolt Base, Toto 2.0 22M, Moirai 2.0 Small입니다. [Nixtla 통계 모델](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html) · [CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor) · [TiDE](https://nixtlaverse.nixtla.io/neuralforecast/models.tide.html) · [PatchTST](https://nixtlaverse.nixtla.io/neuralforecast/models.patchtst.html)
- 사전학습 가중치의 배포 커밋은 실행 이력에 남깁니다. [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) · [Chronos-Bolt](https://huggingface.co/amazon/chronos-bolt-base) · [Toto 2.0 22M](https://huggingface.co/Datadog/Toto-2.0-22m) · [Moirai 2.0 Small](https://huggingface.co/Salesforce/moirai-2.0-R-small)

2차 산출물은 `data/processed/forecast_benchmark/aoi-month-v2-3e5a46208726/`에 저장합니다. `world_metrics.csv`는 전 세계 합계 지표이며, `metrics.csv`에는 대륙·해역별 세부 지표가 있습니다. 모델별 `backtest.npz`와 `forecast.csv`에는 각각 평가 예측과 2026년 10월 말 기준 전망을 저장합니다. 모델 완료 시 아래 결과표가 자동으로 갱신됩니다.

### 2차 재실행

기존 CPU와 GPU 환경에서 통계 모델과 신경망을 실행할 수 있습니다. CatBoost, Toto, Moirai에는 아래 의존성을 추가했습니다. Toto와 Moirai는 의존성을 고정한 별도 환경에서 실행했습니다.

```bash
forecast_env="${XDG_CACHE_HOME:-$HOME/.cache}/planetarium-forecast"
uv pip install --python "$forecast_env/cpu/bin/python" catboost==1.2.10
uv venv --python 3.12 "$forecast_env/toto"
uv pip install --python "$forecast_env/toto/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
uv pip install --python "$forecast_env/toto/bin/python" toto-2==2.0.0 numpy==2.5.4 polars==1.44.2 filelock
uv venv --python 3.12 "$forecast_env/moirai"
uv pip install --python "$forecast_env/moirai/bin/python" --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
uv pip install --python "$forecast_env/moirai/bin/python" \
  uni2ts==2.0.0 huggingface-hub==0.36.0 pyarrow==20.0.0 lightning==2.5.6 polars==1.44.2 filelock

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$forecast_env/cpu/bin/python" \
  -m planetarium.modeling.forecast_benchmark run --model autoets --threads 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 "$forecast_env/gpu/bin/python" \
  -m planetarium.modeling.forecast_benchmark run --model patchtst --device cuda --threads 3 --batch-size 2048
```

기존 시도와 새 시도를 함께 보존하려면 `--force`를 붙입니다. 평가 당시 모델과 가중치 설정은 실험 폴더의 `runs.jsonl`에 기록되어 있습니다.

<!-- benchmark-world-results:start -->

## 전 세계 합계 성능

| 모델 | 유형 | 2020–2023 WAPE | 2024–2025 WAPE | 1개월 | 2개월 | 3개월 | 2026년 1–7월 WAPE | 홀드아웃 편향 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Chronos-Bolt Base | 사전학습 | 10.37% | 15.47% | 13.58% | 16.09% | 16.74% | 7.84% | 1.44% |
| Chronos-2 | 사전학습 | 10.62% | 12.92% | 11.48% | 13.37% | 13.91% | 6.42% | -0.51% |
| Timestral 209M | 사전학습 | 10.65% | 12.08% | 10.96% | 12.05% | 13.24% | 7.79% | 1.18% |
| TimesFM 2.5 | 사전학습 | 10.82% | 12.91% | 11.59% | 13.20% | 13.94% | 6.20% | -0.04% |
| TimesFM 3.0 | 사전학습 | 10.87% | 12.51% | 11.54% | 12.73% | 13.25% | 5.79% | -0.40% |
| Toto 2.0 22M | 사전학습 | 11.04% | 12.62% | 11.60% | 12.96% | 13.30% | 7.87% | -1.37% |
| AutoTheta | 지역별 학습 | 11.15% | 10.89% | 9.54% | 10.49% | 12.65% | 7.90% | 2.25% |
| AutoETS | 지역별 학습 | 11.38% | 11.86% | 11.34% | 11.75% | 12.50% | 7.07% | -0.31% |
| TiRex-2 | 사전학습 | 11.46% | 13.10% | 11.75% | 13.59% | 13.96% | 8.67% | -0.40% |
| Moirai 2.0 Small | 사전학습 | 11.46% | 14.12% | 13.12% | 14.06% | 15.17% | 7.67% | 2.78% |
| PatchTST | 전체 지역 공동 학습 | 11.53% | 13.40% | 12.35% | 14.19% | 13.65% | 12.58% | -1.88% |
| 최근 5년 같은 달 평균 | 기준선 | 11.83% | 13.97% | 13.97% | 13.97% | 13.97% | 11.52% | -0.29% |
| Ridge | 전체 지역 공동 학습 | 11.85% | 14.01% | 13.23% | 14.61% | 14.21% | 11.53% | -0.86% |
| 계절 지수평활 | 지역별 학습 | 12.12% | 14.79% | 14.79% | 14.79% | 14.79% | 12.33% | 0.20% |
| 최근 5년 같은 달 중앙값 | 기준선 | 12.39% | 15.03% | 15.03% | 15.03% | 15.03% | 10.75% | -0.99% |
| 계절 자기회귀 | 지역별 학습 | 14.43% | 17.98% | 16.46% | 19.01% | 18.46% | 15.16% | -0.48% |
| SARIMA | 지역별 학습 | 14.59% | 16.29% | 14.34% | 16.93% | 17.58% | 13.86% | 6.48% |
| 작년 같은 달 | 기준선 | 16.01% | 18.62% | 18.62% | 18.62% | 18.62% | 15.16% | 8.53% |
| DLinear | 전체 지역 공동 학습 | 19.77% | 20.89% | 20.21% | 27.10% | 15.36% | 20.11% | -13.65% |
| TiDE | 전체 지역 공동 학습 | 21.49% | 27.17% | 27.51% | 24.49% | 29.49% | 24.52% | -2.91% |
| XGBoost | 전체 지역 공동 학습 | 22.02% | 24.53% | 24.67% | 24.24% | 24.69% | 17.69% | -17.41% |
| Croston-SBA | 간헐 시계열 | 27.84% | 31.08% | 29.65% | 31.59% | 31.99% | 27.90% | -0.94% |
| LightGBM | 전체 지역 공동 학습 | 28.51% | 30.94% | 30.88% | 30.94% | 31.00% | 19.55% | -25.50% |
| TSB | 간헐 시계열 | 29.30% | 33.01% | 31.45% | 33.52% | 34.06% | 32.20% | 4.27% |
| N-HiTS | 전체 지역 공동 학습 | 32.23% | 38.48% | 45.71% | 39.53% | 30.21% | 30.12% | 27.18% |
| 직전 달 | 기준선 | 34.14% | 36.84% | 25.26% | 37.99% | 47.27% | 26.86% | 1.71% |
| CatBoost | 전체 지역 공동 학습 | 97.50% | 97.25% | 97.22% | 97.25% | 97.28% | 96.77% | -97.25% |

행은 v1.2의 전 세계 합계 `WORLD`에 대한 2020–2023년 WAPE 순입니다. 1·2·3개월 열은 2024–2025년 홀드아웃입니다.

### 전 세계 지역별 성능

| 모델 | 5,051개 지역 검증 log MAE | 지역별 홀드아웃 log MAE | 지역별 2026년 log MAE | 8개 대륙·해역 WAPE | 249개 국가 단위 WAPE |
|---|---:|---:|---:|---:|---:|
| Chronos-Bolt Base | 0.5636 | 0.5610 | 0.5577 | 25.26% | 33.93% |
| Chronos-2 | 0.5284 | 0.5241 | 0.5250 | 24.13% | 32.08% |
| Timestral 209M | 0.4661 | 0.4572 | 0.4726 | 22.76% | 29.95% |
| TimesFM 2.5 | 0.5694 | 0.5564 | 0.5570 | 24.18% | 32.49% |
| TimesFM 3.0 | 0.5485 | 0.5428 | 0.5395 | 24.05% | 32.83% |
| Toto 2.0 22M | 0.5057 | 0.4980 | 0.5061 | 23.08% | 31.65% |
| AutoTheta | 0.4869 | 0.4728 | 0.4928 | 22.57% | 40.39% |
| AutoETS | 0.4740 | 0.4616 | 0.4845 | 22.50% | 38.35% |
| TiRex-2 | 0.5444 | 0.5421 | 0.5363 | 25.66% | 33.46% |
| Moirai 2.0 Small | 0.5644 | 0.5560 | 0.5600 | 24.38% | 33.95% |
| PatchTST | 0.4639 | 0.4557 | 0.4767 | 23.23% | 32.42% |
| 최근 5년 같은 달 평균 | 0.5294 | 0.5135 | 0.5278 | 24.18% | 33.64% |
| Ridge | 0.4901 | 0.4832 | 0.4959 | 25.17% | 35.77% |
| 계절 지수평활 | 0.5351 | 0.5198 | 0.5353 | 24.74% | 34.15% |
| 최근 5년 같은 달 중앙값 | 0.4749 | 0.4654 | 0.4845 | 24.50% | 34.03% |
| 계절 자기회귀 | 0.5034 | 0.4971 | 0.5087 | 28.33% | 36.08% |
| SARIMA | 0.5637 | 0.5529 | 0.5830 | 32.21% | 44.30% |
| 작년 같은 달 | 0.5759 | 0.5663 | 0.5925 | 34.97% | 47.45% |
| DLinear | 0.4737 | 0.4649 | 0.4812 | 28.32% | 34.65% |
| TiDE | 0.4738 | 0.4655 | 0.4810 | 30.57% | 35.14% |
| XGBoost | 0.4754 | 0.4676 | 0.4856 | 75.80% | 64.98% |
| Croston-SBA | 1.0646 | 1.0629 | 1.0133 | 58.99% | 105.76% |
| LightGBM | 0.4681 | 0.4614 | 0.4787 | 77.18% | 59.93% |
| TSB | 1.0675 | 1.0659 | 1.0139 | 60.26% | 108.01% |
| N-HiTS | 0.4646 | 0.4511 | 0.4731 | 39.11% | 38.81% |
| 직전 달 | 0.9344 | 0.9450 | 0.9347 | 71.73% | 105.36% |
| CatBoost | 0.4793 | 0.4732 | 0.4873 | 86.95% | 64.50% |

## 실행 기록

| 모델 | 상태 | 장치 | CPU 스레드/작업 수 | 시간 | 세부 기록 |
|---|---|---|---:|---|---|
| LightGBM | running | cpu | 7 | 2026-11-15 15:14:24+09:00 |  |
| AutoTheta | running | cpu | 12 | 2026-11-15 15:14:24+09:00 |  |
| Chronos-2 | running | cuda | 2 | 2026-11-15 15:14:24+09:00 |  |
| 직전 달 | running | cpu | 1 | 2026-11-15 15:14:24+09:00 |  |
| TimesFM 3.0 | running | cuda | 2 | 2026-11-15 15:14:24+09:00 |  |
| XGBoost | running | cpu | 8 | 2026-11-15 15:14:24+09:00 |  |
| Timestral 209M | running | cuda | 2 | 2026-11-15 15:14:24+09:00 |  |
| 직전 달 | complete | cpu | 1 | 2026-11-15 15:14:25+09:00 | 0.4초 |
| 작년 같은 달 | running | cpu | 1 | 2026-11-15 15:14:26+09:00 |  |
| 작년 같은 달 | complete | cpu | 1 | 2026-11-15 15:14:26+09:00 | 0.3초 |
| 최근 5년 같은 달 중앙값 | running | cpu | 1 | 2026-11-15 15:14:27+09:00 |  |
| 최근 5년 같은 달 중앙값 | complete | cpu | 1 | 2026-11-15 15:14:28+09:00 | 0.7초 |
| 최근 5년 같은 달 평균 | running | cpu | 1 | 2026-11-15 15:14:29+09:00 |  |
| 최근 5년 같은 달 평균 | complete | cpu | 1 | 2026-11-15 15:14:30+09:00 | 0.6초 |
| 계절 지수평활 | running | cpu | 1 | 2026-11-15 15:14:31+09:00 |  |
| 계절 지수평활 | complete | cpu | 1 | 2026-11-15 15:14:32+09:00 | 0.6초 |
| 계절 자기회귀 | running | cpu | 1 | 2026-11-15 15:14:33+09:00 |  |
| 계절 자기회귀 | complete | cpu | 1 | 2026-11-15 15:14:34+09:00 | 1.0초 |
| TSB | running | cpu | 1 | 2026-11-15 15:14:36+09:00 |  |
| TSB | complete | cpu | 1 | 2026-11-15 15:14:37+09:00 | 0.4초 |
| XGBoost | complete | cpu | 8 | 2026-11-15 15:15:29+09:00 | 67.0초 |
| LightGBM | complete | cpu | 7 | 2026-11-15 15:15:35+09:00 | 73.8초 |
| Ridge | running | cpu | 4 | 2026-11-15 15:15:36+09:00 |  |
| Ridge | complete | cpu | 4 | 2026-11-15 15:15:43+09:00 | 7.2초 |
| Timestral 209M | complete | cuda | 2 | 2026-11-15 15:16:17+09:00 | 116.6초 |
| Chronos-2 | complete | cuda | 2 | 2026-11-15 15:16:31+09:00 | 132.3초 |
| TiRex-2 | running | cuda | 2 | 2026-11-15 15:16:44+09:00 |  |
| DLinear | running | cpu | 6 | 2026-11-15 15:16:44+09:00 |  |
| N-HiTS | running | cuda | 3 | 2026-11-15 15:16:44+09:00 |  |
| DLinear | complete | cpu | 6 | 2026-11-15 15:16:50+09:00 | 7.2초 |
| AutoETS | running | cpu | 10 | 2026-11-15 15:17:58+09:00 |  |
| SARIMA | running | cpu | 2 | 2026-11-15 15:17:58+09:00 |  |
| CatBoost | running | cpu | 8 | 2026-11-15 15:17:58+09:00 |  |
| SARIMA | complete | cpu | 2 | 2026-11-15 15:17:58+09:00 | 0.5초 |
| Croston-SBA | running | cpu | 2 | 2026-11-15 15:18:00+09:00 |  |
| Croston-SBA | complete | cpu | 2 | 2026-11-15 15:18:01+09:00 | 0.5초 |
| N-HiTS | complete | cuda | 3 | 2026-11-15 15:18:24+09:00 | 105.4초 |
| AutoTheta | complete | cpu | 12 | 2026-11-15 15:19:33+09:00 | 322.6초 |
| CatBoost | complete | cpu | 8 | 2026-11-15 15:19:41+09:00 | 106.9초 |
| TimesFM 2.5 | running | cuda | 2 | 2026-11-15 15:19:44+09:00 |  |
| Toto 2.0 22M | running | cuda | 2 | 2026-11-15 15:19:45+09:00 |  |
| Moirai 2.0 Small | running | cuda | 2 | 2026-11-15 15:19:46+09:00 |  |
| AutoETS | complete | cpu | 10 | 2026-11-15 15:19:58+09:00 | 126.0초 |
| Toto 2.0 22M | failed | cuda | 2 | 2026-11-15 15:20:04+09:00 | EinopsError:  Error while processing sum-reduction pattern "... (seq patch) -> ... seq".  Input tensor shape: torch.Size([512, 1, 93]). Additional info: {'patch': 32}.  Shape mismatch, can't divide axis of length 93 in chunks of 32 |
| TiRex-2 | complete | cuda | 2 | 2026-11-15 15:20:24+09:00 | 230.2초 |
| Chronos-Bolt Base | running | cuda | 2 | 2026-11-15 15:20:27+09:00 |  |
| TiDE | running | cuda | 3 | 2026-11-15 15:20:27+09:00 |  |
| Moirai 2.0 Small | complete | cuda | 2 | 2026-11-15 15:20:53+09:00 | 69.5초 |
| TimesFM 3.0 | complete | cuda | 2 | 2026-11-15 15:21:36+09:00 | 451.7초 |
| TiDE | complete | cuda | 3 | 2026-11-15 15:21:44+09:00 | 79.8초 |
| PatchTST | running | cuda | 3 | 2026-11-15 15:21:46+09:00 |  |
| Toto 2.0 22M | running | cuda | 2 | 2026-11-15 15:21:51+09:00 |  |
| Chronos-Bolt Base | complete | cuda | 2 | 2026-11-15 15:22:18+09:00 | 114.7초 |
| Toto 2.0 22M | complete | cuda | 2 | 2026-11-15 15:22:43+09:00 | 54.7초 |
| PatchTST | complete | cuda | 3 | 2026-11-15 15:22:52+09:00 | 68.7초 |
| TimesFM 2.5 | complete | cuda | 2 | 2026-11-15 15:23:43+09:00 | 249.5초 |

<!-- benchmark-world-results:end -->

<details>
<summary>1차 비교 기록 (공개 지역 4,988곳)</summary>

## 1차 비교— 2026년 11월 15일

- **전체 지역을 대상으로 한 튜닝 후보는 N-HiTS입니다.** 17개 모델 중 2020–2023년 검증, 2024–2025년 홀드아웃, 2026년 1–7월 정식 처리본에서 모두 log MAE가 가장 낮았습니다. 홀드아웃에서 0.4544로 최근 5년 같은 달 중앙값(0.4699) 대비 3.3% 개선했고, 1·2·3개월 리드 각각에서도 가장 낮았습니다.
- **사전학습 모델 중에는 Timestral 209M이 앞섰습니다.** 홀드아웃 log MAE는 0.4617로 TimesFM 3.0의 0.5484보다 낮았습니다. 네 사전학습 모델의 80% 예측구간 포함률은 Timestral 80.68%, TimesFM 3.0 84.37%, TiRex-2 84.13%, Chronos-2 72.73%였습니다.
- 지역 합계에서는 **AutoTheta가 전 세계 WAPE 10.89%, Timestral이 아르헨티나 WAPE 30.53%, Ridge가 한국 전국 WAPE 82.61%**로 각각 가장 낮았습니다. 한국 2025년 3월 실적은 2,731이었고, 17개 모델의 1개월 앞 예측은 37–440에 그쳤습니다.

## 2026년 10월 말 기준 12월 전망

단위는 S-NPP 기준 탐지 환산량입니다. 구간은 모델의 10–90분위수입니다.

| 지역 | TimesFM 3.0 | TimesFM 3.0 구간 | Timestral 209M | Timestral 구간 | N-HiTS |
|---|---:|---:|---:|---:|---:|
| 한국 | 53.1 | 11.8–132.5 | 22.7 | 4.2–69.3 | 21.9 |
| 아르헨티나 | 5,860.1 | 1,446.2–15,107.9 | 5,054.9 | 2,353.0–14,070.7 | 5,189.7 |
| 전 세계 | 1,241,180.8 | 1,018,519.2–1,500,719.1 | 1,203,446.5 | 985,281.8–1,451,744.0 | 1,239,727.8 |

## 비교 조건

- 사용 가능한 S-NPP 이력은 2012년 2월부터 시작합니다. 월별 누락·중복이 없는 지역 패널을 확인했습니다. 정적 열원 차감 후 남은 음수 143건과 음수 예측은 0으로 처리합니다.
- 2020–2023년은 후보 비교, 2024–2025년은 시계열 홀드아웃, 2026년 1–7월은 추가 정식 처리본(SP) 검증에 사용합니다. 분할은 예측 대상월 기준이며 모델당 1,182,156개 예측을 평가했습니다.
- 각 기준월까지 관측한 뒤 1·2·3개월 앞을 예측합니다. 사전학습 모델 네 개는 가중치를 고정하고 지역별 독립 시계열을 입력했습니다. 지역별 통계 모델은 기준월까지의 이력으로 적합했습니다.
- 공동 학습 모델은 각 기준월 시점에서 가장 최근에 끝난 12월을 학습 마감으로 삼았습니다. Ridge·LightGBM·XGBoost는 시차값, 달력, 이동 집계, 지역 수준과 격자 수를 사용했습니다. DLinear·N-HiTS는 기준월까지 36개월 이력에서 1–3개월 전망을 직접 학습했습니다. 주요 후보마다 한 설정으로 비교했습니다. [DLinear 논문](https://arxiv.org/abs/2205.13504) · [N-HiTS 논문](https://arxiv.org/abs/2201.12886) · [StatsForecast 모델 문서](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html)
- 입력 특성은 마감월까지의 관측 이력과 당시 알 수 있는 달력·지역 정보로 구성합니다. 2026년 8–10월 NRT는 10월 말 기준 전망의 입력에 사용합니다. 11월 진행 중 집계와 NOAA-20 전환기는 실적 평가에서 제외합니다.
- 비교 지표는 `abs(log1p(예측) - log1p(실적))`의 평균입니다. 같은 분할의 전체 지역·월·리드타임에 같은 가중치를 줍니다. WAPE는 지역 또는 지역 수준별 절대 오차 합을 실적 합으로 나눈 값입니다.
- 이 실험은 하나의 배포 스냅샷을 대상으로 한 과거 검증입니다. 관측 자료 정정과 고정 열원 제거가 배포 시점 기준으로 반영되어 있습니다.
- 사전학습 체크포인트는 배포 커밋을 고정했습니다. TimesFM 3.0 공개 가중치는 비상업·비프로덕션 라이선스입니다. [TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch) · [TimesFM 라이선스 안내](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights) · [Chronos-2](https://huggingface.co/amazon/chronos-2) · [Timestral 209M](https://huggingface.co/mistralai/Timestral-209M) · [TiRex-2](https://huggingface.co/NX-AI/TiRex-2)

## 실행과 검증

Ryzen 9 7950X의 논리 CPU 32개와 RTX 4090을 사용했습니다. CPU에서는 LightGBM, XGBoost, Ridge, AutoTheta를 함께 실행했습니다. GPU에서는 TimesFM 3.0을 먼저 실행했고, 이후 Chronos-2, Timestral, TiRex-2 등을 겹쳐 실행하는 동안 GPU 사용률 100%를 확인했습니다.

원본 Parquet와 저장된 예측을 다시 읽어 17개 모델의 평가 예측 20,096,652개에 대해 분할·리드별 전체 log MAE·MAE, 한국 WAPE, 사용 가능한 80% 구간 포함률을 대조했습니다. 미래 관측을 바꿔도 이전 기준월의 입력 특성이 그대로임을 확인했습니다.

## 산출물

실험 결과는 `data/processed/forecast_benchmark/aoi-month-v1-3e5a46208726/` 아래에 있습니다. `metrics.csv`에 지표를 모았고, `runs.jsonl`에는 실행 설정과 상태를 기록했습니다. 모델별 `backtest.npz`는 과거 예측, `forecast.csv`는 2026년 10월 말 기준 11월–2027년 1월 전망입니다.
세부 지표에는 행정 단계, 주요 지역, 과거 활동 빈도, 홀드아웃 연도별 MAE·RMSE·WAPE·편향과 제공되는 경우 80% 구간 포함률이 있습니다.
과거 예측 배열 축은 기준월 × AOI × 리드타임이며, `ids`와 `origins`가 배열 순서를 명시합니다.

## 재실행

WSL에서 아래 환경을 사용했습니다. 기존 환경을 재사용하는 경우 마지막 실행 명령만 필요합니다. 모델 키는 `python -m planetarium.modeling.forecast_benchmark --help`에서 확인할 수 있습니다.

```bash
forecast_cpu="${XDG_CACHE_HOME:-$HOME/.cache}/planetarium-forecast/cpu"
forecast_gpu="${XDG_CACHE_HOME:-$HOME/.cache}/planetarium-forecast/gpu"
uv venv --python 3.12 "$forecast_cpu"
uv pip install --python "$forecast_cpu/bin/python" \
  numpy==2.5.4 polars==1.44.2 filelock pandas==2.3.3 pyarrow==26.0.1 \
  scikit-learn==1.9.1 lightgbm==4.7.0 xgboost==3.4.1 statsforecast==2.1.1
uv venv --python 3.12 "$forecast_gpu"
uv pip install --python "$forecast_gpu/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
uv pip install --python "$forecast_gpu/bin/python" \
  numpy==2.5.4 polars==1.44.2 filelock timesfm==3.0.2 chronos-forecasting==2.3.2 \
  transformers==5.23.0 tirex-2==0.5.0 neuralforecast==3.3.1 \
  'timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff'

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$forecast_gpu/bin/python" \
  -m planetarium.modeling.forecast_benchmark run --model timesfm3 --device cuda --threads 2 --batch-size 512
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 "$forecast_cpu/bin/python" \
  -m planetarium.modeling.forecast_benchmark run --model lightgbm --threads 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 "$forecast_gpu/bin/python" \
  -m planetarium.modeling.forecast_benchmark run --model nhits --device cuda --threads 3 --batch-size 2048
```

모델 완료 시 아래 결과 영역이 갱신됩니다. 완료한 모델을 같은 데이터로 다시 실행할 때 `--force`를 붙이면 별도 시도 폴더에 저장됩니다.

<!-- benchmark-results:start -->

## 모델별 성능

| 모델 | 유형 | 검증 | 홀드아웃 | 홀드아웃 1개월 | 홀드아웃 2개월 | 홀드아웃 3개월 | 2026 정식 자료 |
|---|---|---:|---:|---:|---:|---:|---:|
| N-HiTS | 전체 지역 공동 학습 | 0.4653 | 0.4544 | 0.4370 | 0.4591 | 0.4672 | 0.4758 |
| Timestral 209M | 사전학습 | 0.4707 | 0.4617 | 0.4434 | 0.4633 | 0.4783 | 0.4776 |
| LightGBM | 전체 지역 공동 학습 | 0.4728 | 0.4656 | 0.4546 | 0.4685 | 0.4737 | 0.4838 |
| DLinear | 전체 지역 공동 학습 | 0.4780 | 0.4683 | 0.4522 | 0.4745 | 0.4781 | 0.4879 |
| 최근 5년 같은 달 중앙값 | 기준선 | 0.4795 | 0.4699 | 0.4699 | 0.4699 | 0.4699 | 0.4896 |
| XGBoost | 전체 지역 공동 학습 | 0.4801 | 0.4720 | 0.4591 | 0.4755 | 0.4815 | 0.4905 |
| AutoTheta | 지역별 학습 | 0.4915 | 0.4772 | 0.4544 | 0.4810 | 0.4964 | 0.4979 |
| Ridge | 전체 지역 공동 학습 | 0.4947 | 0.4876 | 0.4738 | 0.4885 | 0.5005 | 0.5008 |
| 계절 자기회귀 | 지역별 학습 | 0.5082 | 0.5019 | 0.4818 | 0.5106 | 0.5133 | 0.5140 |
| Chronos-2 | 사전학습 | 0.5338 | 0.5295 | 0.5197 | 0.5297 | 0.5391 | 0.5307 |
| 최근 5년 같은 달 평균 | 기준선 | 0.5344 | 0.5185 | 0.5185 | 0.5185 | 0.5185 | 0.5331 |
| 계절 지수평활 | 지역별 학습 | 0.5401 | 0.5248 | 0.5248 | 0.5248 | 0.5248 | 0.5408 |
| TiRex-2 | 사전학습 | 0.5499 | 0.5477 | 0.5380 | 0.5508 | 0.5543 | 0.5421 |
| TimesFM 3.0 | 사전학습 | 0.5541 | 0.5484 | 0.5235 | 0.5604 | 0.5614 | 0.5454 |
| 작년 같은 달 | 기준선 | 0.5814 | 0.5715 | 0.5715 | 0.5715 | 0.5715 | 0.5985 |
| 직전 달 | 기준선 | 0.9442 | 0.9548 | 0.6859 | 0.9837 | 1.1948 | 0.9450 |
| TSB | 간헐 시계열 | 1.0786 | 1.0770 | 1.0516 | 1.0842 | 1.0952 | 1.0247 |

표는 log MAE 기준이며 값이 낮을수록 좋습니다. 행은 2020–2023년 검증 점수순입니다.

### 홀드아웃 지역별 WAPE

| 모델 | 한국 | 아르헨티나 | 미국 | 전 세계 | 한국 시군구 | 국가 단위 |
|---|---:|---:|---:|---:|---:|---:|
| N-HiTS | 83.90% | 37.86% | 25.78% | 21.48% | 99.27% | 34.48% |
| Timestral 209M | 88.38% | 30.53% | 21.27% | 12.08% | 99.66% | 29.95% |
| LightGBM | 82.72% | 49.18% | 56.19% | 31.72% | 99.36% | 60.06% |
| DLinear | 84.65% | 32.94% | 21.57% | 12.92% | 107.21% | 32.65% |
| 최근 5년 같은 달 중앙값 | 88.08% | 46.10% | 20.25% | 15.03% | 103.08% | 34.02% |
| XGBoost | 83.03% | 46.14% | 52.79% | 25.10% | 102.26% | 64.79% |
| AutoTheta | 88.58% | 39.81% | 23.06% | 10.89% | 109.13% | 40.39% |
| Ridge | 82.61% | 36.07% | 21.84% | 14.00% | 109.73% | 35.75% |
| 계절 자기회귀 | 90.10% | 39.03% | 26.20% | 17.98% | 108.46% | 36.08% |
| Chronos-2 | 87.95% | 31.03% | 21.42% | 12.92% | 100.40% | 32.07% |
| 최근 5년 같은 달 평균 | 89.10% | 63.32% | 29.71% | 13.97% | 135.47% | 33.63% |
| 계절 지수평활 | 90.17% | 48.89% | 24.99% | 14.79% | 132.34% | 34.15% |
| TiRex-2 | 89.61% | 32.73% | 26.03% | 13.10% | 103.25% | 33.45% |
| TimesFM 3.0 | 91.51% | 33.37% | 23.90% | 12.51% | 113.30% | 32.82% |
| 작년 같은 달 | 90.69% | 47.61% | 35.27% | 18.62% | 116.81% | 47.45% |
| 직전 달 | 156.06% | 82.13% | 56.92% | 36.84% | 185.67% | 105.35% |
| TSB | 125.20% | 68.84% | 47.67% | 33.01% | 142.26% | 108.01% |

## 실행 기록

| 모델 | 상태 | 장치 | CPU 스레드/작업 수 | 시간 | 세부 기록 |
|---|---|---|---:|---|---|
| TimesFM 3.0 | running | cuda | 2 | 2026-11-15 14:50:51+09:00 |  |
| AutoTheta | running | cpu | 9 | 2026-11-15 14:53:56+09:00 |  |
| 직전 달 | running | cpu | 1 | 2026-11-15 14:53:56+09:00 |  |
| XGBoost | running | cpu | 9 | 2026-11-15 14:53:56+09:00 |  |
| Ridge | running | cpu | 4 | 2026-11-15 14:53:57+09:00 |  |
| LightGBM | running | cpu | 8 | 2026-11-15 14:53:57+09:00 |  |
| 직전 달 | complete | cpu | 1 | 2026-11-15 14:53:57+09:00 | 0.3초 |
| 작년 같은 달 | running | cpu | 1 | 2026-11-15 14:53:58+09:00 |  |
| 작년 같은 달 | complete | cpu | 1 | 2026-11-15 14:53:58+09:00 | 0.3초 |
| 최근 5년 같은 달 중앙값 | running | cpu | 1 | 2026-11-15 14:53:59+09:00 |  |
| 최근 5년 같은 달 중앙값 | complete | cpu | 1 | 2026-11-15 14:54:00+09:00 | 0.6초 |
| 최근 5년 같은 달 평균 | running | cpu | 1 | 2026-11-15 14:54:00+09:00 |  |
| 최근 5년 같은 달 평균 | complete | cpu | 1 | 2026-11-15 14:54:00+09:00 | 0.5초 |
| 계절 지수평활 | running | cpu | 1 | 2026-11-15 14:54:02+09:00 |  |
| 계절 지수평활 | complete | cpu | 1 | 2026-11-15 14:54:03+09:00 | 0.7초 |
| TSB | running | cpu | 1 | 2026-11-15 14:54:04+09:00 |  |
| TSB | complete | cpu | 1 | 2026-11-15 14:54:05+09:00 | 0.5초 |
| TimesFM 3.0 | complete | cuda | 2 | 2026-11-15 14:54:09+09:00 | 207.8초 |
| Ridge | complete | cpu | 4 | 2026-11-15 14:54:21+09:00 | 26.1초 |
| XGBoost | complete | cpu | 9 | 2026-11-15 14:54:53+09:00 | 59.1초 |
| LightGBM | complete | cpu | 8 | 2026-11-15 14:54:58+09:00 | 64.4초 |
| TiRex-2 | running | cuda | 2 | 2026-11-15 14:55:19+09:00 |  |
| Timestral 209M | running | cuda | 2 | 2026-11-15 14:55:19+09:00 |  |
| Chronos-2 | running | cuda | 2 | 2026-11-15 14:55:19+09:00 |  |
| DLinear | running | cpu | 8 | 2026-11-15 14:56:44+09:00 |  |
| Ridge | running | cpu | 6 | 2026-11-15 14:56:45+09:00 |  |
| N-HiTS | running | cuda | 3 | 2026-11-15 14:56:45+09:00 |  |
| Ridge | complete | cpu | 6 | 2026-11-15 14:56:52+09:00 | 7.3초 |
| DLinear | complete | cpu | 8 | 2026-11-15 14:56:52+09:00 | 7.7초 |
| Timestral 209M | complete | cuda | 2 | 2026-11-15 14:57:37+09:00 | 145.0초 |
| Chronos-2 | complete | cuda | 2 | 2026-11-15 14:57:46+09:00 | 153.8초 |
| N-HiTS | complete | cuda | 3 | 2026-11-15 14:58:23+09:00 | 102.3초 |
| TiRex-2 | complete | cuda | 2 | 2026-11-15 14:58:58+09:00 | 228.9초 |
| AutoTheta | complete | cpu | 9 | 2026-11-15 14:59:20+09:00 | 338.6초 |
| 계절 자기회귀 | running | cpu | 8 | 2026-11-15 15:00:13+09:00 |  |
| 계절 자기회귀 | complete | cpu | 8 | 2026-11-15 15:00:13+09:00 | 0.5초 |

<!-- benchmark-results:end -->

</details>
