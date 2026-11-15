# 다음 달 열원 탐지 확률 실험

예측값 `p_any`는 기준월 `t` 말에 계산한, 해당 지역에서 **다음 달 `t+1`에 기준 열원이 한 건 이상 탐지될 확률**입니다. 실제 정답은 한 건 이상이면 1, 완전 관측월에 한 건도 없으면 0입니다. [v1.2 정답 정의](../docs/release_v1.2-2026-11-15.md)

## 결과

- 전 세계 육지 격자 **69,215개**에서 기준선 2개를 포함해 **9개 모델**을 비교했습니다. 모델마다 정식 처리본 79개월의 5,467,985개 격자·월을 평가했습니다. 2024–2025년 홀드아웃의 실제 탐지 비율은 26.87%였습니다.
- **XGBoost**는 홀드아웃에서 Brier **0.07196**, PR-AUC **0.90239**, log loss **0.23109**를 기록했습니다. 과거 같은 달 발생률 기준선의 Brier 0.07852 대비 8.36% 낮았습니다. CatBoost의 검증 Brier는 0.07289, LightGBM의 2026년 추가 검증 Brier는 0.07398로 각각 해당 기간에서 가장 낮았습니다.
- 이진 학습한 Hugging Face 후보 가운데 **PatchTST**의 홀드아웃 Brier는 0.07369였습니다. **TimesFM 3.0 + Platt**의 홀드아웃 Brier는 0.10211, PR-AUC는 0.78900이었습니다.
- 입력 Parquet와 경계 파일의 SHA-256을 v1.2 배포 매니페스트와 대조했습니다. 저장된 확률 49,211,865개에서 전 세계 지표를 재계산하고, 각 전망 행의 대상월이 기준월 다음 달인지 확인했습니다.

## 데이터와 평가

- 전 세계 0.5° 격자를 사용합니다. 배포판 국가 경계에서 남극을 제외한 육지와 겹치는 격자를 포함하며 과거 탐지가 없는 격자도 평가합니다. 국가와 대륙 표시는 각 칸에서 면적이 가장 많이 겹치는 국가를 기준으로 합니다.
- `harmonized_cell_day.n_points`의 월간 합으로 정답을 만들었습니다. v1.2 기준점에서 고정 열원 제외가 적용된 자료입니다. S-NPP 관측은 2012년 2월부터 2026년 10월까지 사용합니다. 검증 정답은 2026년 7월까지의 정식 처리본입니다.
- 2020–2023년은 검증, 2024–2025년은 홀드아웃, 2026년 1–7월은 추가 검증입니다. 시험 연도 `Y`의 학습은 `Y-2`년까지, 확률 보정은 `Y-1`년에 합니다. 각 격자의 예측 입력은 대상월 직전까지의 관측입니다.
- 확률 정확도는 Brier score와 log loss, 양성 지역 판별은 PR-AUC로 평가합니다. 전체 격자와 대륙별 성능을 계산합니다.
- 2026년 10월 마감 자료를 마지막 입력으로 사용한 전망의 대상월은 2026년 11월입니다.

## 후보

- 과거 같은 달의 발생률과 직전 달 발생 여부에 조건을 둔 계절 Markov 확률을 기준선으로 사용합니다.
- 로지스틱 회귀, LightGBM, XGBoost, CatBoost는 이진 정답으로 학습합니다. 12개월 주기, 과거 발생률, 앞선 달의 관측과 주변 격자 정보를 입력으로 사용합니다.
- PatchTSMixer와 PatchTST는 과거 36개월 창을 입력해 이진 교차 엔트로피로 학습합니다. [PatchTSMixer](https://huggingface.co/docs/transformers/model_doc/patchtsmixer) · [PatchTST](https://huggingface.co/docs/transformers/model_doc/patchtst)
- TimesFM 3.0은 두 번의 연간 주기를 담는 과거 24개월의 0/1 이력으로 한 시점 앞 점수를 계산합니다. 이전 연도의 실제 이진 정답으로 Platt 보정을 적합해 점수를 확률로 변환합니다. [TimesFM 3.0 공개 가중치](https://huggingface.co/google/timesfm-3.0-pytorch)
- 표 형태의 분류기는 연도별 과거 학습 행에서 최대 60만 행을 표집합니다. 신경망은 연도별로 1,024개 창을 240번 표집해 학습합니다. Platt 보정에는 이전 연도에서 최대 20만 행을 사용합니다. 평가는 해당 기간의 모든 격자·월을 사용합니다.

## 실행과 산출물

Python 3.12에서 NumPy, Polars, SciPy, scikit-learn, Shapely, filelock을 사용했습니다. 트리 모델은 LightGBM, XGBoost, CatBoost 패키지를 사용합니다. GPU 후보에는 CUDA PyTorch와 `transformers==5.23.0`, `timesfm==3.0.2`를 사용했습니다.

```bash
python -m planetarium.modeling.forecast_benchmark prepare
python -m planetarium.modeling.forecast_benchmark run --model seasonal
python -m planetarium.modeling.forecast_benchmark run --model xgboost --device cuda --threads 8
python -m planetarium.modeling.forecast_benchmark run --model patchtsmixer --device cuda
```

`data/interim/occurrence_benchmark/`에 평가 지표 `metrics.csv`, 모델별 검증 확률 `runs/*.npz`, 지역별 전망 `runs/*.parquet`를 저장합니다. 전망은 `p_any`와 `p_none`을 함께 제공합니다. 모델이 완료될 때마다 아래 표가 갱신됩니다.

<!-- occurrence-results:start -->

## 전 세계 비교

| 모델 | 2020–2023 Brier | 2024–2025 Brier | 2024–2025 PR-AUC | 2024–2025 log loss | 2026년 1–7월 Brier |
|---|---:|---:|---:|---:|---:|
| CatBoost | 0.07289 | 0.07203 | 0.90219 | 0.23218 | 0.07411 |
| XGBoost | 0.07294 | 0.07196 | 0.90239 | 0.23109 | 0.07409 |
| LightGBM | 0.07369 | 0.07210 | 0.90202 | 0.23169 | 0.07398 |
| PatchTST 이진 분류 | 0.07460 | 0.07369 | 0.89767 | 0.24010 | 0.07519 |
| 로지스틱 회귀 | 0.07500 | 0.07424 | 0.89850 | 0.24494 | 0.07628 |
| PatchTSMixer 이진 분류 | 0.07529 | 0.07461 | 0.89588 | 0.24405 | 0.07631 |
| 계절 Markov | 0.07946 | 0.07788 | 0.88244 | 0.25721 | 0.07989 |
| 과거 같은 달 발생률 | 0.08011 | 0.07852 | 0.87083 | 0.26283 | 0.08029 |
| TimesFM 3.0 + Platt | 0.10320 | 0.10211 | 0.78900 | 0.34483 | 0.10341 |

<!-- occurrence-results:end -->
