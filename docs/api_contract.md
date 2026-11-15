# API 계약 (데이터 담당 확정본, 작성 중) — 데이터 `data-v1.2-2026-11-15` (변환 모델 FireBridge (MODIS-to-VIIRS Harmonizer, internal id: `m2v_xgb`), method_version `v1-m2v-2026-11-14`)

> 감지 단위 통합(harmonized hotspots) 2·3단계 기준. 6단계에서 응답 예시·오류 형식까지 채워 확정한다.
> 개요·화면 설명은 `docs/backend-handoff.md`.

## 데이터 원천과 API 대응

| API | 읽는 데이터 | 비고 |
|---|---|---|
| `GET /summary` | **`harmonized_cell_day.parquet`** (+ 그 달 `residual_month`) | 일·월 합계 |
| `GET /hotspots` | **`harmonized_hotspots/{reference,other}/year=YYYY/month=MM/`** | 지도를 확대했을 때 점 |
| `GET /calendar` | `aoi_month.parquet` + `aoi_month_compare.parquet` | 연도 × 월 |
| `GET /spread` | `spread_nowcast_latest.parquet` | 내일 확산 가능성 (한국·아르헨티나) |
| `GET /events` | `fire_events.parquet` | 화재 사건 (한국·아르헨티나) |
| `GET /regime` | `aoi_month_fire_regime.parquet` | AOI × 월 불 유형 배지 |
| `POST /harmonize` | 사용자 업로드 (서버 자료 불변) | FIRMS 파일 → 통합 결과 CSV + summary.json |
| `GET /occurrence` | `aoi_occurrence_nextmonth_latest.parquet` | 다음 달 시군구별 열원 1건 이상 확률 (한국) |
| `GET /recurrence` | `recurrence_nextmonth_latest.parquet` (+ `recurrent_burn_sites.parquet`) | 다음 달 전망 (아르헨티나 확률, 한국 반복 횟수) |
| `GET /fwi` | `fwi_aoi_day.parquet` (+ `fwi_clim_aoi_month.parquet`) | 지역 산불 기상 위험도 배지 (한국·아르헨티나) |
| `GET /export.csv` | 위 결과를 CSV로 | |

### `/summary?aoi|bbox|polygon&start&end&freq=day|month`

- `harmonized_cell_day`에서 영역 안 칸(`aoi`는 칸 × AOI 면적 가중치)과 `date_local` 범위로 `value`, `value_lo`, `value_hi`를 더한다.
- **residual 규칙** (residual = 점으로 나눌 수 없는 값, `harmonized_residual_cell_month.residual_type`으로 구분)
  - **residual(`missing_day_fill` 음수 포함)은 합계 계산에만 쓰고, 별도 값·막대·레이어로 화면에 표시하지 말 것.**
  - `modis_unseen` (2000-11 ~ 2012-01): MODIS가 감지하지 못한 칸의 FireBridge 추정분 + 배경 예측
  - `missing_day_fill` (2012-02 ~, 완결된 달): 위성 자료가 빠진 날을 달 전체로 환산한 몫 = 칸 value − Σ 기준 점 가중치.
    빠진 날에 감지가 있었던 칸은 **음수**일 수 있다(81,563행). 점 가중치는 센서 변환만(S-NPP 1.0, NOAA-20 ×n20_bridge,
    NOAA-21 ×n21_bridge, MODIS ×FireBridge 배율)이라 이 몫이 점에 들어가지 않는다.
  - 저장 위치: `harmonized_cell_day.residual_month` — 칸·월마다 그 달 **1일 행에만** 칸·월 합계가 들어 있고(나머지 날은 null), 일수로 배분하지 않았다. **저장 위치가 1일일 뿐 1일의 값이 아니다.**
  - **`freq=day` 응답**: 일별 `value`에는 residual을 **넣지 않는다**. 대신 달마다 `month_residual`(그 달 전체 residual 합)과
    `note`("MODIS 기간 점으로 표시되지 않는 추정분, 월 단위") 필드를 따로 준다. 프론트엔드는 이것을 1일 막대·점에 몰아서 그리지 않는다
    (합계 계산과 안내 문구에만 쓰고 별도 그래프·값으로 그리지 않는다).
  - **`freq=month` 응답**: 그 달 `value` 합 + `residual_month` 합 (= `aoi_month.value`).
  - **기간 합계(start~end)**: 달 전체가 포함되면 그 달 residual 전부, 일부만 포함되면 **(포함 일수 / 그 달 일수)** 비율만큼 더하고,
    응답에 `residual_prorated: true`를 표시한다 (전부 포함이면 `false`).
  - **S-NPP 이후(2012-02 ~)는 대부분의 달에 residual이 없다.** 위성 자료가 빠진 날이 있는 26개 달에만 `missing_day_fill`이 있다
    (예: 2024-07, 2022-08, 2026-07, 2022-07). 응답의 `month_residual`에는 두 유형을 합쳐 주고, 필요하면 `residual_type`별 값을 함께 준다.
  - **진행 중인 달**(status=partial)은 residual이 없다 (`value`가 오늘까지 실제 합계라서).

  ```json
  // GET /summary?aoi_id=KR&start=2005-03-10&end=2005-04-30&freq=day (형식 예시)
  {
    "aoi_id": "KR", "freq": "day", "method_version": "v1-m2v-2026-11-14",
    "series": [
      {"date": "2005-03-10", "value": 3.1, "value_lo": 1.2, "value_hi": 4.9, "value_source": "modis_m2v"}
    ],
    "month_residual": [
      {"year": 2005, "month": 3, "month_residual": 144.4, "note": "MODIS 기간 점으로 표시되지 않는 추정분, 월 단위"},
      {"year": 2005, "month": 4, "month_residual": 98.7, "note": "MODIS 기간 점으로 표시되지 않는 추정분, 월 단위"}
    ],
    "total": {"value": 412.3, "residual_included": 197.5, "residual_prorated": true}
  }
  ```
  (`total.residual_included` = 3월 residual × 22/31 + 4월 residual 전부. 숫자는 형식 예시)
- 월로 묶은 AOI 합(`value` + `residual_month`)은 `aoi_month.value`와 같다 (4단계 검증).
- `status`, `value_source`는 `aoi_month`(월)과 `harmonized_cell_day.value_source`(일)를 함께 준다.

### `/hotspots?bbox&start&end&layer=reference|all&limit`

- 지도를 충분히 확대했을 때(예: bbox가 일정 면적 이하) 점 파일을 읽는다. 넓은 범위는 `harmonized_cell_day`/`harmonized_cell_month` 격자로 보여 준다.
- `layer=reference`: `harmonized_hotspots/reference/` (합계에 쓰는 기준 위성 점).
- **v1.2: 고정 열원 점 제외** — `harmonized_hotspots_static_flags/year=YYYY.parquet`(점 키 `satellite`·`latitude`·`longitude`·`acq_datetime_utc` → `is_static_source`)와
  anti-join 해서 뺀다. 점 파일에는 아직 들어 있지만 칸·AOI 합계에서는 이미 빠졌다.
  `layer=all`: `reference/` + `other/` (2012-02 이후 S-NPP·NOAA-20·NOAA-21 중 기준이 아닌 점, 지도 표시용 — 합계에 쓰지 않음).
- **점 지도에는 MODIS 기간(2000-11 ~ 2012-01) 미감지 추정분(residual)이 표시되지 않는다.**
  같은 영역·기간이라도 점 가중치 합은 `/summary` 월 합계보다 작을 수 있다 (차이 = residual). 화면에 안내 문구를 넣는다.
- MODIS 비기준 점(2012-02 이후 Terra·Aqua)은 가중치가 없어 만들지 않았다.

### 구간(weight_lo / weight_hi, value_lo / value_hi)

| 기간 | 점·칸 단위 | AOI 단위 (`aoi_month`) |
|---|---|---|
| 2000-11 ~ 2002-06 (Terra 단독) | **null** — 이 기간 구간은 AOI 단위로만 제공 | 있음 |
| 2002-07 ~ 2012-01 | 있음 (FireBridge Uncertainty Band) | 있음 (AOI 보정) |
| 2012-02 ~ | value와 같음 | value와 같음 |

- 점·칸 구간을 더한 값은 합계의 정확한 구간이 아니다(오차 상쇄 무시로 넓음). 지역 구간은 `aoi_month.value_lo/hi`를 쓴다.

### 예년 비교 기준 기간 — `baseline_period`, `continuity_ratio` (2026-11-15)

`aoi_month_compare`의 평년(`baseline_median`·`p25`·`p75`)과 백분위·판정은 지역마다 정한 기준 기간으로 계산한다.

- **연속성 점검** `continuity_ratio` = (통합값 2013~2021 ÷ 2003~2011) ÷ (MODIS 원자료 2013~2021 ÷ 2003~2011).
  0.85~1.15 → `baseline_period = "2001-2025"`(전체), 밖이면 `"2012-2025"`(VIIRS 시기만). 예: 한국 0.68 → 2012~, 아르헨티나 1.05·전 세계 1.02 → 2001~.
  두 기간 MODIS 원자료가 각각 50건 미만인 작은 지역은 상위 지역(시군구 → 시도 → 나라)의 비율을 물려받는다 (`continuity_source = "parent:<id>"`, 아니면 `"own"`).
  현재 3,036개 지역 2001~, 2,015개 지역 2012~.
- `baseline_period = "2012-2025"` 지역의 **2012년 이전 달은 `verdict = uncertain`** (그 기간 통합값이 원자료 추세와 어긋나므로).
  기준 연수 8년 미만도 `uncertain` (이전 10년 → 8년; 2012~ 기준은 대상 해를 빼고 13년).
- 화면: 평년 설명에 "예년(2012~2025)" / "예년(2001~2025)"처럼 기간을 함께 보여 준다.
- 검증 (`reports/compare_continuity_validation.csv`): 점검을 통과한 지역은 판정이 하나도 바뀌지 않았고, 실패 지역은 최근 3년(2023~2025) 판정이
  많음 16% / 적음 37% → 20% / 24%로 균형에 가까워졌다
  (한국 8% / 53% → 19% / 33%).
  2012 이후 달 중 약 1.9만 개 지역·월은 평년이 낮아져 `low_sample`(예년 75백분위 < 1)이 되어 `uncertain`으로 바뀌었다.

## 진행 중인 달 (status = partial) — v1.1

| 컬럼 | 의미 |
|---|---|
| `value`, `value_lo`, `value_hi` | **오늘까지 실제 합계** (month-to-date, 환산 없음) |
| `value_projected` | 관측일 수로 환산한 **달 전체 추정** (완결된 달은 `value`와 같음) |
| `as_of_date` | 그 달 자료의 **마지막 관측일** (완결된 달은 월말) |

- 적용 표: `harmonized_cell_month.value_projected`, `aoi_month.value_projected / as_of_date`,
  `aoi_month_compare.value_projected / as_of_date`, `harmonized_cell_day.value_projected_month`(그 달 1일 행, 칸·월 달 전체 추정).
- `/calendar`, `/summary`, `/compare` 응답에 진행 중인 달이 들어가면 `value`(월 누계), `value_projected`, `as_of_date`, `status: "partial"`을
  함께 주고, 판정(`verdict`)은 항상 `uncertain`.
- 화면: "2026-11-13까지 44.3 (이 추세면 한 달 약 102)"처럼 누계와 추정을 구분해 표시.

```json
// GET /calendar?aoi_id=KR&start_year=2026&end_year=2026 중 11월
{"year": 2026, "month": 11, "value": 44.3, "value_projected": 102.3, "as_of_date": "2026-11-13",
 "status": "partial", "value_source": "n20_bridge", "verdict": "uncertain"}
```

## 확산 가능성·화재 사건·불 유형 (2026-11-15 추가, 한국·아르헨티나)

### `/spread?aoi|bbox` — FireOutlook Spread (internal id: `spread_nowcast`)

- 읽는 파일: `data/processed/spread_nowcast_latest.parquet` (모델 `models/spread_nowcast_xgb.json`, 설명 `models/spread_nowcast_meta.json`).
  `fetch_nrt --append --with-spread` 때마다 다시 만든다.
- 한 행 = 0.025° 칸(약 2.5 km). 오늘 활성 칸과 그 주변 반경 3칸(약 7.5 km) 안의 칸만 있다(후보 칸). 없는 칸은 "대상 아님"이지 0%가 아니다.
- `aoi`는 칸 중심이 AOI 경계 안인 칸, `bbox`는 칸 중심이 상자 안인 칸.

| 응답 필드 | 원본 열 | 의미 |
|---|---|---|
| `cell_lat`, `cell_lon`, `cell_size_deg` | 같음 | 칸 중심과 크기(0.025°) |
| `p_spread` | `p_spread` | 다음 현지 날짜에 그 칸에 위성 감지가 생길 확률 (0~1, 보정됨) |
| `level` | `p_spread`에서 계산 | `low` < 0.05 ≤ `moderate` ≤ 0.20 < `high` |
| `as_of_date` | `as_of_date` | 기준일 (마지막 정상 관측일, NOAA-20) |
| `target_date` | `target_date` | 예측일 (= 기준일 + 1) |
| `weather_lag_days` | `weather_lag_days` | 기상 자료 지연(일). **v1.2-ops 모델은 기상을 쓰지 않아 비어 있음** (`weather_available = false`) |
| `fire_event_days` | `fire_event_days` | (v1.2-ops 추가) 주변 불 칸이 속한 화재 사건이 시작된 지 며칠째인가 |
| `active_today` | `active_today` | 오늘 이 칸에 감지가 있었나 |

- **표시 3단계**: `p_spread` < 5% / 5–20% / > 20% (색 3단계). 확률 숫자는 툴팁에만.
- **필수 문구** (지도 범례 또는 레이어 설명에 항상):
  > 위성 감지 기반 단기 확산 가능성입니다. 물리 시뮬레이션이 아니며, 오늘 감지 지점에서 반경 약 7.5 km 밖으로 번지는 경우와 새로 발생하는 불은 예측 대상이 아닙니다.
- **FireOutlook Spread 모델 v1.2-ops (2026-11-15 교체)**: 오늘·어제 주변 감지·FRP·연속 연소 일수·화재 사건 지속일·방향 특성만 쓴다(기상 없음).
  **검증은 운영과 같은 기상 지연 조건**에서 했다: 운영에서는 NASA POWER가 약 4일 늦으므로 "기준일 − 4일 기상"과 "기상 제외"를 비교했고,
  기상 제외가 두 나라 모두 더 나았다. 이전 v1은 기준일 당일 기상으로 검증해 운영 성능보다 조금 높게 나왔을 수 있다.
- 검증 (2012~2025 연도별 분리, 고정 열원 제외 표본, 후보 칸 안, PR-AUC / Brier):

| 그룹 | 새 모델 (기상 제외) | 운영형 (−4일 기상) | 참고: 당일 기상 (이전 v1 방식) |
|---|---|---|---|
| 한국 상자 | **0.147 / 0.00546** | 0.139 / 0.00549 | 0.158 / 0.00541 |
| 한국만 (KOR) | **0.210 / 0.00316** | 0.198 / 0.00320 | 0.212 / 0.00317 |
| 아르헨티나 | **0.190 / 0.01231** | 0.189 / 0.01231 | 0.197 / 0.01226 |

  후보 칸이 덮는 내일 감지 비율(재현율 상한) 한국 25%, 아르헨티나 42% → `reports/spread_ops_metrics.csv`, `models/spread_nowcast_meta.json`.

```json
// GET /spread?bbox=128.5,35.8,129.6,36.4
{"as_of_date": "2026-11-13", "target_date": "2026-11-14", "weather_lag_days": null,
 "notice": "위성 감지 기반 단기 확산 가능성 (물리 시뮬레이션 아님). 반경 약 7.5 km 밖·새 발화는 대상 아님.",
 "cells": [{"cell_lat": 36.0375, "cell_lon": 129.3875, "cell_size_deg": 0.025, "p_spread": 0.217, "level": "high", "active_today": true}]}
```

### `/events?aoi&start&end`

- 읽는 파일: `data/processed/fire_events.parquet` (1,240,978행, 46 MB; v1.2 고정 열원 점 제외). 기준 점을 시공간으로 이은 사건
  (거리 ≤ 750 m VIIRS / 2 km MODIS, 현지 날짜 차 ≤ 1일). 한국 2000-11~, 아르헨티나 2000-11~.
- 필터: 기간이 `[start, end]`와 겹치는 사건(`start_date ≤ end` 그리고 `end_date ≥ start`). `aoi`는
  시군구(`SGG_<cd>`)면 `sgg_all`에 그 이름이 있는 사건, 시도는 `sido_nm`, 국가는 `country`, 아르헨티나 1급은 `adm1_code`
  (모두 감지 점 위치 기준 가중 다수결; `sgg_all`은 점이 하나라도 있는 시군구 전부).
- 사건 영역: `lat_min/lat_max/lon_min/lon_max`(감지 점 상자)와 `centroid_lat/lon`. 지도에는 상자 또는 중심 원으로.
- 응답 필드: `event_id`, `start_date`, `end_date`, `duration_days`, `n_det`(감지 수), `sum_weight`(VIIRS 상당 감지 수),
  `frp_sum`, `frp_max`, 영역, `burned_area_km2`, `burn_confirmed`, `sgg_all`.
- **`burned_area_km2` 표시 문구**: "위성 확인 연소 면적(하한)" — MODIS MCD64A1(500 m)에서 사건과 겹친 불탄 픽셀 면적.
  큰 사건(Σ가중치 ≥ 100 또는 감지 ≥ 50)은 감지 점 볼록 껍질 + 2 km(±1개월), 나머지는 감지 점 1 km 안(±3일).
  의성 2025-03 기준 공식 피해 면적의 약 64%라 **실제보다 작다.** 2012~2025 밖 사건은 null(자료 없음), 0이면 "확인 안 됨".
  `burned_area_km2_near`(1 km 규칙 값)는 비교용이라 화면에 쓰지 않는다.
- 기본 정렬 `sum_weight` 내림차순, `limit` 기본 100. 감지 1개짜리 사건이 한국 81%라 목록에는 `n_det ≥ 3` 필터를 권한다.

### `/regime?aoi`

- 읽는 파일: `data/processed/aoi_month_fire_regime.parquet` (AOI × 월, 한국 KR·시도·시군구, 아르헨티나 ADM0_ARG·1급).
- `fire_regime` ∈ `large_fire`(큰 사건이 그 달 활동의 50% 이상) / `many_small`(20% 미만) / `mixed`(그 사이) / `none`(감지 없음).
  함께 주는 값: `W`(그 달 Σ가중치), `share_large`, `n_events`, `n_large_events`, `top_event_id`(→ `/events`로 연결).
- **캘린더 값과 다를 수 있음**: fire_regime은 감지 점 위치로 시군구를 정하고, `aoi_month`는 칸 면적 비율로 나누기 때문에
  `aoi_month`에 작은 값이 있는 시군구라도 fire_regime이 `none`일 수 있다. **캘린더 값은 `aoi_month`, 불 유형 배지는 `fire_regime` 기준.**

### `/fwi?aoi&date` (2026-11-15 추가)

- 읽는 파일: `data/processed/fwi_aoi_day.parquet` (AOI × 날짜, 한국 KR·시도·시군구, 아르헨티나 ADM0_ARG·1급; 설명 `data/processed/fwi_meta.json`).
- 돌려주는 값: `fwi_mean`(AOI 면적 가중 평균), `fwi_max`, **절대 등급** `fwi_class`, **백분위** `fwi_pct`(0~100), **상대 등급** `fwi_rel_class`.
  - 절대 등급 `fwi_class`(캐나다 전국 기준): `low` < 5.2 ≤ `moderate` < 11.2 ≤ `high` < 21.3 ≤ `very_high` < 38 ≤ `extreme`.
  - 상대 등급 `fwi_rel_class`: 같은 AOI·같은 달의 2012~2025 일 FWI 분포에서 그날 값의 백분위 →
    `below_normal`(평년보다 낮음, < 25) / `normal`(평년, 25~75) / `above_normal`(높음, 75~90) / `well_above_normal`(매우 높음, ≥ 90).
- **앱 기본 배지는 상대 등급(`fwi_rel_class`)**. 절대 등급은 상세 화면에 함께 보여 준다.
  **아르헨티나는 캐나다 절대 등급이 잘 맞지 않음** (건조 지역이 늘 `high` 이상으로 나오고, `extreme`에서 다음 날 새 감지가 오히려 줄어듦 → `reports/fwi_validation.csv`).
  상대 등급은 두 나라 모두 등급이 오를수록 다음 날 새 감지가 늘어남 → `reports/fwi_validation_relative.csv`.
- 2026년 날짜(`fwi extend`로 이어 계산): `fwi_pct`는 비어 있고, 상대 등급은
  `fwi_clim_aoi_month.parquet`(AOI × 달 경계값 `q25`·`q75`·`q90`, 2012~2025)로 정한다.
- **기준일 = 마지막 POWER 자료일** (현재 2026-11-11, 오늘보다 약 4일 늦음). `date`를 안 주면 그 AOI의 마지막 날을 돌려주고,
  응답에 `as_of_date`(= 그 날짜)를 넣는다. **화면에 기준일을 꼭 표시** (예: "산불 기상 위험도 · 11월 11일 기준").
  기준일보다 뒤의 날짜를 요청하면 값 대신 `null` + `as_of_date`.
- 표시 문구(권장): "기상으로 본 산불 위험도(캐나다 FWI, NASA POWER 일 자료 근사). 실제 발화·확산을 예측하지 않음."

```json
// GET /fwi?aoi=SGG_47730&date=2025-03-25
{"aoi_id": "SGG_47730", "name": "경상북도 의성군", "date": "2025-03-25",
 "fwi_mean": 22.3, "fwi_class": "very_high", "fwi_pct": 97.4, "fwi_rel_class": "well_above_normal", "badge": "fwi_rel_class"}
```

### `POST /harmonize` — 사용자 FIRMS 파일 변환 (2026-11-15)

- 하는 일: 사용자가 올린 **FIRMS 원본 CSV 또는 zip**을 통합 핫스팟 형식으로 바꾸고 요약을 돌려준다. 서버 데이터는 바꾸지 않는다.
- 내부 호출: `planetarium.converter.harmonize_file(path, output_dir, fmt="csv")` (CLI: `uv run python -m planetarium.harmonize --input <파일…> --output <폴더> --format csv`).
- 요청: `multipart/form-data`
  - `file` (필수): `.csv` 또는 `.zip` 하나. MODIS C6.1 / VIIRS 375 m C2, 연간 아카이브·Archive Download·최근 7일·area API 형식 모두.
    zip은 안의 MODIS/VIIRS CSV를 모두 읽고 Landsat·readme는 건너뜀.
  - `format` (선택): `csv`(기본) | `parquet`.
- 응답: `200 application/zip` — 아래 4개 파일. 오류는 `400`(형식 아님·Landsat만 있음·빈 파일), `413`(크기 초과).

| 파일 | 내용 |
|---|---|
| `harmonized_hotspots.csv` | 통합 핫스팟과 같은 컬럼 (`harmonized_weight`, `weight_lo/hi`, `weight_source`, `is_reference`, `cell_id`, `date_local` …) |
| `summary_cell_day.csv` | 0.1° 칸 × 현지 날짜: 기준 점 가중치 합 `value`, `value_lo/hi`, `n_points`, `pending` |
| `summary_aoi_month.csv` | AOI × 월: 기준 점 가중치의 면적 가중 합 `value`, `n_points` (입력 파일에 든 점만) |
| `summary.json` | `rows_in`, `rows_unique`, `rows_out`, `reference_rows`, `pending_new_month_rows`, `months`, `rows_by_satellite`, `notes`, `seconds`, `peak_memory_gb` |

- 규칙
  - 같은 키(위성·위도·경도·관측 시각) 중복은 하나만. 고정 열원 제외: SP는 `type 0`만, NRT는 고정 열원 0.01° 칸 제외 (→ `rows_out < rows_unique`).
  - v1.2: SP `type 0`이라도 고정 열원 0.01° 칸 안의 점은 `is_static_source = true`로 표시하고 요약(`summary_*`)에서 뺀다 (`summary.json`의 `static_source_rows`).
  - 점 가중치: 서버에 같은 점이 있으면 그 가중치를 그대로, 없으면 그 달 규칙(VIIRS 기준 위성 = 1, NOAA-20 연결 비율, MODIS = FireBridge 예측)으로.
  - 서버 통합표에 없는 달의 기준 점은 `weight_source = pending_new_month`, `harmonized_weight = null` (새 달 MODIS 예측은 아직 없음).
  - 2012년 이후 MODIS 점은 `is_reference = false`, `weight_source = m2v_predicted`로 함께 나온다. 서버의 `harmonized_hotspots`에는
    VIIRS 시대 MODIS 점이 들어 있지 않으므로 비교할 때는 빼고 본다.
  - 요약은 **입력 파일의 점만** 더한 값: 일부 지역·기간이면 `aoi_month`보다 작고, MODIS 기간 미감지 추정분(residual)은 들어가지 않는다.
- 처리량(실측 2026-11-15, 전 세계 7일 파일): VIIRS 44만 행 1.4~2.1초, MODIS 8.8만 행 13.8초, 최대 메모리 3.7 GB.
  권장 제한: 업로드 200 MB, 그 이상은 비동기(`202` + `GET /harmonize/{job_id}`)로.
- 검증 (2026-11-15): 최근 7일 파일 3개(NOAA-20·NOAA-21·MODIS, 2026-11-06~11-13 UTC)를 변환해 서버 통합 핫스팟과 키로 비교.
  NOAA-20 417,173점·NOAA-21 414,778점이 양쪽에 있고, **기준 여부 불일치 0, 가중치 차이 최대 0.0**, 가중치 합 같음 (421,383.5 / 432,352.4).
  서버에만 있는 점: NOAA-20 1,532개·NOAA-21 4개(모두 11-13, 이후 받은 API 파일분). 변환기에만 있는 점: NOAA-20 8개(11-11 러시아 남부 한 장면).

### 다음 달 평년 수준 (예측 아님) — `/calendar` 또는 `/compare` 응답에 덧붙임

- 뜻: **"예년 이맘때(다음 달)에는 보통 이 정도"** — 2001~2025 같은 달 값의 중앙값과 25~75% 범위. 예측이 아니다.
- 계산 규칙은 `aoi_month_compare`의 기준 분포와 같다: 같은 달, 지역의 `baseline_period`(2001~2025 또는 2012~2025), `status == ok`인 해만.
  - 가장 간단한 방법: `aoi_month`에서 `aoi_id`, `month = 다음 달`, `year ∈ [2001, 2025]`, `status = ok`인 `value`의 중앙값·25%·75%.
  - `aoi_month_compare`의 직전 같은 달 행(예: 2025-12)의 `baseline_median`·`baseline_p25`·`baseline_p75`를 써도 된다.
    다만 그 행은 자기 해(2025)를 뺀 24년 기준이라 값이 조금 다르다 (KR 12월: 29.4 대 29.5).
- 응답 필드(권장): `next_month_normal = {"month": 12, "median": …, "p25": …, "p75": …, "n_years": 25, "label": "평년 수준(예측 아님)"}`.
- 표시 문구: "다음 달 평년 수준 — 지난 25년 같은 달의 가운데 값(예측 아님)". 범위(p25~p75)는 막대로 함께.
- 예 (2026-12 기준, aoi_month 2001~2025 12월):

| AOI | 중앙값 | 25% | 75% | 해 수 |
|---|---|---|---|---|
| KR | 29.4 | 28.0 | 35.2 | 25 |
| ADM0_ARG | 9,992 | 6,609 | 13,658 | 25 |

- 한국 값에는 산업 열원으로 보이는 감지가 섞여 있다(알려진 한계, `backend-handoff.md`). 아르헨티나는 1% 남짓.
- 아르헨티나 칸 단위 다음 달 확률은 `/recurrence`, 한국은 반복 횟수(`/recurrence`)를 쓴다.

### `/occurrence?aoi` — FireOutlook Monthly – District (internal id: `aoi_occurrence`): 다음 달 열원 발생 확률, 한국 시군구 (2026-11-15)

- 읽는 파일: `data/processed/aoi_occurrence_nextmonth_latest.parquet` (한국 시군구 252개; 설명 `models/aoi_occurrence_meta.json`).
  아르헨티나는 시군구급(departamento) AOI가 없어 대상이 아니다 → 아르헨티나는 칸 단위 `/recurrence` 확률을 쓴다.
- 컬럼: `aoi_id`, `target_month`(현재 `2026-12`), **`p_any`**(다음 달 그 시군구 안에서 VIIRS 기준 감지가 1건 이상일 확률, 고정 열원 제외),
  `p_none` = 1 − `p_any`, **`baseline_p`**(예년 같은 달 발생 비율, 2012 ~ 작년), `model`, `as_of_date`(마지막 반영일),
  참고 `rate_5y`, `n_years`, `lag1_projected`(직전 달이 진행 중이라 월말 추정값을 썼으면 true).
- 표시 (권장)
  - 큰 숫자: "다음 달 열원 1건 이상 가능성 **N%**" (`p_any`).
  - 예년 대비: `baseline_p`와 비교 — `p_any − baseline_p ≥ 0.10` "예년보다 높음", `≤ −0.10` "예년보다 낮음", 그 사이 "예년 수준".
    작은 글씨로 "예년 같은 달 발생 비율 M%".
  - 필수 문구: "과거 감지 기록과 최근 활동으로 본 확률입니다. 날씨 예보가 아니며, 공장 등 산업 열원으로 보이는 감지가 일부 섞여 있을 수 있습니다."
- 모델: FireOutlook Monthly – District = XGBoost(이진) + Platt 보정. 확장 창 검증(2016~2025, 30,240 시군구·월, 양성 13%): PR-AUC 0.422 / Brier 0.0927,
  기준선 "예년 같은 달 발생 비율" 0.404 / 0.0964, "작년 같은 달 발생 여부" 0.223 / 0.163 → 채택 (`reports/occurrence_metrics*.csv`,
  `reports/figures/occurrence_calibration.png`). 기준선을 이기지 못하는 갱신에서는 `model = baseline_same_month_rate`, `p_any = baseline_p`로 나간다.
- 참고: 시군구·달력 월 조합의 45%는 2012~ 동안 감지가 한 번도 없었다(양성 비율 < 5%). 95%를 넘는 조합은 0.07%.
  울산 남구·포항 남구·울주·당진·광양처럼 산업 지역이 상위에 오르는데, 고정 열원 목록(2개 연도 이상 type 1·2)에 없는 산업 열원이 남아 있을 수 있다.
- **`suspected_industrial = true`(산업 열원 의심) 시군구는 순위·강조(상위 N, 색 강조, 알림)에서 제외하고 회색으로 표시**한다. 값은 상세 화면에서만 보여 준다.
  기준: 2012-02 이후 감지가 있던 달의 비율 ≥ 25% **그리고** 계절성이 거의 없음(12개 달력 월 발생 비율의 변동계수 ≤ 0.40, 한국 전체는 0.48)
  **그리고** 가장 적은 달도 ≥ 10%. 현재 4곳: 울산 남구, 포항 남구, 함안군, 제주시 (`reports/occurrence_industrial_flags.csv`).
  제주시는 산업 시설이라기보다 1년 내내 이어지는 소각일 가능성도 있어 "의심"으로만 둔다. 울주·당진·광양은 계절성이 있어 기준에 걸리지 않았다.

```json
// GET /occurrence?aoi=SGG_50110
{"aoi_id": "SGG_50110", "target_month": "2026-12", "p_any": 0.34, "p_none": 0.66, "baseline_p": 0.57,
 "model": "aoi_occurrence_xgb+platt", "as_of_date": "2026-11-14", "vs_normal": "예년보다 낮음"}
```

#### `/occurrence?unit=grid&bbox` — FireOutlook Monthly – Grid (internal id: `grid_occurrence`): 전 세계 0.5° 격자 (2026-11-15)

- 읽는 파일: `data/processed/grid_occurrence_nextmonth_latest.parquet` (설명 `models/grid_occurrence_meta.json`).
  대상: 2012-02 이후 한 번이라도 탐지된 육지 0.5° 격자 52,855개 (남극·공해 제외). `unit`을 안 주면 시군구(`aoi`) 응답.
- 컬럼: `cell_id`(0.5° 격자 id = 행 × 720 + 열), `lat`, `lon`(격자 중심), `target_month`, **`p_any`**(다음 달 그 격자에 기준 감지 ≥ 1건일 확률,
  고정 열원 제외), `p_none`, **`baseline_p`**(예년 같은 달 발생 비율), **`persistent_source`**, `model`, `as_of_date`.
- **`persistent_source = true`(지속 열원 의심: 가스 플레어·제철소 등) 격자는 순위·강조에서 빼고 회색**으로 그린다 (시군구의 `suspected_industrial`과 같은 규칙).
  기준: 2012-02 이후 탐지 달 비율 ≥ 0.75, 달력 월별 평균 탐지 수의 변동계수 ≤ 1.0, 기준 점의 상위 3개 0.01° 칸 집중 비율 ≥ 0.12 → 259개 격자.
  울산·포항·바스라·퍼미안(뉴멕시코 쪽)은 잡히고 아마존·콩고 분지·호주 북부 사바나는 걸리지 않는다. 넓게 퍼진 유전(니제르 델타, 퍼미안 델라웨어 쪽)과
  고정 열원 목록으로 이미 대부분 빠진 곳(한티만시, 당진, 광양)은 잡히지 않는다 (`reports/grid_occurrence_persistence_check.csv`).
- 표시: 색 단계 < 10% / 10~40% / 40~70% / ≥ 70%, 예년 대비는 시군구와 같은 규칙(`p_any − baseline_p` ±0.10).
  필수 문구: "과거 감지 기록과 최근 활동으로 본 확률입니다. 날씨 예보가 아닙니다. 회색 격자는 공장·가스 플레어 같은 지속 열원 의심 지역입니다."
- 검증 (확장 창 2020~2025, 지속 열원 제외): 전체 PR-AUC 0.900 / Brier 0.096, 기준선 "예년 같은 달 비율" 0.869 / 0.104 → 6개 대륙 모두와
  한국·아르헨티나에서 기준선을 이겨 채택 (`reports/grid_occurrence_metrics.csv`, `reports/figures/grid_occurrence_calibration.png`). 보정은 isotonic.

### `/recurrence?aoi|bbox` — FireOutlook Recurrence (internal id: `recurrence_nextmonth`), 확정 (2026-11-15)

- 읽는 파일: `data/processed/recurrence_nextmonth_latest.parquet` (0.1° 칸; 설명 `models/recurrence_nextmonth_meta.json`).
  **고정 열원 칸(`static_source_cells`와 겹치는 0.1° 칸)은 파일에서 이미 뺐다** (아르헨티나 744칸, 한국 160칸).
- 대상 달: `year`·`month` (현재 2026-12). 직전 달은 진행 중 값(월말 추정)을 쓰므로 응답에 `as_of_date`(마지막 반영일)를 넣는다.
- 나라마다 레이어가 다르다 (`layer` 컬럼):

| 나라 | `layer` | 보여 줄 값 | 표시 |
|---|---|---|---|
| 아르헨티나 | `probability` | `p_active` (다음 달 VIIRS 기준 감지값 ≥ 1일 확률, FireOutlook Recurrence XGBoost) | "과거 반복 패턴 기반 전망" + 예측 정확도. 단계 < 10% / 10~40% / ≥ 40% |
| 한국 | `recurrence_count` | `active_years` / `n_years` (지난 해들 중 이 달 활성 횟수) | **"지난 25년 중 12월에 N번 활성"** — **확률이라고 부르지 않는다**. `p_active`는 비어 있음 |

- 예측 정확도(아르헨티나, 확장 창 2016~2025, 고정 열원 칸 제외): PR-AUC 0.57 (반복률만 0.49), ROC-AUC 0.92, 확률 보정 양호
  → `reports/recurrence_metrics_no_static.csv`, `reports/figures/recurrence_calibration.png`.
- 필수 문구: 아르헨티나 "과거에 불이 반복된 곳·시기를 바탕으로 한 확률. 기상 예보나 발화 예측이 아니며, 처음 불이 나는 곳은 낮게 나옴."
  / 한국 "위성 감지 기록에서 이 달에 불이 관측된 해의 수. 앞으로의 발생 확률이 아님."
- 함께 주는 값: `active_years_last3`(최근 3년 같은 달 활성 횟수), `rec`(= active_years / n_years).
- 상습 연소 지점 `recurrent_burn_sites.parquet` (고정 열원 칸 제외본): 같은 달 2001~2025 유효한 해의 70% 이상 활성인 칸·달
  (`rate`, `n_active`, `n_years`, `admin_name`). 현재 한국 6건(화천·철원·연천 3~4월), 아르헨티나 6,404건.

```json
// GET /recurrence?aoi=ADM0_ARG  (상위 1칸)
{"target_month": "2026-12", "as_of_date": "2026-11-14", "layer": "probability", "model": "recurrence_nextmonth_xgb",
 "cells": [{"cell_lat": -38.15, "cell_lon": -68.95, "p_active": 0.78, "active_years": 10, "n_years": 25, "active_years_last3": 1}]}
// GET /recurrence?aoi=KR  (상위 1칸)
{"target_month": "2026-12", "as_of_date": "2026-11-14", "layer": "recurrence_count",
 "cells": [{"cell_lat": 33.25, "cell_lon": 126.35, "p_active": null, "active_years": 4, "n_years": 25, "active_years_last3": 0}]}
```


### `POST /forecast` — FireOutlook Planner: 원하는 지역 + 원하는 기간의 열원 전망 (2026-11-15, 실험 기능)

- **모델 이름과 관계** (파일 이름·코드 식별자·`model` 값은 내부 id 그대로)

| 표시 이름 | 내부 id (파일·코드) | /forecast 안에서의 역할 |
|---|---|---|
| **FireOutlook Planner** | `planner` — `planetarium/forecast.py`, `planner_*` 파일 | 전체: 지역 + 기간 질의를 일 단위·월 단위 조각으로 나눠 합친다 |
| FireOutlook Planner 일 단위 | `planner_daily` — `planner_daily_grid.parquet` | 1~7일 뒤 (0.5° 격자 XGBoost + LightGBM) |
| FireOutlook Planner 월 단위 | `planner_monthly` — `planner_monthly_grid.parquet` | 2~12개월 뒤 격자, 그리고 지역 합산의 바탕 |
| **FireOutlook Monthly – Grid** | `grid_occurrence` — `grid_occurrence_nextmonth_latest.parquet` | 0.5° 격자 다음 달(h=1)은 이 값을 그대로 씀 (`/occurrence?unit=grid`와 같은 숫자) |
| **FireOutlook Monthly – District** | `aoi_occurrence` — `aoi_occurrence_nextmonth_latest.parquet` | 한국 시군구 다음 달은 이 값을 그대로 씀 (`/occurrence`와 같은 숫자). 2·3개월은 같은 방식의 `planner_sgg` |
| **FireOutlook Spread** | `spread_nowcast` — `spread_nowcast_latest.parquet` | **쓰지 않음.** 내일 확산 가능성은 별도 API `/spread` (0.025° 후보 칸, 한국·아르헨티나) |
| FireOutlook Recurrence | `recurrence_nextmonth` | 쓰지 않음 (별도 API `/recurrence`) |

- 함수: `planetarium.forecast.forecast(region, start=None, end=None, as_of=None, horizon_days=None, today=None)`
  (CLI `uv run python -m planetarium.forecast --region SIDO_51 --horizon-days 30`, 또는 `--start 2026-11-15 --end 2026-11-21`).
  `/occurrence`와 같은 값인지 확인: `uv run python -m tests.test_forecast_consistency`.
  미리 계산한 0.5° 격자 예측만 모아서 답하므로 응답은 1초 안쪽이다.
  - 월 단위: `data/processed/planner_monthly_grid.parquet` (격자 × 1~12개월 뒤, 기준 달 2026-11)
  - 일 단위: `data/processed/planner_daily_grid.parquet` (격자 × 1~7일 뒤, 기준일 2026-11-14)
  - 지역 보정: `data/interim/planner/monthly/region_calibration.json`, aoi 가중치 `data/interim/planner/aoi_W.npz`
- 요청

```json
// POST /forecast
{"region": "SIDO_51", "start": "2026-11-15", "end": "2026-11-21"}
// region: aoi_id(aoi_list 5,051개) | [서, 남, 동, 북] | GeoJSON Polygon/MultiPolygon
// 기간: 기준일 다음 날부터, 기준 달에서 12개월 뒤 달까지
```

- **30일 고정 질의 (화면 기본)**: `{"region": "...", "horizon_days": 30}` → 기간 = 오늘(KST) ~ 오늘 + 29일.
  `start`를 함께 주면 그날부터 30일. 오늘이 기준일(as_of) 이하이면 as_of + 1일부터 시작한다.
  as_of(마지막 완결 관측일)와 오늘 사이가 비어도 괜찮다: 오늘은 as_of에서 `lead_days_from_as_of`일 앞선 예측으로 계산된다
  (예: as_of 11-14, 오늘 11-15 → 1일 뒤 예측부터; 자료가 3일 늦으면 오늘 = 3일 뒤 예측, 일 단위가 덮는 날은 7일 뒤까지로 줄어든다).

```json
// POST /forecast {"region": "SIDO_51", "horizon_days": 30}     (오늘 2026-11-15 KST; daily[]는 30개 중 4개만 표시)
{"product": "FireOutlook Planner", "region": "aoi_id SIDO_51", "region_group": "Korea sido",
 "start": "2026-11-15", "end": "2026-12-14", "as_of": "2026-11-14", "lead_days_from_as_of": {"first": 1, "last": 30},
 "p_any": 0.619, "expected_count": 1.37, "expected_count_q10": 0.0, "expected_count_q90": 3.02, "expected_fire_days": 0.94,
 "baseline": {"p_any": 0.5385, "count": 1.28, "expected_fire_days": 0.76, "definition": "예년 같은 기간 (2012~ 같은 달 비율·중앙값)"},
 "skill": "high", "count_skill": "none", "value_source": "model_calibrated",
 "skill_basis": "30-day holdout validation 2024-2025 (Korea sido; reports/planner_30d_validation.csv)", "display_hint": "",
 "daily": [
  {"date": "2026-11-15", "lead_days_from_as_of": 1, "p_day": 0.0631, "base_p_day": 0.0647, "expected_count_day": 0.123, "skill": "low", "source": "daily model"},
  {"date": "2026-11-21", "lead_days_from_as_of": 7, "p_day": 0.0441, "base_p_day": 0.0384, "expected_count_day": 0.26, "skill": "low", "source": "daily model"},
  {"date": "2026-11-22", "lead_days_from_as_of": 8, "p_day": 0.0292, "base_p_day": 0.0245, "expected_count_day": 0.016, "skill": "none", "source": "baseline (current month)"},
  {"date": "2026-12-14", "lead_days_from_as_of": 30, "p_day": 0.0173, "base_p_day": 0.0145, "expected_count_day": 0.0, "skill": "none", "source": "monthly model h=1 x daily weights"}],
 "notice": "위성 탐지 기반 확률이며 산불 발생을 보장하지 않습니다. (Probability of satellite fire detections; it does not guarantee that a wildfire will occur.)"}
// skill none 등급의 예 — POST /forecast {"region": "SGG_47730", "horizon_days": 30}
{"p_any": 0.0283, "baseline": {"p_any": 0.0283}, "skill": "none", "value_source": "baseline", "daily": [{"p_day": "= base_p_day", "...": "..."}]}
```

- **30일 요약 skill** (`skill_basis`가 30-day holdout이면): 28~31일 질의이고 시작일이 as_of + 3일 이내일 때, 조각별 최솟값 대신
  "기준일 다음 날부터 30일" 홀드아웃 검증(2024–2025, 기준일 51개)의 등급별 값을 쓴다. 기준선 = 예년 같은 기간.
  bbox·폴리곤은 0.5° 격자 수가 비슷한 등급으로 본다(1칸 = 격자, < 1.5 = 시군구, < 10 = 시도, < 150 = adm1, < 3000 = 나라).

| 단위 | 관측 비율 | 평균 예측 (모델 / 기준선, 보정 전) | BSS raw / 교차 보정 | 확률 skill | 응답 값 (`value_source`) | 건수 WAPE (모델 / 기준선) | 건수 skill |
|---|---|---|---|---|---|---|---|
| 0.5° 격자 한 칸 | 0.350 | 0.363 / 0.369 | +10.1% / +8.8% | high | `model_calibrated` | 0.589 / 0.621 (+5.2%) | low |
| 한국 시군구 | 0.066 | 0.107 / 0.132 | +12.9% / +0.9% | none | `baseline` | 1.376 / 1.118 (−23%) | none |
| 한국 시도 | 0.544 | 0.578 / 0.629 | +10.6% / +6.4% | high | `model_calibrated` | 1.136 / 0.973 (−17%) | none |
| adm1 (아르헨티나 주 등) | 0.534 | 0.565 / 0.582 | +5.7% / +2.1% | low | `model_calibrated` | 0.422 / 0.461 (+8.3%) | low |
| 나라 | 0.741 | 0.727 / 0.733 | −1.3% / −0.7% | none | `baseline` | 0.320 / 0.360 (+11%) | low |
| 대륙 · 전 세계 | 1.000 | 1.0 / 1.0 | 정보 없음 | none | `baseline` | 0.236 / 0.272 (+14%) · 0.149 / 0.184 (+19%) | low |

  - **`value_source`** (30일 질의에서만): **"평년을 못 이기면 평년을 보여 준다."**
    - `baseline`: 확률 skill이 none인 등급. `p_any` = 보정한 평년 확률(`baseline.p_any`와 같음), `daily[].p_day` = `base_p_day`.
    - `model_calibrated`: high·low 등급. `p_any` = 등급별 isotonic으로 보정한 모델 확률. `daily[].p_day`는 날짜 사이 모양을 유지한 채
      hazard 비율로 함께 조정해 `1 − Π(1 − p_day) = p_any`가 되게 한다. 평년(`baseline.p_any`, `base_p_day`)도 같은 등급의 평년용 isotonic으로 보정한다.
    - 그 밖의 질의(30일이 아니거나 시작이 as_of + 3일 뒤)는 `value_source = "model"`(조각 합성 그대로).
    - 보정 곡선은 홀드아웃 2024–2025 전체로 맞춰 `planner_30d_skill.json`에 들어 있다. 보정 후 신뢰도(한 해로 맞추고 다른 해에서 확인,
      `reports/planner_30d_calibration_after.csv`): 평균 예측 / 관측 = 격자 0.351 / 0.350, 시도 0.550 / 0.544, adm1 0.535 / 0.534,
      시군구 0.068 / 0.066. 10분위별 최대 차이는 격자 0.004, adm1 0.02이고, 표본이 적은 시도·시군구 상위 구간은 0.14까지 벌어진다.
    - `p_any ≥ 0.999`처럼 거의 확실한 경우에는 날짜별 값을 조정하지 않는다.
    - **월 조각 확률 ≥ 0.98 (큰 지역)**: 1 − (1 − p)^f로 나누면 하루 확률도 ≈ 1이 되어 예상 탐지일이 부풀려진다. 그런 조각(모델·평년 각각)의
      `p_day`·`base_p_day`는 **그 지역에서 과거 같은 날짜(±7일)에 탐지가 1건 이상 있던 날의 비율**(2012–2023, `planner_day_any.npz`)로 쓴다.
      aoi_id는 지역 자체 기록(0.1° 칸 소속 가중치 ≥ 0.5), bbox·폴리곤은 0.5° 격자 값을 독립 가정으로 합친다. 최근 상태를 반영할 일 단위
      정보가 없으므로 모델·평년 모두 기후값 그대로다 (1~7일은 일 단위 모델이 그대로 쓰임). 30일 예상 탐지일 확인
      (`reports/planner_30d_fire_days_check.csv`, 2026-11-15 ~ 12-14, 실제 = 2016–2025 같은 기간 평균):
      코르도바 수정 전 모델 19.0 · 평년 29.5 → 수정 후 25.3 · 25.4 (실제 25.1); 아르헨티나·전 세계 30 · 30 (실제 30.0);
      강원(월 확률 < 0.98) 0.94 · 0.76 그대로 (실제 1.3).
    - `daily[].skill`은 그날 값의 근거가 된 조각의 skill이고, 요약 `skill`과 다를 수 있다 (요약은 30일 전체 검증 기준).

  - **조각 합성(1 − Π) 과대평가**: 격자는 거의 맞다(예측 0.36 / 관측 0.35, 10분위 모두 +0.02 이내).
    지역은 과대평가다 — 시군구 0.107 / 0.066, adm1 0.5~0.9 구간에서 예측이 관측보다 0.05~0.10 높다 (`reports/planner_30d_calibration.csv`).
    그래서 BSS는 교차 보정(2024로 맞추고 2025에 적용, 반대도)과 raw 중 **작은 값**으로 skill을 정했고, 응답 값은 위 `value_source` 규칙대로
    보정·평년 대체를 한다. 모델은 바꾸지 않았다.
  - 이 검증의 지역 값은 격자 hazard 합(raw)이다. 서비스의 월 조각은 지역 보정·기준선 라우팅을 거치므로 값이 조금 다를 수 있다.
  - 응답 시간 (warmup 뒤, 30일 질의): 시군구·시도·주·bbox·폴리곤 0.006~0.012초, 전 세계 0.26초 (`reports/planner_30d_latency.csv`).

- `daily[]` 필드 (날짜 순, 30일 질의면 30개)

| 필드 | 형식 | 뜻 |
|---|---|---|
| `date` | `YYYY-MM-DD` | 날짜 (현지 날짜 기준 집계) |
| `lead_days_from_as_of` | 정수 | as_of에서 며칠 뒤 예측인가 (1 = as_of 다음 날) |
| `p_day` | 0~1 | 그날 위성 탐지 ≥ 1건 확률 |
| `base_p_day` | 0~1 | 예년 같은 날 확률 (기준선) |
| `expected_count_day` | 실수 | 그날 예상 탐지 수 (S-NPP 기준) |
| `skill` | `high` · `low` · `none` | 그날 값의 근거가 된 조각의 skill (`none`이면 기준선 값) |
| `source` | 문자열 | `daily model` / `monthly model h=N x daily weights` / `baseline (current month)` (참고용) |
  1~7일 뒤는 일 단위 모델, 8일 이후는 월 모델 × 확률용 일별 기후 가중치(진행 중인 달의 남은 날은 예년 기준선).
  날짜별 hazard를 더하면 조각 값과 같다: `1 − Π(1 − p_day) = p_any`. `expected_fire_days = Σ p_day`(탐지가 있을 것으로 기대되는 날 수).
- 응답 필드: `p_any`(기간 안에 위성 탐지 ≥ 1건 확률), `expected_count`(S-NPP 기준 탐지 수) · `expected_count_q10` · `expected_count_q90`,
  `baseline`(예년 같은 기간: `p_any`, `count`), `skill`(확률, `high` | `low` | `none`), `count_skill`(건수), `models`, `horizons`, `as_of`,
  `pieces`(조각별 값과 `route`), `display_hint`, `suspected_industrial_sgg`(한국 시군구 산업 열원 의심),
  `persistent_source_cells`(지속 열원 의심 격자 수·면적 비율), `notice`.
- **기간 조각 규칙 (겹침 처리)**
  1. 기준일 + 1 ~ + 7일 안의 날짜는 일 단위 예측(하루별 hazard)으로 계산한다.
  2. 나머지 날짜는 달별로 묶어 월 단위 예측을 쓴다. 달의 일부만 들어가면 **일별 기후값 가중치**로 비율을 정한다 (2026-11-15 변경):
     - 건수 f_c = 그 날짜들의 기후 탐지 수 합 / 그 달 전체 합, 확률 f_p = 같은 계산을 "탐지가 있던 날 비율" 기후값으로.
       기후값은 칸별 2012–2025 같은 날짜 평균의 ±7일 이동평균. 그 칸·그 달 과거 탐지일이 15일 미만이면 같은 나라 → 대륙 모양으로 대체
       (현재 칸 자체 34%, 나라 65%, 대륙 0.6%).
     - 지역은 칸별 f_c를 예년 중앙값 건수로, f_p를 격자 hazard로 가중 평균한다.
     - p = 1 − (1 − p_달)^f_p, 건수·구간 × f_c. 응답 `pieces[]`에 `fraction_count`, `fraction_prob`, `fraction_method`가 들어 있다.
     - 검증 (홀드아웃 2024–2025, 기준일 91개 × "8~90일 뒤 7일" 구간, 0.5° 격자, 기후값은 2012–2023만):
       건수 WAPE 0.876 → 0.853, Brier 0.0857 → 0.0853. k = 8–14 / 15–30 / 31–60 / 61–90일 모든 구간에서 둘 다 개선
       (`reports/planner_dayweights_validation.csv`). 건수 모양을 확률에도 쓰면 Brier가 나빠져서(0.0870) 확률용 가중치를 따로 둔다.
     - 일 단위가 이미 쓴 날짜는 빼고 계산한다 (같은 날을 두 번 세지 않음). 한 달 전체면 f = 1.
  3. 진행 중인 달(기준 달)의 남은 날짜 중 일 단위가 덮지 못한 날은 월 예측이 없으므로 예년 기준선에 같은 기후 가중 비율을 곱해 쓰고 그 조각은 `skill = none`.
  4. 조각 합치기: `p_any = 1 − Π(1 − p_조각)` (조각끼리 독립 가정 → 긴 기간은 조금 높게 나올 수 있음), 건수·구간은 합.
- 지역 확률: 격자 확률을 hazard로 더한 뒤(`1 − exp(−Σ w·(−log(1 − p)))`) 지역 크기 구간별 isotonic으로 보정 (월 단위; 2020–2023 학습, 2024–2025 확인).
  일 단위는 격자에서만 검증했으므로 여러 칸을 모은 지역은 `skill` 상한이 `low`다.
- **라우팅 (한 대상·한 시점에는 숫자 하나)** — 응답 `pieces[].route`·`model`에 어느 쪽이 나갔는지 적힌다.

| 대상 | 다음 달 (h=1) | 2·3개월 | 4~12개월 | 1~7일 |
|---|---|---|---|---|
| 0.5° 격자 한 칸 | 기존 FireOutlook Monthly – Grid 값 그대로 (= `/occurrence?unit=grid`) | Planner 월 ens (xgb+lgb) | Planner 월 ens | Planner 일 ens |
| 한국 시군구 | 기존 FireOutlook Monthly – District 값 그대로 (= `/occurrence`) | 시군구 직접 모델 (`planner_sgg`, skill low) | 격자 합산 + 보정 (6개월까지 low, 이후 기준선) | 격자 합산 (low) |
| 그 밖의 지역·bbox·폴리곤 | 격자 합산 + 보정 (전체·adm1 low, 국가·한국 시도는 기준선) | 기준선 (skill none) | 기준선 (skill none) | 격자 합산 (low) |

- **건수 라우팅**: 등급 × 시점마다 검증 WAPE로 고르고, 확인 구간에서 예년 중앙값 대비 WAPE 개선이 5% 미만이면 예년 중앙값을 내고 `count_skill = none`
  (`reports/planner_count_routing.csv`). 전 세계는 AutoTheta/AutoETS(개선 29~57%, high), 대륙은 격자 μ 합(6개월까지 7~20%, low),
  아르헨티나 전체는 1개월만 AutoETS(23%), 국가·adm1·한국(전체·시도·시군구)은 예년 중앙값. 일 단위 건수는 격자 μ(기후값 대비 23~31% 개선).
- **skill 표시 규칙**: 조각 중 하나라도 `none`이면 `none`, 아니면 가장 낮은 값.
  - `high`(기준선 대비 Brier 개선 ≥ 5%): 확률을 큰 숫자로.
  - `low`(1~5%): 확률 + "예년 수준과 큰 차이 없음" 보조 문구.
  - `none`(< 1%): 모델 대신 기준선 값이 나간다. "예년 같은 기간 기준" 이라고만 표시하고 "예측"이라고 부르지 않는다.
- **거의 확실한 경우**: `p_any ≥ 0.98`이면 확률 숫자 대신 "거의 확실히 탐지됨 — 예상 건수를 보세요"를 크게 보여 주고
  `expected_count`(q10~q90)를 주 정보로 쓴다. 응답의 `display_hint`에 같은 문구가 들어 있다 (대륙·큰 나라·긴 기간은 대부분 여기에 해당).
- 지속 열원: `persistent_source_cells.count > 0`이면 "공장·가스 플레어 같은 지속 열원 의심 격자 포함 → 값이 높게 나올 수 있음"을 함께 보여 준다.
  한국 시군구 aoi는 `/occurrence`의 `suspected_industrial`도 같이 본다.
- **필수 문구**: "위성 탐지 기반 확률이며 산불 발생을 보장하지 않습니다."

```json
// POST /forecast {"region": "SIDO_51", "start": "2026-11-15", "end": "2026-11-21"}   (강원, 내일~7일, 일 단위)
{"p_any": 0.34, "expected_count": 1.29, "expected_count_q10": 0.0, "expected_count_q90": 2.87,
 "baseline": {"p_any": 0.31, "count": 1.2}, "skill": "low", "count_skill": "low", "as_of": "2026-11-14", "display_hint": "",
 "notice": "위성 탐지 기반 확률이며 산불 발생을 보장하지 않습니다."}
// POST /forecast {"region": "ADM1_ARG-1299", "start": "2026-12-01", "end": "2026-12-31"}   (코르도바, 다음 달)
{"p_any": 0.9999, "expected_count": 139.8, "expected_count_q10": 70.8, "expected_count_q90": 1100.0, "skill": "low", "count_skill": "none",
 "display_hint": "거의 확실히 탐지됨 — 예상 건수를 보세요"}
// POST /forecast {"region": "WORLD", "start": "2027-01-01", "end": "2027-03-31"}
{"p_any": 1.0, "expected_count": 3825013, "expected_count_q10": 3305932, "expected_count_q90": 4827281, "skill": "none", "count_skill": "high",
 "display_hint": "거의 확실히 탐지됨 — 예상 건수를 보세요"}
```
