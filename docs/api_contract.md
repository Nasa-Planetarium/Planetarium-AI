# API 계약 (데이터 담당 확정본, 작성 중) — 데이터 `v1.1-m2v-2026-11-14` (모델 `v1-m2v-2026-11-14`)

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
| `GET /fwi` | `fwi_aoi_day.parquet` (+ `fwi_clim_aoi_month.parquet`) | 지역 산불 기상 위험도 배지 (한국·아르헨티나) |
| `GET /export.csv` | 위 결과를 CSV로 | |

### `/summary?aoi|bbox|polygon&start&end&freq=day|month`

- `harmonized_cell_day`에서 영역 안 칸(`aoi`는 칸 × AOI 면적 가중치)과 `date_local` 범위로 `value`, `value_lo`, `value_hi`를 더한다.
- **residual 규칙** (residual = 점으로 나눌 수 없는 값, `harmonized_residual_cell_month.residual_type`으로 구분)
  - **residual(`missing_day_fill` 음수 포함)은 합계 계산에만 쓰고, 별도 값·막대·레이어로 화면에 표시하지 말 것.**
  - `modis_unseen` (2000-11 ~ 2012-01): MODIS가 감지하지 못한 칸의 m2v 추정분 + 배경 예측
  - `missing_day_fill` (2012-02 ~, 완결된 달): 위성 자료가 빠진 날을 달 전체로 환산한 몫 = 칸 value − Σ 기준 점 가중치.
    빠진 날에 감지가 있었던 칸은 **음수**일 수 있다(81,563행). 점 가중치는 센서 변환만(S-NPP 1.0, NOAA-20 ×n20_bridge,
    NOAA-21 ×n21_bridge, MODIS ×m2v 배율)이라 이 몫이 점에 들어가지 않는다.
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
  `layer=all`: `reference/` + `other/` (2012-02 이후 S-NPP·NOAA-20·NOAA-21 중 기준이 아닌 점, 지도 표시용 — 합계에 쓰지 않음).
- **점 지도에는 MODIS 기간(2000-11 ~ 2012-01) 미감지 추정분(residual)이 표시되지 않는다.**
  같은 영역·기간이라도 점 가중치 합은 `/summary` 월 합계보다 작을 수 있다 (차이 = residual). 화면에 안내 문구를 넣는다.
- MODIS 비기준 점(2012-02 이후 Terra·Aqua)은 가중치가 없어 만들지 않았다.

### 구간(weight_lo / weight_hi, value_lo / value_hi)

| 기간 | 점·칸 단위 | AOI 단위 (`aoi_month`) |
|---|---|---|
| 2000-11 ~ 2002-06 (Terra 단독) | **null** — 이 기간 구간은 AOI 단위로만 제공 | 있음 |
| 2002-07 ~ 2012-01 | 있음 (M2V Uncertainty Band) | 있음 (AOI 보정) |
| 2012-02 ~ | value와 같음 | value와 같음 |

- 점·칸 구간을 더한 값은 합계의 정확한 구간이 아니다(오차 상쇄 무시로 넓음). 지역 구간은 `aoi_month.value_lo/hi`를 쓴다.

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

### `/spread?aoi|bbox`

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
| `weather_lag_days` | `weather_lag_days` | 기상 자료 지연(일). NASA POWER 근실시간이 보통 2~3일 늦다 |
| `active_today` | `active_today` | 오늘 이 칸에 감지가 있었나 |

- **표시 3단계**: `p_spread` < 5% / 5–20% / > 20% (색 3단계). 확률 숫자는 툴팁에만.
- **필수 문구** (지도 범례 또는 레이어 설명에 항상):
  > 위성 감지 기반 단기 확산 가능성입니다. 물리 시뮬레이션이 아니며, 오늘 감지 지점에서 반경 약 7.5 km 밖으로 번지는 경우와 새로 발생하는 불은 예측 대상이 아닙니다.
- 검증(2012~2025 연도별 분리, 후보 칸 안): PR-AUC 한국 0.14(KOR만 0.16)·아르헨티나 0.20, 확률 보정 양호.
  후보 칸이 덮는 내일 감지 비율(재현율 상한) 한국 22%, 아르헨티나 42% → 자세한 숫자 `reports/spread_nowcast_metrics.csv`.

```json
// GET /spread?bbox=128.5,35.8,129.6,36.4
{"as_of_date": "2026-11-13", "target_date": "2026-11-14", "weather_lag_days": 2,
 "notice": "위성 감지 기반 단기 확산 가능성 (물리 시뮬레이션 아님). 반경 약 7.5 km 밖·새 발화는 대상 아님.",
 "cells": [{"cell_lat": 36.0375, "cell_lon": 129.3875, "cell_size_deg": 0.025, "p_spread": 0.217, "level": "high", "active_today": true}]}
```

### `/events?aoi&start&end`

- 읽는 파일: `data/processed/fire_events.parquet` (1,276,028행, 49 MB). 기준 점을 시공간으로 이은 사건
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
- 2026년 이후 날짜: 파일에 아직 없다(POWER 일 자료가 2025-12-31까지). 새 날짜를 계산하면 `fwi_pct`는 비우고,
  `fwi_clim_aoi_month.parquet`(AOI × 달 경계값 `q25`·`q75`·`q90`)로 상대 등급만 정한다.
- 표시 문구(권장): "기상으로 본 산불 위험도(캐나다 FWI, NASA POWER 일 자료 근사). 실제 발화·확산을 예측하지 않음."

```json
// GET /fwi?aoi=SGG_47730&date=2025-03-25
{"aoi_id": "SGG_47730", "name": "경상북도 의성군", "date": "2025-03-25",
 "fwi_mean": 22.3, "fwi_class": "very_high", "fwi_pct": 97.4, "fwi_rel_class": "well_above_normal", "badge": "fwi_rel_class"}
```
