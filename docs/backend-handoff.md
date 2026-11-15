# 백엔드 전달 문서 — Planetarium 통합 화재 데이터 (v1)

작성 2026-11-14 · 데이터 **`v1.1-m2v-2026-11-14`** (모델·method_version `v1-m2v-2026-11-14`) · 저장소 `C:\Planetarium-AI`

> v1 기준으로 고쳤어요. API 확정본은 데이터 담당이 만드는 `docs/api_contract.md`이고, 컬럼 이름이 다르면 그쪽을 따라 주세요.
> v1.1 파일은 `data/processed/release_v1.1-m2v-2026-11-14/`에 있어요 (v1은 `release_v1-m2v-2026-11-14/`) (v0는 `release_v0-b1-2026-11-14/`에 그대로 남아 있어요).


## v1.1에서 바뀐 점 (먼저 읽어 주세요)

| 항목 | 내용 |
|---|---|
| 진행 중인 달 | `value` = **오늘까지 실제 합계**(환산 없음), **`value_projected`** = 달 전체 추정, **`as_of_date`** = 마지막 관측일. `harmonized_cell_month`·`aoi_month`·`aoi_month_compare`·`harmonized_cell_day`(1일 행 `value_projected_month`) 모두 같은 규칙. 판정은 `uncertain`. 예) KR 2026-11: value 44.3, value_projected 102.3, as_of_date 2026-11-13 |
| 점 가중치 | 센서 변환만: S-NPP 1.0, NOAA-20 ×n20_bridge, NOAA-21 ×n21_bridge, MODIS ×m2v 배율. 자료가 빠진 날 환산분은 점에 넣지 않음 |
| `residual_type` | `harmonized_residual_cell_month`(2,288만 행): `modis_unseen`(MODIS 기간 미감지 추정분) / `missing_day_fill`(실측 기간 자료 빠진 날 환산분, 26개 달, 음수 가능) |
| `harmonized_cell_day` (신규) | 칸 × 현지 날짜, 2억 1,277만 행, 965 MB. `/summary`용. `residual_month`(칸·월 residual, 1일 행에만)는 **월 단위로만** 더하기 — `docs/api_contract.md` |
| 검증 | 점 가중치 + residual = 칸 value, AOI 면적 가중 합 = `aoi_month.value`, cell_day 월 합 = `aoi_month.value` (전부 상대 오차 < 1e-6) |

> **residual(`missing_day_fill` 음수 포함)은 합계 계산에만 쓰고, 별도 값·막대·레이어로 화면에 표시하지 말 것.**

---

## 0. 한 줄 요약

MODIS(2000~)와 VIIRS(2012~) 화재 감지를 **VIIRS(S-NPP) 기준 감지량**이라는 하나의 잣대로 통합한 데이터예요. 백엔드는 아래 Parquet/GeoJSON을 읽어 지도(점·격자), 연도×월 캘린더, 예년 대비 비교, CSV 다운로드를 API로 제공하면 돼요.

---

## 1. 백엔드가 읽을 파일 (`data/processed/`)

| 파일 | 한 행 | 크기 | 앱 화면 |
|---|---|---|---|
| `harmonized_cell_month.parquet` | 0.1° 칸 × 월 | **571 MB, 51,562,642행** | 지도 격자(히트맵) |
| `harmonized_cell_day.parquet` | 0.1° 칸 × 현지 날짜 | 965 MB, 212,774,685행 | `/summary` 일·월 합계 |
| `harmonized_residual_cell_month.parquet` | 칸 × 월 (residual_type) | 125 MB, 22,878,611행 | 점으로 나눌 수 없는 값 |
| `aoi_month.parquet` | 지역 × 월 | 31 MB, 1,580,963행 (지역 5,051 × 313개월) | 연도 × 월 캘린더 |
| `aoi_month_compare.parquet` | 지역 × 월 | 31 MB, 1,580,963행 | 예년 대비 비교 |
| `aoi_list.parquet` | 지역 1개 | 5,051행 | 지역 선택 목록 |
| `aoi_boundaries.geojson` | 지역 경계 | 13 MB | 지도 경계선 |
| `satellite_day_status.parquet` | 위성 × 날짜 | — | 자료 상태 안내 |
| `harmonized_hotspots/reference/year=YYYY/month=MM/*.parquet` | 기준 감지 1건 (점) | **4.5 GB, 325,055,860행** (아래 1-1) | 지도 핫스팟 점(기본), 합계 |
| `harmonized_hotspots/other/year=YYYY/month=MM/*.parquet` | 기준이 아닌 VIIRS 감지 (2012-02~) | 3.0 GB, 228,772,249행 | 지도 '모든 감지' 레이어 |
| `harmonized_cell_day.parquet` | 0.1° 칸 × 현지 날짜 | 908 MB, 209,765,318행 | 날짜 범위 합계 (`/summary`) |
| `harmonized_residual_cell_month.parquet` | 기준 점이 없는 칸 × 월 | 109 MB, 19,948,578행 | 합계 보정용 (점으로 안 보이는 추정분) |
| `../../models/m2v_meta.json` | — | — | "방법 설명" 페이지 |
| `spread_nowcast_latest.parquet` (2026-11-15) | 0.025° 후보 칸 (기준일 1일) | 0.2 MB, 10,189행 (11-13 기준) | 확산 가능성 레이어 `/spread` |
| `fire_events.parquet` (2026-11-15) | 화재 사건 1건 (한국·아르헨티나) | 49 MB, 1,276,028행 | 사건 목록·지도 `/events` |
| `aoi_month_fire_regime.parquet` (2026-11-15) | AOI × 월 (한국 KR·시도·시군구, 아르헨티나) | 0.3 MB, 92,335행 | 불 유형 배지 `/regime` |
| `fwi_aoi_day.parquet` (2026-11-15) | AOI × 날짜 (한국 KR·시도·시군구, 아르헨티나), 2012~2025 | 27 MB, 1,508,630행 (AOI 295개) | 산불 기상 위험도 배지 `/fwi` |
| `fwi_clim_aoi_month.parquet` (2026-11-15) | AOI × 달 경계값 (q25·q50·q75·q90) | 0.04 MB, 3,540행 | 새 날짜의 상대 등급 판정 |

> 배경 예측(감지가 없는 칸의 작은 m2v 예측값, 0.5 미만)은 `harmonized_cell_month`에 넣지 않고 **`data/interim/harmonized_background/<연도>.parquet`에 따로** 있어요(2000~2012, 9,016만 행). 지역 합계(`aoi_month`)에는 이미 더해져 있으니 **캘린더·비교는 `aoi_month*`만 읽으면 돼요.** 격자 지도에서 이 작은 값까지 그릴 필요는 없어요(2005년 기준 전체의 약 3%).

### 1-1. `harmonized_hotspots` (통합 핫스팟 점) — 생성·검증 완료 (2026-11-14)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| latitude, longitude | float | 픽셀 중심 |
| acq_datetime_utc | datetime(UTC) | 관측 시각 |
| date_local | date | 현지 태양시 날짜 (날짜 범위 필터는 이 값 기준) |
| satellite | str | N(S-NPP) / N20 / N21 / Terra / Aqua |
| instrument | str | VIIRS / MODIS |
| frp | float | 화재 복사 에너지(MW) |
| confidence | str | l / n / h |
| daynight | str | D / N |
| cell_id | int | 0.1° 칸 ID |
| adm0_a3, adm1_code | str | 국가(ISO3 계열 코드), 1급 행정구역 코드 (칸 겹침 최대) |
| **harmonized_weight** | float | 이 감지가 VIIRS(S-NPP) 기준으로 몇 건에 해당하는지 |
| weight_lo, weight_hi | float | 가중치 범위(10–90%). **Terra 단독 기간(2000-11~2002-06)은 null** |
| weight_source | str | 기준 점은 `value_source`와 같은 값, 그 외 `snpp_unit` / `n20_bridge` / `n21_bridge` / `none` |
| **is_reference** | bool | true = 그 기간 기준 위성(합계에 사용), false = 지도 '모든 감지' 레이어용 |
| source | str | SP(정식) / NRT(빠른 처리본) |
| method_version | str | `v1-m2v-2026-11-14` |

- 실제 규모: 기준 점 **3억 2,506만 행, 4.5 GB** (`reference/`), 기준이 아닌 VIIRS 점 2억 2,877만 행, 3.0 GB (`other/`, 2012-02 이후만. MODIS 비기준 점은 만들지 않음).
- 검증: 점 가중치 합 + residual = 칸 value (최대 차이 0.00015), 지역 합 = `aoi_month.value` (상대 오차 1e-6 미만). 한국 2005-03 = 380.4601, 2025-03 = 2,764 세 방식 모두 일치.
- residual은 MODIS 기간에만 있어요(칸 값의 약 9%). 2013년 이후는 0.
- `harmonized_cell_day`: `cell_id, date_local, value, value_lo, value_hi, n_points, value_source, residual_month`. **residual_month는 그 달 1일 행에 칸·월 합계로만 저장**돼 있어요(1일의 값이 아님). 일별 그래프에 그리지 말고 월 합계에만 더하세요. 자세한 규칙은 `docs/api_contract.md`.
- 기준 점 가중치 합 + `harmonized_residual_cell_month`(점으로 나눌 수 없는 값) = 칸 `value`.

### 1-2. `harmonized_cell_month` (칸 × 월)

`cell_id, cell_lat, cell_lon, year, month, value, value_lo, value_hi, value_projected, raw_modis, raw_viirs, value_source, data_source, status, method_version`

- `value` = 그 칸·달의 VIIRS 기준 감지량, `raw_modis`/`raw_viirs` = 원래 감지 수(참고)
- **0이 아닌 칸·월만 저장**돼 있어요. 행이 없으면 0이에요 (2000~2012는 0.5 미만 배경 예측도 행이 없어요 — 위 안내 참고).
- 칸 단위 `value_lo/hi`: MODIS 기간은 M2V Uncertainty Band(칸별 10·90 분위수), 실측 기간은 value와 같아요. **Terra 단독 기간은 null.**

### 1-3. `aoi_month` (지역 × 월)

`aoi_id, year, month, value, value_lo, value_hi, value_projected, raw_modis, raw_viirs, value_source, data_source, status, as_of_date, n_cells`

- 칸 값을 지역에 **겹침 면적 비율로 나눠서** 더했어요. 그래서 252개 시군구가 모두 값을 가져요. `KR` 합계 = 시도 합 = 시군구 합.
- `value_lo/hi`: MODIS 기간(2000-11~2012-01)은 지역 단위로 보정한 구간(아래 표), 실측 기간은 value와 같아요.
- `status = missing`이면 `value`, `value_lo`, `value_hi`가 null이에요 (v1에서는 2001-06 한 달: Terra 위성 정지).

### 1-4. `aoi_month_compare` (예년 대비)

`aoi_id, year, month, value, baseline_median, baseline_p25, baseline_p75, percentile, ratio_to_median, n_baseline_years, verdict, status, low_sample, as_of_date, value_projected`

### 1-5. `aoi_list` / `aoi_boundaries.geojson`

`aoi_id, name, level, parent_aoi, n_cells, hidden`

- level: `world`(1), `continent`(8), `country`(248), `adm1`(4,524), `kr`(1), `sido`(17), `sgg`(252)
- aoi_id 예: `WORLD`, `CONT_Asia`, `ADM0_ARG`, `ADM1_ARG-1271`, `KR`, `SIDO_11`, `SGG_11110`
- 한국은 `ADM0_KOR`가 아니라 **`KR`**을 써요 (시도·시군구가 그 아래에 있어요).
- **`hidden = true`인 지역(63개)**: 공해(`CONT_Seven_seas__open_ocean`)와 그 아래 작은 섬 국가·행정구역이에요. 칸이 적어 검증이 불안정해서(편향 1.41) **지역 목록·검색에서 빼 주세요.** 데이터 행은 그대로 있고, 비교 판정은 항상 `uncertain`이에요. 직접 `aoi_id`로 요청이 오면 값은 주되 "참고용" 표시를 권해요.

---

## 2. API (제안)

| API | 요청 파라미터 | 응답 |
|---|---|---|
| `GET /aoi` | `level`, `parent_aoi`, `q`(이름 검색) | 지역 목록 (`hidden=false`만, `include_geometry=true`면 경계) |
| `GET /hotspots` | `bbox=서,남,동,북` 또는 `aoi_id`, `start`, `end`(YYYY-MM-DD), `layer=reference\|all`, `limit` | 점 목록 |
| `GET /summary` | `aoi_id` 또는 `bbox` 또는 `polygon`(GeoJSON), `start`, `end`, `freq=day\|month` | 기간별 value, lo, hi, status, value_source |
| `GET /calendar` | `aoi_id`, `start_year`, `end_year` | 연도×월 value + verdict, percentile, ratio_to_median, status |
| `GET /compare` | `aoi_id`, `year`, `month` | 그 달의 비교 상세 |
| `GET /status` | `start`, `end` | 위성별 자료 상태, 최신 갱신 시각 |
| `GET /export.csv` | `type=summary\|calendar\|hotspots` + 해당 API 파라미터 | CSV 다운로드 |
| `GET /method` | — | 모델 이름·계수·성능 요약 (`m2v_meta.json`) |
| `GET /spread` | `aoi_id` 또는 `bbox` | 내일 확산 가능성 칸 목록 (확률·3단계·기준일·예측일·`weather_lag_days`·필수 문구) |
| `GET /events` | `aoi_id`, `start`, `end`, `limit`, `min_det` | 화재 사건 (영역·기간·감지 수·위성 확인 연소 면적(하한)) |
| `GET /regime` | `aoi_id`, `start_year`, `end_year` | 월별 불 유형 배지 (`large_fire`/`many_small`/`mixed`/`none`) |
| `GET /fwi` | `aoi_id`, `date` | 산불 기상 위험도: 절대 등급 `fwi_class` + 상대 등급 `fwi_rel_class` (앱 기본 배지는 상대 등급) |

### 응답 예시 (v1 실제 값)

`GET /calendar?aoi_id=KR&start_year=2025&end_year=2025`
```json
{
  "aoi_id": "KR",
  "method_version": "v1-m2v-2026-11-14",
  "unit": "VIIRS(S-NPP) 기준 화재 감지량",
  "months": [
    {"year": 2025, "month": 3, "value": 2764, "value_lo": 2764, "value_hi": 2764,
     "value_source": "snpp_observed", "status": "ok",
     "verdict": "above", "percentile": 100, "ratio_to_median": 10.9}
  ]
}
```

`GET /calendar?aoi_id=KR&start_year=2005&end_year=2005` 중 3월
```json
{"year": 2005, "month": 3, "value": 380.5, "value_lo": 203.7, "value_hi": 533.4,
 "value_source": "modis_m2v", "status": "ok"}
```

---

## 3. 코드값과 규칙

### value_source (값의 출처)

| 값 | 기간 | 화면 표시 예 |
|---|---|---|
| `modis_m2v_terra` | 2000-11 ~ 2002-06 | 추정 (Terra) |
| `modis_m2v` | 2002-07 ~ **2012-01** | 추정 (MODIS → VIIRS) |
| `snpp_observed` | 2012-02 ~ 2026-10 (S-NPP 마지막 감지 2026-11-02) | 실측 |
| `n20_bridge` | 2026-11 ~ (S-NPP 11-01~02 + NOAA-20 11-03~) | 실측 (NOAA-20 환산) |
| `n21_bridge` | 핫스팟 점에서 NOAA-20 결측일 대체 | 대체 위성 (NOAA-21 환산) |

**2012-01은 `modis_m2v` 값을 써요.** S-NPP가 2012-01-20부터 12일치만 있어 월 환산 오차가 크고, 모델 학습 기간도 2012-02부터라 MODIS 쪽 추정이 더 일관돼요 (`m2v_meta.json` → `harmonization."2012-01"`).

### 구간(value_lo / value_hi) 정책

| 기간 | 칸(`harmonized_cell_month`)·점(`harmonized_hotspots`) | 지역(`aoi_month`) |
|---|---|---|
| 2000-11 ~ 2002-06 (Terra 단독) | **null** (칸 단위 분위수 모델 없음) | 있음 (지역 단위 보정) — **이 기간 구간은 지역 단위로만 제공** |
| 2002-07 ~ 2012-01 | M2V Uncertainty Band (칸별 10·90%) | 있음 (지역 단위 보정) |
| 2012-02 ~ | value와 같음 (실측) | value와 같음 |

지역 구간은 칸 구간을 단순히 더하지 않고, 과거 검증(연도를 하나씩 빼고 학습한 예측)에서 '실제/예측' 비율의 10·90% 분위수를 지역 유형·크기별로 구해 지역 합계에 곱했어요.

**지역 구간 적중률** (목표 80%, 검증 연도를 뺀 나머지로 구간을 만든 뒤 그 해에 적용):

| 지역 유형 | 지역·월 수 | 적중률 | 구간 폭 중앙값 (폭 ÷ 값) |
|---|---|---|---|
| 국가 (KR 포함) | 39,930 | 83.4% | 1.45 |
| 1급 행정구역 | 738,884 | 86.5% | 2.12 |
| 시·도 | 2,805 | 87.3% | 3.18 |
| 시·군·구 | 41,580 | 90.0% | 3.20 |
| 대륙 | 1,320 | 66.6% | 0.15 |
| 전 세계 | 165 | 78.8% | 0.07 |

- 예측 10건 이상인 지역·월은 국가·1급 모두 79.9~80.0%로 목표에 맞아요.
- 작은 지역(예측 10건 미만)은 실제 0인 달이 많아 하한이 0이 되기 쉬워 적중률이 90%로 높고 폭이 넓어요.
- 대륙이 낮은 건 남극·공해처럼 감지가 거의 없는 대륙 때문이에요(1만 건 이상 대륙·월은 79.2%).

### status (자료 상태)

| 값 | 의미 | 화면 처리 |
|---|---|---|
| `ok` | 정상 | 일반 |
| `nrt` | 빠른 처리본, 이후 바뀔 수 있음 | 점선/연한 색 + 안내 |
| `partial` | 진행 중인 달 | "집계 중" |
| `missing` | 위성 자료 부족 (v1: 2001-06) | 회색. **0으로 칠하지 않기** |

`low_sample`은 status가 아니라 `aoi_month_compare`의 **별도 컬럼**(true/false)이에요. 예년 같은 달 감지가 거의 0인 지역이라 백분위가 불안정하다는 뜻이고, 이때 verdict는 값이 3 이상이면 `above`, 아니면 `uncertain`이에요.

### verdict (예년 대비)

| 값 | 기준 | 표시 |
|---|---|---|
| `above` | 백분위 ≥ 80 | 예년보다 많음 |
| `similar` | 20 < 백분위 < 80 | 비슷 |
| `below` | 백분위 ≤ 20 | 예년보다 적음 |
| `uncertain` | 비교 연도 < 10년, 결측·진행 중, `hidden` 지역 | 판단 보류 |

- 기준 분포 = 2001~2025년 같은 달(해당 연도 제외, status=ok인 달만)

### 계수·버전

- `n20_bridge` = 1.013, `n21_bridge` = 1.053 (전 세계 정식 자료 기준, 실제로는 지역·월별 값과 대체 규칙 적용)
- `method_version` = `v1-m2v-2026-11-14` (응답에 항상 포함 추천)
- 모델 이름: `m2v_xgb`(M2V Harmonizer), `m2v_q10_q90`(M2V Uncertainty Band) — `m2v_meta.json`의 `model_names`

### 공간 일반화 한계 (2026-11-14 검증, v1.1 값은 그대로)

- **실제 사용 조건 = leave-year-out**: 학습에 들어 있는 지역의 새 연도를 예측하면 월 합계 편향(Σ예측/Σ실제)이 대륙·한국·아르헨티나 모두 **0.98~0.99**.
  v1.1은 모든 국가가 학습에 들어 있으므로 배포값(`modis_m2v`)은 이 조건이에요.
- **처음 보는 대륙(leave-continent-out)에서는 수준이 틀려요**: 대륙 월 합계 **0.27~0.76배**, 한국(아시아 없이 학습) **3.0배**, 아르헨티나(남미 없이 학습) **0.65배**.
  계절 모양(상관 0.75~0.98)은 따라가지만 합계와 구간은 믿을 수 없어요.
- **안내**: 학습에 없는 새 지역(그 나라·대륙의 S-NPP 학습 자료가 없는 곳)에는 이 모델을 그대로 쓰지 말고 `ratio_country`(국가 → 하위지역 → 위도띠 × 월 비율)를 쓰세요.
  같은 검증에서 편향 0.85~1.34.
- 자세히: `reports/m2v_leave_continent_out.csv`, `reports/figures/m2v_leave_continent_out.png`, `models/m2v_meta.json`의 `spatial_generalization`.

---

## 3-1. 확산 가능성 · 화재 사건 · 불 유형 (2026-11-15 추가, 한국·아르헨티나)

API 형식은 `docs/api_contract.md`의 "확산 가능성·화재 사건·불 유형" 절을 따라 주세요.

### 확산 가능성 레이어 (`spread_nowcast_latest`)

- 오늘 감지 지점에서 반경 약 7.5 km 안의 2.5 km 칸마다 **내일 위성 감지가 생길 확률**. 모델 `spread_nowcast_xgb`(XGBoost),
  감지 특성 + NASA POWER 기상. 기준일·예측일·기상 지연(`weather_lag_days`)을 응답에 꼭 넣어 주세요.
- 표시 3단계: **< 5% / 5–20% / > 20%**.
- **필수 문구**: "위성 감지 기반 단기 확산 가능성입니다. 물리 시뮬레이션이 아니며, 오늘 감지 지점에서 반경 약 7.5 km 밖으로 번지는 경우와 새로 발생하는 불은 예측 대상이 아닙니다."
- 갱신: `uv run python -m planetarium.fetch_nrt --fetch --days 2 --append --with-spread` (정기 실행은 아직 설정하지 않음).

### 화재 사건 (`fire_events`)과 불 유형 (`aoi_month_fire_regime`)

- 사건 = 기준 감지 점을 시공간으로 이은 묶음 (750 m VIIRS / 2 km MODIS, 하루 차이까지). 시군구·시도·국가는 **감지 점 위치** 기준 가중 다수결.
- `burned_area_km2`는 **"위성 확인 연소 면적(하한)"**으로 표시해 주세요 (MCD64A1, 2012~2025; 의성 2025-03은 공식 피해 면적의 약 64%).
- **fire_regime은 감지 점 위치로 시군구를 정하고, aoi_month는 칸 면적 비율로 나눠서, 작은 값이 있는 시군구라도 fire_regime이 none일 수 있음. 캘린더 값은 aoi_month, 불 유형 배지는 fire_regime 기준.**

### 검증 — 화재 사건 vs 산림청 산불통계 (2022~2025)

산림청 산불 하나가 같은 시군구(사건의 `sgg_all`)에서 기간이 겹치는(앞뒤 하루) 사건과 만나면 "탐지"로 셌어요.

| 피해 면적 | 산불 수 | 탐지 | 탐지율 |
|---|---|---|---|
| 100 ha 이상 (큰 산불) | 25 | 23 | **92%** |
| 10~100 ha | 53 | 36 | 68% |
| 1~10 ha | 266 | 66 | 25% |
| 1 ha 미만 | 1,692 | 138 | 8% |

놓친 큰 산불 2건(자료 상태는 모든 위성 정상):
- **영주 평은면 2023-04-03~04 (244.5 ha)**: S-NPP·NOAA-20·Aqua·Terra 모두 35 km 안에 감지 없음 → 위성 관측 시각 사이에 타고 꺼졌거나 연기·구름에 가려진 것으로 추정(미확인).
- **강릉 2023-04-11 (120.7 ha)**: NOAA-20 11개·Terra 6개 감지는 있었지만 기준 위성 S-NPP는 감지 없음(가장 가까운 21 km). 사건은 기준 위성 점만으로 묶기 때문에 빠짐.
- 이전에 놓쳤던 봉화(130 ha)·서산 운산(158 ha)은 0.1° 칸 대표 시군구 대신 점 위치로 시군구를 정한 뒤 잡혔어요 (84% → 92%).
- 자료: `reports/fire_events_kfs_detection.csv`, `reports/fire_events_kfs_large.csv`, `reports/fire_events_kfs_missed_diagnosis.csv`.

### 산불 기상 위험도 배지 (`fwi_aoi_day`, 2026-11-15)

- 캐나다 FWI를 NASA POWER 일 자료로 근사 계산(근사 내용 `data/processed/fwi_meta.json`). 확산 예측 모델에는 넣지 않았어요(넣어도 성능이 오르지 않음).
- **앱 기본 배지는 상대 등급 `fwi_rel_class`** — 같은 AOI·같은 달의 2012~2025 분포에서의 백분위:
  평년보다 낮음(< 25) / 평년(25~75) / 높음(75~90) / 매우 높음(≥ 90). 절대 등급 `fwi_class`는 상세 화면에 함께.
- **아르헨티나는 캐나다 절대 등급이 잘 맞지 않아요** (FWI 중앙값 20.3으로 늘 높게 나오고, `extreme`에서 다음 날 새 감지가 오히려 줄어듦).
- 2026년 이후 날짜는 아직 파일에 없어요(POWER 일 자료가 2025-12-31까지). 새로 계산하면 `fwi_clim_aoi_month.parquet` 경계값(q25·q75·q90)으로 상대 등급만 정해요.

검증 — 등급별 "다음 날 새 감지" 배율 (0.5° 칸, 그날 감지 없던 칸, 지역 평균 = 1.0)

| 상대 등급 | 한국 | 아르헨티나 |
|---|---|---|
| 평년보다 낮음 | 0.56× | 0.63× |
| 평년 | 1.03× | 1.05× |
| 높음 | 1.32× | 1.29× |
| 매우 높음 | 1.53× | 1.32× |
| AUC (백분위) | 0.60 | 0.58 |
| (참고) AUC 절대 FWI | 0.63 | 0.55 |

- 한국은 절대 FWI의 AUC가 조금 더 높아요(봄 건조기 같은 계절 차이가 절대값에 담기고, 상대 등급은 달별로 정규화해 그 차이를 뺌). 그래서 상세 화면에 절대 등급도 같이 보여 주세요.
- 자료: `reports/fwi_validation.csv`(절대), `reports/fwi_validation_relative.csv`(상대).

---

## 4. 꼭 지켜야 할 것

1. **0과 결측 구분**: 칸·월 행이 없으면 0. 그 달이 결측인지는 `aoi_month.status`로 판단.
2. **합계는 `is_reference = true`인 점만** (+ residual): `harmonized_weight` 합 = value. `layer=all`은 지도 표시용.
3. **지역 합계는 `aoi_month`를 그대로** 쓰기: 칸에서 다시 더하면 배경 예측(interim)과 면적 가중이 빠져 숫자가 달라져요.
4. **날짜는 `date_local` 기준**으로 필터. UTC 시각은 `acq_datetime_utc`.
5. **표기**: "산불 건수"가 아니라 **"위성 화재 감지량 (VIIRS 기준)"**. 2000–2012-01은 "추정값"이라고 표시하고 범위를 함께 보여 주기.
6. **FIRMS 키**: `FIRMS_MAP_KEY`는 서버 `.env`에만. 프론트엔드 코드·GitHub·채팅에 절대 넣지 않기.
7. **출처 표시**: 앱 하단 또는 About 페이지에 FIRMS 인정 문구와 경계 데이터 출처(CC BY) 표기 → `docs/references.md`.
8. **성능**: `harmonized_hotspots`는 커요(7.5 GB). 날짜 범위 합계는 `harmonized_cell_day`로 계산하고, 점 파일은 지도를 확대했을 때 연·월 폴더 단위로 필요한 범위만 읽기(Polars `scan_parquet` + 필터). 점 응답에는 `limit`·`bbox`를 꼭 쓰기.

---

## 5. 파일 받는 법

- 데이터 파일은 저장소가 아니라 GitHub Release에 있어요: [data-v1.1-m2v-2026-11-14](https://github.com/Nasa-Planetarium/Planetarium-AI/releases/tag/data-v1.1-m2v-2026-11-14)(칸·AOI 표), [data-v1.1-hotspots-2026-11-14](https://github.com/Nasa-Planetarium/Planetarium-AI/releases/tag/data-v1.1-hotspots-2026-11-14)(통합 핫스팟 점, 연도별 tar). 사용법은 README의 '데이터 받기'.
- v0 → v1 변경점
  - 숫자: 2000~2012-01 추정값이 m2v로 바뀜 (2000~2011 합계: 한국 −24.5%, 아르헨티나 −22.2%, 전 세계 −3.5%). 그림 `reports/figures/v0_vs_v1_korea_annual.png`
  - `value_source` 이름: `modis_harmonized` → `modis_m2v`, `modis_harmonized_terra_only` → `modis_m2v_terra`, `noaa20_bridged` → `n20_bridge`
  - `aoi_list`: `hidden` 컬럼 추가, 시군구 252개 모두 포함(면적 가중 배분)
  - `aoi_month.value_lo/hi`: 지역 단위 보정 구간

문의: 데이터·모델 담당 (팀장)
