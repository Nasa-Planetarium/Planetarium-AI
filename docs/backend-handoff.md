# 백엔드 전달 문서 — Planetarium 통합 화재 데이터 (초안)

작성 2026-11-14 · 데이터 버전 `v1-m2v-2026-11-14` · 저장소 `C:\Planetarium-AI`

> 이 문서는 **계획 기준 초안**이에요. 데이터 담당(Claude Code)이 만드는 `docs/api_contract.md`가 확정본이고, 컬럼 이름이 조금 바뀌면 그쪽을 따라 주세요. 지금 있는 v0 파일(`release_v0-b1-2026-11-14/`)은 구조가 같아서 연결 작업을 바로 시작해도 돼요. v1이 나오면 파일만 교체하면 돼요.

---

## 0. 한 줄 요약

MODIS(2000~)와 VIIRS(2012~) 화재 감지를 **VIIRS(S-NPP) 기준 감지량**이라는 하나의 잣대로 통합한 데이터예요. 백엔드는 아래 Parquet/GeoJSON을 읽어 지도(점·격자), 연도×월 캘린더, 예년 대비 비교, CSV 다운로드를 API로 제공하면 돼요.

---

## 1. 백엔드가 읽을 파일 (`data/processed/`)

| 파일 | 한 행 | 앱 화면 |
|---|---|---|
| `harmonized_hotspots/year=YYYY/month=MM/*.parquet` | 감지 1건 (점) | 지도 핫스팟 점, 임의 날짜 범위 합계 |
| `harmonized_cell_month.parquet` | 0.1° 칸 × 월 | 지도 격자(히트맵) |
| `aoi_month.parquet` | 지역 × 월 | 연도 × 월 캘린더 |
| `aoi_month_compare.parquet` | 지역 × 월 | 예년 대비 비교 |
| `aoi_list.parquet` | 지역 1개 | 지역 선택 목록 |
| `aoi_boundaries.geojson` | 지역 경계 | 지도 경계선 |
| `satellite_day_status.parquet` | 위성 × 날짜 | 자료 상태 안내 |
| `../../models/m2v_meta.json` | — | "방법 설명" 페이지 |

> `harmonized_hotspots`는 다음 작업에서 만들어져요(아직 없음). 그 전에는 `harmonized_cell_month`, `aoi_month*`로 캘린더·비교·격자부터 연결하면 돼요.

### 1-1. `harmonized_hotspots` (통합 핫스팟 점)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| latitude, longitude | float | 픽셀 중심 |
| acq_datetime_utc | datetime(UTC) | 관측 시각 |
| date_local | date | 현지 태양시 날짜 (날짜 범위 필터는 이 값 기준) |
| satellite | str | N(S-NPP) / N20 / N21 / Terra / Aqua |
| instrument | str | VIIRS / MODIS |
| frp | float | 화재 복사 에너지(MW), 불의 세기 |
| confidence | str | l / n / h |
| daynight | str | D / N |
| cell_id | int | 0.1° 칸 ID |
| country, adm1 | str | 국가, 1급 행정구역 |
| **harmonized_weight** | float | 이 감지가 VIIRS(S-NPP) 기준으로 몇 건에 해당하는지 |
| weight_lo, weight_hi | float | 가중치 범위(10–90%) |
| weight_source | str | 아래 `value_source`와 같은 값 |
| **is_reference** | bool | true = 그 기간 기준 위성(합계에 사용), false = 지도 표시용 |
| source | str | SP(정식) / NRT(빠른 처리본) |
| method_version | str | 예: v1-m2v-2026-11-14 |

### 1-2. `harmonized_cell_month` (칸 × 월)

`cell_id, cell_lat, cell_lon, year, month, value, value_lo, value_hi, raw_modis, raw_viirs, value_source, data_source, status, method_version`

- `value` = 그 칸·달의 VIIRS 기준 감지량, `raw_modis`/`raw_viirs` = 원래 감지 수(참고)
- **0이 아닌 칸·월만 저장**돼 있어요. 행이 없으면 0이에요.

### 1-3. `aoi_month` (지역 × 월)

`aoi_id, year, month, value, value_lo, value_hi, raw_modis, raw_viirs, value_source, data_source, status, n_cells`

### 1-4. `aoi_month_compare` (예년 대비)

`aoi_id, year, month, value, baseline_median, baseline_p25, baseline_p75, percentile, ratio_to_median, n_baseline_years, verdict`

### 1-5. `aoi_list` / `aoi_boundaries.geojson`

`aoi_id, name_ko, name_en, level, parent_id, n_cells, hidden`

- level: `world`, `continent`(8), `country`(248), `adm1`(4,524), `kr_sido`(17), `kr_sgg`(252)
- aoi_id 예: `KR`, `SIDO_11`, `SGG_11110`
- `hidden = true`(예: 공해 Seven seas)는 목록에서 숨기기

---

## 2. API (제안)

| API | 요청 파라미터 | 응답 |
|---|---|---|
| `GET /aoi` | `level`, `parent_id`, `q`(이름 검색) | 지역 목록 (+ `include_geometry=true`면 경계) |
| `GET /hotspots` | `bbox=서,남,동,북` 또는 `aoi_id`, `start`, `end`(YYYY-MM-DD), `layer=reference\|all`, `limit` | 점 목록 |
| `GET /summary` | `aoi_id` 또는 `bbox` 또는 `polygon`(GeoJSON), `start`, `end`, `freq=day\|month` | 기간별 value, lo, hi, status, value_source |
| `GET /calendar` | `aoi_id`, `start_year`, `end_year` | 연도×월 value + verdict, percentile, ratio_to_median, status |
| `GET /compare` | `aoi_id`, `year`, `month` | 그 달의 비교 상세 |
| `GET /status` | `start`, `end` | 위성별 자료 상태, 최신 갱신 시각 |
| `GET /export.csv` | `type=summary\|calendar\|hotspots` + 해당 API 파라미터 | CSV 다운로드 |
| `GET /method` | — | 모델 이름·계수·성능 요약 (`m2v_meta.json`) |

### 응답 예시

`GET /calendar?aoi_id=KR&start_year=2024&end_year=2025`
```json
{
  "aoi_id": "KR",
  "method_version": "v1-m2v-2026-11-14",
  "unit": "VIIRS(S-NPP) 기준 화재 감지량",
  "months": [
    {"year": 2025, "month": 3, "value": 2764, "value_lo": 2764, "value_hi": 2764,
     "value_source": "snpp_observed", "status": "ok",
     "verdict": "above", "percentile": 100, "ratio_to_median": 9.8}
  ]
}
```

`GET /summary?aoi_id=KR&start=2005-03-01&end=2005-03-31&freq=day`
```json
{
  "aoi_id": "KR", "freq": "day",
  "series": [
    {"date": "2005-03-01", "value": 12.4, "value_lo": 9.1, "value_hi": 16.0,
     "value_source": "modis_m2v", "status": "ok"}
  ]
}
```
(숫자는 형식 예시예요.)

---

## 3. 코드값과 규칙

### value_source (값의 출처)

| 값 | 기간 | 화면 표시 예 |
|---|---|---|
| `modis_m2v_terra` | 2000-11 ~ 2002-06 | 추정 (Terra) |
| `modis_m2v` | 2002-07 ~ 2011-12 | 추정 (MODIS → VIIRS) |
| `snpp_observed` | 2012-01-20 ~ 2026-11-02 | 실측 |
| `n20_bridge` | 2026-11-03 ~ | 실측 (NOAA-20 환산 × 1.013) |
| `n21_bridge` | NOAA-20 결측일 대체 | 대체 위성 (NOAA-21 × 1.053) |

### status (자료 상태)

| 값 | 의미 | 화면 처리 |
|---|---|---|
| `ok` | 정상 | 일반 |
| `nrt` | 빠른 처리본, 이후 바뀔 수 있음 | 점선/연한 색 + 안내 |
| `partial` | 진행 중인 달 | "집계 중" |
| `missing` | 위성 자료 부족 | 회색. **0으로 칠하지 않기** |
| `low_sample` | 예년 기록이 너무 적음 | 판정 생략 |

### verdict (예년 대비)

| 값 | 기준 | 표시 |
|---|---|---|
| `above` | 백분위 ≥ 80 | 예년보다 많음 |
| `similar` | 20 < 백분위 < 80 | 비슷 |
| `below` | 백분위 ≤ 20 | 예년보다 적음 |
| `uncertain` | 비교 연도 < 10년, 결측·진행 중 | 판단 보류 |

- 기준 분포 = 2001~2025년 같은 달(해당 연도 제외, status=ok인 달만)

### 계수·버전

- `n20_bridge` = 1.013, `n21_bridge` = 1.053
- `method_version` = `v1-m2v-2026-11-14` (응답에 항상 포함 추천)

---

## 4. 꼭 지켜야 할 것

1. **0과 결측 구분**: 칸·월 행이 없으면 0. 그 달이 결측인지는 `aoi_month.status`로 판단.
2. **합계는 `is_reference = true`인 점만**: `harmonized_weight` 합 = value, `weight_lo/hi` 합 = 범위. `layer=all`은 지도 표시용.
3. **날짜는 `date_local` 기준**으로 필터. UTC 시각은 `acq_datetime_utc`.
4. **표기**: "산불 건수"가 아니라 **"위성 화재 감지량 (VIIRS 기준)"**. 2000–2011년은 "추정값"이라고 표시하고 범위를 함께 보여 주기.
5. **FIRMS 키**: `FIRMS_MAP_KEY`는 서버 `.env`에만. 프론트엔드 코드·GitHub·채팅에 절대 넣지 않기.
6. **출처 표시**: 앱 하단 또는 About 페이지에 FIRMS 인정 문구와 경계 데이터 출처(CC BY) 표기 → `docs/references.md`.
7. **성능**: `harmonized_hotspots`는 크기가 커요. 연·월 폴더 단위로 필요한 범위만 읽고(Polars `scan_parquet` + 필터), 점 응답에는 `limit`·`bbox`를 꼭 쓰기.

---

## 5. 파일 받는 법

- 데이터 파일은 GitHub에 없어요(용량). `data/processed/release_v1-m2v-2026-11-14/`를 공유 드라이브로 전달할 예정이에요.
- 그 전에는 `release_v0-b1-2026-11-14/`로 연결 작업을 시작하고, v1 도착 시 교체.
- v0 → v1 변경점: 숫자(2000–2011 추정값), `value_source` 이름(`modis_harmonized` → `modis_m2v`, `noaa20_bridged` → `n20_bridge`, `modis_harmonized_terra_only` → `modis_m2v_terra`).

문의: 데이터·모델 담당 (팀장)
