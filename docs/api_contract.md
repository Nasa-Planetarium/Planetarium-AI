# API 계약 (데이터 담당 확정본, 작성 중) — 데이터 `v1.1-m2v-2026-11-14` (모델 `v1-m2v-2026-11-14`)

> 감지 단위 통합(harmonized hotspots) 2·3단계 기준. 6단계에서 응답 예시·오류 형식까지 채워 확정한다.
> 개요·화면 설명은 `docs/backend-handoff.md`.

## 데이터 원천과 API 대응

| API | 읽는 데이터 | 비고 |
|---|---|---|
| `GET /summary` | **`harmonized_cell_day.parquet`** (+ 그 달 `residual_month`) | 일·월 합계 |
| `GET /hotspots` | **`harmonized_hotspots/{reference,other}/year=YYYY/month=MM/`** | 지도를 확대했을 때 점 |
| `GET /calendar` | `aoi_month.parquet` + `aoi_month_compare.parquet` | 연도 × 월 |
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
