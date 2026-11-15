# 백엔드 빠른 시작 — Planetarium 통합 화재 데이터

작성 2026-11-15 · 데이터 `v1.1-m2v-2026-11-14` (모델 `v1-m2v-2026-11-14`) · 오늘 18:00 이후 `v1.2`로 갱신 예정

> 이 문서는 **"무엇을 받아서, 어떤 파일로, 어떤 API를 만들면 되는지"**만 짧게 정리한 시작 안내예요.
> 컬럼·규칙의 자세한 설명은 `docs/api_contract.md`(확정 명세), 배경·검증은 `docs/backend-handoff.md`를 보세요.

---

## 1. 지금 준비된 것 (한눈에)

| 기능 | 파일 | 범위 | 상태 |
|---|---|---|---|
| 연도 × 월 캘린더 | `aoi_month.parquet` | 전 세계 5,051개 지역, 2000-11 ~ 2026-11 | ✅ |
| 예년 대비 판정 | `aoi_month_compare.parquet` | 같음 | ✅ |
| 날짜 범위 합계·일별 추이 | `harmonized_cell_day.parquet` | 전 세계 0.1° 칸 | ✅ |
| 격자 지도 | `harmonized_cell_month.parquet` | 전 세계 0.1° 칸 | ✅ |
| 통합 핫스팟 점 지도 | `harmonized_hotspots/` | 전 세계 감지 점 | ✅ |
| 지역 목록·경계 | `aoi_list.parquet`, `aoi_boundaries.geojson` | 5,051개 | ✅ |
| 내일 확산 가능성 | `spread_nowcast_latest.parquet` | 한국·아르헨티나 | ✅ (v1.2에 포함) |
| 화재 사건·불탄 면적 | `fire_events.parquet` | 한국·아르헨티나 | ✅ (v1.2에 포함) |
| 불 유형 배지 | `aoi_month_fire_regime.parquet` | 한국·아르헨티나 | ✅ (v1.2에 포함) |
| 산불 기상 위험도 배지 | `fwi_aoi_day.parquet` (+ `fwi_clim_aoi_month.parquet`) | 한국·아르헨티나 | ✅ 2025년까지, 2026년은 오늘 추가 (v1.2) |
| 실시간 갱신 | `uv run python -m planetarium.fetch_nrt ...` | 최근 1~5일 | ✅ |

---

## 2. 파일 받기

### GitHub Release (지금 받을 수 있음)

| Release | 내용 |
|---|---|
| [`data-v1.1-m2v-2026-11-14`](https://github.com/Nasa-Planetarium/Planetarium-AI/releases/tag/data-v1.1-m2v-2026-11-14) | `harmonized_cell_month`, `harmonized_cell_day`, `harmonized_residual_cell_month`, `aoi_month`, `aoi_month_compare`, `aoi_list`, `aoi_boundaries.geojson`, `m2v_meta.json` (약 1.7 GB) |
| [`data-v1.1-hotspots-2026-11-14`](https://github.com/Nasa-Planetarium/Planetarium-AI/releases/tag/data-v1.1-hotspots-2026-11-14) | `harmonized_hotspots_<연도>.tar` 27개 (약 6.9 GB) |

```powershell
# 캘린더·지역·날짜용 (먼저 이것만 받아도 핵심 화면이 돼요)
gh release download data-v1.1-m2v-2026-11-14 -D data/processed
# 점 지도용 (필요한 연도만 받아도 됨)
gh release download data-v1.1-hotspots-2026-11-14 -p "harmonized_hotspots_2025.tar" -D tmp
tar -xf tmp/harmonized_hotspots_2025.tar -C data/processed
```

- 확산·사건·불 유형·FWI 파일은 **오늘 18:00 이후 `v1.2` Release**에 들어가요. 그 전에는 데이터 담당 PC의 `C:\Planetarium-AI\data\processed\`에 있어요(필요하면 요청).
- v1.2가 나오면 같은 이름 파일을 바꿔 끼우기만 하면 돼요(컬럼 구조 동일).

### 서버 폴더 구조 (권장)

```
data/processed/
├── aoi_list.parquet, aoi_boundaries.geojson
├── aoi_month.parquet, aoi_month_compare.parquet
├── harmonized_cell_month.parquet, harmonized_cell_day.parquet, harmonized_residual_cell_month.parquet
├── harmonized_hotspots/reference/year=YYYY/month=MM/part-0.parquet
├── harmonized_hotspots/other/year=YYYY/month=MM/part-0.parquet
├── spread_nowcast_latest.parquet, fire_events.parquet, aoi_month_fire_regime.parquet   (v1.2)
└── fwi_aoi_day.parquet, fwi_clim_aoi_month.parquet                                     (v1.2)
```

---

## 3. API ↔ 파일 대응 (만들 API)

| API | 읽는 파일 | 화면 | 우선순위 |
|---|---|---|---|
| `GET /aoi?level&parent_id&q` | `aoi_list` (+ `aoi_boundaries`) | 지역 선택·검색 | ⭐ 필수 |
| `GET /calendar?aoi_id&start_year&end_year` | `aoi_month` + `aoi_month_compare` | 연도 × 월 캘린더 | ⭐ 필수 |
| `GET /compare?aoi_id&year&month` | `aoi_month_compare` | 예년 대비 상세 | ⭐ 필수 |
| `GET /summary?aoi_id\|bbox&start&end&freq=day\|month` | `harmonized_cell_day` | 날짜 범위 합계·추이 | ⭐ 필수 |
| `GET /export.csv?type=calendar\|summary\|hotspots&...` | 위 결과 | CSV 다운로드 | ⭐ 필수 |
| `GET /hotspots?bbox&start&end&layer=reference\|all&limit` | `harmonized_hotspots` | 확대 지도 점 | 높음 |
| `GET /grid?bbox&year&month` | `harmonized_cell_month` | 넓은 지도 격자 | 높음 |
| `GET /spread?aoi_id\|bbox` | `spread_nowcast_latest` | 내일 확산 가능성 레이어 | 높음 |
| `GET /events?aoi_id&start&end` | `fire_events` | 화재 사건 목록·영역 | 중간 |
| `GET /regime?aoi_id` | `aoi_month_fire_regime` | 불 유형 배지 | 중간 |
| `GET /fwi?aoi_id&date` | `fwi_aoi_day` | 산불 기상 위험도 배지 | 중간 |
| `GET /method` | `m2v_meta.json` | 방법·검증 설명 | 낮음 |
| `GET /weather?lat&lon` | Open-Meteo Forecast API (외부, 실시간) | 현재 바람·습도 | 낮음 |

**지역 ID 예시**: `WORLD`, `KR`(한국), `SIDO_11`(서울), `SGG_47730`(의성군), `ADM0_ARG`(아르헨티나). 전체 목록은 `aoi_list.aoi_id`.

---

## 4. 읽는 법 (Polars 예시)

```python
import polars as pl
P = "data/processed"

# 캘린더: 한 지역의 연도 × 월
cal = (pl.scan_parquet(f"{P}/aoi_month.parquet")
         .filter(pl.col("aoi_id") == "KR")
         .join(pl.scan_parquet(f"{P}/aoi_month_compare.parquet")
                 .select("aoi_id", "year", "month", "percentile", "ratio_to_median", "verdict",
                         "baseline_median", "baseline_p25", "baseline_p75"),
               on=["aoi_id", "year", "month"], how="left")
         .sort("year", "month").collect())

# 점 지도: 확대된 영역·기간만 (폴더째 읽고 필터)
pts = (pl.scan_parquet(f"{P}/harmonized_hotspots/reference/**/*.parquet", hive_partitioning=True)
         .filter((pl.col("year") == 2025) & (pl.col("month") == 3))
         .filter(pl.col("longitude").is_between(128.4, 129.6) & pl.col("latitude").is_between(36.0, 36.8))
         .select("latitude", "longitude", "acq_datetime_utc", "date_local", "satellite", "frp",
                 "confidence", "harmonized_weight")
         .head(5000).collect())

# 내일 확산 가능성 (bbox 안)
spread = (pl.read_parquet(f"{P}/spread_nowcast_latest.parquet")
            .filter(pl.col("cell_lon").is_between(128.4, 129.6) & pl.col("cell_lat").is_between(36.0, 36.8))
            .with_columns(pl.when(pl.col("p_spread") > 0.20).then(pl.lit("high"))
                            .when(pl.col("p_spread") >= 0.05).then(pl.lit("moderate"))
                            .otherwise(pl.lit("low")).alias("level")))
```

- 큰 파일(`harmonized_cell_day` 965 MB, `harmonized_hotspots` 7 GB)은 **`scan_parquet` + 필터**로 필요한 부분만 읽으세요. 서버 시작 시 전체를 메모리에 올리지 마세요.
- 작은 파일(`aoi_month`, `aoi_month_compare`, `aoi_list`, `fire_events`, `fwi_aoi_day`, `spread_nowcast_latest`)은 시작 시 읽어 두면 빨라요.

---

## 5. 꼭 지켜야 할 규칙 (요약)

1. **단위 표기**: "산불 건수" ❌ → **"위성 화재 감지량 (VIIRS 기준)"** ✅. 2000-11 ~ 2012-01은 **추정값**이라 "추정" 표시 + 범위(`value_lo`~`value_hi`)를 함께.
2. **0과 결측 구분**: 칸 표에 행이 없으면 0. 그 달이 결측인지는 `aoi_month.status`로 판단. `missing`은 회색, 0으로 칠하지 않기.
3. **status 표시**: `ok` 일반 / `nrt` 빠른 처리본(바뀔 수 있음) / `partial` 진행 중인 달(`value`=오늘까지 실제, `value_projected`=달 전체 추정, `as_of_date`=마지막 관측일) / `missing` 자료 부족.
4. **verdict 배지**: `above`(예년보다 많음) / `similar` / `below` / `uncertain`(판단 보류: 비교 연도 부족·결측·진행 중).
5. **지역 합계는 `aoi_month`를 그대로**: 칸에서 다시 더하면 배경 추정·면적 가중이 빠져 숫자가 달라져요.
6. **residual은 화면에 그리지 않기**: 합계 계산에만(자세한 규칙은 `api_contract.md`의 `/summary`). 점 지도에는 MODIS 기간의 "점으로 안 보이는 추정분"이 없어서, 점 합이 월 합계보다 작을 수 있다는 안내 문구를 넣어요.
7. **합계에는 기준 점만**: `harmonized_hotspots/reference`의 `harmonized_weight`만 더해요. `other/`는 지도 표시용.
8. **확산 레이어 필수 문구**: "위성 감지 기반 단기 확산 가능성입니다. 물리 시뮬레이션이 아니며, 오늘 감지 지점에서 반경 약 7.5 km 밖으로 번지는 경우와 새로 발생하는 불은 예측 대상이 아닙니다." 색 3단계(<5% / 5–20% / >20%), 파일에 없는 칸은 "대상 아님"(0% 아님).
9. **불탄 면적 문구**: `burned_area_km2` = "위성 확인 연소 면적(하한)" — 실제보다 작아요(의성 약 64%).
10. **FWI 배지**: 기본은 상대 등급 `fwi_rel_class`(평년보다 낮음/평년/높음/매우 높음), 상세 화면에 절대 등급 `fwi_class`. 문구: "기상으로 본 산불 위험도. 실제 발화·확산을 예측하지 않음."
11. **불 유형 배지와 캘린더 값이 다를 수 있음**: 캘린더 값은 `aoi_month`, 불 유형은 `fire_regime` 기준(점 위치로 지역을 정해서 작은 값이 있는 시군구도 `none`일 수 있음).
12. **출처 표시**: 앱 하단 또는 About 페이지에 NASA FIRMS 인정 문구, 경계 데이터(CC BY), NASA POWER, MCD64A1, Open-Meteo(실시간 기상) 출처 → `docs/references.md`.
13. **비밀값**: `FIRMS_MAP_KEY`는 서버 `.env`에만(설정 방법은 6-1). 프론트엔드 코드·GitHub·채팅에 절대 넣지 않기.

---

## 6. 실시간 갱신 (서버에서 주기 실행)

### 6-1. 먼저 `.env`에 FIRMS 키 넣기 (실시간 갱신에 필수)

1. 키 받기: https://firms.modaps.eosdis.nasa.gov/api/map_key/ 에 이메일을 넣으면 MAP_KEY가 메일로 와요(무료, 10분에 5,000회).
   팀 키를 공유받기보다 **서버 담당자가 직접 발급**하는 걸 권해요.
2. 저장소 루트(`C:\Planetarium-AI\` 또는 서버의 프로젝트 폴더)에 `.env` 파일을 만들고 한 줄 넣기:
   ```
   FIRMS_MAP_KEY=여기에_받은_키
   ```
3. 확인: `uv run python -m planetarium.fetch_nrt --status` → 남은 호출 수가 나오면 정상. 키가 없으면 명확한 오류 메시지가 나와요.

- `.env`는 `.gitignore`에 있어서 GitHub에 안 올라가요. **절대 커밋·채팅·프론트엔드 코드·스크린샷에 넣지 마세요.** 노출되면 FIRMS에서 새 키를 받으세요.
- 코드는 환경변수 `FIRMS_MAP_KEY`로만 읽고, 로그의 URL에서는 키를 `***`로 가려요.
- 다른 외부 자료는 키가 필요 없어요: NASA POWER(기상), Open-Meteo(실시간 기상 표시). Earthdata 로그인은 MCD64A1을 새로 받을 때만 필요하고 서버 운영에는 필요 없어요.

### 6-2. 갱신 실행

```powershell
# 최근 2일 감지 받기 → 통합 데이터에 추가 → 내일 확산 가능성 다시 계산 (API 호출 3회)
uv run python -m planetarium.fetch_nrt --fetch --days 2 --append --with-spread
# 키 상태 확인
uv run python -m planetarium.fetch_nrt --status
```

- 3~6시간마다 한 번이면 충분해요(FIRMS NRT는 관측 후 약 3시간 안에 올라와요). 한 번 실행에 약 8분, 최대 메모리 약 18 GB라 **작은 서버에서는 메모리를 먼저 확인**하세요.
- 받는 중인 날(오늘)은 `partial_current_day`로 표시되고, 확산 예측 기준일은 "마지막으로 완결된 정상 관측일"이에요.

---

## 7. 화면에 바로 쓸 수 있는 숫자 (발표·About용)

- MODIS → VIIRS 변환(MODIS-to-VIIRS Harmonizer): 한국 편향 **1.40 → 0.99**, 월 오차 절반, 추정 범위 적중률 **80%**
- 통합 핫스팟 **5.5억 점**, 점 → 칸 → 지역 합계 **모두 일치** 검증
- 산림청 큰 산불(100 ha 이상) **92% 탐지**, 한국 14년 중 대형 산불형 달 = 2022-03 울진·삼척, 2025-03 의성
- 내일 확산 가능성: 단순 규칙 대비 PR-AUC **약 4.6배**(한국), 확률 보정 양호

---

## 8. 문의·참고 문서

- 확정 명세: `docs/api_contract.md`
- 배경·검증·컬럼 전체: `docs/backend-handoff.md`
- 출처·라이선스: `docs/references.md`
- 데이터 배포 노트: `docs/release_v1.1-m2v-2026-11-14.md` (v1.2는 오늘 저녁)
- 데이터·모델 담당: 팀장
