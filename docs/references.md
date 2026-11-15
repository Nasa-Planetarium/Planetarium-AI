# References — Data Sources

Team Planetarium · NASA Space Apps Challenge 2026 · "Harmonization of MODIS and VIIRS Hot Spots"

All datasets below are publicly available. Please keep this list in sync with what the code actually uses (`data/dataset.md`, `data/external/README.md`).
아래 자료는 모두 공개 자료예요. 실제로 쓰는 자료가 바뀌면 이 목록도 같이 고쳐 주세요.

## 1. Active fire hotspots — NASA FIRMS (core data)

- **Products**: MODIS Collection 6.1 (Terra, Aqua; 1 km), VIIRS 375 m Collection 2 (Suomi NPP, NOAA-20, NOAA-21); standard (SP/archive) and near-real-time (NRT) files
- **Used for**: hotspot integration, MODIS→VIIRS harmonization model, burning-activity calendar
- **Location in repo**: `data/raw/firms/`
- **Acknowledgement** (recommended by FIRMS):
  > We acknowledge the use of data from NASA's Fire Information for Resource Management System (FIRMS) (https://firms.modaps.eosdis.nasa.gov), part of NASA's Earth Science Data and Information System (ESDIS).
- **Algorithm papers**:
  - Giglio, L., Schroeder, W., & Justice, C. O. (2016). The collection 6 MODIS active fire detection algorithm and fire products. *Remote Sensing of Environment*, 178, 31–41. https://doi.org/10.1016/j.rse.2016.02.054
  - Schroeder, W., Oliva, P., Giglio, L., & Csiszar, I. A. (2014). The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment. *Remote Sensing of Environment*, 143, 85–96. https://doi.org/10.1016/j.rse.2013.12.008

## 2. Burned area — MODIS MCD64A1 v061 (validation)

- Giglio, L., Justice, C., Boschetti, L., & Roy, D. (2021). *MODIS/Terra+Aqua Burned Area Monthly L3 Global 500m SIN Grid V061* [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MCD64A1.061
- **Used for**: checking whether hotspot detections overlap real burn scars (Korea and Argentina tiles, 2012–2025)
- **Location in repo**: `data/external/burned_area/mcd64a1/`

## 3. Korea Forest Service wildfire statistics (validation, Korea)

- 산림청 (Korea Forest Service). *산림청_산불통계데이터* (Wildfire statistics), 2022–2025. 공공데이터포털 (Public Data Portal of Korea). https://www.data.go.kr/data/15121380/fileData.do — License: 이용허락범위 제한 없음 (no usage restriction)
- **Used for**: comparing satellite detections with officially reported wildfires
- **Location in repo**: `data/external/kfs_wildfire_stats/`

## 4. Administrative boundaries

| Dataset | Citation / source | License | Used for |
|---|---|---|---|
| Korea 시·도 / 시·군·구 / 행정동 | vuski, *admdongkor* ver20261001 (and 20251231). https://github.com/vuski/admdongkor (original data: Statistics Korea, Ministry of the Interior and Safety) | CC BY 4.0 — attribution required on maps | Korea AOIs (province, municipality) |
| geoBoundaries KOR ADM0/1/2 | Runfola, D. et al. (2020). geoBoundaries: A global database of political administrative boundaries. *PLoS ONE*, 15(4), e0231866. https://doi.org/10.1371/journal.pone.0231866 · https://www.geoboundaries.org | ADM1 Public Domain, ADM2 CC BY 3.0 | Korea land mask |
| Natural Earth 1:10m | Natural Earth. *Admin 0 – Countries, Admin 1 – States/Provinces, Land*. https://www.naturalearthdata.com | Public Domain | Global AOIs (country, 1st-level) |
| KOSTAT 2013 (backup) | southkorea/southkorea-maps. https://github.com/southkorea/southkorea-maps (original data: Statistics Korea) | see repository | Backup only |

## 5. Weather

### 5-1. NASA POWER (MERRA-2) — spread nowcast features

- **Service**: NASA POWER Daily API v2.10.3 (point, `community=AG`, time standard LST), source MERRA-2, 0.5° grid. Accessed 2026/11/14.
- **Variables**: WS10M, WS10M_MAX, WD10M, RH2M, T2M_MAX, T2MDEW, PRECTOTCORR (2012-01-01 – 2025-12-31). Daily minimum relative humidity is
  not provided; it is approximated from T2M_MAX and T2MDEW (Magnus formula, see `models/spread_nowcast_meta.json`).
- **Used for**: weather features of the next-day spread nowcast (Korea box, Argentina)
- **Location in repo**: `data/external/weather/power/` (raw JSON cache), `data/external/weather/power_daily.parquet`
- **Acknowledgement** (required by the POWER Referencing Guide, https://power.larc.nasa.gov/docs/referencing/):
  > The data was obtained from National Aeronautics and Space Administration (NASA) Langley Research Center's Prediction Of Worldwide Energy Resources (POWER) project funded through the NASA Earth Science Division.
  > The data was obtained from the POWER Project's Daily 2.10.3 version on 2026/11/14.
- The POWER Project asks to be told about publications and redistribution: larc-power-project@mail.nasa.gov.

### 5-2. Open-Meteo — real-time weather display only (backend)

- Open-Meteo Forecast / Historical Weather API. https://open-meteo.com — data CC BY 4.0 (attribution required: "Weather data by Open-Meteo.com").
- Free API is for **non-commercial use only**: < 10,000 calls/day, 5,000/hour, 600/minute (https://open-meteo.com/en/terms).
- Not used to train models (the free-tier budget cannot cover 2012–2025 for our candidate cells). Keep it for showing current weather in the app.

### 5-3. Canadian Fire Weather Index System — `fwi_cell_day`, `fwi_aoi_day`

- Van Wagner, C. E. (1987). *Development and structure of the Canadian Forest Fire Weather Index System*. Forestry Technical Report 35, Canadian Forestry Service, Ottawa.
- Van Wagner, C. E., & Pickett, T. L. (1985). *Equations and FORTRAN program for the Canadian Forest Fire Weather Index System*. Forestry Technical Report 33 (test values used to check our code).
- Computed from NASA POWER daily statistics instead of noon observations (approximations in `data/processed/fwi_meta.json`).

## 6. Software (main libraries)

Polars, PyArrow, XGBoost, scikit-learn, Shapely, pyproj, uv. (Cite versions from `pyproject.toml` / `uv.lock` when submitting.)

## 7. Team contributions

- **Jonghun Shin ([@ozingmw](https://github.com/ozingmw))** — `handoff/firms-1km-backend` branch: the conversion and installation design was used as a
  reference (팀원 Jonghun Shin(ozingmw)의 firms-1km-backend 작업에서 변환·설치 설계를 참고).
  Adopted ideas, re-implemented under this repository's rules (S-NPP-based `value`, static-source exclusion, sum checks):
  FIRMS CSV/API input checks (satellite code mapping, required columns, header-only responses kept as observed coverage) →
  `planetarium/dataset.py`, `planetarium/converter.py`; release installation with SHA-256 verification → `planetarium/release_assets.py`;
  cross-satellite FRP conversion → `planetarium/frp_equiv.py` (re-derived against S-NPP); bbox query helpers → `planetarium/store.py`.
  The branch itself is kept separate and is not merged.

---

### Notes for the team
- Show the FIRMS acknowledgement on the last slide of the presentation and in the app's "About / Data" page.
- CC BY datasets (admdongkor, geoBoundaries ADM2) need visible attribution under the map in the app.
- FIRMS hotspots are thermal-anomaly detections, not confirmed wildfires. Describe results as "detections" / "observed burning activity".
- Remove any entry we end up not using (e.g., KOSTAT 2013 backup).
