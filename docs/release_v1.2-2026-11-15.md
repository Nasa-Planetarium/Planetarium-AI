# Data release v1.2-2026-11-15

Harmonized MODIS–VIIRS fire activity, 2000-11 to 2026-11 (VIIRS S-NPP–equivalent detection counts), plus new fire-information layers
for Korea and Argentina. Files: `data/processed/release_v1.2-2026-11-15/` (GitHub Release `data-v1.2-2026-11-15`; not in git because of size).

- Data as of: last NRT append 2026-11-15 10:26 KST (detections up to 2026-11-14 22:07 UTC). 2026-11 is in progress (`status = partial`).
- The MODIS→VIIRS conversion model **FireBridge (MODIS-to-VIIRS Harmonizer)** (internal id: `m2v_xgb`) and `method_version`
  (`v1-m2v-2026-11-14`) are **unchanged** — FireBridge was not retrained for v1.2 (only FireOutlook Spread (internal id: `spread_nowcast`) was, see below).
  Display names ↔ internal ids: FireBridge = `m2v_xgb`, FireBridge-Terra = `m2v_xgb_terra`, FireBridge Uncertainty Band = `m2v_q10_q90`,
  N20 Bridge = `n20_bridge`. File names, code identifiers and data values (`value_source`, `method_version`) keep the internal ids.
- File list with sizes and SHA-256: `release_manifest.csv` (also below).

## Changes from v1.1

### 1. Industrial heat sources removed (static-source points)
- v1.1 kept standard-processing (SP) points with FIRMS `type = 0` even when they fell inside `static_source_cells_0.01deg`
  (0.01° cells where SP `type` 1 or 2 appeared in at least two years: steel works, petrochemical and cement plants, gas flares).
- v1.2 uses **one rule for SP and NRT: points inside those cells are `is_static_source` and are removed from all totals.**
  Point files are not rewritten; the per-point flag is in `harmonized_hotspots_static_flags.tar`.
- Method: cell `value` minus the static reference-point weights of that cell and period; `value_lo`/`value_hi` minus the same share
  (in the MODIS period point weights are proportional to the cell value, so this is the same ratio); residual unchanged.
  Removed amounts are kept in `static_removed*` columns, so the step is repeatable (`planetarium/static_exclusion.py`, also part of `fetch_nrt --append`).
- Cell values are not clipped at zero: in the VIIRS period, points on days without usable data still carry weight, so a cell where static points
  dominate can go slightly below zero; the existing negative `missing_day_fill` residual balances it. Draw such cells as 0.
- Removed share of `aoi_month`:

| AOI | 2000-11 – 2012-01 | 2012-02 – 2026-11 |
|---|---|---|
| Korea (KR) | 7.0% | **22.7%** |
| Argentina | 0.7% | 1.3% |
| World | 0.4% | 1.1% |

- Korea examples: 2025-03 (Uiseong fires) 2,764 → 2,731 (−1.2%); 2024-12 26 → 15; 2024-08 162 → 96; 2026-10 (NRT, already clean) 164.6 unchanged.
- Verdict changes: Korea national 74 of 313 months; Korean AOIs (national, province, municipality) 5,529 of 84,510 AOI-months
  (3,826 of them to `uncertain`, because previous-year values became small); Argentina national 1 month.
  NRT months (2026-08 onward) were already free of static points but were compared with baselines that still had them; they now compare fairly.
- Fire events in Korea: 18,586 → 12,397. Korea Forest Service match rate for fires ≥ 100 ha unchanged (23/25, 92%); < 1 ha 138 → 112 detected.

### 2. New layers (Korea box and Argentina)
- **FireOutlook Spread** (`spread_nowcast_latest`): probability that a 0.025° cell near today's fire has a detection tomorrow (XGBoost).
  Base date = last NOAA-20 `ok` day (2026-11-13 → target 2026-11-14).
  **Model v1.2-ops (replaced on 2026-11-15): validated under the same weather delay as operations.** NASA POWER arrives about 4 days late, so
  weather from the base date − 4 days was compared with no weather at all; no weather was better for both countries and is now the default
  (fire features only: detections and FRP today and yesterday, burning streak, fire-event duration, direction). Leave-year-out 2012–2025 on the
  static-source-free sample, PR-AUC / Brier: Korea box 0.147 / 0.00546, Korea only 0.210 / 0.00316, Argentina 0.190 / 0.01231
  (base date − 4 days weather: 0.139 / 0.198 / 0.189). The earlier model was validated with same-day weather (0.158 / 0.212 / 0.197 on the same
  sample), which operations cannot have, so its figures were somewhat optimistic. Details: `reports/spread_ops_metrics.csv`.
- **Fire events and burned area** (`fire_events`): reference points linked within 750 m (VIIRS) / 2 km (MODIS) and one day; MCD64A1 burned area
  (lower bound; Uiseong 2025-03 ≈ 64% of the official area).
- **Fire-type badge** (`aoi_month_fire_regime`).
- **Fire Weather Index** (`fwi_aoi_day`): absolute class (Canadian thresholds) and relative class (percentile within the same AOI and calendar month,
  2012–2025). App default = relative class; the Canadian absolute class does not fit Argentina well.
- **FireOutlook Recurrence** (internal id: `recurrence_nextmonth`; `recurrence_nextmonth_latest`, `recurrent_burn_sites`): Argentina = model probability of activity next month
  (expanding-window PR-AUC 0.57 vs 0.49 for the recurrence rate alone, static cells excluded); Korea = count of years active in that month
  ("active in N of the last 25 years", not a probability).

### 3. FireOutlook Monthly — next-month occurrence outlook (added to the release on 2026-11-15)
- Probability of at least one detection next month for Korean municipalities (FireOutlook Monthly – District, internal id: `aoi_occurrence`) and a global 0.5° grid (FireOutlook Monthly – Grid, internal id: `grid_occurrence`) — see the section below.

### 4. FRP (S-NPP equivalent) (added to the release on 2026-11-15)
- Fire radiative power converted to the S-NPP level for every satellite, as a separate daily table — see the section below.

### 6. FireOutlook Planner — forecast for any region and period (added to the release on 2026-11-15)
- `forecast(region, start, end)` / `POST /forecast`: probability of ≥ 1 satellite detection and expected detections (q10–q90) for an aoi_id,
  a bbox or a GeoJSON polygon, 1–7 days and 1–12 months ahead, from precomputed 0.5° grid forecasts — see the section below.

### 5. Other
- `satellite_day_status` and `satellite_month_status` are now part of the release.
- Verification after all changes: points + residual = cell value, AOI sums = `aoi_month`, daily table monthly sums = `aoi_month` — 0 mismatches for
  every year 2000–2026 (`reports/hotspots_verify_*.csv`).

## FireOutlook Monthly (next-month occurrence outlook)

Two models give the probability that **at least one detection** occurs next month (target 2026-12), after static-source removal.
They describe past patterns and recent activity; they are not weather forecasts.

| | FireOutlook Monthly – District (Korea municipalities) | FireOutlook Monthly – Grid (global) |
|---|---|---|
| File | `aoi_occurrence_nextmonth_latest.parquet` | `grid_occurrence_nextmonth_latest.parquet` |
| Unit | 252 municipalities (시군구, `aoi_list` level `sgg`) | 52,855 land 0.5° cells with any detection since 2012-02 (Antarctica and open ocean excluded) |
| Target | ≥ 1 VIIRS reference detection inside the municipality (point location) in the month | ≥ 1 reference detection in the cell (`harmonized_cell_day.n_points`) in the month |
| Training data | 2012-02 – 2026-10, months with status `ok`/`nrt`; MODIS-era estimates and the month in progress are not used as targets | same |
| Features | same-month occurrence rate (all years / last 5 years, current year excluded), same-month median value, values 1–3 months before and a year before, province same-month rate, previous-month FWI and its percentile among earlier years, month | same, with country same-month rate and 3×3 neighbour same-month rate instead of province and FWI |
| Model | XGBoost + Platt calibration | XGBoost + isotonic calibration |
| Validation | expanding window, test years 2016–2025 | expanding window, test years 2020–2025 |

The month in progress (2026-11) enters as a lag feature through its month-to-date projection (`value_projected`); this is flagged with `lag1_projected`.
Argentina has no municipality-level areas in `aoi_list` (provinces only): use the cell-level recurrence probability or the global grid there.
Department-level (departamento) areas are future work.

**Validation** (pooled over test years; PR-AUC / Brier; lower Brier is better)

| Group | Rows | Positive share | Model | Baseline: same-month rate | Baseline: same month last year |
|---|---|---|---|---|---|
| Korea municipalities | 30,240 | 13.0% | **0.422 / 0.0927** | 0.404 / 0.0964 | 0.223 / 0.163 |
| Grid, all (persistent sources excluded) | 3,786,912 | 35.3% | **0.900 / 0.096** | 0.869 / 0.104 | 0.648 / 0.178 |
| Grid, Africa | 618,192 | 52.7% | **0.956 / 0.093** | 0.941 / 0.101 | 0.787 / 0.173 |
| Grid, Asia | 838,296 | 38.2% | **0.882 / 0.112** | 0.850 / 0.121 | 0.639 / 0.205 |
| Grid, Europe | 944,856 | 16.5% | **0.713 / 0.080** | 0.660 / 0.089 | 0.382 / 0.149 |
| Grid, North America | 682,848 | 28.2% | **0.870 / 0.088** | 0.835 / 0.095 | 0.579 / 0.166 |
| Grid, Oceania | 254,232 | 31.4% | **0.832 / 0.112** | 0.797 / 0.121 | 0.553 / 0.207 |
| Grid, South America | 448,488 | 58.2% | **0.948 / 0.107** | 0.928 / 0.115 | 0.785 / 0.200 |
| Grid, Korea cells | 4,320 | 31.9% | **0.680 / 0.155** | 0.650 / 0.165 | 0.456 / 0.281 |
| Grid, Argentina cells | 79,848 | 43.6% | **0.893 / 0.123** | 0.862 / 0.134 | 0.650 / 0.236 |

- Both models beat both baselines (higher PR-AUC and lower Brier) in every group, so both are adopted. **The gain over the same-month-rate
  baseline is small** (PR-AUC +0.02 to +0.05, Brier −0.004 to −0.010): most of the signal is the past same-month rate itself.
  If a future update does not beat the baseline, the files carry `model = baseline_same_month_rate` and `p_any = baseline_p`.
- Korea: 45% of municipality × calendar-month combinations never had a detection since 2012 (positive share < 5%); 0.07% are above 95%.

**Industrial and persistent sources** (shown grey and left out of rankings and highlights)
- `suspected_industrial` (Korea municipalities): detections in ≥ 25% of months since 2012-02, little seasonality (coefficient of variation of the
  12 calendar-month occurrence rates ≤ 0.40; Korea overall 0.48) and every calendar month ≥ 10%. Flagged: Ulsan Nam-gu, Pohang Nam-gu,
  Haman-gun, Jeju-si (Jeju-si may be year-round burning rather than industry). Ulju, Dangjin and Gwangyang are seasonal and are not flagged.
- `persistent_source` (grid): detections in ≥ 75% of months, coefficient of variation of the mean detection count per calendar month ≤ 1.0, and
  ≥ 12% of the cell's detections inside its top three 0.01° cells. 259 cells (China 58, USA 48, Iran 16, Egypt 12, Saudi Arabia 9, …).
  The concentration share is 0.003–0.015 at checked wildfire sites and 0.05–0.28 at known industrial sites; 0.10 already admitted farmland cells
  in Brazil and Australia, so 0.12 was used.
- Caught: Ulsan, Pohang, Basra, Permian (New Mexico cell). **Missed:** widespread flare fields (Niger Delta, Permian Delaware cell: concentration
  about 0.05), and places whose flares are already removed by the static list so that only seasonal fires remain (Khanty-Mansiysk, Dangjin,
  Gwangyang). Not flagged (as intended): Amazon, Congo basin and northern Australian savanna check cells.
- Details: `reports/occurrence_*.csv`, `reports/grid_occurrence_*.csv`, `reports/occurrence_calibration.png`, `reports/grid_occurrence_calibration.png`.

## FireOutlook Planner (period forecast)

**What it answers.** "In region R between dates A and B, what is the probability of at least one satellite fire detection, and how many detections
(S-NPP equivalent) are expected?" Region = any of the 5,051 aoi_ids, a bbox or a GeoJSON polygon; period = from the day after the reference date
(`as_of`, now 2026-11-14) up to 12 months ahead. Answers come from precomputed 0.5° grid forecasts in under 1 second
(`planetarium.forecast.forecast`, CLI `python -m planetarium.forecast`). Required notice: "위성 탐지 기반 확률이며 산불 발생을 보장하지 않습니다."
(probability of satellite detections; it does not guarantee that a wildfire will occur).

**Period pieces (overlap rule).**
1. Days 1–7 after `as_of`: the **daily model** (per-day hazard from "≥ 1 within k days", k = 1–7).
2. Day 8 onwards: the **monthly model**, month by month. A partial month is scaled with **daily climatology weights**, separately for counts and
   for probability: f_count = share of the month's climatological detections that fall on the requested days, f_prob = the same with the
   climatological share of detection days. Climatology = 2012–2025 mean for the same calendar date, ±7-day moving average, per 0.5° cell; cells with
   fewer than 15 detection days in that calendar month borrow the country (then continent) shape (cell 34%, country 65%, continent 0.6%).
   p = 1 − (1 − p_month)^f_prob, counts × f_count. Days already covered by the daily model are not counted again.
3. Remaining days of the current month that the daily model does not cover: baseline (same calendar month in past years) with the same weights, `skill = none`.
4. Pieces are combined as p = 1 − Π(1 − p_piece); counts and interval bounds are summed.

Partial-month weights, holdout 2024–2025 (91 reference days × windows of 7 days starting 8–90 days ahead, 0.5° cells, 38.3 M cell-windows,
climatology from 2012–2023 only):

| Window start | Count WAPE: uniform → daily weights | Brier: uniform → daily weights |
|---|---|---|
| 8–14 days | 0.775 → 0.766 | 0.0876 → 0.0871 |
| 15–30 days | 0.843 → 0.824 | 0.0845 → 0.0841 |
| 31–60 days | 0.922 → 0.891 | 0.0856 → 0.0853 |
| 61–90 days | 0.933 → 0.904 | 0.0867 → 0.0864 |
| All | **0.876 → 0.853** | **0.0857 → 0.0853** |

Using the count shape for probability as well made Brier worse (0.0870), hence the separate detection-day shape.

**Models and validation** (selection 2020–2023, holdout 2024–2025 without re-selection, extra check 2026-01..10; persistent-source cells excluded).

| Unit | Horizon | Selected | Brier skill vs same-month baseline (validation / holdout / 2026) |
|---|---|---|---|
| 0.5° grid, monthly | 1 month | FireOutlook Monthly – Grid (unchanged; Planner ens: +8.2% / +7.7% / +8.0%, holdout Brier only 0.00015 better) | — |
| 0.5° grid, monthly | 2 months | ens = XGBoost + LightGBM (isotonic) | +5.8% / +5.4% / +5.3% |
| 0.5° grid, monthly | 3 months | ens | +5.3% / +4.8% / +4.6% |
| 0.5° grid, monthly | 6 months | ens | +4.9% / +4.4% / +4.0% |
| 0.5° grid, monthly | 12 months | ens | +4.8% / +4.4% / +4.2% |
| 0.5° grid, daily | 1 day | ens = XGBoost + LightGBM (weighted isotonic) | +16.3% / +15.9% / +17.7% |
| 0.5° grid, daily | 2 days | ens | +14.5% / +13.6% / +15.6% |
| 0.5° grid, daily | 3 days | ens | +13.4% / +12.3% / +14.4% |
| 0.5° grid, daily | 7 days | ens (4–6 days follow the nearest validated choice) | +11.3% / +10.2% / +12.5% |
| Korean municipality | 1 month | FireOutlook Monthly – District (unchanged) | — |
| Korean municipality | 2 months | direct XGBoost + Platt (same method as District) | +3.4% / +1.9% / +4.2% (grid aggregation −3.3% / −0.3% / −1.8%) |
| Korean municipality | 3 months | direct XGBoost + Platt | +3.3% / +3.8% / +4.5% (grid aggregation −3.2% / −0.6% / −2.1%) |

Other candidates (logistic regression, Tweedie count model, 5-year baseline) ranked lower at every horizon. Regions other than single cells and
Korean municipalities use grid hazard sums calibrated per region-size bin; the gain mostly disappears for large regions, so regions get the model value
only where both validation and holdout skill ≥ 1% (all regions and adm1: 1 month; Korean municipalities: up to 6 months) and the baseline elsewhere
(`skill = none`). Counts: chosen per region group and horizon by validation WAPE and used only if the holdout WAPE improves on the
same-month median by ≥ 5% — world AutoTheta/AutoETS (holdout WAPE 0.116–0.192 vs 0.271), continents grid μ (0.256–0.297 vs 0.318, up to 6 months),
Argentina AutoETS (1 month, 0.301 vs 0.388), countries grid μ (1 month); Korea, adm1 and all other cases use the median. Daily counts: grid μ
(WAPE 23–31% lower than climatology). Reports: `planner_*.csv`.

**30-day screen query** (`horizon_days = 30`: today (KST) to today + 29; `daily[]` with `date`, `lead_days_from_as_of`, `p_day`,
`base_p_day`, `expected_count_day`, `skill`). Validation of "the 30 days after the reference date", holdout 2024–2025, 51 reference days,
same composition as the service (days 1–7 daily model, days 8–30 monthly model × daily weights, pieces combined with 1 − Π):

| Unit | Observed rate | Mean prediction (model / baseline) | BSS raw / cross-fitted | Skill | Count WAPE (model / baseline) | Count skill |
|---|---|---|---|---|---|---|
| 0.5° grid cell | 0.350 | 0.363 / 0.369 | +10.1% / +8.8% | high | 0.589 / 0.621 | low |
| Korean municipality | 0.066 | 0.107 / 0.132 | +12.9% / +0.9% | none | 1.376 / 1.118 | none |
| Korean province (sido) | 0.544 | 0.578 / 0.629 | +10.6% / +6.4% | high | 1.136 / 0.973 | none |
| adm1 (e.g. Argentine provinces) | 0.534 | 0.565 / 0.582 | +5.7% / +2.1% | low | 0.422 / 0.461 | low |
| Country | 0.741 | 0.727 / 0.733 | −1.3% / −0.7% | none | 0.320 / 0.360 | low |
| Continent / world | 1.000 | 1.0 / 1.0 | no information | none | 0.236 / 0.272 · 0.149 / 0.184 | low |

The 1 − Π composition is close to calibrated on grid cells (+0.02 at most per decile) but **over-predicts for regions** (municipalities
0.107 vs 0.066; adm1 0.05–0.10 too high in the 0.5–0.9 range), so the summary skill uses the smaller of the raw and the cross-fitted
(isotonic fitted on one year, applied to the other) Brier skill. **Values follow "show the normal when the model does not beat it"**
(`value_source`): units with skill `none` (municipalities, countries, continents, world) return the calibrated baseline as `p_any` and
`daily[].p_day = base_p_day`; `high`/`low` units (grid cells, provinces, adm1) return `p_any` calibrated with a per-unit isotonic curve
(fitted on the 2024–2025 holdout) and scale the daily hazards so that 1 − Π(1 − p_day) = p_any. After cross-fitted calibration the mean
prediction matches the observed rate (grid 0.351 / 0.350, provinces 0.550 / 0.544, adm1 0.535 / 0.534, municipalities 0.068 / 0.066;
`planner_30d_calibration_after.csv`). For month pieces with probability ≥ 0.98 (large regions), daily probabilities come from the
region's historical share of days with at least one detection on the same calendar date (±7 days, 2012–2023; `planner_day_any.npz`)
instead of 1 − (1 − p)^f, which would put every day near 1: Córdoba's expected detection days for 2026-11-15 – 12-14 moved from 19.0 (model) /
29.5 (baseline) to 25.3 / 25.4 against an observed 2016–2025 average of 25.1 (`planner_30d_fire_days_check.csv`). Models were not changed. Response time after `forecast.warmup()`:
0.006–0.012 s for municipalities, provinces, bbox and polygons, 0.26 s for the world (`planner_30d_*.csv`).

**Service files** are listed below (`planner_*`). Daily refresh: `fetch_nrt --fetch --append --with-planner` (about 9 minutes, up to about 17 GB);
monthly refresh: `planner_monthly final --origin YYYY-MM --as-of YYYY-MM-DD` then `planner_publish publish`.
Licenses: service models are XGBoost, LightGBM, scikit-learn and statsforecast (Apache-2.0, optional extra `forecast-stats`). TimesFM
(non-commercial license) was used only for comparison and is not part of the service.

## FRP (S-NPP equivalent)

FRP from different satellites is not directly comparable (pixel size, overpass time). `frp_snpp_equiv` puts it on the S-NPP level, in the same
spirit as `value` (S-NPP-equivalent detections). Existing files are unchanged; the result is a separate table.

**Method.** For every 0.1° cell and local date where S-NPP and another satellite both detected fire, take
r = Σ FRP(satellite) / Σ FRP(S-NPP). The ratio is the median of r over those cell-days, separately for day and night.
Data: FIRMS standard processing (SP), `type = 0`, static-source cells excluded, local dates from 2012-02.
Comparison years: Aqua 2012–2025, Terra 2012–2019, NOAA-20 2018–2025, NOAA-21 2024–2025.

**Conversion: `frp_snpp_equiv = frp ÷ ratio(satellite, day/night)`** (S-NPP: ratio 1). Use the same formula for single points.

| Satellite | Day ratio | Night ratio | Yearly range (day) | Overlapping cell-days |
|---|---|---|---|---|
| Aqua (MODIS) | 1.141 | 0.825 | 1.12–1.19 | 14.75 M |
| Terra (MODIS), 2012–2019 | 0.963 | 2.406 | 0.91–1.00 | 3.65 M |
| NOAA-20 (VIIRS) | 0.999 | 1.018 | 0.99–1.01 | 23.98 M |
| NOAA-21 (VIIRS) | 1.167 | 1.222 | 1.17 | 6.48 M |

File `harmonized_cell_day_frp.parquet` (0.1° cell × local date, reference points, static sources excluded): `frp_sum` (native),
`frp_snpp_equiv_sum`, `frp_snpp_equiv_max`, `n_points`. Its `n_points` total equals the `n_points` total of `harmonized_cell_day` (321,562,117).
Ratios by year: `reports/frp_snpp_ratios.csv`. Point-level values are not distributed; compute them with the formula above.

**Limitations**
- **Terra night ratio (2.41) is low-confidence:** Terra's night overpass is about 3 hours earlier than S-NPP's, so the ratio mixes the
  sensor difference with the evening-to-night decline of fire intensity.
- **Terra uses 2012–2019 only:** its orbit drifted from about 2020 (earlier overpass); the day ratio fell from 0.98 to 0.75 by 2025.
- **2000–2011 MODIS points use the ratios measured after 2012** (there is no S-NPP before 2012): an assumption.
- Ratios are global medians; they do not vary by region or fire type.

## Files

| File | Rows | Size | Description |
|---|---|---|---|
| `aoi_boundaries.geojson` | — | 12.9 MB | AOI boundaries (WGS84) |
| `aoi_list.parquet` | 5,051 | 76.3 KB | AOI id, name, level, parent, `n_cells`, `hidden` |
| `aoi_month.parquet` | 1,580,963 | 30.4 MB | AOI × month totals (5,051 AOIs); recomputed after static-source removal |
| `aoi_month_compare.parquet` | 1,580,963 | 31.0 MB | Same-month comparison with previous years; verdicts recomputed |
| `aoi_month_fire_regime.parquet` | 92,335 | 0.3 MB | **New.** AOI × month fire-type badge (`large_fire`, `many_small`, `mixed`, `none`) |
| `aoi_occurrence_nextmonth_latest.parquet` | 252 | 11.0 KB | **New.** Next month (2026-12) probability of ≥ 1 detection per Korean municipality (`p_any`, `p_none`, `baseline_p`, `suspected_industrial`) |
| `fire_events.parquet` | 1,240,978 | 47.3 MB | **New.** Fire events (space-time linked reference points), Korea + Argentina, satellite-confirmed burned area (lower bound) |
| `fire_events_meta.json` | — | 5.0 KB | **New.** Event rules, burned-area rules, KFS validation |
| `fwi_aoi_day.parquet` | 1,601,555 | 29.1 MB | **New.** Canadian FWI per AOI × day (Korea, Argentina), 2012-01-01 – 2026-11-11, absolute + relative class |
| `fwi_clim_aoi_month.parquet` | 3,540 | 41.9 KB | **New.** AOI × calendar-month FWI quantiles (q25, q50, q75, q90; 2012–2025) for relative classes of new dates |
| `fwi_meta.json` | — | 10.4 KB | **New.** FWI approximations, classes, validation |
| `grid_occurrence_nextmonth_latest.parquet` | 52,855 | 0.3 MB | **New.** Next month (2026-12) probability of ≥ 1 detection per global 0.5° cell (`p_any`, `p_none`, `baseline_p`, `persistent_source`) |
| `harmonized_cell_day.parquet` | 212,790,231 | 973.5 MB | 0.1° cell × local date. v1.2: static-source points removed (`static_removed*`, `static_n_points`) |
| `harmonized_cell_day_frp.parquet` | 97,791,096 | 1,102.1 MB | **New.** 0.1° cell × local date FRP: native and S-NPP equivalent (`frp_sum`, `frp_snpp_equiv_sum`, `frp_snpp_equiv_max`, `n_points`) |
| `harmonized_cell_month.parquet` | 51,565,721 | 575.4 MB | 0.1° cell × month. **v1.2: industrial (static-source) points removed**; removed amounts in `static_removed`, `static_removed_lo`, `static_removed_hi` |
| `harmonized_hotspots_2026.tar` | 45,180,873 | 636.8 MB | Hotspot points for 2026 (`harmonized_hotspots/{reference,other}/year=2026/`) — the only year whose point files changed since v1.1 |
| `harmonized_hotspots_static_flags.tar` | 5,853,179 | 81.2 MB | **New.** `harmonized_hotspots_static_flags/year=YYYY.parquet`: point key → `is_static_source` (2000–2026) |
| `harmonized_residual_cell_month.parquet` | 22,878,611 | 125.0 MB | Cell × month values that cannot be assigned to points (`residual_type`); unchanged rule |
| `models/aoi_occurrence_calibration.json` | — | 2.1 KB | **New.** Its calibration (Platt; isotonic thresholds kept for reference) |
| `models/aoi_occurrence_meta.json` | — | 3.0 KB | **New.** Features, validation, adoption decision |
| `models/aoi_occurrence_xgb.json` | — | 0.3 MB | **New.** FireOutlook Monthly – District model (XGBoost) |
| `models/grid_occurrence_calibration.json` | — | 16.7 KB | **New.** Its calibration (isotonic; Platt parameters kept for reference) |
| `models/grid_occurrence_meta.json` | — | 5.8 KB | **New.** Features, persistent-source rule, validation, adoption decision |
| `models/grid_occurrence_xgb.json` | — | 1.3 MB | **New.** FireOutlook Monthly – Grid model (XGBoost) |
| `models/m2v_meta.json` | — | 18.4 KB | FireBridge model names, metrics, decisions; v1.2 adds `static_exclusion`, `display_name` |
| `models/m2v_q10_q90.json` | — | 60.4 MB | FireBridge Uncertainty Band (cell interval q10–q90), unchanged |
| `models/m2v_xgb.json` | — | 77.1 MB | FireBridge (MODIS→VIIRS, Aqua+Terra), unchanged |
| `models/m2v_xgb_terra.json` | — | 96.6 MB | FireBridge-Terra (MODIS→VIIRS, Terra only, 2000-11 – 2002-06), unchanged |
| `models/recurrence_nextmonth_meta.json` | — | 7.0 KB | **New.** FireOutlook Recurrence method and validation (Argentina model) |
| `models/spread_nowcast_meta.json` | — | 6.5 KB | **New.** FireOutlook Spread features, validation (operations weather delay), caveats |
| `models/spread_nowcast_xgb.json` | — | 8.2 MB | **New.** FireOutlook Spread model v1.2-ops (no weather) |
| `planner_30d_skill.json` | — | 117.2 KB | **New (Planner).** Summary skill for 30-day queries per unit (grid cell, municipality, province, adm1, country, continent, world) and per-unit isotonic calibration (model and baseline) from the 30-day holdout validation |
| `planner_aoi_weights.npz` | — | 3.9 MB | **New (Planner).** aoi_id × 0.5° cell area-share matrix (sparse CSR arrays; no pickle) |
| `planner_count_choice.json` | — | 21.5 KB | **New (Planner).** Count source per region group × horizon (validation choice, holdout ≥ 5% rule, interval ratios) |
| `planner_daily_grid.parquet` | 369,985 | 5.7 MB | **New (Planner).** 0.5° cell × day 1–7 after `as_of`: per-day `p_any`, cumulative `p_cum`, `mu`, baseline, model, skill |
| `planner_day_any.npz` | — | 9.9 MB | **New (Planner).** Share of days with ≥ 1 detection per calendar date (±7-day mean, 2012–2023): `region` (aoi_id × 366) and `cell` (0.5° cell × 366); used for month pieces with probability ≥ 0.98 |
| `planner_day_weights.npz` | — | 28.2 MB | **New (Planner).** Daily climatology weights per 0.5° cell × calendar day (`C` counts, `CA` detection days, ±7-day mean, country/continent fallback) |
| `planner_meta.json` | — | 2.0 KB | **New (Planner).** Names, reference dates, period rule, validation summary, licenses |
| `planner_monthly_grid.parquet` | 634,260 | 5.3 MB | **New (Planner).** 0.5° cell × 1–12 months after 2026-11: `p_any`, `mu`, `base_p`, `base_count_median`, model, skill |
| `planner_region_calibration.json` | — | 0.7 MB | **New (Planner).** Region probability calibration (isotonic per region-size bin × horizon) and count interval ratios |
| `planner_region_counts_sf.parquet` | 143 | 4.3 KB | **New (Planner).** AutoETS / AutoTheta monthly counts 2026-11 – 2027-11 (continents, world, Korea, Argentina) |
| `planner_region_skill.json` | — | 1.6 KB | **New (Planner).** Probability skill per horizon × region group (min of validation and holdout Brier skill) |
| `planner_sgg_choice.json` | — | 0.3 KB | **New (Planner).** Korean municipality 2–3 month model choice |
| `planner_sgg_monthly.parquet` | 504 | 7.9 KB | **New (Planner).** Korean municipality × 2 and 3 months ahead (direct model) `p_any`, `baseline_p` |
| `recurrence_nextmonth_latest.parquet` | 28,808 | 0.3 MB | **New.** Next month (2026-12): Argentina probability (`layer = probability`), Korea recurrence count (`layer = recurrence_count`); static-source cells removed |
| `recurrent_burn_sites.parquet` | 6,410 | 68.4 KB | **New.** Cell × calendar month active in ≥ 70% of valid years 2001–2025; static-source cells removed |
| `satellite_day_status.parquet` | 28,316 | 0.8 MB | **New in release.** Satellite × UTC day data status (`ok`, `low`, `missing`, `partial`) |
| `satellite_month_status.parquet` | 934 | 10.2 KB | **New in release.** Satellite × month status |
| `spread_nowcast_latest.parquet` | 10,189 | 0.1 MB | **New.** Next-day spread likelihood, 0.025° candidate cells, Korea box + Argentina (model v1.2-ops) |

Reports (CSV, PNG): `frp_snpp_ratios.csv`, `grid_occurrence_calibration.png`, `grid_occurrence_metrics.csv`, `grid_occurrence_persistence_by_country.csv`, `grid_occurrence_persistence_check.csv`, `hotspots_verify_aoi_month.csv`, `hotspots_verify_cell_month.csv`, `occurrence_base_rate_extremes.csv`, `occurrence_calibration.png`, `occurrence_industrial_flags.csv`, `occurrence_metrics.csv`, `occurrence_metrics_by_year.csv`, `spread_ops_importance.csv`, `spread_ops_metrics.csv`, `static_exclusion_before_after_kr_examples.csv`, `static_exclusion_before_after_totals.csv`, `static_exclusion_months.csv`, `static_exclusion_verdict_changes_KR_national.csv`, `static_exclusion_verdict_changes_kr.csv`, `static_source_impact_aoi.csv`, `static_source_impact_kr_sites.csv`, `static_source_impact_monthly_points.csv`, `static_source_impact_points.csv`; FireOutlook Planner: `planner_count_routing.csv`, `planner_daily_count_baseline.csv`, `planner_daily_metrics.csv`, `planner_dayweights_validation.csv`, `planner_grid_h1_vs_grid_occurrence.csv`, `planner_monthly_metrics.csv`, `planner_region_count_compare.csv`, `planner_region_metrics.csv`, `planner_sgg_compare.csv`, `planner_30d_validation.csv`, `planner_30d_calibration.csv`, `planner_30d_calibration_after.csv`, `planner_30d_latency.csv`, `planner_30d_fire_days_check.csv`.
`static_source_impact_*` were measured before the fix (v1.1 data); `static_exclusion_*` are before/after the fix.

**Hotspot points.** The point files of 2000–2025 are identical to the v1.1 tars (checked year by year: expected tar size from the current files equals
the v1.1 asset size, and no partition of those years was modified after the v1.1 tars were made). Keep using
`data-v1.1-hotspots-2026-11-14` for 2000–2025 and take `harmonized_hotspots_2026.tar` from this release. Point files still contain the
static-source points: **remove them with an anti-join on `harmonized_hotspots_static_flags`** (key: `satellite`, `latitude`, `longitude`,
`acq_datetime_utc`) so that point sums match the cell values.

## Known limitations

- **Industrial sources before 2012-02 are only partly removed.** In the MODIS period, the part of a cell estimate that cannot be assigned to points
  (`modis_unseen` residual and background predictions) cannot be split by point, so some static contribution can remain (Korea: 7% removed before
  2012-02 vs 23% after). Some 2001–2011 Korean months moved to `above` partly because the baseline (dominated by later years) dropped.
- **Models were trained on data that still included industrial sources** (FireBridge, FireOutlook Recurrence). They were not retrained.
  FireOutlook Spread was retrained on the static-source-free sample (v1.2-ops).
- The static-source list only contains cells where SP `type` 1 or 2 appeared in at least two years; new plants or gas flares are not caught.
- **FireOutlook Spread radius:** it only scores cells within about 7.5 km (3 cells of 0.025°) of today's detections; new ignitions and spread beyond
  that are not covered (share of tomorrow's detections inside the candidate cells: Korea 25%, Argentina 42%). Not a physical fire-spread simulation.
  It uses no weather (weather 4 days old did not help); a 5-cell radius raised the coverage to 31% / 53% but was not adopted.
- **FWI is about 4 days behind:** NASA POWER daily data currently end on 2026-11-11. Show the reference date with the badge. FWI uses daily
  statistics instead of noon observations (see `fwi_meta.json`).
- Spatial generalization of FireBridge (from v1.1): reliable for regions in the training data; do not apply it to a new continent as is.
- FRP S-NPP equivalents: Terra night ratio is low-confidence and 2000–2011 uses post-2012 ratios (see "FRP (S-NPP equivalent)").
- **FireOutlook Planner:** for large regions `p_any` is close to 1 and carries little information (show the expected count instead,
  `display_hint`); daily forecasts were validated on grid cells only, so multi-cell regions are capped at `skill = low`; long periods treat month
  pieces as independent (slightly high); Korean municipalities beyond 3 months use grid aggregation validated against `aoi_month`, not the
  point-based label of `/occurrence`; AutoETS/AutoTheta were run only for continents, world, Korea and Argentina (not every country).

## SHA-256

```
97f3adc0ef8f3711cbb94f270c50cf51346f87879ce739867d2bfe3e23719dac  aoi_boundaries.geojson
f6dd727a7e36bdc86b6e543bd346c840f0c31eabcd0ad39ed3e95358abb30a02  aoi_list.parquet
d0220c39bdc2075e067d255f31ed36002fe18617d56f03e23bfa08bed200031f  aoi_month.parquet
4c73677d8706417f6d198d8e6c0777cf04be1b38fe65b06fd88502750cb37431  aoi_month_compare.parquet
b3b21a178879f2fdd222c983f632e464243bc9fd2890bc2fc90e15f9e37d5ead  aoi_month_fire_regime.parquet
c8953e5049c46321b9d82ed61218c34cf89cea9daddc32d80ce87b4e97d7b2f6  aoi_occurrence_nextmonth_latest.parquet
d3041069967186d15e31397cf24b81bd96a6967dfc890b5af239f624e8bb172f  fire_events.parquet
cd8ee1e2f05299d7e4bdc129fa4dddd11fe5fa4868dca86c80130dd8b06da406  fire_events_meta.json
bcd2c6cd4a81ae186d8b1dd1f833c6bafac4cb9f0b89ca3a1188e016f2ba6bc4  fwi_aoi_day.parquet
fdbf77528419a7d9e603acabd2c5ba2a720271ca78352c3eb5e565a30211f97c  fwi_clim_aoi_month.parquet
98b2373c46fb92a9d3244ed6fc89406691647087ec702751614f14e28472edcd  fwi_meta.json
de1e57583232a7f0da1f60acd8a33bfd571b62c5c52084fcdb63fa4a7405d8ab  grid_occurrence_nextmonth_latest.parquet
413c5dc59bf3e4150f05a9815f708b7e1162ab010d2e4d9aad21645a92711af3  harmonized_cell_day.parquet
4726e000f2425b4e64ef6024e620b092ab24df30643036d9c3c595f226dc28e1  harmonized_cell_day_frp.parquet
ad55f3dce2b486f2ec5352fe29abcac82f942b4b7e1800f8357e8c8b0cf0645a  harmonized_cell_month.parquet
d7773caf5f297bcc634cd8009a85ba5955c9977b141c52b6393d38368d091573  harmonized_hotspots_2026.tar
ed8386d18d627893642bbe208d8d057b337311c1fdb2cbdc4a63d5d2c772255f  harmonized_hotspots_static_flags.tar
1d18880056b7731b082131baeb068d6b21631746c42d1e4290d65559997a32c0  harmonized_residual_cell_month.parquet
1c2e93d450050f17c17cf15786aa72422b4417dca6d1a766f3cac9b13479c187  models/aoi_occurrence_calibration.json
c6148673ad6513679e8f5df98fca2e78d015767810d90d05234f83373cfc58f1  models/aoi_occurrence_meta.json
285fe4a9ef5d52aba109618ab81b27cd2d7945d6523151b8ee2db2aa1e3bc9cb  models/aoi_occurrence_xgb.json
9e6ff10ef01570b4e92f61411f0c82b22fef41f6efb5730e4c884cfc79e7be5b  models/grid_occurrence_calibration.json
b9d492d55c8880fc05719894d6f4b076a43ec2cf3c23d1deabd637e1521c305c  models/grid_occurrence_meta.json
203312a9c76403aac739919a5e81c331c5a7610a5f3dc93f790d96b2604c57a0  models/grid_occurrence_xgb.json
290e4e1273645cf9b037e83e20f57ea7f58989bc36792998501ab0d16a2d749b  models/m2v_meta.json
ff5affb08951b47e786b9626c5e56816a9854a9b523f546fba66008309acd519  models/m2v_q10_q90.json
51f211b9a161b74ab84343f09e6d6e68eac460c391cdedd3f2f71b6c35d873ee  models/m2v_xgb.json
bf682b1f5a159096764fd0dcc3e9e71ced9a3fe52592f5fbd416025b03dc53b2  models/m2v_xgb_terra.json
9aed586aefb280f9a893e0a7e2a43f0b6c7995aeb043bbe4bbdd93c78fafd4a5  models/recurrence_nextmonth_meta.json
c0ae73671ace873ce79424c8dd33ca99c638e4e68a0e36fdb3e4f189dfc29b20  models/spread_nowcast_meta.json
2f2b965a674e0a7d95cc9c828aa34d851cf50813640550154889cb600ca93fbd  models/spread_nowcast_xgb.json
f71d36253fbebe42aae76138a4ded964f334c80b1d9a9fb33890cd892b74b2fb  planner_30d_skill.json
254398f5350949478866cda1dbf5dea8d82fec3094c0384f1469b2885594742e  planner_aoi_weights.npz
7a0da69371450493c3931bd64b1d98e764289b45227c8b6880d7e5179dc42aaf  planner_count_choice.json
b81223f2dafda9f0d511565be4e25af6e87ba0ffc341eb8ac306582d00e930e7  planner_daily_grid.parquet
c04fb8194bac9ce3bc4f89396379e474a5858beebadf77bd3df989c1f5c29c4e  planner_day_any.npz
b1b53848b0a4680d69d1f3b70c7cd602f4f7e8f861ec5aef7f8d4b48e66df585  planner_day_weights.npz
0dca41ce2549a4e21fde982cd162276bfc7f63a2cf4117da216d393983f4156e  planner_meta.json
9dbb106712db1f4fb4e5ec4a219db0f2b3d4c1b4efb871ee8ff48040fa9a8f54  planner_monthly_grid.parquet
9e40deb39e896d49df1939e67be5e9769220c88fa7ac54fca82aa36b44366629  planner_region_calibration.json
916111b9cbdf0048458ba9e5d6ec0fa478d1b2030ae19cae291d554a0cbacf9d  planner_region_counts_sf.parquet
63eddbbdb98e2554ca28bb0a11b3992bb4f59cbec7e799b4de97a3c33f0be2d4  planner_region_skill.json
d0f1b14ab32516ee7b0ae31677dce37e40d419b2a6142d1488ae287047a7111a  planner_sgg_choice.json
0b04c5261618b8c90e3da11f97bccd62f13723153c1c4c89dcfb0754d00d26e8  planner_sgg_monthly.parquet
c654290c9d393399d033c403d4c28fa7f317878b9075a16443d5c935ee6c6009  recurrence_nextmonth_latest.parquet
f58c7643164d4d8d176e30751eebfac02e329ff15f566d7992a4250a262b0719  recurrent_burn_sites.parquet
c5a21d19a189789e1bb15d51d14a9e09b34020d927448fac0691cb5776d6f517  reports/frp_snpp_ratios.csv
0ab4da12eae3639ba52f8bd5eb2f56801c26e088c747a18b8098782ccdeee79b  reports/grid_occurrence_calibration.png
1f6b0347b9d785c13f237bf02bb3c243f90a4fdaed78c3398650c6612826a611  reports/grid_occurrence_metrics.csv
68fb0288baa4192f08da0e59c9f08e41583c8fcb02ae9bc1d4a81e9420965e33  reports/grid_occurrence_persistence_by_country.csv
780905f8f220df0fc19260fea91376ede4c34472c814abb5f1bad0894720e1cd  reports/grid_occurrence_persistence_check.csv
c6b19586cc12fe09fd2c6c166e803499e1d518b64e6066cc44d15227f38b7709  reports/hotspots_verify_aoi_month.csv
c4051d2767d0758d3748f13867f2b962646f147d4e0775ed87df17573b4ce590  reports/hotspots_verify_cell_month.csv
cdbb678aab2841aa89bc7342902aa587ab23ee6a15c5ce0a0685f0d33e631bb1  reports/occurrence_base_rate_extremes.csv
139ff332c92b4b407f7d2c8a425d72d68d090df7e8095d0ec0376774a3021637  reports/occurrence_calibration.png
28972287918782f82959e84e248e4ac77a96716bab01e4d41e4176cc223b4eaa  reports/occurrence_industrial_flags.csv
0deb7f8c02abf0aef644feb11f212bc0b378f02397018ba163fdb24953d6dd92  reports/occurrence_metrics.csv
61eda9c4e8ea8fd3428575a5f4afbafe82b5fca983d27f2ea6d35f10714be74c  reports/occurrence_metrics_by_year.csv
3906f764754a2a742a66e3ac67782d2d0246d2a6d9edf96d778bd89136a2798d  reports/planner_30d_calibration.csv
095780d2866033a6c8922ff7369f1202d90a038c753c739cf387caf877cefefa  reports/planner_30d_calibration_after.csv
8482616fcfcbe680ea4b9e50c49621c5eee316c6ac1e48f6f003ac0e69dbb864  reports/planner_30d_fire_days_check.csv
0d2d33af83982cb53aa8587b51b6cd6765312a8bd616f6a95490620918a88b2f  reports/planner_30d_latency.csv
a7232c1bfb491bdfc8df26d2f5bebf1751ffe757ffe37c7a890facf31b4bfeb2  reports/planner_30d_validation.csv
020648580257ce126194d4c50d912fcf380f1846ce599b9b217139ca42975982  reports/planner_count_routing.csv
0ea560359ea2460f128cd79e7bb88a1d3f0183f003c05c2fdc3037d36e533dc5  reports/planner_daily_count_baseline.csv
5547eb723e703c3cd89c6746735846dc7b867a9a255f19202e56ab537c575731  reports/planner_daily_metrics.csv
bdf26cbd346ba5e19e1c2639d8f3aa56a44bbee08332be81bdafa45356478265  reports/planner_dayweights_validation.csv
a42f469a888a7524d8c90583e85036c9e1aac01fd5b8ded71a94015290e3ae03  reports/planner_grid_h1_vs_grid_occurrence.csv
27068c7c9899b0f98e84acc59330909489573109c6fb6021ce99d61e2feb82ce  reports/planner_monthly_metrics.csv
482020b98f310facab81aeb0a0211afc4233a6c4654e21df6f0cf8d156e4cef9  reports/planner_region_count_compare.csv
c4d6a5cbe38a771e3b041f55e6e4431a1679e8bfb6f0c36b3788c119bf1eca82  reports/planner_region_metrics.csv
d4680217d56d485522ed2112c889e87b5fcd480a7e79104b9172fa11b240cf53  reports/planner_sgg_compare.csv
68f33cec8d6ce472f683c0dbf27342812d6679daf11ee8a16714b43305513df2  reports/spread_ops_importance.csv
4041874e0a32509d40f7f188df2aaf0e3ca1d2c4169e59b22adb7670d4b3168b  reports/spread_ops_metrics.csv
ec7a757e62061731509e25b955babc9255d2d28420e364efa2854c4aae496b10  reports/static_exclusion_before_after_kr_examples.csv
7b50c31d34a72839065971d8b31729b8a850c8849b120d52e2ad7ca6d7637990  reports/static_exclusion_before_after_totals.csv
5524a3522a10f808bc00e20a7a677428d172e9c689b5e250d052445b0c7e3d1f  reports/static_exclusion_months.csv
bc1d212aac452a197141a7ddc21303ed9c8f8d95f72d7ae6a1ddc9cc9a2cd744  reports/static_exclusion_verdict_changes_KR_national.csv
4eb11cad608cd4c01b3283e837ed3c39704bff9c99629c8a173103d283b8fdac  reports/static_exclusion_verdict_changes_kr.csv
4a6ea92bace419be28d2df250409d19b4ec0ac2eba02010e2b320c09602e9d7c  reports/static_source_impact_aoi.csv
61fc69eb8ccaf25b5a34482fe607cb2a7feb3df0730185d193d2a7507f292fa5  reports/static_source_impact_kr_sites.csv
590b9f261468a7b58870a6322f3959c973b50e5b7863ef3f32e0e85b612cc646  reports/static_source_impact_monthly_points.csv
9915fa1e0de4501dc1616b4285de30acf61e1eceb7870370bf361e36f6b344ce  reports/static_source_impact_points.csv
e86ca4c3cffc93e3fa0904b95b758b35c0ef4020a955e0461eadd647010b6d5e  satellite_day_status.parquet
9ce8d68bcbb0127a8a3fd76c2cc6738f590e39ca2b841e338fb126eb34809e36  satellite_month_status.parquet
d4e4657b1b6a30917c6a6aeb1a5c98810f74b5ec2994fa3d7f76a9cc8d8ee84a  spread_nowcast_latest.parquet
```
