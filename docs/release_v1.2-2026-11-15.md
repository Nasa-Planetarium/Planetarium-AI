# Data release v1.2-2026-11-15

Harmonized MODIS–VIIRS fire activity, 2000-11 to 2026-11 (VIIRS S-NPP–equivalent detection counts), plus new fire-information layers
for Korea and Argentina. Files: `data/processed/release_v1.2-2026-11-15/` (GitHub Release `data-v1.2-2026-11-15`; not in git because of size).

- Data as of: last NRT append 2026-11-15 10:26 KST (detections up to 2026-11-14 22:07 UTC). 2026-11 is in progress (`status = partial`).
- The MODIS→VIIRS conversion model **FireBridge (MODIS-to-VIIRS Harmonizer)** (internal id: `m2v_xgb`) and `method_version`
  (`v1-m2v-2026-11-14`) are **unchanged** — FireBridge was not retrained for v1.2 (only the spread nowcast was, see below).
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
- **Spread nowcast** (`spread_nowcast_latest`): probability that a 0.025° cell near today's fire has a detection tomorrow (XGBoost).
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
- **Recurrence layer** (`recurrence_nextmonth_latest`, `recurrent_burn_sites`): Argentina = model probability of activity next month
  (expanding-window PR-AUC 0.57 vs 0.49 for the recurrence rate alone, static cells excluded); Korea = count of years active in that month
  ("active in N of the last 25 years", not a probability).

### 3. Next-month occurrence outlook (added to the release on 2026-11-15)
- Probability of at least one detection next month for Korean municipalities and a global 0.5° grid — see the section below.

### 4. FRP (S-NPP equivalent) (added to the release on 2026-11-15)
- Fire radiative power converted to the S-NPP level for every satellite, as a separate daily table — see the section below.

### 5. Other
- `satellite_day_status` and `satellite_month_status` are now part of the release.
- Verification after all changes: points + residual = cell value, AOI sums = `aoi_month`, daily table monthly sums = `aoi_month` — 0 mismatches for
  every year 2000–2026 (`reports/hotspots_verify_*.csv`).

## Next-month occurrence outlook

Two models give the probability that **at least one detection** occurs next month (target 2026-12), after static-source removal.
They describe past patterns and recent activity; they are not weather forecasts.

| | Korea municipalities | Global grid |
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
| `models/aoi_occurrence_xgb.json` | — | 0.3 MB | **New.** Municipality occurrence model (XGBoost) |
| `models/grid_occurrence_calibration.json` | — | 16.7 KB | **New.** Its calibration (isotonic; Platt parameters kept for reference) |
| `models/grid_occurrence_meta.json` | — | 5.8 KB | **New.** Features, persistent-source rule, validation, adoption decision |
| `models/grid_occurrence_xgb.json` | — | 1.3 MB | **New.** Global 0.5° occurrence model (XGBoost) |
| `models/m2v_meta.json` | — | 18.4 KB | FireBridge model names, metrics, decisions; v1.2 adds `static_exclusion`, `display_name` |
| `models/m2v_q10_q90.json` | — | 60.4 MB | FireBridge Uncertainty Band (cell interval q10–q90), unchanged |
| `models/m2v_xgb.json` | — | 77.1 MB | FireBridge (MODIS→VIIRS, Aqua+Terra), unchanged |
| `models/m2v_xgb_terra.json` | — | 96.6 MB | FireBridge-Terra (MODIS→VIIRS, Terra only, 2000-11 – 2002-06), unchanged |
| `models/recurrence_nextmonth_meta.json` | — | 7.0 KB | **New.** Recurrence layer method and validation (Argentina model) |
| `models/spread_nowcast_meta.json` | — | 6.5 KB | **New.** Spread nowcast features, validation (operations weather delay), caveats |
| `models/spread_nowcast_xgb.json` | — | 8.2 MB | **New.** Spread nowcast model v1.2-ops (no weather) |
| `recurrence_nextmonth_latest.parquet` | 28,808 | 0.3 MB | **New.** Next month (2026-12): Argentina probability (`layer = probability`), Korea recurrence count (`layer = recurrence_count`); static-source cells removed |
| `recurrent_burn_sites.parquet` | 6,410 | 68.4 KB | **New.** Cell × calendar month active in ≥ 70% of valid years 2001–2025; static-source cells removed |
| `satellite_day_status.parquet` | 28,316 | 0.8 MB | **New in release.** Satellite × UTC day data status (`ok`, `low`, `missing`, `partial`) |
| `satellite_month_status.parquet` | 934 | 10.2 KB | **New in release.** Satellite × month status |
| `spread_nowcast_latest.parquet` | 10,189 | 0.1 MB | **New.** Next-day spread likelihood, 0.025° candidate cells, Korea box + Argentina (model v1.2-ops) |

Reports (CSV, PNG): `frp_snpp_ratios.csv`, `grid_occurrence_calibration.png`, `grid_occurrence_metrics.csv`, `grid_occurrence_persistence_by_country.csv`, `grid_occurrence_persistence_check.csv`, `hotspots_verify_aoi_month.csv`, `hotspots_verify_cell_month.csv`, `occurrence_base_rate_extremes.csv`, `occurrence_calibration.png`, `occurrence_industrial_flags.csv`, `occurrence_metrics.csv`, `occurrence_metrics_by_year.csv`, `spread_ops_importance.csv`, `spread_ops_metrics.csv`, `static_exclusion_before_after_kr_examples.csv`, `static_exclusion_before_after_totals.csv`, `static_exclusion_months.csv`, `static_exclusion_verdict_changes_KR_national.csv`, `static_exclusion_verdict_changes_kr.csv`, `static_source_impact_aoi.csv`, `static_source_impact_kr_sites.csv`, `static_source_impact_monthly_points.csv`, `static_source_impact_points.csv`.
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
- **Models were trained on data that still included industrial sources** (FireBridge, recurrence model). They were not retrained.
  The spread nowcast was retrained on the static-source-free sample (v1.2-ops).
- The static-source list only contains cells where SP `type` 1 or 2 appeared in at least two years; new plants or gas flares are not caught.
- **Spread nowcast radius:** it only scores cells within about 7.5 km (3 cells of 0.025°) of today's detections; new ignitions and spread beyond
  that are not covered (share of tomorrow's detections inside the candidate cells: Korea 25%, Argentina 42%). Not a physical fire-spread simulation.
  It uses no weather (weather 4 days old did not help); a 5-cell radius raised the coverage to 31% / 53% but was not adopted.
- **FWI is about 4 days behind:** NASA POWER daily data currently end on 2026-11-11. Show the reference date with the badge. FWI uses daily
  statistics instead of noon observations (see `fwi_meta.json`).
- Spatial generalization of FireBridge (from v1.1): reliable for regions in the training data; do not apply it to a new continent as is.
- FRP S-NPP equivalents: Terra night ratio is low-confidence and 2000–2011 uses post-2012 ratios (see "FRP (S-NPP equivalent)").

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
