# Data release v1.2-2026-11-15

Harmonized MODIS–VIIRS fire activity, 2000-11 to 2026-11 (VIIRS S-NPP–equivalent detection counts), plus new fire-information layers
for Korea and Argentina. Files: `data/processed/release_v1.2-2026-11-15/` (GitHub Release `data-v1.2-2026-11-15`; not in git because of size).

- Data as of: last NRT append 2026-11-15 10:26 KST (detections up to 2026-11-14 22:07 UTC). 2026-11 is in progress (`status = partial`).
- Model and `method_version` (`v1-m2v-2026-11-14`) are **unchanged** — no model was retrained for v1.2.
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
- **Spread nowcast** (`spread_nowcast_latest`): probability that a 0.025° cell near today's fire has a detection tomorrow (XGBoost; leave-year-out
  PR-AUC Korea box 0.138, Argentina 0.201). Base date = last NOAA-20 `ok` day (2026-11-13 → target 2026-11-14); weather from NASA POWER with a 2-day lag.
- **Fire events and burned area** (`fire_events`): reference points linked within 750 m (VIIRS) / 2 km (MODIS) and one day; MCD64A1 burned area
  (lower bound; Uiseong 2025-03 ≈ 64% of the official area).
- **Fire-type badge** (`aoi_month_fire_regime`).
- **Fire Weather Index** (`fwi_aoi_day`): absolute class (Canadian thresholds) and relative class (percentile within the same AOI and calendar month,
  2012–2025). App default = relative class; the Canadian absolute class does not fit Argentina well.
- **Recurrence layer** (`recurrence_nextmonth_latest`, `recurrent_burn_sites`): Argentina = model probability of activity next month
  (expanding-window PR-AUC 0.57 vs 0.49 for the recurrence rate alone, static cells excluded); Korea = count of years active in that month
  ("active in N of the last 25 years", not a probability).

### 3. Other
- `satellite_day_status` and `satellite_month_status` are now part of the release.
- Verification after all changes: points + residual = cell value, AOI sums = `aoi_month`, daily table monthly sums = `aoi_month` — 0 mismatches for
  every year 2000–2026 (`reports/hotspots_verify_*.csv`).

## Files

| File | Rows | Size | Description |
|---|---|---|---|
| `aoi_boundaries.geojson` | — | 12.9 MB | AOI boundaries (WGS84) |
| `aoi_list.parquet` | 5,051 | 76.3 KB | AOI id, name, level, parent, `n_cells`, `hidden` |
| `aoi_month.parquet` | 1,580,963 | 30.4 MB | AOI × month totals (5,051 AOIs); recomputed after static-source removal |
| `aoi_month_compare.parquet` | 1,580,963 | 31.0 MB | Same-month comparison with previous years; verdicts recomputed |
| `aoi_month_fire_regime.parquet` | 92,335 | 0.3 MB | **New.** AOI × month fire-type badge (`large_fire`, `many_small`, `mixed`, `none`) |
| `fire_events.parquet` | 1,240,978 | 47.3 MB | **New.** Fire events (space-time linked reference points), Korea + Argentina, satellite-confirmed burned area (lower bound) |
| `fire_events_meta.json` | — | 5.0 KB | **New.** Event rules, burned-area rules, KFS validation |
| `fwi_aoi_day.parquet` | 1,601,555 | 29.1 MB | **New.** Canadian FWI per AOI × day (Korea, Argentina), 2012-01-01 – 2026-11-11, absolute + relative class |
| `fwi_clim_aoi_month.parquet` | 3,540 | 41.9 KB | **New.** AOI × calendar-month FWI quantiles (q25, q50, q75, q90; 2012–2025) for relative classes of new dates |
| `fwi_meta.json` | — | 10.4 KB | **New.** FWI approximations, classes, validation |
| `harmonized_cell_day.parquet` | 212,790,231 | 973.5 MB | 0.1° cell × local date. v1.2: static-source points removed (`static_removed*`, `static_n_points`) |
| `harmonized_cell_month.parquet` | 51,565,721 | 575.4 MB | 0.1° cell × month. **v1.2: industrial (static-source) points removed**; removed amounts in `static_removed`, `static_removed_lo`, `static_removed_hi` |
| `harmonized_hotspots_2026.tar` | 45,180,873 | 636.8 MB | Hotspot points for 2026 (`harmonized_hotspots/{reference,other}/year=2026/`) — the only year whose point files changed since v1.1 |
| `harmonized_hotspots_static_flags.tar` | 5,853,179 | 81.2 MB | **New.** `harmonized_hotspots_static_flags/year=YYYY.parquet`: point key → `is_static_source` (2000–2026) |
| `harmonized_residual_cell_month.parquet` | 22,878,611 | 125.0 MB | Cell × month values that cannot be assigned to points (`residual_type`); unchanged rule |
| `models/m2v_meta.json` | — | 18.4 KB | Model names, metrics, decisions; v1.2 adds `static_exclusion` |
| `models/m2v_q10_q90.json` | — | 60.4 MB | Cell interval model (q10, q90), unchanged |
| `models/m2v_xgb.json` | — | 77.1 MB | MODIS→VIIRS model (Aqua+Terra), unchanged |
| `models/m2v_xgb_terra.json` | — | 96.6 MB | MODIS→VIIRS model (Terra only, 2000-11 – 2002-06), unchanged |
| `models/recurrence_nextmonth_meta.json` | — | 7.0 KB | **New.** Recurrence layer method and validation (Argentina model) |
| `models/spread_nowcast_meta.json` | — | 9.3 KB | **New.** Spread nowcast features, validation, caveats |
| `models/spread_nowcast_xgb.json` | — | 10.0 MB | **New.** Spread nowcast model |
| `recurrence_nextmonth_latest.parquet` | 28,808 | 0.3 MB | **New.** Next month (2026-12): Argentina probability (`layer = probability`), Korea recurrence count (`layer = recurrence_count`); static-source cells removed |
| `recurrent_burn_sites.parquet` | 6,410 | 68.4 KB | **New.** Cell × calendar month active in ≥ 70% of valid years 2001–2025; static-source cells removed |
| `satellite_day_status.parquet` | 28,316 | 0.8 MB | **New in release.** Satellite × UTC day data status (`ok`, `low`, `missing`, `partial`) |
| `satellite_month_status.parquet` | 934 | 10.2 KB | **New in release.** Satellite × month status |
| `spread_nowcast_latest.parquet` | 10,189 | 0.2 MB | **New.** Next-day spread likelihood, 0.025° candidate cells, Korea box + Argentina |

Reports (CSV): `hotspots_verify_aoi_month.csv`, `hotspots_verify_cell_month.csv`, `static_exclusion_before_after_kr_examples.csv`, `static_exclusion_before_after_totals.csv`, `static_exclusion_months.csv`, `static_exclusion_verdict_changes_KR_national.csv`, `static_exclusion_verdict_changes_kr.csv`, `static_source_impact_aoi.csv`, `static_source_impact_kr_sites.csv`, `static_source_impact_monthly_points.csv`, `static_source_impact_points.csv`.
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
- **Models were trained on data that still included industrial sources** (M2V MODIS→VIIRS, spread nowcast, recurrence model). They were not retrained.
- The static-source list only contains cells where SP `type` 1 or 2 appeared in at least two years; new plants or gas flares are not caught.
- **Spread nowcast radius:** it only scores cells within about 7.5 km (3 cells of 0.025°) of today's detections; new ignitions and spread beyond
  that are not covered (share of tomorrow's detections inside the candidate cells: Korea 22%, Argentina 42%). Not a physical fire-spread simulation.
- **FWI is about 4 days behind:** NASA POWER daily data currently end on 2026-11-11. Show the reference date with the badge. FWI uses daily
  statistics instead of noon observations (see `fwi_meta.json`).
- Spatial generalization of M2V (from v1.1): reliable for regions in the training data; do not apply it to a new continent as is.

## SHA-256

```
97f3adc0ef8f3711cbb94f270c50cf51346f87879ce739867d2bfe3e23719dac  aoi_boundaries.geojson
f6dd727a7e36bdc86b6e543bd346c840f0c31eabcd0ad39ed3e95358abb30a02  aoi_list.parquet
d0220c39bdc2075e067d255f31ed36002fe18617d56f03e23bfa08bed200031f  aoi_month.parquet
4c73677d8706417f6d198d8e6c0777cf04be1b38fe65b06fd88502750cb37431  aoi_month_compare.parquet
b3b21a178879f2fdd222c983f632e464243bc9fd2890bc2fc90e15f9e37d5ead  aoi_month_fire_regime.parquet
d3041069967186d15e31397cf24b81bd96a6967dfc890b5af239f624e8bb172f  fire_events.parquet
cd8ee1e2f05299d7e4bdc129fa4dddd11fe5fa4868dca86c80130dd8b06da406  fire_events_meta.json
bcd2c6cd4a81ae186d8b1dd1f833c6bafac4cb9f0b89ca3a1188e016f2ba6bc4  fwi_aoi_day.parquet
fdbf77528419a7d9e603acabd2c5ba2a720271ca78352c3eb5e565a30211f97c  fwi_clim_aoi_month.parquet
98b2373c46fb92a9d3244ed6fc89406691647087ec702751614f14e28472edcd  fwi_meta.json
413c5dc59bf3e4150f05a9815f708b7e1162ab010d2e4d9aad21645a92711af3  harmonized_cell_day.parquet
ad55f3dce2b486f2ec5352fe29abcac82f942b4b7e1800f8357e8c8b0cf0645a  harmonized_cell_month.parquet
d7773caf5f297bcc634cd8009a85ba5955c9977b141c52b6393d38368d091573  harmonized_hotspots_2026.tar
ed8386d18d627893642bbe208d8d057b337311c1fdb2cbdc4a63d5d2c772255f  harmonized_hotspots_static_flags.tar
1d18880056b7731b082131baeb068d6b21631746c42d1e4290d65559997a32c0  harmonized_residual_cell_month.parquet
290e4e1273645cf9b037e83e20f57ea7f58989bc36792998501ab0d16a2d749b  models/m2v_meta.json
ff5affb08951b47e786b9626c5e56816a9854a9b523f546fba66008309acd519  models/m2v_q10_q90.json
51f211b9a161b74ab84343f09e6d6e68eac460c391cdedd3f2f71b6c35d873ee  models/m2v_xgb.json
bf682b1f5a159096764fd0dcc3e9e71ced9a3fe52592f5fbd416025b03dc53b2  models/m2v_xgb_terra.json
9aed586aefb280f9a893e0a7e2a43f0b6c7995aeb043bbe4bbdd93c78fafd4a5  models/recurrence_nextmonth_meta.json
8a2607f241ef6c38bbd590819e2c195de4a238337ffae92d3e6b638b024ebcbc  models/spread_nowcast_meta.json
42f47fd15c04b3274fca112c08e367497686cbf214817f06df2aea541c63428f  models/spread_nowcast_xgb.json
c654290c9d393399d033c403d4c28fa7f317878b9075a16443d5c935ee6c6009  recurrence_nextmonth_latest.parquet
f58c7643164d4d8d176e30751eebfac02e329ff15f566d7992a4250a262b0719  recurrent_burn_sites.parquet
c6b19586cc12fe09fd2c6c166e803499e1d518b64e6066cc44d15227f38b7709  reports/hotspots_verify_aoi_month.csv
c4051d2767d0758d3748f13867f2b962646f147d4e0775ed87df17573b4ce590  reports/hotspots_verify_cell_month.csv
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
714d44d32d23e15e60a4676bd9359244d401eacc811c66a926f55900a6c79121  spread_nowcast_latest.parquet
```
