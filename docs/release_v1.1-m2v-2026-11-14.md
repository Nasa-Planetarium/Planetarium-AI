# Data release v1.1-m2v-2026-11-14

Harmonized MODIS–VIIRS fire activity, 2000-11 to 2026-11 (VIIRS S-NPP–equivalent detection counts).
The files are in `data/processed/release_v1.1-m2v-2026-11-14/` (to be attached to the GitHub Release `data-v1.1-m2v-2026-11-14`; not stored in git because of size).

**v1.1 vs v1:** the model and `method_version` (`v1-m2v-2026-11-14`) are unchanged. v1.1 changes data rules only:
point weights are sensor conversion only, the missing-day fill is stored as a separate residual, the month in progress reports
month-to-date values plus a projection, and the daily table `harmonized_cell_day` is added.

## Files

| File | Rows | Size | Description |
|---|---|---|---|
| `harmonized_cell_month.parquet` | 51,562,642 | 571 MB | 0.1° cell × month: `value`, `value_lo`/`value_hi`, **`value_projected`**, `raw_modis`, `raw_viirs`, `value_source`, `data_source`, `status`, `method_version` |
| `harmonized_cell_day.parquet` | 212,774,685 | 965 MB | **New.** 0.1° cell × local date: sum of reference hotspot weights (`value`, `value_lo`, `value_hi`, `n_points`, `value_source`) + `residual_month`, `value_projected_month` (see below) |
| `harmonized_residual_cell_month.parquet` | 22,878,611 | 125 MB | **New.** Cell × month values that cannot be assigned to hotspot points, with `residual_type` |
| `aoi_month.parquet` | 1,580,963 | 31 MB | AOI × month totals (5,051 AOIs × 313 months): `value`, `value_lo`/`value_hi`, **`value_projected`**, `status`, **`as_of_date`**, … |
| `aoi_month_compare.parquet` | 1,580,963 | 31 MB | Same-month comparison with previous years (percentile, ratio_to_median, verdict) + `as_of_date`, `value_projected` |
| `aoi_list.parquet` | 5,051 | 0.1 MB | AOI id, name, level, parent, `n_cells`, `hidden` |
| `aoi_boundaries.geojson` | — | 13 MB | AOI boundaries (WGS84) |
| `m2v_meta.json` | — | 10 KB | Model names, coefficients, validation metrics, decisions (`data_release`, `partial_month`, `hotspots`) |

Hotspot points (`harmonized_hotspots/reference|other/year=YYYY/month=MM/`, 325,055,860 reference + 228,772,249 other VIIRS points, ~7.5 GB) are distributed separately.
Background predictions for cells without MODIS detections (`data/interim/harmonized_background/`, 90.2 M rows) are already included in `aoi_month`.

## How `value` is built

| Period | Method | `value_source` |
|---|---|---|
| 2000-11 – 2002-06 | Terra → VIIRS (MODIS-to-VIIRS Harmonizer, Terra-only) | `modis_m2v_terra` |
| 2002-07 – 2012-01 | Aqua+Terra → VIIRS (MODIS-to-VIIRS Harmonizer) | `modis_m2v` |
| 2012-02 – 2026-11-02 | Suomi NPP VIIRS observed | `snpp_observed` |
| 2026-11-03 – | NOAA-20 VIIRS × 1.013 (N20 Bridge) | `n20_bridge` |

Validation (leave-one-year-out, 2012–2025): bias (Σpred/Σobs) Korea 0.986, Argentina 0.986, global 0.992; interval [q10, q90] coverage ≈ 80%.

## New in v1.1

### Month in progress: `value`, `value_projected`, `as_of_date`
- For `status = partial` (currently 2026-11), **`value` is the month-to-date total** of what was actually observed (no scaling), and
  **`value_projected`** is the full-month estimate (observed days scaled to the number of days in the month).
  For complete months, `value_projected = value`.
- **`as_of_date`** is the last observed date of the month (month end for complete months; 2026-11-13 for 2026-11).
- Example, 2026-11 (as of 2026-11-13): Korea `value` 44.3 / `value_projected` 102.3; Argentina 7,146.8 / 16,492.6; world 684,891 / 1,575,941.
- Verdicts for partial months stay `uncertain`.

### Point weights and `residual_type`
- Hotspot weights now convert sensors only: S-NPP 1.0, NOAA-20 × n20_bridge, NOAA-21 × n21_bridge, MODIS × the cell-month M2V ratio.
  Scaling up for missing observation days is no longer put into point weights.
- Values that cannot be assigned to points are kept in `harmonized_residual_cell_month` with `residual_type`:

| `residual_type` | Period | Meaning | Rows | Sum |
|---|---|---|---|---|
| `modis_unseen` | 2000-11 – 2012-01 | M2V estimate for cell-months where MODIS saw no fire (plus background predictions, stored in interim) | 19,948,578 | 26.6 M |
| `missing_day_fill` | 2012-02 onward, complete months | Cell `value` minus the sum of reference point weights: the share added for days without usable satellite data. Can be negative (81,563 rows) when detections fell on days that were excluded from the monthly estimate | 2,930,033 | 4.36 M |

- For every complete month: sum of reference point weights + residual = cell `value` (verified for all cell-months, AOI-months and the daily table; max relative error < 1e-6).
- Months with `missing_day_fill`: 26 S-NPP months with missing days (largest 2024-07, 2022-08, 2026-07, 2022-07). 2026-11 (in progress) has none because its `value` is month-to-date.

### Daily table `harmonized_cell_day`
- One row per cell × local date with reference hotspots: `value` = sum of `harmonized_weight`, `value_lo`/`value_hi`, `n_points`, `value_source`.
- **`residual_month`**: the cell-month residual (all types), stored only on the row for day 1 of the month (null on other days). It is a
  monthly amount, not a day-1 value: add it only when summing whole months; for partial date ranges add it pro rata (days included / days in month).
- **`value_projected_month`**: for the month in progress only, the cell's full-month estimate, on the day-1 row.
- Grouping `value + residual_month` by month and AOI reproduces `aoi_month.value` exactly.

## Known limitation: spatial generalization (added 2026-11-14, after release)

The released values are **not changed**. This note records how far the model can be trusted outside the conditions it was built for.

- **Use condition (what the release relies on):** leave-year-out validation — regions that are in the training data, a year that is not — gives
  monthly-total bias (Σpred / Σtrue) of **0.98–0.99** for every continent, Korea and Argentina. Every country is in the training data of v1.1,
  so the released `modis_m2v` values are in this condition.
- **Outside it:** leave-continent-out validation — a whole continent removed from training — gets the level wrong: monthly totals are
  **0.27–0.76×** the observed values for the six continents, Korea (trained without Asia) is **3.0×**, Argentina (trained without South America) is
  **0.65×**. Correlation stays at 0.75–0.98 (the seasonal shape is followed), and the AOI intervals stop covering the truth (0–24 % of months).
- **Guidance:** for a new region that has no training data (no S-NPP years in that country or continent), do not apply `m2v_xgb` as is;
  use `ratio_country` (country → subregion → latitude band × month ratio), which had bias 0.85–1.34 in the same test.
- Details: `reports/m2v_leave_continent_out.csv`, `reports/figures/m2v_leave_continent_out.png`, `models/m2v_meta.json` → `spatial_generalization`.

## Notes
- `harmonized_cell_month` stores non-zero cells only; a missing row means 0 when the month's `status` is `ok`.
- 2000-11 – 2002-06 (Terra only): cell- and point-level intervals are null; intervals for this period are provided at AOI level only.
- AOIs with `hidden = true` (open ocean "Seven seas" and its small island units, 63) should be left out of lists; their verdict is always `uncertain`.
- `value` is a count of satellite fire detections (observed burning activity), not the number of wildfires or burned area.
- Data sources and licenses: see `docs/references.md` (NASA FIRMS acknowledgement, boundary licenses).
