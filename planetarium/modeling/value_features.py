"""Features available at forecast creation for direct daily leads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetarium.modeling.value_data import ValuePanel

HORIZON = 30
SEASON_LOOKBACK = 1115
CONTEXT = 256
FEATURES = (
    "lead",
    "year_sin",
    "year_cos",
    "lat",
    "lon_sin",
    "lon_cos",
    "log_cells",
    "log_annual_mean",
    "log_last",
    "log_mean3",
    "log_mean7",
    "log_mean14",
    "log_mean30",
    "log_mean90",
    "log_max7",
    "log_max30",
    "active30",
    "active365",
    "coverage30",
    "log_seasonal",
    "log_last_year",
    "log_mean_prev30",
)


@dataclass(frozen=True)
class ForecastBatch:
    x: np.ndarray  # region, forecast lead, feature
    scale: np.ndarray  # region
    seasonal: np.ndarray  # region, forecast lead
    recent: np.ndarray  # region
    context: np.ndarray  # region, historical date


def safe_mean(values: np.ndarray) -> np.ndarray:
    known = np.isfinite(values)
    return np.nansum(values, axis=0) / np.maximum(known.sum(axis=0), 1)


def seasonal_mean(
    history: np.ndarray, horizon: int = HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    width = horizon + 30
    totals = np.zeros((horizon, history.shape[1]), dtype=np.float32)
    numbers = np.zeros_like(totals)
    last_year = None
    for year in range(1, 4):
        start = len(history) - 365 * year - 15
        window = history[max(start, 0) : max(start + width, 0)]
        if len(window) != width:
            continue
        valid = np.isfinite(window)
        values = np.vstack((np.zeros_like(window[0]), np.nancumsum(window, axis=0)))
        count = np.vstack((np.zeros_like(window[0]), np.cumsum(valid, axis=0)))
        sums, counts = values[31:] - values[:-31], count[31:] - count[:-31]
        totals += sums
        numbers += counts
        if year == 1:
            last_year = sums / np.maximum(counts, 1)
    annual = safe_mean(history[-365:])
    mean = np.divide(
        totals,
        numbers,
        out=np.broadcast_to(annual, totals.shape).copy(),
        where=numbers > 0,
    )
    return mean, mean if last_year is None else last_year


def make_batch(
    panel: ValuePanel, origin: int, *, horizon: int = HORIZON
) -> ForecastBatch:
    if origin < CONTEXT - 1:
        raise ValueError(
            f"At least {CONTEXT} calendar days are required before forecasting."
        )
    history = np.asarray(
        panel.values[max(0, origin + 1 - SEASON_LOOKBACK) : origin + 1]
    )
    seasonal, last_year = seasonal_mean(history, horizon)
    annual = safe_mean(history[-365:])
    scale = (1 + np.maximum(annual, 0)).astype(np.float32)
    recent = safe_mean(history[-30:]).astype(np.float32)
    leads = np.arange(1, horizon + 1)
    by_day = panel.dates[origin] + leads.astype("timedelta64[D]")
    year = by_day.astype("datetime64[Y]").astype("datetime64[D]")
    phase = (by_day - year).astype(int) * (2 * np.pi / 365.25)
    phase_lon = np.deg2rad(panel.regions["lon"].to_numpy())
    active = history > 0
    observed = np.isfinite(history)
    last = np.where(observed[-1], history[-1], safe_mean(history[-3:]))
    log = lambda value: np.log1p(np.maximum(value, 0))
    scaled_log = lambda value: log(value / scale)
    variables = [
        leads[:, None],
        np.sin(phase)[:, None],
        np.cos(phase)[:, None],
        panel.regions["lat"].to_numpy() / 90,
        np.sin(phase_lon),
        np.cos(phase_lon),
        log(panel.regions["n_cells"].to_numpy()),
        log(annual),
        scaled_log(last),
        *[scaled_log(safe_mean(history[-n:])) for n in (3, 7, 14, 30, 90)],
        *[
            scaled_log(np.where(observed[-n:], history[-n:], 0).max(axis=0))
            for n in (7, 30)
        ],
        active[-30:].sum(axis=0) / np.maximum(observed[-30:].sum(axis=0), 1),
        active[-365:].sum(axis=0) / np.maximum(observed[-365:].sum(axis=0), 1),
        observed[-30:].mean(axis=0),
        scaled_log(seasonal),
        scaled_log(last_year),
        scaled_log(safe_mean(history[-60:-30])),
    ]
    x = np.empty((len(scale), horizon, len(FEATURES)), dtype=np.float32)
    for j, values in enumerate(variables):
        x[..., j] = np.broadcast_to(values, (horizon, len(scale))).T
    context = np.asarray(history[-CONTEXT:].T, dtype=np.float32)
    context = np.where(
        np.isfinite(context), context, safe_mean(context.T)[:, None]
    ).astype(np.float32)
    return ForecastBatch(x, scale, seasonal.T.astype(np.float32), recent, context)
