"""Area-level occurrence forecasts with one local solar day per step."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

from planetarium.modeling.daily_occurrence_data import WORK, DailyPanel, load_panel

SEASONS = 26
STATE_COUNT = 4
ONE_DAY = np.timedelta64(1, "D")


@dataclass(frozen=True)
class Series:
    events: np.ndarray
    sides: np.ndarray
    lat: np.ndarray
    size: np.ndarray

    def select(self, indices: np.ndarray) -> Series:
        return Series(
            self.events[:, indices],
            self.sides[indices],
            self.lat[indices],
            self.size[indices],
        )


@dataclass(frozen=True)
class Transitions:
    season: np.ndarray
    one: np.ndarray
    two: np.ndarray
    support: np.ndarray
    overall: np.ndarray


def calendar_bin(dates: np.ndarray) -> np.ndarray:
    year = dates.astype("datetime64[Y]")
    width = (
        (year + np.timedelta64(1, "Y")).astype("datetime64[D]")
        - year.astype("datetime64[D]")
    ).astype(int)
    position = (dates - year.astype("datetime64[D]")).astype(int)
    return position * SEASONS // width


def usable(coverage: np.ndarray, sides: np.ndarray) -> np.ndarray:
    return (coverage[:, None] & sides) == sides


def grid_series(panel: DailyPanel) -> Series:
    return Series(
        panel.events,
        panel.sides,
        panel.cells["lat"].to_numpy(),
        np.ones(panel.cells.height, dtype=np.int32),
    )


def block_series(
    panel: DailyPanel,
    span: int,
    *,
    sample_count: int | None = None,
    rng: np.random.Generator | None = None,
) -> Series:
    """Direct area events over nonoverlapping blocks of span by span grid cells."""
    g = panel.cells["g"].to_numpy()
    n_cols = 720 // span
    block = (g // 720 // span) * n_cols + g % 720 // span
    keys, inverse, size = np.unique(block, return_inverse=True, return_counts=True)
    sides = np.zeros(len(keys), dtype=np.uint8)
    np.bitwise_or.at(sides, inverse, panel.sides)
    lat = np.bincount(inverse, weights=panel.cells["lat"].to_numpy()) / size
    selected = (
        np.sort(rng.choice(len(keys), sample_count, replace=False))
        if sample_count is not None and len(keys) > sample_count
        else np.arange(len(keys))
    )
    matrix = np.empty((len(panel.dates), len(selected)), dtype=np.uint8)
    for first in range(0, len(matrix), 32):
        observed = panel.events[first : first + 32]
        dense = np.zeros((len(observed), 360 * 720), dtype=np.uint8)
        dense[:, g] = observed
        active = dense.reshape(len(observed), 360 // span, span, n_cols, span).any(
            axis=(2, 4)
        )
        matrix[first : first + len(observed)] = active.reshape(len(observed), -1)[
            :, keys[selected]
        ]
    return Series(matrix, sides[selected], lat[selected], size[selected])


def history_stats(
    series: Series, dates: np.ndarray, coverage: np.ndarray, stop: int
) -> Transitions:
    """Estimate calendar-conditioned transitions from positions before stop."""
    events = series.events[:stop]
    bins = calendar_bin(dates[:stop])
    count = np.zeros((SEASONS, len(series.sides), STATE_COUNT), dtype=np.float32)
    hits = np.zeros_like(count)
    observed = usable(coverage[:stop], series.sides)
    for season in range(SEASONS):
        t = np.flatnonzero(bins[2:] == season) + 2
        previous = 2 * events[t - 2] + events[t - 1]
        current = events[t] > 0
        known = observed[t] & observed[t - 1] & observed[t - 2]
        for state in range(STATE_COUNT):
            valid = known & (previous == state)
            count[season, :, state] = valid.sum(axis=0)
            hits[season, :, state] = (valid & current).sum(axis=0)
    # A circular three-bin window retains the yearly season without sharp
    # boundary estimates from adjacent fortnight-sized groups.
    count = count + np.roll(count, 1, axis=0) + np.roll(count, -1, axis=0)
    hits = hits + np.roll(hits, 1, axis=0) + np.roll(hits, -1, axis=0)
    sums = count.sum(axis=-1)
    yes = hits.sum(axis=-1)
    overall = (yes.sum(axis=0) + 0.5) / (sums.sum(axis=0) + 1)
    season = (yes + 8 * overall[None, :]) / (sums + 8)
    one = np.empty_like(count)
    for last in (0, 1):
        sample = count[..., last::2].sum(axis=-1)
        positives = hits[..., last::2].sum(axis=-1)
        one[..., last::2] = ((positives + 6 * season) / (sample + 6))[..., None]
    two = (hits + 6 * one) / (count + 6)
    return Transitions(season, one, two, count, overall)


def feature_rows(stats: Transitions, series: Series, bins, grid, states) -> np.ndarray:
    phase = 2 * np.pi * bins / SEASONS
    return np.column_stack(
        (
            stats.season[bins, grid],
            stats.one[bins, grid, states],
            stats.two[bins, grid, states],
            stats.overall[grid],
            np.log1p(stats.support[bins, grid, states]),
            states >> 1,
            states & 1,
            np.sin(phase),
            np.cos(phase),
            series.lat[grid] / 90,
            np.log1p(series.size[grid]),
        )
    ).astype(np.float32)


def transition_table(
    name: str, stats: Transitions, series: Series, bin_index: int, predictor=None
) -> np.ndarray:
    if name == "seasonal":
        return np.repeat(stats.season[bin_index, :, None], STATE_COUNT, axis=-1)
    if name == "markov1":
        return stats.one[bin_index]
    if name == "markov2":
        return stats.two[bin_index]
    if predictor is None:
        raise ValueError(f"Model {name!r} requires a fitted transition predictor.")
    region = np.repeat(np.arange(len(series.size)), STATE_COUNT)
    states = np.tile(np.arange(STATE_COUNT), len(series.size))
    features = feature_rows(
        stats, series, np.full(len(region), bin_index), region, states
    )
    return predictor(features).reshape(-1, STATE_COUNT)


def initial_state(
    series: Series,
    stats: Transitions,
    dates: np.ndarray,
    coverage: np.ndarray,
    origin: int,
) -> np.ndarray:
    older_known = (coverage[origin - 1] & series.sides) == series.sides
    older_probability = np.where(
        older_known,
        series.events[origin - 1],
        stats.season[int(calendar_bin(dates[origin - 1 : origin])[0])],
    )
    last = series.events[origin]
    state = np.zeros((len(last), STATE_COUNT), dtype=np.float64)
    rows = np.arange(len(last))
    state[rows, last] = 1 - older_probability
    state[rows, last + 2] = older_probability
    return state


def roll(state: np.ndarray, bins: np.ndarray, table) -> np.ndarray:
    """Propagate the full probability distribution of the last two events."""
    probabilities = np.empty((len(bins), len(state)), dtype=np.float32)
    for i, b in enumerate(bins):
        q = table(int(b))
        stays = state * (1 - q)
        events = state * q
        state = np.column_stack(
            (
                stays[:, 0] + stays[:, 2],
                events[:, 0] + events[:, 2],
                stays[:, 1] + stays[:, 3],
                events[:, 1] + events[:, 3],
            )
        )
        probabilities[i] = state[:, 1] + state[:, 3]
    return probabilities


def model_predictor(name: str, path: Path):
    config = json.loads((path / f"{name}.json").read_text(encoding="utf-8"))
    if name == "logistic":
        center = np.asarray(config["mean"])
        scale = np.asarray(config["scale"])
        coefficients = np.asarray(config["coef"])
        bias = config["bias"]

        def raw(x):
            return ((x - center) / scale) @ coefficients + bias
    elif name == "xgboost":
        from xgboost import Booster

        estimator = Booster(params={"nthread": 4})
        estimator.load_model(path / "xgboost.ubj")

        def raw(x):
            return estimator.inplace_predict(x, predict_type="margin")
    else:
        raise ValueError(f"Unknown fitted model {name!r}.")
    return lambda x: expit(
        config["cal_slope"] * raw(x) + config["cal_bias"]
    ), np.datetime64(config["cal_until"], "D")


def _region_mask(region: dict, cells: pl.DataFrame) -> np.ndarray:
    lon, lat = cells["lon"].to_numpy(), cells["lat"].to_numpy()
    shapes = set(region) & {"point", "bbox", "bboxes"}
    if len(shapes) != 1:
        raise ValueError("Each region needs exactly one of point, bbox or bboxes.")
    if "point" in shapes:
        x, y = region["point"]
        if not (-180 <= x <= 180 and -90 <= y < 90):
            raise ValueError("Point is outside the geographic coordinate range.")
        x = -180 if x == 180 else x
        g = int((y + 90) * 2) * 720 + int((x + 180) * 2)
        return cells["g"].to_numpy() == g
    bboxes = [region["bbox"]] if "bbox" in shapes else region["bboxes"]
    included = np.zeros(len(cells), dtype=bool)
    for w, s, e, n in bboxes:
        if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s < n <= 90) or w == e:
            raise ValueError(
                "Bounds must be (west, south, east, north) with positive area."
            )
        x = (
            ((lon + 0.25 > w) & (lon - 0.25 < e))
            if w < e
            else ((lon + 0.25 > w) | (lon - 0.25 < e))
        )
        included |= x & (lat + 0.25 > s) & (lat - 0.25 < n)
    return included


def _date(value: str | date) -> np.datetime64:
    parsed = value if isinstance(value, date) else date.fromisoformat(value)
    return np.datetime64(parsed, "D")


def forecast(
    regions: list[dict],
    start: str | date,
    end: str | date,
    *,
    as_of: str | date | None = None,
    model: str = "auto",
    root: Path | None = None,
    work: Path | None = None,
) -> pl.DataFrame:
    """Return one probability per requested region per local solar date."""
    start_date, end_date = _date(start), _date(end)
    if start_date > end_date:
        raise ValueError("Start must be on or before end.")
    anchor = _date(as_of) if as_of is not None else start_date - ONE_DAY
    if anchor >= start_date:
        raise ValueError("The as-of date must precede the output period.")
    if (
        not isinstance(regions, list)
        or not regions
        or any(not isinstance(region, dict) for region in regions)
    ):
        raise ValueError("Provide a nonempty list of region objects.")
    ids = [r.get("id") for r in regions]
    if any(not isinstance(key, str) or not key.strip() for key in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("Region IDs must be nonempty unique strings.")

    base = Path(work or WORK)
    panel = load_panel(root, base)
    mask = [_region_mask(r, panel.cells) for r in regions]
    for key, include in zip(ids, mask, strict=True):
        if not include.any():
            raise ValueError(f"Region {key!r} has no intersecting land cells.")
    sides_by_cell = panel.sides
    sides = np.asarray(
        [np.bitwise_or.reduce(sides_by_cell[m]) for m in mask], dtype=np.uint8
    )
    needed = np.bitwise_or.reduce(sides)
    good = np.flatnonzero(
        (panel.dates <= anchor) & ((panel.coverage & needed) == needed)
    )
    if not len(good) or good[-1] < 2:
        raise ValueError("No usable observation history at or before the as-of date.")
    origin = int(good[-1])
    origin_date = panel.dates[origin]
    # A union is modeled from the area's actual any-detection history. A
    # detection in overlapping boxes is represented by the same grid cell.
    series = Series(
        np.column_stack(
            [panel.events[: origin + 1, m].any(axis=1) for m in mask]
        ).astype(np.uint8),
        sides,
        np.asarray([panel.cells["lat"].to_numpy()[m].mean() for m in mask]),
        np.asarray([m.sum() for m in mask]),
    )
    year_start = origin_date.astype("datetime64[Y]").astype("datetime64[D]")
    before_year = int(np.searchsorted(panel.dates, year_start))
    stats = history_stats(
        series,
        panel.dates,
        panel.coverage,
        before_year if before_year >= 3 else origin + 1,
    )
    pred = None
    directory = base / panel.fingerprint / "models"
    if model == "auto":
        choice = directory / "selected.json"
        if choice.is_file():
            selected = json.loads(choice.read_text(encoding="utf-8"))
            model = (
                selected["model"]
                if origin_date >= np.datetime64(selected["available_from"], "D")
                else "markov2"
            )
        else:
            model = "markov2"
    if model in {"logistic", "xgboost"}:
        pred, available_from = model_predictor(model, directory)
        if origin_date < available_from:
            raise ValueError(
                f"Model {model!r} was fitted using data after the requested origin."
            )
    elif model not in {"seasonal", "markov1", "markov2"}:
        raise ValueError(f"Unknown model {model!r}.")

    future = np.arange(origin_date + ONE_DAY, end_date + ONE_DAY, dtype="datetime64[D]")
    table = {}
    for b in np.unique(calendar_bin(future)):
        table[b] = transition_table(model, stats, series, int(b), pred)
    state = initial_state(series, stats, panel.dates, panel.coverage, origin)
    path = roll(state, calendar_bin(future), table.__getitem__)
    keep = future >= start_date
    dates = future[keep]
    probs = path[keep]
    return pl.DataFrame(
        {
            "region_id": np.repeat(ids, len(dates)),
            "as_of_date": np.repeat(origin_date, len(dates) * len(ids)),
            "date_local": np.tile(dates, len(ids)),
            "step": np.tile((dates - origin_date).astype(int), len(ids)),
            "p_any": probs.T.ravel(),
            "p_none": (1 - probs).T.ravel(),
            "n_cells": np.repeat(series.size, len(dates)),
            "model": np.repeat(model, len(dates) * len(ids)),
        }
    )
