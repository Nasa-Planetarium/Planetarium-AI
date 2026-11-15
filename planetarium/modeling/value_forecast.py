"""30 daily value predictions for a point, bounds, or a union of bounds."""

from __future__ import annotations

from datetime import date
from math import ceil, floor
from pathlib import Path

import numpy as np
import polars as pl

from planetarium.config import PROCESSED_DATA_DIR
from planetarium.modeling.value_data import (
    HEIGHT,
    WIDTH,
    WORK,
    ValuePanel,
    fill_matrix,
    mask_unobserved,
    normal_days,
    scan_values,
)
from planetarium.modeling.value_features import (
    FEATURES,
    HORIZON,
    SEASON_LOOKBACK,
    make_batch,
)
from planetarium.modeling.value_models import MODELS

DEFAULT_MODEL = "lightgbm_l1"
Bounds = tuple[int, int, int, int]


def _bounds(bbox) -> list[Bounds]:
    west, south, east, north = bbox
    if not (
        -180 <= west <= 180
        and -180 <= east <= 180
        and -90 <= south < north <= 90
        and west != east
    ):
        raise ValueError(
            "Bounds require positive area within (west, south, east, north)."
        )

    def first(value, offset):
        return ceil((value + offset) * 10 - 0.5 - 1e-9)

    def stop(value, offset):
        return floor((value + offset) * 10 - 0.5 + 1e-9) + 1

    low, high = first(south, 90), stop(north, 90)
    intervals = [(west, east)] if west < east else [(west, 180), (-180, east)]
    return [
        (first(left, 180), low, stop(right, 180), high)
        for left, right in intervals
        if first(left, 180) < stop(right, 180) and low < high
    ]


def region_boxes(region: dict) -> list[Bounds]:
    kinds = set(region) & {"point", "bbox", "bboxes"}
    if len(kinds) != 1:
        raise ValueError("Each region needs one of point, bbox or bboxes.")
    if "point" in kinds:
        lon, lat = region["point"]
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Point is outside the geographic coordinate range.")
        x = floor((lon + 180) * 10 + 1e-9) % WIDTH
        y = min(floor((lat + 90) * 10 + 1e-9), HEIGHT - 1)
        return [(x, y, x + 1, y + 1)]
    requested = [region["bbox"]] if "bbox" in kinds else region["bboxes"]
    boxes = [box for bbox in requested for box in _bounds(bbox)]
    if not boxes:
        raise ValueError(f"Region {region['id']!r} contains no 0.1° cell centres.")
    return boxes


def region_geometry(boxes: list[Bounds]) -> dict:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for west, south, east, north in boxes:
        mask[south:north, west:east] = True
    rows, cols = mask.sum(axis=1), mask.sum(axis=0)
    count = int(cols.sum())
    lons = np.deg2rad((np.arange(WIDTH) + 0.5) / 10 - 180)
    longitude = np.rad2deg(np.arctan2(cols @ np.sin(lons), cols @ np.cos(lons)))
    return {
        "sides": int(cols[WIDTH // 2 :].any()) + 2 * int(cols[: WIDTH // 2].any()),
        "lat": (rows @ (np.arange(HEIGHT) + 0.5)) / count / 10 - 90,
        "lon": longitude,
        "n_cells": count,
    }


def _filter(boxes: list[Bounds]) -> pl.Expr:
    col, row = pl.col("cell_id") % WIDTH, pl.col("cell_id") // WIDTH
    return pl.any_horizontal(
        col.is_between(west, east - 1) & row.is_between(south, north - 1)
        for west, south, east, north in boxes
    )


def location_history(regions: list[dict], root: Path, as_of=None) -> ValuePanel:
    if (
        not isinstance(regions, list)
        or not regions
        or any(not isinstance(r, dict) for r in regions)
    ):
        raise ValueError("Provide a nonempty list of region objects.")
    names = [r.get("id") for r in regions]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Region IDs must be nonempty unique strings.")
    boxes = [region_boxes(region) for region in regions]
    geometry = pl.DataFrame(
        [
            dict(region_id=name, **region_geometry(bounds))
            for name, bounds in zip(names, boxes, strict=True)
        ]
    )
    dates, coverage = normal_days(root)
    sides = geometry["sides"].to_numpy()
    required = np.bitwise_or.reduce(sides)
    known = (coverage & required) == required
    if as_of is not None:
        cap = np.datetime64(
            date.fromisoformat(as_of) if isinstance(as_of, str) else as_of, "D"
        )
        known &= dates <= cap
    if not known.any():
        raise ValueError(
            "There is no usable reference coverage at or before the requested date."
        )
    end = int(np.flatnonzero(known)[-1]) + 1
    start = max(0, end - SEASON_LOOKBACK)
    dates, coverage = dates[start:end], coverage[start:end]
    scan = scan_values(root, dates[0].astype(date), dates[-1].astype(date))
    frames = [
        scan.filter(_filter(bounds))
        .group_by("date_local")
        .agg(pl.col("value").cast(pl.Float64).sum())
        .with_columns(pl.lit(i, dtype=pl.UInt32).alias("ix"))
        for i, bounds in enumerate(boxes)
    ]
    totals = pl.concat(frames).collect(engine="streaming")
    values = fill_matrix(totals, dates, sides)
    mask_unobserved(values, coverage, sides)
    return ValuePanel(dates, coverage, geometry, values)


def forecast(regions, *, as_of=None, model=DEFAULT_MODEL, root=None, model_dir=None):
    """Predict a 30-day daily path from the latest usable local solar date."""
    from joblib import load

    if model not in MODELS:
        raise ValueError(f"Unknown fitted model {model!r}.")
    path = Path(model_dir or WORK / "results") / f"{model}.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Train it with python -m planetarium.modeling.value_benchmark --models {model}."
        )
    artifact = load(path)
    if artifact["model"] != model or artifact["features"] != list(FEATURES):
        raise ValueError(
            "The model artifact is incompatible with the forecast features."
        )
    panel = location_history(regions, Path(root or PROCESSED_DATA_DIR), as_of)
    origin = panel.dates[-1]
    if origin < np.datetime64(artifact["fit_end"], "D"):
        raise ValueError("The forecast origin predates the model's training cutoff.")
    batch = make_batch(panel, len(panel.dates) - 1)
    prediction = artifact["regressor"].predict(batch.x, batch.scale)
    steps = np.arange(1, HORIZON + 1)
    n = len(prediction)
    return pl.DataFrame(
        {
            "region_id": np.repeat(panel.regions["region_id"].to_numpy(), HORIZON),
            "as_of_date": [origin.astype(date)] * (n * HORIZON),
            "date_local": np.tile(origin + steps.astype("timedelta64[D]"), n),
            "step": np.tile(steps, n),
            "value": prediction.reshape(-1),
            "n_cells": np.repeat(panel.regions["n_cells"].to_numpy(), HORIZON),
            "model": [model] * (n * HORIZON),
        }
    ).with_columns(pl.col("date_local").cast(pl.Date))
