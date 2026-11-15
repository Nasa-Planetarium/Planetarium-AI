"""Fit the backend's portable LightGBM detection, FRP, and calibration models."""
import argparse
from pathlib import Path

import numpy as np

from . import grid_1km, monthly_cells, year_1km
from .models import MODEL_DIRECTORY, save_bundle


def train_daily(cache, output, threads):
    read = grid_1km.archive(cache)
    detector, intensity = grid_1km.fit(read, 48000, threads)
    validation = grid_1km.evaluate_data(read, grid_1km.VALIDATION_ORIGINS, 130000, detector, intensity, None)
    calibration = grid_1km.fit_calibrators(validation, None)["lightgbm_calibrated"]
    save_bundle(output, "daily", detector, intensity, calibration, "2025-12-21")


def train_annual(cache, output, threads):
    read = monthly_cells.reader(cache)
    detector, intensity = year_1km.train(read, 42000, threads)
    validation = year_1km.evaluate_data(
        read, year_1km.VALID, np.datetime64("2025-01", "M"), np.datetime64("2025-12", "M"),
        65000, detector, intensity, None,
    )
    calibration = year_1km.fit_calibration(validation, None)["lightgbm_calibrated"]
    save_bundle(output, "annual", detector, intensity, calibration, "2024-12-31")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("daily", "annual"))
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", default=MODEL_DIRECTORY, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.kind == "daily":
        train_daily(args.cache, args.output, args.threads)
    else:
        train_annual(args.cache, args.output, args.threads)


if __name__ == "__main__":
    main()
