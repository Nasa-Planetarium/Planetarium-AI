"""Request TimesFM 3.0 forecasts by bounding box and UTC date or month."""
import argparse
from contextlib import redirect_stdout
from datetime import date, timedelta
import json
import os
from pathlib import Path
import sys
from threading import Lock

import numpy as np

from ..api import DATA_ROOT, ROOT, _bbox, _day
from .history import HEIGHT, MONTH, WIDTH, History
from .timesfm_forecast import MODEL_ID, MODEL_REVISION, TimesFM, _coordinates, _history


class FirmsForecaster:
    """Load the checkpoint once and reuse it for regional forecast requests."""

    def __init__(self, asset_root=None, *, checkpoint=None, threads=4, batch_size=256):
        if type(threads) is not int or threads < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("threads and batch_size must be positive integers")
        self.root = Path(asset_root or os.environ.get("FIRMS_BACKEND_DATA_HOME", DATA_ROOT)).resolve()
        self.checkpoint = Path(checkpoint or os.environ.get(
            "FIRMS_BACKEND_TIMESFM_CHECKPOINT", ROOT.parent / "models" / "timesfm-3.0"
        )).resolve()
        if not all((self.checkpoint / file).is_file() for file in ("config.json", "model.safetensors")):
            raise FileNotFoundError(f"TimesFM 3.0 checkpoint is missing in {self.checkpoint}. See the project README.")
        self.threads = threads
        self.batch_size = batch_size
        self._lock = Lock()
        self._model = None

    def forecast_daily(self, target_date, *, bbox, as_of=None, updates=(), minimum=0.0, limit=5000):
        """Estimate detection and conditional FRP for one UTC day, 1 to 7 days ahead."""
        bbox = _bbox(bbox)
        _validate_limits(minimum, limit)
        target = _day(target_date)
        history = History(self.root / "dataset", updates, bbox, self.threads)
        cutoff = history.last_day if as_of is None else _day(as_of)
        lead = (target - cutoff).days
        if not 1 <= lead <= 7:
            raise ValueError(f"target_date must be 1 to 7 UTC days after {cutoff}")
        periods = history.periods(cutoff - timedelta(days=63), cutoff)
        recent = [cells for cells, _ in periods[-14:]]
        neighbors = np.unique(np.concatenate(recent[-7:]))
        cells = np.unique(np.concatenate([*recent, *_neighbors(neighbors)]))
        fields = {"date_utc": str(target), "lead_days": lead}
        return self._forecast(periods, cells, cutoff, lead, bbox, minimum, limit, fields, "p_detected")

    def forecast_monthly(self, target_month, *, bbox, as_of=None, updates=(), minimum=0.0, limit=5000):
        """Estimate one UTC month's detection probability and conditional FRP on demand."""
        bbox = _bbox(bbox)
        _validate_limits(minimum, limit)
        if isinstance(target_month, str) and len(target_month) == 7:
            target_month += "-01"
        target = _day(target_month)
        if target.day != 1:
            raise ValueError("target_month must be YYYY-MM or the first day of that UTC month")
        history = History(self.root / "dataset", updates, bbox, self.threads)
        cutoff = history.last_day if as_of is None else _day(as_of)
        if cutoff.day < 14:
            raise ValueError("For monthly forecasting choose an as_of date on or after the 14th UTC")
        origin = np.datetime64(cutoff, "M")
        lead = int((np.datetime64(target, "M") - origin) / MONTH)
        if not 1 <= lead <= 12:
            raise ValueError(f"target_month must be 1 to 12 months after {origin}")
        cutoff = cutoff.replace(day=14)
        first = date.fromisoformat(str(origin - 36 * MONTH) + "-01")
        *periods, partial = history.periods(first, cutoff, monthly=True)
        older = [cells for cells, _ in periods]
        cells, counts = np.unique(np.concatenate(older), return_counts=True)
        candidates = np.unique(np.concatenate((cells[counts >= 2], older[-1], partial[0])))
        fields = {"target_month": str(target), "lead_months": lead}
        # Model step one covers the incomplete month immediately after the 36-month context.
        return self._forecast(periods, candidates, cutoff, lead + 1, bbox, minimum, limit, fields, "p_detected_any")

    def _forecast(self, periods, candidates, cutoff, step, bbox, minimum, limit, fields, probability_column):
        cells, coords = _coordinates(candidates, bbox)
        probabilities = np.empty(len(cells), dtype=np.float32)
        strength = np.empty_like(probabilities)
        with self._lock:
            if len(cells) and self._model is None:
                self._model = TimesFM(self.checkpoint, self.threads, self.batch_size)
            for start in range(0, len(cells), 32768):
                batch = slice(start, start + 32768)
                p, frp = self._model.score(_history(cells[batch], periods), step)
                probabilities[batch], strength[batch] = p[:, -1], frp[:, -1]
        qualifying = np.flatnonzero(probabilities >= minimum)
        selected = qualifying[np.argsort(-probabilities[qualifying], kind="stable")[:limit]]
        rows = []
        for index in selected:
            x, y = int(coords["grid_x"][index]), int(coords["grid_y"][index])
            frp = strength[index]
            rows.append({**fields, "grid_id": f"{x}:{y}",
                         "longitude": float(coords["longitude"][index]), "latitude": float(coords["latitude"][index]),
                         probability_column: float(probabilities[index]),
                         "frp_noaa20_mw_if_detected_p50": float(frp) if np.isfinite(frp) else None})
        return {**fields, "as_of_utc": str(cutoff), "bbox_wsen": list(bbox), "grid_km": 1,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "candidate_cells": len(cells), "rows": rows}


def _neighbors(cells):
    x, y = cells % WIDTH, cells // WIDTH
    for dy in (-1, 0, 1):
        valid = (0 <= y + dy) & (y + dy < HEIGHT)
        for dx in (-1, 0, 1):
            if (dx, dy) != (0, 0):
                yield (y[valid] + dy) * WIDTH + (x[valid] + dx) % WIDTH


def _validate_limits(minimum, limit):
    if type(limit) is not int or not 1 <= limit <= 50000:
        raise ValueError("limit must be an integer between 1 and 50000")
    if not 0 <= minimum <= 1:
        raise ValueError("minimum must be between 0 and 1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    time = parser.add_mutually_exclusive_group(required=True)
    time.add_argument("--date", help="Target UTC date (YYYY-MM-DD)")
    time.add_argument("--month", help="Target UTC month (YYYY-MM)")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"), required=True)
    parser.add_argument("--as-of", help="Latest usable observation date (UTC)")
    parser.add_argument("--update", type=Path, action="append", default=[])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--minimum", type=float, default=0)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    service = FirmsForecaster(args.data_root, checkpoint=args.checkpoint)
    predict = service.forecast_daily if args.date else service.forecast_monthly
    with redirect_stdout(sys.stderr):
        out = predict(args.date or args.month, bbox=args.bbox, as_of=args.as_of, updates=args.update,
                      minimum=args.minimum, limit=args.limit)
    print(json.dumps(out, allow_nan=False))


if __name__ == "__main__":
    main()
