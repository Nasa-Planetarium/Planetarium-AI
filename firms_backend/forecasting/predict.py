"""Request regional TimesFM 3.0 forecasts for one UTC time or a range."""
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

    def forecast_daily(self, target_date, *, end_date=None, bbox, as_of=None, updates=(), minimum=0.0, limit=5000):
        """Estimate detection and conditional FRP for every UTC day in an inclusive range."""
        bbox = _bbox(bbox)
        _validate_limits(minimum, limit)
        target = _day(target_date)
        end = target if end_date is None else _day(end_date)
        history = History(self.root / "dataset", updates, bbox, self.threads)
        cutoff = history.last_day if as_of is None else _day(as_of)
        leads = _leads((target - cutoff).days, (end - cutoff).days, str(cutoff))
        periods = history.periods(cutoff - timedelta(days=63), cutoff)
        recent = [cells for cells, _ in periods[-14:]]
        neighbors = np.unique(np.concatenate(recent[-7:]))
        cells = np.unique(np.concatenate([*recent, *_neighbors(neighbors)]))
        targets = [(lead, {"date_utc": str(cutoff + timedelta(days=lead)), "lead_days": lead}) for lead in leads]
        fields = targets[0][1] if end_date is None else {"date_from_utc": str(target), "date_to_utc": str(end)}
        return self._forecast(periods, cells, cutoff, targets, bbox, minimum, limit, fields, "p_detected")

    def forecast_monthly(self, target_month, *, end_month=None, bbox, as_of=None, updates=(), minimum=0.0, limit=5000):
        """Estimate detection and conditional FRP for every UTC month in an inclusive range."""
        bbox = _bbox(bbox)
        _validate_limits(minimum, limit)
        target = _month(target_month)
        end = target if end_month is None else _month(end_month)
        history = History(self.root / "dataset", updates, bbox, self.threads)
        cutoff = history.last_day if as_of is None else _day(as_of)
        if cutoff.day < 14:
            raise ValueError("For monthly forecasting choose an as_of date on or after the 14th UTC")
        origin = np.datetime64(cutoff, "M")
        leads = _leads(int((target - origin) / MONTH), int((end - origin) / MONTH), str(origin))
        cutoff = cutoff.replace(day=14)
        first = date.fromisoformat(str(origin - 36 * MONTH) + "-01")
        *periods, partial = history.periods(first, cutoff, monthly=True)
        older = [cells for cells, _ in periods]
        cells, counts = np.unique(np.concatenate(older), return_counts=True)
        candidates = np.unique(np.concatenate((cells[counts >= 2], older[-1], partial[0])))
        # Model step one covers the incomplete month immediately after the 36-month context.
        targets = [(lead + 1, {"target_month": f"{origin + lead * MONTH}-01", "lead_months": lead}) for lead in leads]
        fields = targets[0][1] if end_month is None else {"month_from_utc": str(target), "month_to_utc": str(end)}
        return self._forecast(periods, candidates, cutoff, targets, bbox, minimum, limit, fields, "p_detected_any")

    def _forecast(self, periods, candidates, cutoff, targets, bbox, minimum, limit, fields, probability_column):
        cells, coords = _coordinates(candidates, bbox)
        horizon = targets[-1][0]
        # Keep the number of cell × lead predictions in each chunk bounded as the horizon grows.
        chunk_size = max(1, 32768 * 64 // max(64, horizon))
        empty = np.empty(0, dtype=np.float32)
        best = [(np.empty(0, dtype=np.intp), empty, empty)] * len(targets)
        with self._lock:
            if len(cells) and self._model is None:
                self._model = TimesFM(self.checkpoint, self.threads, self.batch_size)
            for start in range(0, len(cells), chunk_size):
                batch = slice(start, start + chunk_size)
                p, frp = self._model.score(_history(cells[batch], periods), horizon)
                for i, (step, _) in enumerate(targets):
                    qualifying = np.flatnonzero(p[:, step - 1] >= minimum)
                    previous_indices, previous_p, previous_frp = best[i]
                    indices = np.concatenate((previous_indices, start + qualifying))
                    probabilities = np.concatenate((previous_p, p[qualifying, step - 1]))
                    strength = np.concatenate((previous_frp, frp[qualifying, step - 1]))
                    order = np.argsort(-probabilities, kind="stable")[:limit]
                    best[i] = indices[order], probabilities[order], strength[order]
        rows = []
        for (_, time), (indices, probabilities, strength) in zip(targets, best, strict=True):
            for index, p, frp in zip(indices, probabilities, strength, strict=True):
                x, y = int(coords["grid_x"][index]), int(coords["grid_y"][index])
                rows.append({**time, "grid_id": f"{x}:{y}",
                             "longitude": float(coords["longitude"][index]), "latitude": float(coords["latitude"][index]),
                             probability_column: float(p),
                             "frp_noaa20_mw_if_detected_p50": float(frp) if np.isfinite(frp) else None})
        return {**fields, "as_of_utc": str(cutoff), "bbox_wsen": list(bbox), "grid_km": 1,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "candidate_cells": len(cells), "rows": rows}


def _month(value):
    if isinstance(value, str) and len(value) == 7:
        value += "-01"
    day = _day(value)
    if day.day != 1:
        raise ValueError("Expected YYYY-MM or the first day of a UTC month")
    return np.datetime64(day, "M")


def _leads(first, last, origin):
    if first < 1 or last < first:
        raise ValueError(f"Forecast start must be after {origin}, and end must be on or after start")
    return range(first, last + 1)


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
    parser.add_argument("--end-date", help="Last UTC date of an inclusive range")
    parser.add_argument("--end-month", help="Last UTC month of an inclusive range")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"), required=True)
    parser.add_argument("--as-of", help="Latest usable observation date (UTC)")
    parser.add_argument("--update", type=Path, action="append", default=[])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--minimum", type=float, default=0)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    if (args.end_date and not args.date) or (args.end_month and not args.month):
        parser.error("--end-date requires --date and --end-month requires --month")
    service = FirmsForecaster(args.data_root, checkpoint=args.checkpoint)
    predict = service.forecast_daily if args.date else service.forecast_monthly
    end = {"end_date": args.end_date} if args.date else {"end_month": args.end_month}
    with redirect_stdout(sys.stderr):
        out = predict(args.date or args.month, **end, bbox=args.bbox, as_of=args.as_of, updates=args.update,
                      minimum=args.minimum, limit=args.limit)
    print(json.dumps(out, allow_nan=False))


if __name__ == "__main__":
    main()
