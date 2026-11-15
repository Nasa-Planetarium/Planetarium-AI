"""Generate new forecast partitions from prepared 1 km histories and saved models."""
import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..api import ROOT
from . import grid_1km, monthly_cells, timesfm_forecast, year_1km
from .models import MODEL_DIRECTORY, MODEL_KIND, load_bundle

LIGHTGBM = "lightgbm_calibrated"


def _checkpoint(path):
    if path is None:
        path = os.environ.get("FIRMS_BACKEND_TIMESFM_CHECKPOINT", ROOT / "timesfm-checkpoint")
    path = Path(path)
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"TimesFM 3.0 checkpoint missing in {path}. See firms_backend/README.md.")
    return path


def prepare_cache(kind, source, cache, threads=8):
    source, cache = Path(source), Path(cache)
    if kind == "annual":
        monthly_cells.extract(source, cache, threads)
    elif kind == "daily":
        manifest = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        cutoff = date.fromisoformat(manifest["date_range_utc_inclusive"][1])
        grid_1km.extract_days(source, cache, cutoff.year, threads)
        (cache / "metadata.json").write_text(json.dumps({"input_through_utc": str(cutoff)}) + "\n", encoding="utf-8")
    else:
        raise ValueError("kind must be daily or annual")


def predict_from_cache(kind, cache, output, *, checkpoint=None, bbox=None, batch_size=256,
                       method=timesfm_forecast.METHOD, models=MODEL_DIRECTORY, threads=8):
    """Write 1 km forecasts with TimesFM 3.0 and return JSON-compatible metadata."""
    if kind not in MODEL_KIND:
        raise ValueError("kind must be daily or annual")
    cache, output = Path(cache), Path(output)
    if output.exists():
        raise FileExistsError(output)
    cutoff = date.fromisoformat(json.loads((cache / "metadata.json").read_text())["input_through_utc"])
    if kind == "annual" and cutoff.day < 14:
        raise ValueError("At least the first 14 days of the origin month are needed")
    if batch_size < 1 or threads < 1:
        raise ValueError("batch_size and threads must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        info = {"grid_km": 1, "selected_detector": method, "forecast_origin_utc": str(cutoff)}
        if method == timesfm_forecast.METHOD:
            model = timesfm_forecast.TimesFM(_checkpoint(checkpoint), threads, batch_size)
            count = timesfm_forecast.forecast(
                kind, cache, cutoff, stage, model=model, bbox=bbox, threads=threads, batch_size=batch_size,
            )
            info.update(model_id=timesfm_forecast.MODEL_ID, model_revision=timesfm_forecast.MODEL_REVISION,
                        inference_device=model.device, zero_shot=True, bbox_wsen=bbox,
                        probability_method="quantile-integrated binary occupancy, uncalibrated")
        elif method == LIGHTGBM:
            if bbox is not None:
                raise ValueError("bbox is supported for TimesFM 3.0")
            count = _lightgbm(kind, cache, cutoff, stage, models, threads)
        else:
            raise ValueError(f"Unknown forecast method: {method}")
        if kind == "daily":
            info.update(
                key=["date_utc", "grid_id"],
                date_range_utc_inclusive=[str(cutoff + timedelta(days=1)),
                                          str(cutoff + timedelta(days=grid_1km.HORIZON))],
                candidate_cells=count, forecast_rows=count * grid_1km.HORIZON,
            )
        else:
            origin = np.datetime64(cutoff, "M")
            cutoff14 = cutoff.replace(day=14)
            info.update(
                forecast_origin_utc=str(cutoff14), key=["target_month", "grid_id"],
                target_months_inclusive=[str(origin + monthly_cells.MONTH),
                                         str(origin + len(year_1km.LEADS) * monthly_cells.MONTH)],
                candidate_cells_per_month=count, forecast_rows=count * len(year_1km.LEADS),
            )
        (stage / "metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return info


def _lightgbm(kind, cache, cutoff, output, models, threads):
    detector, intensity, calibration = load_bundle(kind, models)
    detector.params["num_threads"] = threads
    intensity.params["num_threads"] = threads
    calibration = {LIGHTGBM: calibration}
    if kind == "daily":
        return grid_1km.output_forecast(
            grid_1km.archive(cache), cutoff, detector, intensity, None, calibration, LIGHTGBM, output, threads,
        )
    count, _ = year_1km.forecast(
        monthly_cells.reader(cache), np.datetime64(cutoff, "M"), cutoff.replace(day=14), detector, intensity,
        None, calibration, LIGHTGBM, output, threads,
    )
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(MODEL_KIND))
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("FIRMS_BACKEND_DATA_HOME", ROOT)) / "dataset")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--method", choices=[timesfm_forecast.METHOD, LIGHTGBM], default=timesfm_forecast.METHOD)
    parser.add_argument("--checkpoint-3", type=Path, help="Directory containing the official TimesFM 3.0 checkpoint")
    parser.add_argument("--bbox", metavar=("WEST", "SOUTH", "EAST", "NORTH"), type=float, nargs=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    checkpoint = _checkpoint(args.checkpoint_3) if args.method == timesfm_forecast.METHOD else None
    if not args.reuse_cache:
        if args.output.exists():
            raise FileExistsError(args.output)
        prepare_cache(args.kind, args.data, args.cache, args.threads)
    print(json.dumps(predict_from_cache(
        args.kind, args.cache, args.output, checkpoint=checkpoint, bbox=args.bbox, batch_size=args.batch_size,
        method=args.method, threads=args.threads,
    ), indent=2))


if __name__ == "__main__":
    main()
