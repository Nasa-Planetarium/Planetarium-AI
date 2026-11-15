"""Generate new forecast partitions from prepared 1 km histories and saved models."""
import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..api import ROOT
from . import grid_1km, monthly_cells, year_1km
from .models import MODEL_DIRECTORY, MODEL_KIND, load_bundle

METHOD = "lightgbm_calibrated"


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


def predict_from_cache(kind, cache, output, *, models=MODEL_DIRECTORY, threads=8):
    """Write forecast data and return JSON-compatible metadata."""
    if kind not in MODEL_KIND:
        raise ValueError("kind must be daily or annual")
    cache, output = Path(cache), Path(output)
    if output.exists():
        raise FileExistsError(output)
    cutoff = date.fromisoformat(json.loads((cache / "metadata.json").read_text())["input_through_utc"])
    detector, intensity, calibration = load_bundle(kind, models)
    # Booster parameters govern inference threads when loaded from a file.
    detector.params["num_threads"] = threads
    intensity.params["num_threads"] = threads
    calibration = {METHOD: calibration}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        info = {"grid_km": 1, "selected_detector": METHOD, "forecast_origin_utc": str(cutoff)}
        if kind == "daily":
            read = grid_1km.archive(cache)
            count = grid_1km.output_forecast(
                read, cutoff, detector, intensity, None, calibration, METHOD, stage, threads,
            )
            info.update(
                key=["date_utc", "grid_id"],
                date_range_utc_inclusive=[str(cutoff + timedelta(days=1)),
                                          str(cutoff + timedelta(days=grid_1km.HORIZON))],
                candidate_cells=count, forecast_rows=count * grid_1km.HORIZON,
            )
        else:
            origin = np.datetime64(cutoff, "M")
            cutoff14 = cutoff.replace(day=14)
            if cutoff < cutoff14:
                raise ValueError("At least the first 14 days of the origin month are needed")
            read = monthly_cells.reader(cache)
            count, _ = year_1km.forecast(
                read, origin, cutoff14, detector, intensity, None,
                calibration, METHOD, stage, threads,
            )
            info.update(
                forecast_origin_utc=str(cutoff14), key=["target_month", "grid_id"],
                target_months_inclusive=[str(origin + monthly_cells.MONTH),
                                         str(origin + len(year_1km.LEADS) * monthly_cells.MONTH)],
                candidate_cells_per_month=count, forecast_rows=count * len(year_1km.LEADS),
            )
        (stage / "metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(MODEL_KIND))
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("FIRMS_BACKEND_DATA_HOME", ROOT)) / "dataset")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not args.reuse_cache:
        if args.output.exists():
            raise FileExistsError(args.output)
        prepare_cache(args.kind, args.data, args.cache, args.threads)
    print(json.dumps(predict_from_cache(args.kind, args.cache, args.output, threads=args.threads), indent=2))


if __name__ == "__main__":
    main()
