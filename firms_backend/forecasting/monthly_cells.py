"""Extract monthly series directly from the harmonized archive's 1 km grid-days."""
import argparse
from datetime import date
from functools import lru_cache
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np

from ..converter.convert import sql_string

WIDTH = 34668
HEIGHT = 14714
X_ORIGIN = 17334
Y_ORIGIN = 7357
X_SCALE = 6371.0088 * np.cos(np.pi / 6)
Y_SCALE = 6371.0088 / np.cos(np.pi / 6)
MONTH = np.timedelta64(1, "M")
CONTEXT = 36
HORIZON = 12
FRP_HISTORY = "measured_native_platform_ratio"


@lru_cache(maxsize=1)
def reference_frp():
    """Use measured FRP and the converter's fitted cross-sensor ratios."""
    path = Path(__file__).resolve().parents[1] / "converter/global/world_daily_model.joblib"
    bundle = joblib.load(path)
    ratios = bundle["frp_models"]["ratios"]
    cases = " ".join(f"WHEN {sql_string(bundle['platforms'][key])} THEN {float(ratio)}"
                     for key, ratio in ratios.items())
    return f"native_grid_frp_mw * CASE native_frp_source_platform WHEN 'NOAA-20' THEN 1 {cases} END"


def date_label(month):
    return str(np.datetime64(month, "D"))


def extract(source, target, threads):
    manifest = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if manifest["grid_km"] != 1 or manifest["daily_key"] != ["date_utc", "grid_id"]:
        raise ValueError("Expected the harmonized 1 km grid-day archive")
    last = date.fromisoformat(manifest["date_range_utc_inclusive"][1])
    months = np.arange(np.datetime64("2019-01", "M"), np.datetime64(last, "M") + MONTH, MONTH)
    target.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={"threads": str(threads), "memory_limit": "8GB"}) as db:
        for month in months:
            day = date.fromisoformat(date_label(month))
            path = source / "grid_daily" / f"year={day.year}" / f"month={day.month}" / "data.parquet"
            rows = db.execute(f"""
                SELECT ((grid_y + 7357) * 34668 + grid_x + 17334)::INT AS cell,
                       count(*)::UTINYINT AS days,
                       count_if(day(date_utc) <= 14)::UTINYINT AS first14,
                       median({reference_frp()})::FLOAT AS frp
                FROM read_parquet(?) WHERE detected GROUP BY cell ORDER BY cell
            """, [str(path)]).fetchnumpy()
            temp = target / f"{month}.tmp.npz"
            np.savez(temp, cells=rows["cell"], days=rows["days"], first14=rows["first14"],
                     frp=np.ma.filled(rows["frp"], np.nan))
            temp.replace(target / f"{month}.npz")
            print(month, len(rows["cell"]), "1 km cells", flush=True)
    meta = {"input_through_utc": str(last), "first_month": str(months[0]),
            "last_month": str(months[-1]), "frp_history": FRP_HISTORY}
    (target / "metadata.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")


def reader(folder):
    @lru_cache(maxsize=60)
    def read(month):
        with np.load(folder / f"{month}.npz") as file:
            return file["cells"], file["days"], file["first14"], file["frp"]
    return read


def candidates(read, origin):
    older = [read(origin - i * MONTH)[0] for i in range(1, CONTEXT + 1)]
    cells, counts = np.unique(np.concatenate(older), return_counts=True)
    now_cells, _, first14, _ = read(origin)
    return np.unique(np.concatenate([cells[counts >= 2], older[0], now_cells[first14 > 0]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("final-delivery/dataset"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()
    extract(args.source, args.cache, args.threads)


if __name__ == "__main__":
    main()
