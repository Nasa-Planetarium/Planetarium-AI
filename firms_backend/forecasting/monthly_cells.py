"""Extract monthly series directly from the harmonized archive's 1 km grid-days."""
import argparse
from datetime import date
from functools import lru_cache
import json
from pathlib import Path

import duckdb
import numpy as np

WIDTH = 34668
X_ORIGIN = 17334
Y_ORIGIN = 7357
X_SCALE = 6371.0088 * np.cos(np.pi / 6)
Y_SCALE = 6371.0088 / np.cos(np.pi / 6)
MONTH = np.timedelta64(1, "M")


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
            rows = db.execute("""
                SELECT ((grid_y + 7357) * 34668 + grid_x + 17334)::INT AS cell,
                       count(*)::UTINYINT AS days,
                       count_if(day(date_utc) <= 14)::UTINYINT AS first14,
                       median(frp_noaa20_mw)::FLOAT AS frp
                FROM read_parquet(?) WHERE detected GROUP BY cell ORDER BY cell
            """, [str(path)]).fetchnumpy()
            temp = target / f"{month}.tmp.npz"
            np.savez(temp, cells=rows["cell"], days=rows["days"], first14=rows["first14"],
                     frp=np.ma.filled(rows["frp"], np.nan))
            temp.replace(target / f"{month}.npz")
            print(month, len(rows["cell"]), "1 km cells", flush=True)
    meta = {"input_through_utc": str(last), "first_month": str(months[0]),
            "last_month": str(months[-1])}
    (target / "metadata.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")


def reader(folder):
    @lru_cache(maxsize=60)
    def read(month):
        with np.load(folder / f"{month}.npz") as file:
            return file["cells"], file["days"], file["first14"], file["frp"]
    return read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("final-delivery/dataset"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()
    extract(args.source, args.cache, args.threads)


if __name__ == "__main__":
    main()
