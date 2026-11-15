"""Extract daily histories and select worldwide 1 km forecast candidates."""
from datetime import timedelta
from functools import lru_cache

import duckdb
import numpy as np

from .monthly_cells import WIDTH, HEIGHT, X_ORIGIN, Y_ORIGIN, reference_frp

HORIZON = 7
CONTEXT = 64
CANDIDATE_DAYS = 14
NEIGHBOR_DAYS = 7
OFFSETS = tuple((x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if (x, y) != (0, 0))


def extract_days(source, target, last_year, threads):
    target.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={"threads": str(threads), "memory_limit": "11GB",
                                "preserve_insertion_order": "false"}) as db:
        for year in range(2022, last_year + 1):
            rows = db.execute(f"""
                SELECT date_utc, ((grid_y + 7357) * 34668 + grid_x + 17334)::INT AS cell,
                       ({reference_frp()})::FLOAT AS frp
                FROM read_parquet(?) WHERE detected ORDER BY date_utc, cell
            """, [str(source / "grid_daily" / f"year={year}" / "month=*/*.parquet")]).fetchnumpy()
            days = rows["date_utc"].astype("datetime64[D]")
            dates, starts = np.unique(days, return_index=True)
            frp = np.ma.filled(rows["frp"], np.nan)
            for day, i, j in zip(dates, starts, np.r_[starts[1:], len(days)], strict=True):
                np.savez(target / f"{day}.npz", ids=rows["cell"][i:j], frp=frp[i:j])
            print(f"Extracted {year}: {len(days):,} exact grid-days", flush=True)


def archive(cache):
    @lru_cache(maxsize=130)
    def read(day):
        with np.load(cache / f"{day}.npz") as file:
            return file["ids"], file["frp"]
    return read


def shift(cells, dx, dy):
    x = (cells % WIDTH + dx) % WIDTH
    y = cells // WIDTH + dy
    valid = (y >= 0) & (y < HEIGHT)
    return (y[valid] * WIDTH + x[valid]).astype(np.int32), valid


def candidate_cells(read, origin):
    daily = [read(origin - timedelta(days=i))[0] for i in range(CANDIDATE_DAYS)]
    neighborhood = np.unique(np.concatenate(daily[:NEIGHBOR_DAYS]))
    return np.unique(np.concatenate([*daily, *(shift(neighborhood, dx, dy)[0] for dx, dy in OFFSETS)]))
