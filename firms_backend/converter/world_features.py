"""Extract worldwide FIRMS features across all years and all five platforms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

from .convert import define_grid, sql_string

BLOCK_KM = 250
DAILY_SAMPLE_MODULUS = 64
PLATFORMS = ("Terra", "Aqua", "S-NPP", "NOAA-20", "NOAA-21")


def monthly_features_query(source, block_km=BLOCK_KM):
    return f"""
        WITH cell_day AS (
            SELECT day, x, y, platform,
                   count(*)::INT AS pixels, avg(pixel_area) AS pixel_area,
                   avg(nrt) AS nrt_share, avg(daytime) AS day_share,
                   sum(frp) AS frp_sum, max(frp) AS frp_peak
            FROM {source} GROUP BY day,x,y,platform
        )
        SELECT date_trunc('month',day)::DATE AS month,
               floor(x / {block_km})::INT AS bx, floor(y / {block_km})::INT AS grid_by,
               platform, count(*) AS cell_days, sum(pixels)::BIGINT AS pixels,
               count(DISTINCT day)::INT AS active_days, avg(pixel_area) AS pixel_area,
               avg(nrt_share) AS nrt_share, avg(day_share) AS day_share,
               sum(frp_sum) AS frp_sum, avg(frp_peak) AS mean_cell_day_peak
        FROM cell_day GROUP BY ALL
    """


def extract(input_path, output, is_modis, work, block_km):
    name = "modis" if is_modis else "viirs"
    staging = work / f"{name}-rows"
    if staging.exists():
        raise FileExistsError(f"Use a fresh work directory: {staging}")
    with duckdb.connect(config={
        "threads": "8", "memory_limit": "8GB", "preserve_insertion_order": "false",
        "temp_directory": str(work / f"{name}-spill"),
    }) as db:
        define_grid(db, 1)
        platform = (
            "CASE satellite WHEN 'Terra' THEN 0 WHEN 'Aqua' THEN 1 END" if is_modis else
            "CASE satellite WHEN 'SNPP' THEN 2 WHEN 'N20' THEN 3 WHEN 'N21' THEN 4 END"
        )
        quality = ("try_cast(confidence AS DOUBLE) >= 30" if is_modis else
                   "lower(confidence) IN ('n','h')")
        db.execute(f"""
            COPY (
                SELECT CAST(acq_date AS DATE) AS day,
                       year(CAST(acq_date AS DATE)) AS year,
                       CAST(acq_time AS SMALLINT) AS minute,
                       ({platform})::UTINYINT AS platform,
                       cell_x(longitude::DOUBLE) AS x, cell_y(latitude::DOUBLE) AS y,
                       CASE WHEN isfinite(frp::DOUBLE) AND frp::DOUBLE >= 0
                            THEN frp::FLOAT END AS frp,
                       (scan::DOUBLE * track::DOUBLE)::FLOAT AS pixel_area,
                       CAST(ends_with(version,'NRT') AS UTINYINT) AS nrt,
                       CAST(upper(daynight) = 'D' AS UTINYINT) AS daytime
                FROM read_csv({sql_string(input_path.resolve())}, header=true, all_varchar=true,
                              delim=',', strict_mode=true, sample_size=10000)
                WHERE (type = '0' OR type IS NULL) AND {quality}
            ) TO {sql_string(staging)} (FORMAT PARQUET, PARTITION_BY (year));
        """)
        print(name, "normalized full CSV", flush=True)
        for partition in sorted(staging.glob("year=*")):
            year = int(partition.name.split("=")[1])
            db.execute(f"""
                CREATE OR REPLACE TEMP TABLE source_rows AS
                    SELECT * FROM read_parquet({sql_string(partition / '*.parquet')});
                COPY ({monthly_features_query('source_rows', block_km)})
                    TO {sql_string(output / f'{name}_monthly_{year}.parquet')} (FORMAT PARQUET);
            """)
            if year >= 2012:
                db.execute(f"""
                    COPY (
                        WITH minute AS (
                            SELECT day,x,y,platform,minute,count(*) AS n,avg(pixel_area) AS area,avg(daytime) AS day_share,
                                   sum(frp) AS minute_frp
                            FROM source_rows
                            WHERE hash(day,x,y) % {DAILY_SAMPLE_MODULUS} = 0
                            GROUP BY ALL
                        )
                        SELECT day,x,y,platform,sum(n)::INT AS pixels,
                               avg(area) AS pixel_area, median(minute_frp) AS grid_frp_mw,
                               sum(day_share * n) / sum(n) AS day_share,
                               count(*)::INT AS minutes
                        FROM minute GROUP BY day,x,y,platform
                    ) TO {sql_string(output / f'{name}_daily_{year}.parquet')} (FORMAT PARQUET);
                """)
            print(name, year, db.execute("SELECT count(*) FROM source_rows").fetchone()[0],
                  "qualified input rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modis", required=True, type=Path)
    parser.add_argument("--viirs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--block-km", type=int, choices=(10, 50, 125, 250), default=250)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(2) as pool:
        tasks = [pool.submit(extract, args.modis, args.output, True, args.work, args.block_km),
                 pool.submit(extract, args.viirs, args.output, False, args.work, args.block_km)]
        for task in tasks:
            task.result()


if __name__ == "__main__":
    main()
