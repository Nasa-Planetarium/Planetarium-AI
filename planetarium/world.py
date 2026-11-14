"""전 세계 0.1° 칸 → 국가(admin0)·1급 행정구역(admin1) 매핑 (land_frac 방식, korea_cells와 같은 규칙).

입력  data/external/boundaries/naturalearth_10m/ne_10m_admin_1_states_provinces.geojson (4,596개)
      data/external/boundaries/naturalearth_10m/ne_10m_admin_0_countries.geojson        (258개: 대륙·하위지역·이름)
      data/processed/korea_cells_0.1deg.parquet                                          (한국 시·도·시군구)
출력  data/processed/world_cells_0.1deg.parquet
      cell_id, lat_idx, lon_idx, cell_lat, cell_lon,
      land_frac        칸 면적 중 육지(admin1 합집합) 비율
      adm0_a3, country, land_frac_adm0   겹침이 가장 큰 국가와 그 비율
      continent, subregion               Natural Earth CONTINENT, SUBREGION (육지 없으면 null)
      adm1_code, adm1_name, land_frac_adm1
      is_land          land_frac >= 0.1 (AOI 집계에 쓰는 칸)
      kr_sido_cd, kr_sido_nm, kr_sgg_cd, kr_sgg_nm, kr_land_frac   한국 칸(korea_cells, geoBoundaries 기준)만

방법
  - 한 칸(0.1°) 안에서는 위경도 면적 비율 ≈ 실제 면적 비율 (투영 왜곡이 칸 안에서 무시할 수준)이라 위경도로 바로 잰다.
    (한국은 EPSG:5179로 계산한 korea_cells 값을 따로 둔다)
  - 폴리곤마다 0.1° 위도 띠로 잘라(clip_by_rect) 모양을 단순하게 만든 뒤, 칸 사각형과 겹침 면적을 벡터 연산으로 계산.
  - admin1이 없는 국가 7곳(작은 섬·환초)은 admin0 폴리곤으로 보충.

실행  uv run python -m planetarium.world
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import polars as pl
import shapely
from shapely.geometry import shape
from tqdm import tqdm

from planetarium.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
from planetarium.korea import KOREA_CELLS_PATH
from planetarium.runlog import start_run_log

RES = 0.1
N_LON = 3600
LAND_FRAC_MIN = 0.1
NE_DIR = EXTERNAL_DATA_DIR / "boundaries" / "naturalearth_10m"
WORLD_CELLS_PATH = PROCESSED_DATA_DIR / f"world_cells_{RES:g}deg.parquet"
# 칸 × 1급 행정구역 겹침 면적 가중치 (국가·1급 AOI 합계를 면적 비율로 배분)
WORLD_WEIGHTS_PATH = PROCESSED_DATA_DIR / f"world_cell_adm1_weights_{RES:g}deg.parquet"


def load_ne() -> tuple[pl.DataFrame, list, pl.DataFrame]:
    """admin1 속성·도형 + (admin1 없는 국가의 admin0 도형) / admin0 속성."""
    a1 = json.loads((NE_DIR / "ne_10m_admin_1_states_provinces.geojson").read_text(encoding="utf-8"))["features"]
    a0 = json.loads((NE_DIR / "ne_10m_admin_0_countries.geojson").read_text(encoding="utf-8"))["features"]
    adm0 = pl.DataFrame([{
        "adm0_a3": f["properties"]["ADM0_A3"], "country": f["properties"]["NAME"],
        "continent": f["properties"]["CONTINENT"], "subregion": f["properties"]["SUBREGION"],
    } for f in a0])
    units, geoms = [], []
    for f in a1:
        p = f["properties"]
        units.append({"adm1_code": p["adm1_code"], "adm1_name": p["name"] or p["name_en"] or p["adm1_code"],
                      "adm0_a3": p["adm0_a3"]})
        geoms.append(shapely.make_valid(shape(f["geometry"])))
    have = {u["adm0_a3"] for u in units}
    for f in a0:
        p = f["properties"]
        if p["ADM0_A3"] not in have:
            units.append({"adm1_code": f"{p['ADM0_A3']}-00", "adm1_name": p["NAME"], "adm0_a3": p["ADM0_A3"]})
            geoms.append(shapely.make_valid(shape(f["geometry"])))
    return pl.DataFrame(units), geoms, adm0


def polygon_cells(geom) -> tuple[np.ndarray, np.ndarray]:
    """도형 하나 → (겹치는 cell_id 배열, 각 칸에서 차지하는 비율)."""
    minx, miny, maxx, maxy = geom.bounds
    i0, i1 = math.floor((miny + 90) / RES), math.ceil((maxy + 90) / RES)
    ids, fracs = [], []
    cell_area = RES * RES
    for li in range(i0, i1):
        lat0 = li * RES - 90
        strip = shapely.clip_by_rect(geom, minx, lat0, maxx, lat0 + RES)
        if strip.is_empty:
            continue
        sx0, _, sx1, _ = strip.bounds
        j0, j1 = math.floor((sx0 + 180) / RES), math.ceil((sx1 + 180) / RES)
        js = np.arange(j0, j1)
        boxes = shapely.box(js * RES - 180, lat0, (js + 1) * RES - 180, lat0 + RES)
        areas = shapely.area(shapely.intersection(boxes, strip)) / cell_area
        keep = areas > 1e-9
        ids.append(li * N_LON + (js[keep] % N_LON))
        fracs.append(areas[keep])
    if not ids:
        return np.array([], dtype=np.int64), np.array([])
    return np.concatenate(ids).astype(np.int64), np.concatenate(fracs)


def build_world_cells() -> pl.DataFrame:
    units, geoms, adm0 = load_ne()
    parts = []
    for k, g in enumerate(tqdm(geoms, desc="admin1 → 칸", unit="poly", mininterval=10)):
        ids, fr = polygon_cells(g)
        if len(ids):
            parts.append(pl.DataFrame({"cell_id": ids, "frac": fr, "unit": np.full(len(ids), k, dtype=np.int32)}))
    pieces = pl.concat(parts).join(units.with_row_index("unit").with_columns(pl.col("unit").cast(pl.Int32)), on="unit")
    # 같은 단위가 칸을 여러 번 덮을 일은 없지만(띠 분할), 안전하게 합산
    a1 = pieces.group_by("cell_id", "adm1_code", "adm1_name", "adm0_a3").agg(pl.col("frac").sum())
    land = a1.group_by("cell_id").agg(land_frac=pl.col("frac").sum().clip(upper_bound=1.0))
    top1 = a1.sort("frac", descending=True).group_by("cell_id").first().rename({"frac": "land_frac_adm1"})
    a0 = a1.group_by("cell_id", "adm0_a3").agg(land_frac_adm0=pl.col("frac").sum())
    top0 = a0.sort("land_frac_adm0", descending=True).group_by("cell_id").first()
    df = (
        land.join(top0, on="cell_id").join(top1.select("cell_id", "adm1_code", "adm1_name", "land_frac_adm1"), on="cell_id")
        .join(adm0, on="adm0_a3", how="left")
        .with_columns(
            lat_idx=(pl.col("cell_id") // N_LON).cast(pl.Int32),
            lon_idx=(pl.col("cell_id") % N_LON).cast(pl.Int32),
        )
        .with_columns(
            cell_lat=((pl.col("lat_idx") + 0.5) * RES - 90).cast(pl.Float32),
            cell_lon=((pl.col("lon_idx") + 0.5) * RES - 180).cast(pl.Float32),
            is_land=pl.col("land_frac") >= LAND_FRAC_MIN,
        )
    )
    kc = pl.read_parquet(KOREA_CELLS_PATH).filter("is_korea").select(
        "cell_id", kr_sido_cd="sido_cd", kr_sido_nm="sido_nm", kr_sgg_cd="sgg_cd", kr_sgg_nm="sgg_nm", kr_land_frac="land_frac")
    # 한국 칸(geoBoundaries 기준)이 NE에서 육지로 안 잡힌 경우도 행을 만든다
    df = df.join(kc, on="cell_id", how="full", coalesce=True).with_columns(
        lat_idx=(pl.col("cell_id") // N_LON).cast(pl.Int32), lon_idx=(pl.col("cell_id") % N_LON).cast(pl.Int32))
    df = df.with_columns(
        cell_lat=((pl.col("lat_idx") + 0.5) * RES - 90).cast(pl.Float32),
        cell_lon=((pl.col("lon_idx") + 0.5) * RES - 180).cast(pl.Float32),
        land_frac=pl.col("land_frac").fill_null(0.0),
        is_land=pl.col("is_land").fill_null(False) | pl.col("kr_sgg_cd").is_not_null(),
        # 한국 칸은 국가를 KOR로 맞춘다 (경계 자료 차이로 해안 칸이 다른 나라/바다로 잡히는 것 방지)
        adm0_a3=pl.when(pl.col("kr_sgg_cd").is_not_null()).then(pl.lit("KOR")).otherwise(pl.col("adm0_a3")),
    ).drop("country", "continent", "subregion").join(adm0, on="adm0_a3", how="left")
    df = df.select(
        "cell_id", "lat_idx", "lon_idx", "cell_lat", "cell_lon", "land_frac", "is_land",
        "adm0_a3", "country", "land_frac_adm0", "continent", "subregion",
        "adm1_code", "adm1_name", "land_frac_adm1",
        "kr_sido_cd", "kr_sido_nm", "kr_sgg_cd", "kr_sgg_nm", "kr_land_frac",
    ).sort("cell_id")
    df.write_parquet(WORLD_CELLS_PATH, compression="zstd")
    # 가중치: 육지 칸(is_land)마다 1급 행정구역 겹침 면적을 칸 안에서 합 1로 정규화
    land_ids = df.filter("is_land").select("cell_id")
    w = (a1.join(land_ids, on="cell_id", how="semi").join(adm0.select("adm0_a3", "continent"), on="adm0_a3", how="left")
         .with_columns(overlap_frac=pl.col("frac"), weight=pl.col("frac") / pl.col("frac").sum().over("cell_id"))
         .select("cell_id", "adm1_code", "adm1_name", "adm0_a3", "continent", "overlap_frac", "weight").sort("cell_id"))
    w.write_parquet(WORLD_WEIGHTS_PATH, compression="zstd")
    return df


def load_world_cells(only_land: bool = False) -> pl.DataFrame:
    df = pl.read_parquet(WORLD_CELLS_PATH)
    return df.filter("is_land") if only_land else df


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="전 세계 칸 → 국가·1급 행정구역 매핑").parse_args(argv)
    start_run_log("world")
    t0 = time.perf_counter()
    df = build_world_cells()
    land = df.filter("is_land")
    print(f"[완료] {WORLD_CELLS_PATH.name}: 육지와 겹치는 칸 {df.height:,}개, land_frac>={LAND_FRAC_MIN} {land.height:,}개, "
          f"국가 {land['adm0_a3'].n_unique()}, 1급 행정구역 {land['adm1_code'].n_unique():,}, {time.perf_counter() - t0:,.0f}초")
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, thousands_separator=","):
        print(land.group_by("continent").agg(cells=pl.len(), countries=pl.col("adm0_a3").n_unique(),
                                             adm1=pl.col("adm1_code").n_unique()).sort("continent"))
        print(land.filter(pl.col("adm0_a3").is_in(["KOR", "ARG"])).group_by("adm0_a3", "country").agg(
            cells=pl.len(), adm1=pl.col("adm1_code").n_unique(), kr_sgg=pl.col("kr_sgg_cd").n_unique()))


if __name__ == "__main__":
    main()
