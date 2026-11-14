"""한국 평가 마스크: 0.1° 칸별 한국 육지 비율과 시·도/시군구 배정.

출력  data/processed/korea_cells_0.1deg.parquet
      cell_id, lat_idx, lon_idx, cell_lat, cell_lon,
      land_frac      칸 면적 중 한국 육지(geoBoundaries ADM0) 비율 (EPSG:5179 면적 기준)
      center_in      칸 중심점이 ADM0 안에 있는지 (이전 방식 비교용)
      is_korea       land_frac >= 0.1
      sido_cd, sido_nm, sgg_cd, sgg_nm   겹치는 면적이 가장 큰 시·도/시군구 (admdongkor *_20251231)
      sido_frac, sgg_frac                그 시·도/시군구가 칸에서 차지하는 비율
      island         본토(가장 큰 폴리곤)와 겹치지 않는 칸 (제주·울릉도 등)

방법
  - 칸 사각형(위경도)을 변 따라 촘촘히 점을 찍은 뒤 EPSG:4326 → EPSG:5179로 변환해 면적을 잰다
    (위도에 따라 0.1° 칸 넓이가 달라지므로 투영 좌표에서 비율 계산).
  - 행정구역은 admdongkor(EPSG:5179) 원본 좌표 그대로 겹침 면적 비교.
  - land_frac 기준 0.1: 해안·섬 칸 중 육지가 칸의 10% 이상인 곳만 한국 칸으로 본다.
    (중심점 방식은 해안 칸을 빠뜨리거나 바다 칸을 넣는 문제가 있어 면적 비율로 바꿈)

실행  uv run python -m planetarium.korea
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from shapely.geometry import Polygon, shape

from planetarium.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR
from planetarium.runlog import start_run_log

RES = 0.1
N_LON = round(360 / RES)
LAND_FRAC_MIN = 0.1
KOREA_CELLS_PATH = PROCESSED_DATA_DIR / f"korea_cells_{RES:g}deg.parquet"
# 칸 × 시군구 겹침 면적 가중치 (AOI 합계를 면적 비율로 배분해 252개 시군구가 모두 값을 갖게)
KOREA_WEIGHTS_PATH = PROCESSED_DATA_DIR / f"korea_cell_sgg_weights_{RES:g}deg.parquet"
ADM0_PATH = EXTERNAL_DATA_DIR / "boundaries" / "geoboundaries" / "geoBoundaries-KOR-ADM0.geojson"
ADM_DIR = EXTERNAL_DATA_DIR / "boundaries" / "admdongkor_20261001"

TO_5179 = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
TO_4326 = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)


def load_adm(name: str) -> tuple[pl.DataFrame, np.ndarray]:
    """admdongkor GeoParquet (EPSG:5179 WKB) -> (속성표, shapely 배열)."""
    t = pq.read_table(ADM_DIR / f"{name}.parquet")
    geoms = shapely.from_wkb(t.column("geom").to_pylist())
    return pl.from_arrow(t.drop(["geom"])), shapely.make_valid(geoms)


def cell_polygon_5179(lat_idx: int, lon_idx: int, n: int = 10) -> Polygon:
    lat0, lon0 = lat_idx * RES - 90, lon_idx * RES - 180
    lat1, lon1 = lat0 + RES, lon0 + RES
    t = np.linspace(0, 1, n, endpoint=False)
    lons = np.concatenate([lon0 + t * RES, np.full(n, lon1), lon1 - t * RES, np.full(n, lon0)])
    lats = np.concatenate([np.full(n, lat0), lat0 + t * RES, np.full(n, lat1), lat1 - t * RES])
    x, y = TO_5179.transform(lons, lats)
    return Polygon(zip(x, y))


def build_korea_cells() -> pl.DataFrame:
    g = json.loads(ADM0_PATH.read_text(encoding="utf-8"))
    adm0_ll = shapely.make_valid(shape(g["features"][0]["geometry"]))
    parts = sorted(getattr(adm0_ll, "geoms", [adm0_ll]), key=lambda p: -p.area)
    mainland_ll = parts[0]
    adm0 = shapely.transform(adm0_ll, lambda xy: np.column_stack(TO_5179.transform(xy[:, 0], xy[:, 1])))
    mainland = shapely.transform(mainland_ll, lambda xy: np.column_stack(TO_5179.transform(xy[:, 0], xy[:, 1])))
    shapely.prepare(adm0)

    sido_attr, sido_geom = load_adm("sido_20251231")
    sgg_attr, sgg_geom = load_adm("sgg_20251231")
    sido_tree, sgg_tree = shapely.STRtree(sido_geom), shapely.STRtree(sgg_geom)

    minx, miny, maxx, maxy = adm0_ll.bounds
    rows, wrows = [], []
    for li in range(int((miny + 90) / RES) - 1, int((maxy + 90) / RES) + 2):
        for lj in range(int((minx + 180) / RES) - 1, int((maxx + 180) / RES) + 2):
            cell = cell_polygon_5179(li, lj)
            if not adm0.intersects(cell):
                continue
            land = cell.intersection(adm0).area / cell.area
            clat, clon = (li + 0.5) * RES - 90, (lj + 0.5) * RES - 180

            def best(tree: shapely.STRtree, geoms: np.ndarray) -> tuple[int | None, float]:
                idx = tree.query(cell, predicate="intersects")
                if len(idx) == 0:
                    # ADM0에는 있지만 admdongkor 행정구역 밖인 칸(예: 비무장지대 부근) → 가장 가까운 곳
                    return int(tree.nearest(cell)), 0.0
                areas = shapely.area(shapely.intersection(geoms[idx], cell))
                k = int(np.argmax(areas))
                return int(idx[k]), float(areas[k] / cell.area)

            si, sfrac = best(sido_tree, sido_geom)
            gi, gfrac = best(sgg_tree, sgg_geom)
            # 모든 시군구와의 겹침 면적 (가중치용)
            gidx = sgg_tree.query(cell, predicate="intersects")
            if len(gidx):
                gareas = shapely.area(shapely.intersection(sgg_geom[gidx], cell)) / cell.area
                for k, a in zip(gidx, gareas):
                    if a > 1e-9:
                        wrows.append({"cell_id": li * N_LON + lj, "sgg_cd": sgg_attr["sggcd"][int(k)],
                                      "sgg_nm": sgg_attr["sggnm"][int(k)], "sido_cd": sgg_attr["sidocd"][int(k)],
                                      "sido_nm": sgg_attr["sidonm"][int(k)], "overlap_frac": float(a)})
            rows.append({
                "cell_id": li * N_LON + lj, "lat_idx": li, "lon_idx": lj,
                "cell_lat": round(clat, 4), "cell_lon": round(clon, 4),
                "land_frac": land,
                "center_in": bool(adm0_ll.contains(shapely.Point(clon, clat))),
                "sido_cd": sido_attr["sidocd"][si] if si is not None else None,
                "sido_nm": sido_attr["sidonm"][si] if si is not None else None,
                "sgg_cd": sgg_attr["sggcd"][gi] if gi is not None else None,
                "sgg_nm": sgg_attr["sggnm"][gi] if gi is not None else None,
                "sido_frac": sfrac, "sgg_frac": gfrac,
                "island": not mainland.intersects(cell),
            })
    df = pl.DataFrame(rows).with_columns(
        pl.col("cell_id").cast(pl.Int64), pl.col("lat_idx", "lon_idx").cast(pl.Int32),
        pl.col("cell_lat", "cell_lon").cast(pl.Float32),
        is_korea=pl.col("land_frac") >= LAND_FRAC_MIN,
    ).sort("cell_id")
    df.write_parquet(KOREA_CELLS_PATH)
    build_weights(df, pl.DataFrame(wrows))
    return df


def build_weights(cells: pl.DataFrame, w: pl.DataFrame) -> pl.DataFrame:
    """한국 칸(is_korea)마다 시군구 겹침 면적 비율을 칸 안에서 합이 1이 되게 정규화한 가중치.
    겹치는 시군구가 없는 칸(비무장지대 부근)은 가장 가까운 시군구에 가중치 1."""
    k = cells.filter("is_korea").select("cell_id", "sgg_cd", "sgg_nm", "sido_cd", "sido_nm")
    w = w.with_columns(pl.col("cell_id").cast(pl.Int64)).join(k.select("cell_id"), on="cell_id", how="semi")
    w = w.with_columns(weight=pl.col("overlap_frac") / pl.col("overlap_frac").sum().over("cell_id"))
    lone = k.join(w.select("cell_id").unique(), on="cell_id", how="anti").with_columns(
        overlap_frac=pl.lit(0.0), weight=pl.lit(1.0))
    out = pl.concat([w, lone.select(w.columns)]).sort("cell_id", "sgg_cd")
    out.write_parquet(KOREA_WEIGHTS_PATH)
    return out


def load_korea_cells(only_korea: bool = True) -> pl.DataFrame:
    df = pl.read_parquet(KOREA_CELLS_PATH)
    return df.filter(pl.col("is_korea")) if only_korea else df


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="한국 0.1° 칸 목록 (육지 비율, 시·도/시군구)").parse_args(argv)
    start_run_log("korea")
    df = build_korea_cells()
    k = df.filter("is_korea")
    print(f"[완료] {KOREA_CELLS_PATH.name}: ADM0와 겹치는 칸 {df.height:,}개, 한국 칸(land_frac>={LAND_FRAC_MIN}) {k.height:,}개")
    print(f"  중심점 방식 칸 {df['center_in'].sum():,}개 | 둘 다 {(df['center_in'] & df['is_korea']).sum():,} | "
          f"면적 방식에만 {(~df['center_in'] & df['is_korea']).sum():,} | 중심점 방식에만 {(df['center_in'] & ~df['is_korea']).sum():,}")
    print(f"  섬 칸(본토와 안 겹침) {k['island'].sum():,}개, 시·도 미배정 {k['sido_cd'].is_null().sum()}, 시군구 미배정 {k['sgg_cd'].is_null().sum()}")
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True):
        print(k.group_by("sido_cd", "sido_nm").agg(cells=pl.len(), island_cells=pl.col("island").sum(),
                                                   sgg=pl.col("sgg_cd").n_unique()).sort("sido_cd"))


if __name__ == "__main__":
    main()
