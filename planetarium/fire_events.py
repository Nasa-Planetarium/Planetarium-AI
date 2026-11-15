"""화재 사건 묶기 + 불탄 면적(MCD64A1) + AOI × 월 화재 양상(fire_regime).

1) events  기준 점(harmonized_hotspots/reference)을 시공간 연결로 사건화 (union-find = 연결 성분).
           두 점이 같은 사건: 거리 ≤ D (두 점 센서 중 큰 값; config.EVENT_DIST_M = VIIRS 750 m, MODIS 2 km)
                              그리고 현지 날짜 차 ≤ T (config.EVENT_GAP_DAYS = 1일).
           하루씩 (전날 ∪ 오늘) 점으로 KD-tree를 만들어 가까운 쌍을 찾고, 그날의 연결 성분을 '대표 점과의 간선'으로만
           남긴 뒤(간선 수 ≈ 점 수), 전 기간 간선으로 한 번 더 연결 성분을 구한다 → 사건.
           범위: 한국(adm0 KOR 점)·아르헨티나(adm0 ARG 점), 2000-11 ~ 2026-11 (국경에서 잘린 사건은 그 나라 쪽만).
           감지 픽셀 면적: 원본 통합 파일(interim/firms_hotspots)의 scan × track (km²)를 키(위성·위경도·시각)로 붙인다.
           → data/processed/fire_events.parquet
2) burned  MCD64A1 v061 Burn Date(500 m, 월별)와 겹치기 (2012~2025, 한국 h27v05·h28v05, 아르헨티나 h11~h14 v11~v14).
           불탄 픽셀이 사건 점에서 BURN_MATCH_KM(1 km) 안이고 연소일이 사건 기간 ± BURN_DATE_PAD(3일) 안이면 그 사건 몫
           (여러 사건이면 가장 가까운 점의 사건 하나). burned_area_km2 = 픽셀 수 × 0.2146 km², burn_confirmed = 면적 > 0.
           MCD64A1 기간 밖(2012 이전, 2026) 사건은 null.
3) regime  AOI × 월 fire_regime ∈ {large_fire, many_small, mixed, none} → data/processed/aoi_month_fire_regime.parquet
           점의 0.1° 칸 → AOI 면적 가중으로 사건 몫(Σ harmonized_weight)을 나눠 AOI·월 합계 W를 만든다.
           큰 사건: 사건 전체 Σ harmonized_weight ≥ LARGE_WEIGHT(100, VIIRS 상당 감지 수) 또는 burned_area_km2 ≥ LARGE_BURN_KM2(10).
           share_large = 그 AOI·월 W 중 큰 사건 몫.
             none       W = 0 (그 달 감지 없음)
             large_fire share_large ≥ 0.5  (큰 사건 하나·몇 개가 그 달 활동의 절반 이상)
             many_small share_large < 0.2  (작은 사건 여럿이 대부분)
             mixed      그 사이
           근거: 감지 수 100(VIIRS 375 m)은 대략 수십 km² 규모의 연소이고, 의성·안동(2025-03)처럼 며칠 이어지는 산불은
           수천 감지가 된다. 한국의 일반 산불(산림청 평균 1 ha 내외)은 감지 1~5개라 이 기준에 걸리지 않는다.
4) validate 한국 2025-03 의성·안동 fire_regime, 산림청 산불통계(2022~2025)와 사건 매칭 탐지율 (면적 구간별, 큰 산불 ≥ 100 ha).

메모리: 점은 연 단위로 읽고, MCD64A1은 타일 × 연 단위로 처리해 결과를 캐시(--force로 다시). 최대 메모리 기록.
실행  uv run python -m planetarium.fire_events all        (또는 events / burned / regime / validate)
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm

from planetarium.config import (EVENT_DIST_M, EVENT_GAP_DAYS, EXTERNAL_DATA_DIR, FIRMS_MERGED_DIR, INTERIM_DATA_DIR,
                                PROCESSED_DATA_DIR, REPORTS_DIR)
from planetarium.runlog import peak_memory_gb, start_run_log

REF_DIR = PROCESSED_DATA_DIR / "harmonized_hotspots" / "reference"
EVENTS_PATH = PROCESSED_DATA_DIR / "fire_events.parquet"
REGIME_PATH = PROCESSED_DATA_DIR / "aoi_month_fire_regime.parquet"
META_PATH = PROCESSED_DATA_DIR / "fire_events_meta.json"
WORK_DIR = INTERIM_DATA_DIR / "fire_events"
MCD_DIR = EXTERNAL_DATA_DIR / "burned_area" / "mcd64a1"
KFS_DIR = EXTERNAL_DATA_DIR / "kfs_wildfire_stats"
REGIONS = {"KR": "KOR", "ARG": "ARG"}
BBOX = {"KR": (124.0, 33.0, 132.0, 39.0), "ARG": (-74.0, -56.0, -53.0, -21.0)}
TILES = {"KR": ["h27v05", "h28v05"], "ARG": ["h11v11", "h12v11", "h12v12", "h12v13", "h13v11", "h13v12", "h13v13", "h13v14", "h14v14"]}
YEARS = range(2000, 2027)
MCD_YEARS = range(2012, 2026)
KM_LAT = 110.574
KM_LON = 111.320
BURN_MATCH_KM = 1.0
BURN_DATE_PAD = 3
MCD_PIXEL_KM2 = 0.463312716528 ** 2
LARGE_WEIGHT = 100.0
LARGE_BURN_KM2 = 10.0
NOMINAL_PIXEL_KM2 = {"VIIRS": 0.375 ** 2, "MODIS": 1.0}


# ── 1) 사건 묶기 ─────────────────────────────────────────────────────────

def load_points(region: str, year: int) -> pl.DataFrame:
    from planetarium.static_exclusion import is_static_expr
    return (pl.scan_parquet(REF_DIR / f"year={year}" / "**" / "*.parquet", hive_partitioning=True)
            .filter(pl.col("adm0_a3") == REGIONS[region], ~is_static_expr())  # v1.2: 고정 열원 점 제외
            .select("latitude", "longitude", "acq_datetime_utc", "date_local", "satellite", "instrument", "frp",
                    "cell_id", "adm0_a3", "adm1_code", "harmonized_weight")
            .collect())


def pixel_area(pts: pl.DataFrame, region: str, year: int) -> pl.DataFrame:
    """원본 통합 파일에서 scan × track (km²)를 붙인다. 못 찾으면 센서 명목값(nadir)과 플래그."""
    x0, y0, x1, y1 = BBOX[region]
    keys = ["latitude", "longitude", "acq_datetime_utc", "satellite"]
    src = (pl.scan_parquet(sorted(FIRMS_MERGED_DIR.glob("*.parquet")))
           .filter(pl.col("acq_date").is_between(date(year, 1, 1) - timedelta(days=1), date(year, 12, 31) + timedelta(days=1)),
                   pl.col("latitude").is_between(y0, y1), pl.col("longitude").is_between(x0, x1))
           .select(*keys, px_km2=(pl.col("scan") * pl.col("track")).cast(pl.Float32))
           .unique(subset=keys).collect())
    out = pts.join(src, on=keys, how="left")
    return out.with_columns(px_nominal=pl.col("px_km2").is_null(),
                            px_km2=pl.coalesce("px_km2", pl.col("instrument").replace_strict(NOMINAL_PIXEL_KM2, default=1.0).cast(pl.Float32)))


def link_edges(pts: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """하루씩 (T일 전 ~ 오늘) 점의 가까운 쌍 → 그날 연결 성분 → (점, 성분 대표 점) 간선."""
    lat = pts["latitude"].to_numpy()
    xy = np.column_stack([pts["longitude"].to_numpy() * KM_LON * np.cos(np.radians(lat)), lat * KM_LAT])
    th = pts["instrument"].replace_strict(EVENT_DIST_M, default=max(EVENT_DIST_M.values())).to_numpy() / 1000.0
    rmax = max(EVENT_DIST_M.values()) / 1000.0
    day = (pts["date_local"].cast(pl.Int32)).to_numpy()
    days = np.unique(day)
    starts = np.searchsorted(day, days)
    ends = np.searchsorted(day, days, side="right")
    pos = {d: (s, e) for d, s, e in zip(days, starts, ends)}
    ei, ej = [], []
    for d in tqdm(days, desc="  일별 연결", unit="day", leave=False, dynamic_ncols=True):
        lo = min((pos[d - k][0] for k in range(EVENT_GAP_DAYS, 0, -1) if (d - k) in pos), default=pos[d][0])
        hi = pos[d][1]
        idx = np.arange(lo, hi)
        if len(idx) < 2:
            continue
        tree = cKDTree(xy[idx])
        pr = tree.query_pairs(rmax, output_type="ndarray")
        if len(pr) == 0:
            continue
        a, b = idx[pr[:, 0]], idx[pr[:, 1]]
        dist = np.hypot(*(xy[a] - xy[b]).T)
        ok = dist <= np.maximum(th[a], th[b])
        a, b = a[ok], b[ok]
        if len(a) == 0:
            continue
        # 그날 연결 성분 → 대표 점(성분의 첫 점)과의 간선만 (쌍 수가 점 수의 제곱으로 커지는 것을 막음)
        loc = {g: i for i, g in enumerate(np.unique(np.concatenate([a, b])))}
        nodes = np.fromiter(loc.keys(), dtype=np.int64)
        la = np.fromiter((loc[x] for x in a), dtype=np.int64, count=len(a))
        lb = np.fromiter((loc[x] for x in b), dtype=np.int64, count=len(b))
        g = coo_matrix((np.ones(len(la), dtype=np.int8), (la, lb)), shape=(len(nodes), len(nodes)))
        _, lab = connected_components(g, directed=False)
        first = np.full(lab.max() + 1, -1, dtype=np.int64)
        for i in range(len(lab)):
            if first[lab[i]] < 0:
                first[lab[i]] = i
        ei.append(nodes)
        ej.append(nodes[first[lab]])
    if not ei:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(ei), np.concatenate(ej)


def korea_sgg() -> pl.DataFrame:
    from planetarium.korea import KOREA_WEIGHTS_PATH
    w = pl.read_parquet(KOREA_WEIGHTS_PATH)
    return w.sort("weight", descending=True).unique("cell_id", keep="first").select("cell_id", "sgg_cd", "sgg_nm", "sido_nm")


def build_events(region: str) -> pl.DataFrame:
    t0 = time.perf_counter()
    parts = []
    for y in tqdm(YEARS, desc=f"[{region}] 점 읽기", unit="year", dynamic_ncols=True):
        p = load_points(region, y)
        if p.height:
            parts.append(pixel_area(p, region, y))
    pts = pl.concat(parts).sort("date_local", "acq_datetime_utc").with_row_index("pid")
    del parts
    ei, ej = link_edges(pts)
    n = pts.height
    g = coo_matrix((np.ones(len(ei), dtype=np.int8), (ei, ej)), shape=(n, n))
    ncomp, comp = connected_components(g, directed=False)
    pts = pts.with_columns(comp=pl.Series(comp, dtype=pl.Int64))
    if region == "KR":
        pts = pts.join(korea_sgg(), on="cell_id", how="left")
    else:
        pts = pts.with_columns(sgg_cd=pl.lit(None, dtype=pl.String), sgg_nm=pl.lit(None, dtype=pl.String), sido_nm=pl.lit(None, dtype=pl.String))
    w = pl.col("harmonized_weight")
    ev = pts.group_by("comp").agg(
        start_date=pl.col("date_local").min(), end_date=pl.col("date_local").max(),
        n_det=pl.len().cast(pl.Int32), sum_weight=w.sum(),
        frp_sum=pl.col("frp").sum(), frp_max=pl.col("frp").max(),
        centroid_lat=pl.col("latitude").mean(), centroid_lon=pl.col("longitude").mean(),
        lat_min=pl.col("latitude").min(), lat_max=pl.col("latitude").max(),
        lon_min=pl.col("longitude").min(), lon_max=pl.col("longitude").max(),
        pixel_area_km2=pl.col("px_km2").sum(), pixel_area_nominal_share=pl.col("px_nominal").mean(),
        n_viirs=(pl.col("instrument") == "VIIRS").sum().cast(pl.Int32), n_modis=(pl.col("instrument") == "MODIS").sum().cast(pl.Int32),
        country=pl.col("adm0_a3").mode().first(),
        adm1_code=pl.col("adm1_code").drop_nulls().mode().first(),
        sgg_cd=pl.col("sgg_cd").drop_nulls().mode().first(), sgg_nm=pl.col("sgg_nm").drop_nulls().mode().first(),
        sido_nm=pl.col("sido_nm").drop_nulls().mode().first(),
        sgg_all=pl.col("sgg_nm").drop_nulls().unique().sort().str.join(","),
        first_pid=pl.col("pid").min(),
    ).sort("start_date", "first_pid")
    ev = ev.with_columns(
        event_id=pl.format("{}_{}_{}", pl.lit(region), pl.col("start_date").dt.strftime("%Y%m%d"),
                           (pl.int_range(pl.len()).over("start_date") + 1).cast(pl.String).str.zfill(4)),
        region=pl.lit(region),
        duration_days=((pl.col("end_date") - pl.col("start_date")).dt.total_days() + 1).cast(pl.Int32),
        instrument=pl.when(pl.col("n_modis") == 0).then(pl.lit("VIIRS")).when(pl.col("n_viirs") == 0).then(pl.lit("MODIS")).otherwise(pl.lit("mixed")),
    )
    # 점 → 사건 (burned·regime 단계용, 작업 폴더)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    pts.join(ev.select("comp", "event_id"), on="comp").select(
        "event_id", "latitude", "longitude", "date_local", "instrument", "cell_id", "harmonized_weight").write_parquet(
        WORK_DIR / f"points_{region}.parquet", compression="zstd")
    print(f"[{region}] 점 {n:,} → 사건 {ncomp:,} (간선 {len(ei):,}), 픽셀 면적 명목값 비율 {pts['px_nominal'].mean():.3%}, "
          f"{time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")
    return ev.drop("comp", "first_pid")


# ── 2) MCD64A1 불탄 면적 ─────────────────────────────────────────────────

def mcd_files(tile: str, year: int) -> list[Path]:
    return sorted((MCD_DIR / tile).glob(f"MCD64A1.A{year}*.{tile}.061.*.hdf"))


def burned_pixels(path: Path, tile: str) -> pl.DataFrame:
    from pyhdf.SD import SD, SDC
    sd = SD(str(path), SDC.READ)
    bd = sd.select("Burn Date").get()
    sd.end()
    r, c = np.nonzero(bd > 0)
    if len(r) == 0:
        return pl.DataFrame(schema={"lat": pl.Float64, "lon": pl.Float64, "burn_date": pl.Date})
    year = int(re.search(r"\.A(\d{4})", path.name).group(1))
    h, v = int(tile[1:3]), int(tile[4:6])
    R, size = 6371007.181, 463.312716528
    x = -20015109.354 + (h * 2400 + c + 0.5) * size
    yy = 10007554.677 - (v * 2400 + r + 0.5) * size
    lat = np.degrees(yy / R)
    lon = np.degrees(x / (R * np.cos(np.radians(lat))))
    doy = bd[r, c].astype(np.int32)
    return pl.DataFrame({"lat": lat, "lon": lon}).with_columns(
        burn_date=(pl.lit(date(year, 1, 1)) + pl.duration(days=pl.Series(doy - 1))))


def burned_area(region: str, force: bool) -> pl.DataFrame:
    ev = pl.read_parquet(WORK_DIR / f"events_{region}.parquet").select("event_id", "start_date", "end_date")
    pts = pl.read_parquet(WORK_DIR / f"points_{region}.parquet").join(ev, on="event_id")
    out = []
    for tile in TILES[region]:
        for y in tqdm(MCD_YEARS, desc=f"[{region}] MCD64A1 {tile}", unit="year", leave=False, dynamic_ncols=True):
            cache = WORK_DIR / f"burned_{region}_{tile}_{y}.parquet"
            if cache.exists() and not force:
                out.append(pl.read_parquet(cache))
                continue
            rows = []
            for f in mcd_files(tile, y):
                px = burned_pixels(f, tile)
                if px.height == 0:
                    continue
                d0, d1 = px["burn_date"].min(), px["burn_date"].max()
                cand = pts.filter(pl.col("end_date") >= d0 - timedelta(days=BURN_DATE_PAD),
                                  pl.col("start_date") <= d1 + timedelta(days=BURN_DATE_PAD))
                if cand.height == 0:
                    continue
                clat = cand["latitude"].to_numpy()
                cxy = np.column_stack([cand["longitude"].to_numpy() * KM_LON * np.cos(np.radians(clat)), clat * KM_LAT])
                plat = px["lat"].to_numpy()
                pxy = np.column_stack([px["lon"].to_numpy() * KM_LON * np.cos(np.radians(plat)), plat * KM_LAT])
                k = min(8, cand.height)
                dist, idx = cKDTree(cxy).query(pxy, k=k, distance_upper_bound=BURN_MATCH_KM)
                if k == 1:
                    dist, idx = dist[:, None], idx[:, None]
                bdn = px["burn_date"].cast(pl.Int32).to_numpy()
                s = cand["start_date"].cast(pl.Int32).to_numpy()
                e = cand["end_date"].cast(pl.Int32).to_numpy()
                eid = cand["event_id"].to_numpy()
                chosen = np.full(len(pxy), -1, dtype=np.int64)
                for j in range(k):  # 가까운 순서로, 기간 조건을 만족하는 첫 점
                    ij = idx[:, j]
                    valid = (chosen < 0) & np.isfinite(dist[:, j]) & (ij < cand.height)
                    ijc = np.where(valid, ij, 0)
                    okd = valid & (bdn >= s[ijc] - BURN_DATE_PAD) & (bdn <= e[ijc] + BURN_DATE_PAD)
                    chosen[okd] = ijc[okd]
                m = chosen >= 0
                if m.any():
                    rows.append(pl.DataFrame({"event_id": eid[chosen[m]]}).group_by("event_id").agg(pixels=pl.len()))
            res = (pl.concat(rows).group_by("event_id").agg(pl.col("pixels").sum()) if rows
                   else pl.DataFrame(schema={"event_id": pl.String, "pixels": pl.UInt32}))
            res.write_parquet(cache)
            out.append(res)
    b = pl.concat(out).group_by("event_id").agg(pl.col("pixels").sum())
    return b.with_columns(burned_area_km2=(pl.col("pixels") * MCD_PIXEL_KM2).cast(pl.Float64)).drop("pixels")


# ── 2b) 큰 사건: 볼록 껍질 + 버퍼 ─────────────────────────────────────────
# 1 km 근접 매칭은 위성이 하루 두 번만 지나가 화선 사이 연소 면적이 빠진다(의성 2025-03: MCD64A1 964 km² 중 492 km²만 붙음).
# 큰 사건(Σ가중치 ≥ HULL_MIN_WEIGHT 또는 감지 ≥ HULL_MIN_DET)은 감지 점 볼록 껍질 + HULL_BUFFER_KM 안의 불탄 픽셀 중
# 연소일이 사건 기간 ± HULL_DATE_PAD일 안인 것까지 붙인다. 작은 사건은 기존 근접 규칙(1 km, ±3일).
# 한 픽셀이 여러 사건 후보면 그 픽셀에서 가장 가까운 감지 점의 사건 하나에만 배정(중복 없음).
HULL_MIN_WEIGHT = 100.0
HULL_MIN_DET = 50
HULL_BUFFER_KM = 2.0
HULL_DATE_PAD = 31


def burned_area_hull(region: str, force: bool) -> pl.DataFrame:
    import shapely
    ev = pl.read_parquet(WORK_DIR / f"events_{region}.parquet").select("event_id", "start_date", "end_date", "sum_weight", "n_det")
    pts = pl.read_parquet(WORK_DIR / f"points_{region}.parquet").join(ev, on="event_id")
    lat0 = pts["latitude"].to_numpy()
    pts = pts.with_columns(x=pl.Series(pts["longitude"].to_numpy() * KM_LON * np.cos(np.radians(lat0))), y=pl.Series(lat0 * KM_LAT))
    big = ev.filter((pl.col("sum_weight") >= HULL_MIN_WEIGHT) | (pl.col("n_det") >= HULL_MIN_DET))
    bp = pts.join(big.select("event_id"), on="event_id", how="semi")
    hulls = {}
    for (eid,), g in bp.group_by("event_id"):
        xy = np.column_stack([g["x"].to_numpy(), g["y"].to_numpy()])
        hulls[eid] = (shapely.convex_hull(shapely.multipoints(xy)).buffer(HULL_BUFFER_KM), cKDTree(xy))
    big_ids = list(hulls)
    epoch = date(1970, 1, 1)
    big_win = {r["event_id"]: ((r["start_date"] - epoch).days, (r["end_date"] - epoch).days) for r in big.to_dicts()}
    tree = shapely.STRtree([hulls[e][0] for e in big_ids]) if big_ids else None
    out = []
    for tile in TILES[region]:
        for y in tqdm(MCD_YEARS, desc=f"[{region}] 껍질 {tile}", unit="year", leave=False, dynamic_ncols=True):
            cache = WORK_DIR / f"burned2_{region}_{tile}_{y}.parquet"
            if cache.exists() and not force:
                out.append(pl.read_parquet(cache))
                continue
            rows = []
            for f in mcd_files(tile, y):
                px = burned_pixels(f, tile)
                if px.height == 0:
                    continue
                plat = px["lat"].to_numpy()
                pxy = np.column_stack([px["lon"].to_numpy() * KM_LON * np.cos(np.radians(plat)), plat * KM_LAT])
                bdn = px["burn_date"].cast(pl.Int32).to_numpy()
                best_d = np.full(len(pxy), np.inf)
                best_e = np.full(len(pxy), "", dtype=object)
                # (a) 근접 규칙: 모든 사건, 감지 점 1 km 안, 기간 ±3일
                d0, d1 = px["burn_date"].min(), px["burn_date"].max()
                cand = pts.filter(pl.col("end_date") >= d0 - timedelta(days=BURN_DATE_PAD),
                                  pl.col("start_date") <= d1 + timedelta(days=BURN_DATE_PAD))
                if cand.height:
                    k = min(8, cand.height)
                    dist, idx = cKDTree(np.column_stack([cand["x"].to_numpy(), cand["y"].to_numpy()])).query(
                        pxy, k=k, distance_upper_bound=BURN_MATCH_KM)
                    if k == 1:
                        dist, idx = dist[:, None], idx[:, None]
                    s = cand["start_date"].cast(pl.Int32).to_numpy()
                    e = cand["end_date"].cast(pl.Int32).to_numpy()
                    eid = cand["event_id"].to_numpy()
                    for j in range(k):
                        ij = idx[:, j]
                        valid = np.isfinite(dist[:, j]) & (ij < cand.height)
                        ijc = np.where(valid, ij, 0)
                        okd = valid & (bdn >= s[ijc] - BURN_DATE_PAD) & (bdn <= e[ijc] + BURN_DATE_PAD) & (dist[:, j] < best_d)
                        best_d[okd] = dist[okd, j]
                        best_e[okd] = eid[ijc[okd]]
                # (b) 큰 사건: 볼록 껍질 + 2 km, 기간 ±31일 → 가장 가까운 감지 점까지 거리로 경쟁
                if tree is not None:
                    gi, ti = tree.query(shapely.points(pxy), predicate="intersects")
                    if len(gi):
                        pair = pl.DataFrame({"pix": gi, "ev": [big_ids[i] for i in ti]})
                        for (e_id,), g in pair.group_by("ev"):
                            s0, e0 = big_win[e_id]
                            pix = g["pix"].to_numpy()
                            pix = pix[(bdn[pix] >= s0 - HULL_DATE_PAD) & (bdn[pix] <= e0 + HULL_DATE_PAD)]
                            if len(pix) == 0:
                                continue
                            dd, _ = hulls[e_id][1].query(pxy[pix])
                            better = dd < best_d[pix]
                            best_d[pix[better]] = dd[better]
                            best_e[pix[better]] = e_id
                m = best_e != ""
                if m.any():
                    rows.append(pl.DataFrame({"event_id": best_e[m].astype(str)}).group_by("event_id").agg(pixels=pl.len()))
            res = (pl.concat(rows).group_by("event_id").agg(pl.col("pixels").sum()) if rows
                   else pl.DataFrame(schema={"event_id": pl.String, "pixels": pl.UInt32}))
            res.write_parquet(cache)
            out.append(res)
    b = pl.concat(out).group_by("event_id").agg(pl.col("pixels").sum())
    return b.with_columns(burned_area_km2=(pl.col("pixels") * MCD_PIXEL_KM2).cast(pl.Float64)).drop("pixels")


# ── 3) AOI × 월 화재 양상 ─────────────────────────────────────────────────

# ── 1b) 행정구역: 감지 점 위치로 (0.1° 칸 대표값 대신) ─────────────────────
# 0.1° 칸 대표 시군구는 경계 근처 점을 옆 시군구로 보낸다(예: 봉화읍 안 감지 → 영주시, 서산 운산면 → 예산군).
# 점마다 경계 도형 안에 드는지(point-in-polygon)로 정하고, 도형 밖(해안·경계 오차)은 ADMIN_SNAP_DEG 안의 가장 가까운 도형으로.
# 사건 속성은 harmonized_weight 가중 다수결, sgg_all은 점이 하나라도 있는 시군구 전부.
ADMIN_SNAP_DEG = 0.02  # 약 2 km


def _assign(lon: np.ndarray, lat: np.ndarray, geoms: list, codes: list) -> np.ndarray:
    import shapely
    tree = shapely.STRtree(geoms)
    pts = shapely.points(lon, lat)
    out = np.full(len(lon), None, dtype=object)
    pi, gi = tree.query(pts, predicate="within")
    out[pi] = np.asarray(codes, dtype=object)[gi]
    miss = np.nonzero(out == None)[0]  # noqa: E711
    if len(miss):
        mi, mg = tree.query_nearest(pts[miss], max_distance=ADMIN_SNAP_DEG)
        out[miss[mi]] = np.asarray(codes, dtype=object)[mg]
    return out


def point_admin(region: str) -> pl.DataFrame:
    """points_<region> + 점 위치 기준 adm0·adm1(NE)와 한국 시도·시군구(admdongkor 2025-12, AOI와 같은 판)."""
    import pyarrow.parquet as pq
    import shapely

    from planetarium.korea import ADM_DIR, TO_4326
    from planetarium.world import load_ne
    pts = pl.read_parquet(WORK_DIR / f"points_{region}.parquet")
    lon, lat = pts["longitude"].to_numpy(), pts["latitude"].to_numpy()
    units, geoms, _ = load_ne()
    x0, y0, x1, y1 = BBOX[region]
    sel = [i for i, g in enumerate(geoms) if shapely.intersects(g, shapely.box(x0 - 1, y0 - 1, x1 + 1, y1 + 1))]
    a1 = _assign(lon, lat, [geoms[i] for i in sel], [units["adm1_code"][i] for i in sel])
    a1_to_a0 = dict(zip(units["adm1_code"], units["adm0_a3"]))
    pts = pts.with_columns(adm1_pt=pl.Series(a1, dtype=pl.String)).with_columns(
        adm0_pt=pl.col("adm1_pt").replace_strict(a1_to_a0, default=None))
    if region == "KR":
        for name, col, nm, out_cd, out_nm in (("sido_20251231", "sidocd", "sidonm", "sido_cd_pt", "sido_nm_pt"),
                                               ("sgg_20251231", "sggcd", "sggnm", "sgg_cd_pt", "sgg_nm_pt")):
            t = pq.read_table(ADM_DIR / f"{name}.parquet")
            g = [shapely.make_valid(shapely.transform(x, lambda xy: np.column_stack(TO_4326.transform(xy[:, 0], xy[:, 1]))))
                 for x in shapely.from_wkb(t.column("geom").to_pylist())]
            cd = _assign(lon, lat, g, t.column(col).to_pylist())
            names = dict(zip(t.column(col).to_pylist(), t.column(nm).to_pylist()))
            pts = pts.with_columns(pl.Series(out_cd, cd, dtype=pl.String)).with_columns(
                pl.col(out_cd).replace_strict(names, default=None).alias(out_nm))
        # 한국 점은 시군구 도형 안이면 KOR (NE 국경과 admdongkor 해안선 차이 보정)
        pts = pts.with_columns(adm0_pt=pl.when(pl.col("sgg_cd_pt").is_not_null()).then(pl.lit("KOR")).otherwise("adm0_pt"))
    else:
        pts = pts.with_columns(*[pl.lit(None, dtype=pl.String).alias(c) for c in ("sido_cd_pt", "sido_nm_pt", "sgg_cd_pt", "sgg_nm_pt")])
    pts.write_parquet(WORK_DIR / f"points_admin_{region}.parquet", compression="zstd")
    return pts


def wmode(col: str) -> pl.Expr:
    """harmonized_weight 가중 다수결 (그룹 안에서)."""
    return pl.col(col).filter(pl.col(col).is_not_null()).sort_by(
        pl.col("harmonized_weight").filter(pl.col(col).is_not_null()).sum().over(col), descending=True).first()


def event_admin(region: str) -> pl.DataFrame:
    pts = point_admin(region)
    # 같은 사건·같은 값의 가중치 합 → 최대 값
    def best(col: str, alias: str) -> pl.DataFrame:
        return (pts.filter(pl.col(col).is_not_null()).group_by("event_id", col).agg(w=pl.col("harmonized_weight").sum())
                .sort("w", descending=True).unique("event_id", keep="first").select("event_id", pl.col(col).alias(alias)))
    out = pts.select("event_id").unique()
    for col, alias in (("adm0_pt", "country"), ("adm1_pt", "adm1_code"), ("sido_nm_pt", "sido_nm"), ("sgg_cd_pt", "sgg_cd"), ("sgg_nm_pt", "sgg_nm")):
        out = out.join(best(col, alias), on="event_id", how="left")
    allsgg = pts.filter(pl.col("sgg_nm_pt").is_not_null()).group_by("event_id").agg(
        sgg_all=pl.col("sgg_nm_pt").unique().sort().str.join(","))
    return out.join(allsgg, on="event_id", how="left")


def regime(events: pl.DataFrame) -> pl.DataFrame:
    """점 위치로 AOI를 정한다: KR(시군구 도형 안), SIDO_<cd>, SGG_<cd>, ADM0_ARG(adm0_pt), ADM1_<adm1_code>(아르헨티나)."""
    from planetarium.harmonize import aoi_membership
    _, aoi = aoi_membership()
    keep = aoi.filter(pl.col("level").is_in(["kr", "sido", "sgg"]) | (pl.col("aoi_id") == "ADM0_ARG")
                      | ((pl.col("level") == "adm1") & (pl.col("parent_aoi") == "ADM0_ARG")))
    large = events.select("event_id", is_large=(pl.col("sum_weight") >= LARGE_WEIGHT)
                          | (pl.col("burned_area_km2").fill_null(0) >= LARGE_BURN_KM2))
    parts = []
    for region in REGIONS:
        p = (pl.read_parquet(WORK_DIR / f"points_admin_{region}.parquet")
             .with_columns(year=pl.col("date_local").dt.year().cast(pl.Int16), month=pl.col("date_local").dt.month().cast(pl.Int8)))
        if region == "KR":
            a = pl.concat([
                p.filter(pl.col("sgg_cd_pt").is_not_null()).with_columns(aoi_id=pl.lit("KR")),
                p.filter(pl.col("sido_cd_pt").is_not_null()).with_columns(aoi_id=pl.concat_str(pl.lit("SIDO_"), "sido_cd_pt")),
                p.filter(pl.col("sgg_cd_pt").is_not_null()).with_columns(aoi_id=pl.concat_str(pl.lit("SGG_"), "sgg_cd_pt")),
            ])
        else:
            a = pl.concat([
                p.filter(pl.col("adm0_pt") == "ARG").with_columns(aoi_id=pl.lit("ADM0_ARG")),
                p.filter(pl.col("adm0_pt") == "ARG").with_columns(aoi_id=pl.concat_str(pl.lit("ADM1_"), "adm1_pt")),
            ])
        a = a.join(keep.select("aoi_id"), on="aoi_id", how="semi").join(large, on="event_id")
        parts.append(a.group_by("aoi_id", "year", "month", "event_id").agg(
            w=pl.col("harmonized_weight").sum(), is_large=pl.col("is_large").first()))
    em = pl.concat(parts)
    top = em.sort("w", descending=True).unique(["aoi_id", "year", "month"], keep="first").select(
        "aoi_id", "year", "month", top_event_id="event_id", top_event_w="w")
    agg = em.group_by("aoi_id", "year", "month").agg(
        W=pl.col("w").sum(), W_large=pl.col("w").filter(pl.col("is_large")).sum(),
        n_events=pl.len().cast(pl.Int32), n_large_events=pl.col("is_large").sum().cast(pl.Int32)).join(top, on=["aoi_id", "year", "month"])
    months = pl.DataFrame({"ym": list(range(2000 * 12 + 10, 2026 * 12 + 11))}).select(
        year=(pl.col("ym") // 12).cast(pl.Int16), month=(pl.col("ym") % 12 + 1).cast(pl.Int8))
    full = keep.select("aoi_id", "name", "level").join(months, how="cross").join(agg, on=["aoi_id", "year", "month"], how="left")
    share = pl.col("W_large") / pl.col("W")
    return full.with_columns(
        pl.col("W", "W_large").fill_null(0.0), pl.col("n_events", "n_large_events").fill_null(0),
    ).with_columns(
        share_large=pl.when(pl.col("W") > 0).then(share),
        fire_regime=pl.when(pl.col("W") <= 0).then(pl.lit("none"))
        .when(share >= 0.5).then(pl.lit("large_fire"))
        .when(share < 0.2).then(pl.lit("many_small"))
        .otherwise(pl.lit("mixed")),
    ).sort("aoi_id", "year", "month")


# ── 4) 검증 ──────────────────────────────────────────────────────────────

SIDO_ABBR = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시",
             "울산": "울산광역시", "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
             "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도"}


def load_kfs() -> pl.DataFrame:
    parts = []
    for f in sorted(KFS_DIR.glob("*.csv")):
        for enc in ("cp949", "utf-8"):
            try:
                d = pl.read_csv(f, encoding=enc, infer_schema_length=0)
                break
            except Exception:
                d = None
        if d is not None:
            parts.append(d)
    k = pl.concat(parts, how="diagonal_relaxed").unique()
    num = lambda c: pl.col(c).str.strip_chars().cast(pl.Int32, strict=False)  # noqa: E731
    k = k.with_columns(
        start=pl.date(num("발생일시_년"), num("발생일시_월"), num("발생일시_일")),
        end=pl.date(num("진화종료시간_년"), num("진화종료시간_월"), num("진화종료시간_일")),
        area_ha=pl.col("피해면적_합계").str.replace_all(",", "").cast(pl.Float64, strict=False),
        sido=pl.col("발생장소_시도").str.strip_chars(), sgg=pl.col("발생장소_시군구").str.strip_chars(),
    ).with_columns(end=pl.coalesce("end", "start"))
    return k.select("start", "end", "area_ha", "sido", "sgg", "발생장소_읍면", "발생장소_동리").unique()


def validate(events: pl.DataFrame, reg: pl.DataFrame) -> dict:
    out = {}
    # 의성·안동 2025-03
    names = reg.filter(pl.col("year") == 2025, pl.col("month") == 3,
                       pl.col("name").is_in(["의성군", "안동시", "청송군", "영양군", "영덕군"]) | (pl.col("aoi_id") == "KR"))
    out["uiseong_2025_03"] = names.select("aoi_id", "name", "fire_regime", "W", "share_large", "n_events", "top_event_id").to_dicts()
    kr = events.filter(pl.col("region") == "KR", pl.col("start_date").is_between(date(2025, 3, 15), date(2025, 3, 31)))
    out["kr_2025_03_top_events"] = kr.sort("sum_weight", descending=True).head(5).select(
        "event_id", "start_date", "end_date", "duration_days", "n_det", "sum_weight", "burned_area_km2", "sgg_all").to_dicts()
    # 산림청 산불통계 매칭
    kfs = load_kfs().filter(pl.col("start").is_not_null())
    ev = events.filter(pl.col("region") == "KR", pl.col("end_date") >= kfs["start"].min() - timedelta(days=1)).select(
        "event_id", "start_date", "end_date", "sum_weight", "sgg_all")
    ev_sgg = ev.with_columns(s=pl.col("sgg_all").str.split(",")).explode("s").filter(pl.col("s") != "")
    k = kfs.with_row_index("kid").with_columns(sido_full=pl.col("sido").replace(SIDO_ABBR))
    j = (k.join(ev_sgg, how="cross")
         .filter(pl.col("s").str.starts_with(pl.col("sgg")),
                 pl.col("start_date") <= pl.col("end") + pl.duration(days=1),
                 pl.col("end_date") >= pl.col("start") - pl.duration(days=1)))
    hit = j.group_by("kid").agg(n_events=pl.col("event_id").n_unique(), best_event=pl.col("event_id").sort_by("sum_weight").last())
    k = k.join(hit, on="kid", how="left").with_columns(detected=pl.col("n_events").is_not_null())
    size = (pl.when(pl.col("area_ha") >= 100).then(pl.lit("4) >=100ha (large)"))
            .when(pl.col("area_ha") >= 10).then(pl.lit("3) 10-100ha")).when(pl.col("area_ha") >= 1).then(pl.lit("2) 1-10ha"))
            .otherwise(pl.lit("1) <1ha")))
    rate = k.with_columns(size=size).group_by("size").agg(n=pl.len(), detected=pl.col("detected").sum(),
                                                          rate=pl.col("detected").mean()).sort("size")
    rate.write_csv(REPORTS_DIR / "fire_events_kfs_detection.csv")
    k.filter(pl.col("area_ha") >= 100).sort("start").write_csv(REPORTS_DIR / "fire_events_kfs_large.csv")
    out["kfs_detection_by_size"] = rate.to_dicts()
    out["kfs_period"] = [str(kfs["start"].min()), str(kfs["start"].max())]
    return out


# ── 실행 ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="화재 사건 묶기 + 불탄 면적 + 화재 양상")
    ap.add_argument("step", choices=["events", "burned", "burned2", "admin", "regime", "validate", "all"])
    ap.add_argument("--regions", nargs="+", default=list(REGIONS))
    ap.add_argument("--force", action="store_true", help="이미 만든 작업 파일도 다시 (기본: 있으면 이어서 = --resume)")
    ap.add_argument("--resume", action="store_true", help="기본 동작")
    args = ap.parse_args(argv)
    start_run_log("fire_events")
    t0 = time.perf_counter()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    steps = ["events", "burned", "burned2", "admin", "regime", "validate"] if args.step == "all" else [args.step]
    if "events" in steps:
        for r in args.regions:
            p = WORK_DIR / f"events_{r}.parquet"
            if p.exists() and not args.force:
                print(f"[있음] {p.name}")
                continue
            build_events(r).write_parquet(p, compression="zstd")
    if "burned" in steps:
        allev = []
        for r in args.regions:
            ev = pl.read_parquet(WORK_DIR / f"events_{r}.parquet")
            b = burned_area(r, args.force)
            in_mcd = pl.col("end_date").dt.year().is_between(MCD_YEARS.start, MCD_YEARS.stop - 1)
            ev = ev.join(b, on="event_id", how="left").with_columns(
                burned_area_km2=pl.when(in_mcd).then(pl.col("burned_area_km2").fill_null(0.0)),
            ).with_columns(burn_confirmed=pl.when(in_mcd).then(pl.col("burned_area_km2") > 0))
            allev.append(ev)
            print(f"[{r}] MCD64A1 기간 사건 {ev.filter(in_mcd).height:,}, 불탄 면적 확인 {ev['burn_confirmed'].sum():,}, "
                  f"불탄 면적 합 {ev['burned_area_km2'].sum():,.0f} km², 최대 메모리 {peak_memory_gb() or 0:.2f} GB")
        cols = ["event_id", "region", "start_date", "end_date", "duration_days", "n_det", "sum_weight", "frp_sum", "frp_max",
                "centroid_lat", "centroid_lon", "lat_min", "lat_max", "lon_min", "lon_max", "pixel_area_km2",
                "pixel_area_nominal_share", "burned_area_km2", "burn_confirmed", "instrument", "n_viirs", "n_modis",
                "country", "adm1_code", "sido_nm", "sgg_cd", "sgg_nm", "sgg_all"]
        pl.concat(allev, how="diagonal_relaxed").select(cols).write_parquet(EVENTS_PATH, compression="zstd")
    if "burned2" in steps:
        ev = pl.read_parquet(EVENTS_PATH)
        if "burned_area_km2_near" not in ev.columns:
            ev = ev.rename({"burned_area_km2": "burned_area_km2_near"})
        ev = ev.drop([c for c in ("burned_area_km2", "burn_confirmed") if c in ev.columns])
        b2 = pl.concat([burned_area_hull(r, args.force) for r in args.regions])
        in_mcd = pl.col("end_date").dt.year().is_between(MCD_YEARS.start, MCD_YEARS.stop - 1)
        ev = ev.join(b2, on="event_id", how="left").with_columns(
            burned_area_km2=pl.when(in_mcd).then(pl.col("burned_area_km2").fill_null(0.0))).with_columns(
            burn_confirmed=pl.when(in_mcd).then(pl.col("burned_area_km2") > 0))
        cols = [c for c in ev.columns if c not in ("burned_area_km2", "burn_confirmed")]
        i = cols.index("burned_area_km2_near")
        cols[i:i] = ["burned_area_km2", "burn_confirmed"]
        ev.select(cols).write_parquet(EVENTS_PATH, compression="zstd")
        for r in args.regions:
            e = ev.filter(pl.col("region") == r)
            print(f"[{r}] 불탄 면적 합: 근접 규칙 {e['burned_area_km2_near'].sum():,.0f} km² → 껍질 규칙 {e['burned_area_km2'].sum():,.0f} km², "
                  f"확인 사건 {e['burn_confirmed'].sum():,}, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")
    if "admin" in steps:
        ev = pl.read_parquet(EVENTS_PATH)
        cols = ev.columns
        old = ev.select("event_id", "sgg_nm", "country")
        adm = pl.concat([event_admin(r) for r in args.regions], how="diagonal_relaxed")
        ev = ev.drop("country", "adm1_code", "sido_nm", "sgg_cd", "sgg_nm", "sgg_all").join(adm, on="event_id", how="left").select(cols)
        ev.write_parquet(EVENTS_PATH, compression="zstd")
        ch = ev.select("event_id", "sgg_nm", "country").join(old, on="event_id", suffix="_old")
        print(f"[행정구역] 점 위치 기준으로 바꿈: 시군구가 바뀐 사건 {(ch['sgg_nm'] != ch['sgg_nm_old']).sum():,}, "
              f"국가가 바뀐 사건 {(ch['country'] != ch['country_old']).sum():,} (전체 {ev.height:,}), 최대 메모리 {peak_memory_gb() or 0:.2f} GB")
    events = pl.read_parquet(EVENTS_PATH) if EVENTS_PATH.exists() else None
    if "regime" in steps:
        reg = regime(events)
        reg.write_parquet(REGIME_PATH, compression="zstd")
        print(reg.filter(pl.col("aoi_id").is_in(["KR", "ADM0_ARG"])).group_by("aoi_id", "fire_regime").len().sort("aoi_id", "fire_regime"))
    if "validate" in steps:
        reg = pl.read_parquet(REGIME_PATH)
        v = validate(events, reg)
        summ = events.group_by("region", "instrument").agg(
            n_events=pl.len(), n_det=pl.col("n_det").sum(), median_det=pl.col("n_det").median(),
            share_single=(pl.col("n_det") == 1).mean(), max_duration=pl.col("duration_days").max(),
            burn_confirmed_rate=pl.col("burn_confirmed").mean(),
            large=((pl.col("sum_weight") >= LARGE_WEIGHT) | (pl.col("burned_area_km2").fill_null(0) >= LARGE_BURN_KM2)).sum())
        summ.write_csv(REPORTS_DIR / "fire_events_summary.csv")
        meta = {
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "linking": {"dist_m": EVENT_DIST_M, "gap_days": EVENT_GAP_DAYS,
                        "rule": "두 점 거리 ≤ 두 센서 중 큰 D, 현지 날짜 차 ≤ T → 같은 사건 (연결 성분)"},
            "burned_area": {"product": "MCD64A1 v061 Burn Date", "match_km": BURN_MATCH_KM, "date_pad_days": BURN_DATE_PAD,
                            "pixel_km2": MCD_PIXEL_KM2, "years": [MCD_YEARS.start, MCD_YEARS.stop - 1], "tiles": TILES},
            "fire_regime": {"large_event": f"sum_weight ≥ {LARGE_WEIGHT} 또는 burned_area_km2 ≥ {LARGE_BURN_KM2}",
                            "large_fire": "share_large ≥ 0.5", "many_small": "share_large < 0.2", "mixed": "0.2 ≤ share_large < 0.5",
                            "none": "W = 0", "W": "AOI·월의 Σ harmonized_weight (0.1° 칸 → AOI 면적 가중)",
                            "rationale": "감지 100(VIIRS 375 m)은 수십 km² 규모 연소, 한국 일반 산불(평균 약 1 ha)은 감지 1~5개."},
            "validation": v,
        }
        META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                       tbl_hide_dataframe_shape=True, float_precision=3, thousands_separator=","):
            print(summ)
            print(pl.DataFrame(v["uiseong_2025_03"]))
            print(pl.DataFrame(v["kr_2025_03_top_events"]))
            print(pl.DataFrame(v["kfs_detection_by_size"]))
    print(f"[완료] {args.step} {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
