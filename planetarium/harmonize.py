"""T6: 최종 통합 시계열 (VIIRS 환산 감지 수 = 'S-NPP가 봤다면 셌을 핫스팟 수').

기간별 값 (value_source)
  2000-11 ~ 2002-06  modis_harmonized_terra_only  M1 최종 방법(지금 B1)의 Terra 단독 변형 × Terra (공통 정상일 환산)
  2002-07 ~ 2012-01  modis_harmonized             M1 최종 방법 × (Aqua+Terra). 2012-01은 S-NPP가 01-20에 시작한 부분 월이라
                                                   MODIS 보정 값을 쓴다 (meta에 기록).
  2012-02 ~ 2026-10  snpp_observed                S-NPP 실측: S-NPP status=ok인 날의 감지 수 × days_in_month / ok일수
  2026-11            noaa20_bridged               S-NPP(11-01~02, 마지막 감지 11-02 15:16 UTC) + NOAA-20(11-03~) × M2 비율,
                                                   ok일수로 월 환산. 진행 중인 달이라 status=partial.
  감지 수는 실제 산불 건수가 아니라 '관측 활동'이다.

출력 1  data/processed/harmonized_cell_month.parquet
  키 cell_id, cell_lat, cell_lon, year, month
  value, value_lo, value_hi, raw_modis(Aqua+Terra 전체 일 감지 수), raw_viirs(그 기간 기준 VIIRS 실측, 없으면 null),
  value_source, data_source(SP/NRT), status(ok/nrt/partial/missing), method_version
  ※ 전 세계라 희소 표: value·raw_modis·raw_viirs가 모두 0인 칸·월은 행이 없다(= 0). 월 상태는 status와 aoi_month를 볼 것.
  ※ 구간: B1은 OOF 칸·월 y/예측 비율의 10·90 백분위수(예측 크기 구간별). 실측 구간은 value와 같다.

출력 2  data/processed/aoi_month.parquet  (AOI × 연 × 월, 빠짐없이)
  aoi_id, year, month, value, value_lo, value_hi, raw_modis, raw_viirs, value_source, data_source, status, n_cells
  AOI: WORLD, CONT_<대륙>, ADM0_<A3>(한국 제외), ADM1_<adm1_code>(한국 제외), KR, SIDO_<cd>, SGG_<cd>, USER_<n>(aoi.geojson)
  - 칸 배정: 국가·1급은 world_cells에서 육지 비율 ≥ 0.1이고 겹침이 가장 큰 곳, 한국은 korea_cells(시·도/시군구 2025-12 기준).
  - value_lo/hi는 칸 구간의 단순 합 (합계의 정확한 구간은 아님 — 보수적으로 넓다).
  - status: partial(진행 중인 달) > missing(기준 위성 자료가 그 달 일수의 50% 미만이거나, AOI가 그 달 자료 범위 밖) > nrt > ok.
    missing이면 value·value_lo·value_hi는 null (0과 결측 구분).
  data/processed/aoi_list.parquet        aoi_id, name, level, parent_aoi, n_cells
  data/processed/aoi_boundaries.geojson  앱 지도용 경계 (WGS84, 단순화)

실행  uv run python -m planetarium.harmonize
"""

from __future__ import annotations

import argparse
import calendar
import json
import time
from datetime import date

import numpy as np
import polars as pl
import shapely
from shapely.geometry import mapping, shape
from tqdm import tqdm

from planetarium.config import (
    EAST_ASIA_BBOX, EXTERNAL_DATA_DIR, REPORTS_DIR, NOAA20_BRIDGE_START, PROCESSED_DATA_DIR, REGIONAL_ONLY_PERIODS,
)
from planetarium.korea import ADM_DIR, KOREA_CELLS_PATH, TO_4326
from planetarium.modeling.m2 import M2_RATIOS_PATH
from planetarium.modeling.train_m1 import B1_INTERVAL_PATH, B1_RATIOS_PATH, META_PATH, apply_interval, b1_predict
from planetarium.monthly import CELL_MONTH_PATH, SAT_MONTH_PATH, STATUS_PATH
from planetarium.runlog import start_run_log
from planetarium.train_table import aggregate_common, load_panel, panel_months
from planetarium.world import NE_DIR, WORLD_CELLS_PATH, polygon_cells

HCM_PATH = PROCESSED_DATA_DIR / "harmonized_cell_month.parquet"
AOI_MONTH_PATH = PROCESSED_DATA_DIR / "aoi_month.parquet"
AOI_LIST_PATH = PROCESSED_DATA_DIR / "aoi_list.parquet"
AOI_GEOJSON_PATH = PROCESSED_DATA_DIR / "aoi_boundaries.geojson"
USER_AOI_PATH = EXTERNAL_DATA_DIR / "aoi.geojson"

START = (2000, 11)
SNPP_FIRST = (2012, 2)
BRIDGE_MONTH = (NOAA20_BRIDGE_START.year, NOAA20_BRIDGE_START.month)
N_LON = 3600
SIMPLIFY_DEG = 0.01


def ym(y: int, m: int) -> int:
    return y * 12 + m - 1


def last_data_day() -> date:
    return pl.read_parquet(STATUS_PATH).filter(pl.col("satellite") == "NOAA-20")["date_utc"].max()


def all_months() -> list[tuple[int, int]]:
    end = last_data_day()
    return [(k // 12, k % 12 + 1) for k in range(ym(*START), ym(end.year, end.month) + 1)]


# ── 월 단위 기준표 ────────────────────────────────────────────────────────

def ok_days(sat: str, start: date | None = None, end: date | None = None) -> pl.DataFrame:
    st = pl.read_parquet(STATUS_PATH).filter(pl.col("satellite") == sat, pl.col("status") == "ok")
    if start:
        st = st.filter(pl.col("date_utc") >= start)
    if end:
        st = st.filter(pl.col("date_utc") <= end)
    return st.select(pl.col("date_utc").alias("date_local"), year=pl.col("date_utc").dt.year().cast(pl.Int16),
                     month=pl.col("date_utc").dt.month().cast(pl.Int8))


def month_reference() -> pl.DataFrame:
    """(연, 월)별 value_source, 기준 위성 자료 비율, data_source, status, 자료 범위."""
    pm = panel_months().filter(pl.col("mode") == "apply")
    sm = pl.read_parquet(SAT_MONTH_PATH)
    snpp = {(r["year"], r["month"]): r for r in sm.filter(pl.col("satellite") == "S-NPP").to_dicts()}
    cm_src = (pl.scan_parquet(CELL_MONTH_PATH).filter(pl.col("satellite").is_in(["S-NPP", "NOAA-20"]))
              .group_by("satellite", "year", "month").agg(nrt=(pl.col("source") == "NRT").any()).collect())
    nrt = {(r["satellite"], r["year"], r["month"]): r["nrt"] for r in cm_src.to_dicts()}
    apply = {(r["year"], r["month"]): r for r in pm.to_dicts()}
    end = last_data_day()
    bridge_ok = ok_days("S-NPP", date(*BRIDGE_MONTH, 1), NOAA20_BRIDGE_START).height + \
        ok_days("NOAA-20", NOAA20_BRIDGE_START).height
    rows = []
    for y, m in all_months():
        dim = calendar.monthrange(y, m)[1]
        k = ym(y, m)
        if k < ym(*SNPP_FIRST):
            a = apply.get((y, m))
            src = "modis_harmonized_terra_only" if a and a["terra_only"] else "modis_harmonized"
            frac = (a["common_days"] / dim) if a else 0.0
            ds = "SP"
        elif k < ym(*BRIDGE_MONTH):
            src = "snpp_observed"
            r = snpp.get((y, m))
            frac = (r["obs_days"] / dim) if r else 0.0
            ds = "NRT" if nrt.get(("S-NPP", y, m)) else "SP"
        else:
            src = "noaa20_bridged"
            frac = bridge_ok / dim
            ds = "NRT"
        partial = (y, m) == (end.year, end.month) and end.day < dim
        status = "partial" if partial else ("missing" if frac < 0.5 else ("nrt" if ds == "NRT" else "ok"))
        # 자료가 동아시아만 있는 달 (config.REGIONAL_ONLY_PERIODS)
        regional = any(s <= date(y, m, 1) <= e and bbox == EAST_ASIA_BBOX for s, e, bbox in REGIONAL_ONLY_PERIODS)
        rows.append({"year": y, "month": m, "value_source": src, "ref_frac": frac, "data_source": ds,
                     "status": status, "regional_ea": regional})
    return pl.DataFrame(rows).with_columns(pl.col("year").cast(pl.Int16), pl.col("month").cast(pl.Int8))


# ── 칸 × 월 ───────────────────────────────────────────────────────────────

def m2_ratio_for_cells(cells: pl.DataFrame, month: int) -> pl.DataFrame:
    """칸별 NOAA-20 → S-NPP 비율: 한국·아르헨티나 → 그 지역, 아니면 대륙, 없으면 world (모두 대체 규칙 적용된 used_ratio)."""
    r = pl.read_parquet(M2_RATIOS_PATH).filter(pl.col("pair") == "S-NPP/NOAA-20", pl.col("month") == month)
    rr = dict(zip(r["region"], r["used_ratio"]))
    wc = pl.read_parquet(WORLD_CELLS_PATH, columns=["cell_id", "adm0_a3", "continent"])
    kc = pl.read_parquet(KOREA_CELLS_PATH, columns=["cell_id", "is_korea"])
    c = cells.join(wc, on="cell_id", how="left").join(kc, on="cell_id", how="left")
    reg = (pl.when(pl.col("is_korea").fill_null(False)).then(pl.lit("korea"))
           .when(pl.col("adm0_a3") == "ARG").then(pl.lit("argentina"))
           .otherwise(pl.concat_str(pl.lit("continent:"), pl.col("continent"))))
    return c.with_columns(region=reg).with_columns(
        m2=pl.col("region").replace_strict(rr, default=rr["world"], return_dtype=pl.Float64)).select("cell_id", "m2")


def harmonized_year(year: int, ref: pl.DataFrame, meta: dict) -> pl.DataFrame:
    keys = ["cell_id", "year", "month"]
    yref = ref.filter(pl.col("year") == year)
    # 원 감지 수 (전체 일)
    cm = pl.scan_parquet(CELL_MONTH_PATH).filter(pl.col("year") == year)
    raw_modis = cm.filter(pl.col("satellite").is_in(["Aqua", "Terra"])).group_by(keys).agg(
        raw_modis=pl.col("n_det").sum()).collect()
    raw_snpp = cm.filter(pl.col("satellite") == "S-NPP").group_by(keys).agg(raw_snpp=pl.col("n_det").sum()).collect()
    raw_n20 = cm.filter(pl.col("satellite") == "NOAA-20").group_by(keys).agg(raw_n20=pl.col("n_det").sum()).collect()
    parts = []

    # 1) MODIS 보정 (apply 패널)
    if year <= SNPP_FIRST[0]:  # apply 패널은 2000-11 ~ 2012-01
        ap = load_panel("apply", [year]).collect() if any(
            f for f in (PROCESSED_DATA_DIR / "panel_common_0.1deg").glob(f"apply_{year}.parquet")) else None
        if ap is not None and ap.height:
            ratios = pl.read_parquet(B1_RATIOS_PATH)
            iv = pl.read_parquet(B1_INTERVAL_PATH)
            pred = b1_predict(ap, ratios, pl.col("terra_only"))
            lo, hi = apply_interval(pred, iv)
            parts.append(ap.select(keys).with_columns(value=pred, value_lo=lo, value_hi=hi))

    # 2) S-NPP 실측 (ok일 × 월 환산)
    if year >= SNPP_FIRST[0]:
        d = ok_days("S-NPP").filter(pl.col("year") == year,
                                    (pl.col("year").cast(pl.Int32) * 12 + pl.col("month").cast(pl.Int32) - 1)
                                    .is_between(ym(*SNPP_FIRST), ym(*BRIDGE_MONTH) - 1))
        if d.height:
            nok = d.group_by("year", "month").agg(n_ok=pl.len())
            obs = aggregate_common(d, {"S-NPP": "snpp"}).select(keys + ["n_det"]).join(nok, on=["year", "month"])
            dim = pl.date(pl.col("year"), pl.col("month"), 1).dt.month_end().dt.day()
            v = (pl.col("n_det") * dim / pl.col("n_ok")).cast(pl.Float64)
            parts.append(obs.select(keys, value=v, value_lo=v, value_hi=v))

    # 3) 2026-11: S-NPP(11-01~02) + NOAA-20(11-03~) × M2
    if year == BRIDGE_MONTH[0]:
        s_days = ok_days("S-NPP", date(*BRIDGE_MONTH, 1), NOAA20_BRIDGE_START)
        n_days = ok_days("NOAA-20", NOAA20_BRIDGE_START)
        n_ok = s_days.height + n_days.height
        if n_ok:
            s = aggregate_common(s_days, {"S-NPP": "snpp"}).select(keys, n_s=pl.col("n_det"))
            n = aggregate_common(n_days, {"NOAA-20": "noaa20"}).select(keys, n_n=pl.col("n_det"))
            b = s.join(n, on=keys, how="full", coalesce=True).with_columns(pl.col("n_s", "n_n").fill_null(0))
            b = b.join(m2_ratio_for_cells(b.select("cell_id").unique(), BRIDGE_MONTH[1]), on="cell_id", how="left")
            dim = calendar.monthrange(*BRIDGE_MONTH)[1]
            v = ((pl.col("n_s") + pl.col("n_n") * pl.col("m2")) * dim / n_ok).cast(pl.Float64)
            parts.append(b.select(keys, value=v, value_lo=v, value_hi=v))

    val = pl.concat(parts) if parts else pl.DataFrame(schema={"cell_id": pl.Int64, "year": pl.Int16, "month": pl.Int8,
                                                              "value": pl.Float64, "value_lo": pl.Float64,
                                                              "value_hi": pl.Float64})
    out = (
        val.join(raw_modis, on=keys, how="full", coalesce=True)
        .join(raw_snpp, on=keys, how="full", coalesce=True)
        .join(raw_n20, on=keys, how="full", coalesce=True)
        .join(yref.select("year", "month", "value_source", "data_source", "status", "regional_ea"),
              on=["year", "month"], how="inner")
    )
    in_snpp = pl.col("value_source") == "snpp_observed"
    in_bridge = pl.col("value_source") == "noaa20_bridged"
    out = out.with_columns(
        pl.col("value", "value_lo", "value_hi").fill_null(0.0),
        pl.col("raw_modis").fill_null(0).cast(pl.Int32),
        raw_viirs=pl.when(in_snpp).then(pl.col("raw_snpp").fill_null(0))
        .when(in_bridge).then(pl.col("raw_snpp").fill_null(0) + pl.col("raw_n20").fill_null(0))
        .otherwise(None).cast(pl.Int32),
        cell_lat=((pl.col("cell_id") // N_LON + 0.5) * 0.1 - 90).cast(pl.Float32),
        cell_lon=((pl.col("cell_id") % N_LON + 0.5) * 0.1 - 180).cast(pl.Float32),
        method_version=pl.lit(meta["method_version"]),
    )
    # 자료 범위 밖 칸 (2026-01~09 동아시아만) → missing
    x0, y0, x1, y1 = EAST_ASIA_BBOX
    outside = pl.col("regional_ea") & ~(pl.col("cell_lat").is_between(y0, y1) & pl.col("cell_lon").is_between(x0, x1))
    out = out.with_columns(status=pl.when(outside).then(pl.lit("missing")).otherwise(pl.col("status")))
    out = out.with_columns(*[pl.when(pl.col("status") == "missing").then(None).otherwise(pl.col(c)).alias(c)
                             for c in ("value", "value_lo", "value_hi")])
    out = out.filter((pl.col("value").fill_null(0) > 0) | (pl.col("raw_modis") > 0) | (pl.col("raw_viirs").fill_null(0) > 0))
    return out.select("cell_id", "cell_lat", "cell_lon", "year", "month", "value", "value_lo", "value_hi",
                      "raw_modis", "raw_viirs", "value_source", "data_source", "status", "method_version").sort(
        "cell_id", "year", "month")


def build_harmonized(ref: pl.DataFrame, meta: dict) -> int:
    tmp = HCM_PATH.with_suffix(".parts")
    tmp.mkdir(parents=True, exist_ok=True)
    years = sorted({y for y, _ in all_months()})
    n = 0
    for y in tqdm(years, desc="통합 칸×월", unit="year"):
        df = harmonized_year(y, ref, meta)
        df.write_parquet(tmp / f"{y}.parquet", compression="zstd")
        n += df.height
    pl.scan_parquet(sorted(tmp.glob("*.parquet"))).sink_parquet(HCM_PATH, compression="zstd")
    for f in tmp.glob("*.parquet"):
        f.unlink()
    tmp.rmdir()
    return n


# ── AOI ───────────────────────────────────────────────────────────────────

def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def aoi_membership() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(칸 → AOI 가중 소속표 [cell_id, aoi_id, weight], AOI 목록 [aoi_id, name, level, parent_aoi, n_cells])

    면적 가중 배분: 칸 하나의 값을 그 칸과 겹치는 행정구역들에 겹침 면적 비율(칸 안에서 합 1)로 나눈다.
      - 국가(ADM0)·1급(ADM1)·대륙: world_cell_adm1_weights (Natural Earth, 육지 비율 >= 0.1인 칸)
      - 한국(KR=1)·시·도·시군구: korea_cell_sgg_weights (geoBoundaries 한국 칸 + admdongkor 2025-12), 한국 칸은 NE 배분에서 제외
      - WORLD: 모든 칸 가중치 1. 육지 비율 < 0.1인 해안·바다 칸은 대륙(가장 많이 겹치는 국가의 대륙)·WORLD에만.
    n_cells = 가중치 > 0인 칸 수.
    """
    from planetarium.korea import KOREA_WEIGHTS_PATH
    from planetarium.world import WORLD_WEIGHTS_PATH
    wc = pl.read_parquet(WORLD_CELLS_PATH, columns=["cell_id", "continent", "is_land"])
    kw = pl.read_parquet(KOREA_WEIGHTS_PATH)
    ww = pl.read_parquet(WORLD_WEIGHTS_PATH)
    kor_ids = kw.select("cell_id").unique()
    grid = pl.read_parquet(PROCESSED_DATA_DIR / "grid_cells_0.1deg.parquet", columns=["cell_id"])
    hist = pl.scan_parquet(HCM_PATH).select("cell_id").unique().collect()
    cells = pl.concat([grid, hist, kor_ids]).unique()

    def cslug(c: str) -> pl.Expr:
        return pl.concat_str(pl.lit("CONT_"), pl.col(c).map_elements(slug, return_dtype=pl.String))

    one = pl.lit(1.0)
    ne = ww.join(kor_ids, on="cell_id", how="anti").filter(pl.col("adm0_a3") != "KOR")
    mem = [
        cells.select("cell_id", aoi_id=pl.lit("WORLD"), weight=one),
        ne.select("cell_id", aoi_id=pl.concat_str(pl.lit("ADM1_"), pl.col("adm1_code")), weight="weight"),
        ne.group_by("cell_id", "adm0_a3").agg(weight=pl.col("weight").sum()).select(
            "cell_id", aoi_id=pl.concat_str(pl.lit("ADM0_"), pl.col("adm0_a3")), weight="weight"),
        ne.filter(pl.col("continent").is_not_null()).group_by("cell_id", "continent").agg(weight=pl.col("weight").sum())
        .select("cell_id", aoi_id=cslug("continent"), weight="weight"),
        # 육지 비율 < 0.1인 칸(해안·바다): 대륙만
        cells.join(wc.filter(~pl.col("is_land"), pl.col("continent").is_not_null()), on="cell_id")
        .join(kor_ids, on="cell_id", how="anti").select("cell_id", aoi_id=cslug("continent"), weight=one),
        kor_ids.select("cell_id", aoi_id=pl.lit("KR"), weight=one),
        kor_ids.select("cell_id", aoi_id=pl.lit("CONT_Asia"), weight=one),
        kw.group_by("cell_id", "sido_cd").agg(weight=pl.col("weight").sum()).select(
            "cell_id", aoi_id=pl.concat_str(pl.lit("SIDO_"), pl.col("sido_cd")), weight="weight"),
        kw.select("cell_id", aoi_id=pl.concat_str(pl.lit("SGG_"), pl.col("sgg_cd")), weight="weight"),
    ]
    adm0 = load_ne_attrs()
    lst = [{"aoi_id": "WORLD", "name": "World", "level": "world", "parent_aoi": None}]
    conts = sorted(set(ne["continent"].drop_nulls().to_list()) | set(wc["continent"].drop_nulls().to_list()))
    lst += [{"aoi_id": f"CONT_{slug(c)}", "name": c, "level": "continent", "parent_aoi": "WORLD"} for c in conts]
    a0 = ne.select("adm0_a3").unique().join(adm0, on="adm0_a3", how="left")
    lst += [{"aoi_id": f"ADM0_{r['adm0_a3']}", "name": r["country"] or r["adm0_a3"], "level": "country",
             "parent_aoi": f"CONT_{slug(r['continent'])}" if r["continent"] else "WORLD"} for r in a0.to_dicts()]
    a1 = ne.select("adm1_code", "adm1_name", "adm0_a3").unique("adm1_code")
    lst += [{"aoi_id": f"ADM1_{r['adm1_code']}", "name": r["adm1_name"], "level": "adm1",
             "parent_aoi": f"ADM0_{r['adm0_a3']}"} for r in a1.to_dicts()]
    lst.append({"aoi_id": "KR", "name": "대한민국", "level": "kr", "parent_aoi": "CONT_Asia"})
    lst += [{"aoi_id": f"SIDO_{r['sido_cd']}", "name": r["sido_nm"], "level": "sido", "parent_aoi": "KR"}
            for r in kw.select("sido_cd", "sido_nm").unique("sido_cd").to_dicts()]
    lst += [{"aoi_id": f"SGG_{r['sgg_cd']}", "name": r["sgg_nm"], "level": "sgg", "parent_aoi": f"SIDO_{r['sido_cd']}"}
            for r in kw.select("sgg_cd", "sgg_nm", "sido_cd").unique("sgg_cd").to_dicts()]
    # 사용자 AOI (있으면): 폴리곤과 겹치는 면적 비율을 가중치로
    if USER_AOI_PATH.exists():
        feats = json.loads(USER_AOI_PATH.read_text(encoding="utf-8"))["features"]
        for i, f in enumerate(feats, 1):
            ids, fr = polygon_cells(shapely.make_valid(shape(f["geometry"])))
            aid = f"USER_{i}"
            mem.append(pl.DataFrame({"cell_id": ids, "weight": fr}).select("cell_id", aoi_id=pl.lit(aid), weight="weight"))
            lst.append({"aoi_id": aid, "name": (f.get("properties") or {}).get("name", aid), "level": "user", "parent_aoi": None})
    membership = (
        pl.concat([m.select(pl.col("cell_id").cast(pl.Int64), "aoi_id", pl.col("weight").cast(pl.Float64)) for m in mem])
        .group_by("cell_id", "aoi_id").agg(pl.col("weight").sum()).filter(pl.col("weight") > 0)
    )
    n = membership.group_by("aoi_id").agg(n_cells=pl.len())
    aoi_list = pl.DataFrame(lst).unique("aoi_id").join(n, on="aoi_id", how="left").with_columns(pl.col("n_cells").fill_null(0))
    return membership, aoi_list


def load_ne_attrs() -> pl.DataFrame:
    """Natural Earth admin0 속성 (adm0_a3, country, continent, subregion) — 도형은 읽지 않음."""
    a0 = json.loads((NE_DIR / "ne_10m_admin_0_countries.geojson").read_text(encoding="utf-8"))["features"]
    return pl.DataFrame([{"adm0_a3": f["properties"]["ADM0_A3"], "country": f["properties"]["NAME"],
                          "continent": f["properties"]["CONTINENT"], "subregion": f["properties"]["SUBREGION"]} for f in a0])


def weighting_compare(membership: pl.DataFrame) -> pl.DataFrame:
    """가중 배분 전후 비교: 이전 방식(칸 하나 -> 겹침이 가장 큰 한 곳) vs 면적 가중. 2000-11~ 전체 기간 합계."""
    kc = pl.read_parquet(KOREA_CELLS_PATH).filter("is_korea")
    wc = pl.read_parquet(WORLD_CELLS_PATH).filter("is_land", pl.col("adm0_a3") != "KOR", pl.col("kr_sgg_cd").is_null())
    old = pl.concat([
        kc.select("cell_id", aoi_id=pl.lit("KR")),
        kc.select("cell_id", aoi_id=pl.concat_str(pl.lit("SIDO_"), pl.col("sido_cd"))),
        kc.select("cell_id", aoi_id=pl.concat_str(pl.lit("SGG_"), pl.col("sgg_cd"))),
        wc.select("cell_id", aoi_id=pl.concat_str(pl.lit("ADM0_"), pl.col("adm0_a3"))),
    ]).with_columns(weight=pl.lit(1.0))
    tot = pl.scan_parquet(HCM_PATH).group_by("cell_id").agg(v=pl.col("value").sum(), r=pl.col("raw_modis").sum()).collect()

    def s(m: pl.DataFrame, tag: str) -> pl.DataFrame:
        return m.join(tot, on="cell_id").group_by("aoi_id").agg(
            (pl.col("v") * pl.col("weight")).sum().alias(f"value_{tag}"),
            (pl.col("r") * pl.col("weight")).sum().alias(f"raw_modis_{tag}"),
            pl.len().alias(f"n_cells_{tag}"))

    keep = pl.col("aoi_id").str.contains(r"^(KR|SIDO_|SGG_|ADM0_)")
    cmp = s(old, "top").join(s(membership.filter(keep), "weighted"), on="aoi_id", how="full", coalesce=True).with_columns(
        diff_pct=(pl.col("value_weighted") / pl.col("value_top") - 1) * 100).sort("aoi_id")
    cmp.write_csv(REPORTS_DIR / "aoi_weighting_compare.csv")
    return cmp


def aoi_coverage_ok(membership: pl.DataFrame) -> pl.DataFrame:
    """AOI가 동아시아 상자 안에 완전히 들어가는지 (2026-01~09 자료 범위 판정용)."""
    x0, y0, x1, y1 = EAST_ASIA_BBOX
    lat = (pl.col("cell_id") // N_LON + 0.5) * 0.1 - 90
    lon = (pl.col("cell_id") % N_LON + 0.5) * 0.1 - 180
    return membership.group_by("aoi_id").agg(inside_ea=(lat.is_between(y0, y1) & lon.is_between(x0, x1)).all())


def build_aoi_month(ref: pl.DataFrame, membership: pl.DataFrame, aoi_list: pl.DataFrame) -> pl.DataFrame:
    hcm = pl.scan_parquet(HCM_PATH)
    parts = []
    for y in tqdm(sorted(ref["year"].unique().to_list()), desc="AOI 합계", unit="year"):  # 연도별로 나눠 메모리 절약
        parts.append(
            hcm.filter(pl.col("year") == y).join(membership.lazy(), on="cell_id")
            .group_by("aoi_id", "year", "month")
            .agg(*[(pl.col(c) * pl.col("weight")).sum().alias(c) for c in ("value", "value_lo", "value_hi")],
                 raw_modis=(pl.col("raw_modis") * pl.col("weight")).sum(),
                 raw_viirs=(pl.col("raw_viirs") * pl.col("weight")).sum(),
                 any_nrt=(pl.col("data_source") == "NRT").any())
            .collect()
        )
    sums = pl.concat(parts)
    base = aoi_list.select("aoi_id", "n_cells").join(ref, how="cross").join(aoi_coverage_ok(membership), on="aoi_id", how="left")
    df = base.join(sums, on=["aoi_id", "year", "month"], how="left")
    viirs_era = pl.col("value_source").is_in(["snpp_observed", "noaa20_bridged"])
    out_of_range = pl.col("regional_ea") & ~pl.col("inside_ea").fill_null(False)
    df = df.with_columns(
        pl.col("value", "value_lo", "value_hi").fill_null(0.0),
        # 원 감지 수도 면적 가중 배분 -> 반올림해 정수로 (스키마 유지)
        raw_modis=pl.col("raw_modis").fill_null(0).round(0).cast(pl.Int64),
        raw_viirs=pl.when(viirs_era).then(pl.col("raw_viirs").fill_null(0)).otherwise(None).round(0).cast(pl.Int64),
        data_source=pl.when(pl.col("any_nrt").fill_null(False)).then(pl.lit("NRT")).otherwise(pl.col("data_source")),
        status=pl.when(pl.col("status") == "partial").then(pl.lit("partial"))
        .when(out_of_range).then(pl.lit("missing")).otherwise(pl.col("status")),
    ).with_columns(
        status=pl.when((pl.col("status") == "ok") & (pl.col("data_source") == "NRT")).then(pl.lit("nrt")).otherwise(pl.col("status")),
    ).with_columns(*[pl.when(pl.col("status") == "missing").then(None).otherwise(pl.col(c)).alias(c)
                     for c in ("value", "value_lo", "value_hi")])
    df = df.select("aoi_id", "year", "month", "value", "value_lo", "value_hi", "raw_modis", "raw_viirs",
                   "value_source", "data_source", "status", "n_cells").sort("aoi_id", "year", "month")
    df.write_parquet(AOI_MONTH_PATH, compression="zstd")
    return df


def build_aoi_boundaries(aoi_list: pl.DataFrame) -> int:
    """앱 지도용 경계 (WGS84, 단순화). 국가·1급(NE), 한국 시·도·시군구(admdongkor 2025-12 → 4326)."""
    feats = []
    keep = set(aoi_list["aoi_id"].to_list())
    names = dict(zip(aoi_list["aoi_id"], aoi_list["name"]))
    parents = dict(zip(aoi_list["aoi_id"], aoi_list["parent_aoi"]))

    def add(aid: str, geom, level: str) -> None:
        if aid in keep and geom is not None and not geom.is_empty:
            g = shapely.simplify(geom, SIMPLIFY_DEG, preserve_topology=True)
            feats.append({"type": "Feature", "properties": {"aoi_id": aid, "name": names.get(aid), "level": level,
                                                            "parent_aoi": parents.get(aid)},
                          "geometry": mapping(shapely.set_precision(g, 1e-4))})

    a0 = json.loads((NE_DIR / "ne_10m_admin_0_countries.geojson").read_text(encoding="utf-8"))["features"]
    for f in a0:
        add(f"ADM0_{f['properties']['ADM0_A3']}", shapely.make_valid(shape(f["geometry"])), "country")
    a1 = json.loads((NE_DIR / "ne_10m_admin_1_states_provinces.geojson").read_text(encoding="utf-8"))["features"]
    for f in a1:
        add(f"ADM1_{f['properties']['adm1_code']}", shapely.make_valid(shape(f["geometry"])), "adm1")
    import pyarrow.parquet as pq
    for name, col, prefix, level in (("sido_20251231", "sidocd", "SIDO_", "sido"), ("sgg_20251231", "sggcd", "SGG_", "sgg")):
        t = pq.read_table(ADM_DIR / f"{name}.parquet")
        geoms = shapely.from_wkb(t.column("geom").to_pylist())
        for code, g in zip(t.column(col).to_pylist(), geoms):
            g4326 = shapely.transform(g, lambda xy: np.column_stack(TO_4326.transform(xy[:, 0], xy[:, 1])))
            add(f"{prefix}{code}", shapely.make_valid(g4326), level)
    kr = json.loads((EXTERNAL_DATA_DIR / "boundaries" / "geoboundaries" / "geoBoundaries-KOR-ADM0_simplified.geojson")
                    .read_text(encoding="utf-8"))["features"][0]["geometry"]
    add("KR", shapely.make_valid(shape(kr)), "kr")
    if USER_AOI_PATH.exists():
        for i, f in enumerate(json.loads(USER_AOI_PATH.read_text(encoding="utf-8"))["features"], 1):
            add(f"USER_{i}", shapely.make_valid(shape(f["geometry"])), "user")
    AOI_GEOJSON_PATH.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
                                encoding="utf-8")
    return len(feats)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="최종 통합 시계열과 AOI 월 합계")
    ap.add_argument("--skip-cells", action="store_true", help="harmonized_cell_month는 그대로 두고 AOI만")
    ap.add_argument("--skip-boundaries", action="store_true")
    args = ap.parse_args(argv)
    start_run_log("harmonize")
    t0 = time.perf_counter()
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    ref = month_reference()
    if not args.skip_cells:
        n = build_harmonized(ref, meta)
        print(f"[완료] {HCM_PATH.name}: {n:,}행 ({time.perf_counter() - t0:,.0f}초)")
    membership, aoi_list = aoi_membership()
    aoi_list.write_parquet(AOI_LIST_PATH)
    weighting_compare(membership)
    am = build_aoi_month(ref, membership, aoi_list)
    print(f"[완료] {AOI_LIST_PATH.name}: AOI {aoi_list.height:,}개 / {AOI_MONTH_PATH.name}: {am.height:,}행")
    if not args.skip_boundaries:
        nf = build_aoi_boundaries(aoi_list)
        print(f"[완료] {AOI_GEOJSON_PATH.name}: {nf:,}개, {AOI_GEOJSON_PATH.stat().st_size / 1e6:,.1f} MB")
    # 결정 기록
    meta.setdefault("harmonization", {}).update({
        "2012-01": "S-NPP가 2012-01-20에 시작한 부분 월이라 MODIS 보정 값(modis_harmonized)을 씀",
        "snpp_observed": "S-NPP status=ok인 날의 감지 수 × days_in_month / ok일수 (raw_viirs는 전체 일 원값)",
        "noaa20_bridged": f"{NOAA20_BRIDGE_START} 부터 NOAA-20 × M2 비율 (한국·아르헨티나·대륙·world 순, 대체 규칙 적용)",
        "regional_only": "2026-01~09는 동아시아 자료만 있어 그 밖의 칸·AOI는 missing",
        "sparse_cell_table": "harmonized_cell_month는 value·raw가 모두 0인 칸·월을 저장하지 않음",
    })
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=1, thousands_separator=","):
        print(ref.group_by("value_source", "status").agg(months=pl.len(), first=pl.min_horizontal(pl.col("year").min()),
                                                         last=pl.col("year").max()).sort("first"))
        kr = am.filter(pl.col("aoi_id") == "KR")
        print(f"KR: {kr.height}개월, 값 없는 달 {kr['value'].is_null().sum()}, status 분포 {dict(kr.group_by('status').len().iter_rows())}")


if __name__ == "__main__":
    main()
