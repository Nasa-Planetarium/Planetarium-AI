"""고정 열원 점 제외 (v1.2, 2026-11-15) — 점 파일은 다시 쓰지 않고 집계에서 뺀다.

규칙  0.01° 고정 열원 칸(static_source_cells_0.01deg: SP type 1·2가 2개 연도 이상 나온 칸) 안의 점 = is_static_source.
      SP·NRT 같은 규칙. (NRT는 입력 단계에서 이미 빠져 있어 실제로는 SP type 0 점만 해당.)
처리 (달마다 기준 점 파티션을 읽어)
  harmonized_cell_month   value      -= Σ 고정 열원 기준 점 harmonized_weight (칸·월)
                          value_lo/hi -= Σ weight_lo/hi (MODIS 기간은 칸 안 기준 점 가중치가 value에 비례하므로 value와 같은 비율)
                          value_projected: 완결된 달은 = value, 진행 중인 달은 value 변화 비율만큼
                          residual·raw_modis·raw_viirs는 그대로 (raw는 원 감지 수)
  harmonized_cell_day     value·value_lo/hi·n_points에서 그 칸·날의 고정 열원 기준 점 몫을 뺀다
  뺀 양은 열(static_removed, static_removed_lo, static_removed_hi[, static_n_points])로 남긴다 →
  다시 실행해도 원래 값(= 현재 + 이전에 뺀 양)에서 새로 계산하므로 두 번 빠지지 않는다 (fetch_nrt --append 뒤에도 실행).
  점 플래그 표: data/processed/harmonized_hotspots_static_flags/year=YYYY.parquet
                (점 키 satellite·latitude·longitude·acq_datetime_utc + layer, harmonized_weight, is_static_source=true)

실행  uv run python -m planetarium.static_exclusion [--years 2000 ... 2026]   (기본: 통합표의 모든 연도)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl
from tqdm import tqdm

from planetarium.config import PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.features import static_mask_path
from planetarium.harmonize import HCM_PATH
from planetarium.hotspots import CELL_DAY_PATH, HOTSPOTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

FLAGS_DIR = PROCESSED_DATA_DIR / "harmonized_hotspots_static_flags"
WORK = PROCESSED_DATA_DIR.parent / "interim" / "static_exclusion"
KEYS = ["cell_id", "year", "month"]
PT_KEYS = ["satellite", "latitude", "longitude", "acq_datetime_utc"]


def c001() -> pl.Expr:
    return (((pl.col("latitude") + 90) / 0.01).floor().cast(pl.Int64) * 36000
            + ((pl.col("longitude") + 180) / 0.01).floor().cast(pl.Int64) % 36000)


def mask_ids() -> pl.Series:
    return pl.read_parquet(static_mask_path())["cell_id"]


def is_static_expr(mask: pl.Series | None = None) -> pl.Expr:
    """점이 고정 열원 0.01° 칸 안에 있는가 (다른 모듈이 점을 읽을 때 쓰는 필터)."""
    m = mask if mask is not None else mask_ids()
    return c001().is_in(m.implode())


def month_static(year: int, month: int, mask: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(칸·월 제외량, 칸·일 제외량, 플래그 점) — 기준 점은 제외량, 기준·기타 점 모두 플래그."""
    parts = []
    for layer in ("reference", "other"):
        d = HOTSPOTS_DIR / layer / f"year={year}" / f"month={month:02d}"
        if not d.exists():
            continue
        parts.append(pl.scan_parquet(d / "*.parquet").filter(is_static_expr(mask)).select(
            *PT_KEYS, "date_local", "cell_id", "source", "harmonized_weight", "weight_lo", "weight_hi", "is_reference",
            layer=pl.lit(layer)).collect())
    if not parts:
        empty = pl.DataFrame()
        return empty, empty, empty
    pts = pl.concat(parts, how="diagonal_relaxed")
    ref = pts.filter("is_reference")
    w = pl.col("harmonized_weight").cast(pl.Float64)
    cm = ref.group_by("cell_id").agg(static_removed=w.sum(), static_removed_lo=pl.col("weight_lo").cast(pl.Float64).sum(),
                                     static_removed_hi=pl.col("weight_hi").cast(pl.Float64).sum(),
                                     lo_null=pl.col("weight_lo").is_null().any()).with_columns(
        year=pl.lit(year, dtype=pl.Int16), month=pl.lit(month, dtype=pl.Int8))
    cm = cm.with_columns(static_removed_lo=pl.when("lo_null").then(None).otherwise("static_removed_lo"),
                         static_removed_hi=pl.when("lo_null").then(None).otherwise("static_removed_hi")).drop("lo_null")
    cd = ref.group_by("cell_id", "date_local").agg(
        static_removed=w.sum(), static_removed_lo=pl.col("weight_lo").cast(pl.Float64).sum(),
        static_removed_hi=pl.col("weight_hi").cast(pl.Float64).sum(), static_n_points=pl.len().cast(pl.Int32))
    flags = pts.select(*PT_KEYS, "date_local", "cell_id", "layer", "source", "is_reference", "harmonized_weight",
                       is_static_source=pl.lit(True))
    return cm, cd, flags


def collect(years: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    mask = mask_ids()
    WORK.mkdir(parents=True, exist_ok=True)
    FLAGS_DIR.mkdir(parents=True, exist_ok=True)
    cms, cds, log = [], [], []
    months = (pl.scan_parquet(HCM_PATH).filter(pl.col("year").is_in(years)).select("year", "month").unique().sort("year", "month")
              .collect().rows())
    by_year: dict[int, list] = {}
    for y, m in tqdm(months, desc="고정 열원 점 (달별)", unit="month", dynamic_ncols=True):
        cm, cd, fl = month_static(y, m, mask)
        if cm.height:
            cms.append(cm)
            cds.append(cd)
        if fl.height:
            by_year.setdefault(y, []).append(fl)
        log.append({"year": y, "month": m, "ref_points": int(cd["static_n_points"].sum()) if cd.height else 0,
                    "ref_weight": float(cm["static_removed"].sum()) if cm.height else 0.0,
                    "flag_points": fl.height, "peak_mem_gb": round(peak_memory_gb() or 0, 2)})
    for y in years:
        f = FLAGS_DIR / f"year={y}.parquet"
        if y in by_year:
            pl.concat(by_year[y], how="diagonal_relaxed").write_parquet(f, compression="zstd")
        elif f.exists():
            f.unlink()
    pl.DataFrame(log).write_csv(REPORTS_DIR / "static_exclusion_months.csv")
    sch_cm = {"cell_id": pl.Int64, "static_removed": pl.Float64, "static_removed_lo": pl.Float64,
              "static_removed_hi": pl.Float64, "year": pl.Int16, "month": pl.Int8}
    cm = pl.concat(cms, how="vertical_relaxed") if cms else pl.DataFrame(schema=sch_cm)
    cd = pl.concat(cds, how="vertical_relaxed") if cds else pl.DataFrame()
    return cm, cd


def _restore(lf: pl.LazyFrame, cols: list[str], extra: list[str]) -> pl.LazyFrame:
    """이전에 뺀 양을 되돌린 원래 값 (열이 없으면 그대로)."""
    names = lf.collect_schema().names()
    if "static_removed" not in names:
        return lf
    lf = lf.with_columns(
        value=pl.col("value") + pl.col("static_removed").fill_null(0.0),
        value_lo=pl.col("value_lo") + pl.col("static_removed_lo").fill_null(0.0),
        value_hi=pl.col("value_hi") + pl.col("static_removed_hi").fill_null(0.0),
        **({"n_points": pl.col("n_points") + pl.col("static_n_points").fill_null(0)} if "static_n_points" in names else {}),
    )
    return lf.drop([c for c in ["static_removed", "static_removed_lo", "static_removed_hi", *extra] if c in names])


def apply_cell_month(cm: pl.DataFrame, years: list[int]) -> None:
    lf = pl.scan_parquet(HCM_PATH)
    cols = lf.collect_schema().names()
    base_cols = [c for c in cols if not c.startswith("static_")]
    target = lf.filter(pl.col("year").is_in(years))
    kept = lf.filter(~pl.col("year").is_in(years))
    old_value = pl.col("value")
    t = _restore(target, base_cols, []).join(cm.lazy(), on=KEYS, how="left")
    has = pl.col("static_removed").is_not_null()
    t = t.with_columns(_v0=old_value).with_columns(
        value=pl.when(has).then(pl.col("value") - pl.col("static_removed")).otherwise("value"),
        value_lo=pl.when(has).then(pl.col("value_lo") - pl.col("static_removed_lo").fill_null(0.0)).otherwise("value_lo"),
        value_hi=pl.when(has).then(pl.col("value_hi") - pl.col("static_removed_hi").fill_null(0.0)).otherwise("value_hi"),
    )
    if "value_projected" in cols:
        t = t.with_columns(value_projected=pl.when(pl.col("status") != "partial").then(pl.col("value"))
                           .when(has & (pl.col("_v0") > 0)).then(pl.col("value_projected") * pl.col("value") / pl.col("_v0"))
                           .otherwise("value_projected"))
    # 0으로 자르지 않는다: 관측이 빠진 날의 점도 가중치를 받으므로 value < Σ(고정 열원 점)인 칸이 있고,
    # 그 차이는 residual(missing_day_fill, 음수)이 이미 메우고 있다 → 자르면 점+residual = 칸 등식이 깨진다.
    t = t.drop("_v0")
    out_cols = base_cols + ["static_removed", "static_removed_lo", "static_removed_hi"]
    kept = kept.with_columns(*[pl.lit(None, dtype=pl.Float64).alias(c) for c in out_cols if c not in cols]) if \
        "static_removed" not in cols else kept
    tmp = HCM_PATH.with_suffix(".new.parquet")
    pl.concat([kept.select(out_cols), t.select(out_cols)], how="vertical_relaxed").sink_parquet(tmp, compression="zstd")
    tmp.replace(HCM_PATH)


def apply_cell_day(cd: pl.DataFrame, years: list[int]) -> None:
    lf = pl.scan_parquet(CELL_DAY_PATH)
    cols = lf.collect_schema().names()
    base_cols = [c for c in cols if not c.startswith("static_")]
    target = lf.filter(pl.col("year").is_in(years))
    kept = lf.filter(~pl.col("year").is_in(years))
    t = _restore(target, base_cols, ["static_n_points"])
    if cd.height:
        # 1일에는 점 행과 residual 행(n_points = 0)이 따로 있을 수 있다 → 점 행에만 붙인다
        t = t.join(cd.lazy().with_columns(_pt=pl.lit(True)), on=["cell_id", "date_local"], how="left").with_columns(
            *[pl.when(pl.col("n_points") > 0).then(pl.col(c)).alias(c)
              for c in ("static_removed", "static_removed_lo", "static_removed_hi", "static_n_points")]).drop("_pt")
    else:
        t = t.with_columns(static_removed=pl.lit(None, dtype=pl.Float64), static_removed_lo=pl.lit(None, dtype=pl.Float64),
                           static_removed_hi=pl.lit(None, dtype=pl.Float64), static_n_points=pl.lit(None, dtype=pl.Int32))
    has = pl.col("static_removed").is_not_null()
    t = t.with_columns(
        value=pl.when(has).then(pl.col("value") - pl.col("static_removed")).otherwise("value"),
        value_lo=pl.when(has).then(pl.col("value_lo") - pl.col("static_removed_lo").fill_null(0.0)).otherwise("value_lo"),
        value_hi=pl.when(has).then(pl.col("value_hi") - pl.col("static_removed_hi").fill_null(0.0)).otherwise("value_hi"),
        n_points=pl.when(has).then(pl.col("n_points") - pl.col("static_n_points")).otherwise("n_points"),
    )
    out_cols = base_cols + ["static_removed", "static_removed_lo", "static_removed_hi", "static_n_points"]
    if "static_removed" not in cols:
        kept = kept.with_columns(static_removed=pl.lit(None, dtype=pl.Float64), static_removed_lo=pl.lit(None, dtype=pl.Float64),
                                 static_removed_hi=pl.lit(None, dtype=pl.Float64), static_n_points=pl.lit(None, dtype=pl.Int32))
    tmp = CELL_DAY_PATH.with_suffix(".new.parquet")
    pl.concat([kept.select(out_cols), t.select(out_cols)], how="vertical_relaxed").sink_parquet(tmp, compression="zstd")
    tmp.replace(CELL_DAY_PATH)


def run(years: list[int] | None = None) -> dict:
    t0 = time.perf_counter()
    years = years or sorted(pl.scan_parquet(HCM_PATH).select("year").unique().collect()["year"].to_list())
    cm, cd = collect(years)
    print(f"[제외량] 칸·월 {cm.height:,}행, 가중치 합 {float(cm['static_removed'].sum()) if cm.height else 0:,.1f}; "
          f"칸·일 {cd.height:,}행, 최대 메모리 {peak_memory_gb() or 0:.2f} GB", flush=True)
    apply_cell_month(cm, years)
    print(f"[완료] {HCM_PATH.name} 갱신, 최대 메모리 {peak_memory_gb() or 0:.2f} GB", flush=True)
    apply_cell_day(cd, years)
    print(f"[완료] {CELL_DAY_PATH.name} 갱신, 최대 메모리 {peak_memory_gb() or 0:.2f} GB", flush=True)
    info = {"years": [min(years), max(years)], "cell_months": cm.height, "removed_weight": float(cm["static_removed"].sum()) if cm.height else 0.0,
            "seconds": round(time.perf_counter() - t0), "peak_memory_gb": round(peak_memory_gb() or 0, 2)}
    print(f"[완료] static_exclusion {json.dumps(info, ensure_ascii=False)}")
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="고정 열원 점 제외 (집계 다시 계산)")
    ap.add_argument("--years", nargs="*", type=int)
    args = ap.parse_args(argv)
    start_run_log("static_exclusion")
    run(args.years)


if __name__ == "__main__":
    main()
