"""감지 단위 통합 (harmonized hotspots): 어떤 위성의 감지든 같은 변환을 거쳐 공통 형식 + VIIRS(S-NPP) 기준 가중치로.

harmonize_detections(points) — 입력: 통합 Parquet 스키마(VIIRS 기준) 행. 과거 파일이든 실시간 API든 같은 함수.
  공통 처리
    - 고정 열원 제거: type == 0, type이 없는 NRT는 고정 열원 마스크(0.01°) 밖만 (cell_day와 같은 규칙)
    - date_local(현지 태양시 날짜), cell_id(0.1°), confidence(l/n/h), 국가·1급 행정구역(칸 겹침 최대)
  기준 위성 (합계에 쓰는 점, is_reference=true) — date_local 기준
    - 2000-11 ~ 2012-01: MODIS (Terra·Aqua)   ※ v1에서 2012-01은 m2v 값을 쓰므로 MODIS가 기준
    - 2012-02 ~ 2026-11-02: S-NPP
    - 2026-11-03 ~: NOAA-20. NOAA-20 status ≠ ok인 날은 NOAA-21 → MODIS 순으로 대체 (weight_source에 표시)
    - 나머지 위성 점은 is_reference=false (지도 '모든 감지' 레이어용)
  가중치 (planetarium.modeling.names.WEIGHT_RULES)
    - 기준 점: 칸·월 value를 그 칸·월 기준 점에 나눠 준다.
        harmonized_weight_i = value × b_i / Σ b   (b: S-NPP 1, NOAA-20 n20_bridge, NOAA-21 n21_bridge, MODIS 1)
      → MODIS 기간에는 value / raw_modis (m2v 배율), S-NPP 기간에는 value / raw_viirs (자료가 빠진 날이 없으면 1.0)
      → 어느 칸·월이든 기준 점 가중치 합 = harmonized_cell_month.value (residual 제외)
      weight_lo/hi = value_lo/hi × b_i / Σ b. Terra 단독 기간(2000-11 ~ 2002-06)은 칸 단위 구간이 없어 null
      (이 기간 구간은 AOI 단위로만 제공).
    - 기준이 아닌 점: S-NPP 1.0, NOAA-20 n20_bridge, NOAA-21 n21_bridge, MODIS(2012-02 이후)는 m2v 칸 값이 없어 null.
  residual (점으로 만들 수 없는 값)
    - MODIS 기간에 그 칸·월 MODIS 감지가 없는데 value > 0인 칸·월 (harmonized_cell_month의 감지 없는 칸 예측 ≥ 0.5)
      + 배경 표(interim/harmonized_background, 예측 < 0.5) → harmonized_residual_cell_month
    → 점 가중치 합 + residual = 칸 value, AOI 면적 가중 합 = aoi_month.value

출력 컬럼
  latitude, longitude, acq_datetime_utc, date_local, satellite, instrument, frp, confidence, daynight, cell_id,
  adm0_a3, adm1_code, harmonized_weight, weight_lo, weight_hi, weight_source, is_reference, source, method_version
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

from pathlib import Path

import polars as pl

from planetarium.config import NOAA20_BRIDGE_START, PROCESSED_DATA_DIR
from planetarium.dataset import load_hotspots
from planetarium.features import STATIC_RES, cell_expr, static_mask_path
from planetarium.harmonize import HCM_BACKGROUND_DIR, HCM_PATH
from planetarium.modeling.names import METHOD_VERSION, N20_BRIDGE_PATH, WEIGHT_RULES
from planetarium.monthly import STATUS_PATH
from planetarium.world import WORLD_CELLS_PATH

HOTSPOTS_DIR = PROCESSED_DATA_DIR / "harmonized_hotspots"
RESIDUAL_PATH = PROCESSED_DATA_DIR / "harmonized_residual_cell_month.parquet"
MODIS_REF_END = date(2012, 1, 31)        # v1: 2012-01은 m2v 값
SNPP_REF_END = NOAA20_BRIDGE_START - timedelta(days=1)  # 2026-11-02
SAT_CODE = {"Terra": "Terra", "Aqua": "Aqua", "N": "S-NPP", "N20": "NOAA-20", "N21": "NOAA-21"}
OUT_COLS = ["latitude", "longitude", "acq_datetime_utc", "date_local", "satellite", "instrument", "frp", "confidence",
            "daynight", "cell_id", "adm0_a3", "adm1_code", "harmonized_weight", "weight_lo", "weight_hi",
            "weight_source", "is_reference", "source", "method_version"]


# ── 보조 표 (한 번 읽어 재사용) ───────────────────────────────────────────

class Context:
    def __init__(self) -> None:
        self.static = pl.read_parquet(static_mask_path())["cell_id"]
        self.cells = pl.read_parquet(WORLD_CELLS_PATH, columns=["cell_id", "adm0_a3", "adm1_code", "continent", "kr_sgg_cd"])
        st = pl.read_parquet(STATUS_PATH, columns=["satellite", "date_utc", "status"])
        self.status = {(s, d): v for s, d, v in st.iter_rows()}
        r = pl.read_parquet(N20_BRIDGE_PATH)
        self.bridge = {(p, reg, m): u for p, reg, m, u in r.select("pair", "region", "month", "used_ratio").iter_rows()}

    def ok(self, sat: str, d: date) -> bool:
        return self.status.get((sat, d)) == "ok"

    def usable(self, sat: str, d: date) -> bool:
        """기준 점으로 쓸 수 있는 날: ok 또는 받는 중인 날(partial_current_day)."""
        return self.status.get((sat, d)) in ("ok", "partial")

    _m2v = None

    def m2v(self):
        """새 달 MODIS용 m2v 예측기 (처음 쓸 때 모델을 읽는다)."""
        if self._m2v is None:
            from planetarium.modeling.m2v_predict import M2VPredictor
            self._m2v = M2VPredictor()
        return self._m2v


def bridge_factor(ctx: Context, pair: str) -> pl.Expr:
    """칸 지역(한국·아르헨티나·대륙·world)과 달력 월별 bridge 계수 (n20_bridge / n21_bridge, 대체 규칙 적용된 used_ratio)."""
    keys = {(reg, m): u for (p, reg, m), u in ctx.bridge.items() if p == pair}
    df = pl.DataFrame([{"_reg": reg, "_m": m, "_f": u} for (reg, m), u in keys.items()],
                      schema={"_reg": pl.String, "_m": pl.Int8, "_f": pl.Float64})
    return df  # join용 표 (호출하는 쪽에서 join)


# ── 공통 처리 ─────────────────────────────────────────────────────────────

def prepare(points: pl.LazyFrame | pl.DataFrame, ctx: Context) -> pl.LazyFrame:
    """고정 열원 제거, date_local, cell_id, 국가·1급, 연·월(현지)."""
    lf = points.lazy()
    static_cell = cell_expr(STATIC_RES)["cell_id"]
    return (
        lf.filter((pl.col("type") == 0) | (pl.col("type").is_null() & ~static_cell.is_in(ctx.static.implode())))
        .with_columns(
            cell_id=cell_expr(0.1)["cell_id"],
            date_local=(pl.col("acq_datetime_utc").dt.replace_time_zone(None)
                        + pl.duration(minutes=(pl.col("longitude") * 4).round().cast(pl.Int64))).dt.date(),
        )
        .with_columns(year=pl.col("date_local").dt.year().cast(pl.Int16), month=pl.col("date_local").dt.month().cast(pl.Int8))
        .join(ctx.cells.lazy(), on="cell_id", how="left")
        .with_columns(adm0_a3=pl.when(pl.col("kr_sgg_cd").is_not_null()).then(pl.lit("KOR")).otherwise(pl.col("adm0_a3")))
    )


def reference_flags(lf: pl.LazyFrame, ctx: Context, days: list[date]) -> pl.LazyFrame:
    """is_reference, ref_rule. NOAA-20 기간에 NOAA-20이 ok가 아닌 날은 NOAA-21 → MODIS 순으로 대체."""
    fallback = {}
    for d in days:
        if d >= NOAA20_BRIDGE_START and not ctx.usable("NOAA-20", d):
            fallback[d] = "N21" if ctx.usable("NOAA-21", d) else "MODIS"
    fb = pl.DataFrame({"date_local": list(fallback), "_fb": list(fallback.values())},
                      schema={"date_local": pl.Date, "_fb": pl.String})
    modis = pl.col("instrument") == "MODIS"
    d = pl.col("date_local")
    return lf.join(fb.lazy(), on="date_local", how="left").with_columns(
        ref_rule=pl.when((d <= MODIS_REF_END) & modis).then(pl.lit("modis"))
        .when((d > MODIS_REF_END) & (d <= SNPP_REF_END) & (pl.col("satellite") == "N")).then(pl.lit("snpp"))
        .when((d >= NOAA20_BRIDGE_START) & pl.col("_fb").is_null() & (pl.col("satellite") == "N20")).then(pl.lit("n20"))
        .when((pl.col("_fb") == "N21") & (pl.col("satellite") == "N21")).then(pl.lit("n21_fallback"))
        .when((pl.col("_fb") == "MODIS") & modis).then(pl.lit("modis_fallback"))
        .otherwise(None),
    ).with_columns(is_reference=pl.col("ref_rule").is_not_null()).drop("_fb")


def base_weight(lf: pl.LazyFrame, ctx: Context) -> pl.LazyFrame:
    """b: 위성별 기본 가중치 (S-NPP 1, NOAA-20 n20_bridge, NOAA-21 n21_bridge, MODIS 1)."""
    region = (pl.when(pl.col("kr_sgg_cd").is_not_null()).then(pl.lit("korea"))
              .when(pl.col("adm0_a3") == "ARG").then(pl.lit("argentina"))
              .when(pl.col("continent").is_not_null()).then(pl.concat_str(pl.lit("continent:"), pl.col("continent")))
              .otherwise(pl.lit("world")))
    lf = lf.with_columns(_reg=region, _m=pl.col("month"))
    for sat, pair, col in (("N20", "S-NPP/NOAA-20", "_f20"), ("N21", "S-NPP/NOAA-21", "_f21")):
        t = bridge_factor(ctx, pair)
        world = t.filter(pl.col("_reg") == "world").select("_m", pl.col("_f").alias(f"{col}_w"))
        lf = lf.join(t.lazy().rename({"_f": col}), on=["_reg", "_m"], how="left").join(world.lazy(), on="_m", how="left") \
            .with_columns(pl.coalesce(col, f"{col}_w").alias(col)).drop(f"{col}_w")
    return lf.with_columns(
        b=pl.when(pl.col("satellite") == "N20").then("_f20")
        .when(pl.col("satellite") == "N21").then("_f21")
        .otherwise(pl.lit(1.0)),
    ).drop("_reg", "_m")


MODIS_M2V_SOURCES = ("modis_m2v", "modis_m2v_terra")


def harmonize_detections(points: pl.LazyFrame | pl.DataFrame, cell_values: pl.DataFrame,
                         ctx: Context | None = None, unit_weights: pl.DataFrame | None = None,
                         as_of: date | None = None) -> pl.DataFrame:
    """감지 점 → 통합 핫스팟 (공통 형식 + 가중치). cell_values: 그 기간 harmonized_cell_month 행.

    점 가중치 = 센서 변환만 (관측이 빠진 날을 달 전체로 늘리는 환산은 넣지 않음 — 그 몫은 칸·월 residual
    'missing_day_fill'로 따로 둔다):
      S-NPP 1.0 · NOAA-20 n20_bridge · NOAA-21 n21_bridge (기준·비기준 같음)
      MODIS, m2v 기간(통합 표 value_source가 modis_m2v*): 칸·월 m2v 배율 = value × b / Σb (= value / raw_modis)
        unit_weights(저장된 통합 핫스팟의 칸·월·위성별 점 가중치)가 있으면 그것을 그대로 (입력이 그 달의 일부일 때)
      MODIS, 그 밖의 달(2012-02 이후·새 달): 입력 MODIS 점으로 특성을 만들어 m2v_xgb로 예측한 배율
        (weight_source='m2v_predicted', 공통 정상일에 감지가 없는 칸은 null)
    as_of: 진행 중인 달의 마지막 관측일 (새 달 MODIS 예측의 경과 일수 기준)."""
    ctx = ctx or Context()
    lf = prepare(points, ctx)
    days = lf.select(pl.col("date_local").unique()).collect()["date_local"].to_list()
    lf = base_weight(reference_flags(lf, ctx, days), ctx)
    keys = ["cell_id", "year", "month"]
    df = lf.collect()
    cv = cell_values.select(*keys, "value", "value_lo", "value_hi", "value_source")
    df = df.join(cv, on=keys, how="left")
    if unit_weights is not None and unit_weights.height:
        df = df.join(unit_weights.select(*keys, "satellite", "unit_w", "unit_lo", "unit_hi"), on=[*keys, "satellite"], how="left")
    else:
        df = df.with_columns(unit_w=pl.lit(None, dtype=pl.Float64), unit_lo=pl.lit(None, dtype=pl.Float64),
                             unit_hi=pl.lit(None, dtype=pl.Float64))
    modis = pl.col("instrument") == "MODIS"
    m2v_month = pl.col("value_source").is_in(MODIS_M2V_SOURCES)
    # (1) MODIS, m2v 기간 기준 점: 칸·월 m2v 배율
    sb = (pl.col("b") * (modis & pl.col("is_reference")).cast(pl.Float64)).sum().over(keys)
    has_unit = pl.col("unit_w").is_not_null()
    df = df.with_columns(
        _w_m2v=pl.when(has_unit).then("unit_w").otherwise(pl.col("value") * pl.col("b") / sb),
        _lo_m2v=pl.when(has_unit).then("unit_lo").otherwise(pl.col("value_lo") * pl.col("b") / sb),
        _hi_m2v=pl.when(has_unit).then("unit_hi").otherwise(pl.col("value_hi") * pl.col("b") / sb),
    )
    # (2) MODIS, m2v 값이 없는 달: 입력 MODIS 점으로 예측
    need = df.filter(modis & ~m2v_month.fill_null(False))
    pred_parts = []
    for (y, m), sub in need.group_by("year", "month"):
        r = ctx.m2v().predict_month(sub, int(y), int(m), as_of)
        if r.height:
            pred_parts.append(r.select("cell_id", "ratio", "ratio_lo", "ratio_hi").with_columns(
                year=pl.lit(y, dtype=pl.Int16), month=pl.lit(m, dtype=pl.Int8)))
    if pred_parts:
        df = df.join(pl.concat(pred_parts), on=keys, how="left")
    else:
        df = df.with_columns(ratio=pl.lit(None, dtype=pl.Float64), ratio_lo=pl.lit(None, dtype=pl.Float64),
                             ratio_hi=pl.lit(None, dtype=pl.Float64))
    viirs_w = pl.col("b")
    out = df.with_columns(
        harmonized_weight=pl.when(~modis).then(viirs_w)
        .when(m2v_month.fill_null(False) & pl.col("is_reference")).then("_w_m2v")
        .otherwise("ratio"),
        weight_lo=pl.when(~modis).then(viirs_w)
        .when(m2v_month.fill_null(False) & pl.col("is_reference")).then("_lo_m2v")
        .otherwise("ratio_lo"),
        weight_hi=pl.when(~modis).then(viirs_w)
        .when(m2v_month.fill_null(False) & pl.col("is_reference")).then("_hi_m2v")
        .otherwise("ratio_hi"),
        weight_source=pl.when(pl.col("ref_rule") == "n21_fallback").then(pl.lit("n21_bridge"))
        .when(pl.col("ref_rule") == "modis_fallback").then(pl.lit("modis_fallback_m2v_predicted"))
        .when(pl.col("is_reference") & pl.col("value_source").is_not_null()).then("value_source")
        .when(pl.col("satellite") == "N").then(pl.lit("snpp_observed"))
        .when(pl.col("satellite") == "N20").then(pl.lit("n20_bridge"))
        .when(pl.col("satellite") == "N21").then(pl.lit("n21_bridge"))
        .when(modis & pl.col("ratio").is_not_null()).then(pl.lit("m2v_predicted"))
        .otherwise(pl.lit("m2v_no_common_day")),
    ).with_columns(
        # 기준이 아닌 VIIRS 점은 구간이 없음 (센서 계수만)
        weight_lo=pl.when(~modis & ~pl.col("is_reference")).then(None).otherwise("weight_lo"),
        weight_hi=pl.when(~modis & ~pl.col("is_reference")).then(None).otherwise("weight_hi"),
    ).with_columns(
        method_version=pl.lit(METHOD_VERSION),
        harmonized_weight=pl.col("harmonized_weight").cast(pl.Float32),
        weight_lo=pl.col("weight_lo").cast(pl.Float32), weight_hi=pl.col("weight_hi").cast(pl.Float32),
    )
    return out.select(OUT_COLS).sort("acq_datetime_utc")


def unit_weights_month(year: int, month: int) -> pl.DataFrame | None:
    """저장된 통합 핫스팟(reference)에서 (칸, 월, 위성)별 점 하나의 가중치. 같은 칸·월·위성의 기준 점은 가중치가 같다."""
    d = HOTSPOTS_DIR / "reference" / f"year={year}" / f"month={month:02d}"
    if not d.exists():
        return None
    return (pl.scan_parquet(d / "*.parquet")
            .group_by("cell_id", "satellite")
            .agg(unit_w=pl.col("harmonized_weight").cast(pl.Float64).first(),
                 unit_lo=pl.col("weight_lo").cast(pl.Float64).first(),
                 unit_hi=pl.col("weight_hi").cast(pl.Float64).first())
            .with_columns(year=pl.lit(year, dtype=pl.Int16), month=pl.lit(month, dtype=pl.Int8))
            .collect())


def month_points(year: int, month: int) -> pl.LazyFrame:
    """그 (현지) 달의 감지 점 (중복 제거 적용). 현지 날짜는 UTC와 최대 ±1일 차이라 앞뒤 하루를 더 읽는다."""
    first = date(year, month, 1)
    last = date(year, month, calendar.monthrange(year, month)[1])
    return load_hotspots().filter(pl.col("acq_date").is_between(first - timedelta(days=1), last + timedelta(days=1)))


def month_cell_values(year: int, month: int) -> pl.DataFrame:
    return pl.scan_parquet(HCM_PATH).filter(pl.col("year") == year, pl.col("month") == month).collect()


def harmonize_month(year: int, month: int, ctx: Context | None = None) -> pl.DataFrame:
    ctx = ctx or Context()
    out = harmonize_detections(month_points(year, month), month_cell_values(year, month), ctx)
    return out.filter(pl.col("date_local").dt.year() == year, pl.col("date_local").dt.month() == month)


VIIRS_SOURCES = ("snpp_observed", "n20_bridge")


def residual_month(year: int, month: int, points: pl.DataFrame) -> pl.DataFrame:
    """점으로 나눌 수 없는 값 (칸·월). residual_type:
      modis_unseen     MODIS 기간: 기준 점이 없는 칸·월의 m2v 예측 + 배경 예측(감지 없는 칸)
      missing_day_fill 실측 기간(완결된 달): 관측이 빠진 날을 달 전체로 환산한 몫 = 칸 value − Σ 기준 점 가중치
                       (빠진 날 감지가 있었던 칸은 음수일 수 있음)
    진행 중인 달(status=partial)은 value가 오늘까지 실제 합계라 residual이 없다."""
    keys = ["cell_id", "year", "month"]
    cv = month_cell_values(year, month)
    yl, ml = pl.lit(year, dtype=pl.Int16), pl.lit(month, dtype=pl.Int8)
    ref_w = points.filter("is_reference").group_by("cell_id").agg(w=pl.col("harmonized_weight").cast(pl.Float64).sum()) \
        .with_columns(year=yl, month=ml)
    parts = []
    modis_cv = cv.filter(pl.col("value_source").is_in(MODIS_M2V_SOURCES))
    if modis_cv.height:
        parts.append(modis_cv.join(ref_w, on=keys, how="anti").filter(pl.col("value") > 0).select(
            *keys, "value", "value_lo", "value_hi", "value_source", residual_type=pl.lit("modis_unseen")))
    viirs_cv = cv.filter(pl.col("value_source").is_in(VIIRS_SOURCES), pl.col("status") != "partial")
    if viirs_cv.height:
        f = viirs_cv.join(ref_w, on=keys, how="left").with_columns(fill=pl.col("value") - pl.col("w").fill_null(0))
        parts.append(f.filter(pl.col("fill").abs() > 1e-9).select(
            *keys, value="fill", value_lo="fill", value_hi="fill", value_source="value_source",
            residual_type=pl.lit("missing_day_fill")))
    bg_path = HCM_BACKGROUND_DIR / f"{year}.parquet"
    if bg_path.exists():
        bg = pl.scan_parquet(bg_path).filter(pl.col("year") == year, pl.col("month") == month).collect()
        parts.append(bg.select(*keys, pl.col("value").cast(pl.Float64), value_lo=pl.lit(None, dtype=pl.Float64),
                               value_hi=pl.lit(None, dtype=pl.Float64), value_source=pl.lit("modis_m2v_background"),
                               residual_type=pl.lit("modis_unseen")))
    if not parts:
        return pl.DataFrame(schema={"cell_id": pl.Int64, "year": pl.Int16, "month": pl.Int8, "value": pl.Float64,
                                    "value_lo": pl.Float64, "value_hi": pl.Float64, "value_source": pl.String,
                                    "residual_type": pl.String})
    return pl.concat(parts, how="diagonal_relaxed")


# ── 전체 생성 (연도 단위로 읽어 달별로 나눔) ───────────────────────────────

import argparse  # noqa: E402
import time  # noqa: E402

from tqdm import tqdm  # noqa: E402

from planetarium.config import REPORTS_DIR  # noqa: E402
from planetarium.runlog import start_run_log  # noqa: E402

CELL_DAY_PATH = PROCESSED_DATA_DIR / "harmonized_cell_day.parquet"
OTHER_START = date(2012, 2, 1)   # 기준이 아닌 점은 2012-02 이후 VIIRS만 (MODIS 비기준 점은 가중치가 null이라 만들지 않음)
VIIRS = ["N", "N20", "N21"]


def write_part(df: pl.DataFrame, layer: str, year: int, month: int) -> int:
    if df.height == 0:
        return 0
    d = HOTSPOTS_DIR / layer / f"year={year}" / f"month={month:02d}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "part-0.parquet"
    df.write_parquet(f, compression="zstd", row_group_size=500_000)
    return f.stat().st_size


def cell_day_rows(pts: pl.DataFrame, res: pl.DataFrame, year: int, month: int) -> pl.DataFrame:
    """칸 × 현지 날짜: 기준 점 가중치 합.
    residual(점으로 나눌 수 없는 값)은 칸·월 단위로 그 달 1일 행의 residual_month에만 둔다 (일수로 균등 배분하지 않음).
    residual 칸·월은 정의상 기준 점이 없는 칸이라 점 행과 겹치지 않는다."""
    ref = pts.filter("is_reference")
    day = ref.group_by("cell_id", "date_local").agg(
        value=pl.col("harmonized_weight").cast(pl.Float64).sum(),
        value_lo=pl.col("weight_lo").cast(pl.Float64).sum(),
        value_hi=pl.col("weight_hi").cast(pl.Float64).sum(),
        lo_null=pl.col("weight_lo").is_null().any(),
        n_points=pl.len().cast(pl.Int32),
        value_source=pl.col("weight_source").mode().first(),
    ).with_columns(
        # Terra 단독 기간처럼 점 구간이 null이면 칸·일 구간도 null
        value_lo=pl.when("lo_null").then(None).otherwise("value_lo"),
        value_hi=pl.when("lo_null").then(None).otherwise("value_hi"),
        residual_month=pl.lit(None, dtype=pl.Float64),
    ).drop("lo_null")
    resid = (res.group_by("cell_id").agg(residual_month=pl.col("value").sum(), value_source=pl.col("value_source").first())
             .with_columns(date_local=pl.lit(date(year, month, 1)), value=pl.lit(0.0),
                           value_lo=pl.lit(None, dtype=pl.Float64), value_hi=pl.lit(None, dtype=pl.Float64),
                           n_points=pl.lit(0, dtype=pl.Int32)))
    out = pl.concat([day, resid.select(day.columns)], how="vertical_relaxed")
    return out.with_columns(year=pl.lit(year, dtype=pl.Int16), month=pl.lit(month, dtype=pl.Int8)).select(
        "cell_id", "date_local", "year", "month", "value", "value_lo", "value_hi", "n_points", "value_source",
        "residual_month").sort("cell_id", "date_local")


def build(years: list[int]) -> None:
    ctx = Context()
    tmp_cd = CELL_DAY_PATH.with_suffix(".parts")
    tmp_res = RESIDUAL_PATH.with_suffix(".parts")
    for d in (tmp_cd, tmp_res):
        d.mkdir(parents=True, exist_ok=True)
    log = []
    hcm_months = pl.scan_parquet(HCM_PATH).select("year", "month").unique().collect()
    bar = tqdm(years, desc="harmonized hotspots", unit="year", dynamic_ncols=True)
    for y in bar:
        t0 = time.perf_counter()
        months = sorted(hcm_months.filter(pl.col("year") == y)["month"].to_list())
        # 메모리 절약: 연도 전체를 한 번에 읽지 않고 달마다 그 달(±1일) 점만 읽는다
        cv_year = pl.scan_parquet(HCM_PATH).filter(pl.col("year") == y).collect()
        n_ref = n_oth = size = 0
        for m in tqdm(months, desc=f"  {y}", unit="month", leave=False, dynamic_ncols=True):
            first = date(y, m, 1)
            last = date(y, m, calendar.monthrange(y, m)[1])
            pts_raw = month_points(y, m).collect()
            pts = harmonize_detections(pts_raw, cv_year.filter(pl.col("month") == m), ctx).filter(
                pl.col("date_local").is_between(first, last))
            ref = pts.filter("is_reference")
            oth = pts.filter(~pl.col("is_reference"), pl.col("satellite").is_in(VIIRS), pl.col("date_local") >= OTHER_START)
            size += write_part(ref, "reference", y, m) + write_part(oth, "other", y, m)
            n_ref += ref.height
            n_oth += oth.height
            res = residual_month(y, m, pts)
            # 메모리 절약: 달마다 결과를 바로 디스크에 쓰고 버린다
            res.filter(pl.col("value_source") != "modis_m2v_background").write_parquet(tmp_res / f"{y}-{m:02d}.parquet")
            cell_day_rows(pts, res, y, m).write_parquet(tmp_cd / f"{y}-{m:02d}.parquet", compression="zstd")
            del pts, ref, oth, res
        sec = time.perf_counter() - t0
        from planetarium.runlog import peak_memory_gb
        log.append({"year": y, "reference_rows": n_ref, "other_rows": n_oth, "points_mb": round(size / 1e6, 1),
                    "seconds": round(sec), "peak_mem_gb": round(peak_memory_gb() or 0, 2)})
        tqdm.write(f"[{y}] 기준 {n_ref:,}행, 기타 VIIRS {n_oth:,}행, {size / 1e6:,.0f} MB, {sec:.0f}초, "
                   f"최대 메모리 {log[-1]['peak_mem_gb']} GB")
        pl.DataFrame(log).write_csv(REPORTS_DIR / "hotspots_build_years.csv")
    for parts, path in ((tmp_cd, CELL_DAY_PATH), (tmp_res, RESIDUAL_PATH)):
        files = sorted(parts.glob("*.parquet"))
        if files:
            pl.scan_parquet(files).sink_parquet(path, compression="zstd")


def build_months(months: list[tuple[int, int]]) -> list[dict]:
    """지정한 (연, 월)만 다시 만든다 (월 단위로 읽고 씀). 점 파티션을 덮어쓰고,
    harmonized_cell_day·harmonized_residual_cell_month에서는 그 달 행만 바꾼다. 반환: 달별 전/후 평균 가중치."""
    from planetarium.runlog import peak_memory_gb
    ctx = Context()
    keys = ["year", "month"]
    target = pl.DataFrame(months, schema={"year": pl.Int16, "month": pl.Int8}, orient="row")
    log = []
    import shutil
    import tempfile
    tmp_root = Path(tempfile.mkdtemp(dir=PROCESSED_DATA_DIR))
    (tmp_root / "cd").mkdir()
    (tmp_root / "res").mkdir()
    for y, m in tqdm(months, desc="바뀐 달 다시 만들기", unit="month", dynamic_ncols=True):
        t0 = time.perf_counter()
        old = pl.scan_parquet(HOTSPOTS_DIR / "reference" / f"year={y}" / f"month={m:02d}" / "*.parquet").select(
            w=pl.col("harmonized_weight").cast(pl.Float64)).collect()["w"]
        first, last = date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
        pts = harmonize_detections(month_points(y, m).collect(), month_cell_values(y, m), ctx).filter(
            pl.col("date_local").is_between(first, last))
        ref = pts.filter("is_reference")
        oth = pts.filter(~pl.col("is_reference"), pl.col("satellite").is_in(VIIRS), pl.col("date_local") >= OTHER_START)
        write_part(ref, "reference", y, m)
        write_part(oth, "other", y, m)
        res = residual_month(y, m, pts)
        # 메모리 절약: 달마다 결과를 바로 디스크에 쓰고 버린다
        res.filter(pl.col("value_source") != "modis_m2v_background").write_parquet(tmp_root / "res" / f"{y}-{m:02d}.parquet")
        cell_day_rows(pts, res, y, m).write_parquet(tmp_root / "cd" / f"{y}-{m:02d}.parquet", compression="zstd")
        log.append({"year": y, "month": m, "reference_rows": ref.height, "w_mean_before": float(old.mean()),
                    "w_mean_after": float(ref["harmonized_weight"].cast(pl.Float64).mean()),
                    "missing_day_fill": float(res.filter(pl.col("residual_type") == "missing_day_fill")["value"].sum()),
                    "seconds": round(time.perf_counter() - t0, 1), "peak_mem_gb": round(peak_memory_gb() or 0, 2)})
        tqdm.write(f"[{y}-{m:02d}] 기준 {ref.height:,}점, 평균 가중치 {log[-1]['w_mean_before']:.4f} → "
                   f"{log[-1]['w_mean_after']:.4f}, missing_day_fill {log[-1]['missing_day_fill']:,.1f}")
    # 표 갱신: 그 달 행만 교체 (스트리밍으로 다시 씀)
    for path, sub in ((CELL_DAY_PATH, "cd"), (RESIDUAL_PATH, "res")):
        new = [pl.scan_parquet(f) for f in sorted((tmp_root / sub).glob("*.parquet"))]
        tmp = path.with_suffix(".new.parquet")
        old_lf = pl.scan_parquet(path)
        if "kind" in old_lf.collect_schema().names():  # 이전 형식: kind → residual_type
            old_lf = old_lf.with_columns(residual_type=pl.lit("modis_unseen")).drop("kind")
        kept = old_lf.join(target.lazy(), on=keys, how="anti")
        pl.concat([kept, *new], how="diagonal_relaxed").sink_parquet(tmp, compression="zstd")
        tmp.replace(path)
    shutil.rmtree(tmp_root, ignore_errors=True)
    pl.DataFrame(log).write_csv(REPORTS_DIR / "hotspots_rebuild_months.csv")
    return log


def finalize_projection() -> pl.DataFrame:
    """harmonized_cell_month에 value_projected를 더하고, 진행 중인 달(status=partial)은 value를 오늘까지 실제 합계로 바꾼다.
      완결된 달: value_projected = value
      진행 중인 달: value_projected = 기존 value(관측일 환산 달 전체 추정), value/lo/hi = 기준 점 가중치 합(month-to-date)
    harmonized_cell_day에는 진행 중인 달 1일 행에 value_projected_month(칸·월 달 전체 추정)를 둔다."""
    keys = ["cell_id", "year", "month"]
    hcm = pl.scan_parquet(HCM_PATH)
    cols = hcm.collect_schema().names()
    if "value_projected" not in cols:
        hcm = hcm.with_columns(value_projected=pl.col("value"))
    partial = hcm.filter(pl.col("status") == "partial").select("year", "month").unique().collect()
    mtd = (pl.scan_parquet(CELL_DAY_PATH).join(partial.lazy(), on=["year", "month"], how="semi")
           .group_by(keys).agg(mtd=pl.col("value").sum(), mtd_lo=pl.col("value_lo").sum(), mtd_hi=pl.col("value_hi").sum())
           .collect())
    is_p = pl.col("status") == "partial"
    new = (hcm.join(mtd.lazy(), on=keys, how="left")
           .with_columns(value_projected=pl.when(is_p).then(pl.col("value_projected")).otherwise(pl.col("value")))
           .with_columns(value=pl.when(is_p).then(pl.col("mtd").fill_null(0.0)).otherwise("value"),
                         value_lo=pl.when(is_p).then(pl.col("mtd_lo").fill_null(0.0)).otherwise("value_lo"),
                         value_hi=pl.when(is_p).then(pl.col("mtd_hi").fill_null(0.0)).otherwise("value_hi"))
           .drop("mtd", "mtd_lo", "mtd_hi"))
    order = [c for c in cols if c != "value_projected"]
    order.insert(order.index("value_hi") + 1, "value_projected")
    tmp = HCM_PATH.with_suffix(".new.parquet")
    new.select(order).sink_parquet(tmp, compression="zstd")
    tmp.replace(HCM_PATH)
    # cell_day: 진행 중인 달 1일 행에 칸·월 달 전체 추정
    proj = pl.scan_parquet(HCM_PATH).join(partial.lazy(), on=["year", "month"], how="semi").select(
        *keys, value_projected_month="value_projected").collect()
    cd = pl.scan_parquet(CELL_DAY_PATH)
    if "value_projected_month" in cd.collect_schema().names():
        cd = cd.drop("value_projected_month")
    first_day = pl.col("date_local").dt.day() == 1
    cd2 = cd.join(proj.lazy(), on=keys, how="left").with_columns(
        value_projected_month=pl.when(first_day).then("value_projected_month"))
    # 진행 중인 달에 1일 행이 없는 칸(1일에 점이 없음)은 추정만 담은 행을 하나 만든다
    have_first = cd.filter(first_day).select(*keys).unique()
    extra = (proj.lazy().join(have_first, on=keys, how="anti")
             .with_columns(date_local=pl.date(pl.col("year"), pl.col("month"), 1), value=pl.lit(0.0),
                           value_lo=pl.lit(0.0), value_hi=pl.lit(0.0), n_points=pl.lit(0, dtype=pl.Int32),
                           value_source=pl.lit("n20_bridge"), residual_month=pl.lit(None, dtype=pl.Float64)))
    tmp = CELL_DAY_PATH.with_suffix(".new.parquet")
    pl.concat([cd2, extra], how="diagonal_relaxed").sink_parquet(tmp, compression="zstd")
    tmp.replace(CELL_DAY_PATH)
    return partial


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="통합 핫스팟(감지 단위) 생성")
    ap.add_argument("--years", nargs="*", type=int, help="연도 (기본: 2000~2026 전부)")
    ap.add_argument("--months", nargs="*", help="이 달들만 다시 만들기: YYYY-MM ... (점 파티션·cell_day·residual의 그 달만 교체)")
    ap.add_argument("--finalize", action="store_true", help="value_projected·진행 중인 달 month-to-date 반영")
    args = ap.parse_args(argv)
    start_run_log("hotspots")
    if args.months is not None or args.finalize:
        if args.months:
            build_months([(int(s[:4]), int(s[5:7])) for s in args.months])
        if args.finalize:
            print("진행 중인 달:", finalize_projection().rows())
        return
    years = args.years or list(range(2000, 2027))
    t0 = time.perf_counter()
    build(years)
    print(f"[완료] {HOTSPOTS_DIR} ({time.perf_counter() - t0:,.0f}초)")


if __name__ == "__main__":
    main()
