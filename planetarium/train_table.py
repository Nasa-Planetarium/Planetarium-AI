"""M1 학습·채점·적용용 표: '공통 정상일' 기준 MODIS(Aqua·Terra) 칸 × 월 특성과 S-NPP 감지 수 (분석 범위: config.ANALYSIS_BBOX).

왜 공통 정상일인가
    위성마다 자료가 빠진 날이 다르다. S-NPP가 10일 빠진 달은 S-NPP 월 합계가 작아지는데 MODIS 특성은 그 10일을
    포함하므로 모델이 S-NPP를 실제보다 적게 배운다. 그래서 각 월에서 비교 위성이 '모두 status=ok'인 날(common_days)만으로
    타깃과 특성을 다시 집계하고, 합계 값은 days_in_month / common_days를 곱해 월 기준으로 환산한다 (원래 값은 *_raw).

모드
    train : 2012-02 ~ 2025-12. 공통일 = Aqua·Terra·S-NPP 모두 ok. 타깃 y = S-NPP 감지 수(환산), y_raw = 공통일 원값.
    apply : 2000-11 ~ 2012-01. 공통일 = Aqua·Terra ok (Aqua 운용 전 2000-11 ~ 2002-06은 Terra만, terra_only=True).
            2012-01은 S-NPP 부분 월이라 MODIS 보정 값을 쓰기 위해 apply에 포함.
    공통일이 그 달 일수의 50% 미만이면 제외.

날짜 대응
    위성 상태는 UTC 날짜, cell_day는 현지 태양시 날짜. '현지 날짜 D = UTC 날짜 D'로 대응시키는 근사를 쓴다
    (낮 관측은 대부분 같은 날짜, 밤 관측은 하루 어긋날 수 있음).

희소 패널 (전 세계라 조밀 격자를 저장하지 않는다)
    data/processed/panel_common_0.1deg/<mode>_<year>.parquet
        행 = 공통일에 Aqua·Terra·S-NPP 중 하나라도 감지가 있거나 주변 3×3에 MODIS 감지가 있는 칸·월.
        여기 없는 (격자 칸 × 정상 월)은 모든 MODIS 특성·주변 합·타깃이 0이다.
        → B1, 그리고 '감지·주변 감지가 모두 0이면 예측 0' 규칙을 켠 모델은 이 희소 패널만으로 전체 칸 합계가 정확하다.
    data/processed/grid_cells_0.1deg.parquet
        채점·적용 격자: 2012~2025 감지 이력 칸(어느 위성이든) ∪ 한국 칸, 지역 속성(국가·대륙·하위지역) 포함.
    data/processed/panel_months.parquet   (mode, year, month, days_in_month, common_days, scale, excluded, terra_only)

학습 표
    data/processed/train_modis_snpp.parquet
        positive(세 위성 중 하나라도 공통일 감지)를 --pos-frac 비율로 표본 + 셋 다 0인 이력 칸·월을 positive 표본 수 × --zero-ratio.
        0행은 (이력 칸 × 정상 월)에서 무작위로 뽑는다 (seed 42).

특성은 MODIS에서만 (S-NPP 정보·MODIS 처리 버전 없음). 지역 특성: continent, subregion (Natural Earth, 범주형).

실행
    uv run python -m planetarium.train_table                         # 패널 + 학습 표
    uv run python -m planetarium.train_table --pos-frac 0.3 --zero-ratio 1.0
    uv run python -m planetarium.train_table --skip-panel            # 학습 표만 다시 표본
"""

from __future__ import annotations

import argparse
import calendar
import math
import time
from datetime import date

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium.config import ANALYSIS_BBOX, PROCESSED_DATA_DIR, REPORTS_DIR, SEED, cell_day_dir
from planetarium.korea import KOREA_CELLS_PATH
from planetarium.monthly import CELL_MONTH_PATH, SAT_MONTH_PATH, STATUS_PATH, in_analysis
from planetarium.runlog import start_run_log
from planetarium.world import WORLD_CELLS_PATH

TRAIN_PATH = PROCESSED_DATA_DIR / "train_modis_snpp.parquet"
TRAIN_V1_PATH = PROCESSED_DATA_DIR / "train_modis_snpp_v1_obsdays.parquet"
PANEL_DIR = PROCESSED_DATA_DIR / "panel_common_0.1deg"
PANEL_MONTHS_PATH = PROCESSED_DATA_DIR / "panel_months.parquet"
GRID_PATH = PROCESSED_DATA_DIR / "grid_cells_0.1deg.parquet"
N_LON = 3600

TRAIN_START, TRAIN_END = (2012, 2), (2025, 12)
APPLY_START, APPLY_END = (2000, 11), (2012, 1)
AQUA_FIRST_MONTH = (2002, 7)
MIN_COMMON_FRAC = 0.5

SAT_PREFIX = {"Aqua": "aqua", "Terra": "terra", "S-NPP": "snpp"}
SUM_VALUES = ["n_det", "n_det_conf", "active_days", "frp_sum"]
OTHER_VALUES = ["n_night", "frp_max"]

NUMERIC_FEATURES = [
    "aqua_n_det", "terra_n_det", "aqua_active_days", "terra_active_days",
    "aqua_n_det_conf", "terra_n_det_conf",
    "aqua_frp_sum", "aqua_frp_max", "terra_frp_sum", "terra_frp_max",
    "aqua_night_ratio", "terra_night_ratio",
    "aqua_n_det_nb", "terra_n_det_nb",
    "month_sin", "month_cos", "cell_lat", "cell_lon",
    "terra_drift",
    "aqua_obs_days", "terra_obs_days",
]
CATEGORICAL_FEATURES = ["continent", "subregion"]
MODIS_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TERRA_ONLY_FEATURES = [f for f in MODIS_FEATURES if not f.startswith("aqua_")]
TARGET = "y"
REGION_COLS = ["adm0_a3", "continent", "subregion", "is_korea"]
ID_COLS = ["cell_id", "cell_lat", "cell_lon", "year", "month", *REGION_COLS,
           "days_in_month", "common_days", "scale", "terra_only"]
RAW_COLS = [f"{q}_{c}" for q in ("aqua", "terra") for c in SUM_VALUES + ["n_det_nb"]]


def ym(y: int, m: int) -> int:
    return y * 12 + (m - 1)


def months_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return [(k // 12, k % 12 + 1) for k in range(ym(*a), ym(*b) + 1)]


# ── 공통 정상일 ───────────────────────────────────────────────────────────

def common_days(mode: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(공통일 날짜표[date_local, year, month], 월표)"""
    st = pl.read_parquet(STATUS_PATH).filter(pl.col("satellite").is_in(list(SAT_PREFIX)))
    w = st.pivot(on="satellite", index="date_utc", values="status")
    ok = lambda s: (pl.col(s) == "ok").fill_null(False) if s in w.columns else pl.lit(False)  # noqa: E731
    if mode == "train":
        start, end = TRAIN_START, TRAIN_END
        flag = ok("Aqua") & ok("Terra") & ok("S-NPP")
    else:
        start, end = APPLY_START, APPLY_END
        flag = ok("Terra") & (ok("Aqua") | (pl.col("date_utc") < date(*AQUA_FIRST_MONTH, 1)))
    w = w.with_columns(common=flag, year=pl.col("date_utc").dt.year().cast(pl.Int16),
                       month=pl.col("date_utc").dt.month().cast(pl.Int8))
    cnt = {(y, m): c for y, m, c in w.group_by("year", "month").agg(pl.col("common").sum()).iter_rows()}
    rows = []
    for y, m in months_between(start, end):
        dim = calendar.monthrange(y, m)[1]
        c = int(cnt.get((y, m), 0))
        excluded = c < MIN_COMMON_FRAC * dim
        rows.append({"mode": mode, "year": y, "month": m, "days_in_month": dim, "common_days": c,
                     "scale": dim / c if c else None, "excluded": excluded,
                     "reason": f"common_days {c}/{dim} < {MIN_COMMON_FRAC:.0%}" if excluded else "",
                     "terra_only": mode == "apply" and ym(y, m) < ym(*AQUA_FIRST_MONTH)})
    months = pl.DataFrame(rows).with_columns(pl.col("year").cast(pl.Int16), pl.col("month").cast(pl.Int8),
                                             pl.col("days_in_month", "common_days").cast(pl.Int16))
    good = months.filter(~pl.col("excluded")).select("year", "month")
    dates = w.filter("common").join(good, on=["year", "month"], how="semi").select(
        pl.col("date_utc").alias("date_local"), "year", "month")
    return dates, months


# ── 집계 ──────────────────────────────────────────────────────────────────

def aggregate_common(dates: pl.DataFrame, sats: dict[str, str] = SAT_PREFIX) -> pl.DataFrame:
    """분석 범위 cell_day에서 주어진 날짜만 골라 칸 × 월 × 위성 집계 (원값)."""
    return (
        pl.scan_parquet(cell_day_dir() / "*.parquet")
        .filter(in_analysis("cell_lat", "cell_lon"), pl.col("satellite").is_in(list(sats)))
        .join(dates.lazy().select("date_local"), on="date_local", how="semi")
        .group_by("cell_id", "date_local", "year", "month", "satellite")
        .agg(pl.col("n_det", "n_det_conf", "n_night", "frp_sum").sum(), pl.col("frp_max").max())
        .group_by("cell_id", "year", "month", "satellite")
        .agg(pl.col("n_det", "n_det_conf", "n_night", "frp_sum").sum(), pl.col("frp_max").max(),
             active_days=pl.len())
        .collect()
    )


def to_wide(agg: pl.DataFrame, sats: dict[str, str] = SAT_PREFIX) -> pl.DataFrame:
    keys = ["cell_id", "year", "month"]
    out = None
    for sat, p in sats.items():
        cols = ["n_det"] if p in ("snpp", "noaa20", "noaa21") else SUM_VALUES + OTHER_VALUES
        sub = agg.filter(pl.col("satellite") == sat).select(*keys, *[pl.col(c).alias(f"{p}_{c}") for c in cols])
        out = sub if out is None else out.join(sub, on=keys, how="full", coalesce=True)
    return out


def neighbor_sums(wide: pl.DataFrame) -> pl.DataFrame:
    """Aqua·Terra 감지 수(공통일 원값)의 3×3 칸 합 (가운데 포함). 경도는 날짜변경선에서 이어 붙인다."""
    base = wide.select("cell_id", "year", "month",
                       a=pl.col("aqua_n_det").fill_null(0), t=pl.col("terra_n_det").fill_null(0))
    li, lj = pl.col("cell_id") // N_LON, pl.col("cell_id") % N_LON
    shifted = pl.concat([
        base.with_columns(cell_id=(li + di) * N_LON + (lj + dj) % N_LON) for di in (-1, 0, 1) for dj in (-1, 0, 1)
    ])
    return shifted.group_by("cell_id", "year", "month").agg(
        aqua_n_det_nb=pl.col("a").sum(), terra_n_det_nb=pl.col("t").sum()).filter(
        (pl.col("aqua_n_det_nb") + pl.col("terra_n_det_nb")) > 0)


def cell_attrs() -> pl.DataFrame:
    """칸 → 지역 속성 (world_cells + korea_cells). 바다 등 육지가 아닌 칸은 continent='none'."""
    wc = pl.read_parquet(WORLD_CELLS_PATH).select("cell_id", "adm0_a3", "continent", "subregion")
    kc = pl.read_parquet(KOREA_CELLS_PATH).select("cell_id", "is_korea")
    return wc.join(kc, on="cell_id", how="full", coalesce=True)


def with_attrs(df: pl.DataFrame, attrs: pl.DataFrame) -> pl.DataFrame:
    return df.join(attrs, on="cell_id", how="left").with_columns(
        pl.col("continent").fill_null("none"), pl.col("subregion").fill_null("none"),
        pl.col("is_korea").fill_null(False),
        cell_lat=((pl.col("cell_id") // N_LON + 0.5) * 0.1 - 90).cast(pl.Float32),
        cell_lon=((pl.col("cell_id") % N_LON + 0.5) * 0.1 - 180).cast(pl.Float32),
    )


def build_grid() -> pl.DataFrame:
    """채점·적용 격자: 2012~2025 감지 이력 칸 ∪ 한국 칸."""
    cm = pl.scan_parquet(CELL_MONTH_PATH).filter(
        (pl.col("year").cast(pl.Int32) * 12 + pl.col("month").cast(pl.Int32) - 1).is_between(ym(*TRAIN_START), ym(*TRAIN_END)),
        in_analysis("cell_lat", "cell_lon"))
    hist = cm.select("cell_id").unique().collect().with_columns(in_history=pl.lit(True))
    kor = pl.read_parquet(KOREA_CELLS_PATH).filter("is_korea").select("cell_id")
    cells = pl.concat([hist.select("cell_id"), kor]).unique().join(hist, on="cell_id", how="left").with_columns(
        pl.col("in_history").fill_null(False))
    g = with_attrs(cells, cell_attrs()).sort("cell_id")
    g.write_parquet(GRID_PATH)
    return g


def build_panel(mode: str, attrs: pl.DataFrame) -> pl.DataFrame:
    """연도별 희소 패널 저장, 월표 반환."""
    dates, months = common_days(mode)
    good = months.filter(~pl.col("excluded"))
    sm = pl.read_parquet(SAT_MONTH_PATH).filter(pl.col("satellite").is_in(["Aqua", "Terra"])).pivot(
        on="satellite", index=["year", "month"], values="obs_days")
    sm = sm.rename({k: f"{k.lower()}_obs_days" for k in ("Aqua", "Terra") if k in sm.columns})
    if "aqua_obs_days" not in sm.columns:
        sm = sm.with_columns(aqua_obs_days=pl.lit(None, dtype=pl.Int16))
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for f in PANEL_DIR.glob(f"{mode}_*.parquet"):
        f.unlink()
    keys = ["cell_id", "year", "month"]
    for year in tqdm(sorted(good["year"].unique().to_list()), desc=f"패널 {mode}", unit="year"):
        d = dates.filter(pl.col("year") == year)
        wide = to_wide(aggregate_common(d))
        nb = neighbor_sums(wide)
        p = (
            wide.join(nb, on=keys, how="full", coalesce=True)
            .join(good.filter(pl.col("year") == year).drop("mode", "excluded", "reason"), on=["year", "month"], how="inner")
            .join(sm, on=["year", "month"], how="left")
        )
        if mode == "apply":  # apply 기간에는 S-NPP가 없으므로 MODIS(또는 주변) 감지가 있는 행만
            pass
        for c in RAW_COLS + ["aqua_n_night", "terra_n_night", "aqua_frp_max", "terra_frp_max", "snpp_n_det"]:
            if c not in p.columns:
                p = p.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
        p = p.with_columns(pl.col(RAW_COLS + ["aqua_n_night", "terra_n_night", "aqua_frp_max", "terra_frp_max",
                                              "snpp_n_det"]).fill_null(0))
        p = with_attrs(p, attrs).with_columns(
            *[pl.col(c).alias(f"{c}_raw") for c in RAW_COLS],
            *[(pl.col(c) * pl.col("scale")).alias(c) for c in RAW_COLS],
            aqua_night_ratio=pl.when(pl.col("aqua_n_det") > 0).then(pl.col("aqua_n_night") / pl.col("aqua_n_det")),
            terra_night_ratio=pl.when(pl.col("terra_n_det") > 0).then(pl.col("terra_n_night") / pl.col("terra_n_det")),
            month_sin=(2 * math.pi * pl.col("month").cast(pl.Float64) / 12).sin(),
            month_cos=(2 * math.pi * pl.col("month").cast(pl.Float64) / 12).cos(),
            terra_drift=(pl.col("year") >= 2021).cast(pl.Int8),
        )
        if mode == "train":
            p = p.with_columns(y_raw=pl.col("snpp_n_det").cast(pl.Int32), y=pl.col("snpp_n_det") * pl.col("scale"))
        else:
            p = p.with_columns(y_raw=pl.lit(None, dtype=pl.Int32), y=pl.lit(None, dtype=pl.Float64))
            p = p.with_columns(*[pl.when(pl.col("terra_only")).then(None).otherwise(pl.col(c)).alias(c)
                                 for c in NUMERIC_FEATURES if c.startswith("aqua_")])
        p = p.select(*ID_COLS, *[c for c in MODIS_FEATURES if c not in ID_COLS],
                     *[f"{c}_raw" for c in RAW_COLS], "y", "y_raw").sort("month", "cell_id")
        p.write_parquet(PANEL_DIR / f"{mode}_{year}.parquet", compression="zstd")
    return months


def load_panel(mode: str, years: list[int] | None = None) -> pl.LazyFrame:
    files = sorted(PANEL_DIR.glob(f"{mode}_*.parquet"))
    if years is not None:
        files = [f for f in files if int(f.stem.split("_")[1]) in years]
    return pl.scan_parquet(files)


def panel_months() -> pl.DataFrame:
    return pl.read_parquet(PANEL_MONTHS_PATH)


def is_positive() -> pl.Expr:
    return (pl.col("aqua_n_det_raw") + pl.col("terra_n_det_raw") + pl.col("y_raw")) > 0


# ── 학습 표 ───────────────────────────────────────────────────────────────

def build_train_table(pos_frac: float, zero_ratio: float) -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    panel = load_panel("train")
    keys = ["cell_id", "year", "month"]
    pos_keys = panel.filter(is_positive()).select(keys).collect()
    n_pos_all = pos_keys.height
    # positive 표본: 메모리를 아끼려고 (칸, 연, 월) 해시로 결정적으로 고른 뒤 그 행만 읽는다
    pick = pl.struct(keys).hash(SEED) % 1_000_000 < int(pos_frac * 1_000_000)
    pos = panel.filter(is_positive(), pick if pos_frac < 1.0 else pl.lit(True)).collect()
    pos = pos.with_columns(row_kind=pl.lit("positive"), sample_weight=pl.lit(1.0 / pos_frac))

    # 0행: 이력 칸 × 정상 월에서 무작위 (positive 전체를 피함). 후보가 수억 개라 무작위 추출 후 제외·중복 제거.
    grid = pl.read_parquet(GRID_PATH).filter("in_history")
    months = panel_months().filter(pl.col("mode") == "train", ~pl.col("excluded"))
    n_zero = int(round(pos.height * zero_ratio))
    n_cand = grid.height * months.height - n_pos_all
    picks = []
    need = n_zero
    while need > 0:
        k = int(need * 1.3) + 1000
        ci = rng.integers(0, grid.height, k)
        mi = rng.integers(0, months.height, k)
        cand = pl.DataFrame({"cell_id": grid["cell_id"].gather(ci), "year": months["year"].gather(mi),
                             "month": months["month"].gather(mi)})
        cand = cand.join(pos_keys, on=keys, how="anti")
        picks.append(cand)
        allp = pl.concat(picks).unique(keys, maintain_order=True)
        need = n_zero - allp.height
        picks = [allp]
    zkeys = picks[0].head(n_zero)
    # 0행 특성: 주변 합(희소 패널에 있을 수 있음) 외에는 0
    nbz = panel.join(zkeys.lazy(), on=keys, how="semi").select(
        keys + ["aqua_n_det_nb", "terra_n_det_nb", "aqua_n_det_nb_raw", "terra_n_det_nb_raw"]).collect()
    zero = zkeys.join(nbz, on=keys, how="left")
    mt = months.drop("mode", "excluded", "reason")
    sm = load_panel("train").select("year", "month", "aqua_obs_days", "terra_obs_days").unique().collect()
    zero = with_attrs(zero.join(mt, on=["year", "month"]).join(sm, on=["year", "month"], how="left"), cell_attrs())
    fill = [c for c in RAW_COLS + [f"{c}_raw" for c in RAW_COLS] if c in pos.columns]
    zero = zero.with_columns(
        *[pl.col(c).fill_null(0) if c in zero.columns else pl.lit(0.0).alias(c) for c in fill],
        aqua_frp_max=pl.lit(0.0), terra_frp_max=pl.lit(0.0),
        aqua_night_ratio=pl.lit(None, dtype=pl.Float64), terra_night_ratio=pl.lit(None, dtype=pl.Float64),
        month_sin=(2 * math.pi * pl.col("month").cast(pl.Float64) / 12).sin(),
        month_cos=(2 * math.pi * pl.col("month").cast(pl.Float64) / 12).cos(),
        terra_drift=(pl.col("year") >= 2021).cast(pl.Int8),
        y=pl.lit(0.0), y_raw=pl.lit(0, dtype=pl.Int32),
        row_kind=pl.lit("zero_sample"),
        # 0행 가중치: 전체 0 후보 수 / 표본 수 (합계 보정용; 학습에는 기본으로 쓰지 않음)
        sample_weight=pl.lit(n_cand / max(n_zero, 1)),
    )
    df = pl.concat([pos, zero.select(pos.columns)], how="vertical_relaxed").sort("year", "month", "cell_id")
    leaked = [c for c in MODIS_FEATURES if "snpp" in c]
    assert not leaked, f"특성에 S-NPP 정보: {leaked}"
    assert df.height == df.select(keys).unique().height, "칸·월 중복"
    df.write_parquet(TRAIN_PATH, compression="zstd")
    print(f"positive 전체 {n_pos_all:,} → 표본 {pos.height:,} (pos_frac={pos_frac}), "
          f"0행 후보 약 {n_cand:,} → 표본 {zero.height:,} (zero_ratio={zero_ratio}), seed={SEED}")
    return df


# ── 리포트 ────────────────────────────────────────────────────────────────

def report(df: pl.DataFrame, months: pl.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    months.filter(pl.col("excluded") | (pl.col("common_days") / pl.col("days_in_month") < 0.7)).with_columns(
        common_frac=pl.col("common_days") / pl.col("days_in_month"),
        decision=pl.when("excluded").then(pl.lit("제외")).otherwise(pl.lit("포함(공통일 50~70%)")),
    ).select("mode", "year", "month", "decision", "common_days", "days_in_month", "common_frac", "reason").write_csv(
        REPORTS_DIR / "train_excluded_months.csv")

    def agg(frame: pl.DataFrame, region: str) -> pl.DataFrame:
        return frame.group_by("year").agg(
            rows=pl.len(), positive=(pl.col("row_kind") == "positive").sum(),
            y_zero=(pl.col("y") == 0).sum(), y_sum=pl.col("y").sum(),
            modis_sum=(pl.col("aqua_n_det") + pl.col("terra_n_det")).sum(),
        ).with_columns(region=pl.lit(region))

    parts = [agg(df, "world"), agg(df.filter("is_korea"), "korea"), agg(df.filter(pl.col("adm0_a3") == "ARG"), "argentina")]
    parts += [agg(df.filter(pl.col("continent") == c), c) for c in df["continent"].unique().sort().to_list()]
    s = pl.concat(parts).with_columns(y_zero_ratio=pl.col("y_zero") / pl.col("rows"),
                                      snpp_to_modis=pl.col("y_sum") / pl.col("modis_sum")).sort("region", "year")
    s.write_csv(REPORTS_DIR / "train_table_summary.csv")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="공통 정상일 기준 희소 패널·학습 표 생성")
    ap.add_argument("--skip-panel", action="store_true", help="패널은 그대로 두고 학습 표·리포트만")
    ap.add_argument("--pos-frac", type=float, default=1.0, help="positive 행 표본 비율 (기본 1.0 = 전부)")
    ap.add_argument("--zero-ratio", type=float, default=1.0, help="0행 수 = positive 표본 수 × 이 값 (기본 1.0)")
    args = ap.parse_args(argv)
    start_run_log("train_table")
    t0 = time.perf_counter()
    if not args.skip_panel:
        grid = build_grid()
        print(f"[격자] {GRID_PATH.name}: {grid.height:,}칸 (이력 {grid['in_history'].sum():,}, 한국 {grid['is_korea'].sum():,}), 범위 {ANALYSIS_BBOX}")
        attrs = cell_attrs()
        months = pl.concat([build_panel("train", attrs), build_panel("apply", attrs)])
        months.write_parquet(PANEL_MONTHS_PATH)
    months = panel_months()
    n_panel = {m: load_panel(m).select(pl.len()).collect().item() for m in ("train", "apply")}
    print(f"[패널] train {n_panel['train']:,}행, apply {n_panel['apply']:,}행 ({time.perf_counter() - t0:,.0f}초)")
    df = build_train_table(args.pos_frac, args.zero_ratio)
    report(df, months)
    mem = df.estimated_size("mb")
    print(f"[완료] {TRAIN_PATH.name}: {df.height:,}행, 메모리 약 {mem:,.0f} MB (컬럼 {df.width}개), "
          f"특성 {len(MODIS_FEATURES)}개 ({time.perf_counter() - t0:,.0f}초)")
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, thousands_separator=","):
        print(months.filter("excluded").select("mode", "year", "month", "common_days", "days_in_month"))
        print(df.group_by("continent").agg(rows=pl.len(), positive=(pl.col("row_kind") == "positive").sum(),
                                           y_zero_ratio=(pl.col("y") == 0).mean()).sort("continent"))
        for name, sub in (("한국", df.filter("is_korea")), ("아르헨티나", df.filter(pl.col("adm0_a3") == "ARG"))):
            print(f"  {name}: {sub.height:,}행, positive {(sub['row_kind'] == 'positive').sum():,}, y=0 비율 {(sub['y'] == 0).mean():.1%}")


if __name__ == "__main__":
    main()
