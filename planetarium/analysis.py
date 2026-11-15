"""T7: 같은 달 과거 대비 비교 ('이번 달 감지 수가 예년 같은 달보다 많은가?').

compare_month(aoi_id, year, month) -> dict
  기준 분포: 같은 달 값 (대상 연도 제외, status == ok인 달만). 기준 기간은 지역마다 연속성 점검으로 정한다 (2026-11-15):
    continuity_ratio = (통합값 2013~2021 / 2003~2011) ÷ (MODIS 원자료 2013~2021 / 2003~2011)
    0.85~1.15 → 전체 기간 2001~2025, 밖이면 2012~2025 (VIIRS 시기만; 2012 이전 통합값이 원자료 추세와 어긋나는 지역).
    두 기간 MODIS 원자료가 각각 50건 미만이면 상위 지역(parent_aoi)의 비율을 물려받는다 (continuity_source = parent:<id>).
    2012~ 기준 지역에서 2012 이전 달의 판정은 uncertain (그 값 자체가 어긋난 기간이라).
    기준 연수 < 8이면 uncertain.
  반환: value, baseline_median, baseline_p25, baseline_p75, percentile, ratio_to_median, n_baseline_years,
        verdict ∈ {above, similar, below, uncertain}, status, low_sample
  verdict 규칙
    - n_baseline_years < 10, 또는 대상 월 status가 missing/partial이면 uncertain
    - low_sample(기준 분포 75백분위 < 1: 예년 같은 달 감지가 거의 없는 지역): 값이 3 이상이면 above, 아니면 uncertain
      (작은 지역에서 0과 1의 차이로 백분위가 크게 흔들리는 것을 막기 위함)
    - percentile ≥ 80 → above, ≤ 20 → below, 그 외 similar
  percentile = (기준값 < 값 개수 + 0.5 × 같은 값 개수) / 기준 연도 수 × 100
  감지 수는 산불 건수가 아니라 관측 활동이다.

출력  data/processed/aoi_month_compare.parquet  (모든 AOI × 월, 백엔드가 그대로 읽음)

실행  uv run python -m planetarium.analysis
"""

from __future__ import annotations

import argparse
import time

import polars as pl

from planetarium.config import PROCESSED_DATA_DIR
from planetarium.harmonize import AOI_MONTH_PATH
from planetarium.runlog import start_run_log

COMPARE_PATH = PROCESSED_DATA_DIR / "aoi_month_compare.parquet"
BASE_YEARS = (2001, 2025)
VIIRS_START = 2012
MIN_YEARS = 8
CONT_A, CONT_B = (2003, 2011), (2013, 2021)
CONT_OK = (0.85, 1.15)
CONT_MIN_RAW = 50
ABOVE, BELOW = 80.0, 20.0
LOW_SAMPLE_P75 = 1.0
LOW_SAMPLE_ABOVE_VALUE = 3.0


def _compare(target: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """target(aoi_id, year, month, value, status) × base(aoi_id, year, month, value) -> 비교표."""
    b = base.select("aoi_id", "month", pl.col("year").alias("b_year"), pl.col("value").alias("b"))
    pairs = target.join(b, on=["aoi_id", "month"], how="left").filter(
        pl.col("b_year").is_null() | (pl.col("b_year") != pl.col("year")))
    stats = pairs.group_by("aoi_id", "year", "month").agg(
        value=pl.col("value").first(), status=pl.col("status").first(),
        baseline_median=pl.col("b").median(), baseline_p25=pl.col("b").quantile(0.25, "linear"),
        baseline_p75=pl.col("b").quantile(0.75, "linear"),
        n_baseline_years=pl.col("b").count().cast(pl.Int16),
        n_less=(pl.col("b") < pl.col("value")).sum(), n_equal=(pl.col("b") == pl.col("value")).sum(),
    )
    low = pl.col("baseline_p75") < LOW_SAMPLE_P75
    pct = (pl.col("n_less") + 0.5 * pl.col("n_equal")) / pl.col("n_baseline_years") * 100
    return stats.with_columns(
        percentile=pl.when(pl.col("n_baseline_years") > 0).then(pct),
        ratio_to_median=pl.when(pl.col("baseline_median") > 0).then(pl.col("value") / pl.col("baseline_median")),
        low_sample=low.fill_null(False),
    ).with_columns(
        verdict=pl.when((pl.col("n_baseline_years") < MIN_YEARS) | pl.col("status").is_in(["missing", "partial"])
                        | pl.col("value").is_null()).then(pl.lit("uncertain"))
        .when(pl.col("low_sample")).then(
            pl.when(pl.col("value") >= LOW_SAMPLE_ABOVE_VALUE).then(pl.lit("above")).otherwise(pl.lit("uncertain")))
        .when(pl.col("percentile") >= ABOVE).then(pl.lit("above"))
        .when(pl.col("percentile") <= BELOW).then(pl.lit("below"))
        .otherwise(pl.lit("similar")),
    ).select("aoi_id", "year", "month", "value", "baseline_median", "baseline_p25", "baseline_p75", "percentile",
             "ratio_to_median", "n_baseline_years", "verdict", "status", "low_sample")


def continuity() -> pl.DataFrame:
    """지역별 연속성 점검 → aoi_id, continuity_ratio, continuity_source, base_start(2001 | 2012), baseline_period."""
    from planetarium.harmonize import AOI_LIST_PATH
    am = pl.scan_parquet(AOI_MONTH_PATH).filter(pl.col("status") == "ok")

    def per(ys, sfx):
        return am.filter(pl.col("year").is_between(*ys)).group_by("aoi_id").agg(
            pl.col("value").sum().alias(f"v{sfx}"), pl.col("raw_modis").sum().alias(f"m{sfx}"))
    c = per(CONT_A, "a").join(per(CONT_B, "b"), on="aoi_id").collect()
    c = c.with_columns(own=pl.when((pl.col("ma") >= CONT_MIN_RAW) & (pl.col("mb") >= CONT_MIN_RAW) & (pl.col("va") > 0))
                       .then((pl.col("vb") / pl.col("va")) / (pl.col("mb") / pl.col("ma"))))
    own = dict(zip(c["aoi_id"].to_list(), c["own"].to_list()))
    lst = pl.read_parquet(AOI_LIST_PATH).select("aoi_id", "parent_aoi")
    parent = dict(zip(lst["aoi_id"].to_list(), lst["parent_aoi"].to_list()))
    rows = []
    for a in lst["aoi_id"].to_list():
        r, src, cur, seen = own.get(a), "own", a, set()
        while r is None and parent.get(cur) and parent[cur] not in seen:   # 표본이 적으면 상위 지역 비율
            seen.add(cur)
            cur = parent[cur]
            r, src = own.get(cur), f"parent:{cur}"
        if r is None:
            src = "none"
        ok = r is None or (CONT_OK[0] <= r <= CONT_OK[1])
        start = BASE_YEARS[0] if ok else VIIRS_START
        rows.append({"aoi_id": a, "continuity_ratio": r, "continuity_source": src, "base_start": start,
                     "baseline_period": f"{start}-{BASE_YEARS[1]}"})
    return pl.DataFrame(rows, schema_overrides={"continuity_ratio": pl.Float64, "base_start": pl.Int16})


def _base(am: pl.LazyFrame, cont: pl.DataFrame | None = None) -> pl.LazyFrame:
    b = am.filter(pl.col("year").is_between(*BASE_YEARS), pl.col("status") == "ok", pl.col("value").is_not_null())
    if cont is None:
        return b
    return b.join(cont.lazy().select("aoi_id", "base_start"), on="aoi_id", how="left").filter(
        pl.col("year") >= pl.col("base_start").fill_null(BASE_YEARS[0])).drop("base_start")


def compare_month(aoi_id: str, year: int, month: int) -> dict:
    am = pl.scan_parquet(AOI_MONTH_PATH).filter(pl.col("aoi_id") == aoi_id)
    target = am.filter(pl.col("year") == year, pl.col("month") == month).select("aoi_id", "year", "month", "value", "status")
    cont = continuity().filter(pl.col("aoi_id") == aoi_id)
    res = _apply_continuity(_compare(target, _base(am, cont)).collect(), cont)
    if res.height == 0:
        raise KeyError(f"{aoi_id} {year}-{month:02d} 자료가 없습니다")
    out = res.to_dicts()[0]
    from planetarium.harmonize import AOI_LIST_PATH
    if aoi_id in set(pl.read_parquet(AOI_LIST_PATH).filter(pl.col("hidden").fill_null(False))["aoi_id"].to_list()):
        out["verdict"] = "uncertain"  # 숨김 AOI(Seven seas 등)
    return out


def _apply_continuity(df: pl.DataFrame, cont: pl.DataFrame) -> pl.DataFrame:
    """baseline_period·continuity_ratio·continuity_source 열을 붙이고, 2012~ 기준 지역의 2012 이전 달은 uncertain."""
    df = df.join(cont.select("aoi_id", "baseline_period", "continuity_ratio", "continuity_source", "base_start"), on="aoi_id", how="left")
    pre = (pl.col("base_start") == VIIRS_START) & (pl.col("year") < VIIRS_START)
    return df.with_columns(verdict=pl.when(pre.fill_null(False)).then(pl.lit("uncertain"))
                           .otherwise(pl.col("verdict"))).drop("base_start")


def build_compare(path=None, legacy: bool = False) -> pl.DataFrame:
    """legacy=True: 이전 규칙(전 지역 2001~2025, 기준 연수 10) — 비교 검증용."""
    global MIN_YEARS
    am = pl.scan_parquet(AOI_MONTH_PATH)
    if legacy:
        old, MIN_YEARS = MIN_YEARS, 10
        df = _compare(am.select("aoi_id", "year", "month", "value", "status"), _base(am)).sort("aoi_id", "year", "month").collect()
        MIN_YEARS = old
    else:
        cont = continuity()
        df = _apply_continuity(_compare(am.select("aoi_id", "year", "month", "value", "status"), _base(am, cont))
                               .sort("aoi_id", "year", "month").collect(), cont)
    # 진행 중인 달 응답용: 마지막 관측일과 달 전체 추정 (판정은 partial이라 uncertain)
    extra = [c for c in ("as_of_date", "value_projected") if c in am.collect_schema().names()]
    if extra:
        df = df.join(am.select("aoi_id", "year", "month", *extra).collect(), on=["aoi_id", "year", "month"], how="left")
    # aoi_list.hidden(Seven seas 등)은 판정하지 않음
    from planetarium.harmonize import AOI_LIST_PATH
    hidden = pl.read_parquet(AOI_LIST_PATH).filter(pl.col("hidden").fill_null(False))["aoi_id"]
    df = df.with_columns(verdict=pl.when(pl.col("aoi_id").is_in(hidden.implode())).then(pl.lit("uncertain"))
                         .otherwise(pl.col("verdict")))
    df.write_parquet(path or COMPARE_PATH, compression="zstd")
    return df


def validate_continuity() -> pl.DataFrame:
    """이전 규칙 vs 연속성 기준: 바뀐 판정 개수·방향, 최근 3년(2023~2025, 완결 달) 판정 분포.
    → reports/compare_continuity_validation.csv, reports/compare_continuity_ratios.csv (파일을 바꾸지 않고 메모리에서 비교)"""
    from planetarium.config import INTERIM_DATA_DIR, REPORTS_DIR
    from planetarium.harmonize import AOI_LIST_PATH
    tmp = INTERIM_DATA_DIR / "compare_check"
    tmp.mkdir(parents=True, exist_ok=True)
    old = build_compare(tmp / "old.parquet", legacy=True).select("aoi_id", "year", "month", "status", v_old="verdict")
    new = build_compare(tmp / "new.parquet").select("aoi_id", "year", "month", "baseline_period", "continuity_ratio",
                                                   "continuity_source", v_new="verdict")
    cont = continuity()
    cont.join(pl.read_parquet(AOI_LIST_PATH).select("aoi_id", "name", "level"), on="aoi_id").write_csv(REPORTS_DIR / "compare_continuity_ratios.csv")
    d = old.join(new, on=["aoi_id", "year", "month"]).join(pl.read_parquet(AOI_LIST_PATH).select("aoi_id", "level", "hidden"), on="aoi_id").filter(
        ~pl.col("hidden").fill_null(False))
    d = d.with_columns(fail=pl.col("baseline_period").str.starts_with(str(VIIRS_START)))
    groups = [("KR", pl.col("aoi_id") == "KR"), ("ADM0_ARG", pl.col("aoi_id") == "ADM0_ARG"), ("WORLD", pl.col("aoi_id") == "WORLD"),
              ("continuity pass (all)", ~pl.col("fail")), ("continuity fail (all)", pl.col("fail")),
              ("fail: Korea sgg", pl.col("fail") & (pl.col("level") == "sgg")), ("fail: countries", pl.col("fail") & (pl.col("level") == "country")),
              ("fail: adm1", pl.col("fail") & (pl.col("level") == "adm1"))]
    rows = []
    for name, f in groups:
        x = d.filter(f)
        ch = x.filter(pl.col("v_old") != pl.col("v_new"))
        rec = x.filter(pl.col("year").is_between(2023, 2025), pl.col("status") == "ok")
        row = {"group": name, "regions": x["aoi_id"].n_unique(), "region_months": x.height, "changed": ch.height,
               "changed_share": ch.height / max(x.height, 1)}
        for a_, b_ in (("above", "similar"), ("similar", "below"), ("above", "below"), ("similar", "above"), ("below", "similar"),
                       ("below", "above"), ("uncertain", "above"), ("uncertain", "similar"), ("uncertain", "below")):
            row[f"{a_}->{b_}"] = ch.filter(pl.col("v_old") == a_, pl.col("v_new") == b_).height
        row["to_uncertain"] = ch.filter(pl.col("v_new") == "uncertain").height
        for col, tag in (("v_old", "old"), ("v_new", "new")):
            dec = rec.filter(pl.col(col).is_in(["above", "similar", "below"]))
            n = max(dec.height, 1)
            ab, si, be = (dec.filter(pl.col(col) == v).height / n for v in ("above", "similar", "below"))
            row.update({f"recent3_{tag}_above": ab, f"recent3_{tag}_similar": si, f"recent3_{tag}_below": be,
                        f"recent3_{tag}_decided": dec.height, f"recent3_{tag}_imbalance": abs(ab - 0.2) + abs(be - 0.2)})
        rows.append(row)
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "compare_continuity_validation.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        print(cont.group_by(pl.col("baseline_period"), pl.col("continuity_source").str.starts_with("parent").alias("inherited")).len().sort("baseline_period"))
        print(cont.filter(pl.col("aoi_id").is_in(["KR", "ADM0_ARG", "WORLD", "SIDO_51", "SGG_47730", "ADM1_ARG-1299"])))
        print(m.select("group", "regions", "changed", "changed_share", "above->similar", "similar->below", "similar->above", "below->similar",
                       "to_uncertain", "recent3_old_above", "recent3_old_below", "recent3_new_above", "recent3_new_below",
                       "recent3_old_imbalance", "recent3_new_imbalance"))
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="같은 달 과거 대비 비교표")
    ap.add_argument("--validate", action="store_true", help="이전 규칙과 연속성 기준 비교만 (파일을 바꾸지 않음)")
    a = ap.parse_args(argv)
    start_run_log("analysis")
    if a.validate:
        validate_continuity()
        return
    t0 = time.perf_counter()
    df = build_compare()
    print(f"[완료] {COMPARE_PATH.name}: {df.height:,}행 ({time.perf_counter() - t0:,.0f}초)")
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=2, thousands_separator=","):
        print(df.group_by("verdict").len().sort("verdict"))
        print(df.filter(pl.col("aoi_id").is_in(["KR", "ADM0_ARG", "WORLD"]), pl.col("year") >= 2025)
              .select("aoi_id", "year", "month", "value", "baseline_median", "percentile", "ratio_to_median",
                      "n_baseline_years", "verdict", "status").sort("aoi_id", "year", "month"))
        print(compare_month("KR", 2025, 3))


if __name__ == "__main__":
    main()
