"""m2v_xgb (v1) leave-continent-out 검증 — 처음 보는 대륙에서도 쓸 수 있는지.

방법
  대륙 하나(C)를 통째로 빼고 나머지 대륙으로 학습 → C의 2012-02 ~ 2025-12 전체 예측.
  - m2v_xgb: 채택 모델과 같은 특성·설정(tweedie 1.1, 범주형 국가·하위지역·대륙, sample weight, zero rule 끔).
    조기 종료는 학습 대륙의 VAL_YEAR(2024)로 (빠진 대륙 C의 자료는 학습·조기 종료 어디에도 쓰지 않음).
    C의 국가·하위지역·대륙 범주는 학습에 없으므로 모델은 이 범주를 '처음 보는 값'으로 처리한다.
  - m2v_q10_q90: 같은 방식의 분위수 모델 (칸 구간 적중률용).
  - ratio_country: C를 뺀 학습 대륙으로 국가 × 월 비율 → C의 국가·하위지역은 없으므로 위도띠 × 월 → 월 순으로 대체.
  평가 대륙: 학습 표 행이 10만 개 이상인 6개 대륙 (Antarctica·Seven seas·none은 너무 작아 항상 학습에만 포함).

지표 (leave-year-out 검증과 같은 정의)
  - 지역·월 합계: 전체 칸(희소 패널 + 패널 밖 0행) 합. MAE, bias = Σ예측/Σ실제, Pearson. 전 기간(pooled).
  - 칸·월: 학습 표 표본 행의 MAE, 칸 q10~q90 구간 적중률.
  - 월 합계 구간 적중률: 배포에 쓰는 AOI 보정 구간표(models/m2v_aoi_interval.parquet, continent/country 유형)를
    leave-continent-out 월 합계에 그대로 씌웠을 때 실제 값이 구간 안에 드는 비율.
  - 한국: 아시아를 뺀 모델, 아르헨티나: 남미를 뺀 모델.
  - 비교 기준(LYO): 기존 leave-year-out OOF(data/processed/oof_region_month.parquet)의 같은 지역·같은 달.

기존 모델·데이터는 읽기만 한다. 출력: reports/m2v_leave_continent_out.csv, reports/figures/m2v_leave_continent_out.png

실행  uv run python -m planetarium.modeling.m2v_lco
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium.config import FIGURES_DIR, REPORTS_DIR
from planetarium.modeling.evaluate import cell_metrics, total_metrics
from planetarium.modeling.m1_xgb import (
    M1_FEATURES, OBJECTIVES, b1c_fit, b1c_predict, category_maps, dense_zero_rows, xgb_predict, xgb_train,
)
from planetarium.modeling.m2v_interval import AOI_INTERVAL_PATH, apply_interval
from planetarium.modeling.train_m1 import FOLD_YEARS, OOF_PATH, lat_band
from planetarium.runlog import peak_memory_gb, start_run_log
from planetarium.train_table import GRID_PATH, TRAIN_PATH, load_panel

OUT_CSV = REPORTS_DIR / "m2v_leave_continent_out.csv"
OUT_FIG = FIGURES_DIR / "m2v_leave_continent_out.png"
OBJECTIVE = "tweedie_1.1"
VAL_YEAR = 2024
MIN_ROWS = 100_000
EXTRA = {"korea": ("Asia", pl.col("is_korea")), "argentina": ("South America", pl.col("adm0_a3") == "ARG")}
INTERVAL_LEVEL = {"continent": "continent", "korea": "country", "argentina": "country"}


def country_sums() -> pl.DataFrame:
    """ratio_country 학습용 합계 (대륙 열 포함)."""
    return (
        load_panel("train").select("year", "month", "cell_lat", "adm0_a3", "subregion", "continent", "aqua_n_det", "terra_n_det", "y")
        .with_columns(lat_band=lat_band(), adm0_a3=pl.col("adm0_a3").fill_null("none"),
                      subregion=pl.col("subregion").fill_null("none"), continent=pl.col("continent").fill_null("none"))
        .group_by("year", "continent", "adm0_a3", "subregion", "lat_band", "month")
        .agg(y=pl.col("y").sum(), m=(pl.col("aqua_n_det") + pl.col("terra_n_det")).sum(), t=pl.col("terra_n_det").sum())
        .collect()
    )


def region_rows(df: pl.DataFrame, pred: np.ndarray, cont: str) -> pl.DataFrame:
    """빠진 대륙 cont의 월 합계 + (해당되면) 한국·아르헨티나 월 합계."""
    d = df.select("year", "month", "is_korea", "adm0_a3", "y").with_columns(pred=pl.Series(pred, dtype=pl.Float64))
    parts = [d.group_by("year", "month").agg(true=pl.col("y").sum(), pred=pl.col("pred").sum()).with_columns(region=pl.lit(f"continent:{cont}"))]
    for name, (c, m) in EXTRA.items():
        if c == cont:
            parts.append(d.filter(m).group_by("year", "month").agg(true=pl.col("y").sum(), pred=pl.col("pred").sum())
                         .with_columns(region=pl.lit(name)))
    return pl.concat(parts).with_columns(pl.col("true").cast(pl.Float64))


def interval_hit(mt: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """배포용 AOI 구간표를 월 합계 예측에 씌운 적중률."""
    lv = pl.when(pl.col("region").str.starts_with("continent:")).then(pl.lit("continent")).otherwise(
        pl.col("region").replace_strict(INTERVAL_LEVEL, default="country"))
    d = mt.filter(pl.col("pred") > 0).with_columns(level=lv, value=pl.col("pred"))
    d = apply_interval(d, table, "value").with_columns(hit=(pl.col("true") >= pl.col("value_lo")) & (pl.col("true") <= pl.col("value_hi")))
    return d.group_by("model", "scheme", "region").agg(month_interval_coverage=pl.col("hit").mean())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="m2v_xgb leave-continent-out 검증")
    ap.add_argument("--continents", nargs="*", help="일부 대륙만 (시험용)")
    args = ap.parse_args(argv)
    start_run_log("m2v_lco")
    t_all = time.perf_counter()

    tt = pl.read_parquet(TRAIN_PATH).with_columns(w=pl.col("sample_weight") / pl.col("sample_weight").mean(),
                                                  continent=pl.col("continent").fill_null("none"))
    cats = category_maps()
    sizes = tt.group_by("continent").len()
    conts = sorted(c for c, n in sizes.iter_rows() if n >= MIN_ROWS and c != "none")
    if args.continents:
        conts = [c for c in conts if c in args.continents]
    print(f"[대륙] 평가 {conts} / 항상 학습에만: {sorted(set(sizes['continent']) - set(conts))}")
    obj = OBJECTIVES[OBJECTIVE]
    sums = country_sums()

    # 1) 대륙마다 학습 (모델은 메모리에만 두고 저장하지 않는다)
    models, cell_rows = {}, []
    bar = tqdm(conts, desc="leave-continent-out 학습", unit="continent", dynamic_ncols=True)
    for c in bar:
        bar.set_postfix_str(c)
        t0 = time.perf_counter()
        rest = tt.filter(pl.col("continent") != c)
        tr, va = rest.filter(pl.col("year") != VAL_YEAR), rest.filter(pl.col("year") == VAL_YEAR)
        m = xgb_train(tr, va, M1_FEATURES, cats, obj)
        tqdm.write(f"  [{c}] m2v_xgb iter {m.best_iteration} ({time.perf_counter() - t0:.0f}초)")
        q = xgb_train(tr, va, M1_FEATURES, cats, {}, quantile=True)
        rc = b1c_fit(sums.filter(pl.col("continent") != c), FOLD_YEARS)
        models[c] = (m, q, rc)
        # 칸·월 지표 (학습 표 표본 행)
        te = tt.filter(pl.col("continent") == c)
        yt = te["y"].to_numpy()
        p = xgb_predict(m, te, M1_FEATURES, cats)
        qq = xgb_predict(q, te, M1_FEATURES, cats)
        lo, hi = np.minimum(qq[:, 0], qq[:, 1]), np.maximum(qq[:, 0], qq[:, 1])
        pr = b1c_predict(te, rc)
        groups = {f"continent:{c}": np.ones(te.height, dtype=bool)}
        for name, (cc, mk) in EXTRA.items():
            if cc == c:
                groups[name] = te.select(mk.fill_null(False))[:, 0].to_numpy()
        for region, idx in groups.items():
            if idx.sum() == 0:
                continue
            for model, pp in (("m2v_xgb", p), ("ratio_country", pr)):
                cm = cell_metrics(yt[idx], pp[idx])
                row = {"model": model, "scheme": "LCO", "region": region, "cell_MAE": cm["cell_MAE"], "cell_n": cm["cell_n"]}
                if model == "m2v_xgb":
                    row["cell_interval_coverage_q10_q90"] = float(np.mean((yt[idx] >= lo[idx]) & (yt[idx] <= hi[idx])))
                cell_rows.append(row)
        tqdm.write(f"  [{c}] 학습 {tr.height:,}행, 시험 {te.height:,}행, m2v iter {m.best_iteration}, q iter {q.best_iteration}, "
                   f"{time.perf_counter() - t0:.0f}초, 최대 메모리 {peak_memory_gb() or 0:.1f} GB")

    # 2) 연도마다 전체 칸(패널 + 패널 밖 0행)을 빠진 대륙의 모델로 예측해 월 합계
    grid = pl.read_parquet(GRID_PATH).with_columns(
        pl.col("adm0_a3").fill_null("none"), pl.col("subregion").fill_null("none"), pl.col("continent").fill_null("none"))
    totals = []
    for y in tqdm(FOLD_YEARS, desc="연도별 월 합계", unit="year", dynamic_ncols=True):
        panel = load_panel("train", [y]).collect().with_columns(pl.col("adm0_a3").fill_null("none"),
                                                                pl.col("continent").fill_null("none"))
        dz = dense_zero_rows(y, grid, panel.select("cell_id", "year", "month")).with_columns(y=pl.lit(0.0))
        for c in conts:
            m, _, rc = models[c]
            pc, zc = panel.filter(pl.col("continent") == c), dz.filter(pl.col("continent") == c)
            for model, pp, pz in (("m2v_xgb", xgb_predict(m, pc, M1_FEATURES, cats), xgb_predict(m, zc, M1_FEATURES, cats)),
                                  ("ratio_country", b1c_predict(pc, rc), np.zeros(zc.height))):
                r = pl.concat([region_rows(pc, pp, c), region_rows(zc, pz, c)])
                totals.append(r.group_by("year", "month", "region").agg(pl.col("true", "pred").sum()).with_columns(
                    model=pl.lit(model), scheme=pl.lit("LCO")))
        del panel, dz
    lco = pl.concat(totals)

    # 3) 비교 기준: 기존 leave-year-out OOF의 같은 지역·같은 달
    regions = lco["region"].unique().to_list()
    lyo = (pl.read_parquet(OOF_PATH).filter(pl.col("model").is_in(["m2v_xgb", "ratio_country"]), pl.col("region").is_in(regions))
           .join(lco.select("year", "month", "region").unique(), on=["year", "month", "region"], how="semi")
           .with_columns(scheme=pl.lit("LYO")).select(lco.columns))
    mt = pl.concat([lco, lyo])

    rows = []
    for (model, scheme, region), g in mt.group_by("model", "scheme", "region"):
        g = g.sort("year", "month")
        tm = total_metrics(g["true"].to_numpy(), g["pred"].to_numpy(), "month_total_")
        rows.append({"model": model, "scheme": scheme, "region": region,
                     "month_total_MAE": tm["month_total_MAE"], "month_total_bias": tm["month_total_bias"],
                     "month_total_pearson": tm["month_total_pearson"], "n_months": tm["month_total_n_months"],
                     "mean_month_true": float(g["true"].mean()),
                     "month_total_rel_MAE": tm["month_total_MAE"] / float(g["true"].mean()) if g["true"].mean() else None})
    out = pl.DataFrame(rows)
    out = out.join(interval_hit(mt.filter(pl.col("model") == "m2v_xgb"), pl.read_parquet(AOI_INTERVAL_PATH)),
                   on=["model", "scheme", "region"], how="left")
    # LYO 칸 지표 (기존 검증 요약의 fold 평균)
    summ = pl.read_csv(REPORTS_DIR / "validation_summary.csv")
    lyo_cell = (summ.filter(pl.col("model").is_in(["m2v_xgb", "ratio_country"]), pl.col("region").is_in(regions),
                            pl.col("metric").is_in(["cell_MAE", "cell_interval_coverage_q10_q90"]))
                .pivot(on="metric", index=["model", "region"], values="mean").with_columns(scheme=pl.lit("LYO")))
    cells = pl.concat([pl.DataFrame(cell_rows), lyo_cell], how="diagonal_relaxed").select(
        "model", "scheme", "region", "cell_MAE", "cell_interval_coverage_q10_q90")
    out = out.join(cells, on=["model", "scheme", "region"], how="left").sort("region", "model", "scheme")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(OUT_CSV)
    plot(out, lco)

    cfg = dict(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=260, tbl_formatting="ASCII_MARKDOWN",
               tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, float_precision=3)
    with pl.Config(**cfg):
        print(out.select("region", "model", "scheme", "month_total_MAE", "month_total_rel_MAE", "month_total_bias",
                         "month_total_pearson", "month_interval_coverage", "cell_MAE", "cell_interval_coverage_q10_q90", "n_months"))
    print(f"[완료] {OUT_CSV.name}, {OUT_FIG.name} — {time.perf_counter() - t_all:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.1f} GB")


def plot(out: pl.DataFrame, lco: pl.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(r for r in out["region"].unique().to_list() if r.startswith("continent:")) + ["korea", "argentina"]
    order = [r for r in order if r in out["region"].to_list()]
    labels = [r.replace("continent:", "").replace("North America", "N. America").replace("South America", "S. America")
              .replace("korea", "Korea\n(no Asia)").replace("argentina", "Argentina\n(no S. Am.)") for r in order]
    series = [("m2v_xgb", "LYO", "#9ecae1"), ("m2v_xgb", "LCO", "#08519c"),
              ("ratio_country", "LYO", "#fdd0a2"), ("ratio_country", "LCO", "#d94801")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    x = np.arange(len(order))
    wdt = 0.2
    for k, (metric, title) in enumerate((("month_total_bias", "Bias (Σpred / Σtrue), 1 = unbiased"),
                                          ("month_total_rel_MAE", "Monthly-total MAE / mean monthly total"))):
        ax = axes[k]
        for i, (model, scheme, color) in enumerate(series):
            v = [out.filter(pl.col("region") == r, pl.col("model") == model, pl.col("scheme") == scheme)[metric].to_list() for r in order]
            ax.bar(x + (i - 1.5) * wdt, [a[0] if a and a[0] is not None else np.nan for a in v], wdt, color=color,
                   label=f"{model} {scheme}")
        if metric == "month_total_bias":
            ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    ax = axes[2]
    k = lco.filter(pl.col("region") == "korea").sort("year", "month")
    if k.height:
        t = [f"{a}-{b:02d}" for a, b in k.select("year", "month").iter_rows()]
        xi = np.arange(len(t))
        tr = k.filter(pl.col("model") == "m2v_xgb")
        ax.plot(xi[:tr.height], tr["true"], color="k", lw=1.2, label="S-NPP observed")
        ax.plot(xi[:tr.height], tr["pred"], color="#08519c", lw=1, label="m2v_xgb (trained without Asia)")
        rc = k.filter(pl.col("model") == "ratio_country")
        ax.plot(xi[:rc.height], rc["pred"], color="#d94801", lw=1, alpha=0.8, label="ratio_country (without Asia)")
        step = max(1, len(t) // 12)
        ax.set_xticks(xi[::step][: len(xi[::step])], [t[i] for i in range(0, tr.height, step)], rotation=45, fontsize=7)
        ax.set_title("Korea monthly totals, leave-Asia-out (2012-02 ~ 2025-12)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("m2v_xgb v1: leave-continent-out (LCO) vs leave-year-out (LYO)", fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
