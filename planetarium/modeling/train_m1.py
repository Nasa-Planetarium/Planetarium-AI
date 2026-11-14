"""M1: MODIS → S-NPP 보정 모델 학습·검증 (leave-year-out).

모델
  B1 비율 기준선 : 학습 연도의 Σ S-NPP / Σ (Aqua+Terra)를 (위도 5° 띠 × 월)별로 계산해 곱한다.
                   분모가 0이면 상위 묶음(월만) 값. Terra 단독 변형(B1_terra)은 Σ S-NPP / Σ Terra.
  (B2 로그 선형, M1 LightGBM, 분위수 구간은 이후 단계에서 같은 틀에 추가)

검증
  leave-year-out: 2012~2025 각 연도를 통째로 빼고 나머지로 학습 → 그 해 예측.
  - 칸·월 지표: 학습 표(표본) 행 기준
  - 지역·월 합계 지표: 표본이 아니라 전체 칸(2012~2025 감지 이력 칸 ∪ 한국 칸) × 정상 월에 예측해 합계.
    희소 패널에 없는 칸·월은 MODIS·주변 감지가 모두 0이고 타깃도 0이며, B1 예측도 0이므로 희소 패널 합 = 전체 칸 합.
  지역: world, 대륙, korea, argentina

출력
  reports/validation_m1.csv        (model, region, fold_year, metric, value)  fold_year='pooled'는 전 기간 OOF 월 합계
  reports/validation_summary.csv   (model, region, metric, mean, std)
  reports/figures/m1_korea_monthly.png, m1_yearly_bias.png
  data/processed/oof_region_month.parquet  (model, region, year, month, true, pred)
  models/b1_ratios.parquet, models/b1_interval.parquet, models/m1_meta.json

실행  uv run python -m planetarium.modeling.train_m1 --models b1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, SEED
from planetarium.modeling.evaluate import cell_metrics, region_masks, total_metrics
from planetarium.runlog import start_run_log
from planetarium.train_table import TRAIN_END, TRAIN_PATH, TRAIN_START, load_panel

FOLD_YEARS = list(range(TRAIN_START[0], TRAIN_END[0] + 1))
LAT_BAND = 5
OOF_PATH = PROCESSED_DATA_DIR / "oof_region_month.parquet"
META_PATH = MODELS_DIR / "m1_meta.json"
B1_RATIOS_PATH = MODELS_DIR / "b1_ratios.parquet"
B1_INTERVAL_PATH = MODELS_DIR / "b1_interval.parquet"
INTERVAL_BINS = [0.0, 0.5, 1, 2, 5, 10, 20, 50, 100, 1e12]
PANEL_COLS = ["cell_id", "cell_lat", "year", "month", "adm0_a3", "continent", "is_korea",
              "aqua_n_det", "terra_n_det", "y"]


def lat_band() -> pl.Expr:
    return (pl.col("cell_lat") / LAT_BAND).floor().cast(pl.Int16) * LAT_BAND


# ── B1 ────────────────────────────────────────────────────────────────────

def b1_sums() -> pl.DataFrame:
    """연도 × 위도띠 × 월별 Σy, ΣAqua, ΣTerra (희소 패널 = 전체 칸 합)."""
    return (
        load_panel("train").select(PANEL_COLS).with_columns(lat_band=lat_band())
        .group_by("year", "lat_band", "month")
        .agg(y=pl.col("y").sum(), a=pl.col("aqua_n_det").sum(), t=pl.col("terra_n_det").sum())
        .collect()
    )


def b1_fit(sums: pl.DataFrame, years: list[int]) -> pl.DataFrame:
    """주어진 학습 연도로 비율표. kind ∈ {aqua_terra, terra_only}."""
    s = sums.filter(pl.col("year").is_in(years))
    out = []
    for kind, den in (("aqua_terra", pl.col("a") + pl.col("t")), ("terra_only", pl.col("t"))):
        band = s.group_by("lat_band", "month").agg(num=pl.col("y").sum(), den=den.sum())
        mon = s.group_by("month").agg(num_m=pl.col("y").sum(), den_m=den.sum())
        r = band.join(mon, on="month").with_columns(
            fallback=pl.col("den") <= 0,
            ratio=pl.when(pl.col("den") > 0).then(pl.col("num") / pl.col("den"))
            .otherwise(pl.col("num_m") / pl.col("den_m")),
        ).with_columns(kind=pl.lit(kind))
        out.append(r.select("kind", "lat_band", "month", "ratio", "num", "den", "fallback"))
        out.append(mon.with_columns(kind=pl.lit(kind), lat_band=pl.lit(None, dtype=pl.Int16),
                                    ratio=pl.col("num_m") / pl.col("den_m"), fallback=pl.lit(True))
                   .select("kind", "lat_band", "month", "ratio", pl.col("num_m").alias("num"),
                           pl.col("den_m").alias("den"), "fallback"))
    return pl.concat(out)


def b1_predict(df: pl.DataFrame, ratios: pl.DataFrame, terra_only: bool | pl.Expr = False) -> pl.Series:
    """예측값 Series. terra_only가 True(또는 행별 식)이면 Terra 단독 비율 × Terra."""
    band = ratios.filter(pl.col("lat_band").is_not_null())
    mon = ratios.filter(pl.col("lat_band").is_null()).select("kind", "month", ratio_m="ratio")
    d = df.with_columns(lat_band=lat_band(), _to=terra_only if isinstance(terra_only, pl.Expr) else pl.lit(terra_only))
    d = d.with_columns(kind=pl.when("_to").then(pl.lit("terra_only")).otherwise(pl.lit("aqua_terra")))
    d = d.join(band.select("kind", "lat_band", "month", "ratio"), on=["kind", "lat_band", "month"], how="left").join(
        mon, on=["kind", "month"], how="left")
    r = pl.coalesce("ratio", "ratio_m")
    modis = pl.when(pl.col("_to")).then(pl.col("terra_n_det")).otherwise(
        pl.col("aqua_n_det").fill_null(0) + pl.col("terra_n_det"))
    return d.select((r * modis).fill_null(0).alias("pred"))["pred"]


# ── 검증 루프 ─────────────────────────────────────────────────────────────

def region_month_totals(df: pl.DataFrame, pred: pl.Series, masks: dict[str, pl.Expr]) -> pl.DataFrame:
    d = df.select("year", "month", "is_korea", "adm0_a3", "continent", "y").with_columns(pred=pred)
    parts = []
    for name, m in masks.items():
        parts.append(d.filter(m).group_by("year", "month").agg(true=pl.col("y").sum(), pred=pl.col("pred").sum())
                     .with_columns(region=pl.lit(name)))
    return pl.concat(parts)


def run_b1(train_table: pl.DataFrame, masks: dict[str, pl.Expr]) -> tuple[list[dict], pl.DataFrame, pl.DataFrame]:
    sums = b1_sums()
    rows, oof, cell_oof = [], [], []
    for model, to in (("B1", False), ("B1_terra", True)):
        for y in tqdm(FOLD_YEARS, desc=model, unit="fold", leave=False):
            ratios = b1_fit(sums, [v for v in FOLD_YEARS if v != y])
            panel = load_panel("train", [y]).select(PANEL_COLS).collect()
            p = b1_predict(panel, ratios, to)
            rm = region_month_totals(panel, p, masks).with_columns(model=pl.lit(model))
            oof.append(rm)
            for region in masks:
                r = rm.filter(pl.col("region") == region).sort("month")
                for k, v in total_metrics(r["true"].to_numpy(), r["pred"].to_numpy(), "month_total_").items():
                    rows.append({"model": model, "region": region, "fold_year": str(y), "metric": k, "value": v})
            tt = train_table.filter(pl.col("year") == y)
            tp = b1_predict(tt, ratios, to)
            if model == "B1":
                cell_oof.append(tt.select("year", "month", "cell_id", "is_korea", "adm0_a3", "continent", "y").with_columns(pred=tp))
                # 구간용: 전체 희소 패널의 OOF (y, pred)
            for region, m in masks.items():
                sub = tt.with_columns(pred=tp).filter(m)
                if sub.height:
                    for k, v in cell_metrics(sub["y"].to_numpy(), sub["pred"].to_numpy()).items():
                        rows.append({"model": model, "region": region, "fold_year": str(y), "metric": k, "value": v})
    return rows, pl.concat(oof), pl.concat(cell_oof)


def b1_interval(sums: pl.DataFrame) -> pl.DataFrame:
    """B1 구간: OOF 칸·월에서 y / 예측 비율의 10·90 백분위수를 예측 크기 구간별로 (희소 패널 전체)."""
    parts = []
    for y in FOLD_YEARS:
        ratios = b1_fit(sums, [v for v in FOLD_YEARS if v != y])
        panel = load_panel("train", [y]).select(PANEL_COLS).collect()
        p = b1_predict(panel, ratios, False)
        parts.append(panel.select("y").with_columns(pred=p).filter(pl.col("pred") > 0))
    d = pl.concat(parts).with_columns(r=pl.col("y") / pl.col("pred"))
    # 표본이 1,000개 미만인 작은 구간은 다음 구간에 합친다
    edges = [e for e in INTERVAL_BINS[1:-1]]
    counts = [d.filter(pl.col("pred") < e).height for e in edges]
    keep = [e for e, c in zip(edges, counts) if c >= 1000]
    d = d.with_columns(bin=pl.col("pred").cut(keep, left_closed=True).cast(pl.String))
    iv = d.group_by("bin").agg(n=pl.len(), r10=pl.col("r").quantile(0.1), r50=pl.col("r").quantile(0.5),
                               r90=pl.col("r").quantile(0.9), pred_min=pl.col("pred").min(), pred_max=pl.col("pred").max())
    iv = iv.sort("pred_min")
    return iv.with_columns(pred_min=pl.when(pl.int_range(pl.len()) == 0).then(0.0).otherwise(pl.col("pred_min")))


def apply_interval(pred: pl.Series, iv: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
    """예측값에 구간 비율을 곱해 (lo, hi). 예측 0이면 0."""
    edges = iv["pred_min"].to_list()
    idx = np.clip(np.searchsorted(np.array(edges), pred.to_numpy(), side="right") - 1, 0, len(edges) - 1)
    r10, r90 = iv["r10"].to_numpy()[idx], iv["r90"].to_numpy()[idx]
    p = pred.to_numpy()
    return pl.Series(p * r10), pl.Series(p * r90)


# ── 그림 ──────────────────────────────────────────────────────────────────

def plot_korea_monthly(oof: pl.DataFrame, models: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = oof.filter(pl.col("region") == "korea").with_columns(
        t=pl.date(pl.col("year"), pl.col("month"), 15)).sort("t")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    base = k.filter(pl.col("model") == models[0])
    ax.plot(base["t"], base["true"], color="black", lw=1.6, label="S-NPP observed (common-day scaled)")
    colors = {"B1": "#1f77b4", "B2": "#2ca02c", "M1": "#d62728", "M1_zero_rule": "#d62728"}
    for m in models:
        s = k.filter(pl.col("model") == m)
        ax.plot(s["t"], s["pred"], lw=1.1, alpha=0.85, color=colors.get(m), label=f"{m} (leave-year-out)")
    ax.set_title("Korea monthly detections: S-NPP observed vs MODIS-harmonized (out-of-fold)")
    ax.set_ylabel("detections / month")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "m1_korea_monthly.png", dpi=130)
    plt.close(fig)


def plot_yearly_bias(oof: pl.DataFrame, models: list[str]) -> pl.DataFrame:
    """연도별 Σ예측/Σ실제 (MODIS 처리 버전 단절 확인: 2018 = 6.2→6.03, 2023 = 6.03→61.03)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yb = oof.group_by("model", "region", "year").agg(true=pl.col("true").sum(), pred=pl.col("pred").sum()).with_columns(
        bias=pl.col("pred") / pl.col("true")).sort("model", "region", "year")
    regions = ["world", "korea", "argentina"] + sorted(r for r in yb["region"].unique().to_list() if r.startswith("continent:"))
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4.8), squeeze=False)
    for ax, m in zip(axes[0], models):
        for r in regions:
            s = yb.filter(pl.col("model") == m, pl.col("region") == r)
            if s.height:
                lw = 2.2 if r in ("world", "korea", "argentina") else 0.9
                ax.plot(s["year"], s["bias"], marker="o", ms=3, lw=lw, label=r.replace("continent:", ""))
        for x, lab in ((2018, "MODIS 6.2→6.03"), (2023, "6.03→C6.1"), (2021, "Terra drift")):
            ax.axvline(x - 0.5, color="grey", ls="--", lw=0.8)
            ax.text(x - 0.45, ax.get_ylim()[1] if ax.get_ylim()[1] < 3 else 2.0, lab, fontsize=7, va="top", color="grey")
        ax.axhline(1.0, color="black", lw=0.8)
        ax.set_title(f"{m}: yearly bias = sum(pred) / sum(S-NPP), leave-year-out")
        ax.set_xlabel("year")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "m1_yearly_bias.png", dpi=130)
    plt.close(fig)
    yb.write_csv(REPORTS_DIR / "m1_yearly_bias.csv")
    return yb


# ── 요약·저장 ─────────────────────────────────────────────────────────────

def pooled_rows(oof: pl.DataFrame) -> list[dict]:
    rows = []
    for (model, region), g in oof.group_by("model", "region"):
        g = g.sort("year", "month")
        for k, v in total_metrics(g["true"].to_numpy(), g["pred"].to_numpy(), "month_total_").items():
            rows.append({"model": model, "region": region, "fold_year": "pooled", "metric": k, "value": v})
    return rows


def summarize(val: pl.DataFrame) -> pl.DataFrame:
    folds = val.filter(pl.col("fold_year") != "pooled").group_by("model", "region", "metric").agg(
        mean=pl.col("value").mean(), std=pl.col("value").std())
    pooled = val.filter(pl.col("fold_year") == "pooled").select(
        "model", "region", pl.concat_str(pl.lit("pooled_"), pl.col("metric")).alias("metric"),
        pl.col("value").alias("mean"), pl.lit(None, dtype=pl.Float64).alias("std"))
    return pl.concat([folds, pooled]).sort("model", "region", "metric")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="M1 보정 모델 leave-year-out 검증")
    ap.add_argument("--models", nargs="+", default=["b1"], choices=["b1"], help="이번 단계는 b1만")
    args = ap.parse_args(argv)
    start_run_log("train_m1")

    tt = pl.read_parquet(TRAIN_PATH)
    continents = sorted(c for c in tt["continent"].unique().to_list() if c != "none")
    masks = region_masks(tt, continents)
    rows, oof, _ = run_b1(tt, masks)
    val = pl.DataFrame(rows + pooled_rows(oof))
    val.write_csv(REPORTS_DIR / "validation_m1.csv")
    summ = summarize(val)
    summ.write_csv(REPORTS_DIR / "validation_summary.csv")
    oof.write_parquet(OOF_PATH)

    sums = b1_sums()
    final = b1_fit(sums, FOLD_YEARS)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final.write_parquet(B1_RATIOS_PATH)
    iv = b1_interval(sums)
    iv.write_parquet(B1_INTERVAL_PATH)
    plot_korea_monthly(oof, ["B1"])
    yb = plot_yearly_bias(oof, ["B1", "B1_terra"])

    meta = {
        "stage": "b1_only",
        "final_method": "B1",
        "final_method_reason": "1단계: B1 비율 기준선만 학습. B2·M1 비교 후 채택 규칙으로 다시 결정.",
        "method_version": f"m1-b1-{datetime.now():%Y-%m-%d}",
        "train_period": f"{TRAIN_START[0]}-{TRAIN_START[1]:02d} ~ {TRAIN_END[0]}-{TRAIN_END[1]:02d}",
        "target": "S-NPP 감지 수, 공통 정상일(Aqua·Terra·S-NPP 모두 ok) 집계를 days_in_month/common_days로 월 환산",
        "b1": {"group": f"위도 {LAT_BAND}° 띠 × 월, 분모 0이면 월만", "ratios": B1_RATIOS_PATH.name,
               "interval": B1_INTERVAL_PATH.name,
               "interval_method": "OOF 칸·월 y/예측 비율의 10·90 백분위수 (예측 크기 구간별)"},
        "zero_rule": "B1은 MODIS 감지가 0이면 예측 0 (규칙 켬/끔 결과 동일)",
        "evaluation": "leave-year-out 2012~2025; 지역·월 합계는 전체 칸(이력 칸 ∪ 한국 칸) × 정상 월",
        "seed": SEED,
        "created": datetime.now().isoformat(timespec="seconds"),
        "validation_summary": {
            r["region"]: {k: r[k] for k in ("metric", "mean")}
            for r in summ.filter(pl.col("model") == "B1", pl.col("metric") == "pooled_month_total_MAE").to_dicts()
        },
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = dict(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_formatting="ASCII_MARKDOWN",
               tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, float_precision=3)
    with pl.Config(**cfg):
        show = summ.filter(pl.col("metric").is_in(["pooled_month_total_MAE", "pooled_month_total_bias",
                                                   "pooled_month_total_pearson", "pooled_month_total_spearman",
                                                   "month_total_spearman", "cell_MAE", "cell_RMSE_log1p"]))
        print(show.pivot(on="metric", index=["model", "region"], values="mean").sort("model", "region"))
        print(yb.filter(pl.col("model") == "B1", pl.col("region").is_in(["world", "korea", "argentina"]))
              .pivot(on="region", index="year", values="bias").sort("year"))
        print(iv)


if __name__ == "__main__":
    main()
