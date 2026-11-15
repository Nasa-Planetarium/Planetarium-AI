"""README figures for judges -> docs/figures/*.png

Every number comes from files already in the repository or the data release:
  reports/validation_summary.csv, reports/m2v_yearly_bias.csv, reports/m2v_leave_continent_out.csv,
  reports/m2v_aoi_interval_coverage.csv, reports/planner_daily_metrics.csv, reports/planner_monthly_metrics.csv,
  reports/compare_continuity_validation.csv, data/processed/aoi_month.parquet, data/processed/grid_occurrence_nextmonth_latest.parquet

Figures are drawn on a light card (solid background) so they stay readable on both light and dark GitHub themes.
Run: uv run python scripts/make_readme_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "figures"

# Reference categorical palette (validated light-mode steps), fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY = "#8a8984"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DPI = 160

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    w, h = plt.imread(p).shape[1::-1]
    print(f"{name}: {w}x{h}px")


def title(fig, main: str, sub: str) -> None:
    fig.suptitle(main, x=0.01, y=1.02, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.01, 0.955, sub, ha="left", fontsize=12, color=INK2)


# ① naive splice vs harmonized ---------------------------------------------------------------
def fig_naive_vs_harmonized() -> None:
    am = pl.read_parquet(PROC / "aoi_month.parquet", columns=["aoi_id", "year", "month", "value", "raw_modis", "raw_viirs"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (aid, name) in zip(axes, [("WORLD", "World"), ("ADM0_ARG", "Argentina")]):
        d = am.filter(pl.col("aoi_id") == aid, pl.col("year").is_between(2001, 2025)).sort("year", "month")
        d = d.with_columns(naive=pl.when((pl.col("year") < 2012) | ((pl.col("year") == 2012) & (pl.col("month") == 1)))
                           .then(pl.col("raw_modis")).otherwise(pl.col("raw_viirs")).cast(pl.Float64))
        d = d.with_columns(pl.col("naive", "value").rolling_mean(12), t=pl.col("year") + (pl.col("month") - 0.5) / 12)
        t = d["t"].to_numpy()
        scale = 1e6 if aid == "WORLD" else 1e3
        unit = "million" if aid == "WORLD" else "thousand"
        ax.plot(t, d["naive"] / scale, color=ORANGE, label="Naive splice (raw MODIS → raw VIIRS)")
        ax.plot(t, d["value"] / scale, color=BLUE, label="Harmonized (FireBridge, S-NPP equivalent)")
        ax.axvline(2012.08, color=GREY, ls="--", lw=1.2)
        ax.text(2012.3, ax.get_ylim()[1] * 0.97, "VIIRS S-NPP\nstarts 2012-02", va="top", fontsize=11, color=INK2)
        ax.set_title(name, loc="left")
        ax.set_ylabel(f"Fire detections per month ({unit})\n12-month moving average")
        ax.set_xlabel("Year")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        i = int(np.nanargmax(t > 2006))
        if aid != "WORLD":
            continue
        ax.annotate("naive", (t[i], d["naive"][i] / scale), xytext=(0, -26), textcoords="offset points", color=ORANGE,
                    fontweight="bold", ha="center", bbox=dict(fc=SURFACE, ec="none", pad=1))
        ax.annotate("harmonized", (t[i], d["value"][i] / scale), xytext=(0, 26), textcoords="offset points", color=BLUE,
                    fontweight="bold", ha="center", bbox=dict(fc=SURFACE, ec="none", pad=1))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=12)
    title(fig, "Splicing raw MODIS onto VIIRS creates a false jump in 2012",
          "Monthly fire detections, 2001–2025 (data/processed/aoi_month.parquet). Static industrial sources removed from the harmonized series only.")
    fig.subplots_adjust(top=0.86, wspace=0.22)
    save(fig, "01_naive_vs_harmonized.png")


# ② FireBridge MAE --------------------------------------------------------------------------
MODELS = [("ratio_lat", "Latitude-band ratio", GREY), ("loglinear_ridge", "Log-linear ridge", YELLOW),
          ("ratio_country", "Country ratio", AQUA), ("m2v_xgb", "FireBridge", BLUE)]


def fig_mae() -> None:
    v = pl.read_csv(REP / "validation_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    for ax, (reg, name, unit) in zip(axes, [("korea", "Korea", "detections / month"), ("argentina", "Argentina", "detections / month"),
                                            ("world", "World", "thousand detections / month")]):
        vals = []
        for m, lab, c in MODELS:
            x = v.filter(pl.col("model") == m, pl.col("region") == reg, pl.col("metric") == "pooled_month_total_MAE")["mean"][0]
            vals.append((lab, x / (1e3 if reg == "world" else 1), c))
        y = np.arange(len(vals))
        ax.barh(y, [x for _, x, _ in vals], color=[c for *_, c in vals], height=0.62)
        for yi, (lab, x, _) in zip(y, vals):
            ax.text(x, yi, f"  {x:,.0f}", va="center", fontsize=12, color=INK)
        ax.set_yticks(y, [lab for lab, *_ in vals])
        ax.set_title(name, loc="left")
        ax.set_xlabel(f"MAE of monthly totals ({unit})")
        ax.set_xlim(0, max(x for _, x, _ in vals) * 1.25)
        ax.grid(axis="y", visible=False)
    title(fig, "FireBridge: lowest error on monthly totals (leave-year-out 2012–2025)",
          "Mean absolute error of regional monthly totals vs observed VIIRS S-NPP, pooled over 14 held-out years (reports/validation_summary.csv). Lower is better.")
    fig.subplots_adjust(top=0.83, wspace=0.55)
    save(fig, "02_firebridge_mae.png")


# ③ yearly bias ----------------------------------------------------------------------------
def fig_bias() -> None:
    b = pl.read_csv(REP / "m2v_yearly_bias.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
    for ax, (reg, name) in zip(axes, [("korea", "Korea"), ("argentina", "Argentina"), ("world", "World")]):
        for m, lab, c in [("ratio_lat", "Baseline", GREY), MODELS[3]]:
            d = b.filter(pl.col("model") == m, pl.col("region") == reg).sort("year")
            ax.plot(d["year"], d["bias"], color=c, marker="o", ms=7, label=lab)
            ax.annotate(lab, (d["year"][-1], d["bias"][-1]), xytext=(6, 0), textcoords="offset points", va="center", color=c,
                        fontweight="bold", fontsize=11)
        ax.axhline(1, color=INK2, lw=1.2)
        ax.axhspan(0.9, 1.1, color=GRID, alpha=0.6, lw=0)
        ax.set_title(name, loc="left")
        ax.set_xlabel("Held-out year")
        ax.set_xlim(2011.5, 2028.3)
        ax.set_xticks([2012, 2015, 2018, 2021, 2024])
    axes[0].set_ylabel("Bias = predicted ÷ observed annual total\n(1.0 = unbiased; band ±10%)")
    title(fig, "Year-by-year bias stays near 1.0 for FireBridge",
          "Each year predicted by a model trained without that year (reports/m2v_yearly_bias.csv). Baseline: latitude-band × month ratio.")
    fig.subplots_adjust(top=0.83, wspace=0.08)
    save(fig, "03_firebridge_yearly_bias.png")


# ④ leave-continent-out -----------------------------------------------------------------------
def fig_lco() -> None:
    d = pl.read_csv(REP / "m2v_leave_continent_out.csv")
    regs = ["korea", "argentina", "continent:Africa", "continent:Asia", "continent:Europe", "continent:North America",
            "continent:Oceania", "continent:South America"]
    lab = [r.replace("continent:", "").replace("korea", "Korea").replace("argentina", "Argentina") for r in regs]
    series = [("m2v_xgb", "LYO", "FireBridge — region seen in training (new year)", BLUE, "o"),
              ("m2v_xgb", "LCO", "FireBridge — whole continent held out", ORANGE, "s"),
              ("ratio_country", "LCO", "Country-ratio fallback — continent held out", AQUA, "D")]
    fig, ax = plt.subplots(figsize=(16, 6.4))
    y = np.arange(len(regs))
    for k, (m, s, name, c, mk) in enumerate(series):
        x = [d.filter(pl.col("model") == m, pl.col("scheme") == s, pl.col("region") == r)["month_total_bias"][0] for r in regs]
        ax.scatter(x, y + (k - 1) * 0.22, color=c, marker=mk, s=90, label=name, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.axvline(1, color=INK2, lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 0.75, 1, 1.5, 2, 3], ["0.25×", "0.5×", "0.75×", "1×", "1.5×", "2×", "3×"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y, lab)
    ax.invert_yaxis()
    ax.set_xlabel("Bias of monthly totals = predicted ÷ observed (log scale; 1× = unbiased)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, fontsize=12)
    title(fig, "Spatial generalization: reliable where trained, not on unseen continents",
          "Leave-year-out (LYO) vs leave-continent-out (LCO), monthly totals 2012-02 – 2025-12 (reports/m2v_leave_continent_out.csv).")
    fig.subplots_adjust(top=0.86)
    save(fig, "04_firebridge_leave_continent_out.png")


# ⑤ Planner BSS by lead time -----------------------------------------------------------------
def fig_bss() -> None:
    dd = pl.read_csv(REP / "planner_daily_metrics.csv").filter(pl.col("model") == "ens")
    mm = pl.read_csv(REP / "planner_monthly_metrics.csv").filter(pl.col("model") == "ens", pl.col("group") == "ALL")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True, gridspec_kw={"width_ratios": [1, 1.6]})
    per_d = [("val_2020_2023", "Validation 2020–2023", BLUE, "o"), ("hold_2024_2025", "Holdout 2024–2025", ORANGE, "s"),
             ("extra_2026", "Extra check 2026", AQUA, "D")]
    per_m = [("validation 2020-2023", *per_d[0][1:]), ("holdout 2024-2025", *per_d[1][1:]), ("extra 2026", *per_d[2][1:])]
    for p, name, c, mk in per_d:
        d = dd.filter(pl.col("period") == p).sort("horizon_days")
        axes[0].plot(d["horizon_days"], d["bss_vs_base_clim"] * 100, color=c, marker=mk, ms=8, label=name)
    for p, name, c, mk in per_m:
        d = mm.filter(pl.col("period") == p, pl.col("bss_vs_base_rate").is_not_null()).sort("horizon_months")
        axes[1].plot(d["horizon_months"], d["bss_vs_base_rate"] * 100, color=c, marker=mk, ms=8, label=name,
                     ls="-" if p != "extra 2026" else ":")
    axes[0].set_title("Daily model — detection within k days", loc="left")
    axes[0].set_xlabel("Lead time (days)")
    axes[0].set_xticks([1, 2, 3, 7])
    axes[0].set_ylabel("Brier skill score vs climatology (%)\nhigher is better; 0 = no better than the normal")
    axes[1].set_title("Monthly model — detection in month h", loc="left")
    axes[1].set_xlabel("Lead time (months)")
    axes[1].set_xticks(range(1, 13))
    for ax in axes:
        ax.axhline(0, color=INK2, lw=1.2)
        ax.set_ylim(0, 20)
    axes[1].legend(loc="upper right", fontsize=12)
    title(fig, "FireOutlook Planner keeps positive skill from 1 day to 12 months ahead",
          "0.5° grid, all land cells, persistent industrial cells excluded. Ensemble (XGBoost + LightGBM, isotonic). "
          "reports/planner_daily_metrics.csv, reports/planner_monthly_metrics.csv")
    fig.subplots_adjust(top=0.84, wspace=0.06)
    save(fig, "05_planner_bss_by_lead.png")


# ⑥ next-month occurrence map ----------------------------------------------------------------
def fig_map() -> None:
    g = pl.read_parquet(PROC / "grid_occurrence_nextmonth_latest.parquet",
                        columns=["lat", "lon", "p_any", "persistent_source", "target_month"])
    tm = g["target_month"][0]
    A = np.full((360, 720), np.nan)
    P = np.zeros((360, 720), bool)
    r = ((90 - g["lat"].to_numpy()) / 0.5 - 0.5).round().astype(int)
    c = ((g["lon"].to_numpy() + 180) / 0.5 - 0.5).round().astype(int)
    A[r, c] = g["p_any"].to_numpy()
    P[r, c] = g["persistent_source"].to_numpy()
    A[P] = np.nan
    cmap = LinearSegmentedColormap.from_list("fire", ["#fbe6da", "#f2a07a", "#e2622f", "#a63a12", "#5e1d05"])
    cmap.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(17, 8.2))
    im = ax.imshow(A[:330], extent=(-180, 180, -75, 90), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    pr, pc = np.nonzero(P)
    ax.scatter(-180 + (pc + 0.5) * 0.5, 90 - (pr + 0.5) * 0.5, s=10, color=INK2, label="Persistent industrial source (grey, not ranked)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("P(≥ 1 fire detection next month)")
    cb.outline.set_visible(False)
    ax.legend(loc="lower left", fontsize=12, markerscale=2.5)
    n = g.height
    title(fig, f"FireOutlook Monthly – Grid: probability of burning activity in {tm}",
          f"{n:,} land 0.5° cells with any detection since 2012-02; blank = no detection history "
          "(data/processed/grid_occurrence_nextmonth_latest.parquet).")
    fig.subplots_adjust(top=0.9)
    save(fig, "06_grid_occurrence_map.png")


# ⑦ continuity self-check: Korea verdict mix -------------------------------------------------
def fig_continuity() -> None:
    v = pl.read_csv(REP / "compare_continuity_validation.csv")
    rows = [("KR", "Korea"), ("continuity fail (all)", "All regions that fail\nthe continuity check"),
            ("continuity pass (all)", "All regions that pass\n(unchanged)")]
    cats = [("above", "Above normal", ORANGE), ("similar", "Similar", GREY), ("below", "Below normal", BLUE)]
    fig, ax = plt.subplots(figsize=(16, 6.4))
    labels, ticks, y = [], [], 0
    for g, name in rows:
        r = v.filter(pl.col("group") == g).row(0, named=True)
        for tag, when in (("old", "before"), ("new", "after")):
            left = 0
            for k, lab, c in cats:
                w = r[f"recent3_{tag}_{k}"] * 100
                ax.barh(y, w, left=left, color=c, height=0.7, edgecolor=SURFACE, linewidth=2)
                if w >= 6:
                    ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", color="white" if c != GREY else INK,
                            fontsize=12, fontweight="bold")
                left += w
            ticks.append(y)
            labels.append(f"{name.replace(chr(10), ' ')} — {when}")
            y += 1
        y += 0.5
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of decided verdicts for 2023–2025 months (%)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for *_, c in cats]
    ax.legend(handles, [lab for _, lab, _ in cats], loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, fontsize=12)
    title(fig, "Continuity self-check removes a false 'below normal' tilt",
          "Same-month comparison verdicts; a balanced split is about 20% / 60% / 20% (reports/compare_continuity_validation.csv).")
    fig.subplots_adjust(top=0.86)
    save(fig, "07_continuity_verdicts.png")


# ⑧ uncertainty band coverage ----------------------------------------------------------------
def fig_coverage() -> None:
    d = pl.read_csv(REP / "m2v_aoi_interval_coverage.csv")
    lv = [("sgg", "Korea municipalities"), ("sido", "Korea provinces"), ("adm1", "Provinces / states"),
          ("country", "Countries"), ("continent", "Continents"), ("world", "World")]
    fig, ax = plt.subplots(figsize=(16, 5.6))
    x = np.arange(len(lv))
    cov = [d.filter(pl.col("level_bin") == k)["coverage"][0] * 100 for k, _ in lv]
    ax.bar(x, cov, color=BLUE, width=0.6)
    for xi, c in zip(x, cov):
        inside = 76 <= c < 84
        ax.text(xi, c - 6 if inside else c + 1.5, f"{c:.0f}%", ha="center", fontsize=12,
                color="white" if inside else INK, fontweight="bold" if inside else "normal")
    ax.axhline(80, color=ORANGE, lw=2, ls="--")
    ax.text(4, 82, "nominal 80% (q10–q90)", color=ORANGE, ha="center", fontweight="bold")
    ax.set_xticks(x, [n for _, n in lv])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Observed totals inside the band (%)")
    ax.grid(axis="x", visible=False)
    title(fig, "FireBridge Uncertainty Band: coverage of regional monthly totals",
          "Out-of-fold leave-year-out predictions, 2012–2025 (reports/m2v_aoi_interval_coverage.csv).")
    fig.subplots_adjust(top=0.84)
    save(fig, "08_interval_coverage.png")


if __name__ == "__main__":
    fig_naive_vs_harmonized()
    fig_mae()
    fig_bias()
    fig_lco()
    fig_bss()
    fig_map()
    fig_continuity()
    fig_coverage()
