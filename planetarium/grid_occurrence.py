"""다음 달 '열원 1건 이상' 확률 — 전 세계 0.5° 격자 (실험, 2026-11-15).

타깃   0.5° 격자 × 달: 기준 점 수(harmonized_cell_day.n_points, 고정 열원 제외 v1.2) ≥ 1.
       관측 구간 2012-02 이후, WORLD status가 ok·nrt인 달 (진행 중인 달 제외).
대상   2012-02 이후 한 번이라도 탐지된 격자, 육지(0.1° 칸 중 is_land가 하나라도 있음), 남극·공해 제외.
특성   (대상 달 t, 예측 시점 = t-1 달 말)
  rate_all, rate_5y, n_years   같은 달 발생 비율 (2012 ~ 직전 해 / 최근 5년)
  med_value                    같은 달 값(harmonized_cell_month.value 합) 중앙값
  lag1~3_value, lastyear_value 직전 1~3개월, 작년 같은 달 값 (진행 중인 달은 value_projected)
  country_rate                 같은 나라 격자들의 같은 달 발생 비율 평균
  nb_rate                      이웃 3×3 격자 같은 달 발생 비율 평균 (자기 포함, 대상 밖 격자 = 0)
  month_sin, month_cos
지속 열원 persistent_source (가스 플레어·제철소 등 의심; 학습에는 넣고, 지표는 전체와 제외 둘 다):
       2012-02 ~ 2026-10 탐지 달 비율 ≥ 0.75, 달력 월별 평균 탐지 수의 변동계수 ≤ 1.0 (계절 변동 작음),
       기준 점의 상위 3개 0.01° 칸 집중 비율 ≥ 0.12 (공간 집중). 근거: reports/grid_occurrence_persistence_check.csv
검증   확장 창 2020~2025: 학습 ≤ Y-2, Y-1로 조기 종료·보정(Platt/isotonic 중 Brier 낮은 쪽), Y 예측.

실행  uv run python -m planetarium.grid_occurrence agg|panel|validate|report|latest|all
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date

import numpy as np
import polars as pl
from scipy.ndimage import uniform_filter
from tqdm import tqdm

from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.occurrence import Platt
from planetarium.runlog import peak_memory_gb, start_run_log

ROOT = PROCESSED_DATA_DIR.parent.parent
MODELS_DIR = ROOT / "models"
WORK = INTERIM_DATA_DIR / "grid_occurrence"
AGG = WORK / "agg.parquet"
CELLS = WORK / "cells.parquet"
PANEL = WORK / "panel.parquet"
OOF = WORK / "oof.parquet"
LATEST_PATH = PROCESSED_DATA_DIR / "grid_occurrence_nextmonth_latest.parquet"
MODEL_PATH = MODELS_DIR / "grid_occurrence_xgb.json"
CAL_PATH = MODELS_DIR / "grid_occurrence_calibration.json"
META_PATH = MODELS_DIR / "grid_occurrence_meta.json"
FIRST = 2012 * 12 + 1          # 2012-02
TARGET = 2026 * 12 + 11        # 2026-12
T0 = FIRST - 12
TEST_YEARS = list(range(2020, 2026))
PERS_OCC, PERS_COUNT_CV, PERS_TOP3 = 0.75, 1.0, 0.12
HORIZON = 1   # 예측 시점: 대상 달보다 HORIZON개월 앞 달 말 (1 = 다음 달). 직전 달 특징(lag)만 이만큼 앞당긴다
FEATURES = ["rate_all", "rate_5y", "n_years", "med_value", "lag1_value", "lag2_value", "lag3_value", "lastyear_value",
            "country_rate", "nb_rate", "month_sin", "month_cos"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
          "max_depth": 6, "learning_rate": 0.08, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8,
          "lambda": 1.0, "max_bin": 256, "seed": 42}


def id05(cell_id: pl.Expr) -> pl.Expr:
    return (cell_id // 3600 // 5) * 720 + (cell_id % 3600) // 5


def tcol() -> pl.Expr:
    return pl.col("year").cast(pl.Int32) * 12 + pl.col("month").cast(pl.Int32) - 1


def agg() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    n = (pl.scan_parquet(PROCESSED_DATA_DIR / "harmonized_cell_day.parquet").with_columns(t=tcol())
         .filter(pl.col("t") >= T0).group_by(g=id05(pl.col("cell_id")), t="t").agg(n=pl.col("n_points").sum()))
    v = (pl.scan_parquet(PROCESSED_DATA_DIR / "harmonized_cell_month.parquet").with_columns(t=tcol())
         .filter(pl.col("t") >= T0)
         .group_by(g=id05(pl.col("cell_id")), t="t")
         .agg(v=pl.when(pl.col("status") == "partial").then(pl.col("value_projected")).otherwise(pl.col("value")).sum()))
    a = n.join(v, on=["g", "t"], how="full", coalesce=True).with_columns(pl.col("n").fill_null(0), pl.col("v").fill_null(0.0)).collect(
        engine="streaming")
    a.write_parquet(AGG)
    w = pl.read_parquet(PROCESSED_DATA_DIR / "world_cells_0.1deg.parquet", columns=["cell_id", "is_land", "land_frac", "adm0_a3", "continent", "kr_sgg_cd"])
    w = w.with_columns(g=id05(pl.col("cell_id")), adm0=pl.when(pl.col("kr_sgg_cd").is_not_null()).then(pl.lit("KOR")).otherwise("adm0_a3"))
    attr = w.group_by("g").agg(land=pl.col("is_land").any(), lf=pl.col("land_frac").sum(),
                               continent=pl.col("continent").drop_nulls().mode().first(),
                               adm0=pl.col("adm0").drop_nulls().mode().first())
    ever = a.filter(pl.col("t").is_between(FIRST, TARGET - 2), pl.col("n") >= 1).select("g").unique()
    cells = attr.join(ever, on="g", how="semi").filter(pl.col("land"), ~pl.col("continent").is_in(["Antarctica", "Seven seas (open ocean)"]),
                                                       pl.col("continent").is_not_null())
    cells = cells.with_columns(lat=((pl.col("g") // 720) * 0.5 - 90 + 0.25).cast(pl.Float32),
                               lon=((pl.col("g") % 720) * 0.5 - 180 + 0.25).cast(pl.Float32)).sort("g")
    cells.write_parquet(CELLS)
    print(f"[완료] 집계 {a.height:,}행 / 대상 격자 {cells.height:,} (탐지된 격자 {ever.height:,} 중 육지·남극 제외), "
          f"최대 메모리 {peak_memory_gb() or 0:.2f} GB", flush=True)


def panel() -> None:
    cells = pl.read_parquet(CELLS)
    a = pl.read_parquet(AGG).join(cells.select("g"), on="g", how="semi")
    st = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_month.parquet", columns=["aoi_id", "year", "month", "status"]).filter(pl.col("aoi_id") == "WORLD")
    status = dict(zip(st.with_columns(t=tcol())["t"], st["status"]))
    T = list(range(T0, TARGET + 1))
    nt, nc = len(T), cells.height
    gi = {g: i for i, g in enumerate(cells["g"].to_list())}
    N = np.zeros((nt, nc), np.float32)
    V = np.zeros((nt, nc), np.float32)
    ti = np.array([t - T0 for t in a["t"].to_list()])
    ci = np.array([gi[g] for g in a["g"].to_list()])
    N[ti, ci] = a["n"].to_numpy()
    V[ti, ci] = a["v"].to_numpy()
    valid = np.array([(t >= FIRST) and status.get(t) in ("ok", "nrt") for t in T])
    Y = np.where(valid[:, None], (N >= 1).astype(np.float32), np.nan)
    rr, cc = (cells["g"] // 720).to_numpy(), (cells["g"] % 720).to_numpy()
    adm = cells["adm0"].fill_null("NA").to_list()
    codes = {c: i for i, c in enumerate(sorted(set(adm)))}
    ac = np.array([codes[c] for c in adm])
    cnt_c = np.bincount(ac, minlength=len(codes))
    WORK.mkdir(parents=True, exist_ok=True)
    parts = []
    for k, t in enumerate(tqdm(T, desc="패널", unit="month", dynamic_ncols=True)):
        if t < FIRST:
            continue
        same = [k - 12 * j for j in range(1, 40) if k - 12 * j >= 0 and valid[k - 12 * j]]
        if same:
            ys = Y[same]
            rate_all = ys.mean(axis=0)
            rate_5y = Y[same[:5]].mean(axis=0)
            med = np.median(V[same], axis=0)
            dense = np.zeros((360, 720), np.float32)
            dense[rr, cc] = rate_all
            nb = uniform_filter(dense, size=3, mode="constant")[rr, cc]
            crate = (np.bincount(ac, weights=rate_all, minlength=len(codes)) / np.maximum(cnt_c, 1))[ac]
        else:
            rate_all = rate_5y = med = nb = crate = np.full(nc, np.nan, np.float32)
        m = t % 12 + 1
        parts.append(pl.DataFrame({
            "g": cells["g"].to_numpy(), "year": np.full(nc, t // 12, np.int16), "month": np.full(nc, m, np.int8), "label": Y[k],
            "rate_all": rate_all, "rate_5y": rate_5y, "n_years": np.full(nc, len(same), np.float32), "med_value": med,
            "lag1_value": V[k - HORIZON], "lag2_value": V[k - HORIZON - 1], "lag3_value": V[k - HORIZON - 2], "lastyear_value": V[k - 12],
            "lastyear_any": Y[k - 12] if valid[k - 12] else np.full(nc, np.nan, np.float32),
            "country_rate": crate, "nb_rate": nb,
            "month_sin": np.full(nc, np.sin(2 * np.pi * m / 12), np.float32), "month_cos": np.full(nc, np.cos(2 * np.pi * m / 12), np.float32),
            "lag1_projected": np.full(nc, status.get(t - HORIZON) == "partial"),
        }))
    p = pl.concat(parts, how="vertical_relaxed").fill_nan(None)
    lab = p.filter(pl.col("label").is_not_null())
    p.write_parquet(PANEL)
    print(f"[완료] panel {p.height:,}행 (라벨 {lab.height:,}, 양성 {lab['label'].mean():.3f}), 격자 {nc:,}, "
          f"최대 메모리 {peak_memory_gb() or 0:.2f} GB", flush=True)


def concentration() -> pl.DataFrame:
    """격자별 2012-02 ~ 2026-10 기준 점(고정 열원 제외)이 상위 3개 0.01° 칸에 몰린 비율 (연 단위로 읽음)."""
    from planetarium.static_exclusion import is_static_expr, mask_ids
    mask, parts = mask_ids(), []
    for y in tqdm(range(2012, 2027), desc="공간 집중도", unit="year", dynamic_ncols=True):
        lf = pl.scan_parquet(PROCESSED_DATA_DIR / f"harmonized_hotspots/reference/year={y}/*/*.parquet").filter(
            pl.col("date_local").is_between(pl.date(2012, 2, 1), pl.date(2026, 10, 31)), ~is_static_expr(mask))
        parts.append(lf.select(g=id05(pl.col("cell_id")),
                               c001=((pl.col("latitude") + 90) / 0.01).floor().cast(pl.Int64) * 36000
                               + ((pl.col("longitude") + 180) / 0.01).floor().cast(pl.Int64))
                     .group_by("g", "c001").agg(n=pl.len()).collect(engine="streaming"))
    a = pl.concat(parts).group_by("g", "c001").agg(pl.col("n").sum())
    return (a.sort(["g", "n"], descending=[False, True]).group_by("g", maintain_order=True)
            .agg(n_points=pl.col("n").sum(), top3=pl.col("n").head(3).sum())
            .with_columns(top3_share=pl.col("top3") / pl.col("n_points")).drop("top3"))


CHECK_SITES = {"KR Ulsan": (35.25, 129.25), "KR Pohang": (35.75, 129.25), "KR Dangjin": (36.75, 126.75), "KR Gwangyang": (34.75, 127.75),
               "IQ Basra": (30.25, 47.75), "NG Niger Delta": (4.75, 6.75), "RU Khanty-Mansiysk (Samotlor)": (61.25, 76.75),
               "US Permian (Delaware)": (31.75, -103.75), "US Permian (NM)": (32.25, -103.25),
               "BR Amazon (Novo Progresso)": (-7.25, -55.25), "BR Amazon (Rondonia)": (-9.25, -63.25), "CD Congo basin": (-3.25, 23.25),
               "CD Congo basin 2": (-6.25, 23.75), "AU N savanna (Kimberley)": (-16.25, 126.25), "AU N savanna (Arnhem)": (-12.75, 134.25)}


def persistence() -> None:
    cells = pl.read_parquet(CELLS)
    conc = concentration()
    conc.write_parquet(WORK / "concentration.parquet")
    a = (pl.read_parquet(AGG).filter(pl.col("t").is_between(FIRST, TARGET - 2)).join(cells.select("g"), on="g", how="semi")
         .with_columns(m=pl.col("t") % 12))
    nmon = pl.DataFrame({"t": list(range(FIRST, TARGET - 1))}).with_columns(m=pl.col("t") % 12).group_by("m").agg(n_months=pl.len())
    cal = a.group_by("g", "m").agg(nm=pl.col("n").sum()).join(nmon, on="m").with_columns(mean_n=pl.col("nm") / pl.col("n_months"))
    full = (cells.select("g").join(pl.DataFrame({"m": list(range(12))}), how="cross")
            .join(cal.select("g", "m", "mean_n"), on=["g", "m"], how="left").with_columns(pl.col("mean_n").fill_null(0)))
    lab = pl.read_parquet(PANEL, columns=["g", "label"]).filter(pl.col("label").is_not_null())
    st = (full.group_by("g").agg(count_cv=pl.col("mean_n").std() / pl.col("mean_n").mean())
          .join(lab.group_by("g").agg(occ_share=pl.col("label").mean()), on="g").join(conc, on="g", how="left")
          .with_columns(persistent_source=(pl.col("occ_share") >= PERS_OCC) & (pl.col("count_cv") <= PERS_COUNT_CV)
                        & (pl.col("top3_share").fill_null(0) >= PERS_TOP3)))
    drop = ("persistent_source", "occ_share", "seasonal_cv", "month_rate_min", "count_cv", "top3_share", "n_points")
    cells = cells.select([c for c in cells.columns if c not in drop]).join(st, on="g", how="left").with_columns(
        pl.col("persistent_source").fill_null(False))
    cells.write_parquet(CELLS)
    rows = []
    for k, (la, lo) in CHECK_SITES.items():
        r = cells.filter((pl.col("lat") - la).abs() < 0.01, (pl.col("lon") - lo).abs() < 0.01)
        base = {"site": k, "lat": la, "lon": lo}
        rows.append({**base, **r.select("occ_share", "count_cv", "top3_share", "n_points", "persistent_source").to_dicts()[0]} if r.height
                    else {**base, "persistent_source": None})
    chk = pl.DataFrame(rows)
    chk.write_csv(REPORTS_DIR / "grid_occurrence_persistence_check.csv")
    top = cells.filter("persistent_source").group_by("adm0").len().sort("len", descending=True)
    top.write_csv(REPORTS_DIR / "grid_occurrence_persistence_by_country.csv")
    with pl.Config(tbl_rows=-1, float_precision=3, tbl_width_chars=200, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True):
        print(f"[지속 열원] {int(cells['persistent_source'].sum()):,} / {cells.height:,} 격자 (탐지 달 ≥ {PERS_OCC}, "
              f"탐지 수 계절 변동계수 ≤ {PERS_COUNT_CV}, 상위 3칸 집중 ≥ {PERS_TOP3}), 최대 메모리 {peak_memory_gb() or 0:.2f} GB")
        print(top.head(15))
        print(chk)


def _dm(df, label=True):
    import xgboost as xgb
    return xgb.DMatrix(df.select(FEATURES).to_numpy().astype(np.float32), label=df["label"].to_numpy() if label else None,
                       feature_names=FEATURES, missing=np.nan)


def fit_fold(tr, va):
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    bst = xgb.train(PARAMS, _dm(tr), 3000, evals=[(_dm(va), "va")], early_stopping_rounds=50, verbose_eval=False)
    r = bst.predict(_dm(va, False), iteration_range=(0, bst.best_iteration + 1))
    yv = va["label"].to_numpy()
    return bst, IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(r, yv), Platt().fit(r, yv)


def validate() -> None:
    p = pl.read_parquet(PANEL).filter(pl.col("label").is_not_null())
    outs, log = [], []
    for Y in tqdm(TEST_YEARS, desc="확장 창", unit="fold", dynamic_ncols=True):
        t0 = time.perf_counter()
        bst, iso, platt = fit_fold(p.filter(pl.col("year") <= Y - 2), p.filter(pl.col("year") == Y - 1))
        te = p.filter(pl.col("year") == Y)
        raw = bst.predict(_dm(te, False), iteration_range=(0, bst.best_iteration + 1))
        prior = float(p.filter(pl.col("year") < Y)["label"].mean())
        outs.append(te.select("g", "year", "month", "label").with_columns(
            p_raw=pl.Series(raw), p_iso=pl.Series(iso.predict(raw)), p_platt=pl.Series(platt.predict(raw)),
            p_base_rate=te["rate_all"].fill_null(prior), p_base_lastyear=te["lastyear_any"].fill_null(0.0)))
        log.append({"test_year": Y, "rounds": bst.best_iteration + 1, "sec": round(time.perf_counter() - t0, 1),
                    "peak_mem_gb": round(peak_memory_gb() or 0, 2)})
        tqdm.write(f"[{Y}] rounds {log[-1]['rounds']}, {log[-1]['sec']}초, 최대 메모리 {log[-1]['peak_mem_gb']} GB")
    pl.concat(outs).write_parquet(OOF)
    (WORK / "folds.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


def _m(y, s):
    from sklearn.metrics import average_precision_score, brier_score_loss
    if len(y) == 0 or y.min() == y.max():
        return {"pr_auc": None, "brier": None}
    return {"pr_auc": float(average_precision_score(y, s)), "brier": float(brier_score_loss(y, np.clip(s, 0, 1)))}


METHODS = {"p_iso": "xgboost_isotonic", "p_platt": "xgboost_platt", "p_raw": "xgboost_raw",
           "p_base_rate": "baseline_same_month_rate", "p_base_lastyear": "baseline_last_year_same_month"}


def report() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = pl.read_parquet(CELLS).select("g", "continent", "adm0", "persistent_source")
    oof = pl.read_parquet(OOF).join(cells, on="g")
    groups = [("ALL", pl.lit(True))] + [(c, pl.col("continent") == c) for c in sorted(cells["continent"].unique().to_list())] + \
             [("KOR", pl.col("adm0") == "KOR"), ("ARG", pl.col("adm0") == "ARG")]
    rows = []
    for subset, filt in (("all", pl.lit(True)), ("excl_persistent", ~pl.col("persistent_source"))):
        d0 = oof.filter(filt)
        for gname, gf in groups:
            g = d0.filter(gf)
            y = g["label"].to_numpy()
            for c, n in METHODS.items():
                rows.append({"subset": subset, "group": gname, "method": n, "rows": g.height, "pos_rate": float(y.mean()) if len(y) else None,
                             **_m(y, g[c].to_numpy())})
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "grid_occurrence_metrics.csv")
    ex = m.filter(pl.col("subset") == "excl_persistent")
    br = {n: ex.filter(pl.col("method") == n, pl.col("group") == "ALL")["brier"][0] for n in ("xgboost_platt", "xgboost_isotonic")}
    cal = "platt" if br["xgboost_platt"] <= br["xgboost_isotonic"] else "isotonic"
    mname = f"xgboost_{cal}"
    wide = ex.filter(pl.col("method").is_in([mname, "baseline_same_month_rate", "baseline_last_year_same_month"])).pivot(
        on="method", index=["group", "rows", "pos_rate"], values=["pr_auc", "brier"])
    wide = wide.with_columns(
        beats=(pl.col(f"pr_auc_{mname}") > pl.max_horizontal("pr_auc_baseline_same_month_rate", "pr_auc_baseline_last_year_same_month"))
        & (pl.col(f"brier_{mname}") < pl.min_horizontal("brier_baseline_same_month_rate", "brier_baseline_last_year_same_month")))
    conts = wide.filter(~pl.col("group").is_in(["ALL", "KOR", "ARG"]))
    adopt = bool(wide.filter(pl.col("group") == "ALL")["beats"][0]) and conts["beats"].sum() > conts.height / 2
    # 보정 곡선 (지속 열원 제외)
    d = oof.filter(~pl.col("persistent_source"))
    y = d["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 5))
    bins = np.linspace(0, 1, 11)
    for c, lab, col in ((f"p_{'platt' if cal == 'platt' else 'iso'}", f"XGBoost + {cal}", "#2a6fdb"), ("p_base_rate", "Baseline: same-month rate", "#d9822b")):
        s = d[c].to_numpy()
        idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
        xs, ys_ = zip(*[(s[idx == b].mean(), y[idx == b].mean()) for b in range(10) if (idx == b).sum() >= 100])
        ax.plot(xs, ys_, "o-", color=col, label=lab)
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed share with ≥ 1 detection")
    ax.set_title("Global 0.5° grid, next month, test 2020–2025\n(persistent sources excluded)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "figures" / "grid_occurrence_calibration.png", dpi=150); plt.close(fig)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=4, thousands_separator=","):
        print(wide)
        print(m.filter(pl.col("subset") == "all", pl.col("group") == "ALL").select("method", "pr_auc", "brier"))
    print(f"[채택 판정] 보정 {cal}, 지속 열원 제외 기준: 전체 {'이김' if wide.filter(pl.col('group') == 'ALL')['beats'][0] else '못 이김'}, "
          f"대륙 {int(conts['beats'].sum())}/{conts.height} → {'채택' if adopt else '기준선 사용'}")
    return {"adopt": adopt, "calibration": cal, "table": wide.to_dicts()}


def latest(adopt: bool, cal: str) -> None:
    p = pl.read_parquet(PANEL)
    lab = p.filter(pl.col("label").is_not_null())
    bst, iso, platt = fit_fold(lab.filter(pl.col("year") <= 2024), lab.filter(pl.col("year") == 2025))
    MODELS_DIR.mkdir(exist_ok=True)
    bst.save_model(MODEL_PATH)
    CAL_PATH.write_text(json.dumps({"method": cal, "platt": platt.params(),
                                    "isotonic": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}}), encoding="utf-8")
    te = p.filter((pl.col("year").cast(pl.Int32) * 12 + pl.col("month").cast(pl.Int32) - 1) == TARGET)
    raw = bst.predict(_dm(te, False), iteration_range=(0, bst.best_iteration + 1))
    pm = platt.predict(raw) if cal == "platt" else iso.predict(raw)
    base = te["rate_all"].fill_null(float(lab["label"].mean())).to_numpy()
    as_of = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_month.parquet", columns=["aoi_id", "year", "month", "as_of_date"]).filter(
        pl.col("aoi_id") == "WORLD", pl.col("year") == 2026, pl.col("month") == 11)["as_of_date"][0]
    cells = pl.read_parquet(CELLS)
    out = te.select("g", "rate_5y", "lag1_projected").with_columns(
        p_any=pl.Series((pm if adopt else base).astype(np.float32)), baseline_p=pl.Series(base.astype(np.float32)),
        p_model=pl.Series(pm.astype(np.float32)), target_month=pl.lit("2026-12"),
        model=pl.lit(f"grid_occurrence_xgb+{cal}" if adopt else "baseline_same_month_rate"), as_of_date=pl.lit(as_of)).join(
        cells.select("g", "lat", "lon", "adm0", "continent", "persistent_source"), on="g").with_columns(
        p_none=(1 - pl.col("p_any")).cast(pl.Float32)).rename({"g": "cell_id"}).select(
        "cell_id", "lat", "lon", "target_month", "p_any", "p_none", "baseline_p", "persistent_source", "model", "as_of_date",
        "p_model", "adm0", "continent", "rate_5y", "lag1_projected").sort("cell_id")
    out.write_parquet(LATEST_PATH)
    with pl.Config(tbl_rows=12, float_precision=3):
        print(f"[완료] {LATEST_PATH.name}: {out.height:,}행 (대상 2026-12, as_of {as_of}; 11월은 value_projected)")
        print(out.group_by("continent").agg(n=pl.len(), p_mean=pl.col("p_any").mean(), base_mean=pl.col("baseline_p").mean(),
                                            pers=pl.col("persistent_source").sum()).sort("continent"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["agg", "panel", "persist", "validate", "report", "latest", "all"])
    a = ap.parse_args(argv)
    start_run_log("grid_occurrence")
    t0 = time.perf_counter()
    res = None
    for s in (["agg", "panel", "persist", "validate", "report", "latest"] if a.step == "all" else [a.step]):
        print(f"\n===== {s} =====", flush=True)
        if s == "agg":
            agg()
        elif s == "panel":
            panel()
        elif s == "persist":
            persistence()
        elif s == "validate":
            validate()
        elif s == "report":
            res = report()
            cells = pl.read_parquet(CELLS)
            meta = {"name": "grid_occurrence_nextmonth", "created": date.today().isoformat(), "unit": "0.5° grid cell (id = row*720 + col)",
                    "cells": cells.height, "persistent_sources": int(cells["persistent_source"].sum()),
                    "persistent_rule": {"occ_share_min": PERS_OCC, "monthly_count_cv_max": PERS_COUNT_CV, "top3_c001_share_min": PERS_TOP3,
                                        "check": "reports/grid_occurrence_persistence_check.csv"},
                    "target": "next month >= 1 reference detection (harmonized_cell_day.n_points; static sources excluded v1.2)",
                    "train_period": "2012-02 .. 2026-10 (WORLD status ok/nrt)", "features": FEATURES, "params": PARAMS,
                    "validation": "expanding window 2020-2025, train <= Y-2, early stop + calibration on Y-1",
                    "calibration": res["calibration"], "adopted": res["adopt"],
                    "adoption_rule": "excluding persistent sources: beat both baselines (higher PR-AUC and lower Brier) overall and in most continents",
                    "table_excl_persistent": res["table"]}
            MODELS_DIR.mkdir(exist_ok=True)
            META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        elif s == "latest":
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            latest(meta["adopted"], meta["calibration"])
    print(f"[완료] {a.step} {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
