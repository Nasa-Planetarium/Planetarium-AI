"""다음 달 '열원 1건 이상' 확률 — AOI(한국 시군구 252개) 단위 (실험, 2026-11-15).

타깃   그 달 시군구 안(감지 점 위치 기준, data/interim/fire_events/points_admin_KR.parquet)에
       VIIRS 기준 점(고정 열원 제외 v1.2) ≥ 1. 관측 구간 2012-02 이후만, KR status가 ok·nrt인 달 (진행 중인 달 제외).
특성   (대상 달 t, 예측 시점 = t-1 달 말, 그 이후 정보 없음)
  rate_all, rate_5y     같은 달 발생 비율 (2012 ~ 그 해 직전 해 / 최근 5년), n_years
  med_value             같은 달 aoi_month.value 중앙값 (같은 기간)
  lag1~3_value          직전 1·2·3개월 aoi_month.value (진행 중인 달은 value_projected, lag1_projected 플래그)
  lastyear_value        작년 같은 달 value
  sido_rate             같은 시도 시군구들의 같은 달 발생 비율 평균 (그 해 직전까지)
  fwi_prev, fwi_prev_rel 직전 달 FWI 평균과 그 값의 과거 같은 달(그 해 이전) 대비 백분위 (3년 이상일 때)
  month_sin, month_cos
모델   XGBoost 이진 → 보정 (Platt·isotonic 중 Brier 낮은 쪽). 확장 창: 시험 해 Y, 학습 ≤ Y-2, Y-1로 조기 종료·보정, Y 예측.
기준선 ① rate_all (과거 같은 달 발생 비율; 없으면 전체 과거 비율) ② 작년 같은 달 발생 여부.

실행  uv run python -m planetarium.occurrence panel|validate|report|latest|all
산출  data/interim/occurrence/panel.parquet, oof.parquet
      reports/occurrence_metrics.csv, occurrence_metrics_by_year.csv, occurrence_base_rate_extremes.csv, figures/occurrence_calibration.png
      models/aoi_occurrence_xgb.json, models/aoi_occurrence_calibration.json, models/aoi_occurrence_meta.json
      data/processed/aoi_occurrence_nextmonth_latest.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

ROOT = PROCESSED_DATA_DIR.parent.parent
MODELS_DIR = ROOT / "models"
WORK = INTERIM_DATA_DIR / "occurrence"
PANEL = WORK / "panel.parquet"
OOF = WORK / "oof.parquet"
LATEST_PATH = PROCESSED_DATA_DIR / "aoi_occurrence_nextmonth_latest.parquet"
MODEL_PATH = MODELS_DIR / "aoi_occurrence_xgb.json"
ISO_PATH = MODELS_DIR / "aoi_occurrence_calibration.json"
META_PATH = MODELS_DIR / "aoi_occurrence_meta.json"
FIRST = (2012, 2)
TARGET = (2026, 12)
TEST_YEARS = list(range(2016, 2026))
FEATURES = ["rate_all", "rate_5y", "n_years", "med_value", "lag1_value", "lag2_value", "lag3_value", "lastyear_value",
            "sido_rate", "fwi_prev", "fwi_prev_rel", "month_sin", "month_cos"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
          "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
          "lambda": 1.0, "seed": 42}


def t_of(y, m):
    return y * 12 + m - 1


def panel() -> pl.DataFrame:
    aoi = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_list.parquet").filter(pl.col("level") == "sgg", ~pl.col("hidden"))
    ids = aoi["aoi_id"].to_list()
    sido = dict(zip(aoi["aoi_id"], aoi["parent_aoi"]))
    t0, t1 = t_of(*FIRST) - 12, t_of(*TARGET)          # 특징용으로 한 해 앞부터 (lag)
    T = list(range(t0, t1 + 1))
    # 타깃: 점 위치 기준 시군구 감지 수 (2012-02 이후 기준 점 = VIIRS)
    pts = pl.read_parquet(INTERIM_DATA_DIR / "fire_events" / "points_admin_KR.parquet", columns=["date_local", "instrument", "sgg_cd_pt"])
    cnt = (pts.filter(pl.col("sgg_cd_pt").is_not_null(), pl.col("instrument") == "VIIRS")
           .group_by(aoi_id=pl.concat_str(pl.lit("SGG_"), pl.col("sgg_cd_pt")),
                     t=pl.col("date_local").dt.year().cast(pl.Int32) * 12 + pl.col("date_local").dt.month().cast(pl.Int32) - 1)
           .agg(n_det=pl.len()))
    st = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_month.parquet", columns=["aoi_id", "year", "month", "value", "value_projected", "status"])
    kr = st.filter(pl.col("aoi_id") == "KR").with_columns(t=pl.col("year").cast(pl.Int32) * 12 + pl.col("month").cast(pl.Int32) - 1)
    status = dict(zip(kr["t"], kr["status"]))
    sv = st.filter(pl.col("aoi_id").is_in(ids)).with_columns(
        t=pl.col("year").cast(pl.Int32) * 12 + pl.col("month").cast(pl.Int32) - 1,
        v=pl.when(pl.col("status") == "partial").then(pl.col("value_projected")).otherwise(pl.col("value")))
    na, nt = len(ids), len(T)
    ai = {a: i for i, a in enumerate(ids)}
    ti = {t: i for i, t in enumerate(T)}
    V = np.full((nt, na), np.nan)
    for a, t, v in sv.select("aoi_id", "t", "v").iter_rows():
        if t in ti:
            V[ti[t], ai[a]] = v
    V = np.nan_to_num(V)                              # aoi_month 행이 없으면 0
    N = np.zeros((nt, na))
    for a, t, n in cnt.iter_rows():
        if t in ti and a in ai:
            N[ti[t], ai[a]] = n
    valid = np.array([(t >= t_of(*FIRST)) and status.get(t) in ("ok", "nrt") for t in T])
    Y = np.where(valid[:, None], (N >= 1).astype(float), np.nan)
    # FWI 월평균 (시군구)
    fw = (pl.scan_parquet(PROCESSED_DATA_DIR / "fwi_aoi_day.parquet").filter(pl.col("aoi_id").is_in(ids))
          .group_by("aoi_id", t=pl.col("date").dt.year().cast(pl.Int32) * 12 + pl.col("date").dt.month().cast(pl.Int32) - 1)
          .agg(f=pl.col("fwi_mean").mean()).collect())
    F = np.full((nt, na), np.nan)
    for a, t, f in fw.iter_rows():
        if t in ti:
            F[ti[t], ai[a]] = f
    sido_idx = {}
    for a, i in ai.items():
        sido_idx.setdefault(sido[a], []).append(i)
    rows = []
    for k, t in enumerate(tqdm(T, desc="패널", unit="month", leave=False)):
        if t < t_of(*FIRST):
            continue
        y, m = t // 12, t % 12 + 1
        same = [ti[t - 12 * j] for j in range(1, 40) if (t - 12 * j) in ti and valid[ti[t - 12 * j]]]  # 이전 해 같은 달 (관측 구간)
        ys = Y[same] if same else np.empty((0, na))
        rate_all = ys.mean(axis=0) if len(same) else np.full(na, np.nan)
        rate_5y = Y[same[:5]].mean(axis=0) if len(same) else np.full(na, np.nan)
        med = np.median(V[same], axis=0) if len(same) else np.full(na, np.nan)
        srate = np.full(na, np.nan)
        if len(same):
            for s, idx in sido_idx.items():
                srate[idx] = rate_all[idx].mean()
        prevF = F[k - 1]
        hist = [F[ti[t - 1 - 12 * j]] for j in range(1, 40) if (t - 1 - 12 * j) in ti]
        frel = np.full(na, np.nan)
        if len(hist) >= 3:
            H = np.vstack(hist)
            nh = np.sum(~np.isnan(H), axis=0)
            with np.errstate(invalid="ignore"):
                less = np.nansum(H < prevF, axis=0) + 0.5 * np.nansum(H == prevF, axis=0)
            frel = np.where((nh >= 3) & ~np.isnan(prevF), 100 * less / np.maximum(nh, 1), np.nan)
        rows.append(pl.DataFrame({
            "aoi_id": ids, "year": np.full(na, y, np.int16), "month": np.full(na, m, np.int8), "label": Y[k],
            "n_det": N[k], "rate_all": rate_all, "rate_5y": rate_5y, "n_years": np.full(na, len(same), np.float64),
            "med_value": med, "lag1_value": V[k - 1], "lag2_value": V[k - 2], "lag3_value": V[k - 3],
            "lastyear_value": V[k - 12], "lastyear_any": Y[k - 12] if valid[k - 12] else np.full(na, np.nan),
            "sido_rate": srate, "fwi_prev": prevF, "fwi_prev_rel": frel,
            "month_sin": np.full(na, np.sin(2 * np.pi * m / 12)), "month_cos": np.full(na, np.cos(2 * np.pi * m / 12)),
            "lag1_projected": np.full(na, status.get(t - 1) == "partial"),
        }))
    p = pl.concat(rows).fill_nan(None)
    WORK.mkdir(parents=True, exist_ok=True)
    p.write_parquet(PANEL)
    lab = p.filter(pl.col("label").is_not_null())
    print(f"[완료] panel: {p.height:,}행, 라벨 {lab.height:,}행 ({lab['year'].min()}~{lab['year'].max()}), 양성 비율 {lab['label'].mean():.3f}, "
          f"최대 메모리 {peak_memory_gb() or 0:.2f} GB")
    return p


def _dm(df, label=True):
    import xgboost as xgb
    return xgb.DMatrix(df.select(FEATURES).to_numpy().astype(np.float32), label=df["label"].to_numpy() if label else None,
                       feature_names=FEATURES, missing=np.nan)


def fit_fold(tr: pl.DataFrame, va: pl.DataFrame):
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    bst = xgb.train(PARAMS, _dm(tr), 2000, evals=[(_dm(va), "va")], early_stopping_rounds=50, verbose_eval=False)
    raw_va = bst.predict(_dm(va, False), iteration_range=(0, bst.best_iteration + 1))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_va, va["label"].to_numpy())
    return bst, iso, Platt().fit(raw_va, va["label"].to_numpy())


class Platt:
    """Platt 보정: logit(p_raw)에 1차 로지스틱 회귀."""
    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1e6).fit(self._x(p), y)
        return self

    @staticmethod
    def _x(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, p):
        return self.lr.predict_proba(self._x(p))[:, 1]

    def params(self):
        return {"a": float(self.lr.coef_[0, 0]), "b": float(self.lr.intercept_[0])}


def baseline1(df: pl.DataFrame, prior: float) -> pl.Expr:
    return pl.col("rate_all").fill_null(prior)


def validate() -> pl.DataFrame:
    p = pl.read_parquet(PANEL).filter(pl.col("label").is_not_null())
    outs, log = [], []
    for Y in tqdm(TEST_YEARS, desc="확장 창", unit="fold"):
        tr, va, te = p.filter(pl.col("year") <= Y - 2), p.filter(pl.col("year") == Y - 1), p.filter(pl.col("year") == Y)
        bst, iso, platt = fit_fold(tr, va)
        raw = bst.predict(_dm(te, False), iteration_range=(0, bst.best_iteration + 1))
        prior = float(p.filter(pl.col("year") < Y)["label"].mean())
        outs.append(te.select("aoi_id", "year", "month", "label").with_columns(
            p_raw=pl.Series(raw), p_model=pl.Series(iso.predict(raw)), p_platt=pl.Series(platt.predict(raw)),
            p_base_rate=te.select(baseline1(te, prior))["rate_all"], p_base_lastyear=te["lastyear_any"].fill_null(0.0)))
        log.append({"test_year": Y, "rounds": bst.best_iteration + 1, "train_rows": tr.height})
    oof = pl.concat(outs)
    oof.write_parquet(OOF)
    (WORK / "folds.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"[완료] oof {oof.height:,}행, folds {[f['rounds'] for f in log]}")
    return oof


def _m(y, s):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "brier": float(brier_score_loss(y, np.clip(s, 0, 1)))}


METHODS = {"p_model": "xgboost_isotonic", "p_platt": "xgboost_platt", "p_raw": "xgboost_raw", "p_base_rate": "baseline_same_month_rate",
           "p_base_lastyear": "baseline_last_year_same_month"}


def report() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    oof = pl.read_parquet(OOF)
    y = oof["label"].to_numpy()
    rows = [{"method": n, "rows": oof.height, "pos_rate": float(y.mean()), **_m(y, oof[c].to_numpy())} for c, n in METHODS.items()]
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "occurrence_metrics.csv")
    by = []
    for (yr,), g in oof.group_by("year"):
        for c, n in METHODS.items():
            by.append({"test_year": yr, "method": n, "pos_rate": float(g["label"].mean()), **_m(g["label"].to_numpy(), g[c].to_numpy())})
    by = pl.DataFrame(by).sort("test_year", "method")
    by.write_csv(REPORTS_DIR / "occurrence_metrics_by_year.csv")
    # AOI × 달력 월 양성 비율 극단 (2012-02 ~ 관측 구간 전체)
    p = pl.read_parquet(PANEL).filter(pl.col("label").is_not_null())
    am = p.group_by("aoi_id", "month").agg(rate=pl.col("label").mean(), n=pl.len())
    ext = pl.DataFrame([{"share_gt95": float((am["rate"] > 0.95).mean()), "share_lt05": float((am["rate"] < 0.05).mean()),
                         "share_eq0": float((am["rate"] == 0).mean()), "share_eq1": float((am["rate"] == 1).mean()),
                         "aoi_months": am.height, "overall_pos_rate": float(p["label"].mean())}])
    ext.write_csv(REPORTS_DIR / "occurrence_base_rate_extremes.csv")
    # 보정 곡선
    fig, ax = plt.subplots(figsize=(5.4, 5))
    bins = np.linspace(0, 1, 11)
    for c, lab, col in (("p_platt", "XGBoost + Platt", "#2a6fdb"), ("p_model", "XGBoost + isotonic", "#7a5cd6"), ("p_raw", "XGBoost (raw)", "#9bb7e8"),
                        ("p_base_rate", "Baseline: same-month rate", "#d9822b")):
        s = oof[c].to_numpy()
        idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
        xs, ys_ = [], []
        for b in range(10):
            k = idx == b
            if k.sum() >= 30:
                xs.append(s[k].mean()); ys_.append(y[k].mean())
        ax.plot(xs, ys_, "o-", color=col, label=lab)
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed share with ≥ 1 detection")
    ax.set_title("Korea municipalities (252), next month, test 2016–2025"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "figures" / "occurrence_calibration.png", dpi=150); plt.close(fig)
    wide = by.pivot(on="method", index="test_year", values=["pr_auc", "brier"])
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=4):
        print(m); print(ext)
        print(by.pivot(on="method", index="test_year", values="pr_auc"))
    mm = {r["method"]: r for r in m.to_dicts()}
    best_base_pr = max(mm["baseline_same_month_rate"]["pr_auc"], mm["baseline_last_year_same_month"]["pr_auc"])
    best_base_br = min(mm["baseline_same_month_rate"]["brier"], mm["baseline_last_year_same_month"]["brier"])
    # 보정 방법: 두 보정 중 Brier가 낮은 쪽
    cal = "platt" if mm["xgboost_platt"]["brier"] <= mm["xgboost_isotonic"]["brier"] else "isotonic"
    best = mm[f"xgboost_{cal}"]
    adopt = best["pr_auc"] > best_base_pr and best["brier"] < best_base_br
    print(f"[채택 판정] xgboost_{cal} PR-AUC {best['pr_auc']:.4f} vs 기준선 최고 {best_base_pr:.4f}, "
          f"Brier {best['brier']:.4f} vs 기준선 최저 {best_base_br:.4f} → {'채택' if adopt else '기준선 사용'}")
    return {"adopt": adopt, "calibration": cal, "metrics": m.to_dicts(), "extremes": ext.to_dicts()[0]}


IND_OCC, IND_CV, IND_MIN = 0.25, 0.40, 0.10


def industrial_flags(p: pl.DataFrame | None = None) -> pl.DataFrame:
    """산업 열원 의심 시군구: 2012-02 ~ 관측 구간에서 발생 달 비율 ≥ 0.25 이고 계절성이 거의 없음
    (12개 달력 월 발생 비율의 변동계수 ≤ 0.40, 가장 낮은 달도 ≥ 0.10). 한국 전체 시군구의 같은 변동계수는 0.48."""
    p = (p if p is not None else pl.read_parquet(PANEL)).filter(pl.col("label").is_not_null())
    cal = p.group_by("aoi_id", "month").agg(r=pl.col("label").mean())
    s = cal.group_by("aoi_id").agg(month_rate_min=pl.col("r").min(), seasonal_cv=pl.col("r").std() / pl.col("r").mean()).join(
        p.group_by("aoi_id").agg(occ_share=pl.col("label").mean()), on="aoi_id")
    return s.with_columns(suspected_industrial=(pl.col("occ_share") >= IND_OCC) & (pl.col("seasonal_cv") <= IND_CV)
                          & (pl.col("month_rate_min") >= IND_MIN))


def latest(adopt: bool | None = None, cal: str | None = None) -> pl.DataFrame:
    """대상 2026-12: 시험 해 2026과 같은 구조 (학습 ≤ 2024, 2025로 조기 종료·보정)."""
    p = pl.read_parquet(PANEL)
    lab = p.filter(pl.col("label").is_not_null())
    bst, iso, platt = fit_fold(lab.filter(pl.col("year") <= 2024), lab.filter(pl.col("year") == 2025))
    if cal is None:
        cal = json.loads(META_PATH.read_text(encoding="utf-8")).get("calibration", "platt") if META_PATH.exists() else "platt"
    MODELS_DIR.mkdir(exist_ok=True)
    bst.save_model(MODEL_PATH)
    ISO_PATH.write_text(json.dumps({"method": cal, "platt": platt.params(),
                                    "isotonic": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}}), encoding="utf-8")
    te = p.filter(pl.col("year") == TARGET[0], pl.col("month") == TARGET[1])
    raw = bst.predict(_dm(te, False), iteration_range=(0, bst.best_iteration + 1))
    pm = platt.predict(raw) if cal == "platt" else iso.predict(raw)
    prior = float(lab["label"].mean())
    base = te.select(pl.col("rate_all").fill_null(prior))["rate_all"].to_numpy()
    if adopt is None:
        adopt = json.loads(META_PATH.read_text(encoding="utf-8"))["adopted"] if META_PATH.exists() else True
    st = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_month.parquet", columns=["aoi_id", "year", "month", "as_of_date"]).filter(
        pl.col("aoi_id") == "KR", pl.col("year") == 2026, pl.col("month") == 11)
    as_of = st["as_of_date"][0]
    p_any = pm if adopt else base
    out = te.select("aoi_id", "rate_all", "rate_5y", "n_years", "lag1_value", "lag1_projected").with_columns(
        target_month=pl.lit(f"{TARGET[0]}-{TARGET[1]:02d}"), p_any=pl.Series(p_any.astype(np.float32)),
        baseline_p=pl.Series(base.astype(np.float32)), p_model=pl.Series(pm.astype(np.float32)),
        model=pl.lit(f"aoi_occurrence_xgb+{cal}" if adopt else "baseline_same_month_rate"), as_of_date=pl.lit(as_of))
    ind = industrial_flags(p)
    out = out.with_columns(p_none=(1 - pl.col("p_any")).cast(pl.Float32)).join(
        ind.select("aoi_id", "suspected_industrial", "occ_share", "seasonal_cv"), on="aoi_id", how="left").with_columns(
        pl.col("suspected_industrial").fill_null(False)).select(
        "aoi_id", "target_month", "p_any", "p_none", "baseline_p", "model", "as_of_date", "suspected_industrial", "p_model", "rate_all",
        "rate_5y", "n_years", "lag1_value", "lag1_projected", "occ_share", "seasonal_cv").sort("p_any", descending=True)
    ind.write_csv(REPORTS_DIR / "occurrence_industrial_flags.csv")
    out.write_parquet(LATEST_PATH)
    names = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_list.parquet", columns=["aoi_id", "name"])
    with pl.Config(tbl_rows=12, float_precision=3):
        print(f"[완료] {LATEST_PATH.name}: {out.height}행, 대상 {TARGET}, as_of {as_of}, lag1(11월) value_projected 사용")
        print(out.join(names, on="aoi_id").sort("p_any", descending=True).select("aoi_id", "name", "p_any", "baseline_p", "rate_all").head(12))
        print(out.select(mean_p_any=pl.col("p_any").mean(), mean_base=pl.col("baseline_p").mean(),
                         n_ge50=(pl.col("p_any") >= 0.5).sum(), expected_aoi=pl.col("p_any").sum()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["panel", "validate", "report", "latest", "all"])
    a = ap.parse_args(argv)
    start_run_log("occurrence")
    t0 = time.perf_counter()
    res = None
    if a.step in ("panel", "all"):
        panel()
    if a.step in ("validate", "all"):
        validate()
    if a.step in ("report", "all"):
        res = report()
        meta = {"name": "aoi_occurrence_nextmonth", "created": date.today().isoformat(), "unit": "Korea sgg (252, aoi_list level sgg, not hidden)",
                "target": "next month >= 1 VIIRS reference detection inside the municipality (point location; static sources excluded, v1.2)",
                "train_period": "2012-02 .. 2026-10 (KR status ok/nrt; month in progress excluded; MODIS-era m2v estimates not used as target)",
                "features": FEATURES, "params": PARAMS, "calibration": res["calibration"],
                "calibration_note": "fitted on year Y-1 predictions of a model trained on <= Y-2; the method with lower pooled Brier (Platt vs isotonic) is used",
                "validation": "expanding window 2016-2025", "adopted": res["adopt"],
                "adoption_rule": "PR-AUC higher and Brier lower than both baselines (pooled 2016-2025)",
                "metrics": res["metrics"], "base_rate_extremes": res["extremes"],
                "argentina": "not modelled: aoi_list has provinces only (no departamentos); use cell-level recurrence probability. Departamento level = future work.",
                "latest_note": "target 2026-12; lag1 (2026-11) uses value_projected (month in progress), flagged lag1_projected"}
        MODELS_DIR.mkdir(exist_ok=True)
        META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.step in ("latest", "all"):
        latest(res["adopt"] if res else None, res["calibration"] if res else None)
    print(f"[완료] {a.step} {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
