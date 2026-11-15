"""반복 연소 기반 '다음 달 불 발생 가능성' (실험, 2026-11-15) — 0.1° 칸, 한국(KR)·아르헨티나(ADM0_ARG).

라벨   칸·달 harmonized_cell_month.value ≥ 1 (VIIRS 기준 값; 2012년 이전은 MODIS→VIIRS 변환 값).
       행이 없으면 0. 그 나라 aoi_month.status가 ok가 아닌 달은 라벨에서 뺀다 (2001-06 missing, 2026-08~ nrt/partial).
특성   (대상 달 t 기준, t 이후 정보 없음; 예측 시점 = t-1 달 말)
  rec           같은 달 과거 반복률 (2001 ~ 그 해 직전 해, 유효한 해만)
  n_prior       반복률에 쓴 해 수
  act3, act5    최근 3·5년 같은 달 활성 횟수
  logv_mean     같은 달 과거 평균 log1p(value)
  nb3_rec, nb5_rec  주변 3×3·5×5 칸 같은 달 반복률 평균 (자기 칸 포함)
  lag{1,2,3}_act, lag{1,2,3}_anom  직전 1~3개월 활성 여부, log1p(value) − 그 달 과거 평균 log1p(value)
  fwi_prev, fwi_prev_rel  직전 달 FWI 평균(0.5° 지점), 그 값의 과거 같은 달 대비 백분위(그 해 이전 해들만, 3년 이상)
  month_sin, month_cos, lat, lon, is_kr
검증   확장 창: 2016~2025 각 해 Y를 Y-1년 이전 대상 달로만 학습 (Y-1년으로 조기 종료 → 같은 회수로 Y-1까지 다시 학습).
       기준선 ① rec(반복률) ② 작년 같은 달 활성 여부.

실행  uv run python -m planetarium.recurrence panel|validate|report|final|latest|sites|all
산출  data/interim/recurrence/panel.parquet, oof.parquet
      models/recurrence_nextmonth_xgb.json, models/recurrence_nextmonth_meta.json
      reports/recurrence_metrics.csv, recurrence_metrics_by_year.csv, recurrence_importance.csv, figures/recurrence_calibration.png
      data/processed/recurrence_nextmonth_latest.parquet, data/processed/recurrent_burn_sites.parquet
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
from planetarium.runlog import peak_memory_gb, start_run_log

ROOT = PROCESSED_DATA_DIR.parent.parent
MODELS_DIR = ROOT / "models"
FIG_DIR = REPORTS_DIR / "figures"
WORK = INTERIM_DATA_DIR / "recurrence"
PANEL_PATH = WORK / "panel.parquet"
OOF_PATH = WORK / "oof.parquet"
MODEL_PATH = MODELS_DIR / "recurrence_nextmonth_xgb.json"
META_PATH = MODELS_DIR / "recurrence_nextmonth_meta.json"
LATEST_PATH = PROCESSED_DATA_DIR / "recurrence_nextmonth_latest.parquet"
SITES_PATH = PROCESSED_DATA_DIR / "recurrent_burn_sites.parquet"

Y0, Y1 = 2001, 2026          # 자료 연도 (2026은 특성·최신 예측용)
TEST_YEARS = list(range(2016, 2026))
LATEST = (2026, 12)
SITE_RATE = 0.7
PAD = 3
REGIONS = {"KR": "KR", "ARG": "ADM0_ARG"}
FEATURES = ["rec", "n_prior", "act3", "act5", "logv_mean", "nb3_rec", "nb5_rec",
            "lag1_act", "lag2_act", "lag3_act", "lag1_anom", "lag2_anom", "lag3_anom",
            "fwi_prev", "fwi_prev_rel", "month_sin", "month_cos", "lat", "lon", "is_kr"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "device": "cuda", "tree_method": "hist",
          "max_depth": 6, "learning_rate": 0.08, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
          "lambda": 1.0, "max_bin": 256, "seed": 42}
MAX_ROUNDS, EARLY_STOP = 2000, 50


def static_cells() -> pl.DataFrame:
    """0.1° 칸 중 고정 열원(제철소·정유·가스 플레어 등, features.py의 0.01° 목록)이 하나라도 있는 칸."""
    s = pl.read_parquet(PROCESSED_DATA_DIR / "static_source_cells_0.01deg.parquet")
    return (s.with_columns(cell_id=((pl.col("cell_id") // 36000) // 10) * 3600 + (pl.col("cell_id") % 36000) // 10)
            .group_by("cell_id").agg(n_static_0p01=pl.len().cast(pl.Int16)))


def region_cells(region: str) -> pl.DataFrame:
    if region == "KR":
        w = pl.read_parquet(PROCESSED_DATA_DIR / "korea_cell_sgg_weights_0.1deg.parquet")
        names = (w.sort("weight", descending=True).group_by("cell_id", maintain_order=True)
                 .agg(sgg_cd=pl.col("sgg_cd").first(), sgg_nm=pl.col("sgg_nm").first(), sido_nm=pl.col("sido_nm").first()))
        return names
    w = pl.read_parquet(PROCESSED_DATA_DIR / "world_cell_adm1_weights_0.1deg.parquet").filter(pl.col("adm0_a3") == "ARG")
    return (w.sort("weight", descending=True).group_by("cell_id", maintain_order=True)
            .agg(adm1_nm=pl.col("adm1_name").first()))


def region_status(aoi_id: str) -> np.ndarray:
    """valid[t] (t = (year-Y0)*12 + month-1): 그 나라 aoi_month.status == ok."""
    nt = (Y1 - Y0 + 1) * 12
    a = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_month.parquet", columns=["aoi_id", "year", "month", "status"]).filter(
        pl.col("aoi_id") == aoi_id, pl.col("year").is_between(Y0, Y1))
    v = np.zeros(nt, bool)
    for y, m, s in a.select("year", "month", "status").iter_rows():
        v[(y - Y0) * 12 + m - 1] = s == "ok"
    return v


def fwi_monthly(region: str, iy0: int, ix0: int, ny: int, nx: int) -> np.ndarray:
    """F[t, r, c]: 칸이 속한 0.5° 지점의 그 달 평균 FWI (없으면 NaN)."""
    nt = (Y1 - Y0 + 1) * 12
    F = np.full((nt, ny, nx), np.nan, np.float32)
    f = (pl.scan_parquet(PROCESSED_DATA_DIR / "fwi_cell_day.parquet").filter(pl.col("region") == region)
         .group_by("glat", "glon", y=pl.col("date").dt.year(), m=pl.col("date").dt.month()).agg(fwi=pl.col("FWI").mean()).collect())
    if f.is_empty():
        return F
    # 0.5° 지점 → 그 안의 0.1° 칸 25개
    for glat, glon, y, m, v in f.iter_rows():
        if not (Y0 <= y <= Y1):
            continue
        t = (y - Y0) * 12 + m - 1
        r0 = int(round((glat - 0.25 + 90) * 10)) - iy0
        c0 = int(round((glon - 0.25 + 180) * 10)) - ix0
        rs, cs = slice(max(r0, 0), max(min(r0 + 5, ny), 0)), slice(max(c0, 0), max(min(c0 + 5, nx), 0))
        F[t, rs, cs] = v
    return F


def build_region(region: str) -> pl.DataFrame:
    aoi_id = REGIONS[region]
    cells = region_cells(region)
    iy, ix = (cells["cell_id"] // 3600).to_numpy(), (cells["cell_id"] % 3600).to_numpy()
    iy0, ix0 = iy.min() - PAD, ix.min() - PAD
    ny, nx = iy.max() + PAD + 1 - iy0, ix.max() + PAD + 1 - ix0
    nyr, nt = Y1 - Y0 + 1, (Y1 - Y0 + 1) * 12
    cm = (pl.scan_parquet(PROCESSED_DATA_DIR / "harmonized_cell_month.parquet")
          .with_columns(r=(pl.col("cell_id") // 3600 - iy0).cast(pl.Int32), c=(pl.col("cell_id") % 3600 - ix0).cast(pl.Int32))
          .filter(pl.col("year").is_between(Y0, Y1), pl.col("r").is_between(0, ny - 1), pl.col("c").is_between(0, nx - 1))
          .select("r", "c", "year", "month", "status",
                  v=pl.when(pl.col("status") == "partial").then(pl.col("value_projected")).otherwise(pl.col("value")))
          .collect())
    V = np.zeros((nt, ny, nx), np.float32)
    t_idx = ((cm["year"].to_numpy().astype(int) - Y0) * 12 + cm["month"].to_numpy().astype(int) - 1)
    V[t_idx, cm["r"].to_numpy(), cm["c"].to_numpy()] = cm["v"].to_numpy().astype(np.float32)
    valid = region_status(aoi_id)                    # 라벨용
    feat_ok = valid.copy()                           # 특성용: ok·nrt·partial 사용, missing·미래 달은 NaN
    st = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_month.parquet", columns=["aoi_id", "year", "month", "status"]).filter(
        pl.col("aoi_id") == aoi_id, pl.col("year").is_between(Y0, Y1))
    for y, m, s in st.select("year", "month", "status").iter_rows():
        feat_ok[(y - Y0) * 12 + m - 1] = s in ("ok", "nrt", "partial")
    V[~feat_ok] = np.nan
    A = np.where(np.isnan(V), np.nan, (V >= 1).astype(np.float32))
    L = np.log1p(V)

    # 같은 달·연도 축 [year, month, r, c]
    A4, L4 = A.reshape(nyr, 12, ny, nx), L.reshape(nyr, 12, ny, nx)
    ok4 = ~np.isnan(A4)
    cnt = np.cumsum(np.nan_to_num(A4), axis=0) - np.nan_to_num(A4)          # 그 해 이전 활성 해 수
    npr = np.cumsum(ok4, axis=0) - ok4                                        # 그 해 이전 유효 해 수
    lsum = np.cumsum(np.nan_to_num(L4), axis=0) - np.nan_to_num(L4)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(npr > 0, cnt / npr, np.nan).astype(np.float32)
        lmean = np.where(npr > 0, lsum / npr, np.nan).astype(np.float32)
    a0 = np.nan_to_num(A4)
    cs = np.concatenate([np.zeros_like(a0[:1]), np.cumsum(a0, axis=0)])
    act3 = cs[np.arange(nyr)] - cs[np.maximum(np.arange(nyr) - 3, 0)]
    act5 = cs[np.arange(nyr)] - cs[np.maximum(np.arange(nyr) - 5, 0)]
    nb = {}
    for k in (3, 5):
        box = uniform_filter(cnt.astype(np.float32), size=(1, 1, k, k), mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            nb[k] = np.where(npr > 0, box / npr, np.nan).astype(np.float32)

    # 시간 축으로 펴기
    flat = lambda x: x.reshape(nt, ny, nx)  # noqa: E731
    REC, LMEAN, ACT3, ACT5, NB3, NB5, NPR = map(flat, (rec, lmean, act3.astype(np.float32), act5.astype(np.float32), nb[3], nb[5],
                                                 npr.astype(np.float32)))
    ANOM = L - LMEAN
    F = fwi_monthly(region, iy0, ix0, ny, nx)
    F4 = F.reshape(nyr, 12, ny, nx)
    frel = np.full_like(F4, np.nan)
    for y in range(nyr):
        prev = F4[:y]
        n_prev = np.sum(~np.isnan(prev), axis=0)
        if y == 0:
            continue
        with np.errstate(invalid="ignore"):
            less = np.nansum(prev < F4[y][None], axis=0) + 0.5 * np.nansum(prev == F4[y][None], axis=0)
            frel[y] = np.where((n_prev >= 3) & ~np.isnan(F4[y]), 100 * less / np.maximum(n_prev, 1), np.nan)
    FREL = frel.reshape(nt, ny, nx)

    r, c = iy - iy0, ix - ix0
    parts = []
    t_first = 0   # 2001-01부터 (2001은 반복률이 없어 학습에서는 빼고, 반복 연소지 집계에만 씀)
    t_last = (LATEST[0] - Y0) * 12 + LATEST[1] - 1
    for t in tqdm(range(t_first, t_last + 1), desc=f"패널 {region}", unit="month", leave=False):
        y, m = Y0 + t // 12, t % 12 + 1
        lab = A[t, r, c] if (t < nt and valid[t]) else np.full(len(r), np.nan, np.float32)
        d = {"cell_id": cells["cell_id"].to_numpy(), "year": np.full(len(r), y, np.int16), "month": np.full(len(r), m, np.int8),
             "label": lab, "value": V[t, r, c] if valid[t] else np.full(len(r), np.nan, np.float32),
             "rec": REC[t, r, c], "n_prior": NPR[t, r, c], "act3": ACT3[t, r, c], "act5": ACT5[t, r, c], "logv_mean": LMEAN[t, r, c],
             "nb3_rec": NB3[t, r, c], "nb5_rec": NB5[t, r, c]}
        nan = np.full(len(r), np.nan, np.float32)
        for k in (1, 2, 3):
            d[f"lag{k}_act"] = A[t - k, r, c] if t - k >= 0 else nan
            d[f"lag{k}_anom"] = ANOM[t - k, r, c] if t - k >= 0 else nan
        d["fwi_prev"], d["fwi_prev_rel"] = (F[t - 1, r, c], FREL[t - 1, r, c]) if t >= 1 else (nan, nan)
        parts.append(pl.DataFrame(d))
    p = pl.concat(parts).fill_nan(None).with_columns(
        region=pl.lit(region), is_kr=pl.lit(1.0 if region == "KR" else 0.0, dtype=pl.Float32),
        lat=((pl.col("cell_id") // 3600 + 0.5) * 0.1 - 90).cast(pl.Float32),
        lon=((pl.col("cell_id") % 3600 + 0.5) * 0.1 - 180).cast(pl.Float32),
        month_sin=(2 * np.pi * pl.col("month").cast(pl.Float32) / 12).sin().cast(pl.Float32),
        month_cos=(2 * np.pi * pl.col("month").cast(pl.Float32) / 12).cos().cast(pl.Float32))
    return p


def panel() -> pl.DataFrame:
    WORK.mkdir(parents=True, exist_ok=True)
    p = pl.concat([build_region(r) for r in REGIONS])
    p.write_parquet(PANEL_PATH, compression="zstd")
    lab = p.filter(pl.col("label").is_not_null(), pl.col("year") >= 2002)
    print(f"[완료] {PANEL_PATH.name}: {p.height:,}행, 라벨 있는 행 {lab.height:,}, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=4, thousands_separator=","):
        print(lab.group_by("region").agg(cells=pl.col("cell_id").n_unique(), rows=pl.len(), pos_rate=pl.col("label").mean(),
                                         fwi_cover=pl.col("fwi_prev").is_not_null().mean(), y0=pl.col("year").min(),
                                         y1=pl.col("year").max()).sort("region"))
    return p


def _dmat(df: pl.DataFrame, label: bool = True):
    import xgboost as xgb
    X = df.select(FEATURES).to_numpy().astype(np.float32)
    return xgb.DMatrix(X, label=df["label"].to_numpy() if label else None, feature_names=FEATURES, missing=np.nan)


def _train(tr: pl.DataFrame, va: pl.DataFrame | None, rounds: int | None = None):
    import xgboost as xgb
    dtr = _dmat(tr)
    if va is not None:
        bst = xgb.train(PARAMS, dtr, MAX_ROUNDS, evals=[(_dmat(va), "va")], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        return bst, bst.best_iteration + 1
    return xgb.train(PARAMS, dtr, rounds), rounds


def validate() -> pl.DataFrame:
    p = pl.read_parquet(PANEL_PATH).filter(pl.col("label").is_not_null(), pl.col("year") >= 2002)
    outs, log = [], []
    for Y in tqdm(TEST_YEARS, desc="확장 창", unit="fold", dynamic_ncols=True):
        t0 = time.perf_counter()
        tr_in, va_in = p.filter(pl.col("year") < Y - 1), p.filter(pl.col("year") == Y - 1)
        _, best = _train(tr_in, va_in)
        bst, _ = _train(p.filter(pl.col("year") < Y), None, best)
        te = p.filter(pl.col("year") == Y)
        pred = bst.predict(_dmat(te, label=False))
        outs.append(te.select("region", "cell_id", "year", "month", "label", "rec", "act3")
                    .with_columns(p_model=pl.Series(pred.astype(np.float32))))
        log.append({"test_year": Y, "rounds": best, "train_rows": p.filter(pl.col("year") < Y).height, "sec": round(time.perf_counter() - t0, 1)})
        tqdm.write(f"[{Y}] rounds {best}, {log[-1]['sec']}초")
    oof = pl.concat(outs)
    # 기준선 ②: 작년 같은 달 활성 (패널의 같은 칸·달·전년 label)
    prev = pl.read_parquet(PANEL_PATH, columns=["region", "cell_id", "year", "month", "label"]).with_columns(
        year=pl.col("year") + 1).rename({"label": "p_lastyear"})
    oof = oof.join(prev, on=["region", "cell_id", "year", "month"], how="left").with_columns(pl.col("p_lastyear").fill_null(0.0))
    oof.write_parquet(OOF_PATH, compression="zstd")
    (WORK / "folds.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"[완료] {OOF_PATH.name}: {oof.height:,}행, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")
    return oof


def _metrics(g: pl.DataFrame, col: str) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y, s = g["label"].to_numpy(), g[col].fill_null(0).to_numpy()
    if y.min() == y.max():
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": float(brier_score_loss(y, np.clip(s, 0, 1)))}
    return {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "brier": float(brier_score_loss(y, np.clip(s, 0, 1)))}


METHODS = {"p_model": "xgboost", "rec": "baseline1_recurrence_rate", "p_lastyear": "baseline2_last_year_same_month"}


def report() -> pl.DataFrame:
    oof = pl.read_parquet(OOF_PATH)
    rows, rows_y = [], []
    for (region,), g in oof.group_by("region"):
        for col, name in METHODS.items():
            rows.append({"region": region, "method": name, "rows": g.height, "pos_rate": float(g["label"].mean()), **_metrics(g, col)})
            for (y,), gy in g.group_by("year"):
                rows_y.append({"region": region, "method": name, "test_year": y, "pos_rate": float(gy["label"].mean()), **_metrics(gy, col)})
    rows_m = []
    for (region, mo), g in oof.group_by("region", "month"):
        for col, name in METHODS.items():
            rows_m.append({"region": region, "month": mo, "method": name, "pos_rate": float(g["label"].mean()), **_metrics(g, col)})
    mm = pl.DataFrame(rows_m).sort("region", "month", "method")
    mm.write_csv(REPORTS_DIR / "recurrence_metrics_by_month.csv")
    m = pl.DataFrame(rows).sort("region", "method")
    my = pl.DataFrame(rows_y).sort("region", "method", "test_year")
    m.write_csv(REPORTS_DIR / "recurrence_metrics.csv")
    my.write_csv(REPORTS_DIR / "recurrence_metrics_by_year.csv")
    # 해마다 모델이 기준선 ①보다 PR-AUC가 높은가
    w = my.pivot(on="method", index=["region", "test_year"], values="pr_auc").with_columns(
        gain=pl.col("xgboost") - pl.col("baseline1_recurrence_rate"))
    wins = w.group_by("region").agg(years_better=(pl.col("gain") > 0).sum(), n_years=pl.len(), gain_min=pl.col("gain").min(),
                                    gain_median=pl.col("gain").median())
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=4, thousands_separator=","):
        print(m)
        print(wins.sort("region"))
    calibration(oof)
    return m


def calibration(oof: pl.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, region in zip(axes, ["KR", "ARG"]):
        g = oof.filter(pl.col("region") == region)
        y = g["label"].to_numpy()
        for col, lab, color in (("p_model", "XGBoost", "#2a6fdb"), ("rec", "Baseline 1: recurrence rate", "#d9822b")):
            s = g[col].fill_null(0).to_numpy()
            idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
            xs, ys, ns = [], [], []
            for b in range(10):
                k = idx == b
                if k.sum() >= 50:
                    xs.append(s[k].mean()); ys.append(y[k].mean()); ns.append(k.sum())
            ax.plot(xs, ys, "o-", color=color, label=lab)
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        ax.set_title(f"{region} — test years 2016–2025 (expanding window)")
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed share active next month")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Next-month fire activity (0.1° cell, VIIRS-equivalent value ≥ 1): calibration")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "recurrence_calibration.png", dpi=150)
    plt.close(fig)
    print(f"[그림] {FIG_DIR / 'recurrence_calibration.png'}")


def final() -> None:
    p = pl.read_parquet(PANEL_PATH).filter(pl.col("label").is_not_null(), pl.col("year") >= 2002)
    folds = json.loads((WORK / "folds.json").read_text(encoding="utf-8"))
    rounds = int(np.median([f["rounds"] for f in folds]))
    bst, _ = _train(p, None, rounds)
    MODELS_DIR.mkdir(exist_ok=True)
    bst.save_model(MODEL_PATH)
    imp = bst.get_score(importance_type="gain")
    pl.DataFrame({"feature": list(imp), "gain": list(imp.values())}).sort("gain", descending=True).write_csv(REPORTS_DIR / "recurrence_importance.csv")
    m = pl.read_csv(REPORTS_DIR / "recurrence_metrics.csv")
    meta = {
        "name": "recurrence_nextmonth_xgb", "status": "experiment (2026-11-15) — adoption decided by data lead",
        "created": date.today().isoformat(), "target": "P(next month 0.1° cell harmonized value >= 1), VIIRS-equivalent",
        "display": "과거 반복 패턴 기반 전망(예측 정확도 표시) — not a weather/physics forecast",
        "regions": {"KR": "korea_cell_sgg_weights cells (1,123)", "ARG": "Natural Earth ADM0 ARG cells (28,589)"},
        "label_rule": "label only where aoi_month.status == ok for that country-month; no row in harmonized_cell_month = 0",
        "data_years": [Y0, 2025], "pre_2012_note": "2001–2011 values are MODIS→VIIRS model values (m2v), not VIIRS observations",
        "features": FEATURES,
        "feature_notes": {
            "rec": "same calendar month, all valid years from 2001 to the year before the target",
            "lag_k": "k months before the target; partial current month uses value_projected (month-to-date projection)",
            "fwi_prev_rel": "percentile of previous month mean FWI among the same month of earlier years only (>=3 years); NaN before",
            "fwi_coverage": "FWI exists only at NASA POWER points used by the spread nowcast (0.5°); other cells NaN",
        },
        "params": PARAMS, "rounds": rounds, "validation": "expanding window, test years 2016–2025, inner early stopping on Y-1",
        "fold_rounds": folds, "metrics": m.to_dicts(),
        "baselines": {"baseline1": "recurrence rate (rec)", "baseline2": "same month last year active (0/1)"},
        "caveats": ["Fire is often human-set (agricultural burning); recurrence reflects practice, not hazard.",
                    "Cells with no fire history get low probability by design; new fire locations are not predictable here.",
                    "Latest 2026-12 uses 2026-11 month-to-date (partial) and 2026-08..10 NRT months as lag features."],
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[완료] {MODEL_PATH.name} ({rounds} rounds), {META_PATH.name}")


def latest() -> pl.DataFrame:
    import xgboost as xgb
    p = pl.read_parquet(PANEL_PATH).filter(pl.col("year") == LATEST[0], pl.col("month") == LATEST[1])
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)
    pred = bst.predict(_dmat(p, label=False))
    out = p.select("region", "cell_id", "lat", "lon", "year", "month", "rec", "n_prior", "act3", "lag1_act", "fwi_prev").with_columns(
        p_active=pl.Series(pred.astype(np.float32)), model=pl.lit("recurrence_nextmonth_xgb"),
        lag1_note=pl.lit("2026-11 month-to-date (partial)"))
    out = out.rename({"lat": "cell_lat", "lon": "cell_lon", "act3": "active_years_last3"}).join(static_cells(), on="cell_id", how="left")
    out = out.with_columns(has_static_source=pl.col("n_static_0p01").is_not_null()).drop("n_static_0p01")
    # 채택 (2026-11-15): 아르헨티나 = 모델 확률, 한국 = 반복 횟수 레이어(확률 아님). 고정 열원 칸은 둘 다 뺀다.
    n_static = out.filter(pl.col("has_static_source")).group_by("region").len()
    out = out.filter(~pl.col("has_static_source")).with_columns(
        layer=pl.when(pl.col("region") == "ARG").then(pl.lit("probability")).otherwise(pl.lit("recurrence_count")),
        active_years=(pl.col("rec") * pl.col("n_prior")).round(0).cast(pl.Int16),
        n_years=pl.col("n_prior").cast(pl.Int16),
        p_active=pl.when(pl.col("region") == "ARG").then(pl.col("p_active")),
        model=pl.when(pl.col("region") == "ARG").then(pl.col("model"))).sort(
        "region", "p_active", "active_years", descending=[False, True, True], nulls_last=True)
    print(f"[고정 열원 칸 제외] {n_static.rows()}")
    out.write_parquet(LATEST_PATH, compression="zstd")
    with pl.Config(tbl_rows=-1, float_precision=3):
        print(f"[완료] {LATEST_PATH.name}: {out.height:,}행 (대상 {LATEST[0]}-{LATEST[1]:02d})")
        print(out.group_by("region", "layer").agg(n=pl.len(), p_mean=pl.col("p_active").mean(), n_ge40=(pl.col("p_active") >= 0.4).sum(),
                                                  n_active_ge10y=(pl.col("active_years") >= 10).sum()))
    return out


def sites() -> pl.DataFrame:
    p = pl.read_parquet(PANEL_PATH, columns=["region", "cell_id", "year", "month", "label", "value"]).filter(
        pl.col("label").is_not_null(), pl.col("year") <= 2025)
    s = p.group_by("region", "cell_id", "month").agg(
        n_years=pl.len().cast(pl.Int16), n_active=pl.col("label").sum().cast(pl.Int16),
        mean_value=pl.col("value").mean().cast(pl.Float32), first_year=pl.col("year").min(), last_year=pl.col("year").max()
    ).with_columns(rate=(pl.col("n_active") / pl.col("n_years")).cast(pl.Float32)).filter(pl.col("rate") >= SITE_RATE)
    s = s.with_columns(cell_lat=((pl.col("cell_id") // 3600 + 0.5) * 0.1 - 90).cast(pl.Float32),
                       cell_lon=((pl.col("cell_id") % 3600 + 0.5) * 0.1 - 180).cast(pl.Float32))
    kr = region_cells("KR").select("cell_id", "sido_nm", "sgg_nm")
    arg = region_cells("ARG").select("cell_id", "adm1_nm")
    s = s.join(kr, on="cell_id", how="left").join(arg, on="cell_id", how="left").with_columns(
        admin_name=pl.coalesce(pl.concat_str("sido_nm", "sgg_nm", separator=" "), pl.col("adm1_nm"))).drop("sido_nm", "sgg_nm", "adm1_nm")
    s = s.join(static_cells(), on="cell_id", how="left").with_columns(
        has_static_source=pl.col("n_static_0p01").is_not_null()).drop("n_static_0p01")
    print(f"[고정 열원 칸 제외] {s.filter(pl.col('has_static_source')).group_by('region').len().rows()}")
    s = s.filter(~pl.col("has_static_source"))
    s = s.sort("region", "rate", "mean_value", descending=[False, True, True])
    s.write_parquet(SITES_PATH, compression="zstd")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=2):
        print(f"[완료] {SITES_PATH.name}: {s.height:,}행 (칸·달), 기간 2001~2025 유효한 해, 기준 {SITE_RATE:.0%}")
        print(s.group_by("region").agg(rows=pl.len(), cells=pl.col("cell_id").n_unique()))
        print(s.filter(pl.col("region") == "KR").head(20).select("admin_name", "cell_lat", "cell_lon", "month", "n_active", "n_years", "rate",
                                                               "mean_value", "has_static_source"))
    s.filter(pl.col("region") == "KR").head(20).write_csv(REPORTS_DIR / "recurrence_sites_kr_top20.csv")
    return s


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="반복 연소 기반 다음 달 불 발생 가능성 (실험)")
    ap.add_argument("step", choices=["panel", "validate", "report", "final", "latest", "sites", "all"])
    args = ap.parse_args(argv)
    start_run_log("recurrence")
    t0 = time.perf_counter()
    steps = ["panel", "validate", "report", "final", "latest", "sites"] if args.step == "all" else [args.step]
    for s in steps:
        globals()[s]()
    print(f"[완료] {args.step} {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
