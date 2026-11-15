"""TimesFM 3.0 vs 우리 XGBoost vs 기준선 — 1·2·3개월 뒤 발생 확률 비교 (같은 정답·같은 시험 기간).
TimesFM: 분위수(log1p 척도) → P(value ≥ 1) = 1 − F(log1p 1), F는 분위수 사이 선형 보간(양 끝은 남은 확률의 절반).
         원값(raw)과, 바로 전 해(Y−1) 예측으로 맞춘 Platt 보정(cal; 우리 모델과 같은 방식) 두 가지.
출력: reports/timesfm_compare_metrics.csv, timesfm_compare_by_horizon.csv, timesfm_compare_mae.csv, figures/timesfm_compare_calibration.png
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

from planetarium.config import INTERIM_DATA_DIR, REPORTS_DIR
from planetarium.occurrence import Platt

D = INTERIM_DATA_DIR / "timesfm_compare"
THR = float(np.log1p(1.0))
TEST = {"grid": (2020, 2025), "sgg": (2016, 2025)}
OUR = {"grid": ("p_iso", INTERIM_DATA_DIR / "grid_occurrence" / "oof.parquet"), "sgg": ("p_platt", INTERIM_DATA_DIR / "occurrence" / "oof.parquet")}
KEY = {"grid": "g", "sgg": "aoi_id"}


def p_exceed(q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """q[..., j] 정렬된 분위수 → P(Y ≥ THR)."""
    L = np.r_[levels[0] / 2, levels, 1 - (1 - levels[-1]) / 2]
    lo = np.minimum(q[..., :1], 0.0) - 1e-6
    hi = q[..., -1:] + np.maximum(q[..., -1:] - q[..., :1], 1e-3)
    Q = np.concatenate([lo, q, hi], axis=-1).astype(np.float64)
    F = np.full(q.shape[:-1], L[-1])
    F = np.where(THR <= Q[..., 0], L[0], F)
    for j in range(Q.shape[-1] - 1):
        a, b = Q[..., j], Q[..., j + 1]
        m = (THR > a) & (THR <= b)
        F = np.where(m, L[j] + (THR - a) / np.maximum(b - a, 1e-9) * (L[j + 1] - L[j]), F)
    return 1.0 - F


def tfm_frame(task: str, mode: str) -> pl.DataFrame:
    s = np.load(D / f"series_{task}.npz")
    z = np.load(D / f"tfm_{task}_{mode}.npz")
    q, lv, org = z["q"].astype(np.float32), z["levels"], z["origins"]
    ids, V, T = s["ids"], s["V"], s["t"]
    med_j = int(np.argmin(np.abs(lv - 0.5)))
    rows = []
    for k, o in enumerate(org):
        for h in (1, 2, 3):
            t = o + h
            if t > T[-1]:
                continue
            rows.append(pl.DataFrame({KEY[task]: ids, "year": np.full(len(ids), t // 12, np.int16), "month": np.full(len(ids), t % 12 + 1, np.int8),
                                      "h": np.full(len(ids), h, np.int8), f"p_tfm_{mode}": p_exceed(q[k, :, h - 1, :], lv).astype(np.float32),
                                      f"med_tfm_{mode}": np.expm1(np.minimum(q[k, :, h - 1, med_j], 20.0)).astype(np.float32),   # log1p 20 이상(극단 분위수)은 잘라 넘침 방지
                                      "value": np.expm1(V[int(t - T[0])]).astype(np.float32)}))
    return pl.concat(rows)


def calibrate(df: pl.DataFrame, cols: list[str], years: tuple[int, int]) -> pl.DataFrame:
    """해 Y의 예측을 해 Y−1 예측·정답으로 맞춘 Platt로 보정 (h마다 따로). 시험 기간(years) 행만 돌려준다."""
    out = []
    for (h,), g in df.group_by("h"):
        for y in range(years[0], years[1] + 1):
            tr, te = g.filter(pl.col("year") == y - 1), g.filter(pl.col("year") == y)
            new = []
            for col in cols:
                pl_ = Platt().fit(tr[col].to_numpy(), tr["label"].to_numpy())
                new.append(pl.Series(col.replace("p_tfm", "p_tfmcal"), pl_.predict(te[col].to_numpy()).astype(np.float32)))
            out.append(te.with_columns(new))
    return pl.concat(out)


def metrics(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss
    if len(y) == 0 or y.min() == y.max():
        return {"pr_auc": None, "brier": None}
    return {"pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, np.clip(p, 0, 1)))}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    all_rows, mae_rows, frames = [], [], {}
    for task in ("grid", "sgg"):
        y0, y1 = TEST[task]
        k = KEY[task]
        ours = []
        for h in (1, 2, 3):
            col, p = OUR[task]
            if h > 1:
                p = D / f"xgb_h{h}" / task / "oof.parquet"
            o = pl.read_parquet(p).select(k, "year", "month", "label", p_xgb=col, p_base=pl.col("p_base_rate")).with_columns(h=pl.lit(h, dtype=pl.Int8))
            ours.append(o)
        ours = pl.concat(ours)
        tf = tfm_frame(task, "ctx36").join(tfm_frame(task, "full").select(k, "year", "month", "h", "p_tfm_full", "med_tfm_full"),
                                           on=[k, "year", "month", "h"])
        tf = tf.with_columns(pl.col("year").cast(pl.Int16), pl.col("month").cast(pl.Int8))
        lab = ours.filter(pl.col("h") == 1).select(k, "year", "month", "label")
        tfl = tf.join(lab, on=[k, "year", "month"])  # 정답이 있는 달만 (상태 ok·nrt)
        # 보정용으로 Y0−1 예측이 필요: 정답은 패널에서
        if task == "grid":
            from planetarium.grid_occurrence import PANEL as GP
            labp = pl.read_parquet(GP, columns=["g", "year", "month", "label"]).filter(pl.col("label").is_not_null())
        else:
            from planetarium.occurrence import PANEL as SP
            labp = pl.read_parquet(SP, columns=["aoi_id", "year", "month", "label"]).filter(pl.col("label").is_not_null())
        tfl = tf.join(labp.with_columns(pl.col("year").cast(pl.Int16), pl.col("month").cast(pl.Int8)), on=[k, "year", "month"])
        tfl = calibrate(tfl, ["p_tfm_ctx36", "p_tfm_full"], (y0, y1))
        d = ours.join(tfl.drop("label"), on=[k, "year", "month", "h"], how="inner")
        if task == "grid":
            cells = pl.read_parquet(INTERIM_DATA_DIR / "grid_occurrence" / "cells.parquet").select("g", "continent", "adm0", "persistent_source")
            d = d.join(cells, on="g")
            groups = [("ALL (excl. persistent)", ~pl.col("persistent_source")), ("ALL", pl.lit(True))] + \
                     [(c, (pl.col("continent") == c) & ~pl.col("persistent_source")) for c in sorted(cells["continent"].unique().to_list())] + \
                     [("Korea cells", (pl.col("adm0") == "KOR") & ~pl.col("persistent_source")),
                      ("Argentina cells", (pl.col("adm0") == "ARG") & ~pl.col("persistent_source"))]
        else:
            groups = [("Korea municipalities", pl.lit(True))]
        frames[task] = d
        methods = {"p_xgb": "XGBoost (ours)", "p_base": "Baseline: same-month rate", "p_tfm_ctx36": "TimesFM ctx36 raw",
                   "p_tfmcal_ctx36": "TimesFM ctx36 + Platt", "p_tfm_full": "TimesFM full raw", "p_tfmcal_full": "TimesFM full + Platt"}
        for gname, f in groups:
            for h in (1, 2, 3):
                g = d.filter(f, pl.col("h") == h)
                y = g["label"].to_numpy()
                for c, mname in methods.items():
                    all_rows.append({"task": task, "group": gname, "horizon_months": h, "method": mname, "rows": g.height,
                                     "pos_rate": float(y.mean()) if len(y) else None, **metrics(y, g[c].to_numpy())})
        # MAE (값 자체): TimesFM 중앙값 vs 같은 달 과거 중앙값 (패널 med_value)
        if task == "grid":
            med = pl.read_parquet(GP, columns=["g", "year", "month", "med_value"])
        else:
            med = pl.read_parquet(SP, columns=["aoi_id", "year", "month", "med_value"])
        dm = d.join(med.with_columns(pl.col("year").cast(pl.Int16), pl.col("month").cast(pl.Int8)), on=[k, "year", "month"], how="left")
        if task == "grid":
            dm = dm.filter(~pl.col("persistent_source"))
        for h in (1, 2, 3):
            g = dm.filter(pl.col("h") == h)
            mae_rows.append({"task": task, "horizon_months": h, "rows": g.height,
                             "mae_timesfm_ctx36": float((g["med_tfm_ctx36"] - g["value"]).abs().mean()),
                             "mae_timesfm_full": float((g["med_tfm_full"] - g["value"]).abs().mean()),
                             "mae_same_month_median": float((g["med_value"].fill_null(0) - g["value"]).abs().mean()),
                             # 극단값에 덜 흔들리는 지표: 로그 척도 MAE, 건수 오차의 중앙값
                             "mae_log1p_timesfm_ctx36": float((g["med_tfm_ctx36"].log1p() - g["value"].log1p()).abs().mean()),
                             "mae_log1p_timesfm_full": float((g["med_tfm_full"].log1p() - g["value"].log1p()).abs().mean()),
                             "mae_log1p_same_month_median": float((g["med_value"].fill_null(0).log1p() - g["value"].log1p()).abs().mean()),
                             "medae_timesfm_full": float((g["med_tfm_full"] - g["value"]).abs().median()),
                             "medae_same_month_median": float((g["med_value"].fill_null(0) - g["value"]).abs().median()),
                             "share_tfm_full_over_100x_max": float((g["med_tfm_full"] > 100 * g["value"].max()).mean()),
                             "mean_value": float(g["value"].mean())})
    m = pl.DataFrame(all_rows)
    m.write_csv(REPORTS_DIR / "timesfm_compare_metrics.csv")
    base = m.filter(pl.col("method") == "Baseline: same-month rate").select("task", "group", "horizon_months", base_pr="pr_auc", base_brier="brier")
    gain = m.join(base, on=["task", "group", "horizon_months"]).with_columns(
        pr_gain=pl.col("pr_auc") - pl.col("base_pr"), brier_gain=pl.col("base_brier") - pl.col("brier"))
    hz = gain.filter(pl.col("group").is_in(["ALL (excl. persistent)", "Korea municipalities", "Korea cells", "Argentina cells"]),
                     pl.col("method") != "Baseline: same-month rate").select(
        "task", "group", "method", "horizon_months", "pr_auc", "brier", "pr_gain", "brier_gain").sort("task", "group", "method", "horizon_months")
    hz.write_csv(REPORTS_DIR / "timesfm_compare_by_horizon.csv")
    mae = pl.DataFrame(mae_rows)
    mae.write_csv(REPORTS_DIR / "timesfm_compare_mae.csv")
    # 보정 곡선: 격자(지속 열원 제외) h = 1·2·3
    d = frames["grid"].filter(~pl.col("persistent_source"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    bins = np.linspace(0, 1, 11)
    for ax, h in zip(axes, (1, 2, 3)):
        g = d.filter(pl.col("h") == h)
        y = g["label"].to_numpy()
        for c, lab_, col in (("p_xgb", "XGBoost (ours)", "#2a6fdb"), ("p_base", "Baseline", "#d9822b"), ("p_tfm_full", "TimesFM full raw", "#7a5cd6"),
                             ("p_tfmcal_full", "TimesFM full + Platt", "#2fa36b")):
            s = g[c].to_numpy()
            idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
            pts = [(s[idx == b].mean(), y[idx == b].mean()) for b in range(10) if (idx == b).sum() >= 200]
            if pts:
                ax.plot(*zip(*pts), "o-", color=col, label=lab_)
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        ax.set_title(f"Global 0.5° grid, {h} month(s) ahead, 2020–2025"); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "figures" / "timesfm_compare_calibration.png", dpi=130); plt.close(fig)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=230, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=4, thousands_separator=","):
        print(m.filter(pl.col("horizon_months") == 1).pivot(on="method", index=["task", "group", "rows", "pos_rate"], values="pr_auc"))
        print(m.filter(pl.col("horizon_months") == 1).pivot(on="method", index=["task", "group"], values="brier"))
        print(hz.filter(pl.col("group").is_in(["ALL (excl. persistent)", "Korea municipalities"])))
        print(mae)


if __name__ == "__main__":
    main()
