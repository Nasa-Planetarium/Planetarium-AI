"""FireOutlook Planner — 격자 확률을 지역 확률로 (aoi_list 5,051 지역으로 학습·검증).

격자 확률을 그냥 곱하면(독립 가정) 넓은 지역에서 과대평가된다. 그래서
  hazard H_r = Σ_g w_rg · (−log(1 − p_g)),  raw_r = 1 − exp(−H_r)      (w_rg = 0.5° 격자 g 중 지역 r에 든 면적 비율)
  p_r = isotonic_bin(raw_r)   — 지역 크기 구간(유효 격자 수 Σ w)마다 검증 구간(2020–2023)에서 학습, 홀드아웃(2024–2025)에서 확인.
건수: E_r = Σ_g w_rg · μ_g, 구간 q10/q90 = E_r × (검증 구간 실제/예측 비율의 10·90 분위수, 크기 구간·시점별).
정답(지역): aoi_month.value ≥ 1 (S-NPP 기준, 고정 열원 제외), status ok·nrt인 달.
실행  uv run python -m planetarium.planner_regions fit
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

from planetarium import grid_occurrence as go
from planetarium.config import PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.planner_monthly import HOLD, HS, VAL, WORK, hdir
from planetarium.runlog import peak_memory_gb, start_run_log

SIZE_BINS = [0.0, 0.5, 2.0, 8.0, 32.0, 128.0, 1e9]
SIZE_LABELS = ["<0.5", "0.5-2", "2-8", "8-32", "32-128", ">=128"]
CAL_PATH = WORK / "region_calibration.json"
GROUPS = (("all regions", pl.lit(True)), ("countries", pl.col("level") == "country"), ("Korea sido", pl.col("level") == "sido"),
          ("Korea sgg", pl.col("level") == "sgg"), ("adm1", pl.col("level") == "adm1"))
LEVEL_GROUP = {"country": "countries", "sido": "Korea sido", "sgg": "Korea sgg", "adm1": "adm1"}


def grid_index() -> pl.DataFrame:
    return pl.read_parquet(go.CELLS).select("g", "lat", "lon", "continent", "adm0", "persistent_source").with_row_index("gi")


def aoi_weights() -> tuple[sp.csr_matrix, list[str], np.ndarray]:
    """W[r, gi] = 지역 r에 든 0.5° 격자 gi의 면적 비율 (0.1° 칸 소속 가중치 합 / 25)."""
    from planetarium.harmonize import aoi_membership
    mem, aoi = aoi_membership()
    gix = grid_index().select("g", "gi")
    m = (mem.with_columns(g=go.id05(pl.col("cell_id"))).group_by("aoi_id", "g").agg(w=(pl.col("weight").sum() / 25).cast(pl.Float64))
         .join(gix, on="g"))
    ids = sorted(aoi["aoi_id"].to_list())
    ri = {a: i for i, a in enumerate(ids)}
    W = sp.csr_matrix((m["w"].to_numpy(), (np.array([ri[a] for a in m["aoi_id"].to_list()]), m["gi"].to_numpy())),
                      shape=(len(ids), gix.height))
    return W, ids, np.asarray(W.sum(axis=1)).ravel()


def size_bin(n: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(SIZE_BINS, n, side="right") - 1, 0, len(SIZE_LABELS) - 1)


def selected_p(o: pl.DataFrame, sel: dict) -> np.ndarray:
    if sel["model"] == "ens":
        return sum(o[c].to_numpy() * w for c, w in sel["ens_weights"].items())
    return o[f"p_{sel['model']}"].to_numpy()


def region_frame(h: int, W, ids, nsize, sel) -> pl.DataFrame:
    gix = grid_index().select("g", "gi")
    files = [hdir(h) / "oof.parquet"] + ([hdir(h) / "oof2026.parquet"] if (hdir(h) / "oof2026.parquet").exists() else [])
    o = pl.concat([pl.read_parquet(f).select("g", "year", "month", "count", "med_value", "mu", "p_base_rate", "p_base_5y",
                                             "p_xgb", "p_lgb", "p_logit", "p_count") for f in files], how="vertical_relaxed").join(gix, on="g")
    o = o.with_columns(p_sel=pl.Series(selected_p(o, sel)).clip(0, 0.999999))
    if h == 1:
        # 격자 다음 달은 기존 FireOutlook Monthly – Grid(grid_occurrence)를 그대로 쓰므로 지역 보정도 그 값으로 (2026 행은 없음)
        old = pl.read_parquet(go.OOF).select("g", "year", "month", p_old="p_iso")
        o = o.join(old, on=["g", "year", "month"], how="inner").with_columns(p_sel=pl.col("p_old").clip(0, 0.999999)).drop("p_old")
    parts = []
    for (y, m), g in o.group_by("year", "month"):
        haz = np.zeros(W.shape[1])
        mu = np.zeros(W.shape[1])
        base = np.zeros(W.shape[1])
        med = np.zeros(W.shape[1])
        gi = g["gi"].to_numpy()
        haz[gi] = -np.log1p(-g["p_sel"].to_numpy())
        mu[gi] = g["mu"].to_numpy()
        med[gi] = g["med_value"].fill_null(0).to_numpy()
        base[gi] = -np.log1p(-np.clip(g["p_base_rate"].to_numpy(), 0, 0.999999))
        parts.append(pl.DataFrame({"aoi_id": ids, "year": np.full(len(ids), y, np.int16), "month": np.full(len(ids), m, np.int8),
                                   "raw": 1 - np.exp(-(W @ haz)), "raw_base": 1 - np.exp(-(W @ base)), "exp_count": W @ mu, "exp_base": W @ med, "n_eff": nsize}))
    r = pl.concat(parts)
    am = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_month.parquet", columns=["aoi_id", "year", "month", "value", "status"]).filter(
        pl.col("status").is_in(["ok", "nrt"]))
    return r.join(am.select("aoi_id", "year", "month", "value", label=(pl.col("value") >= 1).cast(pl.Float64)), on=["aoi_id", "year", "month"])


def fit() -> None:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score
    t0 = time.perf_counter()
    W, ids, nsize = aoi_weights()
    sel_all = json.loads((WORK / "selection.json").read_text())
    lv = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_list.parquet", columns=["aoi_id", "level"])
    cal, rows = {}, []
    for h in HS:
        if str(h) not in sel_all:
            continue
        sel = sel_all[str(h)]
        r = region_frame(h, W, ids, nsize, sel).filter(pl.col("n_eff") > 0)
        r = r.with_columns(sb=pl.Series(size_bin(r["n_eff"].to_numpy())))
        val, hold = r.filter(pl.col("year").is_between(*VAL)), r.filter(pl.col("year").is_between(*HOLD))
        cal[h] = {}
        wm = float((val["exp_count"] - val["value"]).abs().sum() / val["value"].sum())
        wb = float((val["exp_base"] - val["value"]).abs().sum() / val["value"].sum())
        src = "exp_count" if wm < wb else "exp_base"
        cal[h]["count_source"] = {"source": "model_mu" if src == "exp_count" else "baseline_median", "val_wape_model": wm, "val_wape_base": wb}
        # 지역 등급별 건수 원천 (검증 구간 WAPE로 고름; 홀드아웃은 확인만)
        vlv = val.join(lv, on="aoi_id")
        cal[h]["count_source_by_group"] = {}
        for gname, gf in GROUPS:
            d = vlv.filter(gf)
            if d.height and d["value"].sum() > 0:
                a_ = float((d["exp_count"] - d["value"]).abs().sum() / d["value"].sum())
                b_ = float((d["exp_base"] - d["value"]).abs().sum() / d["value"].sum())
                cal[h]["count_source_by_group"][gname] = "model_mu" if a_ < b_ else "baseline_median"
        extra = r.filter(pl.col("year") == 2026)

        def fit_bins(v_):
            out = {}
            for b in range(len(SIZE_LABELS)):
                v = v_.filter(pl.col("sb") == b)
                if v.height >= 50:
                    out[b] = (IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(v["raw"].to_numpy(), v["label"].to_numpy()),
                              IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(v["raw_base"].to_numpy(), v["label"].to_numpy()))
            return out

        def apply_bins(d, fitted):
            pp, pb = d["raw"].to_numpy().copy(), d["raw_base"].to_numpy().copy()
            for b, (iso, isob) in fitted.items():
                k = (d["sb"] == b).to_numpy()
                pp[k], pb[k] = iso.predict(d["raw"].to_numpy()[k]), isob.predict(d["raw_base"].to_numpy()[k])
            return d.with_columns(p=pl.Series(pp), p_base=pl.Series(pb))

        fitted = fit_bins(val)
        for b, (iso, isob) in fitted.items():
            v = val.filter(pl.col("sb") == b)
            ratio = (v["value"] / v[src].clip(lower_bound=1e-6)).filter(v[src] > 0.1)
            cal[h][SIZE_LABELS[b]] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(),
                                      "base_x": isob.X_thresholds_.tolist(), "base_y": isob.y_thresholds_.tolist(),
                                      "ratio_q10": float(ratio.quantile(0.1)) if ratio.len() else 0.0,
                                      "ratio_q90": float(ratio.quantile(0.9)) if ratio.len() else 0.0,
                                      **{f"ratio_{nm}_q{q}": float((v["value"] / v[cn].clip(lower_bound=1e-6)).filter(v[cn] > 0.1).quantile(q / 100) or 0.0)
                                         for nm, cn in (("model", "exp_count"), ("base", "exp_base")) for q in (10, 90)}}
        # 검증 구간 자체 값은 해 하나씩 빼고 보정(leave-one-year-out) → 직접 학습 모델과 같은 조건으로 비교할 수 있게
        val_cv = pl.concat([apply_bins(val.filter(pl.col("year") == y_), fit_bins(val.filter(pl.col("year") != y_))) for y_ in range(VAL[0], VAL[1] + 1)])
        preds = pl.concat([val_cv.with_columns(period=pl.lit("validation 2020-2023 (LOYO)")),
                           apply_bins(hold, fitted).with_columns(period=pl.lit("holdout 2024-2025"))]
                          + ([apply_bins(extra, fitted).with_columns(period=pl.lit("extra 2026"))] if extra.height else []))
        preds.drop("sb").write_parquet(WORK / f"region_preds_h{h}.parquet")
        preds = preds.join(lv, on="aoi_id")
        for (period,), pdf in preds.group_by("period", maintain_order=True):
          for gname, gf in GROUPS:
            d = pdf.filter(gf)
            y = d["label"].to_numpy()
            if len(y) == 0 or y.min() == y.max():
                continue
            b_raw = float(np.mean((np.clip(d["raw"].to_numpy(), 0, 1) - y) ** 2))
            b_cal = float(np.mean((d["p"].to_numpy() - y) ** 2))
            b_base = float(np.mean((d["p_base"].to_numpy() - y) ** 2))
            q10 = d.join(pl.DataFrame({"sb": list(range(len(SIZE_LABELS))), "lab": SIZE_LABELS}), on="sb")
            lo = np.array([cal[h].get(l_, {}).get("ratio_q10", 0) for l_ in q10["lab"].to_list()]) * q10[src].to_numpy()
            hi = np.array([cal[h].get(l_, {}).get("ratio_q90", 0) for l_ in q10["lab"].to_list()]) * q10[src].to_numpy()
            act = q10["value"].to_numpy()
            rows.append({"horizon_months": h, "period": period, "group": gname, "rows": d.height, "pos_rate": float(y.mean()),
                         "brier_independent_product": b_raw, "brier_calibrated": b_cal, "brier_base": b_base,
                         "bss_calibrated_vs_base": 1 - b_cal / b_base, "pr_auc_calibrated": float(average_precision_score(y, d["p"].to_numpy())),
                         "count_source": cal[h]["count_source"]["source"],
                         "wape_count_model": float(np.abs(d["exp_count"].to_numpy() - d["value"].to_numpy()).sum() / max(d["value"].sum(), 1e-9)),
                         "wape_count_base_median": float(np.abs(d["exp_base"].to_numpy() - d["value"].to_numpy()).sum() / max(d["value"].sum(), 1e-9)),
                         "logmae_count_model": float(np.abs(np.log1p(d["exp_count"].to_numpy()) - np.log1p(d["value"].clip(lower_bound=0).to_numpy())).mean()),
                         "logmae_count_base_median": float(np.abs(np.log1p(d["exp_base"].to_numpy()) - np.log1p(d["value"].clip(lower_bound=0).to_numpy())).mean()),
                         "interval_q10_q90_coverage": float(((act >= lo) & (act <= hi)).mean())})
        print(f"[h={h}] regions done, {time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB", flush=True)
    CAL_PATH.write_text(json.dumps({str(k): v for k, v in cal.items()}), encoding="utf-8")
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "planner_region_metrics.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, float_precision=4):
        print(m.filter(pl.col("horizon_months").is_in([1, 2, 3, 6, 12])).select("horizon_months", "period", "group", "bss_calibrated_vs_base", "wape_count_model", "wape_count_base_median", "interval_q10_q90_coverage"))


if __name__ == "__main__":
    start_run_log("planner_regions")
    fit()
