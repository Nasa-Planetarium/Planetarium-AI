"""FireOutlook Planner — 1단계: 월 단위 1~12개월 뒤 (0.5° 격자). 기존 채택 모델·산출물은 건드리지 않는다.

정답   격자·달 기준 점 수(고정 열원 제외 v1.2) ≥ 1 (p_any), 건수 = S-NPP 기준 value 합 (expected_count)
특징   FireOutlook Monthly – Grid(grid_occurrence)와 같음. 시점 h: 직전 달 특징(lag)을 h개월 앞당김 (h=1..12).
후보   base_rate(예년 같은 달 비율), base_5y(최근 5년 같은 달 비율), xgb(분류+isotonic), lgb(LightGBM 분류+isotonic),
       logit(로지스틱 회귀+Platt), count(XGBoost Tweedie 건수 → p = 1 − exp(−μ), Y−1로 isotonic), ens(검증 구간에서 정한 가중 평균)
검증   시험 해 Y ∈ 2020..2025: 학습 ≤ Y−2, 조기 종료·보정 = Y−1. 모델 선택 = 2020–2023 Brier, 확인 = 2024–2025 (다시 고르지 않음).
실행   uv run python -m planetarium.planner_monthly folds [--h 1 2 ...] | select | final
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium import grid_occurrence as go
from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.occurrence import Platt
from planetarium.runlog import peak_memory_gb, start_run_log

WORK = INTERIM_DATA_DIR / "planner" / "monthly"
MODELS_DIR = PROCESSED_DATA_DIR.parent.parent / "models"
HS = list(range(1, 13))
TEST = list(range(2020, 2026))
VAL, HOLD = (2020, 2023), (2024, 2025)
ORIGIN = 2026 * 12 + 10          # 2026-11 (진행 중; 값은 월말 추정 value_projected) — 서비스 예측 기준 달 (--origin으로 바꿈)
AS_OF = "2026-11-14"             # 마지막 반영일 (--as-of)
F = go.FEATURES
XGB_CLS = dict(go.PARAMS)
XGB_CNT = {**{k: v for k, v in go.PARAMS.items() if k != "eval_metric"}, "objective": "reg:tweedie", "tweedie_variance_power": 1.3,
           "eval_metric": "tweedie-nloglik@1.3"}
LGB = {"objective": "binary", "learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
       "bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 0}
MODELS = ["base_rate", "base_5y", "xgb", "lgb", "logit", "count"]


def hdir(h: int):
    return WORK / f"h{h}"


def build_panel(h: int) -> pl.DataFrame:
    """grid_occurrence 패널(시점 h) + 건수 정답(value). 대상 달 범위는 서비스 예측(ORIGIN + h)까지."""
    d = hdir(h)
    d.mkdir(parents=True, exist_ok=True)
    go.HORIZON, go.TARGET, go.PANEL = h, ORIGIN + h, d / "panel.parquet"
    go.panel()
    agg = pl.read_parquet(go.AGG).select("g", "t", count=pl.col("v").clip(lower_bound=0).cast(pl.Float32))
    p = (pl.read_parquet(go.PANEL).with_columns(t=pl.col("year").cast(pl.Int32) * 12 + pl.col("month").cast(pl.Int32) - 1)
         .join(agg, on=["g", "t"], how="left").with_columns(pl.col("count").fill_null(0.0)))
    p.write_parquet(go.PANEL)
    return p


def _X(df):
    return df.select([pl.col(c).cast(pl.Float32) for c in F]).to_numpy()


def _logit_X(df):
    cols = []
    for c in F:
        e = pl.col(c).cast(pl.Float64)
        e = e.fill_null(0).clip(lower_bound=0).log1p() if c.endswith("_value") else e.fill_null(-1 if "rate" in c else 0)
        cols.append(e.alias(c))
    cols += [pl.col("rate_all").is_null().cast(pl.Float64).alias("rate_missing")]
    return df.select(cols).to_numpy()


def fit_predict(tr: pl.DataFrame, va: pl.DataFrame, te: pl.DataFrame) -> tuple[dict, dict]:
    """한 fold: 후보별 시험 확률(+ count μ). 보정은 va(Y−1)로."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out, info = {}, {}
    yv = va["label"].to_numpy()
    iso = lambda r_va, r_te: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(r_va, yv).predict(r_te)  # noqa: E731
    # XGBoost 분류
    bst, _, _ = go.fit_fold(tr, va)
    it = (0, bst.best_iteration + 1)
    out["xgb"] = iso(bst.predict(go._dm(va, False), iteration_range=it), bst.predict(go._dm(te, False), iteration_range=it))
    info["xgb_rounds"] = bst.best_iteration + 1
    # LightGBM 분류
    dtr = lgb.Dataset(_X(tr), tr["label"].to_numpy(), free_raw_data=True)
    dva = lgb.Dataset(_X(va), yv, reference=dtr)
    m = lgb.train(LGB, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    out["lgb"] = iso(m.predict(_X(va), num_iteration=m.best_iteration), m.predict(_X(te), num_iteration=m.best_iteration))
    info["lgb_rounds"] = m.best_iteration
    # 로지스틱 회귀 (학습 표본 200만 행)
    sub = tr.sample(min(2_000_000, tr.height), seed=42)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(_logit_X(sub), sub["label"].to_numpy())
    pl_ = Platt().fit(lr.predict_proba(_logit_X(va))[:, 1], yv)
    out["logit"] = pl_.predict(lr.predict_proba(_logit_X(te))[:, 1])
    # 건수 모델 (Tweedie) → p_any
    dtr = xgb.DMatrix(_X(tr), label=tr["count"].to_numpy(), feature_names=F, missing=np.nan)
    dva_ = xgb.DMatrix(_X(va), label=va["count"].to_numpy(), feature_names=F, missing=np.nan)
    cb = xgb.train(XGB_CNT, dtr, 2000, evals=[(dva_, "va")], early_stopping_rounds=50, verbose_eval=False)
    itc = (0, cb.best_iteration + 1)
    mu_va = cb.predict(dva_, iteration_range=itc)
    mu_te = cb.predict(xgb.DMatrix(_X(te), feature_names=F, missing=np.nan), iteration_range=itc)
    out["count"] = iso(1 - np.exp(-mu_va), 1 - np.exp(-mu_te))
    out["mu"] = mu_te
    info["count_rounds"] = cb.best_iteration + 1
    return out, info


def folds(hs: list[int]) -> None:
    for h in hs:
        t0 = time.perf_counter()
        p = build_panel(h)
        lab = p.filter(pl.col("label").is_not_null())
        res, log = [], []
        for Y in tqdm(TEST, desc=f"h={h}", unit="fold", dynamic_ncols=True):
            tr, va, te = lab.filter(pl.col("year") <= Y - 2), lab.filter(pl.col("year") == Y - 1), lab.filter(pl.col("year") == Y)
            o, info = fit_predict(tr, va, te)
            prior = float(lab.filter(pl.col("year") < Y)["label"].mean())
            res.append(te.select("g", "year", "month", "label", "count", "med_value").with_columns(
                p_base_rate=te["rate_all"].fill_null(prior), p_base_5y=te["rate_5y"].fill_null(te["rate_all"]).fill_null(prior),
                **{f"p_{k}": pl.Series(o[k].astype(np.float32)) for k in ("xgb", "lgb", "logit", "count")},
                mu=pl.Series(o["mu"].astype(np.float32))))
            log.append({"test_year": Y, **info})
        oof = pl.concat(res)
        oof.write_parquet(hdir(h) / "oof.parquet")
        (hdir(h) / "folds.json").write_text(json.dumps(log), encoding="utf-8")
        tqdm.write(f"[h={h}] {time.perf_counter() - t0:,.0f}s, 최대 메모리 {peak_memory_gb() or 0:.2f} GB, rounds {log[-1]}")


def extra2026(hs: list[int]) -> None:
    """2026년 정식 자료(1~10월) 추가 확인: 학습 ≤ 2024, 보정 2025, 시험 2026. 모델 선택에는 쓰지 않는다."""
    for h in hs:
        t0 = time.perf_counter()
        p = pl.read_parquet(hdir(h) / "panel.parquet")
        lab = p.filter(pl.col("label").is_not_null())
        te = lab.filter(pl.col("year") == 2026)
        o, _ = fit_predict(lab.filter(pl.col("year") <= 2024), lab.filter(pl.col("year") == 2025), te)
        prior = float(lab.filter(pl.col("year") < 2026)["label"].mean())
        te.select("g", "year", "month", "label", "count", "med_value").with_columns(
            p_base_rate=te["rate_all"].fill_null(prior), p_base_5y=te["rate_5y"].fill_null(te["rate_all"]).fill_null(prior),
            **{f"p_{k}": pl.Series(o[k].astype(np.float32)) for k in ("xgb", "lgb", "logit", "count")},
            mu=pl.Series(o["mu"].astype(np.float32))).write_parquet(hdir(h) / "oof2026.parquet")
        print(f"[2026 h={h}] {te.height:,}행 (달 {sorted(te['month'].unique().to_list())}), {time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB", flush=True)


def _m(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss
    if len(y) == 0 or y.min() == y.max():
        return {"pr_auc": None, "brier": None, "ece": None}
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, np.clip(p, 0, 1))), "ece": float(ece)}


def ens_weights(v: pl.DataFrame, cands=("p_xgb", "p_lgb", "p_count", "p_logit")) -> dict:
    """검증 구간 Brier 최소 가중 평균 (0.1 간격, 합 1)."""
    y = v["label"].to_numpy()
    P = np.column_stack([v[c].to_numpy() for c in cands])
    best = (9, None)
    steps = np.round(np.arange(0, 1.01, 0.1), 2)
    for w in product(steps, repeat=len(cands)):
        if abs(sum(w) - 1) > 1e-9:
            continue
        b = float(np.mean((P @ np.array(w) - y) ** 2))
        if b < best[0]:
            best = (b, w)
    return dict(zip(cands, best[1]))


def select() -> pl.DataFrame:
    cells = pl.read_parquet(go.CELLS).select("g", "continent", "adm0", "persistent_source")
    rows, choice = [], {}
    for h in HS:
        f = hdir(h) / "oof.parquet"
        if not f.exists():
            continue
        o = pl.concat([pl.read_parquet(x) for x in [f, hdir(h) / "oof2026.parquet"] if x.exists()], how="vertical_relaxed").join(
            cells, on="g").filter(~pl.col("persistent_source"))
        val = o.filter(pl.col("year").is_between(*VAL))
        w = ens_weights(val)
        o = o.with_columns(p_ens=sum(pl.col(c) * wt for c, wt in w.items()))
        groups = [("ALL", pl.lit(True))] + [(c, pl.col("continent") == c) for c in sorted(cells["continent"].unique().to_list())] + \
                 [("Korea", pl.col("adm0") == "KOR"), ("Argentina", pl.col("adm0") == "ARG")]
        for period, pf in (("validation 2020-2023", pl.col("year").is_between(*VAL)), ("holdout 2024-2025", pl.col("year").is_between(*HOLD)),
                           ("extra 2026", pl.col("year") == 2026)):
            for gname, gf in groups:
                d = o.filter(pf, gf)
                y = d["label"].to_numpy()
                base_b = float(np.mean((d["p_base_rate"].to_numpy() - y) ** 2)) if len(y) else None
                for mname in MODELS + ["ens"]:
                    mm = _m(y, d[f"p_{mname}"].to_numpy())
                    rows.append({"horizon_months": h, "period": period, "group": gname, "model": mname, "rows": d.height,
                                 "pos_rate": float(y.mean()) if len(y) else None, **mm,
                                 "bss_vs_base_rate": (1 - mm["brier"] / base_b) if mm["brier"] is not None and base_b else None})
                # 건수 지표 (count μ vs 실제, 기준선 = 같은 달 중앙값)
                if gname in ("ALL", "Korea", "Argentina"):
                    c, mu, md = d["count"].to_numpy(), d["mu"].to_numpy(), d["med_value"].fill_null(0).to_numpy()
                    rows.append({"horizon_months": h, "period": period, "group": gname, "model": "count_metrics", "rows": d.height,
                                 "wape_count": float(np.abs(mu - c).sum() / max(c.sum(), 1e-9)),
                                 "wape_base_median": float(np.abs(md - c).sum() / max(c.sum(), 1e-9)),
                                 "logmae_count": float(np.abs(np.log1p(mu) - np.log1p(c)).mean()),
                                 "logmae_base_median": float(np.abs(np.log1p(md) - np.log1p(c)).mean())})
        vm = [r for r in rows if r["horizon_months"] == h and r["period"].startswith("validation") and r["group"] == "ALL" and r.get("brier") is not None]
        best = min(vm, key=lambda r: r["brier"])
        base = next(r for r in vm if r["model"] == "base_rate")
        # 확실히 나은가: 검증 Brier skill ≥ 1% (그보다 작으면 기준선 서비스, skill = none)
        bss = 1 - best["brier"] / base["brier"]
        skill = "high" if bss >= 0.05 else ("low" if bss >= 0.01 else "none")
        choice[h] = {"model": best["model"] if skill != "none" else "base_rate", "validation_bss": bss, "skill": skill, "ens_weights": w}
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "planner_monthly_metrics.csv")
    (WORK / "selection.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    return m


GRID_LATEST = PROCESSED_DATA_DIR / "planner_monthly_grid.parquet"


def final(hs: list[int]) -> None:
    """서비스 예측: 기준 달 ORIGIN(2026-11, 월말 추정)의 h개월 뒤. 시험 해 2026과 같은 구조(학습 ≤ 2024, 보정 2025)."""
    sel = json.loads((WORK / "selection.json").read_text())
    cells = pl.read_parquet(go.CELLS).select("g", "lat", "lon", "continent", "adm0", "persistent_source")
    parts = []
    for h in hs:
        t0 = time.perf_counter()
        p = pl.read_parquet(hdir(h) / "panel.parquet")
        if p.filter(pl.col("t") == ORIGIN + h).height == 0:     # 기준 달이 바뀌면 패널을 새로 (grid_occurrence agg가 먼저 갱신돼 있어야 함)
            p = build_panel(h)
        lab = p.filter(pl.col("label").is_not_null())
        te = p.filter(pl.col("t") == ORIGIN + h)
        o, _ = fit_predict(lab.filter(pl.col("year") <= 2024), lab.filter(pl.col("year") == 2025), te)
        s = sel[str(h)]
        prior = float(lab["label"].mean())
        base = te["rate_all"].fill_null(prior).to_numpy()
        if s["model"] == "ens":
            pa = sum(o[c.removeprefix("p_")] * w for c, w in s["ens_weights"].items())
        elif s["model"].startswith("base"):
            pa = base if s["model"] == "base_rate" else te["rate_5y"].fill_null(te["rate_all"]).fill_null(prior).to_numpy()
        else:
            pa = o[s["model"]]
        parts.append(te.select("g", "year", "month", "med_value", "rate_5y").with_columns(
            h=pl.lit(h, dtype=pl.Int8), p_any=pl.Series(np.clip(pa, 0, 1).astype(np.float32)), mu=pl.Series(o["mu"].astype(np.float32)),
            base_p=pl.Series(base.astype(np.float32)), model=pl.lit(s["model"]), skill=pl.lit(s["skill"])))
        print(f"[final h={h}] {s['model']} ({s['skill']}), {time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB", flush=True)
    out = pl.concat(parts)
    if 1 in hs:
        # h = 1(다음 달) 격자 값은 기존 FireOutlook Monthly – Grid 파일 그대로 (/occurrence?unit=grid와 같은 숫자; ens와 Brier 차이 0.00015)
        old = pl.read_parquet(go.LATEST_PATH).select(g="cell_id", p_old="p_any", m_old="model")
        out = out.join(old, on="g", how="left").with_columns(
            p_any=pl.when(pl.col("h") == 1).then(pl.col("p_old").fill_null(pl.col("p_any"))).otherwise(pl.col("p_any")).cast(pl.Float32),
            model=pl.when(pl.col("h") == 1).then(pl.col("m_old").fill_null(pl.col("model"))).otherwise(pl.col("model"))).drop("p_old", "m_old")
    out = out.join(cells, on="g").rename({"med_value": "base_count_median"}).with_columns(
        target_month=pl.format("{}-{}", pl.col("year"), pl.col("month").cast(pl.String).str.zfill(2)),
        origin_month=pl.lit(f"{ORIGIN // 12}-{ORIGIN % 12 + 1:02d}"), as_of_date=pl.lit(AS_OF).str.to_date())
    out.write_parquet(GRID_LATEST, compression="zstd")
    print(f"[완료] {GRID_LATEST.name}: {out.height:,}행 ({out['h'].n_unique()}개 시점)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["folds", "select", "final", "extra2026"])
    ap.add_argument("--h", nargs="+", type=int, default=HS)
    ap.add_argument("--origin", help="기준 달 YYYY-MM (진행 중인 달). 기본 2026-11")
    ap.add_argument("--as-of", help="마지막 반영일 YYYY-MM-DD. 기본 2026-11-14")
    a = ap.parse_args(argv)
    global ORIGIN, AS_OF
    if a.origin:
        y_, m_ = map(int, a.origin.split("-"))
        ORIGIN = y_ * 12 + m_ - 1
    if a.as_of:
        AS_OF = a.as_of
    start_run_log("planner_monthly")
    t0 = time.perf_counter()
    if a.step == "folds":
        folds(a.h)
    elif a.step == "final":
        final(a.h)
    elif a.step == "extra2026":
        extra2026(a.h)
    else:
        m = select()
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, float_precision=4):
            print(m.filter(pl.col("group") == "ALL", pl.col("model") != "count_metrics").pivot(on="model", index=["horizon_months", "period"], values="brier"))
        print(json.dumps(json.loads((WORK / "selection.json").read_text()), indent=1))
    print(f"[완료] {a.step} {time.perf_counter() - t0:,.0f}s, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
