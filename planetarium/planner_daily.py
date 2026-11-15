"""FireOutlook Planner — 2단계: 일 단위 1~7일 뒤 (0.5° 격자). FireOutlook Spread(spread_ops)와는 별개다.

기준일 d(그날까지의 탐지를 앎) → 정답 y_k = d+1 ~ d+k 안에 점(고정 열원 제외 v1.2) ≥ 1, c_k = 같은 기간 S-NPP 기준 value 합 (k = 1..7)
특징  최근 1·3·7·14일 탐지(value)·FRP(S-NPP 환산), 최근 14·30일 탐지일 수, 연속 탐지 일수(사건 지속), 마지막 탐지 후 일수,
      이웃 8칸의 7일 탐지·오늘 탐지 칸 수, 계절 기후값(그 전 해들까지 같은 달: 하루 탐지율·7일 안 탐지율·하루 평균 건수), 위도, 연중 날짜.
      FWI는 전 지구 일 단위 값이 없어(한국·아르헨티나만) 넣지 않았다. (넣는다면 운영 지연 −4일 값만 써야 함)
표본  매일 모든 격자를 보되, 활동 칸(최근 30일·이웃 7일 탐지, 기후 7일 탐지율 > 5%, 또는 7일 안 양성)은 25%, 나머지는 1%만 남기고
      가중치 1/비율을 붙인다 (학습·평가 모두 가중; Horvitz–Thompson 방식이라 치우침 없음).
후보  base_clim(그 전 해들 같은 달의 y_k 비율), base_5y(최근 5년), xgb, lgb, logit, count(Tweedie → 1 − exp(−μ)), ens
검증  시험 해 Y ∈ 2020..2025 (학습 ≤ Y−2, 보정 Y−1). 선택 2020–2023, 확인 2024–2025. 2026(1~10월)은 추가 확인.
서비스 기준일 L(2026-11-14)에서 k = 1..7 예측 → 하루별로 나눠 저장: hazard H(k) = −log(1 − P(k)), 하루 j의 p = 1 − exp(−(H(j) − H(j−1))).
실행  uv run python -m planetarium.planner_daily build | samples | folds [--k 1 2 3 7] | select | final
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium import grid_occurrence as go
from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

WORK = INTERIM_DATA_DIR / "planner" / "daily"
SAMPLES = WORK / "samples"
D0 = date(2012, 2, 1)
LAST = date(2026, 11, 14)          # 기준일 (11-15는 아직 일부만 들어옴)
KS = list(range(1, 8))
TEST = list(range(2020, 2026))
VAL, HOLD = (2020, 2023), (2024, 2025)
KEEP_ACTIVE, KEEP_QUIET = 0.25, 0.01
F = ["v1", "v3", "v7", "v14", "a14", "a30", "run", "since", "frp1", "frp7", "nb_v7", "nb_a1", "nb_a7",
     "clim_p1", "clim_y7", "clim_v", "n_years", "abs_lat", "doy_sin", "doy_cos"]
XGB = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda", "max_depth": 7,
       "learning_rate": 0.1, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8, "max_bin": 256, "seed": 42}
XGB_CNT = {**{k: v for k, v in XGB.items() if k != "eval_metric"}, "objective": "reg:tweedie", "tweedie_variance_power": 1.3,
           "eval_metric": "tweedie-nloglik@1.3"}
LGB = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
       "bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 0}
MODELS = ["base_clim", "base_5y", "xgb", "lgb", "logit", "count"]
OUT = PROCESSED_DATA_DIR / "planner_daily_grid.parquet"


def cells() -> pl.DataFrame:
    return pl.read_parquet(go.CELLS).select("g", "lat", "lon", "persistent_source").with_row_index("gi")


def build() -> None:
    """0.1° cell_day → 0.5° 격자 × 날짜 밀집 배열 P(점 수)·V(value)·R(FRP S-NPP 환산). 해마다 읽는다."""
    WORK.mkdir(parents=True, exist_ok=True)
    c = cells().select("g", "gi")
    G, D = c.height, (LAST - D0).days + 1
    arr = {k: np.zeros((G, D), np.float32) for k in ("P", "V", "R")}
    for y in tqdm(range(2012, LAST.year + 1), desc="일 배열", unit="year"):
        lo, hi = max(D0, date(y, 1, 1)), min(LAST, date(y, 12, 31))
        cd = (pl.scan_parquet(PROCESSED_DATA_DIR / "harmonized_cell_day.parquet").filter(pl.col("year") == y, pl.col("date_local").is_between(lo, hi))
              .group_by(g=go.id05(pl.col("cell_id")), date_local=pl.col("date_local"))
              .agg(P=pl.col("n_points").sum().cast(pl.Float32), V=pl.col("value").clip(lower_bound=0).sum().cast(pl.Float32)).collect())
        fr = (pl.scan_parquet(PROCESSED_DATA_DIR / "harmonized_cell_day_frp.parquet").filter(pl.col("date_local").is_between(lo, hi))
              .group_by(g=go.id05(pl.col("cell_id")), date_local=pl.col("date_local"))
              .agg(R=pl.col("frp_snpp_equiv_sum").sum().cast(pl.Float32)).collect())
        for df, cols in ((cd, ("P", "V")), (fr, ("R",))):
            df = df.join(c, on="g").with_columns(di=(pl.col("date_local") - pl.lit(D0)).dt.total_days())
            gi, di = df["gi"].to_numpy(), df["di"].to_numpy()
            for k in cols:
                arr[k][gi, di] = df[k].to_numpy()
    for k, a in arr.items():
        np.save(WORK / f"{k}.npy", a)
    print(f"[완료] 일 배열 {G:,} 격자 × {D:,}일, 점 있는 칸-일 {(arr['P'] > 0).sum():,}, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


def neighbors(c: pl.DataFrame) -> np.ndarray:
    g = c["g"].to_numpy()
    pos = {int(v): i for i, v in enumerate(g)}
    r, cc = g // 720, g % 720
    nb = np.full((len(g), 8), len(g), np.int64)   # len(g) = 0으로 채운 가짜 칸
    j = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            nb[:, j] = [pos.get(int((ri + dr) * 720 + (ci + dc) % 720), len(g)) for ri, ci in zip(r, cc)]
            j += 1
    return nb


def _day_index(d: date) -> int:
    return (d - D0).days


def samples(full_days: set[int] | None = None, write_samples: bool = True) -> None:
    """두 번 돈다: (1) 해·달별 정답 합계 → 확장 기후값, (2) 날마다 특징·정답·표본 추출 → 해마다 parquet.
    full_days: 이 날짜 번호(di)들은 모든 격자의 특징을 WORK/fullgrid/d{di}.parquet로 따로 저장 (30일 검증용; 표본과 같은 코드).
    write_samples=False면 표본 파일은 건드리지 않는다."""
    t0 = time.perf_counter()
    c = cells()
    G = c.height
    P, V, R = (np.load(WORK / f"{k}.npy") for k in ("P", "V", "R"))
    D = P.shape[1]
    A = P > 0
    days = [D0 + timedelta(i) for i in range(D)]
    years = list(range(2012, LAST.year + 1))
    # (1) 기후값 합계: 해 × 달 × [y1..y7, p1, v] × 격자
    S = np.zeros((len(years), 12, 9, G), np.float32)
    N = np.zeros((len(years), 12), np.float32)
    for d in tqdm(range(D - 7), desc="기후 합계", unit="day", mininterval=5):
        yi, m = days[d].year - 2012, days[d].month - 1
        fut = np.logical_or.accumulate(A[:, d + 1:d + 8], axis=1)
        S[yi, m, :7] += fut.T
        S[yi, m, 7] += A[:, d]
        S[yi, m, 8] += V[:, d]
        N[yi, m] += 1
    nb = neighbors(c)
    lat = np.abs(c["lat"].to_numpy()).astype(np.float32)
    rng = np.random.default_rng(42)
    run = np.zeros(G, np.float32)
    since = np.full(G, 400, np.float32)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    buf, cur_year, clim = [], None, None
    total = 0

    def flush(y):
        nonlocal buf, total
        if buf:
            df = pl.concat(buf)
            df.write_parquet(SAMPLES / f"year={y}.parquet")
            total += df.height
        buf = []

    for d in tqdm(range(D), desc="표본", unit="day", mininterval=5):
        dt = days[d]
        run = np.where(A[:, d], np.minimum(run + 1, 60), 0).astype(np.float32)
        since = np.where(A[:, d], 0, np.minimum(since + 1, 400)).astype(np.float32)
        if dt.year < 2013:
            continue
        if dt.year != cur_year:
            if write_samples:
                flush(cur_year)
            cur_year = dt.year
            yi = dt.year - 2012
            n_all = N[:yi].sum(0)[:, None, None]
            n_5y = N[max(0, yi - 5):yi].sum(0)[:, None, None]
            clim = (S[:yi].sum(0) / np.maximum(n_all, 1), S[max(0, yi - 5):yi].sum(0) / np.maximum(n_5y, 1))
            n_years = np.float32(yi)
        m = dt.month - 1
        ca, c5 = clim[0][m], clim[1][m]
        w14 = slice(max(0, d - 13), d + 1)
        v7 = V[:, max(0, d - 6):d + 1].sum(1)
        a7 = A[:, max(0, d - 6):d + 1].any(1)
        feat = {
            "v1": np.log1p(V[:, d]), "v3": np.log1p(V[:, max(0, d - 2):d + 1].sum(1)), "v7": np.log1p(v7), "v14": np.log1p(V[:, w14].sum(1)),
            "a14": A[:, w14].sum(1).astype(np.float32), "a30": A[:, max(0, d - 29):d + 1].sum(1).astype(np.float32), "run": run, "since": since,
            "frp1": np.log1p(R[:, d]), "frp7": np.log1p(R[:, max(0, d - 6):d + 1].sum(1)),
            "nb_v7": np.log1p(np.append(v7, 0)[nb].sum(1)), "nb_a1": np.append(A[:, d], False)[nb].sum(1).astype(np.float32),
            "nb_a7": np.append(a7, False)[nb].sum(1).astype(np.float32),
            "clim_p1": ca[7], "clim_y7": ca[6], "clim_v": np.log1p(ca[8]), "n_years": np.full(G, n_years, np.float32), "abs_lat": lat,
            "doy_sin": np.full(G, np.sin(2 * np.pi * dt.timetuple().tm_yday / 365.25), np.float32),
            "doy_cos": np.full(G, np.cos(2 * np.pi * dt.timetuple().tm_yday / 365.25), np.float32),
        }
        if full_days and d in full_days:
            (WORK / "fullgrid").mkdir(exist_ok=True)
            pl.DataFrame({"gi": np.arange(G, dtype=np.int32), "di": np.full(G, d, np.int32), "year": np.full(G, dt.year, np.int16),
                          **{k: v.astype(np.float32) for k, v in feat.items()},
                          **{f"base_clim{k}": ca[k - 1] for k in KS}}).write_parquet(WORK / "fullgrid" / f"d{d}.parquet")
        if not write_samples:
            continue
        if d == D - 1:     # 서비스 기준일: 모든 격자
            o = pl.DataFrame({"gi": np.arange(G, dtype=np.int32), **{k: v.astype(np.float32) for k, v in feat.items()},
                              **{f"base_clim{k}": ca[k - 1] for k in KS}, **{f"base_5y{k}": c5[k - 1] for k in KS}})
            o.write_parquet(WORK / "origin.parquet")
            continue
        if d + 7 >= D:
            continue
        fut = np.logical_or.accumulate(A[:, d + 1:d + 8], axis=1)
        cnt = np.cumsum(V[:, d + 1:d + 8], axis=1)
        active = (feat["a30"] > 0) | (feat["nb_a7"] > 0) | (ca[6] > 0.05) | fut[:, 6]
        prob = np.where(active, KEEP_ACTIVE, KEEP_QUIET).astype(np.float32)
        keep = rng.random(G) < prob
        idx = np.flatnonzero(keep)
        df = pl.DataFrame({"gi": idx.astype(np.int32), "year": np.full(len(idx), dt.year, np.int16), "month": np.full(len(idx), dt.month, np.int8),
                           "di": np.full(len(idx), d, np.int32), "w": 1 / prob[idx],
                           **{k: v[idx].astype(np.float32) for k, v in feat.items()},
                           **{f"y{k}": fut[idx, k - 1].astype(np.uint8) for k in KS}, **{f"c{k}": cnt[idx, k - 1] for k in KS},
                           **{f"base_clim{k}": ca[k - 1][idx] for k in KS}, **{f"base_5y{k}": c5[k - 1][idx] for k in KS}})
        buf.append(df)
    if write_samples:
        flush(cur_year)
    print(f"[완료] 표본 {total:,}행, {time.perf_counter() - t0:,.0f}s, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


def _iso(p, y, w):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y, sample_weight=w)


def _cap(df: pl.DataFrame, n: int, seed=0) -> pl.DataFrame:
    """행이 n을 넘으면 무작위로 줄이고 가중치를 그만큼 올린다."""
    if df.height <= n:
        return df
    return df.sample(n, seed=seed).with_columns(pl.col("w") * (df.height / n))


def fit_predict(k: int, tr: pl.DataFrame, va: pl.DataFrame, te: pl.DataFrame) -> dict:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = lambda df: df.select([pl.col(c).cast(pl.Float32) for c in F]).to_numpy()  # noqa: E731
    yk, ck = f"y{k}", f"c{k}"
    tr = _cap(tr, 12_000_000)
    Xtr, Xva, Xte = X(tr), X(va), X(te)
    ytr, yva = tr[yk].to_numpy(), va[yk].to_numpy()
    wtr, wva = tr["w"].to_numpy(), va["w"].to_numpy()
    out = {}
    dtr, dva = xgb.DMatrix(Xtr, ytr, weight=wtr), xgb.DMatrix(Xva, yva, weight=wva)
    b = xgb.train(XGB, dtr, 600, evals=[(dva, "va")], early_stopping_rounds=30, verbose_eval=False)
    rng_ = (0, b.best_iteration + 1)
    cal = _iso(b.predict(dva, iteration_range=rng_), yva, wva)
    out["xgb"] = cal.predict(b.predict(xgb.DMatrix(Xte), iteration_range=rng_))
    del dtr
    s = _cap(tr, 4_000_000, 1)
    l = lgb.train(LGB, lgb.Dataset(X(s), s[yk].to_numpy(), weight=s["w"].to_numpy()), 600,
                  valid_sets=[lgb.Dataset(Xva, yva, weight=wva)], callbacks=[lgb.early_stopping(30, verbose=False)])
    out["lgb"] = _iso(l.predict(Xva), yva, wva).predict(l.predict(Xte))
    s = _cap(tr, 2_000_000, 2)
    sc = StandardScaler().fit(X(s))
    lr = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(X(s)), s[yk].to_numpy(), sample_weight=s["w"].to_numpy())
    # 보정: 표본 가중을 써야 하므로 Platt 대신 가중 isotonic
    out["logit"] = _iso(lr.predict_proba(sc.transform(Xva))[:, 1], yva, wva).predict(lr.predict_proba(sc.transform(Xte))[:, 1])
    dtr = xgb.DMatrix(Xtr, tr[ck].to_numpy(), weight=wtr)
    dva = xgb.DMatrix(Xva, va[ck].to_numpy(), weight=wva)
    bc = xgb.train(XGB_CNT, dtr, 600, evals=[(dva, "va")], early_stopping_rounds=30, verbose_eval=False)
    rc = (0, bc.best_iteration + 1)
    mu_va, mu_te = bc.predict(dva, iteration_range=rc), bc.predict(xgb.DMatrix(Xte), iteration_range=rc)
    out["count"] = _iso(1 - np.exp(-mu_va), yva, wva).predict(1 - np.exp(-mu_te))
    out["mu"] = mu_te
    return out


def hdir(k: int):
    return WORK / f"k{k}"


def load_years(years, cols=None) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(SAMPLES / f"year={y}.parquet", columns=cols) for y in years if (SAMPLES / f"year={y}.parquet").exists()])


def folds(ks: list[int]) -> None:
    for k in ks:
        hdir(k).mkdir(parents=True, exist_ok=True)
        parts = []
        for Y in tqdm(TEST + [2026], desc=f"k={k}", unit="fold"):
            t0 = time.perf_counter()
            tr, va, te = load_years(range(2013, Y - 1)), load_years([Y - 1]), load_years([Y])
            o = fit_predict(k, tr, va, te)
            parts.append(te.select("gi", "year", "month", "di", "w", y=pl.col(f"y{k}"), count=pl.col(f"c{k}"),
                                   p_base_clim=pl.col(f"base_clim{k}"), p_base_5y=pl.col(f"base_5y{k}").fill_nan(None).fill_null(pl.col(f"base_clim{k}")))
                         .with_columns(**{f"p_{m}": pl.Series(np.clip(o[m], 0, 1).astype(np.float32)) for m in ("xgb", "lgb", "logit", "count")},
                                       mu=pl.Series(o["mu"].astype(np.float32))))
            tqdm.write(f"  Y={Y} {te.height:,}행 {time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB")
        pl.concat(parts).write_parquet(hdir(k) / "oof.parquet")


def _wm(y, p, w):
    from sklearn.metrics import average_precision_score
    b = float(np.average((p - y) ** 2, weights=w))
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(np.average(p[bins == i], weights=w[bins == i]) - np.average(y[bins == i], weights=w[bins == i])) * w[bins == i].sum()
              for i in range(10) if (bins == i).any()) / w.sum()
    return {"brier": b, "pr_auc": float(average_precision_score(y, p, sample_weight=w)), "ece": float(ece)}


def select() -> pl.DataFrame:
    c = cells().select("gi", "persistent_source")
    rows, choice = [], {}
    for k in KS:
        f = hdir(k) / "oof.parquet"
        if not f.exists():
            continue
        o = pl.read_parquet(f).join(c, on="gi").filter(~pl.col("persistent_source")).with_columns(pl.col("p_base_5y").fill_nan(None).fill_null(pl.col("p_base_clim")))
        val = o.filter(pl.col("year").is_between(*VAL))
        # 앙상블: 검증 구간 Brier 최소 가중 (xgb, lgb, count; 0.1 간격)
        best_w, best_b = None, 9
        for a in np.arange(0, 1.01, 0.1):
            for bb in np.arange(0, 1.01 - a, 0.1):
                p = a * val["p_xgb"].to_numpy() + bb * val["p_lgb"].to_numpy() + (1 - a - bb) * val["p_count"].to_numpy()
                br = float(np.average((p - val["y"].to_numpy()) ** 2, weights=val["w"].to_numpy()))
                if br < best_b:
                    best_b, best_w = br, {"p_xgb": round(a, 1), "p_lgb": round(bb, 1), "p_count": round(1 - a - bb, 1)}
        o = o.with_columns(p_ens=sum(pl.col(cn) * wt for cn, wt in best_w.items()))
        res = {}
        for period, d in (("val_2020_2023", o.filter(pl.col("year").is_between(*VAL))), ("hold_2024_2025", o.filter(pl.col("year").is_between(*HOLD))),
                          ("extra_2026", o.filter(pl.col("year") == 2026))):
            y, w = d["y"].to_numpy().astype(float), d["w"].to_numpy()
            base = _wm(y, d["p_base_clim"].to_numpy(), w)["brier"]
            for m in MODELS + ["ens"]:
                r = _wm(y, d[f"p_{m}"].to_numpy(), w)
                cnt = d["count"].to_numpy()
                mu = d["mu"].to_numpy() if m in ("count", "ens", "xgb", "lgb", "logit") else None
                rows.append({"horizon_days": k, "period": period, "model": m, **r, "bss_vs_base_clim": 1 - r["brier"] / base,
                             "wape_count": float(np.sum(w * np.abs(mu - cnt)) / np.sum(w * cnt)) if mu is not None else None,
                             "log_mae_count": float(np.average(np.abs(np.log1p(mu) - np.log1p(cnt)), weights=w)) if mu is not None else None,
                             "pos_rate": float(np.average(y, weights=w)), "rows": d.height})
            if period == "val_2020_2023":
                res = {m: r for m in []}
        m = pl.DataFrame([r for r in rows if r["horizon_days"] == k and r["period"] == "val_2020_2023"])
        best = m.filter(~pl.col("model").str.starts_with("base")).sort("brier").row(0, named=True)
        bss = best["bss_vs_base_clim"]
        skill = "high" if bss >= 0.05 else "low" if bss >= 0.01 else "none"
        va = o.filter(pl.col("year").is_between(*VAL), pl.col("mu") > 0.1)
        ratio = (va["count"] / va["mu"])
        choice[k] = {"model": best["model"] if skill != "none" else "base_clim", "validation_bss": bss, "skill": skill, "ens_weights": best_w,
                     "count_ratio_q10": float(ratio.quantile(0.1)), "count_ratio_q90": float(ratio.quantile(0.9))}
    (WORK / "selection.json").write_text(json.dumps({str(k): v for k, v in choice.items()}, indent=1), encoding="utf-8")
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "planner_daily_metrics.csv")
    print(json.dumps(choice, indent=1))
    return m


def final() -> None:
    """서비스: 학습 ≤ 2025, 보정 2026(1~10월) → 기준일 LAST의 k = 1..7. 하루별 hazard로 나눠 저장."""
    sel = json.loads((WORK / "selection.json").read_text())
    c = cells()
    org = pl.read_parquet(WORK / "origin.parquet").sort("gi")
    tr, va = load_years(range(2013, 2026)), load_years([2026])
    P, MU, PB = {}, {}, {}
    for k in KS:
        t0 = time.perf_counter()
        s = sel.get(str(k)) or sel[str(min(sel, key=lambda x: abs(int(x) - k)))]   # 검증 안 한 k는 가장 가까운 k의 선택을 따름
        o = fit_predict(k, tr, va, org)
        if s["model"] == "ens":
            p = sum(o[cn.removeprefix("p_")] * wt for cn, wt in s["ens_weights"].items())
        elif s["model"].startswith("base"):
            p = org[f"{s['model']}{k}"].fill_nan(None).fill_null(org[f"base_clim{k}"]).to_numpy()
        else:
            p = o[s["model"]]
        P[k], MU[k], PB[k] = np.clip(p, 0, 0.999999), o["mu"], np.clip(org[f"base_clim{k}"].to_numpy(), 0, 0.999999)
        print(f"[final k={k}] {s['model']} ({s['skill']}), {time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB", flush=True)
    # 누적 P(k)를 단조로 맞춘 뒤 하루별로 나눔
    Pm = np.maximum.accumulate(np.stack([P[k] for k in KS], 1), axis=1)
    Bm = np.maximum.accumulate(np.stack([PB[k] for k in KS], 1), axis=1)
    Mm = np.maximum.accumulate(np.stack([MU[k] for k in KS], 1), axis=1)
    H, HB = -np.log1p(-Pm), -np.log1p(-Bm)
    dH = np.diff(np.concatenate([np.zeros((len(org), 1)), H], 1), axis=1)
    dHB = np.diff(np.concatenate([np.zeros((len(org), 1)), HB], 1), axis=1)
    dM = np.diff(np.concatenate([np.zeros((len(org), 1)), Mm], 1), axis=1)
    clim_v = np.expm1(org["clim_v"].to_numpy())
    parts = []
    for j in KS:
        s = sel.get(str(j)) or sel[str(min(sel, key=lambda x: abs(int(x) - j)))]
        parts.append(pl.DataFrame({"gi": org["gi"].to_numpy(), "date": [LAST + timedelta(j)] * len(org), "horizon_days": np.full(len(org), j, np.int8),
                                   "p_any": (1 - np.exp(-dH[:, j - 1])).astype(np.float32), "p_cum": Pm[:, j - 1].astype(np.float32),
                                   "mu": dM[:, j - 1].astype(np.float32), "mu_cum": Mm[:, j - 1].astype(np.float32),
                                   "base_p": (1 - np.exp(-dHB[:, j - 1])).astype(np.float32), "base_count": clim_v.astype(np.float32),
                                   "model": [s["model"]] * len(org), "skill": [s["skill"]] * len(org),
                                   "count_ratio_q10": np.full(len(org), s["count_ratio_q10"], np.float32),
                                   "count_ratio_q90": np.full(len(org), s["count_ratio_q90"], np.float32)}))
    out = pl.concat(parts).join(c, on="gi").drop("gi").with_columns(as_of_date=pl.lit(LAST))
    out.write_parquet(OUT, compression="zstd")
    print(f"[완료] {OUT.name}: {out.height:,}행")


def detect_last() -> date:
    """마지막 완결 날짜 = cell_day 최신 날짜 − 1일 (최신 날짜는 아직 일부만 들어온 경우가 많음)."""
    mx = pl.scan_parquet(PROCESSED_DATA_DIR / "harmonized_cell_day.parquet").select(pl.col("date_local").max()).collect().item()
    return mx - timedelta(1)


def set_last(value: str | None) -> None:
    global LAST
    if value:
        LAST = detect_last() if value == "auto" else date.fromisoformat(value)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "samples", "folds", "select", "final"])
    ap.add_argument("--k", nargs="+", type=int, default=KS)
    ap.add_argument("--last", help="기준일 YYYY-MM-DD 또는 auto (cell_day 최신 − 1일). 기본 2026-11-14")
    a = ap.parse_args(argv)
    set_last(a.last)
    start_run_log("planner_daily")
    t0 = time.perf_counter()
    {"build": build, "samples": samples, "select": select, "final": final}.get(a.step, lambda: folds(a.k))()
    print(f"[{a.step}] {time.perf_counter() - t0:,.0f}s, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
