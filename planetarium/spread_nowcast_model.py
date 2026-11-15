"""다음 날 확산 예측 2·3단계: 기상 특성 결합 → 기준선·XGBoost leave-year-out → 지표·보정·사례 → 최종 모델·최신 예측.

표현: "위성 감지 기반 단기 확산 가능성" (물리 시뮬레이션 아님). 평가는 후보 칸(오늘 활성 칸 반경 3칸) 안에서만 계산하며,
후보 칸이 덮는 '내일 활성 칸' 비율(재현율 상한: 한국 약 25%, 아르헨티나 약 40%)을 함께 보고한다.

단계
  features  표본(samples_*) + NASA POWER 기상(d일, 최근 3·7일 강수) + 바람 정렬 지수 → features_<region>_<year>.parquet
  train     leave-year-out(2012~2025). 시험 연도 y, 조기 종료 = 안쪽 검증 연도(y+1, 2025는 2024), 하이퍼파라미터 고정.
            학습 표본: 양성 전부 + 음성 표본(아르헨티나 10%, 한국 100%), 음성 가중치 = 1/표본 비율 (확률 보정 유지).
            모델: ① 지속(오늘 활성 → 내일 활성) ② 이웃 규칙(3×3 안에 활성) ③ 로지스틱 회귀(log1p·표준화, 학습 표본 중 150만 행)
                  ④ XGBoost 분류(GPU). fold마다 OOF 예측을 체크포인트로 저장(--resume).
  report    지표(PR-AUC, ROC-AUC, Brier, 정밀도@재현율 0.5·0.7; 한국 상자·KOR만·아르헨티나), 보정 그림, 의성 사례 그림.
  final     전 기간 최종 모델(반복 수 = fold 최적 반복 중앙값) → models/spread_nowcast_xgb.json, models/spread_nowcast_meta.json
  latest    최근 관측일(NOAA-20 status=ok인 마지막 날) 기준 칸별 내일 확률 → data/processed/spread_nowcast_latest.parquet

실행  uv run python -m planetarium.spread_nowcast_model features|train|report|final|latest [--resume]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, SEED
from planetarium.runlog import peak_memory_gb, start_run_log
from planetarium.spread_nowcast import DIRS, OUT_DIR, START
from planetarium.weather_power import DAILY_PATH, GRID

YEARS = list(range(2012, 2026))
NEG_RATE = {"ARG": 0.10, "KR": 1.0}
LR_ROWS = 1_500_000
OOF_DIR = OUT_DIR / "oof"
MODEL_PATH = MODELS_DIR / "spread_nowcast_xgb.json"
META_PATH = MODELS_DIR / "spread_nowcast_meta.json"
LATEST_PATH = PROCESSED_DATA_DIR / "spread_nowcast_latest.parquet"

FIRE = ["self_n", "self_frp", "n3", "frp3", "n7", "frp7", "frpmax7", "cells7", "self_streak", "streak7_max", "near_km",
        "prev_self_n", "prev_self_frp", "prev_n3", "prev_frp3", "prev_n7", "prev_frp7", "prev_frpmax7", "prev_cells7", "prev_ok",
        *[f"dir_{d}" for d in DIRS], "spread_mag"]
WEATHER = ["ws", "ws_max", "rh", "rh_min", "t_max", "prcp", "prcp3", "prcp7", "wind_align", "wind_push"]
CONTEXT = ["month_sin", "month_cos", "cell_lat", "cell_lon", "is_kr"]
FEATURES = FIRE + WEATHER + CONTEXT
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "device": "cuda", "tree_method": "hist",
          "max_depth": 8, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
          "lambda": 1.0, "max_bin": 256, "seed": SEED}
MAX_ROUNDS, EARLY_STOP = 3000, 100


def feat_path(region: str, year: int):
    return OUT_DIR / f"features_{region}_{year}.parquet"


def inner_year(y: int) -> int:
    return y + 1 if y < YEARS[-1] else y - 1


# ── features ─────────────────────────────────────────────────────────────

def weather_table() -> pl.DataFrame:
    w = pl.read_parquet(DAILY_PATH).sort("region", "glat", "glon", "date")
    g = ["region", "glat", "glon"]
    return w.with_columns(
        prcp3=pl.col("PRECTOTCORR").rolling_sum(3, min_samples=1).over(g),
        prcp7=pl.col("PRECTOTCORR").rolling_sum(7, min_samples=1).over(g),
    ).select(*g, "date", ws="WS10M", ws_max="WS10M_MAX", wd="WD10M", rh="RH2M", rh_min="RH2M_MIN_EST", t_max="T2M_MAX",
             prcp="PRECTOTCORR", prcp3="prcp3", prcp7="prcp7")


def add_wind(df: pl.DataFrame) -> pl.DataFrame:
    """바람 정렬 지수: 확산 벡터(활성 칸 → 후보 칸)와 바람이 불어 가는 방향(WD10M + 180°)의 cos. 확산 벡터가 0이면 0."""
    to = (pl.col("wd") + 180).radians()
    wx, wy = to.sin(), to.cos()
    mag = (pl.col("sx") ** 2 + pl.col("sy") ** 2).sqrt()
    align = pl.when(mag > 0).then((pl.col("sx") * wx + pl.col("sy") * wy) / mag).otherwise(0.0)
    return df.with_columns(spread_mag=mag.cast(pl.Float32)).with_columns(
        wind_align=align.cast(pl.Float32), wind_push=(align * pl.col("ws")).cast(pl.Float32))


def features(force: bool) -> None:
    w = weather_table()
    for region in ("KR", "ARG"):
        wr = w.filter(pl.col("region") == region).drop("region")
        for y in tqdm(YEARS, desc=f"특성 {region}", unit="year", dynamic_ncols=True):
            out = feat_path(region, y)
            if out.exists() and not force:
                continue
            s = pl.read_parquet(OUT_DIR / f"samples_{region}_{y}.parquet")
            s = s.with_columns(
                glat=((pl.col("cell_lat").cast(pl.Float64) / GRID).floor() * GRID + GRID / 2).round(3),
                glon=((pl.col("cell_lon").cast(pl.Float64) / GRID).floor() * GRID + GRID / 2).round(3),
            ).join(wr.rename({"date": "date_local"}), on=["glat", "glon", "date_local"], how="left")
            s = add_wind(s).with_columns(is_kr=pl.lit(1 if region == "KR" else 0, dtype=pl.Int8),
                                         prev_ok=pl.col("prev_ok").cast(pl.Int8),
                                         u=pl.lit(np.random.default_rng(SEED + y).random(s.height)).cast(pl.Float32))
            keep = ["date_local", "iy", "ix", "region", "cand_adm0", "label", "u", *FEATURES]
            s.select(keep).write_parquet(out, compression="zstd")
    miss = pl.scan_parquet(OUT_DIR / "features_*.parquet").select(pl.col("ws").is_null().mean()).collect().item()
    print(f"[완료] 특성 파일. 기상 결측 비율 {miss:.4%}")


# ── train (leave-year-out) ───────────────────────────────────────────────

def scan_features(years: list[int]) -> pl.LazyFrame:
    files = [feat_path(r, y) for r in ("KR", "ARG") for y in years]
    return pl.scan_parquet(files)


def train_rows(years: list[int]) -> pl.DataFrame:
    rate = pl.col("region").replace_strict(NEG_RATE, return_dtype=pl.Float32)
    return (scan_features(years).filter((pl.col("label") == 1) | (pl.col("u") < rate))
            .with_columns(w=pl.when(pl.col("label") == 1).then(1.0).otherwise(1.0 / rate).cast(pl.Float32))
            .collect())


def X(df: pl.DataFrame) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in FEATURES]).to_numpy()


LR_COUNT = [c for c in FIRE if c not in ("prev_ok", "near_km", "spread_mag")] + ["prcp", "prcp3", "prcp7"]


def lr_X(df: pl.DataFrame) -> np.ndarray:
    cols = []
    for c in FEATURES:
        e = pl.col(c).cast(pl.Float64)
        if c in LR_COUNT or c == "spread_mag":
            e = e.fill_null(0).clip(0, None).log1p()
        elif c == "near_km":
            e = e.fill_null(10.0)
        else:
            e = e.fill_null(0)
        cols.append(e.alias(c))
    return df.select(cols).to_numpy()


def fit_xgb(tr: pl.DataFrame, va: pl.DataFrame):
    import xgboost as xgb
    dtr = xgb.QuantileDMatrix(X(tr), label=tr["label"].to_numpy(), weight=tr["w"].to_numpy(), max_bin=PARAMS["max_bin"])
    dva = xgb.QuantileDMatrix(X(va), label=va["label"].to_numpy(), weight=va["w"].to_numpy(), ref=dtr)
    return xgb.train(PARAMS, dtr, MAX_ROUNDS, evals=[(dva, "val")], early_stopping_rounds=EARLY_STOP, verbose_eval=False)


def predict_xgb(bst, df: pl.DataFrame) -> np.ndarray:
    import xgboost as xgb
    best = bst.attributes().get("best_iteration")
    return bst.predict(xgb.DMatrix(X(df)), iteration_range=(0, int(best) + 1) if best is not None else (0, 0))


def train(force: bool) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    iters = {}
    bar = tqdm(YEARS, desc="leave-year-out", unit="fold", dynamic_ncols=True)
    for y in bar:
        out = OOF_DIR / f"oof_{y}.parquet"
        if out.exists() and not force:
            iters[y] = json.loads((OOF_DIR / f"iter_{y}.json").read_text())["best_iteration"]
            continue
        t0 = time.perf_counter()
        v = inner_year(y)
        tr = train_rows([x for x in YEARS if x not in (y, v)])
        va = train_rows([v])
        bst = fit_xgb(tr, va)
        # 로지스틱 회귀 (학습 표본 중 일부)
        sub = tr.sample(min(LR_ROWS, tr.height), seed=SEED)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0))
        lr.fit(lr_X(sub), sub["label"].to_numpy(), logisticregression__sample_weight=sub["w"].to_numpy())
        del tr, va, sub
        te = scan_features([y]).collect()
        res = te.select("date_local", "iy", "ix", "region", "cand_adm0", "label").with_columns(
            p_xgb=pl.Series(predict_xgb(bst, te), dtype=pl.Float32),
            p_lr=pl.Series(lr.predict_proba(lr_X(te))[:, 1], dtype=pl.Float32),
            persist=(te["self_n"] > 0).cast(pl.Float32),
            neighbor=(te["n3"] > 0).cast(pl.Float32),
        )
        res.write_parquet(out, compression="zstd")
        iters[y] = bst.best_iteration
        (OOF_DIR / f"iter_{y}.json").write_text(json.dumps({"best_iteration": bst.best_iteration, "inner_year": v}))
        tqdm.write(f"[fold {y}] 안쪽 {v}, iter {bst.best_iteration}, 시험 {te.height:,}행, {time.perf_counter() - t0:,.0f}초, "
                   f"최대 메모리 {peak_memory_gb() or 0:.1f} GB")
        del te, res
    print(f"[완료] leave-year-out {len(iters)} fold, 최적 반복 {iters}")


# ── report ───────────────────────────────────────────────────────────────

MODELS = {"persist": "① persistence", "neighbor": "② neighbor rule", "p_lr": "③ logistic", "p_xgb": "④ XGBoost"}


def precision_at_recall(y: np.ndarray, p: np.ndarray, r: float) -> float:
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y, p)
    ok = rec >= r
    return float(prec[ok].max()) if ok.any() else float("nan")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    if y.sum() == 0 or y.sum() == len(y):
        return {}
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            "prec_at_rec_0.5": precision_at_recall(y, p, 0.5), "prec_at_rec_0.7": precision_at_recall(y, p, 0.7),
            "n": len(y), "positives": int(y.sum()), "base_rate": float(y.mean())}


def groups() -> dict[str, pl.Expr]:
    return {"KR_box": pl.col("region") == "KR", "KOR_only": (pl.col("region") == "KR") & (pl.col("cand_adm0") == "KOR"),
            "ARG": pl.col("region") == "ARG"}


def ceiling() -> pl.DataFrame:
    logs = sorted(OUT_DIR.glob("samples_build_log_*.csv"))
    lg = pl.read_csv(logs[-1])
    return lg.group_by("region").agg(next_active=pl.col("next_active_cells").sum(), covered=pl.col("next_active_covered").sum()).with_columns(
        recall_ceiling=pl.col("covered") / pl.col("next_active"))


def report() -> None:
    oof = pl.read_parquet(sorted(OOF_DIR.glob("oof_*.parquet")))
    rows = []
    for gname, g in groups().items():
        d = oof.filter(g)
        for scope, dd in [("pooled", d)] + [(str(y), d.filter(pl.col("date_local").dt.year() == y)) for y in YEARS]:
            yv = dd["label"].to_numpy()
            for m in MODELS:
                mm = metrics(yv, dd[m].to_numpy())
                if mm:
                    rows.append({"group": gname, "scope": scope, "model": m, **mm})
    met = pl.DataFrame(rows)
    met.write_csv(REPORTS_DIR / "spread_nowcast_metrics.csv")
    ceil = ceiling()
    ceil.write_csv(REPORTS_DIR / "spread_nowcast_recall_ceiling.csv")
    imp = None
    if MODEL_PATH.exists():
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(MODEL_PATH)
        sc = b.get_score(importance_type="gain")
        imp = pl.DataFrame({"feature": [FEATURES[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k for k in sc],
                            "gain": list(sc.values())}).sort("gain", descending=True)
        imp.write_csv(REPORTS_DIR / "spread_nowcast_importance.csv")
    plot_calibration(oof)
    plot_case(oof)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=4, thousands_separator=","):
        print(met.filter(pl.col("scope") == "pooled").drop("scope"))
        print(ceil)
        if imp is not None:
            print(imp.head(15))


def plot_calibration(oof: pl.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    bins = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    for ax, (gname, g) in zip(axes, groups().items()):
        d = oof.filter(g)
        for m, color in (("p_xgb", "#08519c"), ("p_lr", "#d94801")):
            p, y = d[m].to_numpy(), d["label"].to_numpy()
            k = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
            mp = np.array([p[k == i].mean() if (k == i).sum() > 50 else np.nan for i in range(len(bins) - 1)])
            my = np.array([y[k == i].mean() if (k == i).sum() > 50 else np.nan for i in range(len(bins) - 1)])
            ax.plot(mp, my, marker="o", color=color, label=MODELS[m])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{gname} (n={d.height:,}, base rate {d['label'].mean():.2%})", fontsize=10)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency (next day)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Spread nowcast calibration (leave-year-out, candidate cells)", fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "spread_nowcast_calibration.png", dpi=130)
    plt.close(fig)


def plot_case(oof: pl.DataFrame) -> None:
    """한국 2025-03-22~28 (의성·안동): 날짜별 예측 확률 지도 vs 다음 날 실제 감지."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from planetarium.spread_nowcast import RES, load_points
    x0, x1, y0, y1 = 128.3, 129.6, 36.0, 36.85
    days = [date(2025, 3, 22) + timedelta(days=i) for i in range(7)]
    d = oof.filter(pl.col("region") == "KR", pl.col("date_local").is_in(days)).with_columns(
        lat=(pl.col("iy").cast(pl.Float64) + 0.5) * RES - 90, lon=(pl.col("ix").cast(pl.Float64) + 0.5) * RES - 180)
    pts = load_points("KR", days[0], days[-1] + timedelta(days=1))
    pts = pts.with_columns(lat=(pl.col("iy").cast(pl.Float64) + 0.5) * RES - 90, lon=(pl.col("ix").cast(pl.Float64) + 0.5) * RES - 180)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.4))
    sc = None
    for ax, day in zip(axes.flat, days):
        dd = d.filter(pl.col("date_local") == day, pl.col("lat").is_between(y0, y1), pl.col("lon").is_between(x0, x1))
        today = pts.filter(pl.col("date_local") == day).unique(["iy", "ix"])
        nxt = pts.filter(pl.col("date_local") == day + timedelta(days=1)).unique(["iy", "ix"])
        sc = ax.scatter(dd["lon"], dd["lat"], c=dd["p_xgb"], s=9, marker="s", cmap="YlOrRd", vmin=0, vmax=0.6)
        ax.scatter(today["lon"], today["lat"], s=6, c="k", marker=".", label="today detections")
        ax.scatter(nxt["lon"], nxt["lat"], s=22, facecolors="none", edgecolors="#08519c", linewidths=0.8, label="next-day detections")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        y, p = dd["label"].to_numpy(), dd["p_xgb"].to_numpy()
        ax.set_title(f"{day} → {day + timedelta(days=1)}  cand {dd.height}, pos {int(y.sum())}", fontsize=9)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7, loc="lower left")
    axes.flat[-1].axis("off")
    fig.colorbar(sc, ax=axes.flat[-1], fraction=0.8, label="P(detection next local day) — XGBoost, fold 2025 (held out)")
    fig.suptitle("Korea 2025-03-22 ~ 28 (Uiseong–Andong): predicted next-day probability vs next-day detections", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spread_nowcast_case_uiseong_2025.png", dpi=130)
    plt.close(fig)


# ── final model ──────────────────────────────────────────────────────────

def final() -> None:
    import xgboost as xgb
    iters = [json.loads((OOF_DIR / f"iter_{y}.json").read_text())["best_iteration"] for y in YEARS]
    n = int(np.median(iters)) + 1
    tr = train_rows(YEARS)
    d = xgb.QuantileDMatrix(X(tr), label=tr["label"].to_numpy(), weight=tr["w"].to_numpy(), max_bin=PARAMS["max_bin"])
    bst = xgb.train({k: v for k, v in PARAMS.items() if k != "eval_metric"}, d, n)
    bst.save_model(MODEL_PATH)
    met = pl.read_csv(REPORTS_DIR / "spread_nowcast_metrics.csv").filter(pl.col("scope") == "pooled")
    ceil = ceiling()
    meta = {
        "name": "spread_nowcast_xgb",
        "description": "위성 감지 기반 단기 확산 가능성 (물리 시뮬레이션 아님): 오늘(현지 날짜) 활성 칸 반경 3칸(0.025° 칸) 안의 칸에 "
                       "다음 현지 날짜 감지가 생길 확률",
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "grid_deg": 0.025, "candidate_radius_cells": 3, "label": "다음 현지 날짜에 그 칸에 기준 점(VIIRS S-NPP) 감지가 있는가",
        "regions": {"KR": "상자 124.5~132E, 33~38.7N (일본·북한 일부 포함; 평가는 상자 전체와 KOR만)", "ARG": "아르헨티나 국경 (adm0 ARG 점)"},
        "train_period": [str(START), "2025-12-31"],
        "data_status_rule": "S-NPP satellite_day_status가 d와 d+1 모두 ok인 날만",
        "features": FEATURES,
        "weather": {"source": "NASA POWER Daily API v2.10.3 (MERRA-2, 0.5°, LST), community=AG",
                    "variables": {"ws": "WS10M", "ws_max": "WS10M_MAX", "wd": "WD10M (불어오는 방향)", "rh": "RH2M (일평균)",
                                  "rh_min": "RH2M_MIN_EST", "t_max": "T2M_MAX", "prcp": "PRECTOTCORR", "prcp3": "최근 3일 합", "prcp7": "최근 7일 합"},
                    "rh_min_method": "RH_min ≈ 100·e_s(T2MDEW)/e_s(T2M_MAX), e_s(T)=6.112·exp(17.62·T/(243.12+T)) hPa (Magnus, WMO 2008). "
                                     "이슬점이 하루 중 거의 일정하고 최저 습도가 최고 기온 때 나타난다는 가정. POWER는 일 최저 상대습도를 주지 않음.",
                    "wind_alignment": "cos(확산 벡터, 바람이 불어 가는 방향 WD10M+180°). 확산 벡터 = Σ 활성 칸→후보 칸 방향 × 감지 수 / 거리². wind_push = 정렬 × 풍속",
                    "attribution": "docs/references.md 5-1"},
        "sampling": {"negative_rate": NEG_RATE, "negative_weight": "1/표본 비율"},
        "params": PARAMS, "n_rounds": n, "fold_best_iterations": dict(zip(map(str, YEARS), iters)),
        "validation": "leave-year-out 2012~2025, 조기 종료 = 안쪽 검증 연도(y+1, 2025는 2024), 후보 칸 안에서 평가",
        "metrics_pooled": met.to_dicts(),
        "recall_ceiling": ceil.to_dicts(),
        "caveats": ["평가·확률은 후보 칸(오늘 활성 칸 반경 3칸) 안에서만 의미가 있다. 반경 밖 새 발화는 예측 대상이 아니다.",
                    "학습 라벨은 S-NPP. 2026-11 이후 최신 예측은 NOAA-20 감지로 특성을 만든다 (해상도 같은 VIIRS 375 m).",
                    "위성 관측 시각(하루 약 2회) 사이의 확산은 보이지 않는다. 구름 아래 감지 누락은 라벨 0으로 들어간다."],
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[완료] {MODEL_PATH.name} (반복 {n}), {META_PATH.name}")


# ── latest ───────────────────────────────────────────────────────────────

def latest() -> None:
    """NOAA-20 status=ok인 마지막 날 d 기준, 후보 칸별 d+1 확률. 기상은 NASA POWER에서 그 지점·기간만 새로 받는다."""
    import xgboost as xgb

    from planetarium.monthly import STATUS_PATH
    from planetarium.spread_nowcast import RES, REGIONS, add_streak, cell_adm0, cell_days, load_points, neighborhood
    from planetarium.weather_power import API, CACHE_DIR, PARAMS as WPARAMS, rh_min_expr
    st = pl.read_parquet(STATUS_PATH).filter(pl.col("satellite") == "NOAA-20")
    d = st.filter(pl.col("status") == "ok")["date_utc"].max()
    prev_ok = st.filter(pl.col("date_utc") == d - timedelta(days=1), pl.col("status") == "ok").height > 0
    adm = cell_adm0()
    parts = []
    for region in REGIONS:
        pts = load_points(region, d - timedelta(days=30), d, sat="N20")
        if pts.height == 0:
            continue
        cd = add_streak(cell_days(pts))
        act = cd.filter(pl.col("src"), pl.col("date_local") == d)
        if act.height == 0:
            continue
        today = neighborhood(act, "")
        prev = neighborhood(cd.filter(pl.col("src"), pl.col("date_local") == d - timedelta(days=1))
                            .with_columns(date_local=pl.col("date_local") + pl.duration(days=1)), "prev_")
        s = today.join(prev, on=["date_local", "iy", "ix"], how="left")
        lat = (pl.col("iy").cast(pl.Float64) + 0.5) * RES - 90
        lon = (pl.col("ix").cast(pl.Float64) + 0.5) * RES - 180
        m = pl.col("date_local").dt.month().cast(pl.Float64)
        s = s.with_columns(cell_lat=lat.cast(pl.Float32), cell_lon=lon.cast(pl.Float32),
                           cell_id01=(((lat + 90) / 0.1).floor() * 3600 + ((lon + 180) / 0.1).floor()).cast(pl.Int64),
                           month_sin=(2 * np.pi * m / 12).sin().cast(pl.Float32), month_cos=(2 * np.pi * m / 12).cos().cast(pl.Float32),
                           region=pl.lit(region), prev_ok=pl.lit(int(prev_ok), dtype=pl.Int8),
                           is_kr=pl.lit(1 if region == "KR" else 0, dtype=pl.Int8)
                           ).join(adm.rename({"cell_id": "cell_id01", "adm0_a3": "cand_adm0"}), on="cell_id01", how="left").drop("cell_id01")
        fill0 = [c for c in s.columns if c.startswith("prev_") and c != "prev_ok"]
        parts.append(s.with_columns(pl.col(fill0).fill_null(0)))
    s = pl.concat(parts, how="diagonal_relaxed")
    # 기상: 필요한 0.5° 지점만 NASA POWER에서 d-6 ~ d. POWER 근실시간은 약 3일 늦으므로 지점마다 가장 최근 값이 있는 날을 쓰고
    # weather_date·weather_lag_days로 남긴다 (강수 3·7일 합도 그날 기준).
    s = s.with_columns(glat=((pl.col("cell_lat").cast(pl.Float64) / GRID).floor() * GRID + GRID / 2).round(3),
                       glon=((pl.col("cell_lon").cast(pl.Float64) / GRID).floor() * GRID + GRID / 2).round(3))
    import urllib.request
    rows = []
    cache = CACHE_DIR.parent / "power_recent" / d.isoformat()
    cache.mkdir(parents=True, exist_ok=True)
    for glat, glon in tqdm(s.select("glat", "glon").unique().iter_rows(), desc="NASA POWER 최근", unit="point"):
        f = cache / f"{glat:.3f}_{glon:.3f}.json"
        if not f.exists():
            url = (f"{API}?parameters={','.join(WPARAMS)}&community=AG&longitude={glon}&latitude={glat}"
                   f"&start={(d - timedelta(days=6)):%Y%m%d}&end={d:%Y%m%d}&format=JSON&time-standard=LST")
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "planetarium-spread-nowcast/1.0"}), timeout=120) as r:
                f.write_bytes(r.read())
        p = json.loads(f.read_text(encoding="utf-8"))["properties"]["parameter"]
        days = sorted(p[WPARAMS[0]])
        rows.append(pl.DataFrame({"date": days, **{k: [p[k].get(x) for x in days] for k in WPARAMS}}).with_columns(glat=pl.lit(glat), glon=pl.lit(glon)))
    w = pl.concat(rows).with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"),
                                     *[pl.when(pl.col(k) <= -998).then(None).otherwise(pl.col(k)).cast(pl.Float32).alias(k) for k in WPARAMS])
    w = w.with_columns(RH2M_MIN_EST=rh_min_expr().cast(pl.Float32)).sort("glat", "glon", "date").with_columns(
        prcp3=pl.col("PRECTOTCORR").rolling_sum(3, min_samples=1).over("glat", "glon"),
        prcp7=pl.col("PRECTOTCORR").rolling_sum(7, min_samples=1).over("glat", "glon"),
    ).filter(pl.col("WS10M").is_not_null()).sort("date").group_by("glat", "glon").last().select(
        "glat", "glon", weather_date="date", ws="WS10M", ws_max="WS10M_MAX", wd="WD10M", rh="RH2M", rh_min="RH2M_MIN_EST",
                                         t_max="T2M_MAX", prcp="PRECTOTCORR", prcp3="prcp3", prcp7="prcp7")
    s = add_wind(s.join(w, on=["glat", "glon"], how="left"))
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)
    p = bst.predict(xgb.DMatrix(X(s)))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    out = s.select(
        pl.col("date_local").alias("as_of_date"), (pl.col("date_local") + pl.duration(days=1)).alias("target_date"),
        "region", "cand_adm0", "cell_lat", "cell_lon", pl.lit(RES).alias("cell_size_deg"),
        p_spread=pl.Series(p, dtype=pl.Float32), active_today=(pl.col("self_n") > 0), n_det_today_7x7="n7",
        near_active_km="near_km", wind_align="wind_align", weather_available=pl.col("ws").is_not_null(),
        weather_date="weather_date",
        weather_lag_days=(pl.col("date_local") - pl.col("weather_date")).dt.total_days().cast(pl.Int16),
    ).with_columns(model=pl.lit(f"spread_nowcast_xgb {meta['created']}"), source_satellite=pl.lit("NOAA-20"))
    out.sort("region", "p_spread", descending=[False, True]).write_parquet(LATEST_PATH, compression="zstd")
    print(f"[완료] {LATEST_PATH.name}: 기준일 {d} → 예측일 {d + timedelta(days=1)}, 후보 칸 {out.height:,} "
          f"(기상 있음 {out['weather_available'].mean():.0%}), 지역별 {dict(out.group_by('region').len().iter_rows())}")


# ── FWI 특성 비교 (채택 여부는 따로 결정; 기존 모델·OOF는 건드리지 않음) ────────

FWI_FEATURES = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
OOF_FWI_DIR = OUT_DIR / "oof_fwi"


def compare_fwi(force: bool) -> None:
    """같은 표본·같은 fold·같은 설정으로 FWI 6개 특성을 더한 XGBoost를 학습해 기존 OOF와 지표를 비교한다."""
    import xgboost as xgb

    from planetarium.fwi import FWI_PATH
    feats = FEATURES + FWI_FEATURES
    fwi = pl.scan_parquet(FWI_PATH).select("region", "glat", "glon", pl.col("date").alias("date_local"), *FWI_FEATURES)

    def with_fwi(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(
            glat=((pl.col("cell_lat").cast(pl.Float64) / GRID).floor() * GRID + GRID / 2).round(3),
            glon=((pl.col("cell_lon").cast(pl.Float64) / GRID).floor() * GRID + GRID / 2).round(3),
        ).join(fwi, on=["region", "glat", "glon", "date_local"], how="left")

    def rows(years):
        rate = pl.col("region").replace_strict(NEG_RATE, return_dtype=pl.Float32)
        return with_fwi(scan_features(years).filter((pl.col("label") == 1) | (pl.col("u") < rate))).with_columns(
            w=pl.when(pl.col("label") == 1).then(1.0).otherwise(1.0 / rate).cast(pl.Float32)).collect()

    def mat(df):
        return df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    OOF_FWI_DIR.mkdir(parents=True, exist_ok=True)
    gains = None
    for y in tqdm(YEARS, desc="leave-year-out (+FWI)", unit="fold", dynamic_ncols=True):
        out = OOF_FWI_DIR / f"oof_{y}.parquet"
        if out.exists() and not force:
            continue
        t0 = time.perf_counter()
        v = inner_year(y)
        tr, va = rows([x for x in YEARS if x not in (y, v)]), rows([v])
        dtr = xgb.QuantileDMatrix(mat(tr), label=tr["label"].to_numpy(), weight=tr["w"].to_numpy(), max_bin=PARAMS["max_bin"])
        dva = xgb.QuantileDMatrix(mat(va), label=va["label"].to_numpy(), weight=va["w"].to_numpy(), ref=dtr)
        bst = xgb.train(PARAMS, dtr, MAX_ROUNDS, evals=[(dva, "val")], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        del tr, va, dtr, dva
        te = with_fwi(scan_features([y])).collect()
        p = bst.predict(xgb.DMatrix(mat(te)), iteration_range=(0, bst.best_iteration + 1))
        te.select("date_local", "iy", "ix", "region", "cand_adm0", "label").with_columns(
            p_xgb_fwi=pl.Series(p, dtype=pl.Float32)).write_parquet(out, compression="zstd")
        if y == YEARS[-1]:
            sc = bst.get_score(importance_type="gain")
            gains = pl.DataFrame({"feature": [feats[int(k[1:])] for k in sc], "gain": list(sc.values())}).sort("gain", descending=True)
        tqdm.write(f"[fold {y} +FWI] iter {bst.best_iteration}, {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.1f} GB")
    base = pl.read_parquet(sorted(OOF_DIR.glob("oof_*.parquet"))).select("date_local", "iy", "ix", "region", "label", "cand_adm0", "p_xgb")
    new = pl.read_parquet(sorted(OOF_FWI_DIR.glob("oof_*.parquet"))).select("date_local", "iy", "ix", "region", "p_xgb_fwi")
    j = base.join(new, on=["date_local", "iy", "ix", "region"], how="inner")
    rows_ = []
    for gname, g in groups().items():
        d = j.filter(g)
        for m in ("p_xgb", "p_xgb_fwi"):
            rows_.append({"group": gname, "model": m, **metrics(d["label"].to_numpy(), d[m].to_numpy())})
    comp = pl.DataFrame(rows_)
    comp.write_csv(REPORTS_DIR / "spread_nowcast_fwi_comparison.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=4, thousands_separator=","):
        print(comp)
        if gains is not None:
            gains.write_csv(REPORTS_DIR / "spread_nowcast_fwi_importance_fold2025.csv")
            print(gains.head(20))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="확산 예측 2·3단계")
    ap.add_argument("step", choices=["features", "train", "report", "final", "latest", "compare_fwi", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true", help="기본 동작 (만든 파일·fold는 건너뜀)")
    args = ap.parse_args(argv)
    start_run_log("spread_nowcast_model")
    t0 = time.perf_counter()
    steps = ["features", "train", "report", "final", "report"] if args.step == "all" else [args.step]
    for s in steps:
        print(f"===== {s} =====", flush=True)
        {"features": lambda: features(args.force), "train": lambda: train(args.force), "report": report,
         "final": final, "latest": latest, "compare_fwi": lambda: compare_fwi(args.force)}[s]()
        print(f"[{s}] 끝 ({time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
