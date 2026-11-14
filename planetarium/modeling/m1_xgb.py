"""M1 비교 검증: B1(위도띠×월 비율) · B1c(국가×월 비율) · B2(로그 선형) · M1(XGBoost GPU) leave-year-out.

모델
  B1   학습 연도의 Σ S-NPP / Σ (Aqua+Terra), (위도 5° 띠 × 월). 분모 0이면 월만. (train_m1.py)
  B1c  같은 비율을 (국가 × 월)로. 그 국가·월의 학습 기간 MODIS 감지 합이 B1C_MIN_DEN 미만이면
       UN 하위지역 × 월 → 위도띠 × 월 → 월 순으로 대체.
  B2   log1p(y) ~ log1p(aqua) + log1p(terra) + month_sin + month_cos + lat + lon (Ridge)
  M1   XGBoost (device="cuda", tree_method="hist"). 목적함수 reg:tweedie(variance_power 1.1~1.7) 또는 count:poisson 중
       안쪽 검증 Poisson deviance가 가장 작은 것. 범주형 특성: 국가(adm0_a3)·UN 하위지역·대륙.
       분위수: reg:quantileerror, quantile_alpha [0.1, 0.9] (한 모델이 두 분위수를 함께 출력).
  M1_terra  Aqua 특성을 뺀 같은 모델 (2000-11 ~ 2002-06 적용용).

학습 표본 가중치
  학습 표는 positive 25%, 0행 약 4%만 뽑은 표본이라 그대로 학습하면 '자기 칸 0, 주변 감지 있음' 같은 행의
  비중이 실제보다 작아져 예측이 위로 치우친다. sample_weight(= 1/표본 비율)로 모집단 비율을 되살린다.

검증 (결과를 좋게 보이려고 지표·기간을 바꾸지 않음)
  - leave-year-out 2012~2025. 조기 종료는 시험 연도가 아닌 '안쪽 검증 연도'(시험 연도+1, 2025는 2024)로.
  - 목적함수 선택도 안쪽 검증 연도만 사용 (4개 선택 fold: 2013, 2017, 2021, 2025의 안쪽 검증 Poisson deviance 평균).
  - 칸·월 지표: 학습 표(표본) 행.
  - 지역·월 합계: 전체 칸(2012~2025 감지 이력 칸 ∪ 한국 칸) × 정상 월.
      zero rule 켬(+zr): MODIS 감지와 주변 3×3 감지가 모두 0이면 예측 0 → 희소 패널 합 = 전체 칸 합.
      zero rule 끔: 희소 패널 + 패널에 없는 모든 (칸, 월)에 특성 0으로 예측해 더한다.
  - 지역: world, 대륙, korea, argentina.
채택 규칙: 한국 월 합계(전 기간 OOF) MAE와 |편향-1| 둘 다 M1(더 나은 zero rule 변형)이 B1보다 나아야 M1 채택.

실행  uv run python -m planetarium.modeling.m1_xgb
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED
from planetarium.modeling.evaluate import cell_metrics, region_masks, total_metrics
from planetarium.modeling.train_m1 import (
    FOLD_YEARS, META_PATH, OOF_PATH, b1_fit, b1_predict, b1_sums, lat_band, plot_korea_monthly, plot_yearly_bias,
    pooled_rows, region_month_totals, summarize,
)
from planetarium.runlog import start_run_log
from planetarium.train_table import (
    GRID_PATH, NUMERIC_FEATURES, TRAIN_PATH, load_panel, panel_months,
)

CAT_FEATURES = ["adm0_a3", "subregion", "continent"]
M1_FEATURES = NUMERIC_FEATURES + CAT_FEATURES
M1_TERRA_FEATURES = [f for f in M1_FEATURES if not f.startswith("aqua_")]
SELECT_FOLDS = [2013, 2017, 2021, 2025]
OBJECTIVES = {
    "tweedie_1.1": {"objective": "reg:tweedie", "tweedie_variance_power": 1.1},
    "tweedie_1.3": {"objective": "reg:tweedie", "tweedie_variance_power": 1.3},
    "tweedie_1.5": {"objective": "reg:tweedie", "tweedie_variance_power": 1.5},
    "tweedie_1.7": {"objective": "reg:tweedie", "tweedie_variance_power": 1.7},
    "poisson": {"objective": "count:poisson", "max_delta_step": 0.7},
}
BASE_PARAMS = {
    "device": "cuda", "tree_method": "hist", "max_bin": 256, "learning_rate": 0.05, "max_depth": 8,
    "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.9, "lambda": 1.0,
    "max_cat_to_onehot": 1, "max_cat_threshold": 64, "seed": SEED,
}
MAX_ROUNDS = 4000
# 분위수 모델은 수렴이 느려(시험: 4,000회에서도 계속 개선, 380초) 학습률 0.1, 최대 2,000회로 둔다.
Q_PARAMS = {"objective": "reg:quantileerror", "quantile_alpha": np.array([0.1, 0.9]), "learning_rate": 0.1}
Q_MAX_ROUNDS = 2000
EARLY_STOP = 100
B1C_MIN_DEN = 200
B1C_PATH = MODELS_DIR / "b1c_ratios.parquet"
M1_MODEL_PATH = MODELS_DIR / "m1_xgb.json"
M1_Q_PATH = MODELS_DIR / "m1_q10_q90_xgb.json"
M1_TERRA_PATH = MODELS_DIR / "m1_terra_xgb.json"
B2_PATH = MODELS_DIR / "b2_ridge.json"
REGION_KEYS = ["year", "month", "is_korea", "adm0_a3", "continent"]


def inner_year(y: int) -> int:
    return y + 1 if y < FOLD_YEARS[-1] else y - 1


# ── 범주형 인코딩 ─────────────────────────────────────────────────────────

def category_maps() -> dict[str, list[str]]:
    g = pl.read_parquet(GRID_PATH, columns=CAT_FEATURES)
    return {c: sorted(set(g[c].fill_null("none").unique().to_list()) | {"none"}) for c in CAT_FEATURES}


def to_matrix(df: pl.DataFrame, feats: list[str], cats: dict[str, list[str]]) -> np.ndarray:
    cols = []
    for f in feats:
        if f in cats:
            cols.append(pl.col(f).fill_null("none").replace_strict({v: i for i, v in enumerate(cats[f])}, default=0,
                                                                   return_dtype=pl.Float32).alias(f))
        else:
            cols.append(pl.col(f).cast(pl.Float32))
    return df.select(cols).to_numpy()


def feature_types(feats: list[str]) -> list[str]:
    return ["c" if f in CAT_FEATURES else "q" for f in feats]


# ── XGBoost ───────────────────────────────────────────────────────────────

def xgb_train(tr: pl.DataFrame, va: pl.DataFrame, feats: list[str], cats: dict, obj: dict, quantile: bool = False):
    import xgboost as xgb
    ft = feature_types(feats)
    dtr = xgb.QuantileDMatrix(to_matrix(tr, feats, cats), label=tr["y"].to_numpy(), weight=tr["w"].to_numpy(),
                              feature_types=ft, enable_categorical=True, max_bin=BASE_PARAMS["max_bin"])
    dva = xgb.QuantileDMatrix(to_matrix(va, feats, cats), label=va["y"].to_numpy(), weight=va["w"].to_numpy(),
                              feature_types=ft, enable_categorical=True, ref=dtr)
    params = {**BASE_PARAMS, **(Q_PARAMS if quantile else obj)}
    rounds = Q_MAX_ROUNDS if quantile else MAX_ROUNDS
    bst = xgb.train(params, dtr, rounds, evals=[(dva, "val")], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    return bst


def xgb_predict(bst, df: pl.DataFrame, feats: list[str], cats: dict) -> np.ndarray:
    import xgboost as xgb
    d = xgb.DMatrix(to_matrix(df, feats, cats), feature_types=feature_types(feats), enable_categorical=True)
    return bst.predict(d, iteration_range=(0, bst.best_iteration + 1))


def poisson_dev(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    p = np.clip(p, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 2 * (np.where(y > 0, y * np.log(y / p), 0.0) - (y - p))
    return float(np.average(d, weights=w))


# ── B1c, B2 ───────────────────────────────────────────────────────────────

def b1c_sums() -> pl.DataFrame:
    return (
        load_panel("train").select("year", "month", "cell_lat", "adm0_a3", "subregion", "aqua_n_det", "terra_n_det", "y")
        .with_columns(lat_band=lat_band(), adm0_a3=pl.col("adm0_a3").fill_null("none"), subregion=pl.col("subregion").fill_null("none"))
        .group_by("year", "adm0_a3", "subregion", "lat_band", "month")
        .agg(y=pl.col("y").sum(), m=(pl.col("aqua_n_det") + pl.col("terra_n_det")).sum(), t=pl.col("terra_n_det").sum())
        .collect()
    )


def b1c_fit(s: pl.DataFrame, years: list[int], den: str = "m") -> dict[str, pl.DataFrame]:
    s = s.filter(pl.col("year").is_in(years))
    out = {}
    for lvl in ("adm0_a3", "subregion", "lat_band"):
        g = s.group_by(lvl, "month").agg(num=pl.col("y").sum(), den=pl.col(den).sum())
        out[lvl] = g.filter(pl.col("den") >= B1C_MIN_DEN).select(lvl, "month", (pl.col("num") / pl.col("den")).alias(f"r_{lvl}"))
    g = s.group_by("month").agg(num=pl.col("y").sum(), den=pl.col(den).sum())
    out["month"] = g.select("month", (pl.col("num") / pl.col("den")).alias("r_month"))
    return out


def b1c_predict(df: pl.DataFrame, r: dict[str, pl.DataFrame], terra_only: bool = False) -> np.ndarray:
    d = df.select("adm0_a3", "subregion", "cell_lat", "month", "aqua_n_det", "terra_n_det").with_columns(
        lat_band=lat_band(), adm0_a3=pl.col("adm0_a3").fill_null("none"), subregion=pl.col("subregion").fill_null("none"))
    for lvl in ("adm0_a3", "subregion", "lat_band"):
        d = d.join(r[lvl], on=[lvl, "month"], how="left")
    d = d.join(r["month"], on="month", how="left")
    ratio = pl.coalesce("r_adm0_a3", "r_subregion", "r_lat_band", "r_month")
    modis = pl.col("terra_n_det") if terra_only else pl.col("aqua_n_det").fill_null(0) + pl.col("terra_n_det")
    return d.select((ratio * modis).fill_null(0))[:, 0].to_numpy()


B2_COLS = ["la", "lt", "month_sin", "month_cos", "cell_lat", "cell_lon"]


def b2_X(df: pl.DataFrame) -> np.ndarray:
    return df.select(la=pl.col("aqua_n_det").fill_null(0).log1p(), lt=pl.col("terra_n_det").log1p(),
                     month_sin="month_sin", month_cos="month_cos", cell_lat="cell_lat", cell_lon="cell_lon").to_numpy()


def b2_fit(tr: pl.DataFrame):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    m.fit(b2_X(tr), np.log1p(tr["y"].to_numpy()), ridge__sample_weight=tr["w"].to_numpy())
    return m


def b2_predict(m, df: pl.DataFrame) -> np.ndarray:
    return np.clip(np.expm1(m.predict(b2_X(df))), 0, None)


# ── 전체 칸 (zero rule 끔용) ──────────────────────────────────────────────

def dense_zero_rows(year: int, grid: pl.DataFrame, panel_keys: pl.DataFrame) -> pl.DataFrame:
    """패널에 없는 (격자 칸 × 정상 월): MODIS·주변 감지·타깃 모두 0인 행."""
    months = panel_months().filter(pl.col("mode") == "train", ~pl.col("excluded"), pl.col("year") == year).select("year", "month")
    obs = load_panel("train", [year]).select("year", "month", "aqua_obs_days", "terra_obs_days").unique().collect()
    z = grid.join(months, how="cross").join(panel_keys, on=["cell_id", "year", "month"], how="anti").join(
        obs, on=["year", "month"], how="left")
    zero_cols = [f for f in NUMERIC_FEATURES if f.startswith(("aqua_", "terra_")) and not f.endswith(("obs_days", "night_ratio", "drift"))]
    return z.with_columns(
        *[pl.lit(0.0).alias(c) for c in zero_cols],
        aqua_night_ratio=pl.lit(None, dtype=pl.Float64), terra_night_ratio=pl.lit(None, dtype=pl.Float64),
        month_sin=(2 * np.pi * pl.col("month").cast(pl.Float64) / 12).sin(),
        month_cos=(2 * np.pi * pl.col("month").cast(pl.Float64) / 12).cos(),
        terra_drift=(pl.col("year") >= 2021).cast(pl.Int8), y=pl.lit(0.0),
    )


def zero_rule_mask(df: pl.DataFrame) -> np.ndarray:
    return df.select(((pl.col("aqua_n_det").fill_null(0) + pl.col("terra_n_det") + pl.col("aqua_n_det_nb").fill_null(0)
                       + pl.col("terra_n_det_nb")) <= 0))[:, 0].to_numpy()


# ── 메인 루프 ─────────────────────────────────────────────────────────────

def select_objective(tt: pl.DataFrame, cats: dict, log: list) -> str:
    scores = {k: [] for k in OBJECTIVES}
    for y in SELECT_FOLDS:
        v = inner_year(y)
        tr = tt.filter(~pl.col("year").is_in([y, v]))
        va = tt.filter(pl.col("year") == v)
        for name, obj in OBJECTIVES.items():
            t0 = time.perf_counter()
            bst = xgb_train(tr, va, M1_FEATURES, cats, obj)
            p = xgb_predict(bst, va, M1_FEATURES, cats)
            dev = poisson_dev(va["y"].to_numpy(), p, va["w"].to_numpy())
            scores[name].append(dev)
            log.append({"select_fold": y, "inner_year": v, "objective": name, "inner_poisson_dev": dev,
                        "best_iteration": bst.best_iteration, "sec": time.perf_counter() - t0})
            print(f"  [선택] fold {y} (안쪽 {v}) {name}: dev={dev:.4f}, iter={bst.best_iteration}, {time.perf_counter() - t0:.0f}초")
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    best = min(mean, key=mean.get)
    print(f"[목적함수 선택] {best} (안쪽 검증 Poisson deviance 평균 {mean})")
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="B1·B1c·B2·M1(XGBoost GPU) leave-year-out 비교")
    ap.add_argument("--objective", choices=list(OBJECTIVES), help="목적함수 선택 단계를 건너뛰고 지정")
    ap.add_argument("--folds", nargs="*", type=int, help="일부 fold만 (시험용)")
    args = ap.parse_args(argv)
    start_run_log("m1_xgb")
    t_all = time.perf_counter()

    tt = pl.read_parquet(TRAIN_PATH).with_columns(w=pl.col("sample_weight") / pl.col("sample_weight").mean())
    cats = category_maps()
    continents = sorted(c for c in tt["continent"].unique().to_list() if c not in ("none",))
    masks = region_masks(tt, continents)
    grid = pl.read_parquet(GRID_PATH).with_columns(
        pl.col("adm0_a3").fill_null("none"), pl.col("subregion").fill_null("none"), pl.col("continent").fill_null("none"))
    folds = args.folds or FOLD_YEARS

    sel_log: list[dict] = []
    obj_name = args.objective or select_objective(tt, cats, sel_log)
    obj = OBJECTIVES[obj_name]

    s1, s1c = b1_sums(), b1c_sums()
    rows, oof, cell_oof, iters = [], [], [], {"M1": [], "M1_terra": [], "M1_q": []}
    for y in folds:
        t0 = time.perf_counter()
        v = inner_year(y)
        train_years = [x for x in FOLD_YEARS if x != y]
        tr = tt.filter(~pl.col("year").is_in([y, v]))
        va = tt.filter(pl.col("year") == v)
        te = tt.filter(pl.col("year") == y)
        panel = load_panel("train", [y]).collect().with_columns(pl.col("adm0_a3").fill_null("none"))
        dz = dense_zero_rows(y, grid, panel.select("cell_id", "year", "month"))

        r1, r1c, r1c_t = b1_fit(s1, train_years), b1c_fit(s1c, train_years), b1c_fit(s1c, train_years, "t")
        b2 = b2_fit(pl.concat([tr, va]))
        m1 = xgb_train(tr, va, M1_FEATURES, cats, obj)
        mq = xgb_train(tr, va, M1_FEATURES, cats, {}, quantile=True)
        mt = xgb_train(tr, va, M1_TERRA_FEATURES, cats, obj)
        iters["M1"].append(m1.best_iteration)
        iters["M1_q"].append(mq.best_iteration)
        iters["M1_terra"].append(mt.best_iteration)

        zr_p, zr_z = zero_rule_mask(panel), np.ones(dz.height, dtype=bool)
        preds = {  # model -> (패널 예측, 패널 밖 0행 예측)
            "B1": (b1_predict(panel, r1).to_numpy(), np.zeros(dz.height)),
            "B1c": (b1c_predict(panel, r1c), np.zeros(dz.height)),
            "B1c_terra": (b1c_predict(panel, r1c_t, True), np.zeros(dz.height)),
            "B2": (b2_predict(b2, panel), b2_predict(b2, dz)),
            "M1": (xgb_predict(m1, panel, M1_FEATURES, cats), xgb_predict(m1, dz, M1_FEATURES, cats)),
            "M1_terra": (xgb_predict(mt, panel, M1_TERRA_FEATURES, cats), xgb_predict(mt, dz, M1_TERRA_FEATURES, cats)),
        }
        for base in ("B2", "M1", "M1_terra"):
            pp, pz = preds[base]
            preds[f"{base}+zr"] = (np.where(zr_p, 0.0, pp), np.where(zr_z, 0.0, pz))

        for model, (pp, pz) in preds.items():
            rm = region_month_totals(panel, pl.Series(pp), masks)
            if pz.any():
                rz = region_month_totals(dz.with_columns(y=pl.lit(0.0)), pl.Series(pz), masks)
                rm = pl.concat([rm, rz]).group_by("year", "month", "region").agg(pl.col("true", "pred").sum())
            rm = rm.with_columns(model=pl.lit(model))
            oof.append(rm.select("year", "month", pl.col("true").cast(pl.Float64), pl.col("pred").cast(pl.Float64),
                                 "region", "model"))
            for region in masks:
                r = rm.filter(pl.col("region") == region).sort("month")
                if r.height:
                    for k, val in total_metrics(r["true"].to_numpy(), r["pred"].to_numpy(), "month_total_").items():
                        rows.append({"model": model, "region": region, "fold_year": str(y), "metric": k, "value": val})

        # 칸·월 지표 (학습 표 표본) + 분위수 구간 적중률
        cp = {
            "B1": b1_predict(te, r1).to_numpy(), "B1c": b1c_predict(te, r1c), "B2": b2_predict(b2, te),
            "M1": xgb_predict(m1, te, M1_FEATURES, cats), "M1_terra": xgb_predict(mt, te, M1_TERRA_FEATURES, cats),
        }
        zr_te = zero_rule_mask(te)
        for base in ("B2", "M1", "M1_terra"):
            cp[f"{base}+zr"] = np.where(zr_te, 0.0, cp[base])
        q = xgb_predict(mq, te, M1_FEATURES, cats)
        q10, q90 = np.minimum(q[:, 0], q[:, 1]), np.maximum(q[:, 0], q[:, 1])
        q10z, q90z = np.where(zr_te, 0.0, q10), np.where(zr_te, 0.0, q90)
        yt = te["y"].to_numpy()
        for region, m in masks.items():
            idx = te.with_columns(m.alias("_m"))["_m"].to_numpy().astype(bool)  # with_columns: 리터럴도 행마다 펼침
            if idx.sum() == 0:
                continue
            for model, p in cp.items():
                for k, val in cell_metrics(yt[idx], p[idx]).items():
                    rows.append({"model": model, "region": region, "fold_year": str(y), "metric": k, "value": val})
            for model, (lo, hi) in (("M1", (q10, q90)), ("M1+zr", (q10z, q90z))):
                cov = float(np.mean((yt[idx] >= lo[idx]) & (yt[idx] <= hi[idx])))
                rows.append({"model": model, "region": region, "fold_year": str(y), "metric": "cell_interval_coverage_q10_q90", "value": cov})
        cell_oof.append(te.select("year", "month", "cell_id", "is_korea", "adm0_a3", "continent", "y").with_columns(
            pred_m1=pl.Series(cp["M1"], dtype=pl.Float64), q10=pl.Series(q10, dtype=pl.Float64),
            q90=pl.Series(q90, dtype=pl.Float64)))
        # fold마다 중간 저장 (뒤 단계에서 오류가 나도 학습을 다시 하지 않도록)
        ck = REPORTS_DIR / "m1_checkpoint"
        ck.mkdir(exist_ok=True)
        pl.concat(oof).write_parquet(ck / "oof.parquet")
        pl.DataFrame(rows).write_parquet(ck / "rows.parquet")
        pl.concat(cell_oof).write_parquet(ck / "cell_oof.parquet")
        (ck / "iters.json").write_text(json.dumps(iters), encoding="utf-8")
        print(f"[fold {y}] 안쪽 {v}, M1 iter {m1.best_iteration}, q iter {mq.best_iteration}, terra iter {mt.best_iteration}, "
              f"패널 {panel.height:,} + 0행 {dz.height:,}, {time.perf_counter() - t0:.0f}초")

    oof_df = pl.concat(oof)
    val = pl.DataFrame(rows + pooled_rows(oof_df))
    val.write_csv(REPORTS_DIR / "validation_m1.csv")
    summ = summarize(val)
    summ.write_csv(REPORTS_DIR / "validation_summary.csv")
    oof_df.write_parquet(OOF_PATH)
    pl.concat(cell_oof).write_parquet(OOF_PATH.with_name("oof_cell_m1.parquet"))
    if sel_log:
        pl.DataFrame(sel_log).write_csv(REPORTS_DIR / "m1_objective_selection.csv")

    # ── 최종 모델 (전 기간; 반복 수 = fold 최적 반복 수 중앙값) ──
    import xgboost as xgb
    full = tt
    n_iter = {k: int(np.median(v)) + 1 for k, v in iters.items()}
    def fit_full(feats, params, n):
        d = xgb.QuantileDMatrix(to_matrix(full, feats, cats), label=full["y"].to_numpy(), weight=full["w"].to_numpy(),
                                feature_types=feature_types(feats), enable_categorical=True, max_bin=BASE_PARAMS["max_bin"])
        return xgb.train(params, d, n)
    final = fit_full(M1_FEATURES, {**BASE_PARAMS, **obj}, n_iter["M1"])
    final.save_model(M1_MODEL_PATH)
    fq = fit_full(M1_FEATURES, {**BASE_PARAMS, **Q_PARAMS}, n_iter["M1_q"])
    fq.save_model(M1_Q_PATH)
    ft = fit_full(M1_TERRA_FEATURES, {**BASE_PARAMS, **obj}, n_iter["M1_terra"])
    ft.save_model(M1_TERRA_PATH)
    c_all = b1c_fit(s1c, FOLD_YEARS)
    pl.concat([c_all[k].with_columns(level=pl.lit(k)) for k in c_all], how="diagonal_relaxed").write_parquet(B1C_PATH)
    b2_all = b2_fit(full)
    B2_PATH.write_text(json.dumps({"columns": B2_COLS, "mean": b2_all[0].mean_.tolist(), "scale": b2_all[0].scale_.tolist(),
                                   "coef": b2_all[1].coef_.tolist(), "intercept": float(b2_all[1].intercept_)}), encoding="utf-8")

    # ── 그림 ──
    best_m1 = choose_variant(summ, "M1")
    plot_korea_monthly(oof_df, ["B1", "B1c", best_m1])
    plot_yearly_bias(oof_df, ["B1", "B1c", best_m1])
    plot_importance(final)

    # ── 채택 규칙 ──
    def kv(model: str, metric: str) -> float:
        r = summ.filter(pl.col("model") == model, pl.col("region") == "korea", pl.col("metric") == f"pooled_month_total_{metric}")
        return float(r["mean"][0])
    b1_mae, b1_bias = kv("B1", "MAE"), kv("B1", "bias")
    m_mae, m_bias = kv(best_m1, "MAE"), kv(best_m1, "bias")
    adopt = (m_mae < b1_mae) and (abs(m_bias - 1) <= abs(b1_bias - 1))
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    meta["m1_evaluation"] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "objective": obj_name, "objective_selection": "안쪽 검증 연도 Poisson deviance (선택 fold 2013·2017·2021·2025)",
        "features": M1_FEATURES, "terra_only_features": M1_TERRA_FEATURES, "categorical": CAT_FEATURES,
        "category_levels": {k: len(v) for k, v in cats.items()},
        "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in BASE_PARAMS.items()},
        "n_rounds_final": n_iter, "sample_weight": "1/표본 비율 (positive 0.25, 0행 후보/표본)",
        "best_m1_variant": best_m1,
        "korea_pooled": {"B1": {"MAE": b1_mae, "bias": b1_bias}, best_m1: {"MAE": m_mae, "bias": m_bias},
                         "B1c": {"MAE": kv("B1c", "MAE"), "bias": kv("B1c", "bias")}},
        "adoption_rule": "한국 월 합계 MAE와 |편향-1| 둘 다 M1이 B1보다 나아야 채택",
        "adopted": best_m1 if adopt else "B1",
        "models": {"m1": M1_MODEL_PATH.name, "q10_q90": M1_Q_PATH.name, "terra_only": M1_TERRA_PATH.name,
                   "b1c": B1C_PATH.name, "b2": B2_PATH.name},
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg = dict(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=240, tbl_formatting="ASCII_MARKDOWN",
               tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True, float_precision=3)
    with pl.Config(**cfg):
        show = summ.filter(pl.col("metric").is_in(["pooled_month_total_MAE", "pooled_month_total_bias",
                                                   "pooled_month_total_pearson", "pooled_month_total_spearman",
                                                   "cell_MAE", "cell_poisson_dev", "cell_interval_coverage_q10_q90"]))
        print(show.pivot(on="metric", index=["region", "model"], values="mean").sort("region", "model"))
    print(f"[채택] {meta['m1_evaluation']['adopted']}  (한국 B1 MAE {b1_mae:.1f} 편향 {b1_bias:.3f} / "
          f"{best_m1} MAE {m_mae:.1f} 편향 {m_bias:.3f}) — 전체 {time.perf_counter() - t_all:,.0f}초")


def choose_variant(summ: pl.DataFrame, base: str) -> str:
    """zero rule 켬/끔 중 한국 전 기간 월 합계 MAE가 작은 쪽."""
    r = summ.filter(pl.col("model").is_in([base, f"{base}+zr"]), pl.col("region") == "korea",
                    pl.col("metric") == "pooled_month_total_MAE").sort("mean")
    return r["model"][0]


def plot_importance(bst) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gain = bst.get_score(importance_type="total_gain")
    names = {f"f{i}": f for i, f in enumerate(M1_FEATURES)}
    items = sorted(((names.get(k, k), v) for k, v in gain.items()), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#d62728")
    ax.set_title("M1 (XGBoost) feature importance: total gain")
    ax.set_xscale("log")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "m1_feature_importance.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
