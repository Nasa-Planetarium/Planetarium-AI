"""확산 예측 — 운영 조건 검증판 (2026-11-15). 기존 v1(당일 기상)을 대신할 새 기본 모델 후보.

운영에서는 NASA POWER가 약 4일 늦으므로, 검증도 같은 조건으로 한다.
  ops   : v1 불 특성 + 화재 사건 지속일(evt_days_self/max) + 기준일 − 4일 기상 (*_l4, 바람 정렬도 −4일 바람으로)
  nowx  : v1 불 특성 + 화재 사건 지속일 (기상 완전 제외)
표본  고정 열원 제외(v1.2) 뒤 다시 만든 반경 3칸 표본 (spread_v2와 같은 규칙), 같은 leave-year-out 분할.
참고  v1(당일 기상) 성능은 spread_v2 r3 v1 (같은 표본)에서 가져온다.
실행  uv run python -m planetarium.spread_ops samples|train|report|final|latest
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium import spread_nowcast as sn
from planetarium import spread_nowcast_model as sm
from planetarium import spread_v2 as sv
from planetarium.config import PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

LAG = 4
R = 3
OPS_DIR = sv.BASE / "ops_r3"
EVT = ["evt_days_self", "evt_days_max"]
W_L4 = [f"{c}_l{LAG}" for c in ("ws", "ws_max", "rh", "rh_min", "t_max", "prcp", "prcp3", "prcp7", "wind_align", "wind_push")]
OPS = sm.FIRE + EVT + W_L4 + sm.CONTEXT
NOWX = sm.FIRE + EVT + sm.CONTEXT
SETS = {"ops": OPS, "nowx": NOWX}
MODELS_DIR = sm.MODELS_DIR
MODEL_PATH = MODELS_DIR / "spread_nowcast_xgb.json"
META_PATH = MODELS_DIR / "spread_nowcast_meta.json"
LATEST_PATH = sm.LATEST_PATH


def _patch():
    sv.out_dir = lambda r: OPS_DIR
    sv.SETS = SETS


def weather_lag(region: str | None, w: pl.DataFrame | None = None) -> pl.DataFrame:
    w = w if w is not None else sm.weather_table().filter(pl.col("region") == region).drop("region")
    return w.with_columns(date=pl.col("date") + pl.duration(days=LAG)).rename(
        {c: f"{c}_l{LAG}" for c in ("ws", "ws_max", "wd", "rh", "rh_min", "t_max", "prcp", "prcp3", "prcp7")})


def samples() -> None:
    _patch()
    ok, adm = sn.ok_days(), sn.cell_adm0()
    log = []
    for region in ("KR", "ARG"):
        wind, evd, wl = sv.wind_l1(region), sv.event_days(region), weather_lag(region)
        for y in tqdm(sm.YEARS, desc=f"표본 {region}", unit="year", dynamic_ncols=True):
            info = sv.build_year(region, y, R, ok, adm, wind, evd)
            p = OPS_DIR / f"samples_{region}_{y}.parquet"
            s = pl.read_parquet(p).with_columns(*sv.grid_cols()).join(wl.rename({"date": "date_local"}), on=["glat", "glon", "date_local"], how="left")
            s = sv.add_wind_cols(s, f"wd_l{LAG}", f"ws_l{LAG}", f"_l{LAG}")
            s = s.with_columns(spread_mag=((pl.col("sx") ** 2 + pl.col("sy") ** 2).sqrt()).cast(pl.Float32),
                               is_kr=pl.lit(1 if region == "KR" else 0, dtype=pl.Int8), prev_ok=pl.col("prev_ok").cast(pl.Int8),
                               u=pl.lit(np.random.default_rng(sm.SEED + y).random(s.height)).cast(pl.Float32))
            s.select("date_local", "iy", "ix", "region", "cand_adm0", "label", "u", *dict.fromkeys(OPS + NOWX)).write_parquet(p, compression="zstd")
            log.append(info)
    pl.DataFrame(log).write_csv(OPS_DIR / "build_log.csv")
    miss = pl.scan_parquet(OPS_DIR / "samples_*.parquet").select(pl.col(f"ws_l{LAG}").is_null().mean()).collect().item()
    print(f"[완료] 표본 (반경 {R}), −{LAG}일 기상 결측 {miss:.3%}, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


def train() -> None:
    _patch()
    sv.train(R, list(SETS))


def report() -> pl.DataFrame:
    rows = []
    srcs = [("v1_same_day_weather", sv.BASE / "r3" / "oof", "v1"), ("ops_lag4_weather", OPS_DIR / "oof", "ops"), ("no_weather", OPS_DIR / "oof", "nowx")]
    imps = []
    for label, od, name in srcs:
        oof = pl.read_parquet(sorted(od.glob(f"{name}_*.parquet")))
        for gname, g in sm.groups().items():
            d = oof.filter(g)
            rows.append({"model": label, "group": gname, **sm.metrics(d["label"].to_numpy(), d["p"].to_numpy())})
        if name != "v1":
            g = {}
            for jf in od.glob(f"{name}_*.json"):
                for k, v in json.loads(jf.read_text())["gain"].items():
                    g.setdefault(k, []).append(v)
            tot = sum(np.mean(v) for v in g.values())
            imps += [{"model": label, "feature": k, "gain_share": float(np.mean(v) / tot)} for k, v in g.items()]
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "spread_ops_metrics.csv")
    pl.DataFrame(imps).sort("model", "gain_share", descending=[False, True]).write_csv(REPORTS_DIR / "spread_ops_importance.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=5, thousands_separator=","):
        print(m.select("group", "model", "pr_auc", "brier", "roc_auc", "prec_at_rec_0.5", "n", "positives").sort("group", "model"))
    return m


def choose(m: pl.DataFrame) -> str:
    """운영형과 기상 제외 중: 두 나라(KR_box, ARG) PR-AUC 합이 큰 쪽, 같으면 Brier 합이 작은 쪽."""
    s = m.filter(pl.col("group").is_in(["KR_box", "ARG"]), pl.col("model").is_in(["ops_lag4_weather", "no_weather"])).group_by("model").agg(
        pr=pl.col("pr_auc").sum(), br=pl.col("brier").sum()).sort(["pr", "br"], descending=[True, False])
    return s["model"][0]


def final(choice: str) -> None:
    import xgboost as xgb
    _patch()
    name = "ops" if choice == "ops_lag4_weather" else "nowx"
    feats = SETS[name]
    iters = [json.loads((OPS_DIR / "oof" / f"{name}_{y}.json").read_text())["best_iteration"] for y in sm.YEARS]
    n = int(np.median(iters)) + 1
    tr = sv.rows(R, sm.YEARS, feats)
    X = tr.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    d = xgb.QuantileDMatrix(X, label=tr["label"].to_numpy(), weight=tr["w"].to_numpy(), max_bin=256, feature_names=feats)
    bst = xgb.train({k: v for k, v in sm.PARAMS.items() if k != "eval_metric"}, d, n)
    MODELS_DIR.mkdir(exist_ok=True)
    bst.save_model(MODEL_PATH)
    m = pl.read_csv(REPORTS_DIR / "spread_ops_metrics.csv")
    ceil = pl.read_csv(OPS_DIR / "build_log.csv").group_by("region").agg(pl.col("next_active_cells", "next_active_covered").sum()).with_columns(
        recall_ceiling=pl.col("next_active_covered") / pl.col("next_active_cells"))
    meta = {
        "name": "spread_nowcast_xgb", "version": "v1.2-ops (2026-11-15)",
        "description": "위성 감지 기반 단기 확산 가능성 (물리 시뮬레이션 아님): 오늘(현지 날짜) 활성 칸 반경 3칸(0.025° 칸) 안의 칸에 다음 현지 날짜 감지가 생길 확률",
        "created": time.strftime("%Y-%m-%d %H:%M"), "grid_deg": 0.025, "candidate_radius_cells": R,
        "label": "다음 현지 날짜에 그 칸에 기준 점(VIIRS S-NPP) 감지가 있는가 (고정 열원 제외 v1.2)",
        "feature_set": choice, "features": feats,
        "weather_rule": (f"기상은 기준일 − {LAG}일 NASA POWER 값 (운영의 자료 지연과 같은 조건으로 학습·검증)" if name == "ops"
                         else "기상 특성 없음 (운영 지연 조건에서 기상 제외가 더 나음)"),
        "validation": "leave-year-out 2012~2025, 조기 종료 = 안쪽 검증 연도(y+1, 2025는 2024), 고정 열원 제외 표본, 운영과 같은 기상 지연 조건",
        "metrics_pooled": m.to_dicts(), "recall_ceiling": ceil.to_dicts(), "n_rounds": n, "fold_best_iterations": dict(zip(map(str, sm.YEARS), iters)),
        "params": sm.PARAMS, "sampling": {"negative_rate": sm.NEG_RATE, "negative_weight": "1/표본 비율"},
        "previous": "v1 (2026-11-15 00:30): 당일 기상으로 검증 → 운영 성능보다 높게 나왔을 수 있음. 비교는 reports/spread_ops_metrics.csv",
        "caveats": ["평가·확률은 후보 칸(오늘 활성 칸 반경 3칸) 안에서만 의미가 있다. 반경 밖 새 발화는 예측 대상이 아니다.",
                    "학습 라벨은 S-NPP. 2026-11 이후 최신 예측은 NOAA-20 감지로 특성을 만든다.",
                    "화재 사건 지속일은 fire_events(전체 기간으로 연결한 사건)의 시작일 기준이라 아주 약한 미래 정보가 들어갈 수 있다."],
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[완료] {MODEL_PATH.name} ({choice}, {n}회), {META_PATH.name}")


def latest() -> None:
    import xgboost as xgb
    from planetarium.monthly import STATUS_PATH
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feats = meta["features"]
    st = pl.read_parquet(STATUS_PATH).filter(pl.col("satellite") == "NOAA-20")
    d = st.filter(pl.col("status") == "ok")["date_utc"].max()
    prev_ok = st.filter(pl.col("date_utc") == d - timedelta(days=1), pl.col("status") == "ok").height > 0
    adm = sn.cell_adm0()
    w26 = sorted((sm.DAILY_PATH.parent / "power_2026").glob("*/power_daily_2026.parquet"))
    w_all = pl.concat([pl.read_parquet(sm.DAILY_PATH), *(pl.read_parquet(f) for f in w26[-1:])], how="diagonal_relaxed").unique(
        ["region", "glat", "glon", "date"], keep="last")
    parts = []
    for region in sn.REGIONS:
        pts = sn.load_points(region, d - timedelta(days=30), d, sat="N20")
        cd = sn.add_streak(sn.cell_days(pts))
        evd = sv.event_days(region)
        act = (cd.filter(pl.col("src"), pl.col("date_local") == d).join(evd, on=["date_local", "iy", "ix"], how="left")
               .with_columns(w_ws=pl.lit(None, dtype=pl.Float64), w_wd=pl.lit(None, dtype=pl.Float64)))
        if act.height == 0:
            continue
        today = sv.neighborhood_v2(act, R, "", True)
        prev = sv.neighborhood_v2(cd.filter(pl.col("src"), pl.col("date_local") == d - timedelta(days=1))
                                  .with_columns(date_local=pl.col("date_local") + pl.duration(days=1)), R, "prev_", False)
        s = today.join(prev, on=["date_local", "iy", "ix"], how="left")
        lat = (pl.col("iy").cast(pl.Float64) + 0.5) * sn.RES - 90
        lon = (pl.col("ix").cast(pl.Float64) + 0.5) * sn.RES - 180
        mo = pl.col("date_local").dt.month().cast(pl.Float64)
        s = (s.with_columns(cell_lat=lat.cast(pl.Float32), cell_lon=lon.cast(pl.Float32),
                            cell_id01=(((lat + 90) / 0.1).floor() * 3600 + ((lon + 180) / 0.1).floor()).cast(pl.Int64),
                            month_sin=(2 * np.pi * mo / 12).sin().cast(pl.Float32), month_cos=(2 * np.pi * mo / 12).cos().cast(pl.Float32),
                            region=pl.lit(region), prev_ok=pl.lit(int(prev_ok), dtype=pl.Int8), is_kr=pl.lit(1 if region == "KR" else 0, dtype=pl.Int8))
             .join(adm.rename({"cell_id": "cell_id01", "adm0_a3": "cand_adm0"}), on="cell_id01", how="left").drop("cell_id01"))
        s = s.with_columns(pl.col([c for c in s.columns if c.startswith("prev_") and c != "prev_ok"]).fill_null(0),
                           spread_mag=((pl.col("sx") ** 2 + pl.col("sy") ** 2).sqrt()).cast(pl.Float32))
        wr = w_all.filter(pl.col("region") == region).drop("region").sort("glat", "glon", "date").with_columns(
            prcp3=pl.col("PRECTOTCORR").rolling_sum(3, min_samples=1).over("glat", "glon"),
            prcp7=pl.col("PRECTOTCORR").rolling_sum(7, min_samples=1).over("glat", "glon")).select(
            "glat", "glon", "date", ws="WS10M", ws_max="WS10M_MAX", wd="WD10M", rh="RH2M", rh_min="RH2M_MIN_EST", t_max="T2M_MAX",
            prcp="PRECTOTCORR", prcp3="prcp3", prcp7="prcp7")
        wl = weather_lag(None, wr).with_columns(weather_date=pl.col("date") - pl.duration(days=LAG))
        s = s.with_columns(*sv.grid_cols()).join(wl.rename({"date": "date_local"}), on=["glat", "glon", "date_local"], how="left")
        s = sv.add_wind_cols(s, f"wd_l{LAG}", f"ws_l{LAG}", f"_l{LAG}")
        parts.append(s)
    s = pl.concat(parts, how="diagonal_relaxed")
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)
    p = bst.predict(xgb.DMatrix(s.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(), feature_names=feats))
    uses_w = any(c.endswith(f"_l{LAG}") for c in feats)
    out = s.select(
        pl.col("date_local").alias("as_of_date"), (pl.col("date_local") + pl.duration(days=1)).alias("target_date"),
        "region", "cand_adm0", "cell_lat", "cell_lon", pl.lit(sn.RES).alias("cell_size_deg"),
        p_spread=pl.Series(p, dtype=pl.Float32), active_today=(pl.col("self_n") > 0), n_det_today_7x7="n7",
        near_active_km="near_km", wind_align=f"wind_align_l{LAG}" if uses_w else pl.lit(None, dtype=pl.Float32),
        weather_available=pl.col(f"ws_l{LAG}").is_not_null() if uses_w else pl.lit(False),
        weather_date=pl.col("weather_date") if uses_w else pl.lit(None, dtype=pl.Date),
        weather_lag_days=pl.lit(LAG if uses_w else None, dtype=pl.Int16),
        fire_event_days=pl.col("evt_days_max"),
    ).with_columns(model=pl.lit(f"spread_nowcast_xgb {meta['version']} {meta['feature_set']}"), source_satellite=pl.lit("NOAA-20"))
    out.sort("region", "p_spread", descending=[False, True]).write_parquet(LATEST_PATH, compression="zstd")
    print(f"[완료] {LATEST_PATH.name}: 기준일 {d} → 예측일 {d + timedelta(days=1)}, 후보 칸 {out.height:,}, "
          f"지역별 {dict(out.group_by('region').len().iter_rows())}, 20% 초과 {dict(out.filter(pl.col('p_spread') > 0.2).group_by('region').len().iter_rows())}, "
          f"기상 있음 {out['weather_available'].mean():.0%}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["samples", "train", "report", "final", "latest", "all"])
    a = ap.parse_args(argv)
    start_run_log("spread_ops")
    t0 = time.perf_counter()
    if a.step in ("samples", "all"):
        samples()
    if a.step in ("train", "all"):
        train()
    if a.step in ("report", "all"):
        m = report()
        c = choose(m)
        print(f"[선택] {c}")
        (OPS_DIR / "choice.txt").write_text(c)
    if a.step in ("final", "all"):
        final((OPS_DIR / "choice.txt").read_text().strip())
    if a.step in ("latest", "all"):
        latest()
    print(f"[완료] {a.step} {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
