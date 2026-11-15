"""확산 예측 v2 후보 (실험, 2026-11-15) — 기존 spread_nowcast(v1) 산출물은 건드리지 않는다.

표본  v1과 같은 규칙(0.025° 칸, 오늘 활성 칸 주변 후보, 다음 현지 날짜 감지 = 라벨, S-NPP d·d+1 ok인 날)으로
      고정 열원 제외(v1.2) 뒤 다시 만든다. 반경 3칸(약 7.5 km)·5칸(약 12.5 km) 두 벌.
특성  V1    = v1과 같은 특성 (기상은 v1처럼 기준일 당일)
      V2    = v1의 불 특성 + 아래 (모든 기상·FWI는 기준일 전날 d-1 값: 누수 금지)
        prev2_*            이틀 전 주변 감지 (추세), trend_n = n7 - prev_n7, trend2_n = n7 - prev2_n7
        evt_days_self/max  불 칸이 속한 화재 사건(fire_events)의 시작일부터 오늘까지 일수
        dw_push, dw_frp, dw_cos_max  불 칸 → 후보 칸 방향과 바람이 불어 가는 방향의 코사인(양수만) × 풍속(또는 FRP) / 거리, 칸별 합·최대
        ws_l1, ws_max_l1, rh_l1, rh_min_l1, t_max_l1, prcp_l1, prcp3_l1, prcp7_l1, wind_align_l1, wind_push_l1
        FWI_l1, ISI_l1, BUI_l1, fwi_rel_l1 (0~3: 평년보다 낮음 ~ 매우 높음; 상대 등급의 기준 분포는 2012~2025라 약한 미래 정보 포함)
검증  v1과 같은 leave-year-out (2012~2025, 안쪽 조기 종료 해 = 다음 해, 2025는 2024), 음성 표본 비율 ARG 10%·KR 100% (가중치 1/비율)
실행  uv run python -m planetarium.spread_v2 samples|train|report|latest --radius 3|5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium import spread_nowcast as sn
from planetarium import spread_nowcast_model as sm
from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

ROOT = PROCESSED_DATA_DIR.parent.parent
MODELS_DIR = ROOT / "models"
BASE = INTERIM_DATA_DIR / "spread_v2"
YEARS = sm.YEARS
V1 = sm.FEATURES
NEW_FIRE = ["prev2_n3", "prev2_n7", "prev2_frp7", "prev2_cells7", "trend_n", "trend2_n", "evt_days_self", "evt_days_max",
            "dw_push", "dw_frp", "dw_cos_max"]
WEATHER_L1 = ["ws_l1", "ws_max_l1", "rh_l1", "rh_min_l1", "t_max_l1", "prcp_l1", "prcp3_l1", "prcp7_l1", "wind_align_l1", "wind_push_l1"]
FWI_L1 = ["FWI_l1", "ISI_l1", "BUI_l1", "fwi_rel_l1"]
V2 = [c for c in sm.FIRE] + NEW_FIRE + WEATHER_L1 + FWI_L1 + sm.CONTEXT
REL = {"below_normal": 0, "normal": 1, "above_normal": 2, "well_above_normal": 3}


def out_dir(r: int):
    return BASE / f"r{r}"


# ── 표본 ─────────────────────────────────────────────────────────────────

def offsets(r: int) -> pl.DataFrame:
    rows = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ang = (math.degrees(math.atan2(-dx, -dy)) + 360) % 360
            rows.append({"dy": dy, "dx": dx, "cheb": max(abs(dx), abs(dy)), "sector": int(((ang + 22.5) % 360) // 45)})
    return pl.DataFrame(rows, schema={"dy": pl.Int32, "dx": pl.Int32, "cheb": pl.Int32, "sector": pl.Int8})


def wind_l1(region: str) -> pl.DataFrame:
    """0.5° 지점 × 날짜의 '전날' 바람 (date = 기준일, 값 = 기준일 - 1일)."""
    w = pl.read_parquet(sn.DAILY_PATH if hasattr(sn, "DAILY_PATH") else sm.DAILY_PATH).filter(pl.col("region") == region)
    return w.select("glat", "glon", date_local=pl.col("date") + pl.duration(days=1), w_ws="WS10M", w_wd="WD10M")


def grid_cols() -> list[pl.Expr]:
    lat = (pl.col("iy").cast(pl.Float64) + 0.5) * sn.RES - 90
    lon = (pl.col("ix").cast(pl.Float64) + 0.5) * sn.RES - 180
    return [((lat / sm.GRID).floor() * sm.GRID + sm.GRID / 2).round(3).alias("glat"),
            ((lon / sm.GRID).floor() * sm.GRID + sm.GRID / 2).round(3).alias("glon")]


def event_days(region: str) -> pl.DataFrame:
    """칸·날짜 → 그 칸 점이 속한 화재 사건의 시작일부터 오늘까지 일수 (최대)."""
    pts = pl.read_parquet(INTERIM_DATA_DIR / "fire_events" / f"points_{region}.parquet", columns=["event_id", "latitude", "longitude", "date_local"])
    ev = pl.read_parquet(PROCESSED_DATA_DIR / "fire_events.parquet", columns=["event_id", "start_date"])
    return (pts.join(ev, on="event_id").with_columns(
        iy=((pl.col("latitude") + 90) / sn.RES).floor().cast(pl.Int32), ix=((pl.col("longitude") + 180) / sn.RES).floor().cast(pl.Int32))
        .group_by("date_local", "iy", "ix").agg(evt_days=((pl.col("date_local") - pl.col("start_date")).dt.total_days() + 1).max().cast(pl.Int16)))


def neighborhood_v2(active: pl.DataFrame, r: int, prefix: str, extra: bool) -> pl.DataFrame:
    off = offsets(r)
    e = active.join(off, how="cross").with_columns(iy=pl.col("iy") + pl.col("dy"), ix=pl.col("ix") + pl.col("dx"))
    coslat = ((pl.col("iy").cast(pl.Float64) + 0.5) * sn.RES - 90).radians().cos()
    ex, ey = pl.col("dx").cast(pl.Float64) * coslat * sn.KM_PER_CELL, pl.col("dy").cast(pl.Float64) * sn.KM_PER_CELL
    e = e.with_columns(dist=(ex ** 2 + ey ** 2).sqrt(), ux=ex, uy=ey)
    is_self, in3, other = pl.col("cheb") == 0, pl.col("cheb") <= 1, pl.col("cheb") > 0
    aggs = [pl.col("n").filter(is_self).sum().alias(f"{prefix}self_n"), pl.col("frp_sum").filter(is_self).sum().alias(f"{prefix}self_frp"),
            pl.col("n").filter(in3).sum().alias(f"{prefix}n3"), pl.col("frp_sum").filter(in3).sum().alias(f"{prefix}frp3"),
            pl.col("n").sum().alias(f"{prefix}n7"), pl.col("frp_sum").sum().alias(f"{prefix}frp7"),
            pl.col("frp_max").max().alias(f"{prefix}frpmax7"), pl.len().alias(f"{prefix}cells7")]
    if extra:
        to = (pl.col("w_wd") + 180).radians()
        cosw = pl.when(pl.col("dist") > 0).then((pl.col("ux") * to.sin() + pl.col("uy") * to.cos()) / pl.col("dist")).otherwise(0.0)
        pos = cosw.clip(lower_bound=0.0)
        aggs += [pl.col("streak").filter(is_self).max().fill_null(0).alias("self_streak"), pl.col("streak").max().alias("streak7_max"),
                 pl.col("dist").filter(other).min().alias("near_km"),
                 (pl.col("ux") / pl.col("dist") ** 2 * pl.col("n")).filter(other).sum().alias("sx"),
                 (pl.col("uy") / pl.col("dist") ** 2 * pl.col("n")).filter(other).sum().alias("sy"),
                 *[(pl.col("sector") == k).filter(other).sum().cast(pl.Int16).alias(f"dir_{d}") for k, d in enumerate(sn.DIRS)],
                 pl.col("evt_days").filter(is_self).max().alias("evt_days_self"), pl.col("evt_days").max().alias("evt_days_max"),
                 (pos * pl.col("w_ws") * pl.col("n") / pl.col("dist").clip(lower_bound=1.0)).filter(other).sum().alias("dw_push"),
                 (pos * pl.col("frp_sum") / pl.col("dist").clip(lower_bound=1.0)).filter(other).sum().alias("dw_frp"),
                 cosw.filter(other).max().alias("dw_cos_max")]
    return e.group_by("date_local", "iy", "ix").agg(aggs)


def build_year(region: str, year: int, r: int, ok: set, adm: pl.DataFrame, wind: pl.DataFrame, evd: pl.DataFrame) -> dict:
    t0 = time.perf_counter()
    d0, d1 = max(date(year, 1, 1), sn.START), min(date(year, 12, 31), sn.END - timedelta(days=1))
    pts = sn.load_points(region, d0 - timedelta(days=2), d1 + timedelta(days=1))      # 고정 열원 제외 (v1.2)
    cd = sn.add_streak(sn.cell_days(pts))
    act = (cd.filter(pl.col("src"), pl.col("date_local").is_between(d0, d1)).with_columns(*grid_cols())
           .join(wind, on=["glat", "glon", "date_local"], how="left").join(evd, on=["date_local", "iy", "ix"], how="left"))
    today = neighborhood_v2(act, r, "", True)
    src = cd.filter(pl.col("src"))
    prev = neighborhood_v2(src.with_columns(date_local=pl.col("date_local") + pl.duration(days=1)).filter(pl.col("date_local").is_between(d0, d1)),
                           r, "prev_", False)
    prev2 = neighborhood_v2(src.with_columns(date_local=pl.col("date_local") + pl.duration(days=2)).filter(pl.col("date_local").is_between(d0, d1)),
                            r, "prev2_", False).select("date_local", "iy", "ix", "prev2_n3", "prev2_n7", "prev2_frp7", "prev2_cells7")
    lab = cd.select(pl.col("date_local") - pl.duration(days=1), "iy", "ix", next_n="n")
    s = (today.join(prev, on=["date_local", "iy", "ix"], how="left").join(prev2, on=["date_local", "iy", "ix"], how="left")
         .join(lab, on=["date_local", "iy", "ix"], how="left").with_columns(label=(pl.col("next_n").fill_null(0) > 0).cast(pl.Int8)))
    okd = pl.DataFrame({"d": sorted(ok)}, schema={"d": pl.Date})
    nxt = (cd.filter(pl.col("src"), (pl.col("date_local") - pl.duration(days=1)).is_in(act["date_local"].unique().implode()))
           .select(pl.col("date_local") - pl.duration(days=1), "iy", "ix")
           .join(okd.rename({"d": "date_local"}), on="date_local", how="semi")
           .join(okd.select(date_local=pl.col("d") - pl.duration(days=1)), on="date_local", how="semi"))
    covered = nxt.join(today.select("date_local", "iy", "ix"), on=["date_local", "iy", "ix"], how="semi").height
    s = (s.join(okd.rename({"d": "date_local"}), on="date_local", how="semi")
         .join(okd.with_columns(date_local=pl.col("d") - pl.duration(days=1)).select("date_local"), on="date_local", how="semi")
         .join(okd.with_columns(date_local=pl.col("d") + pl.duration(days=1), prev_ok=pl.lit(True)).select("date_local", "prev_ok"),
               on="date_local", how="left").with_columns(pl.col("prev_ok").fill_null(False)))
    lat = (pl.col("iy").cast(pl.Float64) + 0.5) * sn.RES - 90
    lon = (pl.col("ix").cast(pl.Float64) + 0.5) * sn.RES - 180
    m = pl.col("date_local").dt.month().cast(pl.Float64)
    fill0 = [c for c in s.columns if c.startswith(("prev_", "prev2_")) and c != "prev_ok"]
    s = (s.with_columns(cell_lat=lat.cast(pl.Float32), cell_lon=lon.cast(pl.Float32),
                        cell_id01=(((lat + 90) / 0.1).floor() * 3600 + ((lon + 180) / 0.1).floor()).cast(pl.Int64),
                        month_sin=(2 * np.pi * m / 12).sin().cast(pl.Float32), month_cos=(2 * np.pi * m / 12).cos().cast(pl.Float32),
                        region=pl.lit(region))
         .join(adm.rename({"cell_id": "cell_id01", "adm0_a3": "cand_adm0"}), on="cell_id01", how="left").drop("cell_id01", "next_n")
         .with_columns(pl.col(fill0).fill_null(0))
         .with_columns(trend_n=pl.col("n7") - pl.col("prev_n7"), trend2_n=pl.col("n7") - pl.col("prev2_n7")))
    out_dir(r).mkdir(parents=True, exist_ok=True)
    s.write_parquet(out_dir(r) / f"samples_{region}_{year}.parquet", compression="zstd")
    return {"radius": r, "region": region, "year": year, "candidates": s.height, "positives": int(s["label"].sum()),
            "next_active_cells": nxt.height, "next_active_covered": covered, "sec": round(time.perf_counter() - t0, 1),
            "peak_mem_gb": round(peak_memory_gb() or 0, 2)}


def weather_both(region: str) -> pl.DataFrame:
    w = sm.weather_table().filter(pl.col("region") == region).drop("region")
    l1 = w.with_columns(date=pl.col("date") + pl.duration(days=1)).rename(
        {c: f"{c}_l1" for c in ("ws", "ws_max", "wd", "rh", "rh_min", "t_max", "prcp", "prcp3", "prcp7")})
    f = (pl.read_parquet(PROCESSED_DATA_DIR / "fwi_cell_day.parquet", columns=["region", "glat", "glon", "date", "FWI", "ISI", "BUI", "fwi_rel_class"])
         .filter(pl.col("region") == region).drop("region")
         .with_columns(date=pl.col("date") + pl.duration(days=1), fwi_rel_l1=pl.col("fwi_rel_class").replace_strict(REL, default=None, return_dtype=pl.Int8))
         .rename({"FWI": "FWI_l1", "ISI": "ISI_l1", "BUI": "BUI_l1"}).drop("fwi_rel_class"))
    return w.join(l1, on=["glat", "glon", "date"], how="left").join(f, on=["glat", "glon", "date"], how="left")


def add_wind_cols(df: pl.DataFrame, wd: str, ws: str, suffix: str) -> pl.DataFrame:
    to = (pl.col(wd) + 180).radians()
    mag = (pl.col("sx") ** 2 + pl.col("sy") ** 2).sqrt()
    align = pl.when(mag > 0).then((pl.col("sx") * to.sin() + pl.col("sy") * to.cos()) / mag).otherwise(0.0)
    return df.with_columns(**{f"wind_align{suffix}": align.cast(pl.Float32), f"wind_push{suffix}": (align * pl.col(ws)).cast(pl.Float32)})


def samples(r: int, years: list[int]) -> None:
    ok, adm = sn.ok_days(), sn.cell_adm0()
    log = []
    for region in ("KR", "ARG"):
        wind, evd = wind_l1(region), event_days(region)
        wb = weather_both(region)
        for y in tqdm(years, desc=f"표본 r{r} {region}", unit="year", dynamic_ncols=True):
            info = build_year(region, y, r, ok, adm, wind, evd)
            p = out_dir(r) / f"samples_{region}_{y}.parquet"
            s = pl.read_parquet(p).with_columns(*grid_cols()).join(wb.rename({"date": "date_local"}), on=["glat", "glon", "date_local"], how="left")
            s = add_wind_cols(s, "wd", "ws", "")
            s = add_wind_cols(s, "wd_l1", "ws_l1", "_l1")
            s = s.with_columns(spread_mag=((pl.col("sx") ** 2 + pl.col("sy") ** 2).sqrt()).cast(pl.Float32),
                               is_kr=pl.lit(1 if region == "KR" else 0, dtype=pl.Int8), prev_ok=pl.col("prev_ok").cast(pl.Int8),
                               u=pl.lit(np.random.default_rng(sm.SEED + y).random(s.height)).cast(pl.Float32))
            keep = ["date_local", "iy", "ix", "region", "cand_adm0", "label", "u", *dict.fromkeys(V1 + V2)]
            s.select(keep).write_parquet(p, compression="zstd")
            log.append(info)
            tqdm.write(f"[r{r} {region} {y}] 후보 {info['candidates']:,}, 양성 {info['positives']:,}, 내일 활성 칸 중 후보 안 "
                       f"{info['next_active_covered']:,}/{info['next_active_cells']:,}, {info['sec']}초, 최대 메모리 {info['peak_mem_gb']} GB")
    pl.DataFrame(log).write_csv(out_dir(r) / "build_log.csv")


# ── 학습·검증 ─────────────────────────────────────────────────────────────

def scan(r: int, years: list[int]) -> pl.LazyFrame:
    return pl.scan_parquet([out_dir(r) / f"samples_{reg}_{y}.parquet" for reg in ("KR", "ARG") for y in years])


def rows(r: int, years: list[int], feats: list[str]) -> pl.DataFrame:
    rate = pl.col("region").replace_strict(sm.NEG_RATE, return_dtype=pl.Float32)
    return (scan(r, years).filter((pl.col("label") == 1) | (pl.col("u") < rate))
            .select("label", *feats, w=pl.when(pl.col("label") == 1).then(1.0).otherwise(1.0 / rate).cast(pl.Float32)).collect())


def fit(tr, va, feats):
    import xgboost as xgb
    Xt = tr.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    Xv = va.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    dtr = xgb.QuantileDMatrix(Xt, label=tr["label"].to_numpy(), weight=tr["w"].to_numpy(), max_bin=256, feature_names=feats)
    dva = xgb.QuantileDMatrix(Xv, label=va["label"].to_numpy(), weight=va["w"].to_numpy(), ref=dtr, feature_names=feats)
    return xgb.train(sm.PARAMS, dtr, sm.MAX_ROUNDS, evals=[(dva, "val")], early_stopping_rounds=sm.EARLY_STOP, verbose_eval=False)


def predict(bst, df, feats):
    import xgboost as xgb
    X = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    return bst.predict(xgb.DMatrix(X, feature_names=feats), iteration_range=(0, bst.best_iteration + 1))


SETS = {"v1": V1, "v2": V2}


def train(r: int, sets: list[str]) -> None:
    od = out_dir(r) / "oof"
    od.mkdir(parents=True, exist_ok=True)
    for name in sets:
        feats = SETS[name]
        imp = {}
        for y in tqdm(YEARS, desc=f"r{r} {name} leave-year-out", unit="fold", dynamic_ncols=True):
            f = od / f"{name}_{y}.parquet"
            if f.exists():
                continue
            t0 = time.perf_counter()
            v = sm.inner_year(y)
            bst = fit(rows(r, [x for x in YEARS if x not in (y, v)], feats), rows(r, [v], feats), feats)
            te = scan(r, [y]).select("date_local", "region", "cand_adm0", "label", *feats).collect()
            te.select("date_local", "region", "cand_adm0", "label").with_columns(p=pl.Series(predict(bst, te, feats), dtype=pl.Float32)).write_parquet(f)
            for k, g in bst.get_score(importance_type="gain").items():
                imp.setdefault(k, []).append(g)
            (od / f"{name}_{y}.json").write_text(json.dumps({"best_iteration": bst.best_iteration,
                                                             "gain": bst.get_score(importance_type="gain")}))
            tqdm.write(f"[r{r} {name} {y}] iter {bst.best_iteration}, {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.1f} GB")


def report(radii: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows_, imps = [], []
    fig, axes = plt.subplots(len(radii), 2, figsize=(10, 4.6 * len(radii)), squeeze=False)
    for i, r in enumerate(radii):
        od = out_dir(r) / "oof"
        for name in SETS:
            files = sorted(od.glob(f"{name}_*.parquet"))
            if len(files) < len(YEARS):
                continue
            oof = pl.read_parquet(files)
            for gname, g in sm.groups().items():
                d = oof.filter(g)
                rows_.append({"radius": r, "model": name, "group": gname, **sm.metrics(d["label"].to_numpy(), d["p"].to_numpy())})
            for j, reg in enumerate(["KR", "ARG"]):
                d = oof.filter(pl.col("region") == reg)
                y, p = d["label"].to_numpy(), d["p"].to_numpy()
                bins = np.quantile(p, np.linspace(0, 1, 16))
                idx = np.clip(np.searchsorted(bins, p, side="right") - 1, 0, 14)
                xs = [p[idx == b].mean() for b in range(15) if (idx == b).sum() > 50]
                ys = [y[idx == b].mean() for b in range(15) if (idx == b).sum() > 50]
                axes[i][j].plot(xs, ys, "o-", label=f"{name}")
                axes[i][j].plot([0, 1], [0, 1], "--", color="#888", lw=1)
                axes[i][j].set_title(f"radius {r}, {reg}")
                axes[i][j].set_xlabel("Predicted"); axes[i][j].set_ylabel("Observed"); axes[i][j].legend(fontsize=8); axes[i][j].grid(alpha=0.3)
            g = {}
            for jf in od.glob(f"{name}_*.json"):
                for k, v in json.loads(jf.read_text())["gain"].items():
                    g.setdefault(k, []).append(v)
            tot = sum(np.mean(v) for v in g.values())
            imps += [{"radius": r, "model": name, "feature": k, "gain_mean": float(np.mean(v)), "gain_share": float(np.mean(v) / tot)} for k, v in g.items()]
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "figures" / "spread_v2_calibration.png", dpi=130); plt.close(fig)
    m = pl.DataFrame(rows_)
    m.write_csv(REPORTS_DIR / "spread_v2_metrics.csv")
    imp = pl.DataFrame(imps).sort("radius", "model", "gain_mean", descending=[False, False, True])
    imp.write_csv(REPORTS_DIR / "spread_v2_importance.csv")
    ceil = []
    for r in radii:
        f = out_dir(r) / "build_log.csv"
        if f.exists():
            lg = pl.read_csv(f)
            ceil.append(lg.group_by("region").agg(pl.col("next_active_cells", "next_active_covered", "candidates", "positives").sum())
                        .with_columns(radius=pl.lit(r), recall_ceiling=pl.col("next_active_covered") / pl.col("next_active_cells")))
    if ceil:
        pl.concat(ceil).sort("radius", "region").write_csv(REPORTS_DIR / "spread_v2_recall_ceiling.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=4, thousands_separator=","):
        print(m.select("radius", "group", "model", "pr_auc", "brier", "roc_auc", "n", "positives"))
        if ceil:
            print(pl.concat(ceil).sort("radius", "region"))
        print(imp.filter(pl.col("model") == "v2").group_by("radius").head(15))
        wind = imp.filter(pl.col("feature").str.contains("wind|dw_|ws"))
        print(wind.group_by("radius", "model").agg(pl.col("gain_share").sum().alias("wind_gain_share")).sort("radius", "model"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["samples", "train", "report"])
    ap.add_argument("--radius", type=int, nargs="+", default=[3])
    ap.add_argument("--sets", nargs="+", default=["v1", "v2"])
    a = ap.parse_args(argv)
    start_run_log("spread_v2")
    t0 = time.perf_counter()
    for r in a.radius:
        if a.step == "samples":
            samples(r, YEARS)
        elif a.step == "train":
            train(r, a.sets)
    if a.step == "report":
        report(a.radius)
    print(f"[완료] {a.step} {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
