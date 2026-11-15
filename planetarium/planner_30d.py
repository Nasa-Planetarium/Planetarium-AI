"""FireOutlook Planner — "기준일 다음 날부터 30일" 구간 검증 (모델 수정 없음, 정직한 skill 표시용).

예측 (서비스와 같은 합성)
  1~7일   일 단위 모델 k=7 (xgb·lgb 앙상블, 선택 가중) — 시험 해 Y마다 학습 ≤ Y−2, 보정 Y−1 (폴드와 같은 규칙), 모든 격자
  8~30일  월 단위 OOF(다음 달 = 기존 grid_occurrence, 2개월 이후 = Planner ens) × 일별 기후 가중치(2012–2023만; 확률용 = 탐지일 모양,
          건수용 = 탐지 수 모양). 진행 중인 달의 남은 날은 예년 같은 달 기준선.
  합성    p30 = 1 − (1 − p7)·Π(1 − p_달)^f  (서비스와 같은 독립 가정), 건수 = μ7 + Σ μ_달·f
기준선   1~7일 예년 같은 달 "7일 안 탐지" 비율, 8~30일 예년 같은 달 비율 × 같은 가중치 / 건수 = 예년 기후값·중앙값
정답     격자: 0.5° 칸의 점 ≥ 1 (고정 열원 제외) / 지역: aoi 소속 가중치로 모은 S-NPP 기준 value ≥ 1 (planner_regions와 같은 정의)
기준일   2024-01-01부터 14일 간격, d + 30 ≤ 2025-12-31 (홀드아웃만)
지역     격자 hazard 합(보정 전 raw). 보정 곡선: 10분위 신뢰도 표, 그리고 2024↔2025 교차 isotonic(한 해로 맞추고 다른 해에 적용)의 BSS.
결과     reports/planner_30d_validation.csv, reports/planner_30d_calibration.csv(보정 전), reports/planner_30d_calibration_after.csv(교차 보정 후),
         interim/planner/d30/iso30.json (서비스용 등급별 isotonic: 홀드아웃 2024–2025 전체로 맞춤; planner_publish가 planner_30d_skill.json에 넣음)
실행     uv run python -m planetarium.planner_30d features | daily | evaluate
"""

from __future__ import annotations

import argparse
import calendar
import json
import time
from datetime import date, timedelta

import numpy as np
import polars as pl

from planetarium import grid_occurrence as go
from planetarium import planner_daily as pdl
from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

WORK = INTERIM_DATA_DIR / "planner" / "d30"
MONTHLY = INTERIM_DATA_DIR / "planner" / "monthly"
ORIGINS = [d for d in (date(2024, 1, 1) + timedelta(14 * i) for i in range(60)) if d + timedelta(30) <= date(2025, 12, 31)]
GROUPS = {"Korea sgg": "sgg", "Korea sido": "sido", "adm1": "adm1", "countries": "country", "continents": "continent", "world": "world"}


def _di(d: date) -> int:
    return (d - pdl.D0).days


def features() -> None:
    """기준일마다 모든 격자의 일 단위 특징 (planner_daily.samples와 같은 코드, 표본 파일은 그대로)."""
    pdl.samples(full_days={_di(d) for d in ORIGINS}, write_samples=False)


def daily() -> None:
    """k=7 일 단위 예측: 시험 해 Y마다 학습 ≤ Y−2, 보정 Y−1 → 기준일 전체 격자."""
    sel = json.loads((pdl.WORK / "selection.json").read_text())["7"]
    WORK.mkdir(parents=True, exist_ok=True)
    parts = []
    for Y in (2024, 2025):
        t0 = time.perf_counter()
        te = pl.concat([pl.read_parquet(pdl.WORK / "fullgrid" / f"d{_di(d)}.parquet") for d in ORIGINS if d.year == Y])
        o = pdl.fit_predict(7, pdl.load_years(range(2013, Y - 1)), pdl.load_years([Y - 1]), te)
        w = sel["ens_weights"]
        p7 = sum(o[c.removeprefix("p_")] * wt for c, wt in w.items()) if sel["model"] == "ens" else o[sel["model"]]
        parts.append(te.select("gi", "di", "base_clim7", "clim_v").with_columns(p7=pl.Series(np.clip(p7, 0, 1).astype(np.float32)),
                                                                              mu7=pl.Series(o["mu"].astype(np.float32))))
        print(f"[daily Y={Y}] {te.height:,}행, {time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB", flush=True)
    pl.concat(parts).write_parquet(WORK / "daily_k7.parquet")


def _monthly_arrays(G: int, gi_of: dict) -> dict:
    """(대상 달 t, h) → p·μ, 그리고 대상 달 t → 기준선 p·중앙값. h=1은 기존 grid_occurrence(서비스와 같음)."""
    sel = json.loads((MONTHLY / "selection.json").read_text())
    out, base = {}, {}
    old = pl.read_parquet(go.OOF).select("g", "year", "month", p_old="p_iso")
    for h in range(1, 3):
        o = pl.read_parquet(MONTHLY / f"h{h}" / "oof.parquet").filter(pl.col("year").is_between(2024, 2025))
        s = sel[str(h)]
        pe = sum(o[c] * w for c, w in s["ens_weights"].items()) if s["model"] == "ens" else o[f"p_{s['model']}"]
        o = o.with_columns(p=pe)
        if h == 1:
            o = o.join(old, on=["g", "year", "month"], how="left").with_columns(p=pl.col("p_old").fill_null(pl.col("p")))
        o = o.with_columns(gi=pl.col("g").replace_strict(gi_of, default=-1, return_dtype=pl.Int64)).filter(pl.col("gi") >= 0)
        for (yy, mm), d in o.group_by("year", "month"):
            t = yy * 12 + mm - 1
            gi = d["gi"].to_numpy()
            a = {k: np.zeros(G) for k in ("p", "mu")}
            a["p"][gi], a["mu"][gi] = d["p"].to_numpy(), d["mu"].to_numpy()
            out[(t, h)] = a
            if t not in base:
                b = {k: np.zeros(G) for k in ("pb", "med")}
                b["pb"][gi], b["med"][gi] = d["p_base_rate"].to_numpy(), d["med_value"].fill_null(0).to_numpy()
                base[t] = b
    return {"pred": out, "base": base}


def _region_labels() -> pl.DataFrame:
    """aoi × 기준일: 30일 S-NPP 기준 value 합 (aoi 소속 가중치, 0.1° cell_day)."""
    from planetarium.harmonize import aoi_membership
    mem, _ = aoi_membership()
    mem = mem.select("cell_id", "aoi_id", pl.col("weight").cast(pl.Float64))
    cd = (pl.scan_parquet(PROCESSED_DATA_DIR / "harmonized_cell_day.parquet").filter(pl.col("year").is_between(2024, 2025))
          .select("cell_id", "date_local", v=pl.col("value").clip(lower_bound=0)).collect())
    parts = []
    for d in ORIGINS:
        x = cd.filter(pl.col("date_local").is_between(d + timedelta(1), d + timedelta(30))).group_by("cell_id").agg(v=pl.col("v").sum())
        parts.append(x.join(mem, on="cell_id").group_by("aoi_id").agg(value=(pl.col("v") * pl.col("weight")).sum())
                     .with_columns(origin=pl.lit(d)))
    return pl.concat(parts)


def _sk(b: float) -> str:
    return "high" if b >= 0.05 else "low" if b >= 0.01 else "none"


def evaluate() -> None:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score
    from planetarium.planner_dayweights import SLOT, climatology
    t0 = time.perf_counter()
    cells = pdl.cells()
    G = cells.height
    gi_of = dict(zip(cells["g"].to_list(), cells["gi"].to_list()))
    pers = pl.read_parquet(go.CELLS)["persistent_source"].to_numpy()
    C, _ = climatology(range(2012, 2024), "count")
    CA, _ = climatology(range(2012, 2024), "any")
    M = _monthly_arrays(G, gi_of)
    dk = pl.read_parquet(WORK / "daily_k7.parquet")
    V, P = np.load(pdl.WORK / "V.npy", mmap_mode="r"), np.load(pdl.WORK / "P.npy", mmap_mode="r")
    z = np.load(PROCESSED_DATA_DIR / "planner_aoi_weights.npz")
    import scipy.sparse as sp
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    ids = list(z["ids"])
    cell_rows, reg_rows = [], []
    for d0 in ORIGINS:
        di = _di(d0)
        x = dk.filter(pl.col("di") == di).sort("gi")
        assert x.height == G
        p7, mu7 = x["p7"].to_numpy().astype(float), x["mu7"].to_numpy().astype(float)
        b7, bc7 = x["base_clim7"].to_numpy().astype(float), np.expm1(x["clim_v"].to_numpy().astype(float)) * 7
        H = -np.log1p(-np.clip(p7, 0, 0.999999))
        HB = -np.log1p(-np.clip(b7, 0, 0.999999))
        E, EB = mu7.copy(), bc7.copy()
        days = [d0 + timedelta(k) for k in range(8, 31)]
        t0m = d0.year * 12 + d0.month - 1
        for (yy, mm) in sorted({(d.year, d.month) for d in days}):
            md = [d for d in days if (d.year, d.month) == (yy, mm)]
            t = yy * 12 + mm - 1
            h = t - t0m
            dim = calendar.monthrange(yy, mm)[1]
            all_s = [SLOT[(mm, k)] for k in range(1, dim + 1)]
            sel = [SLOT[(d.month, d.day)] for d in md]
            fc = C[:, sel].sum(1) / C[:, all_s].sum(1)
            fp = CA[:, sel].sum(1) / CA[:, all_s].sum(1)
            b = M["base"][t]
            if h == 0:
                pm, em = b["pb"], b["med"]
            else:
                a = M["pred"][(t, h)]
                pm, em = a["p"], a["mu"]
            H += fp * -np.log1p(-np.clip(pm, 0, 0.999999))
            HB += fp * -np.log1p(-np.clip(b["pb"], 0, 0.999999))
            E += fc * em
            EB += fc * b["med"]
        rng = [_di(d0 + timedelta(1)) + k for k in range(30)]
        act_c = np.asarray(V[:, rng], dtype=np.float64).sum(1)
        act_y = (np.asarray(P[:, rng]) > 0).any(1).astype(float)
        cell_rows.append(pl.DataFrame({"origin": [d0] * G, "gi": np.arange(G), "p": 1 - np.exp(-H), "pb": 1 - np.exp(-HB), "e": E, "eb": EB,
                                       "y": act_y, "c": act_c, "pers": pers}))
        reg_rows.append(pl.DataFrame({"origin": [d0] * len(ids), "aoi_id": ids, "p": 1 - np.exp(-(W @ H)), "pb": 1 - np.exp(-(W @ HB)),
                                      "e": W @ E, "eb": W @ EB, "n_eff": np.asarray(W.sum(1)).ravel()}))
    cellf = pl.concat(cell_rows).filter(~pl.col("pers"))
    regf = pl.concat(reg_rows).filter(pl.col("n_eff") > 0).join(_region_labels(), on=["origin", "aoi_id"], how="left").with_columns(
        c=pl.col("value").fill_null(0.0)).with_columns(y=(pl.col("c") >= 1).cast(pl.Float64))
    lv = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_list.parquet", columns=["aoi_id", "level"])
    regf = regf.join(lv, on="aoi_id")
    frames = [("0.5° grid cells", cellf)] + [(g, regf.filter(pl.col("level") == lvl)) for g, lvl in GROUPS.items()]
    rows, cal_rows, after_rows, iso = [], [], [], {}
    for name, d in frames:
        d = d.with_columns(yr=pl.col("origin").dt.year())
        y, p, pb = d["y"].to_numpy(), np.clip(d["p"].to_numpy(), 0, 1), np.clip(d["pb"].to_numpy(), 0, 1)
        bm, bb = float(np.mean((p - y) ** 2)), float(np.mean((pb - y) ** 2))
        # 2024↔2025 교차 isotonic (한 해로 맞추고 다른 해에 적용)
        pc, pbc = np.zeros(len(y)), np.zeros(len(y))
        yr = d["yr"].to_numpy()
        for a_, b_ in ((2024, 2025), (2025, 2024)):
            fi, ap = yr == a_, yr == b_
            if fi.sum() > 10 and len(np.unique(y[fi])) > 1:
                pc[ap] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[fi], y[fi]).predict(p[ap])
                pbc[ap] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pb[fi], y[fi]).predict(pb[ap])
            else:
                pc[ap], pbc[ap] = p[ap], pb[ap]
        bmc, bbc = float(np.mean((pc - y) ** 2)), float(np.mean((pbc - y) ** 2))
        c, e, eb = d["c"].to_numpy(), d["e"].to_numpy(), d["eb"].to_numpy()
        wm, wb = float(np.abs(e - c).sum() / max(c.sum(), 1e-9)), float(np.abs(eb - c).sum() / max(c.sum(), 1e-9))
        bss_c = 1 - bmc / bbc if bbc > 0 else 0.0
        rows.append({"unit": name, "rows": d.height, "obs_rate": float(y.mean()), "mean_p": float(p.mean()), "mean_p_base": float(pb.mean()),
                     "brier_model": bm, "brier_base": bb, "bss_raw": 1 - bm / bb if bb > 0 else 0.0,
                     "brier_model_crossfit_iso": bmc, "brier_base_crossfit_iso": bbc, "bss_crossfit_iso": bss_c,
                     "pr_auc_model": float(average_precision_score(y, p)) if 0 < y.mean() < 1 else None,
                     "pr_auc_base": float(average_precision_score(y, pb)) if 0 < y.mean() < 1 else None,
                     "wape_count_model": wm, "wape_count_base": wb, "count_improvement": 1 - wm / wb if wb > 0 else 0.0,
                     "skill_30d": _sk(min(1 - bm / bb if bb > 0 else 0.0, bss_c)),
                     "count_skill_30d": ("high" if 1 - wm / wb >= 0.20 else "low" if 1 - wm / wb >= 0.05 else "none") if wb > 0 else "none"})
        bins = np.clip((p * 10).astype(int), 0, 9)
        for k in range(10):
            m_ = bins == k
            if m_.any():
                cal_rows.append({"unit": name, "bin": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "n": int(m_.sum()), "mean_pred": float(p[m_].mean()),
                                 "obs_rate": float(y[m_].mean()), "mean_pred_base": float(pb[m_].mean())})
        # 보정 후 신뢰도 (교차 보정 = 서비스에 쓰는 방식을 한 해로 맞추고 다른 해에서 확인)
        for lab, q in (("model_calibrated", pc), ("baseline_calibrated", pbc)):
            qb = np.clip((q * 10).astype(int), 0, 9)
            for k in range(10):
                m_ = qb == k
                if m_.any():
                    after_rows.append({"unit": name, "series": lab, "bin": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "n": int(m_.sum()),
                                       "mean_pred": float(q[m_].mean()), "obs_rate": float(y[m_].mean())})
        # 서비스용 보정점: 홀드아웃 두 해 전체로 맞춤 (모델·평년 각각)
        if len(np.unique(y)) > 1:
            im = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
            ib = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pb, y)
            iso[name] = {"x": im.X_thresholds_.tolist(), "y": im.y_thresholds_.tolist(),
                         "base_x": ib.X_thresholds_.tolist(), "base_y": ib.y_thresholds_.tolist()}
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "planner_30d_validation.csv")
    pl.DataFrame(cal_rows).write_csv(REPORTS_DIR / "planner_30d_calibration.csv")
    pl.DataFrame(after_rows).write_csv(REPORTS_DIR / "planner_30d_calibration_after.csv")
    (WORK / "iso30.json").write_text(json.dumps(iso), encoding="utf-8")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=4):
        print(m)
        print(pl.DataFrame(cal_rows).filter(pl.col("unit").is_in(["0.5° grid cells", "Korea sgg", "adm1", "countries"])))
    print(f"[완료] 기준일 {len(ORIGINS)}개, {time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["features", "daily", "evaluate"])
    a = ap.parse_args()
    start_run_log("planner_30d")
    {"features": features, "daily": daily, "evaluate": evaluate}[a.step]()
