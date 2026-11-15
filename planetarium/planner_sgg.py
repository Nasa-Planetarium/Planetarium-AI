"""FireOutlook Planner — 한국 시군구 2·3개월 뒤: 시군구에서 직접 학습(FireOutlook Monthly – District와 같은 방식) vs 격자 합산.

직접 모델  occurrence.panel()을 HORIZON = h로 다시 만든다 (직전 달 특징만 h개월 앞당김) → XGBoost + Platt (aoi_occurrence와 같은 보정).
격자 합산  planner_regions의 지역 예측(region_preds_h{h}.parquet; 검증 구간은 해 하나씩 뺀 보정).
정답      /occurrence와 같은 정답: 시군구 안 VIIRS 점 ≥ 1 (occurrence 패널의 label).
선택      검증 2020–2023 Brier가 낮은 쪽, 확인 2024–2025, 추가 확인 2026(1~10월). h = 1은 비교만 (서비스는 기존 aoi_occurrence 값).
실행      uv run python -m planetarium.planner_sgg panels | folds | compare | final
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl
from tqdm import tqdm

from planetarium import occurrence as occ
from planetarium.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

WORK = INTERIM_DATA_DIR / "planner" / "sgg"
REGION = INTERIM_DATA_DIR / "planner" / "monthly"
HS = [1, 2, 3]
ORIGIN = 2026 * 12 + 10
VAL, HOLD = (2020, 2023), (2024, 2025)
OUT = PROCESSED_DATA_DIR / "planner_sgg_monthly.parquet"


def _set(h: int):
    t = ORIGIN + h
    occ.HORIZON, occ.TARGET = h, (t // 12, t % 12 + 1)
    occ.WORK = WORK / f"h{h}"
    occ.PANEL = occ.WORK / "panel.parquet"


def panels():
    for h in HS:
        _set(h)
        occ.panel()


def folds():
    for h in HS:
        _set(h)
        p = pl.read_parquet(occ.PANEL).filter(pl.col("label").is_not_null())
        outs = []
        for Y in tqdm(range(2020, 2027), desc=f"sgg h={h}", unit="fold"):
            tr, va, te = p.filter(pl.col("year") <= Y - 2), p.filter(pl.col("year") == Y - 1), p.filter(pl.col("year") == Y)
            bst, iso, platt = occ.fit_fold(tr, va)
            raw = bst.predict(occ._dm(te, False), iteration_range=(0, bst.best_iteration + 1))
            prior = float(p.filter(pl.col("year") < Y)["label"].mean())
            outs.append(te.select("aoi_id", "year", "month", "label").with_columns(
                p_direct=pl.Series(platt.predict(raw)), p_base_rate=te["rate_all"].fill_null(prior)))
        pl.concat(outs).write_parquet(occ.WORK / "oof.parquet")
        print(f"[h={h}] 완료, peak {peak_memory_gb() or 0:.2f} GB", flush=True)


def compare() -> dict:
    rows, choice = [], {}
    for h in HS:
        d = pl.read_parquet(WORK / f"h{h}" / "oof.parquet")
        rp = REGION / f"region_preds_h{h}.parquet"
        if rp.exists():
            g = pl.read_parquet(rp, columns=["aoi_id", "year", "month", "p"]).rename({"p": "p_grid_agg"})
            d = d.join(g, on=["aoi_id", "year", "month"], how="left")
        for period, f in (("validation 2020-2023", pl.col("year").is_between(*VAL)), ("holdout 2024-2025", pl.col("year").is_between(*HOLD)),
                          ("extra 2026", pl.col("year") == 2026)):
            x = d.filter(f)
            if "p_grid_agg" in x.columns:
                x = x.filter(pl.col("p_grid_agg").is_not_null())
            y = x["label"].to_numpy()
            if len(y) == 0:
                continue
            from sklearn.metrics import average_precision_score
            bb = float(np.mean((x["p_base_rate"].to_numpy() - y) ** 2))
            for m in ("p_direct", "p_grid_agg", "p_base_rate"):
                if m not in x.columns:
                    continue
                b = float(np.mean((x[m].to_numpy() - y) ** 2))
                rows.append({"horizon_months": h, "period": period, "model": m.removeprefix("p_"), "rows": x.height, "pos_rate": float(y.mean()),
                             "brier": b, "bss_vs_base_rate": 1 - b / bb, "pr_auc": float(average_precision_score(y, x[m].to_numpy()))})
        v = [r for r in rows if r["horizon_months"] == h and r["period"].startswith("validation")]
        best = min(v, key=lambda r: r["brier"])
        skill = "high" if best["bss_vs_base_rate"] >= 0.05 else "low" if best["bss_vs_base_rate"] >= 0.01 else "none"
        choice[h] = {"model": best["model"] if skill != "none" else "base_rate", "validation_bss": best["bss_vs_base_rate"], "skill": skill}
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "planner_sgg_compare.csv")
    (WORK / "choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=4):
        print(m)
    print(json.dumps(choice, indent=1))
    return choice


def final():
    """직접 모델이 뽑힌 h(2·3)만 서비스 예측: 학습 ≤ 2024, 보정 2025 → 대상 달 ORIGIN + h."""
    choice = json.loads((WORK / "choice.json").read_text())
    parts = []
    for h in (2, 3):
        _set(h)
        p = pl.read_parquet(occ.PANEL)
        lab = p.filter(pl.col("label").is_not_null())
        t = ORIGIN + h
        te = p.filter(pl.col("year") == t // 12, pl.col("month") == t % 12 + 1)
        bst, iso, platt = occ.fit_fold(lab.filter(pl.col("year") <= 2024), lab.filter(pl.col("year") == 2025))
        raw = bst.predict(occ._dm(te, False), iteration_range=(0, bst.best_iteration + 1))
        prior = float(lab["label"].mean())
        base = te["rate_all"].fill_null(prior)
        c = choice[str(h)]
        p_any = platt.predict(raw) if c["model"] == "direct" else base.to_numpy()
        parts.append(te.select("aoi_id").with_columns(
            h=pl.lit(h, pl.Int8), target_month=pl.lit(f"{t // 12}-{t % 12 + 1:02d}"), p_any=pl.Series(p_any.astype(np.float32)),
            baseline_p=base.cast(pl.Float32), p_direct=pl.Series(platt.predict(raw).astype(np.float32)),
            model=pl.lit("planner_sgg_direct_xgb+platt" if c["model"] == "direct" else c["model"]), skill=pl.lit(c["skill"]),
            as_of_date=pl.lit("2026-11-14").str.to_date()))
    out = pl.concat(parts)
    out.write_parquet(OUT)
    print(f"[완료] {OUT.name}: {out.height}행")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["panels", "folds", "compare", "final"])
    a = ap.parse_args(argv)
    start_run_log("planner_sgg")
    {"panels": panels, "folds": folds, "compare": compare, "final": final}[a.step]()


if __name__ == "__main__":
    main()
