"""격자 다음 달: FireOutlook Planner 월 단위 h=1 (ens) vs 기존 FireOutlook Monthly – Grid (grid_occurrence, XGBoost + isotonic).

같은 행(격자·달, 지속 열원 제외)에서 비교한다. 결정은 확인 구간(2024–2025) Brier — 사용자가 정한 규칙
("홀드아웃에서 나으면 교체"). 검증 2020–2023도 함께 적는다.
교체하면: 기존 latest 파일을 백업하고, 같은 파일·같은 컬럼으로 Planner h=1 값을 쓴다 (/occurrence?unit=grid 와 /forecast가 같은 값).
실행  uv run python -m planetarium.planner_gridswap compare | apply
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import polars as pl

from planetarium import grid_occurrence as go
from planetarium.config import PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.planner_monthly import GRID_LATEST, WORK, hdir

DECISION = WORK / "gridswap.json"


def compare() -> dict:
    from sklearn.metrics import average_precision_score
    sel = json.loads((WORK / "selection.json").read_text())["1"]
    cells = pl.read_parquet(go.CELLS).select("g", "persistent_source")
    new = pl.read_parquet(hdir(1) / "oof.parquet")
    pcols = [c for c in sel["ens_weights"]] if sel["model"] == "ens" else [f"p_{sel['model']}"]
    new = new.with_columns(p_new=sum(pl.col(c) * w for c, w in sel["ens_weights"].items()) if sel["model"] == "ens" else pl.col(pcols[0]))
    old = pl.read_parquet(go.OOF).select("g", "year", "month", p_old="p_iso")
    d = new.select("g", "year", "month", "label", "p_new", "p_base_rate").join(old, on=["g", "year", "month"]).join(cells, on="g").filter(
        ~pl.col("persistent_source"))
    rows = []
    for period, f in (("validation 2020-2023", pl.col("year").is_between(2020, 2023)), ("holdout 2024-2025", pl.col("year").is_between(2024, 2025))):
        x = d.filter(f)
        y = x["label"].to_numpy()
        for m in ("p_new", "p_old", "p_base_rate"):
            p = x[m].to_numpy()
            rows.append({"period": period, "model": {"p_new": f"planner_{sel['model']}", "p_old": "grid_occurrence_xgb+isotonic", "p_base_rate": "base_rate"}[m],
                         "rows": x.height, "brier": float(np.mean((p - y) ** 2)), "pr_auc": float(average_precision_score(y, p))})
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "planner_grid_h1_vs_grid_occurrence.csv")
    h = m.filter(pl.col("period") == "holdout 2024-2025")
    b_new, b_old = h.filter(pl.col("model").str.starts_with("planner"))["brier"][0], h.filter(pl.col("model").str.starts_with("grid"))["brier"][0]
    dec = {"replace": bool(b_new < b_old), "holdout_brier_new": b_new, "holdout_brier_old": b_old, "model": sel["model"], "ens_weights": sel["ens_weights"]}
    DECISION.write_text(json.dumps(dec, indent=1), encoding="utf-8")
    with pl.Config(tbl_rows=-1, float_precision=5):
        print(m)
    print(dec)
    return dec


def apply() -> None:
    dec = json.loads(DECISION.read_text())
    if not dec["replace"]:
        print("[유지] 기존 grid_occurrence가 홀드아웃에서 같거나 낫다 → 파일을 바꾸지 않음")
        return
    old = pl.read_parquet(go.LATEST_PATH)
    bak = PROCESSED_DATA_DIR / "backup_grid_occurrence_pre_planner_20261115"
    bak.mkdir(exist_ok=True)
    if not (bak / go.LATEST_PATH.name).exists():
        shutil.copy2(go.LATEST_PATH, bak / go.LATEST_PATH.name)
    new = pl.read_parquet(GRID_LATEST).filter(pl.col("h") == 1).select(cell_id="g", p_new="p_any")
    out = old.join(new, on="cell_id", how="left").with_columns(
        p_any=pl.col("p_new").fill_null(pl.col("p_any")).cast(pl.Float32), p_model=pl.col("p_new").fill_null(pl.col("p_model")).cast(pl.Float32),
        model=pl.lit(f"planner_monthly_{dec['model']}(xgb+lgb, isotonic)")).with_columns(
        p_none=(1 - pl.col("p_any")).cast(pl.Float32)).select(old.columns)
    assert out.height == old.height
    out.write_parquet(go.LATEST_PATH)
    meta = json.loads(go.META_PATH.read_text(encoding="utf-8"))
    meta["replaced_2026_11_15"] = {"by": "FireOutlook Planner monthly h=1 (planner_monthly ens)", "reason": "holdout 2024-2025 Brier",
                                   "holdout_brier_new": dec["holdout_brier_new"], "holdout_brier_old": dec["holdout_brier_old"],
                                   "backup": str(bak.name), "report": "reports/planner_grid_h1_vs_grid_occurrence.csv"}
    go.META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[교체] {go.LATEST_PATH.name}: {out.height:,}행, 백업 {bak.name}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["compare", "apply"])
    {"compare": compare, "apply": apply}[ap.parse_args().step]()
