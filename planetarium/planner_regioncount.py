"""FireOutlook Planner — 큰 지역 월 건수 후보 비교: 격자 μ 합 · 예년 중앙값 합 · AutoETS · AutoTheta (statsforecast, 별도 환경).

대상  대륙(CONT_*; 남극·공해 제외), 전 세계(WORLD), 한국(KR), 아르헨티나(ADM0_ARG). 국가 전체는 시간 관계로 이번에 뺐다.
분할  팀원 벤치마크와 같게: 선택 2020–2023, 확인 2024–2025 (+ 2026 1~10월 추가 확인). 기준 달 c = t − h의 자료까지만 사용.
지표  WAPE = Σ|예측 − 실제| / Σ실제 (S-NPP 기준 aoi_month value)
선택  지역 묶음 × 시점마다 검증 WAPE가 가장 낮은 후보 → interim/planner/count_choice.json
실행  uv run python -m planetarium.planner_regioncount
"""

from __future__ import annotations

import json

import polars as pl

from planetarium.config import INTERIM_DATA_DIR, REPORTS_DIR

PL = INTERIM_DATA_DIR / "planner"
HS = [1, 2, 3, 6, 12]
GROUP = {"WORLD": "world", "KR": "Korea", "ADM0_ARG": "Argentina"}
SKIP = {"CONT_Antarctica", "CONT_Seven_seas__open_ocean"}
CANDS = {"model_mu": "exp_count", "baseline_median": "exp_base", "AutoETS": "AutoETS", "AutoTheta": "AutoTheta"}


def group_of(a: str) -> str | None:
    if a in SKIP:
        return None
    return "continents" if a.startswith("CONT_") else GROUP.get(a)


def main():
    sf = pl.read_parquet(PL / "region_sf_backtest_small.parquet")
    ids = [a for a in sf["aoi_id"].unique().to_list() if group_of(a)]
    rows, choice = [], {}
    for h in HS:
        rp = pl.read_parquet(PL / "monthly" / f"region_preds_h{h}.parquet").filter(pl.col("aoi_id").is_in(ids))
        d = (rp.with_columns(t=pl.col("year").cast(pl.Int32) * 12 + pl.col("month").cast(pl.Int32) - 1)
             .join(sf.filter(pl.col("h") == h).select("aoi_id", "t", "AutoETS", "AutoTheta"), on=["aoi_id", "t"], how="inner")
             .with_columns(grp=pl.col("aoi_id").map_elements(group_of, return_dtype=pl.String), value=pl.col("value").clip(lower_bound=0)))
        choice[h] = {}
        for grp in ("continents", "world", "Korea", "Argentina"):
            for period in ("validation 2020-2023 (LOYO)", "holdout 2024-2025", "extra 2026"):
                x = d.filter(pl.col("grp") == grp, pl.col("period") == period)
                if x.height == 0 or x["value"].sum() == 0:
                    continue
                for name, col in CANDS.items():
                    ratio = (x["value"] / x[col].clip(lower_bound=1e-6)).filter(x[col] > 0.1)
                    rows.append({"horizon_months": h, "group": grp, "period": period, "candidate": name, "months": x.height,
                                 "wape": float((x[col] - x["value"]).abs().sum() / x["value"].sum()),
                                 "ratio_q10": float(ratio.quantile(0.1) or 0), "ratio_q90": float(ratio.quantile(0.9) or 0)})
            v = [r for r in rows if r["horizon_months"] == h and r["group"] == grp and r["period"].startswith("validation")]
            if v:
                choice[h][grp] = min(v, key=lambda r: r["wape"])["candidate"]
    m = pl.DataFrame(rows)
    m.write_csv(REPORTS_DIR / "planner_region_count_compare.csv")
    (PL / "count_choice.json").write_text(json.dumps({str(k): v for k, v in choice.items()}, indent=1), encoding="utf-8")
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=3):
        print(m.pivot(on="candidate", index=["horizon_months", "group", "period"], values="wape").sort("horizon_months", "group", "period"))
    print(json.dumps(choice, indent=1))


def routing() -> pl.DataFrame:
    """지역 건수 최종 라우팅: 등급 × 시점마다 검증 WAPE로 고르고, 확인 구간(2024–2025)에서 기준선(예년 중앙값 합) 대비
    WAPE 개선이 5% 미만이면 기준선을 내고 count_skill = none. 개선 ≥ 20% high, 5~20% low.
    → interim/planner/count_choice.json, reports/planner_count_routing.csv"""
    rows = []
    rm = pl.read_csv(REPORTS_DIR / "planner_region_metrics.csv")
    for (h, grp), d in rm.group_by("horizon_months", "group"):
        v = d.filter(pl.col("period").str.starts_with("validation"))
        o = d.filter(pl.col("period") == "holdout 2024-2025")
        if v.height == 0 or o.height == 0:
            continue
        sel = "model_mu" if v["wape_count_model"][0] < v["wape_count_base_median"][0] else "baseline_median"
        rows.append({"horizon_months": int(h), "group": grp, "selected_on_validation": sel,
                     "hold_wape_selected": float(o["wape_count_model"][0] if sel == "model_mu" else o["wape_count_base_median"][0]),
                     "hold_wape_base": float(o["wape_count_base_median"][0])})
    cc = REPORTS_DIR / "planner_region_count_compare.csv"
    if cc.exists():
        c = pl.read_csv(cc)
        for (h, grp), d in c.group_by("horizon_months", "group"):
            v = d.filter(pl.col("period").str.starts_with("validation"))
            o = d.filter(pl.col("period") == "holdout 2024-2025")
            sel = v.sort("wape")["candidate"][0]
            rows.append({"horizon_months": int(h), "group": grp, "selected_on_validation": sel,
                         "hold_wape_selected": float(o.filter(pl.col("candidate") == sel)["wape"][0]),
                         "hold_wape_base": float(o.filter(pl.col("candidate") == "baseline_median")["wape"][0]),
                         # 구간: 검증 구간 실제/예측 비율의 10·90 분위수 (후보별)
                         **{f"{c_}_ratio_q{q}": float(v.filter(pl.col("candidate") == c_)[f"ratio_q{q}"][0])
                            for c_ in ("AutoETS", "AutoTheta", "model_mu", "baseline_median") for q in (10, 90)}})
    r = pl.DataFrame(rows, infer_schema_length=None).with_columns(improvement=1 - pl.col("hold_wape_selected") / pl.col("hold_wape_base")).with_columns(
        source=pl.when((pl.col("selected_on_validation") != "baseline_median") & (pl.col("improvement") >= 0.05))
        .then(pl.col("selected_on_validation")).otherwise(pl.lit("baseline_median")),
        count_skill=pl.when((pl.col("selected_on_validation") == "baseline_median") | (pl.col("improvement") < 0.05)).then(pl.lit("none"))
        .when(pl.col("improvement") >= 0.20).then(pl.lit("high")).otherwise(pl.lit("low"))).sort("group", "horizon_months")
    r.select("horizon_months", "group", "selected_on_validation", "hold_wape_selected", "hold_wape_base", "improvement", "source",
             "count_skill").write_csv(REPORTS_DIR / "planner_count_routing.csv")
    out: dict = {}
    for row in r.iter_rows(named=True):
        ent = {k: row[k] for k in ("source", "count_skill", "selected_on_validation", "hold_wape_selected", "hold_wape_base", "improvement")}
        if row.get(f"{row['source']}_ratio_q10") is not None:
            ent.update(ratio_q10=row[f"{row['source']}_ratio_q10"], ratio_q90=row[f"{row['source']}_ratio_q90"])
        out.setdefault(str(row["horizon_months"]), {})[row["group"]] = ent
    (PL / "count_choice.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=3):
        print(r)
    return r


if __name__ == "__main__":
    import sys
    main()
    routing()
