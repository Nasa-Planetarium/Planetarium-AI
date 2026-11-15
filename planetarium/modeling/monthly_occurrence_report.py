"""Record monthly and annual occurrence comparisons from finished benchmark runs."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from planetarium.config import REPORTS_DIR
from planetarium.modeling.forecast_benchmark import LABELS, MODELS, _atomic
from planetarium.modeling.foundation_occurrence_models import CHECKPOINTS

REPORT = REPORTS_DIR / "forecast_monthly_backtest.md"
MONTHLY_METRICS = REPORTS_DIR / "forecast_monthly_backtest_metrics.csv"
YEARLY_METRICS = REPORTS_DIR / "forecast_monthly_backtest_yearly.csv"
RESULTS_MARKER = "<!-- monthly-backtest-results -->"
MODEL_LABELS = {
    **LABELS,
    "history": "최근 탐지율 보정",
    "timesfm3": "TimesFM 3.0",
    "chronos2": "Chronos-2",
    "chronos_bolt": "Chronos-Bolt Base",
    "tirex2": "TiRex-2",
}
ORDER = (*MODELS, "history", *CHECKPOINTS)


def _brier(row: dict | None) -> str:
    return f"{row['brier']:.5f}" if row else "—"


def _best(rows) -> str:
    winner = min(rows, key=lambda row: (row["brier"], row["model"]), default=None)
    return f"{MODEL_LABELS[winner['model']]} {_brier(winner)}" if winner else "—"


def record(folder: Path) -> None:
    files = sorted(folder.glob("*_metrics.csv"))
    if not files:
        print("Start an evaluation before recording results.")
        return
    frame = pl.concat([pl.read_csv(file, try_parse_dates=True) for file in files]).sort(
        "month", "grain", "model"
    )
    models = [name for name in ORDER if name in frame["model"].unique().to_list()]
    shared = (
        "as_of",
        "first_date",
        "last_date",
        "forecast_days",
        "days_scored",
        "rollout_days",
        "possible_rows",
        "rows",
        "positive_rate",
    )
    coverage_check = frame.group_by("month", "grain").agg(
        pl.len().alias("models"),
        pl.struct(shared).n_unique().alias("coverage_patterns"),
    )
    if coverage_check.filter(
        (pl.col("models") != len(models)) | (pl.col("coverage_patterns") != 1)
    ).height:
        raise ValueError("The models must all score the same observations each month.")
    yearly = (
        frame.with_columns(pl.col("month").str.slice(0, 4).cast(pl.Int32).alias("year"))
        .group_by("year", "grain", "model")
        .agg(
            months=pl.len(),
            rows=pl.col("rows").sum(),
            possible_rows=pl.col("possible_rows").sum(),
            positive_rate=(
                (pl.col("positive_rate") * pl.col("rows")).sum() / pl.col("rows").sum()
            ),
            brier=((pl.col("brier") * pl.col("rows")).sum() / pl.col("rows").sum()),
            log_loss=(
                (pl.col("log_loss") * pl.col("rows")).sum() / pl.col("rows").sum()
            ),
        )
        .sort("year", "grain", "brier", "model")
    )
    observations = frame.unique(subset=["month", "grain"]).sort("month", "grain")
    months = observations["month"].unique(maintain_order=True).to_list()
    years = yearly["year"].unique().sort().to_list()
    lines = [
        "## 실행 결과",
        "",
        f"{months[0]}~{months[-1]}의 {len(months)}개월을 평가했습니다. 완료된 모델은 {len(models)}/{len(ORDER)}종입니다.",
        "",
        "### 연도별 Brier score",
        "",
        "월별 Brier score에 정상 관측 행 수를 가중해 해당 기간 전체의 Brier score를 계산했습니다. 모든 점수는 낮을수록 좋습니다.",
        "",
    ]
    year_cols = []
    for year in years:
        n = sum(month.startswith(str(year)) for month in months)
        for grain, label in (("cell", "격자"), ("box", "범위")):
            year_cols.append((year, grain, f"{year} {label} ({n}개월)"))
    lines.append("| 모델 | " + " | ".join(col[2] for col in year_cols) + " |")
    lines.append("|---|" + "---:|" * len(year_cols))
    annual = {
        (r["year"], r["grain"], r["model"]): r for r in yearly.iter_rows(named=True)
    }
    for name in models:
        scores = [
            _brier(annual.get((year, grain, name))) for year, grain, _ in year_cols
        ]
        lines.append("| " + " | ".join([MODEL_LABELS[name], *scores]) + " |")

    lines.append("")
    for year in years:
        descriptions = []
        for grain, label in (("cell", "격자"), ("box", "범위")):
            group = [annual[(year, grain, name)] for name in models]
            by_log_loss = min(group, key=lambda r: (r["log_loss"], r["model"]))
            descriptions.append(
                f"{label} 최저 Brier는 {_best(group)}, 최저 log loss는 "
                f"{MODEL_LABELS[by_log_loss['model']]} {by_log_loss['log_loss']:.5f}"
            )
        lines.append(f"- {year}년 {descriptions[0]}이며, {descriptions[1]}입니다.")

    lines.extend(
        [
            "",
            "### 월별 Brier score",
            "",
            "격자 XGBoost와 2차 계절 Markov, 사전학습 모델 중 가장 낮은 점수, 두 공간 단위의 월별 최저 점수를 함께 표시했습니다.",
            "",
            "| 목표 월 | 관측 기준일 | 정상 격자·일 | 격자 XGBoost | 격자 2차 Markov | 격자 최저 사전학습 모델 | 격자 최저 | 범위 최저 |",
            "|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for month in months:
        rows = frame.filter(pl.col("month") == month).to_dicts()
        cell = {r["model"]: r for r in rows if r["grain"] == "cell"}
        box = [r for r in rows if r["grain"] == "box"]
        example = next(iter(cell.values()))
        columns = [
            month,
            example["as_of"].isoformat(),
            f"{example['rows'] / example['possible_rows']:.1%}",
            _brier(cell.get("xgboost")),
            _brier(cell.get("markov2")),
            _best(r for k, r in cell.items() if k in CHECKPOINTS),
            _best(cell.values()),
            _best(box),
        ]
        lines.append("| " + " | ".join(columns) + " |")
    lines += [
        "",
        "[월별 전체 지표](forecast_monthly_backtest_metrics.csv) · [연도별 전체 지표](forecast_monthly_backtest_yearly.csv)",
    ]
    source = REPORT.read_text(encoding="utf-8")
    before, marker, _ = source.partition(RESULTS_MARKER)
    if not marker:
        raise ValueError(f"Missing result marker in {REPORT}")
    _atomic(MONTHLY_METRICS, frame.write_csv())
    _atomic(YEARLY_METRICS, yearly.write_csv())
    _atomic(REPORT, before + marker + "\n\n" + "\n".join(lines) + "\n")
    print(f"Recorded {frame.height} monthly rows and {yearly.height} yearly rows.")
