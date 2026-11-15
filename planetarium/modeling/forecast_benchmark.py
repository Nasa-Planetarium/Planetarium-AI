"""Rolling-origin benchmark for monthly S-NPP-equivalent hotspot counts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from filelock import FileLock

from planetarium.config import PROCESSED_DATA_DIR, REPORTS_DIR

WORK_DIR = PROCESSED_DATA_DIR / "forecast_benchmark"
REPORT = REPORTS_DIR / "forecast_benchmark.md"
PROTOCOL = "aoi-month-v2"
FIRST_MONTH = np.datetime64("2012-02", "M")
AS_OF = np.datetime64("2026-10", "M")
HORIZON = 3
PERIODS = {
    "validation": ("2020-01", "2023-12"),
    "test": ("2024-01", "2025-12"),
    "recent_sp": ("2026-01", "2026-07"),
}
FOCUS_IDS = ("KR", "ADM0_ARG", "ADM0_USA", "WORLD")
MODELS = {
    "timesfm3": ("TimesFM 3.0", "사전학습"),
    "last_value": ("직전 달", "기준선"),
    "seasonal_naive": ("작년 같은 달", "기준선"),
    "seasonal_median": ("최근 5년 같은 달 중앙값", "기준선"),
    "seasonal_mean": ("최근 5년 같은 달 평균", "기준선"),
    "seasonal_ets": ("계절 지수평활", "지역별 학습"),
    "seasonal_ar": ("계절 자기회귀", "지역별 학습"),
    "theta": ("AutoTheta", "지역별 학습"),
    "tsb": ("TSB", "간헐 시계열"),
    "ridge": ("Ridge", "전체 지역 공동 학습"),
    "lightgbm": ("LightGBM", "전체 지역 공동 학습"),
    "xgboost": ("XGBoost", "전체 지역 공동 학습"),
    "dlinear": ("DLinear", "전체 지역 공동 학습"),
    "nhits": ("N-HiTS", "전체 지역 공동 학습"),
    "chronos2": ("Chronos-2", "사전학습"),
    "timestral": ("Timestral 209M", "사전학습"),
    "tirex2": ("TiRex-2", "사전학습"),
    "autoets": ("AutoETS", "지역별 학습"),
    "sarima": ("SARIMA", "지역별 학습"),
    "croston_sba": ("Croston-SBA", "간헐 시계열"),
    "catboost": ("CatBoost", "전체 지역 공동 학습"),
    "tide": ("TiDE", "전체 지역 공동 학습"),
    "patchtst": ("PatchTST", "전체 지역 공동 학습"),
    "timesfm25": ("TimesFM 2.5", "사전학습"),
    "chronos_bolt": ("Chronos-Bolt Base", "사전학습"),
    "toto2": ("Toto 2.0 22M", "사전학습"),
    "moirai2": ("Moirai 2.0 Small", "사전학습"),
}
FOUNDATION_MODELS = {
    name for name, (_, family) in MODELS.items() if family == "사전학습"
}
START_MARKER = "<!-- benchmark-world-results:start -->"
END_MARKER = "<!-- benchmark-world-results:end -->"


@dataclass
class Panel:
    dates: np.ndarray
    ids: np.ndarray
    levels: np.ndarray
    sizes: np.ndarray
    values: np.ndarray
    origins: np.ndarray
    hashes: dict[str, str]
    clipped: int
    hidden_count: int

    @property
    def targets(self) -> np.ndarray:
        return self.origins[:, None] + np.arange(1, HORIZON + 1)

    @property
    def key(self) -> str:
        signature = hashlib.sha256("".join(self.hashes.values()).encode()).hexdigest()[
            :12
        ]
        return f"{PROTOCOL}-{signature}"


def read_panel() -> Panel:
    source = PROCESSED_DATA_DIR
    files = ("aoi_month.parquet", "aoi_list.parquet")
    hashes = {}
    for name in files:
        with (source / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    with (source / "release_manifest.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        manifest = {row["file"]: row["sha256"] for row in csv.DictReader(handle)}
    if any(manifest.get(name) != digest for name, digest in hashes.items()):
        raise ValueError("AOI parquet checksums differ from the release manifest.")

    aois = pl.read_parquet(source / "aoi_list.parquet").sort("aoi_id")
    if aois.get_column("aoi_id").n_unique() != aois.height:
        raise ValueError("AOI metadata contains duplicate IDs.")
    dates = np.arange(FIRST_MONTH, AS_OF + np.timedelta64(1, "M"))
    month_keys = np.asarray([int(str(month).replace("-", "")) for month in dates])
    frame = (
        pl.scan_parquet(source / "aoi_month.parquet")
        .select(
            "aoi_id", "year", "month", "value", "value_source", "status", "data_source"
        )
        .with_columns(key=pl.col("year").cast(pl.Int32) * 100 + pl.col("month"))
        .filter(pl.col("key").is_between(int(month_keys[0]), int(month_keys[-1])))
        .sort("aoi_id", "key")
        .collect()
    )
    ids = np.asarray(aois["aoi_id"].to_list(), dtype=str)
    expected_rows = aois.height * dates.size
    if frame.height != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} regular AOI-month rows, found {frame.height}."
        )
    if not np.array_equal(
        frame["key"].to_numpy().reshape(-1, dates.size),
        np.tile(month_keys, (len(ids), 1)),
    ):
        raise ValueError("AOI months have duplicates or gaps.")
    if not np.array_equal(
        np.asarray(frame["aoi_id"].to_list()).reshape(-1, dates.size),
        np.broadcast_to(ids[:, None], (len(ids), dates.size)),
    ):
        raise ValueError("AOI IDs do not align with the monthly panel.")
    expected_status = (
        pl.when(pl.col("key") >= 202608).then(pl.lit("nrt")).otherwise(pl.lit("ok"))
    )
    expected_source = (
        pl.when(pl.col("key") >= 202608).then(pl.lit("NRT")).otherwise(pl.lit("SP"))
    )
    invalid = frame.filter(
        (
            (pl.col("value_source") != "snpp_observed")
            | (pl.col("status") != expected_status)
            | (pl.col("data_source") != expected_source)
        ).fill_null(True)
    )
    values = frame["value"].to_numpy().reshape(-1, dates.size)
    if invalid.height or not np.isfinite(values).all():
        raise ValueError(
            "Selected months must have finite S-NPP values and expected SP/NRT status."
        )
    clipped = int(np.count_nonzero(values < 0))
    values = np.maximum(values, 0).astype(np.float32)
    first_target = np.searchsorted(dates, np.datetime64(PERIODS["validation"][0]))
    last_target = np.searchsorted(dates, np.datetime64(PERIODS["recent_sp"][1]))
    origins = np.append(np.arange(first_target - HORIZON, last_target), dates.size - 1)
    return Panel(
        dates,
        ids,
        np.asarray(aois["level"].to_list(), dtype=str),
        aois["n_cells"].to_numpy(),
        values,
        origins,
        hashes,
        clipped,
        aois["hidden"].sum(),
    )


def score_model(
    panel: Panel,
    forecasts: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> pl.DataFrame:
    targets = panel.targets
    target_dates = FIRST_MONTH + targets.astype("timedelta64[M]")
    truth = (
        panel.values[:, targets.clip(max=len(panel.dates) - 1)]
        .transpose(1, 0, 2)
        .astype(np.float64)
    )
    error = forecasts - truth
    log_error = np.abs(np.log1p(forecasts.astype(np.float64)) - np.log1p(truth))
    origin = int(panel.origins[0])
    active_share = (panel.values[:, : origin + 1] > 0).mean(axis=1)
    groups = {"all": np.ones(len(panel.ids), dtype=bool)}
    groups.update(
        {f"level:{level}": panel.levels == level for level in sorted(set(panel.levels))}
    )
    aoi_groups = (*FOCUS_IDS, *panel.ids[panel.levels == "continent"])
    groups.update({f"aoi:{aoi}": panel.ids == aoi for aoi in aoi_groups})
    groups["global:countries"] = np.isin(panel.levels, ("country", "kr"))
    groups.update(
        {
            "active:0-10%": active_share <= 0.10,
            "active:10-50%": (active_share > 0.10) & (active_share <= 0.50),
            "active:50-100%": active_share > 0.50,
        }
    )
    rows = []
    periods = PERIODS | {
        "test_2024": ("2024-01", "2024-12"),
        "test_2025": ("2025-01", "2025-12"),
    }
    with_intervals = lower is not None and upper is not None
    for period, (start, stop) in periods.items():
        mask = (target_dates >= np.datetime64(start)) & (
            target_dates <= np.datetime64(stop)
        )
        for horizon in range(HORIZON + 1):
            cases = mask.copy()
            if horizon:
                cases[:, np.arange(HORIZON) != horizon - 1] = False
            origin_idx, lead_idx = np.where(cases)
            for group, included in groups.items():
                aoi_idx = np.flatnonzero(included)
                if not aoi_idx.size:
                    continue
                idx = (origin_idx[:, None], aoi_idx[None, :], lead_idx[:, None])
                residuals = error[idx]
                observed = truth[idx]
                total = observed.sum()
                rows.append(
                    {
                        "period": period,
                        "group": group,
                        "horizon": horizon,
                        "n_aois": len(aoi_idx),
                        "n": residuals.size,
                        "log_mae": log_error[idx].mean(),
                        "mae": np.abs(residuals).mean(),
                        "rmse": np.sqrt(np.square(residuals).mean()),
                        "wape": np.abs(residuals).sum() / total if total else None,
                        "bias": residuals.sum() / total if total else None,
                        "coverage80": (
                            (lower[idx] <= observed) & (observed <= upper[idx])
                        ).mean()
                        if with_intervals
                        else None,
                    }
                )
    return pl.DataFrame(rows)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _save_output(
    folder: Path,
    panel: Panel,
    point: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> None:
    metrics = score_model(panel, point, lower, upper)
    archive = {
        "ids": panel.ids,
        "origins": panel.dates[panel.origins[:-1]],
        "point": point[:-1],
    }
    columns = {
        "aoi_id": np.repeat(panel.ids, HORIZON),
        "level": np.repeat(panel.levels, HORIZON),
        "origin": np.repeat(str(AS_OF), len(panel.ids) * HORIZON),
        "target": np.tile(
            (AS_OF + np.arange(1, HORIZON + 1).astype("timedelta64[M]")).astype(str),
            len(panel.ids),
        ),
        "horizon": np.tile(np.arange(1, HORIZON + 1), len(panel.ids)),
        "point": point[-1].ravel(),
    }
    if lower is not None and upper is not None:
        archive.update(q10=lower[:-1], q90=upper[:-1])
        columns.update(q10=lower[-1].ravel(), q90=upper[-1].ravel())
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "backtest.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **archive)
    (folder / "backtest.npz.tmp").replace(folder / "backtest.npz")
    _atomic_text(folder / "forecast.csv", pl.DataFrame(columns).write_csv())
    _atomic_text(folder / "metrics.csv", metrics.write_csv())


def _events(folder: Path) -> list[dict]:
    history = folder / "runs.jsonl"
    if history.exists():
        return [
            json.loads(line)
            for line in history.read_text(encoding="utf-8").splitlines()
            if line
        ]
    return []


def _commit_event(
    folder: Path, panel: Panel, model: str, status: str, **details
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(WORK_DIR / "report.lock"):
        now = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
        event = {"model": model, "status": status, "time": now, **details}
        history = folder / "runs.jsonl"
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        _render(folder, panel)


def _format(value: float | None, percentage: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:.2%}" if percentage else f"{value:.4f}"


def _render(folder: Path, panel: Panel) -> None:
    events = _events(folder)
    latest = {
        event["model"]: event for event in events if event["status"] == "complete"
    }
    sections = [START_MARKER, "", "## 전 세계 합계 성능", ""]
    metrics = []
    for model, event in latest.items():
        saved = folder / event.get("result_dir", model) / "metrics.csv"
        metrics.append(pl.read_csv(saved).with_columns(pl.lit(model).alias("model")))
    if metrics:
        combined = pl.concat(metrics, how="diagonal_relaxed").sort(
            "model", "period", "group", "horizon"
        )
        _atomic_text(folder / "metrics.csv", combined.write_csv())
        world = combined.filter(pl.col("group") == "aoi:WORLD")
        _atomic_text(folder / "world_metrics.csv", world.write_csv())
        world_scores = {
            (r["model"], r["period"], r["horizon"]): r for r in world.to_dicts()
        }
        models = sorted(
            latest,
            key=lambda m: world_scores[m, "validation", 0]["wape"],
        )
        sections += [
            "| 모델 | 유형 | 2020–2023 WAPE | 2024–2025 WAPE | 1개월 | 2개월 | 3개월 | 2026년 1–7월 WAPE | 홀드아웃 편향 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for model in models:
            keys = [
                ("validation", 0),
                ("test", 0),
                ("test", 1),
                ("test", 2),
                ("test", 3),
                ("recent_sp", 0),
            ]
            values = [
                _format(world_scores[model, period, horizon]["wape"], True)
                for period, horizon in keys
            ]
            bias = _format(world_scores[model, "test", 0]["bias"], True)
            sections.append("| " + " | ".join([*MODELS[model], *values, bias]) + " |")
        sections += [
            "",
            "행은 v1.2의 전 세계 합계 `WORLD`에 대한 2020–2023년 WAPE 순입니다. 1·2·3개월 열은 2024–2025년 홀드아웃입니다.",
            "",
            "### 전 세계 지역별 성능",
            "",
            "| 모델 | 5,051개 지역 검증 log MAE | 지역별 홀드아웃 log MAE | 지역별 2026년 log MAE | 8개 대륙·해역 WAPE | 249개 국가 단위 WAPE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        aggregates = combined.filter(pl.col("horizon") == 0).to_dicts()
        scores = {(r["model"], r["period"], r["group"]): r for r in aggregates}
        for model in models:
            values = [
                _format(scores[model, period, "all"]["log_mae"]) for period in PERIODS
            ]
            values += [
                _format(scores[model, "test", group]["wape"], True)
                for group in ("level:continent", "global:countries")
            ]
            sections.append("| " + " | ".join([MODELS[model][0], *values]) + " |")
    else:
        sections.append("완료된 비교 실험의 지표가 여기에 누적됩니다.")
    sections += [
        "",
        "## 실행 기록",
        "",
        "| 모델 | 상태 | 장치 | CPU 스레드/작업 수 | 시간 | 세부 기록 |",
        "|---|---|---|---:|---|---|",
    ]
    for event in events:
        elapsed = f"{event['seconds']:.1f}초" if "seconds" in event else ""
        detail = event.get("error", elapsed).replace("|", " / ").replace("\n", " ")
        fields = [
            MODELS[event["model"]][0],
            event["status"],
            event.get("device", "—"),
            str(event.get("threads", "—")),
            event["time"].replace("T", " "),
            detail,
        ]
        sections.append("| " + " | ".join(fields) + " |")
    if not events:
        sections.append("| — | 준비 완료 | — | — | — | — |")
    sections += ["", END_MARKER]
    generated = "\n".join(sections)
    document = (
        REPORT.read_text(encoding="utf-8") if REPORT.exists() else _report_header(panel)
    )
    pattern = re.escape(START_MARKER) + r"[\s\S]*?" + re.escape(END_MARKER)
    if re.search(pattern, document):
        document = re.sub(pattern, lambda _: generated, document, count=1)
    else:
        document += "\n" + generated + "\n"
    _atomic_text(REPORT, document)


def _report_header(panel: Panel) -> str:
    return f"""# 월간 열점 탐지량 예측 실험

v1.2의 `data/processed/aoi_month.parquet`에서 지역별 월간 S-NPP 기준 탐지 환산량 `value`를 예측합니다.
`aoi_list.parquet`의 {len(panel.ids):,}곳 모두를 사용했습니다. 배포 화면에서 숨김으로 표시된 {panel.hidden_count:,}곳도 포함했습니다.
데이터셋 및 실험 키는 `{panel.key}`입니다. 두 입력 파일의 SHA-256을 배포 매니페스트와 대조했습니다.

## 비교 조건

- 사용 가능한 S-NPP 이력은 2012년 2월부터 시작합니다. 월별 누락·중복이 없는 지역 패널을 확인했습니다. 정적 열원 차감 후 남은 음수 {panel.clipped:,}건과 음수 예측은 0으로 처리합니다.
- 2020–2023년은 후보 비교, 2024–2025년은 시계열 홀드아웃, 2026년 1–7월은 추가 정식 처리본(SP) 검증에 사용합니다. 분할은 예측 대상월 기준입니다.
- 각 기준월까지 관측한 뒤 1·2·3개월 앞을 예측합니다. 사전학습 모델은 가중치를 유지하고, 지역별 통계 모델은 그때까지의 이력으로 적합합니다. 공동 학습 모델은 기준월까지 지나간 가장 최근 12월까지만 학습합니다.
- 입력 특성은 마감월까지의 관측 이력과 당시 알 수 있는 달력·지역 정보로 구성합니다. 2026년 8–10월 NRT는 10월 말 기준 전망의 입력에 사용합니다. 11월 진행 중 집계와 NOAA-20 전환기는 실적 평가에서 제외합니다.
- 전 세계 합계 `WORLD`의 WAPE를 주 지표로 사용합니다. WAPE는 절대 오차 합을 실적 합으로 나눈 값입니다. 지역별 비교에는 `abs(log1p(예측) - log1p(실적))`의 평균을 사용합니다.
- 이 실험은 하나의 배포 스냅샷을 대상으로 한 과거 검증입니다. 관측 자료 정정과 고정 열원 제거가 배포 시점 기준으로 반영되어 있습니다.

## 산출물

`data/processed/forecast_benchmark/{panel.key}/`에 세부 지표 `metrics.csv`, 전 세계 합계 지표 `world_metrics.csv`, 실행 이력 `runs.jsonl`, 모델별 과거 예측 `backtest.npz`, 2026년 10월 말 기준 전망 `forecast.csv`를 저장합니다.
세부 지표에는 행정 단계, 주요 지역, 과거 활동 빈도, 홀드아웃 연도별 MAE·RMSE·WAPE·편향과 제공되는 경우 80% 구간 포함률이 있습니다.
과거 예측 배열 축은 기준월 × AOI × 리드타임이며, `ids`와 `origins`가 배열 순서를 명시합니다.
"""


def run_model(
    panel: Panel, model: str, device: str, threads: int, batch_size: int, force: bool
) -> None:
    folder = WORK_DIR / panel.key
    output = folder / model
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "run.lock", timeout=0):
        with FileLock(WORK_DIR / "report.lock"):
            history = [e for e in _events(folder) if e["model"] == model]
        completed = any(e["status"] == "complete" for e in history)
        if completed and not force:
            print(
                f"{model} already completed ({panel.key}). Pass --force to rerun.",
                flush=True,
            )
            return
        attempt = 1 + sum(e["status"] == "running" for e in history)
        result_dir = output / f"attempt-{attempt}" if completed else output
        settings = {"device": device, "threads": threads, "batch_size": batch_size}
        _commit_event(folder, panel, model, "running", **settings)
        started = time.perf_counter()
        try:
            if model in FOUNDATION_MODELS:
                from planetarium.modeling.forecast_foundation import predict
            else:
                from planetarium.modeling.forecast_candidates import predict
            point, lower, upper, config = predict(
                model, panel, HORIZON, device, threads, batch_size
            )
            shape = (len(panel.origins), len(panel.ids), HORIZON)
            for array in [point, *(x for x in (lower, upper) if x is not None)]:
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(
                        f"Invalid forecasts: expected {shape}, got {array.shape}."
                    )
                np.maximum(array, 0, out=array)
            if (lower is None) != (upper is None) or (
                lower is not None and (np.any(lower > point) or np.any(point > upper))
            ):
                raise ValueError(
                    "The point forecast must be inside the reported interval."
                )
            _save_output(result_dir, panel, point, lower, upper)
            seconds = time.perf_counter() - started
            _commit_event(
                folder,
                panel,
                model,
                "complete",
                seconds=round(seconds, 2),
                **settings,
                config=config,
                result_dir=result_dir.relative_to(folder).as_posix(),
            )
            print(f"{model} complete ({seconds:.1f}s)", flush=True)
        except Exception as exc:
            _commit_event(
                folder,
                panel,
                model,
                "failed",
                **settings,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "run" and args.model is None:
        parser.error("run requires --model")
    if min(args.threads, args.batch_size) < 1:
        parser.error("threads and batch size must be positive")
    panel = read_panel()
    if args.command == "run":
        run_model(
            panel, args.model, args.device, args.threads, args.batch_size, args.force
        )
    else:
        folder = WORK_DIR / panel.key
        folder.mkdir(parents=True, exist_ok=True)
        with FileLock(WORK_DIR / "report.lock"):
            _render(folder, panel)
        print(REPORT)


if __name__ == "__main__":
    main()
