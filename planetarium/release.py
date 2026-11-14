"""배포 파일 묶기와 이전 배포 대비 비교.

compare  : 이전 배포(release_v0-b1-global-2026-11-14-r3) 대비 현재 산출물
           - 한국(KR)·아르헨티나(ADM0_ARG)·전 세계(WORLD) 2000~2011 연간 합계 변화
           - 한국 캘린더 판정(above/similar/below/uncertain) 변화 개수 (KR 전체 월)
           - 그림 reports/figures/v0_vs_v1_korea_annual.png
           - reports/v0_vs_v1_annual.csv, reports/v0_vs_v1_korea_verdict.csv
copy     : data/processed/release_<method_version>/ 에 앱용 산출물 복사

실행  uv run python -m planetarium.release --compare --copy
"""

from __future__ import annotations

import argparse
import json
import shutil

import polars as pl

from planetarium.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from planetarium.modeling.train_m1 import META_PATH

V0_DIR = PROCESSED_DATA_DIR / "release_v0-b1-global-2026-11-14-r3"
RELEASE_FILES = ["harmonized_cell_month.parquet", "aoi_month.parquet", "aoi_month_compare.parquet",
                 "aoi_list.parquet", "aoi_boundaries.geojson"]
FOCUS = {"KR": "Korea", "ADM0_ARG": "Argentina", "WORLD": "World"}


def annual(dirpath, tag: str) -> pl.DataFrame:
    am = pl.read_parquet(dirpath / "aoi_month.parquet").filter(pl.col("aoi_id").is_in(list(FOCUS)))
    return am.group_by("aoi_id", "year").agg(
        pl.col("value").sum().alias(f"value_{tag}"), pl.col("value").is_null().sum().alias(f"missing_months_{tag}"))


def compare() -> tuple[pl.DataFrame, pl.DataFrame]:
    a0, a1 = annual(V0_DIR, "v0"), annual(PROCESSED_DATA_DIR, "v1")
    ann = a0.join(a1, on=["aoi_id", "year"], how="full", coalesce=True).with_columns(
        change_pct=(pl.col("value_v1") / pl.col("value_v0") - 1) * 100).sort("aoi_id", "year")
    ann.write_csv(REPORTS_DIR / "v0_vs_v1_annual.csv")

    c0 = pl.read_parquet(V0_DIR / "aoi_month_compare.parquet").filter(pl.col("aoi_id") == "KR").select("year", "month", v0="verdict")
    c1 = pl.read_parquet(PROCESSED_DATA_DIR / "aoi_month_compare.parquet").filter(pl.col("aoi_id") == "KR").select("year", "month", v1="verdict")
    ver = c0.join(c1, on=["year", "month"], how="full", coalesce=True)
    ver.filter(pl.col("v0") != pl.col("v1")).sort("year", "month").write_csv(REPORTS_DIR / "v0_vs_v1_korea_verdict.csv")
    plot(ann)
    return ann, ver


def plot(ann: pl.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    k = ann.filter(pl.col("aoi_id") == "KR", pl.col("year") <= 2025).sort("year")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(k["year"], k["value_v0"], marker="o", color="#1f77b4", label="v0-r3 (ratio_lat)")
    ax.plot(k["year"], k["value_v1"], marker="o", color="#d62728", label=f"{meta['method_version']} (m2v_xgb)")
    ax.axvspan(1999.5, 2011.5, color="grey", alpha=0.12, label="MODIS-harmonized period (2000-11 ~ 2012-01)")
    ax.set_title("Korea annual VIIRS-equivalent detections: v0 vs v1")
    ax.set_ylabel("detections / year")
    ax.set_xlabel("year (2000 = Nov-Dec only)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "v0_vs_v1_korea_annual.png", dpi=130)
    plt.close(fig)


def copy_release() -> str:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    d = PROCESSED_DATA_DIR / f"release_{meta['method_version']}"
    d.mkdir(parents=True, exist_ok=True)
    for f in RELEASE_FILES:
        shutil.copy2(PROCESSED_DATA_DIR / f, d / f)
    shutil.copy2(META_PATH, d / META_PATH.name)
    return str(d)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="배포 파일 묶기와 이전 배포 대비 비교")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args(argv)
    if args.compare:
        ann, ver = compare()
        with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                       tbl_hide_dataframe_shape=True, float_precision=1, thousands_separator=","):
            print(ann.filter(pl.col("year") <= 2011).pivot(on="aoi_id", index="year", values="change_pct"))
            print(ver.group_by("v0", "v1").len().sort("v0", "v1"))
    if args.copy:
        print("복사:", copy_release())


if __name__ == "__main__":
    main()
