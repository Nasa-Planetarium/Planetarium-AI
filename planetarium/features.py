"""통합 핫스팟(점)을 '0.1° 격자 칸 × 현지 날짜 × 위성' 단위로 집계한다.

왜 칸 단위인가
    MODIS(1km)는 불 하나를 점 1개로, VIIRS(375m)는 여러 개로 잡는다. 점끼리 맞추는 대신
    더 큰 공통 격자(0.1° ≈ 11km)로 묶어 센서를 비교하고, 남는 감지 수 차이는 보정 모델이 다룬다.

입력  data/interim/firms_hotspots/*.parquet   (planetarium.dataset 결과; 파일별 중복 제거 규칙 적용)
출력  data/processed/cell_day_<res>deg/<입력과 같은 이름>.parquet
      data/processed/static_source_cells_0.01deg.parquet   고정 열원 칸 (약 1km 격자)

열원 종류(type) 처리
    SP 자료는 type 0(식생 화재)만 남긴다. NRT에는 type이 없으므로, SP 자료에서
    type 1(화산)·2(기타 고정 열원: 공장·가스 연소탑 등)가 2개 연도 이상 나온 0.01° 칸을
    '고정 열원 칸'으로 만들고 NRT 감지 중 그 칸에 있는 것을 뺀다.

출력은 희소 표: 감지가 1건 이상인 (칸, 날짜, 위성) 조합만 행으로 남는다.
행이 없다고 '감지 0'인지 '관측 못 함'인지는 알 수 없다 → planetarium.monthly의 위성별 자료 상태(status)로 구분.
감지 수는 산불 건수가 아니라 위성이 감지한 열 이상 픽셀 수(관측 활동)다.

실행 예
    uv run python -m planetarium.features                    # 전체 (다섯 위성), 0.1°
    uv run python -m planetarium.features --force            # 마스크·집계 모두 다시
    uv run python -m planetarium.features --files viirs-noaa20_2020 dl_nrt_*
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
import time
from pathlib import Path

import polars as pl
from tqdm import tqdm

from planetarium.config import PROCESSED_DATA_DIR, cell_day_dir
from planetarium.dataset import dedupe_plans, load_coverage, merged_files, sat_display, scan_hotspot_file, _scan_one
from planetarium.runlog import start_run_log

FIRE_TYPE = 0              # 식생 화재
STATIC_TYPES = [1, 2]      # 화산, 기타 고정 열원
STATIC_MIN_YEARS = 2       # 이 연도 수 이상 나온 칸만 고정 열원으로
# 마스크 격자는 집계 격자와 따로 0.01°(약 1km)로 고정.
# (이전 검증: 2012~2024 아카이브로 만든 마스크를 2025 S-NPP에 적용 시
#  0.1°는 고정 열원 97.8% 제거·진짜 산불 4.9%(한국 31%) 같이 제거, 0.01°는 95.5%·0.8%(한국 8%))
STATIC_RES = 0.01

CELL_DAY_COLUMNS = [
    "cell_id", "lat_idx", "lon_idx", "cell_lat", "cell_lon",
    "date_local", "year", "month", "sensor", "satellite", "source",
    "n_det", "n_det_conf", "n_day", "n_night", "frp_sum", "frp_max", "pixel_area_km2_sum",
]


def static_mask_path(res: float = STATIC_RES) -> Path:
    return PROCESSED_DATA_DIR / f"static_source_cells_{res:g}deg.parquet"


def cell_expr(res: float) -> dict[str, pl.Expr]:
    """위경도 -> 격자 인덱스. cell_id = lat_idx * round(360/res) + lon_idx"""
    n_lon = round(360 / res)
    lat_idx = ((pl.col("latitude") + 90) / res).floor().cast(pl.Int32)
    lon_idx = ((pl.col("longitude") + 180) / res).floor().cast(pl.Int32) % n_lon
    return {"lat_idx": lat_idx, "lon_idx": lon_idx, "cell_id": lat_idx.cast(pl.Int64) * n_lon + lon_idx}


def cell_center(res: float) -> dict[str, pl.Expr]:
    return {
        "cell_lat": ((pl.col("lat_idx") + 0.5) * res - 90).cast(pl.Float32),
        "cell_lon": ((pl.col("lon_idx") + 0.5) * res - 180).cast(pl.Float32),
    }


def build_static_mask(res: float = STATIC_RES) -> pl.DataFrame:
    """type이 있는 모든 SP 파일(연간 아카이브 + dl_archive)의 type 1·2 감지로 고정 열원 칸 목록을 만든다."""
    plans = dedupe_plans(load_coverage())
    parts = []
    for p in tqdm(merged_files(), desc="고정 열원 마스크", unit="file", leave=False):
        if p.name.startswith(("dl_nrt_", "nrt7d_")):
            continue  # NRT는 type이 없음
        parts.append(
            _scan_one(p, plans.get(p.name))
            .filter(pl.col("type").is_in(STATIC_TYPES))
            .with_columns(**cell_expr(res), year=pl.col("acq_datetime_utc").dt.year())
            .group_by("cell_id", "year")
            .agg(n=pl.len())
            .collect()
        )
    mask = (
        pl.concat(parts)
        .group_by("cell_id")
        .agg(n_years=pl.col("year").n_unique(), n_det=pl.col("n").sum())
        .filter(pl.col("n_years") >= STATIC_MIN_YEARS)
        .sort("cell_id")
    )
    path = static_mask_path(res)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.write_parquet(path)
    print(f"[완료] {path.name}: 고정 열원 칸 {mask.height:,}개 (type 1·2가 {STATIC_MIN_YEARS}개 연도 이상)")
    return mask


def aggregate_file(path: Path, res: float, static_cells: pl.Series) -> pl.DataFrame:
    """핫스팟 Parquet 1개 (중복 제거 적용) -> 칸 × 현지 날짜 × 위성 × source 집계."""
    static_cell = cell_expr(STATIC_RES)["cell_id"]
    return (
        scan_hotspot_file(path)
        .filter(
            (pl.col("type") == FIRE_TYPE)
            | (pl.col("type").is_null() & ~static_cell.is_in(static_cells.implode()))
        )
        .with_columns(
            **cell_expr(res),
            # 통합 스키마(VIIRS 기준)에서 집계용 값을 계산: 센서 이름, 위성 표시 이름, 픽셀 면적
            sensor=pl.col("instrument"),
            satellite=sat_display(),
            pixel_area_km2=pl.col("scan") * pl.col("track"),
        )
        .with_columns(
            # 현지 태양시 날짜: UTC + 경도/15 시간. 낮·밤 관측이 같은 현지 하루로 묶인다 (한국은 KST와 거의 같음).
            date_local=(
                pl.col("acq_datetime_utc").dt.replace_time_zone(None)
                + pl.duration(minutes=(pl.col("longitude") * 4).round().cast(pl.Int64))
            ).dt.date(),
        )
        .group_by("cell_id", "lat_idx", "lon_idx", "date_local", "sensor", "satellite", "source")
        .agg(
            n_det=pl.len().cast(pl.Int32),
            n_det_conf=(pl.col("confidence") != "l").sum().cast(pl.Int32),  # nominal·high
            n_day=(pl.col("daynight") == "D").sum().cast(pl.Int32),
            n_night=(pl.col("daynight") == "N").sum().cast(pl.Int32),
            frp_sum=pl.col("frp").sum(),
            frp_max=pl.col("frp").max(),
            pixel_area_km2_sum=pl.col("pixel_area_km2").sum(),
        )
        .with_columns(
            **cell_center(res),
            year=pl.col("date_local").dt.year().cast(pl.Int16),
            month=pl.col("date_local").dt.month().cast(pl.Int8),
        )
        .select(CELL_DAY_COLUMNS)
        .sort("cell_id", "date_local", "satellite")
        .collect()
    )


def load_cell_day(res: float = 0.1) -> pl.LazyFrame:
    """칸 × 날짜 집계 전체 (LazyFrame). 바로 collect하지 말고 지역 filter를 먼저 걸 것.

    예) load_cell_day().filter(pl.col("cell_lat").is_between(33, 39), pl.col("cell_lon").is_between(124, 132))
    """
    return pl.scan_parquet(cell_day_dir(res) / "*.parquet")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="핫스팟을 격자 칸 × 현지 날짜로 집계")
    ap.add_argument("--res", type=float, default=0.1, help="격자 크기(도). 기본 0.1")
    ap.add_argument("--files", nargs="*", help="입력 파일 이름 패턴(확장자 제외). 예: viirs-noaa20_* dl_nrt_*")
    ap.add_argument("--force", action="store_true", help="고정 열원 마스크와 집계 파일을 모두 다시 생성")
    args = ap.parse_args(argv)
    start_run_log("features")

    if abs(round(180 / args.res) - 180 / args.res) > 1e-6:
        sys.exit("--res 는 180을 나누어떨어지게 하는 값이어야 합니다 (예: 0.05, 0.1, 0.25, 0.5, 1).")
    inputs = merged_files()
    if args.files:
        inputs = [p for p in inputs if any(fnmatch.fnmatch(p.stem, pat) for pat in args.files)]
    if not inputs:
        sys.exit("입력 Parquet이 없습니다. 먼저 planetarium.dataset 을 실행하세요.")

    mask_path = static_mask_path()
    mask = pl.read_parquet(mask_path) if mask_path.exists() and not args.force else build_static_mask()
    static_cells = mask["cell_id"]

    out_dir = cell_day_dir(args.res)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"입력 {len(inputs)}개, 격자 {args.res:g}° -> {out_dir}")
    for path in inputs:
        out_path = out_dir / path.name
        if out_path.exists() and not args.force:
            print(f"[있음] {out_path.name} (다시 만들려면 --force)")
            continue
        t0 = time.perf_counter()
        df = aggregate_file(path, args.res, static_cells)
        tmp = out_path.with_suffix(".parquet.tmp")
        df.write_parquet(tmp, compression="zstd")
        tmp.replace(out_path)
        print(f"[완료] {out_path.name}: {df.height:,}행 (칸 {df['cell_id'].n_unique():,}개, 감지 {df['n_det'].sum():,}), "
              f"{out_path.stat().st_size / 1e6:,.0f} MB, {time.perf_counter() - t0:,.0f}초")


if __name__ == "__main__":
    main()
