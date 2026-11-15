"""FIRMS 원본 파일 → 통합 핫스팟 형식 변환기 (harmonize_file).

입력: FIRMS 원본 그대로 — CSV 또는 zip
  · MODIS/VIIRS, 연간 아카이브(zip 안 국가별 CSV)·Archive Download(fire_archive/fire_nrt)·최근 7일·area API
  · 위성 표기 N/SNPP/N20/N21/J1/J2/1/2/Terra/Aqua/T/A 모두 (dataset.SATELLITE_CODE)
  · Landsat(fire_*_LS_*)·Readme는 건너뜀
처리: dataset.iter_standardized(원본 읽기 + VIIRS 기준 스키마 변환, 큰 파일도 일정 크기씩)
      → 같은 키(satellite, latitude, longitude, acq_datetime_utc) 중복 제거
      → 현지 달마다 hotspots.harmonize_detections (저장된 통합 핫스팟의 점 가중치가 있으면 그대로 사용)
      메모리를 아끼려고 입력을 임시 Parquet로 한 번 쓴 뒤 달 단위로 읽는다.
출력 (output_dir):
  harmonized_hotspots.<parquet|csv>   통합 핫스팟과 같은 스키마
  summary_cell_day.<…>                칸 × 현지 날짜: 기준 점 가중치 합(value, value_lo, value_hi, n_points)
  summary_aoi_month.<…>               AOI × 월: 기준 점 가중치의 면적 가중 합 (입력 파일에 들어 있는 점만)
  summary.json                        입력·행 수·위성·기간·대기 중(pending) 행 수·주의 사항
주의: 요약은 '입력 파일의 점'만 더한 값이다. 입력이 일부 지역·기간이면 aoi_month(전체)보다 작고,
      MODIS 기간 미감지 추정분(residual)도 들어가지 않는다.

사용 예
    from planetarium.harmonize import harmonize_file
    harmonize_file("data/raw/firms/active_fire_7d/2026-11-14/J1_VIIRS_C2_Global_7d.csv", "out/n20_7d")
"""

from __future__ import annotations

import calendar
import json
import re
import tempfile
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from planetarium.config import INTERIM_DATA_DIR
from planetarium.dataset import DEDUPE_KEY, DL_NAME, SCHEMA, iter_standardized
from planetarium.hotspots import OUT_COLS, Context, harmonize_detections, month_cell_values, unit_weights_month
from planetarium.modeling.names import METHOD_VERSION
from planetarium.runlog import peak_memory_gb

SKIP_MEMBER = re.compile(r"(_LS_|readme|\.txt$)", re.IGNORECASE)


def list_sources(path: Path) -> list[tuple[Path, str | None]]:
    """(파일, zip 안 멤버) 목록. zip이면 CSV 멤버 전부(Landsat 제외), CSV면 그 파일."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv") and not SKIP_MEMBER.search(Path(m).name)]
        if not members:
            raise ValueError(f"zip 안에 MODIS/VIIRS CSV가 없습니다: {path}")
        return [(path, m) for m in sorted(members)]
    if path.suffix.lower() == ".csv":
        if SKIP_MEMBER.search(path.name):
            raise ValueError(f"Landsat 파일은 통합하지 않습니다 (컬럼 뜻이 다름): {path.name}")
        return [(path, None)]
    raise ValueError(f"CSV 또는 zip만 지원합니다: {path.name}")


def instrument_hint(name: str) -> str | None:
    """파일 이름으로 센서를 알 수 있으면 사용 (Archive Download product 코드, 7일·API 파일 이름)."""
    m = DL_NAME.search(name)
    if m:
        return "MODIS" if m["product"].startswith("M-") else "VIIRS"
    up = name.upper()
    if up.startswith("MODIS") or "MODIS_" in up:
        return "MODIS"
    if "VIIRS" in up or up.startswith(("J1_", "J2_", "SUOMI")):
        return "VIIRS"
    return None


def read_to_parquet(path: Path, tmp: Path) -> dict:
    """입력을 통합 스키마 임시 Parquet로 (큰 파일도 일정 크기씩). 반환: 원본 행 수 등."""
    schema = pl.DataFrame(schema=SCHEMA).to_arrow().schema
    n = 0
    members = []
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for p, member in list_sources(path):
            hint = instrument_hint(Path(member or p).name)
            k = 0
            for df in iter_standardized(p, member, hint):
                w.write_table(df.to_arrow().cast(schema))
                k += df.height
            members.append({"member": member or p.name, "rows": k, "instrument_hint": hint})
            n += k
    if n == 0:
        raise ValueError(f"감지 행이 없습니다: {path}")
    return {"rows_in": n, "members": members}


def _write(df: pl.DataFrame, path: Path, fmt: str, writer_state: dict) -> None:
    """같은 출력 파일에 달마다 이어 쓰기 (parquet: ParquetWriter, csv: 헤더 한 번)."""
    if fmt == "parquet":
        if path not in writer_state:
            writer_state[path] = pq.ParquetWriter(path, df.to_arrow().schema, compression="zstd")
        writer_state[path].write_table(df.to_arrow().cast(writer_state[path].schema))
    else:
        with open(path, "a", encoding="utf-8", newline="") as f:
            df.write_csv(f, include_header=path not in writer_state)
        writer_state[path] = True


def harmonize_file(path: str | Path, output_dir: str | Path | None = None, fmt: str = "parquet",
                   ctx: Context | None = None, membership: pl.DataFrame | None = None) -> dict:
    """FIRMS 원본 파일 하나를 통합 핫스팟 형식으로 바꾸고 요약을 만든다. 반환: 요약 dict (summary.json과 같음)."""
    t0 = time.perf_counter()
    path = Path(path)
    if fmt not in ("parquet", "csv"):
        raise ValueError("fmt는 parquet 또는 csv")
    out = Path(output_dir) if output_dir else INTERIM_DATA_DIR / "converted" / path.stem
    out.mkdir(parents=True, exist_ok=True)
    ext = "parquet" if fmt == "parquet" else "csv"
    files = {k: out / f"{k}.{ext}" for k in ("harmonized_hotspots", "summary_cell_day", "summary_aoi_month")}
    for f in files.values():
        f.unlink(missing_ok=True)
    ctx = ctx or Context()
    if membership is None:
        from planetarium.harmonize import aoi_membership
        membership, _ = aoi_membership()

    from planetarium.static_exclusion import is_static_expr, mask_ids
    static_mask = mask_ids()
    with tempfile.TemporaryDirectory(dir=INTERIM_DATA_DIR) as td:
        tmp = Path(td) / "input.parquet"
        info = read_to_parquet(path, tmp)
        lf = pl.scan_parquet(tmp).unique(subset=DEDUPE_KEY, keep="first")
        n_unique = lf.select(pl.len()).collect().item()
        # 현지 달 목록 (현지 날짜 = UTC + 경도/15h)
        local = (pl.col("acq_datetime_utc").dt.replace_time_zone(None)
                 + pl.duration(minutes=(pl.col("longitude") * 4).round().cast(pl.Int64))).dt.date()
        months = (lf.select(y=local.dt.year(), m=local.dt.month()).unique().sort("y", "m").collect().rows())
        writers: dict = {}
        stats = {"rows_out": 0, "reference": 0, "pending": 0, "by_satellite": {}}
        try:
            for y, m in months:
                first, last = date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
                pts = lf.filter(pl.col("acq_date").is_between(first - timedelta(days=1), last + timedelta(days=1)))
                h = harmonize_detections(pts, month_cell_values(y, m), ctx, unit_weights_month(y, m)).filter(
                    pl.col("date_local").is_between(first, last))
                if h.height == 0:
                    continue
                # v1.2: 고정 열원 0.01° 칸 안의 점은 표시만 하고 요약에서 뺀다 (서버 집계와 같은 규칙)
                h = h.with_columns(is_static_source=is_static_expr(static_mask))
                _write(h.select(*OUT_COLS, "is_static_source"), files["harmonized_hotspots"], fmt, writers)
                stats["static"] = stats.get("static", 0) + int(h["is_static_source"].sum())
                ref = h.filter("is_reference", ~pl.col("is_static_source"))
                cd = ref.group_by("cell_id", "date_local").agg(
                    value=pl.col("harmonized_weight").cast(pl.Float64).sum(),
                    value_lo=pl.col("weight_lo").cast(pl.Float64).sum(),
                    value_hi=pl.col("weight_hi").cast(pl.Float64).sum(),
                    n_points=pl.len().cast(pl.Int32),
                    pending=pl.col("harmonized_weight").is_null().sum().cast(pl.Int32),
                    value_source=pl.col("weight_source").mode().first(),
                ).sort("cell_id", "date_local")
                _write(cd, files["summary_cell_day"], fmt, writers)
                am = (ref.lazy().join(membership.lazy(), on="cell_id")
                      .group_by("aoi_id").agg(value=(pl.col("harmonized_weight").cast(pl.Float64) * pl.col("weight")).sum(),
                                              n_points=pl.len())
                      .with_columns(year=pl.lit(y, dtype=pl.Int16), month=pl.lit(m, dtype=pl.Int8))
                      .select("aoi_id", "year", "month", "value", "n_points").collect().sort("aoi_id"))
                _write(am, files["summary_aoi_month"], fmt, writers)
                stats["rows_out"] += h.height
                stats["reference"] += ref.height
                stats["pending"] += int(h["weight_source"].eq("pending_new_month").sum())
                for s, c in h.group_by("satellite").len().iter_rows():
                    stats["by_satellite"][s] = stats["by_satellite"].get(s, 0) + c
        finally:
            for w in writers.values():
                if hasattr(w, "close"):
                    w.close()

    summary = {
        "input": str(path), "output_dir": str(out), "format": fmt, "method_version": METHOD_VERSION,
        "rows_in": info["rows_in"], "rows_unique": n_unique, "members": info["members"],
        "months": [f"{y}-{m:02d}" for y, m in months],
        "rows_out": stats["rows_out"], "reference_rows": stats["reference"], "pending_new_month_rows": stats["pending"],
        "rows_by_satellite": stats["by_satellite"],
        "static_source_rows": stats.get("static", 0),
        "notes": [
            "summary_*는 입력 파일의 점만 더한 값 (일부 지역·기간이면 aoi_month 전체보다 작음)",
            "MODIS 기간 미감지 추정분(residual)은 점이 아니라 요약에 포함되지 않음",
            "고정 열원(type≠0, NRT는 고정 열원 마스크)은 제외됨 → rows_out < rows_unique",
            "v1.2: SP type 0이라도 고정 열원 0.01° 칸 안의 점은 is_static_source=true로 표시하고 summary_*에서 뺌",
            "pending_new_month: 통합 표에 없는 달의 기준 점 (새 달 MODIS 예측은 변환기 2단계에서 추가 예정)",
        ],
        "seconds": round(time.perf_counter() - t0, 1),
        "peak_memory_gb": round(peak_memory_gb() or 0, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
