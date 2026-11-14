"""FIRMS MODIS·VIIRS 핫스팟 원본을 하나의 공통 스키마 Parquet로 통합한다.

입력 세 종류 (data/raw/는 읽기만 한다)
  1) 연간 아카이브 (정식 처리본 SP, 전 세계)
       data/raw/firms/<폴더>/<접두어>_<year>_all_countries.zip
       폴더 modis / viirs-snpp / viirs-noaa20 (NOAA-20은 zip 접두어가 viirs-jpss1)
  2) FIRMS Archive Download (기간·지역 지정; fire_archive=SP, fire_nrt=NRT)
       data/raw/firms/downloads/*.zip|*.csv 안의 fire_<kind>_<product>_<id>.csv
  3) FIRMS 최근 7일 NRT (전 세계)
       data/raw/firms/active_fire_7d/<받은 날짜>/<MODIS_C6_1|J1_VIIRS_C2|J2_VIIRS_C2>_Global_7d.csv

출력
  data/interim/firms_hotspots/<폴더>_<year>.parquet          (1)
  data/interim/firms_hotspots/dl_<kind>_<slug>_<id>.parquet   (2)
  data/interim/firms_hotspots/nrt7d_<slug>_<날짜>.parquet      (3)
  data/interim/firms_coverage.parquet                         파일 × 위성별 시간·공간 범위
  reports/firms_merge_summary.csv                             위성 × source × 연도별 행 수·기간
  reports/firms_dedupe_summary.csv                            중복 제거로 빠진 행 수

중복 제거 (load_hotspots(dedupe=True), 기본값)
  같은 관측이 여러 파일에 들어 있으므로 위성별로 우선순위를 둔다.
      tier 0  SP   : 연간 아카이브, dl_archive_*
      tier 1  NRT  : dl_nrt_* (Archive Download로 받은 NRT)
      tier 2  NRT  : nrt7d_* (최근 7일 파일)
  낮은 순위 파일의 감지는, 더 높은 순위 파일이 '같은 위성 · 같은 시간 범위 · 같은 지역'을 덮고 있으면 버린다.
  날짜 단위가 아니라 파일의 실제 시간 범위(첫 날 00:00 ~ 마지막 감지 시각)로 판단한다.
  예) drive zip NRT는 2026-11-11 10시경(UTC)에 끝나므로, 그 이후 7일 파일 감지는 남는다.
  지역은 파일의 감지 위경도 범위를 5° 단위로 바깥쪽 반올림해 추정한다 (Archive Download는 요청 상자가 파일에 없음).
  같은 순위끼리는 (satellite, latitude, longitude, acq_datetime_utc)가 같은 행을 하나만 남긴다.

주의: 핫스팟 수는 산불 건수가 아니라 위성이 감지한 열 이상 픽셀 수(관측 활동)다.

실행 예
    uv run python -m planetarium.dataset                          # 전체 (아카이브 + downloads + 7일)
    uv run python -m planetarium.dataset --sensor viirs-noaa20    # 아카이브 한 센서만
    uv run python -m planetarium.dataset --sensor modis --years 2000 2012-2014
    uv run python -m planetarium.dataset --downloads-only         # downloads + 7일 NRT만
    uv run python -m planetarium.dataset --force
    uv run python -m planetarium.dataset --summary-only           # 범위·요약·중복 제거 리포트만
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from planetarium.config import (
    ACTIVE_FIRE_7D_DIR,
    ARCHIVE_ZIP_PREFIX,
    DOWNLOADS_DIR,
    FIRMS_COVERAGE_PATH,
    FIRMS_MERGED_DIR,
    FIRMS_RAW_DIR,
    REPORTS_DIR,
    SENSORS,
)
from planetarium.runlog import start_run_log

FLUSH_ROWS = 2_000_000  # Parquet row group에 한 번에 쓰는 최소 행 수
BIG_CSV_BYTES = 300_000_000  # 이보다 큰 CSV는 임시 폴더에 풀어 나눠 읽는다 (메모리 절약)
DEDUPE_KEY = ["satellite", "latitude", "longitude", "acq_datetime_utc"]
BBOX_ROUND_DEG = 5.0

# 통합 스키마는 VIIRS 원본 기준 (MODIS는 2027년 종료 예정이라 VIIRS가 기준이고 MODIS가 여기에 붙는다).
#   VIIRS: 원본 컬럼명·값 표기 그대로 (Archive Download의 brightness/bright_t31은 bright_ti4/bright_ti5로 이름만 맞춤)
#   MODIS: brightness→bright_ti4, bright_t31→bright_ti5, confidence 0~100 → l(<30)/n(30~79)/h(≥80), 원래 값은 confidence_modis
#   ※ MODIS의 bright_ti4/ti5는 채널 21(22)/31이고 VIIRS의 I4/I5와 물리적으로 다르다. 모델·분석에서 섞지 말고 instrument로 구분할 것.
RENAME = {"brightness": "bright_ti4", "bright_t31": "bright_ti5"}

# 위성 표기: FIRMS 원본 코드로 통일 (VIIRS N/N20/N21, MODIS Terra/Aqua)
#   아카이브 N/N20, Archive Download SNPP/N20/N21, 일부 1/2, 7일 MODIS T/A
SATELLITE_CODE = {
    "Terra": "Terra", "T": "Terra",
    "Aqua": "Aqua", "A": "Aqua",
    "N": "N", "SNPP": "N", "S-NPP": "N", "Suomi NPP": "N",
    "N20": "N20", "1": "N20", "NOAA-20": "N20", "J1": "N20",
    "N21": "N21", "2": "N21", "NOAA-21": "N21", "J2": "N21",
}
# 집계·리포트용 표시 이름
SAT_DISPLAY = {"Terra": "Terra", "Aqua": "Aqua", "N": "S-NPP", "N20": "NOAA-20", "N21": "NOAA-21"}
SATELLITES = ["Terra", "Aqua", "S-NPP", "NOAA-20", "NOAA-21"]
SAT_CODE_OF = {v: k for k, v in SAT_DISPLAY.items()}

# VIIRS 신뢰도: 아카이브·다운로드 l/n/h, 7일 파일 low/nominal/high → l/n/h
VIIRS_CONF = {"l": "l", "n": "n", "h": "h", "low": "l", "nominal": "n", "high": "h"}

SCHEMA: dict[str, pl.DataType] = {
    # ── VIIRS 원본 컬럼 (순서도 원본과 같게)
    "latitude": pl.Float64(),
    "longitude": pl.Float64(),
    "bright_ti4": pl.Float32(),       # K (MODIS는 brightness: 채널 21/22)
    "scan": pl.Float32(),             # km
    "track": pl.Float32(),            # km
    "acq_date": pl.Date(),            # UTC 날짜
    "acq_time": pl.String(),          # UTC HHMM
    "satellite": pl.String(),         # N / N20 / N21 / Terra / Aqua
    "instrument": pl.String(),        # VIIRS / MODIS
    "confidence": pl.String(),        # l / n / h
    "version": pl.String(),
    "bright_ti5": pl.Float32(),       # K (MODIS는 bright_t31: 채널 31)
    "frp": pl.Float32(),              # MW
    "daynight": pl.String(),          # D / N
    "type": pl.Int8(),                # 0 식생 화재, 1 화산, 2 기타 고정 열원, 3 해상 (NRT는 null)
    # ── 추가 컬럼 (최소)
    "confidence_modis": pl.Int16(),   # MODIS 원래 신뢰도 0~100 (VIIRS는 null)
    "source": pl.String(),            # SP / NRT
    "acq_datetime_utc": pl.Datetime("us", "UTC"),
}
# 파생값(현지·KST 날짜, 연·월, 픽셀 면적 등)은 저장하지 않고 집계 단계에서 계산한다.

ARCHIVE_ZIP = re.compile(r"^(?P<prefix>[a-z0-9-]+)_(?P<year>\d{4})_all_countries\.zip$")
DL_NAME = re.compile(r"fire_(?P<kind>nrt|archive)_(?P<product>[A-Za-z0-9-]+)_(?P<id>\d+)\.csv$")
DL_PRODUCTS: dict[str, tuple[str, str] | None] = {  # product -> (instrument, slug); None이면 건너뜀
    "M-C61": ("MODIS", "modis"),
    "M-C6": ("MODIS", "modis"),
    "SV-C2": ("VIIRS", "viirs-snpp"),
    "J1V-C2": ("VIIRS", "viirs-noaa20"),
    "J2V-C2": ("VIIRS", "viirs-noaa21"),
    "LS": None,  # Landsat: scan/track이 픽셀 번호이고 밝기온도·FRP가 없어 통합하지 않음
}
NRT7D_NAME = re.compile(r"^(?P<product>MODIS_C6_1|J1_VIIRS_C2|J2_VIIRS_C2|SUOMI_VIIRS_C2)_Global_7d\.csv$")
NRT7D_PRODUCTS = {
    "MODIS_C6_1": ("MODIS", "modis"),
    "SUOMI_VIIRS_C2": ("VIIRS", "viirs-snpp"),
    "J1_VIIRS_C2": ("VIIRS", "viirs-noaa20"),
    "J2_VIIRS_C2": ("VIIRS", "viirs-noaa21"),
}


def sat_display(col: str = "satellite") -> pl.Expr:
    """위성 코드 → 표시 이름 (S-NPP, NOAA-20 ...)."""
    return pl.col(col).replace_strict(SAT_DISPLAY, default=pl.col(col), return_dtype=pl.String)


# ── 공통 스키마 변환 ───────────────────────────────────────────────────────

def standardize(df: pl.DataFrame, instrument: str) -> pl.DataFrame:
    """문자열로 읽은 원본 DataFrame -> 통합 스키마(VIIRS 기준). 아카이브·다운로드·7일 형식 모두 처리."""
    df = df.rename({k: v for k, v in RENAME.items() if k in df.columns})
    need = {"latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "bright_ti4", "frp"}
    if need - set(df.columns):
        raise ValueError(f"MODIS/VIIRS 핫스팟 형식이 아닙니다 (없는 컬럼: {sorted(need - set(df.columns))})")
    optional = {"bright_ti5", "scan", "track", "type", "version", "daynight"} - set(df.columns)
    df = df.with_columns(pl.lit(None, dtype=pl.String).alias(c) for c in optional)

    s = lambda c: pl.col(c).str.strip_chars()  # noqa: E731
    f32 = lambda c: s(c).cast(pl.Float32, strict=False)  # noqa: E731
    acq_time = s("acq_time").str.zfill(4)
    dt = (s("acq_date") + " " + acq_time).str.to_datetime("%Y-%m-%d %H%M", strict=False)
    version = s("version")

    if instrument == "MODIS":
        conf_modis = s("confidence").cast(pl.Int16, strict=False)
        conf = (pl.when(conf_modis < 30).then(pl.lit("l"))
                .when(conf_modis < 80).then(pl.lit("n"))
                .when(conf_modis <= 100).then(pl.lit("h")))
    else:
        conf_modis = pl.lit(None, dtype=pl.Int16)
        conf = s("confidence").str.to_lowercase().replace_strict(VIIRS_CONF, default=None, return_dtype=pl.String)

    out = df.select(
        latitude=s("latitude").cast(pl.Float64, strict=False),
        longitude=s("longitude").cast(pl.Float64, strict=False),
        bright_ti4=f32("bright_ti4"),
        scan=f32("scan"),
        track=f32("track"),
        acq_date=dt.dt.date(),
        acq_time=acq_time,
        satellite=s("satellite").replace_strict(SATELLITE_CODE, default=None, return_dtype=pl.String),
        # Archive Download의 S-NPP 파일은 instrument가 "SNPP"로 들어 있어 센서 이름으로 통일
        instrument=pl.lit(instrument),
        confidence=conf,
        version=version,
        bright_ti5=f32("bright_ti5"),
        frp=f32("frp"),
        daynight=s("daynight"),
        type=s("type").cast(pl.Int8, strict=False),
        confidence_modis=conf_modis,
        # FIRMS 속성 문서: 정식 처리(Standard)는 컬렉션 번호만("6.1", "2"), 빠른 처리본은 접미사
        # URT/RT/NRT가 붙는다. 숫자만이면 SP, 아니면 모두 NRT로 묶는다(URT·RT도 정식 처리 전 자료).
        source=pl.when(version.str.contains(r"^[0-9.]+$")).then(pl.lit("SP")).otherwise(pl.lit("NRT")),
        acq_datetime_utc=dt.dt.replace_time_zone("UTC"),
    ).cast(SCHEMA)

    bad_sat = out.filter(pl.col("satellite").is_null()).height
    if bad_sat:
        raw = df.filter(s("satellite").replace_strict(SATELLITE_CODE, default=None).is_null())["satellite"].unique().to_list()
        raise ValueError(f"모르는 위성 표기 {raw} ({bad_sat:,}행)")
    return out


def write_parquet_atomic(df: pl.DataFrame, out_path: Path) -> None:
    tmp = out_path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(out_path)


def describe(df: pl.DataFrame, out_path: Path, t0: float) -> str:
    rng = f"{df['acq_datetime_utc'].min():%Y-%m-%d %H:%M} ~ {df['acq_datetime_utc'].max():%Y-%m-%d %H:%M} UTC" if df.height else "-"
    sats = ",".join(sorted(SAT_DISPLAY.get(x, x) for x in df["satellite"].unique().to_list()))
    srcs = ",".join(sorted(df["source"].unique().to_list()))
    return (f"[완료] {out_path.name}: {df.height:,}행 [{sats} / {srcs}] {rng}, "
            f"{out_path.stat().st_size / 1e6:,.1f} MB, {time.perf_counter() - t0:,.0f}초")


# ── 1) 연간 아카이브 ──────────────────────────────────────────────────────

def zip_prefix(sensor_dir: str) -> str:
    return ARCHIVE_ZIP_PREFIX.get(sensor_dir, sensor_dir)


def find_archives(sensor_filter: str, years: set[int] | None) -> list[tuple[Path, str, int]]:
    inputs = []
    for sensor_dir in SENSORS:
        if sensor_filter not in ("all", sensor_dir):
            continue
        for zip_path in sorted((FIRMS_RAW_DIR / sensor_dir).glob("*.zip")):
            m = ARCHIVE_ZIP.match(zip_path.name)
            if not m or m["prefix"] != zip_prefix(sensor_dir):
                print(f"[건너뜀] 이름 형식이 다른 파일: {zip_path}", file=sys.stderr)
                continue
            year = int(m["year"])
            if years is None or year in years:
                inputs.append((zip_path, sensor_dir, year))
    return inputs


def merge_archive_zip(zip_path: Path, sensor_dir: str, year: int, out_path: Path) -> int:
    """연간 아카이브 zip 1개 -> Parquet 1개 (zip을 풀지 않고 국가별 CSV를 차례로 읽음)."""
    sensor = SENSORS[sensor_dir]
    schema = pl.DataFrame(schema=SCHEMA).to_arrow().schema
    tmp = out_path.with_suffix(".parquet.tmp")
    total, buffer, buffered = 0, [], 0
    with zipfile.ZipFile(zip_path) as zf, pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        def flush() -> None:
            nonlocal buffer, buffered
            if buffer:
                writer.write_table(pl.concat(buffer).to_arrow().cast(schema))
                buffer, buffered = [], 0

        members = sorted(m for m in zf.namelist() if m.lower().endswith(".csv"))
        for member in tqdm(members, desc=zip_path.name, unit="file", leave=False):
            df = pl.read_csv(io.BytesIO(zf.read(member)), infer_schema=False)
            if df.height == 0:
                continue
            df = standardize(df, sensor)
            buffer.append(df)
            buffered += df.height
            total += df.height
            if buffered >= FLUSH_ROWS:
                flush()
        flush()
    tmp.replace(out_path)
    return total


# ── 2) FIRMS Archive Download ─────────────────────────────────────────────

def find_downloads() -> list[tuple[Path, str | None, re.Match[str]]]:
    """(파일 경로, zip 안 멤버 또는 None, 이름 매치) 목록."""
    found: list[tuple[Path, str | None, re.Match[str]]] = []
    if not DOWNLOADS_DIR.exists():
        return found
    for path in sorted(DOWNLOADS_DIR.iterdir()):
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                pairs = [(n, Path(n).name) for n in sorted(zf.namelist()) if n.lower().endswith(".csv")]
        elif path.suffix.lower() == ".csv":
            pairs = [(None, path.name)]
        else:
            continue
        for member, fname in pairs:
            m = DL_NAME.search(fname)
            if not m:
                print(f"[건너뜀] 다운로드 파일 이름 형식이 다름: {path.name}:{fname}", file=sys.stderr)
                continue
            found.append((path, member, m))
    return found


def merge_download(path: Path, member: str | None, m: re.Match[str], force: bool) -> None:
    product = DL_PRODUCTS.get(m["product"], "unknown")
    where = f"{path.name}:{member or path.name}"
    if product == "unknown":
        print(f"[건너뜀] 모르는 product 코드 {m['product']}: {where}", file=sys.stderr)
        return
    if product is None:
        print(f"[건너뜀] {where}: Landsat은 컬럼 뜻이 달라 통합하지 않음")
        return
    sensor, slug = product
    out_path = FIRMS_MERGED_DIR / f"dl_{m['kind']}_{slug}_{m['id']}.parquet"
    if out_path.exists() and not force:
        print(f"[있음] {out_path.name} (다시 만들려면 --force)")
        return
    t0 = time.perf_counter()
    expected = "SP" if m["kind"] == "archive" else "NRT"
    size = (zipfile.ZipFile(path).getinfo(member).file_size if member else path.stat().st_size)
    if size > BIG_CSV_BYTES:
        n, got = convert_big_csv(path, member, sensor, out_path)
    else:
        if member is None:
            raw = path.read_bytes()
        else:
            with zipfile.ZipFile(path) as zf:
                raw = zf.read(member)
        df = standardize(pl.read_csv(io.BytesIO(raw), infer_schema=False), sensor)
        got = set(df["source"].unique().to_list())
        write_parquet_atomic(df, out_path)
    if got != {expected}:
        print(f"[주의] {where}: 파일 종류({m['kind']})와 version 기반 source {sorted(got)}가 다름", file=sys.stderr)
    print(describe(pl.read_parquet(out_path, columns=["acq_datetime_utc", "satellite", "source"]), out_path, t0)
          + f"  <- {where}")


def convert_big_csv(path: Path, member: str | None, instrument: str, out_path: Path) -> tuple[int, set[str]]:
    """수백 MB~GB CSV: 임시 폴더에 풀어 일정 행씩 읽고 변환해 Parquet에 이어 쓴다 (원본 zip은 그대로)."""
    import shutil
    import tempfile
    schema = pl.DataFrame(schema=SCHEMA).to_arrow().schema
    tmp_out = out_path.with_suffix(".parquet.tmp")
    total, sources = 0, set()
    with tempfile.TemporaryDirectory(dir=FIRMS_MERGED_DIR.parent) as td:
        if member is None:
            csv = path
        else:
            with zipfile.ZipFile(path) as zf, zf.open(member) as src, open(Path(td) / Path(member).name, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
            csv = Path(td) / Path(member).name
        import pyarrow as pa
        import pyarrow.csv as pacsv
        with open(csv, encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
        reader = pacsv.open_csv(  # 모든 컬럼을 문자열로 (standardize가 형 변환)
            csv, read_options=pacsv.ReadOptions(block_size=256 * 1024 * 1024),
            convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in header}))
        try:
            with pq.ParquetWriter(tmp_out, schema, compression="zstd") as writer:
                for batch in reader:
                    df = standardize(pl.from_arrow(pa.Table.from_batches([batch])), instrument)
                    sources |= set(df["source"].unique().to_list())
                    writer.write_table(df.to_arrow().cast(schema))
                    total += df.height
        finally:
            reader.close()  # Windows: 파일 핸들을 닫아야 임시 폴더를 지울 수 있다
            del reader
    tmp_out.replace(out_path)
    return total, sources


# ── 3) 최근 7일 NRT ───────────────────────────────────────────────────────

def find_nrt7d() -> list[tuple[Path, str, str]]:
    """(csv 경로, product, 받은 날짜 폴더) 목록."""
    found = []
    if not ACTIVE_FIRE_7D_DIR.exists():
        return found
    for path in sorted(ACTIVE_FIRE_7D_DIR.glob("*/*.csv")):
        m = NRT7D_NAME.match(path.name)
        if not m:
            print(f"[건너뜀] 7일 파일 이름 형식이 다름: {path}", file=sys.stderr)
            continue
        found.append((path, m["product"], path.parent.name))
    return found


def merge_nrt7d(path: Path, product: str, received: str, force: bool) -> None:
    sensor, slug = NRT7D_PRODUCTS[product]
    out_path = FIRMS_MERGED_DIR / f"nrt7d_{slug}_{received}.parquet"
    if out_path.exists() and not force:
        print(f"[있음] {out_path.name} (다시 만들려면 --force)")
        return
    t0 = time.perf_counter()
    df = standardize(pl.read_csv(path, infer_schema=False), sensor)
    # 전 세계 파일이지만 지역 필터는 하지 않는다: 크기가 작고(수십 MB),
    # T2의 위성별 '자료 있는 날' 판정에 전 세계 감지 수를 써야 하기 때문.
    write_parquet_atomic(df, out_path)
    print(describe(df, out_path, t0) + f"  <- {path.parent.name}/{path.name} (전 세계, 지역 필터 안 함)")


# ── 파일 순위·범위 ────────────────────────────────────────────────────────

def file_tier(path: Path) -> int:
    """중복 제거 우선순위 (작을수록 우선): 0 SP, 1 Archive Download NRT, 2 7일 NRT."""
    name = path.name
    if name.startswith("nrt7d_"):
        return 2
    if name.startswith("dl_nrt_"):
        return 1
    return 0  # 연간 아카이브, dl_archive_*


def merged_files() -> list[Path]:
    return sorted(FIRMS_MERGED_DIR.glob("*.parquet"))


def _round_out(lo: float, hi: float, step: float, lim: float) -> tuple[float, float]:
    import math
    return max(-lim, math.floor(lo / step) * step), min(lim, math.ceil(hi / step) * step)


def build_coverage() -> pl.DataFrame:
    """파일 × 위성별 시간·공간 범위와 같은 키 중복 수. data/interim/firms_coverage.parquet."""
    rows = []
    for path in tqdm(merged_files(), desc="범위 계산", unit="file", leave=False):
        stats = (
            pl.scan_parquet(path)
            .group_by("satellite", "source")
            .agg(
                rows=pl.len(),
                t_min=pl.col("acq_datetime_utc").min(),
                t_max=pl.col("acq_datetime_utc").max(),
                lat_min=pl.col("latitude").min(), lat_max=pl.col("latitude").max(),
                lon_min=pl.col("longitude").min(), lon_max=pl.col("longitude").max(),
                n_unique_key=pl.struct("latitude", "longitude", "acq_datetime_utc").n_unique(),
            )
            .collect()
        )
        for r in stats.iter_rows(named=True):
            lat0, lat1 = _round_out(r["lat_min"], r["lat_max"], BBOX_ROUND_DEG, 90)
            lon0, lon1 = _round_out(r["lon_min"], r["lon_max"], BBOX_ROUND_DEG, 180)
            if lon1 - lon0 >= 300:  # 사실상 전 세계
                lat0, lat1, lon0, lon1 = -90.0, 90.0, -180.0, 180.0
            t_min: datetime = r["t_min"]
            rows.append({
                "file": path.name, "tier": file_tier(path), "satellite": r["satellite"], "source": r["source"],
                "rows": r["rows"], "exact_dups": r["rows"] - r["n_unique_key"],
                # 시작은 첫 감지 날짜 00:00 (요청 기간 시작), 끝은 마지막 감지 시각
                "t_start": t_min.replace(hour=0, minute=0, second=0, microsecond=0), "t_end": r["t_max"],
                "lat_min": lat0, "lat_max": lat1, "lon_min": lon0, "lon_max": lon1,
            })
    cov = pl.DataFrame(rows).sort("satellite", "tier", "t_start", "file")
    FIRMS_COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cov.write_parquet(FIRMS_COVERAGE_PATH)
    return cov


def load_coverage() -> pl.DataFrame:
    if not FIRMS_COVERAGE_PATH.exists():
        return build_coverage()
    cov = pl.read_parquet(FIRMS_COVERAGE_PATH)
    if set(cov["file"].unique().to_list()) != {p.name for p in merged_files()}:
        return build_coverage()  # 파일이 추가·삭제되었으면 다시 계산
    return cov


@dataclass
class DedupePlan:
    """파일 하나에 적용할 중복 제거 규칙."""
    file: str
    tier: int
    drop_expr: pl.Expr | None       # 더 높은 순위 파일이 덮는 감지
    covered_by: list[str]           # 그 상위 파일 이름
    same_tier_prior: list[str]      # 시간이 겹치는 같은 순위 앞선 파일 (키로 anti-join)
    has_exact_dups: bool


def _cover_expr(r: dict) -> pl.Expr:
    return (
        (pl.col("satellite") == r["satellite"])
        & pl.col("acq_datetime_utc").is_between(r["t_start"], r["t_end"], closed="both")
        & pl.col("latitude").is_between(r["lat_min"], r["lat_max"])
        & pl.col("longitude").is_between(r["lon_min"], r["lon_max"])
    )


def _overlaps(a: dict, b: dict) -> bool:
    return (
        a["satellite"] == b["satellite"]
        and a["t_start"] <= b["t_end"] and b["t_start"] <= a["t_end"]
        and a["lat_min"] < b["lat_max"] and b["lat_min"] < a["lat_max"]
        and a["lon_min"] < b["lon_max"] and b["lon_min"] < a["lon_max"]
    )


def dedupe_plans(cov: pl.DataFrame) -> dict[str, DedupePlan]:
    recs = cov.to_dicts()
    plans: dict[str, DedupePlan] = {}
    for f in cov["file"].unique(maintain_order=True).to_list():
        mine = [r for r in recs if r["file"] == f]
        tier = mine[0]["tier"]
        higher = [h for h in recs if h["tier"] < tier and any(_overlaps(h, r) for r in mine)]
        same = [h for h in recs if h["tier"] == tier and h["file"] < f and any(_overlaps(h, r) for r in mine)]
        expr = None
        for h in higher:
            e = _cover_expr(h)
            expr = e if expr is None else (expr | e)
        plans[f] = DedupePlan(
            file=f, tier=tier, drop_expr=expr,
            covered_by=sorted({h["file"] for h in higher}),
            same_tier_prior=sorted({h["file"] for h in same}),
            has_exact_dups=any(r["exact_dups"] > 0 for r in mine),
        )
    return plans


def _scan_one(path: Path, plan: DedupePlan | None) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)
    if plan is None:
        return lf
    if plan.drop_expr is not None:
        lf = lf.filter(~plan.drop_expr)
    if plan.has_exact_dups:
        lf = lf.unique(subset=DEDUPE_KEY, keep="first", maintain_order=True)
    for prior in plan.same_tier_prior:
        lf = lf.join(pl.scan_parquet(FIRMS_MERGED_DIR / prior).select(DEDUPE_KEY), on=DEDUPE_KEY, how="anti")
    return lf


def load_hotspots(dedupe: bool = True, files: list[Path] | None = None) -> pl.LazyFrame:
    """통합 핫스팟 전체를 LazyFrame으로 연다 (기본: 우선순위·중복 제거 적용).

    전 세계 약 5억 행이므로 바로 collect하지 말고 filter를 먼저 걸 것.
    예) load_hotspots().filter(pl.col("latitude").is_between(33, 39)).collect()
    """
    paths = files if files is not None else merged_files()
    plans = dedupe_plans(load_coverage()) if dedupe else {}
    return pl.concat([_scan_one(p, plans.get(p.name)) for p in paths], how="vertical")


def scan_hotspot_file(path: Path, dedupe: bool = True) -> pl.LazyFrame:
    """파일 하나만 (중복 제거 규칙 적용) — 파일 단위 집계용."""
    return _scan_one(path, dedupe_plans(load_coverage()).get(path.name) if dedupe else None)


# ── 리포트 ────────────────────────────────────────────────────────────────

def dedupe_report(cov: pl.DataFrame) -> pl.DataFrame:
    """파일 × 위성별로 우선순위·같은 키 중복 제거에서 빠진 행 수."""
    plans = dedupe_plans(cov)
    out = []
    for f, plan in tqdm(plans.items(), desc="중복 제거 집계", unit="file", leave=False):
        if plan.drop_expr is None and not plan.has_exact_dups and not plan.same_tier_prior:
            sub = cov.filter(pl.col("file") == f)
            for r in sub.iter_rows(named=True):
                out.append({"file": f, "tier": plan.tier, "satellite": r["satellite"], "source": r["source"],
                            "rows_in": r["rows"], "dropped_priority": 0, "dropped_exact_dup": 0,
                            "rows_out": r["rows"], "covered_by": ""})
            continue
        lf = pl.scan_parquet(FIRMS_MERGED_DIR / f)
        covered = plan.drop_expr if plan.drop_expr is not None else pl.lit(False)
        before = lf.group_by("satellite", "source").agg(rows_in=pl.len(), dropped_priority=covered.sum())
        after = _scan_one(FIRMS_MERGED_DIR / f, plan).group_by("satellite", "source").agg(rows_out=pl.len())
        stats = before.join(after, on=["satellite", "source"], how="left").collect()
        for r in stats.iter_rows(named=True):
            rows_out = r["rows_out"] or 0
            out.append({"file": f, "tier": plan.tier, "satellite": r["satellite"], "source": r["source"],
                        "rows_in": r["rows_in"], "dropped_priority": r["dropped_priority"],
                        "dropped_exact_dup": r["rows_in"] - r["dropped_priority"] - rows_out,
                        "rows_out": rows_out,
                        "covered_by": " ".join(plan.covered_by + [f"same:{p}" for p in plan.same_tier_prior])})
    rep = pl.DataFrame(out).with_columns(satellite=sat_display()).sort("satellite", "tier", "file")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rep.write_csv(REPORTS_DIR / "firms_dedupe_summary.csv")
    return rep


def merge_summary() -> pl.DataFrame:
    """위성 × source × 연도별 행 수·기간 (중복 제거 전 원본 기준 + 제거 후 행 수)."""
    plans = dedupe_plans(load_coverage())
    parts = []
    for path in tqdm(merged_files(), desc="요약", unit="file", leave=False):
        agg = dict(
            rows=pl.len(),
            first_utc=pl.col("acq_datetime_utc").min(),
            last_utc=pl.col("acq_datetime_utc").max(),
            null_datetime=pl.col("acq_datetime_utc").is_null().sum(),
            null_latlon=(pl.col("latitude").is_null() | pl.col("longitude").is_null()).sum(),
            conf_low=(pl.col("confidence") == "l").sum(),
            conf_nominal=(pl.col("confidence") == "n").sum(),
            conf_high=(pl.col("confidence") == "h").sum(),
            conf_null=pl.col("confidence").is_null().sum(),
            type0_vegetation=(pl.col("type") == 0).sum(),
            type_null=pl.col("type").is_null().sum(),
            versions=pl.col("version").unique().sort().str.join(" "),
        )
        keys = ["instrument", "satellite", "source", "year"]
        yr = pl.col("acq_datetime_utc").dt.year().cast(pl.Int16).alias("year")
        raw = pl.scan_parquet(path).with_columns(yr).group_by(keys).agg(**agg)
        kept = _scan_one(path, plans.get(path.name)).with_columns(yr).group_by(keys).agg(rows_after_dedupe=pl.len())
        parts.append(raw.join(kept, on=keys, how="left").with_columns(
            file=pl.lit(path.name), tier=pl.lit(file_tier(path), dtype=pl.Int8)).collect())
    s = pl.concat(parts, how="vertical_relaxed").with_columns(pl.col("rows_after_dedupe").fill_null(0),
                                                               satellite=sat_display())
    s = s.select("satellite", "source", "year", "file", "tier", "rows", "rows_after_dedupe", "first_utc", "last_utc",
                 *[c for c in s.columns if c not in {"satellite", "source", "year", "file", "tier", "rows",
                                                      "rows_after_dedupe", "first_utc", "last_utc"}])
    s = s.sort(pl.col("satellite").replace_strict({k: i for i, k in enumerate(SATELLITES)}, default=99),
               "year", "tier", "file")
    s.write_csv(REPORTS_DIR / "firms_merge_summary.csv")
    return s


def satellite_overview(summary: pl.DataFrame) -> pl.DataFrame:
    """위성 × source별 전체 기간·행 수 (중복 제거 후)."""
    return (
        summary.group_by("satellite", "source")
        .agg(files=pl.col("file").n_unique(), rows=pl.col("rows").sum(),
             rows_after_dedupe=pl.col("rows_after_dedupe").sum(),
             first_utc=pl.col("first_utc").min(), last_utc=pl.col("last_utc").max())
        .sort(pl.col("satellite").replace_strict({k: i for i, k in enumerate(SATELLITES)}, default=99), "source")
    )


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_years(spec: list[str] | None) -> set[int] | None:
    if not spec:
        return None
    years: set[int] = set()
    for token in spec:
        for part in token.split(","):
            if "-" in part:
                a, b = part.split("-")
                years.update(range(int(a), int(b) + 1))
            elif part:
                years.add(int(part))
    return years


def print_table(df: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, fmt_str_lengths=60,
                   tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True, thousands_separator=","):
        print(df)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="FIRMS MODIS/VIIRS 핫스팟을 공통 스키마 Parquet로 통합")
    ap.add_argument("--sensor", choices=["all", *SENSORS], default="all",
                    help="아카이브 센서 (지정하면 downloads·7일 파일은 건너뜀)")
    ap.add_argument("--years", nargs="*", help="아카이브 연도. 예: 2000  2012-2014 (지정하면 downloads·7일 파일은 건너뜀)")
    ap.add_argument("--downloads-only", action="store_true", help="downloads와 active_fire_7d만 처리")
    ap.add_argument("--force", action="store_true", help="이미 만든 Parquet도 다시 생성")
    ap.add_argument("--summary-only", action="store_true", help="변환은 건너뛰고 범위·요약·중복 제거 리포트만")
    args = ap.parse_args(argv)
    start_run_log("dataset")

    FIRMS_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    years = parse_years(args.years)
    do_archive = not args.downloads_only
    do_small = args.downloads_only or (args.sensor == "all" and years is None)

    if not args.summary_only:
        archives = find_archives(args.sensor, years) if do_archive else []
        downloads = find_downloads() if do_small else []
        nrt7d = find_nrt7d() if do_small else []
        if not (archives or downloads or nrt7d):
            sys.exit(f"입력 파일이 없습니다: {FIRMS_RAW_DIR}")
        print(f"아카이브 zip {len(archives)}개, 다운로드 CSV {len(downloads)}개, 7일 CSV {len(nrt7d)}개 -> {FIRMS_MERGED_DIR}")
        for path, member, m in downloads:
            merge_download(path, member, m, args.force)
        for path, product, received in nrt7d:
            merge_nrt7d(path, product, received, args.force)
        for zip_path, sensor_dir, year in archives:
            out_path = FIRMS_MERGED_DIR / f"{sensor_dir}_{year}.parquet"
            if out_path.exists() and not args.force:
                print(f"[있음] {out_path.name} (다시 만들려면 --force)")
                continue
            t0 = time.perf_counter()
            rows = merge_archive_zip(zip_path, sensor_dir, year, out_path)
            print(f"[완료] {out_path.name}: {rows:,}행, {out_path.stat().st_size / 1e6:,.0f} MB, "
                  f"{time.perf_counter() - t0:,.0f}초")

    if not merged_files():
        sys.exit("요약할 Parquet이 없습니다.")
    t0 = time.perf_counter()
    cov = build_coverage()
    print(f"\n[범위] {FIRMS_COVERAGE_PATH.name}: 파일×위성 {cov.height}개")
    rep = dedupe_report(cov)
    print("\n[중복 제거] 빠진 행이 있는 파일 (전체는 reports/firms_dedupe_summary.csv)")
    print_table(rep.filter((pl.col("dropped_priority") > 0) | (pl.col("dropped_exact_dup") > 0)))
    print(f"합계: 우선순위로 {rep['dropped_priority'].sum():,}행, 같은 키 중복 {rep['dropped_exact_dup'].sum():,}행 제거")
    summary = merge_summary()
    print("\n[위성 × source 요약] (연도별 상세는 reports/firms_merge_summary.csv)")
    print_table(satellite_overview(summary))
    print(f"리포트 저장: firms_merge_summary.csv, firms_dedupe_summary.csv ({time.perf_counter() - t0:,.0f}초)")


if __name__ == "__main__":
    main()
