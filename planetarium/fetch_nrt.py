"""FIRMS 실시간(NRT) 받기 → 통합 핫스팟 형식 변환 (harmonize_file 사용).

MAP_KEY
  환경변수 FIRMS_MAP_KEY 또는 저장소 루트 .env의 FIRMS_MAP_KEY=... 에서만 읽는다.
  키는 화면·로그·파일 이름에 절대 남기지 않는다. URL을 출력할 때는 키 자리를 *** 로 가린다.

기능
  --status                 키 상태(mapkey_status): 허용 호출 수, 현재 사용량, 기간
  --fetch                  area API로 최근 1~5일 받기 → data/raw/firms/api/<받은 날짜>/<SOURCE>_<AREA>_<N>d.csv
                           → harmonize_file로 변환 → data/interim/converted/api/<받은 날짜>/<SOURCE>/
  --append                 --fetch와 함께: 받은 기간 전체를 기존 점과 합쳐 작업 파일(data/processed)을 갱신
                           (중복 키 satellite·lat·lon·acq_datetime_utc, 우선순위 SP > Archive NRT > 7일 > API).
                           release_* 폴더는 건드리지 않는다. 날짜별 추가 점 수와 그 달 KR·ADM0_ARG·WORLD
                           value / value_projected / as_of_date 전후를 reports/fetch_nrt_append_<받은 날짜>.csv 로 남긴다.
  --with-spread            --append 뒤에 data/processed/spread_nowcast_latest.parquet를 다시 만든다
                           (NOAA-20 status=ok인 마지막 날 기준 내일 확률; 받는 중인 날은 기준일로 쓰지 않음).
  --with-planner           --append 뒤에 FireOutlook Planner 일 단위(1~7일) 예측을 다시 만든다
                           (기준일 = cell_day 최신 날짜 − 1일; 일 배열 → 표본 → 학습·예측 → 서비스 파일 publish, 약 9분·최대 약 17 GB).
                           월 단위(1~12개월)는 달마다 planner_monthly final --origin YYYY-MM --as-of YYYY-MM-DD 로 따로 갱신.

FIRMS area API: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{AREA}/{DAY_RANGE}[/{DATE}]
  SOURCE: VIIRS_NOAA20_NRT, VIIRS_NOAA21_NRT, MODIS_NRT (VIIRS_SNPP_NRT는 S-NPP 종료로 기본 제외)
  AREA: world 또는 서,남,동,북 / DAY_RANGE: 1~5 / DATE: YYYY-MM-DD (그 날부터 DAY_RANGE일)

실행
  uv run python -m planetarium.fetch_nrt --status
  uv run python -m planetarium.fetch_nrt --fetch --days 2 [--area world] [--sources VIIRS_NOAA20_NRT MODIS_NRT]
  uv run python -m planetarium.fetch_nrt --fetch --days 2 --append
  uv run python -m planetarium.fetch_nrt --fetch --days 2 --append --with-spread
  uv run python -m planetarium.fetch_nrt --fetch --days 2 --append --with-spread --with-planner
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from planetarium.config import FIRMS_RAW_DIR, INTERIM_DATA_DIR, PROJ_ROOT
from planetarium.runlog import start_run_log

API_ROOT = "https://firms.modaps.eosdis.nasa.gov"
DEFAULT_SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]
API_DIR = FIRMS_RAW_DIR / "api"
CONVERTED_DIR = INTERIM_DATA_DIR / "converted" / "api"
_KEY_CACHE: str | None = None


# ── 키 (출력 금지) ────────────────────────────────────────────────────────

def map_key() -> str:
    """FIRMS_MAP_KEY: 환경변수 → .env. 값은 반환만 하고 어디에도 출력하지 않는다."""
    global _KEY_CACHE
    if _KEY_CACHE:
        return _KEY_CACHE
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    env = PROJ_ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*FIRMS_MAP_KEY\s*=\s*['\"]?([^'\"\s#]+)", line)
            if m:
                key = m.group(1)
    if not key:
        raise SystemExit("FIRMS_MAP_KEY가 없습니다. 저장소 루트 .env에 FIRMS_MAP_KEY=<키> 한 줄을 직접 넣어 주세요.")
    _KEY_CACHE = key
    return key


def mask(text: str) -> str:
    """문자열 안의 MAP_KEY를 *** 로 가린다 (URL·오류 메시지 출력용)."""
    k = _KEY_CACHE or os.environ.get("FIRMS_MAP_KEY", "")
    return text.replace(k, "***") if k else text


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "planetarium-fetch-nrt/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} — {mask(url)}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"연결 실패 ({e.reason}) — {mask(url)}") from None


# ── API ──────────────────────────────────────────────────────────────────

def mapkey_status() -> dict:
    """키 상태: transaction_limit(기간당 허용 호출), current_transactions(사용), transaction_interval(기간)."""
    url = f"{API_ROOT}/mapserver/mapkey_status/?MAP_KEY={map_key()}"
    print(f"[요청] {mask(url)}")
    raw = _get(url, timeout=60).decode("utf-8", "replace")
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"mapkey_status 응답을 읽을 수 없습니다: {mask(raw[:200])}") from None
    if isinstance(d, dict):
        d.pop("map_key", None)  # 응답에 키가 들어 있으면 지운다
        d.pop("MAP_KEY", None)
    return d


def fetch_area(source: str, area: str = "world", days: int = 1, on: date | None = None,
               received: date | None = None) -> Path:
    """area API 한 번 → raw/firms/api/<받은 날짜>/<SOURCE>_<AREA>_<N>d[_DATE].csv 저장 (키는 파일 이름·내용에 없음)."""
    if not 1 <= days <= 5:
        raise ValueError("days는 1~5")
    url = f"{API_ROOT}/api/area/csv/{map_key()}/{source}/{area}/{days}" + (f"/{on.isoformat()}" if on else "")
    print(f"[요청] {mask(url)}")
    body = _get(url)
    text_head = body[:300].decode("utf-8", "replace")
    if not text_head.lower().startswith("latitude"):
        raise RuntimeError(f"FIRMS 응답이 CSV가 아닙니다 ({source}): {mask(text_head.strip())[:200]}")
    d = API_DIR / (received or date.today()).isoformat()
    d.mkdir(parents=True, exist_ok=True)
    area_tag = "world" if area == "world" else area.replace(",", "_").replace(".", "p").replace("-", "m")
    stamp = datetime.now(timezone.utc).strftime("t%H%M")  # 같은 날 여러 번 받아도 원본(raw)을 덮어쓰지 않음
    f = d / (f"{source}_{area_tag}_{days}d" + (f"_{on.isoformat()}" if on else "") + f"_{stamp}.csv")
    if f.exists():
        raise RuntimeError(f"이미 있는 원본 파일입니다(덮어쓰지 않음): {f.name}")
    f.write_bytes(body)
    n = body.count(b"\n") - 1
    print(f"[저장] {f.relative_to(PROJ_ROOT)} ({n:,}행, {len(body) / 1e6:.1f} MB)")
    return f


def fetch_and_convert(sources: list[str], area: str, days: int, fmt: str = "parquet") -> list[dict]:
    from planetarium.converter import harmonize_file
    from planetarium.harmonize import aoi_membership
    from planetarium.hotspots import Context
    received = date.today()
    ctx, mem = Context(), aoi_membership()[0]
    out = []
    for s in sources:
        f = fetch_area(s, area, days, received=received)
        summ = harmonize_file(f, CONVERTED_DIR / received.isoformat() / s, fmt, ctx, mem)
        print(f"[변환] {s}: 출력 {summ['rows_out']:,}행 (기준 {summ['reference_rows']:,}, 대기 {summ['pending_new_month_rows']:,}), "
              f"달 {summ['months']}, {summ['seconds']}초")
        out.append(summ)
        time.sleep(1)
    return out


# ── --append: 작업 파일 갱신 ──────────────────────────────────────────────

FOCUS_AOI = ["KR", "ADM0_ARG", "WORLD"]


def _overlap_files(first: date, last: date) -> list[Path]:
    """받은 기간(first~last, UTC)과 시간 범위가 겹치는 통합 파일만 (메모리 절약)."""
    import polars as pl

    from planetarium.dataset import FIRMS_MERGED_DIR, load_coverage
    cov = load_coverage().filter(pl.col("t_start").dt.date() <= last, pl.col("t_end").dt.date() >= first)
    return [FIRMS_MERGED_DIR / f for f in cov["file"].unique().sort().to_list()]


def _points_by_date(first: date, last: date):
    """중복 제거를 적용한 감지 점 수: 위성 × UTC 날짜 (first~last). 겹치는 파일만 읽는다."""
    import polars as pl

    from planetarium.dataset import load_hotspots, sat_display
    files = _overlap_files(first, last)
    print(f"[점 수] 겹치는 파일 {len(files)}개만 읽음: {', '.join(f.stem for f in files)}")
    return (load_hotspots(files=files).filter(pl.col("acq_date").is_between(first, last))
            .group_by(satellite=sat_display(), date_utc=pl.col("acq_date")).agg(n=pl.len()).collect())


def _reference_points(months: list[tuple[int, int]]):
    """통합 핫스팟 기준 점 수: 위성 × 현지 날짜 (그 달 파티션만)."""
    import polars as pl

    from planetarium.hotspots import HOTSPOTS_DIR
    parts = [HOTSPOTS_DIR / "reference" / f"year={y}" / f"month={m:02d}" / "part-0.parquet" for y, m in months]
    parts = [f for f in parts if f.exists()]
    if not parts:
        return pl.DataFrame(schema={"satellite": pl.String, "date_local": pl.Date, "n_ref": pl.UInt32, "w_ref": pl.Float64})
    return (pl.scan_parquet(parts).group_by("satellite", "date_local")
            .agg(n_ref=pl.len(), w_ref=pl.col("harmonized_weight").sum()).collect())


def _aoi_snapshot(months: list[tuple[int, int]]):
    import polars as pl

    from planetarium.harmonize import AOI_MONTH_PATH
    am = pl.read_parquet(AOI_MONTH_PATH)
    cols = [c for c in ("aoi_id", "year", "month", "value", "value_projected", "as_of_date", "status") if c in am.columns]
    keep = pl.any_horizontal(*[(pl.col("year") == y) & (pl.col("month") == m) for y, m in months])
    return am.filter(pl.col("aoi_id").is_in(FOCUS_AOI), keep).select(cols)


def _resort_outputs() -> None:
    """정렬이 풀린 큰 산출물을 칸 순으로 다시 써서 용량을 줄인다 (행 수·합계 같은지 확인 후 교체)."""
    from planetarium.harmonize import HCM_PATH
    from planetarium.hotspots import CELL_DAY_PATH
    from planetarium.release import resort
    for path, sort_cols, sum_cols, bins in (
            (HCM_PATH, ["cell_id", "year", "month"], ["value", "value_projected", "raw_modis"], 24),
            (CELL_DAY_PATH, ["cell_id", "date_local"], ["value", "residual_month", "value_projected_month", "n_points"], 64)):
        r = resort(path, sort_cols, sum_cols, bins)
        print(f"[정렬] {r['file']}: {r['rows']:,}행, {r['mb_before']:,} MB → {r['mb_after']:,} MB (최대 메모리 {r['peak_mem_gb']} GB)")


def append(sources: list[str], area: str, days: int) -> None:
    """받기 → 변환(API 순위 3) → 칸×일 → 상태·칸×월 → 그 연도 칸 표 교체 → 그 달 핫스팟 → partial 반영 → AOI·비교 → 검증."""
    import polars as pl

    from planetarium import analysis, dataset, features, harmonize, hotspots, monthly, static_exclusion, verify_hotspots
    from planetarium.config import REPORTS_DIR
    from planetarium.runlog import peak_memory_gb
    received = date.today()
    t0 = time.perf_counter()
    files = []
    for s in sources:
        files.append(fetch_area(s, area, days, received=received))
        time.sleep(1)
    # 받은 기간 (UTC) — 파일의 acq_date 범위
    rng = pl.concat([pl.read_csv(f, columns=["acq_date"], try_parse_dates=True) for f in files]).select(
        lo=pl.col("acq_date").min(), hi=pl.col("acq_date").max()).row(0)
    first, last = rng
    if first is None:
        # 모든 응답이 헤더만 = 요청 지역·기간에 감지 0건. 관측 기록만 남기고 갱신 단계는 건너뛴다.
        for f in files:
            dataset.log_api_response(f, dataset.API_NAME.match(f.name)["source"], f.parent.name, 0)
        print(f"[감지 0건] 받은 {len(files)}개 응답이 모두 헤더만 있음 → {dataset.API_RESPONSES_PATH.name}에 기록, 갱신 없음")
        return
    print(f"[범위] 받은 감지 날짜(UTC) {first} ~ {last}")
    months = sorted({(d.year, d.month) for d in (first + timedelta(days=i) for i in range((last - first).days + 1))})
    before_pts = _points_by_date(first, last)
    before_aoi = _aoi_snapshot(months)
    before_ref = _reference_points(months)

    steps = [
        ("API 파일 변환·범위표", lambda: dataset.main(["--api-only"])),
        ("칸×일 집계 (API 파일)", lambda: features.main(["--files", "api_*"])),
        ("위성 상태·칸×월", lambda: monthly.main([])),
        ("칸×월 통합표 연도 교체", lambda: harmonize.main(["--splice-years", *map(str, sorted({y for y, _ in months}))])),
        ("통합 핫스팟 그 달 다시 만들기", lambda: hotspots.build_months(months)),
        ("진행 중인 달 month-to-date", lambda: hotspots.finalize_projection()),
        ("고정 열원 점 제외 (v1.2)", lambda: static_exclusion.run(sorted({y for y, _ in months}))),
        ("AOI 월 합계", lambda: harmonize.main(["--skip-cells", "--skip-boundaries"])),
        ("과거 대비 비교표", lambda: analysis.main([])),
        ("검증 (점 가중치 + residual = 칸 값)", lambda: verify_hotspots.main([])),
        ("칸×월·칸×일 정렬 재작성 (용량 줄이기)", _resort_outputs),
    ]
    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n===== [{i}/{len(steps)}] {label} =====", flush=True)
        s0 = time.perf_counter()
        fn()
        print(f"[{i}/{len(steps)}] {label} 끝 ({time.perf_counter() - s0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB)", flush=True)

    start_run_log("fetch_nrt")
    after_pts = _points_by_date(first, last)
    after_aoi = _aoi_snapshot(months)
    after_ref = _reference_points(months)
    ref = (before_ref.join(after_ref, on=["satellite", "date_local"], how="full", coalesce=True, suffix="_after")
           .filter(pl.col("date_local") >= first).sort("date_local", "satellite"))
    pts = (before_pts.rename({"n": "before"}).join(after_pts.rename({"n": "after"}), on=["satellite", "date_utc"],
                                                       how="full", coalesce=True)
           .with_columns(pl.col("before", "after").fill_null(0)).with_columns(added=pl.col("after") - pl.col("before"))
           .sort("date_utc", "satellite"))
    keys = ["aoi_id", "year", "month"]
    aoi = before_aoi.join(after_aoi, on=keys, how="full", coalesce=True, suffix="_after").sort(keys)
    tag = received.isoformat()
    pts.write_csv(REPORTS_DIR / f"fetch_nrt_append_{tag}_points.csv")
    aoi.write_csv(REPORTS_DIR / f"fetch_nrt_append_{tag}_aoi.csv")
    ref.write_csv(REPORTS_DIR / f"fetch_nrt_append_{tag}_reference.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True,
                   tbl_hide_dataframe_shape=True, float_precision=2, thousands_separator=",", tbl_width_chars=200):
        print("[날짜별 추가 점 수] (중복 제거 후, UTC)")
        print(pts)
        print("[그 달 AOI 전/후]")
        print(aoi)
        print("[기준 점 수·가중치 합 전/후] (현지 날짜)")
        print(ref)
    print(f"[완료] --append {time.perf_counter() - t0:,.0f}초, 최대 메모리 {peak_memory_gb() or 0:.2f} GB, "
          f"리포트 reports/fetch_nrt_append_{tag}_*.csv")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="FIRMS NRT 받기 → 통합 핫스팟 형식 (MAP_KEY는 .env)")
    ap.add_argument("--status", action="store_true", help="키 상태만 확인")
    ap.add_argument("--fetch", action="store_true", help="area API로 받아 변환")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--area", default="world", help="world 또는 서,남,동,북 (예: 124,33,132,39)")
    ap.add_argument("--days", type=int, default=1, help="최근 1~5일")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--append", action="store_true", help="--fetch와 함께: 기존 점과 합쳐 data/processed 작업 파일 갱신 (release_* 불변)")
    ap.add_argument("--with-spread", action="store_true",
                    help="--append 뒤에 확산 가능성(spread_nowcast_latest.parquet)을 최신 관측일 기준으로 다시 만든다")
    ap.add_argument("--with-planner", action="store_true",
                    help="--append 뒤에 FireOutlook Planner 일 단위(1~7일) 예측을 최신 완결일 기준으로 다시 만든다")
    args = ap.parse_args(argv)
    if args.with_spread and not (args.fetch and args.append):
        ap.error("--with-spread는 --fetch --append와 함께 써야 합니다")
    if args.with_planner and not (args.fetch and args.append):
        ap.error("--with-planner는 --fetch --append와 함께 써야 합니다")
    start_run_log("fetch_nrt")
    if args.status or not args.fetch:
        st = mapkey_status()
        print("[키 상태]", json.dumps(st, ensure_ascii=False))
    if args.fetch and args.append:
        append(args.sources, args.area, args.days)
        if args.with_spread:
            from planetarium import spread_ops   # v1.2-ops 모델 (운영과 같은 기상 지연 조건으로 검증, 기상 미사용)
            t0 = time.perf_counter()
            print("\n===== 확산 가능성 다시 만들기 (spread_nowcast_latest) =====", flush=True)
            spread_ops.latest()
            start_run_log("fetch_nrt")
            print(f"[확산 가능성] 끝 ({time.perf_counter() - t0:,.0f}초)")
        if args.with_planner:
            from planetarium import planner_publish
            t0 = time.perf_counter()
            print("\n===== FireOutlook Planner 일 단위 다시 만들기 (planner_daily_grid) =====", flush=True)
            planner_publish.refresh("auto")
            start_run_log("fetch_nrt")
            print(f"[Planner] 끝 ({time.perf_counter() - t0:,.0f}초)")
    elif args.fetch:
        fetch_and_convert(args.sources, args.area, args.days, args.format)


if __name__ == "__main__":
    main()
