"""NASA POWER(MERRA-2) 일별 기상 받기 — 확산 예측(spread nowcast)용. 키 없음.

지점: 확산 예측 후보 칸(data/interim/spread_nowcast/samples_*.parquet)을 덮는 0.5° 칸의 중심 (한국 상자·아르헨티나).
기간: 2012-01-01 ~ 2025-12-31 (2012-02 첫 표본의 최근 7일 강수까지). 시간 기준: LST(지방 태양시) — 현지 날짜와 맞춤.
변수: WS10M(평균 풍속), WS10M_MAX(최대 풍속), WD10M(풍향, 불어오는 방향, 북=0 시계 방향), RH2M(일평균 상대습도),
      T2M_MAX(최고 기온), T2MDEW(이슬점), PRECTOTCORR(보정 강수, mm/day). community=AG.
상대습도 최저 근사: RH_min ≈ 100 × e_s(T2MDEW) / e_s(T2M_MAX), e_s(T) = 6.112·exp(17.62·T / (243.12 + T)) (Magnus, WMO 2008).
      하루 중 이슬점이 거의 일정하고 최저 습도가 최고 기온 때 나타난다는 가정. 결측값(-999)은 null.

캐시: data/external/weather/power/<region>/<lat>_<lon>.json (원본 응답 그대로) → 합본 data/external/weather/power_daily.parquet.
      이미 받은 지점은 건너뛴다(--resume 기본). 동시 요청 4개, 실패 시 지수 백오프로 재시도.

실행  uv run python -m planetarium.weather_power fetch     # 받기 (백그라운드 권장)
      uv run python -m planetarium.weather_power combine   # 합본 parquet

출처 표기: docs/references.md (NASA POWER 인정 문구).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tqdm import tqdm

from planetarium.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

GRID = 0.5
START, END = "20120101", "20251231"
PARAMS = ["WS10M", "WS10M_MAX", "WD10M", "RH2M", "T2M_MAX", "T2MDEW", "PRECTOTCORR"]
API = "https://power.larc.nasa.gov/api/temporal/daily/point"
CACHE_DIR = EXTERNAL_DATA_DIR / "weather" / "power"
DAILY_PATH = EXTERNAL_DATA_DIR / "weather" / "power_daily.parquet"
SAMPLES = INTERIM_DATA_DIR / "spread_nowcast"
WORKERS = 4


def grid_key(lat: pl.Expr, lon: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """0.5° 칸 중심 (칸 경계 0, 0.5, ...; 중심 0.25, 0.75, ...)."""
    return ((lat / GRID).floor() * GRID + GRID / 2).round(3), ((lon / GRID).floor() * GRID + GRID / 2).round(3)


def points() -> pl.DataFrame:
    glat, glon = grid_key(pl.col("cell_lat").cast(pl.Float64), pl.col("cell_lon").cast(pl.Float64))
    return (pl.scan_parquet(SAMPLES / "samples_*.parquet").select("region", glat.alias("glat"), glon.alias("glon"))
            .unique().collect().sort("region", "glat", "glon"))


def cache_path(region: str, lat: float, lon: float) -> Path:
    return CACHE_DIR / region / f"{lat:.3f}_{lon:.3f}.json"


def fetch_one(region: str, lat: float, lon: float, tries: int = 6) -> tuple[str, float]:
    path = cache_path(region, lat, lon)
    if path.exists():
        return "cached", 0.0
    url = (f"{API}?parameters={','.join(PARAMS)}&community=AG&longitude={lon}&latitude={lat}"
           f"&start={START}&end={END}&format=JSON&time-standard=LST")
    t0, wait = time.perf_counter(), 5.0
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "planetarium-spread-nowcast/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read()
            d = json.loads(body)
            if "properties" not in d:
                raise RuntimeError(f"응답에 자료 없음: {str(d)[:200]}")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(body)
            tmp.replace(path)
            return "ok", time.perf_counter() - t0
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError, ConnectionError) as e:
            if k == tries - 1:
                return f"fail: {e}", time.perf_counter() - t0
            time.sleep(wait)
            wait = min(wait * 2, 120)
    return "fail", time.perf_counter() - t0


def fetch() -> None:
    pts = points()
    todo = [r for r in pts.iter_rows() if not cache_path(*r).exists()]
    print(f"[지점] 전체 {pts.height:,} (한국 {pts.filter(pl.col('region') == 'KR').height}, 아르헨티나 "
          f"{pts.filter(pl.col('region') == 'ARG').height}) / 남은 {len(todo):,} — 요청 1회 = 한 지점 {START}~{END} 변수 {len(PARAMS)}개")
    t0 = time.perf_counter()
    n_ok = n_fail = 0
    secs = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fetch_one, *r): r for r in todo}
        bar = tqdm(as_completed(futs), total=len(futs), desc="NASA POWER", unit="point", dynamic_ncols=True)
        for f in bar:
            status, sec = f.result()
            if status == "ok":
                n_ok += 1
                secs.append(sec)
            elif status != "cached":
                n_fail += 1
                tqdm.write(f"[실패] {futs[f]}: {status}")
            bar.set_postfix(ok=n_ok, fail=n_fail)
    el = time.perf_counter() - t0
    print(f"[완료] 요청 {n_ok + n_fail:,}회 (성공 {n_ok:,}, 실패 {n_fail:,}), {el / 60:,.1f}분, "
          f"요청당 중앙값 {sorted(secs)[len(secs) // 2] if secs else 0:.1f}초. 실패분은 다시 실행하면 이어 받음.")


def rh_min_expr() -> pl.Expr:
    es = lambda t: 6.112 * ((17.62 * t) / (243.12 + t)).exp()  # noqa: E731
    return (100 * es(pl.col("T2MDEW")) / es(pl.col("T2M_MAX"))).clip(0, 100)


def combine() -> pl.DataFrame:
    rows = []
    files = sorted(CACHE_DIR.glob("*/*.json"))
    for f in tqdm(files, desc="합본", unit="point", dynamic_ncols=True):
        region = f.parent.name
        lat, lon = (float(x) for x in f.stem.split("_"))
        p = json.loads(f.read_text(encoding="utf-8"))["properties"]["parameter"]
        days = sorted(p[PARAMS[0]])
        d = {"date": days, **{k: [p[k].get(x) for x in days] for k in PARAMS}}
        rows.append(pl.DataFrame(d).with_columns(region=pl.lit(region), glat=pl.lit(lat), glon=pl.lit(lon)))
    df = pl.concat(rows).with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"),
                                      *[pl.when(pl.col(k) <= -998).then(None).otherwise(pl.col(k)).cast(pl.Float32).alias(k) for k in PARAMS])
    df = df.with_columns(RH2M_MIN_EST=rh_min_expr().cast(pl.Float32)).sort("region", "glat", "glon", "date")
    df.write_parquet(DAILY_PATH, compression="zstd")
    nulls = {k: int(df[k].null_count()) for k in PARAMS}
    print(f"[완료] {DAILY_PATH.name}: {df.height:,}행, 지점 {len(files):,}, 결측 {nulls}")
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="NASA POWER 일별 기상 (확산 예측용)")
    ap.add_argument("step", choices=["fetch", "combine"])
    ap.add_argument("--resume", action="store_true", help="기본 동작 (받은 지점은 건너뜀)")
    args = ap.parse_args(argv)
    start_run_log("weather_power")
    if args.step == "fetch":
        fetch()
    else:
        combine()
    print(f"[메모리] 최대 {peak_memory_gb() or 0:.2f} GB")


if __name__ == "__main__":
    main()
