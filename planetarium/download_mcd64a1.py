"""MODIS 불탄 면적 MCD64A1 v061 내려받기 (NASA Earthdata, earthaccess).

대상   short_name="MCD64A1", version="061", 2012-01-01 ~ 2025-12-31 (월 1개 × 168개월 / 타일)
타일   한국 h27v05, h28v05 / 아르헨티나 h11v11, h12v11, h13v11, h12v12, h13v12, h12v13, h13v13, h13v14, h14v14
저장   data/external/burned_area/mcd64a1/<타일>/<원래 파일 이름>
       이미 있는 파일은 건너뛴다.
로그인 earthaccess.login(strategy="interactive", persist=True)
       아이디·비밀번호는 실행한 사람이 터미널에 직접 입력한다. 코드·로그에 남기지 않는다
       (로그인은 실행 로그 기록을 시작하기 전에 한다). persist=True면 사용자 홈의 .netrc에 저장된다.
검사   파일마다 HDF4 시그니처(첫 4바이트 0x0E 0x03 0x13 0x01)와 크기, 타일별 월(A<연도><일>) 개수 168개.
결과   reports/mcd64a1_download_summary.csv

실행   uv run python -m planetarium.download_mcd64a1
       uv run python -m planetarium.download_mcd64a1 --check-only     # 내려받기 없이 검사만
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import polars as pl

from planetarium.config import EXTERNAL_DATA_DIR, REPORTS_DIR

OUT_DIR = EXTERNAL_DATA_DIR / "burned_area" / "mcd64a1"
TILES = {
    "korea": ["h27v05", "h28v05"],
    "argentina": ["h11v11", "h12v11", "h13v11", "h12v12", "h13v12", "h12v13", "h13v13", "h13v14", "h14v14"],
}
TEMPORAL = ("2012-01-01", "2025-12-31")
EXPECTED_MONTHS = (2025 - 2012 + 1) * 12  # 168
HDF4_MAGIC = b"\x0e\x03\x13\x01"
NAME = re.compile(r"^MCD64A1\.A(?P<year>\d{4})(?P<doy>\d{3})\.(?P<tile>h\d{2}v\d{2})\.061\.\d+\.hdf$")


def all_tiles() -> list[tuple[str, str]]:
    return [(region, t) for region, ts in TILES.items() for t in ts]


def download(threads: int) -> None:
    import earthaccess
    for region, tile in all_tiles():
        d = OUT_DIR / tile
        d.mkdir(parents=True, exist_ok=True)
        have = {p.name for p in d.glob("*.hdf")}
        t0 = time.perf_counter()
        res = earthaccess.search_data(short_name="MCD64A1", version="061", temporal=TEMPORAL,
                                      granule_name=f"MCD64A1.A*.{tile}.061.*")
        todo = []
        for g in res:
            links = [u for u in g.data_links() if u.endswith(".hdf")]
            if links and Path(links[0]).name not in have:
                todo.append(g)
        print(f"[{region} {tile}] 검색 {len(res)}개, 이미 있음 {len(have)}개, 받을 것 {len(todo)}개")
        if todo:
            earthaccess.download(todo, local_path=str(d), threads=threads)
        print(f"[{region} {tile}] 완료 {len(list(d.glob('*.hdf')))}개 ({time.perf_counter() - t0:.0f}초)")


def check() -> pl.DataFrame:
    rows = []
    for region, tile in all_tiles():
        files = sorted((OUT_DIR / tile).glob("*.hdf")) if (OUT_DIR / tile).exists() else []
        months, bad_name, bad_magic, small = set(), [], [], []
        for f in files:
            m = NAME.match(f.name)
            if not m or m["tile"] != tile:
                bad_name.append(f.name)
                continue
            months.add((int(m["year"]), int(m["doy"])))
            with open(f, "rb") as fh:
                if fh.read(4) != HDF4_MAGIC:
                    bad_magic.append(f.name)
            if f.stat().st_size < 10_000:
                small.append(f.name)
        in_range = {k for k in months if 2012 <= k[0] <= 2025}
        rows.append({
            "region": region, "tile": tile, "files": len(files), "months_2012_2025": len(in_range),
            "expected": EXPECTED_MONTHS, "ok": len(in_range) == EXPECTED_MONTHS and not bad_magic and not small and not bad_name,
            "bad_hdf_header": len(bad_magic), "too_small": len(small), "bad_name": len(bad_name),
            "duplicates_same_month": len(files) - len(bad_name) - len(months),
            "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
            "examples_bad": " ".join((bad_magic + small + bad_name)[:3]),
        })
    df = pl.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(REPORTS_DIR / "mcd64a1_download_summary.csv")
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="MCD64A1 v061 내려받기 (한국·아르헨티나 타일)")
    ap.add_argument("--check-only", action="store_true", help="내려받지 않고 개수·HDF 헤더만 검사")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    if not args.check_only:
        import earthaccess
        # 로그인은 실행 로그를 켜기 전에 한다 (입력값이 로그 파일에 남지 않도록)
        auth = earthaccess.login(strategy="interactive", persist=True)
        if not auth.authenticated:
            raise SystemExit("Earthdata 로그인 실패")
    from planetarium.runlog import start_run_log
    start_run_log("mcd64a1_download")
    if not args.check_only:
        download(args.threads)
    df = check()
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_formatting="ASCII_MARKDOWN",
                   tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True):
        print(df.drop("examples_bad"))
    print(f"검사 결과: {REPORTS_DIR / 'mcd64a1_download_summary.csv'}")


if __name__ == "__main__":
    main()
