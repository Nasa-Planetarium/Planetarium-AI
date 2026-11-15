"""GitHub Release 데이터 설치·검증 (data-v1.2-2026-11-15).

release_manifest.csv(파일별 행 수·크기·sha256)를 먼저 받아, 목록의 파일을 하나씩 받고 sha256·크기로 검증한다.
  - 이미 있고 sha256이 맞는 파일은 건너뛴다 (다시 실행하면 이어 받기).
  - --local DIR: 미리 받아 둔 파일(평평한 이름 또는 manifest와 같은 하위 경로)을 받는 대신 복사·검증.
  - tar(harmonized_hotspots_2026.tar, harmonized_hotspots_static_flags.tar)는 검증 뒤 같은 폴더에 푼다
    (안전 검사: 일반 파일만, 절대 경로·".." 금지, 허용된 최상위 폴더만).
  - --hotspot-years 2024 2025: 2000~2025년 점 파일은 v1.1 hotspots Release에서 받는다 (GitHub가 주는 sha256으로 검증).
설계 참고: 팀원 Jonghun Shin(ozingmw)의 firms-1km-backend `release_assets.py` (manifest 검증·--local·안전한 tar 풀기).

실행  uv run python -m planetarium.release_assets install [--dest data/processed] [--local DIR] [--only "*.parquet"]
                                                     [--hotspot-years 2025] [--no-extract]
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

from planetarium.config import PROCESSED_DATA_DIR

REPO = "Nasa-Planetarium/Planetarium-AI"
TAG = "data-v1.2-2026-11-15"
HOTSPOT_TAG = "data-v1.1-hotspots-2026-11-14"
MANIFEST = "release_manifest.csv"
TAR_ROOTS = {"harmonized_hotspots", "harmonized_hotspots_static_flags"}
UA = {"User-Agent": "planetarium-release-assets/1.0"}


def url_of(tag: str, name: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{tag}/{name}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    tmp.replace(target)


def read_manifest(dest: Path, local: Path | None) -> list[dict]:
    if local is not None and (local / MANIFEST).exists():
        text = (local / MANIFEST).read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(urllib.request.Request(url_of(TAG, MANIFEST), headers=UA), timeout=60) as r:
            text = r.read().decode("utf-8")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / MANIFEST).write_text(text, encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    for r in rows:
        p = PurePosixPath(r["file"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"manifest에 잘못된 경로: {r['file']}")
        r["bytes"] = int(r["bytes"])
    return rows


def safe_extract(tar_path: Path, dest: Path) -> int:
    n = 0
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            rel = PurePosixPath(m.name)
            if rel.is_absolute() or ".." in rel.parts or rel.parts[0] not in TAR_ROOTS:
                raise ValueError(f"{tar_path.name}: 허용되지 않은 항목 {m.name}")
            if m.isdir():
                continue
            if not m.isfile():
                raise ValueError(f"{tar_path.name}: 일반 파일이 아닌 항목 {m.name}")
            tf.extract(m, dest, filter="data")
            n += 1
    return n


def install(dest: Path, local: Path | None = None, only: list[str] | None = None, extract: bool = True) -> dict:
    dest = Path(dest)
    rows = read_manifest(dest, local)
    if only:
        rows = [r for r in rows if any(fnmatch.fnmatch(r["file"], pat) or fnmatch.fnmatch(PurePosixPath(r["file"]).name, pat) for pat in only)]
    stats = {"verified": 0, "skipped": 0, "downloaded": 0, "copied": 0, "extracted_files": 0, "failed": []}
    for r in rows:
        rel, name = r["file"], PurePosixPath(r["file"]).name
        target = dest / rel
        if target.exists() and target.stat().st_size == r["bytes"] and sha256(target) == r["sha256"]:
            stats["skipped"] += 1
            print(f"[있음·검증됨] {rel}")
        else:
            if local is not None:
                src = next((p for p in (local / rel, local / name) if p.exists()), None)
                if src is None:
                    stats["failed"].append(f"{rel}: --local 폴더에 없음")
                    print(f"[없음] {rel} (--local)", file=sys.stderr)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
                stats["copied"] += 1
            else:
                print(f"[받기] {rel} ({r['bytes'] / 1e6:,.1f} MB)", flush=True)
                fetch(url_of(TAG, name), target)
                stats["downloaded"] += 1
            if target.stat().st_size != r["bytes"] or sha256(target) != r["sha256"]:
                stats["failed"].append(f"{rel}: 크기 또는 sha256 불일치")
                target.unlink(missing_ok=True)
                print(f"[불일치] {rel} — 지웠습니다", file=sys.stderr)
                continue
            print(f"[검증됨] {rel}")
        stats["verified"] += 1
        if extract and name.endswith(".tar"):
            n = safe_extract(target, dest)
            stats["extracted_files"] += n
            print(f"[풀기] {name}: {n:,}개 파일")
    return stats


def install_hotspot_years(dest: Path, years: list[int], local: Path | None = None) -> dict:
    """v1.1 hotspots Release의 연도별 tar (2000~2025). sha256은 GitHub API가 주는 값으로 검증."""
    with urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/tags/{HOTSPOT_TAG}",
                                                       headers={**UA, "Accept": "application/vnd.github+json"}), timeout=60) as r:
        assets = {a["name"]: a for a in json.loads(r.read())["assets"]}
    stats = {"verified": 0, "failed": []}
    with tempfile.TemporaryDirectory(dir=dest) as td:
        for y in years:
            name = f"harmonized_hotspots_{y}.tar"
            a = assets.get(name)
            if a is None:
                stats["failed"].append(f"{name}: Release에 없음")
                continue
            want = (a.get("digest") or "").removeprefix("sha256:")
            path = (local / name) if local is not None and (local / name).exists() else Path(td) / name
            if path.parent == Path(td):
                print(f"[받기] {name} ({a['size'] / 1e6:,.0f} MB)", flush=True)
                fetch(a["browser_download_url"], path)
            if path.stat().st_size != a["size"] or (want and sha256(path) != want):
                stats["failed"].append(f"{name}: 크기 또는 sha256 불일치")
                continue
            print(f"[검증됨] {name}{'' if want else ' (GitHub digest 없음: 크기만 확인)'}; 풀기 {safe_extract(path, dest):,}개 파일")
            stats["verified"] += 1
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="data-v1.2 Release 받기·sha256 검증·tar 풀기")
    ap.add_argument("action", choices=["install", "verify"])
    ap.add_argument("--dest", type=Path, default=PROCESSED_DATA_DIR)
    ap.add_argument("--local", type=Path, help="미리 받아 둔 파일이 있는 폴더 (받지 않고 복사·검증)")
    ap.add_argument("--only", nargs="+", help='이 패턴의 파일만 (예: "*.parquet" "models/*")')
    ap.add_argument("--hotspot-years", nargs="+", type=int, default=[], help="2000~2025년 점 tar (v1.1 hotspots Release)")
    ap.add_argument("--no-extract", action="store_true")
    a = ap.parse_args(argv)
    if a.action == "verify":  # 받기 없이 이미 있는 파일만 검사
        rows = list(csv.DictReader((a.dest / MANIFEST).open(encoding="utf-8")))
        missing = [r["file"] for r in rows if not (a.dest / r["file"]).exists()]
        bad = [r["file"] for r in rows if (a.dest / r["file"]).exists() and sha256(a.dest / r["file"]) != r["sha256"]]
        print(f"[검사] 일치 {len(rows) - len(bad) - len(missing)}/{len(rows)}, 불일치 {bad}, 없음 {len(missing)}개")
        sys.exit(1 if bad else 0)
    st = install(a.dest, a.local, a.only, not a.no_extract)
    if a.hotspot_years:
        st["hotspot_years"] = install_hotspot_years(a.dest, a.hotspot_years, a.local)
    print(f"[완료] {json.dumps(st, ensure_ascii=False)}")
    sys.exit(1 if st["failed"] or st.get("hotspot_years", {}).get("failed") else 0)


if __name__ == "__main__":
    main()
