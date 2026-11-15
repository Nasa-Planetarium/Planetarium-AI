"""릴리스 폴더의 파일별 행 수·크기·sha256 → release_manifest.csv
실행  uv run python scripts/release_manifest.py data/processed/release_v1.2-2026-11-15
"""
import hashlib
import os
import sys
import tarfile

import polars as pl
import pyarrow.parquet as pq

R = sys.argv[1]
rows = []
for dp, _, fn in os.walk(R):
    for f in sorted(fn):
        if f == "release_manifest.csv":
            continue
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, R).replace(os.sep, "/")
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        n = None
        if f.endswith(".parquet"):
            n = pq.ParquetFile(p).metadata.num_rows
        elif f.endswith(".csv"):
            with open(p, encoding="utf-8") as fh:
                n = sum(1 for _ in fh) - 1
        elif f.endswith(".tar"):
            n = 0
            with tarfile.open(p) as t:
                for m in t.getmembers():
                    if m.name.endswith(".parquet"):
                        n += pq.ParquetFile(t.extractfile(m)).metadata.num_rows
        rows.append({"file": rel, "rows": n, "bytes": os.path.getsize(p), "sha256": h.hexdigest()})
m = pl.DataFrame(rows).sort("file")
m.write_csv(os.path.join(R, "release_manifest.csv"))
with pl.Config(tbl_rows=-1, tbl_width_chars=250, fmt_str_lengths=70):
    print(m.with_columns(mb=(pl.col("bytes") / 1e6).round(1)).select("file", "rows", "mb", "sha256"))
print(f"total {m['bytes'].sum() / 1e9:.2f} GB")
