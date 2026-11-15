"""2·3개월 뒤 발생 확률 XGBoost (비교 실험). 기존 1개월 모델·산출물은 건드리지 않고 data/interim/timesfm_compare/xgb_h{h}/에 쓴다.
같은 특징·같은 확장 창·같은 보정 규칙, 직전 달 특징(lag·FWI)만 h개월 앞당김.
실행  uv run python scripts/horizon_xgb.py 2 3
"""
import sys
import time

from planetarium import grid_occurrence as go
from planetarium import occurrence as oc
from planetarium.config import INTERIM_DATA_DIR
from planetarium.runlog import peak_memory_gb, start_run_log

BASE = INTERIM_DATA_DIR / "timesfm_compare"
start_run_log("horizon_xgb")
for h in map(int, sys.argv[1:]):
    for name, mod in (("sgg", oc), ("grid", go)):
        t0 = time.perf_counter()
        w = BASE / f"xgb_h{h}" / name
        w.mkdir(parents=True, exist_ok=True)
        mod.HORIZON, mod.WORK, mod.PANEL, mod.OOF = h, w, w / "panel.parquet", w / "oof.parquet"
        mod.panel()
        mod.validate()
        print(f"[h={h} {name}] {time.perf_counter() - t0:,.0f}s, peak {peak_memory_gb() or 0:.2f} GB", flush=True)
