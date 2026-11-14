"""실행 로그: 화면 출력을 reports/<name>_run.log에도 남긴다."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import TextIO

from planetarium.config import REPORTS_DIR


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            try:
                st.write(s)
            except UnicodeEncodeError:  # Windows 콘솔(cp949)에서 못 쓰는 문자는 ?로
                enc = getattr(st, "encoding", None) or "ascii"
                st.write(s.encode(enc, "replace").decode(enc))
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()

    def isatty(self) -> bool:
        return False


def start_run_log(name: str) -> None:
    """stdout·stderr를 reports/<name>_run.log(추가 모드, UTF-8)에도 기록."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    f = open(REPORTS_DIR / f"{name}_run.log", "a", encoding="utf-8")  # noqa: SIM115 (프로세스 끝까지 유지)
    f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv)} =====\n")
    sys.stdout = _Tee(sys.__stdout__, f)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.__stderr__, f)  # type: ignore[assignment]
