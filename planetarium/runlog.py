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


def peak_memory_gb() -> float | None:
    """이 프로세스의 최대 메모리 사용량(GB). Windows: Peak Working Set, 그 외: ru_maxrss."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE  # 64비트에서 핸들이 잘리지 않도록
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 1024**3
    except Exception:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    except Exception:
        return None


def log_peak_memory(label: str = "") -> None:
    gb = peak_memory_gb()
    if gb is not None:
        print(f"[메모리] 최대 사용량 {gb:.2f} GB {label}".rstrip())


def start_run_log(name: str) -> None:
    """stdout·stderr를 reports/<name>_run.log(추가 모드, UTF-8)에도 기록."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    f = open(REPORTS_DIR / f"{name}_run.log", "a", encoding="utf-8")  # noqa: SIM115 (프로세스 끝까지 유지)
    f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv)} =====\n")
    sys.stdout = _Tee(sys.__stdout__, f)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.__stderr__, f)  # type: ignore[assignment]
    import atexit
    atexit.register(log_peak_memory, "(프로세스 종료 시)")
