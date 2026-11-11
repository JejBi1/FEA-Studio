"""Memory of the program while a solver runs: what the process does not need at that moment goes back to the system.

Before a solver or helper process starts (ccx, PARDISO / hypre / NGSolve workers, Gmsh) the calling process releases its
working set (Windows EmptyWorkingSet). Clean pages (DLL code of OpenCascade, VTK, Qt, OpenBLAS, the graphics driver, fonts)
go to the standby list at once and are reloaded from the disk cache when the program touches them again. Modified data pages
are written to the pagefile only under memory pressure. Numerics are not touched: the same arrays and the same operations.

Off: FEA_RELEASE_MEMORY=0 (A/B tests).
"""
from __future__ import annotations

import gc
import os
import sys
import threading
import time


MIN_INTERVAL = 2.0          # [s] a garbage collection closer to the previous one is skipped (parallel ccx jobs start at once)
_last_gc = [-1e9]
_lock = threading.Lock()
_api = []                   # (kernel32, psapi) once


def _empty_working_set() -> bool:
    try:
        if not _api:
            import ctypes
            kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
            psapi.EmptyWorkingSet.restype = ctypes.c_int
            _api.append((kernel, psapi))
        kernel, psapi = _api[0]
        return bool(psapi.EmptyWorkingSet(kernel.GetCurrentProcess()))
    except (OSError, AttributeError):          # no psapi (very old system): nothing to release
        return False


def release_unused(force: bool = False) -> bool:
    """Runs the garbage collector and releases the working set of this process (Windows). Returns True when the working set
    was released. On other systems it only runs the garbage collector. The garbage collection (~0.1 s on a big heap) is
    skipped when the previous one ran less than MIN_INTERVAL seconds ago (several solver jobs started together), unless
    `force`; the working set is released every time (cheap right after the previous release).
    """
    if os.environ.get("FEA_RELEASE_MEMORY", "1") == "0":
        return False
    with _lock:
        now = time.monotonic()
        collect = force or now - _last_gc[0] >= MIN_INTERVAL
        if collect:
            _last_gc[0] = now
    if collect:
        gc.collect()
    if sys.platform != "win32":
        return False
    return _empty_working_set()
