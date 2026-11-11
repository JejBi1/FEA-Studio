"""Pictures of the Word report drawn in parallel threads.

The report builds the scenes in order (main thread) and hands each picture to `render` here; while a `Pool` is active
the picture is drawn by a worker thread (the numba kernels release the GIL, every picture has its own matplotlib
Figure without pyplot – the font cache of matplotlib is per thread) and `_Doc.image` leaves a placeholder run that
`Pool.finish` fills with the picture in document order before the report is saved. The pixels are the same as when
drawn one after another (same code, same inputs; the language of the labels is that of the calling thread).
Without an active pool (MCP, other callers) `render` draws at once.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .. import i18n

_active = threading.local()        # Pool of the thread that builds the report
MAX_WORKERS = 4
MEM_PER_WORKER_GB = 0.5             # raster buffers of a 1600 × 1000 picture (supersampled 2×) + scene + Agg canvas


def workers() -> int:
    """Number of drawing threads: physical cores − 1, at most MAX_WORKERS, as free RAM allows (FEA_REPORT_THREADS)."""
    env = os.environ.get("FEA_REPORT_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    from ..config import NUM_THREADS
    n = max(1, min(MAX_WORKERS, NUM_THREADS - 1))
    try:
        import psutil
        free = psutil.virtual_memory().available / 1e9
        n = max(1, min(n, int(free / MEM_PER_WORKER_GB / 2)))
    except Exception:  # noqa: BLE001 – without psutil the core count only
        pass
    return n


def _init_worker():
    from ..mcp.render import keep_buffers
    keep_buffers()                        # raster buffers kept for the life of the worker thread


class Pool:
    """Parallel drawing of the report pictures; use as a context manager in the thread that builds the report.
    One worker (small machine, FEA_REPORT_THREADS=1) = the pictures are drawn at once in the calling thread.
    """

    def __init__(self, n: int | None = None):
        self.n = workers() if n is None else int(n)
        self.ex = ThreadPoolExecutor(self.n, thread_name_prefix="fea-report-img", initializer=_init_worker) \
            if self.n > 1 else None
        self.futures: dict[str, Future] = {}
        self.slots: list[tuple] = []            # (run, path, width) in document order

    def __enter__(self):
        from ..mcp.render import reuse_buffers
        self._prev = getattr(_active, "pool", None)
        _active.pool = self if self.ex is not None else None
        self._reuse = reuse_buffers()           # pictures drawn in this thread (pool of one, other renderers)
        self._reuse.__enter__()
        return self

    def __exit__(self, *exc):
        _active.pool = self._prev
        self._reuse.__exit__(None, None, None)
        if self.ex is not None:
            for f in self.futures.values():
                f.cancel()
            self.ex.shutdown(wait=True)
            self.ex = None
        return False

    def submit(self, scene, path, view, width, height) -> None:
        from ..mcp.render import render as draw
        if str(Path(path)) in self.futures:     # the same file drawn again: the earlier picture goes into its place first
            self._place(str(Path(path)), [])
        lang = i18n.language()

        def job():
            with i18n.using(lang):
                draw(scene, path, view, width, height)
        self.futures[str(Path(path))] = self.ex.submit(job)

    def pending(self, path) -> bool:
        return str(Path(path)) in self.futures

    def _place(self, key: str | None, warnings: list[str]) -> None:
        """Waits for the pictures (all, or only the file key) and puts them into their placeholder runs."""
        keep = []
        for run, path, width in self.slots:
            if key is not None and str(Path(path)) != key:
                keep.append((run, path, width))
                continue
            err = self.futures[str(Path(path))].exception()
            if err is None:
                run.add_picture(str(path), width=width)
            else:                               # drawing failed: the reason stays in place of the picture
                run.add_text(i18n.tr("The figure could not be drawn: ") + str(err))
                warnings.append(f"Figure {Path(path).stem}: {err}")
        self.slots = keep
        if key is not None:
            self.futures.pop(key).exception()   # drawn (the file may now be overwritten)

    def finish(self, warnings: list[str]) -> None:
        """Waits for all pictures and puts them into their placeholder runs (document order)."""
        self._place(None, warnings)


def current() -> Pool | None:
    return getattr(_active, "pool", None)


def numbered(img_dir: Path, prefix: str) -> Path:
    """img_dir/<prefix><n>.png with n = the pictures of this prefix so far + 1 – drawn ones and those still being drawn."""
    names = {p.name for p in img_dir.glob(f"{prefix}*.png")}
    pool = current()
    if pool is not None:
        names |= {Path(p).name for p in pool.futures if Path(p).parent == img_dir and Path(p).name.startswith(prefix)
                  and Path(p).suffix == ".png"}
    return img_dir / f"{prefix}{len(names) + 1}.png"


def render(scene, path, view="iso", width: int = 1280, height: int = 720) -> None:
    """mcp.render.render for the report: in a worker thread while a Pool is active, otherwise at once."""
    pool = current()
    if pool is None:
        from ..mcp.render import render as draw
        draw(scene, path, view, width, height)
    else:
        pool.submit(scene, path, view, width, height)
