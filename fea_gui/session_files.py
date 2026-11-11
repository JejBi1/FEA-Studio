"""Recent files (File menu) and autosave of the open project with a restore offer after a crash.

Pure helpers (no Qt, tested in tests/test_session_files.py): `push_recent`, `existing`, `results_nbytes`, `write_autosave`,
`stale_autosaves`, `discard`. The Qt part `install(main)` adds File → Recent files, the autosave timer and the restore question.

Persistent state is written only in the program started by `python -m fea_gui` (widgets.remember_ui – FEA_UI_REMEMBER=1);
tests and scripts that create the window directly neither read the autosaves nor write the settings.

Autosave: every `autosave_minutes` (QSettings, default 5, 0 = off) when the study has unsaved changes and nothing is computing,
the project is saved to user_data_dir()/autosave/<pid>.fea (+ <pid>.json: source file, time, process). Results larger than
RESULTS_LIMIT_MB are left out (the study only). A clean exit and a save remove the autosave of this process; an autosave of a
process that no longer runs = the program ended abnormally → the restore question at the next start.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

RECENT_KEY = "recent_files"
RECENT_MAX = 10
RESULTS_LIMIT_MB = 200
DEFAULT_MINUTES = 5


# --------------------------------------------------------------------- pure helpers
def _norm(p) -> str:
    s = str(Path(p).resolve()) if str(p) else ""
    return s


def _same(a: str, b: str) -> bool:
    return a.lower() == b.lower() if sys.platform == "win32" else a == b


def push_recent(items, path, limit: int = RECENT_MAX) -> list[str]:
    """New list of recent files: `path` first, without duplicates (case-insensitive on Windows), at most `limit`."""
    p = _norm(path)
    out = [p] + [str(x) for x in (items or []) if str(x) and not _same(str(x), p)]
    return out[:limit]


def existing(items) -> list[str]:
    """Only the files that still exist (a moved / deleted project is not offered)."""
    out = []
    for x in items or []:
        try:
            if str(x) and Path(str(x)).is_file():
                out.append(str(x))
        except (OSError, ValueError):
            pass
    return out


def as_list(value) -> list[str]:
    """QSettings value → list of strings (one string, list, None)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(v) for v in value]


def results_nbytes(results) -> int:
    """Memory size of the numpy arrays of the results (mesh, states, fields, thermal states, modes) – estimate of the .npz size."""
    import numpy as np
    seen: set[int] = set()

    def walk(o, depth: int) -> int:
        if o is None or id(o) in seen or depth > 4:
            return 0
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            return int(o.nbytes)
        if isinstance(o, dict):
            return sum(walk(v, depth + 1) for v in o.values())
        if isinstance(o, (list, tuple)):
            return sum(walk(v, depth + 1) for v in o)
        d = getattr(o, "__dict__", None)
        if d is not None and not isinstance(o, type):
            return sum(walk(v, depth + 1) for k, v in d.items() if not k.startswith("_cache"))
        return 0
    return walk(results, 0)


def autosave_dir() -> Path:
    from fea_core.config import user_data_dir
    return user_data_dir() / "autosave"


def _create_time(pid: int) -> float | None:
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 – the process does not exist / no access
        return None


def write_autosave(folder: Path, study, step_path, results, source=None, limit_mb: float = RESULTS_LIMIT_MB,
                   pid: int | None = None) -> dict:
    """Saves the project to folder/<pid>.fea and the description <pid>.json; results above limit_mb are left out."""
    from fea_core.project import save_project
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pid = os.getpid() if pid is None else int(pid)
    big = results is not None and results_nbytes(results) > limit_mb * 1e6
    fea = folder / f"{pid}.fea"
    save_project(fea, study, step_path or "", None if big else results)
    meta = {"pid": pid, "created": _create_time(pid), "time": time.time(), "source": str(source) if source else "",
            "study": str(getattr(study, "name", "")), "with_results": results is not None and not big,
            "results_left_out": bool(big), "file": str(fea)}
    tmp = folder / f"{pid}.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    tmp.replace(folder / f"{pid}.json")
    return meta


def _alive(meta: dict) -> bool:
    ct = _create_time(int(meta.get("pid", -1)))
    if ct is None:
        return False
    created = meta.get("created")
    return created is None or abs(ct - float(created)) < 2.0        # the same pid reused by another process = not alive


def stale_autosaves(folder: Path) -> list[dict]:
    """Autosaves of processes that no longer run (abnormal end), newest first. A broken description is skipped."""
    folder = Path(folder)
    out = []
    if not folder.is_dir():
        return out
    for js in folder.glob("*.json"):
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        fea = folder / f"{meta.get('pid')}.fea"
        if not fea.is_file() or _alive(meta):
            continue
        meta["file"] = str(fea)
        meta["meta_file"] = str(js)
        out.append(meta)
    return sorted(out, key=lambda m: -float(m.get("time") or 0))


def discard(meta_or_pid, folder: Path | None = None) -> None:
    """Deletes an autosave (description dict from stale_autosaves / write_autosave, or the pid of this process)."""
    if isinstance(meta_or_pid, dict):
        files = [meta_or_pid.get("file"), meta_or_pid.get("meta_file")]
        if not meta_or_pid.get("meta_file") and meta_or_pid.get("file"):
            files.append(str(Path(meta_or_pid["file"]).with_suffix(".json")))
    else:
        f = Path(folder or autosave_dir())
        files = [f / f"{meta_or_pid}.fea", f / f"{meta_or_pid}.json"]
    for p in files:
        if p:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


def in_autosave_dir(path) -> bool:
    try:
        return Path(path).resolve().parent == autosave_dir().resolve()
    except OSError:
        return False


# --------------------------------------------------------------------- Qt
def _remember() -> bool:
    from .widgets import remember_ui
    return remember_ui()


def recent_files(main) -> list[str]:
    return existing(as_list(main.settings.value(RECENT_KEY)))


def note_opened(main, path) -> None:
    """A file was opened (MainWindow.open_file) → first in the recent list (not an autosave copy)."""
    if not _remember() or path is None or in_autosave_dir(path):
        return
    main.settings.setValue(RECENT_KEY, push_recent(as_list(main.settings.value(RECENT_KEY)), path))


def note_saved(main, path) -> None:
    """The project was saved (MainWindow._save) → recent list; the autosave of this process is no longer needed."""
    note_opened(main, path)
    if _remember():
        discard(os.getpid())


def _fill_recent(main, menu) -> None:
    menu.clear()
    items = recent_files(main)
    for p in items:
        a = menu.addAction(Path(p).name, lambda path=p: main.open_file(path))
        a.setToolTip(p)
        a.setProperty("_i18n_done", True)
    if not items:
        a = menu.addAction("(no recent files)")
        a.setEnabled(False)
    menu.addSeparator()
    a = menu.addAction("Clear list", lambda: main.settings.setValue(RECENT_KEY, []))
    a.setEnabled(bool(items))


def _autosave_tick(main) -> None:
    if main.study is None or not main.dirty or main.busy():
        return
    try:
        meta = write_autosave(autosave_dir(), main.study, main.geometry_str(), main.results, main.project_path)
    except Exception as e:  # noqa: BLE001 – autosave must never disturb the work
        main.log_msg(f"Autosave failed: {e}")
        return
    note = " (results left out – too large)" if meta["results_left_out"] else ""
    main.statusBar().showMessage(f"Autosaved{note}", 3000)


def _offer_restore(main) -> None:
    from PySide6.QtWidgets import QMessageBox
    for meta in stale_autosaves(autosave_dir()):
        when = time.strftime("%d. %m. %Y %H:%M", time.localtime(float(meta.get("time") or 0)))
        src = meta.get("source") or ""
        name = meta.get("study") or Path(src).stem or "untitled"
        extra = "" if meta.get("with_results") else "\nThe autosave contains the study only (no results)."
        r = QMessageBox.question(
            main, "Restore autosave",
            f"FEA Studio did not end properly last time.\nAn autosaved copy of “{name}” from {when} exists."
            f"{extra}\n\nRestore it?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if r != QMessageBox.Yes:
            discard(meta)
            continue
        main.open_file(meta["file"])
        if main.project_path is not None and in_autosave_dir(main.project_path):
            main.project_path = Path(src) if src and Path(src).suffix.lower() == ".fea" else None
            main.dirty = True
            main.update_title()
            main.log_msg(f"Restored autosave of “{name}” from {when} – save it to keep it")
            discard(meta)
        break


def install(main) -> None:
    """File → Recent files, autosave timer and the restore offer (the latter two only with remember_ui)."""
    from PySide6.QtCore import QCoreApplication, QTimer
    btn = getattr(main, "menu_buttons", {}).get("File")
    if btn is not None and btn.menu() is not None:
        m = btn.menu()
        sub = m.addMenu("Recent files")
        sub.setToolTip("Recently opened or saved projects and models (only existing files)")
        sub.aboutToShow.connect(lambda: _fill_recent(main, sub))
        acts = m.actions()
        first_sep = next((a for a in acts if a.isSeparator()), None)
        if first_sep is not None:                       # after Open / Save / Save as
            m.removeAction(sub.menuAction())
            m.insertMenu(first_sep, sub)
        main.recent_menu = sub
    if not _remember():
        return
    minutes = float(main.settings.value("autosave_minutes", DEFAULT_MINUTES) or 0)
    if minutes > 0:
        t = QTimer(main)
        t.setInterval(int(minutes * 60000))
        t.timeout.connect(lambda: _autosave_tick(main))
        t.start()
        main.autosave_timer = t
    app = QCoreApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(lambda: discard(os.getpid()))     # clean exit = no restore offer
    QTimer.singleShot(1500, lambda: _offer_restore(main))
