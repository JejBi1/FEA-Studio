"""About FEA Studio: version, licence (GPL-3.0-or-later, no warranty), licence texts and the folder of the diagnostic logs."""
from __future__ import annotations

import html
import platform
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                               QTabWidget, QVBoxLayout)

ROOT = Path(__file__).resolve().parents[1]
# (tab title, file in the program folder)
LICENSE_FILES = (("Licence (GPL-3.0)", "LICENSE"), ("Additional permission", "LICENSE-EXCEPTIONS.md"),
                 ("Third-party components", "THIRD_PARTY_LICENSES.md"))


def log_dir() -> Path:
    """Folder of the crash logs and the redirected output (user data, stays local)."""
    from fea_core.config import user_data_dir
    return user_data_dir() / "logs"


def _read(name: str) -> str | None:
    p = ROOT / name
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from fea_core import __version__
        self.setWindowTitle("About FEA Studio")
        self.resize(720, 560)
        lay = QVBoxLayout(self)
        head = QLabel()
        head.setTextFormat(Qt.RichText)
        head.setWordWrap(True)
        head.setOpenExternalLinks(True)
        head.setText(
            f"<h2>FEA Studio {html.escape(__version__)}</h2>"
            "<p>Strength analysis of parts and assemblies (CalculiX, Gmsh, OpenCascade).</p>"
            "<p>This program is free software: you can redistribute it and/or modify it under the terms of the "
            "GNU General Public License as published by the Free Software Foundation, either version 3 of the License, "
            "or (at your option) any later version (GPL-3.0-or-later), with the additional permission in LICENSE-EXCEPTIONS.</p>"
            "<p><b>This program comes with ABSOLUTELY NO WARRANTY</b>, without even the implied warranty of MERCHANTABILITY "
            "or FITNESS FOR A PARTICULAR PURPOSE. Results of a numerical analysis must be checked by a qualified engineer.</p>"
            "<p><a href=\"https://www.gnu.org/licenses/gpl-3.0.html\">https://www.gnu.org/licenses/gpl-3.0.html</a></p>")
        lay.addWidget(head)
        sysinfo = QLabel()
        sysinfo.setTextFormat(Qt.PlainText)
        sysinfo.setProperty("_i18n_done", True)
        sysinfo.setText(f"Python {platform.python_version()} · {platform.system()} {platform.release()} · {platform.machine()}")
        sysinfo.setProperty("muted", True)
        lay.addWidget(sysinfo)

        tabs = QTabWidget()
        tabs.setToolTip("Licence texts shipped with the program")
        for title, name in LICENSE_FILES:
            text = _read(name)
            ed = QPlainTextEdit()
            ed.setReadOnly(True)
            ed.setProperty("_i18n_done", True)       # licence texts are shown as they are (not translated)
            ed.setPlainText(text if text is not None else f"{name} was not found in the program folder ({ROOT}).")
            tabs.addTab(ed, title)
        lay.addWidget(tabs, 1)

        row = QHBoxLayout()
        b_lic = QPushButton("Open licences folder")
        b_lic.setToolTip("Licence texts of all bundled components (licenses folder of the installation)")
        b_lic.clicked.connect(self._open_licenses)
        b_log = QPushButton("Open log folder")
        b_log.setToolTip("Crash reports and program output (stay on this computer; attach them to a bug report)")
        b_log.clicked.connect(lambda: self._open_dir(log_dir()))
        row.addWidget(b_lic)
        row.addWidget(b_log)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        row.addWidget(bb)
        lay.addLayout(row)

    def _open_licenses(self):
        d = ROOT / "licenses"
        self._open_dir(d if d.is_dir() else ROOT)

    @staticmethod
    def _open_dir(d: Path):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))


def show_about(parent=None) -> None:
    AboutDialog(parent).exec()


# ---------------------------------------------------------------------------------------------- diagnostics
_LOG_MAX = 2 * 1024 * 1024          # redirected output: one rotation at 2 MB (fea_studio.log → fea_studio.log.1)


def _anonymise(text: str) -> str:
    """The user's home folder → "<home>" (paths in tracebacks; the log stays local, but is meant to be attached)."""
    import os
    for key in ("USERPROFILE", "HOME"):
        h = os.environ.get(key)
        if h and len(h) > 3:
            text = text.replace(h, "<home>").replace(h.replace("\\", "/"), "<home>")
    return text


def write_crash_log(exc_type, exc, tb, where: str = "") -> Path | None:
    """crash-<time>.log in log_dir() with the traceback, version and platform. None = could not be written."""
    import datetime
    import traceback
    try:
        from fea_core import __version__
    except Exception:  # noqa: BLE001
        __version__ = "?"
    try:
        d = log_dir()
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        p = d / f"crash-{stamp}.log"
        k = 1
        while p.exists():
            k += 1
            p = d / f"crash-{stamp}-{k}.log"
        body = "".join(traceback.format_exception(exc_type, exc, tb))
        head = (f"FEA Studio {__version__}\nPython {platform.python_version()} ({sys.executable and Path(sys.executable).name})\n"
                f"Platform {platform.platform()}\nTime {datetime.datetime.now().isoformat(timespec='seconds')}\n"
                + (f"Where {where}\n" if where else "") + "\n")
        p.write_text(_anonymise(head + body), encoding="utf-8")
        return p
    except Exception:  # noqa: BLE001 – the crash handler must never raise
        return None


def _redirect_output(d: Path):
    """pythonw has no console (sys.stdout / sys.stderr are None) → output goes to fea_studio.log (one rotation)."""
    p = d / "fea_studio.log"
    try:
        if p.exists() and p.stat().st_size > _LOG_MAX:
            p.replace(d / "fea_studio.log.1")
    except OSError:
        pass
    f = open(p, "a", encoding="utf-8", errors="replace", buffering=1)
    import datetime
    f.write(f"\n===== start {datetime.datetime.now().isoformat(timespec='seconds')} =====\n")
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f
    return f


_last_box = [0.0]


def _show_box(path: Path | None, exc) -> None:
    import time
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance()
        if app is None or time.monotonic() - _last_box[0] < 5.0:
            return
        _last_box[0] = time.monotonic()
        where = (f"Details were saved to:\n{path}\n\nAttach this file to a bug report." if path is not None
                 else "The details could not be saved.")
        box = QMessageBox(QMessageBox.Critical, "FEA Studio – unexpected error",
                          f"An unexpected error occurred: {type(exc).__name__}: {exc}\n\n{where}")
        box.setTextFormat(Qt.PlainText)
        box.exec()
    except Exception:  # noqa: BLE001
        pass


def install_diagnostics() -> Path | None:
    """Crash log for unhandled exceptions (main thread and other threads), faulthandler for hard crashes, output of pythonw
    to a log file. Returns the log folder (None = not available). Everything stays on this computer.
    """
    import faulthandler
    import threading
    try:
        d = log_dir()
        d.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001 – without a writable folder the program still runs
        return None
    out = None
    if sys.stdout is None or sys.stderr is None:
        try:
            out = _redirect_output(d)
        except OSError:
            out = None
    try:
        # hard crash (access violation in a native library): Python traceback of all threads to the log
        target = out if out is not None else open(d / "faulthandler.log", "a", encoding="utf-8")
        faulthandler.enable(file=target, all_threads=True)
        install_diagnostics._fh = target           # keep the file open for the lifetime of the process
    except Exception:  # noqa: BLE001
        pass
    prev = sys.excepthook

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return prev(exc_type, exc, tb)
        path = write_crash_log(exc_type, exc, tb)
        try:
            prev(exc_type, exc, tb)                # also to stderr (console / log file)
        except Exception:  # noqa: BLE001
            pass
        _show_box(path, exc)
    sys.excepthook = hook

    def thook(args):
        path = write_crash_log(args.exc_type, args.exc_value, args.exc_traceback,
                               where=f"thread {getattr(args.thread, 'name', '?')}")
        if sys.stderr is not None:
            try:
                import traceback
                traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=sys.stderr)
                if path is not None:
                    print(f"Crash log: {path}", file=sys.stderr)
            except Exception:  # noqa: BLE001
                pass
    threading.excepthook = thook
    return d
