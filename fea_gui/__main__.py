"""Starting the GUI: python -m fea_gui [file.step|file.fea|study.json].

The splash window appears right after the QApplication is created – loading OpenCascade (OCP) and the other libraries takes 1–3 s
and without it the user would see nothing after starting.
"""
import os
import sys
from pathlib import Path

if sys.flags.ignore_environment:
    # the installer's shortcut runs `pythonw -E`: PYTHONHOME / PYTHONPATH of another program (GIS, CAD, older Python) would
    # not let the .venv start – delete them also for the helper processes (Gmsh, NGSolve, MCP…) that do not get -E
    for _k in ("PYTHONHOME", "PYTHONPATH"):
        os.environ.pop(_k, None)

if sys.platform.startswith("linux"):
    # the OpenCascade 3D window draws into a native X11 window (Xw_Window) – in a Wayland session Qt therefore runs via XWayland
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

ICON_DIR = Path(__file__).resolve().parent.parent / "installer" / "icons"   # make_icon.py, fea_studio.ico/png
APP_ID = "FEAStudio.FEAStudio"


def _app_id():
    """Windows: own AppUserModelID of the process – the taskbar then shows the FEA Studio window icon instead of the
    Python icon (pythonw from .venv starts the base interpreter as a subprocess, which would otherwise group with pythonw.exe).
    It must be set before the first window is created.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:  # noqa: BLE001 – without the ID only the Python icon is shown
        pass


def _app_icon(app):
    """Application icon (all windows, taskbar) from installer/icons; if the files are missing, nothing happens."""
    from PySide6.QtGui import QIcon, QImageReader

    icon = QIcon()
    for name in ("fea_studio.ico", "fea_studio.png"):   # ICO = sharp small sizes 16–128, PNG 256
        p = ICON_DIR / name
        if p.is_file() and QImageReader(str(p)).canRead():
            icon.addFile(str(p))
    if not icon.isNull():
        app.setWindowIcon(icon)
    if sys.platform.startswith("linux"):
        app.setDesktopFileName("fea-studio")            # pairing the window with the fea-studio.desktop shortcut


def _splash():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
    from PySide6.QtWidgets import QApplication, QSplashScreen

    app = QApplication.instance() or QApplication(sys.argv)
    try:
        _app_icon(app)
    except Exception:  # noqa: BLE001 – the icon is not a condition for starting
        pass
    pm = QPixmap(460, 150)
    pm.fill(QColor("#1f2933"))
    p = QPainter(pm)
    p.setPen(QColor("#f5f7fa"))
    f = QFont(app.font())
    f.setPointSize(24)
    f.setBold(True)
    p.setFont(f)
    p.drawText(pm.rect().adjusted(0, 0, 0, -30), Qt.AlignCenter, "FEA Studio")
    f.setPointSize(10)
    f.setBold(False)
    p.setFont(f)
    p.setPen(QColor("#9aa5b1"))
    p.drawText(pm.rect().adjusted(0, 70, 0, 0), Qt.AlignCenter, "CalculiX · Gmsh · OpenCascade")
    p.end()
    s = QSplashScreen(pm, Qt.WindowStaysOnTopHint)
    s.show()
    app.processEvents()
    return app, s


def _diagnostics():
    """Crash log (user data / logs / crash-<time>.log), faulthandler and – under pythonw, where stdout / stderr are None –
    the program output in fea_studio.log. Must not stop the start."""
    try:
        from .about import install_diagnostics
        install_diagnostics()
    except Exception:  # noqa: BLE001
        pass


_diagnostics()
_app_id()
os.environ.setdefault("FEA_UI_REMEMBER", "1")    # remember collapsed sections and pinned steps (not for the MainWindow tests)
try:
    app, splash = _splash()
except Exception:  # noqa: BLE001 – the splash window is not a condition for starting
    app, splash = None, None

from .main import main  # noqa: E402

main(app, splash)
