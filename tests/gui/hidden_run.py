"""Runs a GUI test with the windows off the visible area and without activation (nothing pops up for the user).

python tests/gui/hidden_run.py tests/gui/gui_x_smoke.py OUT_DIR [...]
The offscreen platform of Qt does not work (the OCCT 3D window needs a native window) – the window is therefore only moved off the screen.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

FAR = -30000
_show = QWidget.show
_move = QWidget.move
_setGeometry = QWidget.setGeometry


def _hidden(w):
    if w.isWindow():
        w.setAttribute(Qt.WA_ShowWithoutActivating, True)
        w.setWindowFlag(Qt.WindowDoesNotAcceptFocus, False)
        _move(w, FAR, FAR)


def show(self):
    _hidden(self)
    _show(self)
    if self.isWindow():
        _move(self, FAR, FAR)


def move(self, *a):
    if self.isWindow():
        return _move(self, FAR, FAR)
    return _move(self, *a)


QWidget.show = show
QWidget.move = move
QWidget.activateWindow = lambda self: None
QWidget.raise_ = (lambda orig: (lambda self: None if self.isWindow() else orig(self)))(QWidget.raise_)

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
