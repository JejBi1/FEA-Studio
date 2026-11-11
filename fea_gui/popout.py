"""Pop-out windows: a part of the main window (the step panel, one panel section) moved into its own window and back.

`Popout(content, title, key)` keeps the place of `content` in its layout; `detach()` moves the widget into a `PopoutWindow`
(own window above the main window, any size, also on a second monitor) and leaves a placeholder with a "Return" button;
closing the window or `attach()` puts the widget back to the same place. The widget object stays the same – panel code,
signals and attributes (main.*) do not notice the move.

Window geometry is remembered (QSettings ui/popout/<key>) only with widgets.remember_ui(), like the collapsed sections.
"""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QLayout, QMainWindow, QPushButton,
                               QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from . import theme

_ALL: list["Popout"] = []


def _main_window() -> QMainWindow | None:
    for w in QApplication.topLevelWidgets():
        if isinstance(w, QMainWindow) and w.objectName() != "popoutWindow" and hasattr(w, "panel_stack"):
            return w
    return None


def _settings():
    from .widgets import _ui_settings
    return _ui_settings()


class PopoutWindow(QWidget):
    """Own window for a moved part of the main window. Owned by `parent` (it stays above the main window and closes with it);
    the keyboard shortcuts of the main window (Ctrl+S, Ctrl+Z, F5…) work in it too.
    """
    closing = Signal()

    def __init__(self, title: str, parent: QWidget, key: str = ""):
        super().__init__(parent, Qt.Window)
        self.setObjectName("popoutWindow")
        self.setWindowTitle(title)
        self._key = key
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        main = _main_window()
        if main is not None:
            self.setWindowIcon(main.windowIcon())
            for a in main.actions():                 # QAction with WindowShortcut: active in every window it is added to
                if not a.shortcut().isEmpty():
                    self.addAction(a)

    def set_content(self, w: QWidget) -> None:
        self.layout().addWidget(w, 1)

    def restore_geometry(self, default_size) -> None:
        from .widgets import remember_ui
        if self._key and remember_ui():
            g = _settings().value(f"ui/popout/{self._key}")
            if isinstance(g, (QByteArray, bytes)) and self.restoreGeometry(QByteArray(g)):
                return
        self.resize(*default_size)

    def save_geometry(self) -> None:
        from .widgets import remember_ui
        if self._key and remember_ui():
            s = _settings()
            s.setValue(f"ui/popout/{self._key}", self.saveGeometry())
            s.sync()

    def closeEvent(self, e):
        self.save_geometry()
        self.closing.emit()
        super().closeEvent(e)


class Popout(QObject):
    """Moves `content` between its place in the main window and its own window.

    title – window title; key – ASCII key for the remembered geometry; note – text of the placeholder; owner – parent of the
    window (default = the widget holding `content`, so the window disappears with it).
    """
    changed = Signal(bool)                       # True = in its own window

    def __init__(self, content: QWidget, title: str, key: str = "", *, note: str = "",
                 owner: QWidget | None = None, default_size=(560, 720)):
        super().__init__(content)
        self.content = content
        self.title = title
        self.key = key
        self.note = note or "Opened in a separate window."
        self.owner = owner
        self.default_size = default_size
        self.window: PopoutWindow | None = None
        self.placeholder: QFrame | None = None
        self._layout: QLayout | None = None
        self._index = 0
        self._stretch = 0
        _ALL.append(self)
        self.destroyed.connect(lambda *_: _ALL.remove(self) if self in _ALL else None)

    # --- state
    def is_detached(self) -> bool:
        return self.window is not None

    def toggle(self) -> None:
        self.attach() if self.is_detached() else self.detach()

    def _make_placeholder(self) -> QFrame:
        f = QFrame()
        f.setObjectName("popoutPlaceholder")
        f.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        h = QHBoxLayout(f)
        h.setContentsMargins(8, 6, 8, 6)
        lab = QLabel(self.note)
        lab.setProperty("muted", True)
        lab.setWordWrap(True)
        h.addWidget(lab, 1)
        b = QPushButton("Show window")
        b.setToolTip("Bring the separate window to the front")
        b.clicked.connect(self.raise_window)
        h.addWidget(b)
        b = QPushButton("Return here")
        theme.set_icon(b, "dock_in")
        b.setToolTip("Put the content back into the main window (also by closing its window)")
        b.clicked.connect(self.attach)
        h.addWidget(b)
        return f

    def detach(self) -> None:
        if self.is_detached():
            self.raise_window()
            return
        holder = self.content.parentWidget()
        lay = holder.layout() if holder is not None else None
        idx = lay.indexOf(self.content) if lay is not None else -1
        if idx < 0:
            return
        self._layout, self._index = lay, idx
        self._stretch = lay.stretch(idx) if hasattr(lay, "stretch") else 0
        size = self.content.size()
        self.placeholder = self._make_placeholder()
        lay.insertWidget(idx, self.placeholder)
        lay.removeWidget(self.content)
        win = PopoutWindow(self.title, self.owner or holder, self.key)
        win.set_content(self.content)
        self.content.setVisible(True)
        win.closing.connect(self._on_window_closed)
        self.window = win
        w = max(size.width() + 40, self.default_size[0])
        h = max(size.height() + 40, self.default_size[1])
        win.restore_geometry((w, h))
        win.show()
        win.raise_()
        win.activateWindow()
        self.changed.emit(True)

    def _on_window_closed(self) -> None:
        QTimer.singleShot(0, self.attach)        # after closeEvent – the window must not lose its child during its own event

    def attach(self) -> None:
        win = self.window
        if win is None:
            return
        self.window = None
        lay = self._layout
        if self.placeholder is not None:
            if lay is not None:
                lay.removeWidget(self.placeholder)
            self.placeholder.deleteLater()
            self.placeholder = None
        if lay is not None:
            win.layout().removeWidget(self.content)
            lay.insertWidget(min(self._index, lay.count()), self.content, self._stretch)
        if win.isVisible():
            win.save_geometry()
            win.hide()
        win.deleteLater()
        self.changed.emit(False)

    def raise_window(self) -> None:
        if self.window is not None:
            if self.window.isMinimized():
                self.window.showNormal()
            self.window.raise_()
            self.window.activateWindow()


class CloseToDock(QObject):
    """Closing a floating QDockWidget returns it to the main window instead of hiding it (the dock would be lost)."""

    def __init__(self, dock):
        super().__init__(dock)
        self.dock = dock
        dock.installEventFilter(self)

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if obj is self.dock and ev.type() == QEvent.Close and self.dock.isFloating():
            main = self.dock.parentWidget()
            if main is not None and main.isVisible() and not getattr(main, "_closing", False):
                ev.ignore()
                self.dock.setFloating(False)
                self.dock.show()
                self.dock.raise_()
                return True
        return False


def popout_of(w: QWidget | None) -> Popout | None:
    """The detached Popout whose window contains w (None = w is in the main window)."""
    while w is not None:
        if isinstance(w, PopoutWindow):
            for p in _ALL:
                if p.window is w:
                    return p
            return None
        w = w.parentWidget()
    return None


def raise_for(w: QWidget) -> None:
    """Brings the window holding w to the front (guide, What is this?, field error)."""
    p = popout_of(w)
    while p is not None:
        p.raise_window()
        p = popout_of(p.window.parentWidget()) if p.window is not None else None


def attach_for(w: QWidget) -> None:
    """Puts everything holding w back into the main window (the guide highlights elements in the main window only)."""
    p = popout_of(w)
    while p is not None:
        p.attach()
        p = popout_of(w)


def attach_all() -> None:
    for p in list(_ALL):
        p.attach()


def popout_button(tip: str = "Open in a separate window") -> QToolButton:
    b = QToolButton()
    b.setObjectName("popoutButton")
    b.setAutoRaise(True)
    b.setCursor(Qt.PointingHandCursor)
    theme.set_icon(b, "popout")
    b.setToolTip(tip)
    return b
