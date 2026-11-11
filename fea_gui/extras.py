"""Extra tools of the main window wired at the end of MainWindow.__init__ (one call `install_extras`):
Tools → Parametric study… / Job queue…, View → Python console (dock), File → Recent files + autosave (session_files).

The Python console runs the typed code in the GUI process (full access – intended for the user at the keyboard). It is NOT
reachable from IPC / MCP: `ipc_server` only calls `MainWindow.execute_mcp_action`, which knows a fixed list of GUI commands
and the registered MCP tools (tests/test_console_not_remote.py).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QLabel, QVBoxLayout, QWidget

CONSOLE_NOTE = ("The console runs Python code inside FEA Studio with full access to your files – run only code you trust. "
                "It is not available to MCP / LLM agents.")


def install_extras(main) -> None:
    from . import session_files
    menus = [b.menu() for b in (getattr(main, "menu_buttons", {}).get("Tools"), getattr(main, "tools_button", None))
             if b is not None and b.menu() is not None]
    seen = set()
    acts = []
    for text, slot, tip in (("Parametric study…", lambda: open_param_dialog(main),
                             "Sweep loads, materials, mesh size, print settings – KPI table and chart"),
                            ("Job queue…", lambda: open_queue_dialog(main),
                             "Solve several .fea projects one after another in the background")):
        a = main._act(text, slot, tip=tip)
        acts.append(a)
    for m in menus:
        if id(m) in seen:
            continue
        seen.add(id(m))
        m.addSeparator()
        for a in acts:
            m.addAction(a)
    vm = getattr(main, "view_menu", None)
    if vm is not None:
        a = main.act_console = vm.addAction("Python console")
        a.setCheckable(True)
        a.setToolTip("Dock with a Python console over the open study (code runs in this program)")
        a.toggled.connect(lambda on: show_console(main, on))
    session_files.install(main)


def open_param_dialog(main):
    from .main import APP_NAME, msg
    if main.study is None:
        msg(main, APP_NAME, "Open a model or a project first.")
        return None
    from .param_dialog import ParamDialog
    dlg = getattr(main, "_param_dlg", None)
    if dlg is None:
        dlg = main._param_dlg = ParamDialog(main)
        dlg.finished.connect(lambda *_: setattr(main, "_param_dlg", None))
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
    dlg.show()
    dlg.raise_()
    return dlg


def open_queue_dialog(main):
    from .queue_dialog import QueueDialog
    dlg = getattr(main, "_queue_dlg", None)
    if dlg is None:
        dlg = main._queue_dlg = QueueDialog(main)
        dlg.finished.connect(lambda *_: setattr(main, "_queue_dlg", None))
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
    dlg.show()
    dlg.raise_()
    return dlg


def console_dock(main) -> QDockWidget:
    """The Python console dock (created at the first use, bottom area next to the solver log)."""
    d = getattr(main, "dock_console", None)
    if d is not None:
        return d
    from .console_panel import ConsolePanel
    d = QDockWidget("Python console", main)
    d.setObjectName("dock_console")
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(4, 4, 4, 0)
    note = QLabel(CONSOLE_NOTE)
    note.setWordWrap(True)
    note.setProperty("muted", True)
    note.setToolTip("Security note: the console is for the user at the keyboard only")
    lay.addWidget(note)
    main.console_panel = ConsolePanel(main)
    lay.addWidget(main.console_panel, 1)
    d.setWidget(w)
    main.addDockWidget(Qt.BottomDockWidgetArea, d)
    log = getattr(main, "dock_log", None)
    if log is not None:
        main.tabifyDockWidget(log, d)
    a = getattr(main, "act_console", None)
    if a is not None:
        d.visibilityChanged.connect(lambda vis: (a.blockSignals(True), a.setChecked(vis or not d.isHidden()),
                                                 a.blockSignals(False)))
    main.dock_console = d
    return d


def show_console(main, on: bool = True) -> None:
    if not on and getattr(main, "dock_console", None) is None:
        return
    d = console_dock(main)
    d.setVisible(on)
    if on:
        d.raise_()
        main.console_panel.input_edit.setFocus()
