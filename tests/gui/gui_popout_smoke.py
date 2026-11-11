"""Pop-out windows: the step panel, one panel section and the Tables dock in a separate window and back.

python tests/gui/hidden_run.py tests/gui/gui_popout_smoke.py OUT_DIR [project.fea]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_popout")
OUT.mkdir(parents=True, exist_ok=True)
PROJECT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("examples/example2_assembly.fea")
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)

app = QApplication(sys.argv)
from fea_gui.main import MainWindow  # noqa: E402
from fea_gui.popout import PopoutWindow, attach_all, popout_of  # noqa: E402
from fea_gui.widgets import Section  # noqa: E402

fails = []


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text)
    if not cond:
        fails.append(text)


def settle():
    for _ in range(3):
        app.processEvents()


w = MainWindow()
w.resize(1600, 950)
w.show()
w.open_file(PROJECT)
settle()
stack = w.panel_stack

# --- step panel
w.set_stage(w.STAGE["Materials"])
w.btn_panel_popout.click()
settle()
check(isinstance(stack.window(), PopoutWindow) and stack.window().isVisible(), "panel in its own window")
check(w.act_panel_window.isChecked(), "View menu item checked")
check("Materials" in stack.window().windowTitle(), f"window title = step ({stack.window().windowTitle()})")
w.set_stage(w.STAGE["Constraints"])
settle()
check(stack.currentIndex() == w.STAGE["Constraints"] and "Constraints" in stack.window().windowTitle(),
      "step tab switches the panel in the window")
check(any(a.shortcut().toString() == "Ctrl+S" for a in stack.window().actions()), "Ctrl+S works in the window")
stack.window().grab().save(str(OUT / "panel_window.png"))
stack.window().close()
settle()
check(stack.window() is w and not w.panel_detached(), "closing the window returns the panel")
check(not w.act_panel_window.isChecked(), "View menu item unchecked")
w.act_panel_window.setChecked(True)
settle()
check(w.panel_detached(), "View menu detaches the panel")
w.act_panel_window.setChecked(False)
settle()
check(not w.panel_detached() and stack.window() is w, "View menu returns the panel")

# --- one section
w.set_stage(w.STAGE["Constraints"])
settle()
sec = next(s for s in stack.widget(w.STAGE["Constraints"]).findChildren(Section) if s._pop_btn is not None)
body = sec.body
sec._pop_btn.click()
settle()
check(sec.is_detached() and isinstance(body.window(), PopoutWindow) and body.isVisible(), "section in its own window")
check(sec._pop.placeholder is not None and sec._pop.placeholder.isVisible(), "placeholder in the panel")
sec.set_open(False)
settle()
check(body.isVisible() and not sec._pop.placeholder.isVisible(), "collapsing the header hides only the placeholder")
w.set_stage(w.STAGE["Loads"])
settle()
check(body.window().isVisible(), "section window stays when the step changes")
w.set_stage(w.STAGE["Constraints"])
settle()
Section.reveal(body)
check(popout_of(body) is sec._pop, "popout_of finds the section window")
body.window().close()
settle()
check(not sec.is_detached() and body.window() is w and body.parentWidget() is sec, "closing returns the section")
check(not sec.is_open() or body.isVisible(), "section state kept")
# section in a detached panel: the section window is owned by the panel window, attach_all returns both
w.set_panel_detached(True)
settle()
sec.detach()
settle()
check(sec.is_detached() and stack.window() is not w, "section from a detached panel")
attach_all()
settle()
check(not sec.is_detached() and not w.panel_detached() and body.parentWidget() is sec, "attach_all returns everything")

# --- Tables dock
d = w.dock_tables
w.table_dock.btn_popout.click()
settle()
check(d.isFloating() and d.isVisible() and d.titleBarWidget() is None, "Tables floating with a title bar")
check(w.act_tables_window.isChecked(), "View menu Tables checked")
d.close()
settle()
settle()
check(not d.isFloating() and d.isVisible() and d.titleBarWidget() is not None, "closing returns the Tables dock")
w.act_tables_window.setChecked(True)
settle()
check(d.isFloating(), "View menu floats the tables")
w.act_tables_window.setChecked(False)
settle()
check(not d.isFloating(), "View menu docks the tables")

w.close()
print("FAILS:", len(fails))
sys.exit(1 if fails else 0)
