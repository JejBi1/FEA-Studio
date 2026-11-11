"""Parametric study dialog, job queue, Python console dock, nonlinear convergence monitor, polygon imprint and recent files.

python tests/gui/hidden_run.py tests/gui/gui_extras_smoke.py OUT_DIR
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("FEA_NO_IPC", "1")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_extras").resolve()
OUT.mkdir(parents=True, exist_ok=True)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
_warn = []
QMessageBox.warning = staticmethod(lambda *a, **k: _warn.append(a[2] if len(a) > 2 else "") or QMessageBox.Ok)

app = QApplication(sys.argv)
from fea_core.geometry import Part, read_step, write_step  # noqa: E402
from fea_core.project import save_project  # noqa: E402
from fea_core.selectors import find_faces  # noqa: E402
from fea_core.study import Fixed, Force, LoadCase, MeshSettings, Study  # noqa: E402
from fea_gui.main import MainWindow  # noqa: E402
from verification.cases import STEEL, box  # noqa: E402

fails = []


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text)
    if not cond:
        fails.append(text)


def wait(pred, limit=600.0):
    t0 = time.time()
    while not pred() and time.time() - t0 < limit:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    return pred()


step = OUT / "tyc.step"
write_step([Part("Tyc", box(0, 0, 0, 40, 10, 10))], step)
parts = read_step(step)
st = Study("Tyc", str(step), mesh=MeshSettings(size=5.0))
st.add_material(STEEL, default=True)
st.add_selection("vetknuti", find_faces(parts, x=0))
st.add_selection("konec", find_faces(parts, x=40))
st.constraints.append(Fixed("vetknuti"))
st.load_cases.append(LoadCase("Tah", [Force("konec", [1000.0, 0, 0])]))
proj = OUT / "tyc.fea"
save_project(proj, st, step)

w = MainWindow()
w.resize(1500, 950)
w.show()
w.open_file(proj)
app.processEvents()
check(w.study is not None and w.study.name == "Tyc", "project opened")
check(hasattr(w, "recent_menu"), "File → Recent files exists")

# --- parametric study
from fea_gui.extras import open_param_dialog, open_queue_dialog  # noqa: E402
dlg = open_param_dialog(w)
row = dlg.rows[0]
row.values.setText("1, 2, 3")
app.processEvents()
check("one analysis" in dlg.est.text(), f"estimate says linear shortcut ({dlg.est.text()})")
dlg.btn_run.click()
check(wait(lambda: dlg.thread is None), "load sweep finished")
check(len(dlg.points) == 3 and dlg.result is not None and dlg.result.linear_shortcut, "3 scaled points")
u = [p.max_disp for p in dlg.points]
check(abs(u[2] / u[0] - 3.0) < 1e-9, f"displacement scales with the factor ({u})")
row.kind.setCurrentIndex(row.kind.findData("material_prop"))
row.field.setCurrentText("E")
row.values.setText("105000; 210000")
app.processEvents()
dlg.btn_run.click()
check(wait(lambda: dlg.thread is None), "E sweep finished")
check(len(dlg.points) == 2 and all(p.converged for p in dlg.points), "2 solved points")
check(dlg.points[0].max_disp > 1.9 * dlg.points[1].max_disp, "half E = double displacement")
check(dlg.rtab.rowCount() == 2 and dlg.rtab.columnCount() >= 4, "result table filled")
check(dlg.plot_w.canvas is not None, "chart drawn")
dlg.grab().save(str(OUT / "param_dialog.png"))
dlg.rtab.selectRow(0)
app.processEvents()
dlg.btn_apply.click()
app.processEvents()
check(abs(w.study.materials[w.study.default_material].E - 105000) < 1e-6 and w.dirty, "point taken over into the study")
check(getattr(w, "param_study", None) is not None, "result kept for the report")
w.undo()
check(abs(w.study.materials[w.study.default_material].E - 210000) < 1e-6, "undo restores the study")
dlg.close()
app.processEvents()

# --- Python console
w.act_console.setChecked(True)
app.processEvents()
check(w.dock_console.isVisible(), "console dock shown")
w.console_panel.input_edit.setPlainText("print('study =', study.name)")
w.console_panel.run_code()
check("study = Tyc" in w.console_panel.output_edit.toPlainText(), "console runs code over the open study")
w.dock_console.grab().save(str(OUT / "console.png"))
w.act_console.setChecked(False)
app.processEvents()
check(not w.dock_console.isVisible(), "console dock hidden")

# --- nonlinear monitor (from saved convergence data)
mon = w.nl_monitor
w.set_stage(w.STAGE["Solve"])
check(mon.isHidden(), "monitor hidden for a linear study without data")


class _R:
    mesh = type("M", (), {"nodes": []})()
    steps, thermal, warnings, modes = [], [], [], []
    info = {"nonlinear": {"convergence": {"job": {
        "iterations": [{"step": 1, "inc": 1, "att": 1, "iteration": i, "resid_force_pct": 10.0 ** (1 - i),
                        "corr_disp_pct": 10.0 ** (-i)} for i in range(1, 5)]
        + [{"step": 1, "inc": 2, "att": 1, "iteration": i, "resid_force_pct": 10.0 ** (1 - i), "corr_disp_pct": 1e-3}
           for i in range(1, 4)],
        "increments": [{"step": 1, "inc": 1, "att": 1, "iterations": 4, "tot_time": 0.5},
                       {"step": 1, "inc": 2, "att": 1, "iterations": 3, "tot_time": 1.0}],
        "cutbacks": 0, "status_message": "Analysis converged"}}}}


saved = w.results
w.results = _R()
mon.refresh()
app.processEvents()
check(not mon.isHidden() and "2 increments, 7 iterations" in mon.info.text(), f"monitor shows saved history ({mon.info.text()})")
check(mon.plot.canvas is not None, "convergence chart drawn")
mon.grab().save(str(OUT / "nl_monitor.png"))
w.results = saved
mon.refresh()

# --- polygon imprint dialog
from fea_gui.imprint_dialog import ImprintDialog  # noqa: E402
idlg = ImprintDialog(w)
idlg.combo_shape.setCurrentIndex(idlg.combo_shape.findData("polygon"))
idlg.txt_name.setText("pad")
idlg.sp_cx.setValue(20)
idlg.sp_cy.setValue(5)
idlg.sp_cz.setValue(10)
idlg.txt_points.setPlainText("-5, -3\n5, -3\n5, 3\n-5, 3")
idlg._add_imprint()
imp = next((i for i in w.study.imprints if i.name == "pad"), None)
check(imp is not None and imp.shape == "polygon" and abs(imp.area() - 60.0) < 1e-9, "polygon imprint added (A = 60 mm²)")
idlg.txt_name.setText("bad")
idlg.txt_points.setPlainText("0, 0\n1, 0")
n_warn = len(_warn)
idlg._add_imprint()
check(len(_warn) == n_warn + 1 and not any(i.name == "bad" for i in w.study.imprints), "invalid polygon refused")
idlg.close()

# --- job queue (real solver process)
w.dirty = False
qd = open_queue_dialog(w)
qd.add([proj])
job = qd.q.list_jobs()[0]
check(wait(lambda: job.status not in ("pending", "running"), 600), "queue job finished")
check(job.status == "completed" and job.output.is_file(), f"solved project written ({job.status} {job.error})")
qd.refresh()
check(qd.tab.rowCount() == 1 and qd.tab.item(0, 1).text() in ("done", "hotovo"), "queue table shows the job")
qd.tab.selectRow(0)
app.processEvents()
qd.grab().save(str(OUT / "queue.png"))
qd.open_result()
app.processEvents()
check(w.results is not None and w.project_path == job.output, "result opened from the queue")
qd.close()

w.dirty = False
w.close()
print("FAILED:" if fails else "ALL OK", *fails, sep="\n  ")
sys.exit(1 if fails else 0)
