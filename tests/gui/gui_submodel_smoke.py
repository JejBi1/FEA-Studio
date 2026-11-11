"""A pass through the GUI of the submodel (fea_gui/submodel_dialog.py), with snapshots.

  python tests/gui/hidden_run.py tests/gui/gui_submodel_smoke.py OUT_DIR

Model SM2 (verification/submodel_cases.py): plate with a hole, tension + pressure + hole pressure + gravity. Calculation, in the
results "Submodel…": centre from the maximum outside singularities and by a click, box region in 3D, calculation in a thread,
result item "Submodel: …" in the result menu (quantities, cut check), sphere region, deleting, saving and reopening the .fea.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_submodel_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

FAIL: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "ERROR ") + text)
    if not cond:
        FAIL.append(text)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)   # closing the window: do not save
    import verification.cases  # noqa: F401 – circular import of verification
    from verification.submodel_cases import sm2_build
    from fea_core.project import save_project
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    log = []

    def shot(name, extra=None):
        for _ in range(3):
            app.processEvents()
        w.view.redraw()
        app.processEvents()
        from PySide6.QtGui import QImage, QPainter
        img = w.grab().toImage()
        tmp = OUT / "_view.png"
        w.view.screenshot(str(tmp))
        v = QImage(str(tmp))
        p = QPainter(img)
        p.drawImage(w.view.mapTo(w, w.view.rect().topLeft()), v.scaled(w.view.size()))
        p.end()
        tmp.unlink(missing_ok=True)
        img.save(str(OUT / f"{name}.png"))
        if extra is not None:
            extra.grab().save(str(OUT / f"{name}_dialog.png"))
        log.append(name)

    def click_at(xyz):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        xp, yp = w.view.view.Convert(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        r = w.view.dpr()
        pt = QPoint(int(xp / r), int(yp / r))
        QTest.mouseMove(w.view, pt)
        QTest.mouseClick(w.view, Qt.LeftButton, Qt.NoModifier, pt)
        app.processEvents()

    def s_open():
        wd = OUT / "sm2"
        wd.mkdir(exist_ok=True)
        st, _ = sm2_build(wd)
        save_project(OUT / "sm2.fea", st, st.geometry)
        w.open_file(str(OUT / "sm2.fea"))
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_dialog():
        check(w.results is not None and w.results.steps, "global result")
        w.view.set_orientation("iso")
        w.view.fit_all()
        w.open_submodel_dialog()
        d = w._submodel_dlg
        check(d is not None and d.isVisible(), "submodel dialog")
        d.from_max()
        c = np.array([s.value() for s in d.c])
        from fea_core.mcp.diagnostics import hotspots
        hs = hotspots(w.diag_session(), 0, top_n=8)["hotspots"]
        ref = next(h for h in hs if h.get("classification") != "singularity")["xyz_mm"]
        check(np.allclose(c, ref, atol=1e-3), f"centre = max. stress outside singularities: {np.round(c, 2)}")
        d.start_pick(True)
        click_at((-10.0, 0.0, 10.0))
        c = np.array([s.value() for s in d.c])
        check(w._pick_hook is None and abs(c[2] - 10) < 1.0, f"centre by a click: {np.round(c, 2)}")
        d.set_center((0.0, 0.0, 5.0))
        for s, v in zip(d.sz, (40.0, 40.0, 14.0)):
            s.setValue(v)
        d.h.setValue(2.0)
        check(bool(w.view._marks.get("submodel_region")), "region outline in 3D")
        shot("1_region", d)
        d.run()
        check(d.thread is not None, "submodel analysis in a thread")

    def s_done():
        d = w._submodel_dlg
        subs = w.results.submodels
        check(len(subs) == 1, f"submodel stored in the results: {len(subs)}")
        if not subs:
            return
        sr = subs[0]
        check(sr.summary["max_mises_MPa"] > 100 and sr.summary["equilibrium"]["imbalance"] < 1e-3,
              f"summary: max {sr.summary['max_mises_MPa']:.1f} MPa, imbalance {sr.summary['equilibrium']['imbalance']:.1e}")
        cur = w.current_result()
        check(cur == ("sub", 0), f"result menu shows the submodel: {cur}")
        check("Cut check" in w.res_info.toPlainText() or "Kontrola řezu" in w.res_info.toPlainText(), "cut check in the summary")
        check(d.tab.rowCount() == 1, "table of submodels")
        shot("2_submodel_mises", d)
        names = [w.r_qty.itemText(i) for i in range(w.r_qty.count())]
        k = next(i for i, n in enumerate(names) if n.startswith("Cut check"))
        w.r_qty.setCurrentIndex(k)
        shot("3_cut_check")
        w.r_qty.setCurrentIndex(0)
        d.shape.setCurrentIndex(d.shape.findData("sphere"))
        d.radius.setValue(18.0)
        d.run()

    def s_sphere():
        d = w._submodel_dlg
        check(len(w.results.submodels) == 2 and w.results.submodels[1].region.shape == "sphere", "sphere submodel")
        shot("4_sphere", d)
        d.tab.selectRow(1)
        d.delete_row()
        check(len(w.results.submodels) == 1, "submodel deleted")
        save_project(OUT / "sm2_sub.fea", w.study, w.step_path, w.results)
        d.close()
        w.dirty = False
        w.open_file(str(OUT / "sm2_sub.fea"))
        check(w.results is not None and len(w.results.submodels) == 1, "submodel reopened from the .fea")
        w.set_stage(w.STAGE["Results"])
        w.select_result("sub:0")
        check(w.current_result() == ("sub", 0), "reopened submodel in the result menu")
        shot("5_reopened")

    steps = [s_open, s_dialog, s_done, s_sphere]

    def run_next():
        if w._thread is not None or w._aux_threads:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            print("snapshots:", ", ".join(log))
            app.quit()
            return
        f = steps.pop(0)
        print(f"--- {f.__name__}")
        try:
            f()
        except Exception:
            import traceback
            traceback.print_exc()
            FAIL.append(f.__name__)
            app.quit()
            return
        QTimer.singleShot(600, run_next)

    QTimer.singleShot(800, run_next)
    app.exec()
    print("RESULT:", "OK" if not FAIL else f"{len(FAIL)} errors: " + "; ".join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
