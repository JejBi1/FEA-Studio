"""A pass through the GUI of the voxel detail of printing, with snapshots.

  python tests/gui/gui_voxel_smoke.py OUT_DIR

Model X1 (verification/voxel_cases.py): printed bar 40 × 10 × 10 (PLA, full infill), tension 10 MPa. Calculation, in the results
"Print detail (voxels)…": the centre "lowest safety" and by a click, estimate of the size, calculation in a thread, section through a layer
(slider, field, direction of the beads), table of the summary, box of the region in 3D, the note INDICATIVE.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_voxel_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

FAIL: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "CHYBA ") + text)
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
    from verification.voxel_cases import _bar_study
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
        wd = OUT / "x1"
        wd.mkdir(exist_ok=True)
        st = _bar_study(wd)
        save_project(OUT / "x1.fea", st, st.geometry)
        w.open_file(str(OUT / "x1.fea"))
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_dialog():
        res = w.results
        check(res is not None and "print_safety" in res.steps[0].fields, "výsledek tisku")
        w.view.set_orientation("iso")
        w.view.fit_all()
        w.open_voxel_dialog()
        d = w._voxel_dlg
        check(d is not None and d.isVisible(), "dialog voxelového detailu")
        d.auto_center()
        check(not d.auto.isChecked() and all(np.isfinite([s.value() for s in d.c])), "střed = nejnižší bezpečnost")
        d.start_pick(True)
        click_at((20.0, 5.0, 10.0))
        c = [s.value() for s in d.c]
        check(abs(c[0] - 20) < 3 and abs(c[2] - 10) < 0.5 and w._pick_hook is None, f"střed klikem: {np.round(c, 2)}")
        for s, v in zip(d.c, (20.0, 5.0, 5.0)):
            s.setValue(v)
        e = d.estimate()
        check(e is not None and e["ok"] and e["voxels"] > 1000, f"odhad: {e and {k: e[k] for k in ('voxels', 'dof', 'memory_gb', 'time_s')}}")
        shot("1_odhad", d)
        d.run()
        check(d.thread is not None, "výpočet detailu ve vlákně")

    def s_done():
        d = w._voxel_dlg
        vr = d.vr
        check(vr is not None and vr.steps, "výsledek detailu")
        if vr is None:
            return
        sm = vr.summary["steps"][0]
        check(np.isfinite(sm["min_safety_element_avg"]) and np.isfinite(sm["safety_p1"]) and sm.get("ratio") is not None,
              f"souhrn: průměr prvku {sm['min_safety_element_avg']:.3f}, p1 {sm['safety_p1']:.3f}, poměr {sm.get('ratio')}")
        check(d.tab.rowCount() == 1 and "INDICATIVE" in d.out.text(), "tabulka souhrnu a upozornění")
        check(d.img.pixmap() is not None and not d.img.pixmap().isNull(), "obrázek řezu")
        d.field.setCurrentIndex(d.field.findData("safety"))
        d.slice.setValue(max(d.slice.maximum() - 1, 0))
        check(d.slice_lbl.text().startswith("layer"), f"posuvník vrstvy: {d.slice_lbl.text()}")
        check(bool(w.view._marks.get("voxel_box")), "region box in 3D")
        shot("2_detail_bezpecnost", d)
        d.field.setCurrentIndex(d.field.findData("kind"))
        shot("3_detail_druh", d)
        d.close()

    steps = [s_open, s_dialog, s_done]

    def run_next():
        if w._thread is not None or w._aux_threads:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            print("snímky:", ", ".join(log))
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
    print("VÝSLEDEK:", "OK" if not FAIL else f"{len(FAIL)} chyb: " + "; ".join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
