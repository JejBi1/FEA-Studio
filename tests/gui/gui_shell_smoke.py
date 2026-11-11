"""A pass through the GUI with the shell model, with snapshots.

  python tests/gui/gui_shell_smoke.py OUT_DIR

1) Sh1 (verification/shell_cases.py): a plate 200 × 40 × 5 as a body → the Solve step "Shell model" (recognition of the plate,
   thickness 5 mm), calculation, σx of the top / bottom surface = M/W, reactions, mutual exclusion with 2D.
2) Sh7: a plate part from a STEP (a face without volume) → shells automatically, the thickness given in the table, clamping of the edge and a force
   on the edge in the forms, calculation → the deflection as Sh1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_shell_out").resolve()
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
    from fea_gui.main import MainWindow, FORCE
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)   # closing the window: do not save
    import verification.cases  # noqa: F401 – circular import of verification
    from verification.shell_cases import SH1_B, SH1_F, SH1_L, SH1_T, sh1_build, sh7_build
    from fea_core.project import save_project
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    log = []
    sb = w.shell_box

    def shot(name):
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
        log.append(name)

    def qty(name):
        names = [w.r_qty.itemText(i).split(" [")[0] for i in range(w.r_qty.count())]
        check(name in names, f"veličina „{name}“")
        if name in names:
            w.r_qty.setCurrentIndex(names.index(name))
        app.processEvents()

    def s_open():
        wd = OUT / "sh1"
        wd.mkdir(exist_ok=True)
        st, _ = sh1_build(wd)
        save_project(OUT / "sh1.fea", st, st.geometry)
        w.open_file(str(OUT / "sh1.fea"))
        w.set_stage(w.STAGE["Solve"])
        sb.on.setChecked(True)
        check(w.study.shell.enabled, "skořepinový model zapnut")
        t = sb.table
        check(t.rowCount() == 1 and t.item(0, 1).text() == "plate" and t.item(0, 2).text() == "5",
              f"rozpoznaná deska: {[t.item(0, c).text() for c in range(5)] if t.rowCount() else None}")
        # 2D and shells exclude each other
        w.twod_box.on.setChecked(True)
        check(w.study.twod.enabled and not w.study.shell.enabled, "zapnutí 2D vypne skořepiny")
        sb.on.setChecked(True)
        check(w.study.shell.enabled and not w.study.twod.enabled, "zapnutí skořepin vypne 2D")
        errs, warns = w.check()
        check(not errs and any("Shell model" in x for x in warns), f"připraveno: {errs}")
        w.view.set_orientation("iso")
        w.view.fit_all()
        shot("1_skorepiny_nastaveni")
        w.run_solve()

    def s_res():
        res = w.results
        check(res is not None and res.shell is not None and len(res.shell.steps) == 2, "výsledek skořepin (2 stavy)")
        if res is None or res.shell is None:
            return
        items = [w.r_step.itemText(i) for i in range(w.r_step.count())]
        check(items[:1] == ["Shells: Síla 100 N"], f"nabídka: {items}")
        w.r_scale.setCurrentIndex(0)                    # without deformation (the probe in the undeformed position)
        w.select_result("shell:0")
        qty("Stress σx – top surface")
        W = SH1_B * SH1_T ** 2 / 6
        ref = SH1_F * SH1_L / 2 / W
        p = np.array([SH1_L / 2, SH1_B / 2, SH1_T / 2])
        v = w._probe_exact(p) if w._probe_exact else None
        check(v is not None and abs(v - ref) / ref < 0.03, f"σx horní povrch x = L/2: {v} (M/W {ref:.4g})")
        qty("Stress σx – bottom surface")
        v = w._probe_exact(p) if w._probe_exact else None
        check(v is not None and abs(v + ref) / ref < 0.03, f"σx dolní povrch: {v}")
        qty("Bending stress (von Mises)")
        txt = w.res_info.toPlainText()
        check(("Reactions" in txt and "imbalance" in txt) or ("Reakce" in txt and "nerovnováha" in txt), "souhrn: reakce a rovnováha")
        w.on_probe(tuple(p))
        check("Probe" in w.st_info.text(), f"sonda: {w.st_info.text()}")
        shot("2_skorepiny_ohyb")
        w.r_scale.setCurrentIndex(1)
        qty("von Mises – max of surfaces")
        shot("3_skorepiny_vm")

    def s_surface():
        wd = OUT / "sh7"
        wd.mkdir(exist_ok=True)
        sh7_build(wd)
        step = next(wd.glob("model_*.step"))
        w.dirty = False
        w.open_file(str(step))
        check(w.surface_names == {"Plocha"} and w.study.shell.enabled, f"plošný díl ze STEP: {w.surface_names}")
        w.set_stage(w.STAGE["Solve"])
        t = sb.table
        check(t.rowCount() == 1 and t.item(0, 1).text() == "surface part", "plošný díl v tabulce")
        t.setCurrentCell(0, 0)
        sb.t.setValue(SH1_T)
        sb.set_thickness()
        check(w.study.shell.parts.get("Plocha") == SH1_T, f"tloušťka zadána: {w.study.shell.parts}")
        # clamping of the edge x = 0 and a force on the edge x = L
        w.set_stage(w.STAGE["Constraints"])
        w.set_pick_filter("edge")
        e0 = [(0, i) for i in range(len(w.view.part_edges(0))) if abs(w.edge_prop(0, i)["centroid"][0]) < 1e-6]
        w.view.set_selected_edges(e0, emit=True)
        w.add_fixed()
        c = w.study.constraints[-1] if w.study.constraints else None
        check(c is not None and w.study.selections[c.selection].entity == "edges", "vetknutí hrany")
        w.set_stage(w.STAGE["Loads"])
        w.set_pick_filter("edge")
        e1 = [(0, i) for i in range(len(w.view.part_edges(0))) if abs(w.edge_prop(0, i)["centroid"][0] - SH1_L) < 1e-6]
        w.view.set_selected_edges(e1, emit=True)
        w.ld_type.setCurrentText(FORCE)
        for s_, val in zip(w.ld_vec, (0.0, 0.0, -SH1_F)):
            s_.setValue(val)
        w.add_load()
        w.set_stage(w.STAGE["Solve"])
        errs, _ = w.check()
        check(not errs, f"připraveno: {errs}")
        w.view.fit_all()
        shot("4_plosny_dil")
        w.run_solve()

    def s_surface_res():
        res = w.results
        check(res is not None and res.shell is not None, "výsledek plošného dílu")
        if res is None or res.shell is None:
            return
        st = res.shell.step(0)
        tip = np.abs(res.shell.nodes[:, 0] - SH1_L) < 1e-6
        u = -float(st.U[tip, 2].mean())
        I = SH1_B * SH1_T ** 3 / 12
        G = 210000.0 / 2.6
        ref = SH1_F * SH1_L ** 3 / (3 * 210000.0 * I) + SH1_F * SH1_L / (5 / 6 * G * SH1_B * SH1_T)
        check(abs(u - ref) / ref < 0.03, f"průhyb konce {u:.4f} mm (Timoshenko {ref:.4f})")
        check(abs(st.reactions[next(iter(st.reactions))][2] - SH1_F) < 1e-2, f"reakce {st.reactions}")
        w.select_result("shell:0")
        qty("Displacement Z")
        shot("5_plosny_dil_posuv")

    steps = [s_open, s_res, s_surface, s_surface_res]

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
