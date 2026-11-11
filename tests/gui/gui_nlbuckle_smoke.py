"""GUI pass through the nonlinear buckling with imperfection (F9 C2): settings in the Solve step, analysis, curve and shapes.

  python tests/gui/hidden_run.py tests/gui/gui_nlbuckle_smoke.py OUT_DIR

Model: cantilever 200×20×10 (steel), clamping x = 0, compression 1 kN on the face x = 200; imperfection = 1st buckling mode,
e0 = 0.2 mm, λ max = 15 (below λ_cr ≈ 21.6).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_nlbuckle_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")

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
    from fea_core.geometry import Part, face_props, write_step
    from verification.cases import box
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    step = OUT / "konzola.step"
    write_step([Part("Nosnik", box(0, 0, 0, 200, 20, 10))], step)

    def shot(name):
        for _ in range(3):
            app.processEvents()
        w.grab().save(str(OUT / f"{name}.png"))

    def faces_at_x(x):
        return [(pi, fi) for pi, faces in enumerate(w.view.part_faces) for fi, f in enumerate(faces)
                if abs(face_props(f)["bbox"][0] - x) < 1e-6 and abs(face_props(f)["bbox"][3] - x) < 1e-6]

    def s_setup():
        w.open_file(str(step))
        w.ms_size.setValue(5.0)
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces(faces_at_x(0.0), emit=True)
        w.add_fixed()
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces(faces_at_x(200.0), emit=True)
        w.ld_type.setCurrentText(FORCE)
        for s_, v in zip(w.ld_vec, (-1000.0, 0.0, 0.0)):
            s_.setValue(v)
        w.add_load()
        check(len(w.study.load_cases[0].loads) == 1, "tlak na čele")

    def s_settings():
        w.set_stage(w.STAGE["Solve"])
        eb = w.eigen_box
        eb.n_on.setChecked(True)
        check(eb.n_amp_kind.isEnabled() and eb.n_note.isVisible(), "skupina nelineárního vzpěru zapnuta")
        eb.n_amp_kind.setCurrentIndex(eb.n_amp_kind.findData("mm"))
        eb.n_amp.setValue(0.2)
        eb.n_max.setValue(15.0)
        eb.n_inc.setValue(6)
        eb.n_modes.setText("1")
        eb.n_modes.editingFinished.emit()
        s = w.study.nlbuckling
        check(s.enabled and s.amplitude == 0.2 and s.max_factor == 15.0 and s.increments == 6 and s.modes == [1],
              f"nastavení ve studii: {s}")
        check(any("responsibility" in m["text"] for m in w.messages.messages(source="study check")),
              "varování o amplitudě imperfekce v kontrole")
        w.eigen_box.n_on.setChecked(False)
        check(not w.study.nlbuckling.enabled, "vypnutí")
        w.undo()
        check(w.study.nlbuckling.enabled and w.eigen_box.n_on.isChecked(), "zpět vrátí zapnutí")
        shot("1_nastaveni")
        w.run_solve()

    def s_results():
        res = w.results
        check(res is not None, "výsledky")
        if res is None:
            return
        kinds = [m.kind for m in res.modes]
        check("nlbuckle" in kinds, f"sady tvarů {kinds}")
        info = res.info.get("nlbuckling") or {}
        r = info.get("result") or {}
        check(r.get("status") == "max_factor" and abs(r.get("ultimate_factor", 0) - 15.0) < 1e-6, f"výsledek {r}")
        items = [w.r_step.itemData(i) for i in range(w.r_step.count())]
        j = kinds.index("nlbuckle")
        last = len(res.modes[j].values) - 1
        check(f"mode:{j}:0" in items and f"mode:{j}:{last}" in items, "položky nelineárního vzpěru v nabídce")
        w.select_result(f"mode:{j}:{last}")
        app.processEvents()
        check(w.r_qty.itemText(0).startswith("Nonlinear buckling"), f"veličina {w.r_qty.itemText(0)}")
        check("λ" in w.res_info.toPlainText() and "Ultimate" in w.res_info.toPlainText(), "souhrn se součinitelem")
        w.res_tabs.setCurrentIndex(2)
        app.processEvents()
        mt = w.modes_table
        check(mt.nl_label.isVisible() and mt.nl_plot.isVisible() and mt.nl_plot.canvas is not None, "křivka a λ_u v záložce")
        shot("2_vysledky_krivka")
        w.select_result(f"mode:{j}:0")
        app.processEvents()
        check("Imperfection" in w.res_info.toPlainText(), "imperfekce zobrazena")
        shot("3_imperfekce")

    steps = [s_setup, s_settings, s_results]

    def run_next():
        if w._thread is not None or w._aux_threads:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
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
