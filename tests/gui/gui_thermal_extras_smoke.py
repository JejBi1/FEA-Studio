"""A pass through the Heat step with the contact conductance of an interface and the cavity radiation, with snapshots.

  python tests/gui/hidden_run.py tests/gui/gui_thermal_extras_smoke.py OUT_DIR

Model: three steel blocks 10×10 in a row – A (x 0–50) and B (50–100) touch (bonded interface with h_c = 1000 W/(m²·K)),
C (102–112) is separated by a 2 mm gap (closed cavity between the faces x = 100 (ε 0.8) and x = 102 (ε 0.5)).
T(x = 0) = 100 °C, T(x = 112) = 20 °C. The heat has one path in series → the heat through the interface, the heat of both
cavity faces and the heat at x = 0 are the same number (independent outputs: own integration h_c·∫ΔT dA, RFL of the
cavity faces, reaction of the prescribed temperature).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_thermal_extras_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL = []


def check(cond, text):
    print(("OK   " if cond else "ERROR ") + text)
    if not cond:
        FAIL.append(text)


def make_step(path):
    from fea_core.geometry import Part, write_step
    from verification.cases import box
    write_step([Part("A", box(0, 0, 0, 50, 10, 10)), Part("B", box(50, 0, 0, 50, 10, 10)),
                Part("C", box(102, 0, 0, 10, 10, 10))], path)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")
    QMessageBox.warning = staticmethod(lambda parent, title, text, *a: print(f"[{title}] {text}"))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    step = OUT / "blocks.step"
    make_step(step)
    log = []
    tp = w.thermal_panel

    def shot(name):
        app.processEvents()
        w.view.redraw()
        app.processEvents()
        from PySide6.QtGui import QImage, QPainter
        img = w.grab().toImage()
        tmp = OUT / "_view.png"
        w.view.screenshot(str(tmp))
        v = QImage(str(tmp))
        pos = w.view.mapTo(w, w.view.rect().topLeft())
        p = QPainter(img)
        p.drawImage(pos, v.scaled(w.view.size()))
        p.end()
        tmp.unlink(missing_ok=True)
        img.save(str(OUT / f"{name}.png"))
        log.append(name)

    def plane_x(x):
        from fea_core.geometry import face_props
        out = []
        for pi, faces in enumerate(w.view.part_faces):
            for fi, f in enumerate(faces):
                p = face_props(f)
                if abs(p["bbox"][0] - x) < 1e-6 and abs(p["bbox"][3] - x) < 1e-6:
                    out.append((pi, fi))
        return out

    def add_bc(kind, faces, value, **kw):
        w.view.set_selected_faces(faces, emit=True)
        tp.bc_type.setCurrentIndex(tp.bc_type.findData(kind))
        if "cavity" in kw:
            tp.bc_cavity.setCurrentText(kw["cavity"])
            tp.bc_closed.setChecked(kw.get("closed", True))
        tp.bc_val.setValue(value)
        tp.add_bc()

    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")
        w.study.mesh.size = 4.0
        w.set_stage(w.STAGE["Materials"])
        w.mat_combo.setCurrentText("Ocel S235")
        w.assign_material(True)
        check(all(w.study.material_of(p).k for p in ("A", "B", "C")), "material with λ on all parts")

    def s_heat():
        w.set_stage(w.STAGE["Heat"])
        tp.new_case("Heat")
        add_bc("fixed_temperature", plane_x(0.0), 100.0)
        add_bc("fixed_temperature", plane_x(112.0), 20.0)
        check(not tp.bc_cavity.isVisible() and not tp.bc_closed.isVisible(), "cavity fields hidden for other kinds")
        add_bc("cavity_radiation", plane_x(100.0), 0.8, cavity="Gap", closed=True)
        check(tp.bc_cavity.isVisible() and tp.bc_closed.isVisible() and not tp.bc_amb.isVisible(),
              "cavity fields visible, ambient hidden for a closed cavity")
        add_bc("cavity_radiation", plane_x(102.0), 0.5, cavity="Gap", closed=True)
        c = w.study.thermal.case("Heat")
        kinds = [b.kind for b in c.loads]
        check(kinds == ["fixed_temperature", "fixed_temperature", "cavity_radiation", "cavity_radiation"], f"conditions {kinds}")
        cav = [b for b in c.loads if b.kind == "cavity_radiation"]
        check([b.emissivity for b in cav] == [0.8, 0.5] and all(b.closed and b.cavity_name == "Gap" for b in cav),
              "two cavity faces with their own ε, one closed cavity")
        check("cavity “Gap”" in tp.bc_list.item(2).text(), f"list text: {tp.bc_list.item(2).text()}")
        # the form takes the cavity back from the list
        tp.bc_list.setCurrentRow(3)
        check(tp.bc_type.currentData() == "cavity_radiation" and abs(tp.bc_val.value() - 0.5) < 1e-12
              and tp.bc_cavity.currentText() == "Gap" and tp.bc_closed.isChecked(), "form loaded from the cavity condition")
        # interface conductance A–B with the suggestion dialog
        from fea_gui.thermal_extras import SuggestDialog
        dlg = SuggestDialog(tp, w.study, [p.name for p in w.parts], "A", "B")
        est = dlg.estimate()
        check(est > 0 and "W/(m²·K)" in dlg.est.text(), f"estimate of h_c from the materials: {est}")
        dlg._use_row(0)
        check(dlg.value == 1000.0, f"value from the database row: {dlg.value}")
        cb = tp.conductance
        cb.part_a.setCurrentText("A")
        cb.part_b.setCurrentText("B")
        cb.h.setValue(1000.0)
        cb.add()
        th = w.study.thermal
        check(len(th.conductances) == 1 and th.conductances[0].conductance == 1000.0, "interface conductance in the study")
        check(w.study.mesh.separate == ["B"], f"part B meshed separately: {w.study.mesh.separate}")
        check(cb.list.count() == 1 and "h_c = 1" in cb.list.item(0).text(), "interface in the list")
        w.undo()
        n0 = len(w.study.thermal.conductances)
        w.redo()
        check((n0, len(w.study.thermal.conductances)) == (0, 1), "undo / redo of the interface")
        w.set_stage(w.STAGE["Heat"])
        tp.update_info()
        check("share a mesh" not in tp.info.text(), "no readiness error about the shared mesh")
        shot("h1_heat_step")
        # Contacts step: h_c of a contact pair goes into the pair (not added to the study – it would be nonlinear)
        cp = w.contact_panel
        cp.p_a.setEditText("A")
        cp.p_b.setEditText("B")
        cp.p_kind.setCurrentIndex(cp.p_kind.findData("frictionless"))
        cp.p_hc.setValue(2500.0)
        pair = cp.pair_from_form()
        check(pair is not None and pair.conductance == 2500.0, "contact pair form carries h_c")
        cp.p_kind.setCurrentIndex(cp.p_kind.findData("bonded"))
        check(not cp.p_hc.isEnabled() and cp.pair_from_form().conductance is None, "h_c off for a bonded pair")
        cp.p_hc.setValue(0.0)
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_results():
        res = w.results
        check(res is not None and len(res.thermal) == 1, "thermal result")
        if res is None:
            return
        th = res.thermal[0]
        q_in = max(h["Q_W"] for h in th.heat if h["kind"] == "fixed_temperature")
        ifc = th.balance.get("interfaces") or []
        cav = {f["selection"]: f["Q_W"] for f in th.balance.get("cavity_faces", [])}
        check(len(ifc) == 1 and abs(abs(ifc[0]["Q_W"]) / q_in - 1) < 1e-3,
              f"heat through the interface {ifc[0]['Q_W'] if ifc else None} = heat in {q_in}")
        vals = sorted(cav.values())
        check(len(vals) == 2 and abs(-vals[0] / q_in - 1) < 1e-3 and abs(vals[1] / q_in - 1) < 1e-3,
              f"cavity faces give / take the same heat: {cav}")
        check(th.balance.get("imbalance", 1) < 1e-3, "heat balance")
        i = w.r_step.findText("Thermal: Heat")
        if i >= 0:
            w.r_step.setCurrentIndex(i)
        shot("h2_temperature")
        w._save(OUT / "blocks.fea")
        w.dirty = False

    def s_reload():
        w.open_file(str(OUT / "blocks.fea"))
        st = w.study
        check(len(st.thermal.conductances) == 1 and st.mesh.separate == ["B"], "interface after reopening")
        check(w.results is not None and "interfaces" in w.results.thermal[0].balance, "interface heat after reopening")
        w.set_stage(w.STAGE["Heat"])
        shot("h3_reopened")

    steps = [s_open, s_heat, s_results, s_reload]

    def run_next():
        if w._thread is not None:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            print("snapshots:", ", ".join(log))
            print("RESULT:", "OK" if not FAIL else f"{len(FAIL)} ERRORS: {FAIL}")
            app.quit()
            return
        f = steps.pop(0)
        try:
            f()
        except Exception:
            import traceback
            traceback.print_exc()
            FAIL.append(f.__name__)
            app.quit()
            return
        QTimer.singleShot(700, run_next)

    QTimer.singleShot(800, run_next)
    app.exec()
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
