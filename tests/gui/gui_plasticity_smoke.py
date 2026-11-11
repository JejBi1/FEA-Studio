"""Plasticity through the GUI: Material properties → Plasticity (bilinear, table paste), unloading step, nonlinear solve with
"Stop and keep results" available, PEEQ quantity, Plasticity assessment dialog (EN 1993 / ASME, residual stress).

  python tests/gui/hidden_run.py tests/gui/gui_plasticity_smoke.py OUT_DIR

Model: bar 10×10×100 (geometry of PL1), clamped at z = 0, force 30 kN (σ = 300 MPa) on z = 100, steel E 200 GPa, Re 250,
Et 20 GPa → PEEQ = (σ − Re)/H, H = E·Et/(E − Et) = 22 222 MPa → 0.00225 in the middle of the bar; unloading = elastic,
residual stress ≈ 0 away from the clamp.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_plasticity_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")

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
    from fea_gui.main import MainWindow, FORCE, QUANTITIES
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    step = OUT / "tyc.step"
    from fea_core.geometry import Part, write_step
    from verification.cases import box
    write_step([Part("Tyc", box(0, 0, 0, 10, 10, 100))], step)
    log = []
    H = 200000.0 * 20000.0 / 180000.0

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

    def faces_z(z):
        from fea_core.geometry import face_props
        out = []
        for pi, faces in enumerate(w.view.part_faces):
            for fi, f in enumerate(faces):
                bb = face_props(f)["bbox"]
                if abs(bb[2] - z) < 1e-6 and abs(bb[5] - z) < 1e-6:
                    out.append((pi, fi))
        return out

    def s_open():
        w.open_file(str(step))
        w.ms_size.setValue(5.0)

    def s_material():
        w.set_stage(w.STAGE["Materials"])
        w.open_material_dialog()
        w.m_name.setText("Ocel plast")
        w.m_E.setValue(200000.0)
        w.m_nu.setValue(0.3)
        w.m_Re.setValue(250.0)
        w.m_Rm.setValue(400.0)
        pb = w.plasticity_box
        pb.setChecked(True)
        check(pb.is_open() and pb.inner.isVisible(), "plasticity section opens with the tick")
        pb.model.setCurrentIndex(pb.model.findData("multilinear"))
        pb.paste_text("250\t0\n280\t0.01\n300\t0.03")
        check(pb.table_rows() == [(250.0, 0.0), (280.0, 0.01), (300.0, 0.03)], "table pasted")
        check("complete" in pb.status.text(), f"table valid: {pb.status.text()[:120]}")
        shot("1_plasticity_table", w.mat_dialog)
        pb.model.setCurrentIndex(pb.model.findData("bilinear"))
        pb.bi_kind.setCurrentIndex(pb.bi_kind.findData("Et"))
        pb.bi_val.setValue(20000.0)
        pb.hard.setCurrentIndex(pb.hard.findData("isotropic"))
        check(pb.plot.canvas is not None, "curve drawn")
        w.assign_material(True)
        m = w.study.materials.get("Ocel plast")
        check(m is not None and m.is_plastic and m.plastic["model"] == "bilinear" and m.plastic["Et"] == 20000.0,
              f"Material.plastic stored: {m.plastic if m else None}")
        check("elastoplastic" in w.mat_summary.text(), "summary mentions plasticity")
        check(w.study.is_nonlinear(), "study is nonlinear")
        w.save_user_material()
        from fea_core import materials as matlib
        lib = {x.name: x for x in matlib.user_library()}
        check("Ocel plast" in lib and lib["Ocel plast"].plastic == m.plastic, "user library keeps plasticity")
        w.mat_combo.setCurrentText("Ocel plast")
        check(pb.isChecked() and pb.values() == m.plastic, "form reloads the plasticity")
        shot("2_plasticity_bilinear", w.mat_dialog)
        w.mat_dialog.close()

    def s_bc():
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces(faces_z(0.0), emit=True)
        w.add_fixed()
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces(faces_z(100.0), emit=True)
        w.ld_type.setCurrentText(FORCE)
        for s_, v in zip(w.ld_vec, (0.0, 0.0, 30000.0)):
            s_.setValue(v)
        w.add_load()
        check(len(w.study.load_cases[0].loads) == 1, "force 30 kN")
        w.plasticity_box.add_unloading()
        seq = w.study.analysis.sequence
        check(len(seq) == 2 and seq[-1].scale == 0.0 and not w.study.analysis.branches, f"load history: {seq}")

    def s_solve():
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()
        check(w._thread is not None and w.btn_stop_keep.isEnabled(), "nonlinear run: Stop and keep results enabled")
        shot("3_solving")

    def s_results():
        res = w.results
        check(res is not None and len(res.steps) == 2, f"two steps: {None if res is None else [s.name for s in res.steps]}")
        if res is None or len(res.steps) < 2:
            return
        X = res.mesh.nodes
        mid = np.nonzero((X[:, 2] > 30) & (X[:, 2] < 70))[0]
        pe = float(np.mean(res.steps[0].fields["peeq"][mid]))
        ref = (300.0 - 250.0) / H
        check(abs(pe - ref) / ref < 0.03, f"PEEQ middle {pe:.5g} vs {ref:.5g}")
        names = [q[0] for q in QUANTITIES]
        w.select_result("step:0")
        w.r_qty.setCurrentIndex(names.index("Equivalent plastic strain PEEQ"))
        check(w.legend.isVisible(), "PEEQ map shown")
        shot("4_peeq")
        from fea_gui.plasticity_dialog import PlasticityDialog
        dlg = PlasticityDialog(w)
        dlg.show()
        app.processEvents()
        check(dlg.steps.rowCount() == 2, "assessment rows per step")
        en = dlg.rows[0]["en1993"]
        check(abs(en["utilization"] - en["max_peeq"] / 0.05) < 1e-9 and en["ok"], f"EN 1993 utilization {en['utilization']:.4g}")
        check(dlg.rows[0]["asme"] is not None and dlg.rows[0]["asme"]["utilization"] > 0, "ASME utilization")
        resid = dlg.rows[1]["en1993"].get("residual")
        vm_mid = float(np.max(res.steps[1].mises[mid]))
        check(resid is not None and vm_mid < 5.0, f"residual stress after unloading: max {resid and resid['max_mises_MPa']:.3g},"
                                                   f" middle {vm_mid:.3g} MPa")
        dlg.cls.setCurrentIndex(dlg.cls.findData("austenitic"))
        am = dlg.rows[0]["asme"]["materials"].get("Ocel plast", {})
        check(dlg.mats.rowCount() == 1 and am.get("alpha_sl") == 0.6 and abs(am.get("m2", 0) - 0.75 * (1 - 250 / 400)) < 1e-12,
              f"class change → austenitic constants {am}, cell {dlg.mats.item(0, 5).text() if dlg.mats.rowCount() else None}")
        shot("5_assessment", dlg)
        dlg.show_step(1)
        check(w.r_step.currentData() == "step:1", "show unloading step")
        dlg.close()

    steps = [s_open, s_material, s_bc, s_solve, s_results]

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
