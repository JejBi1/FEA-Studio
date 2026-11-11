"""A pass through the GUI for bolts and welds with snapshots (a manual check of the look).

  python tests/gui/gui_joints_smoke.py OUT_DIR

Assembly: a lap joint with two bolts ISO 4017 M12 (nut, washers) and next to it a T-joint with two fillet welds a5. It recognises the joints, changes the class
of the bolt, enters the constraints and the loads, calculates, saves the snapshots of the Joints step (the utilisation tables in the bottom bar) and the .fea project, reopens it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_joints_out")
OUT.mkdir(parents=True, exist_ok=True)


def make_step(path):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Vec
    from fea_core.geometry import Part, write_step
    from verification import joint_models as jm
    from verification.cases import cyl, solid
    parts = jm.lap_joint(d=12.0, nbolts=2, pitch=40.0)
    # a second bolt without names (recognised by its shape – a proposal to confirm)
    parts = [Part(f"Body{i}", p.shape) if p.name.endswith(":2") else p for i, p in enumerate(parts)]
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(50.0, 100.0, 0.0))
    for p in jm.t_joint_fillet():
        parts.append(Part(p.name, BRepBuilderAPI_Transform(p.shape, tr, True).Shape()))
    # a free magnet without a connection (the calculation offers to exclude it)
    parts.append(Part("Magnet", solid(cyl((20.0, 20.0, 15.0), (0, 0, 1), 5.0, 4.0)), (0.35, 0.35, 0.35)))
    write_step(parts, path)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")   # without modal windows
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    step = OUT / "spoje.step"
    make_step(step)
    log = []

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

    def faces_where(pred):
        from fea_core.geometry import face_props
        out = []
        for pi, faces in enumerate(w.view.part_faces):
            for fi, f in enumerate(faces):
                if pred(w.parts[pi].name, face_props(f)):
                    out.append((pi, fi))
        return out

    def plane_x(part, x):
        return faces_where(lambda n, p: n == part and abs(p["bbox"][0] - x) < 1e-6 and abs(p["bbox"][3] - x) < 1e-6)

    def plane_z(part, z):
        return faces_where(lambda n, p: n == part and abs(p["bbox"][2] - z) < 1e-6 and abs(p["bbox"][5] - z) < 1e-6)

    jp = w.joints_panel

    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")

    import fea_gui.joints_panel as jpm
    from PySide6.QtWidgets import QMessageBox

    def fake_exec(dlg):
        dlg.show()
        app.processEvents()
        dlg.grab().save(str(OUT / "j0_potvrzeni.png"))
        print("návrhy:", [(s.source, round(s.confidence, 2), s.label) for s in dlg.sugg])
        dlg.close()
        return True
    jpm.exec_dialog = fake_exec
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

    def s_detect():
        w.set_stage(w.STAGE["Joints"])
        jp.detect()
        st = w.study
        print("rozpoznáno:", [b.designation for b in st.bolts], [x.part for x in st.welds])
        assert len(st.bolts) == 2 and len(st.welds) == 2, "čekány 2 šrouby a 2 svary"
        assert any("shape" in b.note for b in st.bolts), "druhý šroub má být podle tvaru"
        jp.bolt_list.setCurrentRow(1)
        jp.b_class.setCurrentText("10.9")
        jp.save_bolt()
        print("třída šroubu 2:", st.bolts[1].strength)
        assert st.bolts[1].strength == "10.9"
        jp.bolt_list.setCurrentRow(0)
        shot("j1_spoje_rozpoznano")

    def s_bc():
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces(plane_x("Deska B", 200.0) + plane_z("Pasnice", -10.0), emit=True)
        w.add_fixed()
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces(plane_x("Deska A", 0.0), emit=True)
        w.ld_type.setCurrentText("Force (components)")
        for s, v in zip(w.ld_vec, (-20000.0, 0.0, 0.0)):
            s.setValue(v)
        w.add_load()
        w.view.set_selected_faces(plane_z("Stojina", 100.0), emit=True)
        for s, v in zip(w.ld_vec, (0.0, 0.0, 30000.0)):
            s.setValue(v)
        w.add_load()
        w.set_stage(w.STAGE["Solve"])
        shot("j2_reseni")
        w.run_solve()                     # fails: a free magnet → the offer to exclude (confirmed)

    def s_rerun():
        print("vynechané díly:", w.study.ignored_parts)
        assert w.study.ignored_parts == ["Magnet"], "magnet měl být vynechán"
        w.set_stage(w.STAGE["Joints"])
        shot("j2b_vynechany_magnet")
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_results():
        print("kroky výsledků:", [s.name for s in w.results.steps])
        w.set_stage(w.STAGE["Joints"])
        a = jp.assessment
        for c in a.all():
            print(f"  {c.name}: využití {c.utilization:.2f} ({c.governing})")
        jp.bolt_list.setCurrentRow(0)
        shot("j3_spoje_posouzeni")
        jp.weld_list.setCurrentRow(0)
        w.dock_joints.raise_()
        jp.tabs.setCurrentIndex(1)
        shot("j4_svary")
        w.set_stage(w.STAGE["Results"])
        w.r_step.setCurrentIndex(len(w.results.steps) - 1)      # the state "Bolt preload"
        shot("j5_predpeti")
        w._save(OUT / "spoje.fea")
        w.dirty = False

    def s_reload():
        w.open_file(str(OUT / "spoje.fea"))
        st = w.study
        print("po otevření:", len(st.bolts), "šroubů,", len(st.welds), "svarů; třída 2:", st.bolts[1].strength)
        w.set_stage(w.STAGE["Joints"])
        print("posouzení po otevření:", "yes" if jp.assessment else "ne")
        shot("j6_znovu_otevreno")

    steps = [s_open, s_detect, s_bc, s_rerun, s_results, s_reload]

    def run_next():
        if w._thread is not None:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            print("snímky:", ", ".join(log))
            app.quit()
            return
        f = steps.pop(0)
        try:
            f()
        except Exception:
            import traceback
            traceback.print_exc()
            app.quit()
            return
        QTimer.singleShot(700, run_next)

    QTimer.singleShot(800, run_next)
    app.exec()


if __name__ == "__main__":
    main()
