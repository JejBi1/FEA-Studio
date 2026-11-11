"""A pass through the GUI for 3D printing with snapshots (a manual check of the look).

  python tests/gui/gui_print_smoke.py OUT_DIR

Model: printed bracket 80×20×12 with a hole Ø6 (PLA Bambu, gyroid 20 %), clamping at x = 0, force 100 N
at the end. It assigns the printed material and the print settings, shows the shell / core in the section of the mesh, calculates,
shows the safety Tsai-Wu, compares the variants of the internal structure (the same mesh, taking over of a variant), tries the
calibration dialog, saves and reopens the project.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_print_out")
OUT.mkdir(parents=True, exist_ok=True)


def make_step(path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from fea_core.geometry import Part, write_step
    from verification.cases import box, cyl
    shape = BRepAlgoAPI_Cut(box(0, 0, 0, 80, 20, 12), cyl((60, 10, -1), (0, 0, 1), 3.0, 14)).Shape()
    write_step([Part("Držák tisk", shape)], path)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow, QUANTITIES
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")   # without modal windows
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    step = OUT / "drzak.step"
    make_step(step)
    log = []

    def shot(name, widget=None):
        app.processEvents()
        w.view.redraw()
        app.processEvents()
        from PySide6.QtGui import QImage, QPainter
        if widget is not None:
            widget.grab().save(str(OUT / f"{name}.png"))
            log.append(name)
            return
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
                if pred(face_props(f)):
                    out.append((pi, fi))
        return out

    def plane_x(x):
        return faces_where(lambda p: abs(p["bbox"][0] - x) < 1e-6 and abs(p["bbox"][3] - x) < 1e-6)

    pp = w.print_panel

    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")
        w.study.mesh.size = 1.5

    def s_print():
        w.set_stage(w.STAGE["3D printing"])
        pp.parts.item(0).setSelected(True)
        pp.mat.setCurrentText("Tisk PLA Bambu Basic")
        pp.assign_material()
        pp.parts.item(0).setSelected(True)
        pp.s_infill.setValue(20.0)
        pp.s_pattern.setCurrentIndex(pp.s_pattern.findData("gyroid"))
        pp.apply_spec()
        st = w.study
        sp = st.printing.parts["Držák tisk"]
        print("materiál:", st.material_of("Držák tisk").name, "| tisk:", sp.infill, sp.pattern, sp.build_dir)
        assert st.material_of("Držák tisk").is_print and sp.pattern == "gyroid" and abs(sp.infill - 0.2) < 1e-9
        pp.parts.item(0).setSelected(True)
        shot("p1_nastaveni_tisku")

    def s_bc():
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces(plane_x(0.0), emit=True)
        w.add_fixed()
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces(plane_x(80.0), emit=True)
        w.ld_type.setCurrentText("Force (components)")
        for s, v in zip(w.ld_vec, (0.0, 0.0, -100.0)):
            s.setValue(v)
        w.add_load()
        w.set_stage(w.STAGE["Mesh"])
        w.make_mesh()

    def s_shell():
        w.set_stage(w.STAGE["3D printing"])
        pp.cut_axis.setCurrentIndex(0)          # section X through the centre of the hole
        pp.cut_pos.setValue(750)
        pp.show_shell()
        w.view.set_orientation("iso")
        shot("p2_skorepina_jadro_rez")
        # creep: the first state as the long-term one, 1000 h at 30 °C
        pp.cr_on.setChecked(True)
        pp.cr_hours.setValue(1000.0)
        pp.cr_T.setValue(30.0)
        pp.cr_cases.item(0).setCheckState(Qt.Checked)
        pp.on_creep()
        cr = w.study.printing.creep
        print("tečení:", cr.enabled, cr.hours, cr.temperature, cr.cases)
        assert cr.enabled and cr.cases == [w.study.load_cases[0].name] and cr.temperature == 30.0
        w.set_stage(w.STAGE["Solve"])
        shot("p3_reseni")
        w.run_solve()

    def s_results():
        w.set_stage(w.STAGE["Results"])
        names = [q[0] for q in QUANTITIES]
        w.r_qty.setCurrentIndex(names.index("3D printing – safety factor (Tsai-Wu/Hill)"))
        s = w.results.step(0)
        d = s.printing["parts"]["Držák tisk"]
        print(f"bezpečnost {s.printing['criterion']}: {d['min_safety']:.3f} ({d['region']}), "
              f"max. posuv {s.umag.max():.4f} mm")
        shot("p4_bezpecnost_tisku")
        cs = w.results.info["printing"]["creep"]["cases"][0]
        print(f"tečení: posuv {cs['u_short']:.4f} → {cs['u_long']:.4f} mm (×{cs['ratio']:.3f}), "
              f"kroky: {[st.name for st in w.results.steps]}")
        assert len(w.results.steps) == 2 and cs["ratio"] > 1.2
        w.set_stage(w.STAGE["3D printing"])
        pp.update_info()
        from PySide6.QtWidgets import QScrollArea
        shot("p8_teceni_panel", pp.findChild(QScrollArea).widget())    # the whole content of the panel (also below the slider)
        w.set_stage(w.STAGE["Results"])
        w.r_step.setCurrentIndex(1)
        w.r_qty.setCurrentIndex(names.index("Total displacement"))
        shot("p9_teceni_posuv")
        w.r_qty.setCurrentIndex(0)
        shot("p5_von_mises")
        w._save(OUT / "drzak.fea")
        w.dirty = False

    def s_variants():
        w.set_stage(w.STAGE["3D printing"])
        vb = pp.variants_box
        pp.parts.item(0).setSelected(True)
        mesh_before = w.mesh
        from PySide6.QtWidgets import QInputDialog
        names = iter(["4 perimetry, mřížka 40 %", "2 perimetry, gyroid 10 %"])
        QInputDialog.getText = staticmethod(lambda *a, **k: (next(names), True))    # without a modal window
        pp.s_walls.setValue(4)
        pp.s_infill.setValue(40.0)
        pp.s_pattern.setCurrentIndex(pp.s_pattern.findData("grid"))
        vb.add()
        pp.s_walls.setValue(2)
        pp.s_infill.setValue(10.0)
        pp.s_pattern.setCurrentIndex(pp.s_pattern.findData("gyroid"))
        vb.add()
        print("varianty:", [(v.name, v.parts) for v in w.study.printing.variants])
        assert len(w.study.printing.variants) == 2 and w.results is not None   # the definition does not discard the results
        assert w.valid_mesh() is mesh_before
        vb.k_req.setValue(2.0)
        vb.on_k_req()
        vb.compute()

    def s_variants_done():
        vb = pp.variants_box
        run = vb.valid_run()
        assert run is not None, vb.status.text()
        for o in run.outcomes:
            print(f"  {o.name}: k = {o.min_safety}, {o.mass_g} g, posuv {o.max_disp} mm, ok={o.ok} {o.error}")
        print("doporučeno:", run.recommended, "|", run.reason, "| síť:", run.mesh_info)
        assert run.mesh_info["meshed"] is False and all(o.ok for o in run.outcomes)
        m = {o.name: o for o in run.outcomes}
        assert m["4 perimetry, mřížka 40 %"].mass_g > m["2 perimetry, gyroid 10 %"].mass_g
        assert m["4 perimetry, mřížka 40 %"].max_disp < m["2 perimetry, gyroid 10 %"].max_disp
        from PySide6.QtWidgets import QScrollArea
        vb.rtable.selectRow(1)
        vb.set_open(True)                  # the Variants section is collapsed by default
        shot("p10_varianty", vb)
        vb.adopt()
        sp = w.study.printing.parts["Držák tisk"]
        print("převzato:", sp.walls, sp.infill, sp.pattern, "| výsledky:", w.results is m["4 perimetry, mřížka 40 %"].results)
        assert sp.walls == 4 and abs(sp.infill - 0.4) < 1e-9 and w.results is m["4 perimetry, mřížka 40 %"].results
        assert vb.valid_run() is not None
        w.dirty = False

    def s_calib():
        from fea_gui.print_panel import CalibrationDialog
        from fea_core.printing.calibrate import TensileTest
        dlg = CalibrationDialog(w, w.study, pp.material_from_ui(), pp.spec_from_ui(""))
        dlg.add_row(TensileTest("XY-1", "xy", E=2950, Rm=41, infill=1.0))
        dlg.add_row(TensileTest("XY-2", "xy", E=3010, Rm=43, infill=1.0))
        dlg.add_row(TensileTest("Z-1", "z", E=2400, Rm=24, infill=1.0))
        ok = dlg.compute()
        m = dlg.result_material
        print("kalibrace:", ok, m.E, m.E_z, m.Rm, m.Rm_z)
        # the perimeters of the XY body are loaded along the line (E_L = 2/(1 + r)·E_xy) → the full material E_xy < the measured average
        from fea_core.printing.calibrate import factors
        from fea_core.printing.props import default_props
        pr = default_props(pp.material_from_ui().name)
        e_xy = sum(E / factors(t, pr.line_E_ratio, pr.line_R_ratio)[0]
                   for E, t in ((2950, dlg.tests()[0]), (3010, dlg.tests()[1]))) / 2
        assert ok and abs(m.E - e_xy) < 1e-6 and m.E < 2980 and abs(m.Rm_z - 24) < 1e-6
        dlg.resize(1100, 640)
        dlg.show()
        shot("p6_kalibrace", dlg)
        dlg.close()

    def s_reload():
        w.open_file(str(OUT / "drzak.fea"))
        st = w.study
        sp = st.printing.parts["Držák tisk"]
        print("po otevření:", st.material_of("Držák tisk").name, sp.pattern, sp.infill,
              "| výsledky:", "yes" if w.results is not None else "ne",
              "| pole:", list(w.results.step(0).fields) if w.results else None)
        assert sp.pattern == "gyroid" and w.results is not None and "print_safety" in w.results.step(0).fields
        w.set_stage(w.STAGE["3D printing"])
        shot("p7_znovu_otevreno")

    def s_iso_pla():
        # isotropic "PLA tisk XY (orientačně)" (without E_z) – the print results will not be there: a warning before the calculation
        from fea_core.materials import LIBRARY
        from fea_core.solve import check_printing
        m = next(x for x in LIBRARY if x.name == "PLA tisk XY (orientačně)")
        w.study.materials[m.name] = m
        w.study.part_materials["Držák tisk"] = m.name
        w.study.printing.parts.pop("Držák tisk", None)
        w.study.printing.creep.enabled = False
        w.changed()
        _, warn = check_printing(w.study, ["Držák tisk"])
        print("varování izotropní PLA:", warn)
        assert any("without print properties" in t for t in warn)
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_iso_results():
        # a print quantity without a printed part → the model visible, the explanation in the Summary (not an empty view)
        w.set_stage(w.STAGE["Results"])
        w.res_tabs.setCurrentIndex(1)
        names = [q[0] for q in QUANTITIES]
        w.r_qty.setCurrentIndex(names.index("3D printing – safety factor (Tsai-Wu/Hill)"))
        w.show_result()
        txt = w.res_info.toPlainText()
        print("bez tisku:", txt[:90], "| záložka", w.res_tabs.currentIndex(), "| díly vidět", w.view.parts_visible)
        assert any(x in txt for x in ("printed material", "tištěný materiál")) and w.res_tabs.currentIndex() == 0 and w.view.parts_visible
        shot("p11_bez_tistenoho_materialu")

    steps = [s_open, s_print, s_bc, s_shell, s_results, s_variants, s_variants_done, s_calib, s_reload,
             s_iso_pla, s_iso_results]

    def run_next():
        if w._thread is not None:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            print("snímky:", ", ".join(log))
            w.dirty = False          # closing the window (app.quit) would otherwise wait for the modal "Save changes?"
            app.quit()
            return
        f = steps.pop(0)
        try:
            f()
        except Exception:
            import traceback
            traceback.print_exc()
            w.dirty = False
            app.quit()
            return
        QTimer.singleShot(700, run_next)

    QTimer.singleShot(800, run_next)
    app.exec()


if __name__ == "__main__":
    main()
