"""A pass through the GUI with screenshots (a manual check of the look).

  python tests/gui/gui_smoke.py OUT_DIR [STEP]

Without STEP it creates a plate with a hole (V3, whole), clamps the left face, puts a force on the right one,
creates the mesh, calculates and saves the snapshots of the individual steps + the .fea project.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_out")
OUT.mkdir(parents=True, exist_ok=True)


def make_step(path):
    from verification.cases import box, cyl, solid
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from fea_core.geometry import Part, write_step
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 200, 60, 8), cyl((100, 30, -1), (0, 0, 1), 10, 10)).Shape())
    rib = box(0, 0, 8, 20, 60, 30)
    write_step([Part("Plate", plate), Part("Base plate", rib)], path)


def main():
    from fea_gui.main import MainWindow
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 950)
    w.show()
    step = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "deska.step"
    if len(sys.argv) <= 2:
        make_step(step)
    log = []

    def shot(name):
        app.processEvents()
        w.view.redraw()
        app.processEvents()
        # an OpenGL window is not captured by w.grab(), and a screenshot would capture other windows →
        # the 3D view is rendered by OCCT (Dump) and inserted into the snapshot of the widgets
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
        out = []
        for pi, faces in enumerate(w.view.part_faces):
            from fea_core.geometry import face_props
            for fi, f in enumerate(faces):
                if pred(w.parts[pi].name, face_props(f)):
                    out.append((pi, fi))
        return out

    steps = []

    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")

    def s_material():
        w.set_stage(w.STAGE["Materials"])
        w.mat_parts.selectAll()
        shot("1_materialy")
        # the designation converter: ČSN 12050 = C45 → picks "Ocel C45 +N"; ČSN 11523 = S355; 16MnCr5 is not in the library
        w.mat_find.setText("ČSN 12050")
        print("převodník 12050 →", w.mat_combo.currentText())
        shot("1b_prevodnik")
        w.mat_find.setText("16MnCr5")
        print("převodník 16MnCr5:", w.mat_conv.text().split("<br>")[-2][:120])
        w.mat_find.setText("")
        w.mat_combo.setCurrentText("Ocel S235")

    def s_constraint():
        w.set_stage(w.STAGE["Constraints"])
        sel = faces_where(lambda part, p: part == "Base plate" and abs(p["bbox"][0]) < 1e-6 and abs(p["bbox"][3]) < 1e-6)
        w.view.set_selected_faces(sel, emit=True)
        w.add_fixed()
        shot("2_vazby")

    def click_at(xyz, ctrl=False):
        """A real mouse click on the projection of a point of the model (test of the face selection)."""
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        xp, yp = w.view.view.Convert(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        r = w.view.dpr()
        pt = QPoint(int(xp / r), int(yp / r))
        QTest.mouseMove(w.view, pt)
        QTest.mouseClick(w.view, Qt.LeftButton, Qt.ControlModifier if ctrl else Qt.NoModifier, pt)
        app.processEvents()

    def s_load():
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces([], emit=True)
        click_at((200, 30, 4))           # the face of the plate at the end
        sel = faces_where(lambda part, p: part == "Plate" and abs(p["bbox"][0] - 200) < 1e-6 and abs(p["bbox"][3] - 200) < 1e-6)
        got = w.view.selected_faces()
        print("klik myší vybral:", got, "očekáváno:", sel, "OK" if got == sel else "ERROR")
        click_at((200, 30, 4))           # the second click removes the face
        print("po druhém kliku:", w.view.selected_faces())
        w.view.set_selected_faces(sel, emit=True)
        w.ld_type.setCurrentText("Force (components)")
        w.ld_vec[2].setValue(-400.0)
        w.add_load()
        n1 = len(w.study.load_cases[0].loads)
        w.undo()
        n0 = len(w.study.load_cases[0].loads)
        w.redo()
        print("zpět/znovu: zatížení", n1, "->", n0, "->", len(w.study.load_cases[0].loads),
              "OK" if (n1, n0, len(w.study.load_cases[0].loads)) == (1, 0, 1) else "ERROR")
        w.set_stage(w.STAGE["Loads"])
        w.load_list.setCurrentRow(0)
        w.ld_vec[2].setValue(-500.0)
        w.update_load()
        print("úprava zatížení:", w.study.load_cases[0].loads[0].vector)
        hole = faces_where(lambda part, p: p["type"] == "cylinder")
        w.view.set_selected_faces(hole[:1], emit=True)
        w.select_holes()
        print("celá díra:", len(hole[:1]), "->", len(w.view.selected_faces()), "ploch (válců celkem", len(hole), ")")
        sel = faces_where(lambda part, p: p["type"] == "cylinder")
        w.view.set_selected_faces(sel, emit=True)
        shot("3_zatizeni")
        # cylindrical constraint and bearing force on the hole (forms, markers in 3D), then back (Ctrl+Z)
        n_undo = w._undo_pos
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces(sel, emit=True)
        for c, on in zip(w.cyl_chk, (True, True, False)):
            c.setChecked(on)
        w.add_cylindrical()
        print("válcová vazba:", w.con_list.item(w.con_list.count() - 1).text())
        shot("2b_valcova_vazba")
        # remote constraint (hinge 50 mm beyond the face of the plate, distributing): the point from the centroid of the selection, then moving the point
        end = faces_where(lambda part, p: part == "Plate" and abs(p["bbox"][0] - 200) < 1e-6
                          and abs(p["bbox"][3] - 200) < 1e-6)
        w.view.set_selected_faces(end, emit=True)
        print("vzdálená vazba sbalená:", "OK" if not w.rs_group.isChecked() and not w.rs_pt[0].isVisible() else "ERROR")
        w.rs_group.setChecked(True)
        w.remote_point_from_selection()
        print("těžiště výběru:", [round(s.value(), 3) for s in w.rs_pt])
        w.rs_pt[0].setValue(250.0)
        for c, on in zip(w.rs_chk, (True, True, True, False, False, False)):
            c.setChecked(on)
        w.rs_coupling.setCurrentIndex(1)
        w.add_remote_support()
        rs = w.study.constraints[-1]
        print("vzdálená vazba:", w.con_list.item(w.con_list.count() - 1).text(),
              "OK" if (rs.kind, rs.coupling, rs.dofs[:4], rs.point[0]) == ("remote_support", "distributing",
                                                                          [True, True, True, False], 250.0) else "ERROR")
        shot("2c_vzdalena_vazba")
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces(sel, emit=True)
        w.ld_type.setCurrentText("Bearing load (hole / pin)")
        for s, v in zip(w.ld_vec, (-1000.0, 0.0, 0.0)):
            s.setValue(v)
        w.add_load()
        print("ložisková síla:", w.load_list.item(w.load_list.count() - 1).text())
        w.view.set_selected_faces([], emit=True)
        w.view.set_orientation("top")
        shot("3b_loziskova_sila")
        w.view.set_orientation("iso")
        while w._undo_pos > n_undo:
            w.undo()
        print("zpět po válcové vazbě a ložiskové síle:", len(w.study.constraints), "vazeb,",
              len(w.study.load_cases[0].loads), "load",
              "OK" if (len(w.study.constraints), len(w.study.load_cases[0].loads)) == (1, 1) else "ERROR")

    def s_mesh():
        w.set_stage(w.STAGE["Mesh"])
        w.view.set_selected_faces(faces_where(lambda part, p: p["type"] == "cylinder"), emit=True)
        w.ref_size.setValue(1.0)
        w.add_refinement()
        w.make_mesh()

    def s_mesh_shot():
        shot("4_sit")
        w.set_stage(w.STAGE["Solve"])
        shot("5_reseni")
        w.run_solve()

    def s_results():
        click_at((150, 30, 8))           # probe on the top face of the plate (the deformation moves the point only slightly)
        print("sonda:", w.st_info.text())
        shot("6_vysledky_vm")
        w.r_qty.setCurrentIndex(1)
        shot("7_vysledky_posuv")
        w.r_qty.setCurrentIndex(13)
        w.r_edges.setChecked(True)
        shot("8_bezpecnost")
        w.r_qty.setCurrentIndex(0)
        w.r_edges.setChecked(False)
        w.sec_axis.setCurrentIndex(2)      # section by the plane Y in the middle (coloured section through the mesh, the redraw with a delay)

    def s_section():
        shot("8b_rez")
        t0 = __import__("time").perf_counter()
        w.show_result()
        print(f"řez sítí: překreslení {__import__('time').perf_counter() - t0:.2f} s, sonda:", end=" ")
        click_at((100, 30, 4))           # a point inside the plate in the section plane y = 30 (outside the hole? the centre of the hole)
        click_at((60, 30, 4))
        print(w.st_info.text())
        # section by the plane of a face: the side face of the plate y = 0 (normal −y), the slider at y = 30
        side = faces_where(lambda part, p: part == "Plate" and p["type"] == "plane" and abs(p["bbox"][1]) < 1e-6
                           and abs(p["bbox"][4]) < 1e-6)
        w.view.set_selected_faces(side[:1])
        w.sec_axis.setCurrentIndex(w.SEC_FACE)
        w.view.set_selected_faces([])
        w.sec_pos.setValue(500)

    def s_section_face():
        w.view.set_orientation("front")
        w.show_result()
        shot("8c_rez_plocha")
        w.view.set_orientation("iso")
        w.sec_axis.setCurrentIndex(0)

    def s_fatigue():
        w.set_stage(w.STAGE["Fatigue"])
        fp = w.fat_panel
        fp.ch_kind.setCurrentIndex(0)
        fp.s_amp.setText("1")
        fp.s_mean.setText("0.5")
        fp.s_cyc.setText("100")
        fp.add_channel()
        fp.repeats.setText("1e5")
        fp.on_settings()
        fp.surface.setCurrentIndex(fp.surface.findData("obrobeny"))
        # S-N from the PragTic database: the dialog without a modal loop – 1045 normalized (#83, Basquin)
        from fea_gui import pragtic_dialog

        def fake_exec(dlg):
            dlg.q.setText("1045 normalized")
            row = next(i for i, r in enumerate(dlg.recs) if r["id"] == 83)
            dlg.table.setCurrentCell(row, 0)
            dlg.resize(1250, 780)
            dlg.grab().save(str(OUT / "a2_pragtic_dialog.png"))
            dlg.accept_sn()
            return True
        pragtic_dialog.PragticDialog.exec = fake_exec
        fp.sn_from_db()
        print("S-N z PragTic:", w.study.fatigue.sn[fp.sn_mat.currentText()]["source"])
        shot("a_unava_nastaveni")
        fp.compute()

    def s_fatigue_res():
        fp = w.fat_panel
        r = fp.result
        print("únava: D max", float(r.total_damage.max()), "min bezpečnost", float(r.safety.min()),
              "time", round(r.time_s, 2), "s")
        fp.scroll.ensureWidgetVisible(fp.worst)
        shot("b_unava_poskozeni")
        fp.worst.selectRow(0)
        fp.qty.setCurrentIndex(2)
        shot("c_unava_bezpecnost")
        w._save(OUT / "deska.fea")
        w.dirty = False

    def s_fatigue_cp():
        fp = w.fat_panel
        fp.eq.setCurrentIndex(fp.eq.findData("critical_plane"))
        assert fp.cp_box.isVisibleTo(fp) and w.study.fatigue.critical_plane
        fp.compute()

    def s_fatigue_cp_res():
        fp = w.fat_panel
        r = fp.result
        print("únava – kritická rovina: D max", float(r.total_damage.max()), "time", round(r.time_s, 2), "s;",
              r.method_lines()[-1])
        fp.worst.selectRow(0)
        print("normála v tabulce:", fp.worst.item(0, 6).text())
        fp.scroll.ensureWidgetVisible(fp.worst)
        shot("d_unava_kriticka_rovina")
        fp.eq.setCurrentIndex(fp.eq.findData("signed_mises"))
        w._save(OUT / "deska.fea")
        w.dirty = False

    def s_report():
        from PySide6.QtWidgets import QDialog
        from fea_gui import report_dialog

        def fake_exec(dlg):
            dlg.path.setText(str(OUT / "protokol.docx"))
            dlg.open_after.setChecked(False)
            dlg.notes.setPlainText("Zkušební závěr.")
            return QDialog.Accepted
        report_dialog.ReportDialog.exec = fake_exec
        w.export_report()

    def s_report_done():
        p = OUT / "protokol.docx"
        print("protokol:", p.exists(), p.stat().st_size if p.exists() else 0, "B")

    def s_reload():
        w.open_file(str(OUT / "deska.fea"))
        shot("9_znovu_otevreno")
        fs = w.study.fatigue
        print("po otevření: kanálů únavy", len(fs.history.channels) if fs else 0, "bloků", fs.repeats if fs else None)

    steps = [s_open, s_material, s_constraint, s_load, s_mesh, s_mesh_shot, s_results, s_section, s_section_face,
             s_fatigue, s_fatigue_res, s_fatigue_cp, s_fatigue_cp_res, s_report, s_report_done, s_reload]

    def run_next():
        busy = getattr(w, "_report_thread", None) is not None
        if w._thread is not None or w.fat_panel._thread is not None or busy:   # wait for the mesh / the calculation / the protocol
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
    if w.results is not None:
        st = w.results.step(0)
        print("max vM", float(st.mises.max()), "max u", float(st.umag.max()))


if __name__ == "__main__":
    main()
