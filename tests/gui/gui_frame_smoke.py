"""A pass through the GUI with the frame system and the 2D calculation by a section, with snapshots.

  python tests/gui/gui_frame_smoke.py OUT_DIR

1) Example 6 (hall frame without CAD): opening, the editor of the members (section from the library, node, member, selection by a click in 3D), load
   of the member in the form, calculation, the diagram My, the assessment EN 1993 (the Frames tab), own shape, buckling shape (MYSTRAN),
   saving and reopening.
2) A new frame system without CAD through the whole form: cantilever IPE 200, force at the end → deflection P·L³/(3EI).
3) The beam calculator (simply supported beam q, curved bar, Euler, assessment EN 1993).
3b) Mixed model: plate + bar, section from the end face of the bar (CAD face), "Replace part with a member", members from the wire model
   STEP, calculation (moment at the foot of the member F·L).
4) 2D case A1 (verification/twod_cases.py – tube with internal pressure as axisymmetric): calculation, σθ (Lamé),
   course along a line through the wall + linearisation.
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_frame_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FAIL: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow, FRAME_MEMBER, FRAME_NODE
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)   # closing the window: do not save
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    fea6 = OUT / "ukazka6.fea"
    shutil.copy(ROOT / "examples" / "example6_hall_frame.fea", fea6)
    log = []
    fp = w.frame_panel

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

    def qty(name):
        names = [w.r_qty.itemText(i).split(" [")[0] for i in range(w.r_qty.count())]
        check(name in names, f"veličina „{name}“ v nabídce")
        if name in names:
            w.r_qty.setCurrentIndex(names.index(name))
        app.processEvents()

    # ------------------------------------------------------------------ 1) example 6
    def s_open():
        w.open_file(str(fea6))
        st = w.study
        check(w.frame_only() and st.frame.active, "studie bez CAD s pruty")
        check(not w.stage_enabled("Joints") and w.stage_enabled("Members") and not w.stage_visible("Joints"),
              "u prutové soustavy bez CAD jsou kroky Spoje / 3D tisk / Teplo / Síť vypnuté")
        w.set_stage(w.STAGE["Members"])
        w.view.set_orientation("iso")
        w.view.fit_all()
        check(fp.t_nodes.rowCount() == 10 and fp.t_members.rowCount() == len(st.frame.members) == 19,
              f"tabulky: uzlů {fp.t_nodes.rowCount()}, prutů {fp.t_members.rowCount()}")
        shot("1_pruty_model")

    def s_edit():
        st = w.study
        fr = st.frame
        fp.tabs.setCurrentIndex(2)
        fp.s_src.setCurrentIndex(fp.s_src.findData("lib"))
        fp.s_find.setText("IPE200")
        check(fp.s_list.count() > 0 and fp.s_list.item(0).text() == "IPE 200", "hledání průřezu IPE200")
        fp.preview_section()
        check(fp._sec_preview is not None and abs(fp._sec_preview.Iy - 1.943e7) / 1.943e7 < 0.01,
              f"IPE 200 Iy = {getattr(fp._sec_preview, 'Iy', None)}")
        fp.s_name.setText("")
        fp.add_section()
        check("IPE 200" in fr.sections, "průřez IPE 200 přidán")
        shot("2_prurezy", None)
        fp.tabs.setCurrentIndex(0)
        fp.n_name.setText("T1")
        for s_, v in zip(fp.n_xyz, (6000.0, 3000.0, 7500.0)):
            s_.setValue(v)
        fp.add_node()
        check("T1" in fr.nodes, "uzel T1 přidán")
        fp.tabs.setCurrentIndex(1)
        fp.m_name.setText("test")
        fp.m_start.setCurrentText("A3")
        fp.m_end.setCurrentText("T1")
        fp.m_section.setCurrentText("IPE 200")
        fp.add_member()
        check("test" in fr.members and fr.members["test"].section == "IPE 200", "prut A3 → T1 přidán")
        n_undo = len(w._undo)
        w.undo()
        check("test" not in w.study.frame.members, "zpět odebere prut")
        w.redo()
        check("test" in w.study.frame.members and len(w._undo) == n_undo, "znovu vrátí prut")
        fp.refresh()
        fp.t_members.setCurrentCell(list(w.study.frame.members).index("test"), 0)
        fp.delete_member()
        fp.t_nodes.setCurrentCell(list(w.study.frame.nodes).index("T1"), 0)
        fp.delete_node()
        fr = w.study.frame
        check("test" not in fr.members and "T1" not in fr.nodes, "prut a uzel smazány")
        # selection by a click in 3D: node A2
        fp.tabs.setCurrentIndex(0)
        click_at(fr.nodes["A2"].xyz)
        r = fp.t_nodes.currentRow()
        check(r >= 0 and fp.t_nodes.item(r, 0).text() == "A2", f"klik ve 3D vybral uzel: řádek {r}")
        # in the iso view the projections of the bracings cross in the middle of the crossbar → a click nearer to the node A3 (otherwise it selects the bracing)
        a2, a3 = np.asarray(fr.nodes["A2"].xyz), np.asarray(fr.nodes["A3"].xyz)
        for t in (0.7, 0.6, 0.8):
            click_at(a2 + t * (a3 - a2))
            r = fp.t_members.currentRow()
            if r >= 0 and fp.t_members.item(r, 0).text() == "A příčel L":
                break
        r = fp.t_members.currentRow()
        check(fp.tabs.currentIndex() == 1 and r >= 0 and fp.t_members.item(r, 0).text() == "A příčel L",
              f"klik ve 3D vybral prut: {fp.t_members.item(r, 0).text() if r >= 0 else None}")
        shot("3_vyber_prutu")

    def s_loads():
        w.set_stage(w.STAGE["Loads"])
        k = [lc.name for lc in w.study.load_cases].index(next(n for n in ("Wind from left", "Vítr zleva") if n in [lc.name for lc in w.study.load_cases]))
        w.lc_combo.setCurrentIndex(k)
        texts = [w.load_list.item(i).text() for i in range(w.load_list.count())]
        check(any(t.startswith(("Member A sloup L", "Prut A sloup L")) for t in texts) and any(t.startswith(("Node A2", "Uzel A2")) for t in texts),
              f"popis zatížení prutů: {texts[:2]}")
        w.ld_type.setCurrentText(FRAME_NODE)
        check(w.ld_fr_node.isVisible() and not w.ld_fr_member.isVisible(), "formulář síly v uzlu")
        w.set_pick_filter("face")
        click_at(w.study.frame.nodes["B2"].xyz)
        check(w.ld_fr_node.currentText() == "B2", f"klik na uzel B2 → formulář: {w.ld_fr_node.currentText()}")
        for s_, v in zip(w.ld_vec, (0.0, 1500.0, 0.0)):
            s_.setValue(v)
        w.add_load()
        l = w.study.load_cases[k].loads[-1]
        check(l.kind == "frame_node" and l.node == "B2" and l.force[1] == 1500.0, f"síla v uzlu: {l}")
        w.ld_type.setCurrentText(FRAME_MEMBER)
        w.ld_fr_member.setCurrentText("B sloup L")
        for s_, v in zip(w.ld_vec, (1.0, 0.0, 0.0)):
            s_.setValue(v)
        w.ld_fr_var.setChecked(True)
        for s_, v in zip(w.ld_fr_qend, (2.0, 0.0, 0.0)):
            s_.setValue(v)
        w.add_load()
        l = w.study.load_cases[k].loads[-1]
        check(l.kind == "frame_member" and l.q_end == [2.0, 0.0, 0.0], f"proměnné spojité zatížení: {l}")
        shot("4_zatizeni_prutu")
        w.load_list.setCurrentRow(w.load_list.count() - 1)
        check(w.ld_type.currentText() == FRAME_MEMBER and w.ld_fr_var.isChecked(), "načtení do formuláře")
        w.delete_item(w.load_list, "load")
        w.load_list.setCurrentRow(w.load_list.count() - 1)
        w.delete_item(w.load_list, "load")
        check(len(w.study.load_cases[k].loads) == 3, "zatížení smazána")
        w.set_stage(w.STAGE["Solve"])
        w.eigen_box.b_on.setChecked(True)               # buckling of the frame system (MYSTRAN)
        w.eigen_box.b_case.setCurrentIndex(w.eigen_box.b_case.findData("Sníh"))
        check(w.study.buckling.enabled and w.study.buckling.case == "Sníh", "vzpěr zapnut pro stav Sníh")
        w.run_solve()

    def s_results():
        res = w.results
        check(res is not None and len(res.steps) == 3 and [m.kind for m in res.modes] == ["modal", "buckling"],
              "výsledky 3 stavů + frekvence + vzpěr")
        if res is None:
            return
        w.select_result("step:1")                     # Snow
        qty("Members – My")
        check(w.legend.isVisible() and "My" in w.legend.title, f"obrazec My: {w.legend.title}")
        check(w.res_tabs.isTabVisible(3) and w.frame_box.table.rowCount() == 19, "záložka Pruty (19 prutů)")
        check(any(x in w.res_info.toPlainText() for x in ("EN 1993-1-1 check", "Posouzení EN 1993")), "souhrn s posouzením")
        w.res_tabs.setCurrentIndex(3)
        w.frame_box.table.selectRow(w.frame_box._names.index("A sloup L"))
        app.processEvents()
        check("Interakce" in w.frame_box.detail.toPlainText() or "utilization" in w.frame_box.detail.toPlainText(),
              "detail posudku prutu")
        shot("5_obrazec_My")
        qty("Members – EN 1993 utilization")
        shot("6_vyuziti")
        qty("Members – stress σ_eq")
        w._probe is not None and w.on_probe(w._probe[0].data[0])
        check("member" in w.st_info.text(), f"sonda na prutu: {w.st_info.text()}")
        w.select_result("mode:0:0")
        check(w.legend.title.startswith(("Member shape", "Tvar prutů")) and "Hz" in w.res_info.toPlainText(), "vlastní tvar prutů")
        shot("7_vlastni_tvar")
        w.select_result("mode:1:0")
        check(w.legend.title.startswith(("Member shape", "Tvar prutů")) and "λ" in w.res_info.toPlainText(), "tvar vzpěru prutů")
        shot("8_tvar_vzperu")
        w._save(OUT / "example6_results.fea")
        w.dirty = False
        w.open_file(str(OUT / "example6_results.fea"))
        check(w.results is not None and getattr(w.results.modes[1], "frame_shapes_own", None) is not None,
              "tvary vzpěru prutů uloženy v .fea")
        w.select_result("mode:1:0")
        check(w.legend.title.startswith(("Member shape", "Tvar prutů")), "tvar vzpěru po otevření")

    # ------------------------------------------------------------------ 2) new frame system by the form
    def s_new():
        w.dirty = False
        w.new_frame_study()
        check(w.frame_only() and w.STAGES[w.stage] == "Members", "nová prutová soustava")
        fp.tabs.setCurrentIndex(2)
        fp.s_src.setCurrentIndex(fp.s_src.findData("lib"))
        fp.s_find.setText("IPE 200")
        fp.add_section()
        fp.tabs.setCurrentIndex(0)
        for name, x in (("A", 0.0), ("B", 1000.0)):
            fp.n_name.setText(name)
            for s_, v in zip(fp.n_xyz, (x, 0.0, 0.0)):
                s_.setValue(v)
            fp.add_node()
        fp.tabs.setCurrentIndex(1)
        fp.m_name.setText("konzola")
        fp.m_start.setCurrentText("A")
        fp.m_end.setCurrentText("B")
        fp.add_member()
        fp.tabs.setCurrentIndex(3)
        fp.u_node.setCurrentText("A")
        for cb in fp.u_dof:
            cb.setChecked(True)
        fp.save_support()
        fr = w.study.frame
        check(len(fr.nodes) == 2 and "konzola" in fr.members and len(fr.supports) == 1, "konzola zadaná formulářem")
        w.set_stage(w.STAGE["Loads"])
        w.ld_type.setCurrentText(FRAME_NODE)
        w.ld_fr_node.setCurrentText("B")
        for s_, v in zip(w.ld_vec, (0.0, 0.0, -1000.0)):
            s_.setValue(v)
        w.add_load()
        w.set_stage(w.STAGE["Solve"])
        errs, _ = w.check()
        check(not errs, f"připraveno: {errs}")
        w.run_solve()

    def s_new_done():
        res = w.results
        check(res is not None and res.steps, "konzola spočtena")
        if res is None:
            return
        from fea_core.beams.sections import get
        sec = get("IPE 200")
        u = res.steps[0].frame["nodes"]["B"][2]
        G = 210000.0 / 2.6
        exp = -1000.0 * 1000.0 ** 3 / (3 * 210000.0 * sec.Iy) - 1000.0 * 1000.0 / (G * sec.Av_z)   # Timoshenko
        check(abs(u - exp) / abs(exp) < 0.02, f"průhyb konce {u:.5f} mm (Timoshenko {exp:.5f})")
        qty("Members – displacement |u|")
        shot("9_konzola")

    # ------------------------------------------------------------------ 3) calculator
    def s_calc():
        w.open_beam_calc()
        d = w._calc_dlg
        r = d.calc_beam()
        check(r is not None and abs(r.extremes["M"]["max"][0] - 10 * 6000 ** 2 / 8) < 1e-3 * 45e6,
              f"prostý nosník qL²/8: {r.extremes['M']['max'][0] if r else None}")
        shot("10_kalkulacka", d)
        d.tabs.setCurrentIndex(1)
        c = d.calc_curved()
        check(c is not None and c.K_i > 1, f"křivý prut K_i = {getattr(c, 'K_i', None)}")
        d.tabs.setCurrentIndex(2)
        d.e_sec.setText("IPE 200")
        e = d.calc_euler()
        check(e is not None and e.F_cr > 0, f"Euler F_cr = {getattr(e, 'F_cr', None)}")
        d.tabs.setCurrentIndex(3)
        k = d.calc_ec3()
        check(k is not None and 0 < k.utilization < 5 and d.k_rows.rowCount() > 3,
              f"posudek EN 1993: využití {getattr(k, 'utilization', None)}")
        shot("11_kalkulacka_ec3", d)
        d.close()

    # ------------------------------------------------------------------ 3b) mixed model: a part replaced by a member
    def s_mixed():
        from fea_core.geometry import Part, write_step, face_props
        from verification.cases import box
        stp = OUT / "deska_tyc.step"
        write_step([Part("Plate", box(-50, -50, 0, 100, 100, 10)), Part("Tyc", box(-10, -10, 10, 20, 20, 200))], stp)
        w.dirty = False
        w.open_file(str(stp))
        w.set_stage(w.STAGE["Constraints"])
        sel = [(pi, fi) for pi, faces in enumerate(w.view.part_faces) for fi, f in enumerate(faces)
               if w.parts[pi].name == "Plate" and abs(face_props(f)["bbox"][5]) < 1e-6]
        w.view.set_selected_faces(sel, emit=True)
        w.add_fixed()
        w.set_stage(w.STAGE["Members"])
        # section from the end face of the bar (CAD face)
        top = [(pi, fi) for pi, faces in enumerate(w.view.part_faces) for fi, f in enumerate(faces)
               if w.parts[pi].name == "Tyc" and abs(face_props(f)["bbox"][2] - 210) < 1e-6]
        w.view.set_selected_faces(top, emit=True)
        fp.tabs.setCurrentIndex(2)
        fp.s_src.setCurrentIndex(fp.s_src.findData("face"))
        fp.s_name.setText("čelo tyče")
        fp.add_section()
        sd = w.study.frame.sections.get("čelo tyče")
        check(sd is not None and sd.props and abs(sd.props["A"] - 400) < 1, f"průřez z plochy CAD: A = "
              f"{sd.props['A'] if sd and sd.props else None}")
        fp.replace_part_named("Tyc")
        st = w.study
        check("Tyc" in st.frame.members and "Tyc" in st.ignored_parts and st.frame.connections,
              f"díl nahrazen prutem, spojení {[(c.node, c.selection) for c in st.frame.connections]}")
        # wire model STEP → members
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCP.gp import gp_Pnt
        from OCP.BRep import BRep_Builder
        from OCP.TopoDS import TopoDS_Compound
        c = TopoDS_Compound()
        bb = BRep_Builder()
        bb.MakeCompound(c)
        for a, b in (((200, 0, 0), (200, 0, 300)), ((200, 0, 300), (400, 0, 300))):
            bb.Add(c, BRepBuilderAPI_MakeEdge(gp_Pnt(*a), gp_Pnt(*b)).Edge())
        wr = STEPControl_Writer()
        wr.Transfer(c, STEPControl_AsIs)
        wr.Write(str(OUT / "drat.step"))
        new = fp.import_wires(str(OUT / "drat.step"), "čelo tyče")
        check(len(new) == 2, f"pruty z drátového modelu: {new}")
        for m in new:                                   # remove (a model without supports would be a mechanism)
            fp.t_members.setCurrentCell(list(w.study.frame.members).index(m), 0)
            fp.delete_member()
        for n in [n for n in w.study.frame.nodes if n.startswith("N")]:
            fp.t_nodes.setCurrentCell(list(w.study.frame.nodes).index(n), 0)
            fp.delete_node()
        shot("15_smiseny_model")
        w.set_stage(w.STAGE["Loads"])
        w.ld_type.setCurrentText(FRAME_NODE)
        m = w.study.frame.members["Tyc"]
        conn = {c.node for c in w.study.frame.connections}
        w.ld_fr_node.setCurrentText(next(n for n in (m.start, m.end) if n not in conn))
        for s_, v in zip(w.ld_vec, (100.0, 0.0, 0.0)):
            s_.setValue(v)
        w.add_load()
        w.set_stage(w.STAGE["Solve"])
        errs, _ = w.check()
        check(not errs, f"smíšený model připraven: {errs}")
        w.run_solve()

    def s_mixed_done():
        res = w.results
        check(res is not None and res.steps and res.steps[0].frame, "smíšený model spočten")
        if res is None:
            return
        check(res.mesh.part_names == ["Plate"], f"díl nahrazený prutem se nesíťuje: {res.mesh.part_names}")
        names = [w.r_qty.itemText(i).split(" [")[0] for i in range(w.r_qty.count())]
        check("von Mises" in names and "Members – My" in names, "veličiny těles i prutů")
        qty("von Mises")
        shot("16_smiseny_von_mises")
        qty("Members – My")
        M = max(abs(np.asarray(md["My"])).max() for md in res.steps[0].frame["members"].values())
        M = max(M, max(abs(np.asarray(md["Mz"])).max() for md in res.steps[0].frame["members"].values()))
        check(abs(M - 100.0 * 200.0) / 20000.0 < 0.02, f"moment v patě prutu {M:.1f} N·mm (F·L = 20 000)")
        shot("17_smiseny_My")

    # ------------------------------------------------------------------ 4) 2D
    def s_2d():
        from fea_core.project import save_project
        import verification.cases  # noqa: F401 – twod_cases is imported from cases (circular import)
        from verification.twod_cases import a1_build
        wd = OUT / "a1"
        wd.mkdir(exist_ok=True)
        st, _ = a1_build(wd)
        save_project(OUT / "a1.fea", st, st.geometry)
        w.dirty = False
        w.open_file(str(OUT / "a1.fea"))
        check(w.study.twod.enabled, "studie A1 s 2D řezem")
        w.set_stage(w.STAGE["Solve"])
        tb = w.twod_box
        check(tb.on.isChecked() and tb.kind.currentData() == "axisym", "skupina 2D výpočet řezem")
        w.view.set_orientation("iso")
        w.view.fit_all()
        shot("12_2d_nastaveni")
        w.run_solve()

    def s_2d_res():
        from verification.cases import RI, RO, V4_H
        from verification.twod_cases import _lame_tube
        res = w.results
        check(res is not None and res.twod is not None, "výsledek 2D")
        if res is None or res.twod is None:
            return
        items = [w.r_step.itemText(i) for i in range(w.r_step.count())]
        check(any(t.startswith("2D:") for t in items), f"nabídka: {items}")
        w.select_result("twod:0")
        qty("σθ hoop")
        fr = res.twod.frame2d()
        p = fr.to3d(np.array([RI, V4_H / 2]))
        _, sth_i, _ = _lame_tube()
        v = w._probe_exact(w.displayed(p)) if w._probe_exact else None
        check(v is not None and abs(v - sth_i) / sth_i < 0.01, f"sonda σθ na ri = {v} (Lamé {sth_i:.4g})")
        shot("13_2d_sigma_theta")
        w.open_path_dialog()
        dlg = w._path_dlg
        k = next(i for i in range(dlg.step.count()) if dlg.step.itemData(i)[0] == "twod")
        dlg.step.setCurrentIndex(k)
        dlg.set_point("A", fr.to3d(np.array([RI, V4_H / 2])))
        dlg.set_point("B", fr.to3d(np.array([RO, V4_H / 2])))
        names = [dlg.field.itemText(i) for i in range(dlg.field.count())]
        dlg.field.setCurrentIndex(next(i for i, t in enumerate(names) if t.startswith("σθ")))
        dlg.plot()
        vals = np.asarray(dlg.data["values"], float) if dlg.data else np.array([np.nan])
        check(np.isfinite(vals).all() and abs(vals[0] - sth_i) / sth_i < 0.02, f"průběh σθ přes stěnu: {vals[:2]}")
        dlg.linearize()
        check(dlg.lin is not None and dlg.lin["Pm_mises"] > 0, "linearizace 2D")
        shot("14_2d_prubeh", dlg)
        dlg.close()

    steps = [s_open, s_edit, s_loads, s_results, s_new, s_new_done, s_calc, s_mixed, s_mixed_done, s_2d, s_2d_res]

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
