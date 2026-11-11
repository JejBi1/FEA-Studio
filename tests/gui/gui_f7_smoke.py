"""A pass through the GUI functions of F7 with snapshots: orthotropic material, a part meshed separately (*TIE), elastic support, refinement by a sphere,
natural frequencies and buckling, display of the shapes, estimate of the stress error, progress along a line + linearisation, mesh convergence,
the message table, undo/redo, saving and reopening, print tolerance and two-step refinement.

  python tests/gui/gui_f7_smoke.py OUT_DIR

Model: cantilever 200×20×10 split at x = 100 into the parts A and B (B meshed separately, joined by *TIE – V17), clamping
x = 0, elastic support of the bottom face, force (−1000; 0; −10) N on the face x = 200. The user library of materials goes
into OUT_DIR (APPDATA), not into the profile of the user.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_f7_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")           # material library / presets only into the folder of the test

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

FAIL: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def make_step(path):
    from fea_core.geometry import Part, write_step
    from verification.cases import box
    write_step([Part("Konzola A", box(0, 0, 0, 100, 20, 10)), Part("Konzola B", box(100, 0, 0, 100, 20, 10))], path)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow, FORCE, QUANTITIES
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")      # without modal windows
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)   # closing the window: do not save
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    step = OUT / "konzola.step"
    make_step(step)
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
        if extra is not None:                          # a non-modal dialog separately
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

    def faces_where(pred):
        from fea_core.geometry import face_props
        return [(pi, fi) for pi, faces in enumerate(w.view.part_faces) for fi, f in enumerate(faces)
                if pred(w.parts[pi].name, face_props(f))]

    def bb(p):
        return p["bbox"]

    # ------------------------------------------------------------------ kroky
    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")
        w.ms_size.setValue(5.0)
        check(w.study.mesh.size == 5.0, "velikost prvku 5 mm")

    def s_material():
        w.set_stage(w.STAGE["Materials"])
        w.mat_parts.clearSelection()
        w.mat_parts.item(1).setSelected(True)                 # Konzola B
        w.open_material_dialog()                              # properties and orthotropy are in a pop-up window
        w.m_name.setText("Ortho ocel")
        w.ortho_box.setChecked(True)
        check(w.ortho_box.inner.isVisible() and not w.m_E.isEnabled(), "ortotropní: konstanty vidět, E/ν vypnuto")
        w.ortho_box.fill_isotropic()
        w.ortho_box.spins["E1"].setValue(210000.0)
        w.ortho_box.spins["alpha1"].setValue(12.0)
        w.assign_material(False)
        m = w.study.materials.get("Ortho ocel")
        check(m is not None and m.is_ortho and abs(m.ortho["G12"] - 210000 / 2.6) < 1 and
              abs(m.ortho["alpha1"] - 12e-6) < 1e-12 and "alpha2" not in m.ortho,
              f"ortotropní materiál ve studii: {m.ortho if m else None}")
        check(w.study.part_materials.get("Konzola B") == "Ortho ocel", "přiřazen dílu Konzola B")
        w.save_user_material()                                # knihovna v OUT/appdata
        from fea_core import materials as matlib
        lib = {x.name: x for x in matlib.user_library()}
        check("Ortho ocel" in lib and lib["Ortho ocel"].is_ortho, "uživatelská knihovna uloží a načte ortho")
        w.mat_combo.setCurrentText("Ortho ocel")
        check(w.ortho_box.isChecked() and abs(w.ortho_box.spins["nu12"].value() - 0.3) < 1e-9,
              "formulář načte ortotropní konstanty")
        shot("1_ortotropni_material")

    def s_ties():
        w.set_stage(w.STAGE["Joints"])
        jp = w.joints_panel
        from PySide6.QtCore import Qt
        for i in range(jp.sep_list.count()):
            it = jp.sep_list.item(i)
            if it.data(Qt.UserRole) == "Konzola B":
                it.setCheckState(Qt.Checked)
        check(w.study.mesh.separate == ["Konzola B"], f"síťovat samostatně: {w.study.mesh.separate}")

    def s_constraints():
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces(faces_where(lambda part, p: abs(bb(p)[0]) < 1e-6 and abs(bb(p)[3]) < 1e-6), emit=True)
        w.add_fixed()
        w.view.set_selected_faces(faces_where(lambda part, p: abs(bb(p)[2]) < 1e-6 and abs(bb(p)[5]) < 1e-6), emit=True)
        w.el_group.setChecked(True)
        w.el_k.setValue(0.5)
        w.el_unit.setCurrentIndex(0)
        w.el_dir.setCurrentIndex(w.el_dir.findData("normal"))
        w.add_elastic()
        c = w.study.constraints[-1]
        check(c.kind == "elastic" and c.stiffness == 0.5 and c.direction == "normal" and c.per_area,
              f"pružné uložení: {c}")
        check("elastic support k = " in w.con_list.item(1).text() and "N/mm³" in w.con_list.item(1).text(), f"popis: {w.con_list.item(1).text()}")
        w.con_list.setCurrentRow(1)
        w.el_k.setValue(0.4)
        w.update_elastic()
        check(w.study.constraints[1].stiffness == 0.4, "úprava tuhosti pružného uložení")
        shot("2_pruzne_ulozeni")

    def s_loads():
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces(faces_where(lambda part, p: abs(bb(p)[0] - 200) < 1e-6 and abs(bb(p)[3] - 200) < 1e-6),
                                  emit=True)
        w.ld_type.setCurrentText(FORCE)
        for s_, v in zip(w.ld_vec, (-1000.0, 0.0, -10.0)):
            s_.setValue(v)
        w.add_load()
        check(len(w.study.load_cases[0].loads) == 1, "síla na čele")

    def s_mesh():
        w.set_stage(w.STAGE["Mesh"])
        for s_, v in zip(w.ball_pt, (150.0, 10.0, 5.0)):
            s_.setValue(v)
        w.ball_r.setValue(8.0)
        w.ball_h.setValue(2.0)
        w.add_ball_refinement()
        loc = w.study.mesh.local
        check(len(loc) == 1 and loc[0]["center"] == [150.0, 10.0, 5.0] and loc[0]["radius"] == 8.0,
              f"zjemnění koulí: {loc}")
        check(w.ref_list.count() == 1 and w.ref_list.item(0).text().startswith("Sphere"), "koule v seznamu zjemnění")
        w.make_mesh()

    def s_mesh_done():
        check(w.mesh is not None, "síť vytvořena")
        if w.mesh is None:
            return
        ties = w.mesh.report.get("ties", [])
        check(len(ties) == 1 and abs(ties[0]["area_mm2"] - 200) < 1, f"*TIE nalezen: {ties}")
        check(any("*TIE" in m["text"] for m in w.messages.messages(source="mesh")), "zpráva sítě o *TIE")
        w.set_stage(w.STAGE["Joints"])
        check(w.joints_panel.tie_table.rowCount() == 1, "tabulka *TIE v kroku Spoje")
        shot("3_spojeni_tie")
        w.set_stage(w.STAGE["Mesh"])
        shot("4_sit_koule")

    def s_solve():
        w.set_stage(w.STAGE["Solve"])
        eb = w.eigen_box
        eb.m_on.setChecked(True)
        eb.m_modes.setValue(4)
        eb.b_on.setChecked(True)
        eb.b_modes.setValue(2)
        eb.b_case.setCurrentIndex(eb.b_case.findData(w.study.load_cases[0].name))
        check(w.study.modal.enabled and w.study.modal.modes == 4 and w.study.buckling.enabled and
              w.study.buckling.case == w.study.load_cases[0].name, f"nastavení: {w.study.modal}, {w.study.buckling}")
        check(eb.b_note.isVisible(), "varování lineárního vzpěru")
        check(any("Buckling is linear" in m["text"] for m in w.messages.messages(source="study check")),
              "kontrola studie v tabulce zpráv")
        shot("5_reseni_frekvence_vzper")
        w.run_solve()

    def s_results():
        res = w.results
        check(res is not None, "results")
        if res is None:
            return
        check(len(res.steps) == 1 and [m.kind for m in res.modes] == ["modal", "buckling"],
              f"stavy {len(res.steps)}, módy {[m.kind for m in res.modes]}")
        st = res.steps[0]
        sel = w.study.constraints[1].selection
        check(sel in st.reactions and abs(st.reactions[sel][2]) > 0, f"reakce pružného uložení {st.reactions.get(sel)}")
        check("stress_error" in st.fields, "pole odhadu chyby napětí")
        items = [w.r_step.itemText(i) for i in range(w.r_step.count())]
        check(sum(t.startswith("Mode shape") for t in items) == 4 and sum(t.startswith("Buckling mode") for t in items)
              == 2, f"nabídka výsledků: {items}")
        w.select_result("mode:0:0")
        f1 = res.modes[0].values[0]
        check("Hz" in w.res_info.toPlainText() and w.r_qty.itemText(0).startswith("Mode"),
              f"vlastní tvar 1: f = {f1:.1f} Hz, veličina {w.r_qty.itemText(0)}")
        check(w.res_tabs.isTabVisible(2) and w.modes_table.table.rowCount() == 6, "záložka Vlastní tvary (6 řádků)")
        w.res_tabs.setCurrentIndex(2)
        shot("6_vlastni_tvar")
        w.modes_table.table.selectRow(4)                       # 1st buckling shape
        app.processEvents()
        check(w.r_step.currentData() == "mode:1:0" and "λ" in w.res_info.toPlainText(), "tvar vzpěru z tabulky")
        shot("7_tvar_vzperu")
        w.select_result("step:0")
        names = [q[0] for q in QUANTITIES]
        w.r_qty.setCurrentIndex(names.index("Stress error estimate"))
        check(w.legend.isVisible(), "mapa odhadu chyby napětí")
        w.res_tabs.setCurrentIndex(0)
        shot("8_odhad_chyby")
        w.r_qty.setCurrentIndex(0)

    def s_path():
        w.open_path_dialog()
        dlg = w._path_dlg
        dlg.set_point("A", (50.0, 10.0, 0.0))
        dlg.set_point("B", (50.0, 10.0, 10.0))
        dlg.field.setCurrentIndex(0)
        dlg.plot()
        v = np.asarray(dlg.data["values"], float) if dlg.data else np.array([np.nan])
        check(np.isfinite(v).all() and v.max() > 1, f"průběh von Mises přes tloušťku: {np.nanmin(v):.3g}…{np.nanmax(v):.3g}")
        dlg.linearize()
        check(dlg.lin is not None and dlg.lin_table.rowCount() == 7 and dlg.lin["Pm_mises"] > 0,
              f"linearizace: Pm {dlg.lin['Pm_mises'] if dlg.lin else None}")
        p = dlg.export_csv(str(OUT / "prubeh.csv"))
        txt = Path(p).read_text(encoding="utf-8-sig")
        check(any(x in txt for x in ("Stress linearization", "Linearizace")) and txt.count("\n") > 50, "export CSV")
        # point A by a click in 3D (the top face of the cantilever)
        dlg.start_pick("A", True)
        w.view.fit_all()
        app.processEvents()
        click_at(w.displayed(np.array([150.0, 10.0, 10.0])))
        a = dlg.point("A")
        check(abs(a[0] - 150) < 8 and w._pick_hook is None, f"bod A klikem: {np.round(a, 2)}")
        dlg.set_point("A", (50.0, 10.0, 0.0))
        shot("9_prubeh_po_care", dlg)

    def s_conv():
        w._path_dlg.close()
        w.open_convergence_dialog()
        dlg = w._conv_dlg
        dlg.qty.setCurrentIndex(dlg.qty.findData("max_disp"))
        dlg.levels.setValue(2)
        dlg.ratio.setValue(1.5)
        dlg.run()
        check(dlg.thread is not None, "konvergence běží ve vlákně")

    def s_conv_done():
        dlg = w._conv_dlg
        out = dlg.result
        if out is None:
            print("\n".join(w.log.toPlainText().splitlines()[-25:]))
        check(out is not None and len(out["levels"]) == 2 and out["levels"][1]["nodes"] > out["levels"][0]["nodes"],
              f"konvergence 2 úrovně: {[(r['size_mm'], r['nodes'], r['value']) for r in (out or {}).get('levels', [])]}")
        check(dlg.tab.rowCount() == 2 and any(x in dlg.out.toPlainText() for x in ("onverged", "onvergováno")), f"verdikt: {dlg.out.toPlainText()[:80]}")
        check(w.results is not None and len(w.results.modes) == 2, "výsledky hlavního okna konvergence nezměnila")
        shot("10_konvergence", dlg)
        dlg.close()

    def s_messages():
        mp = w.messages
        w.dock_messages.raise_()
        srcs = {m["source"] for m in mp.messages()}
        check({"study check", "mesh", "solver", "results"} <= srcs, f"zdroje zpráv: {srcs}")
        mp.filter.setCurrentIndex(mp.filter.findData("warning"))
        check(mp.table.rowCount() == len(mp.messages(level="warning")) > 0, "filtr varování")
        txt = mp.copy()
        check("warning" in txt and "\t" in txt, "kopírování zpráv")
        r = next((i for i, m in enumerate(mp._rows) if "elastic support" in m["text"].lower()), None)
        check(r is not None, "varování pružného uložení v tabulce")
        if r is not None:
            mp.goto(r)
            check(w.STAGES[w.stage] == "Constraints", f"dvojklik → krok {w.STAGES[w.stage]}")
        mp.filter.setCurrentIndex(0)
        shot("11_zpravy")

    def s_undo():
        w.set_stage(w.STAGE["Solve"])
        w.eigen_box.m_modes.setValue(6)
        check(w.study.modal.modes == 6 and w.results is None, "změna počtu módů zahodí výsledky")
        w.undo()
        check(w.study.modal.modes == 4 and w.eigen_box.m_modes.value() == 4, "zpět vrátí počet módů")
        w.redo()
        check(w.study.modal.modes == 6, "znovu")
        w.undo()
        w.run_solve()

    def s_save():
        check(w.results is not None and len(w.results.modes) == 2, "přepočteno")
        w._save(OUT / "f7.fea")
        w.dirty = False
        w.open_file(str(OUT / "f7.fea"))
        st = w.study
        check(st.modal.enabled and st.modal.modes == 4 and st.buckling.enabled, "modal / vzpěr uloženy")
        check(any(c.kind == "elastic" for c in st.constraints), "pružné uložení uloženo")
        check(st.materials["Ortho ocel"].is_ortho and st.mesh.separate == ["Konzola B"], "ortho a *TIE uloženy")
        check(any("center" in r for r in st.mesh.local), "zjemnění koulí uloženo")
        check(w.results is not None and len(w.results.modes) == 2 and w.r_step.findData("mode:0:3") >= 0,
              "výsledky s módy po otevření")

    def s_print():
        # print tolerance + two-step refinement (without frequencies and buckling – faster)
        w.eigen_box.m_on.setChecked(False)
        w.eigen_box.b_on.setChecked(False)
        w.set_stage(w.STAGE["3D printing"])
        pp = w.print_panel
        pp.parts.clearSelection()
        pp.parts.item(0).setSelected(True)                     # Konzola A
        pp.mat.setCurrentText("Tisk PLA Bambu Basic")
        pp.assign_material()
        pp.parts.clearSelection()
        pp.parts.item(0).setSelected(True)
        pp.s_tol.setValue(0.1)
        pp.apply_spec()
        sp = w.study.printing.spec("Konzola A")
        check(abs(sp.tolerance - 0.1) < 1e-12 and w.study.material_of("Konzola A").is_print,
              f"tolerance tisku 0,1 mm: {sp.tolerance}")
        shot("12_tisk_tolerance")
        w.run_solve()

    def s_refine():
        res = w.results
        check(res is not None and "print_safety" in res.steps[0].fields, "výsledky tisku")
        if res is None:
            return
        n0 = len(res.mesh.nodes)
        w._refine_n0 = (n0, len(w.study.mesh.local))
        w.set_stage(w.STAGE["3D printing"])
        w.print_panel.refine_n.setValue(1)
        w.print_panel.refine_two_step()

    def s_refine_done():
        n0, k0 = w._refine_n0
        res = w.results
        check(res is not None and len(res.mesh.nodes) > n0 and len(w.study.mesh.local) == k0 + 1,
              f"zjemnění převzato: uzlů {n0} → {len(res.mesh.nodes) if res else None}, "
              f"zjemnění {k0} → {len(w.study.mesh.local)}")
        check(w.valid_mesh() is not None, "převzatá síť platí pro studii")
        txt = w.print_panel.refine_info.text()
        check("print safety" in txt and "– →" not in txt, f"souhrn zjemnění: {txt[:90]}")
        w.set_stage(w.STAGE["3D printing"])
        w.print_panel.scroll_to(w.print_panel.refine_info)
        shot("13_zjemneni_tisku")

    steps = [s_open, s_material, s_ties, s_constraints, s_loads, s_mesh, s_mesh_done, s_solve, s_results, s_path,
             s_conv, s_conv_done, s_messages, s_undo, s_save, s_print, s_refine, s_refine_done]

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
