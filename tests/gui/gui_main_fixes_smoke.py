"""Check of the fixes of the main window (audit 13. 10. 2026) – programmatically, without a calculation.

  python tests/gui/gui_main_fixes_smoke.py OUT_DIR

1 gravity after switching the type always (0; 0; −9810), editing an existing gravity keeps the value; 2 a change of a load on
faces to gravity leaves the selection only when something else uses it, "Force normal to the face" when editing only on
planar faces; 3 restoring the menu of the materials keeps the selected / assigned material; 4 the cylindrical constraint default in
all components; 5 deleting a load state with a question; 6 the approximate check of the rigid motions = a warning, the calculation
allowed; 7 Mesh step: the switch "Show mesh", a change of the mesh settings returns the view to the parts; 8 context menu of the 3D
window, double click = zoom to the selection (the face stays selected); 9 the 3D mouse connected (without a device nothing happens);
the assignment of a material does not delete the materials of the study; undo / redo of the joints by the effect of the change; the axis of each hole of a cylindrical constraint;
opening a study JSON (the metadata from the CAD are not opened as a study).
The dialog windows are replaced (msg, ask) – the test must not wait for a click.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_main_fixes_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL: list[str] = []
MSGS: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def make_step(path):
    from verification.cases import box, cyl
    from fea_core.geometry import Part, write_step
    write_step([Part("Plate", box(0, 0, 0, 100, 20, 10)),
                Part("Cep", cyl((0, 60, 0), (1, 0, 0), 5, 50)),
                Part("Cep2", cyl((0, 90, 0), (1, 0, 0), 5, 50))], path)


def main():
    import fea_gui.main as gm
    from fea_core.study import Force, Gravity, CylindricalSupport, LoadCase

    def fake_msg(parent, title, text, icon=None):
        MSGS.append(str(text))
        print("  [hláška]", " | ".join(str(text).splitlines())[:400])
        return 0
    gm.msg = fake_msg                                    # no modal window

    app = QApplication(sys.argv)
    w = gm.MainWindow()
    w.resize(1400, 900)
    w.show()
    w.ask = lambda *_: True                              # confirm the questions (deleting a state)
    step = OUT / "model.step"
    make_step(step)

    def faces_where(pred):
        from fea_core.geometry import face_props
        return [(pi, fi) for pi, faces in enumerate(w.view.part_faces) for fi, f in enumerate(faces)
                if pred(w.parts[pi].name, face_props(f))]

    def vec():
        return [s.value() for s in w.ld_vec]

    def s_open():
        w.open_file(str(step))
        check(w.study is not None and len(w.parts) == 3, "model načten (3 díly)")

    def s_materials():                                   # 3
        w.set_stage(w.STAGE["Materials"])
        names = [w.mat_combo.itemText(i) for i in range(w.mat_combo.count())]
        other = next(n for n in names[1:] if not n.startswith("Print"))
        w.mat_combo.setCurrentText(other)
        w.m_E.setValue(123456.0)                         # an unfinished value in the form
        w.changed(invalidate_results=False)              # restoring everything (refresh_materials)
        check(w.mat_combo.currentText() == other, f"po obnovení zůstal vybraný materiál '{other}' "
                                                  f"(je '{w.mat_combo.currentText()}')")
        check(w.m_E.value() == 123456.0, "rozpracovaný formulář materiálu se obnovením nepřepsal")
        w.mat_parts.clearSelection()
        w.mat_parts.item(0).setSelected(True)
        w.mat_combo.setCurrentText(other)
        w.m_name.setText("Můj materiál")
        w.m_E.setValue(150000.0)
        from fea_core.study import Material
        w.study.materials["Kalibrovaný PLA"] = Material("Kalibrovaný PLA", 3000.0, 0.35, 1.24e-9, Rm=40.0)
        w.assign_material(False)
        check("Kalibrovaný PLA" in w.study.materials, "přiřazení nesmaže nepřiřazený materiál studie (kalibrace tisku)")
        w.mat_parts.clearSelection()
        w.mat_parts.item(0).setSelected(True)
        w.mat_combo.setCurrentText(other)
        w.m_name.setText("Můj materiál")
        w.m_E.setValue(150000.0)
        w.assign_material(False)
        check(w.study.part_materials.get("Plate") == "Můj materiál", "materiál přiřazen dílu Deska")
        check(w.mat_combo.currentText() == "Můj materiál" and w.m_name.text() == "Můj materiál"
              and w.m_E.value() == 150000.0,
              f"po přiřazení zůstal vybraný přiřazený materiál (nabídka '{w.mat_combo.currentText()}', "
              f"název '{w.m_name.text()}', E {w.m_E.value()})")
        check([i.row() for i in w.mat_parts.selectedIndexes()] == [0], "výběr dílu v seznamu zůstal")

    def s_constraints():                                 # 4
        w.set_stage(w.STAGE["Constraints"])
        check(all(c.isChecked() for c in w.cyl_chk), "válcová vazba: výchozí radiálně + tečně + osově")
        # both pins with one selection (non-coaxial – the core: each hole has its own axis), otherwise Cep2 would be a free body
        w.view.set_selected_faces(faces_where(lambda part, p: part in ("Cep", "Cep2") and p["type"] == "cylinder"),
                                  emit=True)
        w.add_cylindrical()
        c = w.study.constraints[-1]
        check(isinstance(c, CylindricalSupport) and c.radial and c.tangential and c.axial,
              f"válcová vazba z výběru ve všech složkách: {c}")
        w.view.set_selected_faces(faces_where(lambda part, p: part == "Plate" and abs(p["bbox"][3]) < 1e-6),
                                  emit=True)
        w.add_fixed()
        check(len(w.study.constraints) == 2, "vetknutí čela desky")
        both = faces_where(lambda part, p: part in ("Cep", "Cep2") and p["type"] == "cylinder")
        axes = w.cylinders_of(both)
        check(len(axes) == 2 and {round(float(a[0][1])) for a in axes} == {60, 90},
              f"válcová vazba na 2 nesouosých čepech: osa každého ({len(axes)})")

    def s_gravity():                                     # 1
        w.set_stage(w.STAGE["Loads"])
        w.ld_type.setCurrentText(gm.FORCE)
        for s, v in zip(w.ld_vec, (0.0, 0.0, -200.0)):
            s.setValue(v)
        w.ld_type.setCurrentText("Gravity")
        check(vec() == [0.0, 0.0, -9810.0], f"přepnutí na gravitaci: g = {vec()} (ne zbytková síla −200)")
        w.ld_vec[2].setValue(-5000.0)
        w.update_status()                                # on_ld_type(0) is called at every change of the selection
        check(vec() == [0.0, 0.0, -5000.0], "zadaná hodnota g se při obnovení formuláře nepřepíše")
        w.add_load()
        lc = w.study.load_cases[0]
        check(isinstance(lc.loads[-1], Gravity) and lc.loads[-1].vector == [0.0, 0.0, -5000.0], "gravitace přidána")
        w.ld_type.setCurrentText(gm.FORCE)
        check(vec() == [0.0, 0.0, 0.0], f"přepnutí z gravitace na sílu vektor vynuluje: {vec()}")
        w.load_list.setCurrentRow(len(lc.loads) - 1)     # editing an existing gravity
        check(w.ld_type.currentText() == "Gravity" and vec() == [0.0, 0.0, -5000.0],
              f"načtení existující gravitace do formuláře zachová g: {vec()}")

    def s_update_load():                                 # 2
        lc = w.study.load_cases[0]
        top = faces_where(lambda part, p: part == "Plate" and abs(p["bbox"][2] - 10) < 1e-6
                          and abs(p["bbox"][5] - 10) < 1e-6)
        w.view.set_selected_faces(top, emit=True)
        w.ld_type.setCurrentText(gm.FORCE)
        for s, v in zip(w.ld_vec, (0.0, 0.0, -100.0)):
            s.setValue(v)
        w.add_load()
        sel = lc.loads[-1].selection
        check(sel in w.study.selections, f"síla na ploše má výběr '{sel}'")
        row = len(lc.loads) - 1
        w.load_list.setCurrentRow(row)
        w.ld_type.setCurrentText("Gravity")
        w.update_load()
        check(isinstance(lc.loads[row], Gravity), "zatížení změněno na gravitaci")
        check(sel not in w.study.selections and sel not in w.sel_faces, f"osiřelý výběr '{sel}' odstraněn")
        # the selection shared with a constraint stays
        fixed_sel = w.study.constraints[-1].selection
        lc.loads.append(Force(fixed_sel, [0.0, 0.0, -1.0]))
        w.changed()
        w.load_list.setCurrentRow(len(lc.loads) - 1)
        w.ld_type.setCurrentText("Gravity")
        w.update_load()
        check(fixed_sel in w.study.selections, f"výběr '{fixed_sel}' používaný vazbou zůstal")
        # "Force normal to the face" when editing only on planar faces
        w.view.set_selected_faces(faces_where(lambda part, p: part == "Cep" and p["type"] == "cylinder"), emit=True)
        w.ld_type.setCurrentText(gm.FORCE)
        for s, v in zip(w.ld_vec, (0.0, 0.0, -10.0)):
            s.setValue(v)
        w.add_load()
        row = len(lc.loads) - 1
        before = lc.loads[row]
        w.load_list.setCurrentRow(row)
        w.ld_type.setCurrentText("Force normal to face")
        w.ld_val.setValue(50.0)
        n = len(MSGS)
        w.update_load()
        check(lc.loads[row] is before and any("planar" in m for m in MSGS[n:]),
              "úprava na „Sílu kolmo na plochu“ na válcové ploše odmítnuta")

    def s_delete_lc():                                   # 5
        w.study.load_cases.append(LoadCase("Delete"))
        w.changed()
        w.lc_combo.setCurrentIndex(len(w.study.load_cases) - 1)
        asked = []
        w.ask = lambda text: asked.append(text) or False
        w.delete_lc()
        check(len(asked) == 1 and "Delete" in asked[0] and any(lc.name == "Delete" for lc in w.study.load_cases),
              "smazání stavu se ptá; „Ne“ stav ponechá")
        w.ask = lambda *_: True
        w.delete_lc()
        check(not any(lc.name == "Delete" for lc in w.study.load_cases), "po potvrzení stav smazán")
        w.lc_combo.setCurrentIndex(0)

    def s_rigid():                                       # 6
        import fea_core.mcp.tools_mesh as tm
        orig = tm.rigid_body_check
        try:
            tm.rigid_body_check = lambda sess: {"method": "x", "free_rigid_modes": ["rotace kolem osy (1; 0; 0)"],
                                                "ok": False, "note": "výsledek je přibližný."}
            w._rigid = None
            w.set_stage(w.STAGE["Solve"])
            items = [w.ready_list.item(i).text() for i in range(w.ready_list.count())]
            check(any("rigid-body motion" in t for t in items) and w.btn_run.isEnabled(),
                  f"přibližná kontrola tuhých pohybů = varování, výpočet povolen: {items}")
            check(w.rigid_errors() == [], "rigid_errors bez přibližného výsledku (varianty tisku)")
            tm.rigid_body_check = lambda sess: {"method": "x", "free_rigid_modes": ["posuv ve směru (0; 0; 1)"],
                                                "ok": False}
            w._rigid = None
            w.refresh_ready()
            check(not w.btn_run.isEnabled() and len(w.rigid_errors()) == 1, "přesná kontrola tuhých pohybů blokuje")
        finally:
            tm.rigid_body_check = orig
            w._rigid = None

    def s_mesh():                                        # 7
        w.set_stage(w.STAGE["Mesh"])
        w.ms_size.setValue(4.0)
        w.make_mesh()

    def s_mesh_view():
        check(w.mesh is not None and w._mesh_view and not w.view.parts_visible, "krok Síť ukazuje hotovou síť")
        w.ms_show.setChecked(False)
        check(w.view.parts_visible and w.view.mode == "faces" and not w._mesh_view,
              "„Zobrazit síť“ vypnuto → díly s výběrem ploch")
        w.view.set_selected_faces(faces_where(lambda part, p: part == "Cep" and p["type"] == "cylinder"), emit=True)
        w.ref_size.setValue(1.5)
        w.add_refinement()
        check(len(w.study.mesh.local) == 1, "zjemnění výběru v kroku Síť")
        w.ms_show.setChecked(True)
        check(not w._mesh_view and w.view.parts_visible, "síť zahozena zjemněním → zobrazeny díly")
        w.make_mesh()

    def s_joints_undo():                                # undo / redo for the joints (joints_panel.change_level)
        from fea_core.results import Results
        from fea_core.joints.model import BoltSpec
        check(w._mesh_view and not w.view.parts_visible, "nová síť zobrazena")
        b = BoltSpec("B1").to_dict()
        lvl = w.joints_change_level
        jd = w.study.joints.to_dict()
        check(lvl({"joints": jd}, {"joints": {**jd, "category": "B"}}) == "assess", "kategorie spoje = jen posouzení")
        check(lvl({"joints": jd}, {"joints": {**jd, "preload_case": not jd["preload_case"]}}) == "fe",
              "stav předpětí = výpočet MKP (síť platí)")
        check(lvl({"joints": jd}, {"joints": {**jd, "bolt_mesh": 0.5}}) == "mesh", "síť u šroubů = nová síť")
        check(lvl({"joints": jd}, {"joints": {**jd, "gamma_M3": 1.3, "weld_color_tol": 0.05}}) == "assess",
              "γ_M3 a tolerance barvy svarů = jen posouzení (fea_core.joints.levels.SETTINGS_ASSESS)")
        check(lvl({"bolts": [b]}, {"bolts": [{**b, "mu_T": 0.2}]}) == "assess", "μ_T šroubu = jen posouzení")
        check(lvl({"bolts": [b]}, {"bolts": [{**b, "strength": "10.9"}]}) == "fe", "třída šroubu = výpočet MKP")
        check(lvl({"bolts": [b]}, {"bolts": []}) == "mesh", "smazaný šroub = nová síť")
        mesh = w.mesh
        fake = Results(mesh=mesh, steps=[])               # empty results (the GUI reads also twod, shell, modes…)
        w.results = fake
        w.study.joints.category = "B"
        w.push_undo()
        w.undo()
        check(w.results is fake and w.mesh is mesh and w.valid_mesh() is mesh and w.study.joints.category == "A",
              "zpět u posudkového pole spojů: výsledky i síť zůstanou")
        w.redo()
        check(w.results is fake and w.study.joints.category == "B", "znovu: výsledky zůstanou")
        w.study.joints.preload_case = not w.study.joints.preload_case
        w.push_undo()
        w.undo()
        check(w.results is None and w.valid_mesh() is mesh, "zpět u stavu předpětí: výsledky pryč, síť platí")
        w.study.joints.bolt_mesh = 0.5
        w.push_undo()
        w.undo()
        check(w.mesh is None and w.view.parts_visible and not w._mesh_view,
              "zpět u síťování šroubů: síť pryč, pohled na díly")
        w.make_mesh()

    def s_mesh_settings():
        check(w._mesh_view and not w.view.parts_visible, "nová síť zobrazena")
        w.ms_size.setValue(5.0)                          # a change of the settings discards the mesh
        check(w.mesh is None and not w._mesh_view and w.view.parts_visible,
              "změna nastavení sítě → pohled zpět na díly (ne stará síť)")

    def s_view():                                        # 8, 9
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces([])
        m = w.build_view_menu()
        texts = [a.text() for a in m.actions()]
        check(any(t.startswith(("Fit all", "Show all")) for t in texts) and any(t.startswith("Zoom to selection") for t in texts)
              and any(t in ("View", "Pohled") for t in texts), f"místní nabídka 3D okna: {texts}")
        w.view.set_orientation("iso")
        app.processEvents()
        c = (50.0, 10.0, 10.0)                           # centre of the top face of the plate
        xp, yp = w.view.view.Convert(*c)
        r = w.view.dpr()
        pt = QPoint(int(xp / r), int(yp / r))
        QTest.mouseMove(w.view, pt)
        QTest.mouseDClick(w.view, Qt.LeftButton, Qt.NoModifier, pt)
        app.processEvents()
        sel = w.view.selected_faces()
        check(len(sel) == 1, f"dvojklik na plochu ji vybere (a nechá vybranou): {sel}")
        scale = w.view.view.Camera().Scale()
        w.view.fit_all()
        check(w.view.view.Camera().Scale() > scale * 1.01, "dvojklik přiblížil výběr (vše = menší měřítko)")
        if sys.platform == "win32":
            check(w.spacemouse is not None and w.spacemouse.registered, "3D myš napojená (raw input registrován)")

    def s_open_json():                                   # studie JSON (REST /api/open-gui, MCP fea_study_save)
        import os
        from fea_core.study import Study
        cp = Study.from_dict(w.study.to_dict())
        cp.geometry = os.path.relpath(step, OUT)         # STEP next to the JSON, the path relative to the file of the study
        cp.save(OUT / "studie.json")
        w.dirty = False
        w.open_file(str(OUT / "studie.json"))
        check(w.study is not None and len(w.parts) == 3 and len(w.study.constraints) == len(cp.constraints)
              and [lc.name for lc in w.study.load_cases] == [lc.name for lc in cp.load_cases],
              "studie JSON otevřena (díly, vazby, zatěžovací stavy)")
        check(w.project_path is None and w.step_path == step.resolve(),
              "studie JSON: STEP vedle souboru, Uložit = nový projekt .fea (JSON se nepřepíše)")
        meta = OUT / "metadata.json"
        meta.write_text('{"source": "Test", "parts": []}', encoding="utf-8")
        before = len(MSGS)
        w.dirty = False
        w.open_file(str(meta))
        check(len(MSGS) > before and any(x in MSGS[-1] for x in ("CAD metadata", "metadata z CAD")) and w.study.name == cp.name,
              "metadata z CADu se jako studie neotevřou (hláška, otevřená studie zůstane)")

    steps = [s_open, s_materials, s_constraints, s_gravity, s_update_load, s_delete_lc, s_rigid, s_mesh,
             s_mesh_view, s_joints_undo, s_mesh_settings, s_view, s_open_json]

    def run_next():
        if w._thread is not None:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            w.dirty = False                              # quit() in Qt 6 closes the windows – without a question about saving
            app.quit()
            return
        f = steps.pop(0)
        try:
            f()
        except Exception:
            import traceback
            traceback.print_exc()
            FAIL.append(f.__name__)
            w.dirty = False
            app.quit()
            return
        QTimer.singleShot(300, run_next)

    QTimer.singleShot(500, run_next)
    app.exec()
    print("VÝSLEDEK:", "OK" if not FAIL else f"{len(FAIL)} chyb: " + "; ".join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
