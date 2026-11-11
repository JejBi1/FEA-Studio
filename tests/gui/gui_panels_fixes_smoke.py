"""Fixes of the Joints, Heat, Fatigue panels and of the protocol dialog (checked programmatically, without snapshots).

  python tests/gui/gui_panels_fixes_smoke.py OUT_DIR

1. Example examples/example2_assembly.fea (bolts, welds, heat, fatigue, printing – without results):
   - Joints: _filling returns after an exception when filling the form (try/finally); an unknown filler of a weld
     is shown in the menu only temporarily; the description of the automatic preload.
   - Heat: the time / step / frames / temperatures are written only after finishing the edit (editingFinished), without a change
     nothing is discarded; a value set without focus (script) is written at once.
   - Fatigue: default "Cycles in the block" = 1; the S-N materials only of the meshed parts (without bolts and excluded ones),
     a part without a material does not crash.
   - Protocol: the name according to the kind of the calculation (structural / thermal / both), the option "Keep images" → keep_images.
2. The decision function joints_panel.change_level (mesh / FEM calculation / assessment only).
3. A small assembly (a lap joint M12 + a T-joint with welds), calculation; then a change of the FAT of the weld and μ_T of the bolt
   keeps the mesh and the results and only recalculates the assessment; the preload discards the results (the mesh stays); the weld leg the mesh.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_panels_fixes_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL = []


def check(cond, text):
    print(("OK   " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def make_step(path):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf, gp_Vec
    from fea_core.geometry import Part, write_step
    from verification import joint_models as jm
    parts = jm.lap_joint(d=12.0, nbolts=1)
    tr = gp_Trsf()
    tr.SetTranslation(gp_Vec(50.0, 100.0, 0.0))
    for p in jm.t_joint_fillet():
        parts.append(Part(p.name, BRepBuilderAPI_Transform(p.shape, tr, True).Shape()))
    write_step(parts, path)


def test_change_level():
    from fea_core.joints.model import BoltSpec, WeldSpec
    from fea_gui.joints_panel import change_level
    b = BoltSpec("Š")
    w = WeldSpec("S", "Weld")

    def lv(kind, spec, pc=True, **kw):
        new = spec.to_dict()
        new.update(kw)
        return change_level(kind, spec.to_dict(), new, pc)
    check(lv("bolt", b) == "", "šroub beze změny → nic")
    check(lv("bolt", b, mu_T=0.2) == "assess", "μ_T → jen posouzení")
    check(lv("bolt", b, thread_in_shear=False) == "assess", "střih v závitu → jen posouzení")
    check(lv("bolt", b, shank_d=13.0) == "assess", "dřík lícovaného → jen posouzení")
    check(lv("bolt", b, preload_mode="force", preload=20000.0) == "fe", "předpětí se stavem předpětí → MKP")
    check(lv("bolt", b, False, preload_mode="force", preload=20000.0) == "assess",
          "předpětí bez stavu předpětí → jen posouzení")
    check(lv("bolt", b, alpha_A=1.4) == "fe" and lv("bolt", b, mu_G=0.1) == "fe", "α_A, μ_G (F_V) → MKP")
    check(lv("bolt", b, strength="10.9") == "fe", "třída (E, F_M,zul) → MKP")
    check(lv("bolt", b, size=12.0) == "mesh", "velikost → síť")
    check(lv("bolt", b, enabled=False) == "mesh", "vypnutí šroubu → síť")
    check(lv("bolt", b, origin=[1.0, 0.0, 0.0]) == "mesh", "neznámé / geometrické pole → síť (opatrně)")
    bd = BoltSpec("Š", enabled=False)
    check(lv("bolt", bd, strength="10.9") == "assess", "vypnutý šroub – úprava nemění model MKP")
    for k, v in (("fat_sigma", 50.0), ("fat_tau", 71.0), ("sn_curve", "IIW"), ("gamma_Mf", 1.35), ("toe", False),
                 ("fat_toe", 100.0), ("filler", "G 38")):
        check(lv("weld", w, **{k: v}) == "assess", f"svar {k} → jen posouzení")
    check(lv("weld", w, segments=20) == "fe" and lv("weld", w, toe_t=8.0) == "fe", "úseky, t paty → výpočet MKP")
    for k, v in (("a", 4.0), ("open_root", False), ("enabled", False), ("part", "Jiný")):
        check(lv("weld", w, **{k: v}) == "mesh", f"svar {k} → síť")


def main():
    import fea_gui.main as gm
    import fea_gui.joints_panel as jpm
    from fea_gui.main import MainWindow
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")      # without modal windows
    QMessageBox.warning = staticmethod(lambda parent, title, text, *a: print(f"[{title}] {text}"))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    jpm.exec_dialog = lambda dlg: True
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1500, 950)
    w.show()
    jp, tp, fp = w.joints_panel, w.thermal_panel, w.fat_panel
    calls = {"changed": 0}
    orig_changed = w.changed

    def counting_changed(*a, **k):
        calls["changed"] += 1
        return orig_changed(*a, **k)
    w.changed = counting_changed
    keep_settings = w.settings.value("report/keep_images", None)

    # ------------------------------------------------------------------ example
    def s_example():
        w.open_file(str(ROOT / "examples" / "example2_assembly.fea"))
        st = w.study
        check(len(st.bolts) >= 1 and len(st.welds) >= 2 and len(st.thermal.cases) >= 2, "ukázka 2 načtena")
        test_change_level()
        # --- Joints: description of the automatic preload
        txt = jp.b_mode.itemText(0)
        check("90 % Rp0,2" not in txt and "pressure" in txt, f"text automatického předpětí: „{txt}“")
        tip = jp.b_mode.toolTip() + jp.b_mode.whatsThis()                 # short label + details (F11)
        check("p_G" in tip and "min(" in tip, "nápověda automatického předpětí")
        # --- Joints: _filling after an exception (try/finally)
        jp.bolt_list.setCurrentRow(0)
        orig = jp.b_alpha.setValue
        jp.b_alpha.setValue = lambda *_: (_ for _ in ()).throw(RuntimeError("test"))
        try:
            jp.on_bolt_row(0)
        except RuntimeError:
            pass
        jp.b_alpha.setValue = orig
        check(jp._filling is False, "on_bolt_row: _filling vrácen i po výjimce")
        orig = jp.w_gamma.setValue
        jp.w_gamma.setValue = lambda *_: (_ for _ in ()).throw(RuntimeError("test"))
        try:
            jp.on_weld_row(0)
        except RuntimeError:
            pass
        jp.w_gamma.setValue = orig
        check(jp._filling is False, "on_weld_row: _filling vrácen i po výjimce")
        # --- Joints: an unknown filler only temporarily
        n0 = jp.w_filler.count()
        st.welds[0].filler = "XY 99 (neznámý)"
        jp.weld_list.setCurrentRow(1)
        jp.weld_list.setCurrentRow(0)
        check(jp.w_filler.count() == n0 + 1 and jp.w_filler.currentText().startswith("unknown: XY 99"),
              f"neznámý přídavný materiál zobrazen dočasně ({jp.w_filler.currentText()})")
        jp.weld_list.setCurrentRow(1)
        check(jp.w_filler.count() == n0, f"po výběru jiného svaru nabídka bez neznámé položky ({jp.w_filler.count()})")
        jp.weld_list.setCurrentRow(0)
        jp.on_weld_row(0)
        jp.on_weld_row(0)
        check(jp.w_filler.count() == n0 + 1, "opakovaný výběr – neznámá položka jen jednou")
        undo0 = len(w._undo)
        jp.save_weld()                                   # unchanged → the name stays, nothing is discarded
        check(st.welds[0].filler == "XY 99 (neznámý)" and len(w._undo) == undo0,
              "uložení svaru s neznámým materiálem: název zachován, beze změny studie")
        st.welds[0].filler = ""
        jp.weld_list.setCurrentRow(1)

        # --- Heat: writing after finishing the edit
        w.set_stage(w.STAGE["Heat"])
        ci = next(i for i, c in enumerate(st.thermal.cases) if c.transient)
        tp.case_combo.setCurrentIndex(ci)
        case = st.thermal.cases[ci]
        t0, n_out0 = case.time, case.outputs
        tp.t_time.hasFocus = lambda: True                # the user types into the field (without a dependence on the active window)
        tp.t_out.hasFocus = lambda: True
        tp.t_ref.hasFocus = lambda: True
        c0, u0 = calls["changed"], len(w._undo)
        for v in (t0 + 1, t0 + 2, t0 + 3):
            tp.t_time.setValue(v)                        # arrow step / a typed character
        tp.t_out.setValue(n_out0 + 5)
        check(calls["changed"] == c0 and case.time == t0 and len(w._undo) == u0,
              f"při psaní se studie nemění (changed {calls['changed'] - c0}×, doba {case.time})")
        tp.t_time.editingFinished.emit()
        check(case.time == t0 + 3 and case.outputs == n_out0 + 5 and calls["changed"] == c0 + 1,
              f"po dokončení úpravy zapsáno jednou (doba {case.time}, snímky {case.outputs})")
        c1, u1 = calls["changed"], len(w._undo)
        tp.t_time.editingFinished.emit()
        tp.t_out.editingFinished.emit()
        check(calls["changed"] == c1 and len(w._undo) == u1, "editingFinished beze změny nic nezahodí")
        r0 = st.thermal.reference_temperature
        tp.t_ref.setValue(r0 + 7)
        check(st.thermal.reference_temperature == r0, "referenční teplota se při psaní nezapíše")
        tp.t_ref.editingFinished.emit()
        check(st.thermal.reference_temperature == r0 + 7, "referenční teplota po dokončení úpravy")
        for wd in (tp.t_time, tp.t_out, tp.t_ref):
            del wd.hasFocus
        tp.t_dt.setValue(case.dt + 1.0 if case.dt else 2.0)   # without focus (script) → at once
        check(case.dt == tp.t_dt.value(), f"hodnota nastavená skriptem se zapíše hned (dt {case.dt})")

        # --- Fatigue: default cycles and S-N materials
        check(fp.s_cyc.text() == "1", f"výchozí počet kmitů v bloku: {fp.s_cyc.text()}")
        from fea_core.materials import Material
        bolt_part = st.active_bolts()[0].cad_parts[0]
        st.materials["Zkušební šroubová ocel"] = Material("Zkušební šroubová ocel", 210000.0, 0.3, 7.85e-9,
                                                           Re=640.0, Rm=800.0)
        st.part_materials[bolt_part] = "Zkušební šroubová ocel"
        other = next(p.name for p in w.parts if p.name not in st.excluded_parts() and p.name != "Arm (PLA print)")
        st.materials["Zkušební vynechaná"] = Material("Zkušební vynechaná", 70000.0, 0.33, 2.7e-9, Re=200.0, Rm=300.0)
        st.part_materials[other] = "Zkušební vynechaná"
        st.ignored_parts = [other]
        fp.refresh()
        mats = [fp.sn_mat.itemText(i) for i in range(fp.sn_mat.count())]
        check("Zkušební šroubová ocel" not in mats and "Zkušební vynechaná" not in mats and mats,
              f"S-N jen pro síťované díly: {mats}")
        st.part_materials[bolt_part] = "neexistuje"      # a bolt part with an invalid material
        st.ignored_parts = []
        st.part_materials[other] = "neexistuje"          # a meshed part without a material → skipped
        try:
            fp.refresh()
            ok = True
        except KeyError as e:
            ok = False
            print(e)
        check(ok, "díl bez materiálu nespadne (KeyError)")
        res_fake = SimpleNamespace(mesh=SimpleNamespace(part_names=["Rameno (tisk PLA)"]), steps=[], thermal=[])
        w.results = res_fake
        check(fp.fatigue_materials() == ["Tisk PLA Bambu Basic"], f"s výsledky díly sítě: {fp.fatigue_materials()}")
        w.results = None

        # --- Protocol: the name and keeping the images
        from fea_gui.report_dialog import ReportDialog, default_title
        from fea_core.study import Study
        check(default_title(st, None).startswith("FE structural and thermal analysis"),
              f"název ukázky: {default_title(st, None)}")
        th = Study.from_dict(st.to_dict())
        for lc in th.load_cases:
            lc.loads = []
        check(default_title(th, None).startswith("FE thermal analysis"), f"jen teplo: {default_title(th, None)}")
        check(default_title(th, SimpleNamespace(steps=[])).startswith("FE thermal analysis"),
              "jen teplo s výsledky (bez mechanických kroků)")
        me = Study.from_dict(st.to_dict())
        me.thermal.cases = []
        check(default_title(me, None).startswith("FE structural analysis"), f"jen pevnost: {default_title(me, None)}")
        w.study = th
        dlg = ReportDialog(w)
        check(dlg.title.text().startswith("FE thermal analysis"), f"dialog – název: {dlg.title.text()}")
        w.study = st
        dlg = ReportDialog(w)
        dlg.path.setText(str(OUT / "zkouska.docx"))
        check("zkouska_obrazky" in dlg.keep_images.text(), f"popis volby: {dlg.keep_images.text()}")
        dlg.keep_images.setChecked(True)
        o = dlg.options()
        check(o.keep_images is True, "keep_images = True")
        check(ReportDialog(w).keep_images.isChecked(), "volba zapamatována (QSettings)")
        dlg.keep_images.setChecked(False)
        check(dlg.options().keep_images is False, "keep_images = False")
        dlg.keep_images.setChecked(True)
        dlg.images.setChecked(False)
        check(dlg.options().keep_images is False and not dlg.keep_images.isEnabled(), "bez obrázků nic neponechat")
        w.dirty = False

    # ------------------------------------------------------------------ small assembly: calculation
    step = OUT / "spoje_maly.step"

    def s_model():
        make_step(step)
        w.open_file(str(step))
        w.study.mesh.size = 6.0
        w.set_stage(w.STAGE["Joints"])
        jp.detect()
        st = w.study
        check(len(st.bolts) == 1 and len(st.welds) == 2, f"rozpoznáno {len(st.bolts)} šroubů, {len(st.welds)} svarů")
        from fea_core.geometry import face_props

        def faces_where(pred):
            return [(pi, fi) for pi, faces in enumerate(w.view.part_faces) for fi, f in enumerate(faces)
                    if pred(w.parts[pi].name, face_props(f))]

        def plane(part, ax, val):
            return faces_where(lambda n, p: n == part and abs(p["bbox"][ax] - val) < 1e-6
                               and abs(p["bbox"][ax + 3] - val) < 1e-6)
        w.set_stage(w.STAGE["Constraints"])
        xb = max(p["bbox"][3] for p in (face_props(f) for f in w.view.part_faces[[x.name for x in w.parts].index("Deska B")]))
        w.view.set_selected_faces(plane("Deska B", 0, xb) + plane("Pasnice", 2, -10.0), emit=True)
        w.add_fixed()
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces(plane("Deska A", 0, 0.0), emit=True)
        w.ld_type.setCurrentText("Force (components)")
        for s, v in zip(w.ld_vec, (-8000.0, 0.0, 0.0)):
            s.setValue(v)
        w.add_load()
        w.view.set_selected_faces(plane("Stojina", 2, 100.0), emit=True)
        for s, v in zip(w.ld_vec, (0.0, 0.0, 20000.0)):
            s.setValue(v)
        w.add_load()
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_after_solve():
        st = w.study
        res, mesh = w.results, w.mesh
        check(res is not None and mesh is not None and w.valid_mesh() is not None, "výpočet hotov, síť platná")
        if res is None:
            return
        w.set_stage(w.STAGE["Joints"])
        a0 = jp.assessment
        check(a0 is not None, "posouzení spojů po výpočtu")
        # FAT of the weld + filler → only the assessment
        jp.weld_list.setCurrentRow(0)
        jp.w_fs.setValue(50.0)
        jp.w_gamma.setValue(1.35)
        jp.w_filler.setCurrentIndex(1)
        jp.save_weld()
        check(st.welds[0].fat_sigma == 50.0 and st.welds[0].gamma_Mf == 1.35 and st.welds[0].filler,
              "svar: FAT, γ_Mf a přídavný materiál uloženy")
        check(w.results is res and w.mesh is mesh and w.valid_mesh() is mesh,
              "svar (jen posouzení): výsledky i síť zachovány")
        check(jp.assessment is not None and jp.assessment is not a0, "posouzení přepočítáno")
        check(w.dirty and w._undo[w._undo_pos] == json.dumps(st.to_dict()), "změna je ve snímku zpět/znovu")
        # μ_T of the bolt → only the assessment
        a1 = jp.assessment
        jp.bolt_list.setCurrentRow(0)
        jp.b_muT.setValue(0.25)
        jp.save_bolt()
        check(st.bolts[0].mu_T == 0.25 and w.results is res and w.valid_mesh() is mesh and jp.assessment is not a1,
              "šroub μ_T: výsledky i síť zachovány, posouzení znovu")
        # setting of the assessment (category) → the mesh stays valid
        jp.s_cat.setCurrentIndex(jp.s_cat.findData("B"))
        check(st.joints.category == "B" and w.results is res and w.valid_mesh() is mesh,
              "kategorie šroubů: výsledky i síť zachovány")
        # preload (the preload state switched on) → a new calculation, the mesh valid
        check(st.joints.preload_case, "stav předpětí zapnut")
        jp.bolt_list.setCurrentRow(0)
        jp.b_mode.setCurrentIndex(jp.b_mode.findData("force"))
        jp.b_pre.setValue(30000.0)
        jp.save_bolt()
        check(w.results is None and w.mesh is mesh and w.valid_mesh() is mesh,
              "předpětí: výsledky zahozeny, síť zachována")
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()                                    # with the mesh kept

    def s_resolve():
        st = w.study
        res = w.results
        check(res is not None and len(res.steps) == len(st.load_cases) + 1, "nový výpočet se zachovanou sítí")
        if res is None:
            return
        bolt = (res.info.get("joints") or {}).get("bolts") or {}
        print("předpětí – kroky:", [s.name for s in res.steps], "; šrouby:", list(bolt))
        w.set_stage(w.STAGE["Joints"])
        c = next((x for x in jp.assessment.bolts), None) if jp.assessment else None
        fm = c.data.get("basics", {}).get("F_Mmax") if c else None
        check(fm is not None and abs(fm - 30000.0) < 1e-6, f"posouzení se zadaným předpětím (F_M,max {fm})")
        pre = res.steps[-1]
        check(pre.name == "Bolt preload", f"poslední krok = předpětí ({pre.name})")
        # weld leg → a new mesh
        jp.weld_list.setCurrentRow(0)
        jp.weld_list.setCurrentRow(0)
        jp.w_a.setValue(4.0)
        jp.save_weld()
        check(w.mesh is None and w.valid_mesh() is None, "výška svaru: síť zahozena")
        w.dirty = False

    steps = [s_example, s_model, s_after_solve, s_resolve]

    def run_next():
        if w._thread is not None:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            print("VÝSLEDEK:", "OK" if not FAIL else f"{len(FAIL)} CHYB: {FAIL}")
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
        QTimer.singleShot(500, run_next)

    QTimer.singleShot(800, run_next)
    app.exec()
    if keep_settings is None:
        w.settings.remove("report/keep_images")
    else:
        w.settings.setValue("report/keep_images", keep_settings)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
