"""Fixes of the 3D printing step in the GUI (without a calculation, ~20 s):

  python tests/gui/gui_print_fixes_smoke.py [OUT_DIR]

1. the material form keeps the fields outside the form (Re, λ, α, c_p) and the unchanged values without rounding of the spin box,
2. the assignment of a material does not delete the other materials of the study (calibrated, material of a variant),
3. calibration: the infill template in Czech (the key as data), the line ratios from the override in the study (props_of),
4. Advanced: bins / samples / line_bins → study via changed() (undo/redo), nothing without a change,
5. the arrow of the build direction at the selected printed part, "Build direction from face" (selection by a click and an existing selection).
Model: examples/example1_bracket.fea (printed bracket, material Tisk PLA Bambu Basic).
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_print_fixes_out")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")   # without modal windows
    infos = []
    QMessageBox.information = staticmethod(lambda *a, **k: infos.append(a[2] if len(a) > 2 else "") or 0)
    QMessageBox.warning = staticmethod(lambda *a, **k: infos.append(a[2] if len(a) > 2 else "") or 0)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1500, 950)
    w.show()
    src = OUT / "drzak.fea"
    shutil.copy(ROOT / "examples" / "example1_bracket.fea", src)
    failed = []

    def run():
        try:
            checks()
            print("VŠE OK")
        except Exception:
            import traceback
            traceback.print_exc()
            failed.append(1)
        w.dirty = False
        app.quit()

    def flush():
        for _ in range(3):
            app.processEvents()

    def checks():
        from dataclasses import replace
        from fea_core.geometry import face_props
        from fea_core.printing import PrintVariant
        from fea_gui.print_panel import BUILD_KEY, CalibrationDialog
        w.open_file(str(src))
        flush()
        st = w.study
        pp = w.print_panel
        names = [p.name for p in w.parts]
        part = next(n for n in names if st.material_of(n).is_print)
        pidx = names.index(part)
        w.set_stage(w.STAGE["3D printing"])
        flush()

        # ------------------------------------------------------------ 1. Re and the thermal fields survive the editing
        base_name = st.material_of(part).name
        st.materials[base_name] = replace(st.materials[base_name], Re=33.0, k=0.13, alpha=6.8e-5, cp=1800.0,
                                          nu=0.3555)
        w.changed()
        pp.parts.clearSelection()
        pp.parts.item(pidx).setSelected(True)
        pp.mat.setCurrentText(base_name)
        pp.on_mat_combo()
        pp.m["E"].setValue(2500.0)
        pp.save_material()
        m = w.study.materials[base_name]
        print("1:", m.E, m.Re, m.k, m.alpha, m.cp, m.nu)
        assert m.E == 2500.0 and m.Re == 33.0 and (m.k, m.alpha, m.cp) == (0.13, 6.8e-5, 1800.0)
        assert m.nu == 0.3555, "nezměněné ν se nesmí zaokrouhlit na 3 desetinná místa spinu"
        assert w.study.material_of(part).name == base_name

        # ------------------------------------------------------------ 2. the assignment does not delete the other materials
        st = w.study
        st.materials["Kalibrovaný X"] = replace(m, name="Kalibrovaný X")         # without parts (calibration)
        st.materials["Varianta Y"] = replace(m, name="Varianta Y")
        st.printing.variants = [PrintVariant("V1", {part: {"material": "Varianta Y", "walls": 3}})]
        w.changed()
        pp.parts.clearSelection()
        pp.parts.item(pidx).setSelected(True)
        pp.mat.setCurrentText("Tisk PLA Bambu Basic")
        pp.assign_material()
        st = w.study
        print("2:", sorted(st.materials), "→", st.material_of(part).name)
        assert {"Kalibrovaný X", "Varianta Y", base_name} <= set(st.materials)
        assert pp.mat.findText("Kalibrovaný X") >= 0, "kalibrovaný materiál musí zůstat v nabídce"

        # ------------------------------------------------------------ 3. calibration: the infill template in Czech, props from the study
        from fea_core.printing.calibrate import TensileTest, calibrate
        from fea_core.printing.infill import pattern_label
        from fea_core.printing.props import default_props, props_of
        bm = st.materials["Kalibrovaný X"]
        st.printing.materials["Kalibrovaný X"] = {"line_E_ratio": 0.5, "line_R_ratio": 0.5}
        dlg = CalibrationDialog(w, st, bm, pp.spec_from_ui(""))
        dlg.base.setCurrentText("Kalibrovaný X")
        tests = [TensileTest("XY-1", "xy", E=2600, Rm=40, walls=2, infill=0.2, pattern="gyroid"),
                 TensileTest("Z-1", "z", E=2100, Rm=22, infill=1.0)]
        for t in tests:
            dlg.add_row(t)
        cb = dlg.table.cellWidget(0, [c[0] for c in dlg.COLS].index("pattern"))
        print("3: vzor v tabulce:", cb.currentText(), "| data:", cb.currentData())
        assert cb.currentText() == pattern_label("gyroid") != "gyroid" and cb.currentData() == "gyroid"
        assert dlg.tests()[0].pattern == "gyroid"
        assert dlg.compute(), dlg.out.toPlainText()
        exp = calibrate(bm, dlg.tests(), dlg.name.text(), props=props_of(st, bm)).material
        dflt = calibrate(bm, dlg.tests(), dlg.name.text(), props=default_props(bm.name)).material
        print("3: E_xy", dlg.result_material.E, "(props studie", exp.E, ", výchozí", dflt.E, ")")
        assert abs(dlg.result_material.E - exp.E) < 1e-9 and abs(exp.E - dflt.E) > 1.0
        dlg.close()

        # ------------------------------------------------------------ 4. Advanced
        assert not pp.adv.isVisible()
        pp.adv_toggle.setChecked(True)
        flush()
        n_undo = len(w._undo)
        pp.a_bins.setValue(12)
        pp.a_samples.setValue(48)
        pp.a_line_bins.setValue(18)
        pp.on_advanced()
        ps = w.study.printing
        print("4:", ps.bins, ps.samples, ps.line_bins, "| zpět:", n_undo, "→", len(w._undo))
        assert (ps.bins, ps.samples, ps.line_bins) == (12, 48, 18) and len(w._undo) == n_undo + 1
        pp.on_advanced()                                   # unchanged → no new snapshot
        assert len(w._undo) == n_undo + 1
        w.undo()
        flush()
        ps = w.study.printing
        assert (ps.bins, ps.samples, ps.line_bins) == (10, 32, 36), (ps.bins, ps.samples, ps.line_bins)
        assert (w.print_panel.a_bins.value(), w.print_panel.a_line_bins.value()) == (10, 36)
        w.redo()
        flush()
        assert w.study.printing.bins == 12
        for txt in ("1/bins", "180°/line_bins"):
            assert any(txt in s.toolTip() + s.whatsThis() for s in (pp.a_bins, pp.a_samples, pp.a_line_bins)), txt

        # ------------------------------------------------------------ 5. build direction
        v = w.view
        pp.parts.clearSelection()
        pp.parts.item(pidx).setSelected(True)
        flush()
        assert BUILD_KEY in v._sets and v._sets[BUILD_KEY], "šipka směru stavby chybí"
        pp.parts.clearSelection()
        flush()
        assert BUILD_KEY not in v._sets, "po zrušení výběru dílu má šipka zmizet"
        props = [face_props(f) for f in v.part_faces[pidx]]
        planar = [i for i, p in enumerate(props) if p["type"] == "plane"]
        curved = [i for i, p in enumerate(props) if p["type"] != "plane"]
        bottom = min(planar, key=lambda i: props[i]["centroid"] @ np.array([0, 0, 1.0]) + 10 * props[i]["normal"][2])
        side = next(i for i in planar if abs(abs(props[i]["normal"][2]) - 1) > 1e-6)
        # a) selection by a click: the button without a selected face starts the selection
        w.study.printing.parts[part] = replace(w.study.printing.spec(part), part=part, walls=4)
        w.changed()
        pp.parts.item(pidx).setSelected(True)
        pp.btn_face_dir.click()
        assert pp._picking and v.mode == "faces" and pp.face_hint.text()
        if curved:                                         # non-planar face: a message, the selection goes on
            v.set_selected_faces([(pidx, curved[0])], emit=True)
            assert pp._picking and "not planar" in pp.face_hint.text()
        v.set_selected_faces([(pidx, side)], emit=True)
        flush()
        exp = -np.asarray(props[side]["normal"]) / np.linalg.norm(props[side]["normal"])
        got = np.asarray(w.study.printing.spec(part).build_dir)
        print("5a: normála", props[side]["normal"], "→ směr stavby", got)
        assert not pp._picking and v.mode == "model" and np.allclose(got, exp, atol=1e-8)
        assert BUILD_KEY in v._sets
        # the other print settings of the part did not change
        assert w.study.printing.spec(part).walls == 4
        # b) a face already selected (face mode) → at once
        v.set_mode("faces")
        v.set_selected_faces([(pidx, bottom)])
        pp.btn_face_dir.click()
        got = w.study.printing.spec(part).build_dir
        print("5b: spodní plocha → směr stavby", got, "| formulář:", pp.s_dir.currentText())
        assert np.allclose(got, -np.asarray(props[bottom]["normal"]), atol=1e-8)
        if np.allclose(got, [0, 0, 1]):
            assert got == [0.0, 0.0, 1.0] and pp.s_dir.currentIndex() == 0
        # c) a non-planar selected face → a Czech message
        if curved:
            v.set_mode("faces")
            v.set_selected_faces([(pidx, curved[0])])
            infos.clear()
            pp.btn_face_dir.click()
            assert infos and any(x in infos[0] for x in ("not planar", "není rovinn"))  and not pp._picking
            v.set_mode("model")
        # leaving the step → the arrow disappears
        w.set_stage(w.STAGE["Constraints"])
        flush()
        assert BUILD_KEY not in v._sets, "šipka směru stavby má mimo krok 3D tisk zmizet"
        w.set_stage(w.STAGE["3D printing"])
        flush()

        # ------------------------------------------------------------ 6., 7. help
        for s in (pp.s_layer, pp.s_lw, pp.s_walls, pp.s_infill):
            assert s.toolTip(), "chybí nápověda"
        from fea_core.printing.presets import BUILTIN_NAME
        assert pp.preset.findData(BUILTIN_NAME) >= 0, "vestavěná předvolba chybí v nabídce"
        assert "Apply to selected parts" in pp.btn_preset_fill.toolTip() + pp.btn_preset_fill.whatsThis()
        assert pp.s_solid_thin.toolTip() and pp.s_thin_w.toolTip()

        pp.parts.clearSelection()
        pp.parts.item(pidx).setSelected(True)                # arrow of the build direction in 3D (snapshot for a check)
        flush()
        v.set_orientation("iso")
        v.redraw()
        flush()
        v.screenshot(str(OUT / "smer_stavby_3d.png"))
        w.grab().save(str(OUT / "fixes_panel.png"))

    QTimer.singleShot(500, run)
    app.exec()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
