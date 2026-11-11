"""A pass through the GUI for heat, with snapshots (a manual check of the look).

  python tests/gui/gui_thermal_smoke.py OUT_DIR

Model: steel bar 100×20×20. Material with λ, α, c_p from the form of the Materials step; thermal state "Heating"
(x = 0 → 100 °C, the face x = 100 convection h = 500 W/(m²·K) to 20 °C, sides insulated) and a transient "Ramp-up".
First a calculation of the HEAT ONLY (without constraints – the empty load state is left out), then the thermal stress: a clamping
x = 0, a state with the field from "Heating" and a state with a uniform temperature 80 °C. Analytically (1D): q = 80 / (L/λ + 1/h)
= 20 000 W/m², T(x = 100) = 60 °C. Saves and reopens the project.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_thermal_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL = []


def check(cond, text):
    print(("OK   " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def make_step(path):
    from fea_core.geometry import Part, write_step
    from verification.cases import box
    write_step([Part("Tyč", box(0, 0, 0, 100, 20, 20))], path)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow, THERMAL_LOAD, QUANTITIES, THERMAL_QUANTITIES
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")      # without modal windows
    QMessageBox.warning = staticmethod(lambda parent, title, text, *a: print(f"[{title}] {text}"))
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    step = OUT / "tyc.step"
    make_step(step)
    log = []
    tp = w.thermal_panel

    def shot(name):
        app.processEvents()
        w.view.redraw()
        app.processEvents()
        # the 3D view is rendered by OCCT (Dump) and inserted into the snapshot of the widgets (not a screenshot)
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
                if pred(face_props(f)):
                    out.append((pi, fi))
        return out

    def plane_x(x):
        return faces_where(lambda p: abs(p["bbox"][0] - x) < 1e-6 and abs(p["bbox"][3] - x) < 1e-6)

    def click_at(xyz):
        """A real mouse click on the projection of a point of the model."""
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        xp, yp = w.view.view.Convert(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        r = w.view.dpr()
        pt = QPoint(int(xp / r), int(yp / r))
        QTest.mouseMove(w.view, pt)
        QTest.mouseClick(w.view, Qt.LeftButton, Qt.NoModifier, pt)
        app.processEvents()

    def select_result(text):
        i = w.r_step.findText(text)
        check(i >= 0, f"výsledek '{text}' v nabídce ({[w.r_step.itemText(k) for k in range(w.r_step.count())]})")
        w.r_step.setCurrentIndex(i)

    def select_qty(name):
        names = [w.r_qty.itemText(k).split(" [")[0] for k in range(w.r_qty.count())]
        check(name in names, f"veličina '{name}' v nabídce")
        w.r_qty.setCurrentIndex(names.index(name))

    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")
        w.study.mesh.size = 4.0
        # order Constraints → Motion (F8) → Heat → Loads
        check(w.STAGE["Constraints"] < w.STAGE["Heat"] == w.STAGE["Loads"] - 1, "krok Teplo za Vazbami, před Zatížením")
        check(w.panel_stack.widget(w.STAGE["Heat"]) is tp, "panel Teplo na indexu kroku")

    def s_material():
        w.set_stage(w.STAGE["Materials"])
        w.mat_combo.setCurrentText("Ocel S235")
        check(abs(w.m_k.value() - 53.3) < 1e-9 and abs(w.m_alpha.value() - 12.0) < 1e-9,
              f"knihovní materiál: λ {w.m_k.value()}, α {w.m_alpha.value()}·10⁻⁶, c_p {w.m_cp.value()}")
        w.m_name.setText("Ocel (teplo)")
        w.m_k.setValue(50.0)
        w.m_alpha.setValue(12.0)
        w.m_cp.setValue(460.0)
        w.assign_material(True)
        m = w.study.material_of("Tyč")
        check((m.name, m.k, m.alpha, m.cp) == ("Ocel (teplo)", 50.0, 12e-6, 460.0),
              f"materiál ve studii: {m.name}, k {m.k}, α {m.alpha}, c_p {m.cp}")
        w.mat_combo.setCurrentText("Ocel (teplo)")
        shot("t1_materialy")

    def s_thermal():
        w.set_stage(w.STAGE["Heat"])
        tp.new_case("Heating")
        w.view.set_selected_faces(plane_x(0.0), emit=True)
        tp.bc_type.setCurrentIndex(tp.bc_type.findData("fixed_temperature"))
        tp.bc_val.setValue(100.0)
        tp.add_bc()
        w.view.set_selected_faces(plane_x(100.0), emit=True)
        tp.bc_type.setCurrentIndex(tp.bc_type.findData("convection"))
        tp.bc_val.setValue(500.0)
        tp.bc_amb.setValue(20.0)
        tp.add_bc()
        c = w.study.thermal.case("Heating")
        check([b.kind for b in c.loads] == ["fixed_temperature", "convection"], f"okrajové podmínky: {tp.bc_list.count()}")
        # undo / redo
        w.undo()
        n0 = len(w.study.thermal.case("Heating").loads)
        w.redo()
        check((n0, len(w.study.thermal.case("Heating").loads)) == (1, 2), "zpět/znovu tepelné okrajové podmínky")
        w.set_stage(w.STAGE["Heat"])
        # editing the selected condition: convection h 500 → 400 → 500
        tp.case_combo.setCurrentIndex(0)
        tp.bc_list.setCurrentRow(1)
        check(tp.bc_type.currentData() == "convection" and abs(tp.bc_val.value() - 500) < 1e-9, "formulář z vybrané podmínky")
        tp.bc_val.setValue(400.0)
        tp.update_bc()
        check(w.study.thermal.case("Heating").loads[1].h == 400.0, "úprava konvekce")
        tp.bc_list.setCurrentRow(1)
        tp.bc_val.setValue(500.0)
        tp.update_bc()
        # transient state: a jump of 100 °C at x = 0, 600 s, 6 frames
        tp.new_case("Ramp")
        tp.transient.setChecked(True)
        tp.t_time.setValue(600.0)
        tp.t_dt.setValue(10.0)
        tp.t_out.setValue(6)
        w.view.set_selected_faces(plane_x(0.0), emit=True)
        tp.bc_type.setCurrentIndex(tp.bc_type.findData("fixed_temperature"))
        tp.bc_val.setValue(100.0)
        tp.add_bc()
        nb = w.study.thermal.case("Ramp")
        check(nb.transient and nb.time == 600.0 and nb.dt == 10.0 and nb.outputs == 6, f"přechodový stav {nb}")
        tp.case_combo.setCurrentIndex(0)
        w.view.set_selected_faces([], emit=True)
        shot("t2_teplo")
        errors, warnings = w.check()
        check(not errors, f"studie jen s teplem je připravena (chyby: {errors})")
        print("varování:", warnings)
        w.set_stage(w.STAGE["Solve"])
        print("připravenost:", [w.ready_list.item(i).text() for i in range(w.ready_list.count())])
        check(w.btn_run.isEnabled(), "tlačítko Spustit povoleno u studie jen s teplem")
        shot("t3_reseni_jen_teplo")
        w.run_solve()

    def s_thermal_only_results():
        res = w.results
        check(res is not None and res.steps == [] and len(res.thermal) == 2,
              f"výsledky jen tepla: {None if res is None else (len(res.steps), len(res.thermal))}")
        w.set_stage(w.STAGE["Results"])
        select_result("Thermal: Heating")
        select_qty("Temperature")
        th = res.thermal_case("Heating")
        idx = np.nonzero(np.abs(res.mesh.nodes[:, 0] - 100) < 1e-6)[0]
        T100 = float(th.T[idx].mean())
        check(abs(T100 - 60.0) < 0.05, f"T(x = 100) = {T100:.4f} °C (analyticky 60)")
        # maximum places even without a mechanical state: the hottest places (fea_core.mcp.diagnostics.thermal_hotspots)
        hs = w._hs_rows
        check(w.hs_table.rowCount() > 0 and "Hottest" in w.hs_title.text() and abs(hs[0][1] - 100.0) < 1e-6,
              f"tabulka „{w.hs_title.text()}“: {w.hs_table.rowCount()} míst, T1 = {hs[0][1] if hs else None}")
        print("  ", " | ".join(w.hs_table.item(0, c).text() for c in range(w.hs_table.columnCount())))
        w.on_probe((100.0, 10.0, 10.0))
        print("sonda (přímo):", w.st_info.text())
        check("60" in w.st_info.text(), "sonda ukazuje teplotu čela")
        click_at((50.0, 10.0, 20.0))
        print("sonda (klik):", w.st_info.text())
        check("Temperature" in w.st_info.text(), "klik myší do výsledku teploty")
        shot("t4_jen_teplo_teplota")
        select_qty("Heat flux |q|")
        check(w.hs_table.rowCount() > 0 and "heat flux" in w.hs_title.text(),
              f"místa maxim toku: „{w.hs_title.text()}“, {w.hs_table.rowCount()} míst")
        print("info:", w.res_info.toPlainText()[:400].replace("\n", " | "))
        shot("t5_jen_teplo_tok")
        select_result("Thermal: Ramp")
        check(w.r_time.isVisible() and w.r_time.count() == len(res.thermal_case("Ramp").times),
              f"snímky přechodového stavu: {w.r_time.count()}")
        select_qty("Temperature")
        w.r_time.setCurrentIndex(1)
        shot("t6_nabeh_snimek")
        # the other steps with results without a mechanical state must not fail
        for name in ("Fatigue", "Joints", "3D printing", "Results"):
            w.set_stage(w.STAGE[name])
        check(not w.fat_panel.btn.isEnabled(), "únava bez mechanického stavu zakázána")

    def s_mech():
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces(plane_x(0.0), emit=True)
        w.add_fixed()
        w.set_stage(w.STAGE["Loads"])
        w.lc_combo.setCurrentIndex(0)
        w.ld_type.setCurrentText(THERMAL_LOAD)
        w.ld_th_src.setCurrentIndex(w.ld_th_src.findData("Heating"))
        w.add_load()
        w.study.load_cases.append(gm.LoadCase("Rovnoměrně 80 °C"))
        w.changed()
        w.lc_combo.setCurrentIndex(1)
        w.ld_type.setCurrentText(THERMAL_LOAD)
        w.ld_th_src.setCurrentIndex(w.ld_th_src.count() - 1)        # uniform temperature of the parts
        w.ld_val.setValue(80.0)
        w.add_load()
        w.lc_combo.setCurrentIndex(0)
        l0, l1 = w.study.load_cases[0].loads[0], w.study.load_cases[1].loads[0]
        check(l0.case == "Heating" and l0.time is None and l1.value == 80.0 and l1.case is None,
              f"teplotní zatížení: {w.describe_load(l0)} | {w.describe_load(l1)}")
        w.load_list.setCurrentRow(0)
        check(w.ld_type.currentText() == THERMAL_LOAD and w.ld_th_src.currentData() == "Heating", "formulář teplotního zatížení")
        shot("t7_zatizeni_teplota")
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_results():
        res = w.results
        check(res is not None and len(res.steps) == 2 and len(res.thermal) == 2,
              f"výsledky: {len(res.steps)} mech., {len(res.thermal)} tepelné")
        w.set_stage(w.STAGE["Results"])
        select_result("Load case 1")
        select_qty("von Mises")
        st = res.step(0)
        print(f"teplotní napětí: max vM {st.mises.max():.2f} MPa, max posuv {st.umag.max():.4f} mm, "
              f"T {st.fields['T'].min():.2f}…{st.fields['T'].max():.2f}")
        check("T" in st.fields, "mechanický stav má pole T")
        shot("t8_teplotni_napeti_vm")
        select_qty("Temperature")
        check(any(x in w.legend.title for x in ("Temperature", "Teplota")), f"legenda: {w.legend.title}")
        shot("t9_mech_teplota")
        select_result("Rovnoměrně 80 °C")
        select_qty("Displacement X")
        # far from the clamping (which prevents the transverse expansion) the free expansion: u(100) − u(50) = αΔT·50
        U = res.step(1).U[:, 0]
        x = res.mesh.nodes[:, 0]
        du = U[np.abs(x - 100) < 1e-6].mean() - U[np.abs(x - 50) < 1e-6].mean()
        check(abs(du - 12e-6 * 60 * 50) / (12e-6 * 60 * 50) < 0.005, f"volná roztažnost u(100) − u(50) = {du:.6f} mm "
              f"(αΔT·50 = {12e-6 * 60 * 50:.6f})")
        shot("t10_rovnomerne_posuv")
        select_result("Thermal: Heating")
        check(w.r_qty.count() == len(THERMAL_QUANTITIES), "veličiny tepla po přepnutí")
        w.sec_axis.setCurrentIndex(2)            # section by the plane Y in the middle
        w.show_result()
        shot("t11_teplo_rez")
        w.sec_axis.setCurrentIndex(0)
        select_result("Load case 1")
        check(w.r_qty.count() == len(QUANTITIES), "mechanické veličiny po přepnutí zpět")
        w._save(OUT / "tyc.fea")
        w.dirty = False

    def s_reload():
        w.open_file(str(OUT / "tyc.fea"))
        st = w.study
        res = w.results
        check(len(st.thermal.cases) == 2 and st.thermal.case("Ramp").transient, "tepelné stavy po otevření")
        check(res is not None and len(res.thermal) == 2 and res.thermal_case("Ramp").frames is not None,
              "výsledky tepla po otevření")
        check(st.material_of("Tyč").k == 50.0, "λ materiálu po otevření")
        select_result("Thermal: Heating")
        shot("t12_znovu_otevreno")
        w.set_stage(w.STAGE["Heat"])
        shot("t13_teplo_po_otevreni")

    steps = [s_open, s_material, s_thermal, s_thermal_only_results, s_mech, s_results, s_reload]

    def run_next():
        if w._thread is not None:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            print("snímky:", ", ".join(log))
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
        QTimer.singleShot(700, run_next)

    QTimer.singleShot(800, run_next)
    app.exec()
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
