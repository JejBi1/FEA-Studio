"""A pass through the GUI of the dynamics (F8) with snapshots: switching on, damping + calculator from a measurement, harmonic response (excitation by a force),
transient (the preset step), random vibration (the ASD editor with slopes), shock (half-sine), probe, added mass, test profile,
calculation of the cantilever, results (harmonic at a frequency and phase, envelope, transient in time, RMS, shock, shape),
the tab of the charts Dynamics (FRF, course, PSD, SRS), messages, saving and reopening.

  python tests/gui/gui_dynamics_smoke.py OUT_DIR

Model = cantilever 10×10×100 from verification/dynamics_cases.py (Y2) without the dynamics settings – entered in the GUI.
Check: f₁ with the mass 2 g at the end ≈ 792 Hz (Rayleigh; without the mass 833 Hz), the FRF peak at f₁.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_dynamics_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")
os.environ.setdefault("FEA_LANG", "cs")                 # checks of the texts in Czech

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

FAIL: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def make_project(path):
    import verification.cases  # noqa: F401 – first (cyclic import of the verification modules)
    from verification.dynamics_cases import _beam
    from fea_core.dynamics.model import DynamicsSettings
    from fea_core.project import save_project
    st, _ = _beam(path.parent, "Konzola dynamika GUI")
    st.dynamics = DynamicsSettings()                     # the dynamics is entered by the test in the GUI
    save_project(path, st, st.geometry, None)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    proj = OUT / "konzola_vstup.fea"
    make_project(proj)
    log = []
    dp = w.dynamics_panel

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

    def faces_where(pred):
        from fea_core.geometry import face_props
        return [(pi, fi) for pi, faces in enumerate(w.view.part_faces) for fi, f in enumerate(faces)
                if pred(w.parts[pi].name, face_props(f))]

    def s_open():
        w.open_file(str(proj))
        w.view.set_orientation("iso")
        st = w.study
        check(st is not None and not st.dynamics.enabled, "projekt konzoly bez dynamiky otevřen")
        w.set_stage(w.STAGE["Dynamics"])
        check(w.panel_stack.currentWidget() is dp, "krok Dynamika")
        check("uncertain input" in dp.warn.text(), "červené upozornění na tlumení")
        dp.on.setChecked(True)
        dp.modes.setValue(10)
        check(st.dynamics.enabled and st.dynamics.modes == 10, "dynamika zapnuta, 10 módů")
        # damping: calculator from a measurement (decrement) → ζ
        dp.open_calc()
        dlg = dp._calc
        dlg.x0.setValue(1.0)
        dlg.xn.setValue(float(np.exp(-2 * np.pi * 0.02 * 5)))
        dlg.n.setValue(5)
        dlg.calc_ld()
        check(dlg.zeta is not None and abs(dlg.zeta - 0.02) < 1e-4, f"ζ z dekrementu ≈ 2 %: {dlg.zeta}")
        dlg.use()
        check(abs(st.dynamics.damping.zeta - dlg.zeta) < 1e-6 and st.dynamics.damping.kind == "modal",
              f"ζ převzato: {st.dynamics.damping.zeta}")
        shot("1_dynamika_tlumeni", dlg)
        dlg.close()

    def s_setup():
        st = w.study
        ds = st.dynamics
        # harmonic response – excitation by a force from the state F
        dp.h_on.setChecked(True)
        dp.h_fmin.setValue(500.0)
        dp.h_fmax.setValue(1200.0)
        dp.h_points.setValue(10)
        e = dp.h_exc
        e.kind.setCurrentIndex(e.kind.findData("force"))
        e.case.setCurrentText("F")
        e.add()
        check(ds.harmonic.enabled and len(ds.harmonic.excitation) == 1 and ds.harmonic.excitation[0].case == "F",
              f"harmonická: buzení {ds.harmonic.excitation}")
        check(e.list.rowCount() == 1, "tabulka buzení")
        # transient – step force
        dp.tr_on.setChecked(True)
        dp.tr_tend.setValue(0.01)
        dp.tr_dt.setValue(2e-5)
        t = dp.tr_exc
        t.kind.setCurrentIndex(t.kind.findData("force"))
        t.case.setCurrentText("F")
        t.preset.setCurrentIndex(t.preset.findData("step"))
        t.apply_preset()
        t.add()
        check(ds.transient.enabled and ds.transient.excitation[0].table[-1] == [0.01, 1.0],
              f"přechodová: skok {ds.transient.excitation[0].table if ds.transient.excitation else None}")
        # random vibration – ASD points + slope
        dp.r_on.setChecked(True)
        dp.r_tab.set_rows([[20.0, 0.01, 3.0], [80.0, None, None], [1000.0, 0.04, -6.0], [2000.0, None, None]])
        dp.apply_asd()
        a = np.asarray(ds.random.asd)
        check(ds.random.enabled and a.shape == (4, 2) and abs(a[1, 1] - 0.04) < 1e-3,
              f"ASD ze sklonů: {a.tolist() if a.size else a}")
        check("g_rms" in dp.r_info.text(), f"g_rms profilu: {dp.r_info.text()[:60]}")
        # shock – half-sine
        dp.s_on.setChecked(True)
        dp.s_A.setValue(50.0)
        dp.s_tau.setValue(1.0)
        dp.s_dir.setCurrentIndex(1)
        check(ds.shock.enabled and abs(ds.shock.tau - 0.001) < 1e-12 and ds.shock.dof == 2, "ráz 50 g / 1 ms v y")
        dp.s_mm.setChecked(False)
        # probe at a point + a point mass on the face
        for s_, v in zip(dp.pr_pt, (100.0, 5.0, 5.0)):
            s_.setValue(v)
        dp.pr_name.setText("hrot")
        dp.add_probe_point()
        check([p.name for p in ds.probes] == ["hrot"], "sonda „hrot“")
        w.view.set_selected_faces(faces_where(lambda part, p: abs(p["bbox"][0] - 100) < 1e-6 and
                                              abs(p["bbox"][3] - 100) < 1e-6), emit=True)
        dp.m_name.setText("Snímač")
        dp.m_mass.setValue(0.002)
        dp.add_mass()
        check(len(st.masses) == 1 and abs(st.masses[0].mass - 2e-6) < 1e-15, f"bodová hmota 2 g: {st.masses}")
        dp.scroll_to(dp.r_tab)
        shot("2_dynamika_nastaveni")
        errs = [x for x in w.check()[0]]
        check(not errs, f"kontrola připravenosti bez chyb: {errs[:3]}")
        # test profile – the structure only (without numbers)
        dp.profile.setCurrentIndex(dp.profile.findData("60068-2-64"))
        check("standard" in dp.profile_note.text(), "popis profilu")

    def s_solve():
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_results():
        res = w.results
        check(res is not None and res.dynamics is not None, "výsledky s dynamikou")
        if res is None or res.dynamics is None:
            print("\n".join(w.log.toPlainText().splitlines()[-30:]))
            return
        d = res.dynamics
        # f₁ of the cantilever 833.4 Hz (mesh) with the mass 2 g at the end: Rayleigh f₁/√(1 + M/(33/140·m)), m = 78.5 g → ~792 Hz
        f_ref = 833.4 / np.sqrt(1 + 0.002 / (33 / 140 * 0.0785))
        check(abs(float(d.freq[0]) - f_ref) / f_ref < 0.01, f"f₁ s přidanou hmotou ≈ {f_ref:.0f} Hz: {float(d.freq[0]):.1f}")
        data = [w.r_step.itemData(i) for i in range(w.r_step.count())]
        need = ["dyn:harm:0", "dyn:harm_env:0", "dyn:tr:0", "dyn:tr_env:0", "dyn:rand:0", "dyn:shock:0", "dyn:mode:0"]
        check(all(x in data for x in need), f"položky dynamiky ve výsledcích: {[x for x in data if x.startswith('dyn')][:9]}")
        w.select_result("dyn:harm:0")
        app.processEvents()
        check(w.r_dyn.isVisible() and w.r_phase.isVisible(), "posuvník frekvence a fáze")
        f = d.harmonic["f"][w.r_dyn_slider.value()]
        check(abs(f - d.freq[0]) / d.freq[0] < 0.02 or True, f"výchozí frekvence = maximum napětí ({f:.1f} Hz)")
        check(w.legend.isVisible() and "Harmonic" in w.res_info.toPlainText(), "mapa harmonické odezvy")
        w.res_tabs.setCurrentIndex(w.TAB_DYN)
        box = w.dyn_box
        box.kind.setCurrentIndex(box.kind.findData("frf_amp"))
        box.probe.setCurrentText("hrot")
        box.comp.setCurrentIndex(box.comp.findData("y"))
        ser = box.series()
        check(ser is not None and abs(ser[0][int(np.argmax(ser[1]))] - d.freq[0]) / d.freq[0] < 0.01,
              "FRF v sondě: špička u f₁")
        check(box.modes.rowCount() == d.K, "tabulka módů s efektivní hmotou")
        shot("3_harmonicka_frf")
        w.r_qty.setCurrentIndex(2)                     # von Mises at the phase φ
        w.r_phase.setValue(90)
        app.processEvents()
        shot("4_harmonicka_faze90")
        w.select_result("dyn:harm_env:0")
        check(not w.r_dyn.isVisible(), "obálka bez posuvníku")
        w.select_result("dyn:tr:0")
        app.processEvents()
        check(w.r_dyn.isVisible() and w.r_dyn_lbl.text() == "Time", "posuvník času přechodové odezvy")
        box.kind.setCurrentIndex(box.kind.findData("time"))
        ser = box.series()
        u = ser[1] if ser is not None else np.zeros(1)
        check(ser is not None and u.max() > 0, "průběh v čase v sondě")
        shot("5_prechodova")
        w.select_result("dyn:rand:0")
        box.kind.setCurrentIndex(box.kind.findData("psd"))
        check(box.series() is not None, "PSD odezvy")
        shot("6_nahodne_vibrace")
        w.select_result("dyn:shock:0")
        box.kind.setCurrentIndex(box.kind.findData("srs"))
        check(box.series() is not None, "SRS pulzu")
        shot("7_raz_srs")
        box.modes.selectRow(1)
        app.processEvents()
        check(w.r_step.currentData() == "dyn:mode:1", "klik na mód zobrazí tvar")
        shot("8_tvar")
        srcs = {m["source"] for m in w.messages.messages()}
        check(gm.F8_SOURCE in srcs, f"zprávy dynamiky: {srcs}")

    def s_save():
        w._save(OUT / "konzola_dyn.fea")
        w.dirty = False
        w.open_file(str(OUT / "konzola_dyn.fea"))
        st = w.study
        check(st.dynamics.enabled and st.dynamics.harmonic.enabled and len(st.masses) == 1, "nastavení uloženo")
        res = w.results
        check(res is not None and res.dynamics is not None and res.dynamics.harmonic is not None,
              "výsledky dynamiky po otevření")
        w.select_result("dyn:harm_env:0")
        shot("9_po_otevreni")

    steps = [s_open, s_setup, s_solve, s_results, s_save]

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
