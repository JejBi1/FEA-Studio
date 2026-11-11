"""A pass through the GUI of the contacts (F8) with snapshots: Find contacts, adding a pair, editing (type, K, μ from the library), a pair from a selection of
faces, settings of the contact, sequence of steps, calculation of K1 (two blocks, pressure 10 MPa, contact without friction), quantities
of the contact (pressure, state of the contact), the tab Contacts, messages, saving and reopening.

  python tests/gui/gui_contact_smoke.py OUT_DIR

Model = the case K1 from verification/contact_cases.py (without the contact pair – that is entered in the GUI). The expected pressure in the
contact p = 10 MPa, the normal force of the pair 4000 N.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_contact_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")           # user libraries only into the folder of the test
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
    from verification.contact_cases import k1_build
    from fea_core.project import save_project
    st, _ = k1_build(path.parent)
    st.contacts.clear()                                # the pair is entered by the test in the GUI
    save_project(path, st, st.geometry, None)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow, QUANTITIES
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")      # without modal windows
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)   # closing the window: do not save
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    proj = OUT / "k1_vstup.fea"
    make_project(proj)
    log = []
    cp = w.contact_panel

    def shot(name):
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
        log.append(name)

    def faces_where(pred):
        from fea_core.geometry import face_props
        return [(pi, fi) for pi, faces in enumerate(w.view.part_faces) for fi, f in enumerate(faces)
                if pred(w.parts[pi].name, face_props(f))]

    # ------------------------------------------------------------------ kroky
    def s_open():
        w.open_file(str(proj))
        w.view.set_orientation("iso")
        check(w.study is not None and not w.study.contacts, "projekt K1 bez kontaktu otevřen")
        w.set_stage(w.STAGE["Contacts"])
        check(w.STAGES[w.stage] == "Contacts" and w.panel_stack.currentWidget() is cp, "krok Kontakty za Spoji")
        check(w.STAGE["Contacts"] == w.STAGE["Joints"] + 1 and w.STAGE["Dynamics"] == w.STAGE["Loads"] + 1,
              "pořadí kroků (Kontakty za Spoji, Dynamika za Zatížením)")
        check("linear" in cp.banner.text() or "nonlinear" in cp.banner.text(), f"pruh: {cp.banner.text()[:60]}")
        cp.tol.setValue(0.05)
        cp.find_touching()
        check(cp._thread is not None, "Najít dotyky běží ve vlákně")

    def s_found():
        check(len(cp._found) == 1 and cp._found[0]["state"] == "contact" and cp._found[0]["area"] >= 400 - 1,
              f"nalezen dotyk A–B (styk, plocha ≥ 400 mm²): {[(d['a'], d['b'], d['state'], d['area']) for d in cp._found]}")
        check(cp.found.rowCount() == 1, "tabulka nalezených dotyků")
        cp.add_kind.setCurrentIndex(cp.add_kind.findData("frictionless"))
        cp.add_found()
        st = w.study
        check(len(st.contacts) == 1 and st.contacts[0].kind == "frictionless" and st.contacts[0].initial == "contact",
              f"pár přidán: {st.contacts}")
        check("nonlinear" in cp.banner.text(), "pruh: studie je nelineární")
        check(cp.table.rowCount() == 1 and cp.found.item(0, 5).text() == "yes", "pár v tabulce, nalezený označen")
        shot("1_najit_dotyky")

    def s_edit():
        st = w.study
        cp.table.selectRow(0)
        app.processEvents()
        check(cp.p_a.currentText() == "A" and cp.p_kind.currentData() == "frictionless", "formulář načte pár")
        cp.p_K.setValue(1e6)
        cp.update_pair()
        check(st.contacts[0].stiffness == 1e6, f"K = 1e6: {st.contacts[0].stiffness}")
        # friction – μ from the library according to the purpose
        cp.table.selectRow(0)
        cp.p_kind.setCurrentIndex(cp.p_kind.findData("frictional"))
        cp.p_lib.setCurrentIndex(cp.p_lib.findData("steel – steel, dry"))
        cp.p_purpose.setCurrentIndex(cp.p_purpose.findData("resistance"))
        app.processEvents()
        check("0.8" in cp.mu_info.text() or "0,8" in cp.mu_info.text(), f"μ z knihovny pro odpor: {cp.mu_info.text()[:120]}")
        cp.p_purpose.setCurrentIndex(cp.p_purpose.findData("slip"))
        check("0.15" in cp.mu_info.text() or "0,15" in cp.mu_info.text(), "μ_min pro prokluz 0,15")
        shot("2_par_treni")
        cp.p_kind.setCurrentIndex(cp.p_kind.findData("frictionless"))
        cp.p_lib.setCurrentIndex(0)
        cp.update_pair()
        check(st.contacts[0].kind == "frictionless" and st.contacts[0].stiffness == 1e6, "zpět styk bez tření, K zůstává")
        # a pair from a selection of faces (sides A and B) and the removal
        w.view.set_selected_faces(faces_where(lambda part, p: part == "A" and abs(p["bbox"][5] - 10) < 1e-6
                                              and abs(p["bbox"][2] - 10) < 1e-6), emit=True)
        cp.side_from_selection("A")
        w.view.set_selected_faces(faces_where(lambda part, p: part == "B" and abs(p["bbox"][2] - 10) < 1e-6
                                              and abs(p["bbox"][5] - 10) < 1e-6), emit=True)
        cp.side_from_selection("B")
        check(cp.p_a.currentText().startswith(("Contact A", "Kontakt A")) and cp.p_b.currentText().startswith(("Contact B", "Kontakt B")),
              f"strany z výběru: {cp.p_a.currentText()}, {cp.p_b.currentText()}")
        cp.p_name.setText("Plochy A-B")
        cp.p_kind.setCurrentIndex(cp.p_kind.findData("frictionless"))
        cp.add_pair()
        check(len(st.contacts) == 2 and st.contacts[1].a in st.selections, "pár z výběrů ploch přidán")
        shot("3_par_z_vyberu")
        sel_a = st.contacts[1].a
        cp.table.selectRow(1)
        cp.delete_pair()
        check(len(st.contacts) == 1 and sel_a not in st.selections, "pár smazán i s nepoužitým výběrem")
        # the settings of the contact and the sequence of steps
        cp.s_method.setCurrentIndex(cp.s_method.findData("mortar"))
        check(st.contact_settings.method == "mortar", "formulace mortar")
        cp.s_method.setCurrentIndex(cp.s_method.findData("surface_to_surface"))
        cp.a_mode.setCurrentIndex(1)
        check(not st.analysis.branches and [s.name for s in st.analysis.sequence] == ["p"],
              f"posloupnost ze stavů: {[s.name for s in st.analysis.sequence]}")
        cp.seq.selectRow(0)
        app.processEvents()
        cp.k_scale.setValue(0.5)
        cp.k_name.setText("polovina")
        cp.add_step()
        check([s.scale for s in st.analysis.sequence] == [1.0, 0.5], "druhý krok s měřítkem 0,5")
        cp.add_motion()                              # without a rigid remote constraint → only a message
        check(not cp._motions, "pohyb bez tuhé vazby se nepřidá")
        cp.scroll_to(cp.seq)
        shot("4_posloupnost")
        cp.seq.selectRow(1)
        cp.delete_step()
        cp.a_mode.setCurrentIndex(0)
        check(st.analysis.branches and len(st.analysis.sequence) == 1, "zpět větve")
        w.undo()
        w.redo()
        check(len(st.contacts if st is w.study else w.study.contacts) == 1, "zpět / znovu")

    def s_solve():
        w.set_stage(w.STAGE["Solve"])
        errs = [w.ready_list.item(i).text() for i in range(w.ready_list.count())]
        check(any("ready to run" in e for e in errs), f"kontrola připravenosti: {errs[:3]}")
        w.run_solve()
        check(w.btn_stop_keep.isEnabled(), "„Zastavit a ponechat výsledky“ aktivní u nelineární studie")

    def s_results():
        res = w.results
        check(res is not None, "results")
        if res is None:
            print("\n".join(w.log.toPlainText().splitlines()[-30:]))
            return
        st = res.steps[0]
        c = st.contacts.get(w.study.contacts[0].name, {})
        check(abs(c.get("Fn_N", 0) - 4000) < 4, f"normálová síla páru 4000 N: {c.get('Fn_N')}")
        names = [q[0] for q in QUANTITIES]
        w.r_qty.setCurrentIndex(names.index("Contact – pressure"))
        app.processEvents()
        sl = st.fields["CPRESS"] > 0
        check(abs(float(np.mean(st.fields["CPRESS"][sl])) - 10) < 0.1, "tlak ve styku ~10 MPa")
        check(w.legend.isVisible(), "mapa tlaku")
        w.res_tabs.setCurrentIndex(w.TAB_CONTACT)
        check(w.res_tabs.isTabVisible(w.TAB_CONTACT) and w.contact_box.table.rowCount() == 1,
              "záložka Kontakty s párem")
        shot("5_tlak_ve_styku")
        w.r_qty.setCurrentIndex(names.index("Contact – contact state"))
        app.processEvents()
        check("sticking" in str(getattr(w.legend, "note", "")), "poznámka stavů styku")
        shot("6_stav_styku")
        w.r_qty.setCurrentIndex(0)
        srcs = {m["source"] for m in w.messages.messages()}
        check(gm.F8_SOURCE in srcs, f"zprávy nelineárního výpočtu: {srcs}")
        w.dock_messages.raise_()
        shot("7_zpravy")

    def s_save():
        w._save(OUT / "k1_gui.fea")
        w.dirty = False
        w.open_file(str(OUT / "k1_gui.fea"))
        st = w.study
        check(len(st.contacts) == 1 and st.contacts[0].stiffness == 1e6, "kontakty uloženy")
        res = w.results
        check(res is not None and "CPRESS" in res.steps[0].fields and res.steps[0].contacts,
              "výsledky kontaktu po otevření")
        w.set_stage(w.STAGE["Contacts"])
        shot("8_kontakty_po_otevreni")

    steps = [s_open, s_found, s_edit, s_solve, s_results, s_save]

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
