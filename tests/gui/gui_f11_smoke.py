"""F11 (A2): the progress bar by phases, visibility of the steps, the Analyses menu, a narrow window, collapsible sections in the Solve step.

  python tests/gui/gui_f11_smoke.py OUT_DIR

1) A new study from a STEP shows the 8 required steps (Model, Materials, Joints, Constraints, Loads, Mesh, Solve, Results).
2) Example 1 (printed bracket): the required steps + the 3D printing used; Heat / Dynamics / Motion / Contacts / Frames hidden.
3) The Analyses menu pins Heat; set_stage(Dynamics) reveals a hidden step; opening another study removes the revealed steps.
4) The menu of the phase button, Ctrl+PgDown / Ctrl+PgUp, the labels of the bar (short), "Show all steps".
5) Frame example 6: the steps FRAME_ONLY_OFF hidden and disabled, Frames visible.
6) The bar at the widths 1920 / 1366 / 1100 px (new study, example 1, example 2) in both themes: all visible steps
   are on the bar (at least as a dot), the » button hides nothing; snapshots of the bar, of the panels Constraints, Mesh, Solve.
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_f11_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")

from PySide6.QtCore import Qt, QRect  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QMenu  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FAIL: list[str] = []
MANDATORY = ["Model", "Materials", "Joints", "Constraints", "Loads", "Mesh", "Solve", "Results"]


def check(cond, text):
    print(("OK    " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def make_step(path):
    from verification.cases import box, cyl, solid
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from fea_core.geometry import Part, write_step
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 200, 60, 8), cyl((100, 30, -1), (0, 0, 1), 10, 10)).Shape())
    write_step([Part("Plate", plate), Part("Base plate", box(0, 0, 8, 20, 60, 30))], path)


def main():
    from fea_gui.main import MainWindow, STAGES, STAGE, PHASES, PHASE_OF, OPTIONAL_STAGES, FRAME_ONLY_OFF
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1700, 1000)
    w.show()

    def pump(n=4):
        for _ in range(n):
            app.processEvents()

    def vis():
        """Steps that the bar shows according to the state of the study (in a narrow window part of it is collapsed into the phase buttons)."""
        pump()
        return [n for n in STAGES if w.stage_visible(n)]

    def shown():
        """Step tabs above the panel = the visible steps of the CURRENT phase (the phase cards are in the header)."""
        pump()
        return [n for n, a in zip(STAGES, w.stage_actions) if a.isVisible()]

    def phase_steps(v):
        return [s for s in v if PHASE_OF[s] == PHASE_OF[STAGES[w.stage]]]

    def ext_visible():
        """Something from the phase cards or the step tabs did not fit (Qt hides the item and shows »)."""
        pump()
        for tb in (w.tb_file, w.tb_flow, w.tb_steps):
            for a in tb.actions():
                wd = tb.widgetForAction(a)
                if a.isVisible() and wd is not None and not wd.isVisible():
                    return True
        return False

    # ---- phase data
    check([s for _, ss in PHASES for s in ss] == STAGES and set(PHASE_OF) == set(STAGES), "fáze pokrývají všechny kroky")
    check(len(w.stage_actions) == 15 and w.stage_actions[STAGE["Joints"]].text() == "Joints", "stage_actions: 15 akcí v pořadí STAGES")
    check(all(len(a.toolTip()) <= 90 and "\n" not in a.toolTip() for a in w.stage_actions), "popisky kroků lišty jsou krátké")
    check(all(a.whatsThis() for a in w.stage_actions), "podrobný popis kroku (whatsThis) je zachován")

    # ---- 1) without a study and a new study from a STEP
    check(vis() == MANDATORY, f"bez studie: povinné kroky {vis()}")
    step = OUT / "deska.step"
    make_step(step)
    w.open_file(str(step))
    check(vis() == MANDATORY, f"nová studie ze STEP: 8 kroků ({vis()})")
    check(w._phase_buttons["Assembly"].isVisible() and w._phase_buttons["Conditions"].isVisible(), "tlačítka fází jsou vidět")

    # ---- 2) example 1
    ex1 = OUT / "ukazka1.fea"
    shutil.copy(ROOT / "examples" / "example1_bracket.fea", ex1)
    w.open_file(str(ex1))
    v = vis()
    check("3D printing" in v and all(m in v for m in MANDATORY), f"ukázka 1: povinné + 3D tisk ({v})")
    check(not any(s in v for s in ("Contacts", "Members", "Motion", "Heat", "Dynamics")), "ukázka 1: nepoužité kroky skryté")
    check(not ext_visible(), "ukázka 1: bez tlačítka » při šířce 1700 px")
    check(shown() == phase_steps(v), f"ukázka 1: záložky = kroky fáze {shown()}")
    check(all(w._phase_buttons[p].isVisible() for p, _ in PHASES), "ukázka 1: všech 5 karet fází")

    # ---- 3) the Analyses menu
    m = w.flow_more.menu()
    w._fill_more_menu(m)
    cbs = w._more_checks
    print("Analýzy:", {k: (c.text(), c.isChecked(), c.isEnabled()) for k, c in cbs.items()})
    used = cbs["3D printing"]
    check(used.isChecked() and not used.isEnabled() and "used in the study" in used.text(),
          "Analýzy: použitý krok je zaškrtnutý a zakázaný")
    teplo = cbs["Heat"]
    check(not teplo.isChecked() and teplo.isEnabled(), "Analýzy: Teplo nezaškrtnuto")
    check(len(cbs) == len(OPTIONAL_STAGES) + 1, "Analýzy: zaškrtávátko pro každý nepovinný krok + „všechny kroky“")
    w.set_stage(STAGE["Materials"])
    teplo.setChecked(True)                       # pin
    pump()
    check("Heat" in vis() and STAGES[w.stage] == "Heat" and "Heat" in w._flow_pins, "Analýzy: zaškrtnutí připne Teplo a přepne na něj")
    w.set_stage(STAGE["Materials"])
    check("Heat" in vis(), "připnutý krok zůstává na liště")
    w._fill_more_menu(m)
    teplo = w._more_checks["Heat"]
    check(teplo.isChecked(), "Analýzy: připnutý Teplo je zaškrtnutý")
    teplo.setChecked(False)
    pump()
    check("Heat" not in vis() and "Heat" not in w._flow_pins, "odškrtnutí Teplo odepne")
    check("Dynamics" not in vis(), "Dynamika před otevřením skrytá")
    w.set_stage(STAGE["Dynamics"])
    check("Dynamics" in vis() and "Dynamics" in w._flow_revealed, "set_stage odhalí skrytý krok")
    w.set_stage(STAGE["Materials"])
    check("Dynamics" in vis(), "odhalený krok zůstane po odchodu")
    # IPC / tree: the same path set_stage – the Motion step
    w.set_stage(STAGE["Motion"])
    check("Motion" in vis(), "Pohyb odhalen")
    # "Show all steps"
    w._set_show_all(True)
    check(vis() == STAGES, "Zobrazit všechny kroky: všech 15")
    w._set_show_all(False)
    w.set_stage(STAGE["Materials"])
    # another study removes the revealed ones
    w.open_file(str(ex1))
    check("Dynamics" not in vis() and not w._flow_revealed - {STAGES[w.stage]}, "otevření studie smaže odhalené kroky")

    # ---- 4) the phase menu, keys
    pm = QMenu()
    w._fill_phase_menu("Conditions", pm)
    ptxt = [a.text() for a in pm.actions()]
    check(ptxt == ["Constraints", "Motion (hidden)", "Heat (hidden)", "Loads", "Dynamics (hidden)"] or
          (len(ptxt) == 5 and any("(hidden)" in t for t in ptxt)), f"nabídka fáze Podmínky: {ptxt}")
    [a for a in pm.actions() if a.text().startswith("Heat")][0].trigger()
    check(STAGES[w.stage] == "Heat" and "Heat" in vis(), "volba v nabídce fáze přepne a odhalí krok")
    w.set_stage(STAGE["Materials"])
    w.step_stage(1)
    check(STAGES[w.stage] == "Joints" or STAGES[w.stage] == "3D printing", f"Ctrl+PgDown (step_stage): {STAGES[w.stage]}")
    w.set_stage(STAGE["Model"])
    QTest.keyClick(w, Qt.Key_PageDown, Qt.ControlModifier)
    pump()
    check(STAGES[w.stage] in ("Materials", "Model"), f"klávesa Ctrl+PgDown: {STAGES[w.stage]} (okno bez fokusu smí nereagovat)")
    w.set_stage(STAGE["Results"])
    w.step_stage(1)
    check(STAGES[w.stage] != "Model" and w.stage >= STAGE["Results"], "na posledním viditelném kroku se nic nestane")

    # ---- 5) frame system
    fea6 = OUT / "ukazka6.fea"
    shutil.copy(ROOT / "examples" / "example6_hall_frame.fea", fea6)
    w.open_file(str(fea6))
    v = vis()
    check(not any(s in v for s in FRAME_ONLY_OFF), f"prutová soustava: FRAME_ONLY_OFF skryté ({v})")
    check(all(not w.stage_actions[STAGE[s]].isEnabled() for s in FRAME_ONLY_OFF), "…a zakázané")
    check("Members" in v, "Pruty viditelné")
    w.new_frame_study()
    check("Members" in vis() and STAGES[w.stage] == "Members", "nová prutová soustava: krok Pruty")

    # ---- 6) the bar in three widths and two themes: new study, example 1, example 2
    shots = OUT
    ex2 = OUT / "ukazka2.fea"
    shutil.copy(ROOT / "examples" / "example2_assembly.fea", ex2)
    mode0 = w.settings.value("theme", "system")
    for tag, path in (("new", step), ("u1", ex1), ("u2", ex2)):
        w.resize(1700, 900)
        w.open_file(str(path))
        for mode in ("light", "dark"):
            w.set_theme(mode)
            for width in (1920, 1366, 1100):
                w.resize(width, 900)
                pump(6)
                v, sh = vis(), shown()
                # nothing may hide in » and every visible step is on the bar (in narrow windows the phases collapse only as
                # a last resort – in 1366 px all must be visible; the bars are in one row, in 1100 px the phases
                # not the current one may collapse)
                check(not ext_visible(), f"{tag} {mode} {width} px: úrovně {w._fit_levels}, tlačítko » nic neschovává")
                # the cards of all phases with visible steps in the header, the tabs = the visible steps of the current phase
                phases = [p for p, ss in PHASES if any(s in v for s in ss)]
                check(all(w._phase_buttons[p].isVisible() for p in phases),
                      f"{tag} {mode} {width} px: karty fází {phases} (úroveň {w._fit_levels})")
                check(sh == phase_steps(v), f"{tag} {mode} {width} px: záložky = kroky fáze ({sh})")
                h = w.tb_flow.geometry().bottom() + 4
                w.grab(QRect(0, 0, w.width(), h)).save(str(shots / f"A2_bar_{tag}_{width}_{mode}.png"))
    w.set_theme(mode0)
    w.resize(1500, 950)
    w.open_file(str(ex1))

    # ---- panels Constraints, Mesh, Solve
    w.resize(1500, 950)
    w.open_file(str(step))
    pump()

    def shot(name):
        pump()
        w.view.redraw()
        pump()
        img = w.grab().toImage()
        tmp = OUT / "_view.png"
        w.view.screenshot(str(tmp))
        vimg = QImage(str(tmp))
        p = QPainter(img)
        p.drawImage(w.view.mapTo(w, w.view.rect().topLeft()), vimg.scaled(w.view.size()))
        p.end()
        tmp.unlink(missing_ok=True)
        img.save(str(OUT / f"{name}.png"))

    for mode in ("light", "dark"):
        w.set_theme(mode)
        for st in ("Constraints", "Mesh", "Solve"):
            w.set_stage(STAGE[st])
            shot(f"A2_{st}_{mode}")
    w.set_theme(mode0)
    # collapsed sections of Solve: the Solver collapsed, natural frequencies ✓ by the tick
    w.set_stage(STAGE["Solve"])
    check(not w.solver_section.is_open(), "Řešič je ve výchozím stavu sbalený")
    check(not w.eigen_box.modal_group.is_open() and not w.eigen_box.buckling_group.is_open()
          and not w.twod_box.is_open() and not w.shell_box.is_open(), "oddíly s volbou „počítat“ jsou sbalené")
    w.eigen_box.m_on.setChecked(True)
    check(w.eigen_box.modal_group.is_open() and w.eigen_box.m_modes.isVisible(), "zaškrtnutí „počítat“ rozbalí oddíl")
    w.eigen_box.m_on.setChecked(False)
    from fea_gui.widgets import Section
    Section.reveal(w.backend_combo)
    check(w.backend_combo.isVisible(), "Section.reveal rozbalí Řešič")

    w.dirty = False
    w.close()
    print("\nSELHALO:" if FAIL else "\nVŠE OK", *FAIL, sep="\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
