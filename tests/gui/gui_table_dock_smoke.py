"""The bottom dock Tables (Studio × Tables layout, stage 3): the table by step and a click on a row.

python tests/gui/gui_table_dock_smoke.py OUT_DIR [project.fea]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/gui_table_dock")
OUT.mkdir(parents=True, exist_ok=True)
PROJECT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("examples/example2_assembly.fea")
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)

app = QApplication(sys.argv)
from fea_gui.main import MainWindow  # noqa: E402
from fea_gui.widgets import fmt  # noqa: E402

fails = []


def check(cond, text):
    print(("OK   " if cond else "CHYBA ") + text)
    if not cond:
        fails.append(text)


w = MainWindow()
w.resize(1600, 950)
w.show()
w.open_file(PROJECT)
app.processEvents()
td = w.table_dock
tabs = td.tables

w.set_stage(w.STAGE["Model"])
app.processEvents()
check(td.group == "parts" and tabs["parts"].rowCount() == len(w.parts) + 1, "Model: řádek na díl + souhrn")
check(w.tabifiedDockWidgets(w.dock_tables) and not w.dock_tables.visibleRegion().isEmpty(), "Model: dok Tabulka vpředu")
summary = tabs["parts"].item(tabs["parts"].rowCount() - 1, 8).text()
check(summary.startswith("weight") and summary.endswith("N"), f"souhrn dílů s tíhou ({summary})")

w.set_stage(w.STAGE["Loads"])
app.processEvents()
st = w.study
n_rows = len(st.constraints) + sum(len(lc.loads) for lc in st.load_cases)
check(td.group == "loads" and tabs["loads"].rowCount() == n_rows + 1, "Zatížení: vazby + zatížení + souhrn")
check(tabs["loads"].columnCount() == 7 + len(st.load_cases), "Zatížení: Fx, Fy, Fz, |F| a sloupec na zatěžovací stav")
# a click on the last load of the last state selects the state and the row in the panel
ci = max(i for i, lc in enumerate(st.load_cases) if lc.loads)
row = len(st.constraints) + sum(len(lc.loads) for lc in st.load_cases[:ci]) + len(st.load_cases[ci].loads) - 1
td._on_click("loads", row)
app.processEvents()
check(w.lc_combo.currentIndex() == ci and w.load_list.currentRow() == len(st.load_cases[ci].loads) - 1,
      "klik na zatížení vybere stav a položku v panelu")

w.set_stage(w.STAGE["Mesh"])
app.processEvents()
if w.mesh is not None:
    check(td.group == "mesh" and tabs["mesh"].rowCount() == len(w.mesh.part_names) + 1, "Síť: řádek na díl + souhrn")
else:
    check(tabs["mesh"].rowCount() == 0, "Síť: bez sítě prázdná tabulka")

w.dock_log.raise_()
app.processEvents()
w.set_stage(w.STAGE["Solve"])
app.processEvents()
check(td.group == "mesh" and not w.dock_log.visibleRegion().isEmpty() and w.dock_tables.visibleRegion().isEmpty(),
      "Řešení: tabulka sítě obnovena, vpředu zůstal log řešiče")

w.set_stage(w.STAGE["Results"])
app.processEvents()
if w.results is not None:
    steps = [i for i, s in enumerate(w.results.steps) if len(s.mises)]
    check(tabs["results"].rowCount() == len(steps), "Výsledky: řádek na zatěžovací stav")
    if len(steps) > 1:
        td._on_click("results", 1)
        app.processEvents()
        check(w.r_step.currentData() == f"step:{steps[1]}", "klik na stav zobrazí stav ve výsledcích")
    # the Assessment card (step 4): the verdict against the requirement; a change of the requirement does not discard the results
    ov = w.res_overview
    from fea_gui.table_dock import step_summaries
    ks = [d["k"] for d in step_summaries(w) if d["k"] is not None]
    check(ov.isVisible() and bool(ks), "karta Posouzení viditelná, aspoň jeden stav posouzen")
    if ks:
        kmin = min(ks)
        check(ov.k.text() == fmt(kmin, 3), f"karta ukazuje nejmenší bezpečnost ({ov.k.text()} = {fmt(kmin, 3)})")
        res_before = w.results
        ov.req.setValue(min(10.0, round(kmin + 1.0, 1)))        # the requirement above the smallest safety
        app.processEvents()
        check(ov.verdict.text() == "Fails" and ov.verdict.property("state") == "fail",
              "požadavek nad bezpečností → Nevyhovuje")
        check(w.study.assessment.get("required_safety") == round(min(10.0, round(kmin + 1.0, 1)), 3),
              "požadavek uložen do Study.assessment")
        check(w.results is res_before, "změna požadavku nezahodí výsledky")
        ov.req.setValue(1.0)
        app.processEvents()
        check(ov.verdict.text() == "Passes" or kmin < 1.0, "požadavek 1,0 → Vyhovuje (je-li k ≥ 1)")
        cur = next(d for d in step_summaries(w) if f"step:{d['index']}" == w.r_step.currentData())
        check(ov.kpi["umax"].text() == f"{fmt(cur['umax'], 4)} mm", "čísla karty patří zobrazenému stavu")
w.grab().save(str(OUT / "dock_vysledky.png"))

# tabs of the tables: the first tab = the table of the group, the counts in the text; the title of the dock hidden (the heading = the tabs)
from fea_core.i18n import tr  # noqa: E402
from fea_gui.table_dock import TABS, TITLES  # noqa: E402
check(td.current == "results" and td.tabbar.count() == len(TABS["results"])
      and td.tabbar.tabText(0).startswith(tr(TITLES["results"])), f"záložky Výsledků ({td.tabbar.tabText(0)})")
check(w.dock_tables.titleBarWidget() is not None, "spodní dok bez titulku (jeden nadpis)")
td.show_tab("hotspots")
app.processEvents()
check(td.current == "hotspots" and tabs["hotspots"].rowCount() > 0, f"Nejvyšší napětí: {tabs['hotspots'].rowCount()} míst")
td.show_tab("psafety")
app.processEvents()
check(tabs["psafety"].rowCount() > 0, f"Bezpečnost po dílech: {tabs['psafety'].rowCount()} dílů")
td.show_tab("reactions")
app.processEvents()
check(tabs["reactions"].rowCount() > 0, f"Reakce: {tabs['reactions'].rowCount()} řádků")
td.show_tab("results")
sm = w.st_model.text()
if w.results is not None:
    check("Equilibrium" in sm and "elements" in sm and sm.endswith("mm · N · MPa") and w.st_model.property("state") == "ok",
          f"stavový řádek: {sm}")

# optional steps: the shelf and the tables (Heat from the example, the other items added only for the test)
w.set_stage(w.STAGE["Heat"])
app.processEvents()
th = w.study.thermal.cases
check(td.group == "thermal" and tabs["thermal"].rowCount() == sum(len(c.loads) for c in th) + (1 if th else 0)
      and tabs["thermal"].columnCount() == 3 + len(th), "Teplo: podmínka na řádek, sloupec na tepelný stav")
check(any(a.isVisible() for a in w._shelf_acts["Heat"]) and not any(a.isVisible() for a in w._shelf_acts["Loads"]),
      "Teplo: police ukazuje příkazy tepla")
w._shelf_run(("thermal_panel.bc_type", "Heat transfer – convection"))
check(w.thermal_panel.bc_type.currentText() == "Heat transfer – convection", "police Konvekce přepne druh podmínky")

from fea_core.contacts.model import ContactPair  # noqa: E402
from fea_core.dynamics.model import PointMass, SurfaceMass  # noqa: E402
from fea_core.beams.model import FrameMember  # noqa: E402
from fea_core.motion.model import Body, MotionJoint  # noqa: E402
st = w.study
pa, pb = w.parts[0].name, w.parts[1].name
st.contacts.append(ContactPair("Test", pa, pb, kind="frictional", mu=0.15))
st.masses += [PointMass("Test hmota", 1.5), SurfaceMass("Test plocha", 2.0)]
st.frame.add_node("N1", [0, 0, 0])
st.frame.add_node("N2", [300, 400, 0])
st.frame.members["P1"] = FrameMember("P1", "N1", "N2", "IPE100")
st.mechanism.bodies += [Body("Ground", [pa], ground=True), Body("Rameno", [pb])]
st.mechanism.joints.append(MotionJoint("Pin", "revolute", "Ground", "Rameno"))
for stage, g, n in (("Contacts", "contacts", 1), ("Dynamics", "dynamics", 3), ("Members", "frame", 2), ("Motion", "motion", 3)):
    w.set_stage(w.STAGE[stage])
    app.processEvents()
    check(td.group == g and td.current == g and tabs[g].rowCount() == n,
          f"{stage}: tabulka {g} ({tabs[g].rowCount()} řádků)")
    check(any(a.isVisible() for a in w._shelf_acts[stage]), f"{stage}: police má příkazy kroku")
check(tabs["frame"].item(0, 4).text() == "500", f"délka prutu 500 mm ({tabs['frame'].item(0, 4).text()})")
check(tabs["dynamics"].item(2, 2).text() == f"{fmt(1.5, 4)} kg", f"souhrn bodových hmot ({tabs['dynamics'].item(2, 2).text()})")
w.grab().save(str(OUT / "dock_pohyb.png"))

# Model: the tabs of the parts and the editable cells (Count → ignored_parts), the Properties card of the selected part
w.dirty = False
w.set_stage(w.STAGE["Model"])
app.processEvents()
check(td.group == "parts" and td.tabbar.count() == 4, "Model: záložky Díly, Materiály, Kontrola geometrie, Souřadné systémy")
t = tabs["parts"]
name0 = w.parts[td._row_data["parts"][0]].name
it = t.item(0, 2)
it.setCheckState(Qt.Unchecked)
app.processEvents()
check(name0 in w.study.ignored_parts, f"odškrtnutí Počítat vynechá díl {name0}")
t = tabs["parts"]
t.item(0, 2).setCheckState(Qt.Checked)
app.processEvents()
check(name0 not in w.study.ignored_parts, "zaškrtnutí Počítat díl vrátí")
w.view.set_selected_parts([0])
w.on_parts_picked([0])
w.props_card.refresh()
app.processEvents()
check(w.props_card.isVisible() and w.parts[0].name in w.props_card.header.text(), "Vlastnosti vybraného dílu")
w.set_stage(w.STAGE["Joints"])
app.processEvents()
check(td.group == "joints" and tabs["bolts"].rowCount() == len(w.study.bolts), "Spoje: tabulka šroubů")
check(not w.dock_joints.isVisible(), "dok Spoje – využití skrytý (tabulky jsou v doku Tabulky)")
if w.study.bolts:
    cb = tabs["bolts"].cellWidget(0, 2)                    # the class of the bolt in the cell = the form of Joints + save
    old = w.study.bolts[0].strength
    new = "10.9" if old != "10.9" else "8.8"
    cb.setCurrentIndex(cb.findText(new))
    app.processEvents()
    check(w.study.bolts[0].strength == new, f"třída šroubu z tabulky {old} → {w.study.bolts[0].strength}")
w.set_stage(w.STAGE["Loads"])
app.processEvents()
st = w.study
row = next(i for i, d in enumerate(td._row_data["loads"]) if d[0] == "load")
ci, li = td._row_data["loads"][row][1]
other = (ci + 1) % len(st.load_cases)
n_before = len(st.load_cases[other].loads)
tabs["loads"].item(row, 7 + other).setCheckState(Qt.Checked)
app.processEvents()
check(len(st.load_cases[other].loads) == n_before + 1, "zaškrtnutí v matici zkopíruje zatížení do stavu")
w.set_stage(w.STAGE["Mesh"])
app.processEvents()
link = next(i for i in range(td.tabbar.count()) if td.tabbar.tabData(i) == "log")
td.tabbar.setCurrentIndex(link)
app.processEvents()
check(td.current in ("mesh", "refine") and not w.dock_log.visibleRegion().isEmpty(), "záložka Log řešiče přepne na dok logu")

w.dirty = False
w.close()
print("VÝSLEDEK:", "OK" if not fails else f"{len(fails)} chyb")
sys.exit(1 if fails else 0)
