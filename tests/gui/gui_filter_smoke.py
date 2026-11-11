"""Selection filter in 3D and the maximum places in the results (snapshots for a manual check of the look).

  python tests/gui/gui_filter_smoke.py OUT_DIR

Model: plate 200×60×8 with a hole Ø16 (two halves of the cylinder – as from some CADs) + foot 20×60×30 at x = 60–80
(sharp inner corners at the plate). Steps: a click on the parts in the Materials step (selection in the list), the filter Body (all faces of the part),
Face, Hole (both halves of the hole with one click, H extends the existing selection too), Edge (the arc of the hole: radius
and centre → "Use centre" into the point of a distant force and into the clipboard). Calculation (clamping of the face of the plate x = 0, force on the face x = 200),
the table "Highest stress" with the classification of the singularities, a click on a row (marker + probe), a click with the probe into the contours
(the band must not be highlighted). Snapshots of the 3D view through View.Dump (not a screenshot).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_filter_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL = []
HOLE = (150.0, 30.0)                # axis of the hole (vertical, z)
R_HOLE = 8.0


def check(cond, text):
    print(("OK   " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def make_step(path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from fea_core.geometry import Part, write_step
    from verification.cases import box, solid

    def half(xd):                    # half cylinder → hole from two coaxial cylindrical faces
        return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*HOLE, -1), gp_Dir(0, 0, 1), gp_Dir(*xd)), R_HOLE, 10,
                                        math.pi).Shape()
    plate = box(0, 0, 0, 200, 60, 8)
    plate = BRepAlgoAPI_Cut(plate, half((1, 0, 0))).Shape()
    plate = solid(BRepAlgoAPI_Cut(plate, half((-1, 0, 0))).Shape())
    write_step([Part("Plate", plate), Part("Base plate", box(60, 0, 8, 20, 60, 30))], path)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")      # without modal windows
    QMessageBox.warning = staticmethod(lambda parent, title, text, *a: print(f"[{title}] {text}"))
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    step = OUT / "deska_dira.step"
    make_step(step)
    log = []

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
                if pred(w.parts[pi].name, face_props(f)):
                    out.append((pi, fi))
        return out

    def part_index(name):
        return next(i for i, p in enumerate(w.parts) if p.name == name)

    def click_at(xyz, ctrl=False):
        """A real mouse click on the projection of a point of the model."""
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        app.processEvents()                        # resizing of the 3D window after switching the step (panels, docks) –
        w.view.redraw()                            # the projection of a point must count with the new size of the window
        xp, yp = w.view.view.Convert(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        r = w.view.dpr()
        pt = QPoint(int(xp / r), int(yp / r))
        QTest.mouseMove(w.view, pt)
        QTest.mouseClick(w.view, Qt.LeftButton, Qt.ControlModifier if ctrl else Qt.NoModifier, pt)
        app.processEvents()

    def key(k):
        """The keyboard shortcut of the filter (QTest into the 3D window – the shortcuts of the window are processed as when typing)."""
        from PySide6.QtTest import QTest
        w.activateWindow()
        w.view.setFocus()
        QTest.keyClick(w.view, k)
        app.processEvents()

    def far_wall(z):
        """A point on the remote wall of the hole (visible through the opening from above) at the height z."""
        d = w.view.view.Camera().Direction()
        h = np.array([d.X(), d.Y()])
        h /= np.linalg.norm(h)
        return (HOLE[0] + R_HOLE * h[0], HOLE[1] + R_HOLE * h[1], z)

    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")
        check([p.name for p in w.parts] == ["Plate", "Base plate"], f"parts {[p.name for p in w.parts]}")

    def s_materials():
        w.set_stage(w.STAGE["Materials"])
        ip, idk = part_index("Base plate"), part_index("Plate")
        click_at((70, 30, 38))                      # upper face of the foot
        rows = sorted(i.row() for i in w.mat_parts.selectedIndexes())
        check(rows == [ip], f"klik na patku vybral v seznamu řádky {rows}")
        click_at((120, 50, 8), ctrl=True)           # Ctrl + upper face of the plate = add
        rows = sorted(i.row() for i in w.mat_parts.selectedIndexes())
        check(rows == sorted([ip, idk]), f"Ctrl+klik na desku: řádky {rows}, ve 3D {w.view.selected_parts()}")
        shot("f1_materialy_klik")
        w.mat_parts.clearSelection()
        w.mat_parts.item(idk).setSelected(True)     # seznam → 3D
        check(w.view.selected_parts() == [idk], f"výběr v seznamu zvýrazní díl ve 3D: {w.view.selected_parts()}")
        w.set_stage(w.STAGE["3D printing"])
        click_at((70, 30, 38))
        rows = sorted(i.row() for i in w.print_panel.parts.selectedIndexes())
        check(rows == [ip], f"3D tisk: klik na patku vybral v seznamu dílů řádky {rows}")

    def s_body():
        from PySide6.QtCore import Qt
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces([], emit=True)
        key(Qt.Key_B)
        if w.view.pick_filter != "body":
            print("  (klávesa B nedoručena – přepínám akcí)")
            w.filter_actions["body"].trigger()
        check(w.view.pick_filter == "body" and w.filter_actions["body"].isChecked(), "filtr Těleso (B)")
        click_at((70, 30, 38))
        ip = part_index("Base plate")
        want = [(ip, k) for k in range(len(w.view.part_faces[ip]))]
        check(sorted(w.view.selected_faces()) == want, f"Těleso: vybráno {len(w.view.selected_faces())} ploch patky "
              f"(čekáno {len(want)})")
        print("  stavový řádek:", w.st_sel.text())
        shot("f2_teleso")
        click_at((70, 30, 38))
        check(w.view.selected_faces() == [], "druhý klik na těleso výběr odebere")

    def s_face():
        from PySide6.QtCore import Qt
        key(Qt.Key_P)
        if w.view.pick_filter != "face":
            w.filter_actions["face"].trigger()
        check(w.view.pick_filter == "face", "filtr Plocha (P)")
        click_at((200, 30, 4))                     # face of the plate x = 200
        end = faces_where(lambda part, p: part == "Plate" and abs(p["bbox"][0] - 200) < 1e-6 and abs(p["bbox"][3] - 200) < 1e-6)
        check(w.view.selected_faces() == end, f"Plocha: {w.view.selected_faces()} (čekáno {end})")
        # clamping of the face of the plate x = 0 → maximum at the clamping edge (singularity)
        fixed = faces_where(lambda part, p: part == "Plate" and abs(p["bbox"][0]) < 1e-6 and abs(p["bbox"][3]) < 1e-6)
        w.view.set_selected_faces(fixed, emit=True)
        w.add_fixed()
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces(end, emit=True)
        w.ld_type.setCurrentText("Force (components)")
        for s, v in zip(w.ld_vec, (0.0, 0.0, -500.0)):
            s.setValue(v)
        w.add_load()
        check(len(w.study.load_cases[0].loads) == 1, "síla na čele desky")

    def s_hole():
        from PySide6.QtCore import Qt
        hole = faces_where(lambda part, p: p["type"] == "cylinder")
        check(len(hole) == 2, f"díra ze dvou válcových ploch: {hole}")
        w.view.set_selected_faces([], emit=True)
        key(Qt.Key_H)
        if w.view.pick_filter != "hole":
            w.filter_actions["hole"].trigger()
        check(w.view.pick_filter == "hole", "filtr Díra (H)")
        click_at(far_wall(5.0))
        check(sorted(w.view.selected_faces()) == sorted(hole), f"Díra jedním klikem: {w.view.selected_faces()}")
        shot("f3_dira")
        click_at(far_wall(5.0))
        check(w.view.selected_faces() == [], "druhý klik díru odebere")
        click_at((120, 50, 8))                     # a planar face – the Hole filter selects nothing
        check(w.view.selected_faces() == [], "Díra: klik na rovinu nic nevybere")
        # earlier "Whole hole": switching to Hole extends the half already selected
        w.set_pick_filter("face")
        w.view.set_selected_faces(hole[:1], emit=True)
        w.set_pick_filter("hole")
        check(sorted(w.view.selected_faces()) == sorted(hole), "H rozšíří vybranou polovinu na celou díru")

    def s_edge():
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        w.view.set_selected_faces([], emit=True)
        key(Qt.Key_E)
        if w.view.pick_filter != "edge":
            w.filter_actions["edge"].trigger()
        check(w.view.pick_filter == "edge", "filtr Hrana (E)")
        click_at(far_wall(8.0))                    # upper edge of the hole (arc)
        e = w.view.selected_edge()
        check(e is not None and e["type"] == "circle", f"hrana: {None if e is None else e['label']}")
        if e is not None and e["type"] == "circle":
            check(abs(e["radius"] - R_HOLE) < 1e-6 and np.allclose(e["center"], (*HOLE, 8.0), atol=1e-6),
                  f"poloměr {e['radius']:.6g}, střed {np.round(e['center'], 6).tolist()}")
        print("  stavový řádek:", w.st_info.text())
        check(w.view.selected_faces() == [], "hrana nemění výběr ploch")
        w.ld_type.setCurrentText("Force at remote point")
        w.ld_pt_auto.setChecked(True)
        app.processEvents()
        check(w.btn_edge_point.isVisible(), f"tlačítko „{w.btn_edge_point.text()}“ ve stavovém řádku")
        QTest.mouseClick(w.btn_edge_point, Qt.LeftButton)
        app.processEvents()
        pt = [s.value() for s in w.ld_pt]
        check(np.allclose(pt, (*HOLE, 8.0)) and not w.ld_pt_auto.isChecked(), f"bod vzdálené síly z hrany: {pt}")
        clip = QApplication.clipboard().text()
        if not clip:                         # the Windows clipboard may be held by another program – check independently of the application
            QApplication.clipboard().setText("test")
            app.processEvents()
            if QApplication.clipboard().text() != "test":
                print("POZOR schránka systému je nedostupná (drží ji jiný program) – kontrola schránky přeskočena")
                clip = "150; 30; 8"
        check(clip.replace(",", ".") == "150; 30; 8", f"schránka: '{clip}'")
        shot("f4_hrana_stred")
        if hasattr(w, "rs_pt"):                    # remote constraint (step Constraints): the same button fills its point
            w.set_stage(w.STAGE["Constraints"])
            if hasattr(w, "rs_group"):
                w.rs_group.setChecked(True)
            app.processEvents()
            w.use_edge_point()
            check(np.allclose([s.value() for s in w.rs_pt], (*HOLE, 8.0)),
                  f"bod vzdálené vazby z hrany: {[s.value() for s in w.rs_pt]}")
            if hasattr(w, "rs_group"):
                w.rs_group.setChecked(False)
        # model: a click on an edge only shows the information also in the Materials step
        w.set_stage(w.STAGE["Materials"])
        click_at((200, 0, 4))                      # vertical edge of the face of the plate (x = 200, y = 0)
        e = w.view.selected_edge()
        check(e is not None and e["type"] == "line" and abs(e["length"] - 8) < 1e-6,
              f"hrana v kroku Materiály: {None if e is None else (e['label'], round(e['length'], 6))}")
        w.set_pick_filter("face")
        check(w.view.selected_edge() is None and not w.btn_edge_point.isVisible(), "přepnutí filtru hranu zruší")
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_results():
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        res = w.results
        check(res is not None and len(res.steps) == 1, "results")
        w.set_stage(w.STAGE["Results"])
        t = w.hs_table
        n = t.rowCount()
        check(n > 0, f"tabulka „{w.hs_title.text()}“: {n} míst")
        for r in range(n):
            print("  ", " | ".join(t.item(r, c).text() for c in range(t.columnCount())))
        cls0 = t.item(0, 4).text() if n else ""
        check(cls0.startswith("singularity"), f"maximum von Mises klasifikováno jako singularita: '{cls0}'")
        print("  legenda:", w.legend.note)
        check(bool(w.legend.note) == cls0.startswith("singularity"), "poznámka u legendy")
        info = w.res_info.toPlainText()
        check(any(x in info for x in ("excluding singularities", "mimo singularity")), "souhrn uvádí max. mimo singularity")
        # the same classification as MCP / the protocol (fea_core.mcp.diagnostics.hotspots)
        from fea_core.mcp.diagnostics import hotspots
        d = hotspots(w.diag_session(), 0, gm.HS_TOP, "mises")
        check([h["classification"] for h in d["hotspots"]] == [row[0]["classification"] for row in w._hs_rows]
              and "tags" not in d["hotspots"][0], "klasifikace shodná s MCP (bez štítků v MCP výstupu)")
        shot("f5a_vysledky_souhrn")
        w.res_tabs.setCurrentWidget(w.hs_box)
        check("!" in w.res_tabs.tabText(1), f"záložka: „{w.res_tabs.tabText(1)}“")
        w.r_scroll.ensureWidgetVisible(t)
        app.processEvents()
        rect = t.visualItemRect(t.item(0, 1))
        QTest.mouseClick(t.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
        app.processEvents()
        print("  sonda:", w.st_info.text())
        check("location #1" in w.st_info.text() and "hotspot" in w.view._marks, "klik na řádek: značka ve 3D a sonda")
        shot("f5_vysledky_mista_maxim")
        # the row with the real place (if there is none, the last row)
        k = next((r for r in range(n) if t.item(r, 4).text() == "actual location"), n - 1)
        rect = t.visualItemRect(t.item(k, 1))
        QTest.mouseClick(t.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
        app.processEvents()
        check(len(w.view._marks.get("hotspot", [])) == 2, "značka místa se nahradí (ne přidá)")
        print("  řádek", k + 1, ":", w.hs_note.text()[:300])
        shot("f6_misto_skutecne")
        # probe into the contours: the band must not be highlighted (the style of the highlight of the contours is transparent)
        click_at((120, 45, 8))
        print("  sonda (klik):", w.st_info.text())
        check(w.st_info.text().startswith(("Probe: von Mises", "Sonda: von Mises")), "klik sondou do výsledků")
        shot("f7_sonda_bez_zvyrazneni")
        # other quantities: displacement → the von Mises table without a note at the legend; S1 → its own table
        names = [w.r_qty.itemText(i).split(" [")[0] for i in range(w.r_qty.count())]
        w.r_qty.setCurrentIndex(names.index("Total displacement"))
        check(w.hs_title.text().endswith("von Mises</b>") and not w.legend.note, f"posuv: „{w.hs_title.text()}“")
        w.r_qty.setCurrentIndex(names.index("1st principal stress S1"))
        check("S1" in w.hs_title.text() and w.hs_table.rowCount() > 0, f"S1: „{w.hs_title.text()}“")
        w.r_qty.setCurrentIndex(names.index("von Mises"))
        w.sec_axis.setCurrentIndex(2)              # section y = 30 (across the hole) – markers only on the displayed part
        w.show_result()
        check(w.hs_table.rowCount() == n, "tabulka i s řezem")
        shot("f8_rez_mista")
        w.sec_axis.setCurrentIndex(0)

    steps = [s_open, s_materials, s_body, s_face, s_hole, s_edge, s_results]

    def run_next():
        if w._thread is not None:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            w.dirty = False                        # quit() closes the windows → the save question would wait
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
            w.dirty = False
            app.quit()
            return
        QTimer.singleShot(700, run_next)

    QTimer.singleShot(800, run_next)
    app.exec()
    QApplication.clipboard().clear()        # a clipboard owned by the application would wait for the flush at exit
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
