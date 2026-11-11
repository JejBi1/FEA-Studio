"""A pass through the GUI with the loads of an edge, a point, a local / cylindrical system, hydrostatics and rotation (snapshots).

  python tests/gui/gui_loads_smoke.py OUT_DIR

Model: shaft Ø20 × 200 (axis x) with a collar and a cantilever 200×20×10 next to it; clamping of the faces x = 0. The load states
are created by the form of the Loads step (selections of edges and points by the filter Edge / Point with a real mouse click), the mesh, calculation,
saving and reopening of the .fea. It prints the check of the equilibrium and of the reactions.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_loads_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def make_step(path):
    from verification.cases import box, cyl
    from fea_core.geometry import Part, write_step
    write_step([Part("Shaft", cyl((0, 0, 0), (1, 0, 0), 10, 200)),
                Part("Cantilever", box(0, 40, -5, 200, 20, 10))], path)


def main():
    from fea_gui.main import MainWindow, FORCE, ROTATION
    from PySide6.QtWidgets import QInputDialog
    # modal dialogs of the name (a new load state) without a window – the answer = the pre-filled name
    QInputDialog.getText = staticmethod(lambda *a, **k: (k.get("text", "Case"), True))
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 950)
    w.show()
    step = OUT / "model.step"
    make_step(step)
    log = []

    def shot(name):
        app.processEvents()
        w.view.redraw()
        app.processEvents()
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

    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")
        w.study.load_cases[0].name = "Edge"

    def s_constraints():
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces(faces_where(lambda part, p: abs(p["bbox"][0]) < 1e-6 and abs(p["bbox"][3]) < 1e-6),
                                  emit=True)
        w.add_fixed()

    def s_edge():
        w.set_stage(w.STAGE["Loads"])
        w.set_pick_filter("edge")
        w.view.fit_all()
        app.processEvents()
        click_at((200, 50, 5))               # the upper edge of the face of the cantilever (x = 200, z = 5, y 40…60)
        got = w.view.selected_edges()
        e = w.edge_prop(*got[0]) if got else None
        check(len(got) == 1 and e is not None and abs(e["length"] - 20) < 1e-6 and abs(e["centroid"][2] - 5) < 1e-6,
              f"klik filtrem Hrana vybral hranu čela konzoly: {got}")
        w.ld_type.setCurrentText(FORCE)
        check(w.load_target() == "edges", f"síla půjde na hrany: {w.ld_target.text()}")
        for s_, v in zip(w.ld_vec, (0, 0, -200.0)):
            s_.setValue(v)
        w.add_load()
        l = w.study.load_cases[0].loads[-1]
        check(w.study.selections[l.selection].entity == "edges", f"zatížení '{l.selection}' na hraně")
        shot("1_sila_na_hranu")

    def s_point():
        w.add_lc()
        lc = w.study.load_cases[-1]
        lc.name = "Point"
        w.refresh_all()
        w.lc_combo.setCurrentIndex(len(w.study.load_cases) - 1)
        w.set_pick_filter("point")
        click_at((200, 50, 5))               # a point on the edge (not a vertex) – projected onto the face / edge
        click_at((200, 40, 5))               # vrchol
        pts = w.view.selected_points()
        check(len(pts) == 2 and np.allclose(pts[1][1], (200, 40, 5), atol=1e-6),
              f"filtr Bod: 2 body, druhý = vrchol (200; 40; 5): {pts}")
        w.ld_type.setCurrentText(FORCE)
        for s_, v in zip(w.ld_vec, (0, 0, -100.0)):
            s_.setValue(v)
        w.add_load()
        l = w.study.load_cases[-1].loads[-1]
        check(w.study.selections[l.selection].entity == "points", "zatížení do bodů")
        shot("2_sila_do_bodu")

    def s_csys():
        from fea_core.study import CoordinateSystem
        w.study.coordinate_systems["Osa hřídele"] = CoordinateSystem("Osa hřídele", "cylindrical", [0, 0, 0],
                                                                       [1, 0, 0], [0, 1, 0])
        w.add_lc()
        w.study.load_cases[-1].name = "Krut"
        w.changed()
        w.lc_combo.setCurrentIndex(len(w.study.load_cases) - 1)
        w.set_pick_filter("face")
        w.view.set_selected_faces(faces_where(lambda part, p: part == "Shaft" and abs(p["bbox"][0] - 200) < 1e-6
                                              and abs(p["bbox"][3] - 200) < 1e-6), emit=True)
        w.ld_type.setCurrentText(FORCE)
        w.ld_csys.setCurrentIndex(w.ld_csys.findData("Osa hřídele"))
        check("F_θ" in w.ld_vec_lbl.text(), f"popisek složek ve válcovém systému: {w.ld_vec_lbl.text()}")
        for s_, v in zip(w.ld_vec, (0, 15000.0, 0)):
            s_.setValue(v)
        w.add_load()
        w.ld_csys.setCurrentIndex(0)
        shot("3_valcovy_system")

    def s_hydro():
        w.add_lc()
        w.study.load_cases[-1].name = "Hydrostatika"
        w.changed()
        w.lc_combo.setCurrentIndex(len(w.study.load_cases) - 1)
        w.view.set_selected_faces(faces_where(lambda part, p: part == "Cantilever" and abs(p["bbox"][1] - 40) < 1e-6
                                              and abs(p["bbox"][4] - 40) < 1e-6), emit=True)
        w.ld_type.setCurrentText("Pressure")
        w.ld_p_mode.setCurrentIndex(2)
        w.ld_val.setValue(0.0)
        for s_, v in zip(w.ld_p_org, (0, 0, 5)):
            s_.setValue(v)
        for s_, v in zip(w.ld_p_gdir, (0, 0, -1)):
            s_.setValue(v)
        w.ld_p_rho.setValue(1e6)             # "heavy liquid" (max. field), so that the force is felt
        w.add_load()
        l = w.study.load_cases[-1].loads[-1]
        check(l.variable and l.clip and abs(l.gradient[2] + 1e6 * 1e-12 * 9810) < 1e-15, f"hydrostatický tlak: {l}")
        shot("4_hydrostatika")

    def s_rotation():
        w.add_lc()
        w.study.load_cases[-1].name = "Rotation"
        w.changed()
        w.lc_combo.setCurrentIndex(len(w.study.load_cases) - 1)
        w.ld_type.setCurrentText(ROTATION)
        w.view.set_selected_faces(faces_where(lambda part, p: part == "Shaft" and p["type"] == "cylinder"), emit=True)
        w.rotation_axis_from_selection()
        check(np.allclose([s_.value() for s_ in w.ld_rot_ax], (1, 0, 0)) or
              np.allclose([s_.value() for s_ in w.ld_rot_ax], (-1, 0, 0)), "osa otáčení z válcové plochy = osa x")
        w.ld_rot_rpm.setValue(3000.0)
        for k in range(w.ld_rot_parts.count()):
            it = w.ld_rot_parts.item(k)
            from PySide6.QtCore import Qt
            it.setCheckState(Qt.Checked if it.text() == "Shaft" else Qt.Unchecked)
        w.add_load()
        l = w.study.load_cases[-1].loads[-1]
        check(l.kind == "rotation" and abs(l.omega - 3000 * 2 * math.pi / 60) < 1e-9 and l.parts == ["Shaft"],
              f"otáčení 3000 ot/min jen hřídele: {l}")
        shot("5_otaceni")
        w.make_mesh()

    def s_solve():
        shot("6_sit")
        w.run_solve()

    def s_results():
        res = w.results
        check(res is not None and len(res.steps) == 5, "výsledky 5 stavů")
        if res is None:
            return
        for st in res.steps:
            check(st.equilibrium["imbalance"] < 1e-4, f"{st.name}: rovnováha {st.equilibrium['imbalance']:.2e}")
        names = [n for n in res.steps[0].reactions]
        print("reakce:", names)
        tot = lambda k: np.sum([np.asarray(r) for r in res.steps[k].reactions.values()], axis=0)
        check(abs(tot(0)[2] - 200) < 1e-2, f"hrana: ΣRz = {tot(0)[2]:.4f} (200)")
        check(abs(tot(1)[2] - 100) < 1e-2, f"body: ΣRz = {tot(1)[2]:.4f} (100)")
        rho = w.study.material_of("Shaft").rho
        w_ = 3000 * 2 * math.pi / 60
        check(np.linalg.norm(tot(4)) < 1e-3 * rho * w_ ** 2 * 10 * math.pi * 100 * 200,
              f"otáčení kolem osy hřídele: výslednice ≈ 0 ({np.linalg.norm(tot(4)):.3g} N)")
        w.r_step.setCurrentIndex(w.r_step.findData("step:2"))
        shot("7_vysledky_krut")
        w._save(OUT / "zatizeni.fea")

    def s_reload():
        w.open_file(str(OUT / "zatizeni.fea"))
        st = w.study
        kinds = [(l.kind, st.selections[l.selection].entity if getattr(l, "selection", None) else "-")
                 for lc in st.load_cases for l in lc.loads]
        check(kinds == [("force", "edges"), ("force", "points"), ("force", "faces"), ("pressure", "faces"),
                        ("rotation", "-")], f"po otevření .fea: {kinds}")
        check("Osa hřídele" in st.coordinate_systems, "souřadný systém uložen")
        check(all(v for v in w.sel_edges.values()) and all(v for v in w.sel_points.values()),
              "hrany a body výběrů nalezeny v geometrii")
        w.set_stage(w.STAGE["Loads"])
        w.lc_combo.setCurrentIndex(0)
        w.load_list.setCurrentRow(0)
        check(w.view.selected_edges() == list(w.sel_edges[st.load_cases[0].loads[0].selection]),
              "výběr zatížení zvýrazní jeho hranu")
        shot("8_znovu_otevreno")

    steps = [s_open, s_constraints, s_edge, s_point, s_csys, s_hydro, s_rotation, s_solve, s_results, s_reload]

    def run_next():
        if w._thread is not None:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            print("snímky:", ", ".join(log))
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
    print("VÝSLEDEK:", "OK" if not FAIL else f"{len(FAIL)} chyb: " + "; ".join(FAIL))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
