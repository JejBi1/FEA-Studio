"""A pass through the GUI of the Motion step (F8) with snapshots: the crank mechanism MB1 with geometry – bodies from the parts, masses from the CAD,
the proposal of the joints from the geometry (a pin in a hole, with confirmation), manual joints, the drive (preview of the law), the load of the piston, check of the mobility,
calculation, animation, graph + "Find maximum", transfer to the FEM (the instant of the maximum reaction), saving and reopening.

  python tests/gui/gui_motion_smoke.py OUT_DIR

Geometry: connecting rod MB11 (eyes Ø10 at x = 20 and 100), crank with a pin Ø10 at x = 20, piston with a pin at x = 100; frame without parts.
Check: the acceleration of the piston at φ = 60° = −75 022 mm/s² (MB1, ω = 100 rad/s, r 20, l 80).
"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_motion_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")
os.environ.setdefault("FEA_LANG", "cs")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

FAIL: list[str] = []


def check(cond, text):
    print(("OK    " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def make_step(path):
    import verification.cases  # noqa: F401 – first (cyclic import of the verification modules)
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from fea_core.geometry import Part, write_step
    from verification.cases import box, cyl, solid
    from verification.motion_cases import _rod_shape
    klika = solid(BRepAlgoAPI_Fuse(box(-8, -6, -15, 36, 12, 10), cyl((20, 0, -5), (0, 0, 1), 5.0, 10.0)).Shape())
    pist = solid(BRepAlgoAPI_Fuse(box(90, -10, 5, 20, 20, 10), cyl((100, 0, -5), (0, 0, 1), 5.0, 10.0)).Shape())
    write_step([Part("Klika", klika), Part("Ojnice", _rod_shape()), Part("Píst", pist)], path)


def main():
    import fea_gui.main as gm
    import fea_gui.motion_panel as mp
    from fea_gui.main import MainWindow
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")
    for k in ("warning", "information", "critical"):
        setattr(QMessageBox, k, staticmethod(lambda parent, title, text, *a, **kw: print(f"[{title}] {text}")))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
    mp.SuggestionDialog.exec = lambda self: QDialog.Accepted        # confirmation of the proposal of the joints without a modal window
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    step = OUT / "klika.step"
    make_step(step)
    log = []
    p = w.motion_panel

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
        pa = QPainter(img)
        pa.drawImage(w.view.mapTo(w, w.view.rect().topLeft()), v.scaled(w.view.size()))
        pa.end()
        tmp.unlink(missing_ok=True)
        img.save(str(OUT / f"{name}.png"))
        log.append(name)

    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")
        w.set_stage(w.STAGE["Motion"])
        check(w.STAGE["Motion"] == w.STAGE["Constraints"] + 1 and w.panel_stack.currentWidget() is p, "krok Pohyb za Vazbami")
        # bodies: the frame without parts, then a body for each part
        p.b_name.setText("ground")
        p.b_ground.setChecked(True)
        p.add_body()
        p.bodies_from_parts()
        names = [b.name for b in w.study.mechanism.bodies]
        check(names == ["ground", "Klika", "Ojnice", "Píst"], f"tělesa: {names}")
        p.compute_props()
        mo = p._props.get("Ojnice", {}).get("mass", 0)
        check(mo > 0, f"hmotnost ojnice z CAD {1e3 * mo:.4f} kg")
        p.tabs.setCurrentIndex(0)
        p.b_table.selectRow(2)
        shot("1_telesa")
        p.suggest_geometry()
        check(p._thread is not None, "návrh kloubů z geometrie ve vlákně")

    def s_joints():
        m = w.study.mechanism
        kinds = sorted((j.kind, j.body_a, j.body_b) for j in m.joints)
        check(len(m.joints) == 2 and all(j.selection_a and j.selection_b for j in m.joints),
              f"2 klouby z geometrie s plochami: {kinds}")
        # name the joints by their position (A = crank–connecting rod, B = connecting rod–piston)
        for j in m.joints:
            j.name = "čep A" if abs(j.origin[0] - 20) < 1 else "čep B"
        w.changed()
        p.tabs.setCurrentIndex(1)
        # by hand: the pin of the crank (frame → crank, axis z) and the guide of the piston (frame → piston, axis x)
        p.j_name.setText("čep klika")
        p.j_kind.setCurrentIndex(p.j_kind.findData("revolute"))
        p.j_a.setCurrentIndex(p.j_a.findData(""))
        p.j_b.setCurrentText("Klika")
        mp._set(p.j_o, [0, 0, 0])
        mp._set(p.j_ax, [0, 0, 1])
        p._jsel = {"a": None, "b": None}
        p.add_joint()
        p.j_name.setText("vedení")
        p.j_kind.setCurrentIndex(p.j_kind.findData("prismatic"))
        p.j_b.setCurrentText("Píst")
        mp._set(p.j_o, [100, 0, 0])
        mp._set(p.j_ax, [1, 0, 0])
        p.add_joint()
        check([j.name for j in m.joints][-2:] == ["čep klika", "vedení"] and m.joints[-1].body_a == "", "ruční klouby")
        p.j_table.selectRow(0)
        shot("2_klouby")
        # drive and loads
        p.tabs.setCurrentIndex(2)
        p.d_name.setText("motor")
        p.d_joint.setCurrentText("čep klika")
        p.d_qty.setCurrentIndex(p.d_qty.findData("angular_velocity"))
        p.d_law.setCurrentIndex(p.d_law.findData("constant"))
        p.d_const.setValue(100.0)
        p.add_driver()
        p.l_name.setText("síla na píst")
        p.l_body.setCurrentText("Píst")
        mp._set(p.l_pt, [100, 0, 0])
        mp._set(p.l_F, [-1000, 0, 0])
        p.add_load()
        p.g_on.setChecked(False)
        check(m.drivers[0].quantity == "angular_velocity" and m.loads[0].force[0] == -1000 and m.gravity is None,
              "pohon, zatížení, bez tíhy")
        # preview of the S-curve
        p.d_law.setCurrentIndex(p.d_law.findData("scurve"))
        p.preview_law()
        p.d_table.selectRow(0)
        shot("3_pohon_s_krivka")
        # calculation: one revolution, 72 outputs
        p.tabs.setCurrentIndex(3)
        p.r_kind.setCurrentIndex(p.r_kind.findData("kinematic"))
        p.r_tend.setValue(2 * math.pi / 100)
        p.r_out.setValue(72)
        p.on_run()
        check(abs(m.run.t_end - 2 * math.pi / 100) < 1e-6 and m.run.outputs == 72, "nastavení běhu")
        check(p.r_solver.model().item(1).isEnabled() == __import__("fea_core.modules", fromlist=["x"]).available("chrono"),
              "volba Chrono podle modulu")
        p.check_dof()

    def s_dof():
        d = getattr(p, "last_dof", None)
        check(d is not None and d["dof_driven"] == 0, f"kontrola volnosti: {d and {k: d[k] for k in ('grubler', 'dof_free', 'dof_driven', 'redundant')}}")
        check(any(m["source"] == "pohyb" for m in w.messages.messages()), "zpráva kontroly volnosti v doku Zprávy")
        shot("4_kontrola_volnosti")
        p.run_motion()

    def s_results():
        res = w.motion_result
        check(res is not None, "výsledek pohybu")
        if res is None:
            print("\n".join(w.log.toPlainText().splitlines()[-30:]))
            return
        k = 12                                           # 72 outputs per revolution → 60°
        ip = res.body_names.index("Píst")
        a = res.arrays["acc"][k, ip, 0]
        check(abs(a - (-75022.0)) / 75022 < 1e-3, f"zrychlení pístu v 60° = −75 022 mm/s²: {a:.1f}")
        check(p.tabs.currentIndex() == 4 and p.a_slider.maximum() == len(res.arrays["t"]) - 1,
              f"karta Výsledky, posuvník {p.a_slider.maximum() + 1} snímků")
        p.a_slider.setValue(18)                          # 90°
        app.processEvents()
        check(getattr(w.view, "_part_posed", False), "animace: díly přesunuty")
        shot("5_animace_90st")
        p.g_qty.setCurrentText("reaction")
        p.g_item.setCurrentText("čep B")
        p.find_max()
        mx = getattr(p, "last_max", None)
        check(mx is not None and p.a_slider.value() == mx["index"], f"Najít maximum: {mx and (mx['t'], mx['value'])}")
        shot("6_graf_maximum")
        p.a_play.setChecked(True)
        for _ in range(10):
            app.processEvents()
        p.a_play.setChecked(False)
        # transfer to the FEM: the instant of the maximum reaction in the pin B, the body of the connecting rod
        out = p.export({"mode": "max", "times": [], "quantity": "reakce:čep B", "body": "Ojnice",
                        "support_joints": None, "load_kind": "remote_distributing"})
        lcs = out["load_cases"] if out else []
        check(len(lcs) == 1 and lcs[0] in [lc.name for lc in w.study.load_cases], f"zatěžovací stav z MBD: {lcs}")
        check(any(c.kind == "remote_support" for c in w.study.constraints), "uložení = vzdálená vazba")
        check(w.motion_result is not None, "převod do MKP výsledek pohybu nezahodí")
        shot("7_prevod_mkp")
        w.set_stage(w.STAGE["Loads"])
        check(not getattr(w.view, "_part_posed", False), "mimo krok Pohyb díly v poloze CAD")
        shot("8_zatizeni_z_mbd")

    def s_save():
        w._save(OUT / "klika.fea")
        w.dirty = False
        w.open_file(str(OUT / "klika.fea"))
        m = w.study.mechanism
        check(len(m.bodies) == 4 and len(m.joints) == 4 and m.drivers and m.loads, "mechanismus uložen")
        check(w.motion_result is not None and len(w.motion_result.arrays["t"]) >= 72, "výsledek pohybu po otevření")
        w.set_stage(w.STAGE["Motion"])
        p.tabs.setCurrentIndex(4)
        p.a_slider.setValue(36)
        shot("9_po_otevreni")
        # a change of the mechanism discards the result
        m.drivers[0].data = [50.0]
        w.changed()
        check(w.motion_result is None, "změna pohonu zahodí výsledek pohybu")
        w.undo()

    steps = [s_open, s_joints, s_dof, s_results, s_save]

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
