"""F5 step 2 in the GUI (the 3D printing step): thin places, presets, sweep, advice and build directions – with snapshots.

  python tests/gui/gui_print_sweep_smoke.py OUT_DIR

Model: printed bracket 80×20×12 with a hole Ø6 and a rib 2.4 × 8 mm at the top (a thin place → full infill), PLA Bambu,
clamping x = 0, force 100 N on the face x = 80, coarse mesh. Procedure: section shell / core with the thin places, switching
"full infill of thin places", presets (save / load / delete, a faulty file – an own APPDATA in the folder
OUT, the user library is not touched), calculation, sweep 2 × 2 in a thread (table, Pareto, recommendation, chart,
adding to the variants, taking over), a new calculation, advice by the place of failure (a variant from the advice, the
sweep pre-filled), ordering of 2 build directions and taking over the direction.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_print_sweep_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ["APPDATA"] = str(OUT / "appdata")          # presets and the library of materials only for the test

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox, QScrollArea  # noqa: E402


def make_step(path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from fea_core.geometry import Part, write_step
    from verification.cases import box, cyl
    body = BRepAlgoAPI_Fuse(box(0, 0, 0, 80, 20, 12), box(0, 8.8, 12, 80, 2.4, 8)).Shape()
    shape = BRepAlgoAPI_Cut(body, cyl((60, 10, -1), (0, 0, 1), 3.0, 14)).Shape()
    write_step([Part("Držák žebro", shape)], path)


def main():
    import fea_gui.main as gm
    from fea_gui.main import MainWindow
    gm.msg = lambda parent, title, text, icon=None: print(f"[{title}] {text}")   # without modal windows
    msgs = []

    def box_msg(kind):
        return staticmethod(lambda *a, **k: msgs.append((kind, a[2] if len(a) > 2 else "")) or 0)
    QMessageBox.information = box_msg("info")
    QMessageBox.warning = box_msg("warning")
    QMessageBox.critical = box_msg("critical")
    names = []
    QInputDialog.getText = staticmethod(lambda *a, **k: (names.pop(0), True) if names else ("", False))
    app = QApplication(sys.argv)
    w = MainWindow()
    w.ask = lambda *_: True
    w.resize(1600, 1000)
    w.show()
    step = OUT / "drzak_zebro.step"
    make_step(step)
    log = []
    failed = []
    pp = w.print_panel
    sb, ab = pp.sweep_box, pp.advice_box
    PART = "Držák žebro"

    def flush():
        for _ in range(3):
            app.processEvents()

    def shot(name, widget=None):
        flush()
        w.view.redraw()
        flush()
        from PySide6.QtGui import QImage, QPainter
        if widget is not None:
            widget.grab().save(str(OUT / f"{name}.png"))
            log.append(name)
            return
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

    def panel_shot(name):
        shot(name, pp.findChild(QScrollArea).widget())     # the whole content of the panel (also below the slider)

    def plane_x(x):
        from fea_core.geometry import face_props
        out = []
        for pi, faces in enumerate(w.view.part_faces):
            for fi, f in enumerate(faces):
                p = face_props(f)
                if abs(p["bbox"][0] - x) < 1e-6 and abs(p["bbox"][3] - x) < 1e-6:
                    out.append((pi, fi))
        return out

    def select_part():
        pp.parts.clearSelection()
        pp.parts.item(0).setSelected(True)

    # ------------------------------------------------------------------ kroky
    def s_open():
        w.open_file(str(step))
        w.view.set_orientation("iso")
        w.study.mesh.size = 2.0
        w.set_stage(w.STAGE["3D printing"])
        select_part()
        pp.mat.setCurrentText("Tisk PLA Bambu Basic")
        pp.assign_material()
        select_part()
        pp.apply_spec()
        assert w.study.material_of(PART).is_print
        w.set_stage(w.STAGE["Constraints"])
        w.view.set_selected_faces(plane_x(0.0), emit=True)
        w.add_fixed()
        w.set_stage(w.STAGE["Loads"])
        w.view.set_selected_faces(plane_x(80.0), emit=True)
        w.ld_type.setCurrentText("Force (components)")
        for s, v in zip(w.ld_vec, (0.0, 0.0, -100.0)):
            s.setValue(v)
        w.add_load()
        w.set_stage(w.STAGE["Mesh"])
        w.make_mesh()

    def s_thin():
        w.set_stage(w.STAGE["3D printing"])
        select_part()
        flush()
        sp = w.study.printing.spec(PART)
        assert sp.solid_thin and pp.s_solid_thin.isChecked() and pp.s_thin_w.value() == 0.0
        assert any(x in pp.s_thin_auto.text() for x in ("0,84", "0.84")), pp.s_thin_auto.text()
        pp.cut_axis.setCurrentIndex(0)
        pp.cut_pos.setValue(500)
        pp.show_shell()
        thin = w.mesh.cell_data.get("print_thin")
        n_thin = int((thin > 0).sum())
        print("tenká místa:", n_thin, "prvků | legenda:", pp.legend.labels[-1], "|", pp.legend.note)
        assert n_thin > 0 and pp.legend.labels[-1] == "thin region – solid infill" and str(n_thin) in pp.legend.note
        w.view.set_orientation("iso")
        shot("s1_tenka_mista_rez")
        # switch off the full infill of thin places → no thin places in the section, then back
        n_undo = len(w._undo)
        pp.s_solid_thin.setChecked(False)
        assert not pp.s_thin_w.isEnabled()
        pp.apply_spec()
        assert not w.study.printing.spec(PART).solid_thin and len(w._undo) == n_undo + 1
        pp.show_shell()
        thin = w.mesh.cell_data.get("print_thin")
        assert thin is None or not (thin > 0).any()
        assert pp.legend.labels[-1] != "thin region – solid infill"
        select_part()
        pp.s_solid_thin.setChecked(True)
        pp.s_thin_w.setValue(1.5)
        pp.apply_spec()
        sp = w.study.printing.spec(PART)
        assert sp.solid_thin and sp.thin_width == 1.5 and sp.thin_limit == 1.5 and pp.s_thin_auto.text() == ""
        pp.s_thin_w.setValue(0.0)
        pp.apply_spec()
        assert w.study.printing.spec(PART).thin_width == 0.0

    def s_presets():
        from fea_core.printing.presets import BUILTIN_NAME, presets_path
        path = presets_path()
        assert str(path).startswith(str(OUT)), path
        assert pp.preset.findData(BUILTIN_NAME) >= 0 and not pp.btn_preset_del.isEnabled()
        select_part()
        pp.s_walls.setValue(3)
        pp.s_infill.setValue(25.0)
        pp.s_solid_thin.setChecked(False)
        names.append("Moje 3 perimetry")
        pp.save_preset_as()
        data = json.loads(path.read_text(encoding="utf-8"))
        print("předvolby v souboru:", [(d["name"], d["settings"]["walls"], d["settings"]["infill"]) for d in data])
        assert data[0]["name"] == "Moje 3 perimetry" and data[0]["settings"]["walls"] == 3
        assert abs(data[0]["settings"]["infill"] - 0.25) < 1e-9 and data[0]["settings"]["solid_thin"] is False
        assert pp.preset.currentData() == "Moje 3 perimetry" and pp.btn_preset_del.isEnabled()
        # built-in → the form (the build direction stays), then a custom one
        pp.s_dir.setCurrentIndex(2)                         # +X
        pp.preset.setCurrentIndex(pp.preset.findData(BUILTIN_NAME))
        pp.load_preset()
        sp = pp.spec_from_ui("")
        assert sp.walls == 2 and abs(sp.infill - 0.15) < 1e-9 and sp.solid_thin and sp.build_dir == [1.0, 0.0, 0.0]
        pp.preset.setCurrentIndex(pp.preset.findData("Moje 3 perimetry"))
        pp.load_preset()
        sp = pp.spec_from_ui("")
        assert sp.walls == 3 and abs(sp.infill - 0.25) < 1e-9 and not sp.solid_thin
        panel_shot("s2_predvolby_panel")
        pp.delete_preset()
        assert pp.preset.findData("Moje 3 perimetry") < 0 and json.loads(path.read_text(encoding="utf-8")) == []
        # a faulty file → a Czech message, no crash, saving does not overwrite the file
        path.write_text("{nesmysl", encoding="utf-8")
        pp.refresh_presets()
        print("vadný soubor:", pp.preset_warn.text())
        assert pp.preset_warn.isVisibleTo(pp) and "cannot be read" in pp.preset_warn.text()
        assert pp.preset.count() == 1
        msgs.clear()
        names.append("Jiná")
        pp.save_preset_as()
        assert msgs and msgs[0][0] == "critical" and path.read_text(encoding="utf-8") == "{nesmysl"
        path.unlink()
        pp.refresh_presets()
        assert not pp.preset_warn.isVisibleTo(pp)
        # back to the default setting of the part
        pp.preset.setCurrentIndex(pp.preset.findData(BUILTIN_NAME))
        pp.load_preset()
        pp.s_dir.setCurrentIndex(0)
        select_part()
        pp.apply_spec()
        sp = w.study.printing.spec(PART)
        assert sp.walls == 2 and sp.solid_thin and sp.build_dir == [0, 0, 1]
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_sweep():
        assert w.results is not None, "výpočet studie selhal"
        w.set_stage(w.STAGE["3D printing"])
        assert not pp.sweep_section.is_open()
        pp.sweep_section.toggle.click()
        assert pp.sweep_section.is_open() and sb.isVisibleTo(pp)
        res = w.results
        n_undo = len(w._undo)
        sb.walls.setText("2, 4")
        sb.store()
        sb.infill.setText("10 30")
        sb.store()
        sw = w.study.printing.sweep
        print("rozmítání ve studii:", sw.to_dict(), "| zpět:", n_undo, "→", len(w._undo))
        assert sw.walls == [2, 4] and sw.infill == [0.1, 0.3] and len(w._undo) == n_undo + 2
        assert w.results is res, "definice rozmítání nesmí zahodit výsledky studie"
        # a wrong number → a message, the study unchanged
        sb.infill.setText("10, x")
        sb.store()
        assert "is not a number" in sb.count.text() and w.study.printing.sweep.infill == [0.1, 0.3]
        sb.refresh()
        sb.infill.setText("10, 30")
        est = sb.do_estimate()
        print("odhad:", est, "|", sb.status.text())
        assert est["count"] == 4 and est["runs"] == 5 and not est["over_limit"] and est["total_s"] is not None
        sb.max_n.setValue(3)
        sb.store()
        assert sb.do_estimate()["over_limit"] and "exceeds" in sb.status.text()
        sb.max_n.setValue(60)
        sb.store()
        sb.compute()
        assert sb._thread is not None and w._thread is sb._thread and not sb.btn_run.isEnabled()

    def s_sweep_done():
        run = sb.valid_run()
        assert run is not None, sb.status.text()
        rows = [sb.table.item(r, 0).data(0x0100) for r in range(sb.table.rowCount())]
        pareto = [sb.table.item(r, 0).data(0x0100) for r in range(sb.table.rowCount())
                  if sb.table.item(r, 9).text() == "*"]
        rec = [sb.table.item(r, 0).data(0x0100) for r in range(sb.table.rowCount())
               if "(recommended)" in sb.table.item(r, 0).text()]
        for o in run.outcomes:
            print(f"  {o.name}: k = {o.min_safety}, {o.mass_g} g, ok={o.ok} {o.error}")
        print("tabulka:", len(rows), "| Pareto:", pareto, "| doporučená:", rec, "| run:", run.recommended)
        assert len(rows) == 5 and all(o.ok for o in run.outcomes)
        assert pareto and set(pareto) == set(run.sweep["pareto"]) and rec == [run.recommended]
        assert sb.plot.isVisibleTo(pp) and sb.plot.canvas is not None
        assert w.results is not None, "rozmítání nesmí zahodit výsledky studie"
        panel_shot("s3_rozmitani_panel")
        shot("s4_rozmitani", sb)
        # add to the variants
        target = next(n for n in rows if n != "Current settings" and "4 walls" in n and "30 %" in n)
        sb.select_name(target)
        nv = len(w.study.printing.variants)
        names.append("Ze sweepu 4/30")
        sb.add_variant()
        assert len(w.study.printing.variants) == nv + 1 and w.study.printing.variants[-1].name == "Ze sweepu 4/30"
        assert w.study.printing.variants[-1].parts[PART]["walls"] == 4
        assert sb.valid_run() is not None and w.results is not None
        # taking over into the study
        sb.select_name(target)
        sb.adopt()
        sp = w.study.printing.spec(PART)
        print("převzato:", sp.walls, sp.infill)
        assert sp.walls == 4 and abs(sp.infill - 0.3) < 1e-9 and sb.valid_run() is not None
        assert any(sb.table.item(r, 0).data(0x0100) == "Original study settings" for r in range(sb.table.rowCount()))
        w.set_stage(w.STAGE["Solve"])
        w.run_solve()

    def s_advice():
        assert w.results is not None
        w.set_stage(w.STAGE["3D printing"])
        pp.advice_section.set_open(True)
        ab.advise()
        print("rada:", [(i["part"], i["kind"], i["title"], i["suggestion"]) for i in ab.items])
        assert ab.items and ab.list_lay.count() == len(ab.items) + 1
        panel_shot("s5_rada_panel")
        shot("s6_rada", ab)
        nv = len(w.study.printing.variants)
        item = {"part": PART, "kind": "walls", "title": "Test", "text": "", "reason": "", "suggestion": {"walls": 5}}
        sug = next((i for i in ab.items if i.get("suggestion")), item)
        assert ab.try_variant(sug)
        v = w.study.printing.variants[-1]
        print("varianta z rady:", v.name, v.parts)
        assert len(w.study.printing.variants) == nv + 1 and v.parts[sug["part"]] == sug["suggestion"]
        assert w.results is not None
        sitem = next((i for i in ab.items if i.get("sweep")), None) or dict(
            item, sweep={"walls": [2, 4], "infill": [0.15, 0.4]})
        pp.sweep_section.set_open(False)
        ab.sweep_item(sitem)
        sw = w.study.printing.sweep
        print("rozmítání z rady:", sw.to_dict())
        assert pp.sweep_section.is_open() and sw.parts == [sitem["part"]] and sw.walls == sitem["sweep"]["walls"]
        assert sb.walls.text() == ", ".join(str(x) for x in sitem["sweep"]["walls"])
        # build directions (2 directions because of the duration of the test)
        ab.rank_directions = ["+Z", "+X"]
        ab.compute_rank()
        assert ab._thread is not None

    def s_rank_done():
        res = ab.valid_rank()
        assert res is not None, ab.status.text()
        print("směry:", [(r["direction"], r["min_safety"], r["mass_g"], r["current"]) for r in res["ranking"]],
              "| nejlepší:", res["best"])
        assert ab.rtable.rowCount() == 2 and res["best"] in ("+Z", "+X")
        assert "best" in ab.rtable.item(0, 0).text()
        assert any("(current)" in ab.rtable.item(r, 0).text() for r in range(2))
        shot("s7_smery_stavby", ab)
        r = next(i for i in range(2) if ab.rtable.item(i, 0).data(0x0100) == "+X")
        ab.rtable.selectRow(r)
        ab.adopt_rank()
        sp = w.study.printing.spec(PART)
        print("směr převzat:", sp.build_dir)
        assert sp.build_dir == [1.0, 0.0, 0.0] and ab.valid_rank() is not None
        assert "(current)" in ab.rtable.item(r, 0).text()
        select_part()
        flush()
        w.view.set_orientation("iso")
        shot("s8_smer_prevzat")

    steps = [s_open, s_thin, s_presets, s_sweep, s_sweep_done, s_advice, s_rank_done]

    def run_next():
        if w._thread is not None:
            QTimer.singleShot(300, run_next)
            return
        if not steps:
            print("snímky:", ", ".join(log))
            print("VŠE OK")
            w.dirty = False
            app.quit()
            return
        f = steps.pop(0)
        try:
            print(f"--- {f.__name__}")
            f()
        except Exception:
            import traceback
            traceback.print_exc()
            failed.append(f.__name__)
            w.dirty = False
            app.quit()
            return
        QTimer.singleShot(500, run_next)

    QTimer.singleShot(800, run_next)
    app.exec()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
