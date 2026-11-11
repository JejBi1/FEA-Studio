"""Switching of bodies off in the Model step (the columns Show / Count, Study.hidden_parts) – a programmatic check.

  python tests/gui/gui_bodies_smoke.py OUT_DIR

Assembly T6 (a T-joint with welds): hiding a body = 3D only (the mesh / results stay, undo works), switching off from the calculation
= ignored_parts + translucent, in bulk by name, isolation, saving to .fea and back.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_bodies_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL = []


def check(cond, text):
    print(("OK   " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    from fea_gui import main as gm
    gm.msg = lambda *a, **k: None
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
    w = gm.MainWindow()
    w.resize(1500, 950)
    w.show()
    app.processEvents()
    w.open_file(str(ROOT / "examples" / "user_tests" / "T6_t_joint_welds_a5.step"))
    app.processEvents()
    names = [p.name for p in w.parts]
    check(len(names) >= 3, f"díly: {names}")
    t = w.model_table
    check(t.columnCount() == 6 and t.item(0, 0).checkState() == Qt.Checked, "sloupce Zobrazit / Počítat")

    from fea_core.solve import mesh_study                # a real mesh – must not disappear by merely hiding
    w.study.mesh.size = 8.0
    w.mesh, _ = mesh_study(w.study, w.parts, OUT)
    w.push_undo()
    t.item(0, 0).setCheckState(Qt.Unchecked)
    app.processEvents()
    check(w.study.hidden_parts == [names[0]], f"skryto: {w.study.hidden_parts}")
    check(w.view._hidden == {0} and not w.view.ctx.IsDisplayed(w.view.part_ais[0]), "díl 0 ve 3D nezobrazen")
    check(w.mesh is not None, "skrytí nezahodilo síť")
    check("hidden" in t.item(0, 3).text(), "stav „(skryto)“")
    w.undo()
    app.processEvents()
    check(w.study.hidden_parts == [] and w.view.ctx.IsDisplayed(w.view.part_ais[0]), "zpět vrátí zobrazení")
    check(w.mesh is not None, "zpět skrytí nezahodí síť")

    t.item(1, 1).setCheckState(Qt.Unchecked)
    app.processEvents()
    check(w.study.ignored_parts == [names[1]] and w.view._ghost == {1}, "vyřazení z výpočtu + průhledně")
    check(w.mesh is None, "vyřazení zahodí síť")
    w.set_ignored(names, False)

    svary = w.parts_by_pattern("svar")
    check(bool(svary) and all("svar" in n.lower() for n in svary), f"podle názvu: {svary}")
    w.set_hidden(svary, True)
    check(set(w.study.hidden_parts) == set(svary), "skrytí podle názvu")
    w.isolate_parts([names[0]])
    check(set(w.study.hidden_parts) == set(names[1:]), "izolace jednoho dílu")

    from fea_core.study import Study
    st2 = Study.from_dict(w.study.to_dict())
    check(st2.hidden_parts == w.study.hidden_parts, "hidden_parts v JSON studie")
    w.set_hidden(names, False)
    check("hidden_parts" not in w.study.to_dict(), "bez skrytých dílů se klíč nezapisuje")

    # an assembly from Fusion (the native module): bodies and occurrences hidden in Fusion → hidden in 3D and excluded from the calculation
    sample = ROOT / "scratch" / "native_cad" / "fusion" / "samples" / "ada_neokey.f3z"
    if sample.is_file():
        from fea_core import modules, native_cad
        modules._installed = lambda key: True
        native_cad.cache_dir = lambda: OUT / "native_cache"
        w.dirty = False
        w.open_file(str(sample))
        app.processEvents()
        st = w.study
        check(len(w.parts) == 27 and len(st.hidden_parts) == 12, f"neokey: {len(w.parts)} těles, skrytých "
              f"{len(st.hidden_parts)}")
        check(set(st.hidden_parts) <= set(st.ignored_parts), "skrytá tělesa z CAD jsou vyřazená z výpočtu")
        vis = [p.name for p in w.parts if p.name not in st.hidden_parts]
        check("Board:1" in vis and "Body:1" in vis and len(vis) == 15, f"viditelná: {vis}")
        idx = {p.name: i for i, p in enumerate(w.parts)}
        check(w.view._hidden == {idx[n] for n in st.hidden_parts}, "skrytá tělesa nejsou ve 3D")
    else:
        print("(vzorek sestavy Fusionu není k dispozici – přeskočeno)")

    # an assembly from Inventor (the native module): the .ipt parts at the positions of the occurrences, the subassembly
    iam = ROOT / "scratch" / "native_cad" / "inv" / "dl_vna" / "gpr20_vna_holder-main" / "assemblies" / "vna_holder.iam"
    if iam.is_file():
        from fea_core import modules, native_cad
        modules._installed = lambda key: True
        native_cad.cache_dir = lambda: OUT / "native_cache"
        w.dirty = False
        w.open_file(str(iam))
        app.processEvents()
        names = [p.name for p in w.parts]
        check(len(names) == 37 and "Sharp GP2Y:1" in names, f"sestava Inventoru: {len(names)} těles")
        check(w.model_table.rowCount() == 37, "tabulka těles sestavy Inventoru")
    else:
        print("(vzorek sestavy Inventoru není k dispozici – přeskočeno)")
    w.dirty = False
    w.close()
    print("CHYBY:", FAIL if FAIL else "none")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
