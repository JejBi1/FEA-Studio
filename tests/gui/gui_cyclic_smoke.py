"""Cyclic symmetry in the GUI (fea_gui.cyclic_gui): section in the Constraints step, axis and N from the cut faces, 3D markers,
results shown as the full 360° model and the sector again.

  python tests/gui/hidden_run.py tests/gui/gui_cyclic_smoke.py OUT_DIR
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "gui_cyclic_out")
OUT.mkdir(parents=True, exist_ok=True)
FAIL = []


def check(cond, text):
    print(("OK   " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    from fea_gui import main as gm
    from fea_core.geometry import Part, write_step
    from verification.cases import solid
    from verification.cyclic_cases import ring_sector
    gm.msg = lambda *a, **k: None
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Discard)
    th = math.pi / 4
    step = OUT / "sector.step"
    write_step([Part("Sektor", solid(ring_sector(20.0, 60.0, 0.0, 8.0, th)))], step)
    w = gm.MainWindow()
    w.resize(1500, 950)
    w.show()
    app.processEvents()
    w.open_file(str(step))
    app.processEvents()
    from fea_core.selectors import find_faces
    from fea_core.study import Fixed, LoadCase, Rotation
    st = w.study
    nr = [-math.sin(th), math.cos(th), 0.0]
    st.add_selection("rez1", find_faces(w.parts, y=0))
    st.add_selection("rez2", find_faces(w.parts, normal=nr))
    st.add_selection("otvor", find_faces(w.parts, radius=20.0))
    st.constraints.append(Fixed("otvor"))
    st.load_cases.append(LoadCase("w", [Rotation([0, 0, 0], [0, 0, 1], omega=500.0)]))
    w.changed()
    w.set_stage(gm.STAGE["Constraints"])
    app.processEvents()
    box = w.cyclic_box
    box.left.setCurrentIndex(box.left.findText("rez1"))
    box.right.setCurrentIndex(box.right.findText("rez2"))
    box.detect()
    app.processEvents()
    c = w.study.cyclic
    check(c is not None and c.enabled and c.sectors == 8, f"osa a N z řezů: {c}")
    check(c is not None and np.allclose(c.axis_dir, [0, 0, 1]), "směr osy +z")
    try:
        w.refresh_sets()                                  # cut faces + axis drawn in the Constraints step
        check(True, "značky řezů a osy ve 3D")
    except Exception as e:  # noqa: BLE001
        check(False, f"značky ve 3D: {e}")
    box.nds.setText("0, 2-3")
    box.on_change()
    check(w.study.cyclic.nodal_diameters == [0, 2, 3], "uzlové průměry 0, 2-3")
    w.study.cyclic.nodal_diameters = None
    w.study.mesh.size = 8.0
    from fea_core.solve import run_study
    res = run_study(w.study, OUT / "run", parts=w.parts)
    w.results = res
    w.mesh = res.mesh
    w.refresh_all()
    w.set_stage(gm.STAGE["Results"])
    app.processEvents()
    w.show_result()
    app.processEvents()
    n_full = len(w._probe[1]) if w._probe else 0
    sector_surface = len(np.unique(res.mesh.boundary_faces()[0]))
    check(n_full > 4 * sector_surface, f"zobrazen model 360° ({n_full} uzlů povrchu, sektor {sector_surface})")
    check(w.results is res, "výsledky sektoru zůstaly ve stavu okna")
    box.expand.setChecked(False)
    app.processEvents()
    n_sec = len(w._probe[1]) if w._probe else 0
    check(0 < n_sec <= sector_surface + 1, f"zobrazen jen sektor ({n_sec})")
    check(w.results is res and w.mesh is res.mesh, "přepnutí zobrazení výsledky nezahodilo")
    from fea_core.study import Study
    st2 = Study.from_dict(w.study.to_dict())
    check(st2.cyclic == w.study.cyclic, "cyklická symetrie v JSON studie")
    print("HOTOVO" if not FAIL else f"CHYBY: {len(FAIL)}")
    w.dirty = False
    w.close()
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
