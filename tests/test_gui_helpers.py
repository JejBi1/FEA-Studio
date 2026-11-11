"""GUI helper functions without a window (Qt offscreen): the form of a bolt must not silently change the size / class by saving."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_bolt_combo_keeps_unknown_values(app):
    from fea_core.joints import tables
    from fea_gui.joints_panel import _select_or_add
    c = QtWidgets.QComboBox()
    for k in tables.PITCH:
        if k >= 3:
            c.addItem(f"M{k:g}", float(k))
    _select_or_add(c, 2.5, "M2.5")                        # below M3 – not in the menu
    assert c.currentData() == 2.5 and c.currentText().endswith("(from model)")
    _select_or_add(c, 8.0, "M8")                          # the temporary item disappears
    assert c.currentData() == 8.0 and not any(c.itemText(i).endswith("(from model)") for i in range(c.count()))
    k = QtWidgets.QComboBox()
    k.addItems(tables.CLASS_NAMES)
    _select_or_add(k, "8.8", "8.8")
    assert (k.currentData() or k.currentText()) == "8.8"
    _select_or_add(k, "14.9", "14.9")                     # unknown class (a file from a newer version)
    assert (k.currentData() or k.currentText()) == "14.9"


def test_messages_stage_and_sources(app):
    """Message table: the step by the text, replacement of the messages of the source, level filter, copying."""
    from fea_gui.messages import MessagesPanel, stage_for

    class Win:
        STAGE = {"Constraints": 4, "Materiály": 1}
        stage = None

        def set_stage(self, k):
            self.stage = k
    assert stage_for("Elastic support 'x': the stiffness must be positive", "study check") == "Constraints"
    assert stage_for("Orthotropic material 'A': E1 is missing", "study check") == "Materials"
    assert stage_for("something else", "results") == "Results"
    w = Win()
    mp = MessagesPanel(w)
    mp.set_source("study check", [("error", "The study has no support"), ("warning", "Material without density")])
    mp.add("info", "řešič", "Výpočet dokončen")
    assert len(mp.messages()) == 3 and mp.table.rowCount() == 3
    t0 = mp.messages(level="error")[0]["time"]
    mp.set_source("study check", [("error", "The study has no support")])
    assert len(mp.messages()) == 2 and mp.messages(level="error")[0]["time"] == t0
    mp.filter.setCurrentIndex(mp.filter.findData("error"))
    assert mp.table.rowCount() == 1
    assert "error\tstudy check\tThe study has no support" in mp.copy()
    mp.goto(0)
    assert w.stage == 4
    mp.clear()
    assert [m["source"] for m in mp.messages()] == ["study check"]


def test_mode_helpers():
    import numpy as np
    from fea_core.eigen import ModeSet
    from fea_gui.eigen_gui import mass_fractions, mode_label, mode_quantity
    ms = ModeSet("modal", "Natural frequencies", [10.0, 20.5], np.zeros((2, 3, 3), np.float32),
                 eff_mass=[[0.5, 0, 0, 0, 0, 0], [0, 0.25, 0, 0, 0, 0]], total_mass=[1.0, 1.0, 1.0, 0, 0, 0])
    assert mode_label(ms, 1) == "Mode shape 2: f = 20.5 Hz"
    fr = mass_fractions(ms)
    assert np.allclose(fr, [[0.5, 0, 0], [0, 0.25, 0]])
    sh = np.array([[3.0, 4.0, 0.0]])
    assert mode_quantity(sh, 0)[0] == 5.0 and mode_quantity(sh, 2)[0] == 4.0
    b = ModeSet("buckling", "Vzpěr – S", [21.65], np.zeros((1, 3, 3), np.float32), case="S")
    assert mode_label(b, 0) == "Buckling mode 1 (S): λ = 21.65"


def test_frame_helpers():
    """Frames in the GUI: the axis of an arc member, local axes, quantities of the 2D results."""
    import numpy as np
    from fea_core.beams.model import FrameMember, SectionDef
    from fea_core.study import Material, Study
    from fea_gui.frame_panel import member_axes, member_polyline
    from fea_gui.frame_results import FRAME_QUANTITIES, band_colors, is_frame_quantity
    st = Study("t", "")
    st.add_material(Material("Ocel", 210000.0, 0.3, 7.85e-9), default=True)
    fr = st.frame
    fr.sections["IPE 200"] = SectionDef("IPE 200", library="IPE 200")
    fr.add_node("A", [0, 0, 0])
    fr.add_node("B", [2000, 0, 0])
    fr.members["p"] = FrameMember("p", "A", "B", "IPE 200")
    fr.members["o"] = FrameMember("o", "A", "B", "IPE 200", through=[1000, 0, 500])
    P = member_polyline(st, fr.members["o"])
    assert len(P) == 25 and np.allclose(P[0], [0, 0, 0]) and np.allclose(P[-1], [2000, 0, 0])
    R = np.linalg.norm(np.asarray(P) - [1000, 0, -750], axis=1)          # circle through 3 points: centre z = −750
    assert np.allclose(R, 1250.0)
    ax = member_axes(st, fr.members["p"])
    assert np.allclose(ax[0], [1, 0, 0]) and np.allclose(ax[2], [0, 0, 1])   # from the section = global Z
    assert all(is_frame_quantity(q) for q, _ in FRAME_QUANTITIES) and not is_frame_quantity("Von Mises")
    assert len(band_colors(12)) == 12


def test_model_ignore_parts(app):
    """Check of the logic of excluding / returning the parts from the calculation in the model (MainWindow.set_ignored, apply_ghost)."""
    from fea_core.study import Material, Study
    st = Study("test_study", "")
    st.add_material(Material("Ocel", 210000.0, 0.3, 7.85e-9), default=True)
    st.ignored_parts = ["PartB"]

    class FakeView:
        def __init__(self):
            self.ghost = []

        def set_ghost(self, indices):
            self.ghost = list(indices)

    class FakePart:
        def __init__(self, name):
            self.name = name

    class FakeWin:
        def __init__(self):
            self.study = st
            self.view = FakeView()
            self.parts = [FakePart("PartA"), FakePart("PartB"), FakePart("PartC")]
            self.logs = []
            self.changed_called = False
            self.mesh_invalidated = False

        from fea_gui.main import MainWindow
        apply_ghost = MainWindow.apply_ghost
        set_ignored = MainWindow.set_ignored

        def log_msg(self, msg):
            self.logs.append(msg)

        def changed(self, invalidate_mesh=False):
            self.changed_called = True
            self.mesh_invalidated = invalidate_mesh

    win = FakeWin()
    win.apply_ghost()
    assert win.view.ghost == [1]

    # Exclude PartC
    win.set_ignored(["PartC"], True)
    assert "PartC" in st.ignored_parts
    assert "PartB" in st.ignored_parts
    assert win.view.ghost == [1, 2]
    assert win.mesh_invalidated is True

    # Return PartB
    win.set_ignored(["PartB"], False)
    assert "PartB" not in st.ignored_parts
    assert "PartC" in st.ignored_parts
    assert win.view.ghost == [2]
