"""Plasticity in the GUI: PlasticityBox ↔ Material.plastic (PlasticitySpec.to_dict) mapping, table paste / conversion,
validation, and the core evaluation (plasticity_eval) on synthetic results (Qt offscreen – no 3D view)."""
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from fea_core import plasticity_eval as PE                           # noqa: E402
from fea_core.plasticity import PlasticitySpec, plastic_table, validate_material   # noqa: E402
from fea_core.study import LoadCase, Material, Study                 # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeWin:
    def __init__(self, E=200000.0, Re=250.0, Rm=400.0):
        from fea_gui.widgets import dspin
        self.m_name = QtWidgets.QLineEdit("Steel")
        self.m_E = dspin(E, 1, 1e7, 0)
        self.m_nu = dspin(0.3, 0, 0.499, 3)
        self.m_Re = dspin(Re, 0, 1e5, 0)
        self.m_Rm = dspin(Rm, 0, 1e5, 0)
        self.study = None
        self.logs, self.n_changed = [], 0

    def changed(self):
        self.n_changed += 1

    def log_msg(self, t):
        self.logs.append(t)


def box(win=None):
    from fea_gui.plasticity_box import PlasticityBox
    w = win or FakeWin()
    return PlasticityBox(w), w


def mat(plastic=None, **kw):
    return Material(kw.pop("name", "Steel"), kw.pop("E", 200000.0), 0.3, 7.85e-9, Re=kw.pop("Re", 250.0),
                    Rm=kw.pop("Rm", 400.0), plastic=plastic, **kw)


def test_elastic_material_is_off(app):
    b, _ = box()
    b.load(mat())
    assert not b.isChecked() and b.values() is None


@pytest.mark.parametrize("spec", [
    PlasticitySpec("bilinear", "isotropic", Et=2000.0),
    PlasticitySpec("bilinear", "kinematic", Et=0.0, limit_strain=0.04),
    PlasticitySpec("multilinear", "isotropic", table=[(250.0, 0.0), (300.0, 0.01), (340.0, 0.05)]),
    PlasticitySpec("ramberg_osgood", "isotropic", K_prime=700.0, n_prime=0.2),
    PlasticitySpec("hollomon", "isotropic", Rm=420.0, A=0.22),
])
def test_roundtrip_spec(app, spec):
    """Material → form → values() gives exactly the dict the core reads (no second storage)."""
    b, _ = box()
    m = mat(spec.to_dict())
    b.load(m)
    assert b.isChecked()
    got = PlasticitySpec.from_dict(b.values())
    want = PlasticitySpec.from_dict(spec.to_dict())
    assert got.model == want.model and got.hardening == want.hardening
    assert got.limit_strain == pytest.approx(want.limit_strain)
    if want.model == "bilinear":
        assert got.Et == pytest.approx(want.Et)
    if want.model == "multilinear":
        assert got.table == pytest.approx(want.table)
    if want.model == "ramberg_osgood":
        assert (got.K_prime, got.n_prime) == pytest.approx((want.K_prime, want.n_prime))
    if want.model == "hollomon":
        assert (got.Rm, got.A) == pytest.approx((want.Rm, want.A))
    # same *PLASTIC points as the solver gets
    assert plastic_table(got, m.Re, m.E) == pytest.approx(plastic_table(want, m.Re, m.E))
    assert "●" in b.status.text()


def test_plastic_modulus_input_is_stored_as_tangent_modulus(app):
    b, w = box()
    b.load(mat(PlasticitySpec("bilinear", Et=2000.0).to_dict()))
    b.bi_kind.setCurrentIndex(b.bi_kind.findData("H"))       # the same physical slope is kept when switching
    E = w.m_E.value()
    assert b.bi_val.value() == pytest.approx(E * 2000.0 / (E - 2000.0), rel=1e-4)
    b.bi_val.setValue(25000.0)                                 # H = 25 GPa → Et = E·H/(E + H)
    assert b.values()["Et"] == pytest.approx(E * 25000.0 / (E + 25000.0))


def test_paste_and_engineering_conversion(app):
    b, w = box()
    b.load(mat())
    b.setChecked(True)
    b.model.setCurrentIndex(b.model.findData("multilinear"))
    b.paste_text("sigma;eps\n250;0,00125\n300;0,02\n350;0,06\n")
    assert b.table_rows() == [(250.0, 0.00125), (300.0, 0.02), (350.0, 0.06)]
    b.convert_engineering()
    rows = b.table_rows()
    E = w.m_E.value()
    s, ep = 300 * 1.02, math.log(1.02) - 300 * 1.02 / E
    ep0 = max(0.0, math.log(1.00125) - 250 * 1.00125 / E)      # at the yield point ε_pl = 0 (clipped)
    assert rows[0][1] == 0.0 and rows[1] == pytest.approx((s, ep - ep0), rel=1e-4)
    assert not validate_material(b._trial_material())[0]


def test_invalid_table_is_reported(app):
    b, _ = box()
    b.load(mat(PlasticitySpec("multilinear", table=[(250.0, 0.0), (240.0, 0.01)]).to_dict()))
    assert "must not decrease" in b.status.text()


@pytest.mark.parametrize("text, rows", [
    ("250\t0\n300\t0.01", [(250.0, 0.0), (300.0, 0.01)]),
    ("250,0\n300,0.01", [(250.0, 0.0), (300.0, 0.01)]),
    ("250 0\n300 1e-2", [(250.0, 0.0), (300.0, 0.01)]),
    ("250;0\n300,5;0,01", [(250.0, 0.0), (300.5, 0.01)]),
])
def test_parse_table(text, rows):
    from fea_gui.plasticity_box import parse_table
    assert parse_table(text) == pytest.approx(rows)


def test_parse_table_ambiguous():
    from fea_gui.plasticity_box import parse_table
    with pytest.raises(ValueError):
        parse_table("250,5,0,01")


def test_validate_material_rules():
    assert not validate_material(mat(PlasticitySpec("bilinear", Et=2000.0).to_dict()))[0]
    assert validate_material(mat(PlasticitySpec("bilinear", Et=2000.0).to_dict(), Re=None))[0]       # no Re fallback
    assert validate_material(mat(PlasticitySpec("bilinear", Et=None).to_dict()))[0]
    assert validate_material(mat(PlasticitySpec("bilinear", Et=3e5).to_dict()))[0]                  # Et ≥ E
    assert validate_material(mat(PlasticitySpec("ramberg_osgood", K_prime=200.0, n_prime=0.2).to_dict()))[0]
    assert validate_material(mat(PlasticitySpec("hollomon", Rm=200.0, A=0.2).to_dict()))[0]
    e, w = validate_material(mat(PlasticitySpec("multilinear", table=[(260.0, 0.0), (300.0, 0.02)]).to_dict(), Re=200))
    assert not e and w                                                                               # Re ≠ table yield
    assert validate_material(mat(PlasticitySpec("bilinear", Et=2000.0).to_dict(), ortho={"E1": 1}))[0]


def test_add_unloading_step():
    st = Study("s", "x.step")
    st.load_cases += [LoadCase("A", []), LoadCase("B", [])]
    assert PE.add_unloading_step(st)
    seq = st.analysis.sequence
    assert [s.name for s in seq] == ["A", "B", PE.UNLOADING_STEP] and seq[-1].case == "B" and seq[-1].scale == 0.0
    assert not st.analysis.branches and st.analysis.nonlinear
    assert not PE.add_unloading_step(st)                        # already ends with unloading


def _results(S, peeq, names=("P",), seq_names=None):
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    mesh = SimpleNamespace(nodes=nodes, tets=np.array([[0, 1, 2, 3]]), tet_part=np.array([0]), part_names=list(names))
    from fea_core.results import von_mises
    steps = [SimpleNamespace(name=n, S=S, fields={"peeq": peeq}, partial=False, mises=von_mises(S))
             for n in (seq_names or ["L"])]
    return SimpleNamespace(mesh=mesh, steps=steps)


def test_assess_uniaxial_and_triaxial():
    """Uniaxial tension (triaxiality 1/3): ASME ε_L = ε_Lu = m2 = 0.6·(1 − 250/400); triaxial: ε_L by the formula."""
    st = Study("s", "x.step")
    st.add_material(mat(PlasticitySpec("bilinear", Et=2000.0, limit_strain=0.05).to_dict()), default=True)
    S = np.zeros((4, 6))
    S[:, 0] = 300.0
    peeq = np.array([0.01, 0.02, 0.0, -1e-6])
    res = _results(S, peeq)
    en = PE.assess(st, res, 0, "en1993")
    assert en["max_peeq"] == pytest.approx(0.02) and en["utilization"] == pytest.approx(0.02 / 0.05)
    assert en["where_peeq"]["node"] == 2 and en["where_peeq"]["parts"] == ["P"]
    am = PE.assess(st, res, 0, "asme")
    m2 = 0.6 * (1 - 250 / 400)
    assert am["triaxiality_at_max"] == pytest.approx(1 / 3) and am["utilization"] == pytest.approx(0.02 / m2)
    S2 = np.tile([400.0, 300.0, 300.0, 0, 0, 0], (4, 1))
    am2 = PE.assess(st, _results(S2, peeq), 0, "asme")
    T = (1000 / 3) / 100.0
    assert am2["limit_at_max"] == pytest.approx(m2 * math.exp(-2.2 / (1 + m2) * (T - 1 / 3)))
    aus = PE.assess(st, res, 0, "asme", PE.AsmeParams("austenitic", eps_cf=0.01))
    m2a = 0.75 * (1 - 250 / 400)
    assert aus["utilization"] == pytest.approx((0.02 + 0.01) / m2a)
    with pytest.raises(ValueError):
        PE.assess(st, _results(S, peeq), 0, "nonsense")


def test_assess_residual_and_curve_end():
    st = Study("s", "x.step")
    st.add_material(mat(PlasticitySpec("multilinear", table=[(250.0, 0.0), (300.0, 0.01)]).to_dict()), default=True)
    st.load_cases.append(LoadCase("A", []))
    PE.add_unloading_step(st)
    S = np.zeros((4, 6))
    S[:, 0] = 120.0
    res = _results(S, np.full(4, 0.02), seq_names=["A", PE.UNLOADING_STEP])
    a = PE.assess(st, res, 1)
    assert a["residual"]["max_mises_MPa"] == pytest.approx(120.0) and "Steel" in a["curve_end_exceeded"]
    assert "residual" not in PE.assess(st, res, 0)
    rows = PE.summary(st, res)
    assert len(rows) == 2 and rows[0]["asme"] is not None


def test_stress_strain_curve_matches_table():
    m = mat(PlasticitySpec("bilinear", Et=20000.0).to_dict())
    eps, sig = PE.stress_strain(m)
    assert sig[0] == 0 and sig[1] == pytest.approx(250.0) and eps[1] == pytest.approx(250.0 / 200000.0)
    H = 200000 * 20000 / 180000
    assert sig[-1] == pytest.approx(250 + H * 0.2) and eps[-1] == pytest.approx(sig[-1] / 200000 + 0.2)
