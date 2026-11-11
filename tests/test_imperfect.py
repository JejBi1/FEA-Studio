"""Fast tests of the imperfections and the nonlinear-buckling curve evaluation (fea_core.imperfect) – no Gmsh, no ccx."""
from __future__ import annotations

import json

import numpy as np
import pytest

from fea_core.imperfect import curve
from fea_core.imperfect.model import ImperfectionError, NonlinearBucklingSettings
from fea_core.imperfect.shape import (amplitude, bow, from_file, imperfect_mesh, imperfection_field, mode_combination,
                                      reference_length)
from fea_core.mesh import Mesh


def _mesh(n=11, L=100.0):
    x = np.linspace(0, L, n)
    nodes = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    return Mesh(nodes, np.zeros((0, 10), int), np.zeros(0, int), ["P"])


def test_mode_combination_scaled_to_unit_and_weighted():
    sh = np.zeros((2, 4, 3))
    sh[0, :, 2] = [0, 0.5, 1.0, 0.5]
    sh[1, :, 1] = [0, 1.0, 0, -1.0]
    w = mode_combination(sh, [1, 2], [1.0, -0.5])
    assert np.linalg.norm(w, axis=1).max() == pytest.approx(1.0)
    assert w[2, 2] == pytest.approx(1.0 / np.linalg.norm([0, 0, 1.0]))
    assert np.sign(w[1, 1]) == -1                      # negative weight flips mode 2
    with pytest.raises(ImperfectionError):
        mode_combination(sh, [3], [1.0])


def test_bow_is_half_sine_perpendicular_and_unit():
    m = _mesh(21, 200.0)
    w = bow(m.nodes, [0, 0, 0], [200, 0, 0], [0.3, 0, 1.0])      # axial part of the direction is removed
    assert np.allclose(w[:, 0], 0.0) and np.allclose(w[:, 1], 0.0)
    assert w[10, 2] == pytest.approx(1.0)
    assert w[5, 2] == pytest.approx(np.sin(np.pi * 0.25))
    assert w[0, 2] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ImperfectionError):
        bow(m.nodes, [0, 0, 0], [200, 0, 0], [1, 0, 0])


def test_amplitude_mm_or_ratio_and_field_sign():
    m = _mesh(21, 300.0)
    s = NonlinearBucklingSettings(imperfection="bow", bow_a=[0, 0, 0], bow_b=[300, 0, 0], bow_direction=[0, 0, 1],
                                  amplitude_ratio=300.0)
    assert reference_length(m, s) == pytest.approx(300.0)
    assert amplitude(m, s) == pytest.approx(1.0)
    s.amplitude, s.sign = 2.5, -1.0
    w = imperfection_field(m, s, None)
    assert w[10, 2] == pytest.approx(-2.5)
    s.length, s.amplitude = 1000.0, 0.0
    assert amplitude(m, s) == pytest.approx(1000.0 / 300.0)
    s.imperfection = "none"
    assert not imperfection_field(m, s, None).any()


def test_imperfect_mesh_is_a_copy():
    m = _mesh()
    w = np.zeros_like(m.nodes)
    w[:, 2] = 1.0
    mi = imperfect_mesh(m, w)
    assert np.allclose(mi.nodes[:, 2], 1.0) and np.allclose(m.nodes[:, 2], 0.0)
    assert mi.nodes is not m.nodes and mi._cache is not m._cache


def test_shape_from_csv_nearest_point(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("x;y;z;ux;uy;uz\n0;0;0;0;0;0\n50;0;0;0;0;2\n100;0;0;0;0;0\n", encoding="utf-8")
    m = _mesh(11)
    w = from_file(m.nodes, p)
    assert w[5, 2] == pytest.approx(1.0) and w[0, 2] == 0.0
    with pytest.raises(ImperfectionError):
        from_file(m.nodes, tmp_path / "missing.csv")


def test_curve_status_limit_max_and_stopped():
    lam = np.linspace(0, 1, 11)
    u = lam / (1 - 0.9 * lam)                          # stiffening loss towards 1/(1 - 0.9)
    ev = curve.evaluate(lam, u, finished=True, max_factor=1.0)
    assert ev["status"] == "max_factor" and ev["ultimate_factor"] == pytest.approx(1.0)
    # stopped with a vanishing tangent → limit point
    lam2 = np.array([0, 0.5, 0.8, 0.95, 0.99, 0.995])
    u2 = np.array([0, 0.5, 1.0, 2.0, 4.0, 6.0])
    ev2 = curve.evaluate(lam2, u2, finished=False, max_factor=2.0, lam_cr=1.2)
    assert ev2["status"] == "limit_point" and ev2["ultimate_factor"] == pytest.approx(0.995)
    assert ev2["knockdown"] == pytest.approx(0.995 / 1.2)
    # stopped while still stiff → "stopped"
    ev3 = curve.evaluate(lam, lam, finished=False, max_factor=2.0)
    assert ev3["status"] == "stopped"
    # descending branch (a curve from displacement control) → limit point at the maximum
    ev4 = curve.evaluate([0, 1, 2, 1.8], [0, 1, 3, 5], finished=True, max_factor=3.0)
    assert ev4["status"] == "limit_point" and ev4["ultimate_index"] == 2
    assert curve.evaluate([], [], False, 1.0)["status"] == "no_increment"


def test_first_crossing_interpolates():
    assert curve.first_crossing([0, 1, 2], [0, 100, 300], 200.0) == pytest.approx(1.5)
    assert curve.first_crossing([0, 1], [0, 10], 20.0) is None


def test_study_json_only_when_used():
    from fea_core.study import Study
    st = Study("t", "")
    d = st.to_dict()
    assert "nlbuckling" not in d and "nlbuckling" not in d["features"]
    st.nlbuckling = NonlinearBucklingSettings(enabled=True, modes=[1, 2], weights=[1.0, 0.3], amplitude=1.5)
    d = json.loads(json.dumps(st.to_dict()))
    assert d["nlbuckling"]["modes"] == [1, 2] and "nlbuckling" in d["features"]
    st2 = Study.from_dict(d)
    assert st2.nlbuckling == st.nlbuckling and not st2.load_warnings
    d["nlbuckling"]["future_key"] = 1                 # unknown keys of a newer version are ignored
    assert Study.from_dict(d).nlbuckling.amplitude == 1.5


def test_check_reports_missing_inputs():
    from fea_core.imperfect.run import check
    from fea_core.study import Study
    st = Study("t", "")
    st.nlbuckling = NonlinearBucklingSettings(enabled=True, imperfection="bow")
    errors, warnings = check(st, [])
    assert any("load case" in e for e in errors) and any("supports" in e for e in errors)
    assert any("bow" in e for e in errors)
    assert any("responsibility" in w for w in warnings)
    st.nlbuckling = NonlinearBucklingSettings()
    assert check(st, []) == ([], [])
