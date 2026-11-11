"""Fast tests of the core (without Gmsh and ccx)."""
import numpy as np
import pytest

from fea_core.ccx.frd import read_frd
from fea_core.ccx.inp import consistent_nodal_forces
from fea_core.mesh import gmsh_to_ccx_tet10, surface_element_faces, MeshError
from fea_core.results import von_mises, principal, tet10_shape
from fea_core.solve import equilibrium
from fea_core.study import Study, Material, Fixed, Force, Pressure, LoadCase, RemoteLoad
from fea_core.selectors import FaceSelector

FRD = """\
    1C
    2C                             2                                     1
 -1         1 0.00000E+00 0.00000E+00 0.00000E+00
 -1         2 1.00000E+00 0.00000E+00 0.00000E+00
 -3
    1PSTEP                         1           1           1
  100CL  101 1.000000000           2                     0    1           1
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -5  ALL         1    2    0    0    1ALL
 -1         1 1.20000E+02-4.50000E+01 0.00000E+00
 -1         2-1.00000E-03-2.00000E-03-3.00000E-03
 -3
    1PSTEP                         2           1           2
  100CL  101 1.000000000           2                     0    1           2
 -4  DISP        4    1
 -5  D1          1    2    1    0
 -5  D2          1    2    2    0
 -5  D3          1    2    3    0
 -5  ALL         1    2    0    0    1ALL
 -1         1 1.00000E+00 2.00000E+00 3.00000E+00
 -1         2 4.00000E+00 5.00000E+00 6.00000E+00
 -3
 9999
"""


def test_frd_fixed_columns(tmp_path):
    p = tmp_path / "t.frd"
    p.write_text(FRD)
    f = read_frd(p)
    assert f.steps() == [1, 2]
    b = f.get("DISP", 1)
    assert b.components == ["D1", "D2", "D3"]
    np.testing.assert_allclose(b.values[0], [120.0, -45.0, 0.0])   # glued numbers
    np.testing.assert_allclose(f.get("DISP", 2).values[1], [4, 5, 6])


def _tet10(scale=1.0):
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) * scale
    mids_gmsh = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]
    nodes = np.vstack([x] + [0.5 * (x[i] + x[j]) for i, j in mids_gmsh])
    return nodes, np.arange(10)[None, :]


def test_tet10_reorder():
    nodes, tets = _tet10()
    t = gmsh_to_ccx_tet10(tets, nodes)
    x = nodes[t[0]]
    np.testing.assert_allclose(x[8], 0.5 * (x[1] + x[3]))
    np.testing.assert_allclose(x[9], 0.5 * (x[2] + x[3]))
    with pytest.raises(MeshError):  # a double swap must fail
        gmsh_to_ccx_tet10(t, nodes)


def test_shape_functions_partition_of_unity():
    xi = np.random.default_rng(0).random((20, 3)) / 3
    np.testing.assert_allclose(tet10_shape(xi).sum(axis=1), 1.0)


def test_surface_element_faces():
    nodes, tets = _tet10()
    t = gmsh_to_ccx_tet10(tets, nodes)
    tri = np.array([[0, 1, 3, 4, 8, 7]])  # wall 1-2-4 → S2 (1-4-2)
    e, f = surface_element_faces(t, tri)
    assert e[0] == 0 and f[0] == 2


def test_consistent_forces_tri6():
    nodes = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    tri = np.array([[0, 1, 2, 3, 4, 5]])
    f = consistent_nodal_forces(nodes, tri, np.array([0, 0, -1.0]))  # 1 MPa na 2 mm²
    assert set(f) == {3, 4, 5}                                         # rohy 0
    np.testing.assert_allclose(sum(f.values()), [0, 0, -2.0])


def test_stress_invariants():
    s = np.array([100.0, 0, 0, 0, 0, 0])
    assert von_mises(s) == pytest.approx(100.0)
    tau = np.array([0, 0, 0, 50.0, 0, 0])
    assert von_mises(tau) == pytest.approx(50 * np.sqrt(3))
    np.testing.assert_allclose(principal(tau), [50, 0, -50], atol=1e-12)


def test_equilibrium_detects_wrong_reactions():
    x = np.array([[0, 0, 0], [1, 0, 0]], float)
    F = np.array([[0, 0, 0], [0, 0, -100.0]])
    ok = equilibrium(x, F, np.array([[0, 0, 100.0], [0, 0, 0]]), np.zeros(3), np.array([0, -100.0, 0]))
    assert ok["imbalance"] < 1e-12
    bad = equilibrium(x, F, np.array([[0, 0, -136.8], [0, 0, 0]]), np.zeros(3), np.array([0, -100.0, 0]))
    assert bad["imbalance"] > 0.1


def test_study_json_roundtrip(tmp_path):
    st = Study("t", "m.step")
    st.add_material(Material("Ocel", 210000, 0.3, 7.85e-9))
    sel = FaceSelector("A", "plane", 1.0, [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0, 0, 1, 1])
    st.add_selection("s", [sel])
    st.constraints.append(Fixed("s"))
    st.load_cases.append(LoadCase("LC", [Force("s", [1, 2, 3]), Pressure("s", 5.0),
                                         RemoteLoad("s", [0, 0, 0], moment=[1, 0, 0])]))
    st.save(tmp_path / "s.json")
    st2 = Study.load(tmp_path / "s.json")
    assert st2.to_dict() == st.to_dict()
