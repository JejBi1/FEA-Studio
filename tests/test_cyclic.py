"""Cyclic symmetry (fea_core.cyclic): node pairs, equations, per-sector reactions, expansion – synthetic meshes (no Gmsh, no ccx).
The ccx runs are the verification cases CY1, CY2 (python -m verification.run_all CY1 CY2)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fea_core.cyclic import (CyclicError, CyclicSymmetry, axis_from_planes, equations, pairs, post_step, rotation,
                             rotate_stress, summary)
from fea_core.cyclic.expand import expand_mesh, expand_results, _mode_blocks, powers
from fea_core.cyclic.modal import groups, read_dat
from fea_core.mesh import Mesh
from fea_core.study import Study, Fixed

N = 6
TH = 2 * math.pi / N


def sector_mesh(nr=3, nt=4, nz=2, ri=10.0, ro=20.0, h=5.0, theta=TH, with_axis=False):
    """Structured linear tet mesh of a ring sector (hex → 6 tets); ri = 0 with_axis → nodes on the axis."""
    if with_axis:
        ri = 0.0
    rs = np.linspace(ri, ro, nr + 1)
    ts = np.linspace(0.0, theta, nt + 1)
    zs = np.linspace(0.0, h, nz + 1)
    idx = -np.ones((nr + 1, nt + 1, nz + 1), int)
    nodes = []
    for i, r in enumerate(rs):
        for j, t in enumerate(ts):
            for k, z in enumerate(zs):
                if r == 0.0 and j > 0:
                    idx[i, j, k] = idx[i, 0, k]
                    continue
                idx[i, j, k] = len(nodes)
                nodes.append([r * math.cos(t), r * math.sin(t), z])
    tets = []
    split = [(0, 1, 3, 7), (0, 1, 7, 5), (0, 5, 7, 4), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7)]
    for i in range(nr):
        for j in range(nt):
            for k in range(nz):
                c = [idx[i + a, j + b, k + cc] for cc in (0, 1) for b in (0, 1) for a in (0, 1)]
                for s in split:
                    q = [c[v] for v in s]
                    if len(set(q)) == 4:
                        tets.append(q)
    nodes = np.array(nodes, float)
    tets = np.array(tets, int)

    def face(jsel):
        tri = []
        for i in range(nr):
            for k in range(nz):
                a, b = idx[i, jsel, k], idx[i + 1, jsel, k]
                c_, d = idx[i + 1, jsel, k + 1], idx[i, jsel, k + 1]
                for t in ((a, b, c_), (a, c_, d)):
                    if len(set(t)) == 3:
                        tri.append(t)
        return np.array(tri, int)
    bore = []
    for j in range(nt):
        for k in range(nz):
            a, b = idx[0, j, k], idx[0, j + 1, k]
            c_, d = idx[0, j + 1, k + 1], idx[0, j, k + 1]
            if len({a, b, c_, d}) == 4:
                bore += [(a, b, c_), (a, c_, d)]
    surfaces = {"L": face(0), "R": face(nt)}
    if bore:
        surfaces["bore"] = np.array(bore, int)
    return Mesh(nodes, tets, np.zeros(len(tets), np.int32), ["Disc"], surfaces, {})


def study_with(mesh, **kw) -> Study:
    st = Study(name="t", geometry="")
    st.cyclic = CyclicSymmetry(sectors=N, left="L", right="R", **kw)
    return st


def test_pairs_match_rotation_and_reject_reversed_axis():
    m = sector_mesh()
    st = study_with(m)
    p = pairs(st, m)
    R = rotation([0, 0, 1], TH)
    assert len(p["left"]) == len(p["right"]) == 4 * 3
    assert np.abs(m.nodes[p["left"]] @ R.T - m.nodes[p["right"]]).max() < 1e-12
    st.cyclic.axis_dir = [0, 0, -1]
    with pytest.raises(CyclicError, match="reverse the axis"):
        pairs(st, m)
    st.cyclic.axis_dir = [0, 0, 1]
    st.cyclic.sectors = 5
    with pytest.raises(CyclicError, match="do not match"):
        pairs(st, m)


def test_axis_nodes_and_equations():
    m = sector_mesh(with_axis=True)
    st = study_with(m)
    p = pairs(st, m)
    assert len(p["axis"]) == 3                              # nz + 1 nodes on the axis
    eqs = equations(st, m)
    assert len(eqs) == 3 * len(p["left"]) + 2 * len(p["axis"])
    Rt = rotation([0, 0, 1], TH).T
    n0, terms = eqs[0]
    assert terms[0][:2] == (n0, 0) and terms[0][2] == 1.0
    coef = {(n, k): c for n, k, c in terms[1:]}
    r0 = int(p["right"][0])
    assert coef[(r0, 0)] == pytest.approx(-Rt[0, 0]) and coef[(r0, 1)] == pytest.approx(-Rt[0, 1])
    # a cyclic field satisfies every equation
    U = np.zeros((len(m.nodes), 3))
    r = np.hypot(m.nodes[:, 0], m.nodes[:, 1])
    t = np.arctan2(m.nodes[:, 1], m.nodes[:, 0])
    U[:, 0], U[:, 1], U[:, 2] = 1e-3 * r * np.cos(t), 1e-3 * r * np.sin(t), 2e-3 * m.nodes[:, 2]
    for _, terms in eqs:
        assert abs(sum(c * U[n, k] for n, k, c in terms)) < 1e-12


def test_axis_from_planes_and_summary():
    n1 = [0, -1, 0]                                         # face y = 0 (normal out of the sector)
    n2 = [-math.sin(TH), math.cos(TH), 0]
    x0, d, n, ang = axis_from_planes([0, 0, 0], n1, [15, 0, 2.5], [0, 0, 0], n2,
                                     [15 * math.cos(TH), 15 * math.sin(TH), 2.5])
    assert n == N and ang == pytest.approx(60.0)
    assert np.allclose(d, [0, 0, 1]) and np.linalg.norm(x0[:2]) < 1e-12
    m = sector_mesh()
    s = summary(study_with(m), m)
    assert s["sectors"] == N and s["equations"] == 3 * s["cut_pairs"]


def test_post_step_moves_reactions_to_the_right_face():
    m = sector_mesh()
    st = study_with(m)
    st.constraints.append(Fixed("bore"))
    p = pairs(st, m)
    rng = np.random.default_rng(1)
    RF = rng.normal(size=(len(m.nodes), 3))
    F = rng.normal(size=(len(m.nodes), 3)) * 0.1
    S = rng.normal(size=(len(m.nodes), 6))
    allc = np.zeros((len(m.nodes), 3), bool)
    allc[m.surface_nodes("bore")] = True
    RF2, S2, info = post_step(st, m, RF, F, S, allc)
    R = p["R"]
    L, Rn = p["left"], p["right"]
    assert np.allclose(RF2[L], F[L])
    assert np.allclose(RF2[Rn] - F[Rn], (RF - F)[Rn] + (RF - F)[L] @ R.T)
    # total of (RF − F) = reactions seen by the solver + the forces across the cut
    G, G2 = RF - F, RF2 - F
    cut = np.r_[L, Rn]
    assert np.allclose((G[cut] - np.where(allc[cut], G2[cut], 0.0)).sum(axis=0), info["cut_force"])
    # stresses at the cut: one value seen from both sides
    assert np.allclose(S2[Rn], rotate_stress(S2[L], R))


def test_expand_mesh_and_fields():
    m = sector_mesh(with_axis=True)
    st = study_with(m)
    info = summary(st, m)
    full, gid, Rk = expand_mesh(m, info)
    p = pairs(st, m)
    n_exp = N * len(m.nodes) - N * len(p["left"]) - (N - 1) * len(p["axis"])
    assert len(full.nodes) == n_exp
    # no two nodes at the same place, all elements of positive volume
    from scipy.spatial import cKDTree
    assert len(cKDTree(full.nodes).query_pairs(1e-9)) == 0
    X = full.nodes[full.tets]
    vol = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0]) / 6
    Xs = m.nodes[m.tets]
    vs = np.einsum("ij,ij->i", np.cross(Xs[:, 1] - Xs[:, 0], Xs[:, 2] - Xs[:, 0]), Xs[:, 3] - Xs[:, 0]) / 6
    assert np.abs(vol).sum() == pytest.approx(N * np.abs(vs).sum())
    assert "L" not in full.surfaces and "R" not in full.surfaces


def test_mode_expansion_phase():
    R = rotation([0, 0, 1], TH)
    Rk = powers(R, N)
    re = np.array([[1.0, 0.0, 0.0]])
    im = np.array([[0.0, 1.0, 0.0]])
    b = _mode_blocks(re, im, 2, N, Rk)
    phi = 2 * math.pi * 2 / N
    for k in range(N):
        expect = (math.cos(k * phi) * re - math.sin(k * phi) * im) @ Rk[k].T
        assert np.allclose(b[k], expect)


def test_expand_results_of_a_static_state():
    from fea_core.results import Results, StepResult
    m = sector_mesh()
    st = study_with(m)
    U = np.zeros((len(m.nodes), 3))
    U[:, :2] = 1e-3 * m.nodes[:, :2]                         # radial expansion = cyclic field
    S = np.tile([1.0, 2.0, 0.0, 0.5, 0.0, 0.0], (len(m.nodes), 1))
    res = Results(m, [StepResult("a", U, S)], [], {"cyclic": summary(st, m)})
    full = expand_results(res)
    assert full is expand_results(res)                       # cached
    assert np.allclose(full.step(0).U[:, :2], 1e-3 * full.mesh.nodes[:, :2])
    assert np.allclose(full.step(0).S[:len(m.nodes)][0], S[0])


def test_modal_groups_and_dat():
    assert groups([0, 1, 2, 3], has_axis=False) == [(0, 3)]
    assert groups([0, 1, 2, 3, 4], has_axis=True) == [(0, 0), (1, 1), (2, 4)]
    assert groups([0, 2, 3], has_axis=False) == [(0, 0), (2, 3)]
    import tempfile
    from pathlib import Path
    txt = ("  NODAL   MODE NO    EIGENVALUE                      FREQUENCY\n"
           "    2          1   0.9252589E+09   0.3041807E+05   0.4841186E+04   0.0000000E+00\n"
           "    2          2   0.9252589E+09   0.3041807E+05   0.4841186E+04   0.0000000E+00\n"
           "      1   0.5743351E-02  -0.1340001E-01   0.7427091E-17   0.6700018E-01   0.2871708E-01  -0.1202086E+01\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.dat"
        p.write_text(txt)
        rows = read_dat(p)
    assert rows == [(2, 1, 0.9252589e9, 4841.186), (2, 2, 0.9252589e9, 4841.186)]


def test_study_json_roundtrip_and_old_files():
    st = Study(name="t", geometry="x.step")
    assert "cyclic" not in st.to_dict()
    from fea_core.signature import mesh_signature
    sig0 = mesh_signature(st)
    st.cyclic = CyclicSymmetry(sectors=12, left="a", right="b", axis_dir=[1.0, 0, 0], nodal_diameters=[0, 3])
    d = st.to_dict()
    assert "cyclic" in d["features"]
    back = Study.from_dict(d)
    assert back.cyclic == st.cyclic
    assert mesh_signature(st) != sig0


@pytest.mark.slow
def test_sector_run_report_and_saved_results(tmp_path):
    """Small sector with ccx: equilibrium with the cut forces, saved and reloaded cyclic results, Word report section."""
    import math as _m
    from verification.cases import solid, _study
    from verification.cyclic_cases import ring_sector
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.study import MeshSettings, Rotation, LoadCase
    from fea_core.solve import run_study
    from fea_core.results import Results
    from fea_core.report import ReportOptions, build_report
    from fea_core.report.word import session_from
    th = _m.pi / 4
    st, parts = _study("cyc", tmp_path / "s.step", [Part("S", solid(ring_sector(20.0, 60.0, 0.0, 8.0, th)))],
                       MeshSettings(size=8.0))
    nr = np.array([-_m.sin(th), _m.cos(th), 0.0])
    st.add_selection("L", find_faces(parts, y=0))
    st.add_selection("R", find_faces(parts, normal=nr))
    st.add_selection("bore", find_faces(parts, radius=20.0))
    st.constraints.append(Fixed("bore"))
    st.load_cases.append(LoadCase("w", [Rotation([0, 0, 0], [0, 0, 1], omega=500.0)]))
    st.modal.enabled = True
    st.modal.modes = 1
    st.cyclic = CyclicSymmetry(sectors=8, left="L", right="R", nodal_diameters=[0, 1])
    res = run_study(st, tmp_path / "run", parts=parts)
    s = res.step(0)
    assert s.equilibrium["imbalance"] < 1e-6 and "cyclic" in s.equilibrium
    assert [r for r in res.modes[0].nodal_diameters] == [0, 1]
    back = Results.load(tmp_path / "run" / "results.npz")
    assert back.info["cyclic"]["sectors"] == 8 and back.modes[0].nodal_diameters == [0, 1]
    full = expand_results(back)
    assert len(full.mesh.tets) == 8 * len(back.mesh.tets)
    sess = session_from(st, tmp_path / "s.step", res, parts)
    r = build_report(sess, tmp_path / "r.docx", ReportOptions(images=True))
    import zipfile
    xml = zipfile.ZipFile(tmp_path / "r.docx").read("word/document.xml").decode("utf-8")
    assert "Cyclic symmetry" in xml and "nodal diameter" in xml and "Full 360° model" in xml, r.get("warnings")


def test_rigid_rows_leave_axial_motions_free():
    from fea_core.cyclic import rigid_rows
    m = sector_mesh()
    st = study_with(m, axis_point=[5.0, -3.0, 0.0])
    c = np.array([1.0, 2.0, 3.0])
    rows = rigid_rows(st, c)
    assert np.linalg.matrix_rank(rows) == 4
    x0 = np.array([5.0, -3.0, 0.0])
    axial = np.r_[0, 0, 1.0, 0, 0, 0]                        # translation along the axis
    spin = np.r_[np.cross([0, 0, 1.0], c - x0), 0, 0, 1.0]   # rotation about the axis line (displacement at c = ω × (c − x0))
    assert np.allclose(rows @ axial, 0) and np.allclose(rows @ spin, 0)
