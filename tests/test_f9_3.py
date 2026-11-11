"""Unit tests for stage F9.3:
- A7: monitor of the convergence of the nonlinear calculation (parse_cvg, parse_sta, stdout residuals, nodes)
- C1: submodel (Submodeling: geometry, cut nodes, interpolation of the displacements, check of the cut)
- A5: display of the results (
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from fea_core.nl_convergence import (
    parse_cvg, parse_sta, parse_stdout_residuals, read_convergence_job,
    ConvergenceHistory, ConvergenceIteration
)
from fea_core.submodel import (
    SubmodelSpec, identify_cut_nodes, interpolate_cut_displacements, verify_cut_boundary,
    generate_submodel_mesh, create_submodel_shape, locate_batch, write_submodel_inp
)
from fea_core.study import Material
from fea_core.post_vectors import (
    principal_stresses_and_directions, principal_vector_segments
)
from fea_core.isosurface import extract_isosurface
from fea_core.results import Results, StepResult, von_mises
from fea_core.mesh import Mesh


# ===================================================================== A7: Monitor konvergence
def test_parse_cvg():
    cvg_text = (
        " 1  1  1  1  0  0.250000E+03  0.120000E-02  0.000000E+00  0.000000E+00\n"
        " 1  1  1  2  0  0.150000E+01  0.450000E-04  0.000000E+00  0.000000E+00\n"
        " 2  1  1  1  0  0.800000E+02  0.800000E-03  0.000000E+00  0.000000E+00\n"
    )
    records = parse_cvg(cvg_text)
    assert len(records) == 3
    assert records[0].iteration == 1
    assert records[0].step == 1
    assert records[1].iteration == 2
    assert records[1].resid_force_pct == pytest.approx(1.50)
    assert records[2].step == 2


def test_parse_sta():
    sta_text = (
        " SUMMARY OF JOB INFORMATION:\n"
        " STEP  INC ATT ITRS  TOTAL TIME  STEP TIME  INC TIME\n"
        "   1     1   1    2  0.100000    0.100000   0.100000\n"
        "   1     2   1    1  0.300000    0.300000   0.200000\n"
    )
    records = parse_sta(sta_text)
    assert len(records) == 2
    assert records[0]["step"] == 1
    assert records[0]["inc"] == 1
    assert records[0]["iterations"] == 2
    assert records[1]["inc"] == 2
    assert records[1]["step_time"] == pytest.approx(0.3)


def test_parse_stdout_residuals():
    stdout_text = """
 average force= 1.250000e+03
 largest residual force= 4.560000e+01 in node 1420 and dof 1
 largest correction to disp= 1.200000e-02 in node 850 and dof 2
 average force= 1.250000e+03
 largest residual force= 8.100000e-01 in node 1420 and dof 1
 largest correction to disp= 2.300000e-05 in node 850 and dof 2
"""
    res = parse_stdout_residuals(stdout_text)
    assert len(res) == 2
    assert res[0]["worst_force_val"] == pytest.approx(45.6)
    assert res[0]["worst_force_node"] == 1420
    assert res[0]["worst_force_dof"] == 1
    assert res[0]["worst_disp_node"] == 850


def test_read_convergence_job(tmp_path: Path):
    cvg = tmp_path / "test.cvg"
    cvg.write_text(" 1 1 1 1 0 10.0 0.01 0.0 0.0\n")
    sta = tmp_path / "test.sta"
    sta.write_text(" 1 1 1 1 0.1 0.1 0.1\n")
    hist = read_convergence_job(tmp_path, "test")
    assert len(hist.iterations) == 1
    assert len(hist.increments) == 1
    d = hist.to_dict()
    assert d["total_iterations"] == 1


# ===================================================================== C1: Submodel
def test_identify_cut_nodes_box():
    nodes = np.array([
        [0.0, 0.0, 5.0],      # interior point
        [30.0, 0.0, 5.0],     # cut face x = +30
        [-30.0, 15.0, 5.0],   # cut face x = -30
        [10.0, -30.0, 2.0],   # cut face y = -30
        [15.0, 15.0, 5.0],    # interior point
    ])
    spec = SubmodelSpec(center=(0.0, 0.0, 5.0), size=(60.0, 60.0, 10.0), shape="box")
    cut_idx = identify_cut_nodes(nodes, spec, tol=1e-3)
    assert set(cut_idx) == {1, 2, 3}


def test_identify_cut_nodes_sphere():
    nodes = np.array([
        [0.0, 0.0, 0.0],     # centre
        [10.0, 0.0, 0.0],    # na povrchu r=10
        [0.0, -10.0, 0.0],   # na povrchu r=10
        [5.0, 5.0, 0.0],     # inside
    ])
    spec = SubmodelSpec(center=(0.0, 0.0, 0.0), size=10.0, shape="sphere")
    cut_idx = identify_cut_nodes(nodes, spec, tol=1e-3)
    assert set(cut_idx) == {1, 2}


def test_verify_cut_boundary():
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [5.0, 0.0, 0.0],
        [5.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
        [5.0, 0.0, 5.0],
        [0.0, 5.0, 5.0],
    ])
    tets = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    m_glob = Mesh(nodes, tets, np.zeros(1, dtype=np.int32), ["Part"])
    S_glob = np.ones((10, 6)) * 20.0
    step_g = StepResult("Step", np.zeros((10, 3)), S_glob)
    res_glob = Results(m_glob, [step_g], [], {})

    m_sub = Mesh(nodes[:4], np.array([[0, 1, 2, 3]]), np.zeros(1, dtype=np.int32), ["Sub"])
    S_sub = np.ones((4, 6)) * 20.0
    step_s = StepResult("Step", np.zeros((4, 3)), S_sub)
    res_sub = Results(m_sub, [step_s], [], {})

    check = verify_cut_boundary(res_glob, res_sub, np.array([0, 1, 2, 3]), step_index=0)
    assert check["passed"] is True
    assert check["mean_rel_diff"] < 1e-4


def test_locate_batch():
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [5.0, 0.0, 0.0],
        [5.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
        [5.0, 0.0, 5.0],
        [0.0, 5.0, 5.0],
    ])
    tets = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    m_glob = Mesh(nodes, tets, np.zeros(1, dtype=np.int32), ["Part"])
    res_glob = Results(m_glob, [StepResult("Step", np.zeros((10, 3)), np.zeros((10, 6)))], [], {})

    pts = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    best_elems, best_xis = locate_batch(res_glob, pts)
    assert len(best_elems) == 2
    assert best_elems[0] == 0
    assert best_elems[1] == 0
    assert np.all(best_xis >= -1e-9)
    assert np.all(best_xis.sum(axis=1) <= 1.0 + 1e-9)


def test_write_submodel_inp_special_chars(tmp_path: Path):
    """Check that write_submodel_inp safely handles the diacritics / spaces in the name of the material and uses solver_name."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tets = np.array([[0, 1, 2, 3]])
    mat = Material(name="Ocel 11 523, třída 4 (speciální)", E=210000.0, nu=0.3)
    cut_idx = np.array([0, 1])
    cut_u = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])

    inp_path = tmp_path / "submodel.inp"
    write_submodel_inp(coords, tets, mat, cut_idx, cut_u, inp_path)

    text = inp_path.read_text(encoding="ascii")
    assert "*MATERIAL, NAME=M1" in text
    assert "MATERIAL=M1" in text
    assert "*STATIC, SOLVER=" in text


def test_submodel_non_main_thread(tmp_path: Path):
    """Check that the generation of the mesh of the submodel in a separate process works also from a secondary thread without a ValueError."""
    import threading
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir

    box_shape = BRepPrimAPI_MakeBox(gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0, 0, 1)), 20.0, 20.0, 10.0).Shape()
    spec = SubmodelSpec(center=(10.0, 10.0, 5.0), size=(10.0, 10.0, 10.0), shape="box", mesh_size_max=5.0)

    res = {}
    err = []

    def thread_fn():
        try:
            sub_shape = create_submodel_shape(box_shape, spec)
            coords, tets = generate_submodel_mesh(sub_shape, spec, workdir=tmp_path / "sub_thread")
            res["nodes"] = coords
            res["tets"] = tets
        except Exception as e:
            err.append(e)

    th = threading.Thread(target=thread_fn)
    th.start()
    th.join()

    assert not err, f"Chyba ve vedlejším vlákně: {err[0]}"
    assert len(res["nodes"]) > 0
    assert len(res["tets"]) > 0


# ===================================================================== A5: principal stress vectors
def test_principal_stresses_and_directions():
    S = np.array([
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [50.0, 50.0, 0.0, 20.0, 0.0, 0.0],
    ])
    vals, vecs = principal_stresses_and_directions(S)
    
    assert vals[0, 0] == pytest.approx(100.0)
    assert vals[0, 1] == pytest.approx(0.0)
    assert vals[0, 2] == pytest.approx(0.0)
    assert abs(vecs[0, 0, 0]) == pytest.approx(1.0)
    
    assert vals[1, 0] == pytest.approx(0.0)
    assert vals[1, 2] == pytest.approx(-50.0)
    assert abs(vecs[1, 0, 2]) == pytest.approx(1.0)


def test_principal_vector_segments():
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    S = np.array([
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    seg1 = principal_vector_segments(nodes, S, which=1, scale=0.1, min_stress_fraction=0.0)
    assert len(seg1["starts"]) == 2
    assert seg1["is_tension"][0] == True
    
    seg3 = principal_vector_segments(nodes, S, which=3, scale=0.1, min_stress_fraction=0.0)
    assert len(seg3["starts"]) == 2
    assert seg3["is_tension"][1] == False


# ===================================================================== A5: Izoplochy (Marching Tetrahedra)
def test_extract_isosurface():
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    tets = np.array([[0, 1, 2, 3]])
    values = np.array([0.0, 10.0, 0.0, 0.0])

    verts, faces = extract_isosurface(nodes, tets, values, isovalue=5.0)
    assert len(faces) == 1
    assert len(verts) == 3
    assert np.any(np.isclose(verts[:, 0], 0.5))


def test_extract_isosurface_empty():
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    tets = np.array([[0, 1, 2, 3]])
    values = np.array([1.0, 2.0, 3.0, 4.0])

    verts, faces = extract_isosurface(nodes, tets, values, isovalue=10.0)
    assert len(faces) == 0
    assert len(verts) == 0
