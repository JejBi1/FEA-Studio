"""Solver for large models: distributing constraint, hypre against the direct solver, NGSolve+hypre against ccx."""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fea_core import hypre


def test_distributing_forces_resultant():
    """Distribution of a force and a moment to the nodes of a face gives exactly F and M (about the point of application)."""
    from fea_core.mesh import Mesh
    from fea_core.loads import distributing_forces
    rng = np.random.default_rng(2)
    # square 10×10 in the plane z = 0 from 2 tri6 (nodes at the corners + midpoints of the edges)
    P = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 0, 0], [10, 5, 0], [5, 10, 0], [0, 5, 0],
                  [5, 5, 0]], float)
    tri = np.array([[0, 1, 2, 4, 5, 8], [0, 2, 3, 8, 6, 7]])
    m = Mesh(P, np.zeros((0, 10), int), np.zeros(0, int), ["A"], {"S": tri})
    F, M, p = rng.normal(size=3) * 100, rng.normal(size=3) * 1e4, np.array([3.0, 4.0, 50.0])
    f = distributing_forces(m, "S", p, F, M)
    np.testing.assert_allclose(f.sum(axis=0), F, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(np.cross(P - p, f).sum(axis=0), M, rtol=1e-10, atol=1e-6)


def _laplace_3d(n):
    """An elasticity-like SPD system with blocks 3×3 (Laplace ⊗ I3 + a small coupling of the components)."""
    T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    I = sp.identity(n)
    L = sp.kron(sp.kron(T, I), I) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(I, I), T)
    B = np.array([[2.0, 0.3, 0.1], [0.3, 2.0, 0.2], [0.1, 0.2, 2.0]])
    return sp.kron(L, B).tocsr()


@pytest.mark.skipif(not hypre.available(), reason="hypre není nainstalován")
def test_hypre_matches_direct(tmp_path):
    K = _laplace_3d(14)
    n = K.shape[0]
    rng = np.random.default_rng(0)
    F = rng.normal(size=(n, 2))
    fixed = np.zeros(n, bool)
    fixed[:3 * 14 * 14] = True                      # one face of the cube clamped
    fixed[-3::3] = True                             # and one separate component (as a symmetry)
    g = np.zeros(n)
    g[-1] = 0.01                                    # a non-zero prescribed displacement
    Kb, Fb = hypre.apply_dirichlet(K, F, fixed, g)
    X, info = hypre.solve(Kb, Fb, np.zeros((n, 3)), tmp_path, nproc=4, tol=1e-12)
    ref = np.column_stack([spla.spsolve(Kb.tocsc(), Fb[:, k]) for k in range(2)])
    np.testing.assert_allclose(X, ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max())
    np.testing.assert_allclose(X[fixed, 0], g[fixed])
    assert info["nproc"] >= 2                       # a small case: fewer processes (≥ 3000 DOF per process)
    assert all(c["iterations"] < 200 for c in info["cases"])


@pytest.mark.slow
@pytest.mark.skipif(not hypre.available(), reason="hypre není nainstalován")
def test_ngs_hypre_equals_ccx_on_same_mesh(tmp_path):
    """The same mesh → the displacements of NGSolve+hypre and of ccx agree (tet10, curved elements, pressure)."""
    from verification import cases as C
    from fea_core.solve import run_study
    from fea_core.solve_large import run_ngs_hypre
    st, _ = C.v4_build(tmp_path)                  # tube: curved surfaces, pressure, symmetry
    res_c = run_study(st, tmp_path / "ccx")
    res_h = run_ngs_hypre(st, res_c.mesh, tmp_path / "ngs", [], lambda s: None)
    a, b = res_c.step(0), res_h.step(0)
    assert float(np.abs(a.U - b.U).max() / np.abs(a.U).max()) < 2e-3
    assert abs(float(a.mises.max()) - float(b.mises.max())) / float(a.mises.max()) < 0.02
    assert b.equilibrium["imbalance"] < 1e-6
