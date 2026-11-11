"""NGSolve process: stiffness matrix and nodal stresses on our tet10 mesh.

  python -m fea_core.ngs.worker assemble JOB      → JOB/hypre/* (CSR after the elimination of the constraints, order node×3 of our mesh)

The stresses and nodal forces from the displacements are computed by fea_core.recovery in the same way as in ccx (the earlier projection of NGSolve `Set`
differed from ccx at the midpoints of the edges by up to 28 %).

Thermal load (JOB/dT.npy = T − T_ref per node and state, α per part in materials.json):
σ_th = C·α·ΔT·(1,1,1,0,0,0), load f_th = ∫ σ_th : ε(v) dx (added to F); the stresses σ(u) − σ_th and the internal
forces are computed by fea_core.recovery.

Separate process: NGSolve carries its own OpenCASCADE (7.8), it must not meet OCP.
Mesh: points renumbered "corners first" (otherwise NGSolve takes the middle nodes between the corners as isolated vertices),
tet10 elements in netgen order, Curve(2) = isoparametric curvature from our nodes.
The H1 basis of order 2 is hierarchical: the edge coefficient c has the value c·B_MID at the midpoint of the edge, therefore the
conversion u_ngs = T·u_nodes and K_nodes = Tᵀ K T. Check: rigid displacements / rotations must give K·u ≈ 0.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PERM_CCX_TO_NG = [0, 1, 2, 3, 4, 6, 7, 5, 8, 9]   # netgen: midpoints (1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
CCX_EDGES = [(0, 1, 4), (1, 2, 5), (2, 0, 6), (0, 3, 7), (1, 3, 8), (2, 3, 9)]
B_MID = -0.125                                     # value of the H1 edge function of order 2 at the midpoint of the edge


def build(job: Path):
    import ngsolve as ngs
    from netgen.meshing import Mesh as NGMesh
    from fea_core.mesh import Mesh
    m = Mesh.load(job / "mesh.npz")
    if m.tets.shape[1] != 10:
        raise ValueError("The NGSolve backend requires tet10")
    N = len(m.nodes)
    corner = np.unique(m.tets[:, :4])
    mid = np.setdiff1d(np.arange(N), corner)
    order = np.concatenate([corner, mid])          # netgen point p → our node order[p]
    newid = np.empty(N, dtype=np.int64)
    newid[order] = np.arange(N)
    ngm = NGMesh(dim=3)
    ngm.AddPoints(np.ascontiguousarray(m.nodes[order]))
    tets_ng = newid[m.tets]                        # ccx order, netgen point numbers
    for pi in range(len(m.part_names)):
        sel = m.tet_part == pi
        ngm.AddElements(dim=3, index=pi + 1, base=0,
                        data=np.ascontiguousarray(tets_ng[sel][:, PERM_CCX_TO_NG].astype(np.int32)))
        ngm.SetMaterial(pi + 1, f"p{pi}")
    mesh = ngs.Mesh(ngm)
    mesh.Curve(2)
    nv = mesh.nv
    if nv != len(corner):
        raise RuntimeError(f"NGSolve: {nv} vertices, expected {len(corner)}")
    # NGSolve edge → midpoint node (by the pair of vertices)
    keys = []
    mids = []
    for a, b, c in CCX_EDGES:
        lo = np.minimum(tets_ng[:, a], tets_ng[:, b])
        hi = np.maximum(tets_ng[:, a], tets_ng[:, b])
        keys.append(lo * nv + hi)
        mids.append(tets_ng[:, c])
    keys = np.concatenate(keys)
    mids = np.concatenate(mids)
    uk, ui = np.unique(keys, return_index=True)
    ev = np.array([[v.nr for v in e.vertices] for e in mesh.edges], dtype=np.int64)
    ek = np.minimum(ev[:, 0], ev[:, 1]) * nv + np.maximum(ev[:, 0], ev[:, 1])
    pos = np.searchsorted(uk, ek)
    if np.any(uk[np.clip(pos, 0, len(uk) - 1)] != ek):
        raise RuntimeError("NGSolve: edge without a mid-side node")
    edge_mid = mids[ui[pos]]                       # netgen point of the midpoint of each edge
    return ngs, m, mesh, order, nv, ev, edge_mid


def transform(nv, ev, edge_mid, npts):
    """T: scalar degrees of freedom of NGSolve (vertices, edges) ← values at the netgen points."""
    ne = len(ev)
    rows = np.concatenate([np.arange(nv), np.repeat(nv + np.arange(ne), 3)])
    cols = np.concatenate([np.arange(nv), np.column_stack([edge_mid, ev[:, 0], ev[:, 1]]).ravel()])
    w = 1.0 / B_MID
    vals = np.concatenate([np.ones(nv), np.tile([w, -0.5 * w, -0.5 * w], ne)])
    return sp.csr_matrix((vals, (rows, cols)), shape=(nv + ne, npts))


def materials_cf(ngs, mesh, job):
    mats = json.loads((job / "materials.json").read_text(encoding="utf-8"))
    E = mesh.MaterialCF({f"p{i}": m["E"] for i, m in enumerate(mats)}, default=mats[0]["E"])
    nu = mesh.MaterialCF({f"p{i}": m["nu"] for i, m in enumerate(mats)}, default=mats[0]["nu"])
    return E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))


def node_to_ngs(T1, order, N) -> sp.csr_matrix:
    """Q: our vector node×3 (interleaved) → degrees of freedom VectorH1 of NGSolve (components by blocks)."""
    n1 = T1.shape[0]
    cols_of_point = 3 * order                      # netgen point p → our node order[p]
    Tc = T1.tocoo()
    nnz = len(Tc.data)
    rows = np.empty(3 * nnz, dtype=Tc.row.dtype)
    cols = np.empty(3 * nnz, dtype=np.int64)
    data = np.tile(Tc.data, 3)
    cols_base = cols_of_point[Tc.col]
    for comp in range(3):
        rows[comp * nnz : (comp + 1) * nnz] = Tc.row + comp * n1
        cols[comp * nnz : (comp + 1) * nnz] = cols_base + comp
    return sp.csr_matrix((data, (rows, cols)), shape=(3 * n1, 3 * N))


VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))      # order of C_el (xx yy zz xy xz yz, engineering shear)


def ortho_cf(ngs, mesh, job, m):
    """C per element (job/C_el.npy, (E, 6, 6) in global axes, our order of elements) → matrix CF from 21
    piecewise constant L2 fields of order 0. The elements of NGSolve are in the order of addition (per part); check of the
    assignment: ∫ (1/V_e) dx = number of elements.
    """
    C = np.load(job / "C_el.npy")
    ng_order = np.concatenate([np.nonzero(m.tet_part == pi)[0] for pi in range(len(m.part_names))])
    Cn = C[ng_order]
    L2 = ngs.L2(mesh, order=0)
    if L2.ndof != len(ng_order):
        raise RuntimeError(f"NGSolve: L2 has {L2.ndof} DOFs, {len(ng_order)} elements")
    chk = ngs.GridFunction(L2)
    vol = ngs.Integrate(ngs.CF(1.0), mesh, element_wise=True).NumPy()
    chk.vec.FV().NumPy()[:] = 1.0 / vol
    if abs(ngs.Integrate(chk, mesh) - len(ng_order)) > 1e-6 * len(ng_order):
        raise RuntimeError("NGSolve: element numbering does not match the order of addition")
    gf = {}
    for i in range(6):
        for j in range(i, 6):
            g = ngs.GridFunction(L2)
            g.vec.FV().NumPy()[:] = Cn[:, i, j]
            gf[i, j] = gf[j, i] = g
    return ngs.CF(tuple(gf[i, j] for i in range(6) for j in range(6)), dims=(6, 6))


def thermal_stress(ngs, mesh, job, m, gdT):
    """σ_th as 6 Voigt components (xx yy zz xy xz yz) for the scalar field ΔT (GridFunction)."""
    mats = json.loads((job / "materials.json").read_text(encoding="utf-8"))
    alpha = mesh.MaterialCF({f"p{i}": mt.get("alpha", 0.0) for i, mt in enumerate(mats)},
                            default=mats[0].get("alpha", 0.0))
    eth = alpha * gdT
    if (job / "C_el.npy").exists():
        Cm = ortho_cf(ngs, mesh, job, m)
        sv = Cm * ngs.CF((eth, eth, eth, 0, 0, 0))
        return [sv[i] for i in range(6)]
    mu, lam = materials_cf(ngs, mesh, job)
    s0 = (3 * lam + 2 * mu) * eth
    return [s0, s0, s0, 0, 0, 0]


def thermal_gf(ngs, mesh, T1, order, values):
    """Scalar H1 of order 2 from the nodal values of our mesh (the same interpolation as tet10)."""
    g = ngs.GridFunction(ngs.H1(mesh, order=2))
    g.vec.FV().NumPy()[:] = T1 @ values[order]
    return g


def bilinear(ngs, mesh, job, m=None):
    """Bilinear form of elasticity and the stress function σ(u) → 6 components (xx yy zz xy yz zx).
    Isotropic via μ, λ per part; with job/C_el.npy (3D printing) general anisotropy per element.
    """
    fes = ngs.VectorH1(mesh, order=2)
    u, v = fes.TnT()
    eps = lambda w: ngs.Sym(ngs.Grad(w))
    if (job / "C_el.npy").exists():
        Cm = ortho_cf(ngs, mesh, job, m)

        def ev(w):
            e = eps(w)
            return ngs.CF(tuple((1 if a == b else 2) * e[a, b] for a, b in VOIGT))
        a = ngs.BilinearForm(ngs.InnerProduct(Cm * ev(u), ev(v)) * ngs.dx)

        def sigma(gu):
            sv = Cm * ev(gu)
            return [sv[0], sv[1], sv[2], sv[3], sv[5], sv[4]]
        return fes, a, sigma
    mu, lam = materials_cf(ngs, mesh, job)
    a = ngs.BilinearForm(ngs.InnerProduct(2 * mu * eps(u) + lam * ngs.Trace(eps(u)) * ngs.Id(3), eps(v)) * ngs.dx)

    def sigma(gu):
        e = eps(gu)
        s = 2 * mu * e + lam * ngs.Trace(e) * ngs.Id(3)
        return [s[i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))]
    return fes, a, sigma


def assemble(job: Path) -> None:
    """K in the nodal basis → constraints in place → files of the system for hypre (job/hypre/*.npy)."""
    from fea_core.hypre import rotation_modes
    t0 = time.time()
    ngs, m, mesh, order, nv, ev, edge_mid = build(job)
    t_mesh = time.time() - t0
    fes, a, _ = bilinear(ngs, mesh, job, m)
    with ngs.TaskManager():
        a.Assemble()
    t_asm = time.time() - t0 - t_mesh
    vals, cols, ptr = a.mat.CSR()                  # views into the NGSolve memory, without a copy
    Kh = sp.csr_matrix((np.asarray(vals), np.asarray(cols), np.asarray(ptr).astype(np.int64)),
                       shape=(fes.ndof, fes.ndof), copy=False)
    N = len(m.nodes)
    n = 3 * N
    T1 = transform(nv, ev, edge_mid, N)
    Q = node_to_ngs(T1, order, N)
    QT = Q.T.tocsr()
    fixed = np.load(job / "fixed.npy")
    g = np.load(job / "g.npy")
    F = np.load(job / "F.npy")
    if (job / "dT.npy").exists():                  # thermal load: f_th = Qᵀ ∫ σ_th : ε(v) dx
        dT = np.load(job / "dT.npy")
        Fth = np.zeros((n, dT.shape[1]))
        v = fes.TestFunction()
        ev_v = ngs.Sym(ngs.Grad(v))
        for k in range(dT.shape[1]):
            if not np.any(dT[:, k]):
                continue
            st = thermal_stress(ngs, mesh, job, m, thermal_gf(ngs, mesh, T1, order, dT[:, k]))
            f = ngs.LinearForm(fes)
            f += (st[0] * ev_v[0, 0] + st[1] * ev_v[1, 1] + st[2] * ev_v[2, 2]
                  + 2 * (st[3] * ev_v[0, 1] + st[4] * ev_v[0, 2] + st[5] * ev_v[1, 2])) * ngs.dx
            with ngs.TaskManager():
                f.Assemble()
            Fth[:, k] = QT @ f.vec.FV().NumPy()
        F += Fth
        np.save(job / "Fth.npy", Fth)
    # rigid displacements and rotations for the check K·u ≈ 0
    X = m.nodes - m.nodes.mean(axis=0)
    modes = []
    for dd in range(3):
        u0 = np.zeros((N, 3)); u0[:, dd] = 1.0; modes.append(u0.ravel())
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        u0 = np.zeros((N, 3)); u0[:, i] = -X[:, j]; u0[:, j] = X[:, i]; modes.append(u0.ravel())
    modes = np.column_stack(modes)
    # 1st pass: K_nodes = Qᵀ K Q by blocks of rows → temporary files (in memory only K of NGSolve + the block)
    hj = job / "hypre"
    hj.mkdir(exist_ok=True)
    chunk = max(30000, int(3e7 // max(1, Kh.nnz // max(1, Kh.shape[0]))))   # ~3e7 non-zeros per block
    chunk -= chunk % 3
    parts, nnz_total, row_nnz = [], 0, np.zeros(n, dtype=np.int64)
    diag = np.zeros(n)
    rbm_res, kmax, rowmax = 0.0, 0.0, 0
    gnz = np.any(g)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        B = (QT[lo:hi] @ Kh @ Q).tocsr()
        B.sort_indices()
        rbm_res = max(rbm_res, float(np.abs(B @ modes).max()))
        kmax = max(kmax, float(np.abs(B.data).max()))
        rowmax = max(rowmax, int(np.diff(B.indptr).max()))
        diag[lo:hi] = B.diagonal(k=lo)
        if gnz:
            F[lo:hi] -= (B @ g)[:, None]
        row_nnz[lo:hi] = np.diff(B.indptr)
        np.save(hj / f"_c{lo}_i.npy", B.indices.astype(np.int32))
        np.save(hj / f"_c{lo}_d.npy", B.data)
        parts.append(lo)
        nnz_total += B.nnz
        del B
    del Kh, a, vals, cols, ptr, Q, QT
    rbm_err = rbm_res / (kmax * float(np.abs(X).max()) * rowmax)
    # 2nd pass: joining into files for hypre + constraints (symmetric elimination, diag = mean of the free ones)
    d = float(diag[~fixed].mean())
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=indptr[1:])
    ind = np.lib.format.open_memmap(hj / "indices.npy", mode="w+", dtype=np.int32, shape=(nnz_total,))
    dat = np.lib.format.open_memmap(hj / "data.npy", mode="w+", dtype=np.float64, shape=(nnz_total,))
    for lo in parts:
        hi = min(n, lo + chunk)
        ci = np.load(hj / f"_c{lo}_i.npy")
        cd = np.load(hj / f"_c{lo}_d.npy")
        rows = np.repeat(np.arange(lo, hi), row_nnz[lo:hi])
        cd[fixed[rows] | fixed[ci]] = 0.0
        on_diag = (ci == rows) & fixed[rows]
        cd[on_diag] = d
        s, e = indptr[lo], indptr[hi]
        ind[s:e] = ci
        dat[s:e] = cd
        (hj / f"_c{lo}_i.npy").unlink()
        (hj / f"_c{lo}_d.npy").unlink()
    ind.flush()
    dat.flush()
    del ind, dat
    F[fixed, :] = d * g[fixed, None]
    np.save(hj / "indptr.npy", indptr)
    np.save(hj / "rhs.npy", np.ascontiguousarray(F))
    np.save(hj / "rbm.npy", rotation_modes(m.nodes))
    (job / "assemble.json").write_text(json.dumps({
        "ndof": n, "nnz": int(nnz_total), "t_mesh": t_mesh, "t_assemble": t_asm,
        "t_total": time.time() - t0, "rigid_body_residual": rbm_err}), encoding="utf-8")


if __name__ == "__main__":
    {"assemble": assemble}[sys.argv[1]](Path(sys.argv[2]))
