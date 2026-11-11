"""Stresses and nodal forces from the displacements in the same way as ccx (C3D10) – for solvers other than ccx (NGSolve + hypre / PARDISO).

ccx: stresses at the 4 Gauss points (gauss3d4) → linear extrapolation to the corners (IP j lies at corner j), midpoints of edges =
average of the corners, node = plain average over all elements of the node. Verified 13. 10. 2026 (scratch/pardiso_proto/
stress_variants.py): from the displacements at full precision agreement with the .frd of ccx 8·10⁻⁶ (max., plate + foot 197 k DOF);
the projection of NGSolve (`Set`) differed up to 28 % at the midpoints of edges. Nodal forces = Σ_g w·|J|·Bᵀσ (the same 4 points).
This is neither the stiffness matrix nor a solver – only a recomputation from the result (like printing.failure, joints.forces).
"""
from __future__ import annotations

import numpy as np

from .loads import GAUSS4, GAUSS4_A as _A, GAUSS4_B as _B, W_GAUSS4, _tet10_dN, inv3
from .mesh import Mesh

EXTRAP = np.full((4, 4), -_B / (_A - _B)) + np.eye(4) / (_A - _B)   # roh i ← hodnoty v IP j
MID_EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])  # nodes 4–9 of C3D10 (ccx order)
VOIGT_TO_CCX = [0, 1, 2, 3, 5, 4]        # xx yy zz xy xz yz → ccx xx yy zz xy yz zx


def element_stiffness(study, mesh: Mesh, elems: np.ndarray | None = None) -> np.ndarray:
    """C (E, 6, 6) per element (Voigt xx yy zz xy xz yz, engineering shear; 3D printing = C_el as for ccx).

    elems = only the selected elements (isotropic materials do not assemble C for the whole mesh).
    """
    from .solve_large import element_stiffness_all, iso_stiffness
    C = element_stiffness_all(study, mesh)
    if C is not None:
        return C if elems is None else C[elems]
    Cp = np.array([iso_stiffness(study.material_of(p).E, study.material_of(p).nu) for p in mesh.part_names])
    return Cp[mesh.tet_part if elems is None else mesh.tet_part[elems]]


def _chunks(n: int, size: int = 200_000):
    for lo in range(0, n, size):
        yield lo, min(n, lo + size)


def _element_pass(nodes: np.ndarray, conn: np.ndarray, Ue: np.ndarray, Ce: np.ndarray,
                  alpha_e: np.ndarray | None = None, dTe: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Core of the recomputation for a batch of C3D10 elements: (σ at 4 IP (E, 4, 6) in the order xx yy zz xy xz yz, fe (E, 10, 3) = ∫Bᵀσ).

    Ue = displacements of the nodes of the elements (E, 10, 3), Ce = C of the elements (E, 6, 6); thermal load: alpha_e (E,) and dTe = ΔT of the nodes
    of the elements (E, 10) → σ = C·(ε − α·ΔT·(1,1,1,0,0,0)), ΔT at the IP by the shape functions.
    """
    from .results import tet10_shape
    X = nodes[conn]
    n = len(conn)
    dN = [_tet10_dN(*p) for p in GAUSS4]
    Ng = [tet10_shape(np.asarray(p)) for p in GAUSS4] if dTe is not None else None
    ip_sig = np.empty((n, 4, 6))
    fe = np.zeros((n, 10, 3))
    for g in range(4):
        J = np.tensordot(dN[g], X, axes=(1, 1)).swapaxes(0, 1)            # (E, 3, 3)
        dNx = inv3(J) @ dN[g]                                             # (E, 3, 10)
        G = dNx @ Ue                                                      # ∂u_k/∂x_a
        eps = np.stack([G[:, 0, 0], G[:, 1, 1], G[:, 2, 2], G[:, 0, 1] + G[:, 1, 0],
                        G[:, 0, 2] + G[:, 2, 0], G[:, 1, 2] + G[:, 2, 1]], axis=1)
        if dTe is not None:
            eps[:, :3] -= (alpha_e * (dTe @ Ng[g]))[:, None]
        sig = np.einsum("eij,ej->ei", Ce, eps)                            # xx yy zz xy xz yz
        ip_sig[:, g] = sig
        st = np.empty((n, 3, 3))
        st[:, 0, 0], st[:, 1, 1], st[:, 2, 2] = sig[:, 0], sig[:, 1], sig[:, 2]
        st[:, 0, 1] = st[:, 1, 0] = sig[:, 3]
        st[:, 0, 2] = st[:, 2, 0] = sig[:, 4]
        st[:, 1, 2] = st[:, 2, 1] = sig[:, 5]
        detJ = np.abs(np.linalg.det(J))
        fe += (W_GAUSS4 * detJ)[:, None, None] * np.einsum("eja,ejk->eak", dNx, st)
    return ip_sig, fe


def element_nodal_forces(study, mesh: Mesh, U: np.ndarray, elems: np.ndarray,
                         dT: np.ndarray | None = None) -> np.ndarray:
    """Internal nodal forces fe (k, 10, 3) of the selected C3D10 elements: ∫Bᵀσ dV over the 4 Gauss points (as RF in `recover`).

    dT = T − T_ref per node (thermal load) → σ = C(ε − αΔT). The matrix C is assembled only for the selected elements.
    """
    if mesh.tets.shape[1] != 10:
        raise ValueError("Element nodal forces are only available for tet10 (C3D10)")
    elems = np.asarray(elems, int)
    if len(elems) == 0:
        return np.zeros((0, 10, 3))
    conn = mesh.tets[elems]
    alpha_e = dTe = None
    if dT is not None:
        alpha_e = np.array([study.material_of(p).alpha or 0.0 for p in mesh.part_names])[mesh.tet_part[elems]]
        dTe = np.asarray(dT, float)[conn]
    return _element_pass(mesh.nodes, conn, U[conn], element_stiffness(study, mesh, elems), alpha_e, dTe)[1]


def recover(study, mesh: Mesh, U: np.ndarray, dT: np.ndarray | None = None,
            C: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(S (N, 6) in ccx order, RF (N, 3) = internal nodal forces ∫Bᵀσ) from the displacements U (N, 3).

    dT = T − T_ref per node (thermal load): σ = C·(ε − α·ΔT·(1,1,1,0,0,0)), ΔT at the IP by the shape functions.
    """
    if mesh.tets.shape[1] != 10:
        raise ValueError("Stress recovery is available only for tet10 (C3D10)")
    if C is None:
        C = element_stiffness(study, mesh)
    alpha = None
    if dT is not None:
        alpha = np.array([study.material_of(p).alpha or 0.0 for p in mesh.part_names])[mesh.tet_part]
    n = len(mesh.nodes)
    S_acc = np.zeros((n, 6))
    RF = np.zeros((n, 3))
    for lo, hi in _chunks(len(mesh.tets)):
        conn = mesh.tets[lo:hi]
        corner_sig, fe = _element_pass(mesh.nodes, conn, U[conn], C[lo:hi],
                                       None if dT is None else alpha[lo:hi],
                                       None if dT is None else dT[conn])
        corner = np.einsum("ij,ejk->eik", EXTRAP, corner_sig)             # (E, 4, 6)
        mids = 0.5 * (corner[:, MID_EDGES[:, 0]] + corner[:, MID_EDGES[:, 1]])
        allv = np.concatenate([corner, mids], axis=1)                     # (E, 10, 6)
        flat = conn.ravel()
        for k in range(6):
            S_acc[:, k] += np.bincount(flat, weights=allv[:, :, k].ravel(), minlength=n)
        for k in range(3):
            RF[:, k] += np.bincount(flat, weights=fe[:, :, k].ravel(), minlength=n)
    cnt = np.bincount(mesh.tets.ravel(), minlength=n).astype(float)
    S = np.divide(S_acc, cnt[:, None], out=np.zeros_like(S_acc), where=cnt[:, None] > 0)
    return S[:, VOIGT_TO_CCX], RF
