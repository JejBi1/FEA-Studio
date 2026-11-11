"""Printing tolerance (F7): a part printed δ smaller on the free surface – displacement of the nodes of the mesh (PrintSpec.tolerance [mm]).

The geometry really shrinks: the nodes of the free surface of the printed part are moved inwards so that each of their free faces recedes by δ (n_i·v = −δ for the clusters of normals of the free faces at the node, n_j·v = 0 for the faces that stand: constraints, loads, bolt connections / *TIE, contact with another part – least squares), the internal nodes with a linear damping
w = 1 − d/L (d = distance to the nearest node of the free surface, L = max(2.5·h, 4δ)), the nodes on the standing faces only
in their plane. The faces with a load and a constraint remain – the forces act on the real material.
Why not a "layer without material" (the fraction of voids in the element): tried 14. 10. 2026 (P7) – the stiffness came out correctly, but the loaded face above the almost empty surface elements gave an artificial concentration
of the strain (safety 2.6× lower).
The nominal coordinates are kept by Mesh.node_data["nominal"] (the displacement can be reverted / changed without a new meshing).
"""
from __future__ import annotations

import math

import numpy as np

COS_CLUSTER = math.cos(math.radians(20.0))


def parts_with_tolerance(study, mesh) -> dict[int, float]:
    from .region import print_parts
    out = {}
    for pi, pn in enumerate(mesh.part_names):
        if pn in print_parts(study, [pn]):
            d = float(getattr(study.printing.spec(pn), "tolerance", 0.0) or 0.0)
            if d > 0:
                out[pi] = d
    return out


def _fixed_tri_keys(study, mesh) -> set:
    names = {c.selection for c in study.constraints}
    names |= {getattr(l, "selection", None) for lc in study.load_cases for l in lc.loads}
    names |= {n for n in mesh.surfaces if n.startswith("__")}       # bolts, *TIE
    keys = set()
    for nm in names:
        if nm in mesh.surfaces and len(mesh.surfaces[nm]):
            keys |= {tuple(t) for t in np.sort(np.asarray(mesh.surfaces[nm])[:, :3], axis=1).tolist()}
    return keys


def _clusters(normals: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for n in normals:
        for i, c in enumerate(out):
            if c @ n / max(np.linalg.norm(c), 1e-300) > COS_CLUSTER:
                out[i] = c + n
                break
        else:
            out.append(n.copy())
    return [c / np.linalg.norm(c) for c in out]


def displacement(study, mesh, nominal: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Displacement of the nodes (N, 3) for the printing tolerance from the nominal coordinates; warnings."""
    tolp = parts_with_tolerance(study, mesh)
    U = np.zeros_like(nominal)
    if not tolp:
        return U, []
    warnings: list[str] = []
    saved = mesh.nodes
    mesh.nodes = nominal
    try:
        tri, elem = mesh.boundary_faces()
    finally:
        mesh.nodes = saved
    X = nominal
    tpart = mesh.tet_part[elem]
    fixed_keys = _fixed_tri_keys(study, mesh)
    key = [tuple(k) for k in np.sort(tri[:, :3], axis=1).tolist()]
    is_fixed = np.array([k in fixed_keys for k in key], bool)
    # nodes shared with another part (contact) do not move
    owners = np.zeros(len(X), np.int64)
    first = np.full(len(X), -1)
    for pi in range(len(mesh.part_names)):
        nd = np.unique(mesh.tets[mesh.tet_part == pi])
        owners[nd] += 1
        first[nd] = np.where(first[nd] < 0, pi, first[nd])
    shared = owners > 1
    nt = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])
    from scipy.spatial import cKDTree
    for pi, delta in tolp.items():
        mine = tpart == pi
        free = mine & ~is_fixed
        if not free.any():
            continue
        cols = tri.shape[1]
        inc_free: dict[int, list] = {}
        inc_fix: dict[int, list] = {}
        for j in np.nonzero(mine)[0]:
            tgt = inc_fix if is_fixed[j] else inc_free
            for nd in tri[j, :cols]:
                tgt.setdefault(int(nd), []).append(nt[j])
        surf_nodes, surf_v = [], []
        for nd, ns in inc_free.items():
            if shared[nd]:
                continue
            rows = [(c, -delta) for c in _clusters(ns)]
            rows += [(c, 0.0) for c in _clusters(inc_fix.get(nd, []))]
            A = np.array([r[0] for r in rows])
            b = np.array([r[1] for r in rows])
            v = np.linalg.lstsq(A, b, rcond=None)[0]
            if np.linalg.norm(v) > 3 * delta:              # a sharp wedge – limit the displacement
                v *= 3 * delta / np.linalg.norm(v)
            surf_nodes.append(nd)
            surf_v.append(v)
        if not surf_nodes:
            continue
        surf_nodes = np.array(surf_nodes)
        surf_v = np.array(surf_v)
        U[surf_nodes] = surf_v
        # internal nodes of the part (and nodes of the standing faces in their plane) with damping
        elems = np.nonzero(mesh.tet_part == pi)[0]
        h = float(np.mean(np.linalg.norm(X[mesh.tets[elems[:2000], 1]] - X[mesh.tets[elems[:2000], 0]], axis=1)))
        L = max(2.5 * h, 4 * delta)
        nodes = np.unique(mesh.tets[elems])
        rest = np.setdiff1d(nodes, surf_nodes)
        rest = rest[~shared[rest]]
        if len(rest):
            d, k = cKDTree(X[surf_nodes]).query(X[rest])
            w = np.clip(1.0 - d / L, 0.0, 1.0)
            v = surf_v[k] * w[:, None]
            for i, nd in enumerate(rest):
                if nd in inc_fix:                          # on a standing face: only in its plane
                    for c in _clusters(inc_fix[nd]):
                        v[i] -= (v[i] @ c) * c
            U[rest] = v
        # check: no element of the part may invert
        P = X + U
        c4 = mesh.tets[elems, :4]
        vol0 = np.einsum("ij,ij->i", np.cross(X[c4[:, 1]] - X[c4[:, 0]], X[c4[:, 2]] - X[c4[:, 0]]),
                         X[c4[:, 3]] - X[c4[:, 0]])
        vol1 = np.einsum("ij,ij->i", np.cross(P[c4[:, 1]] - P[c4[:, 0]], P[c4[:, 2]] - P[c4[:, 0]]),
                         P[c4[:, 3]] - P[c4[:, 0]])
        bad = (np.sign(vol1) != np.sign(vol0)) | (np.abs(vol1) < 0.05 * np.abs(vol0))
        if bad.any():
            U[nodes] = 0.0
            warnings.append(f"3D printing '{mesh.part_names[pi]}': tolerance {delta:.3g} mm cannot be applied – the surface offset would distort {int(bad.sum())} elements (thin walls relative to the mesh); refine the mesh or reduce the tolerance")
    return U, warnings


def apply(study, mesh) -> list[str]:
    """Sets mesh.nodes = nominal + displacement of the tolerance (a new array – the shallow copies of the mesh of the print variants are not shared)."""
    nominal = mesh.node_data.get("nominal")
    if nominal is None:
        if not parts_with_tolerance(study, mesh):
            return []
        nominal = mesh.nodes.copy()
        mesh.node_data["nominal"] = nominal
    U, w = displacement(study, mesh, nominal)
    mesh.nodes = nominal + U
    if not np.any(U):
        mesh.nodes = nominal.copy()
        mesh.node_data.pop("nominal", None)
    return w
