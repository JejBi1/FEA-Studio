"""Parts meshed separately and connected by a *TIE constraint (chapter 5.5 of the specification: "shared mesh, or *TIE").

`MeshSettings.separate` = parts that are not fragmented in Gmsh together with the others (own faces, independent mesh) – for
parts that Gmsh does not join with the assembly (tangential contact: "PLC Error: segment and facet intersect"), or for another
fineness of the mesh. After meshing `detect` finds the faces of the surface of a separate part that lie on the surface of another part
(distance of the centroid of the face from the surface ≤ tolerance) → faces `__tie{i}_s` (slave = the separate part) and
`__tie{i}_m` (master = the other parts) in Mesh.surfaces; ccx: *TIE, POSITION TOLERANCE (element faces of both surfaces).
Nodes of the slave surface must not be in *BOUNDARY or dependent in another equation (ccx: "dependent side of a MPC and a
SPC") → such faces are left out of the tie with a warning. The NGSolve backend and MYSTRAN cannot (→ ccx / error).
"""
from __future__ import annotations

import numpy as np

PREFIX = "__tie"


def tolerance(study, mesh) -> float:
    t = study.mesh.tie_tolerance
    if t:
        return float(t)
    return max(0.05 * float(study.mesh.size), 1e-3)


def _point_tri_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance of the points p (n, 3) from the triangles (a, b, c) (n, 3) – Ericson, Real-Time Collision Detection 5.1.5."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) > 1e-300, den, 1e-300)
    v, w = vb / den, vc / den
    q = a + ab * v[:, None] + ac * w[:, None]                   # inside
    # regions of the vertices and edges (a simplified approach is enough: limit the barycentric coordinates and project onto the edges)
    out = (v < 0) | (w < 0) | (v + w > 1)
    if out.any():
        best = np.full(len(p), np.inf)
        for s0, s1 in ((a, b), (b, c), (c, a)):
            e = s1 - s0
            t = np.clip(np.einsum("ij,ij->i", p - s0, e) / np.maximum(np.einsum("ij,ij->i", e, e), 1e-300), 0, 1)
            d = np.linalg.norm(p - (s0 + e * t[:, None]), axis=1)
            best = np.minimum(best, d)
        dist = np.linalg.norm(p - q, axis=1)
        return np.where(out, best, dist)
    return np.linalg.norm(p - q, axis=1)


def detect(study, mesh) -> list[str]:
    """Finds the contact faces of separately meshed parts → mesh.surfaces __tie{i}_s / _m. Returns warnings."""
    from .contacts.detect import contact_parts, contact_part_pairs
    cparts = contact_parts(study) if study.active_contacts() else set()
    cpairs = contact_part_pairs(study) if cparts else set()
    sep = [p for p in dict.fromkeys([*study.mesh.separate, *sorted(cparts)]) if p in mesh.part_names]
    if not sep:
        return []
    from scipy.spatial import cKDTree
    warnings = []
    tri, elem = mesh.boundary_faces()
    part = mesh.tet_part[elem]
    X = mesh.nodes
    cen = X[tri[:, :3]].mean(axis=1)
    tol = tolerance(study, mesh)
    from .solve import constraint_masks
    bc = np.zeros(len(X), bool)
    for m in constraint_masks(study, mesh).values():
        bc |= m.any(axis=1)
    k = 0
    for pn in sep:
        pi = mesh.part_names.index(pn)
        mine = np.nonzero(part == pi)[0]
        # a contact pair (F8.5) is not tied by *TIE – between the parts of the pair there is contact
        skip = [mesh.part_names.index(q) for q in mesh.part_names if frozenset((pn, q)) in cpairs]
        other = np.nonzero((part != pi) & ~np.isin(part, skip))[0]
        if not len(mine) or not len(other):
            continue

        def touching(src, dst):
            """Faces src whose centroid lies within tol of some face dst."""
            tree = cKDTree(cen[dst])
            span = np.linalg.norm(X[tri[dst, 0]] - X[tri[dst, 1]], axis=1).max() if len(dst) else 0.0
            cand = tree.query_ball_point(cen[src], r=span + tol)
            ok = np.zeros(len(src), bool)
            for i, cs in enumerate(cand):
                if not cs:
                    continue
                cs = dst[np.asarray(cs)]
                d = _point_tri_dist(np.repeat(cen[src[i]][None], len(cs), 0), X[tri[cs, 0]], X[tri[cs, 1]],
                                    X[tri[cs, 2]])
                ok[i] = d.min() <= tol
            return src[ok]
        slave = touching(mine, other)
        master = touching(other, mine)
        if not len(slave) or not len(master):
            if pn in cparts and pn not in study.mesh.separate:
                continue                          # a part only in contact – without *TIE it is fine
            warnings.append(f"Part '{pn}' is meshed separately but does not touch another part (tolerance {tol:.3g} mm) – it is not connected to anything")
            continue
        fixed = bc[tri[slave, :6]].any(axis=1)
        if fixed.any():
            warnings.append(f"*TIE of part '{pn}': {int(fixed.sum())} contact faces have nodes in a support – omitted from *TIE (ccx does not allow a *BOUNDARY node to also be dependent in *TIE)")
            slave = slave[~fixed]
        k += 1
        mesh.surfaces[f"{PREFIX}{k}_s"] = tri[slave]
        mesh.surfaces[f"{PREFIX}{k}_m"] = tri[master]
        area = float(np.linalg.norm(np.cross(X[tri[slave, 1]] - X[tri[slave, 0]],
                                             X[tri[slave, 2]] - X[tri[slave, 0]]), axis=1).sum() / 2)
        mesh.report.setdefault("ties", []).append({"part": pn, "slave": f"{PREFIX}{k}_s", "master": f"{PREFIX}{k}_m",
                                                   "area_mm2": area, "tolerance_mm": tol})
    return warnings


def pairs(mesh) -> list[tuple[str, str]]:
    return [(t["slave"], t["master"]) for t in mesh.report.get("ties", [])]


def write(f, mesh, m, sel_ids_surface) -> None:
    """*TIE for the pairs from detect (the faces already written as *SURFACE)."""
    tl = mesh.report.get("ties", [])
    for i, t in enumerate(tl, 1):
        f.write(f"*TIE, NAME=T{i}, POSITION TOLERANCE={t['tolerance_mm']:.6g}, ADJUST=NO\n"
                f"{sel_ids_surface(t['slave'])}, {sel_ids_surface(t['master'])}\n")


def tied_part_pairs(mesh) -> list[tuple[int, int]]:
    """Pairs of part indices connected by *TIE (for the check of free bodies)."""
    out = []
    for t in mesh.report.get("ties", []):
        s = np.unique(mesh.surfaces[t["slave"]])
        mst = np.unique(mesh.surfaces[t["master"]])
        owner = np.full(len(mesh.nodes), -1)
        for pi in range(len(mesh.part_names)):
            owner[np.unique(mesh.tets[mesh.tet_part == pi])] = pi
        ps = set(np.unique(owner[s]).tolist()) - {-1}
        pm = set(np.unique(owner[mst]).tolist()) - {-1}
        out += [(a, b) for a in ps for b in pm]
    return out
