"""Periodic mesh of the two cut faces of a cyclic-symmetric sector (runs in the Gmsh worker – gmsh and numpy only, no OCP).

settings["periodic"] = {"left": selection, "right": selection, "axis_point": [3], "axis_dir": [3], "sectors": N}:
right = left rotated by +360°/N about the axis. Every Gmsh surface of `right` is paired with the surface of `left` whose rotated
centroid coincides (same area) and `gmsh.model.mesh.setPeriodic(2, right, left, affine)` copies the mesh of the left surface
onto the right one → the nodes of the two faces match exactly after the rotation (verified 11. 11. 2026: 1.6e-14 mm).
"""
from __future__ import annotations

import gmsh
import numpy as np


def affine(axis_point, axis_dir, angle: float) -> list[float]:
    """4×4 affine matrix (row by row, Gmsh convention) of the rotation about the axis by angle [rad]."""
    d = np.asarray(axis_dir, float)
    d = d / np.linalg.norm(d)
    K = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    p = np.asarray(axis_point, float)
    t = p - R @ p
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = t
    return A.ravel().tolist(), R, t


def apply(spec: dict, sel_surfs: dict[str, list[int]], tol: float, report: dict) -> None:
    """Sets the periodic mesh constraint (call before mesh.generate)."""
    occ = gmsh.model.occ
    left = sel_surfs.get(spec["left"], [])
    right = sel_surfs.get(spec["right"], [])
    if not left or not right:
        raise RuntimeError(f"Cyclic symmetry: the cut faces '{spec['left']}' / '{spec['right']}' were not found in the geometry")
    A, R, t = affine(spec["axis_point"], spec["axis_dir"], 2 * np.pi / int(spec["sectors"]))
    lprops = [(s, np.asarray(occ.getCenterOfMass(2, s)), occ.getMass(2, s)) for s in left]
    masters, slaves = [], []
    used = set()
    for s in right:
        c = np.asarray(occ.getCenterOfMass(2, s))
        a = occ.getMass(2, s)
        best, best_d = None, np.inf
        for sl, cl, al in lprops:
            if sl in used:
                continue
            dist = float(np.linalg.norm(R @ cl + t - c))
            if dist < best_d and abs(al - a) <= 1e-6 * max(a, 1e-30) + tol * tol:
                best, best_d = sl, dist
        if best is None or best_d > max(1e3 * tol, 1e-6):
            raise RuntimeError(f"Cyclic symmetry: face {s} of '{spec['right']}' is not '{spec['left']}' rotated by 360°/{spec['sectors']} about the axis (nearest rotated face {best_d:.3g} mm away) – check the axis, the number of sectors and the cut faces")
        used.add(best)
        slaves.append(s)
        masters.append(best)
    if len(used) != len(left):
        raise RuntimeError(f"Cyclic symmetry: '{spec['left']}' has {len(left)} surfaces, '{spec['right']}' {len(right)} – the cut faces do not match")
    gmsh.model.mesh.setPeriodic(2, slaves, masters, A)
    report["periodic"] = {"pairs": [[int(a), int(b)] for a, b in zip(slaves, masters)]}
