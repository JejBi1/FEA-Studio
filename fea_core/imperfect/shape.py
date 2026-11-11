"""Imperfection shapes (nodal displacement fields w, (N, 3) mm) and the imperfect copy of the mesh.

The original mesh is never changed: `imperfect_mesh` returns a copy with nodes x + w (selections, surfaces and element data are
shared – they reference node numbers, not coordinates).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .model import ImperfectionError


def reference_length(mesh, settings) -> float:
    """L for the amplitude L/ratio: the given length or the largest dimension of the bounding box of the mesh."""
    if settings.length:
        return float(settings.length)
    ext = mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0)
    return float(ext.max())


def amplitude(mesh, settings) -> float:
    """e0 [mm] = amplitude (> 0) or L / amplitude_ratio."""
    if settings.amplitude and float(settings.amplitude) > 0:
        return float(settings.amplitude)
    r = float(settings.amplitude_ratio or 0.0)
    if r <= 0:
        raise ImperfectionError("Imperfection: give an amplitude e0 > 0 mm or a ratio L/xxx with xxx > 0")
    return reference_length(mesh, settings) / r


def mode_combination(shapes: np.ndarray, modes, weights) -> np.ndarray:
    """Σ c_k·φ_k of the (normalised) buckling modes (1-based numbers); scaled to max |w| = 1."""
    shapes = np.asarray(shapes, float)
    modes = [int(m) for m in (modes or [1])]
    weights = list(weights or [])
    weights = [float(weights[i]) if i < len(weights) else 1.0 for i in range(len(modes))]
    w = np.zeros(shapes.shape[1:])
    for m, c in zip(modes, weights):
        if not 1 <= m <= len(shapes):
            raise ImperfectionError(f"Imperfection: buckling mode {m} is not available (computed modes 1–{len(shapes)})")
        w += c * shapes[m - 1]
    return _unit(w)


def bow(nodes: np.ndarray, a, b, direction) -> np.ndarray:
    """Half-sine bow w = sin(π·s) · d between points a (s = 0) and b (s = 1), s = projection on the axis a→b; max |w| = 1.
    Nodes outside the span (s < 0 or s > 1) are not moved.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.asarray(direction, float)
    ax = b - a
    L = float(np.linalg.norm(ax))
    if L <= 0:
        raise ImperfectionError("Imperfection bow: points a and b must differ")
    if not np.linalg.norm(d) > 0:
        raise ImperfectionError("Imperfection bow: the direction must be a nonzero vector")
    e = ax / L
    d = d - (d @ e) * e                               # perpendicular to the axis
    if not np.linalg.norm(d) > 1e-12:
        raise ImperfectionError("Imperfection bow: the direction must not be parallel to the axis a–b")
    d /= np.linalg.norm(d)
    s = (nodes - a) @ e / L
    amp = np.where((s >= 0) & (s <= 1), np.sin(np.pi * np.clip(s, 0.0, 1.0)), 0.0)
    return amp[:, None] * d[None, :]


def from_file(nodes: np.ndarray, path) -> np.ndarray:
    """Nodal shape from a CSV with columns x, y, z, ux, uy, uz (header allowed; separator , ; or tab): each mesh node takes
    the value of the nearest point of the file (nearest neighbour); scaled to max |w| = 1."""
    from ..pathcheck import is_unc
    p = Path(str(path))
    if is_unc(str(p)):
        raise ImperfectionError("Imperfection file: network (UNC) paths are not allowed")
    if not p.is_file():
        raise ImperfectionError(f"Imperfection file not found: {p}")
    rows = []
    for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        sep = ";" if ";" in s else ("\t" if "\t" in s else ",")
        parts = [x.strip() for x in s.split(sep)]
        try:
            vals = [float(x) for x in parts[:6]]
        except ValueError:
            continue                                  # header
        if len(vals) == 6:
            rows.append(vals)
    if len(rows) < 1:
        raise ImperfectionError("Imperfection file: no row with 6 numbers x, y, z, ux, uy, uz")
    arr = np.asarray(rows, float)
    from scipy.spatial import cKDTree
    _, idx = cKDTree(arr[:, :3]).query(nodes)
    return _unit(arr[idx, 3:6])


def _unit(w: np.ndarray) -> np.ndarray:
    m = float(np.linalg.norm(w, axis=1).max()) if len(w) else 0.0
    if not m > 0:
        raise ImperfectionError("Imperfection: the shape is zero everywhere")
    return w / m


def imperfection_field(mesh, settings, buckling_shapes: np.ndarray | None) -> np.ndarray:
    """w (N, 3) [mm] with max |w| = e0 (sign included), or zeros for imperfection "none"."""
    kind = settings.imperfection
    n = len(mesh.nodes)
    if kind == "none":
        return np.zeros((n, 3))
    if kind == "modes":
        if buckling_shapes is None:
            raise ImperfectionError("Imperfection from buckling modes: the linear buckling modes are missing")
        w = mode_combination(buckling_shapes, settings.modes, settings.weights)
    elif kind == "bow":
        if settings.bow_a is None or settings.bow_b is None or settings.bow_direction is None:
            raise ImperfectionError("Imperfection bow: give the points a, b and the direction")
        w = bow(mesh.nodes, settings.bow_a, settings.bow_b, settings.bow_direction)
    elif kind == "file":
        if not settings.shape_file:
            raise ImperfectionError("Imperfection from a file: no file given")
        w = from_file(mesh.nodes, settings.shape_file)
    else:
        raise ImperfectionError(f"Unknown imperfection kind '{kind}' (modes | bow | file | none)")
    sign = -1.0 if float(settings.sign or 1.0) < 0 else 1.0
    return sign * amplitude(mesh, settings) * w


def imperfect_mesh(mesh, w: np.ndarray):
    """Copy of the mesh with the nodes moved by w (the original mesh is not changed; own cache)."""
    w = np.asarray(w, float)
    if w.shape != mesh.nodes.shape:
        raise ImperfectionError("Imperfection: the shape does not match the mesh")
    return dataclasses.replace(mesh, nodes=np.asarray(mesh.nodes, float) + w, _cache={})
