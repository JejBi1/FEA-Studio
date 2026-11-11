"""Expansion of a cyclic-symmetric sector to the full 360° model (display, CY1 reference model).

Sector k = the datum sector rotated by R^k (R = rotation by +360°/N, right = R·left). Its left cut nodes coincide with the right
nodes of sector k − 1 (and the right nodes of the last sector with the left nodes of sector 0) → one node; axis nodes are shared
by all sectors. Vectors are rotated (R^k·v), stresses R^k·σ·R^kᵀ, scalars copied; natural modes with a nodal diameter ND:
u_k = R^k·(cos(kφ)·Re − sin(kφ)·Im), φ = 2π·ND/N (= ccx NGRAPH output, trial 11. 11. 2026). The sector results are not changed.
"""
from __future__ import annotations

import copy

import numpy as np

from . import CyclicSymmetry, _pairs, rotate_stress


def spec_from_info(info: dict) -> CyclicSymmetry:
    return CyclicSymmetry(sectors=int(info["sectors"]), left=info["left"], right=info["right"],
                          axis_point=list(info["axis_point"]), axis_dir=list(info["axis_dir"]))


def sector_map(mesh, info: dict) -> tuple[np.ndarray, int, dict]:
    """(gid (N, n) – node of the full model for node i of sector k, number of nodes of the full model, pairs)."""
    spec = spec_from_info(info)
    p = _pairs(spec, mesh)
    N, n = int(spec.sectors), len(mesh.nodes)
    gid = np.full((N, n), -1, dtype=np.int64)
    cnt = 0
    for k in range(N):
        m = gid[k]
        if len(p["axis"]):
            m[p["axis"]] = gid[0][p["axis"]] if k > 0 else -1
        if k > 0:
            m[p["left"]] = gid[k - 1][p["right"]]
        new = m < 0
        if k == N - 1:
            new[p["right"]] = False
        m[new] = cnt + np.arange(int(new.sum()))
        cnt += int(new.sum())
        if k == N - 1:
            m[p["right"]] = gid[0][p["left"]]
    return gid, cnt, p


def powers(R: np.ndarray, N: int) -> list[np.ndarray]:
    out = [np.eye(3)]
    for _ in range(1, N):
        out.append(R @ out[-1])
    return out


def _scatter(gid, n_full, blocks) -> np.ndarray:
    first = blocks[0]
    out = np.zeros((n_full,) + first.shape[1:], dtype=first.dtype)
    for k in range(len(blocks) - 1, -1, -1):          # sector 0 written last – merged nodes take its value
        out[gid[k]] = blocks[k]
    return out


def expand_field(values: np.ndarray, gid, n_full: int, Rk: list[np.ndarray], x0=None) -> np.ndarray:
    """Nodal field of the sector → full model: (n,) scalar, (n, 3) vector, (n, 6) stress (ccx order), other widths copied."""
    a = np.asarray(values)
    n = gid.shape[1]
    if a.ndim == 0 or len(a) < n:
        return a
    a = a[:n]
    if a.ndim == 2 and a.shape[1] == 3:
        blocks = [a @ R.T for R in Rk]
    elif a.ndim == 2 and a.shape[1] == 6:
        blocks = [rotate_stress(a, R).astype(a.dtype, copy=False) for R in Rk]
    else:
        blocks = [a] * len(Rk)
    return _scatter(gid, n_full, blocks)


def expand_mesh(mesh, info: dict):
    """Mesh of the full 360° model on the rotated sector mesh (+ gid, rotations, pairs)."""
    from ..mesh import Mesh
    gid, n_full, p = sector_map(mesh, info)
    N = gid.shape[0]
    Rk = powers(p["R"], N)
    x0 = p["x0"]
    nodes = _scatter(gid, n_full, [(mesh.nodes - x0) @ R.T + x0 for R in Rk])
    tets = np.vstack([gid[k][mesh.tets] for k in range(N)])
    tet_part = np.tile(mesh.tet_part, N)
    skip = {info["left"], info["right"]}
    surfaces = {name: np.vstack([gid[k][tri] for k in range(N)]) for name, tri in mesh.surfaces.items()
                if name not in skip}
    edges = {name: np.vstack([gid[k][seg] for k in range(N)]) for name, seg in mesh.edges.items()}
    points = {name: np.concatenate([gid[k][pt] for k in range(N)]) for name, pt in mesh.points.items()}
    cell = {k_: np.concatenate([v] * N) for k_, v in mesh.cell_data.items() if len(v) == len(mesh.tets)}
    rep = dict(mesh.report)
    rep.update(nodes=int(n_full), elements=int(len(tets)), cyclic_expanded=N)
    full = Mesh(nodes, tets, tet_part, list(mesh.part_names), surfaces, rep, None, cell, edges, points)
    return full, gid, Rk


def _mode_blocks(shape_re, shape_im, nd: int, N: int, Rk) -> list[np.ndarray]:
    phi = 2 * np.pi * nd / N
    re = np.asarray(shape_re, float)
    im = np.zeros_like(re) if shape_im is None else np.asarray(shape_im, float)
    return [((np.cos(k * phi) * re - np.sin(k * phi) * im) @ Rk[k].T).astype(np.float32) for k in range(N)]


def expand_results(res):
    """Results of the full 360° model for display (cached on the sector results object; None without cyclic symmetry)."""
    info = (res.info or {}).get("cyclic")
    if not info or not len(res.mesh.nodes):
        return None
    hit = getattr(res, "_cyclic_full", None)
    if hit is not None:
        return hit
    from ..results import Results, StepResult, ThermalResult
    from ..eigen import ModeSet
    full, gid, Rk = expand_mesh(res.mesh, info)
    n_full = len(full.nodes)
    N = gid.shape[0]
    steps = []
    for st in res.steps:
        s2 = StepResult(st.name, expand_field(st.U, gid, n_full, Rk), expand_field(st.S, gid, n_full, Rk),
                        dict(st.reactions), dict(st.ref_displacements), copy.deepcopy(st.equilibrium),
                        dict(st.reaction_moments), st.moment_point, st.joints, {}, st.printing, st.remote)
        s2.fields = {k: expand_field(v, gid, n_full, Rk) for k, v in st.fields.items()}
        steps.append(s2)
    thermal = []
    for t in res.thermal:
        frames = None if t.frames is None else np.stack([expand_field(fr, gid, n_full, Rk) for fr in t.frames])
        thermal.append(ThermalResult(t.name, expand_field(t.T, gid, n_full, Rk), expand_field(t.flux, gid, n_full, Rk),
                                     t.transient, list(t.times), frames, t.heat, t.balance))
    modes = []
    for m in res.modes:
        nds = getattr(m, "nodal_diameters", None)
        shapes = []
        for j in range(len(m.shapes)):
            nd = int(nds[j]) if nds is not None else 0
            im = m.shapes_imag[j] if nds is not None else None
            shapes.append(_scatter(gid, n_full, _mode_blocks(m.shapes[j], im, nd, N, Rk)))
        m2 = ModeSet(m.kind, m.name, list(m.values), np.stack(shapes) if shapes else np.zeros((0, n_full, 3), np.float32),
                     m.case, m.participation, m.eff_mass, m.total_mass)
        for attr in ("nodal_diameters", "multiplicity", "sectors"):
            if hasattr(m, attr):
                setattr(m2, attr, getattr(m, attr))
        modes.append(m2)
    info2 = dict(res.info)
    info2["cyclic_expanded"] = True
    out = Results(full, steps, list(res.warnings), info2, thermal, modes)
    res._cyclic_full = out
    return out
