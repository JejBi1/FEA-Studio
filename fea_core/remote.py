"""Remote constraint (RemoteSupport, like Remote Displacement in Ansys): a face connected to a remote point by a
rigid or distributing constraint, with zero displacements / rotations of the selected components at the point.

ccx 2.23 (verified 13. 10. 2026, scratch/remote/exp1.py, exp2.py):
- RIGID: *RIGID BODY (REF NODE = displacements of the point, ROT NODE DOF 1–3 = rotations about the point) + *BOUNDARY on the REF/ROT
  node in every step. Reaction = RF REF NODE (force) and RF ROT NODE (moment about the point) from *NODE PRINT RF in the .dat –
  the .frd does NOT write nodes outside the elements. For ROT NODE in the .dat = rotation [rad].
- DISTRIBUTING: *COUPLING + *DISTRIBUTING with *BOUNDARY on the ref. node holds NOTHING (ccx only distributes the load of the ref.
  node, the displacement of the ref. node is written as 0, the problem stays singular). Therefore an own *EQUATION from the kinematics of RBE3:
  displacement and rotation of the point = weighted least squares of the displacements of the face nodes (weights w_i = ∫N_i dA as the load in
  `loads.distributing_forces`):
      u_c = Σ w_i u_i / W,  θ = Γ⁻¹ Σ w_i r_i × u_i,  Γ = Σ w_i (|r_i|² I − r_i r_iᵀ),  r_i = x_i − c,
      u_p = u_c + θ × (p − c).
  The row of the constraint (u_p·e_d = 0, θ·e_d = 0) = the nodal forces of a unit load through the distributing constraint (principle of
  virtual work) → `rbe3`. The equations are dense over the nodes of the face: ccx fills the block by eliminating the dependent DOF with
  terms² (the ccx manual for *DISTRIBUTING COUPLING recommends ≤ ~50 nodes). Measured (scratch/remote/exp4.py,
  plate 100×100, 6 equations): Σ terms² 2.5·10⁵ → ccx 1.8 s; 1.4·10⁷ → 10 s / 0.47 GB; 2.6·10⁷ → 18 s / 0.78 GB;
  an equation with 11 257 terms → ccx > 11 GB (killed). Therefore the limit `MAX_DIST_FILL` → a clear error (a rigid
  constraint = only 6 independent DOF per node, cheap).
  Reaction = Σ (RF − F) over the nodes of the equations (RF = K·u includes the force of the constraint equation), the moment about the point measured
  in the same way; in the free components it comes out ≈ 0 (RBE3 reproduces rigid motions → the constraint forces do no work on the
  free components) – a check that the constraint holds only the constrained components.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh

DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")
# Σ (terms of the equation)² of the distributing constraints above which ccx would need > ~1.2 GB and tens of seconds more
MAX_DIST_FILL = 4.0e7
# Rotation of a rigid body (*RIGID BODY) is taken by ccx as FINITE even without NLGEOM → in a nonlinear calculation with contact without
# large deformations the imbalance of forces and moments ∝ θ² results (trial A7 17. 10. 2026, scratch/a5/a7_exp.py, blocks
# 20×20×20 with a rigid face: θ = 0.0068 / 0.020 / 0.048 rad → 7.7e-6 / 9.0e-5 / 5.7e-4 ≈ 0.25·θ²; another task at
# 0.05 rad gave 4e-3 → the coefficient depends on the task; the linear calculation does not have this). Warning from 0.03 rad.
RIGID_ROT_WARN = 0.03
RIGID_ROT_ERR_COEF = 0.25


def rigid_rotations(study, ref_nodes: dict, refs: dict | None, rsup: dict | None) -> list[tuple[str, float]]:
    """Rotation [rad] of the rigid remote loads and constraints of one step: [(selection, |θ|)].
    ref_nodes = `InpMap.ref_nodes` (key "state:load" → {"ref", "rot"}), refs = displacements of the nodes NREF (*NODE PRINT),
    rsup = `{selection: {"rot": …}}` of the rigid constraints of the step.
    """
    from .study import RemoteLoad
    loads = [l for lc in study.load_cases for l in lc.loads if isinstance(l, RemoteLoad)]
    out: dict[str, float] = {}
    for key, info in (ref_nodes or {}).items():
        if "rot" not in info or not refs or info["rot"] not in refs:
            continue
        li = int(key.split(":")[1])
        name = loads[li].selection if li < len(loads) else key
        out[name] = max(out.get(name, 0.0), float(np.linalg.norm(refs[info["rot"]])))
    for sel, d in (rsup or {}).items():
        out[sel] = max(out.get(sel, 0.0), float(np.linalg.norm(np.asarray(d.get("rot", (0, 0, 0)), float))))
    return sorted(out.items())


def rigid_rotation_note(items: list[tuple[str, float]], step: str, nlgeom: bool, limit: float = RIGID_ROT_WARN) -> str | None:
    """Warning (None = all right) when a rigid body in a nonlinear calculation WITHOUT NLGEOM rotated by more than `limit` rad."""
    if nlgeom:
        return None
    bad = [(n, t) for n, t in items if t >= limit]
    if not bad:
        return None
    n, t = max(bad, key=lambda x: x[1])
    return (f"Step '{step}': the rigid remote constraint / load '{n}' rotated by {t:.3g} rad ({np.degrees(t):.2f}°), but the analysis is without large deformations (NLGEOM). CalculiX treats the rotation of a rigid body (*RIGID BODY) as finite and the rest of the analysis as small → the force and moment equilibrium is violated (grows with θ²: measured {RIGID_ROT_ERR_COEF * t * t:.1e}, several times more depending on the model, the analysis is rejected above {0.001:g}). Reduce the rotation (stiffer support), switch on large deformations (NLGEOM), or use the distributing coupling (its rotation is linear)")


class RemoteError(ValueError):
    pass


def parse_dofs(dofs) -> list[bool]:
    """Constrained components: a list of 6 bool | names ("ux,uy,uz" / ["ux", "rz"]) | "all" / "vše" / "posuvy" / "natočení" (Czech keywords: all / displacements / rotations)."""
    if isinstance(dofs, str):
        s = dofs.strip().lower()
        if s in ("all", "vse", "vše", "*", ""):
            return [True] * 6
        if s in ("translations", "posuvy", "u"):
            return [True] * 3 + [False] * 3
        if s in ("rotations", "natoceni", "natočení", "rotation", "r"):
            return [False] * 3 + [True] * 3
        dofs = [t for t in s.replace(";", ",").replace(" ", ",").split(",") if t]
    dofs = list(dofs)
    if len(dofs) == 6 and all(isinstance(v, (bool, np.bool_, int)) and not isinstance(v, str) for v in dofs) \
            and all(v in (0, 1, True, False) for v in dofs):
        return [bool(v) for v in dofs]
    out = [False] * 6
    for t in dofs:
        t = str(t).strip().lower()
        if t not in DOF_NAMES:
            raise RemoteError(f"Unknown remote support component '{t}' (allowed: {', '.join(DOF_NAMES)})")
        out[DOF_NAMES.index(t)] = True
    return out


def dofs_text(dofs) -> str:
    """"ux, uy, uz, rx" (constrained components) or "none" (Czech "žádná")."""
    on = [n for n, v in zip(DOF_NAMES, dofs) if v]
    return ", ".join(on) if on else "none"


def coupling_text(coupling: str) -> str:
    return "rigid" if coupling == "rigid" else "distributing"


def remote_supports(study) -> list:
    from .study import RemoteSupport
    return [c for c in study.constraints if isinstance(c, RemoteSupport)]


# ------------------------------------------------------------------ kinematics of RBE3 (distributing constraint)

def rbe3(mesh: Mesh, selection: str, point) -> dict:
    """Operator of the distributing constraint: displacement and rotation of the point `point` from the displacements of the face nodes.

    Returns {"idx" (n,) nodes with weight, "A" (6, n, 3): rows ux uy uz rx ry rz of the point = Σ A[k, i]·u_i,
    "c" weighted centroid, "W" sum of the weights}.
    """
    from .ccx.inp import consistent_nodal_forces
    tri = mesh.surfaces.get(selection)
    if tri is None or len(tri) == 0:
        raise RemoteError(f"Selection '{selection}' has no faces in the mesh")
    w = consistent_nodal_forces(mesh.nodes, tri, np.array([1.0, 0.0, 0.0]))
    idx = np.array(sorted(n for n, f in w.items() if f[0] > 0), dtype=np.int64)
    if len(idx) < 3:
        raise RemoteError(f"Selection '{selection}': too few nodes for a remote support")
    wi = np.array([w[int(n)][0] for n in idx])
    X = mesh.nodes[idx]
    W = float(wi.sum())
    c = (wi[:, None] * X).sum(axis=0) / W
    r = X - c
    G = (wi[:, None, None] * ((r * r).sum(axis=1)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :])).sum(axis=0)
    ev = np.linalg.eigvalsh(G)
    if ev[0] <= 1e-9 * max(ev[-1], 1e-300):
        raise RemoteError(f"Selection '{selection}': the face nodes lie on a line – the rotation of the point cannot be determined (select a face, not an edge)")
    Gi = np.linalg.inv(G)
    p = np.asarray(point, float)
    A = np.zeros((6, len(idx), 3))
    for d in range(3):
        e = np.zeros(3)
        e[d] = 1.0
        A[d] = wi[:, None] * (e / W) + wi[:, None] * np.cross(Gi @ np.cross(p - c, e), r)
        A[3 + d] = wi[:, None] * np.cross(Gi @ e, r)
    return {"idx": idx, "A": A, "c": c, "W": W}


def point_motion(mesh: Mesh, selection: str, point, U: np.ndarray) -> np.ndarray:
    """(6,) displacement [mm] and rotation [rad] of the point of a distributing constraint from the displacements of the mesh (RBE3 kinematics)."""
    op = rbe3(mesh, selection, point)
    return np.einsum("kij,ij->k", op["A"], U[op["idx"]])


def distributing_equations(study, mesh: Mesh, bc: np.ndarray) -> list[list[tuple[int, int, float]]]:
    """Constraint equations of the distributing remote constraints: [[(node 0-based, dof 0..2, coef.), ...]], the first term
    dependent. The rows of all constraints together by Gauss-Jordan elimination (pivot = the largest term outside the DOF held by
    *BOUNDARY `bc` (N, 3)) → every dependent DOF is only in its own equation. Equations satisfied by *BOUNDARY alone
    (all terms in `bc`) or linearly dependent are left out.
    """
    rows, cols = [], {}
    for c in remote_supports(study):
        if c.coupling == "rigid" or not any(c.dofs):
            continue
        op = rbe3(mesh, c.selection, c.point)
        idx, A = op["idx"], op["A"]
        for k in range(6):
            if c.dofs[k]:
                rows.append({(int(n), d): float(A[k, i, d]) for i, n in enumerate(idx) for d in range(3)
                             if A[k, i, d] != 0.0})
    if not rows:
        return []
    keys = sorted({key for r in rows for key in r})
    cols = {key: j for j, key in enumerate(keys)}
    C = np.zeros((len(rows), len(keys)))
    for i, r in enumerate(rows):
        for key, v in r.items():
            C[i, cols[key]] = v
    allowed = np.array([not bc[n, d] for n, d in keys])
    scale = np.abs(C).max(axis=1)
    out_rows = []
    for i in range(len(C)):
        cand = np.where(allowed, np.abs(C[i]), 0.0)
        j = int(np.argmax(cand))
        if cand[j] <= 1e-10 * scale[i]:
            continue                                   # satisfied by *BOUNDARY or dependent on the previous ones
        C[i] /= C[i, j]
        for k in range(len(C)):
            if k != i and C[k, j] != 0.0:
                C[k] -= C[k, j] * C[i]
        allowed[j] = False
        out_rows.append((i, j))
    eqs = []
    for i, j in out_rows:
        row = C[i]
        tol = 1e-12 * np.abs(row).max()
        terms = [(keys[j][0], keys[j][1], 1.0)]
        terms += [(keys[k][0], keys[k][1], float(row[k])) for k in np.nonzero(np.abs(row) > tol)[0] if k != j]
        eqs.append(terms)
    return eqs


def check_overlaps(study, mesh: Mesh, bolts=()) -> None:
    """The nodes of a remote constraint must not share another constraint equation of ccx (remote constraint / load, cylindrical support,
    bolt) – ccx does not allow two dependent relations at one node; with *DISTRIBUTING it gives an imbalance.
    """
    from .study import RemoteLoad, CylindricalSupport
    sups = remote_supports(study)
    if not sups:
        return
    others: list[tuple[str, str]] = []
    for c in study.constraints:
        if isinstance(c, CylindricalSupport):
            others.append((c.selection, "cylindrical support"))
    for lc in study.load_cases:
        for l in lc.loads:
            if isinstance(l, RemoteLoad):
                others.append((l.selection, "remote load"))
    for b in bolts:
        for key, _ in b.couplings:
            others.append((key, f"bolt {b.name}"))
    seen: list[tuple[str, set]] = []
    for c in sups:
        if c.selection not in mesh.surfaces:
            continue
        nodes = set(mesh.surface_nodes(c.selection).tolist())
        for sel, what in others:
            if sel in mesh.surfaces and nodes & set(mesh.surface_nodes(sel).tolist()):
                raise RemoteError(f"Remote support '{c.selection}' shares nodes with face '{sel}' ({what}) – ccx does not allow two constraint equations on one node. Separate the faces (use a different selection).")
        for sel, prev in seen:
            if nodes & prev:
                raise RemoteError(f"Remote supports '{sel}' and '{c.selection}' share nodes – merge them into one support or separate the faces.")
        seen.append((c.selection, nodes))


def rigid_rows(c, center) -> np.ndarray:
    """Rows [e, (p − center) × e] of the constrained components of a remote constraint for the check of rigid motions (motion of a body
    u(x) = t + ω × (x − center)): displacement of the point p in the direction e = e·t + ω·((p − center) × e); rotation = [0, e]·[t, ω].
    """
    p = np.asarray(c.point, float) - np.asarray(center, float)
    out = []
    for k in range(6):
        if not c.dofs[k]:
            continue
        e = np.zeros(3)
        e[k % 3] = 1.0
        out.append(np.concatenate([e, np.cross(p, e)]) if k < 3 else np.concatenate([np.zeros(3), e]))
    return np.array(out).reshape(-1, 6)
