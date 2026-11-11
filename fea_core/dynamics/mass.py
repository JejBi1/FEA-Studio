"""Mass in the dynamics: the consistent product M·v (own integration ∫ρNᵀN dV – evaluation, not a solver; decision 3c of the
specification F8), added mass (PointMass, SurfaceMass) – writing into ccx (*MASS) and the forces in the statics (weight, rotation).

Consistent mass of tet10: a 14-point rule (degree 5, positive weights) – the product of the quadratic shape functions
(degree 4) is integrated exactly on straight elements; ρ per element as in ccx (also the groups of 3D printing, printing.element_density).
"""
from __future__ import annotations

import numpy as np

from .model import PointMass, SurfaceMass
from ..ccx import deck_text

# 14-point rule on a tetrahedron (Walkington / Keast, degree 5), weights on the reference tetrahedron (Σ = 1/6)
_A1, _A2, _C = 0.0927352503108912, 0.3108859192633006, 0.4544962958743504
_W1, _W2, _W3 = 0.01224884051939366, 0.01878132095300264, 0.007091003462846911


def _rule14() -> tuple[np.ndarray, np.ndarray]:
    pts, w = [], []
    for a, wa in ((_A1, _W1), (_A2, _W2)):
        b = 1.0 - 3.0 * a
        for k in range(4):                         # barycentric (λ0..λ3), λk = b
            lam = [a] * 4
            lam[k] = b
            pts.append(lam[1:])
            w.append(wa)
    d = 0.5 - _C
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        lam = [d] * 4
        lam[i] = lam[j] = _C
        pts.append(lam[1:])
        w.append(_W3)
    return np.array(pts), np.array(w)


TET14_PTS, TET14_W = _rule14()


def _element_rho(study, mesh) -> np.ndarray:
    from ..printing import element_density
    return element_density(study, mesh)


def _qp_data(mesh):
    """(N (q, 10), w·|J| (M, q)) – cache of the mesh."""
    from ..loads import _tet10_dN
    from ..results import tet10_shape

    def compute():
        X = mesh.nodes[mesh.tets]
        out = np.empty((len(mesh.tets), len(TET14_W)))
        for q, (p, w) in enumerate(zip(TET14_PTS, TET14_W)):
            J = np.einsum("ij,mjk->mik", _tet10_dN(*p), X)
            out[:, q] = w * np.abs(np.linalg.det(J))
        return out
    N = np.stack([tet10_shape(p) for p in TET14_PTS])
    return N, mesh.cached("tet10_wdet14", ("nodes", "tets"), compute)


def mass_times(study, mesh, V: np.ndarray) -> np.ndarray:
    """M·V for a field of displacements V (N, 3) or (K, N, 3) – consistent mass of the bodies (only the nodes of the mesh; the added mass
    separately, point_mass_terms). The result of the same shape.
    """
    if mesh.tets.shape[1] != 10:
        raise ValueError("Consistent mass is available only for tet10 elements")
    V = np.asarray(V, float)
    single = V.ndim == 2
    VV = V[None] if single else V
    K, n = len(VV), VV.shape[1]
    N, wd = _qp_data(mesh)
    W = wd * _element_rho(study, mesh)[:, None]            # (M, q) weights of the points × ρ
    NN = np.einsum("qa,qb->qab", N, N).reshape(len(N), -1)
    tets = mesh.tets
    V2 = np.ascontiguousarray(VV.transpose(1, 0, 2).reshape(n, 3 * K))   # node × (mode, component)
    out = np.zeros((n, 3 * K))
    step = int(max(1000, 1e7 // (10 * 3 * K)))
    for e0 in range(0, len(tets), step):                  # by batches of elements (memory)
        t = tets[e0:e0 + step]
        Me = (W[e0:e0 + step] @ NN).reshape(len(t), 10, 10)   # mass matrix of the elements Σ_q w ρ N Nᵀ
        fe = Me @ V2[t]                                        # (m, 10, 3K)
        idx = t.ravel()
        fe = fe.reshape(-1, 3 * K)
        for c in range(3 * K):
            out[:, c] += np.bincount(idx, weights=fe[:, c], minlength=n)
    out = out.reshape(n, K, 3).transpose(1, 0, 2)
    return out[0].copy() if single else np.ascontiguousarray(out)


def total_mass(study, mesh) -> float:
    """Mass of the bodies of the mesh [t] (without the added mass)."""
    _, wd = _qp_data(mesh)
    return float((wd.sum(axis=1) * _element_rho(study, mesh)).sum())


# ------------------------------------------------------------------ added mass

def mass_items(study) -> list:
    return [m for m in getattr(study, "masses", []) if isinstance(m, (PointMass, SurfaceMass))]


def surface_mass_nodes(mesh, m: SurfaceMass) -> tuple[np.ndarray, np.ndarray]:
    """(nodes, mass of the nodes [t]) of the surface mass: consistent weights of the area ∫N dA (tri6: corners 0, midpoints A/3) – the sum
    exactly ρ_A·A; kg/m² → t/mm² = 1e-9.
    """
    from ..ccx.inp import consistent_nodal_forces
    w = consistent_nodal_forces(mesh.nodes, mesh.surfaces[m.selection], np.array([1.0, 0.0, 0.0]))
    idx = np.array([n for n, f in w.items() if f[0] > 0], dtype=np.int64)
    a = np.array([w[n][0] for n in idx])
    return idx, a * float(m.mass_per_area) * 1e-9


def point_of(mesh, m: PointMass) -> np.ndarray:
    """Point of the point mass: given, otherwise the centroid of the face (weights ∫N dA)."""
    if m.point is not None:
        return np.asarray(m.point, float)
    from ..ccx.inp import consistent_nodal_forces
    w = consistent_nodal_forces(mesh.nodes, mesh.surfaces[m.selection], np.array([1.0, 0.0, 0.0]))
    idx = np.array(list(w))
    a = np.array([w[n][0] for n in idx])
    return (a[:, None] * mesh.nodes[idx]).sum(axis=0) / a.sum()


def inertia_matrix(m: PointMass) -> np.ndarray | None:
    if m.inertia is None:
        return None
    I = np.asarray(m.inertia, float)
    if I.shape == (3,):
        I = np.diag(I)
    return I.reshape(3, 3)


def rotary_points(m: PointMass, p: np.ndarray, arm: float) -> list[tuple[np.ndarray, float]]:
    """Distribution of the mass with rotary inertia into points (rigid constraint): the centre + a pair ±a along the principal axes.

    A pair of masses μ_k at the distance ±a_k on the principal axis k gives I_ii = Σ_{k≠i} 2μ_k a_k² → c_k = 2μ_k a_k² =
    (tr I)/2 − I_kk (≥ 0 for a physical tensor). a = arm; it needs Σ 2μ_k ≤ m, otherwise the arm is enlarged.
    """
    I = inertia_matrix(m)
    if I is None or not np.any(I):
        return [(p, float(m.mass))]
    vals, vecs = np.linalg.eigh(0.5 * (I + I.T))
    c = 0.5 * vals.sum() - vals
    if (c < -1e-9 * max(vals.max(), 1e-30)).any():
        raise ValueError(f"Point mass '{m.name or m.selection}': the inertia tensor does not satisfy the triangle inequality (I_xx ≤ I_yy + I_zz …)")
    c = np.maximum(c, 0.0)
    a = float(arm)
    need = c.sum() / max(float(m.mass), 1e-300)          # Σ 2μ_k = Σ c_k / a² ≤ m → a² ≥ Σc/m
    if a * a < 1.25 * need:
        a = float(np.sqrt(1.25 * need))
    pts = []
    rest = float(m.mass)
    for k in range(3):
        mu = c[k] / (2 * a * a)
        if mu <= 0:
            continue
        for s in (1.0, -1.0):
            pts.append((p + s * a * vecs[:, k], mu))
        rest -= 2 * mu
    return [(p, rest)] + pts


ROT_ARM = 10.0                    # default arm of the auxiliary masses of the rotary inertia [mm]


def mass_points(study, mesh) -> list[dict]:
    """Points of the added point masses: [{"item": PointMass, "point": p, "nodes": [(x, μ)]}] (centre + auxiliary points
    of the rotary inertia at a rigid constraint; the distributing constraint only the centre).
    """
    out = []
    for m in mass_items(study):
        if isinstance(m, PointMass):
            p = point_of(mesh, m)
            pts = rotary_points(m, p, ROT_ARM) if m.coupling == "rigid" else [(p, float(m.mass))]
            out.append({"item": m, "point": p, "nodes": pts})
    return out


def accel_field(study, l):
    """Acceleration of the field of the body loads l at the points x (k, 3) [mm/s²] – the force on the mass m is m·a; None = the load
    does not act on the mass (force, pressure…).
    """
    from ..study import Gravity, Rotation
    kind = getattr(l, "kind", "")
    if isinstance(l, Gravity):
        from ..loads import global_vector
        g = global_vector(study, l)
        return lambda x: np.broadcast_to(g, np.shape(x)).astype(float)
    if isinstance(l, Rotation):
        a = np.asarray(l.axis, float)
        a = a / np.linalg.norm(a)
        p0 = np.asarray(l.point, float)
        w2, al = float(l.omega) ** 2, float(l.alpha)

        def f(x):
            r = np.asarray(x, float) - p0
            return w2 * (r - np.outer(r @ a, a)) - al * np.cross(a, r)
        return f
    if kind == "inertia":
        a0, al = np.asarray(l.accel, float), np.asarray(l.alpha, float)
        om = np.asarray(l.omega, float)
        W = np.asarray(l.wmat, float).reshape(3, 3) if getattr(l, "wmat", None) is not None else \
            np.outer(om, om) - (om @ om) * np.eye(3)
        p0 = np.asarray(l.point, float)

        def f(x):
            r = np.asarray(x, float) - p0
            return -(a0 + np.cross(al, r) + r @ W.T)
        return f
    return None


def _part_ok(study, mesh, m, l) -> bool:
    """Rotation / inertia only of the chosen parts: the mass belongs to the part of its face."""
    parts = getattr(l, "parts", None)
    if not parts:
        return True
    sel = study.selections.get(m.selection)
    return sel is not None and any(f.part in set(parts) for f in sel.faces)


def static_forces(study, mesh, lc) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]]:
    """Forces of the added mass in the state lc (weight, rotation, inertia): (nodal forces of the mesh (N, 3) from the surface mass,
    [(index of the point mass, index of the point, force (3,))]).
    """
    F = np.zeros((len(mesh.nodes), 3))
    pts = []
    items = mass_items(study)
    if not items:
        return F, pts
    mp = mass_points(study, mesh)
    for l in lc.loads:
        acc = accel_field(study, l)
        if acc is None:
            continue
        for m in items:
            if isinstance(m, SurfaceMass) and _part_ok(study, mesh, m, l):
                idx, mm = surface_mass_nodes(mesh, m)
                F[idx] += mm[:, None] * acc(mesh.nodes[idx])
        for i, info in enumerate(mp):
            if not _part_ok(study, mesh, info["item"], l):
                continue
            for k, (x, mu) in enumerate(info["nodes"]):
                pts.append((i, k, mu * acc(x[None, :])[0]))
    return F, pts


def static_resultants(study, mesh, lc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(nodal forces of the mesh, force and moment about the origin from the point masses) – for the check of the equilibrium (loads.nodal_loads)."""
    F, pts = static_forces(study, mesh, lc)
    Fr, Mr = np.zeros(3), np.zeros(3)
    if pts:
        mp = mass_points(study, mesh)
        for i, k, f in pts:
            x = mp[i]["nodes"][k][0]
            Fr += f
            Mr += np.cross(x, f)
    return F, Fr, Mr


def write(f, study, mesh, m, next_node: int, next_elem: int, fmt, bc=None) -> tuple[int, int]:
    """Writing of the added mass into the .inp: surface = *MASS in the nodes of the face; point = nodes with *MASS, rigid constraint = *RIGID
    BODY (NSET = nodes of the face + auxiliary points of the rotary inertia, REF = centre), distributing = *EQUATION of the displacement
    of the centre from the kinematics of RBE3 (the centre dependent). m.masses = [{"nodes": [node numbers], …}] for the forces in the statics.
    """
    items = mass_items(study)
    m.masses = []
    if not items:
        return next_node, next_elem
    f.write("** ---------------- pridana hmota (fea_core.dynamics.mass)\n")
    k_el = 0
    for si, it in enumerate([x for x in items if isinstance(x, SurfaceMass)], 1):
        idx, mm = surface_mass_nodes(mesh, it)
        # one value *MASS per ELSET → group the nodes by the mass (rounding to a series of 0.5 % as the springs)
        q = np.round(np.log(np.maximum(mm, 1e-300)) / np.log(1.005)).astype(np.int64)
        for g, key in enumerate(np.unique(q)):
            sel = idx[q == key]
            val = float(mm[q == key].mean())
            es = f"EMS{si}_{g + 1}"
            f.write(f"*ELEMENT, TYPE=MASS, ELSET={es}\n")
            for n_ in sel:
                f.write(f"{next_elem}, {int(n_) + 1}\n")
                next_elem += 1
            f.write(f"*MASS, ELSET={es}\n{fmt(val)}\n")
            k_el += 1
    for pi, info in enumerate(mass_points(study, mesh), 1):
        it = info["item"]
        ids = []
        f.write(f"** bodova hmota {pi}: {deck_text(it.name or it.selection)}\n*NODE\n")
        for x, mu in info["nodes"]:
            f.write(f"{next_node}, {fmt(x[0])}, {fmt(x[1])}, {fmt(x[2])}\n")
            ids.append(next_node)
            next_node += 1
        for j, (x, mu) in enumerate(info["nodes"]):
            if mu <= 0:
                continue
            es = f"EMP{pi}_{j + 1}"
            f.write(f"*ELEMENT, TYPE=MASS, ELSET={es}\n{next_elem}, {ids[j]}\n*MASS, ELSET={es}\n{fmt(mu)}\n")
            next_elem += 1
        surf_nodes = mesh.surface_nodes(it.selection)
        if bc is not None and bc[surf_nodes].any():
            raise ValueError(f"Point mass '{it.name or it.selection}': the face shares nodes with a fixed support / prescribed displacement – ccx does not allow a mass coupling node to be in *BOUNDARY at the same time. Choose another face.")
        # ccx (mastruct) does not establish the structure of the matrix according to the elements MASS: the mass in a DEPENDENT node of a constraint (auxiliary points
        # of the rotary inertia in *RIGID BODY, the centre of the distributing constraint in *EQUATION) → "*ERROR in add_bo_st / add_sm_st:
        # coefficient should be 0" (verified 15. 10. 2026, scratch/f8_examples/pm_test.py). An auxiliary spring SPRINGA of
        # negligible stiffness with this node establishes the structure (it unfolds through the constraint); the force ~1e-9·Δl is negligible.
        dummy = [(n_, ids[0]) for n_, (x, mu) in zip(ids[1:], info["nodes"][1:]) if mu > 0] \
            if it.coupling == "rigid" else [(ids[0], int(surf_nodes[0]) + 1)]
        if dummy:
            f.write(f"*ELEMENT, TYPE=SPRINGA, ELSET=EPMS{pi}\n")
            for na, nb in dummy:
                f.write(f"{next_elem}, {na}, {nb}\n")
                next_elem += 1
            f.write(f"*SPRING, ELSET=EPMS{pi}\n\n1.E-9\n")
        if it.coupling == "rigid":
            rot = next_node
            next_node += 1
            p = info["point"]
            f.write(f"*NODE\n{rot}, {fmt(p[0])}, {fmt(p[1])}, {fmt(p[2])}\n*NSET, NSET=NPM{pi}\n")
            from ..ccx.inp import _write_ids
            _write_ids(f, np.concatenate([surf_nodes + 1, np.asarray(ids[1:], dtype=np.int64)]))
            f.write(f"*RIGID BODY, NSET=NPM{pi}, REF NODE={ids[0]}, ROT NODE={rot}\n")
        else:
            from ..remote import rbe3
            op = rbe3(mesh, it.selection, info["point"])
            f.write("*EQUATION\n")
            for d in range(3):
                terms = [(ids[0], d, -1.0)] + [(int(n_) + 1, e, float(op["A"][d, i, e]))
                                               for i, n_ in enumerate(op["idx"]) for e in range(3)
                                               if op["A"][d, i, e] != 0.0]
                f.write(f"{len(terms)}\n")
                for s in range(0, len(terms), 4):
                    f.write(", ".join(f"{n_}, {e + 1}, {fmt(c)}" for n_, e, c in terms[s:s + 4]) + "\n")
        m.masses.append({"nodes": ids, "point": info["point"].tolist(), "name": it.name or it.selection})
    return next_node, next_elem


def node_motion(mesh, info: dict, U: np.ndarray) -> np.ndarray:
    """Displacements of the points of the point mass (k, 3) from the displacements of the mesh U (N, 3): motion of the centre and the rotation from the
    kinematics of the face (RBE3 – exact for a rigid motion of the face), points = u_p + θ × (x − p). The .frd does not write the nodes of the mass (verified
    by the trial scratch/f8_dyn/exp_norm.py).
    """
    from ..remote import point_motion
    p = info["point"]
    mot = point_motion(mesh, info["item"].selection, p, U)
    return np.array([mot[:3] + np.cross(mot[3:], x - p) for x, _ in info["nodes"]])


def check(study, part_names=None) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for m in mass_items(study):
        what = f"Added mass '{m.name or m.selection}'"
        if m.selection not in study.selections:
            errors.append(f"{what}: selection '{m.selection}' does not exist")
        elif study.selections[m.selection].entity != "faces":
            errors.append(f"{what}: the selection must contain faces")
        if isinstance(m, PointMass):
            if not float(m.mass) > 0:
                errors.append(f"{what}: the mass must be positive")
            if m.coupling not in ("rigid", "distributing"):
                errors.append(f"{what}: unknown coupling '{m.coupling}' (rigid | distributing)")
            if m.coupling == "distributing" and inertia_matrix(m) is not None:
                warnings.append(f"{what}: a distributing coupling does not transfer rotary inertia – neglected (use a rigid one)")
            try:
                rotary_points(m, np.zeros(3), 1.0)
            except ValueError as e:
                errors.append(str(e))
        elif not float(m.mass_per_area) > 0:
            errors.append(f"{what}: the mass per unit area must be positive")
    return errors, warnings
