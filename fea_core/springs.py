"""Elastic support of a face (Winkler foundation, as Elastic Support in Ansys) – `study.ElasticSupport`.

Stiffness k [N/mm³] (per area) or total K [N/mm] distributed by the area (k = K/A). Nodal springs k·w_i, weights w_i
= diagonal (HRZ) distribution of the area of tri6: corner A/19, midpoint of an edge 16A/57 (Σ = A; a consistent distribution would give corners 0).
- direction "all": SPRING1 at the node in the directions x, y, z (the same stiffness),
- direction "normal": only perpendicular to the face – an auxiliary node with SPRING1 and the constraint equation u_g = n·u (see write).
ccx wants one constant per *SPRING → the nodal stiffnesses are rounded to a geometric series with a step of 0.5 % (≤ 0.25 %)
and the reactions are computed with the stiffnesses actually written (the equilibrium exactly).
Verified (scratch/springs/exp1.py, 14. 10. 2026): the RF of ccx at a node with a spring includes the force of the spring (= external load,
at a free face 0) → the reaction of the foundation = −k·u (resp. −k·(u·n)·n), computed by `reactions`. Tension and compression alike (linearly,
the foundation does not separate). The NGSolve backend cannot (solve_large.unsupported → ccx).
"""
from __future__ import annotations

import numpy as np

STEP = 1.005                 # rounding step of the stiffnesses (0.5 %)


def node_weights(mesh, sel: str) -> tuple[np.ndarray, np.ndarray]:
    """(nodes of the selection, share of the area per node [mm²]) – HRZ distribution of the area of the triangles tri6 (tri3: A/3 to the corners)."""
    tri = np.asarray(mesh.surfaces[sel])
    X = mesh.nodes
    a = 0.5 * np.linalg.norm(np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]]), axis=1)
    w = np.zeros(len(X))
    if tri.shape[1] >= 6:
        np.add.at(w, tri[:, :3].ravel(), np.repeat(a / 19.0, 3))
        np.add.at(w, tri[:, 3:6].ravel(), np.repeat(a * 16.0 / 57.0, 3))
    else:
        np.add.at(w, tri[:, :3].ravel(), np.repeat(a / 3.0, 3))
    idx = np.unique(tri[:, :6] if tri.shape[1] >= 6 else tri[:, :3])
    return idx, w[idx]


def node_normals(mesh, sel: str, idx: np.ndarray) -> np.ndarray:
    """Unit normals at the nodes of the selection (average of the normals of the neighbouring triangles weighted by area)."""
    tri = np.asarray(mesh.surfaces[sel])
    X = mesh.nodes
    nt = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])      # length = 2A
    acc = np.zeros((len(X), 3))
    cols = tri[:, :6] if tri.shape[1] >= 6 else tri[:, :3]
    for j in range(cols.shape[1]):
        np.add.at(acc, cols[:, j], nt)
    n = acc[idx]
    ln = np.linalg.norm(n, axis=1)
    return n / np.where(ln > 0, ln, 1.0)[:, None]


def supports(study):
    from .study import ElasticSupport
    return [c for c in study.constraints if isinstance(c, ElasticSupport)]


def _quantize(k: np.ndarray) -> np.ndarray:
    lk = np.log(np.maximum(k, 1e-300)) / np.log(STEP)
    return np.where(k > 0, STEP ** np.round(lk), 0.0)


def table(study, mesh) -> list[dict]:
    """Springs per selection: {"selection", "idx", "k" (written stiffness of the node [N/mm]), "direction", "normal"}."""
    out = []
    for c in supports(study):
        idx, w = node_weights(mesh, c.selection)
        area = float(w.sum())
        kk = float(c.stiffness) if c.per_area else float(c.stiffness) / max(area, 1e-30)
        k = _quantize(kk * w)
        d = {"selection": c.selection, "idx": idx, "k": k, "direction": c.direction, "area": area,
             "k_area": kk}
        if c.direction == "normal":
            d["normal"] = node_normals(mesh, c.selection, idx)
        out.append(d)
    return out


def write(f, study, mesh, next_node: int, next_elem: int, fmt) -> tuple[int, int, list[int]]:
    """Spring elements into the .inp; returns (next node, next element, auxiliary nodes of the normal springs – NSET NSPR).

    Normal spring = auxiliary node g (SPRING1 in DOF 1 to the ground) + *EQUATION u_g,1 = n·u (g dependent → it does not collide
    with other equations or with *BOUNDARY of the node). SPRINGA (node – fixed node in the direction of the normal) in ccx 2.23 holds also transversely
    (verified 14. 10. 2026, scratch/springs/exp_springa.py: a tensioned prism on normal SPRINGA has σx at the base 167 instead of
    210 MPa, normal force ≈ 0) → do not use.
    """
    tabs = table(study, mesh)
    if not tabs:
        return next_node, next_elem, []
    aux: list[int] = []
    eqs: list[str] = []
    groups: dict[tuple[str, float], list[str]] = {}
    X = mesh.nodes
    f.write("** pruzne ulozeni (ElasticSupport)\n")
    for t in tabs:
        for j, n in enumerate(t["idx"]):
            k = float(t["k"][j])
            if k <= 0:
                continue
            if t["direction"] == "normal":
                p = X[n]
                f.write(f"*NODE\n{next_node}, {fmt(p[0])}, {fmt(p[1])}, {fmt(p[2])}\n")
                groups.setdefault(("1", k), []).append(f"{next_elem}, {next_node}")
                nv = t["normal"][j]
                terms = [(next_node, 1, 1.0)] + [(n + 1, d + 1, -float(nv[d])) for d in range(3) if abs(nv[d]) > 1e-14]
                eqs.append(f"{len(terms)}\n" + ", ".join(f"{a}, {d}, {fmt(c)}" for a, d, c in terms))
                aux.append(next_node)
                next_node += 1
                next_elem += 1
            else:
                for dof in (1, 2, 3):
                    groups.setdefault((str(dof), k), []).append(f"{next_elem}, {n + 1}")
                    next_elem += 1

    def real(v: float) -> str:
        """ccx tells a DOF line from a stiffness by the decimal point (it deletes empty lines) → always a point."""
        t = fmt(v)
        if "." in t:
            return t
        return t.replace("e", ".e") if "e" in t else t + "."
    for gi, ((dof, k), rows) in enumerate(sorted(groups.items())):
        es = f"ESPR{gi + 1}"
        f.write(f"*ELEMENT, TYPE=SPRING1, ELSET={es}\n" + "\n".join(rows) + "\n")
        f.write(f"*SPRING, ELSET={es}\n{dof}\n{real(k)}\n")
    if eqs:
        f.write("*EQUATION\n" + "\n".join(eqs) + "\n")
    if aux:
        f.write("*NSET, NSET=NSPR\n")
        for q in range(0, len(aux), 16):
            f.write(", ".join(str(g) for g in aux[q:q + 16]) + ",\n")
    return next_node, next_elem, aux


def reactions(study, mesh, U: np.ndarray) -> dict[str, np.ndarray]:
    """Reactions of the elastic support per selection (N, 3): −k·u, resp. −k·(u·n)·n (force of the foundation on the body)."""
    out = {}
    for t in table(study, mesh):
        R = out.setdefault(t["selection"], np.zeros((len(mesh.nodes), 3)))
        idx, k = t["idx"], t["k"]
        u = U[idx]
        if t["direction"] == "normal":
            n = t["normal"]
            R[idx] += -(k * np.einsum("ij,ij->i", u, n))[:, None] * n
        else:
            R[idx] += -k[:, None] * u
    return out


def check(study) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for c in supports(study):
        if not (np.isfinite(c.stiffness) and c.stiffness > 0):
            errors.append(f"Elastic support '{c.selection}': stiffness must be positive")
        if c.direction not in ("all", "normal"):
            errors.append(f"Elastic support '{c.selection}': unknown direction '{c.direction}' (all | normal)")
        s = study.selections.get(c.selection)
        if s is not None and s.entity != "faces":
            errors.append(f"Elastic support '{c.selection}': selection must contain faces")
    if supports(study):
        warnings.append("The elastic support is linear (carries tension and compression – the foundation does not lift off)")
    return errors, warnings
