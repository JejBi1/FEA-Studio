"""Results of the contact (F8.5, specification 4.3, 6.1): block CONTACT of the .frd (COPEN, CSLIP1, CSLIP2, CPRESS, CSHEAR1, CSHEAR2 –
one block per increment; S2S writes all nodes, mortar / N2S only the slave), forces and contact area by pairs,
the state of the contact (open / sliding / sticking – ccx does not print it, derived from CPRESS and |τ| vs μ·p), opening of the joint
from the deformed mesh (COPEN stores only the penetration), series *NODE PRINT by increments (force–displacement curve).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

STATUS = {0: "open", 1: "sliding", 2: "sticking"}


def contact_fields(block, n: int) -> dict[str, np.ndarray]:
    """Nodal fields from the block CONTACT: CPRESS, COPEN (penetration), CSLIP (|·|), CSHEAR (|·|), CSHEAR_V (2 components)."""
    out = {k: np.zeros(n) for k in ("CPRESS", "COPEN", "CSLIP", "CSHEAR")}
    if block is None or not len(block.node_ids):
        return out
    m = block.node_ids <= n
    ids = block.node_ids[m] - 1
    v = block.values[m]
    out["COPEN"][ids] = v[:, 0]
    out["CSLIP"][ids] = np.hypot(v[:, 1], v[:, 2])
    out["CPRESS"][ids] = v[:, 3]
    out["CSHEAR"][ids] = np.hypot(v[:, 4], v[:, 5])
    return out


def node_weights(mesh, tri) -> tuple[np.ndarray, np.ndarray]:
    """Nodes of the face and their area weights ∫N dA (tri6: corners 0, midpoints A/3) – pressure × weight = nodal force."""
    from ..ccx.inp import consistent_nodal_forces
    w = consistent_nodal_forces(mesh.nodes, tri, np.array([1.0, 0.0, 0.0]))
    nodes = np.array(sorted(w), dtype=np.int64)
    return nodes, np.array([w[k][0] for k in nodes])


ACTIVE_FRACTION = 0.05      # a node is "in contact" for the mean pressure from the force if its p ≥ 5 % of the mean nodal pressure
EDGE_PEAK_RATIO = 1.10      # a peak at the edge is conspicuous if it is more than 10 % above the largest pressure inside the face


def free_edge_nodes(tri: np.ndarray) -> np.ndarray:
    """Nodes on the boundary of a triangular face (an edge belongs to a single face: a corner, the midpoint of the edge and the opposite corners). tri6 =
    corners 0–2, midpoints 3 (0–1), 4 (1–2), 5 (2–0); tri3 only the corners.
    """
    tri = np.asarray(tri)
    pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.sort(pairs, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    free = cnt[inv.ravel()] == 1
    nodes = [pairs[free].ravel()]
    if tri.shape[1] >= 6:
        mid = np.concatenate([tri[:, 3], tri[:, 4], tri[:, 5]])
        nodes.append(mid[free])
    return np.unique(np.concatenate(nodes))


def pressure_metrics(study, mesh, tri: np.ndarray, nodes: np.ndarray, w: np.ndarray, p: np.ndarray,
                     fn: float, p_all: np.ndarray | None = None) -> dict:
    """More reliable measures of the pressure than the highest nodal pressure (peaks at a free edge are sensitive to the formulation: K4 ccx
    110 MPa × Abaqus 79 × Lamé 78.75; the nodal pressure of ccx beyond the boundary of the contact fluctuates – K6):

    p_force_MPa = Fn / A_active (Fn = the exact force of the pair from ccx CF, if we have it – `apply_cf` recomputes the value –,
        otherwise Σ p·w; A_active = area of the nodes with p ≥ 5 % of the mean nodal pressure, `area_active_mm2`),
    p_max_interior_MPa = the largest pressure outside the "boundary faces" (faces of the face with a node on the free edge; nodes held by a
        constraint – planes of symmetry – are not counted beyond the edge),
    p_face_max_MPa = the largest mean of the nodal pressure over a face (suppresses isolated nodal peaks; the scatter of the nodal pressure of the
        quadratic faces under a uniform pressure is still ±15 % – K4),
    peak_at_edge = the peak is at the edge and more than 10 % above the interior; peak_edge_ratio = p_max / p_max_interior.
    """
    on = p > 1e-12
    if not on.any():
        return {"p_force_MPa": 0.0, "area_active_mm2": 0.0, "p_max_interior_MPa": 0.0, "p_face_max_MPa": 0.0,
                "peak_at_edge": False, "peak_edge_ratio": 1.0}
    p_nodal = float((p * w)[on].sum() / max(w[on].sum(), 1e-30))
    act = p >= ACTIVE_FRACTION * p_nodal
    p_force = float(fn / max(w[act].sum(), 1e-30))
    try:
        from ..solve import constraint_masks
        held = np.zeros(len(mesh.nodes), bool)
        for msk in constraint_masks(study, mesh).values():
            held |= msk.any(axis=1)
    except Exception:  # noqa: BLE001 – without constraints (an auxiliary calculation) each boundary is free
        held = np.zeros(len(mesh.nodes), bool)
    edge = free_edge_nodes(tri)
    edge = edge[~held[edge]]
    in_edge = np.isin(nodes, edge)
    tri_edge = np.isin(tri, edge).any(axis=1)                    # elements touching the free edge
    layer = np.isin(nodes, np.unique(tri[tri_edge]))
    p_max = float(p.max())
    inner = p[~layer]
    p_int = float(inner.max()) if len(inner) else p_max
    if not len(inner):                                           # small face: only the edge itself is left out
        inner = p[~in_edge]
        p_int = float(inner.max()) if len(inner) else p_max
    at_edge = bool(layer[int(np.argmax(p))] and p_max > EDGE_PEAK_RATIO * max(p_int, 1e-30))
    if p_all is None:
        p_all = np.zeros(len(mesh.nodes))
        p_all[nodes] = p
    return {"p_force_MPa": p_force, "area_active_mm2": float(w[act].sum()), "p_max_interior_MPa": p_int,
            "p_face_max_MPa": float(p_all[tri].mean(axis=1).max()),
            "peak_at_edge": at_edge, "peak_edge_ratio": p_max / max(p_int, 1e-30)}


def pair_summary(study, mesh, report: list[dict], fields: dict, U: np.ndarray, params: list[dict]) -> dict:
    """{pair: {Fn, Ft, area_contact, area, p_max, open_max, slip_max, status_counts}} – per slave faces."""
    from .detect import signed_gap
    out = {}
    pp = {p["pair"]: p for p in params}
    Xd = mesh.nodes + U
    status = np.full(len(mesh.nodes), -1, np.int8)
    opening = np.zeros(len(mesh.nodes))
    for r in report:
        if not r.get("master"):
            continue
        tri = mesh.surfaces[r["slave"]]
        nodes, w = node_weights(mesh, tri)
        p = fields["CPRESS"][nodes]
        tau = fields["CSHEAR"][nodes]
        mu = (pp.get(r["pair"]) or {}).get("mu") or 0.0
        on = p > 1e-12
        slip = on & (tau >= (1 - 1e-3) * mu * p) if mu else on
        st = np.where(~on, 0, np.where(slip, 1, 2))
        status[nodes] = st
        # opening: gap of the deformed mesh (positive = open)
        mesh_d = _Deformed(mesh, Xd)
        gn, g = signed_gap(mesh_d, tri, mesh.surfaces[r["master"]])
        gmap = dict(zip(gn.tolist(), g.tolist()))
        op = np.array([max(gmap.get(int(k), 0.0), 0.0) for k in nodes])
        opening[nodes] = np.where(on, 0.0, op)
        a_on = float(w[on].sum())
        from .detect import node_normals
        nn_, nv = node_normals(mesh, tri)
        nmap = dict(zip(nn_.tolist(), range(len(nn_))))
        act = [nmap[int(k)] for k in nodes[on] if int(k) in nmap]
        spread = 0.0
        if act:
            v = nv[act]
            mn = v.mean(axis=0)
            mn /= max(np.linalg.norm(mn), 1e-300)
            spread = float(np.degrees(np.arccos(np.clip(v @ mn, -1, 1))).max())
        out[r["pair"]] = {"Fn_N": float((p * w).sum()), "Ft_N": float((tau * w).sum()), "area_mm2": float(w.sum()),
                          "Fn_nodal_N": float((p * w).sum()),
                          "area_contact_mm2": a_on, "contact_fraction": a_on / max(float(w.sum()), 1e-30),
                          "p_max_MPa": float(p.max()) if len(p) else 0.0,
                          "p_mean_MPa": float((p * w)[on].sum() / max(a_on, 1e-30)),
                          "open_max_mm": float(op[~on].max()) if (~on).any() else 0.0,
                          "slip_max_mm": float(fields["CSLIP"][nodes].max()) if len(nodes) else 0.0,
                          "nodes_open": int((st == 0).sum()), "nodes_slip": int((st == 1).sum()),
                          "nodes_stick": int((st == 2).sum()), "mu": mu, "normal_spread_deg": spread,
                          **pressure_metrics(study, mesh, tri, nodes, w, p, float((p * w).sum()), fields["CPRESS"])}
    fields["STATUS"] = status.astype(float)
    fields["OPENING"] = opening
    return out


class _Deformed:
    """A light replacement of the mesh with the deformed coordinates (for signed_gap)."""

    def __init__(self, mesh, X):
        self.nodes = X
        self.tets = mesh.tets


_HEAD = re.compile(rb"(displacements|forces) \(\w+,\w+,\w+\) for set (\S+) and time\s+(\S+)", re.I)


def print_series(dat: Path, setname: str) -> list[tuple[float, str, dict[int, np.ndarray]]]:
    """All the blocks *NODE PRINT of the set (by increments): [(time, "U"|"RF", {node: (3,)})]."""
    buf = Path(dat).read_bytes()
    out = []
    for m in _HEAD.finditer(buf):
        if m.group(2).decode().upper() != setname.upper():
            continue
        kind = "U" if m.group(1).lower().startswith(b"disp") else "RF"
        pos = buf.find(b"\n", m.end()) + 1
        cur = {}
        started = False
        while pos < len(buf):
            e = buf.find(b"\n", pos)
            e = len(buf) if e < 0 else e
            parts = buf[pos:e].split()
            pos = e + 1
            if not parts:
                if started:
                    break
                continue
            try:
                cur[int(parts[0])] = np.array([float(v) for v in parts[1:4]])
                started = True
            except ValueError:
                break
        out.append((float(m.group(3).replace(b"D", b"E")), kind, cur))
    return out


def ccx_messages(log_text: str) -> dict:
    """Diagnostics of non-convergence from the output of ccx: number of the cuts of the increment, the last time, messages."""
    cut = len(re.findall(r"increment size decreased|too many cutbacks|reduce the increment", log_text, re.I))
    times = re.findall(r"actual step time=\s*([-+0-9.Ee]+)", log_text)
    errs = [l.strip() for l in log_text.splitlines() if "*ERROR" in l]
    return {"cutbacks": cut, "last_step_time": float(times[-1]) if times else None, "errors": errs[-5:]}


ADVICE = ("Tips: use displacement control instead of force (unconstrained slip does not converge in statics), reduce the initial increment (dt0), switch friction on only in a later step (start with μ ≈ 1e-4), check initial gaps and penetrations, add weak springs against free motion of parts.")


_STAT = re.compile(rb"statistics for slave set (\S+), master set (\S+) and time\s+(\S+)", re.I)
_NUMS = re.compile(rb"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")


def contact_print(dat: Path) -> list[dict]:
    """Blocks *CONTACT PRINT CF (only S2S penalty): [{"slave", "master", "time", "force" (3,) on the slave face,
    "moment" (about the origin), "center", "normal" (mean normal)}].
    """
    buf = Path(dat).read_bytes()
    out = []
    for m in _STAT.finditer(buf):
        seg = buf[m.end(): m.end() + 1200]
        nums = [float(x.replace(b"D", b"E")) for x in _NUMS.findall(seg)[:12]]
        if len(nums) < 12:
            continue
        out.append({"slave": m.group(1).decode(), "master": m.group(2).decode(),
                    "time": float(m.group(3).replace(b"D", b"E")), "force": np.array(nums[0:3]),
                    "moment": np.array(nums[3:6]), "center": np.array(nums[6:9]), "normal": np.array(nums[9:12])})
    return out


def apply_cf(summary: dict, params: list[dict], cf: list[dict], k: int) -> None:
    """Overwrites Fn, Ft of the pair by the force from ccx *CONTACT PRINT CF at the end of the step k (exact – the integration of the nodal CPRESS
    over a line contact or a nonuniform pressure overestimates it).
    """
    by = {(c.get("slave"), c.get("master")): c["pair"] for c in params if c.get("slave")}
    t0, t1 = k if isinstance(k, tuple) else (k, k + 1)
    eps = 1e-9 * max(1.0, abs(t1))
    for c in cf:
        if not (t0 + eps < c["time"] <= t1 + eps):
            continue
        pair = by.get((c["slave"], c["master"]))
        if pair is None or pair not in summary:
            continue
        n = c["normal"] / max(np.linalg.norm(c["normal"]), 1e-300)
        fn = float(c["force"] @ n)
        # resultant (for a curved face the radial forces cancel – then Fn_N from the integral |p| dA applies, p_mean_MPa)
        flat = summary[pair].get("normal_spread_deg", 0.0) <= 10.0      # the active part of the contact is nearly planar
        if not flat:
            summary[pair].update(resultant_N=c["force"].tolist(), source="nodal CPRESS (curved surface)")
            continue
        summary[pair].update(Fn_N=abs(fn), Ft_N=float(np.linalg.norm(c["force"] - fn * n)),
                             force_N=c["force"].tolist(), center=c["center"].tolist(), source="ccx CF")
        if summary[pair].get("area_active_mm2"):               # mean pressure from the exact force
            summary[pair]["p_force_MPa"] = abs(fn) / summary[pair]["area_active_mm2"]


def edge_peak_warnings(steps) -> list[str]:
    """Warning of a nodal pressure peak at the free edge of the contact face: one text per pair, with the list of the steps.
    A peak at an edge (the edge of a hole, the end of a face) is sensitive to the formulation and the mesh – ccx × Abaqus differ up to 5×
    (K6, 10 kN: 579 × 120 MPa), the singularity at a sharp edge grows with the refinement of the mesh; reliable are the mean pressure
    from the force (`p_force_MPa`) and the resultant.
    """
    by: dict[str, list] = {}
    for s in steps:
        for pair, c in (getattr(s, "contacts", None) or {}).items():
            if c.get("peak_at_edge"):
                by.setdefault(pair, []).append((s.name, c))
    out = []
    for pair, rows in by.items():
        name, c = max(rows, key=lambda x: x[1].get("peak_edge_ratio", 0.0))
        steps_txt = ", ".join(f"„{n}“" for n, _ in rows[:3]) + (" …" if len(rows) > 3 else "")
        out.append(f"Contact '{pair}': the largest nodal pressure {c['p_max_MPa']:.4g} MPa is at a free edge of the contact surface (steps {steps_txt}; inside the surface {c['p_max_interior_MPa']:.4g} MPa, mean pressure from the force {c['p_force_MPa']:.4g} MPa) – a peak at the edge depends on the formulation and mesh (another program or a finer mesh gives another value); judge the mean pressure and the resultant, check the peak by mesh refinement")
    return out
