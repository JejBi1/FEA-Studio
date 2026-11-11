"""Forces in the joints after the calculation (each load state separately – linear, superposable).

Bolts: forces and moments in the constraint faces (RF − F), the force in the joint between the clamped parts in the "cell"
of the bolt (nodes of the joint assigned to the nearest axis of the bolt that clamps both parts) from the nodal forces of the elements
of the upper part. Transfer through the joint plane = joint + bolt:
  F_A  = (F_joint + F_bolt)·e   (> 0 = separation), F_SA = F_bolt·e, F_PA = F_joint·e,
  F_Q  = |component ⟂ e|,  M_b = bending moment in the bolt in the plane of the joint.
Welds: free body of the weld, forces in the nodes of the contact faces from the nodal forces of the elements of the weld, by segments
along the root. Force in the effective section of the segment T = (F_A − F_B)/2 (root nodes subtracted),
σ⊥ = −T·n_t/(a·L), τ⊥ = T·t/(a·L), τ∥ = T·e/(a·L); n_t = normal of the plane of the effective section
(contains the root and the perpendicular to the face of the weld), oriented from part A to part B.
"""
from __future__ import annotations

import numpy as np

from ..mesh import Mesh, tri_areas
from .fem import coupling_keys
from .forces import elements_touching, mesh_interfaces, nodal_force_sum


# ===================================================================== geometry (once per mesh)

def _polyline_param(P: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parameter (arc length) and the distance of points X projected onto the polyline P."""
    seg = P[1:] - P[:-1]
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    best_s = np.zeros(len(X))
    best_d = np.full(len(X), np.inf)
    for k in range(len(seg)):
        if L[k] <= 0:
            continue
        t = np.clip(((X - P[k]) @ seg[k]) / L[k] ** 2, 0.0, 1.0)
        q = P[k] + t[:, None] * seg[k]
        d = np.linalg.norm(X - q, axis=1)
        better = d < best_d
        best_d[better] = d[better]
        best_s[better] = cum[k] + t[better] * L[k]
    return best_s, best_d


def _root_chain(tri_a: np.ndarray, root: set[int], nodes: np.ndarray) -> np.ndarray | None:
    """Ordered row of the root nodes (corners of the edges of the triangles of the contact face A lying on the root)."""
    adj: dict[int, set[int]] = {}
    for t in tri_a:
        c = t[:3]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if c[i] in root and c[j] in root:
                adj.setdefault(int(c[i]), set()).add(int(c[j]))
                adj.setdefault(int(c[j]), set()).add(int(c[i]))
    if not adj:
        return None
    ends = [k for k, v in adj.items() if len(v) == 1]
    start = ends[0] if ends else next(iter(adj))
    chain, prev, cur = [start], None, start
    while True:
        nxt = [k for k in adj[cur] if k != prev and k not in chain[-2:]]
        nxt = [k for k in nxt if k not in chain or (k == start and len(chain) > 2)]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur == start:
            chain.append(cur)
            break
        chain.append(cur)
    if len(chain) < 2:
        return None
    return nodes[np.array(chain)]


def _part_boundary(mesh: Mesh, pi: int) -> np.ndarray:
    """Surface of one part (corners of the triangles) incl. contact with other parts – for the rays (thickness of the sheet)."""
    from ..mesh import TET_FACES
    T = mesh.tets[mesh.tet_part == pi][:, :4]
    f = T[:, TET_FACES].reshape(-1, 3)
    key = np.sort(f, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return f[cnt[inv.ravel()] == 1]


def _ray_exit(X: np.ndarray, tri: np.ndarray, o: np.ndarray, d: np.ndarray, eps: float) -> float:
    """Nearest intersection of the ray o + t·d (t > eps) with the triangles (Möller–Trumbore, vectorised)."""
    A, B, C = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    e1, e2 = B - A, C - A
    p = np.cross(d, e2)
    det = (e1 * p).sum(axis=1)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - A
    u = (s * p).sum(axis=1) * inv
    q = np.cross(s, e1)
    v = (q * d).sum(axis=1) * inv
    t = (e2 * q).sum(axis=1) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > eps)
    return float(t[hit].min()) if hit.any() else np.inf


def _locate_in(mesh: Mesh, elems: np.ndarray, tree, p: np.ndarray) -> tuple[int, np.ndarray, float]:
    """Element (of elems) containing the point: (element, natural coordinates, error – 0 = inside)."""
    x = mesh.nodes[mesh.tets[elems, :4]]
    _, cand = tree.query(p, k=min(48, len(elems)))
    best = (-1, None, np.inf)
    for c in np.atleast_1d(cand):
        a = x[c, 0]
        J = (x[c, 1:] - a).T
        xi = np.linalg.solve(J, p - a)
        err = max(-xi.min(), xi.sum() - 1.0, 0.0)
        if err < best[2]:
            best = (int(elems[c]), xi, err)
        if err <= 1e-9:
            break
    return best


HOTSPOT = {"fine": (0.4, 1.0, 1.67, -0.67), "coarse": (0.5, 1.5, 1.5, -0.5)}   # IIW type "a": points /t, weights


def _toe_points(mesh: Mesh, seg: list[dict], sides: list, edges: np.ndarray, toe_t: float | None) -> None:
    """Toe of the weld by segments (for each of the two welded parts): the point of the toe, direction from the weld along the surface of the sheet u,
    thickness of the sheet t (ray perpendicular to the sheet, or given) and the IIW reference points for the extrapolation of the
    structural stress (hot-spot, type "a"): fine mesh (h ≤ 0.6 t) 0.4t/1.0t, σ_hs = 1.67σ₀,₄ − 0.67σ₁,₀;
    coarse mesh 0.5t/1.5t, σ_hs = 1.5σ₀,₅ − 0.5σ₁,₅. The points lie just below the surface of the sheet (elements of that part only).
    Stores seg[k]["toe"] = [{part, x, u, t, elems, xi, w, h, mesh, ok}, ...].
    """
    from scipy.spatial import cKDTree
    X = mesh.nodes
    nseg = len(seg)
    for sg in seg:
        sg["toe"] = []
    for pi, tri, nrm, area, s_tri in sides:
        if not len(tri):
            continue
        elems = np.nonzero(mesh.tet_part == pi)[0]
        tree = cKDTree(X[mesh.tets[elems, :4]].mean(axis=1))
        btri = _part_boundary(mesh, pi)
        size = float(np.linalg.norm(np.ptp(X[mesh.tets[elems, :4]].reshape(-1, 3), axis=0)))
        for k in range(nseg):
            lo, hi = edges[k], edges[k + 1]
            m = (s_tri >= lo) & (s_tri < hi) if k < nseg - 1 else (s_tri >= lo)
            if not m.any():
                continue
            sg = seg[k]
            e = np.asarray(sg["e"])
            r0 = np.asarray(sg["x"])
            n = (nrm[m] * area[m, None]).sum(axis=0)          # normal of the contact face (from the weld into the sheet)
            n -= (n @ e) * e
            n /= max(np.linalg.norm(n), 1e-300)
            u = np.cross(n, e)
            u /= max(np.linalg.norm(u), 1e-300)
            c = X[tri[m][:, :3]].reshape(-1, 3)
            if ((c - r0) @ u).mean() < 0:
                u = -u
            z = float(((c - r0) @ u).max())                     # weld leg on this sheet
            toe = r0 + z * u
            if toe_t:
                t = float(toe_t)
            else:
                off = 1e-6 * size                               # start just above the surface of the sheet
                t = _ray_exit(X, btri, toe + 0.25 * z * u - off * n, n, 10 * off) - off
            rec = {"part": mesh.part_names[pi], "x": toe.tolist(), "u": u.tolist(), "n": n.tolist(), "t": t,
                   "z": z, "ok": bool(np.isfinite(t))}
            if not rec["ok"]:
                sg["toe"].append(rec)
                continue
            for kind in ("fine", "coarse"):
                a1, a2, w1, w2 = HOTSPOT[kind]
                pts = [toe + a * t * u + 1e-3 * t * n for a in (a1, a2)]
                loc = [_locate_in(mesh, elems, tree, p) for p in pts]
                hs = [float(np.mean(np.linalg.norm(np.diff(X[mesh.tets[el, [0, 1, 2, 3, 0]]], axis=0), axis=1)))
                      for el, _, _ in loc]
                h = float(np.mean(hs))
                rec.update(elems=[l[0] for l in loc], xi=[l[1].tolist() for l in loc], w=[w1, w2], h=h, mesh=kind,
                           ok=all(l[2] < 1e-3 for l in loc), points=[p.tolist() for p in pts])
                if kind == "fine" and h <= 0.6 * t:
                    break
            sg["toe"].append(rec)


def weld_geometry_mesh(mesh: Mesh, part: str, nseg: int, a_user: float | None, ifaces=None,
                       toe_t: float | None = None) -> dict:
    """Geometry of the weld from the mesh: contact faces, root, segments, normals of the effective section, toes (hot-spot)."""
    from ..loads import part_volumes
    pw = mesh.part_names.index(part)
    ifaces = ifaces if ifaces is not None else mesh_interfaces(mesh)
    fusion: dict[int, np.ndarray] = {}                   # neighbour -> faces of contact (node numbers of the weld)
    for (a, b), d in ifaces.items():
        if a == pw:
            fusion[b] = d["tri"]
        elif b == pw:
            fusion[a] = d["tri_b"]
    if not fusion:
        raise ValueError(f"Weld '{part}' does not touch any part")
    groups = sorted(fusion.items(), key=lambda kv: -tri_areas(mesh.nodes, kv[1]).sum())
    elems = np.nonzero(mesh.tet_part == pw)[0]
    vol = float(part_volumes(mesh)[pw])
    btri, belem = mesh.boundary_faces()
    free = btri[np.isin(belem, elems)]
    (pa, tri_a) = groups[0]
    (pb, tri_b) = groups[1] if len(groups) > 1 else (None, np.zeros((0, tri_a.shape[1]), int))
    nodes = mesh.nodes
    root = set(np.intersect1d(tri_a, tri_b).tolist()) if pb is not None else set()
    chain = _root_chain(tri_a, root, nodes) if root else None
    kind = "fillet" if chain is not None else "butt"
    A1 = float(tri_areas(nodes, tri_a).sum())
    A2 = float(tri_areas(nodes, tri_b).sum()) if len(tri_b) else 0.0
    if chain is None:                                  # butt weld / one contact face: the axis = the main direction
        c = nodes[np.unique(tri_a)]
        c0 = c.mean(axis=0)
        _, _, vt = np.linalg.svd(c - c0, full_matrices=False)
        ax = vt[0]
        s = (c - c0) @ ax
        chain = np.array([c0 + s.min() * ax, c0 + s.max() * ax])
    seglen = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    L = float(seglen.sum())
    # a from the geometry (fillet: L, contact faces and volume; the angle from the mean normals per segment below)
    def tri_normals(tri):
        X = nodes[tri[:, :3]]
        n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300), 0.5 * np.linalg.norm(n, axis=1)
    edges = np.linspace(0.0, L, nseg + 1)
    seg = []
    sa, _ = _polyline_param(chain, nodes[tri_a[:, :3]].mean(axis=1))
    sb, _ = _polyline_param(chain, nodes[tri_b[:, :3]].mean(axis=1)) if len(tri_b) else (np.zeros(0), None)
    sf, _ = _polyline_param(chain, nodes[free[:, :3]].mean(axis=1))
    na, aa = tri_normals(tri_a)
    nb, ab = tri_normals(tri_b) if len(tri_b) else (np.zeros((0, 3)), np.zeros(0))
    nf, af = tri_normals(free)
    # orientation of the normals outwards from the weld: contact faces – against the centroid of the weld
    cw = nodes[mesh.tets[elems, :4]].mean(axis=(0, 1))
    sign = lambda tri, n: np.where(((nodes[tri[:, :3]].mean(axis=1) - cw) * n).sum(axis=1) < 0, -1.0, 1.0)
    na *= sign(tri_a, na)[:, None]
    if len(tri_b):
        nb *= sign(tri_b, nb)[:, None]
    a_vals = []
    for k in range(nseg):
        lo, hi = edges[k], edges[k + 1]
        ma = (sa >= lo) & (sa < hi) if k < nseg - 1 else (sa >= lo)
        mb = (sb >= lo) & (sb < hi) if k < nseg - 1 else (sb >= lo)
        mf = (sf >= lo) & (sf < hi) if k < nseg - 1 else (sf >= lo)
        smid = 0.5 * (lo + hi)
        # tangent of the root at the middle of the segment
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        j = int(np.clip(np.searchsorted(cum, smid) - 1, 0, len(seglen) - 1))
        e_par = (chain[j + 1] - chain[j]) / max(seglen[j], 1e-300)
        if kind == "fillet":
            # normal of the face of the weld (the surface of the mesh has outward normals); the toes of the weld (normal ∥ axis) are not counted.
            # The sign does not matter – the orientation n_t is set below by the direction from part A to part B.
            keep = mf & (np.abs(nf @ e_par) < 0.7)
            n_free = (nf[keep] * af[keep, None]).sum(axis=0)
            n_free -= (n_free @ e_par) * e_par
            nn = np.linalg.norm(n_free)
            if nn < 1e-12:
                ua = (na[ma] * aa[ma, None]).sum(axis=0)
                ub = (nb[mb] * ab[mb, None]).sum(axis=0)
                n_free = -(ua / max(np.linalg.norm(ua), 1e-300) + ub / max(np.linalg.norm(ub), 1e-300))
                n_free -= (n_free @ e_par) * e_par
                nn = np.linalg.norm(n_free)
            n_free /= max(nn, 1e-300)
            n_t = np.cross(e_par, n_free)
        else:
            ua = (na[ma] * aa[ma, None]).sum(axis=0)
            n_t = -ua / max(np.linalg.norm(ua), 1e-300)      # normal of the contact face A towards the weld
            n_t -= (n_t @ e_par) * e_par
        n_t /= max(np.linalg.norm(n_t), 1e-300)
        ca = nodes[tri_a[ma][:, :3]].mean(axis=1).mean(axis=0) if ma.any() else None
        cb = nodes[tri_b[mb][:, :3]].mean(axis=1).mean(axis=0) if mb.any() else None
        if ca is not None and cb is not None and (cb - ca) @ n_t < 0:
            n_t = -n_t
        t_perp = np.cross(n_t, e_par)
        Aa = float(aa[ma].sum())
        Ab = float(ab[mb].sum()) if len(ab) else 0.0
        if kind == "fillet" and Aa > 0 and Ab > 0:
            ua = (na[ma] * aa[ma, None]).sum(axis=0)
            ub = (nb[mb] * ab[mb, None]).sum(axis=0)
            ua /= max(np.linalg.norm(ua), 1e-300)
            ub /= max(np.linalg.norm(ub), 1e-300)
            cos_t = float(np.clip(-ua @ ub, -1, 1))
            sin_t = np.sqrt(max(1 - cos_t ** 2, 1e-12))
            Ls = hi - lo
            z1, z2 = Aa / Ls, Ab / Ls
            a_vals.append((z1 * z2 * sin_t / np.sqrt(z1 ** 2 + z2 ** 2 - 2 * z1 * z2 * cos_t), Ls))
        seg.append({"s0": float(lo), "s1": float(hi), "e": e_par.tolist(), "n": n_t.tolist(), "t": t_perp.tolist(),
                    "x": (chain[j] + (smid - cum[j]) * e_par).tolist()})
    if a_user:
        a = float(a_user)
        a_src = "entered"
    elif kind == "fillet" and a_vals:
        a = float(sum(v * l for v, l in a_vals) / sum(l for _, l in a_vals))
        a_src = "from geometry"
    else:
        a = A1 / L if L > 0 else 0.0
        a_src = "from geometry (butt weld: thickness of the contact face)"
    # assignment of the nodes of the contact faces to the segments; the root node belongs to the part to which it is connected,
    # (disconnected contact: to one only); a node connected to both parts is not counted
    if kind == "fillet" and pb is not None:
        try:
            _toe_points(mesh, seg, [(pa, tri_a, na, aa, sa), (pb, tri_b, nb, ab, sb)], edges, toe_t)
        except Exception as ex:  # noqa: BLE001 – the toe is a supplement, the root is assessed even without it
            for sg in seg:
                sg["toe"] = []
            seg[0]["toe_error"] = str(ex)
    root_arr = np.array(sorted(root), int)
    in_a = np.zeros(len(nodes), bool)
    in_a[np.unique(mesh.tets[mesh.tet_part == pa])] = True
    in_b = np.zeros(len(nodes), bool)
    if pb is not None:
        in_b[np.unique(mesh.tets[mesh.tet_part == pb])] = True
    root_a = root_arr[in_a[root_arr] & ~in_b[root_arr]]
    root_b = root_arr[in_b[root_arr] & ~in_a[root_arr]]
    root_shared = root_arr[in_a[root_arr] & in_b[root_arr]]
    na_nodes = np.union1d(np.setdiff1d(np.unique(tri_a), root_arr), root_a)
    nb_nodes = (np.union1d(np.setdiff1d(np.unique(tri_b), root_arr), root_b) if len(tri_b)
                else np.zeros(0, int))
    s_na, _ = _polyline_param(chain, nodes[na_nodes])
    s_nb, _ = _polyline_param(chain, nodes[nb_nodes]) if len(nb_nodes) else (np.zeros(0), None)
    bins = lambda s: np.clip(np.floor(s / max(L, 1e-300) * nseg).astype(int), 0, nseg - 1)
    return {
        "part": part, "part_index": pw, "kind": kind, "a": a, "a_source": a_src, "length": L, "volume": vol,
        "part_a": mesh.part_names[pa], "part_b": mesh.part_names[pb] if pb is not None else None,
        "area_a": A1, "area_b": A2, "elems": elems, "segments": seg, "chain": chain,
        "nodes_a": na_nodes, "seg_a": bins(s_na), "nodes_b": nb_nodes,
        "seg_b": bins(s_nb) if len(nb_nodes) else np.zeros(0, int), "root_nodes": root_shared,
        "open_root": bool(len(root_arr)) and not len(root_shared),
    }


def bolt_interfaces(study, mesh: Mesh, ifaces=None) -> dict[str, list[dict]]:
    """Joints between the clamped parts after the bolts: nodes of the joint in the cell of the bolt, elements of the upper part."""
    ifaces = ifaces if ifaces is not None else mesh_interfaces(mesh)
    bolts = study.active_bolts()
    pidx = {n: i for i, n in enumerate(mesh.part_names)}
    # pairs of parts (upper, lower) and the position of the joint plane for each bolt
    planes: list[tuple[int, str, str, float]] = []
    for bi, b in enumerate(bolts):
        stack = [(pl["part"], pl["s0"], pl["s1"]) for pl in b.plates]
        if b.connection == "tapped":
            stack.append((b.nut_part, b.nut_s, b.thread_end))
        for k in range(len(stack) - 1):
            if stack[k][0] != stack[k + 1][0]:
                planes.append((bi, stack[k][0], stack[k + 1][0], 0.5 * (stack[k][2] + stack[k + 1][1])))
    out: dict[str, list[dict]] = {b.name: [] for b in bolts}
    by_pair: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for bi, up, lo, s in planes:
        by_pair.setdefault((up, lo), []).append((bi, s))
    for (up, lo), lst in by_pair.items():
        if up not in pidx or lo not in pidx:
            continue
        key = (min(pidx[up], pidx[lo]), max(pidx[up], pidx[lo]))
        if key not in ifaces:
            for bi, s in lst:
                out[bolts[bi].name].append({"upper": up, "lower": lo, "s": s, "nodes": np.zeros(0, int),
                                            "elems": np.zeros(0, int), "area": 0.0})
            continue
        tri = ifaces[key]["tri"]
        nodes = np.unique(tri)
        X = mesh.nodes[nodes]
        dist = np.full((len(lst), len(nodes)), np.inf)
        for j, (bi, _) in enumerate(lst):
            b = bolts[bi]
            r = X - np.asarray(b.origin)
            dist[j] = np.linalg.norm(r - np.outer(r @ b.e, b.e), axis=1)
        owner = np.argmin(dist, axis=0)
        cent = mesh.nodes[tri[:, :3]].mean(axis=1)
        dc = np.full((len(lst), len(tri)), np.inf)
        for j, (bi, _) in enumerate(lst):
            b = bolts[bi]
            r = cent - np.asarray(b.origin)
            dc[j] = np.linalg.norm(r - np.outer(r @ b.e, b.e), axis=1)
        tri_owner = np.argmin(dc, axis=0)
        areas = tri_areas(mesh.nodes, tri)
        elems = elements_touching(mesh, pidx[up], nodes)
        for j, (bi, s) in enumerate(lst):
            out[bolts[bi].name].append({"upper": up, "lower": lo, "s": s, "nodes": nodes[owner == j],
                                        "elems": elems, "area": float(areas[tri_owner == j].sum())})
    return out


def joint_geometry(study, mesh: Mesh) -> dict:
    """Everything about the joints needed from the mesh (computed once)."""
    ifaces = mesh_interfaces(mesh) if (study.active_bolts() or study.active_welds()) else {}
    welds, warnings = {}, []
    for w in study.active_welds():
        if w.part not in mesh.part_names:
            warnings.append(f"{w.name}: part '{w.part}' is not in the mesh")
            continue
        try:
            welds[w.name] = weld_geometry_mesh(mesh, w.part, max(1, int(w.segments)), w.a, ifaces,
                                               getattr(w, "toe_t", None))
        except ValueError as e:
            warnings.append(f"{w.name}: {e}")
    bolts = bolt_interfaces(study, mesh, ifaces) if study.active_bolts() else {}
    return {"welds": welds, "bolts": bolts, "warnings": warnings}


def required_elements(geom: dict) -> np.ndarray:
    """Elements for which the stresses at the integration points are needed (*EL PRINT)."""
    parts = [g["elems"] for g in geom["welds"].values()]
    parts += [iface["elems"] for lst in geom["bolts"].values() for iface in lst]
    if not parts:
        return np.zeros(0, int)
    return np.unique(np.concatenate(parts))


# ===================================================================== forces in one state

def step_forces(study, mesh: Mesh, geom: dict, RF: np.ndarray, F: np.ndarray, sig: dict | None) -> dict:
    out = {"bolts": {}, "welds": {}}
    for i, b in enumerate(study.active_bolts(), 1):
        ends = []
        for key, s in coupling_keys(b, i, mesh):
            idx = mesh.surface_nodes(key)
            f = (RF - F)[idx]
            p = b.point(s)
            ends.append({"s": s, "F": f.sum(axis=0), "M": np.cross(mesh.nodes[idx] - p, f).sum(axis=0)})
        planes = []
        cache: dict[int, np.ndarray] = {}
        for iface in geom["bolts"].get(b.name, []):
            s = iface["s"]
            p = b.point(s)
            Fb, Mb = np.zeros(3), np.zeros(3)
            for en in ends:                           # the bolt transfers the forces of the constraints above the joint plane
                if en["s"] < s:
                    Fb += en["F"]
                    Mb += en["M"] + np.cross(b.point(en["s"]) - p, en["F"])
            Fi, Mi = np.zeros(3), np.zeros(3)
            if sig is not None and len(iface["nodes"]):
                key = id(iface["elems"])
                if key not in cache:
                    cache[key] = nodal_force_sum(mesh, iface["elems"], sig)
                fn = cache[key][iface["nodes"]]
                Fi = fn.sum(axis=0)
                Mi = np.cross(mesh.nodes[iface["nodes"]] - p, fn).sum(axis=0)
            planes.append({"s": s, "upper": iface["upper"], "lower": iface["lower"], "area": iface["area"],
                           "F_bolt": Fb, "M_bolt": Mb, "F_int": Fi, "M_int": Mi})
        out["bolts"][b.name] = {"ends": ends, "planes": planes}
    for w in study.active_welds():
        g = geom["welds"].get(w.name)
        if g is None or sig is None:
            continue
        fw = nodal_force_sum(mesh, g["elems"], sig)
        nseg = len(g["segments"])
        FA = np.zeros((nseg, 3))
        FB = np.zeros((nseg, 3))
        np.add.at(FA, g["seg_a"], fw[g["nodes_a"]])
        if len(g["nodes_b"]):
            np.add.at(FB, g["seg_b"], fw[g["nodes_b"]])
        T = 0.5 * (FA - FB) if g["kind"] == "fillet" or len(g["nodes_b"]) else FA
        a = g["a"]
        res = []
        for k, sg in enumerate(g["segments"]):
            Ls = sg["s1"] - sg["s0"]
            A = a * Ls
            e, n, t = (np.asarray(sg[x]) for x in ("e", "n", "t"))
            res.append([-T[k] @ n / A, T[k] @ t / A, T[k] @ e / A])
        F_root = fw[g["root_nodes"]].sum(axis=0) if len(g["root_nodes"]) else np.zeros(3)
        total = FA.sum(axis=0) + FB.sum(axis=0) + F_root
        scale = max(np.abs(FA).sum(), np.abs(FB).sum(), 1e-30)
        out["welds"][w.name] = {"T": T, "stress": np.asarray(res), "F_a": FA.sum(axis=0), "F_b": FB.sum(axis=0),
                                "F_root": F_root, "imbalance": float(np.linalg.norm(total) / scale)}
    return out


def to_json(d):
    """numpy → lists (storage in the meta data of the results)."""
    if isinstance(d, dict):
        return {k: to_json(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [to_json(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d
