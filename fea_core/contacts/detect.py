"""Search for the touches and the contact faces (F8.5, specification 4.2).

- `find_touching(parts, tol)` (OCP, before meshing, "Find touches"): pairs of parts within the distance tol with the state
  contact / gap / penetration (volume of the intersection BRepAlgoAPI_Common), area and list of the touching faces;
- `mesh_pairs(study, mesh)` (after meshing): faces of the pair `__c{i}_s` (slave) / `__c{i}_m` (master) from the mesh
  (selections of faces, or the faces of the surface of a part within the tolerance from the other part), slave = the finer mesh (on equality the softer
  material), the initial gap with a sign in the nodes of the slave (− = penetration), contact area;
- `split_interface(mesh, a, b)` – conforming mesh: doubling of the nodes of the common faces of parts a/b (part b gets new
  nodes) → two faces exactly opposite each other, without the initial gap (the interface of a bolted joint).
"""
from __future__ import annotations

import numpy as np

PREFIX = "__c"


def default_tol(study) -> float:
    t = study.contact_settings.detect_tol
    return float(t) if t else max(0.2 * float(study.mesh.size), 0.05)


# ------------------------------------------------------------------ geometrie (OCP)

def find_touching(parts, tol: float = 0.05, names=None) -> list[dict]:
    """Pairs of parts within the distance tol: [{"a", "b", "distance", "state" contact|gap|penetration, "area" mm² (faces
    within tol), "overlap_volume" mm³, "faces_a", "faces_b" (indices of the faces)}].
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    sel = [p for p in parts if names is None or p.name in names]
    boxes = []
    for p in sel:
        b = Bnd_Box()
        BRepBndLib.Add_s(p.shape, b)
        lo, hi = b.CornerMin(), b.CornerMax()             # Bnd_Box.Get() in OCP 8 cannot be converted (TypeError)
        boxes.append((lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()))
    out = []

    def faces(shape):                                     # order = geometry.list_faces (id 'Part#i')
        from ..geometry import list_faces
        return list_faces(shape)

    def area(face):
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, g)
        return g.Mass()

    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            a, b = boxes[i], boxes[j]
            if any(a[k] - tol > b[k + 3] or b[k] - tol > a[k + 3] for k in range(3)):
                continue
            d = BRepExtrema_DistShapeShape(sel[i].shape, sel[j].shape)
            if not d.IsDone() or d.Value() > tol:
                continue
            vol = 0.0
            try:
                com = BRepAlgoAPI_Common(sel[i].shape, sel[j].shape)
                g = GProp_GProps()
                BRepGProp.VolumeProperties_s(com.Shape(), g)
                vol = max(g.Mass(), 0.0)
            except Exception:  # noqa: BLE001 – the intersection not computed (a degenerate contact)
                vol = 0.0
            fa, fb, ar = [], [], 0.0
            for k, f in enumerate(faces(sel[i].shape)):
                if _face_touches(f, sel[j].shape, tol):
                    fa.append(k)
                    ar += area(f)
            ar_b = 0.0
            for k, f in enumerate(faces(sel[j].shape)):
                if _face_touches(f, sel[i].shape, tol):
                    fb.append(k)
                    ar_b += area(f)
            ar = min(ar, ar_b) if fa and fb else max(ar, ar_b)    # contact ≤ the smaller of the touching faces
            vtot = min(_volume(sel[i].shape), _volume(sel[j].shape))
            state = "penetration" if vol > 1e-6 * max(vtot, 1e-30) else ("contact" if d.Value() <= 1e-6 else "gap")
            out.append({"a": sel[i].name, "b": sel[j].name, "distance": float(d.Value()), "state": state,
                        "area": float(ar), "overlap_volume": float(vol), "faces_a": fa, "faces_b": fb,
                        "depth": float(vol / ar) if ar > 0 and state == "penetration" else 0.0})
    return out


def _face_touches(face, other, tol: float) -> bool:
    """A face lies on the second part: it is within tol and its inner point (the centre of the parameters) is within tol too – side faces that
    touch the other part only by an edge are not counted (two blocks on top of each other: 400 mm², not 1200).
    """
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepTools import BRepTools
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    dd = BRepExtrema_DistShapeShape(face, other)
    if not dd.IsDone() or dd.Value() > tol:
        return False
    try:
        u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
        pt = BRep_Tool.Surface_s(face).Value(0.5 * (u0 + u1), 0.5 * (v0 + v1))
        dv = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(pt).Vertex(), other)
        return dv.IsDone() and dv.Value() <= tol * 1.01 + 1e-6
    except Exception:  # noqa: BLE001 – without an inner point as before
        return True


def _volume(shape) -> float:
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return abs(g.Mass())


# ------------------------------------------------------------------ mesh

def _part_of_selection(study, name) -> set[str]:
    sel = study.selections.get(name)
    return {f.part for f in sel.faces} if sel is not None else {name}


def pair_parts(study, pair) -> tuple[set[str], set[str]]:
    """Parts of side a and b of the pair (the name of the part or the selection of faces)."""
    return _part_of_selection(study, pair.a), _part_of_selection(study, pair.b)


def contact_parts(study) -> set[str]:
    """Parts of the active contact pairs (they are meshed separately without *TIE between them – split separate)."""
    out = set()
    if study.contact_split() != "separate":
        return out
    for c in study.active_contacts():
        if c.kind == "compression_only" and not c.b:
            continue
        a, b = pair_parts(study, c)
        out |= a | b
    return out


def contact_part_pairs(study) -> set[frozenset]:
    """Pairs of parts between which there is a contact (*TIE must not connect them)."""
    out = set()
    for c in study.active_contacts():
        a, b = pair_parts(study, c)
        out |= {frozenset((x, y)) for x in a for y in b if x != y}
    return out


def _faces_of_part(mesh, pi: int):
    tri, elem = mesh.boundary_faces()
    m = mesh.tet_part[elem] == pi
    return tri[m]


def _outward(mesh, tri: np.ndarray, n: np.ndarray, what: str) -> np.ndarray:
    """Flips the face normals n (in place) to point OUT of the element (against the opposite corner).

    Without the orientation the sign of the gap / the direction of the normal is undefined – the contact result would be
    wrong, therefore an error (never a silent pass).
    """
    from ..loads import _OPPOSITE
    from ..mesh import surface_element_faces
    X = mesh.nodes
    try:
        el, fno = surface_element_faces(mesh.tets, tri)
    except Exception as e:  # noqa: BLE001 – reported with the context
        raise ValueError(f"Contact {what}: the faces do not match faces of the mesh elements – the outward normal "
                         f"(sign of the gap) cannot be determined ({e})") from None
    opp = X[mesh.tets[el, _OPPOSITE[fno - 1]]]
    a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    n[np.einsum("ij,ij->i", n, (a + b + c) / 3 - opp) < 0] *= -1
    return n


def face_normals(mesh, tri: np.ndarray) -> np.ndarray:
    """Unit OUTWARD normals of the faces (orientation against the opposite corner of the element)."""
    X = mesh.nodes
    a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    return _outward(mesh, tri, n, "faces")


def _touching(X, src_tri, dst_tri, tol, mesh=None, opposing: float = -0.3):
    """Faces src whose centroid lies within tol from the surface of dst (distance point–triangle) and whose nearest
    face dst is OPPOSITE (n_src·n_dst < opposing) – side faces and planes of symmetry do not belong to the pair.
    """
    from scipy.spatial import cKDTree
    from ..ties import _point_tri_dist
    if not len(src_tri) or not len(dst_tri):
        return np.zeros(len(src_tri), bool)
    ns = face_normals(mesh, src_tri) if mesh is not None else None
    nd = face_normals(mesh, dst_tri) if mesh is not None else None
    cs = X[src_tri[:, :3]].mean(axis=1)
    cd = X[dst_tri[:, :3]].mean(axis=1)
    tree = cKDTree(cd)
    span = np.linalg.norm(X[dst_tri[:, 0]] - X[dst_tri[:, 1]], axis=1).max()
    cand = tree.query_ball_point(cs, r=span + tol)
    ok = np.zeros(len(src_tri), bool)
    cnt = np.fromiter((len(c) for c in cand), np.int64, len(cand))
    if not cnt.sum():
        return ok
    si = np.repeat(np.arange(len(cand)), cnt)                      # pair (face src, candidate dst) at once
    di = np.fromiter((j for c in cand for j in c), np.int64, int(cnt.sum()))
    for k in range(0, len(si), 1_000_000):                         # by batches (memory)
        s, d = si[k:k + 1_000_000], di[k:k + 1_000_000]
        near = _point_tri_dist(cs[s], X[dst_tri[d, 0]], X[dst_tri[d, 1]], X[dst_tri[d, 2]]) <= tol
        if ns is not None:
            near &= np.einsum("ij,ij->i", nd[d], ns[s]) < opposing
        ok[s[near]] = True
    return ok


def signed_gap(mesh, slave_tri: np.ndarray, master_tri: np.ndarray, master_part_elems=None) -> tuple[np.ndarray, np.ndarray]:
    """Gap with a sign in the nodes of the slave face (nodes, gap; − = penetration) with respect to the nearest face of the master: perpendicular
    to the master face with the outward normal (orientation according to the opposite corner of the element).
    """
    from scipy.spatial import cKDTree
    from ..ties import _point_tri_dist
    X = mesh.nodes
    nodes = np.unique(slave_tri[:, :6] if slave_tri.shape[1] >= 6 else slave_tri)
    if not len(master_tri):
        return nodes, np.full(len(nodes), np.inf)
    a, b, c = X[master_tri[:, 0]], X[master_tri[:, 1]], X[master_tri[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    # outward normal: against the opposite corner of the element of the face
    _outward(mesh, master_tri, n, "master face")
    cen = (a + b + c) / 3
    tree = cKDTree(cen)
    k = min(8, len(cen))
    _, cand = tree.query(X[nodes], k=k)
    cand = np.atleast_2d(cand)
    if cand.shape[0] != len(nodes):
        cand = cand.T
    best = np.full(len(nodes), np.inf)
    sgn = np.ones(len(nodes))
    for j in range(cand.shape[1]):
        ci = cand[:, j]
        d = _point_tri_dist(X[nodes], a[ci], b[ci], c[ci])
        s = np.sign(np.einsum("ij,ij->i", X[nodes] - cen[ci], n[ci]))
        s[s == 0] = 1.0
        upd = d < best
        best[upd] = d[upd]
        sgn[upd] = s[upd]
    return nodes, best * sgn


def mesh_pairs(study, mesh) -> list[str]:
    """Contact faces of the active pairs in the mesh → mesh.surfaces __c{i}_s/_m, mesh.report["contacts"]. Warnings."""
    warnings = []
    mesh.report["contacts"] = []
    tol = default_tol(study)
    X = mesh.nodes
    for i, c in enumerate(study.all_contacts(), 1):
        if not c.enabled or c.kind in ("bonded", "none"):
            continue
        if c.kind == "compression_only" and not c.b:
            if c.a not in mesh.surfaces:
                warnings.append(f"Contact '{c.name}': selection '{c.a}' is not in the mesh")
                continue
            mesh.report["contacts"].append({"pair": c.name, "index": i, "slave": c.a, "master": None, "kind": c.kind})
            continue
        sides = []
        for side in (c.a, c.b):
            if side in mesh.surfaces:
                sides.append(("sel", side, mesh.surfaces[side]))
            elif side in mesh.part_names:
                sides.append(("part", side, _faces_of_part(mesh, mesh.part_names.index(side))))
            else:
                sides.append((None, side, np.zeros((0, 6), np.int64)))
        if any(s[0] is None for s in sides):
            warnings.append(f"Contact '{c.name}': '{[s[1] for s in sides if s[0] is None][0]}' is neither a part nor a selection in the mesh")
            continue
        ta, tb = sides[0][2], sides[1][2]
        split = next(((mesh.surfaces[f"__split_{x}|{y}_a"], mesh.surfaces[f"__split_{x}|{y}_b"], inv)
                      for x, y, inv in ((c.a, c.b, False), (c.b, c.a, True))
                      if f"__split_{x}|{y}_a" in mesh.surfaces), None)
        if split is not None and sides[0][0] == "part" and sides[1][0] == "part":
            ta, tb = (split[1], split[0]) if split[2] else (split[0], split[1])   # disconnected conforming joint
        else:
            if sides[0][0] == "part":
                ta = ta[_touching(X, ta, tb, tol, mesh)]
            if sides[1][0] == "part":
                tb = tb[_touching(X, tb, ta, tol, mesh)]
        if not len(ta) or not len(tb):
            warnings.append(f"Contact '{c.name}': the faces do not touch in the mesh (tolerance {tol:.3g} mm) – pair skipped; specify face selections or a larger tolerance")
            continue
        size = [float(np.mean(np.linalg.norm(X[t[:, 0]] - X[t[:, 1]], axis=1))) for t in (ta, tb)]
        if c.slave in ("a", "b"):
            s_first = c.slave == "a"
        else:
            s_first = size[0] < size[1] * 0.999
            if abs(size[0] - size[1]) <= 0.001 * max(size):
                try:
                    ea = [study.material_of(p).E for p in pair_parts(study, c)[0]]
                    eb = [study.material_of(p).E for p in pair_parts(study, c)[1]]
                    s_first = min(ea) <= min(eb)
                except KeyError:
                    s_first = True
        st, mt = (ta, tb) if s_first else (tb, ta)
        mesh.surfaces[f"{PREFIX}{i}_s"] = st
        mesh.surfaces[f"{PREFIX}{i}_m"] = mt
        nodes, gap = signed_gap(mesh, st, mt)
        area = float(np.linalg.norm(np.cross(X[st[:, 1]] - X[st[:, 0]], X[st[:, 2]] - X[st[:, 0]]), axis=1).sum() / 2)
        gmin = float(gap.min()) if len(gap) else 0.0
        h = min(size)
        # penetration from 0.1 % of the size of the element already: penalty K = 50·E without ADJUST sends it into the load (p = K·δ, 0.01 mm
        # → 1e5 MPa) and the calculation does not converge (K8); the contact is tolerated up to 2 % h (chord error of curved faces)
        state = "penetration" if gmin < -max(1e-3 * h, 1e-4) else ("contact" if abs(gmin) <= max(0.02 * h, 1e-4)
                                                                   else "gap")
        mesh.report["contacts"].append({"pair": c.name, "index": i, "slave": f"{PREFIX}{i}_s",
                                        "master": f"{PREFIX}{i}_m", "kind": c.kind, "area_mm2": area,
                                        "gap_min_mm": gmin, "gap_max_mm": float(gap.max()) if len(gap) else 0.0,
                                        "state": state, "slave_side": "a" if s_first else "b",
                                        "element_size_mm": h})
    return warnings


def split_interface(mesh, pa: str, pb: str) -> int:
    """Conforming mesh: disconnects the common faces of the parts pa and pb – for the elements of part pb the nodes of the faces get copies
    (mesh.node_alias = the original number). The faces of both sides → mesh.surfaces "__split_{pa}|{pb}_a/_b". Returns the number of
    doubled nodes.
    """
    from ..mesh import TET_FACES
    from ..contacts.bonded import bonded_pairs
    ia, ib = mesh.part_names.index(pa), mesh.part_names.index(pb)
    shared = [p for p in bonded_pairs(mesh) if {p["a"], p["b"]} == {pa, pb}]
    if not shared:
        return 0
    # common faces: faces of the elements of parts a and b with the same corner nodes (vectorised – the key of the face = the sorted corners)
    from ..mesh import TET10_FACE_MIDS
    T = mesh.tets
    el_a, el_b = np.nonzero(mesh.tet_part == ia)[0], np.nonzero(mesh.tet_part == ib)[0]
    ca = np.sort(T[el_a][:, TET_FACES].reshape(-1, 3), axis=1)    # faces by (element, face 0–3)
    cb = np.sort(T[el_b][:, TET_FACES].reshape(-1, 3), axis=1)
    n0 = len(mesh.nodes)
    ka, kb = _face_keys(n0, ca, cb)
    ua, ia_first = np.unique(ka, return_index=True)
    common_a = np.sort(ia_first[np.isin(ua, kb)])                 # order of the faces of part a
    if not len(common_a):
        return 0
    ub, ib_first = np.unique(kb, return_index=True)
    common_b = ib_first[np.searchsorted(ub, ka[common_a])]

    def tri6(els, idx):
        e, f = els[idx // 4], idx % 4
        return np.concatenate([T[e[:, None], TET_FACES[f]], T[e[:, None], TET10_FACE_MIDS[f]]], axis=1)
    tri_a, tb = tri6(el_a, common_a), tri6(el_b, common_b)
    nodes = np.unique(tri_a)
    remap = np.arange(n0 + len(nodes))
    remap[nodes] = n0 + np.arange(len(nodes))
    mesh.nodes = np.vstack([mesh.nodes, mesh.nodes[nodes]])
    alias = mesh.canonical().copy() if mesh.node_alias is not None else np.arange(n0)
    mesh.node_alias = np.concatenate([alias, alias[nodes]])
    mesh.tets[el_b] = remap[mesh.tets[el_b]]
    tb = remap[tb]
    # selections of faces on the faces of part b: renumber the nodes (the face of part b after the renumbering, not the face of part a)
    cb_new = np.sort(remap[cb], axis=1)
    for name, st in list(mesh.surfaces.items()):
        if not len(st) or name.startswith("__split_"):
            continue
        st2 = remap[st]
        k2, kb_new, k1, ka_all = _face_keys(len(remap), np.sort(st2[:, :3], axis=1), cb_new,
                                            np.sort(st[:, :3], axis=1), ca)
        use = np.isin(k2, kb_new) & ~np.isin(k1, ka_all)
        if use.any():
            st = st.copy()
            st[use] = st2[use]
            mesh.surfaces[name] = st
    mesh.surfaces[f"__split_{pa}|{pb}_a"] = tri_a
    mesh.surfaces[f"__split_{pa}|{pb}_b"] = tb
    mesh._cache.clear()
    return len(nodes)


def _face_keys(n: int, *corners) -> list[np.ndarray]:
    """Integer keys of the faces from the sorted corners (the same corners = the same key across all the arrays)."""
    if n < 2 ** 21:
        return [(c[:, 0].astype(np.int64) * n + c[:, 1]) * n + c[:, 2] for c in corners]
    _, inv = np.unique(np.vstack(corners), axis=0, return_inverse=True)
    return np.split(inv.ravel().astype(np.int64), np.cumsum([len(c) for c in corners])[:-1])


def node_normals(mesh, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nodes of the face (corners and midpoints) and the unit OUTWARD normals at them (average of the normals of the faces weighted by area)."""
    X = mesh.nodes
    a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    n = 0.5 * np.cross(b - a, c - a)
    _outward(mesh, tri, n, "slave face")
    cols = tri[:, :6] if tri.shape[1] >= 6 else tri
    nodes = np.unique(cols)
    acc = np.zeros((len(nodes), 3))
    idx = np.searchsorted(nodes, cols)
    for k in range(cols.shape[1]):
        np.add.at(acc, idx[:, k], n)
    acc /= np.maximum(np.linalg.norm(acc, axis=1), 1e-300)[:, None]
    return nodes, acc


_OPPOSITE_NORMALS = node_normals
