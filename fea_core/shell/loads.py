"""Conversion of the constraints and loads of the 3D study to the shell model + integration on tri6 / line3.

A face selection of the 3D body is converted according to the kind of the face of the plate:
  • upper / lower face (skin) → elements of the shell whose centroid lies below the face at the distance t/2
    (projection inwards of the face) – pressure, force on the face, constraint of the whole face,
  • side face → boundary edges of the shell lying on the face (all 3 nodes of line3 within 0.1·t) – constraints of the edge,
    force and pressure along the length (pressure on the side face = p·t per length),
  • planar part from STEP → elements on the face (distance ~0).
Edge of the 3D body (EdgeSelector) → edges of the mesh within 0.55·t (for solids the edge of the upper face lies t/2 from the mid face).
Point (PointSelector) → nearest node of the part.

All loads go as nodal force *CLOAD into the nodes of the shell (consistent integration) – the loads in the
nodes are thus exactly known and the equilibrium check makes sense (ccx distributes the force of a node among the expanded nodes
1/6-2/3-1/6 resp. 1/2-1/2 across the thickness; verified that the sum of the RF of the expanded nodes = the reaction).
Constraints: Fixed = DOF 1–6 (clamping; only DOF 1–3 is a hinge in ccx – verified), Displacement = prescribed displacements;
on a side face additionally zero rotation about a = n × e (n = normal of the shell, e = constrained direction) – for solids the
whole side face is constrained in the direction e, hence also its rotation about a (symmetry, verified on a quarter of a plate).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import ShellError

# Dunavant 6 points (degree 4) on the reference triangle (ξ, η), weights × 1/2 (area of the reference triangle)
_A1, _W1 = 0.445948490915965, 0.223381589678011
_A2, _W2 = 0.091576213509771, 0.109951743655322
TRI_GP = np.array([[_A1, _A1], [1 - 2 * _A1, _A1], [_A1, 1 - 2 * _A1],
                   [_A2, _A2], [1 - 2 * _A2, _A2], [_A2, 1 - 2 * _A2]])
TRI_GW = 0.5 * np.array([_W1] * 3 + [_W2] * 3)
LINE_GP = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
LINE_GW = np.array([5 / 9, 8 / 9, 5 / 9])


def tri6_shape(xi, eta):
    """Shape functions of tri6 (order of ccx S6) and their derivatives with respect to ξ, η: N (6,), dN (6, 2)."""
    l1, l2, l3 = 1 - xi - eta, xi, eta
    N = np.array([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1), 4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1])
    dl = np.array([[-1, -1], [1, 0], [0, 1]], float)
    dN = np.zeros((6, 2))
    for k, (a, la) in enumerate(((0, l1), (1, l2), (2, l3))):
        dN[k] = (4 * la - 1) * dl[a]
    dN[3] = 4 * (l1 * dl[1] + l2 * dl[0])
    dN[4] = 4 * (l2 * dl[2] + l3 * dl[1])
    dN[5] = 4 * (l3 * dl[0] + l1 * dl[2])
    return N, dN


_TRI = [tri6_shape(*p) for p in TRI_GP]
TRI_N = np.array([a for a, _ in _TRI])            # (6 points, 6 nodes)
TRI_DN = np.array([b for _, b in _TRI])           # (6, 6, 2)
TRI_NC = tri6_shape(1 / 3, 1 / 3)[0]              # centroid


def tri6_gauss(nodes: np.ndarray, tri: np.ndarray):
    """Gauss points of the elements: position X (M, 6, 3), area vector dA·n (M, 6, 3) (orientation of the element), N (6, 6)."""
    x = nodes[tri]                                 # (M, 6, 3)
    X = np.einsum("gk,mkc->mgc", TRI_N, x)
    t1 = np.einsum("gk,mkc->mgc", TRI_DN[:, :, 0], x)
    t2 = np.einsum("gk,mkc->mgc", TRI_DN[:, :, 1], x)
    dA = np.cross(t1, t2) * TRI_GW[None, :, None]
    return X, dA, TRI_N


def tri6_integrals(nodes: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(∫N dA (M, 6), obsah prvku (M,))."""
    _, dA, N = tri6_gauss(nodes, tri)
    a = np.linalg.norm(dA, axis=2)                 # (M, 6)
    return a @ N, a.sum(axis=1)


def line3_gauss(nodes: np.ndarray, lines: np.ndarray):
    """line3 [end, end, midpoint]: position X (K, 3, 3), dL (K, 3), tangent (K, 3, 3), N (3, 3)."""
    xi = LINE_GP
    N = np.stack([0.5 * xi * (xi - 1), 0.5 * xi * (xi + 1), 1 - xi ** 2], axis=1)
    dN = np.stack([xi - 0.5, xi + 0.5, -2 * xi], axis=1)
    x = nodes[lines]
    X = np.einsum("gk,mkc->mgc", N, x)
    T = np.einsum("gk,mkc->mgc", dN, x)
    jl = np.linalg.norm(T, axis=2)
    return X, jl * LINE_GW[None, :], T / jl[..., None], N


def scatter(n_nodes: int, conn: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Sum of the values of the elements (K, k, 3) into the nodes (N, 3)."""
    out = np.zeros((n_nodes, 3))
    for c in range(3):
        out[:, c] = np.bincount(conn.ravel(), vals[..., c].ravel(), minlength=n_nodes)
    return out


# ------------------------------------------------------------------ targets of selections

@dataclass
class Target:
    """Selection converted to the mesh of the shell."""
    name: str
    elements: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))     # elements (skin, face)
    # sign n_element · n_face outwards × (area of the face of the body / area of the mid face under it) – for pressure
    elem_sign: np.ndarray = field(default_factory=lambda: np.zeros(0))
    edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))   # line3 (side faces)
    edge_normal: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))        # outward normal of the side face
    edge_thick: np.ndarray = field(default_factory=lambda: np.zeros(0))              # thickness at the edge
    points: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))        # nodes (points)
    side_edges: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))       # edge from a side face (rotation)

    def nodes(self, mesh) -> np.ndarray:
        parts = [mesh.tri[self.elements].ravel(), self.edges.ravel(), self.points]
        return np.unique(np.concatenate(parts).astype(np.int64))

    @property
    def empty(self) -> bool:
        return not (len(self.elements) or len(self.edges) or len(self.points))


class _FaceProbe:
    """Fast queries on an OCP face (precomputed surface and classifier)."""

    def __init__(self, face):
        from OCP.BRep import BRep_Tool
        from OCP.ShapeAnalysis import ShapeAnalysis_Surface
        from ..geometry import bbox
        self.face = face
        self.surf = BRep_Tool.Surface_s(face)
        self.sas = ShapeAnalysis_Surface(self.surf)
        self.bb = bbox(face)
        from OCP.TopAbs import TopAbs_REVERSED
        self.rev = face.Orientation() == TopAbs_REVERSED

    def query(self, p, tol: float):
        """(inside, distance, outward normal) of the point p."""
        from OCP.BRepClass import BRepClass_FaceClassifier
        from OCP.GeomLProp import GeomLProp_SLProps
        from OCP.TopAbs import TopAbs_IN, TopAbs_ON
        from OCP.gp import gp_Pnt, gp_Pnt2d
        uv = self.sas.ValueOfUV(gp_Pnt(*map(float, p)), 1e-7)
        q = self.surf.Value(uv.X(), uv.Y())
        d = float(np.linalg.norm(np.array([q.X(), q.Y(), q.Z()]) - p))
        st = BRepClass_FaceClassifier(self.face, gp_Pnt2d(uv.X(), uv.Y()), max(tol, 1e-7)).State()
        pr = GeomLProp_SLProps(self.surf, uv.X(), uv.Y(), 1, 1e-9)
        n = np.zeros(3)
        if pr.IsNormalDefined():
            v = pr.Normal()
            n = np.array([v.X(), v.Y(), v.Z()]) * (-1.0 if self.rev else 1.0)
        return st in (TopAbs_IN, TopAbs_ON), d, n

    def near_box(self, pts: np.ndarray, gap: float) -> np.ndarray:
        return np.all((pts >= self.bb[:3] - gap) & (pts <= self.bb[3:] + gap), axis=1)


def _edge_distance(edge, pts: np.ndarray) -> np.ndarray:
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt
    out = np.empty(len(pts))
    for i, p in enumerate(pts):
        v = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, p))).Vertex()
        ds = BRepExtrema_DistShapeShape(v, edge)
        out[i] = ds.Value() if ds.IsDone() and ds.NbSolution() else np.inf
    return out


class Mapper:
    """Conversion of the selections of the study (faces / edges / points of the 3D parts) to the elements, edges and nodes of the shell mesh."""

    def __init__(self, mesh, sheets: dict, parts: dict):
        self.mesh, self.sheets, self.parts = mesh, sheets, parts
        self.pidx = {n: i for i, n in enumerate(mesh.part_names)}
        self._faces: dict[str, list] = {}
        self._fcache: dict = {}
        self._ecache: dict = {}
        self._bnd = None
        x = mesh.nodes[mesh.tri]
        self.centroids = np.einsum("k,mkc->mc", TRI_NC, x)
        self.enormal = mesh.normals()

    def faces_of(self, part: str):
        if part not in self._faces:
            from ..geometry import list_faces
            self._faces[part] = list_faces(self.parts[part].shape)
        return self._faces[part]

    def boundary(self):
        if self._bnd is None:
            self._bnd = self.mesh.boundary_edges()
        return self._bnd

    def target(self, name: str, sel) -> Target:
        from ..selectors import resolve, resolve_edge
        tg = Target(name)
        els, signs, edges, enorm, eth, pts, sides = [], [], [], [], [], [], []
        for fs in sel.faces:
            if fs.part not in self.sheets:
                raise ShellError(f"Selection '{name}': part '{fs.part}' is not in the shell model")
            sh = self.sheets[fs.part]
            hit = resolve(fs, [self.parts[fs.part]], self._fcache.setdefault(fs.part, {}))
            if hit is None:
                raise ShellError(f"Selection '{name}': the face of part '{fs.part}' was not found in the geometry")
            fi = hit[1]
            probe = _FaceProbe(self.faces_of(fs.part)[fi])
            pi = self.pidx[fs.part]
            t = sh.thickness
            if sh.kind == "surface" or fi in sh.top or fi in sh.bottom:
                expect = 0.0 if sh.kind == "surface" else 0.5 * t     # planar part: mesh nodes lie on the face of the CAD
                cand = np.nonzero((self.mesh.tri_part == pi) & probe.near_box(self.centroids, t))[0]
                mine, sg = [], []
                for e in cand:
                    ins, d, n = probe.query(self.centroids[e], 1e-3 * t)
                    if ins and abs(d - expect) <= 0.1 * t + 1e-6:
                        mine.append(e)
                        sg.append(np.sign(float(self.enormal[e] @ n)) or 1.0)
                if not mine:
                    raise ShellError(f"Selection '{name}': the face of part '{fs.part}' has no elements in the shell")
                # the pressure acts on the real face of the body: scale area of the face / area of the mid face under it
                # (curved skin – the inner face of a tube is smaller than the mid one; for a planar plate 1)
                a_mid = float(tri6_integrals(self.mesh.nodes, self.mesh.tri[np.asarray(mine)])[1].sum())
                ratio = float(fs.area) / a_mid if a_mid > 0 and sh.kind != "surface" else 1.0
                els += mine
                signs += [g * ratio for g in sg]
            else:
                line, elem = self.boundary()
                lp = self.mesh.tri_part[elem]
                cand = np.nonzero(lp == pi)[0]
                tol = 0.1 * t + 1e-6
                nn = self.mesh.nodes[line[cand]]
                ok = probe.near_box(nn.reshape(-1, 3), tol).reshape(-1, 3).all(axis=1)
                found = 0
                for k in cand[ok]:
                    good, nk = True, None
                    for v in line[k]:
                        ins, d, n = probe.query(self.mesh.nodes[v], tol)
                        if not ins or d > tol:
                            good = False
                            break
                        nk = n
                    if good:
                        edges.append(line[k])
                        enorm.append(nk)
                        eth.append(t)
                        sides.append(True)
                        found += 1
                if not found:
                    raise ShellError(f"Selection '{name}': the side face of part '{fs.part}' does not lie on the shell boundary (no mesh edge)")
        for es in sel.edges:
            if es.part not in self.sheets:
                raise ShellError(f"Selection '{name}': part '{es.part}' is not in the shell model")
            sh = self.sheets[es.part]
            hit = resolve_edge(es, [self.parts[es.part]], self._ecache.setdefault(es.part, {}))
            if hit is None:
                raise ShellError(f"Selection '{name}': the edge of part '{es.part}' was not found in the geometry")
            from ..geometry import list_edges, bbox
            edge = list_edges(self.parts[es.part].shape)[hit[1]]
            tol = (0.55 * sh.thickness if sh.kind != "surface" else 1e-3 * sh.thickness) + 1e-6
            lines, _, _ = self.mesh.edge_table()
            pi = self.pidx[es.part]
            inpart = np.zeros(len(self.mesh.nodes), bool)
            inpart[self.mesh.tri[self.mesh.tri_part == pi].ravel()] = True
            bb = bbox(edge)
            nn = self.mesh.nodes[lines]
            ok = (inpart[lines].all(axis=1)
                  & np.all((nn >= bb[:3] - tol) & (nn <= bb[3:] + tol), axis=(1, 2)))
            cand = np.nonzero(ok)[0]
            found = 0
            if len(cand):
                d = _edge_distance(edge, nn[cand].reshape(-1, 3)).reshape(-1, 3)
                for k in cand[(d <= tol).all(axis=1)]:
                    edges.append(lines[k])
                    enorm.append(np.zeros(3))
                    eth.append(sh.thickness)
                    sides.append(False)
                    found += 1
            if not found:
                raise ShellError(f"Selection '{name}': the edge of part '{es.part}' has no mesh edge in the shell")
        for ps in sel.points:
            if ps.part not in self.sheets:
                raise ShellError(f"Selection '{name}': part '{ps.part}' is not in the shell model")
            pi = self.pidx[ps.part]
            nodes = np.unique(self.mesh.tri[self.mesh.tri_part == pi])
            d = np.linalg.norm(self.mesh.nodes[nodes] - np.asarray(ps.point, float), axis=1)
            k = int(np.argmin(d))
            if d[k] > self.sheets[ps.part].thickness + 1e-6 and d[k] > 0.5 * float(
                    np.mean(np.linalg.norm(self.mesh.nodes[self.mesh.tri[:, 0]] - self.mesh.nodes[self.mesh.tri[:, 1]],
                                           axis=1))):
                raise ShellError(f"Selection '{name}': point {ps.point} is far from the shell mesh ({d[k]:.3g} mm)")
            pts.append(nodes[k])
        tg.elements = np.asarray(els, np.int64)
        tg.elem_sign = np.asarray(signs, float)
        tg.edges = np.asarray(edges, np.int64).reshape(-1, 3)
        tg.edge_normal = np.asarray(enorm, float).reshape(-1, 3)
        tg.edge_thick = np.asarray(eth, float)
        tg.points = np.asarray(pts, np.int64)
        tg.side_edges = np.asarray(sides, bool)
        if len(tg.elements):
            u, i = np.unique(tg.elements, return_index=True)
            tg.elements, tg.elem_sign = u, tg.elem_sign[i]
        return tg


# ------------------------------------------------------------------ constraints

def _axis_of(a: np.ndarray) -> int | None:
    """Index of the global axis parallel to a (|a| > 0), otherwise None."""
    k = int(np.argmax(np.abs(a)))
    return k if abs(a[k]) >= (1 - 1e-6) * np.linalg.norm(a) else None


def constraint_dofs(c, tg: Target, mesh, nnorm: np.ndarray):
    """Constraint → ({node: {DOF 1–6: value}}, {node: local axes (3, 3) – columns x', y', z'}).

    Local axes (*TRANSFORM, verified scratch/shell/exp3b.py – *EQUATION on DOF 4–6 is ignored by ccx): the only constrained
    component e on the side face whose rotation axis a = n × e is not a global axis → x' = e, y' = a, constraint of DOF 1 and 5.
    """
    kind = getattr(c, "kind", "")
    out: dict[int, dict[int, float]] = {}
    frames: dict[int, np.ndarray] = {}
    nodes = tg.nodes(mesh)
    if kind == "fixed":
        for v in nodes:
            out[int(v)] = {d: 0.0 for d in range(1, 7)}
        return out, frames
    if kind != "displacement":
        raise ShellError(f"Support of type '{kind}' is not supported in the shell model (only fixed support and displacement)")
    comps = [(k, float(v)) for k, v in enumerate((c.ux, c.uy, c.uz)) if v is not None]
    if not comps:
        return out, frames
    side_nodes = set(np.unique(tg.edges[tg.side_edges]).tolist()) if len(tg.edges) else set()
    for v in nodes:
        d = {k + 1: val for k, val in comps}
        if int(v) in side_nodes:
            if len(comps) == 3:
                d.update({4: 0.0, 5: 0.0, 6: 0.0})      # the whole side face constrained = clamping
            else:
                n = nnorm[v]
                axes = []
                for k, _ in comps:
                    e = np.zeros(3)
                    e[k] = 1.0
                    a = np.cross(n, e)
                    if np.linalg.norm(a) < 1e-6:
                        continue
                    axes.append((k, e, a / np.linalg.norm(a), _axis_of(a)))
                if all(ax is not None for *_, ax in axes):
                    for *_, ax in axes:
                        d[4 + ax] = 0.0
                elif len(comps) == 1:
                    k, e, a, _ = axes[0]
                    frames[int(v)] = np.column_stack([e, a, np.cross(e, a)])
                    d = {1: comps[0][1], 5: 0.0}
                else:
                    raise ShellError(
                        f"Support '{c.selection}': two displacement components on the edge face of an inclined shell would constrain rotation in a general direction – not supported in the shell model (use a fixed support, all three components or a single component)")
        out[int(v)] = d
    return out, frames


# ------------------------------------------------------------------ loads

def _weights(tg: Target, mesh) -> tuple[np.ndarray, np.ndarray, float]:
    """Nodal weights of the uniform distribution (∫N dA of the elements, t·∫N dL of the edges) and the total "area" of the selection."""
    W = np.zeros(len(mesh.nodes))
    tot = 0.0
    if len(tg.elements):
        nint, area = tri6_integrals(mesh.nodes, mesh.tri[tg.elements])
        W += np.bincount(mesh.tri[tg.elements].ravel(), nint.ravel(), minlength=len(W))
        tot += float(area.sum())
    if len(tg.edges):
        _, dl, _, N = line3_gauss(mesh.nodes, tg.edges)
        w = (dl * tg.edge_thick[:, None]) @ N        # (K, 3)
        W += np.bincount(tg.edges.ravel(), w.ravel(), minlength=len(W))
        tot += float(w.sum())
    return W, tot, float(len(tg.points))


def force_loads(load, tg: Target, mesh) -> np.ndarray:
    F = np.zeros((len(mesh.nodes), 3))
    vec = np.asarray(load.vector, float)
    if getattr(load, "csys", None):
        raise ShellError("A force in a local coordinate system is not supported in the shell model")
    if len(tg.points) and (len(tg.elements) or len(tg.edges)):
        raise ShellError(f"Force '{load.selection}': the selection mixes points with faces/edges")
    if len(tg.points):
        for v in tg.points:
            F[v] += vec / len(tg.points)
        return F
    if len(tg.edges) and not tg.side_edges.all():
        # edges of the 3D body: along the length (without the thickness)
        _, dl, _, N = line3_gauss(mesh.nodes, tg.edges)
        w = dl @ N
        W = np.bincount(tg.edges.ravel(), w.ravel(), minlength=len(mesh.nodes))
        return W[:, None] * vec[None, :] / float(w.sum())
    W, tot, _ = _weights(tg, mesh)
    if tot <= 0:
        raise ShellError(f"Force '{load.selection}': the selection has zero measure")
    return W[:, None] * vec[None, :] / tot


def pressure_value(load, X: np.ndarray) -> np.ndarray:
    p = np.full(X.shape[:-1], float(load.value))
    g = getattr(load, "gradient", None)
    if g is not None and any(float(v) != 0.0 for v in g):
        o = np.asarray(load.origin if load.origin is not None else [0, 0, 0], float)
        lin = (X - o) @ np.asarray(g, float)
        if getattr(load, "clip", False):
            lin = np.maximum(lin, 0.0)
        p = p + lin
    return p


def pressure_loads(load, tg: Target, mesh) -> np.ndarray:
    """Pressure (positive = into the material, against the outward normal of the face): on the skin −p·n_out by the area of the elements, on the side face
    −p·t·n_side along the length of the edge.
    """
    F = np.zeros((len(mesh.nodes), 3))
    if len(tg.points) or (len(tg.edges) and not tg.side_edges.all()):
        raise ShellError(f"Pressure '{load.selection}': faces only")
    if len(tg.elements):
        X, dA, N = tri6_gauss(mesh.nodes, mesh.tri[tg.elements])
        p = pressure_value(load, X)                            # (K, 6)
        f = -(p * tg.elem_sign[:, None])[..., None] * dA        # n_ven = sign · n_prvku
        F += scatter(len(mesh.nodes), mesh.tri[tg.elements], np.einsum("mgc,gk->mkc", f, N))
    if len(tg.edges):
        X, dl, T, N = line3_gauss(mesh.nodes, tg.edges)
        p = pressure_value(load, X)                            # (K, 3)
        n = tg.edge_normal[:, None, :]                          # outward normal of the side face (planar at the node)
        f = -(p * dl * tg.edge_thick[:, None])[..., None] * n
        F += scatter(len(mesh.nodes), tg.edges, np.einsum("mgc,gk->mkc", f, N))
    return F


def body_loads(mesh, rho_t: np.ndarray, accel) -> np.ndarray:
    """Body load: Σ ρ·t·∫N a(x) dA; accel(X (M, 6, 3)) → acceleration (M, 6, 3) [mm/s²]."""
    X, dA, N = tri6_gauss(mesh.nodes, mesh.tri)
    a = np.linalg.norm(dA, axis=2)                             # (M, 6)
    f = accel(X) * (a * rho_t[:, None])[..., None]
    return scatter(len(mesh.nodes), mesh.tri, np.einsum("mgc,gk->mkc", f, N))


def gravity_loads(load, mesh, rho_t: np.ndarray) -> np.ndarray:
    if getattr(load, "csys", None):
        raise ShellError("Gravity in a local coordinate system is not supported in the shell model")
    g = np.asarray(load.vector, float)
    return body_loads(mesh, rho_t, lambda X: np.broadcast_to(g, X.shape))


def rotation_loads(load, mesh, rho_t: np.ndarray) -> np.ndarray:
    p = np.asarray(load.point, float)
    a = np.asarray(load.axis, float)
    a = a / np.linalg.norm(a)
    w2, al = float(load.omega) ** 2, float(load.alpha)
    mask = np.ones(len(mesh.tri), bool)
    if load.parts:
        idx = [mesh.part_names.index(n) for n in load.parts if n in mesh.part_names]
        mask = np.isin(mesh.tri_part, idx)

    def acc(X):
        r = X - p
        rp = r - (r @ a)[..., None] * a
        return w2 * rp - al * np.cross(a, r)
    return body_loads(mesh, rho_t * mask, acc)
