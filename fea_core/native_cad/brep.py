"""Stream of the ShapeManager (ASM BinaryFile4/8 from .f3d and .ipt, ACIS SAB) → exact B-rep OCC (OCP).

Supported: plane, cone (cylinder and cone), sphere, torus, spline-surface with a stored NURBS cache (direct or via
`ref N`); straight, ellipse (circle / ellipse), intcurve with a NURBS cache; tolerant tedge / tvertex. Lengths in the stream are
in cm → the result ×10 (mm). Curves on faces (pcurves) are computed by ShapeFix. Only the solved model (records before
`Begin-of-ASM-History-Data`). Verified against the STEP export from Fusion: volume 1e-14 (planes, cylinders, cones, NURBS).
Not supported (the body is reported as unconverted): procedural surfaces without a stored approximation (helices of threads),
elliptic cylinder / cone, subshell. The input is untrusted – nothing is run, errors are reported by an exception.
"""
from __future__ import annotations

import math
from collections import Counter

from . import sab
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID, TopAbs_SHELL
from OCP.TopExp import TopExp
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedMapOfShape
from OCP.TopoDS import TopoDS_Compound, TopoDS_Face, TopoDS_Shell, TopoDS_Solid, TopoDS_Vertex, TopoDS_Wire
from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Lin, gp_Pnt, gp_Trsf, gp_Vec, gp_Mat, gp_XYZ
from OCP.collections import Array1_gp_Pnt, Array2_gp_Pnt, Array1_double, Array1_int, Array2_double
from OCP.Geom import (Geom_BSplineCurve, Geom_BSplineSurface, Geom_Circle, Geom_ConicalSurface,
                      Geom_CylindricalSurface, Geom_Ellipse, Geom_Line, Geom_Plane, Geom_SphericalSurface,
                      Geom_ToroidalSurface)



def _once(seen: set, k: int):
    """Linked list from the file: a repeated record = a cycle (a forged file) → an error instead of an endless loop."""
    if k in seen:
        raise Unsupported(f"cycle in the record list at {k}")
    seen.add(k)

class Unsupported(Exception):
    pass


def P(v):
    return gp_Pnt(*v)


def D(v):
    return gp_Dir(*v)


def head(name):
    return name.split("-")[-1]


def kind_of(name):
    return name.split("-")[0]


# ---------------------------------------------------------------------------------------------- NURBS
class Cur:
    def __init__(self, toks, i):
        self.t, self.i = toks, i

    def take(self):
        x = self.t[self.i]
        self.i += 1
        return x

    def peek(self):
        return self.t[self.i]

    def num(self):
        k, v = self.take()
        if k not in ("double", "long", "enum", "float", "short", "char", "int64"):
            raise Unsupported(f"number expected, found {k}={v!r}")
        return v


def _arr1(vals, cls):
    a = cls(1, len(vals))
    for k, x in enumerate(vals):
        a.SetValue(k + 1, x)
    return a


def read_bs_curve(cur: Cur, dim: int = 3):
    _, kind = cur.take()
    if kind in ("nullbs", "null_curve"):
        return None
    if kind not in ("nubs", "nurbs"):
        raise Unsupported(f"curve block {kind}")
    deg = int(cur.num())
    closure = int(cur.num())  # 0 open, 1 closed, 2 periodic
    n = int(cur.num())
    kn, mu = [], []
    for _ in range(n):
        kn.append(cur.num())
        mu.append(int(cur.num()))
    rational = kind == "nurbs"
    npol = sum(mu) - (deg - 1)
    pts, w = [], []
    for _ in range(npol):
        c = [cur.num() for _ in range(dim)]
        pts.append(c)
        w.append(cur.num() if rational else 1.0)
    mu[0] += 1
    mu[-1] += 1
    return dict(deg=deg, knots=kn, mults=mu, poles=pts, weights=w, rational=rational, closure=closure)


def bs_curve_geom(b):
    poles = Array1_gp_Pnt(1, len(b["poles"]))
    for k, p in enumerate(b["poles"]):
        poles.SetValue(k + 1, gp_Pnt(*p))
    K, M = _arr1(b["knots"], Array1_double), _arr1(b["mults"], Array1_int)
    if b["rational"]:
        C = Geom_BSplineCurve(poles, _arr1(b["weights"], Array1_double), K, M, b["deg"], False)
    else:
        C = Geom_BSplineCurve(poles, K, M, b["deg"], False)
    if b["closure"] == 2 and C.IsClosed():
        try:
            C.SetPeriodic()  # periodic curve of ASM: the parameters of the edges may lie outside the clamped domain of definition
        except Exception:  # noqa: BLE001
            pass
    return C


def read_bs_surface(cur: Cur):
    _, kind = cur.take()
    if kind != "nubs" and kind != "nurbs":
        raise Unsupported(f"face block {kind}")
    du, dv = int(cur.num()), int(cur.num())
    if cur.peek()[0] in ("ident", "subident"):
        cur.take()  # both | u | v | none
    for _ in range(4):
        cur.num()  # closedness u, v; singularity u, v
    nu, nv = int(cur.num()), int(cur.num())
    ku, mu, kv, mv = [], [], [], []
    for _ in range(nu):
        ku.append(cur.num())
        mu.append(int(cur.num()))
    for _ in range(nv):
        kv.append(cur.num())
        mv.append(int(cur.num()))
    rational = kind == "nurbs"
    us, vs = sum(mu) - (du - 1), sum(mv) - (dv - 1)
    if us < 1 or vs < 1 or us * vs > 4_000_000:      # a forged header would allocate tens of GB
        raise Unsupported(f"nurbs: {us} x {vs} control points")
    Pp = Array2_gp_Pnt(1, us, 1, vs)
    W = Array2_double(1, us, 1, vs)
    for j in range(vs):          # v outer loop, u inner
        for i in range(us):
            x, y, z = cur.num(), cur.num(), cur.num()
            Pp.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))
            W.SetValue(i + 1, j + 1, cur.num() if rational else 1.0)
    mu[0] += 1
    mu[-1] += 1
    mv[0] += 1
    mv[-1] += 1
    KU, KV, MU, MV = (_arr1(ku, Array1_double), _arr1(kv, Array1_double), _arr1(mu, Array1_int),
                      _arr1(mv, Array1_int))
    if rational:
        return Geom_BSplineSurface(Pp, W, KU, KV, MU, MV, du, dv, False, False)
    return Geom_BSplineSurface(Pp, KU, KV, MU, MV, du, dv, False, False)


# ---------------------------------------------------------------------------------------------- model
class Model:
    def __init__(self, data: bytes):
        self.h, recs, self.info = sab.frame(data, stop_at_history=True)   # history is not read (and not tokenized)
        if self.info["error"]:
            raise Unsupported("framing: " + self.info["error"])
        hi = next((r.index for r in recs if r.name.startswith("Begin-of-")), None)
        self.recs = recs
        self.solved = recs if hi is None else recs[:hi]
        # table of subtypes: each `{` followed by a name ≠ ref in the order of the stream
        self.subtypes = []
        for r in recs:
            t = r.tokens
            for i in range(len(t) - 1):
                if t[i][0] == "open" and t[i + 1][0] in ("ident", "subident") and t[i + 1][1] != "ref":
                    self.subtypes.append((r.index, i))
        self.B = BRep_Builder()
        self.cache = {}
        self.stats = Counter()
        self.unsupported = Counter()

    def rec(self, k):
        if k < 0 or k >= len(self.solved):
            raise Unsupported(f"reference outside the solved model ${k}")
        return self.solved[k]

    # -- podtypy
    def subtype_span(self, toks, i):
        """toks[i] == open; returns (toks, i) of the definition (follows `ref N`)."""
        if toks[i + 1] == ("ident", "ref"):
            n = toks[i + 2][1]
            ri, ti = self.subtypes[n]
            return self.subtype_span(self.recs[ri].tokens, ti)
        return toks, i

    def cache_block(self, toks, i, surface):
        """In a subtype starting with toks[i] == open it finds the NURBS cache (block nubs/nurbs at depth 0 of the subtype)."""
        toks, i = self.subtype_span(toks, i)
        name = toks[i + 1][1]
        depth, j = 0, i + 2
        blocks = []
        while j < len(toks):
            k, v = toks[j]
            if k == "open":
                depth += 1
            elif k == "close":
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and k in ("ident", "subident") and v in ("nubs", "nurbs"):
                blocks.append(j)
            j += 1
        if not blocks:
            raise Unsupported(f"{name}: no NURBS cache")
        revision_first = toks[i + 2][0] == "long" and toks[i + 3] == ("enum", 0)
        order = blocks if revision_first else list(reversed(blocks))
        last_err = None
        for b in order:
            try:
                if surface:
                    return read_bs_surface(Cur(toks, b)), name
                bc = read_bs_curve(Cur(toks, b), 3)
                # a 3D block must be followed by a tolerance (double) – a 2D block would be read shifted
                return bs_curve_geom(bc), name
            except (Unsupported, IndexError, TypeError, Exception) as e:  # try the next block
                last_err = e
        raise Unsupported(f"{name}: cache unreadable ({last_err})")

    # -- geometrie
    def surface(self, k):
        key = ("S", k)
        if key in self.cache:
            return self.cache[key]
        r = self.rec(k)
        t = r.tokens
        kind = kind_of(r.name)
        v = [x[1] for x in t]
        if kind == "plane":
            S, rev = Geom_Plane(gp_Ax3(P(v[3]), D(v[4]), D(v[5]))), False
        elif kind == "cone":
            origin, axis, major, ratio = v[3], v[4], v[5], v[6]
            s, c = v[9], v[10]
            R = math.sqrt(sum(x * x for x in major))
            if abs(abs(ratio) - 1.0) > 1e-9:
                raise Unsupported("elliptic cone/cylinder")
            ax = gp_Ax3(P(origin), D(axis), D(major))
            if abs(s) < 1e-12:
                S = Geom_CylindricalSurface(ax, R)
            else:
                S = Geom_ConicalSurface(ax, math.atan(s / c), R)
            rev = c < 0
        elif kind == "sphere":
            center, rad, d1, d2 = v[3], v[4], v[5], v[6]
            S, rev = Geom_SphericalSurface(gp_Ax3(P(center), D(d2), D(d1)), abs(rad)), rad < 0
        elif kind == "torus":
            origin, axis, Rm, rm, ref = v[3], v[4], v[5], v[6], v[7]
            S, rev = Geom_ToroidalSurface(gp_Ax3(P(origin), D(axis), D(ref)), abs(Rm), abs(rm)), rm < 0
        elif kind == "spline":
            rev = t[3][0] == "true"
            S, sub = self.cache_block(t, 4, surface=True)
            self.stats["spline:" + sub] += 1
        else:
            raise Unsupported(f"face {r.name}")
        self.stats["surface:" + kind] += 1
        self.cache[key] = (S, rev)
        return S, rev

    def curve(self, k):
        key = ("C", k)
        if key in self.cache:
            return self.cache[key]
        r = self.rec(k)
        t = r.tokens
        v = [x[1] for x in t]
        kind = kind_of(r.name)
        rev = False
        if kind == "straight":
            C = Geom_Line(gp_Lin(P(v[3]), D(v[4])))
        elif kind == "ellipse":
            center, normal, major, ratio = v[3], v[4], v[5], v[6]
            R = math.sqrt(sum(x * x for x in major))
            n = normal if ratio > 0 else tuple(-x for x in normal)
            ax = gp_Ax2(P(center), D(n), D(major))
            C = Geom_Circle(ax, R) if abs(abs(ratio) - 1) < 1e-12 else Geom_Ellipse(ax, R, R * abs(ratio))
        elif kind == "intcurve":
            rev = t[3][0] == "true"
            C, sub = self.cache_block(t, 4, surface=False)
            self.stats["intcurve:" + sub] += 1
        else:
            raise Unsupported(f"curve {r.name}")
        self.stats["curve:" + kind] += 1
        self.cache[key] = (C, rev)
        return C, rev

    # -- topologie
    def vertex(self, k):
        key = ("V", k)
        if key in self.cache:
            return self.cache[key]
        r = self.rec(k)
        pt = self.rec(r.tokens[5][1]).tokens[3][1]
        tol = 1e-6
        if r.name.startswith("tvertex"):
            tols = [x[1] for x in r.tokens[6:9] if x[0] == "double" and x[1] > 0]
            if tols:
                tol = max(tol, max(tols))
        V = TopoDS_Vertex()
        self.B.MakeVertex(V, P(pt), tol)
        self.cache[key] = (V, pt)
        return V, pt

    def edge(self, k):
        key = ("E", k)
        if key in self.cache:
            return self.cache[key]
        r = self.rec(k)
        t = r.tokens
        v0, ts, v1, te, crv, sense = t[3][1], t[4][1], t[5][1], t[6][1], t[8][1], t[9][0] == "true"
        if crv < 0:
            self.cache[key] = None  # point edge (apex of a cone, pole) – completed by ShapeFix
            self.stats["edge:degenerate"] += 1
            return None
        C, crev = self.curve(crv)
        V0, p0 = self.vertex(v0)
        V1, p1 = self.vertex(v1)
        rev = sense ^ crev
        # 1) parameters of the edge directly from ASM: E(t) = C(t), resp. C(-t) in the reverse sense
        a, b = (ts, te) if not rev else (-te, -ts)
        Va, Vb = (V0, V1) if not rev else (V1, V0)
        pa_, pb_ = (p0, p1) if not rev else (p1, p0)
        if isinstance(C, Geom_Line):  # an OCC line has unit speed, an ASM vector does not
            L = math.sqrt(sum(x * x for x in self.rec(crv).tokens[4][1]))
            a, b = a * L, b * L
        if v0 == v1 and abs(b - a) < 1e-12 and C.IsPeriodic():
            b = a + C.Period()
        tol = max(BRep_Tool.Tolerance_s(V0), BRep_Tool.Tolerance_s(V1), 1e-6) * 50
        ok = (C.Value(a).Distance(P(pa_)) <= tol and C.Value(b).Distance(P(pb_)) <= tol)
        if not ok:
            # 2) backup: projection of the vertices onto the curve
            self.stats["edge:param_by_projection"] += 1
            pr0, pr1 = GeomAPI_ProjectPointOnCurve(P(p0), C), GeomAPI_ProjectPointOnCurve(P(p1), C)
            if pr0.NbPoints() == 0 or pr1.NbPoints() == 0:
                raise Unsupported(f"edge #{k}: vertex does not lie on the curve")
            qa, qb = pr0.LowerDistanceParameter(), pr1.LowerDistanceParameter()
            if C.IsPeriodic():
                a, b, Va, Vb = (qa, qb, V0, V1) if not rev else (qb, qa, V1, V0)
                while b <= a + 1e-12:
                    b += C.Period()
            else:
                rev = qa > qb
                a, b, Va, Vb = (qa, qb, V0, V1) if not rev else (qb, qa, V1, V0)
        mk = BRepBuilderAPI_MakeEdge(C, Va, Vb, a, b)
        if not mk.IsDone():
            # tolerant model: enlarge the tolerance of the vertex to the real deviation (max 10 µm in cm = 1e-3)
            pva, pvb = BRep_Tool.Pnt_s(Va), BRep_Tool.Pnt_s(Vb)
            da, db = C.Value(a).Distance(pva), C.Value(b).Distance(pvb)
            if max(da, db) > 1e-3:
                raise Unsupported(f"edge #{k}: error {mk.Error()} (deviation {max(da, db):.3g} cm)")
            self.B.UpdateVertex(Va, max(BRep_Tool.Tolerance_s(Va), da * 1.5))
            self.B.UpdateVertex(Vb, max(BRep_Tool.Tolerance_s(Vb), db * 1.5))
            self.stats["vertex:tolerance_widened"] += 1
            mk = BRepBuilderAPI_MakeEdge(C, Va, Vb, a, b)
            if not mk.IsDone():
                raise Unsupported(f"edge #{k}: error {mk.Error()} (a={a}, b={b})")
        E = mk.Edge()
        if rev:
            E.Reverse()
        if r.name.startswith("tedge") and len(t) > 11 and t[11][0] == "double":
            self.B.UpdateEdge(E, max(1e-6, t[11][1]))
        self.stats["edge"] += 1
        self.cache[key] = E
        return E

    def face(self, k):
        r = self.rec(k)
        t = r.tokens
        loop, srf, fsense = t[4][1], t[7][1], t[8][0] == "true"
        if len(t) > 9 and t[9][0] == "true":
            self.stats["face:double_sided"] += 1
        S, srev = self.surface(srf)
        frev = fsense ^ srev
        F = TopoDS_Face()
        self.B.MakeFace(F, S, 1e-6)
        seen_l = set()
        while loop >= 0:
            _once(seen_l, loop)
            lr = self.rec(loop)
            c0 = lr.tokens[4][1]
            W = TopoDS_Wire()
            self.B.MakeWire(W)
            c = c0
            guard = 0
            while True:
                cr = self.rec(c)
                E = self.edge(cr.tokens[6][1])
                if E is not None:
                    flip = (cr.tokens[7][0] == "true") ^ frev
                    self.B.Add(W, E.Reversed() if flip else E)
                c = cr.tokens[3][1]
                guard += 1
                if c == c0 or c < 0 or guard > 100000:
                    break
            self.B.Add(F, W)
            loop = lr.tokens[3][1]
        self.stats["face"] += 1
        return F.Reversed() if frev else F

    def transform(self, k):
        if k < 0:
            return None
        v = [x[1] for x in self.rec(k).tokens]
        c0, c1, c2, tr, sc = v[2], v[3], v[4], v[5], v[6]
        T = gp_Trsf()
        # column mapping according to asm.md §5.2: a[0..2] → column 0 …
        T.SetValues(c0[0], c1[0], c2[0], tr[0],
                    c0[1], c1[1], c2[1], tr[1],
                    c0[2], c1[2], c2[2], tr[2])
        if abs(sc - 1.0) > 1e-12:
            self.stats["transform:scaled"] += 1
        return T

    def bodies(self):
        """List (index of the body, key of the body, TopoDS shape or error)."""
        out = []
        top = self.h.entities  # records 1 .. entities−1 = stored entities of the top level
        for r in self.solved[1:top]:
            if head(r.name) != "body":
                self.stats["top:" + r.name] += 1
                continue
            key = r.tokens[1][1]
            try:
                comp = TopoDS_Compound()
                self.B.MakeCompound(comp)
                lump = r.tokens[3][1]
                nsol = 0
                seen_k = set()
                while lump >= 0:
                    _once(seen_k, lump)
                    lr = self.rec(lump)
                    sh = lr.tokens[4][1]
                    sol = TopoDS_Solid()
                    self.B.MakeSolid(sol)
                    while sh >= 0:
                        _once(seen_k, sh)
                        shr = self.rec(sh)
                        if shr.tokens[4][1] >= 0:
                            raise Unsupported("subshell")
                        shell = TopoDS_Shell()
                        self.B.MakeShell(shell)
                        f = shr.tokens[5][1]
                        while f >= 0:
                            _once(seen_k, f)
                            self.B.Add(shell, self.face(f))
                            f = self.rec(f).tokens[3][1]
                        self.B.Add(sol, shell)
                        sh = shr.tokens[3][1]
                    self.B.Add(comp, sol)
                    nsol += 1
                    lump = lr.tokens[3][1]
                T = self.transform(r.tokens[5][1])
                shape = comp
                if T is not None and T.Form() != 0:  # gp_Identity = 0
                    shape = BRepBuilderAPI_Transform(comp, T, True).Shape()
                out.append((r.index, key, shape, None))
            except (Unsupported, Exception) as e:  # noqa: BLE001
                self.unsupported[str(e)[:80]] += 1
                out.append((r.index, key, None, f"{type(e).__name__}: {e}"))
        return out


def count(shape, kind):
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m.Extent()


def props(shape):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    a = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, a)
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, False)
    lo, hi = box.CornerMin(), box.CornerMax()  # Bnd_Box.Get() in OCP 8 cannot be converted
    x0, y0, z0, x1, y1, z1 = lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z()
    return dict(volume_mm3=g.Mass(), area_mm2=a.Mass(), bbox_mm=[round(x, 4) for x in (x0, y0, z0, x1, y1, z1)],
                solids=count(shape, TopAbs_SOLID), shells=count(shape, TopAbs_SHELL), faces=count(shape, TopAbs_FACE),
                edges=count(shape, TopAbs_EDGE), vertices=count(shape, TopAbs_VERTEX),
                valid=BRepCheck_Analyzer(shape).IsValid())


def convert(data: bytes):
    m = Model(data)
    res = dict(header=dict(magic=m.h.magic, version=m.h.version, product=m.h.product, kernel=m.h.kernel,
                           flags=m.h.flags, entities=m.h.entities), bodies=[])
    comp = TopoDS_Compound()
    m.B.MakeCompound(comp)
    ok = 0
    for idx, key, shape, err in m.bodies():
        item = dict(record=idx, asm_body_key=key, error=err)
        if shape is not None:
            raw_valid = BRepCheck_Analyzer(shape).IsValid()
            fix = ShapeFix_Shape(shape)
            fix.Perform()
            fixed = fix.Shape()
            T = gp_Trsf()
            T.SetScale(gp_Pnt(0, 0, 0), 10.0)  # cm → mm
            mm = BRepBuilderAPI_Transform(fixed, T, True).Shape()
            item.update(raw_valid_before_shapefix=raw_valid, **props(mm))
            m.B.Add(comp, mm)
            ok += 1
        res["bodies"].append(item)
    res["stats"] = dict(m.stats)
    res["unsupported"] = dict(m.unsupported)
    res["total"] = props(comp) if ok else None
    return res, comp


def bodies(data: bytes) -> list[tuple[int, list, str | None]]:
    """Bodies of the stream in the order of the stream: [(key of the body, [TopoDS_Solid in mm], error or None)]."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    m = Model(data)
    out = []
    for idx, key, shape, err in m.bodies():
        if shape is None:
            out.append((key, [], err or f"body #{idx}"))
            continue
        fix = ShapeFix_Shape(shape)
        fix.Perform()
        T = gp_Trsf()
        T.SetScale(gp_Pnt(0, 0, 0), 10.0)                     # cm → mm
        mm = BRepBuilderAPI_Transform(fix.Shape(), T, True).Shape()
        ex = TopExp_Explorer(mm, TopAbs_SOLID)
        sol = []
        while ex.More():
            sol.append(TopoDS.Solid(ex.Current()))
            ex.Next()
        out.append((key, sol, None))
    return out


def solids(data: bytes) -> tuple[list, list[str]]:
    """Bodies of the stream in mm: ([(key of the body, index of the body in the body, TopoDS_Solid)], [errors of the unconverted bodies])."""
    out, errors = [], []
    for key, sol, err in bodies(data):
        if err:
            errors.append(err)
        out += [(key, k, s) for k, s in enumerate(sol)]
    return out, errors
