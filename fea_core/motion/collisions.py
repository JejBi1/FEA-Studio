"""Check of the collisions without contact (specification 3.4): bodies in the MBD pass through each other silently → at the output times the smallest
distance of the shapes of OCP (BRepExtrema_DistShapeShape) on the shapes moved to the position of the body, the first collision refined by
bisection of the interval (the position between the outputs: the centroid by the Hermite cubic from the position and the velocity, the rotation along the geodesic
(Slerp)).

check_collisions(study, parts, res, pairs, tol) → {t, pairs: [{a, b, distance (N), min_distance, t_min, first}],
first: {a, b, t} | None, tol}. pairs = [[body, body], …] | "all" (all pairs of bodies not connected by a joint). Distance ≤ tol + 1e-6 [mm] = collision (the shapes touch or penetrate – OCP then returns 0).
"""
from __future__ import annotations

import numpy as np


def _compound(shapes):
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    if len(shapes) == 1:
        return shapes[0]
    c = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


def _moved(shape, R, T):
    from OCP.gp import gp_Trsf
    from OCP.TopLoc import TopLoc_Location
    tr = gp_Trsf()
    tr.SetValues(*R[0], T[0], *R[1], T[1], *R[2], T[2])
    return shape.Moved(TopLoc_Location(tr))


def _distance(s1, s2) -> float:
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    d = BRepExtrema_DistShapeShape(s1, s2)
    if not d.IsDone():
        return float("nan")
    return float(d.Value())


class _Pose:
    """Position of the body at an arbitrary time between the outputs."""

    def __init__(self, res, i):
        from scipy.spatial.transform import Rotation, Slerp
        a = res.arrays
        self.t = a["t"]
        self.c = a["pos"][:, i]
        self.v = a["vel"][:, i]
        self.c0 = np.asarray(res.meta["bodies"][i]["com"], float)
        self.R = a["rot"][:, i]
        self.ground = res.meta["bodies"][i]["ground"]
        if not self.ground and len(self.t) > 1:
            self.slerp = Slerp(self.t, Rotation.from_matrix(self.R))

    def at(self, t: float):
        if self.ground:
            return np.eye(3), np.zeros(3)
        k = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        t0, t1 = self.t[k], self.t[k + 1]
        h = t1 - t0
        s = (t - t0) / h if h > 0 else 0.0
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        c = h00 * self.c[k] + h10 * h * self.v[k] + h01 * self.c[k + 1] + h11 * h * self.v[k + 1]
        R = self.slerp([t]).as_matrix()[0]
        return R, c - R @ self.c0


def check_collisions(study, parts, res, pairs, tol: float = 0.0, refine: bool = True) -> dict:
    mech = study.mechanism
    by_name = {p.name: p for p in parts}
    names = res.body_names
    shapes = {}
    for b in mech.bodies:
        sh = [by_name[p].shape for p in b.parts if p in by_name]
        if sh:
            shapes[b.name] = _compound(sh)
    if pairs == "all" or pairs is True:
        linked = {frozenset((j.body_a or "", j.body_b or "")) for j in mech.joints}
        ground = [b.name for b in mech.bodies if b.ground]
        for j in mech.joints:                       # joint to the frame "" = to all the bodies of the frame
            for g in ground:
                if not j.body_a:
                    linked.add(frozenset((g, j.body_b)))
                if not j.body_b:
                    linked.add(frozenset((j.body_a, g)))
        cand = [(a, b) for k, a in enumerate(names) for b in names[k + 1:]
                if frozenset((a, b)) not in linked and not (mech.body(a).ground and mech.body(b).ground)]
    else:
        cand = [tuple(p) for p in pairs]
    t = res.arrays["t"]
    out = {"t": t.tolist(), "pairs": [], "first": None, "tol": float(tol)}
    tol_eff = float(tol) + 1e-6                     # touch: OCP returns a rounding (1e-15 mm) instead of 0
    for a, b in cand:
        if a not in shapes or b not in shapes:
            out.setdefault("skipped", []).append([a, b])
            continue
        ia, ib = names.index(a), names.index(b)
        pa, pb = _Pose(res, ia), _Pose(res, ib)

        def dist(tt):
            Ra, Ta = pa.at(tt)
            Rb, Tb = pb.at(tt)
            # a body entirely inside another: OCP returns 0 (InnerSolution)
            return _distance(_moved(shapes[a], Ra, Ta), _moved(shapes[b], Rb, Tb))

        d = np.array([dist(float(tt)) for tt in t])
        hit = np.nonzero(d <= tol_eff)[0]
        first = None
        if len(hit):
            k = int(hit[0])
            tc = float(t[k])
            if k > 0 and refine:                    # bisection between the last free and the first colliding output
                lo, hi = float(t[k - 1]), float(t[k])
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if dist(mid) <= tol_eff:
                        hi = mid
                    else:
                        lo = mid
                    if hi - lo < 1e-9 * max(t[-1], 1.0):
                        break
                tc = hi
            first = tc
        km = int(np.nanargmin(d)) if np.isfinite(d).any() else 0
        rec = {"a": a, "b": b, "distance": d.tolist(), "min_distance": float(np.nanmin(d)) if np.isfinite(d).any()
               else None, "t_min": float(t[km]), "first": first}
        out["pairs"].append(rec)
        if first is not None and (out["first"] is None or first < out["first"]["t"]):
            out["first"] = {"a": a, "b": b, "t": first}
    return out
