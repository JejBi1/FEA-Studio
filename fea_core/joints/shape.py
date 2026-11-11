"""Recognition of bolts, nuts and washers by SHAPE (parts without a standard in the name).

The body is decomposed into faces with respect to one axis (axis = the largest cylindrical face):
  - all cylinders and cones must be coaxial (a transverse hole, a key slot … = not a fastener),
  - planar faces only perpendicular or parallel to the axis; parallel ones outside at the same distance ρ with normals
    every 60° = a hexagon with a key width s = 2ρ (inside = the inner hexagon of the head),
  - the course of the radius along the axis (nodes of the triangulation) → head = section with r > 1.15 r_shank at one end.
Bolt: shank (outer cylinder, Ø = d, or d2/d3 for a modelled thread) + head according to ISO tables
(hexagon s, cylindrical dk, countersunk cone 90° dk ≈ 2d), no hole on the axis, volume ≈ head + shank.
Nut: outer hexagon (s according to d) over the whole height + coaxial hole Ø d … D1, height 0.3–1.5 d.
Washer: annulus d1/d2 according to ISO 7089/7093 (h ≤ 0.25 d2).
A cylinder without a head (pin, magnet, rod), a shaft with a collar in the middle, a bushing → None.
The result is only a PROPOSAL with a certainty and a reason – a bolt is created only after confirmation (the joint must also have
clamped parts with a coaxial hole, which detect._bolt_from_part checks).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus

from ..geometry import Part, bbox, face_props
from . import tables
from .naming import FastenerName

MAX_FACES = 80                     # fastener with few faces (frames, plates are skipped quickly)
STANDARD_BY_SHAPE = {("bolt", "hex"): "ISO 4017", ("bolt", "socket"): "ISO 4762",
                     ("bolt", "countersunk"): "ISO 10642", ("nut", ""): "ISO 4032", ("washer", ""): "ISO 7089",
                     ("washer_large", ""): "ISO 7093"}
HEAD_LABEL = {"hex": "hexagon head", "socket": "cylindrical head", "countersunk": "countersunk head"}


@dataclass
class ShapeGuess:
    part: str
    kind: str                       # bolt | nut | washer | washer_large
    d: float
    head: str = ""                  # hex | socket | countersunk (bolt only)
    length: float | None = None
    confidence: float = 0.0         # 0–1
    reason: str = ""

    @property
    def standard(self) -> str:
        return STANDARD_BY_SHAPE.get((self.kind, self.head if self.kind == "bolt" else ""), "")

    def fastener_name(self) -> FastenerName:
        return FastenerName(self.kind, self.standard, self.d, None, self.length, None, self.head or "hex",
                            source=f"shape: {self.reason}")

    def describe(self) -> str:
        lbl = {"bolt": "bolt", "nut": "nut", "washer": "washer", "washer_large": "washer (large)"}[self.kind]
        s = f"{lbl} M{self.d:g}" + (f"×{self.length:g}" if self.length else "")
        return s + (f" ({HEAD_LABEL[self.head]})" if self.head in HEAD_LABEL else "")


# ===================================================================== rozklad ploch

def _surface_axis(sf) -> tuple[np.ndarray, np.ndarray] | None:
    t = sf.GetType()
    if t == GeomAbs_Cylinder:
        ax = sf.Cylinder().Axis()
    elif t == GeomAbs_Cone:
        ax = sf.Cone().Axis()
    elif t == GeomAbs_Torus:
        ax = sf.Torus().Axis()
    else:
        return None
    p, d = ax.Location(), ax.Direction()
    return np.array([p.X(), p.Y(), p.Z()]), np.array([d.X(), d.Y(), d.Z()])


def _faces(part: Part) -> list[dict] | None:
    from .detect import _tri_nodes
    faces = part.faces()
    if not faces or len(faces) > MAX_FACES:
        return None
    out = []
    for f in faces:
        sf = BRepAdaptor_Surface(f)
        t = sf.GetType()
        pr = face_props(f)
        rec = {"face": f, "type": {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
                                   GeomAbs_Torus: "torus", GeomAbs_Sphere: "sphere"}.get(t, "other"),
               "area": pr["area"], "point": np.asarray(pr["point"], float), "normal": np.asarray(pr["normal"], float),
               "nodes": _tri_nodes(f)}
        ax = _surface_axis(sf)
        if ax is not None:
            rec["axis"] = ax
        if t == GeomAbs_Cylinder:
            rec["r"] = sf.Cylinder().Radius()
        elif t == GeomAbs_Cone:
            rec["semi"] = abs(sf.Cone().SemiAngle())
        out.append(rec)
    return out


def _coaxial(ax, p0, e, tol) -> bool:
    p, d = ax
    d = d / np.linalg.norm(d)
    if abs(abs(float(d @ e)) - 1.0) > 1e-4:
        return False
    off = (p - p0) - ((p - p0) @ e) * e
    return float(np.linalg.norm(off)) <= tol


def _hexagon(planes: list[dict], rho_ref: float | None = None) -> tuple[float, float, float] | None:
    """Hexagon from planes parallel to the axis: (s, s0, s1) or None. The planes have 'rho', 'az', 's0', 's1'."""
    if len(planes) < 6:
        return None
    rhos = np.array([p["rho"] for p in planes])
    best = None
    for r0 in (np.unique(np.round(rhos, 3)) if rho_ref is None else [rho_ref]):
        grp = [p for p in planes if abs(p["rho"] - r0) <= max(0.02 * r0, 0.02)]
        if len(grp) < 6:
            continue
        a0 = grp[0]["az"]
        off = [((p["az"] - a0 + 30.0) % 60.0) - 30.0 for p in grp]     # deviation from the grid every 60°
        dirs = {int(round((p["az"] - a0) / 60.0)) % 6 for p in grp}
        if len(dirs) < 6 or max(abs(o) for o in off) > 2.0:
            continue
        s0 = min(p["s0"] for p in grp)
        s1 = max(p["s1"] for p in grp)
        if best is None or len(grp) > best[0]:
            best = (len(grp), 2 * float(np.mean([p["rho"] for p in grp])), s0, s1)
    return None if best is None else best[1:]


def _shank_sizes(diam: float) -> list[tuple[float, float, str]]:
    """Candidates d for the diameter of the shank: (d, certainty, description). Nominal diameter, mean d2 or small d3 (thread in CAD)."""
    out = []
    for k in tables.PITCH:
        if k < 2:
            continue
        th = tables.thread(k)
        for ref, conf, txt in ((th.d, 1.0, "Ø shank = d"), (th.d2, 0.8, "Ø shank = d2 (modeled thread)"),
                               (th.d3, 0.75, "Ø shank = d3 (modeled thread)")):
            if abs(diam - ref) <= max(0.05, 0.012 * ref):
                out.append((float(k), conf, txt))
                break
    return out


# ===================================================================== klasifikace

def analyze(part: Part) -> ShapeGuess | None:
    """Proposal of a fastener by shape, otherwise None."""
    try:
        recs = _faces(part)
    except Exception:  # noqa: BLE001 – strange geometry = not a fastener
        return None
    if not recs:
        return None
    cyls = [r for r in recs if r["type"] == "cylinder"]
    if not cyls:
        return None
    total = sum(r["area"] for r in recs)
    ref = max(cyls, key=lambda r: r["area"])
    p0, e = ref["axis"][0], ref["axis"][1] / np.linalg.norm(ref["axis"][1])
    bb = bbox(part.shape)
    size = float(np.linalg.norm(bb[3:] - bb[:3]))
    tol = max(0.01, 1e-3 * size)
    bad = 0.0
    for r in recs:
        if r["type"] in ("cylinder", "cone", "torus"):
            if not _coaxial(r["axis"], p0, e, tol):
                bad += r["area"]
        elif r["type"] == "plane":
            c = abs(float(r["normal"] @ e))
            if 1e-3 < c < 1 - 1e-4:
                bad += r["area"]
        elif r["type"] == "sphere":
            bad += r["area"]
    if bad > 0.05 * total:
        return None                                          # transverse holes, slots, inclined faces
    allnodes = np.vstack([r["nodes"] for r in recs if len(r["nodes"])])
    rel = allnodes - p0
    s_all = rel @ e
    r_all = np.linalg.norm(rel - np.outer(s_all, e), axis=1)
    smin, smax = float(s_all.min()), float(s_all.max())
    Lt = smax - smin
    if Lt <= 0:
        return None
    # cylinders: outer (shank, head) / inner (hole)
    outer, inner = [], []
    for r in cyls:
        x = r["point"] - p0
        rv = x - (x @ e) * e
        sv = (r["nodes"] - p0) @ e
        item = {"r": r["r"], "s0": float(sv.min()), "s1": float(sv.max()), "area": r["area"]}
        (outer if float(r["normal"] @ rv) > 0 else inner).append(item)
    # planes parallel to the axis (hexagons)
    t1 = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(e, t1)
    par_out, par_in = [], []
    for r in recs:
        if r["type"] != "plane" or abs(float(r["normal"] @ e)) > 1e-3:
            continue
        n = r["normal"]
        rho = float((r["point"] - p0) @ n)
        sv = (r["nodes"] - p0) @ e
        item = {"rho": abs(rho), "az": math.degrees(math.atan2(float(n @ t2), float(n @ t1))) % 360.0,
                "s0": float(sv.min()), "s1": float(sv.max())}
        (par_out if rho > 0 else par_in).append(item)
    vol = part.volume()
    hole = max((c for c in inner if c["s1"] - c["s0"] >= 0.8 * Lt), key=lambda c: c["s1"] - c["s0"], default=None)
    hexo = _hexagon(par_out)

    # ------------------------------------------------------------ washer
    if hole is not None and not par_out and Lt <= 0.25 * 2 * max((c["r"] for c in outer), default=0.0):
        ro = max(c["r"] for c in outer)
        d1, d2 = 2 * hole["r"], 2 * ro
        for kind, tab in (("washer", tables.WASHER), ("washer_large", tables.WASHER_LARGE)):
            for k, (td1, td2, th) in tab.items():
                if abs(d1 - td1) <= max(0.1, 0.03 * td1) and abs(d2 - td2) <= max(0.2, 0.05 * td2):
                    conf = 0.9 if abs(Lt - th) <= max(0.1, 0.25 * th) else 0.7
                    return ShapeGuess(part.name, kind, float(k), confidence=conf,
                                      reason=f"annulus Ø{d1:.4g}/Ø{d2:.4g} × {Lt:.3g} mm ≈ {('ISO 7093' if kind == 'washer_large' else 'ISO 7089')} M{k:g}")
        return None

    # ------------------------------------------------------------ matice
    if hole is not None and hexo is not None:
        s_hex, h0, h1 = hexo
        if h1 - h0 < 0.6 * Lt:
            return None
        if any(c["r"] < 0.8 * s_hex / 2 for c in outer):        # outer cylinder smaller than the hexagon = another part
            return None
        dh = 2 * hole["r"]
        best = None
        for k in tables.PITCH:
            th = tables.thread(k)
            if not (th.d1 - max(0.15, 0.03 * k) <= dh <= th.d + max(0.15, 0.02 * k)):
                continue
            nd = tables.nut_dims(k)
            s_ok = [s for s in (nd["s"], tables.DIN_HEX_S.get(k)) if s and abs(s_hex - s) <= max(0.15, 0.03 * s)]
            if not s_ok or not (0.3 * k <= Lt <= 1.5 * k):
                continue
            conf = 0.9 if abs(dh - k) <= 0.05 * k or abs(dh - th.d1) <= 0.05 * k else 0.75
            if best is None or conf > best[1]:
                best = (float(k), conf)
        if best is None:
            return None
        return ShapeGuess(part.name, "nut", best[0], confidence=best[1],
                          reason=f"hexagon s = {s_hex:.4g} mm, hole Ø{dh:.4g}, height {Lt:.3g} mm ≈ ISO 4032 M{best[0]:g}")
    if hole is not None or any(c["s1"] - c["s0"] > 0.6 * Lt for c in inner):
        return None                                          # pouzdro, trubka

    # ------------------------------------------------------------ bolt
    if not outer:
        return None
    shank = max(outer, key=lambda c: c["s1"] - c["s0"])
    rs = shank["r"]
    if len({round(c["r"], 2) for c in outer}) > 4:
        return None                                          # stepped shaft
    big = r_all > 1.15 * rs
    if not big.any():
        return None                                          # cylinder without a head (pin, magnet, rod)
    hs0, hs1 = float(s_all[big].min()), float(s_all[big].max())
    eps = max(0.02 * Lt, tol)
    if hs0 <= smin + eps:
        e_tip = 1.0                                          # head at the start, tip in the direction +e
        k_head = hs1 - smin
        L = smax - hs1
    elif hs1 >= smax - eps:
        e_tip = -1.0
        k_head = smax - hs0
        L = hs0 - smin
    else:
        return None                                          # collar in the middle (shouldered pin)
    if shank["s1"] - shank["s0"] < 0.5 * L:
        return None
    cands = _shank_sizes(2 * rs)
    if not cands:
        return None
    head_nodes = big
    rk = float(r_all[head_nodes].max())
    cones = [r for r in recs if r["type"] == "cone"]
    best = None
    L_b = L
    for d, c_shank, shank_txt in cands:
        L = L_b
        if L < 0.8 * d or k_head > 1.6 * d:
            continue
        cand = None
        # hexagon head
        if hexo is not None:
            s_hex = hexo[0]
            hd = tables.head_dims(d, "hex")
            s_ok = [s for s in (hd["s"], tables.DIN_HEX_S.get(tables.size_key(d))) if s
                    and abs(s_hex - s) <= max(0.15, 0.03 * s)]
            if s_ok and 0.5 * hd["k"] <= k_head <= 1.6 * hd["k"]:
                cand = ("hex", 0.95, f"hexagon head s = {s_hex:.4g}, k = {k_head:.3g} mm",
                        math.sqrt(3) / 2 * s_hex ** 2 * k_head)
        # countersunk head (cone ~90°)
        if cand is None and cones:
            cn = max(cones, key=lambda r: r["area"])
            dk = 2 * rk
            if 1.7 * d <= dk <= 2.4 * d and abs(math.degrees(cn["semi"]) - 45.0) <= 6.0:
                cand = ("countersunk", 0.85, f"countersunk head 90°, dk = {dk:.4g} mm",
                        math.pi * k_head / 3 * (rk ** 2 + rk * rs + rs ** 2))
                L = L + k_head                                # nominal length of the countersunk bolt incl. the head
        # cylindrical head
        if cand is None and hexo is None:
            dk = 2 * rk
            hd = tables.head_dims(d, "socket")
            if abs(dk - hd["dk"]) <= max(0.15, 0.05 * hd["dk"]) and 0.6 * d <= k_head <= 1.25 * d:
                cand = ("socket", 0.9, f"cylindrical head dk = {dk:.4g}, k = {k_head:.3g} mm (ISO 4762)",
                        math.pi / 4 * dk ** 2 * k_head)
            elif 1.4 * d <= dk <= 2.1 * d and 0.4 * d <= k_head <= 1.25 * d:
                cand = ("socket", 0.6, f"cylindrical / lens head dk = {dk:.4g}, k = {k_head:.3g} mm (outside ISO 4762)",
                        math.pi / 4 * dk ** 2 * k_head)
        if cand is None:
            continue
        head, c_head, head_txt, v_head = cand
        v_shank = math.pi * rs ** 2 * (L - (k_head if head == "countersunk" else 0.0))
        ratio = vol / max(v_head + v_shank, 1e-12)
        if not 0.55 <= ratio <= 1.3:
            continue
        conf = c_shank * c_head * (1.0 if 0.75 <= ratio <= 1.1 else 0.85)
        if best is None or conf > best[0]:
            best = (conf, d, head, round(L, 3), f"{shank_txt} {2 * rs:.4g} mm, {head_txt}, L = {L:.4g} mm")
    if best is None:
        return None
    conf, d, head, L, why = best
    _ = e_tip
    return ShapeGuess(part.name, "bolt", d, head, L, round(conf, 3), why)
