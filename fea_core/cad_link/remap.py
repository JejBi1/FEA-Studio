"""Remapping of the selections of faces, edges and points to the new version of the geometry.

For each face of the old selection (in this order):
  1. identity from the CAD – the token of the face from the table of the last export translated by the add-in onto the faces of the new version (cad);
  2. exact match of the selector (selectors.MATCH_TOL) – the face has not changed (same);
  3. tolerant match – the same part and kind, direction of the normal / axis within 10°, the smallest displacement of the centroid and change of the area;
     only unambiguous (the second best at least 2× worse) – "probable", otherwise ambiguous;
  4. split face – to the found piece parallel pieces (same plane / axis) inside the original face are added (split, to check);
  5. otherwise not found (lost) – the old selector stays, the calculation reports it as missing.

The result is a proposal (RemapReport); the study is changed only by `apply_report` – automatically only the certain matches (same, cad),
the others according to the choice of the user / the parameter `accept`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..selectors import MATCH_TOL, EdgeSelector, FaceSelector, PointSelector, match_score

STATUS = ("same", "cad", "probable", "split", "ambiguous", "lost")
STATUS_LABEL = {"same": "no changes", "cad": "by CAD", "probable": "probable match",
                "split": "split face", "ambiguous": "ambiguous", "lost": "not found"}
CERTAIN = frozenset({"same", "cad"})
DEFAULT_ACCEPT = frozenset({"same", "cad", "probable", "split"})
MAX_ANGLE = 10.0          # ° – the normal / axis of the face may be rotated at most by this much
MAX_SCORE = 1.0           # tolerant match: a worse score = not found
MARGIN = 2.0              # the second candidate must be at least this many times worse


# ===================================================================== result

@dataclass
class ItemMap:
    """One face / edge / point of the old selection → a proposal of the new one."""
    kind: str                              # face | edge | point
    old: dict
    new: list[dict] = field(default_factory=list)
    status: str = "lost"
    method: str = ""
    score: float | None = None
    note: str = ""
    candidates: list[dict] = field(default_factory=list)   # [{"selector", "score"}] – for checking

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "status": self.status, "method": self.method,
             "old": _brief(self.old), "new": [_brief(x) for x in self.new]}
        if self.score is not None:
            d["score"] = round(self.score, 4)
        if self.note:
            d["note"] = self.note
        if self.candidates:
            d["candidates"] = [{"selector": _brief(c["selector"]), "score": round(c["score"], 4)}
                               for c in self.candidates]
        return d


def _brief(sel: dict) -> dict:
    out = {"part": sel.get("part")}
    for k in ("type", "area", "length", "centroid", "point"):
        if k in sel:
            v = sel[k]
            out[k] = [round(float(x), 4) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else (
                round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
    return out


@dataclass
class SelectionMap:
    name: str
    entity: str
    users: list[str] = field(default_factory=list)
    items: list[ItemMap] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.items:
            return "same"
        return max((it.status for it in self.items), key=STATUS.index)

    def changed(self) -> bool:
        return any(it.status != "same" for it in self.items)

    def to_dict(self) -> dict:
        return {"name": self.name, "entity": self.entity, "status": self.status, "users": self.users,
                "items": [it.to_dict() for it in self.items]}


@dataclass
class AnchorMap:
    """Coordinates given when the item was created (the point of a remote force, the centre of an imprint…) – moved with its selection."""
    where: str
    key: tuple                              # path in the dictionary of the study
    old: list[float]
    new: list[float]
    status: str = "same"                    # same | shifted | review
    note: str = ""

    def to_dict(self) -> dict:
        return {"where": self.where, "status": self.status, "old": [round(v, 4) for v in self.old],
                "new": [round(v, 4) for v in self.new], **({"note": self.note} if self.note else {})}


@dataclass
class RemapReport:
    parts: list = field(default_factory=list)            # parts.PartMatch
    selections: list[SelectionMap] = field(default_factory=list)
    anchors: list[AnchorMap] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)        # bolts, contacts, coordinates to check
    unchanged: bool = False                               # the geometry has not changed (the fingerprints and the names)

    def counts(self) -> dict:
        out = {k: 0 for k in STATUS}
        for s in self.selections:
            out[s.status] += 1
        return {k: v for k, v in out.items() if v}

    def needs_review(self, accept=DEFAULT_ACCEPT) -> bool:
        """Is there anything to show the user? (anything but the certain matches, the renamed parts according to the geometry, the displacements of the points)"""
        if any(s.status not in CERTAIN for s in self.selections):
            return True
        if any(p.method in ("geometry", "removed") for p in self.parts):
            return True
        return any(a.status != "same" for a in self.anchors) or bool(self.notes)

    def blocking(self, accept=DEFAULT_ACCEPT) -> list[str]:
        """Selections that are not remapped even with the choice accept (the calculation would report them as missing)."""
        return [s.name for s in self.selections if any(it.status not in accept for it in s.items)]

    def part_mapping(self) -> dict[str, str]:
        return {p.old: p.new for p in self.parts if p.old and p.new}

    def to_dict(self) -> dict:
        return {"unchanged": self.unchanged, "counts": self.counts(),
                "parts": [p.to_dict() for p in self.parts if p.method != "name" or p.changed],
                "selections": [s.to_dict() for s in self.selections],
                "anchors": [a.to_dict() for a in self.anchors], "notes": list(self.notes)}


# ===================================================================== score

def _unit(v) -> np.ndarray | None:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def _angle(a, b, signed: bool) -> float:
    """Angle between the directions [°]; signed=False = axis without orientation."""
    ua, ub = _unit(a), _unit(b)
    if ua is None or ub is None:
        return 0.0
    c = float(ua @ ub)
    if not signed:
        c = abs(c)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def face_score(old: dict, new: dict, size: float) -> float:
    """Deviation of the face of the new version from the old one (0 = identical, inf = it cannot be it)."""
    if old.get("type") != new.get("type"):
        return math.inf
    t = old.get("type")
    ang = 0.0
    if t == "plane":
        ang = _angle(old.get("normal"), new.get("normal"), True)
    elif t in ("cylinder", "cone", "torus") and "axis" in old and "axis" in new:
        ang = _angle(old["axis"], new["axis"], False)
    if ang > MAX_ANGLE:
        return math.inf
    dc = float(np.linalg.norm(np.asarray(old["centroid"], float) - np.asarray(new["centroid"], float))) / size
    da = 0.5 * abs(math.log(max(float(new["area"]), 1e-12) / max(float(old["area"]), 1e-12)))
    extra = 0.0
    if "radius" in old and "radius" in new and old["radius"] and new["radius"]:
        extra += 0.5 * abs(math.log(new["radius"] / old["radius"]))
    if t not in ("plane", "cylinder", "cone", "sphere", "torus"):
        bo, bn = np.asarray(old["bbox"], float), np.asarray(new["bbox"], float)
        extra += float(np.abs(bo - bn).max()) / size
    return ang / MAX_ANGLE + dc + da + extra


def edge_score(old: dict, new: dict, size: float) -> float:
    if old.get("type") != new.get("type"):
        return math.inf
    ang = 0.0
    if old.get("type") == "line":
        ang = _angle(np.asarray(old["end"]) - np.asarray(old["start"]),
                     np.asarray(new["end"]) - np.asarray(new["start"]), False)
    elif "axis" in old and "axis" in new:
        ang = _angle(old["axis"], new["axis"], False)
    if ang > MAX_ANGLE:
        return math.inf
    dc = float(np.linalg.norm(np.asarray(old["centroid"], float) - np.asarray(new["centroid"], float))) / size
    dl = 0.5 * abs(math.log(max(float(new["length"]), 1e-12) / max(float(old["length"]), 1e-12)))
    extra = 0.0
    if old.get("radius") and new.get("radius"):
        extra += 0.5 * abs(math.log(new["radius"] / old["radius"]))
    return ang / MAX_ANGLE + dc + dl + extra


def _same_surface(a: dict, b: dict, size: float) -> bool:
    """Do the two faces lie in the same plane / cylinder (parallel pieces of a split face)?"""
    if a.get("type") != b.get("type"):
        return False
    tol = 1e-6 * size + 1e-9
    if a["type"] == "plane":
        if _angle(a["normal"], b["normal"], True) > 0.05:
            return False
        n = _unit(a["normal"])
        return n is not None and abs(float(n @ (np.asarray(b["point"]) - np.asarray(a["point"])))) <= tol
    if a["type"] in ("cylinder", "cone", "sphere", "torus") and "axis_point" in a and "axis_point" in b:
        if abs(a.get("radius", 0.0) - b.get("radius", 0.0)) > 1e-6 * max(a.get("radius", 1.0), 1.0):
            return False
        if a["type"] == "sphere":
            return float(np.linalg.norm(np.asarray(a["axis_point"]) - np.asarray(b["axis_point"]))) <= tol
        if _angle(a["axis"], b["axis"], False) > 0.05:
            return False
        e = _unit(a["axis"])
        d = np.asarray(b["axis_point"]) - np.asarray(a["axis_point"])
        return float(np.linalg.norm(d - (d @ e) * e)) <= tol
    return False


def _inside(props: dict, box: np.ndarray, pad: float) -> bool:
    c = np.asarray(props["centroid"], float)
    return bool(np.all(c >= box[:3] - pad) and np.all(c <= box[3:] + pad))


# ===================================================================== faces

def _face_key(sel: FaceSelector) -> tuple:
    return (sel.part, round(sel.area, 6), tuple(round(c, 6) for c in sel.centroid))


def _old_props(sel: FaceSelector, old_geo) -> dict:
    """Full property of the old face (axis, radius) from the original geometry; otherwise only the data of the selector."""
    d = sel.to_dict()
    pf = old_geo.part(sel.part) if old_geo is not None else None
    if pf is not None:
        best = min(((match_score(d, fp), fi) for fi, fp in enumerate(pf.faces)), default=None)
        if best is not None and best[0] < MATCH_TOL:
            out = dict(pf.faces[best[1]])
            out["_old_index"] = best[1]
            return out
    return {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in d.items()}


def _cad_faces(old: dict, old_part: int | None, ctx) -> tuple[list[tuple[int, int]], str] | None:
    """Faces of the new version by the identity from the CAD: None = the CAD knows nothing about the face, [] = it disappeared in the CAD."""
    if not ctx.old_rows or not ctx.resolved:
        return None
    from .facetable import _score
    best = None
    for k, row in enumerate(ctx.old_rows):
        if old_part is not None and row.get("part") in ctx.old_row_part and ctx.old_row_part[row["part"]] != old_part:
            continue
        s = _score(row, old, ctx.old_size)
        if s <= 1e-3 and (best is None or s < best[0]):
            best = (s, k)
    if best is None:
        return None
    tok = ctx.old_rows[best[1]].get("token")
    if tok is None or tok not in ctx.resolved:
        return None
    from .facetable import bind_row
    hits = []
    for g in ctx.resolved[tok]:
        h = bind_row(g, ctx.new_geo)
        if h is not None and h not in hits:
            hits.append(h)
    return hits, tok


class _Ctx:
    """Shared data of the remapping."""

    def __init__(self, old_geo, new_geo, part_map: dict[str, str], old_meta, new_meta):
        from .facetable import face_rows, resolved, row_part_map
        self.old_geo, self.new_geo = old_geo, new_geo
        self.part_map = part_map
        self.old_rows = face_rows(old_meta)
        self.resolved = resolved(new_meta)
        self.old_row_part = row_part_map(old_meta, [p.name for p in old_geo.parts]) if old_geo is not None and \
            self.old_rows else {}
        self.old_size = old_geo.size if old_geo is not None else 1.0


def remap_faces(selectors: list[FaceSelector], ctx: _Ctx) -> dict[tuple, ItemMap]:
    """Map of the unique old faces (key _face_key) → proposal."""
    uniq: dict[tuple, FaceSelector] = {}
    for s in selectors:
        uniq.setdefault(_face_key(s), s)
    result: dict[tuple, ItemMap] = {}
    claimed: dict[tuple[int, int], tuple] = {}         # new face → old face (certain match)
    pending: list[tuple] = []
    info: dict[tuple, dict] = {}
    for key, sel in uniq.items():
        it = ItemMap("face", sel.to_dict())
        result[key] = it
        new_name = ctx.part_map.get(sel.part)
        ni = ctx.new_geo.index.get(new_name) if new_name else None
        if ni is None:
            it.status, it.note = "lost", f"part '{sel.part}' is not in the new geometry"
            continue
        pf = ctx.new_geo.part(ni)
        old = _old_props(sel, ctx.old_geo)
        old_pi = ctx.old_geo.index.get(sel.part) if ctx.old_geo is not None else None
        info[key] = {"old": old, "ni": ni, "pf": pf}
        cad = _cad_faces(old, old_pi, ctx)
        # exact match (selector with the new name of the part)
        sd = dict(sel.to_dict(), part=new_name)
        exact = min(((match_score(sd, fp), fi) for fi, fp in enumerate(pf.faces)), default=(math.inf, -1))
        if cad is not None and cad[0]:
            hits = cad[0]
            if exact[0] < MATCH_TOL and (ni, exact[1]) not in hits:
                it.status, it.method = "ambiguous", "CAD × geometry"
                it.note = "CAD and geometry point to different faces"
                it.candidates = [{"selector": _selector(ctx, h), "score": 0.0} for h in hits + [(ni, exact[1])]]
                continue
            it.status, it.method = ("same" if exact[0] < MATCH_TOL and len(hits) == 1 else "cad"), "cad"
            it.new = [_selector(ctx, h) for h in hits]
            for h in hits:
                claimed.setdefault(h, key)
            info[key]["hits"] = hits
            continue
        if cad is not None and not cad[0]:
            info[key]["cad_lost"] = True
        if exact[0] < MATCH_TOL:
            it.status, it.method, it.score = "same", "geometry", float(exact[0])
            it.new = [FaceSelector.from_props(new_name, pf.faces[exact[1]]).to_dict()]
            claimed.setdefault((ni, exact[1]), key)
            info[key]["hits"] = [(ni, exact[1])]
            continue
        pending.append(key)
    # tolerant match – the best pair first, only the faces not yet taken
    scored = []
    for key in pending:
        old, pf, ni = info[key]["old"], info[key]["pf"], info[key]["ni"]
        cands = sorted((face_score(old, fp, pf.size), fi) for fi, fp in enumerate(pf.faces))
        info[key]["cands"] = [(s, fi) for s, fi in cands if math.isfinite(s)]
        if info[key]["cands"]:
            scored.append((info[key]["cands"][0][0], key))
    for _, key in sorted(scored):
        it, ni, pf = result[key], info[key]["ni"], info[key]["pf"]
        free = [(s, fi) for s, fi in info[key]["cands"] if claimed.get((ni, fi), key) == key]
        it.candidates = [{"selector": FaceSelector.from_props(pf.name, pf.faces[fi]).to_dict(), "score": float(s)}
                         for s, fi in free[:3]]
        if not free or free[0][0] > MAX_SCORE:
            continue
        best = free[0]
        second = free[1][0] if len(free) > 1 else math.inf
        if second < MARGIN * best[0] + 1e-9:
            it.status, it.method, it.score = "ambiguous", "geometry", float(best[0])
            it.note = "several similar faces"
            continue
        it.status, it.method, it.score = "probable", "geometry", float(best[0])
        it.new = [FaceSelector.from_props(pf.name, pf.faces[best[1]]).to_dict()]
        claimed[(ni, best[1])] = key
        info[key]["hits"] = [(ni, best[1])]
    for key, it in result.items():
        if it.status == "lost" and not it.note:
            it.note = "the face no longer exists in CAD" if info.get(key, {}).get("cad_lost") else "no matching face found"
            if it.candidates:
                it.note += " – nearest candidates in the proposal"
    # split faces: to the found piece the parallel pieces inside the original face are added
    for key, it in result.items():
        hits = info.get(key, {}).get("hits")
        if not hits or it.status == "ambiguous":
            continue
        old, ni, pf = info[key]["old"], info[key]["ni"], info[key]["pf"]
        area_new = sum(float(ctx.new_geo.part(h[0]).faces[h[1]]["area"]) for h in hits)
        if area_new >= 0.98 * float(old["area"]):
            continue
        base = pf.faces[hits[0][1]] if hits[0][0] == ni else None
        if base is None:
            continue
        box = np.asarray(old["bbox"], float).copy()
        shift = np.asarray(base["centroid"], float) - np.asarray(old["centroid"], float)
        if base["type"] == "plane":                       # displacement only perpendicular to the plane (thickness), not in the plane
            n = _unit(base["normal"])
            shift = (shift @ n) * n if n is not None else np.zeros(3)
        box[:3] += shift
        box[3:] += shift
        # the part may have grown at the same time (an extended plate with a groove) – the envelope of the face grows at most like the envelope of the part
        opf = ctx.old_geo.part(it.old.get("part")) if ctx.old_geo is not None else None
        if opf is not None:
            box[:3] -= np.maximum(np.asarray(opf.bbox[:3]) - np.asarray(pf.bbox[:3]), 0.0)
            box[3:] += np.maximum(np.asarray(pf.bbox[3:]) - np.asarray(opf.bbox[3:]), 0.0)
        sib = [fi for fi, fp in enumerate(pf.faces)
               if (ni, fi) not in hits and (ni, fi) not in claimed and _same_surface(base, fp, pf.size)
               and _inside(fp, box, 1e-3 * pf.size)]
        if sib:
            it.status, it.method = "split", (it.method + " + split").strip(" +")
            it.new = it.new + [FaceSelector.from_props(pf.name, pf.faces[fi]).to_dict() for fi in sib]
            it.note = f"face split into {len(it.new)} pieces – coplanar pieces added, please check"
            for fi in sib:
                claimed[(ni, fi)] = key
    return result


def _selector(ctx: _Ctx, hit: tuple[int, int]) -> dict:
    pf = ctx.new_geo.part(hit[0])
    return FaceSelector.from_props(pf.name, pf.faces[hit[1]]).to_dict()


# ===================================================================== edges

def _edge_key(sel: EdgeSelector) -> tuple:
    return (sel.part, round(sel.length, 6), tuple(round(c, 6) for c in sel.centroid))


def remap_edges(selectors: list[EdgeSelector], ctx: _Ctx) -> dict[tuple, ItemMap]:
    uniq: dict[tuple, EdgeSelector] = {}
    for s in selectors:
        uniq.setdefault(_edge_key(s), s)
    out: dict[tuple, ItemMap] = {}
    claimed: set[tuple[int, int]] = set()
    pending = []
    for key, sel in uniq.items():
        it = ItemMap("edge", sel.to_dict())
        out[key] = it
        new_name = ctx.part_map.get(sel.part)
        ni = ctx.new_geo.index.get(new_name) if new_name else None
        if ni is None:
            it.note = f"part '{sel.part}' is not in the new geometry"
            continue
        pf = ctx.new_geo.part(ni)
        sd = dict(sel.to_dict(), part=new_name)
        exact = min(((match_score(sd, ep), ei) for ei, ep in enumerate(pf.edges)), default=(math.inf, -1))
        if exact[0] < MATCH_TOL:
            it.status, it.method, it.score = "same", "geometry", float(exact[0])
            it.new = [EdgeSelector.from_props(new_name, pf.edges[exact[1]]).to_dict()]
            claimed.add((ni, exact[1]))
            continue
        old = sd
        opf = ctx.old_geo.part(sel.part) if ctx.old_geo is not None else None
        if opf is not None:
            b = min(((match_score(sel.to_dict(), ep), ei) for ei, ep in enumerate(opf.edges)), default=None)
            if b is not None and b[0] < MATCH_TOL:
                old = opf.edges[b[1]]
        cands = sorted((edge_score(old, ep, pf.size), ei) for ei, ep in enumerate(pf.edges))
        pending.append((cands[0][0] if cands else math.inf, key, ni, pf,
                        [(s, ei) for s, ei in cands if math.isfinite(s)]))
    for _, key, ni, pf, cands in sorted(pending, key=lambda x: x[0]):
        it = out[key]
        free = [(s, ei) for s, ei in cands if (ni, ei) not in claimed]
        it.candidates = [{"selector": EdgeSelector.from_props(pf.name, pf.edges[ei]).to_dict(), "score": float(s)}
                         for s, ei in free[:3]]
        if not free or free[0][0] > MAX_SCORE:
            it.note = "no matching edge found"
            continue
        second = free[1][0] if len(free) > 1 else math.inf
        if second < MARGIN * free[0][0] + 1e-9:
            it.status, it.method, it.score, it.note = "ambiguous", "geometry", float(free[0][0]), "several similar edges"
            continue
        it.status, it.method, it.score = "probable", "geometry", float(free[0][0])
        it.new = [EdgeSelector.from_props(pf.name, pf.edges[free[0][1]]).to_dict()]
        claimed.add((ni, free[0][1]))
    return out


# ===================================================================== body

def _host_face(point, pf) -> int | None:
    """Face of the part on which the point lies (nearest, within 1e-6 of the size)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt
    v = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, point))).Vertex()
    best = None
    for fi, f in enumerate(pf.face_objs):
        ds = BRepExtrema_DistShapeShape(v, f)
        if ds.IsDone() and ds.NbSolution() and (best is None or ds.Value() < best[0]):
            best = (ds.Value(), fi)
    if best is None or best[0] > 1e-6 * pf.size + 1e-7:
        return None
    return best[1]


def _project(point, face) -> tuple[np.ndarray, float]:
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt
    from ..geometry import xyz_of
    v = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, point))).Vertex()
    ds = BRepExtrema_DistShapeShape(v, face)
    if not ds.IsDone() or not ds.NbSolution():
        raise ValueError("the point cannot be projected")
    return xyz_of(ds.PointOnShape2(1)), float(ds.Value())


def remap_points(selectors: list[PointSelector], ctx: _Ctx, face_maps: dict) -> list[ItemMap]:
    """Point on the surface: an unchanged surface = the same point; otherwise a displacement with the face on which it lies (difference of the centroids),
    and projection onto its new form.
    """
    out = []
    for sel in selectors:
        it = ItemMap("point", sel.to_dict())
        out.append(it)
        new_name = ctx.part_map.get(sel.part)
        ni = ctx.new_geo.index.get(new_name) if new_name else None
        if ni is None:
            it.note = f"part '{sel.part}' is not in the new geometry"
            continue
        pf = ctx.new_geo.part(ni)
        p = np.asarray(sel.point, float)
        opf = ctx.old_geo.part(sel.part) if ctx.old_geo is not None else None
        host = _host_face(p, opf) if opf is not None else None
        if host is not None:
            old_face = FaceSelector.from_props(sel.part, opf.faces[host])
            fm = face_maps.get(_face_key(old_face))
            if fm is None:
                fm = remap_faces([old_face], ctx)[_face_key(old_face)]
            if fm.status == "same":
                it.status, it.method = "same", "geometry"
                it.new = [PointSelector(new_name, [float(v) for v in p]).to_dict()]
                continue
            if fm.new and fm.status in ("cad", "probable", "split"):
                tgt = fm.new[0]
                nfi = next((fi for fi, fp in enumerate(pf.faces) if match_score(tgt, fp) < MATCH_TOL), None)
                if nfi is not None:
                    q = p + np.asarray(tgt["centroid"], float) - np.asarray(opf.faces[host]["centroid"], float)
                    q, _ = _project(q, pf.face_objs[nfi])
                    it.status, it.method = fm.status if fm.status != "split" else "probable", "moved with its face"
                    it.new = [PointSelector(new_name, [float(v) for v in q]).to_dict()]
                    it.note = f"point moved by {np.linalg.norm(q - p):.3g} mm with the face it lies on"
                    continue
        try:
            from ..geometry import project_to_surface
            q, dist = project_to_surface(pf.part.shape, p)
        except Exception:  # noqa: BLE001
            continue
        if dist <= 1e-6 * pf.size + 1e-7:
            it.status, it.method = "same", "geometry"
            it.new = [PointSelector(new_name, [float(v) for v in p]).to_dict()]
        elif dist <= 0.05 * pf.size:
            it.status, it.method = "probable", "nearest surface point"
            it.new = [PointSelector(new_name, [float(v) for v in q]).to_dict()]
            it.note = f"point projected onto the new surface ({dist:.3g} mm)"
    return out
