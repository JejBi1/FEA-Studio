"""Shared helpers of the MCP tools: compact JSON, rounding, selections of faces, long operations."""
from __future__ import annotations

import ast
import json
import math
import threading
import time
from typing import Any, Callable

import numpy as np

from ..selectors import edge_ok, face_ok
from .session import FeaError, Session


# ===================================================================== output

def rnd(x: Any, sig: int = 5) -> Any:
    """Rounds numbers to sig significant digits (recursively, numpy → lists), inf/nan → text."""
    if isinstance(x, dict):
        return {str(k): rnd(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, sig) for v in x]
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), sig)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if v == 0:
            return 0.0
        return float(f"{v:.{sig}g}")
    return x


def out(d: Any) -> str:
    """Output of a tool = compact JSON (the LLM reads it as text)."""
    return json.dumps(rnd(d), ensure_ascii=False, separators=(",", ":"), default=str)


# ===================================================================== faces and selections

FACE_QUERY_KEYS = {"part", "type", "x", "y", "z", "normal", "plane", "cylinder", "radius", "where", "near",
                   "limit", "tol"}


def face_id(part: str, fi: int) -> str:
    return f"{part}#{fi}"


def parse_face_id(sess: Session, fid: str) -> tuple[int, int]:
    if "#" not in fid:
        raise FeaError(f"'{fid}' is neither a face id (form 'Part#number' from fea_geometry_find_faces) nor a selection name")
    pname, idx = fid.rsplit("#", 1)
    pi = sess.part_index(pname)
    try:
        fi = int(idx)
    except ValueError:
        raise FeaError(f"Invalid face id '{fid}'") from None
    if not 0 <= fi < len(sess.face_props(pi)):
        raise FeaError(f"Part '{pname}' has no face {fi} (it has {len(sess.face_props(pi))} faces)")
    return pi, fi


_SAFE_NODES = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
               ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.Compare, ast.Eq,
               ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Name, ast.Load,
               ast.Constant, ast.Subscript, ast.Tuple, ast.List, ast.Call, ast.Attribute)
_SAFE_FUNCS = {"abs": abs, "min": min, "max": max, "sqrt": math.sqrt}
_TEXT_VARS = {"type", "part"}           # text variables of faces – only for comparison, not arithmetic (repeating the text)


def _maybe_text(node) -> bool:
    """Can the expression give text or a list? (index, slice, min/max with text, and/or with text) – forbidden in the arithmetic."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Subscript)):
        return True
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes))
    if isinstance(node, ast.Name):
        return node.id in _TEXT_VARS
    if isinstance(node, ast.Call):
        return any(_maybe_text(a) for a in node.args) or bool(node.keywords)
    if isinstance(node, ast.BoolOp):
        return any(_maybe_text(v) for v in node.values)
    if isinstance(node, ast.UnaryOp):
        return _maybe_text(node.operand)
    if isinstance(node, ast.BinOp):
        return _maybe_text(node.left) or _maybe_text(node.right)
    return False


def where_filter(expr: str) -> Callable[[dict], bool]:
    """Safe expression over the properties of a face, e.g. "area > 100 and cz > 5 and type == 'plane'".

    Variables: area, type, cx cy cz (centroid), nx ny nz (normal), xmin ymin zmin xmax ymax zmax,
    radius (cylinder, otherwise None), part. Functions abs, min, max, sqrt.
    """
    if len(expr) > 500:
        raise FeaError("The where expression is too long (max. 500 characters)")
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise FeaError(f"Disallowed element in the where expression: {type(node).__name__}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and not (
                isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
                and abs(node.right.value) <= 8 and isinstance(node.left, (ast.Name, ast.Constant))):
            raise FeaError("In a where expression the exponent may only be a number up to 8 and the base a variable or a number")
        if isinstance(node, ast.BinOp) and any(_maybe_text(x) for x in (node.left, node.right)):
            raise FeaError("A where expression may only compute with numbers (text and lists can only be compared)")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS):
            raise FeaError("Only the functions abs, min, max, sqrt are allowed in the where expression")
        if isinstance(node, ast.Attribute):
            raise FeaError("Attributes are not allowed in the where expression")
    code = compile(tree, "<where>", "eval")

    def f(env: dict) -> bool:
        try:
            return bool(eval(code, {"__builtins__": {}}, {**_SAFE_FUNCS, **env}))
        except Exception as e:  # noqa: BLE001
            raise FeaError(f"The where expression '{expr}' cannot be evaluated: {e}") from None
    return f


def face_env(part: str, p: dict) -> dict:
    c, n, bb = p["centroid"], p["normal"], p["bbox"]
    return {"area": float(p["area"]), "type": p["type"], "part": part,
            "cx": float(c[0]), "cy": float(c[1]), "cz": float(c[2]),
            "nx": float(n[0]), "ny": float(n[1]), "nz": float(n[2]),
            "xmin": float(bb[0]), "ymin": float(bb[1]), "zmin": float(bb[2]),
            "xmax": float(bb[3]), "ymax": float(bb[4]), "zmax": float(bb[5]),
            "radius": p.get("radius")}


def face_summary(part: str, fi: int, p: dict, full: bool = False) -> dict:
    d = {"id": face_id(part, fi), "type": p["type"], "area_mm2": float(p["area"]),
         "centroid_mm": list(map(float, p["centroid"]))}
    if p["type"] == "plane":
        d["normal"] = list(map(float, p["normal"]))
    if p.get("radius") is not None:
        d["radius_mm"] = p["radius"]
        d["axis_dir"] = list(map(float, p["axis_dir"]))
    d["bbox_mm"] = list(map(float, p["bbox"]))
    if full:
        d["point_on_face_mm"] = list(map(float, p["point"]))
        d["normal_at_point"] = list(map(float, p["normal"]))
        if p.get("radius") is not None:
            d["axis_point_mm"] = list(map(float, p["axis_point"]))
    return d


def find_faces(sess: Session, part: str | None = None, type: str | None = None, x=None, y=None, z=None,
               normal=None, plane=None, cylinder=None, radius=None, where: str | None = None, near=None,
               limit: int | None = None, tol: float = 1e-6) -> list[tuple[int, int]]:
    """Search of the faces – the same rules as fea_core.selectors.find_faces (x/y/z = the face lies entirely
    in the plane according to the envelope, normal = planar face with the outward normal, radius = cylinder), in addition:
    plane {"point", "normal"} (planar face in the plane, normal ±), cylinder {"point", "axis",
    "radius"?} (coaxial cylindrical faces), where (expression), near [x, y, z] (sorting by the distance of the
    centroid), limit.
    """
    if type is not None and type not in ("plane", "cylinder", "cone", "sphere", "torus", "other"):
        raise FeaError("type must be plane | cylinder | cone | sphere | torus | other")
    cond = where_filter(where) if where else None
    pis = [sess.part_index(part)] if part is not None else range(len(sess.parts))
    nrm = None
    if normal is not None:
        nrm = np.asarray(normal, float)
        if nrm.shape != (3,) or np.linalg.norm(nrm) == 0:
            raise FeaError("normal must be a nonzero vector [nx, ny, nz]")
        nrm = nrm / np.linalg.norm(nrm)
    pl = None
    if plane is not None:
        try:
            pp, pn = np.asarray(plane["point"], float), np.asarray(plane["normal"], float)
        except (KeyError, TypeError):
            raise FeaError('plane must be {"point": [x, y, z], "normal": [nx, ny, nz]}') from None
        pl = (pp, pn / np.linalg.norm(pn))
    cy = None
    if cylinder is not None:
        try:
            cp, ca = np.asarray(cylinder["point"], float), np.asarray(cylinder["axis"], float)
        except (KeyError, TypeError):
            raise FeaError('cylinder must be {"point": [x, y, z], "axis": [ax, ay, az], "radius": r (optional)}') from None
        cy = (cp, ca / np.linalg.norm(ca), cylinder.get("radius"))
    hits = []
    for pi in pis:
        pname = sess.parts[pi].name
        for fi, p in enumerate(sess.face_props(pi)):
            if not face_ok(p, type, nrm, (x, y, z), radius, tol):
                continue
            if pl is not None:
                bb = p["bbox"]
                size = max(float(np.linalg.norm(bb[3:] - bb[:3])), 1e-9)
                if p["type"] != "plane" or abs(abs(float(np.dot(p["normal"], pl[1]))) - 1) > 1e-6:
                    continue
                if abs(float(np.dot(np.asarray(p["point"]) - pl[0], pl[1]))) > max(tol * size, 1e-6):
                    continue
            if cy is not None:
                if p.get("radius") is None:
                    continue
                if abs(abs(float(np.dot(p["axis_dir"], cy[1]))) - 1) > 1e-6:
                    continue
                off = np.asarray(p["axis_point"]) - cy[0]
                off = off - np.dot(off, cy[1]) * cy[1]
                if np.linalg.norm(off) > 1e-4 * max(1.0, p["radius"]):
                    continue
                if cy[2] is not None and abs(p["radius"] - float(cy[2])) > 1e-6 * max(1.0, float(cy[2])):
                    continue
            if cond is not None and not cond(face_env(pname, p)):
                continue
            hits.append((pi, fi))
    if near is not None:
        q = np.asarray(near, float)
        hits.sort(key=lambda h: float(np.linalg.norm(sess.face_props(h[0])[h[1]]["centroid"] - q)))
    if limit:
        hits = hits[: int(limit)]
    return hits


def selector_of(sess: Session, pi: int, fi: int):
    from ..selectors import FaceSelector
    return FaceSelector.from_props(sess.parts[pi].name, sess.face_props(pi)[fi])


def resolve_faces(sess: Session, selection) -> list:
    """Faces from various notations → list of FaceSelector.

    selection: name of an existing selection | id of the face 'Part#i' | query {"x": 0, "part": ...}
    (parameters of fea_geometry_find_faces) | a whole selector {"part", "type", "area", "centroid", ...}
    | a list of any of these.
    """
    from ..selectors import FaceSelector
    st = sess.require_study()
    items = selection if isinstance(selection, list) else [selection]
    faces: list = []
    for it in items:
        if isinstance(it, str):
            if it in st.selections:
                faces += list(st.selections[it].faces)
            else:
                faces.append(selector_of(sess, *parse_face_id(sess, it)))
        elif isinstance(it, dict):
            if "area" in it and "centroid" in it:
                try:
                    faces.append(FaceSelector.from_dict({k: it[k] for k in
                                                         ("part", "type", "area", "centroid", "point", "normal", "bbox")}))
                except KeyError as e:
                    raise FeaError(f"The face selector lacks the key {e}") from None
            elif "id" in it and isinstance(it["id"], str):
                faces.append(selector_of(sess, *parse_face_id(sess, it["id"])))
            else:
                bad = set(it) - FACE_QUERY_KEYS
                if bad:
                    raise FeaError(f"Unknown keys in the face query: {', '.join(sorted(bad))}")
                hits = find_faces(sess, **it)
                if not hits:
                    raise FeaError(f"Query {json.dumps(it, ensure_ascii=False)} found no face")
                faces += [selector_of(sess, pi, fi) for pi, fi in hits]
        else:
            raise FeaError("selection must be a selection name, a face id, a query (dict) or a list of these")
    if not faces:
        raise FeaError("The face selection is empty")
    uniq, seen = [], set()
    for f in faces:                                  # without duplicates (the same face given twice)
        key = face_key(f)
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def resolve_selection(sess: Session, selection, name: str | None) -> str:
    """Returns the name of the selection in the study; the selection name is created / overwritten from the faces selection."""
    st = sess.require_study()
    if isinstance(selection, str) and selection in st.selections and (name is None or name == selection):
        return selection
    faces = resolve_faces(sess, selection)
    if not name:
        raise FeaError("Missing name for the new face selection")
    st.add_selection(name, faces)
    return name


def face_key(f) -> tuple:
    """Identity of a face of the selector (part + area + centroid) – the same faces given differently."""
    return (f.part, round(f.area, 6), tuple(round(c, 6) for c in f.centroid))


def same_faces(a, b) -> bool:
    return {face_key(f) for f in a} == {face_key(f) for f in b}


def selection_users(st, name: str) -> list[tuple[str, object]]:
    """Who uses the selection of faces: [(description, object)] – constraints, loads, thermal boundary conditions, mesh refinement,
    added masses, probes of the dynamics, contact pairs, connections of bars to a body and the elements of the mechanism (transfer to FE).
    """
    u = [(f"constraint {c.kind}", c) for c in st.constraints if c.selection == name]
    u += [(f"load {l.kind} in case '{lc.name}'", l) for lc in st.load_cases for l in lc.loads
          if getattr(l, "selection", None) == name]
    u += [(f"thermal boundary condition {b.kind} in thermal case '{c.name}'", b) for c in st.thermal.cases
          for b in c.loads if getattr(b, "selection", None) == name]
    u += [(f"mesh refinement {loc.get('size')} mm", loc) for loc in st.mesh.local if loc.get("selection") == name]
    u += [(f"added mass '{m.name or m.selection}'", m) for m in getattr(st, "masses", None) or []
          if getattr(m, "selection", None) == name]
    dyn = getattr(st, "dynamics", None)
    u += [(f"dynamics probe '{p.name}'", p) for p in (dyn.probes if dyn is not None else []) if p.selection == name]
    u += [(f"contact pair '{c.name}'", c) for c in getattr(st, "contacts", None) or [] if name in (c.a, c.b)]
    fr = getattr(st, "frame", None)
    u += [(f"member connection at node '{k.node}'", k) for k in (fr.connections if fr is not None else [])
          if k.selection == name]
    mech = getattr(st, "mechanism", None)
    if mech is not None:
        u += [(f"mechanism joint '{j.name}'", j) for j in mech.joints if name in (j.selection_a, j.selection_b)]
        u += [(f"mechanism force element '{f.name}'", f) for f in mech.forces
              if name in (f.selection_a, f.selection_b)]
        u += [(f"mechanism load '{l.name}'", l) for l in mech.loads if l.selection == name]
    return u


def guard_shared_selection(sess: Session, name: str, selection, allowed: list = ()) -> None:
    """A selection name used by other items (constraints, loads, thermal boundary conditions, mesh refinement, masses,
    probes, contacts, bar connections, mechanism – selection_users) must not be overwritten by other faces – it would change
    them too. The same faces (or selection = name) are fine; allowed = the items that are being replaced right now.
    """
    st = sess.require_study()
    if name not in st.selections or (isinstance(selection, str) and selection == name):
        return
    users = [w for w, o in selection_users(st, name) if not any(o is a for a in allowed)]
    if users and not same_faces(resolve_faces(sess, selection), st.selections[name].faces):
        raise FeaError(f"Selection '{name}' is already used by {users[0]} – other faces would change it as well. Choose another name, or specify selection='{name}' (same faces)")


def guarded_selection(sess: Session, selection, name: str, allowed: list = ()) -> str:
    """resolve_selection with the protection of a shared selection (guard_shared_selection) – a new selection name does not overwrite the faces
    that another item of the study uses.
    """
    guard_shared_selection(sess, name, selection, allowed)
    return resolve_selection(sess, selection, name)


def edge_id(part: str, ei: int) -> str:
    return f"{part}#e{ei}"


def parse_edge_id(sess: Session, eid: str) -> tuple[int, int]:
    if "#e" not in eid:
        raise FeaError(f"'{eid}' is neither an edge id (form 'Part#e<number>' from fea_geometry_find_edges) nor an edge selection name")
    pname, idx = eid.rsplit("#e", 1)
    pi = sess.part_index(pname)
    try:
        ei = int(idx)
    except ValueError:
        raise FeaError(f"Invalid edge id '{eid}'") from None
    if not 0 <= ei < len(sess.edge_props(pi)):
        raise FeaError(f"Part '{pname}' has no edge {ei} (it has {len(sess.edge_props(pi))} edges)")
    return pi, ei


EDGE_QUERY_KEYS = {"part", "type", "x", "y", "z", "radius", "direction", "near", "limit", "tol", "where"}


def edge_env(part: str, p: dict) -> dict:
    c, bb = p["centroid"], p["bbox"]
    return {"length": float(p["length"]), "type": p["type"], "part": part,
            "cx": float(c[0]), "cy": float(c[1]), "cz": float(c[2]),
            "xmin": float(bb[0]), "ymin": float(bb[1]), "zmin": float(bb[2]),
            "xmax": float(bb[3]), "ymax": float(bb[4]), "zmax": float(bb[5]), "radius": p.get("radius")}


def edge_summary(part: str, ei: int, p: dict) -> dict:
    d = {"id": edge_id(part, ei), "type": p["type"], "length_mm": float(p["length"]),
         "centroid_mm": list(map(float, p["centroid"])), "start_mm": list(map(float, p["start"])),
         "end_mm": list(map(float, p["end"]))}
    if p.get("radius") is not None:
        d["radius_mm"] = p["radius"]
        d["center_mm"] = list(map(float, p["center"]))
        d["axis_dir"] = list(map(float, p["axis"]))
    return d


def find_edges(sess: Session, part: str | None = None, type: str | None = None, x=None, y=None, z=None,
               radius=None, direction=None, near=None, limit: int | None = None, tol: float = 1e-6,
               where: str | None = None) -> list[tuple[int, int]]:
    """Search of the edges (the same rules as fea_core.selectors.find_edges): x/y/z = the edge lies entirely in the plane (two coordinates = on a line), radius = circle / arc,
    direction = parallel segment, near = sorting by the distance of the centroid, where = expression over length, type, cx…, xmin…zmax, radius.
    """
    if type is not None and type not in ("line", "circle", "ellipse", "curve"):
        raise FeaError("edge type must be line | circle | ellipse | curve")
    cond = where_filter(where) if where else None
    dvec = None
    if direction is not None:
        dvec = np.asarray(direction, float)
        if dvec.shape != (3,) or np.linalg.norm(dvec) == 0:
            raise FeaError("direction must be a nonzero vector [dx, dy, dz]")
        dvec = dvec / np.linalg.norm(dvec)
    pis = [sess.part_index(part)] if part is not None else range(len(sess.parts))
    hits = []
    for pi in pis:
        pname = sess.parts[pi].name
        for ei, p in enumerate(sess.edge_props(pi)):
            if not edge_ok(p, type, (x, y, z), radius, dvec, tol):
                continue
            if cond is not None and not cond(edge_env(pname, p)):
                continue
            hits.append((pi, ei))
    if near is not None:
        q = np.asarray(near, float)
        hits.sort(key=lambda h: float(np.linalg.norm(sess.edge_props(h[0])[h[1]]["centroid"] - q)))
    if limit:
        hits = hits[: int(limit)]
    return hits


def edge_selector_of(sess: Session, pi: int, ei: int):
    from ..selectors import EdgeSelector
    return EdgeSelector.from_props(sess.parts[pi].name, sess.edge_props(pi)[ei])


def resolve_edges(sess: Session, selection) -> list:
    """Edges from various notations → list of EdgeSelector: name of the selection of edges | id 'Part#e<i>' | query {"x": 200, "z": 10}
    (parameters of fea_geometry_find_edges) | a whole selector {"part", "type", "length", "centroid", ...} | a list.
    """
    from ..selectors import EdgeSelector
    st = sess.require_study()
    items = selection if isinstance(selection, list) else [selection]
    edges: list = []
    for it in items:
        if isinstance(it, str):
            if it in st.selections:
                if not st.selections[it].edges:
                    raise FeaError(f"Selection '{it}' contains no edges")
                edges += list(st.selections[it].edges)
            else:
                edges.append(edge_selector_of(sess, *parse_edge_id(sess, it)))
        elif isinstance(it, dict):
            if "length" in it and "centroid" in it:
                try:
                    edges.append(EdgeSelector.from_dict({k: it[k] for k in
                                                         ("part", "type", "length", "centroid", "point", "bbox")}))
                except KeyError as e:
                    raise FeaError(f"The edge selector lacks the key {e}") from None
            elif "id" in it and isinstance(it["id"], str):
                edges.append(edge_selector_of(sess, *parse_edge_id(sess, it["id"])))
            else:
                bad = set(it) - EDGE_QUERY_KEYS
                if bad:
                    raise FeaError(f"Unknown keys in the edge query: {', '.join(sorted(bad))}")
                hits = find_edges(sess, **it)
                if not hits:
                    raise FeaError(f"Query {json.dumps(it, ensure_ascii=False)} found no edge")
                edges += [edge_selector_of(sess, pi, ei) for pi, ei in hits]
        else:
            raise FeaError("edges must be an edge selection name, an edge id, a query (dict) or a list of these")
    if not edges:
        raise FeaError("The edge selection is empty")
    uniq, seen = [], set()
    for e in edges:
        key = (e.part, round(e.length, 6), tuple(round(c, 6) for c in e.centroid))
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq


def resolve_points(sess: Session, points, part: str | None = None) -> tuple[list, list[float]]:
    """Points → (list of PointSelector projected onto the surface of the part, distances of the projection [mm]). points = name of the selection
    of points | [x, y, z] | list [x, y, z] | {"point": [...], "part": ...} | list; part = the part (None = the part with the nearest
    surface). A point farther than 1 % of the size of the part from the surface = error.
    """
    from ..geometry import project_to_surface, bbox
    from ..selectors import PointSelector
    st = sess.require_study()
    if isinstance(points, str):
        if points not in st.selections or not st.selections[points].points:
            raise FeaError(f"'{points}' is not a point selection")
        return list(st.selections[points].points), [0.0] * len(st.selections[points].points)
    items = points
    if isinstance(points, dict) or (isinstance(points, (list, tuple)) and len(points) == 3
                                    and all(isinstance(v, (int, float)) for v in points)):
        items = [points]
    if not isinstance(items, (list, tuple)) or not items:
        raise FeaError("points must be [x, y, z], a list of points or {\"point\": [...], \"part\": ...}")
    out_s, dists = [], []
    for it in items:
        pp, pn = (it.get("point"), it.get("part", part)) if isinstance(it, dict) else (it, part)
        try:
            q = np.asarray([float(v) for v in pp], float)
        except (TypeError, ValueError):
            raise FeaError(f"Point {pp!r} must have 3 coordinates") from None
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            raise FeaError(f"Point {pp!r} must have 3 finite coordinates")
        cands = [sess.part_index(pn)] if pn is not None else range(len(sess.parts))
        best = None
        for pi in cands:
            proj, d = project_to_surface(sess.parts[pi].shape, q)
            if best is None or d < best[2]:
                best = (pi, proj, d)
        pi, proj, d = best
        bb = bbox(sess.parts[pi].shape)
        if d > max(0.01 * float(np.linalg.norm(bb[3:] - bb[:3])), 1e-3):
            raise FeaError(f"Point {q.tolist()} is {d:.3g} mm from the surface of part '{sess.parts[pi].name}' – specify a point on the surface (vertex, point on an edge or face)")
        out_s.append(PointSelector(sess.parts[pi].name, [float(v) for v in proj]))
        dists.append(float(d))
    return out_s, dists


def selection_info(sess: Session, name: str) -> dict:
    st = sess.require_study()
    s = st.selections[name]
    if s.edges:
        L = sum(e.length for e in s.edges)
        c = sum(np.asarray(e.centroid) * e.length for e in s.edges) / max(L, 1e-30)
        return {"name": name, "edges": len(s.edges), "parts": sorted({e.part for e in s.edges}), "length_mm": L,
                "centroid_mm": c, "types": sorted({e.type for e in s.edges})}
    if s.points:
        return {"name": name, "points": [p.point for p in s.points], "parts": sorted({p.part for p in s.points})}
    parts = sorted({f.part for f in s.faces})
    area = sum(f.area for f in s.faces)
    c = sum(np.asarray(f.centroid) * f.area for f in s.faces) / max(area, 1e-30)
    return {"name": name, "faces": len(s.faces), "parts": parts, "area_mm2": area, "centroid_mm": c,
            "types": sorted({f.type for f in s.faces})}


# ===================================================================== long operations

_PROTECTED_PIDS: set[int] = set()       # separate processes started by the server (GUI) – kill_children leaves them
_CALL = threading.local()                # cancellation of the tool call that is running (set by server.py)
ACTIVE_CALLS: set[threading.Event] = set()   # cancellation of all the running tool calls (the end of the server – __main__)


def set_call_cancel(ev: threading.Event | None) -> None:
    """Event of the cancellation of the tool call in this thread (the client sent notifications/cancelled or the server ends)."""
    _CALL.cancel = ev


def call_cancel() -> threading.Event | None:
    return getattr(_CALL, "cancel", None)


def protect_pid(pid: int) -> None:
    """The process (and its children, e.g. ccx started from the GUI) is never ended by kill_children."""
    _PROTECTED_PIDS.add(int(pid))


def kill_children(patterns=("fea_core.mesh.worker", "ccx", "mpiexec", "fea_core.ngs", "fea_core.hypre")) -> list[str]:
    """Ends the child processes of the server (meshing, ccx, MPI) – when the time is exceeded. The protected processes
    (the GUI started through REST /api/open-gui) and their children stay.
    """
    try:
        import psutil
    except ImportError:
        return []
    protected: set[int] = set()
    for pid in list(_PROTECTED_PIDS):
        try:
            p = psutil.Process(pid)
            protected.add(pid)
            protected |= {c.pid for c in p.children(recursive=True)}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            _PROTECTED_PIDS.discard(pid)
    killed = []
    for ch in psutil.Process().children(recursive=True):
        if ch.pid in protected:
            continue
        try:
            cmd = " ".join(ch.cmdline())
            if any(p in cmd for p in patterns):
                ch.kill()
                killed.append(ch.name())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed


def run_with_timeout(fn: Callable[[threading.Event], Any], timeout_s: float, what: str,
                     cancel_event: threading.Event | None = None) -> Any:
    """Runs fn(cancel) in a thread; after timeout_s it sets cancel, ends the subprocesses and raises FeaError.
    cancel_event = external cancellation (the asynchronous job fea_job_cancel) – the same as the exceeded time; without it
    the cancellation of the tool call by the MCP client is used (call_cancel – notifications/cancelled, the end of the server).
    """
    if timeout_s is None or timeout_s <= 0:
        raise FeaError("timeout_s must be a positive number of seconds")
    if cancel_event is None:
        cancel_event = call_cancel()
    cancel = threading.Event()
    box: dict = {}

    def target():
        try:
            box["value"] = fn(cancel)
        except BaseException as e:  # noqa: BLE001 – passed on to the caller
            box["error"] = e

    th = threading.Thread(target=target, daemon=True, name=f"fea-mcp {what}")
    t0 = time.time()
    th.start()
    if cancel_event is None:
        th.join(timeout_s)
    else:
        while th.is_alive() and time.time() - t0 < timeout_s and not cancel_event.is_set():
            th.join(0.25)
        if th.is_alive() and cancel_event.is_set():
            cancel.set()
            th.join(15)
            if th.is_alive():
                kill_children()
                th.join(30)
            raise FeaError(f"{what}: cancelled (fea_job_cancel or call cancelled by the client)")
    if th.is_alive():
        cancel.set()
        kill_children()
        th.join(30)
        raise FeaError(f"{what}: time limit of {timeout_s:g} s exceeded – operation terminated. Increase timeout_s or coarsen the mesh (global_size).")
    if "error" in box:
        e = box["error"]
        if isinstance(e, FeaError):
            raise e
        raise FeaError(f"{what} failed: {e}") from e
    box["elapsed"] = time.time() - t0
    return box["value"]
