"""Bars from the geometry of CAD: (1) edges of the wire model in STEP (lines, arcs, curves → polyline) and (2)
recognition of a prismatic body (rod, profile) → a bar with the section of its face ("Replace by beam").

Prismatic body: a pair of planar faces with opposite normals and the same area, the line joining the centroids ∥ normal, all
other faces parallel to the axis (planar perpendicular, cylindrical with the axis ∥) – the longest of the identical pairs. Section from the face
(sections.section_from_face – sectionproperties); a match with the library (A within 0.5 %, Iy, Iz within 1 %, symmetric section)
→ profile from the library. Ends of the bar: centroids of the faces; a face touching a face of another part → connection to the body
(FrameConnection, rigid constraint to that face). The part is then left out of the mesh (Study.ignored_parts).
"""
from __future__ import annotations

import math

import numpy as np

from .model import FrameMember, FrameConnection, SectionDef


def _edges_not_in_solids(shape) -> list:
    """Edges of the shape that belong to no body (OCP 8 has no TopTools_IndexedMapOfShape → hash + IsSame)."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    seen: dict[int, list] = {}
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        ee = TopExp_Explorer(ex.Current(), TopAbs_EDGE)
        while ee.More():
            e = ee.Current()
            seen.setdefault(hash(e), []).append(e)
            ee.Next()
        ex.Next()
    out, done = [], {}
    ee = TopExp_Explorer(shape, TopAbs_EDGE)
    while ee.More():
        e = ee.Current()
        h = hash(e)
        if not any(e.IsSame(o) for o in seen.get(h, [])) and not any(e.IsSame(o) for o in done.get(h, [])):
            done.setdefault(h, []).append(e)
            out.append(TopoDS.Edge(e))
        ee.Next()
    return out


def wire_edges(path) -> list[dict]:
    """Free edges (outside the bodies) from STEP: [{type, start, end, point (middle of the parameter), length, center, radius}]."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from ..geometry import edge_props, edge_polyline
    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
    r = STEPControl_Reader()
    if r.ReadFile(str(path)) != IFSelect_RetDone:
        raise IOError(f"Cannot read STEP: {path}")
    r.TransferRoots()
    shape = r.OneShape()
    out = []
    for e in _edges_not_in_solids(shape):
        d = edge_props(e)
        if d["type"] == "curve":
            d["polyline"] = edge_polyline(e, 12)
        out.append(d)
    return out


def frame_from_wires(study, edges: list[dict], section: str, tol: float = 1e-3, prefix: str = "") -> list[str]:
    """Nodes and bars from the list of edges (wire_edges). End points within tol are merged. Returns the names of the new bars."""
    fr = study.frame
    if section not in fr.sections:
        raise ValueError(f"Cross-section '{section}' does not exist")
    pts = [np.asarray(n.xyz, float) for n in fr.nodes.values()]
    names = list(fr.nodes)

    def node(p) -> str:
        p = np.asarray(p, float)
        for nm, q in zip(names, pts):
            if np.linalg.norm(q - p) <= tol:
                return nm
        nm = f"{prefix}N{len(names) + 1}"
        while nm in fr.nodes:
            nm += "'"
        fr.add_node(nm, p)
        names.append(nm)
        pts.append(p)
        return nm
    new = []

    def member(a, b, through=None):
        nm = f"{prefix}P{len(fr.members) + 1}"
        while nm in fr.members:
            nm += "'"
        fr.members[nm] = FrameMember(nm, a, b, section, through=None if through is None else list(map(float, through)))
        new.append(nm)
    for e in edges:
        if e["type"] == "line":
            a, b = node(e["start"]), node(e["end"])
            if a != b:
                member(a, b)
        elif e["type"] == "circle":
            s, t = np.asarray(e["start"]), np.asarray(e["end"])
            if np.linalg.norm(s - t) <= tol:          # whole circle → two half-arcs
                c, ax = np.asarray(e["center"]), np.asarray(e["axis"])
                opp = 2 * c - s
                q1 = c + np.cross(ax, s - c)
                q2 = c - np.cross(ax, s - c)
                a, m_ = node(s), node(opp)
                member(a, m_, q1)
                member(m_, a, q2)
            else:
                member(node(s), node(t), e["point"])
        else:
            P = e.get("polyline")
            if P is None:
                continue
            ids = [node(p) for p in P]
            for a, b in zip(ids, ids[1:]):
                if a != b:
                    member(a, b)
    return new


# ------------------------------------------------------------------ prismatic body → bar

def prismatic(part, rel: float = 1e-4) -> dict | None:
    """Recognises a straight prismatic body: {"face_a", "face_b", "start", "end", "length", "area"} | None."""
    from ..geometry import face_props, cylinder_axis
    faces = part.faces()
    props = [face_props(f) for f in faces]
    planes = [i for i, p in enumerate(props) if p["type"] == "plane"]
    best = None
    for ii, i in enumerate(planes):
        for j in planes[ii + 1:]:
            pi, pj = props[i], props[j]
            if abs(pi["area"] - pj["area"]) > rel * max(pi["area"], pj["area"]) * 100:
                continue
            ni, nj = np.asarray(pi["normal"]), np.asarray(pj["normal"])
            if ni @ nj > -0.9999:
                continue
            d = np.asarray(pj["centroid"]) - np.asarray(pi["centroid"])
            L = float(np.linalg.norm(d))
            if L <= 0 or abs(abs(d @ ni) - L) > 1e-6 * L + 1e-6:
                continue
            ax = d / L
            ok = True
            for k, pk in enumerate(props):
                if k in (i, j):
                    continue
                if pk["type"] == "plane":
                    if abs(np.asarray(pk["normal"]) @ ax) > 1e-6:
                        ok = False
                        break
                elif pk["type"] == "cylinder":
                    cyl = cylinder_axis(faces[k])
                    if cyl is None or abs(abs(np.asarray(cyl[1]) @ ax) - 1) > 1e-6:
                        ok = False
                        break
                else:
                    ok = False
                    break
            if ok and (best is None or L > best["length"]):
                best = {"face_a": faces[i], "face_b": faces[j], "start": np.asarray(pi["centroid"]),
                        "end": np.asarray(pj["centroid"]), "length": L, "area": pi["area"],
                        "props_a": pi, "props_b": pj}
    return best


_LIB_CACHE: list | None = None


def library_match(sec, tol_a: float = 5e-3, tol_i: float = 1e-2) -> str | None:
    """Library profile with the same A, Iy, Iz (symmetric sections – alpha = 0)."""
    global _LIB_CACHE
    from . import sections as S
    if _LIB_CACHE is None:
        _LIB_CACHE = []
        for fam, names in S.families().items():
            for nm in names:
                try:
                    p = S.get(nm)
                except Exception:  # noqa: BLE001
                    continue
                if abs(p.alpha) < 1e-9:
                    _LIB_CACHE.append((nm, p.A, p.Iy, p.Iz))
    best = None
    for nm, A, Iy, Iz in _LIB_CACHE:
        e = max(abs(A - sec.A) / A, abs(Iy - sec.Iy) / Iy, abs(Iz - sec.Iz) / Iz)
        if abs(A - sec.A) <= tol_a * A and e <= tol_i and (best is None or e < best[1]):
            best = (nm, e)
    return best[0] if best else None


def member_from_part(study, parts, part_name: str, section_name: str | None = None, use_library: bool = True,
                     connect: bool = True) -> dict:
    """Replaces a prismatic part by a bar: section from the face (library on a match), nodes at the centroids of the faces, connection to the touching
    faces of other parts (selections __prut_<part>_a/_b). The part is left out of the mesh. Returns a description.
    """
    from .sections import section_from_face
    from ..selectors import FaceSelector
    from ..geometry import face_props
    part = next((p for p in parts if p.name == part_name), None)
    if part is None:
        raise ValueError(f"Part '{part_name}' is not in the geometry")
    pr = prismatic(part)
    if pr is None:
        raise ValueError(f"Part '{part_name}' is not a straight prismatic body (bar, profile) – it cannot be replaced by a member")
    sec, place = section_from_face(pr["face_a"], name=f"Cross-section {part_name}")
    lib = library_match(sec) if use_library else None
    fr = study.frame
    sname = section_name or (lib or f"Cross-section {part_name}")
    if sname not in fr.sections:
        if lib:
            fr.sections[sname] = SectionDef(sname, library=lib)
        else:
            d = sec.to_dict()
            d["alpha"] = 0.0                      # the axes of the face are already the main ones (up = main axis z)
            fr.sections[sname] = SectionDef(sname, props=d)
    a, b = f"{part_name}_A", f"{part_name}_B"
    fr.add_node(a, pr["start"])
    fr.add_node(b, pr["end"])
    up = list(map(float, place["z_axis"]))
    mname = part_name
    fr.members[mname] = FrameMember(mname, a, b, sname, material=study.part_materials.get(part_name), up=up)
    conns = []
    if connect:
        for end, key, fp in (("a", a, pr["props_a"]), ("b", b, pr["props_b"])):
            n0, c0 = np.asarray(fp["normal"]), np.asarray(fp["centroid"])
            sels = []
            for other in parts:
                if other.name == part_name or other.name in study.ignored_parts:
                    continue
                for f in other.faces():
                    q = face_props(f)
                    if q["type"] != "plane" or np.asarray(q["normal"]) @ n0 > -0.9999:
                        continue
                    if abs((np.asarray(q["centroid"]) - c0) @ n0) > 1e-4:
                        continue
                    bb = np.asarray(q["bbox"])
                    if np.all(c0 >= bb[:3] - 1e-6) and np.all(c0 <= bb[3:] + 1e-6):
                        sels.append(FaceSelector.from_props(other.name, q))
            if sels:
                sn = f"__prut_{part_name}_{end}"
                study.add_selection(sn, sels)
                fr.connections.append(FrameConnection(key, sn))
                conns.append((key, [s.part for s in sels]))
    if part_name not in study.ignored_parts:
        study.ignored_parts.append(part_name)
    return {"member": mname, "section": sname, "library": lib, "length_mm": pr["length"], "connections": conns,
            "A_mm2": sec.A, "Iy_mm4": sec.Iy, "Iz_mm4": sec.Iz, "J_mm4": sec.J}
