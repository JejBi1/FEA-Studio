"""Proposal of the joints of the mechanism from the geometry of the CAD and from the metadata of Fusion (ALWAYS only a PROPOSAL – the user confirms it).

From the geometry: coaxial cylindrical faces of the parts of two different bodies (a pin in a hole: the same radius within the tolerance, the axes identical,
the axial ranges overlap) → a revolute joint (if the pin has an axial stop = a touching planar face perpendicular to the axis),
otherwise a cylindrical joint. Clusters along the axis separated by a gap = separate joints (a lid on two hinges → 2 coaxial
revolute joints → the check of the freedom reports the redundancy). Certainty 0–1 and a reason; the selections of faces for the transfer to FE
(FaceSelector as elsewhere) are created only when the proposal is accepted (apply_suggestion).

Metadata of Fusion (JSON from the adapter, key "joints" – the adapter does not send them yet, the format is a proposal):
    "joints": [{"name": "Rev1", "type": "revolute" | "rigid" | "slider" | "cylindrical" | "pin_slot" | "planar" |
                "ball", "occurrence_one": "Deska:1", "occurrence_two": "Rameno:1",
                "origin": [x, y, z], "axis": [x, y, z], "axis2": [x, y, z] | null,
                "limits": {"min": -30, "max": 90} | null,
                "units": {"length": "mm", "angle": "deg"}}]
(Fusion API: Joint.jointMotion – RevoluteJointMotion.rotationAxisVector, rotationLimits; SliderJointMotion.
slideDirectionVector, slideLimits; geometryOrOriginOne.origin.) The frame = the body with an occurrence that is in Body.ground.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .model import MotionJoint

FUSION_TYPES = {"rigid": "fixed", "revolute": "revolute", "slider": "prismatic", "cylindrical": "cylindrical",
                "pin_slot": "pin_slot", "pinslot": "pin_slot", "planar": "planar", "ball": "spherical"}


@dataclass
class JointSuggestion:
    """Proposal of a joint (to confirm): MotionJoint + faces (part, index of the face) on both bodies, certainty, reason."""
    joint: MotionJoint
    faces_a: list = field(default_factory=list)      # [(part, index of the face)]
    faces_b: list = field(default_factory=list)
    confidence: float = 0.5
    reason: str = ""
    source: str = "cad"

    def to_dict(self) -> dict:
        return {"joint": self.joint.to_dict(), "faces_a": [list(f) for f in self.faces_a],
                "faces_b": [list(f) for f in self.faces_b], "confidence": round(self.confidence, 3),
                "reason": self.reason, "source": self.source}


def _cyl_faces(part):
    from ..geometry import cylinder_axis
    from ..joints.detect import _tri_nodes
    from OCP.TopAbs import TopAbs_REVERSED
    out = []
    for k, f in enumerate(part.faces()):
        c = cylinder_axis(f)
        if c is None:
            continue
        p, d, r = c
        nodes = _tri_nodes(f)
        out.append({"k": k, "face": f, "p": p, "d": d / np.linalg.norm(d), "r": float(r), "nodes": nodes,
                    "hole": f.Orientation() == TopAbs_REVERSED})
    return out


def _plane_faces(part):
    from ..geometry import face_props
    out = []
    for k, f in enumerate(part.faces()):
        pr = face_props(f)
        if pr["type"] == "plane":
            out.append({"k": k, "c": np.asarray(pr["centroid"], float), "n": np.asarray(pr["normal"], float),
                        "area": pr["area"]})
    return out


def _axial_stop(pa_planes, pb_planes, p0, e, r, tol) -> bool:
    """A touching planar face perpendicular to the axis at the pin (the axial stop) → a revolute, not a cylindrical joint."""
    for fa in pa_planes:
        if abs(abs(fa["n"] @ e) - 1) > 1e-3:
            continue
        ra = np.linalg.norm((fa["c"] - p0) - ((fa["c"] - p0) @ e) * e)
        if ra > 4 * r + 10:
            continue
        for fb in pb_planes:
            if abs(abs(fb["n"] @ e) - 1) > 1e-3 or fa["n"] @ fb["n"] > -0.99:
                continue
            if abs((fa["c"] - fb["c"]) @ e) <= tol:
                return True
    return False


def suggest_from_geometry(study, parts, tol_r: float = 0.05, tol_mm: float = 0.5) -> list[JointSuggestion]:
    """Proposals of the joints from the coaxial cylindrical faces of the parts of different bodies (see the module)."""
    mech = study.mechanism
    by_name = {p.name: p for p in parts}
    body_of = {}
    for b in mech.bodies:
        for p in b.parts:
            body_of[p] = b.name
    cyl = {n: _cyl_faces(by_name[n]) for n in body_of if n in by_name}
    planes: dict = {}
    existing = {frozenset((j.body_a, j.body_b)) for j in mech.joints}
    groups: dict = {}
    names = [n for n in body_of if n in cyl]
    for ia, na in enumerate(names):
        for nb in names[ia + 1:]:
            if body_of[na] == body_of[nb]:
                continue
            for fa in cyl[na]:
                for fb in cyl[nb]:
                    if abs(abs(fa["d"] @ fb["d"]) - 1) > 1e-4:
                        continue
                    r = max(fa["r"], fb["r"])
                    dr = abs(fa["r"] - fb["r"])
                    if dr > max(tol_r * r, tol_mm):
                        continue
                    e = fa["d"]
                    off = (fb["p"] - fa["p"]) - ((fb["p"] - fa["p"]) @ e) * e
                    if np.linalg.norm(off) > max(0.02 * r, 0.05):
                        continue
                    sa, sb = fa["nodes"] @ e, fb["nodes"] @ e
                    lo, hi = max(sa.min(), sb.min()), min(sa.max(), sb.max())
                    if hi - lo <= 1e-6:
                        continue
                    # key of the axis: pair of parts + line (point nearest to the origin, direction with a sign)
                    p0 = fa["p"] - (fa["p"] @ e) * e
                    sgn = 1.0 if e[np.argmax(np.abs(e))] > 0 else -1.0
                    key = (na, nb, tuple(np.round(p0, 3)), tuple(np.round(sgn * e, 6)))
                    groups.setdefault(key, []).append((fa, fb, lo, hi, dr, r))
    out = []
    n = len(mech.joints)
    for (na, nb, _, _), lst in groups.items():
        # clusters along the axis
        lst.sort(key=lambda x: x[2])
        clusters, cur = [], [lst[0]]
        for item in lst[1:]:
            if item[2] <= max(c[3] for c in cur) + 1e-6:
                cur.append(item)
            else:
                clusters.append(cur)
                cur = [item]
        clusters.append(cur)
        ba, bb = body_of[na], body_of[nb]
        if mech.body(bb).ground and not mech.body(ba).ground:     # frame as body_a
            na, nb, ba, bb = nb, na, bb, ba
            clusters = [[(fb, fa, lo, hi, dr, r) for fa, fb, lo, hi, dr, r in c] for c in clusters]
        for cl in clusters:
            fa0 = cl[0][0]
            e = fa0["d"]
            lo, hi = min(c[2] for c in cl), max(c[3] for c in cl)
            r = max(c[5] for c in cl)
            p_axis = fa0["p"] - (fa0["p"] @ e) * e
            origin = p_axis + 0.5 * (lo + hi) * e
            for key in (na, nb):
                if key not in planes:
                    planes[key] = _plane_faces(by_name[key])
            stop = _axial_stop(planes[na], planes[nb], p_axis, e, r, max(0.02 * r, 0.05))
            pin_hole = any(c[0]["hole"] != c[1]["hole"] for c in cl)
            dr = max(c[4] for c in cl)
            conf = 0.6 + 0.2 * pin_hole + 0.15 * (dr <= 0.02 * r) - (0.0 if stop else 0.15)
            kind = "revolute" if stop else "cylindrical"
            n += 1
            j = MotionJoint(f"Joint {n}", kind, ba, bb, [float(x) for x in origin], [float(x) for x in e],
                            source="cad")
            why = (f"coaxial cylinders {na} / {nb}, Ø {2 * r:.4g} mm (difference {2 * dr:.3g} mm), contact length {hi - lo:.4g} mm" + ("; axial stop (planar faces in contact) → revolute" if stop else
                                         "; no axial stop → cylindrical (sliding along the axis free)"))
            if frozenset((ba, bb)) in existing:
                conf -= 0.3
                why += "; the bodies are already connected by another joint"
            out.append(JointSuggestion(j, sorted({(na, c[0]["k"]) for c in cl}), sorted({(nb, c[1]["k"]) for c in cl}),
                                       float(min(max(conf, 0.05), 0.99)), why))
    out.sort(key=lambda s: -s.confidence)
    return out


def _norm_keys(name: str) -> list[str]:
    keys = [str(name or "").strip().casefold()]
    cur = str(name or "").strip()
    for _ in range(3):
        new = re.sub(r"\s*\(\d+\)$|:\d+$|\s*<\d+>$|_\d+$", "", cur).strip()
        if new == cur:
            break
        cur = new
        keys.append(cur.casefold())
    return keys


def suggest_from_metadata(study, meta: dict) -> list[JointSuggestion]:
    """Joints from the metadata of Fusion (the format see the module) – the bodies by the names of the occurrences (without the suffixes ":1", "<1>", " (2)")."""
    mech = study.mechanism
    lookup = {}
    for b in mech.bodies:
        for p in b.parts:
            for k in _norm_keys(p):
                lookup.setdefault(k, b.name)
        for k in _norm_keys(b.name):
            lookup.setdefault(k, b.name)

    def body(occ):
        for k in _norm_keys(occ):
            if k in lookup:
                return lookup[k]
        return None
    out = []
    for k, d in enumerate((meta or {}).get("joints", [])):
        kind = FUSION_TYPES.get(str(d.get("type", "")).lower().replace(" ", "_"))
        ba, bb = body(d.get("occurrence_one")), body(d.get("occurrence_two"))
        units = d.get("units") or {}
        lf = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}.get(units.get("length", "mm"), 1.0)
        ang = math.pi / 180 if units.get("angle", "deg") == "deg" else 1.0
        why = []
        conf = 0.9
        if kind is None:
            why.append(f"unknown type '{d.get('type')}'")
            conf = 0.2
            kind = "fixed"
        if ba is None or bb is None:
            why.append("the occurrence belongs to no body – add the bodies")
            conf = min(conf, 0.3)
        j = MotionJoint(d.get("name") or f"Fusion {k + 1}", kind, ba or "", bb or "",
                        [float(x) * lf for x in (d.get("origin") or [0, 0, 0])],
                        [float(x) for x in (d.get("axis") or [0, 0, 1])],
                        axis2=None if d.get("axis2") is None else [float(x) for x in d["axis2"]], source="fusion")
        lim = d.get("limits")
        if lim and (lim.get("min") is not None or lim.get("max") is not None):
            sc = ang if kind in ("revolute", "cylindrical") else lf
            j.limits = [None if lim.get("min") is None else lim["min"] * sc,
                        None if lim.get("max") is None else lim["max"] * sc]
        out.append(JointSuggestion(j, confidence=conf, reason="; ".join(why) or "joint from Fusion", source="fusion"))
    return out


def apply_suggestion(study, parts, sug: JointSuggestion, name: str | None = None) -> MotionJoint:
    """Accepts a proposal: creates the selections of the faces of the joint in the study (Joint – part) and adds the joint to the mechanism."""
    from ..geometry import face_props
    from ..selectors import FaceSelector
    by_name = {p.name: p for p in parts}
    j = sug.joint
    if name:
        j.name = name
    existing = {x.name for x in study.mechanism.joints}
    base, n = j.name, 1
    while j.name in existing:
        n += 1
        j.name = f"{base} ({n})"

    def make(faces, side):
        if not faces:
            return None
        sels = []
        for pn, k in faces:
            f = by_name[pn].faces()[k]
            sels.append(FaceSelector.from_props(pn, face_props(f)))
        nm = f"{j.name} – {faces[0][0]}"
        i = 1
        while nm in study.selections:
            i += 1
            nm = f"{j.name} – {faces[0][0]} ({i})"
        study.add_selection(nm, sels)
        return nm
    j.selection_a = make(sug.faces_a, "a") or j.selection_a
    j.selection_b = make(sug.faces_b, "b") or j.selection_b
    study.mechanism.joints.append(j)
    return j
