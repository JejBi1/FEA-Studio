"""Update of the geometry of the study with the settings kept: proposal (plan_update) → confirmation → use (apply_report).

Shared by the GUI ("Replace geometry…", CAD add-in), MCP (fea_geometry_replace) and scripts. Without Qt.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..selectors import FaceSelector
from .faces import Geometry
from .model import CadLink, Iteration
from .remap import (DEFAULT_ACCEPT, AnchorMap, RemapReport, SelectionMap, _Ctx, _face_key, _edge_key,
                    _host_face, remap_edges, remap_faces, remap_points)

# fields of the bolt that follow from the geometry (recognition); the others (class, preload, friction…) were entered by the user
BOLT_GEOMETRY = ("origin", "axis", "head_part", "head_dw", "nut_part", "nut_s", "nut_dw", "thread_end", "nut_m",
                 "plates", "washers", "length", "head_cone")


def _users(study, name: str) -> list[str]:
    from ..mcp.common import selection_users
    out = [w for w, _ in selection_users(study, name)]
    out += [f"imprint '{imp.name}'" for imp in study.imprints if getattr(imp, "name", None) == name]
    return out


def _centroid(sels: list[dict]) -> np.ndarray | None:
    """Centroid of a selection of faces (weighted by the area)."""
    if not sels:
        return None
    a = np.array([float(s["area"]) for s in sels])
    c = np.array([s["centroid"] for s in sels], float)
    return (a[:, None] * c).sum(0) / max(a.sum(), 1e-12)


def plan_update(study, old_parts: list | None, new_parts: list, old_meta: dict | None = None,
                new_meta: dict | None = None) -> RemapReport:
    """Proposal of the remapping of the study onto the new parts (changes nothing). old_parts = parts of the current geometry (it refines the search:
    axes of cylinders, the face on which the point lies); None = only the data of the selectors.
    """
    from .parts import map_parts
    old_geo = Geometry(old_parts) if old_parts else None
    new_geo = Geometry(new_parts)
    rep = RemapReport()
    if old_geo is not None:
        rep.parts = map_parts(old_geo, new_geo, old_meta, new_meta)
    else:                                   # without the old geometry: by names only
        from .parts import PartMatch
        names = {p.name for p in new_parts}
        used = {s.part for sel in study.selections.values() for s in sel.faces + sel.edges + sel.points}
        rep.parts = [PartMatch(n, n if n in names else None, "name" if n in names else "removed", True)
                     for n in sorted(used)]
    part_map = rep.part_mapping()
    ctx = _Ctx(old_geo, new_geo, part_map, old_meta, new_meta)
    faces = [f for s in study.selections.values() for f in s.faces]
    edges = [e for s in study.selections.values() for e in s.edges]
    fmap = remap_faces(faces, ctx) if faces else {}
    emap = remap_edges(edges, ctx) if edges else {}
    for name, sel in study.selections.items():
        sm = SelectionMap(name, sel.entity, _users(study, name))
        for f in sel.faces:
            sm.items.append(fmap[_face_key(f)])
        for e in sel.edges:
            sm.items.append(emap[_edge_key(e)])
        if sel.points:
            sm.items += remap_points(sel.points, ctx, fmap)
        rep.selections.append(sm)
    rep.anchors = _anchors(study, rep, ctx)
    rep.notes += _notes(study, rep)
    rep.unchanged = (all(p.method == "name" and not p.changed for p in rep.parts)
                     and len(rep.parts) == len(new_parts)
                     and all(s.status == "same" for s in rep.selections)
                     and all(a.status == "same" for a in rep.anchors))
    rep._ctx = ctx                         # bolts and imprints when used (the same cache of faces)
    return rep


# ===================================================================== coordinates bound to a selection

def _anchors(study, rep: RemapReport, ctx) -> list[AnchorMap]:
    shift: dict[str, np.ndarray] = {}
    for sm in rep.selections:
        old = [it.old for it in sm.items if it.kind == "face"]
        new = [n for it in sm.items if it.kind == "face" for n in it.new]
        c0, c1 = _centroid(old), _centroid(new)
        if c0 is not None and c1 is not None and all(it.new for it in sm.items):
            d = c1 - c0
            # planar faces: the point moves only perpendicular to the plane (change of thickness, length) – growth of the face in its
            # plane (a removed fillet, a longer plate) does not move the point; a hole, a cylinder: displacement of the centroid (a moved hole)
            ns = [np.asarray(s["normal"], float) for s in old + new]
            if all(s.get("type") == "plane" for s in old + new) and ns and \
                    all(abs(float(n @ ns[0])) > 0.9998 * np.linalg.norm(n) * np.linalg.norm(ns[0]) for n in ns):
                n0 = ns[0] / max(np.linalg.norm(ns[0]), 1e-12)
                d = (d @ n0) * n0
            shift[sm.name] = d
    out: list[AnchorMap] = []

    def add(where, key, point, sel):
        if point is None or sel not in shift:
            return
        p = np.asarray(point, float)
        d = shift[sel]
        size = ctx.new_geo.size
        if float(np.linalg.norm(d)) <= 1e-9 * size:
            out.append(AnchorMap(where, key, list(map(float, p)), list(map(float, p)), "same"))
        else:
            out.append(AnchorMap(where, key, list(map(float, p)), list(map(float, p + d)), "shifted",
                                 f"moved with selection '{sel}' by {np.linalg.norm(d):.3g} mm – please check"))
    for i, c in enumerate(study.constraints):
        if c.kind == "remote_support":
            add(f"remote support '{c.selection}' – point", ("constraints", i, "point"), c.point, c.selection)
    for k, lc in enumerate(study.load_cases):
        for i, ld in enumerate(lc.loads):
            if ld.kind == "remote":
                add(f"remote load '{ld.selection}' ({lc.name}) – point", ("load_cases", k, "loads", i, "point"),
                    ld.point, ld.selection)
            elif ld.kind == "pressure" and getattr(ld, "variable", False) and ld.origin is not None:
                add(f"variable pressure '{ld.selection}' ({lc.name}) – origin",
                    ("load_cases", k, "loads", i, "origin"), ld.origin, ld.selection)
    for i, m in enumerate(study.masses):
        if getattr(m, "kind", "") == "point_mass" and m.point is not None:
            add(f"point mass '{m.name or m.selection}' – centre of gravity", ("masses", i, "point"), m.point, m.selection)
    for i, p in enumerate(study.dynamics.probes):
        if p.selection and p.point is not None:
            add(f"probe '{p.name}' – point", ("dynamics", "probes", i, "point"), p.point, p.selection)
    out += _imprint_anchors(study, ctx)
    return out


def _imprint_anchors(study, ctx) -> list[AnchorMap]:
    """Centre of an imprint = projection onto the new form of the face on which it lies (the thickness changed → the centre moves
    perpendicular to the face; a displacement in the plane of the face cannot be estimated – to check when the face has changed).
    """
    out = []
    if ctx.old_geo is None:
        return out
    for i, imp in enumerate(study.imprints):
        opf = ctx.old_geo.part(imp.part)
        new_name = ctx.part_map.get(imp.part)
        npf = ctx.new_geo.part(new_name) if new_name else None
        c = np.asarray(imp.center, float)
        where = f"imprint '{imp.name}' – centre"
        if opf is None or npf is None:
            out.append(AnchorMap(where, ("imprints", i, "center"), list(c), list(c), "review", "part not found"))
            continue
        host = _host_face(c, opf)
        if host is None:
            continue
        old_face = FaceSelector.from_props(imp.part, opf.faces[host])
        fm = remap_faces([old_face], ctx)[_face_key(old_face)]
        if fm.status == "same":
            out.append(AnchorMap(where, ("imprints", i, "center"), list(c), list(c), "same"))
            continue
        if not fm.new:
            out.append(AnchorMap(where, ("imprints", i, "center"), list(c), list(c), "review",
                                 "imprint face not found"))
            continue
        nf = fm.new[0]
        n = np.asarray(nf["normal"], float)
        n = n / max(np.linalg.norm(n), 1e-12)
        q = c + n * float((np.asarray(nf["point"], float) - c) @ n)
        out.append(AnchorMap(where, ("imprints", i, "center"), list(map(float, c)), list(map(float, q)), "shifted",
                             f"projected onto the new face (shift {np.linalg.norm(q - c):.3g} mm) – check the in-plane position"))
    return out


def _notes(study, rep: RemapReport) -> list[str]:
    out = []
    changed = {p.old for p in rep.parts if p.old and (p.changed or p.method in ("geometry", "removed"))}
    removed = [p.old for p in rep.parts if p.method == "removed"]
    if removed:
        from .refs import references
        d = study.to_dict()
        used = [n for n in removed if references(d, n)]
        if used:
            out.append("Parts removed from the geometry have settings in the study (material, printing, bolts…): "
                       + ", ".join(used[:8]) + (" …" if len(used) > 8 else ""))
    if any(c.source == "auto" and (c.a in changed or c.b in changed) for c in study.contacts):
        out.append("Automatically found contact pairs lie on changed parts – run \"Find contacts\" again in the Contacts step (gap and contact area are recalculated)")
    free = []
    if study.coordinate_systems:
        free.append("coordinate systems")
    if any("center" in loc for loc in study.mesh.local):
        free.append("mesh refinement spheres")
    if any(l.kind == "rotation" for lc in study.load_cases for l in lc.loads):
        free.append("rotation axis")
    if study.twod.enabled:
        free.append("2D section plane")
    if free and changed:
        out.append("Coordinates independent of the selections were left unchanged – check: " + ", ".join(free))
    return out


# ===================================================================== use

def _set(d, key: tuple, value):
    for k in key[:-1]:
        d = d[k]
    d[key[-1]] = value


def _merge_bolts(d: dict, rep: RemapReport, new_parts: list) -> list[str]:
    """Bolts on the changed parts: geometry (axis, bearing faces, clamped parts) from the new recognition, the settings
    of the user (class, preload, friction…) remain.
    """
    bolts = d.get("bolts") or []
    if not bolts:
        return []
    changed = {p.new for p in rep.parts if p.new and (p.changed or p.method == "geometry")}
    msgs = []
    todo = [b for b in bolts if changed & ({b.get("head_part"), b.get("nut_part")} | set(b.get("cad_parts") or [])
                                           | {r.get("part") for r in b.get("plates") or []})]
    if not todo:
        return []
    try:
        from ..joints.detect import detect
        found = detect(new_parts, shapes=True)
        cands = [s.spec for s in found.suggestions if s.kind == "bolt"]
    except Exception as e:  # noqa: BLE001 – the recognition is only a help
        return [f"Bolts on changed parts cannot be detected again ({e}) – check them in the Joints step"]
    for b in todo:
        mine = set(b.get("cad_parts") or [])
        hit = next((c for c in cands if mine and mine & set(c.cad_parts)), None)
        if hit is None:
            msgs.append(f"Bolt '{b.get('name')}': not found in the new geometry – check it in the Joints step")
            continue
        hd = hit.to_dict()
        for k in BOLT_GEOMETRY:
            b[k] = hd.get(k, b.get(k))
        b["cad_parts"] = list(hit.cad_parts)
        msgs.append(f"Bolt '{b.get('name')}': position and clamped parts taken from the new geometry")
    return msgs


def apply_report(study, report: RemapReport, accept=DEFAULT_ACCEPT, new_step: str | Path | None = None,
                 new_parts: list | None = None, new_meta: dict | None = None, trigger: str = "step",
                 source: str = "", results=None, old_parts: list | None = None, reject=(), reject_anchors=()):
    """New study with the remapped selections, the renamed parts and the moved points. accept = the states of the items that
    are used (the others keep the old selector – the calculation reports them as missing); reject = selections whose
    uncertain proposals the user rejected (the certain matches are always used); reject_anchors = indices of the displacements of the points
    (report.anchors) that are not used. results = the results of the previous geometry → KPI into the iteration to which they
    belong (the first iteration is created from them).
    """
    from ..study import Study
    from .refs import rename_parts
    d = study.to_dict()
    mapping = report.part_mapping()
    rename_parts(d, mapping)
    applied = {"selections": 0, "kept": 0}
    for sm in report.selections:
        s = d["selections"].get(sm.name)
        if s is None:
            continue
        faces, edges, points = [], [], []
        seen = set()
        for it in sm.items:
            ok = it.status in accept and (sm.name not in reject or it.status in ("same", "cad"))
            use = it.new if ok and it.new else None
            if use is None:
                old = dict(it.old, part=mapping.get(it.old.get("part"), it.old.get("part")))
                use = [old]
                applied["kept"] += 1
            for x in use:
                k = (it.kind, x.get("part"), round(float(x.get("area", x.get("length", 0.0))), 6),
                     tuple(round(float(v), 6) for v in x.get("centroid", x.get("point", []))))
                if k in seen:
                    continue
                seen.add(k)
                {"face": faces, "edge": edges, "point": points}[it.kind].append(x)
        s["faces"], s["edges"], s["points"] = faces, edges, points
        if not edges:
            s.pop("edges", None)
        if not points:
            s.pop("points", None)
        applied["selections"] += 1
    for k, a in enumerate(report.anchors):
        if a.status == "shifted" and k not in reject_anchors:
            try:
                _set(d, a.key, list(a.new))
            except (KeyError, IndexError, TypeError):
                pass
    notes = []
    if new_parts is not None:
        notes = _merge_bolts(d, report, new_parts)
    if new_step is not None:
        d["geometry"] = str(new_step)
    new = Study.from_dict(d)
    new.load_warnings = list(getattr(study, "load_warnings", []))
    _record_iteration(new, study, report, new_step, new_meta, trigger, source, results, old_parts)
    report.notes += [n for n in notes if n not in report.notes]
    return new


def _record_iteration(new, old_study, report, new_step, new_meta, trigger, source, results, old_parts):
    from .digest import step_digest
    from .facetable import parameters
    from .history import kpis
    link = getattr(old_study, "cad_link", None)
    link = CadLink.from_dict(link.to_dict()) if link is not None else CadLink()
    meta = new_meta or {}
    if not link.source:
        link.source = str(meta.get("source") or source or "STEP")
    if meta.get("document") and not link.document:
        link.document = str(meta["document"])[:200]
    did = (meta.get("link") or {}).get("document_id") if isinstance(meta.get("link"), dict) else None
    if isinstance(did, str) and did and not link.document_id:
        link.document_id = did[:256]
    if not link.iterations:                       # first iteration = the current geometry
        first = Iteration(1, trigger="initial", source=link.source,
                          geometry=Path(str(old_study.geometry or "")).name, digest=link.digest,
                          params=dict(link.params))
        if results is not None:
            try:
                first.kpi = kpis(old_study, results, old_parts)
            except Exception as e:  # noqa: BLE001 – the summary must not stop the update
                first.note = f"the results summary cannot be built: {e}"
        link.iterations.append(first)
    elif results is not None and not link.iterations[-1].kpi:
        try:
            link.iterations[-1].kpi = kpis(old_study, results, old_parts)
        except Exception as e:  # noqa: BLE001 – the summary must not stop the update; the gap is recorded
            note = f"the results summary cannot be built: {e}"
            last = link.iterations[-1]
            last.note = f"{last.note}; {note}" if last.note else note
    digest = ""
    if new_step is not None and Path(new_step).is_file():
        try:
            digest = step_digest(new_step)
        except OSError:
            pass
    params = parameters(meta)
    prev = link.iterations[-1].params if link.iterations else {}
    changes = [f"{k}: {prev.get(k, {}).get('expression', '–')} → {v.get('expression', v.get('value'))}"
               for k, v in params.items() if prev.get(k, {}).get("expression") != v.get("expression")
               and k in prev]
    for p in report.parts:
        if p.method == "added":
            changes.append(f"new part {p.new}")
        elif p.method == "removed":
            changes.append(f"removed part {p.old}")
        elif p.old != p.new:
            changes.append(f"part {p.old} → {p.new}")
        elif p.changed:
            changes.append(f"changed part {p.new}")
    link.revision += 1
    link.digest = digest
    link.params = params
    link.iterations.append(Iteration(len(link.iterations) + 1, time.strftime("%Y-%m-%dT%H:%M:%S"), trigger,
                                     link.source, Path(str(new_step)).name if new_step else "", digest, params,
                                     report.counts(), changes[:40]))
    new.cad_link = link
