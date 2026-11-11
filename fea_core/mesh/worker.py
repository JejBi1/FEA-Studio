"""Gmsh meshing process – started separately: python -m fea_core.mesh.worker job.json

Input (job.json):
  parts:      [{"name", "brep"}]                 bodies in global position (mm)
  selections: {"name": [selector, ...]}          see fea_core.selectors
  settings:   MeshSettings as dict
  imprints:   [{"name", "part", "center", "normal", "radius", "size"}]  circle imprint into the plane of
              the part faces (bolt bearing annulus) → new selection "name" (optional)
  volume_sizes: [{"part", "size"}]               local element size in the whole part (welds)
  settings.local: [{"selection", "size"}] (on selection faces) or [{"center", "radius", "size"}] (sphere, Ball field)
  edges:      {"name": [EdgeSelector dict]}       edges for loads → line3 segments (Mesh.edges)
  points:     {"name": [PointSelector dict]}      points for loads → mesh nodes (Mesh.points); the point is projected onto
              the nearest vertex / edge / face of the part and inserted by fragmentation (occ.fragment with the point: on a face =
              inserted node, on an edge it splits it, at a vertex it merges – verified 13. 10. 2026, scratch/edgeload/exp_point.py)
  output:     path to .npz, report: path to .json

Procedure: import BREP part by part → match selectors to faces (content, centroid,
envelope) → copies of the selected faces and imprint discs → fragmentation of all bodies with the discs
(conforming assembly mesh, imprinted annulus) → removal of disc remnants outside the bodies →
assignment of new faces to the copies (points on the face + envelope) → removal of the copies →
tet10 mesh → quality check → write.

The module may depend only on gmsh, numpy and fea_core.selectors (it runs in a clean process – no OCP).
"""
from __future__ import annotations

import json
import sys
import time

import gmsh
import numpy as np

from fea_core.selectors import MATCH_TOL, match_score         # numpy only (no OCP)

HO_OPT_BELOW = 0.2   # min. scaled Jacobian below which curved elements are optimised


def _props(tag: int) -> dict:
    occ = gmsh.model.occ
    return {
        "area": occ.getMass(2, tag),
        "centroid": list(occ.getCenterOfMass(2, tag)),
        "bbox": list(occ.getBoundingBox(2, tag)),
    }


def _score(sel: dict, cand: dict) -> float:
    """Mismatch of a selector and a Gmsh face / curve: content (length) and centroid; the envelope only as a weak criterion – Gmsh
    returns for curved faces (cone, torus) a loose envelope up to mm larger than OCP (verified 12. 10. 2026).
    """
    return match_score(sel, cand, 1e-4)


def _curve_props(tag: int) -> dict:
    occ = gmsh.model.occ
    return {"length": occ.getMass(1, tag), "centroid": list(occ.getCenterOfMass(1, tag)),
            "bbox": list(occ.getBoundingBox(1, tag))}


def _boundary_curves(vols: list[int]) -> list[int]:
    surfs = _boundary_surfaces(vols)
    if not surfs:
        return []
    bnd = gmsh.model.getBoundary([(2, t) for t in surfs], combined=False, oriented=False)
    return sorted({abs(t) for d, t in bnd if d == 1})


def _curve_lies_on(cand: int, ref: int, tol: float) -> bool:
    """Does the curve cand lie entirely on the curve ref? (envelope + points inside the parametrisation)"""
    occ = gmsh.model.occ
    cb = np.asarray(occ.getBoundingBox(1, cand))
    rb = np.asarray(occ.getBoundingBox(1, ref))
    tb = max(100 * tol, 1e-3 * float(np.linalg.norm(rb[3:] - rb[:3])))
    if np.any(cb[:3] < rb[:3] - tb) or np.any(cb[3:] > rb[3:] + tb):
        return False
    lo, hi = gmsh.model.getParametrizationBounds(1, cand)
    for a in np.linspace(0.05, 0.95, 7):
        x = np.asarray(gmsh.model.getValue(1, cand, [lo[0] + a * (hi[0] - lo[0])]))
        q = np.asarray(gmsh.model.getClosestPoint(1, ref, list(x))[0])
        if np.linalg.norm(x - q) > tol:
            return False
    return True


def _snap_point(p: np.ndarray, vols: list[int], snap: float, reach: float) -> tuple[np.ndarray | None, float]:
    """Point on the surface of the part nearest to p: vertex first (within snap), then edge (within snap), then face (within reach).
    Returns (point, distance from p), or (None, distance of the nearest face).
    """
    surfs = _boundary_surfaces(vols)
    curves = _boundary_curves(vols)
    pts = sorted({abs(t) for d, t in gmsh.model.getBoundary([(1, c) for c in curves], combined=False,
                                                             oriented=False) if d == 0}) if curves else []
    for t in pts:
        x = np.asarray(gmsh.model.getValue(0, t, []))
        if np.linalg.norm(x - p) <= snap:
            return x, float(np.linalg.norm(x - p))
    best, best_d = None, np.inf
    for c in curves:
        q = np.asarray(gmsh.model.getClosestPoint(1, c, list(p))[0])
        d = float(np.linalg.norm(q - p))
        if d < best_d:
            best, best_d = q, d
    if best is not None and best_d <= snap:
        return best, best_d
    best, best_d = None, np.inf
    for t in surfs:
        q = np.asarray(gmsh.model.getClosestPoint(2, t, list(p))[0])
        d = float(np.linalg.norm(q - p))
        if d < best_d and gmsh.model.isInside(2, t, list(q)):
            best, best_d = q, d
    if best is None or best_d > reach:
        return None, best_d
    return best, best_d


def _boundary_surfaces(vols: list[int]) -> list[int]:
    if not vols:
        return []
    bnd = gmsh.model.getBoundary([(3, v) for v in vols], combined=False, oriented=False)
    return sorted({abs(t) for d, t in bnd if d == 2})


def _points_on_surface(tag: int, n: int = 6) -> np.ndarray:
    """Several points lying on the (trimmed) face; for narrow frames with a window a denser grid."""
    lo, hi = gmsh.model.getParametrizationBounds(2, tag)
    pts = []
    for nn in (n, 3 * n):
        for a in np.linspace(0.02, 0.98, nn):
            for b in np.linspace(0.02, 0.98, nn):
                uv = [lo[0] + a * (hi[0] - lo[0]), lo[1] + b * (hi[1] - lo[1])]
                if gmsh.model.isInside(2, tag, uv, parametric=True):
                    pts.append(gmsh.model.getValue(2, tag, uv))
        if len(pts) >= 3:
            break
    if len(pts) > 40:
        pts = pts[:: len(pts) // 40 + 1]
    if not pts:
        c = gmsh.model.occ.getCenterOfMass(2, tag)
        pts.append(gmsh.model.getClosestPoint(2, tag, list(c))[0])
    return np.asarray(pts, float).reshape(-1, 3)


def _lies_on(cand: int, ref: int, tol: float) -> bool:
    """Does the face cand lie entirely on the face ref? (envelope + points on the face)"""
    cb = np.asarray(gmsh.model.occ.getBoundingBox(2, cand))
    rb = np.asarray(gmsh.model.occ.getBoundingBox(2, ref))
    ac, ar = gmsh.model.occ.getMass(2, cand), gmsh.model.occ.getMass(2, ref)
    if abs(ac - ar) <= 1e-7 * ar and np.linalg.norm(np.subtract(gmsh.model.occ.getCenterOfMass(2, cand),
                                                                gmsh.model.occ.getCenterOfMass(2, ref))) <= tol:
        return True                                   # face not changed by fragmentation
    # envelope only roughly: fragmentation enlarges the edge tolerance and the OCC envelope includes it (verified 12. 10. 2026)
    tb = max(100 * tol, 1e-3 * float(np.linalg.norm(rb[3:] - rb[:3])))
    if np.any(cb[:3] < rb[:3] - tb) or np.any(cb[3:] > rb[3:] + tb):
        return False
    for p in _points_on_surface(cand):
        q = np.asarray(gmsh.model.getClosestPoint(2, ref, list(p))[0])
        if np.linalg.norm(p - q) > tol:
            return False
        if not gmsh.model.isInside(2, ref, list(p)):
            return False
    return True


def run(job: dict) -> dict:
    t0 = time.time()
    s = job["settings"]
    report: dict = {"warnings": [], "unmatched": [], "selections": {}}
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        # 0 = 1 thread: the mesh is then THE SAME at every meshing (HXT / Delaunay in several threads gives a different mesh each time –
        # example 1: 16 711–16 899 elements, print safety at the corner 1.55–1.91); example 2 only ~1 s slower. More threads
        # = MeshSettings.threads > 1 (large models, speed before repeatability)
        gmsh.option.setNumber("General.NumThreads", int(s.get("threads") or 1))
        gmsh.model.add("fea")
        occ = gmsh.model.occ

        # ---------------------------------------------------------- import
        part_vols: dict[str, list[int]] = {}
        for p in job["parts"]:
            dt = occ.importShapes(p["brep"], highestDimOnly=True)
            vols = [t for d, t in dt if d == 3]
            if not vols:
                raise RuntimeError(f"Part '{p['name']}' contains no body")
            part_vols[p["name"]] = vols
        occ.synchronize()
        all_vols = [v for vs in part_vols.values() for v in vs]
        bb = gmsh.model.getBoundingBox(-1, -1)
        model_size = float(np.linalg.norm(np.subtract(bb[3:], bb[:3])))
        tol = max(1e-6 * model_size, 1e-6)

        # ---------------------------------------------------------- selectors → faces
        sel_surfs: dict[str, list[int]] = {}
        surf_part: dict[int, str] = {}
        for sname, sels in job["selections"].items():
            tags = []
            for sel in sels:
                cands = _boundary_surfaces(part_vols.get(sel["part"], []))
                scored = sorted((_score(sel, _props(t)), t) for t in cands)
                if scored and scored[0][0] < MATCH_TOL:
                    tags.append(scored[0][1])
                    surf_part[scored[0][1]] = sel["part"]
                else:
                    report["unmatched"].append({"selection": sname, "selector": sel,
                                                "best_score": scored[0][0] if scored else None})
            sel_surfs[sname] = tags

        # ---------------------------------------------------------- edge selectors → curves
        sel_curves: dict[str, list[int]] = {}
        curve_part: dict[int, str] = {}
        curve_len: dict[int, float] = {}
        for sname, sels in job.get("edges", {}).items():
            tags = []
            for sel in sels:
                cands = _boundary_curves(part_vols.get(sel["part"], []))
                scored = sorted((_score(sel, _curve_props(t)), t) for t in cands)
                if scored and scored[0][0] < MATCH_TOL:
                    tags.append(scored[0][1])
                    curve_part[scored[0][1]] = sel["part"]
                    curve_len[scored[0][1]] = occ.getMass(1, scored[0][1])
                else:
                    report["unmatched"].append({"selection": sname, "selector": sel, "kind": "edge",
                                                "best_score": scored[0][0] if scored else None})
            sel_curves[sname] = tags

        # ---------------------------------------------------------- points (loads at a point) → vertices
        point_tags: list[tuple[str, int, int]] = []          # (selection, order of the point in the selection, Gmsh point tag)
        snap, reach = max(10 * tol, 1e-4), max(1e-3 * model_size, 0.05)
        for sname, sels in job.get("points", {}).items():
            for k, sel in enumerate(sels):
                p0 = np.asarray(sel["point"], float)
                q, d = _snap_point(p0, part_vols.get(sel["part"], []), snap, reach)
                if q is None:
                    report["unmatched"].append({"selection": sname, "selector": sel, "kind": "point",
                                                "best_score": d})
                    continue
                if d > 10 * tol:
                    report["warnings"].append(f"Selection '{sname}': point ({', '.join((f'{v:.4g}' for v in p0))}) moved onto the surface of part '{sel['part']}' by {d:.3g} mm")
                point_tags.append((sname, k, occ.addPoint(*map(float, q))))
        if point_tags:
            occ.synchronize()

        # ---------------------------------------------------------- discs and imprint tools (bolts, face splitting)
        imprints = job.get("imprints", [])
        disks = []
        for imp in imprints:
            n = np.asarray(imp["normal"], float)
            n_norm = np.linalg.norm(n)
            n = n / n_norm if n_norm > 1e-12 else np.array([0.0, 0.0, 1.0])
            xa = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
            xa /= np.linalg.norm(xa)
            c = np.asarray(imp["center"], float)
            shape = imp.get("shape", "circle")
            if shape == "rectangle":
                dx = float(imp.get("dx", 20.0))
                dy = float(imp.get("dy", 20.0))
                ya = np.cross(n, xa)
                ya /= np.linalg.norm(ya)
                p0 = c - 0.5 * dx * xa - 0.5 * dy * ya
                p1 = c + 0.5 * dx * xa - 0.5 * dy * ya
                p2 = c + 0.5 * dx * xa + 0.5 * dy * ya
                p3 = c - 0.5 * dx * xa + 0.5 * dy * ya
                pt0 = occ.addPoint(float(p0[0]), float(p0[1]), float(p0[2]))
                pt1 = occ.addPoint(float(p1[0]), float(p1[1]), float(p1[2]))
                pt2 = occ.addPoint(float(p2[0]), float(p2[1]), float(p2[2]))
                pt3 = occ.addPoint(float(p3[0]), float(p3[1]), float(p3[2]))
                l0 = occ.addLine(pt0, pt1)
                l1 = occ.addLine(pt1, pt2)
                l2 = occ.addLine(pt2, pt3)
                l3 = occ.addLine(pt3, pt0)
                w = occ.addWire([l0, l1, l2, l3])
                disks.append(occ.addPlaneSurface([w]))
            else:
                r = float(imp.get("radius", 10.0))
                disks.append(occ.addDisk(float(c[0]), float(c[1]), float(c[2]), r, r,
                                         zAxis=n.tolist(), xAxis=xa.tolist()))
        if disks:
            occ.synchronize()

        # ---------------------------------------------------------- fragmentation of the assembly
        # parts "separate" are not fragmented with the others (own faces, independent mesh → connection by *TIE, fea_core.ties)
        separate = set(s.get("separate") or [])
        for imp in imprints:
            if imp["part"] in separate:
                raise RuntimeError(f"Part '{imp['part']}' is meshed separately (*TIE) and carries a bolt imprint – mesh bolted parts together with the assembly")
        for sname, k, _ in point_tags:
            if job["points"][sname][k]["part"] in separate:
                raise RuntimeError(f"Selection '{sname}': a point on a separately meshed part (*TIE) is not yet supported")
        main_vols = [v for n, vs in part_vols.items() if n not in separate for v in vs]
        vol_part: dict[int, str] = {}
        disk_copies: dict[int, int] = {}
        if len(main_vols) > 1 or disks or point_tags:
            originals = sorted({t for ts in sel_surfs.values() for t in ts})
            copies = {}
            for t in originals:
                copies[t] = occ.copy([(2, t)])[0][1]
            for i, t in enumerate(disks):
                disk_copies[i] = occ.copy([(2, t)])[0][1]
            curve_copies = {t: occ.copy([(1, t)])[0][1] for t in sorted({t for ts in sel_curves.values() for t in ts})}
            orig_area = {t: occ.getMass(2, t) for t in originals}
            vol_names = [n for n, vs in part_vols.items() if n not in separate for _ in vs]
            _, out_map = occ.fragment([(3, v) for v in main_vols],
                                      [(2, t) for t in disks] + [(0, t) for _, _, t in point_tags])
            occ.synchronize()
            n0 = len(main_vols) + len(disks)
            frag_points = []
            for (sname, k, _), children in zip(point_tags, out_map[n0:]):
                pts = [t for d, t in children if d == 0]
                if not pts:
                    report["unmatched"].append({"selection": sname, "selector": job["points"][sname][k],
                                                "kind": "point", "best_score": None})
                    continue
                frag_points.append((sname, k, pts[0]))
            point_tags = frag_points
            new_part_vols: dict[str, list[int]] = {n: (list(vs) if n in separate else [])
                                                   for n, vs in part_vols.items()}
            for v in (v for n in separate for v in part_vols.get(n, [])):
                vol_part[v] = next(n for n in separate if v in part_vols.get(n, []))
            for name, children in zip(vol_names, out_map[:len(main_vols)]):
                for d, t in children:
                    if d != 3:
                        continue
                    if t in vol_part and vol_part[t] != name:
                        report["warnings"].append(
                            f"Parts '{vol_part[t]}' and '{name}' overlap (volume {t}); overlap assigned to part '{vol_part[t]}'")
                        continue
                    vol_part[t] = name
                    new_part_vols[name].append(t)
            part_vols = new_part_vols
            # remapping of the selected faces
            for sname in sel_surfs:
                new_tags = []
                for sel_t in sel_surfs[sname]:
                    ref = copies[sel_t]
                    cands = _boundary_surfaces(part_vols[surf_part[sel_t]])
                    found = [c for c in cands if _lies_on(c, ref, tol)]
                    area = sum(occ.getMass(2, c) for c in found)
                    if not found or abs(area - orig_area[sel_t]) > 1e-3 * orig_area[sel_t]:
                        report["warnings"].append(
                            f"Selection '{sname}': after fragmentation found {area:.4g} of {orig_area[sel_t]:.4g} mm²")
                    new_tags.extend(found)
                sel_surfs[sname] = sorted(set(new_tags))
            for sname in sel_curves:                       # remapping of the selected edges (fragmentation splits them)
                new_tags = []
                for sel_t in sel_curves[sname]:
                    ref = curve_copies[sel_t]
                    cands = _boundary_curves(part_vols[curve_part[sel_t]])
                    found = [c for c in cands if c not in curve_copies.values() and _curve_lies_on(c, ref, tol)]
                    length = sum(occ.getMass(1, c) for c in found)
                    if not found or abs(length - curve_len[sel_t]) > 1e-3 * curve_len[sel_t]:
                        report["warnings"].append(
                            f"Edge selection '{sname}': after fragmentation found {length:.4g} of {curve_len[sel_t]:.4g} mm")
                    new_tags.extend(found)
                sel_curves[sname] = sorted(set(new_tags))
            # remnants of discs outside the bodies (above a hole, beyond the edge of a part) removed
            bounding = set(_boundary_surfaces([v for vs in part_vols.values() for v in vs]))
            keep = set(copies.values()) | set(disk_copies.values())
            dangling = [(2, t) for _, t in gmsh.model.getEntities(2) if t not in bounding and t not in keep]
            if dangling:
                occ.remove(dangling, recursive=True)
                occ.synchronize()
            # imprinted annuli
            for i, imp in enumerate(imprints):
                cands = _boundary_surfaces(part_vols.get(imp["part"], []))
                found = [c for c in cands if _lies_on(c, disk_copies[i], tol)]
                if not found:
                    report["warnings"].append(f"Imprint '{imp['name']}' on part '{imp['part']}' failed")
                sel_surfs[imp["name"]] = found
            occ.remove([(2, t) for t in list(copies.values()) + list(disk_copies.values())], recursive=True)
            if curve_copies:
                occ.remove([(1, t) for t in curve_copies.values()], recursive=True)
            occ.synchronize()
        else:
            for n, vs in part_vols.items():
                for v in vs:
                    vol_part[v] = n

        # ---------------------------------------------------------- mesh size
        size = float(s.get("size", 5.0))
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(s.get("min_size") or size / 20))
        cp = int(s.get("curvature_points", 12) or 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", cp)
        gmsh.option.setNumber("Mesh.ElementOrder", int(s.get("order", 2)))
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        if s.get("algorithm3d"):                          # 1 Delaunay (default), 10 HXT (parallel)
            gmsh.option.setNumber("Mesh.Algorithm3D", int(s["algorithm3d"]))
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)  # only when needed, see below (takes ~70 % of the time)
        fields = []

        def threshold(tags, h, dist_max):
            """Size h on faces tags, transition to the global size at distance dist_max."""
            fd = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", tags)
            ft = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(ft, "InField", fd)
            gmsh.model.mesh.field.setNumber(ft, "SizeMin", h)
            gmsh.model.mesh.field.setNumber(ft, "SizeMax", size)
            gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(ft, "DistMax", dist_max)
            fields.append(ft)

        def threshold_curves(tags, h, dist_max):
            """Size h on curves tags, transition to the global size at distance dist_max."""
            fd = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fd, "CurvesList", tags)
            ft = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(ft, "InField", fd)
            gmsh.model.mesh.field.setNumber(ft, "SizeMin", h)
            gmsh.model.mesh.field.setNumber(ft, "SizeMax", size)
            gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(ft, "DistMax", dist_max)
            fields.append(ft)
        for loc in s.get("local", []):
            if loc.get("center") is not None:              # sphere for refinement (convergence, neighbourhood of a point)
                h, r = float(loc["size"]), float(loc["radius"])
                if not (0 < h < size and r > 0):
                    continue
                fb = gmsh.model.mesh.field.add("Ball")
                c = [float(v) for v in loc["center"]]
                gmsh.model.mesh.field.setNumber(fb, "XCenter", c[0])
                gmsh.model.mesh.field.setNumber(fb, "YCenter", c[1])
                gmsh.model.mesh.field.setNumber(fb, "ZCenter", c[2])
                gmsh.model.mesh.field.setNumber(fb, "Radius", r)
                gmsh.model.mesh.field.setNumber(fb, "Thickness", max(r, 2 * size))   # smooth transition
                gmsh.model.mesh.field.setNumber(fb, "VIn", h)
                gmsh.model.mesh.field.setNumber(fb, "VOut", size)
                fields.append(fb)
                continue
            tags = sel_surfs.get(loc["selection"], [])
            if tags and loc.get("size") is not None:
                threshold(tags, float(loc["size"]), 4 * size)
            ctags = sel_curves.get(loc["selection"], [])
            if ctags:
                if loc.get("divisions"):
                    nd = int(loc["divisions"])
                    for ct in ctags:
                        gmsh.model.mesh.setTransfiniteCurve(ct, nd + 1)
                elif loc.get("size") is not None:
                    threshold_curves(ctags, float(loc["size"]), 4 * size)
        for imp in imprints:                              # bolts: fine in the bearing face, transition within 2r
            tags = sel_surfs.get(imp["name"], [])
            h = float(imp.get("size") or 0)
            if tags and 0 < h < size:
                if imp.get("shape") == "rectangle":      # rectangle: half of the diagonal
                    r_eff = 0.5 * float(np.hypot(imp.get("dx", 10.0), imp.get("dy", 10.0)))
                else:
                    r_eff = float(imp.get("radius", 0.0) or 0.5 * max(imp.get("dx", 10.0), imp.get("dy", 10.0)))
                threshold(tags, h, 2.0 * r_eff)
        for vs in job.get("volume_sizes", []):            # welds: element size in the whole body
            vols = part_vols.get(vs["part"], [])
            h = float(vs["size"])
            if not vols or h >= size:
                continue
            fc = gmsh.model.mesh.field.add("Constant")
            gmsh.model.mesh.field.setNumber(fc, "VIn", h)
            gmsh.model.mesh.field.setNumber(fc, "VOut", 1e22)
            gmsh.model.mesh.field.setNumbers(fc, "VolumesList", vols)
            gmsh.model.mesh.field.setNumber(fc, "IncludeBoundary", 1)
            fields.append(fc)
        if fields:
            fm = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(fm, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(fm)

        # ---------------------------------------------------------- mesh
        gmsh.logger.start()
        gen_errors: list[str] = []
        try:
            gmsh.model.mesh.generate(3)
        except Exception as e:  # noqa: BLE001 – HXT sometimes fails on complicated geometry → Delaunay
            if int(s.get("algorithm3d") or 1) == 1:
                raise
            gen_errors.append(f"HXT: {e}")
            report["warnings"].append(f"HXT algorithm failed ({e}); mesh generated with Delaunay")
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            try:
                gmsh.model.mesh.generate(3)
            except Exception as e2:  # noqa: BLE001 – included in the error below when no volume mesh exists
                gen_errors.append(f"Delaunay: {e2}")
        etype = 11 if int(s.get("order", 2)) == 2 else 4
        if not any(len(gmsh.model.mesh.getElementsByType(etype, v)[0]) for _, v in gmsh.model.getEntities(3)):
            msgs = [m for m in gmsh.logger.get() if "rror" in m]
            hint = next((m for m in msgs if "intersect" in m or "PLC" in m), msgs[-1] if msgs else "")
            tried = f" (mesher errors: {'; '.join(gen_errors)})" if gen_errors else ""
            raise RuntimeError(f"The volume mesh was not generated – the geometry probably self-intersects or parts touch tangentially (cylinder in a slot, coincident faces). Gmsh: {hint}{tried}")
        gmsh.logger.stop()
        min_sj = _min_quality(etype)
        if etype == 11 and min_sj < HO_OPT_BELOW and s.get("optimize_high_order", True):
            report["warnings"].append(f"Curved elements min. SJ {min_sj:.3g} – higher-order optimization")
            gmsh.model.mesh.optimize("HighOrder")
            min_sj = _min_quality(etype)
        if min_sj <= 0.0 and etype == 11:
            report["warnings"].append(
                f"Curved elements with non-positive Jacobian (min SJ={min_sj:.3g}); mesh regenerated with straight edges")
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
            gmsh.model.mesh.generate(3)
            min_sj = _min_quality(etype)
        report["min_scaled_jacobian"] = min_sj

        # ---------------------------------------------------------- output
        ntags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(ntags)
        ntags = ntags[order]
        nodes = xyz.reshape(-1, 3)[order]
        tag2idx = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        tag2idx[ntags] = np.arange(len(ntags))

        part_names = list(part_vols.keys())
        tets, tet_part = [], []
        npe = 10 if etype == 11 else 4
        for pi, name in enumerate(part_names):
            for v in part_vols[name]:
                _, enodes = gmsh.model.mesh.getElementsByType(etype, v)
                conn = tag2idx[enodes.astype(np.int64)].reshape(-1, npe)
                tets.append(conn)
                tet_part.append(np.full(len(conn), pi, dtype=np.int32))
        tets = np.vstack(tets)
        tet_part = np.concatenate(tet_part)
        # remove unused nodes (copies of faces etc.)
        used = np.unique(tets)
        remap = np.full(len(nodes), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        nodes = nodes[used]
        tets = remap[tets]

        arrays = {"nodes": nodes, "tets": tets, "tet_part": tet_part}
        ftype = 9 if etype == 11 else 2
        fpe = 6 if etype == 11 else 3
        for i, (sname, tags) in enumerate(sel_surfs.items()):
            tris = [np.zeros((0, fpe), dtype=np.int64)]
            for t in tags:
                _, fn = gmsh.model.mesh.getElementsByType(ftype, t)
                tris.append(remap[tag2idx[fn.astype(np.int64)]].reshape(-1, fpe))
            tri = np.vstack(tris)
            if np.any(tri < 0):
                report["warnings"].append(f"Selection '{sname}': face is not on the body surface")
                tri = tri[np.all(tri >= 0, axis=1)]
            arrays[f"sel_{i}"] = tri
            report["selections"][sname] = {"key": f"sel_{i}", "surfaces": tags,
                                           "faces": int(len(tri))}
        ltype, lpe = (8, 3) if etype == 11 else (1, 2)       # line3 (ends, midpoint) | line2
        report["edges"] = {}
        for i, (sname, tags) in enumerate(sel_curves.items()):
            segs = [np.zeros((0, lpe), dtype=np.int64)]
            for t in tags:
                _, en = gmsh.model.mesh.getElementsByType(ltype, t)
                segs.append(remap[tag2idx[en.astype(np.int64)]].reshape(-1, lpe))
            seg = np.vstack(segs)
            if np.any(seg < 0):
                report["warnings"].append(f"Edge selection '{sname}': edge is not on the body surface")
                seg = seg[np.all(seg >= 0, axis=1)]
            arrays[f"edge_{i}"] = seg
            report["edges"][sname] = {"key": f"edge_{i}", "curves": tags, "segments": int(len(seg))}
        report["points"] = {}
        by_sel: dict[str, list[int]] = {}
        for sname, k, t in point_tags:
            nt, _, _ = gmsh.model.mesh.getNodes(0, t)
            idx = int(remap[tag2idx[int(nt[0])]]) if len(nt) else -1
            if idx < 0:
                report["warnings"].append(f"Point selection '{sname}': point {k + 1} has no node in the body mesh")
                continue
            by_sel.setdefault(sname, []).append(idx)
        for i, sname in enumerate(job.get("points", {})):
            arrays[f"pt_{i}"] = np.asarray(by_sel.get(sname, []), dtype=np.int64)
            report["points"][sname] = {"key": f"pt_{i}", "nodes": len(by_sel.get(sname, []))}
        np.savez(job["output"], **arrays)
        report.update({
            "part_names": part_names, "nodes": int(len(nodes)), "elements": int(len(tets)),
            "element_type": "C3D10" if etype == 11 else "C3D4", "time_s": time.time() - t0,
            "ok": True,
        })
    finally:
        gmsh.finalize()
    return report


def _min_quality(etype: int) -> float:
    q_min = np.inf
    for _, v in gmsh.model.getEntities(3):
        etags, _ = gmsh.model.mesh.getElementsByType(etype, v)
        if len(etags):
            q = gmsh.model.mesh.getElementQualities(etags, "minSJ")
            q_min = min(q_min, float(np.min(q)))
    return q_min


def main(argv: list[str]) -> int:
    job_path = argv[1]
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    try:
        report = run(job)
    except Exception as e:  # message for the main process
        import traceback
        report = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
    with open(job["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
