"""Gmsh meshing process for the 2D section – started separately: python -m fea_core.twod.worker job.json

Input (job.json):
  parts:    [{"name", "brep"}]       faces of the section in the plane x-y (z = 0), mm
  settings: {"size", "min_size", "curvature_points", "order" (2), "local": [{"points": [[x, y], …], "size"}
            (refinement at the curves passing through the points) | {"center": [x, y], "radius", "size"} (circle)]}
  output:   .npz, report: .json
Output (npz): nodes (N, 2), tri (M, 6) tri6 in the order of Gmsh = ccx (corners, midpoints 1-2, 2-3, 3-1; the orientation is corrected by the main
process), tri_part (M,), seg (K, 3) line3 of the boundary curves (ends, midpoint), seg_curve (K,) order of the curve,
curve_pts (C, 7, 2) sampling points of the curves (exact geometry). A boundary curve = adjoins a single face
(the interfaces of the parts are meshed conformingly – occ.fragment – and do not belong to the boundary).

The module may depend only on gmsh and numpy (a clean process, without OCP).
"""
from __future__ import annotations

import json
import sys
import time

import gmsh
import numpy as np


def _curve_samples(tag: int, n: int = 7) -> np.ndarray:
    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
    ts = [lo[0] + a * (hi[0] - lo[0]) for a in np.linspace(0.0, 1.0, n)]
    xyz = np.asarray(gmsh.model.getValue(1, tag, ts)).reshape(-1, 3)
    return xyz[:, :2]


def _min_quality() -> float:
    q = np.inf
    for _, s in gmsh.model.getEntities(2):
        et, _ = gmsh.model.mesh.getElementsByType(9, s)
        if len(et):
            q = min(q, float(np.min(gmsh.model.mesh.getElementQualities(et, "minSJ"))))
    return q


def run(job: dict) -> dict:
    t0 = time.time()
    s = job["settings"]
    report: dict = {"warnings": []}
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.option.setNumber("General.NumThreads", 1)          # repeatable mesh
        gmsh.model.add("fea2d")
        occ = gmsh.model.occ
        part_surfs: list[list[int]] = []
        for p in job["parts"]:
            dt = occ.importShapes(p["brep"], highestDimOnly=True)
            surfs = [t for d, t in dt if d == 2]
            if not surfs:
                raise RuntimeError(f"The section of part '{p['name']}' contains no face")
            part_surfs.append(surfs)
        occ.synchronize()
        flat = [(2, t) for ss in part_surfs for t in ss]
        if len(flat) > 1:                                         # conforming mesh between the parts
            out, omap = occ.fragment(flat[:1], flat[1:])
            occ.synchronize()
            new, k = [], 0
            for ss in part_surfs:
                acc = []
                for _ in ss:
                    acc += [t for d, t in omap[k] if d == 2]
                    k += 1
                new.append(acc)
            owner: dict[int, int] = {}
            for pi, ss in enumerate(new):
                for t in ss:
                    if t in owner and owner[t] != pi:
                        report["warnings"].append(
                            f"Sections of parts '{job['parts'][owner[t]]['name']}' and '{job['parts'][pi]['name']}' overlap (parts intersect in CAD) – the common region belongs to the first")
                    owner.setdefault(t, pi)
            part_surfs = [[t for t in ss if owner[t] == pi] for pi, ss in enumerate(new)]
            part_surfs = [list(dict.fromkeys(ss)) for ss in part_surfs]
        bb = gmsh.model.getBoundingBox(-1, -1)
        model_size = float(np.linalg.norm(np.subtract(bb[3:], bb[:3])))
        tol = max(1e-6 * model_size, 1e-7)

        size = float(s.get("size", 5.0))
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(s.get("min_size") or size / 50))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(s.get("curvature_points", 12) or 0))
        gmsh.option.setNumber("Mesh.ElementOrder", int(s.get("order", 2)))
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)                # Frontal-Delaunay

        all_curves = sorted({abs(t) for _, t in gmsh.model.getBoundary(
            [(2, t) for ss in part_surfs for t in ss], combined=False, oriented=False)})
        fields = []
        for loc in s.get("local", []):
            h = float(loc["size"])
            if not 0 < h < size:
                continue
            if loc.get("center") is not None:
                fb = gmsh.model.mesh.field.add("Ball")
                c = [float(v) for v in loc["center"]]
                r = float(loc["radius"])
                gmsh.model.mesh.field.setNumber(fb, "XCenter", c[0])
                gmsh.model.mesh.field.setNumber(fb, "YCenter", c[1])
                gmsh.model.mesh.field.setNumber(fb, "ZCenter", 0.0)
                gmsh.model.mesh.field.setNumber(fb, "Radius", r)
                gmsh.model.mesh.field.setNumber(fb, "Thickness", max(r, 2 * size))
                gmsh.model.mesh.field.setNumber(fb, "VIn", h)
                gmsh.model.mesh.field.setNumber(fb, "VOut", size)
                fields.append(fb)
                continue
            tags = []
            for q in loc.get("points", []):
                for c in all_curves:
                    cp = np.asarray(gmsh.model.getClosestPoint(1, c, [float(q[0]), float(q[1]), 0.0])[0])
                    if np.hypot(cp[0] - q[0], cp[1] - q[1]) <= 100 * tol:
                        tags.append(c)
            if not tags:
                report["warnings"].append(f"Local refinement '{loc.get('selection', '')}': section curve not found")
                continue
            fd = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fd, "CurvesList", sorted(set(tags)))
            gmsh.model.mesh.field.setNumber(fd, "Sampling", 100)
            ft = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(ft, "InField", fd)
            gmsh.model.mesh.field.setNumber(ft, "SizeMin", h)
            gmsh.model.mesh.field.setNumber(ft, "SizeMax", size)
            gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(ft, "DistMax", 4 * size)
            fields.append(ft)
        if fields:
            fm = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(fm, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(fm)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        gmsh.model.mesh.generate(2)
        q = _min_quality()
        if q <= 0.0:
            report["warnings"].append(f"Curved elements with a non-positive Jacobian (min SJ {q:.3g}) – remeshing with straight edges")
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
            gmsh.model.mesh.generate(2)
            q = _min_quality()
        report["min_scaled_jacobian"] = q

        ntags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(ntags)
        ntags = ntags[order]
        nodes = xyz.reshape(-1, 3)[order][:, :2]
        tag2idx = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        tag2idx[ntags] = np.arange(len(ntags))
        tri, tri_part = [], []
        for pi, ss in enumerate(part_surfs):
            for t in ss:
                _, en = gmsh.model.mesh.getElementsByType(9, t)
                conn = tag2idx[en.astype(np.int64)].reshape(-1, 6)
                tri.append(conn)
                tri_part.append(np.full(len(conn), pi, dtype=np.int32))
        if not tri or not sum(len(c) for c in tri):
            raise RuntimeError("2D mesh was not created")
        tri = np.vstack(tri)
        tri_part = np.concatenate(tri_part)
        used = np.unique(tri)
        remap = np.full(len(nodes), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        nodes = nodes[used]
        tri = remap[tri]

        # boundary curves: adjoin a single face of the mesh
        surf_set = {t for ss in part_surfs for t in ss}
        seg, seg_curve, pts, ctags = [], [], [], []
        for c in all_curves:
            up, _ = gmsh.model.getAdjacencies(1, c)
            if sum(1 for u in up if int(u) in surf_set) != 1:
                continue
            _, en = gmsh.model.mesh.getElementsByType(8, c)
            if not len(en):
                continue
            conn = remap[tag2idx[en.astype(np.int64)]].reshape(-1, 3)
            seg.append(conn)
            seg_curve.append(np.full(len(conn), len(ctags), dtype=np.int64))
            pts.append(_curve_samples(c))
            ctags.append(int(c))
        arrays = {"nodes": nodes, "tri": tri, "tri_part": tri_part,
                  "seg": np.vstack(seg) if seg else np.zeros((0, 3), np.int64),
                  "seg_curve": np.concatenate(seg_curve) if seg_curve else np.zeros(0, np.int64),
                  "curve_pts": np.array(pts) if pts else np.zeros((0, 7, 2))}
        np.savez(job["output"], **arrays)
        report.update({"part_names": [p["name"] for p in job["parts"]], "nodes": int(len(nodes)),
                       "elements": int(len(tri)), "element_type": "tri6", "curves": ctags,
                       "time_s": time.time() - t0, "ok": True})
    finally:
        gmsh.finalize()
    return report


def main(argv: list[str]) -> int:
    with open(argv[1], "r", encoding="utf-8") as f:
        job = json.load(f)
    try:
        report = run(job)
    except Exception as e:  # noqa: BLE001 – message for the main process
        import traceback
        report = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
    with open(job["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
