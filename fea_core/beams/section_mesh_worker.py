"""Gmsh meshing process for a bar section – started separately: python -m fea_core.beams.section_mesh_worker job.json out.npz

Replacement of the Triangle triangulation (J. R. Shewchuk, package cytriangle – non-commercial licence) in `sectionproperties`.
Input (job.json): the same data that sectionproperties passes to Triangle –
  vertices [[x, y], …], segments [[i, j], …] (edges of the outline), holes [[x, y], …] (points inside the holes),
  regions [[x, y, attribute, max. area], …] (a point inside the region, material number, max. element area; ≤ 0 = coarse mesh),
  coarse (true = without the area and quality limits).
Output (npz): vertices (N, 2), triangles (M, 6) in the order of Triangle (corners 0–2, midpoints 1-2, 2-0, 0-1; anticlockwise),
attributes (M,) number of the region.

The module may depend only on gmsh, shapely and numpy (a clean process, without OCP).
"""
from __future__ import annotations

import json
import sys

import numpy as np

SQ3 = 3.0 ** 0.5


def _key(p, scale: float) -> tuple:
    return (round(p[0] / scale), round(p[1] / scale))


def _polygons(job: dict):
    """Regions bounded by edges: [(Polygon, attribute)] – holes and areas outside the regions are discarded."""
    from shapely.geometry import LineString, Point
    from shapely.ops import polygonize, unary_union
    from shapely.geometry.polygon import orient

    V = np.asarray(job["vertices"], float)
    lines = [LineString([V[a], V[b]]) for a, b in job["segments"]]
    faces = list(polygonize(unary_union(lines)))
    holes = [Point(h) for h in job.get("holes", [])]
    regions = job.get("regions", [])
    out = []
    for f in faces:
        if any(f.contains(h) for h in holes):
            continue
        attr = None
        for r in regions:
            if f.contains(Point(r[0], r[1])):
                attr = int(r[2])
                break
        if attr is None:
            if regions:
                continue
            attr = 0
        out.append((orient(f, 1.0), attr))
    if not out:
        raise RuntimeError("The section outline does not enclose any region")
    return out


def _area_limits(job: dict) -> dict:
    return {int(r[2]): float(r[3]) for r in job.get("regions", []) if len(r) > 3 and float(r[3]) > 0}


def _mesh_once(polys, limits: dict, coarse: bool, c: float, scale: float):
    import gmsh

    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("General.NumThreads", 1)            # repeatable mesh
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)                # Frontal-Delaunay (quality triangles)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)        # midpoints of the edges exactly in the middle of the segments
        gmsh.model.add("section")
        geo = gmsh.model.geo

        def size_of(attr: int) -> float:
            if coarse or attr not in limits:
                return 1e10
            return c * (4.0 * limits[attr] / SQ3) ** 0.5          # equilateral triangle of area A

        pt_size: dict[tuple, float] = {}
        for poly, attr in polys:
            for ring in [poly.exterior, *poly.interiors]:
                for p in list(ring.coords)[:-1]:
                    k = _key(p, scale)
                    pt_size[k] = min(pt_size.get(k, 1e10), size_of(attr))
        pts: dict[tuple, int] = {}
        coord: dict[tuple, tuple] = {}
        for poly, attr in polys:
            for ring in [poly.exterior, *poly.interiors]:
                for p in list(ring.coords)[:-1]:
                    coord.setdefault(_key(p, scale), p)
        for k, p in coord.items():
            pts[k] = geo.addPoint(p[0], p[1], 0.0, pt_size[k])
        lns: dict[tuple, int] = {}

        def line(a: int, b: int) -> int:
            if (a, b) in lns:
                return lns[(a, b)]
            if (b, a) in lns:
                return -lns[(b, a)]
            lns[(a, b)] = geo.addLine(a, b)
            return lns[(a, b)]

        def loop(ring) -> int:
            ids = [pts[_key(p, scale)] for p in list(ring.coords)[:-1]]
            return geo.addCurveLoop([line(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))])

        surf_attr: dict[int, int] = {}
        for poly, attr in polys:
            loops = [loop(poly.exterior)] + [loop(r) for r in poly.interiors]
            surf_attr[geo.addPlaneSurface(loops)] = attr
        geo.synchronize()
        gmsh.model.mesh.generate(2)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        tags = np.asarray(tags, np.int64)
        xyz = np.asarray(xyz, float).reshape(-1, 3)
        order = np.argsort(tags)
        tri_all, att_all = [], []
        for s, attr in surf_attr.items():
            _, _, nodes = gmsh.model.mesh.getElements(2, s)
            if not len(nodes):
                continue
            tri_all.append(np.asarray(nodes[0], np.int64).reshape(-1, 6))
            att_all.append(np.full(len(tri_all[-1]), attr, np.int64))
        T = np.vstack(tri_all)
        att = np.concatenate(att_all)
        T = order[np.searchsorted(tags[order], T)]
        return xyz[:, :2], T, att
    finally:
        gmsh.finalize()


def _finish(V: np.ndarray, T: np.ndarray, att: np.ndarray):
    """Order of the nodes of Gmsh (midpoints 0-1, 1-2, 2-0) → order of Triangle (1-2, 2-0, 0-1), orientation anticlockwise,
    only the nodes used.
    """
    T = T[:, [0, 1, 2, 4, 5, 3]].copy()
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    neg = ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]) < 0
    T[neg] = T[neg][:, [0, 2, 1, 3, 5, 4]]                    # swap corners 1 and 2 (and their midpoints)
    used, inv = np.unique(T, return_inverse=True)
    return V[used], inv.reshape(T.shape).astype(np.int32), att


def run(job: dict):
    coarse = bool(job.get("coarse"))
    polys = _polygons(job)
    V0 = np.asarray(job["vertices"], float)
    scale = max(float(np.ptp(V0[:, 0])), float(np.ptp(V0[:, 1])), 1e-12) * 1e-9
    limits = _area_limits(job)
    c = 0.85
    for _ in range(10):
        V, T, att = _mesh_once(polys, limits, coarse, c, scale)
        if coarse or not limits:
            break
        a, b, d = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        area = 0.5 * np.abs((b - a)[:, 0] * (d - a)[:, 1] - (b - a)[:, 1] * (d - a)[:, 0])
        lim = np.array([limits.get(int(k), np.inf) for k in att])
        if np.all(area <= lim * 1.0000001):                       # Triangle: element area ≤ the given one (hard limit)
            break
        c *= 0.85
    return _finish(V, T, att)


def main(argv: list[str]) -> int:
    with open(argv[1], encoding="utf-8") as f:
        job = json.load(f)
    V, T, att = run(job)
    np.savez(argv[2], vertices=V, triangles=T, attributes=att)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
