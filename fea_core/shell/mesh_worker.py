"""Gmsh meshing process for the shells – started separately: python -m fea_core.shell.mesh_worker job.json

Input (job.json):
  parts:    [{"name", "brep"}]        mid faces of the parts (compound of faces, global position, mm)
  settings: {"size", "curvature_points", "threads"}
  output:   .npz, report: .json

Procedure: import of the faces part by part → fragmentation of all faces together (a conforming mesh at the intersections – T joint,
shared edges) → 2D mesh (Frontal-Delaunay), order 2 (tri6, curved) → orientation of the elements according to the normal of the face
in Gmsh (respects the orientation of the OCC face) → writing. The node order of tri6 in Gmsh = ccx S6 (corners, midpoints 12, 23, 31).

The module may depend only on gmsh and numpy (it runs in a clean process – without OCP).
"""
from __future__ import annotations

import json
import sys
import time

import gmsh
import numpy as np


def run(job: dict) -> dict:
    t0 = time.time()
    s = job["settings"]
    report: dict = {"warnings": []}
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.option.setNumber("General.NumThreads", int(s.get("threads") or 1))
        gmsh.model.add("shell")
        occ = gmsh.model.occ
        inputs, owner = [], []
        for pi, p in enumerate(job["parts"]):
            dt = occ.importShapes(p["brep"], highestDimOnly=True)
            surfs = [(d, t) for d, t in dt if d == 2]
            if not surfs:
                raise RuntimeError(f"Part '{p['name']}' has no mid surface")
            inputs += surfs
            owner += [pi] * len(surfs)
        occ.synchronize()
        if len(inputs) > 1:
            _, omap = occ.fragment(inputs, [])
        else:
            omap = [[inputs[0]]]
        occ.synchronize()
        surf_part: dict[int, int] = {}
        for k, new in enumerate(omap):
            for d, t in new:
                if d != 2:
                    continue
                if t in surf_part and surf_part[t] != owner[k]:
                    report["warnings"].append(
                        f"The mid surfaces of parts '{job['parts'][surf_part[t]]['name']}' and '{job['parts'][owner[k]]['name']}' overlap – elements belong to the first part")
                    continue
                surf_part.setdefault(t, owner[k])
        # the remnants (faces without a part) are not meshed
        for d, t in gmsh.model.getEntities(2):
            if t not in surf_part:
                report["warnings"].append(f"Face {t} does not belong to any part after fragmentation – skipped")
        size = float(s["size"])
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(s.get("min_size") or 0.0))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(s.get("curvature_points") or 0))
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
        gmsh.model.mesh.generate(2)

        ntags, coords, _ = gmsh.model.mesh.getNodes()
        nodes_all = coords.reshape(-1, 3)
        idx = np.full(int(ntags.max()) + 1, -1, dtype=np.int64)
        idx[ntags.astype(np.int64)] = np.arange(len(ntags))
        tris, parts, surfs, flipped = [], [], [], 0
        for _, t in gmsh.model.getEntities(2):
            if t not in surf_part:
                continue
            types, etags, enodes = gmsh.model.mesh.getElements(2, t)
            for ty, et, en in zip(types, etags, enodes):
                if ty != 9:                       # 9 = tri6
                    raise RuntimeError(f"Unexpected Gmsh element type {ty} (tri6 expected)")
                tri = idx[en.astype(np.int64)].reshape(-1, 6)
                x = nodes_all[tri[:, :3]]
                n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
                c = x.mean(axis=1)
                par = gmsh.model.getParametrization(2, t, c.ravel().tolist())
                ref = np.asarray(gmsh.model.getNormal(t, par)).reshape(-1, 3)
                flip = np.einsum("ij,ij->i", n, ref) < 0
                tri[flip] = tri[flip][:, [0, 2, 1, 5, 4, 3]]
                flipped += int(flip.sum())
                tris.append(tri)
                parts.append(np.full(len(tri), surf_part[t], dtype=np.int64))
                surfs.append(np.full(len(tri), t, dtype=np.int64))
        if not tris:
            raise RuntimeError("The shell mesh is empty")
        tri = np.vstack(tris)
        used = np.unique(tri)
        remap = np.full(len(nodes_all), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        tri = remap[tri]
        nodes = nodes_all[used]
        qual = []
        for _, t in gmsh.model.getEntities(2):
            types, etags, _ = gmsh.model.mesh.getElements(2, t)
            for et in etags:
                if len(et):
                    qual.append(gmsh.model.mesh.getElementQualities(et, "minSICN"))
        q = np.concatenate(qual) if qual else np.ones(1)
        report.update(ok=True, nodes=int(len(nodes)), elements=int(len(tri)), min_sicn=float(q.min()),
                      flipped=flipped, time_s=time.time() - t0,
                      part_names=[p["name"] for p in job["parts"]])
        np.savez(job["output"], nodes=nodes, tri=tri, tri_part=np.concatenate(parts), tri_surf=np.concatenate(surfs))
        return report
    finally:
        gmsh.finalize()


def main(path: str) -> None:
    job = json.loads(open(path, encoding="utf-8").read())
    try:
        rep = run(job)
    except Exception as e:  # noqa: BLE001 – the error into a message for the main process
        import traceback
        rep = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
    with open(job["report"], "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
