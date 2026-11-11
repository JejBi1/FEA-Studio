"""Mesh of a section for `sectionproperties` from Gmsh (instead of Triangle).

`sectionproperties` (MIT) meshes the section with the Triangle library (J. R. Shewchuk) through the package `cytriangle`, whose licence
forbids including it in commercial products without the author's consent. FEA Studio therefore does not use it: `install()` puts into
`sys.modules["cytriangle"]` this module with a function `triangulate` of the same interface, which creates the mesh in Gmsh
(separate process `section_mesh_worker`, without OCP). It is always done – even when the real cytriangle is installed –
so that the results do not depend on the environment.

Interface used by sectionproperties 3.10 (`pre.create_mesh`, the only place of the call):
  triangulate({"vertices", "segments", "holes", "regions": [[x, y, attribute, max. area]]}, "pAo2" | "pq30.0Aao2")
  → {"vertices" (N, 2), "triangles" (M, 6) corners + midpoints of the edges 1-2, 2-0, 0-1, "triangle_attributes" (M, 1)}.
The switch q (quality) only turns on a quality mesh (Gmsh Frontal-Delaunay), the angle is not passed on; a = limit of the element area
per region (as the hard limit of Triangle: the largest element area ≤ the given one).
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

from .. import config


class SectionMeshError(RuntimeError):
    pass


def triangulate(tri: dict, opts: str = "") -> dict:
    """Replacement of `cytriangle.triangulate` (see the module): mesh of the section by Gmsh in a separate process."""
    job = {
        "vertices": np.asarray(tri["vertices"], float).tolist(),
        "segments": np.asarray(tri["segments"], int).tolist(),
        "holes": np.asarray(tri.get("holes", []), float).reshape(-1, 2).tolist(),
        "regions": [[float(v) for v in r] for r in tri.get("regions", [])],
        "coarse": "a" not in opts and "q" not in opts,
    }
    with tempfile.TemporaryDirectory(prefix="fea_section_") as tmp:
        d = Path(tmp)
        (d / "job.json").write_text(json.dumps(job), encoding="utf-8")
        cmd = [sys.executable, "-m", "fea_core.beams.section_mesh_worker", str(d / "job.json"), str(d / "out.npz")]
        rc = config.run_logged(cmd, d / "worker.log", timeout=300)
        if rc != 0 or not (d / "out.npz").exists():
            log = (d / "worker.log").read_text(encoding="utf-8", errors="replace")[-1500:]
            raise SectionMeshError(f"Section mesh (Gmsh) failed (code {rc}): {log.strip()}")
        with np.load(d / "out.npz") as z:
            V, T, att = z["vertices"], z["triangles"], z["attributes"]
    return {"vertices": V, "triangles": T.astype(np.int32), "triangle_attributes": att.astype(float).reshape(-1, 1)}


def install() -> None:
    """Puts this module in as `cytriangle` (before the first import of sectionproperties; repeated calls do no harm)."""
    shim = types.ModuleType("cytriangle")
    shim.triangulate = triangulate
    shim.__doc__ = "Náhrada cytriangle (FEA Studio): síť průřezu z Gmsh, viz fea_core.beams.section_mesh"
    sys.modules["cytriangle"] = shim
    pre = sys.modules.get("sectionproperties.pre.pre")
    if pre is not None:                                       # already imported with the real cytriangle
        pre.triangle = shim


def import_sectionproperties():
    """Imports `sectionproperties` with the mesh from Gmsh (see `install`) and returns the module."""
    install()
    import sectionproperties.pre.pre as pre                    # noqa: F401 – check that the import succeeded
    import sectionproperties
    return sectionproperties
