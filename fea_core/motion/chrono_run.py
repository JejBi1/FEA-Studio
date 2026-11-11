"""Project Chrono solver for a mechanism with contact (stage F8.9, part C1) – main process.

Environment `tools/chrono_env` (conda-forge pychrono 10.0.0 without MKL, OpenBLAS; installer module `chrono`). NOT inside
`.venv`: the conda interpreter in `.venv/chrono_env` finds `.venv/pyvenv.cfg` in the parent folder and starts as a venv of the main environment
(sys.prefix = .venv, without its own site-packages – verified 15. 10. 2026). The worker process
`fea_core.motion.chrono_worker` runs with the interpreter of the environment (not the venv) with PYTHONPATH = project root; it must not import
OCP nor scipy (they are not in the environment). The shapes of the contacts = triangulation of the parts of the bodies (OCP BRepMesh) in the axes of the body (from the
centroid, m) – this module prepares them into JOB/tri_<body>.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config

CHRONO_ENV = config.TOOLS_DIR / "chrono_env"


def chrono_python() -> Path:
    return CHRONO_ENV / "python.exe" if config.IS_WIN else CHRONO_ENV / "bin" / "python"


def available() -> bool:
    """Is the Chrono environment installed (fast – it runs nothing)?"""
    if not chrono_python().exists():
        return False
    sp = CHRONO_ENV / ("Lib/site-packages" if config.IS_WIN else "lib/python3.11/site-packages")
    return (sp / "pychrono").is_dir()


INSTALL_HINT = ("The Chrono solver (contact in mechanisms) is not installed – module 'chrono': `python installer/fea_setup.py install chrono` (environment tools/chrono_env, ~0.6 GB)")


def triangulate(shapes, deflection_mm: float = 0.01, angle: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (n, 3) [mm], triangles (m, 3)) of the surface of the OCP shapes with the outward normals (orientation of the face)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    V, F = [], []
    off = 0
    for sh in shapes:
        BRepMesh_IncrementalMesh(sh, float(deflection_mm), False, float(angle), True)
        ex = TopExp_Explorer(sh, TopAbs_FACE)
        while ex.More():
            f = TopoDS.Face(ex.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(f, loc)
            if tri is not None:
                tr = loc.Transformation()
                pts = np.array([[p.X(), p.Y(), p.Z()] for p in (tri.Node(i).Transformed(tr)
                                                                 for i in range(1, tri.NbNodes() + 1))])
                tris = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], int) - 1
                if f.Orientation() == TopAbs_REVERSED:
                    tris = tris[:, ::-1]
                V.append(pts)
                F.append(tris + off)
                off += len(pts)
            ex.Next()
    if not V:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(V), np.vstack(F)


def prepare_contacts(study, parts, job: dict, meta: dict, jd: Path) -> None:
    """Triangulation of the bodies of the contacts into JOB/tri_<i>.npz (m, axes of the body from the centroid) and the records of the contacts into the job (SI)."""
    mech = study.mechanism
    names = [b.name for b in mech.bodies]
    by_name = {p.name: p for p in parts or []}
    done: dict[int, str] = {}
    out = []
    for c in mech.contacts:
        idx = []
        for bn in (c.body_a, c.body_b):
            if bn not in names:
                raise ValueError(f"Contact '{c.name or c.body_a + '–' + c.body_b}': body '{bn}' does not exist")
            i = names.index(bn)
            if i not in done:
                b = mech.bodies[i]
                shapes = [by_name[p].shape for p in b.parts if p in by_name]
                if not shapes:
                    raise ValueError(f"Body '{bn}' has no parts from the geometry – contact needs a CAD shape")
                Vm, Fm = triangulate(shapes, c.deflection)
                com = np.asarray(meta["bodies"][i]["com"], float)
                path = jd / f"tri_{i}.npz"
                np.savez(path, v=(Vm - com) * 1e-3, f=Fm.astype(np.int32))
                done[i] = str(path)
            idx.append(i)
        young = c.young
        if young is None:
            try:
                young = study.material_of(mech.bodies[idx[0]].parts[0]).E
            except (KeyError, IndexError):
                young = 2.1e5
        out.append({"name": c.name or f"{c.body_a}–{c.body_b}", "a": idx[0], "b": idx[1], "mu": float(c.mu),
                    "restitution": float(c.restitution), "rolling": float(c.rolling) * 1e-3, "method": c.method,
                    "young": float(young) * 1e6})
    job["contacts"] = out
    job["tri"] = {str(k): v for k, v in done.items()}
    meta["contacts"] = [{"name": c["name"], "a": c["a"], "b": c["b"]} for c in out]
