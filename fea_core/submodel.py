"""Submodel (Submodeling, C1): local detail of a critical place.

Principle:
  1. A coarse global model is computed for the whole assembly / part.
  2. SubmodelSpec defines a geometric cut-out around the critical place (box or sphere).
  3. From the global geometry the submodel geometry is cut out by a Boolean intersection (BRepAlgoAPI_Common).
  4. The submodel geometry is meshed with a fine tet10 mesh in a separate process (fea_core.mesh.generate_mesh),
     which guarantees a safe run also from side threads and a strict separation of Gmsh from OCP.
  5. The nodes on the cut faces (cut boundary) are identified and the displacements U from the global result are interpolated
     into them by a batch localisation in tet10 (locate_batch + tet10_shape).
  6. The submodel is computed in ccx with the prescribed displacements on the cut faces (*BOUNDARY) with the equilibrium check of the reactions.
  7. Check of the cut: comparison of the stresses on the cut face global vs. submodel (criterion delta_sigma < 10 %).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS_Shape
from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir

from .geometry import Part, write_step, read_step, write_brep
from .results import Results, StepResult, tet10_shape, von_mises
from .study import Material
from .mesh import Mesh, generate_mesh
from .ccx.runner import run_ccx
from .ccx.frd import read_frd
from .ccx.inp import deck_text
from .config import solver_name, EQUILIBRIUM_TOL


@dataclass
class SubmodelSpec:
    """Specifikace submodelu."""
    center: tuple[float, float, float]
    size: tuple[float, float, float] | float  # (dx, dy, dz) for a box or radius r for a sphere
    shape: str = "box"                        # "box" | "sphere"
    mesh_size_max: float = 2.0
    mesh_size_min: float | None = 0.5
    target_step: int = 0
    name: str = "Submodel"


def create_submodel_shape(parent_shape, spec: SubmodelSpec):
    """Creates the OCP shape of the submodel by trimming parent_shape."""
    c = spec.center
    if spec.shape == "sphere":
        r = float(spec.size) if isinstance(spec.size, (int, float)) else float(spec.size[0])
        tool = BRepPrimAPI_MakeSphere(gp_Pnt(c[0], c[1], c[2]), r).Shape()
    else:
        # box
        if isinstance(spec.size, (int, float)):
            dx = dy = dz = float(spec.size)
        else:
            dx, dy, dz = [float(v) for v in spec.size]
        tool = BRepPrimAPI_MakeBox(
            gp_Ax2(gp_Pnt(c[0] - dx / 2, c[1] - dy / 2, c[2] - dz / 2), gp_Dir(0, 0, 1)),
            dx, dy, dz
        ).Shape()

    cut = BRepAlgoAPI_Common(parent_shape, tool)
    cut.Build()
    if not cut.IsDone():
        raise RuntimeError("Boolean intersection for the submodel failed")
    return cut.Shape()


def identify_cut_nodes(nodes: np.ndarray, spec: SubmodelSpec, tol: float = 1e-2) -> np.ndarray:
    """Returns the indices of the submodel nodes lying on the outer cut boundary."""
    c = spec.center
    if spec.shape == "sphere":
        r = float(spec.size) if isinstance(spec.size, (int, float)) else float(spec.size[0])
        dist = np.linalg.norm(nodes - np.array(c), axis=1)
        return np.nonzero(np.abs(dist - r) < tol)[0]
    else:
        if isinstance(spec.size, (int, float)):
            dx = dy = dz = float(spec.size)
        else:
            dx, dy, dz = [float(v) for v in spec.size]
        x0, x1 = c[0] - dx / 2, c[0] + dx / 2
        y0, y1 = c[1] - dy / 2, c[1] + dy / 2
        z0, z1 = c[2] - dz / 2, c[2] + dz / 2

        is_cut = (
            (np.abs(nodes[:, 0] - x0) < tol) | (np.abs(nodes[:, 0] - x1) < tol) |
            (np.abs(nodes[:, 1] - y0) < tol) | (np.abs(nodes[:, 1] - y1) < tol) |
            (np.abs(nodes[:, 2] - z0) < tol) | (np.abs(nodes[:, 2] - z1) < tol)
        )
        return np.nonzero(is_cut)[0]


def locate_batch(results: Results, points: np.ndarray, chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Batch localisation of points in the mesh: returns (element_ids, xis).

    Searches the candidate elements with a cKDTree and solves the barycentric coordinates in batches
    with a vectorised np.linalg.solve.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    k_pts = len(pts)
    if k_pts == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=float)

    if results._tree is None:
        from scipy.spatial import cKDTree
        t4 = results.mesh.tets[:, :4]
        centers = results.mesh.nodes[t4].mean(axis=1)
        results._tree = cKDTree(centers)

    k_cands = min(64, len(results.mesh.tets))
    best_elems = np.empty(k_pts, dtype=np.int64)
    best_xis = np.empty((k_pts, 3), dtype=float)

    nodes = results.mesh.nodes
    tets = results.mesh.tets

    for start_idx in range(0, k_pts, chunk_size):
        end_idx = min(start_idx + chunk_size, k_pts)
        pts_chunk = pts[start_idx:end_idx]
        m = len(pts_chunk)

        _, cand = results._tree.query(pts_chunk, k=k_cands)
        cand = np.asarray(cand)
        if cand.ndim == 1:
            cand = cand[:, None]

        elem_tets = tets[cand, :4]                                           # (m, k_cands, 4)
        tets_coords = nodes[elem_tets]                                      # (m, k_cands, 4, 3)
        a = tets_coords[:, :, 0, :]                                         # (m, k_cands, 3)
        J = (tets_coords[:, :, 1:, :] - a[:, :, None, :]).transpose(0, 1, 3, 2)  # (m, k_cands, 3, 3)
        rhs = (pts_chunk[:, None, :] - a)[:, :, :, None]                    # (m, k_cands, 3, 1)

        try:
            xis = np.linalg.solve(J, rhs)[:, :, :, 0]                       # (m, k_cands, 3)
            err = np.maximum(np.maximum(-xis.min(axis=2), xis.sum(axis=2) - 1.0), 0.0)
            for i in range(m):
                inside = np.nonzero(err[i] <= 1e-9)[0]
                j = int(inside[0]) if len(inside) > 0 else int(np.argmin(err[i]))
                best_elems[start_idx + i] = cand[i, j]
                best_xis[start_idx + i] = xis[i, j]
        except np.linalg.LinAlgError:
            for i in range(m):
                el, xi = results.locate(pts_chunk[i])
                best_elems[start_idx + i] = el
                best_xis[start_idx + i] = xi

    return best_elems, best_xis


def interpolate_cut_displacements(global_results: Results, sub_nodes: np.ndarray,
                                  cut_indices: np.ndarray, step_index: int = 0) -> np.ndarray:
    """Interpolates the displacements U from the global solution into the nodes on the cut of the submodel.

    Uses the shape functions of tet10 elements of the global mesh with batch localisation of the points.
    """
    step = global_results.step(step_index)
    U_g = step.U
    tets_g = global_results.mesh.tets

    cut_pts = sub_nodes[cut_indices]
    best_elems, best_xis = locate_batch(global_results, cut_pts)

    cut_U = np.zeros((len(cut_indices), 3))
    for k in range(len(cut_indices)):
        conn = tets_g[best_elems[k]]
        cut_U[k] = tet10_shape(best_xis[k]) @ U_g[conn]
    return cut_U


def generate_submodel_mesh(shape_or_path: TopoDS_Shape | Path | str, spec: SubmodelSpec,
                           workdir: Path | str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generates a fine tet10 mesh of the submodel using the subprocess mesh worker (without importing gmsh into the main process).

    shape_or_path: TopoDS_Shape or the path to a STEP / BREP file
    spec: SubmodelSpec with the mesh settings
    workdir: working folder (if None, a subfolder or a temporary directory is used)
    """
    if isinstance(shape_or_path, (str, Path)):
        p = Path(shape_or_path)
        if workdir is None:
            workdir = p.parent / "mesh_submodel"
        if p.suffix.lower() == ".brep":
            shape = TopoDS_Shape()
            BRepTools.Read_s(shape, str(p))
        else:
            parts = read_step(p)
            shape = parts[0].shape
    else:
        shape = shape_or_path
        if workdir is None:
            import tempfile
            workdir = Path(tempfile.mkdtemp(prefix="fea_submodel_mesh_"))

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    part = Part(name=spec.name or "Submodel", shape=shape)
    settings = {
        "size": float(spec.mesh_size_max),
        "min_size": float(spec.mesh_size_min if spec.mesh_size_min is not None else spec.mesh_size_max / 20.0),
        "order": 2,
        "optimize_high_order": True,
    }

    mesh = generate_mesh([part], selections={}, settings=settings, workdir=workdir)
    return mesh.nodes, mesh.tets


def _write_table(f, fmt: str, cols: list, n: int) -> None:
    """Bulk block writing of a table to a file in batches (efficient I/O for large meshes)."""
    _ROWS = 20000
    nc = len(cols)
    for s in range(0, n, _ROWS):
        e = min(n, s + _ROWS)
        flat = [None] * ((e - s) * nc)
        for j, c in enumerate(cols):
            flat[j::nc] = c[s:e]
        f.write((fmt * (e - s)) % tuple(flat))


def write_submodel_inp(coords: np.ndarray, tets: np.ndarray, material: Material,
                       cut_indices: np.ndarray, cut_U: np.ndarray, inp_path: Path,
                       title: str = "Submodel solution") -> None:
    """Writes the input file .inp for CalculiX for the submodel calculation with block writing and a safe material id."""
    inp_path = Path(inp_path)
    inp_path.parent.mkdir(parents=True, exist_ok=True)

    n_nodes = len(coords)
    n_elems = len(tets)
    n_cut = len(cut_indices)
    mat_id = "M1"
    solver = solver_name()

    with open(inp_path, "w", encoding="ascii", errors="replace") as f:
        f.write(f"*HEADING\n{deck_text(title)}\n")
        f.write("*NODE, NSET=NALL\n")
        node_ids = list(range(1, n_nodes + 1))
        _write_table(f, "%d, %.6f, %.6f, %.6f\n",
                     [node_ids, coords[:, 0].tolist(), coords[:, 1].tolist(), coords[:, 2].tolist()],
                     n_nodes)

        el_order = tets.shape[1]
        el_type = "C3D10" if el_order == 10 else "C3D4"
        f.write(f"*ELEMENT, TYPE={el_type}, ELSET=EALL\n")
        el_ids = list(range(1, n_elems + 1))
        conn_1based = (tets + 1).T.tolist()
        el_fmt = "%d, " + ", ".join(["%d"] * el_order) + "\n"
        _write_table(f, el_fmt,
                     [el_ids, *conn_1based],
                     n_elems)

        f.write(f"** material: {deck_text(material.name)}\n")
        f.write(f"*MATERIAL, NAME={mat_id}\n*ELASTIC\n{material.E:.6f}, {material.nu:.6f}\n")
        f.write(f"*SOLID SECTION, ELSET=EALL, MATERIAL={mat_id}\n")
        f.write(f"*STEP\n*STATIC, SOLVER={solver}\n*BOUNDARY\n")

        cut_nids = (cut_indices + 1).tolist()
        _write_table(f, "%d, 1, 1, %.8e\n", [cut_nids, cut_U[:, 0].tolist()], n_cut)
        _write_table(f, "%d, 2, 2, %.8e\n", [cut_nids, cut_U[:, 1].tolist()], n_cut)
        _write_table(f, "%d, 3, 3, %.8e\n", [cut_nids, cut_U[:, 2].tolist()], n_cut)

        f.write("*NODE FILE\nU, RF\n*EL FILE\nS\n*END STEP\n")


def solve_submodel(coords: np.ndarray, tets: np.ndarray, material: Material,
                   cut_indices: np.ndarray, cut_U: np.ndarray, workdir: Path,
                   job_name: str = "job") -> Results:
    """Runs the submodel calculation in CalculiX and returns a Results object with the equilibrium check."""
    workdir = Path(workdir)
    inp_path = workdir / f"{job_name}.inp"
    write_submodel_inp(coords, tets, material, cut_indices, cut_U, inp_path)

    run_res = run_ccx(workdir, job_name)
    if not run_res.finished or run_res.errors:
        raise RuntimeError(f"ccx submodel analysis failed: {run_res.errors}")

    frd_path = workdir / f"{job_name}.frd"
    frd = read_frd(frd_path)

    n = len(coords)
    U = np.zeros((n, 3))
    S = np.zeros((n, 6))
    RF = np.zeros((n, 3))

    b_u = next((b for b in frd.blocks if b.name in ("DISP", "U")), None)
    if b_u:
        U[b_u.node_ids - 1] = b_u.values[:, :3]

    b_s = next((b for b in frd.blocks if b.name in ("STRESS", "S")), None)
    if b_s:
        S[b_s.node_ids - 1] = b_s.values[:, :6]

    b_rf = next((b for b in frd.blocks if b.name in ("FORC", "RF")), None)
    if b_rf:
        RF[b_rf.node_ids - 1] = b_rf.values[:, :3]

    # Check of the equilibrium of the reactions on the cut faces of the submodel:
    rf_sum = RF.sum(axis=0)
    rf_norm = float(np.linalg.norm(rf_sum))
    scale = float(np.linalg.norm(RF, axis=1).sum())
    scale = max(scale, 1e-12)
    imbalance = rf_norm / scale

    eq_info = {
        "force_residual": rf_sum.tolist(),
        "imbalance": float(imbalance),
        "scale": scale,
    }

    warnings = []
    if imbalance > EQUILIBRIUM_TOL:
        warnings.append(
            f"Submodel: residual force imbalance at the cut {imbalance:.3e} > {EQUILIBRIUM_TOL} (residual force {rf_norm:.4g} N)"
        )

    mesh = Mesh(
        nodes=coords,
        tets=tets,
        tet_part=np.zeros(len(tets), dtype=np.int32),
        part_names=["Submodel"],
    )
    step = StepResult(
        name="Submodel_Result",
        U=U,
        S=S,
        fields={"RF": RF},
        equilibrium=eq_info,
    )
    return Results(mesh=mesh, steps=[step], warnings=warnings, info={"submodel": True, "equilibrium": eq_info})


def verify_cut_boundary(global_results: Results, sub_results: Results,
                        cut_indices: np.ndarray, step_index: int = 0,
                        max_rel_diff: float = 0.10) -> dict[str, Any]:
    """Compares the stresses on the cut face between the global model and the submodel.

    Criterion: the relative deviation on the cut should be < 10 %, which indicates that
    the cut boundary lies at a sufficient distance from the stress concentrator.
    """
    sub_step = sub_results.step(0)
    glob_step = global_results.step(step_index)
    sub_mesh = sub_results.mesh
    glob_mesh = global_results.mesh

    sub_mises_cut = sub_step.mises[cut_indices]
    cut_pts = sub_mesh.nodes[cut_indices]
    best_elems, best_xis = locate_batch(global_results, cut_pts)

    glob_mises_cut = np.zeros(len(cut_indices))
    tets_g = glob_mesh.tets
    for k in range(len(cut_indices)):
        conn = tets_g[best_elems[k]]
        glob_mises_cut[k] = tet10_shape(best_xis[k]) @ glob_step.mises[conn]

    diff = np.abs(sub_mises_cut - glob_mises_cut)
    ref = np.maximum(glob_mises_cut, 1.0)
    rel_diff = diff / ref

    max_diff = float(rel_diff.max())
    mean_diff = float(rel_diff.mean())

    return {
        "max_rel_diff": max_diff,
        "mean_rel_diff": mean_diff,
        "passed": max_diff <= max_rel_diff,
        "tolerance": max_rel_diff,
        "sub_mean_MPa": float(sub_mises_cut.mean()),
        "glob_mean_MPa": float(glob_mises_cut.mean()),
    }
