"""Result of the topology optimisation in the results of the study (saving to the project, display in the GUI, STL after opening).

The nodal density is attached to each load state as the field `StepResult.fields["topo_density"]` (the mesh is the same,
on which the optimisation ran), the summary and the removed elements into `Results.info["topology"]` (saved by `Results.save`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .export import densities_to_nodes, topo_surface

if TYPE_CHECKING:
    from ..results import Results
    from .beso import TopologyOptimizer

FIELD = "topo_density"


def attach(results: "Results", opt: "TopologyOptimizer") -> dict:
    """Attaches the result of the optimisation to the results (mesh of the optimisation = mesh of the results). Returns the summary `info["topology"]`."""
    mesh = results.mesh
    rho = np.asarray(opt.rho, float)
    if len(rho) != len(mesh.tets):
        raise ValueError(f"The optimisation ran on a different mesh ({len(rho)} elements, results {len(mesh.tets)})")
    node_rho = densities_to_nodes(mesh, rho)
    for st in results.steps:
        st.fields[FIELD] = node_rho
    sp = opt.spec
    info = {
        "method": "BESO",
        "target_volume_fraction": float(sp.target_volume_fraction),
        "evolution_rate": float(sp.evolution_rate),
        "filter_radius_mm": float(sp.filter_radius),
        "penalty": float(sp.penalty),
        "non_design_selections": list(sp.non_design_selections),
        "iterations": len(opt.history),
        "converged": bool(opt.history[-1].converged) if opt.history else False,
        "final_volume_fraction": float(opt.current_volume_fraction),
        "initial_compliance_Nmm": opt.initial_compliance,
        "final_compliance_Nmm": opt.final_compliance,
        "volume_history": [round(float(h.volume_fraction), 5) for h in opt.history],
        "compliance_history_Nmm": [float(h.total_compliance) for h in opt.history],
        "void_elements": np.nonzero(rho <= 0.5)[0].astype(int).tolist(),
    }
    results.info["topology"] = info
    return info


def element_densities(results: "Results") -> np.ndarray | None:
    """Element densities (1 = material, 0 = removed) from the saved result; None = the result has no optimisation."""
    info = (results.info or {}).get("topology")
    if not info:
        return None
    rho = np.ones(len(results.mesh.tets))
    void = np.asarray(info.get("void_elements", []), dtype=np.int64)
    rho[void[(void >= 0) & (void < len(rho))]] = 0.0
    return rho


def shape_triangles(results: "Results") -> tuple[np.ndarray, np.ndarray] | None:
    """Closed surface of the resulting shape (ρ ≥ 0.5) as linear triangles (P (k, 3, 3), density at the vertices
    (k, 3)) for the colour display; None = no optimisation.
    """
    rho = element_densities(results)
    if rho is None:
        return None
    mesh = results.mesh
    verts, faces = topo_surface(mesh, rho)
    if len(faces) == 0:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    node_rho = densities_to_nodes(mesh, rho)
    from scipy.spatial import cKDTree
    d, i = cKDTree(mesh.nodes).query(verts)
    tol = 1e-9 * max(float(np.ptp(mesh.nodes, axis=0).max()), 1.0)
    vals = np.where(d <= tol, node_rho[i], 0.5)          # vertices of the surface of the mesh = nodes, isosurface = 0.5
    return verts[faces], vals[faces]
