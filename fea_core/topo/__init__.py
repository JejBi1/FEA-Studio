"""Package of the core of the topology optimisation (BESO / SIMP)."""
from .spec import TopologyOptSpec
from .filter import SensitivityFilter, filter_sensitivities
from .beso import (
    TopologyOptimizer,
    OptimizationIteration,
    compute_tet_volumes,
    compute_tet_centroids,
    compute_mesh_element_volumes_and_centroids,
    compute_strain_energy_density,
    compute_element_strain_energies,
    element_strain_energies,
    element_stiffness_scale,
    MIN_STIFFNESS_RATIO,
    find_threshold_bisection,
    find_threshold_simp,
    identify_non_design_elements,
)
from .export import (
    densities_to_nodes,
    export_topo_stl,
    topo_surface,
    write_binary_stl,
)

__all__ = [
    "TopologyOptSpec",
    "SensitivityFilter",
    "filter_sensitivities",
    "TopologyOptimizer",
    "OptimizationIteration",
    "compute_tet_volumes",
    "compute_tet_centroids",
    "compute_mesh_element_volumes_and_centroids",
    "compute_strain_energy_density",
    "compute_element_strain_energies",
    "element_strain_energies",
    "element_stiffness_scale",
    "MIN_STIFFNESS_RATIO",
    "find_threshold_bisection",
    "find_threshold_simp",
    "identify_non_design_elements",
    "densities_to_nodes",
    "export_topo_stl",
    "topo_surface",
    "write_binary_stl",
]
