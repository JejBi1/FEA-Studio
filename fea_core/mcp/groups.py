"""Tool groups of the MCP server (cheap tool list for agents).

Every client session pays for the whole tool list in context tokens. Only the `core` group is exposed by default; the other groups are
enabled with `FEA_MCP_GROUPS` / `--groups` (comma list or `all`) or at run time by the tool `fea_tools_enable`
(the server then sends `notifications/tools/list_changed`). The REST API always exposes all the tools (`rest.registry`).

Static tables only – no import of the tool modules (they are imported lazily, see `server.build_server`).
"""
from __future__ import annotations

import os

CORE = "core"

# group → one-line description (shown by fea_tools_groups and in the server instructions)
GROUPS: dict[str, str] = {
    "core": "study, geometry/faces, materials, constraints, loads, mesh, solve, results, report, jobs, GUI control",
    "model": "edge/point selections, imprints, coordinate systems, elastic support, orthotropy, bearing/hydrostatic/rotation/"
             "edge/point loads, hide/ignore parts, mesh refinement/convergence",
    "results": "section force, export (VTU/CSV), results along a line + linearization, load combinations (EN 1990), "
               "topology optimization, plasticity assessment, nonlinear convergence",
    "thermal": "heat conduction (steady/transient), thermal stress",
    "modal": "natural frequencies, linear buckling, mode shapes",
    "fatigue": "fatigue (S-N, rainflow, critical plane)",
    "joints": "bolts and welds (VDI 2230, EN 1993-1-8, IIW)",
    "print": "3D printing (FDM): shell/infill, creep, variants, sweep, advice, voxel detail",
    "beams": "section library, analytic beam formulas, curved beams, EN 1993-1-1 member check (no mesh)",
    "frame": "beam/frame/truss systems (also without geometry)",
    "shell2d": "2D axisymmetric/plane models and shell models",
    "contact": "contacts, friction, nonlinear load sequences",
    "dynamics": "harmonic, transient, random, shock, prestressed dynamics, added mass",
    "motion": "rigid/flexible body mechanisms (Exudyn/Chrono), transfer of loads to FEM",
    "calc": "stand-alone calculators (snap-fit, press fit, Hertz, friction, convection, damping, drop)",
    "cad": "geometry replacement keeping settings, geometry iterations",
}

# module → (default group, other groups used by overrides of its tools)
MODULES: list[tuple[str, str, tuple[str, ...]]] = [
    ("tools_study", "core", ("model",)),
    ("tools_geometry", "core", ("model",)),
    ("tools_materials", "core", ("model",)),
    ("tools_bc", "core", ("model", "results")),
    ("tools_thermal", "thermal", ()),
    ("tools_mesh", "core", ("model",)),
    ("tools_results", "core", ("results",)),
    ("tools_eigen", "modal", ()),
    ("tools_paths", "results", ()),
    ("tools_fatigue", "fatigue", ()),
    ("tools_joints", "joints", ("print",)),
    ("tools_print", "print", ()),
    ("tools_voxel", "print", ()),
    ("tools_beams", "beams", ()),
    ("tools_frame", "frame", ()),
    ("tools_reduced", "shell2d", ()),
    ("tools_calc", "calc", ()),
    ("tools_contact", "contact", ()),
    ("tools_dynamics", "dynamics", ()),
    ("tools_jobs", "core", ()),
    ("tools_motion", "motion", ()),
    ("tools_gui", "core", ()),
    ("tools_cad", "cad", ()),
    ("prompts", "core", ()),
]

# tools whose group differs from the default group of their module
OVERRIDES: dict[str, str] = {
    **dict.fromkeys(("fea_study_set_parts_ignored", "fea_study_set_parts_hidden", "fea_geometry_find_edges",
                     "fea_selection_create_edges", "fea_selection_create_points", "fea_imprint_add", "fea_imprint_list",
                     "fea_imprint_delete", "fea_material_set_orthotropic", "fea_constraint_add_elastic_support",
                     "fea_load_add_edge_force", "fea_load_add_point_force", "fea_load_add_hydrostatic",
                     "fea_load_add_bearing", "fea_load_add_rotation", "fea_csys_create", "fea_csys_list",
                     "fea_csys_delete", "fea_mesh_separate_parts", "fea_mesh_refine_point", "fea_mesh_convergence"),
                    "model"),
    **dict.fromkeys(("fea_combination_add", "fea_combination_generate_en1990", "fea_combination_list",
                     "fea_combination_delete", "fea_results_section_force", "fea_results_export",
                     "fea_convergence_history", "fea_topo_opt_run", "fea_plasticity_assess"), "results"),
    **dict.fromkeys(("fea_print_set", "fea_print_results"), "print"),
}

META_TOOLS = ("fea_tools_enable", "fea_tools_groups", "fea_tool_help")


def group_of(tool: str, module_default: str) -> str:
    return OVERRIDES.get(tool, module_default)


def parse(spec: str | list[str] | None) -> set[str]:
    """'all' | 'a,b' | ['a', 'b'] → set of known groups (core always included); an unknown name → ValueError."""
    if spec is None:
        return {CORE}
    items = spec.replace(";", ",").replace(" ", ",").split(",") if isinstance(spec, str) else list(spec)
    names = {x.strip().lower() for x in items if x and x.strip()}
    if "all" in names or "*" in names:
        return set(GROUPS)
    bad = sorted(names - set(GROUPS))
    if bad:
        raise ValueError(f"unknown tool group {', '.join(bad)} (known: {', '.join(GROUPS)}, all)")
    return names | {CORE}


def from_env() -> set[str]:
    return parse(os.environ.get("FEA_MCP_GROUPS") or None)
