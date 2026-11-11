# FEA Studio – MCP Server

Controls the full FEA workflow (STEP → Gmsh → CalculiX / NGSolve+hypre → results, PNG renders) from an LLM client via Model Context Protocol. Headless (no PySide6 requirement), standard I/O (stdio) by default, or streamable HTTP.

```bash
.venv\Scripts\python.exe -m fea_core.mcp                                     # stdio
.venv\Scripts\python.exe -m fea_core.mcp --transport streamable-http        # http://127.0.0.1:8766/mcp
```

Options: `--groups` (tool groups, see below), `--workdir` (workspace directory root, default `%LOCALAPPDATA%\FEAStudio\mcp` or `FEA_MCP_WORKDIR`). Units: mm, N, MPa, t, s. Automatically generated working folders (`<name>_<timestamp>`) older than 30 days without changes are purged on startup (`FEA_MCP_KEEP_DAYS`, 0 = keep all); directories containing `.docx` reports or `.fea` project archives are preserved.

Unknown arguments raise errors. Client cancellation (`notifications/cancelled`) terminates active meshing or solver execution and releases resources. Server shutdown terminates child processes (Gmsh, ccx, MPI).

## Tool groups (context cost)

Every client session pays for the tool list in context tokens. Only the **`core` group** (study, geometry/faces, materials, constraints, loads, mesh, solve, results, report, jobs, GUI control; 46 tools + 3 helpers) is exposed by default: ~6 000 tokens, instead of ~60 000 for all tools (all groups now ~25 000: schemas are compacted and tools outside `core` list a one-sentence summary; `fea_tool_help(tool)` returns the full text).

| Group | Content |
|---|---|
| `model` | edge/point selections, imprints, coordinate systems, elastic support, orthotropy, bearing/hydrostatic/rotation/edge/point loads, hide/ignore parts, mesh refinement/convergence |
| `results` | section force, VTU/CSV export, line results + linearization, EN 1990 load combinations, topology optimization, plasticity, nonlinear convergence |
| `thermal`, `modal`, `fatigue`, `joints` | heat conduction + thermal stress; frequencies + buckling; fatigue; bolts and welds |
| `print` | FDM printing (shell/infill, creep, variants, sweep, advice, voxel detail) |
| `beams`, `frame`, `shell2d` | beam formulas + EN 1993-1-1 check; frame/truss systems; 2D and shell models |
| `contact`, `dynamics`, `motion`, `calc`, `cad` | contacts + nonlinear sequences; harmonic/transient/random/shock; mechanisms; calculators; geometry replacement |

Enable more groups:

- at start: `--groups thermal,print` or `--groups all`, or the environment variable `FEA_MCP_GROUPS` (e.g. in the `env` block of the client configuration);
- at run time: the tool `fea_tools_enable("thermal,print")` registers the tools and sends `notifications/tools/list_changed` (the server announces `tools.listChanged`); `fea_tools_groups` lists the groups. A client that does not refresh its tool list after `list_changed` must be restarted with `FEA_MCP_GROUPS` set. `python -m fea_core.mcp.setup` registers the default (`core`) for all clients, because support for `list_changed` could not be verified per client - add `FEA_MCP_GROUPS=all` to the `env` of a client that ignores it.

**Tool groups are not a permission boundary.** They only save context tokens: any authenticated client (or any client on a loopback transport without a token) can call `fea_tools_enable("all")`. Do not rely on them to restrict what an agent may do; use `--token` / network isolation for that. `fea_tools_enable` imports the group modules in a worker thread, so the server keeps answering pings meanwhile.

Support of `tools/list_changed` per client (the registration writes the default `core` only; for a client that does not refresh, put `FEA_MCP_GROUPS=all` or a group list into its `env` block):

| Client | `list_changed` refresh |
|---|---|
| Claude Desktop, Claude Code, Cursor, Windsurf, Antigravity, Gemini CLI, LM Studio, VS Code, Codex CLI | unknown (not verified) – if the new tools do not appear after `fea_tools_enable`, restart the client with `FEA_MCP_GROUPS` set |

The REST API always exposes all tools. Modules of disabled groups are not imported (idle RSS ~82 MB with `core`, ~93 MB with all groups).

## Client Configuration

Register all local AI desktop clients (Claude Desktop, Claude Code, Cursor, Windsurf, Antigravity, Gemini CLI, LM Studio, VS Code, Codex CLI) via:

```bash
.venv\Scripts\python.exe -m fea_core.mcp.setup              # Register (backups created as <file>.fea-zaloha)
.venv\Scripts\python.exe -m fea_core.mcp.setup --dry-run    # Preview modifications
.venv\Scripts\python.exe -m fea_core.mcp.setup --remove     # Unregister
```

Manual configuration:

### Claude Desktop (`%APPDATA%\Claude\claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "fea-studio": {
      "command": "C:\\FEA Studio\\.venv\\Scripts\\python.exe",
      "args": ["-m", "fea_core.mcp"],
      "cwd": "C:\\FEA Studio",
      "env": {
        "PYTHONPATH": "C:\\FEA Studio",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

### Cursor (`.cursor/mcp.json` or `%USERPROFILE%\.cursor\mcp.json`)

```json
{
  "mcpServers": {
    "fea-studio": {
      "command": "C:\\FEA Studio\\.venv\\Scripts\\python.exe",
      "args": ["-m", "fea_core.mcp"],
      "env": {
        "PYTHONPATH": "C:\\FEA Studio",
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

### Streamable HTTP Transport

Run `.venv\Scripts\python.exe -m fea_core.mcp --transport streamable-http` (default port 8766). In the client, set URL to `http://127.0.0.1:8766/mcp`. Note: Port 8765 belongs to the REST API (`fea_core.rest`). Binding to `--host 0.0.0.0` requires `--token` (or `FEA_MCP_TOKEN`), and requests must provide `Authorization: Bearer <token>`.

## Available Tools

All tools (REST: always all; MCP: by group, see above):

| Category | Tools | Description |
|---|---|---|
| Study | `fea_study_new`, `fea_study_open`, `fea_study_save`, `fea_study_status`, `fea_study_set_parts_ignored`, `fea_study_set_parts_hidden` | Manage FEA studies, CAD metadata matching, and part visibility. |
| Geometry | `fea_geometry_list_parts`, `fea_geometry_find_faces`, `fea_geometry_find_edges`, `fea_geometry_get_face_info`, `fea_selection_create/list/delete`, `fea_selection_create_edges/points` | Query topological entities and manage geometric selections. |
| Materials | `fea_materials_list`, `fea_materials_assign`, `fea_material_set_orthotropic` | Assign isotropic and general orthotropic material models. |
| Constraints | `fea_constraint_add_fixed/displacement/cylindrical/remote/elastic_support`, `fea_constraint_list/remove` | Apply fixed, prescribed displacement, cylindrical, remote, or foundation supports. |
| Loads | `fea_loadcase_create/delete`, `fea_load_add_force/pressure/gravity/remote/bearing/edge_force/point_force/hydrostatic/rotation/temperature`, `fea_load_list/remove` | Apply mechanical and thermal loads across load cases. |
| Coordinate Systems | `fea_csys_create`, `fea_csys_list`, `fea_csys_delete` | Define Cartesian and cylindrical local coordinate systems. |
| Thermal | `fea_thermal_case_create/delete`, `fea_thermal_add_temperature/heat_flux/convection/radiation/heat_generation`, `fea_thermal_remove/list/settings` | Steady-state and transient heat transfer boundary conditions. |
| Meshing & Checks | `fea_mesh_generate`, `fea_mesh_info`, `fea_mesh_separate_parts`, `fea_mesh_refine_point`, `fea_mesh_convergence`, `fea_check_readiness` | Generate tetrahedral meshes, bonded `*TIE` interfaces, local sphere refinements, and GCI mesh convergence checks. |
| Solving & Results | `fea_solve_run`, `fea_results_summary`, `fea_results_probe`, `fea_results_hotspots`, `fea_results_render_image`, `fea_results_line`, `fea_results_linearize` | Run FEA solvers, extract field results, evaluate stress hotspots, render field plots, and linearize stresses across section lines (ASME VIII-2). |
| Modal & Buckling | `fea_modal_set`, `fea_buckling_set`, `fea_results_modes`, `fea_results_render_mode` | Natural frequencies, effective modal mass participation, and linear eigenvalue buckling factors. |
| Visuals | `fea_model_render_image` | Render 3D scene with boundary condition glyphs, loads, and highlighted surfaces. |
| Fatigue | `fea_fatigue_run` | S-N, critical plane (Findley), and strain-life assessment. |
| Joints | `fea_joints_detect`, `fea_joints_list`, `fea_joints_update`, `fea_joints_assess` | Preloaded bolts (VDI 2230) and fillet/butt welds (EN 1993-1-8 / IIW). |
| 3D Printing (FDM) | `fea_print_refine`, `fea_print_get`, `fea_print_set`, `fea_print_creep`, `fea_print_props`, `fea_print_calibrate`, `fea_print_render_shell`, `fea_print_results`, `fea_print_variant_*`, `fea_print_sweep`, `fea_print_advice`, `fea_print_rank_build_direction`, `fea_print_presets` | Anisotropic slicer-derived shell-infill FEA, build direction ranking, parameter sweeps, and ISO 527 calibration. |
| Voxel Submodel | `fea_print_voxel_estimate`, `fea_print_voxel_run`, `fea_print_voxel_render` | High-resolution C3D8I voxel submodeling for printed layers. |
| Beams & Frames | `fea_section_get/search`, `fea_beam_analytic/buckling`, `fea_curved_beam`, `fea_ec3_member_check`, `fea_frame_*` | 1D Timoshenko frame and truss solver with EN standard cross-section library, releases, and EN 1993-1-1 checks. |
| 2D & Shells | `fea_twod_set/results/probe/line/linearize/render`, `fea_shell_set/detect/results/probe/render` | Axisymmetric, plane stress, plane strain, and shell models. |
| Reports | `fea_report_word` | Generate complete Microsoft Word (`.docx`) calculation reports. |
| Contacts | `fea_contact_detect/set/list/remove/check/settings`, `fea_analysis_set_sequence`, `fea_contact_results/render`, `fea_results_history` | Surface-to-surface contacts (bonded, frictionless, frictional, interference fits), multi-step nonlinear sequences. |
| Dynamics | `fea_dyn_settings/damping_set/harmonic_set/transient_set/random_set/shock_set/prestress_set/probe_*/mass_*/frf_get/results/render/plot/profile_presets` | Harmonic base excitation, transient response, random vibration (ASD), shock response spectrum (SRS), point/surface masses. |
| Multibody Motion | `fea_motion_body_*/joint_*/joints_from_cad/driver_add/force_add/load_add/contact_add/remove/settings/check_dof/run/results/render/export_loads` | Rigid multibody dynamics (Exudyn/Chrono), DOF checks, and peak load transfer to FEA. |
| Calculators | `fea_calc_snapfit/pressfit/hertz/friction/friction_library/mechanism/damping/drop` | Engineering analytical formula tools. |
| Async Jobs | `fea_job_start/status/cancel/result` | Background job execution with progress tracking. |

Resources: `fea://study/current`, `fea://results/summary`, `fea://materials/catalog`.
Prompts: `fea_setup_static_analysis`, `fea_setup_thermal_analysis`, `fea_mesh_convergence_check`, `fea_interpret_stress_results`, `fea_setup_contact`, `fea_setup_vibration_test`, `fea_setup_mechanism`.
