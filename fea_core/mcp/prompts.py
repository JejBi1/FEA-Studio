"""Resources and expert templates (prompts) of the MCP server."""
from __future__ import annotations

import json

from .diagnostics import summary_text
from .session import Session
from .tools_materials import catalog

UNITS = "Units: mm, N, MPa, t (tonne), s; density t/mm³ (steel 7.85e-9), g = 9810 mm/s², moment N·mm (1 N·m = 1000 N·mm)."

SETUP = """You are an experienced FEA analyst. Prepare a linear-static analysis in FEA Studio through the MCP tools.
{units}
User's goal: {goal}

Procedure:
1. fea_study_new(step_path) or fea_study_open(path); fea_geometry_list_parts – check the dimensions (mm!) and the part names.
2. Material: fea_materials_list(query) → fea_materials_assign(part or '*', name). Without Re the safety cannot be computed.
3. Faces: fea_geometry_find_faces (x/y/z = a face in a plane, normal, cylinder/radius for holes, where, near).
   Verify the result with an image fea_model_render_image(highlight=[id…]) or show_face_ids=True. Pass the ids 'Díl#i' to the
   other tools (stable selectors are stored in the study).
4. Supports: fea_constraint_add_fixed / fea_constraint_add_displacement. Prevent all 6 rigid-body motions, but do not over-constrain –
   a clamp is infinitely stiff (a singularity at the edge, extra stiffness). For symmetry use a zero displacement normal to the plane.
   A support outside the model (a joint, a bearing, a frame that is not modelled): fea_constraint_add_remote (a point + constrained components
   ux…rz; dofs="ux,uy,uz" = a pin at the point; the reaction = force and moment about the point).
5. Loads: fea_loadcase_create + fea_load_add_force (total force in N distributed over the face) / pressure (MPa) /
   gravity / remote (force and moment at a remote point through RBE3 – for hubs, pins, loads taken from other parts).
   Components in a local / cylindrical system: fea_csys_create + csys=… (cylindrical only for a force). Liquid:
   fea_load_add_hydrostatic, linear pressure: pressure with a gradient. Rotation (centrifugal force): fea_load_add_rotation.
   A force on an edge / at a point (fea_load_add_edge_force / point_force) is singular – prefer a small face or remote.
   Each state is a separate step; for fatigue define unit states.
   Elastic foundation / rubber pad: fea_constraint_add_elastic_support (k N/mm³, normal to the face). Composite, wood:
   fea_material_set_orthotropic. Parts that Gmsh does not join with the assembly: fea_mesh_separate_parts (*TIE).
   Natural frequencies: fea_modal_set (supports and densities suffice); stability: fea_buckling_set (linear buckling).
6. fea_model_render_image() – check visually that the supports and the load arrows are on the right faces and directions.
7. fea_check_readiness() – fix all errors; consider the warnings.
8. fea_mesh_generate(global_size) – at least 2–3 tet10 elements across the wall thickness; minSJ > 0.3 is good.
9. fea_solve_run() – the equilibrium check is automatic (an imbalance > 1e-3 = an error).
10. fea_results_summary → fea_results_hotspots → fea_results_render_image(field='mises', view='multi').
11. Profile through the wall fea_results_line (plot=True), linearization fea_results_linearize; stress convergence
    at the critical place fea_mesh_convergence(point, radius_mm). Verify members analytically with fea_beam_analytic /
    fea_ec3_member_check.
State the assumptions (material, supports, idealization) and the uncertainties to the user."""

CONVERGENCE = """Check the mesh convergence for the monitored quantity ({quantity}) {where}.
{units}

Procedure (h-convergence, automatically with the fea_mesh_convergence tool):
1. Baseline calculation: fea_mesh_generate(global_size=h0) → fea_solve_run → fea_results_hotspots: find the critical place
   and verify that it is NOT a singularity (clamped edge, sharp inner corner, point force) – the stress never converges there.
2. fea_mesh_convergence(quantity=…, point=…, radius_mm=…, levels=3, ratio=1.5, tol=0.02, timeout_s=…):
   - quantity=\"max_mises\" + point + radius_mm = maximum von Mises in a sphere around the place, only the sphere is refined
     (cheap, recommended); without radius_mm globally (DOF ∝ ratio³ per level – watch the DOF and time estimate);
   - quantity=\"probe\" + point + field (mises, s1, sx, ux, umag…) = value at a fixed point;
   - quantity=\"max_disp\" (displacements always converge), \"frequency\" (mode = mode number; fea_modal_set),
     \"print_safety\" (3D printing; for the print shell rather fea_print_refine).
   The tool computes the levels in a copy of the study (the results of the main study stay) and returns a table of levels (element
   size, nodes, value, change, ccx stress error estimate STR %), the Richardson extrapolation (order p, extrapolated
   value, GCI – Roache 1998, ASME V&V 20), a verdict and warnings (singularity, wandering maximum).
   By hand the same can be done with: fea_mesh_refine_point(center, radius_mm, size_mm) or grading=[{{\"selection\": …, \"size\": h/2}}]
   and repeating fea_solve_run.
3. Evaluate: change of the last refinement < 2–5 % = converged; order p close to 2 (tet10) = asymptotic range;
   GCI = estimate of the discretization uncertainty. A value that keeps growing with p < 0.5 = singularity.
4. BEWARE of singularities: the stress at a clamped edge, in a sharp inner corner without a fillet or under a point force does
   NOT settle with refinement (it grows without bound). fea_results_hotspots marks such places as “singularity” – do not judge them
   by the nodal value; watch the quantity at a distance of 2–3 elements (probe), the nominal stress, the linearization through the wall
   (fea_results_linearize) or the displacements (they always converge). The ccx error estimate (field='stress_error') shows the places
   where the mesh is coarse.
5. Summarize the result in a table: level, element size, DOF/nodes, value, change [%], extrapolation, GCI, time.
Displacements and reactions converge quickly; stresses more slowly; in concentrations (fillets, holes) you need ≥ 3–4 elements per quarter circle. """

INTERPRET = """Evaluate the stress results and the safety of the structure{part_txt}.
{units}

1. fea_results_summary(step=None): check the equilibrium (applied_load ≈ −reaction_total, imbalance < 1e-3), the size of the
   displacements (linear theory holds for small deformations – displacement ≪ thickness) and the reactions at the supports (do they match the load?).
2. fea_results_hotspots(top_n=8): separate singularities (support edge, sharp inner corner, point force, bolt connection)
   from real concentrations (fillets, holes, shoulders). Do not judge a singularity by its nodal value.
3. Safety against yield: k = Re / σ_vM (statics, ductile material). Indicatively: k ≥ 1.5 for a static structure,
   ≥ 2 with an uncertain load. Brittle materials (cast iron) – use σ1 and Rm. For printed parts fea_print_results (Tsai-Wu).
   Local plasticity in a concentration above Re is often admissible for ductile steel under static load (the linear model
   overestimates it) – but not in fatigue: use fea_fatigue_run there.
4. Images: fea_results_render_image(field='mises', view='multi'), possibly 'safety', 's1' (tension – fatigue, brittle),
   'displacement'. Describe where the stress is highest and why (bending, notch, contact).
5. Recommended geometry changes: fillets in inner corners (R ≥ 0.1–0.2 × thickness), a stronger section at the place of the maximum bending
   moment (second moment of area ~ h³), ribs, holes moved away from edges, lightening of lightly stressed places (σ < 0.2·Re), a change
   of the support. For each recommendation state the expected effect and whether to verify it with a new calculation.
6. State the limitations: linear material and small deformations, idealized supports (a clamp is stiffer than reality), contacts
   are bonded (no opening of joints), stability only by linear buckling (fea_buckling_set – without imperfections,
   the real capacity is lower), dynamics only by natural frequencies (fea_modal_set). """


THERMAL = """You are an experienced FEA analyst. Prepare a thermal analysis (heat conduction) in FEA Studio through the MCP tools{stress_txt}.
{units}
Heat: temperature °C, heat flux W/m² (the model is in mm: 1 W/m² = 1e-6 W/mm²), power W, heat transfer coefficient W/(m²·K),
conductivity λ = k W/(m·K), expansion α 1/K, specific heat c_p J/(kg·K), time s.
User's goal: {goal}

Procedure:
1. fea_study_new(step_path) or fea_study_open(path); fea_geometry_list_parts – dimensions in mm, part names.
2. A material with thermal properties: fea_materials_list (k_W_mK, alpha_1_K, cp_J_kgK – the library steels, stainless steels,
   aluminiums and plastics have them) → fea_materials_assign; own values custom_props {{"k": 50, "alpha": 12e-6,
   "cp": 460}}. A steady state needs k, a transient one also the density rho and cp, thermal stress needs α.
3. Thermal state: fea_thermal_case_create(name) – steady; transient transient=True, time_s, dt_s (0 = time/100),
   outputs (frames); initial temperature fea_thermal_settings(initial_C=…).
4. Boundary conditions (faces as for supports: id 'Díl#i', query {{"x": 0}}, selection name):
   - fea_thermal_add_temperature – a prescribed temperature (contact with a large body, a regulated heat sink);
   - fea_thermal_add_heat_flux – flux_W_m2 or power_W (heating, losses of a component on a face; + = into the body);
   - fea_thermal_add_convection – h: free air 5–25, fan 25–250, water 500–10 000 W/(m²·K), ambient °C;
   - fea_thermal_add_radiation – emissivity (polished Al 0.05, oxidized steel 0.8, black paint 0.95), ambient °C;
   - fea_thermal_add_heat_generation – power W in the volume of a part (winding, resistor).
   A steady state MUST have heat removal (temperature, convection or radiation) from all connected parts – otherwise it has no solution.
   Faces without a boundary condition are perfectly insulated (adiabatic). Parts with a shared mesh have perfect thermal
   contact (no contact resistance); bolts do not conduct heat.
5. fea_model_render_image(thermal_case=…) – check the faces and values; fea_check_readiness().
6. fea_solve_run() – ccx *HEAT TRANSFER; the heat balance of a steady state is checked automatically (> 1e-3 = an error).
7. fea_results_summary(thermal_case=…) – T max/min and where, fluxes of the boundary conditions (the sum of supply = removal),
   fea_results_render_image(field='temperature', thermal_case=…, view='multi'), field='heat_flux' (flux),
   fea_results_hotspots(field='temperature'), fea_results_probe(point, thermal_case=…, time_s=…).
   Sanity check: a 1D estimate with the resistances R = L/(k·A) and 1/(h·A); the temperature must neither exceed the source nor fall below the ambient
   (without internal sources).
8. Thermal stress (optional): fea_thermal_settings(reference_C=assembly temperature) → supports only against rigid-body motions
   (a statically determinate support, symmetry planes – a full clamp prevents expansion and gives large stresses/singularities) →
   fea_load_add_temperature(case_name, thermal_case=…) (or value_C = a uniform temperature) → fea_solve_run →
   fea_results_summary / hotspots (von Mises). A linear temperature field in a free body causes no stress.
State the assumptions: linear conduction, properties independent of temperature, radiation to the ambient only, perfect thermal contact of parts,
bolts do not conduct heat, h and ε are estimates (the greatest uncertainty)."""

CONTACT = """You are an experienced FEA analyst. Prepare a static analysis of an assembly WITH CONTACT in FEA Studio through the MCP tools.
{units}
User's goal: {goal}

Contact makes the calculation NONLINEAR (CalculiX only): superposition of states does not hold (fatigue by superposition is unavailable), the calculation takes
minutes to hours. Without contact, parts with a common face are rigidly connected by a shared mesh (bonded).
Procedure:
1. fea_study_new / fea_study_open, materials (fea_materials_assign), supports and load cases as for statics.
2. fea_contact_detect() – a proposal of pairs (touch / gap / penetration); check the penetrations (a press fit = interference,
   otherwise a modelled thread or an assembly error) → fea_contact_detect(apply=True).
3. fea_contact_set(pair, type=…): frictionless (contact without friction), frictional (μ), rough (no slip), no_separation
   (slides without separating), interference (an interference fit – clearance = −Δd/2 RADIALLY, nominal geometry only), compression_only
   (a support in compression only, without b = to the ground), bonded / none. μ: purpose='slip' (force transmitted by friction → μ_min) or
   'resistance' (resistance, drive force → μ_max); μ is an ESTIMATE – compute both bounds (fea_calc_friction_library shows the library).
   K (stiffness) and λ (stick_slope) are penalties (default K = 50·E, λ = K/100); for an interference K ≥ 50·E or mortar.
4. fea_contact_settings: method surface_to_surface (default, pressure ±0.1 %) | mortar; split separate (separate meshes,
   required for an interference) | conformal_split (bolted parts); nlgeom only for large deformations (snap-fits).
5. A part loaded by force and held ONLY by friction does not converge (slip in statics) – drive it by displacement: a rigid remote support
   (fea_constraint_add_remote, coupling='rigid') + fea_analysis_set_sequence(branches=False, steps=[…, "prescribed":
   [{{"target": …, "kind": "disp", "dof": 1, "amplitude": [[0, 0], [1, 0.5]]}}]]), or add weak springs
   (fea_contact_settings(stabilize='springs')). Friction only in the next step (mu_change) helps convergence.
6. fea_contact_check() and fea_check_readiness(); fea_model_render_image() shows the pairs coloured by type.
7. fea_mesh_generate → fea_contact_check() again (initial penetrations in the mesh) → fea_job_start(kind='solve') →
   fea_job_status every ~20 s → fea_job_result.
8. Results: fea_contact_results (Fn, Ft, contact area, p_max, opening, slip; a PARTIAL result = non-convergence,
   the last converted increment applies), fea_contact_render(field='tlak' | 'stav' | 'mezera' | 'prokluz' | 'treni' (pressure | state | gap | slip | friction),
   pair=…), fea_results_history(plot=True) (force–displacement), fea_results_summary / hotspots for stresses.
9. Sanity check: Σ Fn ≈ the load transmitted by the contact, pressure ≈ F/A for a planar contact, Hertz (fea_calc_hertz),
   a press fit by Lamé (fea_calc_pressfit). State the sensitivity to μ (μ_min / μ_max) and K (×2) in the conclusion.
State the assumptions: linearly elastic material, penalty contact (K, λ), Coulomb friction with an estimated μ, a nonlinear
calculation without superposition, possible partial results."""

VIBRATION = """You are an experienced FEA and dynamics analyst. Prepare a simulation of a vibration / shock test in FEA Studio through MCP.
{units}
Test: {test}

Procedure (modal superposition: ccx computes the modes, the response is computed in Python – fast and accurate):
1. Fixture model: supports = the clamping faces of the fixture (base excitation moves ALL constrained nodes in the excited direction –
   a symmetry plane in the excited direction is an error; use the whole model). Material densities are mandatory; motors, PCB assemblies,
   omitted parts are added as a mass fea_dyn_mass_add(name, selection, mass_kg=… | mass_per_area_kg_m2=…).
2. fea_dyn_profile_presets(name) – the structure of a test (IEC 60068-2-6 sine, -64 random, -27 shock, -47 mounting).
   Take the NUMERICAL VALUES (frequencies, amplitudes, ASD, durations) from the specification / standard – do not invent them.
3. fea_dyn_settings(enabled=True, modes=…) – modes up to 1.5–2× the highest excitation frequency, Σ m_eff ≥ 90 % in the excited
   direction (fea_dyn_results checks it). verify_ccx=True = a ccx benchmark at the probes.
4. Damping fea_dyn_damping_set: bolted steel ~0.8 %, welded ~0.3 %, PLA 1–2 %, PCB 1/(2√f_n), fixtures 2–5 %;
   an UNCERTAIN input (±50 %) – state the sensitivity or recommend a measurement (hammer, decrement).
5. Probes fea_dyn_probe_add(name, point=…) at the places of interest (components, arm ends).
6. Excitation according to the test:
   - sine: fea_dyn_harmonic_set(f_min_Hz, f_max_Hz, sine_profile={{"displacement_mm": x̂, "acceleration_g": a}}, dof,
     sweep_rate_oct_min, sweeps) – FRF, resonances, fatigue from the sweep;
   - random: fea_dyn_random_set(points=[[f, W|null], …], slopes_db_oct=[…], grms_g, dof, duration_h, method='dirlik');
   - shock: fea_dyn_shock_set(pulse='halfsine', A_g, tau_s, dof, combine='cqc');
   - decay after an axis motion: fea_dyn_transient_set(t_end_s, dt_s ≤ T_min/20, base_acc_g, pulse, settle_tol_mm).
   Compute each axis (x, y, z) separately (one axis per calculation).
7. fea_check_readiness → fea_job_start(kind='solve') → fea_job_status → fea_job_result.
8. Results: fea_dyn_results (modes, m_eff, checks, maxima), fea_dyn_frf_get / fea_dyn_plot(kind='frf' | 'psd' |
   'transient' | 'srs'), fea_dyn_render(field='harmonic_envelope' | 'rms' | '3sigma' | 'damage' | 'life' | 'shock').
   Evaluate: the 3σ stress against the fatigue limit / Re, damage D < 1 (spectral fatigue), resonances in the test band,
   accelerations of the components. Miles √(π/2·f_n·Q·W) as a hand check of the 1st mode.
9. The report fea_report_word – contains modes, profiles, FRF, RMS, 3σ, SRS, fatigue and a red warning about damping.
State the assumptions: a linear model (without contact), modal superposition with a truncated basis (static correction / missing
mass), estimated damping, idealized clamping."""

MECHANISM = """You are an experienced mechanism designer and FEA analyst. Prepare in FEA Studio a rigid-body mechanism calculation
(Exudyn; flexible bodies Craig–Bampton; shape contact in Chrono) and the transfer of its loads to FEM through the MCP tools. {units}
Task: {goal}

Procedure:
1. fea_study_new(step_path) / fea_study_open; fea_geometry_list_parts. Materials with density (fea_materials_assign) –
   the masses of bodies are computed from CAD (printed parts only approximately).
2. Bodies: fea_motion_body_add(name, parts=[…]) – rigidly connected parts = one body; the frame ground=True. Check with
   fea_motion_body_list (masses in kg, parts outside bodies).
3. Joints: fea_motion_joints_from_cad() – a proposal from coaxial cylinders (a pin in a hole); check the certainty and the reason, then
   accept=True. The others by hand fea_motion_joint_add (prismatic, spherical, gear ratio, screw lead_mm, rod…).
   The joint faces (faces_a / faces_b) are required for the transfer to FEM. Friction: friction={{model: 'coulomb', mu_k, r_eff}}.
4. Drives fea_motion_driver_add (angle / speed / torque, law constant | table | sine | poly | scurve), springs
   and dampers fea_motion_force_add, external forces fea_motion_load_add.
5. fea_motion_check_dof: a kinematic problem needs 0 free degrees after the drives are given; redundant constraints →
   the reactions are AMBIGUOUS – offer a spherical / cylindrical joint instead of a second revolute, or redundant joints as
   FEM supports (support_joints).
6. fea_motion_settings(kind='kinematic' | 'static' | 'dynamic', t_end_s, outputs, collisions='all' for a collision
   check) → fea_motion_run. Check the equilibrium of the bodies and the power balance in the summary, the warnings (collisions, assembly).
   A flexible body (a compliant arm, vibrations): fea_motion_body_add(…, flexible=True, modes=8–20) – dynamic problem only,
   isotropic material, body joints with faces (interfaces); spectral_radius=1 leaves the vibrations undamped; results
   'napětí:<těleso>', 'deformace:<těleso>' (stress:<body>, deformation:<body>).
   Shape contact (a cam, a stop by shape, rolling, sliding): fea_motion_contact_add(A, B, mu, deflection_mm) +
   fea_motion_settings(solver='chrono', kind='dynamic', dt_s=1e-4…2.5e-4) – only with the module “chrono” (otherwise an installation
   guide); without friction and stops in joints, without flexible bodies; results 'kontaktní síla:<těleso>' (contact force:<body>).
7. fea_motion_results(series=['reakce:<kloub>', 'moment pohonu:<pohon>'], find_max='reakce:<kloub>'),
   fea_motion_render('reakce:<kloub>') – graphs (reaction:<joint>, drive torque:<drive>); drive torque against the motor catalogue, reactions against the bearings.
8. To FEM: fea_motion_export_loads(mode='max', quantity='reakce:<kloub>', body=…) for the decisive instant
   (the support reaction in FEM = the reaction from MBD – inertia relief), or mode='history' for fatigue (RBE3 unit states +
   channels); then fea_mesh_generate, fea_solve_run, fea_results_summary, fea_fatigue_run.
State the assumptions: rigid bodies (unless flexible), friction as a load from the reaction of the previous step, collisions only
as a check (unless contact in Chrono – then implicit Euler, the error proportional to the step, the shape = a triangulation with the deviation
deflection_mm), the mass of printed parts approximate."""


def register(mcp, sess: Session) -> None:

    @mcp.resource("fea://study/current", name="Current study", mime_type="application/json",
                  description="Live JSON model of the current study (materials, selections, supports, loads, mesh, solver).")
    def study_current() -> str:
        with sess.use():
            if sess.study is None:
                return json.dumps({"study": None, "hint": "fea_study_new(step_path) or fea_study_open(path)"})
            return json.dumps(sess.study.to_dict(), ensure_ascii=False, indent=1, default=str)

    @mcp.resource("fea://results/summary", name="Results summary", mime_type="text/plain",
                  description="Text summary of the latest results (displacements, stresses, reactions, equilibrium, safety; thermal states: temperatures, fluxes of the boundary conditions, heat balance).")
    def results_summary_res() -> str:
        with sess.use():
            return summary_text(sess)

    @mcp.resource("fea://materials/catalog", name="Material catalogue", mime_type="application/json",
                  description="The complete material database (built-in with sources + user).")
    def materials_catalog() -> str:
        return json.dumps(catalog(None), ensure_ascii=False, indent=1)

    @mcp.prompt(name="fea_setup_static_analysis", description="Guide to setting up a static analysis (supports, loads, mesh).")
    def fea_setup_static_analysis(goal: str = "static strength check of a part") -> str:
        return SETUP.format(units=UNITS, goal=goal)

    @mcp.prompt(name="fea_setup_thermal_analysis",
                description="Guide to a thermal analysis: a material with λ, a thermal state, heat removal, thermal stress.")
    def fea_setup_thermal_analysis(goal: str = "temperature field of a part and heat dissipation", thermal_stress: bool = False) -> str:
        return THERMAL.format(units=UNITS, goal=goal,
                              stress_txt=" including thermal stress" if thermal_stress else "")

    @mcp.prompt(name="fea_mesh_convergence_check", description="Adaptive mesh refinement and convergence check.")
    def fea_mesh_convergence_check(quantity: str = "max. von Mises at the critical place", location: str = "") -> str:
        return CONVERGENCE.format(units=UNITS, quantity=quantity, where=f"at the place {location}" if location else "")

    @mcp.prompt(name="fea_interpret_stress_results",
                description="Safety evaluation and recommendations for geometry changes from the stress results.")
    def fea_interpret_stress_results(part: str = "") -> str:
        return INTERPRET.format(units=UNITS, part_txt=f" (part {part})" if part else "")

    @mcp.prompt(name="fea_setup_contact",
                description="Guide to the statics of an assembly with contact and friction (pairs, μ, step sequence, job).")
    def fea_setup_contact(goal: str = "pressure distribution in the contact and forces transferred by friction") -> str:
        return CONTACT.format(units=UNITS, goal=goal)

    @mcp.prompt(name="fea_setup_vibration_test",
                description="Guide to the simulation of a vibration / shock test (modes, damping, sine, PSD, shock, probes).")
    def fea_setup_vibration_test(test: str = "sinusoidal and random vibration per IEC 60068-2-6 / -64") -> str:
        return VIBRATION.format(units=UNITS, test=test)

    @mcp.prompt(name="fea_setup_mechanism",
                description="Guide to a mechanism calculation (bodies, joints, drives, reactions) and the transfer of loads to FEM.")
    def fea_setup_mechanism(goal: str = "reactions in joints and drive torque, the decisive instant into FEM") -> str:
        return MECHANISM.format(units=UNITS, goal=goal)
