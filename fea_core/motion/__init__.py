"""Kinematics and dynamics of mechanisms (stages F8.3, F8.8, F8.9 C1) – Exudyn (rigid and flexible bodies) and Chrono (contact)
in separate processes.

Modules (core without GUI; OCP and Exudyn are loaded lazily – the import of the package is light):
  model.py        Mechanism, Body, MotionJoint, JointFriction, ForceElement, Driver, MotionLoad, MotionRun, ToFea
                  (Study.mechanism, JSON), JOINT_KINDS / JOINT_LABELS (Czech names), JOINT_CONSTRAINTS (Grübler)
  laws.py         courses of the drivers: Law(kind, data) – constant | table | sine | poly | scurve (limited jerk)
  mass.py         body_properties(study, parts) – mass [t], centroid [mm], tensor of inertia [t·mm²] from the CAD
  joints_cad.py   suggest_from_geometry(study, parts) / suggest_from_metadata(study, meta) → [JointSuggestion]
                  (certainty, reason); apply_suggestion(study, parts, s) creates the selections of faces and the joint
  run.py          check_model, grubler, check_dof(study, parts) → {grubler, dof_free, dof_driven, redundant,
                  redundant_list, messages, ok}; run_motion(study, workdir, parts=None, study_dir=None, on_log=None,
                  cancel=None, on_progress=None, timeout=None) → MotionResult (check of the equilibrium of the bodies, power
                  balance, collisions)
  worker.py       separate process `python -m fea_core.motion.worker JOB` (SI), exudyn_adapter.py = model of Exudyn
  results.py      MotionResult: fields by times (units of the project), series("reakce:Kloub 1"), find_max(…), summary(),
                  save/load (npz); in the project .fea the file results_motion.npz (project.save_project(motion=),
                  project.load_motion(workdir))
  flex.py         flexible bodies (Body.flexible, Craig–Bampton): prepare (mesh, interface = faces of the joints), finish
                  (shapes and stresses into MotionResult), displacement / stress / von_mises(res, body, k), stress_history
  chrono_run.py   Chrono solver for contact (MotionRun.solver = "chrono", Mechanism.contacts = [MotionContact]):
                  available(), INSTALL_HINT, triangulate, prepare_contacts; chrono_worker.py = process in tools/chrono_env
  collisions.py   check_collisions(study, parts, res, pairs, tol) – min. distance of the OCP shapes in time, first collision
  to_fea.py       export_loads(study, res, mode, …) – instant / maximum → load state (RemoteLoad RBE3 on the faces
                  of the joints + Inertia, support RemoteSupport distributing = inertia relief); history → unit states +
                  fatigue channels; check_relief(info, step), superpose(res_fe, study, t)

Proposal of the API for the GUI step "Motion" (fea_gui, not implemented yet – everything through Study.mechanism and the functions above):
  - Table of bodies: Body(name, parts, ground) – selection of parts in 3D (filter Body), mass properties
    run.body_props(study, parts) displayed next to it (kg, mm), warning "indicative" for printed parts (mass.py).
  - Joints: button "Suggest from geometry" → joints_cad.suggest_from_geometry → confirmation dialog (certainty, reason,
    highlight faces_a / faces_b in 3D) → apply_suggestion; manually: kind (JOINT_LABELS), bodies A/B, faces (selection of faces
    → axis z cylinder_axis as in the MCP fea_motion_joint_add), limits + restitution, friction (JointFriction).
  - Drivers (Driver: joint, quantity, law + editor of the table / S-curve with the graph Law), force elements (ForceElement),
    external loads (MotionLoad with a table), weight Mechanism.gravity.
  - "Check of the freedom" → run.check_dof in a thread (workers.TaskThread) → dock Messages (redundancy, missing driver).
  - "Run" → run.run_motion(on_progress=…, cancel=threading.Event) in a thread; progress bar from "progress x %".
  - Animation: time slider + play; position of the body k in time i = MotionResult.pose(body, i) → (R, T) as
    gp_Trsf for AIS_Shape.SetLocalTransformation (FeaView.batch()); highlight the collision (meta["collisions"]["first"]).
  - Graphs (widgets.LazyPlot): MotionResult.series(name) – offer the names of MotionResult.SERIES × items; "Find maximum" = find_max → jump
    of the slider to the time of the maximum.
  - "Convert to FE": dialog (instant / maximum of a quantity / the whole course, body or parts, joints of the support) →
    to_fea.export_loads → new load states (and Study.fatigue for the course) → step Loads / Solve; after the FE calculation
    show to_fea.check_relief (reactions of the support = MBD).
"""
from .model import (Mechanism, Body, MotionJoint, JointFriction, ForceElement, Driver, MotionLoad,  # noqa: F401
                    MotionRun, ToFea, MotionContact, JOINT_KINDS, JOINT_LABELS)
