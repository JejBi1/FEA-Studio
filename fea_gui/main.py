"""Main window of FEA Studio.

Layout ("Studio × Tables" design): at the top the header (menus, phase tabs, phase actions, Run calculation), on the left the shelf of step commands in groups (SHELF), in the centre the view bar (selection, colour, section) above the 3D window, on the right the study tree, the step tabs of the phase, the Properties card and the step panel, at the bottom the table tabs (table_dock), log, messages and the status bar. All calculation logic is in fea_core – the GUI only edits Study and calls the core.
"""
from __future__ import annotations

from html import escape
import html
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QFormLayout,
                               QGridLayout, QHBoxLayout, QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                               QPlainTextEdit, QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QStackedWidget, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QAbstractItemView,
                               QFrame, QScrollArea, QTabWidget, QTableWidgetItem)

from fea_core import config
from fea_core import materials as matlib
from fea_core.geometry import legacy_step_names, read_step, face_props, bbox
from fea_core.native_cad import is_native, to_step as native_to_step
from fea_core.joints.levels import LEVELS as JOINT_LEVELS, settings_level as joint_settings_level
from fea_core.project import save_project, load_project, load_study_json
from fea_core.selectors import FaceSelector, EdgeSelector, PointSelector, resolve, resolve_edge
from fea_core.solve import check_study
from fea_core.study import (Study, Material, Fixed, Displacement, Force, Pressure, Gravity, RemoteLoad,
                            LoadCase, CylindricalSupport, BearingLoad, ThermalLoad, RemoteSupport, Rotation,
                            ElasticSupport)

from .safe_browser import SafeTextBrowser
from .contact_panel import ContactPanel
from .dynamics_panel import DynamicsPanel
from .motion_panel import MotionPanel, mech_signature
from . import dyn_results as dynr
from .fatigue_panel import FatiguePanel
from .frame_panel import FramePanel
from . import frame_results as frs
from . import twod_gui
from . import shell_gui
from . import submodel_dialog as smd
from .joints_panel import JointsPanel, change_level, util_color
from .eigen_gui import (MODE_QUANTITIES, EigenBox, ModesTable, mode_info_html, mode_label, mode_quantity,
                        mode_quantities)
from .legend import Legend
from .messages import MessagesPanel
from . import i18n, theme
from fea_core.i18n import decimal, keep_names, tr
from .print_panel import PrintPanel, PRINT_FIELDS
from .thermal_panel import ThermalPanel, KIND_SHORT, describe_bc, describe_case
from .user_settings import app_settings
from . import session_files
from .view import FeaView
from .widgets import (C_HEX, Section, sci, TaskProgress, check_item, checked, compact, dspin, help_tip, panel_header,
                      fit_rows, scroll_body, set_checks, set_placeholder, table)
from .workers import MeshThread, SolveThread, kill_mesh_workers

APP_NAME = "FEA Studio"
# F8: "Contacts" after Joints, "Motion" (mechanism kinematics, fea_core.motion) after Constraints, "Dynamics" after Loads
STAGES = ["Model", "Materials", "3D printing", "Joints", "Contacts", "Members", "Constraints", "Motion", "Heat", "Loads",
          "Dynamics", "Mesh", "Solve", "Results", "Fatigue"]
STAGE = {name: i for i, name in enumerate(STAGES)}      # step index by name (do not hard-code numbers)
# F11: steps grouped into phases (Progress bar); STAGES / STAGE and the panel indices are NOT changed by this
PHASES = [("Geometry", ["Model", "Materials", "3D printing"]),
          ("Assembly", ["Joints", "Contacts", "Members"]),
          ("Conditions", ["Constraints", "Motion", "Heat", "Loads", "Dynamics"]),
          ("Calculation", ["Mesh", "Solve"]),
          ("Evaluation", ["Results", "Fatigue"])]
PHASE_OF = {s: p for p, ss in PHASES for s in ss}
assert [s for _, ss in PHASES for s in ss] == STAGES
# optional steps: on the bar only when the study uses them, the user pinned them (Analyses menu) or opened them
OPTIONAL_STAGES = ("3D printing", "Contacts", "Members", "Motion", "Heat", "Dynamics", "Fatigue")
PHASE_TIPS = {
    "Geometry": "model, materials, 3D printing",
    "Assembly": "bolts and welds, contacts, members",
    "Conditions": "supports, loads, heat, motion, dynamics",
    "Calculation": "mesh and running the analysis",
    "Evaluation": "results and fatigue",
}
# names of optional steps in the Analyses menu
ANALYSIS_LABELS = {
    "3D printing": "3D printing – print settings of plastic parts",
    "Contacts": "Contacts – part contact, friction",
    "Members": "Members – beam structure",
    "Motion": "Motion – mechanisms",
    "Heat": "Heat – heat conduction",
    "Dynamics": "Dynamics – vibration",
    "Fatigue": "Fatigue – life",
}
# phase names on the header tabs and in the panel headings (proposal: Model, Assembly, Conditions, Calculation, Results)
PHASE_LABEL = {"Geometry": "Model", "Evaluation": "Results"}


class ElidedLabel(QLabel):
    """Label that is shortened with "…" in a narrow place (the full text is in the tooltip)."""

    def __init__(self, text=""):
        super().__init__()
        self._full = ""
        self.setMinimumWidth(40)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text):
        text = tr(text)                       # the subclass bypasses the translating wrapper of QLabel.setText
        self._full = text
        self.setToolTip(text)
        self._elide()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _elide(self):
        super().setText(self.fontMetrics().elidedText(self._full, Qt.ElideRight, max(self.width(), 40)))


# header buttons per phase: (text, target as in SHELF, short label, long description)
HEADER_ACTIONS = {
    "Geometry": [("Replace geometry…", "replace_geometry_dialog",
                   "Replace the model with modified geometry and keep the study setup",
                   "Loads a modified STEP / .ipt / .f3d and transfers materials, selections, supports and loads to it (faces are matched by position and shape; whatever is not found is shown in the remapping window).")],
    "Assembly": [("Detect joints…", "joints_panel.detect", "Find bolts, nuts, washers and welds in the model",
                 "Detects bolts, nuts, washers and welds by name, shape, color or layer; you confirm uncertain suggestions in a window.")],
    "Calculation": [("Mesh convergence…", "open_convergence_dialog", "Convergence study: analysis on refined meshes",
                 "Solves the study on several refined meshes (globally or in a sphere around a point), Richardson extrapolation and the GCI index.")],
    "Evaluation": [("Export CSV / VTU", "open_export_dialog", "Export results to CSV (Excel) and VTU (ParaView)",
                     "Nodal fields, reactions and elements to CSV with solver node numbers; VTU + PVD for ParaView."),
                    ("Generate report", "export_report", "Technical report of the analysis in Word (.docx)  [Ctrl+P]",
                     "Word report: input, mesh, results by load case with maps, joint and fatigue assessments, conclusion.")],
}


def element_quality_nodes(mesh) -> np.ndarray:
    """Element quality for the mesh colour: q = 6√2·V / l_rms³ of the corner tetrahedron (1 = regular, 0 = degenerate),
    at a node the worst of the neighbouring elements. Cached on the mesh.
    """
    def calc():
        c = mesh.nodes[mesh.tets[:, :4]]
        e = [c[:, j] - c[:, i] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
        vol = np.abs(np.einsum("ij,ij->i", e[0], np.cross(e[1], e[2]))) / 6.0
        l_rms = np.sqrt(sum(np.einsum("ij,ij->i", v, v) for v in e) / 6.0)
        q = np.clip(6.0 * np.sqrt(2.0) * vol / np.maximum(l_rms, 1e-30) ** 3, 0.0, 1.0)
        node = np.ones(len(mesh.nodes))
        np.minimum.at(node, mesh.tets.ravel(), np.repeat(q, mesh.tets.shape[1]))
        return node
    return mesh.cached("gui_quality", ("nodes", "tets"), calc)


def phase_label(phase: str) -> str:
    return PHASE_LABEL.get(phase, phase)


# material colours (part colour by material) and a part without material
MATERIAL_COLORS = [(169, 188, 203), (232, 196, 120), (150, 196, 150), (214, 160, 160), (180, 160, 214),
                   (120, 190, 200), (220, 180, 130), (190, 200, 120)]
NO_MATERIAL_COLOR = (201, 203, 204)
VIEW_DIRECTIONS = (("iso", "Isometric"), ("front", "Front"), ("top", "Top"), ("right", "Right"))


# step labels of the progress bar: short (level 1), long STAGE_LONG (level 2: What is this? / F1)
STAGE_TIPS = {
    "Model": "load a STEP file or a .fea project, overview of parts",
    "Materials": "material of the parts – library, designation converter",
    "3D printing": "printing of plastic parts – perimeters, infill, build direction",
    "Joints": "bolts and welds – detection, preload, assessment",
    "Contacts": "part contact, friction, nonlinear steps",
    "Members": "beam structure – nodes, members, sections, EN 1993",
    "Constraints": "model supports – fixed, displacement, cylindrical",
    "Motion": "mechanism – bodies, joints, drives",
    "Heat": "heat conduction – steady-state and transient cases",
    "Loads": "load cases – force, pressure, moment, gravity, temperature",
    "Dynamics": "vibration – harmonic, transient, random, shock",
    "Mesh": "element size, refinement, mesh generation",
    "Solve": "readiness check, solver, running the analysis",
    "Results": "stress and displacement maps, safety factor, probe, maxima",
    "Fatigue": "load history, S-N, damage and life",
}
STAGE_LONG = {
    "Model": "loading STEP geometry or a .fea project, parts overview",
    "Materials": "assigning materials to parts (library, grade converter, custom material)",
    "3D printing": "print settings for plastic parts (perimeters, infill, build direction), variants, creep",
    "Joints": "bolts and welds (detection from the model, preload, VDI 2230 / EN 1993-1-8 assessment)",
    "Contacts": "contact of parts – frictionless, frictional, interference, compression only; friction and nonlinear steps (optional)",
    "Members": "frame – nodes, members, cross-sections, supports, EN 1993-1-1 check (optional)",
    "Constraints": "model restraints – fixed support, prescribed displacement, cylindrical and remote support",
    "Motion": "mechanism – bodies, joints, drivers; animation, joint reactions, transfer to FEA (optional)",
    "Heat": "heat conduction – steady-state / transient cases, boundary conditions (optional)",
    "Loads": "load cases and loads – force, pressure, moment, gravity, rotation, temperature",
    "Dynamics": "vibration – damping, harmonic / transient response, random vibration, shock, added mass (optional)",
    "Mesh": "element size, local refinement, mesh generation (optional – the analysis generates it)",
    "Solve": "readiness check, solver selection and running the analysis",
    "Results": "color maps of stress, displacement and safety factor, probe, maximum locations, reactions",
    "Fatigue": "load history, S-N curves, damage and life (optional)",
}
STAGE_COLORS = "Dot: green = done, orange = warning / optional, red = error, gray = nothing yet"
GREEN, ORANGE, RED, GREY = "#2e9e4f", "#e69a1a", "#d23b3b", "#9aa1ab"
C_CONSTRAINT = (40, 110, 230)
C_FORCE = (220, 40, 40)
C_PRESSURE = (190, 60, 200)
C_REFINE = (30, 170, 170)
BEARING = "Bearing load (hole / pin)"
THERMAL_LOAD = "Temperature (thermal stress)"
ROTATION = "Rotation (centrifugal force)"
FORCE = "Force (components)"
FRAME_NODE = "Member – nodal force / moment"
FRAME_MEMBER = "Member – distributed load"
FRAME_ONLY_OFF = ("3D printing", "Joints", "Contacts", "Constraints", "Heat", "Mesh")   # steps without meaning for beams without CAD
# left tool shelf: step commands (label, icon from theme.ICONS, target, description); target = window attribute
# (button of the panel – click, or a method) or (menu, text) = pick a menu item – see MainWindow._shelf_run
# tool shelf (Studio × Tables design): ("#", "NÁZEV") = group heading, otherwise (text, icon, target, label); target =
# window attribute (button → click, method → call; with a dot panel.button), "@Text" = panel button by its text,
# "focus:field" / "popup:menu" / "raise:dock" / "menu:button", (menu, text) = pick a menu item
SHELF = {
    "Model": [("#", "GEOMETRY"),
              ("Load", "open", "open_dialog", "Open STEP or .fea project"),
              ("Replace", "detect", "replace_geometry_dialog", "Replace geometry and keep the setup"),
              ("Check", "check", "show_geometry_check", "Geometry check: table of warnings from loading"),
              ("#", "PARTS"),
              ("Show", "show", "btn_show_all_parts", "Show all parts"),
              ("Isolate", "body", "isolate_selected", "Show only the selected parts"),
              ("Hide", "hide", "btn_toggle_hide_part", "Hide / show selected parts in 3D"),
              ("Exclude", "exclude", "btn_toggle_ignore_part", "Exclude selected parts from analysis / include again")],
    "Materials": [("#", "MATERIAL"),
                  ("Assign", "assign", "btn_assign_sel", "Assign the material from the form to selected parts"),
                  ("All", "assign", "@Assign to all", "Assign the material from the form to all parts"),
                  ("Library", "report", "popup:mat_combo", "Choose a material from the library"),
                  ("Converter", "combine", "focus:mat_find", "Steel and alloy grade converter (ČSN, EN, AISI…)")],
    "Joints": [("#", "JOINTS"),
              ("Detect", "detect", "joints_panel.detect", "Find bolts, nuts, washers and welds in the model"),
              ("From hole", "cylinder", "@Bolt into selected hole…", "Bolt into the hole selected in 3D"),
              ("Weld", "line", "@Mark the part of the selected face as a weld", "Mark the part of the selected face as a weld"),
              ("#", "CONNECTION"),
              ("Contact", "contact", "stage:Kontakty", "Contact pairs (contact, friction) instead of a shared mesh"),
              ("#", "CHECK"),
              ("Exclude", "exclude", "@Exclude the part of the selected face", "Exclude the part of the selected face from the analysis")],
    "Constraints": [("#", "SUPPORTS"),
              ("Fixed support", "fixed", "btn_fixed", "Fix the selected faces rigidly"),
              ("Cylindrical", "cylinder", "add_cylindrical", "Cylindrical support of a hole / pin from selection"),
              ("Remote", "point", "add_remote_support", "Remote support: face connected to a point"),
              ("Displacement", "line", "@Displacement from selection (checked components)", "Prescribed displacement of the checked components")],
    "Loads": [("#", "LOADS"),
                 ("Force", "force", ("ld_type", FORCE), "Load type: force given by components"),
                 ("Pressure", "pressure", ("ld_type", "Pressure"), "Load type: pressure normal to the face"),
                 ("Moment", "moment", ("ld_type", "Moment (remote point)"), "Load type: moment via a remote point"),
                 ("Bearing", "cylinder", ("ld_type", BEARING), "Load type: bearing force in a hole"),
                 ("Gravity", "mass", ("ld_type", "Gravity"), "Load type: self-weight"),
                 ("Add", "add", "btn_add_load", "Add a load to the faces selected in 3D"),
                 ("#", "STATES"),
                 ("New case", "add", "add_lc", "New load case"),
                 ("Combination", "combine", "open_combinations_dialog", "Load case combinations to EN 1990")],
    "3D printing": [("#", "PRINTING"),
                ("Assign", "assign", "print_panel.assign_material", "Assign the printed material to the selected parts"),
                ("Shell", "section", "print_panel.show_shell", "Mesh section: print shell and core"),
                ("Refine", "mesh", "print_panel.btn_refine", "Refine mesh at critical locations and recompute")],
    "Contacts": [("#", "CONTACTS"),
                 ("Find", "detect", "contact_panel.btn_find", "Find touching parts and suggest contact pairs"),
                 ("Add pair", "contact", "contact_panel.add_pair", "Add a contact pair from the form")],
    "Members": [("#", "MEMBERS"),
              ("Node", "point", "frame_panel.add_node", "Add a frame node"),
              ("Member", "line", "frame_panel.add_member", "Add a member between nodes"),
              ("Cross-section", "profile", "frame_panel.add_section", "Add a cross-section from the library"),
              ("From wires", "open", "frame_panel.from_wires", "Members from a STEP wireframe")],
    "Motion": [("#", "MECHANISM"),
              ("Body", "body", "motion_panel.add_body", "Add a mechanism body"),
              ("Joint", "joint", "motion_panel.add_joint", "Add a joint between bodies"),
              ("Driver", "moment", "motion_panel.add_driver", "Add a joint driver"),
              ("Play", "run", "motion_panel.a_play", "Play the mechanism motion")],
    "Heat": [("#", "CONDITIONS"),
              ("Temperature", "temperature", ("thermal_panel.bc_type", "Prescribed temperature"),
               "Boundary condition: prescribed surface temperature"),
              ("Flux", "heat", ("thermal_panel.bc_type", "Heat flux [W/m²]"), "Boundary condition: heat flux through a surface"),
              ("Convection", "convection", ("thermal_panel.bc_type", "Heat transfer – convection"),
               "Boundary condition: heat transfer to the surroundings"),
              ("Radiation", "radiation", ("thermal_panel.bc_type", "Radiation to surroundings"), "Boundary condition: radiation to the surroundings"),
              ("Add", "add", "thermal_panel.add_bc", "Add the boundary condition to the faces selected in 3D")],
    "Dynamics": [("#", "DYNAMICS"),
                 ("Mass", "mass", "dynamics_panel.add_mass", "Add a point or surface mass"),
                 ("Probe", "point", "dynamics_panel.btn_pr_pt", "Response probe at the selected point")],
    "Mesh": [("#", "MESH"),
            ("Create", "mesh", "btn_mesh", "Create the finite element mesh"),
            ("Size", "line", "focus:ms_size", "Global element size [mm]"),
            ("Refine", "add", "@Refine selection", "Refine the mesh on the faces selected in 3D"),
            ("Cancel", "exclude", "btn_mesh_cancel", "Cancel meshing")],
    "Solve": [("#", "SOLVER"),
               ("Run", "run", "btn_run", "Run analysis"),
               ("Stop", "exclude", "btn_cancel", "Stop the analysis"),
               ("#", "OUTPUT"),
               ("Log", "report", "raise:dock_log", "Solver log (Gmsh and CalculiX output)"),
               ("Messages", "check", "raise:dock_messages", "Messages: errors, warnings and information")],
    "Results": [("#", "EXPLORE"),
                 ("Section", "section", "menu:sec_button", "Section the model by a plane"),
                 ("Along line", "line", "btn_path", "Quantity along a line segment, linearization"),
                 ("Isosurface", "isosurface", "btn_iso", "Isosurface of a chosen value"),
                 ("Vectors", "force", "@Vektory σ...", "Principal stress vectors"),
                 ("Section force", "cut_force", "btn_sec_force", "Resultant internal forces in a cut"),
                 ("#", "VIEW"),
                 ("Animation", "animate", "btn_anim", "Deformation animation"),
                 ("Export", "export", "open_export_dialog", "Export results to VTU / CSV")],
    "Fatigue": [("#", "FATIGUE"),
              ("PragTic", "detect", "fat_panel.sn_from_db", "S-N curve from the PragTic database"),
              ("Calculate", "run", "fat_panel.btn", "Run fatigue analysis")],
}
SHELF_LABELS = {"Results": "Explore"}
P_MODES = ["constant", "linear", "hydrostatic (liquid)"]
# labels of the load types (field Type in the Loads step) and their vector
LOAD_TIPS = {
    FORCE: "Force given by components, distributed uniformly over the selected faces (by area), edges (Edge filter,\nby length) or to points (Point filter, each F/n). The total force is entered, not the density.",
    "Force normal to face": "Total force normal to the selected planar faces (positive pushes into the material).\nUse Pressure for curved faces.",
    "Pressure": "Pressure on the selected faces [MPa], positive acts into the material (against the outward normal).\nProfile: constant, linear (gradient) or hydrostatic (liquid below the surface).",
    "Moment (remote point)": "Moment [N·mm] at a remote point attached to the selected faces (rigid or distributing\ncoupling) – e.g. torque on a shaft. Point = face center or user-specified.",
    "Force at remote point": "Force [N] at a point outside the model attached to the selected faces – a moment F × arm is also produced\n(e.g. a weight on an arm, a force at the end of an unmodeled part).",
    BEARING: "Bearing load: cosine pressure distribution on the half of the hole / pin facing the force (like a pin in an eye),\naxial component as uniform shear. Select a cylindrical face (Hole filter H).",
    "Gravity": "Self-weight of all parts from material density: acceleration g [mm/s²], default (0; 0; −9810).\nNeeds no face selection; one per load case.",
    ROTATION: "Centrifugal force of rotating parts (ρ·ω²·r) and optionally angular acceleration α.\nEnter the axis (point + direction or “Axis from selection”) and the speed.",
    THERMAL_LOAD: "Thermal stress: temperature from a thermal case (Heat step) or a uniform part temperature.\nStress arises from ΔT = T − reference temperature and restrained expansion (requires material α).",
    FRAME_NODE: "Point force [N] and moment [N·mm] at a frame node (global axes). Select the node in the list\nor by clicking it in 3D.",
    FRAME_MEMBER: "Distributed member load [N/mm] along its length: uniform, or linearly varying up to the value at the end;\nin global axes (snow, wind) or in the member local axes (x, y, z). Select the member in the list\nor by clicking in 3D. Member self-weight is added by the Gravity load.",
}
VEC_TIPS = {"Moment (remote point)": "Moment components [N·mm] about the point (right-handed about the axes)",
            "Gravity": "Gravitational acceleration [mm/s²] – downward direction; usually (0; 0; −9810)",
            "Force at remote point": "Force components [N] acting at the remote point",
            BEARING: "Components of the bearing load resultant [N]; the radial part becomes pressure on the hole, the axial part becomes shear"}
# short labels of the load types (field Type); the long LOAD_TIPS go to the detailed help (F1 / ⓘ / What is this?)
LOAD_SHORT = {
    FORCE: "Force spread over faces, edges or points (total force, not density)",
    "Force normal to face": "Total force normal to the selected planar faces (positive pushes into the material)",
    "Pressure": "Pressure on the selected faces [MPa] – constant, linear or hydrostatic",
    "Moment (remote point)": "Moment [N·mm] at a remote point attached to the selected faces",
    "Force at remote point": "Force [N] at a point outside the model attached to the faces (also produces a moment)",
    BEARING: "Bearing load: cosine pressure on the half of the hole / pin facing the force",
    "Gravity": "Self-weight of all parts from the material density (one per load case)",
    ROTATION: "Centrifugal force of rotating parts and optionally angular acceleration",
    THERMAL_LOAD: "Thermal stress from a thermal case or from a uniform part temperature",
    FRAME_NODE: "Point force and moment at a node of the beam structure",
    FRAME_MEMBER: "Distributed member load along its length (uniform or linear)",
}
VEC_SHORT = {"Moment (remote point)": "{} component of the moment [N·mm] about the point",
             "Gravity": "{} component of the gravity acceleration [mm/s²]",
             "Force at remote point": "{} component of the force [N] at the remote point",
             BEARING: "{} component of the bearing load resultant [N]"}
G_STD = 9810.0                               # g [mm/s²] for hydrostatic pressure
C_EDGE = (200, 30, 120)
C_ELASTIC = (20, 150, 120)                   # elastic support (springs)
STRESS_ERROR_TIP = ("Stress error estimate [%] (CalculiX, ccx solver only): difference in principal stress between adjacent\nelements at a node (gradient estimate). Below ~5 % is good, above ~10 % refine the mesh there. At singularities\n(sharp corners, fixed support edges) it stays high even after refinement.")

BACKENDS = [("auto", "Automatic (recommended – direct / iterative by model)"),
            ("ccx", "Direct sparse solver – CalculiX"),
            ("ngs_pardiso", "Direct, many cases at once – NGSolve + PARDISO"),
            ("ngs_hypre", "Iterative PCG + AMG (large models) – NGSolve + hypre")]

QUANTITIES = [
    ("von Mises", "MPa"), ("Total displacement", "mm"), ("Displacement X", "mm"), ("Displacement Y", "mm"), ("Displacement Z", "mm"),
    ("1st principal stress S1", "MPa"), ("3rd principal stress S3", "MPa"),
    ("Stress σx", "MPa"), ("Stress σy", "MPa"), ("Stress σz", "MPa"),
    ("Stress τxy", "MPa"), ("Stress τyz", "MPa"), ("Stress τzx", "MPa"),
    ("Safety factor to Re", "–"),
    ("3D printing – safety factor (Tsai-Wu/Hill)", "–"), ("3D printing – failure index", "–"),
    ("Temperature", "°C"),                       # only a state with a thermal load (StepResult.fields["T"])
    ("Stress error estimate", "%"),             # ccx gradient estimate (StepResult.fields["stress_error"], ccx backend only)
    # contact (F8, StepResult.fields CPRESS, STATUS, OPENING, CSLIP, CSHEAR) – at the end, old indices stay valid
    ("Contact – pressure", "MPa"), ("Contact – contact state", "–"), ("Contact – opening", "mm"),
    ("Contact – slip", "mm"), ("Contact – frictional stress", "MPa"),
    # Stress Tool (F9.1)
    ("Safety factor – Tresca", "–"),
    ("Safety factor – Mohr-Coulomb", "–"),
    ("Safety factor (Stress Tool)", "–"),
    # topology optimisation (BESO, StepResult.fields["topo_density"]) – at the end, old indices stay valid
    ("Topology – density", "–"), ("Topology – final shape", "–"),
    # plasticity (StepResult.fields["peeq"], nonlinear CalculiX) – at the end, old indices stay valid
    ("Equivalent plastic strain PEEQ", "–"),
]
PEEQ_Q = "Equivalent plastic strain PEEQ"
TOPO_Q = ("Topology – density", "Topology – final shape")
TOPO_SHAPE_Q = "Topology – final shape"
TOPO_MISSING = ("The result has no topology optimization – run it with the Topology (BESO)… button in the results panel.")
CONTACT_Q = {"Contact – pressure": "CPRESS", "Contact – contact state": "STATUS", "Contact – opening": "OPENING",
             "Contact – slip": "CSLIP", "Contact – frictional stress": "CSHEAR"}
STATUS_NOTE = "Contact status: −1 not a contact node, 0 open, 1 sliding, 2 sticking"
# quantities of the thermal state (Results.thermal)
THERMAL_QUANTITIES = [("Temperature", "°C"), ("Heat flux |q|", "W/m²"), ("Heat flux qx", "W/m²"),
                      ("Heat flux qy", "W/m²"), ("Heat flux qz", "W/m²")]
SAFETY_QTY = ("Safety factor to Re", "3D printing – safety factor (Tsai-Wu/Hill)", "Safety factor – Tresca", "Safety factor – Mohr-Coulomb", "Safety factor (Stress Tool)")
NO_MIN_QTY = ("von Mises", "Total displacement", "3D printing – failure index", "Heat flux |q|", "Stress error estimate",
              MODE_QUANTITIES[0][0], "Contact – pressure", "Contact – contact state", "Contact – opening",
              "Contact – slip", "Contact – frictional stress", "Safety factor – Tresca", "Safety factor – Mohr-Coulomb", "Safety factor (Stress Tool)",
              "Topology – density", "Topology – final shape", "Equivalent plastic strain PEEQ")   # without a minimum marker
# maximum places with singularity classification (fea_core.mcp.diagnostics): GUI quantity → diagnostics field;
# other mechanical quantities (displacement, safety, print) → von Mises
HS_STRESS = {"von Mises": "mises", "1st principal stress S1": "s1", "3rd principal stress S3": "s3", "Stress σx": "sxx",
             "Stress σy": "syy", "Stress σz": "szz", "Stress τxy": "sxy", "Stress τyz": "syz", "Stress τzx": "szx"}
HS_THERMAL = {"Temperature": "temperature", "Heat flux |q|": "heat_flux", "Heat flux qx": "hfx",
              "Heat flux qy": "hfy", "Heat flux qz": "hfz"}
HS_TOP = 8                                   # number of places in the table
F8_SOURCE = "nonlinear analysis and dynamics"  # message source (Messages dock) – partial result, advice, dynamics checks
CAD_SOURCE = "CAD import"        # message source: metadata from the CAD next to the STEP (fea_core.cad_import)
DROP_SUFFIXES = (".step", ".stp", ".fea", ".ipt", ".iam", ".f3d", ".f3z")   # files to drop onto the window (as Open);
# .ipt / .f3d / .f3z = native Autodesk files (module native, fea_core.native_cad)
NATIVE_FILTER = "Inventor / Fusion (*.ipt *.iam *.f3d *.f3z)"
# selection filter in 3D (fea_gui.view.PICK_FILTERS): key, text, shortcut, help
PICK_FILTER_ACTIONS = [
    ("face", "Face", "P", "Selection filter Face: a click adds / removes one face (click on empty space = clear selection)"),
    ("body", "Body", "B", "Selection filter Body: a click adds / removes all faces of a part.\nIn steps without face selection (Model, Materials…) a click always selects whole parts."),
    ("hole", "Hole", "H", "Selection filter Hole: a click on a cylindrical face selects the whole hole / pin\n(all coaxial cylindrical faces of the same radius). Switching also expands the already selected cylinders."),
    ("edge", "Edge", "E", "Selection filter Edge: a click adds / removes an edge (edge load in the Loads step) and shows\nthe length and type of the edge, for a circle the radius and center (“Use center” in the status bar = point\nof a remote load / support). The face selection is not changed."),
    ("point", "Point", "V", "Selection filter Point: a click on a vertex or face adds / removes a point (point load in the Loads\nstep). A point on a face is projected exactly onto the CAD face; a click on empty space clears the point selection."),
]


# short labels of the selection filters (tool shelf); the long text from PICK_FILTER_ACTIONS goes to whatsThis (F1 / What is this?)
PICK_FILTER_SHORT = {
    "face": "Face selection: a click adds / removes one face (click on empty space = clear)",
    "body": "Body selection: a click adds / removes all faces of the part",
    "hole": "Hole selection: a click on a cylindrical face selects the whole hole / pin",
    "edge": "Edge selection: a click adds / removes an edge and shows its length, radius and center",
    "point": "Point selection: a click on a vertex or face adds / removes a point",
}


def fmt(v, dec=4):
    """Number with a decimal comma (decimal point in English)."""
    if v is None:
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", " ")
    s = f"{v:.{dec}g}"
    if "e+" in s and abs(v) < 1e9:           # 20 000 instead of 2e+04 (heat fluxes in W/m²)
        return f"{float(s):,.0f}".replace(",", " ")
    return decimal(s)


def nice_floor(v):
    """Rounding down to 1 / 2 / 2.5 / 5 × 10^n (default element size)."""
    if v <= 0:
        return 1.0
    e = 10.0 ** math.floor(math.log10(v))
    return max(m * e for m in (1.0, 2.0, 2.5, 5.0) if m * e <= v * (1 + 1e-9))


def dot_icon(color):
    pm = QPixmap(12, 12)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(1, 1, 10, 10)
    p.end()
    return QIcon(pm)


def msg(parent, title, text, icon=QMessageBox.Warning):
    b = QMessageBox(icon, title, str(text), QMessageBox.Ok, parent)
    b.setTextFormat(Qt.PlainText)
    return b.exec()



def _prewarm_imports():
    """Optional (FEA_PREWARM=1): import of libraries that the GUI needs only for results / plots (scipy.spatial ~1.2 s,
    matplotlib ~0.6 s), in a background thread – only the import of modules, no widgets (those may be created only in the
    main thread). Off by default: the two libraries add ~45 MB to the idle program, the first chart then waits ~1–2 s.
    """
    import threading

    def run():
        try:
            import matplotlib.figure  # noqa: F401
            import scipy.sparse  # noqa: F401
            import scipy.spatial  # noqa: F401
        except Exception:  # noqa: BLE001 – only a speed-up, the error shows up when the feature is really used
            pass
    threading.Thread(target=run, daemon=True, name="fea-prewarm").start()


def _remember_ui() -> bool:
    """Remember the state of the bar / sections between runs (only a real start of the program, not the tests)."""
    try:
        from .widgets import remember_ui
        return remember_ui()
    except ImportError:
        import os
        return os.environ.get("FEA_UI_REMEMBER") == "1"


def _batched(fn):
    """Method of the main window with batched redraw of the 3D view (one UpdateCurrentViewer at the end)."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        view = getattr(self, "view", None)
        if view is None or not hasattr(view, "batch"):
            return fn(self, *a, **k)
        with view.batch():
            return fn(self, *a, **k)
    return wrapper

def form_hide_empty_rows(form: QFormLayout) -> None:
    """Form rows whose widgets are all hidden are removed from the layout (QFormLayout.setRowVisible) –
    a hidden row with a layout of boxes (QHBoxLayout) would otherwise leave a gap. The caller sets the widget visibility;
    a row with at least one visible widget is shown whole (the widgets of a row are switched together).
    """
    def widgets(item):
        if item is None:
            return []
        w = item.widget()
        if w is not None:
            return [w]
        lay = item.layout()
        return [x for i in range(lay.count()) for x in widgets(lay.itemAt(i))] if lay is not None else []

    for row in range(form.rowCount()):
        ws = []
        for role in (QFormLayout.LabelRole, QFormLayout.FieldRole, QFormLayout.SpanningRole):
            ws += widgets(form.itemAt(row, role))
        form.setRowVisible(row, any(not w.isHidden() for w in ws))


class NearestPoints:
    """Nearest displayed point for the result probe – the part of cKDTree the GUI uses (`data`, `query(x)` → (distance, index)).
    A click asks for one point: a numpy pass over the points (~1 ms for 100 k) instead of building a tree on every redraw (animation
    frames, section moves) and importing scipy.spatial only for that. Several points at once → cKDTree."""

    def __init__(self, pts):
        self.data = np.asarray(pts, float)

    def query(self, x, *args, **kwargs):
        x = np.asarray(x, float)
        if x.ndim == 1 and not args and not kwargs:
            d2 = np.einsum("ij,ij->i", self.data - x, self.data - x)
            j = int(np.argmin(d2))
            return float(np.sqrt(d2[j])), j
        from scipy.spatial import cKDTree
        return cKDTree(self.data).query(x, *args, **kwargs)


class _LazyPanel:
    """Panel of an optional step built at the first access (start-up of the window: these panels are ~40 % of the widgets).
    Non-data descriptor – after the build the instance attribute takes over."""

    def __init__(self, attr: str):
        self.attr = attr

    def __get__(self, obj, cls=None):
        if obj is None:
            return self
        return obj._build_lazy_panel(self.attr)


# attribute → (step, class); the stack holds a placeholder until the panel is needed
LAZY_PANELS = {"print_panel": ("3D printing", PrintPanel), "frame_panel": ("Members", FramePanel),
               "contact_panel": ("Contacts", ContactPanel), "dynamics_panel": ("Dynamics", DynamicsPanel),
               "motion_panel": ("Motion", MotionPanel)}
# the step colour of a panel that is not built yet: color() reads only the study through `win` → the same functions on a light object
_SHELL_ATTRS = ("color", "study", "frame", "mech", "part_names", "errors", "body_names")


class MainWindow(QMainWindow):
    STAGE = STAGE
    STAGES = STAGES
    flowChanged = Signal()           # after restoring the step state (refresh_flow) – setup guide, see flow_state

    def __init__(self):
        i18n.install(QApplication.instance())        # GUI language (cs | en) – before the windows are created (also QLocale for numbers)
        super().__init__()
        self.settings = app_settings()
        self.dark = theme.apply(QApplication.instance(), str(self.settings.value("theme", "system")), stylesheet=False)
        self.setWindowTitle(APP_NAME)
        self.resize(1600, 950)
        # stav dokumentu
        self.project_path: Path | None = None
        self.step_path: Path | None = None
        self.parts = []
        self.study: Study | None = None
        self.mesh = None
        self.results = None
        self.motion_result = None         # mechanism result (fea_core.motion.MotionResult, Motion step)
        self._motion_sig = None           # signature of the mechanism that motion_result belongs to (motion_panel)
        self.face_cache: dict = {}        # part -> [face_props]
        self.sel_faces: dict[str, list] = {}   # selection -> [(part, face)]
        self.dirty = False
        self._thread = None
        self._workdir = Path(tempfile.mkdtemp(prefix="feastudio_"))
        self._undo: list[str] = []
        self._undo_pos = -1
        self._thread_sig = None                   # study signature when the mesh / calculation started (study_signature)
        self._mesh_sig = None                     # signature of the meshing inputs (MESH_KEYS) that self.mesh belongs to
        self._thread_mesh_sig = None
        self._rigid = None                        # cache of the rigid-motion check (signature, result)
        self._rb_sess = None

        self.view = FeaView()
        self.view.facesPicked.connect(self.on_faces_picked)
        self.view.probed.connect(self.on_probe)
        self.view.partsPicked.connect(self.on_parts_picked)
        self.view.edgePicked.connect(self.on_edge_picked)
        self.view.edgesPicked.connect(lambda _: self.update_status())
        self.view.pointsPicked.connect(lambda _: self.update_status())
        self.sel_edges: dict[str, list] = {}   # edge selection -> [(part, edge)]
        self.sel_points: dict[str, list] = {}  # point selection -> [(part, (x, y, z))]
        self.edge_cache: dict = {}             # part -> [edge_props] (order of geometry.list_edges = FeaView.part_edges)
        self.view.pickNote.connect(lambda t: self.statusBar().showMessage(t, 4000))
        self.view.contextMenuAt.connect(self.view_context_menu)     # right click without a drag (drag = rotate)
        self.view.screen_pick = self.frame_screen_pick              # click on a node / member of the frame
        self.spacemouse = None                    # 3D mouse 3Dconnexion (Windows raw input), see showEvent
        if sys.platform == "win32":
            try:
                from .spacemouse import SpaceMouse
                self.spacemouse = SpaceMouse(self.settings, self)
                self.spacemouse.motion.connect(self._space_motion)
                self.spacemouse.button.connect(self._space_button)
            except Exception:  # noqa: BLE001 – the program works without it
                self.spacemouse = None
        self._point_inputs: list[tuple] = []      # point field [mm] in forms (Use centre of edge), see register_point_input
        self._edge_info = None                    # selected edge (filter Edge)
        self._hs_rows: list = []                  # maximum places in the results table
        self._diag_sess = None
        self._pick_hook = None                    # a click in the results is taken by the dialog (progress along a line, convergence)
        self._aux_threads: list = []              # dialog threads (convergence, print refinement) – busy()
        self._path_dlg = self._conv_dlg = None
        self._r_view = (0.0, None)                # (deformation scale, displacements) of the displayed result
        self.surface_names: set[str] = set()      # plate parts from STEP (shells; the 3D mesh does not mesh them)
        self._voxel_dlg = None

        self._build_tree()
        # containers first into the window, then the panels: moving a finished widget tree under a new parent reapplies
        # the style (QSS) to all children – previously 3× (stack → splitter → window), window start ~2 s slower
        self.panel_stack = QStackedWidget()
        self.panel_stack.setMinimumWidth(460)
        # "Studio" layout: left tool shelf (tb_shelf), centre 3D window, right study tree above the step panel
        side = QSplitter(Qt.Vertical)
        side.setObjectName("side_split")
        side.addWidget(self.tree)
        side_panel = QWidget()                           # step tabs of the phase (tb_steps) above the step panel
        self._side_layout = QVBoxLayout(side_panel)
        self._side_layout.setContentsMargins(0, 0, 0, 0)
        self._side_layout.setSpacing(0)
        self._side_layout.addWidget(self.panel_stack, 1)
        side.addWidget(side_panel)
        side.setStretchFactor(0, 1)                      # tree about a third of the height, step panel the rest
        side.setStretchFactor(1, 2)
        side.setChildrenCollapsible(False)
        center = QWidget()                               # view bar (tb_view) above the 3D window
        self._center_layout = QVBoxLayout(center)
        self._center_layout.setContentsMargins(0, 0, 0, 0)
        self._center_layout.setSpacing(0)
        self._center_layout.addWidget(self.view, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(center)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)
        self.side_split = side
        self._build_panels()
        self._build_toolbars()
        self._build_docks()
        theme.apply_stylesheet(QApplication.instance())   # QSS once over the finished window (theme.apply)
        split.setSizes([1180, 520])
        side.setSizes([300, 600])
        self.st_sel = QLabel("")
        self.st_info = QLabel("")
        self.statusBar().addWidget(self.st_info, 1)
        self.btn_edge_point = QPushButton("Use center")
        help_tip(self.btn_edge_point, "Circle / arc center (else edge midpoint) to the clipboard and the point field",
                 "Copy the circle / arc center (edge midpoint for other edges) to the clipboard and to the point field of the open form (remote load / support)")
        self.btn_edge_point.clicked.connect(self.use_edge_point)
        self.btn_edge_point.setVisible(False)
        self.statusBar().addPermanentWidget(self.btn_edge_point)
        self.status_progress = TaskProgress()
        self.status_progress.setFixedWidth(340)
        self.statusBar().addPermanentWidget(self.status_progress)
        self.statusBar().addPermanentWidget(self.st_sel)
        self.st_model = QLabel("")
        self.st_model.setObjectName("modelStatus")
        self.st_model.setTextFormat(Qt.PlainText)
        help_tip(self.st_model, "Result equilibrium, mesh element count, solve time and units",
                 "Model summary: largest relative imbalance of reactions and loads over the load cases (√ = below the check limit, otherwise the result is invalid), number of mesh elements, time of the last solve in this window and the program units (mm, N, MPa, t, s).")
        self.statusBar().addPermanentWidget(self.st_model)
        self.set_stage(0)
        self.refresh_all()
        from .tour import install_help                   # Help bar (F1), example guide, manual
        install_help(self)
        from .extras import install_extras              # Tools: parametric study, job queue; View: Python console; recent files
        install_extras(self)
        compact(self.panel_stack)                        # narrower panels (menus, long form rows)
        self._started = True                             # optional panels built later: compact + form tips in _build_lazy_panel
        # Dynamic IPC Bridge (control from an LLM / MCP)
        from fea_core.mcp.session import Session
        self._mcp_session = Session(base_dir=self._workdir)
        self._mcp_tools_cache = None
        self.ipc_server = None
        import os as _os
        if _os.environ.get("FEA_NO_IPC") != "1":     # tests: the pipe is per user – other FEA Studio clients would reach the test window
            try:
                from .ipc_server import IpcServer
                self.ipc_server = IpcServer(self)
            except Exception as e:  # noqa: BLE001
                self.log_msg(f"Warning: the IPC server could not be started: {e}")
        # dropping .step / .stp / .fea from Explorer anywhere on the window (also over the 3D OCCT window, which is a native window
        # without its own OLE target; Windows passes the drop to the main window and Qt sends it by position: dragEnterEvent below)
        self.setAcceptDrops(True)
        if os.environ.get("FEA_PREWARM") == "1":
            QTimer.singleShot(800, _prewarm_imports)     # result libraries in the background (first display without waiting)

    # ================================================================== 3D mouse, context menu of the 3D window
    def showEvent(self, e):
        super().showEvent(e)
        self._fit_toolbars()
        if not getattr(self, "_docks_sized", False):      # bottom dock low – the 3D window has priority (Studio layout)
            self._docks_sized = True
            QTimer.singleShot(0, lambda: self.resizeDocks(
                [self.dock_log], [max(110, int(self.height() * 0.15))], Qt.Vertical))
            from .widgets import _ui_settings, remember_ui
            if remember_ui() and str(_ui_settings().value("ui/popout/panel_detached", False)).lower() in ("1", "true"):
                QTimer.singleShot(0, lambda: self.set_panel_detached(True))      # step panel in its own window as last time
        sm = self.spacemouse
        if sm is not None and not sm.registered:
            try:
                sm.attach(int(self.winId()))      # raw input registration – with no device / driver it does nothing
            except Exception:  # noqa: BLE001
                pass

    def nativeEvent(self, eventType, message):
        sm = self.spacemouse
        if sm is not None and sm.registered:
            try:
                sm.handle(message)
            except Exception:  # noqa: BLE001 – a faulty message must not crash the window
                pass
        return super().nativeEvent(eventType, message)

    def _space_motion(self, v, dt):
        if QApplication.activeModalWidget() is None:
            self.view.space_motion(v, dt)

    def _space_button(self, action):
        if QApplication.activeModalWidget() is not None:
            return
        if action == "fit":
            self.view.fit_all()
        elif action in ("roll_cw", "roll_ccw"):
            self.view.roll(math.pi / 2 if action == "roll_ccw" else -math.pi / 2)
        else:
            self.view.set_orientation(action)

    def spacemouse_settings(self):
        from .spacemouse import SpaceMouseDialog
        if self.spacemouse is not None:
            SpaceMouseDialog(self.spacemouse, self).exec()

    def build_view_menu(self):
        """Context menu of the 3D window (right click without a drag): show all, zoom to selection, views, 3D mouse."""
        from PySide6.QtWidgets import QMenu
        v = self.view
        m = QMenu(self)
        m.addAction("Fit all\tF", v.fit_all)
        a = m.addAction("Zoom to selection (double-click)", v.zoom_selection)
        a.setEnabled(v.has_selection())
        if v.mode == "faces" and (v.selected_faces() or v.selected_edges() or v.selected_points()):
            def clear():
                v.clear_selection()
                self.update_status()
            m.addAction("Clear selection", clear)

        # context actions for a body / part (exclude from the calculation in the Model section)
        sel_parts = v.selected_parts() if hasattr(v, "selected_parts") else []
        cur_pi = None
        if hasattr(v, "_part_at_cursor"):
            cur_pi = v._part_at_cursor()
        if cur_pi is None and hasattr(v, "_face_at_cursor"):
            cur_face = v._face_at_cursor()
            if cur_face is not None and 0 <= cur_face[0] < len(self.parts):
                cur_pi = cur_face[0]

        if not sel_parts and cur_pi is not None and 0 <= cur_pi < len(self.parts):
            v.set_selected_parts([cur_pi])
            sel_parts = [cur_pi]

        target_parts = [self.parts[i].name for i in sel_parts if 0 <= i < len(self.parts)]

        if self.study and target_parts:
            m.addSeparator()
            ign = set(self.study.ignored_parts)
            all_ignored = all(p in ign for p in target_parts)
            if len(target_parts) == 1:
                pname = target_parts[0]
                if all_ignored:
                    m.addAction(f"Include part '{pname}' in the analysis", lambda parts=target_parts: self.set_ignored(parts, False))
                else:
                    m.addAction(f"Exclude part '{pname}' from the analysis", lambda parts=target_parts: self.set_ignored(parts, True))
            else:
                if all_ignored:
                    m.addAction(f"Include selected parts ({len(target_parts)}) in the analysis", lambda parts=target_parts: self.set_ignored(parts, False))
                else:
                    m.addAction(f"Exclude selected parts ({len(target_parts)}) from the analysis", lambda parts=target_parts: self.set_ignored(parts, True))

        m.addSeparator()
        views = m.addMenu("View")
        for key, txt in (("iso", "Isometric"), ("front", "Front"), ("back", "Back"), ("top", "Top"),
                         ("bottom", "Bottom"), ("left", "Left"), ("right", "Right")):
            views.addAction(txt, lambda k=key: v.set_orientation(k))
        if self.spacemouse is not None:
            m.addSeparator()
            m.addAction("3D mouse (3Dconnexion)…", self.spacemouse_settings)
        return m

    def frame_screen_pick(self, x, y) -> bool:
        """Click in 3D on a node / member of the frame: Frames step = table row, Loads step with a frame load = node / member into the form.
        True = the click is handled (face selection does not change).
        """
        if self.study is None or not self.study.frame.nodes:
            return False
        stage = STAGES[getattr(self, "stage", 0)]
        t = self.ld_type.currentText() if hasattr(self, "ld_type") else ""
        if stage not in ("Members", "Loads") or (stage == "Loads" and t not in (FRAME_NODE, FRAME_MEMBER)):
            return False
        hit = self.frame_panel.pick(x, y)
        if hit is None:
            return False
        kind, name = hit
        if stage == "Members":
            self.frame_panel.select(kind, name)
        elif kind == "node" and t == FRAME_NODE:
            self.ld_fr_node.setCurrentText(name)
        elif kind == "member" and t == FRAME_MEMBER:
            self.ld_fr_member.setCurrentText(name)
        else:
            return False
        self.statusBar().showMessage(("Node " if kind == "node" else "Member ") + name, 4000)
        return True

    def frame_assessment_changed(self):
        """Inputs of the EN 1993 assessment changed – FEM results stay valid, the assessment is recomputed."""
        if self.results is not None and STAGES[self.stage] == "Results":
            self.show_result()

    def view_context_menu(self, pos):
        self.build_view_menu().exec(pos)

    # ================================================================== stavba UI
    def _act(self, text, slot, shortcut=None, tip=None, icon=None):
        a = QAction(text, self)
        if icon:
            theme.set_icon(a, icon)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setToolTip((tip or text) + (f"  [{shortcut}]" if shortcut else ""))
        a.triggered.connect(slot)
        self.addAction(a)
        return a

    def _build_toolbars(self):
        """Header in one line (Studio × Tables design): ▲ FEA Studio, menus File / Edit / View /
        Tools / Help, phase tabs (tb_flow), document name on the right, Replace geometry and Run calculation.
        """
        from PySide6.QtWidgets import QMenu, QToolButton
        tb = QToolBar("Menus")
        tb.setObjectName("tb_file")
        tb.setMovable(False)
        logo = QLabel("▲")
        logo.setObjectName("brandLogo")
        tb.addWidget(logo)
        brand = QLabel("FEA Studio")
        brand.setObjectName("brand")
        brand.setToolTip("FEA Studio – finite element strength analysis (CalculiX, Gmsh)")
        self._brand_act = tb.addWidget(brand)
        a_open = self._act("Open…", self.open_dialog, "Ctrl+O", "Open STEP or .fea project", "open")
        a_save = self._act("Save", self.save, "Ctrl+S", "Save .fea project", "save")
        a_save_as = self._act("Save as…", self.save_as, "Ctrl+Shift+S", icon="save_as")
        a_report = self._act("Report…", self.export_report, "Ctrl+P", "Technical analysis report to Word (.docx)",
                             "report")
        a_undo = self._act("Undo", self.undo, "Ctrl+Z", "Undo the last study change", "undo")
        a_redo = self._act("Redo", self.redo, "Ctrl+Y", "Redo the undone change", "redo")
        self.menu_buttons: dict[str, QToolButton] = {}

        def menu(title, tip):
            b = QToolButton()
            b.setObjectName("menuButton")
            b.setText(title)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setPopupMode(QToolButton.InstantPopup)
            b.setToolTip(tip)
            m = QMenu(b)
            b.setMenu(m)
            tb.addWidget(b)
            self.menu_buttons[title] = b
            return m
        m = menu("File", "Open, save, report, replace geometry")
        self.menu_file = self.menu_buttons["File"]        # target of the guide (Save, Report)
        for a in (a_open, a_save, a_save_as):
            m.addAction(a)
        m.addSeparator()
        m.addAction(a_report)
        m.addSeparator()
        m.addAction("Replace geometry (keep setup)…", self.replace_geometry_dialog)
        m = menu("Edit", "Undo, redo")
        m.addAction(a_undo)
        m.addAction(a_redo)
        m = self.view_menu = menu("Show", "View, section, workflow steps, appearance, language")
        m.addAction(self._act("Fit (whole model)", lambda: self.view.fit_all(), "F", "Fit all", "fit"))
        sub = m.addMenu("View direction")
        for key, txt in VIEW_DIRECTIONS:
            sub.addAction(theme.icon(key), txt, lambda k=key: self.view.set_orientation(k))
        m.addSeparator()
        steps = m.addMenu("Optional workflow steps")
        steps.aboutToShow.connect(lambda: self._fill_more_menu(steps))
        m.addSeparator()
        self.act_panel_window = a = m.addAction("Step panel in a separate window")
        a.setCheckable(True)
        a.setToolTip("Moves the step panel into its own window; closing the window returns it")
        a.toggled.connect(lambda on: self.set_panel_detached(on))
        self.act_tables_window = a = m.addAction("Tables in a separate window")
        a.setCheckable(True)
        a.setToolTip("Bottom Tables dock in its own window; closing or dragging back returns it")
        a.toggled.connect(lambda on: self.set_tables_floating(on))
        m.addSeparator()
        # the Appearance / Language / Tools menus belong to buttons without a parent – _menu_sources keeps them (otherwise Python would delete them)
        self._menu_sources = [self._theme_button(), i18n.language_button(self), self._tools_button()]
        th, lang, tools = (b.menu() for b in self._menu_sources)
        th.setTitle("Appearance")
        th.setIcon(theme.icon("theme"))
        m.addMenu(th)
        lang.setTitle("Language / Jazyk")
        lang.setIcon(self._menu_sources[1].icon())
        m.addMenu(lang)
        mt = menu("Tools", "Calculators, new beam structure, mesh convergence, modules")
        for a in tools.actions():
            mt.addAction(a)
        self.help_menu = menu("Help", "Help, What's this?, tours, manual")
        self.addToolBar(tb)
        self.tb_file = tb
        self._build_section_controls()
        self._build_flow_bar()
        rt = QToolBar("Document")
        rt.setObjectName("tb_right")
        rt.setMovable(False)
        rt.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        spacer = QWidget()                               # the last bar in the row gets the remaining space → everything to the right
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rt.addWidget(spacer)
        self.doc_label = QLabel("")
        self.doc_label.setObjectName("docLabel")
        self.doc_label.setTextFormat(Qt.PlainText)         # name of a study / file from a foreign project
        self._doc_act = rt.addWidget(self.doc_label)
        b = QPushButton("Replace geometry…")             # phase actions (HEADER_ACTIONS, _update_header_actions)
        b.clicked.connect(lambda: self._header_action(0))
        self.btn_replace_geo = self.btn_phase = b
        self._replace_act = rt.addWidget(b)
        b = QPushButton("")
        b.clicked.connect(lambda: self._header_action(1))
        self.btn_phase2 = b
        self._phase2_act = rt.addWidget(b)
        b = QPushButton("Run analysis")
        theme.set_icon(b, "run")
        theme.set_primary(b)
        help_tip(b, "Go to the Solve step and run the analysis  [F5]",
                 "Switches to the Solve step and runs the analysis (the mesh is created if missing). Same as the Run button in the Solve step.")
        b.clicked.connect(self.run_from_header)
        self.btn_run_header = b
        self._run_act = rt.addWidget(b)
        self._act("Run analysis", self.run_from_header, "F5", "Run analysis")
        self.addToolBar(rt)
        self.tb_right = rt
        for t in (self.tb_file, self.tb_flow, rt):     # header 32 px as in the design (the style adds a 6 px border)
            t.layout().setContentsMargins(4, 0, 4, 0)
        self._build_shelf()
        self._build_view_bar()

    def _update_header_actions(self, phase: str):
        """Header buttons by phase (design): Model Replace geometry, Assembly Recognise joints, Calculation mesh
        convergence, Results Export and Create report (instead of Run calculation).
        """
        acts = HEADER_ACTIONS.get(phase, [])
        self._header_acts = acts
        for i, (b, wa) in enumerate(((self.btn_phase, self._replace_act), (self.btn_phase2, self._phase2_act))):
            if i < len(acts):
                text, _slot, short, long = acts[i]
                b.setText(text)
                help_tip(b, short, long)
            wa.setVisible(i < len(acts) and self._header_lvl < 2)
        self._run_act.setVisible(phase != "Evaluation")
        theme.set_primary(self.btn_phase2, phase == "Evaluation")

    def _header_action(self, i: int):
        acts = getattr(self, "_header_acts", [])
        if i < len(acts):
            self._shelf_run(acts[i][1])

    def run_from_header(self):
        """Run calculation from the header: Solve step + its button (same readiness check and availability)."""
        if self.busy() or self.study is None:
            return
        self.set_stage(STAGE["Solve"])
        if self.btn_run.isEnabled():
            self.btn_run.click()

    def _build_section_controls(self):
        """Section through the model: axis menu, position slider and "flip" in the panel that the Section button above the 3D window expands."""
        from PySide6.QtWidgets import QMenu, QToolButton, QWidgetAction
        b = QToolButton()
        b.setObjectName("sec_button")
        b.setText("Section")
        theme.set_icon(b, "section")
        b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b.setPopupMode(QToolButton.InstantPopup)
        b.setCheckable(True)
        help_tip(b, "Section the model by a plane normal to an axis or by the plane of a selected face",
                 "Section by a plane normal to an axis or by the plane of a selected face (select a face, then “face”).<br>The slider moves the plane within the model extent, “flip” shows the other part. In the results the section through the mesh interior is shown in colour too.")
        menu = QMenu(b)
        box = QWidget()
        lay = QGridLayout(box)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.addWidget(QLabel("Plane"), 0, 0)
        self.sec_axis = QComboBox()
        self.sec_axis.addItems(["off", "X", "Y", "Z", "face"])
        help_tip(self.sec_axis, "Section by a plane normal to an axis, or by the plane of the selected face",
                 "Section by a plane normal to an axis or by the plane of the selected face (select a face, then “face”).<br>In the results the section through the inside of the mesh is also shown in color.")
        self.sec_axis.currentIndexChanged.connect(self.on_sec_axis)
        self.sec_axis.currentIndexChanged.connect(self._sync_sec_button)
        lay.addWidget(self.sec_axis, 0, 1)
        lay.addWidget(QLabel("Position"), 1, 0)
        self.sec_pos = QSlider(Qt.Horizontal)
        self.sec_pos.setRange(0, 1000)
        self.sec_pos.setValue(500)
        self.sec_pos.setFixedWidth(180)
        self.sec_pos.setToolTip("Section plane position within the model extent (for a face-based section, offset along its normal)")
        self.sec_pos.valueChanged.connect(self.update_section)
        lay.addWidget(self.sec_pos, 1, 1)
        self._sec_face = None              # (point, normal) of the section plane from a face
        self.sec_flip = QCheckBox("flip")
        self.sec_flip.setToolTip("Show the other part of the model (flip the section side)")
        self.sec_flip.toggled.connect(self.update_section)
        lay.addWidget(self.sec_flip, 2, 1)
        wa = QWidgetAction(menu)
        wa.setDefaultWidget(box)
        menu.addAction(wa)
        b.setMenu(menu)
        self.sec_button = b

    def _sync_sec_button(self, k):
        """The Section button shows that the section is on (checked, "Section Y")."""
        on = k > 0
        self.sec_button.setChecked(on)
        self.sec_button.setText(f"Section {self.sec_axis.currentText()}" if on and k != self.SEC_FACE else
                                ("Face section" if on else "Section"))

    # ------------------------------------------------------------------ tool shelf on the left (Studio layout)
    def _build_shelf(self):
        """Vertical tool shelf: commands of the current step in groups (SHELF, changes in set_stage).
        A shelf command = click on the button of the step panel (same behaviour and availability) or a window method.
        """
        sh = QToolBar("Tools")
        sh.setObjectName("tb_shelf")
        sh.setMovable(False)
        sh.setFloatable(False)
        sh.setOrientation(Qt.Vertical)
        sh.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        sh.setIconSize(QSize(18, 18))

        def label(text):
            lb = QLabel(text)
            lb.setAlignment(Qt.AlignCenter)
            return sh.addWidget(lb)
        # step commands: one set of actions per step, visible only in its own step (_update_shelf)
        self._shelf_label = QLabel("")
        self._shelf_label.setAlignment(Qt.AlignCenter)
        self._shelf_label_act = sh.addWidget(self._shelf_label)
        self._shelf_acts: dict[str, list[QAction]] = {}
        self._shelf_heads: dict[str, list[QAction]] = {}       # group headings (labels) per step
        for stage, cmds in SHELF.items():
            acts, heads = [], []
            for item in cmds:
                if len(item) == 2:                             # ("#", "NÁZEV") = group heading
                    if acts or heads:
                        sep = sh.addSeparator()
                        sep.setVisible(False)
                        heads.append(sep)
                    h = label(item[1])
                    h.setVisible(False)
                    heads.append(h)
                    continue
                txt, ic, target, tip = item
                a = QAction(txt, self)
                theme.set_icon(a, ic)
                a.setToolTip(tip)
                a.triggered.connect(lambda _=False, t=target: self._shelf_run(t))
                a.setVisible(False)
                sh.addAction(a)
                acts.append(a)
            self._shelf_acts[stage] = acts
            self._shelf_heads[stage] = heads
        self._shelf_lvl = 0
        self.addToolBar(Qt.LeftToolBarArea, sh)
        self.tb_shelf = sh

    # ------------------------------------------------------------------ bar above the 3D window
    def _build_view_bar(self):
        """Bar above the 3D window (design): on the left the help for the selection, on the right Select ▾ (selection filter P/B/H/E/V),
        View ▾, Section, Fit. (An overlay directly in the 3D window is not possible – the OCCT window is native.)
        """
        from PySide6.QtWidgets import QMenu, QToolButton
        vb = QFrame()                                    # not a QToolBar: the help on the left is shortened in a narrow window
        vb.setObjectName("tb_view")
        row = QHBoxLayout(vb)
        row.setContentsMargins(6, 3, 6, 3)
        row.setSpacing(3)
        self.view_title = QLabel("")                     # view description per step (update_view_info)
        self.view_title.setObjectName("viewTitle")
        self.view_title.setTextFormat(Qt.PlainText)
        row.addWidget(self.view_title)
        self.view_legend = QLabel("")                    # legend of material colours (colour by material)
        self.view_legend.setObjectName("viewLegend")
        self.view_legend.setTextFormat(Qt.RichText)
        row.addWidget(self.view_legend)
        self.view_hint = ElidedLabel("")
        self.view_hint.setObjectName("viewHint")
        row.addWidget(self.view_hint, 1)

        def vb_add(w):
            w.setIconSize(QSize(16, 16))
            row.addWidget(w)
        pb = QToolButton()
        pb.setObjectName("pick_button")
        pb.setPopupMode(QToolButton.InstantPopup)
        pb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        help_tip(pb, "What a click in 3D selects: face, body, hole, edge, point (P / B / H / E / V)",
                 "3D selection filter: <b>Face</b> (P), <b>Body</b> (B), <b>Hole</b> (H – all coaxial cylindrical faces), <b>Edge</b> (E – length, radius and center), <b>Point</b> (V – vertex or point on a face).")
        pm = QMenu(pb)
        fg = QActionGroup(self)
        self.filter_actions: dict[str, QAction] = {}
        for key, txt, sc, tip in PICK_FILTER_ACTIONS:
            a = self._act(txt, lambda _=False, k=key: self.set_pick_filter(k), sc, PICK_FILTER_SHORT[key], key)
            a.setWhatsThis(tr(tip).replace(".\n", ".<br>").replace("\n", " "))     # tr before editing the rows (catalogue)
            a.setCheckable(True)
            fg.addAction(a)
            pm.addAction(a)
            self.filter_actions[key] = a
        self.filter_actions["face"].setChecked(True)
        pb.setMenu(pm)
        self.pick_button = pb
        vb_add(pb)
        cb = QToolButton()                               # part colour: by part / by material
        cb.setObjectName("color_button")
        cb.setPopupMode(QToolButton.InstantPopup)
        cb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        cb.setToolTip("Part color in 3D: each part a different color, or by material (with legend)")
        cm = QMenu(cb)
        cg = QActionGroup(self)
        self.color_actions = {}
        for key, txt in (("part", "part"), ("material", "material")):
            a = cm.addAction(f"Color: {txt}", lambda k=key: self.set_color_mode(k))
            a.setCheckable(True)
            cg.addAction(a)
            self.color_actions[key] = a
        cb.setMenu(cm)
        self.color_button = cb
        vb_add(cb)
        qb = QToolButton()                               # results: quantity (menu from r_qty)
        qb.setObjectName("qty_button")
        qb.setPopupMode(QToolButton.InstantPopup)
        qb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        qb.setToolTip("Displayed result quantity")
        qm = QMenu(qb)
        qm.aboutToShow.connect(lambda: self._fill_qty_menu(qm))
        qb.setMenu(qm)
        self.qty_button = qb
        vb_add(qb)
        def menu_button(name, tip, items, slot):
            b = QToolButton()
            b.setObjectName(name)
            b.setPopupMode(QToolButton.InstantPopup)
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setToolTip(tip)
            mm = QMenu(b)
            grp = QActionGroup(self)
            acts = {}
            for key, txt in items:
                a = mm.addAction(txt, lambda k=key, bb=b: slot(k, bb))
                a.setCheckable(True)
                grp.addAction(a)
                acts[key] = a
            b.setMenu(mm)
            b._acts = acts
            first = items[0][0]
            acts[first].setChecked(True)
            b.setText(acts[first].text())
            vb_add(b)
            return b
        self.transp_button = menu_button(
            "transp_button", "Parts solid or transparent (bolts and holes inside the assembly are visible)",
            (("solid", "Solid"), ("transparent", "Transparent")), self._on_transparency)
        self.style_button = menu_button(
            "style_button", "Part style: shaded with face edges, shaded, wireframe",
            (("shaded_edges", "Shaded with edges"), ("shaded", "Shaded"), ("wire", "Wireframe")),
            self._on_display_style)
        self.mesh_show_button = menu_button(
            "mesh_show_button", "What to show in 3D in the Mesh and Solve steps: the finite element mesh or the parts",
            (("mesh", "Show: Mesh"), ("parts", "Show: Parts")), self._on_mesh_show_mode)
        self.mesh_color_button = menu_button(
            "mesh_color_button", "Mesh color: uniform, or by element quality (1 = regular tetrahedron)",
            (("plain", "Color: uniform"), ("quality", "Color: element quality")), self._on_mesh_color)
        self.view_edges = QToolButton()                  # results: mesh edges (r_edges)
        self.view_edges.setObjectName("edges_button")
        self.view_edges.setText("Mesh edges")
        self.view_edges.setCheckable(True)
        self.view_edges.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.view_edges.setToolTip("Show mesh element edges over the color map")
        self.view_edges.toggled.connect(lambda on: self.r_edges.setChecked(on))
        vb_add(self.view_edges)
        db = QToolButton()
        db.setObjectName("dir_button")
        db.setText("View")
        theme.set_icon(db, "iso")
        db.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        db.setPopupMode(QToolButton.InstantPopup)
        db.setToolTip("View direction: isometric, front, top, right (more in the 3D view context menu)")
        dm = QMenu(db)
        for key, txt in VIEW_DIRECTIONS:
            dm.addAction(theme.icon(key), txt, lambda k=key: self.view.set_orientation(k))
        db.setMenu(dm)
        vb_add(db)
        vb_add(self.sec_button)
        fit = QToolButton()
        fit.setObjectName("fit_button")
        fit.setText("Fit")
        theme.set_icon(fit, "fit")
        fit.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        fit.setToolTip("Show the whole model  [F]")
        fit.clicked.connect(lambda: self.view.fit_all())
        vb_add(fit)
        self._center_layout.insertWidget(0, vb)
        self.tb_view = vb
        mode = str(self.settings.value("view/color", "part")) if _remember_ui() else "part"
        self._color_mode = mode if mode in self.color_actions else "part"
        self.color_actions[self._color_mode].setChecked(True)
        self.color_button.setText(self.color_actions[self._color_mode].text())
        self._sync_pick_button("face")
        self.r_qty.currentIndexChanged.connect(lambda _: self.update_view_info())
        self.r_step.currentIndexChanged.connect(lambda _: self.update_view_info())
        self.r_edges.toggled.connect(lambda on: self.view_edges.setChecked(on))
        from .properties import PropertiesCard           # Properties of the selected item above the step panel
        self.props_card = PropertiesCard(self)
        self._side_layout.insertWidget(1, self.props_card)
        pc = self.props_card.schedule
        for sig in (self.view.partsPicked, self.view.facesPicked, self.con_list.currentRowChanged,
                    self.load_list.currentRowChanged, self.lc_combo.currentIndexChanged,
                    self.mat_parts.itemSelectionChanged, self.joints_panel.bolt_list.currentRowChanged):
            sig.connect(pc)
        if hasattr(self, "model_table"):
            self.model_table.itemSelectionChanged.connect(pc)

    @staticmethod
    def _pick(button, key):
        button._acts[key].setChecked(True)
        button.setText(button._acts[key].text())

    def _on_transparency(self, key, button):
        self._pick(button, key)
        self.view.set_user_transparency(0.65 if key == "transparent" else 0.0)

    def _on_display_style(self, key, button):
        self._pick(button, key)
        self.view.set_display_style(key)

    def _on_mesh_show_mode(self, key, button):
        self._pick(button, key)
        if self.ms_show.isChecked() != (key == "mesh"):
            self.ms_show.setChecked(key == "mesh")

    def _on_mesh_color(self, key, button):
        self._pick(button, key)
        self._mesh_color = key
        if getattr(self, "_mesh_view", False):
            self.show_mesh()

    def _fill_qty_menu(self, menu):
        menu.clear()
        for i in range(self.r_qty.count()):
            a = menu.addAction(self.r_qty.itemText(i), lambda k=i: self.r_qty.setCurrentIndex(k))
            a.setCheckable(True)
            a.setChecked(i == self.r_qty.currentIndex())

    def set_color_mode(self, mode: str):
        """Part colour in 3D: "part" (each part a different colour) or "material" (by material, legend above the 3D view)."""
        self._color_mode = mode
        self.color_actions[mode].setChecked(True)
        self.color_button.setText(self.color_actions[mode].text())
        if _remember_ui():
            self.settings.setValue("view/color", mode)
        self.apply_part_colors()
        self.update_view_info()

    def material_colors(self) -> dict[str, tuple]:
        """Colour by the materials of the study (order by the first part with a material), a part without a material is grey."""
        out: dict[str, tuple] = {}
        for p in self.parts:
            try:
                m = self.study.material_of(p.name).name
            except (KeyError, AttributeError):
                continue
            if m not in out:
                out[m] = MATERIAL_COLORS[len(out) % len(MATERIAL_COLORS)]
        return out

    def apply_part_colors(self):
        if self.study is None or not self.parts:
            return
        if self._color_mode != "material":
            self.view.set_part_colors(None)
            return
        cols = self.material_colors()
        per = []
        for p in self.parts:
            try:
                per.append(cols[self.study.material_of(p.name).name])
            except (KeyError, AttributeError):
                per.append(NO_MATERIAL_COLOR)
        self.view.set_part_colors(per)

    def update_view_info(self):
        """Bar above the 3D window: description of the view per step, material legend, buttons per step (part colour outside
        results, quantity and mesh edges in results).
        """
        if not hasattr(self, "view_title"):
            return
        import html
        name = STAGES[getattr(self, "stage", 0)]
        st = self.study
        res_view = name in ("Results", "Fatigue") and self.results is not None
        self.color_button.setVisible(name in ("Model", "Materials", "3D printing"))
        self.transp_button.setVisible(name in ("Joints", "Contacts", "Members"))
        self.style_button.setVisible(name in ("Constraints", "Loads", "Heat", "Motion", "Dynamics"))
        mesh_stage = name == "Mesh" and self.mesh is not None
        self.mesh_show_button.setVisible(mesh_stage)
        self.mesh_color_button.setVisible(mesh_stage and self.ms_show.isChecked())
        if mesh_stage:
            self._pick(self.mesh_show_button, "mesh" if self.ms_show.isChecked() else "parts")
        self.qty_button.setVisible(res_view)
        self.view_edges.setVisible(res_view)
        legend = ""
        if st is None:
            title = ""
        elif res_view:
            title = f"{self.r_qty.currentText()} · {self.r_step.currentText()}"
            self.qty_button.setText(f"Quantity: {self.r_qty.currentText()}")
        elif name == "Mesh" or name == "Solve":
            m = self.mesh
            title = (f"{m.element_type} · {fmt(st.mesh.size, 3)} mm · {fmt(len(m.tets))} elements · mesh valid √"
                     if m is not None else f"The mesh is created during the analysis · {fmt(st.mesh.size, 3)} mm")
        elif name in ("Constraints", "Loads", "Heat", "Dynamics", "Motion"):
            lc = st.load_cases[self.lc_combo.currentIndex()] if st.load_cases and self.lc_combo.currentIndex() >= 0 \
                else None
            title = (f"{lc.name} · {len(lc.loads)} loads · {len(st.constraints)} supports" if lc is not None
                     else f"{len(st.constraints)} supports")
        elif name in ("Joints", "Contacts", "Members"):
            title = f"Joints: {len(st.bolts)} bolts · {len(st.welds)} welds · {len(st.contacts)} contacts"
        else:
            title = "Color by material" if self._color_mode == "material" else "Color by part"
        if st is not None and not res_view and self._color_mode == "material" and name not in ("Mesh",):
            cols = self.material_colors()
            counts: dict[str, int] = {}
            missing = 0
            for p in self.parts:
                try:
                    k = st.material_of(p.name).name
                    counts[k] = counts.get(k, 0) + 1
                except (KeyError, AttributeError):
                    missing += 1
            chips = [f"<span style='color:#{r:02x}{g:02x}{b:02x}'>■</span> {html.escape(k)} · {counts[k]}"
                     for k, (r, g, b) in cols.items()]
            if missing:
                r, g, b = NO_MATERIAL_COLOR
                chips.append(f"<span style='color:#{r:02x}{g:02x}{b:02x}'>■</span> no material · {missing}")
            legend = "&nbsp;&nbsp;".join(chips)
        self.view_title.setText(title)
        self.view_title.setVisible(bool(title))
        self.view_legend.setText(legend)
        self.view_legend.setVisible(bool(legend))

    def _sync_pick_button(self, key: str):
        """Text of the Select button and the help on the left above the 3D window by the selection filter."""
        txt = next(t for k, t, _, _ in PICK_FILTER_ACTIONS if k == key)
        self.pick_button.setText(f"Select: {txt}")
        theme.set_icon(self.pick_button, key)
        self.view_hint.setText(PICK_FILTER_SHORT[key])

    def _shelf_run(self, target):
        """Shelf command: "attribute" of the window – button (click, only if enabled) or method; "stage:Step" switches the step;
        a tuple (attribute, value) sets the text of a menu (load type).
        """
        def resolve(path):
            o = self
            for part in path.split("."):
                o = getattr(o, part, None)
                if o is None:
                    return None
            return o
        if isinstance(target, str) and target.startswith("@"):     # panel button of the step by its text
            from PySide6.QtWidgets import QAbstractButton
            panel = self.panel_stack.currentWidget()
            for b in panel.findChildren(QAbstractButton) if panel is not None else []:
                if b.text() == target[1:]:
                    if b.isEnabled():
                        b.click()
                    return
            return
        if isinstance(target, str) and ":" in target:
            kind, path = target.split(":", 1)
            if kind == "stage":
                self.set_stage(STAGE[path])
                return
            obj = resolve(path)
            if obj is None:
                return
            if kind == "focus":
                obj.setFocus()
                if hasattr(obj, "selectAll"):
                    obj.selectAll()
            elif kind == "popup":
                obj.showPopup()
            elif kind == "raise":
                obj.show()
                obj.raise_()
            elif kind == "menu":
                obj.showMenu()
            return
        if isinstance(target, tuple):
            combo = resolve(target[0])
            if combo is not None:
                i = combo.findText(target[1])
                if i >= 0:
                    combo.setCurrentIndex(i)
            return
        obj = resolve(target)
        if obj is None:
            return
        if hasattr(obj, "click"):
            if obj.isEnabled():
                obj.click()
        elif callable(obj):
            obj()

    def _update_shelf(self, name: str):
        """The shelf shows the commands of the current step; a step without commands = only selection and views."""
        if not hasattr(self, "_shelf_acts"):
            return
        for stage, acts in self._shelf_acts.items():
            for a in acts:
                a.setVisible(stage == name)
        self._shelf_stage = name
        has = bool(self._shelf_acts.get(name))
        self._shelf_label.setText(SHELF_LABELS.get(name, name).upper() if has else "")
        self._shelf_has_cmds = has
        self._fit_shelf()

    def _fit_shelf(self):
        """Shelf in a low window: first without the step name, then buttons with the icon only (name in the label), so that the commands
        do not hide in the dropdown button ⋮.
        """
        sh = getattr(self, "tb_shelf", None)
        if sh is None or getattr(self, "_fitting_shelf", False):
            return
        self._fitting_shelf = True
        try:
            avail = sh.height() if sh.isVisible() else 10 ** 6
            for lvl in range(3):
                self._shelf_level(lvl)
                if sh.sizeHint().height() <= avail:
                    break
        finally:
            self._fitting_shelf = False

    def _shelf_level(self, lvl: int):
        self._shelf_lvl = lvl
        self._shelf_label_act.setVisible(False)          # the step name was replaced by the group headings (SHELF ("#", …))
        cur = getattr(self, "_shelf_stage", None)
        for stage, heads in self._shelf_heads.items():
            for h in heads:
                h.setVisible(stage == cur and (lvl < 1 or h.isSeparator()))
        style = Qt.ToolButtonIconOnly if lvl >= 2 else Qt.ToolButtonTextUnderIcon
        if self.tb_shelf.toolButtonStyle() != style:
            self.tb_shelf.setToolButtonStyle(style)

    # ------------------------------------------------------------------ progress bar by phases (F11)
    def _build_flow_bar(self):
        """Phase tabs in the header (tb_flow: Model, Assembly, Conditions, Calculation, Results) and step tabs of the current phase
        above the step panel (tb_steps, coloured dot = state of the step). Visibility is controlled by `_apply_flow_visibility`
        (required steps always, optional ones only when used / pinned / opened).
        """
        from PySide6.QtWidgets import QButtonGroup, QMenu, QToolBar, QToolButton
        wf = QToolBar("Phase")
        wf.setObjectName("tb_flow")
        wf.setMovable(False)
        wf.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._flow_cols = {name: GREY for name in STAGES}   # colours of the step states (refresh_flow)
        self._flow_sig = None
        self._icon_cache: dict = {}
        self._flow_pins: set[str] = set()            # pinned optional steps (+ menu)
        self._flow_show_all = False
        self._flow_revealed: set[str] = set()        # optional steps opened in this session
        self._phase_last: dict[str, str] = {}        # last opened step of the phase (click on the phase tab)
        if _remember_ui():
            try:
                self._flow_pins = {n for n in json.loads(str(self.settings.value("flow/pinned", "[]")))
                                   if n in OPTIONAL_STAGES}
            except (ValueError, TypeError):
                self._flow_pins = set()
            self._flow_show_all = str(self.settings.value("flow/show_all", "0")) in ("1", "true", "True")
        self._phase_buttons: dict[str, QToolButton] = {}
        self._phase_acts: dict[str, QAction] = {}
        pg = QButtonGroup(self)
        pg.setExclusive(True)
        for phase, names in PHASES:
            b = QToolButton()
            b.setObjectName("phaseTab")
            b.setText(phase_label(phase))
            b.setCheckable(True)
            b.setToolTip(f"{phase_label(phase)}: {PHASE_TIPS[phase]}")
            b.setWhatsThis(f"<b>{phase_label(phase)}</b><p>Workflow phase: {PHASE_TIPS[phase]}. The steps of the phase are tabs above the panel on the right.</p>")
            b.clicked.connect(lambda _=False, ph=phase: self.goto_phase(ph))
            pg.addButton(b)
            self._phase_buttons[phase] = b
            self._phase_acts[phase] = wf.addWidget(b)
        self._phase_group = pg
        self.addToolBar(wf)
        self.tb_flow = wf
        st = QToolBar("Steps")
        st.setObjectName("tb_steps")
        st.setMovable(False)
        st.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        st.setIconSize(QSize(14, 14))
        grp = QActionGroup(self)
        self.stage_actions = []                      # in STAGES order (also hidden ones – widgetForAction returns them)
        for name in STAGES:
            i = STAGE[name]
            a = QAction(theme.step_icon(0, GREY), name, self)
            a.setCheckable(True)
            a.setToolTip(f"{name}: {STAGE_TIPS[name]}")
            a.setWhatsThis(f"{STAGE_LONG[name]}<br>{STAGE_COLORS}")
            a.triggered.connect(lambda _=False, k=i: self.set_stage(k))
            grp.addAction(a)
            st.addAction(a)
            self.stage_actions.append(a)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        st.addWidget(spacer)
        mb = QToolButton()
        mb.setObjectName("flowMore")
        theme.set_icon(mb, "plus")
        mb.setToolButtonStyle(Qt.ToolButtonIconOnly)
        mb.setPopupMode(QToolButton.InstantPopup)
        mb.setToolTip("Optional steps: 3D printing, contacts, members, motion, heat, dynamics, fatigue")
        mb.setWhatsThis("<b>Optional steps</b><p>A step the study uses is among the tabs by itself; pin the others here (check box) or show all steps.</p>")
        mm = QMenu(mb)
        mm.aboutToShow.connect(lambda: self._fill_more_menu(mm))
        mb.setMenu(mm)
        self.flow_more = mb
        st.addWidget(mb)
        from .popout import popout_button
        pb = popout_button("Step panel in a separate window (larger, also on a second monitor)")
        pb.setWhatsThis("<b>Step panel in a separate window</b><p>The panel of the current step (materials, supports, loads, member nodes…) moves into its own window that can be enlarged or moved to a second monitor; the step tabs and the 3D view stay in the main window and the 3D window gets the freed space. Closing the window returns the panel.</p><p>Single sections of a panel have the same button in their header; the bottom Tables dock has it next to its tabs.</p>")
        pb.clicked.connect(lambda: self.set_panel_detached(not self.panel_detached()))
        self.btn_panel_popout = pb
        st.addWidget(pb)
        self._side_layout.insertWidget(0, st)
        self.tb_steps = st
        self._fit_levels = (-1, -1)                 # compaction of the header – see _fit_toolbars
        self._header_lvl = 0
        self._header_phase = None
        from PySide6.QtGui import QShortcut
        for key, d in (("Ctrl+PgDown", 1), ("Ctrl+PgUp", -1)):         # next / previous visible step
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda d=d: self.step_stage(d))

    def goto_phase(self, phase: str):
        """Phase tab: the last opened step of the phase, otherwise the first visible and enabled one."""
        names = dict(PHASES)[phase]
        last = self._phase_last.get(phase)
        if last in names and self.stage_visible(last) and self.stage_enabled(last):
            self.set_stage(STAGE[last])
            return
        for s in names:
            if self.stage_visible(s) and self.stage_enabled(s):
                self.set_stage(STAGE[s])
                return

    def stage_visible(self, name: str) -> bool:
        """Is the step on the progress bar? Required always; optional only when the study uses it (colour ≠ grey), it is pinned,
        all steps are shown, it was opened in this session, or it is current. A frame system without CAD hides
        the steps FRAME_ONLY_OFF and shows Frames.
        """
        fo = self.study is not None and self.frame_only()
        if fo and name in FRAME_ONLY_OFF:
            return False
        if name not in OPTIONAL_STAGES:
            return True
        if fo and name == "Members":
            return True
        return (self._flow_cols.get(name, GREY) != GREY or name in self._flow_pins or self._flow_show_all
                or name in self._flow_revealed or STAGES[getattr(self, "stage", 0)] == name)

    def stage_enabled(self, name: str) -> bool:
        """Is the step usable? (A hidden QAction always reports isEnabled() False – hence our own source of truth.)"""
        return not (self.study is not None and self.frame_only() and name in FRAME_ONLY_OFF)

    def _stage_used(self, name: str) -> bool:
        return self._flow_cols.get(name, GREY) != GREY

    def _flow_icon(self, n: int, color: str):
        key = (n, color)
        ic = self._icon_cache.get(key)
        if ic is None:
            ic = self._icon_cache[key] = theme.step_icon(n, color)
        return ic

    @staticmethod
    def _summary_color(colors: list[str]) -> str:
        for c in (RED, ORANGE, GREEN):
            if c in colors:
                return c
        return GREY

    def _apply_flow_visibility(self, fit: bool = True, force: bool = False):
        """Phase tabs (visible if the phase has at least one visible step; checked = phase of the current step) and the step tabs
        of the current phase with a state dot; then measures the header again (_fit_toolbars).
        """
        if not hasattr(self, "_phase_acts"):
            return
        cur = STAGES[getattr(self, "stage", 0)]
        vis = {n: self.stage_visible(n) for n in STAGES}
        sig = (tuple(vis.items()), tuple(self._flow_cols.items()), cur)
        if not force and sig == self._flow_sig:
            return
        self._flow_sig = sig
        for phase, names in PHASES:
            shown = [s for s in names if vis[s]]
            for s in names:
                a = self.stage_actions[STAGE[s]]
                a.setIcon(self._flow_icon(0, self._flow_cols[s]))
                a.setVisible(vis[s] and phase == PHASE_OF[cur])
            self._phase_acts[phase].setVisible(bool(shown))
            b = self._phase_buttons[phase]
            if b.isChecked() != (phase == PHASE_OF[cur]):
                b.setChecked(phase == PHASE_OF[cur])
        if self._header_phase != PHASE_OF[cur] and hasattr(self, "btn_phase"):
            self._header_phase = PHASE_OF[cur]
            self._update_header_actions(PHASE_OF[cur])
        if fit:
            self._fit_toolbars()

    def _fill_phase_menu(self, phase: str, menu):
        """Menu of a phase button: all steps of the phase with a state icon (hidden ones with a suffix), choice = switch the step."""
        menu.clear()
        cur = STAGES[getattr(self, "stage", 0)]
        for s in dict(PHASES)[phase]:
            col = self._flow_cols.get(s, GREY)
            hidden = not self.stage_visible(s)
            act = menu.addAction(self._flow_icon(0, col), s + (" (hidden)" if hidden else ""))
            act.setCheckable(True)
            act.setChecked(s == cur)
            act.setEnabled(self.stage_enabled(s))
            act.setToolTip(f"{s}: {STAGE_TIPS[s]}")
            act.triggered.connect(lambda _=False, k=STAGE[s]: self.set_stage(k))

    def _fill_more_menu(self, menu):
        """Analyses menu: optional steps (tick = pin to the bar), "Show all steps". The items are real check boxes
        (QWidgetAction) – in this menu style a checkable item without a tick would not be recognisable; the choice closes the menu.
        """
        from PySide6.QtWidgets import QCheckBox, QWidgetAction
        menu.clear()
        theme.add_menu_section(menu, "Show steps")
        self._more_checks: dict[str, QCheckBox] = {}

        def add_check(key, text, on, enabled, tip, slot):
            cb = QCheckBox(text)
            cb.setStyleSheet("QCheckBox { padding: 5px 14px 5px 12px; }")
            cb.setChecked(on)
            cb.setEnabled(enabled)
            cb.setToolTip(tip)
            cb.toggled.connect(lambda v, f=slot: (menu.close(), f(v)))
            act = QWidgetAction(menu)
            act.setDefaultWidget(cb)
            menu.addAction(act)
            self._more_checks[key] = cb

        for s in OPTIONAL_STAGES:
            used = self._stage_used(s)
            add_check(s, ANALYSIS_LABELS[s] + (" – used in the study" if used else ""),
                      used or s in self._flow_pins or self._flow_show_all or s in self._flow_revealed,
                      not used and self.stage_enabled(s), f"{s}: {STAGE_TIPS[s]}",
                      lambda on, nm=s: self._pin_stage(nm, on))
        menu.addSeparator()
        add_check("all", "Show all steps", self._flow_show_all, True,
                  "Also show unused optional steps on the bar", self._set_show_all)

    def _save_flow_prefs(self):
        if _remember_ui():
            self.settings.setValue("flow/pinned", json.dumps(sorted(self._flow_pins)))
            self.settings.setValue("flow/show_all", "1" if self._flow_show_all else "0")

    def _pin_stage(self, name: str, on: bool):
        if on:
            self._flow_pins.add(name)
            self._save_flow_prefs()
            self.set_stage(STAGE[name])
        else:
            self._flow_pins.discard(name)
            self._flow_revealed.discard(name)
            self._save_flow_prefs()
            self._apply_flow_visibility()

    def _set_show_all(self, on: bool):
        self._flow_show_all = bool(on)
        self._save_flow_prefs()
        self._apply_flow_visibility()

    def step_stage(self, d: int):
        """Ctrl+PgDown / Ctrl+PgUp: next / previous visible (and enabled) step."""
        k = getattr(self, "stage", 0)
        while True:
            k += d
            if not 0 <= k < len(STAGES):
                return
            if self.stage_visible(STAGES[k]) and self.stage_enabled(STAGES[k]):
                self.set_stage(k)
                return

    # ------------------------------------------------------------------ bars by window width
    def _fit_toolbars(self):
        """Header in a narrower window: first without the document name, then without the Replace geometry button (it is in the File menu),
        without the program name and finally Run calculation as an icon only – the phase tabs and menus always stay.
        """
        if getattr(self, "_fitting", False) or not hasattr(self, "tb_right"):
            return
        from PySide6.QtWidgets import QToolBar, QToolButton
        self._fitting = True
        try:
            width = self.width() - 12
            row = [self.tb_file, self.tb_flow, self.tb_right]
            lvl = 0
            for lvl in range(5):
                self._header_level(lvl)
                if sum(t.sizeHint().width() for t in row) <= width:
                    break
            self._fit_levels = (lvl, 0)
            for tb in self.findChildren(QToolBar):          # the » dropdown button stays visible in the dark theme
                ext = tb.findChild(QToolButton, "qt_toolbar_ext_button")
                if ext is not None and ext.property("feaIcon") != "more":
                    theme.set_icon(ext, "more")                 # vertical dots – the button is narrow (~12 px)
                    ext.setIconSize(QSize(12, 12))
                    ext.setToolTip("More toolbar items that did not fit into the narrow window")
        finally:
            self._fitting = False

    def _header_level(self, lvl: int):
        self._header_lvl = lvl
        self._doc_act.setVisible(lvl < 1)
        n = len(getattr(self, "_header_acts", []))
        self._replace_act.setVisible(lvl < 2 and n > 0)
        self._phase2_act.setVisible(lvl < 2 and n > 1)
        self._brand_act.setVisible(lvl < 3)
        self.btn_run_header.setText("" if lvl >= 4 else tr("Run analysis"))

    @staticmethod
    def _set_qss(w, css: str):
        """Style of one widget; not recomputed when unchanged (polishing costs time at every compaction of the bar)."""
        if w.styleSheet() != css:
            w.setStyleSheet(css)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_toolbars()
        QTimer.singleShot(0, self._fit_shelf)       # the height of the shelf is known only after the window layout

    def _tools_button(self):
        """Tools menu: beam calculator, new frame system, mesh convergence."""
        from PySide6.QtWidgets import QMenu, QToolButton
        b = QToolButton()
        b.setText("Tools")
        b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        theme.set_icon(b, "tools")                 # in a narrow window only the icon (_fit_toolbars)
        b.setPopupMode(QToolButton.InstantPopup)
        help_tip(b, "Calculators, new beam structure, mesh convergence, modules",
                 "Tools: analytic beam calculator (beam, curved bar, buckling, EN 1993 check), friction, Hertz, press-fit, snap-fit, mechanism and damping calculators, new frame structure without CAD, mesh convergence")
        m = QMenu(b)
        m.addAction("Beam calculator…", self.open_beam_calc)
        m.addAction("Calculators (friction, Hertz, press fit, snap fit, heat)…", self.open_calc_dialog)
        m.addAction("Selection by criteria (Worksheet)…", self.open_criteria_selection_dialog)
        m.addAction("Section force (Free Body Cut)…", self.open_section_force_dialog)
        m.addAction("New frame (no CAD)", self.new_frame_study)
        m.addAction("Mesh convergence…", self.open_convergence_dialog)
        m.addSeparator()
        m.addAction("Replace geometry (keep setup)…", self.replace_geometry_dialog)
        m.addAction("Geometry iterations…", self.open_iterations)
        m.addSeparator()
        m.addAction("Program modules…", self.open_modules_dialog)
        b.setMenu(m)
        self.tools_button = b
        return b

    # ------------------------------------------------------------------ new geometry keeping the settings
    def replace_geometry_dialog(self):
        if self.study is None or self.step_path is None:
            msg(self, APP_NAME, "Open a model or a project with STEP geometry first.")
            return
        last = self.settings.value("last_dir", "")
        p, _ = QFileDialog.getOpenFileName(self, "Replace geometry – modified model", last,
                                           f"STEP (*.step *.stp *.STEP *.STP);;{NATIVE_FILTER}")
        if p:
            self.update_geometry(Path(p))

    def update_geometry(self, path: Path, trigger: str = "step", solve: bool = False) -> dict:
        """New STEP (modified model from the CAD) keeping the settings: proposed remapping (fea_core.cad_link) →
        window to check (if there is something to check) → new study, iteration; solve = calculate after a certain remapping.
        """
        from fea_core import cad_import
        from fea_core.cad_link.update import apply_report, plan_update
        if self.busy():
            msg(self, APP_NAME, "Meshing or analysis in progress – wait for it to finish.")
            return {"applied": False, "reason": "busy"}
        if self.study is None or self.step_path is None:
            return {"applied": False, "reason": "no study"}
        path = Path(path)
        warn: list[str] = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if is_native(path):                              # modified Inventor / Fusion part without a STEP export
                path = native_to_step(path, warn)
            new_parts = read_step(path, report=warn)
            if not new_parts:
                raise RuntimeError("The STEP contains no solid")
            old_meta = cad_import.load_sidecar(self.step_path)
            new_meta = cad_import.load_sidecar(path, warn)
            rep = plan_update(self.study, self.solid_parts(), new_parts, old_meta, new_meta)
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            msg(self, APP_NAME, f"The new geometry cannot be loaded:\n{e}", QMessageBox.Critical)
            return {"applied": False, "reason": str(e)}
        QApplication.restoreOverrideCursor()
        reject, reject_anchors = set(), set()
        if rep.needs_review():
            from PySide6.QtWidgets import QDialog
            from .remap_dialog import RemapDialog
            dlg = RemapDialog(self, rep, f"{path.name}: {len(new_parts)} parts")
            if dlg.exec() != QDialog.Accepted:
                self.log_msg("Geometry update cancelled – the current model is kept.")
                return {"applied": False, "reason": "cancelled"}
            reject, reject_anchors = dlg.rejected_selections(), dlg.rejected_anchors()
        new = apply_report(self.study, rep, new_step=str(path), new_parts=new_parts, new_meta=new_meta,
                           trigger=trigger, results=self.results, old_parts=self.solid_parts(),
                           reject=reject, reject_anchors=reject_anchors)
        self.motion_result = self._motion_sig = None
        self._load_model(path, new, None, parts=new_parts, geo_warn=warn)
        self.dirty = True
        self.reset_undo()                         # undo/redo boundary: snapshots do not contain the geometry
        self.update_title()
        it = new.cad_link.iterations[-1]
        self.log_msg(f"Geometry updated ({path.name}) – iteration {it.n}; selections: "
                     + ", ".join(f"{k} {v}" for k, v in it.remap.items()))
        items = [("info", f"Iterace {it.n}: " + ("; ".join(it.changes) or "geometry unchanged"))]
        items += [("warning", n) for n in rep.notes]
        blocking = rep.blocking() + [s for s in reject if s not in rep.blocking()]
        if blocking:
            items.append(("warning", "Selections without remapping (fix the faces): " + ", ".join(blocking)))
        self.messages.set_source("geometry update", items, "Model")
        if solve and not blocking and not rep.notes:
            QTimer.singleShot(0, self.run_solve)
        return {"applied": True, "iteration": it.n, "remap": it.remap, "not_remapped": blocking}

    def open_iterations(self):
        if self.study is None or self.study.cad_link is None or not self.study.cad_link.iterations:
            msg(self, APP_NAME, "The study has no geometry iterations yet – they are created by the first geometry update (Tools → Replace geometry, CAD add-in \"Update in FEA Studio\").")
            return
        from .remap_dialog import IterationsDialog
        IterationsDialog(self, self.study).exec()

    def open_modules_dialog(self):
        from .modules_dialog import ModulesDialog
        if getattr(self, "_modules_dlg", None) is None:
            self._modules_dlg = ModulesDialog(self)
        self._modules_dlg.show()
        self._modules_dlg.raise_()

    def open_calc_dialog(self):
        from .calc_dialog import CalcDialog
        if getattr(self, "_calcs_dlg", None) is None:
            self._calcs_dlg = CalcDialog(self)
        self._calcs_dlg.show()
        self._calcs_dlg.raise_()

    def open_beam_calc(self):
        from .beam_calc import BeamCalcDialog
        if getattr(self, "_calc_dlg", None) is None:
            self._calc_dlg = BeamCalcDialog(self)
        self._calc_dlg.show()
        self._calc_dlg.raise_()

    def _theme_button(self):
        """Appearance switch (by system / light / dark), stored in QSettings."""
        from PySide6.QtWidgets import QMenu, QToolButton
        b = QToolButton()
        b.setText("Appearance")
        b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b.setPopupMode(QToolButton.InstantPopup)
        b.setToolTip("Application appearance: follow Windows, light or dark")
        m = QMenu(b)
        grp = QActionGroup(m)
        cur = str(self.settings.value("theme", "system"))
        for key, txt in theme.MODES.items():
            a = m.addAction(txt, lambda k=key: self.set_theme(k))
            a.setCheckable(True)
            a.setChecked(key == cur)
            grp.addAction(a)
        b.setMenu(m)
        theme.set_icon(b, "theme")
        return b

    def set_theme(self, mode: str):
        """Switches the appearance at run time: palette + style, icons, 3D window background, colours of the step states."""
        self.settings.setValue("theme", mode)
        self.dark = theme.apply(QApplication.instance(), mode)
        theme.refresh_icons(self)
        self.view.set_dark_background(self.dark)
        self.refresh_flow()
        self._flow_sig = None
        self._fit_toolbars()                      # the style changes the widths of the bar buttons (polish once more afterwards)
        QTimer.singleShot(0, self._fit_toolbars)

    def _build_tree(self):
        self.tree = QTreeWidget()
        self.tree.setMinimumHeight(110)               # above the step panel (Studio layout)
        self.tree.setHeaderLabels(["Study", "Info"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setStretchLastSection(False)
        self.tree.itemClicked.connect(self.on_tree_click)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_menu)
        help_tip(self.tree, "Study tree – a click switches to the step, the dot = item state",
                 "Study tree: parts, supports, load cases, mesh and results; the dot = item state (green done, orange warnings, red error). A click switches to the corresponding step.<br>Right-click a part to exclude the body from the analysis.")

    def _build_docks(self):
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setFont(theme.mono_font())
        self.log.setWhatsThis("<b>Solver log</b><p>Progress of meshing and analysis (Gmsh and CalculiX output), warnings and errors. If the analysis fails, the reason is here – most often a missing support (free body) or material. An overview of errors and warnings is in the “Messages” dock.</p>")
        from .table_dock import PhaseTables              # table by step (parts, loads, mesh, results)
        self.table_dock = PhaseTables(self)
        dt = QDockWidget("Tables", self)
        dt.setObjectName("dock_tables")
        dt.setWidget(self.table_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, dt)
        self.dock_tables = dt
        d = QDockWidget("Solver log", self)
        d.setObjectName("dock_log")
        d.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, d)
        self.tabifyDockWidget(dt, d)
        self.resizeDocks([d], [130], Qt.Vertical)
        dj = QDockWidget("Joints – utilization", self)          # tables of bolts and welds (colour by utilisation)
        dj.setObjectName("dock_joints")
        dj.setWidget(self.joints_panel.tabs)
        self.addDockWidget(Qt.BottomDockWidgetArea, dj)
        self.tabifyDockWidget(d, dj)
        dj.hide()                                         # tables of bolts and welds are in the tabs of the Tables dock
        self.messages = MessagesPanel(self)               # message table (check, mesh, solver, results)
        dm = QDockWidget("Messages", self)
        dm.setObjectName("dock_messages")
        dm.setWidget(self.messages)
        self.messages.dock = dm
        self.addDockWidget(Qt.BottomDockWidgetArea, dm)
        self.tabifyDockWidget(d, dm)
        self.resizeDocks([d, dj, dm], [170, 170, 170], Qt.Vertical)
        d.raise_()
        self.dock_log, self.dock_joints, self.dock_messages = d, dj, dm
        dt.topLevelChanged.connect(self._on_tables_floating)
        from .popout import CloseToDock
        self._tables_close = CloseToDock(dt)              # closing the floating Tables window docks it back
        for dk in (dt, d, dj, dm):                      # bottom docks: the title is the dock tab (proposal – no title bar)
            if self.dockWidgetArea(dk) == Qt.BottomDockWidgetArea:
                dk.setTitleBarWidget(QWidget())

    def _build_panels(self):
        self.joints_panel = JointsPanel(self)
        self.thermal_panel = ThermalPanel(self)
        panels = {stage: QWidget() for stage, _ in LAZY_PANELS.values()}     # placeholders of the lazily built panels
        for name, b in (("Model", self._panel_model), ("Materials", self._panel_materials),
                        ("Joints", self.joints_panel),
                        ("Constraints", self._panel_constraints), ("Heat", self.thermal_panel),
                        ("Loads", self._panel_loads), ("Mesh", self._panel_mesh), ("Solve", self._panel_solve),
                        ("Results", self._panel_results)):
            if isinstance(b, QWidget):
                panels[name] = b
                continue
            w = QWidget()
            _, lay = scroll_body(w)           # scroll bar – the panel is below the tree
            b(lay)
            lay.addStretch(1)
            panels[name] = w
        self.fat_panel = FatiguePanel(self)
        panels["Fatigue"] = self.fat_panel
        for name in STAGES:                 # panel order = step order (index only via STAGE)
            self.panel_stack.addWidget(panels[name])
        assert self.panel_stack.count() == len(STAGES)

    # attributes of the lazily built panels (3D printing, Members, Contacts – F8, Dynamics – F8, Motion – F8)
    print_panel = _LazyPanel("print_panel")
    frame_panel = _LazyPanel("frame_panel")
    contact_panel = _LazyPanel("contact_panel")
    dynamics_panel = _LazyPanel("dynamics_panel")
    motion_panel = _LazyPanel("motion_panel")

    def _build_lazy_panel(self, attr: str):
        """Builds the panel of an optional step in place of its placeholder (same index in the stack, same post-processing as at start)."""
        if attr in self.__dict__:
            return self.__dict__[attr]
        stage, cls = LAZY_PANELS[attr]
        panel = cls(self)
        self.__dict__[attr] = panel
        k = STAGE[stage]
        stack = self.panel_stack
        current = stack.currentIndex()
        old = stack.widget(k)
        stack.insertWidget(k, panel)
        stack.removeWidget(old)
        old.deleteLater()
        if current == k:
            stack.setCurrentIndex(k)
        if getattr(self, "_started", False):          # built after the start: what __init__ did for the other panels
            compact(panel)
            if getattr(self, "_form_tips", None) is not None:
                self._form_tips.add(panel)
            if self.study is not None:
                panel.refresh()
        return panel

    def _built(self, attr: str):
        """The panel if it is already built, otherwise None (refreshing a panel nobody has seen is not needed – it refreshes at its build)."""
        return self.__dict__.get(attr)

    def _panel_color(self, attr: str, *args) -> str:
        """Step colour without building the panel: color() of a panel reads only the study (through win)."""
        panel = self._built(attr)
        if panel is not None:
            return panel.color(*args)
        cls = LAZY_PANELS[attr][1]
        shell_type = type("_Shell", (), {n: cls.__dict__[n] for n in _SHELL_ATTRS if n in cls.__dict__})
        shell = shell_type()
        shell.win = self
        return shell.color(*args)

    # ------------------------------------------------------------------ panel Model
    def _panel_model(self, lay):
        lay.addWidget(panel_header("Model", "STEP geometry from CAD (parts = bodies, units mm) or a .fea project",
                                   STAGE["Model"] + 1))
        b = QPushButton("Open STEP / project…")
        theme.set_primary(b)
        b.setToolTip("Load STEP geometry (assembly parts = bodies) or a saved .fea project  [Ctrl+O]")
        b.setWhatsThis("<b>Open model</b><p>STEP (.step, .stp) creates a new study: each body is a part, part names are taken from the CAD (bolts and welds are recognized from them). A .fea project contains geometry, settings and results.</p><p>Model units are mm.</p>")
        self.btn_open_model = b
        b.clicked.connect(self.open_dialog)
        lay.addWidget(b)
        b = QPushButton("New frame (no CAD)")
        b.setToolTip("New study from members only (frame, truss) without STEP geometry")
        b.setWhatsThis("<b>Frame structure without CAD</b><p>A study without a solid: frames, trusses, beams. Nodes, members, cross-sections (EN library, parametric, custom) and supports are entered in the <i>Beams</i> step, member loads (nodal force, distributed load, gravity) in the <i>Loads</i> step. Solution: the program’s own exact beam solver (Timoshenko) – statics, natural frequencies and linear buckling without CalculiX; EN 1993-1-1 check.</p><p>Member material = the default material of the study (Materials step).</p>")
        b.clicked.connect(self.new_frame_study)
        self.btn_new_frame = b
        lay.addWidget(b)

        self.model_table = table(["Show", "Analyze", "Part", "State", "Volume [cm³]", "Material"], stretch_last=True,
                                 placeholder="Open a STEP file or a project…", rows=(3, 12))
        self.model_table.itemChanged.connect(self._on_model_table_check)
        self.model_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.model_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_table.customContextMenuRequested.connect(self._on_model_table_menu)
        self.model_table.itemSelectionChanged.connect(self._on_model_table_selection)
        self.model_table.itemDoubleClicked.connect(lambda _: self.view.zoom_selection())
        self.model_table.setToolTip("Model bodies: Show = in 3D, Analyze = in the analysis; right-click = in bulk")
        self.model_table.setWhatsThis(
            "<b>Model bodies</b><p><i>Show</i> – the body is visible in the 3D view (display only, no effect on the analysis; a hidden body cannot be picked by clicking).</p><p><i>Analyze</i> – the body is in the analysis. A switched-off body is neither meshed nor analyzed (magnets, pins, purchased parts); in 3D it is transparent. The change discards the mesh; selections, supports and loads of the other bodies remain.</p><p>Right-click: switch the selected bodies on / off in bulk, isolate, switch off by name (e.g. <tt>*bolt*</tt>).</p>")
        lay.addWidget(self.model_table)

        row = QHBoxLayout()
        self.btn_toggle_ignore_part = QPushButton("Exclude / include in analysis")
        self.btn_toggle_ignore_part.setToolTip("Excludes the selected bodies from the analysis and meshing, or brings them back")
        self.btn_toggle_ignore_part.clicked.connect(self._toggle_selected_model_parts_ignore)
        row.addWidget(self.btn_toggle_ignore_part)
        self.btn_toggle_hide_part = QPushButton("Hide / show")
        self.btn_toggle_hide_part.setToolTip("Hides the selected bodies in the 3D view or shows them again (no effect on the analysis)")
        self.btn_toggle_hide_part.clicked.connect(self._toggle_selected_model_parts_hidden)
        row.addWidget(self.btn_toggle_hide_part)
        self.btn_show_all_parts = QPushButton("Show all")
        self.btn_show_all_parts.setToolTip("Shows all hidden bodies in 3D")
        self.btn_show_all_parts.clicked.connect(lambda: self.set_hidden([p.name for p in self.parts], False))
        row.addWidget(self.btn_show_all_parts)
        lay.addLayout(row)

        self.model_info = SafeTextBrowser()
        self.model_info.setMaximumHeight(130)
        self.model_info.setToolTip("Model parts with volume and assigned material, STEP geometry check warnings")
        lay.addWidget(self.model_info)

    # ------------------------------------------------------------------ Materials panel
    @staticmethod
    def _scrolled(outer):
        """Content of the panel in a scroll area (without a horizontal scroll bar); returns the content layout."""
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidget(inner)
        outer.addWidget(sc, 100)
        return lay

    def _panel_materials(self, lay):
        lay = self._scrolled(lay)                   # the content (part list + properties + orthotropy) fits only in the scroll area
        lay.addWidget(panel_header("Materials", "Select parts (here or by clicking in 3D, Ctrl = add) and assign a material",
                                   STAGE["Materials"] + 1))
        sec = Section("Parts", key="mat.parts", tip="Select the parts that get the material (Ctrl / Shift = several, or click in 3D)")
        slay = QVBoxLayout(sec.body)
        slay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(sec)
        self.mat_parts = QListWidget()
        fit_rows(self.mat_parts, 3, 10)             # the part list grows with the number of parts, scroll bar above 10 rows
        self.mat_parts.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mat_parts.itemSelectionChanged.connect(self.on_mat_parts_sel)
        help_tip(self.mat_parts, "Parts and their material – select parts (Ctrl / Shift = several, or click in 3D)",
                 "Parts and their material. Select parts (Ctrl / Shift = several, or by clicking in 3D), then choose a material and “Assign to selected”.")
        slay.addWidget(self.mat_parts)
        sec = Section("Material", key="mat.material", tip="Material from the library, converter of designations, summary of properties")
        slay = QVBoxLayout(sec.body)
        slay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(sec)
        mat_lay, lay = lay, slay
        self.mat_combo = QComboBox()
        self.mat_combo.currentIndexChanged.connect(self.on_mat_combo)
        help_tip(self.mat_combo, "Material from the library or the study – values are loaded into the form",
                 "Material from the built-in library (with the source of the values), from your library or from the study.<br>Selecting it loads the properties into the form below – you can edit them before assigning.")
        lay.addWidget(self.mat_combo)
        self.mat_find = QLineEdit()
        self.mat_find.setPlaceholderText("Converter: ČSN / EN / W.Nr. / AISI / JIS… designation (e.g. 12050, St 52-3)")
        self.mat_find.setClearButtonEnabled(True)
        self.mat_find.textChanged.connect(self.on_mat_find)
        help_tip(self.mat_find, "Converter: type a designation in any standard (ČSN, EN, AISI…)",
                 "Designation converter: type a designation in any standard (ČSN 11 373, 1.0038, S235JR, A36, 6061…) – it selects the matching library material, otherwise it offers similar ones (links under the field).")
        lay.addWidget(self.mat_find)
        self.mat_conv = QLabel()
        self.mat_conv.setWordWrap(True)
        self.mat_conv.setVisible(False)             # an empty label would leave a gap in the layout
        self.mat_conv.linkActivated.connect(lambda url: self.mat_combo.setCurrentText(self._mat_links[int(url[4:])]))
        self._mat_links: list[str] = []
        lay.addWidget(self.mat_conv)
        # material summary from the form; full properties and orthotropy are in a pop-up window (clearer
        # than a long scrolling panel) – the fields m_* and ortho_box live in the window, the panel logic reads them unchanged
        self.mat_summary = QLabel()
        self.mat_summary.setWordWrap(True)
        self.mat_summary.setTextFormat(Qt.RichText)
        help_tip(self.mat_summary, "Material in the form – edit with “Material properties…”",
                 "Summary of the material loaded into the form (from the list or edited). Edit values, thermal properties, source and orthotropy in the “Material properties…” window.")
        lay.addWidget(self.mat_summary)
        b = QPushButton("Material properties…")
        help_tip(b, "Edit material properties (E, ν, ρ, Re, Rm, heat, orthotropy) in a window",
                 "Opens a window with the material properties: name, E, ν, ρ, Re, Rm, thermal properties, source and orthotropic constants. The window can stay open – parts can still be selected in 3D and assigned.")
        b.clicked.connect(self.open_material_dialog)
        self.btn_mat_props = b
        lay.addWidget(b)
        self._build_material_dialog()
        sec = Section("Assignment", key="mat.assign", tip="Assign the material from the form to parts, save it to your library")
        lay = QVBoxLayout(sec.body)
        lay.setContentsMargins(0, 0, 0, 0)
        mat_lay.addWidget(sec)
        h = QHBoxLayout()
        b1 = QPushButton("Assign to selected")
        b1.setToolTip("Assign the material from the form to the parts selected in the list (or by clicking in 3D)")
        b1.clicked.connect(lambda: self.assign_material(False))
        self.btn_assign_sel = b1
        theme.set_primary(b1)
        b2 = QPushButton("Assign to all")
        b2.setToolTip("Assign the material from the form to all parts of the model")
        b2.clicked.connect(lambda: self.assign_material(True))
        h.addWidget(b1)
        h.addWidget(b2)
        lay.addLayout(h)
        b3 = QPushButton("Save to my material library")
        help_tip(b3, "Save the material from the form to your own library (offered in other projects too)",
                 "Save the material from the form to your own library (it is offered in other projects too):<br>"
                 + str(matlib.user_library_path()))
        b3.clicked.connect(self.save_user_material)
        lay.addWidget(b3)
        mat_lay.addStretch(1)

    def _build_material_dialog(self):
        """Non-modal "Material properties" window: form (m_*) + orthotropy (ortho_box) + assignment."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Material properties")
        dlg.setModal(False)
        outer = QVBoxLayout(dlg)
        lay = self._scrolled(outer)
        g = Section("Mechanical properties", key="matdlg.mech", detachable=False,
                    tip="Name, elastic constants, density and strengths – editable, then assign")
        f = QFormLayout(g.body)
        g_th = Section("Thermal properties", key="matdlg.thermal", detachable=False,
                       tip="Conductivity, expansion, heat capacity – for the Heat step and thermal stress")
        f_th = QFormLayout(g_th.body)
        g_src = Section("Source", key="matdlg.source", detachable=False,
                        tip="Source of the values (standard, catalog, measurement) – in the report")
        f_src = QFormLayout(g_src.body)
        self.m_name = QLineEdit()
        self.m_E = dspin(210000, 1, 1e7, 0, suffix=" MPa")
        self.m_nu = dspin(0.3, 0.0, 0.499, 3)
        self.m_rho = dspin(7850, 0, 1e5, 0, suffix=" kg/m³")
        self.m_Re = dspin(235, 0, 1e5, 0, suffix=" MPa")
        self.m_Rm = dspin(360, 0, 1e5, 0, suffix=" MPa")
        self.m_src = QLineEdit()
        self.m_name.setToolTip("Material name – assigned to parts under this name (changing the name = new material)")
        self.m_E.setToolTip("Tensile Young's modulus (steel 210 000, aluminum 70 000, PLA ~3 000 MPa)")
        self.m_nu.setToolTip("Poisson's ratio (steel 0.3, aluminum 0.33, plastics 0.35–0.4)")
        self.m_rho.setToolTip("Density – for self-weight, rotation and transient heat conduction (steel 7850 kg/m³)")
        self.m_Re.setToolTip("Yield strength Re / Rp0.2 – safety factor to Re in results, weld assessment")
        self.m_Rm.setToolTip("Tensile strength – S-N curve estimate for fatigue, weld and thread assessment")
        self.m_src.setToolTip("Source of values (standard, catalog, measurement) – included in the report")
        # heat: 0 = not entered (None); α is shown in 10⁻⁶/K, stored in 1/K
        self.m_k = dspin(0.0, 0, 1e4, 3, suffix=" W/(m·K)")
        self.m_alpha = dspin(0.0, 0, 1e4, 3, suffix=" ·10⁻⁶/K")
        self.m_cp = dspin(0.0, 0, 1e6, 1, suffix=" J/(kg·K)")
        for w in (self.m_k, self.m_alpha, self.m_cp):
            w.setSpecialValueText("not specified")
        self.m_k.setToolTip("Thermal conductivity – required for heat conduction (Heat step)")
        self.m_alpha.setToolTip("Coefficient of thermal expansion – required for thermal stress")
        self.m_cp.setToolTip("Specific heat capacity – transient heat conduction only (with density ρ)")
        for lbl, w in (("Name", self.m_name), ("E", self.m_E), ("ν", self.m_nu), ("ρ", self.m_rho),
                       ("Re (Rp0.2)", self.m_Re), ("Rm", self.m_Rm)):
            f.addRow(lbl, w)
        for lbl, w in (("λ (conductivity)", self.m_k), ("α (expansion)", self.m_alpha), ("c_p (heat capacity)", self.m_cp)):
            f_th.addRow(lbl, w)
        f_src.addRow("Source", self.m_src)
        self._mat_print: dict = {}          # print properties (E_z, Rm_z…) of the selected material – keep
        self._mat_form_src = None           # (name, repr) of the material loaded into the form
        self._mat_select = None             # material that should stay selected after the list is refreshed (assigned)
        self.m_print = QLabel()
        self.m_print.setWordWrap(True)
        f.addRow(self.m_print)
        f.setRowVisible(self.m_print, False)        # an empty row would leave a gap (it is shown with text)
        self._mat_form = f
        lay.addWidget(g)
        lay.addWidget(g_th)
        lay.addWidget(g_src)
        from .ortho_box import OrthoBox
        self.ortho_box = OrthoBox(self)          # general orthotropic material (Material.ortho, ortho_csys)
        lay.addWidget(self.ortho_box)
        from .plasticity_box import PlasticityBox
        self.plasticity_box = PlasticityBox(self)    # elastoplastic metal (Material.plastic, fea_core.plasticity)
        lay.addWidget(self.plasticity_box)
        lay.addStretch(1)
        h = QHBoxLayout()
        b1 = QPushButton("Assign to selected")
        b1.setToolTip("Assign the material from the form to the parts selected in the list (or by clicking in 3D)")
        b1.clicked.connect(lambda: self.assign_material(False))
        theme.set_primary(b1)
        b2 = QPushButton("Assign to all")
        b2.setToolTip("Assign the material from the form to all parts of the model")
        b2.clicked.connect(lambda: self.assign_material(True))
        b3 = QPushButton("Save to library")
        b3.setToolTip("Save the material from the form to your own library (offered in other projects too)")
        b3.clicked.connect(self.save_user_material)
        h.addWidget(b1)
        h.addWidget(b2)
        h.addWidget(b3)
        h.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.close)
        h.addWidget(bb)
        outer.addLayout(h)
        dlg.resize(580, 780)
        self.mat_dialog = dlg
        for w in (self.m_E, self.m_nu, self.m_rho, self.m_Re, self.m_Rm, self.m_k, self.m_alpha, self.m_cp):
            w.valueChanged.connect(self.update_mat_summary)
        for w in (self.m_name, self.m_src):
            w.textChanged.connect(self.update_mat_summary)
        self.ortho_box.check.toggled.connect(self.update_mat_summary)
        self.plasticity_box.check.toggled.connect(self.update_mat_summary)
        self.update_mat_summary()

    def open_material_dialog(self):
        """Shows the material properties window (non-modal – selecting parts in 3D keeps working)."""
        self.mat_dialog.show()
        self.mat_dialog.raise_()
        self.mat_dialog.activateWindow()

    def update_mat_summary(self, *_):
        """Short summary of the material form in the panel (full values are in the Material properties window)."""
        if not hasattr(self, "mat_summary"):
            return
        def opt(w, unit):
            return f"{fmt(w.value())} {unit}" if w.value() > 0 else tr("not specified")
        name = html.escape(self.m_name.text().strip() or "—")
        lines = [f"<b>{name}</b>",
                 f"E {fmt(self.m_E.value())} MPa · ν {fmt(self.m_nu.value())} · ρ {fmt(self.m_rho.value())} kg/m³",
                 f"Re {opt(self.m_Re, 'MPa')} · Rm {opt(self.m_Rm, 'MPa')}"]
        if self.m_k.value() > 0 or self.m_alpha.value() > 0 or self.m_cp.value() > 0:
            lines.append(f"λ {opt(self.m_k, 'W/(m·K)')} · α {opt(self.m_alpha, '·10⁻⁶/K')} · "
                         f"c_p {opt(self.m_cp, 'J/(kg·K)')}")
        if self.ortho_box.isChecked():
            lines.append("<i>" + tr("orthotropic material") + "</i>")
        elif self._mat_print:
            lines.append("<i>" + tr("printed material – print properties in the 3D printing step") + "</i>")
        if self.plasticity_box.isChecked():
            lines.append("<i>" + tr("elastoplastic material – nonlinear analysis") + "</i>")
        self.mat_summary.setText("<br>".join(lines))

    # ------------------------------------------------------------------ Constraints panel
    def _panel_constraints(self, lay):
        lay.addWidget(panel_header("Constraints", "Click to select faces in 3D (click empty space = clear), then choose a support",
                                   STAGE["Constraints"] + 1))
        self.con_list = QListWidget()
        self.con_list.currentRowChanged.connect(self.on_con_row)
        help_tip(self.con_list, "Study supports – a click on a row highlights the faces and loads the values",
                 "Study supports. A click on a row highlights their faces and loads the values into the form (edit with the “Save changes…”, “Faces from selection” and “Delete” buttons).")
        set_placeholder(self.con_list, "No supports yet – select faces and choose a support")
        fit_rows(self.con_list, 3, 8)
        lay.addWidget(self.con_list)
        b = QPushButton("Fixed support from selection")
        b.setToolTip("Fix the selected faces (ux = uy = uz = 0)")
        b.setWhatsThis("<b>Fixed support</b><p>The simplest support: the selected faces cannot move in any direction. First select the faces by clicking in 3D (Face filter P), then press the button.</p><p>The model must be supported so that it cannot move or rotate freely – the Solve step checks the rigid body motions.</p><p>Edges of a fixed support tend to be singularities – do not evaluate the stress directly on them.</p>")
        self.btn_fixed = b
        b.clicked.connect(self.add_fixed)
        lay.addWidget(b)
        g = Section("Prescribed displacement", key="con.disp", tip="Support with selected displacement components",
                    help="Support with selected displacement components: symmetry (only the normal component = 0), sliding support, or an imposed displacement (non-zero value).")
        grid = QGridLayout(g.body)
        self.d_chk, self.d_val = [], []
        for i, ax in enumerate("xyz"):
            c = QCheckBox(f"u{ax}")
            c.setChecked(True)
            help_tip(c, f"Prescribe the displacement in the {ax} direction (unticked = free direction)",
                     f"Prescribe displacement in direction {ax} (unchecked = free direction; symmetry: normal component only = 0)")
            v = dspin(0.0, -1e4, 1e4, 4, suffix=" mm")
            v.setToolTip(f"Displacement value u{ax} [mm] (0 = support / symmetry, nonzero = imposed displacement)")
            grid.addWidget(c, i, 0)
            grid.addWidget(v, i, 1)
            self.d_chk.append(c)
            self.d_val.append(v)
        b = QPushButton("Displacement from selection (checked components)")
        b.setToolTip("New support: prescribe the checked displacement components on the selected faces (other directions free)")
        b.clicked.connect(self.add_displacement)
        grid.addWidget(b, 3, 0, 1, 2)
        b = QPushButton("Save changes to the selected displacement")
        b.setToolTip("Overwrite the components and values of the prescribed displacement selected in the supports list")
        b.clicked.connect(self.update_displacement)
        grid.addWidget(b, 4, 0, 1, 2)
        lay.addWidget(g)
        g = Section("Cylindrical support (hole / pin)", key="con.cyl", tip="Support in the cylinder axis components (radial, tangential, axial)",
                    help="Support in the components of the cylinder axes (like Cylindrical Support): a ticked component has zero displacement.<br>The axis is determined from the selected cylindrical faces (coaxial, same radius – see the “Hole” H selection filter).")
        grid = QGridLayout(g.body)
        self.cyl_chk = []
        for i, (txt, tip) in enumerate((("radially", "Prevent radial displacement of the face (normal to the cylinder axis)"),
                                        ("tangentially (rotation)", "Prevent rotation about the cylinder axis"),
                                        ("axially", "Prevent displacement along the cylinder axis"))):
            c = QCheckBox(txt)
            c.setToolTip(tip)
            c.setChecked(True)                   # default as in the core (CylindricalSupport): all three components
            grid.addWidget(c, 0, i)
            self.cyl_chk.append(c)
        b = QPushButton("Cylindrical support from selection")
        b.setToolTip("New cylindrical support on the selected cylindrical faces (hole for a pin / bolt, pin in a bearing)")
        b.clicked.connect(self.add_cylindrical)
        grid.addWidget(b, 1, 0, 1, 3)
        b = QPushButton("Save changes to the selected cylindrical support")
        b.setToolTip("Overwrite the constrained components of the cylindrical support selected in the list")
        b.clicked.connect(self.update_cylindrical)
        grid.addWidget(b, 2, 0, 1, 3)
        lay.addWidget(g)
        self._panel_remote_support(lay)
        self._panel_elastic(lay)
        from .cyclic_gui import CyclicBox             # cyclic symmetry of a sector (fea_core.cyclic)
        self.cyclic_box = CyclicBox(self)
        lay.addWidget(self.cyclic_box)
        h = QHBoxLayout()
        b = QPushButton("Faces from selection")
        b.setToolTip("Replace the faces of the selected support with the current selection")
        b.clicked.connect(lambda: self.update_faces_of(self.con_list, "constraint"))
        h.addWidget(b)
        b = QPushButton("Delete")
        b.setToolTip("Delete the support selected in the list (Ctrl+Z to undo)")
        b.clicked.connect(lambda: self.delete_item(self.con_list, "constraint"))
        h.addWidget(b)
        lay.addLayout(h)

    def _panel_remote_support(self, lay):
        """Remote constraint (like Remote Displacement): point, constrained components ux…rz, rigid / distributing coupling of a face."""
        g = Section("Remote support (point off the face)", key="con.remote", collapsed=True,
                    tip="Faces attached to a remote point (displacements and rotations at the point)",
                    help="Faces attached to a remote point; the ticked displacements (ux, uy, uz) and rotations about the point (rx, ry, rz) are zero at the point. Displacements only = hinge.<br>Rigid: the face does not deform (it must not share nodes with a fixed support / displacement). Distributing: holds only the weighted average of the face, the face deforms (dense equations – suitable for smaller faces). Reaction = force and moment about the point.")
        self.rs_group = g                    # collapsible section (the Constraints panel has no scroll bar – the list of constraints would shrink)
        grid = QGridLayout(g.body)
        grid.setContentsMargins(0, 0, 0, 0)
        self.rs_pt = [dspin(0.0, -1e7, 1e7, 3) for _ in range(3)]
        hp = QHBoxLayout()
        for s, ax in zip(self.rs_pt, "xyz"):
            s.setToolTip(f"{ax} coordinate of the remote point [mm] (circle center: Edge filter → “Use center”)")
            hp.addWidget(s)
        grid.addWidget(QLabel("Point [mm]"), 0, 0)
        grid.addLayout(hp, 0, 1, 1, 3)
        self.register_point_input(self.rs_pt)
        b = QPushButton("Selection centroid")
        b.setToolTip("Point = centroid of the selected faces (weighted by area)")
        b.clicked.connect(self.remote_point_from_selection)
        grid.addWidget(b, 1, 0, 1, 2)
        self.rs_coupling = QComboBox()
        self.rs_coupling.addItems(["rigid", "distributing (RBE3)"])
        help_tip(self.rs_coupling, "Rigid = face acts as a rigid body, distributing (RBE3) = face deforms",
                 "Rigid: the face moves as a rigid body (adds stiffness).<br>Distributing (RBE3): holds only the weighted average of the displacements, the face may deform.")
        grid.addWidget(self.rs_coupling, 1, 2, 1, 2)
        self.rs_chk = []
        hc = QHBoxLayout()
        for i, txt in enumerate(("ux", "uy", "uz", "rx", "ry", "rz")):
            c = QCheckBox(txt)
            c.setChecked(True)
            c.setToolTip(("point displacement in direction " if i < 3 else "rotation about point axis ") + "xyz"[i % 3])
            hc.addWidget(c)
            self.rs_chk.append(c)
        grid.addLayout(hc, 2, 0, 1, 4)
        b = QPushButton("From selection")
        b.setToolTip("New remote support on the selected faces")
        b.clicked.connect(self.add_remote_support)
        grid.addWidget(b, 3, 0, 1, 2)
        b = QPushButton("Save changes")
        b.setToolTip("Save the point, components and type of the selected remote support")
        b.clicked.connect(self.update_remote_support)
        grid.addWidget(b, 3, 2, 1, 2)
        lay.addWidget(g)

    def _panel_elastic(self, lay):
        """Elastic support of a face (Winkler foundation, like Elastic Support): stiffness per area or total."""
        g = Section("Elastic support (foundation, springs)", key="con.elastic", collapsed=True,
                    tip="Faces on an elastic foundation (Winkler foundation)",
                    help="Selected faces resting on an elastic foundation (Winkler foundation, like Elastic Support in Ansys): stiffness k [N/mm³] per unit area, or total stiffness K [N/mm] distributed by area.<br>Normal to the face (foundation, rubber pad) or in all directions. Linear – it also transmits tension.")
        self.el_group = g                    # collapsible section as the remote constraint
        f = QFormLayout(g.body)
        f.setContentsMargins(0, 0, 0, 0)
        h = QHBoxLayout()
        self.el_k = dspin(1.0, 1e-9, 1e12, 6, 0.1)
        help_tip(self.el_k, "Support stiffness: per area k [N/mm³], or total K [N/mm]",
                 "Support stiffness: per area k [N/mm³] (pressure [MPa] per 1 mm of compression; concrete ~0.05–0.2, rubber 5 mm ~1–10), or total stiffness K [N/mm] (distributed by face area)")
        h.addWidget(self.el_k, 1)
        self.el_unit = QComboBox()
        self.el_unit.addItem("N/mm³ (per area)", True)
        self.el_unit.addItem("N/mm (total)", False)
        help_tip(self.el_unit, "Stiffness per area [N/mm³], or total [N/mm]",
                 "Stiffness per unit area k [N/mm³] (foundation), or total stiffness K [N/mm] of all selected faces together (e.g. the catalog stiffness of a rubber bushing)")
        h.addWidget(self.el_unit)
        f.addRow("Stiffness", h)
        self.el_dir = QComboBox()
        self.el_dir.addItem("normal to face", "normal")
        self.el_dir.addItem("in all directions (x, y, z)", "all")
        help_tip(self.el_dir, "Springs normal to the face only (foundation), or in all directions",
                 "Normal to the face: springs only in the normal direction (foundation – the face may slide;<br>for a support without sliding add a support against displacement and rotation in the plane).<br>In all directions: the same stiffness in x, y, z (rubber insert, bonded joint).")
        f.addRow("Direction", self.el_dir)
        h = QHBoxLayout()
        b = QPushButton("From selection")
        b.setToolTip("New elastic support on the faces selected in 3D")
        b.setWhatsThis("<b>Elastic support</b><p>Select faces (e.g. the bottom face of a plate on a foundation), enter the stiffness and direction and press <i>From selection</i>. The support reaction is in the results under the selection name.</p><p>A support only normal to the face does not prevent motion in the plane of the face – the model must also be fixed otherwise (rigid-body motion check in the Solve step).</p>")
        b.clicked.connect(self.add_elastic)
        h.addWidget(b)
        b = QPushButton("Save changes")
        b.setToolTip("Save the stiffness and direction of the elastic support selected in the supports list")
        b.clicked.connect(self.update_elastic)
        h.addWidget(b)
        f.addRow(h)
        lay.addWidget(g)

    def elastic_form(self) -> tuple[float, str, bool]:
        return float(self.el_k.value()), self.el_dir.currentData(), bool(self.el_unit.currentData())

    def add_elastic(self):
        sels = self.need_faces()
        if not sels:
            return
        k, d, per_area = self.elastic_form()
        name = self.new_selection("Elastic support", sels)
        self.study.constraints.append(ElasticSupport(name, k, d, per_area))
        self.view.set_selected_faces([])
        self.changed()

    def update_elastic(self):
        row = self.con_list.currentRow()
        if not (0 <= row < len(self.study.constraints)) or not isinstance(self.study.constraints[row], ElasticSupport):
            msg(self, APP_NAME, "Select an elastic support in the list.")
            return
        c = self.study.constraints[row]
        c.stiffness, c.direction, c.per_area = self.elastic_form()
        self.changed()
        self.con_list.setCurrentRow(row)

    # ------------------------------------------------------------------ Loads panel
    def _panel_loads(self, lay):
        lay.addWidget(panel_header("Loads", "Load case = one analysis step; loads act on the selected faces",
                                   STAGE["Loads"] + 1))
        h = QHBoxLayout()
        self.lc_combo = QComboBox()
        self.lc_combo.currentIndexChanged.connect(lambda _: self.refresh_loads())
        help_tip(self.lc_combo, "Load case the loads are added to (each = one analysis)",
                 "The load case the loads are added to. Each case = a separate analysis (separate results); fatigue combines the cases according to the load history.")
        h.addWidget(self.lc_combo, 1)
        for txt, slot, tip in (("+", self.add_lc, "New load case"),
                               ("Rename", self.rename_lc, "Rename the selected load case"),
                               ("−", self.delete_lc, "Delete the selected load case with its loads")):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        lay.addLayout(h)
        h_tools = QHBoxLayout()
        b_comb = QPushButton("Combinations (EN 1990)...")
        b_comb.setToolTip("Load case combinations and envelopes per ČSN EN 1990 (A2)")
        b_comb.clicked.connect(self.open_combinations_dialog)
        h_tools.addWidget(b_comb)
        b_imp = QPushButton("Imprint on face...")
        b_imp.setToolTip("Split a face for a load / support / contact by a geometric imprint (A10)")
        b_imp.clicked.connect(self.open_imprint_dialog)
        h_tools.addWidget(b_imp)
        lay.addLayout(h_tools)
        self.load_list = QListWidget()
        self.load_list.currentRowChanged.connect(self.on_load_row)
        help_tip(self.load_list, "Loads of the selected case – a click highlights the faces and loads the values",
                 "Loads of the selected case. A click on a row highlights the faces and loads the values into the form (“Save changes of the selected” overwrites them).")
        set_placeholder(self.load_list, "No loads yet – select faces and fill in the form")
        fit_rows(self.load_list, 3, 8)
        lay.addWidget(self.load_list)
        g = Section("New load", key="ld.new", tip="Type, values and target of the load (faces / edges / points)",
                    help="Load form: type, values and target (selected faces / edges / points).<br>“Add from selection” adds it to the load case chosen above.")
        f = QFormLayout(g.body)
        self.ld_type = QComboBox()
        self.ld_type.addItems([FORCE, "Force normal to face", "Pressure", "Moment (remote point)",
                               "Force at remote point", "Bearing load (hole / pin)", "Gravity", ROTATION,
                               THERMAL_LOAD, FRAME_NODE, FRAME_MEMBER])
        self.ld_type.currentIndexChanged.connect(self.on_ld_type)
        f.addRow("Type", self.ld_type)
        self.ld_target = QLabel("")
        self.ld_target.setWordWrap(True)
        self.ld_target.setToolTip("What the force acts on, based on the current 3D selection and selection filter (Face / Edge / Point)")
        self.ld_target_lbl = QLabel("Acts on")
        f.addRow(self.ld_target_lbl, self.ld_target)
        # frame loads (fea_core.beams.model.FrameNodeLoad / FrameMemberLoad)
        self.ld_fr_node = QComboBox()
        self.ld_fr_node.setToolTip("Frame node where the force and moment act (or click a node in 3D)")
        self.ld_fr_node_lbl = QLabel("Node")
        f.addRow(self.ld_fr_node_lbl, self.ld_fr_node)
        self.ld_fr_member = QComboBox()
        self.ld_fr_member.setToolTip("Member with distributed load (or click a member in 3D)")
        self.ld_fr_member_lbl = QLabel("Member")
        f.addRow(self.ld_fr_member_lbl, self.ld_fr_member)
        self.ld_fr_axes = QComboBox()
        self.ld_fr_axes.addItem("global axes (x, y, z)", "global")
        self.ld_fr_axes.addItem("member local axes (x along, y, z of cross-section)", "local")
        help_tip(self.ld_fr_axes, "Distributed load components in global axes, or in the member's local axes",
                 "Distributed load components in the global axes (gravity, snow, wind) or in the member's local axes (x = along the axis, z = the “up” axis of the section)")
        self.ld_fr_axes_lbl = QLabel("Axes")
        f.addRow(self.ld_fr_axes_lbl, self.ld_fr_axes)
        # local coordinate system of the components (force, remote load, bearing force, gravity)
        hc = QHBoxLayout()
        self.ld_csys = QComboBox()
        help_tip(self.ld_csys, "Component coordinate system: global, or local (Cartesian / cylindrical)",
                 "Component coordinate system: the global model axes, or a local system (Cartesian / cylindrical).<br>Cylindrical (r, θ, z) for forces only – the direction changes over the face / edge, the entered force is the sum of the magnitudes (the tangential component around the shaft end face = torque ⅔·F_θ·R).")
        self.ld_csys.currentIndexChanged.connect(lambda _: self.on_ld_type(0))
        hc.addWidget(self.ld_csys, 1)
        b = QPushButton("Systems…")
        b.setToolTip("Create / edit local coordinate systems")
        b.clicked.connect(self.edit_csys)
        hc.addWidget(b)
        self.ld_csys_btn = b
        self.ld_csys_lbl = QLabel("Coord. system")
        f.addRow(self.ld_csys_lbl, hc)
        self.ld_vec = [dspin(0.0, -1e9, 1e9, 3) for _ in range(3)]
        hv = QHBoxLayout()
        for s in self.ld_vec:
            hv.addWidget(s)
        self.ld_vec_lbl = QLabel("F [N]")
        f.addRow(self.ld_vec_lbl, hv)
        self.ld_val = dspin(0.0, -1e9, 1e9, 3)
        self.ld_val_lbl = QLabel("Value")
        f.addRow(self.ld_val_lbl, self.ld_val)
        hm = QHBoxLayout()
        self.ld_fr_mom = [dspin(0.0, -1e12, 1e12, 3) for _ in range(3)]
        for sp_, ax in zip(self.ld_fr_mom, "xyz"):
            sp_.setToolTip(f"{ax} component of the nodal moment [N·mm] (right-handed about the global axis)")
            hm.addWidget(sp_)
        self.ld_fr_mom_lbl = QLabel("M [N·mm]")
        f.addRow(self.ld_fr_mom_lbl, hm)
        hq = QHBoxLayout()
        self.ld_fr_var = QCheckBox("linearly to")
        self.ld_fr_var.setToolTip("Linearly varying load: q at the member start (fields above) → q at the end (these fields)")
        hq.addWidget(self.ld_fr_var)
        self.ld_fr_qend = [dspin(0.0, -1e9, 1e9, 4) for _ in range(3)]
        for sp_, ax in zip(self.ld_fr_qend, "xyz"):
            sp_.setToolTip(f"{ax} component of the distributed load at the member end [N/mm]")
            hq.addWidget(sp_)
        self.ld_fr_qend_lbl = QLabel("q at the end")
        f.addRow(self.ld_fr_qend_lbl, hq)
        # variable pressure: linear p = p0 + g·(x − origin), hydrostatic p = p0 + ρ·g·depth below the level
        self.ld_p_mode = QComboBox()
        self.ld_p_mode.addItems(P_MODES)
        help_tip(self.ld_p_mode, "Pressure distribution: constant, linear or hydrostatic",
                 "Constant pressure, linearly varying p = p₀ + gradient·(x − origin), or hydrostatic liquid pressure p = p₀ + ρ·g·depth below the surface (p₀ above the surface).")
        self.ld_p_mode.currentIndexChanged.connect(lambda _: self.on_ld_type(0))
        self.ld_p_mode_lbl = QLabel("Pressure profile")
        f.addRow(self.ld_p_mode_lbl, self.ld_p_mode)

        def vec_row(label, vals, dec=6, rng=1e7):
            sp = [dspin(v, -rng, rng, dec) for v in vals]
            h = QHBoxLayout()
            for x in sp:
                h.addWidget(x)
            lbl = QLabel(label)
            f.addRow(lbl, h)
            return sp, lbl
        self.ld_p_grad, self.ld_p_grad_lbl = vec_row("Gradient [MPa/mm]", (0, 0, 0), 8, 1e6)
        self.ld_p_org, self.ld_p_org_lbl = vec_row("Origin [mm]", (0, 0, 0), 3)
        for s, ax in zip(self.ld_p_grad, "xyz"):
            s.setToolTip(f"Pressure change per 1 mm in direction {ax} [MPa/mm]: p = p₀ + gradient·(x − origin)")
        for s, ax in zip(self.ld_p_org, "xyz"):
            s.setToolTip(f"{ax} coordinate [mm] of the point where pressure p₀ applies (for hydrostatic pressure, a point on the surface)")
        self.register_point_input(self.ld_p_org)
        self.ld_p_rho = dspin(1000.0, 0.0, 1e6, 2, suffix=" kg/m³")
        self.ld_p_rho.setToolTip("Liquid density (water 1000, oil 850–900 kg/m³); g = 9.81 m/s²")
        self.ld_p_rho_lbl = QLabel("Liquid ρ")
        f.addRow(self.ld_p_rho_lbl, self.ld_p_rho)
        self.ld_p_gdir, self.ld_p_gdir_lbl = vec_row("Gravity direction", (0, 0, -1), 4, 1e3)
        # rotation: axis, speed, angular acceleration, rotating parts
        self.ld_rot_pt, self.ld_rot_pt_lbl = vec_row("Axis point", (0, 0, 0), 3)
        self.register_point_input(self.ld_rot_pt)
        self.ld_rot_ax, self.ld_rot_ax_lbl = vec_row("Axis direction", (0, 0, 1), 4, 1e3)
        for s, ax in zip(self.ld_p_gdir, "xyz"):
            s.setToolTip(f"{ax} component of the gravity direction (downward; length is irrelevant) – pressure increases with depth in this direction")
        for s, ax in zip(self.ld_rot_pt, "xyz"):
            s.setToolTip(f"{ax} coordinate [mm] of any point on the rotation axis")
        for s, ax in zip(self.ld_rot_ax, "xyz"):
            s.setToolTip(f"{ax} component of the rotation axis direction (length is irrelevant; positive rotation is right-handed about it)")
        hr = QHBoxLayout()
        self.ld_rot_rpm = dspin(0.0, -1e7, 1e7, 2, suffix=" ot/min")
        self.ld_rot_rpm.setToolTip("Speed [rpm] – centrifugal force ρ·ω²·r (ω = 2π·n/60)")
        self.ld_rot_alpha = dspin(0.0, -1e9, 1e9, 3, suffix=" rad/s²")
        self.ld_rot_alpha.setToolTip("Angular acceleration (positive = accelerating about the axis direction by the right-hand rule)")
        hr.addWidget(self.ld_rot_rpm)
        self.ld_rot_alpha_lbl = QLabel("α")
        hr.addWidget(self.ld_rot_alpha_lbl)
        hr.addWidget(self.ld_rot_alpha)
        b = QPushButton("Axis from selection")
        b.setToolTip("Axis from the selected cylindrical face or circle (Edge filter)")
        b.clicked.connect(self.rotation_axis_from_selection)
        hr.addWidget(b)
        self.ld_rot_axis_btn = b
        self.ld_rot_w_lbl = QLabel("Speed")
        f.addRow(self.ld_rot_w_lbl, hr)
        self.ld_rot_parts = QListWidget()
        self.ld_rot_parts.setMaximumHeight(90)
        self.ld_rot_parts.setToolTip("The checked parts rotate (nothing checked = all parts)")
        self.ld_rot_parts_lbl = QLabel("Rotates")
        f.addRow(self.ld_rot_parts_lbl, self.ld_rot_parts)
        self.ld_pt = [dspin(0.0, -1e7, 1e7, 3) for _ in range(3)]
        hp = QHBoxLayout()
        for s, ax in zip(self.ld_pt, "xyz"):
            s.setToolTip(f"{ax} coordinate of the remote point [mm] where the force / moment acts")
            hp.addWidget(s)
        self.ld_pt_auto = QCheckBox("face center")
        self.ld_pt_auto.setChecked(True)
        self.ld_pt_auto.setToolTip("Point = centroid of the selected faces (uncheck for a custom point, e.g. the end of a lever)")
        hp.addWidget(self.ld_pt_auto)
        self.ld_pt_lbl = QLabel("Point [mm]")
        f.addRow(self.ld_pt_lbl, hp)
        self.register_point_input(self.ld_pt, self.ld_pt_auto)
        self.ld_coupling = QComboBox()
        self.ld_coupling.addItems(["distributing (does not add stiffness)", "rigid"])
        help_tip(self.ld_coupling, "Distributing = the face deforms, rigid = the face acts as a rigid body",
                 "How the point acts on the faces: distributing = forces distributed by area, the face deforms freely; rigid = the face moves as a rigid body (adds stiffness).")
        self.ld_coupling_lbl = QLabel("Constraint")
        f.addRow(self.ld_coupling_lbl, self.ld_coupling)
        # thermal load: field from a thermal state (for a transient one at a time) or uniform temperature of the parts
        self.ld_th_src = QComboBox()
        help_tip(self.ld_th_src, "Temperature from a thermal case (Heat step), or a uniform part temperature",
                 "Temperature field from a thermal case (Heat step), or a uniform temperature of the selected parts.<br>Stress from ΔT = T − reference temperature (Heat step).")
        self.ld_th_src.currentIndexChanged.connect(lambda _: self.on_ld_type(0))
        self.ld_th_src_lbl = QLabel("Temperature from")
        f.addRow(self.ld_th_src_lbl, self.ld_th_src)
        self.ld_th_time_w = QWidget()
        ht = QHBoxLayout(self.ld_th_time_w)
        ht.setContentsMargins(0, 0, 0, 0)
        self.ld_th_time = dspin(0.0, 0.0, 1e9, 3, suffix=" s")
        self.ld_th_time.setToolTip("Time of the transient thermal case whose temperature field is used (linear between snapshots)")
        self.ld_th_end = QCheckBox("end of the state")
        self.ld_th_end.setToolTip("Use the temperature field at the end of the transient case")
        self.ld_th_end.setChecked(True)
        self.ld_th_end.toggled.connect(lambda on: self.ld_th_time.setEnabled(not on))
        self.ld_th_time.setEnabled(False)
        ht.addWidget(self.ld_th_time, 1)
        ht.addWidget(self.ld_th_end)
        self.ld_th_time_lbl = QLabel("Time")
        f.addRow(self.ld_th_time_lbl, self.ld_th_time_w)
        self.ld_th_parts = QListWidget()
        self.ld_th_parts.setMaximumHeight(90)
        self.ld_th_parts.setToolTip("The checked parts have the specified temperature (nothing checked = all parts)")
        self.ld_th_parts_lbl = QLabel("Parts")
        f.addRow(self.ld_th_parts_lbl, self.ld_th_parts)
        self._ld_form = f
        hb = QHBoxLayout()
        b = QPushButton("Add from selection")
        b.setToolTip("Add the load from the form to the selected case on the selected faces")
        b.setWhatsThis("<b>Add load</b><p>1. Select the load case above. 2. Select faces by clicking in 3D (edges with the E filter, points with the V filter). 3. Choose the load type and fill in the values. 4. Press “Add from selection”. Arrows in 3D show the direction.</p><p>A force also works on edges and points; gravity, rotation and temperature need no selection.</p>")
        self.btn_add_load = b
        theme.set_primary(b)
        b.clicked.connect(self.add_load)
        hb.addWidget(b)
        b = QPushButton("Save changes to selected")
        b.setToolTip("Overwrite the values of the load selected in the list (faces stay)")
        b.clicked.connect(self.update_load)
        hb.addWidget(b)
        f.addRow(hb)
        lay.addWidget(g)
        h = QHBoxLayout()
        b = QPushButton("Faces from selection")
        help_tip(b, "Replace the faces of the selected load with the current selection",
                 "Replace the faces (edges / points for an edge / point force) of the selected load with the current selection")
        b.clicked.connect(lambda: self.update_faces_of(self.load_list, "load"))
        h.addWidget(b)
        b = QPushButton("Delete")
        b.setToolTip("Delete the load selected in the list (Ctrl+Z to undo)")
        b.clicked.connect(lambda: self.delete_item(self.load_list, "load"))
        h.addWidget(b)
        lay.addLayout(h)
        self.on_ld_type(0)

    # ------------------------------------------------------------------ Mesh panel
    def _panel_mesh(self, lay):
        lay.addWidget(panel_header("Mesh", "Quadratic tetrahedra tet10 (C3D10), Gmsh", STAGE["Mesh"] + 1))
        g = Section("Settings", key="mesh.settings", tip="Global mesh settings – the element size applies to the whole model")
        f = QFormLayout(g.body)
        self.ms_size = dspin(5.0, 0.01, 1e5, 3, suffix=" mm")
        self.ms_min = dspin(0.0, 0.0, 1e5, 3, suffix=" mm")
        self.ms_min.setSpecialValueText("auto")
        self.ms_curv = QSpinBox()
        self.ms_curv.setRange(0, 72)
        self.ms_curv.setValue(12)
        self.ms_curv.setSpecialValueText("off")
        help_tip(self.ms_size, "Target element size [mm] – smaller = more accurate and slower",
                 "Target tet10 element size [mm]. Smaller = more accurate and slower. Start with ~1/10–1/20 of the part size, at least 2 elements across thin walls; check the convergence (reduce by ~30 % and compare the results).")
        help_tip(self.ms_min, "Smallest allowed element size [mm] (auto = by Gmsh)",
                 "Smallest allowed element size [mm] (auto = by Gmsh) – limits the refinement in small details (fillets, holes)")
        help_tip(self.ms_curv, "Elements per circle circumference – refines the mesh in holes and fillets",
                 "Number of elements around the whole circle – refines the mesh in holes and fillets (12 is usually enough, more for stress concentration at a hole)")
        f.addRow("Element size", self.ms_size)
        f.addRow("Min. size", self.ms_min)
        f.addRow("Elements on circle", self.ms_curv)
        for w in (self.ms_size, self.ms_min, self.ms_curv):
            w.valueChanged.connect(self.on_mesh_settings)
        lay.addWidget(g)
        g = Section("Local refinement", key="mesh.refine", tip="Smaller elements on the selected faces and in a sphere around a point")
        v = QVBoxLayout(g.body)
        v.setContentsMargins(f.contentsMargins())     # same margins as the Settings section
        self.ref_list = QListWidget()
        self.ref_list.currentRowChanged.connect(self.on_ref_row)
        self.ref_list.setToolTip("Local mesh refinement on selected faces (element size at the face)")
        set_placeholder(self.ref_list, "No local refinement yet")
        fit_rows(self.ref_list, 2, 6)
        v.addWidget(self.ref_list)
        h = QHBoxLayout()
        self.ref_size = dspin(1.0, 0.01, 1e5, 3, suffix=" mm")
        self.ref_size.setToolTip("Element size on the selected faces [mm] (smaller than the global size)")
        b = QPushButton("Refine selection")
        help_tip(b, "Refine the mesh on faces selected in 3D (notches, holes, transitions)",
                 "Refine the mesh on faces selected in 3D – stress concentration spots (notches, holes, transitions).<br>Turn off “Show mesh” to select faces.")
        b.clicked.connect(self.add_refinement)
        h.addWidget(self.ref_size)
        h.addWidget(b)
        v.addLayout(h)
        b = QPushButton("Delete refinement")
        b.setToolTip("Delete the refinement selected in the list (faces and spheres)")
        b.clicked.connect(lambda: self.delete_item(self.ref_list, "refine"))
        v.addWidget(b)
        # refinement in a sphere around a point (MeshSettings.local {"center", "radius", "size"}, Gmsh Ball field)
        gb = Section("Refine around point (sphere)", key="mesh.ball", collapsed=True,
                     tip="Smaller elements inside a sphere around a point – also inside the volume",
                     help="Smaller elements inside a sphere around a point – also inside the volume (notch, weld root, stress maximum spot).<br>Outside the sphere the global element size is kept.")
        fb = QFormLayout(gb.body)
        hp = QHBoxLayout()
        self.ball_pt = [dspin(0.0, -1e7, 1e7, 3) for _ in range(3)]
        for s, ax in zip(self.ball_pt, "xyz"):
            help_tip(s, f"{ax} coordinate of the sphere center [mm]",
                     f"{ax} coordinate of the sphere center [mm] (enter it, “Point from selection”, or the edge / circle center with the Edge filter → “Use center” in the status bar)")
            hp.addWidget(s)
        self.register_point_input(self.ball_pt)
        b = QPushButton("Point from selection")
        help_tip(b, "Sphere center from the selected point, edge / circle, or faces",
                 "Sphere center = the last point selected in 3D with the Point filter (V), otherwise the center of the selected edge / circle (Edge filter), otherwise the centroid of the selected faces")
        b.clicked.connect(self.ball_point_from_selection)
        hp.addWidget(b)
        fb.addRow("Center [mm]", hp)
        hr = QHBoxLayout()
        self.ball_r = dspin(5.0, 0.001, 1e6, 3, 1.0, " mm")
        self.ball_r.setToolTip("Refinement sphere radius [mm] – usually 2–5 × the detail size")
        self.ball_h = dspin(0.5, 0.001, 1e5, 3, 0.1, " mm")
        self.ball_h.setToolTip("Element size inside the sphere [mm] (smaller than the global size)")
        hr.addWidget(QLabel("R"))
        hr.addWidget(self.ball_r, 1)
        hr.addWidget(QLabel("element"))
        hr.addWidget(self.ball_h, 1)
        fb.addRow("Sphere", hr)
        b = QPushButton("Refine around point")
        b.setToolTip("Add a refinement in a sphere (center, radius, element size)")
        b.setWhatsThis("<b>Refine around point</b><p>The mesh inside the sphere will be finer (element size “element”), also inside the volume – suitable for a stress maximum from the results or a weld root. Enter the point as numbers, select it with the Point filter (V) and press “Point from selection”, or take the edge center (Edge filter → “Use center”).</p><p>The refinement is shown in the list above.</p>")
        b.clicked.connect(self.add_ball_refinement)
        self.btn_ball = b
        fb.addRow(b)
        v.addWidget(gb)
        lay.addWidget(g)
        h = QHBoxLayout()
        self.btn_mesh = QPushButton("Generate mesh")
        help_tip(self.btn_mesh, "Generate the mesh (Gmsh, tet10) – the analysis also creates it by itself",
                 "Generate the mesh (Gmsh, quadratic tetrahedra tet10) and show it.<br>Not required: Run analysis creates the mesh by itself.")
        self.btn_mesh.clicked.connect(self.make_mesh)
        theme.set_primary(self.btn_mesh)
        self.btn_mesh_cancel = QPushButton("Cancel")
        self.btn_mesh_cancel.setEnabled(False)
        self.btn_mesh_cancel.setToolTip("Terminate the Gmsh meshing process")
        self.btn_mesh_cancel.clicked.connect(self.cancel_mesh)
        h.addWidget(self.btn_mesh, 1)
        h.addWidget(self.btn_mesh_cancel)
        lay.addLayout(h)
        self.ms_show = QCheckBox("Show mesh")
        self.ms_show.setChecked(True)
        help_tip(self.ms_show, "On = the mesh is shown in 3D; off = parts and face selection",
                 "On: the surface of the finished mesh is shown in 3D (parts hidden).<br>Off: parts and face selection by clicking – e.g. for Refine selection.")
        self.ms_show.toggled.connect(self.on_mesh_show)
        lay.addWidget(self.ms_show)
        self.mesh_info = QLabel("")
        self.mesh_info.setWordWrap(True)
        lay.addWidget(self.mesh_info)

    # ------------------------------------------------------------------ Solve panel
    def _panel_solve(self, lay):
        outer = lay                                 # content in the scroll area (natural frequencies, buckling, 2D)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidget(inner)
        outer.addWidget(sc, 100)
        lay.addWidget(panel_header("Solve", "Statics – direct solver (CalculiX) or iterative PCG/AMG (NGSolve + hypre)", STAGE["Solve"] + 1))
        self.ready_list = QListWidget()
        fit_rows(self.ready_list, 4, 12)
        help_tip(self.ready_list, "Readiness check: errors (cannot run) and warnings",
                 "Study readiness check: errors (the analysis cannot be run) and warnings (materials, supports, rigid body motions, loads…). Fix the errors in the steps listed.")
        lay.addWidget(self.ready_list)
        self.solver_section = Section("Solver", key="solve.solver", collapsed=True,
                                      tip="Solver choice: automatic, direct (CalculiX, PARDISO) or iterative (hypre)")
        f = QFormLayout(self.solver_section.body)
        from fea_core.solvers import backend_status, install_hint, linear_solver_note
        avail = backend_status()                  # solvers not installed are greyed out (cannot be selected; the check reports the error)
        self.backend_combo = QComboBox()
        for key, txt in BACKENDS:
            st_ = avail.get(key)
            missing = st_ is not None and not st_["ok"]
            self.backend_combo.addItem(f"{txt} – not installed" if missing else txt, key)   # template for i18n
            if missing:
                it = self.backend_combo.model().item(self.backend_combo.count() - 1)
                it.setEnabled(False)
                it.setToolTip(f"Missing {', '.join(st_['missing'])}")
        large = f"{config.LARGE_DOF:,}".replace(",", " ")      # space as the thousands separator (in the number only)
        help_tip(self.backend_combo, "Solver: automatic, direct (CalculiX, PARDISO) or iterative PCG + AMG",
                 f"<b>Solver</b><p><b>Automatic</b> (recommended): contacts, bolts, welds, cylindrical and remote supports → CalculiX; from {large} DOF the iterative NGSolve + hypre (if installed); from {config.PARDISO_MIN_CASES} load cases NGSolve + PARDISO (CalculiX if it fails); otherwise CalculiX. The automatic choice skips solvers that are not installed.</p><p><b>Direct sparse solver – CalculiX</b>: verified by all verification problems, memory grows fast (limit ~1–2 million DOF).<br><b>NGSolve + PARDISO</b>: direct, one factorization for all cases – faster from 2 cases (module ngsolve).<br><b>NGSolve + hypre</b>: iterative PCG with AMG, memory grows linearly – large models (modules ngsolve and hypre, Microsoft MPI on Windows).</p><p>In other programs: direct = ANSYS Direct (Sparse), SOLIDWORKS Direct Sparse, the Nastran direct solver; iterative = ANSYS Iterative (PCG), SOLIDWORKS FFEPlus. MUMPS is not offered – PARDISO and PaStiX fill its role.</p><p>Grayed-out solvers are not installed – button <i>Install solvers…</i> (Tools → Program modules…).</p>")
        self.backend_combo.currentIndexChanged.connect(self.on_backend)
        f.addRow("Solver", self.backend_combo)
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["PARDISO", "PASTIX", "SPOOLES"])
        for i in range(self.solver_combo.count()):
            note = linear_solver_note(self.solver_combo.itemText(i))
            if note:                              # PARDISO without MKL, PaStiX in the Linux CalculiX
                it = self.solver_combo.model().item(i)
                it.setEnabled(False)
                it.setToolTip(f"Not available – {note}")
        self.solver_combo.currentTextChanged.connect(self.on_solver)
        help_tip(self.solver_combo, "Equation solver in CalculiX: PARDISO (default), PaStiX, SPOOLES",
                 "<b>Equation solver (CalculiX)</b><p>Direct solver of the equation system inside CalculiX (the NGSolve solvers do not use it): PARDISO (Intel MKL, fastest, default), PaStiX (Windows only), SPOOLES (slower, always available). The results are the same.</p><p>A grayed-out solver is not in this installation – CalculiX then uses a substitute (PARDISO → PaStiX / SPOOLES), which is stated in the log or the result warnings.</p>")
        f.addRow("Equation solver (CalculiX)", self.solver_combo)
        miss = {k: v for k, v in avail.items() if not v["ok"]}
        self.btn_install_solvers = QPushButton("Install solvers…")
        help_tip(self.btn_install_solvers, "Opens Program modules – install the missing solvers",
                 "<b>Install more solvers</b><p>" + "<br>".join(
                     f"{v['title']}: missing {', '.join(v['missing'])}" for v in miss.values())
                 + "</p><p>" + install_hint() + "</p>")
        self.btn_install_solvers.clicked.connect(self.open_modules_dialog)
        f.addRow(self.btn_install_solvers)
        f.setRowVisible(self.btn_install_solvers, bool(miss))     # only when something is missing
        lay.addWidget(self.solver_section)
        self.backend_combo.currentIndexChanged.connect(self._backend_badge)
        self._backend_badge()
        self.eigen_box = EigenBox(self)          # natural frequencies and linear buckling (Study.modal, Study.buckling)
        lay.addWidget(self.eigen_box)
        self.twod_box = twod_gui.TwoDBox(self)    # 2D calculation by a section (Study.twod)
        lay.addWidget(self.twod_box)
        self.shell_box = shell_gui.ShellBox(self)  # shell model of plates and sheets (Study.shell)
        lay.addWidget(self.shell_box)
        h = QHBoxLayout()
        self.btn_run = QPushButton("Run analysis")
        self.btn_run.setToolTip("Run analysis: check → mesh → statics → equilibrium → results")
        self.btn_run.setWhatsThis("<b>Run analysis</b><p>First the study is checked (list above). Then the mesh is generated, the thermal cases are solved and the linear statics of each load case. At the end the program verifies the equilibrium of reactions and loads – a result with an imbalance is not returned.</p><p>When finished, the Results step opens. Progress is in the log below.</p>")
        theme.set_primary(self.btn_run)
        self.btn_run.clicked.connect(self.run_solve)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_solve)
        self.btn_cancel.setToolTip("Abort the running analysis (terminates solver processes)")
        self.btn_stop_keep = QPushButton("Stop and keep results")
        self.btn_stop_keep.setEnabled(False)
        self.btn_stop_keep.clicked.connect(self.stop_keep_solve)
        help_tip(self.btn_stop_keep, "Stop the analysis and keep the last converged increment (partial)",
                 "Nonlinear analysis (contact, step sequence): stops CalculiX and keeps the last converged increment as a PARTIAL result (red label). If there is no converged increment the analysis ends with an error.")
        h.addWidget(self.btn_run)
        h.addWidget(self.btn_cancel)
        h.addWidget(self.btn_stop_keep)
        lay.addLayout(h)
        self.solve_progress = TaskProgress()      # green bar 0–100 % (progress from the core, fea_core.progress)
        lay.addWidget(self.solve_progress)
        self.solve_info = QLabel("")
        self.solve_info.setWordWrap(True)
        lay.addWidget(self.solve_info)
        from .nl_monitor import NlMonitor           # convergence of a nonlinear analysis (A7), live and from the results
        self.nl_monitor = NlMonitor(self)
        lay.addWidget(self.nl_monitor)
        lay.addStretch(1)

    # ------------------------------------------------------------------ Results panel
    def _panel_results(self, lay):
        lay.addWidget(panel_header("Results", "Click in the model to show the value (probe)", STAGE["Results"] + 1))
        from .result_overview import ResultOverview   # verdict of safety against the requirement + numbers of the state
        self.res_overview = ResultOverview(self)
        self.res_overview.setVisible(False)
        lay.addWidget(self.res_overview)
        outer = lay                                 # content in the scroll area (legend + summary + maximum places)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        f = QFormLayout()
        self.r_step = QComboBox()
        self.r_step.setToolTip("Load case (or thermal case “Heat: …”) whose results are displayed")
        self.r_qty = QComboBox()
        help_tip(self.r_qty, "Displayed quantity: stress, displacement, safety factor, temperature…",
                 "Displayed quantity: von Mises stress, displacements, principal stress, stress components, safety factor to Re, for printing the Tsai-Wu/Hill safety factor, temperature, ccx stress error estimate;<br>for a mode / buckling shape the normalized displacement (max |u| = 1)")
        for name, unit in QUANTITIES:
            self.r_qty.addItem(f"{name} [{unit}]")
        self.r_scale = QComboBox()
        self.r_scale.addItems(["undeformed", "automatic", "1×", "10×", "100×", "1000×"])
        self.r_scale.setCurrentIndex(1)
        help_tip(self.r_scale, "Deformation scale in 3D (does not change the result values)",
                 "Deformation scale in 3D (automatic = the largest displacement scaled to 5 % of the model diagonal); the result values are not affected by the scale")
        self.r_bands = QSpinBox()
        self.r_bands.setRange(4, 16)
        self.r_bands.setValue(12)
        self.r_bands.setToolTip("Number of legend color bands")
        self.r_csys = QComboBox()
        self.r_csys.setToolTip("Local coordinate system for stresses and displacements (e.g. cylindrical system)")
        self.r_csys.addItem("Global Cartesian", None)
        self.r_csys_lbl = QLabel("System")
        self.r_edges = QCheckBox("element edges")
        self.r_edges.setToolTip("Show mesh element edges over the color map")
        self.r_outline = QCheckBox("undeformed model outline")
        self.r_outline.setToolTip("Show the undeformed model outline (for comparison with the deformed shape)")
        self.r_part = QComboBox()
        self.r_part.setToolTip("Show only the selected assembly part with recalculated color scale (Display Group, A5)")
        self.r_part.addItem("All parts", None)
        self.r_part.currentIndexChanged.connect(lambda _: self.show_result())
        self.r_time = QComboBox()                   # snapshot of a transient thermal state
        self.r_time.setToolTip("Snapshot time of the transient thermal case")
        self.r_time_lbl = QLabel("Time")
        for lbl, w in (("Load case", self.r_step), ("Assembly part", self.r_part), (self.r_time_lbl, self.r_time),
                       ("Quantity", self.r_qty), (self.r_csys_lbl, self.r_csys),
                       ("Deformation", self.r_scale), ("Number of bands", self.r_bands)):
            f.addRow(lbl, w)
        self.r_time.setVisible(False)
        self.r_time_lbl.setVisible(False)
        # dynamics (F8): frequency / time slider, phase of the harmonic response, absolute displacement (+ base motion)
        self.r_dyn = QWidget()
        hd = QHBoxLayout(self.r_dyn)
        hd.setContentsMargins(0, 0, 0, 0)
        self.r_dyn_slider = QSlider(Qt.Horizontal)
        help_tip(self.r_dyn_slider, "Frequency / time whose field is shown (arrows ← → step by one point)",
                 "Frequency of the harmonic response / time of the transient response whose field is shown (arrows ← → step by one point)")
        self.r_dyn_val = QLabel("")
        self.r_dyn_val.setMinimumWidth(90)
        self.r_phase = QSpinBox()
        self.r_phase.setRange(0, 360)
        self.r_phase.setSingleStep(15)
        self.r_phase.setSuffix(" °")
        self.r_phase.setWrapping(True)
        help_tip(self.r_phase, "Phase φ of the harmonic response for the “at phase φ” quantities",
                 "Phase φ of the harmonic response (the moment of the period ωt = φ) for the “at phase φ” quantities;<br>the amplitude over phase does not depend on the phase")
        self.r_abs = QCheckBox("absolute")
        help_tip(self.r_abs, "Absolute displacement = relative + base motion (base excitation only)",
                 "Absolute displacement = relative + base motion (base excitation only); otherwise displacement relative to the base")
        hd.addWidget(self.r_dyn_slider, 1)
        hd.addWidget(self.r_dyn_val)
        hd.addWidget(self.r_phase)
        hd.addWidget(self.r_abs)
        self.r_dyn_lbl = QLabel("Frequency")
        f.insertRow(2, self.r_dyn_lbl, self.r_dyn)
        self.r_dyn.setVisible(False)
        self.r_dyn_lbl.setVisible(False)
        self.r_dyn_slider.valueChanged.connect(lambda _: self.show_result())
        self.r_phase.valueChanged.connect(lambda _: self.show_result())
        self.r_abs.toggled.connect(lambda _: self.show_result())
        f.addRow(self.r_edges)
        f.addRow(self.r_outline)
        lay.addLayout(f)
        grid = QGridLayout()                      # result tool buttons in 2 columns (full labels)
        grid.setSpacing(6)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        def add_btn(btn):
            n = grid.count()
            grid.addWidget(btn, n // 2, n % 2)

        b = QPushButton("Line plot…")
        help_tip(b, "Values along a line: chart, line in 3D, stress linearization, CSV",
                 "Values of the quantity along a line between two points (click in the model or enter coordinates): chart, line in 3D, stress linearization through the wall (membrane + bending, ASME VIII-2) and CSV export")
        b.clicked.connect(self.open_path_dialog)
        self.btn_path = b
        add_btn(b)
        b = QPushButton("Mesh convergence…")
        help_tip(b, "Run on progressively finer meshes and check that the quantity settles",
                 "Run the study on progressively finer meshes (globally or in a sphere around a point) and check that the monitored quantity settles (Richardson extrapolation, GCI). The results of the main window do not change.")
        b.clicked.connect(self.open_convergence_dialog)
        self.btn_conv = b
        add_btn(b)
        b = QPushButton("Print detail (voxels)…")
        help_tip(b, "Local voxel model of the print beads at the critical spot (indicative)",
                 "Local voxel model of beads (perimeters, layers, infill) around the critical spot of a printed part – boundary displacements from this result, safety factor in the bead axes (INDICATIVE)")
        b.clicked.connect(self.open_voxel_dialog)
        self.btn_voxel = b
        add_btn(b)
        b = QPushButton("Submodel…")
        help_tip(b, "Finer local model of a region driven by this result (box or sphere)",
                 "Submodel: the parts are cut by a box or a sphere around the critical spot and meshed finer; the displacement of this result is prescribed on the cut faces, supports and loads inside the region are transferred. Cut check = submodel vs global stress at the cut.")
        b.clicked.connect(self.open_submodel_dialog)
        self.btn_submodel = b
        add_btn(b)
        b = QPushButton("Section force…")
        help_tip(b, "Resultant internal forces and moments in a cut (Free Body Cut)",
                 "Calculation of resultant internal forces and moments in a section (Free Body Cut, A3): N, Vy, Vz, Mx, My, Mz")
        b.clicked.connect(self.open_section_force_dialog)
        self.btn_sec_force = b
        add_btn(b)
        b = QPushButton("Export (VTU, CSV)...")
        b.setToolTip("Export results to ParaView (VTU/PVD) or CSV tables (nodal values and reactions, A6)")
        b.clicked.connect(self.open_export_dialog)
        add_btn(b)

        b_anim = QPushButton("Animate...")
        b_anim.setToolTip("Deformation animation (A5) - smooth cycle of deflection scaling")
        b_anim.clicked.connect(self.toggle_animation)
        self.btn_anim = b_anim
        add_btn(b_anim)

        b_iso = QPushButton("Isosurface...")
        b_iso.setToolTip("Extraction and export of an isosurface at the chosen value (marching tetrahedra, A5)")
        b_iso.clicked.connect(self.open_isosurface_dialog)
        self.btn_iso = b_iso
        add_btn(b_iso)

        b_vec = QPushButton("Stress vectors...")
        b_vec.setToolTip("Principal stress vectors σ1, σ2, σ3 and export (A5)")
        b_vec.clicked.connect(self.open_vectors_dialog)
        self.btn_vec = b_vec
        add_btn(b_vec)

        b = QPushButton("Plasticity assessment…")
        help_tip(b, "PEEQ check per EN 1993-1-5 (5 %) and ASME VIII-2 local failure, residual stress")
        b.clicked.connect(self.open_plasticity_dialog)
        self.btn_plast = b
        add_btn(b)

        b_topo = QPushButton("Topology (BESO)...")
        b_topo.setToolTip("Topology optimization to lighten the structure for 3D printing (BESO, C4a)")
        b_topo.clicked.connect(self.open_topo_dialog)
        self.btn_topo = b_topo
        add_btn(b_topo)
        lay.addLayout(grid)
        self.r_step.currentIndexChanged.connect(lambda _: self.on_r_step())
        for w in (self.r_qty, self.r_scale, self.r_time, self.r_csys):
            w.currentIndexChanged.connect(lambda _: self.show_result())
        self.r_bands.valueChanged.connect(lambda _: self.show_result())
        self.r_edges.toggled.connect(lambda _: self.show_result())
        self.r_outline.toggled.connect(lambda _: self.show_result())
        self.legend = Legend()
        self.legend.setMinimumHeight(250)           # 16 bands fit, the results panel is tall
        self.legend.setWhatsThis("<b>Legend</b><p>Color bands of the displayed quantity from the minimum to the maximum of the displayed case.</p>")
        lay.addWidget(self.legend)
        self.res_info = SafeTextBrowser()
        self.res_info.setMinimumHeight(200)
        help_tip(self.res_info, "Summary: maxima, reactions, equilibrium, safety factor; a click in the model = probe",
                 "State summary: maxima, support reactions, equilibrium check, safety factor per part;<br>a click in the model gives the value at the point (probe)")
        # maximum places with singularity classification (same as MCP fea_results_hotspots and the report)
        self.hs_box = QWidget()
        v = QVBoxLayout(self.hs_box)
        v.setContentsMargins(2, 4, 2, 2)
        self.hs_title = QLabel("<b>Highest stresses</b>")
        v.addWidget(self.hs_title)
        self.hs_table = table(["#", "value", "part", "location [mm]", "classification"], 160)
        self.hs_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.hs_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.hs_table.setWordWrap(True)             # long classification ("singularity – clamped edge") on 2 lines
        help_tip(self.hs_table, "Spots of the highest values with classification (singularity, boundary condition…)",
                 "Separate spots of the highest values (clusters of ~3 elements) with automatic classification:<br>singularity (support edge, sharp inner edge, point load, bolt) – do not evaluate by the nodal value; boundary condition – check the support; real spot – check by mesh convergence.<br>A click on a row marks the spot in 3D and shows the value (probe).")
        self.hs_table.currentCellChanged.connect(lambda r, *_: self.on_hotspot_row(r))
        v.addWidget(self.hs_table)
        self.hs_note = QLabel("")
        self.hs_note.setWordWrap(True)
        self.hs_note.setTextFormat(Qt.RichText)
        v.addWidget(self.hs_note)
        self.res_tabs = QTabWidget()                # state summary | maximum places (saves panel height)
        self.res_tabs.addTab(self.res_info, "Summary")
        self.res_tabs.addTab(self.hs_box, "Highest stresses")
        self.modes_table = ModesTable(self)          # natural frequency / buckling: frequencies and effective mass fractions
        self.res_tabs.addTab(self.modes_table, "Mode shapes")
        self.res_tabs.setTabToolTip(2, "Natural frequencies and buckling factors, effective mass fractions x/y/z and sums")
        self.res_tabs.setTabVisible(2, False)
        self.frame_box = frs.FrameResultsBox(self)    # frames: extremes per member and EN 1993 assessment
        self.res_tabs.addTab(self.frame_box, "Members")
        self.res_tabs.setTabToolTip(3, "Internal forces per member and EN 1993-1-1 check")
        self.res_tabs.setTabVisible(3, False)
        self.dyn_box = dynr.DynResultsBox(self)      # F8: dynamics plots (FRF, time history, PSD, SRS) and mode table
        self.TAB_DYN = self.res_tabs.addTab(self.dyn_box, "Dynamics")
        self.res_tabs.setTabToolTip(self.TAB_DYN, "Dynamics charts in probes (FRF, time history, response PSD, pulse SRS) and table of modes with effective mass")
        self.res_tabs.setTabVisible(self.TAB_DYN, False)
        self.contact_box = dynr.ContactResultsBox(self)   # F8: contact pair forces, force–displacement curve, partial result
        self.TAB_CONTACT = self.res_tabs.addTab(self.contact_box, "Contacts")
        self.res_tabs.setTabToolTip(self.TAB_CONTACT, "Contact pairs of the displayed case (forces, area, pressure, opening, slip) and force–travel curve")
        self.res_tabs.setTabVisible(self.TAB_CONTACT, False)
        lay.addWidget(self.res_tabs)
        self.r_scroll = QScrollArea()
        self.r_scroll.setWidgetResizable(True)
        self.r_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.r_scroll.setFrameShape(QFrame.NoFrame)
        self.r_scroll.setWidget(inner)
        outer.addWidget(self.r_scroll, 100)

    # ================================================================== kroky postupu
    @_batched
    def set_stage(self, k):
        name = STAGES[k]
        if not self.stage_visible(name):             # a hidden optional step (tree, guide, IPC) appears on the bar
            self._flow_revealed.add(name)
        self.stage = k
        self._phase_last[PHASE_OF[name]] = name
        self.stage_actions[k].setChecked(True)
        self._apply_flow_visibility()                # current phase and step (narrow bar) + possibly a new step
        for attr, (stage, _) in LAZY_PANELS.items():
            if stage == name:
                getattr(self, attr)                   # builds the panel at its first display
        self.panel_stack.setCurrentIndex(k)
        self._update_panel_window_title()
        self._update_shelf(name)                     # step commands in the tool shelf
        self.update_view_info()                      # popis pohledu nad 3D oknem
        if hasattr(self, "props_card"):
            self.props_card.schedule()
        if hasattr(self, "table_dock") and self.table_dock.set_stage(name) and name != "Solve" and not self.busy():
            self.dock_tables.raise_()                # bottom dock: the step table (for Solve and during a calculation the log stays)
        if name == "Results" and hasattr(self, "res_overview"):
            self.res_overview.refresh()
        v = self.view
        # the 3D window shows the mesh (Mesh step, "Show mesh") – after the mesh is discarded it returns to the parts (changed)
        self._mesh_view = name == "Mesh" and self.mesh is not None and self.ms_show.isChecked()
        v.show_persist(not (name == "Results" and self.results is not None))   # frame system (persistent objects)
        if name == "Results" and self.results is not None:
            v.show_parts(False)
            v.show_sets(False)
            v.set_mode("results")
            v.clear_section()
            self.show_result()
        elif name == "Mesh" and self.mesh is not None and self.ms_show.isChecked():
            v.show_parts(False)
            v.show_sets(False)
            v.set_mode("faces")
            self.show_mesh()
        elif name == "Fatigue":
            self.fat_panel.refresh()
            if self.fat_panel.result is not None:
                self.fat_panel.show_map()
            else:
                v.clear_results()
                v.show_parts(True)
                v.show_sets(False)
                v.set_mode("model")
        else:
            v.clear_results()
            v.show_parts(True)
            v.show_sets(name in ("Joints", "Contacts", "Constraints", "Motion", "Heat", "Loads", "Dynamics", "Mesh",
                                 "Solve"))
            v.set_mode("faces" if name in ("Joints", "Contacts", "Members", "Constraints", "Motion", "Heat", "Loads",
                                           "Dynamics", "Mesh") else "model")
        if name == "3D printing":
            self.print_panel.legend.hide()
            self.print_panel.refresh()
        if name == "Joints":
            self.joints_panel.compute()
            if hasattr(self, "table_dock"):
                self.table_dock.refresh()                # joint assessment into the Bolts / Welds tables
        if name == "Solve":
            self.refresh_ready()
        if name == "Members":
            self.frame_panel.refresh()
        if name == "Heat":
            self.thermal_panel.update_info()
        if name == "Contacts":
            self.contact_panel.update_info()
        if name == "Dynamics":
            self.dynamics_panel.update_info()
        if name == "Motion":
            self.motion_panel.update_info()
        elif self._built("motion_panel") is not None:
            self.motion_panel.reset_pose()          # mechanism animation only in the Motion step (otherwise the CAD position)
        if name == "Materials":            # parts selected in the list are highlighted in 3D too
            v.set_selected_parts([i.row() for i in self.mat_parts.selectedIndexes()])
        if not (name == "Results" and self.results is not None):
            self.apply_model_section()
        if v.mode != "faces":          # face selection stays between the Constraints / Loads / Mesh steps
            v.set_selected_faces([])
        self.refresh_sets()
        self.update_status()

    # ================================================================== file
    def geometry_str(self) -> str:
        """Path to the STEP for Study.geometry ("" = frame system without CAD)."""
        return str(self.step_path) if self.step_path else ""

    def frame_only(self) -> bool:
        """Study with frames only (no STEP geometry)."""
        return self.study is not None and not self.parts and self.step_path is None

    @_batched
    def new_frame_study(self):
        """New frame-system study without STEP geometry (Study.geometry == "")."""
        if self.busy():
            msg(self, APP_NAME, "Analysis in progress – wait for it to finish.")
            return
        if not self.confirm_discard():
            return
        st = Study(name="Frame", geometry="")
        st.add_material(matlib.default(), default=True)
        st.load_cases.append(LoadCase("Load case 1"))
        self._load_model(None, st, None)
        self.project_path = None
        self.dirty = False
        self.reset_undo()
        self.update_title()
        self.set_stage(STAGE["Members"])

    def open_dialog(self):
        last = self.settings.value("last_dir", "")
        p, _ = QFileDialog.getOpenFileName(self, "Open", last,
                                           "Model nebo projekt (*.step *.stp *.fea *.ipt *.f3d *.f3z);;Projekt FEA "
                                           f"(*.fea);;STEP (*.step *.stp);;{NATIVE_FILTER}")
        if p:
            self.settings.setValue("last_dir", str(Path(p).parent))
            self.open_file(p)

    # ------------------------------------------------------------------ dropping a file onto the window
    @staticmethod
    def dropped_files(mime) -> list[Path]:
        """Files .step / .stp / .fea dropped (Explorer, desktop, CAD)."""
        if mime is None or not mime.hasUrls():
            return []
        return [Path(u.toLocalFile()) for u in mime.urls()
                if u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in DROP_SUFFIXES]

    @staticmethod
    def _accept_copy(e):
        """Accept a drop always as a copy: the program only reads the file. The suggested action with Shift would be a move
        (MoveAction) and Explorer would then delete the original file.
        """
        e.setDropAction(Qt.CopyAction)
        e.accept()

    def dragEnterEvent(self, e):
        if self.dropped_files(e.mimeData()):
            self._accept_copy(e)
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if self.dropped_files(e.mimeData()):
            self._accept_copy(e)
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        files = self.dropped_files(e.mimeData())
        if not files:
            super().dropEvent(e)
            return
        self._accept_copy(e)
        if len(files) > 1:
            self.log_msg(f"{len(files)} files dropped – only the first one is opened: {files[0].name}")
        self.raise_()
        self.activateWindow()
        # open only after returning from the drop: the "Save changes?" question inside dropEvent would block Explorer
        QTimer.singleShot(0, lambda p=str(files[0]): self.open_file(p))

    def busy(self) -> bool:
        """Meshing, calculation or report is running (the thread works with the study or the mesh)."""
        return (self._thread is not None or getattr(self, "_report_thread", None) is not None
                or getattr(self.fat_panel, "_thread", None) is not None or bool(self._aux_threads))

    @_batched
    def open_file(self, path):
        if self.busy():
            msg(self, APP_NAME, "Meshing, analysis or report generation is in progress – wait for it to finish or cancel the analysis (Solve / Mesh step).")
            return
        if not self.confirm_discard():
            return
        path = opened = Path(path)
        old_study = self.study
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if path.suffix.lower() == ".fea":
                wd = self._workdir / f"proj_{int(time.time() * 1000)}"
                study, step, results = load_project(path, wd)
                from fea_core.project import load_motion
                self.motion_result = load_motion(wd)          # mechanism result (Motion step), may be missing
                self._motion_sig = mech_signature(study) if self.motion_result is not None else None
                self._load_model(step, study, results)
                new_path = path
            elif path.suffix.lower() == ".json":             # studie JSON (fea_core run, MCP fea_study_save, REST)
                study, step = load_study_json(path)          # Save then asks for the .fea project (JSON is not overwritten)
                self.motion_result = self._motion_sig = None
                self._load_model(step, study, None)
                new_path = None
            else:
                self.motion_result = self._motion_sig = None
                native_warn: list[str] = []
                if is_native(path):                          # .ipt / .f3d → STEP in the cache (module native)
                    path = native_to_step(path, native_warn)
                self._load_model(path, None, None)
                for w in native_warn:
                    self.log_msg(f"Native file: {w}")
                    self.messages.add("warning", "geometry check", w, "Model")
                new_path = None
        except Exception as e:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            if self.study is not old_study:
                # model already replaced, error only when the panels are restored: the study belongs to the new file → the path
                # and title of the previous project must not stay (Ctrl+S would overwrite it with a foreign study); save is possible only as a new file
                self.project_path = None
                self.dirty = True
                self.reset_undo()
                self.update_title()
            msg(self, APP_NAME, f"The file cannot be opened:\n{e}", QMessageBox.Critical)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.project_path = new_path               # only after a successful load (Ctrl+S does not overwrite another file)
        self.dirty = False
        self.reset_undo()
        self.update_title()
        session_files.note_opened(self, opened)    # File → Recent files
        self.set_stage(STAGE["Materials"] if self.results is None else STAGE["Results"])

    def _load_model(self, step: Path | None, study: Study | None, results, parts=None, geo_warn=None):
        self._flow_revealed = set()              # optional steps open in the previous study are not kept on the bar
        geo_warn = list(geo_warn or [])          # geometry check (only when a model is loaded, not when calculating)
        legacy = legacy_step_names(study)        # saved study with parts "NAUO1" (SolidWorks STEP, version 1.0.0)
        if step is None:                         # frame system without CAD (Study.geometry == "")
            parts = []
        elif parts is None:                      # parts = parts already loaded (geometry update)
            parts = read_step(step, report=geo_warn, legacy_names=legacy)
        surfaces = []                            # plate parts from STEP (read_step skips them) – shells
        want = set((study.shell.parts if study is not None else {}) or {})
        if step is not None and (not parts or want - {p.name for p in parts}):
            from fea_core.shell.geometry import read_step_surfaces
            try:
                surfaces = [p for p in read_step_surfaces(step) if p.name not in {q.name for q in parts}]
            except Exception as e:  # noqa: BLE001
                geo_warn.append(f"cannot load surface parts: {e}")
        if step is not None and not parts and not surfaces:
            raise RuntimeError("The STEP file contains no body or face")
        self.surface_names = {p.name for p in surfaces}
        if surfaces:                             # faces were loaded as plate parts of shells – the read_step warning does not apply
            geo_warn = [g for g in geo_warn if "no volume" not in g and "no solid" not in g]
        parts = list(parts) + surfaces
        self.step_path = Path(step) if step is not None else None
        self.parts = parts
        # part names from the CAD (and the file name = default study name) are not translated in the English GUI
        keep_names([p.name for p in parts] + ([self.step_path.stem] if self.step_path else []))
        self.face_cache = {}
        self.mesh = results.mesh if results is not None else None
        self.results = results
        self.fat_panel.result = None             # fatigue and joint assessment belong to the previous model
        self.joints_panel.assessment = None
        self._rigid = None
        new_study = study is None and step is not None
        cad_msgs: list[tuple[str, str]] = []
        cad_log: list[str] = []
        if study is None:
            study = Study(name=self.step_path.stem, geometry=str(self.step_path))
            study.add_material(matlib.default(), default=True)
            bb = np.array([bbox(p.shape) for p in parts])
            diag = float(np.linalg.norm(bb[:, 3:].max(axis=0) - bb[:, :3].min(axis=0)))
            study.mesh.size = nice_floor(diag / 30)
            study.load_cases.append(LoadCase("Load case 1"))
            if surfaces and len(surfaces) == len(parts):    # only faces → shell model (the user enters the thickness)
                from fea_core.shell.model import ShellSettings
                study.shell = ShellSettings(True, {p.name: None for p in surfaces})
        if new_study:                            # new study from a STEP: materials from the CAD metadata (<name>.json next to it)
            cad_msgs, cad_log = self._cad_metadata(Path(step), study, [p.name for p in parts])
            try:                                 # fastener components of the CAD assembly → virtual bolts
                from fea_core import cad_import
                added, bw = cad_import.apply_fasteners(study, parts, cad_import.load_sidecar(Path(step)))
            except Exception as e:  # noqa: BLE001 – bolts are an extra, the model opens without them too
                added, bw = [], [f"Virtual bolts from the assembly cannot be created: {e}"]
            if added:
                text = (f"Virtual bolts from the CAD assembly ({len(added)}): {', '.join(added[:8])}"
                        + (" …" if len(added) > 8 else "") + " – assessment in the Joints step")
                cad_msgs.append(("info", text))
                cad_log.append(text)
            for w in bw:
                cad_msgs.append(("warning", w))
                cad_log.append(w)
        self.study = study
        self._mesh_sig = self.study_signature(self.MESH_KEYS) if self.mesh is not None else None
        self.view.set_parts(parts)
        self.view.clear_persist()
        self.log_msg(f"Loaded: {self.step_path.name} – {len(parts)} bodies" if self.step_path else
                     f"Frame without CAD: {len(study.frame.nodes)} nodes, {len(study.frame.members)} members")
        for w in geo_warn:
            self.log_msg(f"Geometry check: {w}")
        if legacy:
            self.log_msg("STEP parts are named as in the older program version (SolidWorks occurrences “NAUO1”…) so that the materials, selections and bolts of the saved study stay valid")
        name_warn: list[str] = []                # only the message and the log, not a window (selected faces are reported by resolve_all)
        if not new_study and step is not None:   # material of a part that is not in the STEP (renamed in the CAD?)
            names = {p.name for p in parts}
            lost = [n for n in study.part_materials if n not in names]
            if lost:
                listing = ", ".join(lost[:8]) + (" …" if len(lost) > 8 else "")
                name_warn.append(f"A material is assigned to parts that are not in the STEP file (renamed in CAD?): {listing}")
                self.log_msg(name_warn[-1])
        for line in cad_log:
            self.log_msg(line)
        for src in ("mesh", "solver", "results"):           # messages belong to the previous model
            self.messages.set_source(src, [])
        if geo_warn or name_warn:
            self.messages.set_source("geometry check", [("warning", w) for w in geo_warn + name_warn], "Model")
        else:
            self.messages.set_source("geometry check", [])
        self.messages.set_source(CAD_SOURCE, cad_msgs, "Materials")
        if self._path_dlg is not None:
            self._path_dlg.close()
        self.messages.set_source(F8_SOURCE, [])
        if results is not None:
            self.mesh_messages(results.mesh)
            self.messages.set_source("results", [("warning", w) for w in results.warnings], unique=True)
            self.f8_messages(results)
        missing = self.resolve_all()
        self.refresh_all()
        if geo_warn:
            msg(self, APP_NAME, "The geometry check found problems (the calculation may fail or be inaccurate):\n\n"
                + "\n".join(geo_warn[:20]) + ("\n…" if len(geo_warn) > 20 else ""))
        if missing:
            msg(self, APP_NAME, "Some selected faces were not found in the model (did the model change?):\n\n"
                + "\n".join(missing))

    def _cad_metadata(self, step: Path, study: Study, part_names: list[str]) -> tuple[list, list[str]]:
        """CAD metadata next to the STEP (<name>.json – Fusion add-in, SolidWorks / Inventor macros, REST import):
        materials of the parts by matching the library (fea_core.cad_import). Returns (messages [(level, text)], log lines);
        nothing blocks – a material outside the library is only a warning, the part gets the default material of the study.
        """
        from fea_core import cad_import
        warn: list[str] = []
        try:
            meta = cad_import.load_sidecar(step, warn)
            if meta is None:
                return [("warning", w) for w in warn], list(warn)
            res = cad_import.apply_metadata(study, part_names, meta)
        except Exception as e:  # noqa: BLE001 – metadata is an extra, the model opens without it too
            text = f"CAD metadata cannot be used: {e}"
            return [("warning", text)], [text]
        items = res.messages(study.default_material)
        lines = [f"{res.summary()} – file {cad_import.sidecar_path(step).name}"]
        for r in res.rows:
            if r.get("material"):
                lines.append(f"Material from CAD: {r['name']} – “{r['material_match']['cad_material']}” → {r['material']}")
        lines += [t for _, t in items[1:]]
        return items, lines

    def resolve_all(self) -> list[str]:
        self.sel_faces, self.sel_edges, self.sel_points = {}, {}, {}
        missing = []
        names = [p.name for p in self.parts]
        for name, sel in self.study.selections.items():
            faces = []
            for fsel in sel.faces:
                r = resolve(fsel, self.parts, self.face_cache)
                if r is None:
                    missing.append(f"{name}: {fsel.describe()}")
                else:
                    faces.append(r)
            self.sel_faces[name] = faces
            if sel.edges:
                edges = []
                for esel in sel.edges:
                    r = resolve_edge(esel, self.parts, self.edge_cache)
                    if r is None:
                        missing.append(f"{name}: {esel.describe()}")
                    else:
                        edges.append(r)
                self.sel_edges[name] = edges
            if sel.points:
                pts = []
                for psel in sel.points:
                    if psel.part in names:
                        pts.append((names.index(psel.part), tuple(psel.point)))
                    else:
                        missing.append(f"{name}: {psel.describe()}")
                self.sel_points[name] = pts
        return missing

    def save(self):
        if self.project_path is None:
            return self.save_as()
        return self._save(self.project_path)

    def export_report(self):
        from .report_dialog import export_report
        if self.study is None:
            msg(self, APP_NAME, "Open a model or project first.")
            return
        export_report(self)

    def save_as(self):
        if self.study is None:
            return False
        last = self.settings.value("last_dir", "")
        default = str(Path(last) / (self.study.name + ".fea")) if last else self.study.name + ".fea"
        p, _ = QFileDialog.getSaveFileName(self, "Save project", default, "FEA project (*.fea)")
        if not p:
            return False
        if not p.lower().endswith(".fea"):
            p += ".fea"
        return self._save(Path(p))

    def _save(self, path: Path):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            save_project(path, self.study, self.geometry_str(), self.results, motion=self.motion_result)
        except Exception as e:  # noqa: BLE001
            msg(self, APP_NAME, f"Saving failed:\n{e}", QMessageBox.Critical)
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.project_path = path
        self.dirty = False
        self.update_title()
        self.log_msg(f"Saved: {path}")
        session_files.note_saved(self, path)       # recent files; the autosave of this session is no longer needed
        link = self.study.cad_link
        if link is not None and link.document_id:    # the CAD add-in finds the project even without a running GUI
            from fea_core.cad_link.registry import remember
            remember(link.document_id, path, link.document)
        return True

    def confirm_discard(self):
        if not self.dirty:
            return True
        r = QMessageBox.question(self, APP_NAME, "The study has unsaved changes. Save?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return bool(self.save())
        return r == QMessageBox.Discard

    def changed(self, invalidate_results=True, invalidate_mesh=False):
        self.dirty = True
        if invalidate_mesh:
            self.mesh = None
        if invalidate_results and self.results is not None:
            self.results = None
            self.log_msg("Study changed – results discarded, recomputation needed.")
            self.messages.set_source("results", [])
            self.messages.set_source(F8_SOURCE, [])
            if self._path_dlg is not None:
                self._path_dlg.close()
        if invalidate_results:
            self.fat_panel.result = None
            self.joints_panel.assessment = None
        self.push_undo()
        self.update_title()
        self.refresh_all()
        self._sync_mesh_view()

    def _sync_mesh_view(self):
        """Mesh discarded (change of mesh settings, refinement, selections…), but the 3D window still shows it → back to the parts
        with the face selection.
        """
        if getattr(self, "_mesh_view", False) and self.mesh is None:
            self.set_stage(self.stage)

    def on_mesh_show(self, _=None):
        """The "Show mesh" switch in the Mesh step: the mesh surface, or the parts with face selection (refinement)."""
        if STAGES[getattr(self, "stage", 0)] == "Mesh":
            self.set_stage(self.stage)

    # ------------------------------------------------------------------ undo / redo
    def reset_undo(self):
        self._undo = [json.dumps(self.study.to_dict())] if self.study else []
        self._undo_pos = len(self._undo) - 1

    def push_undo(self):
        if self.study is None:
            return
        snap = json.dumps(self.study.to_dict())
        if self._undo and self._undo[self._undo_pos] == snap:
            return
        del self._undo[self._undo_pos + 1:]
        self._undo.append(snap)
        if len(self._undo) > 200:
            self._undo.pop(0)
        self._undo_pos = len(self._undo) - 1

    def undo(self):
        if self.study is not None and self._undo_pos > 0:
            self._restore(self._undo_pos - 1)

    def redo(self):
        if self.study is not None and self._undo_pos < len(self._undo) - 1:
            self._restore(self._undo_pos + 1)

    # effect of a change: nothing < assessment only < FEA calculation < mesh; the joint settings by effect are in the core
    # (fea_core.joints.levels – shared with MCP fea_joints_*)
    LEVELS = JOINT_LEVELS

    def joints_change_level(self, old: dict, new: dict) -> str:
        """Effect of the difference in bolts / welds / joint settings between two dictionaries of the study (undo / redo) – the same as
        changes in the Joints step (joints_panel.change_level). Unclear (other bolts / welds, order) = new mesh.
        """
        oj, nj = old.get("joints") or {}, new.get("joints") or {}
        lv = self.LEVELS.index(joint_settings_level(oj, nj))
        preload = bool(nj.get("preload_case", True))
        for key, kind in (("bolts", "bolt"), ("welds", "weld")):
            a, b = old.get(key) or [], new.get(key) or []
            if [x.get("name") for x in a] != [x.get("name") for x in b]:
                return "mesh"
            for x, y in zip(a, b):
                lv = max(lv, self.LEVELS.index(change_level(kind, x, y, preload) or ""))
        return self.LEVELS[lv]

    def _restore(self, pos):
        old = self.study.to_dict()
        new = json.loads(self._undo[pos])
        mesh_ok = self.mesh is not None and self.valid_mesh() is not None     # before switching the study
        self._undo_pos = pos
        link = self.study.cad_link                 # the iteration history is not restored (it is not a settings change)
        self.study = Study.from_dict(new)
        self.study.cad_link = link
        old.pop("cad_link", None)
        new.pop("cad_link", None)
        self.resolve_all()

        def _calc(d, k):
            v = d.get(k)
            if k == "printing" and isinstance(v, dict):   # definition of the sweep, variants and required safety
                v = {x: y for x, y in v.items() if x not in ("sweep", "variants", "required_safety")}   # nejsou vstupy
            return v                                                                                   # calculation

        def differ(keys):
            return [k for k in keys if json.dumps(_calc(old, k), sort_keys=True) != json.dumps(_calc(new, k), sort_keys=True)]
        joints = ("bolts", "welds", "joints")
        try:
            jl = self.joints_change_level(old, new) if differ(joints) else ""
        except Exception:  # noqa: BLE001 – unclear = discard as before
            jl = "mesh"
        level = self.LEVELS.index(jl)
        if differ(("geometry", "mesh", "selections", "ignored_parts")):
            level = 3
        elif differ((set(old) | set(new)) - {"fatigue", "hidden_parts"} - set(joints)):
            level = max(level, 2)                  # other calculation inputs (constraints, loads, materials…)
        if level == 3:
            self.mesh = None
        elif mesh_ok and differ(joints):
            self._mesh_sig = self.study_signature(self.MESH_KEYS)   # joints outside the meshing inputs – the mesh stays valid
        # results are discarded only when the calculation inputs change (fatigue settings and joint assessment fields do not change them)
        if level >= 2:
            self.results = None
            self.joints_panel.assessment = None
            self.fat_panel.result = None
        else:
            if level == 1:                         # only the joint assessment – recalculate from the valid results
                self.joints_panel.assessment = None
                self.joints_panel.compute(force=True)
            if differ(("fatigue",)):
                self.fat_panel.result = None
        self.dirty = True
        self.update_title()
        self.refresh_all()
        self.set_stage(STAGE["Solve"] if STAGES[self.stage] == "Results" and self.results is None else self.stage)
        self.log_msg("Undo" if pos < len(self._undo) - 1 else "Redo")

    def update_title(self):
        name = self.project_path.name if self.project_path else (self.step_path.name if self.step_path else
                                                                 (self.study.name if self.study else ""))
        self.setWindowTitle(f"{name}{' *' if self.dirty else ''} – {APP_NAME}" if name else APP_NAME)
        if hasattr(self, "doc_label"):
            title = self.study.name if self.study is not None and self.study.name else ""
            parts = [t for t in (title, name if name != title else "") if t]
            self.doc_label.setText(" · ".join(parts) + (" *" if self.dirty else ""))

    # ================================================================== face selection
    def on_faces_picked(self, faces):
        self.update_status()

    @staticmethod
    def arrow_point(p):
        """Arrow point: centroid of a planar face (lies on it unless it has holes), otherwise a point on the face."""
        return p["centroid"] if p["type"] == "plane" else p["point"]

    def face_prop(self, pi, fi) -> dict:
        props = self.face_cache.get(pi)
        if props is None:
            props = self.face_cache[pi] = [face_props(f) for f in self.view.part_faces[pi]]
        return props[fi]

    def selected_selectors(self) -> list[FaceSelector]:
        out = []
        for pi, fi in self.view.selected_faces():
            props = self.face_cache.get(pi)
            if props is None:
                props = self.face_cache[pi] = [face_props(f) for f in self.view.part_faces[pi]]
            out.append(FaceSelector.from_props(self.parts[pi].name, props[fi]))
        return out

    def need_faces(self):
        sels = self.selected_selectors()
        if not sels:
            msg(self, APP_NAME, "First select one or more faces in the 3D view.")
        return sels

    def edge_prop(self, pi, ei) -> dict:
        from fea_core.geometry import edge_props
        props = self.edge_cache.get(pi)
        if props is None:
            props = self.edge_cache[pi] = [edge_props(e) for e in self.view.part_edges(pi)]
        return props[ei]

    def selected_edge_selectors(self) -> list[EdgeSelector]:
        return [EdgeSelector.from_props(self.parts[pi].name, self.edge_prop(pi, ei))
                for pi, ei in self.view.selected_edges()]

    def selected_point_selectors(self) -> list[PointSelector]:
        return [PointSelector(self.parts[pi].name, [float(c) for c in q]) for pi, q in self.view.selected_points()]

    def load_target(self) -> str | None:
        """What the force from the form acts on: edges (filter Edge), points (filter Point), otherwise faces; no selection → None."""
        f = self.view.pick_filter
        e, pts, fc = self.view.selected_edges(), self.view.selected_points(), self.view.selected_faces()
        if f == "edge" and e:
            return "edges"
        if f == "point" and pts:
            return "points"
        if fc:
            return "faces"
        return "edges" if e else "points" if pts else None

    def new_entity_selection(self, prefix, entity) -> str:
        """New edge / point selection from the current 3D selection."""
        prefix = tr(prefix)                         # new names in the program language
        k = 1
        while f"{prefix} {k}" in self.study.selections:
            k += 1
        name = f"{prefix} {k}"
        if entity == "edges":
            self.study.add_selection(name, edges=self.selected_edge_selectors())
            self.sel_edges[name] = self.view.selected_edges()
        else:
            self.study.add_selection(name, points=self.selected_point_selectors())
            self.sel_points[name] = self.view.selected_points()
        return name

    def new_selection(self, prefix, sels) -> str:
        prefix = tr(prefix)                         # new names in the program language
        k = 1
        while f"{prefix} {k}" in self.study.selections:
            k += 1
        name = f"{prefix} {k}"
        self.study.add_selection(name, sels)
        self.sel_faces[name] = self.view.selected_faces()
        return name

    def update_model_status(self):
        """Status bar on the right: equilibrium (worst state), number of elements, calculation time, units."""
        if not hasattr(self, "st_model"):
            return
        out, state = [], ""
        fs = getattr(self, "flow_state", None)          # readiness check (refresh_flow)
        if self.study is not None and fs is not None:
            errs, warns = fs[1], fs[2]
            out.append(f"× {tr('errors')}: {len(errs)}" if errs else
                       (f"! {tr('warning')}: {len(warns)}" if warns else "√ " + tr("Readiness check")))
        res = self.results
        imb = [s.equilibrium["imbalance"] for s in (res.steps if res is not None else [])
               if s.equilibrium and "imbalance" in s.equilibrium]
        if imb:
            worst = max(imb)
            ok = worst <= config.EQUILIBRIUM_TOL
            out.append(("√ " if ok else "× ") + "Equilibrium " + sci(worst))
            state = "ok" if ok else "fail"
        mesh = res.mesh if res is not None and getattr(res, "mesh", None) is not None else getattr(self, "mesh", None)
        n_el = len(mesh.tets) if mesh is not None and getattr(mesh, "tets", None) is not None else 0
        if n_el:
            out.append(f"{fmt(n_el)} elements")
        dt = getattr(res, "_gui_solve_s", None) if res is not None else None
        if dt is not None:
            out.append(decimal(f"{dt:.1f} s"))
        out.append("mm · N · MPa")
        self.st_model.setText(" · ".join(out))
        self.st_model.setProperty("state", state)
        self.st_model.style().unpolish(self.st_model)
        self.st_model.style().polish(self.st_model)

    def update_status(self):
        n = len(self.view.selected_faces())
        out = []
        if n:
            area = sum(s.area for s in self.selected_selectors())
            out.append(f"Faces selected: {n}   A = {fmt(area)} mm²")
        ne = len(self.view.selected_edges())
        if ne:
            length = sum(self.edge_prop(pi, ei)["length"] for pi, ei in self.view.selected_edges())
            out.append(f"edges: {ne}   L = {fmt(length)} mm")
        npt = len(self.view.selected_points())
        if npt:
            out.append(f"points: {npt}")
        self.st_sel.setText("   |   ".join(out))
        if hasattr(self, "ld_target") and STAGES[getattr(self, "stage", 0)] == "Loads":
            self.on_ld_type(0)

    # ================================================================== materials
    def refresh_materials(self, select: str | None = None):
        """Part list and material menu after a change of the study. The selected material (or `select`, e.g. the one just
        assigned) stays selected; the form is reloaded only when the selection changes (unfinished values stay).
        """
        rows = {i.row() for i in self.mat_parts.selectedIndexes()} if self.mat_parts.count() == len(self.parts) else set()
        self.mat_parts.blockSignals(True)
        self.mat_parts.clear()
        for i, p in enumerate(self.parts):
            m = self.study.material_of(p.name).name if self.study else "–"
            self.mat_parts.addItem(f"{p.name}   [{m}]")
            if i in rows:
                self.mat_parts.item(i).setSelected(True)
        self.mat_parts.blockSignals(False)
        prev = self.mat_combo.currentText()
        select = select or self._mat_select
        self._mat_select = None
        want = select or prev
        self.mat_combo.blockSignals(True)
        self.mat_combo.clear()
        names = [m.name for m in matlib.all_materials()]
        if self.study:
            names += [n for n in self.study.materials if n not in names]
        self.mat_combo.addItems(names)
        i = self.mat_combo.findText(want) if want else -1
        if i >= 0:
            self.mat_combo.setCurrentIndex(i)
        self.mat_combo.blockSignals(False)
        if i < 0 or select is not None or want != prev or self._mat_form_src != self._mat_source(want):
            self.on_mat_combo(self.mat_combo.currentIndex())

    def _mat_source(self, name):
        """Material by name (the study has priority over the library) as text – the form is reloaded only when
        the material changed (undo / redo, another study).
        """
        m = self.study.materials.get(name) if self.study else None
        if m is None:
            try:
                m = matlib.by_name(name)
            except KeyError:
                return None
        return name, repr(m)

    def on_mat_combo(self, _):
        name = self.mat_combo.currentText()
        if not name:
            return
        m = self.study.materials.get(name) if self.study else None
        if m is None:
            try:
                m = matlib.by_name(name)
            except KeyError:
                return
        self._mat_form_src = (name, repr(m))
        self.m_name.setText(m.name)
        self.m_E.setValue(m.E)
        self.m_nu.setValue(m.nu)
        self.m_rho.setValue(m.rho * 1e12)
        self.m_Re.setValue(m.Re or 0)
        self.m_Rm.setValue(m.Rm or 0)
        self.m_k.setValue(m.k or 0)
        self.m_alpha.setValue((m.alpha or 0) * 1e6)
        self.m_cp.setValue(m.cp or 0)
        self.m_src.setText(m.source)
        self._mat_print = {k: getattr(m, k) for k in PRINT_FIELDS} if m.is_print else {}
        self.m_print.setText(
            f"<i>Printed material (E, Rm = layer plane XY): E_z {fmt(m.E_z)} MPa, Rm_z {fmt(m.Rm_z)} MPa – shell, infill and other properties in the 3D printing step.</i>" if m.is_print else
            f"<i>{self.ortho_box.describe(m)}</i>" if m.is_ortho else "")
        self._mat_form.setRowVisible(self.m_print, bool(self.m_print.text()))
        self.ortho_box.load(m)
        self.plasticity_box.load(m)
        self.update_mat_summary()

    def on_mat_find(self, text):
        """Designation converter: equivalents in other standards, matching / similar material from the library."""
        from fea_core import grades
        self._mat_links = []
        if not text.strip():
            self.mat_conv.setText("")
            self.mat_conv.setVisible(False)
            return
        hits = grades.lookup(text)
        if not hits:
            self.mat_conv.setText("<i>The designation is not in the converter.</i>")
            self.mat_conv.setVisible(True)
            return
        g, approx = hits[0]

        def link(name, extra=""):
            self._mat_links.append(name)
            return f"<a href='mat:{len(self._mat_links) - 1}'>{escape(str(name))}</a>{extra}"
        rows = [f"<b>{g.key}</b> – {g.family_label}{' (approximate match only)' if approx else ''}",
                g.summary(html=True)]
        if len(hits) > 1:
            rows.append("Also: " + ", ".join(f"{h.key}{' (≈)' if a else ''}" for h, a in hits[1:4]))
        mats = {m.name: m for m in matlib.all_materials()}
        if self.study:
            mats.update(self.study.materials)
        same, similar = [], []
        q = grades.profile(g)
        for m in mats.values():
            mg = grades.grade_of(m.name)
            if mg is g:
                same.append(m.name)
            elif mg is not None:
                d = grades.distance(q, grades.profile(mg, rm=m.Rm))
                if d <= 6:
                    similar.append((d, m.name, grades.difference(q, grades.profile(mg, rm=m.Rm))))
        if same:
            rows.append("In the library: " + ", ".join(link(n) for n in same))
            if self.mat_combo.findText(same[0]) >= 0:
                self.mat_combo.setCurrentText(same[0])
        else:
            rows.append("Not in the library" + (" – similar: " + ", ".join(
                link(n, f" <small>({escape(str(why))})</small>" if why else "") for _, n, why in sorted(similar)[:4])
                if similar else " or anything similar – enter the properties and save them to your library."))
        try:
            from fea_core.fatigue import pragtic
            _, exact, _ = pragtic.find(text, sn_only=True)
            rows.append(f"Fatigue data (PragTic): {len(exact)} records with S-N – Fatigue step.")
        except FileNotFoundError:
            pass
        self.mat_conv.setText("<br>".join(rows))
        self.mat_conv.setVisible(True)

    def on_mat_parts_sel(self):
        # valid rows only: when the model is replaced the list briefly keeps the rows of the previous model
        rows = [i.row() for i in self.mat_parts.selectedIndexes() if i.row() < len(self.parts)]
        if STAGES[getattr(self, "stage", 0)] == "Materials":
            self.view.set_selected_parts(rows)
        if rows:
            m = self.study.material_of(self.parts[rows[0]].name)
            i = self.mat_combo.findText(m.name)
            if i >= 0:
                self.mat_combo.setCurrentIndex(i)

    def on_parts_picked(self, idx):
        """Click on parts in 3D in part-selection mode: in the Materials step (and 3D printing) selects the parts in the list."""
        names = [self.parts[i].name for i in idx if 0 <= i < len(self.parts)]
        self.st_info.setText(("Selected parts: " + ", ".join(names)) if names else "")
        stage = STAGES[getattr(self, "stage", 0)]
        if stage == "Model":
            if hasattr(self, "model_table"):
                self.model_table.blockSignals(True)
                self.model_table.clearSelection()
                for i in idx:
                    if i < self.model_table.rowCount():
                        self.model_table.selectRow(i)
                self.model_table.blockSignals(False)
                if idx and idx[0] < self.model_table.rowCount():
                    it = self.model_table.item(idx[0], 0)
                    if it is not None:
                        self.model_table.scrollToItem(it)
            return
        if stage == "Materials":
            lw, after = self.mat_parts, self.on_mat_parts_sel
        elif stage == "3D printing":
            lw, after = self.print_panel.parts, self.print_panel.on_part_sel
        else:
            return
        lw.blockSignals(True)
        lw.clearSelection()
        for i in idx:
            it = lw.item(i)
            if it is not None:
                it.setSelected(True)
        lw.blockSignals(False)
        if idx and lw.item(idx[0]) is not None:
            lw.scrollToItem(lw.item(idx[0]))
        after()

    def material_from_form(self, name: str) -> Material:
        """Material from the form of the Materials step (print properties are taken from the selected material; an orthotropic
        material has E = E1, ν = ν12).
        """
        ortho, ocs = self.ortho_box.values()
        E, nu = (ortho["E1"], ortho["nu12"]) if ortho else (self.m_E.value(), self.m_nu.value())
        return Material(name, E, nu, self.m_rho.value() * 1e-12,
                        Re=self.m_Re.value() or None, Rm=self.m_Rm.value() or None, source=self.m_src.text(),
                        k=self.m_k.value() or None, alpha=(self.m_alpha.value() * 1e-6) or None,
                        cp=self.m_cp.value() or None, ortho=ortho, ortho_csys=ocs, plastic=self.plasticity_box.values(),
                        **self._mat_print)

    def save_user_material(self):
        name = self.m_name.text().strip()
        if not name:
            msg(self, APP_NAME, "Enter a material name.")
            return
        if name in {m.name for m in matlib.LIBRARY}:
            msg(self, APP_NAME, "A built-in material cannot be overwritten – choose a different name.")
            return
        try:
            m = self.material_from_form(name)
            matlib.save_user_material(m)
        except ValueError as e:                  # unreadable plasticity table
            msg(self, APP_NAME, str(e))
            return
        except matlib.UserLibraryError as e:          # an unreadable library file is not overwritten
            msg(self, APP_NAME, str(e), QMessageBox.Critical)
            return
        self.log_msg(f"Material '{name}' saved to {matlib.user_library_path()}")
        self.refresh_materials(select=name)

    # ================================================================== hole selection, section
    def select_holes(self):
        from fea_core.geometry import coaxial_cylinders
        sel = self.view.selected_faces()
        if not sel:
            msg(self, APP_NAME, "Select the cylindrical face of a hole or pin.")
            return
        out = list(sel)
        for pi, fi in sel:
            for j in coaxial_cylinders(self.view.part_faces[pi], fi):
                if (pi, j) not in out:
                    out.append((pi, j))
        self.view.set_selected_faces(out, emit=True)
        self.log_msg(f"Selection extended to {len(out)} faces (coaxial cylindrical faces).")

    # ================================================================== selection filter, edges
    def set_pick_filter(self, key: str):
        """Selection filter in 3D: Face (P) / Body (B) / Hole (H) / Edge (E). Hole also extends the already selected
        cylindrical faces to whole holes (previously the "Whole hole" button).
        """
        a = self.filter_actions.get(key)
        if a is not None and not a.isChecked():
            a.setChecked(True)
        self.view.set_filter(key)
        if hasattr(self, "pick_button"):
            self._sync_pick_button(key)
        if key != "edge":
            self.on_edge_picked(None)
        if key == "hole" and self.view.mode == "faces":
            from fea_core.geometry import cylinder_axis
            sel = self.view.selected_faces()
            if any(cylinder_axis(self.view.part_faces[pi][fi]) is not None for pi, fi in sel):
                self.select_holes()
        txt = next(t for k, t, _, _ in PICK_FILTER_ACTIONS if k == key)
        self.statusBar().showMessage(f"Selection filter: {txt}", 3000)

    @staticmethod
    def _pt(p, dec=6) -> str:
        return "(" + "; ".join(fmt(float(c) + 0.0, dec) for c in p) + ")"

    def describe_edge(self, d) -> str:
        part = self.parts[d["part"]].name if 0 <= d.get("part", -1) < len(self.parts) else ""
        s = f"Edge {d['edge'] + 1} of part {part}: {d['label']}, length {fmt(d['length'], 6)} mm"
        if d["type"] == "circle":
            s += (f", R {fmt(d['radius'], 6)} mm (Ø {fmt(2 * d['radius'], 6)} mm), center {self._pt(d['center'])} mm, axis {self._pt(np.round(d['axis'], 6), 4)}")
        elif d["type"] == "ellipse":
            s += f", semi-axes {fmt(d['major'], 6)} / {fmt(d['minor'], 6)} mm, center {self._pt(d['center'])} mm"
        else:
            s += f", edge center {self._pt(d['mid'])} mm"
            if d["type"] == "line":
                s += f" (from {self._pt(d['start'])} to {self._pt(d['end'])})"
        return s

    def on_edge_picked(self, info):
        self._edge_info = info
        self.btn_edge_point.setVisible(info is not None)
        if info is None and self.st_info.text().startswith("Edge "):
            self.st_info.setText("")
        if info is not None:
            self.btn_edge_point.setText("Use circle center" if "center" in info else "Use edge center")
            self.st_info.setText(self.describe_edge(info))
            self.log_msg(self.describe_edge(info))

    def register_point_input(self, spins, auto=None):
        """Point field [mm] (three QDoubleSpinBox) for "Use centre" of an edge; auto = check box of the automatic point
        ("face centre"), which is cleared when a point is inserted. The first visible field is used (open form).
        """
        self._point_inputs.append((list(spins), auto))

    def use_edge_point(self):
        """Centre of a circle / edge to the clipboard and into the point field of the open form (remote load / constraint)."""
        d = self._edge_info
        if d is None:
            return
        p = [float(c) + 0.0 for c in d["point"]]
        txt = "; ".join(fmt(c, 6) for c in p)
        QApplication.clipboard().setText(txt)
        what = "Circle center" if "center" in d else "Edge center"
        target = next(((s, a) for s, a in self._point_inputs if all(w.isVisible() for w in s)), None)
        if target is None:
            self.statusBar().showMessage(f"{what} ({txt}) mm copied to the clipboard (no form with a point – remote load / support – is open).", 8000)
            return
        spins, auto = target
        if auto is not None:
            auto.setChecked(False)
        for s, c in zip(spins, p):
            s.setValue(c)
        self.statusBar().showMessage(f"{what} ({txt}) mm inserted into the point field and copied to the clipboard.", 8000)

    SEC_FACE = 4                           # the item "face" in sec_axis

    def _sec_range(self, n):
        bb = np.array([bbox(p.shape) for p in self.parts])
        lo3, hi3 = bb[:, :3].min(axis=0), bb[:, 3:].max(axis=0)
        corners = np.array([[x, y, z] for x in (lo3[0], hi3[0]) for y in (lo3[1], hi3[1]) for z in (lo3[2], hi3[2])])
        proj = corners @ n
        return float(proj.min()), float(proj.max())

    def section_plane(self):
        """(point, normal) of the section plane – the side (x − o)·n ≤ 0 is shown; None = section off."""
        k = self.sec_axis.currentIndex()
        if k <= 0 or not self.parts:
            return None
        if k == self.SEC_FACE:
            if self._sec_face is None:
                return None
            n = np.asarray(self._sec_face[1], float)
        else:
            n = np.zeros(3)
            n[k - 1] = 1.0
        lo, hi = self._sec_range(n)
        s = lo + (hi - lo) * self.sec_pos.value() / 1000.0
        return n * s, (-n if self.sec_flip.isChecked() else n)

    def on_sec_axis(self, k):
        """Section by the plane of a face: normal and position from the first selected face (the slider then moves it)."""
        if k == self.SEC_FACE:
            sel = self.view.selected_faces()
            if not sel:
                msg(self, APP_NAME, "For a section along a face plane, first select a face (e.g. in the Supports step), then choose “face”.")
                self.sec_axis.setCurrentIndex(0)
                return
            p = self.face_prop(*sel[0])
            n = np.asarray(p["normal"], float)
            if np.linalg.norm(n) < 1e-12:
                return
            n /= np.linalg.norm(n)
            self._sec_face = (np.asarray(p["point"], float), n)
            lo, hi = self._sec_range(n)
            self.sec_pos.blockSignals(True)
            self.sec_pos.setValue(int(round(1000 * (float(self._sec_face[0] @ n) - lo) / max(hi - lo, 1e-12))))
            self.sec_pos.blockSignals(False)
        self.update_section()

    def update_section(self, *_):
        if STAGES[getattr(self, "stage", 0)] == "Results" and self.results is not None:
            self.view.clear_section()       # results: section through the mesh in numpy (fea_core.section), not OCCT
            if getattr(self, "_sec_timer", None) is None:   # the slider: redraw only after a short pause
                self._sec_timer = QTimer(self)
                self._sec_timer.setSingleShot(True)
                self._sec_timer.timeout.connect(self.show_result)
            self._sec_timer.start(120)
            return
        self.apply_model_section()

    def apply_model_section(self):
        plane = self.section_plane()
        if plane is None:
            self.view.clear_section()
        else:                                    # OCCT keeps the side the plane normal points to
            self.view.set_section_plane(plane[0], -plane[1])

    def assign_material(self, all_parts):
        if self.study is None:
            return
        rows = list(range(len(self.parts))) if all_parts else [i.row() for i in self.mat_parts.selectedIndexes()]
        if not rows and not (all_parts and self.study.frame.active or all_parts and not self.parts):
            msg(self, APP_NAME, "Select parts in the list.")
            return
        name = self.m_name.text().strip() or "Material"
        try:
            m = self.material_from_form(name)
        except ValueError as e:                  # unreadable plasticity table
            msg(self, APP_NAME, str(e))
            return
        self.study.materials[name] = m
        for r in rows:
            self.study.part_materials[self.parts[r].name] = name
        if all_parts:
            self.study.default_material = name
        # study materials are not deleted (as in the 3D printing step): an unassigned one can be a calibrated printed material,
        # a material of print variants (printing.variants) or have overrides of print properties (printing.materials);
        # the report states only the materials used
        self._mat_select = name                  # the assigned material stays selected in the list and in the form
        self.changed()

    # ================================================================== constraints
    def add_fixed(self):
        if self.study is not None and self.study.shell.enabled and self.view.selected_edges() and \
                not self.view.selected_faces():           # shells: clamping of the edge of a plate part / sheet
            name = self.new_entity_selection("Fixed support", "edges")
            self.study.constraints.append(Fixed(name))
            self.view.clear_selection()
            self.changed()
            return
        sels = self.need_faces()
        if sels:
            name = self.new_selection("Fixed support", sels)
            self.study.constraints.append(Fixed(name))
            self.view.set_selected_faces([])
            self.changed()

    def add_displacement(self):
        sels = self.need_faces()
        if not sels:
            return
        vals = [v.value() if c.isChecked() else None for c, v in zip(self.d_chk, self.d_val)]
        if all(v is None for v in vals):
            msg(self, APP_NAME, "Check at least one displacement component.")
            return
        name = self.new_selection("Displacement", sels)
        self.study.constraints.append(Displacement(name, *vals))
        self.view.set_selected_faces([])
        self.changed()

    def on_con_row(self, row):
        if 0 <= row < len(self.study.constraints):
            c = self.study.constraints[row]
            self.view.set_selected_faces(self.sel_faces.get(c.selection, []))
            if isinstance(c, Displacement):
                for chk, val, v in zip(self.d_chk, self.d_val, (c.ux, c.uy, c.uz)):
                    chk.setChecked(v is not None)
                    val.setValue(v or 0.0)
            elif isinstance(c, CylindricalSupport):
                for chk, v in zip(self.cyl_chk, (c.radial, c.tangential, c.axial)):
                    chk.setChecked(v)
            elif isinstance(c, RemoteSupport):
                self.rs_group.setChecked(True)
                for s, v in zip(self.rs_pt, c.point):
                    s.setValue(float(v))
                for chk, v in zip(self.rs_chk, c.dofs):
                    chk.setChecked(bool(v))
                self.rs_coupling.setCurrentIndex(0 if c.coupling == "rigid" else 1)
            elif isinstance(c, ElasticSupport):
                self.el_group.setChecked(True)
                self.el_k.setValue(float(c.stiffness))
                self.el_unit.setCurrentIndex(0 if c.per_area else 1)
                self.el_dir.setCurrentIndex(max(0, self.el_dir.findData(c.direction)))
            self.update_status()

    def cylinder_faces(self, sels) -> bool:
        """Cylindrical constraint / bearing force only on cylindrical faces (hole, pin)."""
        if any(s.type != "cylinder" for s in sels):
            msg(self, APP_NAME, "Select only the cylindrical faces of the hole or pin (the Hole selection filter, H, selects the whole hole).")
            return False
        return True

    def normal_force_dir(self, sels):
        """Unit outward normal for "Force normal to the face" (area-weighted average); only planar faces, otherwise a
        message and None (the same check when adding and when editing a load).
        """
        n = np.sum([np.asarray(s.normal, float) * s.area for s in sels], axis=0) if sels else np.zeros(3)
        if np.linalg.norm(n) < 1e-12 or any(s.type != "plane" for s in sels):
            msg(self, APP_NAME, "A force normal to the face can only be applied to planar faces with the same normal.")
            return None
        return n / np.linalg.norm(n)

    def add_cylindrical(self):
        sels = self.need_faces()
        if not sels or not self.cylinder_faces(sels):
            return
        flags = [c.isChecked() for c in self.cyl_chk]
        if not any(flags):
            msg(self, APP_NAME, "Check at least one constrained component.")
            return
        name = self.new_selection("Cylindrical support", sels)
        self.study.constraints.append(CylindricalSupport(name, *flags))
        self.view.set_selected_faces([])
        self.changed()

    def update_cylindrical(self):
        row = self.con_list.currentRow()
        if not (0 <= row < len(self.study.constraints)) or \
                not isinstance(self.study.constraints[row], CylindricalSupport):
            msg(self, APP_NAME, "Select a cylindrical support in the list.")
            return
        flags = [c.isChecked() for c in self.cyl_chk]
        if not any(flags):
            msg(self, APP_NAME, "Check at least one constrained component.")
            return
        c = self.study.constraints[row]
        c.radial, c.tangential, c.axial = flags
        self.changed()
        self.con_list.setCurrentRow(row)

    def update_displacement(self):
        row = self.con_list.currentRow()
        if not (0 <= row < len(self.study.constraints)) or not isinstance(self.study.constraints[row], Displacement):
            msg(self, APP_NAME, "Select a prescribed displacement in the list.")
            return
        vals = [v.value() if c.isChecked() else None for c, v in zip(self.d_chk, self.d_val)]
        if all(v is None for v in vals):
            msg(self, APP_NAME, "Check at least one displacement component.")
            return
        c = self.study.constraints[row]
        c.ux, c.uy, c.uz = vals
        self.changed()
        self.con_list.setCurrentRow(row)

    def remote_point_from_selection(self):
        """Point of the remote constraint = centroid of the selected faces (weighted by area)."""
        sels = self.need_faces()
        if not sels:
            return
        w = np.array([s.area for s in sels])
        p = (np.array([s.centroid for s in sels]) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-30)
        for s, v in zip(self.rs_pt, p):
            s.setValue(float(v))

    def remote_form(self) -> tuple[list[float], list[bool], str] | None:
        dofs = [c.isChecked() for c in self.rs_chk]
        if not any(dofs):
            msg(self, APP_NAME, "Check at least one constrained component (ux … rz).")
            return None
        return ([s.value() for s in self.rs_pt], dofs,
                "distributing" if self.rs_coupling.currentIndex() == 1 else "rigid")

    def add_remote_support(self):
        sels = self.need_faces()
        if not sels:
            return
        form = self.remote_form()
        if form is None:
            return
        name = self.new_selection("Remote support", sels)
        self.study.constraints.append(RemoteSupport(name, *form))
        self.view.set_selected_faces([])
        self.changed()

    def update_remote_support(self):
        row = self.con_list.currentRow()
        if not (0 <= row < len(self.study.constraints)) or not isinstance(self.study.constraints[row], RemoteSupport):
            msg(self, APP_NAME, "Select a remote support in the list.")
            return
        form = self.remote_form()
        if form is None:
            return
        c = self.study.constraints[row]
        c.point, c.dofs, c.coupling = form
        self.changed()
        self.con_list.setCurrentRow(row)

    # ================================================================== loads
    def current_lc(self) -> LoadCase | None:
        i = self.lc_combo.currentIndex()
        if self.study and 0 <= i < len(self.study.load_cases):
            return self.study.load_cases[i]
        return None

    def lc_name_taken(self, name: str, own=None) -> bool:
        """The name of the load state is already used by another state (also "Bolt preload" from the results)."""
        from fea_core.joints.model import PRELOAD_CASE
        if name == PRELOAD_CASE or any(lc.name == name and lc is not own for lc in self.study.load_cases):
            msg(self, APP_NAME, f"Load case '{name}' already exists – choose a different name.")
            return True
        return False

    def add_lc(self):
        if self.study is None:
            return
        k = len(self.study.load_cases) + 1
        while any(lc.name == f"Load case {k}" for lc in self.study.load_cases):
            k += 1
        name, ok = QInputDialog.getText(self, APP_NAME, "Load case name:", text=f"Load case {k}")
        if ok and name.strip() and not self.lc_name_taken(name.strip()):
            self.study.load_cases.append(LoadCase(name.strip()))
            self.changed()
            self.lc_combo.setCurrentIndex(len(self.study.load_cases) - 1)

    def rename_lc(self):
        lc = self.current_lc()
        if lc is None:
            return
        name, ok = QInputDialog.getText(self, APP_NAME, "New name:", text=lc.name)
        if ok and name.strip():
            self.set_lc_name(lc, name.strip())

    def set_lc_name(self, lc: LoadCase, name: str):
        """Renames the load state including the references (fatigue channels, long-term creep states of 3D printing)."""
        if name == lc.name or self.lc_name_taken(name, lc):
            return
        old = lc.name
        st = self.study
        if st.fatigue is not None:
            for ch in st.fatigue.history.channels:
                if ch.load_case == old:
                    ch.load_case = name
        cr = st.printing.creep
        cr.cases = [name if c == old else c for c in cr.cases]
        lc.name = name
        self.changed()

    def ask(self, text: str) -> bool:
        """Yes / No question (default No). GUI tests replace it (w.ask = lambda *_: True)."""
        return QMessageBox.question(self, APP_NAME, text, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    def delete_lc(self):
        lc = self.current_lc()
        if lc is None:
            return
        st = self.study
        n = len(lc.loads)
        if not self.ask(f"Delete load case '{lc.name}'"
                        + (f" including its loads ({n})" if n else "") + "?\nReferences to it (fatigue channels, long-term creep states) are removed as well.\nYou can undo it with Undo (Ctrl+Z)."):
            return
        for l in lc.loads:
            self._drop_selection(getattr(l, "selection", None))
        st.load_cases.remove(lc)
        dropped = []
        if st.fatigue is not None:                  # fatigue and creep channels referring to a deleted state
            n = len(st.fatigue.history.channels)
            st.fatigue.history.channels = [ch for ch in st.fatigue.history.channels if ch.load_case != lc.name]
            if n != len(st.fatigue.history.channels):
                dropped.append(f"fatigue channels: {n - len(st.fatigue.history.channels)}")
        cr = st.printing.creep
        if lc.name in cr.cases:
            cr.cases = [c for c in cr.cases if c != lc.name]
            dropped.append("long-term creep case")
        if dropped:
            self.log_msg(f"Load case '{lc.name}' and its references deleted ({', '.join(dropped)})")
        self.changed()

    def on_ld_type(self, i):
        t = self.ld_type.currentText()
        th = t == THERMAL_LOAD
        th_case = self.ld_th_src.currentData() if th else None
        uniform = th and th_case is None
        transient = False
        if th_case and self.study is not None:
            c = next((c for c in self.study.thermal.cases if c.name == th_case), None)
            transient = c is not None and c.transient
        frn, frm = t == FRAME_NODE, t == FRAME_MEMBER
        for w in (self.ld_fr_node, self.ld_fr_node_lbl, self.ld_fr_mom_lbl, *self.ld_fr_mom):
            w.setVisible(frn)
        for w in (self.ld_fr_member, self.ld_fr_member_lbl, self.ld_fr_axes, self.ld_fr_axes_lbl, self.ld_fr_var,
                  self.ld_fr_qend_lbl, *self.ld_fr_qend):
            w.setVisible(frm)
        vec = t in (FORCE, "Moment (remote point)", "Force at remote point", "Gravity", BEARING, FRAME_NODE,
                    FRAME_MEMBER)
        val = t in ("Force normal to face", "Pressure") or uniform
        pt = t in ("Moment (remote point)", "Force at remote point")
        pmode = self.ld_p_mode.currentIndex() if t == "Pressure" else -1
        rot = t == ROTATION
        csys_ok = t in (FORCE, "Moment (remote point)", "Force at remote point", "Gravity", BEARING)
        for w in (self.ld_csys, self.ld_csys_btn, self.ld_csys_lbl):
            w.setVisible(csys_ok)
        target = self.load_target() if t == FORCE else None
        for w in (self.ld_target, self.ld_target_lbl):
            w.setVisible(t == FORCE)
        if t == FORCE:
            n = {"faces": len(self.view.selected_faces()), "edges": len(self.view.selected_edges()),
                 "points": len(self.view.selected_points())}
            self.ld_target.setText({"faces": f"faces ({n['faces']}) – uniformly by area",
                                    "edges": f"edges ({n['edges']}) – uniformly by length (Edge filter)",
                                    "points": f"points ({n['points']}) – each F/n (Point filter)"}.get(
                target, "nothing – select faces, edges (Edge filter) or points (Point filter)"))
        for w in (self.ld_p_mode, self.ld_p_mode_lbl):
            w.setVisible(t == "Pressure")
        for w in self.ld_p_grad + [self.ld_p_grad_lbl]:
            w.setVisible(pmode == 1)
        for w in self.ld_p_org + [self.ld_p_org_lbl]:
            w.setVisible(pmode in (1, 2))
        for w in self.ld_p_gdir + [self.ld_p_gdir_lbl, self.ld_p_rho, self.ld_p_rho_lbl]:
            w.setVisible(pmode == 2)
        self.ld_p_org_lbl.setText("Liquid level – point [mm]" if pmode == 2 else "Origin [mm]")
        for w in (self.ld_rot_pt + self.ld_rot_ax + [self.ld_rot_pt_lbl, self.ld_rot_ax_lbl, self.ld_rot_w_lbl,
                                                    self.ld_rot_rpm, self.ld_rot_alpha, self.ld_rot_alpha_lbl,
                                                    self.ld_rot_axis_btn, self.ld_rot_parts,
                                                    self.ld_rot_parts_lbl]):
            w.setVisible(rot)
        for w in self.ld_vec + [self.ld_vec_lbl]:
            w.setVisible(vec)
        for w in (self.ld_val, self.ld_val_lbl):
            w.setVisible(val)
        for w in self.ld_pt + [self.ld_pt_lbl, self.ld_pt_auto, self.ld_coupling, self.ld_coupling_lbl]:
            w.setVisible(pt)
        for w in (self.ld_th_src, self.ld_th_src_lbl):
            w.setVisible(th)
        for w in (self.ld_th_time_w, self.ld_th_time_lbl):
            w.setVisible(th and transient)
        for w in (self.ld_th_parts, self.ld_th_parts_lbl):
            w.setVisible(uniform)
        form_hide_empty_rows(self._ld_form)
        cyl = False
        cs_name = self.ld_csys.currentData()
        if cs_name and self.study is not None and cs_name in self.study.coordinate_systems:
            cyl = self.study.coordinate_systems[cs_name].kind == "cylindrical"
        comp = "F_r, F_θ, F_z" if cyl else "F"
        self.ld_vec_lbl.setText({"Moment (remote point)": "M [N·mm]", "Gravity": "g [mm/s²]",
                                 FRAME_MEMBER: "q [N/mm]"}.get(t, f"{comp} [N]"))
        self.ld_val_lbl.setText(("p [MPa]", "p₀ at origin [MPa]", "p nad hladinou [MPa]")[max(pmode, 0)]
                                if t == "Pressure" else "T [°C]" if th else "F [N]")
        self.ld_val.setToolTip("Positive pressure acts into the material" if t == "Pressure" else
                               "Uniform part temperature (stress from ΔT = T − reference temperature)" if th else
                               "Positive force pushes into the face (against the outward normal)")
        prev = getattr(self, "_ld_prev_type", None)
        self._ld_prev_type = t
        if t != prev:
            # change of type: a residual force from the form must not stay as g [mm/s²] (−200 → g = −200) and vice versa
            # g as force [N]; when an existing load is loaded into the form (fill_load_form) g is not filled in –
            # its vector is written right afterwards
            if t == "Gravity" and not getattr(self, "_ld_filling", False):
                for s, v in zip(self.ld_vec, (0.0, 0.0, -G_STD)):
                    s.setValue(v)
            elif prev == "Gravity":
                for s in self.ld_vec:
                    s.setValue(0.0)
        help_tip(self.ld_type, LOAD_SHORT.get(t, "Load type"),
                 tr(LOAD_TIPS.get(t, "")).replace(".\n", ".<br>").replace("\n", " "))
        tip = VEC_TIPS.get(t, "Force components [N] in the axes of the chosen coordinate system (x, y, z; cylindrical r, θ, z)")
        short = VEC_SHORT.get(t, "{} component of the force [N] in the axes of the selected coordinate system")
        for s, ax in zip(self.ld_vec, "xyz"):
            help_tip(s, short.format(ax), f"{tip}<br>Field: component {ax}")

    def form_csys(self, cylindrical_ok: bool) -> tuple[bool, str | None]:
        """(OK?, system name from the form); cylindrical only for a force."""
        name = self.ld_csys.currentData() or None
        if name and not cylindrical_ok and self.study.coordinate_systems.get(name) is not None \
                and self.study.coordinate_systems[name].kind == "cylindrical":
            msg(self, APP_NAME, "A cylindrical coordinate system is only available for a force (the direction would otherwise depend on the point) – choose a Cartesian system or global axes.")
            return False, None
        return True, name

    def pressure_from_form(self, sel: str) -> Pressure:
        """Pressure from the form: constant / linear / hydrostatic."""
        p0 = self.ld_val.value()
        mode = self.ld_p_mode.currentIndex()
        org = [s.value() for s in self.ld_p_org]
        if mode == 1:
            g = [s.value() for s in self.ld_p_grad]
            return Pressure(sel, p0, g, org) if any(g) else Pressure(sel, p0)
        if mode == 2:
            d = np.array([s.value() for s in self.ld_p_gdir])
            if np.linalg.norm(d) == 0 or self.ld_p_rho.value() <= 0:
                raise ValueError("Hydrostatic pressure: enter a nonzero gravity direction and a positive liquid density.")
            g = d / np.linalg.norm(d) * self.ld_p_rho.value() * 1e-12 * G_STD
            return Pressure(sel, p0, g.tolist(), org, clip=True)
        return Pressure(sel, p0)

    def rotation_from_form(self) -> Rotation:
        ax = np.array([s.value() for s in self.ld_rot_ax])
        if np.linalg.norm(ax) == 0:
            raise ValueError("Rotation: the axis direction must not be zero.")
        w = self.ld_rot_rpm.value() * 2 * math.pi / 60
        if w == 0 and self.ld_rot_alpha.value() == 0:
            raise ValueError("Rotation: enter the speed or angular acceleration.")
        return Rotation([s.value() for s in self.ld_rot_pt], (ax / np.linalg.norm(ax)).tolist(), w,
                        self.ld_rot_alpha.value(), checked(self.ld_rot_parts))

    def rotation_axis_from_selection(self):
        """Rotation axis from the selected circle (filter Edge) or a cylindrical face."""
        from fea_core.geometry import cylinder_axis
        o = d = None
        e = self.view.selected_edge()
        if e is not None and "center" in e and "axis" in e:
            o, d = e["center"], e["axis"]
        else:
            for pi, fi in self.view.selected_faces():
                ax = cylinder_axis(self.view.part_faces[pi][fi])
                if ax is not None:
                    o, d = ax[0], ax[1]
                    break
        if o is None:
            msg(self, APP_NAME, "Select a cylindrical face (shaft, hole) or circle (Edge filter).")
            return
        for sp, v in zip(self.ld_rot_pt, o):
            sp.setValue(float(v) + 0.0)
        for sp, v in zip(self.ld_rot_ax, d):
            sp.setValue(round(float(v), 9) + 0.0)

    def edit_csys(self):
        if self.study is None:
            return
        from .csys_dialog import CsysDialog
        CsysDialog(self).exec()
        self.refresh_csys_combo()

    def refresh_csys_combo(self):
        cur = self.ld_csys.currentData()
        self.ld_csys.blockSignals(True)
        self.ld_csys.clear()
        self.ld_csys.addItem("global axes (x, y, z)", None)
        if self.study is not None:
            for k, c in self.study.coordinate_systems.items():
                self.ld_csys.addItem(f"{k} ({'cylindrical r, θ, z' if c.kind == 'cylindrical' else 'Cartesian'})", k)
        i = self.ld_csys.findData(cur) if cur else 0
        self.ld_csys.setCurrentIndex(max(i, 0))
        self.ld_csys.blockSignals(False)
        self.ortho_box.refresh_csys()               # axes of the orthotropic material (Materials step)

    def add_load(self):
        lc = self.current_lc()
        if lc is None:
            msg(self, APP_NAME, "Create a load case first.")
            return
        t = self.ld_type.currentText()
        vec = [s.value() for s in self.ld_vec]
        ok, csys = self.form_csys(t == FORCE)
        if not ok:
            return
        if t == "Gravity":
            lc.loads.append(Gravity(vec, csys))
            self.changed()
            return
        if t in (FRAME_NODE, FRAME_MEMBER):
            l = self.frame_load_from_form()
            if l is not None:
                lc.loads.append(l)
                self.changed()
            return
        if t == THERMAL_LOAD:
            l = self.thermal_load_from_form()
            if l is not None:
                lc.loads.append(l)
                self.changed()
            return
        if t == ROTATION:
            try:
                l = self.rotation_from_form()
            except ValueError as e:
                msg(self, APP_NAME, str(e))
                return
            lc.loads = [x for x in lc.loads if not isinstance(x, Rotation)] + [l]   # one rotation per state
            self.changed()
            return
        if t == FORCE and self.load_target() in ("edges", "points"):
            target = self.load_target()
            name = self.new_entity_selection("Force on edge" if target == "edges" else "Force at point", target)
            lc.loads.append(Force(name, vec, csys))
            self.view.clear_selection()
            self.changed()
            if any(isinstance(l, Force) and self.study.selections.get(l.selection) is not None
                   and self.study.selections[l.selection].entity in ("edges", "points") for l in lc.loads):
                self.log_msg("A force on an edge / at a point is singular in a solid model – stress near it is not valid (grows with mesh refinement); evaluate away from the point of application.")
            return
        sels = self.need_faces()
        if not sels:
            return
        if t == FORCE:
            name = self.new_selection("Force", sels)
            lc.loads.append(Force(name, vec, csys))
        elif t == "Force normal to face":
            n = self.normal_force_dir(sels)
            if n is None:
                return
            name = self.new_selection("Force", sels)
            lc.loads.append(Force(name, (-self.ld_val.value() * n).tolist()))
        elif t == "Pressure":
            try:
                self.pressure_from_form("")
            except ValueError as e:
                msg(self, APP_NAME, str(e))
                return
            name = self.new_selection("Pressure", sels)
            lc.loads.append(self.pressure_from_form(name))
        elif t == BEARING:
            if not self.cylinder_faces(sels):
                return
            name = self.new_selection("Bearing load", sels)
            lc.loads.append(BearingLoad(name, vec, csys))
        else:
            name = self.new_selection("Moment" if t.startswith("Moment") else "Remote force", sels)
            lc.loads.append(self.remote_load_from_form(name, sels, vec, csys))
        self.view.clear_selection()
        self.changed()

    def remote_load_from_form(self, name, sels, vec, csys) -> RemoteLoad:
        """Remote force / moment from the form; automatic point = centroid of the faces (weighted by area)."""
        if self.ld_pt_auto.isChecked():
            w = np.array([s.area for s in sels])
            pt = ((np.array([s.centroid for s in sels]) * w[:, None]).sum(axis=0) / w.sum()).tolist()
        else:
            pt = [s.value() for s in self.ld_pt]
        coupling = "rigid" if self.ld_coupling.currentIndex() == 1 else "distributing"
        what = "moment" if self.ld_type.currentText().startswith("Moment") else "force"
        return RemoteLoad(name, pt, coupling=coupling, csys=csys, **{what: vec})

    def frame_load_from_form(self):
        """Frame load from the form (force / moment at a node, distributed load of a member)."""
        from fea_core.beams.model import FrameMemberLoad, FrameNodeLoad
        vec = [s.value() for s in self.ld_vec]
        if self.ld_type.currentText() == FRAME_NODE:
            node = self.ld_fr_node.currentText()
            if not node:
                msg(self, APP_NAME, "The study has no frame nodes (Members step).")
                return None
            mom = [s.value() for s in self.ld_fr_mom]
            if not any(vec) and not any(mom):
                msg(self, APP_NAME, "Enter a force or moment.")
                return None
            return FrameNodeLoad(node, vec, mom)
        member = self.ld_fr_member.currentText()
        if not member:
            msg(self, APP_NAME, "The study has no members (Members step).")
            return None
        q_end = [s.value() for s in self.ld_fr_qend] if self.ld_fr_var.isChecked() else None
        if not any(vec) and not (q_end and any(q_end)):
            msg(self, APP_NAME, "Enter the distributed load q [N/mm].")
            return None
        return FrameMemberLoad(member, vec, q_end, self.ld_fr_axes.currentData())

    def refresh_frame_load_form(self):
        fr = self.study.frame
        for cb, names in ((self.ld_fr_node, list(fr.nodes)), (self.ld_fr_member, list(fr.members))):
            cur = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(names)
            if cur in names:
                cb.setCurrentText(cur)
            cb.blockSignals(False)

    def thermal_load_from_form(self) -> ThermalLoad | None:
        """Thermal load from the form: field of a thermal state (time for a transient one), or uniform temperature of the parts."""
        case = self.ld_th_src.currentData()
        if case:
            c = next((c for c in self.study.thermal.cases if c.name == case), None)
            t = None
            if c is not None and c.transient and not self.ld_th_end.isChecked():
                t = self.ld_th_time.value()
                if t > c.time:
                    msg(self, APP_NAME, f"Time {fmt(t)} s is past the end of the transient case ({fmt(c.time)} s).")
                    return None
            return ThermalLoad(case=case, time=t)
        return ThermalLoad(value=self.ld_val.value(), parts=checked(self.ld_th_parts))

    def refresh_thermal_load_form(self):
        """Menu of thermal states and the part list for the thermal load (after a change of the study)."""
        cur = self.ld_th_src.currentData()
        cur_text = self.ld_th_src.currentText()
        self.ld_th_src.blockSignals(True)
        self.ld_th_src.clear()
        for c in self.study.thermal.cases:
            self.ld_th_src.addItem(f"thermal case: {c.name}", c.name)
        self.ld_th_src.addItem("uniform part temperature", None)
        i = self.ld_th_src.findData(cur) if cur else self.ld_th_src.findText(cur_text)
        self.ld_th_src.setCurrentIndex(max(i, 0))
        self.ld_th_src.blockSignals(False)
        for lst in (self.ld_th_parts, self.ld_rot_parts):          # model parts, the tick stays
            on = set(checked(lst))
            lst.clear()
            for p in self.parts:
                check_item(lst, p.name, p.name in on)
        self.refresh_csys_combo()
        self.refresh_frame_load_form()
        self.on_ld_type(0)

    def on_load_row(self, row):
        lc = self.current_lc()
        if lc and 0 <= row < len(lc.loads):
            l = lc.loads[row]
            sel = getattr(l, "selection", None)
            self.view.set_selected_faces(self.sel_faces.get(sel, []) if sel else [])
            self.view.set_selected_edges(self.sel_edges.get(sel, []) if sel else [])
            self.view.set_selected_points(self.sel_points.get(sel, []) if sel else [])
            self.fill_load_form(l)
            self.update_status()

    def fill_load_form(self, l):
        self._ld_filling = True                  # change of type from an existing load – do not overwrite its vector
        try:
            self._fill_load_form(l)
        finally:
            self._ld_filling = False

    def _fill_load_form(self, l):
        def setvec(v, spins=None):
            for s, x in zip(spins or self.ld_vec, v):
                s.setValue(float(x))
        if hasattr(l, "csys"):
            self.ld_csys.setCurrentIndex(max(self.ld_csys.findData(l.csys), 0) if l.csys else 0)
        k = getattr(l, "kind", "")
        if k == "frame_node":
            self.ld_type.setCurrentText(FRAME_NODE)
            self.ld_fr_node.setCurrentText(l.node)
            setvec(l.force)
            setvec(l.moment, self.ld_fr_mom)
            return
        if k == "frame_member":
            self.ld_type.setCurrentText(FRAME_MEMBER)
            self.ld_fr_member.setCurrentText(l.member)
            setvec(l.q)
            self.ld_fr_var.setChecked(l.q_end is not None)
            setvec(l.q_end or [0, 0, 0], self.ld_fr_qend)
            self.ld_fr_axes.setCurrentIndex(max(0, self.ld_fr_axes.findData(l.axes)))
            return
        if isinstance(l, Force):
            self.ld_type.setCurrentText(FORCE)
            setvec(l.vector)
        elif isinstance(l, Pressure):
            self.ld_type.setCurrentText("Pressure")
            self.ld_val.setValue(l.value)
            if l.variable and l.clip:
                g = np.asarray(l.gradient, float)
                self.ld_p_mode.setCurrentIndex(2)
                self.ld_p_rho.setValue(float(np.linalg.norm(g)) / (1e-12 * G_STD))
                setvec(g / np.linalg.norm(g), self.ld_p_gdir)
            elif l.variable:
                self.ld_p_mode.setCurrentIndex(1)
                setvec(l.gradient, self.ld_p_grad)
            else:
                self.ld_p_mode.setCurrentIndex(0)
            if l.variable:
                setvec(l.origin or [0, 0, 0], self.ld_p_org)
        elif isinstance(l, Rotation):
            self.ld_type.setCurrentText(ROTATION)
            setvec(l.point, self.ld_rot_pt)
            setvec(l.axis, self.ld_rot_ax)
            self.ld_rot_rpm.setValue(l.omega * 60 / (2 * math.pi))
            self.ld_rot_alpha.setValue(l.alpha)
            set_checks(self.ld_rot_parts, l.parts)
        elif isinstance(l, BearingLoad):
            self.ld_type.setCurrentText(BEARING)
            setvec(l.vector)
        elif isinstance(l, Gravity):
            self.ld_type.setCurrentText("Gravity")
            setvec(l.vector)
        elif isinstance(l, ThermalLoad):
            self.ld_type.setCurrentText(THERMAL_LOAD)
            i = self.ld_th_src.findData(l.case) if l.case else self.ld_th_src.count() - 1
            self.ld_th_src.setCurrentIndex(max(i, 0))
            self.ld_th_end.setChecked(l.time is None)
            self.ld_th_time.setValue(l.time or 0.0)
            if l.value is not None:
                self.ld_val.setValue(l.value)
            set_checks(self.ld_th_parts, l.parts)
            self.on_ld_type(0)
        elif isinstance(l, RemoteLoad):
            moment = any(l.moment)
            self.ld_type.setCurrentText("Moment (remote point)" if moment else "Force at remote point")
            setvec(l.moment if moment else l.force)
            self.ld_pt_auto.setChecked(False)
            for s, x in zip(self.ld_pt, l.point):
                s.setValue(float(x))
            self.ld_coupling.setCurrentIndex(1 if l.coupling == "rigid" else 0)

    def update_load(self):
        lc = self.current_lc()
        row = self.load_list.currentRow()
        if lc is None or not (0 <= row < len(lc.loads)):
            msg(self, APP_NAME, "Select a load in the list.")
            return
        l = lc.loads[row]
        t = self.ld_type.currentText()
        vec = [s.value() for s in self.ld_vec]
        sel = getattr(l, "selection", None)
        ok, csys = self.form_csys(t == FORCE)
        if not ok:
            return
        entity = self.study.selections[sel].entity if sel and sel in self.study.selections else "faces"
        if t == "Gravity":
            new = Gravity(vec, csys)
        elif t in (FRAME_NODE, FRAME_MEMBER):
            new = self.frame_load_from_form()
            if new is None:
                return
        elif t == THERMAL_LOAD:
            new = self.thermal_load_from_form()
            if new is None:
                return
        elif t == ROTATION:
            try:
                new = self.rotation_from_form()
            except ValueError as e:
                msg(self, APP_NAME, str(e))
                return
        elif sel is None:
            msg(self, APP_NAME, "Gravity, rotation and temperature can only be changed to a load without faces.")
            return
        elif entity in ("edges", "points") and t != FORCE:
            msg(self, APP_NAME, "Only a force (components) can be applied to an edge / point.")
            return
        elif t == FORCE:
            new = Force(sel, vec, csys)
        elif t == "Force normal to face":
            sels = self.study.selections[sel].faces
            n = self.normal_force_dir(sels)
            if n is None:
                return
            new = Force(sel, (-self.ld_val.value() * n).tolist())
        elif t == "Pressure":
            try:
                new = self.pressure_from_form(sel)
            except ValueError as e:
                msg(self, APP_NAME, str(e))
                return
        elif t == BEARING:
            if not self.cylinder_faces(self.study.selections[sel].faces):
                return
            new = BearingLoad(sel, vec, csys)
        else:
            new = self.remote_load_from_form(sel, self.study.selections[sel].faces, vec, csys)
        if sel and getattr(new, "selection", None) != sel:
            # a load on faces converted to gravity / rotation / temperature: its selection is removed, unless something else uses it
            # (constraint, other load, thermal condition, mesh refinement)
            self._drop_selection(sel)
        lc.loads[row] = new
        self.changed()
        self.load_list.setCurrentRow(row)

    # ================================================================== shared list edits
    def _drop_selection(self, name):
        if not name:
            return
        used = [c.selection for c in self.study.constraints]
        used += [getattr(l, "selection", None) for lc in self.study.load_cases for l in lc.loads]
        used += [getattr(bc, "selection", None) for c in self.study.thermal.cases for bc in c.loads]
        used += [r.get("selection") for r in self.study.mesh.local]
        used += [x for c in self.study.contacts for x in (c.a, c.b)]                 # F8: kontakty, hmota, sondy
        used += [m.selection for m in self.study.masses]
        used += [p.selection for p in self.study.dynamics.probes]
        used += [mo.target for s in self.study.analysis.sequence for mo in s.prescribed]
        mech = self.study.mechanism                                                    # F8: mechanismus
        used += [x for j in mech.joints for x in (j.selection_a, j.selection_b)]
        used += [x for f in mech.forces for x in (f.selection_a, f.selection_b)]
        used += [ld.selection for ld in mech.loads]
        if used.count(name) <= 1:
            self.study.selections.pop(name, None)
            self.sel_faces.pop(name, None)
            self.sel_edges.pop(name, None)
            self.sel_points.pop(name, None)

    def _items(self, kind):
        if kind == "constraint":
            return self.study.constraints
        if kind == "load":
            lc = self.current_lc()
            return lc.loads if lc else []
        if kind == "thermal":
            c = self.thermal_panel.current_case()
            return c.loads if c else []
        return self.study.mesh.local

    def delete_item(self, lw, kind):
        if self.study is None:
            return
        row = lw.currentRow()
        items = self._items(kind)
        if not (0 <= row < len(items)):
            return
        it = items[row]
        sel = it.get("selection") if isinstance(it, dict) else getattr(it, "selection", None)
        self._drop_selection(sel)
        items.pop(row)
        self.changed(invalidate_mesh=(kind == "refine"))

    def update_faces_of(self, lw, kind):
        if self.study is None:
            return
        row = lw.currentRow()
        items = self._items(kind)
        if not (0 <= row < len(items)):
            msg(self, APP_NAME, "Select an item in the list.")
            return
        it = items[row]
        name = it.get("selection") if isinstance(it, dict) else getattr(it, "selection", None)
        if not name:
            msg(self, APP_NAME, "The selected item has no faces (gravity, rotation, temperature, heat generation in a part, sphere refinement).")
            return
        entity = self.study.selections[name].entity if name in self.study.selections else "faces"
        if entity in ("edges", "points"):                 # force on an edge / to a point: edges / points from the selection
            new = self.selected_edge_selectors() if entity == "edges" else self.selected_point_selectors()
            if not new:
                msg(self, APP_NAME, "Select edges in 3D (Edge filter)." if entity == "edges" else
                    "Select points in 3D (Point filter).")
                return
            setattr(self.study.selections[name], entity, new)
            if entity == "edges":
                self.sel_edges[name] = self.view.selected_edges()
            else:
                self.sel_points[name] = self.view.selected_points()
            self.changed(invalidate_mesh=True)
            return
        sels = self.need_faces()
        if not sels:
            return
        self.study.selections[name].faces = sels
        self.sel_faces[name] = self.view.selected_faces()
        self.changed(invalidate_mesh=True)

    # ================================================================== mesh
    def on_mesh_settings(self):
        if self.study is None or getattr(self, "_filling", False):
            return
        self.study.mesh.size = self.ms_size.value()
        self.study.mesh.min_size = self.ms_min.value() or None
        self.study.mesh.curvature_points = self.ms_curv.value()
        self.changed(invalidate_mesh=True)

    def add_refinement(self):
        if getattr(self, "_mesh_view", False) and not self.view.selected_faces():
            msg(self, APP_NAME, "The mesh is displayed in 3D – turn off “Show mesh”, select part faces and then click Refine selection.")
            return
        sels = self.need_faces()
        if sels:
            name = self.new_selection("Refinement", sels)
            self.study.mesh.local.append({"selection": name, "size": self.ref_size.value()})
            self.view.set_selected_faces([])
            self.changed(invalidate_mesh=True)

    def on_ref_row(self, row):
        if 0 <= row < len(self.study.mesh.local):
            r = self.study.mesh.local[row]
            if r.get("selection"):
                self.view.set_selected_faces(self.sel_faces.get(r["selection"], []))
            elif r.get("center") is not None:
                for s, v in zip(self.ball_pt, r["center"]):
                    s.setValue(float(v))
                self.ball_r.setValue(float(r.get("radius", 1.0)))
                self.ball_h.setValue(float(r.get("size", 1.0)))

    @staticmethod
    def describe_refinement(r) -> str:
        if r.get("selection"):
            return f"{r['selection']}: {fmt(r['size'])} mm"
        c = r.get("center") or [0, 0, 0]
        return (f"Sphere ({'; '.join((fmt(float(x)) for x in c))}) R {fmt(float(r.get('radius', 0)))} mm: {fmt(float(r['size']))} mm")

    def ball_point_from_selection(self):
        """Sphere centre from the 3D selection: point (filter Point) > centre of an edge / circle > centroid of the faces."""
        p = None
        pts = self.view.selected_points()
        e = self.view.selected_edge()
        if pts:
            p = pts[-1][1]
        elif e is not None:
            p = e["point"]
        elif self.view.selected_edges():
            p = self.edge_prop(*self.view.selected_edges()[-1])["point"]
        elif self.view.selected_faces():
            sels = self.selected_selectors()
            w = np.array([s.area for s in sels])
            p = (np.array([s.centroid for s in sels]) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-30)
        if p is None:
            msg(self, APP_NAME, "Select a point (Point filter V), an edge (Edge filter E) or faces in 3D.")
            return
        for s, v in zip(self.ball_pt, p):
            s.setValue(float(v) + 0.0)

    def add_ball_refinement(self):
        if self.study is None:
            return
        if self.ball_h.value() >= self.study.mesh.size:
            msg(self, APP_NAME, "The element size in the sphere must be smaller than the global element size.")
            return
        self.study.mesh.local.append({"center": [s.value() for s in self.ball_pt], "radius": self.ball_r.value(),
                                      "size": self.ball_h.value()})
        self.changed(invalidate_mesh=True)

    # meshing inputs (Gmsh); the geometry is checked by _load_model, materials and the 3D printing settings do not change the mesh – the assignment
    # of shells and infill is recomputed on the finished mesh (fea_core.printing.region.ensure)
    MESH_KEYS = ("mesh", "selections", "bolts", "welds", "joints", "ignored_parts")

    def study_signature(self, keys=None) -> str:
        """Signature of the study (without the fatigue settings) – the result of a thread started on another study is discarded."""
        if keys is self.MESH_KEYS:                    # meshing inputs – shared with MCP (fea_core.signature)
            from fea_core.signature import mesh_inputs
            return json.dumps(mesh_inputs(self.study, geometry=False), sort_keys=True, default=str)
        d = self.study.to_dict()
        d.pop("fatigue", None)
        if keys:
            d = {k: d.get(k) for k in keys}
        return json.dumps(d, sort_keys=True, default=str)

    def valid_mesh(self):
        """Mesh that can be taken over into the calculation (the meshing inputs have not changed since it was created), otherwise None."""
        if self.mesh is not None and self._mesh_sig is not None and self._mesh_sig == self.study_signature(self.MESH_KEYS):
            return self.mesh
        return None

    def make_mesh(self):
        if self.study is None or self._thread is not None:
            return
        self.btn_mesh.setEnabled(False)
        self.btn_mesh_cancel.setEnabled(True)
        self.mesh_info.setText("Meshing…")
        self.status_progress.start("Meshing (Gmsh)", 0.95)
        self.log_msg("Meshing (Gmsh)…")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        self._thread_sig = self.study_signature(self.MESH_KEYS)
        t = MeshThread(Study.from_dict(self.study.to_dict()), self.solid_parts(),
                       self._workdir / f"mesh_{int(time.time() * 1000)}", self)
        t.done.connect(self.on_mesh_done)
        t.failed.connect(self.on_mesh_failed)
        t.finished.connect(self._thread_finished)
        self._thread = t
        t.start()

    def cancel_mesh(self):
        if isinstance(self._thread, MeshThread):
            self._thread.cancel.set()
            n = kill_mesh_workers()
            self.log_msg("Cancelling meshing…" + (f" (Gmsh processes terminated: {n})" if n else ""))

    def on_mesh_done(self, mesh):
        t = self._thread                         # the done signal arrives before finished
        if isinstance(t, MeshThread) and t.cancel.is_set():
            self.log_msg("Meshing canceled.")
            self.mesh_progress_finish("cancelled", "Meshing canceled")
            self.refresh_all()
            return
        if self.study is None or self.study_signature(self.MESH_KEYS) != self._thread_sig:
            self.log_msg("The study changed during meshing – mesh discarded, generate it again.")
            self.mesh_progress_finish("cancelled", "Mesh discarded – study changed")
            self.refresh_all()
            return
        self.mesh_progress_finish("done", "Mesh done")
        self.mesh = mesh
        self._mesh_sig = self._thread_sig
        r = mesh.report
        self.log_msg(f"Mesh: {fmt(r['nodes'])} nodes, {fmt(r['elements'])} elements, min. SJ {r['min_scaled_jacobian']:.3f}")
        for w in r.get("warnings", []):
            self.log_msg(f"Warning: {w}")
        for t in r.get("ties", []):
            self.log_msg(f"*TIE connection: part {t['part']}, contact {fmt(t['area_mm2'])} mm²")
        self.mesh_messages(mesh)
        self.refresh_all()
        if STAGES[self.stage] == "Mesh":
            self.ms_show.blockSignals(True)          # a new mesh is always shown
            self.ms_show.setChecked(True)
            self.ms_show.blockSignals(False)
            self.set_stage(self.stage)

    def on_mesh_failed(self, text):
        t = self._thread                         # the done signal arrives before finished
        if isinstance(t, MeshThread) and t.cancel.is_set():
            self.log_msg("Meshing canceled.")
            self.mesh_info.setText("Meshing canceled.")
            self.mesh_progress_finish("cancelled", "Meshing canceled")
            return
        self.mesh_progress_finish("error", "Meshing failed – see the log for the reason")
        self.log_msg(f"Mesh ERROR: {text}")
        self.messages.set_source("mesh", [("error", f"Meshing failed: {text}")])
        self.dock_messages.raise_()
        msg(self, APP_NAME, f"Meshing failed:\n{text}", QMessageBox.Critical)
        self.refresh_all()

    def f8_messages(self, res):
        """Messages of the nonlinear calculation and dynamics (F8): partial result with advice, list of tasks, checks of the modal
        basis. Texts that another source already reports (result warnings) are left out (unique).
        """
        items = []
        nl = (getattr(res, "info", None) or {}).get("nonlinear") or {}
        if nl:
            items.append(("info", f"Nonlinear analysis: {('states separately (branches)' if nl.get('mode') == 'branches' else 'step sequence')}, CalculiX jobs {len(nl.get('jobs') or [])}"))
            for job, p in (nl.get("partial") or {}).items():
                t = p.get("last_step_time")
                items.append(("warning", f"Nonlinear calculation ({job}) did not converge – partial result: the last converted step {p.get('last_step')} of {p.get('of_steps')}"
                                         + (f", step time {fmt(float(t), 4)}" if t is not None else "")
                                         + f", increment cut-back {p.get('cutbacks', 0)}×"))
                for e in p.get("errors") or []:
                    items.append(("warning", f"CalculiX ({job}): {e}"))
            if nl.get("partial"):
                from fea_core.contacts.results import ADVICE
                items.append(("info", f"Nonlinear analysis: {ADVICE}"))
        d = getattr(res, "dynamics", None)
        if d is not None:
            items.append(("info", f"Dynamics: {d.K} modes {fmt(float(d.freq[0]), 5)}–{fmt(float(d.freq[-1]), 5)} Hz, model mass {fmt(1000 * d.model_mass)} kg" if d.K else "Dynamics: no modes"))
            items += [("warning", w) for w in d.warnings]
            items.append(("warning", "Dynamics: damping is an uncertain input (sensitivity ±50 %) – verify resonance results by measuring the damping"))
        self.messages.set_source(F8_SOURCE, items, "Results", unique=True)

    def mesh_messages(self, mesh):
        """Messages of the "mesh" source: mesh size, meshing warnings, *TIE connections."""
        r = mesh.report
        items = [("info", f"Mesh: {fmt(r.get('nodes', len(mesh.nodes)))} nodes, {fmt(r.get('elements', len(mesh.tets)))} elements, min. scaled Jacobian {fmt(r.get('min_scaled_jacobian'), 3)}")]
        items += [("warning", w) for w in r.get("warnings", []) if w not in r.get("study_warnings", [])]
        items += [("info", f"*TIE connection: part {t['part']} – contact {fmt(t['area_mm2'])} mm², tolerance {fmt(t['tolerance_mm'], 3)} mm") for t in r.get("ties", [])]
        self.messages.set_source("mesh", items)

    def _thread_finished(self):
        QApplication.restoreOverrideCursor()
        self._thread = None
        self.btn_mesh.setEnabled(True)
        self.btn_mesh_cancel.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_stop_keep.setEnabled(False)
        if getattr(self, "_stop_timer", None) is not None:
            self._stop_timer.stop()
        self.refresh_ready()

    def show_mesh(self):
        if self.mesh is None:
            return
        tri, _ = self.mesh.boundary_faces()
        if getattr(self, "_mesh_color", "plain") == "quality":
            q = element_quality_nodes(self.mesh)
            edges = np.linspace(0.0, 1.0, 11)
            self.view.show_contour(self.mesh.nodes, tri, q, edges, tr("Element quality (1 = regular)"), "%.2f",
                                   edges_on=True, reverse=True)
            return
        self.view.show_mesh(self.mesh.nodes, tri)

    # ================================================================== calculation
    def on_solver(self, name):
        if self.study and self.study.solver.solver != name:
            self.study.solver.solver = name
            self.changed()

    def _backend_badge(self, _=0):
        """Badge of the collapsed Solver section: the chosen solver."""
        from fea_core.solvers import TITLES
        self.solver_section.set_badge(TITLES.get(self.backend_combo.currentData(), ""))

    def on_backend(self, _):
        key = self.backend_combo.currentData()
        if self.study and key and self.study.solver.backend != key:
            self.study.solver.backend = key
            self.changed()

    def study_for_solve(self) -> Study:
        """Copy of the study for the calculation. With thermal states the empty load states are left out (a new study has an
        empty "Load state 1" – a study with heat only then does not need constraints).
        """
        st = Study.from_dict(self.study.to_dict())
        if st.thermal.cases or st.modal.enabled or st.dynamics.enabled:   # also a study with only frequencies / dynamics
            st.load_cases = [lc for lc in st.load_cases if lc.loads]
        return st

    def no_static(self) -> bool:
        """Study without statics (no non-empty load state), but with heat or natural frequencies – loads
        are not needed; constraints only for statics (natural frequencies without constraints = free body, only a warning).
        """
        st = self.study
        return st is not None and (bool(st.thermal.cases) or st.modal.enabled or st.dynamics.enabled) and \
            not any(lc.loads for lc in st.load_cases)

    def check(self) -> tuple[list[str], list[str]]:
        """Readiness check of the study as it will be calculated (see study_for_solve)."""
        st = self.study_for_solve()
        errors, warnings = check_study(st, [p.name for p in self.parts])
        if st.shell.enabled and not st.twod.enabled and not self.frame_only():   # shell model (fea_core.shell)
            from fea_core.shell.run import check_study as check_shell
            errors = [e for e in errors if "supports are currently faces only" not in e]   # shells: constraints also on edges
            errors = errors + [e for e in check_shell(st) if e not in errors]
            warnings = warnings + ["Shell model: parts are analyzed as mid-surfaces with thickness (S6) – results “Shells: …”"]
        return errors, warnings

    def rigid_errors(self) -> list[str]:
        """Rigid-motion check errors that block the calculation (see rigid_check)."""
        return self.rigid_check()[0]

    def rigid_check(self) -> tuple[list[str], list[str]]:
        """Rigid-motion check (fea_core.mcp.tools_mesh.rigid_body_check): constraints must prevent 3 displacements and
        3 rotations. A study with heat only and no constraints is not checked (missing constraints are reported by check_study).
        With a mesh exactly from the constraint nodes, without a mesh from the points of the CAD faces; result cached by constraints and selections.
        Returns (errors, warnings): an approximate result (a constraint that the geometric check does not know exactly – "note")
        is only a warning and does not block the calculation – as MCP fea_check_readiness.
        """
        st = self.study
        if st is None or not st.constraints or self.no_static() or not self.parts or st.shell.enabled or \
                st.twod.enabled:                  # 2D / shells: constraints are converted in the core (own check)
            return [], []
        sig = self.study_signature(("constraints", "selections")) + f"|{id(self.mesh)}|{id(self.parts)}"
        if self._rigid is not None and self._rigid[0] == sig:
            return list(self._rigid[1][0]), list(self._rigid[1][1])
        errs: list[str] = []
        warns: list[str] = []
        try:
            from fea_core.mcp.session import Session
            from fea_core.mcp.tools_mesh import rigid_body_check
            if self._rb_sess is None or self._rb_sess._parts is not self.parts:
                self._rb_sess = Session(self._workdir / "rigid")
                self._rb_sess._parts = self.parts          # GUI parts (the cache of face properties stays)
            sess = self._rb_sess
            sess.study = Study.from_dict(st.to_dict())
            sess.mesh = sess.mesh_sig = None
            m = self.mesh
            if m is not None and all(c.selection in m.surfaces for c in st.constraints):
                sess.mesh = m
                sess.mesh_sig = sess.mesh_signature()
            r = rigid_body_check(sess)
            if not r.get("ok", True):
                text = "The supports do not prevent a rigid-body motion of the model – free: " + "; ".join(r["free_rigid_modes"])
                if r.get("note"):
                    warns.append(text + f" ({r['note']} The calculation can be started; if the problem is really free, the solver ends with an error.)")
                else:
                    errs.append(text)
        except Exception as e:  # noqa: BLE001 – the auxiliary check must not block the calculation
            self.log_msg(f"The rigid-body motion check failed: {e}")
        self._rigid = (sig, (errs, warns))
        return list(errs), list(warns)

    def refresh_ready(self, check_result=None):
        """check_result = (errors, warnings) of self.check() already computed for this study state (refresh_all) – not computed again."""
        self.ready_list.clear()
        if self.study is None:
            if getattr(self, "messages", None) is not None:
                self.messages.set_source("study check", [])
            return
        errors, warnings = check_result if check_result is not None else self.check()
        if STAGES[getattr(self, "stage", 0)] == "Solve":   # rigid-motion check only in the Solve step (cache)
            rb_err, rb_warn = self.rigid_check()
            errors, warnings = errors + rb_err, warnings + rb_warn
        if self.study.thermal.cases or self.study.modal.enabled or self.study.dynamics.enabled:
            empty = [lc.name for lc in self.study.load_cases if not lc.loads]
            if empty:
                warnings = warnings + [f"An empty load case is skipped during the analysis: {', '.join(empty)}"]
        for e in errors:
            QListWidgetItem(dot_icon(RED), e, self.ready_list)
        for w in warnings:
            QListWidgetItem(dot_icon(ORANGE), w, self.ready_list)
        if not errors:
            QListWidgetItem(dot_icon(GREEN), "The study is ready to run", self.ready_list)
        self.btn_run.setEnabled(not errors and self._thread is None)
        if getattr(self, "messages", None) is not None:     # message table: study check
            self.messages.set_source("study check", [("error", e) for e in errors]
                                     + [("warning", w) for w in warnings]
                                     + ([] if errors else [("info", "The study is ready to run")]))

    def run_solve(self):
        if self.study is None or self._thread is not None:
            return
        errors, _ = self.check()
        rb_err, rb_warn = self.rigid_check()
        errors += rb_err
        if errors:
            msg(self, APP_NAME, "The study is not ready:\n\n" + "\n".join(errors))
            return
        for w in rb_warn:
            self.log_msg(f"Warning: {w}")
        self.study.geometry = self.geometry_str()
        self._thread_sig = self.study_signature()
        self._thread_mesh_sig = self.study_signature(self.MESH_KEYS)
        wd = self._workdir / f"run_{int(time.time() * 1000)}"
        self.log_msg(f"Analysis in folder {wd}")
        mesh = self.valid_mesh()
        t = SolveThread(self.study_for_solve(), wd, self, mesh=mesh,
                        parts=self.solid_parts() if mesh is not None else None)
        t.log.connect(self.log_msg)
        t.solver_line.connect(lambda s: self.log.appendPlainText(s) if s.strip() else None)
        t.progress.connect(self.progress_report)
        t.done.connect(self.on_solve_done)
        t.failed.connect(self.on_solve_failed)
        t.finished.connect(self._thread_finished)
        self._thread = t
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_stop_keep.setEnabled(self.study.is_nonlinear())
        self._stop_keep = False
        self.solve_info.setText("Computing…")
        QApplication.setOverrideCursor(Qt.BusyCursor)
        self._t_solve = time.time()
        self.progress_start("Study check")
        t.start()

    # ------------------------------------------------------------------ green progress bar (Solve panel + status bar)
    def progress_bars(self):
        return [b for b in (getattr(self, "solve_progress", None), getattr(self, "status_progress", None)) if b]

    def progress_start(self, text: str, end: float = 0.05):
        for b in self.progress_bars():
            b.start(text, end)

    def progress_report(self, value: float, end: float, text: str):
        for b in self.progress_bars():
            b.report(value, end, text)

    def progress_finish(self, state: str, text: str):
        """End of a job: the bar in the Solve panel keeps the result, the status bar clears it after a while."""
        for b in self.progress_bars():
            b.finish(state, text)
        QTimer.singleShot(6000, lambda: self.status_progress.hide() if self._thread is None else None)

    def mesh_progress_finish(self, state: str, text: str):
        self.status_progress.finish(state, text)
        QTimer.singleShot(6000, lambda: self.status_progress.hide() if self._thread is None else None)

    def cancel_solve(self):
        if isinstance(self._thread, SolveThread):
            self._thread.cancel.set()
            kill_mesh_workers()                  # the calculation may be meshing right now (Gmsh cancel is not checked)
            self.log_msg("Canceling analysis…")

    def stop_keep_solve(self):
        """Stops the nonlinear calculation and keeps the last converted increment (the core returns it as a partial
        result – StepResult.partial). The ccx processes are terminated repeatedly until the calculation thread ends (jobs
        waiting in the parallel queue would otherwise start).
        """
        if not isinstance(self._thread, SolveThread):
            return
        from .workers import kill_ccx_processes
        self._stop_keep = True
        self.btn_stop_keep.setEnabled(False)
        self.log_msg("Stopping the nonlinear analysis – the last converged increment will be kept…")
        if getattr(self, "_stop_timer", None) is None:
            self._stop_timer = QTimer(self)
            self._stop_timer.setInterval(300)
            self._stop_timer.timeout.connect(lambda: kill_ccx_processes() if self._thread is not None else None)
        kill_ccx_processes()
        self._stop_timer.start()

    def on_solve_done(self, res):
        if self.study is None or self.study_signature() != self._thread_sig:
            self.progress_finish("cancelled", "Result discarded – study changed")
            self.solve_info.setText("The study changed during the analysis – result discarded.")
            self.log_msg("The study changed during the analysis – results discarded, run the analysis again.")
            msg(self, APP_NAME, "The study changed during the analysis – the results would not match it and were discarded.\nRun the analysis again.")
            return
        self.results = res
        self.mesh = res.mesh
        self._mesh_sig = self._thread_mesh_sig
        self.dirty = True
        if self.study.cad_link is not None:           # link with the CAD: results summary into the current iteration
            try:
                from fea_core.cad_link.history import record
                record(self.study, res, self.solid_parts())
            except Exception as e:  # noqa: BLE001 – the iteration summary does not discard the result
                self.log_msg(f"The iteration summary cannot be written: {e}")
        dt = time.time() - self._t_solve
        res._gui_solve_s = dt                          # calculation time for the status bar (not saved)
        self.progress_finish("done", decimal(f"Done in {dt:.1f} s"))
        self.solve_info.setText(f"Done in {dt:.1f} s.")
        self.log_msg(f"Analysis completed in {dt:.1f} s.")
        for th in res.thermal:
            bal = th.balance
            self.log_msg(f"Thermal state {th.name}: T {fmt(float(th.T.min()))} … {fmt(float(th.T.max()))} °C"
                         + (f", balance {fmt(100 * bal['imbalance'], 2)} %" if "imbalance" in bal else ""))
        for w in res.warnings:
            self.log_msg(f"Warning: {w}")
        for ms in res.modes:
            self.log_msg(f"{ms.name}: " + ", ".join(fmt(float(v), 5) for v in ms.values)
                         + (" Hz" if ms.kind == "modal" else " (λ factors)"))
        self.mesh_messages(res.mesh)
        self.messages.set_source("results", [("warning", w) for w in res.warnings], unique=True)
        self.f8_messages(res)
        self.messages.set_source("solver", [("info", decimal(f"Analysis completed in {dt:.1f} s"))])
        self.fat_panel.result = None              # fatigue and joint assessment belong to the previous results
        self.joints_panel.assessment = None
        self.update_title()
        self.refresh_all()
        if res.steps and (self.study.active_bolts() or self.study.active_welds()):
            self.joints_panel.on_results()
        self.set_stage(STAGE["Results"])

    def on_solve_failed(self, text):
        if getattr(self, "_stop_keep", False):      # stopped before the first increment was converted
            text = ("Calculation stopped – no converted increment to keep (stop only after the first increments, or reduce dt₀).\n\n" + text)
        cancelled = "cancelled" in text.lower()
        self.progress_finish("cancelled" if cancelled else "error",
                             "Cancelled" if cancelled else "Analysis failed – see the log for the reason")
        self.solve_info.setText("Analysis failed.")
        self.log_msg(f"ERROR: {text}")
        self.messages.set_source("solver", [("warning" if cancelled else "error",
                                            "Analysis canceled" if cancelled else f"Analysis failed: {text}")])
        if not cancelled:
            self.dock_messages.raise_()
        if self.joints_panel.offer_ignore(text):      # free bodies → offer to exclude parts
            return
        msg(self, APP_NAME, f"Analysis failed:\n\n{text}", QMessageBox.Critical)

    # ================================================================== results
    def _post_key(self) -> str:
        """Fingerprint of the study inputs of the post-processing (combinations, load-case names, Stress Tool settings, materials and
        their assignment) – key of the GUI caches on the results (the cached value is the same as a new computation)."""
        st = self.study
        return repr(([c.to_dict() for c in getattr(st, "combinations", [])], [lc.name for lc in st.load_cases],
                     st.assessment, sorted(st.materials.items()), sorted(st.part_materials.items()), st.default_material))

    def _safety_field(self, st, criterion):
        """Safety field of the Stress Tool (assessment.compute_safety_field, capped at 10), cached on the step per (criterion, inputs)
        – redraws (section, animation, colour bands) do not recompute it."""
        key = (criterion, self._post_key(), id(self.results.mesh))
        cache = st.__dict__.setdefault("_gui_safety", {})
        if key not in cache:
            from fea_core.assessment import compute_safety_field
            args = () if criterion is None else (criterion,)
            cache.clear()                              # one entry per step (inputs changed → the old one is useless)
            cache[key] = np.minimum(compute_safety_field(self.study, self.results.mesh, st, *args)[0], 10.0)
        return cache[key]

    def _combination(self, res, ci):
        """compute_combination for combination `ci` of the study, cached on the results (key = definitions of all combinations and the
        inputs of the safety envelopes; at most 4 entries – every combination holds U and S of the whole mesh)."""
        from fea_core.combinations import compute_combination
        comb = self.study.combinations[ci]
        key = (ci, self._post_key())
        cache = res.__dict__.setdefault("_gui_comb_cache", {})
        if key not in cache:
            val = compute_combination(self.study, res, comb)
            while len(cache) >= 4:
                cache.pop(next(iter(cache)))
            cache[key] = val
        return cache[key]

    def result_field(self, st, qi):
        name = QUANTITIES[qi][0]
        if name == "von Mises":
            return st.mises
        if name == "Total displacement":
            return st.umag
        # local coordinate system of the results (A4): displacement and stress components are transformed at each node
        cs_name = None
        if getattr(self, "r_csys", None) is not None and self.r_csys.currentIndex() > 0 and self.study:
            cs_name = self.r_csys.currentData()
            if cs_name not in self.study.coordinate_systems:
                cs_name = None
        comps = {"σx": 0, "σy": 1, "σz": 2, "τxy": 3, "τyz": 4, "τzx": 5}
        if cs_name is not None and (name in ("Displacement X", "Displacement Y", "Displacement Z") or name.removeprefix("Stress ") in comps):
            from fea_core import csys as _csys
            try:
                if name.startswith("Displacement "):
                    U_loc = _csys.transform_displacements(self.study, cs_name, st.U, self.results.mesh.nodes, "results")
                    return U_loc[:, "XYZ".index(name[-1])]
                S_loc = _csys.transform_stress(self.study, cs_name, st.S, self.results.mesh.nodes, "results")
                return S_loc[:, comps[name.split()[-1]]]
            except _csys.CsysError as e:           # e.g. a point on the axis of a cylindrical system → global components
                self.log_msg(f"Coordinate system “{cs_name}”: {e} – global components shown.")
        if name.startswith("Displacement "):
            return st.U[:, "XYZ".index(name[-1])]
        if name.startswith("1st principal"):
            return st.principal[:, 0]
        if name.startswith("3rd principal"):
            return st.principal[:, 2]
        if name.removeprefix("Stress ") in comps:
            return st.S[:, comps[name.split()[-1]]]

        if name == "Safety factor to Re":
            re_node = self.node_re()
            vm = np.maximum(st.mises, 1e-9)
            return np.minimum(re_node / vm, 10.0)
        if name == "Safety factor – Tresca":
            return self._safety_field(st, "tresca")
        if name == "Safety factor – Mohr-Coulomb":
            return self._safety_field(st, "mohr_coulomb")
        if name == "Safety factor (Stress Tool)":
            return self._safety_field(st, None)                   # criterion per material
        if name == "Stress error estimate":
            v = st.fields.get("stress_error")
            return np.full(len(st.U), np.nan) if v is None else np.asarray(v, float)
        if name.startswith("3D printing"):
            # non-printed parts: safety 10 (scale limit), index 0 – only printed parts are assessed
            key = "print_safety" if "safety factor" in name else "print_index"
            v = st.fields.get(key)
            if v is None:
                return np.full(len(st.U), np.nan)
            return np.minimum(np.nan_to_num(v, nan=10.0, posinf=10.0), 10.0) if key == "print_safety" \
                else np.nan_to_num(v, nan=0.0)
        if name == "Temperature":
            T = st.fields.get("T")
            return np.full(len(st.U), np.nan) if T is None else np.asarray(T, float)
        if name in CONTACT_Q:                         # contact (F8): pressure, status, opening, slip, friction stress
            v = st.fields.get(CONTACT_Q[name])
            return np.full(len(st.U), np.nan) if v is None else np.asarray(v, float)
        if name == PEEQ_Q:                            # plasticity: equivalent plastic strain (nodal, from ccx)
            v = st.fields.get("peeq")
            return np.full(len(st.U), np.nan) if v is None else np.maximum(np.asarray(v, float), 0.0)
        if name in TOPO_Q:                            # topology optimisation: nodal density 0 to 1
            v = st.fields.get("topo_density")
            return np.full(len(st.U), np.nan) if v is None else np.asarray(v, float)
        raise KeyError(name)

    @staticmethod
    def thermal_field(th, qi, frame):
        """Field of a thermal state; frame = index of the snapshot of a transient state (None = end). Flux only at the end."""
        name = THERMAL_QUANTITIES[qi][0]
        if name == "Temperature":
            return th.T if frame is None else th.frames[frame]
        if frame is not None:
            return None                       # the flux is stored only at the end of the state
        if name == "Heat flux |q|":
            return th.flux_mag
        return th.flux[:, "xyz".index(name[-1])]

    def current_result(self):
        """("step", i) | ("thermal", j) according to the choice in the Results step, None = nothing."""
        if self.results is None:
            return None
        d = self.r_step.currentData()
        if not d:
            return None
        kind, *rest = d.split(":")
        if kind == "mode":                          # ("mode", (set, shape)) – natural shape / buckling shape
            j, k = int(rest[0]), int(rest[1])
            ok = 0 <= j < len(self.results.modes) and 0 <= k < len(self.results.modes[j].values)
            return ("mode", (j, k)) if ok else None
        if kind == "dyn":                           # ("dyn", (druh, k)) – dynamika (dyn_results.QUANTS)
            ok = getattr(self.results, "dynamics", None) is not None and rest and rest[0] in dynr.QUANTS
            return ("dyn", (rest[0], int(rest[1]) if len(rest) > 1 else 0)) if ok else None
        if kind == "combination":
            i = int(rest[0])
            ok = self.study is not None and hasattr(self.study, "combinations") and 0 <= i < len(self.study.combinations)
            return ("combination", i) if ok else None
        if kind == "sub":                           # submodel (Results.submodels, submodel_dialog)
            return smd.current(self.results, rest)
        i = int(rest[0])
        r2 = self.results.twod
        n = (len(self.results.steps) if kind == "step" else len(self.results.thermal) if kind == "thermal" else
             len(r2.steps) if kind == "twod" and r2 is not None else
             len(self.results.shell.steps) if kind == "shell" and self.results.shell is not None else 0)
        return (kind, i) if 0 <= i < n else None

    def result_qlist(self, cur=None) -> list[tuple[str, str]]:
        """Quantities of the menu by kind of result: bodies, frames (alone / mixed model), heat, modes, 2D."""
        cur = cur if cur is not None else self.current_result()
        kind = cur[0] if cur else "step"
        res = self.results
        solids = res is not None and len(res.mesh.nodes) > 0
        if kind == "thermal":
            return THERMAL_QUANTITIES
        if kind == "twod":
            return [(n, u) for n, u, _ in twod_gui.quantities(res.twod)]
        if kind == "shell":
            return [(n, u) for n, u, _, _ in shell_gui.SHELL_Q]
        if kind == "sub":
            return smd.QLIST
        if kind == "mode":
            return mode_quantities(res.modes[cur[1][0]]) if solids and cur else (MODE_QUANTITIES if solids
                                                                                  else [frs.MODE_FRAME_Q])
        if kind == "dyn":
            return dynr.QUANTS[cur[1][0]]
        has_frame = res is not None and any(st.frame for st in res.steps)
        if not solids and has_frame:
            return frs.FRAME_QUANTITIES
        return QUANTITIES + (frs.FRAME_QUANTITIES if has_frame else [])

    def select_result(self, data: str):
        """Selects a result in the menu of the Results step (data of the r_step item, e.g. "mode:0:2")."""
        if STAGES[self.stage] != "Results" and self.results is not None:
            self.set_stage(STAGE["Results"])
        k = self.r_step.findData(data)
        if k >= 0:
            self.r_step.setCurrentIndex(k)

    def deformation_scale(self, U) -> float:
        """Deformation scale by the choice "Deformation" (automatic = max. displacement at 5 % of the model diagonal)."""
        m = self.results.mesh
        umax = float(np.linalg.norm(U, axis=1).max()) if len(U) else 0.0
        scale = {0: 0.0, 2: 1.0, 3: 10.0, 4: 100.0, 5: 1000.0}.get(self.r_scale.currentIndex())
        if scale is None:
            diag = (float(np.linalg.norm(m.nodes.max(axis=0) - m.nodes.min(axis=0))) if len(m.nodes) else
                    self.view.model_size())                       # frame / 2D without a 3D mesh
            scale = 0.05 * diag / umax if umax > 0 else 0.0
        return scale

    def displayed(self, p) -> np.ndarray:
        """Point of the undeformed mesh in the position shown in 3D (displacement × scale of the displayed result)."""
        p = np.asarray(p, float)
        scale, U = self._r_view
        if not scale or U is None or self.results is None:
            return p
        cur = self.current_result()
        if cur and cur[0] == "twod":
            r2 = self.results.twod
            fr = r2.frame2d()
            try:
                return p + scale * fr.vec3d(r2.probe(fr.to2d(p), "U", cur[1]))
            except Exception:  # noqa: BLE001
                return p
        try:
            from fea_core.results import tet10_shape
            e, xi = self.results.locate(p)
            conn = self.results.mesh.tets[e]
            return p + scale * (tet10_shape(xi) @ U[conn])
        except Exception:  # noqa: BLE001 – outside the mesh → no displacement
            return p

    def undeformed(self, x) -> np.ndarray:
        """Point clicked on the displayed (deformed) model → undeformed position (displacement of the nearest surface node)."""
        x = np.asarray(x, float)
        scale, U = self._r_view
        pr = getattr(self, "_probe", None)
        if not scale or U is None or pr is None:
            return x
        tree, ids = pr[0], pr[1]
        _, j = tree.query(x)
        n = int(ids[j])
        return x - scale * U[n] if n >= 0 else x

    def current_frame(self, th):
        """Index of the snapshot of the transient state selected in r_time; None = end of the state (also a steady one)."""
        if not th.transient or th.frames is None or not len(th.times):
            return None
        k = self.r_time.currentIndex()
        return None if k < 0 or k >= len(th.times) - 1 else k

    def on_r_step(self):
        """Change of the result: quantities by kind (mechanics / heat), times of the transient state, then display."""
        self._sync_result_controls()
        self.show_result()
        self.res_overview.refresh()                 # numbers of the displayed state on the Assessment card

    def _sync_result_controls(self):
        cur = self.current_result()
        kind = cur[0] if cur else "step"
        qlist = self.result_qlist(cur)
        key = (kind, tuple(q[0] for q in qlist))
        self.refresh_r_csys()
        is_step = kind == "step"
        self.r_csys.setVisible(is_step)
        self.r_csys_lbl.setVisible(is_step)
        if getattr(self, "_r_kind", None) != key:
            old = self.r_qty.currentText().split(" [")[0]
            self.r_qty.blockSignals(True)
            self.r_qty.clear()
            for name, unit in qlist:
                self.r_qty.addItem(f"{name} [{unit}]")
            names = [q[0] for q in qlist]
            self.r_qty.setCurrentIndex(names.index(old) if old in names else 0)
            self.r_qty.blockSignals(False)
            self._r_kind = key
        th = self.results.thermal[cur[1]] if kind == "thermal" and cur else None
        transient = th is not None and th.transient and th.frames is not None and len(th.times) > 0
        self.r_time.blockSignals(True)
        if transient:
            items = [f"t = {fmt(float(t))} s" for t in th.times]
            if [self.r_time.itemText(i) for i in range(self.r_time.count())] != items:
                self.r_time.clear()
                self.r_time.addItems(items)
                self.r_time.setCurrentIndex(len(items) - 1)
        else:
            self.r_time.clear()
        self.r_time.blockSignals(False)
        self.r_time.setVisible(transient)
        self.r_time_lbl.setVisible(transient)
        self.r_scale.setEnabled(kind != "thermal")
        # dynamics: frequency slider (harmonic) / time (transient), phase, absolute displacement
        ax = dynr.axis(self.results.dynamics, cur[1][0]) if kind == "dyn" else None
        self.r_dyn.setVisible(ax is not None)
        self.r_dyn_lbl.setVisible(ax is not None)
        if ax is not None:
            vals, label, _unit = ax
            self.r_dyn_lbl.setText(label)
            s = self.r_dyn_slider
            if s.maximum() != len(vals) - 1 or getattr(self, "_dyn_axis", None) != cur[1][0]:
                s.blockSignals(True)
                s.setRange(0, len(vals) - 1)
                d = self.results.dynamics
                if cur[1][0] == "harm":                 # default = frequency of the stress maximum (envelope)
                    h = d.harmonic
                    s.setValue(int(np.asarray(h["env_S_at"])[int(np.argmax(h["env_S"]))]))
                else:
                    tr = d.transient
                    te = float(np.asarray(tr["t_env"])[np.asarray(tr["env_S_at"])[int(np.argmax(tr["env_S"]))]])
                    s.setValue(int(np.argmin(np.abs(vals - te))))
                s.blockSignals(False)
                self._dyn_axis = cur[1][0]
            self.r_phase.setVisible(cur[1][0] == "harm")
            self.r_abs.setVisible(dynr.has_base(self.results.dynamics, cur[1][0]))

    def node_re(self):
        m = self.results.mesh
        re_part = np.array([self.study.material_of(p).Re or np.inf for p in m.part_names])
        re_node = np.full(len(m.nodes), np.inf)
        for pi, re in enumerate(re_part):
            np.minimum.at(re_node, m.tets[m.tet_part == pi].ravel(), re)
        return re_node

    def show_result(self):
        if self.results is None or STAGES[self.stage] != "Results":
            return
        from .cyclic_gui import full_results          # cyclic symmetry: the sector shown as the full 360° model
        full = full_results(self)
        if full is None:
            return self._show_result()
        sector, self.results = self.results, full
        try:
            return self._show_result()
        finally:
            self.results = sector

    def _show_result(self):
        res = self.results
        self._probe = self._probe_exact = None
        if not getattr(self, "_anim_frame", False):   # the displayed result changed → cached animation frames are stale
            self.view.anim_reset()
            self._anim_state = {}
        cur = self.current_result()
        if cur is None:
            self.res_info.setHtml("<i>The results contain no load case or thermal case.</i>")
            self.view.clear_results()
            self.legend.clear()
            self.clear_hotspots()
            return
        kind, si = cur
        qi = max(self.r_qty.currentIndex(), 0)
        m = res.mesh
        anim_frame = kind == "step" and getattr(self, "_anim_frame", False)   # frame of the deformation animation
        qlist = self.result_qlist(cur)
        qname = qlist[min(qi, len(qlist) - 1)][0]
        self._probe_labels = None
        if kind == "twod":
            return self.show_twod_result(si, qi)
        if kind == "shell":
            return self.show_shell_result(si, qi)
        if kind == "sub":
            return smd.show(self, si, qi)
        if kind == "step" and (len(m.nodes) == 0 or frs.is_frame_quantity(qname)):
            return self.show_frame_result(si, qi)
        if kind == "mode" and len(m.nodes) == 0:
            return self.show_frame_mode(si)
        if kind == "dyn":
            return self.show_dyn_result(si, qi)
        tri, elem = m.boundary_faces()
        if hasattr(self, "r_part") and self.r_part.currentIndex() > 0:
            pname = self.r_part.currentText()
            if pname in m.part_names:
                pi = m.part_names.index(pname)
                kept_faces = (m.tet_part[elem] == pi)
                tri = tri[kept_faces]
        surf_nodes = np.unique(tri)
        st = th = frame = None
        if kind == "thermal":
            th = res.thermal[si]
            frame = self.current_frame(th)
            name, unit = THERMAL_QUANTITIES[min(qi, len(THERMAL_QUANTITIES) - 1)]
            vals = self.thermal_field(th, min(qi, len(THERMAL_QUANTITIES) - 1), frame)
            scale = 0.0
            X = m.nodes
            missing = "Heat flux is stored only for the end of a transient case – choose the last time."
            self._r_view = (0.0, None)
        elif kind == "mode":
            ms = res.modes[si[0]]
            shape = np.asarray(ms.shapes[si[1]], float)
            mq = mode_quantities(ms)                  # normalised shape | nonlinear buckling displacement [mm]
            qm = min(qi, len(mq) - 1)
            name, unit = mq[qm]
            vals = mode_quantity(shape, qm)
            scale = self.deformation_scale(shape)
            X = m.nodes + scale * shape
            missing = "The shape is not available."
            self._r_view = (scale, shape)
        elif kind == "combination":
            comb = self.study.combinations[si]
            from fea_core.combinations import EnvelopeResult
            try:
                st = self._combination(res, si)
            except (ValueError, KeyError) as e:
                self.res_info.setHtml(f"<i>Combination “{escape(str(comb.name))}” cannot be computed: {escape(str(e))}</i>")
                self.view.clear_results()
                self.legend.clear()
                self.clear_hotspots()
                return
            if isinstance(st, EnvelopeResult):          # envelope: one field over the nodes, without deformation
                unit = {"umag": "mm", "safety": "–"}.get(st.field_name, "MPa")
                name = f"Envelope {st.envelope_type}: {st.field_name}"
                vals = np.asarray(st.values, float)
                scale = 0.0
                X = m.nodes
                missing = "The envelope has no values."
                self._r_view = (0.0, None)
            else:
                name, unit = QUANTITIES[min(qi, len(QUANTITIES) - 1)]
                vals = self.result_field(st, min(qi, len(QUANTITIES) - 1))
                scale = self.deformation_scale(st.U)
                X = m.nodes + scale * st.U
                missing = "The quantity is not available in the combination."
                self._r_view = (scale, st.U)
        else:
            st = res.step(si)
            name, unit = QUANTITIES[min(qi, len(QUANTITIES) - 1)]
            vals = self.result_field(st, min(qi, len(QUANTITIES) - 1))
            anim_factor = getattr(self, "_anim_factor", 1.0)
            scale = self.deformation_scale(st.U) * anim_factor            # deformation scale
            if name in TOPO_Q:
                scale = 0.0                                                # density belongs to the undeformed shape
            X = m.nodes + scale * st.U
            missing = ("The case has no thermal load." if name == "Temperature" else
                       "The stress error estimate is computed only by CalculiX (solver “Direct sparse solver – CalculiX” or the automatic choice with CalculiX) – a result from NGSolve does not have it." if name == "Stress error estimate" else
                       "The case has no contact pairs (Contacts step) – the contact quantity is not available."
                       if name in CONTACT_Q else
                       TOPO_MISSING if name in TOPO_Q else
                       "The result has no plastic strain – tick Plasticity in the Material properties window and solve again."
                       if name == PEEQ_Q else
                       "The quantity is not available in the result (no printed part).")
            self._r_view = (scale, st.U)
            if not anim_frame:                         # animation frame: only the deformed bands change
                self.contact_box.fill(res, si)
        if vals is None or np.isnan(vals).all():
            if name.startswith("3D printing"):
                missing = ("The 3D print quantity is not in the result – no part has a printed material. A material such as “PLA print XY (approximate)” is a solid isotropic material without layers and infill; to assess the print, choose a printed material in the 3D print step (e.g. “Print PLA Bambu Basic”) and run the calculation again.")
            self.res_info.setHtml(f"<i>{missing}</i>")
            self.view.clear_results()
            self.legend.clear()
            self.clear_hotspots()
            self.res_tabs.setCurrentIndex(0)          # the message is in the Summary – otherwise the view would stay empty without an explanation
            self.view.show_parts(True)                 # the model stays visible (without results)
            self.view.set_mode("results")
            return
        sv = vals[surf_nodes]
        vmin, vmax = float(sv.min()), float(sv.max())
        # section: trimmed surface + coloured section through the mesh interior (plane in the undeformed position)
        plane = self.section_plane()
        tris = outline = None
        probe_pts, probe_vals, probe_ids = X[surf_nodes], sv, surf_nodes
        shape_view = name == TOPO_SHAPE_Q and plane is None and kind == "step"
        if shape_view:
            # resulting shape ρ ≥ 0.5: closed surface of the isosurface instead of the mesh surface, without deformation, legend 0.5–1
            tris = self.topo_shape_tris(res)
            if tris is not None:
                vmin, vmax = 0.5, 1.0
                if len(tris[0]):
                    probe_pts, probe_vals = tris[0].reshape(-1, 3), tris[1].ravel()
                    probe_ids = np.full(len(probe_vals), -1)
        elif name in TOPO_Q:
            vmin, vmax = 0.0, 1.0
        if plane is not None:
            from fea_core.section import plane_section, clip_surface, section_outline
            o, nrm = plane
            Ps, Vs = clip_surface(m.nodes, tri, vals, o, nrm, X)
            Pc, Vc = plane_section(m.nodes, m.tets, vals, o, nrm, X)
            tris = (np.concatenate([Ps, Pc]), np.concatenate([Vs, Vc]))
            outline = section_outline(Pc)
            if len(Vc):
                vmin, vmax = min(vmin, float(Vc.min())), max(vmax, float(Vc.max()))
            kept = (m.nodes[surf_nodes] - o) @ nrm <= 0
            probe_pts = np.vstack([X[surf_nodes][kept], Pc.reshape(-1, 3)])
            probe_vals = np.concatenate([sv[kept], Vc.ravel()])
            probe_ids = np.concatenate([surf_nodes[kept], np.full(Vc.size, -1)])
        from .contour import band_edges
        if name == "Contact – contact state":          # discrete states (−1 not contact, 0 no contact, 1 sliding, 2 sticking)
            edges = np.array([-1.5, -0.5, 0.5, 1.5, 2.5])
        else:
            edges = band_edges(vmin, vmax, self.r_bands.value())
        colors = self.view.show_contour(X, tri, vals, edges, f"{name} [{unit}]", edges_on=self.r_edges.isChecked(),
                                        reverse=(name in SAFETY_QTY), show_scale=False, tris=tris, outline=outline,
                                        deform=(m.nodes, st.U, scale) if anim_frame and tris is None else None)
        self.legend.set_data(f"{name} [{unit}]", edges, colors, lambda v: fmt(float(v)))
        if self.r_outline.isChecked():
            self.view.show_parts(True, 0.85)
            self.view.set_mode("results")
        else:
            self.view.show_parts(False)
        imax = surf_nodes[int(np.argmax(sv))]
        # maximum places with singularity classification (table); note at the legend if the maximum is a singularity
        if kind in ("mode", "combination") or name in TOPO_Q:   # mode / buckling shape, combination, density: no maximum-places table
            self.clear_hotspots()
            self.res_tabs.setTabText(1, "Highest stresses")
            note, sing_xyz, r_cl = "", None, 0.0
        else:
            hot_key = (id(res), kind, si, name, frame, self.section_plane() is None)
            if anim_frame and getattr(self, "_anim_hot", (None,))[0] == hot_key:
                note, sing_xyz, r_cl = self._anim_hot[1]          # the table and the classification do not depend on the scale
                self._hs_X = X                                      # row click marks the current deformed position
            else:
                note, sing_xyz, r_cl = self.fill_hotspots(kind, si, name, unit, vals, X, frame)
                self._anim_hot = (hot_key, (note, sing_xyz, r_cl))
        self.legend.set_note(note)
        # max/min markers only on the displayed part (trimmed surface + section)
        jmax, jmin = int(np.argmax(probe_vals)), int(np.argmin(probe_vals))
        if name not in SAFETY_QTY and name != "Contact – contact state" and name not in TOPO_Q:
            lbl = f"max {fmt(float(probe_vals[jmax]))}"
            if sing_xyz is not None and probe_ids[jmax] >= 0 and \
                    np.linalg.norm(m.nodes[probe_ids[jmax]] - sing_xyz) <= r_cl:
                lbl += " (singularity)"
            self.view.mark(probe_pts[jmax], lbl, (170, 0, 0))
        if name not in NO_MIN_QTY:
            self.view.mark(probe_pts[jmin], f"min {fmt(float(probe_vals[jmin]))}",
                           (170, 0, 0) if name in SAFETY_QTY else (0, 60, 170))
        self._probe = (NearestPoints(probe_pts), probe_ids, probe_vals, unit, name)
        if th is not None:
            # thermal state: geometry without deformation → probe by interpolation with the element shape functions
            if name == "Temperature" and frame is None:
                self._probe_exact = lambda p, _n=th.name: float(res.probe(p, "T", _n)[0])
            else:
                self._probe_exact = lambda p, _v=vals: self.interpolate(_v, p)
            self.fill_thermal_info(th, frame)
        elif kind == "mode":
            self.res_info.setHtml(mode_info_html(res.modes[si[0]], si[1], self.study)
                                  + f"<br>Deformation scale: {fmt(scale, 3)}×")
            self.modes_table.select(*si)
            if self.res_tabs.currentIndex() == 1:
                self.res_tabs.setCurrentIndex(0)
        elif kind == "combination" and hasattr(st, "governing_summary"):
            names = escape(", ".join(str(n) for n in st.combination_names))
            gov = escape(", ".join(f"{k}: {v} nodes" for k, v in st.governing_summary.items()))
            self.res_info.setHtml(f"<b>{escape(str(st.name))}</b><br>Envelope {escape(str(st.envelope_type))} of field {escape(str(st.field_name))} from combinations: {names}<br>Governing combination: {gov}<br>Range: {fmt(st.min_val)} to {fmt(st.max_val)} {escape(str(unit))}")
        else:
            if not anim_frame:
                self.fill_result_info(st, scale, imax, vals)
            if st.frame:                               # mixed model: members via the result of the bodies
                frs.overlay(self, res, st, scale)
        if name == "Stress error estimate":
            self.legend.set_note(tr(STRESS_ERROR_TIP).replace("\n", " "))    # translate before joining the lines
        if name == "Contact – contact state":
            self.legend.set_note(STATUS_NOTE)
        dlg = self._path_dlg                       # progress line (results are redrawn → draw again)
        if dlg is not None and dlg.isVisible():
            dlg.draw_line()

    # ------------------------------------------------------------------ dynamics results (F8)
    def show_dyn_result(self, si, qi):
        """Dynamics field (dyn_results.field) as a colour map on the mesh surface (+ section), summary and plot tab."""
        res = self.results
        d = res.dynamics
        m = res.mesh
        sub, k = si
        self.clear_hotspots()
        self.contact_box.fill(None, None)
        if len(m.nodes) == 0 or d.shapes.shape[1] != len(m.nodes):
            self.res_info.setHtml("<i>Dynamics fields are shown only on the solid mesh – for members see the charts (Dynamics tab).</i>")
            self.view.clear_results()
            self.legend.clear()
            return
        idx = self.r_dyn_slider.value()
        phase = float(self.r_phase.value())
        absolute = self.r_abs.isChecked() and self.r_abs.isVisible()
        try:
            vals, U, name, unit = dynr.field(d, sub, k, idx, qi, phase, absolute)
        except Exception as e:  # noqa: BLE001 – missing data in the result (older file)
            self.res_info.setHtml(f"<i>The dynamics result cannot be shown: {escape(str(e))}</i>")
            self.view.clear_results()
            self.legend.clear()
            return
        ax = dynr.axis(d, sub)
        if ax is not None:
            xs, _lbl, u = ax
            self.r_dyn_val.setText(f"{fmt(float(xs[min(idx, len(xs) - 1)]), 5)} {u}")
        scale = 0.0 if U is None else self.deformation_scale(U)
        X = m.nodes + scale * U if U is not None else m.nodes
        self._r_view = (scale, U) if U is not None else (0.0, None)
        tri, _ = m.boundary_faces()
        surf_nodes = np.unique(tri)
        if not np.isfinite(vals).any():
            self.res_info.setHtml("<i>The quantity is not available in the result.</i>")
            self.view.clear_results()
            self.legend.clear()
            return
        vals = np.nan_to_num(vals, nan=0.0)
        sv = vals[surf_nodes]
        vmin, vmax = float(sv.min()), float(sv.max())
        plane = self.section_plane()
        tris = outline = None
        probe_pts, probe_vals, probe_ids = X[surf_nodes], sv, surf_nodes
        if plane is not None:
            from fea_core.section import plane_section, clip_surface, section_outline
            o, nrm = plane
            Ps, Vs = clip_surface(m.nodes, tri, vals, o, nrm, X)
            Pc, Vc = plane_section(m.nodes, m.tets, vals, o, nrm, X)
            tris = (np.concatenate([Ps, Pc]), np.concatenate([Vs, Vc]))
            outline = section_outline(Pc)
            if len(Vc):
                vmin, vmax = min(vmin, float(Vc.min())), max(vmax, float(Vc.max()))
            kept = (m.nodes[surf_nodes] - o) @ nrm <= 0
            probe_pts = np.vstack([X[surf_nodes][kept], Pc.reshape(-1, 3)])
            probe_vals = np.concatenate([sv[kept], Vc.ravel()])
            probe_ids = np.concatenate([surf_nodes[kept], np.full(Vc.size, -1)])
        from .contour import band_edges
        edges = band_edges(vmin, vmax, self.r_bands.value())
        colors = self.view.show_contour(X, tri, vals, edges, f"{name} [{unit}]", edges_on=self.r_edges.isChecked(),
                                        show_scale=False, tris=tris, outline=outline)
        self.legend.set_data(f"{name} [{unit}]", edges, colors, lambda v: fmt(float(v)))
        self.legend.set_note(dynr.UNCERTAIN if sub != "mode" else "")
        self.view.show_parts(self.r_outline.isChecked(), 0.85)
        if len(probe_vals):
            jmax = int(np.argmax(probe_vals))
            self.view.mark(probe_pts[jmax], f"max {fmt(float(probe_vals[jmax]))}", (170, 0, 0))
        self._probe = (NearestPoints(probe_pts), probe_ids, probe_vals, unit, name)
        self.res_info.setHtml(dynr.info_html(d, sub, k, idx, phase, absolute, vals, name, unit, scale))

    # ------------------------------------------------------------------ results of frames and 2D (F7)
    def _finish_lines(self, title, edges, colors, pts, vals, labels, unit, name, reverse=False):
        """Legend, max / min markers and probe for results drawn as lines (frames)."""
        self.legend.set_data(title, edges, colors, lambda v: fmt(float(v)))
        self.legend.set_note("")
        vals = np.asarray(vals, float)
        if len(vals):
            jmax, jmin = int(np.nanargmax(vals)), int(np.nanargmin(vals))
            self.view.mark(pts[jmax], f"max {fmt(float(vals[jmax]))}", (170, 0, 0))
            if vals[jmin] < 0 or name.split(" – ")[-1] in frs.DIAGRAMS:
                self.view.mark(pts[jmin], f"min {fmt(float(vals[jmin]))}", (0, 60, 170))
            self._probe = (NearestPoints(pts), np.full(len(pts), -1), vals, unit, name)
            self._probe_labels = labels

    def show_frame_result(self, si, qi):
        res = self.results
        st = res.steps[si]
        qlist = self.result_qlist()
        name, unit = qlist[min(qi, len(qlist) - 1)]
        if not frs.is_frame_quantity(name):
            name, unit = frs.FRAME_QUANTITIES[0]
        self.view.clear_results()
        self.clear_hotspots()
        self.view.show_parts(bool(self.parts), 0.85)
        self.view.set_mode("results")
        out = frs.step_draw(self, res, st, name, unit, self.r_bands.value())
        if out is None:
            self.res_info.setHtml("<i>The case has no member results.</i>")
            self.legend.clear()
            return
        title, edges, colors, pts, vals, labels, scale = out
        self._r_view = (0.0, None)
        self._finish_lines(title, edges, colors, pts, vals, labels, unit, name)
        self.res_info.setHtml(frs.step_info_html(self, res, st)
                              + (f"<br>Deformation scale: {fmt(scale, 3)}×" if scale else ""))
        self.frame_box.fill(res, st)
        self.view._update()

    def show_frame_mode(self, si):
        res = self.results
        ms = res.modes[si[0]]
        self.view.clear_results()
        self.clear_hotspots()
        self.view.set_mode("results")
        out = frs.mode_draw(self, res, ms, si[1], self.r_bands.value())
        info = mode_info_html(ms, si[1], self.study)
        if out is None:
            self.legend.clear()
            self.res_info.setHtml(info + "<br><i>The member shape is not available (buckling of members alone is stored since version F7; run the analysis again).</i>")
            return
        title, edges, colors, pts, vals, labels, scale = out
        self._r_view = (0.0, None)
        self._finish_lines(title, edges, colors, pts, vals, None, "–", title)
        self.res_info.setHtml(info + f"<br>Deformation scale: {fmt(scale, 3)}×")
        self.modes_table.select(*si)
        self.view._update()

    def show_twod_result(self, i, qi):
        res = self.results
        r2 = res.twod
        self.clear_hotspots()
        self.view.show_parts(bool(self.parts), 0.85)
        self.view.set_mode("results")
        title, edges, colors, X3, vals, scale, U3, key = twod_gui.draw(self, r2, i, qi, self.r_bands.value())
        if self.r_outline.isChecked():
            self.view.show_parts(True, 0.85)
        self.legend.set_data(title, edges, colors, lambda v: fmt(float(v)))
        self.legend.set_note("2D section: 3D parts shown for orientation only (transparent); values apply in the section plane")
        jmax, jmin = int(np.argmax(vals)), int(np.argmin(vals))
        self.view.mark(X3[jmax], f"max {fmt(float(vals[jmax]))}", (170, 0, 0))
        self.view.mark(X3[jmin], f"min {fmt(float(vals[jmin]))}", (0, 60, 170))
        self._r_view = (scale, U3)
        fr = r2.frame2d()
        self._probe = (NearestPoints(X3), np.arange(len(X3)), vals, title.split("[")[-1].rstrip("]"), title.split(" [")[0])
        self._probe_exact = lambda p, _k=key, _i=i: float(r2.probe(fr.to2d(self.undeformed(p)), _k, _i))
        self.res_info.setHtml(twod_gui.info_html(self, r2, i, key))
        dlg = self._path_dlg
        if dlg is not None and dlg.isVisible():
            dlg.draw_line()

    def show_shell_result(self, i, qi):
        """Result of the shell model (Results.shell) on the mid-surface: contours, markers, probe, summary."""
        rs = self.results.shell
        self.clear_hotspots()
        self.view.show_parts(False)
        self.view.set_mode("results")
        title, edges, colors, X, vals, scale, U, key, layer = shell_gui.draw(self, rs, i, qi, self.r_bands.value())
        if self.r_outline.isChecked():
            self.view.show_parts(True, 0.85)
        self.legend.set_data(title, edges, colors, lambda v: fmt(float(v)))
        self.legend.set_note("Shells: contours on the mid-surface; layer depends on the quantity (top / middle / bottom surface, max of surfaces)")
        jmax, jmin = int(np.argmax(vals)), int(np.argmin(vals))
        self.view.mark(X[jmax], f"max {fmt(float(vals[jmax]))}", (170, 0, 0))
        if key not in ("mises", "membrane", "bending", "u", "thickness"):
            self.view.mark(X[jmin], f"min {fmt(float(vals[jmin]))}", (0, 60, 170))
        self._r_view = (scale, U)
        self._probe = (NearestPoints(X), np.arange(len(X)), vals, title.split("[")[-1].rstrip("]"), title.split(" [")[0])
        self._probe_exact = lambda p, _v=vals: self.interpolate_shell(rs, _v, p)   # displayed field (layer)
        self.res_info.setHtml(shell_gui.info_html(self, rs, i))

    def interpolate_shell(self, rs, vals, p) -> float:
        """Nodal field of the shells at a point (nearest point of the mid-surface, tri6 shape functions)."""
        from fea_core.shell.loads import tri6_shape
        q = rs.probe(self.undeformed(p), "u", 0, "mid")
        e = q["element"]
        X = rs.nodes[rs.tri[e]]
        a, b, d = X[0], X[1], X[2]
        u, v = b - a, d - a
        G = np.array([[u @ u, u @ v], [u @ v, v @ v]])
        xi, eta = np.clip(np.linalg.solve(G, [u @ (np.asarray(q["point"]) - a), v @ (np.asarray(q["point"]) - a)]), 0, 1)
        N, _ = tri6_shape(xi, eta)
        return float(N @ np.asarray(vals)[rs.tri[e]])

    # ------------------------------------------------------------------ result dialogs (F7)
    def open_path_dialog(self):
        if self.results is None:
            msg(self, APP_NAME, "Run the analysis first (Solve step).")
            return
        if not self.results.steps and not self.results.thermal and self.results.twod is None:
            msg(self, APP_NAME, "The results contain only mode shapes – a line plot needs a load case or thermal case.")
            return
        from .path_dialog import PathDialog
        if len(self.results.mesh.nodes) == 0 and self.results.twod is None:
            msg(self, APP_NAME, "Results along a line are for body and 2D section results – for members, internal force diagrams are shown directly in 3D (quantities Members – N, V, M) and in the Members tab.")
            return
        if self._path_dlg is None or self._path_dlg.win is not self:
            self._path_dlg = PathDialog(self)
        else:
            self._path_dlg.refresh()
        self._path_dlg.show()
        self._path_dlg.raise_()
        if STAGES[self.stage] != "Results":
            self.set_stage(STAGE["Results"])
        self._path_dlg.draw_line()

    def open_section_force_dialog(self):
        if not self.results or not self.results.steps:
            msg(self, APP_NAME, "First solve the study.")
            return
        from .section_dialog import SectionForceDialog
        dlg = SectionForceDialog(self)
        dlg.exec()

    def open_plasticity_dialog(self):
        if not self.results or not self.results.steps:
            msg(self, APP_NAME, "First solve the study.")
            return
        from .plasticity_dialog import PlasticityDialog
        PlasticityDialog(self).exec()

    def open_criteria_selection_dialog(self):
        if not getattr(self, "parts", None):
            msg(self, APP_NAME, "First load the STEP geometry.")
            return
        from .selection_dialog import WorksheetDialog
        dlg = WorksheetDialog(self)
        dlg.exec()

    def refresh_r_csys(self):
        if not hasattr(self, "r_csys"):
            return
        cur = self.r_csys.currentData()
        self.r_csys.blockSignals(True)
        self.r_csys.clear()
        self.r_csys.addItem("Global Cartesian", None)
        if self.study and self.study.coordinate_systems:
            for k, cs in self.study.coordinate_systems.items():
                self.r_csys.addItem(f"{k} ({cs.kind})", k)
        k = self.r_csys.findData(cur)
        if k >= 0:
            self.r_csys.setCurrentIndex(k)
        self.r_csys.blockSignals(False)

    def open_convergence_dialog(self):
        if self.study is None:
            return
        if self.frame_only() or self.study.twod.enabled or self.study.shell.enabled:
            msg(self, APP_NAME, "Mesh convergence is for 3D bodies – for a frame, increase the number of elements per member (step Members → Assessment); for a 2D section and shells, reduce the element size and compare the results.")
            return
        from .convergence_dialog import ConvergenceDialog
        if self._conv_dlg is None:
            self._conv_dlg = ConvergenceDialog(self)
        else:
            self._conv_dlg.refresh()
        self._conv_dlg.show()
        self._conv_dlg.raise_()

    def open_voxel_dialog(self):
        """Voxel print detail (fea_core.printing.voxel) – only results with print safety."""
        res = self.results
        if res is None or not any("print_safety" in s.fields for s in res.steps):
            msg(self, APP_NAME, "The voxel detail needs a 3D printing result (a printed part with a print safety factor) – set up printing in the 3D printing step and run the analysis.")
            return
        from .voxel_dialog import VoxelDialog
        if self._voxel_dlg is None or self._voxel_dlg.win is not self:
            self._voxel_dlg = VoxelDialog(self)
        else:
            self._voxel_dlg.refresh()
        self._voxel_dlg.show()
        self._voxel_dlg.raise_()

    def surface_parts_from_step(self) -> list:
        """Plate parts (faces without volume) from the STEP model – fea_core.shell.geometry.read_step_surfaces."""
        if self.step_path is None or not Path(self.step_path).exists():
            return []
        from fea_core.shell.geometry import read_step_surfaces
        try:
            return read_step_surfaces(self.step_path)
        except Exception as e:  # noqa: BLE001
            self.log_msg(f"Cannot load surface parts from STEP: {e}")
            return []

    @_batched
    def load_surface_parts(self) -> int:
        """Adds plate parts from the STEP among the parts of the model (display, edge / face selection for shells). Returns the count."""
        names = {p.name for p in self.parts}
        new = [p for p in self.surface_parts_from_step() if p.name not in names]
        if not new:
            return 0
        self.parts = list(self.parts) + new
        keep_names([p.name for p in self.parts] + ([self.step_path.stem] if self.step_path else []))
        self.surface_names |= {p.name for p in new}
        self.face_cache, self.edge_cache = {}, {}
        self.view.set_parts(self.parts)
        self.resolve_all()
        self.log_msg(f"Surface parts from STEP: {', '.join((p.name for p in new))} (shell model only)")
        self.refresh_all()
        return len(new)

    def solid_parts(self) -> list:
        """Parts for the 3D mesh (without plate parts from the STEP)."""
        return [p for p in self.parts if p.name not in self.surface_names]

    def register_aux(self, t):
        """Dialog thread (convergence, print refinement) – busy() and closing the window wait for it."""
        self._aux_threads.append(t)

    def unregister_aux(self, t):
        if t in self._aux_threads:
            self._aux_threads.remove(t)

    def interpolate(self, vals, p) -> float:
        """Nodal field at point p (undeformed mesh) – shape functions of the element that contains the point."""
        from fea_core.results import tet10_shape
        res = self.results
        e, xi = res.locate(p)
        conn = res.mesh.tets[e]
        N = tet10_shape(xi) if len(conn) == 10 else np.array([1 - xi.sum(), *xi])
        return float(N @ np.asarray(vals)[conn])

    def fill_thermal_info(self, th, frame):
        res = self.results
        m = res.mesh
        T = th.T if frame is None else th.frames[frame]
        i_max, i_min = int(np.argmax(T)), int(np.argmin(T))

        def at(i):
            return ", ".join(fmt(float(c), 5) for c in m.nodes[i])
        rows = [f"<b>{escape(str(th.name))}</b> – " + (f"transient case, time {fmt(float(th.times[frame if frame is not None else -1]))} s"
                                           if th.transient and th.times else "steady state"),
                f"Max. temperature: <b>{fmt(float(T[i_max]))} °C</b> at point ({at(i_max)}) mm",
                f"Min. temperature: <b>{fmt(float(T[i_min]))} °C</b> at point ({at(i_min)}) mm"]
        if frame is None:
            q = th.flux_mag
            rows.append(f"Max. heat flux |q|: <b>{fmt(float(q.max()))} W/m²</b>")
        if self.study is not None:
            rows.append(f"Reference temperature: {fmt(self.study.thermal.reference_temperature)} °C")
        if th.heat:
            rows.append("<br><b>Heat of boundary conditions</b> [W] (+ = into the body"
                        + (", end of the state" if th.transient else "") + "):")
            rows.append("<table cellspacing=3>")
            for h in th.heat:
                rows.append(f"<tr><td>{escape(str(h['name']))}</td><td>{escape(str(KIND_SHORT.get(h['kind'], h['kind'])))}</td>"
                            f"<td align=right><b>{fmt(float(h['Q_W']))}</b></td></tr>")
            rows.append("</table>")
        bal = th.balance or {}
        if "imbalance" in bal:
            rows.append(f"Heat balance: input {fmt(bal['in_W'])} W, output {fmt(bal['out_W'])} W, residual {fmt(bal['residual_W'], 3)} W ({fmt(100 * bal['imbalance'], 2)} %)")
        elif bal.get("note"):
            rows.append(f"Heat balance: {escape(str(bal['note']))}")
        if th.transient and th.times:
            rows.append(f"Snapshots: {len(th.times)} ({fmt(float(th.times[0]))} … {fmt(float(th.times[-1]))} s); flux only at the end of the case.")
        warn = [escape(str(w)) for w in res.warnings]
        if warn:
            rows.append("<br><span style='color:#b36b00'>! " + "<br>! ".join(warn) + "</span>")
        self.res_info.setHtml(self._join_rows(rows))

    @staticmethod
    def _join_rows(rows):
        """HTML lines separated by <br>; a table (<table>…</table>) without a break between rows."""
        out, in_tab = [], False
        for r in rows:
            if r.startswith("<table"):
                in_tab = True
                out.append(r)
            elif r == "</table>":
                in_tab = False
                out.append(r)
            elif in_tab:
                out.append(r)
            else:
                out.append(r + "<br>")
        return "".join(out)

    def fill_result_info(self, st, scale, imax, vals):
        res = self.results
        m = res.mesh
        vm = st.mises
        i_vm = int(np.argmax(vm))
        rows = [f"<b>{escape(str(st.name))}</b>"]
        if st.partial:                              # non-convergence – last converted increment (F8, 4.9)
            rows.append("<span style='color:#d23b3b'><b>PARTIAL RESULT</b> – the calculation did not converge, "
                        "the last converted increment is shown (advice in the Messages dock)</span>")
        if st.contacts:
            rows.append("Contacts: " + "; ".join(f"{escape(str(n))} Fn {fmt(c.get('Fn_N'))} N, p_max {fmt(c.get('p_max_MPa'))} MPa"
                                                 for n, c in list(st.contacts.items())[:4])
                        + " (Contacts tab)")
        rows += [f"Max. displacement: <b>{fmt(float(st.umag.max()))} mm</b>",
                f"Max. von Mises: <b>{fmt(float(vm.max()))} MPa</b> at point ({', '.join((fmt(float(c), 5) for c in m.nodes[i_vm]))}) mm"]
        # classification of the maximum (fea_core.mcp.diagnostics – same as MCP and the report)
        cur = self.current_result()
        hs = self.result_hotspots("step", cur[1], "mises") if cur and cur[0] == "step" else None
        top = hs["hotspots"] if hs else []
        sing = bool(top) and top[0].get("classification") == "singularity"
        if top and top[0].get("classification") != "probably real concentration":
            rows[-1] += f" – <span style='color:#b36b00'>{escape(self.hotspot_label(top[0]))}</span>"
        if sing:
            ns = next((h for h in top if h.get("classification") != "singularity"), None)
            rows.append(f"Max. von Mises excluding singularities: <b>{fmt(ns['mises_MPa'])} MPa</b> at point {self._pt(ns['xyz_mm'], 5)} mm ({escape(self.hotspot_label(ns))})" if ns else
                        f"Max. von Mises excluding singularities: all {len(top)} highest locations are singularities – evaluate away from them (≥ 2–3 elements), in a section, or by nominal stress.")
        re = self.node_re()
        if np.isfinite(re).any():
            sf = float(np.min(re / np.maximum(vm, 1e-9)))
            rows.append(f"Min. safety factor to Re: <b>{fmt(sf, 3)}</b>")
        topo = (res.info or {}).get("topology")
        if topo:
            c0, c1 = topo.get("initial_compliance_Nmm"), topo.get("final_compliance_Nmm")
            comp = f", compliance full design → result: {fmt(c0, 4)} → {fmt(c1, 4)} N·mm (×{fmt(c1 / c0, 3)})" \
                if c0 and c1 else ""
            rows.append(f"Topology optimization (BESO): {topo.get('iterations', 0)} iterations, volume {fmt(100 * float(topo.get('final_volume_fraction', 0)), 3)} % of the original{comp} (quantities Topology – density / final shape)")
        T = st.fields.get("T")
        if T is not None:
            rows.append(f"Thermal load: T {fmt(float(np.min(T)))} … {fmt(float(np.max(T)))} °C (stress-free at {fmt(self.study.thermal.reference_temperature)} °C)")
        if st.printing:
            rows.append(f"<br><b>3D printing</b> – criterion {escape(str(st.printing.get('criterion', '')))} (print axes):")
            for pname, d in st.printing.get("parts", {}).items():
                sl = ", ".join(fmt(float(v), 3) for v in d["stress_local"])
                rows.append(f"{escape(str(pname))}: min. safety factor <b>{fmt(d['min_safety'], 3)}</b> – {escape(str(d['region']))} at point ({', '.join((fmt(float(c), 4) for c in d['at']))}) mm; σ11 σ22 σ33 τ12 τ13 τ23 = ({sl}) MPa")
            rows.append("")
        rows.append(f"Deformation scale: {fmt(scale, 3)}×")
        rows.append("<br><b>Reactions</b> [N]:")
        for k, r in st.reactions.items():
            rows.append(f"{escape(str(k))}: Fx {fmt(float(r[0]))}, Fy {fmt(float(r[1]))}, Fz {fmt(float(r[2]))}")
            rem = st.remote.get(k)
            if rem:                                   # remote constraint: moment about the constraint point
                mm = rem["moment"]
                rows.append(f"&nbsp;&nbsp;moment about point ({'; '.join((fmt(float(x)) for x in rem['point']))}) mm [N·mm]: Mx {fmt(float(mm[0]))}, My {fmt(float(mm[1]))}, Mz {fmt(float(mm[2]))}")
        eq = st.equilibrium
        if eq and "imbalance" in eq:              # a combination carries only the total reaction (no equilibrium check)
            rows.append(f"Equilibrium check: imbalance {fmt(100 * eq['imbalance'], 2)} %")
        # warning for singularities
        warn = ["Stress maxima at sharp internal corners, point loads and on support edges are singularities – do not evaluate by the nodal maximum; refine the mesh and monitor convergence."]
        if sing:
            warn.insert(0, f"<b>The maximum von Mises stress is a singularity ({escape(', '.join(top[0].get('tags') or []) or 'see table')}) – see the “Highest stresses” tab.</b>")
        elif hs is None:                          # classification failed → at least the test of the constraint face
            cons_nodes = set()
            for c in self.study.constraints:
                if c.selection in m.surfaces:
                    cons_nodes.update(np.unique(m.surfaces[c.selection]).tolist())
            if i_vm in cons_nodes:
                warn.insert(0, "<b>The maximum von Mises stress lies on a support face – probably a support singularity.</b>")
        for w in res.warnings:
            warn.append(escape(str(w)))
        rows.append("<br><span style='color:#b36b00'>! " + "<br>! ".join(warn) + "</span>")
        self.res_info.setHtml("<br>".join(rows))

    def on_probe(self, xyz):
        hook = self._pick_hook                     # point picked by the dialog (progress along a line, convergence)
        if hook is not None and self.results is not None:
            p = self.undeformed(xyz)
            self.view.mark(xyz, "(" + "; ".join(fmt(float(c), 5) for c in p) + ")", (20, 120, 220), key="pick")
            hook(p)
            return
        if getattr(self, "_probe", None) is None or self.results is None:
            return
        tree, ids, vals, unit, name = self._probe
        exact = getattr(self, "_probe_exact", None)
        if exact is not None:                  # thermal state: interpolation in the element (Results.probe)
            try:
                v = exact(np.asarray(xyz, float))
            except Exception:  # noqa: BLE001 – outside the mesh → nearest node below
                v = None
            if v is not None and np.isfinite(v):
                txt = f"{fmt(v)} {unit}"
                self.view.mark(xyz, txt, (20, 20, 20))
                self.st_info.setText(f"Probe: {name} = {txt} at point ({', '.join((fmt(float(c), 5) for c in xyz))}) mm")
                return
        d, j = tree.query(xyz)
        n = int(ids[j])
        txt = f"{fmt(float(vals[j]))} {unit}"
        self.view.mark(tree.data[j], txt, (20, 20, 20))
        labels = getattr(self, "_probe_labels", None)
        if labels is not None and j < len(labels) and labels[j][0]:
            self.st_info.setText(f"Probe: {name} = {txt} – member {labels[j][0]}, s = {fmt(float(labels[j][1]), 5)} mm")
            return
        if n >= len(self.results.mesh.nodes):
            n = -1
        if n >= 0:
            x = self.results.mesh.nodes[n]
            self.st_info.setText(f"Probe: {name} = {txt} at node {n + 1} "
                                 f"({', '.join(fmt(float(c), 5) for c in x)}) mm")
        else:
            self.st_info.setText(f"Probe: {name} = {txt} in section ({', '.join((fmt(float(c), 5) for c in tree.data[j]))}) mm")

    # ================================================================== maximum places (singularities)
    def diag_session(self):
        """MCP diagnostics session over the GUI state: maximum places are classified by the same functions as in MCP
        (fea_results_hotspots) and in the report – fea_core.mcp.diagnostics.hotspots / thermal_hotspots.
        """
        from fea_core.mcp.session import Session
        if self._diag_sess is None:
            self._diag_sess = Session(self._workdir / "diag")
        s = self._diag_sess
        s.study, s.results, s._parts = self.study, self.results, self.parts
        s.mesh = self.results.mesh if self.results is not None else None
        s.fatigue = self.fat_panel.result
        return s

    def result_hotspots(self, kind, i, key, time_s=None):
        """Maximum places (cache on the results object): kind "step" → hotspots with labels, "thermal" →
        thermal_hotspots. A heuristic error → None (written to the log), the results are shown anyway.
        """
        res = self.results
        if res is None:
            return None
        cache = getattr(res, "_gui_hotspots", None)
        if cache is None:
            cache = res._gui_hotspots = {}
        k = (kind, i, key, time_s)
        if k not in cache:
            from fea_core.mcp import diagnostics as dg
            try:
                sess = self.diag_session()
                cache[k] = (dg.thermal_hotspots(sess, i, HS_TOP, key, time_s) if kind == "thermal"
                            else dg.hotspots(sess, i, HS_TOP, key, tags=True))
            except Exception as e:  # noqa: BLE001 – the heuristic must not block the display of results
                self.log_msg(f"Could not evaluate the maximum locations: {e}")
                cache[k] = None
        return cache[k]

    @staticmethod
    def hotspot_label(h) -> str:
        """Short classification of a place: "singularity – clamped edge", "boundary condition – …", "real place";
        for temperature (no classification) a boundary condition at the place.
        """
        cls = h.get("classification")
        tags = h.get("tags") or []
        if cls == "singularity":
            return "singularity – " + ", ".join(tags) if tags else "singularity"
        if cls == "affected by boundary condition":
            return "boundary condition – " + ", ".join(tags) if tags else cls
        if cls:
            return "actual location" if cls.startswith("probably real") else cls
        return h["reasons"][0] if h.get("reasons") else "–"

    def clear_hotspots(self):
        self._hs_rows = []
        self.hs_table.blockSignals(True)
        self.hs_table.setRowCount(0)
        self.hs_table.blockSignals(False)
        self.hs_note.setText("")
        self.hs_title.setText("<b>Highest stresses</b>")
        self.res_tabs.setTabText(1, "Highest stresses")
        self.res_tabs.setTabToolTip(1, "")
        self.legend.set_note("")

    def fill_hotspots(self, kind, i, name, unit, vals, X, frame=None):
        """Table of maximum places of the displayed quantity (stress quantities; other mechanical → von Mises; heat →
        hottest places / heat flux). Returns (note for the legend, point of a singular maximum | None, cluster radius);
        the note and the point only if the table belongs to the displayed quantity.
        """
        self.clear_hotspots()
        if kind == "thermal":
            key = HS_THERMAL.get(name, "temperature")
            th = self.results.thermal[i]
            t = float(th.times[frame]) if frame is not None and key == "temperature" else None
            direct = True
            title = "Hottest locations" if key == "temperature" else f"Highest values – {name[:1].lower()}{name[1:]}"
        else:
            key = HS_STRESS.get(name, "mises")
            t = None
            direct = name in HS_STRESS
            title = "Highest stress – " + ("von Mises" if key == "mises" else name.replace("Stress ", "")
                                           + (" (absolute)" if key != "s1" else ""))
        self.hs_title.setText(f"<b>{title}</b>")
        tab_name = title.split(" – ")[0]
        self.res_tabs.setTabText(1, tab_name)
        d = self.result_hotspots(kind, i, key, t)
        if not d:
            return "", None, 0.0
        hs = d["hotspots"]
        u = unit if direct else d["unit"]
        tab = self.hs_table
        tab.blockSignals(True)
        tab.setRowCount(len(hs))
        tab.horizontalHeaderItem(1).setText(f"value [{u}]")
        for r, h in enumerate(hs):
            n = h["node"] - 1
            v = float(vals[n]) if direct else float(h["value"])
            cls = h.get("classification", "")
            tip = "\n".join([x for x in [cls, *h.get("reasons", []), h.get("recommendation", "")] if x])
            for c, txt in enumerate((str(h["rank"]), fmt(v), ", ".join(h["parts"]), self._pt(h["xyz_mm"], 4),
                                     self.hotspot_label(h))):
                it = QTableWidgetItem(txt)
                it.setToolTip(tip)
                if c == 1:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                tab.setItem(r, c, it)
            col = {"singularity": ORANGE, "affected by boundary condition": GREY}.get(cls, GREEN if cls else None)
            if col:
                tab.item(r, 4).setForeground(QColor(col))
            self._hs_rows.append((h, v, u))
        tab.blockSignals(False)
        self._hs_X = X
        self._hs_name = name if direct else "von Mises"
        note, sing_xyz = "", None
        if hs and hs[0].get("classification") == "singularity":
            ns = next((k for k, row in enumerate(self._hs_rows) if row[0].get("classification") != "singularity"), None)
            # two sentences are translated separately (the catalogue has each as a separate template)
            note = tr(f"Max. {fmt(self._hs_rows[0][1])} {u} is a singularity "
                      f"({', '.join(hs[0].get('tags') or []) or 'see table'}) – do not judge by the node value.")
            note += tr(f" Max. outside singularities: {fmt(self._hs_rows[ns][1])} {u} (location #{hs[ns]['rank']})."
                       if ns is not None else f" All {len(hs)} highest locations are singularities.")
            sing_xyz = np.asarray(hs[0]["xyz_mm"], float)
        nsing = sum(1 for h in hs if h.get("classification") == "singularity")
        if nsing:
            self.res_tabs.setTabText(1, tr(tab_name) + f" ! {nsing}")   # translate the name separately
            self.res_tabs.setTabToolTip(1, f"Singularities among the {len(hs)} highest locations: {nsing}")
        info = []
        if note:
            info.append(f"<span style='color:#b36b00'>! {escape(note)}</span>")
        elif any(h.get("classification") for h in hs):
            info.append(f"Singularities among the {len(hs)} highest locations: {nsing}.")
        if d.get("coldest"):
            cd = d["coldest"]
            info.append(f"Coldest location: {fmt(cd['T_C'])} °C at point {self._pt(cd['xyz_mm'], 4)} mm ({escape(', '.join(cd['parts']))}).")
        info.append("<i>Clicking a row marks the location in 3D; classification reasons in the tooltip.</i>")
        self.hs_note.setText("<br>".join(info))
        if not direct:
            return "", None, 0.0
        return note, sing_xyz, float(d.get("cluster_radius_mm", 0.0))

    def on_hotspot_row(self, r):
        """Row of the maximum-places table: marker in 3D (deformed position), value as a probe, reasons of the classification."""
        import html
        if not (0 <= r < len(self._hs_rows)) or self.results is None:
            return
        h, v, u = self._hs_rows[r]
        n = h["node"] - 1
        X = getattr(self, "_hs_X", None)
        pos = X[n] if X is not None and n < len(X) else h["xyz_mm"]
        sing = h.get("classification") == "singularity"
        self.view.mark(pos, f"#{h['rank']}: {fmt(v)} {u}", (200, 100, 0) if sing else (20, 20, 20), key="hotspot",
                       below=True)
        label = self.hotspot_label(h)
        self.st_info.setText(f"Probe: {self._hs_name} = {fmt(v)} {u} at node {n + 1} {self._pt(h['xyz_mm'], 5)} mm – location #{h['rank']}: {label}")
        rows = [f"<b>#{h['rank']}: {html.escape(label)}</b>"] + [f"• {html.escape(x)}" for x in h.get("reasons", [])]
        if h.get("recommendation"):
            rows.append(f"<i>{html.escape(h['recommendation'])}</i>")
        self.hs_note.setText("<br>".join(rows))

    # ================================================================== restoring the view
    @_batched
    def refresh_all(self):
        self._filling = True
        try:
            check_result = self.check() if self.study is not None else None
            self.refresh_tree(check_result)
            self.refresh_flow(check_result)
            if self.study is None:
                return
            self.apply_ghost()
            self.refresh_model_info()
            self.refresh_materials()
            for attr in ("print_panel", "frame_panel"):
                if self._built(attr) is not None:
                    self._built(attr).refresh()
            self.joints_panel.refresh()
            self.twod_box.refresh()
            self.shell_box.refresh()
            self.cyclic_box.refresh()
            self.thermal_panel.refresh()
            for attr in ("contact_panel", "dynamics_panel", "motion_panel"):
                if self._built(attr) is not None:
                    self._built(attr).refresh()
            self.refresh_thermal_load_form()
            self.con_list.clear()
            for c in self.study.constraints:
                self.con_list.addItem(self.describe_constraint(c))
            cur = self.lc_combo.currentIndex()
            self.lc_combo.blockSignals(True)
            self.lc_combo.clear()
            self.lc_combo.addItems([lc.name for lc in self.study.load_cases])
            self.lc_combo.setCurrentIndex(min(max(cur, 0), len(self.study.load_cases) - 1))
            self.lc_combo.blockSignals(False)
            self.refresh_loads(refresh_view=False)
            ms = self.study.mesh
            self.ms_size.setValue(ms.size)
            self.ms_min.setValue(ms.min_size or 0.0)
            self.ms_curv.setValue(ms.curvature_points)
            self.ref_list.clear()
            for r in ms.local:
                self.ref_list.addItem(self.describe_refinement(r))
            if self.mesh is not None:
                r = self.mesh.report
                ties = r.get("ties") or []
                self.mesh_info.setText(f"{fmt(r.get('nodes', len(self.mesh.nodes)))} nodes, {fmt(r.get('elements', len(self.mesh.tets)))} C3D10 elements, min. scaled Jacobian {fmt(r.get('min_scaled_jacobian'), 3)}"
                                       + (f"; *TIE connections: {len(ties)} (Joints step)" if ties else ""))
            else:
                self.mesh_info.setText("The mesh has not been generated yet (it is also generated during the analysis).")
            self.solver_combo.blockSignals(True)
            self.solver_combo.setCurrentText(self.study.solver.solver)
            self.solver_combo.blockSignals(False)
            self.backend_combo.blockSignals(True)
            bi = self.backend_combo.findData(self.study.solver.backend)
            if bi < 0:                                  # unknown solver (file from another version) → automatically
                self.log_msg(f"Unknown solver “{self.study.solver.backend}” in the study – set to Automatic")
                self.study.solver.backend = "auto"
                bi = max(0, self.backend_combo.findData("auto"))
                check_result = None                     # the study changed: refresh_ready checks it again
            self.backend_combo.setCurrentIndex(bi)
            self.backend_combo.blockSignals(False)
            self._backend_badge()                       # signals are blocked → mark the Solver section manually
            self.eigen_box.refresh()
            cur = self.r_step.currentData()
            self.r_step.blockSignals(True)
            self.r_step.clear()
            if self.results is not None:            # mechanical states first (indices as results.steps), then heat
                for i, s in enumerate(self.results.steps):
                    self.r_step.addItem(s.name, f"step:{i}")
                if hasattr(self.study, "combinations") and self.study.combinations:
                    for j, comb in enumerate(self.study.combinations):
                        prefix = "Envelope" if comb.envelope else "Combination"
                        self.r_step.addItem(f"{prefix}: {comb.name}", f"combination:{j}")
                for j, t in enumerate(self.results.thermal):
                    self.r_step.addItem(f"Thermal: {t.name}", f"thermal:{j}")
                if self.results.twod is not None:          # 2D calculation by a section
                    for j, s2 in enumerate(self.results.twod.steps):
                        self.r_step.addItem(f"2D: {s2['name']}", f"twod:{j}")
                if self.results.shell is not None:         # shell model
                    for j, s3 in enumerate(self.results.shell.steps):
                        self.r_step.addItem(f"Shells: {s3.name}", f"shell:{j}")
                for txt, data in smd.r_step_items(self.results):   # submodels ("sub:k")
                    self.r_step.addItem(txt, data)
                for j, ms in enumerate(self.results.modes):  # natural and buckling mode shapes (eigen.ModeSet)
                    for k in range(len(ms.values)):
                        self.r_step.addItem(mode_label(ms, k), f"mode:{j}:{k}")
                for txt, data in dynr.items(self.results):   # dynamika (F8): pole „dyn:druh:k“
                    self.r_step.addItem(txt, data)
                k = self.r_step.findData(cur) if cur is not None else -1
                self.r_step.setCurrentIndex(max(k, 0))
            self.r_step.blockSignals(False)
            if hasattr(self, "r_part"):
                cur_part = self.r_part.currentText()
                self.r_part.blockSignals(True)
                self.r_part.clear()
                self.r_part.addItem("All parts", None)
                if self.results is not None and self.results.mesh is not None:
                    for pname in self.results.mesh.part_names:
                        self.r_part.addItem(pname, pname)
                idx = self.r_part.findText(cur_part)
                self.r_part.setCurrentIndex(max(idx, 0))
                self.r_part.blockSignals(False)
            if self.results is not None:
                self._sync_result_controls()
            self.modes_table.fill(self.results)
            self.res_tabs.setTabVisible(2, bool(self.results is not None and self.results.modes))
            self.res_tabs.setTabVisible(3, bool(self.results is not None and any(s.frame for s in self.results.steps)))
            has_dyn = self.results is not None and getattr(self.results, "dynamics", None) is not None
            self.res_tabs.setTabVisible(self.TAB_DYN, has_dyn)
            if has_dyn:
                self.dyn_box.fill(self.results)
            self.res_tabs.setTabVisible(self.TAB_CONTACT, bool(self.results is not None and any(
                s.contacts or s.partial or s.curves for s in self.results.steps)))
            self.refresh_sets()
            self.refresh_ready(check_result)
            self.update_status()
        finally:
            self._filling = False
            if hasattr(self, "table_dock"):
                try:
                    if STAGES[getattr(self, "stage", 0)] == "Results" or self.results is None:
                        self.res_overview.refresh()      # Assessment card (state summary only when visible)
                    self.table_dock.refresh()            # table of the current step in the bottom dock
                    self.update_model_status()
                    self.apply_part_colors()
                    self.update_view_info()
                    self.props_card.schedule()
                except Exception as e:  # noqa: BLE001 – the table is only an overview, must not block the window
                    self.log_msg(f"The table in the bottom dock could not be refreshed: {e}")

    def refresh_loads(self, refresh_view: bool = True):
        self.load_list.clear()
        lc = self.current_lc()
        if lc:
            for l in lc.loads:
                self.load_list.addItem(self.describe_load(l))
        if refresh_view:
            self.refresh_sets()

    def refresh_sets(self):
        """Colour sets of the constraint and load faces of the current state in 3D (in the Joints step bolts and welds)."""
        if self.study is None:
            return
        v = self.view
        v.clear_sets()
        stage = STAGES[self.stage]
        if stage not in ("Joints", "Contacts", "Constraints", "Motion", "Heat", "Loads", "Dynamics", "Mesh", "Solve"):
            v.show_sets(False)
            return
        if stage != "Results" or self.results is None:              # frame system in all steps with a model
            self.frame_panel.show_3d(self.current_lc() if stage == "Loads" else None)
        if stage == "Solve":
            self.twod_box.draw_preview()
        if STAGES[self.stage] == "Joints":
            self.joints_panel.show_3d()
            v.show_sets(True)
            return
        if STAGES[self.stage] == "Heat":          # boundary conditions of the selected thermal state
            self.thermal_panel.show_3d()
            v.show_sets(True)
            return
        if STAGES[self.stage] == "Contacts":       # contact pair faces (colour by type)
            self.contact_panel.show_3d()
            v.show_sets(True)
            return
        if STAGES[self.stage] == "Motion":          # joints (axes), spring elements, mechanism loads
            self.motion_panel.show_3d()
            v.show_sets(True)
            return
        if STAGES[self.stage] == "Dynamics":       # added mass and output probes
            self.dynamics_panel.show_3d()
            v.show_sets(True)
            return
        L = 0.12 * v.model_size()
        for c in self.study.constraints:
            v.set_face_set(f"c:{c.selection}", self.sel_faces.get(c.selection, []), C_CONSTRAINT,
                           label=c.selection)
            if isinstance(c, CylindricalSupport):
                segs = []                             # axis of each selection hole (extended beyond the face)
                for p0, d, r, t0, t1, _ in self.cylinders_of(self.sel_faces.get(c.selection, [])):
                    m = 0.25 * (t1 - t0) + r
                    segs.append((p0 + (t0 - m) * d, p0 + (t1 + m) * d))
                if segs:
                    v.add_lines(f"c:{c.selection}", segs, C_CONSTRAINT, 2.5)
            elif isinstance(c, RemoteSupport):        # point (cross) + lines to the faces
                pt = np.asarray(c.point, float)
                k = 0.04 * L
                seg = [(pt - k * e, pt + k * e) for e in np.eye(3)]
                for pi, fi in self.sel_faces.get(c.selection, [])[:12]:
                    p = self.face_prop(pi, fi)
                    seg.append((pt, np.asarray(self.arrow_point(p), float)))
                v.add_lines(f"c:{c.selection}", seg, C_CONSTRAINT, 2.0, label=f"point {c.selection}")
            elif isinstance(c, ElasticSupport):       # springs from the face outwards (normal; "all directions" + cross)
                seg = []
                for pi, fi in self.sel_faces.get(c.selection, [])[:12]:
                    p = self.face_prop(pi, fi)
                    seg += self.spring_segments(self.arrow_point(p), p["normal"], 0.35 * L, c.direction == "all")
                v.add_lines(f"c:{c.selection}", seg, C_ELASTIC, 2.0)
        if stage == "Constraints":                    # cyclic symmetry: cut faces and axis
            from .cyclic_gui import show_3d as show_cyclic
            show_cyclic(self)
        lc = self.current_lc()
        if lc:
            from fea_core import csys as csys_mod
            from fea_core.loads import pressure_value

            def gvec(l, at):
                """Global direction of a force at a point (local / cylindrical system)."""
                try:
                    return csys_mod.field_global(self.study, getattr(l, "csys", None), l.vector,
                                                 np.asarray(at, float)[None, :])[0]
                except ValueError:
                    return np.asarray(l.vector, float)
            for l in lc.loads:
                sel = getattr(l, "selection", None)
                if isinstance(l, Rotation):              # rotation axis across the model
                    a = np.asarray(l.axis, float) / max(np.linalg.norm(l.axis), 1e-30)
                    p0 = np.asarray(l.point, float)
                    bb = np.array([bbox(p.shape) for p in self.parts])
                    c0 = 0.5 * (bb[:, :3].min(axis=0) + bb[:, 3:].max(axis=0))
                    t0 = float((c0 - p0) @ a)
                    half = 0.6 * v.model_size()
                    v.add_lines("l:rotation", [(p0 + (t0 - half) * a, p0 + (t0 + half) * a)], C_EDGE, 2.5,
                                label=self.describe_load(l))
                    v.add_arrows("l:rotation", [(p0 + (t0 + half) * a, a, 0.4 * L)], C_EDGE, double=True)
                    continue
                if not sel:
                    continue
                entity = self.study.selections[sel].entity if sel in self.study.selections else "faces"
                key = f"l:{sel}"
                if entity == "edges":                      # force on an edge: arrows along the edges
                    pts = []
                    for pi, ei in self.sel_edges.get(sel, []):
                        e = self.edge_prop(pi, ei)
                        pts += [e["start"], e["point"], e["end"]]
                    if pts:
                        v.add_lines(key, [(pts[i], pts[i + 1]) for i in range(0, len(pts), 3)]
                                    + [(pts[i + 1], pts[i + 2]) for i in range(0, len(pts), 3)], C_EDGE, 4.0,
                                    label=self.describe_load(l))
                        if np.linalg.norm(l.vector) > 0:
                            v.add_arrows(key, [(q, gvec(l, q), 0.8 * L) for q in pts[:24]], C_FORCE)
                    continue
                if entity == "points":                     # force to a point
                    pts = [q for _, q in self.sel_points.get(sel, [])]
                    if pts and np.linalg.norm(l.vector) > 0:
                        v.add_arrows(key, [(q, gvec(l, q), L) for q in pts], C_FORCE)
                    if pts:                                # cross at the point (with a label at the first one)
                        k = 0.04 * L
                        v.add_lines(key, [(np.add(q, -k * e), np.add(q, k * e)) for q in pts for e in np.eye(3)],
                                    C_FORCE, 2.0, label=self.describe_load(l))
                    continue
                faces = self.sel_faces.get(sel, [])
                col = C_PRESSURE if isinstance(l, Pressure) else C_FORCE
                v.set_face_set(key, faces, col, label=self.describe_load(l))
                props = [self.face_prop(pi, fi) for pi, fi in faces[:8]]
                if isinstance(l, Force) and np.linalg.norm(l.vector) > 0:
                    arr = []
                    cs = self.study.coordinate_systems.get(l.csys) if l.csys else None
                    if cs is not None and cs.kind == "cylindrical":    # arrows at points far from the axis
                        for pi, fi in faces[:4]:
                            for q, nq in self.face_points_off_axis(pi, fi, cs, 6):
                                d = gvec(l, q)
                                arr.append((v.place_on_face(q, d, nq, 0.7 * L), d, 0.7 * L))
                    else:
                        for p in props:
                            q = self.arrow_point(p)
                            d = gvec(l, q)
                            arr.append((v.place_on_face(q, d, p["normal"], L), d, L))
                    v.add_arrows(key, arr, col)
                elif isinstance(l, Pressure) and l.variable:
                    arr = []
                    for p in props:
                        q = self.arrow_point(p)
                        pv = float(pressure_value(l, np.asarray(q, float)[None, :])[0])
                        if pv != 0:
                            s_ = np.sign(pv)
                            arr.append((v.place_on_face(q, -s_ * np.asarray(p["normal"]), p["normal"], L),
                                        -s_ * np.asarray(p["normal"]), L))
                    v.add_arrows(key, arr, col)
                elif isinstance(l, Pressure) and l.value != 0:
                    s = np.sign(l.value)       # positive pressure points into the material
                    v.add_arrows(key, [(v.place_on_face(self.arrow_point(p), -s * p["normal"], p["normal"], L),
                                        -s * p["normal"], L) for p in props], col)
                elif isinstance(l, BearingLoad) and np.linalg.norm(l.vector) > 0:
                    v.add_arrows(key, self.bearing_arrows(faces, l.vector, L), col)
                elif isinstance(l, RemoteLoad):
                    if np.linalg.norm(l.force) > 0:
                        v.add_arrows(key, [(l.point, l.force, 1.3 * L)], col)
                    if np.linalg.norm(l.moment) > 0:
                        v.add_arrows(key, [(l.point, l.moment, 1.3 * L)], col, double=True)
        if STAGES[self.stage] == "Loads" and lc is not None:      # local systems used in the state
            used = {getattr(l, "csys", None) for l in lc.loads} - {None}
            for name in used:
                c = self.study.coordinate_systems.get(name)
                if c is None:
                    continue
                from fea_core.csys import matrix
                R = matrix(c)
                o = np.asarray(c.origin, float)
                k = 0.5 * L
                v.add_lines(f"cs:{name}", [(o, o + k * R[i]) for i in range(3)], (90, 90, 90), 2.0,
                            label=f"{name} ({'r, θ, z' if c.kind == 'cylindrical' else 'x, y, z'})")
        if STAGES[self.stage] == "Mesh":
            for k, r in enumerate(self.study.mesh.local):
                if r.get("selection"):
                    v.set_face_set(f"r:{r['selection']}", self.sel_faces.get(r["selection"], []), C_REFINE)
                elif r.get("center") is not None:          # sphere: three circles
                    c, R = np.asarray(r["center"], float), float(r.get("radius", 0.0))
                    t = np.linspace(0, 2 * np.pi, 49)
                    seg = []
                    for a, b in ((0, 1), (1, 2), (0, 2)):
                        P = np.tile(c, (len(t), 1))
                        P[:, a] += R * np.cos(t)
                        P[:, b] += R * np.sin(t)
                        seg += [(P[i], P[i + 1]) for i in range(len(t) - 1)]
                    v.add_lines(f"r:ball{k}", seg, C_REFINE, 2.0, label=f"refinement {fmt(float(r['size']))} mm")
        v.show_sets(STAGES[self.stage] in ("Constraints", "Loads", "Solve") or
                    (STAGES[self.stage] == "Mesh" and self.mesh is None))

    @staticmethod
    def spring_segments(base, normal, L, all_dirs=False) -> list[tuple]:
        """Spring marker: zigzag from a point of the face in the direction of the outward normal and a "ground" dash at the end."""
        n = np.asarray(normal, float)
        n = n / max(np.linalg.norm(n), 1e-30)
        e = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        e /= np.linalg.norm(e)
        b = np.asarray(base, float)
        pts = [b, b + 0.2 * L * n]
        for k in range(6):
            pts.append(b + (0.2 + 0.1 * (k + 0.5)) * L * n + (0.12 if k % 2 == 0 else -0.12) * L * e)
        pts += [b + 0.8 * L * n, b + L * n]
        seg = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        tip = b + L * n
        seg.append((tip - 0.2 * L * e, tip + 0.2 * L * e))
        if all_dirs:                                  # springs also in the plane of the face
            e2 = np.cross(n, e)
            seg += [(b - 0.25 * L * e, b + 0.25 * L * e), (b - 0.25 * L * e2, b + 0.25 * L * e2)]
        return seg

    def face_points_off_axis(self, pi, fi, cs, n: int = 6) -> list[tuple[np.ndarray, np.ndarray]]:
        """Face points (triangulation nodes) far from the axis of a cylindrical system, spread out (the furthest points);
        the face normal at them roughly from the triangulation. For force arrows in a cylindrical system (there is no direction on the axis).
        """
        from OCP.BRep import BRep_Tool
        from OCP.TopLoc import TopLoc_Location
        from fea_core.csys import matrix
        face = self.view.part_faces[pi][fi]
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            p = self.face_prop(pi, fi)
            return [(np.asarray(p["point"], float), np.asarray(p["normal"], float))]
        trsf = loc.Transformation()
        X = np.array([[q.X(), q.Y(), q.Z()] for q in (tri.Node(i).Transformed(trsf)
                                                       for i in range(1, tri.NbNodes() + 1))])
        ez = matrix(cs)[2]
        d = X - np.asarray(cs.origin, float)
        r = np.linalg.norm(d - np.outer(d @ ez, ez), axis=1)
        cand = np.nonzero(r >= 0.6 * r.max())[0]
        pick = [int(cand[np.argmax(r[cand])])]
        dmin = np.linalg.norm(X[cand] - X[pick[0]], axis=1)
        for _ in range(min(n, len(cand)) - 1):
            pick.append(int(cand[np.argmax(dmin)]))
            dmin = np.minimum(dmin, np.linalg.norm(X[cand] - X[pick[-1]], axis=1))
        nrm = np.asarray(self.face_prop(pi, fi)["normal"], float)
        out = []
        for k in pick:                       # normal: planar face → same; cylinder → radial (by type)
            nq = nrm
            if self.face_prop(pi, fi)["type"] != "plane":
                rv = d[k] - (d[k] @ ez) * ez
                nq = rv / max(np.linalg.norm(rv), 1e-30) * (1 if rv @ (X[k] - np.asarray(cs.origin)) >= 0 else -1)
            out.append((X[k], nq))
        return out

    def cylinder_of(self, faces):
        """(point on the axis, direction, radius, t_min, t_max, hole?) of the first group of coaxial cylindrical CAD faces of the selection
        (see cylinders_of); None if the selection has no cylinder.
        """
        groups = self.cylinders_of(faces)
        return groups[0] if groups else None

    def cylinders_of(self, faces) -> list[tuple]:
        """Groups of coaxial cylindrical faces of the selection with the same radius (like fea_core.cylinder.selection_axes – a selection
        with several non-coaxial holes has the axis of each hole): [(point on the axis, direction, radius, t_min, t_max, hole?)], t = position
        along the axis across the faces of the group (from the bounding boxes).
        """
        from fea_core.geometry import cylinder_axis
        groups: list[list] = []                  # [axis (p0, d, r), props of the first face, [t…]]
        for pi, fi in faces:
            ax = cylinder_axis(self.view.part_faces[pi][fi])
            if ax is None:
                continue
            p, dd, rr = np.asarray(ax[0], float), np.asarray(ax[1], float), float(ax[2])
            tol = 1e-3 * max(rr, 1.0)
            g = None
            for cand in groups:
                p0, d, r = cand[0]
                q = p - p0
                if abs(abs(float(dd @ d)) - 1.0) < 1e-6 and np.linalg.norm(q - (q @ d) * d) < tol \
                        and abs(rr - r) < tol:
                    g = cand
                    break
            if g is None:
                g = [(p, dd, rr), self.face_prop(pi, fi), []]
                groups.append(g)
            bb = np.asarray(self.face_prop(pi, fi)["bbox"])
            corners = np.array([[bb[i], bb[j], bb[k]] for i in (0, 3) for j in (1, 4) for k in (2, 5)])
            g[2].append((corners - g[0][0]) @ g[0][1])
        out = []
        for (p0, d, r), props, ts in groups:
            ts = np.concatenate(ts)
            q = np.asarray(props["point"], float) - p0
            er = q - (q @ d) * d
            hole = float(np.asarray(props["normal"]) @ er) < 0
            out.append((p0, d, r, float(ts.min()), float(ts.max()), hole))
        return out

    def bearing_arrows(self, faces, F, L):
        """Bearing force arrows: pressure proportional to cos θ on the facing half of the hole (for a pin on the opposite half)."""
        cyl = self.cylinder_of(faces)
        if cyl is None:
            return []
        p0, d, r, t0, t1, hole = cyl
        F = np.asarray(F, float)
        Fa = (F @ d) * d
        Fr = F - Fa
        c = p0 + 0.5 * (t0 + t1) * d
        out = []
        Lb = min(L, 0.9 * r) if hole else L
        s = 1.0 if hole else -1.0
        if np.linalg.norm(Fr) > 1e-12 * np.linalg.norm(F):
            fh = Fr / np.linalg.norm(Fr)
            e2 = np.cross(d, fh)
            for th in np.radians(np.linspace(-72, 72, 7)):
                w = np.cos(th) * fh + np.sin(th) * e2
                out.append((c + s * r * w, w, Lb * max(np.cos(th), 0.25)))
        else:
            fh = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
            fh /= np.linalg.norm(fh)
        if np.linalg.norm(Fa) > 1e-9 * np.linalg.norm(F):
            out.append((c + s * r * fh, Fa, Lb))
        return out

    @staticmethod
    def describe_constraint(c):
        if isinstance(c, Fixed):
            return f"{c.selection}"
        if isinstance(c, CylindricalSupport):
            comps = [n for n, on in (("radially", c.radial), ("tangentially", c.tangential), ("axially", c.axial)) if on]
            return f"{c.selection}: cylindrical – {', '.join(comps) or 'nothing constrained'}"
        if isinstance(c, RemoteSupport):
            from fea_core.remote import dofs_text
            return (f"{c.selection}: remote {('rigid' if c.coupling == 'rigid' else 'distributing')} – {dofs_text(c.dofs)} at the point (" + "; ".join(fmt(float(x)) for x in c.point) + ")")
        if isinstance(c, ElasticSupport):
            from fea_core.loads_text import constraint_text
            return f"{c.selection}: {constraint_text(c, num=fmt)}"
        comps = ", ".join(f"u{a} = {fmt(v)}" for a, v in zip("xyz", (c.ux, c.uy, c.uz)) if v is not None)
        return f"{c.selection}: {comps} mm"

    @staticmethod
    def describe_load(l):
        from fea_core.loads_text import pressure_text, rotation_text, vec_text

        def vec(v):
            return vec_text(v, fmt)
        cs = f" [{l.csys}]" if getattr(l, "csys", None) else ""
        if isinstance(l, Force):
            return f"{l.selection}: F = {vec(l.vector)} N{cs}"
        if isinstance(l, Pressure) and l.variable:
            return f"{l.selection}: " + pressure_text(l, fmt).split(" (level")[0].split(" ·")[0] + \
                (f", level {vec(l.origin)}" if l.clip else "")
        if isinstance(l, Pressure):
            return f"{l.selection}: p = {fmt(l.value)} MPa"
        if isinstance(l, BearingLoad):
            return f"{l.selection}: bearing F = {vec(l.vector)} N{cs}"
        if isinstance(l, Gravity):
            return f"Gravity: g = {vec(l.vector)} mm/s²{cs}"
        if isinstance(l, Rotation):
            return "Rotation: " + rotation_text(l, fmt)
        if isinstance(l, ThermalLoad):
            if l.case:
                return f"Temperature from thermal state '{l.case}'" + (f" at time {fmt(l.time)} s" if l.time is not None else "")
            return f"Temperature {fmt(l.value)} °C – " + (", ".join(l.parts) if l.parts else "all parts")
        k = getattr(l, "kind", "")
        if k == "frame_node":
            return (f"Node {l.node}: F = {vec(l.force)} N" + (f", M = {vec(l.moment)} N·mm" if any(l.moment) else ""))
        if k == "frame_member":
            q = f"q = {vec(l.q)}" + (f" → {vec(l.q_end)}" if l.q_end is not None else "")
            return f"Member {l.member}: {q} N/mm" + (" (local axes)" if l.axes == "local" else "")
        if k == "frame_temperature":
            g = (f", z gradient {fmt(l.grad_z)} K" if l.grad_z else "") + (f", y gradient {fmt(l.grad_y)} K" if l.grad_y else "")
            return f"Member temperature ({(', '.join(l.members) if l.members else 'all members')}): ΔT = {fmt(l.dT)} K{g}"
        if isinstance(l, RemoteLoad):
            parts = []
            if any(l.force):
                parts.append(f"F = {vec(l.force)} N")
            if any(l.moment):
                parts.append(f"M = {vec(l.moment)} N·mm")
            return f"{l.selection}: {', '.join(parts)} at point {vec(l.point)}{cs}"
        return str(l)

    def refresh_model_info(self):
        if self.study is None:
            if hasattr(self, "model_table"):
                self.model_table.setRowCount(0)
                self.model_table.setVisible(False)
                self.btn_toggle_ignore_part.setVisible(False)
                self.btn_toggle_hide_part.setVisible(False)
                self.btn_show_all_parts.setVisible(False)
            self.model_info.setVisible(True)
            self.model_info.setMaximumHeight(16777215)
            self.model_info.setHtml("Open a STEP model (.step, .stp) or a .fea project.")
            return
        if self.frame_only():
            if hasattr(self, "model_table"):
                self.model_table.setRowCount(0)
                self.model_table.setVisible(False)
                self.btn_toggle_ignore_part.setVisible(False)
                self.btn_toggle_hide_part.setVisible(False)
                self.btn_show_all_parts.setVisible(False)
            self.model_info.setVisible(True)
            self.model_info.setMaximumHeight(16777215)
            fr = self.study.frame
            self.model_info.setHtml(f"<b>Frame without CAD</b><br>Nodes: {len(fr.nodes)}, members: {len(fr.members)}, sections: {len(fr.sections)}<br><br>Edit the frame in the <b>Members</b> step, the member material in the Materials step (study default material).<br><br>Units: mm, N, MPa, t, s")
            return
        if hasattr(self, "model_table"):
            self.model_table.setVisible(True)
            self.btn_toggle_ignore_part.setVisible(True)
            self.btn_toggle_hide_part.setVisible(True)
            self.btn_show_all_parts.setVisible(True)
            self.model_info.setMaximumHeight(130)

            name = html.escape(self.step_path.name) if self.step_path else ""
            ign = set(self.study.ignored_parts)
            hid = set(self.study.hidden_parts)
            excl = self.study.excluded_parts()

            self.model_table.blockSignals(True)
            self.model_table.setRowCount(len(self.parts))
            for r, p in enumerate(self.parts):
                try:
                    m = self.study.material_of(p.name)
                except KeyError:
                    m = None
                is_ign = p.name in ign
                is_bolt = p.name in excl and not is_ign

                for c, on in ((0, p.name not in hid), (1, not is_ign)):
                    chk = QTableWidgetItem()
                    chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                    chk.setCheckState(Qt.Checked if on else Qt.Unchecked)
                    chk.setData(Qt.UserRole, p.name)
                    self.model_table.setItem(r, c, chk)

                it_name = QTableWidgetItem(p.name)
                if is_ign:
                    it_name.setForeground(QColor(140, 140, 140))
                    font = it_name.font()
                    font.setStrikeOut(True)
                    it_name.setFont(font)
                self.model_table.setItem(r, 2, it_name)

                if is_ign:
                    st_txt = "Excluded from analysis"
                    it_st = QTableWidgetItem(st_txt)
                    it_st.setForeground(QColor(ORANGE))
                elif is_bolt:
                    st_txt = "Bolt (beam)"
                    it_st = QTableWidgetItem(st_txt)
                    it_st.setForeground(QColor(GREEN))
                else:
                    st_txt = "Active"
                    it_st = QTableWidgetItem(st_txt)
                    it_st.setForeground(QColor(GREEN))
                if p.name in hid:
                    it_st.setText(st_txt + " (hidden)")
                self.model_table.setItem(r, 3, it_st)

                vol_cm3 = p.volume() / 1000.0
                it_vol = QTableWidgetItem(f"{fmt(vol_cm3, 4)}")
                it_vol.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if is_ign:
                    it_vol.setForeground(QColor(140, 140, 140))
                self.model_table.setItem(r, 4, it_vol)

                it_mat = QTableWidgetItem(m.name if m else "-")
                if is_ign:
                    it_mat.setForeground(QColor(140, 140, 140))
                self.model_table.setItem(r, 5, it_mat)

            self.model_table.blockSignals(False)
            self.model_table.resizeColumnsToContents()
            self.model_table.horizontalHeader().setStretchLastSection(True)

            active_cnt = sum(1 for p in self.parts if p.name not in excl)
            ign_cnt = len(ign)
            summary = [f"<b>{name}</b><br>Total bodies: {len(self.parts)} (active for analysis: {active_cnt}"]
            if ign_cnt:
                summary[0] += f", excluded: {ign_cnt}"
            if hid:
                summary[0] += f", hidden in 3D: {len(hid & {p.name for p in self.parts})}"
            summary[0] += ")<br>Units: mm, N, MPa, t, s"
            if getattr(self, "geometry_warnings", None):
                summary.append("<br><br><b>STEP geometry check warnings:</b><br>"
                               + "<br>".join(html.escape(w) for w in self.geometry_warnings[:10]))
            self.model_info.setHtml("".join(summary))

    def refresh_tree(self, check_result: tuple[list[str], list[str]] | None = None):
        self.tree.clear()
        if self.study is None:
            return
        st = self.study
        root = QTreeWidgetItem(self.tree, [st.name, ""])
        root.setExpanded(True)

        def node(parent, text, info="", color=None, data=None):
            it = QTreeWidgetItem(parent, [text, info])
            if color:
                it.setIcon(0, dot_icon(color))
            it.setData(0, Qt.UserRole, data)
            return it

        errors, warnings = check_result if check_result is not None else self.check()
        excl = st.excluded_parts()
        parts = node(root, "Parts", str(len(self.parts)), RED if self.material_errors(errors) else GREEN,
                     ("stage", STAGE["Materials"]))
        ignored = set(st.ignored_parts)
        for p in self.parts:
            try:
                mname = st.material_of(p.name).name
            except KeyError:
                mname = None
            info = ("excluded from analysis" if p.name in ignored else "bolt (beam)" if p.name in excl
                    else mname or "no material")
            node(parts, p.name, info, GREY if p.name in ignored else (RED if mname is None and p.name not in excl else None),
                 ("part", p.name))
        from fea_core.printing.region import print_parts
        printed = print_parts(st, [p.name for p in self.parts if p.name not in excl])
        if printed:
            pit = node(root, "3D printing", f"{len(printed)} parts", GREEN, ("stage", STAGE["3D printing"]))
            for name in printed:
                sp = st.printing.spec(name)
                node(pit, name, f"infill {fmt(100 * sp.infill, 3)} %" if sp.enabled else "solid", None,
                     ("stage", STAGE["3D printing"]))
        if st.bolts or st.welds:
            jits = node(root, "Joints", f"{len(st.bolts)} bolts, {len(st.welds)} welds", self.joints_color(),
                        ("stage", STAGE["Joints"]))
            chk = {c.name: c for c in self.joints_panel.assessment.all()} if self.joints_panel.assessment else {}
            for i, b in enumerate(st.bolts):
                c = chk.get(b.name)
                node(jits, f"{b.name}: {b.designation}", f"utilization {fmt(c.utilization, 3)}" if c else "",
                     self.util_color(c) if c else None, ("bolt", i))
            for i, w in enumerate(st.welds):
                c = chk.get(w.name)
                node(jits, f"{w.name}: {w.part}", f"utilization {fmt(c.utilization, 3)}" if c else "",
                     self.util_color(c) if c else None, ("weld", i))
        if st.contacts or st.analysis.nonlinear:      # contact pairs (F8)
            from fea_core.contacts.model import CONTACT_LABELS
            cit = node(root, "Contacts", f"{len(st.contacts)} pairs", self._panel_color("contact_panel", errors, warnings),
                       ("stage", STAGE["Contacts"]))
            for i, c in enumerate(st.contacts):
                node(cit, c.name, CONTACT_LABELS.get(c.kind, c.kind) + ("" if c.enabled else " (off)"), None,
                     ("contact", i))
        mech = st.mechanism
        if mech.bodies:                               # mechanismus (F8, krok Pohyb)
            mit = node(root, "Motion", f"{len(mech.bodies)} bodies, {len(mech.joints)} joints",
                       self._panel_color("motion_panel"), ("stage", STAGE["Motion"]))
            for j in mech.joints:
                from fea_core.motion.model import JOINT_LABELS
                node(mit, j.name, JOINT_LABELS.get(j.kind, j.kind), None, ("stage", STAGE["Motion"]))
        thermal_only = self.no_static()
        fr = st.frame
        if fr.nodes or fr.members:
            fit = node(root, "Members", f"{len(fr.members)} members", self._panel_color("frame_panel", errors), ("stage", STAGE["Members"]))
            for mname, m in list(fr.members.items())[:200]:
                node(fit, mname, f"{m.section}", None, ("frame_member", mname))
        cons = node(root, "Constraints", str(len(st.constraints)),
                    GREEN if st.constraints else (GREY if thermal_only or fr.supports else RED),
                    ("stage", STAGE["Constraints"]))
        for i, c in enumerate(st.constraints):
            node(cons, self.describe_constraint(c), "", C_HEX(C_CONSTRAINT), ("constraint", i))
        tit = node(root, "Heat", f"{len(st.thermal.cases)} st." if st.thermal.cases else "–",
                   self.thermal_color(errors, warnings), ("stage", STAGE["Heat"]))
        for ci, c in enumerate(st.thermal.cases):
            pre = f"Thermal case '{c.name}'"
            col = (RED if any(e.startswith(pre) for e in errors) else
                   ORANGE if any(w.startswith(pre) for w in warnings) else GREEN)
            cit = node(tit, c.name, describe_case(c), col, ("tcase", ci))
            for i, bc in enumerate(c.loads):
                node(cit, describe_bc(bc), "", None, ("tbc", (ci, i)))
        lcs = node(root, "Load cases", str(len(st.load_cases)), self.loads_color(errors),
                   ("stage", STAGE["Loads"]))
        for li, lc in enumerate(st.load_cases):
            bad = any(e.startswith(f"State '{lc.name}'") for e in errors)
            lit = node(lcs, lc.name, f"{len(lc.loads)}×", RED if bad else GREEN if lc.loads else
                       (GREY if st.thermal.cases or st.modal.enabled else ORANGE), ("lc", li))
            for i, l in enumerate(lc.loads):
                node(lit, self.describe_load(l), "", None, ("load", (li, i)))
        if st.dynamics.enabled or st.masses:          # dynamics and added mass (F8)
            ds = st.dynamics
            kinds = [t for on, t in ((ds.harmonic.enabled, "harmonic"), (ds.transient.enabled, "transient"),
                                     (ds.random.enabled, "random"), (ds.shock.enabled, "shock")) if on]
            dit = node(root, "Dynamics", (f"{len(kinds)} responses" if len(kinds) > 1 else kinds[0]) if ds.enabled and kinds
                       else ("modes" if ds.enabled else "–"), self._panel_color("dynamics_panel", errors, warnings),
                       ("stage", STAGE["Dynamics"]))
            dit.setToolTip(1, ", ".join(kinds))
            for m in st.masses:
                node(dit, f"Mass: {m.name or m.selection}", f"{fmt(1000 * m.mass)} kg" if m.kind == "point_mass"
                     else f"{fmt(m.mass_per_area)} kg/m²", None, ("stage", STAGE["Dynamics"]))
        mesh_txt = f"{fmt(len(self.mesh.tets))} elements" if self.mesh is not None else "not generated"
        node(root, "Mesh", mesh_txt, GREEN if self.mesh is not None else ORANGE, ("stage", STAGE["Mesh"]))
        res_txt = "computed" if self.results is not None else "not computed"
        rit = node(root, "Results", res_txt, GREEN if self.results is not None else GREY,
                   ("stage", STAGE["Results"]))
        if self.results is not None:
            for i, s in enumerate(self.results.steps):
                if len(s.S):
                    info = f"σmax {fmt(float(s.mises.max()))} MPa"
                else:
                    mm = (s.frame or {}).get("members") or {}
                    info = (f"σ_eq {fmt(max(float(np.max(d['sigma_eq'])) for d in mm.values()))} MPa" if mm else "")
                node(rit, s.name, info, None, ("result", f"step:{i}"))
            if self.results.twod is not None:
                for i, s2 in enumerate(self.results.twod.steps):
                    node(rit, f"2D: {s2['name']}", f"σmax {fmt(float(self.results.twod.mises(i).max()))} MPa", None,
                         ("result", f"twod:{i}"))
            if self.results.shell is not None:
                for i, s3 in enumerate(self.results.shell.summary()["steps"]):
                    node(rit, f"Shells: {s3['name']}", f"σmax {fmt(s3['mises_max'])} MPa", None,
                         ("result", f"shell:{i}"))
            for j, t in enumerate(self.results.thermal):
                node(rit, f"Thermal: {t.name}", f"Tmax {fmt(float(t.T.max()))} °C", None, ("result", f"thermal:{j}"))
            for j, ms in enumerate(self.results.modes):          # natural frequencies, buckling
                info = (f"f₁ {fmt(float(ms.values[0]))} Hz" if ms.kind == "modal" else
                        f"λ₁ {fmt(float(ms.values[0]))}") if len(ms.values) else "–"
                mit = node(rit, ms.name, info, None, ("result", f"mode:{j}:0"))
                for k in range(len(ms.values)):
                    node(mit, mode_label(ms, k), "", None, ("result", f"mode:{j}:{k}"))
            dyn_items = [x for x in dynr.items(self.results) if not x[1].startswith("dyn:mode")]
            if dyn_items:                                          # dynamika (F8)
                dres = self.results.dynamics
                dit = node(rit, "Dynamics", f"f₁ {fmt(float(dres.freq[0]))} Hz" if dres.K else "", None,
                           ("result", dyn_items[0][1]))
                for txt, data in dyn_items:
                    node(dit, txt.replace("Dynamics: ", ""), "", None, ("result", data))
        fr = self.fat_panel.result
        fat_txt = (f"D max {fmt(float(fr.total_damage.max()), 3)}" if fr is not None else
                   ("set" if st.fatigue and st.fatigue.history.channels else "–"))
        node(root, "Fatigue", fat_txt, self.fatigue_color(), ("stage", STAGE["Fatigue"]))
        self.tree.expandAll()

    @staticmethod
    def util_color(check):
        return util_color(check.utilization)

    @staticmethod
    def material_errors(errors) -> list[str]:
        """Readiness errors that are fixed in the Materials step (assignment, properties for heat)."""
        return [e for e in errors if "material" in e or e.startswith("Material")]

    def thermal_only(self) -> bool:
        """A study with heat conduction only (no non-empty mechanical state) – constraints and loads are not needed."""
        st = self.study
        return st is not None and bool(st.thermal.cases) and not any(lc.loads for lc in st.load_cases)

    def thermal_color(self, errors=None, warnings=None):
        """Heat step: grey = no thermal state, red = error, orange = warning, green = ready."""
        st = self.study
        if st is None or not st.thermal.cases:
            return GREY
        if errors is None:
            errors, warnings = self.check()
        if any(e.startswith("Thermal case") or "thermal conductivity" in e or "heat capacity" in e for e in errors):
            return RED
        used = {getattr(bc, "selection", None) for c in st.thermal.cases for bc in c.loads}
        if any(e.startswith("Selection '") and e.split("'")[1] in used for e in errors):
            return RED
        return ORANGE if any(w.startswith("Thermal case") for w in warnings or []) else GREEN

    def loads_color(self, errors):
        st = self.study
        if any(e.startswith("State '") for e in errors):
            return RED
        if self.no_static():
            return GREY
        if not any(lc.loads for lc in st.load_cases):
            return RED
        # empty state: orange (with thermal states it is skipped in the calculation)
        return GREEN if all(lc.loads for lc in st.load_cases) or st.thermal.cases or st.modal.enabled or \
            st.dynamics.enabled else ORANGE

    def joints_color(self):
        """Joints step: grey = none, green = entered / satisfied, orange = close, red = not satisfied."""
        st = self.study
        if st is None or not (st.bolts or st.welds):
            return GREY
        a = getattr(self, "joints_panel", None) and self.joints_panel.assessment
        if a:
            us = [c.utilization for c in a.all() if math.isfinite(c.utilization)]
            return util_color(max(us)) if us else ORANGE
        errors, _ = self.check()
        return RED if any(e.startswith(("Bolt", "Weld")) or "bolt" in e for e in errors) else GREEN

    def fatigue_color(self):
        st = self.study
        fr = getattr(self, "fat_panel", None) and self.fat_panel.result
        if fr is not None:
            return GREEN if fr.total_damage.max() < 1 else RED
        if st is not None and st.fatigue and st.fatigue.history.channels:
            return ORANGE
        return GREY

    def refresh_flow(self, check_result: tuple[list[str], list[str]] | None = None):
        st = self.study
        if st is None:
            cols = {name: GREY for name in STAGES}
            errors, warnings = [], []
        else:
            errors, warnings = check_result if check_result is not None else self.check()
            cols = {
                "Model": GREEN,
                "Materials": RED if self.material_errors(errors) else GREEN,
                "3D printing": RED if any(e.startswith("3D printing") for e in errors) else self._panel_color("print_panel"),
                "Joints": self.joints_color(),
                "Contacts": self._panel_color("contact_panel", errors, warnings),
                "Dynamics": self._panel_color("dynamics_panel", errors, warnings),
                "Motion": self._panel_color("motion_panel"),
                "Members": self._panel_color("frame_panel", errors),
                "Constraints": GREEN if st.constraints else (GREY if self.no_static() or st.frame.supports else RED),
                "Heat": self.thermal_color(errors, warnings),
                "Loads": self.loads_color(errors),
                "Mesh": GREEN if self.mesh is not None else ORANGE,
                "Solve": GREEN if self.results is not None else (ORANGE if not errors else RED),
                "Results": GREEN if self.results is not None else GREY,
                "Fatigue": self.fatigue_color(),
            }
        fo = self.frame_only()
        self._flow_cols = cols
        for name, a in zip(STAGES, self.stage_actions):
            a.setEnabled(not (fo and name in FRAME_ONLY_OFF))       # frame system without CAD: steps hidden and disabled
        self._apply_flow_visibility()
        self.flow_state = (cols, errors if st is not None else [], warnings if st is not None else [])
        self.flowChanged.emit()

    def on_tree_click(self, item, col):
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, k = data
        if kind == "stage":
            self.set_stage(k)
        elif kind == "part":
            pi = next((i for i, p in enumerate(self.parts) if p.name == k), None)
            if pi is not None:
                self.view.set_selected_parts([pi])
                if STAGES[getattr(self, "stage", 0)] == "Model":
                    if hasattr(self, "model_table"):
                        self.model_table.clearSelection()
                        if pi < self.model_table.rowCount():
                            self.model_table.selectRow(pi)
                elif STAGES[getattr(self, "stage", 0)] == "Materials":
                    for r in range(self.mat_parts.count()):
                        it = self.mat_parts.item(r)
                        if it.text() == k:
                            self.mat_parts.setCurrentItem(it)
                            break
        elif kind in ("bolt", "weld"):
            self.joints_panel.select_joint(kind, k)
        elif kind == "constraint":
            self.set_stage(STAGE["Constraints"])
            self.con_list.setCurrentRow(k)
        elif kind == "contact":
            self.set_stage(STAGE["Contacts"])
            self.contact_panel.table.selectRow(k)
        elif kind == "frame_member":
            self.set_stage(STAGE["Members"])
            self.frame_panel.select("member", k)
        elif kind == "lc":
            self.set_stage(STAGE["Loads"])
            self.lc_combo.setCurrentIndex(k)
        elif kind == "load":
            self.set_stage(STAGE["Loads"])
            self.lc_combo.setCurrentIndex(k[0])
            self.load_list.setCurrentRow(k[1])
        elif kind == "tcase":
            self.set_stage(STAGE["Heat"])
            self.thermal_panel.case_combo.setCurrentIndex(k)
        elif kind == "tbc":
            self.set_stage(STAGE["Heat"])
            self.thermal_panel.case_combo.setCurrentIndex(k[0])
            self.thermal_panel.bc_list.setCurrentRow(k[1])
        elif kind == "result":
            self.set_stage(STAGE["Results"])
            self.r_step.setCurrentIndex(max(self.r_step.findData(k), 0))

    # ================================================================== other
    def log_msg(self, text):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    # ================================================================== part management and exclusion from the calculation
    def apply_ghost(self):
        """Excluded parts translucent in 3D (in all steps with a model)."""
        st = self.study
        v = self.view
        if not hasattr(v, "set_ghost"):
            return
        ign = set(st.ignored_parts) if st else set()
        v.set_ghost([i for i, p in enumerate(self.parts or []) if p.name in ign])
        if hasattr(v, "set_hidden"):
            hid = set(st.hidden_parts) if st else set()
            v.set_hidden([i for i, p in enumerate(self.parts or []) if p.name in hid])

    def set_ignored(self, names: list[str], on: bool):
        """Excludes or includes parts in the calculation (ignored_parts)."""
        st = self.study
        if st is None:
            return
        cur = list(st.ignored_parts)
        new = (cur + [n for n in names if n not in cur]) if on else [n for n in cur if n not in names]
        if new == cur:
            return
        st.ignored_parts = new
        self.apply_ghost()
        self.log_msg(("Excluded from analysis: " if on else "Restored to analysis: ") + ", ".join(names))
        self.changed(invalidate_mesh=True)
        if hasattr(self, "joints_panel"):
            self.joints_panel.refresh_ignored()

    def set_hidden(self, names: list[str], on: bool):
        """Hides or shows parts in the 3D view (Study.hidden_parts) – results and mesh are not discarded."""
        st = self.study
        if st is None:
            return
        cur = list(st.hidden_parts)
        new = (cur + [n for n in names if n not in cur]) if on else [n for n in cur if n not in names]
        if new == cur:
            return
        st.hidden_parts = new
        self.apply_ghost()
        self.dirty = True
        self.push_undo()
        self.update_title()
        self.refresh_model_info()
        self.refresh_tree()

    def isolate_selected(self):
        """Isolate command: only the parts selected in 3D (or in the parts table)."""
        names = [self.parts[i].name for i in self.view.selected_parts() if 0 <= i < len(self.parts)]
        if not names and hasattr(self, "model_table"):
            rows = sorted({ix.row() for ix in self.model_table.selectedIndexes()})
            names = [self.parts[r].name for r in rows if r < len(self.parts)]
        if names:
            self.isolate_parts(names)

    def show_geometry_check(self):
        """Check command: Model step and the Geometry check tab of the bottom dock."""
        self.set_stage(STAGE["Model"])
        if hasattr(self, "table_dock"):
            self.table_dock.show_tab("geometry")
            self.dock_tables.raise_()

    def isolate_parts(self, names: list[str]):
        """Shows only the given parts (hides the others)."""
        if self.study is None or not names:
            return
        keep = set(names)
        self.study.hidden_parts = []
        self.set_hidden([p.name for p in self.parts if p.name not in keep], True)
        self.apply_ghost()
        self.refresh_model_info()

    def parts_by_pattern(self, pattern: str) -> list[str]:
        """Part names by pattern (* and ?, case-insensitive; without wildcards = contains the text)."""
        import fnmatch
        pat = pattern.strip().lower()
        if not pat:
            return []
        if not any(ch in pat for ch in "*?["):
            pat = f"*{pat}*"
        return [p.name for p in self.parts if fnmatch.fnmatchcase(p.name.lower(), pat)]

    def _by_pattern_dialog(self, what: str):
        """In bulk by name: what = ignore | include | hide | show."""
        titles = {"ignore": "Exclude from the analysis the bodies whose name contains / matches the pattern:",
                  "include": "Include in the analysis the bodies whose name contains / matches the pattern:",
                  "hide": "Hide in 3D the bodies whose name contains / matches the pattern:",
                  "show": "Show in 3D the bodies whose name contains / matches the pattern:"}
        text, ok = QInputDialog.getText(self, APP_NAME, titles[what] + "\n(e.g. bolt, M6*, *magnet*)")
        if not ok:
            return
        names = self.parts_by_pattern(text)
        if not names:
            msg(self, APP_NAME, f"No body matches the pattern “{text}”.")
            return
        if what in ("ignore", "include"):
            self.set_ignored(names, what == "ignore")
        else:
            self.set_hidden(names, what == "hide")

    def _on_model_table_check(self, item):
        """A tick in the parts table: column 0 = show, 1 = count (applies to all selected rows)."""
        if getattr(self, "_filling", False) or item.column() not in (0, 1):
            return
        name = item.data(Qt.UserRole)
        if not name:
            return
        on = item.checkState() == Qt.Checked
        rows = {i.row() for i in self.model_table.selectionModel().selectedRows()}
        names = [name]
        if item.row() in rows and len(rows) > 1:          # a tick in the selection = the whole selection
            names = [self.parts[r].name for r in sorted(rows) if r < len(self.parts)]
        if item.column() == 0:
            self.set_hidden(names, not on)
        else:
            self.set_ignored(names, not on)

    def _toggle_selected_model_parts_hidden(self):
        sel_rows = [i.row() for i in self.model_table.selectionModel().selectedRows()]
        names = [self.parts[r].name for r in sel_rows if r < len(self.parts)]
        if not names:
            msg(self, APP_NAME, "Select a body in the table or in the 3D window.")
            return
        hid = set(self.study.hidden_parts) if self.study else set()
        self.set_hidden(names, not all(n in hid for n in names))

    def _on_tree_menu(self, pos):
        """Context menu of the tree (right click): exclude / include the body from the calculation."""
        item = self.tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, k = data
        if kind == "part":
            pname = k
            pi = next((i for i, p in enumerate(self.parts) if p.name == pname), None)
            ign = set(self.study.ignored_parts) if self.study else set()
            menu = QMenu(self)
            if pname in ign:
                act = menu.addAction(f"Include part '{pname}' in the analysis")
                act.triggered.connect(lambda: self.set_ignored([pname], False))
            else:
                act = menu.addAction(f"Exclude part '{pname}' from the analysis")
                act.triggered.connect(lambda: self.set_ignored([pname], True))
            hid = set(self.study.hidden_parts) if self.study else set()
            if pname in hid:
                act = menu.addAction(f"Show part '{pname}' in 3D")
                act.triggered.connect(lambda: self.set_hidden([pname], False))
            else:
                act = menu.addAction(f"Hide part '{pname}' in 3D")
                act.triggered.connect(lambda: self.set_hidden([pname], True))
            act = menu.addAction("Show only this part")
            act.triggered.connect(lambda: self.isolate_parts([pname]))
            if pi is not None:
                menu.addSeparator()
                zoom_act = menu.addAction("Zoom in 3D view\tF")
                zoom_act.triggered.connect(lambda: (self.view.set_selected_parts([pi]), self.view.zoom_selection()))
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _on_model_table_selection(self):
        rows = [i.row() for i in self.model_table.selectionModel().selectedRows()]
        if STAGES[getattr(self, "stage", 0)] == "Model":
            self.view.set_selected_parts(rows)
            names = [self.parts[r].name for r in rows if r < len(self.parts)]
            self.st_info.setText(("Selected body: " + ", ".join(names)) if names else "")

    def _on_model_table_menu(self, pos):
        sel_rows = [i.row() for i in self.model_table.selectionModel().selectedRows()]
        item = self.model_table.itemAt(pos)
        if item is not None and item.row() not in sel_rows:
            self.model_table.selectRow(item.row())
            sel_rows = [item.row()]
        if not sel_rows:
            return
        names = [self.parts[r].name for r in sel_rows if r < len(self.parts)]
        if not names:
            return

        ign = set(self.study.ignored_parts) if self.study else set()
        all_ignored = all(n in ign for n in names)

        menu = QMenu(self)
        if len(names) == 1:
            name = names[0]
            if all_ignored:
                act = menu.addAction(f"Include body '{name}' in the analysis")
                act.triggered.connect(lambda: self.set_ignored(names, False))
            else:
                act = menu.addAction(f"Exclude body '{name}' from the analysis")
                act.triggered.connect(lambda: self.set_ignored(names, True))
        else:
            if all_ignored:
                act = menu.addAction(f"Include selected bodies ({len(names)}) in the analysis")
                act.triggered.connect(lambda: self.set_ignored(names, False))
            else:
                act = menu.addAction(f"Exclude selected bodies ({len(names)}) from the analysis")
                act.triggered.connect(lambda: self.set_ignored(names, True))

        hid = set(self.study.hidden_parts) if self.study else set()
        if all(n in hid for n in names):
            act = menu.addAction("Zobrazit ve 3D" + (f" ({len(names)})" if len(names) > 1 else ""))
            act.triggered.connect(lambda: self.set_hidden(names, False))
        else:
            act = menu.addAction("Hide in 3D" + (f" ({len(names)})" if len(names) > 1 else ""))
            act.triggered.connect(lambda: self.set_hidden(names, True))
        act = menu.addAction("Show only the selected bodies")
        act.triggered.connect(lambda: self.isolate_parts(names))
        menu.addSeparator()
        act = menu.addAction("Show all")
        act.triggered.connect(lambda: self.set_hidden([p.name for p in self.parts], False))
        act = menu.addAction("Include all in the analysis")
        act.triggered.connect(lambda: self.set_ignored([p.name for p in self.parts], False))
        sub = menu.addMenu("By name")
        for what, label in (("ignore", "Exclude from analysis…"), ("include", "Include in analysis…"),
                            ("hide", "Hide in 3D…"), ("show", "Show in 3D…")):
            a = sub.addAction(label)
            a.triggered.connect(lambda _=False, w=what: self._by_pattern_dialog(w))
        menu.addSeparator()
        zoom_act = menu.addAction("Zoom in 3D view\tF")
        zoom_act.triggered.connect(self.view.zoom_selection)
        menu.exec(self.model_table.viewport().mapToGlobal(pos))

    def _toggle_selected_model_parts_ignore(self):
        sel_rows = [i.row() for i in self.model_table.selectionModel().selectedRows()]
        names = [self.parts[r].name for r in sel_rows if r < len(self.parts)]
        if not names:
            msg(self, APP_NAME, "Select a body in the table or in the 3D window.")
            return
        ign = set(self.study.ignored_parts) if self.study else set()
        all_ignored = all(n in ign for n in names)
        self.set_ignored(names, not all_ignored)

    # ================================================================== Dynamic IPC Bridge (LLM / MCP)
    def _get_mcp_tools(self):
        if self._mcp_tools_cache is None:
            from fea_core.rest.registry import collect_tools
            self._mcp_tools_cache = collect_tools(self._mcp_session)
        return self._mcp_tools_cache

    def execute_mcp_action(self, tool_name: str, args: dict):
        """Runs an MCP tool or an interactive GUI command in the main thread of the window."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"LLM: {tool_name} ...", 0)
        QApplication.processEvents()
        try:
            # 1. Interactive GUI tools
            if tool_name == "fea_gui_status":
                return {
                    "gui_running": True,
                    "study": self.study.name if self.study is not None else None,
                    "stage": STAGES[self.stage] if hasattr(self, "stage") else None,
                    "parts": len(self.parts) if self.parts else 0,
                    "mesh_ready": self.mesh is not None,
                    "results_ready": self.results is not None,
                }
            if tool_name == "fea_cad_link_info":       # CAD add-in: what is open and which tokens to translate
                return self.cad_link_info()
            if tool_name in ("fea_cad_update", "fea_cad_open"):    # CAD add-in: new geometry / new study
                p = Path(str(args.get("step_path", "")))
                from fea_core.pathcheck import is_unc
                if is_unc(str(p)) or p.suffix.lower() not in (".step", ".stp") or not p.is_file():
                    raise ValueError(f"Invalid STEP file: {p}")
                if tool_name == "fea_cad_open" or self.study is None or self.step_path is None:
                    QTimer.singleShot(0, lambda: self.open_file(str(p)))     # only after the add-in answers
                    return {"accepted": True, "mode": "new_study"}
                solve = bool(args.get("solve", False))
                if not self._same_cad_document(p):        # a different CAD document than the open project → new study
                    QTimer.singleShot(0, lambda: self.open_file(str(p)))
                    return {"accepted": True, "mode": "new_study", "reason": "different CAD document"}
                QTimer.singleShot(0, lambda: self.update_geometry(p, "cad", solve))
                return {"accepted": True, "mode": "update", "study": self.study.name}
            if tool_name == "fea_gui_toast":
                msg_txt = str(args.get("message", ""))
                timeout_ms = int(args.get("timeout_ms", 4000))
                self.statusBar().showMessage(msg_txt, timeout_ms)
                return {"gui_running": True, "toast": msg_txt}
            if tool_name == "fea_gui_stage_switch":
                stage = str(args.get("stage", ""))
                if stage in STAGE:
                    self.set_stage(STAGE[stage])
                    return {"gui_running": True, "stage": stage}
                raise ValueError(f"Unknown step '{stage}'. Allowed: {', '.join(STAGES)}")
            if tool_name == "fea_gui_camera_focus":
                sel = args.get("selection")
                part = args.get("part")
                if part and part != "*":
                    pi = next((i for i, p in enumerate(self.parts) if p.name == part), None)
                    if pi is not None:
                        from fea_core.geometry import bbox
                        bb = bbox(self.parts[pi].shape)
                        from OCC.Core.Bnd import Bnd_Box
                        from OCC.Core.gp import gp_Pnt
                        b = Bnd_Box()
                        b.Add(gp_Pnt(float(bb[0]), float(bb[1]), float(bb[2])))
                        b.Add(gp_Pnt(float(bb[3]), float(bb[4]), float(bb[5])))
                        self.view.fit_box(b)
                    else:
                        self.view.fit_all()
                elif sel and self.study and sel in self.study.selections:
                    faces = self.sel_faces.get(sel, [])
                    if faces:
                        self.view.select_faces(faces)
                        self.view.zoom_selection()
                    else:
                        self.view.fit_all()
                else:
                    self.view.fit_all()
                self.view.update()
                return {"gui_running": True, "focused": True, "target": sel or part or "all"}
            if tool_name == "fea_gui_highlight":
                entities = args.get("entities") or []
                faces_to_show = []
                for ent in entities:
                    if self.study and ent in self.study.selections:
                        faces_to_show.extend(self.sel_faces.get(ent, []))
                    elif "#" in ent:
                        pname, sidx = ent.rsplit("#", 1)
                        try:
                            f_idx = int(sidx)
                            p_idx = next((i for i, p in enumerate(self.parts) if p.name == pname), None)
                            if p_idx is not None:
                                faces_to_show.append((p_idx, f_idx))
                        except ValueError:
                            pass
                self.view.select_faces(faces_to_show)
                self.view.update()
                return {"gui_running": True, "highlighted": entities}
            if tool_name == "fea_gui_show_mode":
                mode_num = int(args.get("mode_number", 1))
                if self.results is not None and getattr(self.results, "modes", None):
                    idx = -1
                    target_suffix = f":{mode_num - 1}"
                    for i in range(self.r_step.count()):
                        data = str(self.r_step.itemData(i) or "")
                        if data.startswith("mode:") and data.endswith(target_suffix):
                            idx = i
                            break
                    if idx >= 0:
                        self.r_step.setCurrentIndex(idx)
                        self.set_stage(STAGE["Results"])
                        self.show_result()
                        return {"gui_running": True, "mode_shown": mode_num}
                return {"gui_running": True, "mode_shown": mode_num, "note": "Results with mode shapes are not available"}

            # 2. Common MCP tools (fea_core.mcp)
            sess = self._mcp_session
            sess.study = self.study
            sess._parts = self.parts
            sess.step_path = self.step_path
            sess.mesh = self.mesh
            sess.results = self.results
            sess.workdir = self._workdir

            tools = self._get_mcp_tools()
            if tool_name not in tools:
                raise ValueError(f"Unknown tool: '{tool_name}'")

            tool = tools[tool_name]
            v_args = tool.validate(args)
            result = tool.fn(**v_args)

            # synchronisation of the state from the Session back into the window
            if sess.study is not self.study or sess.step_path != self.step_path:
                if sess.study is not None and (self.study is None or sess.step_path != self.step_path):
                    self._load_model(sess.step_path, sess.study, sess.results)
                else:
                    self.study = sess.study
            self.study = sess.study
            self.parts = sess.parts or []
            self.mesh = sess.mesh
            self.results = sess.results
            self.dirty = True

            self.refresh_all()
            if self.results is not None and tool_name in ("fea_solve_run", "fea_job_result"):
                self.set_stage(STAGE["Results"])
                self.show_result()
            elif self.mesh is not None and tool_name == "fea_mesh_generate":
                self.set_stage(STAGE["Mesh"])
                self.show_mesh()
            elif "constraint" in tool_name:
                self.set_stage(STAGE["Constraints"])
            elif "load" in tool_name:
                self.set_stage(STAGE["Loads"])
            elif "material" in tool_name:
                self.set_stage(STAGE["Materials"])

            self.view.update()
            self.statusBar().showMessage(f"LLM: {tool_name} done", 4000)
            return result
        finally:
            QApplication.restoreOverrideCursor()

    def cad_update_start(self, step: Path):
        """Start from the CAD add-in (`--cad-update STEP`): opens the project linked to the document (local registry)
        and uses the new geometry; an unknown project = the STEP as a new study.
        """
        from fea_core import cad_import
        from fea_core.cad_link.registry import lookup
        meta = cad_import.load_sidecar(step) or {}
        link = meta.get("link") if isinstance(meta.get("link"), dict) else {}
        fea = lookup(str(link.get("document_id", "")))
        if fea is None:
            self.open_file(str(step))
            return
        self.open_file(str(fea))
        if self.study is not None and self.project_path == fea:
            self.update_geometry(step, "cad")

    def _same_cad_document(self, step: Path) -> bool:
        """Does the new STEP from the add-in belong to the CAD document of the open project? An unknown id on one side = yes
        (STEP without metadata, older project); known and different ids = no – otherwise the settings of project A would be remapped
        to the geometry of document B (and calculated at once).
        """
        from fea_core import cad_import
        meta = cad_import.load_sidecar(step) or {}
        link = meta.get("link") if isinstance(meta.get("link"), dict) else {}
        new_id = str(link.get("document_id", "") or "")
        old_id = self.cad_link_info().get("document_id", "") if self.study is not None else ""
        return not new_id or not old_id or new_id == old_id

    def cad_link_info(self) -> dict:
        """For the CAD add-in: the open study, the CAD document id and the tokens of the faces / bodies the study uses (the add-in
        translates them to the new version of the model – findEntityByToken).
        """
        d = {"gui_running": True, "study": self.study.name if self.study is not None else None,
             "busy": self.busy(), "tokens": [], "body_tokens": []}
        if self.study is None or self.step_path is None:
            return d
        link = self.study.cad_link
        d["document_id"] = link.document_id if link is not None else ""
        try:
            from fea_core import cad_import
            from fea_core.cad_link.facetable import used_tokens
            meta = cad_import.load_sidecar(self.step_path)
            d["tokens"], d["body_tokens"] = used_tokens(self.study, self.solid_parts(), meta)
            ml = (meta or {}).get("link") if isinstance((meta or {}).get("link"), dict) else {}
            d["document_id"] = d["document_id"] or str(ml.get("document_id", ""))
        except Exception as e:  # noqa: BLE001 – the add-in then translates all tokens of the last export
            d["error"] = str(e)
        return d

    # ================================================================== separate windows (step panel, Tables dock)
    def panel_detached(self) -> bool:
        p = getattr(self, "_panel_pop", None)
        return p is not None and p.is_detached()

    def set_panel_detached(self, on: bool) -> None:
        """Step panel (panel_stack) into its own window and back; the step tabs and the 3D view stay in the main window."""
        if bool(on) == self.panel_detached():
            if on:
                self._panel_pop.raise_window()
            return
        if getattr(self, "_panel_pop", None) is None:
            from .popout import Popout
            self._panel_pop = Popout(self.panel_stack, "Step panel", "panel", owner=self,
                                     note="The step panel is open in a separate window.", default_size=(620, 860))
            self._panel_pop.changed.connect(self._on_panel_detached)
        if on:
            self._panel_pop.detach()
        else:
            self._panel_pop.attach()

    def _on_panel_detached(self, detached: bool) -> None:
        from .widgets import _ui_settings, remember_ui
        from . import theme
        a = getattr(self, "act_panel_window", None)
        if a is not None and a.isChecked() != detached:
            a.blockSignals(True)
            a.setChecked(detached)
            a.blockSignals(False)
        b = getattr(self, "btn_panel_popout", None)
        if b is not None:
            theme.set_icon(b, "dock_in" if detached else "popout")
            b.setToolTip("Return the step panel into the main window" if detached
                         else "Step panel in a separate window (larger, also on a second monitor)")
        self.panel_stack.setMinimumWidth(0 if detached else 460)
        if detached:
            self._update_panel_window_title()
            sizes = self.side_split.sizes()          # the tree gets the side space, the placeholder stays small
            self.side_split.setSizes([sum(sizes) - 90, 90])
        else:
            self.side_split.setSizes([300, 600])
        if remember_ui() and not getattr(self, "_closing", False):
            s = _ui_settings()
            s.setValue("ui/popout/panel_detached", bool(detached))
            s.sync()

    def _update_panel_window_title(self) -> None:
        p = getattr(self, "_panel_pop", None)
        if p is not None and p.window is not None:
            p.window.setWindowTitle(f"{STAGES[self.stage]} – {APP_NAME}")

    def set_tables_floating(self, on: bool) -> None:
        """Bottom Tables dock as a floating window (with a title bar – it can be dragged back to the edge) and back."""
        d = self.dock_tables
        if bool(on) == d.isFloating():
            if on:
                d.raise_()
            return
        if on:
            w, h = max(d.width(), 900), max(d.height() + 120, 380)
            d.setTitleBarWidget(None)               # own title bar: moving, docking back by dragging, closing
            d.setFloating(True)
            d.resize(w, h)
            d.show()
            d.raise_()
        else:
            d.setFloating(False)

    def _on_tables_floating(self, floating: bool) -> None:
        from . import theme
        d = self.dock_tables
        if not floating:
            d.setTitleBarWidget(QWidget())          # docked: the tab is the title (no title bar)
        a = getattr(self, "act_tables_window", None)
        if a is not None and a.isChecked() != floating:
            a.blockSignals(True)
            a.setChecked(floating)
            a.blockSignals(False)
        b = getattr(self.table_dock, "btn_popout", None)
        if b is not None:
            theme.set_icon(b, "dock_in" if floating else "popout")
            b.setToolTip("Return the tables to the bottom of the main window" if floating
                         else "Tables in a separate window")

    def closeEvent(self, e):
        if not self.confirm_discard():
            e.ignore()
            return
        self._closing = True
        if getattr(self, "ipc_server", None) is not None:
            try:
                self.ipc_server.close()
            except Exception:  # noqa: BLE001
                pass
            self.ipc_server = None
        if self._thread is not None:
            if isinstance(self._thread, (SolveThread, MeshThread)):
                self._thread.cancel.set()
                kill_mesh_workers()
            self._thread.wait(5000)
        for t in list(self._aux_threads):         # convergence, print refinement
            t.cancel.set()
            kill_mesh_workers()
            t.wait(5000)
        super().closeEvent(e)
        # temporary work folder (meshes, calculations, unpacked projects) – removed only on a normal close, remains after a crash
        import shutil
        shutil.rmtree(self._workdir, ignore_errors=True)


    def toggle_animation(self):
        """Deformation animation: LEVELS deformation levels per cycle; the first visit of a level is a whole `show_result`,
        its result objects are then kept by the view (FeaView.anim_store) and later visits only swap the displayed objects
        (anim_show). Any other redraw of the result drops the cached frames (show_result → anim_reset).
        The timer runs frame by frame: the next is scheduled only after the previous one is drawn, with a period of at
        least 33 ms and an idle time of at least the frame time, so the main thread is not at 100 % for a large model.
        The phase advances by the real time (for slow frames the animation only gets coarser, it does not run slower).
        """
        import time
        if not hasattr(self, "_anim_timer"):
            from PySide6.QtCore import QTimer
            PERIOD = 1.6                 # s, duration of one cycle of the deflection enlargement
            MIN_MS = 33                  # shortest period of the frames (~30 fps)
            LEVELS = 40                  # deformation levels per cycle (21 distinct frames, each drawn once)
            self._anim_timer = QTimer(self)
            self._anim_timer.setSingleShot(True)
            self._anim_timer.setInterval(MIN_MS)
            self._anim_phase = 0.0
            self._anim_last = 0.0

            def stop(redraw=True):
                self._anim_running = False
                self._anim_timer.stop()
                self.view.anim_end()
                self._anim_state = {}
                self._anim_factor = 1.0
                self.btn_anim.setText("Animate...")
                if redraw:
                    self.show_result()

            def on_tick():
                if self.results is None or STAGES[self.stage] != "Results":
                    stop(redraw=False)           # nothing to animate outside results
                    return
                t0 = time.perf_counter()
                dt = min(t0 - self._anim_last, PERIOD / 8)       # a long delay (modal window) does not move the phase
                self._anim_last = t0
                step = 2 * math.pi * dt / PERIOD
                warm = len(self._anim_state) < LEVELS // 2 + 1
                if warm:                             # first cycle: the frames are built level by level (none skipped)
                    step = min(step, 2 * math.pi / LEVELS)
                self._anim_phase = (self._anim_phase + step) % (2 * math.pi)
                k = int(round(self._anim_phase / (2 * math.pi) * LEVELS)) % LEVELS
                level = min(k, LEVELS - k)                       # the factor is symmetric in the phase
                self._anim_factor = 0.5 * (1.0 - math.cos(2 * math.pi * level / LEVELS))
                self._anim_frame = True              # show_result keeps the summary, hotspot table and contact box
                try:
                    with self.view.batch():          # cached frame: only the displayed objects are swapped
                        if self.view.anim_show(level):
                            self._probe, self._hs_X, self._r_view, self._probe_exact = self._anim_state[level]
                        else:
                            self.show_result()
                            self.view.anim_store(level)
                            self._anim_state[level] = (self._probe, getattr(self, "_hs_X", None), self._r_view,
                                                       self._probe_exact)
                except Exception:
                    stop(redraw=False)           # otherwise the error would repeat 20× per second
                    raise
                finally:
                    self._anim_frame = False
                frame = time.perf_counter() - t0
                if self._anim_running:
                    # period ≥ MIN_MS; idle time ≥ the frame (cached frames) / ≥ half of it (first cycle)
                    self._anim_timer.start(max(MIN_MS - int(1000 * frame), int((500 if warm else 1000) * frame)))

            self._anim_timer.timeout.connect(on_tick)
            self._anim_stop = stop
        if getattr(self, "_anim_running", False):
            self._anim_stop()
        else:
            self._anim_running = True
            self._anim_phase = 0.0
            self._anim_last = time.perf_counter()
            self.view.anim_begin()
            self._anim_state = {}
            self.btn_anim.setText("Stop animation")
            self._anim_timer.start()

    def open_isosurface_dialog(self):
        if self.results is None:
            return
        from .isosurface_dialog import IsosurfaceDialog
        cur = self.current_result()
        si = cur[1] if cur and cur[0] == "step" else 0
        dlg = IsosurfaceDialog(self, self.results, si)
        dlg.exec()

    def open_vectors_dialog(self):
        if self.results is None:
            return
        from .vectors_dialog import VectorsDialog
        cur = self.current_result()
        si = cur[1] if cur and cur[0] == "step" else 0
        step = label = None
        if cur and cur[0] == "combination":         # displayed combination (not state 1); the envelope has no vectors
            from fea_core.results import StepResult
            comb = self.study.combinations[cur[1]]
            try:
                r = self._combination(self.results, cur[1])
            except (ValueError, KeyError) as e:
                msg(self, APP_NAME, f"The combination cannot be computed: {e}")
                return
            if isinstance(r, StepResult):
                step, label = r, f"K{cur[1] + 1:02d}: {comb.name}"
        dlg = VectorsDialog(self, self.results, si, step=step, label=label)
        dlg.exec()

    def open_submodel_dialog(self):
        if self.results is None or not self.results.steps or len(self.results.mesh.nodes) == 0:
            msg(self, APP_NAME, "The submodel needs a solved load case of a 3D model (Solve step).")
            return
        dlg = getattr(self, "_submodel_dlg", None)
        if dlg is None:
            dlg = self._submodel_dlg = smd.SubmodelDialog(self)
        else:
            dlg.refresh()
        dlg.show()
        dlg.raise_()
        dlg.draw_region()

    def open_topo_dialog(self):
        if self.study is None:
            return
        mesh = self.results.mesh if self.results else getattr(self, "mesh", None)
        if mesh is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "First create the model mesh or run the analysis.")
            return
        from .topo_dialog import TopologyOptDialog
        dlg = TopologyOptDialog(self, self.study, mesh, self.results)
        dlg.exec()

    def topo_shape_tris(self, res):
        """Surface of the resulting shape of the topology optimisation (cache on the results object, valid for the same summary)."""
        from fea_core.topo.results import shape_triangles
        info = (res.info or {}).get("topology")
        cache = getattr(res, "_topo_tris", None)
        if cache is not None and cache[0] is info:
            return cache[1]
        tris = shape_triangles(res)
        res._topo_tris = (info, tris)
        return tris

    def topo_attached(self):
        """The optimisation was attached to the results: mark the project as changed WITHOUT discarding the results and show the shape."""
        self.dirty = True
        self.update_title()
        if self.results is None:
            return
        self.results._topo_tris = None
        self.select_stage_results()
        names = [n for n, _ in self.result_qlist()]
        if TOPO_SHAPE_Q in names:
            self.r_qty.blockSignals(True)
            self.r_qty.setCurrentIndex(names.index(TOPO_SHAPE_Q))
            self.r_qty.blockSignals(False)
        self.show_result()

    def select_stage_results(self):
        """Switches to the Results step if it is not active."""
        if STAGES[self.stage] != "Results":
            self.set_stage(STAGE["Results"])

    def open_export_dialog(self):
        if not self.results or not self.results.steps:
            msg(self, APP_NAME, "First solve the study.")
            return
        from .export_dialog import ExportResultsDialog
        dlg = ExportResultsDialog(self)
        dlg.exec()

    def open_imprint_dialog(self):
        if not self.study:
            return
        from .imprint_dialog import ImprintDialog
        dlg = ImprintDialog(self)
        dlg.exec()

    def open_combinations_dialog(self):
        if not self.study:
            return
        from .combinations_dialog import CombinationsDialog
        dlg = CombinationsDialog(self)
        dlg.exec()


def main(app=None, splash=None):
    app = app or QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    def excepthook(t, v, tb):
        text = "".join(traceback.format_exception(t, v, tb))
        sys.stderr.write(text)
        QApplication.restoreOverrideCursor()
        msg(None, APP_NAME, "Unexpected error:\n\n" + text[-1800:], QMessageBox.Critical)

    sys.excepthook = excepthook
    w = MainWindow()
    from fea_core import i18n as _i18n
    snapshot = _i18n.startup_snapshot()           # translations of the window texts (no study opened yet – no names from files)
    QTimer.singleShot(1500, lambda: _i18n.persist_startup(snapshot))   # the next start does not load the catalogue
    w.show()
    if splash is not None:
        splash.finish(w)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--cad-update" in sys.argv and args:       # CAD add-in without a running GUI: project linked to the document
        QTimer.singleShot(100, lambda: w.cad_update_start(Path(args[0])))
    elif args:
        QTimer.singleShot(100, lambda: w.open_file(args[0]))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
