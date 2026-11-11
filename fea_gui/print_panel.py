"""3D printing panel (F4): printed material of the parts, print settings (perimeters, layers, infill, build direction),
properties of the full extrusion, failure criterion, calibration from ISO 527 tensile tests and display of the shell
and the core in a mesh section. The GUI only edits the Study; the calculation is done by fea_core.printing.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from html import escape
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSlider, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from fea_core import materials as matlib
from fea_core.printing import PrintSpec
from fea_core.printing.calibrate import TensileTest, calibrate, iso527_modulus, read_curve
from fea_core.printing.failure import CRITERIA
from fea_core.printing.infill import PATTERNS, core, solid, pattern_label
from fea_core.i18n import tr
from fea_core.study import Material

from .safe_browser import SafeTextBrowser
from .widgets import (Section, check_item, checked, dspin, fit_rows, fmt_dash as fmt, help_tip, panel_header, scroll_body,
                      scroll_to, set_placeholder)

BUILD_DIRS = [("+Z (usual)", [0, 0, 1]), ("−Z", [0, 0, -1]), ("+X", [1, 0, 0]), ("−X", [-1, 0, 0]),
              ("+Y", [0, 1, 0]), ("−Y", [0, -1, 0])]
PRINT_FIELDS = ("E_z", "nu_z", "G_z", "Rm_z", "Rc", "Rc_z", "S_xy", "S_z")
# material form fields where 0 means "not entered" (None); the other material fields are taken over from the original
OPTIONAL_FIELDS = ("Rm", "nu_z", "G_z", "Rm_z", "Rc", "Rc_z", "S_xy", "S_z")
GREEN, GREY = "#2e9e4f", "#9aa1ab"
BUILD_KEY = "print:build"                       # selection in 3D: the build direction arrow and the outline of the build plate
C_BUILD = (220, 120, 20)
C_THIN = (186, 60, 200)                         # thin places with full infill in the "Shell / core" section
PART_ROLE = int(Qt.ItemDataRole.UserRole)       # part name in the part list item (the item text also has the print description)


def ispin(val, lo, hi):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    return s


def print_materials(study) -> list[Material]:
    mats = {m.name: m for m in matlib.all_materials() if m.is_print}
    if study is not None:
        mats.update({k: m for k, m in study.materials.items() if m.is_print})
    return list(mats.values())


class PrintPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._filling = False
        self._mat_base: Material | None = None   # material loaded into the form (keep the fields outside the form)
        self._mat_shown: dict = {}               # form values right after loading (unchanged = original value)
        self._picking = False                    # face selection on the build plate in 3D (Build direction from face)
        self._cen_cache = None                   # (mesh, centroids of elements) for the section through the shell
        self.scroll, lay = scroll_body(self)
        lay.addWidget(panel_header("3D printing", "Shell (perimeters, top/bottom layers) + infill, orthotropy by build direction"))
        self.parts = QListWidget()
        self.parts.setSelectionMode(QAbstractItemView.ExtendedSelection)
        set_placeholder(self.parts, "No part yet – open a model")
        fit_rows(self.parts, 2, 6)
        self.parts.setToolTip("Printed parts – selection for the print settings")
        self.parts.setWhatsThis(
            "<b>Parts and their printing</b><p>List of all parts of the model. For printed parts the material, number of perimeters, infill and build direction are shown in parentheses; non-printed parts and bolts are marked.</p><p>Procedure: select parts, choose a printed material and press <i>Assign</i>, then set the printing in the form and save it with the <i>Apply to selected parts</i> button. Clicking a part in 3D selects it here too (Ctrl = add).</p>")
        self.parts.itemSelectionChanged.connect(self.on_part_sel)
        lay.addWidget(self.parts)
        h = QHBoxLayout()
        self.mat = QComboBox()
        help_tip(self.mat, "Printed material: full-extrusion properties in the layer and perpendicular to it",
                 "Printed material = solid-extrusion properties in the layer plane (XY) and normal to the layers (Z)")
        h.addWidget(self.mat, 1)
        b = QPushButton("Assign")
        help_tip(b, "Assigns the printed material to the selected parts",
                 "Assign the printed material to the parts selected in the list.\nA part with a printed material is analyzed as 3D printing (shell + infill per the settings below).")
        b.clicked.connect(self.assign_material)
        h.addWidget(b)
        lay.addLayout(h)

        # ---------------------------------------------------------- print settings
        g = Section("Print settings of the selected parts", key="print.settings", collapsed=False,
                    tip="Sets the print structure and build direction of the selected parts.",
                    help="<b>Print settings of the selected parts</b><p>Sets the perimeters, solid layers, infill, build direction and surface tolerance. Changes are stored only with the “Apply to selected parts” button.</p>")
        f = QFormLayout(g.body)
        self.s_enabled = QCheckBox("shell + infill (otherwise the whole part is solid)")
        self.s_enabled.setChecked(True)
        help_tip(self.s_enabled, "On = shell + sparse infill, off = whole part solid",
                 "On = print model: shell (perimeters, top/bottom layers) + core with sparse infill.\nOff = the whole part is solid extrusion (100 % infill), orthotropy by build direction stays.")
        f.addRow(self.s_enabled)
        self.s_dir = QComboBox()
        for txt, _ in BUILD_DIRS:
            self.s_dir.addItem(txt)
        self.s_dir.addItem("custom")
        self.s_vec = [dspin(0.0, -1, 1, 3, 0.1) for _ in range(3)]
        for s, ax in zip(self.s_vec, "XYZ"):
            help_tip(s, f"{ax} component of the custom build direction",
                     f"Component {ax} of the custom build direction (the vector is normalized; zero vector = +Z).\nThe field is shown when “custom” is selected.")
        hv = QHBoxLayout()
        hv.addWidget(self.s_dir)
        for s in self.s_vec:
            hv.addWidget(s)
        self.s_dir.currentIndexChanged.connect(self.on_dir)
        help_tip(self.s_dir, "Direction in which layers are added (printer Z axis)",
                 "Printer Z axis (the direction in which layers are added) in model coordinates.\nFor selected printed parts an orange arrow from the bed is drawn in 3D.")
        f.addRow("Build direction (printer Z)", hv)
        h = QHBoxLayout()
        self.btn_face_dir = QPushButton("Build direction from face")
        self.btn_face_dir.setCheckable(True)
        help_tip(self.btn_face_dir, "Build direction from a planar face lying on the bed",
                 "Click a planar face in 3D that lies on the printer bed → build direction = opposite outward normal of the face.\nApplies to the parts selected in the list (with no selection, to the part the face belongs to); other print settings are unchanged. Clicking the button again cancels the selection.")
        self.btn_face_dir.clicked.connect(self.build_dir_from_face)
        h.addWidget(self.btn_face_dir)
        self.face_hint = QLabel()
        self.face_hint.setWordWrap(True)
        self.face_hint.setProperty("muted", True)
        h.addWidget(self.face_hint, 1)
        f.addRow(h)
        self.win.view.facesPicked.connect(self.on_faces_picked)
        self.s_layer = dspin(0.2, 0.02, 2.0, 3, 0.02, " mm")
        help_tip(self.s_layer, "Layer height (usually 0.2 mm)",
                 "Layer height (typically 0.2 mm). Together with the number of top/bottom layers it determines the thickness of the solid shell at the top and bottom.")
        self.s_lw = dspin(0.42, 0.1, 3.0, 3, 0.02, " mm")
        help_tip(self.s_lw, "Width of the extruded line (usually 0.42 mm)",
                 "Extruded line width (typically 0.42 mm for a 0.4 nozzle). Wall thickness = perimeters × line width.")
        self.s_walls = ispin(2, 0, 50)
        help_tip(self.s_walls, "Number of wall perimeter lines (perimeters)",
                 "Number of perimeters (outer wall lines). They form the shell on the sides of the part; perimeter lines run along the surface (along / across line anisotropy).")
        self.s_top = ispin(5, 0, 200)
        self.s_top.setToolTip("Number of solid layers on top faces (normal along the build direction)")
        self.s_bottom = ispin(3, 0, 200)
        self.s_bottom.setToolTip("Number of solid layers on bottom faces (normal against the build direction)")
        self.s_infill = dspin(15.0, 0.0, 100.0, 1, 5.0, " %")
        help_tip(self.s_infill, "Core infill density, 0–100 %",
                 "Core infill density (0–100 %). Core properties = solid extrusion × pattern factors by density (Gibson–Ashby, indicative – calibrate with tests).")
        self.s_pattern = QComboBox()
        help_tip(self.s_pattern, "Infill pattern – sets how core stiffness drops with density",
                 "Infill pattern – determines how core stiffness and strength drop with density (2D patterns have no effect in the Z direction)")
        for k in PATTERNS:
            self.s_pattern.addItem(pattern_label(k), k)
        f.addRow("Layer height", self.s_layer)
        f.addRow("Line width", self.s_lw)
        f.addRow("Perimeters", self.s_walls)
        hl = QHBoxLayout()
        hl.addWidget(QLabel("top"))
        hl.addWidget(self.s_top)
        hl.addWidget(QLabel("bottom"))
        hl.addWidget(self.s_bottom)
        f.addRow("Solid layers", hl)
        f.addRow("Infill", self.s_infill)
        f.addRow("Infill pattern", self.s_pattern)
        ht = QHBoxLayout()
        self.s_solid_thin = QCheckBox("solid infill of thin regions")
        self.s_solid_thin.setChecked(True)
        help_tip(self.s_solid_thin, "Thin regions are computed as full extrusion",
                 "Where the interior of the part after subtracting the perimeters is narrower than the width limit, sparse infill does not fit and the slicer fills it with solid infill.\nOn = such elements are treated as solid extrusion. Off = thin regions also have sparse infill (more conservative, slicers usually do not print it that way).\nINDICATIVE – slicers also decide by the area of the region.")
        ht.addWidget(self.s_solid_thin)
        ht.addWidget(QLabel("width limit"))
        self.s_thin_w = dspin(0.0, 0.0, 50.0, 2, 0.1, " mm", special="auto")
        help_tip(self.s_thin_w, "Interior width below which solid infill is printed (0 = auto)",
                 "Width of the interior (after subtracting the perimeters) below which solid infill is printed.\n0 = automatically 2 × line width.")
        ht.addWidget(self.s_thin_w)
        self.s_thin_auto = QLabel()
        self.s_thin_auto.setProperty("muted", True)
        ht.addWidget(self.s_thin_auto, 1)
        f.addRow("Thin regions", ht)
        self.s_tol = dspin(0.0, 0.0, 5.0, 3, 0.05, " mm", special="no loss")
        self.s_tol.setToolTip("Shift of the free surface inwards by the print tolerance [mm]")
        self.s_tol.setWhatsThis("<b>Print tolerance</b><p>A printed part is usually a few tenths of a millimeter smaller than the model at the surface (bead size, rounding, shrinkage). The entered δ shrinks the free surface of the part by δ on each side in the analysis – thin walls get weaker and the analysis is on the safe side.</p><p>Faces with a support or load stay in place (forces act on the real material). The mesh is unchanged, only the nodes are moved.</p>")
        f.addRow("Tolerance (surface loss)", self.s_tol)
        self.s_tol.valueChanged.connect(self.update_effective)
        for w in (self.s_layer, self.s_lw, self.s_infill):
            w.valueChanged.connect(self.update_effective)
        for w in (self.s_walls, self.s_top, self.s_bottom):
            w.valueChanged.connect(self.update_effective)
        self.s_pattern.currentIndexChanged.connect(self.update_effective)
        self.s_enabled.toggled.connect(self.update_effective)
        self.s_solid_thin.toggled.connect(self.update_effective)
        self.s_thin_w.valueChanged.connect(self.update_effective)
        # presets (fea_core.printing.presets): built-in "3D printing standard" + custom ones next to the user material library
        hp = QHBoxLayout()
        self.preset = QComboBox()
        help_tip(self.preset, "Print settings preset (without build direction)",
                 "Print settings preset: layer height, line width, perimeters, top/bottom layers, infill, pattern, thin regions (the build direction belongs to the part, the preset does not change it)")
        hp.addWidget(self.preset, 1)
        self.btn_preset_fill = QPushButton("To form")
        help_tip(self.btn_preset_fill, "Fills the form with the selected preset",
                 "Fills the form with the selected preset (e.g. 3D print standard: 2 perimeters × 0.42 mm, 5 top / 3 bottom layers × 0.2 mm, grid 15 %).\nThe build direction in the form stays. Saved to the study only with the “Apply to selected parts” button.")
        self.btn_preset_fill.clicked.connect(self.load_preset)
        hp.addWidget(self.btn_preset_fill)
        b = QPushButton("Save as…")
        help_tip(b, "Saves the form settings as your own preset",
                 "Saves the form settings (without build direction) as a custom preset – the print_presets.json file next to the user material library")
        b.clicked.connect(self.save_preset_as)
        hp.addWidget(b)
        self.btn_preset_del = QPushButton("Delete")
        self.btn_preset_del.setToolTip("Deletes the selected custom preset (built-in ones cannot be deleted)")
        self.btn_preset_del.clicked.connect(self.delete_preset)
        hp.addWidget(self.btn_preset_del)
        f.addRow("Preset", hp)
        self.preset_warn = QLabel()
        self.preset_warn.setWordWrap(True)
        self.preset_warn.hide()
        f.addRow(self.preset_warn)
        self.preset.currentIndexChanged.connect(self._on_preset_sel)
        h = QHBoxLayout()
        b = QPushButton("Apply to selected parts")
        b.setToolTip("Saves the form settings to the selected parts")
        b.setWhatsThis(
            "<b>Apply to selected parts</b><p>Writes the print settings from the form (build direction, layers, perimeters, infill, pattern, thin regions) to the study for all parts selected in the list above. Until you press it, neither the form nor a loaded preset changes the study.</p><p>Changing the print settings does not require re-meshing – only the shell and infill assignment is recomputed; the results are discarded.</p>")
        b.clicked.connect(self.apply_spec)
        h.addWidget(b)
        h.addStretch(1)
        f.addRow(h)
        self.eff = QLabel()
        self.eff.setWordWrap(True)
        f.addRow(self.eff)
        lay.addWidget(g)

        # ---------------------------------------------------------- varianty (F5)
        from .print_variants import VariantsBox
        self.variants_box = VariantsBox(self)
        lay.addWidget(self.variants_box)

        # ---------------------------------------------------------- sweep and advice (F5, step 2) – collapsible
        from .print_sweep import AdviceBox, SweepBox
        self.sweep_box = SweepBox(self)
        self.sweep_section = Section("Parameter sweep (perimeters × infill × pattern × direction)", self.sweep_box,
                                     key="print.sweep", collapsed=True,
                                     tip="Compares combinations of print settings on one mesh.",
                                     help="<b>Parameter sweep</b><p>Tries combinations of perimeters, infill, pattern and build direction on the same mesh and compares mass with safety.</p>")
        lay.addWidget(self.sweep_section)
        self.advice_box = AdviceBox(self)
        self.advice_section = Section("Advice by failure location and build direction", self.advice_box,
                                      key="print.advice", collapsed=True,
                                      tip="Recommends the build direction according to the failure location.",
                                      help="<b>Advice based on the failure location</b><p>Evaluates the latest results and ranks the build directions by expected safety.</p>")
        lay.addWidget(self.advice_section)
        self.refresh_presets()

        # ---------------------------------------------------------- material
        g = Section("Printed material – solid extrusion (100 % infill)", key="print.material", collapsed=True,
                    tip="Edits the full-extrusion properties of the printed material.",
                    help="<b>Printed material</b><p>The values apply to full extrusion, not to the effective properties of a part with infill. Select the material above and save changes with the buttons below.</p>")
        f = QFormLayout(g.body)
        self.m_name = QLineEdit()
        help_tip(self.m_name, "Material name in the study (new name = new material)",
                 "Material name in the study. A new name = a new material (the original stays);\na built-in library material cannot be overwritten by “To my library” – choose a different name.")
        f.addRow("Name", self.m_name)
        self.m = {}
        rows = [("E", "E in layer plane (XY)", 3000, " MPa"), ("E_z", "E normal to layers (Z)", 2500, " MPa"),
                ("nu", "ν in plane", 0.36, ""), ("nu_z", "ν_xz (0 = same as ν)", 0.0, ""),
                ("G_z", "G between layers (0 = estimate)", 0.0, " MPa"), ("rho", "ρ", 1240, " kg/m³"),
                ("Rm", "Rm tension XY", 45, " MPa"), ("Rm_z", "Rm tension Z (layer bond)", 27, " MPa"),
                ("Rc", "compressive strength XY (0 = Rm)", 0.0, " MPa"), ("Rc_z", "compressive strength Z (0 = Rc)", 0.0, " MPa"),
                ("S_xy", "in-plane shear (0 = Rm/√3)", 0.0, " MPa"), ("S_z", "interlayer shear (0 = Rm_z/√3)", 0.0, " MPa")]
        mtips = {
            "E": "Young's modulus of solid extrusion in the layer plane (XY) [MPa] = average along and across the line.\nE.g. PLA Bambu Basic 2680 MPa (TDS).",
            "E_z": "Young's modulus normal to the layers (build direction Z) [MPa].\nUsually lower than E in the plane (PLA Bambu Basic 2160 MPa).",
            "nu": "Poisson's ratio in the layer plane ν_xy [–] (plastics typically 0.33–0.4).",
            "nu_z": "Poisson's ratio ν_xz [–]: transverse contraction in the Z direction under tension in the layer plane\n(ε_z = −ν_xz·σ_x/E). 0 = same as ν in the plane.",
            "G_z": "Shear modulus between layers (XZ/YZ plane) [MPa].\n0 = estimate E_z / (2·(1 + ν)).",
            "rho": "Density of solid extrusion [kg/m³] (PLA ~1240). The mass of a part with infill is calculated per element\nfrom the shell fraction and the infill density.",
            "Rm": "Tensile strength in the layer plane (XY) [MPa] = average along and across the line.\nRequired (failure cannot be assessed without it). PLA Bambu Basic 39 MPa.",
            "Rm_z": "Tensile strength normal to the layers = layer bond strength [MPa].\nUsually the weakest print direction (PLA Bambu Basic 35 MPa). 0 = same as Rm XY.",
            "Rc": "Compressive strength in the layer plane [MPa]. 0 = same as Rm XY.",
            "Rc_z": "Compressive strength normal to the layers [MPa]. 0 = same as compressive strength XY (if not given, Rm_z).",
            "S_xy": "Shear strength in the layer plane [MPa]. 0 = Rm XY / √3.",
            "S_z": "Interlayer shear strength (XZ/YZ plane) [MPa]. 0 = Rm_z / √3.",
        }
        mshort = {
            "E": "Young's modulus of full extrusion in the layer plane [MPa]",
            "E_z": "Young's modulus perpendicular to the layers [MPa]",
            "nu": "Poisson's ratio in the layer plane ν_xy",
            "nu_z": "Poisson's ratio ν_xz – contraction in the Z direction",
            "G_z": "Shear modulus between layers [MPa] (0 = estimate)",
            "rho": "Density of full extrusion [kg/m³]",
            "Rm": "Tensile strength in the layer plane [MPa]",
            "Rm_z": "Layer bond strength – tension perpendicular to the layers [MPa]",
            "Rc": "Compressive strength in the layer plane [MPa] (0 = Rm)",
            "Rc_z": "Compressive strength perpendicular to the layers [MPa] (0 = as XY)",
            "S_xy": "Shear strength in the layer plane [MPa] (0 = Rm/√3)",
            "S_z": "Interlayer shear strength [MPa] (0 = Rm_z/√3)",
        }
        for key, lbl, val, suf in rows:
            dec = 3 if key.startswith("nu") else 1
            hi = 0.499 if key.startswith("nu") else 1e6
            w = dspin(val, 0.0, hi, dec, 0.01 if key.startswith("nu") else 10.0, suf)
            help_tip(w, mshort[key], mtips[key])
            self.m[key] = w
            f.addRow(lbl, w)
        self.m_src = QLineEdit()
        self.m_src.setToolTip("Source of the values (data sheet, test, literature) – stated in the report.")
        f.addRow("Source", self.m_src)
        h = QHBoxLayout()
        b = QPushButton("Save and assign to selected")
        help_tip(b, "Saves the material to the study and assigns it to the selected parts",
                 "Saves the material from the form to the study and assigns it to the parts selected in the list\n(with no selection to parts that already have a material of the same name). The results are discarded.")
        b.clicked.connect(self.save_material)
        h.addWidget(b)
        b = QPushButton("To my library")
        help_tip(b, "Saves the material to the user library (for other studies)",
                 "Saves the material from the form into the user material library (for further studies):\n"
                 + str(matlib.user_library_path()))
        b.clicked.connect(self.save_user_material)
        h.addWidget(b)
        f.addRow(h)
        lay.addWidget(g)
        self.mat.currentIndexChanged.connect(self.on_mat_combo)

        # ---------------------------------------------------------- assessment, calibration, display
        g = Section("Assessment and calibration", key="print.assessment", collapsed=True,
                    tip="Sets the failure criterion and the material calibration.",
                    help="<b>Assessment and calibration</b><p>Selects the failure criterion, allows calibrating the material from tests and shows the shell fraction in the mesh.</p>")
        f = QFormLayout(g.body)
        self.crit = QComboBox()
        for k, txt in CRITERIA.items():
            self.crit.addItem(txt, k)
        self.crit.setToolTip("Failure criterion of the printed material")
        self.crit.setWhatsThis(
            "<b>Failure criterion</b><p>The stress in the shell and core is converted at the Gauss points into the print axes (layer, line, build direction) and compared with the strengths of solid extrusion and infill, respectively. The result is the <i>print safety factor</i> = load multiplier to failure (node = minimum over elements).</p><p><i>Tsai-Wu</i> distinguishes tension and compression with a single quadratic expression (default), <i>Tsai-Hill</i> takes the strength according to the sign of each component. Changing the criterion discards the results.</p>")
        self.crit.currentIndexChanged.connect(self.on_crit)
        f.addRow("Failure criterion", self.crit)
        b = QPushButton("Calibration from ISO 527 tests…")
        b.setToolTip("Calibration of full extrusion from tensile tests")
        b.setWhatsThis(
            "<b>Calibration from tests</b><p>Opens a dialog where you enter the measured E and Rm of ISO 527 tensile specimens and their print settings (perimeters, layers, infill). From these the program calculates the solid-extrusion properties with the same shell and infill model as in the analysis and creates a calibrated material.</p><p>XY specimens lie on the bed, Z specimens stand upright (tension normal to the layers). Specimens with 100 % infill are the most accurate.</p>")
        b.clicked.connect(self.open_calibration)
        f.addRow(b)
        h = QHBoxLayout()
        self.cut_axis = QComboBox()
        self.cut_axis.addItems(["section X", "section Y", "section Z"])
        help_tip(self.cut_axis, "Axis of the section plane for showing the shell and core",
                 "Section plane axis for the “Shell / core” view (plane normal to the selected model axis)")
        self.cut_pos = QSlider(Qt.Horizontal)
        self.cut_pos.setRange(0, 1000)
        self.cut_pos.setValue(500)
        help_tip(self.cut_pos, "Position of the section plane within the extent of the printed parts",
                 "Section plane position within the extent of the printed parts (left = start, right = end).\nElements on the side of smaller coordinates are shown; redrawn when the slider is released.")
        self.cut_pos.sliderReleased.connect(self.show_shell)
        self.cut_axis.currentIndexChanged.connect(lambda _: self.show_shell())
        b = QPushButton("Shell / core")
        b.setToolTip("Shell fraction in the mesh elements of printed parts (section through the mesh), mesh from the Mesh step")
        b.setWhatsThis(
            "<b>Shell / core</b><p>Shows a section through the mesh of the printed parts colored by the shell fraction in the element (1 = perimeters / solid layers, 0 = sparse infill). Thin regions that the slicer fills with solid infill are shown in purple.</p><p>Requires the mesh from the Mesh step. The section plane is chosen by the axis and the slider next to the button. A mesh coarse relative to the shell thickness only blurs it – consider local refinement.</p>")
        b.clicked.connect(self.show_shell)
        h.addWidget(b)
        h.addWidget(self.cut_axis)
        h.addWidget(self.cut_pos, 1)
        f.addRow(h)
        # two-step refinement (fea_core.printing.refine.two_step): sphere around the places of the lowest print safety
        h = QHBoxLayout()
        self.btn_refine = QPushButton("Refine mesh at critical locations and recompute")
        self.btn_refine.setToolTip("Refines the mesh at critical places and recomputes")
        self.btn_refine.setWhatsThis(
            "<b>Two-step refinement</b><p>A mesh coarse relative to the perimeter thickness (~0.8 mm) makes the shell stress mesh-dependent. Refining the whole surface is too expensive, so after the first analysis the locations with the lowest print safety factor are found and the mesh is refined only in spheres around them; a second analysis follows.</p><p>If the minimum lies at a sharp inner edge, it is a singularity – the value drops with every refinement; the actual corner fillet decides.</p>")
        self.btn_refine.clicked.connect(self.refine_two_step)
        h.addWidget(self.btn_refine, 1)
        h.addWidget(QLabel("locations"))
        self.refine_n = ispin(3, 1, 8)
        self.refine_n.setToolTip("Number of separate locations of the lowest safety factor around which the mesh is refined")
        h.addWidget(self.refine_n)
        f.addRow(h)
        self.refine_info = QLabel("")
        self.refine_info.setWordWrap(True)
        f.addRow(self.refine_info)
        lay.addWidget(g)

        # ---------------------------------------------------------- advanced (discretisation of the print model)
        adv_section = Section("Advanced (discretization of the shell and lines)", key="print.advanced", collapsed=True,
                              tip="Sets the level of detail of the shell and line direction discretization.",
                              help="<b>Advanced settings</b><p>Controls the number of shell fraction levels, element sampling and line direction classes. The defaults are usually sufficient.</p>")
        self.adv_toggle = adv_section.toggle
        self.adv = adv_section.body
        f = QFormLayout(self.adv)
        self.a_bins = ispin(10, 1, 50)
        help_tip(self.a_bins, "Number of shell fraction levels in an element",
                 "Number of shell fraction levels in an element: the fraction f is rounded to multiples of 1/bins (materials per part = bins + 1).\nMore levels = smoother shell/core transition, more materials in the model. Default 10.")
        self.a_samples = ispin(32, 4, 512)
        help_tip(self.a_samples, "Number of sampling points in an element for the shell fraction",
                 "Number of sampling points in an element for estimating the shell fraction (uniformly, Sobol).\nMore points = more accurate fraction for elements cut by the shell boundary, longer assignment. Default 32.")
        self.a_line_bins = ispin(36, 1, 180)
        help_tip(self.a_line_bins, "Number of perimeter line direction classes",
                 "Number of classes of the perimeter line direction in the layer plane: class = 180°/line_bins (36 → 5°, direction error ≤ 2.5°).\nEach class = its own *ORIENTATION in ccx. Default 36.")
        f.addRow("Shell fraction levels (bins)", self.a_bins)
        f.addRow("Sampling points per element", self.a_samples)
        f.addRow("Line direction classes", self.a_line_bins)
        for w in (self.a_bins, self.a_samples, self.a_line_bins):
            w.editingFinished.connect(self.on_advanced)
        self.adv_section = adv_section                  # collapsing handled by Section (adv = body, adv_toggle = header)
        lay.addWidget(adv_section)

        # ---------------------------------------------------------- lines, creep, S-N (per material)
        g = Section("Perimeters, creep and fatigue of the material (indicative)", key="print.properties", collapsed=True,
                    tip="Edits the indicative properties of perimeters, creep and fatigue.",
                    help="<b>Print properties</b><p>Additional anisotropic, creep and S-N properties of the material. The default values are indicative.</p>")
        f = QFormLayout(g.body)
        self.pp_family = QLabel()
        f.addRow("Family (defaults)", self.pp_family)
        self.pp = {
            "line_E_ratio": dspin(1.0, 0.05, 1.0, 3, 0.05),
            "line_R_ratio": dspin(1.0, 0.05, 1.0, 3, 0.05),
            "creep_a": dspin(0.0, 0.0, 10.0, 4, 0.01),
            "creep_n": dspin(0.4, 0.01, 1.0, 3, 0.05),
            "rupture_beta": dspin(0.0, 0.0, 0.5, 3, 0.01),
            "sn_SD_ratio": dspin(0.1, 0.0, 1.0, 3, 0.01),
            "sn_SD": dspin(0.0, 0.0, 1000.0, 2, 1.0, " MPa", special="from ratio"),
            "sn_ND": dspin(2e6, 1e3, 1e10, 0, 1e5),
            "sn_k": dspin(5.5, 1.0, 50.0, 2, 0.5),
            "sn_k2": dspin(10.0, 0.0, 100.0, 2, 1.0),
            "sn_z_ratio": dspin(0.0, 0.0, 2.0, 3, 0.05, special="= Rm_z/Rm"),
        }
        tips = {
            "line_E_ratio": "E across line / E along line in perimeters (1 = no line anisotropy).\nE of the material (XY) = average of along and across.",
            "line_R_ratio": "Tensile strength across / along the line (PLA ~0.55: Khosravani et al. 2022).",
            "creep_a": "Findley: φ(t) = a·t^n (t in hours, 23 °C); E_c = E/(1 + φ)",
            "creep_n": "Exponent of Findley's law φ(t) = a·t^n [–] (PLA, PETG ~0.4).\nDetermines how fast creep grows with load duration.",
            "rupture_beta": "Strength under sustained load: k_R = 1 − β·log10(1 + t)",
            "sn_SD_ratio": "S-N amplitude at ND as a fraction of Rm XY (Ezeh & Susmel 2019: 0.1, P_s 95 %)",
            "sn_SD": "S-N stress amplitude at ND in the layer plane (XY) [MPa].\nThe entered value takes precedence over the SD / Rm ratio; 0 = from the ratio.",
            "sn_ND": "Number of cycles at the S-N knee point to which SD belongs (Ezeh & Susmel 2019: 2·10⁶).",
            "sn_k": "S-N curve slope above the knee (N < ND): N = ND·(SD/σa)^k (PLA 5.5).",
            "sn_k2": "S-N curve slope below the knee (N > ND). Plastics have no fatigue limit → default 10.\n0 = fatigue limit (no damage below SD).",
            "sn_z_ratio": "SD_z / SD_xy (0 = same as the static strength ratio Rm_z/Rm)",
        }
        labels = {"line_E_ratio": "Line: E across / along", "line_R_ratio": "Line: Rm across / along",
                  "creep_a": "Creep: a [h^-n]", "creep_n": "Creep: n", "rupture_beta": "Long-term strength: β",
                  "sn_SD_ratio": "S-N: SD / Rm", "sn_SD": "S-N: SD (priority)", "sn_ND": "S-N: ND",
                  "sn_k": "S-N: slope k", "sn_k2": "S-N: slope k2 below knee", "sn_z_ratio": "S-N: SD_z / SD_xy"}
        tshort = {
            "line_E_ratio": "E across the line / E along the line in perimeters",
            "line_R_ratio": "Strength across / along the line (PLA ~0.55)",
            "creep_a": "Findley: φ(t) = a·t^n (t in hours, 23 °C)",
            "creep_n": "Exponent of the Findley law φ(t) = a·t^n",
            "rupture_beta": "Long-term strength: k_R = 1 − β·log10(1 + t)",
            "sn_SD_ratio": "S-N amplitude at ND as a fraction of Rm XY",
            "sn_SD": "S-N stress amplitude at ND in the layer plane [MPa]",
            "sn_ND": "Number of cycles at the knee of the S-N curve",
            "sn_k": "Slope of the S-N curve above the knee (N < ND)",
            "sn_k2": "Slope of the S-N curve below the knee (0 = fatigue limit)",
            "sn_z_ratio": "SD_z / SD_xy (0 = as the ratio Rm_z/Rm)",
        }
        for k, w in self.pp.items():
            if k in tips:
                help_tip(w, tshort[k], tips[k])
            f.addRow(labels[k], w)
        self.pp_T = QLabel()
        self.pp_T.setWordWrap(True)
        f.addRow(self.pp_T)
        h = QHBoxLayout()
        b = QPushButton("Apply to material")
        b.setToolTip("Saves the values to the study for the current printed material (PrintSettings.materials)")
        b.clicked.connect(self.apply_props)
        h.addWidget(b)
        b = QPushButton("Default (literature)")
        help_tip(b, "Restores the default properties for the material family",
                 "Deletes the override in the study and restores the defaults by material family (PLA/PETG/ABS)\nfrom the sources listed below the buttons")
        b.clicked.connect(self.reset_props)
        h.addWidget(b)
        f.addRow(h)
        self.pp_src = QLabel()
        self.pp_src.setWordWrap(True)
        self.pp_src.setProperty("muted", True)
        f.addRow(self.pp_src)
        lay.addWidget(g)

        self.cr_on = QCheckBox("estimate creep of long-term cases")      # in the section header as "compute"
        g = Section("Creep – long-term load (isochronous modulus)", check=self.cr_on,
                    tip="Estimates creep of the printed parts in long-term states.",
                    help="<b>Creep</b><p>After the calculation the checked states are recomputed on the same mesh with reduced modulus and strength according to time and temperature.</p>")
        f = QFormLayout(g.body)
        help_tip(self.cr_on, "Recomputes long-term states with reduced modulus and strength",
                 "On = after the analysis the long-term cases are recomputed with stiffnesses × k_E/(1 + φ) and strengths × k_R\n(results as additional steps “<case> – creep t h, T °C”).")
        self.cr_hours = dspin(1000.0, 0.01, 1e6, 1, 100.0, " h")
        help_tip(self.cr_hours, "Duration of sustained load [h]",
                 "Duration of sustained load [h] for creep φ(t) = a·t^n and long-term strength k_R.\nE.g. 1000 h ≈ 6 weeks, 8760 h = 1 year.")
        self.cr_T = dspin(23.0, -40.0, 150.0, 1, 1.0, " °C")
        help_tip(self.cr_T, "Part temperature during loading [°C]",
                 "Part temperature during loading [°C] – per the material table it reduces the modulus (k_E) and speeds up creep (k_φ).\nA temperature above the table's validity (e.g. PLA 45 °C, PETG 60 °C) = analysis error.")
        f.addRow("Load duration", self.cr_hours)
        f.addRow("Part temperature", self.cr_T)
        self.cr_cases = QListWidget()
        set_placeholder(self.cr_cases, "No load case yet")
        fit_rows(self.cr_cases, 2, 5)
        help_tip(self.cr_cases, "Sustained states are computed with reduced modulus",
                 "Checked load cases act permanently – they are recomputed with a reduced modulus")
        f.addRow("Long-term cases", self.cr_cases)
        note = QLabel("<i>Estimate by linear viscoelasticity (up to ~1/3 of the strength): E/(1 + φ(t, T)), strength × k_R. It does not capture nonlinear creep, bolt preload relaxation or moisture – indicative only.</i>")
        note.setWordWrap(True)
        f.addRow(note)
        self.cr_on.toggled.connect(self.on_creep)
        self.cr_hours.editingFinished.connect(self.on_creep)
        self.cr_T.editingFinished.connect(self.on_creep)
        self.cr_cases.itemChanged.connect(self.on_creep)
        lay.addWidget(g)
        from .legend import Legend
        self.legend = Legend()
        self.legend.setMinimumHeight(200)
        self.legend.hide()
        lay.addWidget(self.legend)
        self.info = SafeTextBrowser()
        self.info.setMinimumHeight(120)
        self.info.setVisible(False)
        lay.addWidget(self.info)
        lay.addStretch(1)

    # ================================================================== obnova
    @property
    def study(self):
        return self.win.study

    def part_names(self) -> list[str]:
        return [p.name for p in self.win.parts]

    def selected_parts(self) -> list[str]:
        """Names of the selected parts from the item data of the list – NOT an index into win.parts: after another model is opened the list
        keeps the rows of the previous model until refresh() (fewer parts → IndexError, other parts → foreign names).
        """
        items = sorted(self.parts.selectedItems(), key=self.parts.row)
        return [str(n) for n in (it.data(PART_ROLE) for it in items) if n is not None]

    def refresh(self):
        st = self.study
        if st is None:
            return
        self._filling = True
        try:
            sel = set(self.selected_parts())
            self.parts.clear()
            excl = st.excluded_parts()
            for name in self.part_names():
                if name in excl:
                    txt = f"{name}   [bolt / excluded]"
                else:
                    m = st.material_of(name)
                    if not m.is_print:
                        txt = f"{name}   [{m.name} – not printed]"
                    else:
                        sp = st.printing.spec(name)
                        how = (f"{sp.walls}× perimeter, infill {fmt(100 * sp.infill, 3)} % {pattern_label(sp.pattern)}"
                               if sp.enabled else "solid")
                        if getattr(sp, "tolerance", 0.0):
                            how += f", tolerance {fmt(sp.tolerance, 3)} mm"
                        txt = f"{name}   [{m.name}; {how}; Z = {self.dir_text(sp.build_dir)}]"
                item = QListWidgetItem(txt)
                item.setData(PART_ROLE, name)
                self.parts.addItem(item)
            for i, name in enumerate(self.part_names()):
                if name in sel:
                    self.parts.item(i).setSelected(True)
            cur = self.mat.currentText()
            self.mat.blockSignals(True)
            self.mat.clear()
            self.mat.addItems([m.name for m in print_materials(st)])
            if cur:
                self.mat.setCurrentText(cur)
            self.mat.blockSignals(False)
            self.crit.blockSignals(True)
            self.crit.setCurrentIndex(max(0, self.crit.findData(st.printing.criterion)))
            self.crit.blockSignals(False)
            ps = st.printing
            for w, v in ((self.a_bins, ps.bins), (self.a_samples, ps.samples), (self.a_line_bins, ps.line_bins)):
                w.blockSignals(True)
                w.setValue(int(v))
                w.blockSignals(False)
            cr = st.printing.creep
            self.cr_on.setChecked(cr.enabled)
            self.cr_hours.setValue(cr.hours)
            self.cr_T.setValue(cr.temperature)
            self.cr_cases.blockSignals(True)
            self.cr_cases.clear()
            for lc in st.load_cases:
                check_item(self.cr_cases, lc.name, lc.name in cr.cases)
            self.cr_cases.blockSignals(False)
        finally:
            self._filling = False
        self.on_mat_combo()
        self.update_effective()
        self.update_info()
        self.variants_box.refresh()
        self.sweep_box.refresh()
        self.advice_box.refresh()
        if self._picking and self.win.view.mode != "faces":     # the step was left in the meantime
            self.stop_face_pick()
        # the build direction arrow only after the sets are redrawn by the main window (refresh_sets deletes them)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self.show_build_dir)

    @staticmethod
    def dir_text(v) -> str:
        for txt, d in BUILD_DIRS:
            if np.allclose(np.asarray(v, float) / np.linalg.norm(v), d):
                return txt.split()[0]
        return "(" + "; ".join(fmt(float(x), 3) for x in v) + ")"

    def color(self) -> str:
        """Step colour: grey = no printed part, green = set."""
        st = self.study
        if st is None:
            return GREY
        from fea_core.printing.region import print_parts
        return GREEN if print_parts(st, self.part_names()) else GREY

    # ================================================================== parts and settings
    def on_part_sel(self):
        names = self.selected_parts()
        self.show_build_dir()
        if not names or self.study is None:
            return
        st = self.study
        self.fill_spec(st.printing.spec(names[0]))
        m = st.material_of(names[0])
        if m.is_print:
            self.mat.setCurrentText(m.name)

    # ================================================================== build direction in 3D
    def in_stage(self) -> bool:
        win = self.win
        return getattr(win, "stage", -1) == win.STAGE["3D printing"]

    def printed_selected(self) -> list[str]:
        st = self.study
        if st is None:
            return []
        excl = st.excluded_parts()
        out = []
        for n in self.selected_parts():
            try:
                if n not in excl and st.material_of(n).is_print:
                    out.append(n)
            except KeyError:
                pass
        return out

    def build_dir_geometry(self, name: str):
        """Build direction arrow of a part: (foot on the plate, direction, length, radius of the plate outline) or None."""
        from fea_core.geometry import bbox
        part = next((p for p in self.win.parts if p.name == name), None)
        if part is None:
            return None
        d = np.asarray(self.study.printing.spec(name).build_dir, float)
        if np.linalg.norm(d) < 1e-12:
            return None
        d = d / np.linalg.norm(d)
        bb = np.asarray(bbox(part.shape), float)
        lo, hi = bb[:3], bb[3:]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        c = 0.5 * (lo + hi)
        s = corners @ d
        diag = float(np.linalg.norm(hi - lo)) or 1.0
        foot = c + (s.min() - c @ d) * d                       # projection of the centre onto the plane of the build plate (lowest point)
        L = max(1.3 * float(s.max() - s.min()), 0.35 * diag)    # the arrow sticks out above the part
        return foot, d, L, 0.45 * diag

    def show_build_dir(self):
        """Orange build direction arrow (from the plate) and the plate outline for the printed parts selected in the list;
        nothing outside the 3D printing step (the main window deletes the 3D sets when the step changes).
        """
        v = self.win.view
        v.remove_face_set(BUILD_KEY)
        if self.study is not None and self.in_stage():
            for name in self.printed_selected():
                g = self.build_dir_geometry(name)
                if g is None:
                    continue
                foot, d, L, r = g
                tip = foot + L * d
                e1 = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(d, e1)
                q = [foot + r * (a * e1 + b * e2) for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
                v.add_lines(BUILD_KEY, [(tip, foot)], C_BUILD, 3.0, label=f"build direction {self.dir_text(d)}")
                v.add_arrows(BUILD_KEY, [(tip, d, 0.25 * L)], C_BUILD)
                v.add_lines(BUILD_KEY, [(q[i], q[(i + 1) % 4]) for i in range(4)], C_BUILD, 1.5)
        v._update()

    def build_dir_from_face(self):
        """"Build direction from face" button: selected planar face (on the plate) → build_dir = −outward normal;
        without a selected face it starts a selection by click in 3D.
        """
        if self.study is None:
            self.btn_face_dir.setChecked(False)
            return
        if self._picking:                                  # second click = cancel the selection
            self.stop_face_pick()
            return
        v = self.win.view
        faces = v.selected_faces() if v.mode == "faces" else []
        planar = [f for f in faces if self.win.face_prop(*f)["type"] == "plane"]
        if planar:
            self.btn_face_dir.setChecked(False)
            self.set_build_from_face(*planar[-1])
            return
        if faces:
            self.btn_face_dir.setChecked(False)
            QMessageBox.information(self, "3D printing", "The selected face is not planar – select a planar face lying on the printer bed.")
            return
        self._picking = True
        self.btn_face_dir.setChecked(True)
        if v.pick_filter != "face":
            self.win.set_pick_filter("face")
        v.set_mode("faces")
        v.set_selected_faces([])
        self.face_hint.setText("Click a planar face in 3D that lies on the printer bed (click the button again = cancel).")

    def stop_face_pick(self):
        self._picking = False
        self.btn_face_dir.setChecked(False)
        self.face_hint.setText("")
        v = self.win.view
        if self.in_stage():
            v.set_selected_faces([])
            v.set_mode("model")

    def on_faces_picked(self, faces):
        if not self._picking:
            return
        if not self.in_stage():
            self.stop_face_pick()
            return
        if not faces:
            return
        pi, fi = faces[-1]
        if self.win.face_prop(pi, fi)["type"] != "plane":
            self.face_hint.setText("! The face is not planar – click a planar face lying on the bed.")
            self.win.view.set_selected_faces([])
            return
        self.stop_face_pick()
        self.set_build_from_face(pi, fi)

    def set_build_from_face(self, pi: int, fi: int):
        """Build direction = −(outward normal) of the planar face (pi, fi) for the parts selected in the list, otherwise for the part of the face.
        Only build_dir changes, the other print settings of the parts stay.
        """
        st = self.study
        n = np.asarray(self.win.face_prop(pi, fi)["normal"], float)
        if np.linalg.norm(n) < 1e-12:
            QMessageBox.information(self, "3D printing", "The face normal cannot be determined.")
            return
        d = -n / np.linalg.norm(n)
        d = np.where(np.abs(d) < 1e-9, 0.0, np.round(d, 9))   # OCC noise (−0.0, 1e-17) removed, axes exact
        d = (d / np.linalg.norm(d)).tolist()
        targets = self.selected_parts() or [self.win.parts[pi].name]
        changed = False
        for name in targets:
            sp = st.printing.spec(name)
            if np.allclose(sp.build_dir, d, atol=1e-12) and name in st.printing.parts:
                continue
            st.printing.parts[name] = replace(sp, part=name, build_dir=list(d))
            changed = True
        self.win.log_msg(f"Build direction from face {self.dir_text(d)}: {', '.join(targets)}")
        if changed:
            self.win.changed()
        self.fill_spec(st.printing.spec(targets[0]))
        self.show_build_dir()

    def fill_spec(self, sp: PrintSpec):
        self._filling = True
        try:
            self.s_enabled.setChecked(sp.enabled)
            k = next((i for i, (_, d) in enumerate(BUILD_DIRS)
                      if np.allclose(np.asarray(sp.build_dir, float) / np.linalg.norm(sp.build_dir), d)), len(BUILD_DIRS))
            self.s_dir.setCurrentIndex(k)
            for s, v in zip(self.s_vec, sp.build_dir):
                s.setValue(float(v))
            self.on_dir(k)
            self.s_layer.setValue(sp.layer_height)
            self.s_lw.setValue(sp.line_width)
            self.s_walls.setValue(sp.walls)
            self.s_top.setValue(sp.top_layers)
            self.s_bottom.setValue(sp.bottom_layers)
            self.s_infill.setValue(100 * sp.infill)
            self.s_pattern.setCurrentIndex(max(0, self.s_pattern.findData(sp.pattern)))
            self.s_solid_thin.setChecked(bool(getattr(sp, "solid_thin", True)))
            self.s_thin_w.setValue(float(getattr(sp, "thin_width", 0.0) or 0.0))
            self.s_tol.setValue(float(getattr(sp, "tolerance", 0.0) or 0.0))
        finally:
            self._filling = False
        self.update_effective()

    def on_dir(self, k):
        custom = k >= len(BUILD_DIRS)
        for s in self.s_vec:
            s.setVisible(custom)
        if not custom:
            for s, v in zip(self.s_vec, BUILD_DIRS[k][1]):
                s.setValue(float(v))

    def spec_from_ui(self, part: str) -> PrintSpec:
        v = [s.value() for s in self.s_vec]
        if np.linalg.norm(v) == 0:
            v = [0.0, 0.0, 1.0]
        return PrintSpec(part, build_dir=v, layer_height=self.s_layer.value(), line_width=self.s_lw.value(),
                         walls=self.s_walls.value(), top_layers=self.s_top.value(), bottom_layers=self.s_bottom.value(),
                         infill=self.s_infill.value() / 100.0, pattern=self.s_pattern.currentData(),
                         enabled=self.s_enabled.isChecked(), solid_thin=self.s_solid_thin.isChecked(),
                         thin_width=self.s_thin_w.value(), tolerance=self.s_tol.value())

    def apply_spec(self):
        names = self.selected_parts()
        if not names:
            QMessageBox.information(self, "3D printing", "Select parts in the list.")
            return
        for n in names:
            self.study.printing.parts[n] = self.spec_from_ui(n)
        self.win.changed()

    # ================================================================== two-step refinement (F5 step 4)
    def refine_two_step(self):
        import time
        from .workers import TaskThread
        win = self.win
        res = win.results
        if res is None or not any("print_safety" in s.fields for s in res.steps):
            QMessageBox.information(self, "3D printing", "First run the study analysis with printed parts (Solve step) – the refinement is based on the locations of the lowest print safety factor.")
            return
        if win.busy():
            QMessageBox.information(self, "3D printing", "Another analysis is running – wait for it to finish.")
            return
        st = win.study_for_solve()
        st.geometry = win.geometry_str()
        sig = win.study_signature()
        n = int(self.refine_n.value())
        wd = win._workdir / f"zjemneni_tisku_{int(time.time() * 1000)}"

        def job(t):
            from fea_core.printing.refine import two_step
            return two_step(st, wd, results=res, n_spots=n, on_log=t.log.emit, cancel=t.cancel)
        t = TaskThread(job, self)
        t.log.connect(win.log_msg)
        t.done.connect(lambda out: self.on_refine_done(out, sig))
        t.failed.connect(self.on_refine_failed)
        t.finished.connect(lambda: self._refine_finished(t))
        win.register_aux(t)
        self._refine_thread = t
        self.btn_refine.setEnabled(False)
        self.refine_info.setText("Computing the refined mesh…")
        win.status_progress.start("Print refinement – 2nd analysis", 0.95)
        win.log_msg(f"Print refinement: spheres around {n} locations of the lowest safety factor, 2nd analysis")
        t.start()

    def _refine_finished(self, t):
        self.win.unregister_aux(t)
        self._refine_thread = None
        self.btn_refine.setEnabled(True)

    def on_refine_failed(self, text):
        self.win.mesh_progress_finish("error", "Print refinement failed")
        self.refine_info.setText(f"<span style='color:#d23b3b'>Refinement failed: {text}</span>")
        self.win.log_msg(f"ERROR in print refinement: {text}")
        self.win.messages.add("error", "solver", f"Print refinement failed: {text}", "3D printing")

    def on_refine_done(self, out, sig):
        win = self.win
        win.mesh_progress_finish("done", "Print refinement done")
        a, b = out.get("min_safety_before"), out.get("min_safety_after")
        ch = out.get("change")
        rows = [f"Min. print safety: {fmt(a, 4)} → <b>{fmt(b, 4)}</b>"
                + (f" ({'+' if ch >= 0 else ''}{fmt(100 * ch, 3)} %)" if ch is not None else "")
                + f"; nodes {out.get('nodes_before', 0):,} → {out.get('nodes_after', 0):,}".replace(",", " ")
                + (f"; refinement in {len(out['balls'])} spheres (element {fmt(min((x['size'] for x in out['balls'])), 3)} mm)" if out.get("balls") else "")]
        if out.get("singular"):
            rows.append("<span style='color:#d23b3b'>! The minimum is a singularity (a sharp inner edge)"
                        + (f" – outside it the min. safety factor is {fmt(out['min_safety_regular'], 4)}"
                           if out.get("min_safety_regular") is not None else "") + ".</span>")
        for w in out.get("warnings", []):
            rows.append(f"<span style='color:#b36b00'>! {w}</span>")
            win.messages.add("warning", "results", "Print refinement: " + w, "3D printing")
        self.refine_info.setText("<br>".join(rows))
        win.log_msg("Print refinement: " + rows[0].replace("<b>", "").replace("</b>", ""))
        if win.study is None or win.study_signature() != sig:
            win.log_msg("The study changed during refinement – the refined results are not adopted.")
            return
        if not win.ask("Refinement done: min. print safety " + f"{fmt(a, 4)} → {fmt(b, 4)}.\n\nTake over the local mesh refinement into the study (the Mesh step) and show the new results?"):
            return
        win.study.mesh.local = list(win.study.mesh.local) + [
            {"center": list(map(float, x["center"])), "radius": float(x["radius"]), "size": float(x["size"])}
            for x in out["balls"]]
        res2 = out["second"]
        win.results = res2
        win.mesh = res2.mesh
        win._mesh_sig = win.study_signature(win.MESH_KEYS)
        win.fat_panel.result = None
        win.joints_panel.assessment = None
        win.dirty = True
        win.push_undo()
        win.update_title()
        win.messages.set_source("results", [("warning", w) for w in res2.warnings], unique=True)
        win.mesh_messages(res2.mesh)
        win.refresh_all()
        win.set_stage(win.STAGE["Results"])

    def scroll_to(self, w: QWidget):
        """Moves the panel so that the widget is visible (section opened from the advice)."""
        scroll_to(self.scroll, w, deferred=True, margin=0)

    # ================================================================== print settings presets
    @staticmethod
    def preset_problem() -> str:
        """Czech message about a faulty presets file (unreadable / skipped records), otherwise ""."""
        import json
        from fea_core.printing.presets import list_presets, presets_path
        p = presets_path()
        if not p.exists():
            return ""
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            return (f"Preset file {p} cannot be read ({e}) – custom presets are not available; fix or rename it (saving will not overwrite it).")
        if not isinstance(raw, list):
            return f"Preset file {p} does not have the expected form (a list of presets) – custom presets were not loaded."
        bad = len([d for d in raw if isinstance(d, dict)]) - (len(list_presets()) - 1)
        bad += len([d for d in raw if not isinstance(d, dict)])
        return (f"Preset file {p}: {bad} faulty records skipped (they remain in the file)." if bad > 0 else "")

    def refresh_presets(self, select: str | None = None):
        from fea_core.printing.presets import list_presets
        cur = select or self.preset.currentData()
        self.preset.blockSignals(True)
        self.preset.clear()
        for pr in list_presets():
            s = pr.settings
            label = pr.name + ("  (built-in)" if pr.builtin else "")
            self.preset.addItem(label, pr.name)
            tip = (f"{s.get('walls')} perimeters × {fmt(s.get('line_width'), 3)} mm, layers {s.get('top_layers')} / {s.get('bottom_layers')} × {fmt(s.get('layer_height'), 3)} mm, infill {fmt(100 * s.get('infill', 0), 3)} % {pattern_label(s.get('pattern', 'grid'))}"
                   + ("" if s.get("solid_thin", True) else ", thin regions without solid infill"))
            if pr.note:
                tip += "\n" + pr.note
            self.preset.setItemData(self.preset.count() - 1, tip, Qt.ToolTipRole)
        i = self.preset.findData(cur) if cur else -1
        self.preset.setCurrentIndex(max(0, i))
        self.preset.blockSignals(False)
        warn = self.preset_problem()
        self.preset_warn.setText("! " + warn if warn else "")
        self.preset_warn.setVisible(bool(warn))
        self._on_preset_sel()

    def _on_preset_sel(self, *_):
        from fea_core.printing.presets import BUILTIN_NAME
        self.btn_preset_del.setEnabled(self.preset.currentData() not in (None, BUILTIN_NAME))

    def load_preset(self):
        """Preset → form (build direction and "shell + infill" stay); to the study "Apply to selected parts"."""
        from fea_core.printing.presets import PresetError, get_preset
        name = self.preset.currentData()
        if not name:
            return
        try:
            pr = get_preset(name)
        except PresetError as e:
            QMessageBox.warning(self, "Print presets", str(e))
            return
        d = self.spec_from_ui("").to_dict()
        d.update(pr.settings)
        self.fill_spec(PrintSpec.from_dict(d))
        self.win.log_msg(f"Preset '{name}' in the form – “Apply to selected parts” saves it to the study.")

    def save_preset_as(self):
        from fea_core.printing.presets import BUILTIN_NAME, PresetError, list_presets, preset_from_spec, save_preset
        cur = self.preset.currentData()
        name, ok = QInputDialog.getText(self, "Save print preset", "Preset name:",
                                        text="" if cur in (None, BUILTIN_NAME) else cur)
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in {p.name for p in list_presets() if not p.builtin} and \
                not self.win.ask(f"Preset '{name}' already exists. Overwrite?"):
            return
        try:
            save_preset(name, preset_from_spec(self.spec_from_ui("")))
        except PresetError as e:                     # built-in name, unreadable file (it is not overwritten)
            QMessageBox.critical(self, "Print presets", str(e))
            return
        from fea_core.printing.presets import presets_path
        self.win.log_msg(f"Print preset '{name}' saved to {presets_path()}")
        self.refresh_presets(select=name)

    def delete_preset(self):
        from fea_core.printing.presets import BUILTIN_NAME, PresetError, delete_preset
        name = self.preset.currentData()
        if not name:
            return
        if name == BUILTIN_NAME:
            QMessageBox.information(self, "Print presets", "A built-in preset cannot be deleted.")
            return
        if not self.win.ask(f"Delete print preset '{name}'?"):
            return
        try:
            delete_preset(name)
        except PresetError as e:
            QMessageBox.critical(self, "Print presets", str(e))
            return
        self.win.log_msg(f"Print preset '{name}' deleted.")
        self.refresh_presets()

    def update_effective(self):
        """Shell thicknesses and effective core properties for the current settings and material."""
        if self._filling:
            return
        m = self.material_from_ui()
        sp = self.spec_from_ui("")
        self.s_thin_w.setEnabled(sp.solid_thin)
        self.s_thin_auto.setText(f"(auto = 2 × line width = {fmt(2.0 * sp.line_width, 3)} mm)" if sp.thin_width <= 0
                                 else "")
        if m is None:
            self.eff.setText("")
            return
        s = solid(m)
        rows = [f"Shell: walls {fmt(sp.wall_thickness, 3)} mm, top {fmt(sp.top_layers * sp.layer_height, 3)} mm, bottom {fmt(sp.bottom_layers * sp.layer_height, 3)} mm"]
        if sp.enabled and sp.infill < 0.99:
            c = core(m, sp.pattern, sp.infill)
            rows.append(f"Core: E_xy {fmt(c.E1)} MPa ({fmt(100 * c.E1 / s.E1, 3)} %), E_z {fmt(c.E3)} MPa ({fmt(100 * c.E3 / s.E3, 3)} %),<br>Rm_xy {fmt(c.Xt, 3)} MPa, Rm_z {fmt(c.Zt, 3)} MPa")
        else:
            rows.append("Core: solid extrusion")
        if sp.enabled and sp.solid_thin:
            rows.append(f"Thin regions: interior narrower than {fmt(sp.thin_limit, 3)} mm (after subtracting the perimeters) = solid infill")
        rows.append("<i>The infill factors are indicative – calibrate with tests.</i>")
        self.eff.setText("<br>".join(tr(r) for r in rows))     # the "Core" line has <br> inside → line by line

    # ================================================================== material
    def on_mat_combo(self, *_):
        name = self.mat.currentText()
        m = next((x for x in print_materials(self.study) if x.name == name), None)
        if m is None:
            return
        # fields outside the form (Re, λ, α, c_p…) are edited in the Materials step – here they are only kept
        self._mat_base = m
        self._filling = True
        try:
            self.m_name.setText(m.name)
            for k, w in self.m.items():
                v = getattr(m, k)
                w.setValue((v or 0.0) * (1e12 if k == "rho" else 1.0))
            self.m_src.setText(m.source)
            self._mat_shown = {k: w.value() for k, w in self.m.items()}
        finally:
            self._filling = False
        self.fill_props(m.name)
        self.update_effective()

    # ================================================================== lines, creep, S-N
    def fill_props(self, mat_name: str, default: bool = False):
        from fea_core.printing.props import PrintProps, default_props
        st = self.study
        pr = default_props(mat_name)
        if not default and st is not None and mat_name in st.printing.materials:
            pr = PrintProps.from_dict(dict(pr.to_dict(), **st.printing.materials[mat_name]))
        self._filling = True
        try:
            for k, w in self.pp.items():
                v = getattr(pr, k)
                w.setValue(0.0 if v is None else float(v))
        finally:
            self._filling = False
        self.pp_family.setText(pr.family)
        T = "; ".join(f"{fmt(r[0], 3)} °C: k_E {fmt(r[1], 3)}, k_φ {fmt(r[2], 3)}" for r in pr.creep_T)
        fac = pr.creep_factors(1000.0, 23.0)
        self.pp_T.setText(f"Temperature (valid up to {fmt(fac['T_max'], 3)} °C): {T}<br>1000 h at 23 °C: φ = {fmt(fac['phi'], 3)}, E_c/E = {fmt(fac['stiffness'], 3)}, strength × {fmt(fac['strength'], 3)}")
        self.pp_src.setText(pr.source)

    def apply_props(self):
        st = self.study
        name = self.m_name.text().strip()
        if st is None or not name:
            return
        d = {k: w.value() for k, w in self.pp.items()}
        for k in ("sn_SD", "sn_z_ratio"):
            d[k] = d[k] or None
        st.printing.materials[name] = d
        self.win.changed()
        self.fill_props(name)

    def reset_props(self):
        st = self.study
        name = self.m_name.text().strip()
        if st is not None and name in st.printing.materials:
            del st.printing.materials[name]
            self.win.changed()
        self.fill_props(name, default=True)

    def on_creep(self, *_):
        if self._filling or self.study is None:
            return
        cr = self.study.printing.creep
        new = (self.cr_on.isChecked(), self.cr_hours.value(), self.cr_T.value(), checked(self.cr_cases))
        if new == (cr.enabled, cr.hours, cr.temperature, list(cr.cases)):
            return                                  # editingFinished without a change of value does not discard the results
        cr.enabled, cr.hours, cr.temperature, cr.cases = new
        self.win.changed()

    def material_from_ui(self) -> Material | None:
        """Material from the form = copy of the material loaded into the form with only the edited fields changed
        (Re, thermal properties and other fields outside the form are kept).
        """
        g = {k: w.value() for k, w in self.m.items()}
        if g["E"] <= 0 or g["E_z"] <= 0:
            return None
        base = self._mat_base
        vals = {}
        for k, x in g.items():
            if base is not None and k in self._mat_shown and abs(x - self._mat_shown[k]) < 1e-12:
                vals[k] = getattr(base, k)                 # unchanged → original value (without rounding of the spin box)
            elif k == "rho":
                vals[k] = x * 1e-12
            elif k in OPTIONAL_FIELDS:
                vals[k] = x or None
            else:
                vals[k] = x
        name = self.m_name.text().strip() or "Print"
        if base is None:
            return Material(name, source=self.m_src.text(), **vals)
        return replace(base, name=name, source=self.m_src.text(), **vals)

    def _assign(self, m: Material, names: list[str]):
        """Material to the study and to the parts. Unused materials are NOT deleted here: a calibrated or stored material without
        parts would otherwise disappear from the menu (if it is not in the library) and the links of variants / overrides of printing.materials
        would break; an unassigned material in the study does no harm (the report states only the used ones).
        """
        st = self.study
        st.materials[m.name] = m
        for n in names:
            st.part_materials[n] = m.name
            st.printing.parts.setdefault(n, self.spec_from_ui(n))
        self.win.changed()

    def assign_material(self):
        names = self.selected_parts()
        if not names:
            QMessageBox.information(self, "3D printing", "Select parts in the list.")
            return
        name = self.mat.currentText()
        m = next((x for x in print_materials(self.study) if x.name == name), None)
        if m is not None:
            self._assign(Material(**m.__dict__), names)

    def save_material(self):
        if self.study is None:
            return
        m = self.material_from_ui()
        if m is None:
            QMessageBox.warning(self, "3D printing", "Enter both E and E_z.")
            return

        def mat_name(n):
            try:
                return self.study.material_of(n).name
            except KeyError:
                return None
        names = self.selected_parts() or [n for n in self.part_names() if mat_name(n) == m.name]
        self._assign(m, names)

    def save_user_material(self):
        m = self.material_from_ui()
        if m is None:
            QMessageBox.warning(self, "3D printing", "Enter both E and E_z.")
            return
        if m.name in {x.name for x in matlib.LIBRARY}:
            QMessageBox.warning(self, "3D printing", "A built-in material cannot be overwritten – choose a different name.")
            return
        try:
            matlib.save_user_material(m)
        except matlib.UserLibraryError as e:          # unreadable library file is not overwritten
            QMessageBox.critical(self, "3D printing", str(e))
            return
        self.win.log_msg(f"Material '{m.name}' saved to {matlib.user_library_path()}")
        self.refresh()
        self.mat.setCurrentText(m.name)

    def on_crit(self, _):
        if self._filling or self.study is None:
            return
        self.study.printing.criterion = self.crit.currentData()
        self.win.changed()

    def on_advanced(self, *_):
        """bins, samples, line_bins → study (a change = new shell assignment, results are not valid; the mesh stays)."""
        if self._filling or self.study is None:
            return
        ps = self.study.printing
        new = (self.a_bins.value(), self.a_samples.value(), self.a_line_bins.value())
        if new == (int(ps.bins), int(ps.samples), int(ps.line_bins)):
            return                                  # editingFinished without a change of value does not discard the results
        ps.bins, ps.samples, ps.line_bins = new
        self.win.changed()

    # ================================================================== kalibrace
    def open_calibration(self):
        if self.study is None:
            return
        dlg = CalibrationDialog(self, self.study, self.material_from_ui(), self.spec_from_ui(""))
        if dlg.exec() == QDialog.Accepted and dlg.result_material is not None:
            self.study.printing.tests = dlg.tests_dicts()
            m = dlg.result_material
            # the calibration assumed overwriting the print properties of the default material (line ratios) → the same overwrite
            # also for the resulting material, otherwise the calculation would use different ratios than the calibration
            pm = self.study.printing.materials
            base_name = dlg.base.currentText()
            if base_name in pm and m.name not in pm:
                pm[m.name] = dict(pm[base_name])
            names = self.selected_parts()
            if dlg.save_user.isChecked():
                try:
                    matlib.save_user_material(m)
                except matlib.UserLibraryError as e:
                    QMessageBox.critical(self, "3D printing", str(e))
            if names:
                self._assign(m, names)
            else:                                   # without selected parts only into the menu (and the library)
                self.study.materials[m.name] = m
                self.win.changed()
            self.mat.setCurrentText(m.name)
            self.win.log_msg(f"Calibrated material '{m.name}': E {fmt(m.E)} / {fmt(m.E_z)} MPa, Rm {fmt(m.Rm, 3)} / {fmt(m.Rm_z, 3)} MPa (XY / Z)")

    # ================================================================== display
    def show_shell(self):
        """Mesh section of the printed parts, colour = share of shell in the element."""
        win = self.win
        st = self.study
        if st is None:
            return
        mesh = win.mesh
        if mesh is None:
            QMessageBox.information(self, "3D printing", "Generate the mesh first (Mesh step).")
            return
        from fea_core.printing import has_print, shell_of
        from fea_core.printing.region import ensure
        if not has_print(st, mesh.part_names):
            QMessageBox.information(self, "3D printing", "No part has a printed material.")
            return
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            sig = mesh.report.get("print_sig")
            warns = ensure(st, mesh)             # recomputed only after a change of the print settings (signature), the mesh stays
            if mesh.report.get("print_sig") != sig:      # warning only after a new assignment, not at every move of the section
                for w in warns:
                    win.log_msg(f"Warning: {w}")
            sh = shell_of(mesh)
            from fea_core.mesh import Mesh
            ax = self.cut_axis.currentIndex()
            if self._cen_cache is None or self._cen_cache[0]() is not mesh:   # weak reference: it does not keep the old mesh
                import weakref
                self._cen_cache = (weakref.ref(mesh), mesh.nodes[mesh.tets[:, :4]].mean(axis=1))
            cen = self._cen_cache[1]
            printed = sh >= 0
            lo, hi = cen[printed, ax].min(), cen[printed, ax].max()
            pos = lo + (hi - lo) * self.cut_pos.value() / 1000.0
            keep = np.nonzero(printed & (cen[:, ax] <= pos + 1e-9))[0]
            if len(keep) == 0:
                return
            sub = Mesh(mesh.nodes, mesh.tets[keep], mesh.tet_part[keep], mesh.part_names)
            tri, el = sub.boundary_faces()
            # thin places with full infill (region.thin_core, f = 1) separately – otherwise they merge with the shell
            thin = mesh.cell_data.get("print_thin")
            thin_k = (thin[keep] > 0) if thin is not None else np.zeros(len(keep), bool)
            thin_f = thin_k[el]
            conn = mesh.tets[keep][~thin_k] if thin_k.any() and not thin_k.all() else mesh.tets[keep]
            vk = sh[keep][~thin_k] if thin_k.any() and not thin_k.all() else sh[keep]
            n = len(mesh.nodes)
            acc = np.bincount(conn.ravel(), np.repeat(vk.astype(float), conn.shape[1]), minlength=n)
            cnt = np.bincount(conn.ravel(), minlength=n)
            vals = np.where(cnt > 0, acc / np.maximum(cnt, 1), 1.0)
            edges = np.linspace(0.0, 1.0, 11)
            win.view.show_parts(True, 0.85)
            colors = win.view.show_contour(mesh.nodes, tri[~thin_f], vals, edges, "Shell fraction", edges_on=True,
                                           show_scale=False)
            n_thin = int((thin > 0).sum()) if thin is not None else 0
            title = "Shell fraction in the element [–]  (1 = shell, 0 = infill)"
            if thin_f.any():
                self._show_thin_faces(mesh.nodes, tri[thin_f])
                lab = {len(edges): "thin region – solid infill"}
                e2 = list(edges) + [1.1]
                self.legend.set_data(title, e2, list(colors) + [C_THIN],
                                     lambda v: lab.get(e2.index(v), fmt(float(v), 2)))
            else:
                self.legend.set_data(title, edges, colors, lambda v: fmt(float(v), 2))
            self.legend.set_note(f"{n_thin} elements in thin regions with solid infill (purple)" if n_thin else "")
            self.legend.show()
            if n_thin:
                win.log_msg(f"3D printing: {n_thin} elements in thin regions of printed parts with solid infill")
        finally:
            QApplication.restoreOverrideCursor()

    def _show_thin_faces(self, X, tri):
        """Walls of the elements of thin places (full infill) in one colour over the contour of the shell share."""
        from OCP.AIS import AIS_Shape
        from fea_core.contour import linear_triangles
        from .contour import triangles_face
        from .viewer3d import rgb
        v = self.win.view
        face = triangles_face(X[linear_triangles(tri)])
        if face is None:
            return
        o = AIS_Shape(face)
        o.Attributes().SetAutoTriangulation(False)
        o.SetColor(rgb(*C_THIN))
        o.SetMaterial(v._matte())
        v._no_highlight(o)
        asp = o.Attributes().ShadingAspect().Aspect()
        asp.SetDrawEdges(True)
        asp.SetEdgeColor(rgb(60, 60, 60))
        v.ctx.Display(o, 1, 0, False)
        v._result_objs.append(o)
        v._update()

    def update_info(self):
        res = self.win.results
        rows = []
        if res is not None and res.info.get("printing"):
            rows.append("<b>Printed parts (last analysis)</b>")
            for name, d in res.info["printing"]["parts"].items():
                thin = d.get("thin_fraction") or 0.0
                rows.append(f"{name}: shell fraction {fmt(100 * d['shell_fraction'], 3)} %, mass {fmt(d['mass_g'], 4)} g"
                            + (f", thin regions with solid infill {fmt(100 * thin, 3)} % of volume" if thin > 0 else ""))
            for s in res.steps:
                for name, d in (s.printing or {}).get("parts", {}).items():
                    rows.append(f"{s.name} – {name}: min. safety factor ({s.printing['criterion']}) <b>{fmt(d['min_safety'], 3)}</b> – {d['region']}")
            cr = res.info["printing"].get("creep")
            if cr:
                rows.append(f"<b>Creep (estimate, {fmt(cr['hours'])} h, {fmt(cr['temperature'], 3)} °C)</b>")
                for c in cr["cases"]:
                    fac = "; ".join(f"{k}: E × {fmt(v['stiffness'], 3)}, strength × {fmt(v['strength'], 3)}"
                                    for k, v in c["factors"].items())
                    rows.append(f"{c['case']}: max. displacement {fmt(c.get('u_short'))} → <b>{fmt(c['u_long'])} mm</b> (× {fmt(c.get('ratio'), 3)}), min. safety factor {fmt(c.get('min_safety_long'), 3)} <br>&nbsp;&nbsp;({fac})")
        if rows:
            rows.append("<span style='color:#b36b00'>! The linear model does not capture creep of the plastic (PLA) – account for creep under sustained load; the isochronous modulus estimate above is only indicative (linear viscoelasticity).</span>")
            self.info.setHtml("<br>".join(rows))
            self.info.setVisible(True)
        else:
            self.info.setHtml("")
            self.info.setVisible(False)


class CalibrationDialog(QDialog):
    """Table of tensile tests → calibrated printed material (fea_core.printing.calibrate)."""
    COLS = [("name", "Body"), ("direction", "Direction"), ("E", "E [MPa]"), ("Rm", "Rm [MPa]"), ("width", "b [mm]"),
            ("thickness", "h [mm]"), ("walls", "Perimeters"), ("line_width", "Line [mm]"), ("top_layers", "Top"),
            ("bottom_layers", "Bottom"), ("layer_height", "Layer [mm]"), ("infill", "Infill [%]"), ("pattern", "Pattern")]

    def __init__(self, parent, study, base: Material | None, spec: PrintSpec):
        super().__init__(parent)
        self.setWindowTitle("Printed material calibration – ISO 527")
        self.resize(1100, 640)
        self.study = study
        self.spec = spec
        self.result_material = None
        lay = QVBoxLayout(self)
        intro = QLabel(
            "ISO 527-2 tensile specimens (1A/1B: b = 10, h = 4 mm). Direction <b>xy</b> = specimen lies on the bed, <b>z</b> = stands upright (tension normal to the layers). Enter the specimen print settings – the solid extrusion is calculated with the same shell and infill model as in the analysis. Specimens with 100 % infill are best.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels([c[1] for c in self.COLS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setToolTip("Test specimens: measured E, Rm and print settings")
        self.table.setWhatsThis(
            "<b>Test specimens</b><p>Each row is one ISO 527 tensile specimen. Enter the measured modulus E and strength Rm (nominal, on the full cross-section b × h), the cross-section dimensions and the specimen print settings (perimeters, line width, top/bottom layers, layer height, infill, pattern).</p><p>Direction <i>xy</i> = specimen lay on the bed, <i>z</i> = stood upright (tension normal to the layers). The solid extrusion is calculated from each specimen with the shell and infill model; the result is the average over specimens of the same direction.</p>")
        for c, tip in enumerate((
                "Specimen name (empty = “specimen N”)",
                "Tension direction: xy = lying flat, z = standing upright",
                "Measured Young's modulus [MPa] (secant 0.05–0.25 %, ISO 527)",
                "Measured tensile strength [MPa] = max. force / (b × h)",
                "Width of the gauge cross-section b [mm] (ISO 527-2 1A/1B: 10 mm)",
                "Thickness of the gauge cross-section h [mm] (ISO 527-2 1A/1B: 4 mm)",
                "Number of perimeters when printing the specimen",
                "Extruded line width when printing the specimen [mm]",
                "Number of top solid layers of the specimen",
                "Number of bottom solid layers of the specimen",
                "Layer height when printing the specimen [mm]",
                "Specimen infill density [%] (most accurate calibration with 100 %)",
                "Specimen infill pattern")):
            self.table.horizontalHeaderItem(c).setToolTip(tip)
        lay.addWidget(self.table, 1)
        h = QHBoxLayout()
        for txt, slot, tip in (
                ("Add XY", lambda: self.add_row(TensileTest.from_spec("XY", "xy", self.spec)),
                 "Adds a specimen printed flat (tension in the layer plane), print settings from the form"),
                ("Add Z", lambda: self.add_row(TensileTest.from_spec("Z", "z", self.spec)),
                 "Adds a specimen printed upright (tension perpendicular to the layers), print settings from the form"),
                ("Remove", self.remove_row, "Removes the specimen row where the cursor is"),
                ("Load curve (CSV)…", self.load_curve,
                 "Loads a force – elongation curve (CSV/TXT) and fills in E and Rm of the selected row.\nAsks for the gauge length L0 of the extensometer (0 = the second column is strain in %).")):
            b = QPushButton(txt)
            short, _, rest = tip.partition("\n")
            help_tip(b, short, tip) if rest else b.setToolTip(tip)
            b.clicked.connect(slot)
            h.addWidget(b)
        h.addStretch(1)
        lay.addLayout(h)
        f = QFormLayout()
        self.base = QComboBox()
        mats = print_materials(study)
        if base is not None and base.name not in {m.name for m in mats}:
            mats.append(base)
        self._mats = {m.name: m for m in mats}
        self.base.addItems(list(self._mats))
        if base is not None:
            self.base.setCurrentText(base.name)
        help_tip(self.base, "Base material for missing Z/XY ratios, ν and strengths",
                 "Default material: Z/XY ratios (if Z tests are missing), ν, compressive and shear strengths")
        f.addRow("Default material", self.base)
        self.name = QLineEdit("PLA – calibration")
        self.name.setToolTip("Name of the calibrated material to be created (in the study, optionally in the library)")
        f.addRow("Result name", self.name)
        self.save_user = QCheckBox("also save to my material library")
        self.save_user.setChecked(True)
        help_tip(self.save_user, "Also saves the calibrated material to the user library",
                 "The calibrated material is also saved to the user material library – available in later studies")
        f.addRow(self.save_user)
        lay.addLayout(f)
        b = QPushButton("Calculate")
        help_tip(b, "Computes the full-extrusion properties, does not create the material",
                 "Calculates the solid-extrusion properties from the specimens in the table and lists them below (φ_E, φ_R per specimen,\nresulting E and Rm in XY / Z, coefficient of variation). Does not create a material yet.")
        b.clicked.connect(self.compute)
        lay.addWidget(b)
        self.out = SafeTextBrowser()
        self.out.setMinimumHeight(200)
        lay.addWidget(self.out)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Create material")
        help_tip(bb.button(QDialogButtonBox.Ok), "Computes the calibration and creates the material",
                 "Calculates the calibration and creates the material: adds it to the study and assigns it to the parts selected in the panel list\n(with no selection only to the printed materials list). The specimen table is saved in the study.")
        bb.button(QDialogButtonBox.Cancel).setToolTip("Closes the dialog without creating a material")
        bb.accepted.connect(self.accept_result)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        for d in study.printing.tests:
            self.add_row(TensileTest.from_dict(d))

    def add_row(self, t: TensileTest):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, (key, _) in enumerate(self.COLS):
            if key == "direction":
                w = QComboBox()
                w.addItems(["xy", "z"])
                w.setCurrentText(t.direction)
                help_tip(w, "xy = flat (tension in the layer), z = upright (perpendicular to the layers)",
                         "xy = specimen printed flat (tension in the layer plane), z = upright (tension normal to the layers)")
                self.table.setCellWidget(r, c, w)
            elif key == "pattern":                  # Czech name, the pattern key as the item data
                w = QComboBox()
                for k in PATTERNS:
                    w.addItem(pattern_label(k), k)
                w.setCurrentIndex(max(0, w.findData(t.pattern)))
                w.setToolTip("Specimen infill pattern (no effect at 100 % infill)")
                self.table.setCellWidget(r, c, w)
            else:
                v = getattr(t, key)
                if key == "infill":
                    v = 100 * v
                self.table.setItem(r, c, QTableWidgetItem(v if isinstance(v, str) else fmt(float(v), 6)))

    def remove_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    def row_test(self, r) -> TensileTest:
        d = {}
        for c, (key, _) in enumerate(self.COLS):
            w = self.table.cellWidget(r, c)
            if w is not None:
                d[key] = w.currentData() if key == "pattern" else w.currentText()
                continue
            txt = (self.table.item(r, c).text() if self.table.item(r, c) else "").strip()
            if key == "name":
                d[key] = txt or f"specimen {r + 1}"
            else:
                try:
                    v = float(txt.replace(",", ".")) if txt not in ("", "–") else 0.0
                except ValueError:
                    raise ValueError(f"Row {r + 1}, column {self.COLS[c][1]}: '{txt}' is not a number")
                d[key] = int(round(v)) if key in ("walls", "top_layers", "bottom_layers") else v
        d["infill"] = d["infill"] / 100.0
        return TensileTest.from_dict(d)

    def tests(self) -> list[TensileTest]:
        return [self.row_test(r) for r in range(self.table.rowCount())]

    def tests_dicts(self) -> list[dict]:
        return [t.to_dict() for t in self.tests()]

    def load_curve(self):
        r = self.table.currentRow()
        if r < 0:
            QMessageBox.information(self, "Calibration", "Select a specimen row.")
            return
        p, _ = QFileDialog.getOpenFileName(self, "Force–elongation curve", "", "CSV / TXT (*.csv *.txt);;All (*)")
        if not p:
            return
        L0, ok = QInputDialog.getDouble(self, "Calibration", "Gauge length of the extensometer L0 [mm]\n(0 = the second column is strain in %)", 50.0, 0.0, 1e4, 2)
        if not ok:
            return
        try:
            t = self.row_test(r)
            strain, stress = read_curve(p, t.width * t.thickness, L0 or None)
            E = iso527_modulus(strain, stress)
            Rm = float(stress.max())
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Calibration", f"The curve cannot be processed:\n{e}")
            return
        self.table.setItem(r, 2, QTableWidgetItem(fmt(E, 6)))
        self.table.setItem(r, 3, QTableWidgetItem(fmt(Rm, 6)))
        self.out.setHtml(f"Curve {p}: E (secant 0.05–0.25 %) = <b>{fmt(E)} MPa</b>, Rm = <b>{fmt(Rm)} MPa</b>")

    def compute(self) -> bool:
        try:
            from fea_core.printing.props import props_of
            base = self._mats[self.base.currentText()]
            # line ratios across / along from the study (override "Use for material"), otherwise the default by family
            cal = calibrate(base, self.tests(), self.name.text().strip() or None,
                            props=props_of(self.study, base))
        except Exception as e:  # noqa: BLE001
            self.out.setHtml(f"<span style='color:#d23b3b'>{escape(str(e))}</span>")
            self.result_material = None
            return False
        m, s = cal.material, cal.summary
        table = ("<table cellspacing=6><tr><th align=left>Specimen</th><th>direction</th><th>E meas. [MPa]</th><th>Rm meas. [MPa]</th><th>φ_E</th><th>φ_R</th><th>E solid [MPa]</th><th>R solid [MPa]</th></tr>")
        for r in cal.rows:
            table += (f"<tr><td>{escape(str(r['name']))}</td><td>{escape(str(r['direction']))}</td><td>{fmt(r['E_meas'])}</td>"
                      f"<td>{fmt(r['Rm_meas'])}</td><td>{fmt(r['phi_E'], 3)}</td><td>{fmt(r['phi_R'], 3)}</td>"
                      f"<td>{fmt(r['E_solid'])}</td><td>{fmt(r['R_solid'])}</td></tr>")
        rows = [table + "</table>"]

        def cv(k):
            return f" (CoV {fmt(100 * s[k], 2)} %)" if s.get(k) is not None else ""
        rows.append(f"<b>Result:</b> E_xy {fmt(m.E)} MPa{cv('cv_E_xy')}, E_z {fmt(m.E_z)} MPa{cv('cv_E_z')}, Rm_xy {fmt(m.Rm)} MPa{cv('cv_R_xy')}, Rm_z {fmt(m.Rm_z)} MPa{cv('cv_R_z')}; strength Z / XY = {fmt(100 * s['z_ratio_R'], 3)} %")
        for w in cal.warnings:
            rows.append(f"<span style='color:#b36b00'>! {w}</span>")
        self.out.setHtml("<br>".join(rows))
        self.result_material = m
        return True

    def accept_result(self):
        if self.compute():
            self.accept()
