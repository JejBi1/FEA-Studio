"""Contacts panel (step after Joints, F8 – spec 4.1–4.9, 6.3): contact pairs of the assembly, finding contacts, friction
library, settings of the formulation and meshing of the interface, nonlinear stepping (branches / sequence of steps with prescribed motion).

The GUI only edits Study.contacts, Study.contact_settings and Study.analysis; the calculation is done by fea_core.contacts (it runs
with the whole study in the Solve step, the readiness check via solve.check_study). A pair between parts ("A", "B" = part names)
finds the contact faces on the mesh itself (walls within the tolerance of the other part); a pair between face selections is more precise and
faster (only the faces needed).
"""
from __future__ import annotations

from html import escape
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from fea_core.contacts.model import (CONTACT_LABELS, CONTACT_TYPES, ContactPair, LoadStep, Motion)

from .widgets import (Section, dspin, fit_rows, fmt, help_tip, panel_header,
                      scroll_body, scroll_to, set_placeholder, table)

APP = "FEA Studio"
# colours of the contact types in 3D and in the table (RGB 0–255)
TYPE_COLORS = {"bonded": (150, 150, 150), "no_separation": (20, 160, 160), "frictionless": (40, 110, 230),
               "frictional": (230, 130, 20), "rough": (150, 90, 40), "interference": (200, 40, 160),
               "compression_only": (40, 160, 60), "none": (200, 200, 200)}
# descriptions of the types (menu, "What is this?") – behaviour and realisation in CalculiX (spec 4.1)
TYPE_TIPS = {
    "bonded": "Bonded: parts share the mesh (as without contact) – linear, superposition holds.",
    "no_separation": "No separation: surfaces slide on each other without friction but do not separate (ccx TIED + small λ).",
    "frictionless": "Frictionless contact: surfaces can separate, in compression only a normal force is transferred (nonlinear).",
    "frictional": "Frictional contact: Coulomb μ, elastic slip λ while sticking; can slide (nonlinear).",
    "rough": "Rough contact: large μ – surfaces can separate but do not slide (nonlinear).",
    "interference": "Interference (press fit): radial interference δ_r = Δd/2 as a negative *CLEARANCE on the nominal\ngeometry, or the actual overlap in CAD (no value).",
    "compression_only": "Compression only (support): transfers compression only – with an empty side B as a support to ground\n(GAPUNI), otherwise contact of two parts (nonlinear).",
    "none": "No interaction: the parts do not influence each other (may penetrate).",
}
STATE_LABELS = {"contact": "contact", "gap": "gap", "penetration": "penetration", None: "–"}
PURPOSES = [("slip", "joint slip (μ_min)"), ("resistance", "mechanism resistance (μ_max)")]
METHODS = [("surface_to_surface", "surface – surface (penalty, default)"), ("mortar", "mortar (hard contact)"),
           ("node_to_surface", "node – surface (friction convergence only)")]
SPLITS = [("separate", "mesh parts separately (non-conforming mesh)"),
          ("conformal_split", "conforming mesh + split interface (bolts)")]
BOLTS = [("lock", "locked (length after preload)"), ("preload", "preload in this step"), ("free", "free")]
DOF_NAMES = ["x", "y", "z"]


def _item(text, align_right=False, tip=None) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    if align_right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if tip:
        it.setToolTip(tip)
    return it


def _num_or_none(v: float, zero_none=True):
    return None if (zero_none and abs(v) < 1e-300) else float(v)


def describe_pair(c: ContactPair) -> str:
    b = c.b or "ground"
    return f"{c.name}: {c.a} – {b}, {CONTACT_LABELS.get(c.kind, c.kind)}"


class EditTable(QTableWidget):
    """Small editable table of numbers (amplitude course, friction changes): rows [x, y]; an empty cell = None."""

    def __init__(self, cols, tip, rows=0, detailed=""):
        super().__init__(rows, len(cols))
        self.setHorizontalHeaderLabels(cols)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().hide()
        self.setMinimumHeight(110)
        self.setMaximumHeight(170)
        if not detailed and (len(tip) > 80 or "\n" in tip):
            from .guide import short_of
            help_tip(self, short_of(tip, 80), tip)
        else:
            help_tip(self, tip, detailed)

    def set_rows(self, rows):
        self.setRowCount(0)
        for r in rows:
            k = self.rowCount()
            self.insertRow(k)
            for c, v in enumerate(r):
                self.setItem(k, c, QTableWidgetItem("" if v is None else (v if isinstance(v, str) else fmt(float(v), 8))))

    def values(self, col_types=None) -> list[list]:
        out = []
        for r in range(self.rowCount()):
            row = []
            for c in range(self.columnCount()):
                it = self.item(r, c)
                t = (it.text() if it else "").strip().replace(",", ".").replace(" ", "")
                if col_types and col_types[c] is str:
                    row.append((it.text() if it else "").strip())
                    continue
                row.append(float(t) if t else None)
            if any(v not in (None, "") for v in row):
                out.append(row)
        return out

    def add_row(self):
        self.insertRow(self.rowCount())

    def remove_row(self):
        r = self.currentRow()
        self.removeRow(r if r >= 0 else self.rowCount() - 1)


class ContactPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._filling = False
        self._found: list[dict] = []            # result of "Find contacts" (not only the one stored in the study)
        self._det_faces: dict = {}              # (a, b) → (faces a, faces b) from the detection – display in 3D
        self._thread = None
        self._row = -1                          # selected pair (index in Study.contacts)
        self._loaded = None                     # (row, pair) last loaded into the form
        self._last_study = None
        self._last_branches = None
        self._scroll, lay = scroll_body(self)
        lay.addWidget(panel_header("Contacts", "Contact of assembly parts (frictionless, frictional, interference, compression only), friction and nonlinear stepping – optional"))
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setTextFormat(Qt.RichText)
        help_tip(self.banner, "Study state: linear (superposition) or non-linear",
                 "Study state: linear (superposition, parts tied with shared mesh) or non-linear (contact with separation / friction, large deflections, step sequence).")
        lay.addWidget(self.banner)
        intro = QLabel("Without contact pairs, touching parts are <b>rigidly connected</b> by a shared mesh. Contact adds real contact (separation, friction, interference) – the analysis is then nonlinear and slower.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        # ---------------------------------------------------------- finding contacts
        self.sec_find = Section("Find touching parts", key="contact.find", collapsed=False,
                                tip="Finding touching parts of the assembly in CAD geometry",
                                help="<b>Find touching parts</b><p>Searches pairs of parts within the given distance: <i>contact</i> (faces on each other), <i>gap</i> (clearance within tolerance), <i>penetration</i> (the parts overlap – press fit or a CAD error). The pairs found are shown in the table below the button; add the selected ones as contact pairs of the chosen type.</p><p>A pair you do not add stays rigidly connected by the shared mesh.</p>")
        f = QFormLayout(self.sec_find.body)
        self.tol = dspin(0.05, 0.0, 100.0, 3, 0.05, " mm")
        help_tip(self.tol, "Search distance for touching part pairs [mm]",
                 "Search distance [mm]: part pairs closer than this value are offered as a pair (similar to Pinball / Max Activation Distance). Larger values also find clearances.")
        f.addRow("Search distance", self.tol)
        self.btn_find = QPushButton("Find touching parts")
        help_tip(self.btn_find, "Find touching parts in the model (OCP geometry)",
                 "<b>Find touching parts</b><p>Searches pairs of parts within the given distance: <i>contact</i> (faces on each other), <i>gap</i> (clearance within tolerance), <i>penetration</i> (the parts overlap – press fit or a CAD error). The pairs found are shown in the table below the button; add the selected ones as contact pairs of the chosen type.</p><p>A pair you do not add stays rigidly connected by the shared mesh.</p>")
        self.btn_find.clicked.connect(self.find_touching)
        f.addRow(self.btn_find)
        self.found = table(["Part A", "Part B", "State", "Gap / penetration [mm]", "Area [mm²]", "Pair"], 110,
                           placeholder="Run \"Find contacts\" to locate contact faces", rows=(2, 6))
        self.found.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.found.setMaximumHeight(160)
        help_tip(self.found, "Detected part contacts (touch, gap, or interference)",
                 "Found part pairs: status (touch / gap / interference), smallest gap or interference depth, contact area, Pair column = already in the pair list. Ctrl / Shift = multi-select.")
        f.addRow(self.found)
        h = QHBoxLayout()
        self.add_kind = QComboBox()
        for k in CONTACT_TYPES:
            self.add_kind.addItem(CONTACT_LABELS[k], k)
            self.add_kind.setItemData(self.add_kind.count() - 1, TYPE_TIPS[k], Qt.ToolTipRole)
        self.add_kind.setCurrentIndex(self.add_kind.findData("frictionless"))
        self.add_kind.setToolTip("Contact type for the pairs being added (can be changed later for each pair)")
        h.addWidget(self.add_kind, 1)
        b = QPushButton("Add selected")
        help_tip(b, "Add selected pairs as contact pairs",
                 "Add selected pairs as contact pairs of the chosen type (no selection = all not yet in the list).")
        b.clicked.connect(self.add_found)
        self.btn_add_found = b
        h.addWidget(b)
        f.addRow(h)
        lay.addWidget(self.sec_find)

        # ---------------------------------------------------------- pairs
        self.sec_pairs = Section("Contact pairs", key="contact.pairs", collapsed=False,
                                 tip="Model contact pairs and editing of the selected pair",
                                 help="<b>Contact pairs</b><p>Assembly contact pairs: sides A and B (part or face selection), contact type, friction, stiffness and initial status. Clicking a row selects the pair in 3D and loads it into the form below.</p>")
        v_pairs = QVBoxLayout(self.sec_pairs.body)
        v_pairs.setContentsMargins(0, 0, 0, 0)
        self.table = table(["Pair", "A", "B", "Type", "μ", "Purpose", "K [N/mm³]", "λ [N/mm³]", "Interference / clearance [mm]",
                            "Adjust", "Initial", "Note"], 150,
                           placeholder="No contact pairs yet – add from detected contacts or face selection", rows=(3, 10))
        help_tip(self.table, "Study contact pairs (click = select faces in 3D)",
                 "Study contact pairs: sides A and B (part or face selection), type (color = color in 3D), μ (blank = from library by purpose), normal stiffness K and cohesive slope λ (blank = automatic), interference / clearance, adjust to contact and initial status from \"Find contacts\". Click = pair faces in 3D and values in form.")
        self.table.currentCellChanged.connect(lambda r, *_: self.on_row(r))
        v_pairs.addWidget(self.table)

        g = QGroupBox("Pair")
        help_tip(g, "Edit selected pair or define a new one",
                 "Edit selected pair or define a new one (sides from 3D face selection or whole parts).")
        f = QFormLayout(g)
        self.p_name = QLineEdit()
        self.p_name.setToolTip("Pair name (reactions, results and the report use this name)")
        f.addRow("Name", self.p_name)
        self.p_a, self.p_b = QComboBox(), QComboBox()
        for side, cb, tip in (("A", self.p_a, "Side A: part (all its faces touching B) or a face selection"),
                              ("B", self.p_b, "Side B: part or face selection; for the “Compression only” type empty = support to ground")):
            cb.setEditable(True)
            cb.setProperty("_i18n_done", True)          # names of parts / selections (data) are not translated
            cb.setToolTip(tip)
            hb = QHBoxLayout()
            hb.addWidget(cb, 1)
            bt = QPushButton(f"{side} from selection")
            bt.setToolTip(f"Side {side} = faces selected in 3D (a face selection “Contact {side} n” is created)")
            bt.clicked.connect(lambda _=False, s=side: self.side_from_selection(s))
            setattr(self, f"btn_side_{side.lower()}", bt)
            hb.addWidget(bt)
            f.addRow(f"Side {side}", hb)
        self.p_kind = QComboBox()
        for k in CONTACT_TYPES:
            self.p_kind.addItem(CONTACT_LABELS[k], k)
            self.p_kind.setItemData(self.p_kind.count() - 1, TYPE_TIPS[k], Qt.ToolTipRole)
        help_tip(self.p_kind, "Contact type: interface behavior in normal and tangential directions",
                 "<b>Contact type</b><ul>" + "".join(
                     f"<li><b>{CONTACT_LABELS[k]}</b> – {TYPE_TIPS[k]}</li>" for k in CONTACT_TYPES) +
                 "</ul><p>All types except “Bonded” and “No interaction” are nonlinear: states are computed one after another, superposition and fatigue by superposition do not apply. See the help <i>Contacts</i>.</p>")
        self.p_kind.currentIndexChanged.connect(self.on_kind)
        f.addRow("Type", self.p_kind)
        self.p_mu = dspin(0.0, 0.0, 10.0, 4, 0.05, special="from library")
        help_tip(self.p_mu, "Friction coefficient μ (0 = from library by purpose)",
                 "Friction coefficient μ (0 = from friction library by material pair and purpose). μ is an ESTIMATE – computing both bounds (μ_min, μ_max) as variants is recommended.")
        f.addRow("Friction μ", self.p_mu)
        self.p_purpose = QComboBox()
        for k, t in PURPOSES:
            self.p_purpose.addItem(t, k)
        help_tip(self.p_purpose, "Friction purpose: force transmission (μ_min) or mechanism resistance (μ_max)",
                 "Friction purpose: friction load transfer (bolted joint, press fit, brake) → μ_min (conservative), mechanism resistance (drive force, assembly, jamming) → μ_max.")
        f.addRow("Purpose", self.p_purpose)
        self.p_lib = QComboBox()
        help_tip(self.p_lib, "Material pair in the friction library",
                 "Material pair in the friction library (sources in Friction help). Automatic = by part materials; if missing, 0.15 (slip) / 0.3 (resistance) is used – measure if critical.")
        f.addRow("Friction library", self.p_lib)
        self.mu_info = QLabel("")
        self.mu_info.setWordWrap(True)
        self.mu_info.setTextFormat(Qt.RichText)
        f.addRow(self.mu_info)
        self.p_K = dspin(0.0, 0.0, 1e12, 1, 1e5, " N/mm³", special="automatic (50·E)")
        help_tip(self.p_K, "Normal contact stiffness K [N/mm³] (penalty)",
                 "Normal contact stiffness K [N/mm³] (penalty): 0 = 50 × smaller E of the pair (ccx manual 5–50·E). Penetration = p/K – for interference choose large K or mortar formulation.")
        f.addRow("Stiffness K", self.p_K)
        self.p_lam = dspin(0.0, 0.0, 1e12, 1, 1e3, " N/mm³", special="automatic (K/100)")
        self.p_lam.setToolTip("Stick slope λ [N/mm³] (tangential stiffness while sticking, elastic slip = τ/λ): 0 = K/100")
        f.addRow("Stick slope λ", self.p_lam)
        hc = QHBoxLayout()
        self.p_clr_on = QCheckBox("specify")
        help_tip(self.p_clr_on, "Specify interference or clearance for the pair as a single value",
                 "Checked = pair interference / clearance as a single value (*CLEARANCE, overrides CAD gap); unchecked = gap / overlap from actual CAD geometry.")
        self.p_clr = dspin(0.0, -100.0, 100.0, 5, 0.005, " mm")
        help_tip(self.p_clr, "Radial interference (negative value) or clearance [mm]",
                 "Radial interference / clearance [mm]: negative = interference (press fit δ_r = −Δd/2), positive = clearance. Only for nominal geometry and surface-to-surface / mortar formulation.")
        self.p_clr_on.toggled.connect(self.p_clr.setEnabled)
        hc.addWidget(self.p_clr_on)
        hc.addWidget(self.p_clr, 1)
        f.addRow("Interference / clearance", hc)
        self.p_adjust = QCheckBox("snap nodes into contact")
        help_tip(self.p_adjust, "Move the nodes of the slave face onto the master into contact",
                 "Move slave surface nodes to master (eliminates small gaps and penetrations from CAD). Modifies mesh coordinates; never use with interference (interference would vanish).")
        f.addRow(self.p_adjust)
        self.p_slave = QComboBox()
        for k, t in ((None, "automatic (finer mesh)"), ("a", "side A"), ("b", "side B")):
            self.p_slave.addItem(t, k)
        help_tip(self.p_slave, "Slave surface selection (nodes must not penetrate master)",
                 "Slave surface (nodes that must not penetrate master): automatic = finer mesh, if matched then softer material.")
        f.addRow("Slave", self.p_slave)
        from .thermal_extras import conductance_row
        self.p_hc, row = conductance_row(self, lambda: self.pair_sides())
        f.addRow("Thermal conductance", row)
        self.p_enabled = QCheckBox("pair is active")
        self.p_enabled.setChecked(True)
        self.p_enabled.setToolTip("Unchecked = the pair is not passed to the analysis (parts then bonded / without contact)")
        f.addRow(self.p_enabled)
        hb = QHBoxLayout()
        for txt, slot, tip, attr in (
                ("Add new pair", self.add_pair, "New pair from sides A, B and the form settings", "btn_add_pair"),
                ("Save changes", self.update_pair, "Overwrite the selected pair with the form values", "btn_update_pair"),
                ("Delete", self.delete_pair, "Delete the selected pair (also the face selections nothing else uses)",
                 "btn_delete_pair")):
            bt = QPushButton(txt)
            bt.setToolTip(tip)
            bt.clicked.connect(slot)
            setattr(self, attr, bt)
            hb.addWidget(bt)
        f.addRow(hb)
        v_pairs.addWidget(g)
        lay.addWidget(self.sec_pairs)
        self.p_lib.currentIndexChanged.connect(lambda _: self.update_mu_info())
        self.p_purpose.currentIndexChanged.connect(lambda _: self.update_mu_info())
        self.p_mu.valueChanged.connect(lambda _: self.update_mu_info())

        # ---------------------------------------------------------- contact settings
        self.sec_settings = Section("Contact settings", key="contact.settings", collapsed=True,
                                    tip="Global settings for contact formulation and interface meshing",
                                    help="<b>Contact settings</b><p>Contact formulation in CalculiX, joint interface meshing, large deflections and stabilization for the entire study.</p>")
        f = QFormLayout(self.sec_settings.body)
        self.s_method = QComboBox()
        for k, t in METHODS:
            self.s_method.addItem(t, k)
        help_tip(self.s_method, "Contact formulation: surface-to-surface, mortar or node-to-surface",
                 "Formulation (one per study): surface-to-surface = penalty, pressure ±0.1% (default); mortar = hard contact, suited for interference (~2.5× slower); node-to-surface only for friction convergence (pressure unusable on quadratic elements).")
        f.addRow("Formulation", self.s_method)
        self.s_split = QComboBox()
        for k, t in SPLITS:
            self.s_split.addItem(t, k)
        help_tip(self.s_split, "Meshing of contact faces of parts in contact",
                 "Mesh of parts in contact: separate (non-conforming, required for interference) or conforming mesh with split interface (matching nodes, accurate clamping – bolted joints).")
        f.addRow("Interface meshing", self.s_split)
        self.s_tol = dspin(0.0, 0.0, 100.0, 3, 0.05, " mm", special="automatic (0.2 × element)")
        help_tip(self.s_tol, "Tolerance for finding contact faces on mesh [mm]",
                 "Tolerance for detecting contact faces on mesh [mm] (part faces within this distance to other part); 0 = automatic 0.2 × element size.")
        f.addRow("Face tolerance", self.s_tol)
        hs = QHBoxLayout()
        self.s_stab = QComboBox()
        self.s_stab.addItem("no stabilization", "none")
        self.s_stab.addItem("weak springs", "springs")
        help_tip(self.s_stab, "Stabilization of parts held only by contact using ground springs",
                 "Stabilization of parts held only by contact (otherwise model is initially singular): weak ground springs at all nodes of the part.")
        self.s_stab_k = dspin(0.0, 0.0, 1e9, 6, 1e-3, " N/mm", special="automatic")
        help_tip(self.s_stab_k, "Stiffness of stabilization springs [N/mm] (0 = automatic)",
                 "Stiffness of stabilization springs [N/mm] (0 = automatic ~1e-6·EA/L) – must be negligible compared to part stiffness.")
        hs.addWidget(self.s_stab, 1)
        hs.addWidget(self.s_stab_k, 1)
        f.addRow("Stabilization", hs)
        self.s_nlgeom = QCheckBox("large deformations (NLGEOM)")
        help_tip(self.s_nlgeom, "Geometric non-linearity (large deflections and rotations)",
                 "Geometric non-linearity: snap-through, cam, large rotation. Incompatible with bolts, cylindrical support, and distributing remote coupling (linear equations from undeformed geometry).")
        f.addRow(self.s_nlgeom)
        self.s_press = QComboBox()
        self.s_press.addItem("dead (direction does not change)", "dead")
        self.s_press.addItem("follower (normal to the deformed face)", "follower")
        help_tip(self.s_press, "Pressure direction under large deflections (dead / follower)",
                 "Pressure under large deflections: dead = original direction, follower = normal to deformed surface (NLGEOM only).")
        f.addRow("Pressure", self.s_press)
        self.s_bolt = QCheckBox("bolted joint interfaces as contact")
        help_tip(self.s_bolt, "Joint interface of clamped bolt parts as friction contact",
                 "Clamped joint interface of each bolt as friction contact (μ = bolt μ_T, Joints step): joint opening, residual clamping force, slip. Preload in step 1, then fixed; interface mesh conforming with split.")
        f.addRow(self.s_bolt)
        for w in (self.s_method, self.s_split, self.s_stab, self.s_press):
            w.currentIndexChanged.connect(lambda _: self.on_settings())
        self.s_nlgeom.toggled.connect(lambda _: self.on_settings())
        self.s_bolt.toggled.connect(lambda _: self.on_settings())
        for w in (self.s_tol, self.s_stab_k):
            self._commit_on_edit(w, self.on_settings)
        lay.addWidget(self.sec_settings)

        # ---------------------------------------------------------- nonlinear stepping
        self.sec_analysis = Section("Nonlinear analysis – steps", key="contact.analysis", collapsed=True,
                                    tip="Solution control of non-linear study and step sequence",
                                    help="<b>Non-linear analysis – steps</b><p>How load cases of a non-linear study are solved: each separately (branches) or as history (step sequence).</p>")
        f = QFormLayout(self.sec_analysis.body)
        self.a_kind = QComboBox()
        self.a_kind.addItem("linear (nonlinear only with contact / NLGEOM)", "linear")
        self.a_kind.addItem("always nonlinear", "nonlinear")
        help_tip(self.a_kind, "Study calculation type: linear superposition or non-linear",
                 "Calculation type: linear = superposition (contact pair or NLGEOM switches to non-linear automatically); always non-linear = even without contact (e.g. step sequence with prescribed motion).")
        f.addRow("Calculation type", self.a_kind)
        self.a_mode = QComboBox()
        self.a_mode.addItem("cases separately (branches, in parallel)", True)
        self.a_mode.addItem("step sequence (history)", False)
        help_tip(self.a_mode, "Case ordering: separate branches or step sequence",
                 "Branches: each load case solved independently after [bolt preload] – concurrently, path-independent.<br>Sequence: single job, consecutive steps (state preserved – friction is path-dependent), prescribed motion, friction change, pair deactivation, cycles (frictional fatigue ≥ 2 cycles).")
        f.addRow("States", self.a_mode)
        self.a_keep = QCheckBox("keep the last converged increment if the analysis does not converge")
        help_tip(self.a_keep, "Keep last completed increment on non-convergence",
                 "If the solution fails to converge, the result is the last converged increment marked as PARTIAL (red badge in results).")
        f.addRow(self.a_keep)
        self.a_kind.currentIndexChanged.connect(lambda _: self.on_analysis())
        self.a_mode.currentIndexChanged.connect(lambda _: self.on_analysis())
        self.a_keep.toggled.connect(lambda _: self.on_analysis())
        self.seq = table(["#", "Step", "State", "Scale", "Bolts", "Motions", "Cycles"], 120,
                         placeholder="No steps – add a step with the button below", rows=(2, 6))
        help_tip(self.seq, "Load step sequence (click = edit step below)",
                 "Load history step sequence (order = solve order). Loads between steps do not accumulate – each step specifies its load (case × scale), contact state and deformations persist. Click = edit step below.")
        self.seq.currentCellChanged.connect(lambda r, *_: self.on_seq_row(r))
        f.addRow(self.seq)
        hb = QHBoxLayout()
        for txt, slot, tip, attr in (
                ("Add step", self.add_step, "New step from the step form (at the end of the sequence)", "btn_add_step"),
                ("Save step", self.update_step, "Overwrite the selected step with the form values", "btn_update_step"),
                ("Delete", self.delete_step, "Delete the selected step", "btn_delete_step"),
                ("↑", lambda: self.move_step(-1), "Move the step earlier", "btn_step_up"),
                ("↓", lambda: self.move_step(1), "Move the step later", "btn_step_down")):
            bt = QPushButton(txt)
            bt.setToolTip(tip)
            bt.clicked.connect(slot)
            setattr(self, attr, bt)
            hb.addWidget(bt)
        f.addRow(hb)
        bt = QPushButton("Steps from load cases")
        help_tip(bt, "Create step sequence from load cases",
                 "Replace sequence with \"case by case\" steps in load cases order (scale 1).")
        bt.clicked.connect(self.steps_from_cases)
        self.btn_steps_cases = bt
        f.addRow(bt)
        lay.addWidget(self.sec_analysis)

        # ---------------------------------------------------------- krok
        self.sec_step = Section("Step", key="contact.step", collapsed=True,
                                tip="Settings of selected or new sequence step",
                                help="<b>Sequence step</b><p>Step settings: load case, scale factor, ramp, bolts, deactivated pairs and increment parameters.</p>")
        f = QFormLayout(self.sec_step.body)
        self.k_name = QLineEdit()
        self.k_name.setToolTip("Step name (results are named after the step)")
        f.addRow("Name", self.k_name)
        self.k_case = QComboBox()
        help_tip(self.k_case, "Step load case (load × scale)",
                 "Step load case (load × scale); \"none\" = prescribed motion or contact change only.")
        f.addRow("Load case", self.k_case)
        self.k_scale = dspin(1.0, -1e6, 1e6, 4, 0.1)
        self.k_scale.setToolTip("Load multiplier of the case in this step (e.g. 0 = unloading, 2 = overload)")
        f.addRow("Scale", self.k_scale)
        self.k_ramp = QComboBox()
        self.k_ramp.addItem("linear ramp", "linear")
        self.k_ramp.addItem("step", "step")
        self.k_ramp.setToolTip("Load history within the step: linear ramp (better convergence) or step")
        f.addRow("Ramp", self.k_ramp)
        self.k_bolts = QComboBox()
        for k, t in BOLTS:
            self.k_bolts.addItem(t, k)
        help_tip(self.k_bolts, "Bolt behavior in step: preload, locked, or free",
                 "Bolts in step: preload (preload force), locked (fixed length after preload, *BOUNDARY FIXED), or free.")
        f.addRow("Bolts", self.k_bolts)
        self.k_off = QListWidget()
        self.k_off.setMaximumHeight(80)
        self.k_off.setToolTip("Checked pairs are deactivated in this step (*MODEL CHANGE)")
        f.addRow("Deactivated pairs", self.k_off)
        self.k_mu = EditTable(["Pair", "new μ"], "Friction change in step (*CHANGE FRICTION)",
                              detailed="Friction change in step (*CHANGE FRICTION): enter new μ > 0 for pair, empty = unchanged.<br>Only works for pairs with friction from step 1 ('frictionless' = μ 1e-4).")
        self.k_mu.setMinimumHeight(80)
        fit_rows(self.k_mu, 2, 5)
        set_placeholder(self.k_mu, "No friction pairs yet")
        f.addRow("Friction change", self.k_mu)
        hp = QHBoxLayout()
        self.k_dt0 = dspin(0.1, 1e-9, 1.0, 6, 0.05)
        self.k_dt0.setToolTip("First increment as a fraction of the step (0.1 = 10 %); reduce if it does not converge")
        self.k_dtmin = dspin(1e-5, 1e-12, 1.0, 8, 1e-5)
        self.k_dtmin.setToolTip("Smallest allowed increment (fraction of the step) – below it the analysis ends without convergence")
        self.k_dtmax = dspin(0.25, 1e-9, 1.0, 6, 0.05)
        self.k_dtmax.setToolTip("Largest increment (fraction of the step)")
        for lbl, w in (("dt₀", self.k_dt0), ("min", self.k_dtmin), ("max", self.k_dtmax)):
            hp.addWidget(QLabel(lbl))
            hp.addWidget(w, 1)
        f.addRow("Increments", hp)
        hq = QHBoxLayout()
        self.k_maxinc = QSpinBox()
        self.k_maxinc.setRange(1, 100000)
        self.k_maxinc.setValue(200)
        self.k_maxinc.setToolTip("Maximum number of increments of the step (ccx *STEP, INC=)")
        self.k_every = QSpinBox()
        self.k_every.setRange(1, 10000)
        self.k_every.setValue(1)
        self.k_every.setToolTip("Output (frame, point of the force–travel curve) every n increments")
        self.k_cycles = QSpinBox()
        self.k_cycles.setRange(1, 1000)
        self.k_cycles.setValue(1)
        help_tip(self.k_cycles, "Number of step block repetitions (≥ 2 for frictional fatigue)",
                 "Number of repetitions of step block (≥ 2 for frictional fatigue – initial cycle differs due to micro-slip).")
        for lbl, w in (("max. increments", self.k_maxinc), ("output every", self.k_every), ("cycles", self.k_cycles)):
            hq.addWidget(QLabel(lbl))
            hq.addWidget(w, 1)
        f.addRow(hq)
        self.k_direct = QCheckBox("fixed increments (DIRECT)")
        self.k_direct.setToolTip("Fixed increments dt₀ without automatic cutback (*STATIC, DIRECT)")
        f.addRow(self.k_direct)
        lay.addWidget(self.sec_step)

        # ---------------------------------------------------------- prescribed motion
        self.sec_motion = Section("Prescribed motion in the step", key="contact.motion", collapsed=True,
                                  tip="Displacement or rotation of rigid remote support in step",
                                  help="<b>Prescribed motion</b><p>Displacement / rotation of rigid remote support in step (displacement control – superior friction convergence, force-displacement curve).</p>")
        f = QFormLayout(self.sec_motion.body)
        self.mo_list = QListWidget()
        self.mo_list.setMaximumHeight(80)
        set_placeholder(self.mo_list, "No prescribed motion yet")
        self.mo_list.setToolTip("Prescribed motions of the selected step. Click = load into the form.")
        self.mo_list.currentRowChanged.connect(self.on_motion_row)
        f.addRow(self.mo_list)
        self.mo_target = QComboBox()
        self.mo_target.setToolTip("Motion target = face selection with a rigid remote support (Supports step, coupling “rigid”)")
        f.addRow("Rigid support", self.mo_target)
        hm = QHBoxLayout()
        self.mo_kind = QComboBox()
        self.mo_kind.addItem("displacement [mm]", "disp")
        self.mo_kind.addItem("rotation [rad]", "rot")
        self.mo_kind.setToolTip("Type of motion: displacement of the support point [mm] or rotation about the axis [rad]")
        self.mo_dof = QComboBox()
        self.mo_dof.addItems(DOF_NAMES)
        self.mo_dof.setToolTip("Axis of motion (global x, y, z)")
        hm.addWidget(self.mo_kind, 1)
        hm.addWidget(self.mo_dof)
        f.addRow("Motion", hm)
        self.mo_amp = EditTable(["t (0–1)", "value"], "Motion history in step time: pairs [t, value]",
                                detailed="Motion history in step time (0 = start, 1 = end): pairs [t, value] linearly between points.")
        self.mo_amp.set_rows([[0.0, 0.0], [1.0, 0.0]])
        fit_rows(self.mo_amp, 2, 5)
        f.addRow(self.mo_amp)
        hr = QHBoxLayout()
        for txt, slot, tip, attr in (("+ row", self.mo_amp.add_row, "Add history row", "btn_mo_row_add"),
                                     ("− row", self.mo_amp.remove_row, "Delete the selected history row", "btn_mo_row_del"),
                                     ("Add motion", self.add_motion, "Add the motion to the selected step (saved with the step)", "btn_add_motion"),
                                     ("Delete motion", self.delete_motion, "Delete the selected motion", "btn_del_motion")):
            bt = QPushButton(txt)
            bt.setToolTip(tip)
            bt.clicked.connect(slot)
            setattr(self, attr, bt)
            hr.addWidget(bt)
        f.addRow(hr)
        lay.addWidget(self.sec_motion)
        self._motions: list[Motion] = []        # motions of the unfinished step (stored by "Save step" / "Add step")

        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        lay.addWidget(self.info)
        lay.addStretch(1)
        self.win.view.facesPicked.connect(self.on_faces_picked)

    def scroll_to(self, w):
        """Scrolls the panel content so that the element w is visible (guide, test snapshots)."""
        scroll_to(self._scroll, w)

    @staticmethod
    def _commit_on_edit(spin, slot):
        spin.editingFinished.connect(slot)
        spin.valueChanged.connect(lambda *_: None if spin.hasFocus() else slot())

    # ================================================================== state ↔ controls
    @property
    def study(self):
        return self.win.study

    def part_names(self) -> list[str]:
        excl = self.study.excluded_parts() if self.study is not None else set()
        return [p.name for p in self.win.solid_parts() if p.name not in excl]

    def refresh(self):
        st = self.study
        self._filling = True
        try:
            self.p_lib.blockSignals(True)
            cur_lib = self.p_lib.currentData()
            self.p_lib.clear()
            self.p_lib.addItem("automatic by materials", None)
            try:
                from fea_core.calc import friction_lib as fl
                for p in fl.pairs():
                    rng = (f"μ_min {fmt(p.mu_min, 3)}" if p.mu_min is not None else "μ_min ?") + ", " + \
                          (f"μ_max {fmt(p.mu_max, 3)}" if p.mu_max is not None else "μ_max ?")
                    self.p_lib.addItem(f"{p.name} ({rng})", p.name)
                    self.p_lib.setItemData(self.p_lib.count() - 1, f"{p.note}\nSource: {p.source}".strip(),
                                           Qt.ToolTipRole)
            except Exception:  # noqa: BLE001 – library not available
                pass
            k = self.p_lib.findData(cur_lib)
            self.p_lib.setCurrentIndex(max(k, 0))
            self.p_lib.blockSignals(False)
            names = self.part_names() if st is not None else []
            sels = sorted(n for n, s in (st.selections.items() if st is not None else [])
                          if s.entity == "faces" and s.faces)
            for cb in (self.p_a, self.p_b):
                t = cb.currentText()
                cb.clear()
                cb.addItems(names + sels)
                cb.setEditText(t)
            if st is None:
                return
            cs = st.contact_settings
            self.s_method.setCurrentIndex(max(0, self.s_method.findData(cs.method)))
            self.s_split.setCurrentIndex(max(0, self.s_split.findData(cs.split)))
            self.s_tol.setValue(cs.detect_tol or 0.0)
            self.s_stab.setCurrentIndex(max(0, self.s_stab.findData(cs.stabilize)))
            self.s_stab_k.setValue(cs.stabilize_k or 0.0)
            self.s_nlgeom.setChecked(cs.nlgeom)
            self.s_press.setCurrentIndex(max(0, self.s_press.findData(cs.pressure)))
            self.s_press.setEnabled(cs.nlgeom)
            self.s_bolt.setChecked(bool(getattr(st.joints, "contact_joint", False)))
            self.s_bolt.setEnabled(bool(st.bolts))
            an = st.analysis
            self.a_kind.setCurrentIndex(max(0, self.a_kind.findData(an.kind)))
            self.a_mode.setCurrentIndex(0 if an.branches else 1)
            self.a_keep.setChecked(an.stop_keep)
            self.k_case.clear()
            self.k_case.addItem("(none – motion / contact change only)", None)
            for lc in st.load_cases:
                self.k_case.addItem(lc.name, lc.name)
            from fea_core.study import RemoteSupport
            self.mo_target.clear()
            for c in st.constraints:
                if isinstance(c, RemoteSupport) and c.coupling == "rigid":
                    self.mo_target.addItem(c.selection, c.selection)
            self.fill_pairs()
            self.fill_found()
            self.fill_seq()
            if st is not self._last_study or an.branches != self._last_branches:
                if not an.branches:
                    self.sec_analysis.set_open(True)
                self._last_study = st
                self._last_branches = an.branches
        finally:
            self._filling = False
        self.on_kind()
        self.update_banner()
        self.update_info()

    def fill_pairs(self):
        st = self.study
        t = self.table
        t.blockSignals(True)
        t.setRowCount(0)
        from fea_core.contacts.inp import default_mu
        for c in st.contacts:
            r = t.rowCount()
            t.insertRow(r)
            mu_txt = "–"
            if c.kind in ("frictional",):
                mu_txt = fmt(c.mu, 3) if c.mu else f"({fmt(default_mu(st, c), 3)})"
            elif c.kind == "rough":
                mu_txt = "large"
            clr = "–" if c.clearance is None else fmt(c.clearance, 4)
            note = []
            if not c.enabled:
                note.append("off")
            if c.source == "auto":
                note.append("from Find touching parts")
            if c.initial == "penetration" and c.kind != "interference":
                note.append("penetration in CAD!")
            if c.gap is not None:
                note.append(f"{'penetration' if c.gap < 0 else 'gap'} {fmt(abs(c.gap), 3)} mm")
            if c.conductance:
                note.append(f"h_c {fmt(c.conductance, 4)} W/(m²·K)")
            vals = [c.name, c.a, c.b or "ground", CONTACT_LABELS.get(c.kind, c.kind), mu_txt,
                    "slip" if c.purpose == "slip" else "resistance",
                    fmt(c.stiffness, 4) if c.stiffness else "auto", fmt(c.stick_slope, 4) if c.stick_slope else "auto",
                    clr, "yes" if c.adjust else "–", STATE_LABELS.get(c.initial, "–"), ", ".join(note)]
            for k, v in enumerate(vals):
                it = _item(str(v), k in (4, 6, 7, 8))
                if k == 3:
                    it.setBackground(QColor(*TYPE_COLORS.get(c.kind, (200, 200, 200))).lighter(150))
                    it.setForeground(QColor(20, 20, 20))
                    it.setToolTip(TYPE_TIPS.get(c.kind, ""))
                if k == 11 and c.initial == "penetration" and c.kind != "interference":
                    it.setForeground(QColor("#d23b3b"))
                t.setItem(r, k, it)
        if 0 <= self._row < len(st.contacts):
            t.selectRow(self._row)
        t.blockSignals(False)
        if not (0 <= self._row < len(st.contacts)):
            self._row = -1
            self._loaded = None
        elif self._loaded != (self._row, st.contacts[self._row].to_dict()):
            self._load_form(self._row)               # form = selected pair (after adding, undo / redo); unfinished
            # the form (sides from the selection) is not overwritten until the pair changes

    def fill_found(self):
        t = self.found
        t.setRowCount(0)
        have = {frozenset((c.a, c.b)) for c in self.study.contacts} if self.study is not None else set()
        for d in self._found:
            r = t.rowCount()
            t.insertRow(r)
            gap = -d.get("depth", 0.0) if d["state"] == "penetration" else d["distance"]
            vals = [d["a"], d["b"], STATE_LABELS.get(d["state"], d["state"]), fmt(gap, 4), fmt(d["area"], 4),
                    "yes" if frozenset((d["a"], d["b"])) in have else ""]
            for k, v in enumerate(vals):
                it = _item(v, k in (3, 4))
                if k == 2 and d["state"] == "penetration":
                    it.setForeground(QColor("#d23b3b"))
                    help_tip(it, "Part interference in CAD (press fit or assembly error)",
                             "Parts penetrate each other: press fit (Interference type) or CAD assembly error (modeled threads, nominal overlap).")
                t.setItem(r, k, it)

    def fill_seq(self):
        st = self.study
        t = self.seq
        t.blockSignals(True)
        cur = t.currentRow()
        t.setRowCount(0)
        for i, s in enumerate(st.analysis.sequence):
            t.insertRow(i)
            vals = [str(i + 1), s.name, s.case or "–", fmt(s.scale, 4), dict(BOLTS).get(s.bolts, s.bolts),
                    str(len(s.prescribed)) if s.prescribed else "–", str(s.cycles)]
            for k, v in enumerate(vals):
                t.setItem(i, k, _item(v, k in (0, 3, 5, 6)))
        if 0 <= cur < t.rowCount():
            t.selectRow(cur)
        t.blockSignals(False)
        names = [c.name for c in st.contacts]
        self.k_off.clear()
        for n in names:
            it = QListWidgetItem(n)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.k_off.addItem(it)
        self.k_mu.set_rows([[n, None] for n in names if self._friction_pair(n)])
        self.k_mu.setEnabled(not st.analysis.branches)
        enabled = not st.analysis.branches
        for w in (self.seq, self.btn_add_step, self.btn_update_step, self.btn_delete_step, self.btn_step_up,
                  self.btn_step_down, self.btn_steps_cases):
            w.setEnabled(enabled)

    def _friction_pair(self, name) -> bool:
        c = next((c for c in self.study.contacts if c.name == name), None)
        return c is not None and c.kind in ("frictional", "rough", "no_separation", "frictionless")

    def update_banner(self):
        st = self.study
        if st is None:
            self.banner.setText("")
            return
        if st.is_nonlinear():
            why = []
            if st.active_contacts():
                why.append(f"active contacts: {len(st.active_contacts())}")
            if st.contact_settings.nlgeom:
                why.append("large deformations")
            if st.analysis.nonlinear:
                why.append("nonlinear analysis")
            self.banner.setText("<span style='color:#d23b3b'><b>The study is nonlinear</b> (" + ", ".join(why) +
                                ") – states are computed one after another, fatigue by superposition is not available, NGSolve is not used (always CalculiX).</span>")
        else:
            self.banner.setText("<span style='color:#2e9e4f'>The study is linear</span> – touching parts are rigidly connected by a shared mesh, superposition of cases holds.")

    def update_info(self):
        st = self.study
        if st is None:
            self.info.setText("")
            return
        from fea_core.contacts.check import check
        try:
            errors, warnings = check(st, [p.name for p in self.win.parts])
        except Exception as e:  # noqa: BLE001 – the check must not crash the panel
            errors, warnings = [f"Contact check failed: {e}"], []
        rows = [f"<span style='color:#d23b3b'>● {escape(e)}</span>" for e in errors]
        rows += [f"<span style='color:#b36b00'>● {escape(w)}</span>" for w in warnings]
        if st.contacts and not errors:
            rows.append("<span style='color:#2e9e4f'>● Contacts are defined without errors.</span>")
        rows.append("<small>μ is an estimate (bounds μ_min / μ_max by purpose), K and λ are penalties – check the results for sensitivity. Tables of μ with sources: help <i>Friction</i>.</small>")
        self.info.setText("<br>".join(rows))

    def color(self, errors, warnings) -> str:
        """Step colour on the progress bar: grey = no contacts, red = error, orange = warnings, green = ready."""
        from .main import GREEN, GREY, ORANGE, RED
        st = self.study
        if st is None or not (st.contacts or st.analysis.nonlinear or st.contact_settings.nlgeom or
                              getattr(st.joints, "contact_joint", False)):
            return GREY
        pre = ("Contact", "Sequence", "Step '", "Large deformations", "Calculation type")
        if any(e.startswith(pre) for e in errors):
            return RED
        return ORANGE if any(w.startswith(pre) for w in warnings) else GREEN

    # ================================================================== finding contacts
    def find_touching(self):
        if self.study is None or self._thread is not None:
            return
        names = self.part_names()
        if len(names) < 2:
            QMessageBox.information(self, APP, "The model has fewer than two parts – nothing to search.")
            return
        from .workers import TaskThread
        step = self.win.step_path
        tol = self.tol.value()
        from fea_core.geometry import legacy_step_names
        legacy = legacy_step_names(self.study)          # part names as in the saved study (older SolidWorks STEP)

        def job(_th):
            from fea_core.contacts.detect import find_touching
            from fea_core.geometry import read_step
            # parts are reloaded in a thread – OCP operations (intersection of solids) over shapes shown in 3D would collide
            parts = read_step(step, legacy_names=legacy) if step is not None else []
            return find_touching(parts, tol, names=set(names))
        t = TaskThread(job, self)
        t.done.connect(self.on_found)
        t.failed.connect(self.on_find_failed)
        t.finished.connect(self._find_finished)
        self._thread = t
        self.win.register_aux(t)
        self.btn_find.setEnabled(False)
        self.btn_find.setText("Searching for touching parts…")
        self.win.log_msg(f"Searching for touching parts (distance {fmt(tol)} mm)…")
        t.start()

    def _find_finished(self):
        self.win.unregister_aux(self._thread)
        self._thread = None
        self.btn_find.setEnabled(True)
        self.btn_find.setText("Find touching parts")

    def on_found(self, found):
        self._found = list(found)
        names = [p.name for p in self.win.parts]
        self._det_faces = {}
        for d in self._found:
            if d["a"] in names and d["b"] in names:
                self._det_faces[(d["a"], d["b"])] = ([(names.index(d["a"]), k) for k in d["faces_a"]],
                                                     [(names.index(d["b"]), k) for k in d["faces_b"]])
        n_pen = sum(d["state"] == "penetration" for d in self._found)
        self.win.log_msg(f"Pairs of parts in contact found: {len(self._found)}" + (f" (penetration: {n_pen})" if n_pen else ""))
        self.fill_found()
        self.win.refresh_sets()

    def on_find_failed(self, text):
        self.win.log_msg(f"ERROR finding touching parts: {text}")
        QMessageBox.warning(self, APP, f"Finding touching parts failed:\n{text}")

    def unique_name(self, base: str) -> str:
        names = {c.name for c in self.study.contacts}
        if base not in names:
            return base
        k = 2
        while f"{base} ({k})" in names:
            k += 1
        return f"{base} ({k})"

    def add_found(self):
        st = self.study
        if st is None or not self._found:
            QMessageBox.information(self, APP, "First run “Find touching parts”.")
            return
        rows = sorted({i.row() for i in self.found.selectedIndexes()})
        have = {frozenset((c.a, c.b)) for c in st.contacts}
        if not rows:
            rows = [i for i, d in enumerate(self._found) if frozenset((d["a"], d["b"])) not in have]
        kind = self.add_kind.currentData()
        added = 0
        for i in rows:
            d = self._found[i]
            if frozenset((d["a"], d["b"])) in have:
                continue
            gap = -float(d.get("depth", 0.0)) if d["state"] == "penetration" else float(d["distance"])
            st.contacts.append(ContactPair(self.unique_name(f"{d['a']} – {d['b']}"), d["a"], d["b"], kind=kind,
                                           initial=d["state"], gap=gap, area=float(d["area"]), source="auto"))
            have.add(frozenset((d["a"], d["b"])))
            added += 1
        if added:
            self._row = len(st.contacts) - 1
            self.win.changed()
            self.win.log_msg(f"Contact pairs added: {added} ({CONTACT_LABELS[kind]})")

    # ================================================================== pairs
    def side_from_selection(self, side: str):
        if self.study is None:
            return
        sels = self.win.need_faces()
        if not sels:
            return
        name = self.win.new_selection(f"Contact {side}", sels)
        self.win.view.set_selected_faces([])
        (self.p_a if side == "A" else self.p_b).setEditText(name)
        self.win.changed(invalidate_mesh=False)

    def on_kind(self, *_):
        kind = self.p_kind.currentData()
        fr = kind in ("frictional",)
        for w in (self.p_mu, self.p_purpose, self.p_lib):
            w.setEnabled(fr or kind == "rough")
        self.p_mu.setEnabled(fr)
        self.p_clr_on.setEnabled(kind in ("interference", "frictionless", "frictional", "rough", "compression_only",
                                          "no_separation"))
        self.p_clr.setEnabled(self.p_clr_on.isEnabled() and self.p_clr_on.isChecked())
        self.p_adjust.setEnabled(kind != "interference")
        self.p_hc.setEnabled(kind not in ("bonded", "none"))
        self.update_mu_info()

    def pair_from_form(self, name: str | None = None) -> ContactPair | None:
        a, b = self.p_a.currentText().strip(), self.p_b.currentText().strip()
        kind = self.p_kind.currentData()
        if not a:
            QMessageBox.warning(self, APP, "Enter side A (part or face selection from 3D).")
            return None
        if not b and kind != "compression_only":
            QMessageBox.warning(self, APP, "Enter side B (empty only for the “Compression only” type = support to ground).")
            return None
        name = (name or self.p_name.text().strip() or f"{a} – {b or 'ground'}")
        clr = float(self.p_clr.value()) if (self.p_clr_on.isChecked() and self.p_clr_on.isEnabled()) else None
        return ContactPair(name, a, b, kind=kind, mu=_num_or_none(self.p_mu.value()) if kind == "frictional" else None,
                           purpose=self.p_purpose.currentData(), materials=self.p_lib.currentData(),
                           stiffness=_num_or_none(self.p_K.value()), stick_slope=_num_or_none(self.p_lam.value()),
                           clearance=clr, adjust=self.p_adjust.isChecked() and kind != "interference",
                           slave=self.p_slave.currentData(), enabled=self.p_enabled.isChecked(),
                           conductance=_num_or_none(self.p_hc.value()) if kind not in ("bonded", "none") else None)

    def pair_sides(self) -> tuple[str, str]:
        """Sides A, B of the form (for the h_c estimate from the materials)."""
        return self.p_a.currentText().strip(), self.p_b.currentText().strip()

    def add_pair(self):
        st = self.study
        if st is None:
            return
        c = self.pair_from_form()
        if c is None:
            return
        c.name = self.unique_name(c.name)
        st.contacts.append(c)
        self._row = len(st.contacts) - 1
        self.win.changed()

    def update_pair(self):
        st = self.study
        r = self._row
        if st is None or not (0 <= r < len(st.contacts)):
            QMessageBox.warning(self, APP, "Select a pair in the table.")
            return
        old = st.contacts[r]
        c = self.pair_from_form()
        if c is None:
            return
        if c.name != old.name and c.name in {x.name for x in st.contacts}:
            QMessageBox.warning(self, APP, f"Pair '{c.name}' already exists.")
            return
        c.initial, c.gap, c.area, c.source = old.initial, old.gap, old.area, old.source
        if c.name != old.name:                      # references from the sequence of steps
            for s in st.analysis.sequence:
                s.contacts_off = [c.name if n == old.name else n for n in s.contacts_off]
                if old.name in s.mu_change:
                    s.mu_change[c.name] = s.mu_change.pop(old.name)
        if c == old:
            return
        st.contacts[r] = c
        self.win.changed()

    def delete_pair(self):
        st = self.study
        r = self._row
        if st is None or not (0 <= r < len(st.contacts)):
            return
        c = st.contacts[r]
        for side in (c.a, c.b):                     # face selections that nothing else uses (before the pair is removed)
            if side in st.selections:
                self.win._drop_selection(side)
        st.contacts.pop(r)
        for s in st.analysis.sequence:
            s.contacts_off = [n for n in s.contacts_off if n != c.name]
            s.mu_change.pop(c.name, None)
        self._row = min(r, len(st.contacts) - 1)
        self.win.changed()

    def on_row(self, r):
        st = self.study
        if self._filling or st is None or not (0 <= r < len(st.contacts)):
            return
        self._row = r
        self._load_form(r)
        self.win.refresh_sets()

    def _load_form(self, r):
        c = self.study.contacts[r]
        self._loaded = (r, c.to_dict())
        filling = self._filling
        self._filling = True
        try:
            self.p_name.setText(c.name)
            self.p_a.setEditText(c.a)
            self.p_b.setEditText(c.b or "")
            self.p_kind.setCurrentIndex(max(0, self.p_kind.findData(c.kind)))
            self.p_mu.setValue(c.mu or 0.0)
            self.p_purpose.setCurrentIndex(max(0, self.p_purpose.findData(c.purpose)))
            self.p_lib.setCurrentIndex(max(0, self.p_lib.findData(c.materials)))
            self.p_K.setValue(c.stiffness or 0.0)
            self.p_lam.setValue(c.stick_slope or 0.0)
            self.p_clr_on.setChecked(c.clearance is not None)
            self.p_clr.setValue(c.clearance or 0.0)
            self.p_adjust.setChecked(c.adjust)
            self.p_slave.setCurrentIndex(max(0, self.p_slave.findData(c.slave)))
            self.p_hc.setValue(c.conductance or 0.0)
            self.p_enabled.setChecked(c.enabled)
        finally:
            self._filling = filling
        if not filling:
            self.on_kind()
        else:                                        # when filling the panel only the enabling of fields (μ info comes afterwards)
            kind = c.kind
            self.p_mu.setEnabled(kind == "frictional")
            self.p_purpose.setEnabled(kind in ("frictional", "rough"))
            self.p_lib.setEnabled(kind in ("frictional", "rough"))

    def select_pair(self, name: str):
        """Selects the pair in the table by name (click on a face in 3D, study tree)."""
        names = [c.name for c in self.study.contacts]
        if name in names:
            self.table.selectRow(names.index(name))

    def update_mu_info(self):
        st = self.study
        if st is None or self._filling:
            return
        kind = self.p_kind.currentData()
        if kind not in ("frictional", "rough"):
            self.mu_info.setText("<small>Type without friction – μ is not used.</small>" if kind != "no_separation" else
                                 "<small>No separation: tangentially free (ccx TIED, small λ).</small>")
            return
        if kind == "rough":
            self.mu_info.setText("<small>Rough contact = large μ (10).</small>")
            return
        if self.p_mu.value() > 0:
            self.mu_info.setText(f"<small>Given μ = {fmt(self.p_mu.value(), 3)} (takes precedence over the library).</small>")
            return
        try:
            from fea_core.contacts.inp import default_mu
            from fea_core.calc import friction_lib as fl
            c = ContactPair("_", self.p_a.currentText().strip(), self.p_b.currentText().strip(), kind=kind,
                            purpose=self.p_purpose.currentData(), materials=self.p_lib.currentData())
            mu = default_mu(st, c)
            src = ""
            if c.materials:
                p = fl.get(c.materials)
                src = f"pair “{p.name}”; {p.source}"
                if p.unverified_value(c.purpose):
                    src += " – UNVERIFIED"
            else:
                from fea_core.contacts.detect import pair_parts
                pa, pb = pair_parts(st, c)
                try:
                    ma = st.material_of(sorted(pa)[0]).name
                    mb = st.material_of(sorted(pb)[0]).name
                    sug = fl.suggest(ma, mb)
                    src = (f"by materials {ma} – {mb}: “{sug[0]['name']}”" if sug else
                           f"pair {ma} – {mb} is not in the library → {fmt(mu, 3)} (measure it)")
                except (KeyError, IndexError):
                    src = "the part materials are not known"
            self.mu_info.setText(f"<small>μ from library = <b>{fmt(mu, 3)}</b> ({('slip → μ_min' if c.purpose == 'slip' else 'resistance → μ_max')}); {escape(str(src))}</small>")
        except Exception as e:  # noqa: BLE001
            self.mu_info.setText(f"<small>μ cannot be determined: {escape(str(e))}</small>")

    # ================================================================== settings
    def on_settings(self, *_):
        st = self.study
        if self._filling or st is None:
            return
        cs = st.contact_settings
        new = (self.s_method.currentData(), self.s_split.currentData(), _num_or_none(self.s_tol.value()),
               self.s_stab.currentData(), _num_or_none(self.s_stab_k.value()), self.s_nlgeom.isChecked(),
               self.s_press.currentData())
        bolt = self.s_bolt.isChecked()
        if new == (cs.method, cs.split, cs.detect_tol, cs.stabilize, cs.stabilize_k, cs.nlgeom, cs.pressure) and \
                bolt == bool(getattr(st.joints, "contact_joint", False)):
            return
        (cs.method, cs.split, cs.detect_tol, cs.stabilize, cs.stabilize_k, cs.nlgeom, cs.pressure) = new
        st.joints.contact_joint = bolt
        self.win.changed()

    def on_analysis(self, *_):
        st = self.study
        if self._filling or st is None:
            return
        an = st.analysis
        new = (self.a_kind.currentData(), bool(self.a_mode.currentData()), self.a_keep.isChecked())
        if new == (an.kind, an.branches, an.stop_keep):
            return
        branch_changed = (an.branches != new[1])
        an.kind, an.branches, an.stop_keep = new
        self._last_branches = an.branches
        if branch_changed and not an.branches:
            self.sec_analysis.set_open(True)
        if not an.branches and not an.sequence and st.load_cases:
            self._make_steps_from_cases()
        self.win.changed()

    # ================================================================== sequence of steps
    def _make_steps_from_cases(self):
        st = self.study
        st.analysis.sequence = [LoadStep(lc.name, lc.name) for lc in st.load_cases]

    def steps_from_cases(self):
        if self.study is None:
            return
        self._make_steps_from_cases()
        self.win.changed()

    def step_from_form(self) -> LoadStep | None:
        name = self.k_name.text().strip()
        if not name:
            QMessageBox.warning(self, APP, "Enter the step name.")
            return None
        off = [self.k_off.item(i).text() for i in range(self.k_off.count())
               if self.k_off.item(i).checkState() == Qt.Checked]
        muc = {}
        try:
            for row in self.k_mu.values([str, float]):
                if row[1] is not None:
                    muc[row[0]] = float(row[1])
        except ValueError:
            QMessageBox.warning(self, APP, "Friction change: μ must be a number.")
            return None
        return LoadStep(name, self.k_case.currentData(), self.k_scale.value(), self.k_ramp.currentData(),
                        self.k_bolts.currentData(), off, muc, [Motion(**vars(m)) for m in self._motions],
                        self.k_dt0.value(), self.k_dtmin.value(), self.k_dtmax.value(), self.k_maxinc.value(),
                        self.k_direct.isChecked(), self.k_every.value(), self.k_cycles.value())

    def add_step(self):
        st = self.study
        if st is None:
            return
        s = self.step_from_form()
        if s is None:
            return
        if s.name in {x.name for x in st.analysis.sequence}:
            QMessageBox.warning(self, APP, f"Step '{s.name}' already exists.")
            return
        st.analysis.sequence.append(s)
        self.win.changed()
        self.seq.selectRow(len(st.analysis.sequence) - 1)

    def update_step(self):
        st = self.study
        r = self.seq.currentRow()
        if st is None or not (0 <= r < len(st.analysis.sequence)):
            QMessageBox.warning(self, APP, "Select a step in the table.")
            return
        s = self.step_from_form()
        if s is None:
            return
        if s.name != st.analysis.sequence[r].name and s.name in {x.name for x in st.analysis.sequence}:
            QMessageBox.warning(self, APP, f"Step '{s.name}' already exists.")
            return
        st.analysis.sequence[r] = s
        self.win.changed()
        self.seq.selectRow(r)

    def delete_step(self):
        st = self.study
        r = self.seq.currentRow()
        if st is None or not (0 <= r < len(st.analysis.sequence)):
            return
        st.analysis.sequence.pop(r)
        self.win.changed()

    def move_step(self, d: int):
        st = self.study
        r = self.seq.currentRow()
        if st is None or not (0 <= r < len(st.analysis.sequence)) or not (0 <= r + d < len(st.analysis.sequence)):
            return
        seq = st.analysis.sequence
        seq[r], seq[r + d] = seq[r + d], seq[r]
        self.win.changed()
        self.seq.selectRow(r + d)

    def on_seq_row(self, r):
        st = self.study
        if self._filling or st is None or not (0 <= r < len(st.analysis.sequence)):
            return
        Section.reveal(self.k_name)
        s = st.analysis.sequence[r]
        self.k_name.setText(s.name)
        self.k_case.setCurrentIndex(max(0, self.k_case.findData(s.case)))
        self.k_scale.setValue(s.scale)
        self.k_ramp.setCurrentIndex(max(0, self.k_ramp.findData(s.ramp)))
        self.k_bolts.setCurrentIndex(max(0, self.k_bolts.findData(s.bolts)))
        for i in range(self.k_off.count()):
            it = self.k_off.item(i)
            it.setCheckState(Qt.Checked if it.text() in s.contacts_off else Qt.Unchecked)
        self.k_mu.set_rows([[n, s.mu_change.get(n)] for n in [c.name for c in st.contacts] if self._friction_pair(n)])
        self.k_dt0.setValue(s.dt0)
        self.k_dtmin.setValue(s.dt_min)
        self.k_dtmax.setValue(s.dt_max)
        self.k_maxinc.setValue(s.max_inc)
        self.k_every.setValue(s.output_every)
        self.k_cycles.setValue(s.cycles)
        self.k_direct.setChecked(s.direct)
        self._motions = [Motion(**vars(m)) for m in s.prescribed]
        self.fill_motions()

    # ------------------------------------------------------------------ prescribed motion
    def fill_motions(self):
        self.mo_list.blockSignals(True)
        self.mo_list.clear()
        for m in self._motions:
            unit = "mm" if m.kind == "disp" else "rad"
            end = m.amplitude[-1][1] if m.amplitude else 0.0
            self.mo_list.addItem(f"{m.target}: {('displacement' if m.kind == 'disp' else 'rotation')} {DOF_NAMES[m.dof - 1]}, end {fmt(float(end))} {unit}")
        self.mo_list.blockSignals(False)

    def on_motion_row(self, r):
        if not (0 <= r < len(self._motions)):
            return
        Section.reveal(self.mo_target)
        m = self._motions[r]
        self.mo_target.setCurrentIndex(max(0, self.mo_target.findData(m.target)))
        self.mo_kind.setCurrentIndex(max(0, self.mo_kind.findData(m.kind)))
        self.mo_dof.setCurrentIndex(int(m.dof) - 1)
        self.mo_amp.set_rows(m.amplitude)

    def add_motion(self):
        tgt = self.mo_target.currentData()
        if not tgt:
            QMessageBox.warning(self, APP, "Prescribed motion needs a rigid remote support (Supports step: remote support, coupling “rigid”) – the faces that move.")
            return
        try:
            amp = [[float(t), float(v)] for t, v in self.mo_amp.values()]
        except (TypeError, ValueError):
            QMessageBox.warning(self, APP, "Motion history: fill in pairs of numbers [t, value].")
            return
        if len(amp) < 2 or any(b[0] <= a[0] for a, b in zip(amp, amp[1:])):
            QMessageBox.warning(self, APP, "Motion history needs at least 2 points with increasing t (0–1).")
            return
        self._motions.append(Motion(tgt, self.mo_kind.currentData(), self.mo_dof.currentIndex() + 1, amp))
        self.fill_motions()
        self.win.log_msg("Motion added to the step form – save the step (“Save step” / “Add step”).")

    def delete_motion(self):
        r = self.mo_list.currentRow()
        if 0 <= r < len(self._motions):
            self._motions.pop(r)
            self.fill_motions()

    # ================================================================== 3D
    def side_faces(self, side: str, other: str) -> list:
        """Faces of a pair side: selection of faces, or the faces of the part in contact from the last detection, otherwise the whole part."""
        win = self.win
        if not side:
            return []
        if side in self.study.selections:
            return win.sel_faces.get(side, [])
        names = [p.name for p in win.parts]
        if side not in names:
            return []
        det = self._det_faces.get((side, other)) or self._det_faces.get((other, side))
        if det is not None:
            return det[0] if (side, other) in self._det_faces else det[1]
        pi = names.index(side)
        return [(pi, fi) for fi in range(len(win.view.part_faces[pi]))]

    def show_3d(self):
        st = self.study
        v = self.win.view
        for k, c in enumerate(st.contacts):
            col = TYPE_COLORS.get(c.kind, (200, 200, 200))
            sel = k == self._row
            transp = 0.15 if sel else (0.55 if self._row >= 0 else 0.35)
            if not c.enabled:
                transp = 0.8
            for side, other, tag in ((c.a, c.b, "A"), (c.b, c.a, "B")):
                faces = self.side_faces(side, other)
                if faces:
                    v.set_face_set(f"k:{k}:{tag}", faces, col, label=f"{c.name} ({tag})" if sel else None,
                                   transparency=transp)

    def on_faces_picked(self, faces):
        """A click on a pair face in 3D selects the pair in the table (the face selection stays for "A / B from selection")."""
        win = self.win
        if not faces or self.study is None or getattr(win, "stage", -1) != win.STAGE.get("Contacts", -2):
            return
        f = tuple(faces[-1])
        for k, c in enumerate(self.study.contacts):
            if any(tuple(x) == f for side, other in ((c.a, c.b), (c.b, c.a)) for x in self.side_faces(side, other)):
                if k != self._row:
                    self.table.selectRow(k)
                return
