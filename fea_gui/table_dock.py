"""Bottom dock "Tables" (Studio × Tables layout): table tabs by step (TABS), actions on the right
in the tab row, note at the bottom. The parts have editable cells (Show, Count, Material).

- parts (Model, Materials, 3D printing): material, volume, mass of the solid material, whether it is counted; summary and weight,
- loads and constraints (Constraints, Loads, Dynamics): matrix of load states – a constraint applies in all states, a load
  in its own state; summary = number of loads in the state,
- mesh by parts (Mesh, Solve): element size, numbers of elements and nodes; summary with the min. scaled Jacobian,
- results of the states (Results, Fatigue): max. von Mises, max. outside singularities (the same classification of places as the
  "Highest stress" table), safety to Re at that place, max. displacement, resultant of the reactions.

Read-only; a click on a row selects the item in the main window (part in 3D, constraint / load in the panel, state in the
results). The data come from the main window – nothing is recomputed except the summary of the results (maximum places are in the window's cache,
`MainWindow.result_hotspots`).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QStackedWidget, QTabBar, QTableWidgetItem, QVBoxLayout, QWidget)

from . import theme
from fea_core.i18n import tr
from fea_core.richtext import safe_rich

from .widgets import fmt, table, table_text

GROUP_OF = {"Model": "parts", "Materials": "parts", "3D printing": "parts", "Joints": "joints",
            "Constraints": "loads", "Loads": "loads",
            "Heat": "thermal", "Contacts": "contacts", "Members": "frame", "Motion": "motion", "Dynamics": "dynamics",
            "Mesh": "mesh", "Solve": "mesh",
            "Results": "results", "Fatigue": "results"}
# table tabs of the group (first = default; the group key is also the key of the first table)
TABS = {"parts": ["parts", "materials", "geometry", "csys"],
        "joints": ["bolts", "welds", "bonded", "ignored"],
        "loads": ["loads", "cases", "combos"],
        "thermal": ["thermal"], "contacts": ["contacts", "bonded"], "frame": ["frame"], "motion": ["motion"],
        "dynamics": ["dynamics"],
        "mesh": ["mesh", "refine", "log", "messages"],
        "results": ["results", "hotspots", "psafety", "reactions", "jutil"]}
TITLES = {"parts": "Parts", "loads": "Loads and supports", "mesh": "Mesh by part", "results": "Load case results",
          "thermal": "Thermal boundary conditions", "contacts": "Contact pairs", "frame": "Members",
          "motion": "Mechanism", "dynamics": "Added masses",
          "materials": "Study materials", "geometry": "Geometry check", "csys": "Coordinate systems",
          "bolts": "Bolts", "welds": "Welds", "bonded": "Part connections", "ignored": "Excluded parts",
          "cases": "Load cases", "combos": "Combination", "refine": "Refinement",
          "hotspots": "Highest stresses", "reactions": "Reactions", "jutil": "Joints – utilization",
          "psafety": "Safety by part"}
DOCK_TITLE = "Tables"
# tabs that only switch to another bottom dock (Solver log, Messages) – they have no table of their own
LINK_TABS = {"log": "dock_log", "messages": "dock_messages"}
TITLES.update({"log": "Solver log", "messages": "Messages"})
G_MM = 9810.0                                   # mm/s²

CON_KIND = {"Fixed": "Fixed support", "Displacement": "Prescribed displacement", "CylindricalSupport": "Cylindrical support",
            "RemoteSupport": "Remote support", "ElasticSupport": "Elastic support"}
LOAD_KIND = {"Force": "Force", "Pressure": "Pressure", "RemoteLoad": "Remote load", "BearingLoad": "Bearing load",
             "Gravity": "Gravity", "Rotation": "Rotation", "ThermalLoad": "Temperature", "Inertia": "Inertia"}


DEFAULT_REQUIRED = 1.5


def required_safety(study) -> float:
    """Required statics safety (to Re outside singularities) – Study.assessment["required_safety"], default 1.5."""
    try:
        return float((getattr(study, "assessment", None) or {}).get("required_safety", DEFAULT_REQUIRED))
    except (TypeError, ValueError):
        return DEFAULT_REQUIRED


def step_summaries(main) -> list[dict]:
    """Summary of the mechanical states for the Tables dock and the Assessment card (the same numbers in both places): for each state
    max. von Mises, max. outside singularities (the first place of `result_hotspots` that is neither a singularity nor affected
    by a boundary condition) with the parts, safety to Re at that place, max. displacement, |reaction|. Cache on the results object.
    """
    res = getattr(main, "results", None)
    if res is None:
        return []
    cached = getattr(res, "_gui_step_summary", None)
    if cached is not None:
        return cached
    steps = [(i, s) for i, s in enumerate(res.steps) if len(s.mises)]
    try:
        re_node = main.node_re() if steps else None
    except Exception:  # noqa: BLE001 – a missing material must not crash the overview
        re_node = None
    out = []
    for i, s in steps:
        vm = s.mises
        real = None
        d = main.result_hotspots("step", i, "mises")
        for h in (d or {}).get("hotspots", []):
            if h.get("classification", "") not in ("singularity", "affected by boundary condition"):
                real = h
                break
        row = {"index": i, "name": s.name, "vmax": float(vm.max()), "umax": float(s.umag.max()),
               "real": None, "parts": [], "k": None, "reaction": None}
        if real is not None:
            v = float(vm[real["node"] - 1])
            row.update(real=v, parts=list(real.get("parts", [])))
            if re_node is not None:
                k = float(re_node[real["node"] - 1] / max(v, 1e-9))
                row["k"] = k if np.isfinite(k) else None
        rt = (s.equilibrium or {}).get("reaction_total")
        if rt is not None:
            row["reaction"] = float(np.linalg.norm(rt))
        out.append(row)
    # reaction noise (.frd has 6 valid digits) → 0 below 1e-5 × the largest reaction over the states (as in the Word report)
    tiny = 1e-5 * max([d["reaction"] for d in out if d["reaction"] is not None] + [0.0])
    for d in out:
        if d["reaction"] is not None and d["reaction"] < tiny:
            d["reaction"] = 0.0
    res._gui_step_summary = out
    return out


def _int(n) -> str:
    """Integer with a space as the thousands separator (18 412) – fmt would write large numbers with an exponent."""
    return f"{int(n):,}".replace(",", " ")


def _item(text, right=False, color=None, bold=False, tip=""):
    it = QTableWidgetItem(text)
    if right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if color:
        it.setForeground(QColor(color))
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    if tip:
        it.setToolTip(safe_rich(tip))                  # tooltips are rich text and may carry part / case names from foreign files
    return it


class PhaseTables(QWidget):
    """Dock content: a note (the table name is the dock title) + one table per group of steps (QStackedWidget)."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.group = "parts"
        self._row_data: dict[str, list] = {}
        self._tab_of: dict[str, str] = {}               # selected tab per group
        self._notes: dict[str, str] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QWidget()
        head.setObjectName("dockHead")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(4, 0, 4, 0)
        hl.setSpacing(3)
        self.tabbar = QTabBar()
        self.tabbar.setObjectName("dockTabs")
        self.tabbar.setDrawBase(False)
        self.tabbar.setExpanding(False)
        self.tabbar.currentChanged.connect(self._on_tab)
        hl.addWidget(self.tabbar)
        hl.addStretch(1)
        self.actions_box = QWidget()
        self._actions_lay = QHBoxLayout(self.actions_box)
        self._actions_lay.setContentsMargins(0, 2, 0, 2)
        self._actions_lay.setSpacing(3)
        hl.addWidget(self.actions_box)
        from .popout import popout_button
        self.btn_popout = popout_button("Tables in a separate window")
        self.btn_popout.clicked.connect(lambda: main.set_tables_floating(not main.dock_tables.isFloating()))
        hl.addWidget(self.btn_popout)
        lay.addWidget(head)
        self.note = QLabel("")                          # note below the table
        self.note.setObjectName("dockNote")
        self.note.setProperty("muted", True)
        self.note.setTextFormat(Qt.PlainText)
        self._build_actions()
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        lay.addWidget(self.note)
        self.tables = {}
        cols = {"parts": ["Part", "Material", "Volume [cm³]", "Mass [kg]", "Calculation"],
                "loads": ["Name", "Type", "Value"],
                "mesh": ["Part", "Element size [mm]", "Elements", "Nodes", "Share of elements [%]"],
                "results": ["State", "Max. σ von Mises [MPa]", "Max. outside singularities [MPa]", "Location",
                            "Safety factor to Re", "Max. displacement [mm]", "Total reaction [N]"],
                "thermal": ["Face / part", "Type", "Value"],
                "contacts": ["Pair", "Part / selection A", "Part / selection B", "Type", "Friction μ", "Interference / clearance [mm]"],
                "frame": ["Member", "Nodes", "Cross-section", "Material", "Length [mm]", "Shape", "Joints"],
                "motion": ["Name", "Type", "Description", "Parts / bodies"],
                "dynamics": ["Name", "Type", "Mass", "Face / selection"],
                "materials": ["Material", "E [MPa]", "ν", "ρ [kg/m³]", "Re [MPa]", "Rm [MPa]", "Parts", "Type"],
                "geometry": ["#", "Warnings from loading the geometry"],
                "csys": ["Name", "Type", "Origin [mm]", "z axis", "x axis"],
                "bolts": ["Name", "Size", "Class", "Joint", "Preload", "Under head", "Under nut", "Washers",
                          "Utilization", "F_SA max [N]", "F_Q max [N]", "Φ", "Governing"],
                "welds": ["Name", "Weld part", "a [mm]", "Length [mm]", "FAT σ / τ", "S-N curve", "Utilization",
                          "Governing"],
                "bonded": ["Part A", "Part B", "Contact area [mm²]", "Connection"],
                "ignored": ["Part", "Reason"],
                "cases": ["State", "Loads", "Kinds", "Supports"],
                "combos": ["Combination", "Factors", "Situation", "Envelope"],
                "refine": ["Selection / location", "Size [mm]", "Type"],
                "hotspots": ["#", "σ [MPa]", "Assessment", "Part", "x, y, z [mm]", "k to Re"],
                "reactions": ["Constraint", "Rx [N]", "Ry [N]", "Rz [N]", "|R| [N]"],
                "jutil": ["Joint", "Type", "Utilization", "Governing"],
                "psafety": ["Part", "Max. σ [MPa]", "Re [MPa]", "k to Re", "k / required", "Note"]}
        holders = {"parts": "Open a model – its parts will appear here",
                   "loads": "No supports or loads yet",
                   "mesh": "The mesh has not been created yet – it is created when solving",
                   "results": "No results yet – run the analysis",
                   "thermal": "No thermal case or boundary condition yet",
                   "contacts": "No contact pairs yet – “Find” on the shelf suggests them",
                   "frame": "No members yet – add nodes and members or load them from a wireframe",
                   "motion": "No mechanism bodies or joints yet",
                   "dynamics": "No added masses yet",
                   "materials": "No material yet",
                   "geometry": "Geometry check without warnings",
                   "csys": "Only the global coordinate system (local ones are set for loads – Systems…)",
                   "bolts": "No bolts yet – “Detect” on the shelf finds them in the model",
                   "welds": "No welds yet",
                   "bonded": "Part connections appear after meshing",
                   "ignored": "All parts are analysed",
                   "cases": "No load case yet",
                   "combos": "No combinations yet – Combinations (EN 1990) on the shelf",
                   "refine": "No local mesh refinement",
                   "hotspots": "No results yet",
                   "reactions": "No results yet",
                   "jutil": "No joint assessment yet",
                   "psafety": "No results yet"}
        for g, c in cols.items():
            t = table(c, stretch_last=True, placeholder=holders[g])
            t.setObjectName(f"table_{g}")
            t.setAlternatingRowColors(True)
            t.verticalHeader().setDefaultSectionSize(20)
            t.itemClicked.connect(lambda it, gg=g: self._on_click(gg, it.row()))
            self.stack.addWidget(t)
            self.tables[g] = t
        self.tables["parts"].setToolTip("Model parts – click selects the part in 3D, Show / Analyse / Material can be changed")
        self.tables["parts"].itemChanged.connect(self._on_part_item)
        self.tables["loads"].itemChanged.connect(self._on_load_item)
        self.tables["loads"].setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.tables["loads"].setToolTip("Supports and loads by load case – a click selects the item in the panel")
        self.tables["mesh"].setToolTip("Element and node counts by part in the created mesh")
        self.tables["results"].setToolTip("Result summary by load case – a click shows the case")
        self.tables["thermal"].setToolTip("Boundary conditions by thermal case – click selects the case in the panel")
        self.tables["contacts"].setToolTip("Contact pairs – click selects the pair in the panel")
        self.tables["frame"].setToolTip("Frame members with cross-section, length and hinges")
        self.tables["motion"].setToolTip("Mechanism bodies, joints and drivers")
        self.tables["dynamics"].setToolTip("Added masses (equipment, contents) for dynamics and statics")
        for k, tip in (("materials", "Study materials with properties and number of parts"),
                       ("geometry", "Geometry check warnings from loading the STEP"),
                       ("csys", "Local coordinate systems of loads"),
                       ("bolts", "Study bolts with preload, clamped parts and utilization"),
                       ("welds", "Study welds with dimensions, FAT and utilization"),
                       ("bonded", "Parts rigidly connected by a shared mesh and the contact area"),
                       ("ignored", "Parts excluded from the analysis"),
                       ("cases", "Load cases and numbers of loads"),
                       ("combos", "Load case combinations (EN 1990)"),
                       ("refine", "Local mesh refinement (faces, spheres, parts)"),
                       ("hotspots", "Highest stress locations of the displayed case with singularity assessment"),
                       ("psafety", "Safety to Re by part in the displayed case"),
                       ("reactions", "Support reactions of the displayed case"),
                       ("jutil", "Utilization of bolts and welds from the assessment")):
            self.tables[k].setToolTip(tip)

    # ------------------------------------------------------------------ switching and refresh
    def set_stage(self, name: str) -> bool:
        """Table tabs for a step; False = the step has no table (the dock stays as it is)."""
        g = GROUP_OF.get(name)
        if g is None:
            return False
        self.group = g
        keys = TABS[g]
        self.tabbar.blockSignals(True)
        while self.tabbar.count():
            self.tabbar.removeTab(0)
        for k in keys:
            self.tabbar.addTab(TITLES[k])
            self.tabbar.setTabData(self.tabbar.count() - 1, k)
        cur = self._tab_of.get(g, keys[0])
        self.tabbar.setCurrentIndex(keys.index(cur) if cur in keys else 0)
        self.tabbar.blockSignals(False)
        self.refresh()
        return True

    @property
    def current(self) -> str:
        k = self.tabbar.tabData(self.tabbar.currentIndex())
        return k if k else self.group

    def show_tab(self, key: str):
        for i in range(self.tabbar.count()):
            if self.tabbar.tabData(i) == key:
                self.tabbar.setCurrentIndex(i)
                return

    def _on_tab(self, i):
        k = self.tabbar.tabData(i)
        if not k:
            return
        if k in LINK_TABS:                              # Solver log / Messages: switch the dock, the table tab stays
            dock = getattr(self.main, LINK_TABS[k], None)
            prev = self._tab_of.get(self.group, TABS[self.group][0])
            self.tabbar.blockSignals(True)
            self.show_tab(prev)
            self.tabbar.blockSignals(False)
            if dock is not None:
                dock.show()
                dock.raise_()
            return
        self._tab_of[self.group] = k
        self.stack.setCurrentWidget(self.tables[k])
        self.note.setText(self._notes.get(k, ""))
        self._update_actions()

    def refresh(self):
        """Fills all tables of the group (counts in the tabs) and shows the selected one."""
        dock = getattr(self.main, "dock_tables", None)
        if dock is not None and dock.windowTitle() != DOCK_TITLE:
            dock.setWindowTitle(DOCK_TITLE)
        for i in range(self.tabbar.count()):
            k = self.tabbar.tabData(i)
            if k in LINK_TABS:
                self.tabbar.setTabText(i, tr(TITLES[k]) + " ↗")
                continue
            self.note.setText("")
            try:
                getattr(self, "_fill_" + k)()
            except Exception as e:  # noqa: BLE001 – one table must not crash the others
                self._finish(self._start(k, 0))
                self.note.setText(f"The table cannot be filled: {e}")
            self._notes[k] = self.note.text()
            n = self._count(k)
            self.tabbar.setTabText(i, f"{tr(TITLES[k])}  {n}" if n is not None else tr(TITLES[k]))
        k = self.current
        self.stack.setCurrentWidget(self.tables[k])
        self.note.setText(self._notes.get(k, ""))
        self._update_actions()

    def _count(self, k):
        """Number of rows without the summary row (the summary has items read-only)."""
        t = self.tables[k]
        n = t.rowCount()
        if n and t.item(n - 1, 0) is not None and not (t.item(n - 1, 0).flags() & Qt.ItemIsSelectable):
            n -= 1
        return n

    # ------------------------------------------------------------------ actions in the tab row
    def _build_actions(self):
        def btn(text, tip, slot):
            b = QPushButton(text)
            b.setObjectName("dockAction")
            b.setToolTip(tip)
            b.clicked.connect(slot)
            self._actions_lay.addWidget(b)
            return b
        m = self.main
        self.act = {
            "add": btn("+ Row", "New item: load on the faces selected in 3D, bolt into the selected hole", self._add_row),
            "show_all": btn("Show all", "Show all parts in 3D", lambda: m._shelf_run("btn_show_all_parts")),
            "isolate": btn("Isolate", "In 3D only the parts selected in the table", self._isolate_rows),
            "assign": btn("Assign material…", "Go to the Materials step with the selected parts", self._go_materials),
            "detect": btn("Detect…", "Find bolts, nuts, washers and welds in the model",
                          lambda: m._shelf_run("joints_panel.detect")),
            "dup": btn("Duplicate", "Copy the selected load into the same case", self._duplicate_load),
            "delete": btn("Delete", "Delete the selected support or load", self._delete_row),
            "show_model": btn("Show in model", "Marker and probe at the location of the selected row", self._show_hotspot),
            "copy": btn("Copy to Excel", "Whole table to the clipboard (pasting into Excel splits the columns)",
                        self._copy),
        }
        self.filter = QLineEdit()
        self.filter.setObjectName("dockFilter")
        self.filter.setPlaceholderText("Filter: *")
        self.filter.setFixedWidth(110)
        self.filter.setToolTip("Select parts by name (* and ?, without them = contains text) – Enter selects in 3D")
        self.filter.returnPressed.connect(self._filter_parts)
        self._actions_lay.insertWidget(2, self.filter)

    def _update_actions(self):
        k = self.current
        vis = {"add": k in ("loads", "bolts"), "show_all": k == "parts", "isolate": k == "parts", "assign": k == "parts",
               "detect": k in ("bolts", "welds"), "dup": k == "loads", "delete": k == "loads",
               "show_model": k == "hotspots", "copy": True}
        for key, b in self.act.items():
            b.setVisible(vis[key])
        self.filter.setVisible(k == "parts")

    def _add_row(self):
        m = self.main
        if self.current == "loads":
            m._shelf_run("btn_add_load")
        elif self.current == "bolts":
            m._shelf_run("@Bolt into selected hole…")

    def _copy(self):
        QApplication.clipboard().setText(table_text(self.tables[self.current]))
        self.main.statusBar().showMessage(tr("Table copied to the clipboard"), 3000)

    def _selected_rows(self, k):
        return sorted({ix.row() for ix in self.tables[k].selectedIndexes()})

    def _isolate_rows(self):
        names = [self.main.parts[self._row_data["parts"][r]].name for r in self._selected_rows("parts")
                 if r < len(self._row_data.get("parts", []))]
        if names:
            self.main.isolate_parts(names)
        else:
            self.main.isolate_selected()

    def _go_materials(self):
        m = self.main
        rows = [self._row_data["parts"][r] for r in self._selected_rows("parts") if r < len(self._row_data["parts"])]
        m.set_stage(m.STAGE["Materials"])
        if rows:
            m.mat_parts.clearSelection()
            for r in rows:
                if r < m.mat_parts.count():
                    m.mat_parts.item(r).setSelected(True)

    def _filter_parts(self):
        m = self.main
        names = set(m.parts_by_pattern(self.filter.text().strip() or "*"))
        idx = [i for i, p in enumerate(m.parts) if p.name in names]
        m.view.set_selected_parts(idx)
        t = self.tables["parts"]
        t.clearSelection()
        for r, pi in enumerate(self._row_data.get("parts", [])):
            if pi in idx:
                t.selectRow(r)

    def _load_row(self):
        rows = self._selected_rows("loads")
        data = self._row_data.get("loads", [])
        return data[rows[0]] if rows and rows[0] < len(data) else None

    def _duplicate_load(self):
        import copy
        m = self.main
        d = self._load_row()
        if d is None or d[0] != "load" or m.study is None:
            return
        ci, li = d[1]
        lc = m.study.load_cases[ci]
        lc.loads.insert(li + 1, copy.deepcopy(lc.loads[li]))
        m.changed()

    def _delete_row(self):
        m = self.main
        d = self._load_row()
        if d is None or m.study is None:
            return
        self._on_click("loads", self._selected_rows("loads")[0])
        if d[0] == "con":
            m.delete_item(m.con_list, "con")
        else:
            m.delete_item(m.load_list, "load")

    def _show_hotspot(self):
        rows = self._selected_rows("hotspots")
        data = self._row_data.get("hotspots", [])
        if rows and rows[0] < len(data) and hasattr(self.main, "hs_table"):
            r = data[rows[0]]
            if r < self.main.hs_table.rowCount():
                self.main.hs_table.selectRow(r)
                self.main.hs_table.cellClicked.emit(r, 0)

    def _start(self, g, n_rows):
        t = self.tables[g]
        t.blockSignals(True)
        for r in range(t.rowCount()):                  # cells with a widget (material menu)
            for c in range(t.columnCount()):
                if t.cellWidget(r, c) is not None:
                    t.removeCellWidget(r, c)
        t.clearContents()
        t.setRowCount(n_rows)
        self._row_data[g] = []
        return t

    def _finish(self, t, summary: list | None = None):
        if summary is not None:
            r = t.rowCount()
            t.setRowCount(r + 1)
            bg = QBrush(QColor(theme.colors()["header"]))
            for c, val in enumerate(summary):
                it = _item(val, right=c > 0 and val[:1].isdigit(), bold=True)
                it.setBackground(bg)
                it.setFlags(Qt.ItemIsEnabled)
                t.setItem(r, c, it)
        t.blockSignals(False)
        t.resizeColumnsToContents()
        t.horizontalHeader().setStretchLastSection(True)

    # ------------------------------------------------------------------ parts
    def _fill_parts(self):
        m = self.main
        st = m.study
        parts = list(getattr(m, "parts", []) or []) if st is not None else []
        t = self._start("parts", len(parts))
        t.setColumnCount(10)
        t.setHorizontalHeaderLabels([tr(x) for x in ("Part", "Show", "Analyze", "Material", "E [MPa]", "Re [MPa]",
                                                     "Volume [cm³]", "Mass [kg]", "Body type", "Check")])
        excl = st.excluded_parts() if st is not None else set()
        ign = set(st.ignored_parts) if st is not None else set()
        hid = set(st.hidden_parts) if st is not None else set()
        surf = set(getattr(m, "surface_names", set()) or set())
        warn = list(getattr(m, "geometry_warnings", None) or [])
        mats = sorted(st.materials) if st is not None else []
        vol_sum = mass_sum = 0.0
        n_calc = 0
        used = set()
        for r, p in enumerate(parts):
            try:
                mat = st.material_of(p.name)
            except KeyError:
                mat = None
            vol = p.volume()                                    # mm³
            mass = vol * mat.rho * 1000.0 if mat is not None and mat.rho else None   # t → kg
            muted = theme.colors()["muted"]
            col = muted if p.name in ign else None
            if p.name not in ign and p.name not in excl:
                vol_sum += vol
                mass_sum += mass or 0.0
                n_calc += 1
                if mat is not None:
                    used.add(mat.name)
            t.setItem(r, 0, _item(p.name, color=col, bold=False))
            for c, on, tip in ((1, p.name not in hid, "Show the part in 3D (view only – results stay)"),
                               (2, p.name not in ign, "The part goes into the analysis (off = excluded, the mesh is discarded)")):
                it = QTableWidgetItem("")
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)
                it.setToolTip(tr(tip))
                t.setItem(r, c, it)
            if p.name in excl:
                t.setItem(r, 3, _item("bolt (beam)", color=muted))
            else:
                cb = QComboBox()
                cb.setProperty("_i18n_done", True)               # material names are not translated
                cb.addItems(mats if mat is None or mat.name in mats else mats + [mat.name])
                if mat is None:
                    cb.insertItem(0, "–")
                    cb.setCurrentIndex(0)
                else:
                    cb.setCurrentText(mat.name)
                cb.setToolTip(tr("Material of the part (study materials; more from the library in the Materials step)"))
                cb.currentTextChanged.connect(lambda name, part=p.name: self._set_material(part, name))
                t.setCellWidget(r, 3, cb)
            t.setItem(r, 4, _item(fmt(mat.E, 6) if mat is not None and mat.E else "–", right=True, color=col))
            re = getattr(mat, "Re", None) if mat is not None else None
            t.setItem(r, 5, _item(fmt(re, 4) if re else "–", right=True, color=col))
            t.setItem(r, 6, _item(fmt(vol / 1000.0, 4), right=True, color=col))
            t.setItem(r, 7, _item(fmt(mass, 4) if mass is not None else "–", right=True, color=col,
                                  tip="Mass of solid material (for 3D printing without infill and shell)"))
            kind = ("bolt (beam)" if p.name in excl else "surface part" if p.name in surf else
                    "printed part" if mat is not None and mat.is_print else "solid body")
            t.setItem(r, 8, _item(kind, color=col))
            bad = [w for w in warn if p.name in w]
            ok_col, bad_col = ("#7fcf8a", "#e0b060") if theme.is_dark() else ("#2f7d45", "#9a5d00")
            t.setItem(r, 9, _item(("! " + tr("warning")) if bad else "√ OK", color=bad_col if bad else ok_col,
                                  tip="\n".join(bad)))
            self._row_data["parts"].append(r)
        summary = None
        if parts:
            summary = ["Total analysed", str(n_calc), str(n_calc), f"{len(used)} mat.", "", "",
                       fmt(vol_sum / 1000.0, 4), fmt(mass_sum, 4), f"weight {fmt(mass_sum * G_MM / 1000.0, 3)} N", ""]
            geo = getattr(m, "step_path", None)
            self.note.setText(" · ".join(x for x in (
                geo.name if geo is not None else "", f"{len(parts)} parts",
                ("! " + tr("geometry warnings")) + f": {len(warn)}" if warn else "√ " + tr("Geometry check without warnings"),
                tr("Show = only in 3D · Analyse = the part goes into the analysis")) if x))
        self._finish(t, summary)

    def _on_part_item(self, it):
        if it.column() not in (1, 2):
            return
        m = self.main
        data = self._row_data.get("parts", [])
        if not 0 <= it.row() < len(data) or m.study is None:
            return
        name = m.parts[data[it.row()]].name
        on = it.checkState() == Qt.Checked
        if it.column() == 1:
            m.set_hidden([name], not on)
        else:
            m.set_ignored([name], not on)

    def _set_material(self, part: str, name: str):
        m = self.main
        st = m.study
        if st is None or name not in st.materials or st.part_materials.get(part) == name:
            return
        st.part_materials[part] = name
        m.changed()

    # ------------------------------------------------------------------ loads and constraints
    def _fill_loads(self):
        m = self.main
        st = m.study
        t = self.tables["loads"]
        cases = list(st.load_cases) if st is not None else []
        t.setColumnCount(7 + len(cases))
        t.setHorizontalHeaderLabels([tr(x) for x in ("Name", "Type", "Value", "Fx [N]", "Fy [N]", "Fz [N]", "|F| [N]")]
                                    + [lc.name for lc in cases])
        cons = list(st.constraints) if st is not None else []
        rows = [("con", k, c) for k, c in enumerate(cons)]
        rows += [("load", (ci, li), l) for ci, lc in enumerate(cases) for li, l in enumerate(lc.loads)]
        t = self._start("loads", len(rows))
        blue, red = "#2f6fd0", "#c9372c"
        if theme.is_dark():
            blue, red = "#7fa8f0", "#ff7a6a"
        from fea_core.loads_text import constraint_text, load_text
        ro = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        for r, (kind, key, obj) in enumerate(rows):
            cls = type(obj).__name__
            if kind == "con":
                name, druh, col = getattr(obj, "selection", "") or "–", CON_KIND.get(cls, cls), blue
                try:
                    value = constraint_text(obj, st, num=fmt)
                except Exception:  # noqa: BLE001 – the description must not crash the table
                    value = m.describe_constraint(obj)
            else:
                name = getattr(obj, "selection", None) or LOAD_KIND.get(cls, cls)
                druh, col = LOAD_KIND.get(cls, getattr(obj, "kind", cls)), red
                try:
                    value = load_text(obj, st, num=fmt)
                except Exception:  # noqa: BLE001
                    value = m.describe_load(obj)
            for c, v in enumerate(("● " + name, druh, value)):
                it = _item(v, color=col if c == 0 else None)
                it.setFlags(ro)
                t.setItem(r, c, it)
            vec = getattr(obj, "vector", None) if kind == "load" and getattr(obj, "kind", "") == "force" else None
            editable = vec is not None and not getattr(obj, "csys", None)
            for c in range(3):
                it = _item(fmt(float(vec[c]), 6) if vec is not None else "", right=True,
                           tip="Double-click = edit the force component (global axes)" if editable else "")
                it.setFlags(ro | Qt.ItemIsEditable if editable else ro)
                t.setItem(r, 3 + c, it)
            it = _item(fmt(float(np.linalg.norm(vec)), 5) if vec is not None else "", right=True)
            it.setFlags(ro)
            t.setItem(r, 6, it)
            for ci in range(len(cases)):
                on = kind == "con" or key[0] == ci
                it = QTableWidgetItem("" if kind == "load" else "√")
                it.setTextAlignment(Qt.AlignCenter)
                if kind == "load":
                    it.setFlags(ro | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked if on else Qt.Unchecked)
                    it.setToolTip(tr("Check = copy the load into the case, uncheck in its own case = delete"))
                else:
                    it.setFlags(ro)
                    it.setToolTip(tr("the support applies in all load cases"))
                t.setItem(r, 7 + ci, it)
            self._row_data["loads"].append((kind, key))
        summary = None
        if cases:
            summary = ["Number of loads in the case", "", "", "", "", "", ""] + [str(len(lc.loads)) for lc in cases]
            self.note.setText(f"{len(cons)} supports · {len(cases)} load cases · " +
                              tr("Matrix: row = load, column = load case") + " · " +
                              tr("double-click Fx / Fy / Fz to edit the force"))
        self._finish(t, summary)
        hh = t.horizontalHeader()                       # state columns narrow, Value stretches
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(2, hh.ResizeMode.Stretch)

    def _on_load_item(self, it):
        m = self.main
        st = m.study
        data = self._row_data.get("loads", [])
        if st is None or not 0 <= it.row() < len(data):
            return
        kind, key = data[it.row()]
        if kind != "load":
            return
        ci, li = key
        ld = st.load_cases[ci].loads[li]
        c = it.column()
        if 3 <= c <= 5:
            try:
                v = float(it.text().replace(",", ".").replace("−", "-").replace(" ", ""))
            except ValueError:
                self.refresh()
                return
            if float(ld.vector[c - 3]) != v:
                ld.vector[c - 3] = v
                m.changed()
        elif c >= 7:
            target = c - 7
            on = it.checkState() == Qt.Checked
            if on and target != ci:
                import copy
                st.load_cases[target].loads.append(copy.deepcopy(ld))
                m.changed()
            elif not on and target == ci:
                self._on_click("loads", it.row())
                m.delete_item(m.load_list, "load")

    # ------------------------------------------------------------------ mesh
    def _fill_mesh(self):
        m = self.main
        st = m.study
        mesh = getattr(m, "mesh", None)
        if mesh is None or st is None:
            self._finish(self._start("mesh", 0))
            return
        tp = np.asarray(mesh.tet_part)
        n_el = len(tp)
        names = list(mesh.part_names)
        t = self._start("mesh", len(names))
        for pi, name in enumerate(names):
            sel = tp == pi
            ne = int(sel.sum())
            nn = int(len(np.unique(mesh.tets[sel]))) if ne else 0
            size = st.mesh.part_sizes.get(name, st.mesh.size)
            t.setItem(pi, 0, _item(name))
            t.setItem(pi, 1, _item(fmt(size, 3) + (" (custom)" if name in st.mesh.part_sizes else ""), right=True))
            t.setItem(pi, 2, _item(_int(ne), right=True))
            t.setItem(pi, 3, _item(_int(nn), right=True))
            t.setItem(pi, 4, _item(fmt(100.0 * ne / n_el, 3) if n_el else "–", right=True))
            self._row_data["mesh"].append(pi)
        rep = mesh.report or {}
        sj = rep.get("min_scaled_jacobian")
        summary = ["Total", f"global {fmt(st.mesh.size, 3)}", _int(n_el), _int(len(mesh.nodes)),
                   f"min. SJ {fmt(sj, 3)}" if sj is not None else ""]
        self.note.setText(f"elements {mesh.element_type} · local refinements: {len(st.mesh.local)}")
        self._finish(t, summary)

    # ------------------------------------------------------------------ results
    def _fill_results(self):
        m = self.main
        rows = step_summaries(m)
        t = self._start("results", len(rows))
        req = required_safety(m.study)
        warn = theme.colors()["muted"]
        ok_col, bad_col = ("#7fcf8a", "#ff7a6a") if theme.is_dark() else ("#2f7d45", "#c9372c")
        for r, d in enumerate(rows):
            t.setItem(r, 0, _item(d["name"]))
            t.setItem(r, 1, _item(fmt(d["vmax"], 4), right=True))
            if d["real"] is not None:
                t.setItem(r, 2, _item(fmt(d["real"], 4), right=True, bold=True))
                t.setItem(r, 3, _item(", ".join(d["parts"]) or "–"))
                k = d["k"]
                col = None if k is None else (ok_col if k >= req else bad_col)
                t.setItem(r, 4, _item(fmt(k, 3) if k is not None else "–", right=True, color=col, bold=col is not None,
                                      tip=f"Part Re / stress at the maximum outside singularities; required ≥ {fmt(req, 3)}"))
            else:
                for c in (2, 3, 4):
                    t.setItem(r, c, _item("–", right=c != 3, color=warn))
            t.setItem(r, 5, _item(fmt(d["umax"], 4), right=True))
            t.setItem(r, 6, _item(fmt(d["reaction"], 4) if d["reaction"] is not None else "–", right=True))
            self._row_data["results"].append(d["index"])
        if rows:
            self.note.setText(f"required safety factor ≥ {fmt(req, 3)} · singularities (sharp edges, fixed supports) are not counted in “outside singularities” · a click shows the case")
        self._finish(t)

    # ------------------------------------------------------------------ teplo
    def _fill_thermal(self):
        st = self.main.study
        cases = list(st.thermal.cases) if st is not None else []
        t = self.tables["thermal"]
        t.setColumnCount(3 + len(cases))
        from .thermal_panel import KIND_SHORT, describe_bc, describe_case
        t.setHorizontalHeaderLabels(["Face / part", "Type", "Value"] + [f"{c.name} ({describe_case(c)})" for c in cases])
        rows = [(ci, bc) for ci, c in enumerate(cases) for bc in c.loads]
        rows += [(-1, cc) for cc in (st.thermal.conductances if st is not None else [])]   # interfaces: all cases
        t = self._start("thermal", len(rows))
        for r, (ci, bc) in enumerate(rows):
            text = describe_bc(bc)
            name = getattr(bc, "selection", None) or getattr(bc, "part", "") or "–"
            if isinstance(name, list):
                name = ", ".join(name)
            if ci < 0:
                name = f"{bc.selection} – {bc.master}"
            value = text.split(": ", 1)[1] if ": " in text else text
            t.setItem(r, 0, _item(name))
            t.setItem(r, 1, _item(KIND_SHORT.get(getattr(bc, "kind", ""), getattr(bc, "kind", ""))))
            t.setItem(r, 2, _item(value))
            for cj in range(len(cases)):
                it = _item("√" if cj == ci or ci < 0 else "")
                it.setTextAlignment(Qt.AlignCenter)
                t.setItem(r, 3 + cj, it)
            self._row_data["thermal"].append(ci)
        summary = None
        if cases:
            summary = ["Conditions in the case", "", ""] + [str(len(c.loads)) for c in cases]
            ref = getattr(st.thermal, "reference_temperature", None)
            self.note.setText(f"{len(cases)} thermal cases · reference temperature {fmt(ref, 4)} °C · click selects the case"
                              if ref is not None else f"{len(cases)} thermal cases")
        self._finish(t, summary)
        hh = t.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(2, hh.ResizeMode.Stretch)

    # ------------------------------------------------------------------ kontakty
    def _fill_contacts(self):
        st = self.main.study
        pairs = list(st.contacts) if st is not None else []
        t = self._start("contacts", len(pairs))
        from fea_core.contacts.model import CONTACT_LABELS
        for r, p in enumerate(pairs):
            mu = getattr(p, "mu", None)
            clr = getattr(p, "clearance", None)
            t.setItem(r, 0, _item(p.name))
            t.setItem(r, 1, _item(p.a))
            t.setItem(r, 2, _item(p.b))
            t.setItem(r, 3, _item(CONTACT_LABELS.get(p.kind, p.kind)))
            if mu is not None:
                mu_text = fmt(mu, 3)
            elif p.kind == "frictional":
                mu_text = p.materials or "from the friction library"
            elif p.kind == "rough":
                mu_text = "no slip"
            else:
                mu_text = "–"
            t.setItem(r, 4, _item(mu_text, right=mu is not None))
            t.setItem(r, 5, _item(fmt(clr, 4) if clr is not None else "–", right=clr is not None))
            if not p.enabled:
                for c in range(6):
                    t.item(r, c).setForeground(QColor(theme.colors()["muted"]))
            self._row_data["contacts"].append(r)
        if pairs:
            from fea_core.contacts.model import NONLINEAR_TYPES
            nl = sum(p.enabled and p.kind in NONLINEAR_TYPES for p in pairs)
            self.note.setText(f"{len(pairs)} pairs · nonlinear {nl} · μ is an estimate, K and λ are penalties")
        self._finish(t)

    # ------------------------------------------------------------------ frames
    def _fill_frame(self):
        st = self.main.study
        fr = st.frame if st is not None else None
        members = list(fr.members.values()) if fr is not None else []
        t = self._start("frame", len(members))
        total = 0.0
        for r, mb in enumerate(members):
            a, b = fr.nodes.get(mb.start), fr.nodes.get(mb.end)
            length = float(np.linalg.norm(np.subtract(b.xyz, a.xyz))) if a is not None and b is not None else None
            if length is not None and mb.through is None:
                total += length
            rel = [n for n, rs in (("start", mb.releases_start), ("end", mb.releases_end)) if any(rs)]
            kind = "truss" if getattr(mb, "kind", "") == "truss" else ("arc" if mb.through else "beam")
            t.setItem(r, 0, _item(mb.name))
            t.setItem(r, 1, _item(f"{mb.start} – {mb.end}"))
            t.setItem(r, 2, _item(mb.section))
            t.setItem(r, 3, _item(mb.material or "default"))
            t.setItem(r, 4, _item(fmt(length, 5) + (" (chord)" if mb.through else "") if length is not None else "–",
                                  right=True))
            t.setItem(r, 5, _item(kind))
            t.setItem(r, 6, _item(", ".join(rel) or "–"))
            self._row_data["frame"].append(r)
        summary = None
        if members:
            summary = [f"{len(members)} members", f"{len(fr.nodes)} nodes", f"{len(fr.sections)} cross-sections", "",
                       fmt(total, 5), "", f"supports: {len(fr.supports)}"]
            self.note.setText("total length of straight members")
        self._finish(t, summary)

    # ------------------------------------------------------------------ mechanismus
    def _fill_motion(self):
        st = self.main.study
        mech = st.mechanism if st is not None else None
        rows = []
        if mech is not None:
            from fea_core.motion.model import JOINT_LABELS
            for b in mech.bodies:
                rows.append((b.name, "body", "ground (fixed)" if b.ground else
                             (f"mass {fmt(b.mass, 4)} kg" if b.mass else "mass from CAD"), ", ".join(b.parts)))
            for j in mech.joints:
                rows.append((j.name, "hinge", JOINT_LABELS.get(j.kind, j.kind), f"{j.body_a} – {j.body_b}"))
            for d in mech.drivers:
                rows.append((f"driver {d.joint}", "drive", getattr(d, "quantity", ""), d.joint))
        t = self._start("motion", len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                t.setItem(r, c, _item(str(val)))
            self._row_data["motion"].append(r)
        if mech is not None and rows:
            self.note.setText(f"{len(mech.bodies)} bodies · {len(mech.joints)} joints · {len(mech.drivers)} drivers")
        self._finish(t)

    # ------------------------------------------------------------------ added masses
    def _fill_dynamics(self):
        st = self.main.study
        masses = list(st.masses) if st is not None else []
        t = self._start("dynamics", len(masses))
        total = 0.0
        for r, ms in enumerate(masses):
            if getattr(ms, "kind", "") == "surface_mass":
                value, druh = f"{fmt(ms.mass_per_area, 4)} kg/m²", "surface mass"
            else:
                total += float(getattr(ms, "mass", 0.0) or 0.0)
                value, druh = f"{fmt(ms.mass, 4)} kg", "point mass"
            t.setItem(r, 0, _item(getattr(ms, "name", "") or ms.selection))
            t.setItem(r, 1, _item(druh))
            t.setItem(r, 2, _item(value, right=True))
            t.setItem(r, 3, _item(ms.selection))
            self._row_data["dynamics"].append(r)
        summary = None
        if masses:
            summary = ["Point masses total", "", f"{fmt(total, 4)} kg", ""]
        self._finish(t, summary)

    # ------------------------------------------------------------------ materials, geometry, coordinate systems
    def _fill_materials(self):
        st = self.main.study
        mats = st.materials if st is not None else {}
        counts: dict[str, int] = {}
        for p in getattr(self.main, "parts", []) or []:
            try:
                k = st.material_of(p.name).name
                counts[k] = counts.get(k, 0) + 1
            except (KeyError, AttributeError):
                pass
        t = self._start("materials", len(mats))
        for r, (name, mt) in enumerate(mats.items()):
            vals = [name, fmt(mt.E, 6) if mt.E else "–", fmt(mt.nu, 3) if mt.nu is not None else "–",
                    fmt(mt.rho * 1e12, 5) if mt.rho else "–", fmt(getattr(mt, "Re", None), 4) if getattr(mt, "Re", None) else "–",
                    fmt(getattr(mt, "Rm", None), 4) if getattr(mt, "Rm", None) else "–", str(counts.get(name, 0)),
                    "3D printing" if mt.is_print else ("orthotropic" if getattr(mt, "ortho", None) else "isotropic")]
            for c, v in enumerate(vals):
                t.setItem(r, c, _item(v, right=0 < c < 7))
            self._row_data["materials"].append(r)
        self._finish(t)

    def _fill_geometry(self):
        warn = list(getattr(self.main, "geometry_warnings", None) or [])
        t = self._start("geometry", len(warn))
        for r, w in enumerate(warn):
            t.setItem(r, 0, _item(str(r + 1), right=True))
            t.setItem(r, 1, _item(w))
            self._row_data["geometry"].append(r)
        self._finish(t)

    def _fill_csys(self):
        st = self.main.study
        cs = st.coordinate_systems if st is not None else {}
        t = self._start("csys", len(cs))
        v3 = lambda v: "(" + "; ".join(fmt(float(x), 4) for x in v) + ")" if v is not None else "–"  # noqa: E731
        for r, (name, c) in enumerate(cs.items()):
            for k, v in enumerate((name, "cylindrical" if c.kind == "cylindrical" else "Cartesian", v3(c.origin),
                                   v3(c.z_axis), v3(getattr(c, "x_axis", None)))):
                t.setItem(r, k, _item(v))
            self._row_data["csys"].append(r)
        self._finish(t)

    # ------------------------------------------------------------------ spoje
    def _joint_checks(self):
        a = getattr(getattr(self.main, "joints_panel", None), "assessment", None)
        return {c.name: c for c in a.all()} if a is not None else {}

    def _util_item(self, c):
        if c is None:
            return _item("–", right=True)
        u = c.utilization
        ok_col, warn_col, bad_col = (("#7fcf8a", "#e0b060", "#ff7a6a") if theme.is_dark() else
                                     ("#2f7d45", "#9a5d00", "#c9372c"))
        col = ok_col if u <= 0.9 else warn_col if u <= 1.0 else bad_col
        return _item(fmt(u, 3), right=True, color=col, bold=True, tip=c.governing)

    def _fill_bolts(self):
        st = self.main.study
        bolts = list(st.bolts) if st is not None else []
        chk = self._joint_checks()
        t = self._start("bolts", len(bolts))
        conn = {"nut": "with nut", "tapped": "into thread", "fitted": "fitted"}
        jp = getattr(self.main, "joints_panel", None)
        for r, b in enumerate(bolts):
            t.setItem(r, 0, _item(b.name))
            if jp is not None:
                for c, src, cur in ((1, jp.b_size, f"M{b.size:g}"), (2, jp.b_class, b.strength),
                                    (4, jp.b_mode, None)):
                    cb = QComboBox()
                    cb.setProperty("_i18n_done", c != 4)          # sizes and classes are not translated
                    for i in range(src.count()):
                        cb.addItem(src.itemText(i), src.itemData(i))
                    k = cb.findData(b.preload_mode) if c == 4 else cb.findText(cur)
                    if k < 0 and cur:
                        cb.addItem(cur, None)
                        k = cb.count() - 1
                    cb.setCurrentIndex(max(k, 0))
                    cb.setToolTip(src.toolTip())
                    cb.currentIndexChanged.connect(lambda _i, row=r, col=c, box=cb: self._edit_bolt(row, col, box))
                    t.setCellWidget(r, c, cb)
            else:
                t.setItem(r, 1, _item(b.designation))
                t.setItem(r, 2, _item(b.strength))
            t.setItem(r, 3, _item(conn.get(b.connection, b.connection)))
            if jp is None:
                t.setItem(r, 4, _item(b.preload_mode))
            vals = [b.head_part or "–", b.nut_part or "–", str(len(b.washers))]
            for c, v in enumerate(vals):
                t.setItem(r, 5 + c, _item(v))
            c = chk.get(b.name)
            t.setItem(r, 8, self._util_item(c))
            vdi = c.data.get("vdi", {}) if c is not None else {}
            for k, (key, dec) in enumerate((("F_SA", 5), ("F_Q", 5), ("phi", 3))):
                t.setItem(r, 9 + k, _item(fmt(vdi.get(key), dec) if c is not None else "–", right=True))
            t.setItem(r, 12, _item(c.governing if c is not None else ("not computed" if not b.enabled else "")))
            self._row_data["bolts"].append(r)
        if bolts:
            self.note.setText(tr("Bolt = U1 element with preload · Auto = min(F_M,zul; pressure under head)"))
        self._finish(t)

    def _edit_bolt(self, row: int, col: int, box: QComboBox):
        """Editing a bolt in the table = the form of the Joints panel + Save bolt changes (the same effect on the mesh and results)."""
        jp = self.main.joints_panel
        if not 0 <= row < jp.bolt_list.count():
            return
        jp.bolt_list.setCurrentRow(row)
        target = {1: jp.b_size, 2: jp.b_class, 4: jp.b_mode}[col]
        k = target.findData(box.currentData()) if box.currentData() is not None else target.findText(box.currentText())
        if k < 0:
            return
        target.setCurrentIndex(k)
        jp.save_bolt()

    def _fill_welds(self):
        st = self.main.study
        welds = list(st.welds) if st is not None else []
        chk = self._joint_checks()
        t = self._start("welds", len(welds))
        for r, w in enumerate(welds):
            vals = [w.name, w.part, fmt(w.a, 3) if w.a else "from geometry", fmt(w.length, 4) if w.length else "from geometry",
                    f"{fmt(w.fat_sigma, 3)} / {fmt(w.fat_tau, 3)}", w.sn_curve]
            for c, v in enumerate(vals):
                t.setItem(r, c, _item(v))
            c = chk.get(w.name)
            t.setItem(r, 6, self._util_item(c))
            t.setItem(r, 7, _item(c.governing if c is not None else ""))
            self._row_data["welds"].append(r)
        self._finish(t)

    def _fill_bonded(self):
        mesh = getattr(self.main, "mesh", None)
        pairs = []
        if mesh is not None:
            cache = getattr(mesh, "_gui_bonded", None)
            if cache is None:
                from fea_core.contacts import bonded_pairs
                cache = mesh._gui_bonded = bonded_pairs(mesh)
            pairs = cache
        t = self._start("bonded", len(pairs))
        for r, d in enumerate(pairs):
            t.setItem(r, 0, _item(d["a"]))
            t.setItem(r, 1, _item(d["b"]))
            t.setItem(r, 2, _item(fmt(d["area"], 4), right=True))
            t.setItem(r, 3, _item("shared mesh"))
            self._row_data["bonded"].append(r)
        self._finish(t)

    def _fill_ignored(self):
        st = self.main.study
        ign = list(st.ignored_parts) if st is not None else []
        t = self._start("ignored", len(ign))
        for r, n in enumerate(ign):
            t.setItem(r, 0, _item(n))
            t.setItem(r, 1, _item("excluded from the analysis (Model → Analyse)"))
            self._row_data["ignored"].append(r)
        self._finish(t)

    def _fill_jutil(self):
        chk = self._joint_checks()
        t = self._start("jutil", len(chk))
        for r, c in enumerate(chk.values()):
            t.setItem(r, 0, _item(c.name))
            t.setItem(r, 1, _item(c.label))
            t.setItem(r, 2, self._util_item(c))
            t.setItem(r, 3, _item(c.governing))
            self._row_data["jutil"].append(r)
        self._finish(t)

    # ------------------------------------------------------------------ states, combinations, refinement
    def _fill_cases(self):
        st = self.main.study
        cases = list(st.load_cases) if st is not None else []
        t = self._start("cases", len(cases))
        for r, lc in enumerate(cases):
            kinds = sorted({LOAD_KIND.get(type(x).__name__, type(x).__name__) for x in lc.loads})
            t.setItem(r, 0, _item(lc.name))
            t.setItem(r, 1, _item(str(len(lc.loads)), right=True))
            t.setItem(r, 2, _item(", ".join(tr(k) for k in kinds) or "–"))
            t.setItem(r, 3, _item(str(len(st.constraints)), right=True))
            self._row_data["cases"].append(r)
        self._finish(t)

    def _fill_combos(self):
        st = self.main.study
        combos = list(getattr(st, "combinations", None) or []) if st is not None else []
        t = self._start("combos", len(combos))
        for r, c in enumerate(combos):
            f = ", ".join(f"{fmt(v, 3)}·{k}" for k, v in (c.factors or {}).items())
            t.setItem(r, 0, _item(c.name))
            t.setItem(r, 1, _item(f or "–"))
            t.setItem(r, 2, _item(getattr(c, "design_situation", "") or "–"))
            t.setItem(r, 3, _item("yes" if getattr(c, "envelope", False) else "ne"))
            self._row_data["combos"].append(r)
        self._finish(t)

    def _fill_refine(self):
        m = self.main
        st = m.study
        rows = []
        if st is not None:
            for it in (m._items("refine") if hasattr(m, "_items") else []):
                sel = it.get("selection") if isinstance(it, dict) else getattr(it, "selection", None)
                size = it.get("size") if isinstance(it, dict) else getattr(it, "size", None)
                rows.append((sel or "–", size, "faces"))
            for loc in getattr(st.mesh, "local", []) or []:
                c = loc.get("center")
                rows.append(("(" + "; ".join(fmt(float(x), 4) for x in c) + f") R {fmt(loc.get('radius'), 3)}"
                             if c is not None else "–", loc.get("size"), "sphere"))
            for name, size in (st.mesh.part_sizes or {}).items():
                rows.append((name, size, "part"))
        t = self._start("refine", len(rows))
        for r, (a, size, kind) in enumerate(rows):
            t.setItem(r, 0, _item(str(a)))
            t.setItem(r, 1, _item(fmt(size, 3) if size is not None else "–", right=True))
            t.setItem(r, 2, _item(kind))
            self._row_data["refine"].append(r)
        self._finish(t)

    # ------------------------------------------------------------------ results: maximum places, reactions
    def _step_index(self):
        data = self.main.r_step.currentData() if hasattr(self.main, "r_step") else None
        if isinstance(data, str) and data.startswith("step:"):
            return int(data.split(":")[1])
        res = self.main.results
        steps = [i for i, s in enumerate(res.steps) if len(s.mises)] if res is not None else []
        return steps[0] if steps else None

    def _fill_hotspots(self):
        m = self.main
        i = self._step_index()
        d = m.result_hotspots("step", i, "mises") if i is not None else None
        hs = (d or {}).get("hotspots", [])
        re_node = None
        if hs:
            try:
                re_node = m.node_re()
            except Exception:  # noqa: BLE001
                re_node = None
        t = self._start("hotspots", len(hs))
        sing_col = theme.colors()["muted"]
        for r, h in enumerate(hs):
            cls = h.get("classification", "")
            real = cls not in ("singularity", "affected by boundary condition")
            col = None if real else sing_col
            k = None
            if real and re_node is not None:
                k = float(re_node[h["node"] - 1] / max(h["mises_MPa"], 1e-9))
            t.setItem(r, 0, _item(str(h.get("rank", r + 1)), right=True, color=col))
            t.setItem(r, 1, _item(fmt(h["mises_MPa"], 4), right=True, color=col, bold=real))
            t.setItem(r, 2, _item(cls, color=col, tip=h.get("recommendation", "")))
            t.setItem(r, 3, _item(", ".join(h.get("parts", [])), color=col))
            t.setItem(r, 4, _item(" · ".join(fmt(float(x), 4) for x in h["xyz_mm"]), color=col))
            t.setItem(r, 5, _item(fmt(k, 3) if k is not None and np.isfinite(k) else "–", right=True, color=col))
            self._row_data["hotspots"].append(r)
        if hs:
            self.note.setText(tr("Singularities (fixed edges, sharp inner corners) are not assessed – the value grows with mesh refinement."))
        self._finish(t)

    def _fill_psafety(self):
        """Safety to Re by parts in the displayed state: max. σ of the part outside singularities (maximum places classified as
        singularity / boundary condition are left out – as on the Assessment card), bar k / required safety.
        """
        m = self.main
        res = m.results
        i = self._step_index()
        rows = []
        if res is not None and i is not None and len(res.mesh.tets):
            mesh = res.mesh
            vm = res.steps[i].mises
            d = m.result_hotspots("step", i, "mises")
            sing = {h["node"] - 1 for h in (d or {}).get("hotspots", [])
                    if h.get("classification", "") in ("singularity", "affected by boundary condition")}
            excl = m.study.excluded_parts()
            for pi, name in enumerate(mesh.part_names):
                if name in excl:
                    continue
                nodes = np.unique(mesh.tets[mesh.tet_part == pi].ravel())
                if not len(nodes):
                    continue
                keep = nodes[~np.isin(nodes, list(sing))] if sing else nodes
                note = "" if len(keep) == len(nodes) else "no singularities"
                use = keep if len(keep) else nodes
                smax = float(vm[use].max())
                try:
                    re = m.study.material_of(name).Re
                except KeyError:
                    re = None
                k = re / max(smax, 1e-9) if re else None
                rows.append((name, smax, re, k, note))
        req = required_safety(m.study)
        t = self._start("psafety", len(rows))
        ok_col, bad_col = ("#7fcf8a", "#ff7a6a") if theme.is_dark() else ("#2f7d45", "#c9372c")
        for r, (name, smax, re, k, note) in enumerate(rows):
            col = None if k is None else (ok_col if k >= req else bad_col)
            t.setItem(r, 0, _item(name))
            t.setItem(r, 1, _item(fmt(smax, 4), right=True))
            t.setItem(r, 2, _item(fmt(re, 4) if re else "–", right=True))
            t.setItem(r, 3, _item(fmt(k, 3) if k is not None else "–", right=True, color=col, bold=col is not None))
            n = 0 if k is None else int(round(min(k / req, 2.0) * 10))       # 10 units = required safety
            bar = ("■" * min(n, 10) + "|" + "■" * max(n - 10, 0)) if k is not None else ""
            t.setItem(r, 4, _item(bar, color=col, tip=f"k / {fmt(req, 3)} = {fmt(k / req, 3)}" if k is not None else ""))
            t.setItem(r, 5, _item(note))
            self._row_data["psafety"].append(r)
        if rows:
            self.note.setText(tr("Line | = required safety") + f" {fmt(req, 3)} · " +
                              tr("parts without a yield strength (plastics, printing) are not assessed"))
        self._finish(t)

    def _fill_reactions(self):
        res = self.main.results
        i = self._step_index()
        s = res.steps[i] if res is not None and i is not None else None
        reac = dict(s.reactions) if s is not None else {}
        t = self._start("reactions", len(reac))
        for r, (name, R) in enumerate(reac.items()):
            R = np.asarray(R, float)
            t.setItem(r, 0, _item(name))
            for c in range(3):
                t.setItem(r, 1 + c, _item(fmt(float(R[c]), 4), right=True))
            t.setItem(r, 4, _item(fmt(float(np.linalg.norm(R)), 4), right=True, bold=True))
            self._row_data["reactions"].append(r)
        summary = None
        if reac and s is not None:
            rt = (s.equilibrium or {}).get("reaction_total")
            imb = (s.equilibrium or {}).get("imbalance")
            if rt is not None:
                summary = ["Total"] + [fmt(float(v), 4) for v in rt] + [fmt(float(np.linalg.norm(rt)), 4)]
            if imb is not None:
                from .widgets import sci
                self.note.setText(f"{s.name} · " + tr("equilibrium") + f" {sci(imb)}")
        self._finish(t, summary)

    # ------------------------------------------------------------------ click on a row
    def _on_click(self, g: str, row: int):
        m = self.main
        data = self._row_data.get(g, [])
        if not 0 <= row < len(data) or m.busy():
            return
        if g == "parts":
            pi = data[row]
            m.view.set_selected_parts([pi])
            if hasattr(m, "model_table") and pi < m.model_table.rowCount():
                m.model_table.selectRow(pi)
        elif g == "loads":
            kind, key = data[row]
            if kind == "con":
                m.set_stage(m.STAGE["Constraints"])
                m.con_list.setCurrentRow(key)
            else:
                ci, li = key
                m.set_stage(m.STAGE["Loads"])
                m.lc_combo.setCurrentIndex(ci)
                m.load_list.setCurrentRow(li)
        elif g == "results":
            k = m.r_step.findData(f"step:{data[row]}")
            if k >= 0:
                m.r_step.setCurrentIndex(k)
        elif g == "contacts":
            tab = getattr(getattr(m, "contact_panel", None), "table", None)
            if tab is not None and data[row] < tab.rowCount():
                tab.selectRow(data[row])
        elif g == "thermal":
            combo = getattr(m.thermal_panel, "case_combo", None)
            if combo is not None and 0 <= data[row] < combo.count():
                combo.setCurrentIndex(data[row])
