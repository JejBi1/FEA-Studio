"""Heat panel: thermal states (steady / transient heat conduction), their boundary conditions on faces
(selection as for constraints and loads), heat sources in the parts and the reference / initial temperature.

The GUI only edits Study.thermal; the calculation is done by fea_core.thermal (it runs with the whole study in the Solve step).
Thermal stress = the load "Temperature (thermal stress)" in a mechanical load state (Loads step).
"""
from __future__ import annotations

from html import escape
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QMessageBox, QPushButton, QSpinBox, QWidget)

from fea_core.study import ThermalLoad
from fea_core.thermal.model import (ThermalCase, FixedTemperature, HeatFlux, HeatFlow, Convection, Radiation,
                                    HeatGeneration, CavityRadiation, ContactConductance, ABSOLUTE_ZERO)

from .thermal_extras import (ConductanceBox, cavity_from_values, cavity_names, cavity_values, describe_cavity,
                             is_cavity_bc, sync_cavity)
from .widgets import Section, dspin, fit_rows, fmt_dash as fmt, help_tip, panel_header, scroll_body, set_placeholder

APP = "FEA Studio"
# kind of boundary condition: (description in the menu, prefix of the selection name, quantity, default value, colour in 3D)
BC_TYPES = {
    "fixed_temperature": ("Prescribed temperature", "Temperature", "T [°C]", 100.0, (230, 90, 20)),
    "heat_flux": ("Heat flux [W/m²]", "Flux", "q [W/m²]", 1000.0, (215, 30, 110)),
    "heat_flow": ("Heat power [W]", "Power", "Q [W]", 10.0, (215, 30, 110)),
    "convection": ("Heat transfer – convection", "Convection", "h [W/(m²·K)]", 10.0, (20, 150, 210)),
    "radiation": ("Radiation to surroundings", "Radiation", "ε [–]", 0.9, (215, 170, 0)),
    "cavity_radiation": ("Cavity radiation (between faces)", "Cavity", "ε [–]", 0.9, (190, 140, 0)),
    "heat_generation": ("Heat generation in part", None, "P [W]", 10.0, (160, 60, 200)),
}
# conditions without a form (older files: contact conductance in a case) – shown, edited elsewhere
OTHER_COLORS = {"contact_conductance": (150, 95, 40)}
# short names of the kinds for the flux table in the results
KIND_SHORT = {"fixed_temperature": "prescribed temperature", "heat_flux": "heat flux", "heat_flow": "heat power",
              "convection": "convection", "radiation": "radiation", "heat_generation": "heat generation",
              "cavity_radiation": "cavity radiation", "contact_conductance": "contact conductance"}
HINTS = {
    "fixed_temperature": "The face has a constant temperature (contact with a heat sink, bath, heater…).",
    "heat_flux": "Flux through the face: positive = heat enters the body (heating), negative = removal.",
    "heat_flow": "Total power distributed uniformly over the area of the selected faces; positive = into the body.",
    "convection": "q = h·(T ambient − T). Typical values: natural air convection 5–25, forced 25–250, water 500–10 000 W/(m²·K).",
    "radiation": "Radiation to the surroundings without mutual irradiation of faces (the problem is nonlinear). Typical ε: polished aluminum ~0.05, oxidized steel and anodized ~0.8, black paint ~0.95.",
    "cavity_radiation": "Faces with the same cavity name radiate to each other (view factors by CalculiX, nonlinear). Add each face group with its own ε. Closed cavity = fully enclosed (no heat leaves); open = the opening sees the ambient temperature.",
    "heat_generation": "Power uniformly in the part volume (winding losses, resistance, electronics). Select the part in the list or by clicking its face in 3D.",
}
# label of the value field by the kind of condition (changes with the type, on_bc_type)
VALUE_TIPS = {
    "fixed_temperature": "Temperature of the selected faces [°C] (must be above absolute zero −273.15 °C)",
    "heat_flux": "Heat flux density q [W/m²]: positive = heat enters the body, negative = removal.\nTotal power = q × area of the selected faces.",
    "heat_flow": "Total heat power Q [W] distributed uniformly over the area of the selected faces;\npositive = into the body, negative = removal",
    "convection": "Heat transfer coefficient h [W/(m²·K)] > 0. Typical values: natural air convection 5–25,\nforced 25–250, water 500–10 000",
    "radiation": "Surface emissivity ε in the range (0; 1]: polished aluminum ~0.05, oxidized steel and anodized ~0.8,\nblack paint ~0.95",
    "cavity_radiation": "Emissivity ε of the selected cavity faces in the range (0; 1]; other faces of the same cavity\ncan have their own ε (add them as separate conditions with the same cavity name)",
    "heat_generation": "Heat power P [W] generated uniformly in the volume of the selected part (losses, resistance…)",
}


VALUE_SHORT = {
    "fixed_temperature": "Temperature of the selected faces [°C]",
    "heat_flux": "Heat flux density q [W/m²] (positive = into the body)",
    "heat_flow": "Total heat power Q [W] (positive = into the body)",
    "convection": "Heat transfer coefficient h [W/(m²·K)]",
    "radiation": "Surface emissivity ε in the interval (0; 1]",
    "cavity_radiation": "Emissivity ε of the selected cavity faces (0; 1]",
    "heat_generation": "Heat power P [W] in the volume of the selected part",
}


def describe_bc(bc) -> str:
    if is_cavity_bc(bc):
        return describe_cavity(bc)
    if isinstance(bc, ContactConductance):
        return f"{bc.selection}: contact conductance h_c = {fmt(bc.conductance)} W/(m²·K) to {bc.master}"
    if isinstance(bc, FixedTemperature):
        return f"{bc.selection}: T = {fmt(bc.value)} °C"
    if isinstance(bc, HeatFlux):
        return f"{bc.selection}: q = {fmt(bc.value)} W/m²"
    if isinstance(bc, HeatFlow):
        return f"{bc.selection}: Q = {fmt(bc.value)} W"
    if isinstance(bc, Convection):
        return f"{bc.selection}: h = {fmt(bc.h)} W/(m²·K), T ambient {fmt(bc.ambient)} °C"
    if isinstance(bc, Radiation):
        return f"{bc.selection}: radiation ε = {fmt(bc.emissivity, 3)}, T ambient {fmt(bc.ambient)} °C"
    if isinstance(bc, HeatGeneration):
        return f"Heat generation – {bc.part}: P = {fmt(bc.value)} W"
    return str(bc)


def describe_case(c: ThermalCase) -> str:
    if not c.transient:
        return "steady-state"
    return f"transient {fmt(c.time)} s"


class ThermalPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._filling = False
        _, lay = scroll_body(self)
        lay.addWidget(panel_header("Heat", "Heat conduction in parts (steady-state or transient); select the faces of boundary conditions by clicking in 3D"))
        lbl = QLabel("Thermal stress: in the Loads step, add “Temperature (thermal stress)” to the load case.")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        # ---------------------------------------------------------- thermal states
        h = QHBoxLayout()
        self.case_combo = QComboBox()
        help_tip(self.case_combo, "Thermal state being edited",
                 "Thermal case being edited. Each case is a separate heat conduction analysis; a mechanical\nload case can take its temperature field (“Temperature” load).")
        self.case_combo.currentIndexChanged.connect(lambda _: self.on_case())
        h.addWidget(self.case_combo, 1)
        for txt, slot, tip in (("+", self.add_case, "New thermal case"),
                               ("Rename", self.rename_case,
                                "Rename the selected thermal state (including references from temperature loads)"),
                               ("−", self.delete_case, "Delete the thermal case (including thermal loads that reference it)")):
            b = QPushButton(txt)
            b.clicked.connect(slot)
            b.setToolTip(tip)
            h.addWidget(b)
        lay.addLayout(h)
        g = Section("Thermal case", key="thermal.state", collapsed=False,
                    tip="Sets a steady-state or transient thermal state.",
                    help="<b>Thermal state</b><p>Selects steady-state or transient heat conduction and its time settings. A transient state needs the density and specific heat capacity of the materials.</p>")
        f = QFormLayout(g.body)
        self.transient = QCheckBox("transient (time history from the initial temperature)")
        self.transient.setToolTip("Transient state: temperature over time from the initial temperature")
        self.transient.setWhatsThis("<b>Transient thermal case</b><p>Computes the temperature evolution in time from the initial temperature (section “Study temperatures”) to the given time with a fixed time step. The specified number of evenly spaced snapshots is saved; a mechanical case can take the temperature at any time (linear between snapshots).</p><p>A steady-state case is faster, but the heat must have a way out – prescribed temperature, convection or radiation.</p>")
        self.t_time = dspin(3600.0, 1e-6, 1e9, 3, 60.0, " s")
        self.t_time.setToolTip("Duration of the transient process [s] – time of the last snapshot (e.g. 3600 s = 1 h)")
        self.t_dt = dspin(0.0, 0.0, 1e9, 4, 1.0, " s", special="auto (duration / 100)")
        help_tip(self.t_dt, "Time step [s]; 0 = auto (duration / 100)",
                 "Time step [s]; 0 = auto (duration / 100). The number of steps is rounded up to a multiple of\nthe number of snapshots. A finer step = more accurate rapid changes, but a longer computation.")
        self.t_out = QSpinBox()
        self.t_out.setRange(1, 10000)
        self.t_out.setValue(10)
        help_tip(self.t_out, "Number of stored temperature snapshots over time",
                 "Number of stored temperature snapshots evenly spaced in time (the last = end of duration);\na snapshot at t = 0 s with the initial temperature is added to them.")
        f.addRow(self.transient)
        f.addRow("Duration", self.t_time)
        f.addRow("Time step", self.t_dt)
        f.addRow("Number of snapshots", self.t_out)
        self.t_info = QLabel()
        self.t_info.setWordWrap(True)
        f.addRow(self.t_info)
        self.transient.toggled.connect(self.on_case_settings)
        for w in (self.t_time, self.t_dt, self.t_out):
            self._commit_on_edit(w, self.on_case_settings)
        lay.addWidget(g)

        # ---------------------------------------------------------- boundary conditions
        lay.addWidget(QLabel("Case boundary conditions"))
        self.bc_list = QListWidget()
        set_placeholder(self.bc_list, "No boundary condition yet – faces without a condition are insulated")
        fit_rows(self.bc_list, 3, 8)
        help_tip(self.bc_list, "Boundary conditions of the selected thermal state",
                 "Boundary conditions of the selected thermal case. Click = faces are selected in 3D and the values\nare loaded into the form below (edit with the “Save changes to selected” button).\nFaces without a condition are thermally insulated (zero flux).")
        self.bc_list.currentRowChanged.connect(self.on_bc_row)
        lay.addWidget(self.bc_list)
        g = Section("New boundary condition", key="thermal.boundary", collapsed=False,
                    tip="Creates a boundary condition on the faces selected in 3D.",
                    help="<b>New boundary condition</b><p>Select faces in 3D, choose the type and enter the values. Faces without a condition are perfectly insulated.</p>")
        f = QFormLayout(g.body)
        self.bc_type = QComboBox()
        for k, (txt, *_rest) in BC_TYPES.items():
            self.bc_type.addItem(txt, k)
            self.bc_type.setItemData(self.bc_type.count() - 1, HINTS[k], Qt.ToolTipRole)
        self.bc_type.setToolTip("Type of the thermal boundary condition")
        self.bc_type.setWhatsThis("<b>Thermal boundary conditions</b><p><i>Prescribed temperature</i> holds the temperature of a face (contact with a heat sink, bath). <i>Heat flux</i> [W/m²] and <i>power</i> [W] supply (positive) or remove (negative) heat through a face. <i>Convection</i> q = h·(T ambient − T) and <i>radiation</i> (emissivity ε, no mutual irradiation of faces) remove heat to the surroundings. <i>Heat generation</i> distributes power uniformly over the part volume.</p><p>Select faces in 3D and click “Add from selection”; faces without a condition are perfectly insulated. A steady-state case must have a heat sink (temperature, convection or radiation).</p>")
        self.bc_type.currentIndexChanged.connect(self.on_bc_type)
        f.addRow("Type", self.bc_type)
        self.bc_val = dspin(100.0, -1e12, 1e12, 4, 1.0)
        self.bc_val_lbl = QLabel("T [°C]")
        self.btn_calc_h = QPushButton("Calculator h…")
        self.btn_calc_h.setToolTip("Opens the convection heat transfer calculator (D4) for computing the coefficient h")
        self.btn_calc_h.clicked.connect(self.open_convection_calc)
        self.btn_calc_h.setVisible(False)
        h_val = QHBoxLayout()
        h_val.addWidget(self.bc_val, 1)
        h_val.addWidget(self.btn_calc_h)
        f.addRow(self.bc_val_lbl, h_val)
        self.bc_amb = dspin(20.0, -273.0, 1e5, 3, 1.0, " °C")
        help_tip(self.bc_amb, "Ambient temperature [°C] for convection and radiation",
                 "Ambient temperature [°C] for convection and radiation (air, coolant, enclosure walls)")
        self.bc_amb_lbl = QLabel("T ambient")
        f.addRow(self.bc_amb_lbl, self.bc_amb)
        self.bc_part = QComboBox()
        help_tip(self.bc_part, "Part in which the heat is generated",
                 "Part in which heat is generated (power uniformly in its volume);\ncan also be selected by clicking a face of the part in 3D")
        self.bc_part_lbl = QLabel("Part")
        f.addRow(self.bc_part_lbl, self.bc_part)
        self.bc_cavity = QComboBox()                       # cavity radiation: name of the cavity (faces of one cavity see each other)
        self.bc_cavity.setEditable(True)
        self.bc_cavity.setProperty("_i18n_done", True)
        help_tip(self.bc_cavity, "Cavity name – faces with the same name radiate to each other",
                 "Name of the cavity. All faces with the same cavity name exchange heat by radiation (view factors computed by CalculiX); different names = separate cavities.")
        self.bc_cavity_lbl = QLabel("Cavity")
        f.addRow(self.bc_cavity_lbl, self.bc_cavity)
        self.bc_closed = QCheckBox("closed cavity (no opening)")
        help_tip(self.bc_closed, "Closed = fully enclosed; open = the opening sees T ambient",
                 "Closed cavity: the faces enclose it completely, the view factors are normalised to 1 and no heat leaves to the surroundings (the case still needs a heat sink elsewhere). Open cavity: the part of the view not covered by the faces sees the ambient temperature. One setting per cavity – it is copied to its other faces.")
        self.bc_closed.toggled.connect(lambda *_: self._cavity_amb())
        f.addRow(self.bc_closed)
        hb = QHBoxLayout()
        b = QPushButton("Add from selection")
        b.setToolTip("Boundary condition on the faces selected in 3D (heat generation on the part chosen in the list)")
        b.clicked.connect(self.add_bc)
        hb.addWidget(b)
        b = QPushButton("Save changes to selected")
        b.setToolTip("Overwrite the values of the boundary condition selected in the list (faces are kept)")
        b.clicked.connect(self.update_bc)
        hb.addWidget(b)
        f.addRow(hb)
        self.bc_hint = QLabel()
        self.bc_hint.setWordWrap(True)
        f.addRow(self.bc_hint)
        lay.addWidget(g)
        h = QHBoxLayout()
        b = QPushButton("Faces from selection")
        b.setToolTip("Replace the faces of the selected boundary condition with the current selection")
        b.clicked.connect(lambda: self.win.update_faces_of(self.bc_list, "thermal"))
        h.addWidget(b)
        b = QPushButton("Delete")
        b.setToolTip("Delete the boundary condition selected in the list")
        b.clicked.connect(lambda: self.win.delete_item(self.bc_list, "thermal"))
        h.addWidget(b)
        lay.addLayout(h)

        # ---------------------------------------------------------- contact conductance of interfaces (all thermal cases)
        g = Section("Contact conductance (all thermal cases)", key="thermal.conductance", collapsed=True,
                    tip="Temperature jump at the interface of two parts (h_c).",
                    help="<b>Thermal contact conductance</b><p>Interfaces between two parts with a contact conductance h_c [W/(m²·K)]: heat flux q = h_c·ΔT across the interface. Without it touching parts are in perfect thermal contact. One of the two parts is meshed separately (two sets of nodes) and stays tied mechanically by *TIE. The heat through each interface is listed in the results and the report.</p>")
        self.conductance = ConductanceBox(self)
        QHBoxLayout(g.body).addWidget(self.conductance)
        lay.addWidget(g)

        # ---------------------------------------------------------- teploty studie
        g = Section("Study temperatures", key="thermal.temperatures", collapsed=True,
                    tip="Sets the reference and initial temperature of the study.",
                    help="<b>Study temperatures</b><p>The reference temperature defines zero thermal strain. The initial temperature applies to transient states at time zero.</p>")
        f = QFormLayout(g.body)
        self.t_ref = dspin(20.0, -273.0, 1e5, 2, 1.0, " °C")
        self.t_ref.setToolTip("Temperature without thermal strain (assembly) – thermal stress from ΔT = T − T_ref")
        self.t_init = dspin(20.0, -273.0, 1e5, 2, 1.0, " °C")
        help_tip(self.t_init, "Initial model temperature at time 0 [°C]",
                 "Initial temperature of the whole model at time 0 for all transient cases [°C].\nCan be edited after unchecking “= reference”.")
        self.t_init_same = QCheckBox("= reference")
        self.t_init_same.setToolTip("Checked = transient cases start from the reference temperature")
        self.t_init_same.setChecked(True)
        hi = QHBoxLayout()
        hi.addWidget(self.t_init, 1)
        hi.addWidget(self.t_init_same)
        f.addRow("Reference (assembly)", self.t_ref)
        f.addRow("Initial (transient)", hi)
        self._commit_on_edit(self.t_ref, self.on_temps)
        self._commit_on_edit(self.t_init, self.on_temps)
        self.t_init_same.toggled.connect(self.on_temps)
        lay.addWidget(g)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        lay.addWidget(self.info)
        lay.addStretch(1)
        self.win.view.facesPicked.connect(self.on_faces_picked)
        self.on_bc_type()

    @staticmethod
    def _commit_on_edit(spin, slot):
        """The value is written to the study after the edit is finished (Enter, leaving the field) – not at every arrow step
        or typed character (each write discards the results and saves an undo snapshot). A change without focus (setValue
        from a script or a test) is written at once. Without a change of value the slot does nothing.
        """
        spin.editingFinished.connect(slot)
        spin.valueChanged.connect(lambda *_: None if spin.hasFocus() else slot())

    # ================================================================== state ↔ controls
    @property
    def study(self):
        return self.win.study

    def current_case(self) -> ThermalCase | None:
        st = self.study
        i = self.case_combo.currentIndex()
        if st is not None and 0 <= i < len(st.thermal.cases):
            return st.thermal.cases[i]
        return None

    def refresh(self):
        """Fills the controls from the study (calls MainWindow.refresh_all)."""
        st = self.study
        self._filling = True
        try:
            cur = self.case_combo.currentIndex()
            self.case_combo.blockSignals(True)
            self.case_combo.clear()
            if st is not None:
                self.case_combo.addItems([c.name for c in st.thermal.cases])
            if self.case_combo.count():
                self.case_combo.setCurrentIndex(min(max(cur, 0), self.case_combo.count() - 1))
            self.case_combo.blockSignals(False)
            part = self.bc_part.currentText()
            self.bc_part.clear()
            self.bc_part.addItems([p.name for p in self.win.parts])
            if part:
                self.bc_part.setCurrentText(part)
            cav = self.bc_cavity.currentText()
            self.bc_cavity.clear()
            self.bc_cavity.addItems((cavity_names(st) if st is not None else []) or ["Cavity 1"])
            if cav:
                self.bc_cavity.setCurrentText(cav)
            self.conductance.refresh()
            if st is not None:
                th = st.thermal
                self.t_ref.setValue(th.reference_temperature)
                self.t_init_same.setChecked(th.initial_temperature is None)
                self.t_init.setValue(th.t_initial)
                self.t_init.setEnabled(th.initial_temperature is not None)
        finally:
            self._filling = False
        self.fill_case()

    def fill_case(self):
        c = self.current_case()
        self._filling = True
        try:
            self.transient.setEnabled(c is not None)
            if c is not None:
                self.transient.setChecked(c.transient)
                self.t_time.setValue(c.time)
                self.t_dt.setValue(c.dt)
                self.t_out.setValue(c.outputs)
            for w in (self.t_time, self.t_dt, self.t_out):
                w.setEnabled(c is not None and c.transient)
            self.bc_list.clear()
            if c is not None:
                for bc in c.loads:
                    self.bc_list.addItem(describe_bc(bc))
        finally:
            self._filling = False
        self.update_info()

    def update_info(self):
        """Description of the state + errors and warnings of the readiness check that concern the heat."""
        c = self.current_case()
        st = self.study
        if c is None:
            self.t_info.setText("<i>No thermal case – add one with the + button.</i>")
        elif c.transient:
            n = c.time / c.step_size() if c.step_size() > 0 else 0
            self.t_info.setText(f"{fmt(n, 6)} time steps of {fmt(c.step_size())} s, a snapshot every {fmt(c.time / max(c.outputs, 1))} s. Materials need ρ and c_p.")
        else:
            self.t_info.setText("Steady state – the heat must have a way out (prescribed temperature, convection, radiation).")
        rows = []
        if st is not None and st.thermal.cases:
            from fea_core.thermal.check import check
            errors, warnings, _ = check(st, [p.name for p in self.win.parts])
            for e in errors:
                rows.append(f"<span style='color:#d23b3b'>● {escape(e)}</span>")
            for w in warnings:
                if w.startswith(("Thermal case", "Contact conductance")):
                    rows.append(f"<span style='color:#b36b00'>● {escape(w)}</span>")
        rows.append("<small>Thermal units: °C, W, W/m², W/(m²·K); material λ [W/(m·K)], α [1/K], c_p [J/(kg·K)] (Materials step).</small>")
        self.info.setText("<br>".join(rows))

    def on_case(self):
        self.fill_case()
        self.win.refresh_sets()

    def on_case_settings(self, *_):
        if self._filling:
            return
        c = self.current_case()
        if c is None:
            return
        new = (self.transient.isChecked(), self.t_time.value(), self.t_dt.value(), self.t_out.value())
        if new == (c.transient, c.time, c.dt, c.outputs):
            return                                  # editingFinished without a change of value does not discard the results
        c.transient, c.time, c.dt, c.outputs = new
        self.win.changed()

    def on_temps(self, *_):
        if self._filling or self.study is None:
            return
        th = self.study.thermal
        new = (self.t_ref.value(), None if self.t_init_same.isChecked() else self.t_init.value())
        if new == (th.reference_temperature, th.initial_temperature):
            return                                  # editingFinished without a change of value does not discard the results
        th.reference_temperature, th.initial_temperature = new
        self.win.changed()

    # ================================================================== stavy
    def default_case_name(self) -> str:
        names = {c.name for c in self.study.thermal.cases}
        k = len(names) + 1
        while f"Thermal case {k}" in names:
            k += 1
        return f"Thermal case {k}"

    def new_case(self, name: str, transient: bool = False) -> ThermalCase | None:
        """Adds a thermal state (without a dialog – also for scripts and tests) and selects it."""
        st = self.study
        name = name.strip()
        if st is None or not name:
            return None
        if any(c.name == name for c in st.thermal.cases):
            QMessageBox.warning(self, APP, f"Thermal case '{name}' already exists.")
            return None
        c = ThermalCase(name, transient=transient)
        st.thermal.cases.append(c)
        self.win.changed()
        self.case_combo.setCurrentIndex(len(st.thermal.cases) - 1)
        return c

    def add_case(self):
        if self.study is None:
            return
        name, ok = QInputDialog.getText(self, APP, "Thermal case name:", text=self.default_case_name())
        if ok:
            self.new_case(name)

    def set_case_name(self, name: str):
        """Renames the selected state including the references from the thermal loads of the mechanical states."""
        c = self.current_case()
        name = name.strip()
        if c is None or not name or name == c.name:
            return
        if any(x.name == name for x in self.study.thermal.cases):
            QMessageBox.warning(self, APP, f"Thermal case '{name}' already exists.")
            return
        for lc in self.study.load_cases:
            for l in lc.loads:
                if isinstance(l, ThermalLoad) and l.case == c.name:
                    l.case = name
        c.name = name
        self.win.changed()

    def rename_case(self):
        c = self.current_case()
        if c is None:
            return
        name, ok = QInputDialog.getText(self, APP, "New name:", text=c.name)
        if ok:
            self.set_case_name(name)

    def delete_case(self):
        c = self.current_case()
        st = self.study
        if c is None:
            return
        for bc in c.loads:
            self.win._drop_selection(getattr(bc, "selection", None))
        st.thermal.cases.remove(c)
        removed = 0
        for lc in st.load_cases:
            n = len(lc.loads)
            lc.loads = [l for l in lc.loads if not (isinstance(l, ThermalLoad) and l.case == c.name)]
            removed += n - len(lc.loads)
        if removed:
            self.win.log_msg(f"Deleted thermal loads referencing thermal case '{c.name}': {removed}")
        self.win.changed()

    # ================================================================== boundary conditions
    def on_bc_type(self, *_):
        kind = self.bc_type.currentData()
        _txt, _prefix, qty, default, _col = BC_TYPES[kind]
        self.bc_val_lbl.setText(qty)
        for w in (self.bc_val, self.bc_val_lbl):
            help_tip(w, VALUE_SHORT[kind], VALUE_TIPS[kind])
        cav = kind == "cavity_radiation"
        amb = kind in ("convection", "radiation") or (cav and not self.bc_closed.isChecked())
        for w in (self.bc_amb, self.bc_amb_lbl):
            w.setVisible(amb)
        for w in (self.bc_cavity, self.bc_cavity_lbl, self.bc_closed):
            w.setVisible(cav)
        self.btn_calc_h.setVisible(kind == "convection")
        gen = kind == "heat_generation"
        for w in (self.bc_part, self.bc_part_lbl):
            w.setVisible(gen)
        self.bc_val.setDecimals(3 if kind in ("radiation", "cavity_radiation") else 4)
        if not self._filling:
            self.bc_val.setValue(default)
        self.bc_hint.setText(f"<small>{HINTS[kind]}</small>")

    def _cavity_amb(self):
        """Open cavity: the ambient temperature of the opening is shown."""
        if self.bc_type.currentData() == "cavity_radiation":
            for w in (self.bc_amb, self.bc_amb_lbl):
                w.setVisible(not self.bc_closed.isChecked())

    def open_convection_calc(self):
        from .calc_dialog import CalcDialog
        dlg = CalcDialog(self.win)
        dlg.tabs.setCurrentWidget(dlg.forms.get("convection", dlg.tabs.widget(0)))
        dlg.exec()

    def on_faces_picked(self, faces):
        """Heat source: a click on a face selects its part."""
        if faces and self.bc_type.currentData() == "heat_generation" and \
                getattr(self.win, "stage", -1) == self.win.STAGE["Heat"]:
            self.bc_part.setCurrentText(self.win.parts[faces[-1][0]].name)

    def bc_from_form(self, kind: str, selection: str | None):
        v = self.bc_val.value()
        if kind == "fixed_temperature":
            if v <= ABSOLUTE_ZERO:
                QMessageBox.warning(self, APP, "The temperature is below absolute zero.")
                return None
            return FixedTemperature(selection, v)
        if kind == "heat_flux":
            return HeatFlux(selection, v)
        if kind == "heat_flow":
            return HeatFlow(selection, v)
        if kind == "convection":
            if v <= 0:
                QMessageBox.warning(self, APP, "The heat transfer coefficient h must be positive.")
                return None
            return Convection(selection, v, self.bc_amb.value())
        if kind == "radiation":
            if not 0 < v <= 1:
                QMessageBox.warning(self, APP, "Emissivity ε must be in the range (0; 1].")
                return None
            return Radiation(selection, v, self.bc_amb.value())
        if kind == "cavity_radiation":
            try:
                return cavity_from_values(selection, v, self.bc_amb.value(), self.bc_cavity.currentText(),
                                          self.bc_closed.isChecked())
            except ValueError as e:
                QMessageBox.warning(self, APP, str(e))
                return None
        part = self.bc_part.currentText()
        if not part:
            QMessageBox.warning(self, APP, "Select the part with the heat source.")
            return None
        return HeatGeneration(part, v)

    def add_bc(self):
        st = self.study
        if st is None:
            return
        kind = self.bc_type.currentData()
        sels = None
        if kind != "heat_generation":
            sels = self.win.need_faces()
            if not sels:
                return
        bc = self.bc_from_form(kind, "")
        if bc is None:
            return
        c = self.current_case()
        if c is None:                                  # first condition without a state → the state is created automatically
            c = ThermalCase(self.default_case_name())
            st.thermal.cases.append(c)
            self.case_combo.blockSignals(True)
            self.case_combo.addItem(c.name)
            self.case_combo.setCurrentIndex(self.case_combo.count() - 1)
            self.case_combo.blockSignals(False)
            self.win.log_msg(f"Thermal case '{c.name}' created")
        if sels is not None:
            bc.selection = self.win.new_selection(BC_TYPES[kind][1], sels)
            self.win.view.set_selected_faces([])
        c.loads.append(bc)
        self._sync_cavity(c, bc)
        self.win.changed()

    def _sync_cavity(self, case, bc):
        if isinstance(bc, CavityRadiation):
            changed = sync_cavity(case, bc)
            if changed:
                self.win.log_msg(f"Cavity '{bc.cavity_name}': closed / ambient temperature also set on {', '.join(changed)}")

    def update_bc(self):
        c = self.current_case()
        row = self.bc_list.currentRow()
        if c is None or not (0 <= row < len(c.loads)):
            QMessageBox.warning(self, APP, "Select a boundary condition in the list.")
            return
        old = c.loads[row]
        kind = self.bc_type.currentData()
        sel = getattr(old, "selection", None)
        if (kind == "heat_generation") != (sel is None):
            QMessageBox.warning(self, APP, "A heat source in a part cannot be changed to a condition on faces (or vice versa) – add a new one.")
            return
        bc = self.bc_from_form(kind, sel)
        if bc is None:
            return
        c.loads[row] = bc
        self._sync_cavity(c, bc)
        self.win.changed()
        self.bc_list.setCurrentRow(row)

    def on_bc_row(self, row):
        c = self.current_case()
        if c is None or not (0 <= row < len(c.loads)) or self._filling:
            return
        bc = c.loads[row]
        sel = getattr(bc, "selection", None)
        sel = sel if isinstance(sel, str) else None
        self.win.view.set_selected_faces(self.win.sel_faces.get(sel, []) if sel else [])
        if bc.kind not in BC_TYPES and not is_cavity_bc(bc):
            return                                    # contact conductance in a case (older file) – no form
        self._filling = True
        try:
            kind = "cavity_radiation" if is_cavity_bc(bc) else bc.kind
            self.bc_type.setCurrentIndex(max(0, self.bc_type.findData(kind)))
            if kind == "cavity_radiation":
                v = cavity_values(bc)
                self.bc_cavity.setCurrentText(v["name"])
                self.bc_closed.setChecked(v["closed"])
            self.on_bc_type()
            if is_cavity_bc(bc):
                self.bc_val.setValue(bc.emissivity)
                self.bc_amb.setValue(bc.ambient)
            elif isinstance(bc, Convection):
                self.bc_val.setValue(bc.h)
                self.bc_amb.setValue(bc.ambient)
            elif isinstance(bc, Radiation):
                self.bc_val.setValue(bc.emissivity)
                self.bc_amb.setValue(bc.ambient)
            else:
                self.bc_val.setValue(bc.value)
            if isinstance(bc, HeatGeneration):
                self.bc_part.setCurrentText(bc.part)
        finally:
            self._filling = False
        self.win.update_status()

    # ================================================================== 3D
    def show_3d(self):
        """Coloured face sets of the boundary conditions of the selected state (+ flux arrows, parts with a heat source)."""
        win = self.win
        v = win.view
        c = self.current_case()
        if c is None:
            return
        L = 0.12 * v.model_size()
        names = [p.name for p in win.parts]
        self.conductance.show_3d()                     # interfaces with h_c (all thermal cases)
        for k, bc in enumerate(c.loads):
            kind = "cavity_radiation" if is_cavity_bc(bc) else bc.kind
            col = BC_TYPES[kind][4] if kind in BC_TYPES else OTHER_COLORS.get(kind, (128, 128, 128))
            if isinstance(bc, ContactConductance):
                for side in (bc.selection, bc.master):
                    if side in win.sel_faces:
                        v.set_face_set(f"t:{k}:{side}", win.sel_faces[side], col, label=describe_bc(bc))
                continue
            if isinstance(bc, HeatGeneration):
                if bc.part in names:
                    pi = names.index(bc.part)
                    v.set_face_set(f"t:{k}:{bc.part}", [(pi, fi) for fi in range(len(v.part_faces[pi]))], col,
                                   label=describe_bc(bc), transparency=0.6)
                continue
            if isinstance(bc.selection, list):            # cavity with several selections (MCP / older files)
                faces = [f for s in bc.selection for f in win.sel_faces.get(s, [])]
                key = f"t:{k}:" + ",".join(bc.selection)
            else:
                faces = win.sel_faces.get(bc.selection, [])
                key = f"t:{bc.selection}"
            v.set_face_set(key, faces, col, label=describe_bc(bc))
            if isinstance(bc, (HeatFlux, HeatFlow)) and bc.value != 0:
                s = float(np.sign(bc.value))          # a positive flux points into the body (against the outward normal)
                arrows = []
                for pi, fi in faces[:8]:
                    p = win.face_prop(pi, fi)
                    n = np.asarray(p["normal"], float)
                    arrows.append((v.place_on_face(win.arrow_point(p), -s * n, n, L), -s * n, L))
                v.add_arrows(key, arrows, col)
