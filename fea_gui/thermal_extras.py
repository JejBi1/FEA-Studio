"""Heat step additions: thermal contact conductance of interfaces (D1) and cavity radiation (D2).

- `ConductanceBox` – section of the Heat panel: interfaces between two parts with h_c [W/(m²·K)] (ThermalSettings.conductances,
  apply to all thermal cases); one of the parts is meshed separately (fea_core.thermal.interfaces.set_interface).
- `conductance_row` – h_c field + "Suggest…" (also in the Contacts step for a contact pair).
- `SuggestDialog` – typical values (CONTACT_CONDUCTANCE_DATABASE) and the estimate from the materials, pressure, roughness, filler.
- pure mapping of the cavity condition (form ↔ CavityRadiation) – tested without Qt (tests/test_gui_thermal_extras.py).
"""
from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QListWidget,
                               QMessageBox, QPushButton, QTableWidgetItem, QVBoxLayout, QWidget)

from fea_core.thermal.model import CavityRadiation, ContactConductance, Radiation

from .widgets import dspin, fit_rows, fmt_dash as fmt, help_tip, set_placeholder, table

APP = "FEA Studio"
CONDUCTANCE_COLOR = (150, 95, 40)
TIM_CHOICES = (("", "none (dry contact)"), ("paste", "thermal paste"), ("pad", "thermal pad"))


# ------------------------------------------------------------------ pure mapping (no widgets)

def is_cavity_bc(bc) -> bool:
    return isinstance(bc, CavityRadiation) or (isinstance(bc, Radiation) and bool(getattr(bc, "cavity", False)))


def cavity_from_values(selection: str, emissivity: float, ambient: float, name: str, closed: bool) -> CavityRadiation:
    """Cavity condition from the form; ValueError with a message for the user."""
    if not 0 < emissivity <= 1:
        raise ValueError("Emissivity ε must be in the range (0; 1].")
    name = (name or "").strip()
    if not name:
        raise ValueError("Enter the cavity name – faces with the same name radiate to each other.")
    return CavityRadiation(selection, emissivity=float(emissivity), ambient=float(ambient), cavity_name=name,
                           closed=bool(closed))


def cavity_values(bc) -> dict:
    return {"emissivity": bc.emissivity, "ambient": bc.ambient, "name": getattr(bc, "cavity_name", None) or "CAV",
            "closed": bool(getattr(bc, "closed", False))}


def sync_cavity(case, bc) -> list[str]:
    """One cavity = one setting of closed / ambient: copies them from bc to the other faces of the same cavity in the case.
    Returns the selections that changed."""
    out = []
    for b in case.loads:
        if b is bc or not is_cavity_bc(b) or (getattr(b, "cavity_name", None) or "CAV") != bc.cavity_name:
            continue
        if (bool(b.closed), b.ambient) != (bool(bc.closed), bc.ambient):
            b.closed, b.ambient = bool(bc.closed), bc.ambient
            out.append(str(b.selection))
    return out


def cavity_names(study) -> list[str]:
    return sorted({getattr(b, "cavity_name", None) or "CAV" for c in study.thermal.cases for b in c.loads if is_cavity_bc(b)})


def describe_cavity(bc) -> str:
    v = cavity_values(bc)
    sel = ", ".join(bc.selection) if isinstance(bc.selection, list) else bc.selection
    state = "closed" if v["closed"] else f"open, T ambient {fmt(v['ambient'])} °C"
    return f"{sel}: cavity “{v['name']}” ε = {fmt(v['emissivity'], 3)}, {state}"


def describe_conductance(c: ContactConductance) -> str:
    return f"{c.name}: h_c = {fmt(c.conductance, 5)} W/(m²·K) ({c.selection} – {c.master})"


# ------------------------------------------------------------------ h_c suggestion

class SuggestDialog(QDialog):
    """Typical contact conductances and the estimate from the materials of the two sides (INDICATIVE)."""

    def __init__(self, parent, study, part_names: list[str], a: str, b: str, value: float = 0.0):
        super().__init__(parent)
        from fea_core.thermal.contact_conductance import CONTACT_CONDUCTANCE_DATABASE
        self.setWindowTitle("Contact conductance h_c")
        self.study, self.part_names, self.a, self.b = study, part_names, a, b
        self.value = value or None
        lay = QVBoxLayout(self)
        lbl = QLabel("Typical values (double-click = use). h_c depends strongly on pressure, roughness and filling – "
                     "measure or take a datasheet value when the result matters.")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        self.db = table(["Interface", "h_c [W/(m²·K)]", "Source"], 120, rows=(4, 9))
        help_tip(self.db, "Typical contact conductances (double-click = use the value)")
        rows = list(CONTACT_CONDUCTANCE_DATABASE.values())
        self.db.setRowCount(len(rows))
        for r, d in enumerate(rows):
            for k, v in enumerate((d["description"], fmt(d["conductance_W_m2K"], 6), d["source"])):
                it = QTableWidgetItem(v)
                if k == 1:
                    it.setData(Qt.UserRole, float(d["conductance_W_m2K"]))
                self.db.setItem(r, k, it)
        self.db.cellDoubleClicked.connect(self._use_row)
        lay.addWidget(self.db)
        f = QFormLayout()
        mats = []
        from fea_core.thermal.interfaces import side_parts
        for s in (a, b):
            m = ""
            for p in side_parts(study, s, part_names):
                try:
                    m = study.material_of(p).name
                    break
                except KeyError:
                    pass
            mats.append(m or "(no material – steel assumed)")
        self.mats = QLabel(f"{escape(a or '–')}: {escape(mats[0])}<br>{escape(b or '–')}: {escape(mats[1])}")
        self.mats.setTextFormat(Qt.RichText)
        f.addRow("Materials", self.mats)
        self.p = dspin(1.0, 0.01, 1e4, 2, 0.5, " MPa")
        help_tip(self.p, "Contact pressure [MPa] (bolted joint ~10–50, loose contact ~0.1–1)")
        f.addRow("Pressure", self.p)
        self.ra = dspin(3.0, 0.05, 100.0, 2, 0.5, " µm")
        help_tip(self.ra, "Surface roughness Ra [µm] (ground ~0.4–1, machined ~1.6–6.3)")
        f.addRow("Roughness Ra", self.ra)
        self.tim = QComboBox()
        for k, t in TIM_CHOICES:
            self.tim.addItem(t, k)
        help_tip(self.tim, "Filler between the faces (paste / pad replaces the dry estimate)")
        f.addRow("Filler", self.tim)
        self.est = QLabel()
        f.addRow("Estimate", self.est)
        lay.addLayout(f)
        for w in (self.p, self.ra):
            w.valueChanged.connect(lambda *_: self._estimate())
        self.tim.currentIndexChanged.connect(lambda *_: self._estimate())
        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        use = bb.addButton("Use estimate", QDialogButtonBox.AcceptRole)
        use.setToolTip("Take the estimate from the materials, pressure, roughness and filler")
        use.clicked.connect(self._use_estimate)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._estimate()

    def estimate(self) -> float:
        from fea_core.thermal.interfaces import suggest
        return float(suggest(self.study, self.a, self.b, self.part_names, self.p.value(), self.ra.value(),
                             self.tim.currentData() or None)["h_W_m2K"])

    def _estimate(self):
        self.est.setText(f"h_c ≈ {fmt(self.estimate(), 5)} W/(m²·K) (indicative)")

    def _use_estimate(self):
        self.value = self.estimate()
        self.accept()

    def _use_row(self, r, _c=0):
        it = self.db.item(r, 1)
        if it is not None:
            self.value = float(it.data(Qt.UserRole))
            self.accept()


def conductance_row(owner, sides) -> tuple:
    """h_c field (0 = perfect contact) + button "Suggest…"; sides() → (side A, side B) for the estimate.
    Returns (spin, widget for a form row)."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    spin = dspin(0.0, 0.0, 1e9, 1, 500.0, " W/(m²·K)", special="perfect contact")
    help_tip(spin, "Thermal contact conductance h_c [W/(m²·K)]; 0 = perfect contact",
             "<b>Thermal contact conductance h_c</b><p>Heat flux across the interface q = h_c·(T_A − T_B) – a temperature jump "
             "ΔT = q / h_c. 0 = perfect thermal contact (no jump). Typical: steel dry 1 000–3 000, bolted flange 5 000, "
             "thermal paste 25 000 W/(m²·K). Applies to all thermal cases.</p>")
    h.addWidget(spin, 1)
    b = QPushButton("Suggest…")
    b.setToolTip("Typical values and an estimate from the materials and the pressure")

    def ask():
        a, bb = sides()
        win = getattr(owner, "win", None)
        st = getattr(win, "study", None)
        if st is None:
            return
        dlg = SuggestDialog(owner, st, [p.name for p in win.parts], a, bb, spin.value())
        if dlg.exec() and dlg.value:
            spin.setValue(dlg.value)
    b.clicked.connect(ask)
    h.addWidget(b)
    return spin, w


# ------------------------------------------------------------------ section of the Heat panel

class ConductanceBox(QWidget):
    """Interfaces with a contact conductance between two parts (all thermal cases)."""

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.win = panel.win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        set_placeholder(self.list, "No interface – touching parts are in perfect thermal contact")
        fit_rows(self.list, 2, 6)
        help_tip(self.list, "Interfaces with a contact conductance (click = load into the form)",
                 "Interfaces between two parts with a thermal contact conductance h_c. They apply to all thermal cases. "
                 "Contact pairs of the Contacts step can have their own h_c (field “Thermal conductance” there).")
        self.list.currentRowChanged.connect(self.on_row)
        lay.addWidget(self.list)
        f = QFormLayout()
        self.part_a, self.part_b = QComboBox(), QComboBox()
        for cb, tip in ((self.part_a, "First part of the interface"), (self.part_b, "Second part of the interface")):
            cb.setProperty("_i18n_done", True)              # part names (data) are not translated
            help_tip(cb, tip)
        f.addRow("Part A", self.part_a)
        f.addRow("Part B", self.part_b)
        self.h, row = conductance_row(self, lambda: (self.part_a.currentText(), self.part_b.currentText()))
        self.h.setValue(1000.0)
        f.addRow("h_c", row)
        hb = QHBoxLayout()
        for txt, slot, tip, attr in (
                ("Add / update", self.add, "Add the interface A–B, or update h_c of an existing one", "btn_add"),
                ("Delete", self.delete, "Delete the interface selected in the list", "btn_delete")):
            bt = QPushButton(txt)
            bt.setToolTip(tip)
            bt.clicked.connect(slot)
            setattr(self, attr, bt)
            hb.addWidget(bt)
        f.addRow(hb)
        lay.addLayout(f)
        self.note = QLabel("<small>A temperature jump needs two sets of nodes: one of the parts is meshed separately and stays "
                           "mechanically tied by *TIE (non-conforming mesh).</small>")
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.RichText)
        lay.addWidget(self.note)

    @property
    def study(self):
        return self.win.study

    def refresh(self):
        st = self.study
        names = [p.name for p in self.win.parts]
        for cb in (self.part_a, self.part_b):
            cur = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(names)
            if cur in names:
                cb.setCurrentText(cur)
            cb.blockSignals(False)
        if self.part_b.currentIndex() == self.part_a.currentIndex() and self.part_b.count() > 1:
            self.part_b.setCurrentIndex(1 if self.part_a.currentIndex() == 0 else 0)
        row = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for c in (st.thermal.conductances if st is not None else []):
            self.list.addItem(describe_conductance(c))
        if 0 <= row < self.list.count():
            self.list.setCurrentRow(row)
        self.list.blockSignals(False)

    def on_row(self, row):
        st = self.study
        if st is None or not (0 <= row < len(st.thermal.conductances)):
            return
        c = st.thermal.conductances[row]
        self.part_a.setCurrentText(c.selection)
        self.part_b.setCurrentText(c.master)
        self.h.setValue(c.conductance)
        self.win.refresh_sets()

    def add_interface(self, a: str, b: str, h: float) -> bool:
        """Adds / updates the interface (without dialogs except the error message – also for scripts and tests)."""
        from fea_core.thermal.interfaces import ConductanceError, set_interface
        st = self.study
        if st is None:
            return False
        if h <= 0:
            QMessageBox.warning(self, APP, "Enter a positive contact conductance h_c (or use “Suggest…”).")
            return False
        try:
            _item, sep, _replaced = set_interface(st, a, b, h, None, [p.name for p in self.win.parts])
        except ConductanceError as e:
            QMessageBox.warning(self, APP, str(e))
            return False
        if sep:
            self.win.log_msg(f"Part '{sep}' is meshed separately for the contact conductance (tied mechanically by *TIE)")
        self.win.changed()
        return True

    def add(self):
        self.add_interface(self.part_a.currentText(), self.part_b.currentText(), self.h.value())

    def delete(self):
        st = self.study
        row = self.list.currentRow()
        if st is None or not (0 <= row < len(st.thermal.conductances)):
            QMessageBox.warning(self, APP, "Select an interface in the list.")
            return
        st.thermal.conductances.pop(row)
        self.win.changed()

    def show_3d(self):
        """Parts of the interfaces in colour (translucent) with the h_c label."""
        st, v = self.study, self.win.view
        if st is None:
            return
        names = [p.name for p in self.win.parts]
        for k, c in enumerate(st.thermal.conductances):
            for side in (c.selection, c.master):
                if side in names:
                    pi = names.index(side)
                    v.set_face_set(f"tc:{k}:{side}", [(pi, fi) for fi in range(len(v.part_faces[pi]))], CONDUCTANCE_COLOR,
                                   label=describe_conductance(c) if side == c.selection else None, transparency=0.7)
