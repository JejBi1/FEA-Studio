"""General orthotropic material (apart from 3D printing) in the Materials step: switch, constants E1–E3, ν12/13/23, G12/13/23
(optionally α1–α3) in the axes of the coordinate system of the study (Material.ortho, Material.ortho_csys, see fea_core.ortho).
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from .widgets import Section, dspin, fmt, help_tip

KEYS = (("E1", "E2", "E3"), ("nu12", "nu13", "nu23"), ("G12", "G13", "G23"), ("alpha1", "alpha2", "alpha3"))
ROWS = (("E [MPa]", "Young's modulus along axis {}"), ("ν", "Poisson's ratio ν{} (ε_j = −ν_ij·σ_i / E_i)"),
        ("G [MPa]", "shear modulus in plane {}"), ("α [10⁻⁶/K]", "coefficient of thermal expansion along axis {} (0 = isotropic α of the material)"))
SUBS = (("1", "2", "3"), ("12", "13", "23"), ("12", "13", "23"), ("1", "2", "3"))
WARN = ("The safety factor against Re and von Mises does not apply to an orthotropic material – evaluate the stress in the material axes against its strengths (the stress components in the results are in global axes).")


class OrthoBox(Section):
    """Collapsible section with a tick in the header: tick = the material is orthotropic (constants and axes are shown;
    an unticked section is only one header row).
    """

    def __init__(self, win):
        self.win = win
        self.check = QCheckBox()
        self.check.setToolTip("Tick to make the material orthotropic (9 independent constants in axes 1, 2, 3)")
        inner = QWidget()
        super().__init__(
            "Orthotropic material (axes 1, 2, 3)", inner, check=self.check,
            tip="Tick to make the material orthotropic (9 independent constants in axes 1, 2, 3)",
            help="A material with three mutually perpendicular axes of symmetry. Enter the moduli E1–E3, Poisson's ratios ν12, ν13, ν23 (ε<sub>j</sub> = −ν<sub>ij</sub>·σ<sub>i</sub>/E<sub>i</sub>) and the shear moduli G12, G13, G23. The material axes are the axes of the selected coordinate system (Cartesian: 1, 2, 3 = x, y, z; cylindrical: 1 = radial, 2 = tangential, 3 = along the axis).<p>The constants must give a positive definite matrix (ν<sub>ij</sub>² < E<sub>i</sub>/E<sub>j</sub>) – the Solve step checks it.</p><p>The safety factor to Re (von Mises) does not apply to an orthotropic material.</p><p>Ticking the box makes the material orthotropic (wood, composite, laminates…): 9 independent constants. E and ν above are then not used (E = E1, ν = ν12). For 3D printing use the 3D print step (transverse isotropy by build direction).</p>")
        self.inner = inner
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 0, 0)
        grid = QGridLayout()
        self.spins: dict[str, object] = {}
        for r, ((lbl, tip), keys, subs) in enumerate(zip(ROWS, KEYS, SUBS)):
            grid.addWidget(QLabel(lbl), r, 0)
            for c, (k, s) in enumerate(zip(keys, subs)):
                if r == 1:
                    w = dspin(0.3, -0.999, 0.999, 4, 0.01)
                elif r == 3:
                    w = dspin(0.0, 0.0, 1e4, 3, 1.0, special="isotropic")
                else:
                    w = dspin(210000.0, 0.0, 1e8, 0, 1000.0)
                w.setToolTip(f"{k}: {tip.format(s)}")
                self.spins[k] = w
                grid.addWidget(w, r, c + 1)
        v.addLayout(grid)
        h = QHBoxLayout()
        h.addWidget(QLabel("Material axes"))
        self.csys = QComboBox()
        help_tip(self.csys, "Material axes coordinate system: global, local Cartesian or cylindrical",
                 "Material axes coordinate system: global axes (1, 2, 3 = x, y, z), a local Cartesian system (the x, y, z axes of the system) or cylindrical (1 = radial, 2 = tangential, 3 = along the axis – pipes, wound parts, tree trunk). Systems are created with the “Systems…” button.")
        h.addWidget(self.csys, 1)
        b = QPushButton("Systems…")
        b.setToolTip("Create / edit local coordinate systems (same as for loads)")
        b.clicked.connect(self.edit_csys)
        h.addWidget(b)
        v.addLayout(h)
        b = QPushButton("Fill isotropically from E, ν")
        help_tip(b, "Fill E1 = E2 = E3 = E, ν and G = E / (2(1 + ν)) from the fields above",
                 "E1 = E2 = E3 = E, ν12 = ν13 = ν23 = ν, G = E / (2(1 + ν)) from the fields above – a starting point for editing")
        b.clicked.connect(self.fill_isotropic)
        v.addWidget(b)
        self.warn = QLabel(WARN)
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color:#b36b00")
        v.addWidget(self.warn)
        self.check.toggled.connect(self._on_toggle)

    # the tick (orthotropic yes / no) is independent of the expansion of the section (Section.isChecked = expanded)
    def isChecked(self) -> bool:
        return self.check.isChecked()

    def setChecked(self, on: bool) -> None:
        self.check.setChecked(bool(on))

    def _on_toggle(self, on: bool):
        self.inner.setEnabled(True)
        for w in (self.win.m_E, self.win.m_nu):
            w.setEnabled(not on)
        if on and all(self.spins[k].value() == 210000.0 for k in ("E1", "E2", "E3")) and \
                self.win.m_E.value() != 210000.0:
            self.fill_isotropic()

    def fill_isotropic(self):
        E, nu = self.win.m_E.value(), self.win.m_nu.value()
        for k in KEYS[0]:
            self.spins[k].setValue(E)
        for k in KEYS[1]:
            self.spins[k].setValue(nu)
        for k in KEYS[2]:
            self.spins[k].setValue(E / (2 * (1 + nu)))

    def refresh_csys(self, select=None):
        cur = select if select is not None else self.csys.currentData()
        self.csys.blockSignals(True)
        self.csys.clear()
        self.csys.addItem("global axes (1, 2, 3 = x, y, z)", None)
        st = self.win.study
        if st is not None:
            for k, c in st.coordinate_systems.items():
                self.csys.addItem(f"{k} ({'cylindrical: 1 = r, 2 = θ, 3 = z' if c.kind == 'cylindrical' else 'Cartesian'})",
                                  k)
        if cur and self.csys.findData(cur) < 0:
            self.csys.addItem(f"{cur} (does not exist)", cur)
        self.csys.setCurrentIndex(max(0, self.csys.findData(cur)) if cur else 0)
        self.csys.blockSignals(False)

    def edit_csys(self):
        if self.win.study is None:
            return
        from .csys_dialog import CsysDialog
        CsysDialog(self.win).exec()
        self.refresh_csys()
        self.win.refresh_csys_combo()

    def load(self, m):
        """Loads the material into the form (isotropic → switch off, constants from E, ν)."""
        o = getattr(m, "ortho", None) or {}
        self.check.blockSignals(True)
        self.check.setChecked(bool(o))
        self.check.blockSignals(False)
        self.set_open(bool(o))
        self.inner.setEnabled(bool(o))
        for w in (self.win.m_E, self.win.m_nu):
            w.setEnabled(not o)
        if o:
            for k, w in self.spins.items():
                val = o.get(k)
                if k.startswith("alpha"):
                    w.setValue(float(val) * 1e6 if val else 0.0)
                elif val is not None:
                    w.setValue(float(val))
        else:
            E, nu = float(m.E), float(m.nu)
            for k in KEYS[0]:
                self.spins[k].setValue(E)
            for k in KEYS[1]:
                self.spins[k].setValue(nu)
            for k in KEYS[2]:
                self.spins[k].setValue(E / (2 * (1 + nu)))
            for k in KEYS[3]:
                self.spins[k].setValue(0.0)
        self.refresh_csys(getattr(m, "ortho_csys", None) or "")

    def values(self) -> tuple[dict | None, str | None]:
        """(Material.ortho, Material.ortho_csys); isotropic material → (None, None)."""
        if not self.isChecked():
            return None, None
        o = {k: float(self.spins[k].value()) for row in KEYS[:3] for k in row}
        for k in KEYS[3]:
            a = self.spins[k].value()
            if a:
                o[k] = a * 1e-6
        return o, self.csys.currentData() or None

    @staticmethod
    def describe(m) -> str:
        o = getattr(m, "ortho", None)
        if not o:
            return ""
        return ("orthotropic: E1/E2/E3 = " + " / ".join(fmt(float(o[k]), 5) for k in KEYS[0]) + " MPa, ν = "
                + " / ".join(fmt(float(o[k]), 3) for k in KEYS[1]) + ", G = "
                + " / ".join(fmt(float(o[k]), 5) for k in KEYS[2]) + " MPa"
                + (f", axes {m.ortho_csys}" if getattr(m, "ortho_csys", None) else ", global axes"))
