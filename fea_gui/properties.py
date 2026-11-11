""""Properties – <selected item>" card above the step panel (Studio × Tables layout, design kom3).

Shows the properties of what is currently selected: a part (Model, Materials, 3D printing – volume, mass, Show / Count,
material with E, ν, ρ, Re, Rm), a bolt (Joints), a constraint or load (Constraints, Loads – where it acts, face content, value),
the progress of the last calculation (Mesh, Solve). Edits (Show, Count, Material) go the same way as the parts table;
a bolt (size, class, preload) goes through the form of the Joints panel, a force (components) and a cylindrical constraint (constrained directions) directly
into the study; the others are changed in the step form below the card. Without a selection the card is hidden.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout, QLabel,
                               QScrollArea, QWidget)

from fea_core.i18n import tr

from .table_dock import _int
from .widgets import Section, fmt

G_MM = 9810.0
MAX_HEIGHT = 230                                   # px – the rest of the right column belongs to the step form
# calculation phase (fea_core.progress, Results.info["timings"])
PHASE_NAMES = {"check": "Readiness check", "mesh": "Mesh (Gmsh)", "thermal": "Heat conduction",
               "mech": "CalculiX", "eigen": "Natural frequencies / buckling", "dynamics": "Dynamics",
               "post": "Results and equilibrium", "background": "Background tasks"}
APPROX = chr(126)                                  # marker of an approximate equivalent in the converter (fea_core.grades)


class PropertiesCard(Section):
    def __init__(self, main):
        scroll = QScrollArea()                         # content with a limited height – the step form stays below the card
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(MAX_HEIGHT)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("sectionBody")
        scroll.setWidget(body)
        super().__init__("Properties", scroll, key="properties",
                         tip="Properties of the selected item (part, bolt, support, load, analysis)")
        self.setObjectName("propsCard")
        self.main = main
        self._scroll = scroll
        self.form = QFormLayout(body)
        self.form.setContentsMargins(10, 4, 10, 6)
        self.form.setHorizontalSpacing(10)
        self.form.setVerticalSpacing(2)
        self._timer = QTimer(self)                     # the selection changes in batches (click, list) → one refresh
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self.refresh)

    def schedule(self, *_):
        self._timer.start()

    # ------------------------------------------------------------------ obsah
    def _clear(self):
        while self.form.rowCount():
            self.form.removeRow(0)

    def _prop(self, label, value, tip=""):
        if isinstance(value, str):
            w = QLabel()
            w.setTextFormat(Qt.PlainText)                # names from a foreign CAD are never HTML (format before the text:
            w.setText(value)                             # QLabel(text) would already parse it as rich text)
            w.setWordWrap(True)
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
        else:
            w = value
        if tip:
            w.setToolTip(tr(tip))
        lab = QLabel(label)
        lab.setProperty("muted", True)
        self.form.addRow(lab, w)

    def _group(self, text):
        lab = QLabel(text)
        lab.setObjectName("propsGroup")
        self.form.addRow(lab)

    def refresh(self):
        m = self.main
        st = m.study
        stage = m.STAGES[getattr(m, "stage", 0)]
        what = None
        if st is not None:
            if stage in ("Model", "Materials", "3D printing"):
                what = self._sel_part(stage)
            elif stage == "Joints":
                what = self._sel_bolt()
            elif stage == "Constraints":
                r = m.con_list.currentRow()
                what = ("con", r) if 0 <= r < len(st.constraints) else None
            elif stage == "Loads":
                ci, r = m.lc_combo.currentIndex(), m.load_list.currentRow()
                if 0 <= ci < len(st.load_cases) and 0 <= r < len(st.load_cases[ci].loads):
                    what = ("load", (ci, r))
            elif stage in ("Mesh", "Solve"):
                what = ("solve", None)
        if what is None:
            self.setVisible(False)
            return
        self._clear()
        kind, key = what
        getattr(self, "_fill_" + kind)(key)
        self.setVisible(True)
        QTimer.singleShot(0, self._fit)                 # new labels are shown only after the events are processed

    def _fit(self):
        body = self._scroll.widget()
        self._scroll.setFixedHeight(min(MAX_HEIGHT, self.form.sizeHint().height() + 4))
        body.updateGeometry()

    # ------------------------------------------------------------------ selection
    def _sel_part(self, stage):
        m = self.main
        idx = list(m.view.selected_parts())
        if not idx and stage == "Model" and hasattr(m, "model_table"):
            idx = sorted({ix.row() for ix in m.model_table.selectedIndexes()})
        if not idx and stage == "Materials":
            idx = sorted({ix.row() for ix in m.mat_parts.selectedIndexes()})
        idx = [i for i in idx if 0 <= i < len(m.parts)]
        return ("part", idx[0]) if idx else None

    def _sel_bolt(self):
        jp = getattr(self.main, "joints_panel", None)
        r = jp.bolt_list.currentRow() if jp is not None else -1
        return ("bolt", r) if 0 <= r < len(self.main.study.bolts) else None

    # ------------------------------------------------------------------ part
    def _fill_part(self, i):
        m = self.main
        st = m.study
        p = m.parts[i]
        self.header.setText(f"{tr('Properties')} – {p.name}")
        try:
            mat = st.material_of(p.name)
        except KeyError:
            mat = None
        vol = p.volume()
        self._group(tr("Part"))
        self._prop(tr("Name"), p.name)
        try:
            nf = len(p.faces())
        except Exception:  # noqa: BLE001
            nf = None
        self._prop(tr("Volume"), f"{fmt(vol / 1000.0, 4)} cm³" + (f" · {nf} {tr('faces')}" if nf else ""))
        if mat is not None and mat.rho:
            mass = vol * mat.rho * 1000.0
            self._prop(tr("Mass"), f"{fmt(mass, 4)} kg · {tr('weight')} {fmt(mass * G_MM / 1000.0, 3)} N")
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        show = QCheckBox("Show")
        show.setChecked(p.name not in set(st.hidden_parts))
        show.setToolTip(tr("Show the part in 3D (view only – results stay)"))
        show.toggled.connect(lambda on, n=p.name: m.set_hidden([n], not on))
        calc = QCheckBox("Analyze")
        calc.setChecked(p.name not in set(st.ignored_parts))
        calc.setToolTip(tr("The part goes into the analysis (off = excluded, the mesh is discarded)"))
        calc.toggled.connect(lambda on, n=p.name: m.set_ignored([n], not on))
        h.addWidget(show)
        h.addWidget(calc)
        h.addStretch(1)
        self._prop(tr("In analysis"), box)
        self._group(tr("Material"))
        cb = QComboBox()
        cb.setProperty("_i18n_done", True)
        names = sorted(st.materials)
        cb.addItems(names if mat is None or mat.name in names else names + [mat.name])
        if mat is None:
            cb.insertItem(0, "–")
            cb.setCurrentIndex(0)
        else:
            cb.setCurrentText(mat.name)
        cb.currentTextChanged.connect(lambda name, n=p.name: self._set_material(n, name))
        self._prop(tr("Material"), cb, "Material of the part (study materials; more from the library in the Materials step)")
        if mat is not None:
            try:
                from fea_core.grades import lookup
                hits = lookup(mat.name)
                names = []
                if hits and not hits[0][1]:                  # exact match only, not an approximate equivalent
                    for system in ("EN", "W.Nr.", "ČSN"):
                        names += [x for x in hits[0][0].names.get(system, [])[:1] if not x.startswith(APPROX)]
                norm = " · ".join(dict.fromkeys(names))
            except Exception:  # noqa: BLE001 – the converter is only an extra
                norm = ""
            if norm:
                self._prop(tr("Standard"), norm)
            self._prop("E", f"{fmt(mat.E, 6)} MPa" if mat.E else "–")
            self._prop("ν", fmt(mat.nu, 3) if mat.nu is not None else "–")
            self._prop("ρ", f"{fmt(mat.rho * 1e12, 5)} kg/m³" if mat.rho else "–")
            self._prop("Re", f"{fmt(mat.Re, 4)} MPa" if mat.Re else "–")
            self._prop("Rm", f"{fmt(mat.Rm, 4)} MPa" if mat.Rm else "–")
        warn = [w for w in (getattr(m, "geometry_warnings", None) or []) if p.name in w]
        self._prop(tr("Geometry check"), ("! " + "; ".join(warn)) if warn else "√ " + tr("no warnings"))

    def _set_material(self, part, name):
        st = self.main.study
        if st is None or name not in st.materials or st.part_materials.get(part) == name:
            return
        st.part_materials[part] = name
        self.main.changed()

    # ------------------------------------------------------------------ bolt
    def _fill_bolt(self, r):
        m = self.main
        b = m.study.bolts[r]
        self.header.setText(f"{tr('Properties')} – {b.name}")
        conn = {"nut": "with nut", "tapped": "into thread", "fitted": "fitted"}
        jp = m.joints_panel
        self._group(tr("Bolt"))
        self._prop(tr("Designation"), b.designation)
        self._prop(tr("Size"), self._copy_combo(jp.b_size, f"M{b.size:g}", None, r))
        self._prop(tr("Class"), self._copy_combo(jp.b_class, b.strength, None, r))
        self._prop(tr("Joint"), tr(conn.get(b.connection, b.connection)))
        self._group(tr("Preload"))
        self._prop(tr("Mode"), self._copy_combo(jp.b_mode, None, b.preload_mode, r))
        if b.preload_mode in ("force", "torque"):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1e7)
            sp.setDecimals(1)
            sp.setSuffix(" N" if b.preload_mode == "force" else " N·m")
            sp.setValue(b.preload)
            sp.setToolTip(tr("Preload: force F_M,max [N], or tightening torque M_A [N·m], as selected"))
            sp.editingFinished.connect(lambda box=sp, row=r: self._bolt_value(row, box.value()))
            self._prop("F_M,max" if b.preload_mode == "force" else "M_A", sp)
        self._group(tr("Clamped parts"))
        self._prop(tr("Under head"), b.head_part or "–")
        self._prop(tr("Under nut") if b.connection != "tapped" else tr("Thread in part"), b.nut_part or "–")
        if b.washers:
            self._prop(tr("Washers"), str(len(b.washers)))
        a = getattr(m.joints_panel, "assessment", None)
        c = {x.name: x for x in a.bolts}.get(b.name) if a is not None else None
        self._group(tr("Assessment"))
        self._prop(tr("Utilization"), (("√ " if c.utilization <= 1 else "× ") + fmt(c.utilization, 3) + f" · {c.governing}")
                  if c is not None else tr("assessment after the analysis"))

    def _copy_combo(self, src, text, data, row):
        """Menu as in the form of the Joints panel; change = form + Save bolt changes (the same effect on the mesh)."""
        cb = QComboBox()
        cb.setProperty("_i18n_done", data is None)
        for i in range(src.count()):
            cb.addItem(src.itemText(i), src.itemData(i))
        k = cb.findData(data) if data is not None else cb.findText(text)
        if k < 0 and text:
            cb.addItem(text, None)
            k = cb.count() - 1
        cb.setCurrentIndex(max(k, 0))
        cb.setToolTip(src.toolTip())
        cb.currentIndexChanged.connect(lambda _i: self._bolt_combo(row, src, cb))
        return cb

    def _bolt_combo(self, row, src, cb):
        jp = self.main.joints_panel
        jp.bolt_list.setCurrentRow(row)
        k = src.findData(cb.currentData()) if cb.currentData() is not None else src.findText(cb.currentText())
        if k >= 0:
            src.setCurrentIndex(k)
            jp.save_bolt()

    def _bolt_value(self, row, v):
        jp = self.main.joints_panel
        jp.bolt_list.setCurrentRow(row)
        if abs(jp.b_pre.value() - v) > 1e-9:
            jp.b_pre.setValue(v)
            jp.save_bolt()

    # ------------------------------------------------------------------ constraint / load
    def _where(self, sel_name):
        st = self.main.study
        sel = st.selections.get(sel_name) if sel_name else None
        if sel is None:
            return
        n_f = len(sel.faces or [])
        area = sum(getattr(f, "area", 0.0) or 0.0 for f in sel.faces or [])
        parts = sorted({getattr(f, "part", "") for f in sel.faces or []} - {""})
        self._group(tr("Where it acts"))
        self._prop(tr("Selection"), f"{sel_name} · {tr('faces')}: {n_f}" + (f" · {', '.join(parts)}" if parts else ""))
        if area:
            self._prop(tr("Area"), f"{fmt(area, 5)} mm²")

    def _fill_con(self, r):
        m = self.main
        c = m.study.constraints[r]
        self.header.setText(f"{tr('Properties')} – {getattr(c, 'selection', '') or type(c).__name__}")
        try:
            from fea_core.loads_text import constraint_text
            text = constraint_text(c, m.study, num=fmt)
        except Exception:  # noqa: BLE001
            text = m.describe_constraint(c)
        self._group(tr("Constraint"))
        self._prop(tr("Type"), text)
        if getattr(c, "kind", "") == "cylindrical":
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            for attr, label in (("radial", "radially"), ("tangential", "tangentially"), ("axial", "axially")):
                ch = QCheckBox(label)
                ch.setChecked(bool(getattr(c, attr)))
                ch.setToolTip(tr("Constrained displacement component in the cylinder axes"))
                ch.toggled.connect(lambda on, a=attr, obj=c: self._set_attr(obj, a, on))
                h.addWidget(ch)
            h.addStretch(1)
            self._prop(tr("Constrained"), box)
        self._prop(tr("Load cases"), tr("applies in all load cases"))
        self._where(getattr(c, "selection", None))

    def _fill_load(self, key):
        m = self.main
        ci, li = key
        lc = m.study.load_cases[ci]
        ld = lc.loads[li]
        self.header.setText(f"{tr('Properties')} – {getattr(ld, 'selection', None) or type(ld).__name__}")
        try:
            from fea_core.loads_text import load_text
            text = load_text(ld, m.study, num=fmt)
        except Exception:  # noqa: BLE001
            text = m.describe_load(ld)
        self._group(tr("Magnitude and direction"))
        self._prop(tr("Loads"), text)
        vec = getattr(ld, "vector", None)
        if vec is not None and getattr(ld, "kind", "") == "force":
            import numpy as np
            self._prop(tr("Coord. system"), ld.csys or tr("global axes (x, y, z)"))
            for i, ax in enumerate("XYZ"):
                sp = QDoubleSpinBox()
                sp.setRange(-1e9, 1e9)
                sp.setDecimals(3)
                sp.setSuffix(" N")
                sp.setValue(float(vec[i]))
                sp.setToolTip(tr("Force component; the change is saved when leaving the field"))
                sp.editingFinished.connect(lambda box=sp, k=i, obj=ld: self._set_component(obj, k, box.value()))
                self._prop(ax, sp)
            self._prop(tr("Resultant"), f"{fmt(float(np.linalg.norm(vec)), 5)} N")
        self._prop(tr("Load case"), lc.name)
        self._where(getattr(ld, "selection", None))

    def _set_attr(self, obj, attr, value):
        if getattr(obj, attr) != value:
            setattr(obj, attr, value)
            self.main.changed()

    def _set_component(self, obj, k, v):
        if float(obj.vector[k]) != v:
            obj.vector[k] = v
            self.main.changed()

    # ------------------------------------------------------------------ calculation
    def _fill_solve(self, _):
        m = self.main
        st = m.study
        res = m.results
        self.header.setText(f"{tr('Properties')} – {tr('Calculation')}")
        self._group(tr("Analysis progress"))
        if m.busy():
            self._prop(tr("State"), tr("running…"))
        elif res is not None:
            dt = getattr(res, "_gui_solve_s", None)
            self._prop(tr("State"), "√ " + tr("finished") + (f" · {fmt(dt, 3)} s" if dt is not None else ""))
        else:
            self._prop(tr("State"), tr("not solved yet"))
        for key, sec in ((res.info.get("timings") or {}) if res is not None else {}).items():
            self._prop(tr(PHASE_NAMES.get(key, key)), f"√ {fmt(sec, 2)} s")
        self._group(tr("Solver"))
        info = res.info if res is not None else {}
        self._prop(tr("Software"), str(info.get("backend") or st.solver.backend))
        self._prop(tr("System"), str(info.get("ccx_solver") or st.solver.solver))
        mesh = res.mesh if res is not None and len(res.mesh.nodes) else m.mesh
        if mesh is not None and len(mesh.nodes):
            self._prop(tr("Range"), f"{_int(len(mesh.nodes) * 3)} DOF · {_int(len(mesh.tets))} {tr('elements')}")
            sj = (mesh.report or {}).get("min_scaled_jacobian")
            if sj is not None:
                self._prop(tr("Mesh quality"), ("√ " if sj >= 0.2 else "! ") + f"min. SJ {fmt(sj, 3)}")
        self._prop(tr("Mesh settings"), f"{'tet10' if st.mesh.order == 2 else 'tet4'} · {fmt(st.mesh.size, 3)} mm"
                  + (f" · {tr('local refinements')}: {len(st.mesh.local)}" if st.mesh.local else ""))
        if res is not None:
            imb = [s.equilibrium["imbalance"] for s in res.steps if s.equilibrium and "imbalance" in s.equilibrium]
            if imb:
                from .widgets import sci
                self._prop(tr("Equilibrium"), f"√ {sci(max(imb))}" if max(imb) <= 1e-3 else f"× {sci(max(imb))}")
