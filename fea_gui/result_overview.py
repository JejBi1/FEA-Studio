""""Assessment" card at the top of the Results panel (Studio × Tables layout, stage 4).

The verdict of the whole study: the smallest safety to Re outside singularities over all mechanical states against the required
safety (Study.assessment["required_safety"], default 1.5; the field on the card changes it – the results are not discarded),
the deciding state and part; below it the numbers of the displayed state: max. stress outside singularities, max. stress in total (the difference =
singularity), max. displacement, resultant of the reactions. The numbers come from `table_dock.step_summaries` – the same as in the bottom
table. Parts without a yield strength (plastics, 3D printing) are not assessed here – printing has its own safety Tsai-Wu / Hill.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout)

from .table_dock import required_safety, step_summaries
from fea_core.i18n import tr

from .widgets import fmt, help_tip


class ResultOverview(QFrame):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setObjectName("overviewCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)
        head = QHBoxLayout()
        t = QLabel("Assessment")
        t.setObjectName("overviewTitle")
        head.addWidget(t)
        head.addStretch(1)
        lbl = QLabel("required ≥")
        lbl.setProperty("muted", True)
        head.addWidget(lbl)
        self.req = QDoubleSpinBox()
        self.req.setRange(1.0, 10.0)
        self.req.setSingleStep(0.1)
        self.req.setDecimals(2)
        self.req.setValue(1.5)
        self.req.setFixedWidth(76)
        help_tip(self.req, "Required safety factor to yield strength Re (outside singularities)",
                 "Required safety factor to yield strength Re = Re / σ von Mises at the most stressed location outside singularities (sharp edges, fixed supports, point forces). Saved in the study; results are not recomputed. Typically 1.5 (static steel design), per the standard or specification.")
        self.req.valueChanged.connect(self._on_req)
        head.addWidget(self.req)
        lay.addLayout(head)
        row = QHBoxLayout()
        self.k = QLabel("–")
        self.k.setObjectName("overviewValue")
        row.addWidget(self.k)
        col = QVBoxLayout()
        col.setSpacing(0)
        self.verdict = QLabel("")
        self.verdict.setObjectName("overviewVerdict")
        self.where = QLabel("")
        self.where.setProperty("muted", True)
        self.where.setWordWrap(True)
        col.addWidget(self.verdict)
        col.addWidget(self.where)
        row.addLayout(col, 1)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setObjectName("overviewBar")
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 200)
        self.bar.setFixedHeight(6)
        self.bar.setToolTip("Safety factor / required safety factor (full bar = 2× the requirement, half = exactly the requirement)")
        lay.addWidget(self.bar)
        self.step_title = QLabel("")
        self.step_title.setProperty("muted", True)
        lay.addWidget(self.step_title)
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(2)
        self.kpi = {}
        self.kpi_label = {}
        for r, (key, text, tip) in enumerate((
                ("real", "Max. σ outside singularities", "Highest von Mises at a location that is not a singularity"),
                ("vmax", "Max. σ overall", "Highest von Mises in the model including singularities (sharp edges, fixed supports)"),
                ("umax", "Max. displacement", "Largest total displacement"),
                ("reaction", "Total reaction", "Resultant of all support reactions and force equilibrium"),
                ("joints", "Joints", "Largest utilization of bolts and welds (VDI 2230, EN 1993-1-8)"))):
            a = QLabel(text)
            a.setProperty("muted", True)
            a.setToolTip(tip)
            b = QLabel("–")
            b.setTextInteractionFlags(Qt.TextSelectableByMouse)
            b.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            g.addWidget(a, r, 0)
            g.addWidget(b, r, 1)
            self.kpi[key] = b
            self.kpi_label[key] = a
        g.setColumnStretch(0, 1)
        lay.addLayout(g)
        self.setToolTip("Static assessment: safety factor to Re outside singularities against the requirement")
        for w in self.findChildren(QLabel):            # names of parts / states from a foreign CAD: never as HTML (<img src=\\…>)
            w.setTextFormat(Qt.PlainText)

    # ------------------------------------------------------------------
    def _on_req(self, v: float):
        st = self.main.study
        if st is None or getattr(self, "_filling", False):
            return
        if st.assessment is None:
            st.assessment = {}
        st.assessment["required_safety"] = round(float(v), 3)
        self.main.dirty = True                         # assessment change – the mesh and the results are not discarded
        self.main.update_title()
        self.refresh()
        if hasattr(self.main, "table_dock"):
            self.main.table_dock.refresh()

    def _set_state(self, state: str):
        for w in (self.verdict, self.bar, self.k):
            w.setProperty("state", state)
            w.style().unpolish(w)
            w.style().polish(w)

    def refresh(self):
        m = self.main
        rows = step_summaries(m) if m.study is not None else []
        self.setVisible(bool(rows))
        if not rows:
            return
        self._filling = True
        self.req.setValue(required_safety(m.study))
        self._filling = False
        req = self.req.value()
        judged = [d for d in rows if d["k"] is not None]
        if judged:
            worst = min(judged, key=lambda d: d["k"])
            k = worst["k"]
            ok = k >= req
            self.k.setText(fmt(k, 3))
            self.verdict.setText("Passes" if ok else "Fails")
            self.where.setText(f"governed by {worst['name']}" + (f" · {', '.join(worst['parts'])}" if worst["parts"] else "")
                               + f" · {len(judged)} of {len(rows)} states assessed")
            self.bar.setValue(int(min(200.0, 100.0 * k / req)))
            self._set_state("ok" if ok else "fail")
        else:
            self.k.setText("–")
            self.verdict.setText("Cannot be assessed")
            self.where.setText("no part with yield strength Re at the maxima (plastics and 3D printing have their own assessment)")
            self.bar.setValue(0)
            self._set_state("none")
        cur = self._current(rows)
        self.step_title.setText(f"Displayed case: {cur['name']}" if cur else "")
        if cur:
            self.kpi["real"].setText(f"{fmt(cur['real'], 4)} MPa" + (f" · {', '.join(cur['parts'])}" if cur["parts"] else "")
                                     if cur["real"] is not None else "–")
            sing = cur["real"] is not None and cur["vmax"] > cur["real"] * 1.001
            self.kpi["vmax"].setText(f"{fmt(cur['vmax'], 4)} MPa" + (" (singularity)" if sing else ""))
            self.kpi["umax"].setText(f"{fmt(cur['umax'], 4)} mm")
            imb = (m.results.steps[cur["index"]].equilibrium or {}).get("imbalance")
            from .widgets import sci
            self.kpi["reaction"].setText((f"{fmt(cur['reaction'], 4)} N" if cur["reaction"] is not None else "–")
                                         + (f" · {tr('equilibrium')} {sci(imb)}" if imb is not None else ""))
        a = getattr(getattr(m, "joints_panel", None), "assessment", None)
        checks = a.all() if a is not None else []
        if checks:
            worst = max(checks, key=lambda c: c.utilization)
            self.kpi["joints"].setText(f"{fmt(worst.utilization, 3)} · {worst.name}")
        self.kpi["joints"].setVisible(bool(checks))
        self.kpi_label["joints"].setVisible(bool(checks))

    def _current(self, rows):
        data = self.main.r_step.currentData() if hasattr(self.main, "r_step") else None
        if isinstance(data, str) and data.startswith("step:"):
            i = int(data.split(":")[1])
            for d in rows:
                if d["index"] == i:
                    return d
        return rows[0] if rows else None
