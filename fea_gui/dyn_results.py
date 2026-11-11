"""Dynamics and contact results in the Results step (F8): items of the result menu (data "dyn:…"), fields for the colour
map (harmonic at a frequency and phase, envelopes, transient in time, RMS / damage of random vibrations, shock), the
Dynamics plots tab (FRF, time course, response PSD, pulse SRS, ASD, mode table with effective mass) and the Contacts tab
(forces and faces of the pairs, force–displacement curve, partial-result label).

The fields are computed from the modal basis on request (fea_core.dynamics.results.DynamicsResults – harmonic_field,
harmonic_amplitude, transient_field); the GUI computes nothing itself except von Mises from the components.
"""
from __future__ import annotations

from html import escape

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from .widgets import LazyPlot, fmt, help_tip, table

G_MM = 9810.0
UNCERTAIN = "Damping is an uncertain input (sensitivity ±50 %) – the response at resonance is inversely proportional to damping."
COMP = ["x", "y", "z"]

# quantities by the kind of dynamic result: (name, unit)
Q_HARM = [("Von Mises – amplitude over phase", "MPa"), ("Displacement – amplitude |u|", "mm"),
          ("Von Mises at phase φ", "MPa"), ("Total displacement at phase φ", "mm"), ("Displacement X at phase φ", "mm"),
          ("Displacement Y at phase φ", "mm"), ("Displacement Z at phase φ", "mm")]
Q_HARM_ENV = [("Von Mises envelope (over frequencies)", "MPa"), ("Displacement envelope (over frequencies)", "mm"),
              ("Frequency of maximum stress", "Hz"), ("Frequency of the displacement maximum", "Hz")]
Q_SWEEP = [("Sweep damage D", "–")]
Q_TR = [("von Mises", "MPa"), ("Total displacement", "mm"), ("Displacement X", "mm"), ("Displacement Y", "mm"), ("Displacement Z", "mm")]
Q_TR_ENV = [("Von Mises envelope (over time)", "MPa"), ("Displacement envelope (over time)", "mm"),
            ("Time of maximum stress", "s"), ("Time of maximum displacement", "s")]
Q_RAND = [("Von Mises RMS (1σ, Segalman)", "MPa"), ("Von Mises 3σ (informative)", "MPa"), ("Total RMS displacement", "mm"),
          ("RMS displacement X", "mm"), ("RMS displacement Y", "mm"), ("RMS displacement Z", "mm"), ("Damage D (spectral fatigue)", "–"),
          ("Life", "h")]
Q_SHOCK = [("Von Mises (mode combination)", "MPa"), ("Total displacement (combination)", "mm"), ("Displacement X (combination)", "mm"),
           ("Displacement Y (combination)", "mm"), ("Displacement Z (combination)", "mm")]
Q_SHOCK_ENV = [("Von Mises envelope (over time)", "MPa"), ("Displacement envelope (over time)", "mm")]
Q_MODE = [("Shape – displacement (normalized, max 1)", "–"), ("Shape – von Mises (normalized, max 1)", "–")]
QUANTS = {"harm": Q_HARM, "harm_env": Q_HARM_ENV, "sweep": Q_SWEEP, "tr": Q_TR, "tr_env": Q_TR_ENV, "rand": Q_RAND,
          "shock": Q_SHOCK, "shock_env": Q_SHOCK_ENV, "mode": Q_MODE}
NO_DEFORM = {"harm_env", "sweep", "tr_env", "rand", "shock", "shock_env"}


def _vm(S: np.ndarray) -> np.ndarray:
    from fea_core.dynamics.response import P_VM
    return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", S, P_VM, S), 0.0))


def items(res) -> list[tuple[str, str]]:
    """Items of the result menu (text, data) for Results.dynamics."""
    d = getattr(res, "dynamics", None)
    if d is None:
        return []
    out = []
    if d.harmonic is not None:
        out.append(("Dynamics: harmonic – at frequency", "dyn:harm:0"))
        out.append(("Dynamics: harmonic – envelope", "dyn:harm_env:0"))
        if d.harmonic.get("sweep"):
            out.append(("Dynamics: sweep – damage", "dyn:sweep:0"))
    if d.transient is not None:
        out.append(("Dynamics: transient – in time", "dyn:tr:0"))
        out.append(("Dynamics: transient – envelope", "dyn:tr_env:0"))
    if d.random is not None:
        out.append(("Dynamics: random vibration – RMS, 3σ, damage", "dyn:rand:0"))
    if d.shock is not None:
        comb = str((d.shock.get("summary") or {}).get("combine", "srss")).upper()
        out.append((f"Dynamics: shock – {comb}", "dyn:shock:0"))
        if d.shock.get("time"):
            out.append(("Dynamics: shock – transient envelope", "dyn:shock_env:0"))
    for k in range(d.K):
        out.append((f"Dynamics: mode {k + 1} – {fmt(float(d.freq[k]), 5)} Hz", f"dyn:mode:{k}"))
    return out


def axis(d, sub) -> tuple[np.ndarray, str, str] | None:
    """Slider axis (values, name, unit): harmonic frequency, transient time; otherwise None."""
    if sub == "harm" and d.harmonic is not None:
        return np.asarray(d.harmonic["f"], float), "Frequency", "Hz"
    if sub == "tr" and d.transient is not None:
        return np.asarray(d.transient["t"], float), "Time", "s"
    return None


def has_base(d, sub) -> bool:
    r = d.harmonic if sub == "harm" else d.transient if sub == "tr" else None
    return r is not None and "ub" in r and np.abs(np.asarray(r["ub"])).max() > 0


def field(d, sub: str, k: int, idx: int, qi: int, phase: float = 0.0, absolute: bool = False):
    """(values per node (N,), displacements for the deformation (N, 3) | None, name, unit)."""
    q = QUANTS[sub]
    name, unit = q[min(max(qi, 0), len(q) - 1)]
    U = None
    if sub == "harm":
        h = d.harmonic
        i = min(max(idx, 0), len(h["f"]) - 1)
        U, S = d.harmonic_field(i, phase, absolute)
        if name.startswith("Von Mises – amplitude"):
            vals = d.harmonic_amplitude(i)[1]
        elif name.startswith("Displacement – amplitude"):
            vals = d.harmonic_amplitude(i)[0]
        elif name.startswith("Von Mises at"):
            vals = _vm(S)
        elif name.startswith("Total displacement"):
            vals = np.linalg.norm(U, axis=1)
        else:
            vals = U[:, "XYZ".index(name.split()[1])]
    elif sub == "harm_env":
        h = d.harmonic
        f = np.asarray(h["f"], float)
        vals = {0: h["env_S"], 1: h["env_U"], 2: f[np.asarray(h["env_S_at"], int)],
                3: f[np.asarray(h["env_U_at"], int)]}[q.index((name, unit))]
    elif sub == "sweep":
        vals = d.harmonic["sweep"]["D"]
    elif sub == "tr":
        tr = d.transient
        i = min(max(idx, 0), len(tr["t"]) - 1)
        U, S = d.transient_field(i, absolute)
        if name == "von Mises":
            vals = _vm(S)
        elif name == "Total displacement":
            vals = np.linalg.norm(U, axis=1)
        else:
            vals = U[:, "XYZ".index(name[-1])]
    elif sub == "tr_env":
        tr = d.transient
        te = np.asarray(tr["t_env"], float)
        vals = {0: tr["env_S"], 1: tr["env_U"], 2: te[np.asarray(tr["env_S_at"], int)],
                3: te[np.asarray(tr["env_U_at"], int)]}[q.index((name, unit))]
    elif sub == "rand":
        r = d.random
        j = q.index((name, unit))
        Ur = np.asarray(r["U_rms"], float)
        vals = {0: lambda: r["S_rms"], 1: lambda: 3.0 * np.asarray(r["S_rms"], float),
                2: lambda: np.linalg.norm(Ur, axis=1), 3: lambda: Ur[:, 0], 4: lambda: Ur[:, 1],
                5: lambda: Ur[:, 2], 6: lambda: r["D"],
                7: lambda: np.minimum(np.nan_to_num(np.asarray(r["life_h"], float), nan=1e12, posinf=1e12), 1e12)}[j]()
    elif sub == "shock":
        s = d.shock
        Us = np.asarray(s["U"], float)
        j = q.index((name, unit))
        vals = s["mises"] if j == 0 else np.linalg.norm(Us, axis=1) if j == 1 else Us[:, j - 2]
    elif sub == "shock_env":
        t = d.shock["time"]
        vals = t["env_S"] if name.startswith("Envelope von") else t["env_U"]
    else:                                               # mode
        kk = min(max(k, 0), d.K - 1)
        U = np.asarray(d.shapes[kk], float)
        um = np.linalg.norm(U, axis=1)
        U = U / max(float(um.max()), 1e-300)
        if name.startswith("Mode – displacement"):
            vals = np.linalg.norm(U, axis=1)
        else:
            v = _vm(np.asarray(d.stress[kk], float))
            vals = v / max(float(v.max()), 1e-300)
    return np.asarray(vals, float), U, name, unit


def info_html(d, sub: str, k: int, idx: int, phase: float, absolute: bool, vals, name, unit, scale) -> str:
    """Summary of the dynamic result for the Results panel."""
    rows = []
    s = d.summary()
    if sub == "harm":
        f = float(d.harmonic["f"][min(idx, len(d.harmonic["f"]) - 1)])
        rows.append(f"<b>Harmonic response – f = {fmt(f, 5)} Hz</b>, phase φ = {fmt(phase, 4)}°"
                    + (" (absolute displacement)" if absolute else " (displacement relative to the base)"))
        rows.append("Excitation: " + escape("; ".join(str(x) for x in d.harmonic.get("labels", []))))
    elif sub == "harm_env":
        h = s.get("harmonic", {})
        rows.append(f"<b>Harmonic response – envelope over {h.get('points', '?')} frequencies</b> {fmt(h.get('f_range_Hz', [0, 0])[0], 4)}–{fmt(h.get('f_range_Hz', [0, 0])[1], 4)} Hz")
        rows.append(f"Max. displacement {fmt(h.get('max_displacement_mm'))} mm, max. von Mises {fmt(h.get('max_mises_MPa'))} MPa at {fmt(h.get('f_at_max_mises_Hz'), 5)} Hz")
    elif sub == "sweep":
        sw = (s.get("harmonic") or {}).get("sweep_fatigue") or {}
        rows.append("<b>Fatigue from sine sweep</b> (dn/df = 60/(R·ln 2) cycles/Hz per sweep)")
        rows += [f"{escape(str(k_))}: {fmt(v) if isinstance(v, (int, float)) else escape(str(v))}" for k_, v in sw.items() if k_ != "at"]
    elif sub == "tr":
        t = float(d.transient["t"][min(idx, len(d.transient["t"]) - 1)])
        rows.append(f"<b>Transient response – t = {fmt(t, 5)} s</b>"
                    + (" (absolute displacement)" if absolute else ""))
        rows.append("Excitation: " + escape("; ".join(str(x) for x in d.transient.get("labels", []))))
        st_ = (s.get("transient") or {}).get("settle_s") or {}
        if st_:
            rows.append("Settling time at the probes: " + ", ".join(
                f"{escape(str(n))}: {fmt(v, 4) + ' s' if v is not None else 'does not settle'}" for n, v in st_.items()))
    elif sub == "tr_env":
        tr = s.get("transient", {})
        rows.append(f"<b>Transient response – envelope</b> up to {fmt(tr.get('t_end_s'), 4)} s ({tr.get('steps')} steps)")
        rows.append(f"Max. displacement {fmt(tr.get('max_displacement_mm'))} mm, max. von Mises {fmt(tr.get('max_mises_MPa'))} MPa at time {fmt(tr.get('t_at_max_mises_s'), 5)} s")
    elif sub == "rand":
        r = s.get("random", {})
        rows.append(f"<b>Random vibration</b> – input g_rms {fmt(r.get('g_rms_input'), 4)} g, duration {fmt(r.get('duration_h'))} h, method {escape(str(r.get('method', '')))}")
        rows.append(f"Max. RMS displacement {fmt(r.get('max_U_rms_mm'))} mm, von Mises RMS "
                    f"{fmt(r.get('max_mises_rms_MPa'))} MPa, 3σ {fmt(r.get('max_mises_3sigma_MPa'))} MPa")
        rows.append(f"Max. damage D = <b>{fmt(r.get('max_damage'), 4)}</b>, min. life <b>{fmt(r.get('min_life_h'), 4)} h</b>")
        if r.get("note"):
            rows.append(f"<small>{escape(str(r['note']))}</small>")
    elif sub in ("shock", "shock_env"):
        r = s.get("shock", {})
        rows.append(f"<b>Shock</b> – pulse {escape(str(r.get('pulse', '')))}, A {fmt(r.get('A_g'))} g, τ {fmt(1000 * float(r.get('tau_s') or 0), 4)} ms, combination {str(r.get('combine', '')).upper()}, ZPA {fmt(r.get('zpa_g'), 4)} g")
        rows.append(f"Max. displacement {fmt(r.get('max_U_mm'))} mm, max. von Mises {fmt(r.get('max_mises_MPa'))} MPa"
                    + (f"; transient response to the pulse {fmt(r['transient_max_mises_MPa'])} MPa"
                       if r.get("transient_max_mises_MPa") is not None else ""))
        if r.get("note"):
            rows.append(f"<small>{escape(str(r['note']))}</small>")
    else:
        kk = min(max(k, 0), d.K - 1)
        rows.append(f"<b>Mode {kk + 1}</b> of the dynamics modal basis – f = {fmt(float(d.freq[kk]), 6)} Hz, ζ = {fmt(100 * float(d.zeta[kk]), 3)} %")
        if d.model_mass:
            em = np.asarray(d.eff_mass, float)[kk, :3] / d.model_mass * 100
            rows.append("Effective mass x / y / z: " + " / ".join(fmt(float(v), 3) for v in em) + " %")
    v = np.asarray(vals, float)
    fin = v[np.isfinite(v)]
    if len(fin):
        rows.append(f"{escape(str(name))}: max {fmt(float(fin.max()))} {escape(str(unit))}, min {fmt(float(fin.min()))} {escape(str(unit))}")
    if scale:
        rows.append(f"Deformation scale: {fmt(scale, 3)}×")
    rows.append(f"Model: {d.K} modes, mass {fmt(1000 * d.model_mass)} kg")
    rows.append(f"<span style='color:#d23b3b'><b>! {UNCERTAIN}</b></span>")
    warn = [w for w in d.warnings if "uncertain input" not in w]
    if warn:
        rows.append("<span style='color:#b36b00'>! " + "<br>! ".join(escape(str(w)) for w in warn) + "</span>")
    return "<br>".join(rows)


# ====================================================================== Dynamics tab (plots)
PLOTS = [("frf_amp", "FRF – amplitude"), ("frf_phase", "FRF – phase"), ("time", "Time history (transient)"),
         ("shock_time", "Time history (shock)"), ("psd", "Response PSD (random vibration)"), ("asd", "Input ASD"),
         ("srs", "Pulse SRS (shock)"), ("reaction", "Reactions – harmonic (FRF)")]
COMPS = [("mag", "|u| (magnitude)"), ("x", "u_x"), ("y", "u_y"), ("z", "u_z"), ("vm", "σ von Mises"),
         ("ax", "a_x"), ("ay", "a_y"), ("az", "a_z")]


class DynResultsBox(QWidget):
    """Dynamics plots at the probes + the mode table of the modal basis with effective mass."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.d = None
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 4, 2, 2)
        h = QHBoxLayout()
        self.kind = QComboBox()
        for k, t in PLOTS:
            self.kind.addItem(t, k)
        help_tip(self.kind, "Probe plot type (FRF, time history, PSD, SRS)",
                 "Plot type: FRF (amplitude / phase vs. frequency) at probe, time history, response PSD, input ASD, shock response spectrum (SRS) of pulse, harmonic reaction resultant.")
        self.probe = QComboBox()
        help_tip(self.probe, "Output probe (max σ = node with peak stress envelope)",
                 "Probe (Dynamics step; \"max σ\" = node with maximum stress envelope, added automatically).")
        self.comp = QComboBox()
        for k, t in COMPS:
            self.comp.addItem(t, k)
        help_tip(self.comp, "Displayed component (displacement, stress, acceleration)",
                 "Component: displacement magnitude, displacement components, von Mises, acceleration (time domain only – absolute).")
        self.log = QCheckBox("log")
        self.log.setChecked(True)
        help_tip(self.log, "Logarithmic axes for frequency plots",
                 "Logarithmic axes for frequency plots (FRF, PSD, ASD, SRS); time history is always linear.")
        for w in (self.kind, self.probe, self.comp):
            h.addWidget(w, 1)
        h.addWidget(self.log)
        v.addLayout(h)
        self.plot = LazyPlot()
        self.plot.setToolTip("Chart of the selected quantity in the probe")
        v.addWidget(self.plot)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setTextFormat(Qt.RichText)
        v.addWidget(self.note)
        self.modes = table(["Shape", "f [Hz]", "ζ [%]", "m_x [%]", "m_y [%]", "m_z [%]", "Σ m_x", "Σ m_y", "Σ m_z"], 150)
        self.modes.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        help_tip(self.modes, "Modal basis: natural frequencies and effective mass fractions",
                 "Dynamics modal basis: natural frequencies, damping and effective mass fractions (Σ in excited direction should be ≥ 90%).<br>Click a row to display shape in 3D.")
        self.modes.currentCellChanged.connect(lambda r, *_: self.on_mode(r))
        v.addWidget(self.modes)
        self.kind.currentIndexChanged.connect(lambda _: self.draw())
        self.probe.currentIndexChanged.connect(lambda _: self.draw())
        self.comp.currentIndexChanged.connect(lambda _: self.draw())
        self.log.toggled.connect(lambda _: self.draw())

    def fill(self, results):
        d = getattr(results, "dynamics", None) if results is not None else None
        self.d = d
        t = self.modes
        t.blockSignals(True)
        t.setRowCount(0)
        if d is not None:
            em = np.asarray(d.eff_mass, float)[:, :3]
            m = d.model_mass or 1.0
            cum = np.cumsum(em, axis=0)
            for k in range(d.K):
                t.insertRow(k)
                vals = [str(k + 1), fmt(float(d.freq[k]), 6), fmt(100 * float(d.zeta[k]), 3)] + \
                    [fmt(100 * float(x) / m, 3) for x in em[k]] + [fmt(100 * float(x) / m, 3) for x in cum[k]]
                for c, txt in enumerate(vals):
                    it = QTableWidgetItem(txt)
                    if c:
                        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    t.setItem(k, c, it)
        t.blockSignals(False)
        cur = self.probe.currentText()
        self.probe.blockSignals(True)
        self.probe.clear()
        names = []
        if d is not None:
            for r in (d.harmonic, d.transient, d.random, (d.shock or {}).get("time")):
                if r:
                    names += [n for n in (r.get("probes") or {}) if n not in names]
        self.probe.addItems(names)
        if cur in names:
            self.probe.setCurrentText(cur)
        self.probe.blockSignals(False)
        avail = {"frf_amp": d is not None and d.harmonic is not None,
                 "frf_phase": d is not None and d.harmonic is not None,
                 "reaction": d is not None and d.harmonic is not None,
                 "time": d is not None and d.transient is not None,
                 "shock_time": d is not None and bool((d.shock or {}).get("time")),
                 "psd": d is not None and d.random is not None, "asd": d is not None and d.random is not None,
                 "srs": d is not None and (d.shock or {}).get("srs") is not None}
        model = self.kind.model()
        first = None
        for i, (k, _) in enumerate(PLOTS):
            model.item(i).setEnabled(bool(avail.get(k)))
            if avail.get(k) and first is None:
                first = i
        if first is not None and not avail.get(self.kind.currentData()):
            self.kind.setCurrentIndex(first)
        if d is not None and d.model_mass:
            s = np.asarray(d.eff_mass, float)[:, :3].sum(axis=0) / d.model_mass * 100
            self.note.setText(f"Σ effective mass of {d.K} modes: x {fmt(float(s[0]), 3)} %, y {fmt(float(s[1]), 3)} %, z {fmt(float(s[2]), 3)} % (model mass {fmt(1000 * d.model_mass)} kg). <span style='color:#d23b3b'>{UNCERTAIN}</span>")
        self.draw()

    def on_mode(self, r):
        if self.d is not None and 0 <= r < self.d.K:
            self.win.select_result(f"dyn:mode:{r}")

    def series(self):
        """(x, y, x label, y label, log x) for the current choice, or None."""
        d, kind, pn, comp = self.d, self.kind.currentData(), self.probe.currentText(), self.comp.currentData()
        if d is None:
            return None
        if kind in ("frf_amp", "frf_phase", "reaction") and d.harmonic is not None:
            h = d.harmonic
            f = np.asarray(h["f"], float)
            if kind == "reaction":
                R = np.asarray(h["reaction"])
                return f, np.linalg.norm(np.abs(R), axis=1), "f [Hz]", "|R| [N]", True
            p = (h.get("probes") or {}).get(pn)
            if p is None:
                return None
            U = np.asarray(p["U"])
            if comp == "vm" and "S" in p:
                from fea_core.dynamics.response import P_VM, max_over_phase
                S = np.asarray(p["S"])
                y = max_over_phase(S.real, -S.imag, P_VM)
                if kind == "frf_phase":
                    return None
                return f, y, "f [Hz]", "σ_vM amplitude [MPa]", True
            if comp == "mag" or comp.startswith("a") or comp == "vm":
                if kind == "frf_phase":
                    j = int(np.argmax(np.abs(U).max(axis=0)))
                    return f, np.degrees(np.angle(U[:, j])), "f [Hz]", f"phase u_{COMP[j]} [°]", True
                return f, np.sqrt((np.abs(U) ** 2).sum(axis=1)), "f [Hz]", "|u| [mm]", True
            j = COMP.index(comp)
            if kind == "frf_phase":
                return f, np.degrees(np.angle(U[:, j])), "f [Hz]", f"phase u_{comp} [°]", True
            return f, np.abs(U[:, j]), "f [Hz]", f"|u_{comp}| [mm]", True
        if kind in ("time", "shock_time"):
            r = d.transient if kind == "time" else (d.shock or {}).get("time")
            if not r:
                return None
            p = (r.get("probes") or {}).get(pn)
            if p is None:
                return None
            t = np.asarray(r["t"], float)
            U = np.asarray(p["U"], float)
            if comp == "vm" and "S" in p:
                return t, _vm(np.asarray(p["S"], float)), "t [s]", "σ_vM [MPa]", False
            if comp.startswith("a") and "A" in p:
                j = COMP.index(comp[1])
                return t, np.asarray(p["A"], float)[:, j] / G_MM, "t [s]", f"a_{comp[1]} [g]", False
            if comp in COMP:
                j = COMP.index(comp)
                return t, U[:, j], "t [s]", f"u_{comp} [mm]", False
            return t, np.linalg.norm(U, axis=1), "t [s]", "|u| [mm]", False
        if kind == "psd" and d.random is not None:
            p = (d.random.get("probes") or {}).get(pn)
            if p is None:
                return None
            f = np.asarray(d.random["f"], float)
            if comp.startswith("a"):
                j = COMP.index(comp[1])
                return f, np.asarray(p["A_psd_g2"], float)[:, j], "f [Hz]", f"PSD a_{comp[1]} [g²/Hz]", True
            U = np.asarray(p["U_psd"], float)
            if comp in COMP:
                j = COMP.index(comp)
                return f, U[:, j], "f [Hz]", f"PSD u_{comp} [mm²/Hz]", True
            return f, U.sum(axis=1), "f [Hz]", "PSD |u| [mm²/Hz]", True
        if kind == "asd" and d.random is not None:
            return np.asarray(d.random["f"], float), np.asarray(d.random["asd_g2"], float), "f [Hz]", "ASD [g²/Hz]", True
        if kind == "srs" and (d.shock or {}).get("srs") is not None:
            s = d.shock
            return np.asarray(s["srs_f"], float), np.asarray(s["srs"], float) / G_MM, "f [Hz]", "SRS [g]", True
        return None

    def draw(self):
        if not self.isVisible() and self.plot.canvas is None:
            return
        ser = self.series()
        ax = self.plot.axes()
        if ser is None:
            ax.text(0.5, 0.5, "Chart not available (choose another probe / component)", ha="center", va="center",
                    transform=ax.transAxes)
            self.plot.canvas.draw_idle()
            return
        x, y, xl, yl, logx = ser
        ax.plot(x, y, color="#1f77b4", lw=1.3)
        kind = self.kind.currentData()
        if kind == "srs" and self.d is not None:
            s = self.d.shock
            ax.plot(self.d.freq, np.asarray(s["Sa"], float) / G_MM, "o", ms=4, color="#d62728", label="modes")
            ax.legend(fontsize=8)
        if kind in ("frf_amp", "frf_phase", "psd", "reaction") and self.d is not None:
            for fn in self.d.freq:
                if x.min() <= fn <= x.max():
                    ax.axvline(float(fn), color="#999999", lw=0.6, ls=":")
        if self.log.isChecked() and logx:            # log only for the frequency plots (time course linear)
            ax.set_xscale("log")
            if kind != "frf_phase" and np.nanmax(y) > 0 and np.all(y >= 0):
                ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, which="both", alpha=0.3)
        self.plot.canvas.draw_idle()

    def showEvent(self, e):
        super().showEvent(e)
        self.draw()


# ====================================================================== Contacts tab
class ContactResultsBox(QWidget):
    """Forces and faces of the contact pairs of the displayed state, partial-result label, force–displacement curve."""

    def __init__(self, win):
        super().__init__()
        self.win = win
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 4, 2, 2)
        self.partial = QLabel("")
        self.partial.setWordWrap(True)
        self.partial.setTextFormat(Qt.RichText)
        v.addWidget(self.partial)
        self.table = table(["Pair", "Fn [N]", "Ft [N]", "Contact area [mm²]", "Fraction [%]", "p_max [MPa]",
                            "p_mean [MPa]", "Max. opening [mm]", "Max. slip [mm]", "μ", "Stick / slip / open"], 140)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        help_tip(self.table, "Forces, areas, and pressures of contact pairs for displayed case",
                 "Contact pairs of displayed case: normal and friction force, contact area and fraction, peak and mean pressure, peak opening and slip, μ and node counts (stick / slip / open). Contact forces are internal – equilibrium is checked via reactions.")
        v.addWidget(self.table)
        h = QHBoxLayout()
        self.target = QComboBox()
        help_tip(self.target, "Rigid coupling for plotting force-displacement curve",
                 "Prescribed motion (rigid remote coupling) whose force-displacement curve is plotted across all sequence steps.")
        h.addWidget(QLabel("Force–travel curve"))
        h.addWidget(self.target, 1)
        v.addLayout(h)
        self.plot = LazyPlot()
        self.plot.setToolTip("Force in the rigid support vs. prescribed displacement / rotation (increment frames)")
        v.addWidget(self.plot)
        self.target.currentIndexChanged.connect(lambda _: self.draw())
        self.res = None

    def fill(self, results, si: int | None):
        self.res = results
        t = self.table
        t.setRowCount(0)
        st = results.steps[si] if results is not None and si is not None and 0 <= si < len(results.steps) else None
        if st is not None and st.partial:
            self.partial.setText("<span style='color:#d23b3b'><b>PARTIAL RESULT</b> – the calculation did not converge, "
                                 "the last converted increment is shown"
                                 + (f" (time {fmt(getattr(st, 'fields_time', None), 4)})"
                                    if getattr(st, "fields_time", None) is not None else "")
                                 + ". Advice in the Messages dock.</span>")
        else:
            self.partial.setText("")
        for name, c in ((st.contacts or {}).items() if st is not None else []):
            r = t.rowCount()
            t.insertRow(r)
            vals = [name, fmt(c.get("Fn_N")), fmt(c.get("Ft_N")), fmt(c.get("area_contact_mm2")),
                    fmt(100 * float(c.get("contact_fraction") or 0), 3), fmt(c.get("p_max_MPa")),
                    fmt(c.get("p_mean_MPa")), fmt(c.get("open_max_mm")), fmt(c.get("slip_max_mm")),
                    fmt(c.get("mu"), 3) if c.get("mu") else "–",
                    f"{c.get('nodes_stick', 0)} / {c.get('nodes_slip', 0)} / {c.get('nodes_open', 0)}"]
            for k, txt in enumerate(vals):
                it = QTableWidgetItem(str(txt))
                if k:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, k, it)
        cur = self.target.currentText()
        self.target.blockSignals(True)
        self.target.clear()
        names = []
        for s in (results.steps if results is not None else []):
            names += [n for n in (s.curves or {}) if n not in names]
        self.target.addItems(names)
        if cur in names:
            self.target.setCurrentText(cur)
        self.target.blockSignals(False)
        self.target.setEnabled(bool(names))
        self.draw()

    def draw(self):
        if self.res is None or (not self.isVisible() and self.plot.canvas is None):
            return
        name = self.target.currentText()
        ax = self.plot.axes()
        us, fs, lab = [], [], ""
        for s in self.res.steps:
            c = (s.curves or {}).get(name)
            if not c:
                continue
            j = int(c.get("dof", 1)) - 1
            u = np.asarray(c.get("u") or [], float)
            F = np.asarray(c.get("F") or [], float)
            n = min(len(u), len(F))
            if n:
                us += list(u[:n, j])
                fs += list(F[:n, j])
            lab = ("displacement" if c.get("kind") == "disp" else "rotation") + f" {COMP[j]}"
        if us:
            ax.plot(us, fs, "o-", ms=3, color="#d62728")
            ax.set_xlabel(f"{lab} [{'mm' if lab.startswith('displacement') else 'rad'}]")
            ax.set_ylabel("support force [N]" if lab.startswith("displacement") else "support moment [N·mm]")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No prescribed motion (the force–travel curve comes from a step sequence\nwith prescribed motion of a rigid remote support)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        self.plot.canvas.draw_idle()

    def showEvent(self, e):
        super().showEvent(e)
        self.draw()
