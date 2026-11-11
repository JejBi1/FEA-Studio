"""Frame results in the GUI: deformed shape coloured by σ_eq / displacement / utilisation, diagrams of the internal forces
N, Vy, Vz, T, My, Mz (perpendicular to the member in its plane), mode shapes of natural vibration and buckling of the members, table of extremes per member
and the EN 1993-1-1 assessment (fea_core.beams.assess) with the detail of the selected member's rows.
"""
from __future__ import annotations

import html
import json

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QLabel, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget

from .safe_browser import SafeTextBrowser
from .contour import BAND_COLORS, band_edges
from .widgets import fmt, help_tip, table

Q_PREFIX = "Members – "
FRAME_QUANTITIES = [("Members – stress σ_eq", "MPa"), ("Members – displacement |u|", "mm"), ("Members – N", "N"),
                    ("Members – Vy", "N"), ("Members – Vz", "N"), ("Members – T", "N·mm"), ("Members – My", "N·mm"),
                    ("Members – Mz", "N·mm"), ("Members – EN 1993 utilization", "–")]
DIAGRAMS = {"N": 2, "Vy": 1, "Vz": 2, "T": 2, "My": 2, "Mz": 1}     # direction of the diagram: local axis (1 = y, 2 = z)
MODE_FRAME_Q = ("Member shape (normalized)", "–")


def band_colors(n: int, reverse: bool = False) -> list:
    cols = BAND_COLORS[::-1] if reverse else BAND_COLORS
    return [cols[round(k * (len(cols) - 1) / max(n - 1, 1))] for k in range(n)]


def is_frame_quantity(name: str) -> bool:
    return name.startswith(Q_PREFIX)


def frame_info(res) -> dict | None:
    return (res.info or {}).get("frame") if res is not None else None


def frame_fe(win, res):
    """FEM discretisation of the study's members (local axes of the segments for the diagrams) – cache on the results object."""
    from fea_core.beams.fem import build
    from .frame_panel import frame_signature
    sig = frame_signature(win.study)
    c = getattr(res, "_gui_fe", None)
    if c is None or c[0] != sig:
        try:
            fe = build(win.study)
        except Exception:  # noqa: BLE001 – the study changed since the calculation / error → without local axes
            fe = None
        res._gui_fe = c = (sig, fe)
    return c[1]


def frame_assessment(win, res) -> dict | None:
    """EN 1993-1-1 assessment (cache by the assessment inputs); error → {"_error": text}."""
    if res is None or not res.steps or not any(s.frame for s in res.steps):
        return None
    sig = json.dumps(win.study.frame.to_dict(), sort_keys=True, default=str)
    c = getattr(res, "_gui_assess", None)
    if c is None or c[0] != sig:
        from fea_core.beams.assess import assess
        try:
            a = assess(win.study, res)
        except Exception as e:  # noqa: BLE001
            a = {"_error": str(e)}
        res._gui_assess = c = (sig, a)
    return c[1]


def _member_points(md):
    return np.asarray(md["points"], float), np.asarray(md["u"], float)


def step_draw(win, res, st, name, unit, n_bands):
    """Draws the result of the members of state st (quantity name). Returns (title, bounds, colours, probe points, values, point labels,
    deformation scale) or None (state without members).
    """
    fr = st.frame or {}
    members = fr.get("members") or {}
    if not members:
        return None
    v = win.view
    key = name[len(Q_PREFIX):]
    allu = np.vstack([_member_points(md)[1] for md in members.values()])
    scale = win.deformation_scale(allu) if key in ("stress σ_eq", "displacement |u|", "EN 1993 utilization") else 0.0
    segs, vals, pts, pv, plabel = [], [], [], [], []
    if key in DIAGRAMS:
        fe = frame_fe(win, res)
        comp = DIAGRAMS[key]
        vmax = max(float(np.max(np.abs(md[key]))) for md in members.values())
        group = ("T", "My", "Mz") if key in ("T", "My", "Mz") else ("N", "Vy", "Vz")
        ref = max(float(np.max(np.abs(md[k]))) for md in members.values() for k in group)
        zero = vmax <= 1e-6 * ref                      # numerical noise (e.g. My = 1e-9 with bending about Mz only)
        vmax = vmax or 1.0
        Ld = 0.0 if zero else 0.08 * v.model_size() / vmax
        base, ticks = [], []
        for mname, md in members.items():
            P, _ = _member_points(md)
            q = np.zeros(len(P)) if zero else np.asarray(md[key], float)
            n = len(P)
            segi = fe.member_segs.get(mname, []) if fe is not None else []
            D = np.zeros((n, 3))
            for i in range(n):
                j = i // 3
                if j < len(segi):
                    D[i] = fe.segs[segi[j]].R[comp]
                else:                                     # without local axes: perpendicular to the member in the global z
                    t = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
                    t /= max(np.linalg.norm(t), 1e-30)
                    z = np.array([0, 0, 1.0]) - t[2] * t
                    D[i] = z / max(np.linalg.norm(z), 1e-30)
            Q = P + (q * Ld)[:, None] * D
            for i in range(n - 1):
                if np.linalg.norm(P[i + 1] - P[i]) < 1e-9 and np.linalg.norm(Q[i + 1] - Q[i]) < 1e-9:
                    continue
                base.append((P[i], P[i + 1]))
                if max(abs(q[i]), abs(q[i + 1])) < 1e-4 * vmax:     # zero diagram (truss) – member axis only
                    continue
                segs.append((Q[i], Q[i + 1]))
                vals.append(0.5 * (q[i] + q[i + 1]))
            ticks += [(P[i], Q[i]) for i in range(0, n, 3)]
            pts.append(Q)
            pv.append(q)
            plabel += [(mname, s) for s in md["s"]]
        allv = np.concatenate(pv)
        edges = band_edges(float(allv.min()), float(allv.max()), n_bands)
        colors = band_colors(len(edges) - 1)
        v.result_lines(base, color=(110, 115, 125), width=2.0, pickable=False)
        v.result_lines(ticks, color=(150, 155, 165), width=1.0, pickable=False)
        v.result_lines(segs, vals, edges, colors, width=3.5)
        return f"{name} [{unit}]", edges, colors, np.vstack(pts), allv, plabel, 0.0
    util = None
    if key == "EN 1993 utilization":
        a = frame_assessment(win, res) or {}
        util = {k: float(d.get("utilization", np.nan)) for k, d in a.items() if not k.startswith("_")}
    for mname, md in members.items():
        P, U = _member_points(md)
        X = P + scale * U
        if key == "stress σ_eq":
            q = np.asarray(md["sigma_eq"], float)
        elif key == "displacement |u|":
            q = np.linalg.norm(U, axis=1)
        else:
            q = np.full(len(P), util.get(mname, np.nan) if util else np.nan)
        for i in range(len(X) - 1):
            if np.linalg.norm(X[i + 1] - X[i]) < 1e-9:
                continue
            segs.append((X[i], X[i + 1]))
            vals.append(0.5 * (q[i] + q[i + 1]))
        pts.append(X)
        pv.append(q)
        plabel += [(mname, s) for s in md["s"]]
    allv = np.nan_to_num(np.concatenate(pv), nan=0.0)
    vals = np.nan_to_num(np.asarray(vals, float), nan=0.0)
    edges = band_edges(float(allv.min()), float(allv.max()), n_bands)
    colors = band_colors(len(edges) - 1)
    if scale > 0:                                         # outline of the undeformed system
        und = [(a, b) for md in members.values() for a, b in zip(_member_points(md)[0][:-1], _member_points(md)[0][1:])
               if np.linalg.norm(b - a) > 1e-9]
        v.result_lines(und, color=(150, 155, 165), width=1.0, pickable=False)
    v.result_lines(segs, vals, edges, colors, width=5.0)
    return f"{name} [{unit}]", edges, colors, np.vstack(pts), allv, plabel, scale


def overlay(win, res, st, scale):
    """Members of a mixed model over the result of the bodies: deformed axis in grey."""
    members = (st.frame or {}).get("members") or {}
    seg = []
    for md in members.values():
        P, U = _member_points(md)
        X = P + scale * U
        seg += [(a, b) for a, b in zip(X[:-1], X[1:]) if np.linalg.norm(b - a) > 1e-9]
    win.view.result_lines(seg, color=(60, 65, 80), width=3.0, pickable=False)


def mode_geometry(win, res, ms, k):
    """(xyz (n, 3), segments [(a, b)], shape (n, 3)) of the members for mode k, or None. Natural vibration and buckling of the members alone are
    on their own subdivision (`beams.run.eigen_fe`) – the segments from the new discretisation with the same settings.
    """
    fi = frame_info(res)
    if ms.frame_shapes is not None and fi:
        X = np.asarray(fi["xyz"], float)
        sh = np.asarray(ms.frame_shapes[k], float)
        if len(sh) != len(X):
            return None
        segs = [ab for lst in fi["segments"].values() for ab in lst]
        return X, segs, sh
    own = getattr(ms, "frame_shapes_own", None)
    if own is not None and getattr(ms, "frame_xyz", None) is not None:
        from fea_core.beams.run import eigen_fe
        X = np.asarray(ms.frame_xyz, float)
        try:
            fe = eigen_fe(win.study)
        except Exception:  # noqa: BLE001
            return None
        if len(fe.xyz) != len(X):
            return None
        return X, [(sg.a, sg.b) for sg in fe.segs], np.asarray(own[k], float)
    return None


def mode_draw(win, res, ms, k, n_bands):
    g = mode_geometry(win, res, ms, k)
    if g is None:
        return None
    X, segs, sh = g
    scale = win.deformation_scale(sh)
    Y = X + scale * sh
    m = np.linalg.norm(sh, axis=1)
    edges = band_edges(float(m.min()), float(m.max()), n_bands)
    colors = band_colors(len(edges) - 1)
    und = [(X[a], X[b]) for a, b in segs]
    win.view.result_lines(und, color=(150, 155, 165), width=1.0, pickable=False)
    win.view.result_lines([(Y[a], Y[b]) for a, b in segs], [0.5 * (m[a] + m[b]) for a, b in segs], edges, colors,
                          width=5.0)
    return f"{MODE_FRAME_Q[0]} [{MODE_FRAME_Q[1]}]", edges, colors, Y, m, [("", 0.0)] * len(Y), scale


def step_info_html(win, res, st) -> str:
    fr = st.frame or {}
    members = fr.get("members") or {}
    rows = [f"<b>{html.escape(st.name)}</b> – frame"]
    if members:
        u = max((float(np.linalg.norm(np.asarray(md["u"]), axis=1).max()), k) for k, md in members.items())
        s = max((float(np.max(md["sigma_eq"])), k) for k, md in members.items())
        M = max((float(np.max(np.abs(np.r_[md["My"], md["Mz"]]))), k) for k, md in members.items())
        N = max((float(np.max(np.abs(md["N"]))), k) for k, md in members.items())
        rows += [f"Max. displacement: <b>{fmt(u[0])} mm</b> (member {html.escape(u[1])})",
                 f"Max. σ_eq: <b>{fmt(s[0])} MPa</b> (member {html.escape(s[1])}; normal + shear, approximate)",
                 f"Max. |M|: {fmt(M[0] / 1e6)} kN·m (member {html.escape(M[1])}), max. |N|: {fmt(N[0] / 1e3)} kN "
                 f"(member {html.escape(N[1])})"]
    a = frame_assessment(win, res)
    if a and "_error" not in a:
        umax, worst = a.get("_max", (0.0, None))
        if worst:
            d = a[worst]
            col = "#d23b3b" if umax > 1 else "#2e9e4f"
            rows.append(f"EN 1993-1-1 check (all cases): max. utilization <b><span style='color:{col}'>{fmt(umax, 3)}</span></b> – member {html.escape(worst)}, {html.escape(d.get('governing', ''))} (case {html.escape(str(d.get('case', '')))}) – “Members” tab")
    elif a and "_error" in a:
        rows.append(f"<span style='color:#b36b00'>EN 1993 check cannot be performed: {html.escape(a['_error'])}</span>")
    react = fr.get("reactions") or {}
    if react:
        rows.append("<br><b>Support reactions</b> [N, N·mm]:")
        rows.append("<table cellspacing=3><tr><th align=left>node</th>" + "".join(
            f"<th align=right>{c}</th>" for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")) + "</tr>")
        for n, r in react.items():
            rows.append(f"<tr><td>{html.escape(n)}</td>" + "".join(f"<td align=right>{fmt(float(x))}</td>"
                                                                     for x in r) + "</tr>")
        rows.append("</table>")
    for n, r in (fr.get("coupling_forces") or {}).items():
        rows.append(f"Force of the connection to the body at node {html.escape(n)}: " + ", ".join(fmt(float(x)) for x in r[:3]) + " N")
    eq = st.equilibrium
    if eq:
        rows.append(f"Equilibrium check: imbalance {fmt(100 * eq['imbalance'], 2)} %")
    if res.warnings:
        rows.append("<br><span style='color:#b36b00'>! " + "<br>! ".join(html.escape(w) for w in res.warnings)
                    + "</span>")
    return win._join_rows(rows)


class FrameResultsBox(QWidget):
    """"Frames" tab in the results: extremes of the internal forces per member and the EN 1993 assessment (click = detail)."""

    COLS = ["Member", "Cross-section", "|N| [kN]", "|Vz| [kN]", "|My| [kN·m]", "|Mz| [kN·m]", "σ_eq [MPa]", "u [mm]",
            "Utilization", "Governing"]

    def __init__(self, win):
        super().__init__()
        self.win = win
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 4, 2, 2)
        self.table = table(self.COLS, placeholder="Member results will appear after the calculation", rows=(3, 10))
        self.table.horizontalHeader().setSectionResizeMode(len(self.COLS) - 1, QHeaderView.Stretch)
        help_tip(self.table, "Extreme internal forces by member, σ_eq, displacement and EN 1993-1-1 utilization",
                 "Extreme internal forces of the displayed step by member (absolute values), σ_eq, displacement,<br>EN 1993-1-1 utilization (maximum over all steps) and the governing check.<br>Click a row = detailed check lines and highlight the member in 3D.")
        self.table.currentCellChanged.connect(lambda r, *_: self.on_row(r))
        v.addWidget(self.table)
        self.detail = SafeTextBrowser()
        self.detail.setMinimumHeight(140)
        help_tip(self.detail, "Check of the selected member: section, resistance, buckling, lateral-torsional buckling, interaction",
                 "Check of the selected member: section class, cross-section resistance (6.2), buckling and lateral-torsional buckling (6.3),<br>interaction; informative lines are not counted in the utilization")
        v.addWidget(self.detail)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        self._names: list[str] = []

    def fill(self, res, st):
        t = self.table
        t.blockSignals(True)
        t.setRowCount(0)
        self._names = []
        self.detail.setHtml("")
        members = ((st.frame or {}).get("members") or {}) if st is not None else {}
        a = frame_assessment(self.win, res) or {}
        for mname, md in members.items():
            r = t.rowCount()
            t.insertRow(r)
            d = a.get(mname, {})
            u = d.get("utilization")
            vals = [mname, md.get("section", ""), fmt(float(np.max(np.abs(md["N"]))) / 1e3),
                    fmt(float(np.max(np.abs(md["Vz"]))) / 1e3), fmt(float(np.max(np.abs(md["My"]))) / 1e6),
                    fmt(float(np.max(np.abs(md["Mz"]))) / 1e6), fmt(float(np.max(md["sigma_eq"]))),
                    fmt(float(np.linalg.norm(np.asarray(md["u"]), axis=1).max())),
                    fmt(u, 3) if u is not None else "–", d.get("governing", "")]
            for c, txt in enumerate(vals):
                it = QTableWidgetItem(txt)
                if 1 < c < 9:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if c == 8 and u is not None:
                    from .joints_panel import util_color
                    from PySide6.QtGui import QBrush, QColor
                    it.setBackground(QBrush(QColor(util_color(u)).lighter(170)))
                    it.setForeground(QBrush(QColor("#000000")))
                t.setItem(r, c, it)
            self._names.append(mname)
        t.blockSignals(False)
        if "_error" in a:
            self.note.setText(f"<span style='color:#b36b00'>EN 1993 check cannot be performed: {html.escape(a['_error'])}</span>")
        elif a:
            um, w = a.get("_max", (0.0, None))
            self.note.setText(f"Max. utilization {fmt(um, 3)} (member {html.escape(str(w))}). Check inputs (buckling lengths, lateral-torsional buckling, steel) are in the Members step → EN 1993 check.")
        else:
            self.note.setText("")

    def on_row(self, r):
        if not (0 <= r < len(self._names)):
            return
        name = self._names[r]
        res = self.win.results
        a = frame_assessment(self.win, res) or {}
        d = a.get(name, {})
        rows = [f"<b>{html.escape(name)}</b> – {html.escape(d.get('section', ''))}, L = {fmt(d.get('length_mm'), 5)} mm"]
        if d.get("utilization") is not None:
            rows.append(f"Steel {d.get('grade')}, f_y = {fmt(d.get('fy'), 4)} MPa, cross-section class {d.get('section_class')}; governing case “{html.escape(str(d.get('case')))}” at x = {fmt(d.get('x_crit'), 5)} mm")
            rows.append("<table cellspacing=3><tr><th align=left>check</th><th>cl.</th><th align=right>action effect</th><th align=right>resistance</th><th align=right>utilization</th></tr>")
            for row in d.get("rows", []):
                col = "#d23b3b" if row["utilization"] > 1 else ""
                rows.append(f"<tr><td>{html.escape(row['name'])}</td><td>{html.escape(row['clause'])}</td>"
                            f"<td align=right>{fmt(row['effect'], 4)} {html.escape(row['unit'])}</td>"
                            f"<td align=right>{fmt(row['resistance'], 4)}</td>"
                            f"<td align=right><span style='color:{col}'>{fmt(row['utilization'], 3)}</span></td></tr>")
            rows.append("</table>")
        if d.get("curved"):
            c = d["curved"]
            rows.append(f"Curved member R/h = {fmt(c['R_over_h'], 3)}: K_i = {fmt(c['K_i'], 3)}, K_o = {fmt(c['K_o'], 3)} (Winkler–Bach, informative)")
        for w in d.get("warnings", []):
            rows.append(f"<span style='color:#b36b00'>! {html.escape(w)}</span>")
        self.detail.setHtml(self.win._join_rows(rows))
        st = self.win.current_result()
        if st and st[0] == "step":
            md = (res.steps[st[1]].frame or {}).get("members", {}).get(name)
            if md:
                P = np.asarray(md["points"], float)
                self.win.view.mark(P[len(P) // 2], name, (200, 100, 0), key="frame_member")
