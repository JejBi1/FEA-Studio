"""Scenes of the frame system for the headless render (fea_core.mcp.render): bars as thick lines with a z-buffer.

- model: bars by section, supports (markers with the constrained components), connection to a body, loads of the nodes and of the bars
  (arrows), in the mixed model also the bodies (model_scene adds the layer of bars through `frame_overlay`);
- results: the deformed shape (colour |u|), diagrams of the internal forces N, Vy, Vz, T, My, Mz (plotted perpendicular to the bar in its local axes:
  My, Vz, N, T in the direction +z of the section, Mz, Vy in the direction +y), colour according to σ_eq or the utilisation EN 1993;
- shapes of the natural vibrations (ModeSet.frame_shapes on the nodes of Results.info["frame"]) and of the buckling (MYSTRAN – own subdivision
  4 elements per bar, ModeSet.frame_xyz + frame_shapes_own).
Geometry of the bars = nodes of the FE and segments from Results.info["frame"] (beams.run.fe_info), before the calculation beams.fem.build.
"""
from __future__ import annotations

import numpy as np

from ..contour import band_colors, band_edges
from .render import Scene, fmt
from .session import FeaError, Session

SECTION_COLORS = [(0.20, 0.35, 0.65), (0.75, 0.40, 0.10), (0.15, 0.55, 0.35), (0.55, 0.25, 0.60),
                  (0.60, 0.50, 0.10), (0.20, 0.55, 0.65), (0.50, 0.30, 0.25), (0.35, 0.35, 0.35)]
SUPPORT_COLOR = (0.10, 0.30, 0.85)
LOAD_COLOR = (0.85, 0.15, 0.10)
GHOST = (0.62, 0.62, 0.66)
DIAGRAM = {"N": "Axial force N [N] (+ tension)", "Vy": "Shear force Vy [N]", "Vz": "Shear force Vz [N]",
           "T": "Torsional moment T [N·mm]", "My": "Bending moment My [N·mm]", "Mz": "Bending moment Mz [N·mm]"}
AXIS_OF = {"N": 2, "Vy": 1, "Vz": 2, "T": 2, "My": 2, "Mz": 1}       # projection: local axis y (1) / z (2)


def fe_of(study):
    from ..beams.fem import build, FrameError
    try:
        return build(study)
    except FrameError as e:
        raise FeaError(f"Frame structure: {e}") from None
    except Exception as e:  # noqa: BLE001 – a section from the library etc.
        raise FeaError(f"Frame structure: {e}") from None


def member_lengths(study) -> dict[str, float]:
    """Lengths of the bars [mm] (arcs along the length of the arc) – from the discretisation, on an error the direct distance of the nodes."""
    try:
        fe = fe_of(study)
        return {m: float(sum(fe.segs[i].L for i in idx)) for m, idx in fe.member_segs.items()}
    except FeaError:
        fr = study.frame
        return {m.name: float(np.linalg.norm(np.subtract(fr.nodes[m.end].xyz, fr.nodes[m.start].xyz)))
                for m in fr.members.values() if m.start in fr.nodes and m.end in fr.nodes}


def geometry(study, res=None) -> tuple[np.ndarray, dict, dict]:
    """(xyz of the nodes of the FE (n, 3), {bar: [[a, b], …]}, {node of the system: index})."""
    info = (res.info.get("frame") if res is not None else None) or None
    if info:
        return np.asarray(info["xyz"], float), info["segments"], info["joints"]
    fe = fe_of(study)
    segs = {m: [[fe.segs[i].a, fe.segs[i].b] for i in idx] for m, idx in fe.member_segs.items()}
    return np.asarray(fe.xyz, float), segs, dict(fe.joint)


def _edges(segs: dict) -> tuple[np.ndarray, list[str]]:
    E, owner = [], []
    for m, ss in segs.items():
        for a, b in ss:
            E.append((a, b))
            owner.append(m)
    return np.asarray(E, np.int64).reshape(-1, 2), owner


def _diag(P: np.ndarray) -> float:
    return float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) if len(P) else 1.0


def _band_cols(vals_seg: np.ndarray, edges: np.ndarray, reverse: bool = False) -> np.ndarray:
    lut = np.asarray(band_colors(len(edges) - 1), float) / 255.0
    if reverse:
        lut = lut[::-1]
    k = np.clip(np.searchsorted(edges, np.nan_to_num(vals_seg), side="right") - 1, 0, len(lut) - 1)
    return lut[k]


def dofs_label(dofs, stiffness=None) -> str:
    names = ("ux", "uy", "uz", "rx", "ry", "rz")
    fixed = [n for n, on in zip(names, dofs) if on]
    springs = [n for n, k in zip(names, stiffness or [0] * 6) if k and k > 0]
    if len(fixed) == 6:
        txt = "fixed"
    elif fixed == ["ux", "uy", "uz"]:
        txt = "hinge"
    else:
        txt = ", ".join(fixed) or ""
    if springs:
        txt += (", " if txt else "") + "spring " + ", ".join(springs)
    return txt or "no constraint"


def frame_overlay(sess: Session, scene_diag: float | None = None, load_case: str | None = None,
                  show_names: bool = False) -> dict:
    """Bar layer of the model: {"beams", "markers", "arrows", "swatches"} (bars by section, supports, connections,
    loads of the state load_case – None = of all states).
    """
    from ..beams.model import FrameNodeLoad, FrameMemberLoad
    st = sess.study
    fr = st.frame
    X, segs, joints = geometry(st)
    E, owner = _edges(segs)
    diag = max(scene_diag or 0.0, _diag(X))              # mixed model: the larger of the bodies and the bars
    secs = list(fr.sections)
    cols = np.array([SECTION_COLORS[secs.index(fr.members[m].section) % len(SECTION_COLORS)]
                     if fr.members[m].section in secs else (0.3, 0.3, 0.3) for m in owner], float).reshape(-1, 3)
    out = {"beams": [{"V": X, "E": E, "colors": cols, "width": 4.0}], "markers": [], "arrows": [], "swatches": []}
    used = sorted({fr.members[m].section for m in fr.members}, key=lambda s: secs.index(s) if s in secs else 99)
    for sn in used:
        c = SECTION_COLORS[secs.index(sn) % len(SECTION_COLORS)] if sn in secs else (0.3, 0.3, 0.3)
        n_m = sum(1 for m in fr.members.values() if m.section == sn)
        out["swatches"].append((c, f"cross-section {sn} ({n_m} members)"))
    trusses = [m for m in fr.members.values() if m.kind == "truss"]
    if trusses:
        out["swatches"].append(((1, 1, 1), f"truss members (N only): {len(trusses)}"))
    hinged = [m.name for m in fr.members.values() if any(m.releases_start) or any(m.releases_end)]
    for m in hinged:                                   # joints: a ring at the end of the bar
        mem = fr.members[m]
        for end, rel in (("start", mem.releases_start), ("end", mem.releases_end)):
            if any(rel):
                p0 = np.asarray(fr.nodes[mem.start if end == "start" else mem.end].xyz, float)
                p1 = np.asarray(fr.nodes[mem.end if end == "start" else mem.start].xyz, float)
                q = p0 + 0.04 * (p1 - p0)
                out["markers"].append({"xyz": q, "text": "", "color": (1.0, 1.0, 1.0)})
    if hinged:
        out["swatches"].append(((1.0, 1.0, 1.0), "○ hinge (release at member end)"))
    for sup in fr.supports:
        if sup.node in fr.nodes:
            out["markers"].append({"xyz": np.asarray(fr.nodes[sup.node].xyz, float),
                                   "text": f"{sup.node}: {dofs_label(sup.dofs, sup.stiffness)}", "color": SUPPORT_COLOR})
    if fr.supports:
        out["swatches"].append((SUPPORT_COLOR, f"node supports ({len(fr.supports)})"))
    for c in fr.connections:
        if c.node in fr.nodes:
            out["markers"].append({"xyz": np.asarray(fr.nodes[c.node].xyz, float),
                                   "text": f"{c.node} → body ({c.selection})", "color": (0.55, 0.2, 0.7)})
    if fr.connections:
        out["swatches"].append(((0.55, 0.2, 0.7), f"member-to-body connection ({len(fr.connections)})"))
    if show_names:
        for name, nd in fr.nodes.items():
            if not any(s.node == name for s in fr.supports):
                out["markers"].append({"xyz": np.asarray(nd.xyz, float), "text": name, "color": (0.2, 0.2, 0.2)})
        for m, ss in segs.items():
            a, b = ss[len(ss) // 2]
            out["markers"].append({"xyz": 0.5 * (X[a] + X[b]), "text": m, "color": (0.4, 0.4, 0.4)})
    cases = [lc for lc in st.load_cases if load_case is None or lc.name == load_case]
    L = 0.12 * diag
    multi = len(cases) > 1
    nload = 0
    for lc in cases:
        tag = f" [{lc.name}]" if multi else ""
        for l in lc.loads:
            if isinstance(l, FrameNodeLoad) and l.node in fr.nodes:
                p = np.asarray(fr.nodes[l.node].xyz, float)
                f = np.asarray(l.force, float)
                if np.linalg.norm(f) > 0:
                    out["arrows"].append({"tip": p, "vec": f / np.linalg.norm(f) * L, "color": LOAD_COLOR,
                                          "text": f"F = {fmt(float(np.linalg.norm(f)))} N{tag}", "label_at": "tail"})
                mm = np.asarray(l.moment, float)
                if np.linalg.norm(mm) > 0:
                    out["markers"].append({"xyz": p, "text": f"M = ({'; '.join(fmt(float(v)) for v in mm)}) N·mm{tag}",
                                           "color": LOAD_COLOR})
                nload += 1
            elif isinstance(l, FrameMemberLoad) and l.member in segs:
                ss = segs[l.member]
                q0 = np.asarray(l.q, float)
                q1 = q0 if l.q_end is None else np.asarray(l.q_end, float)
                fe = fe_of(st) if l.axes == "local" else None
                for k, t in enumerate((0.15, 0.5, 0.85)):
                    js = min(int(t * len(ss)), len(ss) - 1)          # segment and the position in it
                    a, b = ss[js]
                    p = X[a] + (t * len(ss) - js) * (X[b] - X[a])
                    q = q0 + (q1 - q0) * t
                    if fe is not None:
                        q = fe.segs[fe.member_segs[l.member][js]].R.T @ q
                    nq = np.linalg.norm(q)
                    if nq > 0:
                        out["arrows"].append({"tip": p, "vec": q / nq * 0.6 * L, "color": (0.90, 0.45, 0.05),
                                              "text": (f"q = {fmt(float(np.linalg.norm(q0)))}"
                                                       + ("" if l.q_end is None else f"…{fmt(float(np.linalg.norm(q1)))}")
                                                       + f" N/mm{tag}") if k == 1 else "", "label_at": "tail"})
                nload += 1
            elif getattr(l, "kind", "") == "gravity":
                c0 = 0.5 * (X.min(axis=0) + X.max(axis=0))
                g = np.asarray(l.vector, float)
                if np.linalg.norm(g) > 0:
                    out["arrows"].append({"tip": c0, "vec": g / np.linalg.norm(g) * L, "color": (0.3, 0.3, 0.3),
                                          "text": f"g = {fmt(float(np.linalg.norm(g)))} mm/s²{tag}"})
    if nload:
        out["swatches"].append((LOAD_COLOR, f"node loads (red) and member loads (orange): {nload}"))
    return out


def frame_model_scene(sess: Session, load_case: str | None = None, show_names: bool = True) -> Scene:
    """Model of the frame system alone (study without geometry)."""
    st = sess.study
    if not st.frame.members:
        raise FeaError("The frame has no members (fea_frame_member_set)")
    X, segs, _ = geometry(st)
    ov = frame_overlay(sess, None, load_case, show_names)
    fr = st.frame
    title = [f"Frame system “{st.name}”" + (f" – state {load_case}" if load_case else ""),
             f"{len(fr.nodes)} nodes, {len(fr.members)} members, {len(fr.supports)} supports"]
    return Scene(V=X, T=np.zeros((0, 3), np.int64), title=title,
                 info=["Member color = cross-section; blue markers = supports, red arrows = loads"],
                 legend_title="Legend", swatches=ov["swatches"], markers=ov["markers"], arrows=ov["arrows"],
                 beams=ov["beams"])


def _solid_layer(res, k: float, step: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Surface of the bodies of the mixed model (deformed k·U) as grey triangles."""
    from ..contour import linear_triangles
    from .diagnostics import mesh_cache
    m = res.mesh
    if not len(m.nodes):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    tri, _ = mesh_cache(res).boundary
    V = m.nodes + (k * res.steps[step].U if (k > 0 and step is not None) else 0.0)
    return V, linear_triangles(tri)


def _member_points(fr_member: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(fr_member["points"], float), np.asarray(fr_member["u"], float)


def result_frame_scene(sess: Session, step: int, quantity: str = "deformed", scale: float | None = None,
                       bands: int = 12, utilization: dict | None = None) -> tuple[Scene, dict]:
    """Result of the frame system in the state step. quantity: deformed | sigma | utilization | N | Vy | Vz | T | My | Mz."""
    res = sess.require_results()
    st = res.steps[step]
    fr = st.frame or {}
    if not fr.get("members"):
        raise FeaError(f"Load case '{st.name}' has no member results")
    study = sess.study
    members = fr["members"]
    allP = np.vstack([np.asarray(m["points"], float) for m in members.values()])
    diag = _diag(np.vstack([allP, res.mesh.nodes]) if len(res.mesh.nodes) else allP)
    umax = max(float(np.linalg.norm(np.asarray(m["u"], float), axis=1).max()) for m in members.values())
    if len(res.mesh.nodes):
        umax = max(umax, float(st.umag.max()))
    deformed = quantity in ("deformed",)
    if scale is None:
        k = 0.05 * diag / umax if (deformed and umax > 0) else 0.0
    else:
        k = float(scale) if deformed else 0.0
    beams, markers = [], []
    Vs, Ts = _solid_layer(res, k, step)
    title2 = ""
    meta = {"step": st.name, "quantity": quantity}
    values_leg, edges, reverse, legend = None, None, False, ""
    if deformed or quantity in ("sigma", "utilization"):
        Ps, Es, vals = [], [], []
        off = 0
        for mname, m in members.items():
            P, U = _member_points(m)
            if quantity == "deformed":
                v = np.linalg.norm(U, axis=1)
            elif quantity == "sigma":
                v = np.asarray(m["sigma_eq"], float)
            else:
                u_ = (utilization or {}).get(mname, {}).get("utilization")
                v = np.full(len(P), np.nan if u_ is None else float(u_))
            Ps.append(P + k * U)
            vals.append(v)
            Es.append(np.column_stack([np.arange(len(P) - 1), np.arange(1, len(P))]) + off)
            off += len(P)
        P = np.vstack(Ps)
        v = np.concatenate(vals)
        E = np.vstack(Es)
        lo, hi = float(np.nanmin(v)) if np.isfinite(v).any() else 0.0, float(np.nanmax(v)) if np.isfinite(v).any() else 1.0
        if quantity == "utilization":
            lo, hi = 0.0, max(1.0, hi)
        edges = band_edges(lo, hi, bands)
        seg_v = 0.5 * (v[E[:, 0]] + v[E[:, 1]])
        cols = _band_cols(seg_v, edges)
        if quantity == "utilization":
            cols[~np.isfinite(seg_v)] = (0.7, 0.7, 0.7)
        if k > 0:                                      # undeformed shape in grey (the same lines, the original points)
            beams.append({"V": allP, "E": E, "colors": GHOST, "width": 1.5, "alpha": 0.8, "depth": False})
        beams.append({"V": P, "E": E, "colors": cols, "width": 5.0})
        values_leg = v
        legend = {"deformed": "Displacement |u| [mm]", "sigma": "σ_eq [MPa]", "utilization": "EN 1993 utilization [–]"}[quantity]
        i = int(np.nanargmax(v)) if np.isfinite(v).any() else 0
        us = {"deformed": " mm", "sigma": " MPa", "utilization": ""}[quantity]
        markers.append({"key": "max", "xyz": P[i], "text": f"max {fmt(float(v[i]))}{us}", "color": (0.85, 0.1, 0.1)})
        title2 = f"max {fmt(float(np.nanmax(v)))}{us}" + (f" | deformation {fmt(k, 3)}× (max. displacement {fmt(umax)} mm)"
                                                         if k > 0 else "")
        meta.update(max=float(np.nanmax(v)), max_at_mm=P[i], deformation_scale=k, max_displacement_mm=umax)
        Vplot = P
    elif quantity in DIAGRAM:
        fe = fe_of(study)
        vmax = max(float(np.abs(np.asarray(m[quantity], float)).max()) for m in members.values()) or 1.0
        sc = (0.12 * diag / vmax) if scale is None else float(scale)
        Pall, Eb, Pd, Ed, cd = [], [], [], [], []
        off_b = off_d = 0
        best = (0.0, None, None)
        for mname, m in members.items():
            P = np.asarray(m["points"], float)
            vals = np.asarray(m[quantity], float)
            idx = fe.member_segs.get(mname, [])
            npp = len(P) // max(len(idx), 1)
            ax_ = AXIS_OF[quantity]
            dirs = np.array([fe.segs[idx[min(j // max(npp, 1), len(idx) - 1)]].R[ax_] for j in range(len(P))])
            Q = P + (sc * vals)[:, None] * dirs
            Pall.append(P)
            Eb.append(np.column_stack([np.arange(len(P) - 1), np.arange(1, len(P))]) + off_b)
            off_b += len(P)
            # outline of the diagram + vertical lines (hatching)
            n_ = len(P)
            Pd += [Q, P]
            ed = [np.column_stack([np.arange(n_ - 1), np.arange(1, n_)]) + off_d]
            ed.append(np.column_stack([np.arange(n_) + off_d, np.arange(n_) + off_d + n_]))
            Ed += ed
            sign = np.sign(0.5 * (vals[:-1] + vals[1:]))
            cd.append(np.where(sign[:, None] >= 0, [[0.85, 0.2, 0.15]], [[0.15, 0.35, 0.85]]))
            cd.append(np.where(np.sign(vals)[:, None] >= 0, [[0.95, 0.6, 0.55]], [[0.6, 0.7, 0.95]]))
            off_d += 2 * n_
            j = int(np.argmax(np.abs(vals)))
            if abs(vals[j]) > best[0]:
                best = (abs(vals[j]), Q[j], (mname, float(vals[j]), float(m["s"][j])))
        Pb, Pq = np.vstack(Pall), np.vstack(Pd)
        beams.append({"V": Pb, "E": np.vstack(Eb), "colors": (0.15, 0.15, 0.18), "width": 3.0})
        beams.append({"V": Pq, "E": np.vstack(Ed), "colors": np.vstack(cd), "width": 2.0, "depth": False})
        unit = DIAGRAM[quantity].split("[")[1].rstrip("]")
        if best[1] is not None:
            mname, val, s_ = best[2]
            markers.append({"key": "max", "xyz": best[1], "text": f"{quantity} = {fmt(val)} {unit} ({mname}, s = {fmt(s_)} mm)",
                            "color": (0.85, 0.1, 0.1)})
            meta.update(extreme={"member": mname, "value": val, "s_mm": s_})
        title2 = (f"|{quantity}| max {fmt(vmax)} {unit}; plotted perpendicular to the member along local axis {('z' if AXIS_OF[quantity] == 2 else 'y')} (red +, blue −), scale {fmt(sc, 3)} mm/{unit}")
        meta.update(diagram_scale=sc, max_abs=vmax)
        Vplot = np.vstack([Pb, Pq])
    else:
        raise FeaError("quantity must be deformed | sigma | utilization | N | Vy | Vz | T | My | Mz")
    if len(Ts):
        beams = [dict(b, depth=b.get("depth", True)) for b in beams]
    title = [f"{legend or DIAGRAM.get(quantity, quantity)} – frame, load case “{st.name}”", title2]
    sw = [] if values_leg is not None else [((0.85, 0.2, 0.15), "positive values"), ((0.15, 0.35, 0.85), "negative values")]
    scene = Scene(V=Vs if len(Ts) else Vplot, T=Ts, tri_colors=np.tile((0.86, 0.86, 0.88), (len(Ts), 1)) if len(Ts) else None,
                  values=values_leg, edges=edges, reverse=reverse, legend_title=legend or "Legend",
                  bands_on_mesh=False, title=title, markers=markers, beams=beams, swatches=sw,
                  info=[f"{study.name}: {len(members)} members" + (f", {len(res.mesh.tets)} solid elements" if len(Ts) else ""),
                        "Units: mm, N, MPa"])
    return scene, meta


def mode_frame_scene(sess: Session, j: int, k: int, scale: float | None = None, bands: int = 12) -> tuple[Scene, dict]:
    """Shape of the mode k of the set Results.modes[j] on the bars (+ the bodies of the mixed model, if the shape of the bodies)."""
    from .scenes import mode_label
    res = sess.require_results()
    ms = res.modes[j]
    study = sess.study
    if getattr(ms, "frame_shapes", None) is not None and len(ms.frame_shapes):
        X, segs, _ = geometry(study, res)
        U = np.asarray(ms.frame_shapes[k], float)
    elif getattr(ms, "frame_shapes_own", None) is not None:
        from ..beams.run import eigen_fe               # own subdivision of the shapes (own solver: segments_modal, MYSTRAN: 4)
        try:
            fe = eigen_fe(study)
        except Exception as e:  # noqa: BLE001
            raise FeaError(f"Frame structure: {e}") from None
        X = np.asarray(ms.frame_xyz, float)
        if len(fe.xyz) != len(X):
            raise FeaError("The buckling mode cannot be mapped to the members (the study has changed since the analysis)")
        segs = {m: [[fe.segs[i].a, fe.segs[i].b] for i in idx] for m, idx in fe.member_segs.items()}
        U = np.asarray(ms.frame_shapes_own[k], float)
    else:
        raise FeaError(f"{ms.name}: the mode shape on the members is not available (MYSTRAN buckling modes are not saved in the project – run the analysis again)")
    E, _ = _edges(segs)
    Vs, Ts = (np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    diag = _diag(X)
    if len(res.mesh.nodes) and ms.shapes is not None and np.asarray(ms.shapes).shape[1] == len(res.mesh.nodes):
        from ..contour import linear_triangles
        from .diagnostics import mesh_cache
        tri, _ = mesh_cache(res).boundary
        Ts = linear_triangles(tri)
        diag = _diag(np.vstack([X, res.mesh.nodes]))
    kk = 0.08 * diag if scale is None else float(scale)
    if len(Ts):
        Vs = res.mesh.nodes + kk * np.asarray(ms.shapes[k], float)
    P = X + kk * U
    v = np.linalg.norm(U, axis=1)
    edges = band_edges(0.0, max(float(v.max()), 1e-12), bands)
    cols = _band_cols(0.5 * (v[E[:, 0]] + v[E[:, 1]]), edges)
    beams = [{"V": X, "E": E, "colors": GHOST, "width": 1.5, "alpha": 0.8, "depth": False},
             {"V": P, "E": E, "colors": cols, "width": 5.0, "depth": bool(len(Ts))}]
    title = [f"Mode |u| [–] – {mode_label(ms, k)}", f"shape normalized to max |u| = 1, magnified {fmt(kk, 3)}×"
             + (" | LINEAR BUCKLING (no imperfections)" if ms.kind == "buckling" else "")]
    scene = Scene(V=Vs if len(Ts) else P, T=Ts, tri_colors=np.tile((0.86, 0.86, 0.88), (len(Ts), 1)) if len(Ts) else None,
                  values=v, edges=edges, legend_title="Mode |u| [–]", bands_on_mesh=False, title=title,
                  beams=beams, info=[f"{study.name}: frame", "Undeformed shape in gray"])
    meta = {"mode_set": ms.name, "kind": ms.kind, "mode": k + 1,
            ("frequency_Hz" if ms.kind == "modal" else "buckling_factor"): float(ms.values[k]),
            "deformation_scale": kk}
    return scene, meta


__all__ = ["frame_overlay", "frame_model_scene", "result_frame_scene", "mode_frame_scene", "geometry",
           "dofs_label", "DIAGRAM"]
