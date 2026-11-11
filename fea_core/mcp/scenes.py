"""Assembly of the scenes for the render: results on the (deformed) mesh and the model with the constraints / loads (B-rep from OCP)."""
from __future__ import annotations

import colorsys

import numpy as np

from ..contour import band_colors, band_edges, linear_triangles

from .render import Scene, fmt
from .session import Session

PART_COLORS = [(0.80, 0.82, 0.86), (0.86, 0.80, 0.70), (0.74, 0.84, 0.76), (0.84, 0.76, 0.84),
               (0.76, 0.82, 0.90), (0.90, 0.86, 0.72), (0.78, 0.78, 0.78), (0.88, 0.78, 0.76)]


def _palette(n: int, hue0: float, hue1: float, s=0.75, v=0.85) -> list[tuple]:
    if n <= 0:
        return []
    return [colorsys.hsv_to_rgb(hue0 + (hue1 - hue0) * (i / max(n - 1, 1)), s, v) for i in range(n)]


def _tri6_edges(tri: np.ndarray) -> np.ndarray:
    """Edges of the elements on the surface (corner–midpoint–corner) – unambiguous."""
    if tri.shape[1] == 6:
        e = np.concatenate([tri[:, [0, 3]], tri[:, [3, 1]], tri[:, [1, 4]], tri[:, [4, 2]], tri[:, [2, 5]], tri[:, [5, 0]]])
    else:
        e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


# ===================================================================== results

def result_scene(sess: Session, step: int, key: str, values: np.ndarray, label: str, unit: str, reverse: bool,
                 deformed: bool, scale: float | None, nbands: int, vmin, vmax, show_mesh: bool,
                 signed: bool, thermal: int | None = None, time_s: float | None = None,
                 section: tuple | None = None, mode: tuple[int, int] | None = None) -> tuple[Scene, dict]:
    """Scene of a result: the mechanical state step (the deformed shape), or the thermal state thermal (index into
    Results.thermal; always the undeformed shape, time_s = the time of the frame of the transient state), or the shape of a mode
    mode = (index of the set Results.modes, index of the mode) – a natural shape / a buckling shape normalised to max |u| = 1.
    section = (point, normal): a section by a plane – the side (x − point)·n ≤ 0 is kept, the section face coloured (interior).
    """
    from .diagnostics import mesh_cache
    res = sess.results
    m = res.mesh
    tr = res.thermal[thermal] if thermal is not None else None
    ms = res.modes[mode[0]] if mode is not None and tr is None else None
    st = res.steps[step] if tr is None and ms is None else None
    tri, elem = mesh_cache(res).boundary
    T = linear_triangles(tri)
    rep = 4 if tri.shape[1] == 6 else 1
    group = np.repeat(m.tet_part[elem], rep)
    diag = float(np.linalg.norm(m.nodes.max(axis=0) - m.nodes.min(axis=0)))
    if ms is not None:
        U = np.asarray(ms.shapes[mode[1]], float)
    else:
        U = st.U if st is not None else None
    umax = float(np.linalg.norm(U, axis=1).max()) if U is not None else 0.0
    if not deformed or U is None:
        k = 0.0
    elif scale is None:
        k = (0.08 if ms is not None else 0.05) * diag / umax if umax > 0 else 0.0
    else:
        k = float(scale)
    V = m.nodes + k * U if k > 0 else m.nodes
    surf_nodes = np.unique(tri)
    sv = values[surf_nodes]
    lo = float(np.nanmin(sv)) if vmin is None else float(vmin)
    hi = float(np.nanmax(sv)) if vmax is None else float(vmax)
    edges = band_edges(lo, hi, nbands)
    imax = int(surf_nodes[np.nanargmax(sv)])
    imin = int(surf_nodes[np.nanargmin(sv)])
    markers = []
    us = "" if unit == "–" else f" {unit}"                 # dimensionless quantities without a unit for the numbers
    if key not in ("safety", "print_safety", "fatigue_safety", "fatigue_life"):
        markers.append({"key": "max", "xyz": V[imax], "text": f"max {fmt(values[imax])}{us}", "color": (0.85, 0.1, 0.1)})
    if signed or key in ("safety", "print_safety", "fatigue_safety", "fatigue_life", "temperature"):
        markers.append({"key": "min", "xyz": V[imin], "text": f"min {fmt(values[imin])}{us}",
                        "color": (0.85, 0.1, 0.1) if reverse else (0.1, 0.3, 0.85)})
    short = label.split(" (")[0]
    if tr is not None:
        when = ""
        if tr.transient and tr.times:
            when = f", t = {fmt(tr.times[-1] if time_s is None else float(time_s))} s"
        src = f"thermal case “{tr.name}”{when}"
    elif ms is not None:
        src = mode_label(ms, mode[1])
    else:
        src = f"state “{st.name}“"
    if ms is not None:
        defo = f" | shape scaled {fmt(k, 3)}× (shape normalized to max |u| = 1)" if k > 0 else " | undeformed shape"
    else:
        defo = f" | deformation {fmt(k, 3)}× (max. displacement {fmt(umax)} mm)" if k > 0 else " | undeformed shape"
    title = [f"{label} [{unit}] – {src}",
             f"max {fmt(float(np.nanmax(values)))}{us}, min {fmt(float(np.nanmin(values)))}{us}" + defo]
    info = [f"{sess.study.name}: {len(m.nodes)} nodes, {len(m.tets)} elements {m.element_type}",
            "Units: mm, °C, W" if tr is not None else
            ("Units: mm, N, MPa, °C" if key == "temperature" else "Units: mm, N, MPa")]
    if vmin is not None or vmax is not None:
        info.insert(0, "Scale range set manually – values outside it fall in the end bands")
    ghost = m.nodes if k > 0 else None
    mesh_edges = _tri6_edges(tri) if show_mesh else None
    if section is not None:
        V, T, values_vis, group = _section_geometry(m, tri, elem, values, V, section)
        if len(T) == 0:
            raise ValueError("The section plane left nothing – choose another point or the opposite normal")
        ghost, mesh_edges = None, None                 # outline/mesh only without the section (the section changes the vertices)
        o, n = (np.asarray(x, float) for x in section)
        title[0] += f" | section through point {fmt_vec(o)}, normal {fmt_vec(n)}"
        for mk in markers:                             # max/min on the visible part (surface + section)
            j = int(np.nanargmax(values_vis) if mk["key"] == "max" else np.nanargmin(values_vis))
            mk["xyz"] = V[j]
            mk["text"] = f"{mk['key']} {fmt(values_vis[j])}{us}"
        values = values_vis
    scene = Scene(V=V, T=T, values=values, tri_group=group, ghost=ghost,
                  mesh_edges=mesh_edges, title=title, info=info,
                  legend_title=f"{short}\n[{unit}]" if len(short) > 16 else f"{short} [{unit}]",
                  edges=edges, reverse=reverse, markers=markers)
    meta = {"field": key, "unit": unit, "deformation_scale": k, "max_displacement_mm": umax,
            "surface_max": float(np.nanmax(sv)), "surface_min": float(np.nanmin(sv)), "max_at_mm": m.nodes[imax],
            "min_at_mm": m.nodes[imin], "band_edges": edges}
    if tr is not None:
        meta = {"field": key, "unit": unit, "thermal_case": tr.name, **{k_: v for k_, v in meta.items()
                                                                       if k_ not in ("field", "unit")}}
        if tr.transient and tr.times:
            meta["time_s"] = float(tr.times[-1]) if time_s is None else float(time_s)
    elif ms is not None:
        meta = {"field": key, "unit": unit, "mode_set": ms.name, "kind": ms.kind, "mode": mode[1] + 1,
                ("frequency_Hz" if ms.kind == "modal" else "buckling_factor"): float(ms.values[mode[1]]),
                **{k_: v for k_, v in meta.items() if k_ not in ("field", "unit", "max_displacement_mm")}}
    else:
        meta = {"field": key, "unit": unit, "step": st.name, **{k_: v for k_, v in meta.items()
                                                                if k_ not in ("field", "unit")}}
    return scene, meta


def mode_label(ms, j: int) -> str:
    """Label of the mode for the title of the picture: "natural shape 1: f = 209.4 Hz" / "buckling shape 1: λ = 21.65"."""
    v = float(ms.values[j])
    if ms.kind == "modal":
        return f"mode shape {j + 1}: f = {fmt(v)} Hz"
    return f"buckling mode {j + 1} ({ms.case}): λ = {fmt(v)}"


def fmt_vec(v) -> str:
    return "(" + "; ".join(fmt(float(x)) for x in v) + ")"


def _section_geometry(m, tri, elem, values, V, section):
    """Surface trimmed by the plane + the section face (fea_core.section), welded vertices, group = part (section = part + number of
    parts → the boundary of the section and of the surface is drawn as an edge).
    """
    from ..section import clip_surface, plane_section
    o, n = section
    nparts = len(m.part_names)
    Ps, Vs, Gs = [], [], []
    for pi in range(nparts):
        sel = m.tet_part[elem] == pi
        if np.any(sel):
            P, Vv = clip_surface(m.nodes, tri[sel], values, o, n, V)
            Ps.append(P), Vs.append(Vv), Gs.append(np.full(len(P), pi))
        P, Vv = plane_section(m.nodes, m.tets, values, o, n, V, m.tet_part == pi)
        Ps.append(P), Vs.append(Vv), Gs.append(np.full(len(P), nparts + pi))
    P = np.concatenate(Ps).reshape(-1, 3)
    vals = np.concatenate(Vs).ravel()
    group = np.concatenate(Gs)
    if len(P) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros(0), group
    scale = max(float(np.abs(P).max()), 1e-30)
    key = np.round(P / scale * 1e9).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return P[first], inv.reshape(-1, 3), vals[first], group


# ===================================================================== 3D printing – the share of the shell

NOT_PRINTED = (0.78, 0.78, 0.78)


def print_shell_scene(study, mesh, origin, normal, info_lines: list[str] | None = None) -> tuple[Scene, dict]:
    """Section of the mesh with the share of the shell in the element (mesh.cell_data["print_shell"], 0 = infill, 1 = shell, −1 = unprinted
    part grey). Element field → colour by triangles: the surface trimmed by the plane and the section face are composed by the levels of
    the share (fea_core.section with the mask of the elements), group = part (the boundary of the section and of the surface as an edge).
    """
    from ..section import clip_surface, plane_section
    shell = np.asarray(mesh.cell_data["print_shell"], float)
    o = np.asarray(origin, float)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    tri, elem = mesh.boundary_faces()
    zero = np.zeros(len(mesh.nodes))
    levels = np.unique(np.round(shell, 6))
    bins = max(1, int(getattr(study.printing, "bins", 10)))
    lut = np.asarray(band_colors(bins + 1), float) / 255.0

    def color(f):
        return NOT_PRINTED if f < 0 else tuple(lut[int(round(min(max(f, 0.0), 1.0) * bins))])
    nparts = len(mesh.part_names)
    Ps, Cs, Gs = [], [], []
    area_cut: dict[float, float] = {}
    for f in levels:
        lev = np.abs(shell - f) < 5e-7
        col = color(float(f))
        for pi in range(nparts):
            em = lev & (mesh.tet_part == pi)
            if not em.any():
                continue
            st = em[elem]
            if st.any():
                P, _ = clip_surface(mesh.nodes, tri[st], zero, o, n)
                Ps.append(P), Cs.append(np.tile(col, (len(P), 1))), Gs.append(np.full(len(P), pi))
            P, _ = plane_section(mesh.nodes, mesh.tets, zero, o, n, None, em)
            if len(P):
                Ps.append(P), Cs.append(np.tile(col, (len(P), 1))), Gs.append(np.full(len(P), nparts + pi))
                a = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum()
                area_cut[float(f)] = area_cut.get(float(f), 0.0) + float(a)
    if not Ps or sum(len(p) for p in Ps) == 0:
        raise ValueError("The section plane left nothing – choose another position or the opposite direction")
    P = np.concatenate(Ps)
    colors = np.concatenate(Cs)
    group = np.concatenate(Gs)
    if not area_cut:
        raise ValueError("The section plane does not intersect any part – choose another position")
    flat = P.reshape(-1, 3)
    scale = max(float(np.abs(flat).max()), 1e-30)
    key = np.round(flat / scale * 1e9).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V, T = flat[first], inv.reshape(-1, 3)
    swatches = []
    for f in sorted(area_cut, reverse=True):
        if f < 0:
            continue
        txt = ("shell (perimeters / solid layers)" if f >= 1 - 1e-9 else "core (infill)" if f <= 1e-9
               else f"shell fraction {100 * f:.0f} %")
        swatches.append((color(f), txt))
    if any(f < 0 for f in area_cut):
        swatches.append((NOT_PRINTED, "non-printed part"))
    total = sum(area_cut.values())
    printed_area = sum(a for f, a in area_cut.items() if f >= 0)
    shell_area = sum(a * f for f, a in area_cut.items() if f >= 0)
    title = [f"3D printing – shell fraction in element | section through point {fmt_vec(o)}, normal {fmt_vec(n)}",
             f"{len(mesh.tets)} elements, {bins} fraction levels; shell in the section {100 * shell_area / printed_area:.1f} % of the area of printed parts" if printed_area > 0 else f"{len(mesh.tets)} elements"]
    info = list(info_lines or []) + ["Element color = fraction of its volume in the shell (perimeters, top/bottom layers); section through the undeformed mesh"]
    scene = Scene(V=V, T=T, tri_colors=colors, tri_group=group, title=title, info=info,
                  legend_title="Shell fraction", swatches=swatches, shade_range=(0.62, 1.0))
    meta = {"section_point_mm": o, "section_normal": n, "cut_area_mm2": total,
            "cut_area_by_shell_fraction": {f"{f:g}": a for f, a in sorted(area_cut.items())},
            "shell_fraction_of_printed_cut": shell_area / printed_area if printed_area > 0 else None}
    return scene, meta


# ===================================================================== model (OCP)

def tessellate(sess: Session) -> dict:
    """Triangulation of the parts by faces: welded vertices, triangles, (part, face) of the triangle."""
    if sess._tess is not None:
        return sess._tess
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location
    from ..geometry import bbox
    bbs = np.array([bbox(p.shape) for p in sess.parts])
    diag = float(np.linalg.norm(bbs[:, 3:].max(axis=0) - bbs[:, :3].min(axis=0))) or 1.0
    verts, tris, owner = [], [], []
    off = 0
    for pi, p in enumerate(sess.parts):
        BRepMesh_IncrementalMesh(p.shape, 0.002 * diag, False, 0.25, True)
        for fi, f in enumerate(sess.faces_of(pi)):
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(f, loc)
            if tri is None:
                continue
            trsf = loc.Transformation()
            nn = tri.NbNodes()
            P = np.empty((nn, 3))
            for i in range(1, nn + 1):
                q = tri.Node(i).Transformed(trsf)
                P[i - 1] = (q.X(), q.Y(), q.Z())
            nt = tri.NbTriangles()
            Tq = np.empty((nt, 3), np.int64)
            for i in range(1, nt + 1):
                Tq[i - 1] = tri.Triangle(i).Get()
            Tq -= 1
            if f.Orientation() == TopAbs_REVERSED:
                Tq = Tq[:, [0, 2, 1]]
            verts.append(P)
            tris.append(Tq + off)
            owner.append(np.column_stack([np.full(nt, pi), np.full(nt, fi)]))
            off += nn
    V = np.vstack(verts)
    T = np.vstack(tris)
    own = np.vstack(owner)
    q = np.round(V / (1e-6 * diag)).astype(np.int64)            # welding of the identical vertices (seams, edges of faces)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    T = inv[T]
    keep = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    sess._tess = {"V": V[first], "T": T[keep], "owner": own[keep], "diag": diag}
    return sess._tess


THERMAL_HUES = {"fixed_temperature": (0.76, 0.84), "heat_flux": (0.05, 0.09), "heat_flow": (0.05, 0.09),
                "convection": (0.46, 0.53), "radiation": (0.13, 0.17), "heat_generation": (0.93, 0.97),
                "cavity_radiation": (0.17, 0.21)}


def model_scene(sess: Session, load_case: str | None, highlight, show_face_ids: bool,
                face_id_part: str | None, thermal_case: str | None = None) -> Scene:
    """Model before the calculation. load_case = only this mechanical state; thermal_case = the thermal boundary conditions
    of this state (without load_case only they). Without both = constraints, all loads and all thermal states.
    """
    from ..selectors import resolve
    from .common import resolve_faces, face_id
    from .diagnostics import THERMAL_LABELS, bc_value_text, thermal_load_text
    st = sess.study
    if sess.frame_only:                                # frame system only (without geometry)
        from .frame_scenes import frame_model_scene
        return frame_model_scene(sess, load_case)
    tess = tessellate(sess)
    V, T, own, diag = tess["V"], tess["T"], tess["owner"], tess["diag"]
    npart = len(sess.parts)
    group = own[:, 0] * 100000 + own[:, 1]
    colors = np.array([PART_COLORS[pi % len(PART_COLORS)] for pi in own[:, 0]], float)
    excl = st.excluded_parts()
    for pi, p in enumerate(sess.parts):
        if p.name in excl:
            colors[own[:, 0] == pi] = (0.93, 0.93, 0.93)
    swatches = []
    if npart <= 8:
        swatches += [((0.93, 0.93, 0.93), f"part {p.name} (not meshed – bolt/excluded)") if p.name in excl
                     else (PART_COLORS[pi % len(PART_COLORS)], f"part {p.name}") for pi, p in enumerate(sess.parts)]
    elif excl:
        swatches.append(((0.93, 0.93, 0.93), "bolt parts / excluded (not meshed)"))
    for pi in range(npart):
        sess.face_props(pi)
    cache = sess._face_props

    def faces_mask(sels) -> np.ndarray:
        mask = np.zeros(len(T), bool)
        for f in sels:
            r = resolve(f, sess.parts, cache)
            if r is not None:
                mask |= (own[:, 0] == r[0]) & (own[:, 1] == r[1])
        return mask

    tn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    ta = 0.5 * np.linalg.norm(tn, axis=1)
    tc = V[T].mean(axis=1)

    def anchor(mask) -> tuple[np.ndarray, np.ndarray]:
        """A point ON the chosen faces near the weighted centroid (the centroid of the triangle) and the outward normal."""
        idx = np.nonzero(mask)[0]
        if not len(idx):
            c = 0.5 * (V.min(axis=0) + V.max(axis=0))
            return c, np.array([0.0, 0.0, 1.0])
        c = (tc[idx] * ta[idx, None]).sum(axis=0) / max(ta[idx].sum(), 1e-30)
        # projection of the centroid into the triangle that contains it (a planar face → exactly the centroid)
        A, B, C = V[T[idx, 0]], V[T[idx, 1]], V[T[idx, 2]]
        nn = tn[idx] / np.maximum(np.linalg.norm(tn[idx], axis=1), 1e-30)[:, None]
        dist = np.einsum("ij,ij->i", c - A, nn)
        q = c - dist[:, None] * nn
        v0, v1, v2 = B - A, C - A, q - A
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-300)
        bv = (d11 * d20 - d01 * d21) / den
        bw = (d00 * d21 - d01 * d20) / den
        inside = (bv >= -1e-9) & (bw >= -1e-9) & (bv + bw <= 1 + 1e-9)
        if inside.any():
            k = np.nonzero(inside)[0][np.argmin(np.abs(dist[inside]))]
            return q[k], nn[k]
        j = idx[np.argmin(np.linalg.norm(tc[idx] - c, axis=1))]
        return tc[j], tn[j] / max(np.linalg.norm(tn[j]), 1e-30)

    from .session import FeaError
    show_mech = thermal_case is None or load_case is not None
    cases = st.load_cases if load_case is None else [lc for lc in st.load_cases if lc.name == load_case]
    if load_case is not None and not cases:
        raise FeaError(f"Load case '{load_case}' does not exist")
    if thermal_case is not None:
        th_cases = [c for c in st.thermal.cases if c.name == thermal_case]
        if not th_cases:
            raise FeaError(f"Thermal case '{thermal_case}' does not exist (cases: {', '.join((c.name for c in st.thermal.cases)) or 'none'})")
    else:
        th_cases = list(st.thermal.cases) if load_case is None else []
    if not show_mech:
        cases = []
    cons = list(st.constraints) if show_mech else []
    loads = [(lc.name, l) for lc in cases for l in lc.loads]
    ccol = _palette(len(cons), 0.55, 0.70, 0.8, 0.85)              # blue shades
    lcol = _palette(len([1 for _, l in loads if getattr(l, "selection", None)]), 0.0, 0.11, 0.85, 0.95)
    markers, arrows, lines = [], [], []
    L = 0.16 * diag
    for c, col in zip(cons, ccol):
        if c.selection not in st.selections:
            continue
        sels = st.selections[c.selection].faces
        msk = faces_mask(sels)
        colors[msk] = col
        extra = ""
        if c.kind == "displacement":
            extra = " " + ", ".join(f"{k}={v:g}" for k, v in (("ux", c.ux), ("uy", c.uy), ("uz", c.uz)) if v is not None)
        elif c.kind == "cylindrical":
            extra = " " + ", ".join(k for k, on in (("radially", c.radial), ("tangentially", c.tangential),
                                                    ("axially", c.axial)) if on)
        elif c.kind == "remote_support":
            from ..remote import dofs_text, coupling_text
            extra = f" ({coupling_text(c.coupling)}, {dofs_text(c.dofs)})"
        elif c.kind == "elastic":
            from ..loads_text import constraint_text
            extra = f" ({constraint_text(c, st, fmt)})"
            col = (0.25, 0.65, 0.45)                   # elastic support green (the foundation, not a rigid constraint)
            colors[msk] = col
        swatches.append((col, f"constraint {c.kind}: {c.selection}{extra}"))
        p, _ = anchor(msk)
        if c.kind == "remote_support":                 # point + dashed segments to the face (scattered points of the face)
            pt = np.asarray(c.point, float)
            idx = np.nonzero(msk)[0]
            if len(idx):
                cand = tc[idx]
                pick = [int(np.argmax(np.linalg.norm(cand - p, axis=1)))]
                dmin = np.linalg.norm(cand - cand[pick[0]], axis=1)
                for _ in range(min(7, len(cand) - 1)):        # the most distant points (coverage of the face)
                    pick.append(int(np.argmax(dmin)))
                    dmin = np.minimum(dmin, np.linalg.norm(cand - cand[pick[-1]], axis=1))
                lines += [{"a": pt, "b": cand[i], "color": col} for i in pick] + [{"a": pt, "b": p, "color": col}]
            from ..remote import dofs_text
            markers.append({"xyz": pt, "text": f"point {c.selection} (remote support {dofs_text(c.dofs)})",
                            "color": col})
            continue
        markers.append({"xyz": p, "text": f"{c.selection} ({c.kind})", "color": col})
    j = 0
    multi_case = len(cases) > 1

    def cs(case):
        return f" [{case}]" if multi_case else ""

    def vec_txt(v):
        return "(" + ", ".join(fmt(float(x)) for x in v) + ")"
    from .. import csys as csys_mod
    from ..loads_text import load_text, pressure_text, rotation_text
    edge_cache: dict = {}

    def edge_lines(sel_obj, col) -> np.ndarray:
        """Segments of the chosen edges (lines in the scene), returns the point in the middle of the first edge."""
        from ..geometry import edge_polyline, list_edges
        from ..selectors import resolve_edge
        mid = None
        for e in sel_obj.edges:
            r = resolve_edge(e, sess.parts, edge_cache)
            if r is None:
                continue
            P = edge_polyline(list_edges(sess.parts[r[0]].shape)[r[1]])
            lines.extend({"a": P[i], "b": P[i + 1], "color": col, "width": 3.0, "solid": True}
                         for i in range(len(P) - 1))
            if mid is None:
                mid = np.asarray(e.point, float)
        return mid

    def force_dir(l, at) -> np.ndarray:
        """Global direction of the force at the point at (local / cylindrical system)."""
        try:
            return csys_mod.field_global(st, getattr(l, "csys", None), l.vector, np.asarray(at, float)[None, :])[0]
        except ValueError:
            return np.asarray(l.vector, float)
    for case, l in loads:
        sel = getattr(l, "selection", None)
        if sel and sel in st.selections:
            col = lcol[j]
            j += 1
            sobj = st.selections[sel]
            if sobj.entity in ("edges", "points"):          # force on an edge / at a point
                pts = [np.asarray(q.point, float) for q in sobj.points]
                p = edge_lines(sobj, col) if sobj.edges else pts[0]
                if p is None:
                    continue
                for k, q in enumerate(pts or [p]):
                    f = force_dir(l, q)
                    nf = np.linalg.norm(f)
                    if nf > 0:
                        arrows.append({"tip": q, "vec": f / nf * L, "color": col, "label_at": "tail",
                                       "text": f"{sel}: F = {fmt(np.linalg.norm(l.vector))} N" if k == 0 else ""})
                    if sobj.points:
                        markers.append({"xyz": q, "text": f"point {sel}" if k == 0 else "", "color": col})
                swatches.append((col, f"{sel}{cs(case)}: {load_text(l, st, fmt)}"))
                continue
            sels = sobj.faces
            msk = faces_mask(sels)
            colors[msk] = col
            p, nrm = anchor(msk)
            if l.kind == "force":
                f = force_dir(l, p)
                nf = np.linalg.norm(f)
                if nf > 0:
                    v = f / nf * L                         # tension (force out of the face): the arrow starts on the face
                    pull = float(np.dot(f, nrm)) > 1e-9 * nf
                    arrows.append({"tip": p + v if pull else p, "vec": v,
                                   "text": f"{sel}: F = {fmt(np.linalg.norm(l.vector))} N",
                                   "color": col, "label_at": "tip" if pull else "tail"})
                swatches.append((col, f"force {sel}{cs(case)}: {vec_txt(l.vector)} N"
                                 + (f" ({load_text(l, st, fmt).split(' in system ')[-1]})" if l.csys else "")))
            elif l.kind == "pressure" and getattr(l, "variable", False):
                from ..loads import pressure_value
                pv = float(pressure_value(l, np.asarray(p, float)[None, :])[0])
                dvec = -np.asarray(nrm) * (np.sign(pv) or 1.0) * L * 0.7
                arrows.append({"tip": p + dvec if pv < 0 else p, "vec": dvec, "color": col,
                               "text": f"{sel}: p = {fmt(pv)} MPa (here)", "label_at": "tip" if pv < 0 else "tail"})
                swatches.append((col, f"pressure {sel}{cs(case)}: {pressure_text(l, fmt)}"))
            elif l.kind == "bearing":
                f = np.asarray(l.vector, float)
                nf = np.linalg.norm(f)
                if nf > 0:                                 # the bearing presses on the facing half of the hole
                    arrows.append({"tip": p, "vec": f / nf * L, "text": f"{sel}: F = {fmt(nf)} N (bearing)",
                                   "color": col, "label_at": "tail"})
                swatches.append((col, f"bearing load {sel}{cs(case)}: {vec_txt(f)} N"))
            elif l.kind == "pressure":
                dvec = -np.asarray(nrm) * np.sign(l.value or 1.0) * L * 0.7
                suction = (l.value or 0) < 0                    # suction / tension: the arrow from the face outwards
                arrows.append({"tip": p + dvec if suction else p, "vec": dvec, "text": f"{sel}: p = {fmt(l.value)} MPa",
                               "color": col, "label_at": "tip" if suction else "tail"})
                swatches.append((col, f"pressure {sel}{cs(case)}: {fmt(l.value)} MPa"))
            elif l.kind == "remote":
                pt = np.asarray(l.point, float)
                f = np.asarray(l.force, float)
                nf = np.linalg.norm(f)
                txt = f"{sel}: " + (f"F = {fmt(nf)} N" if nf else "") + \
                      (f" M = {vec_txt(l.moment)} N·mm" if np.any(l.moment) else "")
                if nf > 0:
                    arrows.append({"tip": pt, "vec": f / nf * L, "text": txt, "color": col})
                markers.append({"xyz": pt, "text": f"point {sel} ({l.coupling})" + ("" if nf else " " + txt), "color": col})
                swatches.append((col, f"remote load {sel}{cs(case)}"))
            else:                                          # another kind (extension)
                swatches.append((col, f"{l.kind} {sel}{cs(case)}"))
                markers.append({"xyz": p, "text": f"{sel} ({l.kind})", "color": col})
        elif l.kind == "rotation":                       # axis of rotation (dashed through the model) + description
            a = np.asarray(l.axis, float)
            a = a / max(np.linalg.norm(a), 1e-30)
            p0 = np.asarray(l.point, float)
            c0 = 0.5 * (V.min(axis=0) + V.max(axis=0))
            t = float(np.dot(c0 - p0, a))
            col = (0.55, 0.15, 0.65)
            lines.append({"a": p0 + (t - 0.6 * diag) * a, "b": p0 + (t + 0.6 * diag) * a, "color": col, "width": 2.5})
            markers.append({"xyz": p0 + (t + 0.6 * diag) * a, "text": f"rotation axis{cs(case)}", "color": col})
            swatches.append((col, f"rotation{cs(case)}: {rotation_text(l, fmt)}"))
        elif l.kind == "gravity":
            g = np.asarray(l.vector, float)
            ng = np.linalg.norm(g)
            if ng > 0:
                c0 = 0.5 * (V.min(axis=0) + V.max(axis=0))
                arrows.append({"tip": c0, "vec": g / ng * L, "text": f"g = {fmt(ng)} mm/s²", "color": (0.3, 0.3, 0.3)})
        elif l.kind == "thermal":                          # thermal load – only a description in the legend
            swatches.append(((0.95, 0.55, 0.45), f"temperature{cs(case)}: {thermal_load_text(l)}"))

    # ---------------------------------------------------------------- thermal boundary conditions
    n_th = 0
    th_items = [(c.name, bc) for c in th_cases for bc in c.loads]
    th_items.sort(key=lambda it: it[1].kind != "heat_generation")    # a part with a source first, the faces through it
    by_kind: dict[str, int] = {}
    for _, bc in th_items:
        by_kind[bc.kind] = by_kind.get(bc.kind, 0) + 1
    pal = {k: _palette(n, *THERMAL_HUES.get(k, (0.0, 0.1)), 0.7, 0.9) for k, n in by_kind.items()}
    used = {k: 0 for k in by_kind}
    multi_th = len(th_cases) > 1
    sel_marker: dict[str, dict] = {}                   # more conditions on the same face → one mark
    for case, bc in th_items:
        col = pal[bc.kind][used[bc.kind]]
        used[bc.kind] += 1
        tag = f" [{case}]" if multi_th else ""
        label = THERMAL_LABELS.get(bc.kind, bc.kind)
        val = bc_value_text(bc)
        sel = getattr(bc, "selection", None)
        if sel is not None:
            if sel not in st.selections:
                continue
            msk = faces_mask(st.selections[sel].faces)
            colors[msk] = col
            p, nrm = anchor(msk)
            if bc.kind in ("heat_flux", "heat_flow") and bc.value:
                dvec = -np.asarray(nrm) * np.sign(bc.value) * L * 0.7     # + into the body: the arrow ends on the face
                out_flow = bc.value < 0
                arrows.append({"tip": p + dvec if out_flow else p, "vec": dvec, "text": f"{sel}: {val}{tag}",
                               "color": col, "label_at": "tip" if out_flow else "tail"})
            elif sel in sel_marker:
                sel_marker[sel]["text"] += f"; {val}{tag}"
            else:
                sel_marker[sel] = {"xyz": p, "text": f"{sel}: {val}{tag}", "color": col}
                markers.append(sel_marker[sel])
            swatches.append((col, f"{label} {sel}{tag}: {val}"))
            n_th += 1
        elif bc.kind == "heat_generation":
            names = [p.name for p in sess.parts]
            if bc.part not in names:
                continue
            msk = own[:, 0] == names.index(bc.part)
            colors[msk] = col
            p, _ = anchor(msk)
            markers.append({"xyz": p, "text": f"source {fmt(bc.value)} W ({bc.part}){tag}", "color": col})
            swatches.append((col, f"{label}{tag}: {val}"))
            n_th += 1
    if th_cases and st.thermal.conductances:            # interface conductances (all thermal cases): a mark on side A
        names = [p.name for p in sess.parts]
        for cc in st.thermal.conductances:
            if cc.selection in names:
                msk = own[:, 0] == names.index(cc.selection)
            elif cc.selection in st.selections:
                msk = faces_mask(st.selections[cc.selection].faces)
            else:
                continue
            p, _ = anchor(msk)
            markers.append({"xyz": p, "text": f"{cc.name}: {bc_value_text(cc)}", "color": (0.55, 0.35, 0.1)})
            swatches.append(((0.55, 0.35, 0.1), f"contact conductance {cc.name}: {bc_value_text(cc)}"))
            n_th += 1
    f8_notes: list[str] = []
    if show_mech and (st.contacts or st.masses or st.dynamics.probes):   # F8: contact pairs, probes, added mass
        from .f8_scenes import model_overlay
        try:
            f8_notes = model_overlay(sess, st, colors, own, faces_mask, anchor, markers, swatches)
        except Exception as e:  # noqa: BLE001 – the addition must not bring down the picture of the model
            f8_notes = [f"Contacts / masses cannot be displayed: {e}"]
    if highlight:
        hs = resolve_faces(sess, highlight)
        colors[faces_mask(hs)] = (1.0, 0.0, 1.0)
        swatches.append(((1.0, 0.0, 1.0), "highlighted faces"))
    if show_face_ids:
        for pi in range(npart):
            pname = sess.parts[pi].name
            if face_id_part is not None and pname != face_id_part:
                continue
            for fi in range(len(cache[pi])):
                p, _ = anchor((own[:, 0] == pi) & (own[:, 1] == fi))
                markers.append({"xyz": p, "text": face_id(pname, fi), "color": (0.2, 0.2, 0.2), "only_visible": True})
    if show_mech:
        title = [f"Study model “{st.name}” – supports and loads" + (f" (state {load_case})" if load_case else ""),
                 f"{npart} part(s), {len(cons)} supports, {len(loads)} loads"
                 + (f", {n_th} thermal boundary conditions" if n_th else "")]
        info = ["Colored faces = support selections (blue) and load selections (red/orange); arrows = load direction"]
    else:
        title = [f"Study model “{st.name}” – thermal boundary conditions (thermal case {thermal_case})",
                 f"{npart} part(s), {n_th} thermal boundary conditions"]
        info = []
    if n_th:
        info += ["Heat: purple = prescribed temperature, orange = flux/power (arrow into the body),",
                 "teal = convection, yellow = radiation, pink = part with heat generation"]
    if st.contacts and show_mech:
        info.append("Contact pairs coloured by type (teal = frictionless, yellow = frictional, magenta = interference)")
    info += f8_notes
    beams = []
    if st.frame.members and show_mech:                 # mixed model: bars, supports and loads of the bars
        from .frame_scenes import frame_overlay
        try:
            ov = frame_overlay(sess, diag, load_case)
            beams = ov["beams"]
            markers += ov["markers"]
            arrows += ov["arrows"]
            swatches += ov["swatches"]
            title[1] += f", {len(st.frame.members)} members"
        except Exception as e:  # noqa: BLE001 – a faulty bar must not bring down the picture of the bodies
            info.append(f"Members cannot be displayed: {e}")
    hidden = [pi for pi, p in enumerate(sess.parts) if p.name in set(st.hidden_parts)]
    if hidden and len(hidden) < npart:                 # parts hidden in the GUI (Study.hidden_parts) – only the display
        keep = ~np.isin(own[:, 0], hidden)
        T, colors, group = T[keep], colors[keep], group[keep]
        info.append(f"Hidden parts ({len(hidden)}): " + ", ".join(sess.parts[i].name for i in hidden[:8])
                    + (" …" if len(hidden) > 8 else ""))
    return Scene(V=V, T=T, tri_colors=colors, tri_group=group, title=title, info=info, legend_title="Legend",
                 swatches=swatches, markers=markers, arrows=arrows, lines=lines, shade_range=(0.55, 1.0),
                 beams=beams)
