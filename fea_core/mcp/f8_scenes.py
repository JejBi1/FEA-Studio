"""Scenes F8 for the headless render: the nodal field on the surface of the chosen parts (contact pressure, state of the contact, the field of the dynamics
on the (deformed) mesh) and the additions of the picture of the model (contact pairs, probes, added masses).
"""
from __future__ import annotations

import numpy as np

from ..contour import band_edges, linear_triangles
from .render import Scene, fmt
from .session import FeaError

# state of the contact (contacts.results.STATUS): −1 not a contact node, 0 out of contact, 1 sliding, 2 sticking
STATUS_COLORS = {-1: (0.80, 0.80, 0.82), 0: (0.35, 0.60, 0.95), 1: (0.97, 0.62, 0.15), 2: (0.20, 0.70, 0.35)}
STATUS_LABELS = {-1: "outside contact surfaces", 0: "not in contact (open)", 1: "sliding", 2: "sticking"}


def field_scene(sess, values: np.ndarray | None, label: str, unit: str, source: str, U: np.ndarray | None = None,
                scale: float | None = None, parts: list[str] | None = None, nbands: int = 12, vmin=None, vmax=None,
                show_mesh: bool = False, signed: bool = False, status: np.ndarray | None = None,
                info: list[str] | None = None, mark_min: bool = False) -> tuple[Scene, dict]:
    """Surface of the parts `parts` (None = all) with the nodal field `values` (bands) or the discrete state of the contact `status`;
    U = displacements for the deformed shape (None = undeformed), scale = magnification (None = auto 5 % of the size).
    """
    from .diagnostics import mesh_cache
    from .scenes import _tri6_edges
    res = sess.require_results()
    m = res.mesh
    if not len(m.nodes):
        from .diagnostics import no_solid_hint
        raise FeaError(no_solid_hint(res))
    tri, elem = mesh_cache(res).boundary
    if parts:
        idx = [m.part_names.index(p) for p in parts if p in m.part_names]
        if not idx:
            raise FeaError(f"Parts {', '.join(parts)} are not in the mesh (parts: {', '.join(m.part_names)})")
        keep = np.isin(m.tet_part[elem], idx)
        tri, elem = tri[keep], elem[keep]
    if not len(tri):
        raise FeaError("The selected parts have no surface in the mesh")
    T = linear_triangles(tri)
    rep = 4 if tri.shape[1] == 6 else 1
    group = np.repeat(m.tet_part[elem], rep)
    used = np.unique(T)
    remap = np.full(len(m.nodes), -1, np.int64)
    remap[used] = np.arange(len(used))
    Tc = remap[T]
    X = m.nodes[used]
    diag = float(np.linalg.norm(X.max(axis=0) - X.min(axis=0))) or 1.0
    k = 0.0
    umax = 0.0
    if U is not None:
        Uc = np.asarray(U, float)[used]
        umax = float(np.linalg.norm(Uc, axis=1).max()) if len(Uc) else 0.0
        k = (0.05 * diag / umax if umax > 0 else 0.0) if scale is None else float(scale)
    V = X + k * Uc if k > 0 else X
    title = [f"{label}" + (f" [{unit}]" if unit and unit != "–" else "") + f" – {source}"]
    defo = f"deformation {fmt(k, 3)}× (max. displacement {fmt(umax)} mm)" if k > 0 else "undeformed shape"
    inf = [f"{sess.study.name}: parts {(', '.join(parts) if parts else 'all')}; {len(m.nodes)} mesh nodes",
           "Units: mm, N, MPa"] + list(info or [])
    meta = {"field": label, "unit": unit, "deformation_scale": k, "parts": parts or list(m.part_names)}
    if status is not None:
        st = np.asarray(status, float)[T].astype(int)
        med = np.sort(st, axis=1)[:, 1]                            # majority of three corners
        colors = np.array([STATUS_COLORS.get(int(s), STATUS_COLORS[-1]) for s in med], float)
        present = sorted(set(med.tolist()))
        sw = [(STATUS_COLORS[s], STATUS_LABELS[s]) for s in (2, 1, 0, -1) if s in present]
        counts = {STATUS_LABELS[s]: int((np.asarray(status)[used] == s).sum()) for s in (2, 1, 0)}
        title.append("nodes: " + ", ".join(f"{k_} {v}" for k_, v in counts.items()) + " | " + defo)
        scene = Scene(V=V, T=Tc, tri_colors=colors, tri_group=group, title=title, info=inf,
                      legend_title="Contact state", swatches=sw, shade_range=(0.62, 1.0),
                      mesh_edges=_tri6_edges(remap[tri]) if show_mesh else None)
        meta["node_counts"] = counts
        return scene, meta
    vals = np.asarray(values, float)[used]
    lo = float(np.nanmin(vals)) if vmin is None else float(vmin)
    hi = float(np.nanmax(vals)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0) * 1e-6
    edges = band_edges(lo, hi, nbands)
    imax, imin = int(np.nanargmax(vals)), int(np.nanargmin(vals))
    us = "" if unit in ("", "–") else f" {unit}"
    markers = [{"key": "max", "xyz": V[imax], "text": f"max {fmt(vals[imax])}{us}", "color": (0.85, 0.1, 0.1)}]
    if signed or mark_min:
        markers.append({"key": "min", "xyz": V[imin], "text": f"min {fmt(vals[imin])}{us}", "color": (0.1, 0.3, 0.85)})
    title.append(f"max {fmt(float(np.nanmax(vals)))}{us}, min {fmt(float(np.nanmin(vals)))}{us} | {defo}")
    if vmin is not None or vmax is not None:
        inf.insert(0, "Scale range set manually – values outside it fall in the end bands")
    short = label.split(" (")[0]
    scene = Scene(V=V, T=Tc, values=vals, tri_group=group, ghost=X if k > 0 else None,
                  mesh_edges=_tri6_edges(remap[tri]) if show_mesh else None, title=title, info=inf,
                  legend_title=f"{short}\n[{unit}]" if len(short) > 16 else f"{short} [{unit}]", edges=edges,
                  markers=markers)
    meta.update({"max": float(np.nanmax(vals)), "min": float(np.nanmin(vals)), "max_at_mm": m.nodes[used[imax]],
                 "min_at_mm": m.nodes[used[imin]], "band_edges": edges})
    return scene, meta


def contact_side_parts(study, mesh, pairs: list | None = None, side: str = "slave") -> list[str] | None:
    """Parts on the side of the contact pairs in the mesh of the result: slave | master (the faces of the pair `mesh.report["contacts"]` – ccx
    writes the pressure and the state of the contact on the slave face) or a | b (the definition of the pair). None = cannot be determined.
    """
    names = None if pairs is None else {c.name for c in pairs}
    out: list[str] = []
    if side in ("slave", "master"):
        for r in (mesh.report or {}).get("contacts", []):
            if names is not None and r.get("pair") not in names:
                continue
            key = r.get(side)
            if not key or key not in mesh.surfaces:
                continue
            nodes = np.unique(mesh.surfaces[key])
            hit = np.isin(mesh.tets, nodes).any(axis=1)
            out += [mesh.part_names[i] for i in np.unique(mesh.tet_part[hit])]
    else:
        from ..contacts.detect import pair_parts
        for c in (pairs if pairs is not None else study.active_contacts()):
            try:
                pa, pb = pair_parts(study, c)
            except Exception:  # noqa: BLE001
                continue
            out += sorted(pa if side == "a" or not c.b else pb)
    out = [p for p in dict.fromkeys(out) if p in mesh.part_names]
    return out or None


def contact_view(mesh, pairs: list | None = None, side: str = "slave"):
    """Direction towards the camera from which the contact face is visible (from the centre of the parts of the side through the centroid of the contact faces,
    slightly rotated); a curved / closed face (a pin in a hole) → "iso".
    """
    names = None if pairs is None else {c.name for c in pairs}
    pts, cents = [], []
    for r in (mesh.report or {}).get("contacts", []):
        if names is not None and r.get("pair") not in names:
            continue
        key = r.get(side if side in ("slave", "master") else "slave")
        if not key or key not in mesh.surfaces:
            continue
        nodes = np.unique(mesh.surfaces[key])
        hit = np.isin(mesh.tets, nodes).any(axis=1)
        own = np.unique(mesh.tets[np.isin(mesh.tet_part, np.unique(mesh.tet_part[hit]))])
        pts.append(mesh.nodes[nodes].mean(axis=0))
        cents.append(mesh.nodes[own].mean(axis=0))
    if not pts:
        return "iso"
    d = np.mean(pts, axis=0) - np.mean(cents, axis=0)
    size = float(np.linalg.norm(np.ptp(mesh.nodes, axis=0))) or 1.0
    if np.linalg.norm(d) < 1e-3 * size:
        return "iso"
    d = d / np.linalg.norm(d)
    a = np.array([1.0, -1.0, 1.0]) / np.sqrt(3)
    t = a - (a @ d) * d
    if np.linalg.norm(t) < 1e-6:
        t = np.array([1.0, 0.0, 0.0]) - d[0] * d
    t /= np.linalg.norm(t)
    v = d + 0.45 * t
    return [float(x) for x in v / np.linalg.norm(v)]


# ===================================================================== picture of the model – contacts, probes, masses

def facing_faces(sess, pa: int, pb: int, fa: list[int], fb: list[int], tol: float) -> tuple[list[int], list[int]]:
    """Contact faces without the faces that touch the other part only by an edge (find_touching takes all of them that are
    within the tolerance): a planar face stays if it has a counterpart = a planar face of the other part with the opposite normal in the
    same plane (within tol); curved faces (a pin in a hole) stay all.
    """
    A, B = sess.face_props(pa), sess.face_props(pb)

    def keep(own, other, idx, oidx):
        out = []
        for i in idx:
            p = own[i]
            if p["type"] != "plane":
                out.append(i)
                continue
            n, c = np.asarray(p["normal"], float), np.asarray(p["centroid"], float)
            for j in oidx:
                q = other[j]
                if q["type"] != "plane":
                    continue
                if float(np.dot(n, q["normal"])) < -0.98 and abs(float(np.dot(np.asarray(q["centroid"]) - c, n))) <= tol:
                    out.append(i)
                    break
        return out
    return keep(A, B, fa, fb), keep(B, A, fb, fa)


def model_overlay(sess, st, colors, own, faces_mask, anchor, markers: list, swatches: list) -> list[str]:
    """Adds to the picture of the model the contact pairs (colour by type), the probes of the dynamics and the added masses. Returns the notes."""
    from .f8_common import TYPE_COLORS, type_label, mu_info, mass_text, touching
    notes = []
    names = [p.name for p in sess.parts]
    tol = None
    for c in st.contacts:
        if not c.enabled:
            continue
        col = TYPE_COLORS.get(c.kind, (0.5, 0.5, 0.5))
        sides = [c.a] + ([c.b] if c.b else [])
        mask = np.zeros(len(own), bool)
        part_sides = [s for s in sides if s not in st.selections]
        for s in sides:
            if s in st.selections:
                mask |= faces_mask(st.selections[s].faces)
        if len(part_sides) == 2 and all(s in names for s in part_sides):
            if tol is None:
                from ..contacts.detect import default_tol
                tol = default_tol(st)
            try:
                hit = touching(sess, part_sides[0], part_sides[1], tol)
            except Exception:  # noqa: BLE001 – OCP failed → only the mark
                hit = None
            if hit:
                ia, ib = names.index(part_sides[0]), names.index(part_sides[1])
                fa, fb = facing_faces(sess, ia, ib, hit["faces_a"], hit["faces_b"], tol)
                for pi, fl in ((ia, fa), (ib, fb)):
                    mask |= (own[:, 0] == pi) & np.isin(own[:, 1], fl)
            else:
                notes.append(f"Contact '{c.name}': the parts do not touch within the tolerance {tol:g} mm – faces are not highlighted")
        elif part_sides and not mask.any():
            for pn in part_sides:
                if pn in names:
                    mask |= own[:, 0] == names.index(pn)
        if mask.any():
            colors[mask] = col
        p, _ = anchor(mask)
        mu = mu_info(st, c).get("mu") if c.kind == "frictional" else None
        extra = f", μ = {mu:g}" if mu is not None else ""
        if c.clearance is not None:
            extra += f", interference/clearance {c.clearance:g} mm"
        markers.append({"xyz": p, "text": f"contact {c.name} ({type_label(c.kind)}{extra})", "color": col})
        swatches.append((col, f"contact {c.name}: {type_label(c.kind)} ({c.a}" + (f" – {c.b})" if c.b else ")")
                         + extra))
    ds = st.dynamics
    for pr in ds.probes:
        col = (0.10, 0.45, 0.15)
        if pr.point is not None:
            xyz = np.asarray(pr.point, float)
        elif pr.selection in st.selections:
            xyz, _ = anchor(faces_mask(st.selections[pr.selection].faces))
        else:
            continue
        markers.append({"xyz": xyz, "text": f"probe {pr.name} ({pr.quantity})", "color": col})
    if ds.probes:
        swatches.append(((0.10, 0.45, 0.15), f"dynamics probes ({len(ds.probes)})"))
    for mm in getattr(st, "masses", []):
        col = (0.55, 0.35, 0.15)
        msk = faces_mask(st.selections[mm.selection].faces) if mm.selection in st.selections else None
        if mm.kind == "surface_mass" and msk is not None:
            colors[msk] = (0.80, 0.65, 0.45)
            col = (0.80, 0.65, 0.45)
        if getattr(mm, "point", None) is not None:
            xyz = np.asarray(mm.point, float)
        elif msk is not None:
            xyz, _ = anchor(msk)
        else:
            continue
        markers.append({"xyz": xyz, "text": f"mass {mm.name or mm.selection}: {mass_text(mm)}", "color": col})
        swatches.append((col, f"mass {mm.name or mm.selection}: {mass_text(mm)}"))
    return notes
