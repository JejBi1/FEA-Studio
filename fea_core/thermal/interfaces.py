"""Interfaces with a thermal contact conductance (D1): resolution to mesh faces and the heat that flows through them.

Sources of a conductance (h_c [W/(m²·K)], heat flux q = h_c·(T_slave − T_master) across the interface):
1. `ContactPair.conductance` of an active contact pair (faces `__c{i}_s/_m` of the pair, Contacts step);
2. `ThermalSettings.conductances` (Heat step, MCP `fea_thermal_add_contact_conductance`) – `ContactConductance` with
   - two PART names: a bonded interface. A shared (conforming) mesh has one set of nodes on the interface, there is no
     temperature jump possible → one of the parts must be meshed separately (`MeshSettings.separate`, `choose_separate`); its
     *TIE faces (fea_core.ties) towards the other part become the conductance faces `__tc{i}_s/_m` and are left out of the thermal
     *TIE (the mechanical *TIE stays – the mechanics is unchanged except for the non-conforming mesh);
   - two face selections (faces of separately meshed parts), or a contact pair name in `selection` (then `master` is unused);
3. `ContactConductance` among the conditions of a thermal case (older files) – resolved the same way, silently skipped when
   it cannot be resolved (previous behaviour).
The conductances are model data (*CONTACT PAIR in the model file) – they apply to ALL thermal cases.

Heat through an interface (`interface_flows`): Q = h_c·∫(T_s − T_m) dA over the slave faces with the consistent tri6 weights
(corners 0, mid-side nodes A/3), T_m interpolated on the nearest master triangle (quadratic tri6 functions). Positive Q =
heat flows from the slave side (`from`) to the master side (`to`). An independent check of the ccx *GAP CONDUCTANCE result
(verification TH8, TH8B: Q = ΔT / ΣR to 1e-3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TC_PREFIX = "__tc"


class ConductanceError(ValueError):
    """A defined contact conductance cannot be applied to the mesh (it would silently become a perfect contact)."""


@dataclass
class Interface:
    name: str           # name shown to the user (contact pair / conductance name)
    slave: str          # surface in mesh.surfaces
    master: str
    h_c: float          # W/(m²·K)
    source: str         # "contact_pair" | "interface" | "case"


# ------------------------------------------------------------------ helpers for the GUI / MCP (no mesh)

def is_part_pair(study, a: str, b: str, part_names) -> bool:
    return a in part_names and b in part_names and a != b


def clamped_parts(study) -> set[str]:
    """Parts with a bolt imprint – a separately meshed part cannot have one (contacts.check)."""
    out = set()
    for b in study.active_bolts():
        out |= {pl.get("part") for pl in b.plates if pl.get("part")}
        out |= {p for p in (b.head_part, b.nut_part) if p}
    return out


def choose_separate(study, a: str, b: str) -> str | None:
    """Part to mesh separately for a bonded-interface conductance between parts a and b (None = one already is).
    Prefers b (the second side); a part with a bolt imprint is not chosen. ConductanceError when neither can be separate.
    """
    sep = set(study.mesh.separate)
    if a in sep or b in sep:
        return None
    bolted = clamped_parts(study)
    for p in (b, a):
        if p not in bolted:
            return p
    raise ConductanceError(f"Contact conductance between '{a}' and '{b}': both parts carry bolt imprints and cannot be meshed separately")


def side_parts(study, side: str, part_names) -> list[str]:
    """Parts of a side: the part itself, or the parts of the faces of a selection."""
    if side in part_names:
        return [side]
    sel = study.selections.get(side)
    return list(dict.fromkeys(f.part for f in sel.faces)) if sel is not None else []


def suggest(study, a: str, b: str, part_names, pressure_mpa: float = 1.0, roughness_um: float = 3.0,
            tim: str | None = None) -> dict:
    """INDICATIVE h_c for the sides a, b from their materials (contact_conductance.suggest_contact_conductance)."""
    from .contact_conductance import suggest_contact_conductance

    def mat(side):
        for p in side_parts(study, side, part_names):
            try:
                return study.material_of(p).name
            except KeyError:
                pass
        return ""
    ma, mb = mat(a), mat(b)
    h = suggest_contact_conductance(ma or "steel", mb or "steel", pressure_mpa, roughness_um, tim)
    return {"h_W_m2K": h, "materials": [ma, mb], "pressure_MPa": pressure_mpa, "roughness_um": roughness_um,
            "tim": tim, "note": "indicative estimate (±50 % and more) – measure or take a datasheet value when it matters"}


def set_interface(study, a: str, b: str, h: float, name: str | None, part_names) -> tuple[object, str | None, bool]:
    """Adds / replaces the interface conductance a–b in ThermalSettings.conductances (the same name or the same pair of sides
    = replacement). For two parts one of them is meshed separately (MeshSettings.separate). Returns (item, the part made
    separate or None, replaced). ConductanceError on wrong input.
    """
    from .model import ContactConductance
    if not h or h <= 0:
        raise ConductanceError("h_c must be a positive contact conductance [W/(m²·K)]")
    if not a or not b or a == b:
        raise ConductanceError("Two different sides (parts or face selections) are required")
    for s in (a, b):
        if s not in part_names and s not in study.selections:
            raise ConductanceError(f"'{s}' is neither a part nor a face selection")
        if s in study.excluded_parts():
            raise ConductanceError(f"Part '{s}' is not meshed (bolt / omitted part)")
    name = (name or f"{a} – {b}").strip()
    th = study.thermal
    old = [c for c in th.conductances if c.name == name or {c.selection, c.master} == {a, b}]
    sep = choose_separate(study, a, b) if is_part_pair(study, a, b, part_names) else None
    if sep:
        study.mesh.separate = [*study.mesh.separate, sep]
    item = ContactConductance(a, b, conductance=float(h), name=name)
    th.conductances = [c for c in th.conductances if not any(c is o for o in old)] + [item]
    return item, sep, bool(old)


def check_definitions(study, part_names: list[str]) -> tuple[list[str], list[str]]:
    """Readiness check of ThermalSettings.conductances (errors, warnings) – no mesh needed."""
    errors, warnings = [], []
    sels = set(study.selections)
    pairs = {c.name for c in study.contacts}
    excl = study.excluded_parts()
    names = [c.name for c in study.thermal.conductances]
    for n in sorted({n for n in names if names.count(n) > 1}):
        errors.append(f"Contact conductance '{n}' is defined more than once")
    for c in study.thermal.conductances:
        a, b = c.selection, c.master
        if not c.conductance or c.conductance <= 0:
            errors.append(f"Contact conductance '{c.name}': h_c must be positive")
        if a in pairs and not b:
            continue
        if not a or not b or a == b:
            errors.append(f"Contact conductance '{c.name}': two different sides (parts or face selections) are required")
            continue
        for s in (a, b):
            if s not in part_names and s not in sels:
                errors.append(f"Contact conductance '{c.name}': '{s}' is neither a part nor a face selection")
            elif s in excl:
                errors.append(f"Contact conductance '{c.name}': part '{s}' is not meshed")
        if is_part_pair(study, a, b, part_names):
            if a not in study.mesh.separate and b not in study.mesh.separate:
                errors.append(f"Contact conductance '{c.name}': parts '{a}' and '{b}' share a mesh (perfect contact) – mesh one of them separately (Joints step, *TIE list)")
            if study.contact_split() == "conformal_split" and any(
                    {x.a, x.b} == {a, b} for x in study.active_contacts()):
                warnings.append(f"Contact conductance '{c.name}': the contact pair of these parts uses a conforming split mesh – the conductance applies only to separately meshed parts")
        for x in study.active_contacts():
            if {x.a, x.b} == {a, b} and x.conductance:
                warnings.append(f"Contact conductance '{c.name}' duplicates h_c of contact pair '{x.name}' – the contact pair value is used")
    return errors, warnings


# ------------------------------------------------------------------ resolution on the mesh

def _part_nodes(mesh) -> list[np.ndarray]:
    out = []
    for pi in range(len(mesh.part_names)):
        m = np.zeros(len(mesh.nodes), bool)
        m[np.unique(mesh.tets[mesh.tet_part == pi])] = True
        out.append(m)
    return out


def _touching(mesh, src: np.ndarray, dst: np.ndarray, tol: float) -> np.ndarray:
    """Rows of the triangles src whose centroid lies within tol of some triangle of dst."""
    from scipy.spatial import cKDTree
    from ..ties import _point_tri_dist
    X = mesh.nodes
    if not len(src) or not len(dst):
        return np.zeros(0, np.int64)
    cs, cd = X[src[:, :3]].mean(axis=1), X[dst[:, :3]].mean(axis=1)
    span = float(np.linalg.norm(X[dst[:, 0]] - X[dst[:, 1]], axis=1).max())
    tree = cKDTree(cd)
    ok = np.zeros(len(src), bool)
    for i, cand in enumerate(tree.query_ball_point(cs, r=span + tol)):
        if not cand:
            continue
        t = dst[np.asarray(cand)]
        d = _point_tri_dist(np.repeat(cs[i][None], len(t), 0), X[t[:, 0]], X[t[:, 1]], X[t[:, 2]])
        ok[i] = d.min() <= tol
    return np.nonzero(ok)[0]


def _part_pair(study, mesh, a: str, b: str, k: int, name: str) -> tuple[str, str]:
    """Conductance faces of the bonded interface of the separately meshed parts a, b: outer faces of a lying on the outer
    surface of b and vice versa (tolerance of fea_core.ties). Not the *TIE faces – ties.detect leaves out faces with
    supported nodes, the interface would lose area (an edge strip on a symmetry plane)."""
    from ..ties import tolerance
    ia, ib = mesh.part_names.index(a), mesh.part_names.index(b)
    tri, elem = mesh.boundary_faces()
    part = mesh.tet_part[elem]
    ta, tb = tri[part == ia], tri[part == ib]
    tol = tolerance(study, mesh)
    sl, ms = ta[_touching(mesh, ta, tb, tol)], tb[_touching(mesh, tb, ta, tol)]
    if not len(sl) or not len(ms):
        pn = _part_nodes(mesh)
        if (pn[ia] & pn[ib]).any():
            raise ConductanceError(f"Contact conductance '{name}': parts '{a}' and '{b}' share a mesh – mesh one of them separately")
        raise ConductanceError(f"Contact conductance '{name}': parts '{a}' and '{b}' do not touch (no common faces within {tol:.3g} mm)")
    # slave = side A (the heat through the interface is then reported from A to B, as the user defined it); for the
    # *GAP CONDUCTANCE with TIED overclosure the choice of the slave side does not change the result (TH8B both ways)
    s_name, m_name = f"{TC_PREFIX}{k}_s", f"{TC_PREFIX}{k}_m"
    mesh.surfaces[s_name], mesh.surfaces[m_name] = sl, ms
    return s_name, m_name


def resolve(study, mesh) -> list[Interface]:
    """All the conductance interfaces of the study on the mesh (creates the faces `__tc{i}_s/_m` for part pairs).
    ConductanceError for a ThermalSettings.conductances item that cannot be applied.
    """
    out: list[Interface] = []
    seen = set()
    report: dict[str, dict] = {}
    for r in mesh.report.get("contacts", []):
        for key in (r.get("pair"), r.get("name")):
            if key:
                report[key] = r

    def add(name, sl, ms, hc, source):
        if sl and ms and sl in mesh.surfaces and ms in mesh.surfaces and (sl, ms) not in seen:
            out.append(Interface(name, sl, ms, float(hc), source))
            seen.add((sl, ms))
            return True
        return False

    # 1. contact pairs with h_c
    for c in getattr(study, "contacts", []):
        if not getattr(c, "enabled", True):
            continue
        hc = getattr(c, "conductance", None)
        if hc is None or float(hc) <= 0:
            continue
        sl = ms = None
        if c.name in report:
            sl, ms = report[c.name].get("slave"), report[c.name].get("master")
        elif c.a in mesh.surfaces and c.b in mesh.surfaces:
            sl, ms = c.a, c.b
        elif c.a in mesh.part_names or c.b in mesh.part_names:
            for r in mesh.report.get("contacts", []):
                if (r.get("part_a") in (c.a, c.b) and r.get("part_b") in (c.a, c.b)) or r.get("name") == c.name:
                    sl, ms = r.get("slave"), r.get("master")
                    break
        add(c.name, sl, ms, hc, "contact_pair")

    # 2. interfaces of the Heat step (strict) and 3. conditions in the thermal cases (older files, lenient)
    items = [(c, True) for c in study.thermal.conductances]
    items += [(bc, False) for case in study.thermal.cases for bc in case.loads if getattr(bc, "kind", "") == "contact_conductance"]
    k = 0
    pair_sides = [{c.a, c.b} for c in study.active_contacts() if c.conductance]
    for bc, strict in items:
        if not bc.conductance or bc.conductance <= 0:
            continue
        a, b = bc.selection, bc.master
        if {a, b} in pair_sides:
            continue                                   # the contact pair's own h_c is used (check_definitions warns)
        name = bc.name if strict else (bc.name if bc.name != "contact_cond" else a)
        if a in report:
            ok = add(name, report[a].get("slave"), report[a].get("master"), bc.conductance, "interface" if strict else "case")
        elif a in mesh.surfaces and b in mesh.surfaces:
            ok = add(name, a, b, bc.conductance, "interface" if strict else "case")
        elif a in mesh.part_names and b in mesh.part_names and a != b:
            k += 1
            try:
                sl, ms = _part_pair(study, mesh, a, b, k, name)
            except ConductanceError:
                if strict:
                    raise
                continue
            ok = add(name, sl, ms, bc.conductance, "interface" if strict else "case")
        else:
            ok = False
        if strict and not ok and (a not in report):
            raise ConductanceError(f"Contact conductance '{name}': sides '{a}' and '{b}' are not found in the mesh")
    return out


def tri_keys(tri: np.ndarray) -> set[tuple]:
    return {tuple(r) for r in np.sort(tri[:, :3], axis=1).tolist()}


def without(tri: np.ndarray, keys: set[tuple]) -> np.ndarray:
    """Triangles of tri not in keys (corner sets)."""
    if not keys or not len(tri):
        return tri
    srt = np.sort(tri[:, :3], axis=1).tolist()
    keep = np.array([tuple(r) not in keys for r in srt], bool)
    return tri[keep]


# ------------------------------------------------------------------ heat through the interfaces

def _mid_pairs(X: np.ndarray, tri: np.ndarray) -> list[tuple[int, int]]:
    """Corner pair of each mid-side node 3, 4, 5 of tri6 (determined geometrically on the first triangle)."""
    t = tri[0]
    out = []
    for m in (3, 4, 5):
        best = min(((i, j) for i, j in ((0, 1), (1, 2), (2, 0))),
                   key=lambda ij: np.linalg.norm(X[t[m]] - 0.5 * (X[t[ij[0]]] + X[t[ij[1]]])))
        out.append(best)
    return out


def _interp_on(X: np.ndarray, tri: np.ndarray, vals: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Values at the points pts interpolated on the nearest triangle of tri (tri6 quadratic, tri3 linear)."""
    from scipy.spatial import cKDTree
    a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    cen = (a + b + c) / 3
    k = min(12, len(tri))
    _, cand = cKDTree(cen).query(pts, k=k)
    cand = np.asarray(cand).reshape(len(pts), k)
    best_d = np.full(len(pts), np.inf)
    best_l = np.zeros((len(pts), 3))
    best_t = np.zeros(len(pts), np.int64)
    for j in range(k):
        t = cand[:, j]
        A, B, C = a[t], b[t], c[t]
        e0, e1, v = B - A, C - A, pts - A
        d00, d01, d11 = (e0 * e0).sum(1), (e0 * e1).sum(1), (e1 * e1).sum(1)
        d20, d21 = (v * e0).sum(1), (v * e1).sum(1)
        den = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-300, d00 * d11 - d01 * d01, 1e-300)
        l1 = (d11 * d20 - d01 * d21) / den
        l2 = (d00 * d21 - d01 * d20) / den
        lam = np.clip(np.stack([1 - l1 - l2, l1, l2], 1), 0.0, None)
        lam /= lam.sum(1, keepdims=True)
        q = lam[:, :1] * A + lam[:, 1:2] * B + lam[:, 2:] * C
        d = np.linalg.norm(pts - q, axis=1)
        better = d < best_d
        best_d[better], best_l[better], best_t[better] = d[better], lam[better], t[better]
    L = best_l
    T = tri[best_t]
    if tri.shape[1] >= 6:
        pairs = _mid_pairs(X, tri)
        N = [L[:, i] * (2 * L[:, i] - 1) for i in range(3)] + [4 * L[:, i] * L[:, j] for i, j in pairs]
        return sum(N[m] * vals[T[:, m]] for m in range(6))
    return sum(L[:, i] * vals[T[:, i]] for i in range(3))


def _node_weights(X: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(nodes, ∫N dA [mm²]) of the triangles – consistent weights (tri6: corners 0, mid-side A/3; tri3: A/3)."""
    from ..mesh import tri_areas
    A = tri_areas(X, tri)
    if tri.shape[1] >= 6:
        nodes, w = tri[:, 3:6].ravel(), np.repeat(A / 3, 3)
    else:
        nodes, w = tri[:, :3].ravel(), np.repeat(A / 3, 3)
    u, inv = np.unique(nodes, return_inverse=True)
    return u, np.bincount(inv, weights=w, minlength=len(u))


def _parts_of(mesh, tri: np.ndarray, pn: list[np.ndarray]) -> list[str]:
    nodes = np.unique(tri[:, :3])
    return [mesh.part_names[i] for i, m in enumerate(pn) if m[nodes].mean() > 0.5]


def interface_flows(study, mesh, T: np.ndarray, items: list[Interface] | None = None) -> list[dict]:
    """Heat [W] through each conductance interface (positive = from the `from` parts to the `to` parts)."""
    if items is None:
        items = resolve(study, mesh)
    if not items:
        return []
    X = mesh.nodes
    pn = _part_nodes(mesh)
    rows = []
    for it in items:
        st, mt = mesh.surfaces[it.slave], mesh.surfaces[it.master]
        nodes, w = _node_weights(X, st)
        Tm = _interp_on(X, mt, T, X[nodes])
        dT = T[nodes] - Tm
        area = float(w.sum())
        q = float((w * dT).sum())                     # K·mm²
        rows.append({"name": it.name, "from": ", ".join(_parts_of(mesh, st, pn)) or it.slave,
                     "to": ", ".join(_parts_of(mesh, mt, pn)) or it.master, "h_W_m2K": it.h_c,
                     "area_mm2": area, "dT_mean_K": q / area if area > 0 else 0.0,
                     "Q_W": it.h_c * q * 1e-6, "source": it.source})
    return rows
