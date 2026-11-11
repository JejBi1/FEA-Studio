"""Writing the CalculiX input file (.inp) from the study and the mesh.

Conventions:
  - nodes and elements numbered from 1 (index + 1)
  - each load state = one *STEP (unit states for superposition / fatigue)
  - force on a face = consistent nodal forces (tri6: corners 0, midpoints 1/3 of the element force)
  - pressure (constant and variable – linear, hydrostatic), force on an edge / at a point, force in a cylindrical system and
    rotation = consistent nodal forces *CLOAD (fea_core.loads); *DLOAD only for gravity (GRAV). Pressure was *DLOAD before –
    ccx integrated it inaccurately on irregular / curved tri6 (found by a cross-check 14. 10. 2026)
  - set names are generated (ccx does not take spaces / diacritics), map in InpMap
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import solver_name
from ..mesh import Mesh, tri_areas
from . import deck_text
from ..study import (Study, Fixed, Displacement, Force, Pressure, Gravity, RemoteLoad, BearingLoad, RemoteSupport,
                     Rotation)


@dataclass
class InpMap:
    """Link between the study and the names in the .inp (for reading results)."""
    nsets: dict[str, str] = field(default_factory=dict)        # selection -> NSET
    surfaces: dict[str, str] = field(default_factory=dict)     # selection -> SURFACE
    constraint_sets: list[str] = field(default_factory=list)   # NSET vazeb (reakce)
    ref_nodes: dict[str, dict] = field(default_factory=dict)   # load key -> {ref, rot}
    remote: dict[str, dict] = field(default_factory=dict)      # selection of a rigid remote support -> {ref, rot} (NSET NRSUP)
    steps: list[str] = field(default_factory=list)             # names of load states
    part_sets: dict[str, str] = field(default_factory=dict)    # part -> ELSET
    bolts: dict[str, dict] = field(default_factory=dict)       # bolt -> {"beam": [nodes], "ref": [...], "gap"}
    preload_step: bool = False                                 # last step = bolt preload
    el_print: str | None = None                                # ELSET s *EL PRINT S (spoje)
    el_print_section: str | None = None                        # ELSET with *EL PRINT S (section)
    el_print_strain: str | None = None                         # ELSET s *EL PRINT E/ME (3D tisk)
    frame: dict | None = None                                  # frame: {"fe": FrameFE, "base": 1st node}
    masses: list = field(default_factory=list)                 # point masses: [{"nodes", "point", "name"}]
    contacts: list = field(default_factory=list)               # contact pairs: [{"pair", "interaction", …}]


_ELEMENT_SCALE: "ElementScale | None" = None


class ElementScale:
    """Stiffness multipliers per element (topology optimisation, fea_core.topo): `with ElementScale(s): run_study(...)`.

    Element E = material E · s_e (ν the same, density · s_e); elements with s_e = 1 stay unchanged. Each distinct
    s_e = its own *MATERIAL and ELSET; if there are more distinct values than `max_levels` (continuous SIMP densities),
    they are rounded to `per_decade` levels per decade (BESO: 2 values, exact). Outside the context the .inp is written unchanged.
    `used` = number of .inp writes that used the multipliers (check that the calculation did not take another path,
    e.g. NGSolve). Isotropic materials without plasticity only; printing and orthotropy = error.
    """

    def __init__(self, scale, per_decade: int = 100, max_levels: int = 64) -> None:
        self.scale = np.asarray(scale, dtype=float)
        self.per_decade = per_decade
        self.max_levels = max_levels
        self.used = 0
        self._prev = None

    def __enter__(self) -> "ElementScale":
        global _ELEMENT_SCALE
        self._prev, _ELEMENT_SCALE = _ELEMENT_SCALE, self
        return self

    def __exit__(self, *exc) -> None:
        global _ELEMENT_SCALE
        _ELEMENT_SCALE = self._prev

    def levels(self, n_elements: int) -> tuple[np.ndarray, np.ndarray]:
        """(level indices (M,), level values); level with value 1 = unchanged."""
        s = self.scale
        if s.shape != (n_elements,):
            raise ValueError(f"Element stiffness multipliers have shape {s.shape}, the mesh has {n_elements} elements")
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("Element stiffness multipliers must be positive finite numbers")
        vals, idx = np.unique(s, return_inverse=True)
        if len(vals) > self.max_levels:                  # continuous densities (SIMP): rounding in log space
            q = np.round(np.log10(s) * self.per_decade) / self.per_decade
            vals, idx = np.unique(q, return_inverse=True)
            vals = 10.0 ** vals
        return idx.ravel(), vals


def _fmt(v: float) -> str:
    return f"{v:.12g}"


# Bulk writing of tables: row template × k rows % tuple of values. '%.12g' % x is the same as _fmt(x)
# (both PyOS_double_to_string 'g', precision 12) and '%d' % i is the same as str(i) → output byte-identical to writing
# row by row, but without a Python call on every number (nodes + elements of 197 k DOF: 0.52 → ~0.2 s)
_ROWS = 20000                                   # rows per formatting call (size of the template and of the tuple)


def _write_table(f, fmt: str, cols: list, n: int) -> None:
    """n rows according to the fmt template (one row); cols = columns (lists of int/float or range)."""
    nc = len(cols)
    for s in range(0, n, _ROWS):
        e = min(n, s + _ROWS)
        flat = [None] * ((e - s) * nc)
        for j, c in enumerate(cols):
            flat[j::nc] = c[s:e]
        f.write((fmt * (e - s)) % tuple(flat))


def _write_ids(f, ids, per_line: int = 16) -> None:
    """Numbers, per_line on each line, every line ends with a comma (NSET, ELSET)."""
    ids = np.asarray(ids, dtype=np.int64).ravel().tolist()
    full = len(ids) // per_line
    line = ", ".join(["%d"] * per_line) + ",\n"
    for s in range(0, full, _ROWS):
        k = min(full, s + _ROWS) - s
        f.write((line * k) % tuple(ids[s * per_line:(s + k) * per_line]))
    rest = ids[full * per_line:]
    if rest:
        f.write(", ".join(["%d"] * len(rest)) % tuple(rest) + ",\n")


def write_mesh(f, mesh: Mesh, m: InpMap) -> None:
    """*NODE (NSET=NALL), elements by part (ELSET P1…), ELSET EALL."""
    f.write("*NODE, NSET=NALL\n")
    n = len(mesh.nodes)
    X = np.asarray(mesh.nodes, dtype=float)
    _write_table(f, "%d, %.12g, %.12g, %.12g\n", [range(1, n + 1), *(X[:, k].tolist() for k in range(3))], n)
    etype = mesh.element_type
    npe = mesh.tets.shape[1]
    for pi, pname in enumerate(mesh.part_names):
        eset = f"P{pi + 1}"
        m.part_sets[pname] = eset
        f.write(f"** part {pi + 1}: {deck_text(pname)}\n")
        f.write(f"*ELEMENT, TYPE={etype}, ELSET={eset}\n")
        ids = np.nonzero(mesh.tet_part == pi)[0]
        conn = mesh.tets[ids].astype(np.int64) + 1
        _write_table(f, ", ".join(["%d"] * (npe + 1)) + "\n",
                     [(ids + 1).tolist(), *(conn[:, k].tolist() for k in range(npe))], len(ids))
    f.write("*ELSET, ELSET=EALL\n" + ", ".join(m.part_sets.values()) + "\n")


def write_sets(f, mesh: Mesh, m: InpMap, need_nset, need_surf, sel_ids: dict[str, str]) -> None:
    """NSET (nodes of the selection) and SURFACE (element faces) of the selections needed; names by sel_ids."""
    for name in need_nset | need_surf:
        if name not in mesh.surfaces:
            raise KeyError(f"Selection '{name}' is not in the mesh")
        if len(mesh.surfaces[name]) == 0:
            raise ValueError(f"Selection '{name}' has no faces in the mesh")
    for name in sorted(need_nset, key=lambda s: sel_ids[s]):
        ns = f"N{sel_ids[name]}"
        m.nsets[name] = ns
        f.write(f"** selection: {deck_text(name)}\n*NSET, NSET={ns}\n")
        _write_ids(f, mesh.surface_nodes(name) + 1)
    for name in sorted(need_surf, key=lambda s: sel_ids[s]):
        sn = f"S{sel_ids[name]}"
        m.surfaces[name] = sn
        elems, fno = mesh.element_faces(name)
        f.write(f"*SURFACE, NAME={sn}, TYPE=ELEMENT\n")
        _write_table(f, "%d, S%d\n", [(elems + 1).tolist(), fno.tolist()], len(elems))


def consistent_nodal_forces(nodes: np.ndarray, tri: np.ndarray, traction: np.ndarray) -> dict[int, np.ndarray]:
    """Nodal forces from a uniform load on a face [N/mm²] (tri6 or tri3)."""
    areas = tri_areas(nodes, tri)
    if tri.shape[1] == 6:
        w_nodes, w = tri[:, 3:6], 1.0 / 3.0
    else:
        w_nodes, w = tri[:, :3], 1.0 / 3.0
    unique_nodes, inv = np.unique(w_nodes, return_inverse=True)
    inv = inv.reshape(w_nodes.shape)
    acc = np.zeros((len(unique_nodes), 3))
    w_force = (areas * w)[:, None] * traction[None, :]
    # = np.add.at(acc, inv[:, k], w_force) per column k (same summation order → bitwise equal)
    order = inv.T.ravel()
    for c in range(3):
        acc[:, c] = np.bincount(order, weights=np.tile(w_force[:, c], w_nodes.shape[1]),
                                minlength=len(unique_nodes))
    return dict(zip(unique_nodes.tolist(), acc))


# penalty stiffness of the thermal gap relative to the axial stiffness of the bolt (gap = α·ΔT·L to 1e-4)
THERMAL_GAP_PENALTY = 1e4


def _write_bolts(f, bolts, m: InpMap, next_node: int, next_elem: int, surf_name,
                 thermal: bool = False, mass: bool = False) -> tuple[int, int]:
    """Equivalent bolt beams (U1) + rigid couplings to faces + preload node pairs (see joints.fem).

    thermal: in the cut there is a second gap t (u_M2 = u_M1 + (g + t)·e) with a SPRING1 spring to ground; the thermal
    elongation of the bolt δ = α·ΔT·L = force k·δ on t (penalty). A prescribed δ directly (*BOUNDARY on g ≠ 0) nor the
    expansion of the U1 element (*EXPANSION) WORKS: ccx then gives unbalanced reactions (verified 13. 10. 2026,
    scratch/thermal/bolt2, bolt3: S2 + δ = 0.0115 mm → 28 kN in the supports). A force on the gap (like the
    preload) is fine.
    mass: steel density of the beams (natural frequencies; the weight of bolts is not in statics – U1 cannot take GRAV).
    """
    if not bolts:
        return next_node, next_elem
    f.write("** ---------------- bolts: U1 beams (Timoshenko), rigid couplings, preload\n")
    f.write("*USER ELEMENT, TYPE=U1, INTEGRATION POINTS=2, MAXDOF=6, NODES=2\n")
    for b in bolts:
        ids = []
        f.write(f"** bolt {b.index}: {deck_text(b.name)}\n*NODE\n")
        for p in b.points:
            f.write(f"{next_node}, {_fmt(p[0])}, {_fmt(p[1])}, {_fmt(p[2])}\n")
            ids.append(next_node)
            next_node += 1
        beam = list(ids)
        info: dict = {"beam": beam, "ref": [], "gap": None, "m2": None, "tgap": None, "k_t": 0.0}
        if b.split is not None:
            m2, gap = next_node, next_node + 1
            next_node += 2
            p = b.points[b.split]
            f.write(f"{m2}, {_fmt(p[0])}, {_fmt(p[1])}, {_fmt(p[2])}\n{gap}, {_fmt(p[0])}, {_fmt(p[1])}, {_fmt(p[2])}\n")
            info.update(gap=gap, m2=m2)
            if thermal:
                info["tgap"] = next_node
                info["k_t"] = THERMAL_GAP_PENALTY * b.E * b.A / max(b.length, 1e-9)
                f.write(f"{next_node}, {_fmt(p[0])}, {_fmt(p[1])}, {_fmt(p[2])}\n")
                next_node += 1
        refs = []
        for key, k in b.couplings:
            p = b.points[k]
            f.write(f"{next_node}, {_fmt(p[0])}, {_fmt(p[1])}, {_fmt(p[2])}\n")
            refs.append((next_node, key, ids[k]))
            next_node += 1
        info["ref"] = [r for r, _, _ in refs]
        es = f"BOLT{b.index}"
        f.write(f"*ELEMENT, TYPE=U1, ELSET={es}\n")
        conn = list(ids)
        for j in range(len(conn) - 1):
            a, c = conn[j], conn[j + 1]
            if b.split is not None and j == b.split:
                a = info["m2"]                        # after the cut the second node of the pair follows
            f.write(f"{next_elem}, {a}, {c}\n")
            next_elem += 1
        if info["tgap"] is not None:
            f.write(f"*ELEMENT, TYPE=SPRING1, ELSET=BT{b.index}\n{next_elem}, {info['tgap']}\n"
                    f"*SPRING, ELSET=BT{b.index}\n1\n{_fmt(info['k_t'])}\n")
            next_elem += 1
        f.write(f"*MATERIAL, NAME=MB{b.index}\n*ELASTIC\n{_fmt(b.E)}, {_fmt(b.nu)}\n"
                + ("*DENSITY\n7.85e-09\n" if mass else ""))
        e = b.e / np.linalg.norm(b.e)
        t1 = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
        t1 /= np.linalg.norm(t1)
        f.write(f"*BEAM SECTION, SECTION=GENERAL, ELSET={es}, MATERIAL=MB{b.index}\n"
                f"{_fmt(b.A)}, {_fmt(b.I)}, 0., {_fmt(b.I)}, 0.9\n{_fmt(t1[0])}, {_fmt(t1[1])}, {_fmt(t1[2])}\n")
        for r, key, _ in refs:
            f.write(f"*COUPLING, REF NODE={r}, SURFACE={surf_name(key)}, CONSTRAINT NAME=CB{b.index}_{r}\n"
                    "*KINEMATIC\n1, 3\n")
        f.write("*EQUATION\n")
        for r, _, beam_node in refs:                 # node of the beam = auxiliary node of the constraint (DOF 1–6)
            for dof in range(1, 7):
                f.write(f"2\n{beam_node}, {dof}, 1., {r}, {dof}, -1.\n")
        if b.split is not None:                       # preload cut: u_M2 = u_M1 + g·e, same rotations
            m1, m2, gap, tg = ids[b.split], info["m2"], info["gap"], info["tgap"]
            for dof in range(1, 4):
                if abs(e[dof - 1]) > 1e-12 and tg is not None:
                    f.write(f"4\n{m2}, {dof}, 1., {m1}, {dof}, -1., {gap}, 1, {_fmt(-e[dof - 1])}, "
                            f"{tg}, 1, {_fmt(-e[dof - 1])}\n")
                elif abs(e[dof - 1]) > 1e-12:
                    f.write(f"3\n{m2}, {dof}, 1., {m1}, {dof}, -1., {gap}, 1, {_fmt(-e[dof - 1])}\n")
                else:
                    f.write(f"2\n{m2}, {dof}, 1., {m1}, {dof}, -1.\n")
            for dof in range(4, 7):
                f.write(f"2\n{m2}, {dof}, 1., {m1}, {dof}, -1.\n")
        m.bolts[b.name] = info
    return next_node, next_elem



def _write_bolt_connectors(f, bolts, m: InpMap, next_node: int, next_elem: int, mesh: Mesh) -> tuple[int, int]:
    """Bolts in a NONLINEAR calculation (contact, history): element U1 does not work in the nonlinear iterations of ccx 2.23
    (trial 15. 10. 2026, scratch/f8_cont/u1_nl.py: without contact imbalance 2.7 MN, with contact "*ERROR in
    extrapolate_u1") → connector: bearing faces as *RIGID BODY (REF + ROT node), between the REF nodes an axial spring
    k = E·A/L with the preload gap (u_M2 = u_M1 + g·e – as for the beam, *BOUNDARY, FIXED locks the length), transverse springs
    12·E·I/L³ and rotational E·I/L (bending), G·I_p/L (torsion) in the local bolt axes (SPRING2 + *ORIENTATION).
    """
    if not bolts:
        return next_node, next_elem
    f.write("** ---------------- bolts (nonlinear analysis): connector made of springs between rigid bearing faces\n")
    for b in bolts:
        if len(b.couplings) != 2:
            raise ValueError(f"{b.name}: a bolt with intermediate parts (clamped stack with clearance) is not yet supported in a nonlinear analysis – remove the clearance in CAD or compute linearly")
        e = b.e / np.linalg.norm(b.e)
        t1 = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0])
        t1 /= np.linalg.norm(t1)
        ends = []
        pn = b.points[b.couplings[1][1]]
        arm = None
        for j, (key, k) in enumerate(b.couplings):
            p = b.points[k]
            ref, rot = next_node, next_node + 1
            next_node += 2
            f.write(f"*NODE\n{ref}, {_fmt(p[0])}, {_fmt(p[1])}, {_fmt(p[2])}\n{rot}, {_fmt(p[0])}, {_fmt(p[1])}, "
                    f"{_fmt(p[2])}\n")
            extra = []
            if j == 0:      # arm: the node of the nut rigidly tied to the head – springs between matching nodes (otherwise a force couple)
                arm = next_node
                next_node += 1
                f.write(f"{arm}, {_fmt(pn[0])}, {_fmt(pn[1])}, {_fmt(pn[2])}\n")
                extra = [arm]
            f.write(f"*NSET, NSET=NBC{b.index}_{ref}\n")
            _write_ids(f, np.concatenate([mesh.surface_nodes(key) + 1, np.asarray(extra, dtype=np.int64)]))
            f.write(f"*RIGID BODY, NSET=NBC{b.index}_{ref}, REF NODE={ref}, ROT NODE={rot}\n")
            ends.append((ref, rot, p))
        (rh, roth, ph), (rn, rotn, pn) = ends
        m1, gap = next_node, next_node + 1
        next_node += 2
        f.write(f"*NODE\n{m1}, {_fmt(pn[0])}, {_fmt(pn[1])}, {_fmt(pn[2])}\n{gap}, {_fmt(pn[0])}, {_fmt(pn[1])}, "
                f"{_fmt(pn[2])}\n")
        L = max(float(np.linalg.norm(pn - ph)), 1e-6)
        G = b.E / (2 * (1 + b.nu))
        k_ax = b.E * b.A / L
        k_lat = 12 * b.E * b.I / L ** 3
        k_rot = b.E * b.I / L
        k_tor = G * 2 * b.I / L
        ori = f"OB{b.index}"
        f.write(f"*ORIENTATION, NAME={ori}, SYSTEM=RECTANGULAR\n"
                + ", ".join(_fmt(v) for v in (*e, *t1)) + "\n")
        for dof, k, (na, nb) in ((1, k_ax, (arm, m1)), (2, k_lat, (arm, m1)), (3, k_lat, (arm, m1)),
                                 (1, k_tor, (roth, rotn)), (2, k_rot, (roth, rotn)), (3, k_rot, (roth, rotn))):
            es = f"BS{b.index}_{next_elem}"
            f.write(f"*ELEMENT, TYPE=SPRING2, ELSET={es}\n{next_elem}, {na}, {nb}\n"
                    f"*SPRING, ELSET={es}, ORIENTATION={ori}\n{dof}, {dof}\n{_fmt(k)}\n")
            next_elem += 1
        # preload cut: u_RN = u_M1 + g·e (M1 dependent)
        f.write("*EQUATION\n")
        for dof in range(1, 4):
            if abs(e[dof - 1]) > 1e-12:
                f.write(f"3\n{m1}, {dof}, 1., {rn}, {dof}, -1., {gap}, 1, {_fmt(e[dof - 1])}\n")
            else:
                f.write(f"2\n{m1}, {dof}, 1., {rn}, {dof}, -1.\n")
        m.bolts[b.name] = {"beam": [], "ref": [rh, rn], "gap": gap, "m2": None, "tgap": None, "k_t": 0.0,
                           "connector": True}
    return next_node, next_elem


def _write_cylindrical(f, study: Study, mesh: Mesh, bolts) -> None:
    """Cylindrical supports as *EQUATION in global DOF (not *TRANSFORM – see fea_core.cylinder)."""
    from ..cylinder import support_equations
    eqs = support_equations(study, mesh)
    if not eqs:
        return
    # a dependent DOF must not be in another constraint equation (remote point, bolt) → clear error
    nodes = {n for n, _ in eqs}
    others = {l.selection for lc in study.load_cases for l in lc.loads if isinstance(l, RemoteLoad)}
    others |= {key for b in bolts for key, _ in b.couplings}
    others |= {c.selection for c in study.constraints if isinstance(c, RemoteSupport)}
    for name in sorted(others):
        if name in mesh.surfaces and nodes & set(mesh.surface_nodes(name).tolist()):
            raise ValueError(f"The cylindrical support shares nodes with face '{name}' (remote load / support, bolt) – ccx does not allow two constraint equations on one node. Separate the faces (different selection).")
    f.write("** cylindrical supports: u . e = 0 (e = radial / tangential / axial direction at the node)\n*EQUATION\n")
    for n, terms in eqs:
        f.write(f"{len(terms)}\n" + ", ".join(f"{n + 1}, {dof + 1}, {_fmt(c)}" for dof, c in terms) + "\n")


def _write_remote_supports(f, study: Study, mesh: Mesh, m: InpMap, next_node: int, bolts) -> int:
    """Remote constraints (fea_core.remote): rigid = *RIGID BODY (REF NODE displacements, ROT NODE rotations; *BOUNDARY on them
    is written in every step), distributing = dense *EQUATION from the RBE3 kinematics (point = weighted average of the face).
    """
    from ..cylinder import boundary_dofs
    from ..remote import remote_supports, check_overlaps, distributing_equations, RemoteError, MAX_DIST_FILL
    sups = remote_supports(study)
    if not sups:
        return next_node
    check_overlaps(study, mesh, bolts)
    bc = boundary_dofs(study, mesh)
    for c in sups:
        if c.coupling != "rigid":
            continue
        idx = mesh.surface_nodes(c.selection)
        if bc[idx].any():
            other = next((o.selection for o in study.constraints if isinstance(o, (Fixed, Displacement))
                          and np.intersect1d(mesh.surface_nodes(o.selection), idx).size), "?")
            raise RemoteError(f"Rigid remote support '{c.selection}' shares nodes with support '{other}' (fixed support / prescribed displacement) – ccx does not allow a rigid-support node in *BOUNDARY at the same time. Use a distributing support or separate the faces.")
        ref, rot = next_node, next_node + 1
        next_node += 2
        x, y, z = (float(v) for v in c.point)
        f.write(f"** remote support (rigid): {deck_text(c.selection)}\n"
                f"*NODE\n{ref}, {_fmt(x)}, {_fmt(y)}, {_fmt(z)}\n{rot}, {_fmt(x)}, {_fmt(y)}, {_fmt(z)}\n"
                f"*RIGID BODY, NSET={m.nsets[c.selection]}, REF NODE={ref}, ROT NODE={rot}\n")
        m.remote[c.selection] = {"ref": ref, "rot": rot}
    if m.remote:
        f.write("*NSET, NSET=NRSUP\n")
        _write_ids(f, [v for info in m.remote.values() for v in (info["ref"], info["rot"])])
    eqs = distributing_equations(study, mesh, bc)
    if eqs:
        fill = float(sum(len(t) ** 2 for t in eqs))
        if fill > MAX_DIST_FILL:
            n = len({(t[0]) for e in eqs for t in e})
            raise RemoteError(f"Distributing remote support: the face has too many nodes in the mesh ({n}) – dense coupling equations would fill the matrix in ccx (Σ terms² {fill:.2g} > {MAX_DIST_FILL:.2g}, memory grows with the square). Use a rigid support, a smaller face or a coarser local face mesh.")
        f.write("** remote supports (distributing): displacement / rotation of the point = weighted average of the face (RBE3)\n*EQUATION\n")
        for terms in eqs:
            f.write(f"{len(terms)}\n")
            for s in range(0, len(terms), 4):          # ccx: at most 12 items per line
                f.write(", ".join(f"{n + 1}, {d + 1}, {_fmt(c)}" for n, d, c in terms[s:s + 4]) + "\n")
    return next_node


def write_inp(study: Study, mesh: Mesh, path: str | Path,
              joint_elems: np.ndarray | None = None,
              temperatures: list[np.ndarray | None] | None = None,
              procedure: str = "static", eigen_line: str = "", modal_output: bool = False,
              storage: bool = False, prestress: bool = False, probe_nodes=None,
              nonlinear: dict | None = None, element_scale: "ElementScale | np.ndarray | None" = None) -> InpMap:
    """joint_elems: elements (0-based) for which the stress at integration points is printed (joints).
    temperatures: node temperatures [°C] per load state (None = no thermal load),
    see fea_core.thermal.field.node_temperatures; strain α·(T − reference_temperature).
    procedure: "static" (step per load state) | "modal" (*FREQUENCY) | "buckle" (*BUCKLE with the load of
    the state); eigen_line = data line under *FREQUENCY / *BUCKLE (number of modes…). For modal / buckle fea_core.eigen passes
    a copy of the study with a single state (modal: empty) and without the preload state.
    Dynamics (fea_core.dynamics.modal): modal_output = shapes U, modal reactions RF and stresses S; storage = *FREQUENCY,
    STORAGE=YES (.eig for ccx SSD / *MODAL DYNAMIC); prestress = procedure "modal" with the states of the study as static
    steps and natural frequencies in *STEP, PERTURBATION (stress stiffening, prestressed modal analysis);
    probe_nodes = probe nodes (0-based) → *NSET NPROBE.
    nonlinear = {"steps": [{"name", "kind" preload|case, "case_index", "scale", "spec" (LoadStep | None)}]} –
    nonlinear calculation (contact, NLGEOM, history): steps in this order instead of unit states, *STEP, INC=…
    with increment control, bolt preload as a step and in the other steps *BOUNDARY, FIXED on the cut gap (locked
    length), prescribed motion of rigid bodies, *CHANGE FRICTION / *MODEL CHANGE, contact output (fea_core.contacts).
    """
    from ..joints.fem import bolt_models, PRELOAD_CASE
    path = Path(path)
    m = InpMap()
    if element_scale is None:
        element_scale = _ELEMENT_SCALE                # context `with ElementScale(...)` (topology optimisation)
    elif not isinstance(element_scale, ElementScale):
        element_scale = ElementScale(element_scale)
    if element_scale is not None:
        element_scale.used += 1
    n_nodes = len(mesh.nodes)
    next_node = n_nodes + 1
    bolts = bolt_models(study, mesh) if study.active_bolts() else []
    m.preload_step = study.joints.preload_case and any(b.split is not None for b in bolts)

    # which selections need NSET / SURFACE
    need_nset, need_surf = set(), set()
    for c in study.constraints:
        need_nset.add(c.selection)
    for lc in study.load_cases:
        for l in lc.loads:
            if isinstance(l, Pressure) and not l.variable:
                need_surf.add(l.selection)
            elif isinstance(l, RemoteLoad):
                need_surf.add(l.selection)
                need_nset.add(l.selection)
    sel_ids = {name: f"SEL{i + 1}" for i, name in enumerate(study.selections)}
    for i, imp in enumerate(getattr(study, "imprints", []), 1):
        iname = getattr(imp, "name", None) or (imp.get("name") if isinstance(imp, dict) else "")
        if iname:
            sel_ids.setdefault(iname, f"IMP{i}")
    for b in bolts:                                  # constraint faces of bolts (selections created during meshing)
        for key, _ in b.couplings:
            need_surf.add(key)
            sel_ids.setdefault(key, f"B{b.index}{key.rsplit('_', 1)[-1].upper()}")
    for c in study.frame.connections if study.frame.active else []:   # connection of beams to a body face
        need_surf.add(c.selection)
    for r in mesh.report.get("contacts", []) if study.active_contacts() else []:   # contact faces (F8.5)
        if r.get("master"):
            need_surf |= {r["slave"], r["master"]}
            sel_ids.setdefault(r["slave"], f"C{int(r['index'])}S")    # int: the report may come from a file
            sel_ids.setdefault(r["master"], f"C{int(r['index'])}M")
    from ..ties import pairs as tie_pairs, write as write_ties
    for i, (sl, ms) in enumerate(tie_pairs(mesh), 1):   # *TIE of separately meshed parts (fea_core.ties)
        need_surf |= {sl, ms}
        sel_ids.setdefault(sl, f"TIE{i}S")
        sel_ids.setdefault(ms, f"TIE{i}M")
    with open(path, "w", encoding="ascii", errors="replace") as f:
        f.write(f"** FEA Studio – {deck_text(study.name)}\n** jednotky: mm, N, MPa, t, s\n")
        f.write("*HEADING\n" + deck_text(study.name) + "\n")

        # ------------------------------------------------------------ nodes, elements, sets
        write_mesh(f, mesh, m)
        write_sets(f, mesh, m, need_nset, need_surf, sel_ids)
        write_ties(f, mesh, m, lambda k: m.surfaces[k])

        # ------------------------------------------------------------ materials
        from ..printing import material_groups
        groups = material_groups(study, mesh)              # 3D printing: material by levels of the shell fraction
        printed = {g["part_index"] for g in groups}
        used: dict[str, str] = {}
        part_mid = []
        for pi, pname in enumerate(mesh.part_names):
            mat = study.material_of(pname)
            part_mid.append(None if pi in printed else used.setdefault(mat.name, f"M{len(used) + 1}"))
        thermal = temperatures is not None and any(t is not None for t in temperatures)
        t_ref = study.thermal.reference_temperature

        def expansion(mat) -> None:
            if thermal:
                f.write(f"*EXPANSION, ZERO={_fmt(t_ref)}\n{_fmt(mat.alpha or 0.0)}\n")
        from ..ortho import material_lines, axes as ortho_axes
        ortho_or: dict[str, str] = {}                  # material → *ORIENTATION (orthotropic material)
        for mname, mid in used.items():
            mat = study.materials[mname]
            f.write(f"** material: {deck_text(mname)}\n")
            if mat.is_ortho:
                f.write(f"*MATERIAL, NAME={mid}\n" + material_lines(mat, _fmt, thermal, t_ref))
            else:
                f.write(f"*MATERIAL, NAME={mid}\n*ELASTIC\n{_fmt(mat.E)}, {_fmt(mat.nu)}\n")
                expansion(mat)
            if getattr(mat, "is_plastic", False):
                from ..plasticity import format_plastic_ccx, PlasticitySpec
                pspec = PlasticitySpec.from_dict(mat.plastic) if isinstance(mat.plastic, dict) else mat.plastic
                for pl_line in format_plastic_ccx(pspec, mat.Re or 235.0, mat.E):
                    f.write(pl_line + "\n")
            if mat.rho > 0:
                f.write(f"*DENSITY\n{_fmt(mat.rho)}\n")
            if mat.is_ortho:
                system, vals = ortho_axes(study, mat)
                ortho_or[mid] = f"OR{mid}"
                f.write(f"*ORIENTATION, NAME=OR{mid}, SYSTEM={system}\n" + ", ".join(_fmt(v) for v in vals) + "\n")
        lv_idx = lv_val = None
        if element_scale is not None:
            lv_idx, lv_val = element_scale.levels(len(mesh.tets))
        for pi, mid in enumerate(part_mid):
            if mid is not None:
                orient = f", ORIENTATION={ortho_or[mid]}" if mid in ortho_or else ""
                part_lv = None
                if lv_idx is not None:
                    part_lv = lv_idx[mesh.tet_part == pi]
                    if np.all(lv_val[part_lv] == 1.0):
                        part_lv = None                 # part without a change of stiffness
                if part_lv is None:
                    f.write(f"*SOLID SECTION, ELSET=P{pi + 1}, MATERIAL={mid}{orient}\n\n")
                    continue
                # stiffness per element: the part is split into ELSET P{i}K{level}, each level has its own *MATERIAL
                mat = study.material_of(mesh.part_names[pi])
                if mat.is_ortho or getattr(mat, "is_plastic", False):
                    raise ValueError(f"Element stiffness multipliers (topology optimization) are only for isotropic elastic material; part '{mesh.part_names[pi]}' is orthotropic / plastic")
                ids = np.nonzero(mesh.tet_part == pi)[0]
                for k in np.unique(part_lv):
                    es = f"P{pi + 1}K{int(k)}"
                    f.write(f"*ELSET, ELSET={es}\n")
                    _write_ids(f, ids[part_lv == k] + 1)
                    sc = float(lv_val[k])
                    if sc == 1.0:
                        f.write(f"*SOLID SECTION, ELSET={es}, MATERIAL={mid}\n\n")
                        continue
                    f.write(f"*MATERIAL, NAME={mid}K{int(k)}\n*ELASTIC\n{_fmt(mat.E * sc)}, {_fmt(mat.nu)}\n")
                    expansion(mat)
                    if mat.rho > 0:
                        f.write(f"*DENSITY\n{_fmt(mat.rho * sc)}\n")
                    f.write(f"*SOLID SECTION, ELSET={es}, MATERIAL={mid}K{int(k)}\n\n")
        orients: dict[tuple, str] = {}         # (part, line class) → *ORIENTATION (axes 1, 2 = layer / line, 3 = Z)
        for g in groups:
            key = (g["part_index"], g.get("line_bin", -1))
            if key in orients:
                continue
            R = g["R"]
            orients[key] = f"OP{key[0] + 1}" + (f"L{key[1]}" if key[1] >= 0 else "")
            f.write(f"** 3D tisk: {deck_text(mesh.part_names[key[0]])}\n"
                    f"*ORIENTATION, NAME={orients[key]}, SYSTEM=RECTANGULAR\n"
                    + ", ".join(_fmt(v) for v in (*R[0], *R[1])) + "\n")
        for gi, g in enumerate(groups):
            es = f"P{g['part_index'] + 1}F{gi + 1}"
            C = g["C"]
            d = [C[0, 0], C[0, 1], C[1, 1], C[0, 2], C[1, 2], C[2, 2], C[3, 3], C[4, 4], C[5, 5]]
            f.write(f"** skorepina {g['f']:.3f}\n*ELSET, ELSET={es}\n")
            _write_ids(f, g["elems"] + 1)
            f.write(f"*MATERIAL, NAME=M{es}\n*ELASTIC, TYPE=ORTHO\n" + ", ".join(_fmt(v) for v in d[:8])
                    + f",\n{_fmt(d[8])}, 0.\n")
            expansion(study.material_of(mesh.part_names[g["part_index"]]))
            if g["rho"] > 0:
                f.write(f"*DENSITY\n{_fmt(g['rho'])}\n")
            f.write(f"*SOLID SECTION, ELSET={es}, MATERIAL=M{es}, "
                    f"ORIENTATION={orients[(g['part_index'], g.get('line_bin', -1))]}\n\n")

        if thermal:
            f.write(f"*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, {_fmt(t_ref)}\n")

        # ------------------------------------------------------------ remote points
        # rigid coupling: ONE *RIGID BODY per selection (ccx accepts a node in only one rigid body – a second one is silently not attached,
        # and the load of another state goes to a free node; found by a cross-check 14. 10. 2026, X4). Other rigid
        # loads of the same selection go to the same REF/ROT nodes, the moment recomputed to the body point: M + (p − p_ref) × F.
        rigid: dict[str, dict] = {}                   # selection → {"ref", "rot"}
        rigid_pt: dict[int, np.ndarray] = {}          # REF node of the rigid body → its point
        for li, (lc_i, l) in enumerate([(i, l) for i, lc in enumerate(study.load_cases)
                                        for l in lc.loads if isinstance(l, RemoteLoad)]):
            key = f"{lc_i}:{li}"
            if l.coupling == "rigid" and l.selection in rigid:
                m.ref_nodes[key] = rigid[l.selection]
                continue
            ref = next_node
            next_node += 1
            x, y, z = l.point
            f.write(f"*NODE\n{ref}, {_fmt(x)}, {_fmt(y)}, {_fmt(z)}\n")
            info = {"ref": ref}
            if l.coupling == "rigid":
                rot = next_node
                next_node += 1
                f.write(f"*NODE\n{rot}, {_fmt(x)}, {_fmt(y)}, {_fmt(z)}\n")
                f.write(f"*RIGID BODY, NSET={m.nsets[l.selection]}, REF NODE={ref}, ROT NODE={rot}\n")
                info["rot"] = rot
                rigid[l.selection] = info
                rigid_pt[ref] = np.asarray(l.point, float)
            else:
                f.write(f"*COUPLING, REF NODE={ref}, SURFACE={m.surfaces[l.selection]}, "
                        f"CONSTRAINT NAME=CPL{li + 1}\n*DISTRIBUTING\n1, 6\n")
            m.ref_nodes[key] = info
        ref_ids = list(dict.fromkeys(v for info in m.ref_nodes.values() for v in info.values()))
        if ref_ids:
            f.write("*NSET, NSET=NREF\n")
            _write_ids(f, ref_ids)
        for c in study.constraints:
            ns = m.nsets[c.selection]
            if ns not in m.constraint_sets:
                m.constraint_sets.append(ns)
        next_node = _write_remote_supports(f, study, mesh, m, next_node, bolts)
        if nonlinear is not None and bolts:           # U1 does not work in nonlinear iterations of ccx → connector from springs
            if thermal:
                raise ValueError("Bolts with a thermal load are not yet supported in a nonlinear analysis")
            next_node, next_elem = _write_bolt_connectors(f, bolts, m, next_node, len(mesh.tets) + 1, mesh)
        else:
            next_node, next_elem = _write_bolts(f, bolts, m, next_node, len(mesh.tets) + 1, lambda k: m.surfaces[k],
                                                thermal, mass=procedure == "modal")
        if study.frame.active:                          # frame (U1 elements, fea_core.beams.fem)
            from ..beams.fem import build as build_frame, write as write_frame
            fe = build_frame(study)
            m.frame = {"fe": fe, "base": next_node}
            next_node, next_elem = write_frame(f, study, fe, next_node, next_elem, _fmt, surf_name=lambda k: m.surfaces[k],
                                               declare_u1=not bolts, mass=procedure == "modal")
        from ..springs import write as write_springs
        next_node, next_elem, spr_aux = write_springs(f, study, mesh, next_node, next_elem, _fmt)
        if getattr(study, "masses", None):            # added mass (point / surface, fea_core.dynamics.mass)
            from ..dynamics.mass import write as write_masses
            from ..cylinder import boundary_dofs
            next_node, next_elem = write_masses(f, study, mesh, m, next_node, next_elem, _fmt,
                                                boundary_dofs(study, mesh))
        if spr_aux and thermal:                       # auxiliary spring nodes need an initial temperature
            f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNSPR, " + _fmt(t_ref) + "\n")
        tgaps = [info["tgap"] for info in m.bolts.values() if info.get("tgap") is not None]
        if tgaps:                                     # spring nodes (SPRING1) need an initial temperature
            f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n" + "".join(f"{g}, {_fmt(t_ref)}\n" for g in tgaps))
        _write_cylindrical(f, study, mesh, bolts)
        if study.active_contacts() and mesh.report.get("contacts"):     # contact pairs (fea_core.contacts)
            from ..contacts.inp import write_pairs
            next_node, next_elem = write_pairs(f, study, mesh, m, lambda k: m.surfaces[k], _fmt, next_node, next_elem)
        if joint_elems is not None and len(joint_elems):
            m.el_print = "EJOINT"
            f.write("*ELSET, ELSET=EJOINT\n")
            _write_ids(f, np.asarray(joint_elems) + 1)
        shell = mesh.cell_data.get("print_shell")      # printed elements (parts with printed material, −1 = not printed)
        if shell is not None and np.any(shell >= 0):
            print_elems = np.nonzero(np.asarray(shell) >= 0)[0]
            if len(print_elems):
                m.el_print_strain = "EPRINT"
                f.write("*ELSET, ELSET=EPRINT\n")
                _write_ids(f, print_elems + 1)
        if getattr(study, "section_forces", False):
            m.el_print_section = "ESECTION"
            f.write("*ELSET, ELSET=ESECTION\nEALL\n")
        gaps = [info["gap"] for info in m.bolts.values() if info["gap"] is not None]

        def bolt_elongation(k: int) -> dict[int, float]:
            """Force on the thermal gap of the bolts in state k: k_t·α·(T − T_ref)·L, T = mean of the nodes of the bolt constraints."""
            T = temperatures[k] if thermal and k < len(temperatures) else None
            if T is None:
                return {}
            out = {}
            for b in bolts:
                info = m.bolts[b.name]
                if info["tgap"] is None:
                    continue
                nodes = np.unique(np.concatenate([mesh.surface_nodes(key) for key, _ in b.couplings]))
                out[info["tgap"]] = info["k_t"] * b.alpha * (float(T[nodes].mean()) - t_ref) * b.length
            return out

        def boundary(op: str, lock_gaps: bool, zero: bool = False) -> None:
            """zero = prescribed displacements zero (the preload state is homogeneous, superposes)."""
            f.write(f"*BOUNDARY{op}\n")
            for c in study.constraints:
                ns = m.nsets[c.selection]
                if isinstance(c, Fixed):
                    f.write(f"{ns}, 1, 3\n")
                elif isinstance(c, Displacement):
                    for dof, v in ((1, c.ux), (2, c.uy), (3, c.uz)):
                        if v is not None:
                            f.write(f"{ns}, {dof}, {dof}, {_fmt(0.0 if zero else v)}\n")
                elif isinstance(c, RemoteSupport) and c.selection in m.remote:   # rigid: REF displacements, ROT rotations
                    info = m.remote[c.selection]
                    for k, on in enumerate(c.dofs):
                        if on:
                            f.write(f"{info['ref'] if k < 3 else info['rot']}, {k % 3 + 1}, {k % 3 + 1}, 0.\n")
            if m.frame:                               # support of nodes of the frame
                from ..beams.fem import boundary_lines
                for line in boundary_lines(m.frame["fe"], m.frame["base"]):
                    f.write(line + "\n")
            for c in m.contacts:                      # fixed support nodes in compression only (GAPUNI)
                if c.get("ground_set"):
                    f.write(f"{c['ground_set']}, 1, 3\n")
            if lock_gaps:
                for g in gaps:                        # preload cut closed (bolt continuous)
                    f.write(f"{g}, 1, 1, 0.\n")

        def temperature(k: int | None) -> None:
            """*TEMPERATURE of state k (OP=NEW = other nodes back to the initial = reference temperature)."""
            if not thermal:
                return
            T = temperatures[k] if k is not None and k < len(temperatures) else None
            f.write("*TEMPERATURE, OP=NEW\n")
            if T is None:
                f.write(f"NALL, {_fmt(t_ref)}\n")
                return
            idx = np.nonzero(T != t_ref)[0]
            if len(idx) == 0:
                f.write(f"NALL, {_fmt(t_ref)}\n")
            _write_table(f, "%d, %.12g\n", [(idx + 1).tolist(), np.asarray(T, float)[idx].tolist()], len(idx))

        # in branches the end of the step is enough (ccx always writes the end of a step, FREQUENCY = large number)
        fq = ", FREQUENCY=99999" if nonlinear is not None and study.analysis.branches else ""

        def outputs() -> None:
            # (sums of RF of constraints *NODE PRINT TOTALS=O\nY are not written – nothing reads them, reactions come from RF in the .frd)
            if m.frame:                               # frame: displacements and rotations of U1 nodes only in .dat
                f.write("*NODE PRINT, NSET=NFRAME\nU\n")
            if procedure != "static" and freq_step[0]:
                if modal_output:                      # dynamics: shapes normalised by mass, modal reactions and stresses
                    f.write("*NODE FILE\nU, RF\n*EL FILE\nS\n")
                    return
                f.write("*NODE FILE\nU\n")         # mode shapes: displacements only (normalisation arbitrary)
                return
            el_vars = "S, PEEQ" if any(getattr(m_, "is_plastic", False) for m_ in study.materials.values()) else "S"
            f.write(f"*NODE FILE{fq}\nU, RF\n*EL FILE{fq}\n{el_vars}\n")
            if ref_ids:
                f.write("*NODE PRINT, NSET=NREF\nU\n")
            if m.remote:                              # reaction of a rigid remote constraint: RF REF = force, RF ROT = moment
                f.write("*NODE PRINT, NSET=NRSUP\nU, RF\n")
            if m.el_print:
                el_fq = ", FREQUENCY=99999" if nonlinear is not None else ""     # end of the step only
                f.write(f"*EL PRINT, ELSET={m.el_print}{el_fq}\nS\n")
            if getattr(m, "el_print_section", None):
                el_fq = ", FREQUENCY=99999" if nonlinear is not None else ""
                f.write(f"*EL PRINT, ELSET={m.el_print_section}{el_fq}\nS\n")
            if getattr(m, "el_print_strain", None):
                el_fq = ", FREQUENCY=99999" if nonlinear is not None else ""
                strain_var = "ME" if thermal else "E"
                f.write(f"*EL PRINT, ELSET={m.el_print_strain}, GLOBAL=YES{el_fq}\n{strain_var}\n")

        if probe_nodes is not None and len(probe_nodes):
            f.write("*NSET, NSET=NPROBE\n")
            _write_ids(f, np.asarray(probe_nodes) + 1)
        freq_step = [not prestress]                   # step of natural frequencies (with preload last)
        stor = ", STORAGE=YES" if storage else ""

        def write_loads(si: int, op: str, scale: float = 1.0) -> None:
            """*CLOAD / *DLOAD of state si (× scale – history step); linear calculation scale = 1 (unchanged)."""
            lc = study.load_cases[si]
            # nodal forces: mesh nodes densely (acc + which components are loaded), other nodes (bolts, remote
            # points) in a dictionary; the summation in the load order as before per key (0.0 + v = v) → same values
            cload: dict[tuple[int, int], float] = dict(((g, 1), v) for g, v in bolt_elongation(si).items())
            acc = np.zeros((n_nodes, 3))
            has = np.zeros((n_nodes, 3), bool)

            def add_nodal(Fn: np.ndarray, rows: np.ndarray | None = None) -> None:
                nz = Fn != 0.0
                if rows is not None:
                    nz &= rows[:, None]
                acc[nz] = acc[nz] + Fn[nz]
                has[nz] = True
            dload: list[str] = []
            from ..loads import force_nodal, pressure_nodal, rotation_forces, global_vector, remote_vectors
            for l in lc.loads:
                if isinstance(l, Force):
                    add_nodal(force_nodal(study, mesh, l))
                elif isinstance(l, BearingLoad):
                    from ..cylinder import bearing_forces
                    Fb = bearing_forces(mesh, l.selection, global_vector(study, l))
                    add_nodal(Fb, np.abs(Fb).max(axis=1) > 0)
                elif isinstance(l, Pressure):
                    # even constant pressure by our integration (*CLOAD), not *DLOAD: ccx integrates pressure on irregular /
                    # curved tri6 faces inaccurately (cylinder ΔS 8e-4, print tolerance P7 3.6e-2; our
                    # integration 1e-5 – 14. 10. 2026)
                    Fp = pressure_nodal(mesh, l)
                    add_nodal(Fp, np.abs(Fp).max(axis=1) > 0)
                elif isinstance(l, Rotation):
                    Fw = rotation_forces(study, mesh, l)
                    add_nodal(Fw, np.abs(Fw).max(axis=1) > 0)
                elif l.kind == "inertia":                # inertia of the rigid motion (conversion from the kinematics)
                    from ..loads import inertia_forces
                    Fi = inertia_forces(study, mesh, l)
                    add_nodal(Fi, np.abs(Fi).max(axis=1) > 0)
                elif isinstance(l, Gravity):
                    g = global_vector(study, l)
                    gn = np.linalg.norm(g)
                    if gn > 0 and (bolts or m.frame):
                        # U1 elements (bolts, frames) cannot take body forces even outside their ELSET → nodal forces
                        from ..loads import gravity_nodal_forces
                        Fg = gravity_nodal_forces(study, mesh, g)
                        add_nodal(Fg, np.abs(Fg).max(axis=1) > 0)
                    elif gn > 0:
                        d = g / gn
                        dload.append(f"EALL, GRAV, {_fmt(gn)}, {_fmt(d[0])}, {_fmt(d[1])}, {_fmt(d[2])}")
            if m.masses or any(getattr(x, "kind", "") == "surface_mass" for x in getattr(study, "masses", [])):
                from ..dynamics.mass import static_forces    # weight / rotation / inertia of added mass
                Fm, pts = static_forces(study, mesh, lc)
                add_nodal(Fm, np.abs(Fm).max(axis=1) > 0)
                for i, k, fv in pts:
                    node = m.masses[i]["nodes"][k]
                    for d in range(3):
                        if fv[d]:
                            cload[(node, d + 1)] = cload.get((node, d + 1), 0.0) + float(fv[d])
            if m.frame:                               # loads of frames (nodal + consistent distributed + weight of members)
                from ..beams.fem import nodal_loads as frame_loads
                Ff = frame_loads(study, m.frame["fe"], lc)
                for n_, d in zip(*np.nonzero(Ff)):
                    key = (m.frame["base"] + int(n_), int(d) + 1)
                    cload[key] = cload.get(key, 0.0) + float(Ff[n_, d])
            # remote loads of this step
            remote =[l for l in lc.loads if isinstance(l, RemoteLoad)]
            keys = [k for k in m.ref_nodes if int(k.split(":")[0]) == si]
            for l, key in zip(remote, keys):
                info = m.ref_nodes[key]
                force, moment = remote_vectors(study, l)
                if info["ref"] in rigid_pt:              # shared rigid body: moment to its point
                    moment = np.asarray(moment, float) + np.cross(np.asarray(l.point, float) - rigid_pt[info["ref"]],
                                                                  np.asarray(force, float))
                for d in range(3):
                    if force[d]:
                        cload[(info["ref"], d + 1)] = cload.get((info["ref"], d + 1), 0.0) + float(force[d])
                    if moment[d]:
                        if "rot" in info:
                            k = (info["rot"], d + 1)
                        else:
                            k = (info["ref"], d + 4)
                        cload[k] = cload.get(k, 0.0) + float(moment[d])
            if scale != 1.0:
                acc *= scale
                cload = {k: v * scale for k, v in cload.items()}
                dload = [_scale_grav(line, scale) for line in dload]
            if cload or has.any() or op:
                f.write(f"*CLOAD{op}\n")
                nn, dd = np.nonzero(has)                  # line by line = sorted by (node, component)
                _write_table(f, "%d, %d, %.12g\n", [(nn + 1).tolist(), (dd + 1).tolist(), acc[nn, dd].tolist()],
                             len(nn))
                for (n, d), v in sorted(cload.items()):  # nodes outside the mesh (numbers greater than the mesh nodes) come after them
                    f.write(f"{n}, {d}, {_fmt(v)}\n")
            if dload or op:
                f.write(f"*DLOAD{op}\n")
                for line in dload:
                    f.write(line + "\n")

        # ------------------------------------------------------------ kroky
        for si, lc in enumerate(study.load_cases if nonlinear is None else []):
            m.steps.append(lc.name)
            op = "" if si == 0 else ", OP=NEW"
            f.write(f"** load case {si + 1}: {deck_text(lc.name)}\n")
            if procedure == "modal" and not prestress:
                f.write(f"*STEP\n*FREQUENCY, SOLVER={solver_name(study.solver.solver)}{stor}\n{eigen_line}\n")
            elif procedure == "buckle":
                f.write(f"*STEP\n*BUCKLE, SOLVER={solver_name(study.solver.solver)}\n{eigen_line}\n")
            else:
                f.write(f"*STEP\n*STATIC, SOLVER={solver_name(study.solver.solver)}\n")
            # constraints (in every step again, OP=NEW from the 2nd step)
            boundary(op, True)
            temperature(si)
            write_loads(si, op)
            outputs()
            f.write("*END STEP\n")
        if nonlinear is not None:
            _write_nonlinear_steps(f, study, m, nonlinear, bolts, gaps, boundary, temperature, write_loads, outputs,
                                   solver_name(study.solver.solver))
        if procedure == "modal" and prestress:
            # prestressed modal analysis: natural frequencies with the stiffness from the end of the last static step
            freq_step[0] = True
            m.steps.append("Natural frequencies (prestress)")
            f.write(f"** natural frequencies after the preload\n*STEP, PERTURBATION\n*FREQUENCY, "
                    f"SOLVER={solver_name(study.solver.solver)}{stor}\n{eigen_line}\n")
            outputs()
            f.write("*END STEP\n")
        if m.preload_step and nonlinear is None:
            # separate state: preload F_V in all bolts (force on the cut gap), otherwise no load
            m.steps.append(PRELOAD_CASE)
            op = "" if not study.load_cases else ", OP=NEW"
            f.write(f"** bolt preload\n*STEP\n*STATIC, SOLVER={solver_name(study.solver.solver)}\n")
            boundary(op, False, zero=True)
            temperature(None)
            f.write(f"*CLOAD{op}\n")
            for b in bolts:
                info = m.bolts[b.name]
                if info["gap"] is not None:
                    f.write(f"{info['gap']}, 1, {_fmt(-b.preload)}\n")
            if op:
                f.write(f"*DLOAD{op}\n")
            outputs()
            f.write("*END STEP\n")
    return m


def _scale_grav(line: str, scale: float) -> str:
    """*DLOAD GRAV line × scale (magnitude of the acceleration)."""
    p = [x.strip() for x in line.split(",")]
    if len(p) >= 3 and p[1].upper() == "GRAV":
        p[2] = _fmt(float(p[2]) * scale)
        return ", ".join(p)
    return line


def _write_nonlinear_steps(f, study, m, nl: dict, bolts, gaps, boundary, temperature, write_loads, outputs,
                           solver: str) -> None:
    """Steps of the nonlinear calculation (fea_core.contacts.run): bolt preload, steps of states / history."""
    from ..contacts.inp import step_header, outputs as contact_outputs, change_cards
    steps = nl["steps"]
    # amplitudes of prescribed motions (model data – they cannot be before the first step, so in the step with the step name)
    amp_lines = []
    for k, s in enumerate(steps):
        spec = s.get("spec")
        for j, mo in enumerate(getattr(spec, "prescribed", None) or []):
            name = f"MOT{k + 1}_{j + 1}"
            amp_lines.append((k, j, name, mo))
    if amp_lines:
        for k, j, name, mo in amp_lines:
            f.write(f"*AMPLITUDE, NAME={name}\n" + "".join(f"{_fmt(float(a))}, {_fmt(float(b))}\n"
                                                          for a, b in mo.amplitude))
    preloaded = False
    dyn_any = any(getattr(s.get("spec"), "dynamic", False) for s in steps)
    iv = getattr(study.analysis, "initial_velocity", None) or {}
    if iv:                                            # initial velocity of parts (*DYNAMIC, impact)
        f.write("*INITIAL CONDITIONS, TYPE=VELOCITY\n")
        for part, v in iv.items():
            if part not in nl.get("part_nodes", {}):
                raise ValueError(f"Initial velocity: part '{part}' is not in the mesh")
            for n_ in nl["part_nodes"][part]:
                for d in range(3):
                    if float(v[d]):
                        f.write(f"{int(n_) + 1}, {d + 1}, {_fmt(float(v[d]))}\n")
    for k, s in enumerate(steps):
        m.steps.append(s["name"])
        op = "" if k == 0 else ", OP=NEW"
        spec = s.get("spec")
        head, direct, inc = step_header(study, spec)
        if direct == "DYNAMIC":                       # *DYNAMIC (HHT-α) – implicit dynamics with contact
            f.write(f"** step {k + 1}: {deck_text(s['name'])}\n{head}\n"
                    f"*DYNAMIC, ALPHA={_fmt(float(spec.alpha))}, SOLVER={solver}\n{inc}\n")
        else:
            f.write(f"** step {k + 1}: {deck_text(s['name'])}\n{head}\n"
                    f"*STATIC, SOLVER={solver}{direct}\n{inc}\n")
        mode = "preload" if s["kind"] == "preload" else (getattr(spec, "bolts", "lock") if spec else "lock")
        if mode == "preload":
            boundary(op, False)
        else:
            boundary(op, not preloaded and mode != "free")
            if preloaded and gaps and mode == "lock":
                f.write("*BOUNDARY, FIXED\n" + "".join(f"{g}, 1, 1\n" for g in gaps))
        for kk, j, name, mo in amp_lines:
            if kk != k:
                continue
            tgt = m.remote.get(mo.target)
            if tgt is None:
                raise ValueError(f"Prescribed motion '{mo.target}': the target must be a rigid remote support (RemoteSupport rigid) – the motion is prescribed at its reference point")
            node = tgt["ref"] if mo.kind == "disp" else tgt["rot"]
            f.write(f"*BOUNDARY, AMPLITUDE={name}\n{node}, {int(mo.dof)}, {int(mo.dof)}, 1.\n")
        if spec is not None:
            f.write(change_cards(study, m, spec))
        ci = s.get("case_index")
        temperature(ci)
        if ci is not None:
            write_loads(ci, op, float(s.get("scale", 1.0)))
            if mode == "preload" and gaps:
                f.write("*CLOAD\n")
                for b in bolts:
                    info = m.bolts[b.name]
                    if info["gap"] is not None:
                        f.write(f"{info['gap']}, 1, {_fmt(-b.preload)}\n")
        else:
            f.write(f"*CLOAD{op}\n")
            if mode == "preload":
                for b in bolts:
                    info = m.bolts[b.name]
                    if info["gap"] is not None:
                        f.write(f"{info['gap']}, 1, {_fmt(-b.preload)}\n")
            if op:
                f.write(f"*DLOAD{op}\n")
        if mode == "preload":
            preloaded = True
        outputs()
        f.write(contact_outputs(study, m))
        if k == 0 and dyn_any:
            # ccx: energy output in a nonlinear calculation must be already in the 1st step, otherwise *DYNAMIC after *STATIC ends with an
            # error (verified scratch/f8_review/dyn2.py)
            f.write("*EL PRINT, ELSET=EALL, TOTALS=ONLY\nENER\n")
        f.write("*END STEP\n")
