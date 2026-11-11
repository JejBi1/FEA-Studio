"""Conversion of the 3D study to the 2D problem: edges of the selections, constraints, loads by states, materials.

Rules (see the description of the package):
  Fixed         u1 = u2 = 0 at the nodes of the edges of the selection.
  Displacement  component along e1 / e2 (global axis parallel to the axis of the section) → u1 / u2; a component perpendicular to the plane of the section is
                ignored (non-zero → warning); an oblique component → error.
  Pressure      constant → *DLOAD Pk on the faces of the elements; variable (gradient) → consistent nodal forces (*CLOAD).
  Force         total 3D force on the faces of the selection → uniform traction F / A_3D (A_3D = area of the faces of the selection in the CAD) along the
                edges of the section; axisym only the axial component (the radial one around the circumference cancels), planar only the components in the plane.
  Gravity       ρ·g at the Gauss points (axisym only the axial component); *CLOAD (= *DLOAD GRAV – verified exp2.py).
  Rotation      centrifugal force ρ·ω²·r (axisym: the axis of rotation = the axis of the problem; planar: the axis ⟂ to the plane, also the angular acceleration).
Axisym: nodes on the axis (r = 0) always have u_r = 0 (ccx does not constrain them itself); edges lying on the axis are not assigned to the selections.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh2d import Mesh2D, body_forces, edge_forces, edge_measure
from .model import Frame2D, TwoDError
from .section import FaceSet, selection_faces

AXIS_SET = "__osa"


@dataclass
class Bc2D:
    selection: str
    nodes: np.ndarray                 # 0-based nodes
    dofs: dict[int, float]            # 1 | 2 → prescribed displacement [mm]


@dataclass
class Step2D:
    name: str
    F: np.ndarray                     # (N, 2) all external nodal forces (also the pressure *DLOAD) – reactions and equilibrium
    cload: np.ndarray                 # (N, 2) forces written as *CLOAD
    dload: list[tuple[int, int, float]] = field(default_factory=list)   # (element 0-based, face 1..3, pressure MPa)


@dataclass
class Model2D:
    kind: str
    frame: Frame2D
    mesh: Mesh2D
    materials: list                   # Material per part (order of mesh.part_names)
    thickness: float
    bcs: list[Bc2D]
    steps: list[Step2D]
    edges: dict[str, np.ndarray]      # selection → rows of mesh.seg
    warnings: list[str]
    info: dict = field(default_factory=dict)


def weight_fn(fr: Frame2D, thickness: float):
    if fr.axisym:
        return lambda x: 2.0 * np.pi * x[..., 0]
    t = float(thickness)
    return lambda x: np.full(x.shape[:-1], t)


def _used_selections(study) -> list[str]:
    names = []
    for c in study.constraints:
        if getattr(c, "selection", None):
            names.append(c.selection)
    for lc in study.load_cases:
        for l in lc.loads:
            if getattr(l, "selection", None):
                names.append(l.selection)
    return list(dict.fromkeys(names))


def selection_edges(study, parts, mesh: Mesh2D, fr: Frame2D, names, warnings: list[str]) -> dict[str, np.ndarray]:
    """Selection → rows of mesh.seg (boundary segments on the faces of the selection). An edge belongs to the selection when all sampling points
    of its curve (exact geometry of Gmsh) lie on the faces of the selection.
    """
    size = max(mesh.size(), 1.0)
    tol = max(1e-5 * size, 1e-5)
    faces = selection_faces(study, parts, names)
    pts3 = fr.to3d(mesh.curve_pts)                           # (C, 7, 3)
    on_axis = np.zeros(len(mesh.curve_pts), bool)
    if fr.axisym:
        on_axis = np.all(np.abs(mesh.curve_pts[:, :, 0]) <= tol, axis=1)
    out = {}
    for name in names:
        sel = study.selections.get(name)
        if sel is None:
            raise TwoDError(f"Selection '{name}' does not exist")
        if sel.entity != "faces":
            raise TwoDError(f"Selection '{name}': in 2D only face selections are supported (not edges / points)")
        fs = FaceSet(faces[name], fr, tol)
        curves = [c for c in range(len(pts3)) if not on_axis[c] and fs.contains_all(pts3[c])]
        rows = np.nonzero(np.isin(mesh.seg_curve, curves))[0]
        if fs.skipped:
            warnings.append(f"Selection '{name}': {fs.skipped} planar faces lie in the section plane – not used in 2D")
        if not len(rows):
            warnings.append(f"Selection '{name}' has no edge in the section – omitted in 2D")
        out[name] = rows
    return out


def _seg_nodes(mesh: Mesh2D, rows) -> np.ndarray:
    return np.unique(mesh.seg[rows]) if len(rows) else np.zeros(0, np.int64)


def _component_dofs(fr: Frame2D, d: np.ndarray, v: float, name: str, warnings: list[str]) -> dict[int, float]:
    c1, c2, cn = float(d @ fr.e1), float(d @ fr.e2), float(d @ fr.n)
    if abs(abs(cn) - 1.0) < 1e-9:
        if v:
            warnings.append(f"Support '{name}': prescribed displacement {v:g} mm perpendicular to the section plane is ignored in 2D")
        return {}
    if abs(abs(c1) - 1.0) < 1e-9:
        return {1: v * np.sign(c1)}
    if abs(abs(c2) - 1.0) < 1e-9:
        return {2: v * np.sign(c2)}
    raise TwoDError(f"Support '{name}': the displacement component is oblique to the section axes – not yet supported in 2D (choose xdir / axis parallel to the displacement axis)")


def constraints_2d(study, mesh: Mesh2D, fr: Frame2D, edges, warnings: list[str]) -> list[Bc2D]:
    bcs = []
    for c in study.constraints:
        kind = getattr(c, "kind", type(c).__name__)
        if kind not in ("fixed", "displacement"):
            raise TwoDError(f"Support '{kind}' ({getattr(c, 'selection', '')}) is not yet supported in 2D – only fixed support and prescribed displacement")
        rows = edges.get(c.selection, np.zeros(0, np.int64))
        nodes = _seg_nodes(mesh, rows)
        if not len(nodes):
            continue
        if kind == "fixed":
            bcs.append(Bc2D(c.selection, nodes, {1: 0.0, 2: 0.0}))
            continue
        dofs: dict[int, float] = {}
        for k, comp in enumerate(("ux", "uy", "uz")):
            v = getattr(c, comp)
            if v is None:
                continue
            d = np.eye(3)[k]
            for dof, val in _component_dofs(fr, d, float(v), c.selection, warnings).items():
                if dof in dofs and abs(dofs[dof] - val) > 1e-12:
                    raise TwoDError(f"Support '{c.selection}': inconsistent displacement components in 2D")
                dofs[dof] = val
        if fr.axisym and 1 in dofs and dofs[1] != 0.0:
            warnings.append(f"Support '{c.selection}': a nonzero radial displacement = expansion of the whole circumference")
        if dofs:
            bcs.append(Bc2D(c.selection, nodes, dofs))
    if fr.axisym:
        tol = 1e-9 * max(mesh.size(), 1.0)
        axis = np.nonzero(np.abs(mesh.nodes[:, 0]) <= tol)[0]
        if len(axis):
            bcs.append(Bc2D(AXIS_SET, axis, {1: 0.0}))
    return bcs


def element_density(study, mesh: Mesh2D, materials) -> np.ndarray:
    rho = np.array([float(m.rho or 0.0) for m in materials])
    return rho[mesh.tri_part]


def _check_csys(l, what: str):
    if getattr(l, "csys", None):
        raise TwoDError(f"{what}: a local coordinate system is not yet supported in 2D")


def loads_2d(study, mesh: Mesh2D, fr: Frame2D, edges, materials, thickness: float, warnings: list[str],
             info: dict) -> list[Step2D]:
    from ..study import Force, Gravity, Pressure, Rotation
    w = weight_fn(fr, thickness)
    rho_e = element_density(study, mesh, materials)
    steps = []
    n = len(mesh.nodes)
    info.setdefault("loads", [])
    for lc in study.load_cases:
        F, CL = np.zeros((n, 2)), np.zeros((n, 2))
        dload = []
        for l in lc.loads:
            kind = getattr(l, "kind", "")
            if isinstance(l, Pressure):
                rows = edges.get(l.selection, np.zeros(0, np.int64))
                if l.variable:
                    g = np.asarray(l.gradient, float)
                    o = np.asarray(l.origin if l.origin is not None else [0, 0, 0], float)
                    if fr.axisym and np.linalg.norm(g - (g @ fr.e2) * fr.e2) > 1e-12 * max(np.linalg.norm(g), 1e-30):
                        warnings.append(f"Pressure '{l.selection}': the gradient has a component perpendicular to the axis – in 2D the pressure is evaluated in the section plane (it would vary around the circumference)")

                    def trac(x, nn, l=l, g=g, o=o):
                        lin = (fr.to3d(x) - o) @ g
                        p = l.value + (np.maximum(lin, 0.0) if l.clip else lin)
                        return -p[..., None] * nn
                    Fp = edge_forces(mesh, rows, trac, w)
                    F += Fp
                    CL += Fp
                else:
                    Fp = edge_forces(mesh, rows, lambda x, nn, p=float(l.value): -p * nn, w)
                    F += Fp
                    dload += [(int(mesh.seg_elem[r]), int(mesh.seg_face[r]), float(l.value)) for r in rows]
            elif isinstance(l, Force):
                _check_csys(l, f"Force '{l.selection}'")
                sel = study.selections.get(l.selection)
                if sel is None or sel.entity != "faces":
                    raise TwoDError(f"Force '{l.selection}': in 2D, only forces on faces are supported")
                rows = edges.get(l.selection, np.zeros(0, np.int64))
                parts_in = set(mesh.part_names)
                A3 = sum(fs.area for fs in sel.faces if fs.part in parts_in)
                if not len(rows) or A3 <= 0:
                    continue
                tau = np.asarray(l.vector, float) / A3
                t2 = fr.vec2d(tau)
                if fr.axisym:
                    rest = tau - t2[1] * fr.e2
                    t2 = np.array([0.0, t2[1]])
                    if np.linalg.norm(rest) > 1e-9 * np.linalg.norm(tau):
                        warnings.append(f"Force '{l.selection}': the component perpendicular to the axis cancels around the circumference – only the axial component in 2D")
                else:
                    if abs(float(tau @ fr.n)) > 1e-9 * np.linalg.norm(tau):
                        warnings.append(f"Force '{l.selection}': the component perpendicular to the section plane is ignored in 2D")
                Ff = edge_forces(mesh, rows, lambda x, nn, t2=t2: np.broadcast_to(t2, nn.shape), w)
                F += Ff
                CL += Ff
                meas = edge_measure(mesh, rows, w)
                info["loads"].append({"case": lc.name, "selection": l.selection, "kind": "force",
                                      "area_3d": A3, "area_2d_model": meas, "ratio": meas / A3,
                                      "force_2d": Ff.sum(axis=0).tolist()})
                if abs(meas / A3 - 1.0) > 0.01:
                    warnings.append(f"Force '{l.selection}' ({lc.name}): the 2D model represents {meas / A3:.4g}× the selection area in 3D (in 2D {('for the whole circumference' if fr.axisym else 'per thickness')} {np.linalg.norm(Ff.sum(axis=0)):.6g} N)")
            elif isinstance(l, Gravity):
                _check_csys(l, "Gravity")
                g = np.asarray(l.vector, float)
                g2 = fr.vec2d(g)
                if fr.axisym:
                    if np.linalg.norm(g - g2[1] * fr.e2) > 1e-9 * max(np.linalg.norm(g), 1e-30):
                        warnings.append("Gravity: the component perpendicular to the axis is not axisymmetric – only the axial component in 2D")
                    g2 = np.array([0.0, g2[1]])
                elif abs(float(g @ fr.n)) > 1e-9 * max(np.linalg.norm(g), 1e-30):
                    warnings.append("Gravity: the component perpendicular to the section plane is ignored in 2D")
                Fg = body_forces(mesh, rho_e, lambda x, g2=g2: np.broadcast_to(g2, x.shape), w)
                F += Fg
                CL += Fg
            elif isinstance(l, Rotation):
                Fr = _rotation(l, mesh, fr, rho_e, w)
                F += Fr
                CL += Fr
            else:
                raise TwoDError(f"Load '{kind}' is not yet supported in 2D (only force on face, pressure, gravity, rotation)")
        steps.append(Step2D(lc.name, F, CL, dload))
    if not steps:
        raise TwoDError("The study has no load case at all")
    return steps


def _rotation(l, mesh: Mesh2D, fr: Frame2D, rho_e: np.ndarray, w) -> np.ndarray:
    ax = np.asarray(l.axis, float)
    ax = ax / max(np.linalg.norm(ax), 1e-300)
    p0 = np.asarray(l.point, float)
    rho = rho_e.copy()
    if l.parts:
        keep = np.isin(np.asarray(mesh.part_names)[mesh.tri_part], list(l.parts))
        rho[~keep] = 0.0
    om2, al = float(l.omega) ** 2, float(l.alpha)
    if fr.axisym:
        d = p0 - fr.origin
        if abs(abs(float(ax @ fr.e2)) - 1.0) > 1e-9 or np.linalg.norm(d - (d @ fr.e2) * fr.e2) > 1e-6 * max(mesh.size(), 1):
            raise TwoDError("Rotation: the rotation axis must be the axis of the axisymmetric problem")
        if al:
            raise TwoDError("Rotation: angular acceleration produces hoop forces – not possible in an axisymmetric problem")
        return body_forces(mesh, rho, lambda x: np.stack([om2 * x[..., 0], np.zeros(x.shape[:-1])], -1), w)
    if abs(abs(float(ax @ fr.n)) - 1.0) > 1e-9:
        raise TwoDError("Rotation: the rotation axis must be perpendicular to the section plane")
    c = fr.to2d(p0)
    s = float(np.sign(ax @ fr.n))                     # α kolem +n: a = α n × r (n × (x, y) = (−y, x))

    def acc(x):
        r = x - c
        return om2 * r - al * s * np.stack([-r[..., 1], r[..., 0]], -1)
    return body_forces(mesh, rho, acc, w)


def materials_2d(study, part_names, warnings: list[str]) -> list:
    out = []
    for p in part_names:
        m = study.material_of(p)
        if m.is_print or m.is_ortho:
            warnings.append(f"Part '{p}': material '{m.name}' is anisotropic – treated as isotropic in 2D, E = {m.E:g} MPa, ν = {m.nu:g} (approximate)")
        out.append(m)
    return out


def build_model(study, parts, mesh: Mesh2D, fr: Frame2D, thickness: float, warnings: list[str]) -> Model2D:
    for key in ("bolts", "welds"):
        if any(getattr(b, "enabled", True) for b in getattr(study, key, []) or []):
            warnings.append("Bolts and welds are not modeled in 2D (parts are connected by the shared section mesh)")
            break
    info: dict = {}
    names = _used_selections(study)
    edges = selection_edges(study, parts, mesh, fr, names, warnings)
    materials = materials_2d(study, mesh.part_names, warnings)
    bcs = constraints_2d(study, mesh, fr, edges, warnings)
    if not [b for b in bcs if b.selection != AXIS_SET]:
        raise TwoDError("2D: no support has an edge in the section – the problem would be unsupported")
    steps = loads_2d(study, mesh, fr, edges, materials, thickness, warnings, info)
    return Model2D(fr.kind, fr, mesh, materials, float(thickness), bcs, steps, edges, warnings, info)
