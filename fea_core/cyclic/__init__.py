"""Cyclic symmetry (F9 C3): one sector of a rotationally periodic part (flange with n holes, impeller, disc).

Model: the user models one sector; its two cut faces `left` and `right` lie in half-planes through the axis and
right = left rotated by +360°/N about the axis (right-hand rule; the orientation of the axis is fixed automatically).
The mesh of the two faces is made periodic in Gmsh (`gmsh.model.mesh.setPeriodic`, fea_core.mesh.periodic) → the nodes
match exactly after the rotation.

Statics (and linear buckling – only cyclic-symmetric modes): loads equal on every sector, constraint equations in GLOBAL DOF
u_left = Rᵀ·u_right (ccx *EQUATION, dependent = the left node; the same approach as the cylindrical support – fea_core.cylinder:
*TRANSFORM would switch every *BOUNDARY on the node to local axes). Nodes on the axis (on both faces): (I − R)·u = 0, i.e. no
displacement perpendicular to the axis. The left nodes are left out of the *BOUNDARY sets (a dependent DOF cannot be
held twice); the constraint of the matching right node holds them through the equation – it must be the same constraint
(checked). Reactions are reported PER SECTOR (the reaction of a cut node pair = the node of the right face); the forces
transmitted across the cut faces go to the equilibrium check (`post_step`). Nodal stresses at the cut faces are the
element-count-weighted mean of both sides (= the nodal average of the full 360° model on the same mesh).

Natural frequencies: ccx *CYCLIC SYMMETRY MODEL + *TIE, CYCLIC SYMMETRY + *SELECT CYCLIC SYMMETRY MODES (node136, node175, node258).
Verified by trial 11. 11. 2026 (scratch/feat_cyclic/trial1–4.py, ring sector N = 8 against the full 360° model on the
rotated mesh: all frequencies to 7 digits):
  - ccx reports every eigenvalue TWICE (the complex problem doubled to a real one): for 0 < ND < N/2 the pair is the
    degenerate pair of standing waves (two modes of the full model), for ND = 0 and ND = N/2 the duplicate is spurious
    (one mode of the full model) → modes are read in pairs.
  - .dat table "NODAL DIAMETER, MODE NO, EIGENVALUE, …" (mode numbers restart for every ND); the .frd has one block per
    pair (DISP real part, DISPI imaginary part), the mode number of the block = running index of the FIRST of the pair.
  - the slave (dependent) surface = left, master = right, axis from a to b along the direction for which right = R(+θ)·left;
    expansion to sector k: u_k = R^k·(cos(kφ)·Re − sin(kφ)·Im), φ = 2π·ND/N (= ccx NGRAPH output to 1e-6).
  - nodes of the dependent surface must not be in *BOUNDARY (ccx "*ERROR in cascade … dependent side of a MPC and a SPC").
Results are expanded to the full 360° for display (`expand`) without changing the sector results.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np


class CyclicError(ValueError):
    pass


@dataclass
class CyclicSymmetry:
    """Cyclic symmetry of the study (Study.cyclic; written to the file only when set)."""
    sectors: int = 8                                   # N = number of sectors in 360°
    left: str = ""                                     # selection of the first cut face (dependent side)
    right: str = ""                                    # selection of the second cut face (right = left rotated by +360°/N)
    axis_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_dir: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    enabled: bool = True
    nodal_diameters: list[int] | None = None           # natural frequencies: nodal diameters (None = 0 … N/2)
    expand: bool = True                                # results shown as the full 360° model

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict | None) -> "CyclicSymmetry | None":
        if not isinstance(d, dict):
            return None
        from ..study import _known
        c = _known(cls, d)
        c.sectors = int(c.sectors)
        c.axis_point = [float(v) for v in c.axis_point]
        c.axis_dir = [float(v) for v in c.axis_dir]
        if c.nodal_diameters is not None:
            c.nodal_diameters = [int(v) for v in c.nodal_diameters]
        return c


def active(study) -> CyclicSymmetry | None:
    c = getattr(study, "cyclic", None)
    return c if c is not None and c.enabled else None


def rotation(d, angle: float) -> np.ndarray:
    """Rotation matrix about the unit axis d by angle [rad] (right-hand rule)."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    K = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def max_nodal_diameter(n: int) -> int:
    return n // 2


def nodal_diameters(spec: CyclicSymmetry) -> list[int]:
    top = max_nodal_diameter(spec.sectors)
    if spec.nodal_diameters is None:
        return list(range(top + 1))
    return sorted({int(k) for k in spec.nodal_diameters if 0 <= int(k) <= top})


# ------------------------------------------------------------------ axis from the cut faces

def axis_from_planes(p1, n1, c1, p2, n2, c2) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Axis of a sector from its two planar cut faces (point, normal, centroid of each): the planes intersect in the axis.
    Returns (point on the axis, direction so that face 2 = face 1 rotated by +angle, N = round(360°/angle), angle [°]).
    """
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    n1, n2 = n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2)
    d = np.cross(n1, n2)
    if np.linalg.norm(d) < 1e-6:
        raise CyclicError("The cut faces are parallel (180° sector or not a sector) – enter the axis by hand")
    d /= np.linalg.norm(d)
    A = np.array([n1, n2, d])
    b = np.array([n1 @ np.asarray(p1, float), n2 @ np.asarray(p2, float), 0.0])
    x0 = np.linalg.solve(A, b)
    v1 = np.asarray(c1, float) - x0
    v2 = np.asarray(c2, float) - x0
    v1 -= (v1 @ d) * d
    v2 -= (v2 @ d) * d
    ang = float(np.arctan2(np.cross(v1, v2) @ d, v1 @ v2))
    if ang < 0:
        d, ang = -d, -ang
    if ang < 1e-6:
        raise CyclicError("The cut faces coincide – select the two different cut faces of the sector")
    n = int(round(2 * np.pi / ang))
    k = int(np.argmax(np.abs(d)))
    x0 = x0 - (x0 @ d) * d if abs(d[k]) > 1 - 1e-12 else x0
    return x0, d, n, float(np.degrees(ang))


def detect(study, left: str, right: str) -> dict:
    """Axis and number of sectors from two planar selections (FaceSelectors of the CAD faces): {"axis_point", "axis_dir",
    "sectors", "angle_deg", "exact"} – exact = 360°/angle is an integer within 1e-4."""
    def plane(name):
        sel = study.selections.get(name)
        if sel is None or not sel.faces:
            raise CyclicError(f"Selection '{name}' does not exist or has no faces")
        f0 = sel.faces[0]
        if any(f.type != "plane" for f in sel.faces):
            raise CyclicError(f"Selection '{name}' is not planar – a cut face of a sector must be a plane")
        n = np.asarray(f0.normal, float)
        for f in sel.faces[1:]:
            if abs(abs(np.asarray(f.normal, float) @ n) - 1) > 1e-6:
                raise CyclicError(f"Selection '{name}' has faces in different planes")
        area = np.array([f.area for f in sel.faces])
        c = (np.asarray([f.centroid for f in sel.faces]) * area[:, None]).sum(axis=0) / area.sum()
        return np.asarray(f0.point, float), n, c
    p1, n1, c1 = plane(left)
    p2, n2, c2 = plane(right)
    x0, d, n, ang = axis_from_planes(p1, n1, c1, p2, n2, c2)
    return {"axis_point": x0.tolist(), "axis_dir": d.tolist(), "sectors": n, "angle_deg": ang,
            "exact": abs(360.0 / ang - n) < 1e-4 * n}


# ------------------------------------------------------------------ readiness check (without a mesh)

def check(study, part_names) -> tuple[list[str], list[str]]:
    spec = active(study)
    if spec is None:
        return [], []
    errors, warnings = [], []
    if int(spec.sectors) < 2:
        errors.append("Cyclic symmetry: the number of sectors must be at least 2")
    for name in (spec.left, spec.right):
        if name not in study.selections:
            errors.append(f"Cyclic symmetry: selection '{name}' does not exist")
        elif not study.selections[name].faces:
            errors.append(f"Cyclic symmetry: selection '{name}' has no faces")
    if spec.left == spec.right:
        errors.append("Cyclic symmetry: the two cut faces must be different selections")
    d = np.asarray(spec.axis_dir, float)
    if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-12:
        errors.append("Cyclic symmetry: the axis direction is zero")
        return errors, warnings
    d = d / np.linalg.norm(d)
    if study.dynamics.enabled:
        errors.append("Cyclic symmetry is not available with dynamics (modal superposition of the full model)")
    if study.is_nonlinear():
        errors.append("Cyclic symmetry is not available in a nonlinear analysis (contact, NLGEOM, plasticity, steps)")
    if study.twod.enabled or study.shell.enabled:
        errors.append("Cyclic symmetry is for 3D solid models only (not 2D or shells)")
    if study.frame.active:
        errors.append("Cyclic symmetry is not available with beams")
    if study.mesh.separate:
        errors.append("Cyclic symmetry: separately meshed parts (*TIE) are not supported")
    if study.thermal.cases:
        warnings.append("Cyclic symmetry: heat conduction is computed on the sector with insulated cut faces – exact only if the temperature field is also mirror-symmetric about the cut planes")
    if study.buckling.enabled:
        warnings.append("Cyclic symmetry: linear buckling finds only modes that are the same on every sector (nodal diameter 0)")
    if study.modal.enabled and (study.modal.f_min or study.modal.f_max):
        warnings.append("Cyclic symmetry: the frequency range is ignored – the lowest frequencies of every nodal diameter are computed")
    from ..study import Gravity, Rotation, RemoteLoad, BearingLoad
    x0 = np.asarray(spec.axis_point, float)
    for lc in study.load_cases:
        for l in lc.loads:
            if isinstance(l, Gravity):
                g = np.asarray(l.vector, float)
                if np.linalg.norm(np.cross(g, d)) > 1e-9 * max(np.linalg.norm(g), 1e-30):
                    warnings.append(f"Cyclic symmetry, case '{lc.name}': gravity is not parallel to the axis – it is not the same on every sector (the result is the response to a cyclic load, not to the gravity of the whole part)")
            elif isinstance(l, Rotation):
                a = np.asarray(l.axis, float)
                a = a / max(np.linalg.norm(a), 1e-30)
                off = np.asarray(l.point, float) - x0
                off -= (off @ d) * d
                if np.linalg.norm(np.cross(a, d)) > 1e-6 or np.linalg.norm(off) > 1e-6 * max(1.0, np.linalg.norm(x0)):
                    warnings.append(f"Cyclic symmetry, case '{lc.name}': the rotation axis is not the symmetry axis – the load is not the same on every sector")
            elif isinstance(l, (RemoteLoad, BearingLoad)):
                warnings.append(f"Cyclic symmetry, case '{lc.name}': load '{l.selection}' acts on every sector (the same load in all {spec.sectors} sectors)")
    return errors, warnings


# ------------------------------------------------------------------ node pairs on the mesh

def _crc(a) -> tuple | None:
    if a is None:
        return None
    a = np.ascontiguousarray(a)
    return (a.shape, a.dtype.str, zlib.crc32(a))


def sector_rotation(spec: CyclicSymmetry) -> np.ndarray:
    return rotation(spec.axis_dir, 2 * np.pi / int(spec.sectors))


def pairs(study, mesh) -> dict:
    """Matched nodes of the cut faces (cached on the mesh): {"left" (k,), "right" (k,) – right[i] = R·left[i], "axis" (a,)
    nodes on the axis (on both faces), "R" (3, 3), "d" axis direction, "x0", "n" sectors, "tol"}. Error if the faces do not match."""
    spec = active(study)
    if spec is None:
        raise CyclicError("Cyclic symmetry is not set")
    for name in (spec.left, spec.right):
        if name not in mesh.surfaces or not len(mesh.surfaces[name]):
            raise CyclicError(f"Cyclic symmetry: selection '{name}' has no faces in the mesh")
    key = (repr(spec.to_dict()), _crc(mesh.nodes), _crc(mesh.surfaces[spec.left]), _crc(mesh.surfaces[spec.right]))
    store = getattr(mesh, "_cache", None)
    if isinstance(store, dict):
        hit = store.get("cyclic_pairs")
        if hit is not None and hit[0] == key:
            return hit[1]
    val = _pairs(spec, mesh)
    if isinstance(store, dict):
        store["cyclic_pairs"] = (key, val)
    return val


def _pairs(spec: CyclicSymmetry, mesh) -> dict:
    from scipy.spatial import cKDTree
    d = np.asarray(spec.axis_dir, float)
    d = d / np.linalg.norm(d)
    x0 = np.asarray(spec.axis_point, float)
    R = sector_rotation(spec)
    X = mesh.nodes
    size = float(np.linalg.norm(X.max(axis=0) - X.min(axis=0))) if len(X) else 1.0
    tol = max(1e-6 * size, 1e-9)
    L = mesh.surface_nodes(spec.left)
    Rn = mesh.surface_nodes(spec.right)

    def radius(idx):
        v = X[idx] - x0
        v = v - np.outer(v @ d, d)
        return np.linalg.norm(v, axis=1)
    axis = np.intersect1d(L, Rn)
    if len(axis):
        far = axis[radius(axis) > tol]
        if len(far):
            raise CyclicError(f"Cyclic symmetry: the cut faces share {len(far)} nodes away from the axis – the faces '{spec.left}' and '{spec.right}' are not the two sides of one sector")
    Lo = np.setdiff1d(L, axis)
    Ro = np.setdiff1d(Rn, axis)
    if len(Lo) != len(Ro):
        raise CyclicError(f"Cyclic symmetry: the cut faces have a different number of nodes ({len(Lo)} and {len(Ro)}) – the mesh is not periodic (are the faces really 360°/{spec.sectors} apart?)")
    if len(Lo) == 0:
        raise CyclicError("Cyclic symmetry: the cut faces have no nodes")
    Y = (X[Lo] - x0) @ R.T + x0
    dist, j = cKDTree(X[Ro]).query(Y)
    if dist.max() > tol or len(np.unique(j)) != len(j):
        # the other orientation (right = R⁻¹·left) – then the axis direction is reversed
        Y2 = (X[Lo] - x0) @ R + x0
        dist2, j2 = cKDTree(X[Ro]).query(Y2)
        if dist2.max() <= tol and len(np.unique(j2)) == len(j2):
            raise CyclicError(f"Cyclic symmetry: '{spec.right}' is '{spec.left}' rotated in the negative sense – reverse the axis direction (or swap the faces)")
        raise CyclicError(f"Cyclic symmetry: the nodes of '{spec.left}' rotated by 360°/{spec.sectors} do not match the nodes of '{spec.right}' (deviation up to {dist.max():.3g} mm) – check the axis, the number of sectors and the cut faces")
    return {"left": Lo, "right": Ro[j], "axis": axis, "R": R, "d": d, "x0": x0, "n": int(spec.sectors), "tol": tol}


def rigid_rows(study, center) -> np.ndarray:
    """Rigid-body check (mcp.tools_mesh.rigid_body_check): the cyclic equations hold the translations perpendicular to the axis
    and the rotations about axes perpendicular to it (a constant vector ⟂ axis is not R-periodic) → 4 rows [e, (p − c) × e] / [0, e];
    only the axial translation and the rotation about the symmetry axis stay for the supports."""
    spec = active(study)
    if spec is None:
        return np.zeros((0, 6))
    d = np.asarray(spec.axis_dir, float)
    d = d / np.linalg.norm(d)
    x0 = np.asarray(spec.axis_point, float)
    e1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    c = np.asarray(center, float)
    return np.array([np.r_[e, np.cross(x0 - c, e)] for e in (e1, e2)] + [np.r_[np.zeros(3), e] for e in (e1, e2)])


def dependent_mask(study, mesh) -> np.ndarray | None:
    """(N, 3) bool: DOF dependent in the cyclic equations (the left nodes) – other constraint equations must not use them."""
    if active(study) is None:
        return None
    p = pairs(study, mesh)
    B = np.zeros((len(mesh.nodes), 3), bool)
    B[p["left"]] = True
    return B


# ------------------------------------------------------------------ constraints on the cut faces

def _node_constraints(study, mesh) -> tuple[dict[int, list[np.ndarray]], np.ndarray, np.ndarray]:
    """Constrained directions per node (Fixed / Displacement global axes, cylindrical support directions) and the
    (N, 3) mask + values of the *BOUNDARY DOF."""
    from ..study import Fixed, Displacement, CylindricalSupport
    from ..cylinder import support_directions
    n = len(mesh.nodes)
    B = np.zeros((n, 3), bool)
    V = np.zeros((n, 3))
    dirs: dict[int, list[np.ndarray]] = {}
    for c in study.constraints:
        if isinstance(c, (Fixed, Displacement)) and c.selection in mesh.surfaces:
            idx = mesh.surface_nodes(c.selection)
            vals = (0.0, 0.0, 0.0) if isinstance(c, Fixed) else (c.ux, c.uy, c.uz)
            for k, v in enumerate(vals):
                if v is not None:
                    B[idx, k] = True
                    V[idx, k] = float(v)
    for i, k in zip(*np.nonzero(B)):
        dirs.setdefault(int(i), []).append(np.eye(3)[k])
    if any(isinstance(c, CylindricalSupport) for c in study.constraints):
        for idx, D in support_directions(study, mesh).values():
            for i, node in enumerate(idx):
                dirs.setdefault(int(node), []).extend(D[i])
    return dirs, B, V


def _span_contains(A: list[np.ndarray], Bv: list[np.ndarray]) -> bool:
    """Is span(A) ⊆ span(Bv)?"""
    if not A:
        return True
    if not Bv:
        return False
    Bm = np.array(Bv, float)
    q, r = np.linalg.qr(Bm.T)
    rank = int((np.abs(np.diag(r)) > 1e-9).sum()) if r.size else 0
    q = q[:, :rank]
    for a in A:
        a = np.asarray(a, float)
        if np.linalg.norm(a - q @ (q.T @ a)) > 1e-6 * max(np.linalg.norm(a), 1e-30):
            return False
    return True


def check_mesh(study, mesh) -> list[str]:
    """Checks on the mesh (before writing the solver input): pairs, consistent constraints on both cut faces, no other
    constraint equations on the cut faces. Returns warnings; errors raise CyclicError."""
    spec = active(study)
    if spec is None:
        return []
    p = pairs(study, mesh)
    R = p["R"]
    dirs, B, V = _node_constraints(study, mesh)
    bad = 0
    for l, r in zip(p["left"], p["right"]):
        dl = dirs.get(int(l), [])
        if not dl:
            continue
        dr = [R.T @ v for v in dirs.get(int(r), [])]
        if not _span_contains(dl, dr):
            bad += 1
            continue
        vl, vr = V[l], R.T @ V[r]
        for k in np.nonzero(B[l])[0]:
            if abs(vl[k] - vr[k]) > 1e-9 * max(1.0, abs(vl[k])) and B[r].all():
                bad += 1
                break
    if bad:
        raise CyclicError(f"Cyclic symmetry: {bad} nodes of the cut face '{spec.left}' are supported differently than the matching nodes of '{spec.right}' – a support touching the cut faces must be the same on both sides (rotated by 360°/{spec.sectors})")
    cut = np.concatenate([p["left"], p["right"], p["axis"]])
    from ..study import RemoteSupport, RemoteLoad, ElasticSupport
    others = {c.selection: "remote support" for c in study.constraints if isinstance(c, RemoteSupport)}
    others.update({c.selection: "elastic support" for c in study.constraints if isinstance(c, ElasticSupport)})
    others.update({l.selection: "remote load" for lc in study.load_cases for l in lc.loads if isinstance(l, RemoteLoad)})
    others.update({m.selection: "added mass" for m in getattr(study, "masses", [])})
    for key in mesh.surfaces:
        if key.startswith("__b"):
            others[key] = "bolt"
    for name, what in sorted(others.items()):
        if name in mesh.surfaces and len(np.intersect1d(mesh.surface_nodes(name), cut)):
            raise CyclicError(f"Cyclic symmetry: the {what} '{name}' touches a cut face of the sector – this combination is not supported (keep it away from the cut faces)")
    if study.modal.enabled and len(p["axis"]) and any(int(a) in dirs for a in p["axis"]):
        raise CyclicError("Cyclic symmetry with natural frequencies: nodes on the symmetry axis must not be supported (ccx constrains them by the nodal diameter itself)")
    return []


def boundary_exclusions(study, mesh) -> np.ndarray:
    """Nodes left out of the *BOUNDARY sets (the dependent left nodes; held by the equation from the right node)."""
    return pairs(study, mesh)["left"]


def equations(study, mesh) -> list[tuple[int, list[tuple[int, int, float]]]]:
    """Static cyclic constraint equations [(dependent node, [(node, dof 0..2, coef), ...])], the first term dependent:
    u_left − Rᵀ·u_right = 0 per component; axis nodes: u·e1 = u·e2 = 0 (e ⟂ axis), pivots outside the *BOUNDARY DOF."""
    p = pairs(study, mesh)
    R = p["R"]
    Rt = R.T
    out = []
    for l, r in zip(p["left"].tolist(), p["right"].tolist()):
        for a in range(3):
            terms = [(l, a, 1.0)] + [(r, b, -float(Rt[a, b])) for b in range(3) if abs(Rt[a, b]) > 1e-12]
            out.append((l, terms))
    if len(p["axis"]):
        from ..cylinder import boundary_dofs
        Bc = boundary_dofs(study, mesh)
        d = p["d"]
        e1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        for node in p["axis"].tolist():
            C = np.array([e1, e2])
            free = [k for k in range(3) if not Bc[node, k]]
            r_ = 0
            for col in free:
                if r_ >= len(C):
                    break
                piv = r_ + int(np.argmax(np.abs(C[r_:, col])))
                if abs(C[piv, col]) < 1e-9:
                    continue
                C[[r_, piv]] = C[[piv, r_]]
                C[r_] /= C[r_, col]
                for j in range(len(C)):
                    if j != r_:
                        C[j] -= C[j, col] * C[r_]
                terms = [(node, col, 1.0)] + [(node, k, float(C[r_, k])) for k in range(3)
                                               if k != col and abs(C[r_, k]) > 1e-12]
                out.append((node, terms))
                r_ += 1
    return out


def summary(study, mesh) -> dict:
    """Results.info["cyclic"]: what the expansion and the report need (saved with the results)."""
    spec = active(study)
    p = pairs(study, mesh)
    return {"sectors": int(spec.sectors), "left": spec.left, "right": spec.right,
            "axis_point": p["x0"].tolist(), "axis_dir": p["d"].tolist(), "angle_deg": 360.0 / int(spec.sectors),
            "cut_pairs": int(len(p["left"])), "axis_nodes": int(len(p["axis"])),
            "equations": int(3 * len(p["left"]) + 2 * len(p["axis"])), "expand": bool(spec.expand)}


# ------------------------------------------------------------------ results of a static state

def _voigt_to_tensor(S: np.ndarray) -> np.ndarray:
    """ccx order SXX SYY SZZ SXY SYZ SZX → (n, 3, 3)."""
    T = np.empty((len(S), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = S[:, 0], S[:, 1], S[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = S[:, 3]
    T[:, 1, 2] = T[:, 2, 1] = S[:, 4]
    T[:, 0, 2] = T[:, 2, 0] = S[:, 5]
    return T


def _tensor_to_voigt(T: np.ndarray) -> np.ndarray:
    return np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 1, 2], T[:, 0, 2]], axis=1)


def rotate_stress(S: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Stress (n, 6) rotated by R: R·σ·Rᵀ."""
    T = _voigt_to_tensor(np.asarray(S, float))
    return _tensor_to_voigt(np.einsum("ij,njk,lk->nil", R, T, R))


def _element_counts(mesh) -> np.ndarray:
    return np.bincount(mesh.tets.ravel(), minlength=len(mesh.nodes))


def post_step(study, mesh, RF: np.ndarray, F: np.ndarray, S: np.ndarray, allc: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-sector reactions and the cut forces of a static state.

    RF − F at a node pair (left dependent): left = constraint force λ, right = −R·λ + reaction. The reaction of the full model at the
    node = (RF − F)_right + R·(RF − F)_left → returned RF' with that at the right node and RF' = F at the left node; axis nodes keep
    only the axial component (1/N of the full reaction). The rest = forces across the cut faces (force, moment about the centre of the
    mesh envelope) for the equilibrium check. S' = stresses at the cut faces averaged over the elements of both sides."""
    p = pairs(study, mesh)
    R, d = p["R"], p["d"]
    L, Rn, A = p["left"], p["right"], p["axis"]
    G = RF - F
    G2 = G.copy()
    G2[Rn] = G[Rn] + G[L] @ R.T
    G2[L] = 0.0
    if len(A):
        G2[A] = np.outer(G[A] @ d, d)
    cut = np.concatenate([L, Rn, A])
    lost = G[cut] - np.where(allc[cut], G2[cut], 0.0)
    x = mesh.nodes
    c0 = 0.5 * (x.min(axis=0) + x.max(axis=0))
    force = lost.sum(axis=0)
    moment = np.cross(x[cut] - c0, lost).sum(axis=0)
    free_res = np.where(allc[Rn], 0.0, G2[Rn])
    scale = max(float(np.abs(F).sum()), float(np.abs(G[cut]).sum()), 1e-30)
    RF2 = F + G2
    S2 = S
    if S is not None and len(S):
        cnt = _element_counts(mesh).astype(float)
        nl, nr = cnt[L][:, None], cnt[Rn][:, None]
        SL = (nl * S[L] + nr * rotate_stress(S[Rn], R.T)) / (nl + nr)
        S2 = np.array(S, float, copy=True)
        S2[L] = SL
        S2[Rn] = rotate_stress(SL, R)
        if len(A):
            acc = np.zeros((len(A), 6))
            for k in range(p["n"]):
                acc += rotate_stress(S[A], np.linalg.matrix_power(R, k))
            S2[A] = acc / p["n"]
    info = {"sectors": p["n"], "cut_force": force.tolist(), "cut_moment": moment.tolist(),
            "cut_pairs": int(len(L)), "axis_nodes": int(len(A)),
            "free_residual": float(np.abs(free_res).max() / scale) if len(Rn) else 0.0}
    return RF2, S2, info


# ------------------------------------------------------------------ solver input (ccx)

def write(f, study, mesh, m, procedure: str, fmt) -> None:
    """Cyclic symmetry cards: reduced *BOUNDARY sets (without the dependent nodes) and the static equations (static / buckle)
    or the ccx cyclic symmetry model (modal)."""
    from ..study import Fixed, Displacement
    check_mesh(study, mesh)
    p = pairs(study, mesh)
    excl = p["left"] if procedure != "modal" else np.concatenate([p["left"], p["axis"]])
    m.cyclic_nsets = {}
    for i, c in enumerate(study.constraints):
        if not isinstance(c, (Fixed, Displacement)) or c.selection in m.cyclic_nsets:
            continue
        idx = mesh.surface_nodes(c.selection)
        keep = np.setdiff1d(idx, excl)
        if len(keep) == len(idx):
            continue
        name = f"{m.nsets[c.selection]}Y"
        if len(keep) == 0:
            m.cyclic_nsets[c.selection] = None
            continue
        f.write(f"** cyclic symmetry: support set without the dependent cut nodes\n*NSET, NSET={name}\n")
        _ids(f, keep + 1)
        m.cyclic_nsets[c.selection] = name
    spec = active(study)
    if procedure == "modal":
        d = p["d"]
        a = p["x0"]
        b = a + d * max(1.0, float(np.linalg.norm(mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0))))
        f.write("** cyclic symmetry model (ccx): slave = left cut face, master = right cut face\n*NSET, NSET=NCYCL\n")
        _ids(f, np.concatenate([p["left"], p["axis"]]) + 1)
        f.write("*NSET, NSET=NCYCR\n")
        _ids(f, np.concatenate([p["right"], p["axis"]]) + 1)
        f.write("*SURFACE, NAME=SCYCL, TYPE=NODE\nNCYCL\n*SURFACE, NAME=SCYCR, TYPE=NODE\nNCYCR\n"
                f"*TIE, NAME=TCYC, CYCLIC SYMMETRY, POSITION TOLERANCE={fmt(10 * p['tol'])}\nSCYCL, SCYCR\n"
                f"*CYCLIC SYMMETRY MODEL, N={int(spec.sectors)}, TIE=TCYC\n"
                + ", ".join(fmt(float(v)) for v in (*a, *b)) + "\n")
        return
    eqs = equations(study, mesh)
    f.write(f"** cyclic symmetry: u_left = R^T u_right (N = {int(spec.sectors)}), axis nodes without radial motion\n*EQUATION\n")
    for _, terms in eqs:
        f.write(f"{len(terms)}\n" + ", ".join(f"{n + 1}, {k + 1}, {fmt(c)}" for n, k, c in terms) + "\n")


def _ids(f, ids, per_line: int = 16) -> None:
    ids = np.asarray(ids, dtype=np.int64).ravel().tolist()
    for s in range(0, len(ids), per_line):
        f.write(", ".join(str(v) for v in ids[s:s + per_line]) + ",\n")
