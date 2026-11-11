"""3D printing – homogenised model (F4): shell + infill, orthotropy according to the build direction, Tsai-Wu/Hill."""
from __future__ import annotations

import numpy as np

from .infill import core, solid, line_solid, density_ratio, rotation
from .model import PrintSpec, PrintSettings, PrintVariant, CreepSettings, CREEP_WARNING  # noqa: F401


def has_print(study, part_names) -> bool:
    from .region import print_parts
    return bool(print_parts(study, list(part_names)))


def shell_of(mesh) -> np.ndarray | None:
    """Shell fraction per element (−1 = unprinted part) or None when the mesh does not contain printing."""
    s = mesh.cell_data.get("print_shell")
    return s if s is not None and np.any(s >= 0) else None


def element_density(study, mesh) -> np.ndarray:
    """Density per element [t/mm³]: for printing ρ·(f + (1 − f)·ρ_infill)."""
    rho = np.array([study.material_of(p).rho for p in mesh.part_names])[mesh.tet_part]
    sh = shell_of(mesh)
    if sh is None:
        return rho
    rho = rho.copy()
    for pi, pname in enumerate(mesh.part_names):
        el = np.nonzero((mesh.tet_part == pi) & (sh >= 0))[0]
        if len(el):
            spec = study.printing.spec(pname)
            r = density_ratio(spec.pattern, spec.infill) if spec.enabled else 1.0
            f = sh[el]
            rho[el] *= f + (1 - f) * r
    return rho


def part_materials(study, pname):
    """(shell without line direction, perimeter in the axes of the line, core) as Ortho for the part."""
    from .props import props_of
    mat = study.material_of(pname)
    spec = study.printing.spec(pname)
    pr = props_of(study, mat)
    ms = solid(mat)
    ml = line_solid(mat, pr.line_E_ratio, pr.line_R_ratio) if spec.enabled else ms
    mc = core(mat, spec.pattern, spec.infill) if spec.enabled else ms
    return ms, ml, mc


def line_axes(R: np.ndarray, theta: float) -> np.ndarray:
    """Axes of the perimeter (rows): 1 = line at the angle θ from the print axis 1 in the plane of the layer, 2 = across, 3 = build."""
    a = np.cos(theta) * R[0] + np.sin(theta) * R[1]
    return np.vstack([a, np.cross(R[2], a), R[2]])


def element_orientation(study, mesh, pi: int, el: np.ndarray) -> tuple[np.ndarray, dict]:
    """Class of the line direction of the elements el of the part pi: −1 = no line (core, full layers, full part, ratio 1),
    otherwise the index of the class of the angle in the plane of the layer (180°/line_bins, centre of the class k·Δ – direction error ≤ Δ/2 = 2.5°). Returns (classes, {class: axes R 3×3}),
    class −1 has the print axes of the part.
    """
    from .props import props_of
    pname = mesh.part_names[pi]
    spec = study.printing.spec(pname)
    R = rotation(spec.build_dir)
    kind = np.full(len(el), -1, dtype=np.int64)
    axes = {-1: R}
    pr = props_of(study, study.material_of(pname))
    line = mesh.cell_data.get("print_line")
    sh = shell_of(mesh)
    if (not spec.enabled or line is None or sh is None
            or (pr.line_E_ratio == 1.0 and pr.line_R_ratio == 1.0)):
        return kind, axes
    nb = max(1, int(study.printing.line_bins))
    L = line[el]
    has = (np.linalg.norm(L, axis=1) > 0.5) & (sh[el] > 0)
    th = np.mod(np.arctan2(L @ R[1], L @ R[0]), np.pi)
    b = np.mod(np.rint(th / (np.pi / nb)).astype(np.int64), nb)     # classes with the centres k·Δ (the axes of the part exactly)
    kind[has] = b[has]
    for k in np.unique(kind[has]):
        axes[int(k)] = line_axes(R, k * np.pi / nb)
    return kind, axes


def material_groups(study, mesh) -> list[dict]:
    """Groups of elements of printed parts with the same material and orientation: part, shell fraction f, class of the
    line direction, elements, C (in the axes R), R (material axes in global – rows), density.
    C = f·C_shell + (1 − f)·C_core (Voigt, iso-strain). For perimeters with line anisotropy the shell is orthotropic in the axes
    (line, across, build); the core is defined in the print axes – an infill with in-plane anisotropy
    (infill.INPLANE_SHEAR, lines ±45°) is transformed into the line axes (Tᵀ·C·T), a transversely isotropic one stays unchanged.
    """
    sh = shell_of(mesh)
    if sh is None:
        return []
    out = []
    for pi, pname in enumerate(mesh.part_names):
        mask = (mesh.tet_part == pi) & (sh >= 0)
        if not mask.any():
            continue
        mat = study.material_of(pname)
        spec = study.printing.spec(pname)
        ms, ml, mc = part_materials(study, pname)
        Cs, Cl, Cc = ms.stiffness(), ml.stiffness(), mc.stiffness()
        r = density_ratio(spec.pattern, spec.infill) if spec.enabled else 1.0
        el_all = np.nonzero(mask)[0]
        kind, axes = element_orientation(study, mesh, pi, el_all)
        for k in sorted(axes):
            Ck = Cc
            if k >= 0 and mc.G12v is not None:          # core from the axes of the print (axes[-1]) to the axes of the line axes[k]
                from .infill import strain_matrix
                T = strain_matrix(axes[-1] @ axes[k].T)
                Ck = T.T @ Cc @ T
            for f in np.unique(sh[el_all[kind == k]]):
                el = el_all[(kind == k) & (sh[el_all] == f)]
                if len(el) == 0:
                    continue
                out.append({"part": pname, "part_index": pi, "f": float(f), "line_bin": int(k), "elems": el,
                            "C": f * (Cs if k < 0 else Cl) + (1 - f) * Ck, "R": axes[k],
                            "rho": mat.rho * (f + (1 - f) * r)})
    return out


def element_stiffness(study, mesh) -> tuple[np.ndarray, np.ndarray] | None:
    """C (E_t, 6, 6) in GLOBAL axes (Voigt xx yy zz xy xz yz, engineering shear) for the printed elements
    and their indices (for solvers without *ORIENTATION – NGSolve). C_g = Tᵀ·C·T, T = transformation of the strain.
    """
    groups = material_groups(study, mesh)
    if not groups:
        return None
    from .infill import strain_matrix
    els, Cs = [], []
    for g in groups:
        T = strain_matrix(g["R"])
        Cs.append(np.broadcast_to(T.T @ g["C"] @ T, (len(g["elems"]), 6, 6)))
        els.append(g["elems"])
    return np.concatenate(els), np.concatenate(Cs)
