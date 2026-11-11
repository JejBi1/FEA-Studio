"""Failure of printed parts: Tsai-Wu / Tsai-Hill in the print axes (F4).

The strain is computed from the displacements at the 4 Gauss points of tet10 (recomputation from the result, not the own stiffness
matrix). The shell and the core have the same strain in the element (Voigt, iso-strain – the same assumption as the mixing of stiffness),
the stress σ = C·ε of each component is assessed with its strengths; in the element the worse component decides.
Result = safety λ (multiple of the load to failure according to the criterion) and the index 1/λ.
Nodal value = the worst of the printed-part elements around the node (no averaging across the jumps of material).
"""
from __future__ import annotations

import numpy as np

from ..loads import GAUSS4, _tet10_dN, inv3
from .infill import Ortho, strain_to_local

CRITERIA = {"tsai_wu": "Tsai-Wu", "tsai_hill": "Tsai-Hill"}


def element_strains(nodes: np.ndarray, conn: np.ndarray, U):
    """Strain (E, G, 6) at the Gauss points, Voigt xx yy zz xy xz yz (engineering shear).

    U = displacements (N, 3), or a list of the displacements of several states → a list of fields (∂N/∂x is computed only once).
    """
    many = isinstance(U, (list, tuple))
    Us = list(U) if many else [U]
    X = nodes[conn]                       # (E, 10, 3)
    outs = [np.empty((len(conn), len(GAUSS4), 6)) for _ in Us]
    for g, p in enumerate(GAUSS4):
        dN = _tet10_dN(*p)                # (3, 10)
        J = np.tensordot(dN, X, axes=(1, 1)).swapaxes(0, 1)
        dNx = inv3(J) @ dN                # ∂N/∂x (E, 3, 10)
        for out, u in zip(outs, Us):
            G = dNx @ u[conn]             # G[a, k] = ∂u_k/∂x_a
            out[:, g, 0] = G[:, 0, 0]
            out[:, g, 1] = G[:, 1, 1]
            out[:, g, 2] = G[:, 2, 2]
            out[:, g, 3] = G[:, 0, 1] + G[:, 1, 0]
            out[:, g, 4] = G[:, 0, 2] + G[:, 2, 0]
            out[:, g, 5] = G[:, 1, 2] + G[:, 2, 1]
    return outs if many else outs[0]


def subtract_thermal(eps: np.ndarray, conn: np.ndarray, T: np.ndarray | None, t_ref: float,
                     alpha: float | None) -> np.ndarray:
    """Mechanical strain ε − α·(T − T_ref)·(1,1,1,0,0,0) at the Gauss points (T per node, interpolation by the shape functions
    of tet10 as in ccx). The free thermal expansion of the print must not be a stress (σ = C·(ε − ε_th), as recovery).
    The expansion is isotropic (α of the material) – the same in the shell and in the infill, so it is subtracted in the global axes.
    """
    if T is None or not alpha:
        return eps
    from ..results import tet10_shape
    Ng = tet10_shape(GAUSS4)                                    # (4, 10)
    eth = alpha * ((T[conn] - t_ref) @ Ng.T)                    # (E, 4)
    out = eps.copy()
    out[..., :3] -= eth[..., None]
    return out


def safety_tsai_wu(s: np.ndarray, m: Ortho) -> np.ndarray:
    """Safety λ: Tsai-Wu(λσ) = 1. s (…, 6) in the axes of the material: 11 22 33 12 13 23.
    Interaction terms F_ij = −½√(F_ii·F_jj) (for a transversely isotropic F12 = −½F11).
    """
    s1, s2, s3, t12, t13, t23 = (s[..., i] for i in range(6))
    Yt, Yc = m.Ytv, m.Ycv
    F1 = 1 / m.Xt - 1 / m.Xc
    F2 = 1 / Yt - 1 / Yc
    F3 = 1 / m.Zt - 1 / m.Zc
    F11 = 1 / (m.Xt * m.Xc)
    F22 = 1 / (Yt * Yc)
    F33 = 1 / (m.Zt * m.Zc)
    F12 = -0.5 * np.sqrt(F11 * F22)
    F13 = -0.5 * np.sqrt(F11 * F33)
    F23 = -0.5 * np.sqrt(F22 * F33)
    a = (F11 * s1 ** 2 + F22 * s2 ** 2 + F33 * s3 ** 2 + 2 * F12 * s1 * s2 + 2 * F13 * s1 * s3 + 2 * F23 * s2 * s3
         + (t13 ** 2 + t23 ** 2) / m.Sz ** 2 + t12 ** 2 / m.Sxy ** 2)
    b = F1 * s1 + F2 * s2 + F3 * s3
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(a > 1e-300, (-b + np.sqrt(b * b + 4 * a)) / (2 * a),
                       np.where(b > 0, 1 / b, np.inf))
    return lam


def safety_tsai_hill(s: np.ndarray, m: Ortho) -> np.ndarray:
    """Safety λ = 1/√FI, Hill criterion with the tensile / compressive strength by the sign of the component."""
    s1, s2, s3, t12, t13, t23 = (s[..., i] for i in range(6))
    X1 = np.where(s1 >= 0, m.Xt, m.Xc)
    X2 = np.where(s2 >= 0, m.Ytv, m.Ycv)
    X3 = np.where(s3 >= 0, m.Zt, m.Zc)
    i1, i2, i3 = 1 / X1 ** 2, 1 / X2 ** 2, 1 / X3 ** 2
    F, G, H = 0.5 * (i2 + i3 - i1), 0.5 * (i1 + i3 - i2), 0.5 * (i1 + i2 - i3)
    fi = (F * (s2 - s3) ** 2 + G * (s3 - s1) ** 2 + H * (s1 - s2) ** 2
          + (t23 ** 2 + t13 ** 2) / m.Sz ** 2 + t12 ** 2 / m.Sxy ** 2)
    with np.errstate(divide="ignore"):
        return np.where(fi > 1e-300, 1 / np.sqrt(np.maximum(fi, 1e-300)), np.inf)


def safety(s: np.ndarray, m: Ortho, criterion: str) -> np.ndarray:
    return safety_tsai_hill(s, m) if criterion == "tsai_hill" else safety_tsai_wu(s, m)


def evaluate(study, mesh, U: np.ndarray, T: np.ndarray | None = None,
             eps_ip: dict[int, np.ndarray] | None = None) -> dict | None:
    """Safety of the printed parts for one state. Returns the nodal fields and the summary per part (or None).
    T = nodal temperatures of the state with a thermal load (StepResult.fields["T"]) → the mechanical strain is assessed.
    """
    from . import shell_of
    sh = shell_of(mesh)
    if sh is None:
        return None
    crit = study.printing.criterion
    n = len(mesh.nodes)
    node_lam = np.full(n, np.inf)
    parts = {}
    for pi, pname in enumerate(mesh.part_names):
        el = np.nonzero((mesh.tet_part == pi) & (sh >= 0))[0]
        if len(el) == 0:
            continue
        from . import part_materials, element_orientation
        ms, ml, mc = part_materials(study, pname)
        kind, axes = element_orientation(study, mesh, pi, el)
        if eps_ip is not None and all((e + 1) in eps_ip for e in el):
            eps_g = np.stack([eps_ip[int(e) + 1] for e in el])
        else:
            eps_g = element_strains(mesh.nodes, mesh.tets[el], U)                      # (E, 4, 6) globally
            eps_g = subtract_thermal(eps_g, mesh.tets[el], T, study.thermal.reference_temperature,
                                     study.material_of(pname).alpha)
        eps = strain_to_local(eps_g, axes[-1])                                     # print axes of the part
        f = sh[el]
        sig_c = eps @ mc.stiffness().T
        sig_s = eps @ ms.stiffness().T
        lam_s = np.where(f[:, None] > 0, safety(sig_s, ms, crit), np.inf)
        for k in axes:                       # perimeters with line anisotropy: axes (line, across, build)
            if k < 0:
                continue
            sel = kind == k
            sig_s[sel] = strain_to_local(eps_g[sel], axes[k]) @ ml.stiffness().T
            lam_s[sel] = safety(sig_s[sel], ml, crit)
        lam_s = np.where(f[:, None] > 0, lam_s, np.inf).min(axis=1)
        lam_c = np.where(f[:, None] < 1, safety(sig_c, mc, crit), np.inf).min(axis=1)
        lam = np.minimum(lam_s, lam_c)
        np.minimum.at(node_lam, mesh.tets[el].ravel(), np.repeat(lam, mesh.tets.shape[1]))
        k = int(np.argmin(lam))
        in_core = bool(lam_c[k] < lam_s[k])
        line = not in_core and kind[k] >= 0
        sig_k, m_k = (sig_c[k], mc) if in_core else (sig_s[k], ml if line else ms)
        sig = sig_k[int(np.argmin(safety(sig_k, m_k, crit)))]
        thin = mesh.cell_data.get("print_thin")
        parts[pname] = {
            "min_safety": float(lam[k]),
            "at": mesh.nodes[mesh.tets[el[k], :4]].mean(axis=0).tolist(),
            "region": "core (infill)" if in_core else ("perimeter (line axes, transverse, Z)" if line else "shell"),
            "stress_local": sig.tolist(),          # σ11 σ22 σ33 τ12 τ13 τ23 in the print / line axes [MPa]
            "thin": bool(thin is not None and thin[el[k]] > 0),   # thin place with full infill (region.thin_core)
            "element": int(el[k]),
        }
    printed = np.isfinite(node_lam) | np.isin(np.arange(n), np.unique(mesh.tets[sh >= 0]))
    node_lam = np.where(printed, node_lam, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.where(printed, 1.0 / node_lam, np.nan)
    return {"safety": node_lam, "index": index, "parts": parts, "criterion": CRITERIA.get(crit, crit)}


def part_summary(study, mesh) -> dict:
    """Volume fraction of the shell and mass of the printed parts (for the report)."""
    from ..loads import element_volumes
    from . import shell_of, element_density
    sh = shell_of(mesh)
    if sh is None:
        return {}
    vol = element_volumes(mesh)
    rho = element_density(study, mesh)
    out = {}
    for pi, pname in enumerate(mesh.part_names):
        el = (mesh.tet_part == pi) & (sh >= 0)
        if el.any():
            spec = study.printing.spec(pname)
            V = float(vol[el].sum())
            thin = mesh.cell_data.get("print_thin")
            out[pname] = {"volume_mm3": V, "shell_fraction": float((vol[el] * sh[el]).sum() / V),
                          "thin_fraction": float((vol[el] * (thin[el] > 0)).sum() / V) if thin is not None else 0.0,
                          "mass_g": float((vol[el] * rho[el]).sum() * 1e6), "material": study.material_of(pname).name,
                          "spec": spec.to_dict()}
    return out
