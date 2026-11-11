"""Stress assessment and safety factor (Stress Tool).

Criteria:
- "mises": von Mises equivalent stress, limit Re (metals) or Rm (plastics, cast iron)
- "tresca": maximum shear stress (sigma_1 - sigma_3), limit Re or Rm
- "mohr_coulomb": tension != compression (Rankine / Coulomb-Mohr for brittle materials):
    if sigma_1 > 0 and sigma_3 < 0: 1/s = sigma_1/Rm - sigma_3/Rc
    if sigma_3 >= 0: 1/s = sigma_1/Rm
    if sigma_1 <= 0: 1/s = -sigma_3/Rc
    missing Rc -> warning and Rc = Rm.

Default choice by material:
- metals with Re: mises, limit Re
- grey cast iron (name contains 'litina' or Re is None and Rc is not None): mohr_coulomb, limit Rm (with Rc)
- plastics / unprinted plastics (Re is None, Rc is None): mises, limit Rm
- 3D printing (E_z is not None): this criterion is not used, `print_safety` from the result (Tsai-Wu/Hill) is taken
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .study import Study, Material
from .results import StepResult


def part_nodes(mesh: Mesh, pi: int) -> np.ndarray:
    """Sorted nodes of the elements of part `pi` (= np.unique(mesh.tets[mesh.tet_part == pi])), all parts computed once and
    cached on the mesh (recomputed when tets / tet_part change). The array must not be changed (shared cache)."""
    def compute():
        tp = np.asarray(mesh.tet_part)
        return [np.unique(mesh.tets[tp == k]) for k in range(len(mesh.part_names))]
    cached = getattr(mesh, "cached", None)
    if cached is None:
        return np.unique(mesh.tets[mesh.tet_part == pi])
    allp = cached("assessment_part_nodes", ("tets", "tet_part"), compute)
    if not 0 <= pi < len(allp):
        return np.unique(mesh.tets[mesh.tet_part == pi])
    return allp[pi]


CRITERIA = {
    "mises": "von Mises",
    "tresca": "Tresca (max. shear)",
    "mohr_coulomb": "Mohr–Coulomb (tension/compression)",
}

LIMITS = {
    "Re": "Yield strength Re",
    "Rm": "Tensile strength Rm",
}


def default_assessment_for_material(mat: Material) -> dict:
    """Default assessment for the given material."""
    name_lower = mat.name.lower()
    if "litina" in name_lower or (mat.Re is None and mat.Rc is not None):
        return {"criterion": "mohr_coulomb", "limit": "Rm"}
    if mat.Re is None:
        return {"criterion": "mises", "limit": "Rm"}
    return {"criterion": "mises", "limit": "Re"}


def resolve_assessment(study: Study, part_name: str | None = None,
                       criterion: str | None = None, limit: str | None = None) -> tuple[str, str]:
    """Determines the valid criterion and limit for a part or globally."""
    st_assess = getattr(study, "assessment", {}) or {}
    crit = criterion or st_assess.get("criterion")
    lim = limit or st_assess.get("limit")

    if part_name is not None and (crit is None or lim is None):
        mat = study.material_of(part_name)
        mat_assess = st_assess.get("materials", {}).get(mat.name, {})
        def_assess = default_assessment_for_material(mat)
        crit = crit or mat_assess.get("criterion") or def_assess["criterion"]
        lim = lim or mat_assess.get("limit") or def_assess["limit"]

    crit = crit or "mises"
    lim = lim or "Re"
    return crit, lim


def equivalent_stress(S: np.ndarray, criterion: str = "mises",
                      Rc: float | None = None, Rm: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Computes the equivalent stress and the inverse of the stress / load factor.
    S: (N, 6) in ccx order xx, yy, zz, xy, yz, zx.
    Returns (sigma_eq, inv_safety_factor_unscaled).
    """
    from .results import von_mises, principal
    if criterion == "mises":
        vm = von_mises(S)
        return vm, vm
    elif criterion == "tresca":
        p = principal(S)
        tr = p[:, 0] - p[:, 2] # sigma_1 - sigma_3
        return tr, tr
    elif criterion == "mohr_coulomb":
        p = principal(S)
        s1, s3 = p[:, 0], p[:, 2]
        rm = float(Rm or 250.0)
        rc = float(Rc or rm)
        inv_s = np.zeros(len(S))

        # 1) tension and compression: s1 > 0 > s3
        mask_mixed = (s1 > 0) & (s3 < 0)
        inv_s[mask_mixed] = s1[mask_mixed] / rm - s3[mask_mixed] / rc

        # 2) pure tension: s3 >= 0
        mask_tension = s3 >= 0
        inv_s[mask_tension] = s1[mask_tension] / rm

        # 3) pure compression: s1 <= 0
        mask_compression = s1 <= 0
        inv_s[mask_compression] = -s3[mask_compression] / rc

        # Equivalent stress converted to the tensile strength: sigma_eq = inv_s * rm
        sigma_eq = np.maximum(inv_s * rm, 0.0)
        return sigma_eq, inv_s
    else:
        raise ValueError(f"Unknown assessment criterion: {criterion}")


def _part_limit(study: Study, pname: str, criterion: str | None, limit: str | None
                ) -> tuple[str, str, float, float | None, list[str]]:
    """(criterion, limit type, limit used [MPa], Rc, warning) of a part – also with a substitute value when the given limit is missing."""
    mat = study.material_of(pname)
    crit, lim = resolve_assessment(study, pname, criterion, limit)
    warnings: list[str] = []
    limit_val = getattr(mat, lim, None)
    if limit_val is None:
        if lim == "Re" and mat.Rm is not None:
            limit_val = mat.Rm
            warnings.append(f"Part '{pname}': yield strength Re is missing, tensile strength Rm = {mat.Rm} MPa used for the assessment")
        elif lim == "Rm" and mat.Re is not None:
            limit_val = mat.Re
            warnings.append(f"Part '{pname}': tensile strength Rm is missing, yield strength Re = {mat.Re} MPa used for the assessment")
        else:
            limit_val = 235.0
            warnings.append(f"Part '{pname}': material '{mat.name}' has no {lim} specified, substitute value 235 MPa used")
    rc_val = getattr(mat, "Rc", None)
    if crit == "mohr_coulomb" and rc_val is None:
        rc_val = limit_val
        warnings.append(f"Part '{pname}': material '{mat.name}' has no compressive strength Rc specified, Rc = Rm = {rc_val} MPa used")
    return crit, lim, float(limit_val), rc_val, warnings


def compute_safety_field(study: Study, mesh: Mesh, step: StepResult,
                         criterion: str | None = None, limit: str | None = None,
                         max_safety: float = 10.0) -> tuple[np.ndarray, list[str]]:
    """Computes the safety field (N,) at the mesh nodes.
    If criterion / limit is not given, it is evaluated per part according to the part materials.
    Parts with printed material (3D printing) are not assessed by this criterion: their `print_safety`
    from the result (Tsai-Wu / Hill) is used; if the result has none, max_safety stays and a warning is added.
    Returns (safety_array, warnings).
    """
    n_nodes = len(mesh.nodes)
    safety = np.full(n_nodes, max_safety, dtype=float)
    warnings: list[str] = []
    print_safety = step.fields.get("print_safety") if getattr(step, "fields", None) else None

    for pi, pname in enumerate(mesh.part_names):
        mat = study.material_of(pname)
        nodes_p = part_nodes(mesh, pi)
        if len(nodes_p) == 0:
            continue
        if getattr(mat, "E_z", None) is not None:             # 3D printing: assessment through printing.failure
            if print_safety is not None and len(print_safety) == n_nodes:
                safety[nodes_p] = np.minimum(safety[nodes_p], np.clip(print_safety[nodes_p], 0.0, max_safety))
            else:
                warnings.append(f"Part '{pname}': printed material is assessed with the 3D printing model (print_safety), which is not in the results – this part is not assessed by this tool")
            continue
        crit, lim, limit_val, rc_val, w = _part_limit(study, pname, criterion, limit)
        warnings += w

        Sp = step.S[nodes_p]
        if crit == "mohr_coulomb":
            _, inv_s = equivalent_stress(Sp, criterion="mohr_coulomb", Rc=rc_val, Rm=limit_val)
            s_part = np.where(inv_s > 1e-9, 1.0 / np.maximum(inv_s, 1e-9), max_safety)
        else:
            eq, _ = equivalent_stress(Sp, criterion=crit)
            s_part = limit_val / np.maximum(eq, 1e-9)

        s_part = np.clip(s_part, 0.0, max_safety)
        # Nodes on an interface between parts take the lowest safety of the adjacent parts
        safety[nodes_p] = np.minimum(safety[nodes_p], s_part)

    return safety, warnings


def assessment_summary(study: Study, mesh: Mesh, step: StepResult,
                       criterion: str | None = None, limit: str | None = None) -> dict:
    """Summary of the safety assessment overall and per part (limit_MPa = the limit actually used, including a substitute value)."""
    safety, warnings = compute_safety_field(study, mesh, step, criterion, limit)
    parts_summary = {}

    for pi, pname in enumerate(mesh.part_names):
        mat = study.material_of(pname)
        nodes_p = part_nodes(mesh, pi)
        if len(nodes_p) == 0:
            continue
        s_part = safety[nodes_p]
        min_idx = nodes_p[int(np.argmin(s_part))]
        if getattr(mat, "E_z", None) is not None:
            entry = {"material": mat.name, "criterion": "3D printing (print_safety)", "limit_type": "-", "limit_MPa": None}
        else:
            crit, lim, limit_val, _, _ = _part_limit(study, pname, criterion, limit)
            entry = {"material": mat.name, "criterion": crit, "limit_type": lim, "limit_MPa": limit_val}
        entry["min_safety"] = float(s_part.min())
        entry["min_safety_at"] = mesh.nodes[min_idx].tolist()
        parts_summary[pname] = entry

    overall_min_idx = int(np.argmin(safety))
    return {
        "min_safety": float(safety.min()),
        "min_safety_at": mesh.nodes[overall_min_idx].tolist(),
        "parts": parts_summary,
        "warnings": warnings,
    }
