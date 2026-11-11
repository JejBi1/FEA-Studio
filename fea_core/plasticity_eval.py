"""Evaluation of an elastoplastic result (GUI Results, Word report, MCP fea_plasticity_assess).

Per node (PEEQ = equivalent plastic strain from ccx, nodal extrapolation averaged by ccx):
  - EN 1993-1-5 annex C: PEEQ ≤ limit_strain of the material (default 5 %),
  - ASME VIII-2 Part 5, 5.3.3.1 (local failure, elastic-plastic analysis):
        ε_L = ε_Lu · exp[ −α_sl / (1 + m2) · ( (σ1 + σ2 + σ3) / (3 σ_e) − 1/3 ) ],   PEEQ (+ ε_cf) ≤ ε_L
    Table 5.7 (R = σ_ys / σ_uts): ferritic steel m2 = 0.60·(1 − R), α_sl = 2.2; austenitic stainless steel and
    nickel alloys m2 = 0.75·(1 − R), α_sl = 0.6. ε_Lu = m2 when no elongation is specified, otherwise
    max(m2, ln(1 + A)) with the elongation A of the material (Table 5.7). The forming strain ε_cf is an input (default 0).
A node shared by parts of different materials takes the worst (highest) utilization. Steps of a load-history sequence
with scale 0 (or without a load case) are "unloading" steps – their stress field is the residual stress.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .plasticity import PlasticitySpec, plastic_table
from .results import von_mises

# ASME VIII-2 Table 5.7 – only the two classes whose values are verified here (other classes: enter m2 / α_sl / ε_Lu)
ASME_CLASSES = {
    "ferritic": {"label": "Ferritic steel", "m2": lambda R: 0.60 * (1.0 - R), "alpha_sl": 2.2},
    "austenitic": {"label": "Austenitic stainless steel / nickel alloy", "m2": lambda R: 0.75 * (1.0 - R),
                   "alpha_sl": 0.6},
}


@dataclass
class AsmeParams:
    """Parameters of the ASME local-failure check. None = from Table 5.7 and the material (class from its name)."""
    material_class: str | None = None     # ferritic | austenitic | None = guess from the material name
    m2: float | None = None
    alpha_sl: float | None = None
    eps_Lu: float | None = None
    eps_cf: float = 0.0                   # forming strain (cold work) added to PEEQ


def plastic_materials(study) -> dict[str, object]:
    return {k: m for k, m in study.materials.items() if getattr(m, "is_plastic", False)}


def curve(m) -> list[tuple[float, float]]:
    """(true stress [MPa], plastic strain [–]) table that the solver gets for the material (ValueError if incomplete)."""
    spec = PlasticitySpec.from_dict(m.plastic)
    if spec.model != "multilinear" and not spec.table and not m.Re:
        raise ValueError(f"Material '{m.name}': the plasticity model needs the yield strength Re")
    return plastic_table(spec, m.Re or 0.0, m.E)


def stress_strain(m, n: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic uniaxial curve for a plot: total strain ε = σ/E + ε_pl(σ) and σ (elastic part from 0 to the yield point)."""
    tab = np.asarray(curve(m), float)
    s_pl, e_pl = tab[:, 0], tab[:, 1]
    e_last = max(float(e_pl[-1]), 1e-3)
    ep = np.linspace(0.0, e_last, n)
    s = np.interp(ep, e_pl, s_pl)
    return np.concatenate([[0.0], s / m.E + ep]), np.concatenate([[0.0], s])


def true_table_from_engineering(rows, E: float) -> list[tuple[float, float]]:
    """Engineering tensile curve (σ_eng [MPa], total ε_eng [–]) from the yield point on → hardening table (true σ,
    plastic ε_pl): σ = σ_eng(1 + ε), ε_pl = ln(1 + ε) − σ/E (plasticity.eng_to_true_stress_strain). The first point becomes
    the yield point (ε_pl = 0); points whose ε_pl does not increase are dropped. Valid up to uniform elongation (necking: no).
    """
    from .plasticity import eng_to_true_stress_strain
    a = np.asarray(rows, float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 1:
        raise ValueError("The engineering curve needs rows of two numbers: stress, total strain")
    s, ep = eng_to_true_stress_strain(a[:, 0], a[:, 1], E)
    ep = ep - ep[0]
    out = [(float(s[0]), 0.0)]
    for si, ei in zip(s[1:], ep[1:]):
        if ei > out[-1][1] + 1e-12 and si >= out[-1][0]:
            out.append((float(si), float(ei)))
    return out


def material_class(m) -> str:
    low = m.name.lower()
    return "austenitic" if any(k in low for k in ("austenit", "stainless", "nerez", "1.43", "1.44", "1.45", "304", "316",
                                                      "inconel", "in718", "in625", "nickel")) else "ferritic"


def asme_constants(m, p: AsmeParams | None = None) -> dict:
    """m2, α_sl, ε_Lu, R and the class for one material (Table 5.7 unless given in p)."""
    p = p or AsmeParams()
    spec = PlasticitySpec.from_dict(m.plastic) if getattr(m, "is_plastic", False) else PlasticitySpec()
    cls = p.material_class or material_class(m)
    if cls not in ASME_CLASSES:
        raise ValueError(f"Unknown ASME material class '{cls}' (ferritic, austenitic)")
    rm = m.Rm or spec.Rm
    if not m.Re or not rm:
        raise ValueError(f"Material '{m.name}': the ASME check needs Re and Rm (R = Re / Rm)")
    R = float(m.Re) / float(rm)
    c = ASME_CLASSES[cls]
    m2 = p.m2 if p.m2 is not None else c["m2"](R)
    alpha = p.alpha_sl if p.alpha_sl is not None else c["alpha_sl"]
    if p.eps_Lu is not None:
        eps_lu, src = p.eps_Lu, "entered"
    else:
        a = spec.A
        if a and a > 1.0:
            a = a / 100.0
        eps_lu, src = (max(m2, math.log(1.0 + a)), "max(m2, ln(1 + A))") if a else (m2, "m2 (elongation not specified)")
    if m2 <= 0 or eps_lu <= 0:
        raise ValueError(f"Material '{m.name}': m2 = {m2:.4g}, ε_Lu = {eps_lu:.4g} must be positive (R = Re/Rm = {R:.3f})")
    return {"class": cls, "R": R, "m2": float(m2), "alpha_sl": float(alpha), "eps_Lu": float(eps_lu), "eps_Lu_source": src}


def _node_parts(mesh) -> list[np.ndarray]:
    return [np.unique(mesh.tets[mesh.tet_part == pi]) for pi in range(len(mesh.part_names))]


def unloading_steps(study, results) -> set[int]:
    """Indices of result steps that are unloading steps of the load history (stress = residual stress)."""
    an = study.analysis
    if an.branches or not an.sequence:
        return set()
    zero = {s.name for s in an.sequence if (s.case is None or abs(s.scale) < 1e-12) and not s.prescribed}
    return {i for i, st in enumerate(results.steps) if st.name in zero}


UNLOADING_STEP = "Unloading (residual stress)"


def add_unloading_step(study) -> bool:
    """Load history for residual stresses: sequence = the load cases one after another (scale 1, as the Contacts step
    builds it) + a final step with scale 0 (unloading). False = the sequence already ends with an unloading step / no cases.
    """
    from .contacts.model import LoadStep
    if not study.load_cases:
        return False
    an = study.analysis
    if an.branches or not an.sequence:
        an.kind, an.branches = "nonlinear", False
        an.sequence = [LoadStep(lc.name, lc.name) for lc in study.load_cases]
    last = an.sequence[-1]
    if (last.case is None or abs(last.scale) < 1e-12) and not last.prescribed:
        return False
    name, k = UNLOADING_STEP, 2
    while name in {s.name for s in an.sequence}:
        name, k = f"{UNLOADING_STEP} {k}", k + 1
    an.sequence.append(LoadStep(name, case=last.case, scale=0.0))
    return True


def assess(study, results, step_index: int, standard: str = "en1993", asme: AsmeParams | None = None) -> dict:
    """Assessment of one result step. Returns a JSON-friendly dict:
    {standard, step, max_peeq, where {node, xyz, part}, utilization, ok, node_utilization (np.ndarray, not JSON),
     materials {name: {...constants, max_peeq, utilization}}, curve_end_exceeded [materials], residual (unloading step)}.
    """
    if not results.steps or not 0 <= step_index < len(results.steps):
        raise ValueError(f"Step {step_index} does not exist")
    st = results.steps[step_index]
    peeq = st.fields.get("peeq")
    if peeq is None:
        raise ValueError("The step has no equivalent plastic strain PEEQ – plasticity is not defined in a material of the model")
    peeq = np.maximum(np.asarray(peeq, float), 0.0)          # nodal extrapolation can dip slightly below 0
    std = standard.lower()
    if std not in ("en1993", "asme"):
        raise ValueError(f"Unknown standard '{standard}' (en1993, asme)")
    mesh = results.mesh
    nn = len(mesh.nodes)
    util = np.zeros(nn)
    limit_node = np.full(nn, np.nan)
    mats_out: dict[str, dict] = {}
    exceeded = []
    p = asme or AsmeParams()
    if std == "asme":
        vm = von_mises(st.S)
        hyd = (st.S[:, 0] + st.S[:, 1] + st.S[:, 2]) / 3.0
        triax = np.where(vm > 1e-6, hyd / np.maximum(vm, 1e-12), 0.0)
    for pi, nodes in enumerate(_node_parts(mesh)):
        try:
            m = study.material_of(mesh.part_names[pi])
        except KeyError:
            continue
        if not getattr(m, "is_plastic", False) or nodes.size == 0:
            continue
        spec = PlasticitySpec.from_dict(m.plastic)
        if std == "en1993":
            lim = np.full(nodes.size, float(spec.limit_strain))
            row = {"limit_strain": float(spec.limit_strain)}
        else:
            c = asme_constants(m, p)
            ex = np.clip(-(c["alpha_sl"] / (1.0 + c["m2"])) * (triax[nodes] - 1.0 / 3.0), -50.0, 50.0)
            lim = c["eps_Lu"] * np.exp(ex)
            row = dict(c)
        u = (peeq[nodes] + (p.eps_cf if std == "asme" else 0.0)) / lim
        worse = u > util[nodes]
        util[nodes[worse]] = u[worse]
        limit_node[nodes[worse]] = lim[worse]
        mp, mu = float(peeq[nodes].max()), float(u.max())
        old = mats_out.setdefault(m.name, dict(row, max_peeq=0.0, utilization=0.0))
        old["max_peeq"], old["utilization"] = max(old["max_peeq"], mp), max(old["utilization"], mu)
        try:
            if mp > curve(m)[-1][1] + 1e-12:
                exceeded.append(m.name)
        except ValueError:
            pass
    i_peeq = int(np.argmax(peeq))
    i_util = int(np.argmax(util))

    def where(i):
        parts = [mesh.part_names[pi] for pi in np.unique(mesh.tet_part[np.any(mesh.tets == i, axis=1)])]
        return {"node": i + 1, "xyz": [float(v) for v in mesh.nodes[i]], "parts": parts}
    out = {
        "standard": std, "step": st.name, "step_index": step_index, "partial": bool(st.partial),
        "max_peeq": float(peeq[i_peeq]), "where_peeq": where(i_peeq),
        "utilization": float(util[i_util]), "where": where(i_util), "ok": bool(util[i_util] <= 1.0),
        "limit_at_max": None if np.isnan(limit_node[i_util]) else float(limit_node[i_util]),
        "plastic_nodes": int(np.count_nonzero(peeq > 1e-6)),
        "materials": mats_out, "curve_end_exceeded": sorted(set(exceeded)),
        "node_utilization": util,
    }
    if std == "asme":
        out["triaxiality_at_max"] = float(triax[i_util])
        out["eps_cf"] = float(p.eps_cf)
    if step_index in unloading_steps(study, results):
        vm = st.mises
        k = int(np.argmax(vm))
        out["residual"] = {"max_mises_MPa": float(vm[k]), "where": where(k)}
    return out


def summary(study, results, asme: AsmeParams | None = None) -> list[dict]:
    """Assessment (EN 1993 and ASME where possible) of every step with PEEQ – rows for the report and the GUI table."""
    rows = []
    for i, st in enumerate(results.steps):
        if st.fields.get("peeq") is None:
            continue
        en = assess(study, results, i, "en1993")
        try:
            am = assess(study, results, i, "asme", asme)
            asme_err = None
        except ValueError as e:
            am, asme_err = None, str(e)
        rows.append({"step": st.name, "index": i, "en1993": en, "asme": am, "asme_error": asme_err})
    return rows


def public(d: dict) -> dict:
    """The assessment without the nodal array (JSON for MCP / report)."""
    return {k: v for k, v in d.items() if k != "node_utilization"}
