"""Fatigue analysis over the results of the linear statics (superposition of the load states)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .settings import (FatigueSettings, EQUIVALENT_LABELS, MEAN_LABELS, CP_RATIO_DUCTILE, findley_k,
                       findley_lambda)
from .sn import SNCurve, estimate_sn, marin, apply_marin
from . import kernel


def _is_preload_step(name: str) -> bool:
    return name.startswith(("Bolt preload", "Preload", "Předpětí"))

EQUIVALENTS = {"signed_mises": (EQUIVALENT_LABELS["signed_mises"], kernel.EQ_SIGNED_MISES),
               "max_principal": (EQUIVALENT_LABELS["max_principal"], kernel.EQ_MAX_PRINCIPAL)}
MEAN_METHODS = {
    "none": (MEAN_LABELS["none"], kernel.MEAN_NONE),
    "goodman": (MEAN_LABELS["goodman"], kernel.MEAN_GOODMAN),
    "gerber": (MEAN_LABELS["gerber"], kernel.MEAN_GERBER),
}


@dataclass
class FatigueResult:
    damage: np.ndarray        # damage per block at the nodes
    safety: np.ndarray        # multiple of the stress to failure at the design number of blocks
    amplitude: np.ndarray     # max. amplitude of the equivalent stress at the node [MPa]
    repeats: float
    sn_used: dict             # material -> {"sn": SNCurve dict, "marin": {...}} or {"en": StrainLifeParams dict}
    t: np.ndarray
    factors: np.ndarray
    load_cases: list[str]
    settings: FatigueSettings
    time_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    normal: np.ndarray | None = None      # critical plane: normal (n, 3)
    shear_dir: np.ndarray | None = None   # critical plane: direction of the shear in the plane (Findley), otherwise 0
    cp_used: dict = field(default_factory=dict)   # material -> parameters of the criterion (cp_params)

    @property
    def total_damage(self) -> np.ndarray:
        return self.damage * self.repeats

    @property
    def life_blocks(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.where(self.damage > 0, 1.0 / self.damage, np.inf)

    def method_lines(self) -> list[str]:
        """Description of the method for the report (in Czech, incl. the parameters of the criterion per material)."""
        return self.settings.describe() + [f"{name}: {cp_text(p)}" for name, p in self.cp_used.items()]

    def worst(self, mesh, n: int = 10) -> list[dict]:
        order = np.argsort(-self.damage)[:n]
        return [{"node": int(i), "xyz": mesh.nodes[i].tolist(), "damage_per_block": float(self.damage[i]),
                 "total_damage": float(self.damage[i] * self.repeats),
                 "life_blocks": float(self.life_blocks[i]), "safety": float(self.safety[i]),
                 "amplitude": float(self.amplitude[i]),
                 "normal": self.normal[i].tolist() if self.normal is not None else None,
                 "shear_dir": self.shear_dir[i].tolist() if self.shear_dir is not None else None}
                for i in order]


def sn_for_material(mat, settings: FatigueSettings) -> tuple[SNCurve, dict]:
    """S-N used for the material (given or estimated from Rm) with the Marin factors included."""
    given = settings.sn.get(mat.name)
    if given:
        base, estimated = SNCurve(**given), False
    else:
        from .sn import material_class
        # for plastics the estimate has no point 0.9 Rm at 10³ (Shigley is for metals) → the slope is not recomputed with Marin
        base, estimated = estimate_sn(mat.E, mat.Rm), material_class(mat.E) != "plast"
    m = marin(mat.Rm, settings.surface, settings.size_d, settings.reliability,
              settings.k_load, settings.k_temp, settings.k_misc)
    return apply_marin(base, m["k"], mat.Rm, estimated), m


def en_for_material(mat, settings: FatigueSettings):
    """Parameters of ε-N for the material (given or estimated from Bäumel-Seeger UML)."""
    from .strain_life import StrainLifeParams, estimate_strain_life
    if getattr(mat, "is_print", False) or getattr(mat, "E_z", None) is not None:
        raise ValueError(
            f"Material '{mat.name}': the ε-N (strain-life) method is not supported for 3D printing. Use the standard S-N analysis or the printing fatigue module."
        )
    given = settings.en_params.get(mat.name)
    if given:
        return StrainLifeParams.from_dict(given)
    if getattr(mat, "Rm", None) is None:
        raise ValueError(
            f"Material '{mat.name}': the tensile strength Rm is missing for determining the ε-N parameters."
        )
    rm = float(mat.Rm)
    e = float(getattr(mat, "E", None) or 210000.0)
    return estimate_strain_life(rm, e, material_type=mat.name)


# ------------------------------------------------------------------ critical plane – material parameters
_BRITTLE = ("šedá litina", "seda litina", "šedé litiny", "gjl", "grauguss", "grey iron", "gray iron")


def is_brittle(mat) -> bool:
    """Brittle material (grey cast iron) → failure by the normal stress. According to the name / the converter of designations."""
    name = mat.name.lower()
    if any(s in name for s in _BRITTLE):
        return True
    try:
        from .. import grades
        g = grades.grade_of(mat.name)
    except Exception:  # noqa: BLE001 – the converter is only a hint
        return False
    return bool(g and g.family == "litina-seda")


def cp_params(mat, settings: FatigueSettings) -> dict:
    """Critical-plane criterion for the material: {criterion, ratio, k, lam, source}."""
    brittle = is_brittle(mat)
    crit = settings.cp_criterion
    if crit == "auto":
        crit = "normal" if brittle else "findley"
    ratio = settings.cp_ratio.get(mat.name)
    if ratio:
        src = "user-defined"
    elif brittle:
        ratio, src = 1.0, "brittle material (gray cast iron): f₋₁/t₋₁ ≈ 1"
    else:
        ratio, src = CP_RATIO_DUCTILE, "ductile material: f₋₁/t₋₁ = √3 (von Mises, Shigley)"
    out = {"criterion": crit, "ratio": float(ratio), "k": 0.0, "lam": 1.0, "source": src}
    if crit == "findley":
        try:
            out["k"] = findley_k(float(ratio))
        except ValueError as e:
            raise ValueError(f"{mat.name}: {e}") from None
        out["lam"] = findley_lambda(out["k"])
    return out


def cp_text(p: dict) -> str:
    if p["criterion"] == "findley":
        return (f"Findley, f₋₁/t₋₁ = {p['ratio']:.3f} ({p['source']}), k = {p['k']:.4f}, λ = {p['lam']:.4f} "
                f"(krut: σekv = {p['ratio']:.3f}·τa)")
    return "normal stress in the plane (S-N directly, mean stress correction per settings)"


# ------------------------------------------------------------------ histories at the node
def _node_sig(results, settings: FatigueSettings, node: int) -> tuple[np.ndarray, np.ndarray]:
    lcs = [s.name for s in results.steps]
    t, F = settings.history.factors(lcs)
    S = np.stack([s.S[node] for s in results.steps])           # (K, 6)
    return t, F @ S                                             # (T, 6)


def node_history(results, settings: FatigueSettings, node: int) -> tuple[np.ndarray, np.ndarray]:
    """History of the equivalent stress at the node (for the graph and the check; not for the critical plane)."""
    t, sig = _node_sig(results, settings, node)
    code = EQUIVALENTS.get(settings.equivalent, EQUIVALENTS["signed_mises"])[1]
    eq = np.array([kernel.equivalent(*row, code) for row in sig])
    return t, eq


def node_plane_history(results, fr: FatigueResult, node: int) -> tuple[np.ndarray, dict]:
    """Critical plane: history σn(t) and the shear τ_d(t) in the critical plane and in the direction of the node."""
    from .critical_plane import plane_series
    t, sig = _node_sig(results, fr.settings, node)
    sn, td = plane_series(sig, fr.normal[node], fr.shear_dir[node])
    out = {"σn": sn}
    if np.any(fr.shear_dir[node]):
        out["τ (direction d)"] = td
    return t, out


def proportional(F: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray] | None:
    """F (T, K) = f ⊗ c (all channels proportional)? Returns (f, c) with max |c| = 1, otherwise None."""
    norms = np.linalg.norm(F, axis=0)
    if norms.max() == 0:
        return None
    kmax = int(np.argmax(norms))
    f = F[:, kmax]
    c = F.T @ f / (f @ f)
    if np.abs(F - np.outer(f, c)).max() > tol * np.abs(F).max():
        return None
    return f, c


def run_fatigue(study, results, nodes: np.ndarray | None = None) -> FatigueResult:
    t0 = time.time()
    st = study.fatigue
    if st is None:
        from .settings import FatigueSettings
        st = FatigueSettings()
    is_en = (st.method == "en")

    # Validation of the settings of the mean-stress correction
    if is_en:
        mean_key = st.mean_correction if st.mean_correction in ("none", "morrow", "swt") else st.en_mean_correction
        if mean_key in ("goodman", "gerber"):
            raise ValueError(
                f"Mean stress correction '{mean_key}' is not supported for the ε-N (strain-life) method. Valid options for ε-N: none, morrow, swt."
            )
    else:
        if st.mean_correction in ("morrow", "swt"):
            raise ValueError(
                f"Mean stress correction '{st.mean_correction}' is intended for the ε-N (strain-life) method. For the S-N method choose from: {', '.join(MEAN_METHODS.keys())} (none, goodman, gerber)."
            )
        if st.mean_correction not in MEAN_METHODS:
            raise ValueError(
                f"Unknown mean stress correction '{st.mean_correction}'. Valid options for the S-N method: {', '.join(MEAN_METHODS.keys())}."
            )

    nl = (getattr(results, "info", None) or {}).get("nonlinear")
    pre_warn: list[str] = []
    if nl:
        # contact / history: superposition does not hold → fatigue only over the real snapshots of the sequence (specification F8, 4.5)
        if nl.get("mode") != "sequence":
            raise ValueError("Fatigue with contact / nonlinear analysis in branches is not available – superposition does not apply and branches have no history. Define a sequence of load steps (≥ 2 block cycles).")
        lcs = [s.name for s in results.steps]
        F, t, pre_warn = _sequence_history(results)
    else:
        if not st.history.channels:
            raise ValueError("No load history specified (fatigue)")
        lcs = [s.name for s in results.steps]
        t, F = st.history.factors(lcs)
    if len(t) < 2:
        raise ValueError("The load history must have at least 2 points")
    from ..study import Displacement, BearingLoad
    moving = [c.selection for c in study.constraints if isinstance(c, Displacement)
              and any(v not in (None, 0, 0.0) for v in (c.ux, c.uy, c.uz))]
    # a bearing load is not linear in the direction of the force (pressure only on the half of the hole facing the force) → superposition
    # of states with BearingLoad gives a different stress than the real (e.g. rotating) reaction (specification F8, 3.5)
    used = set(lcs)
    bearing = sorted({l.selection for lc in study.load_cases if lc.name in used for l in lc.loads
                      if isinstance(l, BearingLoad)})
    mesh = results.mesh
    n_all = len(mesh.nodes)
    idx = np.arange(n_all) if nodes is None else np.asarray(nodes)
    S = np.ascontiguousarray(np.stack([s.S[idx] for s in results.steps]))   # (K, n, 6)
    cpm = st.critical_plane

    # parameters of S-N / ε-N per node (a node shared by several parts → the worse material = smaller SD or worse ε-N)
    sn_used, cp_used = {}, {}
    SD = np.full(n_all, np.inf)
    ND = np.ones(n_all)
    K = np.ones(n_all)
    K2 = np.zeros(n_all)
    RM = np.zeros(n_all)
    CR = np.zeros(n_all, np.int64)
    KF = np.zeros(n_all)
    LAM = np.ones(n_all)

    # Parameters for ε-N (strain-life)
    E_arr = np.zeros(n_all)
    Kp_arr = np.zeros(n_all)
    np_arr = np.zeros(n_all)
    sfp_arr = np.zeros(n_all)
    b_arr = np.zeros(n_all)
    efp_arr = np.zeros(n_all)
    c_arr = np.zeros(n_all)
    metric_arr = np.full(n_all, np.inf)

    warnings = list(pre_warn)
    if moving:
        warnings.append("Nonzero prescribed displacement (" + ", ".join(moving) + ") is in EVERY load case – in superposition it is multiplied by the channel multiples and summed. For fatigue, put the displacement into a separate case and calculate the other cases with zero displacement.")
    if bearing:
        warnings.append("Bearing force (BearingLoad – " + ", ".join(bearing) + ") cannot be superposed: the pressure acts "
                        "only on the half of the hole facing the force, so multiples of the states do not match a force of another "
                        "direction (a rotating reaction). For fatigue use a remote load with a distributing coupling (RemoteLoad "
                        "RBE3, 6 unit components – the conversion from a mechanism does this automatically).")

    for pi, pname in enumerate(mesh.part_names):
        mat = study.material_of(pname)
        pn = np.unique(mesh.tets[mesh.tet_part == pi])
        if is_en:
            if mat.name not in sn_used:
                en_p = en_for_material(mat, st)
                sn_used[mat.name] = {"en": en_p.to_dict(), "source": en_p.source}
                if "UML" in en_p.source or "odhad" in en_p.source.lower():
                    warnings.append(f"{mat.name}: ε-N parameters are estimated (UML) from Rm – verification by tests is recommended")
            p_dict = sn_used[mat.name]["en"]
            # Choice of the worse material for shared nodes: the lower limit strain amplitude at N=1e6
            metric = (p_dict["sigma_f_prime"] / p_dict["E"]) * (2e6 ** p_dict["b"]) + p_dict["eps_f_prime"] * (2e6 ** p_dict["c"])
            worse = metric < metric_arr[pn]
            sel = pn[worse]
            metric_arr[sel] = metric
            E_arr[sel] = p_dict["E"]
            Kp_arr[sel] = p_dict["K_prime"]
            np_arr[sel] = p_dict["n_prime"]
            sfp_arr[sel] = p_dict["sigma_f_prime"]
            b_arr[sel] = p_dict["b"]
            efp_arr[sel] = p_dict["eps_f_prime"]
            c_arr[sel] = p_dict["c"]
        else:
            if mat.name not in sn_used:
                sn, m = sn_for_material(mat, st)
                sn_used[mat.name] = {"sn": sn.to_dict(), "marin": m}
                if "KALIBRACE" in sn.source:
                    warnings.append(f"{mat.name}: the S-N curve is only approximate – calibration from tests required")
                if cpm:
                    cp_used[mat.name] = cp_params(mat, st)
                    if "litin" in mat.name.lower() and not is_brittle(mat) and mat.name not in st.cp_ratio:
                        warnings.append(f"{mat.name}: cast iron – check the endurance limit ratio f₋₁/t₋₁ (default √3 applies to ductile steels)")
            sn = SNCurve(**sn_used[mat.name]["sn"])
            worse = sn.SD < SD[pn]
            sel = pn[worse]
            SD[sel], ND[sel], K[sel], K2[sel], RM[sel] = sn.SD, sn.ND, sn.k, sn.k2, (mat.Rm or 0.0)
            if cpm:
                p = cp_used[mat.name]
                CR[sel] = 1 if p["criterion"] == "findley" else 0
                KF[sel], LAM[sel] = p["k"], p["lam"]

    out_D = np.zeros(len(idx))
    out_sf = np.zeros(len(idx))
    out_a = np.zeros(len(idx))
    prop = proportional(F)
    normal = shear = None

    if is_en:
        from . import strain_life
        rule_code = strain_life.RULE_GLINKA if st.en_notch_rule == "glinka" else strain_life.RULE_NEUBER
        mean_key = st.mean_correction if st.mean_correction in ("none", "morrow", "swt") else st.en_mean_correction
        if mean_key == "swt":
            mean_code_en = strain_life.MEAN_EN_SWT
        elif mean_key == "morrow":
            mean_code_en = strain_life.MEAN_EN_MORROW
        else:
            mean_code_en = strain_life.MEAN_EN_NONE

        eq_code = EQUIVALENTS.get(st.equivalent, EQUIVALENTS["signed_mises"])[1]
        if prop is not None:
            f, c = prop
            Seff = np.ascontiguousarray(np.tensordot(c, S, axes=(0, 0)))
            strain_life.fatigue_nodes_en_proportional(
                np.ascontiguousarray(f), Seff, eq_code, rule_code, mean_code_en,
                E_arr[idx], Kp_arr[idx], np_arr[idx], sfp_arr[idx], b_arr[idx],
                efp_arr[idx], c_arr[idx], float(st.repeats), bool(st.closed_blocks),
                out_D, out_sf, out_a
            )
        else:
            strain_life.fatigue_nodes_en(
                np.ascontiguousarray(F), S, eq_code, rule_code, mean_code_en,
                E_arr[idx], Kp_arr[idx], np_arr[idx], sfp_arr[idx], b_arr[idx],
                efp_arr[idx], c_arr[idx], float(st.repeats), bool(st.closed_blocks),
                out_D, out_sf, out_a
            )
    elif cpm:
        from . import critical_plane as cp
        mean_code = MEAN_METHODS.get(st.mean_correction, (None, kernel.MEAN_GOODMAN))[1]
        planes, levels, ndir, ncand, ndp, exact = st.search_params()
        P = cp.hemisphere(planes)
        normal, shear = np.zeros((len(idx), 3)), np.zeros((len(idx), 3))
        common = (CR[idx], KF[idx], LAM[idx], mean_code, SD[idx], ND[idx], K[idx], K2[idx], RM[idx],
                  float(st.repeats), bool(st.closed_blocks), out_D, out_sf, out_a, normal, shear)
        if prop is not None:
            f, c = prop
            Seff = np.ascontiguousarray(np.tensordot(c, S, axes=(0, 0)))
            cp.cp_nodes_proportional(np.ascontiguousarray(f), Seff, P, cp.plane_spacing(planes), levels, *common)
        else:
            cp.cp_nodes(np.ascontiguousarray(F), S, P, cp.plane_spacing(planes), levels, ndir,
                        np.array([ncand, ndp, exact], np.int64), *common)
    else:
        mean_code = MEAN_METHODS.get(st.mean_correction, (None, kernel.MEAN_GOODMAN))[1]
        args = (EQUIVALENTS[st.equivalent][1], mean_code, SD[idx], ND[idx], K[idx],
                K2[idx], RM[idx], float(st.repeats), bool(st.closed_blocks), out_D, out_sf, out_a)
        if prop is not None:      # σ(t) = f(t)·Σ c_k S_k → rainflow only once
            f, c = prop
            Seff = np.ascontiguousarray(np.tensordot(c, S, axes=(0, 0)))
            kernel.fatigue_nodes_proportional(np.ascontiguousarray(f), Seff, *args)
        else:
            kernel.fatigue_nodes(np.ascontiguousarray(F), S, *args)

    # 3D printing: the Hill equivalent in the axes of the print / line with the S-N of the print (printing.fatigue) overwrites the printed nodes
    # (also with the critical plane – the printed parts are assessed in the axes of the print, the normal of the plane does not hold there)
    if not is_en:
        from ..printing.fatigue import apply as print_fatigue
        mean_code_sn = MEAN_METHODS.get(st.mean_correction, (None, kernel.MEAN_GOODMAN))[1]
        used = print_fatigue(study, results, F, st, mean_code_sn, idx, out_D, out_sf, out_a, sn_used, warnings)
        if used is not None and normal is not None:
            normal[used], shear[used] = 0.0, 0.0

    if nodes is not None:                     # extend to all nodes
        D, SF, A = np.zeros(n_all), np.full(n_all, np.inf), np.zeros(n_all)
        D[idx], SF[idx], A[idx] = out_D, out_sf, out_a
        out_D, out_sf, out_a = D, SF, A
        if cpm:
            N3, D3 = np.zeros((n_all, 3)), np.zeros((n_all, 3))
            N3[idx], D3[idx] = normal, shear
            normal, shear = N3, D3

    if cpm and prop is None and len(t) > 2000:
        warnings.append(f"Critical plane with non-proportional loading and {len(t)} time points is slow – consider a shorter block (fewer oscillations of the same shape and more repetitions).")

    return FatigueResult(out_D, out_sf, out_a, float(st.repeats), sn_used, t, F, lcs, st,
                         time.time() - t0, warnings, normal, shear, cp_used)


def _sequence_history(results) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fatigue of a nonlinear sequence: the channel = the ends of the steps of the LAST cycle (the first cycle with micro-slip differs –
    Mindlin–Deresiewicz hysteresis), F = unit matrix over these steps; check of stabilisation between the last
    two cycles (change of the stress < 1e-3) – specification F8, 4.5.
    """
    import re
    names = [s.name for s in results.steps]
    cyc = [int(m.group(1)) if (m := re.search(r"\((?:cycle|cyklus) (\d+)\)$", n)) else 0 for n in names]
    warn = []
    last = max(cyc) if cyc else 0
    rows = [i for i, c in enumerate(cyc) if c == last and not _is_preload_step(results.steps[i].name)] \
        if last else [i for i, s in enumerate(results.steps) if not _is_preload_step(s.name)]
    if last >= 2:
        prev = [i for i, c in enumerate(cyc) if c == last - 1]
        if len(prev) == len(rows):
            num = max(float(np.abs(results.steps[a].S - results.steps[b].S).max()) for a, b in zip(rows, prev))
            den = max(float(np.abs(results.steps[a].S).max()) for a in rows) or 1.0
            if num / den > 1e-3:
                warn.append(f"Fatigue with contact: the response differs by {100 * num / den:.2g} % between the last two cycles (> 0.1 %) – add block cycles until it stabilizes")
    else:
        warn.append("Fatigue with contact: the sequence has only 1 cycle – the first cycle with slip may differ from the stabilized one (enter ≥ 2 cycles)")
    if len(rows) < 2:
        raise ValueError("Fatigue with contact: the sequence needs at least 2 steps per cycle")
    F = np.zeros((len(rows), len(results.steps)))
    for r, i in enumerate(rows):
        F[r, i] = 1.0
    warn.append("Fatigue from a nonlinear analysis: history = ends of the steps of the last cycle (the course within a step is not considered)")
    return F, np.arange(len(rows), dtype=float), warn
