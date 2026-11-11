"""Combinations of load states and result envelopes (A2).

The standard factors are loaded only from `fea_core/data/load_factors.json` (ČSN EN 1990).
They apply strictly to linear calculations (statics in ccx / NGSolve).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .results import Results, StepResult, von_mises, principal
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .study import Study

DATA_PATH = Path(__file__).resolve().parent / "data" / "load_factors.json"


def load_standards_data() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Configuration file with the code load factors not found: {DATA_PATH}")
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


# Action types accepted by the generator (`action_type`); Czech aliases kept next to the English ones.
_PERMANENT_TYPES = ("permanent", "stale", "stálé", "constant", "g")
_VARIABLE_TYPES = ("variable", "promenne", "proměnné", "q")
_EXCLUDED_TYPES = ("excluded", "none", "vynechat")
ACTION_TYPES = ("permanent", "variable", "excluded")

# Whole words (lower case, without diacritics) for the classification guess of a load-case name.
# Never single letters: "g" or "q" inside an arbitrary word ("Loading", "Bearing") says nothing.
_GUESS_WORDS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("excluded", "", frozenset({"accidental", "impact", "explosion", "fire", "seismic", "earthquake",
                               "mimoradne", "mimoradna", "mimoradny", "naraz", "vybuch", "pozar", "seizmicita"})),
    ("variable", "snow_cz_low", frozenset({"snow", "snih", "snehem", "snehu", "snehova", "snehove"})),
    ("variable", "wind", frozenset({"wind", "vitr", "vetrem", "vetru", "vetrne"})),
    ("variable", "temperature", frozenset({"temperature", "thermal", "teplota", "teplotou", "teplotni"})),
    ("variable", "cat_H", frozenset({"roof", "strecha", "strechy"})),
    ("variable", "cat_A", frozenset({"imposed", "live", "occupancy", "uzitne", "uzitna", "uzitny", "uzitim"})),
    ("permanent", "", frozenset({"dead", "selfweight", "weight", "permanent", "gravity",
                                 "tiha", "vlastni", "stale", "stala", "staly"})),
)
# Default for a name without a recognised word: variable action with the largest ψ factors (cat_E: ψ0 = 1.0,
# ψ1 = 0.9, ψ2 = 0.8) – conservative for the accompanying values; the user has to confirm or change it.
_UNKNOWN_CATEGORY = "cat_E"


def _words(name: str) -> set[str]:
    import re
    import unicodedata
    plain = unicodedata.normalize("NFKD", name.lower())
    plain = "".join(ch for ch in plain if not unicodedata.combining(ch))
    words = set(re.findall(r"[a-z]+", plain))
    if "self" in words and "weight" in words:
        words.add("selfweight")
    return words


def guess_action_class(name: str) -> dict[str, Any]:
    """Conservative guess of the EN 1990 classification of one load case from its name (whole words only).

    Returns {"name", "action_type": permanent | variable | excluded, "category": key of the ψ table or "",
    "favorable": False, "recognized": bool, "reason": str}. The result is only a proposal for the user to confirm.
    """
    from .joints.model import is_preload
    words = _words(name)
    if is_preload(name) or words & {"preload", "prestress", "predpeti"}:
        return {"name": name, "action_type": "permanent", "category": "", "favorable": True, "recognized": True,
                "reason": "prestress / bolt preload: permanent, favourable (γ = 1.0 as γ_P) – check"}
    for action, category, keys in _GUESS_WORDS:
        hit = sorted(words & keys)
        if hit:
            reason = f"word '{hit[0]}'"
            if action == "excluded":
                reason += ": accidental actions are not combined (no accidental design situation)"
            return {"name": name, "action_type": action, "category": category, "favorable": False,
                    "recognized": True, "reason": reason}
    return {"name": name, "action_type": "variable", "category": _UNKNOWN_CATEGORY, "favorable": False,
            "recognized": False, "reason": "not recognised: variable action with the largest ψ (category E) – check"}


def guess_action_classes(names: list[str]) -> list[dict[str, Any]]:
    """`guess_action_class` for all load cases (order kept)."""
    return [guess_action_class(n) for n in names]


def psi_categories(standard: str = "CSN_EN_1990") -> dict[str, str]:
    """Keys of the ψ table → description (from load_factors.json)."""
    std = load_standards_data()["standards"][standard]
    return {k: v.get("description", k) for k, v in std["psi"].items()}


@dataclass
class Combination:
    """Combination of load states or envelope.

    name: name of the combination (e.g. 'MSÚ 6.10 - Užitné hlavní')
    factors: multiples of the states {case_name: factor}
    standard: 'CSN_EN_1990'
    design_situation: 'STR_GEO_6_10' | 'STR_GEO_6_10a_6_10b' | 'EQU' | 'SLS_CHAR' | 'SLS_FREQ' | 'SLS_QUASI' | 'custom'
    description: comment / citation of the standard
    envelope: True if it is an envelope (max/min over a set of combinations)
    envelope_type: 'max' | 'min' | 'abs_max'
    envelope_combinations: list of names of the combinations the envelope is made of
    envelope_field: 'von_mises' | 'umag' | 's1' | 's3' | 'safety'
        ('safety' = safety from the Stress Tool (assessment) per node; for the envelope of the worst case use envelope_type='min')
    """
    name: str
    factors: dict[str, float] = field(default_factory=dict)
    standard: str = "CSN_EN_1990"
    design_situation: str = "STR_GEO_6_10"
    description: str = ""
    envelope: bool = False
    envelope_type: str = "max"
    envelope_combinations: list[str] = field(default_factory=list)
    envelope_field: str = "von_mises"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "factors": dict(self.factors),
            "standard": self.standard,
            "design_situation": self.design_situation,
            "description": self.description,
            "envelope": self.envelope,
            "envelope_type": self.envelope_type,
            "envelope_combinations": list(self.envelope_combinations),
            "envelope_field": self.envelope_field,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Combination:
        return cls(
            name=d["name"],
            factors=dict(d.get("factors", {})),
            standard=d.get("standard", "CSN_EN_1990"),
            design_situation=d.get("design_situation", "STR_GEO_6_10"),
            description=d.get("description", ""),
            envelope=bool(d.get("envelope", False)),
            envelope_type=d.get("envelope_type", "max"),
            envelope_combinations=list(d.get("envelope_combinations", [])),
            envelope_field=d.get("envelope_field", "von_mises"),
        )


ENVELOPE_FIELDS = ("von_mises", "umag", "s1", "s3", "safety")
ENVELOPE_TYPES = ("max", "min", "abs_max")


def validate_combination(study: Study, comb: Combination, others: list[Combination] | None = None) -> None:
    """Checks the combination against the study: names of the factors must be load states, an envelope of combinations or states.

    others = combinations the envelope may refer to (default `study.combinations`). Error = ValueError.
    """
    cases = [lc.name for lc in getattr(study, "load_cases", [])]
    if comb.envelope:
        if comb.envelope_type not in ENVELOPE_TYPES:
            raise ValueError(f"Unknown envelope type '{comb.envelope_type}' (expected {', '.join(ENVELOPE_TYPES)})")
        if comb.envelope_field not in ENVELOPE_FIELDS:
            raise ValueError(f"Unknown envelope field '{comb.envelope_field}' (expected {', '.join(ENVELOPE_FIELDS)})")
        if not comb.envelope_combinations:
            raise ValueError(f"Envelope '{comb.name}' has no source combinations defined")
        pool = others if others is not None else list(getattr(study, "combinations", []))
        known = {c.name for c in pool if not c.envelope} | set(cases)
        bad = [n for n in comb.envelope_combinations if n not in known]
        if bad:
            raise ValueError(f"Envelope '{comb.name}': unknown source combinations {bad}. Available: {sorted(known)}")
        return
    if not comb.factors:
        raise ValueError(f"Combination '{comb.name}' has no load case factors defined")
    bad = [n for n in comb.factors if n not in cases]
    if bad:
        raise ValueError(f"Combination '{comb.name}': unknown load cases {bad}. Study load cases: {cases}")


@dataclass
class EnvelopeResult:
    """Result of the nodal envelope over a set of combinations."""
    name: str
    field_name: str
    envelope_type: str
    values: np.ndarray                      # (N,)
    governing_combinations: list[str]       # (N,) name of the governing combination for each node
    combination_names: list[str]
    min_val: float
    max_val: float
    governing_summary: dict[str, int] = field(default_factory=dict)  # number of nodes governed by the given combination


def generate_en1990_combinations(
    cases: list[dict[str, Any]],
    situation: str | None = None,
    standard: str = "CSN_EN_1990",
    formula: str | None = None,
) -> list[Combination]:
    """Generator of combinations according to ČSN EN 1990.

    The argument `cases` contains the specification of each state:
      - 'name': name of the state in Study.load_cases
      - 'kind': 'G' (permanent), 'Q' (variable), 'A' (accidental) or 'action_type': 'permanent'|'variable'
      - 'favorable': bool (True = favourable action, False = unfavourable action)
      - 'category': key from load_factors.json (e.g. 'cat_A', 'cat_B', 'snow_cz_low', 'wind', 'temperature')
    """
    sit = formula or situation or "STR_GEO_6_10"
    if sit == "6.10":
        sit = "STR_GEO_6_10"
    elif sit in ("6.10a_6.10b", "6.10a/b"):
        sit = "STR_GEO_6_10a_6_10b"

    if sit == "all":
        res_all = []
        for s in ("STR_GEO_6_10", "STR_GEO_6_10a_6_10b", "EQU", "SLS_CHAR", "SLS_FREQ", "SLS_QUASI"):
            res_all.extend(generate_en1990_combinations(cases, situation=s, standard=standard))
        if res_all:
            res_all.append(Combination(
                name="EN1990_STR_GEO_Obalka_max",
                envelope=True,
                envelope_type="max",
                envelope_combinations=[c.name for c in res_all if "STR_GEO" in c.name],
                standard=standard,
                description="Envelope of maximum values for ULS STR/GEO per ČSN EN 1990",
            ))
        return res_all

    situation = sit

    # Normalisation of the inputs `cases`
    norm_cases = []
    for c in cases:
        c2 = dict(c)
        if "action_type" in c2 and "kind" not in c2:
            act = str(c2["action_type"]).lower()
            if act in _PERMANENT_TYPES:
                c2["kind"] = "G"
            elif act in _VARIABLE_TYPES:
                c2["kind"] = "Q"
            elif act in _EXCLUDED_TYPES:
                continue                       # deliberately left out of the combinations
            else:
                raise ValueError(
                    f"Load case '{c2.get('name')}': action type '{c2['action_type']}' is not supported "
                    f"(expected permanent, variable or excluded; accidental design situations are not generated)")
        elif str(c2.get("kind", "")).upper() not in ("G", "Q"):
            raise ValueError(f"Load case '{c2.get('name')}': kind '{c2.get('kind')}' is not supported (expected G or Q)")
        cat = c2.get("category", "")
        if cat in ("A", "B", "C", "D", "E", "F", "G", "H"):
            c2["category"] = f"cat_{cat}"
        elif c2.get("subtype") in ("snow_below_1000m", "snih", "sníh"):
            c2["category"] = "snow_cz_low"
        elif c2.get("subtype") in ("wind", "vitr", "vítr"):
            c2["category"] = "wind"
        norm_cases.append(c2)
    cases = norm_cases

    data = load_standards_data()
    std = data["standards"].get(standard)
    if not std:
        raise ValueError(f"Standard '{standard}' is not defined in load_factors.json")
    for c in cases:
        cat = c.get("category")
        if c["kind"].upper() == "Q" and cat and cat not in std["psi"]:
            raise ValueError(f"Load case '{c.get('name')}': unknown category '{cat}' "
                             f"(expected one of {', '.join(std['psi'])})")

    sit_data = std["design_situations"].get(situation)
    if not sit_data:
        raise ValueError(f"Design situation '{situation}' is not in standard '{standard}'")

    psi_table = std["psi"]
    g_cases = [c for c in cases if c.get("kind", "").upper() == "G"]
    q_cases = [c for c in cases if c.get("kind", "").upper() == "Q"]

    combs: list[Combination] = []

    def get_psi(c: dict, idx: int) -> float:
        cat = c.get("category", "cat_A")
        factors = psi_table.get(cat, {})
        key = f"psi_{idx}"
        if key not in factors:
            raise ValueError(f"Unknown factor {key} for category '{cat}'")
        return float(factors[key])

    # 1. ULS STR/GEO – equation 6.10
    if situation == "STR_GEO_6_10":
        g_unfav = sit_data["gamma_G_unfav"]
        g_fav = sit_data["gamma_G_fav"]
        q_unfav = sit_data["gamma_Q_unfav"]
        citation = sit_data.get("citation", "")

        def base_g_factors() -> dict[str, float]:
            res = {}
            for g in g_cases:
                res[g["name"]] = g_fav if g.get("favorable", False) else g_unfav
            return res

        if not q_cases:
            # Permanent loads only
            combs.append(Combination(
                name="EN1990_STR_GEO_6.10_G_only",
                factors=base_g_factors(),
                standard=standard,
                design_situation=situation,
                description=f"ČSN EN 1990 equation 6.10 (permanent loads only). {citation}",
            ))
        else:
            # Each variable load once as the leading one (Q_1)
            for lead in q_cases:
                f_map = base_g_factors()
                lead_name = lead["name"]
                f_map[lead_name] = 0.0 if lead.get("favorable", False) else q_unfav

                for acc in q_cases:
                    if acc["name"] == lead_name:
                        continue
                    if acc.get("favorable", False):
                        f_map[acc["name"]] = 0.0
                    else:
                        psi0 = get_psi(acc, 0)
                        f_map[acc["name"]] = round(q_unfav * psi0, 6)

                combs.append(Combination(
                    name=f"EN1990_STR_GEO_6.10_{lead_name}_hlavni",
                    factors=f_map,
                    standard=standard,
                    design_situation=situation,
                    description=f"ČSN EN 1990 equation 6.10 (leading variable actions: {lead_name}). {citation}",
                ))

    # 2. ULS STR/GEO – equations 6.10a and 6.10b
    elif situation == "STR_GEO_6_10a_6_10b":
        g_unfav = sit_data["gamma_G_unfav"]
        g_fav = sit_data["gamma_G_fav"]
        xi = sit_data["xi"]
        q_unfav = sit_data["gamma_Q_unfav"]
        citation = sit_data.get("citation", "")

        # 6.10a: all variable with psi_0
        f_map_a = {}
        for g in g_cases:
            f_map_a[g["name"]] = g_fav if g.get("favorable", False) else g_unfav
        for q in q_cases:
            if q.get("favorable", False):
                f_map_a[q["name"]] = 0.0
            else:
                f_map_a[q["name"]] = round(q_unfav * get_psi(q, 0), 6)

        combs.append(Combination(
            name="EN1990_STR_GEO_6.10a",
            factors=f_map_a,
            standard=standard,
            design_situation=situation,
            description=f"ČSN EN 1990 equation 6.10a (all variable actions with ψ0). {citation}",
        ))

        # 6.10b: permanent with xi*gamma_G, one variable leading
        if not q_cases:
            f_map_b = {g["name"]: (g_fav if g.get("favorable", False) else round(xi * g_unfav, 6)) for g in g_cases}
            combs.append(Combination(
                name="EN1990_STR_GEO_6.10b_G_only",
                factors=f_map_b,
                standard=standard,
                design_situation=situation,
                description=f"ČSN EN 1990 equation 6.10b (permanent only, with ξ). {citation}",
            ))
        else:
            for lead in q_cases:
                f_map_b = {}
                for g in g_cases:
                    f_map_b[g["name"]] = g_fav if g.get("favorable", False) else round(xi * g_unfav, 6)
                lead_name = lead["name"]
                f_map_b[lead_name] = 0.0 if lead.get("favorable", False) else q_unfav
                for acc in q_cases:
                    if acc["name"] == lead_name:
                        continue
                    if acc.get("favorable", False):
                        f_map_b[acc["name"]] = 0.0
                    else:
                        psi0 = get_psi(acc, 0)
                        f_map_b[acc["name"]] = round(q_unfav * psi0, 6)

                combs.append(Combination(
                    name=f"EN1990_STR_GEO_6.10b_{lead_name}_hlavni",
                    factors=f_map_b,
                    standard=standard,
                    design_situation=situation,
                    description=f"ČSN EN 1990 equation 6.10b (leading variable actions: {lead_name}, permanent reduced by ξ). {citation}",
                ))

    # 3. EQU – loss of static equilibrium
    elif situation == "EQU":
        g_unfav = sit_data["gamma_G_unfav"]
        g_fav = sit_data["gamma_G_fav"]
        q_unfav = sit_data["gamma_Q_unfav"]
        citation = sit_data.get("citation", "")

        def base_equ_g() -> dict[str, float]:
            return {g["name"]: (g_fav if g.get("favorable", False) else g_unfav) for g in g_cases}

        if not q_cases:
            combs.append(Combination(
                name="EN1990_EQU_G_only",
                factors=base_equ_g(),
                standard=standard,
                design_situation=situation,
                description=f"ČSN EN 1990 EQU (permanent loads only). {citation}",
            ))
        else:
            for lead in q_cases:
                f_map = base_equ_g()
                lead_name = lead["name"]
                f_map[lead_name] = 0.0 if lead.get("favorable", False) else q_unfav
                for acc in q_cases:
                    if acc["name"] == lead_name:
                        continue
                    if acc.get("favorable", False):
                        f_map[acc["name"]] = 0.0
                    else:
                        f_map[acc["name"]] = round(q_unfav * get_psi(acc, 0), 6)
                combs.append(Combination(
                    name=f"EN1990_EQU_{lead_name}_hlavni",
                    factors=f_map,
                    standard=standard,
                    design_situation=situation,
                    description=f"ČSN EN 1990 EQU (leading variable actions: {lead_name}). {citation}",
                ))

    # 4. SLS characteristic 6.14
    elif situation == "SLS_CHAR":
        citation = sit_data.get("citation", "")
        base_g = {g["name"]: 1.0 for g in g_cases}
        if not q_cases:
            combs.append(Combination(
                name="EN1990_SLS_CHAR_G_only",
                factors=base_g,
                standard=standard,
                design_situation=situation,
                description=f"ČSN EN 1990 expression 6.14 (permanent only). {citation}",
            ))
        else:
            for lead in q_cases:
                f_map = dict(base_g)
                lead_name = lead["name"]
                f_map[lead_name] = 0.0 if lead.get("favorable", False) else 1.0
                for acc in q_cases:
                    if acc["name"] == lead_name:
                        continue
                    f_map[acc["name"]] = 0.0 if acc.get("favorable", False) else round(get_psi(acc, 0), 6)
                combs.append(Combination(
                    name=f"EN1990_SLS_CHAR_{lead_name}_hlavni",
                    factors=f_map,
                    standard=standard,
                    design_situation=situation,
                    description=f"ČSN EN 1990 characteristic 6.14 (leading variable actions: {lead_name}). {citation}",
                ))

    # 5. SLS frequent 6.15
    elif situation == "SLS_FREQ":
        citation = sit_data.get("citation", "")
        base_g = {g["name"]: 1.0 for g in g_cases}
        if not q_cases:
            combs.append(Combination(
                name="EN1990_SLS_FREQ_G_only",
                factors=base_g,
                standard=standard,
                design_situation=situation,
                description=f"ČSN EN 1990 expression 6.15 (permanent only). {citation}",
            ))
        else:
            for lead in q_cases:
                f_map = dict(base_g)
                lead_name = lead["name"]
                f_map[lead_name] = 0.0 if lead.get("favorable", False) else round(get_psi(lead, 1), 6)
                for acc in q_cases:
                    if acc["name"] == lead_name:
                        continue
                    f_map[acc["name"]] = 0.0 if acc.get("favorable", False) else round(get_psi(acc, 2), 6)
                combs.append(Combination(
                    name=f"EN1990_SLS_FREQ_{lead_name}_hlavni",
                    factors=f_map,
                    standard=standard,
                    design_situation=situation,
                    description=f"ČSN EN 1990 frequent 6.15 (leading variable actions: {lead_name}). {citation}",
                ))

    # 6. SLS quasi-permanent 6.16
    elif situation == "SLS_QUASI":
        citation = sit_data.get("citation", "")
        f_map = {g["name"]: 1.0 for g in g_cases}
        for q in q_cases:
            f_map[q["name"]] = 0.0 if q.get("favorable", False) else round(get_psi(q, 2), 6)
        combs.append(Combination(
            name="EN1990_SLS_QUASI",
            factors=f_map,
            standard=standard,
            design_situation=situation,
            description=f"ČSN EN 1990 quasi-permanent 6.16. {citation}",
        ))

    return combs


def combine_results(study: Study, results: Results, combination: Combination) -> StepResult:
    """Linear combination of the results of the states for the given combination.

    U, S, reactions and displacements of reference nodes are superposed linearly.
    Von Mises and principal stresses are computed from the combined tensor S.
    Results of joints (`StepResult.joints`: forces, utilisation of bolts and welds) are not combined – they contain coordinates, utilisation
    and other non-linear data; the joint assessment must be made from the design forces of the individual states.
    """
    if study.is_nonlinear():
        raise ValueError(
            "Load case combinations are allowed only for linear analyses. Nonlinear contact, large deformations (NLGEOM) and nonlinear stepping do not support superposition."
        )

    if not combination.factors:
        raise ValueError(f"Combination '{combination.name}' has no factors defined for its load cases")

    step_map: dict[str, StepResult] = {s.name: s for s in results.steps}
    # Support for referencing by state number (index 0, 1...) – the real state names take precedence
    for i, s in enumerate(results.steps):
        step_map.setdefault(str(i), s)
        step_map.setdefault(f"LoadCase_{i+1}", s)

    combined_u: np.ndarray | None = None
    combined_s: np.ndarray | None = None
    combined_reactions: dict[str, np.ndarray] = {}
    combined_reaction_moments: dict[str, np.ndarray] = {}
    ref_disps: dict[int, np.ndarray] = {}
    eq_total: dict[str, np.ndarray] | None = {"reaction_total": np.zeros(3), "reaction_moment_total": np.zeros(3)}

    for case_name, factor in combination.factors.items():
        if abs(factor) < 1e-15:
            continue
        step = step_map.get(case_name)
        if step is None:
            raise KeyError(
                f"Combination '{combination.name}' refers to a non-existent load case '{case_name}'. Available load cases: {list(step_map.keys())}"
            )

        # U
        if combined_u is None:
            combined_u = factor * step.U.copy()
        else:
            combined_u += factor * step.U

        # S (tenzor v uzlech)
        if combined_s is None:
            combined_s = factor * step.S.copy()
        else:
            combined_s += factor * step.S

        # Reakce
        for sel, rf in step.reactions.items():
            if sel not in combined_reactions:
                combined_reactions[sel] = factor * rf.copy()
            else:
                combined_reactions[sel] += factor * rf

        # Reaction moments
        for sel, rm in step.reaction_moments.items():
            if sel not in combined_reaction_moments:
                combined_reaction_moments[sel] = factor * rm.copy()
            else:
                combined_reaction_moments[sel] += factor * rm

        # Total reaction from the equilibrium check (each node once – summing the selections would count shared nodes twice)
        eq = step.equilibrium or {}
        if eq_total is not None and eq.get("reaction_total") is not None \
                and eq.get("reaction_moment_total") is not None:
            for key in eq_total:
                eq_total[key] = eq_total[key] + factor * np.asarray(eq[key], float)[:3]
        else:
            eq_total = None

        # Displacements at the reference nodes
        for n_id, u_ref in step.ref_displacements.items():
            if n_id not in ref_disps:
                ref_disps[n_id] = factor * u_ref.copy()
            else:
                ref_disps[n_id] += factor * u_ref


    if combined_u is None or combined_s is None:
        # All factors were zero
        n_nodes = len(results.mesh.nodes)
        combined_u = np.zeros((n_nodes, 3), dtype=np.float64)
        combined_s = np.zeros((n_nodes, 6), dtype=np.float64)

    out = StepResult(
        name=combination.name,
        U=combined_u,
        S=combined_s,
        reactions=combined_reactions,
        reaction_moments=combined_reaction_moments,
        ref_displacements=ref_disps,
        moment_point=results.steps[0].moment_point if results.steps else None,
    )
    if eq_total is not None:          # total reaction only (CSV export) – the combination is not checked for equilibrium
        out.equilibrium = {k: v.tolist() for k, v in eq_total.items()}
    return out


def compute_envelope(
    study: Study,
    results: Results,
    combination: Combination,
    comb_results: dict[str, StepResult] | None = None,
) -> EnvelopeResult:
    """Computes the nodal envelope over a set of combinations.

    For each node the max or min of the chosen field (von_mises, umag, s1, s3, safety) is computed
    and the name of the governing combination is recorded.
    """
    if not combination.envelope:
        raise ValueError(f"Combination '{combination.name}' is not defined as an envelope (envelope=True)")

    target_comb_names = combination.envelope_combinations
    if not target_comb_names:
        raise ValueError(f"Envelope '{combination.name}' has no source combinations defined")

    # We find or compute the results of the combinations
    if comb_results is None:
        comb_results = {}

    study_combs = {c.name: c for c in getattr(study, "combinations", [])}
    for c_name in target_comb_names:
        if c_name not in comb_results:
            if c_name in study_combs:
                if study_combs[c_name].envelope:
                    raise ValueError(f"Envelope '{combination.name}' cannot be based on another envelope '{c_name}'")
                comb_results[c_name] = combine_results(study, results, study_combs[c_name])
            else:
                # Maybe it is directly the name of a state
                step_map = {s.name: s for s in results.steps}
                if c_name in step_map:
                    comb_results[c_name] = step_map[c_name]
                else:
                    raise KeyError(f"Source combination '{c_name}' for the envelope not found")

    field_name = combination.envelope_field or "von_mises"
    env_type = combination.envelope_type or "max"

    def get_field_values(step: StepResult) -> np.ndarray:
        if field_name in ("von_mises", "mises", "vm"):
            return step.mises
        elif field_name in ("umag", "u_mag", "u_sum"):
            return step.umag
        elif field_name in ("s1", "principal_1", "max_principal"):
            return step.principal[:, 0]
        elif field_name in ("s3", "principal_3", "min_principal"):
            return step.principal[:, 2]
        elif field_name in ("safety", "safety factor", "bezpecnost"):
            from .assessment import compute_safety_field
            return compute_safety_field(study, results.mesh, step)[0]
        else:
            raise ValueError(f"Unsupported field for the envelope: '{field_name}'")

    field_arrays = [get_field_values(comb_results[c_name]) for c_name in target_comb_names]
    stacked = np.stack(field_arrays, axis=0)  # (K, N)

    if env_type == "max":
        governing_idx = np.argmax(stacked, axis=0)  # (N,)
        extrema = np.max(stacked, axis=0)
    elif env_type == "min":
        governing_idx = np.argmin(stacked, axis=0)
        extrema = np.min(stacked, axis=0)
    elif env_type == "abs_max":
        governing_idx = np.argmax(np.abs(stacked), axis=0)
        # We return the values with their sign
        n_nodes = stacked.shape[1]
        extrema = stacked[governing_idx, np.arange(n_nodes)]
    else:
        raise ValueError(f"Unknown envelope type: '{env_type}' (expected 'max', 'min' or 'abs_max')")

    gov_comb_names = [target_comb_names[i] for i in governing_idx]

    # Summary of the frequency of the governing combinations
    summary: dict[str, int] = {}
    for c_name in target_comb_names:
        summary[c_name] = 0
    unique, counts = np.unique(governing_idx, return_counts=True)
    for u, cnt in zip(unique, counts):
        summary[target_comb_names[u]] = int(cnt)

    return EnvelopeResult(
        name=combination.name,
        field_name=field_name,
        envelope_type=env_type,
        values=extrema,
        governing_combinations=gov_comb_names,
        combination_names=list(target_comb_names),
        min_val=float(extrema.min()),
        max_val=float(extrema.max()),
        governing_summary=summary,
    )


def compute_combination(study: Study, results: Results, combination: Combination,
                        cache: dict[str, StepResult] | None = None) -> StepResult | EnvelopeResult:
    """Result of a combination (StepResult) or an envelope (EnvelopeResult) according to `combination.envelope`.

    cache = cache of combination results by name (shared between envelopes and export).
    """
    if cache is None:
        cache = {}
    if combination.envelope:
        return compute_envelope(study, results, combination, comb_results=cache)
    if combination.name not in cache:
        cache[combination.name] = combine_results(study, results, combination)
    return cache[combination.name]
