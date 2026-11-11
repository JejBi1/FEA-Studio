"""Material nonlinearity – elastoplasticity of metals (B1).

Supported models:
  - Bilinear isotropic / kinematic hardening (BISO / BKIN)
  - Multilinear isotropic / kinematic hardening (MISO / MKIN)
  - Ramberg-Osgood / Hollomon from the limits Re, Rm, A
  - Assessment according to EN 1993-1-5 annex C (limit 5 % plastic strain)
    and ASME VIII-2 Part 5 §5.3.3 (local failure according to the triaxiality).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class PlasticitySpec:
    """Definition of the plastic properties of a material."""
    model: str = "bilinear"                # "bilinear" | "multilinear" | "ramberg_osgood" | "hollomon"
    hardening: str = "isotropic"          # "isotropic" | "kinematic"
    Et: float | None = None               # tangent hardening modulus [MPa] for the bilinear model
    table: list[tuple[float, float]] = field(default_factory=list)  # [(sigma_true, eps_pl_true), ...]
    K_prime: float | None = None          # Ramberg-Osgood hardening coefficient [MPa]
    n_prime: float | None = None          # Ramberg-Osgood hardening exponent [-]
    limit_strain: float = 0.05            # limit value of plastic strain according to EN 1993 (5 %)
    Rm: float | None = None               # optional tensile strength for Hollomon [MPa]
    A: float | None = None                # optional elongation for Hollomon [- or %]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "hardening": self.hardening,
            "Et": self.Et,
            "table": [list(row) for row in self.table],
            "K_prime": self.K_prime,
            "n_prime": self.n_prime,
            "limit_strain": self.limit_strain,
            "Rm": self.Rm,
            "A": self.A,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlasticitySpec:
        if not d:
            return cls()
        return cls(
            model=d.get("model", "bilinear"),
            hardening=d.get("hardening", "isotropic"),
            Et=d.get("Et"),
            table=[(float(r[0]), float(r[1])) for r in d.get("table", [])],
            K_prime=d.get("K_prime"),
            n_prime=d.get("n_prime"),
            limit_strain=d.get("limit_strain", 0.05),
            Rm=d.get("Rm"),
            A=d.get("A"),
        )


def eng_to_true_stress_strain(sigma_eng: float | np.ndarray, eps_eng: float | np.ndarray, E: float) -> tuple[np.ndarray, np.ndarray]:
    """Converts the engineering (nominal) stress and strain to the true (Cauchy) stress and plastic strain.

    sigma_true = sigma_eng * (1 + eps_eng)
    eps_true = ln(1 + eps_eng)
    eps_pl = max(0, eps_true - sigma_true / E)
    """
    s_eng = np.asarray(sigma_eng, dtype=float)
    e_eng = np.asarray(eps_eng, dtype=float)
    s_true = s_eng * (1.0 + e_eng)
    e_true = np.log(1.0 + e_eng)
    e_pl = np.maximum(0.0, e_true - s_true / E)
    return s_true, e_pl


def make_bilinear_table(Re: float, Et: float, max_eps_pl: float = 0.20, E: float | None = None) -> list[tuple[float, float]]:
    """Creates a 2-point table for the bilinear hardening (*PLASTIC).

    The true plastic hardening modulus in the coordinates (sigma, eps_pl) is:
      H = dsigma / deps_pl = E * Et / (E - Et)
    where Et is the tangent modulus of the tensile test dsigma / deps_tot.
    """
    Re = float(Re)
    Et = max(0.0, float(Et))
    if Et == 0.0:
        H = 0.0
    else:
        if E is None:
            raise ValueError("Young's modulus E is required to compute the plastic hardening modulus H from the tangent modulus Et.")
        E = float(E)
        if Et >= E:
            raise ValueError(f"Tangent modulus Et ({Et:g} MPa) must be less than Young's modulus E ({E:g} MPa).")
        H = (E * Et) / (E - Et)
    sigma_max = Re + H * max_eps_pl
    return [(Re, 0.0), (sigma_max, max_eps_pl)]


def make_ramberg_osgood_table(Re: float, K_prime: float, n_prime: float, E: float | None = None,
                              max_eps_pl: float = 0.20, n_points: int = 20) -> list[tuple[float, float]]:
    """Generates a table of true stress and plastic strain for Ramberg-Osgood.

    eps_pl = (sigma / K')^(1/n')
    The curve is anchored at the yield strength (Re, eps_pl=0):
      sigma(eps_pl) = K' * (eps_pl + (Re / K')^(1 / n'))^n'
    so that sigma(0) = Re and the course is smoothly continuous from the yield strength.
    """
    Re = float(Re)
    K_prime = float(K_prime)
    n_prime = float(n_prime)
    if K_prime <= 0.0 or n_prime <= 0.0:
        raise ValueError(f"Ramberg-Osgood parameters must be positive: K_prime={K_prime}, n_prime={n_prime}")
    eps_0 = (Re / K_prime) ** (1.0 / n_prime)

    points = [(Re, 0.0)]
    for ep in np.linspace(0.0005, max_eps_pl, n_points - 1):
        s = K_prime * ((ep + eps_0) ** n_prime)
        points.append((float(s), float(ep)))
    return points


def make_hollomon_table(Re: float, Rm: float, A: float, E: float, n_points: int = 15) -> list[tuple[float, float]]:
    """Generates a table of true stress and plastic strain from Re, Rm, A and E.

    Uses a power model anchored at the yield strength (Re, eps_pl=0):
      sigma_true = K * (eps_0 + eps_pl)^n_exp
    where n_exp corresponds to the uniform strain Ag (Considère, Ag ≈ 0.5 * A),
    while eps_0 and K are determined so that sigma(eps_pl=0) = Re exactly
    and at the tensile strength point sigma(eps_pl_u) = Rm * (1 + Ag).
    """
    Re = float(Re)
    Rm = float(Rm)
    E = float(E)
    A = float(A) if A is not None and A > 0 else 0.20
    # If A is in percent (e.g. 20 %), convert to the relative value (0.20)
    if A > 1.0:
        A = A / 100.0

    # Estimate of the uniform elongation A_g (typically 40 to 60 % of the total A)
    Ag = max(0.02, min(0.25, 0.5 * A))
    n_exp = Ag  # Considère criterion for the maximum of the engineering stress

    s_u_true = Rm * (1.0 + Ag)
    e_u_true = math.log(1.0 + Ag)
    e_pl_u = max(1e-3, e_u_true - s_u_true / E)

    # Anchoring: sigma(0) = Re and sigma(e_pl_u) = s_u_true
    ratio = s_u_true / max(1.0, Re)
    if ratio > 1.0 and n_exp > 0:
        eps_0 = e_pl_u / (ratio ** (1.0 / n_exp) - 1.0)
        K = Re / (eps_0 ** n_exp)
    else:
        eps_0 = 0.0
        K = Re

    points = [(Re, 0.0)]
    e_pl_max = math.log(1.0 + A) - (Rm * (1.0 + A)) / E
    e_pl_max = max(0.05, e_pl_max)

    for ep in np.linspace(0.001, e_pl_max, n_points - 1):
        s = K * ((ep + eps_0) ** n_exp)
        s = max(Re, min(s, 2.5 * Rm))
        points.append((float(s), float(ep)))

    # Make sure the stress is strictly non-decreasing
    clean = [points[0]]
    for s, ep in points[1:]:
        if s > clean[-1][0] and ep > clean[-1][1]:
            clean.append((s, ep))
        elif ep > clean[-1][1]:
            clean.append((clean[-1][0] + 1e-3, ep))
    return clean


def format_plastic_ccx(spec: PlasticitySpec, Re: float, E: float) -> list[str]:
    """Generates the lines of the card *PLASTIC for CalculiX / Abaqus input."""
    hard = spec.hardening.upper()
    if hard not in ("ISOTROPIC", "KINEMATIC"):
        hard = "ISOTROPIC"
    lines = [f"*PLASTIC, HARDENING={hard}"]

    table = spec.table
    if not table:
        if spec.model == "bilinear":
            et = spec.Et if spec.Et is not None else E / 100.0
            table = make_bilinear_table(Re, et, E=E)
        elif spec.model == "ramberg_osgood":
            if spec.K_prime is None or spec.n_prime is None:
                raise ValueError(
                    f"Model 'ramberg_osgood' requires the parameters K_prime and n_prime (entered K_prime={spec.K_prime}, n_prime={spec.n_prime})."
                )
            table = make_ramberg_osgood_table(Re, spec.K_prime, spec.n_prime, E=E)
        elif spec.model == "hollomon":
            rm = getattr(spec, "Rm", None) or 1.25 * Re
            a = getattr(spec, "A", None) or 0.20
            table = make_hollomon_table(Re, rm, a, E)
        else:
            raise ValueError(
                f"Unknown plasticity model '{spec.model}' (supported: 'bilinear', 'multilinear', 'ramberg_osgood', 'hollomon')."
            )

    # Make sure the first point has a plastic strain of 0.0
    if abs(table[0][1]) > 1e-12:
        table = [(table[0][0], 0.0)] + [(s, ep) for s, ep in table if ep > 0]

    for s, ep in table:
        lines.append(f"{s:.4f}, {ep:.6f}")
    return lines


def assess_plastic_strain_en1993(peeq: np.ndarray, limit: float = 0.05) -> dict[str, float]:
    """Assessment of plastic strain according to EN 1993-1-5 Annex C / EN 1993-1-14.

    Criterion: PEEQ <= 5% (0.05).
    """
    max_peeq = float(np.max(peeq)) if peeq.size > 0 else 0.0
    utilization = max_peeq / limit if limit > 0 else float("inf")
    return {
        "max_peeq": max_peeq,
        "limit_strain": limit,
        "utilization": utilization,
        "ok": utilization <= 1.0,
    }


def assess_local_failure_asme(
    sigma_tensor: np.ndarray,
    peeq: np.ndarray,
    eps_Lu: float = 0.30,
    m2: float | None = None,
    R: float = 0.65,
    material_type: str = "ferritic",
    eps_L0: float | None = None,
    **kwargs,
) -> dict[str, float]:
    """Assessment of local failure according to ASME VIII-2 Part 5 §5.3.3 (Protection Against Local Failure).

    Limit plastic strain at a given stress triaxiality:
      eps_L = eps_Lu * exp[ - (m2 / (1 + m2)) * ((sigma_1 + sigma_2 + sigma_3) / (3 * sigma_e) - 1/3) ]

    where:
      (sigma_1 + sigma_2 + sigma_3) / (3 * sigma_e) = stress triaxiality (s_hyd / s_vM),
      m2 = 0.60 * (1 - R) for ferritic steels (Table 5.7, where R = Sy / Su),
      m2 = 0.75 * (1 - R) for austenitic stainless steels,
      eps_Lu is the material parameter of the ductility limit (Table 5.7, default value 0.30).
    """
    if eps_L0 is not None:
        eps_Lu = eps_L0  # backward compatibility

    if m2 is None:
        mat_lower = material_type.lower()
        if "austenit" in mat_lower or "stainless" in mat_lower:
            m2 = 0.75 * (1.0 - R)
        else:
            m2 = 0.60 * (1.0 - R)

    if sigma_tensor.ndim < 2 or sigma_tensor.shape[-1] != 6 or peeq.size == 0:
        return {"max_utilization": 0.0, "ok": True}

    from .results import von_mises
    vm = von_mises(sigma_tensor)
    s_hyd = (sigma_tensor[:, 0] + sigma_tensor[:, 1] + sigma_tensor[:, 2]) / 3.0

    with np.errstate(divide="ignore", invalid="ignore"):
        triax = np.where(vm > 1e-3, s_hyd / vm, 0.0)
        # Limit according to ASME VIII-2 Part 5 §5.3.3
        exponent = - (m2 / (1.0 + m2)) * (triax - 1.0 / 3.0)
        exponent = np.clip(exponent, -10.0, 10.0)
        eps_limit = eps_Lu * np.exp(exponent)
        utilization = np.where(eps_limit > 1e-6, peeq / eps_limit, 0.0)

    max_util = float(np.max(utilization))
    return {
        "max_utilization": max_util,
        "ok": max_util <= 1.0,
        "eps_Lu": float(eps_Lu),
        "m2": float(m2),
    }


def check(study, part_names: list[str]) -> tuple[list[str], list[str]]:
    """Check of plasticity before the calculation (errors, warnings).

    Kinematic hardening: CalculiX 2.23 computes it correctly only with NLGEOM (in linear geometry it behaves as isotropic,
    or it diverges – verified against analytics and Abaqus, verification PL4).
    If NLGEOM is not switched on, it is switched on automatically with a warning. If the user explicitly disabled NLGEOM,
    an error is reported.
    """
    errors, warnings = [], []
    for p in part_names:
        try:
            m = study.material_of(p)
        except KeyError:
            continue
        spec = PlasticitySpec.from_dict(m.plastic) if isinstance(getattr(m, "plastic", None), dict) else None
        if spec is None or not getattr(m, "is_plastic", False):
            continue
        if spec.hardening.lower() == "kinematic":
            if not study.contact_settings.nlgeom:
                explicitly_disabled = (
                    getattr(study.contact_settings, "explicit_nlgeom", None) is False
                    or getattr(study.contact_settings, "allow_nlgeom", True) is False
                    or getattr(study.contact_settings, "disable_nlgeom", False) is True
                    or getattr(study.contact_settings, "nlgeom_disabled", False) is True
                    or getattr(study, "explicit_nlgeom", None) is False
                    or getattr(study, "allow_nlgeom", True) is False
                    or getattr(study, "disable_nlgeom", False) is True
                    or getattr(study, "nlgeom_disabled", False) is True
                )
                if explicitly_disabled:
                    errors.append(f"Material '{m.name}' (part '{p}'): kinematic hardening is calculated correctly by CalculiX only with NLGEOM (without it it behaves isotropically / diverges) – enable large deformations (NLGEOM) or use isotropic hardening")
                else:
                    study.contact_settings.nlgeom = True
                    warnings.append(f"Material '{m.name}' (part '{p}'): kinematic hardening requires large deformations (NLGEOM) – NLGEOM option was enabled automatically")
    return errors, warnings
