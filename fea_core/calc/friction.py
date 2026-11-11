"""Friction in joints and machine parts (specification F8, section 3.3.2) – closed formulas.

All functions take the units of the project (mm, N, N·mm) and return a dictionary with the results, the text of the formula and the source.
Choose the coefficient of friction by the purpose (friction_lib.RULES): transmission of the force by friction → μ_min, resistance of a mechanism → μ_max.
"""
from __future__ import annotations

import math

from .common import CalcError, choice, positive, result

# Rolling bearings: μ related to the bore diameter d (needle: to the pitch diameter D_m), ~0.1·C, good lubrication
BEARING_MU = {"ball": (0.0015, "single-row ball"), "self_aligning_ball": (0.0010, "self-aligning ball"),
              "angular_contact": (0.0020, "angular contact ball"),
              "cylindrical_cage": (0.0011, "cylindrical roller with cage"), "full_complement": (0.0020, "full-complement cylindrical roller"),
              "tapered": (0.0018, "tapered roller"), "spherical_roller": (0.0018, "spherical roller"),
              "thrust_ball": (0.0013, "thrust ball"), "needle": (0.0030, "needle (on D_m)")}
BEARING_SOURCE = "Roymech – Rolling bearing friction (approximate, ~0.1·C, good lubrication; exact SKF model UNVERIFIED)"

# Rolling resistance: arm e [mm]
ROLLING_E = {"steel_steel": (0.5, "steel – steel"), "polymer_steel": (2.0, "polymer – steel"),
             "rubber_steel": (7.7, "hard rubber – steel")}
ROLLING_SOURCE = "Roymech – Rolling resistance (approximate, sources inconsistent)"


def journal(F: float, r: float, mu: float, state: str = "worn") -> dict:
    """Radial journal of radius r [mm] with the force F [N] (resultant of the normal and friction forces). state: clearance (clearance,
    line contact) | worn (run-in, p ∝ cos θ over 180°) | fitted (new fitted, uniform pressure).
    """
    F, r, mu = positive("F", F, zero=True), positive("r", r), positive("mu", mu, zero=True)
    st = choice("state", state, ("clearance", "worn", "fitted"))
    k = {"clearance": 1.0, "worn": 4.0 / math.pi, "fitted": math.pi / 2.0}[st]
    mue = k * mu / math.sqrt(1.0 + mu * mu)
    return result({"mu_eff": mue, "k": k, "M_Nmm": mue * F * r, "M_Nm": mue * F * r / 1e3,
                   "M_range_Nmm": [mu / math.sqrt(1 + mu * mu) * F * r, math.pi / 2 * mu / math.sqrt(1 + mu * mu) * F * r]},
                  "M = μ'·F·r; μ' = k·μ/√(1+μ²), k = 1 (clearance), 4/π (run-in), π/2 (fitted)",
                  "own derivation (F8 specification, 3.3.2), similar to Meriam & Kraige, Statics ch. 6")


def thrust_collar(F: float, r2: float, mu: float, r1: float = 0.0, state: str = "worn") -> dict:
    """Thrust (axial) journal – an annulus r1–r2 [mm] (r1 = 0 full pad), axial force F. state: new (uniform pressure)
    | worn (run-in – uniform wear, p·r = const.).
    """
    F, r2, mu = positive("F", F, zero=True), positive("r2", r2), positive("mu", mu, zero=True)
    r1 = positive("r1", r1, zero=True)
    if r1 >= r2:
        raise CalcError("r1 must be less than r2")
    st = choice("state", state, ("new", "worn"))
    rm = (2.0 / 3.0 * (r2 ** 3 - r1 ** 3) / (r2 ** 2 - r1 ** 2)) if st == "new" else (r1 + r2) / 2.0
    return result({"r_eff_mm": rm, "M_Nmm": mu * F * rm, "M_Nm": mu * F * rm / 1e3},
                  "new: M = (2/3)μF(r₂³ − r₁³)/(r₂² − r₁²); worn: M = μF(r₁ + r₂)/2",
                  "Shigley ch. 16 (clutches – uniform pressure / uniform wear)")


def ball_joint(F: float, R: float, mu: float, case: str = "perpendicular") -> dict:
    """Ball joint of radius R: case axis (rotation about the axis of the load) | perpendicular (about a perpendicular axis,
    p ∝ cos θ) | uniform (uniform pressure).
    """
    F, R, mu = positive("F", F, zero=True), positive("R", R), positive("mu", mu, zero=True)
    c = choice("case", case, ("axis", "perpendicular", "uniform"))
    k = {"axis": 1.0, "perpendicular": 4.0 / math.pi, "uniform": math.pi / 2.0}[c]
    return result({"k": k, "M_Nmm": k * mu * F * R, "M_Nm": k * mu * F * R / 1e3},
                  "M = k·μ·F·R; k = 1 (load axis), 4/π (perpendicular axis, p ~ cos θ), π/2 (uniform pressure)",
                  "F8 specification, 3.3.2 (numerically scratch/friction_research/ball.py)")


def pivot(F: float, a: float, mu: float, distribution: str = "hertz") -> dict:
    """Spin friction of a circular area of radius a [mm] (rotation about the normal): hertz (elliptic p) | uniform."""
    F, a, mu = positive("F", F, zero=True), positive("a", a), positive("mu", mu, zero=True)
    dist = choice("distribution", distribution, ("hertz", "uniform"))
    k = 3.0 * math.pi / 16.0 if dist == "hertz" else 2.0 / 3.0
    return result({"k": k, "M_Nmm": k * mu * F * a}, "M = (3π/16)·μFa ≈ 0.589·μFa (Hertz), (2/3)·μFa (uniform)",
                  "own derivation (F8 specification, 3.3.2); Hertz contact patch see hertz.sphere")


def rolling_resistance(N: float, r: float, e: float | None = None, pair: str = "steel_steel") -> dict:
    """Rolling resistance of a wheel / roller of radius r [mm] under the load N; e arm of the rolling resistance [mm] (None = according to the
    pair: steel_steel 0.5 | polymer_steel 2 | rubber_steel 7.7 – indicative).
    """
    N, r = positive("N", N, zero=True), positive("r", r)
    if e is None:
        p = choice("pair", pair, tuple(ROLLING_E))
        e = ROLLING_E[p][0]
    e = positive("e", e, zero=True)
    return result({"e_mm": e, "F_N": N * e / r, "M_Nmm": N * e, "mu_equiv": e / r},
                  "F = N·e/r, M = N·e", ROLLING_SOURCE,
                  ["Lever arm e is approximate (Roymech, sources differ up to several times) – measure for design"])


def guide_jamming(e: float, L: float, mu: float) -> dict:
    """Sliding guide (drawer): the force acts with the arm e [mm] from the axis of the guide of length L [mm]; jamming for
    e/L ≥ 1/(2μ). Use μs (μ_max, resistance of the mechanism).
    """
    e, L, mu = positive("e", e, zero=True), positive("L", L), positive("mu", mu, zero=True)
    jam = mu > 0 and e / L >= 1.0 / (2.0 * mu)
    ratio = 2.0 * mu * e / L
    vals = {"e_over_L": e / L, "limit_e_over_L": math.inf if mu == 0 else 1.0 / (2.0 * mu), "jams": bool(jam),
            "L_min_mm": 2.0 * mu * e, "reserve": math.inf if ratio == 0 else 1.0 / ratio,
            "force_factor": math.inf if jam else 1.0 / (1.0 - ratio)}
    warn = ["Risk of jamming – lengthen the guide or reduce the force lever arm"] if jam else []
    if not jam and ratio > 0.5:
        warn.append("Margin against jamming < 2 – design L > 2μs·e with margin")
    return result(vals, "jams for e/L ≥ 1/(2μ); required force = F/(1 − 2μe/L)",
                  "own derivation (F8 specification, 3.3.2)", warn)


def v_groove(mu: float, beta_deg: float, F: float = 0.0) -> dict:
    """V-groove (wedge groove, V-belt, prismatic guide) with the apex angle β."""
    mu = positive("mu", mu, zero=True)
    if not 0 < beta_deg < 180:
        raise CalcError("beta_deg must be in (0; 180)°")
    mue = mu / math.sin(math.radians(beta_deg) / 2.0)
    return result({"mu_eff": mue, "factor": mue / mu if mu else 1.0 / math.sin(math.radians(beta_deg) / 2.0),
                   "F_friction_N": mue * positive("F", F, zero=True)},
                  "μ' = μ/sin(β/2)", "Shigley / Roloff-Matek (wedge surfaces)")


def wedge(alpha_deg: float, mu1: float, mu2: float | None = None) -> dict:
    """Wedge with the angle α and friction on both faces (mu2 None = mu1): self-locking for α ≤ ρ₁ + ρ₂, ρ = arctan μ.
    For a self-locking that must hold, use μ_min.
    """
    mu1 = positive("mu1", mu1, zero=True)
    mu2 = mu1 if mu2 is None else positive("mu2", mu2, zero=True)
    rho = math.degrees(math.atan(mu1) + math.atan(mu2))
    return result({"rho_sum_deg": rho, "self_locking": bool(alpha_deg <= rho), "margin_deg": rho - alpha_deg},
                  "self-locking for α ≤ ρ₁ + ρ₂, ρ = arctan μ", "Shigley / Roloff-Matek (wedge joints)")


def rolling_bearing(F: float, d: float, kind: str = "ball", start: bool = False, sealed: bool = False) -> dict:
    """Friction torque of a rolling bearing M = μ·F·d/2 (d = bore diameter; for needle the pitch diameter D_m); start = run-up (2×),
    sealed = seals (+0.001). kind: ball | self_aligning_ball | angular_contact | cylindrical_cage |
    full_complement | tapered | spherical_roller | thrust_ball | needle (names in BEARING_MU).
    """
    F, d = positive("F", F, zero=True), positive("d", d)
    k = choice("kind", kind, tuple(BEARING_MU))
    mu = BEARING_MU[k][0] * (2.0 if start else 1.0) + (0.001 if sealed else 0.0)
    return result({"mu": mu, "M_Nmm": mu * F * d / 2.0, "M_Nm": mu * F * d / 2e3, "type": BEARING_MU[k][1]},
                  "M = μ·F·d/2 (run-in up to 2×, seal +0.001)", BEARING_SOURCE)


def power_screw(F: float, d2: float, P: float, mu: float, flank_angle_deg: float = 30.0, starts: int = 1,
                mu_c: float = 0.0, r_c: float = 0.0) -> dict:
    """Power screw: F axial force [N], d2 pitch diameter of the thread [mm], P pitch [mm], mu friction in the thread,
    flank_angle_deg apex angle of the profile (Tr 30°, metric 60°, square 0°), starts number of starts, mu_c and r_c friction
    and mean radius of the collar / bearing (additional moment μ_c·F·r_c).
    """
    F, d2, P = positive("F", F, zero=True), positive("d2", d2), positive("P", P)
    mu = positive("mu", mu, zero=True)
    n = int(starts)
    if n < 1:
        raise CalcError("starts must be ≥ 1")
    if not 0 <= flank_angle_deg < 120:
        raise CalcError("flank_angle_deg must be in [0; 120)°")
    lam = math.atan(n * P / (math.pi * d2))
    phi = math.atan(mu / math.cos(math.radians(flank_angle_deg) / 2.0))
    Mc = positive("mu_c", mu_c, zero=True) * F * positive("r_c", r_c, zero=True)
    T_up = F * d2 / 2.0 * math.tan(lam + phi) + Mc
    T_down = F * d2 / 2.0 * math.tan(phi - lam) + Mc
    eta = math.tan(lam) / math.tan(lam + phi)
    eta_back = math.tan(lam - phi) / math.tan(lam)
    locking = lam <= phi
    vals = {"lead_mm": n * P, "lambda_deg": math.degrees(lam), "rho_prime_deg": math.degrees(phi),
            "eta": eta, "eta_back": eta_back, "self_locking": bool(locking),
            "T_raise_Nmm": T_up, "T_raise_Nm": T_up / 1e3, "T_lower_Nmm": T_down, "T_lower_Nm": T_down / 1e3,
            "T_collar_Nmm": Mc, "margin_deg": math.degrees(phi - lam)}
    warn = []
    if locking and T_down - Mc < 0.05 * F * d2 / 2.0 * math.tan(phi):
        warn.append("Self-locking with small margin – verify with μ_min (vibration defeats self-locking)")
    if not locking:
        warn.append("Screw is NOT self-locking – the load lowers by itself (T_lowering < 0 = braking torque)")
    return result(vals, "tan λ = n·P/(π·d₂); tan ρ' = μ/cos(α/2); T = F(d₂/2)tan(λ ± ρ') + μ_c·F·r_c; η = tan λ/tan(λ + ρ'), η' = tan(λ − ρ')/tan λ; self-locking λ ≤ ρ'",
                  "Shigley's Mechanical Engineering Design, ch. 8 (Power screws)", warn)


def belt(mu: float, wrap_deg: float, F2: float | None = None, F1: float | None = None, q: float = 0.0,
         v: float = 0.0, groove_angle_deg: float | None = None) -> dict:
    """Belt, rope, band brake (Euler–Eytelwein): mu, wrap angle [°], known force in the slack side F2 or in the tight
    F1 [N], q mass per length [t/mm], v speed [mm/s] (centrifugal force q·v²), groove_angle_deg = angle of the V-belt
    (μ' = μ/sin(β/2)).
    """
    mu = positive("mu", mu, zero=True)
    th = math.radians(positive("wrap_deg", wrap_deg))
    mue = mu if groove_angle_deg is None else mu / math.sin(math.radians(positive("groove_angle_deg",
                                                                                 groove_angle_deg)) / 2.0)
    ratio = math.exp(mue * th)
    Fc = positive("q", q, zero=True) * positive("v", v, zero=True) ** 2
    vals = {"mu_eff": mue, "ratio": ratio, "F_centrifugal_N": Fc}
    if F2 is not None and F1 is not None:
        raise CalcError("Enter only one force: F2 (slack side) or F1 (tight side)")
    if F2 is not None:
        F2 = positive("F2", F2, zero=True)
        F1 = Fc + (F2 - Fc) * ratio
    elif F1 is not None:
        F1 = positive("F1", F1, zero=True)
        F2 = Fc + (F1 - Fc) / ratio
    if F1 is not None:
        vals.update({"F1_N": F1, "F2_N": F2, "F_transmitted_N": F1 - F2})
    warn = []
    if F2 is not None and F2 < Fc:
        warn.append("Slack-side force is less than the centrifugal force q·v² – the belt lifts off")
    return result(vals, "F₁/F₂ = e^(μ'α); with centrifugal force (F₁ − qv²)/(F₂ − qv²) = e^(μ'α); V-belt μ' = μ/sin(β/2) (β = 38° → factor 3.07)",
                  "Euler–Eytelwein (Shigley ch. 17)", warn)


def worm(gamma_deg: float, mu: float, alpha_n_deg: float = 20.0) -> dict:
    """Worm gear with the worm as the driver: γ lead angle of the worm, α_n normal pressure angle."""
    if not 0 < gamma_deg < 90:
        raise CalcError("gamma_deg must be in (0; 90)°")
    mu = positive("mu", mu, zero=True)
    g = math.radians(gamma_deg)
    rho = math.atan(mu / math.cos(math.radians(alpha_n_deg)))
    eta = math.tan(g) / math.tan(g + rho)
    eta_back = math.tan(g - rho) / math.tan(g)
    return result({"rho_prime_deg": math.degrees(rho), "eta": eta, "eta_back": eta_back,
                   "self_locking": bool(g <= rho)},
                  "η = tan γ/tan(γ + ρ'), η' = tan(γ − ρ')/tan γ, tan ρ' = μ/cos α_n; self-locking γ ≤ ρ'",
                  "Shigley ch. 15 (worm gearing)",
                  ["Gear μ depends on sliding velocity (AGMA tables) – UNVERIFIED, enter from catalog"])


def clutch(F: float, R: float, mu: float, r: float = 0.0, surfaces: int = 2, state: str = "worn",
           cone_half_angle_deg: float | None = None) -> dict:
    """Friction clutch / brake: clamping force F [N], annulus r–R [mm], surfaces number of friction surfaces (single-disc
    clutch 2), state worn (uniform wear) | new (uniform pressure), conical: cone_half_angle_deg
    (M/sin α).
    """
    F, R, mu = positive("F", F, zero=True), positive("R", R), positive("mu", mu, zero=True)
    r = positive("r", r, zero=True)
    if r >= R:
        raise CalcError("r must be less than R")
    i = int(surfaces)
    if i < 1:
        raise CalcError("surfaces must be ≥ 1")
    st = choice("state", state, ("new", "worn"))
    rm = (R + r) / 2.0 if st == "worn" else 2.0 / 3.0 * (R ** 3 - r ** 3) / (R ** 2 - r ** 2)
    M = i * mu * F * rm
    if cone_half_angle_deg is not None:
        if not 0 < cone_half_angle_deg <= 90:
            raise CalcError("cone_half_angle_deg must be in (0; 90]°")
        M /= math.sin(math.radians(cone_half_angle_deg))
    if st == "worn":
        p_max = F / (2.0 * math.pi * r * (R - r)) if r > 0 else math.inf     # p·r = const., the largest at r
    else:
        p_max = F / (math.pi * (R * R - r * r))
    return result({"r_m_mm": rm, "M_Nmm": M, "M_Nm": M / 1e3, "p_max_MPa": p_max},
                  "M = i·μ·F·r_m; r_m = (R + r)/2 (worn) or (2/3)(R³ − r³)/(R² − r²) (new); cone /sin α",
                  "Shigley ch. 16 (clutches and brakes)", ["Lining μ is UNVERIFIED – enter from the manufacturer's catalog"])


def inclined_plane(alpha_deg: float) -> dict:
    """Measurement of the static coefficient of friction by an inclined plane: the body starts to move at the angle α → μs = tan α."""
    if not 0 <= alpha_deg < 90:
        raise CalcError("alpha_deg must be in [0; 90)°")
    return result({"mu_s": math.tan(math.radians(alpha_deg))}, "μs = tan α (angle at which the body starts to slide)",
                  "basic statics; tilt slowly, several repetitions, clean surface")


# Snap-fit: W = P(μ + tan α)/(1 − μ tan α) – see snapfit.snap_fit
from .snapfit import snap_fit  # noqa: E402,F401


KINDS = {"journal": (journal, "journal (clearance / worn / fitted)"),
         "thrust_collar": (thrust_collar, "thrust collar, annulus (new / worn)"),
         "ball_joint": (ball_joint, "ball joint"),
         "pivot": (pivot, "pivoting (spin) friction of contact patch"),
         "rolling_resistance": (rolling_resistance, "rolling resistance"),
         "guide_jamming": (guide_jamming, "sliding guide – jamming"),
         "v_groove": (v_groove, "V-groove"),
         "wedge": (wedge, "wedge – self-locking"),
         "rolling_bearing": (rolling_bearing, "rolling bearing"),
         "power_screw": (power_screw, "power screw"),
         "belt": (belt, "belt, rope, band brake"),
         "worm": (worm, "worm gear"),
         "clutch": (clutch, "friction clutch / brake"),
         "inclined_plane": (inclined_plane, "measuring μs on an inclined plane")}
