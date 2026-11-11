"""Hertzian contact (Johnson, Contact Mechanics 1985, ch. 4 and 11): sphere – sphere / plane, cylinder – cylinder along the length,
gearing (ISO 6336-2 σ_H0 at the rolling point), eccentric cam with a roller, impact of a sphere (quasi-static).

  1/E* = (1 − ν₁²)/E₁ + (1 − ν₂²)/E₂,  1/R = 1/R₁ + 1/R₂ (plane R = ∞, concave surface R < 0)
  sphere: a = (3FR/(4E*))^⅓, p₀ = 3F/(2πa²), δ = a²/R
  cylinder: b = √(4F'R/(πE*)), p₀ = 2F'/(πb)  (F' = force per length)
  gearing: σ_H0 = Z_H·Z_E·Z_ε·Z_β·√(F_t/(d₁b)·(u + 1)/u), Z_E = √(E*/π), Z_H = √(2/(cos²α·tan α)) (without profile shift, β = 0)
  impact: δ_max = (15mv²/(16E*√R))^⅖, F_max = (4/3)E*√R·δ_max^{3/2}, t_c = 2.94·δ_max/v
"""
from __future__ import annotations

import math

import numpy as np

from .common import CalcError, positive, result

SOURCE = "K. L. Johnson: Contact Mechanics (1985), ch. 4 (Hertz), ch. 11 (impact)"
SOURCE_GEAR = "ISO 6336-2 (σ_H0 = Z_H·Z_E·Z_ε·Z_β·√(F_t/(d₁b)·(u+1)/u)); Hertz of cylinders at the pitch point (Johnson ch. 4)"


def e_star(E1: float, nu1: float, E2: float | None, nu2: float | None) -> float:
    """Combined modulus E*; E2 None = rigid body."""
    c = (1.0 - nu1 ** 2) / positive("E1", E1)
    if E2 is not None:
        c += (1.0 - (nu1 if nu2 is None else nu2) ** 2) / positive("E2", E2)
    return 1.0 / c


def r_eff(R1: float, R2: float | None) -> float:
    """Reduced radius; R2 None or 0 = plane, negative = concave surface."""
    k = 1.0 / positive("R1", R1)
    if R2:
        k += 1.0 / float(R2)
    if k <= 0:
        raise CalcError("Concave surface has a smaller radius than the convex one – contact is not Hertzian (conformal)")
    return 1.0 / k


def sphere_subsurface(nu: float) -> tuple[float, float]:
    """(τ_max/p₀, z/a) below the centre of a circular area (Johnson eq. 3.45)."""
    z = np.linspace(1e-4, 3.0, 30001)
    sz = -1.0 / (1.0 + z * z)
    sr = -(1.0 + nu) * (1.0 - z * np.arctan(1.0 / z)) + 0.5 / (1.0 + z * z)
    t = 0.5 * np.abs(sr - sz)
    k = int(np.argmax(t))
    return float(t[k]), float(z[k])


def cylinder_subsurface() -> tuple[float, float]:
    """(τ_max/p₀, z/b) below the centre of a strip (plane strain, does not depend on ν)."""
    z = np.linspace(1e-4, 3.0, 30001)
    s = np.sqrt(1.0 + z * z)
    sx = -((1.0 + 2.0 * z * z) / s - 2.0 * z)
    sz = -1.0 / s
    t = 0.5 * np.abs(sx - sz)
    k = int(np.argmax(t))
    return float(t[k]), float(z[k])


def sphere(F: float, R1: float, R2: float | None = None, E1: float = 210000.0, nu1: float = 0.3,
           E2: float | None = 210000.0, nu2: float | None = 0.3) -> dict:
    """Sphere R1 on sphere R2 (None = plane, negative = concave). F force [N], E [MPa]; E2 None = rigid second body."""
    F = positive("F", F)
    Es = e_star(E1, nu1, E2, nu2)
    R = r_eff(R1, R2)
    a = (3.0 * F * R / (4.0 * Es)) ** (1.0 / 3.0)
    p0 = 3.0 * F / (2.0 * math.pi * a * a)
    d = a * a / R
    tau1, z1 = sphere_subsurface(nu1)
    vals = {"E_star_MPa": Es, "R_eff_mm": R, "a_mm": a, "p0_MPa": p0, "p_mean_MPa": 2.0 * p0 / 3.0,
            "delta_mm": d, "delta_um": 1e3 * d, "tau_max_MPa": tau1 * p0, "tau_max_ratio": tau1,
            "z_tau_max_mm": z1 * a, "z_tau_max_ratio": z1, "k_contact_N_mm": 1.5 * F / d}
    if E2 is not None and nu2 is not None and abs(nu2 - nu1) > 1e-12:
        tau2, z2 = sphere_subsurface(nu2)
        vals.update({"tau_max_body2_MPa": tau2 * p0, "z_tau_max_body2_mm": z2 * a})
    return result(vals, "a = (3FR/(4E*))^⅓; p₀ = 3F/(2πa²); δ = a²/R; τ_max below the surface (ν of body 1)",
                  SOURCE)


def cylinder(F_per_length: float, R1: float, R2: float | None = None, E1: float = 210000.0, nu1: float = 0.3,
             E2: float | None = 210000.0, nu2: float | None = 0.3) -> dict:
    """Cylinder R1 on cylinder R2 (None = plane) with the contact along the length; F_per_length = force per length [N/mm]."""
    Fp = positive("F_per_length", F_per_length)
    Es = e_star(E1, nu1, E2, nu2)
    R = r_eff(R1, R2)
    b = math.sqrt(4.0 * Fp * R / (math.pi * Es))
    p0 = 2.0 * Fp / (math.pi * b)
    tau, z = cylinder_subsurface()
    vals = {"E_star_MPa": Es, "R_eff_mm": R, "b_mm": b, "width_mm": 2.0 * b, "p0_MPa": p0,
            "p_mean_MPa": math.pi * p0 / 4.0, "tau_max_MPa": tau * p0, "z_tau_max_mm": z * b}
    return result(vals, "b = √(4F'R/(πE*)) (half-width of the strip); p₀ = 2F'/(πb); τ_max ≈ 0.30·p₀ at 0.79·b",
                  SOURCE)


def cam_roller(N: float, b: float, R_cam: float, r_roller: float, E1: float = 210000.0, nu1: float = 0.3,
               E2: float | None = 210000.0, nu2: float | None = 0.3) -> dict:
    """Circular (eccentric) cam of radius R_cam with a roller r_roller, normal force N [N], width of the contact b [mm]."""
    r = cylinder(positive("N", N) / positive("b", b), R_cam, r_roller, E1, nu1, E2, nu2)
    r["rho_mm"] = r["R_eff_mm"]
    r["formula"] = "ρ = R_c·r_f/(R_c + r_f); p₀ = √(N/b·E*/(πρ)); " + r["formula"]
    return r


def gear(z1: int, z2: int, m: float, b: float, T1: float, alpha_deg: float = 20.0, E1: float = 210000.0,
         nu1: float = 0.3, E2: float | None = 210000.0, nu2: float | None = 0.3, Z_eps: float = 1.0,
         Z_beta: float = 1.0, K: float = 1.0, ha: float = 1.0) -> dict:
    """External spur gear with straight teeth without profile shift: z1, z2 numbers of teeth, m module [mm], b width [mm], T1 torque of
    the pinion [N·mm], α pressure angle, Z_eps, Z_beta (default 1 as Hertz), K = K_A·K_v·K_Hβ·K_Hα (σ_H = σ_H0·√K),
    ha height factor of the head. Returns σ_H0 (ISO 6336-2), Hertz at the rolling point, the contact ratio ε_α and whether the rolling
    point is in single-pair contact.
    """
    z1, z2 = int(z1), int(z2)
    if z1 < 5 or z2 < 5:
        raise CalcError("Tooth numbers z1, z2 must be ≥ 5")
    m, b, T1 = positive("m", m), positive("b", b), positive("T1", T1)
    if not 10 <= alpha_deg <= 35:
        raise CalcError("alpha_deg must be 10–35°")
    al = math.radians(alpha_deg)
    r1, r2 = z1 * m / 2.0, z2 * m / 2.0
    rb1, rb2 = r1 * math.cos(al), r2 * math.cos(al)
    ra1, ra2 = r1 + ha * m, r2 + ha * m
    pb = math.pi * m * math.cos(al)
    g_a1 = math.sqrt(ra1 ** 2 - rb1 ** 2) - r1 * math.sin(al)      # from the rolling point to the head of the pinion
    g_a2 = math.sqrt(ra2 ** 2 - rb2 ** 2) - r2 * math.sin(al)      # from the rolling point to the head of the gear
    eps_a = (g_a1 + g_a2) / pb
    rho1, rho2 = r1 * math.sin(al), r2 * math.sin(al)
    rho = rho1 * rho2 / (rho1 + rho2)
    Fn = T1 / rb1
    Ft = T1 / r1
    Es = e_star(E1, nu1, E2, nu2)
    bh = math.sqrt(4.0 * Fn / b * rho / (math.pi * Es))
    p0 = 2.0 * Fn / b / (math.pi * bh)
    ZE = math.sqrt(Es / math.pi)
    ZH = math.sqrt(2.0 / (math.cos(al) ** 2 * math.tan(al)))
    u = z2 / z1
    sH0 = ZH * ZE * float(Z_eps) * float(Z_beta) * math.sqrt(Ft / (2.0 * r1 * b) * (u + 1.0) / u)
    single = g_a1 <= pb and g_a2 <= pb
    warn = []
    if eps_a < 1.0:
        warn.append("Contact ratio ε_α < 1 – the gear pair does not mesh continuously")
    if not single:
        warn.append("Pitch point lies in double-tooth-pair contact – σ_H0 is an upper estimate (force is shared by two pairs)")
    vals = {"d1_mm": 2 * r1, "d2_mm": 2 * r2, "u": u, "F_t_N": Ft, "F_n_N": Fn, "rho1_mm": rho1, "rho2_mm": rho2,
            "rho_red_mm": rho, "b_hertz_mm": bh, "p0_MPa": p0, "Z_E_sqrtMPa": ZE, "Z_H": ZH,
            "sigma_H0_MPa": sH0, "sigma_H_MPa": sH0 * math.sqrt(positive("K", K)), "eps_alpha": eps_a,
            "p_b_mm": pb, "single_pair_at_pitch": bool(single)}
    return result(vals, "ρ_red = ρ₁ρ₂/(ρ₁+ρ₂), ρ_i = r_i·sin α; F_n = T₁/r_b1; p₀ = √(F_n/b·E*/(πρ_red)); "
                        "σ_H0 = Z_H·Z_E·Z_ε·Z_β·√(F_t/(d₁b)·(u+1)/u); ε_α = (g_a1 + g_a2)/p_b",
                  SOURCE_GEAR, warn)


def impact(R: float, v0: float, rho: float = 7.85e-9, E1: float = 210000.0, nu1: float = 0.3,
           E2: float | None = None, nu2: float | None = None, R2: float | None = None,
           m: float | None = None) -> dict:
    """Elastic impact of a sphere R [mm] with the velocity v0 [mm/s] on a wall (E2 None = rigid, R2 None = plane), mass m [t]
    (None = a solid sphere of density rho [t/mm³]). Quasi-static Hertz (Johnson ch. 11): valid if the time of passage of the
    wave 2R/c ≪ t_c.
    """
    v = positive("v0", v0)
    Es = e_star(E1, nu1, E2, nu2)
    Re = r_eff(R, R2)
    mass = positive("m", m) if m is not None else positive("rho", rho) * 4.0 / 3.0 * math.pi * positive("R", R) ** 3
    dmax = (15.0 * mass * v * v / (16.0 * Es * math.sqrt(Re))) ** 0.4
    Fmax = 4.0 / 3.0 * Es * math.sqrt(Re) * dmax ** 1.5
    tc = 2.94 * dmax / v
    a = math.sqrt(Re * dmax)
    c0 = math.sqrt(E1 / positive("rho", rho))          # speed of the longitudinal wave in the rod (rough estimate)
    vals = {"m_t": mass, "E_star_MPa": Es, "delta_max_mm": dmax, "delta_max_um": 1e3 * dmax, "F_max_N": Fmax,
            "t_c_s": tc, "t_c_us": 1e6 * tc, "a_max_mm": a, "p0_max_MPa": 3.0 * Fmax / (2.0 * math.pi * a * a),
            "wave_transit_us": 1e6 * 2.0 * R / c0}
    warn = []
    if 2.0 * R / c0 > 0.1 * tc:
        warn.append("Wave transit time is not << t_c – quasi-static Hertz is only approximate")
    return result(vals, "δ_max = (15mv²/(16E*√R))^⅖; F_max = (4/3)E*√R·δ_max^{3/2}; t_c = 2.94·δ_max/v",
                  SOURCE, warn)


KINDS = {"sphere": (sphere, "sphere – sphere / plane"),
         "cylinder": (cylinder, "cylinder – cylinder / plane (force per length)"),
         "gear": (gear, "gear mesh – σ_H0 per ISO 6336-2 at the pitch point"),
         "cam_roller": (cam_roller, "eccentric (circular) cam with roller follower"),
         "impact": (impact, "sphere impact on a wall (Hertz, quasi-static)")}
