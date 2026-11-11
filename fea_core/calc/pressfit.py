"""Press fit of a shaft (solid / hollow) + hub – thick-walled cylinders (Lamé), plane stress, without friction during
the pressing and without the influence of the length of the hub (Shigley, Mechanical Engineering Design, ch. 3-16; Roark).

  p = δ / ( d·[ (1/E_n)·((D² + d²)/(D² − d²) + ν_n) + (1/E_h)·((d² + d_i²)/(d² − d_i²) − ν_h) ] )   δ = diametral interference
  hub at the bore: σ_t = p·(D² + d²)/(D² − d²), σ_r = −p; shaft at the surface: σ_t = −p·(d² + d_i²)/(d² − d_i²), σ_r = −p
  u_r,hub = (d/2)·p/E_n·((D² + d²)/(D² − d²) + ν_n),  u_r,shaft = −(d/2)·p/E_h·((d² + d_i²)/(d² − d_i²) − ν_h)
  T = μ·p·π·d²·L/2,  F = μ·p·π·d·L,  heating of the hub ΔT = (δ + s_m)/(α_n·d)
"""
from __future__ import annotations

import math

from .common import CalcError, positive, result

SOURCE = "Lamé (thick-walled cylinders); Shigley's Mechanical Engineering Design, ch. 3-16 (press and shrink fits)"


def _vm(s1: float, s2: float) -> float:
    return math.sqrt(s1 * s1 - s1 * s2 + s2 * s2)


def press_fit(d: float, D: float, L: float, delta: float, E_hub: float = 210000.0, nu_hub: float = 0.3,
              E_shaft: float | None = None, nu_shaft: float | None = None, d_inner: float = 0.0, mu: float = 0.15,
              mu_press: float | None = None, alpha_hub: float | None = None, mount_clearance: float = 0.0,
              T_ambient: float = 20.0, Re_hub: float | None = None, Re_shaft: float | None = None) -> dict:
    """Press fit. d nominal diameter of the joint [mm], D outer diameter of the hub, L length of the joint, delta diametral interference
    [mm], E/ν of the hub and the shaft (shaft None = as the hub), d_inner bore of the hollow shaft (0 = solid), mu coefficient of
    friction for the transmission (μ_min – slip of the joint), mu_press coefficient during the pressing (μ_max; None = mu), alpha_hub
    coefficient of expansion of the hub [1/K] for the mounting by heating, mount_clearance mounting clearance during the assembly by heating
    [mm], Re_hub / Re_shaft yield strength [MPa] (safety against yielding according to HMH, optional).
    """
    d, D, L = positive("d", d), positive("D", D), positive("L", L)
    delta = positive("delta", delta, zero=True)
    d_i = positive("d_inner", d_inner, zero=True)
    if D <= d:
        raise CalcError("Hub outer diameter D must be greater than the fit diameter d")
    if d_i >= d:
        raise CalcError("Shaft bore d_inner must be less than d")
    Eh, nh = positive("E_hub", E_hub), float(nu_hub)
    Es = Eh if E_shaft is None else positive("E_shaft", E_shaft)
    ns = nh if nu_shaft is None else float(nu_shaft)
    if not (0 <= nh < 0.5 and 0 <= ns < 0.5):
        raise CalcError("Poisson's ratio must be in [0; 0.5)")
    mu = positive("mu", mu, zero=True)
    mup = mu if mu_press is None else positive("mu_press", mu_press, zero=True)
    kh = (D * D + d * d) / (D * D - d * d)
    ks = (d * d + d_i * d_i) / (d * d - d_i * d_i)
    p = delta / (d * ((kh + nh) / Eh + (ks - ns) / Es))
    st_hub = p * kh
    st_hub_out = 2.0 * p * d * d / (D * D - d * d)
    st_sh = -p * ks
    st_sh_in = -2.0 * p * d * d / (d * d - d_i * d_i) if d_i > 0 else st_sh
    u_hub = d / 2.0 * p / Eh * (kh + nh)
    u_sh = -d / 2.0 * p / Es * (ks - ns)
    vm_hub = _vm(st_hub, -p)
    vm_sh = _vm(st_sh, -p)
    vm_sh_in = abs(st_sh_in) if d_i > 0 else vm_sh
    A = math.pi * d * L
    vals = {"p_MPa": p, "sigma_t_hub_bore_MPa": st_hub, "sigma_r_hub_bore_MPa": -p, "sigma_t_hub_outer_MPa": st_hub_out,
            "sigma_t_shaft_surface_MPa": st_sh, "sigma_r_shaft_surface_MPa": -p, "sigma_t_shaft_bore_MPa": st_sh_in,
            "vm_hub_bore_MPa": vm_hub, "vm_shaft_MPa": max(vm_sh, vm_sh_in),
            "u_r_hub_mm": u_hub, "u_r_shaft_mm": u_sh, "u_r_hub_um": 1e3 * u_hub, "u_r_shaft_um": 1e3 * u_sh,
            "dD_hub_outer_mm": st_hub_out * D / Eh,
            "T_Nmm": mu * p * A * d / 2.0, "T_Nm": mu * p * A * d / 2000.0, "F_axial_N": mu * p * A,
            "F_press_N": mup * p * A, "contact_area_mm2": A}
    warn = []
    if alpha_hub is not None:
        a = positive("alpha_hub", alpha_hub)
        s = positive("mount_clearance", mount_clearance, zero=True)
        dT = (delta + s) / (a * d)
        vals.update({"dT_heating_K": dT, "T_heating_C": float(T_ambient) + dT})
        if s == 0:
            warn.append("Assembly clearance 0 – for shrink fitting s ≈ 0.001·d is usually chosen")
    for name, key, Re, vm in (("Hub", "safety_hub", Re_hub, vm_hub),
                              ("Shaft", "safety_shaft", Re_shaft, vals["vm_shaft_MPa"])):
        if Re is not None:
            k = positive(f"Re ({name})", Re) / vm if vm > 0 else math.inf
            vals[key] = k
            if k < 1:
                warn.append(f"{name}: σ_HMH {vm:.4g} MPa > Re – plastic deformation, Lamé not valid")
    if L < 0.5 * d:
        warn.append("Short hub (L < 0.5·d) – plane stress is only approximate, edges have pressure peaks")
    return result(vals, "p = δ/(d·[(1/E_n)((D²+d²)/(D²−d²)+ν_n) + (1/E_h)((d²+d_i²)/(d²−d_i²)−ν_h)]); σ_t,hub = p(D²+d²)/(D²−d²); T = μpπd²L/2; F = μpπdL; ΔT = (δ+s)/(α·d)",
                  SOURCE, warn)


KINDS = {"press_fit": (press_fit, "shaft + hub press fit (Lamé)")}
