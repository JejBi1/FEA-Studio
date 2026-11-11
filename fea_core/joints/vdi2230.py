"""Bolted joint according to VDI 2230 part 1 (2015) – according to publicly available literature.

Simplifications (stated in the report):
  - compliance of the clamped parts by the cone model (DSV / ESV), several materials weighted by thickness,
  - operating forces from the FE (linear model with a connected joint = clamped joint): additional force in the bolt
    F_SA directly from the beam, the axial operating force F_A = F_SA + F_PA (relief of the joint near the bolt),
    transverse force F_Q from the joint and the beam, bending moment M_b from the beam → load ratio Φ = F_SA / F_A,
  - ΔF_Vth = 0 (no thermal changes), q_F = 1 (one friction surface per bolt and joint),
  - fatigue limit σ_ASV = 0.85 (150/d + 45) (rolled before heat treatment), N_D = 2·10⁶, k = 3,
    below the limit Haibach k₂ = 5 (variable amplitudes).
Units: N, mm, MPa.
"""
from __future__ import annotations

import math

from . import tables
from .model import BoltSpec

# settlement f_Z [µm] by the roughness Rz: (thread, each bearing face of the head/nut, each internal joint)
EMBEDDING = {"<10": (3.0, 2.5, 1.5), "10-40": (3.0, 3.0, 2.0), "40-160": (3.0, 4.0, 3.0)}
NU_TIGHTEN = 0.9              # utilisation of the yield strength during tightening
K_TAU = 0.5                   # reduction of the torsional stress in operation
N_D = 2e6
S_F_MIN, S_D_MIN, S_G_MIN_STATIC, S_G_MIN_DYN, S_P_MIN = 1.0, 1.2, 1.2, 1.8, 1.0


def limiting_pressure(Rm: float | None, E: float) -> tuple[float, str]:
    """Limiting pressure under the head p_G – indicative from Rm (steel S235: 490 MPa ≈ 1.44 Rm)."""
    if not Rm:
        return (490.0 if E > 150000 else 300.0), "estimate (Rm missing)"
    k = 1.4 if Rm <= 520 else (1.2 if Rm <= 800 else 0.9)
    if E < 150000:
        k = 1.2
    return k * Rm, f"estimate p_G ≈ {k:g}·Rm"


def basics(b: BoltSpec, E_S: float, plate_E: dict[str, float], E_M: float,
           Rp02: float, D_A: float | None = None, p_G: float | None = None) -> dict:
    """Compliances, settlement, assembly preload, tightening torque (independent of the load).

    p_G = limiting pressure under the head/nut: the automatic preload is the smaller of F_M,zul (bolt, ν = 0.9)
    and p_G·A_p,min (clamped part – decides for plastics and aluminium).
    """
    th = b.thread
    d, P = th.d, th.P
    head = tables.head_dims(d, b.head, b.standard)
    tapped = b.connection == "tapped"
    lK = b.grip
    # --- bolt δ_S (head, shank, free thread, screwed thread + nut/internal thread)
    l_shank = min(tables.shank_length(d, b.length, b.standard), lK)
    l_gew = max(lK - l_shank, 0.0)
    dS = {
        "head": head["lsk"] / (E_S * th.AN),
        "shank": l_shank / (E_S * th.AN),
        "free thread": l_gew / (E_S * th.Ad3),
        "engaged thread": 0.5 * d / (E_S * th.Ad3),
        "nut / internal thread": (0.33 * d if tapped else 0.4 * d) / (E_M * th.AN),
    }
    delta_S = sum(dS.values())
    # --- clamped parts δ_P (cone; DSV w = 1, ESV w = 2)
    d_h = b.hole or tables.CLEARANCE.get(tables.size_key(d), (1.1 * d,) * 3)[1]
    d_W = b.head_dw if tapped or not b.nut_dw else 0.5 * (b.head_dw + b.nut_dw)
    w = 2.0 if tapped else 1.0
    beta = lK / d_W
    DA = D_A if D_A and D_A > d_W else 1e6
    y = DA / d_W
    for _ in range(3):
        if tapped:
            tanphi = 0.348 + 0.013 * math.log(beta) + 0.193 * math.log(y)
        else:
            tanphi = 0.362 + 0.032 * math.log(beta / 2) + 0.153 * math.log(y)
        D_Gr = d_W + w * lK * tanphi
        if DA <= D_Gr * 1.0001:
            break
        y = D_Gr / d_W
    if DA >= D_Gr:
        num = (d_W + d_h) * (d_W + w * lK * tanphi - d_h)
        den = (d_W - d_h) * (d_W + w * lK * tanphi + d_h)
        dP1 = 2 * math.log(num / den) / (w * math.pi * d_h * tanphi)
        model = "full cone"
    elif DA > d_W:
        dP1 = (2 / (w * d_h * tanphi) * math.log((d_W + d_h) * (DA - d_h) / ((d_W - d_h) * (DA + d_h)))
               + 4 / (DA ** 2 - d_h ** 2) * (lK - (DA - d_W) / (w * tanphi))) / math.pi
        model = "cone + sleeve"
    else:
        dP1 = 4 * lK / (math.pi * (DA ** 2 - d_h ** 2))
        model = "pouzdro"
    # several materials: compliance E = 1 times the weighted mean of 1/E by thicknesses
    segs = [(pl["s1"] - pl["s0"], plate_E.get(pl["part"], 210000.0)) for pl in b.plates]
    segs += [(w_["h"], E_S) for w_ in b.washers]
    lsum = sum(t for t, _ in segs) or lK
    inv_E = sum(t / E for t, E in segs) / lsum
    delta_P = dP1 * inv_E
    phi_K = delta_P / (delta_S + delta_P)
    # --- settlement
    fz_thr, fz_bear, fz_if = EMBEDDING.get(b.roughness, EMBEDDING["10-40"])
    n_bear = 1 if tapped else 2
    n_if = max(len(b.plates) - 1, 0) + len(b.washers) + (1 if tapped else 0)
    f_Z = (fz_thr + n_bear * fz_bear + n_if * fz_if) * 1e-3
    F_Z = f_Z / (delta_S + delta_P)
    # --- assembly
    mu_G, mu_K = b.mu_G, b.mu_K
    d2, dS_ = th.d2, th.ds
    A0 = th.As
    k_thr = P / (math.pi * d2) + 1.155 * mu_G
    F_Mzul = A0 * NU_TIGHTEN * Rp02 / math.sqrt(1 + 3 * (1.5 * d2 / dS_ * k_thr) ** 2)
    D_Km = 0.5 * (b.head_dw + d_h)
    torque_k = 0.16 * P + 0.58 * d2 * mu_G + 0.5 * D_Km * mu_K          # M_A = F_M · torque_k [N·mm]
    dw_min = min(b.head_dw, b.nut_dw) if (b.connection == "nut" and b.nut_dw) else b.head_dw
    A_p = math.pi / 4 * (dw_min ** 2 - d_h ** 2)
    F_lim_p = p_G * A_p if p_G and A_p > 0 else math.inf
    mode = b.preload_mode
    limit = ""
    if mode == "force" and b.preload > 0:
        F_Mmax = b.preload
    elif mode == "torque" and b.preload > 0:
        F_Mmax = b.preload * 1000.0 / torque_k
    else:
        mode = "auto"                             # 90 % of the limit – reserve for the operating force (as ν = 0.9 for the bolt)
        F_Mmax = min(F_Mzul, NU_TIGHTEN * F_lim_p)
        limit = "pressure under head/nut (90 % p_G·A_p)" if NU_TIGHTEN * F_lim_p < F_Mzul else "bolt (90 % Rp0.2)"
    F_Mmin = F_Mmax / b.alpha_A
    M_A = F_Mmax * torque_k / 1000.0
    return {
        "d": d, "P": P, "d2": d2, "d3": th.d3, "dS": dS_, "As": th.As, "Ad3": th.Ad3, "AN": th.AN,
        "lK": lK, "l_shank": l_shank, "l_gew": l_gew, "dW": d_W, "dh": d_h, "DA": D_A, "w": w,
        "delta_S_parts": dS, "delta_S": delta_S, "delta_P": delta_P, "cone": model, "tanphi": tanphi,
        "phi_K": phi_K, "E_S": E_S, "E_P_eff": 1.0 / inv_E, "f_Z": f_Z, "F_Z": F_Z,
        "F_Mzul": F_Mzul, "F_Mmax": F_Mmax, "F_Mmin": F_Mmin, "F_Vmin": F_Mmin - F_Z,
        "F_Vnom": 0.5 * (F_Mmax + F_Mmin) - F_Z, "M_A": M_A, "preload_mode": mode, "Rp02": Rp02,
        "k_thr": k_thr, "torque_k": torque_k, "A_p": A_p, "p_G": p_G, "F_lim_p": F_lim_p, "preload_limit": limit,
        # stiffness of the equivalent beam of the FE: compliance of the bolt + washers in series between the surfaces of the clamped parts
        "k_bolt": 1.0 / delta_S,
    }


def assess(b: BoltSpec, bas: dict, w: dict, Rm_plate_min: float | None, E_plate: float,
           fatigue: dict | None = None) -> dict:
    """Assessment for the operating forces w = {F_A, F_SA, F_PA, F_Q, M_b} (maxima over the states, F_A > 0 = separation).

    Returns the steps of the calculation, the safeties and the total utilisation (max. ratio of the demand to the limit).
    """
    d2, dS, As = bas["d2"], bas["dS"], bas["As"]
    Rp = bas["Rp02"]
    F_A = max(w.get("F_A", 0.0), 0.0)
    F_SA = max(w.get("F_SA", 0.0), 0.0)
    F_PA = max(w.get("F_PA", 0.0), 0.0)
    F_Q = abs(w.get("F_Q", 0.0))
    M_b = abs(w.get("M_b", 0.0))
    phi = F_SA / F_A if F_A > 1e-9 else bas["phi_K"] * 0.5
    # R8 – working stress
    F_Smax = bas["F_Mmax"] + F_SA
    sig_z = F_Smax / As
    M_G = bas["F_Mmax"] * d2 / 2 * bas["k_thr"]
    tau = M_G / (math.pi * dS ** 3 / 16)
    sig_red = math.sqrt(sig_z ** 2 + 3 * (K_TAU * tau) ** 2)
    sig_b = M_b / (math.pi * dS ** 3 / 32)
    S_F = Rp / sig_red
    # R7 – assembly
    util_M = bas["F_Mmax"] / bas["F_Mzul"]
    # R12 – residual clamping force, opening, slip
    F_KR = bas["F_Mmin"] - bas["F_Z"] - F_PA
    F_Kerf = F_Q / b.mu_T if b.mu_T > 0 else 0.0
    S_G = F_KR * b.mu_T / F_Q if F_Q > 1e-9 else math.inf
    tau_Q = F_Q / As
    S_A = 0.6 * Rp / tau_Q if F_Q > 1e-9 else math.inf     # shear of the shank after slip (τ_B ≈ 0.6 Rm ~ Rp)
    # R10 – bearing pressure under the head / nut
    A_p = bas["A_p"]
    p_max = (bas["F_Mmax"] + F_SA) / A_p if A_p > 0 else math.inf
    p_G, p_src = limiting_pressure(Rm_plate_min, E_plate)
    if bas.get("p_G"):
        p_G, p_src = bas["p_G"], "part under head/nut"
    S_P = p_G / p_max
    # R9 – fatigue (from the load curve, if there is one)
    sig_ASV = 0.85 * (150.0 / bas["d"] + 45.0)
    out_fat = None
    if fatigue:
        sa = fatigue["sigma_a"]
        out_fat = dict(fatigue)
        out_fat["sigma_ASV"] = sig_ASV
        out_fat["S_D"] = sig_ASV / sa if sa > 1e-12 else math.inf
    checks = [
        ("Assembly preload F_M,max ≤ F_M,zul", util_M,
         f"{bas['F_Mmax']:.0f} / {bas['F_Mzul']:.0f} N, M_A = {bas['M_A']:.2f} N·m"
         + (f" (limited by {bas['preload_limit']})" if bas.get("preload_limit") else "")),
        ("Working stress σ_red,B ≤ R_p0.2 (S_F ≥ 1.0)", S_F_MIN / S_F, f"S_F = {S_F:.2f}"),
        ("Residual clamp force F_KR > 0 (joint does not open)",
         1.0 if F_KR <= 0 else (F_PA + bas["F_Z"]) / (bas["F_Mmin"]), f"F_KR,min = {F_KR:.0f} N"),
        ("Head bearing pressure p ≤ p_G", 1.0 / S_P if S_P > 0 else math.inf, f"p = {p_max:.0f} / {p_G:.0f} MPa ({p_src})"),
    ]
    if F_Q > 1e-9:
        checks.append(("Slip: F_KR,min·μ_T ≥ S_G·F_Q (S_G ≥ 1.2)",
                       math.inf if F_KR <= 0 else S_G_MIN_STATIC / S_G, f"S_G = {S_G:.2f}"))
    if out_fat:
        checks.append(("Fatigue: σ_a ≤ σ_ASV / S_D (S_D ≥ 1.2)", S_D_MIN / out_fat["S_D"] if out_fat["S_D"] > 0 else math.inf,
                       f"σ_a = {out_fat['sigma_a']:.1f} MPa, σ_ASV = {sig_ASV:.1f} MPa, D = {out_fat.get('damage', 0):.3g}"))
    util = max(c[1] for c in checks)
    return {
        "F_A": F_A, "F_SA": F_SA, "F_PA": F_PA, "F_Q": F_Q, "M_b": M_b, "phi": phi,
        "F_Smax": F_Smax, "sigma_z": sig_z, "tau": tau, "sigma_red": sig_red, "sigma_b": sig_b, "S_F": S_F,
        "F_KR": F_KR, "F_Kerf": F_Kerf, "S_G": S_G, "tau_Q": tau_Q, "S_A": S_A, "p_max": p_max, "p_G": p_G,
        "S_P": S_P, "sigma_ASV": sig_ASV, "fatigue": out_fat, "checks": checks, "utilization": util,
    }


TAN30 = math.tan(math.radians(30.0))


def tau_ratio_bolt(strength: str) -> float:
    """τ_B / R_m of the bolt (literature on VDI 2230): 12.9 0.60, 10.9 0.62, 8.8 0.65, lower classes 0.70, stainless 0.80."""
    s = strength.upper()
    if s.startswith("A"):
        return 0.80
    return {"12.9": 0.60, "10.9": 0.62, "9.8": 0.63, "8.8": 0.65}.get(s, 0.70)


def tau_ratio_part(E: float) -> tuple[float, str]:
    """τ_B / R_m of the material of the internal thread by its kind (from E): steel 0.65, Al 0.65, Ti/Cu/cast iron 0.60, plastic 0.60."""
    if E >= 150000:
        return 0.65, "ocel"
    if E >= 60000:
        return (0.65, "aluminum") if E < 90000 else (0.60, "Ti / Cu / cast iron")
    return 0.60, "plastic – approximate only"


def stripping(d: float, P: float, m_eff: float, Rm_S: float, tau_S: float, Rm_M: float, tau_M: float,
              s_nut: float | None, A_S: float) -> dict:
    """Stripping of the thread according to VDI 2230 (Alexander 1977, nominal dimensions of the thread).

    Shear areas per unit length of the screwed-in length (D2 = d2, D1 = d − 1.0825 P):
      A_SGM = π d (m/P) [P/2 + (d − D2) tan 30°]  = 0.875 π d m   (internal thread, shear at d)
      A_SGS = π D1 (m/P) [P/2 + (d2 − D1) tan 30°] = 0.75 π D1 m  (thread of the bolt, shear at D1)
    Ratio of strengths R_S = τ_BM A_SGM / (τ_BS A_SGS); C1 = widening of the nut (s/d), for a hole with a thread 1;
    C2 (bending of the thread of the bolt) = 0.897 for R_S ≤ 1, otherwise 5.594 − 13.682 R_S + 14.107 R_S² − 6.057 R_S³ + 0.9353 R_S⁴;
    C3 (bending of the internal thread) = 0.897 for R_S ≥ 1, otherwise 0.728 + 1.769 R_S − 2.896 R_S² + 1.296 R_S³ (R_S ≥ 0.4).
    F_mGM = τ_BM A_SGM C1 C3, F_mGS = τ_BS A_SGS C1 C2; the necessary length so that the bolt breaks (F_mS = R_m,S A_S)
    before the thread strips: m_eff,min = F_mS / min(F_mGM, F_mGS)/m, m_ges,min = m_eff,min + 0.8 P (chamfer).
    """
    th = tables.Thread(d, P)
    D1, D2 = th.d1, th.d2
    aM = math.pi * d / P * (P / 2 + (d - D2) * TAN30)
    aS = math.pi * D1 / P * (P / 2 + (th.d2 - D1) * TAN30)
    tBS, tBM = tau_S * Rm_S, tau_M * Rm_M
    R_S = tBM * aM / (tBS * aS)
    note = ""
    if s_nut:
        x = s_nut / d
        if x < 1.4:
            note = f"s/d = {x:.2f} < 1.4 – C1 evaluated for 1.4"
            x = 1.4
        C1 = 1.0 if x >= 1.9 else -x ** 2 + 3.8 * x - 2.61
    else:
        C1 = 1.0
    r = min(max(R_S, 0.4), 2.2)
    C2 = 0.897 if r <= 1.0 else 5.594 - 13.682 * r + 14.107 * r ** 2 - 6.057 * r ** 3 + 0.9353 * r ** 4
    C3 = 0.897 if r >= 1.0 else 0.728 + 1.769 * r - 2.896 * r ** 2 + 1.296 * r ** 3
    fM, fS = tBM * aM * C1 * C3, tBS * aS * C1 * C2              # force per 1 mm of screwing
    F_mS = Rm_S * A_S
    m_min = F_mS / min(fM, fS)
    return {"A_SGM": aM * m_eff, "A_SGS": aS * m_eff, "tau_BM": tBM, "tau_BS": tBS, "R_S": R_S, "C1": C1,
            "C2": C2, "C3": C3, "F_mGM": fM * m_eff, "F_mGS": fS * m_eff, "F_mS": F_mS, "m_eff": m_eff,
            "m_eff_min": m_min, "m_ges_min": m_min + 0.8 * P, "governs": "internal thread" if fM <= fS else "bolt thread",
            "note": note}


def fatigue_damage(sigma_a: list[float], counts: list[float], d: float, k: float = 3.0, k2: float = 5.0) -> dict:
    """Damage of the bolt from the stress amplitudes (rainflow) – VDI Wöhler curve (σ_ASV at N_D)."""
    sASV = 0.85 * (150.0 / d + 45.0)
    D = 0.0
    for sa, n in zip(sigma_a, counts):
        if sa <= 0:
            continue
        kk = k if sa >= sASV else k2
        D += n / (N_D * (sASV / sa) ** kk)
    return {"damage": D, "sigma_a": max(sigma_a) if sigma_a else 0.0, "sigma_ASV": sASV}
