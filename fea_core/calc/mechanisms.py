"""Analytical benchmarks of mechanisms (specification F8, tasks MB1–MB6, KK1, Y10): crank mechanism, four-bar linkage
(Freudenstein), eccentric cam with a roller, pendulum (elliptic integral), run-out with friction, stick-slip, rolling of a cylinder,
reduced inertia and run-up, residual vibration after a pulse of acceleration and the S-curve of the profile of the axis (limited jerk).

Units mm, N, t, s, rad; g = 9810 mm/s². The angles at the input in degrees (…_deg), angular velocities in rad/s.
"""
from __future__ import annotations

import math

import numpy as np

from .common import G, CalcError, choice, positive, result


# ------------------------------------------------------------------ crank mechanism
def slider_crank(r: float, l: float, phi_deg: float, omega: float = 0.0, alpha: float = 0.0, e: float = 0.0,
                 F: float = 0.0) -> dict:
    """Crank mechanism: crank r, connecting rod l, crank angle φ from the dead centre (φ = 0 → piston furthest, x = r + l), ω, ε of the crank
    [rad/s, rad/s²], offset e (axis of the piston shifted by e). F = force on the piston against the direction +x (towards the axis of the crank) [N] →
    the required torque on the crank M = F·v/ω (quasi-static, without inertia and friction).
    """
    r, l = positive("r", r), positive("l", l)
    ph = math.radians(phi_deg)
    s = r * math.sin(ph) - e
    if abs(s) >= l:
        raise CalcError("Connecting rod is shorter than the crank offset – the mechanism cannot be assembled in this position")
    w = math.sqrt(l * l - s * s)
    s1, s2 = r * math.cos(ph), -r * math.sin(ph)
    x = r * math.cos(ph) + w
    x1 = -r * math.sin(ph) - s * s1 / w                          # dx/dφ
    x2 = -r * math.cos(ph) - (s1 * s1 + s * s2) / w - s * s * s1 * s1 / w ** 3
    v = x1 * omega
    a = x2 * omega ** 2 + x1 * alpha
    beta = math.asin(s / l)
    om_rod = s1 * omega / (l * math.cos(beta))                  # dβ/dt, β = asin((r sin φ − e)/l)
    lam = r / l
    vals = {"x_mm": x, "v_mm_s": v, "a_mm_s2": a, "dx_dphi_mm": x1, "beta_rod_deg": math.degrees(beta),
            "omega_rod_rad_s": om_rod, "a_approx_mm_s2": -r * omega ** 2 * (math.cos(ph) + lam * math.cos(2 * ph)),
            "stroke_mm": math.sqrt((l + r) ** 2 - e * e) - math.sqrt((l - r) ** 2 - e * e) if l > r + abs(e) else None}
    if F and omega:
        vals["M_Nmm"] = F * v / omega
    elif F:
        vals["M_Nmm"] = F * x1
    return result(vals, "x = r cos φ + √(l² − (r sin φ − e)²); v = ẋ, a = ẍ (exact); approximately a ≈ −rω²(cos φ + λ cos 2φ); M = F·v/ω",
                  "slider-crank kinematics (e.g. Norton, Design of Machinery, ch. 4 and 7)")


# ------------------------------------------------------------------ four-bar linkage
def four_bar(a: float, b: float, c: float, d: float, theta2_deg: float, omega2: float = 0.0) -> dict:
    """Four-bar linkage: a crank (input), b coupler, c rocker (output), d frame; θ₂ angle of the crank from the frame. Freudenstein –
    both branches (open / crossed), angles θ₃, θ₄, transmission angle, Grashof condition, ω₃, ω₄ for ω₂.
    """
    a, b, c, d = (positive(n, v) for n, v in (("a", a), ("b", b), ("c", c), ("d", d)))
    t2 = math.radians(theta2_deg)
    K1, K2, K3 = d / a, d / c, (a * a - b * b + c * c + d * d) / (2.0 * a * c)
    K4, K5 = d / b, (c * c - d * d - a * a - b * b) / (2.0 * a * b)
    A = math.cos(t2) - K1 - K2 * math.cos(t2) + K3
    B = -2.0 * math.sin(t2)
    C = K1 - (K2 + 1.0) * math.cos(t2) + K3
    D = math.cos(t2) - K1 + K4 * math.cos(t2) + K5
    E = -2.0 * math.sin(t2)
    Fq = K1 + (K4 - 1.0) * math.cos(t2) + K5
    disc4, disc3 = B * B - 4 * A * C, E * E - 4 * D * Fq
    if disc4 < -1e-12 or disc3 < -1e-12:
        raise CalcError("Four-bar linkage cannot be assembled at this crank position (outside range of motion)")
    r4, r3 = math.sqrt(max(disc4, 0.0)), math.sqrt(max(disc3, 0.0))
    out = {}
    for key, sgn in (("open", -1.0), ("crossed", 1.0)):
        t4 = 2.0 * math.atan2(-B + sgn * r4, 2.0 * A)
        t3 = 2.0 * math.atan2(-E + sgn * r3, 2.0 * D)
        mu_t = abs(math.degrees(t4 - t3)) % 360.0
        mu_t = 360.0 - mu_t if mu_t > 180 else mu_t
        mu_t = 180.0 - mu_t if mu_t > 90 else mu_t
        br = {"theta3_deg": math.degrees(t3) % 360.0, "theta4_deg": math.degrees(t4) % 360.0,
              "transmission_angle_deg": mu_t}
        if omega2:
            br["omega3_rad_s"] = a * omega2 / b * math.sin(t4 - t2) / math.sin(t3 - t4)
            br["omega4_rad_s"] = a * omega2 / c * math.sin(t2 - t3) / math.sin(t4 - t3)
        out[key] = br
    ls = sorted((a, b, c, d))
    grashof = ls[0] + ls[3] <= ls[1] + ls[2]
    if not grashof:
        kind = "non-Grashof (no link fully rotates – triple rocker)"
    elif a == ls[0]:
        kind = "crank-rocker (crank a fully rotates)"
    elif d == ls[0]:
        kind = "double crank (ground link is shortest)"
    elif b == ls[0]:
        kind = "double rocker (coupler fully rotates)"
    else:
        kind = "rocker-crank (output c fully rotates)"
    vals = {"theta4_open_deg": out["open"]["theta4_deg"], "theta4_crossed_deg": out["crossed"]["theta4_deg"],
            "theta3_open_deg": out["open"]["theta3_deg"], "theta3_crossed_deg": out["crossed"]["theta3_deg"],
            "transmission_angle_open_deg": out["open"]["transmission_angle_deg"],
            "grashof": bool(grashof), "grashof_type": kind, "branches": out}
    warn = []
    if out["open"]["transmission_angle_deg"] < 40:
        warn.append("Transmission angle < 40° (open branch) – poor force transmission")
    return result(vals, "Freudenstein: K₁ = d/a, K₂ = d/c, K₃ = (a² − b² + c² + d²)/(2ac); θ₄ = 2·atan2(−B -/+ √(B² − 4AC), 2A) (− open, + crossed); Grashof s + l ≤ p + q",
                  "R. L. Norton: Design of Machinery, ch. 4 (Freudenstein equations)", warn)


# ------------------------------------------------------------------ eccentric cam with a roller
def eccentric_cam(e: float, R_cam: float, r_roller: float, theta_deg: float, omega: float = 0.0,
                  k: float = 0.0, F0: float = 0.0, b: float | None = None, E: float = 210000.0,
                  nu: float = 0.3) -> dict:
    """Eccentric (circular) cam: eccentricity e, radius of the cam R_cam, roller r_roller, a straight-line follower in the axis of
    rotation; θ = 0 → the largest lift. Lift s from the lowest position = crank mechanism (crank e, connecting rod R_c + r_f).
    Spring k [N/mm] with the preload F0 → normal force N = (F0 + k·s)/cos φ; with the width b → Hertz pressure cam – roller
    (steel E, ν).
    """
    e, Rc, rf = positive("e", e), positive("R_cam", R_cam), positive("r_roller", r_roller)
    L = Rc + rf
    if e >= L:
        raise CalcError("Eccentricity must be less than R_cam + r_roller")
    sc = slider_crank(e, L, theta_deg, omega)
    s = sc["x_mm"] - (L - e)
    phi = math.asin(e * math.sin(math.radians(theta_deg)) / L)
    vals = {"s_mm": s, "lift_mm": 2.0 * e, "pressure_angle_deg": math.degrees(phi),
            "pressure_angle_max_deg": math.degrees(math.asin(e / L)), "v_mm_s": sc["v_mm_s"], "a_mm_s2": sc["a_mm_s2"]}
    warn = []
    if k or F0:
        N = (positive("F0", F0, zero=True) + positive("k", k, zero=True) * s) / math.cos(phi)
        vals["N_N"] = N
        warn.append("Normal force without follower inertia (add m·a)")
        if b is not None:
            from .hertz import cam_roller
            h = cam_roller(N, b, Rc, rf, E, nu, E, nu)
            vals.update({"rho_mm": h["rho_mm"], "p0_MPa": h["p0_MPa"], "b_hertz_mm": h["b_mm"]})
    return result(vals, "s = e cos θ + √((R_c + r_f)² − e² sin² θ) − (R_c + r_f − e); pressure angle φ = asin(e sin θ/(R_c + r_f)); N = (F₀ + k·s)/cos φ; ρ = R_c·r_f/(R_c + r_f)",
                  "eccentric cam kinematics = slider-crank (Norton ch. 8)", warn)


# ------------------------------------------------------------------ kyvadlo
def ellipk(m: float) -> float:
    """Complete elliptic integral of the first kind K(m), m = k² (arithmetic-geometric mean)."""
    if not 0 <= m < 1:
        raise CalcError("Elliptic integral: m must be in [0; 1)")
    a, b = 1.0, math.sqrt(1.0 - m)
    for _ in range(60):
        a, b = 0.5 * (a + b), math.sqrt(a * b)
        if abs(a - b) < 1e-16 * a:
            break
    return math.pi / (2.0 * a)


def pendulum(L: float, theta0_deg: float, m: float = 0.0, g: float = G) -> dict:
    """Mathematical pendulum of length L [mm] with the amplitude θ₀: exact period T = 4√(L/g)·K(sin²(θ₀/2)), small oscillations
    2π√(L/g); force in the suspension at the bottom N = m·g·(3 − 2 cos θ₀) [N] for the mass m [t], the speed at the bottom.
    """
    L = positive("L", L)
    if not 0 < theta0_deg < 180:
        raise CalcError("theta0_deg must be in (0; 180)°")
    t0 = math.radians(theta0_deg)
    T = 4.0 * math.sqrt(L / g) * ellipk(math.sin(t0 / 2.0) ** 2)
    vals = {"T_s": T, "T_small_s": 2.0 * math.pi * math.sqrt(L / g), "ratio": T / (2.0 * math.pi * math.sqrt(L / g)),
            "v_bottom_mm_s": math.sqrt(2.0 * g * L * (1.0 - math.cos(t0))),
            "N_bottom_over_mg": 3.0 - 2.0 * math.cos(t0)}
    if m:
        vals["N_bottom_N"] = positive("m", m) * g * (3.0 - 2.0 * math.cos(t0))
    return result(vals, "T = 4√(L/g)·K(k), k = sin(θ₀/2); N_dole = mg(3 − 2 cos θ₀)",
                  "classical mechanics (elliptic integral of the 1st kind)")


# ------------------------------------------------------------------ run-out with friction
def sliding_stop(v0: float, mu: float, incline_deg: float = 0.0, g: float = G) -> dict:
    """Run-out of a block with the velocity v0 [mm/s] on a plane with friction μ (slope downwards in the direction of motion incline_deg > 0)."""
    v0, mu = positive("v0", v0, zero=True), positive("mu", mu, zero=True)
    a_ = math.radians(incline_deg)
    dec = g * (mu * math.cos(a_) - math.sin(a_))
    if dec <= 0:
        return result({"decel_mm_s2": dec, "stops": False}, "a = g(μ cos α − sin α) ≤ 0 → the body does not stop",
                      "kinematics of uniformly decelerated motion")
    return result({"decel_mm_s2": dec, "s_mm": v0 * v0 / (2.0 * dec), "t_s": v0 / dec, "stops": True,
                   "stays": bool(mu >= math.tan(a_))},
                  "a = g(μ cos α − sin α); s = v₀²/(2a); t = v₀/a; then remains at rest if μs ≥ tan α",
                  "kinematics of uniformly decelerated motion")


# ------------------------------------------------------------------ stick-slip
def stick_slip(m: float, k: float, v_belt: float, mu_s: float, mu_k: float, g: float = G) -> dict:
    """A body of mass m [t] on a belt of speed v_belt [mm/s], tied by a spring k [N/mm] to a fixed point,
    μs > μk (Coulomb). Steady cycle: sticking up to x_s = μs·m·g/k, slip around x_k = μk·m·g/k.
    """
    m, k, vb = positive("m", m), positive("k", k), positive("v_belt", v_belt)
    mus, muk = positive("mu_s", mu_s, zero=True), positive("mu_k", mu_k, zero=True)
    if mus < muk:
        raise CalcError("μs ≥ μk is required")
    N = m * g
    xs, xk = mus * N / k, muk * N / k
    w = math.sqrt(k / m)
    De = xs - xk
    t_stick = 2.0 * De / vb
    t_slip = (2.0 * math.pi - 2.0 * math.atan(w * De / vb)) / w
    vals = {"x_s_mm": xs, "x_k_mm": xk, "x_min_mm": 2.0 * xk - xs, "jump_mm": 2.0 * De,
            "amplitude_mm": math.sqrt(De * De + (vb / w) ** 2), "T_s": t_stick + t_slip, "t_stick_s": t_stick,
            "t_slip_s": t_slip, "omega_n_rad_s": w, "v_max_mm_s": vb + w * math.sqrt(De * De + (vb / w) ** 2)}
    warn = [] if mus > muk else ["μs = μk – no stick-slip (steady slip at x_k)"]
    return result(vals, "x_s = μs·mg/k, x_k = μk·mg/k; t_stick = 2(x_s − x_k)/v; t_slip = (2π − 2 atan(ω(x_s − x_k)/v))/ω; jump Δx ≈ 2(F_S − F_C)/k",
                  "Popp & Stelter 1990 (stick-slip oscillator on a belt)", warn)


# ------------------------------------------------------------------ rolling of a cylinder
SHAPE_K = {"solid": 0.5, "hollow": 1.0, "sphere": 0.4, "hollow_sphere": 2.0 / 3.0}


def rolling_cylinder(alpha_deg: float, mu: float, R: float = 1.0, shape: str = "solid", k: float | None = None,
                     g: float = G) -> dict:
    """Rolling down an inclined plane α: shape solid (solid cylinder, J = mR²/2) | hollow (thin tube) | sphere |
    hollow_sphere, or k = J/(mR²). It rolls without slip for μ ≥ tan α·k/(1 + k), otherwise it slips.
    """
    if not 0 <= alpha_deg < 90:
        raise CalcError("alpha_deg must be in [0; 90)°")
    mu, R = positive("mu", mu, zero=True), positive("R", R)
    kk = SHAPE_K[choice("shape", shape, tuple(SHAPE_K))] if k is None else positive("k", k)
    a_ = math.radians(alpha_deg)
    mu_min = math.tan(a_) * kk / (1.0 + kk)
    if mu >= mu_min:
        acc = g * math.sin(a_) / (1.0 + kk)
        vals = {"rolls": True, "a_mm_s2": acc, "eps_rad_s2": acc / R, "mu_required": mu_min,
                "F_friction_over_mg": kk / (1 + kk) * math.sin(a_)}
    else:
        acc = g * (math.sin(a_) - mu * math.cos(a_))
        vals = {"rolls": False, "a_mm_s2": acc, "eps_rad_s2": mu * g * math.cos(a_) / (kk * R), "mu_required": mu_min}
    return result(vals, "rolling a = g sin α/(1 + k), k = J/(mR²); condition μ ≥ tan α·k/(1 + k); slipping a = g(sin α − μ cos α), ε = μg cos α/(kR)",
                  "rigid-body dynamics (rolling with slipping)")


# ------------------------------------------------------------------ reduced inertia and run-up
def reduced_inertia(J_motor: float, J_load: float, i: float, M_load: float = 0.0, eps_motor: float = 0.0,
                    eta: float = 1.0, M_motor: float | None = None, n_target_rpm: float | None = None) -> dict:
    """Reduction to the motor shaft: J = J_m + J_z/i², required torque M = J·ε + M_z/(i·η) (i = n_motor/n_load,
    J [t·mm²], M [N·mm], ε [rad/s²]). With M_motor and n_target_rpm the time of the run-up t = J·ω/(M_m − M_z/(iη)).
    """
    Jm, Jz, i = positive("J_motor", J_motor, zero=True), positive("J_load", J_load, zero=True), positive("i", i)
    eta = positive("eta", eta)
    if eta > 1:
        raise CalcError("eta must be in (0; 1]")
    J = Jm + Jz / (i * i)
    Mz = float(M_load) / (i * eta)
    vals = {"J_red_tmm2": J, "J_red_kgm2": J * 1e-3, "M_load_red_Nmm": Mz, "M_required_Nmm": J * float(eps_motor) + Mz,
            "inertia_ratio": (Jz / (i * i)) / Jm if Jm else math.inf}
    warn = []
    if M_motor is not None and n_target_rpm is not None:
        w = positive("n_target_rpm", n_target_rpm) * 2.0 * math.pi / 60.0
        Macc = float(M_motor) - Mz
        if Macc <= 0:
            warn.append("Motor torque is insufficient for the load torque – it will not start")
            vals["t_start_s"] = math.inf
        else:
            vals["t_start_s"] = J * w / Macc
            vals["eps_start_rad_s2"] = Macc / J
    if Jm and vals["inertia_ratio"] > 10:
        warn.append("Load/motor inertia ratio > 10 – difficult servo drive control")
    return result(vals, "J = J_m + J_z/i²; M = J·ε + M_z/(i·η); t_start-up = J·ω/(M_m − M_z/(i·η))",
                  "inertia reduced to the motor shaft (e.g. Roloff/Matek, drives)", warn,
                  units="J v t·mm² (1 t·mm² = 10⁻³ kg·m²)")


# ------------------------------------------------------------------ residual vibration after a pulse
def residual_vibration(a: float, tau: float, f_n: float) -> dict:
    """Residual amplitude of an undamped oscillator of natural frequency f_n [Hz] after a rectangular pulse of acceleration a [mm/s²] of duration τ [s]
    (start of the axis): X = (2a/ω²)·|sin(ωτ/2)|; zero for τ = n/f_n.
    """
    a, tau, f = positive("a", a, zero=True), positive("tau", tau), positive("f_n", f_n)
    w = 2.0 * math.pi * f
    X = 2.0 * a / (w * w) * abs(math.sin(w * tau / 2.0))
    return result({"X_mm": X, "X_static_mm": a / (w * w), "ratio": X / (a / (w * w)) if a else 0.0,
                   "tau_zero_s": [n / f for n in range(1, 4)]},
                  "X = (2a/ω²)·|sin(ωτ/2)|, ω = 2πf_n (undamped)",
                  "1-DOF response to a rectangular pulse (Harris, Shock and Vibration Handbook, ch. 8)")


# ------------------------------------------------------------------ S-curve
def s_curve(distance: float, v_max: float, a_max: float, j_max: float) -> dict:
    """Profile of the axis with limited jerk (7 sections): distance [mm], v_max [mm/s], a_max [mm/s²], j_max [mm/s³]. Returns the times of the
    sections, the total time and the reached v, a; the course see s_curve_profile.
    """
    D, V, A, J = (positive(n, x) for n, x in (("distance", distance), ("v_max", v_max), ("a_max", a_max),
                                               ("j_max", j_max)))

    def acc_phase(v):
        if v * J >= A * A:
            return A / J, v / A - A / J, A            # t_j, t_a, a_peak
        ap = math.sqrt(v * J)
        return ap / J, 0.0, ap

    v = V
    tj, ta, ap = acc_phase(v)
    d_acc = v * (2.0 * tj + ta) / 2.0
    if 2.0 * d_acc > D:                                  # v_max is not reached
        v1 = A * (-A / J + math.sqrt(A * A / (J * J) + 4.0 * D / A)) / 2.0
        v = v1 if v1 * J >= A * A else (D * D * J / 4.0) ** (1.0 / 3.0)
        tj, ta, ap = acc_phase(v)
        d_acc = v * (2.0 * tj + ta) / 2.0
    tv = max(D - 2.0 * d_acc, 0.0) / v
    segs = [tj, ta, tj, tv, tj, ta, tj]
    vals = {"T_s": sum(segs), "t_j_s": tj, "t_a_s": ta, "t_v_s": tv, "v_peak_mm_s": v, "a_peak_mm_s2": ap,
            "d_acc_mm": d_acc, "segments_s": segs, "reaches_v_max": bool(abs(v - V) < 1e-9 * V)}
    return result(vals, "7 jerk segments +j, 0, −j, 0, −j, 0, +j; t_j = a/j, t_a = v/a − a/j, t_v = (D − 2d_acc)/v",
                  "jerk-limited profile (e.g. Biagiotti & Melchiorri, Trajectory Planning for Automatic Machines)")


def s_curve_profile(distance: float, v_max: float, a_max: float, j_max: float, n: int = 501) -> dict:
    """Course of the S-curve in time: t, s, v, a, j (numpy arrays) – exact integration over the sections of a constant jerk."""
    r = s_curve(distance, v_max, a_max, j_max)
    segs = r["segments_s"]
    jerks = [1, 0, -1, 0, -1, 0, 1]
    J = positive("j_max", j_max)
    # states at the starts of the sections
    starts = []
    s = v = a = 0.0
    t0 = 0.0
    for dt, js in zip(segs, jerks):
        j = js * J
        starts.append((t0, s, v, a, j))
        s, v, a = s + v * dt + a * dt * dt / 2 + j * dt ** 3 / 6, v + a * dt + j * dt * dt / 2, a + j * dt
        t0 += dt
    t = np.linspace(0.0, t0, max(int(n), 2))
    out = {k: np.zeros_like(t) for k in ("s", "v", "a", "j")}
    for k, (ts, s0, v0, a0, j) in enumerate(starts):
        te = starts[k + 1][0] if k + 1 < len(starts) else t0 + 1e-12
        m = (t >= ts) & (t <= te) if k + 1 == len(starts) else (t >= ts) & (t < te)
        dt = t[m] - ts
        out["s"][m] = s0 + v0 * dt + a0 * dt * dt / 2 + j * dt ** 3 / 6
        out["v"][m] = v0 + a0 * dt + j * dt * dt / 2
        out["a"][m] = a0 + j * dt
        out["j"][m] = j
    out["t"] = t
    return out


KINDS = {"slider_crank": (slider_crank, "slider-crank (x, v, a, torque)"),
         "four_bar": (four_bar, "four-bar linkage – Freudenstein, Grashof"),
         "eccentric_cam": (eccentric_cam, "eccentric cam with roller follower (lift, pressure angle, force, Hertz)"),
         "pendulum": (pendulum, "pendulum – exact period"),
         "sliding_stop": (sliding_stop, "coast-down of a body with friction"),
         "stick_slip": (stick_slip, "stick-slip on a belt"),
         "rolling_cylinder": (rolling_cylinder, "rolling down an inclined plane"),
         "reduced_inertia": (reduced_inertia, "reduced inertia and start-up"),
         "residual_vibration": (residual_vibration, "residual vibration after an acceleration pulse"),
         "s_curve": (s_curve, "S-curve axis motion profile (limited jerk)")}
