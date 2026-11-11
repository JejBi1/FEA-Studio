"""Analytic calculators (stage F8.2, fea_core.calc): check numbers from the F8 spec chapter 7 (K2, K3, K4, KK1–KK4, R5,
R7, MB1–MB6, Y10), the library of friction coefficients (purpose, material families, user library) and the selection by kind.
"""
import json
import math

import pytest

from fea_core import config
from fea_core.calc import CalcError, call, kinds
from fea_core.calc import (damping, friction, friction_lib as fl, hertz, mechanisms as mech,
                           pressfit, pulses, snapfit, convection)


def close(a, b, rel=1e-3):
    assert a == pytest.approx(b, rel=rel), (a, b)


# ------------------------------------------------------------------ snap-fit (KK2)
def test_snapfit_kk2():
    r = snapfit.snap_fit(l=15, h=1.5, b=5, E=2000, y=0.5, alpha_deg=30, mu=0.3)
    close(r["eps"], 0.005, 1e-9)
    close(r["P_N"], 1.25, 1e-9)
    close(r["W_N"], 1.326, 1e-3)
    close(r["theta_tip_deg"], 2.86, 2e-3)
    close(r["W_tip_rotation_N"], 1.467, 1e-3)
    assert r["W_design_N"] == r["W_tip_rotation_N"]
    assert snapfit.snap_fit(15, 1.5, 5, 2000, 0.5, tip_rotation=False)["W_design_N"] == pytest.approx(r["W_N"])


def test_snapfit_bayer_taper_and_release():
    # Bayer type 2 (h → h/2): y = 1.09·ε·l²/h → 2·C_y ≈ 1.09 (exactly 2(3.5 − 6 + 4 ln 2) = 1.0904)
    r = snapfit.snap_fit(15, 1.5, 5, 2000, 0.5, h_ratio=0.5)
    close(2 * r["C_y"], 4 * (4 * math.log(2) - 2.5), 1e-9)
    close(2 * r["C_y"], 1.09, 1e-3)
    assert r["eps_max_at_x_mm"] == 0.0
    # non-separability μ·tan α' ≥ 1
    r = snapfit.snap_fit(15, 1.5, 5, 2000, 0.5, mu=0.6, alpha_release_deg=60, tip_rotation=False)
    assert r["permanent"] and math.isinf(r["W_release_N"])
    r = snapfit.snap_fit(15, 1.5, 5, 2000, 0.5, mu=0.3, alpha_release_deg=45, eps_allow_pct=0.4)
    assert not r["permanent"] and r["utilization"] == pytest.approx(1.25) and r["warnings"]
    with pytest.raises(CalcError):
        snapfit.snap_fit(15, 1.5, 5, 2000, -0.5)


# ------------------------------------------------------------------ press fit (K4)
def test_pressfit_k4():
    r = pressfit.press_fit(d=20, D=40, L=20, delta=0.02, mu=0.15)
    close(r["p_MPa"], 78.75, 1e-9)
    close(r["sigma_t_hub_bore_MPa"], 131.25, 1e-9)
    close(r["u_r_hub_um"], 7.375, 1e-9)
    close(r["u_r_shaft_um"], -2.625, 1e-9)
    close(r["T_Nm"], 148.4, 1e-3)
    close(r["F_axial_N"], 14840, 1e-3)
    close(r["u_r_hub_um"] - r["u_r_shaft_um"], 10.0, 1e-9)        # radial interference
    r = pressfit.press_fit(20, 40, 20, 0.02, alpha_hub=11.5e-6, mount_clearance=0.02)
    close(r["dT_heating_K"], 0.04 / (11.5e-6 * 20), 1e-12)


def test_pressfit_hollow_and_materials():
    # a hollow shaft is more compliant → a smaller pressure; an Al hub → a smaller pressure
    p0 = pressfit.press_fit(20, 40, 20, 0.02)["p_MPa"]
    assert pressfit.press_fit(20, 40, 20, 0.02, d_inner=10)["p_MPa"] < p0
    assert pressfit.press_fit(20, 40, 20, 0.02, E_hub=70000, nu_hub=0.33, E_shaft=210000, nu_shaft=0.3)["p_MPa"] < p0
    with pytest.raises(CalcError):
        pressfit.press_fit(20, 18, 20, 0.02)


# ------------------------------------------------------------------ Hertz (K2, K3, KK1, KK3, KK4)
def test_hertz_sphere_k2():
    r = hertz.sphere(F=1000, R1=10)
    close(r["E_star_MPa"], 115385, 1e-5)
    close(r["a_mm"], 0.4021, 1e-3)
    close(r["p0_MPa"], 2953, 1e-3)
    close(r["delta_um"], 16.17, 1e-3)
    close(r["tau_max_ratio"], 0.31, 1e-2)
    close(r["z_tau_max_ratio"], 0.48, 1e-2)


def test_hertz_cylinder_k3_and_cam_kk1():
    r = hertz.cylinder(F_per_length=100, R1=10)
    close(r["b_mm"], 0.1050, 1e-3)
    close(r["p0_MPa"], 606.0, 1e-3)
    close(r["tau_max_MPa"] / r["p0_MPa"], 0.30, 1e-2)
    r = hertz.cam_roller(N=100, b=10, R_cam=30, r_roller=10)
    close(r["rho_mm"], 7.5, 1e-12)
    close(r["p0_MPa"], 221.3, 1e-3)


def test_hertz_gear_kk3():
    r = hertz.gear(z1=20, z2=40, m=3, b=10, T1=20000)
    close(r["rho_red_mm"], 6.840, 1e-3)
    close(r["F_n_N"], 709.5, 1e-3)
    close(r["p0_MPa"], 617.2, 1e-3)
    close(r["sigma_H0_MPa"], r["p0_MPa"], 1e-9)          # Hertz at a rolling point = ISO 6336-2 (Z_ε = Z_β = 1)
    close(r["eps_alpha"], 1.635, 1e-3)
    assert r["single_pair_at_pitch"]


def test_hertz_impact_kk4():
    r = hertz.impact(R=10, v0=1000)
    close(r["delta_max_um"], 17.80, 1e-3)
    close(r["t_c_us"], 52.3, 1e-3)
    close(r["F_max_N"], 2310, 1e-3)
    assert r["wave_transit_us"] < r["t_c_us"] / 10


# ------------------------------------------------------------------ friction (R5, R7, …)
def test_power_screw_r7():
    r = friction.power_screw(F=1000, d2=18, P=4, mu=0.1, flank_angle_deg=30)
    close(r["lambda_deg"], 4.046, 1e-3)
    close(r["rho_prime_deg"], 5.911, 1e-3)
    close(r["eta"], 0.403, 1e-3)
    close(r["T_raise_Nm"], 1.580, 1e-3)
    close(r["T_lower_Nm"], 0.293, 2e-3)
    assert r["self_locking"] and r["eta"] < 0.5 and r["eta_back"] < 0


def test_journal_r5_and_elements():
    mu, F, r = 0.2, 1000, 5
    close(friction.journal(F, r, mu, "clearance")["M_Nmm"], 980.6, 1e-3)       # 0,981 N·m
    close(friction.journal(F, r, mu, "worn")["M_Nmm"], 1248.5, 1e-3)           # 1,248 N·m
    close(friction.journal(F, r, mu, "fitted")["mu_eff"], math.pi / 2 * mu / math.sqrt(1 + mu * mu), 1e-12)
    close(friction.thrust_collar(100, 20, 0.1, r1=10, state="worn")["M_Nmm"], 150, 1e-12)
    close(friction.thrust_collar(100, 20, 0.1, state="new")["r_eff_mm"], 40 / 3, 1e-12)
    close(friction.ball_joint(100, 10, 0.1, "perpendicular")["M_Nmm"], 400 / math.pi, 1e-12)
    close(friction.pivot(100, 1, 0.1)["M_Nmm"], 0.589 * 10, 1e-3)
    close(friction.belt(0.2, 180, F2=100)["ratio"], 1.8745, 1e-4)                 # R6 kapstan
    close(friction.v_groove(0.1, 38)["factor"], 3.07, 1e-3)
    assert friction.wedge(10, 0.1)["self_locking"] and not friction.wedge(15, 0.1)["self_locking"]
    j = friction.guide_jamming(e=100, L=50, mu=0.3)
    assert j["jams"] and j["L_min_mm"] == pytest.approx(60)
    assert not friction.guide_jamming(e=50, L=100, mu=0.3)["jams"]
    close(friction.rolling_bearing(1000, 20, "ball")["M_Nmm"], 15, 1e-12)
    close(friction.rolling_bearing(1000, 20, "ball", start=True, sealed=True)["mu"], 0.004, 1e-12)
    close(friction.rolling_resistance(1000, 50, pair="steel_steel")["F_N"], 10, 1e-12)
    w = friction.worm(5, 0.05)
    assert w["self_locking"] is False and 0 < w["eta"] < 1
    close(friction.clutch(1000, 50, 0.3, r=30, surfaces=2)["M_Nmm"], 2 * 0.3 * 1000 * 40, 1e-12)
    close(friction.inclined_plane(16.699)["mu_s"], 0.3, 1e-3)
    b = friction.belt(0.3, 180, F1=1000, q=1e-7, v=10000)                        # q·v² = 10 N
    close((b["F1_N"] - 10) / (b["F2_N"] - 10), math.exp(0.3 * math.pi), 1e-12)


# ------------------------------------------------------------------ mechanismy (MB1–MB6, KK1, Y10)
def test_slider_crank_mb1():
    r = mech.slider_crank(r=20, l=80, phi_deg=60, omega=100, F=500)
    close(r["x_mm"], 88.102, 1e-5)
    close(r["v_mm_s"], -1953.8, 1e-4)
    close(r["a_mm_s2"], -75022, 1e-4)
    close(r["M_Nmm"], 500 * r["v_mm_s"] / 100, 1e-12)
    # derivace numericky
    h = 1e-4
    xp = mech.slider_crank(20, 80, 60 + math.degrees(h))["x_mm"]
    xm = mech.slider_crank(20, 80, 60 - math.degrees(h))["x_mm"]
    close(r["v_mm_s"], (xp - xm) / (2 * h) * 100, 1e-6)
    close(r["a_mm_s2"], (xp - 2 * r["x_mm"] + xm) / h ** 2 * 1e4, 1e-4)


def test_four_bar_mb2():
    r = mech.four_bar(a=40, b=120, c=80, d=100, theta2_deg=60)
    close(r["theta4_open_deg"], 64.943, 1e-5)
    close(r["theta4_crossed_deg"], 248.230, 1e-5)
    assert r["grashof"] and "crank" in r["grashof_type"]
    # the closing condition of both branches
    t2 = math.radians(60)
    for br in r["branches"].values():
        t3, t4 = math.radians(br["theta3_deg"]), math.radians(br["theta4_deg"])
        assert abs(40 * math.cos(t2) + 120 * math.cos(t3) - 80 * math.cos(t4) - 100) < 1e-9
        assert abs(40 * math.sin(t2) + 120 * math.sin(t3) - 80 * math.sin(t4)) < 1e-9


def test_eccentric_cam_kk1():
    close(mech.eccentric_cam(5, 30, 10, 60)["s_mm"], 7.265, 1e-4)
    r = mech.eccentric_cam(5, 30, 10, 90, k=1, F0=10, b=10)
    close(r["s_mm"], 4.686, 1e-4)
    close(r["pressure_angle_deg"], 7.18, 1e-3)
    close(r["N_N"], (10 + r["s_mm"]) / math.cos(math.radians(r["pressure_angle_deg"])), 1e-12)


def test_pendulum_mb3_stop_mb5_stickslip_mb6_rolling_mb4():
    r = mech.pendulum(L=500, theta0_deg=60, m=0.001)
    close(r["T_s"], 1.5223, 1e-4)
    close(r["T_small_s"], 1.4185, 1e-4)
    close(r["N_bottom_N"], 2 * 0.001 * 9810, 1e-12)
    close(mech.ellipk(0.25), 1.685750354812596, 1e-12)
    r = mech.sliding_stop(v0=1000, mu=0.2)
    close(r["s_mm"], 254.8, 1e-3)
    close(r["t_s"], 0.5097, 1e-3)
    r = mech.stick_slip(m=0.001, k=1.0, v_belt=100, mu_s=0.4, mu_k=0.3)
    close(r["x_k_mm"], 2.943, 1e-4)
    close(r["x_s_mm"], 3.924, 1e-4)
    close(r["T_s"], 0.1993, 1e-3)
    r = mech.rolling_cylinder(30, 0.3)
    assert r["rolls"]
    close(r["a_mm_s2"], 3270, 1e-3)
    r = mech.rolling_cylinder(30, 0.1, R=50)
    assert not r["rolls"]
    close(r["a_mm_s2"], 4055, 1e-3)
    close(r["eps_rad_s2"], 2 * 0.1 * 9810 * math.cos(math.radians(30)) / 50, 1e-12)


def test_inertia_residual_scurve():
    r = mech.reduced_inertia(J_motor=100, J_load=10000, i=10, M_load=1000, eps_motor=50)
    close(r["J_red_tmm2"], 200, 1e-12)
    close(r["M_required_Nmm"], 200 * 50 + 100, 1e-12)
    r = mech.residual_vibration(a=1000, tau=0.01, f_n=30)
    w = 2 * math.pi * 30
    close(r["X_mm"], 2000 / w ** 2 * abs(math.sin(w * 0.005)), 1e-12)
    assert mech.residual_vibration(1000, 1 / 30, 30)["X_mm"] < 1e-15
    for D in (100.0, 5.0, 0.01):
        s = mech.s_curve(D, 200, 2000, 50000)
        p = mech.s_curve_profile(D, 200, 2000, 50000, n=2001)
        close(p["s"][-1], D, 1e-9)
        assert abs(p["v"][-1]) < 1e-9 * 200 and abs(p["a"][-1]) < 1e-6 * 2000
        assert p["v"].max() <= 200 * (1 + 1e-9) and p["a"].max() <= 2000 * (1 + 1e-9)
        close(p["t"][-1], s["T_s"], 1e-12)
    assert mech.s_curve(100, 200, 2000, 50000)["reaches_v_max"]


# ------------------------------------------------------------------ damping and fall
def test_damping():
    r = damping.half_power(100, 98, 102)
    close(r["zeta"], 0.02, 1e-12)
    close(r["Q"], 25, 1e-12)
    d = 2 * math.pi * 0.05 / math.sqrt(1 - 0.05 ** 2)
    close(damping.log_decrement(1.0, math.exp(-3 * d), 3)["zeta"], 0.05, 1e-12)
    close(damping.convert(Q=10)["zeta"], 0.05, 1e-12)
    close(damping.convert(eta=0.04)["zeta"], 0.02, 1e-12)
    close(damping.convert(delta=d)["zeta"], 0.05, 1e-12)
    r = damping.rayleigh(10, 100, 0.02, f_check=[10, 100])
    w1, w2 = 2 * math.pi * 10, 2 * math.pi * 100
    close(r["alpha_1_s"], 2 * 0.02 * w1 * w2 / (w1 + w2), 1e-12)
    close(r["beta_s"], 2 * 0.02 / (w1 + w2), 1e-12)
    assert all(v == pytest.approx(0.02) for v in r["zeta_at"].values())
    close(damping.pcb(100)["zeta_pct"], 5.0, 1e-12)
    assert len(damping.table()["rows"]) >= 7
    with pytest.raises(CalcError):
        damping.convert(zeta=0.1, Q=5)


def test_drop_pulse():
    r = pulses.drop(h=1000, tau=0.002, e=0.5)
    v = math.sqrt(2 * 9810 * 1000)
    close(r["v_mm_s"], v, 1e-12)
    close(r["A_mm_s2"], math.pi * 1.5 * v / 0.004, 1e-12)
    close(2 * r["A_mm_s2"] * r["tau_s"] / math.pi, r["dv_mm_s"], 1e-12)          # area of the half-sine = Δv
    close(pulses.drop(1000, m=0.001, k=100)["tau_s"], math.pi * math.sqrt(1e-5), 1e-12)
    with pytest.raises(CalcError):
        pulses.drop(1000)


# ------------------------------------------------------------------ selection by kind (MCP, GUI)
def test_call_by_kind():
    assert set(kinds("friction")) >= {"journal", "power_screw", "belt", "worm", "clutch", "guide_jamming"}
    r = call("friction", "power_screw", {"F": 1000, "d2": 18, "P": 4, "mu": 0.1})
    close(r["T_raise_Nm"], 1.580, 1e-3)
    with pytest.raises(CalcError, match="unknown parameters"):
        call("friction", "power_screw", {"F": 1000, "d2": 18, "P": 4, "mu": 0.1, "x": 1})
    with pytest.raises(CalcError, match="missing"):
        call("hertz", "sphere", {"F": 1})
    with pytest.raises(CalcError, match="Unknown type"):
        call("mechanism", "xx", {})
    for g in ("snapfit", "pressfit", "hertz", "friction", "mechanism", "damping", "pulse", "convection"):
        for k, (fn, desc) in kinds(g).items():
            assert callable(fn) and desc and fn.__doc__, (g, k)


def test_convection_finding1_and_textbook():
    """Finding 1: check of the accuracy of the properties of air and of the convection correlations."""
    # 1. 300 mm plate 60/20 °C: gives ~5.25 W/(m²K), not the earlier underestimated 5.056 W/(m²K)
    r = convection.natural_vertical_plate(L_mm=300, Ts=60, Tinf=20)
    assert 5.20 <= r["h"] <= 5.35
    assert r["h"] == pytest.approx(5.249, rel=0.01)

    # 2. Properties of air: k ~ 0.0257 W/(m·K), Pr ~ 0.709 at 20 °C
    rho, cp, k, nu, Pr = convection.air_properties(20.0)
    assert 0.0250 <= k <= 0.0265
    assert 0.70 <= Pr <= 0.72

    # 3. Incropera Churchill-Chu vertical plate (Incropera ch. 9)
    r_cc = call("convection", "vertical_plate", {"L_mm": 300, "Ts": 70, "Tinf": 25})
    assert 4.8 <= r_cc["h"] <= 5.6

    # 4. Flat plate laminar forced convection Nu = 0.664 Re^0.5 Pr^(1/3)
    r_fp = call("convection", "forced_flat_plate", {"L_mm": 500, "velocity_m_s": 2.0, "Ts": 50, "Tinf": 20})
    assert r_fp["regime"] == "laminar"
    Re = r_fp["Re"]
    Pr_val = r_fp["Pr"]
    Nu_expected = 0.664 * (Re ** 0.5) * (Pr_val ** (1.0 / 3.0))
    close(r_fp["Nu"], Nu_expected, rel=0.02)

    # 5. Flow in a pipe (Gnielinski)
    r_pipe = call("convection", "forced_internal_pipe", {"D_mm": 50, "velocity_m_s": 5.0, "Ts": 50, "Tinf": 20})
    assert "turbulent" in r_pipe["regime"]
    assert r_pipe["h"] > 10.0


def test_calc_has_no_heavy_imports():
    import subprocess
    import sys
    code = ("import sys, fea_core.calc.friction_lib, fea_core.calc.mechanisms, fea_core.calc.hertz, fea_core.calc.convection; "
            "bad = [m for m in ('OCP', 'PySide6', 'scipy', 'gmsh') if m in sys.modules]; print(bad); "
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(config.PROJECT_ROOT))
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------ friction library
def test_friction_library_builtin():
    names = [p.name for p in fl.builtin_pairs()]
    assert len(names) == len(set(names)) and len(names) >= 18
    for p in fl.builtin_pairs():
        assert p.source and p.a in fl.FAMILIES and p.b in fl.FAMILIES, p.name
    assert fl.mu("ocel – ocel, suchá", "slip") == 0.15
    assert fl.mu("ocel – ocel, suchá", "resistance") == 0.8
    assert fl.mu("PLA – ocel", "slip") == 0.2 and fl.mu("PLA – ocel", "resistance") == 0.3
    assert fl.get("ocel – hliník").unverified_value("slip")
    with pytest.raises(fl.FrictionLibError, match="measure"):
        fl.mu("PLA – PLA", "slip")
    with pytest.raises(fl.FrictionLibError, match="Purpose"):
        fl.mu("PLA – ocel", "x")
    with pytest.raises(fl.FrictionLibError):
        fl.get("neexistuje")
    assert fl.mu("třída povrchu C (EN 1090-2)") == 0.3
    assert fl.RULES and set(fl.PURPOSES) == {"slip", "resistance"}


def test_friction_suggest_families():
    assert fl.family("Ocel S235") == "steel" and fl.family("Tisk PLA Bambu Basic") == "PLA"
    assert fl.family("Tisk PETG (orientačně)") == "PETG" and fl.family("Hliník EN AW-6061 T6") == "aluminium"
    assert fl.family("POM-C") == "POM" and fl.family("PA6") == "PA" and fl.family("nic") is None

    class M:
        name = "Ocel C45"
    s = fl.suggest(M(), "Tisk PLA Bambu Basic")
    assert [x["name"] for x in s] == ["PLA – ocel"]
    s = fl.suggest("ocel", "Ocel S355")
    assert s[0]["name"] == "ocel – ocel, suchá" and s[-1].get("surface_class")
    assert fl.suggest("ocel", "ocel", lubricated=True)[0]["name"] == "ocel – ocel, mazaná"
    assert fl.suggest("ocel", "ocel", surface_class="B")[0]["name"] == "třída povrchu B (EN 1090-2)"
    assert fl.suggest("PLA", "POM") == []


def test_friction_user_library(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "user_config_dir", lambda: tmp_path / "FEAStudio")
    assert fl.user_pairs() == []
    p = fl.save_pair({"name": "PLA – PLA (změřeno)", "a": "PLA", "b": "PLA", "mu_s": [0.3, 0.35], "mu_min": 0.25,
                      "mu_max": 0.4, "source": "vlastní měření nakloněnou rovinou", "budouci_klic": 1})
    assert p.mu("slip") == 0.25
    assert fl.mu("PLA – PLA (změřeno)", "resistance") == 0.4
    assert "PLA – PLA (změřeno)" in [x["name"] for x in fl.suggest("PLA", "PLA")]
    path = tmp_path / "FEAStudio" / "friction.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.append({"name": "vadný", "mu_min": "x"})                  # a faulty record is skipped, it stays in the file
    raw.append(42)
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    assert [x.name for x in fl.user_pairs()] == ["PLA – PLA (změřeno)"]
    fl.save_pair({"name": "pryž – ocel (změřeno)", "a": "rubber", "b": "steel", "mu_min": 0.6, "mu_max": 0.9})
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert {"name": "vadný", "mu_min": "x"} in raw and 42 in raw
    with pytest.raises(fl.FrictionLibError, match="built-in"):
        fl.save_pair({"name": "ocel – ocel, suchá", "a": "steel", "b": "steel"})
    with pytest.raises(fl.FrictionLibError):
        fl.save_pair({"name": "x", "a": "steel", "b": "steel", "mu_min": 0.5, "mu_max": 0.2})
    fl.delete_pair("pryž – ocel (změřeno)")
    with pytest.raises(fl.FrictionLibError):
        fl.delete_pair("pryž – ocel (změřeno)")
    with pytest.raises(fl.FrictionLibError):
        fl.delete_pair("ocel – ocel, suchá")
    # unreadable file: reading empty, saving refused (it is not overwritten)
    path.write_text("{nečitelné", encoding="utf-8")
    assert fl.user_pairs() == []
    assert len(fl.pairs()) == len(fl.builtin_pairs())
    with pytest.raises(fl.FrictionLibError, match="Cannot read user friction library"):
        fl.save_pair({"name": "y", "a": "PLA", "b": "PLA"})
    assert path.read_text(encoding="utf-8") == "{nečitelné"


# ------------------------------------------------------------------ MCP
CALC_TOOLS = {"fea_calc_snapfit", "fea_calc_pressfit", "fea_calc_hertz", "fea_calc_friction",
              "fea_calc_friction_library", "fea_calc_mechanism", "fea_calc_damping", "fea_calc_drop",
              "fea_calc_convection"}


def test_mcp_calc_tools(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from tests.test_mcp import run
    monkeypatch.setattr(config, "user_config_dir", lambda: tmp_path / "FEAStudio")

    async def body(s, c):
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert CALC_TOOLS <= set(tools), CALC_TOOLS - set(tools)
        # outside the core group the listing carries a summary; the kinds are in the full text (fea_tool_help)
        help_txt = (await c.call_tool("fea_tool_help", {"tool": "fea_calc_friction"})).content[0].text
        assert "power_screw" in help_txt
        d = await s.ok("fea_calc_snapfit", l_mm=15, h_mm=1.5, b_mm=5, E_MPa=2000, y_mm=0.5)
        close(d["W_tip_rotation_N"], 1.467, 1e-3)
        d = await s.ok("fea_calc_pressfit", d_mm=20, D_mm=40, L_mm=20, delta_mm=0.02)
        close(d["p_MPa"], 78.75, 1e-4)
        d = await s.ok("fea_calc_hertz", kind="sphere", params={"F": 1000, "R1": 10})
        close(d["p0_MPa"], 2953, 1e-3)
        d = await s.ok("fea_calc_friction", kind="power_screw", params={"F": 1000, "d2": 18, "P": 4, "mu": 0.1})
        assert d["self_locking"] is True
        assert "unknown parameters" in await s.err("fea_calc_friction", kind="journal", params={"F": 1, "x": 2})
        assert "Unknown type" in await s.err("fea_calc_mechanism", kind="xx")
        d = await s.ok("fea_calc_mechanism", kind="four_bar",
                       params={"a": 40, "b": 120, "c": 80, "d": 100, "theta2_deg": 60})
        close(d["theta4_open_deg"], 64.943, 1e-5)
        d = await s.ok("fea_calc_damping", kind="half_power", params={"f_n": 100, "f1": 98, "f2": 102})
        close(d["zeta"], 0.02, 1e-4)
        d = await s.ok("fea_calc_drop", params={"h": 1000, "tau": 0.002})
        assert d["A_g"] > 0
        d = await s.ok("fea_calc_convection", kind="vertical_plate", params={"L_mm": 300, "Ts": 60, "Tinf": 20})
        assert d["h"] > 5.0
        d = await s.ok("fea_calc_friction_library")
        assert len(d["pairs"]) >= 18 and d["rules"]
        d = await s.ok("fea_calc_friction_library", action="mu", name="PLA – ocel", purpose="resistance")
        assert d["mu"] == 0.3
        d = await s.ok("fea_calc_friction_library", action="suggest", material_a="Ocel S235",
                       material_b="Tisk PETG (orientačně)")
        assert d["pairs"][0]["name"] == "PETG – ocel"
        assert "measure" in await s.err("fea_calc_friction_library", action="mu", name="PLA – PLA")
        await s.ok("fea_calc_friction_library", action="save",
                   pair={"name": "test", "a": "PLA", "b": "PLA", "mu_min": 0.3, "mu_max": 0.5})
        assert (tmp_path / "FEAStudio" / "friction.json").exists()
        await s.ok("fea_calc_friction_library", action="delete", name="test")
    run(tmp_path, body)
