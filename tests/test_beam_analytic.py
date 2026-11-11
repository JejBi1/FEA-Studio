"""Analytic beams and curved bars (fea_core.beams.analytic) – fast tests against textbook values.

References verified independently: ODR solve_bvp, the reduction theorem with quadrature and a hand derivation
(scratch/beams_analytic/verify.py).
"""
from math import pi, log, sqrt, tan

import numpy as np
import pytest

from fea_core.beams import SectionProps
from fea_core.beams import analytic as an

E, I, L = 210000.0, 8.0e6, 3000.0
EI = E * I
q, P = 2.0, 1000.0


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# ---------------------------------------------------------------------- 1. single-span beams
def test_simple_uniform():
    r = an.beam(L, [an.uniform(q)], "simple", E=E, I=I)
    assert rel(r.reactions["R_A"], q * L / 2) < 1e-12 and rel(r.reactions["R_B"], q * L / 2) < 1e-12
    w, x = r.extremes["w"]["max"]
    assert rel(w, 5 * q * L ** 4 / (384 * EI)) < 1e-12 and abs(x - L / 2) < 1e-3
    M, x = r.extremes["M"]["max"]
    assert rel(M, q * L ** 2 / 8) < 1e-12 and abs(x - L / 2) < 1e-3
    assert rel(r.at(0.0)["phi"], q * L ** 3 / (24 * EI)) < 1e-12


def test_cantilever_force():
    r = an.beam(L, [an.Force(P, L)], "cantilever", E=E, I=I)
    assert rel(r.at(L)["w"], P * L ** 3 / (3 * EI)) < 1e-12
    assert rel(r.at(L)["phi"], P * L ** 2 / (2 * EI)) < 1e-12
    assert rel(r.reactions["R_A"], P) < 1e-12 and rel(r.reactions["M_A"], -P * L) < 1e-12
    assert r.extremes["w"]["abs"][1] == pytest.approx(L)


def test_fixed_uniform():
    r = an.beam(L, [an.uniform(q)], "fixed", E=E, I=I)
    assert rel(r.reactions["M_A"], -q * L ** 2 / 12) < 1e-12 and rel(r.reactions["M_B"], -q * L ** 2 / 12) < 1e-12
    assert rel(r.at(L / 2)["w"], q * L ** 4 / (384 * EI)) < 1e-12
    assert rel(r.at(L / 2)["M"], q * L ** 2 / 24) < 1e-12


def test_propped_uniform():
    r = an.beam(L, [an.uniform(q)], "propped", E=E, I=I)
    assert rel(r.reactions["R_B"], 3 * q * L / 8) < 1e-12 and rel(r.reactions["R_A"], 5 * q * L / 8) < 1e-12
    assert rel(r.reactions["M_A"], -q * L ** 2 / 8) < 1e-12
    M, x = r.extremes["M"]["max"]                       # 9qL²/128 at 5L/8 from the clamping
    assert rel(M, 9 * q * L ** 2 / 128) < 1e-12 and abs(x - 5 * L / 8) < 1e-3


def test_triangular_simple():
    r = an.beam(L, [an.triangular(q, 0, L)], "simple", E=E, I=I)
    assert rel(r.reactions["R_A"], q * L / 6) < 1e-12 and rel(r.reactions["R_B"], q * L / 3) < 1e-12
    M, x = r.extremes["M"]["max"]
    assert rel(M, q * L ** 2 / (9 * sqrt(3))) < 1e-12 and abs(x - L / sqrt(3)) < 1e-3
    w, x = r.extremes["w"]["max"]                       # Roark: 0,01304·W·L³/EI (W = qL/2) v 0,5193 L
    xm = L * sqrt(1 - sqrt(8 / 15))                     # exactly: w = qx(7L⁴ − 10L²x² + 3x⁴)/(360·EI·L)
    assert rel(w, q * xm * (7 * L ** 4 - 10 * L ** 2 * xm ** 2 + 3 * xm ** 4) / (360 * EI * L)) < 1e-12
    assert rel(w, 0.01304 * (q * L / 2) * L ** 3 / EI) < 5e-4 and abs(x - xm) < 1e-3


def test_triangular_cantilever():
    r1 = an.beam(L, [an.triangular(q, 0, L, "a")], "cantilever", E=E, I=I)   # max at the clamping
    r2 = an.beam(L, [an.triangular(q, 0, L)], "cantilever", E=E, I=I)        # max at the free end
    assert rel(r1.at(L)["w"], q * L ** 4 / (30 * EI)) < 1e-12
    assert rel(r2.at(L)["w"], 11 * q * L ** 4 / (120 * EI)) < 1e-12
    assert rel(r2.reactions["M_A"], -q * L ** 2 / 3) < 1e-12


def test_point_loads_classic():
    a = 1000.0
    b = L - a
    r = an.beam(L, [an.Force(P, a)], "simple", E=E, I=I)
    assert rel(r.at(a)["w"], P * a * a * b * b / (3 * EI * L)) < 1e-12
    i = np.nonzero(np.isclose(r.x, a))[0]               # jump of V at the place of the force (left and right limit)
    assert len(i) == 2 and rel(r.V[i[0]] - r.V[i[1]], P) < 1e-12
    r = an.beam(L, [an.Force(P, a)], "fixed", E=E, I=I)
    assert rel(r.reactions["M_A"], -P * a * b * b / L ** 2) < 1e-12
    assert rel(r.reactions["M_B"], -P * a * a * b / L ** 2) < 1e-12
    r = an.beam(L, [an.Force(P, L / 2)], "fixed", E=E, I=I)
    assert rel(r.at(L / 2)["w"], P * L ** 3 / (192 * EI)) < 1e-12
    r = an.beam(L, [an.Force(P, L / 2)], "propped", E=E, I=I)
    assert rel(r.reactions["R_B"], 5 * P / 16) < 1e-12 and rel(r.reactions["M_A"], -3 * P * L / 16) < 1e-12


def test_moment_and_partial():
    C = 5e5
    r = an.beam(L, [an.Moment(C, L / 2)], "simple", E=E, I=I)
    assert rel(r.reactions["R_A"], -C / L) < 1e-12 and abs(r.at(L / 2)["w"]) < 1e-15 * L
    assert rel(r.at(0.0)["phi"], -C * L / (24 * EI)) < 1e-12           # ∫(L−s)/L·M/EI ds
    # uniform load on part of a simply supported beam: the reactions from statics
    r = an.beam(L, [an.uniform(q, 500, 2200)], "simple", E=E, I=I)
    Q = q * 1700
    assert rel(r.reactions["R_B"], Q * (500 + 850) / L) < 1e-12
    # a load extending beyond the beam is cut
    r2 = an.beam(L, [an.uniform(q, 500, 5000)], "simple", E=E, I=I)
    assert rel(r2.reactions["R_A"] + r2.reactions["R_B"], q * (L - 500)) < 1e-12


def test_superposition():
    loads = [an.Force(P, 700), an.uniform(q, 0, 2000), an.Moment(3e5, 2500), an.triangular(1.0, 1000, L)]
    r = an.beam(L, loads, "propped", E=E, I=I)
    parts = [an.beam(L, [ld], "propped", E=E, I=I) for ld in loads]
    for xx in (0.0, 650.0, 1500.0, 2900.0):
        assert rel(r.at(xx)["w"], sum(p.at(xx)["w"] for p in parts)) < 1e-11
    assert rel(r.reactions["M_A"], sum(p.reactions["M_A"] for p in parts)) < 1e-12


def test_timoshenko():
    G, As = 80769.0, 3000.0
    r = an.beam(L, [an.Force(P, L)], "cantilever", E=E, I=I, shear=True, G=G, A_s=As)
    assert rel(r.at(L)["w"], P * L ** 3 / (3 * EI) + P * L / (G * As)) < 1e-12
    r = an.beam(L, [an.uniform(q)], "simple", E=E, I=I, shear=True, G=G, A_s=As)
    assert rel(r.at(L / 2)["w"], 5 * q * L ** 4 / (384 * EI) + q * L ** 2 / (8 * G * As)) < 1e-12
    r = an.beam(L, [an.uniform(q)], "fixed", E=E, I=I, shear=True, G=G, A_s=As)
    assert rel(r.at(L / 2)["w"], q * L ** 4 / (384 * EI) + q * L ** 2 / (8 * G * As)) < 1e-12
    assert rel(r.reactions["M_A"], -q * L ** 2 / 12) < 1e-12
    # section: I and the shear area from SectionProps (Av_z for bending about y), G = E/2(1+ν)
    sec = SectionProps("x", "RECT", A=6000.0, Iy=I, Iz=1e6, Av_z=As)
    r2 = an.beam(L, [an.Force(P, L)], "cantilever", E=E, section=sec, shear=True, nu=0.3)
    assert rel(r2.at(L)["w"], P * L ** 3 / (3 * EI) + P * L / (E / 2.6 * As)) < 1e-12


def test_timoshenko_propped_vs_ode():
    from scipy.integrate import solve_bvp
    G, As = 80769.0, 3000.0
    cs = 1 / (G * As)
    r = an.beam(L, [an.uniform(q)], "propped", E=E, I=I, shear=True, G=G, A_s=As)
    s = solve_bvp(lambda x, y: np.vstack([y[1] + cs * y[3], -y[2] / EI, y[3], -q + 0 * x]),
                  lambda a, b: np.array([a[0], a[1], b[0], b[2]]), np.linspace(0, L, 50), np.zeros((4, 50)),
                  tol=1e-10)
    assert rel(r.reactions["R_B"], -s.sol(L)[3]) < 1e-7
    assert rel(r.at(1200.0)["w"], s.sol(1200.0)[0]) < 1e-7


def test_mechanism():
    with pytest.raises(ValueError):
        an.beam(L, [an.Force(P, 100)], ("free", "free"), E=E, I=I)
    with pytest.raises(ValueError):
        an.beam(L, [an.Force(P, 100)], ("guided", "free"), E=E, I=I)


# ---------------------------------------------------------------------- 2. continuous beam
def test_continuous_two_three_spans():
    r = an.continuous_beam([L, L], [an.uniform(q)], E=E, I=I)
    assert rel(r.reactions["M_2"], -q * L ** 2 / 8) < 1e-12
    assert rel(r.reactions["R_1"], 3 * q * L / 8) < 1e-12 and rel(r.reactions["R_2"], 10 * q * L / 8) < 1e-12
    r = an.continuous_beam([L, L, L], [an.uniform(q)], E=E, I=I)
    assert rel(r.reactions["M_2"], -q * L ** 2 / 10) < 1e-12 and rel(r.reactions["M_3"], -q * L ** 2 / 10) < 1e-12
    assert rel(r.reactions["R_1"], 0.4 * q * L) < 1e-12 and rel(r.reactions["R_2"], 1.1 * q * L) < 1e-12


def test_continuous_vs_macaulay():
    """Constant EI: the three-moment equation = Macaulay with the internal supports (an independent method)."""
    spans, oh = [2000.0, 3000.0, 2500.0], (600.0, 400.0)
    Lt = sum(spans) + sum(oh)
    xs = oh[0] + np.cumsum([0.0] + spans)
    loads = [an.uniform(q, 0, 4000), an.Force(P, 300.0), an.Force(2 * P, 4000.0), an.Moment(4e5, 5000.0),
             an.triangular(1.5, 5000, Lt), an.Force(P, xs[1]), an.Moment(-2e5, xs[2]), an.Force(P, Lt)]
    sett = [0.0, 1.5, -0.5, 0.8]
    r = an.continuous_beam(spans, loads, E=E, I=I, overhang=oh, settlement=sett)
    m = an.beam(Lt, loads, ("free", "free"), E=E, I=I, inner=xs, settlement=sett)
    for xx in np.linspace(0, Lt, 37):
        a, b = r.at(xx), m.at(xx)
        for k in ("w", "phi", "M", "V"):
            assert abs(a[k] - b[k]) <= 1e-9 * (np.abs(getattr(m, k)).max()), (k, xx)
    for i in range(4):
        assert rel(r.reactions[f"R_{i + 1}"], m.reactions[f"R_{i + 1}"]) < 1e-9
    # clamped ends: 2 spans clamped at both ends = Macaulay fixed–fixed with an internal support
    r = an.continuous_beam([2000.0, 3000.0], [an.uniform(q), an.Force(P, 3500.0)], E=E, I=I,
                           left="fixed", right="fixed")
    m = an.beam(5000.0, [an.uniform(q), an.Force(P, 3500.0)], "fixed", E=E, I=I, inner=[2000.0])
    assert rel(r.reactions["M_1"], m.reactions["M_A"]) < 1e-10
    assert rel(r.reactions["M_3"], m.reactions["M_B"]) < 1e-10
    assert rel(r.reactions["R_2"], m.reactions["R_1"]) < 1e-10
    # one span clamped–pinned = propped
    r = an.continuous_beam([L], [an.uniform(q)], E=E, I=I, left="fixed")
    assert rel(r.reactions["M_1"], -q * L ** 2 / 8) < 1e-12 and rel(r.reactions["R_2"], 3 * q * L / 8) < 1e-12


def test_continuous_variable_EI():
    """Different EI per span: the result satisfies all the conditions (supports, continuity of φ, equilibrium)."""
    spans, Is = [2000.0, 3000.0, 1500.0], [8e6, 3e6, 12e6]
    loads = [an.uniform(q), an.Force(P, 2600.0), an.Moment(3e5, 6000.0)]
    sett = [0.0, 2.0, 0.0, -1.0]
    r = an.continuous_beam(spans, loads, E=E, I=Is, right="fixed", overhang=(500.0, 0.0), settlement=sett)
    xs = r.info["supports_x"]
    for xi, d in zip(xs, sett):
        assert abs(r.at(xi)["w"] - d) < 1e-9
        assert abs(r.at(xi - 1e-7)["phi"] - r.at(xi + 1e-7)["phi"]) < 1e-9 or xi == xs[-1]
    assert abs(r.at(xs[-1])["phi"]) < 1e-12                       # clamping on the right
    assert abs(r.at(0.0)["M"]) < 1e-6 and abs(r.at(0.0)["V"]) < 1e-9   # free end of the overhang
    Ltot = r.info["length"]
    Fz = q * Ltot + P
    R = [r.reactions[f"R_{i + 1}"] for i in range(4)]
    assert rel(sum(R), Fz) < 1e-12
    # moment equilibrium about x = 0, positive clockwise: the load down +F·x, right-handed
    # moment +C, reaction upwards −R·x, the clamping on the right acts on the beam with the moment M_4 opposed → −M_4
    Mz = q * Ltot ** 2 / 2 + P * 2600.0 + 3e5 - sum(Ri * xi for Ri, xi in zip(R, xs)) - r.reactions["M_4"]
    assert abs(Mz) < 1e-9 * q * Ltot ** 2
    # EI·w'' = −M inside span 2 (check by a numerical derivative)
    x0, h = 3800.0, 1e-1
    w2 = (r.at(x0 + h)["w"] - 2 * r.at(x0)["w"] + r.at(x0 - h)["w"]) / h ** 2
    assert rel(-E * Is[1] * w2, r.at(x0)["M"]) < 1e-5


# ---------------------------------------------------------------------- 3. curved bar
def test_winkler_rect_roark():
    c = an.curved_beam(an.rect(10, 100), M=1e6, R=100)       # R/h = 1 (R/c = 2)
    assert c.K_i == pytest.approx(1.52, abs=0.005) and c.K_o == pytest.approx(0.73, abs=0.005)
    assert rel(c.r_n, 100 / log(3)) < 1e-13
    assert c.sigma_i < 0 < c.sigma_o                          # M > 0 increases the curvature: the outer fibres in tension
    assert rel(c.sigma_i, c.K_i * c.sigma_i_straight) < 1e-12
    assert rel(float(c.stress(c.r_o)), c.sigma_o) < 1e-12 and abs(float(c.stress(c.r_n))) < 1e-9
    c2 = an.curved_beam(an.rect(10, 100), M=0, N=5000, R=100)
    assert rel(c2.sigma_i, 5000 / 1000) < 1e-12


def test_winkler_closed_forms():
    ri, h = 40.0, 60.0
    ro = ri + h
    c = an.curved_beam(an.trapezoid(30, 10, h), r_inner=ri)
    S = 10 - 30 + (30 * ro - 10 * ri) / h * log(ro / ri)      # Shigley tab. 3-4
    assert rel(c.r_n, 20 * h / S) < 1e-12
    c = an.curved_beam(an.circle(50), r_inner=ri)
    assert rel(c.r_n, (sqrt(ri + 50) + sqrt(ri)) ** 2 / 4) < 1e-12
    c = an.curved_beam(an.tee(40, 10, 8, 60), r_inner=ri)
    assert rel(c.r_n, 800 / (40 * log(50 / 40) + 8 * log(100 / 50))) < 1e-12
    c = an.curved_beam(an.tube(50, 5), R=60)
    S = 2 * pi * ((60 - sqrt(60 ** 2 - 25 ** 2)) - (60 - sqrt(60 ** 2 - 20 ** 2)))
    assert rel(c.r_n, pi * (25 ** 2 - 20 ** 2) / S) < 1e-12
    # I of the section = the table of strips = polygon
    ci = an.curved_beam(an.ishape(80, 50, 10, 30, 8, 6), r_inner=30)
    ct = an.curved_beam(an.strip_table([(0, 10, 50, 50), (10, 72, 6, 6), (72, 80, 30, 30)]), r_inner=30)
    pts = [(-25, 0), (25, 0), (25, 10), (3, 10), (3, 72), (15, 72), (15, 80), (-15, 80), (-15, 72), (-3, 72),
           (-3, 10), (-25, 10)]
    cp = an.curved_beam(an.polygon(pts), r_inner=30)
    assert rel(ci.e, ct.e) < 1e-12 and rel(ci.e, cp.e) < 1e-10 and rel(ci.I, cp.I) < 1e-10
    cs = an.curved_beam(an.strip_table([(0, 30), (60, 10)]), r_inner=ri)   # samples of the width = trapezium
    assert rel(cs.r_n, 20 * h / (10 - 30 + (30 * ro - 10 * ri) / h * log(ro / ri))) < 1e-12


def test_winkler_large_radius_stable():
    for R in (1e3, 1e6, 1e9):
        c = an.curved_beam(an.rect(10, 1.0), M=1.0, R=R)
        assert rel(c.e, 1.0 / (12 * R)) < 1e-7                # e → I/(A·R)
        assert c.warnings and abs(c.K_i - 1) < 1e-3
    assert not an.curved_beam(an.rect(10, 100), R=300).warnings
    with pytest.raises(ValueError):
        an.curved_beam(an.rect(10, 100), R=40)                # inner fibre r < 0


# ---------------------------------------------------------------------- 4. arcs and rings
R, F = 200.0, 100.0


def test_arc_quarter_and_half():
    a = an.arc_cantilever(R, pi / 2, Fr=F, E=E, I=I)
    assert rel(a.u_r, pi * F * R ** 3 / (4 * EI)) < 1e-12
    assert rel(a.u_t, -F * R ** 3 / (2 * EI)) < 1e-12
    assert rel(a.rot, -F * R ** 2 / EI) < 1e-12
    assert rel(a.ux, F * R ** 3 / (2 * EI)) < 1e-12 and rel(a.uy, a.u_r) < 1e-12
    assert rel(abs(a.M_fixed), F * R) < 1e-12
    a = an.arc_cantilever(R, pi / 2, Ft=F, E=E, I=I)          # ∫(1 − sin φ)² = 3π/4 − 2
    assert rel(a.u_t, (3 * pi / 4 - 2) * F * R ** 3 / EI) < 1e-12
    h = an.arc_cantilever(R, pi, Fr=F, E=E, I=I)
    assert rel(h.u_r, pi * F * R ** 3 / (2 * EI)) < 1e-12
    h = an.arc_cantilever(R, pi, Ft=F, E=E, I=I)
    assert rel(h.u_t, 1.5 * pi * F * R ** 3 / EI) < 1e-12
    C = an.arc_cantilever(R, pi, E=E, I=I).compliance        # symetrie poddajnosti (Maxwell)
    assert np.allclose(C, C.T, rtol=1e-12, atol=0)
    m = an.arc_cantilever(R, pi / 3, M=1e4, E=E, I=I)        # moment: rotation M·Rθ/EI
    assert rel(m.rot, 1e4 * R * pi / 3 / EI) < 1e-12


def test_arc_axial_shear_and_straight_limit():
    A, G, As = 600.0, 80769.0, 500.0
    a = an.arc_cantilever(R, pi / 2, Fr=F, E=E, I=I, A=A, axial=True, shear=True, G=G, A_s=As)
    ref = pi * F * R ** 3 / (4 * EI) + pi * R * F / (4 * E * A) + pi * R * F / (4 * G * As)
    assert rel(a.u_r, ref) < 1e-12
    Ls, Rb = 1000.0, 1e7                                     # nearly straight bar → cantilever FL³/3EI
    s = an.arc_cantilever(Rb, Ls / Rb, Fr=F, E=E, I=I)
    assert rel(s.u_r, F * Ls ** 3 / (3 * EI)) < 1e-4


def test_ring_pinch():
    r = an.ring_pinch(F, R, E=E, I=I)
    assert rel(r.dD_load, -0.1488 * F * R ** 3 / EI) < 2e-4 and rel(r.dD_perp, 0.1366 * F * R ** 3 / EI) < 2e-4
    assert rel(r.dD_load, -(pi / 4 - 2 / pi) * F * R ** 3 / EI) < 1e-12
    assert rel(r.dD_perp, (2 / pi - 0.5) * F * R ** 3 / EI) < 1e-12
    assert rel(r.M_load, -F * R / pi) < 1e-12 and rel(r.M_side, F * R * (0.5 - 1 / pi)) < 1e-12


def test_ring_vs_arc():
    """A ring with tension and shear = a quarter as a cantilever arc (clamping at the side, rotation at the top 0)."""
    kw = dict(E=E, I=I, A=600.0, axial=True, shear=True, A_s=500.0, nu=0.3)
    r = an.ring_pinch(F, R, **kw)
    a0 = an.arc_cantilever(R, pi / 2, Fr=-F / 2, **kw)
    a1 = an.arc_cantilever(R, pi / 2, M=1.0, **kw)
    Mt = -a0.rot / a1.rot
    ur, ut = a0.u_r + Mt * a1.u_r, a0.u_t + Mt * a1.u_t
    assert rel(2 * ur, r.dD_load) < 1e-12 and rel(2 * ut, r.dD_perp) < 1e-12


# ---------------------------------------------------------------------- 5. buckling
def test_euler():
    for case, b in (("fixed-free", 2.0), ("pinned-pinned", 1.0), ("fixed-fixed", 0.5)):
        r = an.euler_buckling(L, E=E, I=I, case=case)
        assert rel(r.F_cr, pi ** 2 * EI / (b * L) ** 2) < 1e-14
    r = an.euler_buckling(L, E=E, I=I, case="fixed-pinned", A=5000.0, fy=235.0)
    kl = pi / r.beta
    assert abs(tan(kl) - kl) < 1e-9 and r.beta == pytest.approx(0.6992, abs=1e-4)
    assert rel(r.slenderness, r.L_cr / sqrt(I / 5000.0)) < 1e-14
    assert rel(r.rel_slenderness, sqrt(5000.0 * 235.0 / r.F_cr)) < 1e-14
    sec = SectionProps("x", "I", A=2850.0, Iy=19.43e6, Iz=1.42e6)
    assert rel(an.euler_buckling(L, E=E, section=sec).F_cr, pi ** 2 * E * 1.42e6 / L ** 2) < 1e-14
