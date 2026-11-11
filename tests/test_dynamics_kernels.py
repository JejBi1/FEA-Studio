"""Fast tests of the numerical cores of dynamics (without mesh and ccx): exact solution of the modal equations (modal_ode), PSD and
modal moments (psd), shocks and the response spectrum (shock). Verification cases F8: Y1 (structural damping), Y3 (DLF),
Y4 (SRS of the half-sine), Y5 (Miles), Y6 (decay), Y10 (run-up of an axis), Y12 (CQC).
"""
import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from fea_core.dynamics import modal_ode, psd, shock

TWO_PI = 2.0 * np.pi


# ------------------------------------------------------------------ modal_ode

@pytest.mark.parametrize("zeta", [0.0, 0.02, 0.1])
def test_step_dlf(zeta):
    """Y3: step force on 1 DOF – DLF = 2, damped 1 + exp(−πζ/√(1 − ζ²)) (ζ 2 % → 1.939)."""
    w = TWO_PI * 10.0
    dt = 0.1 / 4000
    q, _, _ = modal_ode.integrate(dt, np.ones(int(0.3 / dt)), w, zeta)
    dlf = q.max() * w * w
    assert dlf == pytest.approx(1.0 + math.exp(-math.pi * zeta / math.sqrt(1 - zeta * zeta)), rel=1e-6)
    if zeta == 0.02:
        assert dlf == pytest.approx(1.939, abs=5e-4)


def test_free_decay_log_decrement():
    """Y6: free decay – q(t + T_d)/q(t) = e^(−ζωT_d), δ = 2πζ/√(1 − ζ²)."""
    fn, z = 25.0, 0.0197
    w = TWO_PI * fn
    Td = 1.0 / (fn * math.sqrt(1 - z * z))
    n = 400
    dt = Td / n
    q, v, a = modal_ode.integrate(dt, np.zeros(5 * n + 1), w, z, q0=0.0, v0=1.0)
    i = n // 7
    delta = math.log(q[i] / q[i + n])
    assert delta == pytest.approx(2 * math.pi * z / math.sqrt(1 - z * z), rel=1e-9)
    assert np.allclose(a, -(2 * z * w * v + w * w * q), rtol=0, atol=1e-9 * np.abs(a).max())


def test_harmonic_steady_amplitude():
    """Harmonic excitation – the steady amplitude = |H| = 1/√((ω_n² − Ω²)² + (2ζω_nΩ)²)."""
    fn, z = 10.0, 0.05
    w = TWO_PI * fn
    for fe in (6.0, 10.0, 17.0):
        W = TWO_PI * fe
        dt = 1.0 / fe / 2000
        t = np.arange(int(12.0 / dt)) * dt                  # ζωt ≈ 38 → the transient has decayed
        q, _, _ = modal_ode.integrate(dt, np.sin(W * t), w, z)
        amp = np.abs(q[-int(3 / fe / dt):]).max()
        assert amp == pytest.approx(1.0 / math.hypot(w * w - W * W, 2 * z * w * W), rel=2e-5)


def test_against_solve_ivp():
    """Random piecewise linear excitation, non-zero initial state, ζ = 0, rigid mode and ζ > 1: against solve_ivp."""
    rng = np.random.default_rng(1)
    dt, T = 1e-3, 400
    P = rng.normal(size=(T, 4))
    om = np.array([TWO_PI * 7.3, TWO_PI * 55.0, 0.0, TWO_PI * 3.0])
    ze = np.array([0.03, 0.0, 0.0, 1.5])
    q0 = np.array([0.01, -0.02, 0.1, 0.05])
    v0 = np.array([0.5, 0.3, -0.2, 0.0])
    q, v, a = modal_ode.integrate(dt, P, om, ze, q0=q0, v0=v0)
    t = np.arange(T) * dt
    for j in range(4):
        def rhs(tt, y, j=j):
            p = np.interp(tt, t, P[:, j])
            return [y[1], p - 2 * ze[j] * om[j] * y[1] - om[j] ** 2 * y[0]]
        s = solve_ivp(rhs, (0, t[-1]), [q0[j], v0[j]], t_eval=t, rtol=1e-12, atol=1e-14, method="DOP853",
                      max_step=dt)
        assert np.abs(s.y[0] - q[:, j]).max() <= 1e-8 * np.abs(q[:, j]).max()
        assert np.abs(s.y[1] - v[:, j]).max() <= 1e-8 * np.abs(v[:, j]).max()


def test_rigid_mode_double_integration():
    """ω = 0: q = double integral of a piecewise linear p (p = t → q = t³/6)."""
    dt = 0.01
    t = np.arange(101) * dt
    q, v, a = modal_ode.integrate(dt, t, 0.0, 0.0)
    assert np.allclose(q, t ** 3 / 6, rtol=1e-12, atol=1e-15)
    assert np.allclose(v, t ** 2 / 2, rtol=1e-12, atol=1e-15)
    assert np.allclose(a, t)


@pytest.mark.parametrize("tau_T", [0.3, 0.5, 0.77, 1.0, 2.0])
def test_rect_pulse_residual(tau_T):
    """Y10: residual amplitude after a rectangular pulse of the base acceleration X = (2a/ω²)|sin(ωτ/2)| (ζ = 0)."""
    fn, acc = 40.0, 2000.0
    w = TWO_PI * fn
    tau = tau_T / fn
    dt = tau / 4000
    n = int(round(tau / dt))
    a_b = np.zeros(n + 2000)
    a_b[:n + 1] = acc                                     # the drop down is spread over one step dt
    q, v, _ = modal_ode.integrate_base(dt, a_b, [1.0], [w], 0.0)
    X = math.hypot(q[-1, 0], v[-1, 0] / w)
    X_ref = 2 * acc / w ** 2 * abs(math.sin(w * (tau + 0.5 * dt) / 2))   # the pulse ends in the middle of the ramp
    assert X == pytest.approx(X_ref, rel=1e-5, abs=1e-9 * acc / w ** 2)
    assert X == pytest.approx(2 * acc / w ** 2 * abs(math.sin(w * tau / 2)), rel=5e-3, abs=5e-3 * acc / w ** 2)


def test_integrate_base_sign():
    """p_j = −Γ_j·a_b: quasi-static response q = −Γ·a/ω²."""
    w = np.array([TWO_PI * 100.0, TWO_PI * 300.0])
    g = np.array([2.0, -0.5])
    dt = 1e-4
    a_b = np.minimum(np.arange(20000) * dt / 0.5, 1.0) * 9810.0  # slow ramp-up
    q, _, _ = modal_ode.integrate_base(dt, a_b, g, w, 0.05)
    assert np.allclose(q[-1], -g * 9810.0 / w ** 2, rtol=1e-4)


# ------------------------------------------------------------------ psd: profil ASD

def test_asd_interp_loglog():
    t = [[20.0, 0.01], [80.0, 0.04], [2000.0, 0.04]]
    W = psd.asd_interp(t, [10.0, 20.0, 40.0, 80.0, 500.0, 2000.0, 2500.0])
    assert W[0] == 0 and W[-1] == 0
    assert W[2] == pytest.approx(0.02, rel=1e-12)          # geometric centre log-log
    assert W[4] == pytest.approx(0.04)


def test_asd_from_slopes_and_rms():
    pts = [[20.0, None], [80.0, 0.04], [350.0, 0.04], [2000.0, None]]
    t = psd.asd_from_slopes(pts, [3.0, None, -6.0])
    assert t[0, 1] == pytest.approx(0.04 * (20 / 80) ** (3 / psd.LOG2_DB))
    assert t[3, 1] == pytest.approx(0.04 * (2000 / 350) ** (-6 / psd.LOG2_DB))
    # exact integral log-log against a fine numerical one
    f = np.geomspace(20, 2000, 400001)
    num = math.sqrt(np.trapz(psd.asd_interp(t, f), f))
    assert psd.asd_rms(t) == pytest.approx(num, rel=1e-8)
    t2 = psd.asd_from_slopes(pts, [3.0, None, -6.0], grms=5.0)
    assert psd.asd_rms(t2) == pytest.approx(5.0, rel=1e-12)
    with pytest.raises(ValueError):
        psd.asd_from_slopes([[20.0, None], [80.0, None]], [3.0])


def test_miles_and_abs_acc_rms():
    """Y5: Miles 7.9266 g (100 Hz, ζ 5 %, 0.04 g²/Hz); modally the absolute acceleration = Miles·√(1 + 4ζ²) (white noise)."""
    fn, z, W = 100.0, 0.05, 0.04
    g_m = psd.miles(fn, 1 / (2 * z), W)
    assert g_m == pytest.approx(7.9266, abs=1e-4)
    f = psd.freq_grid(0.05, 20000.0, [fn], z)
    M = psd.modal_moments(f, [fn], z, [1.0], psd.asd_to_mm(np.full(f.shape, W)), powers=(0,), kind="acc_abs")
    g_rms = math.sqrt(M[0, 0, 0].real) / psd.G_MM
    assert g_rms == pytest.approx(g_m * math.sqrt(1 + 4 * z * z), rel=3e-4)
    assert g_rms == pytest.approx(g_m, rel=5e-3)
    # 3σ: share of the time above 3σ for a Gaussian process
    from scipy.stats import norm
    assert 2 * norm.sf(3.0) == pytest.approx(0.0027, abs=1e-4)


@pytest.mark.parametrize("zeta", [0.005, 0.02, 0.1])
def test_freq_grid_integrates_resonance(zeta):
    """The trapezium on freq_grid integrates |H|² and f²|H|² of the resonance with an error < 3e-4."""
    fn = 100.0
    wn = TWO_PI * fn
    f = psd.freq_grid(1.0, 10000.0, [fn], zeta)
    M = psd.modal_moments(f, [fn], zeta, [1.0], np.ones_like(f), powers=(0, 2))

    def h2(x):
        w = TWO_PI * x
        return 1.0 / ((wn * wn - w * w) ** 2 + (2 * zeta * wn * w) ** 2)
    for i, p in enumerate((0, 2)):
        ex = quad(lambda x: x ** p * h2(x), 1.0, 10000.0, points=[fn], limit=500, epsabs=0, epsrel=1e-12)[0]
        assert M[i, 0, 0].real == pytest.approx(ex, rel=3e-4)


def test_ssd_grid_matches_ccx():
    """Points of *STEADY STATE DYNAMICS identical with ccx 2.23 (attempt scratch/f8_kernels/ssd_bias.py, C3D8, natural
    frequencies 65 261.97 (double) and 88 423.74 Hz).
    """
    eig = [65261.97, 65261.97, 88423.74, 146897.2]
    cases = {
        (5, 1.0): [32630.99, 40788.73, 48946.48, 57104.22, 65261.97, 68525.07, 71788.17, 75051.27, 78314.36],
        (5, 3.0): [32630.99, 35996.86, 48946.48, 61896.09, 65261.97, 66608.32, 71788.17, 76968.01, 78314.36],
        (6, 2.0): [32630.99, 36308.55, 41649.97, 56242.99, 61584.4, 65261.97, 66733.0, 68869.56, 74706.77,
                   76843.34, 78314.36],
        (7, 3.0): [32630.99, 34693.58, 37633.95, 48946.48, 60259.01, 63199.37, 65261.97, 66087.01, 67263.15,
                   71788.17, 76313.18, 77489.33, 78314.36],
    }
    for (n, b), ref in cases.items():
        assert np.allclose(psd.ssd_grid(32630.99, 78314.36, eig, n, b), ref, rtol=1e-6)   # the .dat has 7 digits
    # no natural frequencies in the range (n = 5, bias 3)
    assert np.allclose(psd.ssd_grid(6526.197, 32630.99, eig, 5, 3.0),
                       [6526.197, 9218.899, 19578.59, 29938.28, 32630.99], rtol=1e-6)
    m = 2
    assert psd.ssd_grid(10.0, 1000.0, [100.0, 300.0], 20, 3.0).size == 20 * m - m + 20


def test_structural_damping_peak():
    """Y1: structural damping η – the peak 1/(ω_n²η); η = 2ζ gives the same peak as the viscous ζ."""
    fn, z = 50.0, 0.02
    wn = TWO_PI * fn
    Hs = psd.modal_frf([fn], [fn], eta=2 * z)[0, 0]
    Hv = psd.modal_frf([fn], [fn], z)[0, 0]
    assert abs(Hs) == pytest.approx(1 / (wn * wn * 2 * z), rel=1e-12)
    assert abs(Hs) == pytest.approx(abs(Hv), rel=1e-12)
    assert np.angle(Hv, deg=True) == pytest.approx(-90.0, abs=1e-9)
    r = 0.7
    Hr = psd.modal_frf([r * fn], [fn], z)[0, 0] * wn * wn
    assert abs(Hr) == pytest.approx(1 / math.sqrt((1 - r * r) ** 2 + (2 * z * r) ** 2), rel=1e-12)
    assert -np.angle(Hr) == pytest.approx(math.atan2(2 * z * r, 1 - r * r), rel=1e-12)


def test_modal_moments_equal_direct_psd():
    """Quadratic form Aᵀ·Re(M^(p))·A = ∫ f^p·|Σ A_j Z_j|²·W df (CQC in the full form)."""
    rng = np.random.default_rng(2)
    fn = np.array([80.0, 86.0, 240.0])
    ze = np.array([0.02, 0.03, 0.05])
    g = rng.normal(size=3)
    f = psd.freq_grid(20.0, 1000.0, fn, ze)
    W = psd.asd_to_mm(psd.asd_interp([[20, 0.01], [100, 0.04], [1000, 0.01]], f))
    for kind in psd.KINDS:
        M = psd.modal_moments(f, fn, ze, g, W, kind=kind)
        A = rng.normal(size=(5, 3))
        S = psd.response_psd(f, fn, ze, g, W, A, kind=kind)
        qf = psd.quad_forms(A[None], M)[0]
        for i, p in enumerate((0, 1, 2, 4)):
            assert np.allclose(qf[:, i], np.trapz(f[:, None] ** p * S, f, axis=0), rtol=1e-11)
        assert np.allclose(psd.response_rms(A[None], M[0])[0], np.sqrt(qf[:, 0]), rtol=1e-12)


def test_equivalent_moments_preumont():
    """m_p of the equivalent = Σ Q_ab Sᵀ_a Re(M) S_b (numpy einsum), float32 and float64; uniaxial state = RMS of the component."""
    rng = np.random.default_rng(3)
    K = 7
    fn = np.sort(rng.uniform(50, 500, K))
    f = psd.freq_grid(10.0, 1000.0, fn, 0.03)
    M = psd.modal_moments(f, fn, 0.03, rng.normal(size=K), np.ones_like(f))
    S = rng.normal(size=(40, 6, K))
    for dt in (np.float64, np.float32):
        Sx = S.astype(dt)
        m = psd.equivalent_moments(Sx, M)
        ref = np.einsum("ab,naj,pjk,nbk->np", psd.Q_MISES, Sx.astype(float), M.real, Sx.astype(float))
        assert np.allclose(m, ref, rtol=1e-10)
        r = psd.rms_components(Sx, M[0])
        assert np.allclose(r ** 2, np.einsum("naj,jk,nak->na", Sx.astype(float), M[0].real, Sx.astype(float)),
                           rtol=1e-10)
    S1 = np.zeros((1, 6, K))
    S1[0, 0] = S[0, 0]
    assert psd.vm_rms_segalman(S1, M[0])[0] == pytest.approx(psd.rms_components(S1, M[0])[0, 0], rel=1e-12)


# ------------------------------------------------------------------ shock

def test_pulses():
    A, tau, dt = 50.0, 0.011, 1e-5
    t, a = shock.pulse("halfsine", A, tau, dt, t_end=0.03)
    assert a.max() == pytest.approx(A, rel=1e-6)
    assert np.trapz(a, t) == pytest.approx(2 * A * tau / math.pi, rel=1e-6)
    t, a = shock.pulse("sawtooth", A, tau, dt)
    assert a[int(round(tau / dt))] == pytest.approx(A) and a[-1] == 0
    t, a = shock.pulse("trapezoid", A, tau, dt, rise=0.002)
    assert np.trapz(a, t) == pytest.approx(A * (tau - 0.002), rel=1e-6)
    d = shock.drop_pulse(1000.0, 1e-3, 1e4, 0.0)
    assert d["v"] == pytest.approx(4429.4, rel=1e-4)            # 4,43 m/s z 1 m
    assert d["tau"] == pytest.approx(math.pi * math.sqrt(1e-3 / 1e4))
    assert 2 * d["A"] * d["tau"] / math.pi == pytest.approx(d["v"])


def _halfsine_srs_exact0(fnt, tau=0.01):
    """Exact SRS of the half-sine for ζ = 0 (maximax of the absolute acceleration / A): the primary from the closed form, the residual
    amplitude from the state at t = τ.
    """
    w = TWO_PI * fnt / tau
    W = math.pi / tau
    tt = np.linspace(0, tau, 400001)
    k = w * w / (w * w - W * W)
    y = k * (np.sin(W * tt) - W / w * np.sin(w * tt))
    yd = k * (W * math.cos(W * tau) - W * math.cos(w * tau))
    return max(np.abs(y).max(), math.hypot(y[-1], yd / w))


def test_srs_halfsine_Y4():
    """Y4: half-sine → max SRS 1.768·A at f_nτ = 0.81 (ζ = 0), 1.652 at 0.82 (ζ = 5 %)."""
    tau = 0.01
    t, a = shock.pulse("halfsine", 1.0, tau, tau / 2000, t_end=1.5 * tau)
    fnt = np.arange(0.70, 0.95, 0.0025)
    r0 = shock.srs(t, a, fnt / tau, 0.0)
    i = r0["maximax"].argmax()
    assert r0["maximax"][i] == pytest.approx(1.768, rel=5e-3)
    assert fnt[i] == pytest.approx(0.81, abs=0.01)
    assert r0["maximax"][i] == pytest.approx(1.7684577, rel=1e-5)      # analytic maximum (ζ = 0)
    r5 = shock.srs(t, a, fnt / tau, 0.05)
    i = r5["maximax"].argmax()
    assert r5["maximax"][i] == pytest.approx(1.652, rel=5e-3)
    assert fnt[i] == pytest.approx(0.82, abs=0.01)
    # the whole spectrum ζ = 0 against the exact solution (without f_nτ = 0.5 – resonance in closed form)
    fs = np.array([0.1, 0.3, 0.45, 0.6, 0.81, 1.2, 2.0, 5.0])
    r = shock.srs(t, a, fs / tau, 0.0)
    ex = np.array([_halfsine_srs_exact0(x) for x in fs])
    assert np.allclose(r["maximax"], ex, rtol=2e-4)
    # low frequencies: the residual response dominates, high: the primary → A
    assert r["residual"][0] > r["primary"][0]
    assert 1.0 < r["maximax"][-1] < 1.15


def test_srs_equals_smallwood_and_substeps():
    """The same results as the Smallwood ramp-invariant filter; the division of the step does not change the excitation (piecewise linear)."""
    dt = 1e-4
    t, a = shock.pulse("halfsine", 1.0, 0.01, dt, t_end=0.05)
    for fn, z in ((50.0, 0.05), (81.0, 0.0), (400.0, 0.03)):
        w = TWO_PI * fn
        E = math.exp(-z * w * dt)
        K = w * math.sqrt(1 - z * z) * dt
        C, Sp = E * math.cos(K), E * math.sin(K) / K
        b0, b1, b2, a1, a2 = 1 - Sp, 2 * (Sp - C), E * E - Sp, 2 * C, -E * E
        y = np.zeros_like(a)
        for i in range(len(a)):
            y[i] = (b0 * a[i] + (b1 * a[i - 1] + a1 * y[i - 1] if i >= 1 else 0.0)
                    + (b2 * a[i - 2] + a2 * y[i - 2] if i >= 2 else 0.0))
        r = shock.srs(t, a, [fn], z, points_per_period=4, tail_periods=0.0)
        assert r["maximax"][0] == pytest.approx(np.abs(y).max(), rel=1e-11)
    # coarse record: division of the step = calculation on the linearly refined record
    tc, ac = shock.pulse("halfsine", 1.0, 0.01, 1e-3, t_end=0.05)
    tf = np.arange(0, tc[-1] + 1e-12, 1e-5)
    af = np.interp(tf, tc, ac)
    f = np.array([20.0, 150.0, 900.0])
    rc = shock.srs(tc, ac, f, 0.05, points_per_period=200)
    rf = shock.srs(tf, af, f, 0.05, points_per_period=200)
    assert np.allclose(rc["maximax"], rf["maximax"], rtol=2e-4)


def test_cqc_rho_vs_white_noise_Y12():
    """Y12: ρ_ij of Der Kiureghian = correlation of the modal displacements under white noise (exactly from the modal moments) – 2 %."""
    for fn, z in (([100.0, 105.0], [0.05, 0.05]), ([100.0, 110.0], [0.02, 0.05]), ([100.0, 130.0], [0.05, 0.05])):
        rho = shock.cqc_rho(fn, z)
        f = psd.freq_grid(1.0, 5000.0, fn, z, per_peak=81, ratio=1.01)
        M = psd.modal_moments(f, fn, z, [1.0, 1.0], np.ones_like(f), powers=(0,))[0].real
        r_ex = M[0, 1] / math.sqrt(M[0, 0] * M[1, 1])
        assert rho[0, 1] == pytest.approx(r_ex, rel=0.02, abs=2e-3)
    rho = shock.cqc_rho([10.0, 10.0, 100.0], 0.05)
    assert np.allclose(np.diag(rho), 1.0) and rho[0, 1] == pytest.approx(1.0) and rho[0, 2] < 0.01
    assert np.allclose(rho, rho.T)


def test_cqc_time_domain_Y12():
    """Y12 in time: two close modes excited by broadband noise – the sample correlation q₁, q₂ corresponds to ρ (CQC) and the RMS of the
    sum = the CQC combination of the RMS of the modes.
    """
    rng = np.random.default_rng(11)
    fn = np.array([100.0, 106.0])
    z = 0.05
    dt = 1.0 / 4000
    n = 2_000_000                                          # 500 s
    a = rng.normal(size=n)
    q, _, _ = modal_ode.integrate_base(dt, a, [1.0, 1.0], TWO_PI * fn, z)
    q = q[4000:]
    s = q.std(axis=0)
    r_t = np.mean(q[:, 0] * q[:, 1]) / (s[0] * s[1])
    rho = shock.cqc_rho(fn, z)
    assert r_t == pytest.approx(rho[0, 1], abs=0.03)
    comb = shock.combine(s[:, None], rho)[0]
    assert comb == pytest.approx((q[:, 0] + q[:, 1]).std(), rel=0.02)


def test_combine():
    rng = np.random.default_rng(4)
    R = rng.normal(size=(5, 7, 3))
    assert np.allclose(shock.combine(R), np.sqrt((R ** 2).sum(0)))
    assert np.allclose(shock.combine(R, np.eye(5)), np.sqrt((R ** 2).sum(0)))
    assert np.allclose(shock.combine(R, np.ones((5, 5))), np.abs(R.sum(0)))
    qp = shock.modal_peaks([10.0], [2.0], [9810.0])
    assert qp[0] == pytest.approx(2.0 * 9810.0 / (TWO_PI * 10.0) ** 2)
    assert np.allclose(shock.srs_interp([[10, 5.0], [1000, 50.0]], [100.0]), [5.0 * 10 ** 0.5])
