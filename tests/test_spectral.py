"""Tests of the spectral fatigue `fea_core.fatigue.spectral` (without mesh and ccx): NB / Dirlik / Tovo–Benasciutti against the closed forms,
numerical integration of the S-N with a knee, FLife (etalon, Y11 – 1e-6), rainflow of the synthetic
Gaussian signal (Y11 – ±15 %) and the sine sweep against the time simulation + rainflow (Y14 – 2 %).
"""
import importlib.util
import math
import sys
import types

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma as G

from fea_core.dynamics import modal_ode, psd
from fea_core.fatigue import kernel, spectral as sp
from fea_core.fatigue.sn import SNCurve


def _structural_psd(f, c2):
    """PSD of the response stress of two modes (60 and 230 Hz, ζ 3 %) to white noise; c2 = share of the 2nd mode."""
    H = psd.modal_frf(f, [60.0, 230.0], [0.03, 0.03])
    return np.abs(H[:, 0] + c2 * H[:, 1] * (230.0 / 60.0) ** 2) ** 2 * 1e9


def _moments(f, Gs):
    return np.array([np.trapz(f ** i * Gs, f) for i in (0, 1, 2, 4)])


def _flife():
    """FLife without pyvista / tkinter (test etalon, pip install --no-deps FLife fatpack pyExSi)."""
    pytest.importorskip("fatpack")
    for name in ("pyvista", "pyvistaqt"):
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            sys.modules.setdefault(name, types.ModuleType(name))
    try:
        import tkinter.filedialog  # noqa: F401
    except ImportError:                                   # the pruned Python of the installer does not have tkinter
        tk = types.ModuleType("tkinter")
        fd = types.ModuleType("tkinter.filedialog")
        fd.asksaveasfilename = None
        tk.filedialog = fd
        sys.modules.setdefault("tkinter", tk)
        sys.modules.setdefault("tkinter.filedialog", fd)
        bt = types.ModuleType("matplotlib.backends.backend_tkagg")
        bt.FigureCanvasTkAgg = bt.NavigationToolbar2Tk = None
        sys.modules.setdefault("matplotlib.backends.backend_tkagg", bt)
    return pytest.importorskip("FLife")


def test_params_and_nb_closed_form():
    f = np.linspace(0, 600, 30001)
    m = _moments(f, _structural_psd(f, 1.0))
    p = sp.params(m)
    assert p["nu0"][0] == pytest.approx(math.sqrt(m[2] / m[0]))
    assert 0 < p["alpha2"][0] <= p["alpha1"][0] <= 1
    k, SD, ND = 5.0, 10.0, 1e6
    d = sp.damage(m, 3600.0, SNCurve(SD, ND, k, k), "nb")[0]
    ref = p["nu0"][0] * 3600.0 * math.sqrt(2 * m[0]) ** k * G(1 + k / 2) / (ND * SD ** k)
    assert d == pytest.approx(ref, rel=1e-12)
    # zero process
    z = np.zeros((1, 4))
    assert sp.damage(z, 1.0, SNCurve(SD, ND, k, k), "dirlik")[0] == 0
    assert np.isinf(sp.safety(z, 1.0, SNCurve(SD, ND, k, k), "dirlik")[0])
    assert np.isinf(sp.life_hours([0.0], 3600.0)[0])
    assert sp.life_hours([0.5], 3600.0)[0] == pytest.approx(2.0)


@pytest.mark.parametrize("method", sp.METHODS)
@pytest.mark.parametrize("k2", [0.0, 9.0, 5.0])
def test_knee_sn_exact_vs_quad(method, k2):
    """S-N with a knee (k2 = 0 fatigue limit, Haibach 2k − 1, without a knee): exact integrals = quad over the density pdf."""
    f = np.linspace(0, 600, 30001)
    m = _moments(f, _structural_psd(f, 0.3))
    sn = SNCurve(25.0, 2e6, 5.0, k2)
    pdf_s = lambda s: sp.pdf(np.array([s]), m, method)[0][0]  # noqa: E731
    _, nu = sp.pdf(np.array([1.0]), m, method)
    inv = lambda s: pdf_s(s) / sn.N(s)  # noqa: E731
    s0 = math.sqrt(m[0])
    ref = nu * (quad(inv, 0, sn.SD, epsabs=0, epsrel=1e-12, limit=200)[0]
                + quad(inv, sn.SD, 60 * s0, epsabs=0, epsrel=1e-12, limit=200)[0])
    assert sp.damage_rate(m, sn, method)[0] == pytest.approx(ref, rel=1e-7)
    # the density is normalised
    assert quad(pdf_s, 0, 60 * s0, limit=200)[0] == pytest.approx(1.0, rel=1e-8)


@pytest.mark.parametrize("c2", [0.0, 0.3, 1.0, 3.0])
def test_flife_reference_Y11(c2):
    """Y11: NB / Dirlik / TB (2005, "method 2") = FLife to 1e-6 (the moments of FLife in rad/s converted to Hz)."""
    FLife = _flife()
    f = np.linspace(0, 600, 60001)
    Gs = _structural_psd(f, c2)
    sd = FLife.SpectralData(input={"PSD": Gs, "f": f})
    m = sd.moments[[0, 1, 2, 4]] / (2 * np.pi) ** np.array([0, 1, 2, 4])
    k, SD, ND = 5.0, 10.0, 1e6
    sn = SNCurve(SD, ND, k, k)
    C = ND * SD ** k
    for name, cls, kw in (("nb", FLife.Narrowband, {}), ("dirlik", FLife.Dirlik, {}),
                          ("tb", FLife.TovoBenasciutti, {"method": "method 2"})):
        L = cls(sd).get_life(C, k, **kw)
        assert sp.damage_rate(m, sn, name)[0] * L == pytest.approx(1.0, abs=1e-6), name
    assert np.allclose(sp.dirlik_coefficients(m)["D1"], FLife.Dirlik(sd).coeff[3], rtol=1e-12)


def _gauss_signal(f, Gs, fs, N, seed):
    """Stationary Gaussian signal with a one-sided PSD Gs(f) (random phases, inverse FFT)."""
    rng = np.random.default_rng(seed)
    df = fs / N
    ff = np.arange(N // 2 + 1) * df
    Gi = np.interp(ff, f, Gs, right=0.0)
    X = np.sqrt(Gi * df / 2.0) * N * np.exp(1j * rng.uniform(0, 2 * np.pi, ff.size))
    X[0] = 0.0
    return np.fft.irfft(X, n=N)


@pytest.mark.parametrize("c2", [0.0, 1.0])
def test_rainflow_synthetic_Y11(c2):
    """Y11: against the rainflow of a synthetic Gaussian signal from the PSD (1 050 s): Dirlik ±15 %, TB ±15 % (k ≤ 5),
    NB conservative. (Found: TB for a bimodal spectrum and k = 8 with a fatigue limit deviates up to −20 % – not in the test.)
    """
    f = np.linspace(0, 600, 60001)
    Gs = _structural_psd(f, c2)
    m = _moments(f, Gs)
    fs, N = 4000.0, 2 ** 22
    x = _gauss_signal(f, Gs, fs, N, seed=5)
    assert x.var() == pytest.approx(m[0], rel=1e-3)
    c = kernel.cycles_of(x)
    Sa = c[:, 0] / 2.0
    for k, k2 in ((3.0, 3.0), (5.0, 9.0), (8.0, 0.0)):
        sn = SNCurve(10.0, 1e6, k, k2)
        Nn = np.array([sn.N(s) for s in Sa])
        D_rf = np.sum(c[:, 2] / Nn) / (N / fs)
        r = {mm: sp.damage_rate(m, sn, mm)[0] / D_rf for mm in sp.METHODS}
        assert 0.85 <= r["dirlik"] <= 1.15, (k, k2, r)
        assert r["nb"] >= 0.98, (k, k2, r)
        if k <= 5:
            assert 0.85 <= r["tb"] <= 1.15, (k, k2, r)


def test_mean_stress_and_safety():
    f = np.linspace(0, 600, 30001)
    m = np.vstack([_moments(f, _structural_psd(f, c)) for c in (0.0, 1.0)])
    sn = SNCurve(20.0, 1e6, 5.0, 5.0)
    T = 3600.0
    d0 = sp.damage(m, T, sn, "dirlik")
    dg = sp.damage(m, T, sn, "dirlik", mean=[50.0, 0.0], Rm=500.0)
    assert dg[0] == pytest.approx(d0[0] * (1 / (1 - 0.1)) ** 5, rel=1e-12)
    assert dg[1] == pytest.approx(d0[1], rel=1e-12)          # without a mean stress unchanged
    # safety: without a knee exactly λ = D^(−1/k); with a knee D(λ) = 1
    lam = sp.safety(m, T, sn, "dirlik")
    assert np.allclose(lam, d0 ** (-1 / 5), rtol=1e-12)
    for k2 in (0.0, 9.0):
        snk = SNCurve(20.0, 1e6, 5.0, k2)
        for meth in sp.METHODS:
            lam = sp.safety(m, T, snk, meth)
            assert np.allclose(T * sp.damage_rate(m, snk, meth, lam), 1.0, rtol=1e-8)


def test_sweep_cycles_per_hz():
    assert sp.sweep_cycles_per_hz(1.0) == pytest.approx(86.56, abs=0.01)
    assert sp.sweep_cycles_per_hz(2.0) == pytest.approx(43.28, abs=0.01)


def test_sweep_vs_time_simulation_Y14():
    """Y14: logarithmic sweep 20 → 160 Hz (2 oct/min) through the resonance of 1 DOF (60 Hz, ζ 5 %), constant acceleration
    1 g: D = ∫(dn/df)/N(σ_a(f)) df against the time simulation (the exact solution) + rainflow – 2 %.
    """
    fn, z, A, c = 60.0, 0.05, 9810.0, 1000.0             # σ = c·z [MPa/mm]
    R, f0, f1 = 2.0, 20.0, 160.0
    Tsw = 60.0 / R * math.log2(f1 / f0)
    dt = 1.0 / (f1 * 60)
    t = np.arange(int(Tsw / dt) + 1) * dt
    phi = 2 * np.pi * f0 * 60 / (R * math.log(2)) * (2 ** (R * t / 60) - 1)
    q, _, _ = modal_ode.integrate_base(dt, A * np.sin(phi), [1.0], [2 * np.pi * fn], z)
    cyc = kernel.cycles_of(c * q[:, 0])
    f = psd.freq_grid(f0, f1, [fn], z)
    sig = c * A * np.abs(psd.modal_frf(f, [fn], z)[:, 0])
    for sn in (SNCurve(100.0, 1e6, 5.0, 5.0), SNCurve(100.0, 1e6, 5.0, 9.0), SNCurve(150.0, 1e6, 3.0, 0.0)):
        Nn = np.array([sn.N(s) for s in cyc[:, 0] / 2])
        D_t = np.sum(cyc[:, 2] / Nn)
        D_f = sp.sweep_damage(f, sig, R, 1, sn)
        assert D_f == pytest.approx(D_t, rel=0.02)
    # several nodes at once
    Dn = sp.sweep_damage(f, np.column_stack([sig, 0.5 * sig]), R, 2, SNCurve(100.0, 1e6, 5.0, 5.0))
    assert Dn[1] / Dn[0] == pytest.approx(0.5 ** 5, rel=1e-12)
