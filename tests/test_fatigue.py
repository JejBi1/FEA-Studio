"""Fatigue tests: rainflow, S-N, Miner, mean-stress correction, superposition."""
import math

import numpy as np
import pytest
import rainflow

from fea_core.fatigue import kernel
from fea_core.fatigue.history import Channel, LoadHistory, read_csv_points
from fea_core.fatigue.settings import FatigueSettings
from fea_core.fatigue.sn import SNCurve, estimate_sn, marin, mean_corrected


def _sorted_cycles(c):
    return sorted((round(r, 9), round(m, 9), n) for r, m, n in c)


def test_rainflow_matches_package():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(3, 500))
        x = np.round(rng.normal(size=n).cumsum() * 10, int(rng.integers(0, 3)))
        ref = _sorted_cycles((r, m, c) for r, m, c, *_ in rainflow.extract_cycles(x))
        assert _sorted_cycles(kernel.cycles_of(x)) == ref


def test_closed_block():
    # block 0 → 100 → −100 → 0 repeated: one full cycle of the range 200
    c = kernel.cycles_of(np.array([0.0, 100.0, -100.0, 0.0]), closed=True)
    assert np.allclose(c[:, 0], 200.0)                      # only the range 200 …
    assert c[:, 2].sum() == pytest.approx(1.0)               # … 1 cycle in total (2 half cycles)
    # without closing, the half cycles of the range 100 and 200 remain
    assert kernel.cycles_of(np.array([0.0, 100.0, -100.0, 0.0]))[:, 2].sum() == pytest.approx(1.5)


def _single_node(F, sn: SNCurve, repeats, mean=kernel.MEAN_NONE, Rm=0.0, closed=True, S=None):
    S = np.array([[[1.0, 0, 0, 0, 0, 0]]]) if S is None else S
    D, sf, a = np.zeros(1), np.zeros(1), np.zeros(1)
    kernel.fatigue_nodes(np.ascontiguousarray(F), S, kernel.EQ_MAX_PRINCIPAL, mean,
                         np.array([sn.SD]), np.array([sn.ND]), np.array([sn.k]), np.array([sn.k2]),
                         np.array([Rm]), float(repeats), closed, D, sf, a)
    return D[0], sf[0], a[0]


def test_miner_constant_amplitude_and_safety():
    hist = LoadHistory([Channel("LC", "sine", amplitude=30.0, cycles=1000)], duration=1.0)
    t, F = hist.factors(["LC"])
    sn = SNCurve(20.0, 1e6, 5.0, 0.0)
    D, sf, a = _single_node(F, sn, repeats=100)
    D_ref = 1000 / (1e6 * (20 / 30) ** 5)
    assert a == pytest.approx(30.0, rel=1e-9)
    assert D == pytest.approx(D_ref, rel=1e-9)
    assert sf == pytest.approx((1 / (100 * D_ref)) ** (1 / 5), rel=1e-5)


def test_endurance_limit_and_haibach():
    hist = LoadHistory([Channel("LC", "sine", amplitude=15.0, cycles=10)])
    t, F = hist.factors(["LC"])
    D, sf, _ = _single_node(F, SNCurve(20.0, 1e6, 5.0, 0.0), repeats=1e6)
    assert D == 0.0                          # below the fatigue limit
    assert sf > 1.33                         # s · 15 must exceed 20 MPa
    D2, _, _ = _single_node(F, SNCurve(20.0, 1e6, 5.0, 9.0), repeats=1)
    assert D2 == pytest.approx(10 / (1e6 * (20 / 15) ** 9), rel=1e-9)


def test_goodman_gerber():
    assert mean_corrected(30, 100, 400, "goodman") == pytest.approx(40.0)
    assert mean_corrected(30, 100, 400, "gerber") == pytest.approx(30 / (1 - 1 / 16))
    assert mean_corrected(30, -100, 400, "goodman") == 30          # pressure without benefit
    hist = LoadHistory([Channel("LC", "sine", amplitude=30.0, mean=100.0, cycles=1000)])
    t, F = hist.factors(["LC"])
    sn = SNCurve(20.0, 1e6, 5.0, 0.0)
    D, _, _ = _single_node(F, sn, 1, kernel.MEAN_GOODMAN, Rm=400.0)
    assert D == pytest.approx(1000 / (1e6 * (20 / 40) ** 5), rel=1e-9)


def test_signed_equivalent():
    assert kernel.equivalent(-100.0, 0, 0, 0, 0, 0, kernel.EQ_SIGNED_MISES) == pytest.approx(-100.0)
    assert kernel.equivalent(100.0, 0, 0, 0, 0, 0, kernel.EQ_SIGNED_MISES) == pytest.approx(100.0)
    assert kernel.equivalent(0, 0, 0, 50.0, 0, 0, kernel.EQ_MAX_PRINCIPAL) == pytest.approx(50.0)


def test_superposition_two_cases():
    # two states: σx from A (sine), σy from B (sine in antiphase, smaller)
    hist = LoadHistory([Channel("A", "sine", amplitude=1.0, cycles=4),
                        Channel("B", "sine", amplitude=0.5, cycles=4, phase_deg=180)])
    t, F = hist.factors(["A", "B"])
    np.testing.assert_allclose(F[:, 0], -2 * F[:, 1], atol=1e-12)
    S = np.array([[[100.0, 0, 0, 0, 0, 0]], [[0, 100.0, 0, 0, 0, 0]]])
    D, sf, a = _single_node(F, SNCurve(50.0, 1e6, 5.0, 0.0), 1, S=S)
    # σx = 100 sin, σy = −50 sin → the dominant principal stress oscillates ±100 MPa
    assert a == pytest.approx(100.0, rel=1e-9)
    # note: for pure shear (σx = −σy) the sign of the equivalent stress is ambiguous –
    # a known weakness of the sign methods, solved only by the critical plane method


def test_proportional_fast_path_equals_general():
    from fea_core.fatigue.analysis import proportional
    rng = np.random.default_rng(5)
    f = np.round(rng.normal(size=300).cumsum(), 2)
    F = np.outer(f, [1.0, -0.4, 0.25])
    pr = proportional(F)
    assert pr is not None
    assert proportional(np.column_stack([f, rng.normal(size=300)])) is None
    S = rng.normal(size=(3, 200, 6)) * 40
    n = S.shape[1]
    par = (np.full(n, 30.0), np.full(n, 1e6), np.full(n, 6.0), np.full(n, 11.0), np.full(n, 400.0))
    for eq in (kernel.EQ_SIGNED_MISES, kernel.EQ_MAX_PRINCIPAL):
        a = [np.zeros(n) for _ in range(3)]
        b = [np.zeros(n) for _ in range(3)]
        kernel.fatigue_nodes(F, S, eq, kernel.MEAN_GOODMAN, *par, 1e3, True, *a)
        Seff = np.ascontiguousarray(np.tensordot(pr[1], S, axes=(0, 0)))
        kernel.fatigue_nodes_proportional(np.ascontiguousarray(pr[0]), Seff, eq, kernel.MEAN_GOODMAN, *par,
                                          1e3, True, *b)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=1e-9)


def test_marin_and_estimate():
    m = marin(360, "obrobeny", None, 50.0)
    assert m["ka"] == pytest.approx(4.51 * 360 ** -0.265)
    assert m["kb"] == 1.0 and m["ke"] == 1.0
    assert marin(360, "lesteny", 30, 90.0)["kb"] == pytest.approx(1.24 * 30 ** -0.107)
    sn = estimate_sn(210000, 360)
    assert sn.SD == pytest.approx(180.0) and sn.N(0.9 * 360) == pytest.approx(1e3, rel=1e-9)


def test_csv_and_json(tmp_path):
    p = tmp_path / "k.csv"
    p.write_text("čas;síla\n0;0\n0,5;1,5\n1;-1\n", encoding="utf-8")
    assert read_csv_points(p) == [[0.0, 0.0], [0.5, 1.5], [1.0, -1.0]]
    fs = FatigueSettings(LoadHistory([Channel("LC", points=[[0, 0], [1, 2]])]), repeats=1e5)
    assert FatigueSettings.from_dict(fs.to_dict()) == fs


# ====================================================================== critical plane
from fea_core.fatigue import critical_plane as cp  # noqa: E402
from fea_core.fatigue.settings import findley_k, findley_lambda  # noqa: E402

SQ3 = math.sqrt(3.0)


def _cp(F, S, crit, ratio=SQ3, prop=False, mean=kernel.MEAN_NONE, sn=SNCurve(100.0, 1e6, 5.0, 0.0), Rm=600.0,
        repeats=1.0, planes=100, levels=4, search=(3, 4, 1)):
    """Critical plane for the nodes S (K, n, 6) and the multiples F (T, K); prop → F[:, 0] and S[0]."""
    n = S.shape[1]
    kF = findley_k(ratio) if crit == cp.CP_FINDLEY else 0.0
    lam = findley_lambda(kF) if crit == cp.CP_FINDLEY else 1.0
    out = [np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3))]
    common = (np.full(n, crit, np.int64), np.full(n, kF), np.full(n, lam), mean, np.full(n, sn.SD),
              np.full(n, sn.ND), np.full(n, sn.k), np.full(n, sn.k2), np.full(n, Rm), float(repeats), True, *out)
    P = cp.hemisphere(planes)
    if prop:
        cp.cp_nodes_proportional(np.ascontiguousarray(F[:, 0]), np.ascontiguousarray(S[0]), P,
                                 cp.plane_spacing(planes), levels, *common)
    else:
        cp.cp_nodes(np.ascontiguousarray(F), np.ascontiguousarray(S), P, cp.plane_spacing(planes), levels, 12,
                    np.array(search, np.int64), *common)
    return out


def _sine(cycles=5, n=24, phase=0.0):
    t = np.linspace(0.0, 1.0, cycles * n + 1)
    return np.sin(2 * np.pi * cycles * t + phase)


def test_findley_k_from_ratio():
    for s in (1.2, 1.5, SQ3, 1.9, 2.0):
        k = findley_k(s)
        q = math.sqrt(1 + k * k)
        assert 2 * q / (k + q) == pytest.approx(s, rel=1e-12)     # f₋₁/t₋₁ back
    assert findley_k(2.0) == 0.0 and findley_lambda(0.0) == 0.5
    with pytest.raises(ValueError):
        findley_k(1.0)


def test_rainflow_window_matches_kernel():
    rng = np.random.default_rng(3)
    for _ in range(100):
        x = np.round(rng.normal(size=int(rng.integers(3, 300))).cumsum(), 1)
        n = len(x)
        rev, rr, mm, cc, yw = (np.empty(n + 1) for _ in range(5))
        ridx, stk = np.empty(n + 1, np.int64), np.empty(n + 1, np.int64)
        nc = cp.rainflow_win(x, x, n, rev, ridx, stk, rr, mm, cc, yw, False)
        ref = kernel.cycles_of(x)
        np.testing.assert_allclose(np.column_stack([rr[:nc], mm[:nc], cc[:nc]]), ref)
        # window of the loop at u y = x: maximum = the upper vertex of the cycle
        np.testing.assert_allclose(yw[:nc], mm[:nc] + rr[:nc] / 2, atol=1e-9)


@pytest.mark.parametrize("prop", [False, True])
def test_cp_uniaxial_equals_principal(prop):
    F = _sine()[:, None]
    S = np.array([[[100.0, 0, 0, 0, 0, 0]]])
    sn = SNCurve(80.0, 1e6, 5.0, 0.0)
    D0, _, a0 = _single_node(F, sn, 1.0, S=S)                   # max. principal stress
    for crit in (cp.CP_NORMAL, cp.CP_FINDLEY):                   # Findley is calibrated on alternating tension
        D, sf, a, nrm, _ = _cp(F, S, crit, prop=prop, sn=sn)
        assert a[0] == pytest.approx(100.0, rel=1e-5)
        assert D[0] == pytest.approx(D0, rel=5e-5)
    assert abs(nrm[0, 0]) < 0.9                                  # Findley: the plane out of the principal direction


@pytest.mark.parametrize("prop", [False, True])
def test_cp_pure_torsion(prop):
    # pure shear σ1 = −σ3: no ambiguity of the sign; normal criterion σa = τa, Findley σeq = s·τa
    F = _sine()[:, None]
    S = np.array([[[0, 0, 0, 60.0, 0, 0]]])
    a_n = _cp(F, S, cp.CP_NORMAL, prop=prop)
    assert a_n[2][0] == pytest.approx(60.0, rel=1e-6)
    nx, ny, nz = np.abs(a_n[3][0])
    assert nz < 1e-6 and abs(nx - ny) < 1e-6                     # the plane at 45° (σ1 and σ3 equivalent)
    for s in (SQ3, 1.6):
        a_f = _cp(F, S, cp.CP_FINDLEY, ratio=s, prop=prop)
        assert a_f[2][0] == pytest.approx(s * 60.0, rel=5e-5)     # the plane refined to ~1°


def _op_reference(sa, ta, s):
    """Hand calculation of the 90° out of phase tension σa·sin + torsion τa·cos: a plane with the normal in the plane xy at the angle θ,
    τ(θ) = −σx·sin2θ/2 + τ·cos2θ (a line), σn(θ) = σx(1+cos2θ)/2 + τ·sin2θ; max over θ (finely).
    """
    th = np.linspace(0, np.pi, 400001)
    c, si = np.cos(2 * th), np.sin(2 * th)
    tau_a = np.hypot(sa * si / 2, ta * c)
    sn_max = np.hypot(sa * (1 + c) / 2, ta * si)
    k = findley_k(s)
    return float(((tau_a + k * sn_max) / findley_lambda(k)).max()), float(sn_max.max())


def test_cp_out_of_phase_tension_torsion():
    sa, ta = 100.0, 80.0
    # fine sampling: the peak σn in a general plane lies between the samples (24 points per cycle → up to −0.9 %)
    F = np.column_stack([_sine(5, 360), _sine(5, 360, phase=np.pi / 2)])
    S = np.array([[[sa, 0, 0, 0, 0, 0]], [[0, 0, 0, ta, 0, 0]]])
    ref_f, ref_n = _op_reference(sa, ta, SQ3)
    a_f = _cp(F, S, cp.CP_FINDLEY)[2][0]
    a_n = _cp(F, S, cp.CP_NORMAL)[2][0]
    assert a_f == pytest.approx(ref_f, rel=2e-4)
    assert a_n == pytest.approx(ref_n, rel=1e-4)
    # out of phase Findley is lower than in phase (the shear and σn,max amplitudes do not occur together)
    a_ip = _cp(F[:, :1].repeat(2, axis=1), S, cp.CP_FINDLEY)[2][0]
    assert a_f < a_ip


def test_cp_general_equals_proportional_and_principal():
    rng = np.random.default_rng(11)
    f = np.round(rng.normal(size=200).cumsum(), 2)
    S = rng.normal(size=(1, 60, 6)) * 40
    F = f[:, None]
    sn = SNCurve(30.0, 1e6, 6.0, 11.0)
    g = _cp(F, S, cp.CP_NORMAL, sn=sn)
    p = _cp(F, S, cp.CP_NORMAL, sn=sn, prop=True)
    ref = [np.zeros(60) for _ in range(3)]
    kernel.fatigue_nodes(np.ascontiguousarray(F), S, kernel.EQ_MAX_PRINCIPAL, kernel.MEAN_NONE, np.full(60, sn.SD),
                         np.full(60, sn.ND), np.full(60, sn.k), np.full(60, sn.k2), np.full(60, 600.0), 1.0, True,
                         *ref)
    # σn criterion under proportional loading = the principal stress with the largest damage ≥ max. |principal|
    np.testing.assert_allclose(g[0], p[0], rtol=1e-9)
    assert np.all(g[0] >= ref[0] * (1 - 1e-9))
    gf = _cp(F, S, cp.CP_FINDLEY, sn=sn)
    pf = _cp(F, S, cp.CP_FINDLEY, sn=sn, prop=True)
    np.testing.assert_allclose((gf[0] / pf[0]) ** (1 / sn.k), 1.0, atol=2e-3)


def test_cp_search_close_to_exhaustive():
    # strongly non-proportional variable loading: fast search against brute force (2000 planes × 24 directions)
    rng = np.random.default_rng(2)
    F = np.ascontiguousarray(np.round(rng.normal(size=(120, 3)).cumsum(axis=0) * 0.3, 3))
    n = 40
    S = np.ascontiguousarray(rng.normal(size=(3, n, 6)) * 30)
    sn = SNCurve(30.0, 1e6, 6.0, 11.0)
    for crit in (cp.CP_NORMAL, cp.CP_FINDLEY):
        kF = findley_k(SQ3) if crit else 0.0
        lam = findley_lambda(kF) if crit else 1.0
        Dref, nref = np.zeros(n), np.zeros((n, 3))
        cp.cp_nodes_exhaustive(F, S, cp.hemisphere(2000), 24, np.full(n, crit, np.int64), np.full(n, kF),
                               np.full(n, lam), kernel.MEAN_NONE, np.full(n, sn.SD), np.full(n, sn.ND),
                               np.full(n, sn.k), np.full(n, sn.k2), np.full(n, 600.0), True, Dref, nref)
        D = _cp(F, S, crit, sn=sn)[0]
        r = (D / Dref) ** (1 / sn.k)                              # ratio of the equivalent stresses
        assert r.min() > 0.97 and np.median(r) > 0.995


def test_cp_settings_json_and_compat():
    fs = FatigueSettings(LoadHistory([Channel("LC", points=[[0, 0], [1, 2]])]), equivalent="critical_plane",
                         cp_criterion="findley", cp_ratio={"Ocel": 1.6}, cp_planes=150, cp_thorough=True)
    assert FatigueSettings.from_dict(fs.to_dict()) == fs
    old = fs.to_dict()
    for k in [k for k in old if k.startswith("cp_")]:
        del old[k]
    old["equivalent"] = "signed_mises"
    st = FatigueSettings.from_dict(old)                          # an old file of the study
    assert st.cp_criterion == "auto" and st.cp_planes == 100 and not st.critical_plane
    text = " ".join(fs.describe())
    assert "critical plane" in text and "Findley" in text and "600 planes" in text
