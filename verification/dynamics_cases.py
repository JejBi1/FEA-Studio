"""Verification of structural dynamics (stage F8.1, fea_core.dynamics) – cases Y of the F8 spec (chapter 7).

Y1  1-DOF oscillator (rigid block on an elastic support + point mass 1 kg): |H| and the phase of the harmonic force at resonance
    and at the half-power points against 1/√((1 − r²)² + (2ζr)²); Y1s structural damping – the peak 1/η.
Y2  cantilever 10×10×100 with a harmonic force: f₁ (E–B), peak 0.9707/(2ζ) (E–B, shape of the 1st mode), Python × ccx SSD at the probes.
Y3  step force: 1-DOF DLF 2 (ζ = 0) and 1 + e^(−πζ/√(1−ζ²)) (ζ 2 %); cantilever Python × ccx *MODAL DYNAMIC.
Y6  free decay of 1 DOF after a short impulse: logarithmic decrement → ζ.
Y10 run-up of an axis: rectangular pulse of the base acceleration a, τ → residual amplitude X = (2a/ω²)|sin(ωτ/2)| (ζ = 0).
Y13 base displacement (half-sine) of the cantilever: Python × ccx *MODAL DYNAMIC + *BASE MOTION (displacement).
Y15 cantilever with a point mass M at the end: Rayleigh f = √(3EI/((M + 33/140·m)L³))/(2π), Σ masses = m + M.
Y16 base excitation: a rigid remote constraint as the base = clamping (the same f and response); half of the model with a plane of
    symmetry in the excited direction → warning.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fea_core.study import Fixed, Force, LoadCase, MeshSettings, Material, ElasticSupport, RemoteSupport, Displacement
from fea_core.geometry import Part
from fea_core.selectors import find_faces
from fea_core.dynamics.model import Excitation, Probe, PointMass
from verification.cases import Case, Check, E, RHO, _study, box

# ------------------------------------------------------------------ 1-DOF oscillator
K_SPR = 1000.0          # N/mm (total stiffness of the support in the vertical direction)
M_OSC = 1e-3            # t (1 kg)
W_OSC = math.sqrt(K_SPR / M_OSC)      # 1000 rad/s
F_OSC = W_OSC / (2 * math.pi)         # 159,155 Hz
RIGID = Material("Tuhý blok", 2.1e8, 0.3, 1e-15, source="ověřovací úloha")


def _osc(wd: Path, name: str, zeta: float = 0.02):
    st, parts = _study(name, wd / "model.step", [Part("Blok", box(0, 0, 0, 10, 10, 10))], MeshSettings(size=5.0))
    st.materials.clear()
    st.default_material = None
    st.add_material(RIGID, default=True)
    st.add_selection("dno", find_faces(parts, z=0))
    st.add_selection("vrch", find_faces(parts, z=10))
    st.constraints.append(ElasticSupport("dno", K_SPR, direction="all", per_area=False))
    st.masses.append(PointMass("vrch", M_OSC, point=[5.0, 5.0, 5.0], name="Hmota 1 kg"))
    st.load_cases.append(LoadCase("F", [Force("vrch", [0, 0, 100.0])]))
    ds = st.dynamics
    ds.enabled = True
    ds.modes = 6
    ds.damping.zeta = zeta
    ds.probes = [Probe("hmota", point=[5.0, 5.0, 10.0])]
    return st, parts


def _zmode(d) -> int:
    """Index of the vertical mode (largest Γ_z)."""
    return int(np.argmax(np.abs(d.gamma[:, 2])))


def y1_build(wd: Path):
    st, parts = _osc(wd, "Y1 oscilator harmonicky")
    h = st.dynamics.harmonic
    h.enabled = True
    h.f_min, h.f_max, h.points = 20.0, 400.0, 10
    h.excitation = [Excitation(kind="force", case="F")]
    return st, parts


def y1_eval(res, st):
    d = res.dynamics
    h = d.harmonic
    u_st = 100.0 / K_SPR
    j = _zmode(d)
    fn = float(d.freq[j])
    z = float(d.zeta[j])
    U = h["probes"]["hmota"]["U"][:, 2]
    f = h["f"]
    out = [Check("f svislého módu √(k/m)/2π", fn, F_OSC, 1e-3, "Hz")]
    for lab, r in (("rezonance r = 1", 1.0), ("r = 1 − ζ", 1 - z), ("r = 1 + ζ", 1 + z)):
        i = int(np.argmin(np.abs(f - r * fn)))
        H = 1 / math.sqrt((1 - r * r) ** 2 + (2 * z * r) ** 2)
        out.append(Check(f"|u|/u_st, {lab}", abs(U[i]) / u_st, H, 1e-3, ""))
    i = int(np.argmin(np.abs(f - fn)))
    out.append(Check("fáze v rezonanci (90°)", -math.degrees(np.angle(U[i])), 90.0, 2e-3, "°"))
    i = 0
    r = f[0] / fn
    out.append(Check(f"|u|/u_st při {f[0]:.0f} Hz (kvazistatika)", abs(U[i]) / u_st,
                     1 / math.sqrt((1 - r * r) ** 2 + (2 * z * r) ** 2), 1e-3, ""))
    return out


def y1s_build(wd: Path):
    st, parts = y1_build(wd)
    st.name = "Y1s strukturalni tlumeni"
    st.dynamics.damping.kind = "structural"
    st.dynamics.damping.eta = 0.04
    return st, parts


def y1s_eval(res, st):
    d = res.dynamics
    h = d.harmonic
    j = _zmode(d)
    U = h["probes"]["hmota"]["U"][:, 2]
    i = int(np.argmin(np.abs(h["f"] - d.freq[j])))
    return [Check("špička |u|/u_st = 1/η (η = 0,04)", abs(U[i]) / (100.0 / K_SPR), 1 / 0.04, 1e-3, "")]


# ------------------------------------------------------------------ konzola 10×10×100
L, B = 100.0, 10.0
I_B = B ** 4 / 12


def _beam(wd: Path, name: str, size: float = 2.5, mass: float | None = None):
    st, parts = _study(name, wd / "model.step", [Part("Konzola", box(0, 0, 0, L, B, B))], MeshSettings(size=size))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("cel", find_faces(parts, x=L))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("F", [Force("cel", [0, 100.0, 0])]))
    if mass:
        st.masses.append(PointMass("cel", mass, point=[L, 5.0, 5.0], name="Hmota na konci"))
    ds = st.dynamics
    ds.enabled = True
    ds.modes = 10
    ds.damping.zeta = 0.02
    ds.probes = [Probe("hrot", point=[L, 5.0, 5.0])]
    return st, parts


def _f_eb(beta_l: float) -> float:
    return beta_l ** 2 / (2 * math.pi * L ** 2) * math.sqrt(E * I_B / (RHO * B * B))


def y2_build(wd: Path):
    st, parts = _beam(wd, "Y2 konzola harmonicka sila")
    h = st.dynamics.harmonic
    h.enabled = True
    h.f_min, h.f_max, h.points = 500.0, 1200.0, 10
    h.excitation = [Excitation(kind="force", case="F")]
    st.dynamics.verify_ccx = True
    return st, parts


def y2_eval(res, st):
    d = res.dynamics
    h = d.harmonic
    u_st = res.probe([L, 5.0, 5.0], "U", 0)[1]
    U = np.abs(h["probes"]["hrot"]["U"][:, 1])
    et = d.info.get("etalon", {}).get("harmonic", {})
    return [Check("f₁ (Euler–Bernoulli, Timoshenko −0,3 %)", float(d.freq[0]), _f_eb(1.875104), 5e-3, "Hz"),
            Check("špička |u|/u_st (E–B 0,9707/(2ζ))", float(U.max()) / u_st, 0.9707 / 0.04, 1e-2, ""),
            Check("Python × ccx SSD v sondách (1 + chyba)", 1 + et.get("max_rel_error", 1.0), 1.0, 1e-4, ""),
            Check("normování φᵀMφ = 1 (1 + chyba)", 1 + d.checks["mass_norm_error"], 1.0, 1e-3, ""),
            Check("rovnováha modálních reakcí (1 + chyba)", 1 + d.checks["mode_equilibrium"], 1.0, 1e-3, "")]


def y3_build(wd: Path):
    st, parts = _osc(wd, "Y3 skok sily 1 DOF", zeta=0.0)
    tr = st.dynamics.transient
    tr.enabled = True
    tr.t_end = 5 / F_OSC
    tr.dt = 1 / F_OSC / 200
    tr.excitation = [Excitation(kind="force", case="F", table=[[0.0, 1.0], [1.0, 1.0]])]
    return st, parts


def y3_eval(res, st):
    u = res.dynamics.transient["probes"]["hmota"]["U"][:, 2]
    return [Check("DLF skoku 1 DOF, ζ = 0", float(u.max()) / (100.0 / K_SPR), 2.0, 2e-3, "")]


def y3d_build(wd: Path):
    st, parts = y3_build(wd)
    st.name = "Y3d skok sily 1 DOF tlumeny"
    st.dynamics.damping.zeta = 0.02
    return st, parts


def y3d_eval(res, st):
    u = res.dynamics.transient["probes"]["hmota"]["U"][:, 2]
    z = 0.02
    return [Check("DLF skoku 1 DOF, ζ = 2 % (1 + e^(−πζ/√(1−ζ²)))", float(u.max()) / (100.0 / K_SPR),
                  1 + math.exp(-math.pi * z / math.sqrt(1 - z * z)), 5e-3, "")]


def y3b_build(wd: Path):
    st, parts = _beam(wd, "Y3b konzola skok sily")
    T1 = 1 / 833.3
    tr = st.dynamics.transient
    tr.enabled = True
    tr.t_end, tr.dt = 5 * T1, T1 / 125
    tr.excitation = [Excitation(kind="force", case="F", table=[[0.0, 1.0], [1.0, 1.0]])]
    st.dynamics.damping.zeta = 0.0
    st.dynamics.verify_ccx = True
    return st, parts


def y3b_eval(res, st):
    d = res.dynamics
    u = d.transient["probes"]["hrot"]["U"][:, 1]
    u_st = res.probe([L, 5.0, 5.0], "U", 0)[1]
    et = d.info.get("etalon", {}).get("transient", {})
    return [Check("konzola max u/u_st (ccx pokus 3a 1,9456)", float(u.max()) / u_st, 1.9456, 3e-3, ""),
            Check("Python × ccx *MODAL DYNAMIC v sondách (1 + chyba)", 1 + et.get("max_rel_error", 1.0), 1.0, 1e-4, "")]


def y6_build(wd: Path):
    st, parts = _osc(wd, "Y6 volne doznivani", zeta=0.02)
    T = 1 / F_OSC
    tr = st.dynamics.transient
    tr.enabled = True
    tr.t_end, tr.dt = 12 * T, T / 400
    tr.excitation = [Excitation(kind="force", case="F", table=[[0.0, 0.0], [T / 20, 1.0], [T / 10, 0.0]])]
    return st, parts


def y6_eval(res, st):
    tr = res.dynamics.transient
    t = tr["t"]
    u = tr["probes"]["hmota"]["U"][:, 2]
    T = 1 / F_OSC
    pk = [i for i in range(1, len(u) - 1) if u[i] >= u[i - 1] and u[i] > u[i + 1] and t[i] > T / 5]
    n = len(pk) - 1
    delta = math.log(u[pk[0]] / u[pk[-1]]) / n
    zeta = delta / math.sqrt(4 * math.pi ** 2 + delta ** 2)
    return [Check(f"ζ z log. dekrementu ({n} period)", zeta, 0.02, 1e-2, "")]


def y10_build(wd: Path):
    st, parts = _osc(wd, "Y10 rozjezd osy", zeta=0.0)
    T = 1 / F_OSC
    tau = 0.3 * T
    tr = st.dynamics.transient
    tr.enabled = True
    tr.t_end, tr.dt = 4 * T, T / 2000
    tr.excitation = [Excitation(kind="base_acc", dof=3, unit="g", table=[[0.0, 1.0], [tau, 1.0], [tau + tr.dt, 0.0]])]
    return st, parts


def y10_eval(res, st):
    tr = res.dynamics.transient
    T = 1 / F_OSC
    tau = 0.3 * T + 0.5 * tr["t"][1]               # rectangle with a leading edge of 1 step ≈ centre of the edge
    t = tr["t"]
    u = tr["probes"]["hmota"]["U"][:, 2]            # relative displacement of the mass
    after = t > tau + T / 10
    amp = 0.5 * (u[after].max() - u[after].min())
    w = 2 * math.pi * F_OSC
    return [Check("zbytková amplituda (2a/ω²)|sin(ωτ/2)|", amp, 2 * 9810.0 / w ** 2 * abs(math.sin(w * tau / 2)),
                  5e-3, "mm")]


def y13_build(wd: Path):
    st, parts = _beam(wd, "Y13 pulz posuvu zakladny")
    T1 = 1 / 833.3
    tr = st.dynamics.transient
    tr.enabled = True
    tr.t_end, tr.dt = 6 * T1, T1 / 100
    tp = 2 * T1                                     # half-sine of the base displacement 0.01 mm, duration 2·T₁
    ts = np.arange(0.0, tp + 1e-12, tr.dt)
    tr.excitation = [Excitation(kind="base_disp", dof=2, unit="mm",
                                table=[[float(x), float(0.01 * math.sin(math.pi * x / tp))] for x in ts])]
    st.dynamics.verify_ccx = True
    return st, parts


def y13_eval(res, st):
    et = res.dynamics.info.get("etalon", {}).get("transient", {})
    return [Check("Python × ccx *MODAL DYNAMIC + *BASE MOTION (1 + chyba)", 1 + et.get("max_rel_error", 1.0), 1.0,
                  1e-3, "")]


M_TIP = 10 * RHO * L * B * B


def y15_build(wd: Path):
    st, parts = _beam(wd, "Y15 konzola s bodovou hmotou", mass=M_TIP)
    st.dynamics.modes = 4
    h = st.dynamics.harmonic
    h.enabled = True
    h.f_min, h.f_max, h.points = 50.0, 400.0, 6
    h.excitation = [Excitation(kind="force", case="F")]
    return st, parts


def y15_eval(res, st):
    d = res.dynamics
    m = RHO * L * B * B
    f_r = math.sqrt(3 * E * I_B / ((M_TIP + 33 / 140 * m) * L ** 3)) / (2 * math.pi)
    return [Check("f₁ s hmotou na konci (Rayleigh)", float(d.freq[0]), f_r, 1e-2, "Hz"),
            Check("hmotnost modelu m + M", d.model_mass, m + M_TIP, 1e-6, "t"),
            Check("normování s bodovou hmotou (1 + chyba)", 1 + d.checks["mass_norm_error"], 1.0, 1e-3, ""),
            Check("Γ vlastní integrací × ccx (1 + chyba)", 1 + d.checks["gamma_error"], 1.0, 1e-3, "")]


def y16_build(wd: Path):
    st, parts = _beam(wd, "Y16 tuha zakladna")
    st.constraints = [RemoteSupport("vetknuti", [0.0, 5.0, 5.0], coupling="rigid")]
    h = st.dynamics.harmonic
    h.enabled = True
    h.f_min, h.f_max, h.points = 500.0, 1200.0, 6
    h.excitation = [Excitation(kind="base_acc", dof=2, unit="g")]
    return st, parts


def y16_eval(res, st):
    d = res.dynamics
    U = np.abs(d.harmonic["probes"]["hrot"]["U"][:, 1])
    # reference clamping on the same mesh (Y2 has the same mesh and model)
    return [Check("f₁ s tuhou základnou = vetknutí (E–B)", float(d.freq[0]), _f_eb(1.875104), 5e-3, "Hz"),
            Check("špička relativního posuvu hrotu > 0 (buzení přes tuhou vazbu)", float(U.max() > 0), 1.0, 0.0, "")]


def y16s_build(wd: Path):
    st, parts = _beam(wd, "Y16s rovina symetrie v buzenem smeru")
    st.add_selection("sym", find_faces(parts, y=0))
    st.constraints.append(Displacement("sym", uy=0.0))
    h = st.dynamics.harmonic
    h.enabled = True
    h.f_min, h.f_max, h.points = 500.0, 1200.0, 4
    h.excitation = [Excitation(kind="base_acc", dof=2, unit="g")]
    return st, parts


def y16s_eval(res, st):
    hit = any("symmetry plane" in w and "excit" in w.lower() for w in res.warnings)
    return [Check("varování: rovina symetrie v buzeném směru", float(hit), 1.0, 0.0, "")]


DYN_CASES = {
    "Y1": Case("Y1", "Oscilátor 1 DOF – harmonická síla (|H|, fáze)", y1_build, y1_eval),
    "Y1s": Case("Y1s", "Oscilátor 1 DOF – strukturální tlumení η", y1s_build, y1s_eval),
    "Y2": Case("Y2", "Konzola – harmonická síla, Python × ccx SSD", y2_build, y2_eval),
    "Y3": Case("Y3", "Skok síly 1 DOF (DLF 2)", y3_build, y3_eval),
    "Y3d": Case("Y3d", "Skok síly 1 DOF tlumený", y3d_build, y3d_eval),
    "Y3b": Case("Y3b", "Konzola – skok síly, Python × ccx *MODAL DYNAMIC", y3b_build, y3b_eval),
    "Y6": Case("Y6", "Volné doznívání – logaritmický dekrement", y6_build, y6_eval),
    "Y10": Case("Y10", "Rozjezd osy – zbytková amplituda", y10_build, y10_eval),
    "Y13": Case("Y13", "Posuv základny – Python × ccx *BASE MOTION", y13_build, y13_eval),
    "Y15": Case("Y15", "Konzola s bodovou hmotou (Rayleigh)", y15_build, y15_eval),
    "Y16": Case("Y16", "Buzení základnou přes tuhou vzdálenou vazbu", y16_build, y16_eval),
    "Y16s": Case("Y16s", "Rovina symetrie v buzeném směru → varování", y16s_build, y16s_eval),
}


def y4_build(wd: Path):
    st, parts = _osc(wd, "Y4 raz pulsinus", zeta=0.05)
    s = st.dynamics.shock
    s.enabled = True
    s.pulse, s.A, s.tau, s.dof = "halfsine", 10.0, 0.82 / F_OSC, 3
    s.combine = "srss"
    s.missing_mass = True
    return st, parts


def y4_eval(res, st):
    sh = res.dynamics.shock
    d = res.dynamics
    j = _zmode(d)
    w = 2 * math.pi * float(d.freq[j])
    A = 10.0 * 9810.0
    # 1 DOF: absolute acceleration of the mass max = S_a = 1.652·A (ζ 5 %, f_nτ = 0.82); relative displacement S_a/ω²
    u = float(sh["U"][:, 2].max())
    tr = float(np.abs(sh["time"]["probes"]["hmota"]["U"][:, 2]).max())
    return [Check("S_a(f_n)/A půlsinus f_nτ = 0,82, ζ 5 % (1,652)", float(sh["Sa"][j]) / A, 1.652, 3e-3, ""),
            Check("posuv SRS = S_a/ω² (1 DOF, + chybějící hmota ≈ 0)", u, float(sh["Sa"][j]) / w ** 2, 2e-3, "mm"),
            Check("přechodová odezva na pulz × SRS (max posuv)", tr, float(sh["Sa"][j]) / w ** 2, 5e-3, "mm")]


def y5_build(wd: Path):
    st, parts = _osc(wd, "Y5 nahodne vibrace", zeta=0.05)
    r = st.dynamics.random
    r.enabled = True
    r.asd = [[20.0, 0.04], [2000.0, 0.04]]
    r.dof = 3
    r.duration_h = 1.0
    return st, parts


def y5_eval(res, st):
    rr = res.dynamics.random
    from fea_core.dynamics import psd
    f = rr["f"]
    A = rr["probes"]["hmota"]["A_psd_g2"][:, 2]
    grms = math.sqrt(float((psd.trapz_weights(f) * A).sum()))
    d = res.dynamics
    fn = float(d.freq[_zmode(d)])
    z = 0.05
    miles = psd.miles(fn, 1 / (2 * z), 0.04) * math.sqrt(1 + 4 * z * z)
    return [Check("g_rms absolutního zrychlení hmoty (Miles·√(1+4ζ²))", grms, miles, 5e-3, "g")]


def y14_build(wd: Path):
    st, parts = _osc(wd, "Y14 rozmitani sinusu", zeta=0.05)
    h = st.dynamics.harmonic
    h.enabled = True
    h.f_min, h.f_max, h.points = 20.0, 400.0, 30
    h.excitation = [Excitation(kind="base_acc", dof=3, unit="g")]
    h.sweep_rate, h.sweeps = 1.0, 2
    return st, parts


def y14_eval(res, st):
    sw = res.dynamics.harmonic["sweep"]["summary"]
    return [Check("poškození z rozmítání > 0 (dn/df = 60/(R·ln 2), ověřeno jádrem Y14)", float(sw["max_damage"] >= 0), 1.0,
                  0.0, "")]


DYN_CASES.update({
    "Y4": Case("Y4", "Ráz půlsinus – SRS, SRSS + chybějící hmota, přechodová odezva", y4_build, y4_eval),
    "Y5": Case("Y5", "Náhodné vibrace 1 DOF – Miles", y5_build, y5_eval),
    "Y14": Case("Y14", "Sinusové rozmítání – únava dn/df", y14_build, y14_eval),
})


# ------------------------------------------------------------------ Y9 prestressed modal analysis (D5)
L9 = 200.0
N_CR9 = math.pi ** 2 * E * I_B / L9 ** 2


def _y9(wd: Path, name: str, N: float):
    from fea_core.study import Pressure
    st, parts = _study(name, wd / "model.step", [Part("Nosnik", box(0, 0, 0, L9, B, B))], MeshSettings(size=2.5))
    st.add_selection("a", find_faces(parts, x=0))
    st.add_selection("b", find_faces(parts, x=L9))
    c = [0.0, 5.0, 5.0]
    st.constraints += [RemoteSupport("a", c, dofs=[True, True, True, True, False, False], coupling="distributing"),
                       RemoteSupport("b", [L9, 5.0, 5.0], dofs=[False, True, True, False, False, False],
                                     coupling="distributing")]
    st.load_cases.append(LoadCase("Osová síla", [Pressure("b", -N / (B * B))]))   # tension = negative pressure
    st.modal.enabled, st.modal.modes = True, 4
    ds = st.dynamics
    ds.enabled, ds.modes = True, 4
    ds.prestress.enabled, ds.prestress.case = True, "Osová síla"
    return st, parts


def y9_build(wd: Path):
    return _y9(wd, "Y9 predpjaty nosnik tah", 0.5 * N_CR9)


def y9c_build(wd: Path):
    return _y9(wd, "Y9c predpjaty nosnik tlak", -0.5 * N_CR9)


def _y9_eval(N):
    def ev(res, st):
        f0 = float(res.modes[0].values[0])
        f = float(res.dynamics.freq[0])
        return [Check(f"f₁(N)/f₁(0) = √(1 + N/N_kr), N = {N / N_CR9:+.1f}·N_kr", f / f0, math.sqrt(1 + N / N_CR9), 5e-3, ""),
                # shapes after the prestress (numbers of the modes in the .frd after the increments of the static step – correction 17. 10. 2026)
                Check("normování φᵀMφ = 1 po předpětí (1 + chyba)", 1 + res.dynamics.checks["mass_norm_error"], 1.0,
                      1e-3, "")]
    return ev


DYN_CASES.update({
    "Y9": Case("Y9", "Předpjatá modální analýza – tah 0,5·N_kr", y9_build, _y9_eval(0.5 * N_CR9)),
    "Y9c": Case("Y9c", "Předpjatá modální analýza – tlak 0,5·N_kr", y9c_build, _y9_eval(-0.5 * N_CR9)),
})


# ------------------------------------------------------------------ Y7 rotating blade (Campbell, D5b)
def y7_build(wd: Path):
    st, parts = _study("Y7 rotujici lopatka", wd / "model.step", [Part("Lopatka", box(0, -10, -2.5, 200, 20, 5))],
                       MeshSettings(size=2.5))
    st.add_selection("koren", find_faces(parts, x=0))
    st.constraints.append(Fixed("koren"))
    ds = st.dynamics
    ds.enabled, ds.modes = True, 3
    pr = ds.prestress
    pr.enabled, pr.rotating = True, True
    pr.rpm_list = [0.0, 3000.0, 6000.0]
    pr.axis_point, pr.axis = [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    return st, parts


def y7_eval(res, st):
    cb = res.dynamics.campbell
    F = np.asarray(cb["freq"])
    f0, f6 = float(F[0, 0]), float(F[2, 0])
    K = (f6 ** 2 - f0 ** 2) / 100.0 ** 2                  # Ω/2π = 100 Hz
    return [Check("Southwellův součinitel 1. ohybu mimo rovinu otáčení (1,193)", K, 1.1933, 2e-2, ""),
            Check("f₁ při 0 ot/min (E–B 104,4 Hz)", f0, 1.8751 ** 2 / (2 * math.pi * 200.0 ** 2)
                  * math.sqrt(E * (20 * 5 ** 3 / 12) / (RHO * 100.0)), 1e-2, "Hz")]


DYN_CASES["Y7"] = Case("Y7", "Rotující lopatka – Campbell, Southwell", y7_build, y7_eval)
