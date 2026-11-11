"""Assessment of the members to EN 1993-1-1 (`fea_core.beams.ec3`) against hand calculations.

The references were calculated independently step by step according to the standard (`scratch/beams_ec3/ref.py`, without fea_core); the sections
are built in the test (table values of IPE 300 and HEB 200, formulas for tubes with sharp corners) – they do not depend on
`beams/sections.py`.
"""
from math import pi

import numpy as np
import pytest

from fea_core.beams.ec3 import (EC3Error, INFO, MemberForces, MemberSpec, c1_factor, check_member, classify,
                                cm_factor, fy_of, moment_diagram)
from fea_core.beams.props import SectionProps


def rel(a, b):
    return abs(a - b) / abs(b)


def ipe300() -> SectionProps:
    A, b, tf, tw, r = 5380.0, 150.0, 10.7, 7.1, 15.0
    return SectionProps("IPE 300", "I", {"h": 300.0, "b": b, "tw": tw, "tf": tf, "r": r}, A=A, Iy=8356e4, Iz=604e4,
                        J=20.1e4, Iw=125.9e9, Wel_y=557e3, Wel_z=80.5e3, Wpl_y=628e3, Wpl_z=125e3,
                        Av_z=A - 2 * b * tf + (tw + 2 * r) * tf, Av_y=2 * b * tf, Wt=20.1e4 / tf)


def heb200() -> SectionProps:
    return SectionProps("HEB 200", "I", {"h": 200.0, "b": 200.0, "tw": 9.0, "tf": 15.0, "r": 18.0}, A=7808.0,
                        Iy=5696e4, Iz=2003e4, J=59.28e4, Iw=171.1e9, Wel_y=569.6e3, Wel_z=200.3e3, Wpl_y=642.5e3,
                        Wpl_z=305.8e3, Av_z=2483.0, Av_y=6000.0, Wt=59.28e4 / 15.0)


def rhs(h, b, t, name=None) -> SectionProps:
    """Thin-walled rectangular tube with sharp corners (properties by formulas)."""
    hi, bi = h - 2 * t, b - 2 * t
    A = h * b - hi * bi
    Am, p = (h - t) * (b - t), 2 * (h + b - 2 * t)
    return SectionProps(name or f"RHS {h:g}×{b:g}×{t:g}", "RHS", {"h": h, "b": b, "t": t}, A=A,
                        Iy=(b * h ** 3 - bi * hi ** 3) / 12, Iz=(h * b ** 3 - hi * bi ** 3) / 12,
                        J=4 * Am ** 2 * t / p, Iw=0.0,
                        Wel_y=(b * h ** 3 - bi * hi ** 3) / (6 * h), Wel_z=(h * b ** 3 - hi * bi ** 3) / (6 * b),
                        Wpl_y=(b * h ** 2 - bi * hi ** 2) / 4, Wpl_z=(h * b ** 2 - hi * bi ** 2) / 4,
                        Av_z=A * h / (b + h), Av_y=A * b / (b + h), Wt=2 * Am * t)


def chs(D, t) -> SectionProps:
    d = D - 2 * t
    A = pi / 4 * (D ** 2 - d ** 2)
    I = pi / 64 * (D ** 4 - d ** 4)
    return SectionProps(f"CHS {D:g}×{t:g}", "CHS", {"D": D, "t": t}, A=A, Iy=I, Iz=I, J=2 * I, Wel_y=2 * I / D,
                        Wel_z=2 * I / D, Wpl_y=(D ** 3 - d ** 3) / 6, Wpl_z=(D ** 3 - d ** 3) / 6,
                        Av_y=2 * A / pi, Av_z=2 * A / pi, Wt=4 * I / D)


def const(L, **v):
    x = np.linspace(0.0, L, 21)
    return MemberForces(x, **{k: np.full_like(x, val) for k, val in v.items()})


# --------------------------------------------------------------------------- material

def test_fy_table():
    assert fy_of("S235", 10) == 235 and fy_of("S235JR", 50) == 215
    assert fy_of("S355J2H", 40) == 355 and fy_of("S355", 40.5) == 335
    assert fy_of("S460", 12) == 460 and fy_of("S275", 60) == 255
    with pytest.raises(EC3Error):
        fy_of("S355", 100)
    with pytest.raises(EC3Error):
        fy_of("S690", 10)


# --------------------------------------------------------------------------- (a) IPE 300 S235 – compression, buckling

def test_a_ipe300_compression_buckling():
    sec = ipe300()
    L = 5000.0
    chk = check_member(sec, const(L, N=-350e3), MemberSpec(L=L, grade="S235"))
    assert chk.fy == 235
    assert chk.section_class == 2                       # web in compression c/t = 35.01 ∈ (33ε; 38ε]
    web = next(p for p in chk.classification if p["part"] == "web")
    assert web["ratio"] == pytest.approx(35.0141, rel=1e-4)
    d = chk.data
    assert d["curve_y"] == "a" and d["curve_z"] == "b"
    assert rel(d["Ncr_z"], 500744.2) < 1e-5
    assert rel(d["lambda_z"], 1.58898) < 1e-5
    assert rel(d["chi_z"], 0.31146) < 1e-4
    assert rel(d["chi_y"], 0.94548) < 1e-4
    rz = chk.row("Buckling about z")
    assert rel(rz.resistance, 393775.5) < 1e-5
    assert rel(rz.utilization, 0.888831) < 1e-5
    rt = chk.row("Torsional / torsional-flexural buckling")
    assert rel(d["Ncr_T"], 1604314.7) < 1e-5
    assert rel(rt.resistance, 845897.1) < 1e-4
    assert chk.governing.startswith("Buckling about z")
    assert rel(chk.utilization, 0.888831) < 1e-5
    assert rel(chk.row("Compression").resistance, 1264300.0) < 1e-9


def test_a2_ipe300_bending_axial_section():
    """6.2.9.1: n = 0,316 > 0,25 → MN,y,Rd = Mpl,y,Rd(1 − n)/(1 − 0,5a) = 126,37 kN·m."""
    sec = ipe300()
    chk = check_member(sec, const(3000.0, N=-400e3, My=100e6), MemberSpec(L=3000.0, lt_restrained=True,
                                                                           Lcr_z=1.0, Lcr_y=1.0, Lcr_T=1.0))
    web = next(p for p in chk.classification if p["part"] == "web")
    assert web["alpha"] == pytest.approx(0.9822, abs=2e-3)   # α = (c/2 + N/(2·tw·fy))/c
    assert chk.section_class == 2
    r = chk.row("Bending and axial force")
    assert rel(r.utilization, 0.79130) < 1e-4


# --------------------------------------------------------------------------- (b) HEB 200 S355 – lateral-torsional buckling

def test_b_heb200_lateral_torsional_uniform():
    sec = heb200()
    L = 6000.0
    chk = check_member(sec, const(L, My=150e6), MemberSpec(L=L, grade="S355", C1=1.0))
    assert chk.section_class == 1
    d = chk.data
    assert rel(d["Mcr"], 258.3248e6) < 1e-5
    assert rel(d["lambda_LT"], 0.93965) < 1e-5
    assert d["curve_LT"] == "b" and d["lt_method"] == "rolled"
    assert rel(d["chi_LT"], 0.73633) < 1e-4
    r = chk.row("Lateral-torsional buckling")
    assert rel(r.resistance, 167.9475e6) < 1e-5
    assert rel(r.utilization, 150 / 167.9475) < 1e-5
    # general case 6.3.2.2 (curve a)
    g = check_member(sec, const(L, My=150e6), MemberSpec(L=L, grade="S355", C1=1.0, lt_method="general"))
    assert g.data["curve_LT"] == "a"
    assert rel(g.row("Lateral-torsional buckling").resistance, 161.3387e6) < 1e-5


def test_b_heb200_lateral_torsional_linear_psi0():
    """Linear moment ψ = 0: C1 = 1.88 from the diagram, kc = 1/1.33, f = 0.8792 → χLT,mod = 0.9974."""
    sec = heb200()
    L = 6000.0
    x = np.linspace(0, L, 41)
    My = 200e6 * (1 - x / L)
    Vz = np.full_like(x, -200e6 / L)
    chk = check_member(sec, MemberForces(x, My=My, Vz=Vz), MemberSpec(L=L, grade="S355"))
    d = chk.data
    assert d["diagram_y"]["kind"] == "linear" and d["diagram_y"]["psi"] == pytest.approx(0.0, abs=1e-12)
    assert d["C1"] == pytest.approx(1.88)
    assert rel(d["Mcr"], 485.6506e6) < 1e-5
    assert rel(d["f"], 0.87920) < 1e-4
    assert rel(d["chi_LT"], 0.99740) < 1e-4
    assert rel(chk.row("Lateral-torsional buckling").resistance, 227.4937e6) < 1e-5
    # section checks along the length: the worst section at x = 0
    assert chk.x_crit == 0.0
    assert rel(chk.row("Bending My (6.2.5)").utilization, 200e6 / (642.5e3 * 355)) < 1e-9


# --------------------------------------------------------------------------- (c) RHS – interaction, annex B

def test_c_rhs_compression_biaxial_annex_b():
    sec = rhs(150.0, 100.0, 5.0)
    L = 3000.0
    x = np.linspace(0, L, 31)
    f = MemberForces(x, N=np.full_like(x, -200e3), My=15e6 * (1 - x / L), Mz=np.full_like(x, 5e6),
                     Vz=np.full_like(x, -15e6 / L))
    chk = check_member(sec, f, MemberSpec(L=L, grade="S355", fabrication="cold_formed"))
    d = chk.data
    assert chk.section_class == 2                       # web h in compression c/t = 27 ∈ (33ε; 38ε]
    assert d["curve_y"] == "c" and "Klopení" not in [r.name.split(" (")[0] for r in chk.rows]
    assert rel(d["chi_y"], 0.72454) < 1e-4 and rel(d["chi_z"], 0.56209) < 1e-4
    assert d["Cmy"] == pytest.approx(0.6) and d["Cmz"] == pytest.approx(1.0)
    assert not d["susceptible"]
    k = d["annex_b"]
    assert rel(k["kyy"], 0.69725) < 1e-4 and rel(k["kzz"], 1.31837) < 1e-4
    assert rel(k["kyz"], 0.79102) < 1e-4 and rel(k["kzy"], 0.41835) < 1e-4
    assert rel(chk.row("N + M interaction – buckling about y").utilization, 0.68823) < 1e-4
    assert rel(chk.row("N + M interaction – buckling about z").utilization, 0.76605) < 1e-4
    r = chk.row("Bending and axial force")
    assert r.x == 0.0 and rel(r.utilization, 0.19466) < 1e-4
    assert chk.governing.startswith("N + M interaction – buckling about z")


# --------------------------------------------------------------------------- (d) CHS – classification, compression

@pytest.mark.parametrize("grade,cls,curve,Nb", [("S355", 1, "a", 788689.7), ("S460", 2, "a0", 980114.9)])
def test_d_chs_class_compression(grade, cls, curve, Nb):
    sec = chs(168.3, 6.0)
    L = 4000.0
    chk = check_member(sec, const(L, N=-500e3), MemberSpec(L=L, grade=grade, fabrication="hot_finished"))
    assert chk.section_class == cls                     # d/t = 28,05: 50ε² = 33,1 (S355), 25,5 / 35,8 (S460)
    assert chk.classification[0]["ratio"] == pytest.approx(28.05)
    assert chk.data["curve_y"] == curve
    r = chk.row("Buckling about y")
    assert rel(r.resistance, Nb) < 1e-5
    assert chk.row("Torsional / torsional-flexural buckling") is None and chk.row("Lateral-torsional buckling") is None


# --------------------------------------------------------------------------- (e) class 4 – RHS in compression

def test_e_shs_class4_effective_area():
    sec = rhs(200.0, 200.0, 4.0, "SHS 200×4")
    L = 3000.0
    chk = check_member(sec, const(L, N=-600e3), MemberSpec(L=L, grade="S355", fabrication="cold_formed"))
    assert chk.section_class == 4
    assert rel(chk.data["effective"]["Aeff"], 2445.864) < 1e-5
    assert rel(chk.row("Compression").resistance, 868281.6) < 1e-5
    assert rel(chk.row("Buckling about y").resistance, 763733.7) < 1e-5
    assert any("Class 4" in w for w in chk.warnings)


def welded_i(h=800.0, b=250.0, tw=6.0, tf=15.0) -> SectionProps:
    hw = h - 2 * tf
    Iy = (b * h ** 3 - (b - tw) * hw ** 3) / 12
    Iz = 2 * tf * b ** 3 / 12 + hw * tw ** 3 / 12
    return SectionProps("Svařovaný I 800×250", "I", {"h": h, "b": b, "tw": tw, "tf": tf, "r": 0.0},
                        A=2 * b * tf + hw * tw, Iy=Iy, Iz=Iz, J=(2 * b * tf ** 3 + hw * tw ** 3) / 3,
                        Iw=Iz * (h - tf) ** 2 / 4, Wel_y=Iy / (h / 2), Wel_z=Iz / (b / 2),
                        Wpl_y=b * tf * (h - tf) + tw * hw ** 2 / 4, Wpl_z=2 * tf * b ** 2 / 4 + hw * tw ** 2 / 4,
                        Av_z=hw * tw, Av_y=2 * b * tf, Wt=1.0)


def test_f_welded_i_class4_bending():
    """Stojina c/t = 128,3 > 124ε, ψ = −1: ρ = 0,795, W_eff,y = 3322,3 cm³ (EN 1993-1-5, 4.4)."""
    sec = welded_i()
    chk = check_member(sec, const(8000.0, My=400e6), MemberSpec(L=8000.0, grade="S355", fabrication="welded",
                                                                lt_restrained=True))
    assert chk.section_class == 4
    eff = chk.data["effective"]
    assert rel(eff["zc_y"], -9.0715) < 1e-4
    assert rel(eff["Weff_y"], 3.32229e6) < 1e-5
    assert rel(chk.row("Bending My (6.2.5)").resistance, 1179.4123e6) < 1e-5
    assert chk.row("Lateral-torsional buckling") is None


def test_class4_unsupported_shape():
    sec = chs(323.9, 2.0)                              # d/t = 162 > 90ε²
    with pytest.raises(EC3Error, match="Class 4 – not supported"):
        check_member(sec, const(2000.0, N=-100e3), MemberSpec(L=2000.0, grade="S355"))


# --------------------------------------------------------------------------- other parts

def test_classification_web_bending_vs_compression():
    sec = ipe300()
    assert classify(sec, 235.0, My=50e6).cls == 1          # web in bending c/t = 35 < 72ε
    assert classify(sec, 235.0, N=-1e3).cls == 2
    assert classify(sec, 235.0, N=+1e5).cls == 1           # tah
    c3 = classify(sec, 460.0, N=-1e3)                       # 42ε = 30.1 < 35 → class 4
    assert c3.cls == 4


def test_shear_and_bending_shear_interaction():
    """6.2.6 a 6.2.8: V > 0,5 Vpl → ρ = (2V/Vpl − 1)², M_y,V,Rd = (Wpl,y − ρ·Aw²/(4tw))·fy/γM0."""
    sec = ipe300()
    fy = 235.0
    Vpl = sec.Av_z * fy / 3 ** 0.5
    V = 0.8 * Vpl
    chk = check_member(sec, const(1000.0, Vz=V, My=50e6), MemberSpec(L=1000.0, lt_restrained=True))
    assert rel(chk.row("Shear Vz").utilization, 0.8) < 1e-9
    rho = (2 * 0.8 - 1) ** 2
    hw = 300 - 2 * 10.7
    MV = (sec.Wpl_y - rho * (hw * 7.1) ** 2 / (4 * 7.1)) * fy
    r = chk.row("Bending My and shear Vz")
    assert rel(r.resistance, MV) < 1e-9


def test_rows_informative_and_governing():
    sec = ipe300()
    chk = check_member(sec, const(2000.0, N=-100e3), MemberSpec(L=2000.0))
    info = [r for r in chk.rows if r.informative]
    assert info and info[0].name.startswith(INFO + "Classification")
    assert not chk.governing.startswith(INFO)
    assert chk.ok
    assert "PASS" in chk.text()


def test_moment_diagrams_cm_c1():
    L = 1000.0
    x = np.linspace(0, L, 101)
    q = 1.0
    fixed = q * L * x / 2 - q * x ** 2 / 2 - q * L ** 2 / 12     # clamped beam, distributed load
    dg = moment_diagram(x, fixed)
    assert dg["kind"] == "uniform" and dg["psi"] == pytest.approx(1.0)
    assert cm_factor(dg) == pytest.approx(0.5)                  # 0,1 − 0,8·αs, αs = −0,5
    simple = q * L * x / 2 - q * x ** 2 / 2
    dgs = moment_diagram(x, simple)
    assert cm_factor(dgs) == pytest.approx(0.95)
    assert c1_factor(dgs, x, simple)[0] == pytest.approx(12.5 / 11.0)   # Cb parabola ≈ 1,136
    point = np.where(x <= L / 2, x / 2, (L - x) / 2)                    # single force in the middle
    dgp = moment_diagram(x, point)
    assert dgp["kind"] == "concentrated" and cm_factor(dgp) == pytest.approx(0.90)
    lin = 1.0 - 1.5 * x / L                                              # ψ = −0,5
    dgl = moment_diagram(x, lin)
    assert dgl["kind"] == "linear" and cm_factor(dgl) == pytest.approx(0.4)
    assert c1_factor(dgl)[0] == pytest.approx(min(1.88 + 0.70 + 0.13, 2.7))
