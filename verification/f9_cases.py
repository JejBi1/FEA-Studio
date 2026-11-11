"""Verification cases of stages F9.1 and F9.2:
- AS1: Extended Stress Tool (Tresca, von Mises on the torsion V7, Mohr-Coulomb for cast iron tension / compression / shear)
- FB1: Force by a cut (Free Body Cut on the cantilever V1 in 5 cuts and on the torsion V7)
- CS1: Local coordinate system of the results (V4 cylindrical coordinates: sigma_theta, sigma_r, u_r)
- HT1: Heat transfer calculator D4 (natural convection Churchill-Chu, forced Gnielinski)
- LC1: Load combinations and envelopes (superposition LC1+LC2 vs the direct state LC3, EN 1990 factors, nodal envelope)
- IM1: Face split (Imprint / Split Face: circular imprint r=10 mm with pressure 1 MPa)
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from fea_core.study import Study, Material, MeshSettings, Force, Fixed, Pressure, LoadCase
from fea_core.results import Results
from fea_core.selectors import find_faces
from fea_core.geometry import Part
from fea_core.assessment import compute_safety_field
from fea_core.section_forces import section_forces
from fea_core.csys import transform_stress, transform_displacements
from fea_core.calc.convection import natural_vertical_plate, natural_horizontal_cylinder, forced_internal_pipe
from fea_core.combinations import Combination, combine_results, compute_envelope
from fea_core.imprint import Imprint
from fea_core.mesh import tri_areas

from .cases import AtLeast, Check, Case, _study, box, v1_build, v7_build, v4_build, E, NU, RI, RO, V4_H, P4


F9_CASES: dict[str, Case] = {}


# ===================================================================== AS1
def as1_build(wd: Path):
    """AS1: Shaft from V7 for checking Tresca vs Mises and a body of grey cast iron for Mohr-Coulomb."""
    return v7_build(wd)


def as1_eval(res: Results, st: Study):
    D7 = 20.0
    T7 = 100000.0
    tau_theor = 16.0 * T7 / (math.pi * (D7 ** 3))
    Re = 235.0

    s_mises_theor = Re / (math.sqrt(3.0) * tau_theor)
    s_tresca_theor = Re / (2.0 * tau_theor)

    step = res.step(0)
    mesh = res.mesh

    r = np.hypot(mesh.nodes[:, 1], mesh.nodes[:, 2])
    mid_surf = np.nonzero((np.abs(r - 10.0) < 0.2) & (mesh.nodes[:, 0] > 30.0) & (mesh.nodes[:, 0] < 70.0))[0]

    import copy
    s_mises_field, _ = compute_safety_field(st, mesh, step, criterion="mises", limit="Re")
    s_tresca_field, _ = compute_safety_field(st, mesh, step, criterion="tresca", limit="Re")

    s_mises_num = float(s_mises_field[mid_surf].mean())
    s_tresca_num = float(s_tresca_field[mid_surf].mean())

    mat_cast_iron = Material("LITINA_GJL250", E=110000.0, nu=0.26, rho=7.2e-9, Rm=250.0, Rc=800.0)
    st_iron = copy.copy(st)
    st_iron.materials = dict(st.materials)
    st_iron.materials["STEEL"] = mat_cast_iron
    s_mc_field, _ = compute_safety_field(st_iron, mesh, step, criterion="mohr_coulomb", limit="Rm")
    s_mc_num = float(s_mc_field[mid_surf].mean())
    s_mc_theor = 1.0 / (tau_theor / 250.0 + tau_theor / 800.0)

    return [
        Check("AS1 Mises bezpecnost na krutu V7", s_mises_num, s_mises_theor, 0.05, "-"),
        Check("AS1 Tresca bezpecnost na krutu V7", s_tresca_num, s_tresca_theor, 0.05, "-"),
        Check("AS1 Mohr-Coulomb litina na krutu V7", s_mc_num, s_mc_theor, 0.07, "-"),
    ]


F9_CASES["AS1"] = Case("AS1", "Rozsireny Stress Tool: Tresca a Mohr-Coulomb na krutu V7", as1_build, as1_eval)


# ===================================================================== FB1
def fb1_build(wd: Path):
    """FB1: Cantilever V1 for checking the Free Body Cut (internal forces in 5 cuts)."""
    st, parts = v1_build(wd)
    st.section_forces = True
    return st, parts


def fb1_eval(res: Results, st: Study):
    checks = []
    cuts_x = [30.0, 60.0, 100.0, 140.0, 170.0]

    for x in cuts_x:
        origin = [x, 10.0, 5.0]
        normal = [1.0, 0.0, 0.0]
        res_cut = section_forces(st, res.mesh, res.step(0), origin, normal)
        F = res_cut["F_local"]
        M = res_cut["M_local"]

        M_theor = 100.0 * (200.0 - x)
        checks.append(Check(f"FB1 rez x={x:.0f} Vz", F[2], -100.0, 0.05, "N"))
        checks.append(Check(f"FB1 rez x={x:.0f} My k tezisti", M[1], M_theor, 0.02, "N*mm"))
        checks.append(Check(f"FB1 rez x={x:.0f} N", abs(F[0]), 0.0, 0.5, "N"))

    # Second test: torsion V7 with a section perpendicular to the shaft axis
    import tempfile
    wd = Path(tempfile.mkdtemp(prefix="fb1_v7_"))
    (wd / "v7").mkdir(parents=True, exist_ok=True)
    st7, parts7 = v7_build(wd / "v7")
    st7.section_forces = True
    from fea_core.solve import run_study
    res7 = run_study(st7, workdir=wd / "v7_run", parts=parts7)
    res_cut7 = section_forces(st7, res7.mesh, res7.step(0), [50.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    checks.append(Check("FB1 V7 kroutici moment Mx", res_cut7["M_local"][0], 100000.0, 0.02, "N*mm"))

    return checks


F9_CASES["FB1"] = Case("FB1", "Free Body Cut: sily a momenty v rezech na konzole V1 a krutu V7", fb1_build, fb1_eval)


# ===================================================================== CS1
def cs1_build(wd: Path):
    """CS1: Thick-walled cylinder V4 under internal pressure for checking the transformation to the cylindrical CSYS."""
    return v4_build(wd)


def cs1_eval(res: Results, st: Study):
    step = res.step(0)
    mesh = res.mesh

    # Cylindrical system with the centre on the axis of the cylinder V4 (axis of the cylinder is global Z)
    origin = [0.0, 0.0, 0.0]
    z_axis = [0.0, 0.0, 1.0]
    x_axis = [1.0, 0.0, 0.0]   # theta = 0 along X

    from fea_core.study import CoordinateSystem
    st.coordinate_systems["Valec"] = CoordinateSystem("Valec", "cylindrical", origin, z_axis, x_axis)
    S_cyl = transform_stress(st, "Valec", step.S, mesh.nodes)
    U_cyl = transform_displacements(st, "Valec", step.U, mesh.nodes)

    r = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
    x = mesh.nodes[:, 2]

    # Inner and outer surface at the middle of the length (avoid the end effects)
    mid_len = np.abs(x - V4_H / 2.0) < 1.3                 # middle height of V4 (axis of the cylinder = Z, height 10 mm)
    inner_nodes = np.nonzero((np.abs(r - RI) < 0.2) & mid_len)[0]
    outer_nodes = np.nonzero((np.abs(r - RO) < 0.2) & mid_len)[0]

    # Lamé analytic solution
    # sigma_theta(r) = P * ri^2 / (ro^2 - ri^2) * (1 + ro^2 / r^2)
    # sigma_r(r)     = P * ri^2 / (ro^2 - ri^2) * (1 - ro^2 / r^2)
    s_th_in_theor = P4 * (RI ** 2) / (RO ** 2 - RI ** 2) * (1.0 + (RO / RI) ** 2)
    s_th_out_theor = P4 * (RI ** 2) / (RO ** 2 - RI ** 2) * (1.0 + 1.0)
    s_r_in_theor = -P4
    s_r_out_theor = 0.0

    # Averages from the transformed field
    # Cylindrical tensor: S_rr (index 0), S_th_th (index 1), S_zz (index 2)
    s_th_in_num = float(S_cyl[inner_nodes, 1].mean())
    s_th_out_num = float(S_cyl[outer_nodes, 1].mean())
    s_r_in_num = float(S_cyl[inner_nodes, 0].mean())
    s_r_out_num = float(S_cyl[outer_nodes, 0].mean())

    # Radial displacement u_r (index 0 in U_cyl)
    u_r_in_theor = (P4 * RI / E) * ((RO ** 2 + RI ** 2) / (RO ** 2 - RI ** 2) + NU)
    u_r_in_num = float(U_cyl[inner_nodes, 0].mean())

    return [
        Check("CS1 sigma_theta vnitrni povrch", s_th_in_num, s_th_in_theor, 0.05, "MPa"),
        Check("CS1 sigma_theta vnejsi povrch", s_th_out_num, s_th_out_theor, 0.05, "MPa"),
        Check("CS1 sigma_r vnitrni tlak (-100 MPa)", s_r_in_num, s_r_in_theor, 0.07, "MPa"),
        # free outer surface: σr = 0 (ref 0 → absolute tolerance, 0.5 MPa = 0.5 % of the 100 MPa pressure)
        Check("CS1 sigma_r vnejsi povrch (0 MPa)", abs(s_r_out_num), 0.0, 0.5, "MPa"),
        Check("CS1 radialni posuv u_r vnitrni", u_r_in_num, u_r_in_theor, 0.05, "mm"),
    ]


F9_CASES["CS1"] = Case("CS1", "Mistni valcovy souradny system: Lameho napeti na valci V4", cs1_build, cs1_eval)


# ===================================================================== HT1
def ht1_build(wd: Path):
    """HT1: Analytic check of the heat transfer calculator D4 without an FEM run."""
    return v1_build(wd)


def ht1_eval(res: Results, st: Study):
    # 1. Natural convection on a vertical plate H = 1 m, Ts = 60 °C, Tinf = 20 °C in air
    h_vert, Nu_vert, Ra_vert = natural_vertical_plate(height_m=1.0, t_surface_c=60.0, t_ambient_c=20.0, fluid="air")
    # Hand calculation Churchill-Chu with the air properties from Incropera (table A.4, T_film = 313 K: ν 17.2e-6, α 24.4e-6,
    # k 0.0273, Pr 0.705): Ra = 2.98e9, Nu = 172, h = 4.70 W/(m2*K) (the earlier 4.41 came from a wrong conductivity of air)
    h_vert_ref = 4.70

    # 2. Natural convection on a horizontal cylinder D = 0.05 m, Ts = 80 °C, Tinf = 20 °C in air
    h_cyl, Nu_cyl, Ra_cyl = natural_horizontal_cylinder(diameter_m=0.05, t_surface_c=80.0, t_ambient_c=20.0, fluid="air")
    # Churchill-Chu cylinder: typically ~6.8–7.2 W/(m2*K)
    h_cyl_ref = 6.95

    # 3. Forced convection in a pipe D = 0.02 m, v = 1.0 m/s, water at 40 °C
    h_pipe, Nu_pipe, Re_pipe = forced_internal_pipe(diameter_m=0.02, velocity_m_s=1.0, t_bulk_c=40.0, fluid="water")
    # Hand calculation Dittus-Boelter (n = 0.4, wall 5 K warmer, T_film 42.5 °C; Incropera table A.6: Re = 3.1e4,
    # Pr = 4.16, k = 0.634): Nu = 161, h = 5110 W/(m2*K) (the earlier 4650 was a "typical" estimate)
    h_pipe_ref = 5100.0

    return [
        Check("HT1 svisla deska prirozena konvekce h", h_vert, h_vert_ref, 0.06, "W/(m2*K)"),
        Check("HT1 vodorovny valec prirozena konvekce h", h_cyl, h_cyl_ref, 0.06, "W/(m2*K)"),
        Check("HT1 nucena konvekce voda v trubce h", h_pipe, h_pipe_ref, 0.08, "W/(m2*K)"),
    ]


F9_CASES["HT1"] = Case("HT1", "Kalkulacka prestupu tepla D4: Churchill-Chu a Gnielinski", ht1_build, ht1_eval)


# ===================================================================== LC1 (Load combinations and envelopes)
def lc1_build(wd: Path):
    """LC1: Cantilever V1 with two load states: permanent G (60 N) and variable Q (40 N)."""
    wd.mkdir(parents=True, exist_ok=True)
    beam = box(0, 0, 0, 200, 20, 10)
    geo = [Part("Konzola", beam)]
    st, parts = _study("LC1 kombinace konzola", wd / "model.step", geo, MeshSettings(size=2.5))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=200))
    st.constraints.append(Fixed("vetknuti"))

    st.load_cases.append(LoadCase("G_60N", [Force("konec", [0, 0, -60.0])]))
    st.load_cases.append(LoadCase("Q_40N", [Force("konec", [0, 0, -40.0])]))
    st.load_cases.append(LoadCase("GQ_100N_direct", [Force("konec", [0, 0, -100.0])]))
    # asymmetric pair of states for the envelope: tension N (Fx) and bending G (Fz); +N/−N gives tension/compression on the upper/lower face
    st.load_cases.append(LoadCase("N_tah_4kN", [Force("konec", [4000.0, 0, 0])]))

    st.combinations.append(Combination(
        name="Comb_1_to_1",
        factors={"G_60N": 1.0, "Q_40N": 1.0},
        standard="custom",
        design_situation="custom",
        description="Přímý součet 1.0*G + 1.0*Q",
    ))
    st.combinations.append(Combination(
        name="Comb_EN1990_STR",
        factors={"G_60N": 1.35, "Q_40N": 1.50},
        standard="CSN_EN_1990",
        design_situation="STR_GEO_6_10",
        description="ČSN EN 1990 MSÚ STR 6.10: 1.35*G + 1.50*Q",
    ))
    st.combinations.append(Combination(
        name="Envelope_max",
        envelope=True,
        envelope_type="max",
        envelope_combinations=["G_60N", "Q_40N", "GQ_100N_direct"],
        envelope_field="von_mises",
        description="Obálka max von Mises přes stavy",
    ))
    # non-proportional combination: bending G + tension N (upper face tension+tension) and bending G − tension N (lower face compression+compression)
    st.combinations.append(Combination(
        name="Comb_G_plus_N", factors={"G_60N": 1.0, "N_tah_4kN": 0.5}, standard="custom", design_situation="custom"))
    st.combinations.append(Combination(
        name="Comb_G_minus_N", factors={"G_60N": 1.0, "N_tah_4kN": -0.5}, standard="custom", design_situation="custom"))
    st.combinations.append(Combination(
        name="Envelope_neprop",
        envelope=True,
        envelope_type="max",
        envelope_combinations=["Comb_G_plus_N", "Comb_G_minus_N"],
        envelope_field="von_mises",
        description="Obálka max von Mises přes dvě neproporcionální kombinace",
    ))

    return st, parts


def lc1_eval(res: Results, st: Study):
    s_direct = res.step(2)

    comb1 = st.combinations[0]
    comb1_res = combine_results(st, res, comb1)

    max_delta_u = float(np.max(np.abs(comb1_res.U - s_direct.U)))
    max_delta_s = float(np.max(np.abs(comb1_res.S - s_direct.S)))
    rf_diff = float(np.max(np.abs(comb1_res.reactions["vetknuti"] - s_direct.reactions["vetknuti"])))

    u_scale = float(np.max(np.abs(s_direct.U)))
    s_scale = float(np.max(s_direct.mises))
    rf_scale = 100.0

    comb_en = st.combinations[1]
    comb_en_res = combine_results(st, res, comb_en)

    m = res.mesh
    end_nodes = np.nonzero(np.abs(m.nodes[:, 0] - 200) < 1e-6)[0]
    u_direct_end = float(s_direct.U[end_nodes, 2].mean())
    u_en_end = float(comb_en_res.U[end_nodes, 2].mean())
    u_en_theor = 1.41 * u_direct_end

    # proportional envelope (only a check that the state with the largest load decides)
    env_comb = [c for c in st.combinations if c.name == "Envelope_max"][0]
    env_res = compute_envelope(st, res, env_comb)
    gov_ratio = env_res.governing_summary.get("GQ_100N_direct", 0) / len(m.nodes)

    # non-proportional envelope: independent reference from the nodal tensors of the states (numpy in the test, own von Mises formula)
    def vm(S):
        return np.sqrt(0.5 * ((S[:, 0] - S[:, 1]) ** 2 + (S[:, 1] - S[:, 2]) ** 2 + (S[:, 2] - S[:, 0]) ** 2)
                       + 3.0 * (S[:, 3] ** 2 + S[:, 4] ** 2 + S[:, 5] ** 2))
    S_g = res.step("G_60N").S
    S_n = res.step("N_tah_4kN").S
    v_plus = vm(S_g + 0.5 * S_n)
    v_minus = vm(S_g - 0.5 * S_n)
    v_env = np.maximum(v_plus, v_minus)
    env2 = compute_envelope(st, res, [c for c in st.combinations if c.name == "Envelope_neprop"][0])
    s_scale2 = float(v_env.max())
    d_env = float(np.max(np.abs(env2.values - v_env)) / s_scale2)
    # governing combination per node (without nodes with almost equal values, where the order is noise)
    clear = np.abs(v_plus - v_minus) > 1e-4 * s_scale2
    ref_gov = np.where(v_plus >= v_minus, "Comb_G_plus_N", "Comb_G_minus_N")
    gov_match = float(np.mean(np.asarray(env2.governing_combinations)[clear] == ref_gov[clear]))
    gov_min_frac = min(env2.governing_summary.values()) / len(m.nodes)

    return [
        Check("LC1 superpozice posuv delta_u relativne", max_delta_u / u_scale, 0.0, 1e-4, "-"),
        Check("LC1 superpozice napeti delta_S relativne", max_delta_s / s_scale, 0.0, 1e-4, "-"),
        Check("LC1 reakce Fz superpozice relativne", rf_diff / rf_scale, 0.0, 1e-4, "-"),
        Check("LC1 EN 1990 pruhyb konce (1.41x)", u_en_end, u_en_theor, 1e-5, "mm"),
        Check("LC1 obalka (proporcionalni): podil uzlu ovladanych 100N stavem", gov_ratio, 1.0, 0.001, "-"),
        Check("LC1 obalka (neproporcionalni G+-N): max over uzly vs numpy z tenzoru, relativne", d_env, 0.0, 1e-6, "-"),
        Check("LC1 obalka (neproporcionalni): shoda rozhodujici kombinace po uzlech", gov_match, 1.0, 1e-9, "-"),
        # asymmetric load → each of the two combinations governs a substantial part of the nodes (upper × lower face), otherwise the test is trivial
        AtLeast("LC1 obalka (neproporcionalni): min. podil uzlu jedne kombinace", gov_min_frac, 0.2, 1e-9, "-"),
    ]


F9_CASES["LC1"] = Case("LC1", "Kombinace zateze a obalky: superpozice vs primy vypocet a EN 1990", lc1_build, lc1_eval)


# ===================================================================== IM1 (Imprint on a face / Split Face)
def im1_build(wd: Path):
    """IM1: Plate 100x100x10 clamped at x=0 with a circular imprint r=10 mm in the middle."""
    wd.mkdir(parents=True, exist_ok=True)
    plate = Part("Deska", box(0, 0, 0, 100, 100, 10))
    st, parts = _study("IM1 otisk kruhu", wd / "model.step", [plate], MeshSettings(size=5.0))
    st.imprints.append(Imprint(
        name="raznik_d20",
        part="Deska",
        center=[50.0, 50.0, 10.0],
        normal=[0.0, 0.0, 1.0],
        shape="circle",
        radius=10.0,
        size=2.5,
    ))
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Tlak_razniku", [Pressure("raznik_d20", 1.0)]))
    return st, parts


def im1_eval(res: Results, st: Study):
    step = res.step(0)
    mesh = res.mesh
    tris = mesh.surfaces["raznik_d20"]
    area = float(np.sum(tri_areas(mesh.nodes, tris)))
    theor_area = math.pi * (10.0 ** 2)  # 314.159 mm2
    theor_force = 1.0 * theor_area     # 314.159 N

    rf_z = step.reactions["vetknuti"][2]
    rm_y = float(step.reaction_moments["vetknuti"][1]) if step.reaction_moments else 0.0

    end_nodes = np.nonzero(np.abs(mesh.nodes[:, 0] - 100) < 1e-6)[0]
    u_end = float(step.U[end_nodes, 2].mean())

    # Analytic estimate of the deflection of the free end: cantilever of width b = 100, thickness h = 10, length L = 100, force F = p·π·r²
    # at the position a = 50 (centre of the imprint): w = F·a²·(3L − a)/(6·D·b) with the stiffness of the wide strip D = E·h³/(12·(1 − ν²)) (cylindrical
    # bending) + the shear contribution F·a/(κ·G·b·h), κ = 5/6. Estimate for a surface force uniform across the width; the real load is
    # concentrated (r/b = 0.1) and the sides are free → deviation of the order of 10 %, hence the tolerance 10 %.
    Lc, ac, bc, hc = 100.0, 50.0, 100.0, 10.0
    Dc = E * hc ** 3 / (12.0 * (1.0 - NU ** 2))
    w_est = theor_force * ac ** 2 * (3.0 * Lc - ac) / (6.0 * Dc * bc) \
        + theor_force * ac / (5.0 / 6.0 * (E / (2.0 * (1.0 + NU))) * bc * hc)

    return [
        Check("IM1 plocha otisku kruhu (pi*r^2)", area, theor_area, 0.03, "mm2"),
        Check("IM1 reakce Fz na vetknuti (p*A)", rf_z, theor_force, 0.005, "N"),
        Check("IM1 reakcni moment My k c0 (v ose zatizeni = 0)", abs(rm_y), 0.0, 0.05, "N*mm"),
        Check("IM1 pruhyb volneho konce x=100 (konzola, siroky pas + smyk)", u_end, -w_est, 0.10, "mm"),
    ]


F9_CASES["IM1"] = Case("IM1", "Rozdeleni plochy: otisk kruhu r=10mm s tlakem 1 MPa", im1_build, im1_eval)


# ===================================================================== SM1 (Submodel)
SM1_DX, SM1_DY, SM1_DZ, SM1_R, SM1_F = 200.0, 100.0, 10.0, 10.0, 10000.0


def sm1_peterson() -> tuple[float, float]:
    """Reference maximum stress at the edge of a hole in a strip of finite width (Heywood / Peterson, Roark table 17.1):
    Kt,net = 3.00 − 3.14·(d/W) + 3.667·(d/W)² − 1.527·(d/W)³ related to the net section σ_net = F / ((W − d)·t).
    Returns (Kt,net, σ_max). For d/W = 0.2: Kt,net = 2.507 (Kt,gross = Kt,net / (1 − d/W) = 3.13 related to the gross section).
    """
    dW = 2.0 * SM1_R / SM1_DY
    kt_net = 3.00 - 3.14 * dW + 3.667 * dW ** 2 - 1.527 * dW ** 3
    s_net = SM1_F / ((SM1_DY - 2.0 * SM1_R) * SM1_DZ)
    return kt_net, kt_net * s_net


def sm1_build(wd: Path):
    """SM1: Plate 200x100x10 with a hole r=10 mm in tension. Global model with a coarse mesh (h=14 mm)."""
    wd.mkdir(parents=True, exist_ok=True)
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir

    dx, dy, dz, r = SM1_DX, SM1_DY, SM1_DZ, SM1_R
    box_s = BRepPrimAPI_MakeBox(gp_Ax2(gp_Pnt(-dx/2, -dy/2, 0.0), gp_Dir(0, 0, 1)), dx, dy, dz).Shape()
    cyl_s = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1.0), gp_Dir(0, 0, 1)), r, dz + 2.0).Shape()
    cut = BRepAlgoAPI_Cut(box_s, cyl_s)
    cut.Build()
    plate = Part("Deska", cut.Shape())

    st, parts = _study("SM1 globalni deska", wd / "model.step", [plate], MeshSettings(size=14.0))
    st.add_selection("vetknuti", find_faces(parts, x=-100.0))
    st.add_selection("tah", find_faces(parts, x=100.0))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Tah", [Force("tah", [SM1_F, 0.0, 0.0])]))
    return st, parts


def sm1_run(study: Study, workdir: Path, parts: list[Part]):
    """Runs the global model and then the submodel in the detail around the hole."""
    from fea_core.solve import run_study
    from fea_core.submodel import (SubmodelSpec, create_submodel_shape, generate_submodel_mesh,
                                   identify_cut_nodes, interpolate_cut_displacements,
                                   solve_submodel, verify_cut_boundary)

    # 1. Global calculation
    res_glob = run_study(study, workdir=workdir / "run_global", parts=parts)

    # 2. Submodel geometry and mesh (cut-out 60x60x10 mm around the centre)
    spec = SubmodelSpec(center=(0.0, 0.0, 5.0), size=(60.0, 60.0, 11.0), shape="box",
                        mesh_size_max=2.0, mesh_size_min=0.6)
    sub_shape = create_submodel_shape(parts[0].shape, spec)
    sub_coords, sub_tets = generate_submodel_mesh(sub_shape, spec, workdir=workdir / "mesh_submodel")

    # 3. Identification of the cut nodes and interpolation of the displacements from the global solution
    cut_nodes = identify_cut_nodes(sub_coords, spec, tol=1e-3)
    cut_U = interpolate_cut_displacements(res_glob, sub_coords, cut_nodes, step_index=0)

    # 4. Solution of the submodel
    mat = study.materials[study.default_material]
    res_sub = solve_submodel(sub_coords, sub_tets, mat, cut_nodes, cut_U, workdir / "run_submodel")

    # 5. Check of the cut
    cut_check = verify_cut_boundary(res_glob, res_sub, cut_nodes, step_index=0)

    class CombinedResult:
        def __init__(self, glob, sub, check):
            self.glob = glob
            self.sub = sub
            self.check = check
            self.mesh = sub.mesh
            self.warnings = glob.warnings + sub.warnings
            self.steps = sub.steps
            self.step = sub.step

    return CombinedResult(res_glob, res_sub, cut_check)


def sm1_eval(res, st: Study):
    kt_net, sigma_theor = sm1_peterson()          # 2,507 · 12,5 MPa = 31,33 MPa

    if hasattr(res, "glob"):
        glob_max_vm = float(res.glob.step(0).mises.max())
        # largest PRINCIPAL stress σ1 (= σθ at the edge of the hole); von Mises of the 3D body at the middle of the thickness decreases (σz)
        sub_max_s1 = float(res.sub.step(0).principal[:, 0].max())
        mean_cut_diff = res.check["mean_rel_diff"]
    else:                      # without its own run (e.g. only run_study) the submodel does not exist → the checks will FAIL
        glob_max_vm = float(res.step(0).mises.max())
        sub_max_s1 = float("nan")
        mean_cut_diff = float("nan")

    return [
        # a coarse mesh (h = 14 mm) underestimates the notch peak; the reference is the same Peterson relation, tolerance 8 %
        Check("SM1 hruby globalni model max von Mises (h=14mm) vs Peterson", glob_max_vm, sigma_theor, 0.08, "MPa"),
        # Comparison of σ1 (principal stress) with the Heywood / Peterson polynomial (kt_net 2.507), plate t = 10 mm, Ø20 (t/d = 0.5).
        # Breakdown of the measured deviation 32.82 MPa = +4.7 % (scratch/a10/sm1_bc2d.py, sm1_thick.py, h at the hole 0.15–0.3 mm):
        # polynomial 31.33 → converged 2D infinite strip (half model, roller support, L = 600): 31.98 MPa = +2.1 %;
        # boundary conditions of SM1 (clamped end, L = 200): 32.18 MPa = +0.6 %;
        # 3D middle of the thickness t/d = 0.5 (h = 0.3 mm): 32.59 MPa = +1.3 %; submodel against the coarse 3D model: +0.7 %.
        # The tolerance 5 % = the sum of these components (reference + 3D + mesh) with a margin of ~0.3 %. The surface of the hole 30.15 MPa (σθ) is lower.
        Check(f"SM1 submodel max σ1 vs Peterson/Heywood Kt,net = {kt_net:.3f}, h=0.6mm", sub_max_s1, sigma_theor,
              0.05, "MPa"),
        Check("SM1 kontrola rezu: stredni odchylka napeti na rezu < 10%", mean_cut_diff, 0.0, 0.10, "-"),
    ]


F9_CASES["SM1"] = Case("SM1", "Submodel (C1): deska s otvorem pod tahem, Peterson/Heywood Kt,net = 2,51 (Kt,gross = 3,13)",
                       sm1_build, sm1_eval, run=sm1_run)

from .submodel_cases import SUBMODEL_CASES  # noqa: E402 – SM2: submodel of a solved study (GUI / MCP path)

F9_CASES.update(SUBMODEL_CASES)
