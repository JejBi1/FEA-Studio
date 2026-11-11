"""Verification cases for the material nonlinearity – plasticity (B1).

Case PL1:
  Tensile test of a bar 10x10x100 mm with bilinear hardening (BISO).
  E = 200 000 MPa, Re = 250 MPa, Et = 20 000 MPa, nu = 0.3.
  Prescribed displacement u = 0.5 mm -> total strain eps = 0.005.
  Plastic hardening modulus: H = E·Et / (E - Et) = 200 000 · 20 000 / 180 000 = 22 222.22 MPa.
  Increment of the total strain above the yield limit: Delta eps = eps - Re/E = 0.005 - 0.00125 = 0.00375.
  Theoretical stress: sigma = Re + Et * Delta eps = 250 + 20 000 * 0.00375 = 325.0 MPa.
  Tensile reaction: RF = sigma * A = 325 * 100 = 32 500 N.
  1D plastic strain: eps_pl = eps - sigma / E = 0.005 - 325 / 200 000 = 0.003375 mm/mm.
  (The original value 0.00375 in ref_peeq wrongly neglected the elastic increment Delta sigma/E = 0.000375;
   in 3D tet10 with the Poisson contraction the PEEQ is about 0.00345, i.e. a deviation of about 2.2 %).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from fea_core.study import Study, Material, Fixed, Displacement, LoadCase, MeshSettings
from fea_core.selectors import find_faces
from fea_core.geometry import Part, write_step, read_step
from fea_core.plasticity import PlasticitySpec
from verification.cases import Case, Check


def build_pl1(wd: Path) -> tuple[Study, list[Part]]:
    step_file = wd / "bar_10x10x100.step"
    shape = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 10.0, 10.0, 100.0).Shape()
    write_step([Part("Solid1", shape)], step_file)
    parts = read_step(step_file)

    mat = Material(
        name="Steel_Bilinear",
        E=200000.0,
        nu=0.3,
        rho=7.85e-9,
        Re=250.0,
        Rm=400.0,
        plastic=PlasticitySpec(
            model="bilinear",
            hardening="isotropic",
            Et=20000.0,
        ).to_dict(),
    )

    study = Study(name="PL1", geometry=str(step_file), mesh=MeshSettings(size=5.0))
    study.add_material(mat, default=True)
    study.contact_settings.nlgeom = True

    # Constraints via find_faces
    study.add_selection("vetknuti", find_faces(parts, z=0))
    study.add_selection("konec", find_faces(parts, z=100))
    study.constraints.append(Fixed("vetknuti"))
    study.constraints.append(Displacement("konec", uz=0.5))

    study.load_cases.append(LoadCase("Tension", []))
    return study, parts


def evaluate_pl1(res, study) -> list[Check]:
    step = res.steps[0]
    S = step.S  # (N, 6): SXX, SYY, SZZ, SXY, SYZ, SZX
    s_zz = S[:, 2]

    # Mean stress in the middle of the bar (to eliminate the end effect of the clamping by the Poisson contraction)
    mid_nodes = np.where((res.mesh.nodes[:, 2] > 20.0) & (res.mesh.nodes[:, 2] < 80.0))[0]
    avg_szz = float(np.mean(s_zz[mid_nodes]))

    # Tensile reaction at the support Z=0
    rf_z = 0.0
    for name, r in step.reactions.items():
        if "vetknuti" in name:
            rf_z += float(r[2])

    # Plastic strain PEEQ: 1D analytics eps_pl = 0.005 - 325/200000 = 0.003375
    peeq = step.fields.get("peeq", np.zeros(len(res.mesh.nodes)))
    avg_peeq = float(np.mean(peeq[mid_nodes]))

    ref_sigma = 325.0       # MPa
    ref_rf = 32500.0        # N
    ref_peeq = 0.003375     # mm/mm (1D analytic reference limit)

    return [
        Check("PL1: Napětí SZZ v plastické oblasti", avg_szz, ref_sigma, tol=0.03, unit="MPa"),
        Check("PL1: Tahová reakce RF_Z (se znaménkem: podpora tlačí proti tahu)", rf_z, -ref_rf, tol=0.03, unit="N"),
        Check("PL1: Plastické přetvoření PEEQ", avg_peeq, ref_peeq, tol=0.08, unit="mm/mm"),
    ]


CASES: dict[str, Case] = {
    "PL1": Case("PL1", "PL1: Tah tyče s bilineární plasticitou (BISO)", build_pl1, evaluate_pl1),
}


# ================================================================== PL2–PL4: bending, tables and kinematic hardening
# PL2  Cantilever 60×10×10 mm loaded by a pure moment (rigid face, RemoteLoad) with bilinear isotropic hardening
# (E = 200 GPa, Re = 250 MPa, Et = E/100): steps 0.8 / 1.2 / 1.45·M_y and unloading. Analytics: moment–curvature
# of the rectangular section M(κ) with an elastic core and a hardening plastic zone (M_p = σ_y·b·h²/4 = 1.5·M_y is the
# asymptote for Et → 0), the elastic unloading = residual curvature κ_B − M_B/(EI) and residual stress (Hodge).
# PL3  Two tensile bars (σ controlled by the force, homogeneous state): Ramberg–Osgood and Hollomon, point by point the curve
# ε(σ) = σ/E + ε_pl(σ) from the same *PLASTIC table; unloading = residual strain ε_pl(σ_max), stress 0.
# PL4  Two bars, cycle +1.3·Re → 0 → −1.3·Re → 0: kinematic hardening (Bauschinger effect: reverse yielding at
# σ_max − 2·Re) × isotropic (reverse yielding only at −σ_max). Analytics of linear hardening with H = E·Et/(E − Et).
from scipy.optimize import brentq

from fea_core.study import Force, RemoteLoad
from fea_core.contacts.model import LoadStep
from fea_core.plasticity import make_hollomon_table, make_ramberg_osgood_table
from verification.cases import box

PE, PNU = 200000.0, 0.3


def _plastic_mat(name: str, plastic: PlasticitySpec, Re: float = 250.0, Rm: float = 400.0) -> Material:
    return Material(name, PE, PNU, 7.85e-9, Re=Re, Rm=Rm, plastic=plastic.to_dict())


def _plastic_study(name: str, wd: Path, geo: list[Part], size: float):
    step_file = wd / "model.step"
    write_step(geo, step_file)
    parts = read_step(step_file)
    st = Study(name=name, geometry=str(step_file), mesh=MeshSettings(size=size))
    return st, parts


def _seq(st: Study, case: str, scales: list[float], names: list[str], dt0: float = 0.1, dt_max: float = 0.25) -> None:
    an = st.analysis
    an.kind, an.branches = "nonlinear", False
    an.sequence = [LoadStep(n, case=case, scale=s, dt0=dt0, dt_max=dt_max, max_inc=200) for n, s in zip(names, scales)]


# ------------------------------------------------------------------ PL2 bending with load–unload
PL2_L, PL2_B, PL2_H, PL2_RE, PL2_ET = 60.0, 10.0, 10.0, 250.0, 2000.0
PL2_SCALES = [0.8, 1.2, 1.45, 0.0]
PL2_NAMES = ["0,8 My (pružně)", "1,2 My", "1,45 My", "Odlehčení"]


def bil_moment(kappa: float, b: float, h: float, E: float, sy: float, Et: float) -> float:
    """M(κ) of a rectangle b×h, bilinear material (elastic E, above ε_y the tangent modulus Et)."""
    c, ey = h / 2.0, sy / E
    ye = min(ey / kappa, c)
    M = 2 * b * E * kappa * ye ** 3 / 3.0
    if ye < c:
        M += 2 * b * ((sy - Et * ey) * (c ** 2 - ye ** 2) / 2.0 + Et * kappa * (c ** 3 - ye ** 3) / 3.0)
    return M


def bil_curvature(M: float, b: float, h: float, E: float, sy: float, Et: float) -> float:
    ky = sy / E / (h / 2.0)
    if M <= bil_moment(ky, b, h, E, sy, Et):
        return M / (E * b * h ** 3 / 12.0)
    return brentq(lambda k: bil_moment(k, b, h, E, sy, Et) - M, ky, 200 * ky)


def bil_stress(eps, E, sy, Et):
    """Stress and plastic strain from the total strain (loading from zero)."""
    eps = np.asarray(eps, float)
    ey = sy / E
    a = np.abs(eps)
    s = np.where(a <= ey, E * a, sy + Et * (a - ey)) * np.sign(eps)
    return s, np.maximum(a - np.abs(s) / E, 0.0)


def _section_rotation(X: np.ndarray, U: np.ndarray, x0: float, h: float) -> float:
    """Rotation of the section at x ≈ x0: slope of the regression u_x(z) of the nodes within ±1 mm of the section (planar sections)."""
    sel = np.nonzero(np.abs(X[:, 0] - x0) < 1.0)[0]
    return float(np.polyfit(X[sel, 2] - h / 2, U[sel, 0], 1)[0])


def build_pl2(wd: Path):
    st, parts = _plastic_study("PL2 ohyb load-unload", wd, [Part("Nosnik", box(0, 0, 0, PL2_L, PL2_B, PL2_H))], 1.5)
    st.add_material(_plastic_mat("Ocel BISO", PlasticitySpec(model="bilinear", hardening="isotropic", Et=PL2_ET),
                                 PL2_RE), default=True)
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=PL2_L))
    st.constraints.append(Fixed("vetknuti"))
    My = PL2_RE * PL2_B * PL2_H ** 2 / 6.0                       # yield limit in bending
    st.load_cases.append(LoadCase("Moment M_y", [RemoteLoad("konec", [PL2_L, PL2_B / 2, PL2_H / 2], moment=[0.0, My, 0.0],
                                                           coupling="distributing")]))
    _seq(st, "Moment M_y", PL2_SCALES, PL2_NAMES)
    return st, parts


def evaluate_pl2(res, st) -> list[Check]:
    X = res.mesh.nodes
    b, h, E, sy, Et = PL2_B, PL2_H, PE, PL2_RE, PL2_ET
    My = sy * b * h ** 2 / 6.0
    I = b * h ** 3 / 12.0
    mid = np.nonzero(np.abs(X[:, 0] - PL2_L / 2) < 0.8)[0]
    z = X[mid, 2] - h / 2
    surf_m = np.abs(z) > h / 2 - 0.5
    out = []
    kap = {}
    for k, (sc, name) in enumerate(zip(PL2_SCALES, PL2_NAMES)):
        s = res.steps[k]
        M = sc * My
        # the sense of the moment is given (M_y > 0 at the face with the outward normal +x): M_y = ∫ z·σ_xx dA > 0 → tension at z > 0, so
        # the expected σ_xx(z) and the rotations of the sections (u_x grows with z) have a positive sign – it is not taken from the FEM result
        sg = 1.0
        # curvature in the middle: rotation of the sections (regression of u_x(z) in the vicinity of x = 20 and 40 mm) / distance of the sections
        kfe = (_section_rotation(X, s.U, 40.0, h) - _section_rotation(X, s.U, 20.0, h)) / 20.0   # with the sign
        if sc > 0:
            ka = bil_curvature(M, b, h, E, sy, Et)
            kap[k] = ka
            sa, pa = bil_stress(sg * ka * z, E, sy, Et)
            dev = np.sqrt(np.mean((s.S[mid, 0] - sa) ** 2)) / sy
            pe_fe = float(np.max(s.fields["peeq"][mid[surf_m]]))
            pe_an = float(pa[surf_m].max())
            out += [Check(f"PL2 {name}: křivost uprostřed se znaménkem (moment–křivost)", kfe, ka, 0.03, "1/mm"),
                    Check(f"PL2 {name}: σ_xx uprostřed – RMS odchylka od analytiky / Re", dev, 0.0, 0.03, "")]
            if pe_an > 1e-4:
                out.append(Check(f"PL2 {name}: PEEQ povrch", pe_fe, pe_an, 0.08, ""))
            else:
                out.append(Check(f"PL2 {name}: pružně – PEEQ = 0", pe_fe, 0.0, 1e-6, ""))
        else:                                                    # unloading: elastic, residual curvature and stress
            kb = kap[2]
            Mb = PL2_SCALES[2] * My
            kres = kb - Mb / (E * I)
            sB, pB = bil_stress(sg * kb * z, E, sy, Et)
            sres = sB - sg * Mb * z / I
            dev = np.sqrt(np.mean((s.S[mid, 0] - sres) ** 2)) / sy
            out += [Check("PL2 odlehčení: zbytková křivost uprostřed se znaménkem", kfe, kres, 0.05, "1/mm"),
                    Check("PL2 odlehčení: zbytkové σ_xx – RMS odchylka od analytiky / Re", dev, 0.0, 0.03, ""),
                    Check("PL2 odlehčení: PEEQ povrch = z zatěžování", float(np.max(s.fields["peeq"][mid[surf_m]])),
                          float(pB[surf_m].max()), 0.08, ""),
                    Check("PL2 odlehčení: zbytkové σ_xx horní vlákno (Hodge, tlak)",
                          float(np.mean(s.S[mid[surf_m & (z > 0)], 0])), float(sres[surf_m & (z > 0)].mean()), 0.08, "MPa"),
                    Check("PL2 odlehčení: zbytkové σ_xx dolní vlákno (Hodge, tah)",
                          float(np.mean(s.S[mid[surf_m & (z < 0)], 0])), float(sres[surf_m & (z < 0)].mean()), 0.08, "MPa")]
    return out


# ------------------------------------------------------------------ PL3 / PL4: two tensile bars
PB, PH, PLEN = 10.0, 10.0, 100.0
BAR_X = {"A": 0.0, "B": 30.0}


def _bars(name: str, wd: Path, mats: dict[str, Material], scales: list[float], names: list[str], sig_ref: float,
          size: float = 5.0, dt0: float = 0.1, dt_max: float = 0.25):
    """Two prismatic bars 10×10×100 (parts Prut_A, Prut_B) – each a different material; force F = sig_ref·A at the face (force control,
    homogeneous state: displacement only in the axis, lateral expansion free). The sequence of steps = scales × F.
    """
    geo = [Part(f"Prut_{k}", box(x0, 0, 0, PB, PH, PLEN)) for k, x0 in BAR_X.items()]
    st, parts = _plastic_study(name, wd, geo, size)
    first = True
    for k, m in zip(BAR_X, mats.values()):
        st.add_material(m, default=first)
        st.part_materials[f"Prut_{k}"] = m.name
        first = False
    loads = []
    for k, x0 in BAR_X.items():
        st.add_selection(f"dno_{k}", find_faces(parts, part=f"Prut_{k}", z=0))
        st.add_selection(f"konec_{k}", find_faces(parts, part=f"Prut_{k}", z=PLEN))
        st.add_selection(f"x0_{k}", find_faces(parts, part=f"Prut_{k}", x=x0))
        st.add_selection(f"y0_{k}", find_faces(parts, part=f"Prut_{k}", y=0))
        st.constraints += [Displacement(f"dno_{k}", uz=0.0), Displacement(f"x0_{k}", ux=0.0),
                           Displacement(f"y0_{k}", uy=0.0)]
        loads.append(Force(f"konec_{k}", [0.0, 0.0, sig_ref * PB * PH]))
    st.load_cases.append(LoadCase("Tah", loads))
    _seq(st, "Tah", scales, names, dt0, dt_max)
    return st, parts


def _bar_nodes(res, k: str):
    X = res.mesh.nodes
    x0 = BAR_X[k]
    inx = (X[:, 0] >= x0 - 1e-6) & (X[:, 0] <= x0 + PB + 1e-6)
    return np.nonzero(inx & (np.abs(X[:, 2] - PLEN) < 1e-6))[0], np.nonzero(inx)[0]


def _curve_checks(res, tag: str, k: str, table: list[tuple[float, float]], sigmas: list[float], names: list[str],
                  idx0: int) -> list[Check]:
    """Point by point: u_tip/L = σ/E + ε_pl(σ) from the *PLASTIC table, PEEQ = ε_pl, stress = σ."""
    tab = np.array(table)
    out = []
    tip, allb = _bar_nodes(res, k)
    eps_pl_max = 0.0
    for j, (sg, nm) in enumerate(zip(sigmas, names)):
        s = res.steps[idx0 + j]
        ep = float(np.interp(sg, tab[:, 0], tab[:, 1])) if sg > tab[0, 0] else 0.0
        if sg > 0:
            eps_pl_max = max(eps_pl_max, ep)
        e_tot = sg / PE + ep
        if sg == 0:                                              # unloading: residual plastic deformation
            e_tot = ep = eps_pl_max
        u = float(s.U[tip, 2].mean()) / PLEN
        szz = float(np.mean(s.S[allb, 2]))
        pe = float(np.mean(s.fields["peeq"][allb]))
        out.append(Check(f"{tag} {nm}: přetvoření u/L", u, e_tot, 2e-3, ""))
        if sg != 0:
            out.append(Check(f"{tag} {nm}: σ_zz = F/A", szz, sg, 2e-3, "MPa"))
        else:
            out.append(Check(f"{tag} {nm}: zbytkové σ_zz = 0", szz, 0.0, 0.1, "MPa"))
        if ep > 1e-5:
            out.append(Check(f"{tag} {nm}: PEEQ = ε_pl(σ)", pe, ep, 5e-3, ""))
    return out


# ---- PL3
PL3_SIG = [200.0, 270.0, 300.0, 330.0, 0.0]
PL3_NAMES = ["200 MPa (pružně)", "270 MPa", "300 MPa", "330 MPa", "Odlehčení"]
PL3_REF = 330.0
PL3_RO = dict(K=700.0, n=0.2)
PL3_HOL = dict(Rm=400.0, A=0.2)


def pl3_tables():
    ro = make_ramberg_osgood_table(250.0, PL3_RO["K"], PL3_RO["n"], E=PE)
    hol = make_hollomon_table(250.0, PL3_HOL["Rm"], PL3_HOL["A"], PE)
    return ro, hol


def build_pl3(wd: Path):
    mats = {"A": _plastic_mat("Ocel Ramberg-Osgood", PlasticitySpec(model="ramberg_osgood", K_prime=PL3_RO["K"],
                                                                    n_prime=PL3_RO["n"])),
            "B": _plastic_mat("Ocel Hollomon", PlasticitySpec(model="hollomon", Rm=PL3_HOL["Rm"], A=PL3_HOL["A"]))}
    return _bars("PL3 Ramberg-Osgood a Hollomon", wd, mats, [s / PL3_REF for s in PL3_SIG], PL3_NAMES, PL3_REF)


def evaluate_pl3(res, st) -> list[Check]:
    ro, hol = pl3_tables()
    return (_curve_checks(res, "PL3 R-O", "A", ro, PL3_SIG, PL3_NAMES, 0)
            + _curve_checks(res, "PL3 Hollomon", "B", hol, PL3_SIG, PL3_NAMES, 0))


# ---- PL4
PL4_RE, PL4_ET = 250.0, 20000.0
PL4_SIG = [325.0, 0.0, -325.0, 0.0]
PL4_NAMES = ["+325 MPa", "Odlehčení 1", "−325 MPa", "Odlehčení 2"]


def build_pl4(wd: Path):
    mats = {"A": _plastic_mat("Ocel kinematicke", PlasticitySpec(model="bilinear", hardening="kinematic", Et=PL4_ET), PL4_RE),
            "B": _plastic_mat("Ocel izotropni", PlasticitySpec(model="bilinear", hardening="isotropic", Et=PL4_ET), PL4_RE)}
    st, parts = _bars("PL4 kinematicke vs izotropni zpevneni", wd, mats, [s / 325.0 for s in PL4_SIG], PL4_NAMES, 325.0,
                      dt0=0.5, dt_max=0.5)
    st.contact_settings.nlgeom = True            # kinematic hardening in ccx is computed correctly only with NLGEOM (plasticity.check)
    return st, parts


def cyclic_expect(kinematic: bool):
    """(ε, ε_pl, PEEQ) by the steps of PL4 (linear hardening with H = E·Et/(E − Et))."""
    H = PE * PL4_ET / (PE - PL4_ET)
    s1 = PL4_SIG[0]
    ep1 = (s1 - PL4_RE) / H
    out = [(s1 / PE + ep1, ep1, ep1), (ep1, ep1, ep1)]
    if kinematic:
        y_rev = s1 - 2 * PL4_RE                                  # reverse yielding
        d = (y_rev - PL4_SIG[2]) / H
        ep3 = ep1 - d
        out += [(PL4_SIG[2] / PE + ep3, ep3, ep1 + d), (ep3, ep3, ep1 + d)]
    else:                                                        # isotropically: yielding only at −σ_max, i.e. exactly at the target
        out += [(PL4_SIG[2] / PE + ep1, ep1, ep1), (ep1, ep1, ep1)]
    return out


def evaluate_pl4(res, st) -> list[Check]:
    out = []
    for k, kin, tag in (("A", True, "PL4 kinematické"), ("B", False, "PL4 izotropní")):
        tip, allb = _bar_nodes(res, k)
        for j, (e, ep, pe) in enumerate(cyclic_expect(kin)):
            s = res.steps[j]
            u = float(s.U[tip, 2].mean()) / PLEN
            # NLGEOM: force F/A0 and u/L are nominal quantities, the table is in the true stress / logarithmic strain
            # (difference of the order ε·ν ≈ 1.5 %) → tolerance 3 %; the Bauschinger effect differs from isotropy by hundreds of %
            out.append(Check(f"{tag} {PL4_NAMES[j]}: přetvoření u/L", u, e, 5e-2, ""))
            out.append(Check(f"{tag} {PL4_NAMES[j]}: PEEQ", float(np.mean(s.fields['peeq'][allb])), pe, 3e-2, ""))
    return out


CASES["PL2"] = Case("PL2", "PL2: Ohyb konzoly do plastického rozsahu, odlehčení (BISO)", build_pl2, evaluate_pl2)
CASES["PL3"] = Case("PL3", "PL3: Tah – Ramberg-Osgood a Hollomon bod po bodu", build_pl3, evaluate_pl3)
CASES["PL4"] = Case("PL4", "PL4: Cyklus tah–tlak – kinematické × izotropní zpevnění", build_pl4, evaluate_pl4)
