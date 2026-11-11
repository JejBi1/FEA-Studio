"""3D printing (F4): failure criteria, homogenisation, calibration, storage. Fast (without Gmsh/ccx)."""
import math

import numpy as np
import pytest

from fea_core import materials as matlib
from fea_core.printing import PrintSpec, PrintSettings
from fea_core.printing.calibrate import TensileTest, calibrate, factors, iso527_modulus, section
from fea_core.printing.failure import inv3, safety_tsai_hill, safety_tsai_wu, element_strains
from fea_core.printing.infill import PATTERNS, core, solid, rotation, strain_to_local, ratios
from fea_core.study import Study

PLA = matlib.by_name("Tisk PLA Bambu Basic")


def _s(i, v):
    s = np.zeros(6)
    s[i] = v
    return s


@pytest.mark.parametrize("crit", [safety_tsai_wu, safety_tsai_hill])
def test_uniaxial_strengths(crit):
    m = solid(PLA)
    for i, ref in ((0, m.Xt), (1, m.Xt), (2, m.Zt)):
        assert crit(_s(i, 10.0), m) == pytest.approx(ref / 10.0, rel=1e-12)
    for i, ref in ((0, m.Xc), (2, m.Zc)):
        assert crit(_s(i, -10.0), m) == pytest.approx(ref / 10.0, rel=1e-12)
    assert crit(_s(3, 5.0), m) == pytest.approx(m.Sxy / 5.0, rel=1e-12)
    assert crit(_s(4, 5.0), m) == pytest.approx(m.Sz / 5.0, rel=1e-12)
    assert crit(_s(5, -5.0), m) == pytest.approx(m.Sz / 5.0, rel=1e-12)


def test_tsai_hill_off_axis():
    """Tension at the angle θ in the plane 1–3: 1/σ² = c⁴/X² − c²s²/Z² + s⁴/Z² + c²s²/S² (Hill, X1 = X2 = X)."""
    m = solid(PLA)
    th = math.radians(30)
    c, s = math.cos(th), math.sin(th)
    sig = np.array([c * c, 0, s * s, 0, c * s, 0]) * 10.0
    inv = c ** 4 / m.Xt ** 2 - c * c * s * s / m.Zt ** 2 + s ** 4 / m.Zt ** 2 + c * c * s * s / m.Sz ** 2
    assert safety_tsai_hill(sig, m) * 10.0 == pytest.approx(1 / math.sqrt(inv), rel=1e-12)


def test_stiffness_positive_and_consistent():
    for key in PATTERNS:
        for rho in (0.05, 0.15, 0.4, 1.0):
            C = core(PLA, key, rho).stiffness()
            assert np.allclose(C, C.T)
            assert np.linalg.eigvalsh(C).min() > 0, (key, rho)
    C = solid(PLA).stiffness()
    S = np.linalg.inv(C)
    assert 1 / S[0, 0] == pytest.approx(PLA.E)
    assert 1 / S[2, 2] == pytest.approx(PLA.E_z)
    assert ratios("grid", 1.0) == {"E_xy": 1.0, "E_z": 1.0, "X_xy": 1.0, "X_z": 1.0}


def test_rotation_and_strain_transform():
    R = rotation([1, 0, 0])
    assert np.allclose(R @ R.T, np.eye(3)) and np.allclose(R[2], [1, 0, 0])
    eps = np.array([1e-3, 0, 0, 0, 0, 0])            # ε_xx globally = ε_33 in the print axes
    assert np.allclose(strain_to_local(eps, R), [0, 0, 1e-3, 0, 0, 0])
    eps = np.array([0, 0, 0, 2e-3, 0, 0])            # γ_xy
    loc = strain_to_local(eps, rotation([0, 0, 1]))
    assert np.allclose(loc, [0, 0, 0, 2e-3, 0, 0])


def test_inv3_and_strains():
    J = np.random.default_rng(1).normal(size=(50, 3, 3)) + 3 * np.eye(3)
    assert np.allclose(inv3(J), np.linalg.inv(J))
    # exact linear displacement u = A·x → constant strain in tet10
    from fea_core.results import tet10_shape  # noqa: F401
    x = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 1.5]], float)
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    nodes = np.vstack([x] + [0.5 * (x[a] + x[b]) for a, b in mids])
    A = np.array([[1e-3, 2e-4, 0], [0, -3e-4, 5e-4], [1e-4, 0, 2e-3]])
    eps = element_strains(nodes, np.arange(10)[None], nodes @ A.T)
    ref = [A[0, 0], A[1, 1], A[2, 2], A[0, 1] + A[1, 0], A[0, 2] + A[2, 0], A[1, 2] + A[2, 1]]
    assert np.allclose(eps[0], ref)


def test_calibration_roundtrip():
    """Synthetic bodies made of the "true" material → the calibration returns the true full material."""
    true_E, true_Ez, true_R, true_Rz = 3100.0, 2300.0, 48.0, 26.0
    spec = PrintSpec("x", walls=2, infill=0.15, pattern="grid")
    tests = []
    from fea_core.printing.props import default_props
    pr = default_props(PLA.name)                      # anisotropy of the line of the perimeters (along the tension of the body xy)
    for d, E, R in (("xy", true_E, true_R), ("z", true_Ez, true_Rz)):
        t = TensileTest.from_spec(f"{d}", d, spec)
        fe, fr = factors(t, pr.line_E_ratio, pr.line_R_ratio)
        t.E, t.Rm = E * fe, R * fr
        tests.append(t)
    cal = calibrate(PLA, tests, "PLA kalibrace")
    m = cal.material
    assert (m.E, m.E_z, m.Rm, m.Rm_z) == pytest.approx((true_E, true_Ez, true_R, true_Rz))
    assert m.is_print and m.name == "PLA kalibrace"
    # 100 % infill: measured = full
    t = TensileTest("plny", "xy", E=2900, Rm=40, infill=1.0)
    assert factors(t) == pytest.approx((1.0, 1.0))
    As, Ac = section(TensileTest("z", "z", width=10, thickness=4, walls=2, line_width=0.5))
    assert Ac == pytest.approx((10 - 2) * (4 - 2)) and As + Ac == pytest.approx(40)
    # body xy of full extrusion with only perimeters along the tension: measured E = E_L = 2/(1 + r)·E_xy
    t = TensileTest("stena", "xy", width=1.68, thickness=4, walls=2, line_width=0.42, infill=1.0)
    assert factors(t, 0.8, 0.55) == pytest.approx((2 / 1.8, 2 / 1.55))


def test_iso527_modulus():
    e = np.linspace(0, 0.02, 400)
    s = 3000 * e - 2e4 * e ** 2
    assert iso527_modulus(e, s) == pytest.approx(3000 - 2e4 * (0.0005 + 0.0025), rel=1e-6)


def test_json_roundtrip():
    st = Study("t", "x.step")
    st.add_material(PLA, default=True)
    st.printing = PrintSettings({"A": PrintSpec("A", build_dir=[0, 1, 0], infill=0.3, pattern="gyroid")}, "tsai_hill")
    st2 = Study.from_dict(st.to_dict())
    assert st2.printing.parts["A"].pattern == "gyroid" and st2.printing.criterion == "tsai_hill"
    assert st2.materials[PLA.name].E_z == PLA.E_z and st2.materials[PLA.name].is_print
    assert Study.from_dict({k: v for k, v in st.to_dict().items() if k != "printing"}).printing.parts == {}


# ------------------------------------------------------------------ F4 completion: line, creep, fatigue
def test_line_solid_and_criteria_axis2():
    from fea_core.printing.infill import line_solid
    m = line_solid(PLA, 0.8, 0.55)
    assert 0.5 * (m.E1 + m.E2v) == pytest.approx(PLA.E) and m.E2v / m.E1 == pytest.approx(0.8)
    assert 0.5 * (m.Xt + m.Ytv) == pytest.approx(PLA.Rm) and m.Ytv / m.Xt == pytest.approx(0.55)
    S = np.linalg.inv(m.stiffness())
    assert 1 / S[0, 0] == pytest.approx(m.E1) and 1 / S[1, 1] == pytest.approx(m.E2v)
    assert 1 / S[2, 2] == pytest.approx(PLA.E_z) and -S[0, 1] * m.E1 == pytest.approx(PLA.nu)
    assert np.linalg.eigvalsh(m.stiffness()).min() > 0
    for crit in (safety_tsai_wu, safety_tsai_hill):
        assert crit(_s(1, 10.0), m) == pytest.approx(m.Ytv / 10.0, rel=1e-12)
        assert crit(_s(0, 10.0), m) == pytest.approx(m.Xt / 10.0, rel=1e-12)
        assert crit(_s(1, -10.0), m) == pytest.approx(m.Xc / 10.0, rel=1e-12)
    assert line_solid(PLA, 1.0, 1.0) == solid(PLA)


def test_strain_matrix_and_core_invariance():
    from fea_core.printing import line_axes
    from fea_core.printing.infill import strain_matrix
    rng = np.random.default_rng(3)
    R = rotation([0.3, -0.2, 1.0])
    T = strain_matrix(R)
    C = solid(PLA).stiffness()
    e = rng.normal(size=6)
    assert e @ T.T @ C @ T @ e == pytest.approx(strain_to_local(e, R) @ C @ strain_to_local(e, R))
    # a core transversely isotropic about the build axis (gyroid) has the same C in the axes of the line as in the print axes
    Cc = core(PLA, "gyroid", 0.2).stiffness()
    Rl = line_axes(R, 0.7)
    assert np.allclose(strain_matrix(Rl).T @ Cc @ strain_matrix(Rl), T.T @ Cc @ T, rtol=1e-10, atol=1e-8)
    assert np.allclose(Rl @ Rl.T, np.eye(3)) and np.linalg.det(Rl) == pytest.approx(1.0)
    # a grid (lines ±45°) is anisotropic in the plane → it is transformed to the axes of the line Tᵀ·C·T (printing.material_groups)
    Cg = core(PLA, "grid", 0.2).stiffness()
    Tr = strain_matrix(R @ Rl.T)
    assert np.allclose(strain_matrix(Rl).T @ (Tr.T @ Cg @ Tr) @ strain_matrix(Rl), T.T @ Cg @ T, rtol=1e-10,
                       atol=1e-8)
    assert not np.allclose(strain_matrix(Rl).T @ Cg @ strain_matrix(Rl), T.T @ Cg @ T, rtol=1e-6)


def test_wall_lines():
    from fea_core.printing.region import wall_lines
    sp = PrintSpec("x")
    n = np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0], [0, -0.6, 0.8]])
    L = wall_lines(n, sp)
    assert np.allclose(np.abs(L[0]), [1, 0, 0]) and np.allclose(L[1], 0) and np.allclose(np.abs(L[2]), [0, 1, 0])
    # inclined face: wall 0.84·0.6 = 0.50 < layers 5·0.2·0.8 = 0.8 → full layers, without a line
    assert np.allclose(L[3], 0)


def test_props_creep_factors():
    from fea_core.printing.props import default_props, family_of
    assert family_of("Tisk PETG (orientačně)") == "PETG" and family_of("Tisk PLA Bambu Basic") == "PLA"
    p = default_props("Tisk PLA Bambu Basic")
    f = p.creep_factors(1000.0, 23.0)
    assert f["phi"] == pytest.approx(p.creep_a * 1000 ** p.creep_n) and f["valid"]
    assert f["stiffness"] == pytest.approx(1 / (1 + f["phi"]))
    assert f["strength"] == pytest.approx(1 - p.rupture_beta * math.log10(1001))
    f35 = p.creep_factors(1000.0, 35.0)                        # between 30 and 40 °C
    assert f35["phi"] == pytest.approx(f["phi"] * 2.25) and f35["k_E"] == pytest.approx(0.945)
    assert not p.creep_factors(1000.0, 50.0)["valid"]
    assert p.creep_factors(0.0, 23.0)["stiffness"] == pytest.approx(1.0)


def test_creep_study_and_json():
    from fea_core.printing.creep import creep_study, check
    from fea_core.study import LoadCase
    st = Study("t", "x.step")
    st.add_material(PLA, default=True)
    st.load_cases = [LoadCase("A"), LoadCase("B")]
    assert creep_study(st) is None
    st.printing.creep.enabled = True
    st.printing.creep.cases = ["B"]
    st.printing.creep.temperature = 40.0
    st.printing.materials[PLA.name] = {"line_R_ratio": 0.5}
    st2 = Study.from_dict(st.to_dict())
    assert st2.printing.creep.cases == ["B"] and st2.printing.materials[PLA.name]["line_R_ratio"] == 0.5
    sc, fac = creep_study(st2)
    k = fac[PLA.name]
    assert [lc.name for lc in sc.load_cases] == ["B"] and not sc.printing.creep.enabled
    m = sc.materials[PLA.name]
    assert m.E == pytest.approx(PLA.E * k["stiffness"]) and m.E_z == pytest.approx(PLA.E_z * k["stiffness"])
    assert m.Rm == pytest.approx(PLA.Rm * k["strength"]) and m.nu == PLA.nu
    assert st.materials[PLA.name].E == PLA.E                    # the original unchanged
    st.printing.creep.temperature = 60.0
    errs, _ = check(st, ["A"])
    assert errs and "outside the validity" in errs[0]


def test_print_fatigue_kernels():
    from fea_core.printing.fatigue import _points, _points_prop, _hill_eq, fatigue_strengths
    m = solid(PLA)
    SD = 3.9
    c, inv = fatigue_strengths(m, SD / m.Xt, SD / m.Xt, SD)
    assert _hill_eq(_s(0, 7.0), c, inv) == pytest.approx(7.0)
    assert _hill_eq(_s(2, -7.0), c, inv) == pytest.approx(-7.0 * m.Xt / m.Zt)
    rng = np.random.default_rng(5)
    P = 7
    S = rng.normal(size=(1, P, 6)) * 5
    f = np.sin(np.linspace(0, 6 * np.pi, 61)) * (1 + 0.3 * np.cos(np.linspace(0, 2, 61)))
    C = np.ascontiguousarray(np.broadcast_to(c, (P, 6)))
    I = np.ascontiguousarray(np.broadcast_to(inv, (P, 3)))
    o1 = [np.zeros(P) for _ in range(3)]
    o2 = [np.zeros(P) for _ in range(3)]
    args = (SD, 2e6, 5.5, 10.0, 39.0, 1, 1e5, True)
    _points(np.ascontiguousarray(f[:, None]), S, C, I, *args, *o1)
    _points_prop(f, np.ascontiguousarray(S[0]), C, I, *args, *o2)
    for a, b in zip(o1, o2):
        assert np.allclose(a, b, rtol=1e-9)


def _tet10_unit():
    x = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 1.5]], float)
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    return np.vstack([x] + [0.5 * (x[a] + x[b]) for a, b in mids])


def test_free_thermal_expansion_is_not_stress():
    """Free expansion u = α·ΔT·x → mechanical strain 0 (previously the total ε was assessed as stress)."""
    from fea_core.printing.failure import subtract_thermal
    nodes = _tet10_unit()
    conn = np.arange(10)[None]
    alpha, dT = 68e-6, 30.0
    U = alpha * dT * nodes
    eps = element_strains(nodes, conn, U)
    assert np.allclose(eps[0, :, :3], alpha * dT)
    T = np.full(10, 20.0 + dT)
    mech = subtract_thermal(eps, conn, T, 20.0, alpha)
    assert np.allclose(mech, 0.0, atol=1e-15)
    # linear temperature field: tet10 interpolates it exactly
    T = 20.0 + 10.0 * nodes[:, 0]
    U2 = np.zeros_like(nodes)
    m2 = subtract_thermal(element_strains(nodes, conn, U2), conn, T, 20.0, alpha)
    from fea_core.printing.failure import GAUSS4
    from fea_core.results import tet10_shape
    xg = tet10_shape(GAUSS4) @ nodes
    assert np.allclose(m2[0, :, 0], -alpha * 10.0 * xg[:, 0])
    # several states at once = the same as one by one
    multi = element_strains(nodes, conn, [U, U2])
    assert np.allclose(multi[0], eps) and np.allclose(multi[1], 0.0)


def test_hollow_infill_and_settings_check():
    from fea_core.printing.check import check_settings
    from fea_core.printing.failure import safety
    pla = matlib.by_name("Tisk PLA Bambu Basic")
    c = core(pla, "gyroid", 0.0)                 # previously ZeroDivisionError
    assert c.E1 < 1e-3 * pla.E and np.all(np.linalg.eigvalsh(c.stiffness()) > 0)
    s = np.array([[10.0, -3.0, 2.0, 1.0, 0.5, 0.2]])
    for crit in ("tsai_wu", "tsai_hill"):
        assert np.isinf(safety(s, c, crit)).all()
    st = Study("t", "x.step")
    st.add_material(pla, default=True)
    st.printing.parts["A"] = PrintSpec("A", infill=0.0)
    e, w = check_settings(st, ["A"])
    assert not e and any("hollow" in x for x in w)
    st.printing.bins = 0
    st.printing.parts["A"] = PrintSpec("A", walls=-1)
    e, _ = check_settings(st, ["A"])
    assert any("bins" in x for x in e) and any("perimeters" in x for x in e)
    from dataclasses import replace
    bad = replace(pla, name="Tisk bez Rm", Rm=None)
    st2 = Study("t", "x.step")
    st2.add_material(bad, default=True)
    e, _ = check_settings(st2, ["A"])
    assert any("has no strength" in x for x in e)
    weird = replace(pla, name="Tisk PLA vadný", E=50.0, nu_z=0.45)
    st3 = Study("t", "x.step")
    st3.add_material(weird, default=True)
    e, _ = check_settings(st3, ["A"])
    assert any("not positive definite" in x for x in e)


def test_check_printing_warns_plastic_without_print_model():
    """A part with a plastic "printing" material without E_z (e.g. "PLA tisk XY") → a warning that the print results will not be there;
    a printed material with E_z nor the L-PBF metals (isotropic) get no warning.
    """
    from fea_core.solve import check_printing
    msg = "without print properties"

    def warnings(mat_name):
        st = Study("t", "x.step")
        st.add_material(matlib.by_name(mat_name), default=True)
        return check_printing(st, ["A"])[1]
    assert any(msg in w and "'A'" in w for w in warnings("PLA tisk XY (orientačně)"))
    assert not any(msg in w for w in warnings("Tisk PLA Bambu Basic"))
    lpbf = [m.name for m in matlib.LIBRARY if "L-PBF" in m.name]
    assert lpbf
    for name in lpbf:
        assert not any(msg in w for w in warnings(name)), name
