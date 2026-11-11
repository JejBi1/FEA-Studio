"""Fast tests of the bolts and welds (without Gmsh and ccx): names, tables, VDI 2230, EN 1993, nodal forces."""
import math

import numpy as np
import pytest

from fea_core.joints import tables
from fea_core.joints.assess import beta_w, weld_sn_damage, _dominant
from fea_core.joints.forces import element_nodal_forces, mesh_interfaces, read_el_print
from fea_core.joints.model import BoltSpec, WeldSpec, JointSettings
from fea_core.joints.naming import parse_name
from fea_core.joints.vdi2230 import basics, assess, fatigue_damage
from fea_core.mesh import Mesh
from fea_core.study import Study, Material


@pytest.mark.parametrize("name, kind, std, d, L, cls", [
    ("ISO 4017 - M8 x 30 - 8.8", "bolt", "ISO 4017", 8, 30, "8.8"),
    ("ISO 4762 M6x20", "bolt", "ISO 4762", 6, 20, None),
    ("ISO 4032 M8", "nut", "ISO 4032", 8, None, None),
    ("ISO 7089 8", "washer", "ISO 7089", 8, None, None),
    ("ISO 7089 - 8,4 - 200 HV", "washer", "ISO 7089", 8, None, None),
    ("DIN 933 M10x40 10.9", "bolt", "DIN 933", 10, 40, "10.9"),
    ("DIN 912 M12 x 50 - A4-80", "bolt", "DIN 912", 12, 50, "A4-80"),
    ("ISO 4762 M8 x 25 --- 25N", "bolt", "ISO 4762", 8, 25, None),
    ("ISO 4017 - M12 x 40 - 8.8:2", "bolt", "ISO 4017", 12, 40, "8.8"),
    ("ISO 7089 - 12 - 200 HV:1 (2)", "washer", "ISO 7089", 12, None, None),
    ("DIN 125-1 A 8,4", "washer", "DIN 125-1", 8, None, None),           # SOLIDWORKS Toolbox: part of the standard, shape A
    ("DIN 125-1 A 8.4 - 140 HV-1", "washer", "DIN 125-1", 8, None, None),
    ("DIN 125 - B 10,5", "washer", "DIN 125", 10, None, None),
    ("DIN 7968 M16x60 5.6", "fitted", "DIN 7968", 16, 60, "5.6"),
    ("EN 14399-4 M20x80 10.9", "bolt", "EN 14399-4", 20, 80, "10.9"),
    ("Šroub M10x1.25x40 8.8", "bolt", "", 10, 40, "8.8"),
    ("ISO 4762 M6x20 A2", "bolt", "ISO 4762", 6, 20, "A2-70"),
])
def test_parse_name(name, kind, std, d, L, cls):
    fn = parse_name(name)
    assert fn is not None
    assert (fn.kind, fn.standard, fn.d, fn.length, fn.strength) == (kind, std, d, L, cls)


@pytest.mark.parametrize("name", ["Deska A", "Držák M8", "Svar a5", "Plech 10 mm", "Stojina"])
def test_parse_name_not_fastener(name):
    assert parse_name(name) is None


def test_fine_thread_and_socket_head():
    fn = parse_name("ISO 4762 M10x1.25x30 12.9")
    assert fn.pitch == 1.25 and fn.length == 30 and fn.head == "socket"


def test_thread_tables():
    th = tables.thread(10)
    assert th.P == 1.5
    assert th.As == pytest.approx(58.0, abs=0.05)         # ISO 898-1: M10 A_s = 58,0 mm²
    assert tables.thread(8).As == pytest.approx(36.6, abs=0.05)
    assert tables.thread(16).As == pytest.approx(157.0, abs=0.5)      # ISO 898-1 gives it rounded
    assert tables.thread(20).As == pytest.approx(245.0, abs=0.5)
    assert tables.strength_class("8.8", 20).Rp02 == 660
    assert tables.strength_class("10.9").fub == 1000
    assert tables.clearance_class(10, 11.0) == "medium"
    assert tables.clearance_class(10, 10.5) == "fine"
    with pytest.raises(KeyError):
        tables.strength_class("7.7")


def _bolt(**kw):
    b = BoltSpec("B", standard="ISO 4017", size=10.0, length=40.0, strength="8.8", origin=[0, 0, 10],
                 axis=[0, 0, -1], head_part="A", head_dw=20.0, nut_part="B", nut_s=20.0, nut_dw=20.0,
                 plates=[{"part": "A", "s0": 0, "s1": 10, "hole": 11.0}, {"part": "B", "s0": 10, "s1": 20, "hole": 11.0}],
                 washers=[{"part": "w1", "side": "head", "h": 2.0, "d1": 10.5, "d2": 20.0},
                          {"part": "w2", "side": "nut", "h": 2.0, "d1": 10.5, "d2": 20.0}])
    for k, v in kw.items():
        setattr(b, k, v)
    return b


def test_vdi_preload_and_torque():
    b = _bolt()
    bas = basics(b, 210000.0, {"A": 210000.0, "B": 210000.0}, 210000.0, 640.0)
    th = tables.thread(10)
    k = 1.5 / (math.pi * th.d2) + 1.155 * 0.12
    fm = th.As * 0.9 * 640 / math.sqrt(1 + 3 * (1.5 * th.d2 / th.ds * k) ** 2)
    assert bas["F_Mzul"] == pytest.approx(fm)
    assert bas["lK"] == pytest.approx(24.0)
    assert bas["F_Mmin"] == pytest.approx(bas["F_Mmax"] / 1.6)
    # tightening torque M10 8.8 at μ = 0.12 about 50 N·m (common tables 46–51 N·m)
    assert 44 < bas["M_A"] < 55
    assert 0.05 < bas["phi_K"] < 0.4
    # preload given by the torque
    b2 = _bolt(preload_mode="torque", preload=bas["M_A"])
    assert basics(b2, 210000.0, {"A": 210000.0, "B": 210000.0}, 210000.0, 640.0)["F_Mmax"] == pytest.approx(fm)


def test_vdi_compliance_shank():
    """Compliance of the bolt: sum of the sections; a fully threaded ISO 4017 has no smooth shank."""
    b = _bolt()
    bas = basics(b, 210000.0, {}, 210000.0, 640.0)
    th = tables.thread(10)
    exp = (0.5 * 10 / th.AN + 24.0 / th.Ad3 + 0.5 * 10 / th.Ad3 + 0.4 * 10 / th.AN) / 210000.0
    assert bas["delta_S"] == pytest.approx(exp)
    b4014 = _bolt(standard="ISO 4014", length=80.0)
    bas2 = basics(b4014, 210000.0, {}, 210000.0, 640.0)
    assert bas2["l_shank"] == pytest.approx(min(80 - 26, 24.0))


def test_vdi_assess_opening_and_slip():
    b = _bolt()
    bas = basics(b, 210000.0, {}, 210000.0, 640.0)
    ok = assess(b, bas, {"F_A": 2000, "F_SA": 300, "F_PA": 1700, "F_Q": 500, "M_b": 0}, 360, 210000.0)
    assert ok["F_KR"] > 0 and ok["S_G"] > 1.2
    bad = assess(b, bas, {"F_A": 40000, "F_SA": 6000, "F_PA": 34000, "F_Q": 0, "M_b": 0}, 360, 210000.0)
    assert bad["F_KR"] < 0
    assert max(c[1] for c in bad["checks"]) >= 1.0


def test_vdi_fatigue_limit():
    d = 10.0
    sASV = 0.85 * (150 / d + 45)
    assert fatigue_damage([sASV], [2e6], d)["damage"] == pytest.approx(1.0)
    assert fatigue_damage([2 * sASV], [2e6 / 8], d)["damage"] == pytest.approx(1.0)


def test_weld_sn_en1993():
    # Δσ_C at 2·10⁶ cycles = damage 1
    assert weld_sn_damage(np.array([36.0]), np.array([2e6]), 36.0, False, "EN 1993-1-9", 1.0) == pytest.approx(1.0)
    assert weld_sn_damage(np.array([80.0]), np.array([2e6]), 80.0, True, "EN 1993-1-9", 1.0) == pytest.approx(1.0)
    # below the cut-off limit without damage
    assert weld_sn_damage(np.array([10.0]), np.array([1e9]), 36.0, False, "EN 1993-1-9", 1.0) == 0.0
    # γ_Mf increases the damage: (1.15)³
    assert weld_sn_damage(np.array([36.0]), np.array([2e6]), 36.0, False, "EN 1993-1-9", 1.15) == pytest.approx(1.15 ** 3)
    assert weld_sn_damage(np.array([36.0]), np.array([2e6]), 36.0, False, "IIW", 1.0) == pytest.approx(1.0)


def test_beta_w():
    assert beta_w(235) == 0.8 and beta_w(355) == 0.9 and beta_w(460) == 1.0


def test_dominant_projection_proportional():
    c = np.sin(np.linspace(0, 6, 50))
    v = np.array([3.0, 4.0])
    p = _dominant(np.outer(c, v))
    assert np.ptp(np.abs(p)) == pytest.approx(5.0 * np.ptp(np.abs(c)), rel=1e-9)


def _two_tets():
    """Two tet10 tetrahedra (corners + midpoints of the edges) with a common face, different parts."""
    from fea_core.loads import _tet10_dN  # noqa: F401
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1.0]])
    tets_c = [[0, 1, 2, 3], [0, 2, 1, 4]]
    nodes = list(X)
    mids = {}

    def mid(a, b):
        k = tuple(sorted((a, b)))
        if k not in mids:
            mids[k] = len(nodes)
            nodes.append(0.5 * (X[a] + X[b]))
        return mids[k]
    tets = []
    for a, b, c, d in tets_c:      # ccx order: 5 (0-1), 6 (1-2), 7 (2-0), 8 (0-3), 9 (1-3), 10 (2-3)
        tets.append([a, b, c, d, mid(a, b), mid(b, c), mid(c, a), mid(a, d), mid(b, d), mid(c, d)])
    return Mesh(np.array(nodes, float), np.array(tets), np.array([0, 1]), ["A", "B"])


def test_nodal_forces_uniform_stress():
    """Uniform σ_zz: nodal forces = ∫N σ·n dA over the faces (tri6: corners 0, midpoints 1/3), the sum zero."""
    m = _two_tets()
    sig = np.zeros((1, 4, 6))
    sig[:, :, 2] = 100.0
    f = element_nodal_forces(m, np.array([0]), sig)[0]
    assert np.abs(f.sum(axis=0)).max() < 1e-9
    # wall z = 0 (σ·n·A = −50 N): midpoints 0-1, 2-0 → −50/3; inclined wall (+50 N): midpoints 1-3, 2-3 → +50/3;
    # midpoint 1-2 lies on both → 0; corners 0
    assert f[[4, 6], 2] == pytest.approx([-50 / 3, -50 / 3])
    assert f[[8, 9], 2] == pytest.approx([50 / 3, 50 / 3])
    assert np.abs(f[[0, 1, 2, 3, 5, 7], 2]).max() < 1e-9


def test_mesh_interfaces_and_alias():
    m = _two_tets()
    itf = mesh_interfaces(m)
    assert list(itf) == [(0, 1)]
    assert len(itf[(0, 1)]["tri"]) == 1
    # a disconnected contact: part B gets its own nodes, the wall is found through node_alias
    shared = np.unique(itf[(0, 1)]["tri"])
    n0 = len(m.nodes)
    remap = {int(k): n0 + i for i, k in enumerate(shared)}
    m.nodes = np.vstack([m.nodes, m.nodes[shared]])
    m.node_alias = np.concatenate([np.arange(n0), shared])
    m.tets[1] = [remap.get(int(k), int(k)) for k in m.tets[1]]
    itf2 = mesh_interfaces(m)
    assert list(itf2) == [(0, 1)]
    tri, _ = m.boundary_faces()
    assert len(tri) == 6                     # an internal wall is not counted as a surface


def test_read_el_print(tmp_path):
    dat = tmp_path / "job.dat"
    dat.write_text(
        "\n stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EJOINT and time  0.1000000E+01\n\n"
        "         1   1  1.0E+00  2.0E+00  3.0E+00  4.0E+00  5.0E+00  6.0E+00\n"
        "         1   2  1.0E+00  2.0E+00  3.0E+00  4.0E+00  5.0E+00  6.0E+00\n\n"
        " stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EJOINT and time  0.1000000E+01\n\n"
        "         1   1  7.0E+00  0.0E+00  0.0E+00  0.0E+00  0.0E+00  0.0E+00\n\n")
    steps = read_el_print(dat, "EJOINT")
    assert len(steps) == 2
    # order of the .dat xy xz yz → Results xy yz zx
    assert steps[0][1][0].tolist() == [1, 2, 3, 4, 6, 5]
    assert steps[1][1][0, 0] == 7


def test_detect_countersunk_with_gap(tmp_path):
    """Countersunk head (cone) and clearance between the plates: origin on the top face of the head, bundle without the preload state."""
    from fea_core.geometry import read_step, write_step
    from fea_core.joints.detect import detect
    from fea_core.joints.fem import has_gap, coupling_keys
    from verification.joint_models import countersunk_joint
    write_step(countersunk_joint(), tmp_path / "cs.step")
    rep = detect(read_step(tmp_path / "cs.step"))
    assert len(rep.bolts) == 1, rep.warnings
    b = rep.bolts[0]
    assert b.head == "countersunk" and b.head_cone == pytest.approx(2.5, abs=0.05)
    assert b.origin[2] == pytest.approx(6.0, abs=1e-6) and b.axis[2] == pytest.approx(-1.0)
    assert [p["part"] for p in b.plates] == ["Deska A", "Deska B"]
    assert b.plates[0]["s0"] == 0.0 and b.nut_s == pytest.approx(12.3, abs=1e-6)
    assert has_gap(b)
    assert coupling_keys(b, 1)[0] == ("__b1_h", pytest.approx(1.25, abs=0.05))


def test_read_step_generic_names(tmp_path):
    """A "SOLID" body in a component gets the name of the component (Fusion), not "SOLID"."""
    from fea_core.geometry import GENERIC_NAMES
    assert "SOLID" in GENERIC_NAMES and "BODY" in GENERIC_NAMES


def test_auto_preload_limited_by_plastic():
    """Automatic preload in PLA: 90 % of p_G·A_p, not the F_M,zul of the bolt."""
    b = _bolt()
    pG = 60.0
    bas = basics(b, 210000.0, {"A": 3500.0, "B": 3500.0}, 210000.0, 640.0, None, pG)
    A_p = math.pi / 4 * (20.0 ** 2 - 11.0 ** 2)
    assert bas["F_Mmax"] == pytest.approx(min(bas["F_Mzul"], 0.9 * pG * A_p))
    assert "pressure" in bas["preload_limit"]


def test_study_json_roundtrip_joints():
    st = Study("t", "x.step")
    st.add_material(Material("Ocel", 210000, 0.3), default=True)
    st.bolts.append(_bolt(cad_parts=["šroub", "matice"]))
    st.welds.append(WeldSpec("Svar 1", "Svar a5", a=5.0, fat_sigma=40.0))
    st.joints = JointSettings(category="B", slip_mu=0.4)
    st2 = Study.from_dict(st.to_dict())
    assert st2.bolts[0].to_dict() == st.bolts[0].to_dict()
    assert st2.welds[0].fat_sigma == 40.0 and st2.joints.category == "B"
    assert st2.excluded_parts() == {"šroub", "matice"}
    assert st2.bolts[0].grip == pytest.approx(24.0)
    # an old file without joints
    d = st.to_dict()
    for k in ("bolts", "welds", "joints"):
        d.pop(k)
    assert Study.from_dict(d).bolts == []


# ===================================================================== F3 completion: thread stripping, shape, colours

def test_thread_stripping_nut_hand():
    """M10 8.8 + nut ISO 4032 of class 8 (hand calculation according to VDI 2230 / Alexander, nominal dimensions):
    P = 1.5; d2 = D2 = 10 − 0.649519·1.5 = 9.025722; D1 = 10 − 1.082532·1.5 = 8.376202.
    A_SGM/m = π·10·[0.75 + (10 − 9.025722)·tan30°]/1.5 = π·10·0.875 = 27.48894 mm²/mm,
    A_SGS/m = π·8.376202·[0.75 + (9.025722 − 8.376202)·tan30°]/1.5 = π·8.376202·0.75 = 19.73596 mm²/mm.
    τ_BS = τ_BM = 0.65·800 = 520 MPa → R_S = 27.48894/19.73596 = 1.392835 (> 1 → C3 = 0.897),
    C2 = 5.594 − 13.682 R_S + 14.107 R_S² − 6.057 R_S³ + 0.9353 R_S⁴ = 1.05819, s/d = 1.6 → C1 = 0.91.
    m_eff = 8.4 − 0.8·1.5 = 7.2 mm: F_mGM = 520·27.48894·7.2·0.91·0.897 = 84 009 N,
    F_mGS = 520·19.73596·7.2·0.91·1.05819 = 71 154 N (the thread of the bolt decides);
    F_mS = 800·57.99 = 46 393 N → m_eff,min = 46 392/(71 154/7.2) = 4.694 mm, m_ges,min = 5.894 mm < m = 8.4.
    """
    from fea_core.joints.vdi2230 import stripping
    th = tables.thread(10)
    r = stripping(10.0, 1.5, 7.2, 800.0, 0.65, 800.0, 0.65, 16.0, th.As)
    assert r["A_SGM"] / 7.2 == pytest.approx(math.pi * 10 * 0.875, rel=1e-6)
    assert r["A_SGS"] / 7.2 == pytest.approx(math.pi * 8.376202 * 0.75, rel=1e-6)
    assert r["R_S"] == pytest.approx(1.392835, rel=1e-6)
    assert r["C1"] == pytest.approx(0.91) and r["C3"] == pytest.approx(0.897)
    assert r["C2"] == pytest.approx(1.05819, abs=1e-5)
    assert r["F_mGM"] == pytest.approx(84009, rel=1e-5)
    assert r["F_mGS"] == pytest.approx(71154, rel=1e-5)
    assert r["governs"] == "bolt thread"
    assert r["m_ges_min"] == pytest.approx(5.894, abs=1e-3)


def test_thread_stripping_tapped_aluminium():
    """M8 8.8 into aluminium R_m = 260 MPa (C1 = 1): A_SGM/m = 0.875·π·8 = 21.99115, A_SGS/m = 0.75·π·6.646835 = 15.66142;
    τ_BM = 0.65·260 = 169, τ_BS = 520 → R_S = 0.45636, C3 = 0.728 + 1.769 R_S − 2.896 R_S² + 1.296 R_S³ = 1.05535;
    F_mGM/m = 169·21.99115·1.05535 = 3922.3 N/mm < F_mGS/m = 520·15.66142·0.897 = 7305.2 N/mm;
    F_mS = 800·36.61 = 29 289 N → m_eff,min = 7.467 mm, m_ges,min = 7.467 + 0.8·1.25 = 8.467 mm (≈ 1.06 d).
    """
    from fea_core.joints.vdi2230 import stripping
    th = tables.thread(8)
    r = stripping(8.0, 1.25, 10.0, 800.0, 0.65, 260.0, 0.65, None, th.As)
    assert r["R_S"] == pytest.approx(0.45636, rel=2e-5)
    assert r["C1"] == 1.0 and r["C3"] == pytest.approx(1.05535, abs=2e-5)
    assert r["F_mGM"] / 10 == pytest.approx(3922.3, rel=2e-4)
    assert r["governs"] == "internal thread"
    assert r["m_ges_min"] == pytest.approx(8.467, abs=3e-3)


def test_thread_stripping_rows_pla():
    """Bolt into PLA: the thread length row is informative only, a warning about the plastic."""
    from fea_core.joints.assess import thread_strip_rows
    st = Study("t", "x.step")
    st.add_material(Material("PLA", 3500.0, 0.36, 1.24e-9, Rm=50.0), default=True)
    b = _bolt(connection="tapped", nut_part="B", nut_s=10.0, thread_end=18.0, size=3.0, washers=[])
    rows, warns = [], []
    r = thread_strip_rows(st, b, 1000.0, rows, warns)
    assert r is not None and len(rows) == 2
    assert rows[1][0].startswith("(informative)") and rows[1][1] > 1.0      # PLA: the bolt does not break first
    assert rows[0][1] == pytest.approx(1000.0 / min(r["F_mGM"], r["F_mGS"]))
    assert any("plastic" in w for w in warns)


def _anon(parts):
    from fea_core.geometry import Part
    return [Part(f"Body{i}", p.shape) for i, p in enumerate(parts)]


def test_shape_detection_lap_joint():
    """Without names: bolt, nut and washers by shape → only a proposal (source "shape"), the same joint as by the names."""
    from fea_core.joints.detect import detect
    from verification.joint_models import lap_joint
    parts = lap_joint()
    ref = detect(parts).bolts[0]
    rep = detect(_anon(parts))
    assert rep.bolts == [] and len(rep.shape_bolts) == 1
    s = rep.suggestions[0]
    b = s.spec
    assert s.source == "shape" and s.accept and s.confidence >= 0.85
    assert (b.size, b.head, b.connection, len(b.washers)) == (10.0, "hex", "nut", 2)
    assert b.grip == pytest.approx(ref.grip) and np.allclose(b.origin, ref.origin)
    assert [p["part"] for p in b.plates] == ["Body0", "Body1"]
    assert "8.8" in b.note


def test_shape_detection_rejects_pins_and_shafts():
    """A pin (a cylinder without a head), a stepped pin with a collar in the middle and a bushing are not fastening parts."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from fea_core.geometry import Part
    from fea_core.joints.shape import analyze
    from verification.cases import cyl, solid
    pin = Part("a", solid(cyl((0, 0, 0), (0, 0, 1), 4.0, 30.0)))
    collar = Part("b", solid(BRepAlgoAPI_Fuse(cyl((0, 0, 0), (0, 0, 1), 5.0, 40.0),
                                              cyl((0, 0, 15), (0, 0, 1), 8.0, 6.0)).Shape()))
    sleeve = Part("c", solid(BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), 8.0, 20.0),
                                             cyl((0, 0, -1), (0, 0, 1), 5.0, 22.0)).Shape()))
    assert analyze(pin) is None and analyze(collar) is None and analyze(sleeve) is None


def test_shape_detection_socket_countersunk_nut():
    from fea_core.joints.shape import analyze
    from verification.joint_models import countersunk_joint, tapped_joint
    g = analyze(tapped_joint()[2])
    assert (g.kind, g.d, g.head, g.length) == ("bolt", 10.0, "socket", 30.0)
    cs = countersunk_joint()
    g = analyze(cs[2])
    assert (g.kind, g.d, g.head) == ("bolt", 5.0, "countersunk")
    g = analyze(cs[3])
    assert (g.kind, g.d) == ("nut", 5.0)


def test_step_colors_layers_attributes_and_weld_rule(tmp_path):
    """XCAF: colour, layer and user attribute are written and read; a weld by colour / layer / attribute."""
    from fea_core.geometry import Part, read_step, write_step
    from fea_core.joints.detect import detect, weld_rule
    from verification.joint_models import t_joint_fillet
    ps = t_joint_fillet()
    ps[0] = Part("Pasnice", ps[0].shape, (0.2, 0.3, 0.4))
    ps[2] = Part("Body3", ps[2].shape, (1.0, 0.5, 0.0))
    ps[3] = Part("Body4", ps[3].shape, None, ["SVARY"], ["weld"])
    write_step(ps, tmp_path / "tj.step")
    rp = {p.name: p for p in read_step(tmp_path / "tj.step")}
    assert rp["Body3"].color == pytest.approx((1.0, 0.5, 0.0), abs=1e-3)
    assert rp["Pasnice"].color == pytest.approx((0.2, 0.3, 0.4), abs=1e-3)
    assert rp["Body4"].layers == ["SVARY"] and rp["Body4"].attributes == ["weld"]
    js = JointSettings(weld_colors=["#ff8000"])
    assert weld_rule(rp["Body3"], js)[0] == "color"
    assert weld_rule(rp["Body3"]) is None                      # without a set colour
    assert weld_rule(rp["Body4"])[0] == "layer"
    assert weld_rule(rp["Pasnice"], js) is None
    rep = detect(list(rp.values()), settings=js)
    assert sorted(w.part for w in rep.welds) == ["Body3", "Body4"]
    assert {s.source for s in rep.suggestions} == {"color", "layer"}


def test_intermittent_weld_groups():
    """Intermittent weld: segments between the same parts on one line → a group with gaps; the opposite weld separately."""
    from types import SimpleNamespace
    from fea_core.joints.assess import intermittent_groups, _intermittent_rows
    from fea_core.joints.model import WeldSpec
    from fea_core.study import Study

    def info(x, y0, y1):
        n = 4
        L = (y1 - y0) / n
        seg = [{"s0": k * L, "s1": (k + 1) * L, "e": [0, 1, 0], "x": [x, y0 + (k + 0.5) * L, 0.0],
                "toe": [{"t": 8.0, "ok": True}]} for k in range(n)]
        return {"kind": "fillet", "a": 4.0, "part_a": "A", "part_b": "B", "segments": seg}
    st = Study("t", "x.step")
    st.welds = [WeldSpec(name=f"W{i}", part=f"S{i}") for i in range(4)]
    res = SimpleNamespace(info={"joints": {"welds": {"W0": info(5, 0, 25), "W1": info(5, 40, 65),
                                                    "W2": info(5, 80, 100), "W3": info(-5, 0, 100)}}})
    g = intermittent_groups(st, res)
    assert len(g) == 1 and g[0]["welds"] == ["W0", "W1", "W2"]
    assert g[0]["gaps"] == pytest.approx([15.0, 15.0]) and g[0]["t_min"] == 8.0
    rows, warns = _intermittent_rows(g[0])
    assert rows[0][1] == pytest.approx(15.0 / 96.0)            # min(12t; 200) = 96 mm
    assert rows[0][0].startswith("(informative)")


def test_weld_filler_table():
    from fea_core.joints.tables import filler, FILLERS
    assert filler("") is None
    assert filler("ER70S-6")[1] == 480.0
    assert filler("G 42")[0] == 420.0
    assert filler("316LSi")[1] == 510.0
    assert filler("neexistuje") is None
    assert all(re_ < rm for re_, rm, _ in FILLERS.values())
