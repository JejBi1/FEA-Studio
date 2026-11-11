"""Section library (fea_core.beams.sections): check against the tables of the manufacturers and the standards, exact formulas
against the integration of the outline, parametric shapes, notation of the names; slow (slow): comparison with the FEM `sectionproperties`
and a section from a planar OCP face.

Table values (ArcelorMittal Sections and Merchant Bars, EN 10219-2 / 10210-2 tables) in cm, cm², cm⁴, cm⁶.
"""
import math

import numpy as np
import pytest

from fea_core.beams import sections as S
from fea_core.beams.props import SectionProps

CM2, CM3, CM4, CM6 = 1e2, 1e3, 1e4, 1e6

# name: A [cm²], Iy [cm⁴], Iz [cm⁴], It [cm⁴], mass [kg/m]
TABLE = {
    "IPE 80": (7.64, 80.14, 8.49, 0.70, 6.0),
    "IPE 100": (10.32, 171.0, 15.92, 1.20, 8.1),
    "IPE 120": (13.21, 317.8, 27.67, 1.74, 10.4),
    "IPE 140": (16.43, 541.2, 44.92, 2.45, 12.9),
    "IPE 160": (20.09, 869.3, 68.31, 3.60, 15.8),
    "IPE 180": (23.95, 1317, 100.9, 4.79, 18.8),
    "IPE 200": (28.48, 1943, 142.4, 6.98, 22.4),
    "IPE 220": (33.37, 2772, 204.9, 9.07, 26.2),
    "IPE 240": (39.12, 3892, 283.6, 12.88, 30.7),
    "IPE 270": (45.94, 5790, 419.9, 15.94, 36.1),
    "IPE 300": (53.81, 8356, 603.8, 20.12, 42.2),
    "IPE 330": (62.61, 11770, 788.1, 28.15, 49.1),
    "IPE 360": (72.73, 16270, 1043, 37.32, 57.1),
    "IPE 400": (84.46, 23130, 1318, 51.08, 66.3),
    "IPE 450": (98.82, 33740, 1676, 66.87, 77.6),
    "IPE 500": (115.5, 48200, 2142, 89.29, 90.7),
    "IPE 550": (134.4, 67120, 2668, 123.2, 106),
    "IPE 600": (156.0, 92080, 3387, 165.4, 122),
    "HEA 100": (21.24, 349.2, 133.8, 5.24, 16.7),
    "HEA 200": (53.83, 3692, 1336, 20.98, 42.3),
    "HEA 300": (112.5, 18260, 6310, 85.17, 88.3),
    "HEA 1000": (346.8, 553800, 14000, 822.4, 272),
    "HEB 100": (26.04, 449.5, 167.3, 9.25, 20.4),
    "HEB 200": (78.08, 5696, 2003, 59.28, 61.3),
    "HEB 300": (149.1, 25170, 8563, 185.0, 117),
    "HEB 600": (270.0, 171000, 13530, 667.2, 212),
    "HEM 200": (131.3, 10640, 3651, 259.4, 103),
    "HEM 300": (303.1, 59200, 19400, 1408, 238),
    "UPE 200": (29.0, 1910, 187, None, 22.8),
    "UPN 100": (13.5, 206, 29.3, None, 10.6),
    "UPN 200": (32.2, 1910, 148, 11.9, 25.3),
    # L: Iy = I_u (principal strong), Iz = I_v
    "L 50×5": (4.80, 17.4, 4.54, None, 3.77),
    "L 60×6": (6.91, 36.1, 9.43, None, 5.42),
    "L 100×10": (19.2, 280, 73.0, None, 15.0),
    "CHS 60,3×4": (7.07, 28.2, 28.2, 56.3, 5.55),
    "CHS 114,3×5": (17.2, 257, 257, 514, 13.5),
    "CHS 168,3×8": (40.3, 1297, 1297, 2595, 31.6),
    "SHS 100×5": (18.4, 271, 271, 441, 14.4),
    "SHS 100×5 HF": (18.7, 279, 279, None, 14.7),
}

# other quantities: Iw [cm⁶], Wpl,y [cm³], Wpl,z [cm³], Wel,y [cm³], Av,z [cm²]
TABLE2 = {
    "IPE 200": (12.99e3, 220.6, 44.61, 194.3, 14.00),
    "IPE 300": (125.9e3, 628.4, 125.2, 557.1, 25.68),
    "IPE 400": (490.0e3, 1307, 229.0, 1156, 42.69),
    "HEB 200": (171.1e3, 642.5, 305.8, 569.6, 24.83),
    "HEA 300": (1200e3, 1383, 641.2, 1260, 37.28),
}

# areas of the whole series [cm²] (ArcelorMittal) – check of the table of the dimensions
A_SERIES = {
    "HEA": {100: 21.2, 120: 25.3, 140: 31.4, 160: 38.8, 180: 45.3, 200: 53.8, 220: 64.3, 240: 76.8, 260: 86.8,
            280: 97.3, 300: 112.5, 320: 124.4, 340: 133.5, 360: 142.8, 400: 159.0, 450: 178.0, 500: 197.5,
            550: 211.8, 600: 226.5, 650: 241.6, 700: 260.5, 800: 285.8, 900: 320.5, 1000: 346.8},
    "HEB": {100: 26.0, 120: 34.0, 140: 43.0, 160: 54.3, 180: 65.3, 200: 78.1, 220: 91.0, 240: 106.0, 260: 118.4,
            280: 131.4, 300: 149.1, 320: 161.3, 340: 170.9, 360: 180.6, 400: 197.8, 450: 218.0, 500: 238.6,
            550: 254.1, 600: 270.0},
    "HEM": {100: 53.2, 120: 66.4, 140: 80.6, 160: 97.1, 180: 113.3, 200: 131.3, 220: 149.4, 240: 199.6,
            260: 219.6, 280: 240.2, 300: 303.1, 320: 312.0, 340: 315.8, 360: 318.8, 400: 325.8, 450: 335.4,
            500: 344.3, 550: 354.4, 600: 363.7},
    "UPE": {80: 10.1, 100: 12.5, 120: 15.4, 140: 18.4, 160: 21.7, 180: 25.1, 200: 29.0, 220: 33.9, 240: 38.5,
            270: 44.8, 300: 56.6, 330: 67.8, 360: 77.9, 400: 91.9},
}
# UPN: A [cm²], Iy [cm⁴] (DIN 1026-1, sloped flanges)
UPN_TAB = {80: (11.0, 106), 100: (13.5, 206), 120: (17.0, 364), 140: (20.4, 605), 160: (24.0, 925),
           180: (28.0, 1350), 200: (32.2, 1910), 220: (37.4, 2690), 240: (42.3, 3600), 260: (48.3, 4820),
           280: (53.3, 6280), 300: (58.8, 8030), 320: (75.8, 10870), 350: (77.3, 12840), 380: (80.4, 15760),
           400: (91.5, 20350)}
# L equal-leg: A [cm²], I (about the axis ∥ to the leg) [cm⁴]
L_TAB = {"L 20×3": (1.12, 0.39), "L 30×3": (1.74, 1.40), "L 40×4": (3.08, 4.47), "L 50×5": (4.80, 11.0),
         "L 60×6": (6.91, 22.8), "L 80×8": (12.3, 72.2), "L 100×10": (19.2, 177), "L 120×12": (27.5, 368),
         "L 150×15": (43.0, 898), "L 200×20": (76.3, 2852)}


def rel(a, b):
    return abs(a / b - 1.0)


# ---------------------------------------------------------------- tabulky
@pytest.mark.parametrize("name", list(TABLE))
def test_against_tables(name):
    p = S.get(name)
    A, Iy, Iz, It, m = TABLE[name]
    tol_a = 0.01 if A >= 10 else 0.006 / A + 0.005     # tables of small sections have 3 valid digits
    assert rel(p.A, A * CM2) <= max(tol_a, 0.01), (p.A, A)
    assert rel(p.Iy, Iy * CM4) <= 0.01, (p.Iy, Iy)
    assert rel(p.Iz, Iz * CM4) <= 0.01, (p.Iz, Iz)
    if It is not None:
        assert rel(p.J, It * CM4) <= 0.03, (p.J, It)
    assert rel(p.mass_per_m(), m) <= 0.012, (p.mass_per_m(), m)
    assert p.Iy >= p.Iz
    assert p.source


@pytest.mark.parametrize("name", list(TABLE2))
def test_against_tables_more(name):
    p = S.get(name)
    Iw, Wy, Wz, Wely, Avz = TABLE2[name]
    assert rel(p.Iw, Iw * CM6) <= 0.01
    assert rel(p.Wpl_y, Wy * CM3) <= 0.01
    assert rel(p.Wpl_z, Wz * CM3) <= 0.01
    assert rel(p.Wel_y, Wely * CM3) <= 0.01
    assert rel(p.Av_z, Avz * CM2) <= 0.01


@pytest.mark.parametrize("fam", list(A_SERIES))
def test_series_areas(fam):
    for k, A in A_SERIES[fam].items():
        p = S.get(f"{fam} {k}")
        assert rel(p.A, A * CM2) <= 0.006, (fam, k, p.A / CM2, A)


def test_upn_tapered_flanges():
    for k, (A, Iy) in UPN_TAB.items():
        p = S.get(f"UPN {k}")
        assert rel(p.A, A * CM2) <= 0.006, (k, p.A / CM2, A)
        assert rel(p.Iy, Iy * CM4) <= 0.006, (k, p.Iy / CM4, Iy)
        assert p.ys < 0                   # shear centre behind the web


def test_angles_geometric_axes():
    for name, (A, I) in L_TAB.items():
        p = S.get(name)
        assert rel(p.A, A * CM2) <= 0.006 + 0.006 / A, (name, p.A / CM2, A)
        assert abs(p.alpha - math.pi / 4) < 1e-9
        # I about the axis parallel to the leg = average of the principal moments (equal-leg L)
        assert rel(0.5 * (p.Iy + p.Iz), I * CM4) <= 0.01 + 0.006 / I, (name, 0.5 * (p.Iy + p.Iz) / CM4, I)


def test_ipe200_reference_values():
    """IPE 200 as specified: A 28.5; Iy 1943; Iz 142; It 6.98; Iw 12.99·10³; Wpl,y 221; 22.4 kg/m."""
    p = S.get("IPE 200")
    assert round(p.A / CM2, 1) == 28.5
    assert round(p.Iy / CM4) == 1943
    assert round(p.Iz / CM4) == 142
    assert round(p.J / CM4, 2) == 6.98
    assert round(p.Iw / CM6 / 1e3, 2) == 12.99
    assert round(p.Wpl_y / CM3) == 221
    assert round(p.mass_per_m(), 1) == 22.4


# ---------------------------------------------------------------- exact formulas = integration of the outline
@pytest.mark.parametrize("name", ["IPE 200", "HEB 300", "HEM 600", "RHS 120×60×4", "SHS 100×5",
                                  "SHS 200×12,5", "RHS 200×100×6 HF", "CHS 60,3×4"])
def test_closed_form_matches_outline(name):
    p = S.get(name)
    g = S.polygon_props(S._rings(p.shape, p.dims, 256))
    for k in ("A", "Iy", "Iz", "Wpl_y", "Wpl_z"):
        assert rel(getattr(p, k), g[k]) < 2e-4, (k, getattr(p, k), g[k])


def test_outline_centroid_and_fibers():
    for name in ["IPE 200", "UPE 200", "L 100×50×6", "RHS 120×60×4", "CHS 60,3×4"]:
        p = S.get(name)
        rings = S.outline(p, nseg=64)
        g = S._integrals(S._orient(rings))
        assert abs(g[1] / g[0]) < 1e-6 * p.A and abs(g[2] / g[0]) < 1e-6 * p.A
        assert abs(g[5]) < 1e-4 * p.Iy                     # principal axes → deviation moment 0
        assert rel(g[0], p.A) < 1e-3
        f = np.asarray(p.fibers)
        assert len(f) >= 4
        # the extreme fibres give Wel
        assert rel(p.Wel_y, p.Iy / np.abs(f[:, 1]).max()) < 0.01
        assert rel(p.Wel_z, p.Iz / np.abs(f[:, 0]).max()) < 0.01
    assert len(S.get("CHS 60,3×4").fibers) == 16


# ---------------------------------------------------------------- names, series, search, cache
@pytest.mark.parametrize("text,name", [
    ("IPE200", "IPE 200"), ("ipe 200", "IPE 200"), ("HE 200 B", "HEB 200"), ("HE200A", "HEA 200"),
    ("HEM200", "HEM 200"), ("U 200", "UPN 200"), ("UPE200", "UPE 200"),
    ("L50x5", "L 50×5"), ("L 50×5", "L 50×5"), ("L 50x50x5", "L 50×5"), ("L45x4,5", "L 45×4,5"),
    ("L 50x100x6", "L 100×50×6"), ("CHS 60.3x4", "CHS 60,3×4"), ("TR 60,3×4", "CHS 60,3×4"),
    ("TR Ø60,3x4", "CHS 60,3×4"), ("JEKL 100x100x5", "SHS 100×5"), ("JEKL 120x60x4", "RHS 120×60×4"),
    ("RHS 60x120x4", "RHS 120×60×4"), ("TR 4HR 100x5", "SHS 100×5"), ("SHS 100x5 HF", "SHS 100×5 HF"),
    ("SHS 100x5 EN 10210", "SHS 100×5 HF"), ("SHS 100x5 CF", "SHS 100×5"), ("CHS 273x10", "CHS 273×10"),
])
def test_name_parsing(text, name):
    assert S.get(text).name == name
    assert S.canonical(text) == name


def test_unknown_and_out_of_series():
    with pytest.raises(S.SectionError):
        S.get("IPE 210")
    with pytest.raises(S.SectionError):
        S.get("XYZ 12")
    with pytest.raises(S.SectionError):
        S.get("L 55×5")                       # radii unknown → parametric
    p = S.get("SHS 110×4")                    # tubes outside the series are calculated from the dimensions
    assert "outside the library range" in p.source and p.dims["ro"] == 8.0
    hot = S.get("SHS 100×5 HF")
    assert hot.dims["fab"] == "hot" and hot.dims["ro"] == 7.5 and hot.dims["ri"] == 5.0
    cold = S.get("SHS 100×10")
    assert cold.dims["ro"] == 25.0 and cold.dims["ri"] == 15.0    # 2.5t for 6 < t ≤ 10
    assert S.get("SHS 200×12,5").dims["ro"] == 37.5                # 3t for t > 10


def test_families_and_search():
    f = S.families()
    assert f["IPE"][0] == "IPE 80" and f["IPE"][-1] == "IPE 600" and len(f["IPE"]) == 18
    assert len(f["HEA"]) == 24 and len(f["HEB"]) == 19 and len(f["HEM"]) == 19
    assert "L 50×5" in f["L equal"] and "L 100×50×6" in f["L unequal"]
    assert "CHS 60,3×4" in f["CHS (circular tubes)"]
    for names in f.values():
        assert names
        for n in names[:3] + names[-2:]:
            assert S.get(n).name == n
    assert S.search("IPE200")[0] == "IPE 200"
    r = S.search("HEB 2")
    assert "HEB 200" in r and "HEB 220" in r and all(n.startswith("HEB 2") for n in r)
    assert S.search("HE 300 B")[0] == "HEB 300"
    assert "SHS 100×5" in S.search("100×5")


def test_get_returns_copy():
    a = S.get("IPE 200")
    a.A = -1.0
    a.fibers.append((0.0, 0.0))
    b = S.get("IPE200")
    assert b.A > 0 and len(b.fibers) == 4


def test_to_dict_roundtrip():
    p = S.get("L 100×50×6")
    q = SectionProps.from_dict(p.to_dict())
    assert q.name == p.name and q.alpha == p.alpha and q.fibers == [tuple(x) for x in p.fibers]


# ---------------------------------------------------------------- parametric shapes
def test_rect_saint_venant():
    # β (J = β·a·b³) and the factor Wt = α·a·b² for the ratios of the sides (Timoshenko & Goodier)
    for ratio, beta, alpha in [(1.0, 0.1406, 0.208), (2.0, 0.229, 0.246), (4.0, 0.281, 0.282),
                               (10.0, 0.312, 0.312)]:
        p = S.parametric("RECT", b=10.0, h=10.0 * ratio)
        assert rel(p.J, beta * 10.0 * ratio * 1e3) < 2e-3, ratio
        assert rel(p.Wt, alpha * 10.0 * ratio * 1e2) < 4e-3, ratio
    p = S.parametric("RECT", b=40.0, h=20.0)        # strong axis vertical → rotation by 90°
    assert p.Iy == pytest.approx(20 * 40 ** 3 / 12) and p.alpha == pytest.approx(math.pi / 2)
    assert p.Wpl_y == pytest.approx(20 * 40 ** 2 / 4)


def test_circle_and_tube():
    c = S.parametric("CIRC", d=20.0)
    assert c.J == pytest.approx(math.pi * 20 ** 4 / 32) and c.Wt == pytest.approx(math.pi * 20 ** 3 / 16)
    assert c.Wpl_y == pytest.approx(20 ** 3 / 6)
    t = S.parametric("TUBE", D=60.3, t=4.0)
    g = S.get("CHS 60,3×4")
    assert t.A == pytest.approx(g.A) and t.J == pytest.approx(2 * t.Iy)
    assert t.Av_z == pytest.approx(2 * t.A / math.pi)


def test_rhs_parametric_and_shear_area():
    p = S.parametric("RHS", h=120, b=60, t=4)
    assert p.A == pytest.approx(S.get("RHS 120×60×4").A)
    assert p.Av_z == pytest.approx(p.A * 120 / 180) and p.Av_y == pytest.approx(p.A * 60 / 180)
    q = S.parametric("RHS", h=100, b=100, t=5, ro=0.0)      # sharp corners = difference of rectangles
    assert q.A == pytest.approx(100 ** 2 - 90 ** 2)
    assert q.Iy == pytest.approx((100 ** 4 - 90 ** 4) / 12)


def test_welded_i_symmetric_and_mono():
    s = S.parametric("I", h=400, b=200, tw=8, tf=12)
    A = 2 * 200 * 12 + 376 * 8
    assert s.A == pytest.approx(A)
    assert s.Iy == pytest.approx((200 * 400 ** 3 - 192 * 376 ** 3) / 12)
    assert s.Av_z == pytest.approx(376 * 8) and s.dims["fab"] == "welded"
    m = S.parametric("I", h=400, b=200, tw=8, tf=12, b2=300, tf2=20)
    A = 200 * 12 + 300 * 20 + 368 * 8
    zc = (200 * 12 * 394 + 300 * 20 * 10 + 368 * 8 * 204) / A            # od spodu
    Iy = (200 * 12 ** 3 / 12 + 200 * 12 * (394 - zc) ** 2 + 300 * 20 ** 3 / 12 + 300 * 20 * (10 - zc) ** 2
          + 8 * 368 ** 3 / 12 + 368 * 8 * (204 - zc) ** 2)
    assert m.A == pytest.approx(A) and m.Iy == pytest.approx(Iy, rel=1e-9)
    I1, I2 = 12 * 200 ** 3 / 12, 20 * 300 ** 3 / 12
    hs = 400 - 6 - 10
    assert m.Iw == pytest.approx(hs ** 2 * I1 * I2 / (I1 + I2))
    # shear centre closer to the larger (bottom) flange: from the top flange hs·I2/(I1 + I2)
    z_top = 394 - zc
    assert m.zs == pytest.approx(z_top - hs * I2 / (I1 + I2))
    assert m.zs < 0


def test_tee_and_sheet_channel():
    t = S.parametric("T", h=100, b=80, tw=8, tf=10)
    A = 80 * 10 + 90 * 8
    zc = (800 * 95 + 720 * 45) / A
    assert t.A == pytest.approx(A)
    assert t.Iy == pytest.approx(80 * 1000 / 12 + 800 * (95 - zc) ** 2 + 8 * 90 ** 3 / 12 + 720 * (45 - zc) ** 2)
    assert t.zs == pytest.approx(95 - zc)                                  # shear centre at the intersection of the axes
    u = S.parametric("U", h=100, b=50, t=2)                                # z plechu, ri = t
    rm = 3.0
    Ls = 2 * (50 - 1 - rm) + (100 - 2 - 2 * rm) + math.pi * rm
    assert u.A == pytest.approx(Ls * 2, rel=1e-4)                          # area = midline × t
    assert u.ys < 0 and u.Iw > 0


def test_angle_parametric_principal_axes():
    p = S.parametric("L", h=100, b=100, t=10)
    assert p.alpha == pytest.approx(math.pi / 4)
    q = S.parametric("L", h=100, b=50, t=6, r1=9, r2=4.5)
    g = S.get("L 100×50×6")
    assert q.Iy == pytest.approx(g.Iy) and q.alpha == pytest.approx(g.alpha)
    # unequal-leg L: the principal axis y makes with the leg b the angle tan 2α = 2Iyz/(Iz0 − Iy0)
    assert 0 < q.alpha < math.pi / 4


# ---------------------------------------------------------------- sectionproperties (slow)
def _sp(p, mesh=None):
    rings = S._orient(S._rings(p.shape, p.dims, 32))
    return S.custom_section("kontrola", rings[0].tolist(), [r.tolist() for r in rings[1:]], mesh_size=mesh)


@pytest.mark.slow
@pytest.mark.parametrize("name,tol_j", [("IPE 200", 0.025), ("HEB 200", 0.01), ("UPE 80", 0.01), ("UPE 200", 0.01),
                                        ("UPN 80", 0.01), ("UPN 200", 0.01), ("UPN 400", 0.01), ("L 20×3", 0.015),
                                        ("L 50×5", 0.015), ("L 100×50×6", 0.015), ("L 200×20", 0.015),
                                        ("RHS 120×60×4", 0.02), ("SHS 100×5 HF", 0.02), ("CHS 60,3×4", 0.002)])
def test_against_sectionproperties(name, tol_j):
    p = S.get(name)
    c = _sp(p)
    for k in ("A", "Iy", "Iz", "Wpl_y", "Wpl_z"):
        assert rel(getattr(p, k), getattr(c, k)) < 1e-3, (k, getattr(p, k), getattr(c, k))
    assert rel(p.J, c.J) < tol_j, (p.J, c.J)
    if rel(p.Iy, p.Iz) > 1e-6:                     # for a circle / square the angle of the principal axes is arbitrary
        assert abs(p.alpha - c.alpha) < 1e-6
    # I_w and the shear centre are for a thin-walled U (FEM: UPE −1 … −5 %, UPN +4 … +8 %, shear centre 3 … 7 %)
    if p.shape == "I":
        assert rel(p.Iw, c.Iw) < 0.03, (p.Iw, c.Iw)
    if p.shape == "U":
        assert rel(p.Iw, c.Iw) < 0.09, (p.Iw, c.Iw)
        assert abs(p.ys - c.ys) < 0.08 * abs(c.ys) and abs(c.zs) < 1e-4 * p.dims["h"]
    if p.shape == "L":
        assert math.hypot(p.ys - c.ys, p.zs - c.zs) < 0.1 * p.dims["t"] + 0.08 * math.hypot(c.ys, c.zs)


@pytest.mark.slow
@pytest.mark.parametrize("dims", [dict(h=200, b=150, tw=10, tf=15), dict(h=200, b=150, tw=10, tf=15, r=12),
                                  dict(h=100, b=100, tw=20, tf=10), dict(h=60, b=60, tw=7, tf=7, r=7)])
def test_tee_against_sectionproperties(dims):
    p = S.parametric("T", **dims)
    c = _sp(p)
    assert rel(p.A, c.A) < 1e-3 and rel(p.Iy, c.Iy) < 1e-3 and rel(p.Wpl_y, c.Wpl_y) < 1e-3
    assert rel(p.J, c.J) < 0.01, (p.J, c.J)
    assert abs(p.zs - c.zs) < 0.2 * abs(c.zs)


@pytest.mark.slow
def test_custom_ipe200_polygon():
    """IPE 200 as a polygon with the arcs approximated by segments (8 per quarter circle) against the exact formulas."""
    p = S.get("IPE 200")
    pts = S._rings("I", p.dims, 8)[0]
    c = S.custom_section("IPE 200 – obrys", pts.tolist())
    assert rel(c.A, p.A) < 2e-3 and rel(c.Iy, p.Iy) < 2e-3 and rel(c.Iz, p.Iz) < 2e-3
    assert rel(c.Wpl_y, p.Wpl_y) < 2e-3
    assert rel(c.J, p.J) < 0.03 and rel(c.Iw, p.Iw) < 0.03
    assert abs(c.ys) < 0.02 and abs(c.zs) < 0.02 and abs(c.alpha) < 1e-9
    # FEM shear areas (elastic) – for a force along the web smaller than along the flanges
    assert c.Av_z < c.Av_y


@pytest.mark.slow
def test_custom_rect_and_tube_exact():
    r = S.custom_section("obdélník", [(0, 0), (50, 0), (50, 100), (0, 100)])
    e = S.parametric("RECT", b=50, h=100)
    assert rel(r.A, e.A) < 1e-9 and rel(r.Iy, e.Iy) < 1e-9 and rel(r.Iz, e.Iz) < 1e-9
    assert rel(r.J, e.J) < 2e-3 and rel(r.Wpl_y, e.Wpl_y) < 1e-6
    assert rel(r.Wt, e.Wt) < 0.02
    assert abs(r.alpha) < 1e-9
    D, t = 60.3, 4.0
    o = S._circle(D / 2, 256)
    i = S._circle(D / 2 - t, 256)
    c = S.custom_section("trubka", o.tolist(), [i.tolist()])
    e = S.parametric("TUBE", D=D, t=t)
    assert rel(c.A, e.A) < 1e-3 and rel(c.Iy, e.Iy) < 1e-3 and rel(c.J, e.J) < 2e-3


def _ocp_face(outer, holes=(), z=0.0, rot=False):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pnt

    def wire(pts):
        mp = BRepBuilderAPI_MakePolygon()
        for (x, y) in pts:
            mp.Add(gp_Pnt(float(x), float(y), z) if not rot else gp_Pnt(z, float(x), float(y)))
        mp.Close()
        return mp.Wire()
    mf = BRepBuilderAPI_MakeFace(wire(outer), True)
    for h in holes:
        w = wire(h[::-1])
        mf.Add(w)
    return mf.Face()


@pytest.mark.slow
def test_section_from_face_rect_and_tube():
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Circ, gp_Ax2, gp_Pnt, gp_Dir

    # rectangle 40 × 100 in the plane x = 5 (normal ±x): strong axis along 100 → the y axis ∥ to the edge 40
    f = _ocp_face([(0, 0), (40, 0), (40, 100), (0, 100)], z=5.0, rot=True)
    sec, place = S.section_from_face(f, "plech")
    e = S.parametric("RECT", b=40, h=100)
    assert rel(sec.A, e.A) < 1e-9 and rel(sec.Iy, e.Iy) < 1e-9 and rel(sec.J, e.J) < 2e-3
    assert np.allclose(place["centroid"], (5.0, 20.0, 50.0))
    assert abs(abs(place["normal"][0]) - 1) < 1e-12
    assert abs(abs(place["y_axis"][1]) - 1) < 1e-9          # strong axis y horizontally along the edge 40
    assert abs(np.dot(place["y_axis"], place["z_axis"])) < 1e-12
    # tube – true circles (discretisation of the arcs)
    D, t = 60.3, 4.0
    ax = gp_Ax2(gp_Pnt(10, 0, 0), gp_Dir(0, 0, 1))
    wo = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Circ(ax, D / 2)).Edge()).Wire()
    wi = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Circ(ax, D / 2 - t)).Edge()).Wire()
    wi.Reverse()
    mf = BRepBuilderAPI_MakeFace(wo, True)
    mf.Add(wi)
    sec, place = S.section_from_face(mf.Face(), "trubka")
    e = S.parametric("TUBE", D=D, t=t)
    assert rel(sec.A, e.A) < 2e-3 and rel(sec.Iy, e.Iy) < 2e-3 and rel(sec.J, e.J) < 3e-3
    assert np.allclose(place["centroid"], (10.0, 0.0, 0.0), atol=1e-6)
