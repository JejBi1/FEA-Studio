"""Geometry of the example models of FEA Studio (OCP, part names as from the CAD).

Example 1 "Printed bracket" – one part of PLA (a wall plate with 2 holes for screws, an arm with an eye and a rib).
Example 2 "Assembly" – a steel foot with a welded column (2 fillet welds a4), a printed arm of PLA screwed onto the foot (2× ISO 4017 M6 + washers + nuts), the foot fixed by 4 holes Ø9 to the floor.
Example 3 "Flange" – a steel pressure pipe with a flange and a solid cover, joined by 4× screws ISO 4017 M10 × 45 8.8
with washers ISO 7089 and nuts ISO 4032.
Example 4 "Jib" – a welded crane jib: an anchor plate with 4 holes Ø13 for the anchors, a horizontal load-bearing arm
with an eye Ø20 for the load, a triangular gusset and 5 fillet welds a4.
Example 5 "Heat sink" – an aluminium finned heat sink with a stainless power module (heat transfer, convection, heating
and thermal stress).
Example 7 "Vibration fixture with a PCB" – an aluminium bracket with ribs, a printed circuit board FR-4 on 4 spacer posts,
a pad of a heavy component (transformer as a point mass).
Example 8 "Crank mechanism" – a frame with a guide, a crank with pins, a connecting rod with two eyes, a piston (slider) with a pin.
Example 9 "Bolted base" – a foundation plate and a base with a column connected by 2× ISO 4017 M12 8.8 with washers and nuts.
Example 10 "Bracket – topology optimisation" – a solid plate as the design space of a wall bracket (2 holes for screws, an eye for the load).

Coordinates in mm, z up (= the build direction of the print). Run: `python -m examples.create` (geometry + studies).
"""
from __future__ import annotations

import math

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec

from fea_core.geometry import Part
from fea_core.joints import tables

# colours (light, printed parts yellow)
TISK = (0.98, 0.82, 0.20)
OCEL = (0.78, 0.80, 0.83)
SVAR = (0.93, 0.45, 0.25)
SROUB = (0.55, 0.57, 0.60)
HLINIK = (0.82, 0.86, 0.90)
MODUL_BARVA = (0.85, 0.45, 0.25)


def box(x0, y0, z0, dx, dy, dz):
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), dx, dy, dz).Shape()


def cyl(p, d, r, h):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p), gp_Dir(*d)), r, h).Shape()


def solid(shape):
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    out = []
    while ex.More():
        out.append(TopoDS.Solid(ex.Current()))
        ex.Next()
    if len(out) != 1:
        raise ValueError(f"čekáno 1 těleso, je {len(out)}")
    return out[0]


def fuse(*shapes):
    s = shapes[0]
    for t in shapes[1:]:
        s = BRepAlgoAPI_Fuse(s, t).Shape()
    return s


def cut(s, *tools):
    for t in tools:
        s = BRepAlgoAPI_Cut(s, t).Shape()
    return s


def prism(points, height, direction=(0, 0, 1)):
    """A prism from a polygon (points in a plane) extruded by height in the direction direction."""
    poly = BRepBuilderAPI_MakePolygon()
    for p in points:
        poly.Add(gp_Pnt(*map(float, p)))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    d = [c * height for c in direction]
    return BRepPrimAPI_MakePrism(face, gp_Vec(*d)).Shape()


def _hexagon(x, y, s, z):
    r = s / math.sqrt(3)
    return [(x + r * math.cos(math.radians(30 + 60 * k)), y + r * math.sin(math.radians(30 + 60 * k)), z)
            for k in range(6)]


def hex_bolt(x, y, z_bear, d, L):
    """Screw ISO 4017: bearing face of the head at z_bear, the head up, the shank down."""
    h = tables.head_dims(d, "hex")
    return solid(fuse(prism(_hexagon(x, y, h["s"], z_bear), h["k"]), cyl((x, y, z_bear), (0, 0, -1), d / 2, L)))


def hex_nut(x, y, z_bear, d):
    """Nut ISO 4032 under the joint: bearing face at z_bear, the body down."""
    n = tables.nut_dims(d)
    body = prism(_hexagon(x, y, n["s"], z_bear), n["m"], (0, 0, -1))
    return solid(cut(body, cyl((x, y, z_bear - n["m"] - 1), (0, 0, 1), d / 2, n["m"] + 2)))


def washer(x, y, z0, d, up=True):
    w = tables.washer_dims(d)
    z = z0 if up else z0 - w["h"]
    return solid(cut(cyl((x, y, z), (0, 0, 1), w["d2"] / 2, w["h"]), cyl((x, y, z - 1), (0, 0, 1), w["d1"] / 2, w["h"] + 2)))


# ------------------------------------------------------------------ example 1
def drzak() -> list[Part]:
    """Printed bracket (PLA): wall plate 60×40×6 (vertical, x 0..6) with 2 holes Ø5 for the screws, a horizontal arm
    64×24×8, at the end a boss with an eye Ø8 (axis y – the load hung on a pin) and a triangular rib under the arm.
    It is printed as modelled (build z).
    """
    wall = box(0, 0, 0, 6, 40, 60)                       # x 0..6, y 0..40, z 0..60
    arm = box(6, 8, 44, 64, 24, 8)                        # x 6..70, y 8..32, z 44..52
    lug = box(54, 8, 30, 16, 24, 14)                      # boss x 54..70, z 30..44
    # rib thk. 6 (y 17..23) from the wall up to the boss – without a sharp end (notch) under the arm
    rib = prism([(6, 17, 10), (6, 17, 44), (54, 17, 44), (54, 17, 30)], 6, (0, 1, 0))
    s = fuse(wall, arm, lug, rib)
    holes = [cyl((-1, 20, z), (1, 0, 0), 2.5, 8) for z in (12, 30)]       # holes for the screws Ø5
    eye = cyl((62, 7, 39), (0, 1, 0), 4.0, 26)                            # eye Ø8, axis y (the pin of the load)
    return [Part("Drzak (tisk PLA)", solid(cut(s, *holes, eye)), color=TISK)]


# ------------------------------------------------------------------ example 2
PATKA = (0.0, 0.0, -10.0, 180.0, 70.0, 10.0)     # x0, y0, z0, dx, dy, dz
D_KOTVA = 9.0                                    # holes for the anchors (cylindrical constraint)
KOTVY = [(12, 12), (12, 58), (168, 12), (168, 58)]
M = 6.0                                          # screws of the arm
SROUBY = [(115.0, 23.0), (115.0, 47.0)]     # outside the rib of the arm (y 32..38)
T_RAMENO = 8.0                                   # thickness of the flange of the printed arm
A_SVAR = 4.0


def sestava() -> list[Part]:
    """Steel foot 180×70×10 (z −10..0) with 4 holes Ø9, a column 10×70×90 (x 30..40) welded by two fillet welds a4 along y, a printed arm (a flange x 95..155 on the foot + a vertical wall with an eye Ø10) screwed by
    2× ISO 4017 M6 x 30 - 8.8 with washers ISO 7089 and nuts ISO 4032 under the foot.
    """
    x0, y0, z0, dx, dy, dz = PATKA
    dh = tables.CLEARANCE[tables.size_key(M)][1]
    patka = box(x0, y0, z0, dx, dy, dz)
    patka = cut(patka, *[cyl((x, y, z0 - 1), (0, 0, 1), D_KOTVA / 2, dz + 2) for x, y in KOTVY],
                *[cyl((x, y, z0 - 1), (0, 0, 1), dh / 2, dz + 2) for x, y in SROUBY])
    sloupek = cut(box(30, 0, 0, 10, dy, 90), cyl((29, 35, 75), (1, 0, 0), 5.0, 12))   # hole Ø10 at the top
    z = A_SVAR * math.sqrt(2)                     # leg of the fillet weld
    svar_l = prism([(30, 0, 0), (30, 0, z), (30 - z, 0, 0)], dy, (0, 1, 0))
    svar_p = prism([(40, 0, 0), (40 + z, 0, 0), (40, 0, z)], dy, (0, 1, 0))

    t = T_RAMENO
    priruba = box(95, 15, 0, 60, 40, t)                               # x 95..155, y 15..55, z 0..8
    stena = box(145, 15, t, 10, 40, 52)                               # x 145..155, z 8..60
    zebro = prism([(145, 32, t), (145, 32, 50), (105, 32, t)], 6, (0, 1, 0))   # y 32..38
    rameno = fuse(priruba, stena, zebro)
    rameno = cut(rameno, *[cyl((x, y, -1), (0, 0, 1), dh / 2, t + 2) for x, y in SROUBY],
                 cyl((144, 35, 48), (1, 0, 0), 5.0, 12))              # eye Ø10 in the wall (axis x)

    wd = tables.washer_dims(M)
    nut = tables.nut_dims(M)
    L = math.ceil((t + dz + 2 * wd["h"] + nut["m"] + 3) / 5) * 5
    parts = [Part("Patka", solid(patka), color=OCEL), Part("Sloupek", solid(sloupek), color=OCEL),
             Part(f"Svar a{A_SVAR:g} vlevo", solid(svar_l), color=SVAR),
             Part(f"Svar a{A_SVAR:g} vpravo", solid(svar_p), color=SVAR),
             Part("Rameno (tisk PLA)", solid(rameno), color=TISK)]
    for k, (x, y) in enumerate(SROUBY, 1):
        parts += [Part(f"ISO 4017 - M{M:g} x {L:g} - 8.8:{k}", hex_bolt(x, y, t + wd["h"], M, L), color=SROUB),
                  Part(f"ISO 7089 - {M:g} - 200 HV:{2 * k - 1}", washer(x, y, t, M, up=True), color=SROUB),
                  Part(f"ISO 7089 - {M:g} - 200 HV:{2 * k}", washer(x, y, z0, M, up=False), color=SROUB),
                  Part(f"ISO 4032 - M{M:g} - 8:{k}", hex_nut(x, y, z0 - wd["h"], M), color=SROUB)]
    return parts


# ------------------------------------------------------------------ example 3: Flange
PRIRUBA_SROUBY = [(-30.0, -30.0), (30.0, -30.0), (30.0, 30.0), (-30.0, 30.0)]
M_PRIRUBA = 10.0
T_FLANGE = 15.0
T_VIKO = 15.0
R_PIPE_OUT = 25.0
R_PIPE_IN = 18.0
R_FLANGE = 60.0
L_PIPE = 60.0


def priruba() -> list[Part]:
    """A pressure pipe with a flange and a solid cover joined by 4× screws ISO 4017 M10 × 45 - 8.8.
    Outer pipe Ø50, inner Ø36, flange and cover Ø120, thickness 15 mm, 4 holes Ø11 at the spacing 84.85 mm.
    """
    dh = tables.CLEARANCE[tables.size_key(M_PRIRUBA)][1]  # 11.0 mm

    # lower flange with the pipe: z from −60 to 0
    trubka_vnejsi = cyl((0, 0, -L_PIPE), (0, 0, 1), R_PIPE_OUT, L_PIPE)
    priruba_kotouc = cyl((0, 0, -T_FLANGE), (0, 0, 1), R_FLANGE, T_FLANGE)
    priruba_telo = fuse(trubka_vnejsi, priruba_kotouc)

    vnitrni_dira = cyl((0, 0, -L_PIPE - 1), (0, 0, 1), R_PIPE_IN, L_PIPE + 2)
    sroub_diry_dole = [cyl((x, y, -T_FLANGE - 1), (0, 0, 1), dh / 2, T_FLANGE + 2) for x, y in PRIRUBA_SROUBY]
    priruba_geo = solid(cut(priruba_telo, vnitrni_dira, *sroub_diry_dole))

    # upper cover: z from 0 to 15
    viko_kotouc = cyl((0, 0, 0), (0, 0, 1), R_FLANGE, T_VIKO)
    sroub_diry_nahore = [cyl((x, y, -1), (0, 0, 1), dh / 2, T_VIKO + 2) for x, y in PRIRUBA_SROUBY]
    viko_geo = solid(cut(viko_kotouc, *sroub_diry_nahore))

    wd = tables.washer_dims(M_PRIRUBA)
    L_sroub = 45.0

    parts = [
        Part("Priruba potrubi", priruba_geo, color=OCEL),
        Part("Viko priruby", viko_geo, color=OCEL),
    ]

    for k, (x, y) in enumerate(PRIRUBA_SROUBY, 1):
        parts += [
            Part(f"ISO 4017 - M{M_PRIRUBA:g} x {L_sroub:g} - 8.8:{k}",
                 hex_bolt(x, y, T_VIKO + wd["h"], M_PRIRUBA, L_sroub), color=SROUB),
            Part(f"ISO 7089 - {M_PRIRUBA:g} - 200 HV:{2 * k - 1}",
                 washer(x, y, T_VIKO, M_PRIRUBA, up=True), color=SROUB),
            Part(f"ISO 7089 - {M_PRIRUBA:g} - 200 HV:{2 * k}",
                 washer(x, y, -T_FLANGE, M_PRIRUBA, up=False), color=SROUB),
            Part(f"ISO 4032 - M{M_PRIRUBA:g} - 8:{k}",
                 hex_nut(x, y, -T_FLANGE - wd["h"], M_PRIRUBA), color=SROUB),
        ]
    return parts


# ------------------------------------------------------------------ example 4: Jib
A_SVAR_VYLOZNIK = 4.0
KOTVY_VYLOZNIK = [(-45.0, -80.0), (45.0, -80.0), (45.0, 80.0), (-45.0, 80.0)]
D_KOTVA_VYLOZNIK = 13.0


def vyloznik() -> list[Part]:
    """A welded jib: an anchor plate 15×140×240 with 4 holes Ø13 for the anchors, a horizontal arm
    50×60 of length 220 with an eye Ø20 for the pin of the load, a triangular gusset thk. 12 and 5 fillet welds a5.
    """
    # anchor plate: x −15..0, y −70..70, z −120..120
    deska = box(-15, -70, -120, 15, 140, 240)
    kotva_diry = [cyl((-16, y, z), (1, 0, 0), D_KOTVA_VYLOZNIK / 2, 17) for y, z in KOTVY_VYLOZNIK]
    deska = solid(cut(deska, *kotva_diry))

    # Rameno: x 0..220, y -25..25, z 40..100
    rameno_telo = box(0, -25, 40, 220, 50, 60)
    # lug at the end with an eye: x 220..260, y −15..15, z 30..100
    lug = box(220, -15, 30, 40, 30, 70)
    oko = cyl((240, -16, 65), (0, 1, 0), 10.0, 32)
    rameno_geo = solid(cut(fuse(rameno_telo, lug), oko))

    # triangular gusset (thickness 12 mm, y −6..6):
    pts_zebro = [(0, -6, 40), (0, -6, -60), (140, -6, 40)]
    zebro_geo = solid(prism(pts_zebro, 12, (0, 1, 0)))

    # fillet welds a = 4 mm
    zw = A_SVAR_VYLOZNIK * math.sqrt(2)
    # 1. top weld arm–plate: x 0..zw, z 100..100+zw, y −25..25 (length 50)
    svar_r_h = prism([(0, -25, 100), (0, -25, 100 + zw), (zw, -25, 100)], 50, (0, 1, 0))
    # 2. weld of the gusset to the plate on the left: in the plane YZ at x=0, from z=−60 to z=40 (length 100)
    svar_z_d_l = prism([(0, -6, -60), (zw, -6, -60), (0, -6 - zw, -60)], 100, (0, 0, 1))
    # 3. weld of the gusset to the plate on the right: corner between the plate (x=0, y ≥ 6) and the gusset (y=6, x ≥ 0)
    svar_z_d_p = prism([(0, 6, -60), (0, 6 + zw, -60), (zw, 6, -60)], 100, (0, 0, 1))
    # 4. weld of the gusset to the arm on the left: under the arm z=40, y=−6, x from zw to 130
    svar_z_r_l = prism([(zw, -6, 40), (zw, -6 - zw, 40), (zw, -6, 40 - zw)], 130 - zw, (1, 0, 0))
    # 5. weld of the gusset to the arm on the right: under the arm z=40, y=6, x from zw to 130
    svar_z_r_p = prism([(zw, 6, 40), (zw, 6, 40 - zw), (zw, 6 + zw, 40)], 130 - zw, (1, 0, 0))

    return [
        Part("Kotevni deska", deska, color=OCEL),
        Part("Rameno vylozniku", rameno_geo, color=OCEL),
        Part("Zebro vylozniku", zebro_geo, color=OCEL),
        Part(f"Svar a{A_SVAR_VYLOZNIK:g} rameno horni", solid(svar_r_h), color=SVAR),
        Part(f"Svar a{A_SVAR_VYLOZNIK:g} zebro svisly vlevo", solid(svar_z_d_l), color=SVAR),
        Part(f"Svar a{A_SVAR_VYLOZNIK:g} zebro svisly vpravo", solid(svar_z_d_p), color=SVAR),
        Part(f"Svar a{A_SVAR_VYLOZNIK:g} zebro vodorovny vlevo", solid(svar_z_r_l), color=SVAR),
        Part(f"Svar a{A_SVAR_VYLOZNIK:g} zebro vodorovny vpravo", solid(svar_z_r_p), color=SVAR),
    ]


# ------------------------------------------------------------------ example 5: Heat sink
CHLADIC_MONT_OTVORY = [(10.0, 14.0), (80.0, 14.0), (10.0, 66.0), (80.0, 66.0)]
D_MONT_OTVOR = 4.5


def chladic() -> list[Part]:
    """An aluminium finned heat sink 90×80×30 mm with 6 fins and a stainless power module 30×30×4 mm."""
    zakladna = box(0, 0, 0, 90, 80, 6)
    zebra_y = [4.0, 18.0, 32.0, 46.0, 60.0, 73.0]
    zebra = [box(0, y, 6, 90, 3, 24) for y in zebra_y]
    telo_chladice = fuse(zakladna, *zebra)

    otvory = [cyl((x, y, -1), (0, 0, 1), D_MONT_OTVOR / 2, 8) for x, y in CHLADIC_MONT_OTVORY]
    chladic_geo = solid(cut(telo_chladice, *otvory))

    modul_geo = solid(box(30, 25, -4, 30, 30, 4))

    return [
        Part("Chladic (Al)", chladic_geo, color=HLINIK),
        Part("Vykonovy modul", modul_geo, color=MODUL_BARVA),
    ]


# ------------------------------------------------------------------ example 7: vibration fixture with a PCB
DPS = (10.0, 45.0, 12.0, 100.0, 1.6, 70.0)          # x0, y of the back side (at the posts), z0, dx, thickness, dz
SLOUPKY_DPS = [(15.0, 17.0), (105.0, 17.0), (15.0, 77.0), (105.0, 77.0)]   # (x, z) of the spacers Ø6
PATICE = (48.0, 40.0, 24.0, 2.0)                     # x0, z0, side of the square, height of the pad above the PCB
DPS_BARVA = (0.30, 0.62, 0.38)


def pripravek_dps() -> list[Part]:
    """An aluminium bracket (a base 120×60×6 screwed to the table of the vibrator, a vertical wall 120×5×80 at the back (y 55..60),
    two ribs thk. 5), in front of the wall a printed circuit board FR-4 100×70×1.6 on 4 spacers Ø6 × 10 and on it
    a pad of a heavy component 24×24×2 (transformer – as a point mass in the example). The component faces the observer (−Y).
    """
    x0, yb, z0, dx, t, dz = DPS
    zakladna = box(0, 0, 0, 120, 60, 6)                                  # z 0..6
    stena = box(0, 55, 6, 120, 5, 80)                                    # y 55..60, z 6..86
    zebra = [prism([(x, 55, 6), (x, 15, 6), (x, 55, 46)], 5, (1, 0, 0)) for x in (5.0, 110.0)]
    konzola = solid(fuse(zakladna, stena, *zebra))
    dps = solid(box(x0, yb - t, z0, dx, t, dz))                          # y 43,4..45
    px, pz, a, h = PATICE
    patice = solid(box(px, yb - t - h, pz, a, h, a))
    parts = [Part("Konzola (Al)", konzola, color=HLINIK), Part("DPS (FR-4)", dps, color=DPS_BARVA)]
    for k, (x, z) in enumerate(SLOUPKY_DPS, 1):
        parts.append(Part(f"Sloupek {k}", solid(cyl((x, 55.0, z), (0, -1, 0), 3.0, 55.0 - yb)),
                          color=(0.85, 0.72, 0.35)))
    parts.append(Part("Patice transformatoru", patice, color=(0.25, 0.25, 0.28)))
    return parts


# ------------------------------------------------------------------ example 8: crank mechanism (+ test T10)
def klikovy_mechanismus(r: float = 25.0, l: float = 100.0) -> list[Part]:
    """The crank mechanism in the plane xy (the axes of the pins z): the frame (a plate z −10..0 with a bearing Ø12 at the origin and the guide of the slider),
    the crank (an arm z 0..8, a shaft Ø12 into the bearing of the frame, a pin Ø10 at x = r), the connecting rod (eyes R11 with holes Ø10, pitch l,
    z 8..18), the piston – the slider 30×36×26 as a fork around the eye B with a pin Ø10. CAD position: the crank at the angle 0° (bottom dead centre).
    """
    xb = r + l                                           # oko B v poloze CAD
    x_ved = l - r - 20                                   # the guide covers the travel of the slider (l − r − 15 … l + r + 15) + 5
    vedeni = [box(x_ved, y, -3, 2 * r + 40, 7, 37) for y in (18.0, -25.0)]
    # plate z −13..−3, bearing eye R16 up to z = 0 (the support of the crank – axial support → rotary joint), hole Ø12
    ram = fuse(box(-35, -30, -13, r + l + 75, 60, 10), cyl((0, 0, -13), (0, 0, 1), 16.0, 13), *vedeni)
    ram = solid(cut(ram, cyl((0, 0, -14), (0, 0, 1), 6.0, 15)))
    klika = solid(fuse(cyl((0, 0, 0), (0, 0, 1), 14.0, 8), cyl((r, 0, 0), (0, 0, 1), 11.0, 8), box(0, -8, 0, r, 16, 8),
                       cyl((0, 0, -13), (0, 0, 1), 6.0, 13), cyl((r, 0, 8), (0, 0, 1), 5.0, 10)))
    ojnice = fuse(box(r, -5, 8, l, 10, 10), cyl((r, 0, 8), (0, 0, 1), 11.0, 10), cyl((xb, 0, 8), (0, 0, 1), 11.0, 10))
    ojnice = solid(cut(ojnice, cyl((r, 0, 7), (0, 0, 1), 5.0, 12), cyl((xb, 0, 7), (0, 0, 1), 5.0, 12)))
    # slider = fork around the eye of the connecting rod (z 0..26, groove z 8..18 open towards the connecting rod) → the centre of mass and the force in the plane of the connecting rod
    # z = 13 – otherwise the joints of the connecting rod would carry the tilting moment of the slider (statically indeterminate) and the connecting rod would bend out of the plane
    pist = cut(box(xb - 15, -18, 0, 30, 36, 26), box(xb - 16, -13, 8, 29, 26, 10))
    pist = solid(fuse(pist, cyl((xb, 0, 0), (0, 0, 1), 5.0, 26)))
    return [Part("Ram", ram, color=(0.62, 0.66, 0.72)), Part("Klika", klika, color=(0.30, 0.50, 0.80)),
            Part("Ojnice", ojnice, color=(0.95, 0.60, 0.20)), Part("Pist", pist, color=(0.55, 0.75, 0.45))]


# ------------------------------------------------------------------ example 9: bolted column base (contact)
M_PATKA = 10.0
SROUBY_PATKA = [(-35.0, 0.0), (35.0, 0.0)]
T_PATKA = 10.0                                       # thickness of the base
T_ZAKLAD = 15.0                                      # thickness of the foundation plate
H_SLOUPEK = 70.0                                     # top face of the column (z)
X_ZAKLAD = 60.0                                      # foundation plate x −60..60 (clamped at the ends)


def patka_sroubovana() -> list[Part]:
    """A foundation plate 120×60×15 (z −15..0, clamped at the ends x = ±60), on it a base 100×60×10 with a column
    24×24 (top face z = 70), connected by 2× ISO 4017 - M10 x 45 - 8.8 with washers ISO 7089 and nuts ISO 4032
    under the foundation plate. The gap between the base and the plate = a contact with friction (preload, opening).
    """
    dh = tables.CLEARANCE[tables.size_key(M_PATKA)][1]
    zaklad = cut(box(-X_ZAKLAD, -30, -T_ZAKLAD, 2 * X_ZAKLAD, 60, T_ZAKLAD),
                 *[cyl((x, y, -T_ZAKLAD - 1), (0, 0, 1), dh / 2, T_ZAKLAD + 2) for x, y in SROUBY_PATKA])
    patka = fuse(box(-50, -30, 0, 100, 60, T_PATKA), box(-12, -12, T_PATKA, 24, 24, H_SLOUPEK - T_PATKA))
    patka = cut(patka, *[cyl((x, y, -1), (0, 0, 1), dh / 2, T_PATKA + 2) for x, y in SROUBY_PATKA])
    wd = tables.washer_dims(M_PATKA)
    nut = tables.nut_dims(M_PATKA)
    L = math.ceil((T_PATKA + T_ZAKLAD + 2 * wd["h"] + nut["m"] + 3) / 5) * 5
    parts = [Part("Zakladova deska", solid(zaklad), color=OCEL),
             Part("Patka sloupku", solid(patka), color=(0.70, 0.76, 0.86))]
    m = f"{M_PATKA:g}"
    for k, (x, y) in enumerate(SROUBY_PATKA, 1):
        parts += [Part(f"ISO 4017 - M{m} x {L:g} - 8.8:{k}", hex_bolt(x, y, T_PATKA + wd["h"], M_PATKA, L), color=SROUB),
                  Part(f"ISO 7089 - {m} - 200 HV:{2 * k - 1}", washer(x, y, T_PATKA, M_PATKA, up=True), color=SROUB),
                  Part(f"ISO 7089 - {m} - 200 HV:{2 * k}", washer(x, y, -T_ZAKLAD, M_PATKA, up=False), color=SROUB),
                  Part(f"ISO 4032 - M{m} - 8:{k}", hex_nut(x, y, -T_ZAKLAD - wd["h"], M_PATKA), color=SROUB)]
    return parts


# ------------------------------------------------------------------ example 10
KONZOLA = (160.0, 12.0, 90.0)                        # design space: length x, thickness y, height z
KONZOLA_DIRY = [(15.0, 15.0), (15.0, 75.0)]          # (x, z) of the holes Ø11 for the screws M10 into the wall
D_KONZOLA_DIRA = 11.0
KONZOLA_OKO = (142.0, 45.0)                          # (x, z) of the eye Ø14 for the pin of the load
D_KONZOLA_OKO = 14.0


def konzola_navrh() -> list[Part]:
    """The design space of a wall bracket for topology optimisation: a solid plate 160×90×12 (x 0..160, y 0..12,
    z 0..90) with 2 holes Ø11 for the screws into the wall (x = 15) and an eye Ø14 for the pin of the load at the free end.
    """
    lx, t, lz = KONZOLA
    holes = [cyl((x, -1, z), (0, 1, 0), D_KONZOLA_DIRA / 2, t + 2) for x, z in KONZOLA_DIRY]
    ox, oz = KONZOLA_OKO
    eye = cyl((ox, -1, oz), (0, 1, 0), D_KONZOLA_OKO / 2, t + 2)
    return [Part("Konzola (navrhovy prostor)", solid(cut(box(0, 0, 0, lx, t, lz), *holes, eye)), color=OCEL)]
