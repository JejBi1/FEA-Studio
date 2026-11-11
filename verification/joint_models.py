"""Test assemblies for bolts and welds (OCP, part names as from the CAD)."""
from __future__ import annotations

import math

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Pnt, gp_Vec

from fea_core.geometry import Part
from fea_core.joints import tables
from verification.cases import box, cyl, solid


def prism(points, height: float, direction=(0, 0, 1)):
    """Prism from a polygon (points in a plane) extruded by height in the direction direction."""
    poly = BRepBuilderAPI_MakePolygon()
    for p in points:
        poly.Add(gp_Pnt(*map(float, p)))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    d = np.asarray(direction, float) * height
    return BRepPrimAPI_MakePrism(face, gp_Vec(*d)).Shape()


def hexagon(center, s: float, z: float):
    """Hexagon with a key opening s (distance between opposite sides) in the plane z."""
    r = s / math.sqrt(3)
    return [(center[0] + r * math.cos(math.radians(30 + 60 * k)), center[1] + r * math.sin(math.radians(30 + 60 * k)), z)
            for k in range(6)]


def hex_bolt(x, y, z_bear, d, L, down=True):
    """Bolt with a hexagon head: the bearing face of the head at z_bear, the shank downwards (down) of length L."""
    h = tables.head_dims(d, "hex")
    sg = -1 if down else 1
    head = prism(hexagon((x, y), h["s"], z_bear), h["k"], (0, 0, -sg))
    shank = cyl((x, y, z_bear), (0, 0, sg), d / 2, L)
    return solid(BRepAlgoAPI_Fuse(head, shank).Shape())


def hex_nut(x, y, z_bear, d, up=False):
    """Nut ISO 4032: bearing face at z_bear, the body downwards (up=False)."""
    n = tables.nut_dims(d)
    sg = 1 if up else -1
    body = prism(hexagon((x, y), n["s"], z_bear), n["m"], (0, 0, sg))
    z0 = z_bear if up else z_bear - n["m"]
    return solid(BRepAlgoAPI_Cut(body, cyl((x, y, z0 - 1), (0, 0, 1), d / 2, n["m"] + 2)).Shape())


def washer(x, y, z0, d, up=True):
    w = tables.washer_dims(d)
    zz = z0 if up else z0 - w["h"]
    return solid(BRepAlgoAPI_Cut(cyl((x, y, zz), (0, 0, 1), w["d2"] / 2, w["h"]),
                                 cyl((x, y, zz - 1), (0, 0, 1), w["d1"] / 2, w["h"] + 2)).Shape())


def plate_with_holes(x0, y0, z0, dx, dy, t, holes, d_hole):
    s = box(x0, y0, z0, dx, dy, t)
    for (hx, hy) in holes:
        s = BRepAlgoAPI_Cut(s, cyl((hx, hy, z0 - 1), (0, 0, 1), d_hole / 2, t + 2)).Shape()
    return solid(s)


def lap_joint(d=10.0, t=10.0, strength="8.8", nbolts=1, washers=True, pitch=40.0, gap=0.0):
    """Lap joint: plate A (z 0..t) on top, plate B (z −t−gap..−gap) at the bottom, bolts in the axis y=20.

    Overlap x ∈ [60, 60 + pitch·n], bolts at x = 80 + pitch·k; plate A starts at x = 0, B ends 60 beyond the overlap.
    """
    dh = tables.CLEARANCE[tables.size_key(d)][1]
    xs = [80.0 + pitch * k for k in range(nbolts)]
    W = 40.0
    holes = [(x, W / 2) for x in xs]
    x1 = 60.0 + pitch * nbolts
    A = plate_with_holes(0, 0, 0, x1, W, t, holes, dh)
    B = plate_with_holes(60, 0, -t - gap, x1, W, t, holes, dh)
    parts = [Part("Deska A", A), Part("Deska B", B)]
    wd = tables.washer_dims(d)
    wh = wd["h"] if washers else 0.0
    nut = tables.nut_dims(d)
    L = math.ceil((2 * t + gap + 2 * wh + nut["m"] + 3) / 5) * 5
    for k, (x, y) in enumerate(holes, 1):
        sfx = f":{k}" if nbolts > 1 else ""
        z_head = t + wh
        parts.append(Part(f"ISO 4017 - M{d:g} x {L:g} - {strength}{sfx}", hex_bolt(x, y, z_head, d, L)))
        if washers:
            parts.append(Part(f"ISO 7089 - {d:g} - 200 HV{sfx}", washer(x, y, t, d, up=True)))
            parts.append(Part(f"ISO 7089 - {d:g} - 200 HV{sfx}", washer(x, y, -t - gap, d, up=False)))
        parts.append(Part(f"ISO 4032 - M{d:g} - 8{sfx}", hex_nut(x, y, -t - gap - wh, d)))
    return parts


def tapped_joint(d=10.0, t=10.0, strength="10.9"):
    """Plate A (z 0..t) screwed by ISO 4762 into block B with a thread (z −30..0), hole in B Ø d (thread)."""
    dh = tables.CLEARANCE[tables.size_key(d)][1]
    A = plate_with_holes(0, 0, 0, 60, 40, t, [(30, 20)], dh)
    B = solid(BRepAlgoAPI_Cut(box(0, 0, -30, 60, 40, 30), cyl((30, 20, -2.0 * d), (0, 0, 1), d / 2, 2.0 * d + 1)).Shape())
    sh = tables.head_dims(d, "socket")
    L = 30.0
    head = cyl((30, 20, t), (0, 0, 1), sh["dk"] / 2, sh["k"])
    shank = cyl((30, 20, t - L), (0, 0, 1), d / 2, L)
    bolt = solid(BRepAlgoAPI_Fuse(head, shank).Shape())
    return [Part("Deska", A), Part("Blok", B), Part(f"ISO 4762 M{d:g}x{L:g} {strength}", bolt)]


def countersunk_joint(d=5.0, t=6.0, gap=0.3):
    """Plate A with a 90° countersink (z 0..t) + plate B (z −t−gap..−gap) with clearance, ISO 10642 M5 + nut ISO 4032."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt as P
    dk = 2.0 * d                                          # head diameter ≈ 2d, angle 90° → depth (dk − d)/2
    k = 0.5 * (dk - d)
    dh = tables.CLEARANCE[tables.size_key(d)][1]
    cone_hole = BRepPrimAPI_MakeCone(gp_Ax2(P(30, 20, t - k), gp_Dir(0, 0, 1)), dh / 2, dk / 2 + 0.05, k + 0.05).Shape()
    A = BRepAlgoAPI_Cut(box(0, 0, 0, 60, 40, t), cyl((30, 20, -1), (0, 0, 1), dh / 2, t + 2)).Shape()
    A = solid(BRepAlgoAPI_Cut(A, cone_hole).Shape())
    B = plate_with_holes(0, 0, -t - gap, 60, 40, t, [(30, 20)], dh)
    head = BRepPrimAPI_MakeCone(gp_Ax2(P(30, 20, t - k), gp_Dir(0, 0, 1)), d / 2, dk / 2, k).Shape()
    L = 16.0
    bolt = solid(BRepAlgoAPI_Fuse(head, cyl((30, 20, t - L), (0, 0, 1), d / 2, L - k + 0.01)).Shape())
    nut = hex_nut(30, 20, -t - gap, d)
    return [Part("Deska A", A), Part("Deska B", B), Part(f"Sroub ISO 10642 M{d:g}x{L:g}", bolt),
            Part(f"Matice ISO 4032 M{d:g}", nut)]


def plate_attachment(a=4.0, t=10.0, L=200.0, b=40.0, t_stub=10.0, h_stub=40.0, x_stub=100.0):
    """Sheet (x 0..L, y 0..b, z −t..0) with a non-load-bearing stub (x_stub ± t_stub/2, z 0..h_stub) welded
    by two fillet welds along y (leg a·√2) – the weld toe on the upper surface of the sheet.
    """
    z = a * math.sqrt(2)
    x1, x2 = x_stub - t_stub / 2, x_stub + t_stub / 2
    plate = box(0, 0, -t, L, b, t)
    stub = box(x1, 0, 0, t_stub, b, h_stub)
    w1 = prism([(x2, 0, 0), (x2 + z, 0, 0), (x2, 0, z)], b, (0, 1, 0))
    w2 = prism([(x1, 0, 0), (x1, 0, z), (x1 - z, 0, 0)], b, (0, 1, 0))
    return [Part("Plech", solid(plate)), Part("Nadstavec", solid(stub)), Part(f"Svar a{a:g} vpravo", solid(w1)),
            Part(f"Svar a{a:g} vlevo", solid(w2))]


def t_joint_fillet(a=5.0, t_flange=10.0, t_web=10.0, L=100.0, h_web=100.0):
    """T-joint: flange (z −t..0, x ∈ [−50, 50]) + web (x ∈ [−t_web/2, t_web/2], z 0..h_web),
    two fillet welds along y of length L with leg z = a·√2 (equal legs, 90°).
    """
    z = a * math.sqrt(2)
    fl = box(-50, 0, -t_flange, 100, L, t_flange)
    web = box(-t_web / 2, 0, 0, t_web, L, h_web)
    w1 = prism([(t_web / 2, 0, 0), (t_web / 2 + z, 0, 0), (t_web / 2, 0, z)], L, (0, 1, 0))
    w2 = prism([(-t_web / 2, 0, 0), (-t_web / 2, 0, z), (-t_web / 2 - z, 0, 0)], L, (0, 1, 0))
    return [Part("Pasnice", solid(fl)), Part("Stojina", solid(web)), Part(f"Svar a{a:g} vpravo", solid(w1)),
            Part(f"Svar a{a:g} vlevo", solid(w2))]
