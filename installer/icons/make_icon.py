"""Generator of the FEA Studio icon – installer/icons/fea_studio.ico and fea_studio.png.

Motif: a flat angle (plate of L shape) as the result of a structural FEM calculation.
  * colour map of the stress in concentric bands around the inner fillet – blue (indigo) at the ends of the arms,
    through azure, green, yellow and orange to red in the fillet (stress concentration),
  * triangular finite element mesh refined towards the fillet (lines = darker shade of the band colour, not black),
  * purple force arrow – the tip ends on the part (pressure on the face, as the arrows in the program),
  * hatched fixed support under the lower arm (from 48 px) – the load case makes sense: the lower arm is
    bolted to the base, the force pushes the upper end of the vertical arm, the bending peaks in the fillet,
  * 128 and 256 px: the vertical arm slightly deflected in the direction of the force and the dashed undeformed shape,
  * 256 px: several element nodes at the concentration (white dots with a dark edge).

The sizes (16, 20, 24, 32, 40, 48, 64, 128, 256) are each drawn separately in 4× resampling:
  * 16–40 px: simplified drawing aligned to pixels. Every pixel gets a single colour (the prevailing one
    in the resampled block) – no semi-transparent or mixed pixels (earlier dark wine "smears" on the
    steps of the fillet). 16–24 px only bands, the outline (almost over the whole surface) and a short arrow (tip 5–7 px),
    32–40 px additionally a few whole triangles of the mesh concentrated at the fillet (drawn by hand, lines along the
    pixels, red band without lines).
  * 48 px: mesh, one hole, arrow, fixed support; reduction BOX (exact pixel coverage, no ripple).
  * 64–256 px: full drawing, reduction LANCZOS.

The background is TRANSPARENT (not a tile): the outline of the L is unmistakable and at 16–24 px it covers the whole surface
(a tile would take 2–3 px). On a light taskbar the dark-blue outline holds the shape, on a dark one the saturated band colours
and from 48 px a thin light outer edge (also inside the holes – the hole is then visible as a ring on the dark background).

Run:  python installer/icons/make_icon.py [--out FOLDER] [--preview PREVIEW.png]
Without parameters it overwrites fea_studio.ico and fea_studio.png next to the script. The preview (light, dark and blue
strip, small sizes enlarged) is saved only on request – it does not belong to installer/.
Only PIL + numpy; deterministic (fixed seed, no external inputs – images, fonts), PNG without metadata.
"""
from __future__ import annotations

import argparse
import io
import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SS = 4                                          # resampling
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

OUTLINE = (27, 40, 68)                          # outline – dark navy blue #1B2844
ARROW = (200, 44, 196)                          # force – purple (outside the colour bands)
SUPPORT = (112, 128, 156)                       # fixed support and the dashed shape – slate, readable on light and dark
HALO = (200, 210, 224)                          # light outer edge (because of the dark panel)
HALO_ALPHA = 0.6
NODE = (255, 255, 255)                          # nodes of the elements

# Stress bands from the lowest to the highest (indigo → blue → azure → green → yellow → orange → red).
PAL7 = [(42, 63, 208), (30, 124, 240), (22, 178, 222), (70, 198, 92),
        (252, 214, 40), (250, 136, 28), (228, 38, 30)]
PAL6 = [(42, 63, 208), (24, 150, 236), (70, 198, 92), (252, 214, 40), (250, 136, 28), (228, 38, 30)]
PAL5 = [(42, 63, 208), (22, 160, 234), (80, 200, 72), (252, 210, 36), (228, 38, 30)]


# ----------------------------------------------------------------------------- design by sizes
# Coordinates in pixels of the target size (pixel i covers [i, i + 1)). X0..X1 × Y0..Y1 = envelope of the angle,
# T = thickness of the arms, Ri = inner fillet, Ro = outer corner, rc = other corners, ow = outline width.
BIG = dict(X0=24, Y0=10, X1=244, Y1=218, T=78, Ri=34, Ro=16, rc=12,
           holes=[(63, 49, 15), (205, 179, 15)],
           ow=7.0, halo=3.0, lw=2.0, shade=0.62, h=24.0, refine=0.5,
           thresholds=[0.08, 0.2, 0.34, 0.5, 0.7, 0.95], pal=PAL7,
           arrow=dict(y=49, sw=6.0, hl=44.0, hw=24.0, tail=246.0),
           ground=dict(x0=14, x1=250, line=7.0, depth=24.0, period=18.0, width=5.0),
           deflect=16.0, dash=dict(w=3.0, on=11.0, off=7.0), nodes=True)


def _scaled(size: int) -> dict:
    """Large drawing (64–256 px) = the 256 px master reduced to `size`."""
    s = size / 256.0
    g = dict(BIG)
    for k in ("X0", "Y0", "X1", "Y1", "T", "Ri", "Ro", "rc"):
        g[k] = BIG[k] * s
    g["holes"] = [(x * s, y * s, r * s) for x, y, r in BIG["holes"]]
    g["ow"] = max(2.0, BIG["ow"] * s)
    g["halo"] = max(1.0, BIG["halo"] * s)
    g["lw"] = max(1.0, BIG["lw"] * s)
    g["h"] = max(12.0, BIG["h"] * s)
    a = BIG["arrow"]
    g["arrow"] = dict(y=a["y"] * s, sw=max(1.0, a["sw"] * s), hl=a["hl"] * s, hw=a["hw"] * s, tail=a["tail"] * s)
    gr = BIG["ground"]
    g["ground"] = {k: v * s for k, v in gr.items()}
    g["deflect"] = BIG["deflect"] * s
    g["dash"] = {k: v * s for k, v in BIG["dash"].items()}
    g["resample"] = "lanczos"
    if size < 256:
        g["nodes"] = False
    if size < 128:
        # 64 px: without deformation, dashed shape and nodes; coarser mesh, the fixed support aligned to pixels
        g.update(deflect=0.0, dash=None, nodes=False, refine=0.75, shade=0.66,
                 Y1=55.0, holes=[(15.75, 12.5, 3.75), (51.25, 45.0, 3.75)],
                 ground=dict(x0=3, x1=62, line=2.0, depth=6.0, period=4, pixel=True))
        g["arrow"]["y"] = 12.5                   # the shank exactly on 3 pixel rows
    return g


def design(size: int) -> dict:
    """Drawing parameters for one size."""
    if size >= 64:
        return _scaled(size)
    if size == 48:
        return dict(X0=3, Y0=2, X1=46, Y1=40, T=15, Ri=6.5, Ro=3.5, rc=2.5,
                    holes=[(10.5, 10.0, 3.5)], ow=2.0, halo=1.0, lw=1.0, shade=0.72, h=10.5, refine=0.75,
                    thresholds=[0.09, 0.22, 0.38, 0.56, 0.78, 1.0], pal=PAL7,
                    arrow=dict(y=10.0, sw=1.0, hl=8.0, hw=5.5, tail=46.0),
                    ground=dict(x0=1, x1=47, line=2.0, depth=4.0, period=4, pixel=True),
                    resample="box")
    # 16–40 px: aligned to pixels, every pixel in one colour (crisp); mesh = hand-drawn lines along the pixels
    if size == 40:
        return dict(X0=1, Y0=1, X1=39, Y1=39, T=15, Ri=6, Ro=3.5, rc=2, holes=[], ow=1.0,
                    thresholds=[0.1, 0.24, 0.42, 0.64, 0.95], pal=PAL6,
                    arrow=dict(y=8.5, sw=1.5, hl=6.0, hw=5.5, tail=39.0),
                    mesh=MESH40, shade=0.5, crisp=True)
    if size == 32:
        return dict(X0=1, Y0=1, X1=31, Y1=31, T=12, Ri=5, Ro=3, rc=2, holes=[], ow=1.0,
                    thresholds=[0.1, 0.26, 0.46, 0.7, 1.0], pal=PAL6,
                    arrow=dict(y=7.0, sw=1.0, hl=5.0, hw=4.5, tail=31.0),
                    mesh=MESH32, shade=0.5, crisp=True)
    if size == 24:
        return dict(X0=1, Y0=1, X1=23, Y1=23, T=9, Ri=4, Ro=2, rc=1.5, holes=[], ow=1.0,
                    thresholds=[0.14, 0.4, 0.72, 1.1], pal=PAL5,
                    arrow=dict(y=5.0, sw=1.0, hl=4.0, hw=3.5, tail=23.0),
                    mesh=None, crisp=True)
    if size == 20:
        return dict(X0=0, Y0=0, X1=20, Y1=20, T=8, Ri=3.5, Ro=2, rc=1, holes=[], ow=1.0,
                    thresholds=[0.14, 0.38, 0.66, 0.95], pal=PAL5,
                    arrow=dict(y=4.0, sw=1.0, hl=3.5, hw=3.0, tail=20.0),
                    mesh=None, crisp=True)
    if size == 16:
        return dict(X0=0, Y0=0, X1=16, Y1=16, T=7, Ri=3, Ro=1.5, rc=1, holes=[], ow=1.0,
                    thresholds=[0.14, 0.36, 0.58, 0.8], pal=PAL5,
                    arrow=dict(y=3.5, sw=0.5, hl=3.5, hw=3.0, tail=16.0),     # shank 1 px, tip 5 px
                    mesh=None, crisp=True)
    raise ValueError(size)


# Hand-drawn mesh 32 and 40 px: segments between the CENTRES of the pixels (column, row). Nodes lie on the outline
# (column / row of the outline) or inside; the triangles shrink towards the inner fillet.
MESH32 = [((12, 1), (1, 12)), ((1, 12), (12, 14)),            # vertical arm – large elements
          ((1, 12), (6, 23)), ((12, 14), (6, 23)),             # triangle at the fillet
          ((6, 23), (15, 18)), ((6, 23), (18, 19)),            # small elements in the fillet (fan)
          ((6, 23), (1, 30)), ((6, 23), (12, 30)), ((18, 19), (12, 30)),
          ((18, 19), (24, 30)), ((24, 30), (30, 19))]          # horizontal arm – large elements
MESH40 = [((15, 1), (1, 15)), ((1, 15), (15, 18)),
          ((1, 15), (8, 29)), ((15, 18), (8, 29)),
          ((8, 29), (18, 22)), ((8, 29), (22, 24)),
          ((8, 29), (1, 38)), ((8, 29), (15, 38)), ((22, 24), (15, 38)),
          ((22, 24), (29, 38)), ((29, 38), (38, 24))]


# ----------------------------------------------------------------------------- geometrie
def _unit(v):
    v = np.asarray(v, float)
    return v / np.hypot(*v)


def _seg_dist(X, Y, a, b):
    """Distance of the points from the segment a–b (vector form)."""
    d = b - a
    L2 = float(d @ d)
    if L2 == 0:
        return np.hypot(X - a[0], Y - a[1])
    t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / L2, 0.0, 1.0)
    return np.hypot(X - a[0] - t * d[0], Y - a[1] - t * d[1])


class Shape:
    """Angle: a right-angled polygon with rounded corners (convex and concave) and circular holes.

    Everything analytically – membership of a point in the shape and the distance from the boundary (outline, edge, mesh) without raster
    substitutes, so that the edges at small sizes lie exactly on the pixel grid.
    """

    def __init__(self, g: dict):
        X0, Y0, X1, Y1, T = g["X0"], g["Y0"], g["X1"], g["Y1"], g["T"]
        self.v = np.array([(X0, Y0), (X0 + T, Y0), (X0 + T, Y1 - T), (X1, Y1 - T), (X1, Y1), (X0, Y1)], float)
        radii = [g["rc"], g["rc"], g["Ri"], g["rc"], g["rc"], g["Ro"]]
        n = len(self.v)
        self.corners = []                        # (vertex, direction to the previous, direction to the next, r, centre of the arc)
        for i in range(n):
            v = self.v[i]
            dp = _unit(self.v[i - 1] - v)
            dn = _unit(self.v[(i + 1) % n] - v)
            r = float(radii[i])
            self.corners.append((v, dp, dn, r, v + r * (dp + dn)))
        self.segments = []                       # straight parts between the tangent points of the arcs
        for i in range(n):
            v, _, dn, r, _ = self.corners[i]
            w, dp2, _, r2, _ = self.corners[(i + 1) % n]
            self.segments.append((v + r * dn, w + r2 * dp2))
        self.holes = [tuple(map(float, h)) for h in g.get("holes", ())]

    def inside(self, X, Y):
        """A point inside the material (without holes)."""
        sharp = np.zeros(X.shape, bool)
        n = len(self.v)
        for i in range(n):                       # ray to the right × vertical edges (even / odd number)
            (xa, ya), (xb, yb) = self.v[i], self.v[(i + 1) % n]
            if xa == xb:
                sharp ^= (X < xa) & (Y >= min(ya, yb)) & (Y < max(ya, yb))
        flip = np.zeros(X.shape, bool)
        for v, dp, dn, r, c in self.corners:     # fillet: a convex corner removes material, a concave one adds it
            if r <= 0:
                continue
            a = (X - v[0]) * dp[0] + (Y - v[1]) * dp[1]
            b = (X - v[0]) * dn[0] + (Y - v[1]) * dn[1]
            flip |= (a >= 0) & (a < r) & (b >= 0) & (b < r) & (np.hypot(X - c[0], Y - c[1]) > r)
        inside = sharp ^ flip
        for hx, hy, hr in self.holes:
            inside &= np.hypot(X - hx, Y - hy) > hr
        return inside

    def distance(self, X, Y):
        """Distance from the boundary (outline and holes), unsigned."""
        d = np.full(X.shape, np.inf)
        for a, b in self.segments:
            d = np.minimum(d, _seg_dist(X, Y, a, b))
        for v, dp, dn, r, c in self.corners:
            if r <= 0:
                continue
            ux, uy = X - c[0], Y - c[1]
            span = (-(ux * dn[0] + uy * dn[1]) >= 0) & (-(ux * dp[0] + uy * dp[1]) >= 0)
            d = np.where(span, np.minimum(d, np.abs(np.hypot(ux, uy) - r)), d)
        for hx, hy, hr in self.holes:
            d = np.minimum(d, np.abs(np.hypot(X - hx, Y - hy) - hr))
        return d

    def outline(self, step=0.5):
        """Outline as a dense closed polyline."""
        pts = []
        for i, (v, dp, dn, r, c) in enumerate(self.corners):
            if r > 0:
                t0, t1 = v + r * dp - c, v + r * dn - c
                a0, a1 = math.atan2(t0[1], t0[0]), math.atan2(t1[1], t1[0])
                da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
                m = max(2, int(abs(da) * r / step))
                pts += [(c[0] + r * math.cos(a0 + da * k / m), c[1] + r * math.sin(a0 + da * k / m))
                        for k in range(m + 1)]
            else:
                pts.append(tuple(v))
            a, b = self.segments[i]
            L = float(np.hypot(*(b - a)))
            m = max(1, int(L / step))
            pts += [tuple(a + (b - a) * k / m) for k in range(1, m)]
        return np.asarray(pts, float)


def fillet_distance(X, Y, g):
    """Distance from the inner fillet (the place of the stress concentration) in multiples of the arm thickness."""
    cx = g["X0"] + g["T"] + g["Ri"]
    cy = g["Y1"] - g["T"] - g["Ri"]
    r = np.hypot(X - cx, Y - cy) - g["Ri"] - g["ow"]
    return np.maximum(r, 0.0) / g["T"]


def deflection(y, g):
    """Deflection of the vertical arm to the left (in the direction of the force): mainly the rotation in the fillet (the edges stay
    almost straight and the shape clean) + a little of the deflection line of the cantilever u²(3 − u)/2.
    """
    y = np.asarray(y, float)
    d = g.get("deflect") or 0.0
    if not d:
        return np.zeros_like(y)
    yb = g["Y1"] - g["T"]
    u = np.clip((yb - y) / (yb - g["Y0"]), 0.0, 1.0)
    return d * (0.65 * u + 0.35 * u * u * (3.0 - u) / 2.0)


# ----------------------------------------------------------------------------- finite element mesh
def delaunay(P):
    """Delaunay triangulation (Bowyer–Watson) – for ~300 points pure Python is enough, deterministic."""
    P = [tuple(map(float, p)) for p in P]
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    D = 20 * max(max(xs) - min(xs), max(ys) - min(ys))
    n = len(P)
    pts = P + [(mx - D, my - D), (mx + D, my - D), (mx, my + D)]

    def circ(i, j, k):
        (ax, ay), (bx, by), (cx, cy) = pts[i], pts[j], pts[k]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        return ux, uy, (ax - ux) ** 2 + (ay - uy) ** 2

    tris = [((n, n + 1, n + 2), circ(n, n + 1, n + 2))]
    for i in range(n):
        x, y = pts[i]
        bad, keep = [], []
        for t in tris:
            ux, uy, r2 = t[1]
            (bad if (x - ux) ** 2 + (y - uy) ** 2 < r2 else keep).append(t)
        count = {}
        for (a, b, c), _ in bad:
            for e in ((a, b), (b, c), (c, a)):
                k = (min(e), max(e))
                count[k] = count.get(k, 0) + 1
        for (a, b), cnt in sorted(count.items()):
            if cnt == 1:
                keep.append(((a, b, i), circ(a, b, i)))
        tris = keep
    return P, [t for t, _ in tris if max(t) < n]


def fem_mesh(shape: Shape, g: dict):
    """Unstructured triangle mesh refined towards the fillet. Returns the points and the triangles (indices)."""

    def hloc(x, y):
        r = float(fillet_distance(np.array(x), np.array(y), g))
        return g["h"] * min(1.0, g["refine"] + (1 - g["refine"]) * r / 0.8)

    pts = []
    ring = shape.outline(0.25)
    acc = 0.0
    for j, p in enumerate(ring):                 # points on the outline – step by the local element size
        if j == 0:
            pts.append(tuple(p))
            continue
        acc += float(np.hypot(*(p - ring[j - 1])))
        if acc >= hloc(*p):
            pts.append(tuple(p))
            acc = 0.0
    if len(pts) > 2 and math.dist(pts[-1], pts[0]) < 0.5 * hloc(*pts[0]):
        pts.pop()
    for hx, hy, hr in shape.holes:               # body na otvorech
        m = max(6, int(round(2 * math.pi * hr / (0.85 * hloc(hx, hy)))))
        pts += [(hx + hr * math.cos(2 * math.pi * (k + 0.25) / m), hy + hr * math.sin(2 * math.pi * (k + 0.25) / m))
                for k in range(m)]
    # interior points – greedy Poisson sampling with a fixed seed
    rng = np.random.default_rng(11)
    step = g["h"] / 5
    gx, gy = np.meshgrid(np.arange(g["X0"], g["X1"], step), np.arange(g["Y0"], g["Y1"], step))
    cand = np.column_stack([gx.ravel(), gy.ravel()])
    cand = cand + rng.uniform(-0.5, 0.5, cand.shape) * step
    cand = cand[rng.permutation(len(cand))]
    ins = shape.inside(cand[:, 0], cand[:, 1])
    dist = shape.distance(cand[:, 0], cand[:, 1])
    acc_pts = np.asarray(pts, float)
    for (x, y), ok, d in zip(cand, ins, dist):
        h = hloc(x, y)
        if not ok or d < 0.62 * h:
            continue
        if np.min(np.hypot(acc_pts[:, 0] - x, acc_pts[:, 1] - y)) < 0.9 * h:
            continue
        acc_pts = np.vstack([acc_pts, [x, y]])
    P, tris = delaunay(acc_pts)
    P = np.asarray(P, float)
    keep = []
    for t in tris:                               # only triangles inside the material (not across the fillets and holes)
        q = P[list(t)]
        test = np.vstack([q.mean(0), (q[0] + q[1]) / 2, (q[1] + q[2]) / 2, (q[0] + q[2]) / 2])
        if shape.inside(test[:, 0], test[:, 1]).all():
            keep.append(tuple(sorted(t)))
    return P, sorted(keep)


def mesh_edges(tris):
    edges = set()
    for a, b, c in tris:
        edges |= {(a, b), (b, c), (a, c)}
    return sorted(edges)


def deform_points(P, g):
    P = np.asarray(P, float).copy()
    P[:, 0] -= deflection(P[:, 1], g)
    return P


# ----------------------------------------------------------------------------- raster helpers
def grid(size):
    """Centres of the resampled pixels in the coordinates of the target size."""
    c = (np.arange(size * SS) + 0.5) / SS
    return np.meshgrid(c, c)


def seg_mask(n, segs, width):
    """Segments of the given width (target pixels) into the mask of the resampled raster n × n."""
    out = np.zeros((n, n), bool)
    hw = width / 2
    for a, b in segs:
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        lo = np.floor((np.minimum(a, b) - hw) * SS).astype(int).clip(0, n)
        hi = np.ceil((np.maximum(a, b) + hw) * SS).astype(int).clip(0, n)
        if lo[0] >= hi[0] or lo[1] >= hi[1]:
            continue
        X, Y = np.meshgrid((np.arange(lo[0], hi[0]) + 0.5) / SS, (np.arange(lo[1], hi[1]) + 0.5) / SS)
        out[lo[1]:hi[1], lo[0]:hi[0]] |= _seg_dist(X, Y, a, b) <= hw + 1e-9
    return out


def dash_mask(n, poly, width, on, off):
    """Dashed polyline (phase of the dashes by the arc length)."""
    out = np.zeros((n, n), bool)
    hw = width / 2
    s0 = 0.0
    period = on + off
    for a, b in zip(poly[:-1], poly[1:]):
        d = b - a
        L = float(np.hypot(*d))
        if L == 0:
            continue
        lo = np.floor((np.minimum(a, b) - hw) * SS).astype(int).clip(0, n)
        hi = np.ceil((np.maximum(a, b) + hw) * SS).astype(int).clip(0, n)
        if lo[0] < hi[0] and lo[1] < hi[1]:
            X, Y = np.meshgrid((np.arange(lo[0], hi[0]) + 0.5) / SS, (np.arange(lo[1], hi[1]) + 0.5) / SS)
            t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / (L * L), 0.0, 1.0)
            near = np.hypot(X - a[0] - t * d[0], Y - a[1] - t * d[1]) <= hw
            out[lo[1]:hi[1], lo[0]:hi[0]] |= near & (((s0 + t * L) % period) < on)
        s0 += L
    return out


def arrow_mask(X, Y, g, tip_x):
    """Force arrow to the left: triangular tip (the tip on the part) + shank."""
    a = g["arrow"]
    y0, sw, hl, hw, tail = a["y"], a["sw"], a["hl"], a["hw"], a["tail"]
    xb = tip_x + hl                               # base of the tip
    head = (X >= tip_x) & (X <= xb) & (np.abs(Y - y0) <= hw * (X - tip_x) / hl)
    shaft = (X >= xb - 0.5) & (X <= tail) & (np.abs(Y - y0) <= sw)
    return head | shaft


def ground_mask(X, Y, g):
    """Fixed support under the lower arm: a solid line and hatching "/"."""
    gr = g["ground"]
    y0 = g["Y1"]
    xin = (X >= gr["x0"]) & (X <= gr["x1"])
    line = xin & (Y >= y0) & (Y < y0 + gr["line"])
    hy = (Y >= y0 + gr["line"]) & (Y < y0 + gr["line"] + gr["depth"])
    if gr.get("pixel"):                           # 48–64 px: hatching by whole pixels (otherwise grey noise)
        hatch = xin & hy & (((np.floor(X) + np.floor(Y)) % gr["period"]) == 0)
    else:
        hatch = xin & hy & (((X + Y) % gr["period"]) < gr["width"] * math.sqrt(2))
    return line | hatch


def over(rgb, alpha, mask, color):
    """Overlay of a colour (alpha compositing "over", straight alpha) – mask = coverage 0..1, color = RGB or an array n × n × 3."""
    m = mask.astype(float)
    a_out = m + alpha * (1 - m)
    col = np.broadcast_to(np.asarray(color, float), rgb.shape)
    num = col * m[..., None] + rgb * (alpha * (1 - m))[..., None]
    rgb = np.where(a_out[..., None] > 0, num / np.maximum(a_out, 1e-12)[..., None], 0.0)
    return rgb, a_out


def downscale(rgba, size, how):
    im = Image.fromarray(rgba, "RGBA").convert("RGBa")          # shrinks in premultiplied alpha
    flt = Image.LANCZOS if how == "lanczos" else Image.BOX
    return im.resize((size, size), flt).convert("RGBA")


# ----------------------------------------------------------------------------- kresba
def render_smooth(size: int) -> Image.Image:
    """48–256 px: drawing at 4× supersampling and reduction."""
    g = design(size)
    shape = Shape(g)
    X, Y = grid(size)
    n = size * SS
    Xo = X + deflection(Y, g)                    # deformed shape = back mapping to the original coordinates
    inside = shape.inside(Xo, Y)
    dist = shape.distance(Xo, Y)

    rgb = np.zeros((n, n, 3))
    alpha = np.zeros((n, n))

    # light outer edge (also inside the holes) – because of the dark main panel
    halo = ~inside & (dist <= g["halo"])
    rgb, alpha = over(rgb, alpha, halo.astype(float) * HALO_ALPHA, HALO)

    # dashed undeformed shape – only where it is clearly outside the deformed part
    if g.get("dash") and g.get("deflect"):
        dsh = g["dash"]
        poly = shape.outline(0.5)
        poly = np.vstack([poly, poly[:1]])
        dm = dash_mask(n, poly, dsh["w"], dsh["on"], dsh["off"])
        dm &= ~inside & (dist > g["halo"] + 0.5 * dsh["w"] + 0.5)
        rgb, alpha = over(rgb, alpha, dm, SUPPORT)

    # fixed support
    if g.get("ground"):
        gm = ground_mask(X, Y, g) & ~inside
        rgb, alpha = over(rgb, alpha, gm, SUPPORT)

    # coloured stress bands (in the original coordinates – the bands move with the material)
    r = fillet_distance(Xo, Y, g)
    pal = np.asarray(g["pal"], float)
    k = len(pal) - 1 - np.searchsorted(np.asarray(g["thresholds"]), r)
    part = pal[np.clip(k, 0, len(pal) - 1)]

    # finite element mesh – darker shade of the band colour
    P, tris = fem_mesh(shape, g)
    Pd = deform_points(P, g)
    segs = []
    for i, j in mesh_edges(tris):
        a, b = P[i], P[j]
        m = 4 if g.get("deflect") else 1           # the deformation is nonlinear → segments piece by piece
        q = deform_points([a + (b - a) * t / m for t in range(m + 1)], g)
        segs += list(zip(q[:-1], q[1:]))
    lines = seg_mask(n, segs, g["lw"])
    part = np.where(lines[..., None], part * g["shade"], part)

    # outline (inner stroke – the edge of the shape stays on the pixel grid)
    part = np.where((dist <= g["ow"])[..., None], np.asarray(OUTLINE, float), part)
    rgb, alpha = over(rgb, alpha, inside, part)

    # nodes of the elements at the concentration (256 px only)
    if g.get("nodes"):
        for x, y in node_points(P, Pd, tris, g):
            d = np.hypot(X - x, Y - y)
            rgb, alpha = over(rgb, alpha, d <= 5.6, OUTLINE)
            rgb, alpha = over(rgb, alpha, d <= 3.6, NODE)

    # force arrow – the tip on the deformed edge of the vertical arm
    if g.get("arrow"):
        tip = g["X0"] + g["T"] - float(deflection(g["arrow"]["y"], g))
        rgb, alpha = over(rgb, alpha, arrow_mask(X, Y, g, tip), ARROW)

    rgba = np.dstack([rgb, alpha * 255.0]).round().clip(0, 255).astype(np.uint8)
    return downscale(rgba, size, g["resample"])


def node_points(P, Pd, tris, g):
    """Nodes of two elements nearest to the centre of the inner fillet (the deformed positions)."""
    cx = g["X0"] + g["T"] + g["Ri"]
    cy = g["Y1"] - g["T"] - g["Ri"]
    mid = np.array([cx - g["Ri"] * math.sqrt(0.5), cy + g["Ri"] * math.sqrt(0.5)])
    mid = mid + np.array([-1, 1]) * math.sqrt(0.5) * 0.22 * g["T"]  # a little into the material (red / orange band)
    cent = np.array([P[list(t)].mean(0) for t in tris])
    order = np.argsort(np.hypot(cent[:, 0] - mid[0], cent[:, 1] - mid[1]))
    idx = sorted({i for t in order[:2] for i in tris[t]})
    return [tuple(Pd[i]) for i in idx]


def render_crisp(size: int) -> Image.Image:
    """16–40 px: every pixel with a single colour – the prevailing label in the block SS × SS (no mixed colours)."""
    g = design(size)
    shape = Shape(g)
    X, Y = grid(size)
    inside = shape.inside(X, Y)
    dist = shape.distance(X, Y)
    r = fillet_distance(X, Y, g)
    npal = len(g["pal"])
    k = npal - 1 - np.searchsorted(np.asarray(g["thresholds"]), r)
    lab = np.zeros(X.shape, int)                                 # 0 = background
    lab[inside] = 10 + np.clip(k, 0, npal - 1)[inside]           # 10 + k = band k
    lab[inside & (dist <= g["ow"])] = 1                          # 1 = obrys
    if g.get("arrow"):
        lab[arrow_mask(X, Y, g, g["X0"] + g["T"])] = 2           # 2 = arrow
    # the prevailing label; on a tie the order decides (arrow, outline, bands from red, background)
    order = [2, 1] + [10 + i for i in range(npal - 1, -1, -1)] + [0]
    blocks = lab.reshape(size, SS, size, SS).transpose(0, 2, 1, 3).reshape(size, size, SS * SS)
    counts = np.stack([(blocks == L).sum(-1) for L in order], -1)
    small = np.asarray(order)[counts.argmax(-1)]

    out = np.zeros((size, size, 4), np.uint8)
    for i, col in enumerate(g["pal"]):
        out[small == 10 + i] = (*col, 255)
    out[small == 1] = (*OUTLINE, 255)
    out[small == 2] = (*ARROW, 255)
    # mesh: lines along the pixels (only over the bands, the outline and the arrow stay)
    for a, b in g.get("mesh") or ():
        for x, y in pixel_line(a, b):
            # the highest (red) band stays clean – dark lines at the outline would only be a "smear"
            if 0 <= x < size and 0 <= y < size and 10 <= small[y, x] < 9 + npal:
                col = np.asarray(g["pal"][small[y, x] - 10], float) * g["shade"]
                out[y, x] = (*np.round(col).astype(np.uint8), 255)
    return Image.fromarray(out, "RGBA")


def pixel_line(a, b):
    """Pixels of the segment between the pixel centres a, b (DDA, 8-neighbourhood)."""
    (x0, y0), (x1, y1) = a, b
    m = max(abs(x1 - x0), abs(y1 - y0))
    if m == 0:
        return [(x0, y0)]
    return [(int(math.floor(x0 + (x1 - x0) * t / m + 0.5)), int(math.floor(y0 + (y1 - y0) * t / m + 0.5)))
            for t in range(m + 1)]


def render(size: int) -> Image.Image:
    return render_crisp(size) if design(size).get("crisp") else render_smooth(size)


# ----------------------------------------------------------------------------- outputs
def clean(im: Image.Image) -> Image.Image:
    """New RGBA image without metadata (info, ICC, text)."""
    return Image.frombytes("RGBA", im.size, im.convert("RGBA").tobytes())


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    clean(im).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def write_ico(path: Path, icons: dict) -> None:
    """ICO: sizes below 256 as a 32-bit BMP (+ AND mask), 256 as PNG – the widest compatibility
    (Explorer, Inno Setup, Qt, older tools). Without metadata, deterministic.
    """
    entries = []
    for s in sorted(icons):
        im = icons[s]
        if s >= 256:
            data = png_bytes(im)
        else:
            a = np.asarray(im.convert("RGBA"))
            bgra = a[::-1, :, [2, 1, 0, 3]].tobytes()          # rows from the bottom up
            row = ((s + 31) // 32) * 4
            mask = bytearray()
            for y in range(s - 1, -1, -1):
                bits = np.packbits(a[y, :, 3] == 0).tobytes()  # 1 = transparent
                mask += bits + bytes(row - len(bits))
            hdr = struct.pack("<IiiHHIIiiII", 40, s, 2 * s, 1, 32, 0, len(bgra) + len(mask), 0, 0, 0, 0)
            data = hdr + bgra + bytes(mask)
        entries.append((s, data))
    out = struct.pack("<HHH", 0, 1, len(entries))
    off = 6 + 16 * len(entries)
    body = b""
    for s, data in entries:
        out += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), off)
        off += len(data)
        body += data
    path.write_bytes(out + body)


def preview(icons: dict) -> Image.Image:
    """Preview: a light, a dark and a blue (Windows accent) strip – sizes 1:1 and small enlarged ones."""
    sizes = sorted(icons, reverse=True)
    gap = 20
    w = sum(sizes) + gap * (len(sizes) + 1)
    zoom = [(48, 4), (40, 4), (32, 6), (24, 8), (20, 8), (16, 10)]
    zw = sum(s * z for s, z in zoom) + gap * (len(zoom) + 1)
    W = max(w, zw)
    h1 = max(sizes) + 2 * gap
    h2 = max(s * z for s, z in zoom) + gap
    bgs = ((243, 243, 243), (32, 32, 32), (0, 103, 192))
    img = Image.new("RGBA", (W, len(bgs) * (h1 + h2)), (0, 0, 0, 255))
    y = 0
    for bg in bgs:
        strip = Image.new("RGBA", (W, h1 + h2), bg + (255,))
        x = gap
        for s in sizes:
            strip.alpha_composite(icons[s], (x, gap + (max(sizes) - s) // 2))
            x += s + gap
        x = gap
        for s, z in zoom:
            strip.alpha_composite(icons[s].resize((s * z, s * z), Image.NEAREST), (x, h1))
            x += s * z + gap
        img.alpha_composite(strip, (0, y))
        y += h1 + h2
    return clean(img)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Vytvoří ikonu FEA Studia (fea_studio.ico, fea_studio.png).")
    ap.add_argument("--out", type=Path, default=HERE, help="cílová složka (výchozí = složka skriptu)")
    ap.add_argument("--preview", type=Path, default=None, help="uložit i náhled všech velikostí (PNG)")
    args = ap.parse_args(argv)
    icons = {s: clean(render(s)) for s in SIZES}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fea_studio.png").write_bytes(png_bytes(icons[256]))
    write_ico(args.out / "fea_studio.ico", icons)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        args.preview.write_bytes(png_bytes(preview(icons)))
    print("hotovo:", args.out / "fea_studio.ico", args.out / "fea_studio.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
