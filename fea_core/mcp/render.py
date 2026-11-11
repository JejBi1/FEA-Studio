"""Headless render of a scene into PNG (without Qt, without OpenGL): rasterisation numba + labels of matplotlib (Agg).

Scene = surface linear triangles (for tet10 4 sub-triangles per face of tri6), either with a nodal
field (colour bands as in the GUI, fea_core.contour.BAND_COLORS) or with colours of the triangles (a model
with the chosen faces). Edges: outline (silhouette), sharp edges (angle of normals > feature_angle) and boundaries
of groups (parts / CAD faces). Labels, legend, load arrows, the axis cross and the marks max/min are drawn
through matplotlib in pixel coordinates – the module must not import pyplot (choice of the backend → Qt).
"""
from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..contour import band_colors
from .raster import raster_triangles, raster_lines, downsample, to_rgb8  # noqa: F401 – downsample re-exported

# views: direction towards the camera, direction up (axis Z up as in the CAD)
VIEWS = {
    "iso": ((1, -1, 1), (0, 0, 1)),
    "iso_back": ((-1, 1, 1), (0, 0, 1)),
    "iso_left": ((-1, -1, 1), (0, 0, 1)),
    "iso_bottom": ((1, -1, -1), (0, 0, 1)),
    "front": ((0, -1, 0), (0, 0, 1)),
    "back": ((0, 1, 0), (0, 0, 1)),
    "right": ((1, 0, 0), (0, 0, 1)),
    "left": ((-1, 0, 0), (0, 0, 1)),
    "top": ((0, 0, 1), (0, 1, 0)),
    "bottom": ((0, 0, -1), (0, -1, 0)),
}
VIEW_LABELS = {
    "iso": "isometric (from +X −Y +Z)", "iso_back": "isometric from behind (from −X +Y +Z)",
    "iso_left": "isometric from left (from −X −Y +Z)", "iso_bottom": "isometric from below (from +X −Y −Z)",
    "front": "front (from −Y)", "back": "back (from +Y)", "right": "right (from +X)", "left": "left (from −X)",
    "top": "top (from +Z)", "bottom": "bottom (from −Z)",
}


def camera(view) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """(right, up, towards the camera, description). view = name from VIEWS or a vector of the direction towards the camera [x, y, z]."""
    if isinstance(view, str):
        key = view.strip().lower()
        if key not in VIEWS:
            raise ValueError(f"Unknown view '{view}' – use {', '.join(VIEWS)} or a vector [x, y, z]")
        d, up = VIEWS[key]
        label = VIEW_LABELS[key]
    else:
        d = [float(v) for v in view]
        if len(d) != 3 or np.linalg.norm(d) == 0:
            raise ValueError("A view given as a vector must have 3 components, not all zero (direction to the camera)")
        up = (0, 0, 1) if abs(d[2]) < 0.95 * np.linalg.norm(d) else (0, 1, 0)
        label = f"direction to the camera ({', '.join((f'{v:g}' for v in d))})"
    d = np.asarray(d, float)
    d /= np.linalg.norm(d)
    u = np.asarray(up, float)
    u = u - (u @ d) * d
    u /= np.linalg.norm(u)
    r = np.cross(u, d)
    return r, u, d, label


def fmt(v: float, digits: int = 4) -> str:
    """Compact number for the labels (4 valid digits, the exponent only for the extremes)."""
    if v is None or not np.isfinite(v):
        return "∞" if v is not None and v > 0 else "–"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e6):
        return f"{v:.{digits - 1}e}"
    return f"{v:.{digits}g}"


@dataclass
class Scene:
    V: np.ndarray                         # (N, 3) displayed positions of the nodes [mm]
    T: np.ndarray                         # (n, 3) linear triangles
    values: np.ndarray | None = None      # (N,) nodal field → bands
    tri_colors: np.ndarray | None = None  # (n, 3) 0..1 (without the field)
    tri_group: np.ndarray | None = None   # (n,) part / face – boundaries of the groups = edges
    ghost: np.ndarray | None = None       # (N, 3) undeformed shape (outline in the background)
    mesh_edges: np.ndarray | None = None  # (m, 2) edges of the mesh to draw (indices into V)
    title: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)          # rows at the bottom left
    legend_title: str = ""
    edges: np.ndarray | None = None       # boundaries of the bands
    reverse: bool = False                 # red = small values (safety)
    swatches: list[tuple] = field(default_factory=list)    # [(barva 0..1, text)] – legenda modelu
    markers: list[dict] = field(default_factory=list)      # {"xyz", "text", "color"}
    arrows: list[dict] = field(default_factory=list)       # {"tip", "vec" (mm), "text", "color"}
    lines: list[dict] = field(default_factory=list)        # {"a", "b" (mm), "color", "width"?, "solid"?} segments
    # (a remote constraint dashed, a loaded edge solid)
    feature_angle: float = 30.0
    shade_range: tuple[float, float] = (0.72, 1.0)
    # bars as thick lines with a z-buffer (the rasteriser): [{"V" (n, 3), "E" (m, 2), "colors" (m, 3) | (3,),
    # "width" px, "alpha", "depth" (the body hides them)}] – drawn in order (the outline of the undeformed shape first)
    beams: list[dict] = field(default_factory=list)
    bands_on_mesh: bool = True            # False = values/edges only for the legend (they colour the bars), bodies tri_colors


def edge_adjacency(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique edges (E, 2) and the adjacent triangles (E, 2), −1 = free edge."""
    n = len(T)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    tri = np.tile(np.arange(n), 3)
    e.sort(axis=1)
    key = e[:, 0].astype(np.int64) * (int(T.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    k = key[order]
    start = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(start) - 1
    E = e[order][start]
    adj = np.full((len(E), 2), -1, dtype=np.int64)
    adj[:, 0] = tri[order][start]
    second = np.nonzero(~start)[0]
    if len(second):
        g2 = gid[second]
        _, first = np.unique(g2, return_index=True)
        adj[g2[first], 1] = tri[order][second[first]]
    return E, adj


def tri_normals(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    ln = np.linalg.norm(n, axis=1)
    return n / np.maximum(ln, 1e-300)[:, None]


class Outline:
    """Outline edges of the mesh: the free, sharp and group boundaries do not depend on the view (computed once), the silhouette
    is computed for each direction of the view. adjacency = the finished result of edge_adjacency(T) (the deformed and the undeformed
    shape share it – the same topology), normals = tri_normals(V, T), if already computed.
    """

    def __init__(self, V, T, group, feature_angle, adjacency=None, normals=None):
        E, adj = adjacency if adjacency is not None else edge_adjacency(T)
        n = normals if normals is not None else tri_normals(V, T)
        a, b = adj[:, 0], adj[:, 1]
        free = b < 0
        bb = np.where(free, a, b)
        cosang = np.einsum("ij,ij->i", n[a], n[bb])
        sharp = cosang < math.cos(math.radians(feature_angle))
        grp = np.zeros(len(E), bool) if group is None else group[a] != group[bb]
        self.E, self.a, self.bb, self.n = E, a, bb, n
        self.static = free | sharp | grp

    def edges(self, d) -> np.ndarray:
        nd = self.n @ d
        sil = (nd[self.a] * nd[self.bb]) < 0
        return self.E[self.static | sil]


MULTI_VIEWS = ("iso", "iso_back", "front", "top")


def _legend_width(scene: Scene, W: int, fs: float) -> int:
    if scene.values is not None:
        return int(max(170, 0.17 * W))
    if scene.swatches:                         # fs v bodech, 1 pt = 100/72 px; znak ≈ 0,58 em
        n = max(len(t) for _, t in scene.swatches[:24])
        return int(min(0.45 * W, max(170, (n * 0.58 * 0.8 + 2.0) * fs * 100 / 72 + 16)))
    return 0


def _fit(scene: Scene, r, u, rect):
    """Scale and centre so that the scene (incl. the outline and the arrows) fits into the rectangle rect (px)."""
    vx, vy, vw, vh = rect
    pts = [scene.V]
    if scene.ghost is not None:
        pts.append(scene.ghost)
    for a in scene.arrows:
        tip = np.asarray(a["tip"], float)
        pts.append(np.vstack([tip, tip - np.asarray(a["vec"], float)]))
    for ln in scene.lines:
        pts.append(np.vstack([np.asarray(ln["a"], float), np.asarray(ln["b"], float)]))
    for bm in scene.beams:
        if len(bm["V"]):
            pts.append(np.asarray(bm["V"], float))
    allp = np.vstack([p_ for p_ in pts if len(p_)])
    sx, sy = allp @ r, allp @ u
    cx, cy = 0.5 * (sx.min() + sx.max()), 0.5 * (sy.min() + sy.max())
    s = 0.9 * min(vw / max(np.ptp(sx), 1e-9), vh / max(np.ptp(sy), 1e-9))
    return s, cx, cy, vx + 0.5 * vw, vy + 0.5 * vh


_tls = threading.local()


@contextmanager
def reuse_buffers():
    """Within the block this thread keeps its raster buffers between pictures (a report draws dozens of them: a fresh
    3200 × 1800 float64 colour buffer costs ~0.3 s of page faults). Outside the block every picture allocates its own
    and nothing stays in memory (MCP server, GUI).
    """
    old = getattr(_tls, "bufs", None)
    if old is None:
        _tls.bufs = {}
    try:
        yield
    finally:
        if old is None:
            _tls.bufs = None


def keep_buffers() -> None:
    """This thread keeps its raster buffers until it ends (worker threads of the report pictures)."""
    if getattr(_tls, "bufs", None) is None:
        _tls.bufs = {}


def _buffers(Hs: int, Ws: int):
    """z-buffer (−inf), colour buffer (white) and id buffer (−1) of the supersampled picture."""
    cache = getattr(_tls, "bufs", None)
    if cache is None:
        return (np.full((Hs, Ws), -np.inf), np.ones((Hs, Ws, 3)), np.full((Hs, Ws), -1, dtype=np.int64))
    b = cache.get((Hs, Ws))
    if b is None:
        cache.clear()                                   # one size at a time (memory)
        b = cache[(Hs, Ws)] = (np.empty((Hs, Ws)), np.empty((Hs, Ws, 3)), np.empty((Hs, Ws), dtype=np.int64))
    b[0].fill(-np.inf)
    b[1].fill(1.0)
    b[2].fill(-1)
    return b


def render(scene: Scene, path: str | Path, view="iso", width: int = 1280, height: int = 720,
           supersample: int = 2) -> dict:
    """Renders the scene into PNG. view = name (VIEWS), a vector of the direction towards the camera, or "multi" (2×2: iso,
    iso from behind, from the front, from above). Returns info (views, scale px/mm, visibility of the marks).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle, FancyArrowPatch

    W, H = int(width), int(height)
    if not (200 <= W <= 4096 and 150 <= H <= 4096):
        raise ValueError("Image size must be 200–4096 × 150–4096 px")
    multi = isinstance(view, str) and view.strip().lower() == "multi"
    views = list(MULTI_VIEWS) if multi else [view]
    fs = max(8.0, min(W / 1280, H / 720) * 13.0)
    legend_w = _legend_width(scene, W, fs)
    title_h = int(0.105 * H) + 8
    bottom = int(0.07 * H)
    area = (12, title_h, W - legend_w - 24, H - title_h - bottom)
    if multi:
        ax0, ay0, aw, ah = area
        gw, gh = (aw - 10) / 2, (ah - 10) / 2
        rects = [(ax0 + (i % 2) * (gw + 10), ay0 + (i // 2) * (gh + 10), gw, gh) for i in range(4)]
    else:
        rects = [area]
    k = int(supersample)
    Hs, Ws = H * k, W * k
    zbuf, cbuf, idbuf = _buffers(Hs, Ws)
    V, T = scene.V, np.asarray(scene.T, np.int64).reshape(-1, 3)
    has_tri = len(T) > 0                                # bars only → no triangles
    n = tri_normals(V, T) if has_tri else np.zeros((0, 3))
    if has_tri:
        adjacency = edge_adjacency(T)                   # topology once per render (also for the outline of the undeformed shape)
        outline = Outline(V, T, scene.tri_group, scene.feature_angle, adjacency, n)
        ghost_outline = (Outline(scene.ghost, T, scene.tri_group, scene.feature_angle, adjacency)
                         if scene.ghost is not None else None)
    else:
        outline = ghost_outline = None
    group = scene.tri_group if scene.tri_group is not None else np.zeros(len(T), np.int64)
    group = np.asarray(group, np.int64)
    allv = np.vstack([V.reshape(-1, 3)] + [np.asarray(bm["V"], float).reshape(-1, 3) for bm in scene.beams])
    diag = float(np.linalg.norm(allv.max(axis=0) - allv.min(axis=0))) if len(allv) else 1.0
    diag = diag or 1.0
    if scene.values is not None:
        band_e = np.asarray(scene.edges, float)
        lut = np.asarray(band_colors(len(band_e) - 1), float) / 255.0
        if scene.reverse:
            lut = lut[::-1]
        vals = np.nan_to_num(np.asarray(scene.values, float))[T] if scene.bands_on_mesh else None
    views_out = []
    for vw_, rect in zip(views, rects):
        r, u, d, vlabel = camera(vw_)
        fit_rect = (rect[0], rect[1] + 2.0 * fs, rect[2], rect[3] - 2.0 * fs) if multi else rect
        s, cx, cy, ox, oy = _fit(scene, r, u, fit_rect)

        def proj(P, ss=1, r=r, u=u, d=d, s=s, cx=cx, cy=cy, ox=ox, oy=oy):
            P = np.atleast_2d(np.asarray(P, float))
            return ((P @ r - cx) * s + ox) * ss, (-(P @ u - cy) * s + oy) * ss, P @ d

        # clipping to the rectangle of the view (in a 2×2 the views must not overlap)
        x0, y0, w0, h0 = (int(round(v * k)) for v in rect)
        zb, cb, ib = zbuf[y0:y0 + h0, x0:x0 + w0], cbuf[y0:y0 + h0, x0:x0 + w0], idbuf[y0:y0 + h0, x0:x0 + w0]
        px, py, pz = proj(V, k)
        px, py = px - x0, py - y0
        lo, hi = scene.shade_range
        light = d + 0.35 * u - 0.25 * r
        light /= np.linalg.norm(light)
        shade = lo + (hi - lo) * (0.7 * np.abs(n @ d) + 0.3 * np.clip(n @ light, 0, 1))
        if not has_tri:
            pass
        elif scene.values is not None and scene.bands_on_mesh:
            raster_triangles(px[T], py[T], pz[T], vals, np.zeros((len(T), 3)), shade, group, 0,
                             band_e, np.ascontiguousarray(lut), zb, cb, ib)
        else:
            raster_triangles(px[T], py[T], pz[T], np.zeros((len(T), 3)), np.asarray(scene.tri_colors, float),
                             shade, group, 1, np.zeros(2), np.zeros((1, 3)), zb, cb, ib)
        bias = max(3.0 / s, 0.003 * diag)

        def lines(P, Eidx, color, alpha, width_px, depth=True, background=False):
            if len(Eidx) == 0:
                return
            x, y, z = proj(P, k)
            x, y = x - x0, y - y0
            a, b = Eidx[:, 0], Eidx[:, 1]
            col = np.tile(np.asarray(color, float), (len(Eidx), 1))
            raster_lines(x[a], y[a], z[a], x[b], y[b], z[b], col, float(alpha), float(width_px * k), bias,
                         depth, background, zb, cb, ib)

        if ghost_outline is not None:
            lines(scene.ghost, ghost_outline.edges(d), (0.45, 0.45, 0.5), 0.9, 1.0, depth=False, background=True)
        if scene.mesh_edges is not None:
            lines(V, scene.mesh_edges, (0.1, 0.1, 0.1), 0.35, 0.6)
        if outline is not None:
            lines(V, outline.edges(d), (0.08, 0.08, 0.1), 1.0, 1.25)
        for bm in scene.beams:                         # bars: thick segments (colour per segment)
            E_ = np.asarray(bm["E"], np.int64).reshape(-1, 2)
            if not len(E_):
                continue
            x, y, z = proj(bm["V"], k)
            x, y = x - x0, y - y0
            col = np.asarray(bm.get("colors", (0.2, 0.2, 0.25)), float)
            col = np.ascontiguousarray(np.tile(col, (len(E_), 1)) if col.ndim == 1 else col)
            a_, b_ = E_[:, 0], E_[:, 1]
            raster_lines(x[a_], y[a_], z[a_], x[b_], y[b_], z[b_], col, float(bm.get("alpha", 1.0)),
                         float(bm.get("width", 3.0) * k), bias, bool(bm.get("depth", has_tri)), False, zb, cb, ib)

        def visible(xyz, proj=proj, zb=zb, bias=bias, x0=x0, y0=y0) -> bool:
            x, y, z = proj(xyz, k)
            xi, yi = int(x[0]) - x0, int(y[0]) - y0
            win = zb[max(yi - k, 0):max(yi + k + 1, 0), max(xi - k, 0):max(xi + k + 1, 0)]
            if win.size == 0:
                return False
            return bool(z[0] >= win.max() - 2 * bias)
        views_out.append((vlabel, rect, proj, visible, r, u, d, s))
    img = to_rgb8(cbuf, k)

    # ---------------------------------------------------------------- popisky (matplotlib Agg)
    # The raster picture fills the whole canvas 1:1: it is written straight into the Agg buffer (the same pixels as
    # ax.imshow(interpolation="none") at this size, without its resampling ~0.4 s) and the labels are drawn over it.
    fig = Figure(figsize=(W / 100, H / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_visible(False)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    for i, line in enumerate(scene.title[:3]):
        ax.text(14, 10 + i * fs * 1.75, line, fontsize=fs * (1.25 if i == 0 else 0.95), va="top",
                weight="bold" if i == 0 else "normal", color="#111")
    info = list(scene.info) + ([] if multi else [f"View: {views_out[0][0]}"])
    for i, line in enumerate(reversed(info)):
        ax.text(area[0] + area[2], H - 8 - i * fs * 1.45, line, fontsize=fs * 0.8, ha="right", va="bottom",
                color="#333")
    sm = 0.85 if multi else 1.0
    vis: dict[str, bool] = {}
    pending: dict[str, tuple] = {}

    def triad(x0, y0, L, r, u, d, f):
        for e, name, col in ((np.eye(3)[0], "X", "#d62728"), (np.eye(3)[1], "Y", "#2ca02c"),
                             (np.eye(3)[2], "Z", "#1f4fd1")):
            ddx, ddy = (e @ r) * L, -(e @ u) * L
            flat = math.hypot(ddx, ddy) <= 0.15 * L
            if not flat:
                ax.add_patch(FancyArrowPatch((x0, y0), (x0 + ddx, y0 + ddy), arrowstyle="-|>",
                                             mutation_scale=f, lw=1.8, color=col))
            ax.text(x0 + ddx * 1.35, y0 + ddy * 1.35,
                    name + ("" if not flat else (" ⊙" if e @ d > 0 else " ⊗")),
                    fontsize=f * 0.85, color=col, ha="center", va="center", weight="bold")

    for vlabel, rect, proj, visible, r, u, d, _ in views_out:
        rx, ry, rw, rh = rect
        if multi:
            ax.add_patch(Rectangle((rx, ry), rw, rh, fill=False, ec="#bbb", lw=0.8))
            ax.text(rx + 6, ry + 4, vlabel, fontsize=fs * 0.8, va="top", color="#333",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        for ln in scene.lines:                             # segments point → face (through the model, dashed)
            (xa,), (ya,), _ = proj(ln["a"])
            (xb,), (yb,), _ = proj(ln["b"])
            ax.plot([xa, xb], [ya, yb], color=ln.get("color", (0.2, 0.3, 0.8)), lw=ln.get("width", 1.3) * sm,
                    ls="-" if ln.get("solid") else "--", zorder=4)
        for a in scene.arrows:
            tip = np.asarray(a["tip"], float)
            tail = tip - np.asarray(a["vec"], float)
            (x1,), (y1,), _ = proj(tip)
            (xa,), (ya,), _ = proj(tail)
            col = a.get("color", (0.85, 0.1, 0.1))
            if math.hypot(x1 - xa, y1 - ya) < 4:              # the arrow points into the view → a ring
                ax.plot([x1], [y1], marker="o", ms=fs * 0.9 * sm, mfc="none", mec=col, mew=2)
            else:
                ax.add_patch(FancyArrowPatch((xa, ya), (x1, y1), arrowstyle="-|>", mutation_scale=fs * 1.6 * sm,
                                             lw=2.2 * sm, color=col, zorder=5))
            if a.get("text"):                              # label at the free end of the arrow, outwards from the model
                lx_, ly_ = (x1, y1) if a.get("label_at") == "tip" else (xa, ya)
                tw = (len(a["text"]) + 2) * 0.6 * fs * 0.85 * sm * 100 / 72     # estimate of the width of the text [px]
                tx = min(max(lx_ - 0.5 * tw, rx + 4), rx + rw - tw - 4)      # above the end of the arrow, inside the view
                below = ly_ < ry + 2.5 * fs                                    # at the upper edge → below the arrow
                ax.text(tx, ly_ + (8 if below else -8), " " + a["text"] + " ", fontsize=fs * 0.85 * sm, color=col,
                        weight="bold", zorder=6, va="top" if below else "bottom", ha="left",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
        for m in scene.markers:
            (x,), (y,), _ = proj(m["xyz"])
            v = visible(m["xyz"])
            if m.get("only_visible"):                          # labels of the faces: only the visible ones, small
                if v:
                    ax.text(x, y, m.get("text", ""), fontsize=fs * 0.62 * sm, ha="center", va="center", zorder=7,
                            bbox=dict(boxstyle="round,pad=0.12", fc="#ffffcc", ec="#555", lw=0.5, alpha=0.85))
                continue
            key = m.get("key", m.get("text", ""))
            vis[key] = vis.get(key, False) or v
            if multi and not v:
                pending.setdefault(key, m)
                continue
            _marker(ax, m, x, y, v, rect, fs * sm)
        if multi:
            triad(rx + 2.4 * fs * sm, ry + rh - 2.4 * fs * sm, 1.9 * fs * sm, r, u, d, fs * sm)
        else:
            triad(14 + 3.2 * fs, H - 14 - 3.2 * fs, 2.6 * fs, r, u, d, fs)
    for key, m in pending.items():                       # not visible in any view → into the first
        if not vis.get(key):
            (x,), (y,), _ = views_out[0][2](m["xyz"])
            _marker(ax, m, x, y, False, views_out[0][1], fs * sm)

    # legenda
    lx = area[0] + area[2] + 12
    vy = area[1]
    if scene.values is not None:
        nb = len(band_e) - 1
        nl = scene.legend_title.count("\n") + 1
        top = vy + (1.0 + 1.4 * nl) * fs
        bh = min((area[3] - (top - vy) - fs) / nb, 2.6 * fs)
        ax.text(lx, vy, scene.legend_title, fontsize=fs, weight="bold", va="top", color="#111")
        bw = 1.8 * fs
        for i in range(nb):                                 # maximum at the top
            j = nb - 1 - i
            ax.add_patch(Rectangle((lx, top + i * bh), bw, bh, fc=lut[j], ec="#333", lw=0.5))
        for i in range(nb + 1):
            ax.text(lx + bw + 6, top + i * bh, fmt(float(band_e[nb - i])), fontsize=fs * 0.85, va="center",
                    color="#111")
    elif scene.swatches:
        ax.text(lx, vy, scene.legend_title or "Legend", fontsize=fs, weight="bold", va="top")
        bh = 1.6 * fs
        for i, (col, txt) in enumerate(scene.swatches[:24]):
            y = vy + 2.2 * fs + i * bh
            ax.add_patch(Rectangle((lx, y), 1.2 * fs, 1.0 * fs, fc=col, ec="#333", lw=0.6))
            ax.text(lx + 1.6 * fs, y + 0.5 * fs, txt, fontsize=fs * 0.8, va="center")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_png(fig, canvas, img, path)
    return {"path": str(path), "width": W, "height": H, "views": [v[0] for v in views_out],
            "px_per_mm": [round(float(v[7]), 4) for v in views_out], "markers_visible": vis}


def _save_png(fig, canvas, img: np.ndarray, path: Path) -> None:
    """Figure → PNG as fig.savefig(path, dpi=100) with the picture img (H, W, 3) as the background of the whole canvas:
    the Agg buffer is filled with img, the artists are drawn over it and the buffer is written by the same
    matplotlib.image.imsave call that print_png uses.
    """
    import matplotlib.image as mimage
    renderer = canvas.get_renderer()
    renderer.clear()
    buf = np.asarray(renderer.buffer_rgba())
    if buf.shape[:2] != img.shape[:2]:                   # cannot happen for W, H in px at dpi 100 – plain way then
        ax = fig.axes[0]
        ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0), interpolation="none", zorder=-1)
        fig.patch.set_visible(True)
        fig.savefig(path, dpi=100, format="png")
        return
    buf[..., :3] = img
    buf[..., 3] = 255
    fig.draw(renderer)
    mimage.imsave(path, renderer.buffer_rgba(), format="png", origin="upper", dpi=fig.dpi)


def _marker(ax, m, x, y, v, rect, fs):
    col = m.get("color", "black")
    ax.plot([x], [y], marker="o", ms=fs * 0.75, mfc=col if v else "white", mec="black", mew=1.3, zorder=7)
    rx, ry, rw, rh = rect
    dx = 18 if x < rx + 0.7 * rw else -18
    dy = -18 if y > ry + 0.2 * rh else 18
    if not m.get("text"):                                       # a point without a label (further points of the force)
        return
    txt = m.get("text", "") + ("" if v else " (hidden)")
    ax.annotate(txt, (x, y), (x + dx, y + dy), textcoords="data", fontsize=fs * 0.85, zorder=8,
                ha="left" if dx > 0 else "right", va="center", color="black",
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.2, alpha=0.9))
