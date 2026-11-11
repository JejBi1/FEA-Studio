"""Software rasteriser of triangles and segments with a z-buffer (numba) – headless, without OpenGL and display.

Orthogonal projection: screen coordinates (x to the right, y downwards in pixels) and depth z (larger = nearer
to the camera). In the mode of bands the value is interpolated linearly in the pixel (for an orthogonal projection exactly
as on the linear triangle) and only then classified into the band → the boundaries of the bands are sharp without clipping
of the polygons. Smoothing of the edges = resampling (rendered at a larger resolution and reduced by the mean of the blocks).
"""
from __future__ import annotations

import functools
import threading

import numpy as np

from ..modules import need


def _kernel(fn):
    """Numba kernel compiled at the first call (same code, same cache). Numba and llvmlite are not imported when the
    MCP server or the GUI starts – only when a picture is really drawn (about 30 MB less idle memory). A missing
    module gives the clear error with the installation procedure (modules.ModuleMissing) at that first call.
    """
    compiled: list = []

    @functools.wraps(fn)
    def run(*args):
        if not compiled:
            with _compile_lock:                  # pictures drawn in parallel threads compile the kernel once
                if not compiled:
                    # nogil: the kernels touch only their arrays → pictures can be rasterised in parallel threads
                    compiled.append(need("numba", "accel").njit(cache=True, nogil=True)(fn))
        return compiled[0](*args)
    return run


_compile_lock = threading.Lock()


@_kernel
def raster_triangles(px, py, pz, vals, cols, shade, ids, mode, edges, lut, zbuf, cbuf, idbuf):
    """Draws the triangles.

    px, py, pz (T, 3) – vertices in pixels and depth; vals (T, 3) – values of the field (mode 0);
    cols (T, 3) – colour 0..1 (mode 1); shade (T,) – multiple of the brightness; ids (T,) – number of the part / face
    to idbuf; edges (nb+1,) boundaries of the bands, lut (nb, 3) colours of the bands.
    """
    H, W = zbuf.shape
    nb = lut.shape[0]
    T = px.shape[0]
    for t in range(T):
        x0, y0, z0 = px[t, 0], py[t, 0], pz[t, 0]
        x1, y1, z1 = px[t, 1], py[t, 1], pz[t, 1]
        x2, y2, z2 = px[t, 2], py[t, 2], pz[t, 2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        inv = 1.0 / area
        xmin = max(int(np.floor(min(x0, min(x1, x2)))), 0)
        xmax = min(int(np.ceil(max(x0, max(x1, x2)))), W - 1)
        ymin = max(int(np.floor(min(y0, min(y1, y2)))), 0)
        ymax = min(int(np.ceil(max(y0, max(y1, y2)))), H - 1)
        if xmin > xmax or ymin > ymax:
            continue
        v0, v1, v2 = vals[t, 0], vals[t, 1], vals[t, 2]
        sh = shade[t]
        for yy in range(ymin, ymax + 1):
            cy = yy + 0.5
            for xx in range(xmin, xmax + 1):
                cx = xx + 0.5
                w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) * inv
                w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) * inv
                w2 = 1.0 - w0 - w1
                if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z <= zbuf[yy, xx]:
                    continue
                zbuf[yy, xx] = z
                idbuf[yy, xx] = ids[t]
                if mode == 0:
                    v = w0 * v0 + w1 * v1 + w2 * v2
                    k = 0
                    while k < nb - 1 and v >= edges[k + 1]:
                        k += 1
                    for c in range(3):
                        cbuf[yy, xx, c] = lut[k, c] * sh
                else:
                    for c in range(3):
                        cbuf[yy, xx, c] = cols[t, c] * sh


@_kernel
def raster_lines(ax, ay, az, bx, by, bz, cols, alpha, width, bias, depth_test, only_background,
                 zbuf, cbuf, idbuf):
    """Segments (L,) with the width [px]; visible if z >= zbuf − bias (depth_test).

    only_background = draw only into the empty background (the outline of the undeformed shape).
    """
    H, W = zbuf.shape
    L = ax.shape[0]
    r = width * 0.5
    for i in range(L):
        dx, dy = bx[i] - ax[i], by[i] - ay[i]
        n = int(max(abs(dx), abs(dy)) * 2.0) + 1
        for s in range(n + 1):
            f = s / n
            x = ax[i] + f * dx
            y = ay[i] + f * dy
            z = az[i] + f * (bz[i] - az[i])
            for yy in range(int(np.floor(y - r + 0.5)), int(np.floor(y + r + 0.5)) + 1):
                if yy < 0 or yy >= H:
                    continue
                for xx in range(int(np.floor(x - r + 0.5)), int(np.floor(x + r + 0.5)) + 1):
                    if xx < 0 or xx >= W:
                        continue
                    if only_background and idbuf[yy, xx] >= 0:
                        continue
                    if depth_test and z < zbuf[yy, xx] - bias:
                        continue
                    for c in range(3):
                        cbuf[yy, xx, c] = cbuf[yy, xx, c] * (1.0 - alpha) + cols[i, c] * alpha


@_kernel
def _to_rgb8(img, k, out):
    """uint8 picture of the colour buffer: block mean k×k, clip 0..1, ×255, truncation – the same operations in the same
    order as `(np.clip(downsample(img, k), 0, 1) * 255).astype(np.uint8)` (bit-identical), without the float64 temporaries.
    """
    H, W, C = out.shape
    inv = k * k
    for y in range(H):
        for x in range(W):
            for c in range(C):
                a = 0.0
                for i in range(k):
                    for j in range(k):
                        a += img[y * k + i, x * k + j, c]
                if k != 1:
                    a = a / inv
                if a < 0.0:
                    a = 0.0
                elif a > 1.0:
                    a = 1.0
                out[y, x, c] = np.uint8(a * 255.0)


def to_rgb8(img: np.ndarray, k: int) -> np.ndarray:
    """Final picture (H, W, 3) uint8 from the supersampled colour buffer (see _to_rgb8)."""
    out = np.empty((img.shape[0] // k, img.shape[1] // k, img.shape[2]), np.uint8)
    _to_rgb8(img, int(k), out)
    return out


def downsample(img: np.ndarray, k: int) -> np.ndarray:
    """Reduction by the mean of the blocks k×k (smoothing of the edges)."""
    if k == 1:
        return img
    H, W = img.shape[0] // k, img.shape[1] // k
    acc = np.zeros((H, W) + img.shape[2:])
    for i in range(k):                    # the sum of the shifted sections is ~10× faster than mean(axis=(1, 3))
        for j in range(k):
            acc += img[i:H * k:k, j:W * k:k]
    return acc / (k * k)
