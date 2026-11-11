"""Section mesh from Gmsh instead of Triangle (cytriangle has a non-commercial licence) – `fea_core.beams.section_mesh`.

Fast tests: the triangulation itself (area, holes, regions, node order of tri6) and the section itself in a subprocess WITHOUT the
cytriangle package (`sys.modules["cytriangle"] = None` before the import) – a rectangle: A = b·h, J according to Roark to 1 %.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("shapely")

ROOT = Path(__file__).resolve().parents[1]


def _areas(m):
    V, T = m["vertices"], m["triangles"]
    x = V[T[:, :3]]
    return 0.5 * ((x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1]) - (x[:, 1, 1] - x[:, 0, 1]) * (x[:, 2, 0] - x[:, 0, 0]))


def test_triangulate_rectangle_with_hole_and_regions():
    from fea_core.beams.section_mesh import triangulate
    # plate 20 × 10 divided at x = 10 into two regions (materials), in the left one a square hole 4 × 4
    pts = [[0, 0], [10, 0], [20, 0], [20, 10], [10, 10], [0, 10], [3, 3], [7, 3], [7, 7], [3, 7]]
    seg = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [1, 4], [6, 7], [7, 8], [8, 9], [9, 6]]
    m = triangulate({"vertices": pts, "segments": seg, "holes": [[5, 5]],
                     "regions": [[1, 1, 0, 1.0], [15, 5, 1, 2.0]]}, "pq30.0Aao2")
    A = _areas(m)
    assert m["triangles"].shape[1] == 6 and m["triangle_attributes"].shape == (len(A), 1)
    assert np.all(A > 0)                                     # anticlockwise
    att = m["triangle_attributes"][:, 0].astype(int)
    assert A[att == 0].sum() == pytest.approx(100 - 16, rel=1e-12)
    assert A[att == 1].sum() == pytest.approx(100, rel=1e-12)
    assert A[att == 0].max() <= 1.0 + 1e-9 and A[att == 1].max() <= 2.0 + 1e-9    # hard limit of the area like Triangle
    # order of the nodes as Triangle: midpoints of the edges 1-2, 2-0, 0-1
    V, T = m["vertices"], m["triangles"]
    assert np.allclose(V[T[:, 3]], 0.5 * (V[T[:, 1]] + V[T[:, 2]]))
    assert np.allclose(V[T[:, 4]], 0.5 * (V[T[:, 2]] + V[T[:, 0]]))
    assert np.allclose(V[T[:, 5]], 0.5 * (V[T[:, 0]] + V[T[:, 1]]))
    # conforming mesh between the regions: every node is used (none isolated) and the edges shared by the regions have no duplicate nodes
    assert len(np.unique(T)) == len(V)
    assert len(np.unique(np.round(V, 9), axis=0)) == len(V)


def test_triangulate_is_deterministic():
    from fea_core.beams.section_mesh import triangulate
    t = {"vertices": [[0, 0], [30, 0], [30, 10], [0, 10]], "segments": [[0, 1], [1, 2], [2, 3], [3, 0]],
         "regions": [[5, 5, 0, 2.0]]}
    import copy
    a = triangulate(copy.deepcopy(t), "pq30.0Aao2")
    b = triangulate(copy.deepcopy(t), "pq30.0Aao2")
    assert np.array_equal(a["vertices"], b["vertices"]) and np.array_equal(a["triangles"], b["triangles"])


_CODE = r"""
import sys
sys.modules["cytriangle"] = None                 # cytriangle není (import by selhal)
sys.path.insert(0, r"%s")
from fea_core.beams import sections as S
p = S.custom_section("obdélník", [(0, 0), (40, 0), (40, 20), (0, 20)])
shim = sys.modules["cytriangle"]
assert shim is not None and shim.triangulate.__module__ == "fea_core.beams.section_mesh", shim
print(repr((p.A, p.Iy, p.Iz, p.J)))
"""


def test_custom_section_without_cytriangle():
    r = subprocess.run([sys.executable, "-c", _CODE % ROOT], capture_output=True, text=True, cwd=ROOT, timeout=240)
    assert r.returncode == 0, r.stdout + r.stderr
    A, Iy, Iz, J = eval(r.stdout.strip().splitlines()[-1])
    b, h = 20.0, 40.0                                          # shorter and longer side
    assert A == pytest.approx(b * h, rel=1e-9)
    assert Iy == pytest.approx(b * h ** 3 / 12, rel=1e-9) and Iz == pytest.approx(h * b ** 3 / 12, rel=1e-9)
    # Roark: J = a·b³·[1/3 − 0.21·(b/a)·(1 − b⁴/(12a⁴))], a = longer, b = shorter side
    J_roark = h * b ** 3 * (1 / 3 - 0.21 * (b / h) * (1 - b ** 4 / (12 * h ** 4)))
    assert J == pytest.approx(J_roark, rel=0.01)
