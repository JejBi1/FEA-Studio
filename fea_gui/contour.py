"""Coloured contours of the results in bands and their conversion to OCCT faces.

Cutting the triangles into bands (numpy) is in fea_core.contour (shared with the headless renderer of MCP).
Each band is one face with a triangulation (through a temporary binary STL – filling the
Poly_Triangulation node by node from Python is slow).
"""
from __future__ import annotations

import os
import struct
import tempfile

import numpy as np

from fea_core.contour import (  # noqa: F401 – re-export for the GUI
    BAND_COLORS, SUBTRI, linear_triangles, band_edges, contour_bands, triangle_bands, _clip,
)


def triangles_face(P: np.ndarray):
    """TopoDS_Face only with a triangulation from triangles (n, 3, 3), through a temporary binary STL."""
    from OCP.BRep import BRep_Builder
    from OCP.RWStl import RWStl
    from OCP.TopoDS import TopoDS_Face
    if len(P) == 0:
        return None
    rec = np.zeros(len(P), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["v"] = P
    fd, path = tempfile.mkstemp(prefix="fea_", suffix=".stl")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(bytes(80) + struct.pack("<I", len(rec)) + rec.tobytes())
        tri = RWStl.ReadFile_s(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if tri is None or tri.NbTriangles() == 0:
        return None
    tri.ComputeNormals()
    face = TopoDS_Face()
    BRep_Builder().MakeFace(face, tri)
    return face
