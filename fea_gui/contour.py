"""Coloured contours of the results in bands and their conversion to OCCT faces.

Cutting the triangles into bands (numpy) is in fea_core.contour (shared with the headless renderer of MCP).
Each band is one face with a triangulation (through a binary STL read from memory – filling the
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


_STL_DTYPE = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
_stream_ok = True


def _stl_triangulation(P: np.ndarray):
    """Poly_Triangulation of the triangles (n, 3, 3) through a binary STL – read from memory (RWStl.ReadBinaryStream,
    ~7× faster than a temporary file, the same nodes: the reader merges equal float32 vertices either way); a temporary
    file only if the stream reader is not available.
    """
    global _stream_ok
    from OCP.RWStl import RWStl
    rec = np.zeros(len(P), dtype=_STL_DTYPE)
    rec["v"] = P
    data = bytes(80) + struct.pack("<I", len(rec)) + rec.tobytes()
    if _stream_ok:
        import io
        try:
            return RWStl.ReadBinaryStream_s(io.BytesIO(data))
        except Exception:  # noqa: BLE001 – binding without std::istream support
            _stream_ok = False
    fd, path = tempfile.mkstemp(prefix="fea_", suffix=".stl")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return RWStl.ReadFile_s(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def triangles_face(P: np.ndarray):
    """TopoDS_Face only with a triangulation from triangles (n, 3, 3), through a binary STL in memory."""
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Face
    if len(P) == 0:
        return None
    tri = _stl_triangulation(P)
    if tri is None or tri.NbTriangles() == 0:
        return None
    tri.ComputeNormals()
    face = TopoDS_Face()
    BRep_Builder().MakeFace(face, tri)
    return face
