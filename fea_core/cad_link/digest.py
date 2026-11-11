"""Fingerprint of the geometry: "the model has not changed" and the signature of the mesh according to the content of the STEP (not according to the path of the file)."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

_HEADER = re.compile(rb"HEADER\s*;.*?ENDSEC\s*;", re.S | re.I)


def step_digest(path: str | Path) -> str:
    """SHA-256 of the content of the STEP without the header (FILE_NAME carries the time of the export) and without the difference of the line ends – the same model
    exported again has the same fingerprint, if the exporter does not number the entities differently. The decisive test "the model has not changed" is the
    `fingerprint` of the parts; this is only a quick match of the file.
    """
    data = Path(path).read_bytes()
    data = _HEADER.sub(b"", data, count=1)
    data = b"\n".join(line.strip() for line in data.splitlines() if line.strip())
    return hashlib.sha256(data).hexdigest()[:32]


def part_fingerprint(part) -> dict:
    """Volume, surface area, number of faces and the envelope of the part (rounded) – comparison of the versions of the geometry."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from ..geometry import bbox, list_faces
    vp, sp = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(part.shape, vp)
    BRepGProp.SurfaceProperties_s(part.shape, sp)
    c = vp.CentreOfMass()
    return {"volume": float(vp.Mass()), "area": float(sp.Mass()), "faces": len(list_faces(part.shape)),
            "center": [c.X(), c.Y(), c.Z()], "bbox": [float(v) for v in bbox(part.shape)]}


def same_shape(a: dict, b: dict, tol: float = 1e-6) -> bool:
    """Matching fingerprints of a part (relative tolerance to the size of the part)."""
    if a.get("faces") != b.get("faces"):
        return False
    bb = np.asarray(a["bbox"], float)
    size = max(float(np.linalg.norm(bb[3:] - bb[:3])), 1e-9)
    if abs(a["volume"] - b["volume"]) > tol * max(abs(a["volume"]), 1e-12) \
            or abs(a["area"] - b["area"]) > tol * max(abs(a["area"]), 1e-12):
        return False
    return (np.abs(np.asarray(a["bbox"], float) - np.asarray(b["bbox"], float)).max() <= tol * size
            and np.linalg.norm(np.asarray(a["center"]) - np.asarray(b["center"])) <= tol * size)
