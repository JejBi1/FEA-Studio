"""Table of the faces from the CAD in the metadata (Fusion add-in): identity of the faces and bodies across the versions of the geometry.

STEP carries no marks of faces, therefore the add-in sends a table next to the STEP (everything in mm, global position of the assembly):

    "faces":    [{"token", "part": index of the row "parts", "type", "area", "centroid", "point", "normal"}, …]
    "resolved": [{"token": token from the last export, "faces": [{"type", "area", "centroid", "point", "normal"}]}]
    "resolved_bodies": [{"token", "bodies": [{"volume", "center", "bbox"}]}]
    "parts"[i]["token"] = token of the body

The tokens (Fusion entityToken) are NEVER compared with each other – only the add-in translates them in the CAD
(findEntityByToken) to the faces of the current version and sends their geometry. FEA Studio then matches the rows with the faces of OCP
of the same version by the area and the centroid (different kernels – tolerance 1e-3). A face split in the CAD (a groove) may return more pieces.
"""
from __future__ import annotations

import math
import re

import numpy as np

MAX_ROWS = 200_000
TOKEN_RE = re.compile(r"^[\x21-\x7e]{1,4096}$")       # printable ASCII (Fusion base64), without spaces
TYPES = {"plane", "cylinder", "cone", "sphere", "torus", "other"}
BIND_TOL = 1e-3


def _vec(v, n=3):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(x) for x in out) else None


def _geom_row(r) -> dict | None:
    """Geometry of the face from the metadata (verified numbers) or None."""
    if not isinstance(r, dict):
        return None
    try:
        area = float(r.get("area"))
    except (TypeError, ValueError):
        return None
    c, p = _vec(r.get("centroid")), _vec(r.get("point"))
    if not math.isfinite(area) or area <= 0 or c is None:
        return None
    out = {"area": area, "centroid": c, "point": p or c, "normal": _vec(r.get("normal")) or [0.0, 0.0, 0.0],
           "type": r.get("type") if r.get("type") in TYPES else "other"}
    if isinstance(r.get("part"), int):
        out["part"] = r["part"]
    tok = r.get("token")
    if isinstance(tok, str) and TOKEN_RE.match(tok):
        out["token"] = tok
    return out


def face_rows(meta: dict | None) -> list[dict]:
    rows = (meta or {}).get("faces") or []
    if not isinstance(rows, list):
        return []
    return [g for g in (_geom_row(r) for r in rows[:MAX_ROWS]) if g is not None]


def resolved(meta: dict | None) -> dict[str, list[dict]]:
    """Token of the last version → geometry of the faces of the current version (empty list = the face disappeared in the CAD)."""
    out: dict[str, list[dict]] = {}
    items = (meta or {}).get("resolved") or []
    if not isinstance(items, list):
        return out
    for it in items[:MAX_ROWS]:
        if not isinstance(it, dict) or not isinstance(it.get("token"), str) or not TOKEN_RE.match(it["token"]):
            continue
        faces = it.get("faces") if isinstance(it.get("faces"), list) else []
        out[it["token"]] = [g for g in (_geom_row(f) for f in faces[:64]) if g is not None]
    return out


def resolved_bodies(meta: dict | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for it in ((meta or {}).get("resolved_bodies") or [])[:MAX_ROWS]:
        if not isinstance(it, dict) or not isinstance(it.get("token"), str) or not TOKEN_RE.match(it["token"]):
            continue
        rows = []
        for b in (it.get("bodies") or [])[:64]:
            if not isinstance(b, dict):
                continue
            try:
                vol = float(b.get("volume"))
            except (TypeError, ValueError):
                continue
            c = _vec(b.get("center"))
            if c is None or not math.isfinite(vol):
                continue
            rows.append({"volume": vol, "center": c, "bbox": _vec(b.get("bbox"), 6)})
        out[it["token"]] = rows
    return out


def parameters(meta: dict | None) -> dict:
    """Parameters of the CAD {name: {"expression", "value", "unit"}} (only text and numbers, limited length)."""
    out = {}
    raw = (meta or {}).get("params") or {}
    if not isinstance(raw, dict):
        return out
    for k, v in list(raw.items())[:2000]:
        if not isinstance(k, str) or len(k) > 128 or not isinstance(v, dict):
            continue
        row = {}
        for key in ("expression", "unit", "comment"):
            if isinstance(v.get(key), str):
                row[key] = v[key][:256]
        try:
            val = float(v.get("value"))
            if math.isfinite(val):
                row["value"] = val
        except (TypeError, ValueError):
            pass
        out[k] = row
    return out


# ===================================================================== matching of the rows with the faces of OCP

def _score(row: dict, props: dict, size: float) -> float:
    a0, a1 = row["area"], float(props["area"])
    da = abs(a0 - a1) / max(a0, 1e-12)
    dc = float(np.linalg.norm(np.asarray(row["centroid"]) - np.asarray(props["centroid"]))) / size
    return max(da, dc)


def bind_row(row: dict, geo, parts: list[int] | None = None, tol: float = BIND_TOL) -> tuple[int, int] | None:
    """Face of OCP (index of the part, index of the face) corresponding to a row of the table of the CAD (the same version of the geometry).

    parts = parts in which to search (None = all). Match of the area and the centroid within the tolerance (different geometric kernels);
    the same kind of face takes precedence, two equally good faces = ambiguous (None).
    """
    best: list[tuple[float, int, int]] = []
    for pi in (parts if parts is not None else range(len(geo.parts))):
        pf = geo.part(pi)
        if pf is None:
            continue
        # the centroid of a curved face may lie outside the body, but not far beyond the envelope of the part
        c = np.asarray(row["centroid"])
        if np.any(c < np.asarray(pf.bbox[:3]) - 0.5 * pf.size) or np.any(c > np.asarray(pf.bbox[3:]) + 0.5 * pf.size):
            continue
        for fi, props in enumerate(pf.faces):
            s = _score(row, props, pf.size)
            if s <= tol:
                pen = 0.0 if props["type"] == row["type"] or row["type"] == "other" else tol
                best.append((s + pen, pi, fi))
    if not best:
        return None
    best.sort()
    if len(best) > 1 and best[1][0] <= best[0][0] * 1.0001 + 1e-12 and best[1][1:] != best[0][1:]:
        return None
    return best[0][1], best[0][2]


def bind_rows(rows: list[dict], geo, part_of_row=None, tol: float = BIND_TOL) -> dict[int, tuple[int, int]]:
    """Rows of the table → faces of OCP; part_of_row(row) = index of the part or None (search everywhere)."""
    out = {}
    for k, r in enumerate(rows):
        pi = part_of_row(r) if part_of_row is not None else None
        hit = bind_row(r, geo, None if pi is None else [pi], tol)
        if hit is None and pi is not None:
            hit = bind_row(r, geo, None, tol)
        if hit is not None:
            out[k] = hit
    return out


def row_part_map(meta: dict | None, part_names: list[str]) -> dict[int, int]:
    """Index of a row of the "parts" of the metadata → index of the part of the STEP (matching by names as cad_import.match_parts)."""
    from ..cad_import import match_parts
    mparts = (meta or {}).get("parts") or []
    mapping, _ = match_parts(list(part_names), mparts)
    pos = {n: i for i, n in enumerate(part_names)}
    return {mi: pos[pn] for pn, mi in mapping.items()}


def used_tokens(study, parts: list, meta: dict | None) -> tuple[list[str], list[str]]:
    """Tokens of faces and bodies from the table of the CAD (the last export = the current geometry of the study) which the study uses:
    the faces of the selections and the bodies of all parts. The CAD add-in translates them to the new version (findEntityByToken).
    """
    from .faces import Geometry
    rows = face_rows(meta)
    if not rows:
        return [], []
    geo = Geometry(parts)
    names = [p.name for p in parts]
    rp = row_part_map(meta, names)
    wanted = {(f.part, round(f.area, 6), tuple(round(c, 6) for c in f.centroid))
              for s in study.selections.values() for f in s.faces}
    face_tok = []
    for k, r in enumerate(rows):
        if "token" not in r:
            continue
        pi = rp.get(r.get("part"))
        pf = geo.part(pi) if pi is not None else None
        if pf is None:
            continue
        for fp in pf.faces:
            key = (pf.name, round(float(fp["area"]), 6), tuple(round(float(c), 6) for c in fp["centroid"]))
            if key in wanted and _score(r, fp, pf.size) <= BIND_TOL:
                face_tok.append(r["token"])
                break
    body_tok = [p["token"] for p in (meta or {}).get("parts") or []
                if isinstance(p, dict) and isinstance(p.get("token"), str) and TOKEN_RE.match(p["token"])]
    return face_tok, body_tok
