"""Virtual bolts from the fastener components of the assembly that are not loaded as bodies (native CAD assemblies).

A fastener component = an occurrence whose file / name `naming.parse_name` recognises (bolt, fitted bolt, nut,
washer) – e.g. parts of the Content Center library "ISO 4762 M6 x 20". Only the kind, the dimensions from the name and the position of the occurrence
(origin [mm] and rotation) are written to the CAD metadata (`<STEP>.json`, key "fasteners"). `bolts_from_fasteners` builds from them
`BoltSpec` (a U1 beam as the other bolts, assessment according to VDI 2230 / EN 1993-1-8 in the Joints step):

- axis = the axis of the occurrence (the rotation column) on which the coaxial holes in the loaded parts lie; direction from the head = the side
  where the holes start at the origin of the occurrence (the origin of the library bolt lies on the axis at the head);
- clamped parts = parts intersected by lines parallel to the axis in the middle of the bearing annulus, from the bearing face of the head
  to the length of the bolt; washers (coaxial fasteners) are added from the ISO tables;
- coaxial nut → joint with a nut, otherwise a bolt into the thread in the last part (hole < 1.02 d); a through hole without a nut →
  a nut is assumed (warning).
A bolt that cannot be placed (no holes on the axis, missing size) is only reported – the calculation of the rest of the model goes on.
"""
from __future__ import annotations

import numpy as np

from . import tables
from .detect import _clamp_stack, _coaxial_cylinders, _hole
from .model import BoltSpec
from .naming import parse_name

KINDS = ("bolt", "fitted", "nut", "washer", "washer_large")


def classify(*names: str):
    """Fastener by the name of the file / occurrence → FastenerName (kind in KINDS, with a known d), otherwise None."""
    for n in names:
        if not n:
            continue
        fn = parse_name(n)
        if fn is not None and fn.kind in KINDS and fn.d:
            return fn
    return None


def fastener_record(fn, occurrence: str, source: str, matrix_cm: list[float]) -> dict:
    """Record for the CAD metadata: dimensions from the name + position of the occurrence (4×4 matrix by rows, the offset in cm)."""
    m = matrix_cm
    return {"name": occurrence, "source": source, "kind": fn.kind, "standard": fn.standard, "d": fn.d,
            "pitch": fn.pitch, "length": fn.length, "strength": fn.strength, "head": fn.head,
            "origin": [m[3] * 10.0, m[7] * 10.0, m[11] * 10.0],
            "axes": [[m[0], m[4], m[8]], [m[1], m[5], m[9]], [m[2], m[6], m[10]]]}


def _bbox_all(parts):
    from .detect import _bbox
    b = np.array([_bbox(p.shape) for p in parts])
    return float(np.linalg.norm(b[:, 3:].max(axis=0) - b[:, :3].min(axis=0)))


def _axis_score(parts, p0, e, d, reach):
    """Number of parts with a coaxial hole of diameter 0.8–1.6 d within the reach `reach` from p0 (both sides)."""
    tol = max(0.05, 0.02 * d)
    n = 0
    for part in parts:
        for c in _coaxial_cylinders(part, p0, e, tol):
            if 0.4 * d <= c["r"] <= 0.8 * d and min(abs(c["s0"]), abs(c["s1"]), abs(c["s0"] + c["s1"]) / 2) <= reach:
                n += 1
                break
    return n


def bolts_from_fasteners(parts, fasteners: list[dict], taken: set[str] | None = None,
                         preload: bool = True) -> tuple[list[BoltSpec], list[str]]:
    """Virtual bolts from the records `fastener_record` over the loaded parts → (bolts, warnings)."""
    warnings: list[str] = []
    taken = set(taken or ())
    bolts: list[BoltSpec] = []
    shafts = [f for f in fasteners if f.get("kind") in ("bolt", "fitted")]
    nuts = [f for f in fasteners if f.get("kind") == "nut"]
    washers = [f for f in fasteners if f.get("kind") in ("washer", "washer_large")]
    used_nuts: set[int] = set()
    for f in shafts:
        name = _unique(str(f.get("name") or "Bolt"), taken)
        try:
            b, w = _bolt(parts, f, nuts, washers, used_nuts, name)
        except Exception as e:  # noqa: BLE001 – one bolt must not stop the others
            b, w = None, [f"{name}: the bolt cannot be placed ({e})"]
        warnings += w
        if b is not None:
            taken.add(name)
            bolts.append(b)
    orphan = [f.get("name") for i, f in enumerate(nuts) if i not in used_nuts]
    if orphan:
        warnings.append("Nut without bolt (not loaded, no effect on the calculation): " + ", ".join(map(str, orphan[:8])))
    return bolts, warnings


def _unique(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    k = 2
    while f"{name} ({k})" in taken:
        k += 1
    return f"{name} ({k})"


def _on_axis(f: dict, p0, e, d) -> float | None:
    """Position of the component f on the axis of the bolt (s from p0), if its origin lies on the axis and some of its axes ∥ e."""
    o = np.asarray(f["origin"], float)
    v = o - p0
    s = float(v @ e)
    if np.linalg.norm(v - s * e) > 0.2 * d + 0.05:
        return None
    if not any(abs(abs(float(np.asarray(a) @ e)) - 1.0) < 1e-3 for a in f.get("axes", [])):
        return None
    return s


def _bolt(parts, f, nuts, washers, used_nuts, name):
    warnings: list[str] = []
    d = float(f["d"])
    L = float(f["length"]) if f.get("length") else None
    head = f.get("head") or "hex"
    std = f.get("standard") or ""
    p0 = np.asarray(f["origin"], float)
    axes = [np.asarray(a, float) / np.linalg.norm(a) for a in f.get("axes", [])]
    span = _bbox_all(parts)
    reach = (L or 3 * d) + 2 * d
    scored = sorted(((_axis_score(parts, p0, a, d, reach), i) for i, a in enumerate(axes)), reverse=True)
    if not scored or scored[0][0] == 0:
        return None, [f"{name}: there are no coaxial holes in the loaded parts on the occurrence axis – bolt skipped"]
    a = axes[scored[0][1]]
    tol = max(0.05, 0.02 * d)
    hd = tables.head_dims(d, head, std)
    # washers on the axis (not loaded) – thicknesses from the ISO tables
    on_w = []
    for w in washers:
        s = _on_axis(w, p0, a, d)
        if s is not None:
            dims = tables.washer_dims(float(w.get("d") or d), w.get("kind") == "washer_large")
            on_w.append((s, dims))
    clear = tables.CLEARANCE.get(tables.size_key(d), (1.1 * d,) * 3)[2]
    best = None
    for sign in (1.0, -1.0):                              # direction from the head towards the tip
        e = sign * a
        r_probe = 0.25 * (clear + hd["dw"])
        L_eff = L or span
        stack = _clamp_stack(parts, set(), p0, e, r_probe, -tol, L_eff + tol, None)
        ivs = sorted((lo, hi, n) for n, iv in stack.items() for lo, hi in iv if hi > tol)
        if not ivs:
            continue
        first = ivs[0][0]
        if first > 0.8 * d + sum(x[1]["h"] for x in on_w) + 2 * tol:   # head far from the first part
            continue
        if best is None or first < best[1]:
            best = (e, first, ivs)
    if best is None:
        return None, [f"{name}: the head does not seat on any part at the occurrence origin – bolt skipped"]
    e, s_plate, ivs = best
    on_w = [(float(s * (e @ a)), w) for s, w in on_w]      # position of the washer in the axes of the bolt (from the head)
    head_w = [w for s, w in sorted(on_w, key=lambda x: x[0]) if s <= s_plate + tol]
    nut_w = [w for s, w in sorted(on_w, key=lambda x: x[0]) if s > s_plate + tol]
    # continuous bundle from the bearing face
    plates, s_end = [], s_plate
    for lo, hi, n in ivs:
        if lo > s_end + max(tol, 0.5):                     # clearance in the bundle = end (or the play at the nut)
            break
        plates.append({"part": n, "s0": lo - s_plate, "s1": min(hi, (L or hi)) - s_plate})
        s_end = max(s_end, min(hi, L or hi))
    if L is not None:
        plates = [pl for pl in plates if pl["s0"] < L - s_plate - tol]
    pq = {p.name: p for p in parts}
    origin = p0 + s_plate * e
    for pl in plates:
        pl["hole"] = _hole(pq[pl["part"]], origin, e, pl["s0"], pl["s1"], tol)
    if not plates:
        return None, [f"{name}: no clamped parts found – bolt skipped"]
    # matice na ose
    nut = None
    for i, nf in enumerate(nuts):
        if i in used_nuts:
            continue
        s = _on_axis(nf, p0, e, d)
        if s is not None and s > s_plate and (L is None or s <= L + tol):
            if nut is None or s < nut[1]:
                nut = (i, s, nf)
    connection = "fitted" if f.get("kind") == "fitted" else "nut"
    thread_end = nut_s = 0.0
    nut_dw = 0.0
    last = plates[-1]
    if nut is not None:
        used_nuts.add(nut[0])
        nd = tables.nut_dims(d, nut[2].get("standard") or "")
        nut_dw = nut_w[-1]["d2"] if nut_w else nd["dw"]
        nut_s = last["s1"]
    elif last.get("hole") is not None and last["hole"] < 1.02 * d and len(plates) > 1:
        connection = "tapped"
        tapped = plates.pop()
        nut_s = tapped["s0"]
        thread_end = tapped["s1"] if L is None else min(tapped["s1"], L - s_plate)
        last = tapped
    else:
        nut_s = last["s1"]
        nd = tables.nut_dims(d, std)
        nut_dw = nut_w[-1]["d2"] if nut_w else nd["dw"]
        warnings.append(f"{name}: the nut is not in the assembly – an ISO 4032 nut under the last part is assumed")
    if connection == "tapped":
        nut_part = last["part"]
    else:
        nut_part = plates[-1]["part"]
    for pl in plates:
        if pl.get("hole") is None:
            warnings.append(f"{name}: no coaxial hole found in part '{pl['part']}'")
    strength = f.get("strength")
    note = "virtual bolt from the CAD assembly (the part is not loaded)"
    if not strength:
        strength = "8.8"
        note += "; the strength class is missing in the name – 8.8 is assumed"
        warnings.append(f"{name}: the strength class is missing in the name – 8.8 is assumed")
    dw_head = head_w[-1]["d2"] if head_w else hd["dw"]
    wl = [{"part": "", "side": "head", "h": w["h"], "d1": w["d1"], "d2": w["d2"]} for w in head_w]
    if connection != "tapped":
        wl += [{"part": "", "side": "nut", "h": w["h"], "d1": w["d1"], "d2": w["d2"]} for w in nut_w]
    b = BoltSpec(name=name, standard=std, size=d, pitch=f.get("pitch"), length=L, strength=strength, head=head,
                 connection=connection, origin=[float(v) for v in origin], axis=[float(v) for v in e],
                 head_part=plates[0]["part"], head_dw=float(dw_head), nut_part=nut_part, nut_s=float(nut_s),
                 nut_dw=float(nut_dw), thread_end=float(thread_end), plates=plates, washers=wl, cad_parts=[],
                 note=note)
    if connection == "tapped" and b.thread_end - b.nut_s < 0.8 * d:
        warnings.append(f"{name}: short thread engagement {b.thread_end - b.nut_s:.3g} mm (< 0.8 d)")
    return b, warnings
