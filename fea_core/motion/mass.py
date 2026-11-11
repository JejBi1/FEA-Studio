"""Mass properties of the bodies of the mechanism from the CAD (OCP volume integration × ρ of the material of the part).

All in the units of the project: mass [t], centroid [mm], tensor of inertia about the centroid in the global axes of the position of the CAD
[t·mm²] (OCCT GProp_GProps.MatrixOfInertia = tensor about the centroid, verified on a block, also with negative deviation
moments). Printed parts (material with E_z) INDICATIVELY without the mesh: the shell fraction f ≈ Σ A_face·t_shell / V
(t = max(perimeters·line·√(1−c²), layers·h·|c|), c = n·build direction – as printing.region), density
ρ·(f + (1 − f)·infill), centroid and inertia of the full part × ratio of densities (decision F8: question 16 variant a).
The left-out parts (Study.ignored_parts) and the parts of the bolts (BoltSpec.cad_parts) are added to the nearest body
(include_extra=True). Body.mass / com / inertia override the calculation.
"""
from __future__ import annotations

import math

import numpy as np


class MassError(ValueError):
    """Missing density of the material, an unknown part of the body…"""


def part_props(shape) -> tuple[float, np.ndarray, np.ndarray]:
    """(volume [mm³], centroid [mm], tensor of inertia about the centroid for the density 1 [mm⁵])."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, p)
    c = p.CentreOfMass()
    M = p.MatrixOfInertia()
    J = np.array([[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    return float(p.Mass()), np.array([c.X(), c.Y(), c.Z()]), J


def printed_fraction(part, spec) -> float:
    """Estimate of the share of the full material of a printed part without a mesh: shell + infill (indicative)."""
    from ..geometry import face_props
    from ..printing.infill import density_ratio
    if not spec.enabled:
        return 1.0
    V = abs(part.volume())
    if V <= 0:
        return 1.0
    b = np.asarray(spec.build_dir, float)
    b = b / max(np.linalg.norm(b), 1e-300)
    wall = spec.walls * spec.line_width
    Vs = 0.0
    for f in part.faces():
        p = face_props(f)
        n = np.asarray(p["normal"], float)
        c = float(abs(n @ b)) if np.linalg.norm(n) > 0 else 0.0
        layers = spec.top_layers if (n @ b) > 0 else spec.bottom_layers
        t = max(wall * math.sqrt(max(1.0 - c * c, 0.0)), layers * spec.layer_height * c)
        Vs += p["area"] * t
    fs = min(Vs / V, 1.0)
    r = density_ratio(spec.pattern, spec.infill)
    return fs + (1.0 - fs) * r


def combine(items: list[tuple[float, np.ndarray, np.ndarray]]) -> tuple[float, np.ndarray, np.ndarray]:
    """Sum (mass, centroid, tensor about the centroid) – the Steiner theorem."""
    m = sum(i[0] for i in items)
    if m <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    c = sum(i[0] * i[1] for i in items) / m
    J = np.zeros((3, 3))
    for mi, ci, Ji in items:
        d = ci - c
        J += Ji + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return m, c, J


def _nearest_body(part, bodies_parts: dict[str, list]) -> str | None:
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    best, bd = None, np.inf
    for name, plist in bodies_parts.items():
        for p in plist:
            d = BRepExtrema_DistShapeShape(part.shape, p.shape)
            if d.IsDone() and d.Value() < bd:
                best, bd = name, d.Value()
    return best


def body_properties(study, parts, include_extra: bool = True) -> dict:
    """Mass properties of all bodies of study.mechanism: {body: {mass, com, inertia, parts, extra, printed,
    source}} + "_warnings", "_check" (the sum of the masses against the CAD).
    """
    mech = study.mechanism
    by_name = {p.name: p for p in parts}
    warnings: list[str] = []
    out: dict = {}
    cache: dict[str, tuple] = {}

    def props_of(pname: str):
        if pname in cache:
            return cache[pname]
        p = by_name[pname]
        try:
            mat = study.material_of(pname)
        except KeyError:
            raise MassError(f"Part '{pname}' has no material – the body mass cannot be computed") from None
        if not mat.rho:
            raise MassError(f"Material '{mat.name}' of part '{pname}' has no density ρ – enter it (t/mm³)")
        V, c, J1 = part_props(p.shape)
        k = 1.0
        printed = False
        if mat.is_print:
            k = printed_fraction(p, study.printing.spec(pname))
            printed = True
        rho = mat.rho * k
        cache[pname] = (rho * V, c, rho * J1, printed, k)
        return cache[pname]

    assigned: set[str] = set()
    bparts: dict[str, list] = {}
    for b in mech.bodies:
        missing = [n for n in b.parts if n not in by_name]
        if missing:
            raise MassError(f"Body '{b.name}': parts {', '.join(missing)} are not in the geometry")
        dup = [n for n in b.parts if n in assigned]
        if dup:
            raise MassError(f"Parts {', '.join(dup)} are in several bodies (body '{b.name}')")
        assigned |= set(b.parts)
        bparts[b.name] = [by_name[n] for n in b.parts]
    extra: dict[str, list[str]] = {b.name: [] for b in mech.bodies}
    if include_extra:
        extras = list(study.ignored_parts) + [p for bs in study.active_bolts() for p in bs.cad_parts]
        movable = {b.name: bparts[b.name] for b in mech.bodies if not b.ground and bparts[b.name]}
        for pname in dict.fromkeys(extras):
            if pname in assigned or pname not in by_name:
                continue
            tgt = _nearest_body(by_name[pname], {**movable, **{b.name: bparts[b.name] for b in mech.bodies if b.ground}})
            if tgt is not None:
                extra[tgt].append(pname)
                assigned.add(pname)
    for b in mech.bodies:
        names = list(b.parts) + extra[b.name]
        items = []
        printed = []
        for n in names:
            try:
                m, c, J, pr, k = props_of(n)
            except MassError:
                if b.ground:                     # the frame does not need the mass
                    continue
                raise
            items.append((m, c, J))
            if pr:
                printed.append(f"{n} ({100 * k:.0f} % of solid)")
        m, c, J = combine(items) if items else (0.0, np.zeros(3), np.zeros((3, 3)))
        src = "cad"
        if b.mass is not None:
            m, src = float(b.mass), "user"
        if b.com is not None:
            c, src = np.asarray(b.com, float), "user"
        if b.inertia is not None:
            J, src = np.asarray(b.inertia, float).reshape(3, 3), "user"
        if not b.ground and not m > 0:
            raise MassError(f"Body '{b.name}' has no mass (no parts and no mass given)")
        out[b.name] = {"mass": m, "com": c, "inertia": J, "parts": list(b.parts), "extra": extra[b.name],
                       "printed": printed, "source": src}
        if printed and not b.ground:
            warnings.append(f"Body '{b.name}': the mass of printed parts {', '.join(printed)} is only approximate (shell and infill fraction estimated without a mesh)")
    # check: the sum of the masses against the CAD
    try:
        cad_total = sum(props_of(n)[0] for n in by_name if n not in study.ignored_parts or include_extra)
    except MassError:
        cad_total = None
    unused = [n for n in by_name if n not in assigned and n not in study.ignored_parts]
    if unused:
        warnings.append(f"Parts outside the mechanism bodies (not included in the motion): {', '.join(unused)}")
    total = sum(v["mass"] for v in out.values())
    out["_check"] = {"bodies_t": total, "cad_t": cad_total, "unassigned": unused}
    out["_warnings"] = warnings
    return out
