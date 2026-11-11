"""Shared helpers F8 for MCP and the report: contact pairs (types, μ with its source, K, λ), excitation and profiles of the dynamics,
added mass. Without GUI and without OCP (OCP only lazily in the search of the touches).
"""
from __future__ import annotations

import math

import numpy as np

from .session import FeaError

# ===================================================================== kontakty

TYPE_ALIASES = {
    "bonded": "bonded", "pevne spojeno": "bonded", "rigidly bonded": "bonded", "spojeno": "bonded", "tie": "bonded",
    "no_separation": "no_separation", "no separation": "no_separation", "no separation": "no_separation",
    "no separation (sliding)": "no_separation", "sliding": "no_separation",
    "frictionless": "frictionless", "frictionless": "frictionless", "frictionless": "frictionless",
    "frictionless contact": "frictionless", "frictionless contact": "frictionless", "separation": "frictionless",
    "frictional": "frictional", "s trenim": "frictional", "with friction": "frictional", "friction": "frictional",
    "friction": "frictional", "frictional contact": "frictional", "styk s trenim": "frictional", "friction": "frictional",
    "rough": "rough", "no slip": "rough", "no-slip contact": "rough",
    "interference": "interference", "presah": "interference", "interference": "interference", "shrink_fit": "interference",
    "lisovany spoj": "interference", "press fit": "interference", "interference (press fit)": "interference",
    "compression_only": "compression_only", "compression only": "compression_only", "support": "compression_only",
    "compression only (support)": "compression_only", "compression only": "compression_only",
    "none": "none", "no interaction": "none", "free": "none",
    "pevně spojeno": "bonded", "bez odtrzeni": "no_separation", "bez odtržení": "no_separation",
    "bez odtržení (klouže)": "no_separation", "bez treni": "frictionless", "bez tření": "frictionless",
    "styk bez tření": "frictionless", "styk bez treni": "frictionless", "s třením": "frictional", "treni": "frictional",
    "tření": "frictional", "styk s třením": "frictional", "bez prokluzu": "rough", "styk bez prokluzu": "rough",
    "přesah": "interference", "lisovaný spoj": "interference", "přesah (lisovaný spoj)": "interference",
    "jen tlak": "compression_only", "podpora": "compression_only", "jen tlak (podpora)": "compression_only",
    "bez interakce": "none",
}
# colours of the types (pictures of the model and the report) – fixed assignment of the type, not the order
TYPE_COLORS = {"bonded": (0.55, 0.60, 0.70), "no_separation": (0.25, 0.55, 0.90), "frictionless": (0.0, 0.70, 0.72),
               "frictional": (0.93, 0.68, 0.10), "rough": (0.62, 0.40, 0.20), "interference": (0.85, 0.20, 0.60),
               "compression_only": (0.30, 0.72, 0.30), "none": (0.80, 0.80, 0.80)}
METHOD_LABELS = {"surface_to_surface": "surface-to-surface (penalty)", "mortar": "mortar",
                 "node_to_surface": "node-to-surface (penalty)"}
SPLIT_LABELS = {"separate": "parts of the pair meshed separately (non-conforming mesh, no *TIE)",
                "conformal_split": "conforming mesh with a split interface (matching nodes)"}
PURPOSE_LABELS = {"slip": "joint slip – μ_min", "resistance": "mechanism resistance – μ_max"}
INITIAL_LABELS = {"contact": "contact", "gap": "gap", "penetration": "penetration", None: "–"}


def contact_type(text) -> str:
    """Type of contact from the text (key CONTACT_TYPES, Czech name or alias)."""
    from ..contacts.model import CONTACT_TYPES, CONTACT_LABELS
    t = str(text or "").strip().lower().replace("-", "_")
    if t in CONTACT_TYPES:
        return t
    for k, lab in CONTACT_LABELS.items():
        if lab.lower() == t:
            return k
    t2 = t.replace("_", " ")
    if t2 in TYPE_ALIASES:
        return TYPE_ALIASES[t2]
    if t in TYPE_ALIASES:
        return TYPE_ALIASES[t]
    raise FeaError(f"Unknown contact type '{text}'. Types: " + ", ".join(f"{k} ({v})" for k, v in CONTACT_LABELS.items()))


def type_label(kind: str) -> str:
    from ..contacts.model import CONTACT_LABELS
    return CONTACT_LABELS.get(kind, kind)


def mu_info(study, c) -> dict:
    """Coefficient of friction of the pair with the source: {"mu", "source", "library_pair", "range", "unverified"}.
    The same order as the calculation (contacts.inp.default_mu): the given μ → the pair of the library → by materials → estimate.
    """
    from ..contacts.model import MU_NONE, MU_ROUGH
    if c.kind == "rough":
        return {"mu": MU_ROUGH, "source": "rough contact (large μ = 10)"}
    if c.kind == "no_separation":
        return {"mu": MU_NONE, "source": "no separation (TIED) – technical μ 1e-4"}
    if c.kind != "frictional":
        changed = any(c.name in s.mu_change for s in study.analysis.sequence)
        return {"mu": MU_NONE if changed else None,
                "source": "frictionless (μ 1e-4 due to a friction change in the history)" if changed else "frictionless"}
    if c.mu is not None:
        return {"mu": float(c.mu), "source": "user-defined"}
    try:
        from ..calc import friction_lib as fl
        p = None
        if c.materials:
            p = fl.get(c.materials)
        else:
            from ..contacts.detect import pair_parts
            pa, pb = pair_parts(study, c)
            ma = study.material_of(sorted(pa)[0]).name
            mb = study.material_of(sorted(pb)[0]).name
            s = fl.suggest(ma, mb)
            if s:
                p = fl.get(s[0]["name"])
        if p is not None:
            v = p.mu(c.purpose)
            rng = [p.mu_min, p.mu_max]
            return {"mu": float(v), "library_pair": p.name, "range": rng,
                    "source": f"friction library: {p.name} ({PURPOSE_LABELS.get(c.purpose, c.purpose)})"
                              + (f"; source: {p.source}" if p.source else ""),
                    "unverified": bool(p.unverified_value(c.purpose))}
    except Exception:  # noqa: BLE001 – the library unavailable / a pair without a value → estimate
        pass
    v = 0.15 if c.purpose == "slip" else 0.3
    return {"mu": v, "source": f"default estimate {v:g} – the material pair is not in the friction library (measure: μs = tan α)",
            "unverified": True}


def pair_effective(study, c) -> dict:
    """K, λ (written explicitly) and μ of the pair as in the calculation."""
    out = {}
    try:
        from ..contacts.inp import pair_params
        pp = pair_params(study, None, c)
        out["K_N_mm3"], out["lambda_N_mm3"] = pp["K"], pp["lambda"]
    except Exception:  # noqa: BLE001 – a part without material
        out["K_N_mm3"] = out["lambda_N_mm3"] = None
    out["K_auto"] = c.stiffness is None
    out["lambda_auto"] = c.stick_slope is None
    return out


def pair_describe(study, c) -> dict:
    mi = mu_info(study, c)
    d = {"name": c.name, "a": c.a, "b": c.b or None, "type": c.kind, "type_label": type_label(c.kind),
         "enabled": c.enabled, "mu": mi.get("mu"), "mu_source": mi["source"], "purpose": c.purpose}
    if mi.get("range"):
        d["mu_range_library"] = mi["range"]
    if c.materials:
        d["friction_pair"] = c.materials
    d.update(pair_effective(study, c))
    for k, key in (("clearance", "clearance_mm"), ("slave", "slave"), ("initial", "initial_state"),
                   ("gap", "initial_gap_mm"), ("area", "area_mm2")):
        v = getattr(c, k)
        if v is not None:
            d[key] = v
    if c.adjust:
        d["adjust"] = True
    d["source"] = c.source
    return d


def find_pair(study, name: str):
    for c in study.contacts:
        if c.name == name:
            return c
    raise FeaError(f"Contact pair '{name}' does not exist (pairs: {', '.join((c.name for c in study.contacts)) or 'none'})")


def touching(sess, a: str, b: str, tol: float) -> dict | None:
    """Result of find_touching for a pair of parts (cache of the session – the pictures of the model and the detection)."""
    cache = sess.__dict__.setdefault("_f8_touch", {})
    key = (a, b, round(float(tol), 9))
    if key not in cache:
        from ..contacts.detect import find_touching
        r = find_touching(sess.parts, tol, names={a, b})
        hit = next((x for x in r if {x["a"], x["b"]} == {a, b}), None)
        if hit is not None and hit["a"] != a:
            hit = {**hit, "a": a, "b": b, "faces_a": hit["faces_b"], "faces_b": hit["faces_a"]}
        cache[key] = hit
    return cache[key]


# ===================================================================== dynamics – excitation and profiles

EXC_KINDS = {"force": "force", "force": "force", "force": "force", "base_acc": "base_acc", "zrychleni": "base_acc",
             "acceleration": "base_acc", "acceleration": "base_acc", "base_disp": "base_disp", "displacement": "base_disp",
             "displacement": "base_disp", "sila": "force", "síla": "force", "zrychlení": "base_acc", "posuv": "base_disp"}
EXC_LABELS = {"force": "force (load case × profile)", "base_acc": "base acceleration",
              "base_disp": "base displacement"}
EXC_KEYS = {"kind", "case", "dof", "direction", "table", "unit", "scale", "phase_deg", "phase", "sine_profile"}


def dof_of(v) -> int:
    if isinstance(v, str):
        k = v.strip().lower()
        if k in ("x", "y", "z"):
            return "xyz".index(k) + 1
        if k.isdigit():
            v = int(k)
    try:
        d = int(v)
    except (TypeError, ValueError):
        raise FeaError(f"Direction '{v}' must be 1|2|3 or x|y|z") from None
    if d not in (1, 2, 3):
        raise FeaError("The dof direction must be 1, 2 or 3 (x, y, z)")
    return d


def vec3(v, what: str) -> list[float]:
    try:
        a = [float(x) for x in v]
    except (TypeError, ValueError):
        raise FeaError(f"{what} must be [x, y, z]") from None
    if len(a) != 3 or not all(math.isfinite(x) for x in a) or not any(a):
        raise FeaError(f"{what} must be a nonzero vector [x, y, z]")
    return a


def table2(v, what: str, positive_x: bool = False) -> list[list[float]]:
    try:
        t = [[float(r[0]), float(r[1])] for r in v]
    except (TypeError, ValueError, IndexError):
        raise FeaError(f"{what} must be a list of pairs [[x, value], …]") from None
    if len(t) < 1:
        raise FeaError(f"{what} is empty")
    xs = [r[0] for r in t]
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise FeaError(f"{what}: x must increase")
    if positive_x and xs[0] <= 0:
        raise FeaError(f"{what}: frequencies must be positive")
    return t


def sine_profile_table(f_min: float, f_max: float, displacement_mm: float | None = None,
                       acceleration_g: float | None = None, segments: list | None = None) -> tuple[list, dict]:
    """Sinusoidal profile of the test (IEC 60068-2-6) → table of the acceleration of the base [[f Hz, a g], …] for the excitation base_acc.

    displacement_mm = amplitude of the displacement x̂ below the transition frequency, acceleration_g = amplitude of the acceleration above it
    (f_c = √(a/x̂)/(2π)); segments = in general [{"f_from_Hz", "f_to_Hz", "displacement_mm" | "acceleration_g"}, …].
    A section of a constant displacement: a = (2πf)²·x̂ / g (points every 1/60 decade – linear interpolation of the table).
    """
    from ..dynamics.model import G_MM
    info = {}
    if segments is None:
        if displacement_mm is None and acceleration_g is None:
            raise FeaError("sine_profile: give displacement_mm and/or acceleration_g (values from the standard / specification)")
        if displacement_mm is not None and acceleration_g is not None:
            fc = math.sqrt(float(acceleration_g) * G_MM / float(displacement_mm)) / (2 * math.pi)
            info["crossover_Hz"] = fc
            segments = [{"f_from_Hz": f_min, "f_to_Hz": min(max(fc, f_min), f_max), "displacement_mm": displacement_mm},
                        {"f_from_Hz": min(max(fc, f_min), f_max), "f_to_Hz": f_max, "acceleration_g": acceleration_g}]
        elif displacement_mm is not None:
            segments = [{"f_from_Hz": f_min, "f_to_Hz": f_max, "displacement_mm": displacement_mm}]
        else:
            segments = [{"f_from_Hz": f_min, "f_to_Hz": f_max, "acceleration_g": acceleration_g}]
    pts: dict[float, float] = {}
    for s in segments:
        try:
            a, b = float(s["f_from_Hz"]), float(s["f_to_Hz"])
        except (KeyError, TypeError, ValueError):
            raise FeaError("A sine profile segment needs f_from_Hz and f_to_Hz") from None
        if not 0 < a <= b:
            raise FeaError("Sine profile segment: 0 < f_from_Hz ≤ f_to_Hz")
        if b - a < 1e-12:
            continue
        if s.get("displacement_mm") is not None:
            x = float(s["displacement_mm"])
            n = max(2, int(math.ceil(60 * math.log10(b / a))) + 1)
            for f in np.geomspace(a, b, n):
                pts[round(float(f), 9)] = (2 * math.pi * f) ** 2 * x / G_MM
        elif s.get("acceleration_g") is not None:
            for f in (a, b):
                pts[round(f, 9)] = float(s["acceleration_g"])
        else:
            raise FeaError("A sine profile segment needs displacement_mm or acceleration_g")
    tab = [[f, pts[f]] for f in sorted(pts)]
    info["points"] = len(tab)
    return tab, info


def parse_excitation(item, cases: list[str], domain: str):
    """Item of the excitation (dict) → dynamics.model.Excitation; domain harmonic (table by f) | transient (by t)."""
    from ..dynamics.model import Excitation
    if not isinstance(item, dict):
        raise FeaError("The excitation must be an object {\"kind\": force|base_acc|base_disp, …}")
    bad = set(item) - EXC_KEYS
    if bad:
        raise FeaError(f"Unknown excitation keys: {', '.join(sorted(bad))} (allowed: {', '.join(sorted(EXC_KEYS))})")
    kind = EXC_KINDS.get(str(item.get("kind", "base_acc")).strip().lower())
    if kind is None:
        raise FeaError("excitation kind: force | base_acc | base_disp")
    e = Excitation(kind=kind)
    if kind == "force":
        case = item.get("case")
        if case not in cases:
            raise FeaError(f"Force excitation needs case = an existing load case (cases: {', '.join(cases) or 'none'})")
        e.case = case
        e.unit = "N"
        e.dof = None
    else:
        if item.get("direction") is not None:
            e.direction = vec3(item["direction"], "excitation direction")
            e.dof = None
        else:
            e.dof = dof_of(item.get("dof", 3))
        e.unit = str(item.get("unit") or ("g" if kind == "base_acc" else "mm")).strip().lower().replace("²", "2")
        if kind == "base_acc" and e.unit not in ("g", "mm/s2"):
            raise FeaError("base acceleration unit: g | mm/s2")
        if kind == "base_disp":
            e.unit = "mm"
    if item.get("sine_profile") is not None:
        if domain != "harmonic" or kind != "base_acc":
            raise FeaError("sine_profile only for a harmonic response with base_acc excitation")
        sp = item["sine_profile"]
        tab, _ = sine_profile_table(float(sp.get("f_min_Hz", 1.0)), float(sp.get("f_max_Hz", 2000.0)),
                                    sp.get("displacement_mm"), sp.get("acceleration_g"), sp.get("segments"))
        e.table = tab
        e.unit = "g"
    elif item.get("table") is not None:
        e.table = table2(item["table"], "excitation table", positive_x=(domain == "harmonic"))
    e.scale = float(item.get("scale", 1.0))
    ph = item.get("phase_deg", item.get("phase", 0.0))
    e.phase = float(ph or 0.0)
    if e.phase and domain != "harmonic":
        raise FeaError("The excitation phase only for a harmonic response")
    return e


def excitation_text(e, t=None) -> str:
    """Description of the excitation; t = translation of the sections (the report in another language – i18n.tr), MCP without translation."""
    t = t or (lambda s: s)
    if e.kind == "force":
        s = t(f"force – load case “{e.case}” × {e.scale:g}")
    else:
        d = (t(f"direction ({', '.join((f'{v:g}' for v in e.direction))})") if e.direction is not None
             else t(f"direction {'xyz'[int(e.dof or 3) - 1]}"))
        s = f"{t(EXC_LABELS[e.kind])}, {d}, {e.scale:g} {e.unit}"
    if e.table:
        s += t(f", history by table ({len(e.table)} points)")
    if e.phase:
        s += t(f", phase {e.phase:g}°")
    return s


PROFILE_PRESETS = {
    "IEC 60068-2-6": {
        "title": "IEC 60068-2-6:2007 (Fc) – sinusoidal vibration",
        "kind": "harmonic",
        "fields": {"f_min_Hz": None, "f_max_Hz": None, "displacement_mm": "displacement amplitude x̂ below the crossover frequency", "acceleration_g": "acceleration amplitude above the crossover frequency",
                   "sweep_rate_oct_min": "log sweep rate [oct/min]", "sweep_cycles": "number of sweep cycles per axis", "axes": "test axes (x, y, z)", "dwell": "dwell at resonance (duration, number of cycles)"},
        "formula": "crossover frequency f_c = √(a/x̂)/(2π) (a in mm/s²); constant displacement below f_c, constant acceleration above it",
        "tool": "fea_dyn_harmonic_set(f_min_Hz=…, f_max_Hz=…, sine_profile={\"displacement_mm\": …, "
                "\"acceleration_g\": …}, dof=…, sweep_rate_oct_min=…, sweeps=…)",
        "fatigue": "fatigue from the sweep dn/df = 60/(R·ln 2) cycles/Hz per pass",
    },
    "IEC 60068-2-64": {
        "title": "IEC 60068-2-64:2008 + A1:2019 (Fh) – broadband random vibration",
        "kind": "random",
        "fields": {"points": "ASD points [[f Hz, g²/Hz or null], …]", "slopes_db_oct": "segment slopes [dB/oct]",
                   "grms_g": "total g_rms value (check / rescaling)", "duration_h": "test duration per axis",
                   "axes": "test axes"},
        "formula": "W₂ = W₁·(f₂/f₁)^(s/3,0103); g_rms = √∫W df",
        "tool": "fea_dyn_random_set(points=[[f, W], …], slopes_db_oct=[…], grms_g=…, dof=…, duration_h=…)",
    },
    "IEC 60068-2-27": {
        "title": "IEC 60068-2-27 (Ea) – shock",
        "kind": "shock",
        "fields": {"pulse": "pulse shape halfsine | trapezoid | sawtooth (sawtooth with a final peak)",
                   "A_g": "peak acceleration [g]", "tau_s": "doba pulzu [s]", "count": "number of shocks per direction",
                   "directions": "directions (±x, ±y, ±z)"},
        "tool": "fea_dyn_shock_set(pulse=…, A_g=…, tau_s=…, dof=…)",
        "note": "check the currency of the edition (ed. 2008)",
    },
    "IEC 60068-2-47": {
        "title": "IEC 60068-2-47:2005 – mounting of the test specimen",
        "kind": "check",
        "fields": {"f1_fixture_Hz": "first natural frequency of the fixture (fea_modal_set on the fixture model)",
                   "f_max_test_Hz": "upper test frequency"},
        "note": "check that the fixture does not resonate in the test range (criterion per the standard / specification)",
    },
    "MIL-STD-810H": {
        "title": "MIL-STD-810H methods 514.8 (vibration) / 516.8 (shock)",
        "kind": "reference",
        "note": "reference only – take the profiles from the text of the standard after verification (functional shock = sawtooth with a final peak)",
    },
}
PRESET_NOTE = ("The presets are only the STRUCTURE of a test – enter the numerical values (frequencies, amplitudes, ASD, durations, counts) from the standard or the test specification (the program does not invent them).")


# ===================================================================== added mass

def mass_describe(m) -> dict:
    if m.kind == "point_mass":
        d = {"name": m.name, "kind": "point_mass", "selection": m.selection, "mass_kg": 1000.0 * m.mass,
             "coupling": m.coupling, "point_mm": m.point}
        if m.inertia is not None:
            d["inertia_kg_mm2"] = (1000.0 * np.asarray(m.inertia, float)).tolist()
        return d
    return {"name": m.name, "kind": "surface_mass", "selection": m.selection, "mass_per_area_kg_m2": m.mass_per_area}


def mass_text(m, t=None) -> str:
    """Description of the added mass; t = translation of the sections (the report), MCP without translation."""
    t = t or (lambda s: s)
    if m.kind == "point_mass":
        s = t(f"point mass {1000 * m.mass:g} kg ({('rigid' if m.coupling == 'rigid' else 'distributing')} coupling")
        s += t(f", centre of gravity ({'; '.join((f'{v:g}' for v in m.point))}) mm)") if m.point else t(", face centroid)")
        if m.inertia is not None:
            s += t(" + rotary inertia")
        return s
    return t(f"distributed mass {m.mass_per_area:g} kg/m²")
