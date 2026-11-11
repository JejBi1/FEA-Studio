"""Library of the coefficients of friction with sources (specification F8, table 3.3.3) + the user library.

Built-in data `fea_core/data/friction.json`; the user one `config.user_config_dir()/friction.json` (a list of records in the
same form). The user one is read as robustly as the library of materials: a bad record is skipped (it stays in the file),
an unreadable file → FrictionLibError when saving (it is not overwritten); a built-in record cannot be overwritten.

Choice of μ by the purpose (RULES): transmission of the force by friction (slip of the joint) → μ_min, resistance of a mechanism → μ_max; it is advisable to compute
both limits as variants.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "friction.json"

PURPOSES = {"slip": "joint slip (force transfer by friction) → μ_min",
            "resistance": "mechanism resistance (drive force, assembly, jamming) → μ_max"}

RULES = [
    "Transmission of force by friction (bolted joint F_Q, press fit, brake, snap-fit against pull-out) → μ_min.",
    "Drive force, assembly force, jamming of a guide, tightening torque → μ_max and μs; rolling bearings at start-up 2×.",
    "Self-locking that must hold (screw, wedge) → verify with μ_min.",
    "It is recommended to compute both limits as variants – μ is an estimate.",
    "Plastics: μ falls with load (PLA μs 0.30 → 0.20 at 30 → 70 N), rises sharply near Tg, the PV limit applies.",
    "Pairs without a source (PLA – PLA, rubber – steel, FR4 – steel): measure on an inclined plane (μs = tan α).",
]

FAMILIES = {          # family: (description, regular expression on the name of the material – lower case)
    "steel": ("ocel", r"ocel|steel|nerez|stainless|\bs\d{3}\b|\bc\d{2}e?\b|\d{2}crmo|\b1\.[04]\d{3}\b|\b1[1-7]\s?\d{3}\b"),
    "aluminium": ("hliník", r"hlin[ií]k|alumin|\bal\b|en ?aw|\baw-|\b(?:2024|5083|6060|6061|6082|7075)\b|alsi|almg"),
    "brass": ("mosaz", r"mosaz|brass|cuzn"),
    "POM": ("POM", r"\bpom\b|delrin|acetal|polyoxymet"),
    "PTFE": ("PTFE", r"ptfe|teflon"),
    "PETG": ("PETG", r"\bpetg\b"),
    "PLA": ("PLA", r"\bpla\b"),
    "ABS": ("ABS", r"\babs\b"),
    "PA": ("PA", r"\bpa\s?\d*\b|nylon|polyamid"),
    "rubber": ("pryž", r"pryž|guma|rubber|epdm|\bnbr\b|silikon"),
    "FR4": ("FR4", r"\bfr-?4\b|\bdps\b|\bpcb\b"),
}


class FrictionLibError(ValueError):
    pass


@dataclass
class FrictionPair:
    name: str
    a: str
    b: str
    lubricated: bool = False
    mu_s: list | None = None              # [min, max] from the sources
    mu_k: list | None = None
    mu_min: float | None = None           # prokluz spoje
    mu_max: float | None = None           # odpor mechanismu
    note: str = ""
    source: str = ""
    unverified: list = field(default_factory=list)
    surface_class: str = ""
    builtin: bool = False

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("name", "a", "b", "lubricated", "mu_s", "mu_k", "mu_min", "mu_max",
                                           "note", "source", "unverified")}
        if self.surface_class:
            d["surface_class"] = self.surface_class
        return d

    def mu(self, purpose: str = "slip") -> float:
        """μ by the purpose: slip → μ_min, resistance → μ_max (missing → FrictionLibError with an instruction to measure it)."""
        if purpose not in PURPOSES:
            raise FrictionLibError(f"Purpose must be {' | '.join(PURPOSES)}")
        v = self.mu_min if purpose == "slip" else self.mu_max
        if v is None:
            raise FrictionLibError(f"Pair “{self.name}” has no {('μ_min' if purpose == 'slip' else 'μ_max')} – measure it (inclined plane: μs = tan α) and save it to the user library")
        return float(v)

    def unverified_value(self, purpose: str) -> bool:
        return ("mu_min" if purpose == "slip" else "mu_max") in self.unverified


def _range(name: str, v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v, v]
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise FrictionLibError(f"{name}: [min, max] expected")
    lo, hi = float(v[0]), float(v[1])
    if not 0 <= lo <= hi <= 3:
        raise FrictionLibError(f"{name}: range must satisfy 0 ≤ min ≤ max ≤ 3")
    return [lo, hi]


def _value(name: str, v):
    if v is None:
        return None
    x = float(v)
    if isinstance(v, bool) or not 0 <= x <= 3:
        raise FrictionLibError(f"{name} must be in [0; 3]")
    return x


def pair_from_dict(d: dict, builtin: bool = False) -> FrictionPair:
    """A verified record (unknown keys are ignored – a file from a newer version)."""
    if not isinstance(d, dict):
        raise FrictionLibError("Record must be an object")
    name = str(d.get("name", "")).strip()
    if not name:
        raise FrictionLibError("Record must have a name")
    p = FrictionPair(name, str(d.get("a", "")).strip(), str(d.get("b", "")).strip(), bool(d.get("lubricated", False)),
                     _range("mu_s", d.get("mu_s")), _range("mu_k", d.get("mu_k")), _value("mu_min", d.get("mu_min")),
                     _value("mu_max", d.get("mu_max")), str(d.get("note", "")), str(d.get("source", "")),
                     [str(x) for x in d.get("unverified") or []], str(d.get("surface_class", "")), builtin)
    if p.mu_min is not None and p.mu_max is not None and p.mu_min > p.mu_max:
        raise FrictionLibError(f"{name}: μ_min > μ_max")
    return p


_BUILTIN: list[FrictionPair] | None = None


def builtin_pairs() -> list[FrictionPair]:
    global _BUILTIN
    if _BUILTIN is None:
        data = json.loads(DATA.read_text(encoding="utf-8"))
        _BUILTIN = [pair_from_dict(d, builtin=True) for d in data["pairs"]]
    return list(_BUILTIN)


def user_library_path() -> Path:
    from ..config import user_config_dir
    return user_config_dir() / "friction.json"


def _raw(strict: bool) -> list:
    p = user_library_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        if strict:
            raise FrictionLibError(f"Cannot read user friction library {p} ({e}) – fix or rename it, otherwise saving would overwrite it") from None
        return []
    return data if isinstance(data, list) else []


def user_pairs() -> list[FrictionPair]:
    """User pairs; bad records are skipped."""
    out = []
    for d in _raw(strict=False):
        try:
            out.append(pair_from_dict(d))
        except (FrictionLibError, TypeError, ValueError):
            continue
    return out


def pairs() -> list[FrictionPair]:
    """Built-in + user pairs (the user one does not overwrite the built-in)."""
    out = builtin_pairs()
    names = {p.name for p in out}
    for p in user_pairs():
        if p.name not in names:
            out.append(p)
            names.add(p.name)
    return out


def get(name: str) -> FrictionPair:
    for p in pairs():
        if p.name == name:
            return p
    low = str(name).strip().lower()
    for p in pairs():
        if p.name.lower() == low:
            return p
    raise FrictionLibError(f"Pair “{name}” is not in the friction library (pairs: {', '.join((p.name for p in pairs()))})")


def mu(name: str, purpose: str = "slip") -> float:
    """μ of the pair by the purpose: slip (slip of the joint, μ_min) | resistance (resistance of the mechanism, μ_max)."""
    return get(name).mu(purpose)


def _write(raw: list) -> None:
    p = user_library_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def save_pair(d: dict | FrictionPair) -> FrictionPair:
    """Adds / replaces a user pair (a built-in one cannot be overwritten; unreadable records stay in the file)."""
    p = pair_from_dict(d.to_dict() if isinstance(d, FrictionPair) else d)
    if p.name in {b.name for b in builtin_pairs()}:
        raise FrictionLibError(f"“{p.name}” is a built-in pair – choose another name")
    raw = [x for x in _raw(strict=True) if not (isinstance(x, dict) and x.get("name") == p.name)]
    _write(raw + [p.to_dict()])
    return p


def delete_pair(name: str) -> None:
    if name in {b.name for b in builtin_pairs()}:
        raise FrictionLibError(f"“{name}” is a built-in pair – cannot be deleted")
    raw = _raw(strict=True)
    keep = [x for x in raw if not (isinstance(x, dict) and x.get("name") == name)]
    if len(keep) == len(raw):
        raise FrictionLibError(f"User pair “{name}” does not exist")
    _write(keep)


def family(material) -> str | None:
    """Family of the material by name (str or an object with .name): steel | aluminium | brass | POM | PTFE | PETG | PLA |
    ABS | PA | rubber | FR4; a family given directly ("steel", "ocel") also applies; unknown → None.
    """
    text = str(getattr(material, "name", material) or "").strip()
    low = text.lower()
    for k, (cz, _) in FAMILIES.items():
        if low in (k.lower(), cz.lower()):
            return k
    for k, (_, rx) in FAMILIES.items():
        if re.search(rx, low):
            return k
    return None


def suggest(material_a, material_b, lubricated: bool | None = None, surface_class: str | None = None) -> list[dict]:
    """Pairs of the library for two materials (names, Material objects or families). Sorted: first the match of the lubrication
    (lubricated None = dry first), the surface classes A–D (EN 1090-2) only for steel – steel and only on request
    surface_class or at the end. An empty list = the pair is not in the library (measure it).
    """
    fa, fb = family(material_a), family(material_b)
    out = []
    if fa is None or fb is None:
        return out
    want = {fa, fb}
    for p in pairs():
        if {p.a, p.b} != want:
            continue
        if p.surface_class and surface_class and p.surface_class.upper() != str(surface_class).upper():
            continue
        score = 0
        if p.surface_class:
            score += -10 if surface_class else 10
        if p.lubricated != bool(lubricated):
            score += 5
        if p.mu_min is None and p.mu_max is None:
            score += 1
        out.append((score, p))
    out.sort(key=lambda t: t[0])
    res = []
    for _, p in out:
        d = p.to_dict()
        d["families"] = [fa, fb]
        d["builtin"] = p.builtin
        res.append(d)
    return res


def table() -> list[dict]:
    """All pairs as dictionaries (MCP, GUI, report)."""
    res = []
    for p in pairs():
        d = p.to_dict()
        d["builtin"] = p.builtin
        res.append(d)
    return res
