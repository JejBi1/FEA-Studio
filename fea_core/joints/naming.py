"""Recognition of standard fasteners by the name of the part in CAD.

Examples: "ISO 4017 - M8 x 30 - 8.8", "ISO 4762 M6x20", "ISO 4032 M8", "ISO 7089 8",
"DIN 933 M10x40 10.9", "Šroub M12x50 ISO 4014 A2-70", "ISO 4762 M8 x 25 --- 25N" (SolidWorks).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# norma -> (druh, hlava)
STANDARDS: dict[str, tuple[str, str]] = {
    # bolts with a hexagon head
    "ISO 4014": ("bolt", "hex"), "ISO 4017": ("bolt", "hex"), "ISO 4015": ("bolt", "hex"),
    "ISO 4016": ("bolt", "hex"), "ISO 4018": ("bolt", "hex"), "ISO 8765": ("bolt", "hex"),
    "ISO 8676": ("bolt", "hex"), "DIN 931": ("bolt", "hex"), "DIN 933": ("bolt", "hex"),
    "DIN 960": ("bolt", "hex"), "DIN 961": ("bolt", "hex"), "DIN 6914": ("bolt", "hex"),
    "EN 14399-4": ("bolt", "hex"), "EN 14399-3": ("bolt", "hex"), "EN 15048-1": ("bolt", "hex"),
    # cylindrical and other heads
    "ISO 4762": ("bolt", "socket"), "DIN 912": ("bolt", "socket"), "ISO 14579": ("bolt", "socket"),
    "ISO 7380": ("bolt", "socket"), "ISO 1207": ("bolt", "socket"), "ISO 14580": ("bolt", "socket"),
    "ISO 7045": ("bolt", "socket"), "ISO 7984": ("bolt", "socket"), "DIN 7984": ("bolt", "socket"),
    "ISO 10642": ("bolt", "countersunk"), "DIN 7991": ("bolt", "countersunk"),
    "ISO 7046": ("bolt", "countersunk"), "ISO 2009": ("bolt", "countersunk"),
    # fitted
    "ISO 7379": ("fitted", "socket"), "DIN 609": ("fitted", "hex"), "DIN 7968": ("fitted", "hex"),
    "DIN 610": ("fitted", "hex"),
    # matice
    "ISO 4032": ("nut", ""), "ISO 4033": ("nut", ""), "ISO 4034": ("nut", ""), "ISO 4035": ("nut", ""),
    "ISO 8673": ("nut", ""), "ISO 7040": ("nut", ""), "ISO 7042": ("nut", ""), "ISO 10511": ("nut", ""),
    "ISO 10512": ("nut", ""), "DIN 934": ("nut", ""), "DIN 985": ("nut", ""), "DIN 982": ("nut", ""),
    "DIN 439": ("nut", ""), "DIN 936": ("nut", ""), "DIN 1587": ("nut", ""), "DIN 6915": ("nut", ""),
    "EN 14399-4N": ("nut", ""), "DIN 6923": ("nut", ""), "ISO 4161": ("nut", ""),
    # washers
    "ISO 7089": ("washer", ""), "ISO 7090": ("washer", ""), "ISO 7091": ("washer", ""),
    "ISO 7092": ("washer", ""), "ISO 7093": ("washer_large", ""), "ISO 7094": ("washer_large", ""),
    "DIN 125": ("washer", ""), "DIN 126": ("washer", ""), "DIN 9021": ("washer_large", ""),
    "DIN 125-1": ("washer", ""), "DIN 125-2": ("washer", ""),     # parts of the standard (SOLIDWORKS Toolbox "DIN 125-1 A 8,4")
    "DIN 433": ("washer", ""), "DIN 433-1": ("washer", ""), "DIN 433-2": ("washer", ""),
    "DIN 6916": ("washer", ""), "EN 14399-6": ("washer", ""),
    "DIN 127": ("washer", ""), "DIN 6798": ("washer", ""),
    # threaded rods
    "DIN 975": ("rod", ""), "DIN 976": ("rod", ""),
}

KIND_LABELS = {"bolt": "bolt", "fitted": "fitted bolt", "nut": "nut", "washer": "washer",
               "washer_large": "washer (large)", "rod": "threaded rod"}

_KEYWORDS = [
    (r"\b(matice|matka|nut|mutter)\b", "nut"),
    (r"\b(podlo[žz]ka|washer|scheibe)\b", "washer"),
    (r"\b(l[íi]covan\w*|fitted|passschraube|shoulder)\b", "fitted"),
    (r"\b(z[áa]vitov[áa] ty[čc]|threaded rod|gewindestange|stud)\b", "rod"),
    (r"\b(imbus|inbus|socket|innensechskant)\b", "socket"),
    (r"\b([šs]roub|bolt|screw|schraube)\b", "bolt"),
]

_CLASS_RE = re.compile(r"(?<![\d.])(4\.6|4\.8|5\.6|5\.8|6\.8|8\.8|9\.8|10\.9|12\.9)(?![\d.])")
_STAINLESS_RE = re.compile(r"\b(A[1-5])\s*(?:-\s*(50|70|80))?\b", re.I)
_STD_RE = re.compile(r"\b(ISO|DIN|EN|CSN|ČSN)\s*[-_]?\s*(\d{3,5})(?:\s*-\s*(\d)(?!\d))?", re.I)
_SIZE_RE = re.compile(r"\bM\s*(\d+(?:\.\d+)?)(?:\s*[xX×]\s*(\d+(?:\.\d+)?))?(?:\s*[xX×]\s*(\d+(?:\.\d+)?))?")


@dataclass
class FastenerName:
    kind: str                  # bolt | fitted | nut | washer | washer_large | rod
    standard: str = ""         # "ISO 4017"
    d: float | None = None     # nominal diameter of the thread
    pitch: float | None = None  # fine thread only
    length: float | None = None
    strength: str | None = None
    head: str = "hex"          # hex | socket | countersunk
    source: str = ""           # original name

    @property
    def size(self) -> str:
        return f"M{self.d:g}" if self.d else ""

    def describe(self) -> str:
        parts = [KIND_LABELS.get(self.kind, self.kind)]
        if self.standard:
            parts.append(self.standard)
        if self.d:
            s = f"M{self.d:g}"
            if self.pitch:
                s += f"×{self.pitch:g}"
            if self.length:
                s += f"×{self.length:g}"
            parts.append(s)
        if self.strength:
            parts.append(self.strength)
        return " ".join(parts)


def _normalize(name: str) -> str:
    s = name.replace("×", "x").replace("–", "-").replace("_", " ")
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)            # decimal comma
    s = re.sub(r":\d+$", "", s.strip())              # Fusion: instance ":1"
    s = re.sub(r"\s*\(\d+\)$", "", s)               # FEA Studio: repeated part "(2)"
    return s


def parse_name(name: str) -> FastenerName | None:
    """Recognises a fastener from the name; None if the name does not look like a standard part."""
    s = _normalize(name)
    kind, head, std = None, "hex", ""
    m = _STD_RE.search(s)
    std_end = 0
    if m:
        org = m.group(1).upper().replace("ČSN", "CSN")
        std = f"{org} {m.group(2)}" + (f"-{m.group(3)}" if m.group(3) else "")
        std_end = m.end()
        info = STANDARDS.get(std)
        if info is None and m.group(3):            # "ISO 7089 - 8": the number after the dash is the size, not a part of the standard
            std = f"{org} {m.group(2)}"
            std_end = m.end(2)
            info = STANDARDS.get(std)
        if info:
            kind, head = info[0], info[1] or "hex"
    low = s.lower()
    if kind is None:
        for pat, k in _KEYWORDS:
            if re.search(pat, low):
                if k == "socket":
                    kind, head = "bolt", "socket"
                else:
                    kind = k
                break
    d = pitch = length = None
    ms = _SIZE_RE.search(s)
    if ms:
        d = float(ms.group(1))
        a = float(ms.group(2)) if ms.group(2) else None
        b = float(ms.group(3)) if ms.group(3) else None
        if b is not None:
            if a <= 4 and a < d / 2:             # fine thread "M12 x 1.5 x 40"
                pitch, length = a, b
            else:                                # SolidWorks Toolbox "M12 x 80 x 30" = length × length of the thread
                length = a
        elif a is not None:
            if a <= 4 and a < d / 2:
                pitch = a
            else:
                length = a
    elif kind in ("washer", "washer_large", "nut") and m:
        # "ISO 7089 8" / "ISO 7089 - 8 - 200 HV": the number after the standard = nominal size; DIN 125 / 433 have it before
        # shape A (without chamfer) / B (with chamfer): "DIN 125-1 A 8,4", "DIN 125 - B 10,5"
        rest = s[std_end:]
        mn = re.match(r"[\s\-]*(?:[AB]\b[\s\-]*)?(\d+(?:\.\d+)?)", rest)
        if mn:
            d = float(mn.group(1))
            if kind.startswith("washer"):        # "8.4" = inner diameter of the washer → size M8
                from .tables import WASHER
                d = next((float(k) for k, (d1, _, _) in WASHER.items() if abs(d1 - d) < 1e-6), d)
    if kind is None:
        if d is None or length is None:
            return None
        kind = "bolt"                            # "M8x30 8.8" without a standard and keyword – a bolt
    if d is None:
        return None
    strength = None
    mc = _CLASS_RE.search(s[ms.end():] if ms else s) or _CLASS_RE.search(s)
    if mc and kind in ("bolt", "fitted", "rod"):
        strength = mc.group(1)
    ma = _STAINLESS_RE.search(s)
    if ma and kind in ("bolt", "fitted", "rod", "nut"):
        strength = f"{ma.group(1).upper()}-{ma.group(2) or '70'}"
    return FastenerName(kind, std, d, pitch, length, strength, head, name)
