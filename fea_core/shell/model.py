"""Settings of the shell model (without OCP, without Gmsh) – a part of the study serialisable to JSON.

The shell model replaces the thin-walled parts of the 3D study by the mid surface with a thickness (elements S6 in ccx).
Source of the geometry of a part:
  • a plate from a body – two parallel planar faces of the largest area with opposite normal, constant thickness
    (mid face = the shifted upper face); also a sheet with bends (cylindrical bends, mid face via an offset),
  • a planar body from STEP (a face / shell without volume) – the thickness is given by the user, the position of the face
    relative to the shell according to `offset`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

OFFSETS = {"mid": 0.0, "top": 0.5, "bot": -0.5}     # ccx *SHELL SECTION, OFFSET (in multiples of the thickness)


@dataclass
class ShellSettings:
    enabled: bool = False
    # part → thickness [mm] | None (= thickness of the plate / sheet recognised in the body). Empty = all parts
    # (the bodies must be plates, the planar parts from STEP then need a thickness).
    parts: dict[str, float | None] = field(default_factory=dict)
    size: float | None = None        # velikost prvku [mm]; None = Study.mesh.size
    offset: str = "mid"              # planar parts from STEP: CAD face = mid | upper (top) | lower (bot) face
    element: str = "S6"              # S6 (tri6, expansion C3D15); S8R not yet (quad8 mesh, hourglass)

    def to_dict(self) -> dict:
        return {"enabled": bool(self.enabled),
                "parts": {k: (None if v is None else float(v)) for k, v in self.parts.items()},
                "size": None if self.size is None else float(self.size), "offset": self.offset,
                "element": self.element}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ShellSettings":
        d = d or {}
        parts = {str(k): (None if v is None else float(v)) for k, v in (d.get("parts") or {}).items()}
        size = d.get("size")
        return cls(enabled=bool(d.get("enabled", False)), parts=parts,
                   size=None if size is None else float(size), offset=str(d.get("offset", "mid")),
                   element=str(d.get("element", "S6")))

    def check(self) -> list[str]:
        """Errors of the settings (in Czech); empty list = all right."""
        out = []
        if self.offset not in OFFSETS:
            out.append(f"Unknown shell reference surface position '{self.offset}' (mid | top | bot)")
        if self.element != "S6":
            out.append(f"Shell element type '{self.element}' is not supported (S6 only)")
        if self.size is not None and not self.size > 0:
            out.append("Shell element size must be positive")
        for k, v in self.parts.items():
            if v is not None and not v > 0:
                out.append(f"Shell thickness of part '{k}' must be positive")
        return out


class ShellError(RuntimeError):
    """Error of the shell model (in Czech, for the user)."""
