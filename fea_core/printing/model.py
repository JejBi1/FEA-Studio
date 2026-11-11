"""Settings of the 3D printing in the study (F4, homogenised model, serialisable to JSON).

A part with printed material (Material.E_z ≠ None) is computed as a shell (perimeters × width of the line,
upper/lower layers × height of the layer) + a core with the effective properties of the infill. The material is assigned
per element according to the volume fraction of the element in the shell (fea_core.printing.region). Default values =
Bambu Studio, profile 0.20 mm Standard (2 perimeters, 5 upper and 3 lower layers, 15 % infill).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields


@dataclass
class PrintSpec:
    part: str
    build_dir: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])  # Z axis of the printer in the coordinates of the model
    layer_height: float = 0.2       # mm
    line_width: float = 0.42        # mm
    walls: int = 2                  # number of perimeters
    top_layers: int = 5
    bottom_layers: int = 3
    infill: float = 0.15            # density of the infill 0–1
    pattern: str = "grid"           # key in printing.infill.PATTERNS
    enabled: bool = True
    note: str = ""
    # thin places (F5): the interior narrower than thin_width after subtracting the perimeters is printed with a full infill (as the slicer –
    # a sparse infill does not fit there); thin_width = 0 → automatically 2 × width of the line. Older files without the key → True.
    solid_thin: bool = True
    thin_width: float = 0.0         # mm, 0 = auto
    # printing tolerance (F7): loss of material at the surface [mm] – the part is printed δ smaller on each side (undersized
    # walls, rounded edges of the beads); the layer δ below the surface is computed without material (conservative, region.assign)
    tolerance: float = 0.0

    @property
    def thin_limit(self) -> float:
        """Width of the interior [mm] below which the core is printed with a full infill."""
        return float(self.thin_width) if self.thin_width and self.thin_width > 0 else 2.0 * self.line_width

    @property
    def wall_thickness(self) -> float:
        return self.walls * self.line_width

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PrintSpec":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class CreepSettings:
    """Estimate of the creep (isochronous modulus, printing.props): the long-term load states are recomputed
    with the reduced stiffnesses of the printed materials (and the reduced strength – creep rupture).
    """
    enabled: bool = False
    hours: float = 1000.0           # duration of the load [h]
    temperature: float = 23.0       # temperature of the part [°C]
    cases: list[str] = field(default_factory=list)   # long-term (permanent) load states

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "CreepSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class PrintVariant:
    """Variant of the internal structure: overrides of the print settings per part (PrintSpec fields without part/note + "material" =
    another printed material of the part). A part without an override has the setting of the study. Computed on the same mesh (printing.variants).
    """
    name: str
    parts: dict[str, dict] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "parts": {k: dict(v) for k, v in self.parts.items()}, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "PrintVariant":
        return cls(str(d.get("name", "")), {k: dict(v) for k, v in (d.get("parts") or {}).items()},
                   str(d.get("note", "")))


AXIS_DIRS = {"+X": [1.0, 0.0, 0.0], "-X": [-1.0, 0.0, 0.0], "+Y": [0.0, 1.0, 0.0], "-Y": [0.0, -1.0, 0.0],
             "+Z": [0.0, 0.0, 1.0], "-Z": [0.0, 0.0, -1.0]}


@dataclass
class SweepSpec:
    """Sweep of the print settings (F5, printing.sweep): grid perimeters × infill × pattern × build direction, the same overrides
    for the parts `parts` (empty = all printed). An empty list of a parameter = the value from the study (unchanged).
    infill 0–1; build_dirs = keys of AXIS_DIRS (+X … −Z).
    """
    parts: list[str] = field(default_factory=list)
    walls: list[int] = field(default_factory=list)
    infill: list[float] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    build_dirs: list[str] = field(default_factory=list)
    max_variants: int = 60

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SweepSpec":
        d = d or {}
        return cls([str(x) for x in d.get("parts") or []], [int(x) for x in d.get("walls") or []],
                   [float(x) for x in d.get("infill") or []], [str(x) for x in d.get("patterns") or []],
                   [str(x) for x in d.get("build_dirs") or []], int(d.get("max_variants", 60)))

    def is_empty(self) -> bool:
        return not (self.walls or self.infill or self.patterns or self.build_dirs)


@dataclass
class PrintSettings:
    parts: dict[str, PrintSpec] = field(default_factory=dict)   # part -> print settings
    criterion: str = "tsai_wu"      # tsai_wu | tsai_hill (failure criterion)
    bins: int = 10                  # number of levels of the shell fraction in the element (materials per part = bins + 1)
    samples: int = 32               # number of sampling points in the element (shell fraction)
    tests: list[dict] = field(default_factory=list)   # tensile tests for the calibration (calibrate.TensileTest)
    materials: dict[str, dict] = field(default_factory=dict)   # material -> override of printing.props.PrintProps
    creep: CreepSettings = field(default_factory=CreepSettings)
    line_bins: int = 36             # number of classes of the line direction of the perimeters in the plane of the layer (180°/36 = 5°)
    variants: list[PrintVariant] = field(default_factory=list)   # variants of the internal structure (printing.variants)
    required_safety: float = 1.5    # required safety of the printing (Tsai-Wu/Hill) for the recommendation of the variant
    sweep: SweepSpec = field(default_factory=SweepSpec)   # sweep of the settings (printing.sweep)

    def spec(self, part: str) -> PrintSpec:
        """Settings of the part (default if the part has no settings)."""
        return self.parts.get(part) or PrintSpec(part)

    def to_dict(self) -> dict:
        return {"parts": {k: v.to_dict() for k, v in self.parts.items()}, "criterion": self.criterion,
                "bins": self.bins, "samples": self.samples, "tests": [dict(t) for t in self.tests],
                "materials": {k: dict(v) for k, v in self.materials.items()}, "creep": self.creep.to_dict(),
                "line_bins": self.line_bins, "variants": [v.to_dict() for v in self.variants],
                "required_safety": self.required_safety, "sweep": self.sweep.to_dict()}

    @classmethod
    def from_dict(cls, d: dict | None) -> "PrintSettings":
        d = d or {}
        return cls({k: PrintSpec.from_dict(v) for k, v in d.get("parts", {}).items()},
                   d.get("criterion", "tsai_wu"), int(d.get("bins", 10)), int(d.get("samples", 32)),
                   [dict(t) for t in d.get("tests", [])], {k: dict(v) for k, v in d.get("materials", {}).items()},
                   CreepSettings.from_dict(d.get("creep")), int(d.get("line_bins", 36)),
                   [PrintVariant.from_dict(v) for v in d.get("variants", [])], float(d.get("required_safety", 1.5)),
                   SweepSpec.from_dict(d.get("sweep")))


CREEP_WARNING = ("3D printing: the linear model does not capture plastic creep – for sustained loading of PLA expect creep and a loss of strength over time (especially above ~40 °C); the isochronous-modulus estimate (Creep in the 3D printing step) is only indicative.")
