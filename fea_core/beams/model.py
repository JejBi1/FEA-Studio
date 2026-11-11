"""Model of a frame system (frames, trusses, curved bars) in the study: `Study.frame` (Frame) and the loads of the bars
in the load states (FrameNodeLoad, FrameMemberLoad, FrameTemperature). The frame system itself is computed by its own solver
with the exact Timoshenko element (fea_core.beams.timoshenko); mixed models (bars + bodies) and the choice
`FrameSettings.solver = "ccx_u1"` use the U1 element in ccx (fea_core.beams.fem).

Axes of a bar (EN 1993): x = axis of the bar (from the node start to the end), z = "up" in the section (web of an I profile), y = z × x
(strong axis of an I profile: bending by the moment My, deflection in the direction z). Orientation: `up` = a vector whose projection perpendicular to the axis gives
z (default global Z; for a vertical bar global X); for an arc also "normal" (z = normal of the plane of the arc) or
"radial" (z = towards the centre of the arc – height of the section in the plane of curvature). `roll` = rotation of the section about the axis [°].
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields

DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")
LOCAL_NAMES = ("N", "Vy", "Vz", "T", "My", "Mz")


def _known(klass, d: dict | None):
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class SectionDef:
    """Section of the bar: from the library (`library` = "IPE 200"), parametric (`shape` + `dims`, see sections.parametric)
    or own (`props` = SectionProps.to_dict() – from a face of CAD or an outline).
    """
    name: str
    library: str | None = None
    shape: str | None = None
    dims: dict = field(default_factory=dict)
    props: dict | None = None

    def resolve(self):
        from . import sections as S
        from .props import SectionProps
        if self.props:
            return SectionProps.from_dict(self.props)
        if self.library:
            return S.get(self.library)
        if self.shape:
            return S.parametric(self.shape, **self.dims)
        raise ValueError(f"Cross-section '{self.name}' has no definition (library, shape or properties)")


@dataclass
class FrameNode:
    name: str
    xyz: list[float]


@dataclass
class FrameMember:
    name: str
    start: str                                  # node
    end: str
    section: str                                # Study.frame.sections
    material: str | None = None                 # Study.materials (None = default material of the study)
    up: list[float] | str | None = None         # orientation of the axis z of the section (vector | "normal" | "radial")
    roll: float = 0.0                           # rotation of the section about the axis [°]
    through: list[float] | None = None          # point on the arc (a curved bar by the circle start – through – end)
    releases_start: list[bool] = field(default_factory=lambda: [False] * 6)   # release N Vy Vz T My Mz
    releases_end: list[bool] = field(default_factory=lambda: [False] * 6)
    segments: int = 0                           # number of elements (0 = automatic)
    kind: str = "beam"                          # beam | truss (axial force only – hinges at both ends)


@dataclass
class FrameSupport:
    """Support of a node: dofs = constrained components ux uy uz rx ry rz (globally); stiffness = spring instead of a constraint
    (N/mm, N·mm/rad; 0 = no spring) – a component with stiffness > 0 is elastic, even if it is not in dofs.
    """
    node: str
    dofs: list[bool] = field(default_factory=lambda: [True] * 6)
    stiffness: list[float] = field(default_factory=lambda: [0.0] * 6)


@dataclass
class FrameConnection:
    """Connection of a node of the bar to a face of a body (selection of faces): rigid constraint (*COUPLING/*KINEMATIC, as bolts)."""
    node: str
    selection: str
    coupling: str = "rigid"


@dataclass
class FrameSettings:
    solver: str = "timoshenko"                  # timoshenko (own exact element) | ccx_u1 (U1 element in ccx – comparison)
    segments_static: int = 1                    # elements per straight bar in the statics of the own solver (with 1 element it is exact)
    segments_modal: int = 8                     # elements per straight bar for natural frequencies and buckling (consistent mass / K_G)
    buckling_torsion: bool = False              # buckling: term N·i_p² of the St. Venant torsion (without warping it overestimates torsion)
    segments_straight: int = 20                 # elements per straight bar in ccx_u1 (U1: shear deformation converges ~9 %/n²)
    max_arc_angle: float = 5.0                  # max. angle of an arc element [°]
    shear_deformation: bool = True              # Timoshenko (κ = Av_z/A, jinak 5/6); False = Euler–Bernoulli
    steel_grade: str = "S235"                   # for the assessment according to EN 1993-1-1 (fy by thickness)
    gamma_M0: float = 1.0
    gamma_M1: float = 1.0


@dataclass
class Frame:
    nodes: dict[str, FrameNode] = field(default_factory=dict)
    members: dict[str, FrameMember] = field(default_factory=dict)
    sections: dict[str, SectionDef] = field(default_factory=dict)
    supports: list[FrameSupport] = field(default_factory=list)
    connections: list[FrameConnection] = field(default_factory=list)
    settings: FrameSettings = field(default_factory=FrameSettings)
    checks: dict = field(default_factory=dict)  # assessment EN 1993 by bars: {bar: {Lcr_y, Lcr_z, L_LT, C1, …}}

    @property
    def active(self) -> bool:
        return bool(self.members)

    def add_node(self, name: str, xyz) -> FrameNode:
        self.nodes[name] = FrameNode(name, [float(v) for v in xyz])
        return self.nodes[name]

    def to_dict(self) -> dict:
        return {"nodes": {k: asdict(v) for k, v in self.nodes.items()},
                "members": {k: asdict(v) for k, v in self.members.items()},
                "sections": {k: asdict(v) for k, v in self.sections.items()},
                "supports": [asdict(s) for s in self.supports],
                "connections": [asdict(c) for c in self.connections],
                "settings": asdict(self.settings), "checks": dict(self.checks)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Frame":
        d = d or {}
        return cls(nodes={k: _known(FrameNode, v) for k, v in d.get("nodes", {}).items()},
                   members={k: _known(FrameMember, v) for k, v in d.get("members", {}).items()},
                   sections={k: _known(SectionDef, v) for k, v in d.get("sections", {}).items()},
                   supports=[_known(FrameSupport, s) for s in d.get("supports", [])],
                   connections=[_known(FrameConnection, c) for c in d.get("connections", [])],
                   settings=_known(FrameSettings, d.get("settings")), checks=dict(d.get("checks") or {}))


# ------------------------------------------------------------------ loads of the bars (in LoadCase.loads)

@dataclass
class FrameNodeLoad:
    """Concentrated force [N] and moment [N·mm] at a node of the frame system (global axes)."""
    node: str
    force: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    moment: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    kind: str = "frame_node"


@dataclass
class FrameMemberLoad:
    """Distributed load of a bar [N/mm] along the length: q_start → q_end linearly (q_end None = uniform), in global
    (axes="global") or local axes of the bar (axes="local": x, y, z). On an arc along the arc length.
    """
    member: str
    q: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    q_end: list[float] | None = None
    axes: str = "global"
    kind: str = "frame_member"


@dataclass
class FrameTemperature:
    """Thermal load of the bars (own solver only): change of temperature `dT` [K] relative to the state without stress (uniform →
    axial strain α·dT) and the temperature difference across the section `grad_z` = T(+z edge) − T(−z edge) [K] in the direction of the local axis z
    (bending My), `grad_y` likewise in the direction y (bending Mz); linear distribution over the height of the section. `members` empty = all bars.
    α is taken from the material of the bar.
    """
    members: list[str] = field(default_factory=list)
    dT: float = 0.0
    grad_z: float = 0.0
    grad_y: float = 0.0
    kind: str = "frame_temperature"
