"""Model of the study: materials, parts, face selections, constraints, load states, mesh.

Everything is serialisable to JSON (study.json inside the .fea file).
Units: mm, N, MPa, t, s.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

from .fatigue.settings import FatigueSettings
from .joints.model import BoltSpec, WeldSpec, JointSettings
from .printing.model import PrintSettings
from .thermal.model import ThermalSettings
from .eigen import ModalSettings, BucklingSettings
from .beams.model import Frame, FrameNodeLoad, FrameMemberLoad, FrameTemperature
from .twod.model import TwoDSettings
from .shell.model import ShellSettings
from .selectors import FaceSelector, EdgeSelector, PointSelector
from .dynamics.model import DynamicsSettings, masses_from_list
from .contacts.model import ContactPair, ContactSettings, AnalysisSettings
from .motion.model import Mechanism
from .combinations import Combination
from .imprint import Imprint
from .cad_link.model import CadLink

# version of the study file: 2 = F8 (contacts, nonlinear steps, mechanism, dynamics, added mass); older versions
# of the program ignore the keys of the new version → "features" says what would be lost (Study.load_warnings)
FILE_VERSION = 2
TOP_KEYS = {"format", "version", "features", "name", "geometry", "materials", "default_material", "part_materials",
            "selections", "constraints", "load_cases", "mesh", "solver", "fatigue", "bolts", "welds", "joints",
            "ignored_parts", "printing", "thermal", "coordinate_systems", "modal", "buckling", "frame", "twod",
            "shell", "masses", "contacts", "contact_settings", "analysis", "mechanism", "dynamics", "assessment", "combinations", "imprints",
            "cad_link", "hidden_parts"}
FEATURE_LABELS = {
    "contacts": "contacts", "analysis": "nonlinear stepping", "mechanism": "motion (mechanism)",
    "dynamics": "dynamics", "masses": "added mass", "plastic": "material plasticity",
    "combinations": "load combination", "imprints": "face imprints", "assessment": "stress assessment",
    "cad_link": "CAD link and geometry iterations"
}


class StudyFormatError(ValueError):
    """The study file contains a kind of item that this version of the program does not know (a file from a newer version)."""


def _known(klass, d: dict | None):
    """Dataclass from a dictionary without unknown keys (a file from a newer version can be opened)."""
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class Material:
    name: str
    E: float                  # MPa (for a printed material E in the plane of the layer XY)
    nu: float
    rho: float = 0.0          # t/mm³ (ocel 7.85e-9)
    Re: float | None = None   # MPa
    Rm: float | None = None   # MPa (for printing: tensile strength XY)
    source: str = ""
    # printed material (F4): transversely isotropic, axis 3 = build direction Z. E_z = None → isotropic material.
    # values of the full extrusion (100 % infill); the shell / infill are handled by fea_core.printing.
    E_z: float | None = None      # MPa
    nu_z: float | None = None     # ν_xz: ε_z = −ν_xz·σ_x/E (ccx ν13)
    G_z: float | None = None      # shear modulus between the layers G_xz = G_yz [MPa]
    Rm_z: float | None = None     # tensile strength perpendicular to the layers (layer bond) [MPa]
    Rc: float | None = None       # compressive strength XY [MPa]
    Rc_z: float | None = None     # compressive strength Z [MPa]
    S_xy: float | None = None     # shear strength in the plane of the layer [MPa]
    S_z: float | None = None      # shear strength between the layers [MPa]
    # heat (fea_core.thermal; "engineering" units, conversion to ccx in thermal.model)
    k: float | None = None        # thermal conductivity [W/(m·K)]
    alpha: float | None = None    # coefficient of thermal expansion [1/K]
    cp: float | None = None       # specific heat capacity [J/(kg·K)] (transient conduction only)
    # general orthotropic material (fea_core.ortho): {E1, E2, E3, nu12, nu13, nu23, G12, G13, G23 [, alpha1–3]}
    # in the axes of the coordinate system ortho_csys (cylindrical: 1 = r, 2 = θ, 3 = z; None = global axes); E, nu = E1, ν12
    ortho: dict | None = None
    ortho_csys: str | None = None
    plastic: dict | None = None   # plasticita (fea_core.plasticity): {model, hardening, Et, table}

    @property
    def is_print(self) -> bool:
        return self.E_z is not None

    @property
    def is_ortho(self) -> bool:
        return bool(self.ortho)

    @property
    def is_plastic(self) -> bool:
        return bool(self.plastic)


@dataclass
class Selection:
    """Named selection: faces, or edges, or points (edge / point load). Constraints and other loads
    take only faces; a force (Force) depends on what the selection contains (face → area, edge → length, points → per point).
    """
    name: str
    faces: list[FaceSelector] = field(default_factory=list)
    edges: list[EdgeSelector] = field(default_factory=list)
    points: list[PointSelector] = field(default_factory=list)

    @property
    def entity(self) -> str:
        """"faces" | "edges" | "points" | "mixed" | "empty"."""
        kinds = [k for k in ("faces", "edges", "points") if getattr(self, k)]
        return kinds[0] if len(kinds) == 1 else "empty" if not kinds else "mixed"

    def to_dict(self) -> dict:
        d = {"name": self.name, "faces": [f.to_dict() for f in self.faces]}
        if self.edges:
            d["edges"] = [e.to_dict() for e in self.edges]
        if self.points:
            d["points"] = [p.to_dict() for p in self.points]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Selection":
        return cls(d["name"], [FaceSelector.from_dict(f) for f in d.get("faces", [])],
                   [EdgeSelector.from_dict(e) for e in d.get("edges", [])],
                   [PointSelector.from_dict(p) for p in d.get("points", [])])


ENTITY_LABEL = {"faces": "faces", "edges": "edges", "points": "points", "mixed": "faces, edges and points", "empty": "nothing"}


@dataclass
class CoordinateSystem:
    """Local coordinate system for the components of a load. cartesian: axes x, y, z (z = z_axis, x = x_axis orthogonalised
    to the plane ⟂ z); cylindrical: components (r, θ, z) – the axis of the cylinder = z_axis through origin, θ = 0 in the direction of x_axis.
    The cylindrical components are evaluated at each point separately (see fea_core.csys).
    """
    name: str
    kind: str = "cartesian"             # cartesian | cylindrical
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    z_axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    x_axis: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])


# ------------------------------------------------------------------ constraints

@dataclass
class Fixed:
    selection: str
    kind: str = "fixed"


@dataclass
class Displacement:
    """Prescribed displacement; None = the component is free."""
    selection: str
    ux: float | None = None
    uy: float | None = None
    uz: float | None = None
    kind: str = "displacement"


@dataclass
class CylindricalSupport:
    """Cylindrical support of a hole / pin (as Cylindrical Support in Ansys): zero displacement in the selected
    components in the axes of the cylinder. True = component constrained, False = free. The axis is determined from the face of the selection
    (fit of a cylinder through the nodes of the mesh, see fea_core.cylinder); ccx: *EQUATION at the nodes (not *TRANSFORM).
    """
    selection: str
    radial: bool = True
    tangential: bool = True       # circumferential component (rotation about the axis)
    axial: bool = True
    kind: str = "cylindrical"


@dataclass
class RemoteSupport:
    """Constraint at a remote point (as Remote Displacement in Ansys): a face connected to the point `point` by a rigid
    or distributing constraint; at the point zero displacements / rotations of the selected components (ux uy uz rx ry rz).
    ccx: rigid = *RIGID BODY + *BOUNDARY at the REF/ROT node, distributing = *EQUATION from the kinematics of RBE3
    (see fea_core.remote).
    """
    selection: str
    point: list[float]
    dofs: list[bool] = field(default_factory=lambda: [True] * 6)
    coupling: str = "rigid"            # rigid | distributing
    kind: str = "remote_support"


@dataclass
class ElasticSupport:
    """Elastic support of a face (Winkler foundation, as Elastic Support in Ansys): stiffness = k [N/mm³]
    (per_area) or total stiffness K [N/mm] distributed by the area; direction "all" = the same stiffness in x, y, z,
    "normal" = only perpendicular to the face. Linear (tension and compression). See fea_core.springs.
    """
    selection: str
    stiffness: float
    direction: str = "normal"          # normal | all
    per_area: bool = True
    kind: str = "elastic"


# ------------------------------------------------------------------ loads

@dataclass
class Force:
    """Total force [N] on the selection: faces → uniformly by area, edges → uniformly by length, points → evenly
    to each point (a node of the mesh; a point and edge force is singular in the solid model – the stress there does not hold).
    csys = name of the local coordinate system (Study.coordinate_systems), None = global axes. In a cylindrical system
    the components are (F_r, F_θ, F_z) and the direction changes over the face / edge: the given force is the sum of the magnitudes (distributed density
    F/A by area, F/L by length), the resultant generally ≠ F (e.g. a tangential component around the whole circumference = torque).
    """
    selection: str
    vector: list[float]
    csys: str | None = None
    kind: str = "force"


@dataclass
class Pressure:
    """Pressure [MPa]; positive acts against the outward normal (into the material).

    Variable pressure: p(x) = value + gradient·(x − origin) [gradient MPa/mm]; clip = True → the linear term only positive
    (hydrostatic pressure: p = value + max(0, ρ·g·depth), value = pressure above the level, origin = point on the level,
    gradient = ρ·g in the direction of gravity). Variable pressure → consistent nodal forces (*CLOAD), constant → *DLOAD.
    """
    selection: str
    value: float
    gradient: list[float] | None = None
    origin: list[float] | None = None
    clip: bool = False
    kind: str = "pressure"

    @property
    def variable(self) -> bool:
        return self.gradient is not None and any(float(g) != 0.0 for g in self.gradient)


@dataclass
class Gravity:
    vector: list[float] = field(default_factory=lambda: [0.0, 0.0, -9810.0])  # mm/s²
    csys: str | None = None           # Cartesian local system only
    kind: str = "gravity"


@dataclass
class Rotation:
    """Rotation of parts about an axis (point `point`, direction `axis`): angular velocity omega [rad/s] → centrifugal force ρ·ω²·r⊥,
    angular acceleration alpha [rad/s²] (positive = right-handed about `axis`) → inertia force −ρ·α × r (the drive must supply
    the torque J·α). Forces in the coordinate system rotating with the body (d'Alembert), a static problem – without Coriolis and
    without softening by the centrifugal force. parts = rotating parts (empty = all). Nodal forces see loads.rotation_forces.
    """
    point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    omega: float = 0.0
    alpha: float = 0.0
    parts: list[str] = field(default_factory=list)
    kind: str = "rotation"


@dataclass
class RemoteLoad:
    """Force and moment at a remote point transferred to the face by the constraint.

    coupling: "distributing" (does not increase stiffness) or "rigid" (rigid constraint).
    """
    selection: str
    point: list[float]
    force: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    moment: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # N·mm
    coupling: str = "distributing"
    csys: str | None = None           # components of force and moment in the Cartesian local system (the point stays global)
    kind: str = "remote"


@dataclass
class BearingLoad:
    """Bearing load [N] on a cylindrical face (hole / pin, as Bearing Load in Ansys).

    The component perpendicular to the axis = normal pressure p ~ cos θ only on the half of the cylinder facing the force
    (cosine distribution, the resultant exactly F, zero moment about the axis); the component in the direction of the axis =
    uniform shear load over the whole face. See fea_core.cylinder.bearing_forces.
    """
    selection: str
    vector: list[float]
    csys: str | None = None           # Cartesian local system only
    kind: str = "bearing"


@dataclass
class ThermalLoad:
    """Thermal load (thermal stress): temperature field from the thermal state `case` (frame at time `time`,
    None = end of the state), or a uniform temperature `value` [°C] of the parts `parts` (empty = all parts; it overwrites
    the field of the state). Strain α·(T − Study.thermal.reference_temperature), see fea_core.thermal.field.
    """
    case: str | None = None
    time: float | None = None
    value: float | None = None
    parts: list[str] = field(default_factory=list)
    kind: str = "thermal"


@dataclass
class Inertia:
    """Inertia forces of the rigid motion of parts (d'Alembert – transfer of an instant from the kinematics of a mechanism to FEM):
    body force −ρ·(a₀ + α × r + ω × (ω × r)), r = x − point; accel = acceleration of the point `point` [mm/s²], omega
    [rad/s], alpha [rad/s²] as vectors in global axes. parts = parts of the body (empty = all). The weight
    is entered separately (Gravity). Nodal forces see loads.inertia_forces.
    wmat = matrix of the centrifugal term W (3×3, ω × (ω × r) = W·r, W = ω·ωᵀ − |ω|²·I) instead of omega – unit
    states for the superposition of the history (W is linear in the six products ω_i·ω_j).
    """
    point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    accel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    omega: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    alpha: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    parts: list[str] = field(default_factory=list)
    wmat: list | None = None
    name: str = ""
    kind: str = "inertia"


CONSTRAINT_KINDS = {"fixed": Fixed, "displacement": Displacement, "cylindrical": CylindricalSupport,
                    "remote_support": RemoteSupport, "elastic": ElasticSupport}
LOAD_KINDS = {"force": Force, "pressure": Pressure, "gravity": Gravity, "remote": RemoteLoad,
              "bearing": BearingLoad, "thermal": ThermalLoad, "rotation": Rotation,
              "frame_node": FrameNodeLoad, "frame_member": FrameMemberLoad, "inertia": Inertia,
              "frame_temperature": FrameTemperature}


@dataclass
class LoadCase:
    name: str
    loads: list = field(default_factory=list)


@dataclass
class MeshSettings:
    size: float = 5.0                 # global element size [mm]
    min_size: float | None = None
    order: int = 2                    # 2 = tet10 (C3D10)
    curvature_points: int = 12        # number of elements per 2π on curved faces (0 = off)
    optimize_high_order: bool = True
    local: list[dict] = field(default_factory=list)  # [{"selection", "size"}] or spheres [{"center", "radius", "size"}]
    threads: int = 0                  # Gmsh threads; 0 = 1 thread (the mesh is then the same at every run), > 1 faster, but the mesh differs between runs
    algorithm3d: int = 10             # 10 = HXT (parallel, 2× faster), 1 = Delaunay (backup)
    separate: list[str] = field(default_factory=list)   # parts meshed separately, connected by *TIE (fea_core.ties)
    tie_tolerance: float | None = None                  # tolerance dotyku *TIE [mm] (None = 5 % velikosti prvku)
    part_sizes: dict[str, float] = field(default_factory=dict)  # element size per part


@dataclass
class SolverSettings:
    solver: str = "PARDISO"           # PARDISO | PASTIX | SPOOLES (only the ccx backend)
    exe: str | None = None
    backend: str = "auto"             # ccx | ngs_hypre | ngs_pardiso | auto (up to config.LARGE_DOF ccx, above it
    # NGSolve+hypre); ngs_pardiso = NGSolve + direct PARDISO, 1 factorisation for all states
    nproc: int = 0                    # MPI processes for hypre (0 = physical cores)


@dataclass
class Study:
    name: str
    geometry: str                     # path to the STEP (relative to study.json)
    materials: dict[str, Material] = field(default_factory=dict)
    default_material: str | None = None
    part_materials: dict[str, str] = field(default_factory=dict)
    selections: dict[str, Selection] = field(default_factory=dict)
    constraints: list = field(default_factory=list)
    load_cases: list[LoadCase] = field(default_factory=list)
    mesh: MeshSettings = field(default_factory=MeshSettings)
    solver: SolverSettings = field(default_factory=SolverSettings)
    fatigue: FatigueSettings | None = None
    bolts: list[BoltSpec] = field(default_factory=list)
    welds: list[WeldSpec] = field(default_factory=list)
    joints: JointSettings = field(default_factory=JointSettings)
    ignored_parts: list[str] = field(default_factory=list)   # parts excluded from the calculation (magnets, pins…)
    hidden_parts: list[str] = field(default_factory=list)    # parts hidden only in the 3D view (not concerned with the calculation)
    printing: PrintSettings = field(default_factory=PrintSettings)   # 3D printing (parts with printed material)
    thermal: ThermalSettings = field(default_factory=ThermalSettings)   # thermal states (heat conduction)
    coordinate_systems: dict[str, CoordinateSystem] = field(default_factory=dict)   # local systems of loads
    modal: ModalSettings = field(default_factory=ModalSettings)          # natural frequencies (fea_core.eigen)
    buckling: BucklingSettings = field(default_factory=BucklingSettings)  # linear buckling (fea_core.eigen)
    frame: Frame = field(default_factory=Frame)           # frame system (fea_core.beams; geometry "" = members only)
    twod: TwoDSettings = field(default_factory=TwoDSettings)  # 2D calculation by a section (axisymmetric / plane, fea_core.twod)
    shell: ShellSettings = field(default_factory=ShellSettings)  # shell model of plates and sheets (fea_core.shell)
    # ---- F8: added mass, contacts, nonlinear stepping, mechanism, dynamics
    masses: list = field(default_factory=list)                   # PointMass | SurfaceMass (fea_core.dynamics.mass)
    contacts: list[ContactPair] = field(default_factory=list)    # contact pairs (fea_core.contacts)
    contact_settings: ContactSettings = field(default_factory=ContactSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)   # linear | nonlinear (branches, sequence)
    mechanism: Mechanism = field(default_factory=Mechanism)      # kinematics of rigid bodies (fea_core.motion)
    dynamics: DynamicsSettings = field(default_factory=DynamicsSettings)   # dynamika konstrukce (fea_core.dynamics)
    assessment: dict = field(default_factory=dict)               # posouzeni / Stress Tool (fea_core.assessment)
    combinations: list[Combination] = field(default_factory=list) # kombinace (A2)
    imprints: list = field(default_factory=list)                 # otisky na plose (A10)
    cad_link: CadLink | None = None                              # link to CAD + geometry iterations (cad_link)

    def __post_init__(self):
        self.load_warnings: list[str] = []      # warnings on loading (file from a newer version) – outside the JSON

    # ---------------------------------------------------------------- helpers
    def all_contacts(self) -> list[ContactPair]:
        """Pairs of the study + automatic pairs of the bolt interfaces (JointSettings.contact_joint, μ = μ_T of the bolt)."""
        if not self.joints.contact_joint or not self.active_bolts():
            return list(self.contacts)
        from .contacts.auto import bolt_pairs
        names = {c.name for c in self.contacts}
        return list(self.contacts) + [p for p in bolt_pairs(self) if p.name not in names]

    def active_contacts(self) -> list[ContactPair]:
        return [c for c in self.all_contacts() if c.enabled and c.kind not in ("bonded", "none")]

    def contact_split(self) -> str:
        """Way of meshing the interface: bolt interfaces with contact require a conforming mesh with a split (imprints of the bolts)."""
        if self.joints.contact_joint and self.active_bolts():
            return "conformal_split"
        return self.contact_settings.split

    def is_nonlinear(self, mesh=None, parts=None) -> bool:
        """Nonlinear calculation (contact with separation / friction, NLGEOM, sequence, plasticity) – superposition does not hold."""
        if bool(self.active_contacts()) or self.analysis.nonlinear or self.contact_settings.nlgeom:
            return True
        excluded = self.excluded_parts()
        if mesh is not None and getattr(mesh, "part_names", None):
            active_parts = [p for p in mesh.part_names if p not in excluded]
            for pname in active_parts:
                try:
                    if getattr(self.material_of(pname), "is_plastic", False):
                        return True
                except KeyError:
                    pass
            return False
        if parts is not None:
            active_parts = [p.name for p in parts if p.name not in excluded]
            for pname in active_parts:
                try:
                    if getattr(self.material_of(pname), "is_plastic", False):
                        return True
                except KeyError:
                    pass
            return False
        # Without a mesh: we check the materials assigned to the parts not left out
        for part_name, mat_name in self.part_materials.items():
            if part_name not in excluded:
                mat = self.materials.get(mat_name)
                if mat and getattr(mat, "is_plastic", False):
                    return True
        if self.default_material and not self.part_materials:
            def_mat = self.materials.get(self.default_material)
            if def_mat and getattr(def_mat, "is_plastic", False):
                return True
        return False

    def features(self) -> list[str]:
        """Used F8/F9 extensions (they are written to the file – older versions of the program report them as ignored)."""
        out = []
        if self.contacts:
            out.append("contacts")
        if self.analysis.nonlinear:
            out.append("analysis")
        if self.mechanism.active:
            out.append("mechanism")
        if self.dynamics.enabled:
            out.append("dynamics")
        if self.masses:
            out.append("masses")
        if any(getattr(m, "is_plastic", False) for m in self.materials.values()):
            out.append("plastic")
        if self.combinations:
            out.append("combinations")
        if self.imprints:
            out.append("imprints")
        if self.assessment:
            out.append("assessment")
        if self.cad_link is not None:
            out.append("cad_link")
        return out

    def active_bolts(self) -> list[BoltSpec]:
        return [b for b in self.bolts if b.enabled]

    def active_welds(self) -> list[WeldSpec]:
        return [w for w in self.welds if w.enabled]

    def excluded_parts(self) -> set[str]:
        """Parts that are not meshed (bolts, nuts, washers replaced by a beam + left out)."""
        return {p for b in self.active_bolts() for p in b.cad_parts} | set(self.ignored_parts)

    def add_material(self, m: Material, default: bool = False) -> Material:
        self.materials[m.name] = m
        if default or self.default_material is None:
            self.default_material = m.name
        return m

    def add_selection(self, name: str, faces: list[FaceSelector] | None = None,
                      edges: list[EdgeSelector] | None = None, points: list[PointSelector] | None = None) -> str:
        sel = Selection(name, list(faces or []), list(edges or []), list(points or []))
        if sel.entity == "empty":
            raise ValueError(f"Selection '{name}' is empty")
        if sel.entity == "mixed":
            raise ValueError(f"Selection '{name}' mixes faces, edges, and points – use one kind per selection")
        self.selections[name] = sel
        return name

    def thermal_loads(self, lc: LoadCase) -> list[ThermalLoad]:
        return [l for l in lc.loads if isinstance(l, ThermalLoad)]

    def has_thermal_loads(self) -> bool:
        """Some mechanical state has a thermal load (thermal stress)."""
        return any(isinstance(l, ThermalLoad) for lc in self.load_cases for l in lc.loads)

    def geometry_path(self, study_dir=None) -> Path:
        """Path to the STEP: the relative path is taken relative to the folder of the study study_dir (None = as written)."""
        geo = Path(self.geometry)
        return Path(study_dir) / geo if not geo.is_absolute() and study_dir is not None else geo

    def material_of(self, part: str) -> Material:
        key = self.part_materials.get(part, self.default_material)
        if key is None or key not in self.materials:
            raise KeyError(f"Part '{part}' has no material")
        return self.materials[key]

    # ---------------------------------------------------------------- JSON
    def to_dict(self) -> dict:
        d = {
            "format": "fea-studio-study", "version": FILE_VERSION, "features": self.features(),
            "name": self.name, "geometry": self.geometry,
            "materials": {k: asdict(v) for k, v in self.materials.items()},
            "default_material": self.default_material,
            "part_materials": dict(self.part_materials),
            "selections": {k: s.to_dict() for k, s in self.selections.items()},
            "constraints": [asdict(c) for c in self.constraints],
            "load_cases": [{"name": lc.name, "loads": [asdict(l) for l in lc.loads]}
                           for lc in self.load_cases],
            "mesh": asdict(self.mesh),
            "solver": {k: v for k, v in asdict(self.solver).items() if k != "exe"},
            "fatigue": self.fatigue.to_dict() if self.fatigue else None,
            "bolts": [b.to_dict() for b in self.bolts],
            "welds": [w.to_dict() for w in self.welds],
            "joints": self.joints.to_dict(),
            "ignored_parts": list(self.ignored_parts),
            "printing": self.printing.to_dict(),
            "thermal": self.thermal.to_dict(),
            "coordinate_systems": {k: asdict(c) for k, c in self.coordinate_systems.items()},
            "modal": self.modal.to_dict(),
            "buckling": self.buckling.to_dict(),
            "frame": self.frame.to_dict(),
            "twod": self.twod.to_dict(),
            "shell": self.shell.to_dict(),
        }
        # F8 only when used (a file without F8 stays as before)
        if self.masses:
            d["masses"] = [asdict(m) for m in self.masses]
        if self.contacts:
            d["contacts"] = [c.to_dict() for c in self.contacts]
        if self.contacts or self.contact_settings != ContactSettings():
            d["contact_settings"] = asdict(self.contact_settings)
        if self.analysis != AnalysisSettings():
            d["analysis"] = self.analysis.to_dict()
        if self.mechanism != Mechanism():
            d["mechanism"] = self.mechanism.to_dict()
        if self.dynamics != DynamicsSettings():
            d["dynamics"] = self.dynamics.to_dict()
        if self.assessment:
            d["assessment"] = dict(self.assessment)
        if self.combinations:
            d["combinations"] = [c.to_dict() for c in self.combinations]
        if self.imprints:
            d["imprints"] = [asdict(imp) if hasattr(imp, "__dataclass_fields__") else dict(imp) for imp in self.imprints]
        if self.cad_link is not None:
            d["cad_link"] = self.cad_link.to_dict()
        if self.hidden_parts:
            d["hidden_parts"] = list(self.hidden_parts)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Study":
        def build(kinds, item):
            item = dict(item)
            klass = kinds.get(item.get("kind"))
            if klass is None:
                raise StudyFormatError(f"Unknown support / load type '{item.get('kind')}' – the file is from a newer version of FEA Studio; open it there (or update the program)")
            names = {f.name for f in fields(klass)}
            return klass(**{k: v for k, v in item.items() if k in names})
        warnings = []
        unknown = [k for k, v in d.items() if k not in TOP_KEYS and v not in (None, [], {}, "", False)]
        unsupported = [f for f in d.get("features", []) if f not in FEATURE_LABELS]
        if unknown or unsupported or int(d.get("version", 1) or 1) > FILE_VERSION:
            desc = list(unknown) + list(unsupported)
            warnings.append(f"The study file is from a newer program version (version {d.get('version')}) – ignored: " + (", ".join(desc) or "new options")
                            + ". The results may not correspond to the original study; saving will delete these data.")
        try:
            masses = masses_from_list(d.get("masses"))
        except KeyError as e:
            raise StudyFormatError(str(e)) from None
        st = cls(
            name=d["name"], geometry=d["geometry"],
            materials={k: _known(Material, v) for k, v in d.get("materials", {}).items()},
            default_material=d.get("default_material"),
            part_materials=dict(d.get("part_materials", {})),
            selections={k: Selection.from_dict(s) for k, s in d.get("selections", {}).items()},
            constraints=[build(CONSTRAINT_KINDS, c) for c in d.get("constraints", [])],
            load_cases=[LoadCase(lc["name"], [build(LOAD_KINDS, l) for l in lc["loads"]])
                        for lc in d.get("load_cases", [])],
            mesh=_known(MeshSettings, d.get("mesh", {})),
            # the "exe" from the file is not taken: a foreign .fea would otherwise choose the program that runs as the solver
            # (ccx is searched only by config.find_ccx() / the variable FEA_CCX)
            solver=_known(SolverSettings, {k: v for k, v in (d.get("solver") or {}).items() if k != "exe"}),
            fatigue=FatigueSettings.from_dict(d["fatigue"]) if d.get("fatigue") else None,
            bolts=[BoltSpec.from_dict(b) for b in d.get("bolts", [])],
            welds=[WeldSpec.from_dict(w) for w in d.get("welds", [])],
            joints=JointSettings.from_dict(d.get("joints")),
            ignored_parts=list(d.get("ignored_parts", [])),
            hidden_parts=[str(x) for x in d.get("hidden_parts") or [] if isinstance(x, str)],
            printing=PrintSettings.from_dict(d.get("printing")),
            thermal=ThermalSettings.from_dict(d.get("thermal")),
            coordinate_systems={k: _known(CoordinateSystem, c) for k, c in d.get("coordinate_systems", {}).items()},
            modal=ModalSettings.from_dict(d.get("modal")),
            buckling=BucklingSettings.from_dict(d.get("buckling")),
            frame=Frame.from_dict(d.get("frame")),
            twod=TwoDSettings.from_dict(d.get("twod")),
            shell=ShellSettings.from_dict(d.get("shell")),
            masses=masses,
            contacts=[ContactPair.from_dict(c) for c in d.get("contacts", [])],
            contact_settings=ContactSettings.from_dict(d.get("contact_settings")),
            analysis=AnalysisSettings.from_dict(d.get("analysis")),
            mechanism=Mechanism.from_dict(d.get("mechanism")),
            dynamics=DynamicsSettings.from_dict(d.get("dynamics")),
            assessment=dict(d.get("assessment", {})),
            combinations=[Combination.from_dict(c) for c in d.get("combinations", [])],
            imprints=[Imprint.from_dict(imp) if isinstance(imp, dict) else imp for imp in d.get("imprints", [])],
            cad_link=CadLink.from_dict(d.get("cad_link")) if isinstance(d.get("cad_link"), dict) else None,
        )
        st.load_warnings = warnings
        return st

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Study":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
