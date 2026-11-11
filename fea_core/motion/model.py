"""Rigid-body mechanism (stages F8.3, F8.8): Study.mechanism.

Units of the project (mm, N, t, s, rad); the working process of the MBD library (Exudyn) computes in SI – the conversion only at the boundary
(fea_core.motion.worker), like the units of heat. "Joint" = MotionJoint (not Joint – it would clash with joints/ = bolts, welds).

Kinds of joints (Czech names according to Fusion): fixed Pevný, revolute Rotační, prismatic Posuvný, cylindrical Válcový,
pin_slot Čep v drážce, planar Rovinný, spherical Kulový, universal Kardan, screw Šroubový (lead),
gear Ozubený převod (ratio), rack_pinion Pastorek–hřeben (ratio = mm/rad), distance Vzdálenost (a link).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields

JOINT_KINDS = ("fixed", "revolute", "prismatic", "cylindrical", "pin_slot", "planar", "spherical", "universal",
               "screw", "gear", "rack_pinion", "distance")
JOINT_LABELS = {"fixed": "Rigid", "revolute": "Revolute", "prismatic": "Slider", "cylindrical": "Cylindrical",
                "pin_slot": "Pin-slot", "planar": "Planar", "spherical": "Ball", "universal": "Universal",
                "screw": "Screw", "gear": "Gear", "rack_pinion": "Rack and pinion",
                "distance": "Distance (link)"}
# number of the constraints of the joint (Grübler): 6 − degrees of freedom of the joint
JOINT_CONSTRAINTS = {"fixed": 6, "revolute": 5, "prismatic": 5, "cylindrical": 4, "pin_slot": 3, "planar": 3,
                     "spherical": 3, "universal": 4, "screw": 5, "gear": 1, "rack_pinion": 1, "distance": 1}
FORCE_KINDS = ("spring", "damper", "spring_damper", "torsion", "gas_spring", "bushing")
DRIVER_QUANTITIES = ("angle", "disp", "velocity", "angular_velocity", "torque", "force")
DRIVER_LAWS = ("constant", "table", "sine", "poly", "scurve")


def _known(klass, d):
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class Body:
    """Body = a part or a group of parts (rigidly connected). ground = frame (motionless). Mass properties from the CAD (OCP
    volume integration × ρ), printed parts indicatively × the share of the shell and the infill; mass/com/inertia override the calculation
    (mass [t], com [mm], inertia 3×3 about the centroid [t·mm²]).
    """
    name: str
    parts: list[str] = field(default_factory=list)
    ground: bool = False
    mass: float | None = None
    com: list[float] | None = None
    inertia: list | None = None
    color: list[float] | None = None
    v0: list[float] | None = None      # initial velocity of the centroid [mm/s] (dynamic task)
    w0: list[float] | None = None      # initial angular velocity [rad/s]
    # flexible body (Craig–Bampton, only dynamic task, isotropic materials): the interface = faces of the joints of the body
    # (selection_a/b), modes = number of the natural shapes with the clamped interfaces in addition to the static shapes of the interface,
    # interface rbe2 (rigid interface) | rbe3 (distributing), mesh_size [mm] (None = mesh of the study), zeta = relative damping of the
    # 1st shape (proportional to the stiffness)
    flexible: bool = False
    modes: int = 10
    interface: str = "rbe2"
    mesh_size: float | None = None
    zeta: float = 0.0

    @classmethod
    def from_dict(cls, d) -> "Body":
        return _known(cls, d)


@dataclass
class JointFriction:
    """Friction in the joint (decision 2: the moment / force as a load from the reaction of the previous step).

    model: coulomb (μ, r_eff – a pin: moment μ'·|R|·r, sliding: force μ·|R_n|) | viscous (c_v) | stribeck
    (μ_s, μ_k, v_s, δ, σ2) | bearing (rolling bearing M = μ·F·d/2, r_eff = d/2) | thread (a power screw –
    the transmission and efficiency analytically); fit for a pin: clearance (line contact μ/√(1+μ²)) | worn (p ∝ cos θ, 4/π) |
    fitted (uniform pressure, π/2); preload = constant normal force [N] (seals, preload).
    """
    model: str = "coulomb"
    mu_s: float = 0.15
    mu_k: float = 0.1
    c_v: float = 0.0
    v_s: float = 1.0                  # mm/s (translation) | rad/s (rotation)
    delta: float = 2.0
    sigma2: float = 0.0
    r_eff: float = 5.0                # mm
    fit: str = "clearance"
    preload: float = 0.0
    v_reg: float = 0.5                # regularisation at zero velocity [mm/s or rad/s·r]

    @classmethod
    def from_dict(cls, d) -> "JointFriction":
        return _known(cls, d)


@dataclass
class MotionJoint:
    """Joint between the bodies body_a and body_b: origin `origin` [mm], axis `axis` (rotation / translation / screw); for a pin
    in a slot `axis2` = the direction of the slot; for a gear `joint_a`, `joint_b` = the coupled joints and the ratio (ω_b = ratio·ω_a);
    for a link point_a, point_b. selection_a/b = the faces from which the joint was proposed (the axis of the cylinder) – for the transfer to FE
    (the face of the load of the joint). limits [lo, hi] (rad | mm) = stops.
    """
    name: str
    kind: str = "revolute"
    body_a: str = ""
    body_b: str = ""
    origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    axis2: list[float] | None = None
    selection_a: str | None = None
    selection_b: str | None = None
    ratio: float | None = None
    lead: float | None = None
    joint_a: str | None = None
    joint_b: str | None = None
    point_a: list[float] | None = None
    point_b: list[float] | None = None
    limits: list | None = None
    clearance: float | None = None
    friction: JointFriction | None = None
    source: str = "user"              # user | cad | fusion
    # stops (limits): stiffness [N/mm | N·mm/rad], None = automatic (the contact time ~1 ms according to the inertia of the body);
    # coefficient of restitution e (0 = plastic impact, 1 = elastic) → damping of the stop
    limit_stiffness: float | None = None
    restitution: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "MotionJoint":
        j = _known(cls, d)
        if isinstance(j.friction, dict):
            j.friction = JointFriction.from_dict(j.friction)
        return j


@dataclass
class ForceElement:
    """Force element between the bodies: spring / damper / spring_damper (axis = the line of the points; F = f0 + k·(l − l0) +
    c·l̇; l0 None = the length in the position of the CAD), torsion (moment about the axis of the joint `joint`: M = k·(φ − φ0) + c·φ̇),
    gas_spring (table F(s) [[stroke mm, force N], …] – positive = pushes), bushing (6 stiffnesses [k_x k_y k_z
    k_rx k_ry k_rz] and 6 dampings at the point point_a).
    """
    name: str
    kind: str = "spring"
    body_a: str = ""
    body_b: str = ""
    point_a: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    point_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    k: float = 0.0                    # N/mm | N·mm/rad
    c: float = 0.0                    # N·s/mm | N·mm·s/rad
    f0: float = 0.0
    l0: float | None = None
    joint: str | None = None
    table: list | None = None
    k6: list[float] | None = None
    c6: list[float] | None = None
    # faces of the attachment of the element on the bodies (transfer to FE: the force of the element as a remote load of these faces)
    selection_a: str | None = None
    selection_b: str | None = None

    @classmethod
    def from_dict(cls, d) -> "ForceElement":
        return _known(cls, d)


@dataclass
class Driver:
    """Driver of the joint: quantity angle [rad] | disp [mm] | velocity [mm/s] | angular_velocity [rad/s] (kinematic)
    or torque [N·mm] | force [N] (force); law constant (data = [value]) | table ([[t, value], …], linearly)
    | sine (data = [amplitude, f Hz, phase rad, centre]) | poly (data = coefficients c0 + c1·t + …) | scurve
    (profile of the axis with limited jerk: data = [stroke, v_max, a_max, j_max, t_start]).
    """
    joint: str
    quantity: str = "angle"
    law: str = "constant"
    data: list = field(default_factory=lambda: [0.0])
    name: str = ""

    @classmethod
    def from_dict(cls, d) -> "Driver":
        return _known(cls, d)


@dataclass
class MotionLoad:
    """External force / moment on the body at a point (global axes or csys), the course by the multiple of the table [[t, k], …]."""
    body: str
    point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    force: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    moment: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    table: list | None = None
    name: str = ""
    selection: str | None = None      # the face on which the load acts (transfer to FE)

    @classmethod
    def from_dict(cls, d) -> "MotionLoad":
        return _known(cls, d)


@dataclass
class ToFea:
    """Transfer to FE: mode instants (the times `times`) | max (the instant of the maximum of the quantity `quantity`, e.g.
    "reakce:Kloub 1") | history (the whole course → unit states of the RemoteLoad distributing per joint / element + channels of fatigue);
    parts = the parts that are assessed; support_joints = joints transferred to the constraints of the FE (statically indeterminate,
    redundant); inertia_relief = inertia forces of the body (accelerations a₀, α, ω) as a load.
    """
    mode: str = "instants"
    times: list[float] = field(default_factory=list)
    quantity: str | None = None
    parts: list[str] = field(default_factory=list)
    support_joints: list[str] = field(default_factory=list)
    inertia_relief: bool = True
    load_kind: str = "remote_distributing"

    @classmethod
    def from_dict(cls, d) -> "ToFea":
        return _known(cls, d)


@dataclass
class MotionRun:
    """Calculation of the mechanism: kinematic (prescribed drivers, inverse dynamics – forces including the inertia) |
    static (quasi-static equilibrium in a series of positions, without the inertia) | dynamic (direct dynamics in time);
    t_end [s], outputs = number of output times, collisions = pairs of bodies [[a, b], …] or "all" (check of the collisions
    without contact – the minimum distance of the shapes), collision_tol [mm].
    """
    kind: str = "kinematic"
    t_end: float = 1.0
    outputs: int = 200
    dt: float | None = None
    solver: str = "exudyn"
    collisions: list | str | None = None
    collision_tol: float = 0.0
    to_fea: ToFea = field(default_factory=ToFea)
    spectral_radius: float | None = None     # ρ∞ generalized-α (None = kinematic 0.6, dynamic 0.8)
    include_extra_mass: bool = True          # mass of the left-out parts and bolts to the nearest body

    @classmethod
    def from_dict(cls, d) -> "MotionRun":
        r = _known(cls, d)
        r.to_fea = ToFea.from_dict((d or {}).get("to_fea"))
        return r


@dataclass
class MotionContact:
    """Contact of the bodies in the MBD (stage F8.9, C1 – only the Chrono solver): shapes = triangulation of the parts from the CAD (OCP), method nsc
    (complementarity, hard contact) | smc (penalty), mu = coefficient of friction, restitution e, rolling = rolling
    resistance [mm] (arm of the rolling friction), young [MPa] (only smc, None = from the material of the part), deflection = fineness of the
    triangulation [mm] (chordal deviation).
    """
    body_a: str
    body_b: str
    mu: float = 0.3
    restitution: float = 0.0
    rolling: float = 0.0
    method: str = "nsc"
    young: float | None = None
    deflection: float = 0.01
    name: str = ""

    @classmethod
    def from_dict(cls, d) -> "MotionContact":
        return _known(cls, d)


@dataclass
class Mechanism:
    bodies: list[Body] = field(default_factory=list)
    joints: list[MotionJoint] = field(default_factory=list)
    forces: list[ForceElement] = field(default_factory=list)
    drivers: list[Driver] = field(default_factory=list)
    loads: list[MotionLoad] = field(default_factory=list)
    run: MotionRun = field(default_factory=MotionRun)
    gravity: list[float] | None = field(default_factory=lambda: [0.0, 0.0, -9810.0])   # mm/s², None = without weight
    contacts: list[MotionContact] = field(default_factory=list)   # contacts of the bodies (only run.solver = "chrono")

    @property
    def active(self) -> bool:
        return bool(self.bodies)

    def body(self, name: str) -> Body:
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(f"Body '{name}' does not exist")

    def joint(self, name: str) -> MotionJoint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"Joint '{name}' does not exist")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Mechanism":
        d = d or {}
        m = cls()
        m.bodies = [Body.from_dict(b) for b in d.get("bodies", [])]
        m.joints = [MotionJoint.from_dict(j) for j in d.get("joints", [])]
        m.forces = [ForceElement.from_dict(f) for f in d.get("forces", [])]
        m.drivers = [Driver.from_dict(x) for x in d.get("drivers", [])]
        m.loads = [MotionLoad.from_dict(x) for x in d.get("loads", [])]
        m.run = MotionRun.from_dict(d.get("run"))
        m.contacts = [MotionContact.from_dict(x) for x in d.get("contacts", [])]
        if "gravity" in d:
            m.gravity = d["gravity"]
        return m
