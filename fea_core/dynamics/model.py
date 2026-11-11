"""Structural dynamics (stage F8.1, F8.4, F8.6): settings in Study.dynamics and the added mass (Study.masses).

Units of the project (mm, N, t, s). The acceleration is entered in g or mm/s² (g = 9810 mm/s²), the spectral density of the acceleration (ASD) in
g²/Hz – the conversion to (mm/s²)²/Hz only in the computation (fea_core.dynamics.units), as the units of heat.

Procedure (specification F8, decision 3 – the modal superposition in Python over the modes from ccx is the "evaluation"):
  - ccx *FREQUENCY (the same mesh and constraints as the statics) → shapes φ_j normalised by the mass, modal stresses σ_j, Γ_j, m_eff;
  - the harmonic / transient / random response and the shock = independent scalar equations of the modes (exact solution), the field
    u = Σ φ_j q_j, σ = Σ σ_j q_j; ccx *STEADY STATE DYNAMICS / *MODAL DYNAMIC serve as the etalon in the probes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields

G_MM = 9810.0                     # g [mm/s²]


def _known(klass, d):
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in (d or {}).items() if k in names})


# ------------------------------------------------------------------ added mass

@dataclass
class PointMass:
    """Point mass (motor, a purchased part, a left-out part) connected to the face `selection`.

    mass [t] (1 kg = 1e-3 t); point = centroid [mm] (None = centroid of the face); inertia = tensor of inertia
    about the centroid [t·mm²] as 3×3 or [Ixx, Iyy, Izz] (None = without rotary inertia); coupling: rigid (rigid
    constraint of the face – *RIGID BODY, the rotary inertia by distributing the mass into auxiliary nodes) | distributing (RBE3,
    does not increase the stiffness, only the translational mass). It acts in the natural frequencies and the dynamics and in the statics (weight,
    rotation – the force m·a at the point).
    """
    selection: str
    mass: float
    point: list[float] | None = None
    inertia: list | None = None
    coupling: str = "rigid"
    name: str = ""
    kind: str = "point_mass"


@dataclass
class SurfaceMass:
    """Surface mass [kg/m²] distributed over the area of the face (population of a PCB, paint, insulation); conversion to t/mm² only
    at the writing (1 kg/m² = 1e-9 t/mm²). Nodal masses *MASS with the weights of the face (like the elastic support).
    """
    selection: str
    mass_per_area: float
    name: str = ""
    kind: str = "surface_mass"


MASS_KINDS = {"point_mass": PointMass, "surface_mass": SurfaceMass}


# ------------------------------------------------------------------ damping

@dataclass
class DampingSpec:
    """Damping of the modal system.

    kind: modal = damping ratio ζ of all modes (table `table` [[f Hz, ζ], …] linearly by the frequency of the mode,
    otherwise `zeta`); rayleigh = α·M + β·K from ζ at f1 and f2 (ζ_j = α/(2ω_j) + β·ω_j/2); structural = hysteretic
    loss factor η (only harmonic response; η = 2ζ – careful, ccx *DAMPING, STRUCTURAL expects η, not ζ);
    per_material = {material: ζ} weighted by the modal strain energy (Johnson & Kienholz 1982, needs ENER).
    ALWAYS an uncertain input – the results of the dynamics are given with a warning and the sensitivity ±50 %.
    """
    kind: str = "modal"               # modal | rayleigh | structural | per_material
    zeta: float = 0.02
    eta: float | None = None
    table: list = field(default_factory=list)
    f1: float | None = None
    f2: float | None = None
    per_material: dict = field(default_factory=dict)


# ------------------------------------------------------------------ excitation and kinds of response

@dataclass
class Excitation:
    """Excitation: force = load state `case` × course (amplitude table [[f or t, multiple], …], empty =
    1); base_acc = acceleration of the base in the direction `direction` (global, unit vector or the axis dof 1–3) in the
    units `unit` (g | mm/s2); base_disp = displacement of the base [mm]. phase [°] only the harmonic response (forces with a
    different phase).
    """
    kind: str = "base_acc"            # force | base_acc | base_disp
    case: str | None = None
    dof: int | None = 3
    direction: list[float] | None = None
    table: list = field(default_factory=list)
    unit: str = "g"                   # g | mm/s2 (base_acc), mm (base_disp), N (multiple of the state)
    scale: float = 1.0
    phase: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "Excitation":
        return _known(cls, d)


@dataclass
class HarmonicSpec:
    """Harmonic response (D1): frequency range, points between the natural frequencies (densened at the resonances as in
    ccx – `points` per interval, `bias`), excitation. sweep_rate [oct/min] → fatigue from the sweep (dn/df), sweeps =
    number of passes.
    """
    enabled: bool = False
    f_min: float = 5.0
    f_max: float = 500.0
    points: int = 20
    bias: float = 3.0
    excitation: list[Excitation] = field(default_factory=list)
    sweep_rate: float = 1.0
    sweeps: int = 1
    static_correction: bool = True   # residual static contribution of the higher modes (mode acceleration)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d) -> "HarmonicSpec":
        h = _known(cls, d)
        h.excitation = [Excitation.from_dict(e) for e in (d or {}).get("excitation", [])]
        return h


@dataclass
class TransientSpec:
    """Transient response (D2): t_end, dt [s]; modal = exact solution of the equations of the modes (piecewise linear excitation),
    direct = ccx *DYNAMIC (HHT-α) – for nonlinearities. Excitation by a table in time [[t, multiple], …] (a step = [[0, 1],
    [t_end, 1]]). settle_tol [mm] = tolerance of the settling time in the probes (the decay after the motion of the axis).
    """
    enabled: bool = False
    t_end: float = 0.1
    dt: float = 1e-4
    method: str = "modal"             # modal | direct
    alpha: float = -0.05
    excitation: list[Excitation] = field(default_factory=list)
    settle_tol: float | None = None
    fatigue: bool = False             # fatigue from the history (q_j(t) as channels, modal stresses as unit states)

    @classmethod
    def from_dict(cls, d) -> "TransientSpec":
        t = _known(cls, d)
        t.excitation = [Excitation.from_dict(e) for e in (d or {}).get("excitation", [])]
        return t


@dataclass
class RandomSpec:
    """Random vibrations (D3): ASD of the base [[f Hz, g²/Hz], …] (between the points log-log as in the standards), direction,
    duration [h] per axis, method of the spectral fatigue nb | dirlik | tb.
    """
    enabled: bool = False
    asd: list = field(default_factory=list)
    dof: int = 3
    direction: list[float] | None = None
    duration_h: float = 1.0
    method: str = "dirlik"
    segalman: bool = True

    @classmethod
    def from_dict(cls, d) -> "RandomSpec":
        return _known(cls, d)


@dataclass
class ShockSpec:
    """Shock (D4): pulse of the base halfsine | sawtooth (end peak) | trapezoid | srs_table, A [g], tau [s];
    the response spectrum (SRS) → peaks of the modes, combination srss | cqc, missing mass (ZPA) as a static state.
    """
    enabled: bool = False
    pulse: str = "halfsine"
    A: float = 50.0
    tau: float = 0.011
    rise: float | None = None         # trapezium: rise time [s]
    dof: int = 3
    direction: list[float] | None = None
    srs_table: list = field(default_factory=list)    # [[f Hz, S_a g], …] for pulse = srs_table
    combine: str = "cqc"
    missing_mass: bool = True
    transient: bool = True            # in addition the transient response to the pulse (D2)

    @classmethod
    def from_dict(cls, d) -> "ShockSpec":
        return _known(cls, d)


@dataclass
class PrestressSpec:
    """Prestressed modal analysis (D5): natural frequencies after the static state `case` (stress stiffness, *STEP,
    PERTURBATION). rotating + omega_list [rpm] = Campbell diagram (a loop over the speeds, D5b).
    """
    enabled: bool = False
    case: str | None = None
    nlgeom: bool = False
    rotating: bool = False
    rpm_list: list[float] = field(default_factory=list)
    axis_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    coriolis: bool = True

    @classmethod
    def from_dict(cls, d) -> "PrestressSpec":
        return _known(cls, d)


@dataclass
class Probe:
    """Probe of the output in time / frequency: a point [mm] (projected onto the mesh) or a selection (mean of the nodes)."""
    name: str
    point: list[float] | None = None
    selection: str | None = None
    quantity: str = "U"               # U | S | A

    @classmethod
    def from_dict(cls, d) -> "Probe":
        return _known(cls, d)


@dataclass
class DynamicsSettings:
    """Structural dynamics: number of modes (modal basis), damping, kinds of response, probes.

    modes = number of the modes of the basis (recommended to cover 1.5–2× the highest excitation frequency and Σ m_eff ≥ 90 % in the excited
    direction – it is checked), f_max [Hz] = upper limit of the search of the modes (None = only the number).
    """
    enabled: bool = False
    modes: int = 30
    f_max: float | None = None
    damping: DampingSpec = field(default_factory=DampingSpec)
    harmonic: HarmonicSpec = field(default_factory=HarmonicSpec)
    transient: TransientSpec = field(default_factory=TransientSpec)
    random: RandomSpec = field(default_factory=RandomSpec)
    shock: ShockSpec = field(default_factory=ShockSpec)
    prestress: PrestressSpec = field(default_factory=PrestressSpec)
    probes: list[Probe] = field(default_factory=list)
    max_frames: int = 50              # frames of the fields of the transient response stored in the results
    verify_ccx: bool = False          # in addition ccx SSD / *MODAL DYNAMIC in the probes (etalon of the modal superposition)

    @property
    def active(self) -> bool:
        return self.enabled and any(x.enabled for x in (self.harmonic, self.transient, self.random, self.shock))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "DynamicsSettings":
        d = d or {}
        s = _known(cls, {k: v for k, v in d.items() if k not in ("damping", "harmonic", "transient", "random",
                                                                    "shock", "prestress", "probes")})
        s.damping = _known(DampingSpec, d.get("damping"))
        s.harmonic = HarmonicSpec.from_dict(d.get("harmonic"))
        s.transient = TransientSpec.from_dict(d.get("transient"))
        s.random = RandomSpec.from_dict(d.get("random"))
        s.shock = ShockSpec.from_dict(d.get("shock"))
        s.prestress = PrestressSpec.from_dict(d.get("prestress"))
        s.probes = [Probe.from_dict(p) for p in d.get("probes", [])]
        return s


def masses_from_list(items) -> list:
    out = []
    for it in items or []:
        klass = MASS_KINDS.get((it or {}).get("kind"))
        if klass is None:
            raise KeyError(f"Unknown added mass type '{(it or {}).get('kind')}' (file from a newer version?)")
        out.append(_known(klass, it))
    return out
