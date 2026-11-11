"""Contacts of assemblies and nonlinear stepping (stages F8.5a/b, F8.7): Study.contacts, Study.contact_settings,
Study.analysis.

Realisation in ccx 2.23 (specification F8, 4.1–4.5; verified trials scratch/f8_contact, scratch/ccx_manual):
  - contact = *CONTACT PAIR, TYPE=SURFACE TO SURFACE (penalty, default) or MORTAR; N2S only as a backup;
  - the normal stiffness K and the slope of the sticking λ are ALWAYS written explicitly (the default of ccx takes the E of the first material in the input);
  - friction *FRICTION μ > 0 (μ = 0 is refused by ccx) – "without friction" in the history with a later friction = μ 1e-4;
  - interference = *CLEARANCE, VALUE = −δ_r (radially, overwrites the geometric gap of the whole pair);
  - nonlinear calculation: branches ([preload] + k-th state, in parallel) or a sequence (history, one job).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields

CONTACT_TYPES = ("bonded", "no_separation", "frictionless", "frictional", "rough", "interference",
                 "compression_only", "none")
CONTACT_LABELS = {"bonded": "Bonded", "no_separation": "No separation (sliding)",
                  "frictionless": "Frictionless contact", "frictional": "Frictional contact", "rough": "No-slip contact",
                  "interference": "Interference (press fit)", "compression_only": "Compression only (support)",
                  "none": "No interaction"}
NONLINEAR_TYPES = ("no_separation", "frictionless", "frictional", "rough", "interference", "compression_only")
MU_NONE = 1e-4                    # "without friction" in a study where the friction is switched on later (*CHANGE FRICTION)
MU_ROUGH = 10.0                   # contact without slip = large μ


def _known(klass, d):
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class ContactPair:
    """Contact pair between the parts or the face selections `a`, `b` (name of a selection, otherwise the name of a part = all its faces
    in the contact). slave = the finer mesh, on equality the softer material (None = automatically, otherwise "a" | "b").

    mu = coefficient of friction (None = from the library by the purpose: slip = μ_min for the transmission of the force by friction,
    resistance = μ_max for the resistance / the force of a drive); stiffness K [N/mm³] (None = 50·min(E) of the pair);
    stick_slope λ [N/mm³] (None = K/100); clearance [mm] RADIAL (−Δd/2 = interference) overwrites the geometric gap of the
    pair by one value; adjust = bring the slave nodes into contact (change of the coordinates of the mesh, never for an interference);
    initial = initial state from "Find touches" (contact | gap | penetration).
    """
    name: str
    a: str
    b: str
    kind: str = "frictionless"
    mu: float | None = None
    purpose: str = "slip"
    materials: str | None = None      # pair in the library of friction (e.g. "steel–steel dry")
    stiffness: float | None = None
    stick_slope: float | None = None
    clearance: float | None = None
    adjust: bool = False
    slave: str | None = None
    initial: str | None = None
    gap: float | None = None          # min. gap / penetration from the detection [mm] (negative = penetration)
    area: float | None = None         # contact area from detection [mm²]
    source: str = "user"              # auto | user
    enabled: bool = True
    conductance: float | None = None  # W/(m²·K) thermal contact conductance (stage F9.6)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "ContactPair":
        return _known(cls, d)


@dataclass
class ContactSettings:
    """method: surface_to_surface (default, pressure ±0.1 %) | mortar (hard contact, interference) | node_to_surface
    (only for the convergence of the friction – pressure with tet10 is not usable). One formulation per study (ccx does not allow mixing).
    split: separate = the parts of the pair meshed separately without *TIE (non-conforming mesh) | conformal_split = conforming mesh
    + split of the interface (nodes opposite each other, bolted joints). stabilize = weak springs against free bodies.
    nlgeom = large deformations (snap-fit, cam); pressure = dead | following pressure (only with NLGEOM).
    """
    method: str = "surface_to_surface"
    detect_tol: float | None = None
    split: str = "separate"
    stabilize: str = "none"
    stabilize_k: float | None = None
    nlgeom: bool = False
    pressure: str = "dead"

    @classmethod
    def from_dict(cls, d) -> "ContactSettings":
        return _known(cls, d)


@dataclass
class Motion:
    """Prescribed motion of a rigid body / selection in the sequence (C2): target = a selection with a rigid constraint (remote
    constraint rigid) or a selection of faces; kind disp [mm] | rot [rad]; dof 1–3 in global axes (or csys);
    amplitude [[t, value], …] in the time of the step (0–1).
    """
    target: str
    kind: str = "disp"
    dof: int = 1
    amplitude: list = field(default_factory=lambda: [[0.0, 0.0], [1.0, 0.0]])
    csys: str | None = None

    @classmethod
    def from_dict(cls, d) -> "Motion":
        return _known(cls, d)


@dataclass
class LoadStep:
    """Step of the load history (nonlinear sequence). case = load state (None = only the prescribed motion / a change of contact),
    scale = multiple of the load of the state, ramp linear | step; bolts preload (preload of the bolts) | lock
    (length locked by *BOUNDARY, FIXED) | free; contacts_off = pairs switched off (*MODEL CHANGE); mu_change = {pair: μ}
    (*CHANGE FRICTION); prescribed = prescribed motions; control of the increments dt0, dt_min, dt_max (share of the step),
    max_inc; output_every = a frame every n increments; cycles ≥ 2 for fatigue with friction (repetition of the block).
    """
    name: str
    case: str | None = None
    scale: float = 1.0
    ramp: str = "linear"
    bolts: str = "lock"
    contacts_off: list[str] = field(default_factory=list)
    mu_change: dict = field(default_factory=dict)
    prescribed: list[Motion] = field(default_factory=list)
    dt0: float = 0.1
    dt_min: float = 1e-5
    dt_max: float = 0.25
    max_inc: int = 200
    direct: bool = False
    output_every: int = 1
    cycles: int = 1
    # implicit dynamics with contact (C3/D7, ccx *DYNAMIC HHT-α): time = duration of the step [s], dt0/dt_min/dt_max then
    # in seconds; alpha ∈ [−1/3, 0] (0 for impacts – specification KK4)
    dynamic: bool = False
    time: float = 1.0
    alpha: float = -0.05

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "LoadStep":
        s = _known(cls, d)
        s.prescribed = [Motion.from_dict(m) for m in (d or {}).get("prescribed", [])]
        return s


@dataclass
class AnalysisSettings:
    """kind linear (today's superposition) | nonlinear. Nonlinear: branches = True → each load state separately
    after [bolt preload] (in parallel, without history); False → the sequence `sequence` (one job, history).
    """
    kind: str = "linear"
    branches: bool = True
    sequence: list[LoadStep] = field(default_factory=list)
    stop_keep: bool = True            # on non-convergence keep the last converted increment (partial result)
    initial_velocity: dict = field(default_factory=dict)   # {part: [vx, vy, vz] mm/s} – initial velocity (*DYNAMIC)

    @property
    def nonlinear(self) -> bool:
        return self.kind == "nonlinear"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "AnalysisSettings":
        a = _known(cls, d)
        a.sequence = [LoadStep.from_dict(s) for s in (d or {}).get("sequence", [])]
        return a
