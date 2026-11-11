"""Heat in the study: thermal states (steady / transient heat conduction) and their boundary conditions.

The units in the Study and in the results are "engineering": temperature °C, power W, flux W/m², heat transfer
coefficient W/(m²·K), conductivity W/(m·K), specific heat capacity J/(kg·K), time s. The conversion to the consistent
units of ccx (mm, N, s, t → power mW) is ONLY in fea_core.thermal.inp / .run (constants below).

Thermal stresses: a mechanical load state with ThermalLoad (study.LOAD_KINDS["thermal"]) – the temperature
from the result of a thermal state or a uniform temperature of the parts; without stress at reference_temperature.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields

# "engineering" units → ccx (mm, N, s, t; energy mJ = N·mm, power mW = N·mm/s)
K_TO_CCX = 1.0            # W/(m·K) = mW/(mm·K)
FLUX_TO_CCX = 1e-3        # W/m² → mW/mm²  (also the heat transfer coefficient W/(m²·K) → mW/(mm²·K))
POWER_TO_CCX = 1e3        # W → mW
CP_TO_CCX = 1e6           # J/(kg·K) → mJ/(t·K) = mm²/(s²·K)
STEFAN_BOLTZMANN = 5.670374419e-8   # W/(m²·K⁴)
# σ in the units of ccx (mW/(mm²·K⁴)): 1 W/m² = 1e3 mW / 1e6 mm² = 1e-3 mW/mm² → σ·FLUX_TO_CCX = 5.670374419e-11;
# exactly this value is written by thermal.inp into *PHYSICAL CONSTANTS (temperatures in °C, ABSOLUTE ZERO below)
STEFAN_BOLTZMANN_CCX = STEFAN_BOLTZMANN * FLUX_TO_CCX
ABSOLUTE_ZERO = -273.15   # °C
MAX_CAVITIES = 99         # ccx distinguishes the names of cavities by three characters → codes C01…C99 (thermal.inp.cavity_codes)


# ------------------------------------------------------------------ boundary conditions of the thermal state

@dataclass
class FixedTemperature:
    """Prescribed temperature of the face [°C]."""
    selection: str
    value: float
    kind: str = "fixed_temperature"


@dataclass
class HeatFlux:
    """Heat flux through the face [W/m²]; positive = heat enters the body."""
    selection: str
    value: float
    kind: str = "heat_flux"


@dataclass
class HeatFlow:
    """Total thermal power [W] distributed uniformly over the area of the face; positive = into the body."""
    selection: str
    value: float
    kind: str = "heat_flow"


@dataclass
class Convection:
    """Heat transfer (Newton): q = h·(T_surroundings − T) [W/m²]."""
    selection: str
    h: float                      # W/(m²·K)
    ambient: float = 20.0         # °C
    kind: str = "convection"


@dataclass
class Radiation:
    """Radiation to the surroundings or cavity radiation: q = ε·σ·(T_surroundings⁴ − T⁴), absolute temperatures."""
    selection: str
    emissivity: float = 1.0       # 0–1
    ambient: float = 20.0         # °C
    cavity: bool = False          # whether it is cavity radiation (CR)
    cavity_name: str | None = None  # name of the cavity (arbitrary; written to ccx as C01, C02…)
    closed: bool = False          # closed cavity: a negative temperature of the "surroundings" → ccx normalises the view factors to 1
    kind: str = "radiation"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Radiation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CavityRadiation:
    """Cavity radiation between faces (cavity radiation); `selection` is the name of a selection or a list of names of selections.
    The cavities are distinguished by `cavity_name` (ccx gets a unique code C01…).
    """
    selection: str | list[str] = "CAV"
    emissivity: float = 1.0       # 0–1
    ambient: float = 20.0         # °C
    cavity_name: str = "CAV"
    closed: bool = False
    kind: str = "cavity_radiation"

    def __init__(self, selection: str | list[str] | None = None, emissivity: float = 1.0,
                 ambient: float = 20.0, cavity_name: str = "CAV", closed: bool = False,
                 kind: str = "cavity_radiation", selections: list[str] | None = None, **kwargs):
        if selection is None and selections is not None:
            selection = selections
        self.selection = selection if selection is not None else "CAV"
        self.emissivity = emissivity
        self.ambient = ambient
        self.cavity_name = cavity_name
        self.closed = closed
        self.kind = kind

    @property
    def selections(self) -> list[str]:
        if isinstance(self.selection, list):
            return self.selection
        return [self.selection] if self.selection else []

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CavityRadiation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ or k == "selections"})


@dataclass
class ContactConductance:
    """Thermal resistance of a contact / contact conductance h_c [W/(m²·K)]."""
    selection: str = ""           # slave face / name of the contact
    master: str = ""              # master face
    conductance: float = 1000.0   # W/(m²·K)
    pressure: float | None = None
    temperature: float | None = None
    pressure_overclosure: str = "TIED"
    stiffness: float = 1e6
    name: str = "contact_cond"
    kind: str = "contact_conductance"

    def __init__(self, selection: str = "", master: str = "", conductance: float = 1000.0,
                 pressure: float | None = None, temperature: float | None = None,
                 pressure_overclosure: str = "TIED", stiffness: float = 1e6,
                 name: str = "contact_cond", kind: str = "contact_conductance",
                 slave: str | None = None, **kwargs):
        self.selection = slave if slave is not None else selection
        self.master = master
        self.conductance = conductance
        self.pressure = pressure
        self.temperature = temperature
        self.pressure_overclosure = pressure_overclosure
        self.stiffness = stiffness
        self.name = name
        self.kind = kind

    @property
    def slave(self) -> str:
        return self.selection

    @property
    def resistance(self) -> float:
        return 1.0 / self.conductance if self.conductance > 0 else float("inf")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ContactConductance":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ or k == "slave"})


@dataclass
class HeatGeneration:
    """Internal heat source [W] uniform in the volume of the part (losses in a winding, resistance…)."""
    part: str
    value: float                  # W
    kind: str = "heat_generation"


THERMAL_KINDS = {
    "fixed_temperature": FixedTemperature,
    "heat_flux": HeatFlux,
    "heat_flow": HeatFlow,
    "convection": Convection,
    "radiation": Radiation,
    "cavity_radiation": CavityRadiation,
    "contact_conductance": ContactConductance,
    "heat_generation": HeatGeneration,
}
# kinds of boundary conditions that CAN remove heat to the surroundings (a steady state without a heat sink has no solution);
# is_heat_sink decides – a closed cavity or radiation only between faces does not remove heat from the body
SINK_KINDS = ("fixed_temperature", "convection", "radiation", "cavity_radiation")


def is_cavity(bc) -> bool:
    """Cavity radiation (CavityRadiation, or Radiation with cavity=True)."""
    return isinstance(bc, CavityRadiation) or bool(getattr(bc, "cavity", False))


def bc_selections(bc) -> list[str]:
    """Names of the selections of the boundary condition (CavityRadiation may have several); HeatGeneration and the like = []."""
    sel = getattr(bc, "selection", None)
    if not sel:
        return []
    return list(sel) if isinstance(sel, (list, tuple)) else [sel]


def cavity_key(bc) -> str:
    """Name of the cavity of the condition (default "CAV")."""
    return getattr(bc, "cavity_name", None) or "CAV"


def is_heat_sink(bc) -> bool:
    """True if the boundary condition removes heat out of the body (prescribed temperature, convection, radiation to the surroundings).
    A closed cavity (closed=True, the "surroundings" at −300 °C is only a trick for normalising the view factors) does not remove heat out of the body –
    it only redistributes it between the faces of the cavity. An open cavity exchanges heat with the surroundings through the opening.
    """
    k = getattr(bc, "kind", "")
    if k in ("fixed_temperature", "convection"):
        return True
    if k in ("radiation", "cavity_radiation"):
        return not (is_cavity(bc) and bool(getattr(bc, "closed", False)))
    return False


def build_bc(item: dict):
    item = dict(item)
    klass = THERMAL_KINDS[item["kind"]]
    if hasattr(klass, "from_dict"):
        return klass.from_dict(item)
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in item.items() if k in names})


@dataclass
class ThermalCase:
    """Thermal state. Steady (transient=False) or transient from the initial temperature
    (ThermalSettings.initial_temperature) with a fixed time step dt up to the time time; output at `outputs` uniform frames
    (the last = the time time).
    """
    name: str
    loads: list = field(default_factory=list)
    transient: bool = False
    time: float = 3600.0          # s
    dt: float = 0.0               # s (0 = time/100)
    outputs: int = 10             # number of stored frames

    def step_size(self) -> float:
        return self.dt if self.dt > 0 else self.time / 100.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["loads"] = [asdict(l) for l in self.loads]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ThermalCase":
        names = {f.name for f in fields(cls)} - {"loads"}
        return cls(loads=[build_bc(l) for l in d.get("loads", [])], **{k: v for k, v in d.items() if k in names})


@dataclass
class ThermalSettings:
    cases: list[ThermalCase] = field(default_factory=list)
    reference_temperature: float = 20.0   # °C – temperature without thermal strain (assembly)
    initial_temperature: float | None = None   # °C – start of the transient state (None = reference)
    # contact conductances of interfaces (model data – they apply to ALL thermal cases; ccx *CONTACT PAIR is model data);
    # sides = two part names (bonded interface, one part meshed separately) or two face selections; written only when used
    conductances: list[ContactConductance] = field(default_factory=list)

    def case(self, name: str) -> ThermalCase:
        for c in self.cases:
            if c.name == name:
                return c
        raise KeyError(f"Thermal case '{name}' does not exist")

    @property
    def t_initial(self) -> float:
        return self.reference_temperature if self.initial_temperature is None else self.initial_temperature

    def to_dict(self) -> dict:
        d = {"cases": [c.to_dict() for c in self.cases], "reference_temperature": self.reference_temperature,
             "initial_temperature": self.initial_temperature}
        if self.conductances:
            d["conductances"] = [c.to_dict() for c in self.conductances]
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "ThermalSettings":
        d = d or {}
        return cls(cases=[ThermalCase.from_dict(c) for c in d.get("cases", [])],
                   reference_temperature=float(d.get("reference_temperature", 20.0)),
                   initial_temperature=d.get("initial_temperature"),
                   conductances=[ContactConductance.from_dict(c) for c in d.get("conductances") or []])
