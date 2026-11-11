"""Thermal resistance of a contact / contact conductance (stage F9.6, D1).

Analytical relations and a library of typical values (engineering units):
- contact thermal conductance h_c [W/(m²·K)] and the specific area resistance R_c = 1 / h_c [(m²·K)/W]
- absolute thermal resistance of the interface R_th = R_c / A = 1 / (h_c · A) [K/W]
- temperature drop at the interface ΔT = q / h_c [K]
- equivalent thin layer λ = h_c · t (R = t / λ)
- library of typical h_c (steel–steel, aluminium, pastes, washers) and the estimate `suggest_contact_conductance`

The model `ContactConductance` (boundary condition of a thermal state) is in `thermal.model`; writing into ccx
(`*SURFACE INTERACTION` + `*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TIED` + `*GAP CONDUCTANCE`
+ `*CONTACT PAIR`, h_c·1e-3 [mW/(mm²·K)]) is done only by `thermal.inp.write_model` – one source.
"""
from __future__ import annotations

import re
from typing import Any

CONTACT_CONDUCTANCE_DATABASE: dict[str, dict[str, Any]] = {
    "steel_steel_rough_dry": {
        "conductance_W_m2K": 1000.0,
        "resistance_m2K_W": 0.001,
        "description": "Steel–steel, dry contact, rough surface (Ra ~ 3–5 µm), low pressure (1–5 MPa)",
        "source": "Incropera & DeWitt, ch. 3.1.3",
    },
    "steel_steel_smooth_dry": {
        "conductance_W_m2K": 3000.0,
        "resistance_m2K_W": 0.000333,
        "description": "Steel–steel, dry contact, smooth surface (Ra ~ 1 µm), medium pressure (5–10 MPa)",
        "source": "Incropera & DeWitt",
    },
    "steel_steel_high_pressure": {
        "conductance_W_m2K": 10000.0,
        "resistance_m2K_W": 0.0001,
        "description": "Steel–steel, dry contact, high pressure (> 20 MPa) / tightened bolted joint",
        "source": "Černoch, Machine Design Handbook",
    },
    "aluminum_aluminum_dry": {
        "conductance_W_m2K": 4000.0,
        "resistance_m2K_W": 0.00025,
        "description": "Aluminum–aluminum, dry contact, pressure 1–5 MPa",
        "source": "Incropera & DeWitt",
    },
    "aluminum_steel_dry": {
        "conductance_W_m2K": 2500.0,
        "resistance_m2K_W": 0.0004,
        "description": "Aluminum–steel, dry contact, pressure 1–5 MPa",
        "source": "Incropera & DeWitt",
    },
    "copper_copper_dry": {
        "conductance_W_m2K": 10000.0,
        "resistance_m2K_W": 0.0001,
        "description": "Copper–copper, dry contact, high thermal conductivity",
        "source": "Incropera & DeWitt",
    },
    "thermal_paste_silicone": {
        "conductance_W_m2K": 25000.0,
        "resistance_m2K_W": 0.00004,
        "description": "Contact filled with thermal paste (TIM / silicone paste, thickness ~0.05 mm)",
        "source": "Madhusudana, Thermal Contact Conductance",
    },
    "thermal_pad": {
        "conductance_W_m2K": 5000.0,
        "resistance_m2K_W": 0.0002,
        "description": "Thermal pad (silicone pad, thickness ~0.5 mm, k ~ 2.5 W/(m·K))",
        "source": "Manufacturer datasheets (Bergquist / Laird)",
    },
    "bolted_flange_contact": {
        "conductance_W_m2K": 5000.0,
        "resistance_m2K_W": 0.0002,
        "description": "Preloaded bolted flange joint, steel–steel",
        "source": "VDI Wärmeatlas",
    },
}


def contact_resistance(conductance: float, area_m2: float | None = None) -> float:
    """Thermal resistance of the contact R_c.

    If the area area_m2 [m²] is given, returns the absolute thermal resistance R_th [K/W].
    If no area is given, returns the specific area resistance R_c [(m²·K)/W].
    """
    if conductance <= 0:
        raise ValueError(f"Contact conductance must be positive, entered {conductance}")
    r_unit = 1.0 / conductance
    if area_m2 is not None:
        if area_m2 <= 0:
            raise ValueError(f"Area must be positive, entered {area_m2}")
        return r_unit / area_m2
    return r_unit


def contact_temperature_drop(heat_flux_W_m2: float, conductance: float) -> float:
    """Temperature drop at the contact interface: ΔT = q / h_c [K]."""
    if conductance <= 0:
        raise ValueError(f"Contact conductance must be positive, entered {conductance}")
    return abs(heat_flux_W_m2) / conductance


def thin_layer_equivalent_conductivity(conductance: float, thickness_mm: float) -> float:
    """Backup model of the interface: equivalent thermal conductivity of a thin layer λ [W/(m·K)].

    λ = h_c · t  (where t is the thickness in metres).
    R = t / λ = 1 / h_c.
    """
    if conductance <= 0:
        raise ValueError(f"Contact conductance must be positive, entered {conductance}")
    if thickness_mm <= 0:
        raise ValueError(f"Layer thickness must be positive, entered {thickness_mm}")
    t_m = thickness_mm / 1000.0
    return conductance * t_m


def suggest_contact_conductance(mat_a: str, mat_b: str, pressure_mpa: float = 1.0,
                                roughness_um: float = 3.0, tim: str | None = None) -> float:
    """Estimate of the contact thermal conductance h_c [W/(m²·K)] from the materials, the pressure and the roughness (INDICATIVE).

    The material is recognised by whole words (not by substrings: "galvanized" is not aluminium, "medium" is not copper).
    """
    if tim:
        tim_key = tim.strip().lower()
        if "pad" in tim_key:
            return CONTACT_CONDUCTANCE_DATABASE["thermal_pad"]["conductance_W_m2K"]
        return CONTACT_CONDUCTANCE_DATABASE["thermal_paste_silicone"]["conductance_W_m2K"]

    pair = [mat_a.strip().lower(), mat_b.strip().lower()]

    al_tokens = {"al", "alu", "aluminium", "aluminum", "hlinik", "hliník", "hliníku", "hliníkový"}
    cu_tokens = {"cu", "copper", "med", "měď", "mědi", "měděný", "mosaz", "brass", "bronz", "bronze"}

    def is_mat(name: str, exact_tokens: set[str], prefix: str) -> bool:
        tokens = re.findall(r"\w+", name)   # whole words (also with diacritics)
        for t in tokens:
            if t in exact_tokens:
                return True
            if t.startswith(prefix) and len(t) > len(prefix) and t[len(prefix):].isdigit():
                return True
            if prefix == "al" and t.startswith("aw") and len(t) > 2 and t[2:].isdigit():
                return True
        return False

    is_al = any(is_mat(m, al_tokens, "al") for m in pair)
    is_cu = any(is_mat(m, cu_tokens, "cu") for m in pair)

    if is_cu:
        base_hc = 10000.0
    elif is_al:
        base_hc = 4000.0 if len(set(pair)) == 1 else 2500.0
    else:
        # steel / cast iron / general metals
        base_hc = 2000.0

    # influence of the pressure (the contact conductance grows roughly with the power 0.6–0.9 of the pressure)
    p_factor = max(pressure_mpa, 0.1) ** 0.7
    # influence of the roughness (h_c decreases with the roughness)
    r_factor = max(2.0 / max(roughness_um, 0.1), 0.3)

    hc = base_hc * p_factor * r_factor
    return round(float(hc), 1)
