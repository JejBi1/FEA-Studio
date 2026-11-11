"""Radiation of heat – analytical relations for checking and design (stage F9.6, D2).

- radiation between parallel plates and concentric cylinders, linearised radiative heat transfer coefficient h_r,
  combined heat transfer by convection + radiation h = h_c + h_r (engineering units: °C, W, m²).

The boundary conditions of radiation (Radiation to the surroundings, CavityRadiation cavity radiation) are in `thermal.model`,
their writing into ccx (`*RADIATE`, `*RADIATE, CAVITY=…`, `*PHYSICAL CONSTANTS`) in `thermal.inp` – one source.
σ for ccx (mm, N, s, t → mW): `model.STEFAN_BOLTZMANN_CCX` = σ·1e-3 = 5.670374419e-11 mW/(mm²·K⁴).

Sources:
- Stefan-Boltzmann law of radiation
- Incropera, F.P., DeWitt, D.P.: Fundamentals of Heat and Mass Transfer, ch. 13
- Černoch, S.: Strojně technická příručka (mechanical engineering handbook), SNTL Praha
- CalculiX 2.23 User's Manual (cards *RADIATE, *PHYSICAL CONSTANTS)
"""
from __future__ import annotations

import math

from .model import STEFAN_BOLTZMANN, ABSOLUTE_ZERO


def parallel_plates_radiation(T1_C: float, T2_C: float,
                              eps1: float = 1.0, eps2: float = 1.0,
                              area_m2: float | None = None) -> dict[str, float]:
    """Analytical calculation of the radiation between two parallel large planar plates.

    Heat flux:
        q = ε_eff · σ · (T1_K⁴ − T2_K⁴)
    where the effective emissivity:
        ε_eff = 1 / (1/ε1 + 1/ε2 − 1)

    Parameters:
        T1_C: temperature of the 1st plate [°C]
        T2_C: temperature of the 2nd plate [°C]
        eps1: emissivity of the 1st plate (0 to 1]
        eps2: emissivity of the 2nd plate (0 to 1]
        area_m2: optional area of the plate [m²] for the calculation of the total flux Q [W]
    """
    if not (0.0 < eps1 <= 1.0 and 0.0 < eps2 <= 1.0):
        raise ValueError("Emissivity must lie in the interval (0, 1]")

    T1_k = T1_C - ABSOLUTE_ZERO
    T2_k = T2_C - ABSOLUTE_ZERO

    eps_eff = 1.0 / (1.0 / eps1 + 1.0 / eps2 - 1.0)
    q = eps_eff * STEFAN_BOLTZMANN * ((T1_k ** 4) - (T2_k ** 4))
    h_r = radiation_heat_transfer_coefficient(T1_C, T2_C, eps_eff)

    res = {
        "q_W_m2": q,
        "eps_eff": eps_eff,
        "eps_effective": eps_eff,
        "h_rad": h_r,
        "T1_K": T1_k,
        "T2_K": T2_k,
    }
    if area_m2 is not None:
        if area_m2 <= 0:
            raise ValueError(f"Area must be positive, entered {area_m2}")
        res["Q_W"] = q * area_m2
        res["area_m2"] = area_m2

    return res


def concentric_cylinders_radiation(T1_C: float, T2_C: float,
                                   r1: float, r2: float,
                                   eps1: float = 1.0, eps2: float = 1.0,
                                   L_m: float | None = None,
                                   length_m: float | None = None) -> dict[str, float]:
    """Radiation between two infinitely long coaxial cylinders of radii r1 < r2.

    Effective emissivity related to the area of the inner cylinder A1:
        1 / ε_eff1 = 1/ε1 + (1 - ε2)/ε2 · (r1/r2)
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError("Cylinder radii must be positive")
    if r1 >= r2:
        raise ValueError(f"Inner radius r1 ({r1}) must be less than outer radius r2 ({r2})")
    if not (0.0 < eps1 <= 1.0 and 0.0 < eps2 <= 1.0):
        raise ValueError("Emissivity must lie in the interval (0, 1]")

    L = L_m if L_m is not None else length_m
    T1_k = T1_C - ABSOLUTE_ZERO
    T2_k = T2_C - ABSOLUTE_ZERO

    inv_eps = (1.0 / eps1) + ((1.0 - eps2) / eps2) * (r1 / r2)
    eps_eff = 1.0 / inv_eps
    q1 = eps_eff * STEFAN_BOLTZMANN * ((T1_k ** 4) - (T2_k ** 4))

    res = {
        "q1_W_m2": q1,
        "eps_eff1": eps_eff,
        "eps_effective": eps_eff,
        "r1_m": r1,
        "r2_m": r2,
    }
    if L is not None:
        if L <= 0:
            raise ValueError(f"Cylinder length must be positive, entered {L}")
        a1 = 2.0 * math.pi * r1 * L
        res["Q_W"] = q1 * a1
        res["length_m"] = L

    return res


def radiation_heat_transfer_coefficient(T_surface_C: float, T_ambient_C: float,
                                        emissivity: float = 1.0) -> float:
    """Linearised radiative heat transfer coefficient h_r [W/(m²·K)].

    h_r = ε · σ · (T_s + T_amb) · (T_s² + T_amb²)  [temperatures in kelvin]
    q = h_r · (T_s − T_amb).
    """
    if not (0.0 < emissivity <= 1.0):
        raise ValueError("Emissivity must lie in the range (0, 1]")

    Ts_k = T_surface_C - ABSOLUTE_ZERO
    Ta_k = T_ambient_C - ABSOLUTE_ZERO

    if abs(Ts_k - Ta_k) < 1e-6:
        return 4.0 * emissivity * STEFAN_BOLTZMANN * (Ts_k ** 3)

    return emissivity * STEFAN_BOLTZMANN * (Ts_k + Ta_k) * (Ts_k ** 2 + Ta_k ** 2)


def linearized_radiation_convection(T_surface_C: float, T_ambient_C: float,
                                    emissivity: float = 1.0,
                                    h_conv: float = 0.0) -> dict[str, float]:
    """Combined heat transfer coefficient of radiation and convection:
    h_total = h_conv + h_rad.
    """
    h_r = radiation_heat_transfer_coefficient(T_surface_C, T_ambient_C, emissivity)
    h_tot = h_conv + h_r
    q_tot = h_tot * (T_surface_C - T_ambient_C)
    return {
        "h_total": h_tot,
        "h_radiation": h_r,
        "h_convection": h_conv,
        "q_total_W_m2": q_tot,
    }
