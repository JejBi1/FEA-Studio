"""Calculator of the heat transfer by convection (D4).

A thin layer / adapter over the module `fea_core.calc.thermal_convection` with a common interface for
KINDS, the GUI calculator dialog and the MCP tool `fea_calc_convection`.

All physical properties of the fluids (air, water) and the correlations are implemented
in `fea_core.calc.thermal_convection` as the single source of truth.
"""
from __future__ import annotations

import math
from .common import CalcError, positive, number, choice, result
from .thermal_convection import (
    G,
    fluid_properties,
    natural_convection_vertical_plate,
    natural_convection_horizontal_cylinder,
    natural_convection_horizontal_plate,
    natural_convection_finned_heatsink,
    forced_convection_flat_plate,
    forced_convection_internal_pipe,
    gnielinski_correlation,
    dittus_boelter_correlation,
)


def air_properties(T_film_C: float) -> tuple[float, float, float, float, float]:
    """Properties of dry air at 1 atm from `fluid_properties`.
    Returns (rho [kg/m³], cp [J/(kg·K)], k [W/(m·K)], nu [m²/s], Pr [-]).
    """
    try:
        p = fluid_properties("air", T_film_C)
    except ValueError as e:
        raise CalcError(str(e))
    return p["rho"], p["cp"], p["k"], p["nu"], p["Pr"]


def natural_vertical_plate(L_mm: float = 1000.0, Ts: float = 60.0, Tinf: float = 20.0,
                           W_mm: float | None = None, **kwargs) -> dict | tuple[float, float, float]:
    """Natural convection on a vertical plate of height L_mm.
    Churchill & Chu (1975) – valid for the whole region of the laminar and turbulent flow.
    The named arguments height_m / t_surface_c / t_ambient_c (m, °C) return the bare triple
    (h, Nu, Ra) without rounding for the verification task HT1.
    """
    if "height_m" in kwargs or "t_surface_c" in kwargs:
        H_m = float(kwargs.get("height_m", L_mm / 1000.0 if L_mm > 20 else L_mm))
        ts = float(kwargs.get("t_surface_c", Ts))
        ti = float(kwargs.get("t_ambient_c", Tinf))
        fl = kwargs.get("fluid", "air")
        raw = natural_convection_vertical_plate(H_m, ts, ti, fluid=fl)
        return raw["h"], raw["Nu"], raw["Ra"]

    L = positive("L_mm (plate height)", L_mm) / 1000.0  # m
    ts = number("Ts (surface temperature)", Ts)
    ti = number("Tinf (ambient temperature)", Tinf)

    try:
        raw = natural_convection_vertical_plate(L, ts, ti, fluid="air")
    except ValueError as e:
        raise CalcError(str(e))

    props = fluid_properties("air", raw["T_film"])
    h = raw["h"]
    q = raw["q"]

    res = {
        "h": round(h, 3),
        "Nu": round(raw["Nu"], 2),
        "Ra": float(f"{raw['Ra']:.3e}"),
        "q_W_m2": round(q, 2),
        "regime": raw["regime"],
        "k_air": round(props["k"], 4),
        "T_film_C": round(raw["T_film"], 1),
    }
    if W_mm is not None:
        W = positive("W_mm (plate width)", W_mm) / 1000.0
        area = L * W  # 1 strana
        res["area_m2"] = round(area, 4)
        res["Q_1side_W"] = round(q * area, 2)
        res["Q_2sides_W"] = round(2.0 * q * area, 2)

    return result(res,
                  "Nu = (0.825 + 0.387·Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27))^2; h = Nu·k/L",
                  "Churchill & Chu (1975), Incropera ch. 9.6")


def natural_horizontal_cylinder(D_mm: float = 50.0, Ts: float = 80.0, Tinf: float = 20.0,
                                L_mm: float | None = None, **kwargs) -> dict | tuple[float, float, float]:
    """Natural convection on a horizontal cylinder / pipe of outer diameter D_mm.
    Churchill & Chu (1975) for horizontal cylinders (Ra <= 10^12).
    The named arguments diameter_m / t_surface_c / t_ambient_c (m, °C) return the bare triple (h, Nu, Ra) (HT1).
    """
    if "diameter_m" in kwargs or "t_surface_c" in kwargs:
        D = float(kwargs.get("diameter_m", D_mm / 1000.0 if D_mm > 1 else D_mm))
        ts = float(kwargs.get("t_surface_c", Ts))
        ti = float(kwargs.get("t_ambient_c", Tinf))
        fl = kwargs.get("fluid", "air")
        raw = natural_convection_horizontal_cylinder(D, ts, ti, fluid=fl)
        return raw["h"], raw["Nu"], raw["Ra"]

    D = positive("D_mm (cylinder diameter)", D_mm) / 1000.0
    ts = number("Ts (surface temperature)", Ts)
    ti = number("Tinf (ambient temperature)", Tinf)

    try:
        raw = natural_convection_horizontal_cylinder(D, ts, ti, fluid="air")
    except ValueError as e:
        raise CalcError(str(e))

    props = fluid_properties("air", raw["T_film"])
    h = raw["h"]
    q = raw["q"]

    res = {
        "h": round(h, 3),
        "Nu": round(raw["Nu"], 2),
        "Ra": float(f"{raw['Ra']:.3e}"),
        "q_W_m2": round(q, 2),
        "regime": raw["regime"],
        "k_air": round(props["k"], 4),
        "T_film_C": round(raw["T_film"], 1),
    }
    if L_mm is not None:
        L = positive("L_mm (cylinder length)", L_mm) / 1000.0
        area = math.pi * D * L
        res["area_m2"] = round(area, 4)
        res["Q_W"] = round(q * area, 2)

    return result(res,
                  "Nu = (0.60 + 0.387·Ra^(1/6) / [1 + (0.559/Pr)^(9/16)]^(8/27))^2; h = Nu·k/D",
                  "Churchill & Chu (1975), Incropera ch. 9.6")


def natural_horizontal_plate(L_mm: float, W_mm: float, Ts: float, Tinf: float = 20.0,
                             orientation: str = "upper_hot") -> dict:
    """Natural convection on a horizontal plate:
    orientation:
    - 'upper_hot': upper surface of a hot plate (Ts > Tinf) or lower of a cold one (intensive buoyancy)
    - 'lower_hot': lower surface of a hot plate (Ts > Tinf) or upper of a cold one (suppressed buoyancy)
    """
    L = positive("L_mm", L_mm) / 1000.0
    W = positive("W_mm", W_mm) / 1000.0
    ts = number("Ts", Ts)
    ti = number("Tinf", Tinf)

    orient = choice("orientation", orientation, ("upper_hot", "lower_hot"))

    try:
        raw = natural_convection_horizontal_plate(L, W, ts, ti, fluid="air", orientation=orient)
    except ValueError as e:
        raise CalcError(str(e))

    area = raw["area"]
    q = raw["q"]
    return result({
        "h": round(raw["h"], 3),
        "Nu": round(raw["Nu"], 2),
        "Ra": float(f"{raw['Ra']:.3e}"),
        "L_char_mm": round(raw["L_char"] * 1000.0, 1),
        "orientation": raw["orientation"],
        "regime": raw["regime"],
        "q_W_m2": round(q, 2),
        "Q_W": round(q * area, 2),
        "area_m2": round(area, 4),
        "T_film_C": round(raw["T_film"], 1),
    }, "L_char = A/P; Nu = C·Ra^n (Lloyd & Moran, Goldstein); h = Nu·k/L_char",
       "Incropera & DeWitt ch. 9.5.2")


def forced_flat_plate(L_mm: float, velocity_m_s: float, Ts: float = 50.0, Tinf: float = 20.0,
                      W_mm: float | None = None) -> dict:
    """Forced convection along a flat plate of length L_mm in the direction of the flow, velocity velocity_m_s.
    Laminar (Re < 5e5) and mixed / turbulent flow.
    """
    L = positive("L_mm", L_mm) / 1000.0
    v = positive("velocity_m_s (velocity)", velocity_m_s)
    ts = number("Ts", Ts)
    ti = number("Tinf", Tinf)

    try:
        raw = forced_convection_flat_plate(L, v, ts, ti, fluid="air")
    except ValueError as e:
        raise CalcError(str(e))

    q = raw["q"]
    res = {
        "h": round(raw["h"], 3),
        "Nu": round(raw["Nu"], 2),
        "Re": float(f"{raw['Re']:.3e}"),
        "Pr": round(raw["Pr"], 3),
        "regime": raw["regime"],
        "q_W_m2": round(q, 2),
        "T_film_C": round(raw["T_film"], 1),
    }
    if W_mm is not None:
        W = positive("W_mm", W_mm) / 1000.0
        area = L * W
        res["area_m2"] = round(area, 4)
        res["Q_1side_W"] = round(q * area, 2)

    return result(res,
                  "Re = v·L/ν; Nu = f(Re, Pr); h = Nu·k/L",
                  "Incropera & DeWitt ch. 7.2–7.3")


def forced_internal_pipe(D_mm: float = 50.0, velocity_m_s: float = 1.0, L_mm: float = 1000.0,
                         Ts: float = 50.0, Tinf: float = 20.0, **kwargs) -> dict | tuple[float, float, float]:
    """Forced flow inside a smooth pipe of diameter D_mm (Gnielinski for turbulent flow).
    The named arguments diameter_m / velocity_m_s / t_bulk_c (m, m/s, °C; default water, Dittus-Boelter, the wall 5 K
    warmer than the fluid) return the bare triple (h, Nu, Re) (verification task HT1).
    """
    if "diameter_m" in kwargs or "t_bulk_c" in kwargs:
        D = float(kwargs.get("diameter_m", D_mm / 1000.0 if D_mm > 1 else D_mm))
        v = float(kwargs.get("velocity_m_s", velocity_m_s))
        tb = float(kwargs.get("t_bulk_c", Tinf))
        fl = kwargs.get("fluid", "water")
        corr = kwargs.get("correlation", "dittus_boelter")
        raw = forced_convection_internal_pipe(D, v, Ts=tb + 5.0, Tinf=tb, fluid=fl, L=None, correlation=corr)
        return raw["h"], raw["Nu"], raw["Re"]

    D = positive("D_mm", D_mm) / 1000.0
    L = positive("L_mm", L_mm) / 1000.0
    v = positive("velocity_m_s", velocity_m_s)
    ts = number("Ts", Ts)
    ti = number("Tinf", Tinf)

    try:
        raw = forced_convection_internal_pipe(D, v, ts, ti, fluid="air", L=L, correlation="gnielinski")
    except ValueError as e:
        raise CalcError(str(e))

    q = raw["q"]
    area = math.pi * D * L
    return result({
        "h": round(raw["h"], 3),
        "Nu": round(raw["Nu"], 2),
        "Re": float(f"{raw['Re']:.3e}"),
        "Pr": round(raw["Pr"], 3),
        "friction_factor_f": round(raw["f"], 4),
        "regime": raw["regime"],
        "q_W_m2": round(q, 2),
        "Q_W": round(q * area, 2),
        "area_m2": round(area, 4),
    }, "Gnielinski (1976): Nu = (f/8)(Re-1000)Pr / [1 + 12.7(f/8)^0.5(Pr^(2/3) - 1)]; h = Nu·k/D",
       "Gnielinski (1976), VDI Wärmeatlas ch. G1")


def finned_heatsink_natural(H_mm: float, L_mm: float, S_mm: float,
                            t_fin_mm: float = 2.0, N_fins: int = 10,
                            Ts: float = 60.0, Tinf: float = 20.0,
                            k_fin: float = 170.0) -> dict:
    """Natural convection on a vertical finned heat sink (plate fins):
    - H_mm: height of the fins in the direction of the flow (vertical)
    - L_mm: length of the fins from the base (projection)
    - S_mm: pitch between the fins
    - t_fin_mm: thickness of one fin
    - N_fins: number of fins
    - k_fin: thermal conductivity of the material of the fin [W/(m·K)], default aluminium 170 W/(m·K)
    Bar-Cohen & Rohsenow (1984) optimum pitch + efficiency of the fin.
    """
    H = positive("H_mm", H_mm) / 1000.0
    L = positive("L_mm", L_mm) / 1000.0
    S = positive("S_mm", S_mm) / 1000.0
    t = positive("t_fin_mm", t_fin_mm) / 1000.0
    n_fins = int(positive("N_fins", N_fins))
    ts = number("Ts", Ts)
    ti = number("Tinf", Tinf)
    k_m = positive("k_fin", k_fin)

    try:
        raw = natural_convection_finned_heatsink(H, L, S, t, n_fins, ts, ti, k_fin=k_m, fluid="air")
    except ValueError as e:
        raise CalcError(str(e))

    return result({
        "h_channel": round(raw["h_channel"], 3),
        "h_effective": round(raw["h_effective"], 3),
        "fin_efficiency": round(raw["fin_efficiency"], 3),
        "S_opt_mm": round(raw["S_opt"] * 1000.0, 1),
        "Ra_S": float(f"{raw['Ra_S']:.3e}"),
        "A_total_m2": round(raw["A_total"], 4),
        "Q_total_W": round(raw["Q_total"], 2),
        "heat_flux_avg_W_m2": round(raw["q_avg"], 1),
    }, "Bar-Cohen & Rohsenow (1984); fin efficiency η = tanh(m·Lc)/(m·Lc); Q = h(η·A_fin + A_base)·ΔT",
       "Bar-Cohen & Rohsenow (1984), Incropera ch. 9.7 and 3.6")


KINDS = {
    "vertical_plate": (natural_vertical_plate, "natural convection – vertical plate (Churchill & Chu)"),
    "horizontal_cylinder": (natural_horizontal_cylinder, "natural convection – horizontal cylinder / pipe"),
    "horizontal_plate": (natural_horizontal_plate, "natural convection – horizontal plate"),
    "finned_heatsink": (finned_heatsink_natural, "natural convection – finned heat sink (Bar-Cohen)"),
    "forced_flat_plate": (forced_flat_plate, "forced convection – flow over a plate"),
    "forced_internal_pipe": (forced_internal_pipe, "forced convection – pipe flow (Gnielinski)"),
}
