"""Calculator of the heat transfer by convection without the need of FEA (stage F9.6, D4).

It contains analytical and semi-tabular relations for the heat transfer by natural and forced convection:
- Properties of the fluids (air, water) according to the temperature (density, viscosity, thermal conductivity, Pr, beta)
- Natural convection:
    * vertical plate (Churchill & Chu for the whole range of Ra)
    * horizontal cylinder / pipe (Churchill & Chu)
    * horizontal plate (the heated surface upwards / downwards)
    * finned heat sink (Elenbaas, Bar-Cohen & Rohsenow optimum pitch + efficiency of the fin)
- Forced convection:
    * flow over a flat plate (laminar / turbulent)
    * internal flow in a pipe (Gnielinski and Dittus-Boelter for the transitional and turbulent region)

Sources:
- Churchill, S.W. and Chu, H.H.S. (1975): Correlating equations for laminar and turbulent free convection from a vertical plate / horizontal cylinder
- Gnielinski, V. (1976): New equations for heat and mass transfer in turbulent pipe and channel flow
- Dittus, F.W. and Boelter, L.M.K. (1930): Heat transfer in automobile radiators of the tubular type
- Bar-Cohen, A. and Rohsenow, W.M. (1984): Thermally optimum spacing of vertical, natural convection cooled, parallel plates
- Incropera, F.P., DeWitt, D.P.: Fundamentals of Heat and Mass Transfer (Wiley)
- Černoch, S.: Strojně technická příručka, SNTL Praha
- VDI Wärmeatlas (11th edition)
"""
from __future__ import annotations

import math
from typing import Literal

G = 9.80665  # gravitational acceleration [m/s²]


def _water_density(T_C: float) -> float:
    """Hustota vody [kg/m³] (Tanaka et al. 2001, 0–40 °C; do 150 °C odchylka do ~0,5 %)."""
    return 1000.0 * (1.0 - (T_C + 288.9414) / (508929.2 * (T_C + 68.12963)) * ((T_C - 3.9863) ** 2))


def fluid_properties(fluid: Literal["air", "water"] = "air", T_C: float = 20.0) -> dict[str, float]:
    """Physical properties of the fluid at 1 atm as a function of the film temperature T_C [°C].

    Returns a dictionary with the keys:
        rho: density [kg/m³]
        cp: specific heat capacity [J/(kg·K)]
        k: thermal conductivity [W/(m·K)]
        mu: dynamic viscosity [Pa·s]
        nu: kinematic viscosity [m²/s]
        Pr: Prandtl number [-]
        beta: coefficient of the volumetric thermal expansion [1/K]
        T_C: temperature [°C]
        fluid: name of the fluid
    """
    fluid_name = fluid.strip().lower()
    T_k = T_C + 273.15

    if fluid_name in ("air", "vzduch"):
        if T_k <= 50.0 or T_k >= 1500.0:
            raise ValueError(f"Temperature {T_C:.1f} °C is outside the validity range of the air approximation (50 K – 1500 K)")

        # density of the ideal gas at p = 101325 Pa
        rho = 101325.0 / (287.058 * T_k)

        # specific heat cp [J/(kg·K)] (approximation from VDI / Incropera)
        cp = 1005.0 + 0.05 * max(T_C, 0.0) + 1.2e-4 * max(T_C, 0.0) ** 2

        # dynamic viscosity mu [Pa·s] (Sutherland)
        mu = 1.458e-6 * (T_k ** 1.5) / (T_k + 110.4)

        # kinematic viscosity nu [m²/s]
        nu = mu / rho

        # thermal conductivity k [W/(m·K)] (Sutherland / Eucken formula for air)
        k = 2.646e-3 * (T_k ** 1.5) / (T_k + 245.4 * (10.0 ** (-12.0 / T_k)))

        # Prandtl number Pr [-]
        Pr = cp * mu / k

        # thermal expansion beta [1/K] for the ideal gas
        beta = 1.0 / T_k

        return {
            "rho": rho,
            "cp": cp,
            "k": k,
            "mu": mu,
            "nu": nu,
            "Pr": Pr,
            "beta": beta,
            "T_C": T_C,
            "fluid": "air",
        }

    elif fluid_name in ("water", "voda"):
        if T_C < 0.0 or T_C > 150.0:
            raise ValueError(f"Water temperature {T_C:.1f} °C is outside the liquid range (0 °C – 150 °C)")

        rho = _water_density(T_C)

        # dynamic viscosity [Pa·s] (Vogel equation)
        mu = 2.414e-5 * (10.0 ** (247.8 / (T_C + 133.15)))

        # kinematic viscosity [m²/s]
        nu = mu / rho

        # thermal conductivity [W/(m·K)] (quadratic approximation of the tables, 0–150 °C to ~1 %: 20 °C 0.603; 100 °C 0.682)
        k = 0.5706 + 0.001756 * T_C - 6.46e-6 * (T_C ** 2)

        # specific heat cp [J/(kg·K)]
        cp = 4217.4 - 3.720283 * T_C + 0.1412855 * (T_C ** 2) - 0.002654387 * (T_C ** 3) + 2.093236e-5 * (T_C ** 4)

        # Prandtl number
        Pr = cp * mu / k

        # Coefficient of the volumetric expansion beta = −(1/ρ)·dρ/dT [1/K] from the derivative of the density (20 °C: 2.1e-4, 50 °C: 4.6e-4);
        # below ~4 °C it is negative (the anomaly of water) – natural convection cannot be handled there by this formula → the lower limit
        beta = max(1e-6, -(_water_density(T_C + 0.5) - _water_density(T_C - 0.5)) / rho)

        return {
            "rho": rho,
            "cp": cp,
            "k": k,
            "mu": mu,
            "nu": nu,
            "Pr": Pr,
            "beta": beta,
            "T_C": T_C,
            "fluid": "water",
        }

    else:
        raise ValueError(f"Unknown fluid '{fluid}'. Supported are 'air' and 'water'.")


def natural_convection_vertical_plate(L: float, Ts: float, Tinf: float, fluid: str = "air") -> dict[str, float]:
    """Natural convection on a vertical plate of height L [m] according to Churchill & Chu (1975).

    Valid for the whole region of the laminar and turbulent flow (all Ra).

    Parameters:
        L: characteristic height of the plate [m]
        Ts: surface temperature [°C]
        Tinf: temperature of the surrounding fluid [°C]
        fluid: 'air' or 'water'

    Returns:
        dict with the keys: h [W/(m²·K)], Nu [-], Ra [-], Gr [-], Pr [-], q [W/m²], T_film [°C], regime
    """
    if L <= 0:
        raise ValueError(f"Dimension L must be positive, entered {L}")

    dT = abs(Ts - Tinf)
    T_film = (Ts + Tinf) / 2.0
    props = fluid_properties(fluid, T_film)

    if dT < 1e-6:
        return {
            "h": 0.0,
            "Nu": 0.0,
            "Ra": 0.0,
            "Gr": 0.0,
            "Pr": props["Pr"],
            "q": 0.0,
            "T_film": T_film,
            "regime": "equilibrium (dT=0)",
        }

    beta = props["beta"]
    nu = props["nu"]
    Pr = props["Pr"]
    k = props["k"]

    Gr = (G * beta * dT * (L ** 3)) / (nu ** 2)
    Ra = Gr * Pr

    # Churchill & Chu (1975) correlation for the vertical plate
    f_Pr = (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    Nu = (0.825 + (0.387 * (Ra ** (1.0 / 6.0))) / f_Pr) ** 2
    h = Nu * k / L
    q = h * (Ts - Tinf)
    regime = "laminar" if Ra < 1e9 else "turbulent"

    return {
        "h": h,
        "Nu": Nu,
        "Ra": Ra,
        "Gr": Gr,
        "Pr": Pr,
        "q": q,
        "T_film": T_film,
        "regime": regime,
    }


def natural_convection_horizontal_cylinder(D: float, Ts: float, Tinf: float, fluid: str = "air") -> dict[str, float]:
    """Natural convection on a horizontal cylinder / pipe of diameter D [m] according to Churchill & Chu (1975).

    Valid for Ra <= 10^12.

    Parameters:
        D: outer diameter of the cylinder [m]
        Ts: surface temperature [°C]
        Tinf: temperature of the surrounding fluid [°C]
        fluid: 'air' or 'water'

    Returns:
        dict with the keys: h [W/(m²·K)], Nu [-], Ra [-], Gr [-], Pr [-], q [W/m²], T_film [°C], regime
    """
    if D <= 0:
        raise ValueError(f"Diameter D must be positive, entered {D}")

    dT = abs(Ts - Tinf)
    T_film = (Ts + Tinf) / 2.0
    props = fluid_properties(fluid, T_film)

    if dT < 1e-6:
        return {
            "h": 0.0,
            "Nu": 0.0,
            "Ra": 0.0,
            "Gr": 0.0,
            "Pr": props["Pr"],
            "q": 0.0,
            "T_film": T_film,
            "regime": "equilibrium (dT=0)",
        }

    beta = props["beta"]
    nu = props["nu"]
    Pr = props["Pr"]
    k = props["k"]

    Gr = (G * beta * dT * (D ** 3)) / (nu ** 2)
    Ra = Gr * Pr

    # Churchill & Chu (1975) correlation for the horizontal cylinder
    f_Pr = (1.0 + (0.559 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    Nu = (0.60 + (0.387 * (Ra ** (1.0 / 6.0))) / f_Pr) ** 2
    h = Nu * k / D
    q = h * (Ts - Tinf)
    regime = "laminar" if Ra < 1e9 else "turbulent"

    return {
        "h": h,
        "Nu": Nu,
        "Ra": Ra,
        "Gr": Gr,
        "Pr": Pr,
        "q": q,
        "T_film": T_film,
        "regime": regime,
    }


def natural_convection_horizontal_plate(L: float, W: float, Ts: float, Tinf: float,
                                       hot_side_up: bool = True, orientation: str | None = None,
                                       fluid: str = "air") -> dict[str, float]:
    """Natural convection on a horizontal plate of dimensions L × W [m] (Incropera ch. 9.6).

    Characteristic length L_c = A/P. A surface warmer than the surroundings and facing upwards (or colder and facing downwards) =
    favourable buoyancy (Nu = 0.54·Ra^¼ up to Ra = 1e7, beyond 0.15·Ra^⅓); the opposite suppressed (Nu = 0.52·Ra^⅕).

    Parameters:
        L: length of the plate [m]
        W: width of the plate [m]
        Ts: surface temperature [°C]
        Tinf: temperature of the surroundings [°C]
        hot_side_up: True = the warm surface upwards (or the cold one downwards; favourable buoyancy)
        orientation: 'upper_hot' or 'lower_hot' (overrides hot_side_up, if given)
        fluid: 'air' or 'water'
    """
    if L <= 0 or W <= 0:
        raise ValueError(f"Dimensions L and W must be positive, entered L={L}, W={W}")

    dT = abs(Ts - Tinf)
    T_film = (Ts + Tinf) / 2.0
    props = fluid_properties(fluid, T_film)

    area = L * W
    P = 2.0 * (L + W)
    Lc = area / P

    if dT < 1e-6:
        return {
            "h": 0.0,
            "Nu": 0.0,
            "Ra": 0.0,
            "Gr": 0.0,
            "Pr": props["Pr"],
            "q": 0.0,
            "Lc": Lc,
            "L_char": Lc,
            "area": area,
            "T_film": T_film,
            "orientation": orientation or ("upper_hot" if hot_side_up else "lower_hot"),
            "regime": "equilibrium (dT=0)",
        }

    beta = props["beta"]
    nu = props["nu"]
    Pr = props["Pr"]
    k = props["k"]

    Gr = (G * beta * dT * (Lc ** 3)) / (nu ** 2)
    Ra = Gr * Pr

    if orientation is not None:
        is_favorable = (orientation == "upper_hot" and Ts >= Tinf) or (orientation == "lower_hot" and Ts < Tinf)
    else:
        is_favorable = (hot_side_up and Ts >= Tinf) or ((not hot_side_up) and Ts < Tinf)

    if is_favorable:
        if Ra <= 1e7:
            Nu = 0.54 * (Ra ** 0.25)
            regime = "laminar (favorable buoyancy)"
        else:
            Nu = 0.15 * (Ra ** (1.0 / 3.0))
            regime = "turbulent (favorable buoyancy)"
    else:
        Nu = 0.52 * (Ra ** 0.2)          # Incropera 9.32 (7. vyd.): 1e4 ≤ Ra ≤ 1e9
        regime = "suppressed buoyancy"

    h = Nu * k / Lc
    q = h * (Ts - Tinf)

    return {
        "h": h,
        "Nu": Nu,
        "Ra": Ra,
        "Gr": Gr,
        "Pr": Pr,
        "Lc": Lc,
        "L_char": Lc,
        "area": area,
        "q": q,
        "T_film": T_film,
        "orientation": orientation or ("upper_hot" if hot_side_up else "lower_hot"),
        "regime": regime,
    }


def natural_convection_finned_heatsink(H: float, L: float, S: float, t_fin: float,
                                       N_fins: int, Ts: float, Tinf: float,
                                       k_fin: float = 170.0, fluid: str = "air") -> dict[str, float]:
    """Natural convection on a vertical finned heat sink (plate fins, Bar-Cohen & Rohsenow 1984).

    Parameters:
        H: height of the fins [m]
        L: length of the fins in the direction of the flow [m]
        S: gap (pitch) between the fins [m]
        t_fin: thickness of one fin [m]
        N_fins: number of fins
        Ts: temperature of the base / of the fins [°C]
        Tinf: temperature of the surroundings [°C]
        k_fin: thermal conductivity of the material of the fins [W/(m·K)]
        fluid: 'air' or 'water'
    """
    if min(H, L, S, t_fin) <= 0 or N_fins < 1:
        raise ValueError("All geometric dimensions and the number of fins must be positive")

    dT = abs(Ts - Tinf)
    T_film = (Ts + Tinf) / 2.0
    props = fluid_properties(fluid, T_film)

    if dT < 1e-6:
        return {
            "h_channel": 0.0,
            "h_effective": 0.0,
            "fin_efficiency": 1.0,
            "S_opt": 0.0,
            "Ra_S": 0.0,
            "A_total": 0.0,
            "Q_total": 0.0,
            "q_avg": 0.0,
            "T_film": T_film,
        }

    beta = props["beta"]
    nu = props["nu"]
    Pr = props["Pr"]
    k = props["k"]

    # Elenbaas Rayleigh number per channel of pitch S:
    Ra_S = (G * beta * dT * (S ** 4) * Pr) / ((nu ** 2) * H)

    # Bar-Cohen & Rohsenow (1984) composite Nusselt number for symmetric channels:
    Nu_S = ((576.0 / max(Ra_S ** 2, 1e-12)) + (2.873 / max(math.sqrt(Ra_S), 1e-6))) ** (-0.5)
    h_channel = Nu_S * k / S

    # Optimum pitch of the fins (Bar-Cohen & Rohsenow):
    c_opt = (G * beta * dT * Pr) / (nu ** 2)
    S_opt = 2.714 * ((H / max(c_opt, 1e-12)) ** 0.25)

    # Efficiency of one fin:
    m = math.sqrt(2.0 * h_channel / (max(k_fin, 1e-3) * t_fin))
    Lc = L + t_fin / 2.0
    eta_fin = math.tanh(m * Lc) / max(m * Lc, 1e-6)

    # Faces:
    A_fins = 2.0 * L * H * N_fins
    A_base = max(N_fins - 1, 1) * S * H
    A_total = A_fins + A_base

    # Total heat flow Q and the effective h:
    Q = h_channel * (eta_fin * A_fins + A_base) * (Ts - Tinf)
    h_eff = Q / max(A_total * (Ts - Tinf), 1e-9)

    return {
        "h_channel": h_channel,
        "h_effective": h_eff,
        "fin_efficiency": eta_fin,
        "S_opt": S_opt,
        "Ra_S": Ra_S,
        "A_total": A_total,
        "Q_total": Q,
        "q_avg": Q / max(A_total, 1e-6),
        "T_film": T_film,
    }


def forced_convection_flat_plate(L: float, u: float, Ts: float, Tinf: float, fluid: str = "air") -> dict[str, float]:
    """Forced convection in the longitudinal flow over a flat plate of length L [m] with the velocity u [m/s].

    Laminar (Re <= 5e5) and mixed / turbulent flow with the transition at Re_crit = 5e5.

    Parameters:
        L: length of the plate in the direction of the flow [m]
        u: velocity of the free stream [m/s]
        Ts: surface temperature of the plate [°C]
        Tinf: temperature of the incoming stream of the fluid [°C]
        fluid: 'air' or 'water'

    Returns:
        dict with the keys: h [W/(m²·K)], Nu [-], Re [-], Pr [-], q [W/m²], T_film [°C], regime
    """
    if L <= 0:
        raise ValueError(f"Length L must be positive, entered {L}")
    if u < 0:
        raise ValueError(f"Velocity u must not be negative, entered {u}")

    T_film = (Ts + Tinf) / 2.0
    props = fluid_properties(fluid, T_film)

    nu = props["nu"]
    Pr = props["Pr"]
    k = props["k"]
    Re = u * L / nu

    Re_crit = 5e5
    if Re <= Re_crit:
        # Laminar boundary layer: Nu = 0.664 * Re^(1/2) * Pr^(1/3)
        Nu = 0.664 * (Re ** 0.5) * (Pr ** (1.0 / 3.0))
        regime = "laminar"
    else:
        # Mixed laminar-turbulent: Nu = (0.037 * Re^(4/5) - 871) * Pr^(1/3)
        Nu = (0.037 * (Re ** 0.8) - 871.0) * (Pr ** (1.0 / 3.0))
        regime = "transitional / turbulent"

    h = Nu * k / L
    q = h * (Ts - Tinf)

    return {
        "h": h,
        "Nu": Nu,
        "Re": Re,
        "Pr": Pr,
        "q": q,
        "T_film": T_film,
        "regime": regime,
    }


def gnielinski_correlation(Re: float, Pr: float, f: float | None = None) -> float:
    """Gnielinski (1976) correlation for turbulent flow in a pipe:
    Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * sqrt(f / 8) * (Pr^(2/3) - 1))
    Validity: 3000 <= Re <= 5e6, 0.5 <= Pr <= 2000.
    """
    if Re < 2300:
        return 3.66  # asymptote for the fully developed laminar flow
    if f is None:
        # Petukhov relation for smooth pipes: f = (0.790 * ln(Re) - 1.64)^(-2)
        f = (0.790 * math.log(Re) - 1.64) ** (-2)

    top = (f / 8.0) * (Re - 1000.0) * Pr
    bot = 1.0 + 12.7 * math.sqrt(f / 8.0) * ((Pr ** (2.0 / 3.0)) - 1.0)
    return top / bot


def dittus_boelter_correlation(Re: float, Pr: float, heating: bool = True) -> float:
    """Dittus & Boelter (1930) correlation for fully turbulent flow in a pipe:
    Nu = 0.023 * Re^(0.8) * Pr^n
    where n = 0.4 for heating of the fluid (Ts > Tinf), n = 0.3 for cooling (Ts < Tinf).
    Validity: Re >= 10000, 0.6 <= Pr <= 160.
    """
    n = 0.4 if heating else 0.3
    return 0.023 * (Re ** 0.8) * (Pr ** n)


def forced_convection_internal_pipe(D: float, u: float, Ts: float, Tinf: float,
                                    fluid: str = "air", L: float | None = None,
                                    correlation: Literal["gnielinski", "dittus_boelter"] = "gnielinski"
                                    ) -> dict[str, float]:
    """Forced convection in the flow inside a smooth pipe of diameter D [m] with the velocity u [m/s].

    Parameters:
        D: inner diameter of the pipe [m]
        u: mean velocity of the flow [m/s]
        Ts: temperature of the wall of the pipe [°C]
        Tinf: mean temperature of the fluid in the core of the flow [°C]
        fluid: 'air' or 'water'
        L: length of the pipe [m] (optional, for the correction of the entrance length L/D)
        correlation: 'gnielinski' (default, more accurate) or 'dittus_boelter'

    Returns:
        dict with the keys: h [W/(m²·K)], Nu [-], Re [-], Pr [-], f [-], q [W/m²], T_film [°C], regime
    """
    if D <= 0:
        raise ValueError(f"Diameter D must be positive, entered {D}")
    if u < 0:
        raise ValueError(f"Velocity u must not be negative, entered {u}")

    T_film = (Ts + Tinf) / 2.0
    props = fluid_properties(fluid, T_film)

    nu = props["nu"]
    Pr = props["Pr"]
    k = props["k"]
    Re = u * D / nu

    if Re < 2300:
        Nu = 3.66  # constant wall temperature
        f = 64.0 / max(Re, 1e-4)
        regime = "laminar (Nu = 3.66)"
    else:
        # Coefficient of friction (Petukhov)
        f = (0.790 * math.log(Re) - 1.64) ** (-2)
        if correlation == "dittus_boelter":
            heating = Ts > Tinf
            Nu = dittus_boelter_correlation(Re, Pr, heating=heating)
            regime = "turbulent (Dittus-Boelter)"
            if Re < 1e4:
                regime += ", Re < 1e4 outside the validity range"
        else:
            Nu = gnielinski_correlation(Re, Pr, f=f)
            # Correction for the ratio L/D (entrance length according to VDI Wärmeatlas)
            if L is not None and L > 0 and (L / D) < 100:
                Nu *= (1.0 + (D / L) ** (2.0 / 3.0))
            regime = "transitional / turbulent (Gnielinski)"

    h = Nu * k / D
    q = h * (Ts - Tinf)

    return {
        "h": h,
        "Nu": Nu,
        "Re": Re,
        "Pr": Pr,
        "f": f,
        "q": q,
        "T_film": T_film,
        "regime": regime,
    }
