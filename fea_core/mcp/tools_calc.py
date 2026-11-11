"""Tools: analytical calculators without a mesh (stage F8.2, fea_core.calc) – snap-fit, press fit, Hertz, friction in joints and machine parts, friction coefficient library, mechanism references, damping from measurement, a fall as a pulse.

They do not need an open study. Units mm, N, MPa, t, s (g = 9810 mm/s², moment N·mm); result keys carry the unit ("p_MPa", "T_Nmm"), "formula" = formula, "source" = source.
"""
from __future__ import annotations

from .common import out
from .session import FeaError, Session


def _kind_doc(group: str, head: str) -> str:
    from ..calc.common import kinds, signature_text
    lines = [head, "", "kind and parameters (params = {name: value}):"]
    for k, (fn, desc) in kinds(group).items():
        lines.append(f"- {k}: {desc} – ({signature_text(fn)})")
    return "\n".join(lines)


def _call(group: str, kind: str, params: dict | None) -> str:
    from ..calc import CalcError, call
    if params is not None and not isinstance(params, dict):
        raise FeaError("params must be an object {parameter name: value}")
    try:
        r = call(group, kind, params or {})
    except CalcError as e:
        raise FeaError(str(e)) from None
    return out({"kind": kind, **r})


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_calc_snapfit(l_mm: float, h_mm: float, b_mm: float, E_MPa: float, y_mm: float, alpha_deg: float = 30.0,
                         mu: float = 0.3, alpha_release_deg: float | None = None, h_ratio: float = 1.0,
                         b_ratio: float = 1.0, tip_rotation: bool = True, eps_allow_pct: float | None = None) -> str:
        """Rectangular snap-fit hook (Bayer): root arm length l, thickness h, width b, secant modulus E, deflection y, lead-in angle alpha, friction mu (assembly: mu_max), retention angle alpha_release (separation force; non-releasable when mu*tan >= 1), taper h_ratio/b_ratio (tip/root, Bayer type 2: h_ratio 0.5), tip_rotation (own tip-rotation correction alpha + theta, NOT VERIFIED), eps_allow_pct [%] -> utilisation. Returns strain, P, W, W', theta, warnings."""
        from ..calc import CalcError, snapfit
        try:
            r = snapfit.snap_fit(l_mm, h_mm, b_mm, E_MPa, y_mm, alpha_deg, mu, alpha_release_deg, h_ratio, b_ratio,
                                 tip_rotation, eps_allow_pct)
        except CalcError as e:
            raise FeaError(str(e)) from None
        return out(r)

    @mcp.tool(structured_output=False)
    def fea_calc_pressfit(d_mm: float, D_mm: float, L_mm: float, delta_mm: float, E_hub_MPa: float = 210000.0,
                          nu_hub: float = 0.3, E_shaft_MPa: float | None = None, nu_shaft: float | None = None,
                          d_inner_mm: float = 0.0, mu: float = 0.15, mu_press: float | None = None,
                          alpha_hub_1_K: float | None = None, mount_clearance_mm: float = 0.0,
                          T_ambient_C: float = 20.0, Re_hub_MPa: float | None = None,
                          Re_shaft_MPa: float | None = None) -> str:
        """Press fit shaft/hub (Lame, plane stress): joint d, hub outer D, length L, MEAN interference delta [mm]; E, nu of hub and shaft (shaft None = hub), d_inner = hollow shaft bore. mu = friction for torque (mu_min), mu_press = during pressing (mu_max). Returns pressure p, sigma_t/sigma_r at surfaces, von Mises, u_r, torque T = mu*p*pi*d^2*L/2, axial force, pressing force; with alpha_hub_1_K the assembly heating dT = (delta + s)/(alpha*d); with Re the safety."""
        from ..calc import CalcError, pressfit
        try:
            r = pressfit.press_fit(d_mm, D_mm, L_mm, delta_mm, E_hub_MPa, nu_hub, E_shaft_MPa, nu_shaft, d_inner_mm,
                                   mu, mu_press, alpha_hub_1_K, mount_clearance_mm, T_ambient_C, Re_hub_MPa,
                                   Re_shaft_MPa)
        except CalcError as e:
            raise FeaError(str(e)) from None
        return out(r)

    def fea_calc_hertz(kind: str, params: dict | None = None) -> str:
        return _call("hertz", kind, params)
    fea_calc_hertz.__doc__ = _kind_doc(
        "hertz", "Hertz contact (Johnson): sphere, cylinder, gears (ISO 6336-2 sigma_H0), eccentric cam with roller, sphere impact. E [MPa], F [N], lengths [mm], v [mm/s], rho [t/mm3]; E2=None = rigid counterpart, R2=None = plane, R2<0 = concave.")
    mcp.tool(structured_output=False)(fea_calc_hertz)

    def fea_calc_friction(kind: str, params: dict | None = None) -> str:
        return _call("friction", kind, params)
    fea_calc_friction.__doc__ = _kind_doc(
        "friction", "Friction in joints/machine parts (F8 3.3.2): torque/force from reaction, self-locking, efficiency. Choose mu by purpose (fea_calc_friction_library): joint slip -> mu_min, mechanism resistance -> mu_max. N, mm, deg, N*mm. Snap-fit: fea_calc_snapfit.")
    mcp.tool(structured_output=False)(fea_calc_friction)

    @mcp.tool(structured_output=False)
    def fea_calc_friction_library(action: str = "list", name: str | None = None, purpose: str = "slip",
                                  material_a: str | None = None, material_b: str | None = None,
                                  lubricated: bool | None = None, surface_class: str | None = None,
                                  pair: dict | None = None) -> str:
        """Friction coefficient library with sources (built-in + user friction.json). action: list | get (name) | mu (name, purpose slip -> mu_min | resistance -> mu_max) | suggest (material_a, material_b: names or families steel/aluminium/brass/POM/PTFE/PLA/PETG/ABS/PA/rubber/FR4; lubricated; surface_class A-D per EN 1090-2) | save (pair {name,a,b,lubricated,mu_s [min,max],mu_k,mu_min,mu_max,note,source}) | delete (name, user pairs only). INDICATIVE; "unverified" = estimate."""
        from ..calc import friction_lib as fl
        a = action.strip().lower()
        try:
            if a == "list":
                return out({"pairs": fl.table(), "purposes": fl.PURPOSES, "rules": fl.RULES,
                            "user_library": str(fl.user_library_path())})
            if a in ("get", "mu", "delete") and not name:
                raise FeaError(f"action={a} needs name (fea_calc_friction_library action=list)")
            if a == "get":
                p = fl.get(name)
                return out({**p.to_dict(), "builtin": p.builtin})
            if a == "mu":
                p = fl.get(name)
                return out({"name": p.name, "purpose": purpose, "mu": p.mu(purpose),
                            "unverified": p.unverified_value(purpose), "rule": fl.PURPOSES.get(purpose),
                            "source": p.source, "note": p.note})
            if a == "suggest":
                if not material_a or not material_b:
                    raise FeaError("action=suggest needs material_a and material_b")
                s = fl.suggest(material_a, material_b, lubricated, surface_class)
                return out({"families": [fl.family(material_a), fl.family(material_b)], "pairs": s,
                            **({} if s else {"hint": "The pair is not in the library – measure it (inclined plane μs = tan α, fea_calc_friction kind=inclined_plane) and store it with action=save"})})
            if a == "save":
                if not isinstance(pair, dict):
                    raise FeaError("action=save needs pair = {name, a, b, mu_min, mu_max, …}")
                p = fl.save_pair(pair)
                return out({"saved": p.name, "path": str(fl.user_library_path())})
            if a == "delete":
                fl.delete_pair(name)
                return out({"deleted": name})
        except fl.FrictionLibError as e:
            raise FeaError(str(e)) from None
        raise FeaError("action must be list | get | mu | suggest | save | delete")

    def fea_calc_mechanism(kind: str, params: dict | None = None) -> str:
        return _call("mechanism", kind, params)
    fea_calc_mechanism.__doc__ = _kind_doc(
        "mechanism", "Analytic mechanism benchmarks (no solver): crank, four-bar (Freudenstein, Grashof), cam with roller, pendulum, coast-down with friction, stick-slip, rolling, reduced inertia/run-up, residual vibration after a pulse, axis S-curve. Angles deg, omega rad/s, mm, t, J t*mm2, g = 9810.")
    mcp.tool(structured_output=False)(fea_calc_mechanism)

    def fea_calc_convection(kind: str, params: dict | None = None) -> str:
        return _call("convection", kind, params)
    fea_calc_convection.__doc__ = _kind_doc(
        "convection", "Convective heat transfer (D4): natural (vertical plate, horizontal cylinder/plate, finned sink) and forced (flat plate, pipe). Returns h [W/(m2*K)], Nu, Ra/Re, q [W/m2].")
    mcp.tool(structured_output=False)(fea_calc_convection)

    def fea_calc_damping(kind: str, params: dict | None = None) -> str:
        return _call("damping", kind, params)
    fea_calc_damping.__doc__ = _kind_doc(
        "damping", "Damping from measurement and conversions: -3 dB bandwidth, log decrement, Rayleigh alpha/beta from two frequencies, zeta <-> Q <-> eta <-> delta (zeta as fraction), PCB Q = sqrt(f_n), table of indicative zeta. ccx *DAMPING STRUCTURAL: eta = 2*zeta.")
    mcp.tool(structured_output=False)(fea_calc_damping)

    def fea_calc_drop(kind: str = "drop", params: dict | None = None) -> str:
        return _call("pulse", kind, params)
    fea_calc_drop.__doc__ = _kind_doc(
        "pulse", "Drop as an equivalent half-sine acceleration pulse (shock input): h [mm], tau [s] or m [t] + impact stiffness k [N/mm], restitution e. Returns v, tau, A [mm/s2, g], dv.")
    mcp.tool(structured_output=False)(fea_calc_drop)
