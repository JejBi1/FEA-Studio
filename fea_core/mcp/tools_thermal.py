"""Tools: heat – thermal states (steady / transient heat conduction, ccx *HEAT TRANSFER), their boundary
conditions and the thermal load of the mechanical states (thermal stress).

A boundary condition on a face is identified by (thermal state, kind, selection): the same kind on the same selection in the same
state = replacement (flux W/m² and power W are one kind); convection and radiation on the same face may coexist. The heat source
is per part. Selection of faces: name of a selection | id 'Díl#i' | query | selector (as for constraints); a new selection gets
the name name, otherwise an automatic one (teplota_1, konvekce_1 … = temperature_1, convection_1 …). The fluxes of the boundary conditions in the results are by selection.
"""
from __future__ import annotations

import math
from dataclasses import asdict

from .common import out, resolve_faces, same_faces, selection_info, selection_users
from .session import FeaError, Session

ABS_ZERO = -273.15
AUTO_NAMES = {"fixed_temperature": "temperature", "heat_flux": "tok", "heat_flow": "tok", "convection": "convection",
              "radiation": "radiation", "cavity_radiation": "cavity"}
UNITS = ("Heat: temperature °C, heat flux W/m², power W, heat transfer coefficient W/(m²·K), conductivity λ W/(m·K), expansion α 1/K, specific heat c_p J/(kg·K), time s.")


def _num(v, what: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise FeaError(f"{what}: a number expected") from None
    if not math.isfinite(x):
        raise FeaError(f"{what}: a finite number expected")
    return x


def _temp(v, what: str) -> float:
    x = _num(v, what)
    if x <= ABS_ZERO:
        raise FeaError(f"{what}: {x:g} °C is below absolute zero (temperatures are given in °C)")
    return x


def _group(kind: str) -> str:
    return "flux" if kind in ("heat_flux", "heat_flow") else kind


def bc_key(bc) -> tuple:
    return _group(bc.kind), getattr(bc, "selection", None) or getattr(bc, "part", "")


def describe_bc(bc) -> dict:
    from .diagnostics import THERMAL_LABELS, bc_value_text
    d = asdict(bc)
    d["name"] = getattr(bc, "selection", None) or bc.part
    d["label"] = THERMAL_LABELS.get(bc.kind, bc.kind)
    d["value_text"] = bc_value_text(bc)
    return d


def describe_case(st, c) -> dict:
    d = {"name": c.name, "transient": c.transient, "bcs": [describe_bc(b) for b in c.loads]}
    if c.transient:
        from ..thermal.inp import time_stepping
        d.update({"time_s": c.time, "dt_s": c.dt, "outputs": c.outputs})
        if c.time > 0 and c.outputs >= 1:
            n, freq, dt = time_stepping(c)
            d["increments"] = {"count": n, "dt_s": dt, "snapshot_every": freq}
    used = [lc.name for lc in st.load_cases for l in lc.loads if l.kind == "thermal" and l.case == c.name]
    if used:
        d["used_by_load_cases"] = used
    return d


def conductance_overview(st) -> list[dict]:
    """Contact conductances: interfaces of the Heat step + contact pairs with h_c (they apply to all thermal cases)."""
    rows = [{"name": c.name, "side_a": c.selection, "side_b": c.master, "h_W_m2K": c.conductance,
             "R_m2K_W": c.resistance, "source": "interface"} for c in st.thermal.conductances]
    rows += [{"name": c.name, "side_a": c.a, "side_b": c.b, "h_W_m2K": c.conductance, "R_m2K_W": 1.0 / c.conductance,
              "source": "contact_pair"} for c in st.contacts if c.conductance]
    return rows


def thermal_overview(st) -> dict:
    from .tools_study import describe_load
    th = st.thermal
    d = {"reference_temperature_C": th.reference_temperature,
         "initial_temperature_C": th.initial_temperature if th.initial_temperature is not None
         else f"= reference ({th.reference_temperature:g})",
         "cases": [describe_case(st, c) for c in th.cases],
         "temperature_loads": [{"load_case": lc.name, **describe_load(l)} for lc in st.load_cases
                               for l in lc.loads if l.kind == "thermal"]}
    cond = conductance_overview(st)
    if cond:
        d["contact_conductances"] = cond
    return d


def get_thermal_case(st, name: str, create: bool = True):
    for c in st.thermal.cases:
        if c.name == name:
            return c, False
    if not create:
        raise FeaError(f"Thermal state '{name}' does not exist (thermal states: {', '.join((c.name for c in st.thermal.cases)) or 'none'})")
    if not str(name).strip():
        raise FeaError("The thermal state name must not be empty")
    from ..thermal.model import ThermalCase
    c = ThermalCase(name)
    st.thermal.cases.append(c)
    return c, True


def _thermal_selection(sess: Session, selection, name: str | None, kind: str, replacing: list) -> str:
    """Name of the selection for a boundary condition; a new selection is created (the name name or an automatic one)."""
    st = sess.require_study()
    if isinstance(selection, str) and selection in st.selections and (name is None or name == selection):
        return selection
    faces = resolve_faces(sess, selection)
    if name is None:
        base = AUTO_NAMES.get(kind, "teplo")
        for old in st.selections:                         # the same faces already have a selection with an automatic name
            if old.startswith(base + "_") and same_faces(faces, st.selections[old].faces):
                return old
        i = 1
        while f"{base}_{i}" in st.selections:
            i += 1
        name = f"{base}_{i}"
    elif name in st.selections:
        users = [w for w, o in selection_users(st, name) if not any(o is r for r in replacing)]
        if users and not same_faces(faces, st.selections[name].faces):
            raise FeaError(f"Selection '{name}' is already used by {users[0]} – other faces would change it as well. Choose another name (name), or give selection='{name}' (the same faces)")
    st.add_selection(name, faces)
    return name


def add_thermal_bc(sess: Session, case_name: str, kind: str, selection=None, name: str | None = None,
                   part: str | None = None, **params) -> dict:
    """General addition of a thermal boundary condition of the kind kind (fea_core.thermal.model.THERMAL_KINDS) into the state
    (it creates the state as steady if it is missing). The same kind on the same selection/part = replacement.
    """
    from ..thermal.model import THERMAL_KINDS
    st = sess.require_study()
    if kind not in THERMAL_KINDS:
        raise FeaError(f"Unknown kind of thermal boundary condition '{kind}' (known: {', '.join(THERMAL_KINDS)})")
    klass = THERMAL_KINDS[kind]
    if kind == "heat_generation":
        names = sess.part_names()
        if part not in names:
            raise FeaError(f"Part '{part}' does not exist. Parts: {', '.join(names)}")
        if part in st.excluded_parts():
            raise FeaError(f"Part '{part}' is not meshed (bolt / omitted part) – a heat source cannot be given in it")
        case, created = get_thermal_case(st, case_name)
        old = [b for b in case.loads if bc_key(b) == (kind, part)]
        obj = klass(part=part, **params)
        info = None
    else:
        case = next((c for c in st.thermal.cases if c.name == case_name), None)
        guess = name or (selection if isinstance(selection, str) and selection in st.selections else None)
        old = [b for b in (case.loads if case else []) if guess and bc_key(b) == (_group(kind), guess)]
        sel = _thermal_selection(sess, selection, name, kind, old)
        case, created = get_thermal_case(st, case_name)
        old = [b for b in case.loads if bc_key(b) == (_group(kind), sel)]
        obj = klass(selection=sel, **params)
        info = selection_info(sess, sel)
    case.loads = [b for b in case.loads if not any(b is o for o in old)] + [obj]
    sess.changed()
    d = {"thermal_case": case.name, "bc": describe_bc(obj), "replaced": bool(old), "case_created": created}
    if info:
        d["selection"] = info
    return d


def register(mcp, sess: Session) -> None:

    # ------------------------------------------------------------------ thermal states
    @mcp.tool(structured_output=False)
    def fea_thermal_case_create(name: str, transient: bool = False, time_s: float = 3600.0, dt_s: float = 0.0,
                                outputs: int = 10) -> str:
        """Creates/changes a thermal state (CalculiX heat conduction). transient=False: steady (needs a heat sink: prescribed temperature, convection or radiation). transient=True: from the initial temperature to time_s with fixed step dt_s (0 = time_s/100); outputs = stored snapshots (last = time_s). Materials need k [W/(m*K)]; transient also rho and cp [J/(kg*K)]. Conditions: fea_thermal_add_*."""
        t = _num(time_s, "time_s")
        dt = _num(dt_s, "dt_s")
        if transient:
            if t <= 0:
                raise FeaError("time_s must be a positive duration of the transient state [s]")
            if not 0 <= dt <= t:
                raise FeaError("dt_s must be 0 (automatic time_s/100) up to time_s")
            if int(outputs) < 1:
                raise FeaError("outputs = the number of frames, at least 1")
            if dt > 0 and t / dt > 20000:
                raise FeaError(f"{t / dt:.0f} time steps are too many (ccx ~0.1 s/step even for a small model) – increase dt_s")
        with sess.use():
            st = sess.require_study()
            case, created = get_thermal_case(st, name)
            case.transient = bool(transient)
            if transient:
                case.time, case.dt, case.outputs = t, dt, int(outputs)
            sess.changed()
            d = {"thermal_case": describe_case(st, case), "created": created,
                 "thermal_cases": [c.name for c in st.thermal.cases]}
            missing = [m.name for m in _materials_used(st, sess.part_names())
                       if not m.k or (transient and (not m.cp or not m.rho))]
            if missing:
                d["warning"] = ("Materials without thermal conductivity k" + (" / c_p / hustoty" if transient else "")
                                + f": {', '.join(sorted(set(missing)))} – add fea_materials_assign custom_props")
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_thermal_case_delete(name: str) -> str:
        """Deletes a thermal state with its conditions (refused while a mechanical state takes its temperature from it)."""
        with sess.use():
            st = sess.require_study()
            case, _ = get_thermal_case(st, name, create=False)
            used = [lc.name for lc in st.load_cases for l in lc.loads if l.kind == "thermal" and l.case == name]
            if used:
                raise FeaError(f"Thermal state '{name}' is used by the temperature load of state {', '.join(used)} – remove that first (fea_load_remove(case_name, name='temperature'))")
            st.thermal.cases.remove(case)
            sess.changed()
            return out({"deleted": name, "thermal_cases": [c.name for c in st.thermal.cases]})

    @mcp.tool(structured_output=False)
    def fea_thermal_settings(reference_C: float | None = None, initial_C: float | str | None = None) -> str:
        """reference_C: reference temperature [degC] where thermal strain is zero (alpha*(T - T_ref)); initial_C: initial temperature of transient states [degC] or 'reference'. No arguments: returns settings and thermal states."""
        with sess.use():
            st = sess.require_study()
            th = st.thermal
            changed = False
            if reference_C is not None:
                th.reference_temperature = _temp(reference_C, "reference_C")
                changed = True
            if initial_C is not None:
                if isinstance(initial_C, str) and initial_C.strip().lower() in ("reference", "ref", "reference",
                                                                                 "none", "", "referenční"):
                    th.initial_temperature = None
                else:
                    th.initial_temperature = _temp(initial_C, "initial_C")
                changed = True
            if changed:
                sess.changed()
            return out({**thermal_overview(st), "changed": changed, "units": UNITS})

    @mcp.tool(structured_output=False)
    def fea_thermal_list() -> str:
        """Heat setup: reference/initial temperature, thermal states with conditions (values with units, transient steps) and thermal loads of mechanical states."""
        with sess.use():
            return out({**thermal_overview(sess.require_study()), "units": UNITS})

    # ------------------------------------------------------------------ boundary conditions
    @mcp.tool(structured_output=False)
    def fea_thermal_add_temperature(case: str, selection: str | list | dict, value_C: float,
                                    name: str | None = None) -> str:
        """Prescribed temperature [degC] on faces in thermal state `case` (created as steady if missing). selection: face ids 'Díl#i', query {"x": 0} or selection name (a constraint's selection can be shared). name: new selection name (default auto). Heat through the face is reported under the selection name (+ into the body)."""
        v = _temp(value_C, "value_C")
        with sess.use():
            return out(add_thermal_bc(sess, case, "fixed_temperature", selection, name, value=v))

    @mcp.tool(structured_output=False)
    def fea_thermal_add_heat_flux(case: str, selection: str | list | dict, flux_W_m2: float | None = None,
                                  power_W: float | None = None, name: str | None = None) -> str:
        """Heat into the body through faces: flux_W_m2 [W/m2] OR power_W [W] (spread uniformly over the area). Positive = into the body, negative = out. Note 1 W/m2 = 1e-6 W/mm2. The same selection in a state replaces."""
        if (flux_W_m2 is None) == (power_W is None):
            raise FeaError("Give exactly one: flux_W_m2 [W/m²] or power_W [W]")
        with sess.use():
            if flux_W_m2 is not None:
                return out(add_thermal_bc(sess, case, "heat_flux", selection, name, value=_num(flux_W_m2, "flux_W_m2")))
            return out(add_thermal_bc(sess, case, "heat_flow", selection, name, value=_num(power_W, "power_W")))

    @mcp.tool(structured_output=False)
    def fea_thermal_add_convection(case: str, selection: str | list | dict, h_W_m2K: float, ambient_C: float = 20.0,
                                   name: str | None = None) -> str:
        """Convection q = h*(T_ambient - T). h_W_m2K [W/(m2*K)]: free air 5-25, forced 25-250, water 500-10000; ambient_C [degC]. Can be combined with radiation on the same face."""
        h = _num(h_W_m2K, "h_W_m2K")
        if h <= 0:
            raise FeaError("h_W_m2K must be a positive heat transfer coefficient [W/(m²·K)]")
        amb = _temp(ambient_C, "ambient_C")
        with sess.use():
            return out(add_thermal_bc(sess, case, "convection", selection, name, h=h, ambient=amb))

    @mcp.tool(structured_output=False)
    def fea_thermal_add_radiation(case: str, selection: str | list | dict, emissivity: float, ambient_C: float = 20.0,
                                  name: str | None = None) -> str:
        """Radiation to surroundings at ambient_C [degC] (Stefan-Boltzmann, nonlinear, no mutual irradiation). emissivity 0-1: polished Al 0.05, oxidized steel 0.8, black paint 0.95."""
        e = _num(emissivity, "emissivity")
        if not 0 < e <= 1:
            raise FeaError("emissivity must be in the interval (0; 1]")
        amb = _temp(ambient_C, "ambient_C")
        with sess.use():
            return out(add_thermal_bc(sess, case, "radiation", selection, name, emissivity=e, ambient=amb))

    @mcp.tool(structured_output=False)
    def fea_thermal_add_heat_generation(case: str, part: str, power_W: float) -> str:
        """Internal heat source power_W [W] spread uniformly in a part's volume. The same part in a state replaces."""
        p = _num(power_W, "power_W")
        with sess.use():
            return out(add_thermal_bc(sess, case, "heat_generation", part=part, value=p))

    @mcp.tool(structured_output=False)
    def fea_thermal_add_cavity(case: str, selection: str | list | dict, emissivity: float, cavity: str = "cavity",
                               closed: bool = False, ambient_C: float = 20.0, name: str | None = None) -> str:
        """Cavity radiation (surface-to-surface, view factors by CalculiX) of faces in thermal state `case`; call once per face group with its own emissivity, faces with the same `cavity` name radiate to each other. closed=True: fully enclosed cavity (view factors normalized to 1, no exchange with ambient – the state still needs a heat sink); closed=False: open cavity, the opening sees ambient_C [degC]. emissivity 0-1. Nonlinear (ccx iterations). Heat per face in the results (balance.cavity_faces)."""
        e = _num(emissivity, "emissivity")
        if not 0 < e <= 1:
            raise FeaError("emissivity must be in the interval (0; 1]")
        amb = _temp(ambient_C, "ambient_C")
        cav = str(cavity or "").strip()
        if not cav:
            raise FeaError("cavity: a cavity name is required (faces with the same name radiate to each other)")
        with sess.use():
            d = add_thermal_bc(sess, case, "cavity_radiation", selection, name, emissivity=e, ambient=amb,
                               cavity_name=cav, closed=bool(closed))
            c, _ = get_thermal_case(sess.require_study(), d["thermal_case"], create=False)
            same = [b for b in c.loads if b.kind == "cavity_radiation" and b.cavity_name == cav]
            changed = [b.selection for b in same if (b.closed, b.ambient) != (bool(closed), amb)]
            for b in same:                                # one cavity = one setting of closed / ambient
                b.closed, b.ambient = bool(closed), amb
            d["cavity"] = {"name": cav, "closed": bool(closed), "faces": [b.selection for b in same],
                           "emissivities": [b.emissivity for b in same]}
            if changed:
                d["note"] = f"closed / ambient_C also updated on the other faces of cavity '{cav}': {', '.join(map(str, changed))}"
            if len(same) < 2 and closed:
                d["warning"] = "A closed cavity needs at least two faces (or a face that sees itself) – add the other faces with the same cavity name"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_thermal_add_contact_conductance(side_a: str, side_b: str | None = None, h_W_m2K: float | None = None,
                                            pressure_MPa: float = 1.0, roughness_um: float = 3.0,
                                            tim: str | None = None, name: str | None = None) -> str:
        """Thermal contact conductance h_c [W/(m2*K)] of an interface (temperature jump dT = q/h_c); applies to ALL thermal states. side_a, side_b: two part names (bonded interface – one part is then meshed separately and tied mechanically by *TIE) or two face selections; side_a = a contact pair name with side_b omitted sets h_c of that pair. h_W_m2K omitted = indicative estimate from the materials, pressure_MPa, roughness_um and tim ('paste' | 'pad'). Typical: steel dry 1000-3000, bolted 5000, paste 25000. Remove: fea_thermal_remove(case='', name, kind='contact_conductance')."""
        from ..thermal.interfaces import ConductanceError, set_interface, suggest
        with sess.use():
            st = sess.require_study()
            parts = sess.part_names()
            pair = next((c for c in st.contacts if c.name == side_a), None)
            if pair is not None and not side_b:
                sug = suggest(st, pair.a, pair.b, parts, _num(pressure_MPa, "pressure_MPa"),
                              _num(roughness_um, "roughness_um"), tim)
                h = _num(h_W_m2K, "h_W_m2K") if h_W_m2K is not None else sug["h_W_m2K"]
                if h <= 0:
                    raise FeaError("h_W_m2K must be a positive contact conductance [W/(m²·K)]")
                pair.conductance = h
                sess.changed()
                d = {"contact_pair": pair.name, "h_W_m2K": h, "applies_to": "all thermal states"}
                if h_W_m2K is None:
                    d["suggestion"] = sug
                if pair.kind in ("bonded", "none"):
                    d["warning"] = f"Contact pair '{pair.name}' is of type '{pair.kind}' – it creates no contact faces, h_c has no effect; use the two part names instead"
                return out(d)
            if not side_b:
                raise FeaError(f"side_b is required ('{side_a}' is not a contact pair; pairs: {', '.join(c.name for c in st.contacts) or 'none'})")
            sug = suggest(st, side_a, side_b, parts, _num(pressure_MPa, "pressure_MPa"), _num(roughness_um, "roughness_um"), tim)
            h = _num(h_W_m2K, "h_W_m2K") if h_W_m2K is not None else sug["h_W_m2K"]
            try:
                item, sep, replaced = set_interface(st, side_a, side_b, h, name, parts)
            except ConductanceError as e:
                raise FeaError(str(e)) from None
            sess.changed()
            d = {"conductance": {"name": item.name, "side_a": item.selection, "side_b": item.master,
                                 "h_W_m2K": item.conductance, "R_m2K_W": item.resistance},
                 "replaced": replaced, "applies_to": "all thermal states"}
            if h_W_m2K is None:
                d["suggestion"] = sug
            if sep:
                d["separate_mesh_part"] = sep
                d["note"] = (f"Part '{sep}' is now meshed separately (the interface needs two sets of nodes); mechanically it stays tied by *TIE "
                             "(non-conforming mesh, ccx backend). The mesh will be regenerated.")
            if not st.thermal.cases:
                d["warning"] = "No thermal state yet – create one (fea_thermal_case_create) with a heat sink"
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_thermal_remove(case: str, name: str, kind: str | None = None) -> str:
        """Removes a thermal condition by selection name (part name for a source); kind fixed_temperature | heat_flux | convection | radiation | cavity_radiation | heat_generation restricts it (default: all with that name). kind='contact_conductance' removes an interface conductance by its name or a contact pair's h_c (case ignored). The face selection stays."""
        with sess.use():
            st = sess.require_study()
            if kind == "contact_conductance":
                th = st.thermal
                n0 = len(th.conductances)
                th.conductances = [c for c in th.conductances if c.name != name]
                pairs = [c for c in st.contacts if c.name == name and c.conductance]
                for c in pairs:
                    c.conductance = None
                if len(th.conductances) == n0 and not pairs:
                    raise FeaError(f"Contact conductance '{name}' does not exist (defined: {', '.join(r['name'] for r in conductance_overview(st)) or 'none'})")
                sess.changed()
                return out({"removed": name, "contact_conductances": conductance_overview(st),
                            "note": "parts made separate for the interface stay separately meshed (fea_mesh_separate_parts)"})
            c, _ = get_thermal_case(st, case, create=False)
            grp = _group(kind) if kind else None
            keep = [b for b in c.loads if not (bc_key(b)[1] == name and (grp is None or bc_key(b)[0] == grp))]
            if len(keep) == len(c.loads):
                raise FeaError(f"Boundary condition '{name}'" + (f" ({kind})" if kind else "")
                               + f" does not exist in the thermal state '{case}'")
            c.loads = keep
            sess.changed()
            return out({"removed": name, "thermal_case": case, "bcs": [describe_bc(b) for b in c.loads]})

    # ------------------------------------------------------------------ thermal load of the statics
    @mcp.tool(structured_output=False)
    def fea_load_add_temperature(case_name: str, thermal_case: str | None = None, time_s: float | None = None,
                                 value_C: float | None = None, parts: list[str] | None = None) -> str:
        """Thermal load of mechanical state case_name (strain alpha*(T - T_ref)): thermal_case = temperature field of that thermal state (time_s = transient snapshot, None = end) OR value_C = uniform temperature [degC] of `parts` (None = all; overrides the field). T_ref from fea_thermal_settings (20 degC). Creates the state if missing. Fully clamped parts have restrained expansion (large stresses). Materials need alpha. Remove with fea_load_remove(case_name, name='teplota')."""
        from ..study import ThermalLoad
        from .tools_bc import get_case
        from .tools_study import describe_load
        if (thermal_case is None) == (value_C is None):
            raise FeaError("Give exactly one: thermal_case (a field from a thermal state) or value_C (a uniform temperature)")
        if thermal_case is not None and parts:
            raise FeaError("parts applies only to a uniform temperature value_C – a field from a thermal state applies to the whole model")
        if value_C is not None and time_s is not None:
            raise FeaError("time_s applies only with thermal_case (a frame of a transient state)")
        with sess.use():
            st = sess.require_study()
            names = sess.part_names()
            for p in parts or []:
                if p not in names:
                    raise FeaError(f"Part '{p}' does not exist. Parts: {', '.join(names)}")
            if thermal_case is not None:
                tc, _ = get_thermal_case(st, thermal_case, create=False)
                t = None
                if time_s is not None:
                    if not tc.transient:
                        raise FeaError(f"Thermal state '{thermal_case}' is steady – time_s makes sense only for a transient one")
                    t = _num(time_s, "time_s")
                    if not 0 <= t <= tc.time:
                        raise FeaError(f"time_s = {t:g} s is outside the transient state '{thermal_case}' (0–{tc.time:g} s)")
                obj = ThermalLoad(case=thermal_case, time=t)
            else:
                obj = ThermalLoad(value=_temp(value_C, "value_C"), parts=list(parts or []))
            lc, created = get_case(st, case_name)
            if obj.case:
                old = [l for l in lc.loads if l.kind == "thermal" and l.case]
            else:
                old = [l for l in lc.loads if l.kind == "thermal" and not l.case and set(l.parts) == set(obj.parts)]
            lc.loads = [l for l in lc.loads if not any(l is o for o in old)] + [obj]
            sess.changed()
            d = {"load_case": lc.name, "load": describe_load(obj), "replaced": bool(old), "case_created": created,
                 "reference_temperature_C": st.thermal.reference_temperature}
            warn = []
            no_alpha = sorted({m.name for m in _materials_used(st, names) if m.alpha is None})
            if no_alpha:
                warn.append(f"Materials without the expansion coefficient α: {', '.join(no_alpha)} – add fea_materials_assign custom_props {{'alpha': 12e-6}}")
            tl = [l for l in lc.loads if l.kind == "thermal"]
            if any(l.case for l in tl) and any(not l.case and not l.parts for l in tl):
                warn.append("A uniform temperature of all parts overrides the temperature field from a thermal state in the whole model")
            if not st.constraints:
                warn.append("The study has no supports – thermal stress needs a support (fea_constraint_add_*)")
            if warn:
                d["warnings"] = warn
            return out(d)


def _materials_used(st, part_names: list[str]) -> list:
    """Materials of the meshed parts (without bolts and left-out parts)."""
    excl = st.excluded_parts()
    out_ = []
    for p in part_names:
        if p in excl:
            continue
        try:
            out_.append(st.material_of(p))
        except KeyError:
            pass
    return out_
