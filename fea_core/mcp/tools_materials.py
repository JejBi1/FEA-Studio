"""Tools: catalogue of materials (built-in + user, designation converter) and assignment to parts."""
from __future__ import annotations

import math
from dataclasses import asdict, fields

import numpy as np

from .common import out
from .session import FeaError, Session

PROP_KEYS = {f.name for f in fields(__import__("fea_core.study", fromlist=["Material"]).Material)} - {"name"}


def material_row(m) -> dict:
    d = {"name": m.name, "E_MPa": m.E, "nu": m.nu, "rho_t_mm3": m.rho, "Re_MPa": m.Re, "Rm_MPa": m.Rm,
         "source": m.source}
    if m.is_print:
        d.update({"printed": True, "E_z_MPa": m.E_z, "Rm_z_MPa": m.Rm_z})
    d.update({"k_W_mK": m.k, "alpha_1_K": m.alpha, "cp_J_kgK": m.cp})
    if getattr(m, "ortho", None):
        d.update({"orthotropic": dict(m.ortho), "ortho_csys": m.ortho_csys})
    return {k: v for k, v in d.items() if v is not None}


ORTHO_KEYS = ("E1", "E2", "E3", "nu12", "nu13", "nu23", "G12", "G13", "G23")
ORTHO_OPT = ("alpha1", "alpha2", "alpha3")


def check_ortho(st, ortho: dict | None, csys: str | None) -> dict | None:
    """Verified constants of an orthotropic material (keys, positive moduli, positive definite compliance, system)."""
    if not ortho:
        if csys:
            raise FeaError("ortho_csys only makes sense for an orthotropic material (ortho)")
        return None
    if not isinstance(ortho, dict):
        raise FeaError("ortho must be an object {E1, E2, E3, nu12, nu13, nu23, G12, G13, G23 [, alpha1–3]}")
    bad = set(ortho) - set(ORTHO_KEYS) - set(ORTHO_OPT)
    if bad:
        raise FeaError(f"Unknown orthotropic material constants: {', '.join(sorted(bad))} (allowed: {', '.join(ORTHO_KEYS + ORTHO_OPT)})")
    miss = [k for k in ORTHO_KEYS if ortho.get(k) is None]
    if miss:
        raise FeaError(f"The orthotropic material is missing {', '.join(miss)}")
    o = {}
    for k, v in ortho.items():
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise FeaError(f"{k} must be a number")
        o[k] = float(v)
    if min(o[k] for k in ("E1", "E2", "E3", "G12", "G13", "G23")) <= 0:
        raise FeaError("The moduli E1–E3 and G12–G23 must be positive [MPa]")
    if any(abs(o.get(k, 0.0)) > 1e-3 for k in ORTHO_OPT):
        raise FeaError("give alpha1–3 in 1/K (e.g. 12e-6), not in 10⁻⁶/K")
    from ..ortho import compliance
    if np.linalg.eigvalsh(compliance(o)).min() <= 0:
        raise FeaError("The constants are not physically admissible (the compliance matrix is not positive definite) – check ν_ij (ε_j = −ν_ij·σ_i/E_i, ν_ij² < E_i/E_j must hold)")
    if csys is not None and csys not in st.coordinate_systems:
        raise FeaError(f"Coordinate system '{csys}' does not exist (fea_csys_create; systems: {', '.join(st.coordinate_systems) or 'none'})")
    return o


def assign(sess: Session, st, part: str | None, m) -> list[str]:
    """Saves a material into the study and assigns it to a part ('*' = default, None = only save); returns the parts without a material."""
    st.materials[m.name] = m
    if part == "*":
        st.default_material = m.name
    elif part:
        st.part_materials[part] = m.name
        if st.default_material is None:
            st.default_material = m.name
    sess.changed()
    missing = []
    for p in sess.part_names():
        try:
            st.material_of(p)
        except KeyError:
            missing.append(p)
    return missing


ALIASES = {"E_MPa": "E", "Re_MPa": "Re", "Rm_MPa": "Rm", "rho_t_mm3": "rho", "density": "rho",
           "k_W_mK": "k", "lambda": "k", "conductivity": "k", "alpha_1_K": "alpha", "cte": "alpha",
           "cp_J_kgK": "cp", "c_p": "cp", "specific_heat": "cp"}


def check_thermal_props(m) -> None:
    """Thermal properties in the units of FEA Studio (catches typical mix-ups of units)."""
    if m.k is not None and not m.k > 0:
        raise FeaError("The thermal conductivity k must be positive [W/(m·K)] (steel ~50, aluminium ~170, PLA ~0.13)")
    if m.alpha is not None and abs(m.alpha) > 1e-3:
        raise FeaError(f"The expansion coefficient α = {m.alpha:g} is probably in 10⁻⁶/K – give it in 1/K (steel 12e-6)")
    if m.cp is not None and not m.cp > 0:
        raise FeaError("The specific heat c_p must be positive [J/(kg·K)]")
    if m.cp is not None and m.cp < 50:
        raise FeaError(f"c_p = {m.cp:g} is probably in kJ/(kg·K) – give it in J/(kg·K) (steel 460)")


def catalog(query: str | None = None) -> dict:
    from ..materials import all_materials
    mats = all_materials()
    d: dict = {}
    if query:
        q = query.lower()
        hit = [m for m in mats if q in m.name.lower() or q in m.source.lower()]
        try:                                   # designation converter (ČSN, W.Nr., AISI…) as in the GUI
            from .. import grades
            hits = grades.lookup(query)
            if hits:
                g, approx = hits[0]
                d["grade"] = {"group": g.key, "approximate": approx,
                              "equivalents": {k: v[:4] for k, v in g.names.items()}}
                q_prof = grades.profile(g)
                similar = []
                for m in mats:
                    mg = grades.grade_of(m.name)
                    if mg is g and m not in hit:
                        hit.append(m)
                    elif mg is not None and mg is not g:
                        dist = grades.distance(q_prof, grades.profile(mg, rm=m.Rm))
                        if dist <= 6:
                            similar.append((dist, m.name, grades.difference(q_prof, grades.profile(mg, rm=m.Rm))))
                if similar:
                    d["similar_in_library"] = [{"name": n, "difference": why} for _, n, why in sorted(similar)[:4]]
        except Exception:  # noqa: BLE001 – the converter is only a help
            pass
        mats = hit
    d["materials"] = [material_row(m) for m in mats]
    d["units"] = ("E, Re, Rm v MPa; hustota v t/mm³ (ocel 7.85e-9); k W/(m·K), alpha 1/K, cp J/(kg·K)")
    return d


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_materials_list(query: str | None = None) -> str:
        """Material catalogue (built-in with sources + user library). query filters by name ('S355', 'hliník', '1.4301', '11 523'; the grade converter is used). Thermal: k_W_mK, alpha_1_K, cp_J_kgK."""
        return out(catalog(query))

    @mcp.tool(structured_output=False)
    def fea_materials_assign(part: str, material_name: str, custom_props: dict | None = None) -> str:
        """Assigns a material to a part ('*' = default for parts without their own). material_name from the catalogue, or a new name with custom_props {E [MPa], nu required; rho [t/mm3], Re, Rm [MPa], source, k [W/(m*K)], alpha [1/K], cp [J/(kg*K)]}; custom_props on a catalogue material override it (copy with the same name)."""
        from ..materials import by_name
        from ..study import Material
        with sess.use():
            st = sess.require_study()
            if part != "*" and part not in sess.part_names():
                raise FeaError(f"Part '{part}' does not exist. Parts: {', '.join(sess.part_names())}")
            props = dict(custom_props or {})
            props = {ALIASES.get(k, k): v for k, v in props.items()}
            bad = set(props) - PROP_KEYS
            if bad:
                raise FeaError(f"Unknown material properties: {', '.join(sorted(bad))} (allowed: {', '.join(sorted(PROP_KEYS))})")
            try:
                m = by_name(material_name)
                m = Material(**{**asdict(m), **props})
            except KeyError:
                if material_name in st.materials:
                    m = Material(**{**asdict(st.materials[material_name]), **props})
                elif "E" in props and "nu" in props:
                    m = Material(name=material_name, **props)
                else:
                    raise FeaError(f"Material '{material_name}' is not in the catalogue – give custom_props with E and nu or choose from fea_materials_list") from None
            if not (m.E > 0 and -1 < m.nu < 0.5):
                raise FeaError("Invalid E or nu (E > 0, −1 < ν < 0.5)")
            if m.rho and m.rho > 1e-6:
                raise FeaError(f"The density {m.rho} is probably in kg/m³ – give it in t/mm³ (steel 7.85e-9)")
            check_thermal_props(m)
            if m.ortho or m.ortho_csys:
                m.ortho = check_ortho(st, m.ortho, m.ortho_csys)
            missing = assign(sess, st, part, m)
            return out({"assigned": {"part": part, "material": material_row(m)}, "parts_without_material": missing})

    @mcp.tool(structured_output=False)
    def fea_material_set_orthotropic(name: str, constants: dict, csys: str | None = None,
                                     part: str | None = None, rho: float | None = None, Re: float | None = None,
                                     Rm: float | None = None, base_material: str | None = None,
                                     source: str = "") -> str:
        """General orthotropic material (composites, wood; not FDM - see fea_print_*). constants {E1,E2,E3,G12,G13,G23 [MPa], nu12,nu13,nu23, optional alpha1..3 [1/K]} in axes 1,2,3. csys: coordinate system name (Cartesian: 1,2,3 = x,y,z; cylindrical: 1 radial, 2 tangential, 3 axial); None = global. part ('*' = default; None = only store, assign via fea_materials_assign). base_material supplies rho, k, cp. Positive-definiteness checked. Von Mises safety is not valid - assess stresses in material axes. ccx only."""
        from ..materials import by_name
        from ..study import Material
        with sess.use():
            st = sess.require_study()
            if part is not None and part != "*" and part not in sess.part_names():
                raise FeaError(f"Part '{part}' does not exist. Parts: {', '.join(sess.part_names())}")
            o = check_ortho(st, constants, csys)
            if o is None:
                raise FeaError("constants must not be empty (E1…G23)")
            base = {}
            if base_material:
                if base_material in st.materials:
                    base = asdict(st.materials[base_material])
                else:
                    try:
                        base = asdict(by_name(base_material))
                    except KeyError:
                        raise FeaError(f"Material '{base_material}' is neither in the study nor in the catalogue") from None
                for k in ("E_z", "Rm_z", "nu_z", "G_z", "Rc", "Rc_z", "S_xy", "S_z"):
                    base.pop(k, None)                         # printed material → general orthotropic
            nu12 = o["nu12"]
            given = {k: float(v) for k, v in (("rho", rho), ("Re", Re), ("Rm", Rm)) if v is not None}
            m = Material(**{**base, "name": name, "E": o["E1"], "nu": nu12 if -1 < nu12 < 0.5 else 0.3,
                            "ortho": o, "ortho_csys": csys, **given,
                            "source": source or base.get("source", "") or "orthotropic material given by the user"})
            if m.rho and m.rho > 1e-6:
                raise FeaError(f"The density {m.rho} is probably in kg/m³ – give it in t/mm³ (steel 7.85e-9)")
            check_thermal_props(m)
            missing = assign(sess, st, part, m)
            d = {"material": material_row(m), "assigned_to": part, "parts_without_material": missing,
                 "warning": "The safety against Re (von Mises) does not apply to an orthotropic material – evaluate the stress components in the material axes (the stresses in the results are in global axes)"}
            if csys:
                from ..csys import describe
                d["axes"] = describe(st.coordinate_systems[csys])
            if o.get("alpha1") is None and (st.thermal.cases or st.has_thermal_loads()):
                d["note"] = "Without alpha1–3 the isotropic alpha of the material is used for thermal loading"
            return out(d)
