"""Tools: library of sections, analytical calculations of beams and curved bars, assessment of a bar according to EN 1993-1-1.

Wrappers over fea_core.beams (sections, analytic, ec3) – a quick calculator without a mesh (check of the FE, preliminary design).
They do not require an open study. Units mm, N, MPa, N·mm; section quantities mm², mm⁴, mm⁶, mm³.
"""
from __future__ import annotations

import math
from dataclasses import fields

import numpy as np

from .common import out
from .session import FeaError, Session

SECTION_KEYS = (("A", "A_mm2"), ("Iy", "Iy_mm4"), ("Iz", "Iz_mm4"), ("J", "It_mm4"), ("Iw", "Iw_mm6"),
                ("Wel_y", "Wel_y_mm3"), ("Wel_z", "Wel_z_mm3"), ("Wpl_y", "Wpl_y_mm3"), ("Wpl_z", "Wpl_z_mm3"),
                ("Av_y", "Av_y_mm2"), ("Av_z", "Av_z_mm2"), ("Wt", "Wt_mm3"), ("ys", "shear_center_y_mm"),
                ("zs", "shear_center_z_mm"))


def section_of(spec):
    """Section from the name of the library ("IPE 200") or parametrically {"shape": "RECT", "b": 20, "h": 40} → SectionProps."""
    from ..beams import sections
    try:
        if isinstance(spec, str):
            return sections.get(spec)
        if isinstance(spec, dict):
            d = dict(spec)
            if "name" in d and "shape" not in d:
                return sections.get(d["name"])
            shape = d.pop("shape", None)
            if not shape:
                raise FeaError('Parametric section: {"shape": RECT | CIRC | TUBE | RHS | I | T | U | L, dimensions mm}')
            return sections.parametric(str(shape), **{k: (float(v) if isinstance(v, (int, float)) else v)
                                                       for k, v in d.items()})
    except FeaError:
        raise
    except KeyError as e:
        raise FeaError(f"The parametric section is missing the dimension {e}") from None
    except (ValueError, TypeError) as e:          # SectionError is a ValueError
        raise FeaError(str(e)) from None
    raise FeaError("section = a profile name (“IPE 200”, “HE 200 B”, “L50x5”, “RHS 100x50x4”) or {\"shape\": …, dimensions}")


def section_dict(sec, rho: float = 7.85e-9) -> dict:
    d = {"name": sec.name, "shape": sec.shape, "dims_mm": sec.dims}
    for k, key in SECTION_KEYS:
        d[key] = getattr(sec, k)
    d.update({"iy_mm": sec.iy, "iz_mm": sec.iz, "alpha_principal_deg": math.degrees(sec.alpha),
              "mass_kg_m": sec.mass_per_m(rho), "source": sec.source,
              "axes": "y = strong principal axis, z = weak (EN 1993), origin at the centroid"})
    return d


def _load(ld: dict):
    from ..beams import analytic as an
    if not isinstance(ld, dict):
        raise FeaError("Load: an object {type: force|moment|distributed, …}")
    t = str(ld.get("type", "")).lower()
    try:
        if t in ("force", "force", "force", "point", "sila", "síla"):
            return an.Force(float(ld["P_N"]), float(ld["x_mm"]))
        if t in ("moment",):
            return an.Moment(float(ld["C_Nmm"]), float(ld["x_mm"]))
        if t in ("distributed", "uniform", "spojite", "distributed", "q", "spojité"):
            q1 = float(ld["q_N_mm"])
            b = ld.get("to_mm")
            q2 = ld.get("q2_N_mm")
            return an.Distributed(q1, float(ld.get("from_mm", 0.0)), None if b is None else float(b),
                                  q1 if q2 is None else float(q2))
    except KeyError as e:
        raise FeaError(f"Load {t}: {e} is missing") from None
    raise FeaError("Load type: force {P_N, x_mm} | moment {C_Nmm, x_mm} | distributed {q_N_mm, from_mm, to_mm, q2_N_mm}")


def _beam_out(r, n_points: int) -> dict:
    ex = {}
    names = {"V": "V_N", "M": "M_Nmm", "phi": "phi_rad", "w": "w_mm"}
    for q, key in names.items():
        e = r.extremes[q]
        ex[key] = {k: {"value": e[k][0], "x_mm": e[k][1]} for k in ("max", "min", "abs")}
    L = float(np.max(r.x))
    xs = np.linspace(0.0, L, max(int(n_points), 2))
    at = r.at(xs)
    table = [{"x_mm": float(x), "V_N": float(at["V"][i]), "M_Nmm": float(at["M"][i]), "w_mm": float(at["w"][i]),
              "phi_rad": float(at["phi"][i])} for i, x in enumerate(xs)]
    return {"reactions": r.reactions, "extremes": ex, "table": table, "warnings": r.warnings,
            "sign_convention": "load and deflection w positive DOWN, reaction R positive UP [N], M > 0 = tension at the bottom [N·mm], moments at supports M_A/M_B/M_i as M(x); x from the left end"}


def _stiff(section, I_mm4, E_MPa):
    if section is None and I_mm4 is None:
        raise FeaError("Give section (profile / parametric) or I_mm4")
    sec = section_of(section) if section is not None else None
    if not (E_MPa and E_MPa > 0):
        raise FeaError("E_MPa must be positive")
    return sec


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_section_get(section: str | dict, rho_t_mm3: float = 7.85e-9) -> str:
        """Bar section properties. section: library name (IPE, HE A/B/M, UPE, UPN, equal/unequal L, CHS "TR 60,3x4", SHS/RHS cold+hot "JEKL 100x100x5"; EN 10365/10056/10219/10210, properties computed from dimensions) or parametric {"shape": RECT (b,h) | CIRC (d) | TUBE (D,t) | RHS (h,b,t,ro,ri,fab) | I (h,b,tw,tf,r | b2,tf2) | T (h,b,tw,tf,r) | U (h,b,t | tw,tf,r) | L (h,b,t,r1,r2)} [mm]. Returns A, Iy, Iz (y strong), It, Iw, Wel, Wpl, shear areas Av (EN 1993-1-1 6.2.6), Wt, radii of gyration, shear centre, mass [kg/m] at rho_t_mm3."""
        sec = section_of(section)
        return out(section_dict(sec, rho_t_mm3))

    @mcp.tool(structured_output=False)
    def fea_section_search(text: str = "", limit: int = 50) -> str:
        """Searches the section library ('IPE 2', 'HEB', 'L50', 'CHS 60', 'RHS 100x50'); empty text = series overview. Names go to fea_section_get / fea_beam_analytic / fea_ec3_member_check."""
        from ..beams import sections
        if not text.strip():
            fam = sections.families()
            return out({"families": {k: {"count": len(v), "first": v[0] if v else None, "last": v[-1] if v else None}
                                     for k, v in fam.items()}})
        names = sections.search(text, limit=max(1, min(int(limit), 500)))
        return out({"query": text, "sections": names, "count": len(names)})

    @mcp.tool(structured_output=False)
    def fea_beam_analytic(loads: list[dict], length_mm: float | None = None, spans_mm: list[float] | None = None,
                          supports: str | list[str] = "simple", section: str | dict | None = None,
                          I_mm4: float | None = None, E_MPa: float = 210000.0, axis: str = "y",
                          inner_supports_mm: list[float] | None = None, overhang_mm: list[float] | None = None,
                          settlement_mm: list[float] | None = None, shear: bool = False, nu: float = 0.3,
                          points: int = 11) -> str:
        """Closed-form beam (no mesh): reactions and courses/extremes of V, M, rotation, deflection w. Single span: length_mm + supports "simple" | "cantilever" (clamped left) | "fixed" | "propped" (clamped left, hinged right) | [left, right] of free|pinned|fixed|guided; inner_supports_mm (x); shear = Timoshenko. Continuous: spans_mm (three-moment), supports [left,right] pinned|fixed, overhang_mm [left,right], settlement_mm (positive down). loads (x from the left end incl. overhang): {"type":"force","P_N","x_mm"} | {"type":"moment","C_Nmm","x_mm"} (right-handed +) | {"type":"distributed","q_N_mm","from_mm","to_mm","q2_N_mm"} (linear q->q2). Loads and deflection positive DOWN, reactions UP. Stiffness: section (+ axis y|z) or I_mm4, E_MPa. points = table size."""
        from ..beams import analytic as an
        if (length_mm is None) == (spans_mm is None):
            raise FeaError("Give length_mm (single-span beam) OR spans_mm (continuous beam)")
        if axis not in ("y", "z"):
            raise FeaError("axis must be 'y' (strong axis) or 'z'")
        if not 2 <= int(points) <= 401:
            raise FeaError("points must be 2–401")
        sec = _stiff(section, I_mm4, E_MPa)
        lds = [_load(x) for x in loads or []]
        try:
            if spans_mm is not None and len(spans_mm) > 1 or (spans_mm is not None and overhang_mm):
                sup = supports if isinstance(supports, (list, tuple)) else ("pinned", "pinned") \
                    if supports in ("simple", "pinned") else None
                if sup is None or len(sup) != 2:
                    raise FeaError("Continuous beam: supports = [left, right] from 'pinned' | 'fixed'")
                r = an.continuous_beam([float(s) for s in spans_mm], lds, E=float(E_MPa),
                                       I=None if I_mm4 is None else float(I_mm4), section=sec, axis=axis,
                                       left=sup[0], right=sup[1], overhang=tuple(overhang_mm or (0.0, 0.0)),
                                       settlement=settlement_mm)
                kind = "continuous beam (three-moment equation, Euler–Bernoulli)"
            else:
                L = float(length_mm if length_mm is not None else spans_mm[0])
                if not L > 0:
                    raise FeaError("length_mm must be positive")
                sup = supports if isinstance(supports, str) else tuple(supports)
                r = an.beam(L, lds, sup, E=float(E_MPa), I=None if I_mm4 is None else float(I_mm4), section=sec,
                            axis=axis, inner=[float(x) for x in inner_supports_mm or []], settlement=settlement_mm,
                            shear=bool(shear), nu=float(nu))
                kind = "beam (Macaulay, " + ("Timoshenko" if shear else "Euler–Bernoulli") + ")"
        except FeaError:
            raise
        except (ValueError, KeyError) as e:
            raise FeaError(str(e)) from None
        d = {"kind": kind, **_beam_out(r, points)}
        d["EI_Nmm2"] = r.info.get("EI")
        if sec is not None:
            d["section"] = sec.name
            Wy = sec.Wel_y if axis == "y" else sec.Wel_z
            if Wy:
                Mabs = abs(r.extremes["M"]["abs"][0])
                d["max_bending_stress_MPa"] = Mabs / Wy
                d["stress_note"] = "σ = |M|max / Wel (elastic, extreme fibre) – without shear and stability"
        return out(d)

    @mcp.tool(structured_output=False)
    def fea_beam_buckling(length_mm: float, case: str = "pinned-pinned", section: str | dict | None = None,
                          I_mm4: float | None = None, A_mm2: float | None = None, E_MPa: float = 210000.0,
                          axis: str | None = None, beta: float | None = None, fy_MPa: float | None = None) -> str:
        """Euler F_cr = pi^2*E*I/(beta*L)^2. case fixed-free (beta 2) | pinned-pinned (1) | fixed-pinned (0.699) | fixed-fixed (0.5); beta overrides; section without axis = weaker axis. Returns F_cr [N], buckling length, slenderness, sigma_cr; with fy_MPa the relative slenderness. Ideal bar - for steel design use fea_ec3_member_check."""
        from ..beams import analytic as an
        if case not in an.BUCKLING_BETA:
            raise FeaError(f"case must be {' | '.join(an.BUCKLING_BETA)}")
        if not length_mm > 0:
            raise FeaError("length_mm must be positive")
        sec = _stiff(section, I_mm4, E_MPa)
        try:
            r = an.euler_buckling(float(length_mm), E=float(E_MPa), I=I_mm4, section=sec, axis=axis, case=case,
                                  beta=beta, A=A_mm2, fy=fy_MPa)
        except (ValueError, KeyError) as e:
            raise FeaError(str(e)) from None
        return out({"F_cr_N": r.F_cr, "beta": r.beta, "L_cr_mm": r.L_cr, "i_mm": r.i, "slenderness": r.slenderness,
                    "sigma_cr_MPa": r.sigma_cr, "rel_slenderness": r.rel_slenderness,
                    "section": sec.name if sec is not None else None,
                    "warning": "Ideal (Euler) member without imperfections – the real capacity is lower (EN 1993-1-1, 6.3.1)"})

    @mcp.tool(structured_output=False)
    def fea_curved_beam(kind: str = "winkler", profile: dict | None = None, M_Nmm: float = 0.0, N_N: float = 0.0,
                        r_inner_mm: float | None = None, R_mm: float | None = None, theta_deg: float | None = None,
                        Fr_N: float = 0.0, Ft_N: float = 0.0, F_N: float = 0.0, E_MPa: float = 210000.0,
                        section: str | dict | None = None, I_mm4: float | None = None, axial: bool = False,
                        shear: bool = False) -> str:
        """Curved bars. kind "winkler": inner/outer fibre stresses from M_Nmm (> 0 increases curvature) and N_N (> 0 tension); profile {"shape": rect (b,h) | circle (d) | tube (D,t) | trapezoid (b_in,b_out,h) | tee (b_f,t_f,t_w,h,flange inner|outer) | ishape (h,b_in,t_in,b_out,t_out,t_w) | polygon (points [[x,y]], y radially outward)}; position r_inner_mm OR R_mm (centroid). Returns sigma_i, sigma_o, vs straight bar, K_i, K_o, neutral axis, eccentricity e. "arc_cantilever": arc R_mm, theta_deg, end loads Fr_N (radial out), Ft_N, M_Nmm; section/I_mm4, E_MPa (axial, shear = include those compliances) -> end displacements, rotation, clamp moment (Castigliano). "ring_pinch": thin ring R_mm squeezed by two opposite F_N (Roark 9.2) -> diameter changes, moments at the load and at 90 deg."""
        from ..beams import analytic as an
        k = kind.strip().lower()
        try:
            if k == "winkler":
                if not isinstance(profile, dict) or "shape" not in profile:
                    raise FeaError("winkler needs profile {shape: rect | circle | tube | trapezoid | tee | ishape | polygon, dimensions mm}")
                p = dict(profile)
                shape = str(p.pop("shape")).lower()
                makers = {"rect": lambda: an.rect(float(p["b"]), float(p["h"])),
                          "circle": lambda: an.circle(float(p["d"])),
                          "tube": lambda: an.tube(float(p["D"]), float(p["t"])),
                          "trapezoid": lambda: an.trapezoid(float(p["b_in"]), float(p["b_out"]), float(p["h"])),
                          "tee": lambda: an.tee(float(p["b_f"]), float(p["t_f"]), float(p["t_w"]), float(p["h"]),
                                                p.get("flange", "inner")),
                          "ishape": lambda: an.ishape(float(p["h"]), float(p["b_in"]), float(p["t_in"]),
                                                      float(p["b_out"]), float(p["t_out"]), float(p["t_w"])),
                          "polygon": lambda: an.polygon(p["points"])}
                if shape not in makers:
                    raise FeaError(f"profile.shape must be {' | '.join(makers)}")
                prof = makers[shape]()
                r = an.curved_beam(prof, float(M_Nmm), float(N_N), r_inner=r_inner_mm, R=R_mm)
                return out({"kind": "Winkler–Bach (curved member)", "A_mm2": r.A, "R_centroid_mm": r.R,
                            "r_neutral_mm": r.r_n, "e_mm": r.e, "r_inner_mm": r.r_i, "r_outer_mm": r.r_o,
                            "I_mm4": r.I, "sigma_inner_MPa": r.sigma_i, "sigma_outer_MPa": r.sigma_o,
                            "sigma_inner_straight_MPa": r.sigma_i_straight,
                            "sigma_outer_straight_MPa": r.sigma_o_straight, "K_inner": r.K_i, "K_outer": r.K_o,
                            "warnings": r.warnings,
                            "sign": "M > 0 increases the curvature (tension in the outer fibres), N > 0 tension; σ(r) = N/A + M(r − r_n)/(A·e·r)"})
            sec = section_of(section) if section is not None else None
            if sec is None and I_mm4 is None:
                raise FeaError(f"{k}: give section or I_mm4")
            if not (R_mm and R_mm > 0):
                raise FeaError(f"{k}: R_mm (radius of the centreline) must be positive")
            if k == "arc_cantilever":
                if theta_deg is None or not 0 < theta_deg <= 360:
                    raise FeaError("arc_cantilever: theta_deg v (0; 360]")
                r = an.arc_cantilever(float(R_mm), math.radians(float(theta_deg)), Fr=float(Fr_N), Ft=float(Ft_N),
                                      M=float(M_Nmm), E=float(E_MPa), I=I_mm4, section=sec, axial=bool(axial),
                                      shear=bool(shear))
                return out({"kind": "cantilever circular arc (Castigliano)", "u_radial_mm": r.u_r,
                            "u_tangential_mm": r.u_t, "rotation_rad": r.rot, "ux_mm": r.ux, "uy_mm": r.uy,
                            "M_fixed_Nmm": r.M_fixed, "M_absmax_Nmm": r.M_absmax,
                            "geometry": "centre at the origin, clamped at (R, 0), arc counter-clockwise"})
            if k == "ring_pinch":
                r = an.ring_pinch(float(F_N), float(R_mm), E=float(E_MPa), I=I_mm4, section=sec, axial=bool(axial),
                                  shear=bool(shear))
                return out({"kind": "ring compressed by two forces (Roark 9.2)", "dD_load_mm": r.dD_load,
                            "dD_perp_mm": r.dD_perp, "M_load_Nmm": r.M_load, "M_side_Nmm": r.M_side,
                            "N_side_N": r.N_side, "sign": "ΔD < 0 = shortening; M > 0 increases the curvature"})
        except FeaError:
            raise
        except KeyError as e:
            raise FeaError(f"Dimension {e} is missing") from None
        except ValueError as e:
            raise FeaError(str(e)) from None
        raise FeaError("kind must be winkler | arc_cantilever | ring_pinch")

    @mcp.tool(structured_output=False)
    def fea_ec3_member_check(section: str | dict, length_mm: float, forces: dict, grade: str = "S235",
                             member: dict | None = None) -> str:
        """EN 1993-1-1 steel member check (national annex: gamma_M0 = gamma_M1 = 1.0): classification, section resistance along x (tension, compression, bending, shear, torsion, 6.2 interaction), buckling 6.3.1 (also torsional), LTB 6.3.2, interaction 6.3.3 (annex B). section as fea_section_get; grade S235|S275|S355|S420|S460. forces (design values): {"x_mm":[...], "N_N" (>0 tension), "Vy_N", "Vz_N", "T_Nmm", "My_Nmm", "Mz_Nmm"} - lists matching x_mm or constants. member (MemberSpec, optional): Lcr_y, Lcr_z, Lcr_T, L_LT [mm] or ky, kz, kT; C1, C2, zg, Mcr [N*mm], lt_restrained, lt_method auto|general|rolled, fabrication rolled|welded|hot_finished|cold_formed|auto, sway_y, sway_z, Cmy, Cmz, CmLT, load_type auto|uniform|concentrated, fy, E, G, gamma_M0, gamma_M1, section_class. Returns utilisation, governing check, rows."""
        from ..beams import ec3
        if not length_mm > 0:
            raise FeaError("length_mm must be positive")
        sec = section_of(section)
        if not isinstance(forces, dict):
            raise FeaError("forces must be an object {x_mm, N_N, Vy_N, Vz_N, T_Nmm, My_Nmm, Mz_Nmm}")
        keymap = {"x_mm": "x", "N_N": "N", "Vy_N": "Vy", "Vz_N": "Vz", "T_Nmm": "T", "My_Nmm": "My", "Mz_Nmm": "Mz"}
        bad = set(forces) - set(keymap) - set(keymap.values())
        if bad:
            raise FeaError(f"Unknown forces keys: {', '.join(sorted(bad))} (allowed: {', '.join(keymap)})")
        f = {keymap.get(k, k): v for k, v in forces.items()}
        if f.get("x") is None:
            f["x"] = [0.0, float(length_mm)]
        mspec = dict(member or {})
        names = {fl.name for fl in fields(ec3.MemberSpec)} - {"L", "grade"}
        bad = set(mspec) - names
        if bad:
            raise FeaError(f"Unknown member parameters: {', '.join(sorted(bad))} (allowed: {', '.join(sorted(names))})")
        try:
            mf = ec3.MemberForces(**f)
            spec = ec3.MemberSpec(L=float(length_mm), grade=grade, **mspec)
            c = ec3.check_member(sec, mf, spec)
        except (ec3.EC3Error, ValueError, TypeError, KeyError) as e:
            raise FeaError(str(e)) from None
        rows = [{"check": r.name, "clause": r.clause, "effect": r.effect, "resistance": r.resistance, "unit": r.unit,
                 "utilization": r.utilization if math.isfinite(r.utilization) else None, "x_mm": r.x,
                 "informative": r.informative, **({"note": r.note} if r.note else {})} for r in c.rows]
        return out({"section": c.section, "grade": c.grade, "fy_MPa": c.fy, "section_class": c.section_class,
                    "utilization": c.utilization, "status": "PASS" if c.ok else "FAIL",
                    "governing": c.governing, "x_crit_mm": c.x_crit, "checks": rows, "classification": c.classification,
                    "warnings": c.warnings,
                    "note": "Rows (informative) are not counted in the utilization; the effect/resistance units follow unit"})
