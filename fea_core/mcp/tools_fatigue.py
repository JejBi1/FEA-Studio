"""Tool: fatigue analysis (ch. 7) over the results of the statics – superposition of the states, rainflow, Miner."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import out
from .session import FeaError, Session, user_file

METHODS = {"mises_signed": "signed_mises", "signed_mises": "signed_mises", "max_principal": "max_principal",
           "principal": "max_principal", "critical_plane": "critical_plane", "findley": "critical_plane"}


def parse_history(hc, load_cases: list[str]):
    """Load curve from various notations → LoadHistory.

    - {"channels": [{"load_case", "kind": "points"|"sine", "points": [[t, f], …], "amplitude", "mean",
      "cycles", …}], "duration": T}  (study format)
    - {"load_case": "…", "points": [[t, f], …]} or {"load_case", "kind": "sine", "amplitude", "mean", "cycles"}
    - [[t, f], …] or [f1, f2, …] → the first load state
    - "path.csv" or {"csv": "path.csv", "load_case": "…"} (columns time; multiple)
    """
    from ..fatigue.history import LoadHistory, Channel, read_csv_points
    if not load_cases:
        raise FeaError("The results have no load case")
    if isinstance(hc, str):
        hc = {"csv": hc}
    if isinstance(hc, list):
        if not hc:
            raise FeaError("history_curve is empty")
        pts = [[float(i), float(v)] for i, v in enumerate(hc)] if not isinstance(hc[0], (list, tuple)) else \
              [[float(a), float(b)] for a, b in hc]
        hc = {"load_case": load_cases[0], "points": pts}
    if not isinstance(hc, dict):
        raise FeaError("history_curve: a dictionary, a list of points or a path to a CSV expected")
    if "channels" in hc:
        try:
            hist = LoadHistory.from_dict(hc)
        except TypeError as e:
            raise FeaError(f"history_curve.channels: {e}") from None
    else:
        ch = dict(hc)
        if "csv" in ch:
            p = user_file(ch.pop("csv"), 256 << 20, "CSV")
            ch["points"] = read_csv_points(p)
            ch.setdefault("kind", "points")
        ch.setdefault("load_case", load_cases[0])
        duration = float(ch.pop("duration", 1.0))
        try:
            hist = LoadHistory([Channel(**ch)], duration)
        except TypeError as e:
            raise FeaError(f"history_curve: {e}") from None
        if hist.channels[0].kind == "points" and hist.channels[0].points:
            ts = [p[0] for p in hist.channels[0].points]
            hist.duration = max(duration, max(ts) - min(ts)) if "duration" not in hc else duration
    for c in hist.channels:
        if c.load_case not in load_cases:
            raise FeaError(f"The channel refers to state '{c.load_case}', the results have: {', '.join(load_cases)}")
        if c.kind not in ("points", "sine"):
            raise FeaError("channel kind must be points or sine")
    return hist


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_fatigue_run(history_curve: dict | list | str, method: str = "mises_signed",
                        sn_overrides: dict | None = None, mean_correction: str = "goodman",
                        repeats: float = 1e6, surface: str | None = None, reliability: float | None = None,
                        top_n: int = 10, cp_criterion: str | None = None, cp_ratio: dict | float | None = None,
                        cp_thorough: bool | None = None) -> str:
        """Fatigue: sigma(t) = sum f_k(t)*sigma_k over states -> signed equivalent stress -> rainflow (ASTM E1049) -> Miner -> damage, life (blocks), safety (stress multiple to failure at `repeats` blocks). history_curve: {"load_case","points":[[t,multiple],...]} | {"load_case","kind":"sine","amplitude","mean","cycles"} | {"channels":[...],"duration":T} | [[t,f],...] | CSV path. method: mises_signed | max_principal | critical_plane (multiaxial/non-proportional/shear; cp_criterion auto (grey iron: normal, else findley) | findley | normal; cp_ratio = f-1/t-1 in 1-2, default sqrt(3), number or {"material": s}; cp_thorough). mean_correction none | goodman | gerber. sn_overrides: {"Material": {"SD" MPa, "ND" cycles, "k", "k2": 0}} or one S-N for all (default: estimate from Rm with Marin factors). surface machined | ground | polished | rolled | forged ...; reliability [%]; top_n worst places. Result fields fatigue_* are renderable."""
        from ..fatigue.settings import FatigueSettings, EQUIVALENT_LABELS, MEAN_LABELS
        from ..fatigue.sn import SURFACE_LABELS
        from ..fatigue.analysis import run_fatigue
        if method in ("en", "strain_life"):
            fs_method = "en"
            meth = "signed_mises"
            if mean_correction in ("goodman", "gerber"):
                raise FeaError(f"Mean stress correction '{mean_correction}' is not supported for the ε-N method (options: none | morrow | swt)")
        else:
            fs_method = "sn"
            meth = METHODS.get(method, method)
            if meth not in EQUIVALENT_LABELS:
                raise FeaError(f"Unknown method '{method}' (mises_signed | max_principal | critical_plane | en)")
            if mean_correction in ("morrow", "swt"):
                raise FeaError(f"Mean stress correction '{mean_correction}' is meant for the ε-N method. For S-N choose from: none | goodman | gerber")
        if mean_correction not in MEAN_LABELS:
            raise FeaError(f"mean_correction must be {' | '.join(MEAN_LABELS)}")
        if surface is not None and surface not in SURFACE_LABELS:
            raise FeaError(f"surface must be {' | '.join(SURFACE_LABELS)}")
        if repeats <= 0:
            raise FeaError("repeats must be positive")
        with sess.use(op="fatigue"):
            st = sess.require_study()
            res = sess.require_results()
            lcs = [s.name for s in res.steps]
            hist = parse_history(history_curve, lcs)
            # the settings are checked on a copy – they are written to the study only after a successful calculation
            fs = FatigueSettings.from_dict(st.fatigue.to_dict()) if st.fatigue else FatigueSettings()
            fs.method, fs.history, fs.equivalent, fs.mean_correction, fs.repeats = fs_method, hist, meth, mean_correction, float(repeats)
            if surface is not None:
                fs.surface = surface
            if cp_criterion is not None:
                from ..fatigue.settings import CP_CRITERION_LABELS
                if cp_criterion not in CP_CRITERION_LABELS:
                    raise FeaError(f"cp_criterion must be {' | '.join(CP_CRITERION_LABELS)}")
                fs.cp_criterion = cp_criterion
            if cp_ratio is not None:
                ratios = cp_ratio if isinstance(cp_ratio, dict) else {m: cp_ratio for m in st.materials}
                for mname, v in ratios.items():
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        fv = float("nan")
                    if mname not in st.materials or not 1.0 < fv <= 2.0:
                        raise FeaError("cp_ratio: f₋₁/t₋₁ in the range (1; 2] for the study material")
                    fs.cp_ratio[mname] = fv
            if cp_thorough is not None:
                fs.cp_thorough = bool(cp_thorough)
            if reliability is not None:
                fs.reliability = float(reliability)
            if sn_overrides:
                mats = {st.material_of(p).name for p in res.mesh.part_names}
                if {"SD"} <= set(sn_overrides):                       # one S-N for all
                    sn_overrides = {m: dict(sn_overrides) for m in mats}
                for mname, sn in sn_overrides.items():
                    if mname not in st.materials:
                        raise FeaError(f"sn_overrides: material '{mname}' is not in the study ({', '.join(st.materials)})")
                    bad = set(sn) - {"SD", "ND", "k", "k2", "source"} if isinstance(sn, dict) else {"?"}
                    if bad or "SD" not in sn:
                        raise FeaError("Give the S-N as {\"SD\": MPa, \"ND\": cycles, \"k\": slope, \"k2\": slope below the knee (0 = fatigue limit)}")
                    try:
                        fs.sn[mname] = {"source": "given via MCP", **{k: float(v) if k != "source" else v
                                                                         for k, v in sn.items()}}
                    except (TypeError, ValueError):
                        raise FeaError(f"sn_overrides '{mname}': the values SD, ND, k, k2 must be numbers") from None
            old = st.fatigue
            st.fatigue = fs
            try:
                fr = run_fatigue(st, res)
            except (KeyError, ValueError) as e:
                st.fatigue = old
                raise FeaError(f"Fatigue cannot be computed: {e}") from None
            except Exception:
                st.fatigue = old
                raise
            sess.fatigue = fr
            sess.joints = None                     # the joint assessment (incl. fatigue of welds/bolts) is recalculated
            n_top = max(1, min(int(top_n), 50))
            if fr.damage.max() > 0:
                worst = fr.worst(res.mesh, n_top)
                i = int(np.argmax(fr.damage))
            else:                                   # no damage (below the fatigue limit) → sort by safety
                order = np.argsort(fr.safety)[:n_top]
                worst = [{"node": int(j), "xyz": res.mesh.nodes[j], "damage_per_block": 0.0, "total_damage": 0.0,
                          "life_blocks": float("inf"), "safety": float(fr.safety[j]),
                          "amplitude": float(fr.amplitude[j])} for j in order]
                i = int(order[0])
            for w in worst:
                w["node"] += 1
            return out({
                "method": fr.method_lines(), "time_points": len(fr.t), "load_cases": fr.load_cases,
                "repeats_blocks": fr.repeats,
                "max_total_damage": float(fr.total_damage.max()),
                "min_life_blocks": float(fr.life_blocks.min()),
                "min_safety": float(np.min(fr.safety)), "worst_at_mm": res.mesh.nodes[i],
                "verdict": ("PASS" if fr.total_damage.max() < 1 else "FAIL") + " (D < 1 for the given number of blocks)",
                "sn_used": fr.sn_used, "worst": worst, "warnings": fr.warnings, "time_s": fr.time_s,
                "hint": "Mapy: fea_results_render_image(field='fatigue_damage' | 'fatigue_safety' | 'fatigue_life')",
            })
