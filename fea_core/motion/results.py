"""Results of the mechanism (MotionResult): fields at the output times in the units of the project (mm, N, N·mm, t, s, rad).

Fields (N = number of times, nb = bodies, nj = joints, nd = drivers, nf = force elements, nl = loads):
  t (N)                                    times [s]
  pos, vel, acc (N, nb, 3)                 centroid of the body [mm, mm/s, mm/s²] (globally)
  rot (N, nb, 3, 3)                        rotation of the body with respect to the position in the CAD (x = c(t) + R·(X − c_CAD))
  omega, alpha (N, nb, 3)                  angular velocity and acceleration [rad/s, rad/s²] (globally)
  jF, jM (N, nj, 3)                        TOTAL force [N] and moment [N·mm] of the joint acting on body_b from body_a (constraint + driver + friction + stop + torsion spring + transfer) about the point jP
  jP, jPa (N, nj, 3)                       point of the joint on body_b / body_a [mm]
  jFl, jMl (N, nj, 3)                      the same in the frame of the joint (connected with body_a; z = axis of the joint)
  jq, jv (N, nj, 2)                        coordinate of the joint [rotation rad, displacement mm] and their velocities
  jfr (N, nj, 2)                           friction moment [N·mm] / force [N] on the coordinate (negative = brakes the positive direction)
  jlim (N, nj)                             force / moment of the stop on the main coordinate
  dQ, dq, dv (N, nd)                       moment [N·mm] / force [N] of the driver, coordinate and velocity of the driver
  fval, flen (N, nf)                       force of the element [N] (tension > 0) / moment of the torsion spring [N·mm]; length [mm] / angle
  fF, fM, fPa, fPb (N, nf, 3)              force [N] and moment [N·mm] of the element on body_b, the points of the attachments [mm]
  lF, lM, lP (N, nl, 3)                    external load and its point of application
  ekin, epot (N)                           kinetic and potential energy [N·mm = mJ] (weight + springs)
  wdrv, wfr, wlim, wdamp, wbush, wload (N) work of the drivers, friction, stops, dampers, bushings, loads from t = 0 [N·mm]
  pdrv, pfr, plim, pdamp, pbush, pload, pelem (N) instantaneous powers [N·mm/s] (pelem = all force elements)
`meta` = names, kinds, masses, checks (freedom, equilibrium, power balance, collisions), warnings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARRAYS = ("t", "pos", "rot", "vel", "omega", "acc", "alpha", "jF", "jM", "jP", "jPa", "jFl", "jMl", "jq", "jv",
          "jfr", "jlim", "dQ", "dq", "dv", "fval", "flen", "fF", "fM", "fPa", "fPb", "lF", "lM", "lP", "ekin", "epot",
          "wdrv", "wfr", "wlim", "wdamp", "wbush", "wload", "pdrv", "pfr", "plim", "pdamp", "pbush", "pload", "pelem")
WORKS = ("wdrv", "wfr", "wlim", "wdamp", "wbush", "wload")
WORK_LABELS = {"wdrv": "drives", "wfr": "friction", "wlim": "stops", "wdamp": "dampers", "wbush": "bushings",
               "wload": "external loads"}


@dataclass
class MotionResult:
    arrays: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- access
    def __getattr__(self, name):
        if name in ARRAYS and name in self.__dict__.get("arrays", {}):
            return self.arrays[name]
        raise AttributeError(name)

    @property
    def body_names(self) -> list[str]:
        return [b["name"] for b in self.meta.get("bodies", [])]

    @property
    def joint_names(self) -> list[str]:
        return [j["name"] for j in self.meta.get("joints", [])]

    @property
    def driver_names(self) -> list[str]:
        return [d["name"] for d in self.meta.get("drivers", [])]

    @property
    def force_names(self) -> list[str]:
        return [f["name"] for f in self.meta.get("forces", [])]

    @property
    def warnings(self) -> list[str]:
        return self.meta.setdefault("warnings", [])

    def index(self, kind: str, name: str) -> int:
        names = {"body": self.body_names, "joint": self.joint_names, "driver": self.driver_names,
                 "force": self.force_names}[kind]
        if name in names:
            return names.index(name)
        if kind == "driver":                        # driver also by the name of the joint
            for i, d in enumerate(self.meta.get("drivers", [])):
                if d.get("joint_name") == name:
                    return i
        raise KeyError(f"'{name}' is not in the results ({', '.join(names) or 'nic'})")

    def time_index(self, t: float) -> int:
        return int(np.argmin(np.abs(self.arrays["t"] - float(t))))

    def pose(self, body: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(R, T) transformation of a point from the CAD to the time k: x = R·X + T."""
        R = self.arrays["rot"][k, body]
        c0 = np.asarray(self.meta["bodies"][body]["com"], float)
        return R, self.arrays["pos"][k, body] - R @ c0

    # ---------------------------------------------------------------- series for the graphs
    SERIES = {
        "position": ("body", "pos", "mm", ["x", "y", "z"]),
        "velocity": ("body", "vel", "mm/s", ["vx", "vy", "vz"]),
        "acceleration": ("body", "acc", "mm/s²", ["ax", "ay", "az"]),
        "angular velocity": ("body", "omega", "rad/s", ["ωx", "ωy", "ωz"]),
        "angular acceleration": ("body", "alpha", "rad/s²", ["αx", "αy", "αz"]),
        "reaction": ("joint", "jF", "N", ["Fx", "Fy", "Fz"]),
        "joint moment": ("joint", "jM", "N·mm", ["Mx", "My", "Mz"]),
        "reaction in joint frame": ("joint", "jFl", "N", ["F1", "F2", "F3 (axis)"]),
        "moment in joint frame": ("joint", "jMl", "N·mm", ["M1", "M2", "M3 (axis)"]),
        "angle": ("joint", "jq0", "rad", ["φ"]),
        "translation": ("joint", "jq1", "mm", ["s"]),
        "joint velocity": ("joint", "jv", "rad/s | mm/s", ["ω", "v"]),
        "friction": ("joint", "jfr", "N·mm | N", ["moment", "force"]),
        "stop": ("joint", "jlim", "N·mm | N", ["stop"]),
        "drive": ("driver", "dQ", "N·mm | N", ["M/F"]),
        "drive moment": ("driver", "dQ", "N·mm", ["M"]),
        "drive force": ("driver", "dQ", "N", ["F"]),
        "drive power": ("driver", "dP", "W", ["P"]),
        "element force": ("force", "fval", "N | N·mm", ["F"]),
        "element length": ("force", "flen", "mm | rad", ["L"]),
    }
    # Czech names (tool documentation, studies and scripts from older versions) → the names above
    ALIASES = {
        "pozice": "position", "poloha": "position", "rychlost": "velocity", "zrychlení": "acceleration",
        "zrychleni": "acceleration", "úhlová rychlost": "angular velocity", "uhlova rychlost": "angular velocity",
        "úhlové zrychlení": "angular acceleration", "uhlove zrychleni": "angular acceleration", "reakce": "reaction",
        "moment kloubu": "joint moment", "reakce v rámu kloubu": "reaction in joint frame",
        "moment v rámu kloubu": "moment in joint frame", "úhel": "angle", "uhel": "angle", "posuv": "translation",
        "rychlost kloubu": "joint velocity", "tření": "friction", "treni": "friction", "doraz": "stop",
        "pohon": "drive", "moment pohonu": "drive moment", "síla pohonu": "drive force", "sila pohonu": "drive force",
        "výkon pohonu": "drive power", "vykon pohonu": "drive power", "síla prvku": "element force",
        "sila prvku": "element force", "délka prvku": "element length", "delka prvku": "element length",
        "energie": "energy", "práce": "work", "prace": "work", "vzdálenost": "distance", "vzdalenost": "distance",
        "napětí": "stress", "napeti": "stress", "deformace": "deformation", "kontaktní síla": "contact force",
        "kontaktni sila": "contact force", "kontakt": "contact force", "kontaktní moment": "contact moment",
        "kontaktni moment": "contact moment",
    }

    @classmethod
    def canonical(cls, quantity: str) -> str:
        """English name of a quantity (Czech aliases accepted); the item after ':' is kept."""
        q, sep, item = quantity.partition(":")
        q = q.strip().lower()
        return cls.ALIASES.get(q, q) + (sep + item if sep else "")

    def series(self, name: str) -> dict:
        """Series for a graph: "quantity:item" (e.g. "reakce:Kloub 1", "pozice:Ojnice", "moment pohonu:Motor") or
        "energie", "práce", "vzdálenost:A|B". Returns {t, values (N, k), labels, unit, magnitude?}.
        """
        a = self.arrays
        q, _, item = self.canonical(name).partition(":")
        item = item.strip()
        if q == "energy":
            ek, ep = a["ekin"], a["epot"]
            return {"t": a["t"], "values": np.column_stack([ek, ep, ek + ep]), "labels": ["kinetic", "potential",
                    "total"], "unit": "mJ"}
        if q == "work":
            keys = [k for k in WORKS if np.abs(a[k]).max() > 0] or ["wdrv"]
            return {"t": a["t"], "values": np.column_stack([a[k] for k in keys]),
                    "labels": [WORK_LABELS[k] for k in keys], "unit": "mJ"}
        if q == "distance":
            col = self.meta.get("collisions") or {}
            for p in col.get("pairs", []):
                if f"{p['a']}|{p['b']}" == item or f"{p['b']}|{p['a']}" == item:
                    return {"t": np.asarray(col["t"]), "values": np.asarray(p["distance"])[:, None],
                            "labels": ["distance"], "unit": "mm"}
            raise KeyError(f"Pair '{item}' is not in the collision check")
        if q in ("stress", "deformation") and self.meta.get("flex"):
            from .flex import stress_history
            h = stress_history(self, item)
            if q == "deformation":
                return {"t": a["t"], "values": h["u_max"][:, None], "labels": ["max |u|"], "unit": "mm"}
            return {"t": a["t"], "values": h["von_mises_max"][:, None], "labels": ["max σ_vM"], "unit": "MPa",
                    "node_max": h["node_max"], "point_mm": h["point_mm"]}
        if q in ("contact force", "contact moment") and "cF" in a:
            i = self.index("body", item)
            if q == "contact moment":                   # about the centroid of the body
                v = a["cM"][:, i]
                return {"t": a["t"], "values": v, "labels": ["Mx", "My", "Mz"], "unit": "N·mm",
                        "magnitude": np.linalg.norm(v, axis=1)}
            v = a["cF"][:, i]
            return {"t": a["t"], "values": v, "labels": ["Fx", "Fy", "Fz"], "unit": "N",
                    "magnitude": np.linalg.norm(v, axis=1)}
        if q not in self.SERIES:
            extra = ["energy", "work", "distance"] + (["contact force", "contact moment"] if "cF" in a else []) \
                + (["stress", "deformation"] if self.meta.get("flex") else [])
            raise KeyError(f"Unknown quantity '{q}' – known: {', '.join(list(self.SERIES) + extra)}")
        kind, key, unit, labels = self.SERIES[q]
        i = self.index(kind, item)
        if key in ("jq0", "jq1"):
            v = a["jq"][:, i, 0 if key == "jq0" else 1][:, None]
        elif key == "dP":
            v = (a["dQ"][:, i] * a["dv"][:, i])[:, None] * (1e-3 if self.meta["drivers"][i]["coord"] == "rot" else 1e-3)
        else:
            v = a[key][:, i]
            v = v[:, None] if v.ndim == 1 else v
        out = {"t": a["t"], "values": v, "labels": labels[:v.shape[1]], "unit": unit}
        if key in ("dQ", "jlim", "fval", "flen") or v.shape[1] == 1:
            out["unit"] = self._unit(kind, i, key, unit)
        if v.shape[1] == 3:
            out["magnitude"] = np.linalg.norm(v, axis=1)
        return out

    def _unit(self, kind, i, key, unit):
        if kind == "driver":
            return "N·mm" if self.meta["drivers"][i]["coord"] == "rot" else "N"
        if kind == "force":
            tors = self.meta["forces"][i]["kind"] == "torsion"
            return ("N·mm" if tors else "N") if key == "fval" else ("rad" if tors else "mm")
        if kind == "joint" and key == "jlim":
            return "N·mm" if self.meta["joints"][i].get("primary") == "rot" else "N"
        return unit

    def find_max(self, quantity: str, component: int | None = None) -> dict:
        """"Find maximum": the instant of the largest magnitude of a quantity (for a vector |v|, otherwise |value|)."""
        s = self.series(quantity)
        v = s["values"]
        if component is not None:
            mag = np.abs(v[:, component])
        elif "magnitude" in s:
            mag = s["magnitude"]
        else:
            mag = np.abs(v).max(axis=1)
        k = int(np.argmax(mag))
        return {"quantity": quantity, "index": k, "t": float(s["t"][k]), "value": float(mag[k]),
                "vector": [float(x) for x in v[k]], "unit": s["unit"]}

    # ---------------------------------------------------------------- souhrn
    def summary(self) -> dict:
        a, m = self.arrays, self.meta
        out = {"kind": m.get("kind"), "t_end_s": float(a["t"][-1]) if len(a["t"]) else 0.0, "frames": int(len(a["t"])),
               "dof": m.get("dof"), "time_s": m.get("time_s")}
        bodies = []
        for i, b in enumerate(m.get("bodies", [])):
            if b.get("ground"):
                continue
            bodies.append({"body": b["name"], "mass_kg": b["mass"] * 1e3,
                           "max_velocity_mm_s": float(np.linalg.norm(a["vel"][:, i], axis=1).max()),
                           "max_acceleration_mm_s2": float(np.linalg.norm(a["acc"][:, i], axis=1).max()),
                           "max_omega_rad_s": float(np.linalg.norm(a["omega"][:, i], axis=1).max())})
        out["bodies"] = bodies
        joints = []
        for i, j in enumerate(m.get("joints", [])):
            F = np.linalg.norm(a["jF"][:, i], axis=1)
            M = np.linalg.norm(a["jM"][:, i], axis=1)
            k = int(np.argmax(F))
            d = {"joint": j["name"], "kind": j["kind"], "max_force_N": float(F.max()), "t_max_force_s": float(a["t"][k]),
                 "max_moment_Nmm": float(M.max())}
            if np.abs(a["jfr"][:, i]).max() > 0:
                d["max_friction"] = float(np.abs(a["jfr"][:, i]).max())
            if np.abs(a["jlim"][:, i]).max() > 0:
                d["max_stop_force"] = float(np.abs(a["jlim"][:, i]).max())
            joints.append(d)
        out["joints"] = joints
        drivers = []
        for i, d in enumerate(m.get("drivers", [])):
            Q = a["dQ"][:, i]
            P = Q * a["dv"][:, i] * 1e-3
            drivers.append({"driver": d["name"], "joint": d.get("joint_name"), "quantity": d["quantity"],
                            "unit": "N·mm" if d["coord"] == "rot" else "N", "max_abs": float(np.abs(Q).max()),
                            "t_max_s": float(a["t"][int(np.argmax(np.abs(Q)))]), "rms": float(np.sqrt(np.mean(Q * Q))),
                            "max_power_W": float(np.abs(P).max())})
        out["drivers"] = drivers
        if m.get("forces"):
            out["forces"] = [{"force": f["name"], "kind": f["kind"], "max_abs": float(np.abs(a["fval"][:, i]).max())}
                             for i, f in enumerate(m["forces"])]
        if m.get("flex"):
            from .flex import stress_history
            out["flexible"] = []
            for f in m["flex"]:
                h = stress_history(self, f["body"])
                out["flexible"].append({"body": f["body"], "modes": f["modes"], "freq_hz": f["freq_hz"][:6],
                                        "nodes": f["nodes"], "max_von_mises_MPa": float(h["von_mises_max"].max()),
                                        "t_max_s": float(h["t"][h["k_max"]]), "at_mm": h["point_mm"],
                                        "max_deformation_mm": float(h["u_max"].max())})
        if "cF" in a and m.get("contacts"):
            out["solver"] = m.get("solver", "chrono")
            out["contacts"] = []
            for c in m["contacts"]:
                rec = {"contact": c["name"]}
                for side in ("a", "b"):
                    i = c[side]
                    if i is not None and i >= 0 and not m["bodies"][i]["ground"]:
                        F = np.linalg.norm(a["cF"][:, i], axis=1)
                        k = int(np.argmax(F))
                        rec[f"max_force_on_{m['bodies'][i]['name']}_N"] = float(F[k])
                        rec["t_max_s"] = float(a["t"][k])
                out["contacts"].append(rec)
            out["contacts_max_points"] = m.get("contacts_max")
        for key in ("equilibrium", "power_balance", "collisions_summary", "mass_check"):
            if m.get(key):
                out[key] = m[key]
        if m.get("warnings"):
            out["warnings"] = m["warnings"]
        return out

    # ---------------------------------------------------------------- file
    def save(self, path: str | Path) -> None:
        """npz without compression (fields + meta as JSON)."""
        arr = {k: np.asarray(v) for k, v in self.arrays.items()}
        arr["_meta"] = np.array(json.dumps(self.meta, ensure_ascii=False, default=_json_default))
        np.savez(path, **arr)

    @classmethod
    def load(cls, path: str | Path) -> "MotionResult":
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "_meta"}
            meta = json.loads(str(z["_meta"])) if "_meta" in z.files else {}
        return cls(arrays, meta)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o).__name__)
