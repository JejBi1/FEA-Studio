"""Results of the dynamics (Results.dynamics): modal basis (float32), residual vectors, modal coordinates of the harmonic
and transient response (the field can be computed at any time u = Σφq + Σrc), envelopes, probes, random vibrations, shock.
Storage: <stem>_dyn.npz (without compression) + meta JSON in the file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .response import P_VM


@dataclass
class DynamicsResults:
    freq: np.ndarray
    zeta: np.ndarray
    shapes: np.ndarray                    # (K, N, 3) float32 normalised by the mass
    stress: np.ndarray                    # (K, N, 6) float32
    gamma: np.ndarray                     # (K, 6)
    eff_mass: np.ndarray                  # (K, 6)
    rf_sum: np.ndarray                    # (K, 3)
    checks: dict = field(default_factory=dict)
    model_mass: float = 0.0
    prestress: str | None = None
    residual: dict | None = None          # {"U" (R, N, 3), "S" (R, N, 6), "RF" (R, 3), "labels"}
    harmonic: dict | None = None
    transient: dict | None = None
    random: dict | None = None
    shock: dict | None = None
    campbell: dict | None = None          # Campbell diagram (rotor.campbell): rpm, freq (n, K), critical
    probes: list = field(default_factory=list)   # [{"name", "point", "nodes", "w", "quantity"}]
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.freq)

    # ---------------------------------------------------------------- fields on request
    def _res(self, key):
        return self.residual if self.residual is not None else None

    def harmonic_complex(self, i: int, stress: bool = True):
        h = self.harmonic
        q, c = h["q"][i], h["c"][i]
        U = np.tensordot(q, self.shapes, axes=(0, 0))
        S = np.tensordot(q, self.stress, axes=(0, 0)) if stress else None
        res = h.get("res_index")
        if self.residual is not None and len(c):
            ru, rs = self.residual["U"][res], self.residual["S"][res]
            U = U + np.tensordot(c, ru, axes=(0, 0))
            if stress:
                S = S + np.tensordot(c, rs, axes=(0, 0))
        return U, S

    def harmonic_field(self, i: int, phase_deg: float = 0.0, absolute: bool = False):
        """Real field in the time ωt = phase (displacement relative to the base, absolute = + displacement of the base)."""
        U, S = self.harmonic_complex(i)
        e = np.exp(1j * np.deg2rad(phase_deg))
        Ur = (U * e).real
        if absolute:
            Ur = Ur + (self.harmonic["ub"][i] * e).real[None, :]
        return Ur, (S * e).real

    def harmonic_amplitude(self, i: int):
        """Amplitudes over the phase: |u| (N,), σ_vM (N,) – the maximum over the phase."""
        from .response import max_over_phase
        U, S = self.harmonic_complex(i)
        return max_over_phase(U.real, -U.imag), max_over_phase(S.real, -S.imag, P_VM)

    def transient_field(self, i: int, absolute: bool = False):
        tr = self.transient
        q, c = tr["q"][i], tr["c"][i]
        U = np.tensordot(q, self.shapes.astype(float), axes=(0, 0))
        S = np.tensordot(q, self.stress.astype(float), axes=(0, 0))
        res = tr.get("res_index")
        if self.residual is not None and len(c):
            U = U + np.tensordot(c, self.residual["U"][res].astype(float), axes=(0, 0))
            S = S + np.tensordot(c, self.residual["S"][res].astype(float), axes=(0, 0))
        if absolute:
            U = U + tr["ub"][i][None, :]
        return U, S

    # ---------------------------------------------------------------- souhrn
    def summary(self) -> dict:
        d = {"modes": len(self.freq), "frequencies_Hz": [round(float(f), 6) for f in self.freq[:50]],
             "zeta": [round(float(z), 6) for z in self.zeta[:50]], "model_mass_kg": 1000 * self.model_mass,
             "checks": self.checks, "prestress": self.prestress, "warnings": self.warnings}
        if self.harmonic is not None:
            h = self.harmonic
            d["harmonic"] = {"f_range_Hz": [float(h["f"][0]), float(h["f"][-1])], "points": int(len(h["f"])),
                             "max_displacement_mm": float(h["env_U"].max()),
                             "max_mises_MPa": float(h["env_S"].max()),
                             "f_at_max_mises_Hz": float(h["f"][h["env_S_at"][int(h["env_S"].argmax())]]),
                             "probes": {k: _probe_peak(v, h["f"], True) for k, v in h["probes"].items()}}
            if "sweep" in h:
                d["harmonic"]["sweep_fatigue"] = h["sweep"]["summary"]
        if self.transient is not None:
            tr = self.transient
            d["transient"] = {"t_end_s": float(tr["t"][-1]), "steps": int(len(tr["t"])),
                              "max_displacement_mm": float(tr["env_U"].max()),
                              "max_mises_MPa": float(tr["env_S"].max()),
                              "t_at_max_mises_s": float(tr["t_env"][tr["env_S_at"][int(tr["env_S"].argmax())]]),
                              "probes": {k: _probe_peak(v, tr["t"], False) for k, v in tr["probes"].items()},
                              "settle_s": tr.get("settle", {})}
        if self.random is not None:
            d["random"] = self.random.get("summary", {})
        if self.shock is not None:
            d["shock"] = self.shock.get("summary", {})
        if self.campbell is not None:
            d["campbell"] = {"rpm": np.asarray(self.campbell["rpm"]).tolist(),
                             "critical_speeds": self.campbell.get("critical", []), "note": self.campbell.get("note")}
        return d

    # ---------------------------------------------------------------- storage
    def save(self, path: str | Path) -> None:
        arrays = {"freq": self.freq, "zeta": self.zeta, "shapes": self.shapes.astype(np.float32),
                  "stress": self.stress.astype(np.float32), "gamma": self.gamma, "eff_mass": self.eff_mass,
                  "rf_sum": self.rf_sum}
        meta = {"checks": self.checks, "model_mass": self.model_mass, "prestress": self.prestress,
                "warnings": self.warnings, "info": self.info,
                "probes": [{k: (np.asarray(v).tolist() if k in ("nodes", "w") else v) for k, v in p.items()}
                           for p in self.probes]}
        if self.residual is not None:
            for k in ("U", "S", "RF"):
                arrays[f"res_{k}"] = self.residual[k]
            meta["residual_labels"] = self.residual.get("labels", [])
        for name in ("harmonic", "transient", "random", "shock", "campbell"):
            d = getattr(self, name)
            if d is None:
                continue
            meta[name] = _pack(d, name, arrays)
        arrays["meta"] = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "DynamicsResults":
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta = json.loads(bytes(d["meta"]).decode("utf-8"))
        r = cls(d["freq"], d["zeta"], d["shapes"], d["stress"], d["gamma"], d["eff_mass"], d["rf_sum"],
                meta.get("checks", {}), meta.get("model_mass", 0.0), meta.get("prestress"))
        r.warnings = meta.get("warnings", [])
        r.info = meta.get("info", {})
        r.probes = [{k: (np.asarray(v) if k in ("nodes", "w") else v) for k, v in p.items()}
                    for p in meta.get("probes", [])]
        if "res_U" in d:
            r.residual = {"U": d["res_U"], "S": d["res_S"], "RF": d["res_RF"], "labels": meta.get("residual_labels", [])}
        for name in ("harmonic", "transient", "random", "shock", "campbell"):
            if name in meta:
                setattr(r, name, _unpack(meta[name], name, d))
        return r


def _probe_peak(v: dict, x: np.ndarray, cplx: bool) -> dict:
    U = v["U"]
    mag = np.abs(U) if cplx else np.abs(U)
    m = np.linalg.norm(mag, axis=1)
    i = int(m.argmax())
    out = {"max_U_mm": float(m[i]), "at": float(x[i])}
    if "S" in v:
        from .response import max_over_phase
        S = v["S"]
        vm = max_over_phase(S.real, -S.imag, P_VM) if cplx else np.sqrt(np.maximum(np.einsum("ti,ij,tj->t", S, P_VM, S), 0))
        out["max_mises_MPa"] = float(vm.max())
    return out


def _pack(d: dict, prefix: str, arrays: dict) -> dict:
    """Dictionary of the result → arrays into the npz (keys prefix__...) + the rest in JSON; nested probes {name: {U, S, A}}."""
    meta = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            arrays[f"{prefix}__{k}"] = v
            meta[k] = {"__array__": f"{prefix}__{k}"}
        elif k == "probes" and isinstance(v, dict):
            pm = {}
            for pn, series in v.items():
                pm[pn] = {}
                for sk, sv in series.items():
                    key = f"{prefix}__probe__{len(arrays)}"
                    arrays[key] = np.asarray(sv)
                    pm[pn][sk] = {"__array__": key}
            meta[k] = pm
        elif isinstance(v, dict):
            meta[k] = _pack(v, f"{prefix}_{k}", arrays) if any(isinstance(x, np.ndarray) for x in v.values()) else v
        else:
            meta[k] = v
    return meta


def _unpack(meta, prefix: str, d: dict):
    if isinstance(meta, dict):
        if "__array__" in meta and len(meta) == 1:
            return d[meta["__array__"]]
        return {k: _unpack(v, prefix, d) for k, v in meta.items()}
    return meta
