"""Load curve: channels = multiples of the unit load states in time.

The stress in time is composed by superposition (linear elasticity): σ(t) = Σ_k f_k(t) · σ_k,
where σ_k is the stress from the load state k (computed as one ccx step).
Several channels on the same state are added.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Channel:
    load_case: str                    # name of the load state in the study
    kind: str = "points"              # "points" | "sine"
    points: list[list[float]] = field(default_factory=list)   # [[t, multiple], ...]
    amplitude: float = 1.0            # sinus: f = mean + amplitude·sin(2π·cycles·t/T + phase)
    mean: float = 0.0
    cycles: float = 1.0               # number of cycles per block
    phase_deg: float = 0.0
    samples_per_cycle: int = 24
    scale: float = 1.0                # additional multiple of the whole channel

    def times(self, duration: float) -> np.ndarray:
        if self.kind == "sine":
            n = max(int(np.ceil(self.cycles * self.samples_per_cycle)), 2)
            t = np.linspace(0.0, duration, n + 1)
            # exact peaks of the sine so that the amplitude is not cut by sampling
            k = np.arange(-1, int(np.ceil(2 * self.cycles)) + 2)
            tp = (k * 0.5 + 0.25 - self.phase_deg / 360.0) * duration / max(self.cycles, 1e-12)
            return np.union1d(t, tp[(tp >= 0) & (tp <= duration)])
        if not self.points:
            return np.array([0.0])
        return np.array([p[0] for p in self.points], float)

    def value(self, t: np.ndarray, duration: float) -> np.ndarray:
        if self.kind == "sine":
            w = 2 * np.pi * self.cycles / max(duration, 1e-12)
            f = self.mean + self.amplitude * np.sin(w * t + np.radians(self.phase_deg))
        elif not self.points:
            f = np.zeros_like(t)
        else:
            p = np.asarray(self.points, float)
            order = np.argsort(p[:, 0], kind="stable")
            f = np.interp(t, p[order, 0], p[order, 1])
        return self.scale * f


@dataclass
class LoadHistory:
    channels: list[Channel] = field(default_factory=list)
    duration: float = 1.0             # length of the block (for the sine and the time axis of the graph)

    def time_grid(self) -> np.ndarray:
        ts = [np.array([0.0])]
        for ch in self.channels:
            ts.append(ch.times(self.duration))
        t = np.unique(np.concatenate(ts))
        if all(ch.kind == "sine" for ch in self.channels) or not self.channels:
            t = t[(t >= 0) & (t <= self.duration)]
        return t

    def factors(self, load_cases: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(t, F) – F[i, k] = multiple of the load state load_cases[k] at the time t[i]."""
        t = self.time_grid()
        F = np.zeros((len(t), len(load_cases)))
        for ch in self.channels:
            if ch.load_case not in load_cases:
                raise KeyError(f"Load case '{ch.load_case}' is not in the results")
            F[:, load_cases.index(ch.load_case)] += ch.value(t, self.duration)
        return t, F

    @classmethod
    def from_dict(cls, d: dict) -> "LoadHistory":
        names = set(Channel.__dataclass_fields__)          # ignore unknown keys (newer version)
        return cls([Channel(**{k: v for k, v in c.items() if k in names}) for c in d.get("channels", [])],
                   d.get("duration", 1.0))


def read_csv_points(path: str | Path) -> list[list[float]]:
    """CSV with two columns (time; multiple) or one (multiples, time = order).

    Separator ; , or tab, decimal comma and decimal point both allowed, the heading is skipped.
    """
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # separator for the whole file: ; or tab takes precedence (then the comma is the decimal point); a bare comma is
    # a separator only when the file contains decimal points or non-numeric headings – otherwise "1,5" is ambiguous
    delim = next((d for d in ";\t" if any(d in l for l in lines)), None)
    if delim is None and any("," in l for l in lines):
        def numeric(tok):
            try:
                float(tok)
                return True
            except ValueError:
                return False
        dots = any("." in l for l in lines)
        header = any(not all(numeric(p.strip()) for p in l.split(",") if p.strip()) for l in lines)
        if dots or header or any(l.count(",") > 1 for l in lines):
            delim = ","
        else:
            raise ValueError("The CSV is ambiguous (a comma can be a separator or a decimal comma, e.g. “1,5”) – use the ; separator or a decimal point")
    rows = []
    for line in lines:
        parts = [p.strip() for p in (line.split(delim) if delim else [line])]
        try:
            vals = [float(p.replace(",", ".")) for p in parts if p]
        except ValueError:
            continue                              # heading
        if vals:
            rows.append(vals)
    if not rows:
        raise ValueError("The CSV contains no numbers")
    if all(len(r) == 1 for r in rows):
        return [[float(i), r[0]] for i, r in enumerate(rows)]
    return [[r[0], r[1]] for r in rows if len(r) >= 2]
