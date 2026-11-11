"""Courses of the drivers and the loads of the mechanism in time (without Exudyn – shared by the main process and the worker).

Laws (Driver.law): constant [value] | table [[t, value], …] (linearly, outside the range constantly) | sine
[amplitude, f Hz, phase rad, centre] | poly [c0, c1, …] (c0 + c1·t + …) | scurve [stroke, v_max, a_max, j_max,
t_start] (profile of the axis with limited jerk – 7 sections, jerk ±j; unreachable v_max / a_max are reduced).

Law.value(t), d1(t), d2(t) = the value and its derivatives; integral(t) = ∫₀ᵗ value dτ (a driver by the speed → position).
`scale` = conversion of the units (project → SI) multiplies the values, not the time.
"""
from __future__ import annotations

import math

import numpy as np

POSITION_QUANTITIES = ("angle", "disp")
VELOCITY_QUANTITIES = ("velocity", "angular_velocity")
FORCE_QUANTITIES = ("torque", "force")
KINEMATIC_QUANTITIES = POSITION_QUANTITIES + VELOCITY_QUANTITIES


class LawError(ValueError):
    """Invalid course of the driver (the data do not match the law)."""


def scurve_segments(stroke: float, v_max: float, a_max: float, j_max: float) -> list[tuple[float, float]]:
    """Sections (time, jerk) of the profile with limited jerk from rest to rest over the distance |stroke| (the sign according to stroke)."""
    s = abs(float(stroke))
    v, a, j = abs(float(v_max)), abs(float(a_max)), abs(float(j_max))
    if s == 0:
        return []
    if not (v > 0 and a > 0 and j > 0):
        raise LawError("S-curve: v_max, a_max and j_max must be positive")
    if v * j < a * a:                         # a_max is not reached even with v_max
        a = math.sqrt(v * j)
    ta = v / a + a / j                        # doba rozjezdu na v
    if v * ta > s:                            # v_max is not reached → a lower speed
        v = 0.5 * a * (-a / j + math.sqrt((a / j) ** 2 + 4.0 * s / a))
        if v * j < a * a:                     # also the acceleration only triangular
            v = (s * math.sqrt(j) / 2.0) ** (2.0 / 3.0)
            a = math.sqrt(v * j)
        ta = v / a + a / j
    tj = a / j
    tc = ta - 2 * tj                          # constant acceleration
    tv = max((s - v * ta) / v, 0.0)           # driving at a constant speed
    sg = 1.0 if stroke > 0 else -1.0
    seg = [(tj, j), (tc, 0.0), (tj, -j), (tv, 0.0), (tj, -j), (tc, 0.0), (tj, j)]
    return [(d, sg * jj) for d, jj in seg if d > 1e-15]


class Law:
    """Course of a quantity in time (see the module)."""

    def __init__(self, kind: str, data, scale: float = 1.0):
        self.kind = kind
        self.scale = float(scale)
        try:
            if kind == "constant":
                self.c = float(np.ravel(data)[0]) if data is not None and len(np.ravel(data)) else 0.0
            elif kind == "table":
                p = np.asarray(data, float).reshape(-1, 2)
                p = p[np.argsort(p[:, 0], kind="stable")]
                if len(p) < 1:
                    raise LawError("the table is empty")
                self.tt, self.vv = p[:, 0], p[:, 1]
                dt = np.diff(self.tt)
                self.slope = np.where(dt > 0, np.diff(self.vv) / np.where(dt > 0, dt, 1.0), 0.0)
                # ∫ from the first point to each point (trapezium – exact for linear sections)
                self.cum = np.concatenate([[0.0], np.cumsum(0.5 * (self.vv[1:] + self.vv[:-1]) * dt)])
            elif kind == "sine":
                d = list(np.ravel(data)) + [0.0] * 4
                self.A, self.f, self.ph, self.mean = (float(x) for x in d[:4])
            elif kind == "poly":
                self.coef = np.asarray(np.ravel(data), float)
                if not len(self.coef):
                    self.coef = np.zeros(1)
            elif kind == "scurve":
                d = list(np.ravel(data))
                if len(d) < 4:
                    raise LawError("S-curve needs [stroke, v_max, a_max, j_max, t_start]")
                self.t0 = float(d[4]) if len(d) > 4 else 0.0
                self.seg = scurve_segments(*d[:4])
                # states at the starts of the sections (p, v, a)
                st = [(0.0, 0.0, 0.0)]
                for dur, j in self.seg:
                    st.append(self._advance(st[-1], j, dur))
                self.states = st
                self.tb = np.concatenate([[0.0], np.cumsum([s[0] for s in self.seg])])
            else:
                raise LawError(f"unknown drive law '{kind}' (constant | table | sine | poly | scurve)")
        except (TypeError, ValueError, IndexError) as e:
            if isinstance(e, LawError):
                raise
            raise LawError(f"invalid data of law '{kind}': {e}") from None

    @staticmethod
    def _advance(state, j, d):
        p, v, a = state
        return (p + v * d + a * d * d / 2 + j * d ** 3 / 6, v + a * d + j * d * d / 2, a + j * d)

    @property
    def duration(self) -> float:
        """Duration of the profile of the S-curve [s] (from t_start), otherwise 0."""
        return float(self.tb[-1]) if self.kind == "scurve" else 0.0

    # ---------------------------------------------------------------- evaluation
    def _scurve(self, t: float, order: int) -> float:
        tau = t - self.t0
        if tau <= 0:
            return 0.0
        k = int(np.searchsorted(self.tb, tau, side="right") - 1)
        if k >= len(self.seg):
            p, v, a = self.states[-1]
            return (p, 0.0, 0.0)[order]
        st = self._advance(self.states[k], self.seg[k][1], tau - self.tb[k])
        return st[order]

    def _raw(self, t: float, order: int) -> float:
        k = self.kind
        if k == "constant":
            return self.c if order == 0 else 0.0
        if k == "table":
            if order >= 2:
                return 0.0
            if t <= self.tt[0] or t >= self.tt[-1]:
                return (self.vv[0] if t <= self.tt[0] else self.vv[-1]) if order == 0 else 0.0
            i = int(np.searchsorted(self.tt, t, side="right") - 1)
            return self.vv[i] + self.slope[i] * (t - self.tt[i]) if order == 0 else self.slope[i]
        if k == "sine":
            w = 2 * math.pi * self.f
            x = w * t + self.ph
            return (self.mean + self.A * math.sin(x), self.A * w * math.cos(x), -self.A * w * w * math.sin(x))[order]
        if k == "poly":
            c = self.coef
            for _ in range(order):
                c = c[1:] * np.arange(1, len(c))
            return float(np.polyval(c[::-1], t)) if len(c) else 0.0
        return self._scurve(t, order)

    def value(self, t: float) -> float:
        return self.scale * self._raw(float(t), 0)

    def d1(self, t: float) -> float:
        return self.scale * self._raw(float(t), 1)

    def d2(self, t: float) -> float:
        return self.scale * self._raw(float(t), 2)

    def _prim(self, t: float) -> float:
        """Primitive function (an arbitrary constant)."""
        k = self.kind
        if k == "constant":
            return self.c * t
        if k == "table":
            if t <= self.tt[0]:
                return self.vv[0] * (t - self.tt[0])
            if t >= self.tt[-1]:
                return self.cum[-1] + self.vv[-1] * (t - self.tt[-1])
            i = int(np.searchsorted(self.tt, t, side="right") - 1)
            d = t - self.tt[i]
            return self.cum[i] + self.vv[i] * d + 0.5 * self.slope[i] * d * d
        if k == "sine":
            w = 2 * math.pi * self.f
            return self.mean * t - (self.A / w * math.cos(w * t + self.ph) if w else -self.A * math.sin(self.ph) * t)
        if k == "poly":
            c = self.coef / np.arange(1, len(self.coef) + 1)
            return float(np.polyval(np.concatenate([[0.0], c])[::-1], t))
        raise LawError("the S-curve is not integrated (it is the position itself)")

    def integral(self, t: float) -> float:
        """∫₀ᵗ value dτ."""
        return self.scale * (self._prim(float(t)) - self._prim(0.0))


class DriverMotion:
    """Prescribed coordinate of the joint from the driver (measured from the position in the CAD = 0): quantity angle/disp → coordinate =
    the law (non-zero at time 0 → the mechanism is assembled at the start into this position); velocity → coordinate =
    ∫₀ᵗ law (the S-curve is always directly the profile of the position); coord(t), rate(t), accel(t) in SI.
    """

    def __init__(self, quantity: str, law: Law):
        self.q, self.law = quantity, law
        self.integrate = quantity in VELOCITY_QUANTITIES and law.kind != "scurve"

    def coord(self, t):
        return self.law.integral(t) if self.integrate else self.law.value(t)

    def rate(self, t):
        return self.law.value(t) if self.integrate else self.law.d1(t)

    def accel(self, t):
        return self.law.d1(t) if self.integrate else self.law.d2(t)
