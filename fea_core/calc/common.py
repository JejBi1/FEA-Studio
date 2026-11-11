"""Shared helpers of the calculators: error, check of the inputs, shape of the result and selection of the function by the kind (MCP, GUI)."""
from __future__ import annotations

import inspect
import math

G = 9810.0          # mm/s²


class CalcError(ValueError):
    """Invalid input of the calculator (text understandable for the user)."""


def positive(name: str, v, zero: bool = False) -> float:
    """Positive finite number (zero=True allows 0 too)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise CalcError(f"{name}: number expected, got {v!r}") from None
    if not math.isfinite(x) or x < 0 or (x == 0 and not zero):
        raise CalcError(f"{name} must be {('non-negative' if zero else 'positive')} (got {x:g})")
    return x


def number(name: str, v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise CalcError(f"{name}: number expected, got {v!r}") from None
    if not math.isfinite(x):
        raise CalcError(f"{name} must be a finite number")
    return x


def choice(name: str, v, options) -> str:
    s = str(v).strip().lower()
    if s not in options:
        raise CalcError(f"{name} must be {' | '.join(options)} (got {v!r})")
    return s


def result(values: dict, formula: str, source: str, warnings: list | None = None, **extra) -> dict:
    """Result of a calculator: values + text of the formula + source (+ warnings)."""
    d = dict(values)
    d["formula"] = formula
    d["source"] = source
    if warnings:
        d["warnings"] = list(warnings)
    d.update(extra)
    return d


# ---------------------------------------------------------------- selection of a function by the kind
GROUPS = {"snapfit": "snapfit", "pressfit": "pressfit", "hertz": "hertz", "friction": "friction",
          "mechanism": "mechanisms", "damping": "damping", "pulse": "pulses", "convection": "convection"}
KINDS: dict[str, dict] = {}


def kinds(group: str) -> dict:
    """{druh: (funkce, popis)} skupiny (snapfit | pressfit | hertz | friction | mechanism | damping | pulse)."""
    if group not in GROUPS:
        raise CalcError(f"Unknown calculator group '{group}' ({' | '.join(GROUPS)})")
    if group not in KINDS:
        import importlib
        KINDS[group] = importlib.import_module(f"{__package__}.{GROUPS[group]}").KINDS
    return KINDS[group]


def parameters(fn) -> list[inspect.Parameter]:
    return [p for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]


def call(group: str, kind: str, params: dict | None = None) -> dict:
    """Calls a calculator of the group by the kind with the parameters {name: value}; an unknown / missing parameter = CalcError
    with the list of the allowed ones.
    """
    table = kinds(group)
    k = str(kind).strip()
    if k not in table:
        raise CalcError(f"Unknown type '{kind}' ({' | '.join(table)})")
    fn = table[k][0]
    ps = parameters(fn)
    names = [p.name for p in ps]
    params = dict(params or {})
    bad = sorted(set(params) - set(names))
    if bad:
        raise CalcError(f"{k}: unknown parameters {', '.join(bad)} (allowed: {', '.join(names)})")
    missing = [p.name for p in ps if p.default is p.empty and p.name not in params]
    if missing:
        raise CalcError(f"{k}: missing parameters {', '.join(missing)} (allowed: {', '.join(names)})")
    try:
        return fn(**params)
    except CalcError:
        raise
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
        raise CalcError(f"{k}: {e}") from None


def signature_text(fn) -> str:
    """"a, b=1.0, …" for the help of MCP."""
    out = []
    for p in parameters(fn):
        out.append(p.name if p.default is p.empty else f"{p.name}={p.default!r}")
    return ", ".join(out)
