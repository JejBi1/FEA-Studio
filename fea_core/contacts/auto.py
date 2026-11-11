"""Automatic contact pairs (F8.5b, specification 4.6): the interfaces of the clamped parts of the bolted joints as contact with friction
μ = μ_T of the bolt (BoltSpec.mu_T), a conforming mesh with the interface split (the nodes opposite each other, without the initial gap).
Preload = the first step, in the other steps the length locked by *BOUNDARY, FIXED (contacts.run, ccx/inp).
"""
from __future__ import annotations

from .model import ContactPair


def bolt_pairs(study) -> list[ContactPair]:
    """Pairs (upper part, lower part) of the clamped bundle of all active bolts; μ = the smallest μ_T of the bolts of the interface."""
    pairs: dict[tuple[str, str], float] = {}
    for b in study.active_bolts():
        stack = [pl["part"] for pl in b.plates]
        if b.connection == "tapped" and b.nut_part:
            stack.append(b.nut_part)
        for up, lo in zip(stack[:-1], stack[1:]):
            if up == lo:
                continue
            key = (up, lo) if up < lo else (lo, up)
            pairs[key] = min(pairs.get(key, float(b.mu_T)), float(b.mu_T))
    return [ContactPair(f"Interface {a}–{b}", a, b, kind="frictional", mu=mu, purpose="slip", source="auto")
            for (a, b), mu in sorted(pairs.items())]
