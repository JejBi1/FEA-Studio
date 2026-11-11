"""Matching of the parts of the old and the new version of the geometry: which part is which (renaming, new and removed parts).

Order: identity of the body from the CAD (token translated by the add-in) → the same name → the same name without the suffixes of the occurrence
("Deska:1" ↔ "Deska:1_1") → geometry (volume, position of the centroid, number of faces; the assignment with the smallest total
deviation). A match by geometry alone is "probable" and is shown for checking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .digest import part_fingerprint, same_shape


@dataclass
class PartMatch:
    old: str | None
    new: str | None
    method: str                  # cad | name | name_key | geometry | added | removed
    changed: bool = False        # geometry of the part changed (volume, faces, envelope)
    note: str = ""
    fp_old: dict = field(default_factory=dict, repr=False)
    fp_new: dict = field(default_factory=dict, repr=False)

    @property
    def certain(self) -> bool:
        return self.method in ("cad", "name", "name_key")

    def to_dict(self) -> dict:
        d = {"old": self.old, "new": self.new, "method": self.method, "changed": self.changed}
        if self.note:
            d["note"] = self.note
        if self.fp_old and self.fp_new:
            d["volume_change_pct"] = round(100.0 * (self.fp_new["volume"] / max(self.fp_old["volume"], 1e-12) - 1), 3)
        return d


def _cost(a: dict, b: dict, size: float) -> float:
    va, vb = max(a["volume"], 1e-12), max(b["volume"], 1e-12)
    dv = abs(np.log(vb / va))
    dc = float(np.linalg.norm(np.asarray(a["center"]) - np.asarray(b["center"]))) / size
    df = abs(a["faces"] - b["faces"]) / max(a["faces"], b["faces"], 1)
    return dv + dc + 0.2 * df


def _cad_pairs(old_geo, new_geo, old_meta, new_meta, fp_new) -> dict[int, int]:
    """Parts matched by the token of the body: old row of the metadata → token → bodies of the new version (volume + centroid)."""
    from .facetable import resolved_bodies, row_part_map
    res = resolved_bodies(new_meta)
    if not res or not old_meta:
        return {}
    rows = (old_meta or {}).get("parts") or []
    row_to_old = row_part_map(old_meta, [p.name for p in old_geo.parts])
    out: dict[int, int] = {}
    for ri, oi in row_to_old.items():
        tok = rows[ri].get("token") if isinstance(rows[ri], dict) else None
        bodies = res.get(tok) if isinstance(tok, str) else None
        if not bodies or len(bodies) != 1:          # the body disappeared / was split → the geometry decides
            continue
        b = bodies[0]
        best = None
        for ni, fp in enumerate(fp_new):
            size = max(float(np.linalg.norm(np.asarray(fp["bbox"][3:]) - np.asarray(fp["bbox"][:3]))), 1e-9)
            dv = abs(fp["volume"] - b["volume"]) / max(abs(b["volume"]), 1e-12)
            dc = float(np.linalg.norm(np.asarray(fp["center"]) - np.asarray(b["center"]))) / size
            if dv <= 1e-3 and dc <= 1e-2:
                s = dv + dc
                if best is None or s < best[0]:
                    best = (s, ni)
        if best is not None and best[1] not in out.values():
            out[oi] = best[1]
    return out


def map_parts(old_geo, new_geo, old_meta: dict | None = None, new_meta: dict | None = None) -> list[PartMatch]:
    from ..cad_import import name_keys
    fp_old = [part_fingerprint(p) for p in old_geo.parts]
    fp_new = [part_fingerprint(p) for p in new_geo.parts]
    old_names = [p.name for p in old_geo.parts]
    new_names = [p.name for p in new_geo.parts]
    size = max(old_geo.size, new_geo.size)
    pairs: dict[int, tuple[int, str]] = {}

    def free_new() -> list[int]:
        taken = {n for n, _ in pairs.values()}
        return [j for j in range(len(new_names)) if j not in taken]

    for oi, ni in _cad_pairs(old_geo, new_geo, old_meta, new_meta, fp_new).items():
        pairs[oi] = (ni, "cad")
    pos_new = {n: j for j, n in enumerate(new_names)}
    for oi, n in enumerate(old_names):
        if oi not in pairs and n in pos_new and pos_new[n] in free_new():
            pairs[oi] = (pos_new[n], "name")
    # same name without the suffixes of the occurrence / order of the body – several candidates, the geometry decides
    for oi, n in enumerate(old_names):
        if oi in pairs:
            continue
        keys = set(name_keys(n))
        cands = [j for j in free_new() if keys & set(name_keys(new_names[j]))]
        if not cands:
            continue
        costs = sorted((_cost(fp_old[oi], fp_new[j], size), j) for j in cands)
        if len(costs) == 1 or costs[1][0] > 2 * costs[0][0] + 1e-9:
            pairs[oi] = (costs[0][1], "name_key")
    # the rest by geometry (volume 0.5–2×, centroid within a quarter of the size of the assembly)
    rest_old = [i for i in range(len(old_names)) if i not in pairs]
    rest_new = free_new()
    if rest_old and rest_new:
        # greedily from the best pair (import scipy.optimize takes seconds; for a few parts it does not matter)
        cand = sorted((_cost(fp_old[i], fp_new[j], size), i, j) for i in rest_old for j in rest_new)
        used_o, used_n = set(), set()
        for _, i, j in cand:
            if i in used_o or j in used_n:
                continue
            a, b = fp_old[i], fp_new[j]
            ratio = b["volume"] / max(a["volume"], 1e-12)
            dc = float(np.linalg.norm(np.asarray(a["center"]) - np.asarray(b["center"])))
            if 0.5 <= ratio <= 2.0 and dc <= 0.25 * size:
                pairs[i] = (j, "geometry")
                used_o.add(i)
                used_n.add(j)
    out: list[PartMatch] = []
    for oi, n in enumerate(old_names):
        if oi in pairs:
            ni, how = pairs[oi]
            out.append(PartMatch(n, new_names[ni], how, not same_shape(fp_old[oi], fp_new[ni]),
                                 fp_old=fp_old[oi], fp_new=fp_new[ni]))
        else:
            out.append(PartMatch(n, None, "removed", True, fp_old=fp_old[oi]))
    taken = {ni for ni, _ in pairs.values()}
    for j, n in enumerate(new_names):
        if j not in taken:
            out.append(PartMatch(None, n, "added", True, fp_new=fp_new[j]))
    return out
