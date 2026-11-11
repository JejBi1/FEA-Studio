"""References of the study to the parts by name – renaming of the parts after the update of the geometry (a part renamed in the CAD,
a new body in the component: "Deska:1" → "Deska:1_1").

Works on the dictionary of the study (Study.to_dict → modification → Study.from_dict): the only place that knows all the fields
with the name of a part. The test `tests/test_cad_link.py::test_part_fields_covered` goes through the dataclasses reachable from Study and
fails if a new field with a part is there that is missing here.
"""
from __future__ import annotations

# kind of load with the list of parts "parts" (empty = all)
_LOAD_PARTS = ("rotation", "inertia", "thermal")


def _ren(v, m: dict[str, str]):
    return m.get(v, v) if isinstance(v, str) else v


def _ren_list(xs, m: dict[str, str]) -> list:
    out = []
    for x in xs or []:
        y = _ren(x, m)
        if y not in out:
            out.append(y)
    return out


def _ren_keys(d, m: dict[str, str]) -> dict:
    return {_ren(k, m): v for k, v in (d or {}).items()}


def rename_parts(d: dict, mapping: dict[str, str]) -> list[str]:
    """Renames the parts in the dictionary of the study in place (mapping old → new name). Returns the descriptions of the changed places."""
    m = {a: b for a, b in mapping.items() if a != b}
    if not m:
        return []
    touched: list[str] = []

    def note(where: str, before, after):
        if before != after:
            touched.append(where)

    pm = d.get("part_materials") or {}
    d["part_materials"] = _ren_keys(pm, m)
    note("part materials", pm, d["part_materials"])
    for key, what in (("ignored_parts", "ignored parts"), ("hidden_parts", "hidden parts")):
        if key not in d:
            continue
        old = d.get(key) or []
        d[key] = _ren_list(old, m)
        note(what, old, d[key])
    for s in (d.get("selections") or {}).values():
        for kind in ("faces", "edges", "points"):
            for sel in s.get(kind) or []:
                sel["part"] = _ren(sel.get("part"), m)
    for b in d.get("bolts") or []:
        before = (b.get("head_part"), b.get("nut_part"), list(b.get("cad_parts") or []))
        b["head_part"] = _ren(b.get("head_part"), m)
        b["nut_part"] = _ren(b.get("nut_part"), m)
        b["cad_parts"] = _ren_list(b.get("cad_parts"), m)
        for row in (b.get("plates") or []) + (b.get("washers") or []):
            row["part"] = _ren(row.get("part"), m)
        note(f"bolt {b.get('name')}", before, (b["head_part"], b["nut_part"], b["cad_parts"]))
    for w in d.get("welds") or []:
        old = w.get("part")
        w["part"] = _ren(old, m)
        note(f"weld {w.get('name')}", old, w["part"])
    pr = d.get("printing") or {}
    if pr:
        old = dict(pr.get("parts") or {})
        pr["parts"] = {_ren(k, m): {**v, "part": _ren(v.get("part", k), m)} if isinstance(v, dict) else v
                       for k, v in old.items()}
        note("3D print settings", list(old), list(pr["parts"]))
        for var in pr.get("variants") or []:
            var["parts"] = _ren_keys(var.get("parts"), m)
        sw = pr.get("sweep")
        if isinstance(sw, dict):
            sw["parts"] = _ren_list(sw.get("parts"), m)
    mesh = d.get("mesh") or {}
    if mesh:
        mesh["separate"] = _ren_list(mesh.get("separate"), m)
        mesh["part_sizes"] = _ren_keys(mesh.get("part_sizes"), m)
    sh = d.get("shell") or {}
    if sh:
        sh["parts"] = _ren_keys(sh.get("parts"), m)
    tw = d.get("twod") or {}
    if tw:
        tw["parts"] = _ren_list(tw.get("parts"), m)
    for case in (d.get("thermal") or {}).get("cases") or []:
        for bc in case.get("loads") or []:
            if "part" in bc:
                bc["part"] = _ren(bc["part"], m)
    for lc in d.get("load_cases") or []:
        for ld in lc.get("loads") or []:
            if ld.get("kind") in _LOAD_PARTS and ld.get("parts"):
                ld["parts"] = _ren_list(ld["parts"], m)
    sel_names = set((d.get("selections") or {}).keys())
    for c in d.get("contacts") or []:                    # a / b = selection, otherwise the name of the part
        for k in ("a", "b"):
            if c.get(k) not in sel_names:
                c[k] = _ren(c.get(k), m)
    mech = d.get("mechanism") or {}
    for body in mech.get("bodies") or []:
        body["parts"] = _ren_list(body.get("parts"), m)
    tf = mech.get("to_fea")
    if isinstance(tf, dict):
        tf["parts"] = _ren_list(tf.get("parts"), m)
    for imp in d.get("imprints") or []:
        imp["part"] = _ren(imp.get("part"), m)
    return touched


def references(d: dict, part: str) -> bool:
    """Does the study refer to the part `part` (material, selection, bolt, weld, printing, mesh …)?"""
    import copy
    import json
    probe = copy.deepcopy(d)
    rename_parts(probe, {part: "\x00odebráno\x00"})
    return json.dumps(probe, sort_keys=True, default=str) != json.dumps(d, sort_keys=True, default=str)
