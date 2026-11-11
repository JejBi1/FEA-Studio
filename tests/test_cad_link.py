"""Geometry update keeping the settings (fea_core.cad_link): remapping of the selections, parts, points, CAD tokens,
renaming of the parts in the study, iterations.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from fea_core.cad_link import update
from fea_core.cad_link.faces import Geometry
from fea_core.cad_link.refs import references, rename_parts
from fea_core.selectors import find_faces, resolve
from fea_core.study import Fixed, Force, LoadCase, Pressure, RemoteLoad, Study


# ===================================================================== geometrie

def plate(L=100.0, T=10.0, D=10.0, HX=30.0, W=60.0, fillet=5.0, slot=None, hole2=None):
    """Plate L×W×T with a hole Ø D at (HX, W/2), vertical edge x=L, y=W rounded with R fillet; slot = (x0, x1, depth)."""
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from examples.geometry import box, cut, cyl, solid
    from fea_core.geometry import edge_props
    s = cut(box(0, 0, 0, L, W, T), cyl((HX, W / 2, -1), (0, 0, 1), D / 2, T + 2))
    if hole2 is not None:
        s = cut(s, cyl((hole2[0], W / 2, -1), (0, 0, 1), hole2[1] / 2, T + 2))
    if slot is not None:
        x0, x1, h = slot
        s = cut(s, box(x0, -1, T - h, x1 - x0, W + 2, h + 1))
    s = solid(s)
    if fillet:
        mk = BRepFilletAPI_MakeFillet(s)
        ex = TopExp_Explorer(s, TopAbs_EDGE)
        while ex.More():
            e = TopoDS.Edge(ex.Current())
            p = edge_props(e)
            if p["type"] == "line" and abs(p["centroid"][0] - L) < 1e-6 and abs(p["centroid"][1] - W) < 1e-6:
                mk.Add(fillet, e)
            ex.Next()
        s = solid(mk.Shape())
    return s


def parts(name="Deska:1", **kw):
    from fea_core.geometry import Part
    return [Part(name, plate(**kw))]


def base_study(ps):
    st = Study("deska", "deska.step")
    st.add_selection("Vetknutí", find_faces(ps, type="plane", x=0.0))
    st.add_selection("Díra", find_faces(ps, type="cylinder", where=lambda p: abs(p["centroid"][0] - 30) < 1))
    st.add_selection("Horní", find_faces(ps, type="plane", normal=(0, 0, 1)))
    st.add_selection("Čelo", find_faces(ps, type="plane", normal=(1, 0, 0)))
    st.add_selection("Zaoblení", find_faces(ps, type="cylinder", where=lambda p: p["centroid"][0] > 90))
    st.constraints.append(Fixed("Vetknutí"))
    st.load_cases.append(LoadCase("LC1", [Force("Díra", [0, 0, -1000]), Pressure("Horní", 0.1),
                                          RemoteLoad("Čelo", [120.0, 30.0, 5.0], [0, 0, 100])]))
    st.part_materials["Deska:1"] = "S235"
    return st


@pytest.fixture(scope="module")
def old():
    return parts()


def statuses(rep) -> dict[str, str]:
    return {s.name: s.status for s in rep.selections}


def unresolved(st, ps) -> list[str]:
    return sorted({n for n, s in st.selections.items() for f in s.faces if resolve(f, ps) is None})


# ===================================================================== geometric remapping

@pytest.mark.parametrize("kw, expect", [
    (dict(D=14.0), {"Díra": "probable", "Vetknutí": "same", "Zaoblení": "same"}),
    (dict(T=12.0), {"Horní": "probable", "Čelo": "probable"}),
    (dict(L=130.0), {"Díra": "same", "Čelo": "probable", "Zaoblení": "probable"}),
    (dict(HX=60.0), {"Díra": "probable", "Čelo": "same"}),
    (dict(hole2=(70.0, 8.0)), {"Díra": "same", "Horní": "probable"}),
    (dict(slot=(60.0, 70.0, 3.0)), {"Horní": "split", "Díra": "same"}),
    (dict(L=130.0, slot=(80.0, 90.0, 3.0)), {"Horní": "split"}),
])
def test_remap_parametric_edits(old, kw, expect):
    st = base_study(old)
    new = parts(**kw)
    rep = update.plan_update(st, old, new)
    got = statuses(rep)
    for k, v in expect.items():
        assert got[k] == v, (k, got)
    st2 = update.apply_report(st, rep, new_parts=new)
    assert unresolved(st2, new) == []
    assert st2.cad_link is not None and [it.n for it in st2.cad_link.iterations] == [1, 2]
    # split top face: pressure on both pieces
    if "slot" in kw:
        assert len(st2.selections["Horní"].faces) == 2


def test_hole_moves_with_its_selection(old):
    st = base_study(old)
    new = parts(HX=60.0)
    st2 = update.apply_report(st, update.plan_update(st, old, new), new_parts=new)
    c = st2.selections["Díra"].faces[0].centroid
    assert abs(c[0] - 60.0) < 1e-6


def test_lost_face_keeps_old_selector(old):
    st = base_study(old)
    new = parts(fillet=0.0)
    rep = update.plan_update(st, old, new)
    assert statuses(rep)["Zaoblení"] == "lost"
    assert rep.blocking() == ["Zaoblení"]
    st2 = update.apply_report(st, rep, new_parts=new)
    assert unresolved(st2, new) == ["Zaoblení"]                     # the calculation reports the missing face


def test_accept_only_certain(old):
    st = base_study(old)
    new = parts(D=14.0)
    rep = update.plan_update(st, old, new)
    st2 = update.apply_report(st, rep, accept={"same", "cad"}, new_parts=new)
    assert "Díra" in unresolved(st2, new) and "Horní" in unresolved(st2, new)


def test_unchanged_geometry(old):
    st = base_study(old)
    rep = update.plan_update(st, old, parts())
    assert rep.unchanged
    assert not rep.needs_review()


def test_renamed_part_carries_settings(old):
    st = base_study(old)
    new = parts("Plech:1")
    rep = update.plan_update(st, old, new)
    assert [(p.old, p.new, p.method) for p in rep.parts] == [("Deska:1", "Plech:1", "geometry")]
    assert all(s == "same" for s in statuses(rep).values())
    st2 = update.apply_report(st, rep, new_parts=new)
    assert st2.part_materials == {"Plech:1": "S235"}
    assert {f.part for s in st2.selections.values() for f in s.faces} == {"Plech:1"}


def test_new_body_suffix_maps_by_name_key(old):
    """Fusion: a second body in the component → "Deska:1" is named "Deska:1_1"."""
    from fea_core.geometry import Part
    from examples.geometry import box
    st = base_study(old)
    new = parts("Deska:1_1") + [Part("Deska:1_2", box(0, 100, 0, 10, 10, 10))]
    rep = update.plan_update(st, old, new)
    m = {p.old: (p.new, p.method) for p in rep.parts if p.old}
    assert m["Deska:1"] == ("Deska:1_1", "name_key")
    assert any(p.method == "added" and p.new == "Deska:1_2" for p in rep.parts)


def test_remote_point_follows_plane_normal_only(old):
    st = base_study(old)
    for kw, x in ((dict(L=130.0), 150.0), (dict(fillet=0.0), 120.0)):
        new = parts(**kw)
        st2 = update.apply_report(st, update.plan_update(st, old, new), new_parts=new)
        p = st2.load_cases[0].loads[2].point
        assert np.allclose(p, [x, 30.0, 5.0]), (kw, p)


# ===================================================================== CAD tokens (the face table of the add-in)

def _rows(ps, prefix):
    """Face table as from the add-in: token, part, geometry (here from OCP – in the CAD from the kernel of the CAD)."""
    geo = Geometry(ps)
    rows = []
    for pi in range(len(ps)):
        for fi, fp in enumerate(geo.part(pi).faces):
            rows.append({"token": f"{prefix}{pi}_{fi}", "part": pi, "type": fp["type"], "area": float(fp["area"]),
                         "centroid": [float(v) for v in fp["centroid"]], "point": [float(v) for v in fp["point"]],
                         "normal": [float(v) for v in fp["normal"]]})
    return rows


def _meta(ps, rows, resolved=None):
    return {"source": "Autodesk Fusion", "parts": [{"name": p.name} for p in ps], "faces": rows,
            **({"resolved": resolved} if resolved is not None else {})}


def test_cad_tokens_follow_moved_hole_and_split(old):
    """Token of the hole → a new hole (shifted by 30 mm = otherwise only "probable"); the top face split by a groove:
    the CAD returns only one piece (the behaviour of Fusion verified in a live CAD) → the second piece is supplied by the geometry.
    """
    st = base_study(old)
    new = parts(HX=60.0, slot=(75.0, 85.0, 3.0))
    old_rows = _rows(old, "o")
    new_geo = Geometry(new)
    res = []
    for r in old_rows:
        g = None
        if r["type"] == "cylinder" and abs(r["centroid"][0] - 30) < 1:          # hole
            g = [fp for fp in new_geo.part(0).faces if fp["type"] == "cylinder" and abs(fp["centroid"][0] - 60) < 1]
        elif r["type"] == "plane" and r["normal"][2] > 0.99:                     # top: only the piece x < 75
            g = [fp for fp in new_geo.part(0).faces if fp["type"] == "plane" and fp["normal"][2] > 0.99
                 and fp["centroid"][0] < 75 and abs(fp["centroid"][2] - 10) < 1e-6]
        if g is not None:
            res.append({"token": r["token"], "faces": [{"type": fp["type"], "area": float(fp["area"]),
                                                         "centroid": list(map(float, fp["centroid"])),
                                                         "point": list(map(float, fp["point"]))} for fp in g]})
    rep = update.plan_update(st, old, new, _meta(old, old_rows), _meta(new, _rows(new, "n"), res))
    got = {s.name: (s.status, s.items[0].method) for s in rep.selections}
    assert got["Díra"] == ("cad", "cad")
    assert got["Horní"][0] == "split" and got["Horní"][1].startswith("cad")
    st2 = update.apply_report(st, rep, new_parts=new)
    assert len(st2.selections["Horní"].faces) == 2
    assert abs(st2.selections["Díra"].faces[0].centroid[0] - 60) < 1e-6


def test_cad_deleted_face_falls_back_to_geometry(old):
    """An element deleted and created again: the token returns nothing → the geometry decides."""
    st = base_study(old)
    new = parts(D=12.0)
    old_rows = _rows(old, "o")
    res = [{"token": r["token"], "faces": []} for r in old_rows if r["type"] == "cylinder"]
    rep = update.plan_update(st, old, new, _meta(old, old_rows), _meta(new, _rows(new, "n"), res))
    assert statuses(rep)["Díra"] == "probable"


def test_cad_body_token_maps_renamed_part(old):
    st = base_study(old)
    new = parts("Úplně jiný název")
    from fea_core.cad_link.digest import part_fingerprint
    fp = part_fingerprint(new[0])
    old_meta = _meta(old, _rows(old, "o"))
    old_meta["parts"][0]["token"] = "body0"
    new_meta = _meta(new, _rows(new, "n"))
    new_meta["resolved_bodies"] = [{"token": "body0", "bodies": [{"volume": fp["volume"], "center": fp["center"]}]}]
    rep = update.plan_update(st, old, new, old_meta, new_meta)
    assert [(p.old, p.new, p.method) for p in rep.parts] == [("Deska:1", "Úplně jiný název", "cad")]


def test_face_table_rejects_bad_rows():
    from fea_core.cad_link.facetable import face_rows, parameters, resolved
    meta = {"faces": [{"area": "x"}, {"area": 1.0, "centroid": [0, 0, float("nan")]}, {"area": -1, "centroid": [0, 0, 0]},
                      {"area": 2.0, "centroid": [1, 2, 3], "token": "bad token with spaces", "type": "evil"}],
            "resolved": [{"token": "ok", "faces": [{"area": 1.0, "centroid": [0, 0, 0]}]}, {"token": 5}],
            "params": {"Len": {"expression": "130 mm", "value": 13.0, "unit": "mm"}, "x" * 500: {}}}
    rows = face_rows(meta)
    assert len(rows) == 1 and "token" not in rows[0] and rows[0]["type"] == "other"
    assert list(resolved(meta)) == ["ok"]
    assert list(parameters(meta)) == ["Len"]


# ===================================================================== study, renaming, iterations

def test_study_roundtrip_and_old_files(old):
    st = base_study(old)
    assert "cad_link" not in st.to_dict()                      # without a link the file does not change
    new = parts(D=14.0)
    st2 = update.apply_report(st, update.plan_update(st, old, new), new_parts=new)
    d = st2.to_dict()
    assert "cad_link" in d["features"]
    st3 = Study.from_dict(json.loads(json.dumps(d)))
    assert st3.cad_link.to_dict() == st2.cad_link.to_dict()
    assert st3.cad_link.iterations[1].remap == {"same": 3, "probable": 2}


def test_rename_parts_everywhere():
    d = {"part_materials": {"A": "S235", "B": "Al"}, "ignored_parts": ["A"],
         "selections": {"s": {"faces": [{"part": "A"}]}},
         "bolts": [{"name": "Š1", "head_part": "A", "nut_part": "B", "cad_parts": ["A", "C"],
                    "plates": [{"part": "A"}], "washers": [{"part": "C"}]}],
         "welds": [{"name": "W", "part": "A"}],
         "printing": {"parts": {"A": {"part": "A"}}, "variants": [{"parts": {"A": {}}}], "sweep": {"parts": ["A"]}},
         "mesh": {"separate": ["A"], "part_sizes": {"A": 2.0}}, "shell": {"parts": {"A": 2.0}},
         "twod": {"parts": ["A"]}, "thermal": {"cases": [{"loads": [{"kind": "heat_generation", "part": "A"}]}]},
         "load_cases": [{"loads": [{"kind": "rotation", "parts": ["A"]}, {"kind": "force", "selection": "A"}]}],
         "contacts": [{"a": "A", "b": "s"}],
         "mechanism": {"bodies": [{"parts": ["A"]}]}, "imprints": [{"part": "A"}]}
    assert references(d, "A") and not references(d, "Z")
    rename_parts(d, {"A": "A2", "B": "A"})                     # also the exchange of names
    s = json.dumps(d, ensure_ascii=False)
    assert '"A2"' in s and d["part_materials"] == {"A2": "S235", "A": "Al"}
    assert d["contacts"][0] == {"a": "A2", "b": "s"}           # "s" is a selection, not a part
    assert d["load_cases"][0]["loads"][1]["selection"] == "A"  # the name of the selection is not renamed


PART_FIELDS = {"part", "parts", "cad_parts", "head_part", "nut_part", "a", "b"}
COVERED = {("Selection", "faces"), ("FaceSelector", "part"), ("EdgeSelector", "part"), ("PointSelector", "part"),
           ("BoltSpec", "head_part"), ("BoltSpec", "nut_part"), ("BoltSpec", "cad_parts"), ("WeldSpec", "part"),
           ("WeldSpec", "a"), ("PrintSpec", "part"), ("PrintSettings", "parts"), ("PrintVariant", "parts"),
           ("SweepSpec", "parts"), ("ShellSettings", "parts"), ("TwoDSettings", "parts"), ("HeatGeneration", "part"),
           ("Rotation", "parts"), ("Inertia", "parts"), ("ThermalLoad", "parts"), ("ContactPair", "a"),
           ("ContactPair", "b"), ("Body", "parts"), ("ToFea", "parts"), ("Imprint", "part"),
           ("MeshSettings", "part_sizes"), ("Study", "ignored_parts"), ("Study", "hidden_parts"), ("Study", "part_materials")}


def test_part_fields_covered():
    """Every field with the name of a part in the dataclasses of the study must be renamed by refs.rename_parts (the COVERED list)."""
    import importlib
    seen, todo, found = set(), [Study], set()
    mods = ["fea_core.thermal.model", "fea_core.study", "fea_core.joints.model", "fea_core.printing.model",
            "fea_core.motion.model", "fea_core.contacts.model", "fea_core.imprint", "fea_core.dynamics.model",
            "fea_core.shell.model", "fea_core.twod.model", "fea_core.beams.model", "fea_core.eigen",
            "fea_core.combinations", "fea_core.fatigue.settings"]
    for m in mods:
        mod = importlib.import_module(m)
        todo += [v for v in vars(mod).values() if dataclasses.is_dataclass(v) and isinstance(v, type)
                 and v.__module__ == mod.__name__]
    while todo:
        k = todo.pop()
        if k in seen:
            continue
        seen.add(k)
        for f in dataclasses.fields(k):
            if f.name in PART_FIELDS or ("part" in f.name and f.name != "participation"):
                found.add((k.__name__, f.name))
    missing = {x for x in found if x not in COVERED}
    # field "a" elsewhere than in the contact is a dimension (the effective weld throat a), not a part
    missing -= {("WeldSpec", "a")}
    assert not missing, f"pole s dílem bez přejmenování v refs.rename_parts: {sorted(missing)}"
