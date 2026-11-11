"""Fusion add-in: sending only the selected parts / subassemblies (the plan of hiding and restoring the visibility) on a fake assembly tree."""
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

FUSION = Path(__file__).resolve().parent.parent / "adapters" / "fusion" / "FEAStudio" / "FEAStudio.py"


@pytest.fixture(scope="module")
def fus():
    loader = importlib.machinery.SourceFileLoader("fea_fusion_scope", str(FUSION))
    spec = importlib.util.spec_from_loader("fea_fusion_scope", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class Obj:
    def __init__(self, name, typ, ctx=None):
        self.name, self.objectType, self.assemblyContext = name, "adsk::fusion::" + typ, ctx
        self.entityToken = typ + ":" + name
        self.isLightBulbOn = True
        self.bRepBodies, self.childOccurrences, self.occurrences = [], [], []

    @property
    def body(self):                       # face / edge → body
        return self._body


def make_tree():
    """root: body r1; occurrences A (bodies a1, a2; sub-occurrence A1 with x1), B (b1), C (c1)."""
    root = Obj("Root", "Component")
    r1 = Obj("r1", "BRepBody")
    A, B, C = Obj("A:1", "Occurrence"), Obj("B:1", "Occurrence"), Obj("C:1", "Occurrence")
    a1, a2 = Obj("a1", "BRepBody", A), Obj("a2", "BRepBody", A)
    A1 = Obj("A1:1", "Occurrence", A)
    x1 = Obj("x1", "BRepBody", A1)
    b1, c1 = Obj("b1", "BRepBody", B), Obj("c1", "BRepBody", C)
    root.bRepBodies = [r1]
    root.occurrences = [A, B, C]
    A.bRepBodies, A.childOccurrences = [a1, a2], [A1]
    A1.bRepBodies = [x1]
    B.bRepBodies, C.bRepBodies = [b1], [c1]
    everything = [r1, A, B, C, a1, a2, A1, x1, b1, c1]
    return root, {o.name: o for o in everything}, everything


class Design:
    def __init__(self, root):
        self.rootComponent = root


def names(lst):
    return sorted(o.name for o in lst)


def test_select_subassembly(fus):
    root, o, _ = make_tree()
    occ, body, items = fus.plan_isolation(root, [o["A:1"]])
    assert names(occ) == ["B:1", "C:1"] and names(body) == ["r1"] and items == ["A:1"]


def test_select_nested_subassembly_hides_siblings_in_parent(fus):
    root, o, _ = make_tree()
    occ, body, _ = fus.plan_isolation(root, [o["A1:1"]])
    assert names(occ) == ["B:1", "C:1"]
    assert names(body) == ["a1", "a2", "r1"]            # parent A stays only as the context


def test_select_single_body_in_component(fus):
    root, o, _ = make_tree()
    occ, body, _ = fus.plan_isolation(root, [o["a2"]])
    assert names(occ) == ["A1:1", "B:1", "C:1"] and names(body) == ["a1", "r1"]


def test_select_body_of_other_occurrence_and_root_body(fus):
    root, o, _ = make_tree()
    occ, body, _ = fus.plan_isolation(root, [o["b1"], o["r1"]])
    assert names(occ) == ["A:1", "C:1"] and body == []


def test_select_two_items_and_face(fus):
    root, o, _ = make_tree()
    face = Obj("f", "BRepFace")
    face._body = o["c1"]
    occ, body, items = fus.plan_isolation(root, [o["B:1"], face])
    assert names(occ) == ["A:1"] and body == [o["r1"]] and sorted(items) == ["B:1", "c1"]


def test_root_component_means_whole_design(fus):
    root, _, _ = make_tree()
    assert fus.plan_isolation(root, [root]) is None


def test_component_definition_and_empty_selection_rejected(fus):
    root, _, _ = make_tree()
    with pytest.raises(RuntimeError):
        fus.plan_isolation(root, [Obj("Def", "Component")])
    with pytest.raises(RuntimeError):
        fus.plan_isolation(root, [Obj("x", "SketchPoint")])


def test_isolate_hides_and_restores(fus):
    root, o, everything = make_tree()
    o["C:1"].isLightBulbOn = False                       # already hidden before the export
    with fus.isolate(Design(root), [o["A:1"]]) as scope:
        assert scope == {"mode": "selection", "items": ["A:1"]}
        assert not o["B:1"].isLightBulbOn and not o["r1"].isLightBulbOn
        assert o["A:1"].isLightBulbOn and o["a1"].isLightBulbOn and o["x1"].isLightBulbOn
    assert [e.name for e in everything if not e.isLightBulbOn] == ["C:1"]     # only the originally hidden one stays hidden


def test_isolate_restores_after_error(fus):
    root, o, everything = make_tree()
    with pytest.raises(ValueError):
        with fus.isolate(Design(root), [o["b1"]]):
            raise ValueError("export selhal")
    assert all(e.isLightBulbOn for e in everything)


def test_isolate_noop_without_selection(fus):
    root, o, everything = make_tree()
    for sel in (None, [], [root]):
        with fus.isolate(Design(root), sel) as scope:
            assert scope is None
            assert all(e.isLightBulbOn for e in everything)


def test_scope_title(fus):
    assert fus.scope_title("Držák", {"items": ["Deska:1"]}) == "Držák – Deska:1"
    assert fus.scope_title("Držák", {"items": ["Deska:1", "b", "c"]}) == "Držák – Deska:1 + 2 more"
