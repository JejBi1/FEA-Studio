"""Variants of the internal structure of 3D printing (F5): the mesh independent of the print settings, calculation of the variants on one mesh.

Fast tests without Gmsh and ccx (model, validation, signatures, ensure on a synthetic mesh, MCP definitions);
slow: variants of the cantilever (one mesh, bit-identical agreement with the pure assignment, monotonicity), the MCP calculation and taking over.
"""
import copy
import json
from pathlib import Path

import anyio
import numpy as np
import pytest

from fea_core import materials as matlib
from fea_core.mesh import Mesh
from fea_core.printing import PrintSpec, PrintVariant
from fea_core.printing.model import PrintSettings
from fea_core.printing.region import ensure, print_signature
from fea_core.printing.variants import (BASE_NAME, VariantOutcome, apply_variant, effective, normalize, recommend,
                                        validate)
from fea_core.study import Study

PLA = "Tisk PLA Bambu Basic"


def _study(parts=("A",)):
    st = Study(name="t", geometry="x.step")
    st.add_material(matlib.by_name(PLA), default=True)
    for p in parts:
        st.printing.parts[p] = PrintSpec(p)
    return st


def _tet_mesh():
    x = np.array([[0, 0, 0], [8, 0, 0], [0, 8, 0], [0, 0, 8]], float)
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    nodes = np.vstack([x] + [0.5 * (x[a] + x[b]) for a, b in mids])
    return Mesh(nodes, np.arange(10)[None], np.zeros(1, dtype=np.int64), ["A"])


# ===================================================================== model a validace

def test_json_roundtrip_and_old_files():
    ps = PrintSettings()
    ps.variants = [PrintVariant("V1", {"A": {"walls": 4, "infill": 0.4}}, "pozn.")]
    ps.required_safety = 2.5
    d = json.loads(json.dumps(ps.to_dict()))
    back = PrintSettings.from_dict(d)
    assert back.variants[0].to_dict() == ps.variants[0].to_dict() and back.required_safety == 2.5
    d.pop("variants")
    d.pop("required_safety")                    # a file from an older version
    old = PrintSettings.from_dict(d)
    assert old.variants == [] and old.required_safety == 1.5


def test_normalize_validate_apply():
    st = _study()
    assert normalize({"infill": 40, "walls": 3.0, "build_dir": (1, 0, 0)}) == \
        {"infill": 0.4, "walls": 3, "build_dir": [1.0, 0.0, 0.0]}
    good = PrintVariant("4 perimetry", {"A": {"walls": 4, "infill": 40, "pattern": "gyroid"}})
    assert validate(st, good, ["A"]) == []
    bad = PrintVariant(BASE_NAME, {"B": {"walls": 2, "foo": 1, "pattern": "neni", "material": "Ocel"}})
    errs = "\n".join(validate(st, bad, ["A"]))
    for s in ("reserved", "part 'B' does not exist", "'foo'", "'neni'", "is not in the study"):
        assert s in errs, s
    v2 = apply_variant(st, good)
    sp = v2.printing.spec("A")
    assert (sp.walls, sp.infill, sp.pattern, sp.top_layers) == (4, 0.4, "gyroid", 5)
    assert st.printing.spec("A").walls == 2 and v2.printing.variants == []      # the original unchanged
    eff = effective(st, good, ["A", "B"])
    assert list(eff) == ["A", "B"] and eff["A"]["material"] == PLA and eff["A"]["walls"] == 4
    assert eff["B"]["walls"] == 2                       # a part without an override = the setting of the study


def test_variant_material_override():
    st = _study()
    m2 = matlib.by_name("Tisk PETG (orientačně)")
    st.materials[m2.name] = m2
    v = PrintVariant("PETG", {"A": {"material": m2.name}})
    assert validate(st, v, ["A"]) == []
    st.printing.variants.append(v)
    assert apply_variant(st, v).material_of("A").name == m2.name
    steel = matlib.by_name("Ocel S235")
    st.materials[steel.name] = steel
    assert "is not a printed material" in validate(st, PrintVariant("x", {"A": {"material": steel.name}}), ["A"])[0]


def test_recommend():
    def o(name, k, m, ok=True):
        return VariantOutcome(name, ok=ok, min_safety=k, mass_g=m)
    outs = [o(BASE_NAME, 1.2, 10), o("lehká", 1.6, 8), o("těžká", 3.0, 15), o("chyba", None, None, ok=False)]
    assert recommend(outs, 1.5)[0] == "lehká"
    name, why = recommend(outs, 5.0)
    assert name == "těžká" and "No variant satisfies" in why
    assert recommend([o("x", None, None, ok=False)], 1.0)[0] is None


# ===================================================================== mesh independent of the printing

def test_print_signature_fields():
    st = _study()
    s0 = print_signature(st, ["A"])
    st.printing.parts["A"].infill = 0.5
    st.printing.parts["A"].pattern = "gyroid"
    assert print_signature(st, ["A"]) == s0              # the infill changes only the material of the core, not the assignment
    st.printing.parts["A"].walls = 3
    assert print_signature(st, ["A"]) != s0


def test_ensure_recomputes_only_on_change(monkeypatch):
    import fea_core.printing.region as region
    st = _study()
    mesh = _tet_mesh()
    calls = []
    real = region.assign
    monkeypatch.setattr(region, "assign", lambda s, m: calls.append(1) or real(s, m))
    ensure(st, mesh)
    f0 = mesh.cell_data["print_shell"].copy()
    ensure(st, mesh)
    assert len(calls) == 1
    st.printing.parts["A"].infill = 0.6
    ensure(st, mesh)
    assert len(calls) == 1
    st.printing.parts["A"].enabled = False                # the whole part solid
    ensure(st, mesh)
    assert len(calls) == 2 and np.all(mesh.cell_data["print_shell"] == 1.0)
    st.printing.parts["A"].enabled = True
    ensure(st, mesh)
    assert np.array_equal(mesh.cell_data["print_shell"], f0)
    steel = matlib.by_name("Ocel S235")                    # unprinted material → the print assignment removed
    st.materials[steel.name] = steel
    st.part_materials["A"] = steel.name
    ensure(st, mesh)
    assert "print_shell" not in mesh.cell_data


def test_session_mesh_signature_ignores_printing(tmp_path):
    from fea_core.mcp.session import Session
    s = Session(tmp_path)
    s.study = _study()
    sig = s.mesh_signature()
    s.study.printing.parts["A"].walls = 6
    s.study.printing.variants.append(PrintVariant("v", {"A": {"infill": 0.3}}))
    s.study.materials[PLA] = copy.copy(s.study.materials[PLA])
    s.study.materials[PLA].E_z = 1000.0
    assert s.mesh_signature() == sig
    s.study.mesh.size = 1.234
    assert s.mesh_signature() != sig


# ===================================================================== MCP (definition, without calculation)

def _mcp(fn):
    from mcp import Client
    from fea_core.mcp.server import build_server

    async def main():
        async with Client(build_server(base_dir=fn.tmp), mode="legacy") as c:
            await fn(c)
    anyio.run(main)


async def _call(c, tool, **args):
    r = await c.call_tool(tool, args)
    txt = r.content[0].text
    return (r.is_error, json.loads(txt) if not r.is_error else txt)


def _beam(tmp_path):
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    p = tmp_path / "beam.step"
    write_step([Part("Konzola", box(0, 0, 0, 60, 10, 10))], p)
    return p


def test_mcp_variant_definitions(tmp_path):
    pytest.importorskip("mcp")
    step = _beam(tmp_path)

    async def body(c):
        e, d = await _call(c, "fea_study_new", step_path=str(step), workdir=str(tmp_path / "wd"))
        assert not e, d
        e, d = await _call(c, "fea_print_variant_set", name="x", settings={"walls": 4})
        assert e and "printed parts" in d
        e, d = await _call(c, "fea_materials_assign", part="Konzola", material_name=PLA)
        assert not e, d
        e, d = await _call(c, "fea_print_variant_set", name="4 perimetry 40 %", settings={"walls": 4, "infill": 40})
        assert not e, d
        assert d["settings"]["Konzola"]["walls"] == 4 and d["settings"]["Konzola"]["infill"] == 0.4
        e, d = await _call(c, "fea_print_variant_set", name="gyroid", copy_from="4 perimetry 40 %",
                           part_settings={"Konzola": {"pattern": "gyroid"}})
        assert not e and d["overrides"]["Konzola"] == {"walls": 4, "infill": 0.4, "pattern": "gyroid"}, d
        e, d = await _call(c, "fea_print_variant_set", name="zle", settings={"pattern": "neni"})
        assert e and "neni" in d
        e, d = await _call(c, "fea_print_variant_list")
        assert [v["name"] for v in d["variants"]] == ["4 perimetry 40 %", "gyroid"]
        assert d["study_settings"]["Konzola"]["walls"] == 2 and d["required_safety"] == 1.5
        e, d = await _call(c, "fea_print_variant_remove", name="gyroid")
        assert not e and d["variants"] == ["4 perimetry 40 %"]
        e, d = await _call(c, "fea_print_variants_run")
        assert e and "no load cases" in d.lower()
    body.tmp = tmp_path
    _mcp(body)


# ===================================================================== slow: calculation

def _cantilever(wd: Path):
    from verification.cases import _study as vstudy, box
    from fea_core.geometry import Part
    from fea_core.selectors import find_faces
    from fea_core.study import Fixed, Force, LoadCase, MeshSettings
    st, parts = vstudy("Varianty tisku", wd / "model.step", [Part("Konzola", box(0, 0, 0, 60, 10, 10))],
                       MeshSettings(size=2.0))
    st.add_material(matlib.by_name(PLA), default=True)
    st.materials.pop("Ocel S235")
    st.printing.parts["Konzola"] = PrintSpec("Konzola", walls=2, infill=0.15)
    st.add_selection("vetknuti", find_faces(parts, x=0))
    st.add_selection("konec", find_faces(parts, x=60))
    st.constraints.append(Fixed("vetknuti"))
    st.load_cases.append(LoadCase("Ohyb", [Force("konec", [0.0, 0.0, -20.0])]))
    st.printing.variants = [PrintVariant("lehká", {"Konzola": {"walls": 1, "infill": 0.05}}),
                            PrintVariant("tuhá", {"Konzola": {"walls": 5, "infill": 0.5}}),
                            PrintVariant("plná", {"Konzola": {"enabled": False}})]
    return st, parts


@pytest.mark.slow
def test_variants_one_mesh_and_equivalence(tmp_path, monkeypatch):
    import fea_core.solve as solve
    from fea_core.printing.variants import run_variants
    st, parts = _cantilever(tmp_path)
    calls = []
    real = solve.generate_mesh
    monkeypatch.setattr(solve, "generate_mesh", lambda *a, **k: calls.append(1) or real(*a, **k))
    vr = run_variants(st, tmp_path / "var", parts=parts)
    assert len(calls) == 1 and vr.mesh_info["meshed"]
    o = {x.name: x for x in vr.outcomes}
    assert all(x.ok for x in vr.outcomes), [x.error for x in vr.outcomes]
    order = [BASE_NAME, "lehká", "tuhá", "plná"]
    order_by_mass = sorted(order, key=lambda n: o[n].mass_g)
    assert order_by_mass == ["lehká", BASE_NAME, "tuhá", "plná"]
    disp = [o[n].max_disp for n in order_by_mass]
    assert all(a > b for a, b in zip(disp, disp[1:]))           # more material → stiffer
    assert o["tuhá"].min_safety > o["lehká"].min_safety
    # bit-identical agreement with the pure assignment of the print on the same mesh
    tuha = o["tuhá"].results
    m = copy.deepcopy(tuha.mesh)
    m.cell_data = {}
    m.report = {k: v for k, v in m.report.items() if k not in ("print_sig", "print_warnings")}
    ref = solve.run_study(apply_variant(st, st.printing.variants[1]), tmp_path / "ref", parts=parts, mesh=m)
    assert len(calls) == 1
    assert (tmp_path / "ref/job.inp").read_bytes() == (Path(o["tuhá"].workdir) / "job.inp").read_bytes()
    assert np.array_equal(ref.steps[0].U, tuha.steps[0].U)
    assert np.array_equal(ref.steps[0].fields["print_safety"], tuha.steps[0].fields["print_safety"])
    # recommendation = the lightest with k ≥ the requirement
    st.printing.required_safety = 0.5 * (o["lehká"].min_safety + o[BASE_NAME].min_safety)
    assert recommend(vr.outcomes, st.printing.required_safety)[0] == BASE_NAME


@pytest.mark.slow
def test_mcp_variants_run_and_apply(tmp_path):
    pytest.importorskip("mcp")
    st, _ = _cantilever(tmp_path)
    st.printing.variants = st.printing.variants[1:2]          # only "stiff"
    sfile = tmp_path / "studie.json"
    st.geometry = str(tmp_path / "model.step")
    sfile.write_text(json.dumps(st.to_dict()), encoding="utf-8")

    async def body(c):
        e, d = await _call(c, "fea_study_open", path=str(sfile), workdir=str(tmp_path / "wd"))
        assert not e, d
        e, d = await _call(c, "fea_print_variants_run", required_safety=1.0)
        assert not e, d
        names = [r["variant"] for r in d["results"]]
        assert names == [BASE_NAME, "tuhá"] and all(r["ok"] for r in d["results"])
        e, s = await _call(c, "fea_results_summary")
        assert not e                                               # results of the study = the setting of the study
        e, d = await _call(c, "fea_print_variant_apply", name="tuhá")
        assert not e and d["results_reused"] and d["print"]["Konzola"]["walls"] == 5, d
        e, r = await _call(c, "fea_print_results")
        assert not e and r["parts_info"]["parts"]["Konzola"]["spec"]["walls"] == 5
        # change of the perimeters after the calculation: the calculation takes over the mesh (without Gmsh)
        e, d = await _call(c, "fea_print_set", part="Konzola", settings={"walls": 3})
        e, d = await _call(c, "fea_solve_run")
        assert not e and "mesh_generated" not in d, d
    body.tmp = tmp_path
    _mcp(body)
