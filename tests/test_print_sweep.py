"""F5 step 2: sweep, advice by the place of failure, thin places with full infill, planar anisotropy of the infills,
print presets. Fast tests without Gmsh and ccx (structured tetrahedral mesh, synthetic results); slow: the sweep,
the advice, the build directions and the protocol through MCP on a small bar.
"""
import json
from types import SimpleNamespace

import anyio
import numpy as np
import pytest

from fea_core import materials as matlib
from fea_core.mesh import Mesh
from fea_core.printing import PrintSpec
from fea_core.printing.infill import FULL, INPLANE_SHEAR, PATTERNS, core, ratios, solid
from fea_core.printing.model import PrintSettings, SweepSpec
from fea_core.printing.variants import VariantOutcome, pareto
from fea_core.study import Study

PLA_NAME = "Tisk PLA Bambu Basic"
PLA = matlib.by_name(PLA_NAME)


def _study(parts=("A",), **spec):
    st = Study(name="t", geometry="x.step")
    st.add_material(matlib.by_name(PLA_NAME), default=True)
    for p in parts:
        st.printing.parts[p] = PrintSpec(p, **spec)
    return st


def box_mesh(boxes, h):
    """Structured mesh of tet10 blocks [(name, (x0, y0, z0), (dx, dy, dz))] – every cube into 6 tetrahedra
    (Kuhn division, conforming), the mid-edge nodes in the order of ccx.
    """
    tets, tpart, names = [], [], []
    kuhn = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    mids = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for pi, (name, x0, d) in enumerate(boxes):
        names.append(name)
        n = [max(1, int(round(d[k] / h))) for k in range(3)]
        g = [np.linspace(x0[k], x0[k] + d[k], n[k] + 1) for k in range(3)]
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    c = np.array([[g[0][i + a], g[1][j + b], g[2][k + e]]
                                  for a in (0, 1) for b in (0, 1) for e in (0, 1)])
                    # cube corners: index = 4a + 2b + e → translate to the Kuhn numbering (bit x = 1, y = 2, z = 4)
                    cube = np.array([c[4 * (q & 1) + 2 * ((q >> 1) & 1) + ((q >> 2) & 1)] for q in range(8)])
                    for t in kuhn:
                        X = cube[list(t)]
                        if np.linalg.det(X[1:] - X[0]) < 0:
                            X = X[[0, 2, 1, 3]]
                        pts = np.vstack([X] + [0.5 * (X[a] + X[b]) for a, b in mids])
                        tets.append(pts)
                        tpart.append(pi)
    P = np.concatenate(tets)
    key = np.round(P, 9)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    conn = inv.reshape(-1, 10)
    return Mesh(uniq, conn.astype(np.int64), np.array(tpart, dtype=np.int64), names)


# ===================================================================== model, JSON
def test_json_roundtrip_new_fields_and_old_files():
    sp = PrintSpec("A", solid_thin=False, thin_width=1.5)
    assert PrintSpec.from_dict(json.loads(json.dumps(sp.to_dict()))) == sp
    old = sp.to_dict()
    old.pop("solid_thin")
    old.pop("thin_width")
    o = PrintSpec.from_dict(old)                         # an older file → full infill of thin places switched on, limit auto
    assert o.solid_thin is True and o.thin_width == 0.0 and o.thin_limit == pytest.approx(2 * o.line_width)
    assert sp.thin_limit == 1.5
    ps = PrintSettings()
    ps.sweep = SweepSpec(["A"], [2, 4], [0.1, 0.3], ["grid", "gyroid"], ["+Z", "-X"], 30)
    back = PrintSettings.from_dict(json.loads(json.dumps(ps.to_dict())))
    assert back.sweep == ps.sweep
    d = ps.to_dict()
    d.pop("sweep")
    assert PrintSettings.from_dict(d).sweep == SweepSpec() and SweepSpec().is_empty()


def test_signature_contains_thin_fields():
    from fea_core.printing.region import print_signature
    st = _study()
    s0 = print_signature(st, ["A"])
    st.printing.parts["A"].infill = 0.4                 # the infill does not change the assignment (the limit of the thin place does not depend on it)
    assert print_signature(st, ["A"]) == s0
    st.printing.parts["A"].solid_thin = False
    s1 = print_signature(st, ["A"])
    assert s1 != s0
    st.printing.parts["A"].thin_width = 2.0
    assert print_signature(st, ["A"]) != s1


# ===================================================================== thin places
def test_thin_plate_solid_thick_block_unchanged():
    from fea_core.loads import element_volumes
    from fea_core.printing.region import assign, thin_core
    st = _study(("Deska", "Blok"), infill=0.15)
    mesh = box_mesh([("Deska", (0, 0, 0), (2.0, 12.0, 8.0)), ("Blok", (10, 0, 0), (6.0, 6.0, 6.0))], 0.5)
    assign(st, mesh)
    vol = element_volumes(mesh)
    sh, th = mesh.cell_data["print_shell"], mesh.cell_data["print_thin"]

    def frac(pi, a):
        el = mesh.tet_part == pi
        return float((vol[el] * a[el]).sum() / vol[el].sum())
    assert frac(0, sh) == pytest.approx(1.0) and frac(0, th) > 0.2      # plate 2 mm: entirely solid
    blok_with = frac(1, sh)
    assert frac(1, th) == 0.0
    for p in ("Deska", "Blok"):
        st.printing.parts[p].solid_thin = False
    assign(st, mesh)
    sh = mesh.cell_data["print_shell"]
    assert frac(1, sh) == blok_with                                      # thick block unchanged
    assert frac(0, sh) < 0.95 and not mesh.cell_data["print_thin"].any()
    # width of the inside of the plate = 2 − 2·0.84 = 0.32 mm; the limit can be given
    el = np.nonzero(mesh.tet_part == 0)[0]
    t, w = thin_core(mesh, el, PrintSpec("Deska"))
    assert np.nanmin(w) == pytest.approx(0.32, abs=1e-6) and t.mean() > 0.9
    t0, _ = thin_core(mesh, el, PrintSpec("Deska", thin_width=0.1))
    assert not t0.any()
    elb = np.nonzero(mesh.tet_part == 1)[0]
    tb, _ = thin_core(mesh, elb, PrintSpec("Blok", thin_width=5.0))     # limit larger than the inside of the block (6 − 1.68)
    assert tb.mean() > 0.5


# ===================================================================== anisotropy of the infills
def test_inplane_anisotropy_average_and_posdef():
    from fea_core.printing.check import _posdef
    s = solid(PLA)
    for pat in PATTERNS:
        for rho in (0.01, 0.05, 0.15, 0.4, 0.8):
            c = core(PLA, pat, rho)
            r = ratios(pat, rho)
            assert c.E1 == pytest.approx(s.E1 * r["E_xy"]) and c.E2v == c.E1     # E in the print axes unchanged
            assert c.Xt == pytest.approx(s.Xt * r["X_xy"]) and c.Sxy == pytest.approx(s.Sxy * r["X_xy"])
            assert _posdef(c), (pat, rho)
            if pat in INPLANE_SHEAR:
                assert c.G12 == pytest.approx(INPLANE_SHEAR[pat] * c.E1)
            else:
                assert c.G12 == pytest.approx(c.E1 / (2 * (1 + c.nu12)))
    # grid ±45°: modulus along the lines (45° to the print axes) ≈ ρ/2·E_s (Gibson–Ashby, a grid loaded in tension)
    c = core(PLA, "grid", 0.2)
    S = np.linalg.inv(c.stiffness())
    a = np.array([1, 1, 0]) / np.sqrt(2)
    eps_dir = np.array([a[0] ** 2, a[1] ** 2, 0, a[0] * a[1], 0, 0])     # σ = 1 in the direction a (Voigt, stress)
    E45 = 1.0 / (eps_dir @ S @ eps_dir)
    assert E45 == pytest.approx(0.5 * 0.2 * s.E1, rel=0.08)
    assert core(PLA, "grid", FULL).G12v is None


# ===================================================================== sweep
def test_sweep_definition_count_names_limits():
    from fea_core.printing.sweep import SweepError, check, count, estimate, norm_dir, normalized, sweep_variants
    st = _study(("A", "B"))
    sw = SweepSpec([], [2, 4, 4], [10, 0.3], ["grid"], ["z", "-x"], 60)
    n = normalized(sw)
    assert n.walls == [2, 4] and n.infill == [0.1, 0.3] and n.build_dirs == ["+Z", "-X"]
    assert count(n) == 8 and count(SweepSpec()) == 0
    vs = sweep_variants(st, sw, ["A", "B"])
    assert len(vs) == 8 and set(vs[0].parts) == {"A", "B"}
    assert vs[0].parts["A"] == {"walls": 2, "infill": 0.1, "pattern": "grid", "build_dir": [0.0, 0.0, 1.0]}
    assert vs[1].parts["A"]["build_dir"] == [-1.0, 0.0, 0.0] and "build -X" in vs[1].name
    assert len({v.name for v in vs}) == 8
    assert check(st, sw, ["A", "B"]) == []
    big = SweepSpec([], list(range(1, 9)), [0.1 * i for i in range(1, 9)], [], [], 60)
    errs = check(st, big, ["A", "B"])
    assert any("64 variants" in e and "at most 60" in e for e in errs)
    assert any("no parameter" in e for e in check(st, SweepSpec(), ["A"]))
    assert any("pattern" in e for e in check(st, SweepSpec(patterns=["neni"]), ["A"]))
    assert any("do not exist" in e for e in check(st, SweepSpec(["C"], [2]), ["A", "B"]))
    with pytest.raises(SweepError):
        norm_dir("w")
    assert norm_dir([0, -1, 0]) == "-Y"
    e = estimate(st, sw, None, None)
    assert e["count"] == 8 and e["runs"] == 9 and e["total_s"] is None
    e = estimate(st, sw, None, 12.0)
    assert e["total_s"] == pytest.approx(108.0) and e["source"] == "measured"
    fake = SimpleNamespace(nodes=np.zeros((71_000, 3)), tets=np.zeros((50_000, 10)))
    e = estimate(st, sw, fake)
    assert e["dof"] == 213_000 and 15 < e["single_s"] < 40 and e["source"].startswith("estimated")


def test_pareto():
    def o(name, k, m, ok=True):
        return VariantOutcome(name, ok=ok, min_safety=k, mass_g=m)
    outs = [o("a", 1.0, 10), o("b", 2.0, 12), o("c", 1.5, 13), o("d", 3.0, 20), o("e", 3.0, 25),
            o("f", 1.0, 10), o("x", None, None, ok=False)]
    assert pareto(outs) == ["a", "b", "d"]              # c and e are dominated, f = the same point as a
    assert pareto([]) == []


# ===================================================================== rada
def _res(parts: dict, S=None, mesh=None):
    """Synthetic result: one state, printing.parts = parts."""
    mesh = mesh if mesh is not None else box_mesh([("A", (0, 0, 0), (1.0, 1.0, 1.0))], 1.0)
    n = len(mesh.nodes)
    S = np.zeros((n, 6)) if S is None else np.broadcast_to(np.asarray(S, float), (n, 6))
    st = SimpleNamespace(name="Stav 1", printing={"parts": parts}, S=S)
    return SimpleNamespace(steps=[st], mesh=mesh)


def _g(k, region, sig, thin=False):
    return {"min_safety": k, "region": region, "stress_local": sig, "at": [0, 0, 0], "thin": thin, "element": 0}


def test_advice_rules():
    from fea_core.printing.advice import advise, contributions
    st = _study(walls=2, infill=0.15, pattern="grid")
    kinds = lambda items: [i["kind"] for i in items]   # noqa: E731
    # perimeter along the line → more perimeters
    it = advise(st, _res({"A": _g(1.0, "perimeter (line axes, transverse, Z)", [30, 1, 0, 0, 0, 0])}))
    assert kinds(it) == ["walls"] and it[0]["suggestion"] == {"walls": 4} and it[0]["orientational"]
    assert "INDICATIVE" in it[0]["note"] and "σ11" in it[0]["reason"]
    # layer bond (σ33) → build direction; the principal stress globally in z → candidates ±X, ±Y (not the current +Z)
    it = advise(st, _res({"A": _g(1.0, "shell", [1, 0, 20, 0, 0, 0])}, S=[0, 0, 20, 0, 0, 0]))
    assert it[0]["kind"] == "build_dir"
    d = it[0]["suggestion"]["build_dir"]
    assert abs(d[2]) < 1e-12 and max(abs(d[0]), abs(d[1])) == 1.0
    # infill, uniaxial tension → higher density (strength ~ρ^1 → k_req/k = 1.5 → 22.5 % → 25 %), + perimeters
    it = advise(st, _res({"A": _g(1.0, "core (infill)", [1.0, 0.05, 0, 0, 0, 0])}))
    infill = [i for i in it if i["kind"] == "infill"][0]
    assert infill["suggestion"]["infill"] == pytest.approx(0.25)
    assert "pattern" not in kinds(it) and "walls" in kinds(it)
    # multiaxial infill → gyroid
    it = advise(st, _res({"A": _g(1.0, "core (infill)", [1.0, 0.9, 0, 0.1, 0, 0])}))
    assert [i for i in it if i["kind"] == "pattern"][0]["suggestion"] == {"pattern": "gyroid"}
    # across the line → perimeters help little (build direction)
    it = advise(st, _res({"A": _g(1.0, "perimeter (line axes, transverse, Z)", [1, 20, 0, 0, 0, 0])}))
    assert it[0]["kind"] == "build_dir" and "across" in it[0]["title"].lower()
    # full layers → more layers
    it = advise(st, _res({"A": _g(1.0, "shell", [20, 0, 0, 0, 0, 0])}))
    assert it[0]["kind"] == "layers" and it[0]["suggestion"] == {"top_layers": 7, "bottom_layers": 5}
    # thin place
    it = advise(st, _res({"A": _g(1.0, "shell", [20, 0, 0, 0, 0, 0], thin=True)}))
    assert "thin" in kinds(it)
    # satisfies / large reserve
    assert kinds(advise(st, _res({"A": _g(2.0, "shell", [5, 0, 0, 0, 0, 0])}))) == ["ok"]
    it = advise(st, _res({"A": _g(5.0, "shell", [5, 0, 0, 0, 0, 0])}))
    assert kinds(it) == ["save"] and it[0]["suggestion"] == {"infill": pytest.approx(0.05)}
    # shares in the failure index: strength by the sign
    m = solid(PLA)
    c = contributions([-m.Xc, 0, 0, 0, 0, 0], m)
    assert c[0] == pytest.approx(1.0)


def test_advice_solid_thin_off():
    from fea_core.printing.advice import advise
    from fea_core.printing.region import assign
    st = _study(("Deska",), solid_thin=False)
    mesh = box_mesh([("Deska", (0, 0, 0), (2.0, 8.0, 6.0))], 0.5)
    assign(st, mesh)
    it = advise(st, _res({"Deska": _g(5.0, "shell", [5, 0, 0, 0, 0, 0])}, mesh=mesh))
    sug = [i for i in it if i["kind"] == "solid_thin"]
    assert sug and sug[0]["suggestion"] == {"solid_thin": True}


# ===================================================================== presets
def test_presets_roundtrip_and_corrupt_file(tmp_path, monkeypatch):
    from fea_core.printing import presets as pz
    monkeypatch.setenv("APPDATA", str(tmp_path))
    names = [p.name for p in pz.list_presets()]
    assert names == [pz.BUILTIN_NAME] and pz.list_presets()[0].builtin
    assert pz.get_preset(pz.BUILTIN_NAME).settings["walls"] == 2
    pz.save_preset("Pevnost", {"walls": 4, "infill": 40, "pattern": "gyroid", "top_layers": 6}, "moje")
    p = pz.get_preset("Pevnost")
    assert p.settings == {"walls": 4, "infill": 0.4, "pattern": "gyroid", "top_layers": 6} and p.note == "moje"
    st = _study(("A", "B"), build_dir=[1, 0, 0], walls=2)
    pz.apply_preset(st, "Pevnost", ["A"])
    a = st.printing.spec("A")
    assert (a.walls, a.infill, a.pattern, a.top_layers, a.build_dir) == (4, 0.4, "gyroid", 6, [1, 0, 0])
    assert st.printing.spec("B").walls == 2
    assert pz.preset_from_spec(a)["walls"] == 4 and "build_dir" not in pz.preset_from_spec(a)
    with pytest.raises(pz.PresetError):
        pz.save_preset(pz.BUILTIN_NAME, {"walls": 3})
    with pytest.raises(pz.PresetError):
        pz.save_preset("x", {"walls": 2.5})
    with pytest.raises(pz.PresetError):
        pz.save_preset("x", {"foo": 1})
    with pytest.raises(pz.PresetError):
        pz.delete_preset(pz.BUILTIN_NAME)
    # a faulty record is skipped and stays in the file
    f = pz.presets_path()
    raw = json.loads(f.read_text(encoding="utf-8"))
    f.write_text(json.dumps(raw + [{"name": "vadná", "settings": {"walls": -1}}, 42]), encoding="utf-8")
    assert [p.name for p in pz.list_presets()] == [pz.BUILTIN_NAME, "Pevnost"]
    pz.save_preset("Lehká", {"infill": 0.1})
    assert len(json.loads(f.read_text(encoding="utf-8"))) == 4
    pz.delete_preset("Pevnost")
    assert [p.name for p in pz.list_presets()] == [pz.BUILTIN_NAME, "Lehká"]
    # unreadable file: the list only the built-in one, saving does not overwrite it
    f.write_text("{nejde", encoding="utf-8")
    assert [p.name for p in pz.list_presets()] == [pz.BUILTIN_NAME]
    with pytest.raises(pz.PresetError):
        pz.save_preset("y", {"walls": 3})
    assert f.read_text(encoding="utf-8") == "{nejde"


# ===================================================================== MCP (without calculation)
def _mcp(fn, tmp):
    from mcp import Client
    from fea_core.mcp.server import build_server

    async def main():
        async with Client(build_server(base_dir=tmp), mode="legacy") as c:
            await fn(c)
    anyio.run(main)


async def _call(c, tool, **args):
    r = await c.call_tool(tool, args)
    txt = r.content[0].text
    return (r.is_error, json.loads(txt) if not r.is_error else txt)


def _bar(tmp_path):
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    p = tmp_path / "bar.step"
    write_step([Part("Tyc", box(0, 0, 0, 40, 10, 10))], p)
    return p


def test_mcp_presets_sweep_estimate_advice(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    step = _bar(tmp_path)

    async def body(c):
        e, d = await _call(c, "fea_study_new", step_path=str(step), workdir=str(tmp_path / "wd"))
        assert not e, d
        e, d = await _call(c, "fea_print_presets")
        assert not e and [p["name"] for p in d["presets"]] == ["3D print standard"]
        e, d = await _call(c, "fea_print_sweep", walls=[2, 3])
        assert e and "printed parts" in d
        e, d = await _call(c, "fea_materials_assign", part="Tyc", material_name=PLA_NAME)
        assert not e, d
        e, d = await _call(c, "fea_print_presets", action="save", name="Moje", settings={"walls": 5, "infill": 30})
        assert not e and d["saved"]["settings"]["infill"] == 0.3, d
        e, d = await _call(c, "fea_print_presets", action="apply", name="Moje")
        assert not e and d["parts"]["Tyc"]["walls"] == 5 and d["parts"]["Tyc"]["infill"] == 0.3, d
        e, d = await _call(c, "fea_print_presets", action="save", name="Z dílu", from_part="Tyc",
                           settings={"walls": 6})
        assert not e and d["saved"]["settings"]["walls"] == 6 and d["saved"]["settings"]["infill"] == 0.3
        e, d = await _call(c, "fea_print_presets", action="delete", name="Z dílu")
        assert not e and d["presets"] == ["3D print standard", "Moje"]
        e, d = await _call(c, "fea_print_set", part="Tyc", settings={"solid_thin": False, "thin_width": 1.2})
        assert not e and d["print"]["solid_thin"] is False and d["print"]["thin_width"] == 1.2, d
        e, d = await _call(c, "fea_print_sweep", walls=[1, 2, 3], infill=[10, 20], patterns=["grid", "gyroid"],
                           run=False)
        assert not e and d["estimate"]["count"] == 12 and d["estimate"]["total_s"] is None, d
        e, d = await _call(c, "fea_print_get")
        assert not e and d["parts"]["Tyc"]["thin_regions"]["solid_thin"] is False
        e, d = await _call(c, "fea_print_sweep", run=False)        # saved definition from the study
        assert not e and d["sweep"]["walls"] == [1, 2, 3] and d["sweep"]["infill"] == [0.1, 0.2]
        e, d = await _call(c, "fea_print_sweep", walls=list(range(1, 9)), infill=[0.1 * i for i in range(1, 9)],
                           run=False)
        assert e and "at most 60" in d
        e, d = await _call(c, "fea_print_advice")
        assert e and "results" in d.lower()
        e, d = await _call(c, "fea_print_rank_build_direction", directions=["+Q"])
        assert e and "build direction" in d.lower()
    _mcp(body, tmp_path)


# ===================================================================== slow: calculation through MCP
@pytest.mark.slow
def test_mcp_sweep_advice_rank_report(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    docx = pytest.importorskip("docx")
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    step = _bar(tmp_path)

    async def body(c):
        for tool, args in [("fea_study_new", dict(step_path=str(step), workdir=str(tmp_path / "wd"))),
                           ("fea_materials_assign", dict(part="Tyc", material_name=PLA_NAME)),
                           ("fea_constraint_add_fixed", dict(name="vetknuti", selection={"x": 0})),
                           ("fea_loadcase_create", dict(name="Ohyb")),
                           ("fea_load_add_force", dict(case_name="Ohyb", name="konec", selection={"x": 40},
                                                       force=[0, 0, -30])),
                           ("fea_mesh_generate", dict(global_size=3.0))]:
            e, d = await _call(c, tool, **args)
            assert not e, (tool, d)
        e, d = await _call(c, "fea_print_sweep", walls=[1, 3], infill=[0.1, 0.4], required_safety=1.5,
                           add_recommended=True)
        assert not e, d
        pts = {p["name"]: p for p in d["points"]}
        assert len(pts) == 5 and all(p["ok"] for p in pts.values())
        ok = [p for p in pts.values() if p["min_safety"] >= 1.5]
        exp = min(ok, key=lambda p: p["mass_g"])["name"] if ok else max(pts.values(),
                                                                        key=lambda p: p["min_safety"])["name"]
        assert d["recommended"] == exp and d["pareto"]
        light = min(pts.values(), key=lambda p: p["mass_g"])
        assert light["name"] in d["pareto"] and light["pareto"]
        assert pts["3 walls · 40 %"]["max_disp"] < pts["1 walls · 10 %"]["max_disp"]
        e, a = await _call(c, "fea_print_advice")
        assert not e and a["advice"] and all(i["orientational"] for i in a["advice"]), a
        e, r = await _call(c, "fea_print_rank_build_direction", directions=["+Z", "+X"])
        assert not e and [x["direction"] for x in r["ranking"]] != [] and r["best"] in ("+Z", "+X"), r
        assert r["ranking"][0]["min_safety"] >= r["ranking"][1]["min_safety"]
        e, d = await _call(c, "fea_print_sweep", walls=[1, 3], infill=[0.1, 0.4])   # again – the results into the protocol
        assert not e
        out = tmp_path / "p.docx"
        e, d = await _call(c, "fea_report_word", output_path=str(out), images=False, language="cs")
        assert not e, d
        doc = docx.Document(str(out))
        text = "\n".join([p.text for p in doc.paragraphs]
                         + [" | ".join(cell.text for cell in row.cells) for t in doc.tables for row in t.rows])
        assert "porovnání variant" in text and "Doporučení:" in text and "Pareto" in text
        assert "Rada podle místa porušení" in text or "Vyhovuje" not in text
        e, d = await _call(c, "fea_print_variant_apply", name=exp) if exp != "Current settings" else (False, {})
        assert not e, d
    _mcp(body, tmp_path)
