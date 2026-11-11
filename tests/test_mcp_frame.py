"""MCP tools of the frame system (fea_frame_*): study without geometry, nodes / members / sections / supports / hinges / arcs,
loads, conversion from the wire model and from the prismatic part, results, assessment EN 1993-1-1, images, protocol.

Fast tests without a calculation; calculations of the members (ccx, ~1–3 s) and the mixed model marked slow.
"""
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from tests.test_mcp import png_size, run  # noqa: E402

E = 210000.0
FRAME_TOOLS = {"fea_frame_section_set", "fea_frame_node_set", "fea_frame_member_set", "fea_frame_support_set",
               "fea_frame_connection_set", "fea_frame_remove", "fea_frame_settings", "fea_frame_list",
               "fea_frame_load_node", "fea_frame_load_member", "fea_frame_from_wires", "fea_frame_from_part",
               "fea_frame_checks_set", "fea_frame_results", "fea_frame_assess", "fea_frame_render"}


@pytest.fixture(scope="module")
def geo(tmp_path_factory):
    """STEP: a wire model (segments L 1000 + an arc), a bar 500×20×10 welded to a cube, a cube 100×100×20."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Pnt
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    d = tmp_path_factory.mktemp("frame_geo")
    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    for p, q in (((0, 0, 0), (0, 0, 1000)), ((0, 0, 1000), (1000, 0, 1000))):
        b.Add(comp, BRepBuilderAPI_MakeEdge(gp_Pnt(*p), gp_Pnt(*q)).Edge())
    arc = GC_MakeArcOfCircle(gp_Pnt(1000, 0, 1000), gp_Pnt(1500, 0, 1500), gp_Pnt(2000, 0, 1000)).Value()
    b.Add(comp, BRepBuilderAPI_MakeEdge(arc).Edge())
    w = STEPControl_Writer()
    w.Transfer(comp, STEPControl_AsIs)
    w.Write(str(d / "wires.step"))
    write_step([Part("Kostka", box(0, 0, 0, 100, 100, 20)), Part("Tyc", box(40, 45, 20, 20, 10, 500))],
               d / "rod.step")
    write_step([Part("Kostka", box(0, 0, 0, 100, 100, 20))], d / "block.step")
    return {"wires": d / "wires.step", "rod": d / "rod.step", "block": d / "block.step"}


async def _cantilever(s, tmp_path, L=2000.0):
    await s.ok("fea_study_new", name="Konzola z prutu", workdir=str(tmp_path / "wd"))
    await s.ok("fea_materials_assign", part="*", material_name="Ocel S235", custom_props={"E": E, "nu": 0.3})
    await s.ok("fea_frame_section_set", name="IPE 200", library="IPE 200")
    await s.ok("fea_frame_node_set", nodes={"A": [0, 0, 0], "B": [L, 0, 0]})
    await s.ok("fea_frame_member_set", name="m", start="A", end="B", section="IPE 200")
    await s.ok("fea_frame_support_set", node="A", dofs="fixed")


# ===================================================================== fast

def test_frame_tools_registered(tmp_path):
    async def body(s, c):
        names = {t.name for t in (await c.list_tools()).tools}
        assert FRAME_TOOLS <= names, FRAME_TOOLS - names
    run(tmp_path, body)


def test_frame_only_study_editing(tmp_path):
    async def body(s, c):
        r = await s.ok("fea_study_new", name="Rám", workdir=str(tmp_path / "wd"), material="Ocel S235")
        assert r["frame_only"] and r["parts"] == []
        st = await s.ok("fea_study_status")
        assert st["parts"] == [] and st["step_file"] is None
        sec = await s.ok("fea_frame_section_set", name="R", shape="RECT", dims={"b": 20, "h": 40})
        assert sec["props"]["A_mm2"] == 800
        assert "library" in await s.err("fea_frame_section_set", name="X")
        assert "dimension" in await s.err("fea_frame_section_set", name="X", shape="RECT", dims={"b": 1})
        await s.ok("fea_frame_section_set", name="CHS", library="CHS 76.1x3.6")
        await s.ok("fea_frame_node_set", nodes=[{"name": "A", "xyz": [0, 0, 0]}, {"name": "B", "xyz": [0, 0, 3000]},
                                               {"name": "C", "xyz": [4000, 0, 3000]}, {"name": "D", "xyz": [4000, 0, 0]}])
        assert "start, end and section" in await s.err("fea_frame_member_set", name="x", start="A")
        assert "does not exist" in await s.err("fea_frame_member_set", name="x", start="A", end="Q", section="R")
        await s.ok("fea_frame_member_set", name="sloup L", start="A", end="B", section="R", up=[1, 0, 0])
        m = await s.ok("fea_frame_member_set", name="příčel", start="B", end="C", section="R",
                       releases_end=["My", "Mz"])
        assert m["member"]["length_mm"] == 4000 and m["member"]["releases_end"] == "My, Mz"
        assert "unknown component" in await s.err("fea_frame_member_set", name="příčel", releases_start=["Mx"])
        await s.ok("fea_frame_member_set", name="sloup P", start="D", end="C", section="R", up=[1, 0, 0])
        arc = await s.ok("fea_frame_member_set", name="oblouk", start="B", end="C", section="R",
                         through=[2000, 0, 4000], up="normal")
        assert abs(arc["member"]["length_mm"] - 2 * 2500 * math.asin(2000 / 2500)) < 5     # R = 2500, through the vertex
        await s.ok("fea_frame_member_set", name="diag", start="A", end="C", section="CHS", kind="truss")
        assert "straight line" in await s.err("fea_frame_member_set", name="oblouk", through=[2000, 0, 3000])
        assert (await s.ok("fea_frame_list"))["members"][-2]["through_mm"] == [2000, 0, 4000]   # change reverted
        # supports and readiness (one hinged support: rotation free → rigid motions)
        await s.ok("fea_frame_support_set", node="A", dofs="pinned")
        await s.ok("fea_frame_load_member", case_name="Sníh", member="příčel", q=[0, 0, -5])
        rd = await s.ok("fea_check_readiness")
        assert not rd["ready"] and "rigid-body motions" in " ".join(rd["errors"]), rd
        sup = await s.ok("fea_frame_support_set", node="D", dofs=["ux", "uy", "uz"], stiffness={"rz": 1e6})
        assert sup["support"]["stiffness"] == {"rz": 1e6}
        await s.ok("fea_frame_support_set", node="A", dofs="fixed")
        rd = await s.ok("fea_check_readiness")
        assert rd["ready"], rd["errors"]
        ld = await s.ok("fea_frame_load_node", case_name="Vítr", node="B", force=[2000, 0, 0], moment=[0, 1e5, 0])
        assert ld["case_created"] and "M = [0, 100000, 0]" in ld["load"]
        lm = await s.ok("fea_frame_load_member", case_name="Sníh", member="příčel", q=[0, 0, -5], q_end=[0, 0, -10])
        assert lm["replaced"] and abs(lm["resultant_N"][2] + 30000) < 1e-6
        assert "axes" in await s.err("fea_frame_load_member", case_name="Sníh", member="příčel", q=[0, 0, 1],
                                     axes="xx")
        lst = await s.ok("fea_frame_list")
        assert len(lst["members"]) == 5 and lst["sections"]["R"]["A_mm2"] == 800 and len(lst["loads"]) == 2
        assert {x["node"] for x in lst["supports"]} == {"A", "D"}
        ch = await s.ok("fea_frame_checks_set", member="*", settings={"L_LT": 1000, "lt_restrained": False})
        assert ch["checks"]["sloup L"]["L_LT"] == 1000
        assert "Unknown assessment parameters" in await s.err("fea_frame_checks_set", member="diag", settings={"x": 1})
        assert "used by members" in await s.err("fea_frame_remove", kind="node", names="B")
        assert "used by members" in await s.err("fea_frame_remove", kind="section", names="R")
        rm = await s.ok("fea_frame_remove", kind="member", names=["příčel"])
        assert rm["removed"] == ["příčel"]
        lst = await s.ok("fea_frame_list")
        assert not any(l["kind"] == "frame_member" for l in lst["loads"]) and "příčel" not in lst["checks"]
        st_ = await s.ok("fea_frame_settings", segments_straight=10, steel_grade="s355")
        assert st_["settings"]["segments_straight"] == 10 and st_["settings"]["steel_grade"] == "S355"
        img = tmp_path / "frame_model.png"
        r = await s.raw("fea_frame_render", quantity="model", output_path=str(img), view="front")
        assert not r.is_error and [x.type for x in r.content] == ["text", "image"] and png_size(img) == (1280, 720)
        r = await s.raw("fea_model_render_image", output_path=str(tmp_path / "m2.png"), return_image=False)
        assert not r.is_error
        assert "results" in (await s.err("fea_frame_results")).lower()
        assert "connection to a solid" in await s.err("fea_frame_connection_set", node="A", selection={"z": 0})
        # supports and reopening without geometry
        await s.ok("fea_study_save", path=str(tmp_path / "ram.json"))
        await s.ok("fea_study_save", path=str(tmp_path / "ram.fea"))
        o = await s.ok("fea_study_open", path=str(tmp_path / "ram.fea"), workdir=str(tmp_path / "w2"))
        assert o["frame"]["members"] == 4 and o["step_file"] is None
        o = await s.ok("fea_study_open", path=str(tmp_path / "ram.json"), workdir=str(tmp_path / "w3"))
        assert o["frame"]["members"] == 4
    run(tmp_path, body)


def test_frame_from_wires_and_part(tmp_path, geo):
    async def body(s, c):
        await s.ok("fea_study_new", name="Dráty", workdir=str(tmp_path / "wd"), material="Ocel S235")
        await s.ok("fea_frame_section_set", name="T", library="CHS 76.1x3.6")
        r = await s.ok("fea_frame_from_wires", section="T", step_path=str(geo["wires"]))
        assert r["edges"] == {"line": 2, "circle": 1} and len(r["members_created"]) == 3 and r["nodes_created"] == 4
        lst = await s.ok("fea_frame_list")
        arcs = [m for m in lst["members"] if "through_mm" in m]
        assert len(arcs) == 1 and abs(arcs[0]["length_mm"] - math.pi * 500) < 2
        assert "not found" in await s.err("fea_frame_from_wires", section="T")         # the study has no STEP
        # prismatic part → member + connection with the cube
        await s.ok("fea_study_new", step_path=str(geo["rod"]), workdir=str(tmp_path / "wd2"), material="Ocel S235")
        d = await s.ok("fea_frame_from_part", part="Tyc")
        assert abs(d["length_mm"] - 500) < 1e-6 and abs(d["A_mm2"] - 200) < 1e-3 and d["connections"]
        st = await s.ok("fea_study_status")
        assert "Tyc" in st["ignored_parts"] and st["frame"]["connections"] == 1
    run(tmp_path, body)


# ===================================================================== slow (ccx)

@pytest.mark.slow
def test_e2e_frame_cantilever(tmp_path):
    """Cantilever IPE 200 L = 2 m, force 1 kN: reaction, M = F·L, deflection ≈ FL³/3EI, assessment, images, frequencies,
    protocol (a study without geometry).
    """
    import docx

    async def body(s, c):
        await _cantilever(s, tmp_path)
        await s.ok("fea_frame_load_node", case_name="F", node="B", force=[0, 0, -1000])
        await s.ok("fea_modal_set", modes=2)
        assert (await s.ok("fea_check_readiness"))["ready"]
        r = await s.ok("fea_solve_run", timeout_s=300)
        assert r["steps"][0]["equilibrium"]["ok"] and "fea_frame_results" in r["next"]
        fr = await s.ok("fea_frame_results", step="F", member="m", points=5)
        assert abs(fr["reactions"]["A"]["force_N"][2] - 1000) < 0.1
        assert abs(abs(fr["members"]["m"]["My"]["value"]) - 2e6) < 200 and fr["members"]["m"]["My"]["s_mm"] == 0
        Iy = 19.432e6
        w = 1000 * 2000 ** 3 / (3 * E * Iy)
        assert abs(fr["nodes"]["B"]["u_mm"][2] + w) / w < 0.03
        mc = fr["member_course"]
        assert len(mc["s_mm"]) == 5 and abs(mc["My"][-1]) < 200          # end of the cantilever M ≈ 0
        summ = await s.ok("fea_results_summary", step=None)
        assert summ["steps"][0]["frame"]["max_sigma_eq_MPa"]["member"] == "m"
        txt = (await c.read_resource("fea://results/summary")).contents[0].text
        assert "node reaction 'A'" in txt
        await s.ok("fea_frame_checks_set", member="m", settings={"L_LT": 2000, "C1": 1.0})
        a = await s.ok("fea_frame_assess")
        assert a["governing_member"] == "m" and 0 < a["max_utilization"] < 1 and a["detail"]["rows"]
        for q in ("deformed", "My", "Vz", "sigma", "utilization"):
            p = tmp_path / f"f_{q}.png"
            rr = await s.raw("fea_frame_render", quantity=q, output_path=str(p), view="front", return_image=False)
            assert not rr.is_error, rr.content[0].text
            assert png_size(p) == (1280, 720)
        meta = json.loads((await s.raw("fea_frame_render", quantity="My", output_path=str(tmp_path / "x.png"),
                                       return_image=False)).content[0].text)
        assert abs(abs(meta["extreme"]["value"]) - 2e6) < 200
        assert "quantity" in await s.err("fea_frame_render", quantity="xx")
        md = (await s.ok("fea_results_modes"))["mode_sets"][0]
        f_eb = 1.875104 ** 2 / (2 * math.pi * 2000 ** 2) * math.sqrt(E * 1.4237e6 / (7.85e-9 * 2848.4))
        assert abs(md["frequencies_Hz"][0] - f_eb) / f_eb < 0.03          # 1st mode – bending about the weak axis
        rr = await s.raw("fea_results_render_mode", mode=1, output_path=str(tmp_path / "mode.png"), return_image=False)
        assert not rr.is_error
        assert "solid bodies" in await s.err("fea_results_render_image")
        assert "solid bodies" in await s.err("fea_results_probe", point=[0, 0, 0])
        rep = await s.ok("fea_report_word", output_path=str(tmp_path / "konzola.docx"), language="en")
        assert rep["figures"] >= 5
        await s.ok("fea_study_save", path=str(tmp_path / "k.fea"))
        await s.ok("fea_study_open", path=str(tmp_path / "k.fea"), workdir=str(tmp_path / "w2"))
        fr2 = await s.ok("fea_frame_results", step=0)
        assert abs(fr2["reactions"]["A"]["force_N"][2] - 1000) < 0.1
    run(tmp_path, body)
    d = docx.Document(str(tmp_path / "konzola.docx"))
    text = "\n".join(p.text for p in d.paragraphs) + "\n".join(c.text for t in d.tables for r in t.rows for c in r.cells)
    heads = [p.text for p in d.paragraphs if p.style.name.startswith(("Heading", "Title"))]
    assert heads[0].startswith("Frame analysis")
    assert "1.3 Frame" in heads and any(h.startswith("4 Member check per EN 1993-1-1") for h in heads)
    assert "IPE 200" in text and "Bending moment My diagram" in text and "Most utilised member “m”" in text
    assert "Natural frequencies and effective mass fractions" in text and "design loads" in text


@pytest.mark.slow
def test_e2e_mixed_frame_and_solid(tmp_path, geo):
    """Cube 100×100×20 clamped at the bottom + a column of a member (RECT 20×20, 500 mm) rigidly connected to the top face, force
    1 kN horizontal at the top: reaction of the clamping = −F, force in the connection = F, moment F·L.
    """
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(geo["block"]), workdir=str(tmp_path / "wd"), material="Ocel S235")
        await s.ok("fea_constraint_add_fixed", name="dno", selection={"z": 0})
        await s.ok("fea_frame_section_set", name="R20", shape="RECT", dims={"b": 20, "h": 20})
        await s.ok("fea_frame_node_set", nodes={"A": [50, 50, 20], "B": [50, 50, 520]})
        await s.ok("fea_frame_member_set", name="sloup", start="A", end="B", section="R20", up=[1, 0, 0])
        await s.ok("fea_frame_connection_set", node="A", selection={"z": 20})
        await s.ok("fea_frame_load_node", case_name="F", node="B", force=[1000, 0, 0])
        assert (await s.ok("fea_check_readiness"))["ready"]
        r = await s.raw("fea_model_render_image", output_path=str(tmp_path / "mix.png"), view="iso", return_image=False)
        assert not r.is_error
        await s.ok("fea_mesh_generate", global_size=10)
        res = await s.ok("fea_solve_run", timeout_s=600, backend="ccx")
        st = res["steps"][0]
        assert st["equilibrium"]["ok"] and abs(st["reactions"]["dno"]["force_N"][0] + 1000) < 0.5
        fr = await s.ok("fea_frame_results", step=0)
        cf = fr["connection_forces"]["A"]
        assert abs(abs(cf["force_N"][0]) - 1000) < 0.5 and abs(abs(cf["moment_Nmm"][1]) - 5e5) < 500
        rr = await s.raw("fea_frame_render", quantity="deformed", output_path=str(tmp_path / "mixd.png"),
                         return_image=False)
        assert not rr.is_error
        rep = await s.ok("fea_report_word", output_path=str(tmp_path / "mix.docx"), images=False)
        assert rep["with_results"]
    run(tmp_path, body)


def test_e2e_frame_temperature_and_buckling(tmp_path):
    """Own solver through MCP: temperature of the restrained bar (N = −EAαΔT), buckling + frequencies (shapes on the subdivision for the modes), settings
    of the solver and a protocol with the description of the temperature load.
    """
    async def body(s, c):
        await _cantilever(s, tmp_path)
        await s.ok("fea_frame_support_set", node="B", dofs="fixed")
        t = await s.ok("fea_frame_load_temperature", case_name="T", dT=40.0)
        assert "ΔT = 40" in t["load"] and not t["replaced"]
        assert "does not exist" in await s.err("fea_frame_load_temperature", case_name="T", dT=1.0, members=["x"])
        await s.ok("fea_modal_set", modes=2)
        r = await s.ok("fea_solve_run", timeout_s=300)
        assert r["steps"][0]["equilibrium"]["ok"]
        fr = await s.ok("fea_frame_results", step="T", member="m", points=3)
        assert abs(fr["members"]["m"]["N"]["value"] + E * 2848.41 * 12e-6 * 40) < 0.5
        lst = await s.ok("fea_frame_list")
        assert any(l["kind"] == "frame_temperature" for l in lst["loads"]) and lst["settings"]["solver"] == "timoshenko"
        md = (await s.ok("fea_results_modes"))["mode_sets"][0]
        assert len(md["frequencies_Hz"]) == 2
        rr = await s.raw("fea_results_render_mode", mode=1, output_path=str(tmp_path / "mode.png"), return_image=False)
        assert not rr.is_error, rr.content[0].text
        st = await s.ok("fea_frame_settings", segments_modal=4, buckling_torsion=True)
        assert st["settings"]["segments_modal"] == 4 and st["settings"]["buckling_torsion"]
        assert "solver must" in await s.err("fea_frame_settings", solver="x")
        rep = await s.ok("fea_report_word", output_path=str(tmp_path / "t.docx"))
        assert rep["figures"] >= 1
        await s.ok("fea_frame_settings", solver="ccx_u1")
        assert "built-in solver" in await s.err("fea_solve_run", timeout_s=60)
    run(tmp_path, body)
