"""MCP: force on an edge and at a point, local / cylindrical coordinate system, hydrostatic and linear pressure, rotation.
Fast tests through the protocol without a calculation; e2e (mesh + ccx) marked slow.
"""
import math

import pytest

pytest.importorskip("mcp")

from tests.test_mcp import run, beam_study, png_size, steps  # noqa: E402,F401 (fixture steps)


def test_find_edges_and_entity_selections(tmp_path, steps):
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        d = await s.ok("fea_geometry_find_edges", x=200, z=10)
        assert d["count"] == 1 and abs(d["edges"][0]["length_mm"] - 20) < 1e-9 and d["edges"][0]["id"].startswith("Konzola#e")
        d = await s.ok("fea_geometry_find_edges", direction=[1, 0, 0])
        assert d["count"] == 4
        assert (await s.ok("fea_geometry_find_edges", where="length > 150"))["count"] == 4
        d = await s.ok("fea_selection_create_edges", name="hrana", edges={"x": 200, "z": 10})
        assert d["edges"] == 1 and abs(d["length_mm"] - 20) < 1e-9
        d = await s.ok("fea_selection_create_points", name="bod", points=[200, 10, 5.2])   # it is projected onto the face
        assert d["points"] == [[200, 10, 5.2]]
        d = await s.ok("fea_selection_create_points", name="bod2", points=[[201, 10, 5]])
        assert abs(d["projected_mm"][0] - 1) < 1e-9 and d["points"] == [[200, 10, 5]]
        assert "surface" in await s.err("fea_selection_create_points", name="x", points=[100, 10, 50])
        sel = {x["name"]: x for x in (await s.ok("fea_selection_list"))["selections"]}
        assert sel["hrana"]["edges"] == 1 and len(sel["bod"]["points"]) == 1
    run(tmp_path, body)


def test_edge_point_force_tools(tmp_path, steps):
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        await s.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
        d = await s.ok("fea_load_add_edge_force", case_name="H", name="hrana", edges={"x": 200, "z": 10},
                       force=[0, 0, -100])
        assert d["load"]["applied_to"] == "edges" and "singular" in d["warning"]
        assert "edge" in d["load"]["description"]
        d = await s.ok("fea_load_add_point_force", case_name="B", name="body", points=[[200, 0, 10], [200, 20, 10]],
                       force=[0, 0, -100])
        assert d["load"]["applied_to"] == "body" and "2 points of F/2 each" in d["load"]["description"]
        # an existing selection of edges can be used also in fea_load_add_force
        d = await s.ok("fea_load_add_force", case_name="H2", name="hrana2", selection="hrana", force=[0, 0, -1])
        assert d["load"]["applied_to"] == "edges" and d["selection"]["edges"] == 1
        assert "faces only" in await s.err("fea_load_add_pressure", case_name="H3", name="p", selection="hrana",
                                              pressure=1.0)
        r = await s.ok("fea_check_readiness")
        assert r["ready"], r
        assert any("singular" in w for w in r["warnings"])
        r = await s.raw("fea_model_render_image", output_path=str(tmp_path / "m.png"), return_image=False)
        assert not r.is_error and png_size(tmp_path / "m.png") == (1280, 720)
    run(tmp_path, body)


def test_csys_tools(tmp_path, steps):
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(steps["plate"]), workdir=str(tmp_path / "wd"))
        d = await s.ok("fea_csys_create", name="dira", kind="cylindrical", from_face={"radius": 5})
        assert abs(d["axes_global"][2][2]) > 0.999 and "cylindrical" in d["description"]
        assert abs(d["description"].count("30")) >= 1
        await s.ok("fea_csys_create", name="K", origin=[0, 0, 0], z_axis=[1, 0, 0], x_axis=[0, 1, 0])
        await s.ok("fea_load_add_force", case_name="A", name="dira_tecne", selection={"radius": 5}, force=[0, 100, 0],
                   csys="dira")
        d = await s.ok("fea_load_add_remote", case_name="A", name="cela", selection={"x": 60}, point=[70, 20, 2.5],
                       force=[0, 0, 10], csys="K")
        assert "“K”" in d["load"]["description"]
        assert "cylindrical" in await s.err("fea_load_add_remote", case_name="A", name="cela2", selection={"x": 0},
                                        point=[0, 0, 0], force=[1, 0, 0], csys="dira")
        assert "does not exist" in await s.err("fea_load_add_force", case_name="A", name="x", selection={"x": 0},
                                           force=[1, 0, 0], csys="nic")
        lst = (await s.ok("fea_csys_list"))["coordinate_systems"]
        assert {x["name"] for x in lst} == {"dira", "K"} and lst[0]["used_by"]
        assert "is used by" in await s.err("fea_csys_delete", name="dira")
        await s.ok("fea_load_remove", case_name="A", name="dira_tecne")
        await s.ok("fea_csys_delete", name="dira")
        assert "parallel" in await s.err("fea_csys_create", name="Z", z_axis=[0, 0, 1], x_axis=[0, 0, 3])
    run(tmp_path, body)


def test_hydrostatic_rotation_tools(tmp_path, steps):
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        d = await s.ok("fea_load_add_hydrostatic", case_name="V", name="voda", selection={"y": 0},
                       surface_point=[0, 0, 8])
        ld = d["load"]
        assert ld["clip"] and abs(ld["gradient"][2] + 9.81e-6) < 1e-15 and "hydrostatic" in ld["description"]
        d = await s.ok("fea_load_add_pressure", case_name="V", name="lin", selection={"y": 20}, pressure=0.1,
                       gradient=[1e-3, 0, 0], origin=[100, 0, 0])
        assert "linear" in d["load"]["description"]
        d = await s.ok("fea_load_add_rotation", case_name="R", axis_point=[0, 10, 5], axis_direction=[1, 0, 0],
                       rpm=6000)
        assert abs(d["load"]["omega"] / (200 * math.pi) - 1) < 1e-4 and "rpm" in d["load"]["description"]
        assert "rpm OR omega" in await s.err("fea_load_add_rotation", case_name="R", axis_point=[0, 0, 0],
                                               axis_direction=[1, 0, 0], rpm=1, omega=1)
        d = await s.ok("fea_load_add_rotation", case_name="R", axis_point=[0, 10, 5], axis_direction=[0, 0, 2],
                       alpha=100.0)                       # one rotation per state → replacement
        assert d["replaced"] and (await s.ok("fea_load_list", case_name="R"))["load_cases"][0]["loads"][0]["alpha"] == 100
        await s.ok("fea_load_remove", case_name="R", name="otáčení")
        assert (await s.ok("fea_load_list", case_name="R"))["load_cases"][0]["loads"] == []
    run(tmp_path, body)


@pytest.mark.slow
def test_e2e_new_loads(tmp_path, steps):
    """Cantilever: force on an edge, at a point, cylindrical system, hydrostatics, rotation – mesh, ccx, equilibrium, hotspots."""
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        await s.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
        await s.ok("fea_load_add_edge_force", case_name="Hrana", name="hrana", edges={"x": 200, "z": 10},
                   force=[0, 0, -100])
        await s.ok("fea_load_add_point_force", case_name="Bod", name="bod", points=[200, 10, 5], force=[0, 0, -100])
        await s.ok("fea_csys_create", name="osa", kind="cylindrical", origin=[0, 10, 5], z_axis=[1, 0, 0])
        await s.ok("fea_load_add_force", case_name="Krut", name="celo", selection={"x": 200}, force=[0, 1000, 0],
                   csys="osa")
        await s.ok("fea_load_add_hydrostatic", case_name="Voda", name="voda", selection={"y": 0},
                   surface_point=[0, 0, 8], density_kg_m3=1e6)
        await s.ok("fea_load_add_rotation", case_name="Rotace", axis_point=[0, 10, 5], axis_direction=[1, 0, 0],
                   rpm=10000)
        await s.ok("fea_mesh_generate", global_size=4.0)
        r = await s.ok("fea_solve_run", timeout_s=900)
        assert all(st["equilibrium"]["ok"] for st in r["steps"]), r["steps"]
        summ = {st["step"]: st for st in (await s.ok("fea_results_summary", step=None))["steps"]}
        assert abs(summ["Hrana"]["reactions"]["vetknuti"]["force_N"][2] - 100) < 1e-2
        assert abs(summ["Bod"]["reactions"]["vetknuti"]["force_N"][2] - 100) < 1e-2
        # a point force on a plate supported by the whole lower face: the maximum stress is at the place of the force → singularity
        await s.ok("fea_constraint_remove", name="vetknuti")
        await s.ok("fea_constraint_add_fixed", name="spodek", selection={"z": 0})
        for lc in ("Hrana", "Krut", "Voda", "Rotace"):
            await s.ok("fea_loadcase_delete", name=lc)
        await s.ok("fea_load_add_point_force", case_name="Bod", name="bod", points=[100, 10, 10], force=[0, 0, -1000])
        await s.ok("fea_mesh_generate", global_size=4.0)
        await s.ok("fea_solve_run", timeout_s=900)
        hs = (await s.ok("fea_results_hotspots", step="Bod", top_n=3))["hotspots"][0]
        assert hs["classification"] == "singularity" and any("force into a point" in r for r in hs["reasons"]), hs
        assert abs(hs["xyz_mm"][0] - 100) < 8 and abs(hs["xyz_mm"][2] - 10) < 8, hs
    run(tmp_path, body)
