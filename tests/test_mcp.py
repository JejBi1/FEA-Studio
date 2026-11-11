"""MCP server (F2.5): tools through the real MCP protocol (SDK client in memory) + e2e console V1.

Fast tests without a calculation; e2e (mesh + ccx) and fatigue are marked slow.
"""
import json
import struct
import subprocess
import sys
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from fea_core.mcp.server import build_server  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

SPEC_TOOLS = {
    "fea_study_new", "fea_study_open", "fea_study_save", "fea_study_status",
    "fea_geometry_list_parts", "fea_geometry_find_faces", "fea_geometry_get_face_info",
    "fea_materials_list", "fea_materials_assign",
    "fea_constraint_add_fixed", "fea_constraint_add_displacement",
    "fea_loadcase_create", "fea_load_add_force", "fea_load_add_pressure", "fea_load_add_gravity", "fea_load_add_remote",
    "fea_mesh_generate", "fea_check_readiness", "fea_solve_run",
    "fea_results_summary", "fea_results_probe", "fea_results_hotspots", "fea_results_render_image",
    "fea_fatigue_run",
}


# ===================================================================== helpers

@pytest.fixture(scope="module")
def steps(tmp_path_factory):
    """STEP: console V1 (200×20×10) and a plate with a hole Ø10 (geometry for the selectors)."""
    from verification.cases import box, cyl, solid
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from fea_core.geometry import Part, write_step
    d = tmp_path_factory.mktemp("mcp_geo")
    beam = d / "v1.step"
    write_step([Part("Konzola", box(0, 0, 0, 200, 20, 10))], beam)
    plate = d / "plate.step"
    write_step([Part("Deska", solid(BRepAlgoAPI_Cut(box(0, 0, 0, 60, 40, 5),
                                                   cyl((30, 20, -1), (0, 0, 1), 5, 7)).Shape()))], plate)
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    u = ShapeUpgrade_UnifySameDomain(BRepAlgoAPI_Fuse(box(0, 0, 0, 80, 40, 8), box(0, 0, 0, 8, 40, 68)).Shape(),
                                     True, True, True)
    u.Build()
    bracket = d / "bracket.step"
    write_step([Part("Uhelnik", solid(u.Shape()))], bracket)
    return {"beam": beam, "plate": plate, "bracket": bracket}


class Session:
    """MCP client in memory (legacy mode = JSON-RPC through streams, as stdio)."""

    def __init__(self, client):
        self.c = client

    async def call(self, tool, **args):
        r = await self.c.call_tool(tool, args)
        txt = r.content[0].text
        if r.is_error:
            return {"__error__": txt}
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return {"__text__": txt}

    async def ok(self, tool, **args):
        d = await self.call(tool, **args)
        assert "__error__" not in d, f"{tool}: {d}"
        return d

    async def err(self, tool, **args) -> str:
        d = await self.call(tool, **args)
        assert "__error__" in d, f"{tool} měl skončit chybou: {d}"
        return d["__error__"]

    async def raw(self, tool, **args):
        return await self.c.call_tool(tool, args)


def run(tmp_path, body, mode="legacy"):
    async def main():
        srv = build_server(base_dir=tmp_path / "mcp")
        async with Client(srv, mode=mode) as c:
            await body(Session(c), c)
    anyio.run(main)


def png_size(path: Path) -> tuple[int, int]:
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "není PNG"
    return struct.unpack(">II", data[16:24])


async def beam_study(s, steps, tmp_path, material=True):
    await s.ok("fea_study_new", step_path=str(steps["beam"]), workdir=str(tmp_path / "wd"))
    if material:
        await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")


# ===================================================================== fast tests

def test_headless_import_without_qt():
    """Importing the server and the renderer does not pull in PySide6 (a clean process)."""
    code = ("import sys; from fea_core.mcp.server import build_server; build_server(); "
            "import fea_core.mcp.render, fea_core.mcp.scenes, fea_core.solve, fea_core.mcp.diagnostics; "
            "print('QT' if any(m.startswith(('PySide6', 'fea_gui')) for m in sys.modules) else 'OK')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.stdout.strip().endswith("OK"), r.stdout + r.stderr


def test_tools_resources_prompts(tmp_path):
    async def body(s, c):
        names = {t.name for t in (await c.list_tools()).tools}
        assert SPEC_TOOLS <= names, SPEC_TOOLS - names
        res = {str(r.uri) for r in (await c.list_resources()).resources}
        assert {"fea://study/current", "fea://results/summary", "fea://materials/catalog"} <= res
        prompts = {p.name for p in (await c.list_prompts()).prompts}
        assert {"fea_setup_static_analysis", "fea_mesh_convergence_check", "fea_interpret_stress_results"} <= prompts
        cat = json.loads((await c.read_resource("fea://materials/catalog")).contents[0].text)
        assert any(m["name"] == "Ocel S235" for m in cat["materials"])
        txt = (await c.read_resource("fea://results/summary")).contents[0].text
        assert "study" in txt.lower()
        p = await c.get_prompt("fea_mesh_convergence_check", {"quantity": "σ v rohu"})
        assert "Richardson" in p.messages[0].content.text and "singular" in p.messages[0].content.text.lower()
        p = await c.get_prompt("fea_interpret_stress_results", {})
        assert "Re" in p.messages[0].content.text
    run(tmp_path, body)


def test_no_study_errors(tmp_path):
    async def body(s, c):
        assert "study" in (await s.err("fea_study_status")).lower()
        assert "no results" in (await s.err("fea_results_summary")).lower()
        assert "does not exist" in (await s.err("fea_study_new", step_path=str(tmp_path / "nic.step")))
    run(tmp_path, body)


def test_study_geometry_and_selectors(tmp_path, steps):
    async def body(s, c):
        d = await s.ok("fea_study_new", step_path=str(steps["plate"]), workdir=str(tmp_path / "wd"))
        assert d["parts"][0]["name"] == "Deska" and d["parts"][0]["faces"] == 7
        parts = await s.ok("fea_geometry_list_parts")
        assert parts["parts"][0]["size_mm"] == [60.0, 40.0, 5.0]
        assert abs(parts["parts"][0]["volume_mm3"] - (60 * 40 * 5 - 3.14159265 * 25 * 5)) < 1.0   # 5 valid digits
        f = await s.ok("fea_geometry_find_faces", x=0)
        assert f["count"] == 1 and f["faces"][0]["normal"] == [-1.0, 0.0, 0.0]
        f = await s.ok("fea_geometry_find_faces", radius=5)
        assert f["count"] == 1 and f["faces"][0]["type"] == "cylinder" and f["faces"][0]["radius_mm"] == 5.0
        hole = f["faces"][0]["id"]
        f = await s.ok("fea_geometry_find_faces", cylinder={"point": [30, 20, 0], "axis": [0, 0, 1]})
        assert [x["id"] for x in f["faces"]] == [hole]
        f = await s.ok("fea_geometry_find_faces", normal=[0, 0, 1])
        assert f["count"] == 1 and f["faces"][0]["centroid_mm"][2] == 5.0
        f = await s.ok("fea_geometry_find_faces", plane={"point": [0, 0, 5], "normal": [0, 0, -1]})
        assert f["count"] == 1
        f = await s.ok("fea_geometry_find_faces", where="type == 'plane' and area > 1000", near=[0, 0, 0], limit=2)
        assert f["count"] == 2 and all(x["area_mm2"] > 1000 for x in f["faces"])
        assert "Only the functions" in await s.err("fea_geometry_find_faces", where="__import__('os')")
        assert "Disallowed" in await s.err("fea_geometry_find_faces", where="[x for x in (1,)]")
        f = await s.ok("fea_geometry_find_faces", x=123)
        assert f["count"] == 0 and "hint" in f
        info = await s.ok("fea_geometry_get_face_info", selector=hole)
        assert info["radius_mm"] == 5.0 and len(info["neighbors"]) == 2 and info["selector"]["part"] == "Deska"
        # selections: id, query, selector; they survive as FaceSelector in the study
        sel = await s.ok("fea_selection_create", name="otvor", faces=[hole])
        assert sel["faces"] == 1
        sel = await s.ok("fea_selection_create", name="boky", faces=[{"x": 0}, {"x": 60}])
        assert sel["faces"] == 2 and sel["area_mm2"] == 400.0
        sel = await s.ok("fea_selection_create", name="kopie", faces=info["selector"])
        assert sel["faces"] == 1
        lst = await s.ok("fea_selection_list")
        assert {x["name"] for x in lst["selections"]} == {"otvor", "boky", "kopie"}
        await s.ok("fea_selection_delete", name="kopie")
        assert "does not exist" in await s.err("fea_selection_delete", name="kopie")
        assert "Unknown" in await s.err("fea_selection_create", name="x", faces={"foo": 1})
        assert "found no face" in await s.err("fea_selection_create", name="x", faces={"x": 999})
        assert "Part 'Nic'" in await s.err("fea_selection_create", name="x", faces="Nic#0")
    run(tmp_path, body)


def test_materials(tmp_path, steps):
    async def body(s, c):
        cat = await s.ok("fea_materials_list")
        assert len(cat["materials"]) >= 10
        q = await s.ok("fea_materials_list", query="S355")
        assert [m["name"] for m in q["materials"]] == ["Ocel S355"]
        await beam_study(s, steps, tmp_path, material=False)
        st = await s.ok("fea_study_status")
        assert st["parts"][0]["material"] is None
        rd = await s.ok("fea_check_readiness")
        assert not rd["ready"] and any("material" in e for e in rd["errors"])
        d = await s.ok("fea_materials_assign", part="Konzola", material_name="Moje ocel",
                       custom_props={"E": 200000, "nu": 0.29, "rho": 7.8e-9, "Re": 300})
        assert d["assigned"]["material"]["E_MPa"] == 200000 and d["parts_without_material"] == []
        d = await s.ok("fea_materials_assign", part="*", material_name="Ocel S235", custom_props={"Re": 250})
        assert d["assigned"]["material"]["Re_MPa"] == 250
        assert "kg/m³" in await s.err("fea_materials_assign", part="*", material_name="X",
                                      custom_props={"E": 1, "nu": 0.3, "rho": 7850})
        assert "catalogue" in await s.err("fea_materials_assign", part="*", material_name="Neznámý")
        assert "does not exist" in await s.err("fea_materials_assign", part="Nic", material_name="Ocel S235")
    run(tmp_path, body)


def test_constraints_loads_readiness(tmp_path, steps):
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        # only uz = 0 on the face x = 0 holds the displacement z and the rotation about x → free displacements x, y and rotations about y, z
        await s.ok("fea_constraint_add_displacement", name="podpora", selection={"x": 0}, uz=0.0)
        await s.ok("fea_load_add_force", case_name="F", name="konec", selection={"x": 200}, force=[0, 0, -100])
        rd = await s.ok("fea_check_readiness")
        assert not rd["ready"] and any("rigid-body motions" in e for e in rd["errors"])
        modes = rd["rigid_body"]["free_rigid_modes"]
        assert len(modes) == 4 and any("displacement" in m for m in modes) and any("rotation" in m for m in modes)
        assert "at least one" in await s.err("fea_constraint_add_displacement", name="x", selection={"x": 0})
        d = await s.ok("fea_constraint_add_fixed", name="podpora", selection="podpora")
        assert d["replaced"] and d["constraint"]["kind"] == "fixed"
        rd = await s.ok("fea_check_readiness")
        assert rd["ready"] and rd["rigid_body"]["ok"] and rd["dof"] > 0
        lst = await s.ok("fea_constraint_list")
        assert [x["name"] for x in lst["constraints"]] == ["podpora"]
        # loads of all kinds
        await s.ok("fea_loadcase_create", name="G")
        g = await s.ok("fea_load_add_gravity", case_name="G")
        assert g["load"]["vector"] == [0.0, 0.0, -9810.0]
        await s.ok("fea_load_add_pressure", case_name="G", name="horni", selection={"z": 10}, pressure=0.5)
        r = await s.ok("fea_load_add_remote", case_name="G", name="cep", selection={"x": 200}, point=[250, 10, 5],
                       force=[0, 100, 0], moment=[1000, 0, 0])
        assert r["load"]["coupling"] == "distributing"
        assert "coupling" in await s.err("fea_load_add_remote", case_name="G", name="x", selection={"x": 200},
                                         point=[0, 0, 0], force=[1, 0, 0], coupling="tuha")
        assert "vector" in await s.err("fea_load_add_force", case_name="G", name="x", selection={"x": 200}, force=[1, 2])
        assert "already used by" in await s.err("fea_load_add_force", case_name="G", name="podpora", selection={"x": 200},
                                           force=[1, 0, 0])
        lst = await s.ok("fea_load_list")
        kinds = {lc["name"]: sorted(l["kind"] for l in lc["loads"]) for lc in lst["load_cases"]}
        assert kinds == {"F": ["force"], "G": ["gravity", "pressure", "remote"]}
        await s.ok("fea_load_remove", case_name="G", name="Gravity")
        await s.ok("fea_load_remove", case_name="G", name="cep")
        assert "does not exist" in await s.err("fea_load_remove", case_name="G", name="cep")
        await s.ok("fea_loadcase_delete", name="G")
        st = await s.ok("fea_study_status")
        assert [lc["name"] for lc in st["load_cases"]] == ["F"]
        await s.ok("fea_constraint_remove", name="podpora")
        rd = await s.ok("fea_check_readiness")
        assert any("support" in e for e in rd["errors"])
    run(tmp_path, body)


def test_cylindrical_support_and_bearing(tmp_path, steps):
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(steps["plate"]), workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
        # hole Ø10 (axis z through (30, 20)) only radially + axially → the rotation about the axis of the hole stays free
        d = await s.ok("fea_constraint_add_cylindrical", name="cep", selection={"radius": 5}, tangential=False)
        assert d["constraint"]["kind"] == "cylindrical" and not d["constraint"]["tangential"]
        await s.ok("fea_load_add_bearing", case_name="F", name="oko", selection="cep", force=[100, 0, 0])
        rd = await s.ok("fea_check_readiness")
        modes = rd["rigid_body"]["free_rigid_modes"]
        assert len(modes) == 1 and "rotation" in modes[0] and "through the point (30, 20," in modes[0], modes
        ax = [float(x) for x in modes[0].split("the axis (")[1].split(")")[0].split(",")]
        assert abs(abs(ax[2]) - 1) < 1e-6, modes
        await s.ok("fea_constraint_add_cylindrical", name="cep", selection="cep")
        rd = await s.ok("fea_check_readiness")
        assert rd["rigid_body"]["ok"], rd
        assert "holds nothing" in await s.err("fea_constraint_add_cylindrical", name="x", selection={"radius": 5},
                                          radial=False, tangential=False, axial=False)
        lst = await s.ok("fea_load_list")
        assert lst["load_cases"][0]["loads"][0]["kind"] == "bearing"
        img = await s.ok("fea_model_render_image", output_path=str(tmp_path / "cyl.png"))
        assert png_size(tmp_path / "cyl.png")[0] > 0, img
    run(tmp_path, body)


def test_remote_support(tmp_path, steps):
    """Remote constraint: point = centroid of the selection, check of the rigid motions (6 components = no free motion, only displacements =
    3 free rotations about the point), parameter errors, a marker in the image of the model.
    """
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        await s.ok("fea_load_add_force", case_name="F", name="konec", selection={"x": 200}, force=[0, 0, -100])
        d = await s.ok("fea_constraint_add_remote", name="celo", selection={"x": 0})
        con = d["constraint"]
        assert con["kind"] == "remote_support" and con["point"] == [0.0, 10.0, 5.0] and con["coupling"] == "rigid"
        assert con["dofs"] == [True] * 6 and con["dofs_fixed"] == "ux, uy, uz, rx, ry, rz"
        rd = await s.ok("fea_check_readiness")
        assert rd["ready"] and rd["rigid_body"]["ok"], rd
        # only displacements of a point → the 3 rotations about the axes through the point (0, 10, 5) are free
        d = await s.ok("fea_constraint_add_remote", name="celo", selection="celo", point=[0, 10, 5], dofs="ux,uy,uz",
                       coupling="distributing")
        assert d["replaced"] and d["constraint"]["dofs"] == [True, True, True, False, False, False]
        modes = (await s.ok("fea_check_readiness"))["rigid_body"]["free_rigid_modes"]
        assert len(modes) == 3 and all("rotation" in m and "through the point (0, 10, 5)" in m for m in modes), modes
        # a list of bool + a point off the face: uz and rx, ry, rz at the point [−50, 10, 5] → free displacements x, y and rotation about z?
        # (displacement of the point z + rotations x, y, z: only the displacements x and y are free)
        await s.ok("fea_constraint_add_remote", name="celo", selection="celo", point=[-50, 10, 5],
                   dofs=[False, False, True, True, True, True])
        modes = (await s.ok("fea_check_readiness"))["rigid_body"]["free_rigid_modes"]
        assert len(modes) == 2 and all("displacement" in m for m in modes), modes
        assert "coupling" in await s.err("fea_constraint_add_remote", name="x", selection={"x": 200}, coupling="tuha")
        assert "Unknown remote support component" in await s.err("fea_constraint_add_remote", name="x", selection={"x": 200}, dofs="ux,uq")
        assert "holds nothing" in await s.err("fea_constraint_add_remote", name="x", selection={"x": 200},
                                          dofs=[False] * 6)
        st = await s.ok("fea_study_status")
        assert "remote support" in st["constraints"][0]["description"]
        img = await s.ok("fea_model_render_image", output_path=str(tmp_path / "remote.png"))
        assert png_size(tmp_path / "remote.png")[0] > 0, img
    run(tmp_path, body)


@pytest.mark.slow
def test_e2e_remote_support(tmp_path, steps):
    """Cantilever with a rigid remote constraint of the face (6 components) through MCP: reaction at the point 100 N and the moment F·L about the point."""
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        await s.ok("fea_constraint_add_remote", name="celo", selection={"x": 0})
        await s.ok("fea_load_add_force", case_name="F", name="konec", selection={"x": 200}, force=[0, 0, -100])
        await s.ok("fea_mesh_generate", global_size=5.0)
        r = await s.ok("fea_solve_run", timeout_s=600)
        st = r["steps"][0]
        assert r["backend"] == "ccx" and st["equilibrium"]["ok"]
        rs = (await s.ok("fea_results_summary"))["steps"][0]["remote_supports"]["celo"]
        assert abs(rs["force_N"][2] - 100) < 1e-3 and abs(rs["moment_about_point_Nmm"][1] + 20000) < 0.1, rs
        assert rs["coupling"] == "rigid" and rs["fixed_dofs"] == ["ux", "uy", "uz", "rx", "ry", "rz"]
    run(tmp_path, body)


def test_save_open_and_model_image(tmp_path, steps):
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        await s.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
        await s.ok("fea_load_add_force", case_name="F", name="konec", selection={"x": 200}, force=[0, 0, -100])
        await s.ok("fea_load_add_gravity", case_name="F")
        r = await s.raw("fea_model_render_image", output_path=str(tmp_path / "model.png"), view="multi",
                        highlight={"z": 10}, width=1000, height=600)
        assert not r.is_error and [x.type for x in r.content] == ["text", "image"]
        assert png_size(tmp_path / "model.png") == (1000, 600)
        r = await s.raw("fea_model_render_image", output_path=str(tmp_path / "ids.png"), view="front",
                        show_face_ids=True, return_image=False)
        assert [x.type for x in r.content] == ["text"] and png_size(tmp_path / "ids.png") == (1280, 720)
        assert "view" in (await s.err("fea_model_render_image", view="sikmo")).lower()
        await s.ok("fea_study_save", path=str(tmp_path / "out" / "study.json"))
        await s.ok("fea_study_save", path=str(tmp_path / "out" / "projekt.fea"))
        cur = json.loads((await c.read_resource("fea://study/current")).contents[0].text)
        assert cur["constraints"][0]["selection"] == "vetknuti"
        d = await s.ok("fea_study_open", path=str(tmp_path / "out" / "study.json"))
        assert d["constraints"][0]["name"] == "vetknuti" and d["parts"][0]["material"] == "Ocel S235"
        d = await s.ok("fea_study_open", path=str(tmp_path / "out" / "projekt.fea"), workdir=str(tmp_path / "w2"))
        assert [lc["name"] for lc in d["load_cases"]] == ["F"] and d["results"] == "nejsou"
    run(tmp_path, body)


def test_joints_print_tools(tmp_path, steps):
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        d = await s.ok("fea_joints_detect")
        assert d["bolts"] == [] and d["welds"] == []
        assert "exists" in await s.err("fea_joints_update", name="M8", changes={"enabled": False})
        d = await s.ok("fea_joints_update", name="*settings*", changes={"category": "C"})
        assert d["settings"]["category"] == "C"
        d = await s.ok("fea_print_set", part="Konzola", settings={"infill": 20, "walls": 3})
        assert d["print"]["infill"] == 0.2 and d["print"]["walls"] == 3 and "warning" in d
        await s.ok("fea_materials_assign", part="Konzola", material_name="Tisk PLA Bambu Basic")
        d = await s.ok("fea_print_set", part="Konzola", criterion="tsai_hill")
        assert "warning" not in d and d["criterion"] == "tsai_hill"
        assert "pattern" in await s.err("fea_print_set", part="Konzola", settings={"pattern": "spirála"})
        d = await s.ok("fea_joints_list")
        assert d["bolts"] == [] and d["welds"] == [] and d["settings"]["category"] == "C"
    run(tmp_path, body)


def test_parts_ignored(tmp_path, steps):
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(steps["plate"]), workdir=str(tmp_path / "wd"))
        assert (await s.ok("fea_study_set_parts_ignored", parts=["Deska"]))["ignored_parts"] == ["Deska"]
        d = await s.ok("fea_study_set_parts_ignored", parts=["Deska"], ignored=False)
        assert d["ignored_parts"] == []
        assert "does not exist" in await s.err("fea_study_set_parts_ignored", parts=["X"])
    run(tmp_path, body)


def test_unknown_argument_is_error(tmp_path):
    """A typo in the name of an argument = an error (the SDK would otherwise silently discard it – e.g. a mesh with an automatic size)."""
    async def body(s, c):
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert "additionalProperties" not in tools["fea_mesh_generate"].input_schema   # not listed (compact schema); still enforced:
        assert "querry" in await s.err("fea_materials_list", querry="S235")
        assert "Ocel S235" in json.dumps(await s.ok("fea_materials_list", query="S235"), ensure_ascii=False)
    run(tmp_path, body)


def test_failed_open_keeps_study(tmp_path, steps):
    """An unreadable project / STEP does not discard the current (unsaved) study."""
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        bad = tmp_path / "poskozeny.fea"
        bad.write_bytes(b"neni zip")
        assert "cannot be opened" in await s.err("fea_study_open", path=str(bad))
        badstep = tmp_path / "vadny.step"
        badstep.write_text("ISO-10303-21;\nNESMYSL\n", encoding="ascii")
        await s.err("fea_study_new", step_path=str(badstep))
        await s.err("fea_study_new", step_path=str(steps["beam"]), material="Neznámý materiál")
        meta = tmp_path / "metadata.json"                     # metadata z CADu nejsou studie (project.load_study_json)
        meta.write_text(json.dumps({"source": "Test", "parts": []}), encoding="utf-8")
        assert "CAD metadata" in await s.err("fea_study_open", path=str(meta))
        st = await s.ok("fea_study_status")
        assert st["study"] == "v1" and st["parts"][0]["material"] == "Ocel S235"
    run(tmp_path, body)


def test_study_new_reads_cad_metadata(tmp_path, steps):
    """CAD metadata next to the STEP (<name>.json) assign the material as the GUI does (fea_core.cad_import)."""
    async def body(s, c):
        import shutil
        step = tmp_path / "drzak.step"
        shutil.copy(steps["beam"], step)
        (tmp_path / "drzak.json").write_text(json.dumps({"source": "Test", "parts": [
            {"name": "Konzola", "material": "Ocel S235"}]}), encoding="utf-8")
        d = await s.ok("fea_study_new", step_path=str(step), workdir=str(tmp_path / "wd"))
        assert d["cad_metadata"]["assigned"] == {"Konzola": "Ocel S235"}
        assert (await s.ok("fea_study_status"))["parts"][0]["material"] == "Ocel S235"
        d = await s.ok("fea_study_new", step_path=str(step), workdir=str(tmp_path / "wd2"), apply_cad_metadata=False)
        assert "cad_metadata" not in d
    run(tmp_path, body)


def test_selection_used_by_mass_is_protected(tmp_path, steps):
    """The selection of an added mass cannot be overwritten by other faces or deleted (previously the mass silently moved)."""
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        await s.ok("fea_dyn_mass_add", name="motor", selection={"x": 200}, mass_kg=2.0)
        e = await s.err("fea_load_add_pressure", case_name="LC1", name="motor", selection={"z": 10}, pressure=1.0)
        assert "added mass" in e
        assert "is in use" in await s.err("fea_selection_delete", name="motor")
        sel = {x["name"]: x for x in (await s.ok("fea_selection_list"))["selections"]}
        assert sel["motor"]["centroid_mm"][0] == pytest.approx(200) and sel["motor"]["used_by"]
        await s.ok("fea_dyn_mass_add", name="motor", selection={"x": 0}, mass_kg=2.0)    # the replacement of the mass is allowed
    run(tmp_path, body)


def test_mesh_backend_and_joint_settings_keep_mesh(tmp_path, steps):
    """solver_backend of the mesh = the real choice auto (solve.choose_backend, also ngs_pardiso from 2 states); an assessment change
    of the joint settings does not discard the mesh.
    """
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        await s.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
        for lc in ("A", "B"):
            await s.ok("fea_load_add_force", case_name=lc, name=f"F{lc}", selection={"x": 200}, force=[0, 0, -100])
        m = await s.ok("fea_mesh_generate", global_size=10)
        from fea_core.solve import choose_backend
        sess = srv_sess[0]
        assert m["solver_backend"] == f"{choose_backend(sess.study, sess.mesh)} (auto)"
        d = await s.ok("fea_joints_update", name="*settings*", changes={"gamma_M2": 1.3})
        assert d["effect"].startswith("assessment only") and sess.valid_mesh() is not None
        d = await s.ok("fea_joints_update", name="*settings*", changes={"bolt_mesh": 0.3})
        assert d["effect"].startswith("new mesh") and sess.valid_mesh() is None

    srv_sess = []

    async def main():
        srv = build_server(base_dir=tmp_path / "mcp")
        srv_sess.append(srv.fea_session)
        async with Client(srv, mode="legacy") as c:
            await body(Session(c), c)
    anyio.run(main)


def test_cancelled_call_stops_meshing(tmp_path, steps):
    """Cancellation of a call by the client (notifications/cancelled) ends the meshing and releases the session (previously Gmsh kept running
    and the other tools reported "busy" until the end).
    """
    import time

    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        with anyio.move_on_after(2) as scope:
            await s.call("fea_mesh_generate", global_size=0.6, timeout_s=600)
        assert scope.cancelled_caught
        t0 = time.time()
        await s.ok("fea_study_status")
        assert time.time() - t0 < 8
        assert "No mesh" in await s.err("fea_mesh_info")
    run(tmp_path, body)


def test_http_requires_token_outside_loopback(tmp_path):
    """HTTP MCP: a non-loopback address without a token does not start; Host/Origin also for 127.0.0.2; token → 401/200."""
    from starlette.testclient import TestClient
    from fea_core.mcp import __main__ as mm
    assert mm.main(["--transport", "streamable-http", "--host", "0.0.0.0"]) == 2
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
    hdr = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}

    def post(host, token, base, extra):
        app = mm.http_app(build_server(base_dir=tmp_path / "mcp"), "streamable-http", host, token)
        with TestClient(app, base_url=base) as tc:
            return tc.post("/mcp", content=json.dumps(init), headers={**hdr, **extra}).status_code
    assert post("127.0.0.2", None, "http://127.0.0.2:8799", {"Host": "attacker.example"}) == 421
    assert post("127.0.0.2", None, "http://127.0.0.2:8799", {"Origin": "http://attacker.example"}) == 403
    assert post("127.0.0.2", None, "http://127.0.0.2:8799", {}) == 200
    assert post("0.0.0.0", "T", "http://192.168.1.50:8799", {}) == 401
    assert post("0.0.0.0", "T", "http://192.168.1.50:8799", {"Authorization": "Bearer T"}) == 200


def test_joints_detect_shape_suggestions_and_weld_colors(tmp_path):
    """fea_joints_detect: a bolt without a standard in the name = a proposal by shape (suggestions, acceptance accept_shape), a weld by
    colour from the settings of the joints of the study (weld_colors) – as the GUI and the CLI.
    """
    from fea_core.geometry import Part, write_step
    from verification.joint_models import lap_joint, t_joint_fillet
    lap = tmp_path / "lap.step"
    write_step([Part(f"Body{i}", p.shape) for i, p in enumerate(lap_joint())], lap)
    ps = t_joint_fillet()
    ps[2] = Part("Body3", ps[2].shape, (1.0, 0.5, 0.0))
    tj = tmp_path / "tj.step"
    write_step(ps, tj)

    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(lap), workdir=str(tmp_path / "wd1"))
        d = await s.ok("fea_joints_detect", apply=False)
        assert d["bolts"] == [] and d["suggestions"] and d["suggestions"][0]["source"] == "shape"
        name = d["suggestions"][0]["bolt"]["name"]
        d = await s.ok("fea_joints_detect", accept_shape=[name])
        assert [b["name"] for b in d["bolts"]] == [name] and d["suggestions"][0]["accepted"]
        assert "does not exist" in await s.err("fea_joints_detect", accept_shape=["Šroub 99"])
        await s.ok("fea_study_new", step_path=str(tj), workdir=str(tmp_path / "wd2"))
        await s.ok("fea_joints_update", name="*settings*", changes={"weld_colors": ["#ff8000"]})
        d = await s.ok("fea_joints_detect", apply=False)
        assert "Body3" in [w["part"] for w in d["welds"]]
    run(tmp_path, body)


def test_cleanup_old_workdirs(tmp_path):
    """Cleanup: only the automatically created folders (marker) and those not changed for a long time are deleted; the user's folders and the folders
    with a protocol .docx / project .fea stay.
    """
    import os
    import time
    from fea_core.mcp.session import Session, cleanup_workdirs
    base = tmp_path / "mcp"
    sess = Session(base)
    old_auto = sess.new_workdir("stara")
    (old_auto / "run").mkdir()
    with_doc = sess.new_workdir("s_protokolem_x")
    (with_doc / "protokol_1.docx").write_bytes(b"x")
    user = sess.new_workdir("x", base / "uzivatel")
    new_auto = sess.new_workdir("nova_y")
    past = time.time() - 40 * 86400
    for d in (old_auto, with_doc, user):
        for p in [d, *d.iterdir()]:
            os.utime(p, (past, past))
    removed = cleanup_workdirs(base, days=30)
    assert removed == [str(old_auto)] and not old_auto.exists()
    assert with_doc.exists() and user.exists() and new_auto.exists()
    assert cleanup_workdirs(base, days=0) == []


def test_mesh_timeout_kills_worker(tmp_path, steps):
    """Exceeded limit: the meshing process is terminated and the server answers with an error (it does not hang)."""
    async def body(s, c):
        await beam_study(s, steps, tmp_path)
        e = await s.err("fea_mesh_generate", global_size=1.0, timeout_s=0.5)
        assert "time limit" in e
        st = await s.ok("fea_study_status")              # the server still answers
        assert st["mesh"] == "none"
    run(tmp_path, body)


def test_stdio_server_starts(tmp_path):
    """python -m fea_core.mcp over stdio: handshake, list of tools, call."""
    from mcp.client.stdio import StdioServerParameters

    async def main():
        params = StdioServerParameters(command=sys.executable, args=["-m", "fea_core.mcp", "--groups", "all", "--workdir",
                                                                     str(tmp_path / "mcp")], cwd=str(ROOT))
        with anyio.fail_after(120):
            async with Client(params, mode="legacy") as c:
                names = {t.name for t in (await c.list_tools()).tools}
                assert SPEC_TOOLS <= names
                r = await c.call_tool("fea_materials_list", {"query": "S235"})
                assert "Ocel S235" in r.content[0].text
    anyio.run(main)


# ===================================================================== e2e (mesh + ccx)

@pytest.mark.slow
def test_e2e_v1_cantilever(tmp_path, steps):
    """Console V1 through the whole chain via MCP: clamping, 100 N, mesh, calculation, deflection 0.7634 mm (3 %), PNG."""
    qt_before = "PySide6" in sys.modules

    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(steps["beam"]), name="V1 konzola", workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Ocel S235",
                   custom_props={"E": 210000, "nu": 0.3})
        faces = await s.ok("fea_geometry_find_faces", x=0)
        await s.ok("fea_constraint_add_fixed", name="vetknuti", selection=[faces["faces"][0]["id"]])
        await s.ok("fea_loadcase_create", name="Sila 100 N")
        await s.ok("fea_load_add_force", case_name="Sila 100 N", name="konec", selection={"x": 200},
                   force=[0, 0, -100])
        assert (await s.ok("fea_check_readiness"))["ready"]
        m = await s.ok("fea_mesh_generate", global_size=2.5)
        assert m["element_type"] == "C3D10" and m["min_scaled_jacobian"] > 0.3 and m["dof"] > 30000
        r = await s.ok("fea_solve_run", timeout_s=600, backend="ccx")
        assert r["backend"] == "ccx" and "mesh_generated" not in r          # mesh taken over from fea_mesh_generate
        assert (await s.ok("fea_study_status"))["solver"]["backend"] == "auto"   # backend only for this run
        st = r["steps"][0]
        assert st["equilibrium"]["ok"] and abs(st["reactions"]["vetknuti"]["force_N"][2] - 100) < 0.01
        # deflection of the end (centre of the face) – Timoshenko 0.7634 mm
        p = await s.ok("fea_results_probe", point=[200, 10, 5])
        uz = -p["probes"][0]["u_mm"][2]
        assert abs(uz - 0.7634) / 0.7634 < 0.03, uz
        p = await s.ok("fea_results_probe", point=[[100, 10, 10], [100, 10, 0], [300, 0, 0]])
        sx_top, sx_bot = p["probes"][0]["stress_MPa"]["sxx"], p["probes"][1]["stress_MPa"]["sxx"]
        assert abs(sx_top - 30) < 1.5 and abs(sx_bot + 30) < 1.5
        assert not p["probes"][2]["inside_mesh"] and "outside" in p["probes"][2]["warning"]
        summ = await s.ok("fea_results_summary")
        s0 = summ["steps"][0]
        assert abs(s0["max_displacement_mm"]["value"] - 0.76) < 0.03
        assert s0["parts"][0]["Re_MPa"] == 235 and s0["min_safety_Re"] > 3
        hs = await s.ok("fea_results_hotspots", top_n=3)
        top = hs["hotspots"][0]
        assert top["xyz_mm"][0] < 10 and top["classification"] == "singularity"     # edge of the clamping
        assert any("constraint face 'vetknuti'" in r for r in top["reasons"])
        img = tmp_path / "v1_mises.png"
        r = await s.raw("fea_results_render_image", output_path=str(img), field="mises", view="iso",
                        width=1280, height=720)
        assert not r.is_error and [x.type for x in r.content] == ["text", "image"]
        meta = json.loads(r.content[0].text)
        assert meta["deformation_scale"] > 1 and abs(meta["surface_max"] - s0["max_von_mises_MPa"]["value"]) < 1e-6
        assert png_size(img) == (1280, 720) and img.stat().st_size > 20000
        r = await s.raw("fea_results_render_image", output_path=str(tmp_path / "uz.png"), field="uz", view="multi",
                        deformed=False, show_mesh=True, width=800, height=500)
        assert not r.is_error and png_size(tmp_path / "uz.png") == (800, 500)
        assert "Unknown quantity" in await s.err("fea_results_render_image", field="napeti")
        assert "contain no temperature" in await s.err("fea_results_render_image", field="teplota")
        # fatigue: alternating cycle 0 → 1 → 0 (S-N estimate from Rm), then the damage map
        f = await s.ok("fea_fatigue_run", history_curve=[[0, 0], [1, 1], [2, 0]], repeats=1e6)
        assert f["min_safety"] > 1 and f["worst"][0]["node"] >= 1 and "PASS" in f["verdict"]
        f = await s.ok("fea_fatigue_run", history_curve={"load_case": "Sila 100 N", "kind": "sine", "amplitude": 3.0},
                       method="max_principal", sn_overrides={"SD": 100, "ND": 1e6, "k": 5, "k2": 0})
        assert f["max_total_damage"] > 1 and f["sn_used"]["Ocel S235"]["sn"]["k"] == 5
        # critical plane (Findley): the worst node at the clamping has the transverse contraction prevented (σy, σz ≈ ν·σx) →
        # smaller shear than the uniaxial state → Findley slightly less strict than the max. principal (measured 0.485 vs 0.396)
        fc = await s.ok("fea_fatigue_run", history_curve={"load_case": "Sila 100 N", "kind": "sine", "amplitude": 3.0},
                        method="critical_plane", cp_criterion="findley",
                        sn_overrides={"SD": 100, "ND": 1e6, "k": 5, "k2": 0})
        assert "critical plane" in " ".join(fc["method"]) and fc["worst"][0]["normal"] is not None
        assert 1.0 <= fc["min_safety"] / f["min_safety"] < 1.4, (fc["min_safety"], f["min_safety"])
        assert "cp_criterion" in await s.err("fea_fatigue_run", history_curve=[0, 1], method="critical_plane",
                                             cp_criterion="xx")
        assert "cp_ratio" in await s.err("fea_fatigue_run", history_curve=[0, 1], method="max_principal", cp_ratio=5)
        cur = json.loads((await c.read_resource("fea://study/current")).contents[0].text)
        assert cur["fatigue"]["equivalent"] == "critical_plane"           # an invalid call did not change the study
        r = await s.raw("fea_results_render_image", output_path=str(tmp_path / "fat.png"), field="fatigue_damage",
                        return_image=False)
        assert not r.is_error
        txt = (await c.read_resource("fea://results/summary")).contents[0].text
        assert "vetknuti" in txt and "equilibrium" in txt
        # saving with results and reopening
        await s.ok("fea_study_save", path=str(tmp_path / "v1.fea"))
        await s.ok("fea_study_open", path=str(tmp_path / "v1.fea"), workdir=str(tmp_path / "w2"))
        summ2 = await s.ok("fea_results_summary")
        assert abs(summ2["steps"][0]["max_displacement_mm"]["value"] - s0["max_displacement_mm"]["value"]) < 1e-9
    run(tmp_path, body)
    if not qt_before:
        assert "PySide6" not in sys.modules


@pytest.mark.slow
def test_e2e_bolted_joint(tmp_path):
    """Lap joint M10: recognition of the bolt from the names, preload, calculation (+ the preload state), assessment."""
    from fea_core.geometry import write_step
    from verification.joint_models import lap_joint
    step = tmp_path / "lap.step"
    write_step(lap_joint(), step)

    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(step), workdir=str(tmp_path / "wd"), material="Ocel S235")
        d = await s.ok("fea_joints_detect")
        assert len(d["bolts"]) == 1 and d["bolts"][0]["plates"] == ["Deska A", "Deska B"]
        name = d["bolts"][0]["name"]
        await s.ok("fea_joints_update", name=name, changes={"preload_mode": "force", "preload": 15000})
        assert "nknown" in await s.err("fea_joints_update", name=name, changes={"barva": 1})
        await s.ok("fea_constraint_add_fixed", name="B_konec", selection={"part": "Deska B", "x": 160})
        await s.ok("fea_load_add_force", case_name="Tah", name="A_konec", selection={"part": "Deska A", "x": 0},
                   force=[-5000, 0, 0])
        await s.ok("fea_mesh_generate", global_size=4)
        r = await s.ok("fea_solve_run")
        assert [x["step"] for x in r["steps"]] == ["Tah", "Bolt preload"]
        a = await s.ok("fea_joints_assess")
        b = a["bolts"][0]
        assert b["status"] == "PASS" and 0 < b["utilization"] < 1
        shear = next(x for x in b["checks"] if x["check"].startswith("EN 1993-1-8: shear F_v"))
        assert shear["detail"].startswith("5000 /")                    # the whole force goes through a single bolt
    run(tmp_path, body)


@pytest.mark.slow
def test_hotspot_classification(tmp_path, steps):
    """A hole in a tensioned plate (symmetry = sliding constraints) = the real concentration; the inner corner of the angle = singularity."""
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(steps["plate"]), workdir=str(tmp_path / "p"), material="Ocel S355")
        await s.ok("fea_constraint_add_displacement", name="sym_x", selection={"x": 0}, ux=0.0)
        await s.ok("fea_constraint_add_displacement", name="sym_y", selection={"y": 0}, uy=0.0)
        await s.ok("fea_constraint_add_displacement", name="sym_z", selection={"z": 0}, uz=0.0)
        await s.ok("fea_load_add_pressure", case_name="Tah", name="konec", selection={"x": 60}, pressure=-50.0)
        await s.ok("fea_mesh_generate", global_size=2.0, grading=[{"selection": {"radius": 5}, "size": 0.8}])
        await s.ok("fea_solve_run")
        top = (await s.ok("fea_results_hotspots", top_n=2))["hotspots"][0]
        assert abs(((top["xyz_mm"][0] - 30) ** 2 + (top["xyz_mm"][1] - 20) ** 2) ** 0.5 - 5) < 0.1   # na otvoru
        assert top["classification"] == "probably real concentration", top
        assert any("curved" in r for r in top["reasons"])
        await s.ok("fea_study_new", step_path=str(steps["bracket"]), workdir=str(tmp_path / "b"), material="Ocel S355")
        await s.ok("fea_constraint_add_fixed", name="stena", selection={"x": 0})
        await s.ok("fea_load_add_force", case_name="F", name="konec", selection={"x": 80}, force=[0, 0, -2000])
        await s.ok("fea_mesh_generate", global_size=3.0)
        await s.ok("fea_solve_run")
        top = (await s.ok("fea_results_hotspots", top_n=3))["hotspots"][0]
        assert top["classification"] == "singularity" and any("concave" in r for r in top["reasons"]), top
        assert abs(top["xyz_mm"][2] - 8) < 1e-6 and top["xyz_mm"][0] < 12                 # inner corner x = 8, z = 8
    run(tmp_path, body)
