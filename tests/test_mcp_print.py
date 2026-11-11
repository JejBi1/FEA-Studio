"""MCP tools of 3D printing (F4): fea_print_get / _set / _creep / _props / _calibrate / _render_shell.

Without a ccx calculation; the render of the section of the shell meshes (Gmsh, coarse mesh of the cantilever – several seconds).
"""
import json
import struct

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from fea_core.mcp.server import build_server  # noqa: E402

PLA = "Tisk PLA Bambu Basic"


@pytest.fixture(scope="module")
def beam_step(tmp_path_factory):
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    d = tmp_path_factory.mktemp("print_geo")
    p = d / "beam.step"
    write_step([Part("Drzak", box(0, 0, 0, 60, 20, 12)), Part("Patka", box(60, 0, 0, 10, 20, 12))], p)
    return p


class S:
    def __init__(self, c):
        self.c = c

    async def call(self, tool, **args):
        r = await self.c.call_tool(tool, args)
        txt = r.content[0].text
        if r.is_error:
            return {"__error__": txt}
        return json.loads(txt)

    async def ok(self, tool, **args):
        d = await self.call(tool, **args)
        assert "__error__" not in d, f"{tool}: {d}"
        return d

    async def err(self, tool, **args) -> str:
        d = await self.call(tool, **args)
        assert "__error__" in d, f"{tool} měl skončit chybou: {d}"
        return d["__error__"]


def run(tmp_path, body):
    async def main():
        srv = build_server(base_dir=tmp_path / "mcp")
        async with Client(srv, mode="legacy") as c:
            await body(S(c), c, srv.fea_session)
    anyio.run(main)


async def _study(s, beam_step, tmp_path):
    await s.ok("fea_study_new", step_path=str(beam_step), workdir=str(tmp_path / "wd"))
    await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
    await s.ok("fea_materials_assign", part="Drzak", material_name=PLA)


def test_print_set_get_validation(tmp_path, beam_step):
    async def body(s, c, sess):
        await _study(s, beam_step, tmp_path)
        d = await s.ok("fea_print_set", part="Drzak", settings={"infill": 20, "walls": 3})
        assert d["print"]["infill"] == 0.2 and d["print"]["walls"] == 3 and "notes" in d and "warning" not in d
        d = await s.ok("fea_print_set", part="Drzak", settings={"infill_percent": 35})
        assert d["print"]["infill"] == 0.35 and "notes" not in d
        d = await s.ok("fea_print_set", part="Drzak", settings={"infill": 0.5})
        assert d["print"]["infill"] == 0.5 and "notes" not in d
        assert "build_dir" in await s.err("fea_print_set", part="Drzak", settings={"build_dir": [0, 1]})
        assert "build_dir" in await s.err("fea_print_set", part="Drzak", settings={"build_dir": "z"})
        assert "nonzero" in await s.err("fea_print_set", part="Drzak", settings={"build_dir": [0, 0, 0]})
        assert "integer" in await s.err("fea_print_set", part="Drzak", settings={"walls": 2.5})
        assert "0–1" in await s.err("fea_print_set", part="Drzak", settings={"infill": 150})
        assert "unknown print setting" in await s.err("fea_print_set", part="Drzak", settings={"perimetry": 3})
        assert "part" in await s.err("fea_print_set", settings={"walls": 3})
        d = await s.ok("fea_print_set", criterion="tsai_hill")             # only the criterion, without a part
        assert d["criterion"] == "tsai_hill" and "part" not in d
        d = await s.ok("fea_print_set", part="Patka", settings={"walls": 4})
        assert "warning" in d                                              # a steel part
        d = await s.ok("fea_print_set", part="*", settings={"build_dir": [0, 1, 0]})
        assert list(d["parts"]) == ["Drzak"] and d["parts"]["Drzak"]["build_dir"] == [0.0, 1.0, 0.0]
        g = await s.ok("fea_print_get")
        assert g["printed_parts"] == ["Drzak"] and g["criterion"] == "tsai_hill"
        dr = g["parts"]["Drzak"]
        assert dr["printed"] and dr["shell_thickness_mm"]["walls"] == pytest.approx(3 * 0.42)
        assert dr["core"]["infill"] == 0.5 and dr["core"]["E_xy"] < dr["solid"]["E_xy"]
        assert dr["perimeter_line"]["E_along"] > dr["perimeter_line"]["E_across"]      # PLA: linie 0,8
        assert not g["parts"]["Patka"]["printed"]
        assert g["materials"][PLA]["family"] == "PLA" and g["materials"][PLA]["props"]["source"]
        g1 = await s.ok("fea_print_get", part="Drzak")
        assert list(g1["parts"]) == ["Drzak"]
    run(tmp_path, body)


def test_print_creep_and_props(tmp_path, beam_step):
    async def body(s, c, sess):
        await _study(s, beam_step, tmp_path)
        await s.ok("fea_loadcase_create", name="Trvalé")
        assert "Unknown" in await s.err("fea_print_creep", enabled=True, cases=["neni"])
        assert "positive" in await s.err("fea_print_creep", hours=-5)
        d = await s.ok("fea_print_creep", enabled=True, hours=8760, temperature=30, cases=["Trvalé"])
        assert d["creep"] == {"enabled": True, "hours": 8760.0, "temperature": 30.0, "cases": ["Trvalé"]}
        f = d["factors"][PLA]
        assert 0 < f["stiffness"] < 1 and f["valid"]
        e = await s.ok("fea_print_creep", temperature=60)
        assert e["errors"] and "outside the validity" in e["errors"][0]
        await s.ok("fea_print_creep", temperature=23)
        assert sess.study.printing.creep.hours == 8760.0                  # the ones not given stay unchanged
        # the print properties of the material
        assert "family" in await s.err("fea_print_props", material=PLA, overrides={"family": "PETG"})
        assert "Unknown print properties" in await s.err("fea_print_props", material=PLA, overrides={"line_ratio": 0.7})
        assert "is not printed" in await s.err("fea_print_props", material="Ocel S235", overrides={"creep_a": 0.1})
        assert "positive" in await s.err("fea_print_props", material=PLA, overrides={"sn_k": 0})
        d = await s.ok("fea_print_props", material=PLA, overrides={"line_E_ratio": 0.7, "creep_a": 0.05,
                                                                   "creep_T": [[23, 1, 1], [40, 0.9, 2.5]]})
        assert d["props"]["line_E_ratio"] == 0.7 and d["overrides"]["creep_a"] == 0.05
        assert d["defaults"]["line_E_ratio"] == 0.8
        assert sess.study.printing.materials[PLA]["creep_T"] == [[23.0, 1.0, 1.0], [40.0, 0.9, 2.5]]
        g = await s.ok("fea_print_get")
        assert g["materials"][PLA]["overrides"]["line_E_ratio"] == 0.7
        assert g["materials"][PLA]["creep_factors"]["valid"]
        d = await s.ok("fea_print_props", material=PLA, reset=True)
        assert d["overrides"] == {} and d["props"]["line_E_ratio"] == 0.8
        assert PLA not in sess.study.printing.materials
    run(tmp_path, body)


def test_print_calibrate(tmp_path, beam_step):
    from fea_core.printing.calibrate import TensileTest, calibrate
    from fea_core.printing.props import props_of

    async def body(s, c, sess):
        await _study(s, beam_step, tmp_path)
        tests = [{"name": "XY-1", "direction": "xy", "E": 2900, "Rm": 44, "infill": 100},
                 {"name": "XY-2", "direction": "xy", "E": 3000, "Rm": 46, "infill": 1.0},
                 {"name": "Z-1", "direction": "z", "E": 2500, "Rm": 30}]
        assert "direction" in await s.err("fea_print_calibrate", tests=[{"direction": "q", "E": 1}],
                                          base_material=PLA)
        assert "E or Rm" in await s.err("fea_print_calibrate", tests=[{"direction": "xy"}], base_material=PLA)
        assert "unknown items" in await s.err("fea_print_calibrate", tests=[{"E": 1, "modul": 3}], base_material=PLA)
        assert "is not printed" in await s.err("fea_print_calibrate", tests=tests, base_material="Ocel S235")
        d = await s.ok("fea_print_calibrate", tests=tests, base_material=PLA, name="PLA moje",
                       assign_parts=["*"])
        assert d["assigned_parts"] == ["Drzak"] and d["material"]["name"] == "PLA moje"
        assert len(d["rows"]) == 3 and d["summary"]["n_xy"] == 2 and d["summary"]["n_z"] == 1
        st = sess.study
        assert st.material_of("Drzak").name == "PLA moje" and len(st.printing.tests) == 3
        ref = calibrate(st.materials[PLA], [TensileTest.from_dict({**t, "infill": 1.0}) for t in tests], "x",
                        props=props_of(st, st.materials[PLA]))
        assert st.materials["PLA moje"].E == pytest.approx(ref.material.E, rel=1e-12)
        assert st.materials["PLA moje"].E_z == pytest.approx(ref.material.E_z, rel=1e-12)
        g = await s.ok("fea_print_get", part="Drzak")
        assert g["tests"] == 3 and g["parts"]["Drzak"]["material"] == "PLA moje"
        assert "built-in" in await s.err("fea_print_calibrate", tests=tests, base_material=PLA, name=PLA)
    run(tmp_path, body)


def test_print_render_shell_png(tmp_path, beam_step):
    async def body(s, c, sess):
        await _study(s, beam_step, tmp_path)
        await s.ok("fea_print_set", part="Drzak", settings={"walls": 2, "infill_percent": 20})
        await s.ok("fea_mesh_generate", global_size=3.0)
        out = tmp_path / "shell.png"
        r = await c.call_tool("fea_print_render_shell", {"output_path": str(out), "axis": "x", "position": 30,
                                                          "width": 900, "height": 600})
        assert not r.is_error, r.content[0].text
        assert [x.type for x in r.content] == ["text", "image"]
        data = out.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", data[16:24]) == (900, 600)
        meta = json.loads(r.content[0].text)
        fr = meta["cut_area_by_shell_fraction"]
        assert "1" in fr and "0" in fr                                   # shell and core in the section
        assert meta["cut_area_mm2"] == pytest.approx(20 * 12, rel=1e-6)    # section x = 30 only through the part Drzak
        assert 0 < meta["shell_fraction_of_printed_cut"] < 1
        # section through both parts (an unprinted foot in grey), default position and view, without an image
        d = await s.ok("fea_print_render_shell", axis="y", return_image=False,
                       output_path=str(tmp_path / "shell_y.png"))
        assert "-1" in d["cut_area_by_shell_fraction"] and (tmp_path / "shell_y.png").exists()
        assert "outside" in await s.err("fea_print_render_shell", axis="x", position=500)
        assert "axis" in await s.err("fea_print_render_shell", axis="q")
    run(tmp_path, body)


async def _plastic_study(s, beam_step, tmp_path):
    """A plastic "for printing" without the print model (without E_z) + a constraint and a load."""
    await s.ok("fea_study_new", step_path=str(beam_step), workdir=str(tmp_path / "wd"))
    await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
    await s.ok("fea_materials_assign", part="Drzak", material_name="PLA tisk XY (orientačně)")
    await s.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
    await s.ok("fea_load_add_force", case_name="A", name="F", selection={"x": 70}, force=[0, 0, -10])


def test_readiness_warns_plastic_without_print_model(tmp_path, beam_step):
    async def body(s, c, sess):
        await _plastic_study(s, beam_step, tmp_path)
        rd = await s.ok("fea_check_readiness")
        assert any("without print properties" in w and "Drzak" in w for w in rd["warnings"])
    run(tmp_path, body)


@pytest.mark.slow
def test_solve_keeps_study_warnings_with_ready_mesh(tmp_path, beam_step):
    """A calculation with a finished mesh (MCP always passes it) does not discard the warnings of the study check (previously warnings = [])."""
    async def body(s, c, sess):
        await _plastic_study(s, beam_step, tmp_path)
        await s.ok("fea_mesh_generate", global_size=4)
        r = await s.ok("fea_solve_run", backend="ccx", timeout_s=300)
        assert any("without print properties" in w and "Drzak" in w for w in r["warnings"])
    run(tmp_path, body)
