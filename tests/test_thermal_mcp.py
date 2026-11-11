"""Heat through the MCP server, REST, the Word protocol and the CLI.

Fast: the heat tools through the MCP protocol (validation, replacement of the conditions, shared selections, readiness of a study with heat only,
the mesh signature, the image of the model). Slow: a bar 100×10×10, λ = 50, T = 100 °C at x = 0, convection h = 500 / 20 °C
at x = 100 → T(100) = (k/L·T0 + h·T∞)/(k/L + h) = 60 °C (k/L = 500 W/(m²·K)); thermal stress of a free bar
(linear field → no stress, ux(L) = α·(T̄ − T_ref)·L); Word protocols; CLI run + report.
"""
import json
import struct
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from fea_core.mcp.server import build_server  # noqa: E402

THERMAL_TOOLS = {"fea_thermal_case_create", "fea_thermal_case_delete", "fea_thermal_add_temperature",
                 "fea_thermal_add_heat_flux", "fea_thermal_add_convection", "fea_thermal_add_radiation",
                 "fea_thermal_add_heat_generation", "fea_thermal_settings", "fea_thermal_list", "fea_thermal_remove",
                 "fea_load_add_temperature"}
STEEL = {"E": 210000, "nu": 0.3, "rho": 7.85e-9, "Re": 235, "k": 50.0, "alpha": 12e-6, "cp": 460.0}


@pytest.fixture(scope="module")
def bar_step(tmp_path_factory):
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    p = tmp_path_factory.mktemp("th_geo") / "tyc.step"
    write_step([Part("Tyc", box(0, 0, 0, 100, 10, 10))], p)
    return p


class S:
    def __init__(self, client, srv):
        self.c, self.srv = client, srv

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


def run(tmp_path, body):
    async def main():
        srv = build_server(base_dir=tmp_path / "mcp")
        async with Client(srv, mode="legacy") as c:
            await body(S(c, srv), c)
    anyio.run(main)


def png_size(path: Path) -> tuple[int, int]:
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "není PNG"
    return struct.unpack(">II", data[16:24])


async def bar_study(s, bar_step, tmp_path, name="Tyč teplo"):
    await s.ok("fea_study_new", step_path=str(bar_step), name=name, workdir=str(tmp_path / "wd"))
    await s.ok("fea_materials_assign", part="*", material_name="Ocel (teplo)", custom_props=STEEL)


# ===================================================================== fast tests

def test_thermal_tools_protocol(tmp_path, bar_step):
    async def body(s, c):
        names = {t.name for t in (await c.list_tools()).tools}
        assert THERMAL_TOOLS <= names, THERMAL_TOOLS - names
        prompts = {p.name for p in (await c.list_prompts()).prompts}
        assert "fea_setup_thermal_analysis" in prompts
        p = await c.get_prompt("fea_setup_thermal_analysis", {"goal": "chladič", "thermal_stress": "true"})
        txt = p.messages[0].content.text
        assert "fea_thermal_add_convection" in txt and "thermal stress" in txt and "chladič" in txt

        await bar_study(s, bar_step, tmp_path)
        cat = await s.ok("fea_materials_list", query="S235")
        assert cat["materials"][0]["k_W_mK"] > 0 and cat["materials"][0]["alpha_1_K"] == 12e-6
        assert "10⁻⁶/K" in await s.err("fea_materials_assign", part="*", material_name="X",
                                         custom_props={"E": 1, "nu": 0.3, "alpha": 12})
        assert "kJ" in await s.err("fea_materials_assign", part="*", material_name="X",
                                   custom_props={"E": 1, "nu": 0.3, "cp": 0.46})
        d = await s.ok("fea_materials_assign", part="*", material_name="Ocel (teplo)", custom_props={"lambda": 50})
        assert d["assigned"]["material"]["k_W_mK"] == 50
        rd = await s.ok("fea_check_readiness")
        assert not rd["ready"] and any("no load case or thermal case" in e for e in rd["errors"])

        # thermal state and boundary conditions
        d = await s.ok("fea_thermal_case_create", name="Ohřev")
        assert d["created"] and d["thermal_case"]["transient"] is False
        d = await s.ok("fea_thermal_add_temperature", case="Ohřev", selection={"x": 0}, value_C=100)
        assert d["bc"]["name"] == "temperature_1" and d["selection"]["area_mm2"] == 100.0
        d = await s.ok("fea_thermal_add_convection", case="Ohřev", selection={"x": 100}, h_W_m2K=500, ambient_C=20)
        assert d["bc"]["name"] == "convection_1" and not d["replaced"]
        d = await s.ok("fea_thermal_add_convection", case="Ohřev", selection={"x": 100}, h_W_m2K=400)
        assert d["replaced"] and d["bc"]["name"] == "convection_1" and d["bc"]["h"] == 400   # the same faces → replacement
        d = await s.ok("fea_thermal_add_radiation", case="Ohřev", selection="convection_1", emissivity=0.8)
        assert not d["replaced"]                                              # radiation next to the convection
        lst = await s.ok("fea_thermal_list")
        bcs = lst["cases"][0]["bcs"]
        assert [b["kind"] for b in bcs] == ["fixed_temperature", "convection", "radiation"]
        assert bcs[1]["value_text"] == "h = 400 W/(m²·K), ambient 20 °C"
        await s.ok("fea_thermal_remove", case="Ohřev", name="convection_1", kind="radiation")
        assert "does not exist" in await s.err("fea_thermal_remove", case="Ohřev", name="convection_1", kind="radiation")
        # validace
        assert "exactly one" in await s.err("fea_thermal_add_heat_flux", case="Ohřev", selection={"y": 0})
        assert "exactly one" in await s.err("fea_thermal_add_heat_flux", case="Ohřev", selection={"y": 0},
                                            flux_W_m2=1, power_W=1)
        assert "positive" in await s.err("fea_thermal_add_convection", case="Ohřev", selection={"y": 0}, h_W_m2K=0)
        assert "(0; 1]" in await s.err("fea_thermal_add_radiation", case="Ohřev", selection={"y": 0}, emissivity=1.5)
        assert "absolute zero" in await s.err("fea_thermal_add_temperature", case="Ohřev", selection={"y": 0},
                                                value_C=-300)
        assert "does not exist" in await s.err("fea_thermal_add_heat_generation", case="Ohřev", part="Nic", power_W=1)
        assert "positive duration" in await s.err("fea_thermal_case_create", name="P", transient=True, time_s=0)
        assert "dt_s" in await s.err("fea_thermal_case_create", name="P", transient=True, time_s=10, dt_s=20)
        d = await s.ok("fea_thermal_case_create", name="Přechod", transient=True, time_s=60, dt_s=2, outputs=3)
        assert d["thermal_case"]["increments"] == {"count": 30, "dt_s": 2.0, "snapshot_every": 10}
        d = await s.ok("fea_thermal_add_heat_flux", case="Přechod", selection={"x": 100}, power_W=1.0)
        assert d["bc"]["kind"] == "heat_flow" and d["bc"]["name"] == "tok_1"
        d = await s.ok("fea_thermal_add_heat_generation", case="Přechod", part="Tyc", power_W=2.0)
        assert d["bc"]["name"] == "Tyc"
        d = await s.ok("fea_thermal_settings", reference_C=22, initial_C=30)
        assert d["reference_temperature_C"] == 22 and d["initial_temperature_C"] == 30
        d = await s.ok("fea_thermal_settings", initial_C="reference")
        assert "reference" in d["initial_temperature_C"]

        # a study with heat only: ready without constraints, without the check of the rigid motions
        await s.ok("fea_thermal_case_delete", name="Přechod")
        rd = await s.ok("fea_check_readiness")
        assert rd["ready"], rd
        assert rd["rigid_body"] is None and "Heat-conduction-only" in rd["note"]
        st = await s.ok("fea_study_status")
        assert st["thermal"]["cases"][0]["name"] == "Ohřev" and st["materials"]["Ocel (teplo)"]["k_W_mK"] == 50

        # thermal load in statics
        assert "exactly one" in await s.err("fea_load_add_temperature", case_name="TN")
        assert "does not exist" in await s.err("fea_load_add_temperature", case_name="TN", thermal_case="Nic")
        assert "steady" in await s.err("fea_load_add_temperature", case_name="TN", thermal_case="Ohřev", time_s=5)
        assert "parts" in await s.err("fea_load_add_temperature", case_name="TN", thermal_case="Ohřev", parts=["Tyc"])
        d = await s.ok("fea_load_add_temperature", case_name="TN", thermal_case="Ohřev")
        assert d["case_created"] and d["load"]["description"].startswith("temperature field from thermal state 'Ohřev'")
        assert any("no supports" in w for w in d["warnings"])
        d = await s.ok("fea_load_add_temperature", case_name="TN", thermal_case="Ohřev")
        assert d["replaced"]
        lst = await s.ok("fea_load_list", case_name="TN")
        assert [l["kind"] for l in lst["load_cases"][0]["loads"]] == ["thermal"]
        assert "fea_load_remove" in await s.err("fea_thermal_case_delete", name="Ohřev")
        rd = await s.ok("fea_check_readiness")
        assert not rd["ready"] and any("support" in e for e in rd["errors"])
        await s.ok("fea_load_remove", case_name="TN", name="teplota")
        await s.ok("fea_loadcase_delete", name="TN")

        # shared selection: a constraint must not overwrite the faces of the thermal condition; the same faces are shared
        assert "thermal boundary condition" in await s.err("fea_constraint_add_fixed", name="temperature_1",
                                                          selection={"x": 100})
        await s.ok("fea_constraint_add_fixed", name="temperature_1", selection="temperature_1")
        await s.ok("fea_constraint_remove", name="temperature_1")
        assert "is in use" in await s.err("fea_selection_delete", name="convection_1")
        sl = await s.ok("fea_selection_list")
        assert any("thermal boundary condition" in u for x in sl["selections"] for u in x["used_by"])

        # signature of the mesh: the heat values and the thermal properties of the material do not invalidate the mesh; the new faces do
        sess = s.srv.fea_session
        sig = sess.mesh_signature()
        await s.ok("fea_thermal_add_convection", case="Ohřev", selection={"x": 100}, h_W_m2K=500)
        await s.ok("fea_thermal_settings", reference_C=20)
        await s.ok("fea_materials_assign", part="*", material_name="Ocel (teplo)", custom_props={"k": 45, "cp": 470})
        assert sess.mesh_signature() == sig
        await s.ok("fea_thermal_add_convection", case="Ohřev", selection={"y": 0}, h_W_m2K=5, name="bok")
        assert sess.mesh_signature() != sig
        await s.ok("fea_thermal_remove", case="Ohřev", name="bok")

        # image of the model with the thermal boundary conditions
        r = await s.ok("fea_model_render_image", output_path=str(tmp_path / "model_th.png"), thermal_case="Ohřev",
                       return_image=False)
        assert png_size(tmp_path / "model_th.png") == (1280, 720) and r["views"]
        assert "does not exist" in await s.err("fea_model_render_image", thermal_case="Nic")
    run(tmp_path, body)


# ===================================================================== e2e (mesh + ccx)

T_END = (500 * 100 + 500 * 20) / (500 + 500)          # 60 °C
Q_W = 500 * (100 - T_END) * 1e-4                      # k/L·ΔT·A = 2 W


@pytest.mark.slow
def test_e2e_bar_heat_and_thermal_stress(tmp_path, bar_step):
    import docx
    out1, out2 = tmp_path / "teplo.docx", tmp_path / "teplo_napeti.docx"

    async def body(s, c):
        await bar_study(s, bar_step, tmp_path)
        await s.ok("fea_thermal_case_create", name="Ohřev")
        await s.ok("fea_thermal_add_temperature", case="Ohřev", selection={"x": 0}, value_C=100, name="horke_celo")
        await s.ok("fea_thermal_add_convection", case="Ohřev", selection={"x": 100}, h_W_m2K=500, ambient_C=20,
                   name="chlazeni")
        assert (await s.ok("fea_check_readiness"))["ready"]
        m = await s.ok("fea_mesh_generate", global_size=2.5)
        assert m["solver_backend"].startswith("ccx")
        r = await s.ok("fea_solve_run", timeout_s=600)
        assert r["steps"] == [] and "mesh_generated" not in r
        th = r["thermal"][0]
        assert th["T_max_C"]["value"] == pytest.approx(100.0, abs=1e-6)
        assert th["T_min_C"]["value"] == pytest.approx(T_END, rel=1e-4)
        q = {f["name"]: f["Q_W"] for f in th["heat_flows_W"]}
        assert q["horke_celo"] == pytest.approx(Q_W, rel=1e-3) and q["chlazeni"] == pytest.approx(-Q_W, rel=1e-3)
        assert th["balance"]["ok"]
        p = await s.ok("fea_results_probe", point=[[100, 5, 5], [50, 5, 5]])
        assert p["thermal_case"] == "Ohřev"
        assert p["probes"][0]["T_C"] == pytest.approx(T_END, rel=1e-4)
        assert p["probes"][1]["T_C"] == pytest.approx((100 + T_END) / 2, rel=1e-4)
        assert p["probes"][1]["heat_flux_W_m2"][0] == pytest.approx(Q_W / 1e-4, rel=1e-3)
        summ = await s.ok("fea_results_summary")
        assert summ["steps"] == [] and summ["thermal"][0]["parts"][0]["T_max_C"] == pytest.approx(100.0, abs=1e-6)
        hs = await s.ok("fea_results_hotspots", top_n=2)          # heat only → the hottest places
        assert hs["field"] == "temperature" and hs["hotspots"][0]["T_C"] == pytest.approx(100.0, abs=1e-6)
        assert hs["hotspots"][0]["xyz_mm"][0] == 0 and any("horke_celo" in x for x in hs["hotspots"][0]["reasons"])
        assert hs["coldest"]["T_C"] == pytest.approx(T_END, rel=1e-4) and "notes" in hs
        img = tmp_path / "teplota.png"
        rr = await c.call_tool("fea_results_render_image", {"output_path": str(img), "field": "temperature",
                                                           "view": "multi", "width": 1000, "height": 600})
        assert not rr.is_error and [x.type for x in rr.content] == ["text", "image"]
        meta = json.loads(rr.content[0].text)
        assert meta["thermal_case"] == "Ohřev" and meta["deformation_scale"] == 0
        assert meta["surface_max"] == pytest.approx(100.0, abs=1e-6) and png_size(img) == (1000, 600)
        await s.ok("fea_results_render_image", output_path=str(tmp_path / "tok.png"), field="heat_flux",
                   return_image=False)
        assert png_size(tmp_path / "tok.png") == (1280, 720)
        assert "The results contain no mechanical" in await s.err("fea_results_render_image", field="mises")
        txt = (await c.read_resource("fea://results/summary")).contents[0].text
        assert "Thermal case 'Ohřev'" in txt and "chlazeni" in txt
        rep = await s.ok("fea_report_word", output_path=str(out1), author="Test", language="cs")
        assert rep["with_results"] and rep["figures"] == 2                 # model with conditions + temperature map
        # a change of the value of a thermal condition (the same faces) → the results are discarded, the mesh stays
        await s.ok("fea_thermal_add_convection", case="Ohřev", selection="chlazeni", h_W_m2K=500, ambient_C=20)
        assert "results" in (await s.err("fea_results_summary")).lower()
        r = await s.ok("fea_solve_run", timeout_s=600)
        assert "mesh_generated" not in r

        # thermal stress: a free bar (symmetry planes) with a linear field → no stress, ux(L) = α·(T̄ − 20)·L
        await s.ok("fea_constraint_add_displacement", name="sym_x", selection="horke_celo", ux=0.0)
        await s.ok("fea_constraint_add_displacement", name="sym_y", selection={"y": 0}, uy=0.0)
        await s.ok("fea_constraint_add_displacement", name="sym_z", selection={"z": 0}, uz=0.0)
        await s.ok("fea_load_add_temperature", case_name="Teplotní napětí", thermal_case="Ohřev")
        rd = await s.ok("fea_check_readiness")
        assert rd["ready"] and rd["rigid_body"]["ok"], rd
        r = await s.ok("fea_solve_run", timeout_s=600)
        assert "mesh_generated" in r                                        # new faces of the constraints (y = 0, z = 0)
        st0 = r["steps"][0]
        assert st0["temperature_C"]["max"] == pytest.approx(100.0, abs=1e-6)
        # a free body with T = a + b·x is stress-free: u_x = α(a·x + b·x²/2) − α·b·(y² + z²)/2; the planar constraint of the face
        # x = 0 prevents its curvature → stress only at the face (Saint-Venant), further ≈ 0 (E·α·ΔT = 201.6 MPa)
        assert 1 < st0["max_von_mises_MPa"]["value"] < 30 and st0["max_von_mises_MPa"]["at_mm"][0] < 10
        p = await s.ok("fea_results_probe", point=[[80, 5, 5], [100, 5, 5]], step="Teplotní napětí")
        assert p["probes"][0]["mises_MPa"] < 0.2
        a_, b_, al = 80.0, (T_END - 100) / 100, 12e-6
        mean_r2 = 2 * 100 / 3                                    # the average of y² + z² at the face 10×10
        ux = al * (a_ * 100 + b_ * 100 ** 2 / 2) - al * b_ * 50 / 2 + al * b_ * mean_r2 / 2
        assert p["probes"][1]["u_mm"][0] == pytest.approx(ux, rel=2e-3)
        assert p["probes"][1]["T_C"] == pytest.approx(T_END, rel=1e-4)
        await s.ok("fea_results_render_image", output_path=str(tmp_path / "t_mech.png"), field="temperature",
                   step=0, return_image=False)
        await s.ok("fea_results_render_image", output_path=str(tmp_path / "t_th.png"), field="temperature",
                   thermal_case="Ohřev", return_image=False)
        await s.ok("fea_report_word", output_path=str(out2), images=False, language="cs")
        await s.ok("fea_study_save", path=str(tmp_path / "tyc.fea"))
        await s.ok("fea_study_save", path=str(tmp_path / "tyc.json"))
        d = await s.ok("fea_study_open", path=str(tmp_path / "tyc.fea"), workdir=str(tmp_path / "w2"))
        assert d["results"]["thermal_cases"] == ["Ohřev"]
        summ = await s.ok("fea_results_summary", step=None)
        assert summ["thermal"][0]["T_min_C"]["value"] == pytest.approx(T_END, rel=1e-4)
    run(tmp_path, body)

    def text_of(path):
        d = docx.Document(str(path))
        heads = [p.text for p in d.paragraphs if p.style.name.startswith(("Heading", "Title"))]
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts), heads, len(d.inline_shapes)
    text, heads, nimg = text_of(out1)
    assert heads[0].startswith("Tepelný výpočet MKP") and nimg == 2
    assert "1.3 Tepelné stavy" in heads and "3.1 Tepelný stav „Ohřev“" in heads
    assert not any(h.startswith("3.2") for h in heads) and "Přehled výsledků statiky" not in text
    assert "předepsaná teplota | horke_celo | T = 100 °C" in text and "h = 500 W/(m²·K), okolí 20 °C" in text
    assert "λ [W/(m·K)]" in text and "Tepelná bilance: přívod 2 W" in text
    assert "dokonalém tepelném styku" in text and "Lineární statika" not in text
    text, heads, nimg = text_of(out2)
    assert heads[0].startswith("Pevnostní a tepelný výpočet MKP") and nimg == 0
    assert "3.1 Tepelný stav „Ohřev“" in heads and "3.2 Stav „Teplotní napětí“" in heads
    assert "teplotní pole z tepelného stavu „Ohřev“; teplota v modelu" in text
    assert "Teplotní napětí: přetvoření α·(T − T_ref)" in text

    # CLI: run (summary of the heat) and the report of a study with heat only
    from fea_core.cli import main
    from fea_core.study import Study
    stj = Study.load(tmp_path / "tyc.json")
    stj.load_cases, stj.constraints = [], []
    stj.save(tmp_path / "jen_teplo.json")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["run", str(tmp_path / "jen_teplo.json"), "-o", str(tmp_path / "cli_run")]) == 0
    log = buf.getvalue()
    assert "Thermal state 'Ohřev' (steady-state): T 60 … 100 °C" in log and "heat 'chlazeni'" in log
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["report", str(tmp_path / "jen_teplo.json"), "-o", str(tmp_path / "cli.docx"), "--no-images",
                     "--workdir", str(tmp_path / "cli_rep"), "--lang", "cs"]) == 0
    text, heads, _ = text_of(tmp_path / "cli.docx")
    assert heads[0].startswith("Tepelný výpočet MKP") and "Tepelná bilance" in text


@pytest.mark.slow
def test_rest_thermal_tools(tmp_path, bar_step):
    """REST: the thermal tools from the MCP registry (the same schema), calls over HTTP."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from fea_core.rest import create_app
    c = TestClient(create_app(base_dir=tmp_path / "rest", gui_launcher=lambda p: 0),
                   base_url="http://localhost:8765")
    tools = {t["name"] for t in c.get("/api/tools").json()["tools"]}
    assert THERMAL_TOOLS <= tools

    def call(tool, status=200, **args):
        r = c.post(f"/api/tools/{tool}", json=args)
        assert r.status_code == status, (tool, r.text)
        return r.json()
    call("fea_study_new", step_path=str(bar_step), workdir=str(tmp_path / "wd"))
    call("fea_materials_assign", part="*", material_name="Ocel S235")
    call("fea_thermal_add_temperature", case="A", selection={"x": 0}, value_C=80)
    call("fea_thermal_add_convection", case="A", selection={"x": 100}, h_W_m2K=500)
    assert call("fea_check_readiness")["result"]["ready"]
    r = call("fea_solve_run", timeout_s=600)["result"]
    assert r["thermal"][0]["T_max_C"]["value"] == pytest.approx(80.0, abs=1e-6)
    img = call("fea_results_render_image", field="temperature", width=400, height=300)["images"][0]
    assert png_size_bytes(c.get(img["url"]).content) == (400, 300)
    assert "absolute zero" in call("fea_thermal_add_temperature", 400, case="A", selection={"x": 0},
                                     value_C=-400)["error"]


def png_size_bytes(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])
