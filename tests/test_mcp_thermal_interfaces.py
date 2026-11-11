"""MCP: contact conductance of interfaces (D1) and cavity radiation (D2) – tools, readiness check, solve with the heat
through the interface and the cavity faces, report."""
import pytest

pytest.importorskip("mcp")

from tests.test_mcp import run  # noqa: E402

K = 50.0
HC = 1000.0


@pytest.fixture(autouse=True)
def _no_gui(monkeypatch):
    """The tools must not be redirected to a possibly running GUI window (Dynamic IPC Bridge)."""
    monkeypatch.setattr("fea_core.ipc.is_gui_running", lambda *a, **k: False)


@pytest.fixture(scope="module")
def two_plates(tmp_path_factory):
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    d = tmp_path_factory.mktemp("mcp_th_ifc")
    p = d / "plates.step"
    write_step([Part("A", box(0, 0, 0, 50, 10, 10)), Part("B", box(50, 0, 0, 50, 10, 10))], p)
    gap = d / "gap.step"
    write_step([Part("Hot", box(0, 0, 0, 10, 20, 20)), Part("Cold", box(12, 0, 0, 10, 20, 20))], gap)
    return {"plates": p, "gap": gap}


async def _plates(s, geo, tmp_path):
    steel = {"E": 210000, "nu": 0.3, "rho": 7.85e-9, "Re": 235, "Rm": 360, "k": K, "alpha": 12e-6, "cp": 460.0}
    await s.ok("fea_study_new", step_path=str(geo["plates"]), workdir=str(tmp_path / "wd"))
    await s.ok("fea_materials_assign", part="*", material_name="Steel (heat)", custom_props=steel)
    await s.ok("fea_tools_enable", groups="thermal")


def test_contact_conductance_tools(tmp_path, two_plates):
    async def body(s, c):
        await _plates(s, two_plates, tmp_path)
        d = await s.ok("fea_thermal_add_contact_conductance", side_a="A", side_b="B", pressure_MPa=2.0)
        assert d["suggestion"]["h_W_m2K"] == d["conductance"]["h_W_m2K"] > 0
        assert d["separate_mesh_part"] == "B" and "warning" in d                     # no thermal state yet
        d = await s.ok("fea_thermal_add_contact_conductance", side_a="A", side_b="B", h_W_m2K=HC, name="joint")
        assert d["replaced"] and d["conductance"]["R_m2K_W"] == pytest.approx(1e-3) and "separate_mesh_part" not in d
        lst = await s.ok("fea_thermal_list")
        assert [r["name"] for r in lst["contact_conductances"]] == ["joint"]
        assert "side" in await s.err("fea_thermal_add_contact_conductance", side_a="A", side_b="A", h_W_m2K=10)
        assert "neither" in await s.err("fea_thermal_add_contact_conductance", side_a="A", side_b="X", h_W_m2K=10)
        assert "positive" in await s.err("fea_thermal_add_contact_conductance", side_a="A", side_b="B", h_W_m2K=-1)
        r = await s.ok("fea_check_readiness")
        assert any("no thermal case" in w for w in r["warnings"]), r
        await s.ok("fea_thermal_remove", case="", name="joint", kind="contact_conductance")
        assert "contact_conductances" not in await s.ok("fea_thermal_list")
        assert "does not exist" in await s.err("fea_thermal_remove", case="", name="joint", kind="contact_conductance")
    run(tmp_path, body)


def test_cavity_tools(tmp_path, two_plates):
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(two_plates["gap"]), workdir=str(tmp_path / "wd"))
        await s.ok("fea_tools_enable", groups="thermal")
        d = await s.ok("fea_thermal_add_cavity", case="Rad", selection={"x": 10}, emissivity=0.8, cavity="Gap",
                       closed=True, name="hot")
        assert d["case_created"] and "warning" in d                                  # one face of a closed cavity
        d = await s.ok("fea_thermal_add_cavity", case="Rad", selection={"x": 12}, emissivity=0.5, cavity="Gap",
                       closed=True, name="cold")
        assert d["cavity"]["faces"] == ["hot", "cold"] and d["cavity"]["emissivities"] == [0.8, 0.5]
        assert "cavity Gap, closed" in d["bc"]["value_text"]
        assert "interval" in await s.err("fea_thermal_add_cavity", case="Rad", selection="hot", emissivity=1.5)
        r = await s.ok("fea_check_readiness")
        assert not r["ready"] and any("dissipate" in e for e in r["errors"])          # closed cavity is not a heat sink
        await s.ok("fea_thermal_remove", case="Rad", name="cold", kind="cavity_radiation")
        assert [b["name"] for b in (await s.ok("fea_thermal_list"))["cases"][0]["bcs"]] == ["hot"]
    run(tmp_path, body)


@pytest.mark.slow
def test_conductance_and_cavity_solve_report(tmp_path, two_plates):
    """E2E: bonded interface A–B with h_c (series resistances) – heat through the interface in the results and the report."""
    async def body(s, c):
        await _plates(s, two_plates, tmp_path)
        await s.ok("fea_thermal_add_temperature", case="Heat", selection={"x": 0}, value_C=100, name="hot")
        await s.ok("fea_thermal_add_temperature", case="Heat", selection={"x": 100}, value_C=20, name="cold")
        await s.ok("fea_thermal_add_contact_conductance", side_a="A", side_b="B", h_W_m2K=HC, name="joint")
        assert (await s.ok("fea_check_readiness"))["ready"]
        await s.ok("fea_mesh_generate", global_size=4.0, timeout_s=300)
        await s.ok("fea_solve_run", timeout_s=600)
        summ = await s.ok("fea_results_summary", thermal_case="Heat")
        A = 100e-6
        Q = 80.0 / (2 * 0.05 / (K * A) + 1 / (HC * A))
        ifc = summ["thermal"][0]["balance"]["interfaces"][0]
        assert ifc["name"] == "joint" and abs(abs(ifc["Q_W"]) / Q - 1) < 1e-3, ifc
        out = tmp_path / "r.docx"
        await s.ok("fea_report_word", output_path=str(out), images=False)
        import docx
        text = "\n".join(p.text for p in docx.Document(str(out)).paragraphs)
        cells = "\n".join(cl.text for t in docx.Document(str(out)).tables for row in t.rows for cl in row.cells)
        assert "joint" in cells and "Heat through contact conductance interfaces" in text + cells
        assert "temperature jump" in text
    run(tmp_path, body)
