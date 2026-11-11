"""MCP tools of the mechanism (fea_motion_*): bodies from the CAD, a proposal of the joint from coaxial cylinders, check of the mobility, calculation
(Exudyn – without CalculiX), results, a graph PNG and the transfer to the FEM (load state with Inertia and a remote constraint).
"""
import pytest

pytest.importorskip("mcp")
pytest.importorskip("exudyn")

from tests.test_mcp import png_size, run  # noqa: E402

MOTION_TOOLS = {"fea_motion_body_add", "fea_motion_body_list", "fea_motion_body_remove", "fea_motion_joint_add",
                "fea_motion_joint_list", "fea_motion_joint_remove", "fea_motion_remove", "fea_motion_joints_from_cad",
                "fea_motion_force_add", "fea_motion_driver_add", "fea_motion_load_add", "fea_motion_settings",
                "fea_motion_check_dof", "fea_motion_run", "fea_motion_results", "fea_motion_render",
                "fea_motion_export_loads", "fea_motion_contact_add"}


@pytest.fixture(scope="module")
def lever_step(tmp_path_factory):
    from fea_core.geometry import write_step
    from tests.test_motion import _pin_parts
    p = tmp_path_factory.mktemp("motion_geo") / "paka.step"
    write_step(_pin_parts(), p)
    return p


def test_motion_tools_registered(tmp_path):
    async def body(s, c):
        names = {t.name for t in (await c.list_tools()).tools}
        assert MOTION_TOOLS <= names, MOTION_TOOLS - names
        p = await c.get_prompt("fea_setup_mechanism", {"goal": "páka"})
        txt = p.messages[0].content.text
        assert "fea_motion_joints_from_cad" in txt and "fea_motion_export_loads" in txt
    run(tmp_path, body)


def test_motion_lever_end_to_end(tmp_path, lever_step):
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(lever_step), workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
        await s.ok("fea_motion_body_add", name="rám", parts=["Deska"], ground=True)
        d = await s.ok("fea_motion_body_add", name="páka", parts=["Páka"])
        assert d["body"]["mass_kg"] == pytest.approx((105 * 100 + 3.14159265 * 25 * 10) * 7.85e-6, rel=1e-4)
        e = await s.err("fea_motion_body_add", name="jiné", parts=["Páka"])
        assert "already belong" in e
        d = await s.ok("fea_motion_joints_from_cad", accept=True)
        assert len(d["accepted"]) == 1 and d["suggestions"][0]["joint"]["kind"] == "revolute"
        dof = await s.ok("fea_motion_check_dof")
        assert dof["dof_free"] == 1 and dof["redundant"] == 0
        await s.ok("fea_motion_settings", kind="dynamic", t_end_s=0.1, outputs=20, gravity_mm_s2=[0, -9810, 0])
        assert "__error__" in await s.call("fea_motion_results")            # without a calculation
        r = await s.ok("fea_motion_run", timeout_s=300)
        assert r["equilibrium"]["ok"] and r["joints"][0]["max_force_N"] > 0
        jn = r["joints"][0]["joint"]
        res = await s.ok("fea_motion_results", series=[f"reakce:{jn}", "energie"], find_max=f"reakce:{jn}",
                         time_s=0.05)
        assert res["series"][f"reakce:{jn}"]["unit"] == "N" and res["maximum"]["value"] > 0
        assert res["instant"]["joints"][0]["joint"] == jn
        img = await c.call_tool("fea_motion_render", {"quantity": f"reakce:{jn}", "output_path": str(tmp_path / "r.png")})
        assert not img.is_error and png_size(tmp_path / "r.png") == (1280, 720)
        img = await c.call_tool("fea_motion_render", {"quantity": "location", "time_s": 0.1,
                                                      "output_path": str(tmp_path / "p.png"), "return_image": False})
        assert not img.is_error
        ex = await s.ok("fea_motion_export_loads", mode="time", times_s=[0.05], body="páka")
        lc = ex["load_cases"][0]
        assert lc.startswith("MBD t = ")
        status = await s.ok("fea_study_status")
        assert lc in str(status)                                             # the state is in the study
        # the .fea project carries the result of the mechanism (saving and opening – as in the GUI)
        import zipfile
        sv = await s.ok("fea_study_save", path=str(tmp_path / "paka.fea"))
        assert sv["with_motion"] and "results_motion.npz" in zipfile.ZipFile(tmp_path / "paka.fea").namelist()
        # the Word protocol has a Motion section (spec F8 6.4)
        pytest.importorskip("docx")
        rep = await s.ok("fea_report_word", output_path=str(tmp_path / "paka.docx"), images=False, language="en")
        import re
        xml = zipfile.ZipFile(rep["path"]).read("word/document.xml").decode("utf-8")
        txt = re.sub(r"<[^>]+>", "", xml)
        assert "Motion (mechanism)" in txt and jn in txt and "Transfer to FEA" in txt and lc in txt
        await s.ok("fea_study_open", path=str(tmp_path / "paka.fea"))
        res2 = await s.ok("fea_motion_results")
        assert res2["joints"][0]["max_force_N"] == pytest.approx(r["joints"][0]["max_force_N"], rel=1e-9)
        # a change of the mechanism discards the result
        await s.ok("fea_motion_settings", outputs=10)
        assert "__error__" in await s.call("fea_motion_results")
    run(tmp_path, body)


def test_motion_driver_and_errors(tmp_path, lever_step):
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(lever_step), workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
        await s.ok("fea_motion_body_add", name="rám", parts=["Deska"], ground=True)
        await s.ok("fea_motion_body_add", name="páka", parts=["Páka"])
        e = await s.err("fea_motion_joint_add", name="J", kind="hinge", body_a="rám", body_b="páka")
        assert "kind" in e
        d = await s.ok("fea_motion_joint_add", name="čep", kind="revolute", body_a="rám", body_b="páka",
                       faces_b={"part": "Páka", "type": "cylinder", "radius": 5})
        assert abs(abs(d["joint"]["axis"][2]) - 1) < 1e-9 and d["joint"]["selection_b"]
        e = await s.err("fea_motion_driver_add", joint="čep", quantity="angle", law="scurve", data=[1, 2])
        assert "S-curve" in e
        await s.ok("fea_motion_driver_add", joint="čep", quantity="angular_velocity", law="constant", data=[2.0])
        await s.ok("fea_motion_settings", kind="kinematic", t_end_s=0.5, outputs=10)
        r = await s.ok("fea_motion_run", timeout_s=300)
        assert r["drivers"][0]["unit"] == "N·mm" and r["power_balance"]["ok"]
        await s.ok("fea_motion_joint_remove", name="čep")
        lst = await s.ok("fea_motion_joint_list")
        assert lst["joints"] == [] and lst["drivers"] == []
    run(tmp_path, body)


def test_motion_flexible_and_contact_tools(tmp_path, lever_step):
    """A flexible body and contact (Chrono) through MCP: parameters, checks, removal (without a calculation)."""
    async def body(s, c):
        from fea_core.motion.chrono_run import available
        await s.ok("fea_study_new", step_path=str(lever_step), workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
        await s.ok("fea_motion_body_add", name="rám", parts=["Deska"], ground=True)
        e = await s.err("fea_motion_body_add", name="x", parts=["Deska"], ground=True, flexible=True)
        assert "rám" in e
        d = await s.ok("fea_motion_body_add", name="páka", parts=["Páka"], flexible=True, modes=6, zeta=0.01)
        assert d["body"]["flexible"]["modes"] == 6 and d["body"]["flexible"]["interface"] == "rbe2"
        e = await s.err("fea_motion_contact_add", body_a="páka", body_b="nic")
        assert "does not exist" in e
        d = await s.ok("fea_motion_contact_add", body_a="rám", body_b="páka", mu=0.2, deflection_mm=0.02)
        assert d["contacts"] == ["rám–páka"] and "chrono" in d.get("next", "solver chrono")
        st = await s.ok("fea_motion_settings", kind="dynamic")
        assert st["contacts"][0]["mu"] == 0.2 and st["flexible_bodies"] == ["páka"]
        if available():
            st = await s.ok("fea_motion_settings", solver="chrono")
            assert st["run"]["solver"] == "chrono"
            e = await s.err("fea_motion_run", timeout_s=60)            # a flexible body in Chrono is not possible
            assert "Craig–Bampton" in e
        else:
            e = await s.err("fea_motion_settings", solver="chrono")
            assert "chrono" in e
        await s.ok("fea_motion_remove", kind="contact", name="rám–páka")
        d = await s.ok("fea_motion_body_remove", name="páka")
        assert d["removed"] == "páka"
    run(tmp_path, body)
