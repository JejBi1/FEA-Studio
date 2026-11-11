"""MCP tools of the 2D calculation by a section (fea_twod_*) and of the shell model (fea_shell_*) + sections of the protocol.

Fast: settings, validation, recognition of the plate, readiness. Slow: A1 (tube Lamé as axisymmetric) and Sh1 (cantilever
from a plate) through MCP – calculation, results, probes, segment, linearisation, images, errors of the 3D tools, the protocol.
"""
import json
import math

import pytest

pytest.importorskip("mcp")

from tests.test_mcp import png_size, run  # noqa: E402

TOOLS = {"fea_twod_set", "fea_twod_results", "fea_twod_probe", "fea_twod_line", "fea_twod_linearize",
         "fea_twod_render", "fea_shell_set", "fea_shell_detect", "fea_shell_results", "fea_shell_probe",
         "fea_shell_render"}


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """Study JSON: A1 (quarter of the tube V4, internal pressure) and Sh1 (plate 200×40×5 clamped, force at the end)."""
    import verification.cases  # noqa: F401 – order of the imports (twod_cases ↔ cases)
    from verification.twod_cases import a1_build
    from verification.shell_cases import sh1_build
    d = tmp_path_factory.mktemp("reduced")
    (d / "a1").mkdir()
    (d / "sh1").mkdir()
    st, _ = a1_build(d / "a1")
    st.twod.enabled = False                          # switched on by the tool
    st.save(d / "a1" / "study.json")
    st, _ = sh1_build(d / "sh1")
    st.shell.enabled = False
    st.save(d / "sh1" / "study.json")
    return {"a1": d / "a1" / "study.json", "sh1": d / "sh1" / "study.json"}


def test_reduced_settings(tmp_path, models):
    async def body(s, c):
        names = {t.name for t in (await c.list_tools()).tools}
        assert TOOLS <= names
        await s.ok("fea_study_open", path=str(models["a1"]), workdir=str(tmp_path / "wd"))
        assert "kind" in await s.err("fea_twod_set", kind="xx")
        assert "parallel" in await s.err("fea_twod_set", kind="axisym", axis=[0, 0, 1], xdir=[0, 0, 2])
        r = await s.ok("fea_twod_set", kind="axisym", origin=[0, 0, 0], axis=[0, 0, 1], xdir=[1, 1, 0], size=1.0)
        e1 = r["frame"]["e1 (2D x)"]
        assert abs(e1[0] - math.sqrt(0.5)) < 1e-4 and "OUTSIDE the symmetry planes" in r["hint_xdir"]
        rd = await s.ok("fea_check_readiness")
        assert rd["ready"] and "2D section analysis" in rd["note"] and rd["rigid_body"] is None, rd
        assert "2D calculation" in await s.err("fea_shell_set")
        assert "results" in (await s.err("fea_twod_results")).lower()
        await s.ok("fea_twod_set", enabled=False)
        st = await s.ok("fea_study_status")
        assert st is not None
        await s.ok("fea_study_open", path=str(models["sh1"]), workdir=str(tmp_path / "wd2"))
        det = await s.ok("fea_shell_detect")
        p = det["parts"][0]
        assert p["is_sheet"] and abs(p["thickness_mm"] - 5) < 1e-6 and p["kind"] == "plate"
        assert "Unknown shell reference surface position" in await s.err("fea_shell_set", offset="xx")
        assert "does not exist" in await s.err("fea_shell_set", parts={"Nic": 2})
        r = await s.ok("fea_shell_set", parts={"Deska": None})
        assert r["shell"]["enabled"] and "readiness_errors" not in r
        rd = await s.ok("fea_check_readiness")
        assert rd["ready"] and "Shell model" in rd["note"], rd
        assert "field" in await s.err("fea_shell_results", field="xx")
    run(tmp_path, body)


@pytest.mark.slow
def test_e2e_twod_tube(tmp_path, models):
    """A1: tube ri 20 / ro 40, p = 100 MPa as axisymmetric – σθ at ri according to Lamé (within 2 %), reactions, linearisation."""
    import docx
    P4, RI, RO = 100.0, 20.0, 40.0
    a = P4 * RI ** 2 / (RO ** 2 - RI ** 2)
    sth_i = a * (1 + RO ** 2 / RI ** 2)

    async def body(s, c):
        await s.ok("fea_study_open", path=str(models["a1"]), workdir=str(tmp_path / "wd"))
        await s.ok("fea_twod_set", kind="axisym", origin=[0, 0, 0], axis=[0, 0, 1], xdir=[1, 1, 0], size=1.0)
        r = await s.ok("fea_solve_run", timeout_s=600)
        assert r["twod"]["kind"] == "axisymmetric problem" and "fea_twod_results" in r["next"]
        tr = await s.ok("fea_twod_results", field="st")
        res = tr["result"]
        assert res["equilibrium"]["ok"] and abs(res["field_max"]["value"] - sth_i) / sth_i < 0.02, res
        assert "WHOLE CIRCUMFERENCE" in tr["reactions"]
        zc = 10.0
        pr = await s.ok("fea_twod_probe", point=[[RI, zc], [RI * math.sqrt(0.5), RI * math.sqrt(0.5), zc]])
        st1, st2 = pr["probes"][0]["stress_MPa"]["stheta"], pr["probes"][1]["stress_MPa"]["stheta"]
        assert abs(st1 - sth_i) / sth_i < 0.02 and abs(st1 - st2) < 1e-6 * sth_i      # 3D point = the same place
        ln = await s.ok("fea_twod_line", p0=[RI, zc], p1=[RO, zc], field="st", n=11)
        assert ln["values"][0] > ln["values"][-1] > 0 and ln["points_outside"] == 0
        lin = await s.ok("fea_twod_linearize", p0=[RI, zc], p1=[RO, zc])
        mem = P4 * RI / (RO - RI)                     # σθ membrane = p·ri/t (equilibrium of the section)
        assert abs(lin["Pm_MPa"]["local"][1] - mem) / mem < 0.02, lin
        rr = await s.raw("fea_twod_render", field="mises", output_path=str(tmp_path / "t.png"))
        assert not rr.is_error and [x.type for x in rr.content] == ["text", "image"]
        assert png_size(tmp_path / "t.png")[0] > 300
        assert "2D section" in await s.err("fea_results_render_image")
        assert "2D section" in await s.err("fea_results_probe", point=[RI, 0, zc])
        assert "2D section" in await s.err("fea_results_hotspots")
        summ = await s.ok("fea_results_summary", step=None)
        assert summ["twod"]["steps"] and summ["steps"] == []
        rep = await s.ok("fea_report_word", output_path=str(tmp_path / "a1.docx"), language="en")
        assert rep["figures"] >= 2
        await s.ok("fea_study_save", path=str(tmp_path / "a1.fea"))
        await s.ok("fea_study_open", path=str(tmp_path / "a1.fea"), workdir=str(tmp_path / "w2"))
        tr2 = await s.ok("fea_twod_results", field="st")
        assert abs(tr2["result"]["field_max"]["value"] - res["field_max"]["value"]) < 1e-6 * sth_i
    run(tmp_path, body)
    d = docx.Document(str(tmp_path / "a1.docx"))
    text = "\n".join(p.text for p in d.paragraphs) + "\n".join(c.text for t in d.tables for r in t.rows for c in r.cells)
    heads = [p.text for p in d.paragraphs if p.style.name.startswith(("Heading", "Title"))]
    assert heads[0].startswith("2D FEM section analysis (axisymmetric problem)")
    assert any("2D section analysis" in h for h in heads) and any("– 2D analysis" in h for h in heads)
    assert "for the full circumference" in text and "CAX6" in text and "hoop stress σθ" in text


@pytest.mark.slow
def test_e2e_shell_cantilever(tmp_path, models):
    """Sh1: plate 200×40×5 clamped, force 100 N at the end – σx of the top / bottom face = ±M/W at the middle, reactions, images."""
    import docx
    L, B, T, F = 200.0, 40.0, 5.0, 100.0
    W = B * T ** 2 / 6

    async def body(s, c):
        await s.ok("fea_study_open", path=str(models["sh1"]), workdir=str(tmp_path / "wd"))
        await s.ok("fea_shell_set", parts={"Deska": None})
        r = await s.ok("fea_solve_run", timeout_s=600)
        assert r["shell"]["elements"] > 0 and "fea_shell_results" in r["next"]
        sr = await s.ok("fea_shell_results", step=0, field="sxx", layer="top")
        res = sr["result"]
        assert res["equilibrium"]["ok"] and abs(res["reactions"]["vetknuti"]["force_N"][2] - F) < 0.1
        assert sr["parts"]["Deska"]["thickness"] == T
        top = await s.ok("fea_shell_probe", point=[L / 2, B / 2, T / 2], field="sxx", layer="top")
        bot = await s.ok("fea_shell_probe", point=[L / 2, B / 2, T / 2], field="sxx", layer="bot")
        ref = F * L / 2 / W
        assert abs(top["probes"][0]["value"] - ref) / ref < 0.02 and abs(bot["probes"][0]["value"] + ref) / ref < 0.02
        mid = await s.ok("fea_shell_probe", point=[L / 2, B / 2, T / 2], field="membrane", layer="mid")
        assert mid["probes"][0]["value"] < 0.05 * ref
        rr = await s.raw("fea_shell_render", field="mises", layer="top", deform=True,
                         output_path=str(tmp_path / "s.png"))
        assert not rr.is_error and png_size(tmp_path / "s.png")[0] > 300
        assert "shell model" in await s.err("fea_results_render_image")
        assert (await s.ok("fea_results_summary", step=0))["shell"]["steps"][0]["imbalance"] < 1e-3
        rep = await s.ok("fea_report_word", output_path=str(tmp_path / "sh1.docx"), language="en")
        assert rep["figures"] >= 3
    run(tmp_path, body)
    d = docx.Document(str(tmp_path / "sh1.docx"))
    text = "\n".join(p.text for p in d.paragraphs) + "\n".join(c.text for t in d.tables for r in t.rows for c in r.cells)
    heads = [p.text for p in d.paragraphs if p.style.name.startswith(("Heading", "Title"))]
    assert heads[0].startswith("FEM shell analysis") and any("Shell model" in h for h in heads)
    assert "Stress by layer and part" in text and "membrane" in text and "C3D15" in text
    assert "Shell model parts" in text and "deska" in text
