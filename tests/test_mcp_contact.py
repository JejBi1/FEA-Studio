"""MCP tools of F8 – contacts, nonlinear stepping and asynchronous jobs (tools_contact, tools_jobs).

Fast tests through the MCP protocol in memory without a calculation; e2e K1 (patch test of the contact, ccx) through an asynchronous job is slow.
"""
import json
import time

import anyio
import pytest

pytest.importorskip("mcp")

from tests.test_mcp import Session, png_size, run  # noqa: E402,F401

CONTACT_TOOLS = {
    "fea_contact_detect", "fea_contact_set", "fea_contact_list", "fea_contact_remove", "fea_contact_check",
    "fea_contact_settings", "fea_contact_results", "fea_contact_render", "fea_analysis_set_sequence",
    "fea_results_history", "fea_job_start", "fea_job_status", "fea_job_cancel", "fea_job_result",
}


@pytest.fixture(scope="module")
def blocks(tmp_path_factory):
    """STEP: two blocks 20×20×10 on top of each other (A below, B above – contact at z = 10) and a block C with a gap 0.5 mm."""
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    d = tmp_path_factory.mktemp("contact_geo")
    p = d / "blocks.step"
    write_step([Part("A", box(0, 0, 0, 20, 20, 10)), Part("B", box(0, 0, 10, 20, 20, 10)),
                Part("C", box(0, 0, 20.5, 20, 20, 5))], p)
    return p


async def _study(s, blocks, tmp_path):
    await s.ok("fea_study_new", step_path=str(blocks), workdir=str(tmp_path / "wd"))
    await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
    await s.ok("fea_constraint_add_fixed", name="dno", selection={"z": 0})
    await s.ok("fea_load_add_pressure", case_name="p", name="vrch", selection={"z": 25.5}, pressure=10.0)


# ===================================================================== fast

def test_contact_tools_registered(tmp_path):
    async def body(s, c):
        names = {t.name for t in (await c.list_tools()).tools}
        assert CONTACT_TOOLS <= names, CONTACT_TOOLS - names
        prompts = {p.name for p in (await c.list_prompts()).prompts}
        assert {"fea_setup_contact", "fea_setup_vibration_test"} <= prompts
        txt = (await c.get_prompt("fea_setup_contact", {"goal": "press fit"})).messages[0].content.text
        assert "fea_contact_set" in txt and "fea_job_start" in txt and "μ_min" in txt
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert "fea_job_start" in tools["fea_solve_run"].description
    run(tmp_path, body)


def test_contact_detect_set_list_remove(tmp_path, blocks):
    async def body(s, c):
        await _study(s, blocks, tmp_path)
        d = await s.ok("fea_contact_detect", tol_mm=0.1)
        by = {(p["a"], p["b"]): p for p in d["pairs"]}
        # area = sum of the areas within the tolerance – the core also counts the side walls touching only at an edge (400 + 4·200)
        assert by[("A", "B")]["state"] == "contact" and by[("A", "B")]["area_mm2"] >= 400 - 1e-6
        assert "A#5" in by[("A", "B")]["faces_a"] or len(by[("A", "B")]["faces_a"]) >= 1
        assert by[("A", "B")]["proposed_type"] == "bonded" and not d["applied"]
        assert ("B", "C") not in by                       # mezera 0,5 mm > tolerance 0,1 mm
        assert (await s.ok("fea_contact_list"))["pairs"] == []
        d = await s.ok("fea_contact_detect", tol_mm=1.0, apply=True, type="frictionless")
        assert len(d["applied"]) == 2
        bc = next(p for p in d["pairs"] if p["b"] == "C")
        assert bc["state"] == "gap" and abs(bc["distance_mm"] - 0.5) < 1e-6
        again = await s.ok("fea_contact_detect", tol_mm=1.0, apply=True)
        assert again["applied"] == [] and all(p["exists"] for p in again["pairs"])
        lst = await s.ok("fea_contact_list")
        assert {p["type"] for p in lst["pairs"]} == {"frictionless"} and lst["nonlinear"] is True
        ab = next(p for p in lst["pairs"] if p["b"] == "B")
        assert ab["K_N_mm3"] == pytest.approx(50 * 210000.0) and ab["lambda_N_mm3"] == pytest.approx(50 * 2100.0)
        # friction: μ given and μ from the library (steel – steel) with the source
        p = await s.ok("fea_contact_set", pair=ab["name"], type="frictional", mu=0.25, stiffness=1e6)
        assert p["pair"]["type"] == "frictional" and p["pair"]["mu"] == 0.25 and p["pair"]["mu_source"] == "user-defined"
        assert p["pair"]["K_N_mm3"] == 1e6 and p["pair"]["lambda_N_mm3"] == pytest.approx(1e4)
        p = await s.ok("fea_contact_set", pair=ab["name"], reset=["mu"], purpose="resistance")
        assert p["pair"]["mu"] > 0 and p["pair"]["mu_source"] != "user-defined"
        # new pair from a selection of faces, errors
        p = await s.ok("fea_contact_set", pair="podpora", type="compression_only", a={"part": "A", "z": 0})
        assert p["created"] and p["pair"]["b"] is None and p["pair"]["a"] == "podpora_a"
        assert "both a and b" in await s.err("fea_contact_set", pair="new", type="frictional", a="A")
        assert "Unknown contact type" in await s.err("fea_contact_set", pair=ab["name"], type="lepidlo")
        assert "positive" in await s.err("fea_contact_set", pair=ab["name"], mu=0.0)
        assert "adjust" in await s.err("fea_contact_set", pair=ab["name"], type="interference", adjust=True)
        # settings and sequence
        st = await s.ok("fea_contact_settings", method="mortar")
        assert st["settings"]["method"] == "mortar"
        assert "method" in await s.err("fea_contact_settings", method="pokus")
        await s.ok("fea_contact_settings", method="surface_to_surface")
        seq = await s.ok("fea_analysis_set_sequence", kind="nonlinear", branches=False,
                         steps=[{"name": "load", "case": "p", "scale": 0.5},
                                {"name": "plné", "case": "neni", "mu_change": {ab["name"]: 0.3}}])
        assert [x["name"] for x in seq["analysis"]["sequence"]] == ["load", "plné"]
        assert any("does not exist" in e for e in seq["errors"])
        assert "unknown keys" in await s.err("fea_analysis_set_sequence", steps=[{"name": "x", "zatez": 1}])
        chk = await s.ok("fea_contact_check")
        assert chk["ok"] is False and "mesh" in chk
        # image of the model with the pairs coloured by type
        r = await s.raw("fea_model_render_image", output_path=str(tmp_path / "model.png"), view="iso")
        assert not r.is_error and png_size(tmp_path / "model.png") == (1280, 720)
        rm = await s.ok("fea_contact_remove", pair=ab["name"])
        assert rm["removed"] == [ab["name"]]
        seq2 = (await s.ok("fea_contact_list"))["analysis"]
        assert seq2["kind"] == "nonlinear"
        assert "does not exist" in await s.err("fea_contact_remove", pair="neni")
        rm = await s.ok("fea_contact_remove", pair="*")
        assert rm["remaining"] == []
        # without results
        assert "results" in (await s.err("fea_contact_results")).lower()
        assert "results" in (await s.err("fea_results_history")).lower()
    run(tmp_path, body)


def test_readiness_contact_rigid_warning(tmp_path, blocks):
    """A free motion of the constraints in a study with a contact is only a warning (the contact may hold the part, but it does not replace the constraint)."""
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(blocks), workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
        await s.ok("fea_constraint_add_displacement", name="dno", selection={"z": 0}, uz=0.0)
        await s.ok("fea_load_add_pressure", case_name="p", name="vrch", selection={"z": 25.5}, pressure=10.0)
        rd = await s.ok("fea_check_readiness")
        assert any("rigid-body motions" in e for e in rd["errors"])
        await s.ok("fea_contact_set", pair="AB", type="frictionless", a="A", b="B")
        rd = await s.ok("fea_check_readiness")
        assert not any("rigid-body motions" in e for e in rd["errors"])
        assert any("rigid-body motions" in w and "contact" in w for w in rd["warnings"])
    run(tmp_path, body)


def test_jobs_without_study(tmp_path):
    async def body(s, c):
        assert "No job" in await s.err("fea_job_status")
        assert "study" in (await s.err("fea_job_start")).lower()
        assert "kind" in await s.err("fea_job_start", kind="pokus")
    run(tmp_path, body)


def test_job_lifecycle_error(tmp_path, blocks):
    """A job over an unprepared study: the start returns the id at once, the state ends with 'error', the result = the error with the text."""
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(blocks), workdir=str(tmp_path / "wd"))
        j = await s.ok("fea_job_start", kind="solve")
        assert j["job_id"].startswith("job-") and j["timeout_s"] == 1800
        for _ in range(100):
            st = await s.ok("fea_job_status", job_id=j["job_id"])
            if st["status"] not in ("starting", "running"):
                break
            await anyio.sleep(0.1)
        assert st["status"] == "error" and "not ready" in st["error"]
        assert "not ready" in await s.err("fea_job_result", job_id=j["job_id"])
        assert "no longer running" in (await s.ok("fea_job_cancel"))["note"]
        await s.ok("fea_study_status")                          # the session lock is released
    run(tmp_path, body)


# ===================================================================== e2e (ccx)

def _poll(s, job_id, limit=900):
    async def wait():
        import anyio
        t0 = time.time()
        while time.time() - t0 < limit:
            st = await s.ok("fea_job_status", job_id=job_id)
            if st["status"] not in ("starting", "running"):
                return st
            await anyio.sleep(1.0)
        raise AssertionError("úloha nedoběhla")
    return wait()


@pytest.mark.slow
def test_e2e_k1_contact_job(tmp_path):
    """K1 through an asynchronous job: p = 10 MPa in the contact, Fn = 4000 N, images of the pressure and of the state, history, protocol."""
    import verification.cases  # noqa: F401 – order of the imports (cyclic import of the verification modules)
    from verification.contact_cases import k1_build
    wd = tmp_path / "k1"
    wd.mkdir()
    st, _ = k1_build(wd)
    sj = wd / "study.json"
    st.save(sj)

    async def body(s, c):
        await s.ok("fea_study_open", path=str(sj), workdir=str(tmp_path / "wd"))
        lst = await s.ok("fea_contact_list")
        assert lst["pairs"][0]["K_N_mm3"] == 1e6 and lst["nonlinear"]
        j = await s.ok("fea_job_start", kind="solve")
        assert j["timeout_s"] == 14400 and j["detail"].startswith("nonlinear")
        if (await s.ok("fea_job_status"))["status"] == "running":
            d = await s.call("fea_contact_list")               # the lock is held by the job (or it has already finished)
            assert "__error__" not in d or "busy" in d["__error__"]
        st_ = await _poll(s, j["job_id"])
        assert st_["status"] == "done" and st_["progress"] == 1.0, st_
        r = await s.ok("fea_job_result", job_id=j["job_id"])
        assert r["ok"] and r["steps"][0]["contacts"]["A-B"]["Fn_N"] == pytest.approx(4000, rel=1e-3)
        assert "nonlinear_warning" in r
        cr = await s.ok("fea_contact_results")
        c0 = cr["steps"][0]["contacts"]["A-B"]
        assert c0["p_max_MPa"] == pytest.approx(10, rel=5e-3) and c0["nodes_stick"] + c0["nodes_slip"] > 0
        for f in ("tlak", "stav", "mezera"):
            rr = await s.raw("fea_contact_render", field=f, output_path=str(tmp_path / f"{f}.png"))
            assert not rr.is_error, rr.content[0].text
            assert png_size(tmp_path / f"{f}.png") == (1280, 720)
        meta = json.loads(rr.content[0].text)
        assert meta["parts"] in (["A"], ["B"])
        h = await s.ok("fea_results_history")
        assert h["steps"][0]["contacts"]["A-B"]["Fn_N"] == pytest.approx(4000, rel=1e-3)
        out = tmp_path / "k1.docx"
        await s.ok("fea_report_word", output_path=str(out), images=True, language="en")
        from tests.test_report import doc_text
        text, heads, _, nimg = doc_text(out)
        assert any("Contacts" in x for x in heads) and "WARNING – NONLINEAR ANALYSIS" in text
        assert "Frictionless contact" in text and nimg >= 3
    run(tmp_path, body)
