"""MCP tools of F8 – structural dynamics (tools_dynamics): settings, damping, excitation, profiles, probes, added mass.

Fast tests through the MCP protocol in memory without a calculation; e2e 1-DOF oscillator (Y1, ccx + modal superposition).
"""
import json
import math
import time

import anyio
import pytest

pytest.importorskip("mcp")

from tests.test_mcp import Session, png_size, run  # noqa: E402,F401

DYN_TOOLS = {
    "fea_dyn_settings", "fea_dyn_damping_set", "fea_dyn_harmonic_set", "fea_dyn_transient_set", "fea_dyn_random_set",
    "fea_dyn_shock_set", "fea_dyn_prestress_set", "fea_dyn_probe_add", "fea_dyn_probe_list", "fea_dyn_probe_remove",
    "fea_dyn_mass_add", "fea_dyn_mass_list", "fea_dyn_mass_remove", "fea_dyn_frf_get", "fea_dyn_results",
    "fea_dyn_render", "fea_dyn_plot", "fea_dyn_profile_presets",
}


@pytest.fixture(scope="module")
def beam(tmp_path_factory):
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    p = tmp_path_factory.mktemp("dyn_geo") / "beam.step"
    write_step([Part("Konzola", box(0, 0, 0, 100, 10, 10))], p)
    return p


async def _study(s, beam, tmp_path):
    await s.ok("fea_study_new", step_path=str(beam), workdir=str(tmp_path / "wd"))
    await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
    await s.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
    await s.ok("fea_load_add_force", case_name="F", name="konec", selection={"x": 100}, force=[0, 0, 100])


def test_dynamics_tools_registered(tmp_path):
    async def body(s, c):
        names = {t.name for t in (await c.list_tools()).tools}
        assert DYN_TOOLS <= names, DYN_TOOLS - names
        tools = {t.name: t for t in (await c.list_tools()).tools}
        props = tools["fea_dyn_mass_add"].input_schema["properties"]
        assert "mass_kg" in props and "mass_per_area_kg_m2" in props
        assert "f_min_Hz" in tools["fea_dyn_harmonic_set"].input_schema["properties"]
        txt = (await c.get_prompt("fea_setup_vibration_test", {})).messages[0].content.text
        assert "fea_dyn_profile_presets" in txt and "do not invent" in txt
    run(tmp_path, body)


def test_dynamics_settings_and_errors(tmp_path, beam):
    async def body(s, c):
        await _study(s, beam, tmp_path)
        d = await s.ok("fea_dyn_settings", enabled=True, modes=12, f_max_Hz=5000)
        assert d["dynamics"]["modes"] == 12 and "±50 %" in d["damping_warning"]
        assert any("response" in w for w in d["warnings"])           # only the basis
        dm = await s.ok("fea_dyn_damping_set", kind="rayleigh", zeta=0.03, f1_Hz=50, f2_Hz=500)
        assert dm["dynamics"]["damping"]["kind"] == "rayleigh" and dm["typical_zeta"]
        assert "zeta" in await s.err("fea_dyn_damping_set", zeta=1.5)
        # sine profile: f_c = √(a/x̂)/2π – 0.075 mm / 1 g → 57.6 Hz
        h = await s.ok("fea_dyn_harmonic_set", f_min_Hz=10, f_max_Hz=500, sine_profile={"displacement_mm": 0.075,
                                                                                         "acceleration_g": 1.0}, dof="z")
        assert h["sine_profile"]["crossover_Hz"] == pytest.approx(math.sqrt(9810 / 0.075) / (2 * math.pi), rel=1e-4)
        assert h["dynamics"]["harmonic"]["excitation"][0].startswith("base acceleration")
        h = await s.ok("fea_dyn_harmonic_set", excitation=[{"kind": "force", "case": "F", "table": [[10, 1], [500, 2]]}],
                       sweep_rate_oct_min=1.0)
        assert h["dynamics"]["harmonic"]["excitation"][0].startswith("force")
        assert "case" in await s.err("fea_dyn_harmonic_set", excitation=[{"kind": "force", "case": "neni"}])
        assert "Unknown excitation keys" in await s.err("fea_dyn_harmonic_set", excitation=[{"kind": "base_acc", "amp": 1}])
        assert "f_min" in await s.err("fea_dyn_harmonic_set", f_min_Hz=600)
        t = await s.ok("fea_dyn_transient_set", t_end_s=0.05, dt_s=1e-5, base_acc_g=10, dof=3,
                       pulse={"shape": "halfsine", "tau_s": 0.005}, settle_tol_mm=0.01)
        assert t["dynamics"]["transient"]["settle_tol_mm"] == 0.01
        assert "dt_s" in await s.err("fea_dyn_transient_set", dt_s=1.0)
        r = await s.ok("fea_dyn_random_set", points=[[20, None], [80, 0.04], [350, 0.04], [2000, None]],
                       slopes_db_oct=[3, 0, -6], dof="z", duration_h=0.5)
        assert r["grms_g"] > 0 and r["dynamics"]["random"]["asd_g2_Hz"][0][1] < 0.04
        r2 = await s.ok("fea_dyn_random_set", asd_g2_Hz=[[20, 0.01], [2000, 0.01]], grms_g=5.0)
        assert r2["grms_g"] == pytest.approx(5.0, rel=1e-9)
        sh = await s.ok("fea_dyn_shock_set", pulse="halfsine", A_g=50, tau_s=0.011, dof=3, combine="srss")
        assert sh["dynamics"]["shock"]["combine"] == "srss"
        assert "pulse" in await s.err("fea_dyn_shock_set", pulse="ctverec")
        pr = await s.ok("fea_dyn_prestress_set", enabled=True, case="F")
        assert pr["dynamics"]["prestress"]["case"] == "F"
        assert "does not exist" in await s.err("fea_dyn_prestress_set", case="neni")
        await s.ok("fea_dyn_prestress_set", enabled=False)
        # sondy a hmoty
        p = await s.ok("fea_dyn_probe_add", name="konec", point=[100, 5, 5])
        assert p["probe"]["point"] == [100.0, 5.0, 5.0]
        p = await s.ok("fea_dyn_probe_add", name="čelo", selection={"x": 100}, quantity="A")
        assert p["probe"]["selection"] == "sonda čelo"
        assert len((await s.ok("fea_dyn_probe_list"))["probes"]) == 2
        await s.ok("fea_dyn_probe_remove", name="čelo")
        assert "does not exist" in await s.err("fea_dyn_probe_remove", name="čelo")
        m = await s.ok("fea_dyn_mass_add", name="motor", selection={"x": 100}, mass_kg=1.5, inertia_kg_mm2=[10, 20, 30])
        assert m["mass"]["mass_kg"] == pytest.approx(1.5) and m["mass"]["inertia_kg_mm2"] == pytest.approx([10, 20, 30])
        st = json.loads((await c.read_resource("fea://study/current")).contents[0].text)
        assert st["masses"][0]["mass"] == pytest.approx(1.5e-3)          # t v modelu
        m = await s.ok("fea_dyn_mass_add", name="DPS", selection={"z": 10}, mass_per_area_kg_m2=3.0)
        assert m["mass"]["kind"] == "surface_mass"
        assert "exactly one" in await s.err("fea_dyn_mass_add", name="x", selection={"z": 10})
        assert len((await s.ok("fea_dyn_mass_list"))["masses"]) == 2
        await s.ok("fea_dyn_mass_remove", name="DPS")
        r = await s.raw("fea_model_render_image", output_path=str(tmp_path / "m.png"), view="iso", return_image=False)
        assert not r.is_error and "probe konec" in r.content[0].text
        # without results
        assert "no dynamics" in (await s.err("fea_dyn_results")) or "results" in (await s.err("fea_dyn_results")).lower()
        assert "results" in (await s.err("fea_dyn_frf_get")).lower()
        pres = await s.ok("fea_dyn_profile_presets")
        assert set(pres["presets"]) >= {"IEC 60068-2-6", "IEC 60068-2-64", "IEC 60068-2-27", "MIL-STD-810H"}
        assert "from the standard" in pres["note"]
        one = await s.ok("fea_dyn_profile_presets", name="iec 60068-2-64")
        assert one["kind"] == "random" and "fea_dyn_random_set" in one["tool"]
        rd = await s.ok("fea_check_readiness")
        assert rd["ready"], rd["errors"]
    run(tmp_path, body)


# ===================================================================== e2e

@pytest.mark.slow
def test_e2e_oscillator_y1(tmp_path):
    """Y1 through an asynchronous job: f = √(k/m)/2π, |u|/u_st at resonance = 1/(2ζ), FRF plot, image, protocol."""
    import verification.cases  # noqa: F401 – order of the imports (cyclic import of the verification modules)
    from verification.dynamics_cases import y1_build, F_OSC, K_SPR
    wd = tmp_path / "y1"
    wd.mkdir()
    st, _ = y1_build(wd)
    sj = wd / "study.json"
    st.save(sj)

    async def body(s, c):
        await s.ok("fea_study_open", path=str(sj), workdir=str(tmp_path / "wd"))
        j = await s.ok("fea_job_start")
        assert j["timeout_s"] == 7200
        t0 = time.time()
        while time.time() - t0 < 900:
            js = await s.ok("fea_job_status", job_id=j["job_id"])
            if js["status"] not in ("starting", "running"):
                break
            await anyio.sleep(1.0)
        assert js["status"] == "done", js
        r = await s.ok("fea_job_result")
        assert r["dynamics"]["modes"] >= 1 and "damping_warning" in r
        d = await s.ok("fea_dyn_results")
        fz = [m for m in d["modes_table"] if m["eff_mass_fraction_xyz"][2] > 0.5]
        assert fz and fz[0]["f_Hz"] == pytest.approx(F_OSC, rel=2e-3)
        frf = await s.ok("fea_dyn_frf_get", probe="hmota", kind="harmonic", max_points=1000, reaction=True)
        zeta = fz[0]["zeta"]
        peak = frf["max"]["value"] / (100.0 / K_SPR)
        assert peak == pytest.approx(1 / (2 * zeta), rel=2e-2) and frf["max"]["at"] == pytest.approx(F_OSC, rel=1e-2)
        assert "reaction_max_N" in frf
        rr = await s.raw("fea_dyn_plot", kind="frf", probe="hmota", component="z", output_path=str(tmp_path / "frf.png"))
        assert not rr.is_error and png_size(tmp_path / "frf.png")[0] > 500
        rr = await s.raw("fea_dyn_render", field="harmonic_at", frequency_Hz=F_OSC, quantity="displacement",
                         output_path=str(tmp_path / "h.png"))
        assert not rr.is_error, rr.content[0].text
        rr = await s.raw("fea_dyn_render", field="mode", mode=1, output_path=str(tmp_path / "m.png"))
        assert not rr.is_error, rr.content[0].text
        out = tmp_path / "y1.docx"
        await s.ok("fea_report_word", output_path=str(out), images=True, language="en")
        from tests.test_report import doc_text
        text, heads, _, nimg = doc_text(out)
        assert any("Dynamics" in x for x in heads) and "WARNING – DAMPING" in text and nimg >= 2
    run(tmp_path, body)
