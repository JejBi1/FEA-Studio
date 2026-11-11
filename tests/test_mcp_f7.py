"""MCP tools F7: natural frequencies and buckling, elastic support, orthotropic material, *TIE, sphere refinement, results
along a line and linearisation, mesh convergence, tolerance and refinement of the 3D printing, members (sections, analytics, EN 1993-1-1).

Fast tests through the MCP protocol without a calculation; e2e (Gmsh + ccx) marked slow.
"""
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from tests.test_mcp import Session, png_size, run  # noqa: E402,F401

F7_TOOLS = {
    "fea_modal_set", "fea_buckling_set", "fea_results_modes", "fea_results_render_mode",
    "fea_constraint_add_elastic_support", "fea_material_set_orthotropic",
    "fea_mesh_info", "fea_mesh_separate_parts", "fea_mesh_refine_point", "fea_mesh_convergence",
    "fea_results_line", "fea_results_linearize", "fea_print_refine",
    "fea_section_get", "fea_section_search", "fea_beam_analytic", "fea_beam_buckling", "fea_curved_beam",
    "fea_ec3_member_check",
}
E = 210000.0


@pytest.fixture(scope="module")
def geo(tmp_path_factory):
    """STEP: cantilever V1 200×20×10, cube 10³, a split cantilever (A 0–100, B 100–200), a printed bar 40×10×10."""
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    d = tmp_path_factory.mktemp("f7_geo")
    out = {}
    for name, parts in (("beam", [Part("Konzola", box(0, 0, 0, 200, 20, 10))]),
                        ("cube", [Part("Kostka", box(0, 0, 0, 10, 10, 10))]),
                        ("split", [Part("A", box(0, 0, 0, 100, 20, 10)), Part("B", box(100, 0, 0, 100, 20, 10))]),
                        ("bar", [Part("Tyc", box(0, 0, 0, 40, 10, 10))])):
        out[name] = d / f"{name}.step"
        write_step(parts, out[name])
    return out


async def _beam(s, geo, tmp_path, load=None):
    await s.ok("fea_study_new", step_path=str(geo["beam"]), workdir=str(tmp_path / "wd"))
    await s.ok("fea_materials_assign", part="*", material_name="Ocel S235", custom_props={"E": E, "nu": 0.3})
    await s.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
    if load is not None:
        await s.ok("fea_load_add_force", case_name=load[0], name="konec", selection={"x": 200}, force=load[1])


# ===================================================================== fast

def test_f7_tools_registered(tmp_path):
    async def body(s, c):
        names = {t.name for t in (await c.list_tools()).tools}
        assert F7_TOOLS <= names, F7_TOOLS - names
        p = await c.get_prompt("fea_mesh_convergence_check", {"quantity": "σ v rohu"})
        txt = p.messages[0].content.text
        assert "fea_mesh_convergence" in txt and "Richardson" in txt and "GCI" in txt
    run(tmp_path, body)


def test_sections_and_beams(tmp_path):
    async def body(s, c):
        ipe = await s.ok("fea_section_get", section="IPE 200")
        assert ipe["name"] == "IPE 200" and abs(ipe["A_mm2"] - 2848) / 2848 < 0.01
        assert abs(ipe["Iy_mm4"] - 19.43e6) / 19.43e6 < 0.01 and abs(ipe["mass_kg_m"] - 22.4) < 0.3
        rect = await s.ok("fea_section_get", section={"shape": "RECT", "b": 20, "h": 40})
        assert rect["A_mm2"] == 800 and abs(rect["Iy_mm4"] / (20 * 40 ** 3 / 12) - 1) < 1e-4
        assert "dimension" in (await s.err("fea_section_get", section={"shape": "RECT", "b": 20}))
        assert "IPE 200" in (await s.ok("fea_section_search", text="IPE 2"))["sections"]
        fam = await s.ok("fea_section_search")
        assert fam["families"] and all(v["count"] > 0 for v in fam["families"].values())
        # simply supported beam, uniform q: R = qL/2, M = qL²/8, w = 5qL⁴/(384EI)
        L, q, I = 2000.0, 2.0, 2e6
        b = await s.ok("fea_beam_analytic", length_mm=L, loads=[{"type": "distributed", "q_N_mm": q}], I_mm4=I)
        assert abs(b["reactions"]["R_A"] - q * L / 2) < 1e-3 and abs(b["reactions"]["R_B"] - q * L / 2) < 1e-3
        assert abs(b["extremes"]["M_Nmm"]["max"]["value"] - q * L ** 2 / 8) / (q * L ** 2 / 8) < 1e-4
        w = 5 * q * L ** 4 / (384 * E * I)
        assert abs(b["extremes"]["w_mm"]["max"]["value"] - w) / w < 1e-4
        assert abs(b["extremes"]["w_mm"]["max"]["x_mm"] - L / 2) < 1e-3 and len(b["table"]) == 11
        # cantilever with a force at the end, section from the library: w = PL³/(3EI), σ = PL/Wel
        c = await s.ok("fea_beam_analytic", length_mm=1000, supports="cantilever", section="IPE 200",
                       loads=[{"type": "force", "P_N": 1000, "x_mm": 1000}])
        assert abs(c["extremes"]["w_mm"]["abs"]["value"] - 1000 * 1e9 / (3 * E * ipe["Iy_mm4"])) < 1e-3
        assert abs(c["max_bending_stress_MPa"] - 1e6 / ipe["Wel_y_mm3"]) < 1e-3
        # continuous beam of 2 spans, q: internal support 1.25·qL
        cb = await s.ok("fea_beam_analytic", spans_mm=[1000, 1000], supports=["pinned", "pinned"], I_mm4=I,
                        loads=[{"type": "distributed", "q_N_mm": 1.0}])
        assert abs(cb["reactions"]["R_2"] - 1250) < 1e-3 and "continuous" in cb["kind"]
        assert "length_mm" in await s.err("fea_beam_analytic", loads=[], I_mm4=I)
        assert "Load type" in await s.err("fea_beam_analytic", length_mm=1, loads=[{"type": "x"}], I_mm4=I)
        e = await s.ok("fea_beam_buckling", length_mm=2000, I_mm4=I, case="pinned-pinned")
        assert abs(e["F_cr_N"] / (math.pi ** 2 * E * I / 2000 ** 2) - 1) < 1e-4 and "imperfections" in e["warning"]
        # curved bar: the inner fibre more than a straight bar
        cv = await s.ok("fea_curved_beam", profile={"shape": "rect", "b": 10, "h": 20}, M_Nmm=1e5, r_inner_mm=20)
        assert cv["K_inner"] > 1 and abs(cv["sigma_inner_MPa"]) > abs(cv["sigma_inner_straight_MPa"])
        rp = await s.ok("fea_curved_beam", kind="ring_pinch", F_N=100, R_mm=50, I_mm4=1000)
        ref = -(math.pi / 4 - 2 / math.pi) * 100 * 50 ** 3 / (E * 1000)
        assert abs(rp["dD_load_mm"] - ref) / abs(ref) < 1e-4
        ac = await s.ok("fea_curved_beam", kind="arc_cantilever", R_mm=100, theta_deg=90, Ft_N=10, I_mm4=1000)
        assert ac["M_fixed_Nmm"] != 0 and "kind" in ac
        assert "winkler" in await s.err("fea_curved_beam", kind="xx", R_mm=1, I_mm4=1)
        # EN 1993-1-1: IPE 200 S235, constant My, restrained against lateral-torsional buckling → M_pl = Wpl·fy
        ec = await s.ok("fea_ec3_member_check", section="IPE 200", length_mm=3000, grade="S235",
                        forces={"My_Nmm": 30e6}, member={"lt_restrained": True})
        u = 30e6 / (ipe["Wpl_y_mm3"] * 235)
        assert ec["section_class"] == 1 and abs(ec["utilization"] - u) / u < 0.01 and ec["status"] == "PASS"
        assert any("6.2" in r["clause"] for r in ec["checks"])
        assert "Unknown member parameters" in await s.err("fea_ec3_member_check", section="IPE 200", length_mm=1,
                                                         forces={"N_N": 1}, member={"xx": 1})
        assert "Unknown forces keys" in await s.err("fea_ec3_member_check", section="IPE 200", length_mm=1,
                                                     forces={"F": 1})
    run(tmp_path, body)


def test_modal_buckling_settings(tmp_path, geo):
    async def body(s, c):
        await _beam(s, geo, tmp_path)
        m = await s.ok("fea_modal_set", modes=5, f_max_Hz=5000)
        assert m["modal"]["enabled"] and m["modal"]["modes"] == 5 and m["modal"]["f_max"] == 5000
        rd = await s.ok("fea_check_readiness")
        assert rd["ready"] and "natural frequencies" in rd["note"] and rd["eigen"]["modal"]["modes"] == 5
        assert "load case" in await s.err("fea_buckling_set")
        assert "f_max_Hz" in await s.err("fea_modal_set", f_min_Hz=100, f_max_Hz=50)
        assert "modes" in await s.err("fea_modal_set", modes=0)
        await s.ok("fea_load_add_force", case_name="Tlak", name="konec", selection={"x": 200}, force=[-1000, 0, 0])
        assert "does not exist" in await s.err("fea_buckling_set", case_name="Nic")
        b = await s.ok("fea_buckling_set", case_name="Tlak", modes=3)
        assert b["buckling"]["enabled"] and b["case_used"] == "Tlak" and "Linear buckling" in b["warning"]
        st = await s.ok("fea_study_status")
        assert st["modal"]["enabled"] and st["buckling"]["case"] == "Tlak"
        cur = json.loads((await c.read_resource("fea://study/current")).contents[0].text)
        assert cur["modal"]["modes"] == 5 and cur["buckling"]["modes"] == 3
        assert "results" in (await s.err("fea_results_modes")).lower()
        assert "results" in (await s.err("fea_results_line", p0=[0, 0, 0], p1=[1, 0, 0])).lower()
        assert "results" in (await s.err("fea_results_linearize", p0=[0, 0, 0], p1=[1, 0, 0])).lower()
    run(tmp_path, body)


def test_elastic_support_rigid_check(tmp_path, geo):
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(geo["cube"]), workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
        await s.ok("fea_load_add_pressure", case_name="Tlak", name="vrch", selection={"z": 10}, pressure=1.0)
        assert "positive" in await s.err("fea_constraint_add_elastic_support", name="dno", selection={"z": 0},
                                       stiffness=-1)
        assert "direction" in await s.err("fea_constraint_add_elastic_support", name="dno", selection={"z": 0},
                                          stiffness=1, direction="xy")
        r = await s.ok("fea_constraint_add_elastic_support", name="dno", selection={"z": 0}, stiffness=100)
        assert r["constraint"]["kind"] == "elastic" and "elastic support k = 100 N/mm³" in r["constraint"]["description"]
        rd = await s.ok("fea_check_readiness")                 # only perpendicular to the base: displacements x, y and the rotation about z free
        assert not rd["ready"] and "rigid-body motions" in " ".join(rd["errors"])
        assert len(rd["rigid_body"]["free_rigid_modes"]) == 3 and "note" not in rd["rigid_body"]
        await s.ok("fea_constraint_add_elastic_support", name="dno", selection={"z": 0}, stiffness=1e4,
                   direction="all", per_area=False)
        rd = await s.ok("fea_check_readiness")
        assert rd["ready"], rd["errors"]
        lst = await s.ok("fea_constraint_list")
        assert "total" in lst["constraints"][0]["description"]
        r = await s.raw("fea_model_render_image", output_path=str(tmp_path / "model.png"), view="iso",
                        return_image=False)
        assert not r.is_error and png_size(tmp_path / "model.png") == (1280, 720)
    run(tmp_path, body)


def test_orthotropic_material(tmp_path, geo):
    ok_c = {"E1": 140000, "E2": 10000, "E3": 10000, "nu12": 0.3, "nu13": 0.3, "nu23": 0.4, "G12": 5000,
            "G13": 5000, "G23": 3500, "alpha1": -0.5e-6, "alpha2": 30e-6, "alpha3": 30e-6}

    async def body(s, c):
        await _beam(s, geo, tmp_path)
        await s.ok("fea_csys_create", name="mat", kind="cartesian", origin=[0, 0, 0], z_axis=[0, 0, 1],
                   x_axis=[1, 1, 0])
        r = await s.ok("fea_material_set_orthotropic", name="CFRP UD", constants=ok_c, csys="mat", part="*",
                       rho=1.6e-9)
        mrow = r["material"]
        assert mrow["E_MPa"] == 140000 and mrow["orthotropic"]["G23"] == 3500 and mrow["ortho_csys"] == "mat"
        assert "Re" in r["warning"] and "axes" in r
        st = await s.ok("fea_study_status")
        assert st["materials"]["CFRP UD"]["orthotropic"]["E2"] == 10000 and st["parts"][0]["material"] == "CFRP UD"
        assert "missing G23" in await s.err("fea_material_set_orthotropic", name="X",
                                         constants={k: v for k, v in ok_c.items() if k != "G23"})
        assert "positive definite" in await s.err("fea_material_set_orthotropic", name="X",
                                                    constants={**ok_c, "nu23": 1.2})
        assert "does not exist" in await s.err("fea_material_set_orthotropic", name="X", constants=ok_c, csys="nic")
        assert "1/K" in await s.err("fea_material_set_orthotropic", name="X", constants={**ok_c, "alpha1": 12})
        assert "Unknown orthotropic material constants" in await s.err("fea_materials_assign", part="*", material_name="Y",
                                                  custom_props={"E": 1, "nu": 0.3, "ortho": {"E9": 1}})
        assert "material 'CFRP UD'" in await s.err("fea_csys_delete", name="mat")
        rd = await s.ok("fea_check_readiness")
        assert any("Re" in w for w in rd["warnings"])
    run(tmp_path, body)


def test_mesh_settings_tools(tmp_path, geo):
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(geo["split"]), workdir=str(tmp_path / "wd"))
        assert "does not exist" in await s.err("fea_mesh_separate_parts", parts=["C"])
        r = await s.ok("fea_mesh_separate_parts", parts=["B"], tie_tolerance_mm=0.2)
        assert r["separate_parts"] == ["B"] and r["tie_tolerance_mm"] == 0.2
        assert (await s.ok("fea_mesh_separate_parts", parts=["B"], separate=False))["separate_parts"] == []
        assert "No mesh" in await s.err("fea_mesh_info")
        assert "positive" in await s.err("fea_mesh_refine_point", center=[50, 10, 5], radius_mm=0, size_mm=1)
        assert "smaller" in await s.err("fea_mesh_refine_point", center=[50, 10, 5], radius_mm=5, size_mm=50)
        r = await s.ok("fea_mesh_refine_point", center=[50, 10, 5], radius_mm=5, size_mm=0.8)
        assert r["added"] == {"center": [50, 10, 5], "radius": 5, "size": 0.8} and r["dof_estimate"] > 0
        st = await s.ok("fea_study_status")
        assert st["mesh_settings"]["local"][0]["radius"] == 5
        assert (await s.ok("fea_mesh_refine_point", remove_all=True))["removed"] == 1
        assert "quantity" in await s.err("fea_mesh_convergence", quantity="xx")
        assert "point" in await s.err("fea_mesh_convergence", quantity="probe")
    run(tmp_path, body)


def test_print_tolerance_setting(tmp_path, geo):
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(geo["bar"]), workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Tisk PLA Bambu Basic")
        r = await s.ok("fea_print_set", part="Tyc", settings={"tolerance": 0.1})
        assert r["print"]["tolerance"] == 0.1
        assert "tolerance" in await s.err("fea_print_set", part="Tyc", settings={"tolerance": 3})
        assert (await s.ok("fea_print_get", part="Tyc"))
        assert "not ready" in await s.err("fea_print_refine")          # without constraints and loads
        assert "n_spots" in await s.err("fea_print_refine", n_spots=0)
    run(tmp_path, body)


# ===================================================================== slow (Gmsh + ccx)

@pytest.mark.slow
def test_e2e_modal_buckling_line_linearize_convergence(tmp_path, geo):
    """Cantilever V1: f₁ (E–B 208.9 Hz), buckling by a compression at the end (Euler β = 2: 21.59 kN), images of the shapes, course of σx over
    the height (±30 MPa), linearisation (bending 30 MPa, membrane ≈ 0), convergence in the sphere without loss of the results.
    """
    I = 20 * 10 ** 3 / 12

    async def body(s, c):
        await _beam(s, geo, tmp_path, ("Tlak", [-1000, 0, 0]))
        await s.ok("fea_load_add_force", case_name="Ohyb", name="konec_ohyb", selection={"x": 200}, force=[0, 0, -100])
        await s.ok("fea_modal_set", modes=4)
        await s.ok("fea_buckling_set", case_name="Tlak", modes=2)
        await s.ok("fea_mesh_generate", global_size=2.5)
        r = await s.ok("fea_solve_run", timeout_s=900, backend="ccx")
        assert {m["kind"] for m in r["modes"]} == {"modal", "buckling"} and "buckling_warning" in r
        md = (await s.ok("fea_results_modes", kind="modal"))["mode_sets"][0]
        f1 = md["frequencies_Hz"][0]
        f_eb = 1.875104 ** 2 / (2 * math.pi * 200 ** 2) * math.sqrt(E * I / (7.85e-9 * 200))
        assert abs(f1 - f_eb) / f_eb < 0.01, f1
        m1 = md["modes"][0]
        assert abs(m1["eff_mass_fraction_xyz"][2] - 0.6131) < 0.03 and m1["cumulative_xyz"][2] <= 1.0001
        bk = (await s.ok("fea_results_modes", kind="buckling"))["mode_sets"][0]
        pcr = math.pi ** 2 * E * I / (4 * 200 ** 2)
        assert abs(bk["critical_load_N"][0] - pcr) / pcr < 0.02 and "LINEAR BUCKLING" in " ".join(bk["notes"])
        summ = await s.ok("fea_results_summary", step=None)
        assert len(summ["modes"]) == 2 and len(summ["steps"]) == 2
        img = tmp_path / "mode1.png"
        rr = await s.raw("fea_results_render_mode", mode=1, output_path=str(img), width=800, height=500)
        meta = json.loads(rr.content[0].text)
        assert not rr.is_error and abs(meta["frequency_Hz"] - f1) < 1e-6 and png_size(img) == (800, 500)
        rr = await s.raw("fea_results_render_mode", kind="buckling", mode=1, field="uz", view="front",
                         output_path=str(tmp_path / "buck.png"), return_image=False)
        assert not rr.is_error and "warning" in json.loads(rr.content[0].text)
        assert "mode must be" in await s.err("fea_results_render_mode", mode=9)
        # course of σx over the height at x = 100 (M = 100·100 N·mm, W = 333.3 mm³ → ±30 MPa)
        ln = await s.ok("fea_results_line", p0=[100, 10, 0], p1=[100, 10, 10], field="sxx", step="Ohyb", n=11)
        assert abs(ln["values"][0] + 30) < 1.5 and abs(ln["values"][-1] - 30) < 1.5 and ln["points_outside"] == 0
        rr = await s.raw("fea_results_line", p0=[100, 10, 5], p1=[300, 10, 5], field="uz", step="Ohyb", plot=True,
                         output_path=str(tmp_path / "line.png"))
        d = json.loads(rr.content[0].text)
        assert not rr.is_error and d["points_outside"] > 0 and png_size(tmp_path / "line.png") == (960, 540)
        lin = await s.ok("fea_results_linearize", p0=[100, 10, 0], p1=[100, 10, 10], step="Ohyb", hint=[1, 0, 0],
                         allowable_S_MPa=100)
        assert abs(abs(lin["Pb_MPa"]["local"][1]) - 30) < 1.5 and lin["Pm_MPa"]["mises"] < 1.0
        assert abs(lin["PmPb_mises_max_MPa"] - 30) < 1.5 and lin["check_informative"]["ok"]
        assert "outside the body" in await s.err("fea_results_linearize", p0=[100, 10, 0], p1=[100, 10, 30], step="Ohyb")
        hs = await s.ok("fea_results_hotspots", step="Ohyb", top_n=2)
        assert hs["hotspots"] and "stress_error_pct" in hs["hotspots"][0]           # odhad chyby ccx (STR %)
        so = (await s.ok("fea_results_summary", step="Ohyb"))["steps"][0]
        assert so["max_stress_error_hotspots_pct"] >= 0 and "stress_error_note" in so
        rr = await s.raw("fea_results_render_image", field="stress_error", step="Ohyb", return_image=False,
                         output_path=str(tmp_path / "err.png"))
        assert not rr.is_error and json.loads(rr.content[0].text)["unit"] == "%"
        # convergence: max. von Mises in the sphere around x = 100 at the top (a smooth place) – the results of the main study stay
        cv = await s.ok("fea_mesh_convergence", quantity="probe", field="mises", point=[100, 10, 10], radius_mm=6,
                        step="Ohyb", levels=3, ratio=1.5, tol=0.05, timeout_s=900)
        assert len(cv["levels"]) == 3 and cv["levels"][2]["size_mm"] < cv["levels"][0]["size_mm"]
        assert all(abs(lv["value"] - 30) < 2 for lv in cv["levels"]) and cv["converged"], cv
        assert (await s.ok("fea_results_summary", step="Ohyb"))["steps"][0]["step"] == "Ohyb"
    run(tmp_path, body)


@pytest.mark.slow
def test_e2e_elastic_support(tmp_path, geo):
    """Cube 10³ on a foundation k = 100 N/mm³ normal, pressure 1 MPa: u_z of the bottom = −p/k, reaction of the foundation 100 N."""
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(geo["cube"]), workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Ocel S235")
        await s.ok("fea_constraint_add_elastic_support", name="dno", selection={"z": 0}, stiffness=100)
        await s.ok("fea_constraint_add_displacement", name="x0", selection={"x": 0}, ux=0)
        await s.ok("fea_constraint_add_displacement", name="y0", selection={"y": 0}, uy=0)
        await s.ok("fea_load_add_pressure", case_name="Tlak", name="vrch", selection={"z": 10}, pressure=1.0)
        assert (await s.ok("fea_check_readiness"))["ready"]
        await s.ok("fea_mesh_generate", global_size=2.5)
        r = await s.ok("fea_solve_run", timeout_s=600)
        st = r["steps"][0]
        assert abs(st["reactions"]["dno"]["force_N"][2] - 100) < 0.01 and st["equilibrium"]["ok"]
        p = await s.ok("fea_results_probe", point=[5, 5, 0])
        assert abs(p["probes"][0]["u_mm"][2] + 0.01) < 2e-4
        info = await s.ok("fea_mesh_info")
        assert info["nodes"] > 100 and info["source"] == "valid study mesh"
    run(tmp_path, body)


@pytest.mark.slow
def test_e2e_tie_separate_part(tmp_path, geo):
    """Split cantilever: part B meshed separately, *TIE 20×10 = 200 mm², deflection as V1 (0.7634 mm)."""
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(geo["split"]), workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Ocel S235", custom_props={"E": E, "nu": 0.3})
        await s.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
        await s.ok("fea_load_add_force", case_name="Sila", name="konec", selection={"x": 200}, force=[0, 0, -100])
        await s.ok("fea_mesh_separate_parts", parts=["B"])
        m = await s.ok("fea_mesh_generate", global_size=2.5)
        assert m["ties"] and abs(m["ties"][0]["area_mm2"] - 200) < 0.5 and m["ties"][0]["part"] == "B"
        info = await s.ok("fea_mesh_info")
        assert info["separate_parts"] == ["B"] and len(info["ties"]) == 1
        await s.ok("fea_solve_run", timeout_s=600, backend="auto")
        p = await s.ok("fea_results_probe", point=[200, 10, 5])
        assert abs(-p["probes"][0]["u_mm"][2] - 0.7634) / 0.7634 < 0.03
    run(tmp_path, body)


@pytest.mark.slow
def test_e2e_print_refine(tmp_path, geo):
    """A printed cantilever 40×10×10 (PLA): two-step refinement in the places of the minimum safety, taking over into the session."""
    async def body(s, c):
        await s.ok("fea_study_new", step_path=str(geo["bar"]), workdir=str(tmp_path / "wd"))
        await s.ok("fea_materials_assign", part="*", material_name="Tisk PLA Bambu Basic")
        await s.ok("fea_print_set", part="Tyc", settings={"build_dir": [0, 0, 1]})
        await s.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
        await s.ok("fea_load_add_force", case_name="Sila", name="konec", selection={"x": 40}, force=[0, 0, -20])
        await s.ok("fea_mesh_generate", global_size=2.5)
        await s.ok("fea_solve_run", timeout_s=600)
        r = await s.ok("fea_print_refine", n_spots=2, timeout_s=1200)
        assert r["min_safety_before"] > 0 and r["min_safety_after"] > 0 and r["nodes_after"] > r["nodes_before"]
        assert 1 <= len(r["balls"]) <= 2 and "adopted" in r
        st = await s.ok("fea_study_status")
        assert any("center" in loc for loc in st["mesh_settings"]["local"])
        assert st["mesh"] != "none" and st["results"]["nodes"] == r["nodes_after"]
        pr = await s.ok("fea_print_results")
        assert pr["printing"]
    run(tmp_path, body)
