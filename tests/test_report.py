"""Word protocol (F6, fea_core.report): number format, a protocol without results (MCP), e2e console with fatigue,
bolts (S2) and 3D printing with creep (P4). The calculations are marked slow.
"""
import json
from pathlib import Path

import anyio
import pytest

docx = pytest.importorskip("docx")

from fea_core.report import ReportOptions, build_report, session_from  # noqa: E402
from fea_core.report.word import num  # noqa: E402


def doc_text(path: Path) -> tuple[str, list[str], int, int]:
    """(the whole text incl. the tables, headings, number of tables, number of images)."""
    d = docx.Document(str(path))
    heads = [p.text for p in d.paragraphs if p.style.name.startswith(("Heading", "Title"))]
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text for c in row.cells))
    return "\n".join(parts), heads, len(d.tables), len(d.inline_shapes)


@pytest.fixture(autouse=True)
def _czech_report():
    """The tests check the Czech report (the default language of the core is English)."""
    from fea_core import i18n
    i18n.set_language("cs")
    yield
    i18n.set_language("en")


def test_num_format():
    assert num(1234.5) == "1235" or num(1234.5) == "1234"
    assert num(0.7585) == "0,7585"
    assert num(-0.5) == "−0,5"
    assert num(210000.0) == "2,1·10⁵"
    assert num(1.2e-5, 2) == "1,2·10⁻⁵"
    assert num(None) == "–" and num(float("nan")) == "–" and num(float("inf")) == "∞"


@pytest.fixture(scope="module")
def beam_step(tmp_path_factory):
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    p = tmp_path_factory.mktemp("rep_geo") / "v1.step"
    write_step([Part("Konzola", box(0, 0, 0, 200, 20, 10))], p)
    return p


def _mcp(tmp_path, body):
    pytest.importorskip("mcp")
    from mcp import Client
    from fea_core.mcp.server import build_server

    async def main():
        srv = build_server(base_dir=tmp_path / "mcp")
        async with Client(srv, mode="legacy") as c:
            async def ok(tool, **args):
                r = await c.call_tool(tool, args)
                assert not r.is_error, f"{tool}: {r.content[0].text}"
                return json.loads(r.content[0].text)
            await body(ok)
    anyio.run(main)


async def _beam(ok, beam_step, tmp_path):
    await ok("fea_study_new", step_path=str(beam_step), name="V1 konzola", workdir=str(tmp_path / "wd"))
    await ok("fea_materials_assign", part="*", material_name="Ocel S235")
    faces = await ok("fea_geometry_find_faces", x=0)
    await ok("fea_constraint_add_fixed", name="vetknuti", selection=[faces["faces"][0]["id"]])
    await ok("fea_loadcase_create", name="Sila 100 N")
    await ok("fea_load_add_force", case_name="Sila 100 N", name="konec", selection={"x": 200}, force=[0, 0, -100])


def test_report_without_results_mcp(tmp_path, beam_step):
    """Without a calculation: a protocol with the setup (parts, material, constraints, states, model image), without results."""
    out = tmp_path / "zadani.docx"

    async def body(ok):
        await _beam(ok, beam_step, tmp_path)
        r = await ok("fea_report_word", output_path=str(out), author="Test", subject="Konzola – zadání", language="cs")
        assert r["with_results"] is False and Path(r["path"]) == out and r["figures"] == 1
    _mcp(tmp_path, body)
    text, heads, ntab, nimg = doc_text(out)
    assert heads[0].startswith("Pevnostní výpočet MKP") and "1 Zadání výpočtu" in heads and "3 Výsledky" in heads
    assert "Výsledky nejsou k dispozici" in text and "Ocel S235" in text and "vetknutí – všechny posuvy nulové" in text
    assert "F = (0; 0; −100) N" in text and nimg == 1
    assert not (tmp_path / "zadani_obrazky").exists()                  # temporary images deleted


@pytest.mark.slow
def test_report_e2e_beam_fatigue(tmp_path, beam_step):
    """Console V1 through MCP: calculation + fatigue → protocol with the summary, reactions, maximum places and fatigue."""
    out = tmp_path / "v1.docx"

    async def body(ok):
        await _beam(ok, beam_step, tmp_path)
        await ok("fea_solve_run", timeout_s=600)
        await ok("fea_fatigue_run", history_curve={"load_case": "Sila 100 N", "kind": "sine", "amplitude": 3.0},
                 repeats=1e6)
        r = await ok("fea_report_word", output_path=str(out), required_safety=1.5, keep_images=True,
                     notes="První odstavec.\n\nDruhý odstavec.", language="cs")
        assert r["with_results"] and r["with_fatigue"] and not r["with_joints"]
        assert len(r["images"]) == 6 and all(Path(p).exists() for p in r["images"])   # model, vM, u, Re, 2× fatigue
    _mcp(tmp_path, body)
    text, heads, ntab, nimg = doc_text(out)
    for h in ("Souhrn", "2 Síť a řešič", "3.1 Stav „Sila 100 N“", "4 Únava", "5 Předpoklady, omezení a varování",
              "6 Závěr"):
        assert h in heads, h
    assert nimg == 6 and ntab >= 10
    row = next(l for l in text.splitlines() if l.startswith("vetknuti |") and "| 100 | 100 |" in l)   # Rz, |R|
    assert "·10⁻⁸" not in row and "·10⁻⁹" not in row                 # numerical noise → 0
    assert "hodnocení (k ≥ 1,5)" in text and "VYHOVUJE" in text
    assert "singularita" in text and "Palmgren–Miner" in text and "Druhý odstavec." in text


@pytest.mark.slow
def test_report_bolts_and_print(tmp_path):
    """S2 (bolt M10, assessment VDI/EN) and P4 (PLA printing, creep) – the chapters of the joints and of the 3D printing."""
    import verification.cases  # noqa: F401 – order of the imports (joint_cases ↔ run_all)
    from verification.joint_cases import s2_build
    from verification.print_cases import p4_build
    from fea_core.joints.assess import assess_joints
    from fea_core.solve import run_study
    for d in ("s2", "p4"):
        (tmp_path / d).mkdir()
    st, parts = s2_build(tmp_path / "s2")[:2]
    res = run_study(st, tmp_path / "s2" / "run", parts=parts)
    sess = session_from(st, st.geometry, res, parts, None, assess_joints(st, res, parts))
    r = build_report(sess, tmp_path / "s2.docx", ReportOptions(images=False))
    text, heads, _, nimg = doc_text(Path(r["path"]))
    assert "4 Šroubové a svarové spoje" in heads and nimg == 0
    assert "ISO 4017 M10×40 8.8" in text and "VDI 2230" in text and "EN 1993-1-8" in text
    assert "Šroub 1" in heads and "Šroub Šroub 1" not in heads

    st, parts = p4_build(tmp_path / "p4")[:2]
    res = run_study(st, tmp_path / "p4" / "run", parts=parts)
    r = build_report(session_from(st, st.geometry, res, parts), tmp_path / "p4.docx", ReportOptions(images=False))
    text, heads, _, _ = doc_text(Path(r["path"]))
    assert "4 3D tisk" in heads and "UPOZORNĚNÍ – TEČENÍ" in text and "Dlouhodobé zatížení (tečení)" in text


@pytest.mark.slow
def test_report_modal_buckling(tmp_path, beam_step):
    """Console V1: natural frequencies (table of f and effective masses, 3 shapes) and linear buckling (λ, critical force,
    1st shape, a warning about imperfections); then a study only with frequencies and an elastic support → "FEM modal analysis".
    """
    out1, out2 = tmp_path / "modal_buckling.docx", tmp_path / "modal_only.docx"

    async def body(ok):
        await _beam(ok, beam_step, tmp_path)
        await ok("fea_load_add_force", case_name="Tlak", name="tlak", selection={"x": 200}, force=[-1000, 0, 0])
        await ok("fea_modal_set", modes=4)
        await ok("fea_buckling_set", case_name="Tlak", modes=2)
        await ok("fea_solve_run", timeout_s=900)
        r = await ok("fea_report_word", output_path=str(out1), language="cs")
        assert r["figures"] >= 1 + 2 * 2 + 3 + 1                   # model×2, state maps, 3 natural shapes, buckling
        # natural frequencies only: without load states, the elastic support of the bottom instead of the clamping
        await ok("fea_buckling_set", enabled=False)
        await ok("fea_loadcase_delete", name="Tlak")
        await ok("fea_loadcase_delete", name="Sila 100 N")
        await ok("fea_constraint_add_elastic_support", name="dno", selection={"z": 0}, stiffness=50.0,
                 direction="all")
        await ok("fea_solve_run", timeout_s=900)
        await ok("fea_report_word", output_path=str(out2), images=False, language="cs")
    _mcp(tmp_path, body)
    text, heads, ntab, nimg = doc_text(out1)
    assert any(h.startswith("3.3 Vlastní frekvence") for h in heads), heads
    assert any(h.startswith("3.4 Vzpěr (lineární)") for h in heads), heads
    assert "Vlastní frekvence a podíly efektivní hmoty" in text and "m_ef,z [%]" in text
    assert "Vlastní tvar 3:" in text and "1. tvar ztráty stability" in text
    assert "UPOZORNĚNÍ – LINEÁRNÍ VZPĚR" in text and "imperfekc" in text and "F = (−1000; 0; 0) N" in text
    lines = text.splitlines()
    row1 = lines[next(i for i, l in enumerate(lines) if l.startswith("mód | f [Hz]")) + 1]
    assert row1.startswith("1 | ") and "·10⁻" not in row1, row1               # noise of the effective masses → 0
    assert "Vlastní frekvence (počet módů 4): f₁ = " in text and "λ₁ = " in text
    assert "CalculiX *FREQUENCY, počet módů 4" in text and "CalculiX *BUCKLE, počet tvarů 2" in text
    text, heads, _, nimg = doc_text(out2)
    assert heads[0].startswith("Modální analýza MKP") and nimg == 0
    assert "pružné uložení k = 50 N/mm³, ve všech směrech" in text and "Winklerovo podloží" in text
    assert any(h.startswith("3.1 Vlastní frekvence") for h in heads) and "Vzpěr" not in " ".join(heads)


def test_report_f7_input_without_results(tmp_path, beam_step):
    """Without a calculation: orthotropic material (table of the constants, axes, a warning about Re), elastic support, a refinement sphere,
    a separately meshed part and the settings of the natural frequencies in the chapter of the mesh.
    """
    out = tmp_path / "f7.docx"
    c = {"E1": 140000, "E2": 10000, "E3": 10000, "nu12": 0.3, "nu13": 0.3, "nu23": 0.4, "G12": 5000, "G13": 5000,
         "G23": 3500}

    async def body(ok):
        await _beam(ok, beam_step, tmp_path)
        await ok("fea_csys_create", name="valec", kind="cylindrical", origin=[0, 10, 5], z_axis=[1, 0, 0])
        await ok("fea_material_set_orthotropic", name="Kompozit", constants=c, csys="valec", part="*", rho=1.6e-9)
        await ok("fea_constraint_add_elastic_support", name="podlozi", selection={"z": 0}, stiffness=2.0)
        await ok("fea_mesh_refine_point", center=[100, 10, 10], radius_mm=5, size_mm=1)
        await ok("fea_mesh_separate_parts", parts=["Konzola"])
        await ok("fea_modal_set", modes=6)
        await ok("fea_report_word", output_path=str(out), images=False, language="cs")
    _mcp(tmp_path, body)
    text, heads, _, _ = doc_text(out)
    assert "Ortotropní materiály – inženýrské konstanty" in text and "1,4·10⁵ / 1·10⁴ / 1·10⁴" in text
    assert "(1 = r, 2 = θ, 3 = z)" in text and "neplatí hodnocení bezpečnosti vůči Re" in text
    assert "pružné uložení k = 2 N/mm³, kolmo k ploše" in text
    assert "Místní zjemnění (koule)" in text and "poloměr 5 mm: 1 mm" in text
    assert "Samostatně síťované díly | Konzola" in text and "počet módů 6" in text


def test_report_frame_input_without_results(tmp_path):
    """A study only with frames (example 6 – a hall frame) without a calculation: the section Frame system (nodes, members, sections, supports,
    loads of the members), an image of the model of the members.
    """
    from examples.frames import build
    st = build()
    sess = session_from(st, "", None, [], workdir=tmp_path)
    r = build_report(sess, tmp_path / "ram.docx", ReportOptions())
    text, heads, ntab, nimg = doc_text(Path(r["path"]))
    assert heads[0].startswith("Výpočet prutové soustavy") and "1.3 Prutová soustava" in heads
    assert "bez geometrie těles" in text and "IPE 300" in text and "HEA 260" in text
    assert "příhradový (jen N)" in text and ("začátek: My" in text or "konec: My" in text)
    assert "spojité zatížení prutu (globální osy)" in text and "vetknutí" in text
    assert nimg == len(st.load_cases) and not r["warnings"], r["warnings"]


@pytest.mark.slow
def test_report_frame_hall(tmp_path):
    """Example 6 (hall frame): calculation (statics of 3 states, frequencies, buckling by MYSTRAN if available) → protocol
    with the extremes per member, reactions, the diagrams My, the assessment EN 1993-1-1 and the mode shapes of the members.
    """
    from examples.frames import build
    from fea_core.mystran import MYSTRAN_EXE
    from fea_core.solve import run_study
    st = build()
    if not MYSTRAN_EXE.exists():
        st.buckling.enabled = False
    res = run_study(st, tmp_path / "run")
    sess = session_from(st, "", res, [], workdir=tmp_path)
    r = build_report(sess, tmp_path / "hala.docx", ReportOptions(keep_images=False))
    text, heads, ntab, nimg = doc_text(Path(r["path"]))
    assert any(h.startswith("3.1 Stav „Vlastní tíha“ – prutová soustava") for h in heads), heads
    assert any(h.startswith("Posouzení prutů podle EN 1993-1-1") for h in [h.split(" ", 1)[1] for h in heads if " " in h])
    assert "Extrémy vnitřních sil po prutech" in text and "Reakce uložení prutů" in text
    assert "Obrazec ohybového momentu My" in text and "Posouzení prutů EN 1993-1-1 – přehled využití" in text
    assert "Vlastní tvar 1:" in text and "γM0 = 1" in text
    if st.buckling.enabled:
        assert "1. tvar ztráty stability" in text and "UPOZORNĚNÍ – LINEÁRNÍ VZPĚR" in text
    assert nimg >= 3 * 2 + 3 + 1 and not [w for w in r["warnings"] if "Obrázek" in w], r["warnings"]



# ===================================================================== F8: contacts and dynamics (synthetic results)

def _f8_results():
    """A study and results without a calculation: a mesh of two blocks A/B (tests.test_perf_equivalence.box_mesh), a contact pair
    with friction, a sequence of two steps (the second partial, force–displacement curve) and dynamics (3 modes, harmonic, random
    vibration, shock) with the added mass and a probe.
    """
    import numpy as np
    from tests.test_perf_equivalence import box_mesh
    from fea_core.study import Study, Material, Fixed, LoadCase, Selection, Pressure
    from fea_core.results import Results, StepResult
    from fea_core.contacts.model import ContactPair, LoadStep, Motion
    from fea_core.dynamics.model import Excitation, Probe, PointMass
    from fea_core.dynamics.results import DynamicsResults
    m = box_mesh()
    n = len(m.nodes)
    st = Study("F8 syntetická", "x.step")
    st.add_material(Material("Ocel S235", 210000.0, 0.3, 7.85e-9, Re=235.0, Rm=360.0), default=True)
    for k in m.surfaces:
        st.selections[k] = Selection(k, [])
    st.constraints.append(Fixed("x0"))
    st.load_cases.append(LoadCase("p", [Pressure("x1", 1.0)]))
    st.contacts.append(ContactPair("A-B", "A", "B", kind="frictional", mu=0.2, initial="contact", gap=0.0))
    st.analysis.kind, st.analysis.branches = "nonlinear", False
    st.analysis.sequence = [LoadStep("náběh", "p", 0.5),
                            LoadStep("plné", "p", 1.0, prescribed=[Motion("x1", "disp", 1, [[0, 0], [1, 0.1]])])]
    ds = st.dynamics
    ds.enabled, ds.modes = True, 3
    ds.harmonic.enabled, ds.harmonic.f_min, ds.harmonic.f_max = True, 10.0, 1000.0
    ds.harmonic.excitation = [Excitation("base_acc", dof=3, table=[[10, 1.0], [1000, 1.0]])]
    ds.random.enabled, ds.random.asd = True, [[20.0, 0.01], [2000.0, 0.01]]
    ds.shock.enabled = True
    ds.probes = [Probe("p1", point=[40.0, 15.0, 10.0])]
    st.masses.append(PointMass("x1", 1e-3, point=[45.0, 15.0, 10.0], name="motor"))
    rng = np.random.default_rng(1)
    steps = []
    for k, name in enumerate(("náběh", "plné")):
        U = rng.normal(0, 1e-3, (n, 3))
        S = rng.normal(0, 10.0, (n, 6))
        s = StepResult(name, U, S)
        s.fields.update(CPRESS=np.abs(rng.normal(5, 1, n)), STATUS=rng.integers(-1, 3, n).astype(float),
                        OPENING=np.zeros(n), CSLIP=np.zeros(n), CSHEAR=np.zeros(n), COPEN=np.zeros(n))
        s.contacts = {"A-B": {"Fn_N": 600.0 * (k + 1), "Ft_N": 50.0, "area_mm2": 600.0, "area_contact_mm2": 580.0,
                              "contact_fraction": 0.97, "p_max_MPa": 7.5, "p_mean_MPa": 5.0, "open_max_mm": 0.0,
                              "slip_max_mm": 1e-4, "nodes_open": 1, "nodes_slip": 2, "nodes_stick": 30, "mu": 0.2}}
        steps.append(s)
    steps[1].partial = True
    steps[1].curves = {"x1": {"t": [1.25, 1.5, 1.75], "u": [[0.025, 0, 0], [0.05, 0, 0], [0.07, 0, 0]],
                              "F": [[100, 0, 0], [180, 0, 0], [210, 0, 0]], "dof": 1, "kind": "disp"}}
    res = Results(m, steps, ["ČÁSTEČNÝ VÝSLEDEK (nl_seq): test"],
                  {"backend": "ccx", "nonlinear": {"mode": "sequence", "jobs": ["nl_seq"],
                                                   "partial": {"nl_seq": {"last_step": 2, "of_steps": 2, "cutbacks": 3,
                                                                          "last_step_time": 0.75}}}})
    K, nf = 3, 30
    f = np.geomspace(10, 1000, nf)
    shapes = rng.normal(0, 1, (K, n, 3)).astype(np.float32)
    stress = rng.normal(0, 100, (K, n, 6)).astype(np.float32)
    dr = DynamicsResults(np.array([120.0, 340.0, 800.0]), np.full(K, 0.02), shapes, stress, rng.normal(0, 1e-3, (K, 6)),
                         np.abs(rng.normal(0, 1e-5, (K, 6))), np.zeros((K, 3)),
                         {"mass_norm_error": 1e-7, "gamma_error": 1e-6, "mode_equilibrium": 1e-8,
                          "mass_fraction_xyz": [0.5, 0.6, 0.95]}, 2.4e-5)
    U = (rng.normal(0, 1e-3, (nf, 3)) + 1j * rng.normal(0, 1e-3, (nf, 3)))
    Sp = (rng.normal(0, 5, (nf, 6)) + 1j * rng.normal(0, 5, (nf, 6)))
    dr.harmonic = {"f": f, "q": rng.normal(0, 1, (nf, K)) + 0j, "c": np.zeros((nf, 0)), "ub": np.zeros((nf, 3)) + 0j,
                   "env_U": np.abs(rng.normal(0, 1e-3, n)), "env_S": np.abs(rng.normal(0, 20, n)),
                   "env_U_at": np.zeros(n, int), "env_S_at": np.full(n, 5), "probes": {"p1": {"U": U, "S": Sp}},
                   "reaction": U * 100, "res_index": np.zeros(0, int),
                   "sweep": {"D": np.full(n, 1e-4), "summary": {"max_damage": 1e-4, "at": [0, 0, 0], "rate_oct_min": 1,
                                                                "sweeps": 1}}}
    dr.random = {"f": f, "asd_g2": np.full(nf, 0.01), "U_rms": np.abs(rng.normal(0, 1e-3, (n, 3))),
                 "S_rms": np.abs(rng.normal(0, 5, n)), "D": np.full(n, 1e-3), "life_h": np.full(n, 500.0),
                 "probes": {"p1": {"U_psd": np.full((nf, 3), 1e-8), "A_psd_g2": np.full((nf, 3), 0.02)}},
                 "summary": {"g_rms_input": 4.4, "duration_h": 1.0, "method": "dirlik", "max_U_rms_mm": 0.01,
                             "max_mises_rms_MPa": 12.0, "max_mises_3sigma_MPa": 36.0, "mises_rms_at": [0, 0, 0],
                             "max_damage": 1e-3, "min_life_h": 500.0, "damage_at": [0, 0, 0]}}
    t = np.linspace(0, 0.022, 50)
    dr.shock = {"Sa": np.ones(K), "q_peak": np.ones(K), "U": np.abs(rng.normal(0, 1e-3, (n, 3))),
                "S": np.abs(rng.normal(0, 5, (n, 6))), "mises": np.abs(rng.normal(0, 30, n)), "pulse_t": t,
                "pulse_a": 50 * 9810 * np.sin(np.pi * np.minimum(t / 0.011, 1)), "srs_f": f,
                "srs": 9810 * 60 * np.ones(nf),
                "summary": {"pulse": "halfsine", "A_g": 50, "tau_s": 0.011, "combine": "cqc", "missing_mass": True,
                            "zpa_g": 50, "max_U_mm": 0.01, "max_mises_MPa": 80.0, "at": [0, 0, 0]}}
    dr.probes = [{"name": "p1", "nodes": np.array([0]), "w": np.array([1.0]), "point": [40, 15, 10], "quantity": "U"}]
    dr.info["etalon"] = {"harmonic": {"max_rel_error": 3e-5, "note": "bez statické korekce"}}
    res.dynamics = dr
    return st, res


@pytest.mark.parametrize("lang", ["cs", "en"])
def test_report_f8_sections(tmp_path, lang):
    """The sections Contacts and Dynamics from synthetic results: tables, maps, charts and the mandatory red warnings;
    the conditional texts "no contact" / "perfect thermal contact"; the protocol also goes in English.
    """
    st, res = _f8_results()
    sess = session_from(st, None, res, workdir=tmp_path)
    r = build_report(sess, tmp_path / f"f8_{lang}.docx", ReportOptions(language=lang))
    text, heads, ntab, nimg = doc_text(Path(r["path"]))
    assert not [w for w in r["warnings"] if not w.startswith("Model figure")], r["warnings"]
    if lang == "en":
        assert nimg >= 10
        return
    assert any("Kontakty a nelineární výpočet" in h for h in heads) and any(h.endswith("Dynamika") for h in heads)
    assert heads[0].startswith("Nelineární pevnostní výpočet MKP (kontakt) a dynamika")
    for t in ("UPOZORNĚNÍ – NELINEÁRNÍ VÝPOČET", "UPOZORNĚNÍ – SOUČINITEL TŘENÍ", "UPOZORNĚNÍ – PENALIZACE",
              "UPOZORNĚNÍ – PŘÍRŮSTKY A KONVERGENCE", "UPOZORNĚNÍ – TLUMENÍ", "ČÁSTEČNÝ VÝSLEDEK (nl_seq)",
              "Styk s třením", "zadáno uživatelem", "Posloupnost kroků zatížení", "Křivky síla–dráha",
              "Vlastní frekvence a efektivní hmoty", "Odezva v sondách – harmonická", "Spektrální únava",
              "Kontrola modální superpozice proti CalculiXu", "Přidané hmoty", "bodová hmota 1 kg"):
        assert t in text, t
    assert "bez kontaktu, bez tření, bez rozevření" not in text and "nelze superponovat" in text
    assert nimg >= 10                    # pressure + state × 2 steps, curve, profiles, FRF, envelope, RMS, D, PSD, SRS, shock
