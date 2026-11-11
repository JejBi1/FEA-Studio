"""F9 – connection of the core to the GUI and MCP: every new F9 tool through the MCP protocol and the core functions with the same arguments
as the GUI passes to them (section_forces, combine_results, compute_envelope, transform_*, compute_safety_field).

The fast tests are synthetic (without Gmsh and ccx); the model cantilever (mesh + ccx) is in the tests marked slow.
Cantilever 200×20×10, clamped at x = 0, three states at the end x = 200: G = (0; 0; −100), Q = (0; −50; 0), A = (+100; 0; 0).
"""
from __future__ import annotations

import base64
import csv
import json
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import anyio
import numpy as np
import pytest

from fea_core.combinations import (Combination, EnvelopeResult, combine_results, compute_combination,
                                   compute_envelope, validate_combination)
from fea_core.export_csv import export_envelope_csv, export_nodal_csv
from fea_core.export_vtu import (export_combination_vtu, export_results_vtu, export_step_vtu, safe_name)
from fea_core.imprint import Imprint
from fea_core.mesh import Mesh
from fea_core.results import Results, StepResult
from fea_core.study import Material, Study


# ===================================================================== helpers

def read_vtu_array(path: Path, name: str) -> np.ndarray:
    """Minimal reader of VTK XML inline binary (like vtkXMLDataParser): the header UInt32 and the data are base64 SEPARATELY."""
    root = ET.parse(path).getroot()
    assert root.attrib.get("header_type") == "UInt32"
    for da in root.iter("DataArray"):
        if da.attrib["Name"] != name:
            continue
        txt = "".join(da.text.split())
        head_chars = 8                                   # 4 bytes of the header = 8 base64 characters (with the padding ==)
        n_bytes = struct.unpack("<I", base64.b64decode(txt[:head_chars]))[0]
        raw = base64.b64decode(txt[head_chars:])
        assert len(raw) == n_bytes, f"{name}: hlavička {n_bytes} B, dat {len(raw)} B"
        dt = {"Float32": "<f4", "Int32": "<i4", "UInt8": "u1"}[da.attrib["type"]]
        a = np.frombuffer(raw, dtype=dt)
        nc = int(da.attrib["NumberOfComponents"])
        return a.reshape(-1, nc) if nc > 1 else a
    raise KeyError(name)


def tet_model(n_steps: int = 2):
    """One tet10 element with two states (like tests/test_export.py) and a study with steel."""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5],
                      [.5, 0, .5], [0, .5, .5]], float)
    mesh = Mesh(nodes=nodes, tets=np.arange(10, dtype=np.int32)[None, :], tet_part=np.array([0], np.int32),
                part_names=["D1"])
    steps = []
    for k in range(n_steps):
        S = np.zeros((10, 6))
        S[:, k % 3] = 50.0 * (k + 1)
        steps.append(StepResult(name=f"S{k + 1}", U=np.full((10, 3), 0.1 * (k + 1)), S=S,
                                reactions={"v": np.array([0.0, 0.0, 100.0 * (k + 1)])}))
    st = Study(name="T", geometry="")
    st.add_material(Material("Ocel", 210000.0, 0.3, 7.85e-9, Re=235.0, Rm=360.0, alpha=12e-6), default=True)
    return mesh, Results(mesh=mesh, steps=steps), st


class _Lc:
    def __init__(self, name):
        self.name = name


# ===================================================================== fast tests (without mesh and ccx)

def test_vtu_binary_encoding_roundtrip(tmp_path):
    """Header and data base64 separately – a VTK-style reader returns exactly the fields written."""
    mesh, res, st = tet_model()
    p = export_step_vtu(mesh, res.steps[0], tmp_path / "a.vtu")
    assert np.allclose(read_vtu_array(p, "Points"), mesh.nodes, atol=1e-6)
    assert np.allclose(read_vtu_array(p, "Displacement"), 0.1, atol=1e-6)
    assert np.allclose(read_vtu_array(p, "von_Mises_Stress"), 50.0, atol=1e-4)
    assert np.array_equal(read_vtu_array(p, "connectivity"), np.arange(10))
    assert read_vtu_array(p, "types")[0] == 24
    # number of bytes that is not a multiple of three (remainder 1 and 2 in base64)
    for n in (1, 2, 4, 5):
        assert np.allclose(read_vtu_array(export_step_vtu(mesh, None, tmp_path / f"n{n}.vtu",
                                                          {"f": np.arange(10.0)}), "f"), np.arange(10.0))


def test_vtu_names_sanitized_and_pvd_escaped(tmp_path):
    mesh, res, st = tet_model()
    res.steps[0].name = 'a\\..\\b:c"<>'
    st.combinations.append(Combination(name='../x"y&', factors={res.steps[0].name: 1.0, "S2": 1.0}))
    out = tmp_path / "out"
    warns: list[str] = []
    files = export_results_vtu(mesh, res, out, base_name="../ev il", study=st, warnings=warns)
    assert not warns
    for f in files:
        assert f.parent == out and f.exists()
    assert {p.name for p in tmp_path.iterdir()} == {"out"}               # nothing outside the output folder
    pvd = ET.parse(out / (safe_name("../ev il") + ".pvd")).getroot()     # valid XML even with quotes in the name
    assert len(pvd.findall(".//DataSet")) == 3
    assert safe_name("a\\..\\b:c") == "a_.._b_c" and safe_name("..") == "x" and safe_name("Příklad 1") == "Příklad_1"


def test_vtu_unusable_combination_is_reported(tmp_path):
    mesh, res, st = tet_model()
    st.combinations.append(Combination(name="dobra", factors={"S1": 1.0}))
    st.combinations.append(Combination(name="zla", factors={"neexistuje": 1.0}))
    warns: list[str] = []
    files = export_results_vtu(mesh, res, tmp_path, study=st, warnings=warns)
    assert len(warns) == 1 and "zla" in warns[0] and "neexistuje" in warns[0]
    assert any("_K01_dobra" in f.name for f in files)
    with pytest.warns(UserWarning, match="zla"):                         # without the list of warnings, warnings.warn is printed
        export_results_vtu(mesh, res, tmp_path / "w", study=st)


def test_envelope_export_vtu_and_csv(tmp_path):
    mesh, res, st = tet_model()
    st.combinations += [Combination(name="K1", factors={"S1": 1.0}), Combination(name="K2", factors={"S2": 1.0}),
                        Combination(name="Obalka", envelope=True, envelope_combinations=["K1", "K2"])]
    env = compute_combination(st, res, st.combinations[2])
    assert isinstance(env, EnvelopeResult)                               # ne StepResult
    p = export_combination_vtu(mesh, env, tmp_path / "env.vtu")
    assert np.allclose(read_vtu_array(p, "Envelope_von_mises"), 100.0, atol=1e-4)
    assert (read_vtu_array(p, "Governing_Combination_Index") == 1).all()
    c = export_envelope_csv(mesh, env, tmp_path / "env.csv")
    rows = list(csv.reader(open(c, encoding="utf-8")))
    assert rows[0][-2:] == ["envelope_von_mises", "governing_combination"]
    assert len(rows) == 11 and rows[1][-1] == "K2" and float(rows[1][-2]) == pytest.approx(100.0)
    files = export_results_vtu(mesh, res, tmp_path / "all", study=st)
    assert sum("_K0" in f.name for f in files) == 3


def test_csv_extra_fields_validated_and_vector(tmp_path):
    mesh, res, st = tet_model()
    p = export_nodal_csv(mesh, res.steps[0], tmp_path / "a.csv",
                         {"skalar": np.arange(10.0), "vektor": np.ones((10, 3))})
    rows = list(csv.reader(open(p, encoding="utf-8")))
    assert rows[0][-4:] == ["skalar", "vektor_0", "vektor_1", "vektor_2"]
    assert len(rows) == 11 and float(rows[3][-4]) == 2.0
    assert rows[1][0] == "1" and float(rows[1][rows[0].index("von_mises_MPa")]) == pytest.approx(50.0)
    with pytest.raises(ValueError, match="kratke"):
        export_nodal_csv(mesh, res.steps[0], tmp_path / "b.csv", {"kratke": np.arange(3.0)})


def test_csv_fast_for_large_meshes(tmp_path):
    """200 thousand nodes previously took 4.5 s (f-strings per node)."""
    import time
    n = 200_000
    mesh = Mesh(nodes=np.random.default_rng(0).random((n, 3)), tets=np.zeros((0, 10), np.int32),
                tet_part=np.zeros(0, np.int32), part_names=["D1"])
    step = StepResult(name="s", U=np.random.default_rng(1).random((n, 3)), S=np.random.default_rng(2).random((n, 6)))
    t0 = time.time()
    export_nodal_csv(mesh, step, tmp_path / "big.csv")
    assert time.time() - t0 < 6.0


def test_combination_step_named_like_index_and_joints_dropped():
    mesh, res, st = tet_model()
    res.steps[0].name = "1"                        # the real name of the state "1" must not overwrite index 1 (= the second state)
    res.steps[1].name = "druhy"
    res.steps[0].joints = {"bolts": {"b1": {"utilisation": 0.5, "F": [1.0, 2.0]}}}
    c = combine_results(st, res, Combination(name="k", factors={"1": 1.0}))
    assert np.allclose(c.U, 0.1)                   # a state named "1" is the first, not index 1
    c0 = combine_results(st, res, Combination(name="k0", factors={"0": 1.0, "LoadCase_2": 1.0}))
    assert np.allclose(c0.U, 0.3)                  # an alias by index still works
    assert c.joints == {} and c0.joints == {}      # the utilisation of the joints is not combined linearly


def test_validate_combination_and_safety_envelope():
    mesh, res, st = tet_model()
    st.load_cases = [_Lc("S1"), _Lc("S2")]
    validate_combination(st, Combination(name="ok", factors={"S1": 1.35, "S2": 1.5}))
    with pytest.raises(ValueError, match="unknown load cases"):
        validate_combination(st, Combination(name="x", factors={"S3": 1.0}))
    with pytest.raises(ValueError, match="has no load case factors defined"):
        validate_combination(st, Combination(name="x"))
    with pytest.raises(ValueError, match="Unknown envelope field"):
        validate_combination(st, Combination(name="e", envelope=True, envelope_combinations=["S1"],
                                             envelope_field="nesmysl"))
    with pytest.raises(ValueError, match="unknown source"):
        validate_combination(st, Combination(name="e", envelope=True, envelope_combinations=["nic"]))
    # envelope_field='safety' (previously ValueError): the minimum of the safeties of the combinations, Re = 235
    st.combinations += [Combination(name="K1", factors={"S1": 1.0}), Combination(name="K2", factors={"S2": 1.0}),
                        Combination(name="E", envelope=True, envelope_type="min", envelope_field="safety",
                                    envelope_combinations=["K1", "K2"])]
    validate_combination(st, st.combinations[2])
    env = compute_envelope(st, res, st.combinations[2])
    assert env.field_name == "safety" and np.allclose(env.values, 235.0 / 100.0)
    assert set(env.governing_combinations) == {"K2"}


def test_assessment_reports_used_limit_and_print_parts():
    from fea_core.assessment import assessment_summary, compute_safety_field
    mesh, res, st = tet_model()
    st.materials["Ocel"] = Material("Ocel", 210000.0, 0.3, 7.85e-9)             # without Re and Rm → replacement 235 MPa
    summ = assessment_summary(st, mesh, res.steps[0])
    assert summ["parts"]["D1"]["limit_MPa"] == 235.0 and summ["warnings"]
    # printed material: the print_safety from the result (Tsai-Wu) is taken, not the criterion of the Stress Tool
    st.materials["Ocel"] = Material("Print", 3000.0, 0.36, 1.2e-9, Rm=45.0, E_z=2500.0, Rm_z=27.0)
    res.steps[0].fields["print_safety"] = np.full(10, 1.7)
    s, w = compute_safety_field(st, mesh, res.steps[0])
    assert np.allclose(s, 1.7) and not w
    res.steps[0].fields.pop("print_safety")
    s, w = compute_safety_field(st, mesh, res.steps[0])
    assert np.allclose(s, 10.0) and "print_safety" in w[0]


def test_principal_vectors_edge_cases():
    from fea_core.post_vectors import principal_vector_segments
    rng = np.random.default_rng(3)
    S = rng.normal(size=(500, 6)) * 50
    nodes = rng.random((500, 3))
    d1 = principal_vector_segments(nodes, S, which=1, max_points=1)      # previously division by zero
    assert len(d1["starts"]) <= 2
    d = principal_vector_segments(nodes, S, which=3, max_points=100)
    assert len(d["starts"]) == 100
    from fea_core.post_vectors import principal_stresses_and_directions
    w, v = principal_stresses_and_directions(S)
    idx = d["indices"]
    assert np.allclose(d["values"], w[idx, 2])
    assert np.allclose(np.linalg.norm(d["ends"] - d["starts"], axis=1), np.abs(d["values"]))
    e = principal_vector_segments(nodes[:5], np.zeros((5, 6)), which=1)   # zero stress
    assert len(e["starts"]) == 5 or len(e["starts"]) == 0


def test_imprint_dict_matches_study_serialization():
    imp = Imprint(name="O", part="D", center=[1, 2, 3], normal=[0, 0, 1], shape="rectangle", dx=4, dy=5, size=0.5)
    assert imp.to_dict() == imp.to_mesh_dict()
    assert imp.to_dict() == json.loads(json.dumps(asdict(imp)))          # Study.to_dict zapisuje asdict
    assert Imprint.from_dict(imp.to_dict()) == imp


def test_element_forces_with_temperature():
    """Free thermal expansion (u = αΔT·x): with ΔT zero forces of the element, without ΔT non-zero (σ = C(ε − αΔT))."""
    from fea_core.recovery import element_nodal_forces
    mesh, res, st = tet_model(1)
    alpha, dT = 12e-6, 50.0
    U = alpha * dT * mesh.nodes
    assert np.abs(element_nodal_forces(st, mesh, U, np.array([0]), np.full(10, dT))).max() < 1e-8
    assert np.abs(element_nodal_forces(st, mesh, U, np.array([0]))).max() > 1.0
    # the sum of the element forces is the equilibrium (no external load)
    f = element_nodal_forces(st, mesh, np.random.default_rng(0).random((10, 3)) * 1e-3, np.array([0]))
    assert np.abs(f.sum(axis=1)).max() < 1e-6 * np.abs(f).max() + 1e-9


# ===================================================================== cantilever through MCP (mesh + ccx)

def _make_model(tmp_path_factory):
    from mcp import Client
    from fea_core.mcp.server import build_server
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    d = tmp_path_factory.mktemp("f9glue")
    step = d / "v1.step"
    write_step([Part("Konzola", box(0, 0, 0, 200, 20, 10))], step)
    srv = build_server(base_dir=d / "mcp")

    async def setup():
        async with Client(srv, mode="legacy") as c:
            async def ok(tool, **a):
                r = await c.call_tool(tool, a)
                assert not r.is_error, f"{tool}: {r.content[0].text}"
                return json.loads(r.content[0].text)
            await ok("fea_study_new", step_path=str(step), name="Konzola F9", workdir=str(d / "wd"))
            await ok("fea_materials_assign", part="*", material_name="Ocel S235",
                     custom_props={"E": 210000, "nu": 0.3})
            f0 = await ok("fea_geometry_find_faces", x=0)
            await ok("fea_constraint_add_fixed", name="vetknuti", selection=[f0["faces"][0]["id"]])
            for name, force in (("G", [0, 0, -100]), ("Q", [0, -50, 0]), ("A", [100, 0, 0])):
                await ok("fea_loadcase_create", name=name)
                await ok("fea_load_add_force", case_name=name, name=f"konec_{name}", selection={"x": 200}, force=force)
            await ok("fea_csys_create", name="Valec", kind="cylindrical", origin=[0, 10, 5], z_axis=[1, 0, 0])
            await ok("fea_mesh_generate", global_size=5.0)
            r = await ok("fea_solve_run", timeout_s=600, backend="ccx")
            assert all(s["equilibrium"]["ok"] for s in r["steps"])
    anyio.run(setup)
    return srv, d


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return _make_model(tmp_path_factory)


def call(srv, body):
    """body(ok, err, c): ok(tool, **args) → JSON, err(tool, **args) → text chyby."""
    from mcp import Client

    async def main():
        async with Client(srv, mode="legacy") as c:
            async def raw(tool, **a):
                r = await c.call_tool(tool, a)
                return r.is_error, r.content[0].text

            async def ok(tool, **a):
                e, t = await raw(tool, **a)
                assert not e, f"{tool}: {t}"
                return json.loads(t)

            async def err(tool, **a):
                e, t = await raw(tool, **a)
                assert e, f"{tool} měl skončit chybou: {t}"
                return t
            await body(ok, err)
    anyio.run(main)


def test_section_force_exact_on_polynomial_fields(model):
    """Exact test of the section without the influence of the solver: a linear and a quadratic field of the displacements of tet10 is reproduced exactly
    (uniaxial tension, pure bending) → the section forces and moments = analytics to 1e-9, even if the section zigzags through the elements.
    """
    srv, _ = model
    res, study = srv.fea_session.results, srv.fea_session.study
    mesh = res.mesh
    from fea_core.section_forces import section_forces
    E, nu = 210000.0, 0.3
    X = mesh.nodes
    A = 20.0 * 10.0
    # a) uniaxial tension εx = 1e-3 (σx = E·ε with free transverse contraction)
    eps = 1e-3
    U = np.column_stack([eps * X[:, 0], -nu * eps * X[:, 1], -nu * eps * X[:, 2]])
    r = section_forces(study, mesh, StepResult(name="t", U=U, S=np.zeros((len(X), 6))), [101.3, 0, 0], [1, 0, 0])
    assert r["N"] == pytest.approx(E * eps * A, rel=1e-9)
    assert abs(r["Vy"]) < 1e-6 and abs(r["Vz"]) < 1e-6
    # b) pure bending about y: ux = κ·x·(z − 5), σx = E·κ·(z − 5) → My = E·κ·Iy, N = 0
    kap = 1e-5
    zc = X[:, 2] - 5.0
    U = np.column_stack([kap * X[:, 0] * zc, -nu * kap * X[:, 1] * zc,
                         -0.5 * kap * X[:, 0] ** 2 - 0.5 * nu * kap * (zc ** 2 - (X[:, 1] - 10.0) ** 2)])
    Iy = 20.0 * 10.0 ** 3 / 12.0
    for x0 in (60.7, 101.3):
        r = section_forces(study, mesh, StepResult(name="b", U=U, S=np.zeros((len(X), 6))), [x0, 0, 0], [1, 0, 0])
        assert r["My"] == pytest.approx(E * kap * Iy, rel=1e-9) and abs(r["N"]) < 1e-6 * E * kap * Iy
        assert abs(r["Vy"]) < 1e-6 and abs(r["Vz"]) < 1e-6
    # c) opposite normal = opposite sign of the internal forces
    r1 = section_forces(study, mesh, StepResult(name="b", U=U, S=np.zeros((len(X), 6))), [101.3, 0, 0], [-1, 0, 0])
    assert r1["My"] == pytest.approx(E * kap * Iy, rel=1e-9) or r1["My"] == pytest.approx(-E * kap * Iy, rel=1e-9)


@pytest.mark.slow
def test_section_force_equilibrium_cantilever(model):
    """Section of the cantilever at x = 100: the forces and moments correspond to the load of the free body x > 100. The displacements in the .frd have 6 valid
    digits → residuals of the nodal forces up to ~3 N at a load of 100 N (noise 5e-7 in u gives the same residual), hence the tolerance
    3 %; the exact agreement 1e-9 is in the test with the polynomial fields above.
    """
    srv, _ = model
    sess = srv.fea_session
    res, study = sess.results, sess.study
    from fea_core.section_forces import section_forces
    r = section_forces(study, res.mesh, res.steps[0], [100, 0, 0], [1, 0, 0], ["Konzola"])   # shapes as in the GUI
    c = np.array(r["centroid"])
    F = np.array([0.0, 0.0, -100.0])
    M = np.cross(np.array([200.0, 10.0, 5.0]) - c, F)                   # moment of the load about the centroid of the section
    assert abs(r["N"]) < 3.0 and abs(r["Vy"]) < 3.0
    assert r["Vz"] == pytest.approx(-100.0, rel=3e-2)
    assert np.allclose(r["F_global"], F, atol=3.0)
    assert np.allclose(r["M_global"], M, rtol=3e-2, atol=300.0), (r["M_global"], M)
    assert r["My"] == pytest.approx(M[1], rel=3e-2) and abs(r["My"] - 1e4) < 300      # M = P·(200 − 100)
    # second state: transverse force in y → Vy, Mz;  third: tension N
    r2 = section_forces(study, res.mesh, res.steps[1], [100, 0, 0], [1, 0, 0])
    assert r2["Vy"] == pytest.approx(-50.0, rel=3e-2) and r2["Mz"] == pytest.approx(-5000.0, rel=3e-2)
    r3 = section_forces(study, res.mesh, res.steps[2], [100, 0, 0], [1, 0, 0])
    assert r3["N"] == pytest.approx(100.0, rel=3e-2) and abs(r3["Vz"]) < 3.0
    # another section: the same shear force, the moment grows towards the clamping
    r4 = section_forces(study, res.mesh, res.steps[0], [50, 0, 0], [1, 0, 0])
    assert r4["Vz"] == pytest.approx(-100.0, rel=3e-2) and r4["My"] == pytest.approx(15000.0, rel=3e-2)
    with pytest.raises(ValueError, match="does not cut"):
        section_forces(study, res.mesh, res.steps[0], [500, 0, 0], [1, 0, 0])

    async def body(ok, err):
        d = await ok("fea_results_section_force", point=[100, 0, 0], normal=[1, 0, 0], step="G")
        assert d["step"] == "G" and d["Vz"] == pytest.approx(-100.0, rel=3e-2)
        assert "nonzero" in await err("fea_results_section_force", point=[100, 0, 0], normal=[0, 0, 0])
        assert "does not cut" in await err("fea_results_section_force", point=[500, 0, 0], normal=[1, 0, 0])
    call(srv, body)


@pytest.mark.slow
def test_combination_tools_keep_results_and_export(model, tmp_path):
    srv, _ = model
    sess = srv.fea_session
    res_before = sess.results

    async def body(ok, err):
        a = await ok("fea_combination_add", name="MSU", factors={"G": 1.35, "Q": 1.5})
        assert a["total_combinations"] == 1
        assert sess.results is res_before                                  # a combination is only post-processing
        assert "unknown load cases" in await err("fea_combination_add", name="zla", factors={"X": 1.0})
        await ok("fea_combination_add", name="MSP", factors={"G": 1.0, "Q": 0.7})
        await ok("fea_combination_add", name="Obalka", factors={}, envelope=True, envelope_type="min",
                 envelope_combinations=["MSU", "MSP"], envelope_field="safety")
        assert "unknown source combinations" in await err("fea_combination_add", name="O2", factors={}, envelope=True,
                                               envelope_combinations=["nic"])
        g = await ok("fea_combination_generate_en1990", cases=[
            {"name": "G", "action_type": "permanent", "subtype": "self_weight"},
            {"name": "Q", "action_type": "variable", "category": "B", "leading": True}], formula="6.10")
        assert g["generated_count"] >= 1 and sess.results is res_before
        assert "do not exist" in await err("fea_combination_generate_en1990", cases=[
            {"name": "Neni", "action_type": "permanent", "subtype": "self_weight"}])
        lst = await ok("fea_combination_list")
        names = [c["name"] for c in lst["combinations"]]
        assert {"MSU", "MSP", "Obalka"} <= set(names)
        assert "is used in the envelope" in await err("fea_combination_delete", name="MSU")      # used by the envelope
        # the summary of the results contains the assessment (previously never)
        sm = await ok("fea_results_summary", step="G")
        ass = sm["steps"][0]["assessment"]
        assert "assessment_error" not in sm["steps"][0] and ass["min_safety"] > 0 and ass["parts"]["Konzola"]["limit_MPa"] == 235

        # export: all states + combinations + envelope, VTU and CSV
        ex = await ok("fea_results_export", format="all", output_dir=str(tmp_path / "ex"))
        assert not ex.get("warnings"), ex
        fn = {Path(f).name for f in ex["exported_files"]}
        assert any("_K0" in n and n.endswith("_Obalka.csv") for n in fn) and any(n.endswith(".pvd") for n in fn)
        env_vtu = [f for f in ex["exported_files"] if "_K0" in f and f.endswith("_Obalka.vtu")]
        assert env_vtu
        one = await ok("fea_results_export", format="vtu", output_dir=str(tmp_path / "ex1"), step="Obalka")
        assert len(one["exported_files"]) == 1
        one_csv = await ok("fea_results_export", format="csv", output_dir=str(tmp_path / "ex1"), step="G")
        assert Path(one_csv["exported_files"][0]).exists()
        assert "Nejprve" not in await err("fea_results_export", format="vtu", step="neexistuje")
    call(srv, body)

    # VTU reader: the displacement of the state G in the file = the displacement in the result
    vtu = next(f for f in (tmp_path / "ex").glob("*_01_G.vtu"))
    U = read_vtu_array(vtu, "Displacement")
    assert np.allclose(U, res_before.steps[0].U, atol=1e-5 + 1e-5 * np.abs(res_before.steps[0].U).max())
    assert np.allclose(read_vtu_array(vtu, "Points"), res_before.mesh.nodes, atol=1e-4)
    env = next((tmp_path / "ex").glob("*_K*_Obalka.vtu"))
    assert read_vtu_array(env, "Envelope_safety").shape[0] == len(res_before.mesh.nodes)


@pytest.mark.slow
def test_core_functions_with_gui_argument_shapes(model):
    """Core functions exactly as the GUI calls them (main.py, export_dialog, section_dialog)."""
    srv, _ = model
    sess = srv.fea_session
    res, study = sess.results, sess.study
    mesh = res.mesh
    from fea_core import csys
    from fea_core.assessment import compute_safety_field
    st0 = res.steps[0]

    comb = Combination(name="GUI", factors={"G": 1.35, "Q": 1.5})
    c = combine_results(study, res, comb)
    assert isinstance(c, StepResult)
    assert np.allclose(c.U, 1.35 * res.steps[0].U + 1.5 * res.steps[1].U)
    assert np.allclose(c.S, 1.35 * res.steps[0].S + 1.5 * res.steps[1].S)
    env_c = Combination(name="OB", envelope=True, envelope_type="max", envelope_combinations=["GUI", "G"])
    study.combinations = [comb, env_c]
    env = compute_envelope(study, res, env_c)
    assert isinstance(env, EnvelopeResult) and env.values.shape == (len(mesh.nodes),)
    assert env.governing_summary["GUI"] > 0
    assert isinstance(compute_combination(study, res, env_c), EnvelopeResult)
    study.combinations = []

    # local system (cylindrical, axis x): σz = σxx, ur/uθ from the components in the plane yz
    S_loc = csys.transform_stress(study, "Valec", st0.S, mesh.nodes)
    U_loc = csys.transform_displacements(study, "Valec", st0.U, mesh.nodes)
    assert S_loc.shape == (len(mesh.nodes), 6) and np.allclose(S_loc[:, 2], st0.S[:, 0], atol=1e-6)
    assert np.allclose(U_loc[:, 2], st0.U[:, 0])
    assert np.allclose(np.hypot(U_loc[:, 0], U_loc[:, 1]), np.hypot(st0.U[:, 1], st0.U[:, 2]), atol=1e-9)
    assert csys.stress_labels(study, "Valec")[0] == "σr" and csys.displacement_labels(study, None)[0] == "ux"

    # safety as a field (N,) – the GUI takes [0] and clips it to 10
    for crit in ("tresca", "mohr_coulomb", None):
        sf, warns = compute_safety_field(study, mesh, st0, crit)
        assert sf.shape == (len(mesh.nodes),) and 0 < sf.min() <= 10 and isinstance(warns, list)
    sf, _ = compute_safety_field(study, mesh, st0)
    assert sf.min() == pytest.approx(235.0 / st0.mises.max(), rel=1e-6)

    # export dialog: everything through the shared helpers
    from fea_core.export_vtu import export_combination_vtu as ecv
    from fea_core.export_csv import export_combination_csv as ecc
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        study.combinations = [comb, env_c]
        for cm in study.combinations:
            r = compute_combination(study, res, cm)
            ecv(mesh, r, Path(t) / f"{cm.name}.vtu")
            ecc(mesh, r, Path(t) / f"{cm.name}.csv")
        study.combinations = []
        assert len(list(Path(t).iterdir())) == 4


@pytest.mark.slow
def test_probe_with_csys_and_criterion_through_mcp(model):
    srv, _ = model

    async def body(ok, err):
        p0 = await ok("fea_results_probe", point=[100, 10, 10], step="G")
        pr = p0["probes"][0]
        # named cylindrical system in the study (axis x): σz = sxx
        p1 = await ok("fea_results_probe", point=[100, 10, 10], step="G", csys="Valec")
        row = p1["probes"][0]
        assert row["csys"] == "Valec" and row["u_labels"] == ["ur", "uθ", "uz"]
        assert row["stress_MPa"]["σz"] == pytest.approx(pr["stress_MPa"]["sxx"], rel=1e-4)
        assert row["u_mm"][2] == pytest.approx(pr["u_mm"][0], abs=1e-6)
        assert row["u_mm"][0] == pytest.approx(pr["u_mm"][2], abs=1e-6)       # a point above the axis: radial = +z
        # definition of the system directly in the argument (does not change the study)
        p2 = await ok("fea_results_probe", point=[100, 10, 10], step="G",
                      csys={"kind": "cylindrical", "origin": [0, 10, 5], "z_axis": [1, 0, 0]})
        assert p2["probes"][0]["stress_MPa"]["σz"] == pytest.approx(pr["stress_MPa"]["sxx"], rel=1e-4)
        assert "Coordinate system" in await err("fea_results_probe", point=[100, 10, 10], csys="nic")
        # criterion: tresca >= mises (uniaxial state, agreement), safety to Re
        for crit in ("mises", "tresca", "mohr_coulomb"):
            q = (await ok("fea_results_probe", point=[100, 10, 10], step="G", criterion=crit))["probes"][0]
            assert q["criterion"] == crit and q["equivalent_stress_MPa"] > 0
            assert q["safety_factor"] == pytest.approx(q["limit_MPa"] / q["equivalent_stress_MPa"], rel=1e-4)
        q = (await ok("fea_results_probe", point=[100, 10, 10], step="G", criterion="mises"))["probes"][0]
        assert q["limit_MPa"] == 235 and q["equivalent_stress_MPa"] == pytest.approx(pr["mises_MPa"], rel=1e-4)
        q2 = (await ok("fea_results_probe", point=[100, 10, 10], step="G", criterion="mises", limit_MPa=100.0))["probes"][0]
        assert q2["limit_MPa"] == 100.0
        assert "criterion" in await err("fea_results_probe", point=[100, 10, 10], criterion="nesmysl")
    call(srv, body)


def test_imprint_tools_through_mcp(tmp_path):
    from mcp import Client
    from fea_core.mcp.server import build_server
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    step = tmp_path / "b.step"
    write_step([Part("Deska", box(0, 0, 0, 60, 40, 10))], step)
    srv = build_server(base_dir=tmp_path / "mcp")

    async def body(ok, err):
        await ok("fea_study_new", step_path=str(step), workdir=str(tmp_path / "wd"))
        a = await ok("fea_imprint_add", name="Raznik", part="Deska", center=[30, 20, 10], normal=[0, 0, 1], radius=5.0)
        assert a["added_imprint"]["shape"] == "circle" and a["added_imprint"]["radius"] == 5.0
        b = await ok("fea_imprint_add", name="Patka", part="Deska", shape="rectangle", center=[30, 20, 0],
                     normal=[0, 0, -1], width=12.0, height=8.0)
        assert b["added_imprint"]["dx"] == 12.0 and b["added_imprint"]["dy"] == 8.0
        assert "already exists" in await err("fea_imprint_add", name="Raznik", part="Deska")
        assert "does not exist" in await err("fea_imprint_add", name="X", part="Nic")
        assert "Unknown imprint shape" in await err("fea_imprint_add", name="X", part="Deska", shape="trojuhelnik")
        await err("fea_imprint_add", name="Y", part="Deska", angle_deg=30.0)         # rotation is not supported
        lst = await ok("fea_imprint_list")
        assert [i["name"] for i in lst["imprints"]] == ["Raznik", "Patka"]
        # saving into the study: Study.to_dict writes the same fields that the mesher reads
        d = srv.fea_session.study.to_dict()
        assert d["imprints"][1]["dx"] == 12.0 and "width" not in d["imprints"][1]
        dl = await ok("fea_imprint_delete", name="Raznik")
        assert dl["remaining"] == 1
        assert "does not exist" in await err("fea_imprint_delete", name="Raznik")

    async def main():
        async with Client(srv, mode="legacy") as c:
            async def raw(tool, **a):
                r = await c.call_tool(tool, a)
                return r.is_error, r.content[0].text

            async def ok(tool, **a):
                e, t = await raw(tool, **a)
                assert not e, f"{tool}: {t}"
                return json.loads(t)

            async def err(tool, **a):
                e, t = await raw(tool, **a)
                assert e, f"{tool} měl skončit chybou: {t}"
                return t
            await body(ok, err)
    anyio.run(main)


@pytest.mark.slow
def test_gui_result_field_branches_f9(model):
    """GUI branches of the results from F9 (local system, safety Tresca / Mohr-Coulomb / Stress Tool) called as
    MainWindow.result_field – with a fake self (without a window).
    """
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from fea_gui import main as gm
    srv, _ = model
    sess = srv.fea_session
    st0 = sess.results.steps[0]

    class _Combo:
        def __init__(self, name):
            self.name = name

        def currentIndex(self):
            return 1 if self.name else 0

        def currentData(self):
            return self.name

    class Stub:
        study = sess.study
        results = sess.results
        r_csys = _Combo("Valec")
        logs: list[str] = []

        def log_msg(self, t):
            self.logs.append(t)

        def node_re(self):
            return gm.MainWindow.node_re(self)

    names = [q[0] for q in gm.QUANTITIES]
    stub = Stub()
    # cylindrical system with axis x: "σz" (index 9) = global σxx, "Displacement Z" = uz of the local system = ux
    assert np.allclose(gm.MainWindow.result_field(stub, st0, names.index("Stress σz")), st0.S[:, 0], atol=1e-6)
    assert np.allclose(gm.MainWindow.result_field(stub, st0, names.index("Displacement Z")), st0.U[:, 0])
    stub.r_csys = _Combo(None)                                            # global system
    assert np.allclose(gm.MainWindow.result_field(stub, st0, names.index("Stress σz")), st0.S[:, 2])
    for q in ("Safety factor – Tresca", "Safety factor – Mohr-Coulomb", "Safety factor (Stress Tool)"):
        v = gm.MainWindow.result_field(stub, st0, names.index(q))
        assert v.shape == (len(st0.U),) and 0 < v.min() <= 10 and v.max() <= 10
    assert not stub.logs
