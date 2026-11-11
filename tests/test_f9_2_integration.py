"""Integration tests for stage F9.2:
- MCP tools for the export (VTU/CSV), the imprints (imprint) and the combinations (ČSN EN 1990)
- GUI dialogs for the export, the imprints and the combinations
- Word protocol with the table of the combinations
"""
import numpy as np
import pytest

from fea_core.study import Study, LoadCase, Material
from fea_core.combinations import Combination, generate_en1990_combinations
from fea_core.imprint import Imprint
from fea_core.mesh import Mesh
from fea_core.results import Results, StepResult
from fea_core.mcp.session import Session


def test_mcp_combinations_and_imprints(tmp_path):
    sess = Session(tmp_path)
    sess.study = Study(name="TestStudy", geometry="model.step")
    sess.study.load_cases = [
        LoadCase(name="Stala"),
        LoadCase(name="Uzitna"),
        LoadCase(name="Snih"),
    ]

    # Test combination generation via MCP function logic
    cases = [
        {"name": "Stala", "action_type": "permanent", "subtype": "self_weight"},
        {"name": "Uzitna", "action_type": "variable", "category": "A", "leading": True},
        {"name": "Snih", "action_type": "variable", "subtype": "snow_below_1000m"},
    ]
    combs = generate_en1990_combinations(cases, formula="6.10")
    assert len(combs) >= 2
    sess.study.combinations.extend(combs)

    # Test Imprint add
    imp = Imprint(name="Raznik", part="D1", shape="circle", center=[0, 0, 0], normal=[0, 0, 1], radius=10.0)
    sess.study.imprints.append(imp)
    assert len(sess.study.imprints) == 1
    assert sess.study.imprints[0].to_dict()["name"] == "Raznik"

    # Test study serialization with combinations and imprints
    d = sess.study.to_dict()
    assert "combinations" in d
    assert "imprints" in d
    assert len(d["combinations"]) == len(combs)
    assert len(d["imprints"]) == 1

    st2 = Study.from_dict(d)
    assert len(st2.combinations) == len(combs)
    assert len(st2.imprints) == 1
    assert st2.imprints[0].name == "Raznik"
    assert st2.combinations[0].name == combs[0].name


def test_report_with_combinations(tmp_path):
    docx = pytest.importorskip("docx")
    from fea_core.report import ReportOptions, build_report

    sess = Session(tmp_path)
    sess.study = Study(name="ReportStudy", geometry="model.step")
    mat = Material(name="Ocel", E=210000.0, nu=0.3, rho=7.85e-9, Re=235.0)
    sess.study.materials["Ocel"] = mat
    sess.study.part_materials["D1"] = "Ocel"
    sess.study.load_cases = [LoadCase(name="G"), LoadCase(name="Q")]
    sess.study.combinations = [
        Combination(name="MSD_1", factors={"G": 1.35, "Q": 1.5}, description="ČSN EN 1990 6.10"),
        Combination(name="Obalka", envelope=True, envelope_type="max", envelope_combinations=["MSD_1"]),
    ]

    # Dummy mesh & results
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [5.0, 0.0, 0.0],
        [5.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
        [5.0, 0.0, 5.0],
        [0.0, 5.0, 5.0],
    ])
    tets = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int32)
    mesh = Mesh(nodes=nodes, tets=tets, tet_part=np.zeros(1, dtype=np.int32), part_names=["D1"])
    step = StepResult(
        name="G",
        U=np.zeros((10, 3)),
        S=np.zeros((10, 6)),
        reactions={"Fix": np.zeros(3)},
    )
    res = Results(mesh=mesh, steps=[step])
    sess.results = res

    doc_path = tmp_path / "test_report.docx"
    opts = ReportOptions(title="Testovací protokol", images=False, language="cs")
    rep_res = build_report(sess, doc_path, opts)

    assert doc_path.exists()
    doc = docx.Document(str(doc_path))
    full_text = "\n".join(p.text for p in doc.paragraphs)
    # Check that combinations heading is present
    assert "Kombinace zatěžovacích stavů" in full_text          # language="cs"
