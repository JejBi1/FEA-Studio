import xml.etree.ElementTree as ET
import csv
import numpy as np
import pytest

from fea_core.mesh import Mesh
from fea_core.results import Results, StepResult
from fea_core.study import Study
from fea_core.combinations import Combination
from fea_core.export_vtu import export_step_vtu, export_results_vtu
from fea_core.export_csv import export_nodal_csv, export_reactions_csv


@pytest.fixture
def sample_fe_model(tmp_path):
    # 1 tet10 element (10 nodes)
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ], dtype=np.float64)

    tets = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int32)
    tet_part = np.array([0], dtype=np.int32)

    mesh = Mesh(
        nodes=nodes,
        tets=tets,
        tet_part=tet_part,
        part_names=["Part1"],
    )

    u1 = np.ones((10, 3)) * 0.1
    s1 = np.zeros((10, 6))
    s1[:, 0] = 50.0  # SXX = 50 MPa
    step1 = StepResult(
        name="Step1",
        U=u1,
        S=s1,
        reactions={"support": np.array([0.0, 0.0, 100.0])},
        reaction_moments={"support": np.array([10.0, 20.0, 0.0])},
    )

    u2 = np.ones((10, 3)) * 0.2
    s2 = np.zeros((10, 6))
    s2[:, 1] = 80.0  # SYY = 80 MPa
    step2 = StepResult(
        name="Step2",
        U=u2,
        S=s2,
        reactions={"support": np.array([0.0, 0.0, 200.0])},
        reaction_moments={"support": np.array([20.0, 40.0, 0.0])},
    )

    results = Results(mesh=mesh, steps=[step1, step2])
    study = Study(name="TestStudy", geometry="")
    study.combinations.append(Combination(name="Comb1", factors={"Step1": 1.0, "Step2": 1.0}))

    return mesh, results, study


def test_export_step_vtu(tmp_path, sample_fe_model):
    mesh, results, study = sample_fe_model
    out_file = tmp_path / "test_step.vtu"

    export_step_vtu(mesh, results.step(0), out_file)
    assert out_file.exists()

    # We check that the output is valid XML
    tree = ET.parse(out_file)
    root = tree.getroot()
    assert root.tag == "VTKFile"
    assert root.attrib.get("type") == "UnstructuredGrid"

    piece = root.find(".//Piece")
    assert piece is not None
    assert int(piece.attrib["NumberOfPoints"]) == 10
    assert int(piece.attrib["NumberOfCells"]) == 1

    # Check of the presence of the fields
    arrays = {da.attrib["Name"]: da for da in root.findall(".//DataArray")}
    assert "Points" in arrays
    assert "connectivity" in arrays
    assert "offsets" in arrays
    assert "types" in arrays
    assert "Displacement" in arrays
    assert "von_Mises_Stress" in arrays
    assert "PartId" in arrays


def test_export_results_vtu_with_pvd(tmp_path, sample_fe_model):
    mesh, results, study = sample_fe_model
    out_dir = tmp_path / "vtu_export"

    files = export_results_vtu(mesh, results, out_dir, base_name="cantilever", study=study)
    assert len(files) >= 4  # step 0, step 1, comb 1, pvd
    # state numbers from 1 as step_index / *STEP, combinations K01…
    assert [f.name for f in files] == ["cantilever_01_Step1.vtu", "cantilever_02_Step2.vtu",
                                       "cantilever_K01_Comb1.vtu", "cantilever.pvd"]

    pvd_file = out_dir / "cantilever.pvd"
    assert pvd_file.exists()

    tree = ET.parse(pvd_file)
    root = tree.getroot()
    assert root.tag == "VTKFile"
    assert root.attrib.get("type") == "Collection"
    datasets = root.findall(".//DataSet")
    assert len(datasets) == 3
    assert [d.attrib["timestep"] for d in datasets] == ["1", "2", "3"]


def test_export_csv(tmp_path, sample_fe_model):
    mesh, results, study = sample_fe_model

    # 1. Nodal CSV
    nodal_csv = tmp_path / "nodal.csv"
    export_nodal_csv(mesh, results.step(0), nodal_csv)
    assert nodal_csv.exists()

    with open(nodal_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 11  # header + 10 nodes
        header = reader[0]
        assert header[:4] == ["node_id", "part_id", "part", "n_parts"]
        assert "von_mises_MPa" in header
        assert "ux_mm" in header
        # Values of the first node
        row1 = reader[1]
        assert row1[0] == "1"  # node 1
        assert row1[1:4] == ["1", "Part1", "1"]
        assert float(row1[header.index("von_mises_MPa")]) == pytest.approx(50.0)  # von Mises = 50 MPa

    # 2. Reactions CSV
    reac_csv = tmp_path / "reactions.csv"
    export_reactions_csv(results, reac_csv)
    assert reac_csv.exists()

    with open(reac_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 5  # header + (step1 + TOTAL) + (step2 + TOTAL)
        header = reader[0]
        assert header[:4] == ["step_index", "step_name", "selection", "constraint_kind"]
        assert reader[1][:3] == ["1", "Step1", "support"]
        assert float(reader[1][header.index("Fz_N")]) == pytest.approx(100.0)  # Fz = 100 N
        assert reader[2][:3] == ["1", "Step1", "TOTAL"]
        assert reader[3][:3] == ["2", "Step2", "support"]
