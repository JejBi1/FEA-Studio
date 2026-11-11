import math
from pathlib import Path
import numpy as np
import pytest

from fea_core.imprint import Imprint
from fea_core.study import Study, MeshSettings, Fixed, Pressure, LoadCase, Material
from fea_core.geometry import Part, write_step
from fea_core.mesh import tri_areas
from fea_core.selectors import find_faces
from fea_core.solve import run_study
from verification.cases import box, STEEL


def test_imprint_serialization():
    imp = Imprint(
        name="raznik",
        part="Deska",
        center=[50.0, 50.0, 10.0],
        normal=[0.0, 0.0, 1.0],
        shape="circle",
        radius=10.0,
        dx=20.0,
        dy=20.0,
        size=2.5,
    )
    d = imp.to_dict()
    assert d["name"] == "raznik"
    assert d["radius"] == 10.0
    assert d["shape"] == "circle"

    imp2 = Imprint.from_dict(d)
    assert imp2.name == imp.name
    assert imp2.center == imp.center
    assert imp2.radius == imp.radius


def test_imprint_circle_meshing_and_solve(tmp_path):
    # Plate 100x100x10, clamped at x=0
    step_file = tmp_path / "model.step"
    plate = Part("Deska", box(0, 0, 0, 100, 100, 10))
    write_step([plate], step_file)

    st = Study("ImprintTest", str(step_file), mesh=MeshSettings(size=5.0))
    st.add_material(STEEL, default=True)

    # Imprint of a punch Ø20 in the middle of the top face (z=10, x=50, y=50)
    st.imprints.append(Imprint(
        name="raznik_d20",
        part="Deska",
        center=[50.0, 50.0, 10.0],
        normal=[0.0, 0.0, 1.0],
        shape="circle",
        radius=10.0,
        size=2.5,
    ))

    st.add_selection("vetknuti", find_faces([plate], x=0))
    st.constraints.append(Fixed("vetknuti"))

    # Pressure 1 MPa on the imprint
    st.load_cases.append(LoadCase("Tlak_razniku", [
        Pressure("raznik_d20", 1.0),
    ]))

    res = run_study(st, workdir=tmp_path / "run")

    # 1. Check that the imprint is in the mesh
    assert "raznik_d20" in res.mesh.surfaces
    tris = res.mesh.surfaces["raznik_d20"]
    area = float(np.sum(tri_areas(res.mesh.nodes, tris)))
    theor_area = math.pi * (10.0 ** 2)  # 314.159 mm2
    assert area == pytest.approx(theor_area, rel=0.03)

    # 2. The total reaction at the clamping Fz corresponds to the pressure 1 MPa * circle area 314.159 N
    step = res.step(0)
    rz = step.reactions["vetknuti"][2]
    assert rz == pytest.approx(theor_area, rel=0.005)
