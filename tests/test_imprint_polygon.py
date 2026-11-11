"""Polygon imprint (A10): face frame, validation, JSON round trip and the meshed face (area = shoelace area)."""
import numpy as np
import pytest

from fea_core.imprint import Imprint, ImprintError, check_polygon, face_frame, polygon_area_centroid
from fea_core.study import Study

L_SHAPE = [[-15, -10], [15, -10], [15, 0], [0, 0], [0, 10], [-15, 10]]      # area 30·10 + 15·10 = 450 mm²


def test_area_centroid_and_validation():
    a, c = polygon_area_centroid(L_SHAPE)
    assert a == pytest.approx(450.0)
    assert c == pytest.approx([(300 * 0 + 150 * -7.5) / 450, (300 * -5 + 150 * 5) / 450])
    for bad in ([[0, 0], [1, 0]], [[0, 0], [1, 0], [2, 0]], [[0, 0], [1, 1], [1, 0], [0, 1]],
                [[0, 0], [0, 0], [1, 0], [0, 1]]):
        with pytest.raises(ImprintError):
            check_polygon(bad)


def test_frame_is_orthonormal_and_3d_round_trip():
    for n in ([0, 0, 1], [1, 0, 0], [0.3, -0.5, 0.8]):
        nn, xa, ya = face_frame(n)
        m = np.array([nn, xa, ya])
        assert m @ m.T == pytest.approx(np.eye(3))
    imp = Imprint("p", "Deska", [10, 20, 5], [0, 0, 1], shape="polygon", points=L_SHAPE)
    p3 = imp.points_3d()
    assert np.allclose(p3[:, 2], 5.0)
    back = Imprint.polygon_from_3d("p", "Deska", p3, normal=[0, 0, 1])
    assert back.area() == pytest.approx(450.0)
    assert back.center == pytest.approx(p3.mean(axis=0))
    tilted = p3 + np.array([0, 0, 1.0]) * np.arange(6)[:, None]
    with pytest.raises(ImprintError):
        Imprint.polygon_from_3d("p", "Deska", tilted)                    # not in one plane


def test_json_only_writes_points_for_polygon():
    st = Study("s", "m.step")
    st.imprints += [Imprint("c", "Deska", [0, 0, 0], [0, 0, 1]),
                    Imprint("p", "Deska", [0, 0, 0], [0, 0, 1], shape="polygon", points=L_SHAPE)]
    d = st.to_dict()["imprints"]
    assert "points" not in d[0] and len(d[1]["points"]) == 6
    st2 = Study.from_dict(st.to_dict())
    assert st2.imprints[1].shape == "polygon" and st2.imprints[1].area() == pytest.approx(450.0)
    assert "points3d" in st2.imprints[1].to_mesh_dict()


def test_polygon_imprint_meshes_with_exact_area(tmp_path):
    from fea_core.geometry import Part, write_step, read_step
    from fea_core.mesh import tri_areas
    from fea_core.solve import mesh_study
    from fea_core.study import MeshSettings
    from verification.cases import box, STEEL
    step = tmp_path / "m.step"
    write_step([Part("Deska", box(0, 0, 0, 100, 60, 10))], step)
    parts = read_step(step)
    st = Study("poly", str(step), mesh=MeshSettings(size=6.0))
    st.add_material(STEEL, default=True)
    st.imprints.append(Imprint("lpad", "Deska", [50, 30, 10], [0, 0, 1], shape="polygon", points=L_SHAPE, size=2.0))
    mesh, _ = mesh_study(st, parts, tmp_path / "mesh")
    area = float(np.sum(tri_areas(mesh.nodes, mesh.surfaces["lpad"])))
    assert area == pytest.approx(450.0, rel=1e-6)                     # straight edges → exact
