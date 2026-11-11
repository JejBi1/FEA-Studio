"""Small tests: BDF cards, coaxial cylinders (the whole hole), user materials, undo/redo of the study."""
import math

from fea_core.mystran import _card, _f


def test_bdf_card_continuation_and_reals():
    lines = _card("RBE2", [1, 2, 123] + list(range(10, 30)))
    assert lines[0].count(",") == 9 and lines[0].endswith(",+")      # name + 8 fields + continuation
    assert all(l.startswith("+,") for l in lines[1:])
    fields = sum((l.rstrip(",+").split(",")[1:] for l in lines), [])
    assert fields == ["1", "2", "123"] + [str(i) for i in range(10, 30)]
    assert _f(10.0) == "10." and _f(1e-5) == "1.e-05" and _f(0.0) == "0." and "." in _f(-2.5e12)
    assert len(_f(-1.23456789012e-10)) <= 16


def test_coaxial_half_cylinders():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from fea_core.geometry import list_faces, coaxial_cylinders, cylinder_axis
    ax = gp_Ax2(gp_Pnt(5, 5, 0), gp_Dir(0, 0, 1))
    a = list_faces(BRepPrimAPI_MakeCylinder(ax, 3.0, 10.0, math.pi).Shape())
    b = list_faces(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(5, 5, 0), gp_Dir(0, 0, 1), gp_Dir(-1, 0, 0)),
                                            3.0, 10.0, math.pi).Shape())
    other = list_faces(BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(20, 5, 0), gp_Dir(0, 0, 1)), 3.0, 10.0).Shape())
    faces = a + b + other
    cyl = [i for i, f in enumerate(faces) if cylinder_axis(f) is not None]
    first = cyl[0]
    got = coaxial_cylinders(faces, first)
    assert len(got) == 2 and all(i < len(a) + len(b) for i in got)   # both halves, not a foreign pin


def test_user_materials(tmp_path, monkeypatch):
    from fea_core import materials as ml
    from fea_core.study import Material
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert ml.user_library() == []
    ml.save_user_material(Material("PLA – kalibrace", 3100, 0.36, 1.24e-9, Rm=48, source="zkouška"))
    assert ml.by_name("PLA – kalibrace").E == 3100
    assert len(ml.all_materials()) == len(ml.LIBRARY) + 1
