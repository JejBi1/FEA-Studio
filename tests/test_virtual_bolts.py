"""Virtual bolts from the fastening components of a CAD assembly (fea_core.joints.virtual): without mesh and solver.

A bolt built only from the record (name + position of the occurrence) must agree with the bolt recognised from the full geometry
(joints.detect) – a lap joint with a nut and washers, and a bolt into a thread. The whole calculation: verification case S4.
"""
from __future__ import annotations

import pytest

import verification.cases  # noqa: F401 – order of the imports of the verification modules
from fea_core.joints import tables
from fea_core.joints.naming import parse_name
from fea_core.joints.virtual import bolts_from_fasteners, classify, fastener_record

pytest.importorskip("OCP")


def _rec(name, origin_mm, R=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    m = [R[0][0], R[0][1], R[0][2], origin_mm[0] / 10, R[1][0], R[1][1], R[1][2], origin_mm[1] / 10,
         R[2][0], R[2][1], R[2][2], origin_mm[2] / 10, 0, 0, 0, 1]
    return fastener_record(parse_name(name), name + ":1", name + ".ipt", m)


def _same(a, b):
    for k in ("connection", "size", "length", "strength", "head_part", "nut_part"):
        assert getattr(a, k) == getattr(b, k), k
    for k in ("head_dw", "nut_s", "nut_dw", "thread_end"):
        assert getattr(a, k) == pytest.approx(getattr(b, k), abs=1e-6), k
    assert a.origin == pytest.approx(b.origin, abs=1e-6) and a.axis == pytest.approx(b.axis, abs=1e-9)
    assert [(p["part"], round(p["s0"], 6), round(p["s1"], 6), round(p["hole"], 6)) for p in a.plates] == \
           [(p["part"], round(p["s0"], 6), round(p["s1"], 6), round(p["hole"], 6)) for p in b.plates]
    assert [(w["side"], w["h"], round(w["d2"], 6)) for w in a.washers] == \
           [(w["side"], w["h"], round(w["d2"], 6)) for w in b.washers]


def test_classify():
    assert classify("ISO 4762 M6 x 20").kind == "bolt"
    assert classify("ISO 4032 M8").kind == "nut"
    assert classify("T8_NUT_HOUSING", "T8_NUT_HOUSING:1") is None       # without a size M → an ordinary part
    assert classify("side_panel") is None


def test_virtual_lap_joint_equals_detected():
    from fea_core.joints.detect import detect
    from verification import joint_models as jm
    full = jm.lap_joint(d=12.0)
    ref = detect(full).bolts[0]
    wh = tables.washer_dims(12.0)["h"]
    recs = [_rec(full[2].name, (80, 20, 10 + wh)), _rec(full[3].name, (80, 20, 10)), _rec(full[4].name, (80, 20, -10)),
            _rec(full[5].name, (80, 20, -10 - wh))]
    virt, warn = bolts_from_fasteners(full[:2], recs)
    assert warn == [] and len(virt) == 1
    _same(virt[0], ref)


def test_virtual_tapped_equals_detected():
    from fea_core.joints.detect import detect
    from verification import joint_models as jm
    full = jm.tapped_joint(d=10.0, t=10.0)
    ref = detect(full).bolts[0]
    virt, warn = bolts_from_fasteners(full[:2], [_rec(full[2].name, (30, 20, 10.0), ((1, 0, 0), (0, -1, 0), (0, 0, -1)))])
    assert warn == [] and len(virt) == 1
    _same(virt[0], ref)


def test_virtual_bolt_without_holes_is_reported():
    from verification import joint_models as jm
    full = jm.lap_joint(d=12.0)
    virt, warn = bolts_from_fasteners(full[:2], [_rec(full[2].name, (20, 20, 12.5))])     # outside the holes
    assert virt == [] and "coaxial holes" in warn[0]


def test_apply_fasteners_from_metadata():
    from fea_core import cad_import
    from fea_core.study import Study
    from verification import joint_models as jm
    full = jm.lap_joint(d=12.0)
    wh = tables.washer_dims(12.0)["h"]
    meta = {"parts": [], "fasteners": [_rec(full[2].name, (80, 20, 10 + wh)), _rec(full[5].name, (80, 20, -10 - wh))]}
    st = Study(name="x", geometry="x.step")
    added, warn = cad_import.apply_fasteners(st, full[:2], meta)
    assert added == [full[2].name + ":1"] and [b.name for b in st.bolts] == added
    assert st.bolts[0].connection == "nut" and st.bolts[0].cad_parts == []
