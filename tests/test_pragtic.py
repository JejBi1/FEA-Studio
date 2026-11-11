"""PragTic database: loading, search, conversion of a record to S-N."""
import math

import pytest

from fea_core.fatigue import pragtic
from fea_core.fatigue.sn import SNCurve

pytestmark = pytest.mark.skipif(not pragtic.DATA.exists(), reason="chybí data/pragtic_vmat.json")


def _rec(data, rid=1, mark="X"):
    return {"id": rid, "mark": mark, "references": ["ID 1 : zdroj"],
            "data": {mode: {k: {"value": v, "unit": "", "desc": d} for k, (v, d) in rows.items()}
                     for mode, rows in data.items()}}


def test_database_loads():
    recs = pragtic.records()
    assert len(recs) > 500
    assert len({r["id"] for r in recs}) == len(recs)
    with_sn = [r for r in recs if pragtic.sn_options(r)]
    assert len(with_sn) > 300


def test_1045_normalized_basquin():
    r = pragtic.by_id(83)                     # 1045 normalized, SAE J1099: σf' = 1439, b = −0.127
    assert pragtic.value(r, pragtic.STATIC, "Ultimate tensile strength") == 718
    lbl, sn = pragtic.sn_options(r)[0]
    assert "Basquin" in lbl
    assert sn.ND == 1e6 and sn.k == pytest.approx(1 / 0.127)
    assert sn.SD == pytest.approx(1439 * (2e6) ** -0.127)
    assert sn.k2 == pytest.approx(2 * sn.k - 1)
    assert sn.N(sn.SD * 1.5) == pytest.approx(0.5 * (1.5 * sn.SD / 1439) ** (1 / -0.127), rel=1e-9)
    assert "#83" in sn.source and "2002-000067" in sn.source


def test_limit_with_basquin_meets_curve():
    r = _rec({pragtic.PUSH_PULL: {"f1": (300.0, "Fatigue limit"), "sf": (1200.0, "Fatigue strength coefficient"),
                                  "b": (-0.1, "Fatigue strength exponent")}})
    lbl, sn = pragtic.sn_options(r)[0]
    assert "tension-compression" in lbl and sn.k2 == 0 and sn.SD == 300
    assert sn.k == pytest.approx(10.0)
    assert 1200 * (2 * sn.ND) ** -0.1 == pytest.approx(300)        # the knee lies on the Basquin line
    assert len(pragtic.sn_options(r)) == 2


def test_limit_slope_from_rm():
    r = _rec({pragtic.STATIC: {"Rm": (600.0, "Ultimate tensile strength")},
              "Rotating bending": {"rb1": (270.0, "Fatigue limit"), "N": (1e7, "Number of cycles at fatigue limit")}})
    (lbl, sn), = pragtic.sn_options(r)
    assert "ohyb za rotace" in lbl
    assert sn.N(0.9 * 600) == pytest.approx(1e3)
    assert math.isinf(sn.N(269.0))


def test_find_by_any_standard():
    """ČSN, EN and AISI find the same records; S355 is not in the database as S355, but finds 11523 / St52-3 / 18G2A."""
    ids = [{r["id"] for r in pragtic.find(q)[1]} for q in ("12050", "C45", "AISI 1045", "1.0503")]
    assert ids[0] == ids[1] == ids[2] == ids[3] and len(ids[0]) >= 30
    hits, exact, similar = pragtic.find("S355")
    assert {r["mark"] for r in exact} >= {"11523", "St52-3", "18G2A"}
    assert similar and all(d <= 6 for _, d, _ in similar)
    q = pragtic.find("1045 quenched")[1]                  # XC48 (≈ C45) hardened too
    assert "1045" in {r["mark"] for r in q} and all("quench" in r["treatment"] for r in q)
    assert all(pragtic.grade_of_record(r)[0].key == "C45" for r in q)
    # stainless-like ones are only the stainless ones
    for r, _, _ in pragtic.find("1.4404")[2]:
        p = pragtic.record_profile(r)
        assert p.cls == "nerez"


def test_search_and_html():
    hits = pragtic.search("1045")
    assert hits and all("1045" in r["mark"] or "1045" in r.get("alt_mark", "") or "1045" in r.get("standard", "")
                        or "1045" in r.get("comment", "") for r in hits)
    html = pragtic.details_html(pragtic.by_id(83))
    assert "Mez pevnosti v tahu Rm" in html and "Basquin" in html and "SAE J1099" in html
    assert isinstance(pragtic.sn_options(pragtic.by_id(83))[0][1], SNCurve)
