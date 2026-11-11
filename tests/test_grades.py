"""Converter of material designations and similar materials."""
import math

import pytest

from fea_core import grades as G
from fea_core.materials import LIBRARY


@pytest.mark.parametrize("text, key", [
    ("12050", "C45"), ("ČSN 12 050", "C45"), ("12050.6", "C45"), ("1.0503", "C45"), ("Ck45", "C45"),
    ("AISI 1045", "C45"), ("SAE1045", "C45"), ("S45C", "C45"), ("SM45C (SAE 1045)", "C45"), ("45", "C45"),
    ("XC45", "C45"), ("St 52-3", "S355"), ("S355J2", "S355"), ("11523.1", "S355"), ("ČSN 11 375", "S235"),
    ("15142", "42CrMo4"), ("SCM440", "42CrMo4"), ("4140", "42CrMo4"), ("1.7225", "42CrMo4"),
    ("SAE 52100", "100Cr6"), ("En 31", "100Cr6"), ("SUJ2", "100Cr6"),
    ("X5CrNi18-10", "1.4301"), ("AISI 304", "1.4301"), ("SUS304", "1.4301"), ("17240", "1.4301"),
    ("316L", "1.4404"), ("347stainless-steel", "1.4550"), ("17-4 PH (H900)", "1.4542"),
    ("EN AW-2024", "EN AW-2024"), ("AlCuMg2", "EN AW-2024"), ("24S-T3", "EN AW-2024"), ("ČSN 42 4203", "EN AW-2024"),
    ("6061Al", "EN AW-6061"), ("75S-T6", "EN AW-7075"), ("AlZnMgCu1,5", "EN AW-7075"),
    ("GGG-40", "EN-GJS-400-15"), ("FGS 800-2", "EN-GJS-800-2"), ("GTS-45", "EN-GJMB-450-6"),
    ("6Al-4V", "Ti-6Al-4V"), ("Inconel X", "Inconel X-750"), ("brass 7:3", "CuZn30"),
    ("2.4856", "Inconel 625"), ("Alloy 625", "Inconel 625"), ("IN718", "Inconel 718"), ("AISI H13", "1.2344"),
    ("X40CrMoV5-1", "1.2344"), ("Co28Cr6Mo", "CoCrMo"), ("ASTM F75", "CoCrMo"), ("CW106C", "CuCrZr"),
    ("CuCr1Zr", "CuCrZr"), ("Scalmalloy", "Scalmalloy"), ("Ti-6Al-4V L-PBF HIP", "Ti-6Al-4V"),
])
def test_lookup(text, key):
    assert G.grade_of(text).key == key


def test_no_false_hits():
    for text in ("GTS-45 cast iron", "PLA tisk XY", "St 100", "Ti 24V", "0.4%C steel"):
        hits = G.lookup(text)
        assert all(g.key != "C45" for g, _ in hits), text
    assert G.lookup("PLA tisk XY") == []
    assert G.grade_of("045") is None                         # GB "45" only as the whole text


def test_aluminium_number_is_not_steel():
    """SOLIDWORKS "1060 Alloy", "1060-H12" = aluminium 1060 (series 1xxx), not steel AISI 1060 (C60). The aluminium group
    1060 is not in the converter → nothing; a steel with the word for steel or without any further data remains steel.
    """
    from fea_core.cad_import import library_material
    for text in ("1060 Alloy", "1060-H12", "1060-H12 Rod (SS)", "1060-O", "Aluminium 1060", "Hliník 1060",
                 "EN AW-1060", "AA 1060", "1070 Alloy"):
        assert G.lookup(text) == [], text
        assert G.aluminium_text(text), text
    _, info = library_material("1060 Alloy")
    assert "grade" not in info and "similar_in_library" not in info          # not a similar steel either
    assert G.lookup("1050 Alloy") == [(G.by_key("EN AW-1050A"), True)]
    assert G.grade_of("1100-H12 Rod (SS)").key == "EN AW-1100"
    assert G.grade_of("6061 Alloy").key == "EN AW-6061"
    assert G.grade_of("6061-T6 (SS)").key == "EN AW-6061"
    for text, key in (("AISI 1060", "C60"), ("1060", "C60"), ("SAE 1060 steel", "C60"), ("C60", "C60"),
                      ("4140 alloy steel", "42CrMo4"), ("AISI 1045 Steel, cold drawn", "C45"),
                      ("9254 AL FG", "54SiCr6")):           # steel calmed with aluminium (PragTic), 9xxx is not aluminium
        assert G.grade_of(text).key == key, text


def test_approximate_flag_and_order():
    hits = G.lookup("34CrNiMo6")
    assert hits[0] == (G.by_key("34CrNiMo6"), False)
    assert (G.by_key("AISI 4340"), True) in hits
    assert G.lookup("A36") == [(G.by_key("S235"), True)]


def test_unique_designations():
    """A full (not approximate) designation may belong only to one group."""
    tok, _ = G._index()
    for k, lst in tok.items():
        full = [g.key for g, a in lst if not a]
        assert len(full) <= 1, (k, full)
    assert len({g.key for g in G.GRADES}) == len(G.GRADES)
    for g in G.GRADES:
        assert g.family in G.FAMILIES
        assert all(s in G.SYSTEM_LABELS for s in g.names)


def test_library_materials_known():
    known = {m.name: G.grade_of(m.name) for m in LIBRARY}
    assert known["Ocel C45 +N"].key == "C45"
    assert known["Ocel 42CrMo4 +QT"].key == "42CrMo4"
    assert known["Hliník EN AW-6082 T6"].key == "EN AW-6082"
    assert known["Nerez 1.4404"].key == "1.4404"
    assert known["PLA tisk XY (orientačně)"] is None


def test_similar():
    sim = [g.key for g, _ in G.similar(G.by_key("C45"))]
    assert sim[:2] == ["C50", "C40"]
    assert "S275" == G.similar(G.by_key("S355"))[0][0].key
    assert all(g.base == "al" for g, _ in G.similar(G.by_key("EN AW-7075")))
    assert G.similar(G.by_key("2A12"))[0][0].key in ("EN AW-2024", "EN AW-2124")
    # stainless × carbon steel nor steel × aluminium is offered
    assert math.isinf(G.distance(G.profile(G.by_key("1.4404")), G.profile(G.by_key("C45"))))
    assert math.isinf(G.distance(G.profile(G.by_key("EN AW-6082")), G.profile(G.by_key("S235"))))


def test_incomplete_chemistry_ignored():
    """A record without Cr (incomplete composition) is not "dissimilar" to stainless steel because of that."""
    q = G.profile(G.by_key("1.4301"))
    rec = G.Profile("nerez-aust", "ocel", {"C": 0.05, "Mn": 1.2, "Si": 0.4}, 600, complete=False)
    assert G.distance(q, rec) < 1.5
