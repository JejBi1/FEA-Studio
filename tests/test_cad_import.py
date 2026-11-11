"""Metadata from the CAD next to the STEP (fea_core.cad_import): matching of names (also the SolidWorks occurrence "-1"), materials from the library,
reading of the .json file next to the STEP, the names of the SolidWorks occurrences "NAUO1" in the STEP and the names of bolts with the suffix "-1".
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fea_core.cad_import import (apply_metadata, library_material, load_sidecar, match_parts, name_keys,
                                 parse_metadata, sidecar_path, MetadataError)

ROOT = Path(__file__).resolve().parent.parent


# ===================================================================== matching of names

def test_name_keys_solidworks_suffix():
    assert name_keys("Deska-1") == ["deska-1", "deska"]
    assert name_keys("Deska-12") == ["deska-12", "deska"]
    assert name_keys("Deska:1") == ["deska:1", "deska"]                     # Fusion / Inventor
    assert name_keys("Deska (2)") == ["deska (2)", "deska"]                 # read_step: second part with the same name
    assert name_keys("Deska-1 (2)") == ["deska-1 (2)", "deska-1", "deska"]
    assert name_keys("Konzola-1_2") == ["konzola-1_2", "konzola-1", "konzola"]   # several bodies of an occurrence
    # "-number" only once and only the last one
    assert name_keys("Profil 40-40-4-1") == ["profil 40-40-4-1", "profil 40-40-4"]
    # standard names without an occurrence suffix are not shortened (hyphen with a space, decimal point)
    assert name_keys("ISO 4017 - M8 x 30 - 8.8") == ["iso 4017 - m8 x 30 - 8.8"]
    assert name_keys("ISO 7089 - 8") == ["iso 7089 - 8"]
    assert name_keys("ISO 4017 - M8 x 30 - 8.8-1") == ["iso 4017 - m8 x 30 - 8.8-1", "iso 4017 - m8 x 30 - 8.8"]
    assert name_keys("Díl-A") == ["díl-a"]


def test_match_parts_exact_before_dash():
    # an exact match has priority: "M8-30" is not shortened to "M8" when it has its own record
    m, un = match_parts(["M8-30", "M8"], [{"name": "M8"}, {"name": "M8-30"}])
    assert m == {"M8-30": 1, "M8": 0} and un == []
    # the suffixes of read_step / Fusion have priority over the "-1" of SolidWorks
    m, _ = match_parts(["Deska_1"], [{"name": "Deska-1"}, {"name": "Deska"}])
    assert m == {"Deska_1": 1}
    # STEP with occurrences "Deska-1" (SolidWorks), metadata with the name of the part file
    m, un = match_parts(["Deska-1", "Deska-2", "Patka-1"], [{"name": "Deska"}, {"name": "Patka"}])
    assert m == {"Deska-1": 0, "Deska-2": 0, "Patka-1": 1} and un == []


def test_match_parts_solidworks_occurrences():
    """A STEP from SolidWorks carries the name of the part (an occurrence without a name "NAUO1"); the macro writes the occurrences "Deska-1", "Deska-2"
    with the name of the part in "names" → every part of the STEP gets its own record (the first free one), nothing is left over.
    """
    meta = [{"name": "Deska-1", "names": ["Deska"]}, {"name": "Deska-2", "names": ["Deska"]},
            {"name": "Sroub-1", "names": ["Sroub"]}, {"name": "Vysunout1", "names": "Konzola"}]
    m, un = match_parts(["Deska", "Deska (2)", "Sroub", "Konzola_1"], meta)
    assert m == {"Deska": 0, "Deska (2)": 1, "Sroub": 2, "Konzola_1": 3} and un == []


# ===================================================================== materials

def test_library_material():
    assert library_material("S235JR")[0] == "Ocel S235"
    assert library_material("AISI 304")[0] == "Nerez 1.4301"
    assert library_material("ocel s355")[0] == "Ocel S355"                   # exact name regardless of the case
    lib, info = library_material("Plain Carbon Steel")
    assert lib is None and info["cad_material"] == "Plain Carbon Steel" and "fea_materials_assign" not in str(info)
    assert library_material("")[0] is None


def test_library_material_state():
    """If both names give the material state (T6 / O, +N / +QT), they must agree; SolidWorks "6061 Alloy" is annealed
    (O, Re 55 MPa), not T6. Without a state in the CAD it is assigned as before (Fusion "Aluminum 6061" = T6).
    """
    from fea_core.cad_import import material_states
    assert material_states("Hliník EN AW-6061 T6") == {"T6"} and material_states("EN AW-6082-T651") == {"T6"}
    assert material_states("6061-O") == {"O"} and material_states("Ocel 42CrMo4 +QT") == {"+QT"}
    assert material_states("6061 Alloy") == {"O"} and material_states("Aluminum 6061") == set()
    assert material_states("Ocel S235") == set() and material_states("Titan Grade 5 (Ti-6Al-4V)") == set()
    assert library_material("6061-T6 (SS)")[0] == "Hliník EN AW-6061 T6"
    assert library_material("EN AW-6082-T651")[0] == "Hliník EN AW-6082 T6"
    assert library_material("42CrMo4+QT")[0] == "Ocel 42CrMo4 +QT"
    assert library_material("42CrMo4")[0] == "Ocel 42CrMo4 +QT"              # state not given → unchanged
    assert library_material("Aluminum 6061")[0] == "Hliník EN AW-6061 T6"
    for name in ("6061 Alloy", "6061-O", "6061-T4 (SS)", "7075 Alloy", "C45+QT"):
        lib, info = library_material(name)
        assert lib is None and "material condition" in info["hint"], (name, info)
    # another state confirms / refutes the properties from the CAD (Fusion: material_props)
    assert library_material("6061-T651 Plate", {"Re_MPa": 255})[0] == "Hliník EN AW-6061 T6"
    assert library_material("6061 Alloy", {"Re_MPa": 55.1, "Rm_MPa": 124})[0] is None
    assert library_material("6061 Alloy", {"Re_MPa": 250})[0] == "Hliník EN AW-6061 T6"
    lib, info = library_material("S235JR", {"Re_MPa": 620})
    assert lib is None and "do not match" in info["hint"]
    assert library_material("S235JR", {"Re_MPa": 235, "E_MPa": 210000})[0] == "Ocel S235"
    # O and H111 = one state (EN 485-2 "O/H111"); other combinations ("T4/T6") are not assigned
    for name in ("EN AW-5083 O/H111", "5083-O/H111", "AlMg4,5Mn0,7 O/H111", "EN AW-5083 O", "EN AW-5083 H111"):
        assert library_material(name)[0] == "Hliník EN AW-5083 H111", name
    assert library_material("EN AW-6082 T4/T6")[0] is None
    assert library_material("6061 Alloy")[0] is None                 # O ≠ T6


def _study():
    from fea_core import materials as matlib
    from fea_core.study import Study
    st = Study(name="t", geometry="x.step")
    st.add_material(matlib.default(), default=True)          # as a new study in the GUI (Ocel S235)
    return st


def test_apply_metadata():
    st = _study()
    meta = {"source": "SolidWorks", "document": "Držák",
            "parts": [{"name": "Deska-1", "names": ["Deska"], "material": "AISI 304", "color": [0.5, 0.5, 0.5]},
                      {"name": "Patka-1", "names": ["Patka"], "material": "Plain Carbon Steel",
                       "material_props": {"E_MPa": 210000}},
                      {"name": "Skryty-1", "material": "S355"}]}
    res = apply_metadata(st, ["Deska", "Patka", "Podlozka"], meta)
    assert st.part_materials == {"Deska": "Nerez 1.4301"} and "Nerez 1.4301" in st.materials
    assert st.default_material == "Ocel S235"                        # the default material does not change
    assert st.material_of("Patka").name == "Ocel S235"
    assert res.assigned == {"Deska": "Nerez 1.4301"} and res.matched == 2
    assert res.not_in_library == {"Plain Carbon Steel": ["Patka"]}
    assert res.no_metadata == ["Podlozka"] and res.unmatched == ["Skryty-1"]
    rows = {r["name"]: r for r in res.rows}
    assert rows["Deska"]["metadata"] == "Deska-1" and rows["Deska"]["color"] == [0.5, 0.5, 0.5]
    assert rows["Patka"]["cad_material_props"]["E_MPa"] == 210000 and "material" not in rows["Patka"]
    assert "2 of 3 parts matched" in res.summary() and "SolidWorks" in res.summary()
    msgs = res.messages(st.default_material)
    assert msgs[0][0] == "info"
    warn = [t for lv, t in msgs if lv == "warning"]
    assert any("Plain Carbon Steel" in t and "Ocel S235" in t and "Patka" in t for t in warn)
    # the help is part of one sentence (one translation template with 4 variables)
    res3 = apply_metadata(_study(), ["Kryt"], {"parts": [{"name": "Kryt", "material": "6061 Alloy"}]})
    (_, t3), = [m for m in res3.messages("Ocel S235") if m[0] == "warning"]
    assert t3.startswith("CAD material “6061 Alloy” is not in the library – parts Kryt have the default material Ocel S235, ") \
        and "assign the correct one in the Materials step; same grade as" in t3
    assert any("Podlozka" in t for t in warn)
    assert any(lv == "info" and "Skryty-1" in t for lv, t in msgs)
    # only the finding of the match, without an assignment
    st2 = _study()
    res2 = apply_metadata(st2, ["Deska"], meta, assign=False)
    assert st2.part_materials == {} and res2.assigned == {} and res2.rows[0]["material_match"]["match"]


# ===================================================================== file next to the STEP

def test_sidecar(tmp_path):
    step = tmp_path / "Držák_20261015_101500.step"
    step.write_text("ISO-10303-21;")
    assert sidecar_path(step) == tmp_path / "Držák_20261015_101500.json"
    warn = []
    assert load_sidecar(step, warn) is None and warn == []                  # no file
    meta = {"source": "Inventor", "document": "Držák", "units": "mm",
            "parts": [{"name": "Deska:1", "names": ["Deska"], "material": "Ocel, uhlíková"}]}
    # UTF-8 s BOM (ADODB.Stream z VBA / iLogic)
    sidecar_path(step).write_bytes(b"\xef\xbb\xbf" + json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    assert load_sidecar(step, warn) == meta and warn == []
    # UTF-16 with BOM – it is read anyway
    sidecar_path(step).write_bytes(json.dumps(meta, ensure_ascii=False).encode("utf-16"))
    assert load_sidecar(step, warn)["parts"][0]["material"] == "Ocel, uhlíková"
    # a JSON of the study with the same name (CLI: run konzola.json) and a foreign JSON without "parts" are not metadata – without a warning
    sidecar_path(step).write_text(json.dumps({"format": "fea-studio-study", "geometry": "x.step", "parts": []}),
                                  encoding="utf-8")
    assert load_sidecar(step, warn) is None and warn == []
    sidecar_path(step).write_text(json.dumps({"nastaveni": 1}), encoding="utf-8")
    assert load_sidecar(step, warn) is None and warn == []
    # invalid JSON and a faulty structure → None with a warning
    sidecar_path(step).write_text("{nic", encoding="utf-8")
    assert load_sidecar(step, warn) is None and len(warn) == 1 and "ignored" in warn[0]
    sidecar_path(step).write_text(json.dumps({"parts": "Deska"}), encoding="utf-8")
    assert load_sidecar(step, warn) is None and len(warn) == 2 and "must be a list" in warn[1]


def test_parse_metadata_errors():
    assert parse_metadata(None) is None and parse_metadata("  ") is None
    with pytest.raises(MetadataError, match="JSON"):
        parse_metadata("{nic")
    with pytest.raises(MetadataError, match="JSON object"):
        parse_metadata("[1, 2]")


def test_core_import_without_rest_mcp_qt():
    """The GUI imports cad_import even without the mcp module (fastapi, SDK mcp) and without PySide6."""
    code = ("import sys; from fea_core import cad_import as c; c.library_material('S235JR'); "
            "c.match_parts(['Deska-1'], [{'name': 'Deska'}]); "
            "bad = [m for m in ('fastapi', 'mcp', 'PySide6', 'fea_core.mcp', 'fea_core.rest') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


# ===================================================================== STEP from SolidWorks, bolts

def _assembly_step(path: Path, names: list[str]):
    """Assembly XCAF: occurrences of the parts named as products (OCCT writes the name of the occurrence = the name of the part)."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.Interface import Interface_Static
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    asm = st.NewShape()
    TDataStd_Name.Set_s(asm, TCollection_ExtendedString("Assembly", True))
    labels = {}
    for i, n in enumerate(names):
        if n not in labels:
            labels[n] = st.AddShape(BRepPrimAPI_MakeBox(10, 10, 10).Shape(), False)
            TDataStd_Name.Set_s(labels[n], TCollection_ExtendedString(n, True))
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(20.0 * i, 0, 0))
        st.AddComponent(asm, labels[n], TopLoc_Location(t))
    st.UpdateAssemblies()
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    w = STEPCAFControl_Writer()
    w.SetNameMode(True)
    w.Transfer(doc, STEPControl_AsIs)
    w.Write(str(path))


def test_read_step_solidworks_unnamed_occurrences(tmp_path):
    """SolidWorks writes occurrences without a name NEXT_ASSEMBLY_USAGE_OCCURRENCE('NAUO1',' ',' ',…) → OCCT would name the parts
    "NAUO1"…; read_step takes the name of the part (the product) – the bolts and the metadata are then recognised by the name.
    """
    import re
    from fea_core.geometry import read_step
    p = tmp_path / "occt.step"
    _assembly_step(p, ["Deska", "Deska", "ISO 4017 - M8 x 30 - 8.8"])
    txt = p.read_text(encoding="utf-8")
    n = iter(range(1, 100))
    sw = re.sub(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\('[^']*','[^']*','[^']*',",
                lambda _m: f"NEXT_ASSEMBLY_USAGE_OCCURRENCE('NAUO{next(n)}',' ',' ',", txt)
    assert sw != txt
    q = tmp_path / "solidworks.step"
    q.write_text(sw, encoding="utf-8")
    assert [x.name for x in read_step(q)] == ["Deska", "Deska (2)", "ISO 4017 - M8 x 30 - 8.8"]
    # a saved study of version 1.0.0 (parts "NAUO1"…) is read the old way – materials, selections and the result mesh stay valid
    assert [x.name for x in read_step(q, legacy_names=True)] == ["NAUO1", "NAUO2", "NAUO3"]


def test_legacy_step_names_detects_old_studies():
    from fea_core import materials as matlib
    from fea_core.geometry import legacy_step_names
    from fea_core.selectors import FaceSelector
    from fea_core.study import Selection
    st = _study()
    assert not legacy_step_names(st) and not legacy_step_names(None)
    st.add_material(matlib.by_name("Nerez 1.4301"))
    st.part_materials["Deska"] = "Nerez 1.4301"
    assert not legacy_step_names(st)
    st.part_materials["NAUO2"] = "Nerez 1.4301"                       # key of the dictionary
    assert legacy_step_names(st)
    st2 = _study()
    sel = Selection("Čelo")
    sel.faces.append(FaceSelector("NAUO7_2", "plane", 1.0, [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0, 1, 1, 0]))
    st2.selections["Čelo"] = sel                                     # value (part of the selection, several bodies "_2")
    assert legacy_step_names(st2)
    st3 = _study()
    st3.ignored_parts = ["NAUO", "Deska NAUO1", "nauo1x"]             # only the whole name "NAUO<number>"
    assert not legacy_step_names(st3)


def test_solidworks_macro_records_do_not_collide_with_part_files():
    """SOLIDWORKS macro: name = the part file (the name in the STEP), the occurrences only in "instances" (they do not match). The occurrence
    "Deska-2" of the part Deska must not take the record of the separate file "Deska-2.sldprt" – in any order.
    """
    a = {"name": "Deska", "names": ["Deska_Default"], "instances": ["Deska-1", "Deska-2"], "material": "S235JR"}
    b = {"name": "Deska-2", "instances": ["Deska-2-1"], "material": "AISI 304"}
    for meta in ([a, b], [b, a]):
        m, un = match_parts(["Deska-2", "Deska", "Deska (2)"], meta)
        assert un == [] and meta[m["Deska-2"]] is b and meta[m["Deska"]] is a and meta[m["Deska (2)"]] is a


@pytest.mark.parametrize("name, kind, d, length, strength, head", [
    ("ISO 4017 - M8 x 30 - 8.8-1", "bolt", 8, 30, "8.8", "hex"),           # occurrence of SolidWorks
    ("ISO 4017 - M8 x 30 - 8.8-12", "bolt", 8, 30, "8.8", "hex"),
    ("ISO 4017 - M8 x 30 - 10.9-1", "bolt", 8, 30, "10.9", "hex"),
    ("ISO 4762 M8 x 25 --- 25N-1", "bolt", 8, 25, None, "socket"),          # Toolbox (thread length "25N")
    ("ISO 4017 - M8 x 30 - A2-70-1", "bolt", 8, 30, "A2-70", "hex"),
    ("ISO 10642 - M6 x 16 - 16N-1", "bolt", 6, 16, None, "countersunk"),
    ("ISO 4014 - M12 x 80 x 30-N-1", "bolt", 12, 80, None, "hex"),          # Toolbox: length × thread length
    ("ISO 4032 - M8 - W - N-1", "nut", 8, None, None, "hex"),
    ("ISO 7089 - 8 - 200 HV-1", "washer", 8, None, None, "hex"),
])
def test_parse_name_solidworks_suffix(name, kind, d, length, strength, head):
    from fea_core.joints.naming import parse_name
    f = parse_name(name)
    assert f is not None, name
    assert (f.kind, f.d, f.length, f.strength, f.head) == (kind, d, length, strength, head), f
    assert f.pitch is None


def test_parse_name_fine_pitch_still_works():
    from fea_core.joints.naming import parse_name
    f = parse_name("ISO 4762 M12 x 1.5 x 40:1")
    assert (f.d, f.pitch, f.length) == (12, 1.5, 40)
