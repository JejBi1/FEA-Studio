"""STEP external references: the text pre-check decodes ISO 10303-21 escapes, scans every EXTERNAL / DOCUMENT /
IDENTIFICATION entity with a real string tokenizer, and the names OCCT itself resolved are checked before Transfer."""
from __future__ import annotations

import os
import re
import shutil

import pytest

from fea_core import geometry
from fea_core.geometry import check_step_external_refs, decode_step_string

BS = "\\"
HEAD = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"


def _write(tmp_path, body: str):
    p = tmp_path / "a.stp"
    p.write_text(HEAD + body + "\nENDSEC;\nEND-ISO-10303-21;\n", encoding="latin-1")
    return p


def test_decode_escapes():
    assert decode_step_string("C" + BS + "X" + BS + "3A/x") == "C:/x"
    assert decode_step_string(BS + "X2" + BS + "003A005C" + BS + "X0" + BS) == ":" + BS
    assert decode_step_string("a''b") == "a'b"
    assert decode_step_string(BS + BS + "srv") == BS + "srv"


@pytest.mark.parametrize("body", [
    # the name only in APPLIED_EXTERNAL_IDENTIFICATION_ASSIGNMENT (OCCT reads it from there)
    "#5=APPLIED_EXTERNAL_IDENTIFICATION_ASSIGNMENT('C:/elsewhere/p.stp',#6,#7,(#8));",
    # drive colon as an escape
    "#5=DOCUMENT_FILE('C" + BS + "X" + BS + "3A/elsewhere/p.stp','',$,#6,'',$);",
    # UNC with backslashes as X2 escape
    "#5=EXTERNAL_SOURCE(IDENTIFIER('" + BS + "X2" + BS + "005C005C" + BS + "X0" + BS + "srv/s/p.stp'));",
    # a ';' inside an earlier string must not end the entity
    "#5=DOCUMENT_FILE('x;y','',$,#6,'//srv/s/p.stp',$);",
    # complex instance, other keyword spelling, lower case
    "#5=(applied_external_identification_assignment('../up/p.stp',#6,#7,(#8)));",
    # undecodable escape
    "#5=DOCUMENT_FILE('C" + BS + "X" + BS + "Q/p.stp','',$,#6,'',$);",
])
def test_text_precheck_rejects(tmp_path, body):
    with pytest.raises(ValueError):
        check_step_external_refs(_write(tmp_path, body))


@pytest.mark.parametrize("body", [
    "#5=DOCUMENT_FILE('dil.stp','',$,#6,'',$);",
    "#5=APPLIED_EXTERNAL_IDENTIFICATION_ASSIGNMENT('sub/dil.stp',#6,#7,(#8));",
    "#5=PRODUCT('C:/not/a/reference','','',(#1));",          # strings of other entities are not references
    "#5=DOCUMENT_FILE('Deska " + BS + "X2" + BS + "00E9" + BS + "X0" + BS + ".stp','',$,#6,'',$);",
])
def test_text_precheck_allows(tmp_path, body):
    check_step_external_refs(_write(tmp_path, body))


def _multi_file(folder):
    """A two-component STEP written by OCCT itself (component files next to the root)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopoDS import TopoDS_Compound
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    b.Add(comp, BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 10, 10, 10).Shape())
    b.Add(comp, BRepPrimAPI_MakeBox(gp_Pnt(20, 0, 0), 5, 5, 5).Shape())
    st.AddShape(comp, True)
    w = STEPCAFControl_Writer()
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        w.Transfer(doc, STEPControl_AsIs, "part")
        w.Write("root.stp")
    finally:
        os.chdir(cwd)
    return folder / "root.stp"


def test_occt_view_is_checked_too(tmp_path, monkeypatch):
    a = tmp_path / "a"
    other = tmp_path / "elsewhere"
    a.mkdir()
    other.mkdir()
    root = _multi_file(a)
    assert len(geometry.read_step(root)) == 2                 # relative references inside the folder still work
    text = root.read_bytes()
    comps = sorted(p for p in a.iterdir() if p != root)
    assert comps
    for p in comps:
        shutil.move(str(p), str(other / p.name))

    def to_abs(m):
        body = m.group(0)
        for p in comps:
            body = body.replace(b"'" + p.name.encode() + b"'", b"'" + (other / p.name).as_posix().encode() + b"'")
        return body
    bad = a / "bad.stp"
    bad.write_bytes(re.sub(rb"APPLIED_EXTERNAL_IDENTIFICATION_ASSIGNMENT\s*\([^;]*;", to_abs, text))
    with pytest.raises(ValueError):
        geometry.read_step(bad)
    # even if the text pre-check were bypassed, the names OCCT resolved are refused before Transfer
    monkeypatch.setattr(geometry, "check_step_external_refs", lambda path: None)
    with pytest.raises(ValueError):
        geometry.read_step(bad)


def test_part_volume_cached():
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    p = geometry.Part("box", BRepPrimAPI_MakeBox(2.0, 3.0, 4.0).Shape())
    assert p.volume() == pytest.approx(24.0)
    assert p.__dict__["_volume_cache"][1] == pytest.approx(24.0)
    p.shape = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    assert p.volume() == pytest.approx(1.0)
