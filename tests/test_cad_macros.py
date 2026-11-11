"""The macros "Send to FEA Studio" for SOLIDWORKS (VBA) and Inventor (iLogic): static check of the content
and the pattern of the JSON metadata that the macros write next to the STEP (the import format from the CAD, fea_core/rest/README.md).

The macros cannot be run without the CAD – it is checked that they contain the key parts (STEP export, JSON, search for the folder of
FEA Studio, starting the GUI), that they avoid constructions that would not compile in older CAD versions, and that the
JSON in the form the macros write it passes json.loads and the matching of the parts of FEA Studio.
"""
import json
import re
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SW_DIR = ROOT / "adapters" / "solidworks"
INV_DIR = ROOT / "adapters" / "inventor"
SW = SW_DIR / "FEAStudio_Export.bas"
INV = INV_DIR / "FEAStudio_Export.iLogicVb"


def _split_vb(text: str) -> tuple[str, str]:
    """(code, comments) – an apostrophe outside a string starts a comment (VBA and VB.NET)."""
    code, comments = [], []
    for line in text.splitlines():
        in_str = False
        cut = len(line)
        for i, ch in enumerate(line):
            if ch == '"':
                in_str = not in_str
            elif ch == "'" and not in_str:
                cut = i
                break
        code.append(line[:cut])
        comments.append(line[cut:])
    return "\n".join(code), "\n".join(comments)


def _sw_text() -> str:
    return SW.read_bytes().decode("ascii")


def _inv_text() -> str:
    raw = INV.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "pravidlo iLogic má být v UTF-8 s BOM (čeština)"
    return raw.decode("utf-8-sig")


# ===================================================================== soubory

def test_macro_files_exist():
    for p in (SW, SW_DIR / "README.md", SW_DIR / "FEAStudio_Export.bmp",
              INV, INV_DIR / "README.md", INV_DIR / "FEAStudio_Export.png"):
        assert p.is_file(), p


def test_line_endings_are_consistent():
    """VBA import and iLogic expect CRLF; mixed line ends never (a Linux clone without .gitattributes has LF)."""
    for p in (SW, INV):
        raw = p.read_bytes()
        crlf, lf = raw.count(b"\r\n"), raw.count(b"\n")
        assert crlf in (0, lf), f"{p.name}: míchané konce řádků"
        if sys.platform == "win32":
            assert crlf == lf, f"{p.name}: na Windows má být CRLF"
    # git stores the macros without changing the line ends (otherwise the commit and the package from git would have LF and the VBA editor would
    # misread the module header "Attribute VB_Name")
    attrs = ROOT / ".gitattributes"
    if attrs.is_file():
        rules = {tuple(r.split()) for r in attrs.read_text(encoding="utf-8").splitlines() if r.strip()}
        assert ("*.bas", "-text") in rules and ("*.iLogicVb", "-text") in rules


# ===================================================================== SOLIDWORKS (VBA)

def test_solidworks_macro_is_ascii():
    """The VBA editor imports .bas in the code page of the system → only ASCII (Czech via \\\\uXXXX and T())."""
    raw = SW.read_bytes()
    bad = [i for i, b in enumerate(raw) if b > 127]
    assert not bad, f"neASCII bajt na pozici {bad[0]}"


def test_solidworks_macro_structure():
    text = _sw_text()
    code, comments = _split_vb(text)
    assert text.splitlines()[0] == 'Attribute VB_Name = "FEAStudio_Export"'
    assert re.search(r"^Option Explicit\s*$", code, re.M)
    # "Tools > Macro > Run" runs the last Sub without parameters in the module created last
    subs = re.findall(r"^\s*(?:Public\s+|Private\s+)?Sub\s+(\w+)\s*\(([^)]*)\)", code, re.M)
    assert [n for n, args in subs if not args.strip()] == ["main"]
    procs = re.findall(r"^\s*(?:Public\s+|Private\s+)?(?:Sub|Function)\s+(\w+)", code, re.M)
    assert procs[-1] == "main"
    # Win32 declarations for 64-bit VBA 7
    for d in re.findall(r"^.*\bDeclare\b.*$", code, re.M):
        assert "PtrSafe" in d, d
    # key parts
    for key in ("swStepAP, 214", "swAcisOutputAsSolidAndSurface", "swStepExportSplitPeriodic, False",
                "ClearSelection2 True", "Extension.SaveAs(", ".step\"", ".json\"", "RestorePrefs",
                '"FEA_STUDIO_HOME"', '"\\FEAStudio\\install.json"', '"home"', '"\\FEAStudio"', '"\\export"',
                '"\\Programs\\FEA Studio"', '"\\.venv\\Scripts\\pythonw.exe"', '"-m fea_gui ', '"\\FEA Studio.bat"',
                "ShellExecuteW", "GetEnvironmentVariableW", "IsSuppressed", "GetMaterialPropertyName2",
                "ResolveAllLightWeightComponents", "SendMsgToUser2", "Format$(Now, \"yyyymmdd_hhnnss\")",
                '""units"": ""mm""', '""parts"": [', '""instances"": '):
        assert key in code, key
    assert "%TEMP%\\FEAStudio\\export" in comments
    assert "%APPDATA%\\FEAStudio\\install.json" in comments
    # the record of the part is named after the part file (the name in the STEP), the occurrences "Deska-1" go only into "instances"
    assert "NewEntry(key, baseNm)" in code and "AddInstance e, shortNm" in code
    assert not re.search(r"AddName e, (shortNm|full)\b", code)


def test_solidworks_macro_avoids_version_and_ansi_pitfalls():
    code, _ = _split_vb(_sw_text())
    # the API only from SOLIDWORKS 2023 / 2020 SP02 – with Option Explicit older versions could not compile it
    for name in ("swStepExportAppearances", "swStepExportAtomicSave", "SaveAs3"):
        assert name not in code, name
    # ANSI functions of VBA (diacritics in the paths and messages) and VBScript (shutdown in Windows)
    for rx in (r"\bMsgBox\b", r"\bEnviron\$?\s*\(", r"\bShell\s*[ (]", r"\bMkDir\b", r"\bDir\$?\s*\(",
               r"VBScript\.RegExp", r"\bOpen[ \t]+\S.*[ \t]For[ \t]+(Output|Input|Binary|Append)\b"):
        assert not re.search(rx, code), rx


def test_solidworks_escaped_texts_go_through_T():
    """Strings with \\\\uXXXX must go through T(), otherwise the user would see escape sequences."""
    code, _ = _split_vb(_sw_text())
    logical = re.sub(r" _[ \t]*\n", " ", code)    # continuation lines
    for line in logical.splitlines():
        for lit in re.findall(r'"((?:[^"]|"")*)"', line):
            if re.search(r"\\u[0-9A-Fa-f]{4}", lit):
                assert "T(" in line, line.strip()
    for esc in re.findall(r"\\u([0-9A-Za-z]{1,4})", logical):     # the bare "\u" (in JsonStr) is not counted
        assert re.fullmatch(r"[0-9A-F]{4}", esc), esc


# ===================================================================== Inventor (iLogic, VB.NET)

def test_inventor_rule_structure():
    text = _inv_text()
    code, comments = _split_vb(text)
    assert re.search(r"^Sub Main\(\)", code, re.M)
    for key in ('ItemById("{90AF7F40-0C01-11D5-8E83-0010B541CD80}")', 'opts.Value("ApplicationProtocolType")',
                'opts.Value("IncludeSketches") = False', "IOMechanismEnum.kFileBrowseIOMechanism",
                "tr.SaveCopyAs(", "New System.Text.UTF8Encoding(False)", '"FEA_STUDIO_HOME"', '"install.json"',
                '"home"', '"FEAStudio", "export"', '"Programs", "FEA Studio"', '".venv", "Scripts", "pythonw.exe"',
                '"-m fea_gui ', '"FEA Studio.bat"', "UseShellExecute = False", "WorkingDirectory = home",
                "AllLeafOccurrences", "p.Suppressed", "VirtualComponentDefinition", "ActiveMaterial",
                '"Part Number"', '""units"": ""mm""', '""parts"": ['):
        assert key in code, key
    assert "%TEMP%\\FEAStudio\\export" in comments
    assert "%APPDATA%\\FEAStudio\\install.json" in comments


def test_inventor_rule_uses_vb2012_syntax_only():
    """iLogic (Inventor 2021–2026) translates only VB 2012: no interpolation, ?. and NameOf; no Imports."""
    code, _ = _split_vb(_inv_text())
    no_strings = re.sub(r'"(?:[^"]|"")*"', '""', code)
    assert '$"' not in no_strings
    assert not re.search(r"\?\.|\?\(", no_strings)
    assert not re.search(r"\bNameOf\s*\(", no_strings)
    assert not re.search(r"^\s*Imports\s", code, re.M)


# ===================================================================== without personal data

def test_no_personal_paths_in_adapters():
    for d in (SW_DIR, INV_DIR):
        for p in d.iterdir():
            if p.suffix.lower() not in (".bas", ".ilogicvb", ".md"):
                continue
            t = p.read_bytes().decode("utf-8-sig")
            assert not re.search(r"[A-Za-z]:[\\/]+Users[\\/]", t, re.I), p.name
            assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", t), p.name


# ===================================================================== JSON metadat

def vba_json_str(s: str) -> str:
    """A Python counterpart of JsonStr from the SOLIDWORKS macro: characters outside ASCII and control characters as \\\\uXXXX (UTF-16)."""
    out = []
    raw = s.encode("utf-16-le")
    for (code,) in struct.iter_unpack("<H", raw):
        if code == 0x22:
            out.append('\\"')
        elif code == 0x5C:
            out.append("\\\\")
        elif code < 32 or code > 126:
            out.append(f"\\u{code:04X}")
        else:
            out.append(chr(code))
    return '"' + "".join(out) + '"'


# The JSON pattern as the SOLIDWORKS macro writes it (all in ASCII, CRLF, outside ASCII \uXXXX)
SW_EXAMPLE = (
    '{"source": "SOLIDWORKS",\r\n'
    ' "source_version": "32.1.0",\r\n'
    ' "adapter": "FEAStudio_Export.bas",\r\n'
    ' "document": "Dr\\u017E\\u00E1k \\u201Ezkou\\u0161ka\\u201C",\r\n'
    ' "document_type": "assembly",\r\n'
    ' "configuration": "V\\u00FDchoz\\u00ED",\r\n'
    ' "units": "mm",\r\n'
    ' "step_protocol": "AP214",\r\n'
    ' "exported": "2026-10-15 12:00:00",\r\n'
    ' "parts": [\r\n'
    # name = the part file (as the STEP writes it), the occurrences from the tree only in "instances" (they do not match)
    '  {"name": "Deska", "names": ["Deska_Default"], "component": "Deska", '
    '"material": "AISI 304", "material_database": "SOLIDWORKS Materials", "configuration": "Default", '
    '"color": [0.5000, 0.2500, 1], "count": 2, "instances": ["Deska-1", "Deska-2"]},\r\n'
    '  {"name": "ISO 4017 - M8 x 30 - 8.8", "names": [], "component": "ISO 4017 - M8 x 30 - 8.8", "count": 1, '
    '"instances": ["ISO 4017 - M8 x 30 - 8.8-1", "Pod-1/ISO 4017 - M8 x 30 - 8.8-1"]},\r\n'
    '  {"name": "Dr\\u017E\\u00E1k", "names": ["Dr\\u017E\\u00E1k_Tisk 5mm"], '
    '"component": "Dr\\u017E\\u00E1k", "material": "PLA", "configuration": "Tisk 5mm", "count": 1, "hidden": 1, '
    '"body_materials": "Telo1: PLA; Vlo\\u017Eka: S235JR", "instances": ["Dr\\u017E\\u00E1k-1"]}\r\n'
    ' ],\r\n'
    ' "notes": ["1 skryt\\u00FDch sou\\u010D\\u00E1st\\u00ED: \\u2026 C:\\\\Data\\\\x"]\r\n'
    '}\r\n')

# The JSON pattern of the Inventor rule (UTF-8 without BOM, Czech directly)
INV_EXAMPLE = (
    '{"source": "Autodesk Inventor",\r\n'
    ' "source_version": "2025.2",\r\n'
    ' "adapter": "FEAStudio_Export.iLogicVb",\r\n'
    ' "document": "Držák \\"v1\\"",\r\n'
    ' "document_type": "assembly",\r\n'
    ' "units": "mm",\r\n'
    ' "step_protocol": "AP214",\r\n'
    ' "exported": "2026-10-15 12:00:00",\r\n'
    ' "parts": [\r\n'
    '  {"name": "Deska:1", "names": ["DSK-001", "Deska", "Deska:2"], "component": "Deska", '
    '"part_number": "DSK-001", "material": "S235JR", "material_key": "Steel-S235", '
    '"color": [0.7843, 0.3922, 0.1961], "count": 2},\r\n'
    '  {"name": "ISO 4017 M8 x 30:1", "names": ["ISO 4017 M8 x 30", "Podsestava:1/ISO 4017 M8 x 30:1"], '
    '"component": "ISO 4017 M8 x 30", "part_number": "ISO 4017 M8 x 30", "material": "Ocel, měkká", '
    '"material_key": "Steel", "count": 1},\r\n'
    '  {"name": "Kryt \\"A\\":1", "names": ["Kryt (A) – zkouška"], "component": "Kryt (A) – zkouška", '
    '"material": "Hliník 6061", "count": 1, "hidden": 1}\r\n'
    ' ],\r\n'
    ' "notes": ["1 skrytých výskytů: Inventor je do STEP zapisuje"]\r\n'
    '}\r\n')


def _check_structure(d: dict) -> None:
    """Struktura metadat importu z CADu (fea_core/rest/README.md → „Import z CADu“)."""
    assert isinstance(d, dict)
    assert isinstance(d["source"], str) and d["source"]
    assert isinstance(d["document"], str) and d["document"]
    assert d["units"] == "mm"
    assert isinstance(d["parts"], list) and d["parts"]
    generic = {"default", "body1", "solid1", "part1"}
    for p in d["parts"]:
        assert isinstance(p, dict)
        assert isinstance(p["name"], str) and p["name"]
        assert isinstance(p["names"], list) and all(isinstance(n, str) and n for n in p["names"])
        assert not generic & {n.casefold() for n in p["names"]}, "obecné názvy dělají falešné shody"
        if "material" in p:
            assert isinstance(p["material"], str) and p["material"]
        if "color" in p:
            assert len(p["color"]) == 3 and all(0 <= c <= 1 for c in p["color"])
        assert isinstance(p["count"], int) and p["count"] >= 1
        assert all(isinstance(n, str) and n for n in p.get("instances", []))
    assert all(isinstance(n, str) for n in d.get("notes", []))


@pytest.mark.parametrize("example", [SW_EXAMPLE, INV_EXAMPLE], ids=["solidworks", "inventor"])
def test_metadata_example_parses(example):
    raw = example.encode("utf-8")
    assert not raw.startswith(b"\xef\xbb\xbf")
    d = json.loads(raw)
    _check_structure(d)


def test_solidworks_example_is_ascii_and_decodes_czech():
    assert all(ord(c) < 128 for c in SW_EXAMPLE)
    d = json.loads(SW_EXAMPLE)
    assert d["document"] == "Držák „zkouška“"
    assert d["parts"][2]["body_materials"] == "Telo1: PLA; Vložka: S235JR"
    assert d["notes"][0].endswith("… C:\\Data\\x")


def test_vba_json_escaping_roundtrip():
    """Escaping as JsonStr in the macro: quotes, backslashes, control characters, Czech and characters outside the BMP."""
    for s in ("Držák „A“", 'Kryt "B" \\ C:\\Data\\díl.step', "tab\tnový\r\nřádek", "😀 emoji", "", "\u2028\uffff"):
        enc = vba_json_str(s)
        assert all(ord(c) < 128 for c in enc)
        assert json.loads(enc) == s


def _metadata_module():
    for name in ("fea_core.cad_metadata", "fea_core.rest.imports"):
        try:
            return __import__(name, fromlist=["match_parts"])
        except ImportError:
            continue
    return None


@pytest.mark.parametrize("example, step_names", [
    # SOLIDWORKS: the STEP carries the name of the part file (+ "_<configuration>" for a non-default configuration), occurrences without a name
    (SW_EXAMPLE, ["Deska", "Deska (2)", "ISO 4017 - M8 x 30 - 8.8", "Držák_Tisk 5mm"]),
    # Inventor: the part = the name of the product (the part number), assemblies probably the names of the occurrences
    (INV_EXAMPLE, ["DSK-001", "DSK-001 (2)", "ISO 4017 M8 x 30", "Kryt (A) – zkouška"]),
    (INV_EXAMPLE, ["Deska:1", "Deska:2", "ISO 4017 M8 x 30:1", "Kryt \"A\":1"]),
], ids=["solidworks", "inventor-produkt", "inventor-vyskyt"])
def test_metadata_example_matches_step_parts(example, step_names):
    mod = _metadata_module()
    if mod is None:
        pytest.skip("modul párování metadat není k dispozici")
    meta = json.loads(example)
    if hasattr(mod, "parse_metadata"):
        meta = mod.parse_metadata(example.encode("utf-8"))
    found, unused = mod.match_parts(step_names, meta["parts"])
    assert set(found) == set(step_names), f"nespárované díly: {set(step_names) - set(found)}"
    assert unused == [], f"záznamy metadat bez dílu: {unused}"
