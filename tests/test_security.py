"""Safety guards (review 17. 10. 2026): inputs from a foreign project, MCP and REST."""
import zipfile

import pytest

from fea_core.ccx import deck_text
from fea_core.export_csv import text_cell
from fea_core.mcp.common import where_filter
from fea_core.mcp.session import FeaError, safe_name, user_path
from fea_core.pathcheck import is_unc
from fea_core.study import Study, StudyFormatError


@pytest.mark.parametrize("s, unc", [
    ("\\\\evil\\s\\x.step", True), ("/\\evil\\s", True), ("\\/evil/s", True), ("//evil/s", True),
    ("\\\\?\\UNC\\evil\\s", True), ("C:\\data\\y.step", False), ("rel/x.fea", False), ("/home/u/x.fea", False)])
def test_unc_paths(s, unc):
    assert is_unc(s) == unc
    if unc:
        with pytest.raises(FeaError):
            user_path(s)


@pytest.mark.parametrize("expr, ok", [
    ('area > 100 and type == "plane"', True), ("cz**2 > 4", True), ('part in ("a", "b")', True),
    ("abs(nz) > 0.9", True), ("(9**8)**8 > 1", False), ("type*10 == 1", False), ("(1,)*10 == 1", False),
    ("'a'*5 == 'x'", False), ("x" * 600, False), ("area.__class__", False)])
def test_where_limits(expr, ok):
    if ok:
        where_filter(expr)
    else:
        with pytest.raises(FeaError):
            where_filter(expr)


def test_deck_text_single_line():
    assert deck_text("Nosník\n*INCLUDE, INPUT=x\r\n") == "Nosn?k *INCLUDE, INPUT=x "      # one line
    assert deck_text("*BOUNDARY") == "BOUNDARY"


def test_solver_exe_not_taken_from_file():
    st = Study.from_dict({"name": "x", "geometry": "", "solver": {"exe": "evil.exe", "solver": "PARDISO"}})
    assert st.solver.exe is None
    st.solver.exe = "ccx.exe"
    assert "exe" not in st.to_dict()["solver"]


def test_csv_formula_cells():
    assert text_cell("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert text_cell("Stav 1") == "Stav 1"
    assert safe_name("CON") == "_CON" and safe_name("Stav 1") == "Stav_1"


def test_project_extracts_only_known_members(tmp_path):
    from fea_core.project import load_project
    f = tmp_path / "x.fea"
    st = Study(name="x", geometry="")
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("study.json", st.to_json() if hasattr(st, "to_json") else __import__("json").dumps(st.to_dict()))
        z.writestr("evil.bat", "echo")
        z.writestr("sub/evil.dll", "x")
    wd = tmp_path / "wd"
    load_project(f, wd)
    assert sorted(p.name for p in wd.iterdir()) == ["study.json"]


def test_project_bomb_rejected(tmp_path):
    from fea_core.project import load_project
    f = tmp_path / "bomb.fea"
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("results.npz", b"\0" * (120 << 20))       # 120 MB of zeros → compression ratio ~1000
    with pytest.raises(StudyFormatError):
        load_project(f, tmp_path / "wd")
