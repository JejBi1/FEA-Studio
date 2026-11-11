"""STEP references to external files (DOCUMENT_FILE) outside the folder of the file are refused before the OCCT reads them."""
from __future__ import annotations

import pytest

from fea_core import config
from fea_core.geometry import check_step_external_refs

BS = "\\"
HEAD = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"


def _step(tmp_path, ref: str):
    ref = ref.replace(BS, BS * 2)                        # a backslash written in a STEP
    p = tmp_path / "a.stp"
    p.write_text(HEAD + f"#10=DOCUMENT_FILE('{ref}','',$,#11,'',$);\n#20=PRODUCT('Deska \\X2\\00E9\\X0\\','','',"
                        "(#1));\nENDSEC;\nEND-ISO-10303-21;\n", encoding="latin-1")
    return p


@pytest.mark.parametrize("ref", [BS * 2 + "server" + BS + "s" + BS + "a.stp", "//server/s/a.stp",
                                 BS + "??" + BS + "UNC" + BS + "server" + BS + "a.stp", "D:" + BS + "x" + BS + "a.stp",
                                 "../jinde/a.stp"])
def test_external_ref_outside_folder_rejected(tmp_path, ref):
    with pytest.raises(ValueError):
        check_step_external_refs(_step(tmp_path, ref))


@pytest.mark.parametrize("ref", ["dil.stp", "podslozka/dil.stp"])
def test_external_ref_inside_folder_allowed(tmp_path, ref):
    check_step_external_refs(_step(tmp_path, ref))


def test_solver_name_never_returns_input():
    bad = "SPOOLES\n*INCLUDE, INPUT=" + BS * 2 + "server" + BS + "x.inp"
    assert config.solver_name(bad) in config.KNOWN_SOLVERS
    assert "\n" not in config.solver_name(bad)
    assert config.solver_name("spooles") == "SPOOLES"


def test_which_ignores_current_dir(tmp_path, monkeypatch):
    exe = tmp_path / ("podvrh" + (".exe" if config.IS_WIN else ""))
    exe.write_bytes(b"")
    exe.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    assert config.which("podvrh") is None
