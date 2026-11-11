"""The cross-check against Abaqus is private: never in the release packages, never imported by shipped code."""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

mp = importlib.import_module("make_package")

PRIVATE = [
    "verification/abaqus_check.py", "verification/abaqus_all.py", "verification/abaqus_extract.py",
    "verification/abaqus_report.py", "verification/abaqus_dynamics.py", "verification/abaqus_new_case.py",
    "verification/build_docs_abaqus_report.py", "verification/make_abaqus_drawings_full.py",
    "verification/out_abaqus/report.json", "verification/out_abaqus/srovnani_abaqus.html",
    "tests/test_abaqus_all.py", "docs/Srovnani_Abaqus.html", "docs/projekt/F9_zadani_abaqus_ansys.md",
    "verification/payload.json", "verification/generate_interactive_html.py", "verification/inline_slim.html",
]
PUBLIC = [
    "verification/run_all.py", "verification/cases.py", "verification/frame_cases.py", "tests/test_modules.py",
    "fea_core/ccx/inp.py", "docs/FEA Studio Manual.pdf", "installer/make_package.py", "LICENSE-EXCEPTIONS.md",
]


@pytest.mark.parametrize("path", PRIVATE)
def test_private_paths_are_not_packaged(path):
    assert not mp.packaged(path), path


@pytest.mark.parametrize("path", PUBLIC)
def test_public_paths_are_packaged(path):
    assert mp.packaged(path), path


def test_file_list_has_no_private_file():
    files = mp.file_list(None)
    assert "verification/run_all.py" in files
    leaked = [f for f in files if "abaqus" in f.lower() or f in mp.PRIVATE_FILES]
    assert not leaked, leaked


PRIVATE_IMPORT = re.compile(
    r"(?:from|import)\s+verification(?:\.|\s+import\s+)(abaqus_\w+|build_docs_abaqus_report|make_abaqus_drawings_full)"
    r"|-m['\"]?,?\s*['\"]?verification\.abaqus")


def test_shipped_code_does_not_import_the_private_modules():
    bad = []
    for f in mp.file_list(None):
        if f.endswith(".py"):
            text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
            if PRIVATE_IMPORT.search(text):
                bad.append(f)
    assert not bad, bad


def test_github_export_excludes_the_private_paths():
    excl = ROOT / ".git" / "info" / "github-export-exclude.txt"
    if not excl.is_file():
        pytest.skip("local export configuration missing (only in the development tree)")
    lines = [l.strip() for l in excl.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    for need in ("verification/abaqus_*", "verification/out_abaqus/*", "tests/test_abaqus_all.py", "docs/Srovnani_Abaqus.html",
                 "verification/build_docs_abaqus_report.py", "verification/make_abaqus_drawings_full.py", "docs/projekt/*"):
        assert need in lines, need
