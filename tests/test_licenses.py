"""Collection of the third-party licences (installer/collect_licenses.py, connected in make_package) – fast, without network and without Gmsh/CalculiX."""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

import collect_licenses as cl  # noqa: E402

VENV = ROOT / ".venv"
HAS_VENV = (VENV / "Lib" / "site-packages").is_dir() or any(VENV.glob("lib/python3*/site-packages"))


def _fake_project(root: Path, win: str, linux: str) -> None:
    (root / "installer" / "locks").mkdir(parents=True)
    (root / "installer" / "locks" / "windows-x86_64.txt").write_text(win, encoding="utf-8")
    (root / "installer" / "locks" / "linux-x86_64.txt").write_text(linux, encoding="utf-8")


def _dist(site: Path, name: str, version: str, meta: str, files: dict[str, str] | None = None) -> None:
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{meta}", encoding="utf-8")
    for rel, text in (files or {}).items():
        (info / rel).parent.mkdir(parents=True, exist_ok=True)
        (info / rel).write_text(text, encoding="utf-8")


def test_synthetic_venv_collects_and_warns(tmp_path):
    root = tmp_path / "proj"
    _fake_project(root, "fakepkg==1.0\nnolic==2.0\n", "fakepkg==1.0\nlinuxonly==3.0\n")
    venv = tmp_path / "venv"
    site = venv / "Lib" / "site-packages"
    _dist(site, "fakepkg", "1.0", "License-Expression: MIT\nLicense-File: licenses/LICENSE.txt\n",
          {"licenses/LICENSE.txt": "MIT text\n"})
    _dist(site, "nolic", "2.0", "License: BSD\n")
    out = tmp_path / "licenses"
    warnings = cl.collect(out, venv, root=root)
    assert any("nolic" in w and "has no licence file" in w for w in warnings), warnings
    assert not any("fakepkg" in w for w in warnings)
    assert (out / "python" / "fakepkg-1.0" / "licenses" / "LICENSE.txt").read_text(encoding="utf-8") == "MIT text\n"
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "| Package | Version | Platform | Licence | Texts / note |" in readme
    assert "| fakepkg | 1.0 | Windows + Linux | MIT |" in readme
    assert "| linuxonly | 3.0 | Linux | unknown – not installed |" in readme
    assert "licence file MISSING" in readme
    assert "CalculiX" in (out / "EXTERNAL.md").read_text(encoding="utf-8")


def test_site_packages_layouts(tmp_path):
    (tmp_path / "win" / "Lib" / "site-packages").mkdir(parents=True)
    assert cl.site_packages(tmp_path / "win")[1] == "windows-x86_64"
    (tmp_path / "lin" / "lib" / "python3.11" / "site-packages").mkdir(parents=True)
    assert cl.site_packages(tmp_path / "lin")[1] == "linux-x86_64"


def test_license_files_from_dist_info_root(tmp_path):
    root = tmp_path / "proj"
    _fake_project(root, "rootlic==1.0\n", "rootlic==1.0\n")
    site = tmp_path / "venv" / "Lib" / "site-packages"
    _dist(site, "rootlic", "1.0", "License: MIT\n", {"COPYING": "copying\n", "README.txt": "nic\n"})
    out = tmp_path / "licenses"
    assert cl.collect(out, tmp_path / "venv", root=root) == []
    assert (out / "python" / "rootlic-1.0" / "COPYING").is_file()
    assert not (out / "python" / "rootlic-1.0" / "README.txt").exists()


def test_missing_lock_is_error(tmp_path):
    (tmp_path / "installer" / "locks").mkdir(parents=True)
    (tmp_path / "installer" / "locks" / "windows-x86_64.txt").write_text("a==1\n", encoding="utf-8")
    site = tmp_path / "venv" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        cl.collect(tmp_path / "out", tmp_path / "venv", root=tmp_path)


def test_main_exit_code_stays_zero_with_warnings(tmp_path):
    site = tmp_path / "venv" / "Lib" / "site-packages"
    site.mkdir(parents=True)
    assert cl.main(["--out", str(tmp_path / "out"), "--venv", str(tmp_path / "venv")]) == 0
    assert cl.main(["--out", str(tmp_path / "out2"), "--venv", str(tmp_path / "chybi")]) == 1


@pytest.mark.skipif(not HAS_VENV, reason="chybí vývojové .venv")
def test_real_venv_covers_all_locks(tmp_path):
    out = tmp_path / "licenses"
    warnings = cl.collect(out, VENV)
    assert all(isinstance(w, str) for w in warnings)
    pkgs = cl.read_locks(ROOT)
    readme = (out / "README.md").read_text(encoding="utf-8")
    rows = [l for l in readme.splitlines() if l.startswith("| ") and not l.startswith("| Package ")]
    for entry in pkgs.values():
        assert f"| {entry['name']} |" in readme, f"{entry['name']} chybí v README"
    assert len(rows) >= len(pkgs)
    # installed packages with a licence file have the folder python/<name>-<version>
    folders = {p.name for p in (out / "python").iterdir()}
    assert len(folders) >= 50
    ext = (out / "EXTERNAL.md").read_text(encoding="utf-8")
    assert "https://www.dhondt.de/ccx_2.23.src.tar.bz2" in ext
    assert "https://gmsh.info/src/" in ext
    if (VENV / "Lib" / "site-packages" / "mkl-2023.2.0.dist-info" / "LICENSE.txt").is_file():
        assert "mkl-2023.2.0" in folders


@pytest.mark.skipif(not HAS_VENV, reason="chybí vývojové .venv")
def test_make_package_stage_gets_licenses(tmp_path):
    import make_package  # noqa: E402  (installer/make_package.py)

    make_package.make_licenses(tmp_path)
    assert (tmp_path / "licenses" / "README.md").is_file()
    assert (tmp_path / "licenses" / "EXTERNAL.md").is_file()
    assert any((tmp_path / "licenses" / "python").iterdir())


# --- licence texts downloaded from the official sources (installer/license_texts/, SOURCES.md) ---

TEXTS = ROOT / "installer" / "license_texts"
# Exceptions to the rule "every component of EXTERNAL has a text in external/<slug>/" (only with a justification). Today none: even MS-MPI has
# the MIT text from the sources (a redistributable runtime with its own terms, the user installs it).
EXTERNAL_WITHOUT_TEXT: dict[str, str] = {}


def test_every_external_component_has_a_text(tmp_path):
    copied = cl.copy_external(tmp_path, ROOT)
    for e in cl.EXTERNAL:
        if e["name"] in EXTERNAL_WITHOUT_TEXT:
            continue
        folder = tmp_path / "external" / cl.ext_slug(e)
        assert copied[e["name"]], f"{e['name']}: v external/ není žádný text licence"
        assert all((folder / n).is_file() and (folder / n).stat().st_size > 0 for n in copied[e["name"]])
    slugs = [cl.ext_slug(e) for e in cl.EXTERNAL]
    assert len(slugs) == len(set(slugs)), "názvy složek external/ se opakují"


def test_external_patterns_match_files():
    # the pattern in EXTERNAL["texts"] and in PYTHON_EXTRA must find something (a typo in the pattern would silently leave the text out)
    for e in cl.EXTERNAL:
        for pattern in e.get("texts", []):
            assert cl.license_text_files(ROOT, [pattern]), f"{e['name']}: vzor {pattern} nic nenašel"
    for key, patterns in cl.PYTHON_EXTRA.items():
        for pattern in patterns:
            assert cl.license_text_files(ROOT, [pattern]), f"{key}: vzor {pattern} nic nenašel"


def test_python_extra_texts_for_wheels_without_license(tmp_path):
    root = tmp_path / "proj"
    _fake_project(root, "gmsh==4.15.2\ncadquery-ocp==8.0.1.1.0\nngsolve-openblas==0.3.33\nnolic==2.0\n",
                  "gmsh==4.15.2\ncadquery-ocp==8.0.1.1.0\nngsolve-openblas==0.3.33\n")
    shutil.copytree(TEXTS, root / "installer" / "license_texts")
    site = tmp_path / "venv" / "Lib" / "site-packages"
    _dist(site, "gmsh", "4.15.2", "License: GPLv2+\n")
    _dist(site, "cadquery_ocp", "8.0.1.1.0", "License: Apache-2.0\n")
    _dist(site, "ngsolve_openblas", "0.3.33", "License: BSD-3-Clause\n")
    _dist(site, "nolic", "2.0", "License: BSD\n")
    out = tmp_path / "licenses"
    warnings = cl.collect(out, tmp_path / "venv", root=root)
    assert [w for w in warnings if "has no licence file" in w] and all("nolic" in w for w in warnings if "has no licence file" in w)
    assert {p.name for p in (out / "python" / "gmsh-4.15.2").iterdir()} >= {"LICENSE.txt", "gpl-2.0.txt"}
    assert {p.name for p in (out / "python" / "cadquery-ocp-8.0.1.1.0").iterdir()} >= {
        "LICENSE", "LICENSE_LGPL_21.txt", "OCCT_LGPL_EXCEPTION.txt"}
    assert (out / "python" / "ngsolve-openblas-0.3.33" / "LICENSE").is_file()


@pytest.mark.skipif(not HAS_VENV, reason="chybí vývojové .venv")
def test_real_venv_has_texts_for_known_gaps(tmp_path):
    out = tmp_path / "licenses"
    warnings = cl.collect(out, VENV)
    folders = {p.name: p for p in (out / "python").iterdir()}
    for prefix in ("gmsh-", "cadquery-ocp-", "cadquery-ocp-proxy-", "ngsolve-openblas-"):
        hit = [n for n in folders if n.startswith(prefix) and n[len(prefix)].isdigit()]
        assert hit, f"{prefix}: chybí složka python/{prefix}<verze>"
        assert any(f.is_file() for f in folders[hit[0]].iterdir())
    for w in warnings:
        assert not any(x in w for x in ("gmsh", "cadquery-ocp", "ngsolve-openblas")), w
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "## Data" in readme and "pragtic_vmat.json" in readme and "until the author agrees" in readme
    assert "## Data" in (out / "EXTERNAL.md").read_text(encoding="utf-8")


def test_sources_md_lists_every_text_with_correct_hash():
    sources = (TEXTS / "SOURCES.md").read_text(encoding="utf-8")
    files = [p for p in TEXTS.rglob("*") if p.is_file() and p.name != "SOURCES.md"]
    assert len(files) >= 25
    table_rows = [l for l in sources.splitlines() if l.startswith("| ") and l.split("|")[2].strip().startswith("`")]
    assert len(table_rows) == len(files), "SOURCES.md popisuje jiný počet souborů, než je ve složce"
    for f in files:
        rel = f.relative_to(TEXTS).as_posix()
        rows = [l for l in sources.splitlines() if l.startswith("|") and l.split("|")[2].strip() == f"`{rel}`"]
        assert len(rows) == 1, f"{rel}: v SOURCES.md má být právě jeden řádek"
        assert f"`{hashlib.sha256(f.read_bytes()).hexdigest()}`" in rows[0], f"{rel}: SHA-256 v SOURCES.md nesedí"
        assert "16. 10. 2026" in rows[0], f"{rel}: chybí datum stažení"


def test_license_texts_are_real_texts():
    # downloaded correctly: no HTML page with an error, no empty file
    for f in TEXTS.rglob("*"):
        if f.is_file() and f.name != "SOURCES.md" and f.suffix != ".html":
            head = f.read_bytes()[:2000].lower()
            assert b"<!doctype" not in head and b"<html" not in head, f"{f} vypadá jako HTML stránka"
        if f.is_file():
            assert f.stat().st_size > 200, f"{f} je podezřele malý"
