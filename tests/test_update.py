"""Update of the installation: version, manifest of the program files, migrations, deleting obsolete files (synthetic trees,
without network, uv and CalculiX).
"""
from __future__ import annotations

import gzip
import importlib
import io
import json
import os
import re
import sys
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

upgrade = importlib.import_module("upgrade")
migrations = importlib.import_module("migrations")


def _tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ verze a manifest
def test_parse_version_and_program_version(tmp_path):
    assert upgrade.parse_version("1.0.1") == (1, 0, 1)
    assert upgrade.parse_version("1.2") == (1, 2, 0)
    assert upgrade.parse_version("1.1.0rc1") == (1, 1, 0)
    assert upgrade.parse_version("1.0.10") > upgrade.parse_version("1.0.9")
    _tree(tmp_path, {"fea_core/__init__.py": '"""x"""\n__version__ = "2.3.4"\n'})
    assert upgrade.program_version(tmp_path) == "2.3.4"
    assert upgrade.program_version(ROOT).count(".") == 2


def test_manifest_of_dir_skips_tools_cache_and_itself(tmp_path):
    _tree(tmp_path, {"fea_core/a.py": "a", "fea_core/__pycache__/a.cpython-311.pyc": "b", "tools/uv/uv.exe": "c",
                     "installer/manifest.json": "{}", "README.md": "r"})
    m = upgrade.manifest_of_dir(tmp_path, "1.2.3")
    assert m["version"] == "1.2.3"
    assert set(m["files"]) == {"fea_core/a.py", "README.md"}
    assert m["files"]["README.md"] == upgrade.bytes_hash(b"r")
    upgrade.write_manifest(tmp_path / "m.json", m)
    assert upgrade.read_manifest(tmp_path / "m.json") == m
    assert upgrade.read_manifest(tmp_path / "chybi.json") is None


def test_history_manifests_are_relative_and_anonymous():
    """The bundled manifests of older releases: only relative paths and digests (nothing personal, no absolute paths)."""
    hist = upgrade.history_manifests(ROOT)
    assert {"0.1.0", "1.0.0"} <= set(hist)
    for ver, m in hist.items():
        for rel, digest in m["files"].items():
            assert not rel.startswith(("/", "\\")) and ":" not in rel and ".." not in rel.split("/"), (ver, rel)
            assert re.fullmatch(r"[0-9a-f]{16}", digest), (ver, rel)
    assert "fea_core/contacts.py" in hist["0.1.0"]["files"]          # replaced by the package contacts/ in version 1.0.0


# ------------------------------------------------------------------ deleting obsolete files
def test_remove_obsolete_keeps_user_data_and_modified_files(tmp_path):
    root = tmp_path / "app"
    files = {"fea_core/stary.py": "old", "fea_core/upraveny.py": "old2", "fea_core/novy.py": "n",
             "stara_slozka/x.txt": "x", "examples/ukazka1.fea": "fea", "moje.fea": "uziv", ".venv/lib/a.py": "v",
             "tools/uv/uv.exe": "u", "scratch/pokus.py": "s", "dist/a.zip": "z"}
    _tree(root, files)
    (root / "fea_core/__pycache__").mkdir()
    (root / "fea_core/__pycache__/stary.cpython-311.pyc").write_bytes(b"pyc")
    old = {"version": "1.0.0", "files": {k: upgrade.bytes_hash(v.encode()) for k, v in files.items()}}
    old["files"]["../mimo.txt"] = upgrade.bytes_hash(b"m")
    (tmp_path / "mimo.txt").write_text("m", encoding="utf-8")
    (root / "fea_core/upraveny.py").write_text("upraveno uživatelem", encoding="utf-8")
    new = {"version": "1.0.1", "files": {"fea_core/novy.py": upgrade.bytes_hash(b"n")}}
    obs = upgrade.obsolete_files(old, new)
    assert "fea_core/novy.py" not in obs
    removed, kept = upgrade.remove_obsolete(root, obs, say=lambda *_: None)
    assert sorted(removed) == ["fea_core/stary.py", "stara_slozka/x.txt"]
    assert kept == ["fea_core/upraveny.py"]
    assert not (root / "fea_core/__pycache__/stary.cpython-311.pyc").exists()
    assert not (root / "stara_slozka").exists()                           # empty folder after the deletion
    for rel in ("fea_core/upraveny.py", "fea_core/novy.py", "examples/ukazka1.fea", "moje.fea", ".venv/lib/a.py",
                "tools/uv/uv.exe", "scratch/pokus.py", "dist/a.zip"):
        assert (root / rel).exists(), rel
    assert (tmp_path / "mimo.txt").exists()                               # never outside the program folder


def test_is_protected():
    for rel in (".venv/x", "tools/a", "scratch/b", "dist/c", ".git/HEAD", "a/b.fea", "../x", "/abs", "C:/x", ""):
        assert upgrade.is_protected(rel), rel
    for rel in ("fea_core/a.py", "docs/x.pdf", "examples/geometry.py"):
        assert not upgrade.is_protected(rel), rel


def test_orphan_bytecode(tmp_path):
    _tree(tmp_path, {"fea_core/a.py": "a"})
    c = tmp_path / "fea_core" / "__pycache__"
    c.mkdir()
    (c / "a.cpython-311.pyc").write_bytes(b"1")
    (c / "smazany.cpython-311.pyc").write_bytes(b"2")
    assert [p.name for p in upgrade.orphan_bytecode(tmp_path)] == ["smazany.cpython-311.pyc"]


# ------------------------------------------------------------------ migrace
def test_pending_range_and_order():
    calls = []
    ms = [migrations.Migration("1.0.3", "post", "c", lambda c: calls.append("c")),
          migrations.Migration("1.0.1", "post", "a2", lambda c: calls.append("a2")),
          migrations.Migration("1.0.1", "pre", "a1", lambda c: calls.append("a1")),
          migrations.Migration("1.0.2", "pre", "b", lambda c: (_ for _ in ()).throw(RuntimeError("chyba")))]
    assert [m.title for m in migrations.pending("1.0.0", "1.0.3", migrations=ms)] == ["a1", "a2", "b", "c"]
    assert [m.title for m in migrations.pending("1.0.1", "1.0.2", migrations=ms)] == ["b"]
    assert migrations.pending("1.0.3", "1.0.3", migrations=ms) == []
    warned = []
    ctx = migrations.Context(root=Path("."), venv=Path("."), vpy=Path("."), modules=["core"], say=lambda *_: None,
                             warn=warned.append)
    done = migrations.run(ctx, "1.0.0", "1.0.3", "pre", migrations=ms)
    assert calls == ["a1"] and done == ["1.0.1: a1"] and len(warned) == 1          # an error of a step does not stop the next ones
    migrations.run(ctx, "1.0.0", "1.0.3", "post", migrations=ms)
    assert calls == ["a1", "a2", "c"]


def test_builtin_migrations_cover_101():
    assert {m.phase for m in migrations.pending("1.0.0", "1.0.1")} == {"pre", "post"}


def test_remove_old_release_files_is_idempotent(tmp_path):
    """Step 1.0.1: files of the releases 0.1.0 / 1.0.0 without a saved manifest (according to installer/manifests) – only the unchanged ones."""
    root = tmp_path
    _tree(root, {"fea_core/contacts.py": "stary modul", "tests/gui_smoke.py": "upraveno", "fea_core/novy.py": "n"})
    (root / "installer" / "manifests").mkdir(parents=True)
    upgrade.write_manifest(root / "installer/manifests/0.1.0.json", {"version": "0.1.0", "files": {
        "fea_core/contacts.py": upgrade.bytes_hash(b"stary modul"), "tests/gui_smoke.py": upgrade.bytes_hash(b"puvodni"),
        "fea_core/novy.py": upgrade.bytes_hash(b"n")}})
    upgrade.write_manifest(root / upgrade.PACKAGE_MANIFEST, {"version": "1.0.1", "files": {
        "fea_core/novy.py": upgrade.bytes_hash(b"n")}})
    ctx = migrations.Context(root=root, venv=root / ".venv", vpy=root / "nic", modules=["core"], say=lambda *_: None)
    for _ in range(2):
        migrations.remove_old_release_files(ctx)
        assert not (root / "fea_core/contacts.py").exists()
        assert (root / "tests/gui_smoke.py").exists() and (root / "fea_core/novy.py").exists()


def test_broken_packages(tmp_path):
    site = tmp_path
    _tree(site, {"PySide6_Essentials-6.8.0.dist-info/RECORD": "PySide6/__init__.py,sha256=x,1\nPySide6/QtCore.pyd,,\n",
                 "numpy-2.1.0.dist-info/RECORD": "numpy/__init__.py,,\n", "numpy/__init__.py": "",
                 "vtk-9.3.0.dist-info/RECORD": "vtkmodules/__init__.py,,\n"})
    assert migrations.broken_packages(site) == ["PySide6_Essentials==6.8.0"]      # vtkmodules is deleted by slim on purpose
    _tree(site, {"PySide6/__init__.py": ""})
    assert migrations.broken_packages(site) == []


# ------------------------------------------------------------------ fea_setup: posloupnost aktualizace
@pytest.fixture
def fs(tmp_path, monkeypatch):
    """fea_setup over a synthetic installation in tmp_path (all heavy steps replaced by a record of the calls)."""
    fs = importlib.import_module("fea_setup")
    root = tmp_path / "app"
    venv = root / ".venv"
    _tree(root, {"fea_core/__init__.py": '__version__ = "1.0.1"\n', "fea_core/novy.py": "n", "fea_core/stary.py": "s",
                 "moje_studie.fea": "uziv"})
    monkeypatch.setattr(fs, "ROOT", root)
    monkeypatch.setattr(fs, "VENV", venv)
    monkeypatch.setattr(fs, "VPY", venv / "Scripts" / "python.exe")
    monkeypatch.setattr(fs, "STATE", venv / "fea_modules.json")
    monkeypatch.setattr(fs, "TOOLS", root / "tools")                      # uninstall --all deletes TOOLS – never the real ones
    monkeypatch.setattr(fs, "DL", root / "tools" / "_download")
    monkeypatch.setattr(fs, "PROBLEMS", [])
    monkeypatch.setattr(fs, "LOG", None)
    calls: list[str] = []
    for name in ("ensure_venv", "write_pth", "vc_runtime_local", "launchers", "ensure_sys_libs", "compile_project",
                 "check_install", "make_shortcuts", "install_fusion_addin"):
        monkeypatch.setattr(fs, name, lambda *a, _n=name, **k: calls.append(_n))
    monkeypatch.setattr(fs, "sync_packages", lambda keys: calls.append("sync:" + " ".join(keys)))
    monkeypatch.setattr(fs, "remove_ext", lambda k: calls.append("remove_ext:" + k))
    monkeypatch.setattr(fs, "fusion_dir", lambda: None)
    monkeypatch.setattr(fs, "site_packages", lambda: venv / "Lib" / "site-packages")
    monkeypatch.setattr(fs.slim, "slim_all", lambda *a, **k: calls.append("slim"))
    monkeypatch.setattr(fs.slim, "prune_mkl", lambda *a, **k: None)
    monkeypatch.setattr(fs, "EXT", {})
    ran: list[str] = []
    ms = [migrations.Migration("1.0.1", "pre", "pre", lambda c: ran.append("pre:" + ",".join(c.modules))),
          migrations.Migration("1.0.1", "post", "post", lambda c: ran.append("post"))]
    monkeypatch.setattr(migrations, "MIGRATIONS", ms)
    upgrade.write_manifest(root / upgrade.PACKAGE_MANIFEST, {"version": "1.0.1", "files": {
        "fea_core/__init__.py": upgrade.bytes_hash(b'__version__ = "1.0.1"\n'), "fea_core/novy.py": upgrade.bytes_hash(b"n")}})
    ns = types.SimpleNamespace(root=root, venv=venv, calls=calls, ran=ran, mod=fs)
    return ns


def _args(**kw):
    a = dict(modules=[], profile=None, exact=False, force=False, no_slim=False, no_shortcuts=True, no_mcp=True,
             no_fusion=True, no_check=True, quick_check=False, keep_downloads=False)
    a.update(kw)
    return types.SimpleNamespace(**a)


def test_update_from_legacy_100(fs):
    """A state without a version (release 1.0.0) → it is taken as 1.0.0: obsolete files according to the manifest 1.0.0, migrations pre/post,
    modules as last time (without a question), the new version and the manifest saved.
    """
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core", "gui"], "platform": "x"}), encoding="utf-8")
    hist = fs.root / "installer" / "manifests"
    upgrade.write_manifest(hist / "1.0.0.json", {"version": "1.0.0", "files": {
        "fea_core/stary.py": upgrade.bytes_hash(b"s"), "fea_core/novy.py": upgrade.bytes_hash(b"n"),
        "moje_studie.fea": upgrade.bytes_hash(b"uziv")}})
    assert fs.mod.installed_version() == "1.0.0"
    assert fs.mod.cmd_update(_args()) == 0
    assert not (fs.root / "fea_core/stary.py").exists()
    assert (fs.root / "fea_core/novy.py").exists() and (fs.root / "moje_studie.fea").exists()
    assert fs.ran == ["pre:core,plots,gui", "post"]
    assert "sync:core plots gui" in fs.calls
    assert fs.calls.index("sync:core plots gui") < fs.calls.index("slim")
    st = json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))
    assert st["version"] == "1.0.1" and st["modules"] == ["core", "plots", "gui"]
    saved = upgrade.read_manifest(fs.venv / upgrade.MANIFEST_NAME)
    assert saved and saved["version"] == "1.0.1"
    # a second run of the same version: no migrations, nothing is deleted
    fs.ran.clear()
    assert fs.mod.cmd_update(_args()) == 0
    assert fs.ran == [] and (fs.root / "fea_core/novy.py").exists()


def test_update_uses_stored_manifest_and_exact_removes_modules(fs):
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core", "gui", "report"], "version": "1.0.0"}),
                                              encoding="utf-8")
    upgrade.write_manifest(fs.venv / upgrade.MANIFEST_NAME, {"version": "1.0.0", "files": {
        "fea_core/stary.py": upgrade.bytes_hash(b"s")}})
    assert fs.mod.cmd_install(_args(modules=["gui"], exact=True)) == 0
    assert not (fs.root / "fea_core/stary.py").exists()
    assert "sync:core plots gui" in fs.calls and "remove_ext:report" in fs.calls


def test_downgrade_needs_force(fs):
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": "9.0.0"}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        fs.mod.cmd_update(_args())
    assert e.value.code == 2 and fs.calls == []                         # nothing has changed
    assert fs.mod.cmd_update(_args(force=True)) == 0
    assert json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["version"] == "1.0.1"


def test_fresh_install_without_state(fs):
    assert fs.mod.installed_version() is None
    assert fs.mod.cmd_install(_args(modules=["core"])) == 0
    assert fs.ran == [] and (fs.root / "fea_core/stary.py").exists()      # without a previous version nothing is deleted
    assert json.loads((fs.venv / "fea_modules.json").read_text(encoding="utf-8"))["version"] == "1.0.1"


def test_install_kind():
    fs = importlib.import_module("fea_setup")
    assert fs.install_kind(None, "1.0.1") == "install"
    assert fs.install_kind("1.0.1", "1.0.1") == "repair"
    assert fs.install_kind("1.0.0", "1.0.1") == "update"
    assert fs.install_kind("1.0.2", "1.0.1") == "downgrade"


def test_repair_install_same_version_message(fs, capsys):
    """The same version again (setup.exe over a finished installation) = "repair installation of version X", not "update X → X"."""
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": "1.0.1"}), encoding="utf-8")
    assert fs.mod.cmd_update(_args()) == 0
    out = capsys.readouterr().out
    assert "1.0.1 → 1.0.1" not in out
    assert fs.mod.T("opravná instalace verze 1.0.1", "repair installation of version 1.0.1") in out
    assert fs.mod.T("Opravná instalace verze 1.0.1 dokončena.", "Repair installation of version 1.0.1 finished.") in out


def test_kept_modified_file_reported_once(fs, capsys, monkeypatch):
    """A modified obsolete file is in the saved manifest and in the manifest of an older release (migration 1.0.1) – it is listed once."""
    fs.venv.mkdir()
    (fs.venv / "fea_modules.json").write_text(json.dumps({"modules": ["core"], "version": "1.0.0"}), encoding="utf-8")
    (fs.root / "fea_core/stary.py").write_text("upraveno uživatelem", encoding="utf-8")
    digest = {"fea_core/stary.py": upgrade.bytes_hash(b"s")}
    upgrade.write_manifest(fs.venv / upgrade.MANIFEST_NAME, {"version": "1.0.0", "files": digest})
    upgrade.write_manifest(fs.root / "installer/manifests/1.0.0.json", {"version": "1.0.0", "files": digest})
    monkeypatch.setattr(migrations, "MIGRATIONS", [migrations.Migration("1.0.1", "pre", "staré soubory",
                                                                        migrations.remove_old_release_files)])
    assert fs.mod.cmd_update(_args()) == 0
    out = capsys.readouterr().out
    assert (fs.root / "fea_core/stary.py").exists()
    assert out.count("fea_core/stary.py") == 1, out
    kept = fs.mod.T("upravené zastaralé soubory ponechány", "modified obsolete files kept")
    assert out.count(kept) == 1 and fs.mod.T("žádné zastaralé soubory", "no obsolete files") not in out


def test_remove_obsolete_reported_set(tmp_path):
    _tree(tmp_path, {"a.py": "upraveno"})
    files = {"a.py": upgrade.bytes_hash(b"puvodni")}
    said, seen = [], set()
    for _ in range(2):
        assert upgrade.remove_obsolete(tmp_path, files, say=said.append, reported=seen) == ([], ["a.py"])
    assert len(said) == 1 and seen == {"a.py"}
    upgrade.remove_obsolete(tmp_path, files, say=said.append)            # without a set it always prints
    assert len(said) == 2


def test_installer_messages_follow_language(tmp_path, monkeypatch):
    """Messages of upgrade / migrations / slim through T(cs, en) (setup_lang) like the rest of the installer."""
    lang = importlib.import_module("setup_lang")
    import slim
    (tmp_path / ("Library/bin" if slim.IS_WIN else "lib")).mkdir(parents=True)
    for cs, start in ((True, "zastaralé soubory programu smazány"), (False, "obsolete program files deleted")):
        monkeypatch.setattr(lang, "CS", cs)
        _tree(tmp_path, {"a.py": "x"})
        said = []
        upgrade.remove_obsolete(tmp_path, {"a.py": upgrade.bytes_hash(b"x")}, say=said.append)
        assert said and said[0].startswith(start), said
        said.clear()
        slim.prune_mkl(tmp_path, say=said.append)
        assert said and said[0].startswith("MKL: " + ("uvolněno" if cs else "freed")), said
    # no Czech text of a message outside T(): say/warn with the literal text with diacritics
    import ast
    czech = set("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")
    for mod in ("upgrade", "migrations", "slim"):
        tree = ast.parse((ROOT / "installer" / f"{mod}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in ("say", "warn", "_say"):
                for arg in node.args[:1]:
                    bare = [n for n in ast.walk(arg) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                            and not any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "T" and n in list(ast.walk(c))
                                        for c in ast.walk(arg))]
                    assert not any(czech & set(n.value) for n in bare), (mod, node.lineno)


# ------------------------------------------------------------------ uv and micromamba only into the program folder
def test_uv_and_mamba_write_only_into_program_folder(monkeypatch, tmp_path):
    fs = importlib.import_module("fea_setup")
    e = fs.uv_env()
    assert e["UV_PYTHON_INSTALL_BIN"] == "0" and e["UV_PYTHON_INSTALL_REGISTRY"] == "0"     # without .local/bin and the registry
    assert Path(e["UV_PYTHON_INSTALL_DIR"]) == fs.TOOLS / "python"
    ps1 = (ROOT / "installer" / "install.ps1").read_text(encoding="utf-8-sig")
    assert '$env:UV_PYTHON_INSTALL_BIN = "0"' in ps1 and '$env:UV_PYTHON_INSTALL_REGISTRY = "0"' in ps1
    assert '"--no-bin", "--no-registry"' in ps1 and "$env:TEMP" not in ps1
    sh = (ROOT / "install.sh").read_text(encoding="utf-8")
    assert "UV_PYTHON_INSTALL_BIN=0" in sh and "UV_CACHE_DIR=" in sh
    monkeypatch.setattr(fs, "DL", tmp_path / "dl")
    m = fs.mamba_env()
    for k in ("MAMBA_ROOT_PREFIX", "CONDA_PKGS_DIRS", "XDG_CACHE_HOME", "HOME") + (("USERPROFILE",) if fs.IS_WIN else ()):
        assert Path(m[k]).is_relative_to(tmp_path / "dl"), k                # cache, environments.txt… into _download
    assert set(fs.MAMBA_DIRS) >= {"mamba-root", "mamba-home", "mamba-cache"}
    fs.mamba_cleanup()
    assert not (tmp_path / "dl" / "mamba-home").exists()


# ------------------------------------------------------------------ uninstall: empty program folders
def _junction(link: Path, target: Path) -> bool:
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
        return True
    except (ImportError, AttributeError, OSError):
        try:
            link.symlink_to(target, target_is_directory=True)
            return True
        except OSError:
            return False


def test_uninstall_removes_empty_program_dirs(fs, monkeypatch, tmp_path):
    """uninstall --pycache: the bytecode and the empty program folders bottom-up; folders with files, .venv / tools (without --all),
    development folders and the targets of links / junctions stay. --all deletes .venv and tools too.
    """
    monkeypatch.setattr(fs.mod, "remove_shortcuts", lambda: None)
    monkeypatch.setattr(fs.mod, "remove_fusion_addin", lambda: None)
    r = fs.root
    _tree(r, {"fea_core/__pycache__/novy.cpython-311.pyc": "b", "stary_balicek/__pycache__/x.cpython-311.pyc": "b",
              "examples/moje.fea": "uziv", "examples/ukazka1.fea.bak": "zaloha", ".venv/lib/a.py": "v", "tools/uv/uv.exe": "u",
              "scratch/pokus.py": "s"})
    for d in ("stara_slozka/prazdna/hloubeji", "docs/prazdna", ".venv/prazdna", "tools/prazdna", ".git/refs/prazdna",
              "scratch/prazdna"):
        (r / d).mkdir(parents=True)
    outside = tmp_path / "mimo" / "prazdna"
    outside.mkdir(parents=True)
    linked = _junction(r / "odkaz", tmp_path / "mimo")
    args = types.SimpleNamespace(pycache=True, all=False)
    assert fs.mod.cmd_uninstall(args) == 0
    for gone in ("stara_slozka", "stary_balicek", "docs", "fea_core/__pycache__"):
        assert not (r / gone).exists(), gone
    for kept in ("fea_core/novy.py", "examples/moje.fea", "examples/ukazka1.fea.bak", ".venv/prazdna", "tools/prazdna",
                 ".git/refs/prazdna", "scratch/prazdna"):
        assert (r / kept).exists(), kept
    assert outside.is_dir()                                                # a junction is not entered
    if linked:
        assert os.path.lexists(r / "odkaz")
    assert fs.mod.cmd_uninstall(types.SimpleNamespace(pycache=False, all=True)) == 0
    assert not (r / ".venv").exists() and not (r / "tools").exists()
    assert (r / "examples/moje.fea").exists() and (r / ".git/refs/prazdna").exists() and outside.is_dir()


# ------------------------------------------------------------------ packages: .run and setup.exe
def test_run_header_and_payload(tmp_path):
    mp = importlib.import_module("make_package")
    tgz = tmp_path / "p.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        data = b"hello"
        ti = tarfile.TarInfo("FEAStudio-9.9.9/a.txt")
        ti.size = len(data)
        t.addfile(ti, io.BytesIO(data))
    run = mp.write_run(tgz, tmp_path / "x.run", "9.9.9")
    raw = run.read_bytes()
    marker = b"\n__FEA_STUDIO_PAYLOAD__\n"
    head, payload = raw.split(marker, 1)
    head.decode("ascii")                                                  # the header purely ASCII, LF
    assert b"\r" not in head and head.startswith(b"#!/bin/sh\n")
    assert payload == tgz.read_bytes()
    assert mp.sha(tgz).encode() in head and b'VERSION="9.9.9"' in head
    assert gzip.decompress(payload)


def test_normalize_eol_follows_gitattributes(tmp_path):
    """Line ends in the package: .ps1 / .bat CRLF (PowerShell 5.1, cmd), .sh and executables LF, the others unchanged; the extensions
    with eol=crlf / eol=lf in .gitattributes must be known to make_package.
    """
    mp = importlib.import_module("make_package")
    st = tmp_path / "stage"
    files = {"installer/install.ps1": b"\xef\xbb\xbfa\nb\r\nc\n", "x.bat": b"a\nb\r\n", "install.sh": b"a\r\nb\r\n",
             "installer/fea_setup.py": b"#!/usr/bin/env python3\r\nx\r\n", "pkg/a.py": b"a\r\nb\n", "fea-studio": b"#!/bin/sh\r\n"}
    for rel, data in files.items():
        (st / rel).parent.mkdir(parents=True, exist_ok=True)
        (st / rel).write_bytes(data)
    mp.normalize_eol(st)
    assert (st / "installer/install.ps1").read_bytes() == b"\xef\xbb\xbfa\r\nb\r\nc\r\n"     # the BOM stays
    assert (st / "x.bat").read_bytes() == b"a\r\nb\r\n"
    assert (st / "install.sh").read_bytes() == b"a\nb\n"
    assert (st / "installer/fea_setup.py").read_bytes() == b"#!/usr/bin/env python3\nx\n"
    assert (st / "fea-studio").read_bytes() == b"#!/bin/sh\n"
    assert (st / "pkg/a.py").read_bytes() == b"a\r\nb\n"
    for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\S+)\s.*\beol=(crlf|lf)\b", line)
        if not m:
            continue
        pat, eol = m.groups()
        if pat.startswith("*."):
            assert pat[1:] in (mp.CRLF if eol == "crlf" else mp.LF), line
        else:
            assert eol == "lf" and pat in mp.EXEC, line


def test_sha256sums_have_lf(tmp_path):
    """SHA256SUMS.txt for `sha256sum -c`: LF even on Windows, the format "digest␣␣name"."""
    mp = importlib.import_module("make_package")
    a, b = tmp_path / "a.zip", tmp_path / "b c.tar.gz"
    a.write_bytes(b"A")
    b.write_bytes(b"B")
    raw = mp.write_sums(tmp_path / "SHA256SUMS.txt", [a, b]).read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode("utf-8").splitlines()
    assert lines == [f"{mp.sha(a)}  a.zip", f"{mp.sha(b)}  b c.tar.gz"]
    assert all(re.fullmatch(r"[0-9a-f]{64}  \S.*", l) for l in lines)


def test_iss_consistent_with_modules():
    iss = (ROOT / "installer" / "FEAStudio.iss").read_text(encoding="utf-8-sig")
    assert "AppId={{6F1E7C2A-6E0B-4B7B-9C55-FEA5D1000001}" in iss            # the same AppId = update over the old one
    assert "{6F1E7C2A-6E0B-4B7B-9C55-FEA5D1000001}_is1" in iss
    assert "PrivilegesRequired=lowest" in iss and r"{localappdata}\Programs\FEA Studio" in iss
    mods = json.loads((ROOT / "fea_core" / "modules.json").read_text(encoding="utf-8"))["modules"]
    listed = re.search(r"MODULES = '([^']+)'", iss).group(1).split()
    comps = re.findall(r'^Name: "(\w+)"; Description: "[^"]*"; Types:', iss.split("[Components]")[1].split("[")[0], re.M)
    optional = {k for k, m in mods.items() if k not in ("core", "docs")}
    assert set(listed) == optional and set(comps) == optional | {"core"}
    # examples: a modified one is backed up before being overwritten (not onlyifdoesntexist – new versions of the examples would not be updated);
    # uninstall: empty program folders (also from the zip) – verified with a test setup.exe (scratch/fix_install/iss_test.py)
    files = iss.split("[Files]")[1].split("\n[")[0]
    assert r'Excludes: "\examples\*.fea,\tools\python\*"' in files and "BeforeInstall: BackupModifiedExample" in files
    # onlyifdoesntexist only for the bundled Python (the same-named folder = the same content; during an update it is held by
    # the running MCP server / GUI) – not for the examples
    only = [ln for ln in files.splitlines() if "onlyifdoesntexist" in ln]
    assert len(only) == 1 and r'DestDir: "{app}\tools\python"' in files.split("onlyifdoesntexist")[0].splitlines()[-2]
    assert 'DestName: "fea_new_manifest.json"; Flags: dontcopy' in files
    assert "function RunningFromDir" in iss and "CustomMessage('Running')" in iss
    assert "usPostUninstall" in iss and "RemoveEmptyDirs(ExpandConstant('{app}'))" in iss
    assert "skipifdoesntexist" in iss.split("[UninstallRun]")[1].split("\n[")[0]
    assert not (ROOT / "installer" / "uninstall.ps1").exists()          # obsolete – uninstall only through fea_setup.py
