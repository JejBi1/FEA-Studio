"""Program modules, the installer and the platform layer (without network, without Gmsh and CalculiX)."""
from __future__ import annotations

import importlib
import re
import sys
import tomllib
import types
from pathlib import Path

import pytest

from fea_core import config, modules

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(lines):
    out = {}
    for l in lines:
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)", l.strip())
        if m:
            out[_norm(m.group(1))] = m.group(2)
    return out


# ------------------------------------------------------------------ module definitions
def test_modules_json_consistent():
    mods = modules.all_modules()
    assert "core" in mods and mods["core"].required
    for k, m in mods.items():
        assert m.title("cs") and m.title("en") and m.desc("cs") and m.desc("en"), k
        for r in m.requires:
            assert r in mods, f"{k} vyžaduje neznámý modul {r}"
        assert m.size() > 0, k
    for name, keys in modules.profiles().items():
        assert set(keys) <= set(mods), name
        assert modules.closure(keys)[0] == "core"


def test_closure_order_and_unknown():
    c = modules.closure(["hypre"])
    assert c.index("core") < c.index("pardiso") < c.index("ngsolve") < c.index("hypre")
    with pytest.raises(KeyError):
        modules.closure(["neexistuje"])


@pytest.mark.parametrize("plat", ["windows-x86_64", "linux-x86_64"])
def test_locks_cover_all_pins(plat):
    """Every package from modules.json has the same version in the lock of the platform (otherwise the installer would install other versions)."""
    lock = _pins((ROOT / "installer" / "locks" / f"{plat}.txt").read_text(encoding="utf-8").splitlines())
    assert lock["mkl"] == "2023.2.0", "MKL 2024.2 + PARDISO dává tiše chybné výsledky"
    for k, m in modules.all_modules().items():
        for k2, v in _pins(list(m.pip) + list(m.pip_nodeps)).items():
            assert lock.get(k2) == v, f"{plat}: {k2}=={v} (modul {k}) nesedí se zámkem ({lock.get(k2)})"


def test_pyproject_extras_match_modules():
    d = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = d["project"]["optional-dependencies"]
    for k, m in modules.all_modules().items():
        if k == "core":
            continue
        assert _pins(extras[k]) == _pins(list(m.pip) + list(m.pip_nodeps)), k


def test_imports_of_core_are_light():
    """The core must be importable without numba, matplotlib, PySide6, mcp, docx (optional modules)."""
    import subprocess
    code = ("import sys\n"
            "for n in ('numba', 'PySide6', 'mcp', 'docx', 'fastapi', 'matplotlib'):\n"
            "    sys.modules[n] = None\n"      # the import fails → ImportError
            "import fea_core.solve, fea_core.cli, fea_core.study, fea_core.thermal.run, fea_core.shell.run\n"
            "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


def test_need_raises_module_missing():
    with pytest.raises(modules.ModuleMissing) as e:
        modules.need("fea_neexistujici_balicek", "accel")
    assert e.value.key == "accel" and "accel" in str(e.value)
    assert isinstance(e.value, ImportError)


def test_numba_users_raise_friendly_error(monkeypatch):
    """Without numba the fatigue / print / rasteriser report ModuleMissing (not a bare ImportError without instructions)."""
    import subprocess
    code = ("import sys\nsys.modules['numba'] = None\n"
            "import fea_core.modules as m\n"
            "try:\n    import fea_core.fatigue.kernel\nexcept m.ModuleMissing as e:\n    print('MISSING', e.key)\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert "MISSING accel" in r.stdout, r.stdout + r.stderr[-500:]


def test_status_core_available():
    assert modules.available("core")
    st = modules.status()
    assert set(st) == set(modules.all_modules())


# ------------------------------------------------------------------ platform layer
def test_solver_name(monkeypatch):
    monkeypatch.setattr(config, "CCX_PARDISO", True)
    assert config.solver_name("PARDISO") == "PARDISO"
    monkeypatch.setattr(config, "CCX_PARDISO", False)
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_static.exe"))
    assert config.solver_name("PARDISO") == "PASTIX"
    monkeypatch.setattr(config, "CCX_EXE", Path("x/ccx_spooles"))
    assert config.solver_name("PARDISO") == "SPOOLES"
    assert config.solver_name("SPOOLES") == "SPOOLES"        # the explicit choice does not change


def test_find_tool_depth(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "ccx_static").write_text("x")
    assert config._find_tool(("ccx_static",), tmp_path) == tmp_path / "a" / "ccx_static"
    assert config._find_tool(("nic",), tmp_path) is None
    assert config._find_tool(("x",), tmp_path / "neexistuje") is None


def test_find_ccx_env_override(monkeypatch):
    monkeypatch.setenv("FEA_CCX", "/opt/ccx_i4")
    p, pardiso = config.find_ccx()
    assert p == Path("/opt/ccx_i4") and pardiso
    monkeypatch.setenv("FEA_CCX", "/opt/ccx_2.23")
    assert config.find_ccx() == (Path("/opt/ccx_2.23"), False)
    monkeypatch.setenv("FEA_CCX_PARDISO", "1")
    assert config.find_ccx()[1]


def test_user_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "IS_WIN", False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert config.user_config_dir() == tmp_path / "cfg" / "FEAStudio"
    assert config.user_data_dir() == tmp_path / "data" / "FEAStudio"
    monkeypatch.setattr(config, "IS_WIN", True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert config.user_config_dir() == tmp_path / "roaming" / "FEAStudio"
    assert config.user_data_dir() == tmp_path / "local" / "FEAStudio"


def test_threads_env(monkeypatch):
    monkeypatch.setenv("FEA_THREADS", "3")
    assert config._physical_cores() == 3
    monkeypatch.delenv("FEA_THREADS")
    assert 1 <= config._physical_cores() <= (config.os.cpu_count() or 1)


def test_child_env_adds_sys_libs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "IS_WIN", False)
    sysl = tmp_path / "lib"
    sysl.mkdir()
    monkeypatch.setattr(config, "SYS_LIBS", sysl)
    e = config.child_env({"LD_LIBRARY_PATH": "/x"})
    assert e["LD_LIBRARY_PATH"].split(config.os.pathsep) == [str(sysl), "/x"]


def test_mcp_clients_linux_paths(monkeypatch, tmp_path):
    from fea_core.mcp import setup
    monkeypatch.setattr(setup, "os", types.SimpleNamespace(name="posix", environ={}))   # not os.name globally (it would break pathlib)
    cl = {c.key: c for c in setup.clients({"HOME": str(tmp_path)})}
    assert cl["vscode"].path == tmp_path / ".config" / "Code" / "User" / "mcp.json"
    assert cl["cursor"].path == tmp_path / ".cursor" / "mcp.json"
    assert "claude-desktop-store" not in cl


# ------------------------------------------------------------------ installer and pruning
def test_slim_prune_mkl_keeps_runtime(tmp_path, monkeypatch):
    import slim
    monkeypatch.setattr(slim, "IS_WIN", False)
    lib = tmp_path / "lib"
    lib.mkdir()
    keep = ["libmkl_rt.so.2", "libmkl_core.so.2", "libmkl_intel_thread.so.2", "libmkl_gnu_thread.so.2",
            "libmkl_def.so.2", "libmkl_avx2.so.2", "libmkl_avx512.so.2", "libmkl_vml_def.so.2", "libmkl_intel_lp64.so.2",
            "libiomp5.so", "mkl_msg.cat"]
    drop = ["libmkl_pgi_thread.so.2", "libmkl_tbb_thread.so.2", "libmkl_scalapack_lp64.so.2", "libmkl_blacs_lp64.so.2",
            "libmkl_avx.so.2", "libmkl_mc.so.2", "libmkl_mc3.so.2", "libmkl_vml_avx.so.2", "libmkl_cdft_core.so.2",
            "libmkl_sequential.so.2"]
    for n in keep + drop:
        (lib / n).write_bytes(b"x" * 10)
    slim.prune_mkl(tmp_path, say=lambda *_: None)
    assert sorted(p.name for p in lib.iterdir()) == sorted(keep)


def test_slim_prune_python_refuses_foreign(tmp_path):
    import slim
    venv = tmp_path / ".venv"
    venv.mkdir()
    other = tmp_path / "system_python"
    (other / "Lib" / "test").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {other}\n", encoding="utf-8")
    assert slim.prune_python(venv, say=lambda *_: None) == 0
    assert (other / "Lib" / "test").exists()                 # a foreign Python is never deleted
    ours = tmp_path / "tools" / "python" / "cpython-3.11-x" / "bin"
    (ours.parent / "lib" / "python3.11" / "test").mkdir(parents=True)
    (ours.parent / "lib" / "python3.11" / "os.py").write_text("x")
    ours.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = {ours}\n", encoding="utf-8")
    slim.prune_python(venv, say=lambda *_: None)
    assert not (ours.parent / "lib" / "python3.11" / "test").exists()
    assert (ours.parent / "lib" / "python3.11" / "os.py").exists()


def test_fea_setup_profiles_and_quiet_flags(monkeypatch):
    fs = importlib.import_module("fea_setup")
    ns = type("A", (), {"modules": [], "profile": "standard"})()
    keys = fs.pick_modules(ns)
    assert keys[0] == "core" and "gui" in keys and "pardiso" in keys and "mcp" not in keys
    ns2 = type("A", (), {"modules": ["hypre"], "profile": None})()
    assert {"core", "pardiso", "ngsolve", "hypre"} <= set(fs.pick_modules(ns2))


def test_make_package_file_list_excludes():
    mp = importlib.import_module("make_package")
    files = mp.file_list(None)
    assert "fea_core/modules.json" in files and "installer/fea_setup.py" in files and "install.sh" in files
    assert not any(f.startswith(("scratch/", "tools/", ".venv/", "dist/")) for f in files)
    assert not any(f.endswith(".fea") for f in files)
    assert mp.version().count(".") == 2


def test_make_package_linux_ccx_neutral(tmp_path):
    """The bundled CalculiX for Linux: the owner of the building user is repacked to 0:0 without names, a path from a home folder = an error."""
    import io
    import tarfile
    mp = importlib.import_module("make_package")

    def archive(path, payload):
        with tarfile.open(path, "w:gz") as t:
            for name, data, mode in (("ccx_spooles", payload, 0o755), ("lib/libgomp.so.1", b"/home/conda/x/", 0o755)):
                ti = tarfile.TarInfo(name)
                ti.size, ti.mode, ti.mtime = len(data), mode, 1_700_000_000
                ti.uid = ti.gid = 1000
                ti.uname = ti.gname = "builder"
                t.addfile(ti, io.BytesIO(data))

    archive(tmp_path / "ok.tar.gz", b"\x7fELF ... /build/arpack-src/UTIL/dvout.f")
    mp.linux_ccx(tmp_path / "ok.tar.gz", tmp_path / "out.tar.gz")
    with tarfile.open(tmp_path / "out.tar.gz") as t:
        ms = t.getmembers()
        assert [m.name for m in ms] == ["ccx_spooles", "lib/libgomp.so.1"]
        assert all((m.uid, m.gid, m.uname, m.gname) == (0, 0, "", "") for m in ms)
        assert all(m.mode == 0o755 and m.mtime == 1_700_000_000 for m in ms)
        assert t.extractfile("ccx_spooles").read().endswith(b"dvout.f")
    assert (tmp_path / "out.tar.gz").read_bytes()[3] == 0              # gzip without the file name
    archive(tmp_path / "bad.tar.gz", b"\x7fELF ... /home/builder/fea/ccx_build/arpack-src/UTIL/dvout.f")
    with pytest.raises(SystemExit, match="home folder"):
        mp.linux_ccx(tmp_path / "bad.tar.gz", tmp_path / "out2.tar.gz")


def test_verification_imports_without_numba():
    """The check calculation of the installation (fea_core.warmup → verification.run_all) must also run without the accel module (Numba)."""
    import subprocess
    import sys
    code = ("import sys; sys.modules['numba'] = None\n"
            "import verification.run_all as r, verification.cases as C\n"
            "assert 'V1' in C.CASES\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    assert p.returncode == 0, p.stderr[-2000:]
