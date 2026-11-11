"""Robustness of the installer on foreign computers (1.3.2): shortcuts without PowerShell, certificates / proxy for uv, a clean environment
of Python, offline packages, repeated uv pip, setup.exe without PowerShell. Without network, uv and a real installation.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer"))

fea_setup = importlib.import_module("fea_setup")
win = pytest.mark.skipif(sys.platform != "win32", reason="jen Windows")


@win
def test_winlink_roundtrip(tmp_path):
    winlink = importlib.import_module("winlink")
    lnk = tmp_path / "Zkouška ř ñ.lnk"
    target = Path(sys.executable)
    winlink.create(str(lnk), str(target), '-E -m fea_gui "a b"', str(tmp_path), "popis", "")
    got = winlink.read(str(lnk))
    assert Path(got["target"]) == target
    assert got["args"] == '-E -m fea_gui "a b"'
    assert Path(got["workdir"]) == tmp_path
    for name in winlink.FOLDERS:
        assert Path(winlink.known_folder(name)).is_absolute()


@win
def test_make_shortcuts_without_powershell(tmp_path, monkeypatch):
    """Shortcuts through IShellLinkW into (planted) known folders; PowerShell is not called, the registry and install.json are not changed."""
    winlink = importlib.import_module("winlink")
    folders = {k: tmp_path / k for k in winlink.FOLDERS}
    monkeypatch.setattr(winlink, "known_folder", lambda name: str(folders[name]))
    calls = []
    monkeypatch.setattr(fea_setup, "run", lambda cmd, **kw: calls.append(cmd) or types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(fea_setup, "write_locator", lambda: None)
    reg = []
    fake = types.SimpleNamespace(HKEY_CURRENT_USER=0, REG_SZ=1, CreateKey=lambda *a: _Key(), SetValueEx=lambda *a: reg.append(a))
    monkeypatch.setitem(sys.modules, "winreg", fake)
    fea_setup.make_shortcuts()
    assert not calls, calls
    desk = winlink.read(str(folders["Desktop"] / "FEA Studio.lnk"))
    assert desk["args"] == "-P -E -m fea_gui" and Path(desk["workdir"]) == fea_setup.ROOT
    assert (folders["Programs"] / "FEA Studio" / "Moduly FEA Studia.lnk").is_file()
    assert (folders["SendTo"] / "FEA Studio.lnk").is_file()
    assert any('-E -m fea_gui "%1"' in str(a[-1]) for a in reg)
    # uninstallation: only the shortcuts of this installation (foreign ones stay)
    winlink.create(str(folders["Desktop"] / "cizí.lnk"), sys.executable, "", str(tmp_path), "", "")
    monkeypatch.setattr(fea_setup, "remove_locator", lambda: None)
    fake.OpenKey = lambda *a: (_ for _ in ()).throw(OSError())
    fea_setup.remove_shortcuts()
    assert not (folders["Desktop"] / "FEA Studio.lnk").exists()
    assert not (folders["Programs"] / "FEA Studio").exists()
    assert (folders["Desktop"] / "cizí.lnk").exists()
    assert not calls


class _Key:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_uv_env_certs_and_proxy(monkeypatch):
    for k in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "UV_SYSTEM_CERTS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(fea_setup.urllib.request, "getproxies",
                        lambda: {"https": "http://proxy.firma:8080", "http": "http://proxy.firma:8080", "no": "localhost"})
    env = fea_setup.uv_env()
    if sys.platform == "win32":
        assert env["UV_SYSTEM_CERTS"] == "1"
    assert env["HTTPS_PROXY"] == "http://proxy.firma:8080" and env["NO_PROXY"] == "localhost"
    assert int(env["UV_HTTP_TIMEOUT"]) >= 120
    monkeypatch.setenv("HTTPS_PROXY", "http://vlastni:3128")         # the user's own proxy has priority
    assert "HTTPS_PROXY" not in fea_setup.proxy_env()
    monkeypatch.setattr(fea_setup.urllib.request, "getproxies", lambda: 1 / 0)
    monkeypatch.delenv("HTTPS_PROXY")
    assert fea_setup.proxy_env() == {}                                # a faulty proxy record does not stop the installation


def test_installer_ignores_foreign_python_env(tmp_path):
    env = dict(os.environ, PYTHONHOME=str(tmp_path / "nic"), PYTHONPATH=str(tmp_path / "nic"))
    code = ("import sys, os; sys.path.insert(0, r'%s'); import fea_setup; "
            "print(os.environ.get('PYTHONHOME'), os.environ.get('PYTHONPATH'))" % (ROOT / "installer"))
    r = subprocess.run([sys.executable, "-E", "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.split() == ["None", "None"], r.stderr


def test_uv_pip_retries_then_explains(monkeypatch):
    seen = []

    def fake_uv(*args, env=None, check=True, **kw):
        seen.append(env or {})
        return types.SimpleNamespace(returncode=1 if len(seen) < 3 else 0)
    monkeypatch.setattr(fea_setup, "uv", fake_uv)
    monkeypatch.setattr(fea_setup.time, "sleep", lambda s: None)
    fea_setup.uv_pip("sync", "x.txt")
    assert len(seen) == 3 and seen[0] == {} and seen[1]["UV_CONCURRENT_INSTALLS"] == "1"
    seen.clear()
    monkeypatch.setattr(fea_setup, "uv", lambda *a, **k: seen.append(1) or types.SimpleNamespace(returncode=2))
    with pytest.raises(RuntimeError, match="antivir|antivirus"):
        fea_setup.uv_pip("sync", "x.txt")
    assert len(seen) == 3


def test_offline_wheels_detection(tmp_path, monkeypatch):
    w = tmp_path / "offline" / "wheels"
    monkeypatch.setattr(fea_setup, "OFFLINE", w)
    assert not fea_setup.offline_wheels()
    w.mkdir(parents=True)
    (w / "numpy-1.26.4-cp311-cp311-win_amd64.whl").write_bytes(b"")
    assert fea_setup.offline_wheels()


def test_setup_exe_runs_python_not_powershell():
    iss = (ROOT / "installer" / "FEAStudio.iss").read_text(encoding="utf-8-sig")
    post = iss[iss.index("procedure CurStepChanged"):iss.index("function GetCustomSetupExitCode")]
    assert "tools\\python\\{#PyDir}\\python.exe" in post and "-E -s" in post
    assert post.index("FileExists(Exe)") < post.index("powershell.exe")          # PowerShell only as a fallback
    assert "MinVersion=10.0" in iss and "ExtraDiskSpaceRequired" in iss and "OutName" in iss
    assert f'"{fea_setup.PY_DIR["windows-x86_64"]}"' in iss                      # default PyDir = fea_setup.PY_DIR
    bat = (ROOT / "Install FEA Studio.bat").read_text(encoding="utf-8")
    assert "fea_setup.py" in bat and bat.index("fea_setup.py") < bat.index("install.ps1")
    assert "-E -s" in bat and 'set "PYTHONHOME="' in bat
    gui = (ROOT / "FEA Studio.bat").read_text(encoding="utf-8")
    assert "-E -m fea_gui" in gui      # the .bat changes to the folder of the program – -P is not needed
