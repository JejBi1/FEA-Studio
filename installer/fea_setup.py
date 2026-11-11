#!/usr/bin/env python3
"""Installer of FEA Studio (Windows and Linux) – the modules are installed separately, so that the basic FEM does not take unnecessary space.

    python installer/fea_setup.py install [module …] [--profile minimal|standard|full] [--exact]
    python installer/fea_setup.py update  [--force]   (after unpacking a new version: migrations, obsolete files, module resync)
    python installer/fea_setup.py remove  module …
    python installer/fea_setup.py list | doctor | slim | uninstall
    python installer/fea_setup.py fusion [--remove]     (the "Send to FEA Studio" add-in for Autodesk Fusion)
    python installer/fea_setup.py mpi                   (only Microsoft MPI for the NGSolve + hypre solver; UAC, administrator)

Runs through Python 3.9+ (only the standard library is needed); if needed it downloads `uv` (one executable file),
which downloads Python 3.11 to `tools/python` and creates `.venv`. Everything stays in the program folder (nothing is installed
into the system; exceptions: shortcuts, the .fea extension and optionally MS-MPI). Module definitions: fea_core/modules.json,
version lock: installer/locks/<platform>.txt (verified sets; mkl MUST stay 2023.2.0).
Answers to questions: --yes (default choices), --quiet (no questions). MS-MPI (Windows, the hypre module) is installed without a question
only with the explicit consent --msmpi (setup.exe with the hypre component, the Program modules window asks).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup_lang import CS, T  # noqa: E402  (installer/setup_lang.py – language of the output)
import slim  # noqa: E402  (installer/slim.py – pruning of packages)
import upgrade  # noqa: E402  (installer/upgrade.py – version and manifest of the program files)
import migrations  # noqa: E402  (installer/migrations.py – kroky aktualizace)

IS_WIN = sys.platform == "win32"
EXE = ".exe" if IS_WIN else ""
MACHINE = platform.machine().lower()
PLAT = ("windows" if IS_WIN else "linux") + "-" + ("x86_64" if MACHINE in ("amd64", "x86_64") else MACHINE)
VENV = ROOT / ".venv"
VPY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
TOOLS = ROOT / "tools"
DL = TOOLS / "_download"
UV = TOOLS / "uv" / ("uv.exe" if IS_WIN else "uv")
STATE = VENV / "fea_modules.json"
PY_VERSION = "3.11"

# Downloaded programs: pinned versions and SHA-256 digests (verified 17. 10. 2026; uv and micromamba from the .sha256 of the release
# on GitHub, ccx_i4.exe in the CalculiX zip = the binary the program was verified with, MS-MPI with a valid Microsoft signature).
# The packages carry only ccx_static.exe from the CalculiX zip; ccx_i4.exe (CalculiX + proprietary Intel MKL PARDISO – GPL-2.0
# does not allow us to redistribute it) is extracted from the ORIGINAL zip of dhondt.de on the user's machine with the pardiso
# module (Linux: built locally by installer/linux/build_ccx.sh against the mkl wheel).
# New file = new version in the URL and a new digest (otherwise the installation fails with "digest does not match").
UV_VERSION = "0.13.1"
UV_URL = {
    "windows-x86_64": f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip",
    "linux-x86_64": f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz",
}
UV_SHA256 = {"windows-x86_64": "3b2661a463a5baea7f5076ea2b25ab71d77f36a6c0d809e5350651792f80a672",
             "linux-x86_64": "eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220"}
# Python for the case that `uv python install` fails while extracting (an antivirus on a test PC: "Failed to extract archive … os error 3"
# in tools/python/.temp, 20. 10. 2026) – the same build that uv 0.13.1 downloads; digests from the SHA256SUMS of release 20261013
PY_FULL = "3.11.17"
PBS_RELEASE = "20261013"
_PBS = f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_RELEASE}/cpython-{PY_FULL}%2B{PBS_RELEASE}-"
PY_URL = {"windows-x86_64": _PBS + "x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
          "linux-x86_64": _PBS + "x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"}
PY_SHA256 = {"windows-x86_64": "54a38c5e7cc1a5fd2b0182b495f29606dc1455d18ebfd3b5bf0a3c09253b5031",
             "linux-x86_64": "3b8e5a248bcc351265ffe68f5e457668a2e09c99a47f81f658988c7d3d107522"}
# folder name by which uv recognises the managed Python in UV_PYTHON_INSTALL_DIR
PY_DIR = {"windows-x86_64": f"cpython-{PY_FULL}-windows-x86_64-none", "linux-x86_64": f"cpython-{PY_FULL}-linux-x86_64-gnu"}
CCX_WIN_URL = "https://www.dhondt.de/calculix_2.23_4win.zip"
CCX_WIN_SHA256 = "87fc4e1721df37370ec67d0a244bb14f9bf804b0a95df49e2ae7369d30ec72e8"
MAMBA_VERSION = "2.9.0-0"
MAMBA_URL = {p: f"https://github.com/mamba-org/micromamba-releases/releases/download/{MAMBA_VERSION}/micromamba-{a}.tar.bz2"
             for p, a in (("windows-x86_64", "win-64"), ("linux-x86_64", "linux-64"))}
# digests of the .tar.bz2 ARCHIVES (the publisher's .tar.bz2.sha256 carries the digest of the binary inside: win a6d80439…, linux 366cd9cd… –
# verified that the binaries in these archives have exactly those digests)
MAMBA_SHA256 = {"windows-x86_64": "97a336f4ab794bd96a6a4da5e6ed63e75a1d31830414a182419b23d3b36f3fe0",
                "linux-x86_64": "8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd"}
MSMPI_URL = "https://download.microsoft.com/download/7/2/7/72731ebb-b63c-4170-ade7-836966263a8f/msmpisetup.exe"
MSMPI_SHA256 = "47443829114d8d8670f77af98939fe876d33eceb35d0ce4e0e85efeec4d87213"

QUIET = False
YES = False
MSMPI = False                       # --msmpi: consent to install Microsoft MPI (UAC) even without a question (--quiet, setup.exe, GUI)
PROBLEMS: list[str] = []
LOG: Path | None = None             # install.log in the program folder (only the installer's output, not the output of uv / pip)


def _log(text: str) -> None:
    if LOG is not None:
        try:
            with open(LOG, "a", encoding="utf-8") as f:
                print(text, file=f)
        except OSError:
            pass


# ------------------------------------------------------------------ output (language: setup_lang.CS / T)
def step(text: str) -> None:
    print(f"\n== {text}", flush=True)
    _log(f"== {text}")


def ok(text: str) -> None:
    print(f"   OK  {text}", flush=True)
    _log(f"   OK  {text}")


def info(text: str) -> None:
    print(f"       {text}", flush=True)
    _log(f"       {text}")


def warn(text: str) -> None:
    print(f"   !   {text}", flush=True)
    _log(f"   !   {text}")
    PROBLEMS.append(text)


def ask(text: str, default: bool = True) -> bool:
    if QUIET or YES or not sys.stdin or not sys.stdin.isatty():
        return default
    a = input(f"   ?   {text} [{'A/n' if CS else 'Y/n'}] " if default else f"   ?   {text} [{'a/N' if CS else 'y/N'}] ")
    if not a.strip():
        return default
    return a.strip().lower() in ("a", "ano", "y", "yes")


def die(text: str, code: int = 1):
    print(f"\nCHYBA: {text}", file=sys.stderr, flush=True)
    _log(f"CHYBA: {text}")
    sys.exit(code)


# ------------------------------------------------------------------ moduly
def load_modules() -> tuple[dict, dict]:
    raw = json.loads((ROOT / "fea_core" / "modules.json").read_text(encoding="utf-8"))
    return raw["modules"], raw["profiles"]


MODS, PROFILES = load_modules()


def mtitle(k: str) -> str:
    return MODS[k]["title"]["cs" if CS else "en"]


def size_of(k: str) -> int:
    return int(MODS[k].get("size_mb", {}).get("win" if IS_WIN else "linux", 0))


def closure(keys: list[str]) -> list[str]:
    out: list[str] = []

    def add(k: str) -> None:
        if k not in MODS:
            die(T(f"Neznámý modul: {k} (moduly: {', '.join(MODS)})", f"Unknown module: {k} (modules: {', '.join(MODS)})"))
        for r in MODS[k].get("requires", []):
            add(r)
        if k not in out:
            out.append(k)

    for k in keys:
        add(k)
    return out


def read_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"modules": []}


def write_state(mods: list[str], version: str | None = None) -> None:
    """Installation state: modules, program version (for the update; a state without a version = release 1.0.0), platform, time."""
    old = read_state()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    ver = version or old.get("version") or (upgrade.LEGACY_VERSION if old.get("modules") else upgrade.program_version(ROOT))
    STATE.write_text(json.dumps({"modules": mods, "version": ver, "platform": PLAT,
                                 "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, what: str = "", sha256: str | None = None) -> Path:
    """Downloads a file (3 attempts); with `sha256` the digest is verified – also for a file downloaded earlier (otherwise deleted and an error)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if sha256 is None or sha256_file(dest) == sha256:
            return dest
        dest.unlink()                                   # older / damaged copy – download again
    info(T(f"stahuji {what or url}", f"downloading {what or url}"))
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FEAStudio-installer"})
            with urllib.request.urlopen(req, timeout=60) as r, open(str(dest) + ".part", "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if sha256 is not None:
                got = sha256_file(Path(str(dest) + ".part"))
                if got != sha256:
                    os.remove(str(dest) + ".part")
                    raise RuntimeError(T(f"Otisk SHA-256 staženého souboru nesouhlasí ({got} ≠ {sha256}) – soubor "
                                         "na serveru se změnil nebo byl podvržen; nic se nespustí.",
                                         f"SHA-256 of the downloaded file does not match ({got} ≠ {sha256}) – the file "
                                         "on the server changed or was tampered with; nothing will be run."))
            os.replace(str(dest) + ".part", dest)
            return dest
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(T(f"Stažení selhalo ({url}): {last}", f"Download failed ({url}): {last}"))


# Variables of another Python (GIS, CAD, an older Python installation set them for the whole system) would
# break .venv and the bundled Python ("Failed to import encodings", foreign packages on sys.path) – the installer and all its child processes without them
for _k in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV", "CONDA_PREFIX"):
    os.environ.pop(_k, None)
os.environ["PYTHONNOUSERSITE"] = "1"

# child Pythons (the .venv environment, checks) write UTF-8 to the pipe – it is read back the same way (encoding="utf-8" with text=True);
# without it an installation path with diacritics ("…\\FEA Studio Ř") would break the installation in a cp1250 console (UnicodeDecodeError)
os.environ["PYTHONIOENCODING"] = "utf-8"


def run(cmd: list, env: dict | None = None, cwd: Path | None = None, check: bool = True, quiet: bool = False):
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    if env:
        e.update(env)
    p = subprocess.run([str(c) for c in cmd], env=e, cwd=str(cwd or ROOT),
                       stdout=subprocess.DEVNULL if quiet else None)
    if check and p.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} {' '.join(str(c) for c in cmd[1:4])} … → {p.returncode}")
    return p


def extract_archive(archive: Path, dest: Path, members: list[str] | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    root = dest.resolve()

    def inside(rel: str) -> bool:                      # a member of the archive must not point outside the target folder
        try:
            (root / rel).resolve().relative_to(root)
            return not Path(rel).is_absolute()
        except ValueError:
            return False
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():
                if (members is None or any(m.endswith(x) for x in members)) and inside(m):
                    z.extract(m, dest)
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not (members is None or any(m.name.endswith(x) for x in members)):
                    continue
                if not inside(m.name) or m.isdev() or ((m.issym() or m.islnk())
                                                        and not inside(str(Path(m.name).parent / m.linkname))):
                    continue
                if hasattr(tarfile, "data_filter"):     # Python 3.11.4+ / 3.12: safe extraction filter
                    t.extract(m, dest, filter="data")
                else:
                    t.extract(m, dest)


# ------------------------------------------------------------------ uv and the environment
def uv_env() -> dict:
    """uv writes only into the program folder: Python to tools/python, cache to tools/_download (deleted after the installation).
    UV_PYTHON_INSTALL_BIN=0 – without the launcher python3.11(.exe) in ~/.local/bin (`uv python install` would otherwise create it),
    UV_PYTHON_INSTALL_REGISTRY=0 – without registering Python in the Windows registry (PEP 514, HKCU\\\\Software\\\\Python).
    """
    env = {"UV_CACHE_DIR": os.environ.get("FEA_UV_CACHE_DIR") or str(DL / "uv-cache"), "UV_PYTHON_INSTALL_DIR": str(TOOLS / "python"),
           "UV_PYTHON_INSTALL_BIN": "0", "UV_PYTHON_INSTALL_REGISTRY": "0",
           "UV_LINK_MODE": "copy", "UV_NO_CONFIG": "1", "UV_PYTHON_PREFERENCE": "only-managed",
           "UV_NO_PROGRESS": "1" if QUIET else "0",
           # slow network / proxy / antivirus checking the downloaded files (mkl ~200 MB): longer timeout and more attempts
           "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "300"), "UV_HTTP_RETRIES": os.environ.get("UV_HTTP_RETRIES", "6")}
    if IS_WIN:
        # certificates from the Windows store: antiviruses that inspect HTTPS (Avast, ESET, Kaspersky, Bitdefender…) and corporate
        # proxies sign the connection with their own root authority – with the certificates bundled in uv the download would fail
        # ("invalid peer certificate: UnknownIssuer"); urllib and PowerShell always use the Windows store
        env.setdefault("UV_SYSTEM_CERTS", os.environ.get("UV_SYSTEM_CERTS", "1"))
    env.update(proxy_env())
    return env


def proxy_env() -> dict:
    """Proxy for uv: uv reads only the variables HTTPS_PROXY / HTTP_PROXY; a proxy set in Windows (Internet options,
    group policy) is passed to it from urllib.request.getproxies() – the installer's downloads use it in the same way.
    """
    if any(os.environ.get(k) for k in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")):
        return {}
    try:
        px = urllib.request.getproxies()
    except Exception:  # noqa: BLE001  (a damaged registry entry must not stop the installation)
        return {}
    out = {}
    for scheme, var in (("https", "HTTPS_PROXY"), ("http", "HTTP_PROXY")):
        if px.get(scheme):
            out[var] = px[scheme]
    if out and px.get("no"):
        out["NO_PROXY"] = px["no"]
    return out


def which_path(name: str) -> str | None:
    """Program from PATH without the working folder (shutil.which on Windows starts in it – a planted uv.exe / mpiexec.exe)."""
    dirs = [d.strip().strip('"') for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip()]
    dirs = [d for d in dirs if os.path.isabs(d)]
    if not IS_WIN:
        return shutil.which(name, path=os.pathsep.join(dirs))
    exts = [""] if Path(name).suffix else [e for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if e]
    for d in dirs:
        for e in exts:
            p = Path(d) / (name + e)
            if p.is_file():
                return str(p)
    return None


def ensure_uv() -> Path:
    if UV.exists():
        return UV
    found = os.environ.get("FEA_UV") or which_path("uv")
    if found:
        return Path(found)
    if PLAT not in UV_URL:
        die(T(f"Platforma {PLAT} není podporována (jen Windows x64 a Linux x86_64).",
              f"Platform {PLAT} is not supported (Windows x64 and Linux x86_64 only)."))
    step(T("Stahuji správce balíčků uv", "Downloading package manager uv"))
    arch = download(UV_URL[PLAT], DL / Path(UV_URL[PLAT]).name, f"uv {UV_VERSION}", UV_SHA256[PLAT])
    tmp = DL / "uv-extract"
    slim.rmtree(tmp)
    extract_archive(arch, tmp)
    src = next(tmp.rglob("uv" + EXE))
    UV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, UV)
    UV.chmod(0o755)
    slim.rmtree(tmp)
    ok(f"uv → {UV}")
    return UV


def uv(*args, **kw):
    return run([ensure_uv(), *args], env={**uv_env(), **kw.pop("env", {})}, **kw)


def ensure_python() -> None:
    """Python 3.11 in tools/python: if present → nothing; otherwise `uv python install` and, if it fails (antivirus / synchronisation of the folder
    broke the extraction of uv into tools/python/.temp), its own download with a verified digest and extraction without the temporary uv folder.
    """
    find = [ensure_uv(), "python", "find", PY_VERSION]
    if subprocess.run([str(c) for c in find], env={**os.environ, **uv_env(), "UV_PYTHON_DOWNLOADS": "never"},
                      capture_output=True).returncode == 0:
        return
    for attempt in range(2):
        if uv("python", "install", PY_VERSION, "--no-bin", "--no-registry", check=False).returncode == 0:
            return
        time.sleep(3)
    if PLAT not in PY_URL:
        die(T(f"Python {PY_VERSION} nejde nainstalovat (uv python install selhal).",
              f"Python {PY_VERSION} cannot be installed (uv python install failed)."))
    info(T("uv python install selhal – stahuji Python přímo (antivir nebo synchronizace složky mohly zasáhnout do "
           "rozbalování uv)", "uv python install failed – downloading Python directly (an antivirus or folder sync may "
           "have interfered with uv extraction)"))
    target = TOOLS / "python" / PY_DIR[PLAT]
    tmp = TOOLS / "python" / "_fea_extract"
    try:
        arch = download(PY_URL[PLAT], DL / f"cpython-{PY_FULL}-{PLAT}.tar.gz", f"Python {PY_FULL}", PY_SHA256[PLAT])
        slim.rmtree(tmp)
        slim.rmtree(target)
        extract_archive(arch, tmp)
        os.replace(tmp / "python", target)
        slim.rmtree(tmp)
    except (OSError, RuntimeError, tarfile.TarError) as e:
        die(T(f"Python {PY_VERSION} nejde rozbalit do {target}: {e}. Zkontrolujte antivir (výjimka pro složku programu) "
              "a zda složka není na síťovém disku nebo v synchronizované složce.",
              f"Python {PY_VERSION} cannot be extracted to {target}: {e}. Check the antivirus (exclusion for the program "
              "folder) and that the folder is not on a network drive or in a synced folder."))
    ok(f"Python {PY_FULL} → {target}")


def ensure_venv(force: bool = False) -> None:
    if VPY.exists() and not force:
        r = subprocess.run([str(VPY), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0 and r.stdout.strip() == PY_VERSION:
            return
    step(T(f"Python {PY_VERSION} a prostředí .venv", f"Python {PY_VERSION} and the .venv environment"))
    if VENV.exists():
        keep = {"hypre_env", STATE.name, upgrade.MANIFEST_NAME}      # the installation state survives a new environment
        for p in VENV.iterdir():
            if p.name not in keep:
                slim.rmtree(p) if p.is_dir() else p.unlink()
    ensure_python()
    uv("venv", str(VENV), "--python", PY_VERSION, "--allow-existing")
    ok(T("prostředí vytvořeno", "environment created"))


def lock_file() -> Path:
    p = ROOT / "installer" / "locks" / f"{PLAT}.txt"
    if not p.exists():
        die(T(f"Chybí zámek verzí {p.name}.", f"Version lock {p.name} is missing."))
    return p


OFFLINE = TOOLS / "offline" / "wheels"     # Python packages bundled in the offline installer (make_package.py --offline)
INDEX_ARGS: list[str] = []                  # --offline --no-index --find-links … for uv pip, when the offline packages are enough


def offline_wheels() -> bool:
    return OFFLINE.is_dir() and any(OFFLINE.glob("*.whl"))


def uv_pip(*args, attempts: int = 3) -> None:
    """uv pip with retries: an antivirus briefly locks freshly extracted files (os error 5 / 32) and the network sometimes drops.
    Further attempts without parallel installation (UV_CONCURRENT_INSTALLS=1) – fewer parallel writes for the antivirus check.
    """
    for i in range(attempts):
        env = {} if i == 0 else {"UV_CONCURRENT_INSTALLS": "1", "UV_CONCURRENT_DOWNLOADS": "2"}
        if uv("pip", *args, *INDEX_ARGS, "--python", str(VPY), env=env, check=False).returncode == 0:
            return
        if i + 1 < attempts:
            info(T(f"uv pip {args[0]} selhal – nový pokus ({i + 2}/{attempts}) za {5 * (i + 1)} s",
                   f"uv pip {args[0]} failed – retrying ({i + 2}/{attempts}) in {5 * (i + 1)} s"))
            time.sleep(5 * (i + 1))
    raise RuntimeError(T(
        f"Instalace balíčků Pythonu selhala (uv pip {args[0]}). Nejčastější příčiny: antivir blokuje / maže soubory ve "
        f"složce {ROOT} (přidejte výjimku pro složku a spusťte instalaci znovu), firemní proxy nebo firewall blokuje "
        "pypi.org a files.pythonhosted.org (nastavte HTTPS_PROXY, nebo použijte offline instalátor), složka na síťovém / "
        "synchronizovaném disku. Výpis uv je výše v okně, průběh instalace v install.log.",
        f"Installing the Python packages failed (uv pip {args[0]}). Most common causes: an antivirus blocks / deletes files "
        f"in {ROOT} (add a folder exclusion and run the installation again), a company proxy or firewall blocks pypi.org and "
        "files.pythonhosted.org (set HTTPS_PROXY, or use the offline installer), a network / synced folder. The uv output is "
        "above in the window, the installation log in install.log."))


def sync_packages(keys: list[str]) -> None:
    """Environment = exactly the packages of the selected modules (uv pip sync): missing ones are installed, surplus ones removed.
    With the bundled packages (offline installer) without network; if that is not enough (a module outside the offline set), from the internet.
    """
    global INDEX_ARGS
    reqs = sorted({r for k in keys for r in MODS[k].get("pip", [])})
    nodeps = sorted({r for k in keys for r in MODS[k].get("pip_nodeps", [])})
    with tempfile.TemporaryDirectory(prefix="fea_req_") as td:
        rin = Path(td) / "req.in"
        out = Path(td) / "resolved.txt"
        rin.write_text("\n".join(reqs) + "\n", encoding="utf-8")
        compile_args = ["compile", str(rin), "-c", str(lock_file()), "--no-header", "--no-annotate", "-o", str(out), "-q"]
        INDEX_ARGS = []
        if offline_wheels():
            local = ["--offline", "--no-index", "--find-links", str(OFFLINE)]
            if uv("pip", *compile_args, *local, "--python", str(VPY), check=False, quiet=True).returncode == 0:
                INDEX_ARGS = local
                info(T("balíčky z offline instalátoru (bez internetu)", "packages from the offline installer (no internet)"))
            else:
                INDEX_ARGS = ["--find-links", str(OFFLINE)]
                info(T("offline balíčky nepokrývají vybrané moduly – zbytek se stáhne z internetu",
                       "the offline packages do not cover the selected modules – the rest is downloaded"))
        uv_pip(*compile_args)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
        out.write_text("\n".join(sorted(set(lines + nodeps), key=str.lower)) + "\n", encoding="utf-8")
        info(T(f"{len(lines) + len(nodeps)} balíčků", f"{len(lines) + len(nodeps)} packages"))
        before = installed_records()
        uv_pip("sync", str(out))
        repair_shared_files(before, out)


def _norm(name: str) -> str:
    return name.lower().replace("-", "_").replace(".", "_")


def installed_records() -> dict[str, set[str]]:
    """Installed packages of the venv → files from RECORD (relative to site-packages, lower case)."""
    if not VPY.exists():
        return {}
    import csv
    recs: dict[str, set[str]] = {}
    for di in site_packages().glob("*.dist-info"):
        rec = di / "RECORD"
        if rec.is_file():
            with open(rec, encoding="utf-8", newline="") as f:
                name = di.name[: -len(".dist-info")].rsplit("-", 1)[0]      # „PySide6_Essentials-6.11.2.dist-info“
                recs[_norm(name)] = {r[0].lower() for r in csv.reader(f) if r}
    return recs


def repair_shared_files(before: dict[str, set[str]], resolved: Path) -> None:
    """Uninstalling a package deletes ALL files from its RECORD – also those it shares with another package (PySide6 +
    PySide6_Addons → PySide6/__init__.py disappeared and the GUI would not start: "DLL load failed while importing QtCore").
    Packages whose files were taken away by the removal of another package are reinstalled.
    """
    after = installed_records()
    removed = set(before) - set(after)
    if not removed:
        return
    gone = set().union(*(before[n] for n in removed))
    hit = sorted(n for n, files in after.items() if files & gone)
    if not hit:
        return
    pins = {_norm(l.split("==")[0].split("@")[0].strip()): l.strip() for l in resolved.read_text(encoding="utf-8").splitlines()
            if _norm(l.split("==")[0].split("@")[0].strip()) in hit}
    for di in site_packages().glob("*.dist-info"):               # safety net: version from the installed package
        name, _, ver = di.name[: -len(".dist-info")].rpartition("-")
        if _norm(name) in hit and _norm(name) not in pins:
            pins[_norm(name)] = f"{name}=={ver}"
    pins = sorted(pins.values())
    info(T(f"obnova souborů sdílených s odebranými balíčky: {', '.join(hit)}",
           f"restoring files shared with removed packages: {', '.join(hit)}"))
    if pins:
        uv_pip("install", "--reinstall", "--no-deps", *pins)


def restore_pruned_qt(keys: list[str]) -> None:
    """The docs module (Qt WebEngine) needs QML / Quick, which an earlier installation without docs pruned (slim.prune_qt) – the package
    PySide6_Essentials counts as installed, so the sync does not bring the files back → reinstall it (same version)."""
    if "docs" not in keys:
        return
    base = site_packages() / "PySide6"
    if not base.is_dir() or any(base.glob("Qt6Qml.*")) or any(base.glob("Qt/lib/libQt6Qml.so*")):
        return
    pins = []
    for di in site_packages().glob("*.dist-info"):
        name, _, ver = di.name[: -len(".dist-info")].rpartition("-")
        if _norm(name) == _norm("PySide6_Essentials"):
            pins.append(f"{name}=={ver}")
    if pins:
        info(T("Qt: obnova knihoven QML / Quick pro modul docs (dřívější prořezání)",
               "Qt: restoring the QML / Quick libraries for the docs module (pruned earlier)"))
        uv_pip("install", "--reinstall", "--no-deps", *pins)


def site_packages() -> Path:
    r = subprocess.run([str(VPY), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return Path(r.stdout.strip())


def write_pth() -> None:
    """Root of the program on the sys.path of the venv → `python -m fea_gui` works from any folder."""
    p = site_packages() / "fea_studio.pth"
    # ASCII only, no path: Python 3.11 reads .pth files in the national encoding (ANSI) and a path with a character outside the code page
    # (another script in the user / folder name) would not start the .venv at all ("Failed to import the site module").
    # Program root = parent of .venv = parent of sys.prefix (site sets it before processing the .pth).
    p.write_text("import os, sys; _r = os.path.dirname(sys.prefix); _r in sys.path or sys.path.append(_r)\n",
                 encoding="ascii")


# Visual C++ runtime (not on a clean Windows; the base Python 3.11 carries only vcruntime140*.dll)
VC_RUNTIME = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
              "concrt140.dll", "vcomp140.dll")
VC_NEEDED_BY = ("numba", "exudyn")          # packages whose .pyd want msvcp140 / vcomp140 without their own copy


def vc_runtime_local() -> None:
    """Runtime C++ next to python.exe of the base Python (the application folder = the first place for the DLL search in every process).

    numba and exudyn do not bundle it; in the GUI PySide6 loads it from its folder by chance first, but the MCP server, the command line
    and the background calculations would end with "DLL load failed" on a clean Windows (verified by an analysis of the imports of the installation).
    """
    if not IS_WIN or not VPY.exists():
        return
    r = subprocess.run([str(VPY), "-c", "import sys; print(sys.base_prefix)"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    home = Path(r.stdout.strip()) if r.returncode == 0 else None
    if home is None or not (home / "python.exe").exists():
        return
    site = site_packages()
    src = next((d for d in (site / "shiboken6", site / "PySide6") if (d / "msvcp140.dll").exists()), None)
    if src is None:
        if any((site / n).is_dir() for n in VC_NEEDED_BY) and not (home / "msvcp140.dll").exists():
            warn(T("chybí runtime Visual C++ (msvcp140.dll) – numba / exudyn na čistých Windows nemusí jít načíst; "
                   "nainstalujte Microsoft Visual C++ Redistributable x64 (https://aka.ms/vs/17/release/vc_redist.x64.exe)",
                   "Visual C++ runtime (msvcp140.dll) missing – numba / exudyn may fail to load on a clean Windows; install "
                   "the Microsoft Visual C++ Redistributable x64 (https://aka.ms/vs/17/release/vc_redist.x64.exe)"))
        return
    for n in VC_RUNTIME:
        if (src / n).exists() and not (home / n).exists():
            try:
                shutil.copy2(src / n, home / n)
            except OSError as ex:                   # the Python folder is held by a running process – the runtime from the system stays
                warn(f"{n}: {ex}")


# ------------------------------------------------------------------ external components
def ccx_zip() -> Path:
    return download(CCX_WIN_URL, DL / "calculix_2.23_4win.zip", "CalculiX 2.23 (dhondt.de)", CCX_WIN_SHA256)


def ccx_win_dir() -> Path:
    return TOOLS / "calculix" / "calculix_2.23_4win"


def install_ext_ccx(which: str) -> None:
    """ccx_static (base, without MKL; in the packages) or ccx_i4 (PARDISO, module pardiso). ccx_i4 is never part of a package
    (GPL-2.0 CalculiX + proprietary MKL): Windows extracts it from the original dhondt.de zip (download, digest CCX_WIN_SHA256),
    Linux builds it with build_ccx.sh – both need internet; without it the program uses ccx_static / ccx_spooles.
    """
    if IS_WIN:
        name = {"ccx_static": "ccx_static.exe", "ccx_i4": "ccx_i4.exe"}[which]
        d = ccx_win_dir()
        if (d / name).exists():
            ok(T(f"{name} už je v tools\\calculix", f"{name} is already in tools\\calculix"))
            return
        if which == "ccx_i4":
            info(T("ccx_i4.exe (CalculiX s Intel MKL PARDISO) se nešíří s FEA Studiem – rozbalí se z původního archivu "
                   "dhondt.de (potřebuje internet)",
                   "ccx_i4.exe (CalculiX with Intel MKL PARDISO) is not distributed with FEA Studio – it is extracted from "
                   "the original dhondt.de archive (needs internet)"))
        try:
            z = ccx_zip()
        except RuntimeError as e:
            if which != "ccx_i4":
                raise
            raise RuntimeError(T(f"{e} – bez ccx_i4.exe počítá CalculiX řešičem PaStiX / SPOOLES (ccx_static.exe, ~40 % "
                                 "pomaleji); doinstalovat později: python installer/fea_setup.py install pardiso",
                                 f"{e} – without ccx_i4.exe CalculiX uses PaStiX / SPOOLES (ccx_static.exe, ~40 % slower); "
                                 "add it later: python installer/fea_setup.py install pardiso")) from e
        extract_archive(z, d.parent, members=[name, "README_Install"])
        if not (d / name).exists():
            raise RuntimeError(T(f"Po rozbalení chybí {d / name} – změnila se struktura archivu dhondt.de?",
                                 f"{d / name} missing after extraction – did the dhondt.de archive change?"))
        ok(f"{name} → {d}")
        return
    d = TOOLS / "calculix" / "ccx_2.23_linux"
    name = {"ccx_static": "ccx_spooles", "ccx_i4": "ccx_i4"}[which]
    if (d / name).exists():
        ok(T(f"{name} už je v tools/calculix", f"{name} is already in tools/calculix"))
        return
    # 1) a prebuilt archive in the package / next to the installer, 2) build from the sources (build_ccx.sh)
    for cand in [TOOLS / "calculix" / "ccx_2.23_linux_x86_64.tar.gz", ROOT / "ccx_2.23_linux_x86_64.tar.gz"]:
        if cand.exists():
            extract_archive(cand, d)
            if (d / name).exists():
                (d / name).chmod(0o755)
                ok(f"{name} z {cand.name}")
                return
    if which == "ccx_i4":
        info(T("ccx_i4 (CalculiX s Intel MKL PARDISO) se nešíří s FEA Studiem – sestaví se zde ze zdrojů proti knihovně mkl",
               "ccx_i4 (CalculiX with Intel MKL PARDISO) is not distributed with FEA Studio – it is built here from source "
               "against the mkl library"))
        if not (VENV / "lib" / "libmkl_rt.so.2").exists():
            raise RuntimeError(T("ccx_i4: chybí .venv/lib/libmkl_rt.so.2 (balíček mkl modulu pardiso) – CalculiX zůstane ccx_spooles",
                                 "ccx_i4: .venv/lib/libmkl_rt.so.2 (mkl package of the pardiso module) is missing – CalculiX "
                                 "stays ccx_spooles"))
    if not (d / "ccx_spooles").exists() or which == "ccx_i4":
        build_ccx_linux(d)
    if not (d / name).exists():
        raise RuntimeError(T(f"CalculiX ({name}) se nepodařilo získat.", f"Could not obtain CalculiX ({name})."))


def build_ccx_linux(out: Path) -> None:
    for tool in ("curl", "tar", "bzip2", "perl"):
        if not shutil.which(tool):
            raise RuntimeError(T(f"Pro sestavení CalculiXu chybí „{tool}“ (apt install curl bzip2 perl).",
                                 f"'{tool}' is required to build CalculiX (apt install curl bzip2 perl)."))
    info(T("Sestavuji CalculiX 2.23 ze zdrojů (několik minut, překladače se stáhnou z conda-forge)…",
           "Building CalculiX 2.23 from source (a few minutes, compilers come from conda-forge)…"))
    env = {"MKL_LIB_DIR": str(VENV / "lib")} if (VENV / "lib" / "libmkl_rt.so.2").exists() else {}
    env.update(mamba_env())             # the micromamba script (compilers): cache and home folder only in tools/_download
    try:
        run(["bash", str(ROOT / "installer" / "linux" / "build_ccx.sh"), str(out), str(DL / "ccx_build")], env=env)
    finally:
        mamba_cleanup()
    for f in out.glob("ccx_*"):
        f.chmod(0o755)
    slim.rmtree(DL / "ccx_build")


def micromamba() -> Path:
    mm = DL / "micromamba"
    exe = mm / ("Library/bin/micromamba.exe" if IS_WIN else "bin/micromamba")
    if not exe.exists():
        arch = download(MAMBA_URL[PLAT], DL / "micromamba.tar.bz2", f"micromamba {MAMBA_VERSION}", MAMBA_SHA256[PLAT])
        extract_archive(arch, mm)
    return exe


MAMBA_DIRS = ("mamba-root", "mamba-home", "mamba-cache", "micromamba")      # in tools/_download, deleted after use


def mamba_env() -> dict:
    """Environment for micromamba: EVERYTHING in tools/_download. Without it micromamba 2.x (verified 2.9.0, Windows and Linux) writes also
    outside MAMBA_ROOT_PREFIX: repodata cache to $XDG_CACHE_HOME/conda (~/.cache/conda, Windows %USERPROFILE%\\\\AppData\\\\
    Local\\\\conda), pkgs/urls.txt to ~/.mamba (Windows also %APPDATA%\\\\.mamba) and the list of environments ~/.conda/environments.txt
    (path to the program folder). The registration of the environment cannot be switched off by configuration → the home folder only for micromamba.
    """
    home = DL / "mamba-home"
    home.mkdir(parents=True, exist_ok=True)
    e = {"MAMBA_ROOT_PREFIX": str(DL / "mamba-root"), "CONDA_PKGS_DIRS": str(DL / "mamba-root" / "pkgs"),
         "XDG_CACHE_HOME": str(DL / "mamba-cache"), "HOME": str(home)}
    if IS_WIN:
        e["USERPROFILE"] = str(home)
    return e


def mamba_cleanup() -> None:
    for d in MAMBA_DIRS:
        slim.rmtree(DL / d)


def conda_create(prefix: Path, specs: list[str], cleanup: bool = True) -> None:
    """Creates the conda-forge environment through micromamba (downloaded to tools/_download, deleted after use together with the package cache
    that has hundreds of MB).
    """
    exe = micromamba()
    try:
        run([exe, "create", "-y", "-q", "-p", prefix, "-c", "conda-forge", "--no-rc", *specs], env=mamba_env())
    finally:
        if cleanup:
            mamba_cleanup()


def conda_plan(prefix: Path, specs: list[str]) -> list[dict]:
    """Packages that micromamba would install (dry run, JSON): [{name, version, build_string, url}]."""
    exe = micromamba()
    e = dict(os.environ)
    e.update({**mamba_env(), "PYTHONIOENCODING": "utf-8"})
    r = subprocess.run([str(exe), "create", "-y", "--dry-run", "--json", "-p", str(prefix), "-c", "conda-forge",
                        "--no-rc", *specs], env=e, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"micromamba --dry-run → {r.returncode}: {r.stderr[-400:]}")
    out = json.loads(r.stdout[r.stdout.index("{"):])
    pk = out.get("actions", {}).get("LINK", [])
    for x in pk:
        if not x.get("url"):
            x["url"] = f"https://conda.anaconda.org/conda-forge/{x.get('subdir', '')}/{x['fn']}"
    return pk


def install_ext_hypre() -> None:
    env = VENV / "hypre_env"
    lib = env / ("Library/bin/HYPRE.dll" if IS_WIN else "lib/libHYPRE.so")
    if lib.exists():
        ok(T("hypre už je v .venv/hypre_env", "hypre is already in .venv/hypre_env"))
        return
    specs = ["hypre=3.2.0", "libblas=*=*openblas"] if IS_WIN else ["hypre=3.2.0", "mpich", "libblas=*=*openblas"]
    info(T("vytvářím prostředí hypre 3.2.0 (conda-forge)…", "creating the hypre 3.2.0 environment (conda-forge)…"))
    conda_create(env, specs)
    slim.prune_conda_env(env)
    if not lib.exists():
        raise RuntimeError(T("hypre: po instalaci chybí knihovna HYPRE.", "hypre: the HYPRE library is missing after install."))
    ok(T("hypre nainstalován", "hypre installed"))


CHRONO_SPEC = "pychrono=10.0.0"
CHRONO_PKGS = ("chrono", "pychrono")              # extracted by hand (without the C++ headers and sample data)
CHRONO_SKIP = ("eigen",)                          # headers only – not needed at run time


def chrono_env() -> Path:
    return TOOLS / "chrono_env"


def install_ext_chrono() -> None:
    """Chrono solver for contact in the mechanism (F8.9 C1): tools/chrono_env (NOT in .venv – the conda interpreter would find
    .venv/pyvenv.cfg and start as a venv of the main environment). micromamba on Windows does not unpack the chrono package
    (headers with paths over 260 characters) → the environment is created from all dependencies WITHOUT chrono/pychrono and these two packages
    are unpacked by fea_core.motion.conda_extract (it leaves out headers, tests and sample data – ~1.2 GB).
    """
    env = chrono_env()
    sp = env / ("Lib/site-packages" if IS_WIN else "lib/python3.11/site-packages")
    py = env / ("python.exe" if IS_WIN else "bin/python")
    if (sp / "pychrono").is_dir() and py.exists():
        ok(T("Chrono už je v tools/chrono_env", "Chrono is already in tools/chrono_env"))
        return
    info(T("vytvářím prostředí Chrono 10.0 (conda-forge, ~0,6 GB)…", "creating the Chrono 10.0 environment (conda-forge, ~0.6 GB)…"))
    specs = [CHRONO_SPEC, "python=3.11", "numpy", "zstandard", "libblas=*=*openblas"]
    try:
        plan = conda_plan(env, specs)
        manual = [x for x in plan if x["name"] in CHRONO_PKGS]
        if len(manual) != len(CHRONO_PKGS):
            raise RuntimeError(T("Chrono: conda-forge nenabízí pychrono 10.0.0 pro tuto platformu",
                                 "Chrono: conda-forge has no pychrono 10.0.0 for this platform"))
        rest = [f"{x['name']}={x['version']}={x['build_string']}" for x in plan
                if x["name"] not in CHRONO_PKGS + CHRONO_SKIP]
        conda_create(env, rest, cleanup=False)
        for x in manual:
            if not x.get("sha256"):                      # without a digest it is not downloaded (integrity through TLS alone is not enough)
                raise RuntimeError(T(f"Chrono: repodata neuvádí otisk SHA-256 balíčku {x['name']}",
                                     f"Chrono: repodata has no SHA-256 for package {x['name']}"))
            pkg = download(x["url"], DL / Path(x["url"]).name, f"{x['name']} {x['version']}",
                           x["sha256"])                      # otisk z repodata conda-forge (micromamba --json)
            run([py, "-m", "fea_core.motion.conda_extract", pkg, env], env={"PYTHONPATH": str(ROOT)})
            pkg.unlink(missing_ok=True)
    finally:
        mamba_cleanup()
    r = subprocess.run([str(py), "-c", "import pychrono; print(pychrono.ChSystemNSC)"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "VIRTUAL_ENV")})
    if r.returncode != 0:
        raise RuntimeError(T(f"Chrono: import pychrono selhal: {r.stderr[-300:]}", f"Chrono: import pychrono failed: {r.stderr[-300:]}"))
    ok(T("Chrono nainstalován (tools/chrono_env)", "Chrono installed (tools/chrono_env)"))


def find_mpiexec() -> str | None:
    """mpiexec from PATH or from the MS-MPI folder (MSMPI_BIN, Program Files) – the same as fea_core.solvers.find_mpiexec
    (the installer runs even without .venv, fea_core is not imported).
    """
    w = which_path("mpiexec")
    if w or not IS_WIN:
        return w
    pf = os.environ.get("ProgramFiles") or os.environ.get("ProgramW6432") or "C:\\Program Files"
    for d in (os.environ.get("MSMPI_BIN", ""), str(Path(pf) / "Microsoft MPI" / "Bin")):
        if d and (Path(d) / "mpiexec.exe").is_file():
            return str(Path(d) / "mpiexec.exe")
    return None


def install_ext_mpi() -> None:
    if not IS_WIN:
        return                                              # MPICH is part of the hypre environment
    if find_mpiexec():
        ok(T("mpiexec nalezen (MS-MPI)", "mpiexec found (MS-MPI)"))
        return
    # without a question (--quiet / --yes / no terminal) only with the explicit consent --msmpi – installation into the system with UAC
    if not MSMPI and not ask(T("MS-MPI chybí (bez něj nepůjde řešič NGSolve + hypre). Stáhnout a nainstalovat (UAC, správce)?",
                               "MS-MPI is missing (the NGSolve + hypre solver needs it). Download and install (UAC, administrator)?"),
                             False):
        warn(T(f"MS-MPI chybí – řešič NGSolve + hypre nepůjde. Doinstalujte: python installer/fea_setup.py mpi "
               f"(nebo {MSMPI_URL})",
               f"MS-MPI missing – the NGSolve + hypre solver will not run. Install: python installer/fea_setup.py mpi "
               f"(or {MSMPI_URL})"))
        return
    setup = download(MSMPI_URL, DL / "msmpisetup.exe", "MS-MPI", MSMPI_SHA256)
    ps = ("$p = Start-Process -FilePath $env:FEA_MSMPI -ArgumentList '-unattend' -Verb RunAs -Wait -PassThru; "
          "exit $p.ExitCode")
    r = run(["powershell", "-NoProfile", "-Command", ps], env={"FEA_MSMPI": str(setup)}, check=False)
    if r.returncode == 0:
        ok(T("MS-MPI nainstalováno – restartujte FEA Studio (nové okno uvidí mpiexec v PATH)",
             "MS-MPI installed – restart FEA Studio (a new terminal will see mpiexec)"))
    else:
        warn(T(f"instalace MS-MPI skončila kódem {r.returncode} (zamítnutý dotaz UAC?)",
               f"MS-MPI setup exited with code {r.returncode} (UAC prompt declined?)"))


def install_ext_mystran() -> None:
    """MYSTRAN (MIT) has no public release with a permanent link (github.com/MYSTRAN/MYSTRAN → Releases is empty, the binaries are on
    mystran.com) – what is in tools/mystran is used (the Windows zip package contains it); otherwise an instruction.
    """
    exe = TOOLS / "mystran" / ("mystran" + EXE)
    if exe.exists():
        exe.chmod(0o755)
        ok("MYSTRAN")
        return
    warn(T("MYSTRAN: binárka není součástí tohoto balíčku – stáhněte ji z https://www.mystran.com (fórum → Download) a uložte jako "
           f"{exe}; bez ní nepůjdou jen křížové kontroly MYSTRAN a vzpěr prutů (F9, F3M)",
           "MYSTRAN: the binary is not part of this package – download it from https://www.mystran.com (forum → Download) and save it as "
           f"{exe}; without it only the MYSTRAN cross-checks and beam buckling (F9, F3M) are unavailable"))


EXT = {"ccx_static": lambda: install_ext_ccx("ccx_static"), "ccx_i4": lambda: install_ext_ccx("ccx_i4"),
       "hypre": install_ext_hypre, "mpi": install_ext_mpi, "mystran": install_ext_mystran, "chrono": install_ext_chrono}


def remove_ext(key: str) -> None:
    if key == "pardiso":
        for p in (TOOLS / "calculix").glob("*/ccx_i4*"):
            p.unlink(missing_ok=True)
    elif key == "hypre":
        slim.rmtree(VENV / "hypre_env")
    elif key == "chrono":
        slim.rmtree(chrono_env())
    elif key == "verify":
        slim.rmtree(TOOLS / "mystran")


# ------------------------------------------------------------------ shortcuts and start
def launchers() -> None:
    """Linux: the launchers from the package must be executable (zip / copying loses the bits). Windows has `FEA Studio.bat`."""
    if IS_WIN:
        return
    for name in ("fea-studio", "install.sh"):
        if (ROOT / name).exists():
            (ROOT / name).chmod(0o755)


def make_shortcuts() -> None:
    icon_ico = ROOT / "installer" / "icons" / "fea_studio.ico"
    icon_png = ROOT / "installer" / "icons" / "fea_studio.png"
    if IS_WIN:
        pyw = VENV / "Scripts" / "pythonw.exe"
        py = VENV / "Scripts" / "python.exe"
        # -E: the variables PYTHONHOME / PYTHONPATH of another program (GIS, CAD, older Python) must not break the start
        # (fea_gui/__main__ then deletes them also for the helper processes)
        links = [("Desktop", "FEA Studio.lnk", pyw, "-P -E -m fea_gui", "FEA Studio"),
                 ("Programs", r"FEA Studio\FEA Studio.lnk", pyw, "-P -E -m fea_gui", "FEA Studio"),
                 ("Programs", r"FEA Studio\FEA Studio - REST API.lnk", py, "-m fea_core.rest", "REST API (localhost:8765)"),
                 ("Programs", r"FEA Studio\Moduly FEA Studia.lnk", py, "-m fea_core.modules", "Instalace a odebrani modulu"),
                 ("SendTo", "FEA Studio.lnk", pyw, "-P -E -m fea_gui", "Otevrit STEP / .fea ve FEA Studiu")]
        failed = []
        try:
            import winlink
            for folder, name, exe, arg, desc in links:
                try:
                    dest = Path(winlink.known_folder(folder)) / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    winlink.create(str(dest), str(exe), arg, str(ROOT), desc, str(icon_ico) if icon_ico.exists() else "")
                except OSError as e:          # e.g. Controlled folder access (Defender) on the desktop
                    failed.append((folder, name, exe, arg, desc, e))
        except (OSError, ImportError, AttributeError) as e:
            failed = [(*l, e) for l in links]
        if failed:
            _log("shortcuts via IShellLinkW failed: " + "; ".join(f"{f[1]}: {f[5]}" for f in failed))
            shortcuts_powershell(failed, icon_ico)
        import winreg
        cls = r"Software\Classes"
        for key, val in ((cls + r"\.fea", "FEAStudio.Project"), (cls + r"\FEAStudio.Project", "Projekt FEA Studio"),
                         (cls + r"\FEAStudio.Project\shell\open\command", f'"{pyw}" -P -E -m fea_gui "%1"'),
                         (cls + r"\FEAStudio.Project\DefaultIcon", str(icon_ico))):
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key) as k:
                winreg.SetValueEx(k, "", 0, winreg.REG_SZ, val)
        write_locator()
        ok(T("zástupci (Plocha, nabídka Start, Odeslat do) a přípona .fea",
             "shortcuts (Desktop, Start menu, Send to) and the .fea extension"))
        return
    write_locator()
    apps = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    (apps / "applications").mkdir(parents=True, exist_ok=True)
    icon = apps / "icons" / "hicolor" / "256x256" / "apps" / "fea-studio.png"
    if icon_png.exists():
        icon.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(icon_png, icon)
    (apps / "applications" / "fea-studio.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=FEA Studio\nComment=FEM analysis on CalculiX / Gmsh\n"
        f"Exec={ROOT / 'fea-studio'} %f\nPath={ROOT}\nIcon={'fea-studio' if icon_png.exists() else 'applications-engineering'}\n"
        "Terminal=false\nCategories=Engineering;Science;\nMimeType=application/x-fea-studio;model/step;\n", encoding="utf-8")
    mime = apps / "mime" / "packages"
    mime.mkdir(parents=True, exist_ok=True)
    (mime / "fea-studio.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">\n'
        '  <mime-type type="application/x-fea-studio"><comment>FEA Studio project</comment>\n'
        '    <glob pattern="*.fea"/></mime-type>\n</mime-info>\n', encoding="utf-8")
    for tool, args in (("update-mime-database", [str(apps / "mime")]), ("update-desktop-database", [str(apps / "applications")])):
        if shutil.which(tool):
            run([tool, *args], check=False, quiet=True)
    ok(T("položka v nabídce aplikací a přípona .fea", "application menu entry and the .fea file type"))


def shortcuts_powershell(links: list, icon_ico: Path) -> None:
    """Backup of the shortcuts through WScript.Shell in PowerShell (only when IShellLinkW fails)."""
    lines = ["$sh = New-Object -ComObject WScript.Shell"]
    env = {"FEA_ROOT": str(ROOT), "FEA_ICO": str(icon_ico)}
    for i, (folder, name, exe, arg, desc, _e) in enumerate(links):
        env.update({f"FEA_N{i}": name, f"FEA_X{i}": str(exe), f"FEA_A{i}": arg, f"FEA_D{i}": desc})
        lines.append(f"$p = Join-Path ([Environment]::GetFolderPath('{folder}')) $env:FEA_N{i}; "
                     "New-Item -ItemType Directory -Force -Path (Split-Path $p) | Out-Null; "
                     f"$s = $sh.CreateShortcut($p); $s.TargetPath = $env:FEA_X{i}; $s.Arguments = $env:FEA_A{i}; "
                     f"$s.WorkingDirectory = $env:FEA_ROOT; $s.Description = $env:FEA_D{i}; "
                     "if (Test-Path $env:FEA_ICO) { $s.IconLocation = $env:FEA_ICO }; $s.Save()")
    r = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", "\n".join(lines)], env=env, check=False)
    if r.returncode != 0:
        warn(T("některé zástupce nejde vytvořit (antivir / Řízený přístup ke složkám?) – program spustíte i přes "
               "„FEA Studio.bat“ ve složce programu", "some shortcuts cannot be created (antivirus / Controlled folder "
               "access?) – you can also start the program with “FEA Studio.bat” in the program folder"))


def locator_path() -> Path:
    """install.json in the user settings (as fea_core.config.user_config_dir) – the CAD macros
    (adapters/solidworks, adapters/inventor) find the program folder by it.
    """
    if IS_WIN:
        base = os.environ.get("APPDATA") or str(Path.home())
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "FEAStudio" / "install.json"


def language_path() -> Path:
    """language.txt in the user settings (as fea_core.i18n.language_path) – the program language chosen at installation; the switch in
    the program writes the same file, so the last choice wins."""
    return locator_path().parent / "language.txt"


def pick_language(args) -> None:
    """Program language: --lang (setup.exe passes the language of the setup wizard), otherwise the earlier choice stays; at the first
    installation a question (default by the system language)."""
    lang = getattr(args, "lang", None)
    p = language_path()
    if not lang:
        if p.exists():
            return
        lang = "cs" if CS else "en"
        if not (QUIET or YES or not sys.stdin or not sys.stdin.isatty()):
            print(T("\nJazyk programu (lze později přepnout v programu):", "\nProgram language (can be switched later in the program):"))
            print("  1) Čeština\n  2) English")
            c = input(T(f"Volba [{1 if lang == 'cs' else 2}]: ", f"Choice [{1 if lang == 'cs' else 2}]: ")).strip()
            lang = {"1": "cs", "2": "en"}.get(c, lang)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(lang + "\n", encoding="utf-8")
        ok(T(f"jazyk programu: {'čeština' if lang == 'cs' else 'angličtina'}", f"program language: {'Czech' if lang == 'cs' else 'English'}"))
    except OSError as ex:
        warn(T(f"jazyk programu nejde uložit: {ex}", f"cannot save the program language: {ex}"))


def write_locator() -> None:
    p = locator_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"home": str(ROOT), "version": upgrade.program_version(ROOT)}, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    except OSError as ex:
        warn(T(f"install.json (cesta pro makra CAD) nejde zapsat: {ex}", f"cannot write install.json (path for CAD macros): {ex}"))


def remove_locator() -> None:
    p = locator_path()
    try:
        if p.is_file() and json.loads(p.read_text(encoding="utf-8")).get("home", "").lower() == str(ROOT).lower():
            p.unlink()
    except (OSError, ValueError, AttributeError):
        pass


def _ours(text: str) -> bool:
    """The shortcut / record belongs to THIS installation (target in the program folder) – another installation on the same PC is not removed."""
    return str(ROOT).lower() in (text or "").lower()


def remove_shortcuts() -> None:
    """Removes the shortcut and the .fea extension only if they point to this installation (uninstalling a test copy / a second installation
    must not break the main shortcut).
    """
    if IS_WIN:
        import winreg
        cls = r"Software\Classes\FEAStudio.Project"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, cls + r"\shell\open\command") as k:
                cmd = winreg.QueryValueEx(k, "")[0]
        except OSError:
            cmd = ""
        if _ours(cmd):
            for key in (r"Software\Classes\.fea", cls + r"\shell\open\command", cls + r"\shell\open", cls + r"\shell",
                        cls + r"\DefaultIcon", cls):
                try:
                    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
                except OSError:
                    pass
        try:
            import winlink
            start = Path(winlink.known_folder("Programs")) / "FEA Studio"
            cands = [Path(winlink.known_folder("Desktop")) / "FEA Studio.lnk",
                     Path(winlink.known_folder("SendTo")) / "FEA Studio.lnk"] + sorted(start.glob("*.lnk"))
            for lnk in cands:
                try:
                    if lnk.is_file() and os.path.normcase(winlink.read(str(lnk))["workdir"]) == os.path.normcase(str(ROOT)):
                        lnk.unlink()
                except OSError:
                    pass
            if start.is_dir() and not any(start.iterdir()):
                start.rmdir()
        except (OSError, ImportError, AttributeError) as e:
            _log(f"shortcuts via IShellLinkW: {e} – PowerShell fallback")
            ps = r"""
    $sh = New-Object -ComObject WScript.Shell
    $start = Join-Path ([Environment]::GetFolderPath('Programs')) 'FEA Studio'
    $links = @(Join-Path ([Environment]::GetFolderPath('Desktop')) 'FEA Studio.lnk') + @(Join-Path ([Environment]::GetFolderPath('SendTo')) 'FEA Studio.lnk') + @(Get-ChildItem -Path $start -Filter *.lnk -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    foreach ($l in $links) {
      if ((Test-Path $l) -and ($sh.CreateShortcut($l).WorkingDirectory -eq $env:FEA_ROOT)) { Remove-Item -Force $l }
    }
    if ((Test-Path $start) -and -not (Get-ChildItem $start)) { Remove-Item -Force $start }
    """
            run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], env={"FEA_ROOT": str(ROOT)},
                check=False)
    else:
        apps = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        desk = apps / "applications" / "fea-studio.desktop"
        if desk.is_file() and _ours(desk.read_text(encoding="utf-8", errors="replace")):
            for p in (desk, apps / "mime" / "packages" / "fea-studio.xml",
                      apps / "icons" / "hicolor" / "256x256" / "apps" / "fea-studio.png"):
                p.unlink(missing_ok=True)
    remove_locator()
    ok(T("zástupci a přípona .fea této instalace odebrány", "shortcuts and the .fea file type of this installation removed"))


# ------------------------------------------------------------------ Autodesk Fusion add-in
FUSION_SRC = ROOT / "adapters" / "fusion" / "FEAStudio"
FUSION_SKIP = ("__pycache__", "settings.json")


def fusion_dir() -> Path | None:
    """User data folder of Fusion (%APPDATA%/Autodesk/Autodesk Fusion 360) or None (no Fusion / Linux)."""
    if not IS_WIN or not os.environ.get("APPDATA"):
        return None
    d = Path(os.environ["APPDATA"]) / "Autodesk" / "Autodesk Fusion 360"
    return d if d.is_dir() else None


def install_fusion_addin() -> Path:
    """Copies the add-in to the API/AddIns of Fusion and writes settings.json with the path to this installation (fea_studio_home).
    Own settings in an existing settings.json stay. It takes effect after restarting Fusion (manifest runOnStartup).
    """
    base = fusion_dir()
    if base is None:
        raise RuntimeError(T("Autodesk Fusion není nainstalován (chybí %APPDATA%\\Autodesk\\Autodesk Fusion 360).",
                             "Autodesk Fusion is not installed (%APPDATA%\\Autodesk\\Autodesk Fusion 360 is missing)."))
    dest = base / "API" / "AddIns" / FUSION_SRC.name
    settings = dest / "settings.json"
    old = {}
    if settings.is_file():
        try:
            old = json.loads(settings.read_text(encoding="utf-8"))
        except ValueError:
            warn(T(f"nečitelný {settings} – přepisuje se", f"unreadable {settings} – overwriting"))
    shutil.copytree(FUSION_SRC, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*FUSION_SKIP))
    old["fea_studio_home"] = str(ROOT)
    settings.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
    ok(T(f"doplněk Fusion: {dest} (tlačítko FEA STUDIO se objeví po restartu Fusionu)",
         f"Fusion add-in: {dest} (the FEA STUDIO button appears after restarting Fusion)"))
    return dest


def remove_fusion_addin() -> None:
    base = fusion_dir()
    dest = base / "API" / "AddIns" / FUSION_SRC.name if base is not None else None
    if dest is not None and (dest / "FEAStudio.manifest").is_file():
        try:                                    # an add-in of another installation (fea_studio_home elsewhere) is left alone
            home = json.loads((dest / "settings.json").read_text(encoding="utf-8")).get("fea_studio_home", "")
        except (OSError, ValueError):
            home = ""
        if home and Path(home).resolve() != ROOT.resolve():
            info(T(f"doplněk Fusion patří jiné instalaci ({home}) – ponechán", f"Fusion add-in belongs to another installation ({home}) – kept"))
            return
        slim.rmtree(dest)
        ok(T(f"doplněk Fusion odebrán: {dest}", f"Fusion add-in removed: {dest}"))


# soname → (conda-forge package, Debian/Ubuntu package)
SYSLIB_PKGS = {
    "libGLU.so": ("libglu", "libglu1-mesa"), "libGL.so": ("libgl", "libgl1"), "libEGL.so": ("libegl", "libegl1"),
    "libOpenGL.so": ("libopengl", "libopengl0"), "libGLX.so": ("libglx", "libglx0"),
    "libxkbcommon-x11.so": ("libxkbcommon", "libxkbcommon-x11-0"), "libxkbcommon.so": ("libxkbcommon", "libxkbcommon0"),
    "libfontconfig.so": ("fontconfig", "libfontconfig1"), "libdbus-1.so": ("dbus", "libdbus-1-3"),
    "libxcb-cursor.so": ("xcb-util-cursor", "libxcb-cursor0"), "libxcb-icccm.so": ("xcb-util-wm", "libxcb-icccm4"),
    "libxcb-image.so": ("xcb-util-image", "libxcb-image0"), "libxcb-keysyms.so": ("xcb-util-keysyms", "libxcb-keysyms1"),
    "libxcb-render-util.so": ("xcb-util-renderutil", "libxcb-render-util0"), "libxcb-shape.so": ("libxcb", "libxcb-shape0"),
    "libxcb-randr.so": ("libxcb", "libxcb-randr0"), "libxcb-xinerama.so": ("libxcb", "libxcb-xinerama0"),
    "libxcb-xkb.so": ("libxcb", "libxcb-xkb1"), "libxcb-sync.so": ("libxcb", "libxcb-sync1"),
    "libxcb-xfixes.so": ("libxcb", "libxcb-xfixes0"), "libxcb-render.so": ("libxcb", "libxcb-render0"),
    "libxcb-shm.so": ("libxcb", "libxcb-shm0"), "libxcb-util.so": ("xcb-util", "libxcb-util1"),
    "libX11-xcb.so": ("xorg-libx11", "libx11-xcb1"), "libSM.so": ("xorg-libsm", "libsm6"), "libICE.so": ("xorg-libice", "libice6"),
    "libXrender.so": ("xorg-libxrender", "libxrender1"), "libXext.so": ("xorg-libxext", "libxext6"),
}


def _lib_key(soname: str) -> str:
    return soname.split(".so")[0] + ".so"


def _not_found(files: list[Path]) -> set[str]:
    if not shutil.which("ldd"):
        return set()
    env = child_env_linux()
    miss: set[str] = set()
    for f in files:
        out = subprocess.run(["ldd", str(f)], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env).stdout
        miss |= {l.split("=>")[0].strip() for l in out.splitlines() if "not found" in l}
    return miss


def child_env_linux() -> dict:
    e = dict(os.environ)
    sysl = VENV / "sys_libs" / "lib"
    if sysl.is_dir():
        e["LD_LIBRARY_PATH"] = str(sysl) + os.pathsep + e.get("LD_LIBRARY_PATH", "")
    return e


def trim_sys_libs(miss: set[str]) -> None:
    """From the conda environment it keeps only the libraries that the system lacked, and their dependencies inside the environment."""
    lib = VENV / "sys_libs" / "lib"
    names = {f.name: f for f in lib.iterdir() if f.is_file() or f.is_symlink()}
    todo = [n for n in names if any(n.startswith(_lib_key(m)) for m in miss)]
    keep: set[str] = set()
    while todo:
        n = todo.pop()
        if n in keep:
            continue
        keep.add(n)
        real = (lib / n).resolve()
        keep.add(real.name)
        out = subprocess.run(["ldd", str(real)], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        for line in out.splitlines():
            if "=>" in line and str(lib) in line:
                todo.append(line.split("=>")[1].split()[0].rsplit("/", 1)[-1])
    for n, f in names.items():
        if n not in keep:
            f.unlink(missing_ok=True)
    for d in list(lib.iterdir()):
        if d.is_dir():
            slim.rmtree(d)


def ensure_sys_libs(keys: list[str]) -> None:
    """Linux: gmsh needs libGLU, the Qt (GUI) libraries xcb / EGL / fontconfig. The missing ones are downloaded from conda-forge
    into .venv/sys_libs (without administrator rights); what fails is printed as an apt command for your distribution.
    """
    if IS_WIN:
        return
    site = site_packages()
    targets = list((VENV / "lib").glob("libgmsh*.so*")) + list((site / "OCP").glob("OCP*.so"))
    if "gui" in keys:
        targets += list(site.glob("PySide6/Qt/plugins/platforms/libqxcb.so")) + list(site.glob("PySide6/Qt/lib/libQt6Gui.so*"))
    miss = _not_found(targets)
    if not miss:
        ok(T("systémové knihovny jsou k dispozici", "system libraries are present"))
        return
    pkgs = sorted({SYSLIB_PKGS[_lib_key(m)][0] for m in miss if _lib_key(m) in SYSLIB_PKGS})
    if pkgs:
        info(T(f"chybí systémové knihovny ({', '.join(sorted(miss))}) – stahuji z conda-forge: {', '.join(pkgs)}",
               f"missing system libraries ({', '.join(sorted(miss))}) – fetching from conda-forge: {', '.join(pkgs)}"))
        try:
            slim.rmtree(VENV / "sys_libs")
            conda_create(VENV / "sys_libs", pkgs)
            for rel in ("include", "share", "man", "conda-meta", "pkgs"):
                slim.rmtree(VENV / "sys_libs" / rel)
            trim_sys_libs(miss)
        except Exception as ex:  # noqa: BLE001
            info(f"conda-forge: {ex}")
    miss = _not_found(targets)
    if miss:
        deb = sorted({SYSLIB_PKGS[_lib_key(m)][1] for m in miss if _lib_key(m) in SYSLIB_PKGS})
        warn(T("Chybí systémové knihovny: " + ", ".join(sorted(miss)) + (f" – Ubuntu/Debian: sudo apt install {' '.join(deb)}" if deb else ""),
               "Missing system libraries: " + ", ".join(sorted(miss)) + (f" – Ubuntu/Debian: sudo apt install {' '.join(deb)}" if deb else "")))
    else:
        ok(T("systémové knihovny doplněny do .venv/sys_libs", "system libraries added to .venv/sys_libs"))


# ------------------------------------------------------------------ commands
def pick_modules(args) -> list[str]:
    keys = list(args.modules)
    if args.profile:
        keys += PROFILES[args.profile]
    if not keys:
        if QUIET or YES or not sys.stdin or not sys.stdin.isatty():
            keys = PROFILES["standard"]
        else:
            print(T("\nVyberte rozsah instalace:", "\nChoose what to install:"))
            names = list(PROFILES)
            for i, n in enumerate(names, 1):
                ks = closure(PROFILES[n])
                print(f"  {i}) {n:9s} ~{sum(size_of(k) for k in ks):5d} MB   {', '.join(PROFILES[n])}")
            print(f"  {len(names) + 1}) " + T("vlastní výběr modulů", "custom selection"))
            c = input(T("Volba [2]: ", "Choice [2]: ")).strip() or "2"
            if c.isdigit() and 1 <= int(c) <= len(names):
                keys = PROFILES[names[int(c) - 1]]
            else:
                keys = ["core"]
                for k in MODS:
                    if k != "core" and ask(f"{k}: {mtitle(k)} (~{size_of(k)} MB)", k in PROFILES["standard"]):
                        keys.append(k)
    return closure(["core"] + keys)


def installed_version() -> str | None:
    """Version of the installed program: from the installation state; a state without a version = release 1.0.0; without a state = nothing."""
    st = read_state()
    if not st.get("modules"):
        return None
    return st.get("version") or upgrade.LEGACY_VERSION


def install_kind(old_ver: str | None, new_ver: str) -> str:
    """Kind of installer run: "install" (nothing installed), "repair" (the same version again), "update", "downgrade"."""
    if not old_ver:
        return "install"
    o, n = upgrade.parse_version(old_ver), upgrade.parse_version(new_ver)
    return "repair" if o == n else ("update" if o < n else "downgrade")


def upgrade_files(old_ver: str, new_ver: str, reported: set[str] | None = None) -> None:
    """Obsolete program files (old manifest − new manifest) – only unchanged ones, never user data (installer/upgrade.py).
    `reported` = the modified files already listed (shared by migration 1.0.1, which also goes through the manifests of older releases).
    """
    new = upgrade.read_manifest(ROOT / upgrade.PACKAGE_MANIFEST)
    old = upgrade.previous_manifest(ROOT, VENV, old_ver)
    if new is None:
        info(T("balíček nemá seznam souborů (installer/manifest.json) – zastaralé soubory se nemažou",
               "the package has no file list (installer/manifest.json) – obsolete files are not removed"))
        return
    if old is None:
        info(T(f"seznam souborů verze {old_ver} není k dispozici – zastaralé soubory se nemažou",
               f"no file list for version {old_ver} – obsolete files are not removed"))
        return
    removed, kept = upgrade.remove_obsolete(ROOT, upgrade.obsolete_files(old, new), say=info, reported=reported)
    if not removed and not kept:
        ok(T("žádné zastaralé soubory", "no obsolete files"))


def save_manifest() -> None:
    """Manifest of the installed version into .venv – the next update uses it to find the files that the new version no longer has."""
    new = upgrade.read_manifest(ROOT / upgrade.PACKAGE_MANIFEST)
    if new is not None:
        upgrade.write_manifest(VENV / upgrade.MANIFEST_NAME, new)


def migration_context(keys: list[str]) -> "migrations.Context":
    return migrations.Context(root=ROOT, venv=VENV, vpy=VPY, modules=keys, say=info, warn=warn,
                              setup=sys.modules[__name__])


def cmd_update(args) -> int:
    return cmd_install(args, update=True)


def cmd_install(args, update: bool = False) -> int:
    """install: adds modules (removes only with `remove`, with --exact exactly the selection); over an older version it also performs an update.
    update: modules of the last installation (+ those given, if any) – after unpacking a new version over the old one.
    """
    st = read_state()
    old_ver = installed_version()
    new_ver = upgrade.program_version(ROOT)
    if args.exact:
        keys = pick_modules(args)
    elif update and old_ver and not (args.modules or args.profile):
        keys = closure(st["modules"])
    else:
        keys = closure(st.get("modules", []) + pick_modules(args))      # install adds; only `remove` removes
    pick_language(args)
    if old_ver and upgrade.parse_version(old_ver) > upgrade.parse_version(new_ver):
        if not args.force:
            die(T(f"Nainstalovaná verze {old_ver} je novější než tento balíček ({new_ver}). Návrat ke starší verzi jen "
                  "s --force (prostředí .venv se vytvoří znovu).",
                  f"The installed version {old_ver} is newer than this package ({new_ver}). Downgrade only with --force "
                  "(the .venv environment is recreated)."), 2)
        warn(T(f"návrat ke starší verzi {old_ver} → {new_ver} (--force)", f"downgrade {old_ver} → {new_ver} (--force)"))
    total = sum(size_of(k) for k in keys)
    kind = install_kind(old_ver, new_ver)
    mods = T(f"\nModuly: {', '.join(keys)} (~{total} MB)", f"\nModules: {', '.join(keys)} (~{total} MB)")
    if kind == "repair":                    # the same version again (setup.exe / update over a finished installation)
        print(T(f"\nFEA Studio – opravná instalace verze {new_ver} v {ROOT}", f"\nFEA Studio – repair installation of version "
                f"{new_ver} in {ROOT}") + mods)
    elif kind == "downgrade":
        print(T(f"\nFEA Studio – návrat ke starší verzi {old_ver} → {new_ver} v {ROOT}",
                f"\nFEA Studio – downgrading {old_ver} → {new_ver} in {ROOT}") + mods)
    elif kind == "update":
        print(T(f"\nFEA Studio – aktualizace {old_ver} → {new_ver} v {ROOT}", f"\nFEA Studio – updating {old_ver} → {new_ver} in {ROOT}")
              + mods)
    else:
        print(T(f"\nFEA Studio {new_ver} – instalace do {ROOT}", f"\nFEA Studio {new_ver} – installing into {ROOT}") + mods)
    _log(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} {'update' if update else 'install'} {old_ver or '-'} → {new_ver}: "
         + " ".join(keys))
    if not IS_WIN and platform.libc_ver()[0] == "glibc" and tuple(int(x) for x in platform.libc_ver()[1].split(".")[:2]) < (2, 35):
        warn(T(f"glibc {platform.libc_ver()[1]} je starší než 2.35 – balíčky OCP / PySide6 nemusí jít nainstalovat",
               f"glibc {platform.libc_ver()[1]} is older than 2.35 – the OCP / PySide6 wheels may not install"))
    ctx = migration_context(keys)
    if old_ver:
        step(T(f"Opravná instalace verze {new_ver} – soubory programu", f"Repair installation of version {new_ver} – program files")
             if kind == "repair" else T(f"Aktualizace z verze {old_ver}", f"Update from version {old_ver}"))
        upgrade_files(old_ver, new_ver, ctx.kept_reported)
        migrations.run(ctx, old_ver, new_ver, "pre")
    ensure_venv(args.force)
    step(T("Balíčky Pythonu", "Python packages"))
    sync_packages(keys)
    restore_pruned_qt(keys)
    write_pth()
    vc_runtime_local()
    if old_ver:
        migrations.run(ctx, old_ver, new_ver, "post")
    for k in [k for k in st.get("modules", []) if k not in keys]:      # --exact: removed modules
        remove_ext(k)
    step(T("Prořezání (zbytečné soubory balíčků)", "Slimming (unneeded package files)"))
    if not args.no_slim:
        slim.slim_all(VENV, site_packages(), keys, say=info)
    step(T("Vnější komponenty", "External components"))
    for k in keys:
        for e in MODS[k].get("ext", []):
            if e in EXT:
                try:
                    EXT[e]()
                except Exception as ex:  # noqa: BLE001
                    warn(f"{e}: {ex}")
    if "pardiso" in keys and not args.no_slim:
        slim.prune_mkl(VENV, say=info)
    write_state(keys, new_ver)
    save_manifest()
    launchers()
    ensure_sys_libs(keys)
    step(T("Zástupci a kontrola", "Shortcuts and check"))
    if not args.no_shortcuts and "gui" in keys:                 # shortcuts make sense only with the graphical interface
        try:
            make_shortcuts()
        except Exception as ex:  # noqa: BLE001
            warn(T(f"zástupci: {ex}", f"shortcuts: {ex}"))
    if "mcp" in keys and not args.no_mcp:
        r = run([VPY, "-m", "fea_core.mcp.setup"], check=False)
        (ok if r.returncode == 0 else warn)(T("MCP server zaregistrován u nalezených klientů", "MCP server registered with the clients found")
                                            if r.returncode == 0 else T("MCP: u některých klientů doplňte ručně", "MCP: configure some clients manually"))
    if "gui" in keys and not args.no_fusion and fusion_dir() is not None:      # the button "Send to FEA Studio"
        try:
            install_fusion_addin()
        except Exception as ex:  # noqa: BLE001
            warn(f"Fusion: {ex}")
    compile_project()
    if not args.no_check:
        check_install(keys, not args.quick_check and "accel" in keys)
    if not args.keep_downloads:
        slim.rmtree(DL)
        if INDEX_ARGS and INDEX_ARGS[0] == "--offline":      # the bundled packages are installed – the space on the disk back
            slim.rmtree(OFFLINE.parent)
    print()
    if PROBLEMS:
        print(T("Instalace dokončena s upozorněními:", "Installation finished with warnings:"))
        for p in PROBLEMS:
            print(f"  - {p}")
    else:
        py = r".venv\Scripts\python" if IS_WIN else ".venv/bin/python"
        if "gui" in keys:
            start = ("zástupce „FEA Studio“ nebo „FEA Studio.bat“" if IS_WIN else "příkaz ./fea-studio nebo položka FEA Studio v nabídce aplikací")
            start_en = ("shortcut “FEA Studio” or “FEA Studio.bat”" if IS_WIN else "./fea-studio or the FEA Studio menu entry")
        else:
            start = f"{py} -m fea_core run studie.json (grafické rozhraní: python -m fea_core.modules install gui)"
            start_en = f"{py} -m fea_core run study.json (graphical interface: python -m fea_core.modules install gui)"
        done = {"repair": T(f"Opravná instalace verze {new_ver} dokončena.", f"Repair installation of version {new_ver} finished."),
                "update": T(f"Aktualizace na verzi {new_ver} dokončena.", f"Update to version {new_ver} finished."),
                "downgrade": T(f"Návrat k verzi {new_ver} dokončen.", f"Downgrade to version {new_ver} finished.")
                }.get(kind, T("Instalace dokončena.", "Installation finished."))
        print(done, T("Spuštění: " + start, "Start: " + start_en))
    print(T("Další moduly: python -m fea_core.modules install <modul>", "More modules: python -m fea_core.modules install <module>"))
    return 0


def compile_project() -> None:
    # -f: after an update the bytecode of a source with the same time and size (files from the package carry the release time) would stay old
    run([VPY, "-m", "compileall", "-f", "-q", "-j", "0", str(ROOT / "fea_core"), str(ROOT / "fea_gui")], check=False,
        quiet=True)


def check_install(keys: list[str], full: bool) -> None:
    if "gui" in keys:       # the check calculation does not load the GUI – a broken Qt would otherwise be found only at start
        r = run([VPY, "-c", "import PySide6.QtWidgets, PySide6.QtSvg, OCP.AIS"], check=False, quiet=True)
        (ok if r.returncode == 0 else warn)(T("grafické rozhraní (Qt, OCP) se načte", "graphical interface (Qt, OCP) loads")
                                            if r.returncode == 0 else
                                            T("grafické rozhraní nejde načíst (Qt) – spusťte instalaci s --force",
                                              "graphical interface does not load (Qt) – run the installation with --force"))
    info(T("kontrolní výpočet V1…", "check calculation V1…"))
    cmd = [VPY, "-m", "fea_core.warmup"] + ([] if full and "accel" in keys else ["--quick"])
    r = run(cmd, check=False)
    (ok if r.returncode == 0 else warn)(T("kontrolní výpočty prošly", "check calculations passed") if r.returncode == 0
                                        else T(f"kontrolní výpočty selhaly (kód {r.returncode})", f"check calculations failed (code {r.returncode})"))


def cmd_remove(args) -> int:
    st = read_state()["modules"]
    if not st:
        die(T("Žádné moduly nejsou nainstalovány.", "No modules are installed."))
    drop = set(args.modules)
    if "core" in drop:
        die(T("Modul core nejde odebrat (odinstalace: fea_setup.py uninstall).", "The core module cannot be removed (use: fea_setup.py uninstall)."))
    # removing a module also removes the modules that depend on it
    changed = True
    while changed:
        changed = False
        for k in st:
            if k not in drop and any(r in drop for r in MODS[k].get("requires", [])):
                drop.add(k)
                changed = True
    keep = [k for k in st if k not in drop]
    print(T(f"Odebírám: {', '.join(sorted(drop & set(st)))}", f"Removing: {', '.join(sorted(drop & set(st)))}"))
    sync_packages(keep)
    for k in drop:
        remove_ext(k)
    write_state(keep)
    ok(T(f"zbývají moduly: {', '.join(keep)}", f"remaining modules: {', '.join(keep)}"))
    return 0


def cmd_list(_args) -> int:
    st = read_state()["modules"]
    print(T("Moduly (✓ nainstalováno):\n", "Modules (✓ installed):\n"))
    for k in MODS:
        print(f"  {'✓' if k in st else '–'} {k:8s} ~{size_of(k):4d} MB  {mtitle(k)}")
    print()
    for n, ks in PROFILES.items():
        print(f"  profil {n}: {', '.join(ks)}")
    return 0


def cmd_mpi(_args) -> int:
    """Only Microsoft MPI (Windows) for the NGSolve + hypre solver – the command itself is the consent to install it (UAC)."""
    global MSMPI
    if not IS_WIN:
        print(T("Na Linuxu je MPI (MPICH) součástí modulu hypre: python -m fea_core.modules install hypre",
                "On Linux MPI (MPICH) is part of the hypre module: python -m fea_core.modules install hypre"))
        return 0
    MSMPI = True
    install_ext_mpi()
    slim.rmtree(DL)
    return 1 if PROBLEMS else 0


def cmd_shortcuts(_args) -> int:
    make_shortcuts()
    return 0


def cmd_fusion(args) -> int:
    if args.remove:
        remove_fusion_addin()
    else:
        install_fusion_addin()
    return 0


def cmd_slim(_args) -> int:
    st = read_state()["modules"]
    slim.slim_all(VENV, site_packages(), st, say=info)
    if "pardiso" in st:
        slim.prune_mkl(VENV, say=info)
    return 0


def cmd_doctor(_args) -> int:
    st = read_state()
    print(f"FEA Studio {upgrade.program_version(ROOT)} {ROOT}\nplatform {PLAT}, Python {platform.python_version()} ({sys.executable})")
    print(T("nainstalovaná verze: ", "installed version: ") + (installed_version() or "-")
          + (T(", seznam souborů uložen", ", file list stored") if (VENV / upgrade.MANIFEST_NAME).exists() else ""))
    print(T("moduly: ", "modules: ") + (", ".join(st.get("modules", [])) or "-"))
    if VPY.exists():
        sys.stdout.flush()                                         # listing of the helper processes only after the header
        run([VPY, "-m", "fea_core.modules"], check=False)
        print(flush=True)
        run([VPY, "-m", "fea_core.solvers"], check=False)          # solvers: what works, what is missing, how to install it
        mb = slim.dir_size(VENV) / 1e6
        print(f".venv: {mb:.0f} MB, tools: {slim.dir_size(TOOLS) / 1e6:.0f} MB")
    else:
        print(T(".venv neexistuje – spusťte install.", ".venv does not exist – run install."))
    return 0


CLEAN_SKIP = (".git", "scratch", ".claude", "dist")      # development folders – the uninstallation cleanup does not touch them


def _is_link(p: Path) -> bool:
    """Symbolic link or junction (generally a reparse point) – the cleanup does not enter such a folder (os.walk
    in Python < 3.12 walks a junction on Windows as an ordinary folder).
    """
    try:
        st = os.lstat(p)
    except OSError:
        return True
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


def remove_empty_dirs(root: Path, skip: tuple = (".venv", "tools") + CLEAN_SKIP) -> list[Path]:
    """Deletes empty program folders bottom-up (they remain after a deleted bytecode and after an update from a zip – the Inno uninstaller
    deletes only the folders it created itself). Only inside `root` (the root itself stays), without the top-level `skip` folders, without
    entering links / junctions; a folder with any file (also in a subfolder) stays – files are never deleted.
    """
    removed: list[Path] = []

    def walk(d: Path) -> bool:                      # True = the folder is empty (after cleaning the subfolders)
        try:
            entries = list(os.scandir(d))
        except OSError:
            return False
        empty = True
        for e in entries:
            p = Path(e.path)
            if e.is_dir(follow_symlinks=False) and not _is_link(p) and walk(p):
                try:
                    p.rmdir()
                    removed.append(p)
                    continue
                except OSError:
                    pass
            empty = False
        return empty

    for top in sorted(root.iterdir()):
        if top.is_dir() and top.name not in skip and not _is_link(top) and walk(top):
            try:
                top.rmdir()
                removed.append(top)
            except OSError:
                pass
    return removed


def cmd_uninstall(args) -> int:
    remove_shortcuts()
    remove_fusion_addin()
    if args.pycache or args.all:            # bytecode of the program (the Inno uninstaller deletes only the files it wrote itself)
        n = 0
        for top in ROOT.iterdir():
            if top.is_dir() and top.name not in (".venv", "tools") + CLEAN_SKIP and not _is_link(top):
                for d in [top] if top.name == "__pycache__" else list(top.rglob("__pycache__")):
                    slim.rmtree(d)
                    n += 1
        info(T(f"složky __pycache__ smazány ({n})", f"__pycache__ folders deleted ({n})"))
    if args.all:
        for d in (VENV, TOOLS):
            slim.rmtree(d)
            if d.exists():                  # Windows: the files of the running Python (.venv, tools/python) are locked
                warn(T(f"{d} nejde smazat celá (část souborů používá běžící program, např. Python tohoto instalátoru) – "
                       "smažte ji ručně po jeho skončení",
                       f"{d} cannot be deleted completely (some files are in use, e.g. by the Python running this installer) – "
                       "delete it manually afterwards"))
            else:
                info(T(f"smazáno: {d}", f"deleted: {d}"))
    if args.pycache or args.all:
        gone = remove_empty_dirs(ROOT, CLEAN_SKIP if args.all else (".venv", "tools") + CLEAN_SKIP)
        if gone:
            info(T(f"prázdné složky programu smazány ({len(gone)})", f"empty program folders deleted ({len(gone)})"))
    if args.all:
        info(T(f"Složku {ROOT} lze nyní smazat ručně (vlastní soubory .fea zůstávají).",
               f"You can now delete {ROOT} manually (your own .fea files are untouched)."))
    return 0


def main(argv: list[str] | None = None) -> int:
    global QUIET, YES, MSMPI
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true", help=T("bez dotazů", "no questions"))
    ap.add_argument("--yes", "-y", action="store_true", help=T("odpovídat výchozími volbami", "answer with the default choices"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("install", help=T("nainstalovat moduly (nad starší verzí zároveň aktualizace)", "install modules (over an older version also an update)"))
    pu_ = sub.add_parser("update", help=T("aktualizace po rozbalení nové verze: migrace, zastaralé soubory, moduly jako minule",
                                             "update after unpacking a new version: migrations, obsolete files, modules as last time"))
    for p in (pi, pu_):
        p.add_argument("modules", nargs="*")
        p.add_argument("--lang", choices=["cs", "en"],
                       help=T("jazyk programu (jinak dotaz při první instalaci, později zůstává)",
                              "program language (otherwise asked at the first installation, kept later)"))
        p.add_argument("--profile", choices=list(PROFILES))
        p.add_argument("--exact", action="store_true", help=T("moduly přesně podle výběru (ostatní odebrat) – setup.exe", "modules exactly as selected (remove the others) – setup.exe"))
        p.add_argument("--force", action="store_true", help=T("znovu vytvořit .venv; povolí návrat ke starší verzi", "recreate .venv; allows a downgrade"))
        p.add_argument("--no-slim", action="store_true", help=T("neprořezávat balíčky", "do not slim the packages"))
        p.add_argument("--no-shortcuts", action="store_true")
        p.add_argument("--no-mcp", action="store_true")
        p.add_argument("--no-fusion", action="store_true", help=T("neinstalovat doplněk do Autodesk Fusion", "do not install the Autodesk Fusion add-in"))
        p.add_argument("--no-check", action="store_true", help=T("bez kontrolního výpočtu", "no check calculation"))
        p.add_argument("--quick-check", action="store_true",
                       help=T("jen výpočet V1 (jinak se s modulem accel předkompiluje Numba: únava, tisk, obrázky – několik minut)",
                              "only the V1 calculation (otherwise Numba is precompiled with the accel module: fatigue, printing, images – several minutes)"))
        p.add_argument("--keep-downloads", action="store_true")
        p.add_argument("--msmpi", action="store_true",
                       help=T("nainstalovat i Microsoft MPI (modul hypre, Windows, UAC) bez dotazu",
                              "also install Microsoft MPI (hypre module, Windows, UAC) without asking"))
    pr = sub.add_parser("remove", help=T("odebrat moduly", "remove modules"))
    pr.add_argument("modules", nargs="+")
    sub.add_parser("list")
    sub.add_parser("slim")
    sub.add_parser("shortcuts", help=T("znovu vytvořit zástupce a příponu .fea pro tuto složku", "recreate the shortcuts and the .fea association for this folder"))
    sub.add_parser("doctor")
    sub.add_parser("mpi", help=T("nainstalovat jen Microsoft MPI pro řešič NGSolve + hypre (Windows, UAC, správce)",
                                         "install only Microsoft MPI for the NGSolve + hypre solver (Windows, UAC, administrator)"))
    pf = sub.add_parser("fusion", help=T("nainstalovat (--remove odebrat) doplněk „Odeslat do FEA Studia“ do Autodesk Fusion",
                                               "install (--remove: remove) the “Send to FEA Studio” add-in into Autodesk Fusion"))
    pf.add_argument("--remove", action="store_true")
    pu = sub.add_parser("uninstall", help=T("odebrat zástupce, příponu .fea a doplněk Fusion", "remove the shortcuts, the .fea association and the Fusion add-in"))
    pu.add_argument("--all", action="store_true", help=T("smazat i .venv a tools (a bajtkód a prázdné složky jako --pycache)",
                                                    "also delete .venv and tools (and bytecode and empty folders as --pycache)"))
    pu.add_argument("--pycache", action="store_true", help=T("smazat i bajtkód programu (__pycache__) a prázdné složky programu",
                                                        "also delete the program bytecode (__pycache__) and empty program folders"))
    argv = list(sys.argv[1:] if argv is None else argv)
    glob = {"--quiet", "--yes", "-y"} & set(argv)             # global switches may also stand after the command
    argv = [a for a in argv if a not in ("--quiet", "--yes", "-y")]
    args = ap.parse_args(argv)
    QUIET, YES = args.quiet or "--quiet" in glob, args.yes or bool({"--yes", "-y"} & glob)
    MSMPI = bool(getattr(args, "msmpi", False))
    global LOG
    if args.cmd in ("install", "update", "remove", "mpi"):
        LOG = ROOT / "install.log"
    fn = {"install": cmd_install, "update": cmd_update, "remove": cmd_remove, "list": cmd_list, "slim": cmd_slim, "shortcuts": cmd_shortcuts, "doctor": cmd_doctor,
          "fusion": cmd_fusion, "uninstall": cmd_uninstall, "mpi": cmd_mpi}[args.cmd]
    try:
        return fn(args)
    except KeyboardInterrupt:
        die(T("Přerušeno.", "Interrupted."), 130)
    except RuntimeError as e:
        die(str(e))
    except Exception:  # noqa: BLE001  (the setup.exe console window closes after an error – the output stays in install.log)
        import traceback
        _log(traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
