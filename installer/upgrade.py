"""Update of a FEA Studio installation – version and list of the program files (manifest). Only the standard library.

The installation remembers (`.venv/fea_modules.json` → `version`, `.venv/fea_install_manifest.json`) the version and the program files
it installed. A package carries the manifest of its version (`installer/manifest.json`, created by `make_package.py`); the manifests
of older releases without their own record are in `installer/manifests/<version>.json`.

An obsolete file (it was in the old manifest, not in the new one) is deleted only when:
  - it lies in the program folder and is not in a protected part (`.venv`, `tools`, `scratch`, `dist`, `.git`, user data…),
  - it is not a project `*.fea` (nor an example – the user may have saved work into it),
  - its contents have not changed from the released one (the SHA-256 digest matches) – a modified file stays and is listed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from setup_lang import T

MANIFEST_NAME = "fea_install_manifest.json"         # in .venv – the manifest of the INSTALLED version
PACKAGE_MANIFEST = Path("installer") / "manifest.json"
HISTORY_DIR = Path("installer") / "manifests"
LEGACY_VERSION = "1.0.0"                             # installation without a version record (release 1.0.0 and older)

PROTECTED_PREFIX = (".venv/", "tools/", "scratch/", "dist/", ".git/", ".claude/", "verification/out", "user/",
                    "data_user/", "projects/", "projekty/")
PROTECTED_SUFFIX = (".fea",)
# data that the package does not carry, but the user has them (PragTic – L2, downloaded by pragtic_fetch): the update does not delete them
PROTECTED_FILES = ("fea_core/fatigue/data/pragtic_vmat.json",)
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


# ------------------------------------------------------------------ verze
def parse_version(v: str) -> tuple:
    """"1.0.1" → (1, 0, 1); the suffix ("1.1.0rc1") is ignored, missing parts = 0."""
    parts = [int(m) for m in re.findall(r"\d+", (v or "").split("+")[0])[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


def program_version(root: Path) -> str:
    """Version of the program in the folder (without import – fea_core/__init__.py is only read)."""
    m = re.search(r'__version__ = "(.+?)"', (root / "fea_core" / "__init__.py").read_text(encoding="utf-8"))
    return m.group(1) if m else "0.0.0"


# ------------------------------------------------------------------ manifest
def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def manifest_of_dir(root: Path, version: str, skip_prefix: tuple = ("tools/",)) -> dict:
    """Manifest of an unpacked package: {"version", "files": {relative path: digest}}."""
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not SKIP_DIRS & set(p.parts):
            rel = p.relative_to(root).as_posix()
            if not rel.startswith(skip_prefix) and rel != PACKAGE_MANIFEST.as_posix():
                files[rel] = file_hash(p)
    return {"version": version, "files": files}


def manifest_of_zip(path: Path, version: str, skip_prefix: tuple = ("tools/",)) -> dict:
    """Manifest of a released zip package (FEAStudio-<version>/…) – for installer/manifests/<version>.json."""
    import zipfile
    files = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith("/") or "/" not in n:
                continue
            rel = n.split("/", 1)[1]
            if rel.startswith(skip_prefix) or rel == PACKAGE_MANIFEST.as_posix() or SKIP_DIRS & set(rel.split("/")):
                continue
            files[rel] = bytes_hash(z.read(n))
    return {"version": version, "files": files}


def read_manifest(path: Path) -> dict | None:
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict) or not isinstance(m.get("files"), dict):
        return None
    return m


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")


def history_manifests(root: Path) -> dict[str, dict]:
    """Manifests of older releases bundled with the installer (version → manifest)."""
    out = {}
    d = root / HISTORY_DIR
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        m = read_manifest(p)
        if m:
            out[m.get("version") or p.stem] = m
    return out


def previous_manifest(root: Path, venv: Path, old_version: str) -> dict | None:
    """Manifest of the installed version: saved at the installation, otherwise the manifest of the release from installer/manifests."""
    m = read_manifest(venv / MANIFEST_NAME)
    if m:
        return m
    return history_manifests(root).get(old_version)


# ------------------------------------------------------------------ deleting obsolete files
def is_protected(rel: str) -> bool:
    if not rel or rel.startswith(("/", "\\")) or ":" in rel or ".." in Path(rel).parts:
        return True
    low = rel.replace("\\", "/").lower()
    return low.startswith(PROTECTED_PREFIX) or low.endswith(PROTECTED_SUFFIX) or low in PROTECTED_FILES


def obsolete_files(old: dict, new: dict) -> dict[str, str]:
    """Files of the old manifest that the new one does not have (path → digest of the released form)."""
    newf = {k.lower() for k in new.get("files", {})}         # Windows: the letter case in the path does not matter
    return {k: v for k, v in old.get("files", {}).items() if k.lower() not in newf}


def remove_obsolete(root: Path, old_files: dict[str, str], say=print,
                    reported: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Deletes the obsolete program files (see the module). Returns (deleted, kept modified). `reported` = the set of kept files
    already listed (shared between the calls of one update – a modified file is listed only once).
    """
    removed, kept = [], []
    root = root.resolve()
    dirs = set()
    for rel, digest in sorted(old_files.items()):
        if is_protected(rel):
            continue
        p = (root / rel)
        try:
            p.resolve().relative_to(root)
        except ValueError:
            continue                                           # a link outside the program folder
        if not p.is_file():
            continue
        if digest and file_hash(p) != digest:
            kept.append(rel)
            continue
        p.unlink()
        removed.append(rel)
        dirs.add(p.parent)
        cache = p.parent / "__pycache__"                       # bytecode of a deleted module
        if p.suffix == ".py" and cache.is_dir():
            for c in cache.glob(p.stem + ".*.pyc"):
                c.unlink(missing_ok=True)
    for d in sorted(dirs, key=lambda x: len(x.parts), reverse=True):  # empty folders after the deleted files
        while d != root and d.is_dir():
            rest = [x for x in d.iterdir() if not (x.name == "__pycache__" and x.is_dir() and not any(x.iterdir()))]
            if rest:
                break
            for x in d.iterdir():
                x.rmdir()
            d.rmdir()
            d = d.parent
    if removed:
        say(T(f"zastaralé soubory programu smazány ({len(removed)}): ", f"obsolete program files deleted ({len(removed)}): ")
            + ", ".join(removed[:8]) + (" …" if len(removed) > 8 else ""))
    new_kept = [k for k in kept if reported is None or k not in reported]
    if reported is not None:
        reported.update(kept)
    if new_kept:
        say(T(f"upravené zastaralé soubory ponechány ({len(new_kept)}): ", f"modified obsolete files kept ({len(new_kept)}): ")
            + ", ".join(new_kept[:8]) + (" …" if len(new_kept) > 8 else ""))
    return removed, kept


def orphan_bytecode(root: Path, dirs=("fea_core", "fea_gui", "installer", "examples", "verification", "tests", "adapters",
                                      "stepview")) -> list[Path]:
    """`__pycache__/x.cpython-311.pyc` without the source `x.py` (a module deleted in the new version)."""
    out = []
    for top in dirs:
        base = root / top
        if not base.is_dir():
            continue
        for pyc in base.rglob("__pycache__/*.pyc"):
            src = pyc.parent.parent / (pyc.name.split(".")[0] + ".py")
            if not src.exists():
                out.append(pyc)
    return out
