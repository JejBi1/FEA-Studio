"""Migration steps of a FEA Studio update (run by `fea_setup.py update` / `install` over an older installation).

Each step: the version in which it was added, the phase ("pre" = before the package synchronisation, "post" = after it), a description and the function `fn(ctx)`.
The steps with a version > the previously installed one and ≤ the new one run, sorted by version and phase. The steps MUST be idempotent
(an update interrupted in the middle runs again completely). `ctx` = `Context` below (paths, modules, output, installer).

A new step = add a function and an entry to MIGRATIONS (version = the program version that brings it).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import upgrade
from setup_lang import T


@dataclass
class Context:
    root: Path
    venv: Path
    vpy: Path
    modules: list[str]
    say: Callable[[str], None] = print
    warn: Callable[[str], None] = print
    setup: object = None                 # module fea_setup (uv, site_packages…) – may be missing in the tests
    done: list[str] = field(default_factory=list)
    kept_reported: set[str] = field(default_factory=set)   # modified obsolete files already listed (list once)


@dataclass(frozen=True)
class Migration:
    version: str
    phase: str                           # "pre" | "post"
    title: str
    fn: Callable[[Context], None]


# ------------------------------------------------------------------ 1.0.1
def remove_old_release_files(ctx: Context) -> None:
    """Files of older releases (0.1.0, 1.0.0) that the new version does not have – also from installations without a saved manifest
    (e.g. the moved tests/gui_*.py, fea_core/contacts.py replaced by the package contacts/). Only unchanged files.
    """
    new = upgrade.read_manifest(ctx.root / upgrade.PACKAGE_MANIFEST)
    if new is None:
        ctx.say(T("balíček nemá seznam souborů – staré soubory se nemažou",
                  "the package has no file list – old files are not removed"))
        return
    old: dict[str, str] = {}
    for m in upgrade.history_manifests(ctx.root).values():
        old.update(upgrade.obsolete_files(m, new))
    upgrade.remove_obsolete(ctx.root, old, say=ctx.say, reported=ctx.kept_reported)
    orphans = upgrade.orphan_bytecode(ctx.root)
    for p in orphans:
        p.unlink(missing_ok=True)
    if orphans:
        ctx.say(T(f"osiřelý bajtkód smazán ({len(orphans)} souborů)", f"orphaned bytecode deleted ({len(orphans)} files)"))


# packages whose __init__.py the installer deletes on purpose (installer/slim.py)
_SLIMMED_TOP = {"vtkmodules"}


def broken_packages(site: Path) -> list[str]:
    """Installed packages whose `<package>/__init__.py` from their RECORD is missing – the state after uninstalling a package with
    shared files (PySide6 + PySide6_Addons from an old environment deleted PySide6/__init__.py of PySide6_Essentials).
    Returns "name==version" to reinstall.
    """
    import csv
    out = []
    for di in sorted(site.glob("*.dist-info")):
        rec = di / "RECORD"
        if not rec.is_file():
            continue
        with open(rec, encoding="utf-8", newline="") as f:
            rows = [r[0] for r in csv.reader(f) if r]
        inits = [r for r in rows if r.count("/") == 1 and r.endswith("/__init__.py") and r.split("/")[0] not in _SLIMMED_TOP]
        if any(not (site / r).exists() for r in inits):
            name, _, ver = di.name[: -len(".dist-info")].rpartition("-")
            out.append(f"{name}=={ver}")
    return out


def repair_shared_package_files(ctx: Context) -> None:
    """Repair of 1.0.0 installations after `uv pip sync` (PySide6): reinstalls the packages with missing files and checks that Qt loads."""
    if ctx.setup is None or not ctx.vpy.exists():
        return
    site = ctx.setup.site_packages()
    broken = broken_packages(site)
    if broken:
        ctx.say(T("přeinstalace balíčků s chybějícími soubory: ", "reinstalling packages with missing files: ") + ", ".join(broken))
        ctx.setup.uv("pip", "install", "--reinstall", "--no-deps", *broken, "--python", str(ctx.vpy))
    if "gui" in ctx.modules:
        r = subprocess.run([str(ctx.vpy), "-c", "import PySide6.QtWidgets"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            pins = [p for p in (_installed_pin(site, n) for n in ("PySide6_Essentials", "shiboken6")) if p]
            ctx.say(T("Qt nejde načíst – přeinstalace ", "Qt does not load – reinstalling ") + ", ".join(pins))
            if pins:
                ctx.setup.uv("pip", "install", "--reinstall", "--no-deps", *pins, "--python", str(ctx.vpy))
            r = subprocess.run([str(ctx.vpy), "-c", "import PySide6.QtWidgets"], capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                ctx.warn(T("grafické rozhraní (Qt) stále nejde načíst – spusťte instalaci s --force: ",
                           "the graphical interface (Qt) still does not load – run the installation with --force: ")
                         + (r.stderr.strip().splitlines() or ["?"])[-1])


def _installed_pin(site: Path, name: str) -> str | None:
    key = name.lower().replace("-", "_")
    for di in site.glob("*.dist-info"):
        n, _, ver = di.name[: -len(".dist-info")].rpartition("-")
        if n.lower().replace("-", "_") == key:
            return f"{n}=={ver}"
    return None


MIGRATIONS: list[Migration] = [
    Migration("1.0.1", "pre", T("smazání souborů starších vydání a osiřelého bajtkódu",
                               "removal of files from older releases and orphaned bytecode"), remove_old_release_files),
    Migration("1.0.1", "post", T("oprava souborů sdílených balíčky (PySide6) a kontrola načtení Qt",
                                "repair of files shared by packages (PySide6) and Qt load check"), repair_shared_package_files),
]


def pending(old_version: str, new_version: str, phase: str | None = None,
            migrations: list[Migration] | None = None) -> list[Migration]:
    """Steps for the transition old → new (old < step version ≤ new), sorted by version."""
    lo, hi = upgrade.parse_version(old_version), upgrade.parse_version(new_version)
    ms = MIGRATIONS if migrations is None else migrations
    sel = [m for m in ms if lo < upgrade.parse_version(m.version) <= hi and (phase is None or m.phase == phase)]
    return sorted(sel, key=lambda m: (upgrade.parse_version(m.version), m.phase != "pre"))


def run(ctx: Context, old_version: str, new_version: str, phase: str,
        migrations: list[Migration] | None = None) -> list[str]:
    """Runs the steps of the phase; an error of a step is reported (ctx.warn), the next steps continue. Returns the descriptions of the steps performed."""
    done = []
    for m in pending(old_version, new_version, phase, migrations):
        ctx.say(T(f"migrace {m.version}: {m.title}", f"migration {m.version}: {m.title}"))
        try:
            m.fn(ctx)
            done.append(f"{m.version}: {m.title}")
        except Exception as ex:  # noqa: BLE001
            ctx.warn(T(f"migrace {m.version} ({m.title}) selhala: {ex}", f"migration {m.version} ({m.title}) failed: {ex}"))
    ctx.done.extend(done)
    return done
