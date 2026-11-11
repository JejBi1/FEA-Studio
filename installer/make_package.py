#!/usr/bin/env python3
"""Builds the distribution packages of FEA Studio for Windows and Linux (run on the development machine in the venv of the project):

    python installer/make_package.py [--ref HEAD] [--out dist] [--no-docs] [--no-examples] [--examples-no-results] [--no-linux-ccx]
                                     [--iscc path] [--no-setup] [--no-run] [--offline]

Output in `dist/`:
    FEAStudio-<version>.zip                 Windows (and universal): program files + installer, Python 3.11, uv.exe, CalculiX ccx_static.exe
    FEAStudio-<version>.tar.gz              Linux: the program files with execute permissions and the prebuilt CalculiX ccx_spooles (if available)
                                          (ccx_i4 = CalculiX + Intel MKL PARDISO is never in a package: GPL-2.0 does not allow conveying
                                          it – the installer gets / builds it on the user's machine with the pardiso module)
    FEAStudio-<version>-setup.exe           Windows installer (Inno Setup 6: ISCC from tools/innosetup – installer/get_innosetup.py,
                                          from PATH or --iscc); choice of modules, update over an old installation, bundled uv.exe
    FEAStudio-<version>-offline-setup.exe   (--offline) setup.exe plus all Python packages of the standard profile
                                          (tools/offline/wheels, digests from PyPI) – installation without internet
    FEAStudio-<version>-linux-x86_64.run    Linux self-extracting installer (sh header + .tar.gz): --target, --update
    SHA256SUMS.txt

Each package carries `installer/manifest.json` = the program files of this version with digests (the update uses it to delete files
that the new version no longer has – installer/upgrade.py) and the folder `licenses/` with the texts of the third-party licences (installer/collect_licenses.py),
`licenses/SOURCES.md` (sources of the shipped GPL / LGPL binaries) and `licenses/sources/` (CalculiX source archive; the offline setup also Gmsh).
The private cross-check against Abaqus (verification/abaqus_*, its report and dashboard files) is never packaged (PRIVATE_*).

Without the example results the package is small (~10 MB): Python, libraries, CalculiX etc. are downloaded by the installer (`Install FEA Studio.bat` / `install.sh`)
according to the selected modules (fea_core/modules.json). Content = versioned + new (uncommitted) files without ignored ones;
`--ref HEAD` takes only the committed state. The examples `examples/*.fea` go into the package WITH results (zip ~135 MB; `--examples-no-results` = only the input, ~10 MB),
the PDF manuals are generated if Qt WebEngine is available (docs module).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
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
import upgrade  # noqa: E402  (installer/upgrade.py – manifest of the program files)
import collect_licenses  # noqa: E402  (installer/collect_licenses.py – third-party licences into licenses/)

EXCLUDE_PREFIX = ("scratch/", ".claude/", "dist/", "tools/", ".venv/", "verification/out", "stepview/", "docs/projekt/",
                  "installer/linux/prebuilt/", "adapters/fusion/FEAStudio/__pycache__")
EXCLUDE_SUFFIX = (".pyc", ".fea", ".orig", ".rej")
EXCLUDE_FILES = {"installer/manifest.json", "install.log",      # the manifest is created when packing
                 "CLAUDE.md", "CLAUDE.local.md",                # internal development notes (do not distribute)
                 "fea_core/fatigue/data/pragtic_vmat.json"}     # PragTic: do not distribute until the author agrees (L2)
PRAGTIC = "fea_core/fatigue/data/pragtic_vmat.json"
# The cross-check against Abaqus is PRIVATE (user decision 11. 11. 2026): its scripts, results, report and the generators /
# intermediate files of its HTML dashboard are not distributed (the same paths are excluded from the GitHub export,
# .git/info/github-export-exclude.txt). Nothing that is shipped imports them (tests/test_fix_lic_private.py).
PRIVATE_PREFIX = ("verification/abaqus_", "verification/out_abaqus/")
PRIVATE_FILES = {
    "tests/test_abaqus_all.py", "docs/Srovnani_Abaqus.html",
    "verification/build_docs_abaqus_report.py", "verification/make_abaqus_drawings_full.py",
    # generators and intermediate files of the Abaqus comparison dashboard (read verification/out_abaqus)
    "verification/check_dev.py", "verification/drawings_source.js", "verification/escaped_html.txt",
    "verification/generate_drawings_js.py", "verification/generate_interactive_html.py", "verification/inline_compact.html",
    "verification/inline_component.html", "verification/inline_slim.html", "verification/make_inline.py",
    "verification/make_inline_slim.py", "verification/patch_build_docs.py", "verification/payload.json",
    "verification/prepare_dashboard.py", "verification/preview_args.json", "verification/render_payload.json",
    "verification/slim_preview_args.json", "verification/slim_render_payload.json", "verification/update_report_solvers.py",
}
EXEC ={"install.sh", "fea-studio", "installer/fea_setup.py", "installer/linux/build_ccx.sh", "installer/make_package.py",
        "installer/make_locks.py"}
CRLF = (".bat", ".cmd", ".ps1")         # like .gitattributes (eol=crlf); a .ps1 from the working tree is often LF
LF = (".sh",)


def version() -> str:
    return re.search(r'__version__ = "(.+)"', (ROOT / "fea_core" / "__init__.py").read_text(encoding="utf-8")).group(1)


def file_list(ref: str | None) -> list[str]:
    # -z: paths without quotes and escape sequences (otherwise git writes diacritics as "P\305\231…" and the file was left out)
    if ref:
        cmd = ["git", "ls-tree", "-r", "-z", "--name-only", ref]
    else:
        cmd = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, check=True)
    names = sorted({n for n in out.stdout.decode("utf-8").split("\0") if n})
    return [n for n in names if packaged(n) and (ROOT / n).is_file()]


def packaged(name: str) -> bool:
    """Does the file (path relative to the root, with /) belong to the release packages? Internal notes, tools, PragTic and the
    private Abaqus comparison do not.
    """
    return not (name.startswith(EXCLUDE_PREFIX + PRIVATE_PREFIX) or name.endswith(EXCLUDE_SUFFIX)
                or name in EXCLUDE_FILES or name in PRIVATE_FILES)


def stage_files(files: list[str], stage: Path, ref: str | None) -> None:
    for n in files:
        dst = stage / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        if ref:
            data = subprocess.run(["git", "show", f"{ref}:{n}"], cwd=ROOT, capture_output=True, check=True).stdout
            dst.write_bytes(data)
        else:
            shutil.copy2(ROOT / n, dst)


def normalize_eol(stage: Path) -> None:
    for p in stage.rglob("*"):
        rel = p.relative_to(stage).as_posix()
        if p.is_file() and (p.suffix.lower() in CRLF + LF or rel in EXEC):     # executable: LF (shebang on Linux)
            raw = p.read_bytes().replace(b"\r\n", b"\n")
            if p.suffix.lower() in CRLF:
                raw = raw.replace(b"\n", b"\r\n")
            p.write_bytes(raw)


def has_results(fea: Path) -> bool:
    try:
        with zipfile.ZipFile(fea) as z:
            return any(n.startswith("results") for n in z.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def make_examples(stage: Path, with_results: bool = True) -> None:
    """Examples `examples/*.fea`: with results (default; ~85 MB packed – the example opens finished at once) are taken from the working
    tree where `examples.create --solve` created them; an example without results in it = a warning (the guide computes it).
    Without results (`--examples-no-results`): they are created again without results (~0.2 MB).
    """
    ex = stage / "examples"
    if with_results:
        for f in sorted((ROOT / "examples").glob("example*.fea")):
            shutil.copy2(f, ex / f.name)
            if not has_results(f) and f.stat().st_size > 100_000:
                print(f"  ! {f.name}: has no results in the source tree (python -m examples.create --solve --only N)")
    else:
        for cmd in (["-m", "examples.create", "--out", str(ex)], ["-m", "examples.frames", "--out", str(ex)]):
            r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  ! examples: {' '.join(cmd)} failed: {r.stderr[-300:]}")
    fea = sorted(ex.glob("*.fea"))
    print(f"  examples: {len(fea)} × .fea, {sum(has_results(f) for f in fea)} with results ({sum(f.stat().st_size for f in fea) / 1e6:.1f} MB)")


def make_docs(stage: Path) -> None:
    try:
        import PySide6.QtWebEngineCore  # noqa: F401
    except ImportError:
        print("  ! the manual PDFs will not be created (PySide6 Addons / Qt WebEngine missing – docs module)")
        return
    r = subprocess.run([sys.executable, "-m", "fea_gui.help_pdf", "-o", str(stage / "docs")], cwd=ROOT, capture_output=True,
                       text=True)
    pdfs = list((stage / "docs").glob("*.pdf"))
    print(f"  manual PDFs: {len(pdfs)} files" + ("" if r.returncode == 0 else f" (error: {r.stderr[-200:]})"))


def make_licenses(stage: Path) -> None:
    """Texts of the third-party licences into `licenses/` (from the dist-info in .venv, EXTERNAL.md, README.md). All packages (zip,
    tar.gz, .run, setup.exe) carry them as part of the program; missing texts are warnings of the output.
    """
    try:
        warnings = collect_licenses.collect(stage / "licenses", ROOT / ".venv")
    except FileNotFoundError as e:
        raise SystemExit(f"licences: {e} (prepare the development .venv first)")
    for w in warnings:
        print(f"  ! licence: {w}")
    print(f"  licenses/: {sum(1 for _ in (stage / 'licenses' / 'python').iterdir())} packages with texts, {len(warnings)} warnings")


def add_all(files_root: Path, add) -> None:
    for p in sorted(files_root.rglob("*")):
        if p.is_file():
            add(p, p.relative_to(files_root).as_posix())


def write_zip(stage: Path, prefix: str, out: Path, skip: tuple = ()) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        def add(p: Path, rel: str) -> None:
            if not rel.startswith(skip):
                z.write(p, f"{prefix}/{rel}")
        add_all(stage, add)


def neutral(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    """A tar member without the building user: owner 0:0 without user and group names (the package and the bundled CalculiX)."""
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def write_tar(stage: Path, prefix: str, out: Path) -> None:
    with tarfile.open(out, "w:gz", compresslevel=9) as t:
        def add(p: Path, rel: str) -> None:
            ti = neutral(t.gettarinfo(str(p), f"{prefix}/{rel}"))
            ti.mode = 0o755 if rel in EXEC else 0o644
            with open(p, "rb") as f:
                t.addfile(ti, f)
        add_all(stage, add)


LINUX_CCX = ROOT / "installer" / "linux" / "prebuilt" / "ccx_2.23_linux_x86_64.tar.gz"
# absolute paths from home folders (build on the development machine); /home/conda/ = the conda-forge build farm
# (static libgfortran, libgomp) – foreign, does not identify us
PRIVATE_PATH = re.compile(rb"/home/(?!conda/)[A-Za-z0-9_.-]+/|/[Uu]sers/[^/\x00]+/|/mnt/[a-z]/")


def linux_ccx_member(name: str) -> bool:
    """Members of the prebuilt Linux CalculiX that may be distributed: `ccx_spooles` (GPL-2.0 CalculiX + public-domain SPOOLES, BSD
    ARPACK / LAPACK) and the GCC runtime in `lib/`. NOT `ccx_i4`: GPL-2.0-only CalculiX linked with the proprietary Intel MKL
    (PARDISO) cannot be conveyed under GPL-2 – the installer builds it on the user's machine (pardiso module, build_ccx.sh).
    """
    name = name.lstrip("./")
    return name in ("ccx_spooles", "lib") or name.startswith("lib/")


def linux_ccx(src: Path, dst: Path) -> None:
    """Prebuilt CalculiX for Linux (installer/linux/build_ccx.sh … ARCHIVE) into the package – only `linux_ccx_member` (without
    ccx_i4). A binary with a path from a home folder = an error (build again: build_ccx.sh maps it to /build); an archive with other
    members or with an owner other than 0:0 (packed by hand) is repacked – contents, permissions and times unchanged, gzip without
    name and time.
    """
    with tarfile.open(src, "r:gz") as t:
        everything = t.getmembers()
        members = [m for m in everything if linux_ccx_member(m.name)]
        dropped = [m.name for m in everything if not linux_ccx_member(m.name)]
        if not any(m.name.lstrip("./") == "ccx_spooles" for m in members):
            raise SystemExit(f"{src.name}: ccx_spooles is missing – rebuild with installer/linux/build_ccx.sh, or use --no-linux-ccx")
        bad = [m.name for m in members if m.isfile() and PRIVATE_PATH.search(t.extractfile(m).read())]
        if bad:
            raise SystemExit(f"{src.name}: {', '.join(bad)} contains an absolute path from the home folder – rebuild with "
                             "installer/linux/build_ccx.sh OUTPUT WORKDIR ARCHIVE, or package with --no-linux-ccx")
        owned = [m for m in members if m.uid or m.gid or m.uname or m.gname]
        if not owned and not dropped:
            shutil.copy2(src, dst)
            return
        if dropped:
            print(f"  {src.name}: not distributed → left out: {', '.join(dropped)}")
        if owned:
            print(f"  {src.name}: {len(owned)} members owned by the builder → repacked with owner 0:0")
        with open(dst, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz, \
                tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as out:
            for m in members:
                out.addfile(neutral(m), t.extractfile(m) if m.isfile() else None)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sums(path: Path, files: list[Path]) -> Path:
    """SHA256SUMS.txt in the sha256sum format (digest, 2 spaces, name) with LF line ends – `sha256sum -c` on Linux would
    look for the file "name\\\\r" with CRLF (writing text on Windows would otherwise give CRLF).
    """
    path.write_text("".join(f"{sha(p)}  {p.name}\n" for p in files), encoding="utf-8", newline="\n")
    return path


RUN_HEADER = r'''#!/bin/sh
# FEA Studio @VER@ - samorozbalovaci instalator pro Linux x86_64 (glibc 2.35+) / self-extracting installer.
#   sh FEAStudio-@VER@-linux-x86_64.run [--target SLOZKA] [--update | --install] [--extract-only] [parametry fea_setup.py]
# Vychozi cil: XDG_DATA_HOME/fea-studio (obvykle .local/share/fea-studio v domovske slozce). Je-li v cili instalace,
# provede se aktualizace (fea_setup.py update: migrace, zastarale soubory, moduly jako minule), jinak instalace.
# Dalsi parametry jdou do installer/fea_setup.py, napr. --profile standard --quiet --no-shortcuts
# Navrat ke starsi verzi jen s --force.
set -e
VERSION="@VER@"
PAYLOAD_SHA256="@SHA@"
target="${XDG_DATA_HOME:-${HOME}/.local/share}/fea-studio"
mode=""
extract_only=0
while [ $# -gt 0 ]; do
    case "$1" in
        --target) [ $# -ge 2 ] || { echo "--target: chybi slozka" >&2; exit 2; }; target="$2"; shift 2 ;;
        --target=*) target="${1#--target=}"; shift ;;
        --update) mode=update; shift ;;
        --install) mode=install; shift ;;
        --extract-only|--noexec) extract_only=1; shift ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
case "$target" in /*) ;; *) target="$PWD/$target" ;; esac
force=0
for a in "$@"; do [ "$a" = "--force" ] && force=1; done
line=$(awk '/^__FEA_STUDIO_PAYLOAD__$/ { print NR + 1; exit }' "$0")
if command -v sha256sum >/dev/null 2>&1; then
    got=$(tail -n +"$line" "$0" | sha256sum | cut -d ' ' -f 1)
elif command -v shasum >/dev/null 2>&1; then
    got=$(tail -n +"$line" "$0" | shasum -a 256 | cut -d ' ' -f 1)
else
    echo "CHYBA: chybi sha256sum i shasum - nelze overit neporusenost instalatoru" >&2; exit 1
fi
if [ "$got" != "$PAYLOAD_SHA256" ]; then echo "CHYBA: instalator je poskozeny (kontrolni soucet nesedi)" >&2; exit 1; fi
state="$target/.venv/fea_modules.json"
if [ -f "$state" ]; then
    old=$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' "$state" | head -n 1)
    [ -n "$old" ] || old="1.0.0"
    newest=$(printf '%s\n%s\n' "$old" "$VERSION" | sort -V | tail -n 1)
    if [ "$old" != "$VERSION" ] && [ "$newest" = "$old" ] && [ "$force" = 0 ]; then
        echo "CHYBA: v $target je novejsi verze $old nez tento instalator ($VERSION); navrat ke starsi verzi jen s --force" >&2
        exit 2
    fi
    [ -n "$mode" ] || mode=update
fi
[ -n "$mode" ] || mode=install
mkdir -p "$target"
echo "FEA Studio $VERSION -> $target ($mode)"
tail -n +"$line" "$0" | tar -xzf - -C "$target" --strip-components=1 --no-same-owner
if [ "$extract_only" = 1 ]; then
    echo "Rozbaleno do $target; instalace: sh $target/install.sh $mode"
    exit 0
fi
cd "$target"
exec sh ./install.sh "$mode" "$@"
__FEA_STUDIO_PAYLOAD__
'''


def write_run(tar_path: Path, out: Path, ver: str) -> Path:
    """Self-extracting installer for Linux: sh header (ASCII, LF) + the data .tar.gz (makeself style, without dependencies)."""
    head = RUN_HEADER.replace("@VER@", ver).replace("@SHA@", sha(tar_path)).encode("ascii")
    with open(out, "wb") as f, open(tar_path, "rb") as t:
        f.write(head)
        shutil.copyfileobj(t, f, 1 << 20)
    out.chmod(0o755)
    return out


def bundle_python(stage: Path) -> Path | None:
    """Python 3.11 for Windows unpacked into tools/python/<PY_DIR> (zip and setup.exe): uv finds it during the installation as a managed
    Python, so nothing is downloaded or extracted through tools/python/.temp (which fails with some antiviruses).
    """
    import fea_setup
    import slim
    plat = "windows-x86_64"
    target = stage / "tools" / "python" / fea_setup.PY_DIR[plat]
    try:
        arch = fea_setup.download(fea_setup.PY_URL[plat], ROOT / "tools" / "_download" / f"cpython-{fea_setup.PY_FULL}-{plat}.tar.gz",
                                  f"Python {fea_setup.PY_FULL}", fea_setup.PY_SHA256[plat])
        tmp = stage / "tools" / "python" / "_extract"
        shutil.rmtree(tmp, ignore_errors=True)
        fea_setup.extract_archive(arch, tmp)
        for i in range(30):                      # synchronisation of the folder / antivirus briefly holds freshly extracted files
            try:
                (tmp / "python").replace(target)
                break
            except PermissionError:
                if i == 29:
                    raise
                time.sleep(1)
        shutil.rmtree(tmp, ignore_errors=True)
    except (OSError, RuntimeError, tarfile.TarError) as e:
        # a package without Python would depend on a download through uv (antiviruses) on clean Windows – better not to release it
        shutil.rmtree(stage / "tools" / "python", ignore_errors=True)
        raise SystemExit(f"Python {fea_setup.PY_FULL} could not be bundled ({e}); run again, "
                         "or use --no-python explicitly")
    slim.prune_python_home(target)
    print(f"  Python {fea_setup.PY_FULL} bundled: {slim.dir_size(target) / 1e6:.0f} MB")
    return target


def bundle_windows_tools(stage: Path) -> list[Path]:
    """uv.exe and CalculiX `ccx_static.exe` (from the verified archive of dhondt.de) into the packages for Windows: the installation then
    downloads no executable files for the base program (antiviruses and corporate web filters block downloads of .exe files / of
    little-known servers). `ccx_i4.exe` (CalculiX linked with the proprietary Intel MKL PARDISO) is NOT redistributed – GPL-2.0-only
    does not allow conveying it together with a non-GPL library; the installer extracts it from the original dhondt.de zip only when
    the pardiso module is installed (fea_setup.install_ext_ccx, needs internet).
    """
    import fea_setup
    made = []
    uv = ROOT / "tools" / "uv" / "uv.exe"
    if uv.exists():
        (stage / "tools" / "uv").mkdir(parents=True, exist_ok=True)
        shutil.copy2(uv, stage / "tools" / "uv" / "uv.exe")
        made.append(stage / "tools" / "uv")
    else:
        print("  ! tools/uv/uv.exe is missing – an installation from the zip downloads uv")
    try:
        z = fea_setup.ccx_zip()                         # tools/_download, otisk CCX_WIN_SHA256
        fea_setup.extract_archive(z, stage / "tools" / "calculix", members=list(WIN_CCX_MEMBERS))
        d = stage / "tools" / "calculix" / "calculix_2.23_4win"
        if not (d / "ccx_static.exe").exists():
            raise RuntimeError("ccx_static.exe missing in the archive")
        if (d / "ccx_i4.exe").exists():                 # never in a package (see above)
            (d / "ccx_i4.exe").unlink()
        made.append(stage / "tools" / "calculix")
        print(f"  CalculiX bundled: {slim_size(d) / 1e6:.0f} MB")
    except (OSError, RuntimeError) as e:
        print(f"  ! CalculiX will not be bundled ({e}) – the installation downloads it from dhondt.de")
    return made


WIN_CCX_MEMBERS = ("ccx_static.exe", "README_Install")   # from calculix_2.23_4win.zip into the Windows packages (no ccx_i4.exe)


def add_sources(stage: Path, kind: str) -> list[Path]:
    """Source archives of the shipped GPL binaries into `licenses/sources/` (GPL-2 §3(a) – the source travels with the binary;
    list and digests in collect_licenses.SOURCE_ARCHIVES, described in licenses/SOURCES.md). `kind` = "all" (every package:
    CalculiX for ccx_static.exe / ccx_spooles) or "offline" (only the offline setup: Gmsh, whose wheel it carries).
    Downloaded once into tools/_download/sources (digest checked); a missing archive = an error (do not release without it).
    """
    import fea_setup
    dst = stage / "licenses" / "sources"
    dst.mkdir(parents=True, exist_ok=True)
    made = []
    for name, s in collect_licenses.SOURCE_ARCHIVES.items():
        if s["packages"] != kind:
            continue
        try:
            arch = fea_setup.download(s["url"], ROOT / "tools" / "_download" / "sources" / name, name, s["sha256"])
        except (OSError, RuntimeError) as e:
            raise SystemExit(f"source archive {name} could not be obtained ({e}) – the GPL binaries must not ship without it")
        shutil.copy2(arch, dst / name)
        made.append(dst / name)
    print(f"  licenses/sources ({kind}): {', '.join(p.name for p in made) or '–'} "
          f"({sum(p.stat().st_size for p in made) / 1e6:.1f} MB)")
    return made


def slim_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


INTEL_WHEELS = {"mkl", "intel-openmp", "tbb", "intel-cmplr-lib-ur", "tcmlib", "umf"}


def _req_name(line: str) -> str:
    """Package name from a requirement line ("mkl==2023.2.0" → "mkl"), normalised as on PyPI."""
    name = re.match(r"[A-Za-z0-9._-]+", line.strip())
    return re.sub(r"[-_.]+", "-", name.group(0)).lower() if name else ""


def offline_requirements() -> list[str]:
    """Exact versions of the packages of the standard profile for Windows (the same procedure as fea_setup.sync_packages)."""
    import fea_setup
    keys = fea_setup.closure(fea_setup.PROFILES["standard"])
    reqs = sorted({r for k in keys for r in fea_setup.MODS[k].get("pip", [])})
    nodeps = sorted({r for k in keys for r in fea_setup.MODS[k].get("pip_nodeps", [])})
    with tempfile.TemporaryDirectory(prefix="fea_off_") as td:
        rin, out = Path(td) / "req.in", Path(td) / "resolved.txt"
        rin.write_text("\n".join(reqs) + "\n", encoding="utf-8")
        subprocess.run([str(fea_setup.ensure_uv()), "pip", "compile", str(rin), "-c", str(ROOT / "installer" / "locks" /
                        "windows-x86_64.txt"), "--python-platform", "x86_64-pc-windows-msvc", "--python-version", "3.11",
                        "--no-header", "--no-annotate", "--no-config", "-q", "-o", str(out)], check=True)
        lines = [l.strip() for l in out.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    # Intel (mkl, intel-openmp, tbb): intel-openmp has the EULA of the Intel Developer Tools – distribution only in "Your Product" under
    # a licence without a reciprocal part, not with GPL → the offline set does not carry them, the pardiso module downloads them from PyPI itself
    lines = [l for l in lines if _req_name(l) not in INTEL_WHEELS]
    return sorted(set(lines + nodeps), key=str.lower)


def fetch_wheels(reqs: list[str], dest: Path) -> list[Path]:
    """Wheels for Python 3.11 / win_amd64 from PyPI with the SHA-256 digest from PyPI (JSON API) into the cache `dest`. A package
    without a wheel = an error (an installation without internet could not build it). Returns the files of this set.
    """
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11):
        raise SystemExit("--offline: run make_package.py with Python 3.11 for Windows (development .venv)")
    order = {t: i for i, t in enumerate(sys_tags())}
    dest.mkdir(parents=True, exist_ok=True)
    got, missing = [], []
    for r in reqs:
        name, _, ver = r.partition("==")
        name, ver = name.split("[")[0].strip(), ver.split(";")[0].strip()
        req = urllib.request.Request(f"https://pypi.org/pypi/{name}/{ver}/json", headers={"User-Agent": "FEAStudio-make_package"})
        with urllib.request.urlopen(req, timeout=60) as f:
            files = json.load(f)["urls"]
        best = None
        for u in files:
            if u["packagetype"] != "bdist_wheel":
                continue
            rank = min((order.get(t, 10 ** 6) for t in parse_wheel_filename(u["filename"])[3]), default=10 ** 6)
            if rank < 10 ** 6 and (best is None or rank < best[0]):
                best = (rank, u)
        if best is None:
            missing.append(r)
            continue
        u = best[1]
        out = dest / u["filename"]
        if not (out.exists() and sha(out) == u["digests"]["sha256"]):
            with urllib.request.urlopen(urllib.request.Request(u["url"], headers={"User-Agent": "FEAStudio-make_package"}),
                                        timeout=120) as f, open(str(out) + ".part", "wb") as o:
                shutil.copyfileobj(f, o, 1 << 20)
            if sha(Path(str(out) + ".part")) != u["digests"]["sha256"]:
                raise SystemExit(f"{u['filename']}: SHA-256 digest does not match PyPI")
            Path(str(out) + ".part").replace(out)
        got.append(out)
    if missing:
        raise SystemExit("--offline: bez kola pro cp311 / win_amd64: " + ", ".join(missing))
    return got


def find_iscc(given: str) -> str:
    for c in (given, str(ROOT / "tools" / "innosetup" / "ISCC.exe"), shutil.which("ISCC.exe"), shutil.which("iscc")):
        if c and Path(c).exists():
            return c
    return ""


def write_setup_exe(stage: Path, out: Path, ver: str, iscc: str, offline: bool = False) -> Path | None:
    """setup.exe (Inno Setup 6) from the unpacked package for Windows (Python, uv.exe and CalculiX are already in stage/tools);
    `offline` = also the Python packages in stage/tools/offline/wheels → FEAStudio-<version>-offline-setup.exe.
    """
    import fea_setup
    args = [iscc, "/Q", f"/DAppVersion={ver}", f"/DSourceDir={stage}", f"/O{out}", f"/DPyDir={fea_setup.PY_DIR['windows-x86_64']}"]
    uv = ROOT / "tools" / "uv" / "uv.exe"
    if not (stage / "tools" / "uv" / "uv.exe").exists() and uv.exists():
        args.append(f"/DUvExe={uv}")
    if offline:
        args.append("/DOffline")
    r = subprocess.run(args + [str(ROOT / "installer" / "FEAStudio.iss")])
    exe = out / f"FEAStudio-{ver}-{'offline-' if offline else ''}setup.exe"
    if r.returncode != 0 or not exe.exists():
        print(f"  ! ISCC ended with code {r.returncode} – setup.exe was not created")
        return None
    return exe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", help="git ref (default: the working tree incl. uncommitted files)")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--no-docs", action="store_true")
    ap.add_argument("--no-examples", action="store_true")
    ap.add_argument("--examples-no-results", action="store_true", help="examples without results (small package)")
    ap.add_argument("--no-linux-ccx", action="store_true", help="without the prebuilt CalculiX for Linux")
    ap.add_argument("--iscc", default="", help="path to ISCC.exe (default tools/innosetup, PATH)")
    ap.add_argument("--no-setup", action="store_true", help="without setup.exe")
    ap.add_argument("--no-run", action="store_true", help="without the self-extracting .run for Linux")
    ap.add_argument("--no-python", action="store_true", help="without the bundled Python 3.11 in the zip and setup.exe (the installation downloads it)")
    ap.add_argument("--with-pragtic", action="store_true",
                    help="include the PragTic database (PRIVATE build for own use only – the data have no licence for redistribution)")
    ap.add_argument("--offline", action="store_true",
                    help="also FEAStudio-<version>-offline-setup.exe with all Python packages of the standard profile (no internet)")
    a = ap.parse_args()
    if a.with_pragtic:
        EXCLUDE_FILES.discard(PRAGTIC)
        print("  ! PRIVATE build with the PragTic database – do not distribute")
    ver = version()
    prefix = f"FEAStudio-{ver}"
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "stage"
    shutil.rmtree(stage, ignore_errors=True)
    print(f"FEA Studio {ver}: building packages into {out}")
    files = file_list(a.ref)
    stage_files(files, stage, a.ref)
    print(f"  {len(files)} files")
    if not a.no_examples:
        make_examples(stage, not a.examples_no_results)
    if not a.no_docs:
        make_docs(stage)
    make_licenses(stage)
    add_sources(stage, "all")                                           # CalculiX source (ccx_static.exe, ccx_spooles)
    normalize_eol(stage)
    manifest = upgrade.manifest_of_dir(stage, ver)                     # program files of this version (without tools/)
    upgrade.write_manifest(stage / upgrade.PACKAGE_MANIFEST, manifest)
    print(f"  manifest: {len(manifest['files'])} program files")
    zip_path = out / f"{prefix}.zip"
    tar_path = out / f"{prefix}.tar.gz"
    made = [zip_path, tar_path]
    my = ROOT / "tools" / "mystran" / "mystran.exe"                   # MYSTRAN (MIT) only in the Windows package
    if my.exists():
        (stage / "tools" / "mystran").mkdir(parents=True, exist_ok=True)
        shutil.copy2(my, stage / "tools" / "mystran" / my.name)
    if not a.no_python:
        bundle_python(stage)                                            # Python 3.11 only in the Windows packages
    win_tools = bundle_windows_tools(stage)                             # uv.exe, CalculiX for Windows
    write_zip(stage, prefix, zip_path)
    iscc = "" if a.no_setup else find_iscc(a.iscc)
    if iscc:
        exe = write_setup_exe(stage, out, ver, iscc)
        if exe:
            made.append(exe)
        if a.offline:
            wheels = stage / "tools" / "offline" / "wheels"
            reqs = offline_requirements()
            files = fetch_wheels(reqs, ROOT / "tools" / "_download" / "offline-wheels")     # cache between the builds
            wheels.mkdir(parents=True, exist_ok=True)
            for w in files:
                shutil.copy2(w, wheels / w.name)
            print(f"  offline: {len(files)} wheels, {sum(w.stat().st_size for w in files) / 1e6:.0f} MB")
            extra_src = add_sources(stage, "offline")                   # Gmsh source next to the Gmsh wheel (GPL-2+)
            exe = write_setup_exe(stage, out, ver, iscc, offline=True)
            if exe:
                made.append(exe)
            shutil.rmtree(stage / "tools" / "offline", ignore_errors=True)
            for p in extra_src:
                p.unlink(missing_ok=True)
    elif not a.no_setup:
        print("  Inno Setup (ISCC.exe) not found – setup.exe is not created (python installer/get_innosetup.py)")
    for d in win_tools:
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(stage / "tools" / "mystran", ignore_errors=True)
    shutil.rmtree(stage / "tools" / "python", ignore_errors=True)
    if not any((stage / "tools").iterdir() if (stage / "tools").exists() else []):
        shutil.rmtree(stage / "tools", ignore_errors=True)
    pre = LINUX_CCX
    if pre.exists() and not a.no_linux_ccx:
        (stage / "tools" / "calculix").mkdir(parents=True, exist_ok=True)
        linux_ccx(pre, stage / "tools" / "calculix" / pre.name)
    else:
        print("  ! the prebuilt CalculiX for Linux is missing (installer/linux/prebuilt/) – the Linux installer builds it from source")
    write_tar(stage, prefix, tar_path)
    if not a.no_run:
        made.append(write_run(tar_path, out / f"{prefix}-linux-x86_64.run", ver))
    write_sums(out / "SHA256SUMS.txt", made)
    shutil.rmtree(stage, ignore_errors=True)
    for p in made:
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
