#!/usr/bin/env python3
"""Collects the licences of the third-party packages of FEA Studio (standard library only).

    python installer/collect_licenses.py [--out FOLDER] [--venv .venv]

Output (by default `licenses/` in the project root; `make_package.py` packs it into every package as `licenses/`):
    python/<package>-<version>/   licence files of the Python packages from their dist-info (locks windows-x86_64 and linux-x86_64)
    external/<component>/         texts of the components that are not Python packages – from installer/license_texts/ (downloaded from the official
                                  sources, see license_texts/SOURCES.md) and from local files (tools/, .venv/hypre_env);
                                  nothing is downloaded from the internet
    EXTERNAL.md                   components outside Python and data: version, licence, source, how they are used
    SOURCES.md                    sources of the shipped GPL / LGPL binaries (archives in sources/ – added by make_package.py –,
                                  URLs with SHA-256, written offer)
    README.md                    what the folder contains, links to the sources of the GPL components, summary table of the Python packages

Python packages whose wheel has no licence file (gmsh, cadquery-ocp, ngsolve-openblas) get their texts from `PYTHON_EXTRA`.
An installed package without a licence file = a warning (the function `collect` returns them; the exit code stays 0).
Missing venv or lock = exit code 1.
"""
from __future__ import annotations

import argparse
import datetime
import re
import shutil
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ("windows-x86_64", "linux-x86_64")
NOT_INSTALLED = "unknown – not installed"
TEXTS_DIR = ("installer", "license_texts")                  # downloaded from the official sources, SHA-256 in SOURCES.md
LICENSE_NAME = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|COPYRIGHT|THIRD.?PARTY)", re.IGNORECASE)
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;#]+)")

# Components outside the Python packages (EXTERNAL.md). "texts" = file patterns in installer/license_texts/ (downloaded from the official sources,
# see license_texts/SOURCES.md; they are copied into external/<component>/); "local" = patterns of files with a licence text that are
# only in this tree (tools/, .venv/hypre_env) – the same content is not copied twice.
EXTERNAL = [
    {"name": "CalculiX (ccx)", "slug": "calculix", "version": "2.23", "licence": "GPL-2.0",
     "source": "https://www.dhondt.de/ccx_2.23.src.tar.bz2 (source, also in `sources/` of the package – see SOURCES.md); "
               "Linux build: installer/linux/build_ccx.sh",
     "use": "separate process – FEM solver (`tools/calculix`): packages carry `ccx_static.exe` (Windows) and `ccx_spooles` (Linux); "
            "`ccx_i4` (with Intel MKL PARDISO) is NOT distributed – the installer extracts it from the dhondt.de zip (Windows) or "
            "builds it (Linux) on the user's machine when the pardiso module is installed",
     "texts": ["calculix/*", "_gnu/gpl-2.0.txt"], "local": []},
    {"name": "SPOOLES", "slug": "spooles", "version": "2.2",
     "licence": "public domain (statement in the SPOOLES 2.2 documentation: “entirely within the public domain; there are no licensing restrictions”)",
     "source": "https://www.netlib.org/linalg/spooles/spooles.2.2.tgz (netlib); used by the build in installer/linux/build_ccx.sh",
     "use": "equation solver in `ccx_spooles` (Linux, static)", "texts": ["spooles/*"], "local": []},
    {"name": "ARPACK-ng", "slug": "arpack-ng", "version": "–", "licence": "BSD-3-Clause", "source": "https://github.com/opencollab/arpack-ng",
     "use": "eigenvalues in `ccx_spooles` (Linux, static)", "texts": ["arpack-ng/*"], "local": []},
    {"name": "LAPACK / BLAS (reference)", "slug": "lapack-blas", "version": "–", "licence": "BSD-3-Clause",
     "source": "https://www.netlib.org/lapack/ (text: https://github.com/Reference-LAPACK/lapack)",
     "use": "linear algebra in `ccx_spooles` (Linux, static)", "texts": ["lapack/*"], "local": []},
    {"name": "libgomp, libquadmath (GCC)", "slug": "libgomp-libquadmath", "version": "–",
     "licence": "libgomp: GPL-3.0 with GCC Runtime Library Exception 3.1; libquadmath: LGPL-2.1+ (libquadmath/COPYING.LIB in the GCC sources)",
     "source": "https://gcc.gnu.org/ (GCC source; see SOURCES.md)",
     "use": "OpenMP and quad-precision runtime libraries of `ccx_spooles` (Linux package) and of the locally built `ccx_i4`",
     "texts": ["gcc-runtime/*"], "local": []},
    {"name": "gcc-libs, libgfortran (hypre_env)", "slug": "gcc-libs-libgfortran", "version": "–",
     "licence": "GPL-3.0 with GCC Runtime Library Exception 3.1", "source": "https://gcc.gnu.org/ (GCC source)",
     "use": "runtime libraries of the hypre environment (`.venv/hypre_env`, Windows)",
     "texts": ["gcc-runtime/COPYING3", "gcc-runtime/COPYING.RUNTIME"],
     "local": [".venv/hypre_env/Library/share/licenses/**/*"]},
    {"name": "Gmsh", "slug": "gmsh", "version": "4.15.2",
     "licence": "GPL-2.0-or-later (exception for combining with Netgen, METIS, OpenCASCADE, ParaView)",
     "source": "https://gmsh.info/src/ (source); texts: external/gmsh/ (the PyPI wheel has no licence file in its dist-info)",
     "use": "meshing in a separate process (Python API, `fea_core.mesh.worker`)", "texts": ["gmsh/*", "_gnu/gpl-2.0.txt"], "local": []},
    {"name": "MYSTRAN", "slug": "mystran", "version": "19", "licence": "MIT", "source": "https://github.com/dr-bill-c/MYSTRAN",
     "use": "cross-check in a separate process (`tools/mystran`, Windows package only)", "texts": ["mystran/*"], "local": []},
    {"name": "hypre", "slug": "hypre", "version": "3.2.0", "licence": "Apache-2.0 or MIT", "source": "https://github.com/hypre-space/hypre",
     "use": "AMG solver in a separate MPI process (hypre module; conda-forge `.venv/hypre_env`)", "texts": ["hypre/*"], "local": []},
    {"name": "MPICH", "slug": "mpich", "version": "5 (conda-forge)", "licence": "MPICH (BSD-like)", "source": "https://www.mpich.org/",
     "use": "`mpiexec` and MPI on Linux (hypre module)", "texts": ["mpich/*"], "local": []},
    {"name": "MS-MPI", "slug": "ms-mpi", "version": "10", "licence": "Microsoft MS-MPI Redistributable (Microsoft licence; sources and SDK MIT)",
     "source": "https://learn.microsoft.com/message-passing-interface/microsoft-mpi (MIT text: https://github.com/microsoft/Microsoft-MPI)",
     "use": "`mpiexec` on Windows (hypre module; installed by the user with consent, not in the package – the user accepts the terms of the "
            "redistributable runtime when installing it)", "texts": ["ms-mpi/*"], "local": []},
    {"name": "Intel MKL", "slug": "intel-mkl", "version": "2023.2.0", "licence": "Intel Simplified Software License",
     "source": "text of the mkl 2023.2.0 wheel (the same file is in python/mkl-2023.2.0/)",
     "use": "PARDISO in `ccx_i4` and `fea_core.pardiso` (pardiso module); NOT in any package – the installer downloads the mkl "
            "wheel from PyPI on the user's machine (also with the offline setup) and `installer/slim.py` then removes unused DLLs",
     "texts": ["intel-mkl/*"], "local": []},
    {"name": "Project Chrono (PyChrono)", "slug": "project-chrono", "version": "10.0.0", "licence": "BSD-3-Clause",
     "source": "https://github.com/projectchrono/chrono (third-party component licences: Chrono sources and the pychrono package, conda-forge)",
     "use": "contact in mechanisms (`tools/chrono_env`, optional chrono module)", "texts": ["project-chrono/*"], "local": []},
    {"name": "OpenCASCADE (OCCT) – core of cadquery-ocp", "slug": "occt", "version": "per cadquery-ocp 8.0.1.1.0",
     "licence": "LGPL-2.1 with a linking exception (OCCT)",
     "source": "https://dev.opencascade.org (text: https://github.com/Open-Cascade-SAS/OCCT, identical in V7_8_1 and V8_0_0)",
     "use": "STEP, geometry, faces and the 3D window (dynamically linked libraries in OCP; replaceable)", "texts": ["occt/*"], "local": []},
    {"name": "uv", "slug": "uv", "version": "–", "licence": "MIT or Apache-2.0", "source": "https://github.com/astral-sh/uv",
     "use": "installer (`tools/uv`, bundled in setup.exe)", "texts": ["uv/*"], "local": ["tools/uv/LICENSE*"]},
    {"name": "python-build-standalone (Python 3.11)", "slug": "python-build-standalone", "version": "3.11",
     "licence": "Python: PSF-2.0 (and others per package); python-build-standalone build scripts: MPL-2.0",
     "source": "https://github.com/astral-sh/python-build-standalone",
     "use": "interpreter `tools/python` (installer)", "texts": ["python-build-standalone/*"], "local": ["tools/python/LICENSE*"]},
    {"name": "micromamba", "slug": "micromamba", "version": "–", "licence": "BSD-3-Clause", "source": "https://github.com/mamba-org/mamba",
     "use": "installer: hypre, Chrono and Linux system libraries (installer tool)", "texts": ["micromamba/*"], "local": []},
    {"name": "Inno Setup", "slug": "inno-setup", "version": "6.7.1", "licence": "Inno Setup License (LICENSE.TXT); the authors ask commercial users to buy a licence",
     "source": "https://jrsoftware.org/isinfo.php", "use": "building setup.exe (`tools/innosetup`); setup.exe carries the runtime part",
     "texts": ["inno-setup/*"], "local": ["tools/innosetup/license.txt"]},
]

# Python packages whose wheel has no licence file in the dist-info: additional texts from installer/license_texts/ (key = normalised name).
# They are used only when the dist-info has no licence file at all.
_OCP_TEXTS = ["cadquery-ocp/LICENSE", "occt/LICENSE_LGPL_21.txt", "occt/OCCT_LGPL_EXCEPTION.txt"]
PYTHON_EXTRA = {
    "gmsh": ["gmsh/LICENSE.txt", "gmsh/CREDITS.txt", "_gnu/gpl-2.0.txt"],
    "cadquery-ocp": _OCP_TEXTS,
    "cadquery-ocp-proxy": _OCP_TEXTS,
    "ngsolve-openblas": ["openblas/LICENSE"],
}

# Packages of the locks that are NOT shipped in any package (only installed from PyPI on request): missing in the venv = a note,
# not a warning (their licence texts come with them from PyPI).
NOT_SHIPPED = {
    "pyside6-addons": "docs module only (developers – PDF manuals); not in any package, installed from PyPI on request",
    "pypdfium2": "docs module only (developers – PDF manuals); not in any package, installed from PyPI on request",
}

# Source archives of the GPL binaries that the packages carry (GPL-2.0 §3(a): the complete source accompanies the binary).
# make_package.add_sources puts them into licenses/sources/: "all" = every package (CalculiX: ccx_static.exe in the Windows
# packages, ccx_spooles in the Linux packages), "offline" = only the offline setup (it carries the Gmsh wheel).
# Digests verified 11. 11. 2026 (the CalculiX digest = the one pinned in installer/linux/build_ccx.sh).
SOURCE_ARCHIVES = {
    "ccx_2.23.src.tar.bz2": {"url": "https://www.dhondt.de/ccx_2.23.src.tar.bz2", "packages": "all",
                             "sha256": "9c88385c10fb04f5dc6c4e98027a51bebdd8aee3920e05190d6c1dd08357d6e7"},
    "gmsh-4.15.2-source.tgz": {"url": "https://gmsh.info/src/gmsh-4.15.2-source.tgz", "packages": "offline",
                               "sha256": "be3f66f225d27ba9fa014f07e83169285da8a051b0e8ab7103d88066b39bdd3e"},
}
BUILD_CCX = ("installer", "linux", "build_ccx.sh")
_GET_MIRRORS = re.compile(r'^\s*get_mirrors\s+(\S+)\s+([0-9a-f]{64})\s*\\?\s*\n((?:\s*"[^"]+"\s*\\?\s*\n)+)', re.MULTILINE)


def build_ccx_sources(root: Path = ROOT) -> list[tuple[str, str, list[str]]]:
    """Source archives that installer/linux/build_ccx.sh downloads (file, SHA-256, URLs) – the same list as the script pins."""
    p = root.joinpath(*BUILD_CCX)
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8")
    ver = re.search(r"^CCX_VER=(\S+)", text, re.MULTILINE)
    out = []
    for m in _GET_MIRRORS.finditer(text):
        if m.group(1).startswith("micromamba"):            # the tool that installs the compilers, not a source
            continue
        urls = re.findall(r'"([^"]+)"', m.group(3))
        if ver:
            urls = [u.replace("${CCX_VER}", ver.group(1)) for u in urls]
        out.append((m.group(1), m.group(2), urls))
    return out


def write_sources(out: Path, root: Path, today: str) -> None:
    """licenses/SOURCES.md: where to get the complete corresponding source of every shipped GPL / LGPL binary (archives in
    `sources/`, URLs with SHA-256, written offer for the rest).
    """
    arch = SOURCE_ARCHIVES
    ccx, gmsh = "ccx_2.23.src.tar.bz2", "gmsh-4.15.2-source.tgz"
    lines = [
        "# Sources of the GPL / LGPL components in this package", "",
        f"Generated by `installer/collect_licenses.py` ({today}). This is not a legal opinion.", "",
        "FEA Studio itself is GPL-3.0-or-later; its complete source code is this package (Python sources, nothing compiled).", "",
        "## Archives included in `sources/`", "",
        "| Archive | Component | In which packages | Original URL | SHA-256 |", "|---|---|---|---|---|",
        f"| `sources/{ccx}` | CalculiX 2.23 (GPL-2.0) – source of `ccx_static.exe` (Windows) and `ccx_spooles` (Linux) | "
        f"all packages | {arch[ccx]['url']} | `{arch[ccx]['sha256']}` |",
        f"| `sources/{gmsh}` | Gmsh 4.15.2 (GPL-2.0-or-later) – source of the `gmsh` wheel | offline setup only (the other packages "
        f"do not carry Gmsh; the installer downloads the wheel from PyPI) | {arch[gmsh]['url']} | `{arch[gmsh]['sha256']}` |",
        "",
        "## CalculiX binaries", "",
        "- **Windows `tools/calculix/calculix_2.23_4win/ccx_static.exe`**: taken unchanged from "
        "https://www.dhondt.de/calculix_2.23_4win.zip (third-party build published on the CalculiX site; per its README_Install "
        "ccx_static = CalculiX + statically linked SPOOLES, TAUCS and PaStiX). CalculiX source: `sources/" + ccx + "`. "
        "SPOOLES 2.2 is public domain; PaStiX for CalculiX: https://github.com/Dhondtguido/PaStiX4CalculiX ; "
        "TAUCS: https://www.tau.ac.il/~stoledo/taucs/ .",
        "- **Linux `tools/calculix/ccx_spooles`** (+ `lib/libgomp.so.1`, `lib/libquadmath.so.0`): built by "
        "`installer/linux/build_ccx.sh` (in this package – the complete build script) from these pinned archives:", "",
        "| File | SHA-256 | URL |", "|---|---|---|",
    ]
    for name, digest, urls in build_ccx_sources(root):
        lines.append(f"| `{name}` | `{digest}` | {' or '.join(urls)} |")
    lines += [
        "",
        "  Compilers and the GCC runtime (`libgomp` GPL-3.0 with the GCC Runtime Library Exception, `libquadmath` LGPL-2.1+) come "
        "from conda-forge (`gcc_linux-64`, `gfortran_linux-64`; sources: https://gcc.gnu.org/ and "
        "https://github.com/conda-forge/ctng-compilers-feedstock).",
        "- **`ccx_i4` (CalculiX + Intel MKL PARDISO) is not in any package.** GPL-2.0-only CalculiX may not be conveyed together "
        "with the proprietary MKL; the installer obtains it on the user's machine only with the pardiso module (Windows: extracted "
        "from the original dhondt.de zip, Linux: built by build_ccx.sh against the MKL wheel). It therefore needs internet.", "",
        "## LGPL libraries carried as Python wheels (offline setup only)", "",
        "The offline setup (`FEAStudio-<version>-offline-setup.exe`) carries unmodified wheels from PyPI (other packages download them "
        "during the installation). The libraries are dynamically linked and replaceable. Sources:", "",
        "- Qt 6.11.2 / PySide6 / shiboken6 6.11.2 (LGPL-3.0): https://download.qt.io/official_releases/qt/6.11/6.11.2/ and "
        "https://download.qt.io/official_releases/QtForPython/pyside6/ (pyside-setup-everywhere-src-6.11.2)",
        "- OpenCASCADE (OCCT, LGPL-2.1 with exception) in cadquery-ocp 8.0.1.1.0: https://github.com/Open-Cascade-SAS/OCCT and "
        "https://github.com/CadQuery/OCP",
        "- Gmsh 4.15.2 (GPL-2.0-or-later): `sources/" + gmsh + "` (see above)", "",
        "## Written offer", "",
        "For three years after the release of this version, the complete corresponding source code of every GPL / LGPL binary in "
        "this package that is not included in `sources/` will be provided on request for no more than the cost of physically "
        "performing the distribution. Request it through the issue tracker of the FEA Studio repository.", "",
    ]
    (out / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Data shipped with the program (README.md and EXTERNAL.md, section Data).
DATA = [
    {"name": "PragTic – fatigue properties of materials", "path": "fea_core/fatigue/data/pragtic_vmat.json",
     "source": "https://pragtic.com (J. Papuga)",
     "licence": "NOT part of the package (until the author agrees); for own use download it with "
                "python -m fea_core.fatigue.pragtic_fetch"},
]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def read_locks(root: Path) -> dict[str, dict]:
    """Packages from the locks: {normalised name: {"name": name from the lock, "pins": {platform: version}}}."""
    pkgs: dict[str, dict] = {}
    for plat in PLATFORMS:
        lock = root / "installer" / "locks" / f"{plat}.txt"
        if not lock.is_file():
            raise FileNotFoundError(f"lock {lock} is missing")
        for line in lock.read_text(encoding="utf-8").splitlines():
            m = PIN.match(line.strip())
            if m:
                entry = pkgs.setdefault(norm(m.group(1)), {"name": m.group(1), "pins": {}})
                entry["pins"][plat] = m.group(2)
    return pkgs


def site_packages(venv: Path) -> tuple[Path, str]:
    """site-packages of the venv and its platform (Windows: Lib/site-packages, Linux: lib/python3.x/site-packages)."""
    win = venv / "Lib" / "site-packages"
    if win.is_dir():
        return win, "windows-x86_64"
    for p in sorted(venv.glob("lib/python3*/site-packages")):
        if p.is_dir():
            return p, "linux-x86_64"
    raise FileNotFoundError(f"venv {venv} has no site-packages (neither Lib\\site-packages nor lib/python3.x/site-packages)")


def installed(site: Path) -> dict[str, tuple[metadata.Distribution, Path]]:
    """Installed distributions by normalised name: (distribution, *.dist-info folder).

    Like `importlib.metadata.distributions(path=[site])`, but with a known dist-info folder (the licence files).
    """
    out: dict[str, tuple[metadata.Distribution, Path]] = {}
    for info in sorted(site.glob("*.dist-info")):
        dist = metadata.PathDistribution(info)
        try:
            name = dist.metadata.get("Name")
        except Exception:                                   # damaged dist-info (without METADATA) – skip
            name = None
        if name:
            out.setdefault(norm(name), (dist, info))
    return out


def license_files(info: Path, dist: metadata.Distribution) -> list[Path]:
    """Licence files of a distribution: License-File from the metadata, the licenses/ folder and LICENSE*/COPYING*/NOTICE* in the dist-info."""
    found: set[Path] = set()
    for rel in dist.metadata.get_all("License-File") or []:
        p = info / rel
        if p.is_file():
            found.add(p)
    for p in info.rglob("*"):
        if p.is_file() and p.suffix != ".pyc":
            if p.relative_to(info).parts[0] == "licenses" or LICENSE_NAME.match(p.name):
                found.add(p)
    return sorted(found)


def licence_of(dist: metadata.Distribution) -> str:
    md = dist.metadata
    expr = md.get("License-Expression")
    if expr:
        return expr.strip()
    classifiers = sorted({c.split("::")[-1].strip() for c in md.get_all("Classifier") or [] if c.startswith("License ::")})
    if classifiers:
        return "; ".join(classifiers)
    text = (md.get("License") or "").strip()
    first = text.splitlines()[0].strip() if text else ""
    if not first:
        return "not stated in the metadata"
    return first[:80] + ("…" if len(first) > 80 else "")


def _platforms(pins: dict) -> str:
    if set(pins) == set(PLATFORMS):
        return "Windows + Linux"
    return "Windows" if "windows-x86_64" in pins else "Linux"


def _unique_name(dst: Path, src: Path) -> str:
    name = src.name
    if (dst / name).exists():
        name = f"{src.parent.name}_{src.name}"
    return name


def license_text_files(root: Path, patterns: list[str]) -> list[Path]:
    """Files from installer/license_texts/ by patterns (relative to that folder), without duplicates, in the order of the patterns."""
    base = root.joinpath(*TEXTS_DIR)
    found: list[Path] = []
    for pattern in patterns:
        for src in sorted(base.glob(pattern)):
            if src.is_file() and src not in found:
                found.append(src)
    return found


def copy_unique(dst: Path, sources: list[Path]) -> list[str]:
    """Copies the files to `dst` (the same content again is not copied; the same name with other content gets a prefix); returns the file names."""
    names: list[str] = []
    seen: set[bytes] = set()
    for src in sources:
        data = src.read_bytes()
        if data in seen:
            continue
        seen.add(data)
        dst.mkdir(parents=True, exist_ok=True)
        target = _unique_name(dst, src)
        (dst / target).write_bytes(data)
        names.append(target)
    return names


def copy_external(out: Path, root: Path) -> dict[str, list[str]]:
    """Licence texts of the components outside Python (license_texts/ + local files): {component name: [file names]}."""
    copied: dict[str, list[str]] = {}
    for e in EXTERNAL:
        sources = license_text_files(root, e.get("texts", []))
        for pattern in e["local"]:
            sources += [src for src in sorted(root.glob(pattern)) if src.is_file()]
        copied[e["name"]] = copy_unique(out / "external" / ext_slug(e), sources)
    return copied


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def ext_slug(e: dict) -> str:
    """Folder name external/<slug>/ of the component from EXTERNAL."""
    return e.get("slug") or slug(e["name"])


def data_lines() -> list[str]:
    lines = ["", "## Data", "", "| Data | File in the program | Source | Licence |", "|---|---|---|---|"]
    for d in DATA:
        lines.append(f"| {_cell(d['name'])} | `{d['path']}` | {_cell(d['source'])} | {_cell(d['licence'])} |")
    return lines


def write_external(out: Path, copied: dict[str, list[str]], today: str) -> None:
    lines = [
        "# Components outside the Python packages", "",
        "Components that are not Python packages (CalculiX, Gmsh, MPI, MKL…). FEA Studio's own code is licensed under "
        "GNU GPL v3 (file `../LICENSE`).",
        "Full list with notes: `../THIRD_PARTY_LICENSES.md`. Generated by `installer/collect_licenses.py` (" + today + ").",
        "", "| Component | Version | Licence | Source and licence text | How it is used | Text in the package |",
        "|---|---|---|---|---|---|",
    ]
    for e in EXTERNAL:
        names = copied.get(e["name"]) or []
        where = f"`external/{ext_slug(e)}/` (files: {len(names)})" if names else "link (the text is not in this tree – attach by hand)"
        lines.append(f"| {_cell(e['name'])} | {_cell(e['version'])} | {_cell(e['licence'])} | {_cell(e['source'])} | "
                     f"{_cell(e['use'])} | {where} |")
    lines += data_lines()
    (out / "EXTERNAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(out: Path, rows: list[dict], warnings: list[str], today: str) -> None:
    n_text = sum(1 for r in rows if r["texts"].startswith("`python/"))
    lines = [
        "# Third-party licences (folder `licenses/`)", "",
        f"Generated by `installer/collect_licenses.py` ({today}) from `.venv` and the locks `installer/locks/` (windows-x86_64, "
        "linux-x86_64). This is not a legal opinion.", "",
        "FEA Studio's own code is licensed under **GNU GPL v3 or later** (file `../LICENSE`). The licences of the other components "
        "are listed below and in `EXTERNAL.md`. List of components and notes on obligations: `../THIRD_PARTY_LICENSES.md`.", "",
        "## Contents", "",
        "- `python/<package>-<version>/` – licence files of the Python packages (from their dist-info; structure as in the package)",
        "- `external/<component>/` – texts of components that are not Python packages (downloaded from the official sources, "
        "`installer/license_texts/SOURCES.md`)",
        "- `EXTERNAL.md` – components outside Python and data: version, licence, source, how they are used",
        "- `SOURCES.md` – where the complete source of every shipped GPL / LGPL binary is (with SHA-256 digests) and a written offer",
        "- `sources/` – source archives included in the package (CalculiX; the offline setup also Gmsh)", "",
        "## Sources of the GPL components", "",
        "- CalculiX 2.23 (GPL-2.0): `sources/ccx_2.23.src.tar.bz2` = https://www.dhondt.de/ccx_2.23.src.tar.bz2 – Linux build: "
        "`installer/linux/build_ccx.sh`. `ccx_i4` (with Intel MKL) is not distributed (see SOURCES.md).",
        "- Gmsh 4.15.2 (GPL-2.0-or-later): https://gmsh.info/src/ (in the offline setup also `sources/gmsh-4.15.2-source.tgz`)",
        "- Qt / PySide6, OCCT, NGSolve (LGPL): the libraries are separate and replaceable (their licences in `python/`, sources in "
        "SOURCES.md).", "",
        f"## Python packages ({len(rows)} in the locks, {n_text} with licence texts)", "",
        "| Package | Version | Platform | Licence | Texts / note |", "|---|---|---|---|---|",
    ]
    for r in rows:
        cell = " – ".join(x for x in (r["texts"] if r["texts"] != "–" else "", r["note"]) if x)
        lines.append(f"| {_cell(r['name'])} | {_cell(r['version'])} | {r['platforms']} | {_cell(r['licence'])} | {_cell(cell)} |")
    lines += data_lines()
    if warnings:
        lines += ["", "## Warnings from the collection (add the texts before distribution)", ""] + [f"- {_cell(w)}" for w in warnings]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect(out: Path, venv: Path, root: Path = ROOT) -> list[str]:
    """Collects the licences into `out`; returns the list of warnings (an installed package without a licence file, a missing lock package)."""
    site, venv_plat = site_packages(venv)
    pkgs = read_locks(root)
    inst = installed(site)
    out = Path(out)
    for sub in ("python", "external"):
        shutil.rmtree(out / sub, ignore_errors=True)
    (out / "python").mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    warnings: list[str] = []
    for key in sorted(pkgs, key=lambda k: pkgs[k]["name"].lower()):
        name, pins = pkgs[key]["name"], pkgs[key]["pins"]
        hit = inst.get(key)
        if hit is None:
            if key in NOT_SHIPPED:
                note = NOT_SHIPPED[key]
            elif venv_plat in pins:
                warnings.append(f"{name}=={pins[venv_plat]} is in the lock {venv_plat} but missing in the venv ({venv.name}) – "
                                "licences not copied")
                note = "missing in the venv"
            else:
                note = "other platform"
            rows.append({"name": name, "version": pins.get(venv_plat) or next(iter(pins.values())),
                         "platforms": _platforms(pins), "licence": NOT_INSTALLED, "texts": "–", "note": note})
            continue
        dist, info = hit
        ver = dist.version
        note = f"{ver} in the venv, {pins[venv_plat]} in the lock" if venv_plat in pins and pins[venv_plat] != ver else ""
        lic = licence_of(dist)
        files = license_files(info, dist)
        extra = [] if files else license_text_files(root, PYTHON_EXTRA.get(key, []))
        if extra:
            dst = out / "python" / f"{name}-{ver}"
            names = copy_unique(dst, extra)
            texts = f"`python/{dst.name}/` (files: {len(names)}, added from installer/license_texts)"
        elif files:
            dst = out / "python" / f"{name}-{ver}"
            for f in files:
                try:
                    rel = f.relative_to(info).as_posix()
                except ValueError:                          # License-File outside the dist-info
                    rel = f.name
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, target)
            texts = f"`python/{dst.name}/` (files: {len(files)})"
        else:
            texts = "**licence file MISSING** – add it by hand from the author's website"
            warnings.append(f"{name} {ver}: the dist-info has no licence file (licence from the metadata: {lic})")
        rows.append({"name": name, "version": ver, "platforms": _platforms(pins), "licence": lic, "texts": texts, "note": note})
    copied = copy_external(out, root)
    today = datetime.date.today().isoformat()
    write_external(out, copied, today)
    write_sources(out, root, today)
    write_readme(out, rows, warnings, today)
    return warnings


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Collects third-party licences for the FEA Studio packages.")
    ap.add_argument("--out", default=str(ROOT / "licenses"), help="output folder (default licenses/ in the project root)")
    ap.add_argument("--venv", default=str(ROOT / ".venv"), help="development or installed venv (default .venv)")
    a = ap.parse_args(argv)
    try:
        warnings = collect(Path(a.out), Path(a.venv))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"Licenses collected into {Path(a.out).resolve()} ({len(warnings)} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
