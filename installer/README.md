# Installing FEA Studio (Windows x64 and Linux x86_64)

FEA Studio installs **by modules**: the base solver core (STEP → mesh → CalculiX → results) is small; optional modules (GUI, fatigue and 3D printing, MKL PARDISO, Word reports, MCP server, large model solvers) can be added or removed at any time. Everything resides in the application directory (`.venv`, `tools`). No administrator privileges are required, and no system files are modified (except desktop shortcuts, `.fea` file association, and optional MS-MPI).

## Modules (`fea_core/modules.json`)

| Module | Contents | Disk Space* |
|---|---|---|
| `core` | FEA core, CLI, CalculiX (PaStiX / SPOOLES), Gmsh, OpenCASCADE; statics, heat transfer, modal, buckling, beams, 2D, shells, joints, welds | ~693 MB |
| `plots` | Charts and diagrams (matplotlib) | ~43 MB |
| `gui` | Graphical interface (PySide6 / Qt, without WebEngine) | ~80 MB |
| `accel` | Numba: fatigue (rainflow, critical plane), 3D printing, result images for MCP and reports | ~129 MB |
| `pardiso` | Intel MKL 2023.2 + `ccx_i4`: ~40% faster CalculiX, NGSolve + PARDISO direct solver. Needs internet: MKL comes from PyPI and `ccx_i4` is not distributed with FEA Studio (GPL-2.0 CalculiX + proprietary MKL) – Windows extracts it from the original dhondt.de archive, Linux builds it locally | ~396 MB |
| `beams` | Custom beam section meshing (sectionproperties via Gmsh); EN standard profiles work without it | ~15 MB |
| `report` | Microsoft Word engineering reports (`.docx`) | ~11 MB |
| `motion` | Multibody dynamics (Exudyn worker process) | ~22 MB |
| `mcp` | Model Context Protocol server (180+ tools) and REST API for CAD adapters | ~42 MB |
| `ngsolve` | NGSolve solver (C++ matrices, direct PARDISO for multiple load cases) | ~176 MB |
| `hypre` | Large model iterative solver (hypre BoomerAMG + MPI): MS-MPI on Windows, MPICH on Linux | ~72 MB |
| `chrono` | Rigid/flexible contact in mechanisms (Project Chrono 10 in `tools/chrono_env`); manual install only | ~600 MB |
| `verify` | Verification benchmarks, MYSTRAN cross-checks, pytest | ~51 MB |
| `docs` | Developer tools: PDF manual generation (Qt WebEngine); pre-built PDFs are included in `docs/` | ~384 MB |

\* Approximate Windows footprint after slimming (`installer/slim.py`). Module dependencies resolve automatically (e.g., `hypre` → `ngsolve` → `pardiso` → `core`).

Profiles:
- **minimal**: `core` (~0.7 GB)
- **standard**: `core gui plots accel pardiso beams report motion` (~1.4 GB)
- **full**: standard + `mcp ngsolve hypre` (~1.7 GB)

## Windows Installation

### Option A: Portable Archive (.zip)

1. Extract `FEAStudio-<version>.zip` to your target directory (avoid cloud-synced folders like OneDrive).
2. Run **`Install FEA Studio.bat`** and select an installation profile (minimal, standard, full, or custom).
   - Silent run: `Install FEA Studio.bat --profile standard --quiet`
   - Specific modules: `Install FEA Studio.bat gui accel`
3. Launch via the desktop shortcut or `FEA Studio.bat`. Drag-and-drop `.step` or `.fea` files directly onto the launcher or window.

### Option B: Windows Installer (setup.exe)

- Run `FEAStudio-<version>-setup.exe` to install to `%LOCALAPPDATA%\Programs\FEA Studio` without admin rights.
- Supports module selection, updates over existing installations, and clean uninstallation via Windows Settings → Apps.
- Silent flags: `FEAStudio-<version>-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /TYPE=standard /DIR="C:\FEA Studio"`
- Options: `/NOSHORTCUTS`, `/NOMCP`, `/NOFUSION`, `/NOCHECK`, `/FORCE`.

### Enterprise Networks and Offline Installations

- **No PowerShell required**: Both setup.exe and `.bat` execute via bundled Python without triggering restrictive PowerShell execution policies.
- **Corporate proxies & HTTPS inspection**: Installer uses the Windows certificate store (`UV_SYSTEM_CERTS=1`) and system proxy settings (`HTTPS_PROXY`).
- **Offline environments**: Use `FEAStudio-<version>-offline-setup.exe` (~0.7 GB), which bundles pre-cached wheels for the standard profile and installs without internet connectivity. Exception: the `pardiso` module (Intel MKL and `ccx_i4.exe` are not redistributed) needs internet; offline, CalculiX runs with `ccx_static.exe` (PaStiX / SPOOLES, ~40 % slower) and `pardiso` can be added later (`python installer/fea_setup.py install pardiso`).

## Linux Installation

Supported distributions: x86_64, glibc ≥ 2.35 (Ubuntu 22.04+, Debian 12+, Fedora 36+, openSUSE 15.5+).

```bash
tar xzf FEAStudio-<version>.tar.gz && cd FEAStudio-<version>
./install.sh                      # Interactive prompt, or: ./install.sh --profile standard --quiet
./fea-studio                      # Launch GUI
```

The package carries CalculiX `ccx_spooles` (built by `installer/linux/build_ccx.sh`). With the `pardiso` module the installer builds `ccx_i4` (PARDISO) on your machine from source (a few minutes; needs `curl`, `bzip2`, `perl` and internet).

## Licences and sources

Every package carries `licenses/` (texts of the third-party licences, `EXTERNAL.md`, `SOURCES.md`) and `licenses/sources/` with the CalculiX 2.23 source archive (the source of the shipped `ccx_static.exe` / `ccx_spooles`); the offline setup also the Gmsh source. See `THIRD_PARTY_LICENSES.md`.

## Maintenance & Uninstallation

- **Add/remove modules**: `python -m fea_core.modules install <module>` or GUI menu Tools → Application Modules.
- **Update**: Run new setup.exe over old installation or `Install FEA Studio.bat update`.
- **Uninstall**: Run `fea_setup.py uninstall` or Windows Settings → Apps. Preserves user-created `.fea` files.
