# FEA Studio

**🇬🇧 English** | [🇨🇿 Čeština](README_cs.md)

FEA built on CalculiX and Gmsh: linear statics and nonlinear contact (friction, interference, large deformations), plasticity (Python API / MCP only), heat transfer and thermal stress, natural frequencies and buckling, dynamics (harmonic, transient, random vibration, shock), mechanisms (Exudyn), BESO topology optimization, fatigue (S-N, ε-N, critical plane), bolts and welds, beams, 2D, shells, 3D printing, Word reports. Import of STEP and native Inventor / Fusion parts (.ipt, .iam, .f3d) without CAD, Fusion add-in, REST API and an MCP server for AI agents (tool groups: only `core` is listed by default, more via `FEA_MCP_GROUPS` or `fea_tools_enable`). Windows and Linux.

---

## Installation & Running

| | Windows | Linux |
|---|---|---|
| Installation (modular, no admin rights) | `Install FEA Studio.bat` | `./install.sh` |
| Launch GUI | `FEA Studio.bat` | `./fea-studio` |
| Command line | `.venv\Scripts\python -m fea_core --help` | `.venv/bin/python -m fea_core --help` |
| Register MCP server in AI clients | `Register MCP.bat` | `.venv/bin/python -m fea_core.mcp.setup` |

Modules (core, GUI, fatigue...) and disk footprint: [installer/README.md](installer/README.md). Add modules later:
`python -m fea_core.modules install <module>` or in GUI via Tools → Application Modules.

## Structure

| Directory / File | Content |
|---|---|
| [`fea_core/`](fea_core/) | Computational core without GUI (studies, mesh, ccx, results, fatigue, joints, printing, thermal, beams...), MCP server, REST API, Word reports |
| [`fea_gui/`](fea_gui/) | Graphical interface (Qt + OpenCASCADE), help, wizards |
| [`installer/`](installer/) | Modular installer (`fea_setup.py`), version locks, slimming, packaging, CalculiX for Linux |
| [`adapters/`](adapters/) | Autodesk Fusion add-in (send full design or selected part/subassembly); SolidWorks and Inventor export macros (not yet verified in real CAD) |
| [`examples/`](examples/) | Example projects 1–10 (with results; example 10 = BESO topology optimization), test geometry |
| [`verification/`](verification/) | Benchmark problems against analytical solutions and cross-checks (MYSTRAN) – `python -m verification.run_all` |
| [`tests/`](tests/) | Automated tests (`python -m pytest`), `tests/gui/` = manual GUI smoke test scripts |
| [`docs/`](docs/) | Manuals (PDF / Word, HTML with a Czech / English switch) |
| `tools/` | Downloaded binaries (uv, Python, CalculiX, MYSTRAN) – not in git |
| `.venv/`, `dist/`, `scratch/` | Environment, packages, scratch work – not in git |
| [`CLAUDE.md`](CLAUDE.md) | Agent instructions and verified project facts |
| [`LICENSE`](LICENSE) | Application license (GNU GPL v3) |
| [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) | Third-party component licenses |

## Development

```bash
python -m pytest -m "not slow"         # Fast tests (plain `python -m pytest` also runs the slow benchmarks)
python -m verification.run_all [V1 …]  # Verification benchmarks
python tests/gui/gui_smoke.py OUT      # GUI smoke test walkthrough with screenshots
python installer/make_package.py       # dist/FEAStudio-<version>.zip + .tar.gz
```

## Documentation

Full manuals and guides (English and Czech):
- **User Manual**: [HTML](docs/FEA%20Studio%20Manual.html) (download and open in a browser; Czech / English switch in the header); the PDF and Word versions are in the installation packages
- **Testing and Verification Guide**: [PDF](docs/FEA%20Studio%20Testing%20Guide.pdf), [HTML](docs/FEA%20Studio%20Testing%20Guide.html)

## License

Copyright (C) 2026 FEA Studio authors. Licensed under GNU GPL version 3 or later (GPL-3.0-or-later), with an additional permission under section 7 in [`LICENSE-EXCEPTIONS.md`](LICENSE-EXCEPTIONS.md). Third-party components: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

### Data Sources

- **PragTic fatigue database** – S-N curve data in *Fatigue* step: Papuga, J.: PragTic – material database of fatigue properties (Fatigue Data Repository). Czech Technical University in Prague, Faculty of Mechanical Engineering. https://www.pragtic.com/vmat.php. The database belongs to its author and is **not distributed** with the software; fetch it for personal use via `python -m fea_core.fatigue.pragtic_fetch`. Every imported curve preserves record ID and literature source, and the Word report cites the database.
- Materials, friction, beam sections, and bolt data cite standards, data sheets, and literature per entry.

ANSYS, Abaqus, SOLIDWORKS, Autodesk Fusion, Inventor, Intel, Microsoft, and other marks belong to their respective owners; FEA Studio is not affiliated with them.
