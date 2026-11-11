# FEA Studio – Changelog

Upgrade: run setup.exe (Windows) or `.run` (Linux) over the existing installation, or extract a new archive and run `Install FEA Studio.bat update` / `./install.sh update`. The installer applies migrations (`installer/migrations.py`), removes obsolete files, and synchronizes modules.

## 1.4.0 (2026-11-06)

- **Fixes found by the full test run after the English conversion**: bolt clamping force with a contact joint (+13 % before), print advice never proposed more perimeters, GUI crash when showing the stress error estimate, shell edge supports rejected by the readiness check, mechanism energy plot, Czech input values accepted again by MCP (`tlak`, `teplota`, contact types, excitation, supports…), 361 texts shown in English in the Czech UI, translation engine no longer garbles multi-line summaries, F1 in a dialog shows the help of the focused field.
- **MCP**: `fea_report_word` writes English by default (`language="cs"` for Czech).
- **Installer**: adding the docs module after a slimmed installation restores the Qt QML libraries; remaining Czech-only installer texts bilingual; licence overviews (`licenses/`) in English.
- **PDF manuals** regenerated in Czech and English.
- **Program language chosen at installation** (setup.exe asks for the language of the wizard, the zip / .run installation asks in the console); the choice is kept in `language.txt` in the user settings and can be switched later in the program (Čeština / English).
- **Faster start of the program**: ~15 s → ~6 s (Czech), ~11 s → ~6 s (English) – optional step panels built at their first display, translation templates compiled on demand, stylesheet applied once.
- **English file names** (2026-11-07): launchers `Install FEA Studio.bat`, `Register MCP.bat`; examples `example1_bracket.fea` … `example10_bracket_topology.fea`, test models `examples/user_tests/T1_cantilever.step` …; help pictures of the examples in English. An update removes the files with the old names.
- **Licences**: a short `THIRD_PARTY_LICENSES.md` with the sources of the material, friction and print data.
- **Pop-out windows** (2026-11-07): the step panel, every panel section and the bottom Tables dock can be opened in a separate window (button next to the step tabs / in the section header / next to the table tabs, or View menu); closing the window returns it. The Materials panel and the Members tabs (nodes, members, cross-sections, supports, connections) are split into collapsible sections.

### Also in 1.4.0 (2026-11-05)

- **Project in English**: code, comments, messages and documentation are in English; the Czech UI, help and reports come from translation (`fea_core/i18n/cs.json`, `fea_gui/help/cs`).
- **Shorter help** pages and tooltips.
- **MCP tool groups and shorter descriptions**: only the `core` group is listed by default (`FEA_MCP_GROUPS` / `--groups` / `fea_tools_enable` with `tools/list_changed`, `fea_tools_groups`, `fea_tool_help`); tool-list context ~60k → ~27k tokens with all groups, ~6k by default; MCP idle RSS 95 → 82 MB. Groups save tokens only, they are not a permission boundary.
- **Program RAM released before solvers**: main process ~245 MB → 10–30 MB at solver start; MCP idle 118 → 93 MB; GUI idle 761 → 698 MB.
- **Security fixes**: HTML escaping of names, rich-text resource loading blocked (file / UNC), `tr()` ReDoS cap, `config` import NameError.

### Also in 1.4.0 – pre-release preparation (2026-11-04)

- **PragTic citation**: Dialogs, help, reports, and documentation explicitly cite the database (Papuga, J.: PragTic – material database of fatigue properties (Fatigue Data Repository). Czech Technical University in Prague, Faculty of Mechanical Engineering. https://www.pragtic.com/vmat.php). The database is not bundled; users download it via `python -m fea_core.fatigue.pragtic_fetch`.
- **Security hardening**: Restricted solver input injection, sanitized STEP relative and UNC path references, enforced HMAC session authentication for GUI IPC, secured REST endpoints, and isolated Python execution flags (`-P`).
- **License compliance**: GNU GPL-3.0-or-later with Section 7 exceptions (`LICENSE-EXCEPTIONS.md`), updated `THIRD_PARTY_LICENSES.md`, and clean decoupling of proprietary runtime dependencies.

### Also in 1.4.0 – first build (2026-10-30)

- **Native CAD import without CAD licenses**: Direct reading of Inventor (`.ipt`, `.iam`) and Fusion (`.f3d`, `.f3z`) models with BRep verification.
- **Assembly features & updates**: Automatic virtual bolt generation from assembly fasteners and geometry replacement with boundary condition remapping.
- **Security & export**: Hardened CAD archive unpacking, sanitized entity names against HTML injection, and added standard CSV/VTU result exports.

## 1.3.2 (2026-10-31)

- **Installer reliability**: Replaced PowerShell scripts with bundled Python execution, added support for Windows system root certificates and corporate HTTPS proxies, and added offline installer variant (`FEAStudio-<version>-offline-setup.exe`).
- **UI layout**: Finalized Studio × Tabulky unified header, phase tabs (Model – Assembly – Conditions – Mesh – Solve – Results), dockable tables, and standalone material property window.

## 1.3.1 (2026-10-20)

- **Bundled Python**: Included pre-configured Python 3.11 standalone runtime in Windows archives to eliminate runtime downloads.
- **Extraction resilience**: Added direct fallback extraction when archive unpacking encounters temporary antivirus file locks.

## 1.3.0 (2026-10-18)

- **BESO topology optimization**: Integrated BESO structural optimization with 3D density visualization, STL export, and Example 10.
- **Workflow & dynamics**: Condensed UI phases to 5 core stages, exact integration of piecewise-linear base excitation, and deterministic nonlinear contact solving.

## 1.2.0 (2026-10-17)

- **Fast Timoshenko beam solver**: Standalone 1D solver for beam assemblies with linear buckling, thermal loads, and EN 1993-1-1 checks.
- **Integration point recovery**: Evaluated section forces and 3D printing safety factors directly from solver integration points.

## 1.1.0 (2026-10-17)

- **Advanced physics**: Added plasticity models (bilinear, Ramberg-Osgood, Hollomon), local coordinate systems, EN 1990 load combinations, submodeling, and convection/radiation thermal tools.
- **Security & validation**: Added authenticated session keys for IPC, sanitized CSV injection characters, and extended the verification cases.

## 1.0.1 (2026-10-15)

- **Packaging & updates**: Fixed PySide6 shared file collisions during package sync, introduced manifest-driven updates, and generated Inno Setup and Linux `.run` installers.
- **CAD adapters & 3D printing**: Added SolidWorks and Inventor export macros, direct drag-and-drop STEP loading, and metal L-PBF material library entries.

## 1.0.0 (2026-10-15)

- **Initial release**: Linear statics, heat transfer, fatigue, bolted and welded joints, 3D printing analysis, beams/shells, modal/buckling, contacts, dynamics, mechanisms, MCP server, REST API, Word reports, and modular packaging.
