# CAD Adapters for FEA Studio

The "Send to FEA Studio" integration exports a CAD model as STEP (AP214) alongside part metadata (name, material, color) in `<name>.json` next to the STEP file, then opens it in FEA Studio. When loading a STEP file, FEA Studio reads the adjacent JSON, matches parts by name, and assigns library materials (`fea_core/cad_import.py`). The Fusion add-in can also transfer models directly over the REST API (`python -m fea_core.rest`, see `fea_core/rest/README.md`).

For detailed user instructions on manual exports and formatting, see help topic `fea_gui/help/cad.html` (or `fea_gui/help/en/cad.html`).

| Directory | CAD System | Integration Type | Verification Status |
|---|---|---|---|
| `fusion/` | Autodesk Fusion | Add-in `FEAStudio/` – entire design or selected parts/subassemblies | Export and metadata verified on real assembly via Fusion MCP; selection tree verified in unit tests. |
| `solidworks/` | SOLIDWORKS 2020–2025 | VBA Macro `FEAStudio_Export.bas` – STEP AP214 + adjacent JSON | Verified via mock object tests (`tests/test_cad_macros.py`). |
| `inventor/` | Autodesk Inventor 2021–2026 | External iLogic rule `FEAStudio_Export.iLogicVb` – STEP AP214 + JSON | Verified via mock object tests (`tests/test_cad_macros.py`). |
| — | Other CAD (Solid Edge, Creo, CATIA, NX, Onshape, FreeCAD, Rhino...) | Manual STEP AP214 export | Standard STEP import; drag-and-drop into FEA Studio window or Send To shortcut. |

The Fusion add-in includes a standalone REST client that runs within Fusion's internal Python runtime. SOLIDWORKS and Inventor macros save STEP and JSON to `%TEMP%\FEAStudio\export` and launch FEA Studio directly. The application path is resolved via `FEA_STUDIO_HOME`, `%APPDATA%\FEAStudio\install.json`, or the default install directory.
