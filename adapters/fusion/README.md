# FEA Studio Add-in for Autodesk Fusion

Adds a **Send to FEA Studio** button under the **FEA STUDIO** panel in Design workspaces (SOLID, SURFACE, MESH, FORM, SHEET METAL, PLASTIC, SKETCH, ASSEMBLY) and in UTILITIES → ADD-INS.

Workflow:
1. Exports the active design to STEP (AP214) – entire model or selected parts/subassemblies.
2. Extracts visible body metadata: instance name, body name, component hierarchy, material properties (elastic modulus, Poisson's ratio, density, yield/tensile strength converted to MPa and t/mm³), color, and attributes.
3. If the FEA Studio REST server is active, posts STEP + JSON to `POST /api/import` and opens the GUI; otherwise saves files to `%TEMP%\FEAStudio\export` and launches FEA Studio locally.

## Partial Export (Parts or Subassemblies)

In the command dialog, select **Entire design** or **Selected parts / subassemblies**. When selecting components in Fusion before clicking the button, the selection pre-populates automatically.
- Selected parts retain their exact global assembly coordinates so constraints, loads, and joints align properly.
- Unselected components are temporarily hidden during STEP export and restored immediately afterward.

## Installation

Run via the FEA Studio installer:
```bash
.venv\Scripts\python.exe installer\fea_setup.py fusion          # Uninstall: fusion --remove
```
Restart Fusion to load the add-in.

Manual installation:
1. Copy the `FEAStudio` folder into `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\`.
2. In Fusion, open Tools → Add-Ins → Scripts and Add-Ins → Add-Ins tab, select `FEAStudio`, and check "Run on Startup".
3. Configure settings via `settings.json` (copy from `settings.example.json`) or environment variables:
   - `FEA_REST_URL`: REST API address (default: `http://127.0.0.1:8765`).
   - `FEA_REST_TOKEN`: Optional authentication token.
   - `FEA_STUDIO_HOME`: Path to FEA Studio installation root.
