# FEA Studio – Autodesk Inventor iLogic Rule

The external iLogic rule `FEAStudio_Export.iLogicVb` (VB.NET) exports open parts or assemblies to FEA Studio:
1. Exports active document to **STEP AP214** in `%TEMP%\FEAStudio\export\<name>_<timestamp>.step`.
2. Writes adjacent metadata JSON `<name>_<timestamp>.json` (UTF-8 without BOM): occurrence name, part number, component file, material, and color.
3. Launches FEA Studio with the exported STEP file. FEA Studio reads the JSON file and pairs materials against its library.

Supported versions: Inventor 2021–2026. Uses VB 2012 syntax required by iLogic.

> **Verification note**: Validated via mock object unit tests (`tests/test_cad_macros.py`) and compiler checks outside Inventor.

## Setup

1. **Configure external rules directory**:
   - In Inventor, navigate to **Tools → Options → iLogic Configuration**.
   - Under **External Rule Directories**, click **+**, select `<FEA Studio Path>\adapters\inventor`, and confirm with OK.
2. **Access rule**:
   - Open **Manage → iLogic → iLogic Browser**, go to **External Rules** tab, and locate **FEAStudio_Export**.
   - Right-click and choose **Run Rule**.
3. **Add toolbar button (Inventor 2023+)**:
   - Right-click ribbon → **Customize User Commands** → **Ribbon** tab.
   - Filter by **iLogic rules**, select environment (**Part** or **Assembly**), add `FEAStudio_Export`, and assign button size.

## Application Path Resolution

The rule searches for FEA Studio in this order:
1. Environment variable `FEA_STUDIO_HOME`.
2. `%APPDATA%\FEAStudio\install.json` (`"home"` key).
3. Default path `%LOCALAPPDATA%\Programs\FEA Studio`.
4. Hardcoded constant `FEA_STUDIO_HOME_PRAVIDLO` at the top of the script.
