# FEA Studio – SOLIDWORKS Macro

The VBA macro `FEAStudio_Export.bas` exports open parts or assemblies to FEA Studio in one click:
1. Exports the active document to **STEP AP214** in `%TEMP%\FEAStudio\export\<name>_<timestamp>.step`.
2. Writes adjacent metadata JSON `<name>_<timestamp>.json`: component file name, configurations, occurrence names, materials, and colors.
3. Launches FEA Studio with the STEP file. FEA Studio reads the JSON file and pairs materials against its library.

Supported versions: SOLIDWORKS 2020–2025 (64-bit, VBA 7.1).

> **Verification note**: Validated via mock object unit tests (`tests/test_cad_macros.py`).

## Setup

Save the macro to a persistent directory, preferably `<FEA Studio Path>\adapters\solidworks`.

1. Open a part in SOLIDWORKS and choose **Tools → Macro → New**.
2. Save as `FEAStudio.swp` (do not name it `FEAStudio_Export.swp` to avoid module name conflicts).
3. In the VBA editor, select **File → Import File...** (Ctrl+M) and choose `FEAStudio_Export.bas`.
4. Remove the empty initial module (e.g., `Module1`) without saving it.
5. Select **Debug → Compile VBAProject**, save (Ctrl+S), and close the editor.
6. Test run via **Tools → Macro → Run** → `FEAStudio.swp`.

### Add Toolbar Button

1. Open **Tools → Customize**, navigate to **Commands** tab, and select category **Macro**.
2. Drag the **New Macro Button** onto a toolbar or the CommandManager.
3. Configure the button:
   - **Macro**: browse to `FEAStudio.swp`; **Method**: `FEAStudio_Export.main`.
   - **Appearance**: select `FEAStudio_Export.bmp` (16 × 16 px).
   - **Tooltip**: "Send to FEA Studio".

## Application Path Resolution

The macro searches for FEA Studio in this order:
1. Environment variable `FEA_STUDIO_HOME`.
2. `%APPDATA%\FEAStudio\install.json` (`"home"` key).
3. Default path `%LOCALAPPDATA%\Programs\FEA Studio`.
4. Hardcoded constant `FEA_STUDIO_HOME_MAKRO` at the beginning of the macro file.
