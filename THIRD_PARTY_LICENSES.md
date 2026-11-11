# Third-party licenses

FEA Studio is licensed under the **GNU GPL v3 or later** (`LICENSE`), with an additional permission for Intel MKL,
MS-MPI and CAD APIs (`LICENSE-EXCEPTIONS.md`). The components below keep their own licenses. Installation packages
contain the full license texts in the `licenses/` folder.

## What this means for you

- **You may use FEA Studio for any work, including commercial engineering work.**
- **You may share and modify it** under the GPL: keep the license, publish your changes under the GPL as well.
- **You may not make it closed-source.** Gmsh and CalculiX are GPL; a proprietary product would need other licenses.
- Your own CAD licenses apply to those programs (Fusion Personal, for example, is non-commercial).

## Components

| Component | License | Used for |
|---|---|---|
| CalculiX 2.23 | GPL-2.0 | Solver (separate program). Packages carry `ccx_static.exe` (Windows, from dhondt.de) and `ccx_spooles` (Linux, `installer/linux/build_ccx.sh`) together with the CalculiX source archive |
| CalculiX `ccx_i4` (with Intel MKL PARDISO) | GPL-2.0 + Intel licence | **Not distributed**: the installer extracts it from the original dhondt.de archive (Windows) or builds it (Linux) on your machine, only with the `pardiso` module |
| Gmsh | GPL-2.0-or-later | Meshing (separate process); the offline setup also carries its source |
| Open CASCADE (cadquery-ocp), VTK | LGPL-2.1 with exception, BSD-3 | STEP import, 3D view |
| Qt (PySide6) | LGPL-3.0 | User interface |
| NGSolve / Netgen | LGPL-2.1 | Solver for large models (optional) |
| hypre, mpi4py, MPICH | Apache-2.0 / MIT, BSD-3, BSD-like | Parallel solver (optional) |
| MS-MPI | Microsoft redistributable | Parallel solver on Windows (installed only with your consent) |
| Intel MKL, OpenMP, TBB | Intel licenses | PARDISO solver – downloaded from PyPI during installation, not shipped |
| numpy, scipy, numba, matplotlib | BSD / PSF-style | Computation and plots |
| sectionproperties, shapely | MIT, BSD-3 | Beam cross-sections |
| Exudyn, Project Chrono | BSD-3 | Mechanisms (optional) |
| python-docx, lxml | MIT, BSD-3 | Word reports |
| mcp, fastapi, uvicorn, pydantic | MIT / BSD-3 | MCP server and REST API |
| pyNastran, MYSTRAN | BSD-3, MIT | Verification only |
| Python, uv, micromamba | PSF, MIT / Apache-2.0, BSD-3 | Runtime and installer |

The exact versions are in `installer/locks/`.

## Source code of the GPL / LGPL binaries

Every package carries `licenses/SOURCES.md` (where the complete source of each shipped GPL / LGPL binary is, with SHA-256
digests, and a written offer for the rest) and `licenses/sources/` with the CalculiX 2.23 source archive
(https://www.dhondt.de/ccx_2.23.src.tar.bz2). The offline setup additionally carries the Gmsh 4.15.2 source
(https://gmsh.info/src/). The Linux CalculiX build script and the digests of the SPOOLES, ARPACK and LAPACK archives it
uses are in `installer/linux/build_ccx.sh`.

## Data

- **PragTic fatigue database** – *Papuga, J.: PragTic – material database of fatigue properties (Fatigue Data
  Repository). Czech Technical University in Prague, Faculty of Mechanical Engineering. https://www.pragtic.com/vmat.php*
  The database has no license for redistribution, so it is **not included**. You can download it for your own use:
  `python -m fea_core.fatigue.pragtic_fetch`.
- Materials, friction coefficients, steel sections, bolt and weld tables: factual values with the source given for each
  entry in the program. Texts of standards are not included.

### Sources of the material and friction data

Part of the built-in material library and of the other tables was taken from these databases, data sheets and
publications (each material in the program shows its own source):

- **Standards** – EN 10025-2, EN 10025-6, EN ISO 683-1, -2, -17, EN 10084, EN 10088-1, -3, EN 10089,
  EN ISO 4957, EN 1561, EN 1563, EN 754-2, EN 755-2, EN 1706, EN 12163, EN 12164, EN 13601, ASTM B94, B221, B348, B637,
  F2924, AMS 5662, ISO 5832-3, EN 1993-1-1, EN 1993-1-2, EN 1993-1-8, EN 1090-2, DIN 68364.
- **Handbooks** – MIL-HDBK-17-2F (Composite Materials Handbook); ASM Handbook; USDA Forest Products Laboratory: Wood
  Handbook.
- **Books** – Daniel, I. M., Ishai, O.: Engineering Mechanics of Composite Materials. Barbero, E. J.: Finite Element
  Analysis of Composite Materials. Gibson, L. J., Ashby, M. F.: Cellular Solids (1997).
- **Manufacturer data sheets** – metal 3D printing: Renishaw (RenAM 500 material data sheets), AddUp (Ti6Al4V ELI),
  3D Systems (Certified Scalmalloy), APWORKS; plastic 3D printing: Bambu Lab, Prusa Research (Prusament), Stratasys
  (Ultem 9085), EOS (PA 2200), ThermaX (PEEK); others: SSAB (Hardox 450), Toray (M40J).
- **Papers** – Gao et al. 2025, Materials 18(15):3475, doi:10.3390/ma18153475 (Ti-6Al-4V L-PBF);
  Xie et al. 2023, J. Mater. Res. Technol. 23:3322–3336 (CuCrZr L-PBF);
  Ezeh, O. H., Susmel, L.: Int. J. Fatigue 126 (2019) 319–326 (fatigue of printed PLA);
  Song et al., Mater. Des. 123 (2017) and Khosravani et al., Sci. Rep. 12 (2022) 1016 (anisotropy of printed lines);
  Waseem et al. 2020 (creep of PLA); Stoimenov et al. 2024 (friction).
- **Friction coefficients** – Roymech (Coefficients of friction), The Engineering ToolBox (Friction coefficients),
  Bayer: Snap-Fit Joints for Plastics, Ensinger, DuPont (Delrin), EN 1090-2 and EN 1993-1-8 (slip factors).
- **Grade converter** – designations compiled from the standards (EN 10027 and national standards); Czech ČSN pairs
  checked against cadforum.cz (the table itself is not copied).

Only the numerical values are used; the values are approximate and must be checked against the data of the actual
material supplier.

## Not included

Autodesk Fusion, Inventor and SOLIDWORKS are not part of FEA Studio. The add-ins and macros for them are GPL code;
the CAD programs need their own licenses. The native reader of `.ipt`, `.iam`, `.f3d` files is independent code without
any Autodesk code or libraries; always check critical models against a STEP export.

Trademarks belong to their owners. FEA Studio is not affiliated with them. This summary is not legal advice.
