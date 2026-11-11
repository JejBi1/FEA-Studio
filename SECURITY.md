# Security policy

FEA Studio is a single-maintainer, volunteer project. Reports are handled on a best-effort basis; there is no bug bounty.

## Reporting a vulnerability

Please report security problems **privately** through GitHub's private vulnerability reporting:
open the repository on GitHub → **Security** → **Report a vulnerability**. Do not open a public issue, pull request or
discussion for a suspected vulnerability.

Useful content of a report:

- the affected version (Help → About FEA Studio, or `python -c "import fea_core; print(fea_core.__version__)"`) and the
  platform (Windows / Linux),
- the component (see the scope below) and the steps or a minimal file that reproduce the problem,
- what an attacker gains (for example code execution, reading or writing files outside the work folder, access to the
  REST / MCP interface of another user).

Please do not include personal data, credentials or confidential CAD models in the report; a small synthetic file is enough.

You can expect an acknowledgement when the report has been read, and a note when a fix is released (see
[`CHANGELOG.md`](CHANGELOG.md)). Please give the maintainer reasonable time to release a fix before you publish details.

## Scope

In scope – the code in this repository:

- reading of untrusted input files: project files (`.fea`), study JSON, STEP / BREP, native CAD files (`.ipt`, `.iam`,
  `.f3d`, `.f3z`), CSV load histories, material and preset libraries,
- the REST API (`python -m fea_core.rest`) and the MCP server (`python -m fea_core.mcp`), including their token and
  Host / Origin checks, and the local IPC bridge between the MCP server and the GUI,
- the CAD add-in and macros in `adapters/` (Autodesk Fusion add-in, SOLIDWORKS / Inventor macros),
- the installer and the update mechanism in `installer/` (downloads, digests, extraction of archives),
- text that ends up in solver input decks (CalculiX `.inp`, MYSTRAN `.bdf`).

Out of scope:

- vulnerabilities in third-party programs and libraries (CalculiX, Gmsh, OpenCASCADE / OCP, Qt, NGSolve, hypre, MKL,
  Python packages) – please report those to their projects; a report is welcome when FEA Studio uses them in an unsafe way,
- the REST / MCP interface exposed on a network on purpose without a token, or a local administrator / another process
  of the same user account (the program trusts its own user account),
- the correctness of engineering results – please open a normal issue with the bug report template instead.

## Supported versions

Only the latest release receives security fixes.
