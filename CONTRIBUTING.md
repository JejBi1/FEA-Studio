# Contributing to FEA Studio

Thank you for your interest. FEA Studio is maintained by a single volunteer, so reviews may take a while; small, focused
pull requests with tests are the easiest to accept.

- Bugs and feature requests: open an issue (templates: bug report / feature request). For a crash, attach the crash log –
  **Help → About FEA Studio → Open log folder** (the `crash-<time>.log` files).
- Security problems: **do not** open an issue – see [`SECURITY.md`](SECURITY.md).

## Licence of contributions

FEA Studio is licensed under the **GNU GPL version 3 or later** (`GPL-3.0-or-later`, [`LICENSE`](LICENSE)) with an
additional permission under section 7 ([`LICENSE-EXCEPTIONS.md`](LICENSE-EXCEPTIONS.md): linking with Intel oneMKL,
Microsoft MPI and the CAD / Abaqus scripting runtimes). By submitting a contribution you agree that it is licensed under the
GPL-3.0-or-later **together with that additional permission**. Only submit code you wrote yourself or that is available under
a compatible licence – name its source and licence in the pull request; third-party components are listed in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

Do not add personal data, company names, real customer models or data without a clear licence (the PragTic fatigue database
is deliberately not distributed).

## Development setup

```bash
python installer/fea_setup.py install verify --profile standard --no-shortcuts --no-mcp --no-fusion   # .venv + CalculiX
.venv/bin/python -m pytest -m "not slow"          # Windows: .venv\Scripts\python
.venv/bin/python -m verification.run_all V1 S1    # verification cases against analytical references
```

Optional modules (GUI, Numba acceleration, MCP / REST, NGSolve, …) are listed in `fea_core/modules.json`; tests of modules
that are not installed are skipped. The same steps run in CI (`.github/workflows/tests.yml`) on Linux and Windows.

## Code rules

- **English** in code, comments, messages, docstrings and documentation. Czech appears only through the translation catalogue
  (`fea_core/i18n/cs.json`) and the Czech help pages (`fea_gui/help/cs/`).
- Units mm, N, MPa, t, s everywhere in the core.
- Keep the core (`fea_core`) free of GUI imports; optional dependencies are imported through `fea_core.modules.need()`.
- User text that reaches a solver input deck goes through `fea_core.ccx.deck_text()`.
- Names from files (parts, load cases, selections) are untrusted: show them in Qt as plain text or escaped.
- Lint: `ruff check .` (rule set in `pyproject.toml` – real errors only, no formatter). Do not reformat unrelated code.
- Every new computational feature needs an **independent reference** (analytical solution, literature or a full model) as a
  verification case in `verification/` with real tolerances, plus fast unit tests in `tests/`.
- Every interactive GUI element gets a short tooltip (≤ 80 characters, `widgets.help_tip`); UI strings use only characters
  present in Segoe UI (`tests/test_ui_glyphs.py`).

## Translations (i18n workflow)

The source language is English; the Czech text is looked up at display time.

1. Write new user-visible texts in English in the code. Do not translate persisted identifiers (JSON / `.fea` keys, enum
   values, MCP tool and argument names, solver cards).
2. `python -m fea_core.i18n.extract --todo todo.json --where` lists the texts without a Czech translation and where they are.
3. Translate them in `todo.json` (`""` = not a text for the user; terminology in `fea_core/i18n/glossary.md`).
4. `python -m fea_core.i18n.extract --merge todo.json` adds them to `fea_core/i18n/cs.json`; `tests/test_i18n.py` checks
   that nothing is missing.
5. Help: the English page `fea_gui/help/<slug>.html` is the source; update the Czech page `fea_gui/help/cs/<slug>.html` and
   run `python -m fea_gui.help_pdf --mark-translated <slug>`.

## Pull requests

- One topic per pull request; describe what changed and how it was verified (tests, verification cases, GUI check).
- Run the fast tests and the related verification cases before submitting.
- Add a line to `installer/CHANGELOG.md` under "Unreleased" for user-visible changes.
