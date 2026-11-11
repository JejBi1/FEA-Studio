# FEA Studio

[🇬🇧 English](README.md) | **🇨🇿 Čeština**

MKP nad CalculiX a Gmsh: lineární statika a nelineární kontakt (tření, přesah, velké deformace), plasticita (jen Python API / MCP), teplo a teplotní napětí, vlastní frekvence a vzpěr, dynamika (harmonická, přechodová, náhodné kmitání, ráz), mechanismy (Exudyn), topologická optimalizace BESO, únava (S-N, ε-N, kritická rovina), šrouby a svary, pruty, 2D, skořepiny, 3D tisk, protokol Word. Import STEP i nativních dílů Inventoru / Fusionu (.ipt, .iam, .f3d) bez CAD, doplněk pro Fusion, REST API a server MCP pro AI agenty (skupiny nástrojů: ve výchozím stavu jen `core`, další přes `FEA_MCP_GROUPS` nebo `fea_tools_enable`). Windows i Linux.

---

## Instalace a spuštění

| | Windows | Linux |
|---|---|---|
| instalace (po modulech, bez práv správce) | `Install FEA Studio.bat` | `./install.sh` |
| spuštění GUI | `FEA Studio.bat` | `./fea-studio` |
| příkazová řádka | `.venv\Scripts\python -m fea_core --help` | `.venv/bin/python -m fea_core --help` |
| registrace MCP serveru do AI klientů | `Register MCP.bat` | `.venv/bin/python -m fea_core.mcp.setup` |

Moduly (jádro, GUI, únava…) a jejich velikosti: [installer/README.md](installer/README.md). Další moduly později:
`python -m fea_core.modules install <modul>` nebo v GUI Nástroje → Moduly programu.

## Struktura

| Složka / soubor | Obsah |
|---|---|
| [`fea_core/`](fea_core/) | výpočetní jádro bez GUI (studie, síť, ccx, výsledky, únava, spoje, tisk, teplo, pruty…), MCP server, REST API, protokol Word |
| [`fea_gui/`](fea_gui/) | grafické rozhraní (Qt + OpenCascade), nápověda, průvodci |
| [`installer/`](installer/) | instalátor (`fea_setup.py`), zámky verzí, prořezání, sestavení balíčků, CalculiX pro Linux |
| [`adapters/`](adapters/) | doplněk pro Autodesk Fusion (odeslání celého designu nebo vybraného dílu / podsestavy); exportní makra pro SolidWorks a Inventor (zatím neověřená ve skutečném CAD) |
| [`examples/`](examples/) | ukázkové projekty 1–10 (s výsledky; ukázka 10 = topologická optimalizace BESO konzoly), testovací geometrie |
| [`verification/`](verification/) | ověřovací úlohy proti analytickým řešením, křížové kontroly (MYSTRAN) – `python -m verification.run_all` |
| [`tests/`](tests/) | automatické testy (`python -m pytest`), `tests/gui/` = skripty GUI smoke testů (spouští se ručně) |
| [`docs/`](docs/) | příručky (PDF / Word anglicky, HTML s přepínačem čeština / angličtina) |
| `tools/` | stažené nástroje (uv, Python, CalculiX, MYSTRAN) – není v gitu |
| `.venv/`, `dist/`, `scratch/` | prostředí, hotové balíčky, pracovní pokusy – nejsou v gitu |
| [`CLAUDE.md`](CLAUDE.md) | podrobné pokyny a ověřená fakta pro agenty (architektura, konvence, výsledky ověření) |
| [`LICENSE`](LICENSE) | licence programu (GNU GPL v3) |
| [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) | licence použitých komponent |

## Vývoj

```bash
python -m pytest -m "not slow"         # rychlé testy (samotné `python -m pytest` spustí i pomalé úlohy)
python -m verification.run_all [V1 …]  # ověřovací úlohy
python tests/gui/gui_smoke.py OUT      # průchod celým GUI se snímky
python installer/make_package.py       # dist/FEAStudio-<verze>.zip + .tar.gz
```

## Dokumentace

Kompletní příručky a návody (česky i anglicky):
- **Uživatelská příručka**: [HTML](docs/FEA%20Studio%20Manual.html) (stáhnout a otevřít v prohlížeči, v záhlaví přepnout na češtinu); verze PDF a Word jsou v instalačních balíčcích
- **Návod k testování a verifikaci**: [HTML](docs/FEA%20Studio%20Testing%20Guide.html), [PDF anglicky](docs/FEA%20Studio%20Testing%20Guide.pdf)

## Licence

Copyright (C) 2026 autoři FEA Studia. Licence GNU GPL verze 3 nebo (podle vaší volby) jakékoli pozdější (GPL-3.0-or-later), s doplňkovým svolením podle čl. 7 v [`LICENSE-EXCEPTIONS.md`](LICENSE-EXCEPTIONS.md). Součásti třetích stran: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

### Zdroje dat

- **Databáze PragTic** – únavová data S-N v kroku *Únava*: Papuga, J.: PragTic – material database of fatigue properties (Fatigue Data Repository). Czech Technical University in Prague, Faculty of Mechanical Engineering. https://www.pragtic.com/vmat.php. Databáze patří autorovi a s programem se **nešíří**; pro vlastní potřebu ji stáhnete příkazem `python -m fea_core.fatigue.pragtic_fetch`. U každé převzaté křivky zůstává číslo záznamu a původní pramen, protokol Word databázi cituje.
- Materiály, tření, profily a šrouby mají zdroj (norma, technický list, literatura) u každého záznamu.

ANSYS, Abaqus, SOLIDWORKS, Autodesk Fusion, Inventor, Intel, Microsoft a další názvy jsou ochranné známky svých vlastníků; FEA Studio s nimi není spojeno.
