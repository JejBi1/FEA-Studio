# REST API FEA Studia

Tenká vrstva (FastAPI) nad nástroji MCP serveru (`fea_core/mcp`): stejné nástroje, názvy, popisy, schémata
parametrů i chybové hlášky. Určeno pro CAD adaptéry (`adapters/`), skripty a jiné aplikace. Jedna relace
(studie, síť, výsledky) na proces serveru – jako MCP server.

## Spuštění

```
.venv\Scripts\python.exe -m fea_core.rest [--host 127.0.0.1] [--port 8765] [--workdir SLOŽKA] [--token TOKEN]
```

- Výchozí adresa `127.0.0.1` = přístup jen z tohoto počítače. Jiná adresa (např. `0.0.0.0`) se bez tokenu
  odmítne – server spouští výpočty, čte a zapisuje soubory.
- `--token` (nebo proměnná `FEA_REST_TOKEN`) → každý požadavek na `/api/*` musí mít hlavičku
  `Authorization: Bearer <token>`, jinak 401 (s tokenem jsou za ověřením i `/docs`, `/redoc`, `/openapi.json`).
- Bez tokenu server přijme jen hlavičku `Host` 127.0.0.1 / localhost / [::1] (ochrana proti DNS rebinding – cizí
  stránka s doménou přesměrovanou na 127.0.0.1) → 403. POST/PUT/PATCH/DELETE s hlavičkou `Origin` cizí stránky
  (např. multipart `/api/import` z webu – CSRF) se odmítnou vždy (403); povolený `Origin` = loopback nebo stejný
  server. Adaptéry (urllib, bez `Origin`) nejsou dotčené.
- GUI spuštěné přes `/api/open-gui` je odpojený proces – časový limit nástrojů (ukončení podprocesů) ho ani jeho
  výpočty neukončí.
- `/api/import` vrací ve `warnings` i varování kontroly geometrie STEP (plochy bez tělesa, neplatná tělesa).
- `--workdir` = kořen pracovních složek (výchozí `%LOCALAPPDATA%/FEAStudio/mcp` nebo `FEA_MCP_WORKDIR`).
  Importy z CADu jdou do `<workdir>/imports/`, studie do `<workdir>/<název>_<čas>/`.
- Interaktivní dokumentace (OpenAPI): `http://127.0.0.1:8765/docs`.
- Port 8765 používají i CAD adaptéry (`FEA_REST_URL`). MCP server v režimu HTTP má výchozí port 8766 → oba servery
  mohou běžet souběžně bez `--port`.

## Endpointy

| Metoda a cesta | Popis |
|---|---|
| `GET /api/status` | verze, otevřená studie, `busy` (běžící operace), výsledky ano/ne, poslední import. Odpovídá i během výpočtu. |
| `GET /api/tools` | seznam nástrojů `{name, description, parameters}` (JSON schéma parametrů = MCP `inputSchema`) |
| `GET /api/tools/{název}` | jeden nástroj |
| `POST /api/tools/{název}?images=url\|base64\|both` | zavolá nástroj, tělo = JSON objekt argumentů → `{tool, result, images?, time_s}` |
| `GET /api/study` | aktuální definice studie (JSON modelu `Study`) |
| `POST /api/import` | multipart: `file` (STEP), volitelně `metadata` (JSON z adaptéru), `open` (výchozí true), `name`, `assign_materials` (výchozí true) |
| `GET /api/files/{cesta}` | stažení souboru z pracovní složky (PNG, .docx, .fea, .step…); `@workdir/…` = pracovní složka studie mimo kořen |
| `POST /api/open-gui` | `{"path": …}` (relativně ke kořeni nebo absolutně; .step/.stp/.fea/.json – .json = studie z `fea_study_save`, ne metadata z CADu → 400), bez cesty poslední import, jinak uložená studie relace → spustí `pythonw -m fea_gui <soubor>` jako samostatný proces; studii JSON GUI otevře jako novou neuloženou studii (Uložit = projekt .fea) |

`result` = JSON, který nástroj vrací LLM (u MCP text). Každý nástroj, který vrací obrázek – render a grafy
(`fea_model_render_image`, `fea_results_render_image`, `fea_results_render_mode`, `fea_frame_render`,
`fea_print_render_shell`, `fea_print_voxel_run`, `fea_print_voxel_render`, `fea_twod_render`, `fea_shell_render`,
`fea_contact_render`, `fea_dyn_render`, `fea_dyn_plot`, `fea_motion_render`, `fea_results_line` a
`fea_results_history` s `plot=true`) – vrací navíc `images: [{url, path, mime_type, base64?}]`; obrázek mimo
pracovní složku (`output_path` jinde) se vrátí vždy jako base64.

Chyby: `{"error": "česká zpráva"}` – 400 chyba nástroje nebo neplatné argumenty (i neznámý název argumentu),
404 neznámý nástroj / soubor, 403 cesta mimo pracovní složku, 409 server zaneprázdněn (síťování, výpočet),
401 token, 500 neočekávaná chyba jádra (typ + text).

### Import z CADu

Metadata (vše volitelné):

```json
{"source": "Autodesk Fusion", "document": "Držák", "units": "mm",
 "parts": [{"name": "Deska:1", "names": ["Deska", "Body1"], "material": "Steel",
            "material_props": {"E_MPa": 210000, "nu": 0.3, "rho_t_mm3": 7.85e-9, "Re_MPa": 235, "Rm_MPa": 360},
            "color": [0.63, 0.63, 0.63], "attributes": {"skupina.klíč": "hodnota"}}]}
```

- Stejný JSON vedle STEP (`<název>.json`, „sidecar“) čte i GUI při otevření STEP – párování a materiály jsou společné
  (`fea_core.cad_import`). Zapisuje ho doplněk Fusion i makra SOLIDWORKS a Inventor (`adapters/`).
- Díly ze STEP (`fea_core.geometry.read_step`) se párují s `parts` podle názvu: `name`, `names`, `component`,
  `occurrence`, `body` (ne `instances` – výskyty ze stromu jsou jen informace); bez ohledu na velikost písmen
  nejdřív přesně, pak bez přípon výskytu (`:1`, `<1>`, ` (2)`, `_2`) a nakonec i bez přípony výskytu SolidWorks
  `-1` (odebere se jen poslední „-číslo“, jednou a až po ostatních příponách: „Deska-1“ → „Deska“). Z více
  vyhovujících záznamů se vezme první dosud nepoužitý.
- Materiál se přiřadí (`fea_materials_assign`), jen když odpovídá knihovně: přesný název, nebo stejná skupina
  převodníku značení `grades` (např. „S235“ → Ocel S235, „Stainless Steel AISI 304“ → Nerez 1.4301, „Aluminum 6061“
  → Hliník EN AW-6061 T6), ne přibližný ekvivalent. Uvádí-li stav materiálu (T6, O, H111, +N, +QT…) oba názvy, musí
  se shodovat (O = H111); SOLIDWORKS „6061 Alloy“ je žíhaný stav (O) a k T6 se nepřiřadí. `material_props`
  (`Re_MPa`, jinak `Rm_MPa`) ±20 % jiný stav potvrdí, a pokud nesouhlasí, přiřazení zruší. Obecné a neznámé názvy
  („Steel“, „Plain Carbon Steel“) se jen vrátí v `parts[].material_match` s nápovědou;
  vlastnosti z CADu jsou v `cad_material_props` (nepřebírají se automaticky). Pozor: první přiřazený materiál se
  stane i výchozím materiálem studie (chování `fea_materials_assign`) – `material_effective` a upozornění to ukazují.
  S `assign_materials=false` se materiál nepřiřadí; nalezený materiál knihovny je v `parts[].material_suggested`
  a upozornění „není v knihovně“ se týká jen materiálů, které v knihovně opravdu nejsou.
- Odpověď: `step_path`, `metadata_path`, `step_url`, `study` (výsledek `fea_study_new`), `parts` (spárování,
  materiál, barva, atributy), `warnings`.

## Příklady

curl (Git Bash):

```bash
curl http://127.0.0.1:8765/api/status
curl -F "file=@drzak.step" -F "metadata=<drzak.json" http://127.0.0.1:8765/api/import
curl -X POST -H "Content-Type: application/json" -d '{"part": "*", "material_name": "Ocel S235"}' \
     http://127.0.0.1:8765/api/tools/fea_materials_assign
curl -X POST -H "Content-Type: application/json" -d '{"x": 0}' http://127.0.0.1:8765/api/tools/fea_geometry_find_faces
```

Python (httpx):

```python
import httpx, json
c = httpx.Client(base_url="http://127.0.0.1:8765", timeout=900)   # headers={"Authorization": "Bearer …"}
r = c.post("/api/import", files={"file": open("konzola.step", "rb")},
           data={"metadata": json.dumps({"document": "Konzola", "parts": [{"name": "Konzola", "material": "S235"}]})})
tool = lambda name, **kw: c.post(f"/api/tools/{name}", json=kw).raise_for_status().json()["result"]
face = tool("fea_geometry_find_faces", x=0)["faces"][0]["id"]
tool("fea_constraint_add_fixed", name="vetknuti", selection=[face])
tool("fea_loadcase_create", name="Sila")
tool("fea_load_add_force", case_name="Sila", name="konec", selection={"x": 200}, force=[0, 0, -100])
print(tool("fea_solve_run")["steps"][0]["max_von_mises_MPa"])
img = c.post("/api/tools/fea_results_render_image", json={"field": "mises"}).json()["images"][0]
open("mises.png", "wb").write(c.get(img["url"]).content)
```

## Postup pro CAD adaptéry

1. `GET /api/status` (krátký timeout) – běží server? Ne → uložit STEP + JSON do složky a spustit GUI
   (`<FEA_STUDIO_HOME>\.venv\Scripts\pythonw.exe -m fea_gui <step>`, záloha `FEA Studio.bat <step>`).
2. Export STEP s názvy těles (STEP nese názvy, barvy, případně vrstvy; materiály a atributy CADu obvykle ne →
   posílají se v metadatech).
3. `POST /api/import` (multipart: `file`, `metadata`, `open=true`).
4. Volitelně `POST /api/open-gui` (otevře poslední import v GUI).

Adaptér: `adapters/fusion` (doplněk Fusion). Jiné CADy: STEP + metadata přímo na `POST /api/import`.

## Implementace

- `registry.py` – registrátor, který funkce nástrojů z `fea_core.mcp.tools_*` jen posbírá; schéma a ověření
  argumentů přes `Tool.from_function` SDK `mcp` (shodné s MCP serverem). Resource a prompty se ignorují.
- `app.py` – `create_app(session=None, base_dir=None, token=None, gui_launcher=None)`; synchronní endpointy běží ve
  vláknech FastAPI, souběh hlídá zámek relace (`Session.use`); `busy` → 409 hned.
- `imports.py` – uložení importu, párování názvů, hledání materiálu v knihovně.
- Testy: `tests/test_rest.py` (TestClient; slow e2e konzola V1), `tests/test_adapters.py` (klienti adaptérů proti
  skutečnému serveru ve vlákně).
