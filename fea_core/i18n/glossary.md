# FEA Studio – EN → CS translation guide (Czech UI)

Source language of the program is English; `cs.json` maps the English texts to Czech. Translate UI texts, tooltips, messages,
errors and report (Word) texts into natural, concise, professional Czech as used in Czech engineering software
(Ansys / SolidWorks Simulation / Abaqus localisations, Czech engineering practice).

## Rules
- Output = JSON with exactly the same keys as the todo file ("exact" and "templates" objects), values = Czech text.
- Value "" (empty) = do not translate: the text is not human-readable UI (file formats, solver input lines, codes,
  identifiers, regex, standard designations like "EN 1993-1-8" alone, material grades).
- Templates: keep every placeholder {0}, {1}… exactly (each one must appear in the translation, order may change).
  Keep "{{" / "}}" as is.
- Keep: leading/trailing spaces, "
" line breaks (same count where possible), HTML tags and attributes, "&" accelerators
  (move them to a sensible letter), keyboard shortcuts (Ctrl+Z, Shift+F1), units (mm, N, MPa, °C, W/m²…), symbols
  (σ, τ, λ, Ø, ×, ≤, →, ⚠, •), standard references (VDI 2230, EN 1993-1-9, ISO 898-1, IIW), solver names (ccx,
  CalculiX, NGSolve, hypre, PARDISO, MYSTRAN, Gmsh), file extensions.
- Quotes “…” → Czech „…“; decimal point inside text numbers (0.7·t, 1.5) → decimal comma (0,7·t, 1,5).
- Keep similar length (UI space is limited); keep title-case/sentence-case like the source ("Add" → "Přidat").
- Imperative for actions ("Generate mesh" → "Vytvořit síť"), sentence case for labels, no trailing period where the source has none.
- "Step n" = "Krok n".
- The "Solve" step is "Řešení"; the phase "Calculation" is "Výpočet" (do not confuse with "Analyses" = optional steps menu).

## Glossary (use consistently)
| English | Czech |
|---|---|
| study | studie |
| part / parts | díl / díly |
| assembly | sestava |
| face, edge, point (vertex), body | plocha, hrana, bod, těleso |
| selection | výběr (ploch) |
| support / supports (step "Supports") | vazba / vazby (krok) |
| fixed support | vetknutí |
| prescribed displacement | předepsaný posuv |
| cylindrical support | válcová vazba |
| remote support / remote load (remote displacement) | vzdálená vazba / vzdálené zatížení |
| rigid / distributing (coupling) | tuhá / distribuční (vazba) |
| elastic support (elastic foundation) | pružné uložení |
| load / loads | zatížení |
| load case | zatěžovací stav |
| force, pressure, moment, gravity | síla, tlak, moment, tíha |
| bearing load | ložisková síla |
| hydrostatic pressure | hydrostatický tlak |
| rotation (centrifugal force) | otáčení (odstředivá síla) |
| coordinate system (Cartesian, cylindrical) | souřadný systém (kartézský, válcový) |
| mesh, meshing, element, node | síť, síťování, prvek, uzel |
| element size, refinement | velikost prvku, zjemnění |
| solver, analysis/computation, solution ("Řešení" step = "Solve") | řešič, výpočet, řešení |
| results | výsledky |
| displacement / displacements | posuv / posuvy |
| stress | napětí (mechanické) |
| von Mises stress | napětí von Mises |
| principal stress | hlavní napětí |
| strain | přetvoření |
| reaction | reakce |
| equilibrium | rovnováha |
| safety factor | bezpečnost (součinitel) |
| yield strength Re, tensile strength Rm | mez kluzu Re, pevnost Rm |
| Young's modulus E, Poisson's ratio ν | modul pružnosti E, Poissonovo číslo ν |
| density | hustota |
| fatigue, damage, life | únava, poškození, životnost |
| load history | zatěžovací křivka / průběh |
| critical plane | kritická rovina |
| S-N curve | Wöhlerova křivka (S-N) |
| endurance limit / fatigue limit | mez únavy |
| bolt, nut, washer | šroub, matice, podložka |
| preload | předpětí |
| clamped parts | sevřené díly |
| joint interface | spára |
| weld (fillet, butt), weld toe, weld root | svar (koutový, tupý), pata, kořen svaru |
| filler metal | přídavný materiál |
| resistance (capacity) | únosnost |
| utilization | využití |
| check / assessment | posouzení |
| bearing | otlačení |
| shear | střih |
| slip | prokluz |
| thread stripping | stržení závitu |
| fitted bolt | lícovaný šroub |
| 3D printing, printed part | 3D tisk, tištěný díl |
| build direction | směr stavby |
| layer, line (line width) | vrstva, linie (šířka linie) |
| perimeters (walls), top/bottom layers | perimetry (stěny), horní/dolní vrstvy |
| infill, infill pattern | výplň, vzor výplně |
| shell / core | skořepina / jádro (tisku) |
| creep | tečení |
| variant, sweep | varianta, rozmítání |
| heat, heat conduction (thermal) | teplo, vedení tepla |
| thermal case | tepelný stav |
| steady-state / transient | ustálený / přechodový |
| heat flux, convection, radiation, heat generation | tepelný tok, konvekce, sálání, zdroj tepla |
| thermal stress | teplotní napětí |
| natural frequency, mode shape | vlastní frekvence, tvar kmitu |
| (linear) buckling | vzpěr (lineární) |
| bar/member, beam, cross-section | prut, nosník, průřez |
| lateral-torsional buckling | klopení |
| hinge (release) | kloub |
| section (section view) | řez (pohled) |
| probe | sonda |
| report | protokol (Word) |
| manual, help, guide/tour, example | příručka, nápověda, průvodce, ukázka |
| project | projekt (.fea) |
| material library | knihovna materiálů |
| grade converter | převodník značení |
| singularity | singularita |
| readiness check | kontrola připravenosti |
| rigid-body motion (free) | tuhé pohyby (volné) |
| Undo / Redo | Zpět / Znovu |
| Appearance (light/dark/system) | Vzhled (světlý/tmavý/podle systému) |
| warning / error | varování / chyba |
