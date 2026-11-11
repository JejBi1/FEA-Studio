"""Material library (built in). Values for thicknesses / diameters up to 16 mm, the source is given.

Units: E, Re, Rm in MPa; ρ in t/mm³.
Printed materials (E_z ≠ None) = properties of the FULL extrusion (100 % infill) in the plane of the layer XY and perpendicular
to the layers Z; the shell and the infill are computed by fea_core.printing. Calibration from own ISO 527 tests:
fea_core.printing.calibrate. The isotropic "print XY (indicative)" stay because of older studies.
Metals for 3D printing (L-PBF) are isotropic materials – the printing model (perimeters, infill) does not apply to them.
"""
from __future__ import annotations

from .study import Material

# thermal properties: k [W/(m·K)], α [1/K], c_p [J/(kg·K)] at ~20 °C (fea_core.thermal)
_TH_EC3 = "; thermal EN 1993-1-2 (λ, c_p at 20 °C), α EN 1993-1-1"
_TH_PLASTIC = "; thermal from literature (approximate: plastics λ 0.13–0.2 W/(m·K), α 60–70e-6/K, c_p 1200–1800 J/(kg·K))"
# metals for 3D printing (15. 10. 2026): isotropic model, the values depend on the machine, the parameters and the heat treatment
_AM = ("; isotropic model – indicative, depends on machine, parameters, build direction and heat treatment (verify by testing)")

LIBRARY: list[Material] = [
    Material("Ocel S235", 210000, 0.30, 7.85e-9, Re=235, Rm=360, k=53.3, alpha=12e-6, cp=440, source="EN 10025-2" + _TH_EC3),
    Material("Ocel S355", 210000, 0.30, 7.85e-9, Re=355, Rm=470, k=53.3, alpha=12e-6, cp=440, source="EN 10025-2" + _TH_EC3),
    Material("Ocel C45 +N", 210000, 0.30, 7.85e-9, Re=340, Rm=620, k=46.0, alpha=11.5e-6, cp=460,
             source="EN ISO 683-1; thermal typical values for unalloyed 0.45 % C steel (approximate)"),
    Material("Ocel 42CrMo4 +QT", 210000, 0.30, 7.85e-9, Re=900, Rm=1100, k=42.6, alpha=11.1e-6, cp=460,
             source="EN ISO 683-2; thermal typical manufacturer values (λ 42.6 W/(m·K), α 20–100 °C) – approximate"),
    Material("Nerez 1.4301", 200000, 0.30, 7.90e-9, Re=210, Rm=520, k=15.0, alpha=16.0e-6, cp=500,
             source="EN 10088-3; teplo EN 10088-1 (20 °C, α 20–100 °C)"),
    Material("Nerez 1.4404", 200000, 0.30, 8.00e-9, Re=220, Rm=530, k=15.0, alpha=16.0e-6, cp=500,
             source="EN 10088-3; teplo EN 10088-1 (20 °C, α 20–100 °C)"),
    Material("Hliník EN AW-6082 T6", 70000, 0.33, 2.70e-9, Re=260, Rm=310, k=170.0, alpha=23.4e-6, cp=900,
             source="EN 755-2; thermal typical values (approximate, λ 170–180 W/(m·K))"),
    Material("Hliník EN AW-7075 T6", 71700, 0.33, 2.81e-9, Re=480, Rm=540, k=130.0, alpha=23.4e-6, cp=960,
             source="EN 754-2; thermal ASM Handbook (approximate)"),
    Material("PLA tisk XY (orientačně)", 3500, 0.36, 1.24e-9, Re=None, Rm=50, k=0.13, alpha=68e-6, cp=1800,
             source="indicative value – calibration required (ISO 527)" + _TH_PLASTIC),
    Material("PETG tisk XY (orientačně)", 2000, 0.38, 1.27e-9, Re=None, Rm=45, k=0.20, alpha=60e-6, cp=1200,
             source="indicative value – calibration required (ISO 527)" + _TH_PLASTIC),
    # ---- printed materials (orthotropic, F4)
    Material("Tisk PLA Bambu Basic", 2680, 0.36, 1.24e-9, Re=None, Rm=39, E_z=2160, Rm_z=35, Rc=60,
             k=0.13, alpha=68e-6, cp=1800,
             source="Bambu Lab TDS PLA Basic V2.0 (ISO 527, 100 % infill, annealed 55 °C/8 h): E 2680/2160 MPa, Rm 39/35 MPa (XY/Z); ν, compressive and shear strength estimated" + _TH_PLASTIC),
    Material("Tisk PLA obecně (Z 60 %)", 3000, 0.36, 1.24e-9, Re=None, Rm=45, E_z=2500, Rm_z=27, Rc=60,
             k=0.13, alpha=68e-6, cp=1800,
             source="indicative value from literature (Z strength typically 50–70 % of XY) – calibration required" + _TH_PLASTIC),
    Material("Tisk PETG (orientačně)", 2000, 0.38, 1.27e-9, Re=None, Rm=45, E_z=1700, Rm_z=30, Rc=55,
             k=0.20, alpha=60e-6, cp=1200,
             source="indicative value – calibration required (ISO 527)" + _TH_PLASTIC),

    # ---- structural, alloy and tool steels
    Material("Ocel S275", 210000, 0.30, 7.85e-9, Re=275, Rm=430, k=53.3, alpha=12e-6, cp=440,
             source="EN 10025-2" + _TH_EC3),
    Material("Ocel S690QL", 210000, 0.30, 7.85e-9, Re=690, Rm=800, k=40.0, alpha=11.5e-6, cp=460,
             source="EN 10025-6; high-strength quenched and tempered fine-grain steel (thickness up to 50 mm)"),
    Material("Ocel 16MnCr5", 210000, 0.30, 7.85e-9, Re=600, Rm=850, k=44.0, alpha=11.5e-6, cp=460,
             source="EN 10084; case-hardening steel (core properties after hardening and tempering)"),
    Material("Ocel 34CrNiMo6 +QT", 210000, 0.30, 7.85e-9, Re=800, Rm=1000, k=38.0, alpha=11.3e-6, cp=460,
             source="EN ISO 683-2; quenched and tempered (+QT, diameter up to 100 mm)"),
    Material("Ocel 51CrV4", 210000, 0.30, 7.85e-9, Re=1100, Rm=1350, k=40.0, alpha=11.5e-6, cp=460,
             source="EN 10089; spring steel, quenched and tempered (+QT)"),
    Material("Ocel S960QL", 210000, 0.30, 7.85e-9, Re=960, Rm=1050, k=38.0, alpha=11.5e-6, cp=460,
             source="EN 10025-6; high-strength quenched and tempered steel (thickness up to 50 mm)"),
    Material("Ocel Hardox 450", 210000, 0.30, 7.85e-9, Re=1200, Rm=1450, k=40.0, alpha=11.2e-6, cp=460,
             source="SSAB Hardox 450 data sheet; wear-resistant steel 450 HBW"),
    Material("Ocel 100Cr6", 210000, 0.30, 7.83e-9, Re=1700, Rm=2000, k=43.0, alpha=11.0e-6, cp=460,
             source="EN ISO 683-17 / ČSN 14 109; ložisková ocel po kalení a nízkém popuštění"),
    Material("Ocel 1.2379", 210000, 0.30, 7.70e-9, Re=1650, Rm=1950, k=20.0, alpha=10.5e-6, cp=460,
             source="EN ISO 4957 (X153CrMoV12 / AISI D2 / ČSN 19 573; nástrojová ocel pro stříhání a lisování)"),
    Material("Ocel 1.2343", 215000, 0.30, 7.80e-9, Re=1300, Rm=1520, k=26.0, alpha=11.5e-6, cp=460,
             source="EN ISO 4957 (X37CrMoV5-1 / AISI H11; hot-work tool steel for die casting)"),

    # ---- stainless steels and nickel alloys
    Material("Nerez 1.4542", 196000, 0.30, 7.80e-9, Re=1000, Rm=1070, k=18.0, alpha=10.8e-6, cp=460,
             source="EN 10088-3 (17-4 PH / stav +P1070 / H1025)"),
    Material("Nerez 1.4462", 200000, 0.30, 7.80e-9, Re=460, Rm=700, k=15.0, alpha=13.0e-6, cp=500,
             source="EN 10088-3; duplex austenitic-ferritic steel 2205"),
    Material("Inconel 718", 205000, 0.29, 8.19e-9, Re=1030, Rm=1240, k=11.4, alpha=13.0e-6, cp=435,
             source="AMS 5662 / ASTM B637 (Inconel 718 / W.Nr. 2.4668, solution annealed and aged)"),

    # ---- litiny
    Material("Šedá litina EN-GJL-250", 115000, 0.26, 7.20e-9, Re=None, Rm=250, Rc=800, k=46.0, alpha=11.0e-6, cp=480,
             source="EN 1561 (GG-25, flake graphite; wall thickness 10–20 mm)"),
    Material("Tvárná litina EN-GJS-500-7", 170000, 0.275, 7.10e-9, Re=320, Rm=500, k=36.0, alpha=12.0e-6, cp=500,
             source="EN 1563 (GGG-50, spheroidal graphite; cast test block specimen)"),
    Material("Nerez 1.4016", 220000, 0.30, 7.70e-9, Re=260, Rm=450, k=25.0, alpha=10.0e-6, cp=460,
             source="EN 10088-3 (AISI 430 / ČSN 17 040; feritická korozivzdorná ocel, magnetická)"),
    Material("Nerez 1.4057", 215000, 0.30, 7.70e-9, Re=600, Rm=800, k=25.0, alpha=10.5e-6, cp=460,
             source="EN 10088-3 (AISI 431 / ČSN 17 145; zušlechtitelná martenzitická korozivzdorná ocel, stav +QT800)"),
    Material("Šedá litina EN-GJL-150", 100000, 0.26, 7.15e-9, Re=None, Rm=150, Rc=600, k=50.0, alpha=11.0e-6, cp=480,
             source="EN 1561 (GG-15, flake graphite; soft iron, good machinability and damping)"),
    Material("Tvárná litina EN-GJS-400-15", 165000, 0.275, 7.10e-9, Re=250, Rm=400, k=38.0, alpha=12.0e-6, cp=500,
             source="EN 1563 (GGG-40, spheroidal graphite; ferritic ductile iron with high elongation A=15 %)"),

    # ---- aluminium, titanium and copper alloys
    Material("Hliník EN AW-6061 T6", 69000, 0.33, 2.70e-9, Re=240, Rm=290, k=167.0, alpha=23.5e-6, cp=900,
             source="EN 755-2 / ASTM B221; bars and profiles, temper T6"),
    Material("Hliník EN AW-2024 T3", 73000, 0.33, 2.78e-9, Re=290, Rm=440, k=121.0, alpha=23.2e-6, cp=875,
             source="EN 755-2 (AlCu4Mg1, dural, vytvrzeno za studena stav T3 / T351)"),
    Material("Hliník EN AC-43000", 75000, 0.33, 2.67e-9, Re=230, Rm=340, k=150.0, alpha=21.0e-6, cp=910,
             source="EN 1706 (AlSi10Mg(a), temper T6 for permanent mould / castings)"),
    Material("Titan Grade 5 (Ti-6Al-4V)", 114000, 0.34, 4.43e-9, Re=830, Rm=900, k=6.7, alpha=8.6e-6, cp=526,
             source="ASTM B348 / ISO 5832-3 (Ti-6Al-4V annealed)"),
    Material("Titan Grade 2", 105000, 0.34, 4.51e-9, Re=275, Rm=345, k=16.4, alpha=8.6e-6, cp=528,
             source="ASTM B348 (Grade 2, commercially pure titanium CP-Ti)"),
    Material("Měď Cu-ETP", 115000, 0.34, 8.94e-9, Re=200, Rm=260, k=390.0, alpha=16.8e-6, cp=385,
             source="EN 13601 (CW004A / E-Cu58, pure electrical copper, temper R250)"),
    Material("Mosaz CuZn40", 105000, 0.35, 8.40e-9, Re=250, Rm=440, k=120.0, alpha=20.0e-6, cp=380,
             source="EN 12164 (CW509L, two-phase brass, temper R440)"),
    Material("Mosaz CuZn39Pb3", 105000, 0.35, 8.44e-9, Re=250, Rm=420, k=113.0, alpha=20.0e-6, cp=380,
             source="EN 12164 (CW614N / ČSN 42 3213 / Ms58, automatová mosaz)"),
    Material("Bronz CuSn8", 115000, 0.35, 8.80e-9, Re=300, Rm=450, k=67.0, alpha=18.5e-6, cp=380,
             source="EN 12163 (CW453K / ČSN 42 3123; fosforový cínový bronz, stav R450 pro pružné kontakty a pouzdra)"),
    Material("Hliníkový bronz CuAl10Fe3Mn2", 110000, 0.33, 7.60e-9, Re=350, Rm=700, k=42.0, alpha=16.0e-6, cp=440,
             source="EN 12163 (CW306G / ČSN 42 3046 / Ampco; vysoce pevný kluzný bronz odolný proti kavitaci)"),
    Material("Hliník EN AW-5083 H111", 70000, 0.33, 2.66e-9, Re=125, Rm=275, k=120.0, alpha=23.7e-6, cp=900,
             source="EN 755-2 (AlMg4.5Mn0.7; weldable corrosion-resistant alloy for marine and chemical use)"),
    Material("Hliník EN AW-6060 T6", 69000, 0.33, 2.70e-9, Re=150, Rm=190, k=200.0, alpha=23.4e-6, cp=900,
             source="EN 755-2 (AlMgSi0.5; extruded profiles and precision frames)"),
    Material("Hořčík AZ91D", 45000, 0.35, 1.81e-9, Re=150, Rm=230, k=72.0, alpha=26.0e-6, cp=1020,
             source="ASTM B94 (AZ91D / EN-MC21120; structural magnesium die-casting alloy)"),

    # ---- printed materials FDM / FFF (orthotropic in the Z axis)
    Material("Tisk ASA Prusament", 1600, 0.38, 1.07e-9, Re=None, Rm=42, E_z=1700, Rm_z=11, Rc=55,
             k=0.18, alpha=80e-6, cp=1300,
             source="Prusament ASA TDS (ISO 527, 100 % infill): E 1600/1700 MPa, Rm 42/11 MPa (XY/Z)" + _TH_PLASTIC),
    Material("Tisk ABS Bambu", 2200, 0.38, 1.04e-9, Re=None, Rm=33, E_z=1960, Rm_z=28, Rc=50,
             k=0.18, alpha=85e-6, cp=1400,
             source="Bambu Lab TDS ABS (ISO 527, 100 % infill): E 2200/1960 MPa, Rm 33/28 MPa (XY/Z)" + _TH_PLASTIC),
    Material("Tisk PC Blend Prusament", 1900, 0.38, 1.19e-9, Re=None, Rm=63, E_z=2000, Rm_z=63, Rc=75,
             k=0.20, alpha=65e-6, cp=1200,
             source="Prusament PC Blend TDS (ISO 527, 100 % infill): E 1900/2000 MPa, Rm 63/63 MPa (XY/Z)" + _TH_PLASTIC),
    Material("Tisk PETG-CF Bambu", 2460, 0.38, 1.29e-9, Re=None, Rm=35, E_z=1340, Rm_z=29, Rc=60,
             k=0.24, alpha=45e-6, cp=1200,
             source="Bambu Lab TDS PETG-CF (ISO 527, 100 % infill): E 2460/1340 MPa, Rm 35/29 MPa (XY/Z)" + _TH_PLASTIC),
    Material("Tisk PAHT-CF Bambu", 3860, 0.39, 1.21e-9, Re=None, Rm=92, E_z=2180, Rm_z=47, Rc=85,
             k=0.30, alpha=35e-6, cp=1500,
             source="Bambu Lab TDS PAHT-CF (ISO 527, dry state, 100 % infill): E 3860/2180 MPa, Rm 92/47 MPa (XY/Z)" + _TH_PLASTIC),
    Material("Tisk TPU 95A Bambu", 80, 0.45, 1.20e-9, Re=None, Rm=30, E_z=65, Rm_z=23, Rc=25,
             k=0.16, alpha=150e-6, cp=1800,
             source="Bambu Lab TDS TPU 95A (ISO 527, initial modulus): E ~80/65 MPa, Rm 30/23 MPa (XY/Z)" + _TH_PLASTIC),
    Material("Tisk PEEK", 3800, 0.38, 1.30e-9, Re=None, Rm=95, E_z=2700, Rm_z=45, Rc=110,
             k=0.25, alpha=50e-6, cp=1350,
             source="ThermaX PEEK TDS (ISO 527, annealed 200 °C/4 h, 100 % infill): E 3800/2700 MPa, Rm 95/45 MPa (XY/Z)" + _TH_PLASTIC),
    Material("Tisk PEI Ultem 9085", 2400, 0.38, 1.27e-9, Re=None, Rm=68, E_z=2000, Rm_z=40, Rc=85,
             k=0.22, alpha=55e-6, cp=1250,
             source="Stratasys Ultem 9085 TDS (ASTM D638, aviation certificate FAR 25.853): E 2400/2000 MPa, Rm 68/40 MPa (XY/Z)" + _TH_PLASTIC),
    Material("Tisk PLA-CF Bambu", 4200, 0.36, 1.26e-9, Re=None, Rm=52, E_z=2300, Rm_z=28, Rc=70,
             k=0.24, alpha=40e-6, cp=1600,
             source="Bambu Lab TDS PLA-CF (ISO 527, 100 % infill): E 4200/2300 MPa, Rm 52/28 MPa (XY/Z)" + _TH_PLASTIC),
    Material("Tisk SLS PA12", 1650, 0.38, 0.95e-9, Re=None, Rm=48, E_z=1550, Rm_z=42, Rc=60,
             k=0.25, alpha=110e-6, cp=1700,
             source="EOS PA 2200 MDS (ISO 527, selective laser sintering of powder SLS): E 1650/1550 MPa, Rm 48/42 MPa (XY/Z)" + _TH_PLASTIC),

    # ---- metals for 3D printing (L-PBF / DED)
    # Isotropic Material (NOT a printed material with E_z – the model of fea_core.printing is for FDM plastics with perimeters and infill).
    # Re, Rm = the lowest set of the listed sources (usually the build direction Z), E = typical value from the scatter.
    Material("Ti-6Al-4V L-PBF bez TZ", 111000, 0.34, 4.42e-9, Re=1100, Rm=1250, k=7.1, alpha=9.2e-6, cp=526,
             source="AddUp MDS Ti6Al4V ELI Grade 23 (FormUp 350, layer 60 µm, ASTM E8, US-04052023-0923): as-built Re 1104/1126 MPa, Rm 1265/1252 MPa, A 11/10 %, E 111/113 GPa (XY/Z); α′ martensite and residual stresses – anneal load-bearing parts; heat k 7.1 W/(m·K), α 9.2e-6/K (same sheet), c_p wrought state"
                    + _AM),
    Material("Ti-6Al-4V L-PBF žíhaný", 115000, 0.34, 4.42e-9, Re=900, Rm=1010, k=7.1, alpha=9.2e-6, cp=526,
             source="Renishaw RenAM 500 MDS Ti6Al4V Grade 23 (H-5800-6797-03-A, ASTM E8, annealed 800 °C/4 h in vacuum, layers 30–90 µm): Re 908–994 MPa, Rm 1018–1065 MPa, A 16–22 %, E 115–124 GPa – the lowest set used (Z, 30 µm), rounded down; Gao et al. 2025, Materials 18(15):3475, doi:10.3390/ma18153475 (800 °C/2 h): Re 955, Rm 1030 MPa; heat AddUp MDS Ti6Al4V ELI" + _AM),
    Material("Ti-6Al-4V L-PBF HIP", 115000, 0.34, 4.42e-9, Re=810, Rm=910, k=7.1, alpha=9.2e-6, cp=526,
             source="Gao et al. 2025, Materials 18(15):3475, doi:10.3390/ma18153475 (HIP 920 °C/150 MPa/3 h): Re 850, Rm 935 MPa, A 16 %; AddUp MDS Ti6Al4V ELI (annealed 940 °C/4 h): Re 812–816 MPa, Rm 911 MPa, E 117–118 GPa (XY = Z) – the lower value used; ASTM F2924 requires min. Rm 895, Rp0.2 825 MPa; nearly isotropic, best fatigue (closed pores)" + _AM),
    Material("AlSi10Mg L-PBF bez TZ", 70000, 0.33, 2.67e-9, Re=230, Rm=395, k=120.0, alpha=20.5e-6, cp=910,
             source="Renishaw RenAM 500 MDS AlSi10Mg (H-5800-6790-03-A, ASTM E8, as-built, layers 30–60 µm): Re 244–283 / 228–264 MPa, Rm 392–424 / 394–442 MPa, A 8–10 / 4–8 %, E 69–73 GPa (XY/Z) – min. Z used; elongation in Z significantly lower; heat: λ as-built indicative (lower than annealed, wrought state 130–190 W/(m·K)), α 20–21e-6/K (same sheet)" + _AM),
    Material("AlSi10Mg L-PBF odpnutý", 70000, 0.33, 2.67e-9, Re=155, Rm=260, k=150.0, alpha=20.5e-6, cp=910,
             source="Renishaw RenAM 500 MDS AlSi10Mg (H-5800-6790-03-A, stress-relief annealing 275 °C/2 h in argon, ASTM E8): Re 157–212 MPa, Rm 262–316 MPa, A 12–19 %, E 67–75 GPa – the lowest set used (Z, 60 µm); λ indicative; other annealing (T6) changes the strength – verify" + _AM),
    Material("316L L-PBF bez TZ", 190000, 0.30, 7.99e-9, Re=495, Rm=595, k=16.0, alpha=16e-6, cp=500,
             source="Renishaw RenAM 500 MDS 316L (1.4404) (H-5800-6796-03-A, ASTM E8, as-built, layers 30–70 µm): Re 555–591 / 495–532 MPa, Rm 594–708 / 631–669 MPa (XY/Z), A 40–60 %, E 177–208 GPa – minimum used; heat wrought state (k 16 W/(m·K), α 16e-6/K, same sheet)" + _AM),
    Material("316L L-PBF žíhaný", 185000, 0.30, 7.99e-9, Re=395, Rm=625, k=16.0, alpha=16e-6, cp=500,
             source="Renishaw RenAM 500 MDS 316L (1.4404) (H-5800-6796-03-A, solution annealing 900 °C/2 h, ASTM E8): Re 418–421 / 395–396 MPa, Rm 667–674 / 629–636 MPa (XY/Z), A 53–67 %, E 170–196 GPa – min. Z used" + _AM),
    Material("17-4PH L-PBF vytvrzený H900", 165000, 0.30, 7.80e-9, Re=890, Rm=1115, k=18.0, alpha=10.8e-6, cp=460,
             source="Renishaw RenAM 500 MDS 17-4PH (1.4542) (H-5800-6832-02-A, ASTM E8, solution annealing 1040 °C/30 min + ageing 480 °C/1 h ≈ H900): Re 893/1046 MPa, Rm 1119/1299 MPa, A 17 %, E 164/180 GPa (XY/Z) – the XY plane is weaker, minimum used; wrought steel ASTM A564 H900 has min. Rm 1310 MPa; heat EN 10088-1 (k 18 W/(m·K), α 10.8e-6/K)" + _AM),
    Material("Inconel 718 L-PBF bez TZ", 170000, 0.29, 8.19e-9, Re=585, Rm=950, k=11.4, alpha=13.0e-6, cp=435,
             source="Renishaw RenAM 500 MDS Inconel 718 (H-5800-6794-03-A, ASTM E8, as-built, layer 60 µm): Re 731–735 / 588–632 MPa, Rm 1048–1051 / 951–1002 MPa (XY/Z), A 33–40 %, E 171–191 GPa – min. Z used; heat wrought state" + _AM),
    Material("Inconel 718 L-PBF vytvrzený", 190000, 0.29, 8.19e-9, Re=1160, Rm=1385, k=11.4, alpha=13.0e-6, cp=435,
             source="Renishaw RenAM 500 MDS Inconel 718 (H-5800-6794-03-A, ASTM E8, solution annealing 980 °C/1 h + ageing 720 °C/8 h and 620 °C/8 h): Re 1275–1350 / 1163–1264 MPa, Rm 1480–1539 / 1389–1461 MPa (XY/Z), A 16–27 %, E 177–213 GPa – min. Z used; heat wrought state" + _AM),
    Material("Inconel 625 L-PBF bez TZ", 170000, 0.30, 8.44e-9, Re=585, Rm=860, k=10.0, alpha=13.0e-6, cp=410,
             source="Renishaw RenAM 500 MDS Inconel 625 (H-5800-6792-03-A, ASTM E8, as-built, layers 30–60 µm): Re 718–724 / 585–639 MPa, Rm 1040–1042 / 864–976 MPa (XY/Z), A 37–59 %, E 139–218 GPa (large scatter) – min. Z used; heat wrought state (k 9–11 W/(m·K), α 13e-6/K, same sheet), c_p indicative" + _AM),
    Material("Inconel 625 L-PBF žíhaný", 180000, 0.30, 8.44e-9, Re=505, Rm=825, k=10.0, alpha=13.0e-6, cp=410,
             source="Renishaw RenAM 500 MDS Inconel 625 (H-5800-6792-03-A, annealed 1048 °C/1 h, ASTM E8): Re 584–592 / 507–539 MPa, Rm 971–981 / 827–900 MPa (XY/Z), A 49–69 %, E 147–194 GPa – min. Z used" + _AM),
    Material("Maraging 1.2709 L-PBF bez TZ", 160000, 0.30, 8.10e-9, Re=820, Rm=975, k=14.0, alpha=10.0e-6, cp=450,
             source="Renishaw RenAM 500 MDS Maraging M300 (1.2709) (H-5800-6795-03-A, ASTM E8, as-built): Re 972 / 822–902 MPa, Rm 1146 / 979–1056 MPa, A 13 / 14–16 %, E 191 / 140–169 GPa (XY/Z) – min. Z used; heat wrought state (same sheet), c_p indicative" + _AM),
    Material("Maraging 1.2709 L-PBF vytvrzený", 175000, 0.30, 8.10e-9, Re=1850, Rm=1915, k=14.0, alpha=10.0e-6, cp=450,
             source="Renishaw RenAM 500 MDS Maraging M300 (1.2709) (H-5800-6795-03-A, aged 490 °C/6 h, ASTM E8): Re 1969 / 1854–1911 MPa, Rm 2029 / 1916–1974 MPa, A 5–8 %, E 168–192 GPa (XY/Z) – min. Z used; low ductility, notch-sensitive" + _AM),
    Material("CoCrMo L-PBF bez TZ", 220000, 0.30, 8.40e-9, Re=685, Rm=1190, k=13.0, alpha=13.0e-6, cp=450,
             source="Renishaw RenAM 500 MDS Cobalt chrome Co28Cr6Mo (H-5800-6829-02-A, ASTM E8, as-built, layer 60 µm, Z direction only): Re 689 MPa, Rm 1193 MPa, A 25 %, E 224 GPa; heat wrought state (same sheet), c_p indicative" + _AM),
    Material("CoCrMo L-PBF žíhaný", 230000, 0.30, 8.40e-9, Re=560, Rm=1065, k=13.0, alpha=13.0e-6, cp=450,
             source="Renishaw RenAM 500 MDS Cobalt chrome Co28Cr6Mo (H-5800-6829-02-A, ASTM E8): solution annealing 1050 °C/2 h Re 573/566 MPa, Rm 1092/1068 MPa, E 236–238 GPa (XY/Z); annealing 1150 °C/6 h Re 560, Rm 1104 MPa (Z) – minimum used" + _AM),
    Material("Ocel 1.2344 (H13) L-PBF kalená", 198000, 0.30, 7.80e-9, Re=1485, Rm=1760, k=28.6, alpha=10.4e-6,
             cp=460,
             source="Renishaw RenAM 500 MDS H13 (1.2344) (H-5800-6830-01-A, ASTM E8, hardened 1020 °C/30 min in air + tempered 560 °C/2 h): Re 1506/1489 MPa, Rm 1773/1764 MPa, A 14.5/12.5 %, E 201/198 GPa (XY/Z); heat wrought state (same sheet), c_p indicative" + _AM),
    Material("Scalmalloy L-PBF", 70000, 0.33, 2.67e-9, Re=470, Rm=500, k=95.0, alpha=23.5e-6, cp=900,
             source="3D Systems Certified Scalmalloy (A) datasheet (PN 10123B 02-22, DMP Flex/Factory 350, annealed 325 °C/4 h per APWORKS, ASTM E8, mean and 95 % tolerance interval): Re 490–500 / 490 MPa, Rm 520–530 / 520 MPa (XY/Z), A 13–16 %, k 95–100 W/(m·K), α 23.5e-6/K; Re, Rm lowered to the minima of the powder catalogue (> 470 / > 500 MPa), E ≈ 70 GPa (APWORKS), c_p indicative as for Al alloys" + _AM),
    Material("CuCrZr L-PBF vytvrzený", 120000, 0.33, 8.90e-9, Re=470, Rm=495, k=255.0, alpha=17.0e-6, cp=390,
             source="Xie et al. 2023, J. Mater. Res. Technol. 23:3322–3336 (L-PBF with green laser, after heat treatment): Re 505/472 MPa, Rm 585/495 MPa, A 14/18 %, k 307/255 W/(m·K) (XY/Z) – Z direction used; strength and conductivity depend strongly on ageing (450–550 °C); E, α, c_p wrought alloy CuCr1Zr (CW106C) indicative" + _AM),

    # ---- orthotropic composites (continuous fibres and wood)
    Material("Kompozit CFRP UD", 140000, 0.30, 1.55e-9, Rm=1500, Rc=1200, k=5.0, alpha=0.0, cp=1100,
             ortho={
                 "E1": 140000, "E2": 10000, "E3": 10000,
                 "nu12": 0.30, "nu13": 0.30, "nu23": 0.40,
                 "G12": 5000, "G13": 5000, "G23": 3500,
                 "alpha1": -0.5e-6, "alpha2": 30.0e-6, "alpha3": 30.0e-6,
             },
             source="MIL-HDBK-17-2F / Tsai-Hahn (standard carbon/epoxy UD laminate AS4/3501-6, V_f ≈ 60 %); axis 1 = along the fibers"),
    Material("Kompozit CFRP Tkanina 2D", 65000, 0.05, 1.50e-9, Rm=600, Rc=550, k=3.0, alpha=2.5e-6, cp=1100,
             ortho={
                 "E1": 65000, "E2": 65000, "E3": 10000,
                 "nu12": 0.05, "nu13": 0.30, "nu23": 0.30,
                 "G12": 5000, "G13": 3800, "G23": 3800,
                 "alpha1": 2.5e-6, "alpha2": 2.5e-6, "alpha3": 35.0e-6,
             },
             source="Daniel & Ishai: Engineering Mechanics of Composite Materials (uhlík/epoxid tkanina 2D, vyvážená v rovině 1–2)"),
    Material("Kompozit GFRP UD", 43000, 0.28, 1.95e-9, Rm=1000, Rc=700, k=0.40, alpha=7.0e-6, cp=1050,
             ortho={
                 "E1": 43000, "E2": 11000, "E3": 11000,
                 "nu12": 0.28, "nu13": 0.28, "nu23": 0.42,
                 "G12": 4500, "G13": 4500, "G23": 3500,
                 "alpha1": 7.0e-6, "alpha2": 26.0e-6, "alpha3": 26.0e-6,
             },
             source="MIL-HDBK-17-2F (E-glass/epoxy UD laminate, V_f ≈ 55 %); axis 1 = along the fibers"),
    Material("Kompozit GFRP Tkanina 2D", 24000, 0.13, 1.85e-9, Rm=380, Rc=320, k=0.35, alpha=12.0e-6, cp=1050,
             ortho={
                 "E1": 24000, "E2": 24000, "E3": 10000,
                 "nu12": 0.13, "nu13": 0.30, "nu23": 0.30,
                 "G12": 4200, "G13": 3500, "G23": 3500,
                 "alpha1": 12.0e-6, "alpha2": 12.0e-6, "alpha3": 28.0e-6,
             },
             source="Barbero: FEA of Composite Materials (balanced 2D glass fabric in the 1–2 plane)"),
    Material("Dřevo smrk", 11000, 0.38, 0.45e-9, Rm=80, Rc=40, k=0.13, alpha=4.0e-6, cp=1600,
             ortho={
                 "E1": 11000, "E2": 750, "E3": 450,
                 "nu12": 0.38, "nu13": 0.45, "nu23": 0.55,
                 "G12": 700, "G13": 600, "G23": 50,
                 "alpha1": 4.0e-6, "alpha2": 30.0e-6, "alpha3": 50.0e-6,
             },
             source="DIN 68364 / Wood Handbook (smrk, vlhkost 12 %); osy 1=L (vlákna), 2=R (radiální), 3=T (tečná)"),
    Material("Kompozit AFRP UD (Kevlar 49)", 76000, 0.34, 1.38e-9, Rm=1400, Rc=280, k=0.30, alpha=-4.0e-6, cp=1400,
             ortho={
                 "E1": 76000, "E2": 5500, "E3": 5500,
                 "nu12": 0.34, "nu13": 0.34, "nu23": 0.40,
                 "G12": 2300, "G13": 2300, "G23": 1800,
                 "alpha1": -4.0e-6, "alpha2": 60.0e-6, "alpha3": 60.0e-6,
             },
             source="MIL-HDBK-17-2F (aramid/epoxy Kevlar 49 UD, Vf ≈ 60 %); high toughness and impact resistance"),
    Material("Kompozit CFRP HM (High Modulus)", 210000, 0.30, 1.60e-9, Rm=1800, Rc=1000, k=8.0, alpha=-1.0e-6, cp=1050,
             ortho={
                 "E1": 210000, "E2": 7000, "E3": 7000,
                 "nu12": 0.30, "nu13": 0.30, "nu23": 0.40,
                 "G12": 4500, "G13": 4500, "G23": 3000,
                 "alpha1": -1.0e-6, "alpha2": 30.0e-6, "alpha3": 30.0e-6,
             },
             source="Toray M40J / epoxy UD laminate (high-modulus carbon for satellite and aerospace structures)"),
    Material("Dřevo buk", 14000, 0.45, 0.68e-9, Rm=120, Rc=60, k=0.17, alpha=4.0e-6, cp=1600,
             ortho={
                 "E1": 14000, "E2": 1600, "E3": 900,
                 "nu12": 0.45, "nu13": 0.50, "nu23": 0.70,
                 "G12": 1200, "G13": 850, "G23": 250,
                 "alpha1": 4.0e-6, "alpha2": 35.0e-6, "alpha3": 60.0e-6,
             },
             source="DIN 68364 / Wood Handbook (buk lesní, vlhkost 12 %); osy 1=L (podél vláken), 2=R, 3=T"),
]


def user_library_path():
    from .config import user_config_dir
    return user_config_dir() / "materials.json"


def material_from_dict(d: dict) -> Material:
    """Material from a dictionary; unknown keys (newer versions) are ignored."""
    from dataclasses import fields
    names = {f.name for f in fields(Material)}
    return Material(**{k: v for k, v in d.items() if k in names})


class UserLibraryError(RuntimeError):
    pass


def _raw_user_library(strict: bool) -> list:
    import json
    p = user_library_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        if strict:
            raise UserLibraryError(f"The user material library {p} cannot be read ({e}) – fix or rename the file, otherwise saving would overwrite it") from None
        return []
    return data if isinstance(data, list) else []


def _loads(d) -> Material | None:
    try:
        return material_from_dict(d) if isinstance(d, dict) else None
    except (TypeError, ValueError):
        return None


def user_library() -> list[Material]:
    """User materials (%APPDATA%/FEAStudio/materials.json); bad records are skipped."""
    return [m for m in map(_loads, _raw_user_library(strict=False)) if m is not None]


def save_user_material(m: Material) -> None:
    """Adds / replaces a material. An unreadable file is not overwritten (UserLibraryError); records that this version
    cannot load stay in the file unchanged.
    """
    import json
    from dataclasses import asdict
    p = user_library_path()
    raw = [d for d in _raw_user_library(strict=True) if not (isinstance(d, dict) and d.get("name") == m.name)]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw + [asdict(m)], indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def all_materials() -> list[Material]:
    names = {m.name for m in LIBRARY}
    return LIBRARY + [m for m in user_library() if m.name not in names]


def by_name(name: str) -> Material:
    for m in all_materials():
        if m.name == name:
            return material_from_dict(m.__dict__)
    raise KeyError(name)


def default() -> Material:
    return by_name("Ocel S235")
