"""Designation converter for materials: ČSN ↔ EN ↔ W.Nr. ↔ DIN ↔ AISI/SAE ↔ UNS ↔ JIS ↔ GOST … and similar materials.

A group (`Grade`) = one material under the designations of different standards. A designation with "~" is only an approximate
equivalent (the composition range / strength differ), "=" is searched only as the whole text (e.g. GB "45").
`hidden` = further spellings, only for searching (e.g. misspellings in the PragTic database).

Sources: EN 10027-2, EN 10025-2, EN 10083, EN 10084, EN 10088, EN 573-3, EN 1561/1563/1562,
steel comparison table ČSN–DIN–EN–GOST (cadforum.cz/cz/oceli.asp), UNS / SAE J403
and J404 catalogues, STN 42 4203 (AlCu4Mg1) and 42 4222 (AlZn6Mg2Cu), ČSN 42 2306–2308 (ductile iron).
Uncertain pairs (grey cast iron ČSN 42 24xx, aluminium 42 44xx, cast steels 42 26xx) are deliberately missing.
An equivalence is always approximate (other standards = other tolerances) – check the standard before ordering.

Similarity: the same family (structural, quenched and tempered, austenitic stainless…) and the distance of the
nominal composition + strength; a different base metal (steel × aluminium) is never offered.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

SYSTEMS = [("EN", "EN"), ("WNR", "W.Nr."), ("CSN", "ČSN"), ("DIN", "DIN (dříve)"), ("AISI", "AISI/SAE"),
           ("UNS", "UNS"), ("ASTM", "ASTM"), ("JIS", "JIS"), ("KS", "KS"), ("GB", "GB (Čína)"),
           ("GOST", "GOST"), ("PN", "PN (Polsko)"), ("AFNOR", "AFNOR"), ("BS", "BS"), ("ALCOA", "Alcoa (dříve)"),
           ("TRADE", "obchodní")]
SYSTEM_LABELS = dict(SYSTEMS)

# family -> (base metal, description)
FAMILIES = {
    "ocel-konstr": ("ocel", "konstrukční ocel"),
    "ocel-mikro": ("ocel", "jemnozrnná / mikrolegovaná ocel"),
    "ocel-plech": ("ocel", "ocel pro tváření za studena (plech)"),
    "ocel-tlak": ("ocel", "ocel pro tlakové nádoby / žárupevná"),
    "ocel-c": ("ocel", "uhlíková ocel k zušlechťování"),
    "ocel-cem": ("ocel", "ocel k cementování"),
    "ocel-leg": ("ocel", "legovaná ocel k zušlechťování"),
    "ocel-pruz": ("ocel", "pružinová ocel"),
    "ocel-loz": ("ocel", "ložisková ocel"),
    "ocel-auto": ("ocel", "automatová ocel"),
    "ocel-nastr": ("ocel", "nástrojová ocel"),
    "nerez-aust": ("ocel", "austenitická korozivzdorná ocel"),
    "nerez-ferit": ("ocel", "feritická korozivzdorná ocel"),
    "nerez-mart": ("ocel", "martenzitická korozivzdorná ocel"),
    "nerez-ph": ("ocel", "precipitačně vytvrditelná korozivzdorná ocel"),
    "nerez-duplex": ("ocel", "duplexní korozivzdorná ocel"),
    "litina-seda": ("litina", "šedá litina (lupínkový grafit)"),
    "litina-tvarna": ("litina", "tvárná litina (kuličkový grafit)"),
    "litina-temper": ("litina", "temperovaná litina"),
    "litina-vermik": ("litina", "litina s vermikulárním grafitem"),
    "al-1": ("al", "hliník 1xxx (čistý)"),
    "al-2": ("al", "hliník 2xxx (Al-Cu, dural)"),
    "al-5": ("al", "hliník 5xxx (Al-Mg)"),
    "al-6": ("al", "hliník 6xxx (Al-Mg-Si)"),
    "al-7": ("al", "hliník 7xxx (Al-Zn-Mg)"),
    "al-lit": ("al", "slévárenská slitina hliníku"),
    "ti": ("ti", "slitina titanu"),
    "ni": ("ni", "niklová superslitina"),
    "co": ("co", "slitina kobaltu"),
    "cu-zn": ("cu", "mosaz"),
    "cu": ("cu", "měď"),
    "mg": ("mg", "slitina hořčíku"),
}


@dataclass
class Grade:
    key: str
    family: str
    names: dict[str, list[str]]          # system -> designations (with possibly "~")
    chem: dict[str, float]               # nominal composition [%]
    rm: tuple[float, float] | None       # typical Rm [MPa]
    note: str = ""
    hidden: list[str] = field(default_factory=list)

    @property
    def base(self) -> str:
        return FAMILIES[self.family][0]

    @property
    def family_label(self) -> str:
        return FAMILIES[self.family][1]

    @property
    def rm_mid(self) -> float | None:
        return 0.5 * (self.rm[0] + self.rm[1]) if self.rm else None

    def summary(self, html: bool = False) -> str:
        parts = []
        for sysk, label in SYSTEMS:
            vals = [("≈ " if n.startswith("~") else "") + n.lstrip("~=") for n in self.names.get(sysk, [])]
            if vals:
                v = ", ".join(vals)
                parts.append(f"<i>{label}</i> {v}" if html else f"{label} {v}")
        return " · ".join(parts)


def _chem(s: str) -> dict[str, float]:
    return {e: float(v) for e, v in re.findall(r"([A-Z][a-z]?)(\d*\.?\d+)", s)}


def G(key, family, chem="", rm=None, note="", hidden="", **names) -> Grade:
    nm = {k: [x.strip() for x in v.split(",") if x.strip()] for k, v in names.items()}
    return Grade(key, family, nm, _chem(chem), rm, note, [x.strip() for x in hidden.split(",") if x.strip()])


GRADES: list[Grade] = [
    # ------------------------------------------------------------ structural steels (EN 10025-2)
    G("S235", "ocel-konstr", "C.14 Mn.7 Si.2", (360, 510),
      EN="S235JR, S235J0, S235J2, S235JRG2, S235JRG1, Fe360B", WNR="1.0038, 1.0036, 1.0114, 1.0116, 1.0117",
      CSN="11375, 11373, 11378", DIN="St37-2, RSt37-2, USt37-2, St37-3, St37", ASTM="~A36", JIS="~SS400",
      GOST="~St3sp, ~St3ps"),
    G("St42-2", "ocel-konstr", "C.2 Mn.6", (410, 540), note="older grade between S235 and S275",
      WNR="1.0040, 1.0042", CSN="11423, 11425", DIN="St42-2, USt42-2, RSt42-2, St42", GOST="~St4sp"),
    G("S275", "ocel-konstr", "C.18 Mn1.0 Si.2", (410, 560),
      EN="S275JR, S275J0, S275J2, S275J2G3, Fe430B", WNR="1.0044, 1.0143, 1.0145", CSN="11443, 11448",
      DIN="St44-2, St44-3, St44"),
    G("S355", "ocel-konstr", "C.18 Mn1.4 Si.3", (470, 630),
      EN="S355JR, S355J0, S355J2, S355J2G3, S355K2, Fe510", WNR="1.0045, 1.0553, 1.0577, 1.0570, 1.0596",
      CSN="11523, 11503", DIN="St52-3, St52", ASTM="~A572 Gr.50", JIS="~SM490", GOST="~09G2S",
      PN="~18G2A", AFNOR="~E36-3, ~E36"),
    G("E295", "ocel-konstr", "C.3 Mn.6", (470, 610), EN="E295, Fe490", WNR="1.0050", CSN="11500",
      DIN="St50-2, St50"),
    G("E335", "ocel-konstr", "C.4 Mn.7", (570, 710), EN="E335, Fe590", WNR="1.0060", CSN="11600",
      DIN="St60-2, St60"),
    G("E360", "ocel-konstr", "C.5 Mn.7", (670, 830), EN="E360, Fe690", WNR="1.0070", CSN="11700",
      DIN="St70-2, St70"),
    G("S355J2WP", "ocel-konstr", "C.1 Mn.4 Si.4 Cr.7 Cu.35 Ni.4 P.1", (490, 630),
      note="weathering (atmospheric corrosion-resistant) steel", EN="S355J2WP, S355J2W", WNR="1.8946, 1.8965",
      ASTM="~A242", TRADE="~Cor-Ten A", PN="~10HNAP"),
    G("E235", "ocel-tlak", "C.12 Mn.5", (340, 480), note="precision / seamless tubes",
      EN="E235, P235TR1", WNR="1.0308", DIN="St35"),
    G("P295GH", "ocel-tlak", "C.16 Mn1.1 Si.3", (460, 580), EN="P295GH", WNR="1.0481", CSN="13030",
      DIN="17Mn4"),
    G("16Mo3", "ocel-tlak", "C.16 Mn.6 Mo.3", (440, 590), EN="16Mo3", WNR="1.5415", CSN="15020",
      DIN="15Mo3", ASTM="~A204 Gr.A"),
    G("13CrMo4-5", "ocel-tlak", "C.13 Mn.6 Cr1.0 Mo.5", (440, 590), EN="13CrMo4-5", WNR="1.7335",
      DIN="13CrMo44", PN="15HM", ASTM="~A387 Gr.12"),
    G("10CrMo9-10", "ocel-tlak", "C.1 Mn.5 Cr2.25 Mo1.0", (480, 630), EN="10CrMo9-10, 11CrMo9-10",
      WNR="1.7380, 1.7383", CSN="15313", DIN="10CrMo910", ASTM="A387 Gr.22, ~A182 F22, ~A335 P22",
      hidden="2.25Cr-1Mo"),
    G("14MoV6-3", "ocel-tlak", "C.14 Mn.55 Cr.45 Mo.55 V.27", (460, 610), EN="14MoV6-3", WNR="1.7715",
      CSN="15128", DIN="14MoV63", GOST="~12Ch1MF"),
    G("X11CrMo9-1", "ocel-tlak", "C.1 Cr9.0 Mo1.0", (590, 740), EN="X11CrMo9-1", WNR="1.7386",
      ASTM="A335 P9, A213 T9", hidden="9Cr1Mo"),
    G("S460N", "ocel-mikro", "C.18 Mn1.5 V.15 Ni.5", (540, 720), EN="S460N, S460NL", WNR="1.8901, 1.8903",
      DIN="StE460, TStE460"),
    G("S690Q", "ocel-mikro", "C.16 Mn1.0 Cr.5 Mo.4 Ni.5", (770, 940), EN="S690Q, S690QL", WNR="1.8931, 1.8928",
      DIN="StE690", ASTM="~A514", TRADE="~RQC-100"),
    G("S960Q", "ocel-mikro", "C.20 Mn1.6 Cr.8 Mo.7 Ni2.0", (980, 1150), EN="S960Q, S960QL", WNR="1.8941, 1.8933",
      DIN="StE960", TRADE="~Weldox 960"),
    G("Hardox 450", "ocel-mikro", "C.26 Mn1.6 Cr1.4 Mo.6 Ni1.5", (1400, 1600), note="wear-resistant plate 450 HBW",
      TRADE="Hardox 450, Raex 450, XAR 450", hidden="Hardox"),
    # ------------------------------------------------------------ sheets for cold forming
    G("DC01", "ocel-plech", "C.06 Mn.3", (270, 410), EN="DC01, FeP01", WNR="1.0330", CSN="11321",
      DIN="St12", AISI="~1008", JIS="SPCC", hidden="CQ"),
    G("DC03", "ocel-plech", "C.06 Mn.3", (270, 370), EN="DC03, FeP03", WNR="1.0347", CSN="11301",
      DIN="St13, RRSt13", JIS="~SPCD", hidden="DQSK"),
    G("DC04", "ocel-plech", "C.05 Mn.3", (270, 350), EN="DC04, FeP04", WNR="1.0338", CSN="11305",
      DIN="St14", JIS="~SPCE", hidden="DDQ"),
    # ------------------------------------------------------------ carbon steels for quenching and tempering / case hardening
    G("C10", "ocel-cem", "C.1 Mn.45", (340, 490), EN="C10, C10E, C10R", WNR="1.0301, 1.1121, 1.1207",
      CSN="12010", DIN="Ck10", AISI="1010", UNS="G10100", JIS="S10C"),
    G("C15", "ocel-cem", "C.15 Mn.45", (370, 520), EN="C15, C15E, C15R", WNR="1.0401, 1.1141, 1.1140",
      CSN="12020, 12023", DIN="Ck15", AISI="1015", UNS="G10150", JIS="S15C"),
    G("C22", "ocel-c", "C.2 Mn.55", (410, 560), EN="C22, C22E, C22R", WNR="1.0402, 1.1151, 1.1149",
      CSN="12024", DIN="Ck22", AISI="1020, 1023", UNS="G10200", JIS="S20C, S22C", AFNOR="~XC18",
      BS="~En3, ~En3B"),
    G("C25", "ocel-c", "C.25 Mn.55", (440, 600), EN="C25, C25E, C25R", WNR="1.0406, 1.1158, 1.1163",
      CSN="12030", DIN="Ck25", AISI="1025", UNS="G10250", JIS="S25C"),
    G("C30", "ocel-c", "C.3 Mn.65", (500, 650), EN="C30, C30E, C30R", WNR="1.0528, 1.1178, 1.1179",
      CSN="12031", DIN="Ck30", AISI="1030", UNS="G10300", JIS="S30C"),
    G("C35", "ocel-c", "C.35 Mn.65", (550, 700), EN="C35, C35E, C35R", WNR="1.0501, 1.1181, 1.1180",
      CSN="12040", DIN="Ck35, Cm35, Cf35", AISI="1035", UNS="G10350", JIS="S35C", AFNOR="~XC38",
      hidden="C35N"),
    G("C40", "ocel-c", "C.4 Mn.65", (580, 730), EN="C40, C40E, C40R", WNR="1.0511, 1.1186, 1.1189",
      CSN="12041", DIN="Ck40", AISI="1040", UNS="G10400", JIS="S40C"),
    G("C45", "ocel-c", "C.45 Mn.65 Si.25", (630, 800), EN="C45, C45E, C45R", WNR="1.0503, 1.1191, 1.1201",
      CSN="12050", DIN="Ck45, Cm45, Cf45", AISI="1045, ~1043", UNS="G10450", JIS="S45C", KS="SM45C",
      GB="=45", AFNOR="XC45, ~XC48", BS="~080M46"),
    G("C50", "ocel-c", "C.5 Mn.7", (650, 850), EN="C50, C50E, C50R", WNR="1.0540, 1.1206", CSN="12051",
      DIN="Ck50", AISI="1050", UNS="G10500", JIS="S50C"),
    G("C55", "ocel-c", "C.55 Mn.75", (700, 900), EN="C55, C55E, C55R", WNR="1.0535, 1.1203", CSN="12060",
      DIN="Ck55", AISI="1055", UNS="G10550", JIS="S55C"),
    G("C60", "ocel-c", "C.6 Mn.75", (750, 950), EN="C60, C60E, C60R", WNR="1.0601, 1.1221", CSN="12061",
      DIN="Ck60", AISI="1060", UNS="G10600", JIS="S58C"),
    G("AISI 1070", "ocel-c", "C.7 Mn.75", (800, 1000), AISI="1070", UNS="G10700"),
    G("11SMn30", "ocel-auto", "C.11 Mn1.1 S.3", (380, 570), EN="11SMn30", WNR="1.0715", CSN="11109",
      DIN="9SMn28", AISI="~1215"),
    G("AISI 1144", "ocel-auto", "C.44 Mn1.5 S.28", (650, 850), AISI="1144", UNS="G11440", EN="~44SMn28"),
    G("AISI 1141", "ocel-auto", "C.41 Mn1.5 S.1", (650, 850), AISI="1141", UNS="G11410"),
    # ------------------------------------------------------------ alloy steels for quenching and tempering / case hardening
    G("28Mn6", "ocel-leg", "C.28 Mn1.5", (700, 900), EN="28Mn6", WNR="1.1170", CSN="13141", AISI="~1330"),
    G("34Cr4", "ocel-leg", "C.34 Mn.75 Cr1.05", (800, 950), EN="34Cr4, 34CrS4", WNR="1.7033, 1.7037",
      AISI="~5132", JIS="SCr430"),
    G("41Cr4", "ocel-leg", "C.41 Mn.75 Cr1.05", (900, 1100), EN="41Cr4, 41CrS4, ~37Cr4",
      WNR="1.7035, 1.7039, ~1.7034", CSN="14140", AISI="5140", UNS="G51400", JIS="SCr440"),
    G("25CrMo4", "ocel-leg", "C.25 Mn.75 Cr1.05 Mo.23", (700, 900), EN="25CrMo4, 25CrMoS4",
      WNR="1.7218, 1.7213", CSN="15130", AISI="4130", UNS="G41300", JIS="~SCM430", AFNOR="25CD4"),
    G("34CrMo4", "ocel-leg", "C.34 Mn.75 Cr1.05 Mo.23", (900, 1100), EN="34CrMo4, 34CrMoS4",
      WNR="1.7220, 1.7226", CSN="15131, 15141", AISI="4135, 4137", UNS="G41350", JIS="SCM435",
      AFNOR="35CD4", GB="35CrMo", hidden="35CrMo4"),
    G("42CrMo4", "ocel-leg", "C.42 Mn.75 Cr1.05 Mo.23", (900, 1200), EN="42CrMo4, 42CrMoS4",
      WNR="1.7225, 1.7227", CSN="15142", AISI="4140, 4142", UNS="G41400, G41420", JIS="SCM440",
      AFNOR="42CD4", GB="42CrMo", hidden="42CrMo4V"),
    G("50CrMo4", "ocel-leg", "C.5 Mn.75 Cr1.05 Mo.23", (1000, 1300), EN="50CrMo4", WNR="1.7228", AISI="4150",
      UNS="G41500", JIS="~SCM445"),
    G("34CrNiMo6", "ocel-leg", "C.34 Mn.65 Cr1.5 Ni1.5 Mo.23", (1000, 1200), EN="34CrNiMo6", WNR="1.6582",
      CSN="16343", AISI="~4340"),
    G("AISI 4340", "ocel-leg", "C.4 Mn.7 Cr.8 Ni1.8 Mo.25", (900, 1300), AISI="4340", UNS="G43400",
      BS="817M40, En24", JIS="SNCM439", EN="~34CrNiMo6, ~36CrNiMo4", hidden="En24T, SNCM8"),
    G("39NiCrMo3", "ocel-leg", "C.39 Mn.65 Cr.8 Ni.85 Mo.2", (900, 1100), EN="39NiCrMo3", WNR="1.6510"),
    G("36CrNiMo4", "ocel-leg", "C.36 Mn.65 Cr1.05 Ni1.05 Mo.23", (900, 1100), EN="36CrNiMo4", WNR="1.6511",
      AISI="~9840"),
    G("30CrNiMo8", "ocel-leg", "C.3 Mn.45 Cr2.0 Ni2.0 Mo.4", (1000, 1300), EN="30CrNiMo8", WNR="1.6580"),
    G("30NiCrMo16-6", "ocel-leg", "C.3 Mn.45 Cr1.35 Ni4.0 Mo.45", (1000, 1500), EN="30NiCrMo16-6",
      AFNOR="30NCD16", DIN="30NiCrMo16"),
    G("26NiCrMoV11-5", "ocel-leg", "C.26 Mn.3 Cr1.25 Ni2.75 Mo.35 V.1", (750, 950), EN="26NiCrMoV11-5",
      WNR="1.6948"),
    G("20NiCrMo2-2", "ocel-cem", "C.2 Mn.8 Cr.5 Ni.55 Mo.2", (700, 1000), EN="20NiCrMo2-2", WNR="1.6523",
      AISI="8620", UNS="G86200", JIS="SNCM220"),
    G("AISI 8630", "ocel-leg", "C.3 Mn.8 Cr.5 Ni.55 Mo.2", (750, 1050), AISI="8630", UNS="G86300",
      EN="~30NiCrMo2-2"),
    G("16MnCr5", "ocel-cem", "C.16 Mn1.15 Cr.95", (800, 1100), EN="16MnCr5, 16MnCrS5", WNR="1.7131, 1.7139",
      CSN="14220", AISI="5115", JIS="~SCr415"),
    G("20MnCr5", "ocel-cem", "C.2 Mn1.25 Cr1.15", (1000, 1300), EN="20MnCr5, 20MnCrS5", WNR="1.7147, 1.7149",
      CSN="14221", AISI="~5120", JIS="~SMnC420"),
    G("18CrNiMo7-6", "ocel-cem", "C.18 Mn.7 Cr1.65 Ni1.55 Mo.3", (1000, 1300), EN="18CrNiMo7-6", WNR="1.6587",
      DIN="17CrNiMo6", CSN="~16326"),
    G("38MnVS6", "ocel-leg", "C.38 Mn1.4 Si.6 V.1 S.05", (850, 1000), EN="38MnVS6", WNR="1.1303"),
    G("27MnVS6", "ocel-leg", "C.27 Mn1.4 Si.6 V.1 S.05", (750, 900), EN="27MnVS6", WNR="1.1104"),
    G("49MnVS3", "ocel-leg", "C.49 Mn.8 V.1 S.05", (800, 950), EN="49MnVS3", WNR="1.1199"),
    # ------------------------------------------------------------ spring, bearing and tool steels
    G("100Cr6", "ocel-loz", "C1.0 Mn.35 Si.25 Cr1.45", (2000, 2400), EN="100Cr6", WNR="1.3505", CSN="14109",
      AISI="52100", UNS="G52986", JIS="SUJ2", BS="En31, 535A99", GOST="ShKh15, ŠCh15"),
    G("51CrV4", "ocel-pruz", "C.51 Mn.9 Cr1.05 V.15", (1100, 1600), EN="51CrV4", WNR="1.8159", CSN="15260",
      DIN="50CrV4", AISI="6150", UNS="G61500", JIS="SUP10"),
    G("54SiCr6", "ocel-pruz", "C.55 Mn.7 Si1.4 Cr.65", (1450, 1750), EN="54SiCr6", WNR="1.7102", AISI="9254",
      UNS="G92540", JIS="SUP12"),
    G("61SiCr7", "ocel-pruz", "C.61 Mn.85 Si1.8 Cr.3", (1500, 1800), EN="61SiCr7", WNR="1.7108", AISI="~9262"),
    G("55Cr3", "ocel-pruz", "C.55 Mn.85 Cr.85", (1300, 1600), EN="55Cr3", WNR="1.7176", AISI="~5155", JIS="SUP9"),
    G("AISI 5160", "ocel-pruz", "C.6 Mn.85 Cr.8", (1300, 1800), AISI="5160", UNS="G51600", JIS="SUP9A",
      EN="~60Cr3"),
    G("X37CrMoV5-1", "ocel-nastr", "C.37 Si1.0 Cr5.0 Mo1.3 V.4", (1500, 2000), EN="X37CrMoV5-1", WNR="1.2343",
      AISI="H11", UNS="T20811", JIS="SKD6"),
    G("1.2379", "ocel-nastr", "C1.55 Si.3 Mn.3 Cr11.5 Mo.7 V1.0", (1800, 2200), EN="X153CrMoV12", WNR="1.2379",
      CSN="19573", AISI="D2", UNS="T30402", JIS="SKD11"),
    G("1.2709", "ocel-nastr", "Ni18 Co9 Mo5 Ti.9 Al.1", (1900, 2150), note="maraging steel (tooling, SLM powder)",
      EN="X3NiCoMoTi18-9-5", WNR="1.2709", AISI="~Maraging 300", TRADE="MS1, Maraging 300"),
    G("1.2344", "ocel-nastr", "C.4 Si1.0 Cr5.2 Mo1.4 V1.0", (1500, 2000), EN="X40CrMoV5-1", WNR="1.2344",
      AISI="H13", UNS="T20813", JIS="SKD61"),
    # ------------------------------------------------------------ stainless (EN 10088)
    G("1.4301", "nerez-aust", "C.05 Cr18.25 Ni9.25", (500, 700), EN="X5CrNi18-10", WNR="1.4301",
      CSN="17240, ~17241", DIN="X5CrNi1810, ~V2A", AISI="304", UNS="S30400", JIS="SUS304", KS="STS304",
      GOST="08Ch18N10, 08Kh18N10"),
    G("1.4307", "nerez-aust", "C.02 Cr18.5 Ni9.5", (500, 670), EN="X2CrNi18-9, X2CrNi19-11", WNR="1.4307, 1.4306",
      CSN="17249", AISI="304L", UNS="S30403", JIS="SUS304L"),
    G("1.4401", "nerez-aust", "C.05 Cr17.25 Ni11.25 Mo2.25", (500, 700), EN="X5CrNiMo17-12-2", WNR="1.4401",
      CSN="17346", AISI="316", UNS="S31600", JIS="SUS316"),
    G("1.4404", "nerez-aust", "C.02 Cr17.25 Ni11.25 Mo2.25", (500, 700), EN="X2CrNiMo17-12-2", WNR="1.4404",
      CSN="17349", AISI="316L", UNS="S31603", JIS="SUS316L"),
    G("1.4571", "nerez-aust", "C.05 Cr17.25 Ni11.75 Mo2.25 Ti.4", (500, 700), EN="X6CrNiMoTi17-12-2",
      WNR="1.4571", CSN="17348", AISI="316Ti", UNS="S31635", GOST="~10Ch17N13M2T"),
    G("1.4541", "nerez-aust", "C.05 Cr18 Ni10.5 Ti.4", (500, 700), EN="X6CrNiTi18-10", WNR="1.4541",
      CSN="17248, 17247", DIN="X10CrNiTi18-9, X6CrNiTi1810", AISI="321", UNS="S32100", JIS="SUS321",
      GOST="08Ch18N10T, ~12Ch18N10T", hidden="X10CrNiTi"),
    G("1.4550", "nerez-aust", "C.05 Cr18 Ni10.5 Nb.6", (510, 740), EN="X6CrNiNb18-10", WNR="1.4550",
      AISI="347", UNS="S34700", JIS="SUS347"),
    G("1.4006", "nerez-mart", "C.12 Cr12.5", (450, 650), EN="X12Cr13", WNR="1.4006", CSN="17021",
      DIN="X10Cr13", AISI="410, ~403", UNS="S41000", JIS="SUS410", GOST="12Ch13"),
    G("1.4021", "nerez-mart", "C.2 Cr13", (650, 950), EN="X20Cr13", WNR="1.4021", CSN="17022", AISI="420",
      UNS="S42000", JIS="SUS420J1", GOST="20Ch13"),
    G("1.4016", "nerez-ferit", "C.05 Cr17", (430, 630), EN="X6Cr17", WNR="1.4016", CSN="17040", AISI="430",
      UNS="S43000", JIS="SUS430", GOST="12Ch17"),
    G("1.4057", "nerez-mart", "C.17 Cr16 Ni2.0", (800, 1000), EN="X17CrNi16-2", WNR="1.4057", CSN="17145",
      DIN="X20CrNi172", AISI="431", UNS="S43100", JIS="SUS431", GOST="14Ch17N2"),
    G("1.4313", "nerez-mart", "C.03 Cr12.75 Ni4.0 Mo.5", (780, 980), EN="X3CrNiMo13-4", WNR="1.4313",
      ASTM="~CA6NM", hidden="13Cr-4Ni-Mo, 13Cr-4Ni"),
    G("1.4542", "nerez-ph", "C.04 Cr15.5 Ni4.25 Cu3.5 Nb.3", (930, 1310), EN="X5CrNiCuNb16-4", WNR="1.4542",
      AISI="630", UNS="S17400", JIS="SUS630", TRADE="17-4PH"),
    G("1.4545", "nerez-ph", "C.04 Cr14.75 Ni4.5 Cu3.5 Nb.3", (930, 1310), EN="X5CrNiCu15-5", WNR="1.4545",
      UNS="S15500", TRADE="15-5PH"),
    G("1.4534", "nerez-ph", "C.03 Cr12.75 Ni8.0 Mo2.25 Al1.1", (1100, 1500), EN="X3CrNiMoAl13-8-2",
      WNR="1.4534", UNS="S13800", TRADE="PH13-8Mo"),
    G("1.4532", "nerez-ph", "C.07 Cr15 Ni7.0 Mo2.5 Al1.1", (1100, 1600), EN="X8CrNiMoAl15-7-2", WNR="1.4532",
      UNS="S15700", TRADE="PH15-7Mo"),
    G("Custom 450", "nerez-ph", "C.03 Cr15 Ni6.0 Mo.75 Cu1.5 Nb.5", (1000, 1300), UNS="S45000", AISI="XM-25",
      TRADE="Custom 450"),
    G("Custom 455", "nerez-ph", "C.03 Cr11.75 Ni8.5 Cu2.0 Ti1.2 Nb.3", (1300, 1700), UNS="S45500",
      AISI="XM-16", TRADE="Custom 455"),
    G("1.4462", "nerez-duplex", "C.02 Cr22.5 Ni5.5 Mo3.1 N.17", (650, 880), EN="X2CrNiMoN22-5-3",
      WNR="1.4462", UNS="S32205, S31803", TRADE="2205"),
    # ------------------------------------------------------------ litiny (EN 1561, 1563, 1562, 16079)
    G("EN-GJL-150", "litina-seda", "C3.5 Si2.2", (150, 250), EN="EN-GJL-150", WNR="0.6015", DIN="GG15, GG-15",
      JIS="FC150"),
    G("EN-GJL-200", "litina-seda", "C3.4 Si2.0", (200, 300), EN="EN-GJL-200", WNR="0.6020", DIN="GG20, GG-20",
      JIS="FC200"),
    G("EN-GJL-250", "litina-seda", "C3.3 Si1.9", (250, 350), EN="EN-GJL-250", WNR="0.6025", DIN="GG25, GG-25",
      JIS="FC250"),
    G("EN-GJL-300", "litina-seda", "C3.2 Si1.8", (300, 400), EN="EN-GJL-300", WNR="0.6030", DIN="GG30, GG-30",
      JIS="FC300"),
    G("EN-GJS-400-15", "litina-tvarna", "C3.6 Si2.5", (400, 500), EN="EN-GJS-400-15, EN-GJS-400",
      WNR="0.7040", DIN="GGG-40, GGG40", JIS="FCD400", ASTM="~60-40-18"),
    G("EN-GJS-500-7", "litina-tvarna", "C3.6 Si2.4", (500, 600), EN="EN-GJS-500-7, EN-GJS-500", WNR="0.7050",
      DIN="GGG-50, GGG50", JIS="FCD500"),
    G("EN-GJS-600-3", "litina-tvarna", "C3.6 Si2.3", (600, 700), EN="EN-GJS-600-3, EN-GJS-600", WNR="0.7060",
      CSN="42 2306", DIN="GGG-60, GGG60", JIS="FCD600"),
    G("EN-GJS-700-2", "litina-tvarna", "C3.6 Si2.3", (700, 800), EN="EN-GJS-700-2, EN-GJS-700", WNR="0.7070",
      CSN="42 2307", DIN="GGG-70, GGG70", JIS="FCD700", ASTM="~100-70-03"),
    G("EN-GJS-800-2", "litina-tvarna", "C3.6 Si2.3", (800, 900), EN="EN-GJS-800-2, EN-GJS-800", WNR="0.7080",
      CSN="42 2308", DIN="GGG-80, GGG80", JIS="FCD800", AFNOR="FGS 800-2"),
    G("EN-GJMB-350-10", "litina-temper", "C2.5 Si1.3", (350, 450), EN="EN-GJMB-350-10", WNR="0.8135",
      DIN="GTS-35-10, GTS-35"),
    G("EN-GJMB-450-6", "litina-temper", "C2.5 Si1.3", (450, 550), EN="EN-GJMB-450-6", WNR="0.8145",
      DIN="GTS-45-06, GTS-45"),
    G("EN-GJMB-550-4", "litina-temper", "C2.5 Si1.3", (550, 650), EN="EN-GJMB-550-4", WNR="0.8155",
      DIN="GTS-55-04, GTS-55"),
    G("EN-GJMB-700-2", "litina-temper", "C2.5 Si1.3", (700, 800), EN="EN-GJMB-700-2", WNR="0.8170",
      DIN="GTS-70-02, GTS-70"),
    G("EN-GJV-450", "litina-vermik", "C3.6 Si2.3", (450, 550), EN="EN-GJV-450", hidden="GJV-450"),
    G("EN-GJV-500", "litina-vermik", "C3.6 Si2.3", (500, 600), EN="EN-GJV-500", hidden="GJV-500"),
    # ------------------------------------------------------------ aluminium (EN 573-3)
    G("EN AW-1050A", "al-1", "", (65, 140), EN="EN AW-1050A, Al99.5", WNR="3.0255", AISI="~1050"),
    G("EN AW-1100", "al-1", "Cu.12", (90, 150), EN="EN AW-1100, Al99.0Cu", WNR="3.0205", UNS="A91100",
      hidden="1100"),
    G("EN AW-2014", "al-2", "Cu4.4 Si.8 Mn.8 Mg.5", (420, 500), EN="EN AW-2014, AlCu4SiMg", WNR="3.1255",
      UNS="A92014", ALCOA="14S", hidden="2014"),
    G("EN AW-2017A", "al-2", "Cu4.0 Mg.7 Mn.7 Si.5", (380, 440), EN="EN AW-2017A, AlCu4MgSi(A)",
      WNR="3.1325", DIN="AlCuMg1", ALCOA="17S", hidden="2017, duralumin, dural"),
    G("EN AW-2024", "al-2", "Cu4.4 Mg1.5 Mn.6", (430, 490), EN="EN AW-2024, AlCu4Mg1", WNR="3.1355",
      CSN="42 4203", DIN="AlCuMg2", UNS="A92024", ALCOA="24S", hidden="2024"),
    G("2A12", "al-2", "Cu4.4 Mg1.5 Mn.6", (420, 480), note="Chinese equivalent of 2024", GB="2A12, LY12",
      hidden="LY12"),
    G("EN AW-2124", "al-2", "Cu4.4 Mg1.5 Mn.6", (440, 490), EN="EN AW-2124", UNS="A92124", hidden="2124"),
    G("EN AW-2219", "al-2", "Cu6.3 Mn.3 Zr.15", (400, 460), EN="EN AW-2219, AlCu6Mn", UNS="A92219",
      hidden="2219"),
    G("EN AW-2618A", "al-2", "Cu2.3 Mg1.6 Ni1.0 Fe1.1", (400, 450), EN="EN AW-2618A, AlCu2Mg1.5Ni",
      UNS="A92618", hidden="2618"),
    G("AA 2025", "al-2", "Cu4.4 Si.8 Mn.8", (380, 420), ALCOA="25S", UNS="A92025", hidden="2025"),
    G("EN AW-5083", "al-5", "Mg4.45 Mn.7", (270, 350), EN="EN AW-5083, AlMg4.5Mn0.7", WNR="3.3547",
      DIN="AlMg4.5Mn", UNS="A95083", hidden="5083"),
    G("EN AW-5086", "al-5", "Mg4.0 Mn.45", (240, 310), EN="EN AW-5086, AlMg4", WNR="3.3545", UNS="A95086",
      hidden="5086, Al5086"),
    G("EN AW-6060", "al-6", "Mg.5 Si.45", (150, 245), EN="EN AW-6060, AlMgSi", WNR="3.3206", DIN="AlMgSi0.5",
      hidden="6060"),
    G("EN AW-6061", "al-6", "Mg1.0 Si.6 Cu.28 Cr.2", (260, 320), EN="EN AW-6061, AlMg1SiCu", WNR="3.3211",
      UNS="A96061", ALCOA="61S", hidden="6061"),
    G("EN AW-6082", "al-6", "Si1.0 Mg.9 Mn.7", (290, 340), EN="EN AW-6082, AlSi1MgMn", WNR="3.2315",
      DIN="AlMgSi1", hidden="6082"),
    G("EN AW-7020", "al-7", "Zn4.5 Mg1.2", (340, 410), EN="EN AW-7020, AlZn4.5Mg1", WNR="3.4335",
      hidden="7020"),
    G("EN AW-7075", "al-7", "Zn5.6 Mg2.5 Cu1.6 Cr.23", (520, 580), EN="EN AW-7075, AlZn5.5MgCu",
      WNR="3.4365", DIN="AlZnMgCu1.5", UNS="A97075", ALCOA="75S", PN="PA9", hidden="7075"),
    G("ČSN 42 4222", "al-7", "Zn6.0 Mg2.0 Cu1.0", (500, 560), note="composition approximate, based on grade AlZn6Mg2Cu",
      CSN="42 4222", EN="~AlZn6Mg2Cu"),
    G("7A04", "al-7", "Zn6.0 Mg2.4 Cu1.7 Mn.35 Cr.15", (500, 600), note="Chinese equivalent of 7075",
      GB="7A04, LC4", hidden="LC4"),
    G("AA 7076", "al-7", "Zn7.5 Mg1.6 Cu.65 Mn.5", (480, 540), ALCOA="76S", UNS="A97076", hidden="7076"),
    G("EN AC-43000", "al-lit", "Si10 Mg.35", (220, 300), EN="EN AC-43000, EN AC-AlSi10Mg(a)", WNR="3.2381",
      DIN="G-AlSi10Mg", hidden="AlSi10Mg"),
    G("EN AC-44100", "al-lit", "Si12", (150, 200), EN="EN AC-44100, EN AC-AlSi12(b)", DIN="G-AlSi12",
      TRADE="~Silumin", hidden="AlSi12"),
    G("Scalmalloy", "al-5", "Mg4.6 Sc.7 Zr.3 Mn.5", (500, 540), note="AlMgScZr pro L-PBF (APWORKS)",
      TRADE="Scalmalloy"),
    G("AZ91D", "mg", "Al9.0 Zn.7 Mn.2", (200, 260), note="magnesium die casting",
      EN="EN-MC21120, MgAl9Zn1(A)", ASTM="AZ91D", UNS="M11916", hidden="AZ91"),
    # ------------------------------------------------------------ titanium, nickel, copper
    G("Ti-6Al-4V", "ti", "Al6.0 V4.0", (900, 1100), EN="TiAl6V4", WNR="3.7165", UNS="R56400",
      ASTM="Grade 5, Gr5", GB="TC4", GOST="~VT6", BS="IMI 318", hidden="Ti6Al4V, 6Al-4V, Ti64"),
    G("Ti Grade 2", "ti", "", (345, 480), WNR="3.7035", UNS="R50400", ASTM="Grade 2", hidden="CP-Ti"),
    G("Inconel 718", "ni", "Ni52 Cr19 Fe18 Nb5.1 Mo3.0 Ti.9 Al.5", (1240, 1450), EN="NiCr19Fe19Nb5Mo3",
      WNR="2.4668", UNS="N07718", TRADE="Inconel 718, Alloy 718", hidden="IN718"),
    G("Inconel 625", "ni", "Ni61 Cr21.5 Mo9.0 Nb3.6 Fe2.5", (830, 1030), EN="NiCr22Mo9Nb", WNR="2.4856",
      UNS="N06625", TRADE="Inconel 625, Alloy 625", hidden="IN625"),
    G("CoCrMo", "co", "Co64 Cr28 Mo6", (900, 1300), note="CoCr28Mo6 – implants, dental prostheses, L-PBF powder",
      ASTM="F75, F1537", UNS="R30075, R31537", hidden="Co28Cr6Mo, CoCr28Mo6"),
    G("Inconel X-750", "ni", "Ni70 Cr15.5 Fe7.0 Ti2.5 Al.7 Nb1.0", (1100, 1300), EN="NiCr15Fe7TiAl",
      WNR="2.4669", UNS="N07750", TRADE="Inconel X-750, Inconel X"),
    G("Inconel 751", "ni", "Ni70 Cr15.5 Fe7.0 Ti2.3 Al1.2 Nb1.0", (1100, 1300), UNS="N07751",
      TRADE="Inconel 751"),
    G("Waspaloy", "ni", "Ni58 Cr19.5 Co13.5 Mo4.3 Ti3.0 Al1.4", (1200, 1400), WNR="2.4654", UNS="N07001",
      TRADE="Waspaloy"),
    G("Nimonic 90", "ni", "Ni57 Cr19.5 Co18 Ti2.5 Al1.4", (1100, 1300), WNR="2.4632", UNS="N07090",
      TRADE="Nimonic 90"),
    G("Udimet 500", "ni", "Ni53 Cr18 Co18.5 Mo4.0 Ti2.9 Al2.9", (1200, 1400), UNS="N07500", TRADE="Udimet 500"),
    G("Incoloy 901", "ni", "Ni42.5 Fe36 Cr12.5 Mo5.8 Ti2.8", (1100, 1300), UNS="N09901", TRADE="Incoloy 901"),
    G("CuZn30", "cu-zn", "Cu70 Zn30", (280, 550), EN="CuZn30, CW505L", WNR="2.0265", DIN="Ms70",
      UNS="C26000", hidden="brass 7:3, brass 70/30"),
    G("CuZn40", "cu-zn", "Cu60 Zn40", (340, 500), EN="CuZn40, CW509L", WNR="2.0360", DIN="Ms60",
      UNS="C28000", TRADE="Muntz metal"),
    G("CuZn39Pb3", "cu-zn", "Cu58 Zn39 Pb3", (360, 560), note="free-cutting brass",
      EN="CuZn39Pb3, CW614N", WNR="2.0401", CSN="42 3213", DIN="CuZn39Pb3, Ms58",
      UNS="C38500", hidden="Ms58"),
    G("CuSn8", "cu", "Cu92 Sn8", (350, 550), note="tin bronze",
      EN="CuSn8, CW453K", WNR="2.1030", CSN="42 3123", DIN="CuSn8", UNS="C52100", hidden="bronz"),
    G("CuAl10Fe3Mn2", "cu", "Cu85 Al10 Fe3 Mn2", (550, 750), note="aluminum bronze",
      EN="CuAl10Fe3Mn2, CW306G", WNR="2.0936", CSN="42 3046", DIN="CuAl10Fe", hidden="Ampco"),
    G("CuCrZr", "cu", "Cu98.9 Cr.8 Zr.1", (370, 500), note="precipitation-hardenable copper (electrodes, cooling, L-PBF powder)",
      EN="CuCr1Zr, CW106C", WNR="2.1293", UNS="C18150"),
    G("Cu-ETP", "cu", "Cu99.9", (200, 300), EN="Cu-ETP, CW004A", WNR="2.0065", DIN="E-Cu58", UNS="C11000",
      hidden="copper electrolytic"),
]


# ---------------------------------------------------------------------- search by designation
def _ascii_upper(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper().replace(",", ".")


def norm(s: str) -> str:
    """Normalised designation: capital letters without diacritics, only A-Z, 0-9 and a dot between digits."""
    s = re.sub(r"[^A-Z0-9.]", "", _ascii_upper(s))
    return re.sub(r"(?<!\d)\.|\.(?!\d)", "", s)


def keys(text: str, max_join: int = 4) -> set[str]:
    """Keys of the text: joining neighbouring words (ČSN "12 050", "St 52-3"), without suffixes and prefixes of standards."""
    toks = [t for t in re.split(r"[^A-Z0-9.]+", _ascii_upper(text)) if t.strip(".")]
    out = set()
    for i in range(len(toks)):
        for j in range(i + 1, min(len(toks), i + max_join) + 1):
            k = norm("".join(toks[i:j]))
            if not k:
                continue
            out.add(k)
            if m := re.match(r"^(\d{5})\.\d{1,2}$", k):           # ČSN 12050.6 → 12050
                out.add(m[1])
            if m := re.match(r"^(?:AISI|SAE|CSN|STN)(\d.*)$", k):   # SAE1045 → 1045
                out.add(m[1])
            if m := re.match(r"^(\d{3,5})[A-Z]{2,}$", k):         # 6061AL, 347STAINLESS → 6061, 347
                out.add(m[1])
    return out


@lru_cache(maxsize=1)
def _index() -> tuple[dict[str, list[tuple[Grade, bool]]], dict[str, list[tuple[Grade, bool]]]]:
    tok: dict[str, list] = {}
    exact: dict[str, list] = {}
    for g in GRADES:
        entries = [(n, n.startswith("~")) for vals in g.names.values() for n in vals] + [(g.key, False)]
        entries += [(h, False) for h in g.hidden]
        for n, approx in entries:
            target = exact if n.startswith("=") else tok
            k = norm(n.lstrip("~="))
            if k and all(x[0] is not g for x in target.get(k, [])):
                target.setdefault(k, []).append((g, approx))
    return tok, exact


# A four-digit number is used in AISI/SAE steels (1060 = C60) and also for aluminium (Aluminum Association, series 1xxx–8xxx, 1060 =
# Al99.6; "9254 AL FG" = steel killed with aluminium). Aluminium
# is recognised by the word (Aluminium, Al, EN AW, AA), by the state after the number ("1060-H12", "6061-T6", "1100-O") or by the name
# of the SolidWorks library "1060 Alloy" (steels there have "AISI …" / "… Steel"); the word for steel takes precedence.
_AL_WORDS = {"AL", "ALU", "ALUMINIUM", "ALUMINUM", "HLINIK", "AW", "AA"}
_STEEL_WORDS = {"STEEL", "OCEL", "STAHL", "ACIER", "AISI", "SAE"}
_AL_NUMBER = re.compile(r"(?<![\dA-Z.])[1-8]\d{3}(?:[\s-]*(?:O|H\d{1,3}|T\d{1,4})(?![A-Z\d])|\s+ALLOY\b)")


def aluminium_text(text: str) -> bool:
    """The text denotes an aluminium alloy (a four-digit number is then not an AISI/SAE steel)."""
    up = _ascii_upper(text)
    words = set(re.split(r"[^A-Z0-9]+", up))
    if words & _STEEL_WORDS:
        return False
    return bool(words & _AL_WORDS) or _AL_NUMBER.search(up) is not None


def lookup(text: str) -> list[tuple[Grade, bool]]:
    """Groups matching the text, [(group, only_approximately)] – the longest match first."""
    tok, exact = _index()
    hits: dict[int, tuple[Grade, bool, int]] = {}
    cand = [(k, tok[k]) for k in keys(text) if k in tok]
    whole = norm(text)
    if whole in exact:
        cand.append((whole, exact[whole]))
    al = aluminium_text(text)
    for k, lst in cand:
        for g, approx in lst:
            if al and g.base != "al" and len(k) == 4 and k.isdigit() and "1" <= k[0] <= "8":
                continue                    # "1060 Alloy", "1060-H12" = aluminium 1060, not steel AISI 1060
            prev = hits.get(id(g))
            if prev is None or len(k) > prev[2] or (len(k) == prev[2] and prev[1] and not approx):
                hits[id(g)] = (g, approx, len(k))
    return [(g, a) for g, a, _ in sorted(hits.values(), key=lambda x: (-x[2], x[1]))]


def grade_of(text: str) -> Grade | None:
    hits = lookup(text)
    return hits[0][0] if hits else None


def by_key(key: str) -> Grade:
    for g in GRADES:
        if g.key == key:
            return g
    raise KeyError(key)


# ---------------------------------------------------------------------- podobnost
@dataclass
class Profile:
    """What is compared: family, composition, strength (for a database record its own values).

    complete = the composition is complete (nominal for a group: a missing element = 0); records from the literature
    often have only some elements → the missing ones are not compared.
    """
    family: str | None
    base: str | None
    chem: dict[str, float]
    rm: float | None
    complete: bool = True

    @property
    def cls(self) -> str | None:
        return self.family.split("-")[0] if self.family else None


def numeric_chem(chem: dict | None) -> dict[str, float]:
    return {k: float(v) for k, v in (chem or {}).items() if isinstance(v, (int, float)) and k in _ELEMENTS}


_ELEMENTS = {"C", "Mn", "Si", "P", "S", "Cr", "Ni", "Mo", "V", "W", "Ti", "Cu", "Al", "Nb", "Co", "Mg", "Fe",
             "Zn", "Zr", "N", "B", "Sn", "Ta", "Hf"}


def profile(g: Grade, chem: dict | None = None, rm: float | None = None) -> Profile:
    chem = numeric_chem(chem)
    own = len(chem) >= 2
    return Profile(g.family, g.base, chem if own else g.chem, rm or g.rm_mid, complete=not own)


def guess_family(chem: dict) -> str | None:
    """Family by composition (for records without a known designation)."""
    c = numeric_chem(chem)
    if "C" not in c:
        return None
    if c["C"] >= 1.8:
        return "litina-seda"
    if c.get("Cr", 0) >= 10.5:
        return "nerez-aust" if c.get("Ni", 0) >= 6 else "nerez-mart"
    if c.get("Cr", 0) + c.get("Ni", 0) + 3 * c.get("Mo", 0) + 5 * c.get("V", 0) > 1.0:
        return "ocel-leg"
    return "ocel-c" if c["C"] >= 0.25 else "ocel-konstr"


# elements below the limit are not compared in steel (Mn, Si, Cu are almost everywhere as accompanying elements)
_IGNORE_BELOW = {"Mn": 1.0, "Si": 1.0, "Cu": 0.5, "P": 0.2, "S": 0.2, "Fe": 2.0}
_SCALE = {"C": 0.05}


def _pairs(a: Profile, b: Profile):
    """Compared elements: (element, x, y). An element that an incomplete composition does not state is left out."""
    if not a.chem or not b.chem:
        return
    for e in sorted(set(a.chem) | set(b.chem)):
        if (e not in a.chem and not a.complete) or (e not in b.chem and not b.complete):
            continue
        x, y = a.chem.get(e, 0.0), b.chem.get(e, 0.0)
        limit = 2.1 if (e == "Mn" and a.cls == "nerez") else _IGNORE_BELOW.get(e, 0.0)
        if a.base in ("ocel", "litina") and max(x, y) < limit:
            continue
        yield e, x, y


def _w(e: str, x: float, y: float) -> float:
    return (x - y) / (_SCALE.get(e, 0.3) + 0.1 * max(x, y))


def distance(a: Profile, b: Profile) -> float:
    """0 = same; ~1 = very close; > 6 dissimilar. Another class (steel × stainless × cast iron × Al…) = ∞."""
    if not a.cls or a.cls != b.cls:
        return math.inf
    d = 0.0 if a.family == b.family else 1.5
    if a.chem and b.chem:
        d += math.sqrt(sum(_w(e, x, y) ** 2 for e, x, y in _pairs(a, b)))
    else:
        d += 2.0                                       # composition unknown
    if a.rm and b.rm:
        d += abs(math.log(a.rm / b.rm)) / 0.2
    else:
        d += 0.5                                       # strength unknown
    return d


def similar(g: Grade, n: int = 8, max_dist: float = 6.0) -> list[tuple[Grade, float]]:
    p = profile(g)
    out = [(o, distance(p, profile(o))) for o in GRADES if o is not g]
    out = sorted((x for x in out if x[1] <= max_dist), key=lambda x: x[1])
    return out[:n]


def difference(a: Profile, b: Profile) -> str:
    """Short description of the main difference (for the list of similar materials)."""
    parts = []
    if a.family != b.family and b.family:
        parts.append(FAMILIES[b.family][1])
    from fea_core.i18n import decimal
    diffs = [(abs(_w(e, x, y)), e, y - x) for e, x, y in _pairs(a, b)
             if abs(x - y) > 0.1 * max(x, y, 0.05) and abs(x - y) >= 0.03]
    for _, e, dv in sorted(diffs, reverse=True)[:2]:
        parts.append(decimal(f"{e} {dv:+.2g} %"))
    if a.rm and b.rm and abs(b.rm / a.rm - 1) > 0.08:
        parts.append(f"Rm {100 * (b.rm / a.rm - 1):+.0f} %")
    return ", ".join(parts)
