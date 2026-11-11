"""Material database PragTic (J. Papuga, https://www.pragtic.com/vmat.php) – fatigue and static
properties from the literature (SAE J1099, Automotive Steel Design Manual, articles…), with the source for each record.

Data: `data/pragtic_vmat.json` (downloaded by `pragtic_fetch.py`). The quantities are recognised by the description
("Fatigue limit"…), not by the designation – the designations differ between the loading regimes (f1, b1, t1…).

Conversion to S-N (amplitudes, R = −1, smooth specimen – the Marin factors are added in the analysis):
  1. fatigue limit in tension-compression σC (+ k, NC from the record; if missing, computed from Basquin)  → k2 = 0
  2. Basquin σa = σf'·(2N)^b from tension-compression: SD at 10⁶, k = −1/b, fatigue limit unknown → Haibach 2k − 1
  3. fatigue limit in bending (rotating / alternating) – higher than in tension-compression (stress gradient),
     corresponds to Se' according to Shigley (the same basis as the estimate from Rm).
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from .sn import SNCurve

DATA = Path(__file__).with_name("data") / "pragtic_vmat.json"
URL = "https://www.pragtic.com/vmat.php"

PUSH_PULL = "Fully reversed push-pull"
STATIC = "Static data"
MODES = {
    STATIC: "Static properties",
    PUSH_PULL: "Fully reversed tension-compression (R = −1)",
    "Repeated tension": "Pulsating tension (R = 0)",
    "Rotating bending": "Ohyb za rotace (R = −1)",
    "Fully reversed bending": "Fully reversed bending (R = −1)",
    "Repeated bending": "Pulsating bending (R = 0)",
    "Fully reversed torsion": "Fully reversed torsion (R = −1)",
    "Repeated torsion": "Pulsating torsion (R = 0)",
}
QUANTITIES = {
    "Ultimate tensile strength": "Mez pevnosti v tahu Rm",
    "Tensile yield stress": "Mez kluzu v tahu Re (Rp0,2)",
    "Ultimate shear strength": "Mez pevnosti ve smyku",
    "Shear yield stress": "Mez kluzu ve smyku",
    "Tensile elasticity modulus": "Young's modulus E",
    "Shear elasticity modulus": "Shear modulus G",
    "Poisson's ratio": "Poisson's ratio ν",
    "Elongation at fracture": "Elongation A",
    "Reduction of area at fracture": "Kontrakce Z",
    "Density": "Hustota",
    "Hardness (acc. to Brinell)": "Tvrdost HB",
    "Hardness (acc. to Vickers)": "Tvrdost HV",
    "Hardness (acc. to Rockwell)": "Tvrdost HRC",
    "Coefficient of static strength": "Hardening coefficient K (tension)",
    "Exponent of static strength": "Hardening exponent n (tension)",
    "Coefficient of static strength in torsion": "Hardening coefficient (torsion)",
    "Exponent of static strength in torsion": "Hardening exponent (torsion)",
    "Ambient temperature": "Teplota zkoušky",
    "True fracture strain": "True fracture strain εf",
    "Diameter of specimen at active cross-section": "Specimen diameter",
    "Load frequency": "Loading frequency",
    "Fatigue limit": "Endurance limit (amplitude)",
    "Exponent of S-N curve": "S-N curve exponent k",
    "Number of cycles at fatigue limit": "Cycles at the endurance limit",
    "Fatigue strength coefficient": "Fatigue strength coefficient σf' (Basquin)",
    "Fatigue strength exponent": "Fatigue strength exponent b (Basquin)",
    "Fatigue ductility coefficient": "Fatigue ductility coefficient εf' (Manson-Coffin)",
    "Fatigue ductility exponent": "Fatigue ductility exponent c (Manson-Coffin)",
    "Cyclic hardnening coefficient": "Cyclic hardening coefficient K'",
    "Cyclic hardnening exponent": "Cyclic hardening exponent n'",
    "Cyclic yield stress": "Cyclic yield stress",
}
BASQUIN_ND = 1e6

# Citation of the database – the same text in the S-N dialogue, the help, the Word report and THIRD_PARTY_LICENSES.md.
AUTHOR = "Ing. Jan Papuga, Ph.D. (Czech Technical University in Prague)"
CITATION = ("Papuga, J.: PragTic – material database of fatigue properties (Fatigue Data Repository). Czech Technical University in Prague, Faculty of Mechanical Engineering. https://www.pragtic.com/vmat.php")
NOTICE = ("The fatigue and static data of the PragTic database belong to its author (J. Papuga, pragtic.com). FEA Studio only reads them and states the source of every value taken over; the original source (paper, standard, handbook) is given for each record.")


def available() -> bool:
    """Is the database downloaded (without loading it)?"""
    return DATA.exists()


def citation(db: dict | None = None) -> str:
    """Citation with the date of the download of the data, if the database is available."""
    if db is None and available():
        try:
            db = database()
        except (OSError, ValueError):
            db = None
    date = (db or {}).get("downloaded")
    return CITATION + (f" (data downloaded {date})" if date else "")


@lru_cache(maxsize=1)
def database() -> dict:
    if not DATA.exists():
        raise FileNotFoundError(
            f"The PragTic database is not part of the installation (it is not distributed without the author's consent). For personal use, download it with the command: python -m fea_core.fatigue.pragtic_fetch (it is saved to {DATA}). Fatigue also works without it – S-N estimated from Rm or a manually entered S-N.")
    return json.loads(DATA.read_text(encoding="utf-8"))


def records() -> list[dict]:
    return database()["records"]


def by_id(rid: int) -> dict:
    for r in records():
        if r["id"] == rid:
            return r
    raise KeyError(rid)


def value(rec: dict, mode: str, desc: str) -> float | None:
    """Numerical value according to the loading regime and the English description of the quantity."""
    for row in rec["data"].get(mode, {}).values():
        if row["desc"] == desc and isinstance(row["value"], (int, float)):
            return float(row["value"])
    return None


def title(rec: dict) -> str:
    parts = [rec["mark"]]
    if rec.get("alt_mark"):
        parts.append(f"({rec['alt_mark']})")
    if rec.get("treatment"):
        parts.append(f"– {rec['treatment']}")
    return " ".join(parts)


def _hay(r: dict) -> str:
    return " ".join(str(r.get(k, "")) for k in ("mark", "alt_mark", "standard", "group", "treatment",
                                                  "comment")).lower()


def search(text: str = "", sn_only: bool = False) -> list[dict]:
    """Records whose designation / alt. designation / standard / group / treatment contain all the words."""
    words = text.lower().split()
    return [r for r in records() if all(w in _hay(r) for w in words) and (not sn_only or sn_options(r))]


_NOISE = {"CSN", "STN", "EN", "DIN", "AISI", "SAE", "UNS", "ASTM", "JIS", "GOST", "AFNOR", "BS", "GB", "PN", "KS",
          "WNR", "NR", "AW", "AC", "OCEL", "STEEL", "ALLOY", "SLITINA", "NEREZ", "HLINIK", "ALUMINIUM", "ALUMINUM",
          "TYPE", "GRADE", "LITINA", "MATERIAL"}


# ---------------------------------------------------------------------- designation converter, similar
# designations without a composition that the converter does not know – the family for the search of similar ones
_HINTS = [(r"^HF\d|^9[5-8]0[CX]$|^\d+Y\d+T$|MAN-TEN|^A5(15|16)|^RQC", "ocel-mikro"),
          (r"^(DQSK|DDQ\+?|CQ|100[4-9])", "ocel-plech"),
          (r"^1\d{3}$|^10B\d|^15B\d|CARBON STEEL|MILD STEEL|HARD STEEL", "ocel-c"),
          (r"NODULAR|^GGG|GJS|FGS", "litina-tvarna"), (r"^TE \d|^GTS|TEMPER", "litina-temper"),
          (r"CAST IRON|^GE \d|^GG", "litina-seda"), (r"STAINLESS|^STS", "nerez-aust"),
          # ČSN 41 classes: 10, 11 structural, 12 carbon, 13–16 alloy, 17 0xx/1xx Cr, 17 2xx/3xx CrNi
          (r"^1[01]\d{3}\b", "ocel-konstr"), (r"^12\d{3}\b", "ocel-c"), (r"^1[3-6]\d{3}\b", "ocel-leg"),
          (r"^17[01]\d{2}\b", "nerez-mart"), (r"^17[23]\d{2}\b", "nerez-aust")]


@lru_cache(maxsize=1)
def _grades_of_records() -> dict[int, tuple]:
    from .. import grades
    out = {}
    for r in records():
        hit = None
        for fld in ("mark", "alt_mark", "standard"):
            hits = grades.lookup(r.get(fld) or "")
            if hits:
                hit = hits[0]
                break
        out[r["id"]] = hit
    return out


def grade_of_record(rec: dict):
    """(Grade, only_approximately) by the designation of the record, otherwise None."""
    return _grades_of_records().get(rec["id"])


def record_profile(rec: dict):
    from .. import grades
    hit = grade_of_record(rec)
    rm = value(rec, STATIC, "Ultimate tensile strength")
    if hit:
        return grades.profile(hit[0], rec.get("chemistry"), rm)
    fam = grades.guess_family(rec.get("chemistry") or {})
    if fam is None:
        up = rec["mark"].upper()
        fam = next((f for pat, f in _HINTS if re.search(pat, up)), None)
    if fam is None:
        return None
    return grades.Profile(fam, grades.FAMILIES[fam][0], grades.numeric_chem(rec.get("chemistry")), rm,
                          complete=False)


def find(text: str, sn_only: bool = False, n_similar: int = 20, max_dist: float = 6.0):
    """Search by the designation of any standard.

    Returns (groups [(Grade, approximately)], matching records, similar [(record, distance, description of the difference)]).
    Matching = records of the same group (across all standards) + text match; similar = the nearest other
    records (family, composition, Rm) to the first group found.
    """
    from .. import grades
    hits = grades.lookup(text) if text.strip() else []
    ids = {id(g) for g, _ in hits}
    # words that are neither a designation nor a name of a standard narrow the result ("1045 quenched")
    filt = [w.lower() for w in text.split() if re.fullmatch(r"[^\W\d_]{3,}", w)
            and grades.norm(w) not in _NOISE and not grades.lookup(w)]

    def ok(r):
        return all(w in _hay(r) for w in filt)
    exact, seen = [], set()
    if hits:
        for r in records():
            h = grade_of_record(r)
            if h and id(h[0]) in ids and ok(r):
                exact.append(r)
                seen.add(r["id"])
    exact += [r for r in search(text) if r["id"] not in seen]
    if sn_only:
        exact = [r for r in exact if sn_options(r)]
    similar = []
    if hits:
        q = grades.profile(hits[0][0])
        done = {r["id"] for r in exact}
        for r in records():
            if r["id"] in done or (sn_only and not sn_options(r)) or not ok(r):
                continue
            p = record_profile(r)
            if p is None:
                continue
            d = grades.distance(q, p)
            if d <= max_dist:
                similar.append((r, d, grades.difference(q, p)))
        similar.sort(key=lambda x: x[1])
        similar = similar[:n_similar]
    return hits, exact, similar


def _ref(rec: dict) -> str:
    refs = rec.get("references") or []
    return refs[0].split(":")[0].strip() if refs else ""


def _slope(rec: dict, limit: float, nc: float) -> tuple[float, str]:
    """Missing slope: a line from 0.9 Rm at 10³ to the fatigue limit (Shigley), otherwise k = 10."""
    rm = value(rec, STATIC, "Ultimate tensile strength")
    if rm and 0.9 * rm > limit and nc > 1e3:
        return math.log10(nc / 1e3) / math.log10(0.9 * rm / limit), "k z 0,9 Rm při 10³"
    return 10.0, "k = 10 (odhad)"


def sn_options(rec: dict) -> list[tuple[str, SNCurve]]:
    """Possible S-N curves from the record, the first = recommended. Values without the Marin factors."""
    out = []
    head = f"PragTic #{rec['id']} {title(rec)}"
    ref = _ref(rec)
    tail = f" [{ref}]" if ref else ""
    sf, b = value(rec, PUSH_PULL, "Fatigue strength coefficient"), value(rec, PUSH_PULL, "Fatigue strength exponent")
    basq = sf is not None and b is not None and sf > 0 and b < 0
    limit = value(rec, PUSH_PULL, "Fatigue limit")
    if limit and limit > 0:
        k = value(rec, PUSH_PULL, "Exponent of S-N curve")
        nc = value(rec, PUSH_PULL, "Number of cycles at fatigue limit")
        how = []
        nc_b = 0.5 * (limit / sf) ** (1.0 / b) if basq else 0.0
        if 1e4 <= nc_b <= 1e10 and (k is None or nc is None):   # the limit lies reasonably on the Basquin line
            if k is None:
                k, how = -1.0 / b, how + ["k = −1/b"]
            if nc is None:
                nc, how = nc_b, how + ["NC z Basquina"]
        if nc is None:
            nc, how = 1e7, how + ["NC = 10⁷ (odhad)"]
        if k is None:
            k, txt = _slope(rec, limit, nc)
            how.append(txt)
        out.append(("Fatigue limit in tension-compression" + (f" ({', '.join(how)})" if how else ""),
                    SNCurve(limit, nc, k, 0.0, f"{head}: tension-compression, fatigue limit {limit:g} MPa"
                            + (f" ({', '.join(how)})" if how else "") + tail)))
    if basq:
        k = -1.0 / b
        sd = sf * (2 * BASQUIN_ND) ** b
        out.append((f"Basquin from tension-compression (σf' = {sf:g} MPa, b = {b:g}), no endurance limit",
                    SNCurve(sd, BASQUIN_ND, k, 2 * k - 1,
                            f"{head}: Basquin σf' = {sf:g} MPa, b = {b:g}; pod 10⁶ Haibach 2k−1" + tail)))
    for mode, lbl in (("Rotating bending", "ohyb za rotace"), ("Fully reversed bending", "střídavý ohyb")):
        lim = value(rec, mode, "Fatigue limit")
        if lim and lim > 0:
            k = value(rec, mode, "Exponent of S-N curve")
            nc = value(rec, mode, "Number of cycles at fatigue limit")
            how = [] if nc else ["NC = 10⁷ (odhad)"]
            nc = nc or 1e7
            if k is None:
                k, txt = _slope(rec, lim, nc)
                how.append(txt)
            out.append((f"Endurance limit – {lbl} (higher than tension-compression, Se' per Shigley)",
                        SNCurve(lim, nc, k, 0.0, f"{head}: {lbl}, fatigue limit {lim:g} MPa"
                                + (f" ({', '.join(how)})" if how else "") + tail)))
    return [(lbl, sn) for lbl, sn in out          # inconsistent data of the record → leave the variant out
            if math.isfinite(sn.SD) and sn.SD > 0 and math.isfinite(sn.ND) and sn.ND >= 1 and 0 < sn.k < 1e3]


def summary(rec: dict) -> dict:
    """Short summary for the list: Rm, Re, E and what is available for the S-N."""
    avail = []
    if value(rec, PUSH_PULL, "Fatigue limit"):
        avail.append("σC tah-tlak")
    if value(rec, PUSH_PULL, "Fatigue strength coefficient"):
        avail.append("Basquin")
    if value(rec, PUSH_PULL, "Fatigue ductility coefficient"):
        avail.append("ε-N")
    if value(rec, "Rotating bending", "Fatigue limit") or value(rec, "Fully reversed bending", "Fatigue limit"):
        avail.append("σC ohyb")
    if value(rec, "Fully reversed torsion", "Fatigue limit"):
        avail.append("τC krut")
    return {"Rm": value(rec, STATIC, "Ultimate tensile strength"), "Re": value(rec, STATIC, "Tensile yield stress"),
            "E": value(rec, STATIC, "Tensile elasticity modulus"), "available": ", ".join(avail)}


def _fmt(v) -> str:
    if isinstance(v, float):
        from fea_core.i18n import decimal
        return decimal(f"{v:.6g}")
    return str(v)


def details_html(rec: dict) -> str:
    """Whole record (all quantities with the sources) as HTML for the GUI / report."""
    from html import escape
    h = [f"<h3>{escape(title(rec))}</h3>"]
    hit = grade_of_record(rec)
    if hit:
        g, approx = hit
        h.append(f"<p><i>Grade converter{(' (přibližně)' if approx else '')}:</i> <b>{escape(g.key)}</b> ({escape(g.family_label)}) – {g.summary(html=True)}</p>")
    h.append("<table>")
    for lbl, key in (("Alternativní značka", "alt_mark"), ("Norma", "standard"), ("Skupina", "group"),
                     ("Zpracování", "treatment"), ("Poznámka", "comment")):
        if rec.get(key):
            h.append(f"<tr><td><i>{lbl}</i></td><td>{escape(rec[key])}</td></tr>")
    if rec.get("chemistry"):
        chem = ", ".join(f"{k} {_fmt(v)}" for k, v in rec["chemistry"].items())
        h.append(f"<tr><td><i>Chemical composition [%]</i></td><td>{escape(chem)}</td></tr>")
    h.append("</table>")
    for mode, rows in rec["data"].items():
        h.append(f"<p><b>{escape(MODES.get(mode, mode))}</b></p>"
                 "<table border='1' cellspacing='0' cellpadding='2'>")
        for mk, row in rows.items():
            name = QUANTITIES.get(row["desc"], row["desc"])
            if mode.startswith("Repeated") and row["desc"] == "Fatigue limit":
                name = "Endurance limit (upper stress = stress range)"   # e.g. 42CrMo4: 750 MPa at Rm 1142
            unit = "" if row["unit"] in ("-", "") else row["unit"].replace("degC", "°C")
            h.append(f"<tr><td>{escape(name)}</td><td align='right'>{escape(_fmt(row['value']))}</td>"
                     f"<td>{escape(unit)}</td><td><small>{escape(mk)}</small></td></tr>")
        h.append("</table>")
    if rec.get("references"):
        h.append("<p><b>Zdroje</b></p><ol>" + "".join(f"<li><small>{escape(r)}</small></li>"
                                                     for r in rec["references"]) + "</ol>")
    h.append(f"<p><small>PragTic database, record ID {rec['id']} – {escape(CITATION)}</small></p>")
    return "".join(h)
