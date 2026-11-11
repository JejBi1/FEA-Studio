"""Metadata from CAD for a STEP file: matching parts by name and assigning materials from the library.

Metadata (JSON from the Fusion add-in, SolidWorks / Inventor macros or the REST import, all optional):
    {"source": "SolidWorks", "document": "Držák", "units": "mm",
     "parts": [{"name": "Deska-1", "names": ["Deska"], "component": "…", "occurrence": "…", "body": "…",
                "material": "AISI 304", "material_props": {...}, "color": [r, g, b] | "#rrggbb",
                "attributes": {"skupina.klíč": "hodnota"}, "layers": ["…"], "instances": ["…"],
                "hidden": true}]}
"hidden" = the body is hidden in CAD: the part is hidden in 3D and excluded from the calculation (Study.hidden_parts, ignored_parts).
Occurrence names in "instances" (SolidWorks macro) are information only – they are not matched. The file has the same name as the STEP with the suffix .json and lies next to it ("sidecar", `sidecar_path`). Parts are matched
by name (`match_parts`): exact first, then without occurrence suffixes ":1", "<1>", " (2)", "_2", and finally without the
SolidWorks occurrence suffix "-1". A material is assigned only if it matches the library (exact name or the same group
of the designation converter `grades`, not an approximate equivalent); a material state given (T6 / O / +QT) must match (O = H111) and
the CAD properties must not contradict it (`library_material`), otherwise only a warning remains.

Shared by the REST import (`fea_core.rest.imports`) and the GUI (opening a STEP). Without fastapi, the mcp SDK and PySide6.
"""
from __future__ import annotations

import json
import locale
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

STEP_SUFFIXES = (".step", ".stp")
# A study JSON (`python -m fea_core run bracket.json`) may lie next to a STEP file with the same name – it is not metadata
STUDY_KEYS = ("geometry", "load_cases", "format")
MAX_SIDECAR_BYTES = 50 * 1024 * 1024


class MetadataError(ValueError):
    """Invalid metadata from CAD (error text in Czech)."""


# ===================================================================== reading

def parse_metadata(text: str | bytes | None) -> dict | None:
    """Text JSON → metadata (None = empty); invalid JSON or structure → MetadataError."""
    if text is None or (isinstance(text, (str, bytes)) and not text.strip()):
        return None
    try:
        d = json.loads(text)
    except (ValueError, TypeError) as e:
        raise MetadataError(f"Metadata is not valid JSON: {e}") from None
    return _validate(d)


def _validate(d) -> dict:
    if not isinstance(d, dict):
        raise MetadataError("Metadata must be a JSON object {source, document, parts: [...], units}")
    parts = d.get("parts", [])
    if not isinstance(parts, list) or not all(isinstance(p, dict) for p in parts):
        raise MetadataError("Metadata: 'parts' must be a list of objects {name, color, material, attributes}")
    return d


def sidecar_path(step: str | Path) -> Path:
    """Metadata file for a STEP: the same name with the extension .json ("Držák_20261015_101500.step" → "…json")."""
    return Path(step).with_suffix(".json")


def _decode(data: bytes) -> str:
    """UTF-8 (also with BOM – written by ADODB.Stream from VBA), UTF-16 with BOM, otherwise the national code page (ANSI)."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False) or "cp1250", errors="replace")


def load_sidecar(step: str | Path, warnings: list[str] | None = None) -> dict | None:
    """Metadata from CAD next to the STEP, or None.

    None without a warning: the file does not exist, it is a study JSON or another JSON without a "parts" list.
    None with a warning in `warnings`: the file cannot be read, is not valid JSON or has a bad structure.
    """
    p = sidecar_path(step)
    if not p.is_file():
        return None
    try:
        if p.stat().st_size > MAX_SIDECAR_BYTES:
            raise MetadataError("the file is too large")
        d = json.loads(_decode(p.read_bytes()))
    except (OSError, ValueError) as e:            # MetadataError and JSONDecodeError are ValueError
        if warnings is not None:
            warnings.append(f"CAD metadata ({p.name}) cannot be read – ignored: {e}")
        return None
    if not isinstance(d, dict) or "parts" not in d or any(k in d for k in STUDY_KEYS):
        return None
    try:
        return _validate(d)
    except MetadataError as e:
        if warnings is not None:
            warnings.append(f"CAD metadata ({p.name}) ignored: {e}")
        return None


# ===================================================================== matching parts

_SUFFIXES = (re.compile(r"\s*\(\d+\)$"),          # read_step: second part with the same name "Deska (2)"
             re.compile(r":\d+$"),                 # Fusion / Inventor / Onshape occurrence "Deska:1", "<1>"
             re.compile(r"\s*<\d+>$"),
             re.compile(r"_\d+$"))                 # read_step: several bodies of one component "Deska_2"
# SolidWorks occurrence "Deska-1": only the last "-number" is removed, and only once ("Profil 40-40-4-1" → "Profil
# 40-40-4"); the key is used only after an exact match and after the other suffixes ("M8-30" is not shortened if it has its own record)
_SW_SUFFIX = re.compile(r"(?<=\S)-\d{1,4}$")


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "")).strip().casefold()


def _keys(name: str, dash: bool) -> list[str]:
    keys = [norm(name)]
    cur = str(name or "").strip()
    for _ in range(3):
        new = cur
        for rx in _SUFFIXES:
            new = rx.sub("", new).strip()
        if new == cur:
            break
        cur = new
        keys.append(norm(cur))
    if dash:
        short = _SW_SUFFIX.sub("", cur).strip()
        if short != cur:
            keys.append(norm(short))
    return [k for k in dict.fromkeys(keys) if k]


def name_keys(name: str) -> list[str]:
    """Name variants for matching: exact → without occurrence suffixes / body order → without "-1" (SolidWorks)."""
    return _keys(name, True)


def match_parts(part_names: list[str], meta_parts: list[dict]) -> tuple[dict[str, int], list[int]]:
    """STEP part → index of the metadata record. First exact names, then without occurrence suffixes, finally also without "-1";
    of several matching records the first not yet used one (otherwise the first).

    Returns (map part name → metadata index, metadata indices without a part).
    """
    cands: list[list[str]] = []
    for mp in meta_parts:
        extra = mp.get("names") or []
        extra = [extra] if isinstance(extra, str) else extra if isinstance(extra, list) else []
        names = [mp.get("name"), mp.get("component"), mp.get("occurrence"), mp.get("body"), *extra]
        cands.append([str(n) for n in names if n and isinstance(n, (str, int, float))])
    out: dict[str, int] = {}
    used: set[int] = set()
    for level in (0, 1, 2):
        index: dict[str, list[int]] = {}               # key → metadata records (ascending)
        for i, c in enumerate(cands):
            keys = {norm(n) for n in c} if level == 0 else {k for n in c for k in _keys(n, level == 2)}
            for k in keys:
                index.setdefault(k, []).append(i)
        for pn in part_names:
            if pn in out:
                continue
            for k in ([norm(pn)] if level == 0 else _keys(pn, level == 2)):
                hit = index.get(k)
                if hit:                         # more occurrences of the same part ("Deska-1", "Deska-2") → the first free one
                    out[pn] = next((i for i in hit if i not in used), hit[0])
                    used.add(out[pn])
                    break
    return out, [i for i in range(len(meta_parts)) if i not in used]


# ===================================================================== materials

def _similar(cad_name: str, mats) -> list[dict]:
    """Similar library materials by the designation converter (family, composition, Rm) – like the MCP catalogue."""
    from . import grades
    hits = grades.lookup(cad_name)
    if not hits:
        return []
    g = hits[0][0]
    q = grades.profile(g)
    out = []
    for m in mats:
        mg = grades.grade_of(m.name)
        if mg is not None and mg is not g:
            p = grades.profile(mg, rm=m.Rm)
            dist = grades.distance(q, p)
            if dist <= 6:
                out.append((dist, m.name, grades.difference(q, p)))
    return [{"name": n, "difference": why} for _, n, why in sorted(out)[:4]]


# material state in the name – the strength of the same alloy differs several times by it (EN AW-6061 O: Re 55 MPa, T6: 240):
# aluminium T6 / T651 / H111 / O / F, steel +N / +QT / +A …
_STATE_AL = re.compile(r"(?<![^\s\-_(/])(O|F|T\d{1,4}|H\d{2,3})(?![^\s\-_)/,])")
_STATE_STEEL = re.compile(r"\+\s*(QT|AR|AT|SR|N|A|C|Q|T|P)(?![A-Za-z])", re.I)
# SolidWorks library: a wrought aluminium alloy without state "6061 Alloy", "7075 Alloy" = annealed (O, 6061: Re 55 MPa);
# state T6 is named "6061-T6 (SS)"
_SW_ALLOY_O = re.compile(r"^\s*[1-8]\d{3}\s+alloy\s*$", re.I)
PROPS_TOLERANCE = (0.8, 1.25)          # Re (Rm) from CAD / library – "the same material"


def material_states(name: str) -> set[str]:
    """Material states from the name: {"T6"}, {"O"}, {"+QT"}…; T651 = T6 with stress relief (same strength)."""
    if _SW_ALLOY_O.match(name or ""):
        return {"O"}
    out = set()
    for m in _STATE_AL.finditer(name or ""):
        t = m.group(1).upper()
        out.add(t[:2] if t[0] == "T" else t)
    for m in _STATE_STEEL.finditer(name or ""):
        out.add("+" + m.group(1).upper())
    return out


def same_state(a: set[str], b: set[str]) -> bool:
    """Same material state. O and H111 are one state: EN 485-2 lists them together ("O/H111") with the same strength limits,
    so "EN AW-5083 O/H111" and "5083-O" match the library "… H111". Other pairs ("T4/T6")
    remain different – better not to assign than to assign a stronger state.
    """
    def canon(s: set[str]) -> set[str]:
        return {"O" if t == "H111" else t for t in s}
    return canon(a) == canon(b)


def _props_value(props: dict | None, *keys) -> float | None:
    for k in keys:
        try:
            v = float((props or {}).get(k))
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return None


def props_agree(props: dict | None, m) -> bool | None:
    """Does the yield strength (otherwise the tensile strength) from CAD agree with the library material? None = cannot be compared."""
    for cad, lib in ((_props_value(props, "Re_MPa", "Re"), m.Re), (_props_value(props, "Rm_MPa", "Rm"), m.Rm)):
        if cad and lib:
            return PROPS_TOLERANCE[0] <= cad / lib <= PROPS_TOLERANCE[1]
    return None


def library_material(cad_name: str, props: dict | None = None) -> tuple[str | None, dict]:
    """Library material for a CAD name: (name or None, information: match, group, hint, similar).

    The exact name (also from the user library) or the same group of the designation converter is assigned; if both names give a material
    state (T6, O, +QT…), it must be the same – a different state assigns only the properties from CAD that confirm it
    (props Re_MPa / Rm_MPa ±20 %); properties that do not agree cancel the assignment even with the same state.
    """
    from .materials import all_materials
    from . import grades
    info: dict = {"cad_material": cad_name}
    if not cad_name:
        return None, info
    mats = all_materials()
    for m in mats:
        if norm(m.name) == norm(cad_name):
            info["match"] = "name"
            return m.name, info
    try:
        hits = grades.lookup(cad_name)
    except Exception:  # noqa: BLE001 – the converter is only a help
        hits = []
    if hits:
        g, approx = hits[0]
        info["grade"] = g.key
        same = [m for m in mats if grades.grade_of(m.name) is g]
        if same and not approx:
            cad_st = material_states(cad_name)
            fit = [m for m in same
                   if not cad_st or not material_states(m.name) or same_state(material_states(m.name), cad_st)]
            other = [m for m in same if m not in fit]
            for m in fit:
                if props_agree(props, m) is not False:
                    info["match"] = f"grade converter ({g.key})"
                    return m.name, info
            for m in other:
                if props_agree(props, m):
                    info["match"] = f"grade converter ({g.key}), condition confirmed by CAD properties"
                    return m.name, info
            names = ", ".join(m.name for m in same)
            if fit:
                info["hint"] = f"CAD properties do not match {names} – check and assign manually"
            else:
                info["hint"] = (f"same grade as {names}, but a different material condition (T6 / O, +N / +QT…) – check the yield strength and assign manually")
        elif same:
            info["hint"] = f"approximate equivalent {g.key}: {', '.join((m.name for m in same))} – assign manually"
    try:
        sim = _similar(cad_name, mats)
        if sim:
            info["similar_in_library"] = sim
    except Exception:  # noqa: BLE001
        pass
    return None, info


# ===================================================================== assignment to the study

def _listing(names: list[str], n: int = 8) -> str:
    return ", ".join(names[:n]) + (f" … ({len(names)} in total)" if len(names) > n else "")


@dataclass
class MetadataResult:
    """Result of `apply_metadata`: rows per STEP part (as the answer of the REST import) and a summary."""
    source: str | None = None
    document: str | None = None
    rows: list[dict] = field(default_factory=list)        # {name, metadata, material, material_match, …}
    unmatched: list[str] = field(default_factory=list)    # metadata records without a part in the STEP
    no_metadata: list[str] = field(default_factory=list)  # STEP parts without a record in the metadata
    no_material: list[str] = field(default_factory=list)  # parts with a record but without a material from CAD
    assigned: dict[str, str] = field(default_factory=dict)            # part → library material
    not_in_library: dict[str, list[str]] = field(default_factory=dict)  # material from CAD → parts
    hints: dict[str, str] = field(default_factory=dict)               # material from CAD → hint
    hidden: list[str] = field(default_factory=list)       # parts hidden in CAD (hidden in 3D and excluded from the calculation)

    @property
    def matched(self) -> int:
        return sum(1 for r in self.rows if "metadata" in r)

    def summary(self) -> str:
        n, total, k = self.matched, len(self.rows), len(self.assigned)
        src = ", ".join(str(s) for s in (self.source, self.document) if s)
        if src:
            return f"CAD metadata ({src}): {n} of {total} parts matched, CAD material assigned to {k} parts"
        return f"CAD metadata: {n} of {total} parts matched, CAD material assigned to {k} parts"

    def messages(self, default_material: str | None = None) -> list[tuple[str, str]]:
        """Messages [(level, text)] for the GUI: summary, materials outside the library, parts without a material, unmatched ones."""
        out = [("info", self.summary())]
        dflt = default_material or "–"
        for cad, parts in self.not_in_library.items():
            hint = self.hints.get(cad)
            # whole sentence as one string (also with the hint) – translation (fea_core.i18n) recognises it as one template
            if hint:
                text = (f"CAD material “{cad}” is not in the library – parts {_listing(parts)} have the default material {dflt}, assign the correct one in the Materials step; {hint}")
            else:
                text = (f"CAD material “{cad}” is not in the library – parts {_listing(parts)} have the default material {dflt}, assign the correct one in the Materials step")
            out.append(("warning", text))
        if self.no_material:
            out.append(("warning", f"Parts without a material in CAD (default material {dflt}): {_listing(self.no_material)}"))
        if self.no_metadata:
            out.append(("warning", f"Parts without an entry in the CAD metadata (default material {dflt}): {_listing(self.no_metadata)}"))
        if self.unmatched:
            out.append(("info", f"CAD metadata entries without a part in the STEP file: {_listing(self.unmatched)}"))
        if self.hidden:
            out.append(("info", f"Bodies hidden in CAD – hidden in 3D and excluded from the analysis (Model step, Show / Analyze columns): {_listing(self.hidden)}"))
        return out


def apply_fasteners(study, parts, meta: dict | None) -> tuple[list[str], list[str]]:
    """Virtual bolts from fasteners of the assembly (metadata "fasteners", parts not loaded) → into the study.

    Returns (names of the added bolts, warnings). A bolt of the same name already in the study is not added.
    """
    recs = (meta or {}).get("fasteners") or []
    if not isinstance(recs, list) or not recs:
        return [], []
    from .joints.virtual import bolts_from_fasteners
    recs = [r for r in recs if isinstance(r, dict) and r.get("origin") and r.get("axes") and r.get("d")]
    bolts, warn = bolts_from_fasteners(parts, recs, taken={b.name for b in study.bolts})
    study.bolts += bolts
    return [b.name for b in bolts], warn


def assign_library_material(study, part: str, lib_name: str) -> None:
    """Assigns the library material to the part (a material of the same name in the study is kept – it may have been modified)."""
    if lib_name not in study.materials:
        from .materials import by_name
        study.materials[lib_name] = by_name(lib_name)
    study.part_materials[part] = lib_name
    if study.default_material is None:
        study.default_material = lib_name


def apply_metadata(study, part_names: list[str], meta: dict | None, assign: bool = True,
                   assign_fn: Callable[[str, str], None] | None = None,
                   lookup: Callable[[str, dict | None], tuple[str | None, dict]] | None = None) -> MetadataResult:
    """Matches STEP parts with metadata records and assigns materials that are in the library.

    assign = False: only checks the match (assigns nothing). assign_fn(part, material) replaces the default assignment
    `assign_library_material` (REST: tool fea_materials_assign), lookup replaces `library_material`.
    """
    lookup = lookup or library_material
    meta = meta or {}
    res = MetadataResult(meta.get("source"), meta.get("document"))
    meta_parts = meta.get("parts") or []
    mapping, unmatched = match_parts(list(part_names), meta_parts)
    res.unmatched = [str(meta_parts[i].get("name") or meta_parts[i].get("body") or f"#{i + 1}") for i in unmatched]
    cache: dict[tuple, tuple[str | None, dict]] = {}
    for pn in part_names:
        row: dict = {"name": pn}
        res.rows.append(row)
        if pn not in mapping:
            res.no_metadata.append(pn)
            continue
        mp = meta_parts[mapping[pn]]
        row["metadata"] = mp.get("name")
        if mp.get("hidden") is True:
            res.hidden.append(pn)
            row["hidden"] = True
            if assign:
                for key in ("hidden_parts", "ignored_parts"):
                    cur = getattr(study, key, None)
                    if isinstance(cur, list) and pn not in cur:
                        cur.append(pn)
        for k in ("color", "attributes", "layers"):
            if mp.get(k):
                row[k] = mp[k]
        cad_mat = str(mp.get("material") or "").strip()
        if not cad_mat:
            res.no_material.append(pn)
            continue
        props = mp.get("material_props") if isinstance(mp.get("material_props"), dict) else None
        key = (cad_mat, json.dumps(props, sort_keys=True, default=str))
        if key not in cache:
            cache[key] = lookup(cad_mat, props)
        lib, info = cache[key]
        row["material_match"] = dict(info)
        if lib and assign:
            (assign_fn or (lambda p, m: assign_library_material(study, p, m)))(pn, lib)
            row["material"] = lib
            res.assigned[pn] = lib
        elif not lib:
            res.not_in_library.setdefault(cad_mat, []).append(pn)
            hint = info.get("hint") or ""
            if not hint and info.get("similar_in_library"):
                hint = "similar in the library: " + ", ".join(s["name"] for s in info["similar_in_library"])
            if hint:
                res.hints[cad_mat] = hint
        if mp.get("material_props"):
            row["cad_material_props"] = mp["material_props"]
    return res
