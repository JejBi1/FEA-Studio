"""Tree of the assembly of Fusion from the design data (Design1/MetaStream.dat + BulkStream.dat) – occurrences of the components, their positions,
visibility of the bodies and nested documents (XREF in the .f3z).

Format (found by analysis, unofficial):
- BulkStream = a series of objects [u32 length][class number as ASCII digits][u64 id][data]; the class numbers differ in each
  document. MetaStream gives for each class its GUID (stable) and the list of ids of its objects → the class of an object
  is determined by the GUID, not by the number.
- A reference to another object = byte 01 + u64 id; texts = u32 number of characters + UTF-16LE.
- Definition of a component (DEF) → containers of bodies (CONTAINER: stream "BREP.<guid>.smbh|smb" + list of the bodies in the order
  of the bodies of the stream); a body (BODY) carries the visibility flag (light bulb). Component (COMP) = list of the child occurrences
  + reference to the DEF; occurrence (OCC) = name "Component:n", reference to the DEF, matrix 4×4 (16 × f64 by rows, offset in cm)
  relative to the parent component; an occurrence of a nested document carries the GUID "neutronRole" from RedirectionsStream.dat.
Verified against the STEP export from Fusion (positions, number and volumes of the visible bodies). What cannot be read reliably
raises DesignError – the caller then uses the streams without the assembly (the earlier behaviour with a warning).
"""
from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass, field

from ._scan import one_candidates, ref_values, utf16_texts

CLS_BODY = "D3937028-C20C-4E65-B010-94AD418A5C20"
CLS_DEF = "E03784ED-5E19-4E14-B9F2-3B07017018CD"
CLS_COMP = "904F885E-D859-4755-A7EF-6EBED8732FAD"
CLS_OCC = "CE2913AA-CFE0-4F04-9102-24424ED3BCFA"
CLS_CONTAINER = "CD57BC48-50EC-47DC-975A-FB6DEA72F4DA"
# path to the occurrence (parent occurrence → component → occurrence); in the verified sample exactly at the occurrences with the light bulb
# switched off (Fusion did not export them to STEP) → the occurrence is read as hidden
CLS_HIDDEN_PATH = "2BB3BB69-B0FD-4D56-B4CD-DEBD30847BE3"

_GUID = re.compile(rb"\$\x00\x00\x00([0-9A-Fa-f-]{36})")
_HDR = re.compile(rb"(?=([\x01-\x06])\x00\x00\x00([0-9]{1,6}))")
# flag of the visibility of a body: u32 version of the record (0x0022xxxx, 0x0023xxxx…) + 01 + flag + six zeros + −1.0 (f64)
_VIS = re.compile(rb"[\x00-\xff]{2}[\x20-\x2f]\x00\x01([\x00\x01])\x00{6}\xf0\xbf", re.DOTALL)
_PLAIN = re.compile(r"\d+_\d+|[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}.*|(Prism|Material).*")
_OCC_NAME = re.compile(r"[^:\x00]{1,120}:[^:\x00]{1,60}")
_GUID_TXT = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")



MAX_VISITS = 100_000           # visited occurrences of the components in the whole design
MAX_BODIES = 20_000            # bodies (instances) in the design
MAX_IDS = 5_000_000            # object ids listed in the MetaStream

class DesignError(ValueError):
    pass


@dataclass
class Obj:
    id: int
    cls: str
    data: bytes

    def refs(self, known) -> list[int]:
        """References 01 + u64 id to known objects, each once, in the order of the data."""
        import numpy as np
        vals = ref_values(self.data)
        if not len(vals):
            return []
        _, first = np.unique(vals, return_index=True)
        return [v for v in vals[np.sort(first)].tolist() if v in known]

    def texts(self) -> list[str]:
        """Texts u32 number of characters + UTF-16LE (printable, 2–400 characters)."""
        return utf16_texts(self.data, 400)


def class_ids(meta: bytes) -> dict[str, list[int]]:
    """MetaStream: GUID of a class → ids of the objects."""
    out: dict[str, list[int]] = {}
    total = 0
    for m in _GUID.finditer(meta):
        g = m.group(1).decode().upper()
        o = m.end()
        try:
            ln = struct.unpack_from("<I", meta, o)[0]
            if ln not in (36, 37):
                continue
            p = o + 4 + ln + 4
            ml = struct.unpack_from("<I", meta, p)[0]
            if not 1 <= ml <= 40:
                continue
            p += 4 + ml
            cnt = struct.unpack_from("<I", meta, p)[0]
            if not 0 < cnt < 1_000_000 or p + 4 + 8 * cnt > len(meta):
                continue
            total += cnt
            if total > MAX_IDS:                      # records may share one id area – bound records × cnt
                raise DesignError(f"MetaStream lists more than {MAX_IDS} object ids (suspicious file)")
            out.setdefault(g, []).extend(struct.unpack_from(f"<{cnt}Q", meta, p + 4))
        except struct.error:
            continue
    return out


def objects(bulk: bytes, meta: bytes) -> dict[int, Obj]:
    """Objects of the BulkStream with the class according to MetaStream. An id may stand in the lists of several classes (parent classes) –
    the class of an object is the one whose number prevails in the headers of its objects and matches the header.
    """
    from collections import Counter
    sets = {g: set(ids) for g, ids in class_ids(meta).items()}
    if not sets:
        raise DesignError("MetaStream without classes")
    member: dict[int, list[str]] = {}
    for g, ids in sets.items():
        for i in ids:
            member.setdefault(i, []).append(g)
    heads = []
    for m in _HDR.finditer(bulk):
        n = m.group(1)[0]
        if len(m.group(2)) < n:
            continue
        t, o = m.group(2)[:n], m.start() + 4 + n               # further digits already belong to the id of the object
        if o + 8 > len(bulk):
            continue
        oid = struct.unpack_from("<Q", bulk, o)[0]
        if oid in member:
            heads.append((m.start(), oid, t, o + 8))
    votes: dict[str, Counter] = {}
    for _, oid, t, _ in heads:
        for g in member[oid]:
            votes.setdefault(g, Counter())[t] += 1
    best = {g: c.most_common(1)[0][0] for g, c in votes.items()}
    owner: dict[bytes, str] = {}
    for g, t in best.items():             # number → the class with the most votes (the parent class has its votes scattered)
        if t not in owner or votes[g][t] > votes[owner[t]][t] or (
                votes[g][t] == votes[owner[t]][t] and len(sets[g]) < len(sets[owner[t]])):
            owner[t] = g
    keep = []
    for h in heads:
        g = owner.get(h[2])
        if g is not None and h[1] in sets[g]:
            keep.append((h, g))
    keep.sort()
    out: dict[int, Obj] = {}
    for k, ((start, oid, _, body), g) in enumerate(keep):
        end = keep[k + 1][0][0] if k + 1 < len(keep) else len(bulk)
        out.setdefault(oid, Obj(oid, g, bulk[body:end]))
    return out


def _matrix(d: bytes):
    """First matrix of a rigid motion 4×4 in the data (rows; offset in cm) or None."""
    for off in one_candidates(d, 120, 128).tolist():      # m[15] = 1 (numpy prefilter, same order as a loop over bytes)
        m = struct.unpack_from("<16d", d, off)
        if abs(m[15] - 1.0) > 1e-9 or abs(m[12]) > 1e-9 or abs(m[13]) > 1e-9 or abs(m[14]) > 1e-9:
            continue
        if not all(math.isfinite(x) for x in m):
            continue
        R = (m[0:3], m[4:7], m[8:11])
        if all(abs(sum(R[i][k] * R[j][k] for k in range(3)) - (1.0 if i == j else 0.0)) < 1e-7
               for i in range(3) for j in range(3)):
            det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
                   + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
            if det > 0:
                return m
    return None


def _mul(a, b):
    """Composition of the matrices 4×4 (rows, 16 numbers): a·b."""
    if a is None:
        return b
    if b is None:
        return a
    return tuple(sum(a[4 * i + k] * b[4 * k + j] for k in range(4)) for i in range(4) for j in range(4))


@dataclass
class Instance:
    """Body of an occurrence in the position of the assembly."""
    name: str               # name of the occurrence ("KAILH_SOCKET:SW1")
    stream: str             # name of the stream of the B-rep ("BREP.<guid>.smbh")
    body: int               # order of the body in the stream (0…)
    visible: bool
    matrix: tuple | None    # 4×4 by rows, offset in cm (None = unit)
    document: str = ""


@dataclass
class DocGraph:
    objs: dict[int, Obj]
    defs: dict[int, list[tuple[str, list[int]]]] = field(default_factory=dict)   # DEF → [(stream, bodies)]
    comps: dict[int, tuple[list[int], int | None]] = field(default_factory=dict)  # COMP → (occurrences, DEF)
    occs: dict[int, dict] = field(default_factory=dict)
    visible: dict[int, bool] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)        # DEF → name of the component
    hidden_occs: set[int] = field(default_factory=set)         # occurrences with the light bulb switched off


def graph(bulk: bytes, meta: bytes) -> DocGraph:
    objs = objects(bulk, meta)
    known = set(objs)
    g = DocGraph(objs)
    by_cls: dict[str, list[Obj]] = {}
    for o in objs.values():
        by_cls.setdefault(o.cls, []).append(o)
    for o in by_cls.get(CLS_BODY, []):
        m = _VIS.search(o.data)
        g.visible[o.id] = bool(m and m.group(1) == b"\x01")
    defs = {o.id for o in by_cls.get(CLS_DEF, [])}
    if not defs:
        raise DesignError("design without component definitions")
    for o in by_cls.get(CLS_DEF, []):                  # name = the last "ordinary" text (not a GUID, id, material)
        plain = [t for t in o.texts() if len(t) <= 120 and not _PLAIN.fullmatch(t) and t.strip()
                 and all(ord(ch) < 0x2000 for ch in t)]
        if plain:
            g.names[o.id] = plain[-1].strip()
    for o in by_cls.get(CLS_CONTAINER, []):
        rs = o.refs(known)
        dfn = next((r for r in rs if r in defs), None)
        stream = None
        for r in rs:
            t = [x for x in objs[r].texts() if x.startswith("BREP.")] if objs[r].cls != CLS_BODY else []
            if t:
                stream = t[0]
                break
        if stream is None:
            stream = next((x for x in o.texts() if x.startswith("BREP.")), None)
        bodies = [r for r in rs if objs[r].cls == CLS_BODY]
        if dfn is not None and stream is not None:
            g.defs.setdefault(dfn, []).append((stream, bodies))
    occ_ids = {o.id for o in by_cls.get(CLS_OCC, [])}
    for o in by_cls.get(CLS_COMP, []):
        rs = o.refs(known)
        g.comps[o.id] = ([r for r in rs if r in occ_ids], next((r for r in rs if r in defs), None))
    for o in by_cls.get(CLS_OCC, []):
        rs = o.refs(known)
        tx = o.texts()
        names = [t for t in tx if _OCC_NAME.fullmatch(t) and "urn:" not in t]
        g.occs[o.id] = {"def": next((r for r in rs if r in defs), None), "name": names[-1] if names else None,
                        "matrix": _matrix(o.data), "guids": {t.lower() for t in tx if _GUID_TXT.fullmatch(t)}}
    for o in by_cls.get(CLS_HIDDEN_PATH, []):
        rs = [r for r in o.refs(known) if r in g.occs]
        if rs:
            g.hidden_occs.add(rs[-1])                    # the last occurrence of the path = the hidden one
    return g


def _root(g: DocGraph) -> int:
    """Root component: its definition is referenced by an occurrence that is not a child of any component (the root occurrence)."""
    child = {o for occs, _ in g.comps.values() for o in occs}
    top_defs = {o["def"] for oid, o in g.occs.items() if oid not in child and o["def"] is not None}
    cands = [c for c, (occs, dfn) in g.comps.items() if dfn in top_defs]
    if not cands:
        owned = {g.occs[o]["def"] for o in child if o in g.occs}
        cands = [c for c, (occs, dfn) in g.comps.items() if dfn not in owned and occs]
    if not cands:
        raise DesignError("root component not found")
    return max(cands, key=lambda c: len(g.comps[c][0]))


def instances(docs: dict[str, tuple[bytes, bytes]], root_doc: str, roles: dict[str, str] | None = None,
              labels: dict[str, str] | None = None, max_depth: int = 32) -> list[Instance]:
    """Bodies of the assembly. docs = {document: (BulkStream, MetaStream)}, roles = {neutronRole GUID: document},
    labels = {document: displayed name} (bodies of the root component). An occurrence without a stored name gets
    "<component>:<n>" as in Fusion (n = order among the occurrences of the same component).
    """
    labels = labels or {}
    roles = {k.lower(): v for k, v in (roles or {}).items()}
    graphs: dict[str, DocGraph] = {}

    def gr(name: str) -> DocGraph:
        if name not in graphs:
            graphs[name] = graph(*docs[name])
        return graphs[name]

    out: list[Instance] = []
    seen_docs: list[str] = []
    work = [0]                  # number of visited occurrences – a forged tree (A → 2× B → 2× A …) grows exponentially

    def tick():
        work[0] += 1
        if work[0] > MAX_VISITS or len(out) > MAX_BODIES:
            raise DesignError(f"the design has too many occurrences / bodies (> {MAX_VISITS} / {MAX_BODIES})")

    def walk_doc(doc: str, M, prefix: str, depth: int, visible: bool):
        if depth > max_depth or doc in seen_docs:
            raise DesignError("cyclic or too deep nesting of documents")
        seen_docs.append(doc)
        g = gr(doc)
        root = _root(g)
        _, rdef = g.comps[root]
        emit(g, doc, rdef, M, prefix, visible, name=labels.get(doc) or g.names.get(rdef) or doc)
        walk_comp(g, doc, root, M, prefix, depth, visible)
        seen_docs.pop()

    def def_comp(g: DocGraph, dfn) -> int | None:
        return next((c for c, (_, d) in g.comps.items() if d == dfn), None)

    def emit(g: DocGraph, doc: str, dfn, M, prefix, visible, name):
        conts = g.defs.get(dfn, [])
        if any(s.lower().endswith(".smbh") for s, _ in conts):      # model with history; .smb = a snapshot of the same bodies
            conts = [c for c in conts if c[0].lower().endswith(".smbh")]
        for stream, bodies in conts:
            for k, b in enumerate(bodies):
                out.append(Instance(prefix + name, stream, k, visible and g.visible.get(b, False), M, doc))

    def walk_comp(g: DocGraph, doc: str, comp: int, M, prefix, depth, visible):
        if depth > max_depth:
            raise DesignError("too deep nesting of components")
        seen: dict = {}
        for oid in g.comps.get(comp, ([], None))[0]:
            occ = g.occs.get(oid)
            if occ is None:
                continue
            tick()
            seen[occ["def"]] = n = seen.get(occ["def"], 0) + 1
            Mo = _mul(M, occ["matrix"])
            vis = visible and oid not in g.hidden_occs
            xdoc = next((roles[x] for x in occ["guids"] if x in roles), None)
            if xdoc is not None and xdoc in docs:
                walk_doc(xdoc, Mo, prefix, depth + 1, vis)
                continue
            if occ["def"] is None:
                continue
            emit(g, doc, occ["def"], Mo, prefix, vis, occ["name"] or f"{g.names.get(occ['def']) or 'Komponenta'}:{n}")
            sub = def_comp(g, occ["def"])
            if sub is not None and sub != comp:
                walk_comp(g, doc, sub, Mo, prefix, depth + 1, vis)

    walk_doc(root_doc, None, "", 0, True)
    return out


def redirection_roles(redir: bytes | None, path_to_doc: dict[str, str]) -> dict[str, str]:
    """RedirectionsStream.dat (JSON): neutronRole GUID → name of the nested document."""
    if not redir:
        return {}
    try:
        d = json.loads(redir.decode("utf-8", "replace"))
    except ValueError:
        return {}
    out = {}
    for ref in d.get("references") or []:
        doc = path_to_doc.get(str(ref.get("relativePath")))
        if not doc:
            continue
        for prop in ref.get("properties") or []:
            for key in ("neutronRole", "neutronData"):
                v = (prop.get(key) or {}).get("value") if isinstance(prop.get(key), dict) else None
                if isinstance(v, str):
                    out[v.lower()] = doc
    return out
