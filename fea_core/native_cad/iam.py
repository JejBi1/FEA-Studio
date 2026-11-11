"""Autodesk Inventor .iam: occurrences of parts and subassemblies with the positions → list of (name of the occurrence, file .ipt, matrix).

Format (found by analysis, unofficial):
- Table of the referenced files = texts (u32 number of characters + UTF-16LE) with the extension .ipt / .iam in the stream UFRxDoc; the first
  record is the assembly itself (its original path – according to it the references are converted to relative paths).
- Occurrences = names "<file>:<n>" in the segment AmBrowserSegment in the order of the browser (mates / flush / rigid… drop out,
  because their name matches no file).
- Positions = matrices in the segment AmRxSegment in the order of the occurrences: u32 mask (lower 16 b = elements equal to 1, upper 16 b =
  elements equal to 0, both = −1) + the remaining elements of 4×4 by rows as f64; offset in cm, relative to the parent assembly.
- Checks: number of occurrences = number of matrices; world positions of the parts of the subassemblies must lie in AmDcSegment of the assembly;
  positions of the parts must match the graphics stored by Inventor (CacheGraphics/OGSCache, f32) – a mismatch = a warning.
Verified on an assembly with 21 occurrences and a subassembly (3 parts): the positions of the subassembly = AmDcSegment exactly, the parts do not interpenetrate.
"""
from __future__ import annotations

import bz2
import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from .cfb import CFB, CFBError
from ._scan import mask_candidates, utf16_texts
from .ipt import MAX_FILE, IptError, _inflate, _meta_name

MAX_DEPTH = 8
MAX_INSTANCES = 5000
MAX_VISITS = 50_000                 # all occurrences including the fasteners and subassemblies
MAX_SCENE = 256 << 20               # unpacked scene of the graphics
MAX_SCAN = 64 << 20                 # segment scanned byte by byte (texts, matrices) – real segments are kB to a few MB
MAX_MATRICES = 2 * MAX_VISITS       # positions in one segment
MAX_SEGMENTS = 512 << 20            # all unpacked segments of one assembly file
MAX_LISTING = 200_000               # files listed when searching for a referenced part
_REF = re.compile(r".+\.(ipt|iam)$", re.I)
_OCC = re.compile(r".+:\d+")


class IamError(ValueError):
    pass


@dataclass
class Instance:
    name: str               # name of the occurrence ("side_panel:2")
    path: Path              # file of the part .ipt
    matrix: list            # 4×4 by rows, offset in cm (world position)
    depth: int = 0


def _texts(d: bytes) -> list[str]:
    _scan_limit(d)
    return utf16_texts(d, 1000)


def _scan_limit(d: bytes) -> None:
    if len(d) > MAX_SCAN:
        raise IamError(f"an assembly segment of {len(d) >> 20} MB is too large (suspicious file)")


def masked_matrices(d: bytes) -> list[list[float]]:
    """Rigid-motion matrices written with the mask (see the description of the module) in the order in the data."""
    _scan_limit(d)
    out = []
    for off in mask_candidates(d).tolist():            # numpy prefilter of the mask bits (same offsets as a loop over bytes)
        mask = struct.unpack_from("<I", d, off)[0]
        ones, zeros = mask & 0xFFFF, mask >> 16
        if not (ones >> 15) & 1 or (zeros >> 15) & 1 or (zeros >> 12) & 7 != 7 or (ones >> 12) & 7:
            continue
        m, p = [], off + 4
        for k in range(16):
            if (ones >> k) & 1 and (zeros >> k) & 1:
                m.append(-1.0)
            elif (ones >> k) & 1:
                m.append(1.0)
            elif (zeros >> k) & 1:
                m.append(0.0)
            else:
                if p + 8 > len(d):
                    break
                m.append(struct.unpack_from("<d", d, p)[0])
                p += 8
        if len(m) != 16 or not all(math.isfinite(x) for x in m) or max(abs(x) for x in m) > 1e7:
            continue
        R = (m[0:3], m[4:7], m[8:11])
        if all(abs(sum(R[i][k] * R[j][k] for k in range(3)) - (1.0 if i == j else 0.0)) < 1e-6
               for i in range(3) for j in range(3)):
            det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
                   + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
            if det > 0:
                out.append(m)
                if len(out) > MAX_MATRICES:
                    raise IamError(f"an assembly segment holds more than {MAX_MATRICES} positions (suspicious file)")
    return out


def _mul(a, b):
    if a is None:
        return list(b)
    return [sum(a[4 * i + k] * b[4 * k + j] for k in range(4)) for i in range(4) for j in range(4)]


@dataclass
class _Doc:
    own: str
    files: dict[str, str]          # stem of the file (in lower case) → the original path
    occurrences: list[str]
    matrices: list[list[float]]
    dc_world: list[list[float]]
    scene: list[tuple[float, float, float]]


def _read(path: Path) -> _Doc:
    if path.stat().st_size > MAX_FILE:
        raise IamError(f"the file {path.name} is too large")
    try:
        cfb = CFB(path.read_bytes())
    except CFBError as e:
        raise IamError(f"the file {path.name} is not an Inventor assembly (.iam): {e}") from None
    store = {e.name: e for e in cfb.iter_entries() if e.path[:1] == ("RSeStorage",) and len(e.path) == 2
             and e.is_stream}
    segs: dict[str, bytes] = {}
    for name, e in store.items():
        if name.startswith("M") and "B" + name[1:] in store:
            try:
                seg = _meta_name(cfb.read(e))
            except (IptError, struct.error, UnicodeError):
                continue
            if seg in ("AmBrowserSegment", "AmRxSegment", "AmDcSegment"):
                segs[seg] = _inflate(cfb.read(store["B" + name[1:]])[18:])
                if sum(len(v) for v in segs.values()) > MAX_SEGMENTS:
                    raise IamError(f"assembly {path.name}: the unpacked segments are too large (suspicious file)")
    ufrx = next((cfb.read(e) for e in cfb.iter_entries() if e.is_stream and e.name == "UFRxDoc"), None)
    missing = [s for s in ("AmBrowserSegment", "AmRxSegment") if s not in segs]
    if ufrx is None or missing:
        raise IamError(f"assembly {path.name}: missing " + ", ".join(missing + ([] if ufrx is not None else ["UFRxDoc"]))
                       + " – unknown Inventor version")
    refs = [t for t in _texts(ufrx) if _REF.fullmatch(t)]
    own = next((r for r in refs if PureWindowsPath(r).name.lower() == path.name.lower()), refs[0] if refs else "")
    files: dict[str, str] = {}
    for r in refs:
        if r != own:
            files.setdefault(PureWindowsPath(r).stem.lower(), r)
    occ = [t for t in _texts(segs["AmBrowserSegment"]) if _OCC.fullmatch(t) and t.rsplit(":", 1)[0].lower() in files]
    seen, uniq = set(), []
    for t in occ:                                        # the browser may repeat the name (node + label)
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    scene = []
    for e in cfb.iter_entries():
        if e.is_stream and "DefaultScene" in e.path:
            raw = cfb.read(e)
            if raw[:3] == b"BZh" and len(raw) < 64 << 20:
                try:
                    dec = bz2.BZ2Decompressor()               # output limited – a bz2 "bomb" would unpack GB
                    scene = _scene_translations(dec.decompress(raw, max_length=MAX_SCENE))
                except (OSError, ValueError, EOFError):
                    scene = []
    return _Doc(own, files, uniq, masked_matrices(segs["AmRxSegment"]),
                masked_matrices(segs.get("AmDcSegment", b"")), scene)


def _scene_translations(d: bytes) -> list[tuple[float, float, float]]:
    """Offsets of the matrices 4×4 (f32, by columns) in the scene of the graphics stored by Inventor."""
    import numpy as np
    out = []
    for a in range(4):
        n = (len(d) - a) // 4
        f = np.frombuffer(d[a:a + 4 * n], "<f4")
        if len(f) < 16:
            continue
        w = np.lib.stride_tricks.sliding_window_view(f, 16)
        with np.errstate(invalid="ignore", over="ignore"):
            ok = (np.abs(w[:, 15] - 1) < 1e-4) & (np.abs(w[:, 3]) < 1e-4) & (np.abs(w[:, 7]) < 1e-4) & \
                 (np.abs(w[:, 11]) < 1e-4)
            for i in np.nonzero(ok)[0]:
                R = w[i, [0, 1, 2, 4, 5, 6, 8, 9, 10]].reshape(3, 3).astype(float)
                if np.allclose(R @ R.T, np.eye(3), atol=1e-3):
                    out.append(tuple(float(x) for x in w[i, 12:15]))
    return out


def _inside(p: Path, roots: list[Path]) -> bool:
    return any(p == b or b in p.parents for b in roots)


def resolve(ref: str, own: str, iam: Path, roots: list[Path]) -> Path | None:
    """File of the reference: relative as in the original location of the assembly, otherwise a unique match of the name in the folders roots.

    The result must lie in the folders roots – a reference "..\\\\..\\\\" from a foreign assembly must not load an arbitrary file from the disk.
    """
    r, o = PureWindowsPath(ref), PureWindowsPath(own)
    roots = [b.resolve() for b in roots]
    if o.name and r.drive.lower() == o.drive.lower():
        try:
            rel = os.path.relpath(str(r), str(o.parent))
        except ValueError:
            rel = None
        if rel:
            cand = (iam.parent / Path(*PureWindowsPath(rel).parts)).resolve()
            if cand.is_file() and _inside(cand, roots):
                return cand
    cand = (iam.parent / r.name).resolve()
    if cand.is_file() and _inside(cand, roots):
        return cand
    for base in roots:
        hits = _listing(base).get(r.name, [])
        if len(hits) == 1 and _inside(hits[0].resolve(), roots):
            return hits[0].resolve()
    return None


_LISTINGS: dict = {}


def _listing(base: Path) -> dict:
    """File name → paths under base (bounded walk: 2000 folders, MAX_LISTING files); cached for a short time – an assembly
    with thousands of references would otherwise walk the folder tree once per reference."""
    import time
    hit = _LISTINGS.get(base)
    if hit is not None and time.monotonic() - hit[0] < 30.0:
        return hit[1]
    out: dict = {}
    n = 0
    try:
        for depth, (dirpath, _, names) in enumerate(os.walk(base)):
            for nm in names:
                lst = out.setdefault(nm, [])
                if len(lst) < 2:                     # a name found twice is ambiguous – more copies are not needed
                    lst.append(Path(dirpath) / nm)
            n += len(names)
            if depth > 2000 or n > MAX_LISTING:
                break
    except OSError:
        pass
    if len(_LISTINGS) > 16:
        _LISTINGS.clear()
    _LISTINGS[base] = (time.monotonic(), out)
    return out


def instances(path: str | Path, report: list[str] | None = None,
              fasteners: list[dict] | None = None) -> tuple[list[Instance], list[Path]]:
    """Parts of the assembly in the position of the assembly and all files from which it is read (for the key of the cache).

    Fasteners (bolts, nuts, washers by name – `joints.virtual.classify`) are not loaded and their file need not exist; into `fasteners` a record
    for the virtual bolt is written (`joints.virtual.fastener_record`).
    """
    from ..joints.virtual import classify, fastener_record
    top = Path(path).resolve()
    roots = [top.parent, top.parent.parent]
    used: list[Path] = [top]
    out: list[Instance] = []
    missing: list[str] = []
    notes: list[str] = []
    world_checks: list[tuple[list, list]] = []
    fast: list[dict] = []
    docs: dict[Path, _Doc] = {}
    visits = [0]

    def walk(iam: Path, M, depth: int, stack: tuple):
        if depth > MAX_DEPTH or iam in stack:
            raise IamError("cyclic or too deep nesting of subassemblies")
        if iam not in docs:                              # a subassembly is read once, even if it is in the assembly many times
            docs[iam] = _read(iam)
        doc = docs[iam]
        if len(doc.occurrences) != len(doc.matrices):
            raise IamError(f"assembly {iam.name}: {len(doc.occurrences)} occurrences but {len(doc.matrices)} positions – renamed or suppressed occurrences cannot be read yet")
        if depth == 0:
            world_checks.append((doc.dc_world, doc.scene))
        for name, m in zip(doc.occurrences, doc.matrices):
            visits[0] += 1
            if visits[0] > MAX_VISITS:
                raise IamError(f"the assembly has more than {MAX_VISITS} occurrences")
            ref = doc.files[name.rsplit(":", 1)[0].lower()]
            Mw = _mul(M, m)
            fn = classify(PureWindowsPath(ref).stem, name)
            if fn is not None:                           # fastener component → only virtually (bolted joint)
                fast.append(fastener_record(fn, name, PureWindowsPath(ref).name, Mw))
                continue
            p = resolve(ref, doc.own, iam, roots)
            if p is None:
                missing.append(PureWindowsPath(ref).name)
                continue
            if p not in used:
                used.append(p)
            if p.suffix.lower() == ".iam":
                walk(p, Mw, depth + 1, stack + (iam,))
            else:
                out.append(Instance(name, p, Mw, depth))
                if len(out) > MAX_INSTANCES:
                    raise IamError(f"the assembly has more than {MAX_INSTANCES} part occurrences")

    _LISTINGS.clear()                                    # the folder listings are shared only within this one reading
    try:
        walk(top, None, 0, ())
    finally:
        _LISTINGS.clear()
    if missing:
        raise IamError("The assembly refers to files that are not next to it: " + ", ".join(sorted(set(missing))[:12])
                       + " – copy them next to the assembly (same folder structure as in Inventor)")
    if not out:
        raise IamError(f"the assembly {top.name} contains no part")
    if fasteners is not None:
        fasteners += fast
    dc, scene = world_checks[0]
    nested = [i for i in out if i.depth > 0]
    if nested and dc:
        bad = [i.name for i in nested if not any(max(abs(a - b) for a, b in zip(i.matrix, m)) < 1e-6 for m in dc)]
        if bad:
            raise IamError("positions of the subassembly parts do not match the stored position in the assembly: " + ", ".join(bad[:6]))
    if scene:
        far = [i.name for i in out if not any(max(abs(i.matrix[3] - t[0]), abs(i.matrix[7] - t[1]),
                                                   abs(i.matrix[11] - t[2])) < 2e-3 for t in scene)]
        if far:
            notes.append("Position of occurrences " + ", ".join(far[:6]) + " does not match the graphics stored by Inventor – the assembly may not have been saved after the last change; check the position of the parts")
    if report is not None:
        report.append(f"Inventor assembly: {len(out)} part occurrences from {len(used) - 1} files – positions from the assembly data (unofficial reading; suppressed and hidden occurrences are not distinguished)")
        if fast:
            kinds = {}
            for f in fast:
                kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
            report.append(f"Fasteners not loaded ({len(fast)}: bolts {kinds.get('bolt', 0) + kinds.get('fitted', 0)}, nuts {kinds.get('nut', 0)}, washers {kinds.get('washer', 0) + kinds.get('washer_large', 0)}) – the bolts are replaced virtually (Joints step, bolted joint assessment)")
        report += notes
    return out, used
