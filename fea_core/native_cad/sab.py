"""Own reader of the binary stream of the ShapeManager (ASM BinaryFile4/8) and of ACIS SAB – stdlib only.

Framing of the records according to the tags of SAB:
  02 char, 03 short, 04 long(ref_width), 05 float, 06 double, 07/08/09 text with a length of 1/2/4 B,
  0a true, 0b false, 0c ref(ref_width), 0d ident, 0e subident, 0f/10 opening / closing of a subtype,
  11 end of the record, 12 text with a length of ref_width, 13 position 3×f64, 14 vector 3×f64, 15 enum(ref_width),
  16 2×f64, 17 int64.
Source of the table of tags: public description in the README of the crate cadmpeg-asm (Apache-2.0) + the known structure of ACIS SAB.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAX_TOKENS = 20_000_000          # tokens per stream (a tuple ~64 B → ~1.3 GB); more = the file is rejected

MAGICS = (b"ASM BinaryFile4", b"ASM BinaryFile8", b"ACIS BinaryFile")


@dataclass
class Header:
    magic: str
    ref_width: int
    version: int
    records: int
    entities: int
    flags: int
    product: str
    kernel: str
    date: str
    scale: float
    resabs: float
    resnor: float
    start: int


@dataclass
class Record:
    index: int
    name: str
    offset: int
    tokens: list = field(default_factory=list)

    def refs(self):
        return [v for t, v in self.tokens if t == "ref"]


class SabError(Exception):
    pass


def _string(d, i, tag, rw):
    if tag in (0x07, 0x0D, 0x0E):
        n = d[i]
        i += 1
    elif tag == 0x08:
        n = struct.unpack_from("<H", d, i)[0]
        i += 2
    elif tag == 0x09:
        n = struct.unpack_from("<I", d, i)[0]
        i += 4
    elif tag == 0x12:
        n = struct.unpack_from("<q" if rw == 8 else "<i", d, i)[0]
        i += rw
    else:
        raise SabError(f"not a text {tag:#x}")
    if n < 0 or i + n > len(d):
        raise SabError(f"text outside the data @{i}")
    return d[i:i + n].decode("latin-1"), i + n


def parse_header(d: bytes) -> Header:
    magic = d[:15]
    if magic not in MAGICS:
        raise SabError(f"unknown start {magic!r}")
    if magic == b"ASM BinaryFile8":
        rw = 8
        version, records, entities, flags = struct.unpack_from("<4Q", d, 15)
        i = 47
    else:
        rw = 4
        version, records, entities, flags = struct.unpack_from("<4I", d, 15)
        i = 31
    strs = []
    for _ in range(3):
        tag = d[i]
        s, i = _string(d, i + 1, tag, rw)
        strs.append(s)
    dbl = []
    for _ in range(3):
        if d[i] != 0x06:
            raise SabError(f"double expected @{i}")
        dbl.append(struct.unpack_from("<d", d, i + 1)[0])
        i += 9
    return Header(magic.decode(), rw, version, records, entities, flags, *strs, *dbl, i)


def tokenize(d: bytes, i: int, rw: int, end: int | None = None):
    """Generator (offset, kind, value)."""
    end = len(d) if end is None else end
    lfmt = "<q" if rw == 8 else "<i"
    while i < end:
        o = i
        tag = d[i]
        i += 1
        if tag == 0x02:
            v = struct.unpack_from("<b", d, i)[0]; i += 1; yield o, "char", v
        elif tag == 0x03:
            v = struct.unpack_from("<h", d, i)[0]; i += 2; yield o, "short", v
        elif tag == 0x04:
            v = struct.unpack_from(lfmt, d, i)[0]; i += rw; yield o, "long", v
        elif tag == 0x05:
            v = struct.unpack_from("<f", d, i)[0]; i += 4; yield o, "float", v
        elif tag == 0x06:
            v = struct.unpack_from("<d", d, i)[0]; i += 8; yield o, "double", v
        elif tag in (0x07, 0x08, 0x09, 0x12):
            v, i = _string(d, i, tag, rw); yield o, "str", v
        elif tag == 0x0A:
            yield o, "true", True
        elif tag == 0x0B:
            yield o, "false", False
        elif tag == 0x0C:
            v = struct.unpack_from(lfmt, d, i)[0]; i += rw; yield o, "ref", v
        elif tag == 0x0D:
            v, i = _string(d, i, tag, rw); yield o, "ident", v
        elif tag == 0x0E:
            v, i = _string(d, i, tag, rw); yield o, "subident", v
        elif tag == 0x0F:
            yield o, "open", None
        elif tag == 0x10:
            yield o, "close", None
        elif tag == 0x11:
            yield o, "end", None
        elif tag == 0x13:
            v = struct.unpack_from("<3d", d, i); i += 24; yield o, "pos", v
        elif tag == 0x14:
            v = struct.unpack_from("<3d", d, i); i += 24; yield o, "vec", v
        elif tag == 0x15:
            v = struct.unpack_from(lfmt, d, i)[0]; i += rw; yield o, "enum", v
        elif tag == 0x16:
            v = struct.unpack_from("<2d", d, i); i += 16; yield o, "vec2", v
        elif tag == 0x17:
            v = struct.unpack_from("<q", d, i)[0]; i += 8; yield o, "int64", v
        else:
            raise SabError(f"unknown tag {tag:#04x} @{o:#x}")


def frame(d: bytes, h: Header | None = None, stop_at_history: bool = False):
    """Splits the stream into records. Returns (header, records, info).
    stop_at_history: stop at the first "Begin-of-…" marker (history of the model; brep reads only the records before it) –
    the marker is kept as the last record (empty), so the record indices before it do not change."""
    h = h or parse_header(d)
    recs: list[Record] = []
    info = {"end_marker": None, "history_at": None, "error": None, "stop_offset": None}
    cur = None
    name_parts: list[str] = []
    depth = 0
    ntok = 0
    in_name = True
    try:
        for o, kind, v in tokenize(d, h.start, h.ref_width):
            if in_name:
                if kind == "subident":
                    if not name_parts:
                        start = o
                    name_parts.append(v)
                    continue
                if kind == "ident":
                    if not name_parts:
                        start = o
                    name_parts.append(v)
                    name = "-".join(name_parts)
                    name_parts = []
                    in_name = False
                    cur = Record(len(recs), name, start)
                    if name.startswith("End-of-"):
                        info.setdefault("markers", []).append((name, o))
                        recs.append(cur)
                        cur = None
                        in_name = True
                        if name == "End-of-ASM-data" or name == "End-of-ACIS-data":
                            info["end_marker"] = (name, o)
                            info["stop_offset"] = o
                            break
                        continue
                    if name.startswith("Begin-of-"):
                        info["history_at"] = o
                        info.setdefault("markers", []).append((name, o))
                        if stop_at_history:
                            recs.append(cur)
                            cur = None
                            info["stop_offset"] = o
                            break
                    continue
                raise SabError(f"record name expected, found {kind} @{o:#x}")
            # body of the record
            if kind == "open":
                depth += 1
            elif kind == "close":
                depth -= 1
            elif kind == "end" and depth == 0:
                recs.append(cur)
                cur = None
                in_name = True
                continue
            cur.tokens.append((kind, v))
            ntok += 1
            if ntok > MAX_TOKENS:
                raise SabError(f"the stream has more than {MAX_TOKENS} tokens – file rejected (memory)")
    except (SabError, struct.error, IndexError) as e:
        info["error"] = f"{type(e).__name__}: {e}"
        if cur is not None:
            recs.append(cur)
    return h, recs, info
