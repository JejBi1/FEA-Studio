"""Autodesk Inventor .ipt: OLE (Compound File) → RSeStorage, a pair of streams M (meta) / B (data, zlib) → a stream of the
ShapeManager (ASM BinaryFile) of the segment PmBRepSegment = the resulting body of the part.

The assembly .iam carries only the references to the parts and their positions – it is read by `iam.py`.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

from .cfb import CFB, CFBError

MARKERS = (b"ASM BinaryFile4", b"ASM BinaryFile8", b"ACIS BinaryFile")
MAX_INFLATE = 512 * 1024 * 1024
MAX_TOTAL_INFLATE = 1 << 30            # all geometry segments of one file
MAX_FILE = 2 << 30
MAX_STREAMS = 256                       # streams of the ASM in one segment


class IptError(ValueError):
    pass


def _inflate(buf: bytes) -> bytes:
    if buf[:4] == b"\x28\xb5\x2f\xfd":                     # Zstandard (newer version)
        from .zstd import _lib                             # always ctypes: the size from the header is limited below
        import ctypes
        lib = _lib()
        lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        lib.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        n = lib.ZSTD_getFrameContentSize(buf, len(buf))
        if n > MAX_INFLATE:
            raise IptError("the unpacked segment is too large")
        from .zstd import decompress
        return decompress(buf, int(n))
    z = zlib.decompressobj()
    out = z.decompress(buf, MAX_INFLATE)
    if z.unconsumed_tail:
        raise IptError("the unpacked segment is too large")
    return out


def _meta_name(m: bytes) -> str:
    """Name of the segment from the stream M: text 'RSe Meta Stream', version, 8×u16, name (UTF-16)."""
    n = struct.unpack_from("<I", m, 0)[0]
    i = 4 + n + 2 + 16
    k = struct.unpack_from("<I", m, i)[0]
    if k > 4096:
        raise IptError("invalid segment stream")
    return m[i + 4:i + 4 + 2 * k].decode("utf-16le", "replace").rstrip("\0\n")


def brep_streams(path: str | Path) -> list[bytes]:
    """Streams of the ASM of the segment PmBRepSegment (the resulting body of the part)."""
    p = Path(path)
    if p.suffix.lower() == ".iam":
        raise IptError("An Inventor assembly (.iam) only holds references to parts – open the individual .ipt parts, or export the assembly to STEP")
    if p.stat().st_size > MAX_FILE:
        raise IptError("the file is too large")
    try:
        cfb = CFB(p.read_bytes())
    except CFBError as e:
        raise IptError(f"the file is not an Inventor part (.ipt): {e}") from None
    store = {e.name: e for e in cfb.iter_entries() if e.path[:1] == ("RSeStorage",) and len(e.path) == 2
             and e.is_stream}
    out = []
    total = 0
    for name, e in sorted(store.items()):
        if not name.startswith("M") or ("B" + name[1:]) not in store:
            continue
        try:
            seg = _meta_name(cfb.read(e))
        except (IptError, struct.error, UnicodeError):
            continue
        if seg != "PmBRepSegment":
            continue
        data = _inflate(cfb.read(store["B" + name[1:]])[18:])
        total += len(data)
        if total > MAX_TOTAL_INFLATE:
            raise IptError("the unpacked geometry segments are too large (suspicious file)")
        hits = []
        for mk in MARKERS:
            k = data.find(mk)
            while k >= 0:
                hits.append(k)
                k = data.find(mk, k + 1)
        hits = sorted(set(hits))
        if len(hits) > MAX_STREAMS:
            raise IptError(f"the geometry segment has {len(hits)} streams (suspicious file)")
        # the stream ends at the latest at the next one – the copy together ≤ the length of the segment (before it was data[k:] = quadratic)
        out += [data[k:e] for k, e in zip(hits, hits[1:] + [len(data)])]
    if not out:
        raise IptError("The .ipt file has no resulting solid (segment PmBRepSegment) – a part without a solid, or an unknown Inventor version")
    return out
