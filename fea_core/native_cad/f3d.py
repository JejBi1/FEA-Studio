"""Autodesk Fusion .f3d / .f3z: zip (members also Zstandard) → streams of the B-rep of the ShapeManager (Breps.BlobParts).

Choice of the streams: if the document has streams .smbh (a model with history), they form the model; otherwise .smb (a snapshot). .f3z = zip
with nested .f3d (each document separately). The streams carry bodies in the coordinates of their component – positions of the occurrences,
visibility of the bodies and nested documents are assembled by `design_documents` + `f3d_design.instances` from the design data.
"""
from __future__ import annotations

import io
import struct
import zipfile
import zlib
from pathlib import Path

MAX_MEMBER = 512 * 1024 * 1024          # largest unpacked member (protection against a "zip bomb")
MAX_NESTED = 64                         # most nested .f3d in a .f3z
MAX_TOTAL = 3 << 30                     # unpacked in total for one reading of the file (nested archives, many members)
MAX_JSON = 4 << 20                      # descriptive JSON members


class F3dError(ValueError):
    pass


def _raw(zf: zipfile.ZipFile, zi: zipfile.ZipInfo) -> bytes:
    fp = zf.fp
    fp.seek(zi.header_offset)
    h = fp.read(30)
    if h[:4] != b"PK\x03\x04":
        raise F3dError("damaged archive (local header missing)")
    nlen, xlen = struct.unpack("<HH", h[26:30])
    fp.seek(zi.header_offset + 30 + nlen + xlen)
    return fp.read(zi.compress_size)


def read_member(zf: zipfile.ZipFile, zi: zipfile.ZipInfo, budget: list | None = None) -> bytes:
    """Unpacked member; `budget` = [bytes left] common for the whole reading of the file."""
    if zi.file_size > MAX_MEMBER:
        raise F3dError(f"member {zi.filename} is too large ({zi.file_size} B)")
    if budget is not None:
        budget[0] -= zi.file_size
        if budget[0] < 0:
            raise F3dError("the archive unpacks to more than 3 GB in total (suspicious file)")
    if zi.compress_type in (0, 8, 12, 14):
        return zf.read(zi)
    if zi.compress_type != 93:
        raise F3dError(f"unknown compression of member {zi.filename} ({zi.compress_type})")
    from .zstd import decompress
    data = decompress(_raw(zf, zi), zi.file_size)
    if zlib.crc32(data) & 0xFFFFFFFF != zi.CRC:
        raise F3dError(f"damaged member {zi.filename} (CRC)")
    return data


def _small(zf: zipfile.ZipFile, name: str) -> bytes:
    """Small descriptive member – the size is checked BEFORE unpacking (KeyError if missing)."""
    zi = zf.getinfo(name)
    if zi.file_size > MAX_JSON:
        raise ValueError(f"{name} is too large")
    return read_member(zf, zi)


def _display_names(zf: zipfile.ZipFile) -> dict[str, str]:
    """.f3z: names of the nested documents from DesignDescription.json (relativePath → displayName)."""
    import json
    try:
        d = json.loads(_small(zf, "DesignDescription.json").decode("utf-8", "replace"))
        objs = d["designDescription"]["designGraphs"][0]["designObjects"]
        return {str(o["relativePath"]): str(o.get("displayName") or o.get("friendlyName"))[:120]
                for o in objs if o.get("relativePath")}
    except (KeyError, IndexError, TypeError, ValueError):
        return {}


def documents(path: str | Path) -> list[tuple[str, dict[str, bytes]]]:
    """[(name of the document, {member: data of the B-rep stream})] for each .f3d (also nested in a .f3z)."""
    out: list[tuple[str, dict[str, bytes]]] = []
    names: dict[str, str] = {}
    budget = [MAX_TOTAL]

    def walk(zf: zipfile.ZipFile, label: str, depth: int):
        streams = {}
        for zi in zf.infolist():
            if zi.filename.endswith("/"):
                continue
            if zi.filename.lower().endswith(".f3d"):
                if depth > 0 or len(out) >= MAX_NESTED:
                    continue
                walk(zipfile.ZipFile(io.BytesIO(read_member(zf, zi, budget))),
                     names.get(zi.filename) or Path(zi.filename).stem, depth + 1)
            elif "Breps.BlobParts/" in zi.filename and zi.filename.lower().endswith((".smb", ".smbh")):
                streams[zi.filename] = read_member(zf, zi, budget)
        if streams:
            out.append((label, streams))

    try:
        with zipfile.ZipFile(path) as zf:
            names.update(_display_names(zf))
            walk(zf, Path(path).stem, 0)
    except zipfile.BadZipFile as e:
        raise F3dError(f"the file is not a Fusion archive (.f3d / .f3z): {e}") from None
    return out


def design_documents(path: str | Path) -> tuple[dict[str, dict], str | None]:
    """Documents with the design data: ({name: {"label", "streams", "bulk", "meta", "redir"}}, root document).

    Name = path of the nested .f3d in the .f3z (relativePath), for a standalone .f3d the file name. Without the design data → F3dError.
    """
    import json
    docs: dict[str, dict] = {}
    budget = [MAX_TOTAL]

    def load(zf: zipfile.ZipFile, key: str, label: str):
        d = {"label": label, "streams": {}, "bulk": None, "meta": None, "redir": None}
        for zi in zf.infolist():
            fn = zi.filename
            if fn.endswith("/"):
                continue
            if "Breps.BlobParts/" in fn and fn.lower().endswith((".smb", ".smbh")):
                d["streams"][fn.rsplit("/", 1)[-1]] = read_member(zf, zi, budget)
            elif fn.endswith("/Design1/BulkStream.dat"):
                d["bulk"] = read_member(zf, zi, budget)
            elif fn.endswith("/Design1/MetaStream.dat"):
                d["meta"] = read_member(zf, zi, budget)
            elif fn == "RedirectionsStream.dat":
                d["redir"] = read_member(zf, zi, budget)
        if d["bulk"] is None or d["meta"] is None:
            raise F3dError(f"document {label} has no design data (Design1)")
        docs[key] = d

    root = None
    try:
        with zipfile.ZipFile(path) as zf:
            names = _display_names(zf)
            inner = [zi for zi in zf.infolist() if zi.filename.lower().endswith(".f3d")]
            if inner:
                if len(inner) > MAX_NESTED:
                    raise F3dError(f"too many nested documents ({len(inner)})")
                for zi in inner:
                    load(zipfile.ZipFile(io.BytesIO(read_member(zf, zi, budget))), zi.filename,
                         names.get(zi.filename) or Path(zi.filename).stem)
                try:
                    root = json.loads(_small(zf, "Manifest.json").decode("utf-8", "replace")).get("root")
                except (KeyError, ValueError, AttributeError):
                    root = None
                if root not in docs:
                    root = inner[0].filename if len(inner) == 1 else None
            else:
                load(zf, Path(path).name, Path(path).stem)
                root = Path(path).name
    except zipfile.BadZipFile as e:
        raise F3dError(f"the file is not a Fusion archive (.f3d / .f3z): {e}") from None
    return docs, root


def model_streams(streams: dict[str, bytes]) -> list[tuple[str, bytes]]:
    """Streams that form the model: .smbh (with history), otherwise .smb."""
    smbh = sorted(n for n in streams if n.lower().endswith(".smbh"))
    names = smbh or sorted(streams)
    return [(n, streams[n]) for n in names]
