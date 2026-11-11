# -*- coding: utf-8 -*-
"""Minimal reader of OLE2 / Compound File Binary (MS-CFB) – stdlib only, read only.

The input is an UNTRUSTED file: all sector chains are checked for a length limit and cycle detection,
the sizes of the streams are limited by the size of the file.

Specification: [MS-CFB] https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-cfb/
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")
FREESECT, ENDOFCHAIN, FATSECT, DIFSECT = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFC
NOSTREAM = 0xFFFFFFFF


class CFBError(Exception):
    pass


@dataclass
class Entry:
    sid: int
    name: str
    type: int  # 1 storage, 2 stream, 5 root
    left: int
    right: int
    child: int
    clsid: bytes
    start: int
    size: int
    path: tuple = ()
    children: list = field(default_factory=list)

    @property
    def is_stream(self) -> bool:
        return self.type == 2

    @property
    def is_storage(self) -> bool:
        return self.type in (1, 5)


class CFB:
    def __init__(self, data: bytes):
        if len(data) < 512 or data[:8] != SIGNATURE:
            raise CFBError("not an OLE2/CFB file (signature D0CF11E0 missing)")
        self.data = data
        (self.minor, self.major, bom, sshift, mshift) = struct.unpack_from("<HHHHH", data, 24)
        if bom != 0xFFFE:
            raise CFBError("wrong byte order")
        if sshift not in (9, 12) or mshift != 6:
            raise CFBError(f"unexpected sector size 2^{sshift} / mini 2^{mshift}")
        self.ssize = 1 << sshift
        self.msize = 1 << mshift
        (n_dir, n_fat, first_dir, _tx, self.cutoff, first_minifat, n_minifat,
         first_difat, n_difat) = struct.unpack_from("<IIIIIIIII", data, 40)
        self.n_sectors = (len(data) - self.ssize + self.ssize - 1) // self.ssize
        # --- DIFAT -> seznam sektoru FAT
        difat = list(struct.unpack_from("<109I", data, 76))
        sec, seen = first_difat, set()
        for _ in range(n_difat):
            if sec in (ENDOFCHAIN, FREESECT) or sec in seen or sec >= self.n_sectors:
                break
            seen.add(sec)
            vals = struct.unpack_from(f"<{self.ssize // 4}I", data, self._off(sec))
            difat.extend(vals[:-1])
            sec = vals[-1]
        # each FAT sector once, and no more FAT sectors than the file has sectors to describe (a forged DIFAT listing
        # one sector many times would otherwise grow the table with the square of the file size)
        max_fat = self.n_sectors // (self.ssize // 4) + 1
        fat_secs = list(dict.fromkeys(s for s in difat if s < self.n_sectors))[:min(n_fat, max_fat)]
        fat = []
        for s in fat_secs:
            fat.extend(struct.unpack_from(f"<{self.ssize // 4}I", data, self._off(s)))
        self.fat = fat
        # --- adresar
        dir_bytes = self._read_chain(first_dir, self.fat, self.ssize, self._big_sector)
        self.entries: list[Entry] = []
        for k in range(len(dir_bytes) // 128):
            e = dir_bytes[k * 128:(k + 1) * 128]
            nlen = struct.unpack_from("<H", e, 64)[0]
            name = e[:max(0, min(nlen, 64) - 2)].decode("utf-16le", "replace")
            typ = e[66]
            left, right, child = struct.unpack_from("<III", e, 68)
            clsid = e[80:96]
            start = struct.unpack_from("<I", e, 116)[0]
            size = struct.unpack_from("<Q", e, 120)[0]
            if self.major == 3:
                size &= 0xFFFFFFFF
            self.entries.append(Entry(k, name, typ, left, right, child, clsid, start, size))
        if not self.entries or self.entries[0].type != 5:
            raise CFBError("root directory entry missing")
        root = self.entries[0]
        # --- mini FAT a mini proud (v retezu korene)
        self.minifat = []
        if n_minifat and first_minifat not in (ENDOFCHAIN, FREESECT):
            mf = self._read_chain(first_minifat, self.fat, self.ssize, self._big_sector)
            self.minifat = list(struct.unpack_from(f"<{len(mf) // 4}I", mf, 0))
        self.ministream = self._read_chain(root.start, self.fat, self.ssize, self._big_sector,
                                           limit=root.size) if root.size else b""
        # --- strom (cervenocerny strom sourozencu)
        self._visited = {0}                      # each item only once in the whole tree (forged shared subtrees)
        self._walk(root, (), 0)

    # pomocne
    def _off(self, sec: int) -> int:
        return (sec + 1) * self.ssize

    def _big_sector(self, sec: int) -> bytes:
        o = self._off(sec)
        if o + self.ssize > len(self.data) + self.ssize:  # posledni sektor smi byt zkraceny
            raise CFBError(f"sector {sec} outside the file")
        return self.data[o:o + self.ssize]

    def _mini_sector(self, sec: int) -> bytes:
        o = sec * self.msize
        if o >= len(self.ministream):
            raise CFBError(f"mini sector {sec} outside the mini stream")
        return self.ministream[o:o + self.msize]

    def _read_chain(self, start, table, ssize, getter, limit=None) -> bytes:
        out, sec, seen = [], start, set()
        while sec not in (ENDOFCHAIN, FREESECT):
            if sec in seen:
                raise CFBError(f"cycle in the sector chain at {sec}")
            if sec >= len(table):
                raise CFBError(f"sector {sec} outside the table")
            seen.add(sec)
            out.append(getter(sec))
            sec = table[sec]
            if limit is not None and len(out) * ssize >= limit:
                break
        b = b"".join(out)
        return b[:limit] if limit is not None else b

    def _walk(self, storage: Entry, path: tuple, depth: int):
        if depth > 64:
            raise CFBError("storage tree is too deep")
        stack, seen = [storage.child], self._visited
        while stack:
            sid = stack.pop()
            if sid == NOSTREAM or sid >= len(self.entries) or sid in seen:
                continue
            seen.add(sid)
            e = self.entries[sid]
            e.path = path + (e.name,)
            storage.children.append(e)
            stack.extend([e.left, e.right])
            if e.is_storage:
                self._walk(e, e.path, depth + 1)
        storage.children.sort(key=lambda x: x.name.upper())

    # verejne
    def iter_entries(self):
        def rec(st):
            for c in st.children:
                yield c
                if c.is_storage:
                    yield from rec(c)
        yield from rec(self.entries[0])

    def read(self, e: Entry) -> bytes:
        if not e.is_stream:
            raise CFBError(f"{'/'.join(e.path)} is not a stream")
        if e.size > len(self.data) * 2 + 1024:
            raise CFBError("stream size does not match the file")
        if e.size < self.cutoff:
            return self._read_chain(e.start, self.minifat, self.msize, self._mini_sector, limit=e.size)
        return self._read_chain(e.start, self.fat, self.ssize, self._big_sector, limit=e.size)

    def find(self, *path):
        for e in self.iter_entries():
            if e.path == tuple(path):
                return e
        return None
