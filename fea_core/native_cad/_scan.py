"""Vectorised byte scans of the native readers (the input is untrusted and may be hundreds of MB: a Python loop over every
byte would take minutes). Each function returns exactly what the original byte-by-byte loop returned."""
from __future__ import annotations

import numpy as np

_CHUNK = 8 << 20


def _u8(d) -> np.ndarray:
    return np.frombuffer(d, np.uint8)


def text_candidates(d: bytes, nmax: int) -> np.ndarray:
    """Offsets k with u16 length n = d[k] | d[k+1] << 8 in 2…nmax, d[k+2] = d[k+3] = 0 and k ≤ len − 8."""
    a = _u8(d)
    last = len(a) - 8
    if last < 0:
        return np.zeros(0, np.int64)
    out = []
    for s in range(0, last + 1, _CHUNK):
        e = min(s + _CHUNK, last + 1)                  # candidate offsets s … e−1
        n = a[s:e].astype(np.uint16) | (a[s + 1:e + 1].astype(np.uint16) << 8)
        ok = (n >= 2) & (n <= nmax) & (a[s + 2:e + 2] == 0) & (a[s + 3:e + 3] == 0)
        out.append(np.nonzero(ok)[0] + s)
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def utf16_texts(d: bytes, nmax: int) -> list[str]:
    """Texts u32 number of characters + UTF-16LE (printable, 2…nmax characters), skipping over the texts found."""
    out, k, n_all = [], 0, len(d)
    for c in text_candidates(d, nmax).tolist():
        if c < k:
            continue
        n = d[c] | d[c + 1] << 8
        if c + 4 + 2 * n > n_all:
            continue
        try:
            t = d[c + 4:c + 4 + 2 * n].decode("utf-16-le")
        except UnicodeDecodeError:
            t = ""
        if t and t.isprintable():
            out.append(t)
            k = c + 4 + 2 * n
    return out


def mask_candidates(d: bytes) -> np.ndarray:
    """Offsets 0 … len − 5 of a u32 mask of Inventor matrices: ones bit 15 set, ones bits 12–14 clear,
    zeros bit 15 clear, zeros bits 12–14 set, i.e. (d[k+1] & 0xF0) == 0x80 and (d[k+3] & 0xF0) == 0x70."""
    a = _u8(d)
    last = len(a) - 5
    if last < 0:
        return np.zeros(0, np.int64)
    out = []
    for s in range(0, last + 1, _CHUNK):
        e = min(s + _CHUNK, last + 1)
        ok = ((a[s + 1:e + 1] & 0xF0) == 0x80) & ((a[s + 3:e + 3] & 0xF0) == 0x70)
        out.append(np.nonzero(ok)[0] + s)
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def one_candidates(d: bytes, at: int, span: int, tol: float = 1e-9) -> np.ndarray:
    """Offsets off (0 ≤ off ≤ len − span) where the f64 at off + at is 1 within tol (sorted)."""
    n_all = len(d)
    out = []
    with np.errstate(invalid="ignore", over="ignore"):
        for r in range(8):
            cnt = (n_all - r) // 8
            if cnt <= 0:
                continue
            f = np.frombuffer(d, "<f8", count=cnt, offset=r)
            pos = np.nonzero(np.abs(f - 1.0) <= tol)[0] * 8 + r - at
            out.append(pos[(pos >= 0) & (pos <= n_all - span)])
    if not out:
        return np.zeros(0, np.int64)
    return np.sort(np.concatenate(out))


def ref_values(d: bytes) -> np.ndarray:
    """u64 values after each byte 01 (references "01 + u64 id"), in the order of the data."""
    a = _u8(d)
    if len(a) < 9:
        return np.zeros(0, np.uint64)
    pos = np.nonzero(a[:len(a) - 8] == 1)[0]
    if not len(pos):
        return np.zeros(0, np.uint64)
    out = np.empty(len(pos), np.uint64)
    for s in range(0, len(pos), 1 << 20):           # gather 8 bytes per reference in blocks
        p = pos[s:s + (1 << 20)]
        b = a[(p[:, None] + 1 + np.arange(8)[None, :])]
        out[s:s + len(p)] = np.ascontiguousarray(b).view("<u8").ravel()
    return out


