"""Unpacking of Zstandard (members of the .f3d with method 93, which zipfile in Python 3.11 does not know) without a new dependency.

Always ctypes on the zstd library shipped in the wheel of cadquery-ocp (cadquery_ocp.libs, module core): the output buffer has
the size from the zip header (the caller limits it) – the modules of Python (compression.zstd, zstandard) would unpack the frame without
a limit ("zip bomb" in a foreign file).
"""
from __future__ import annotations

import ctypes
import glob
import os
import site
import sys

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        pats = ("zstd*.dll", "libzstd*.so*", "libzstd*.dylib")
        dirs = [os.path.join(p, "cadquery_ocp.libs") for p in site.getsitepackages() + [site.getusersitepackages()]]
        dirs += [os.path.join(p, "cadquery_ocp.libs") for p in sys.path if p]
        for d in dict.fromkeys(dirs):
            for pat in pats:
                for c in glob.glob(os.path.join(d, pat)):
                    try:
                        lib = ctypes.CDLL(c)
                    except OSError:
                        continue
                    lib.ZSTD_decompress.restype = ctypes.c_size_t
                    lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
                    lib.ZSTD_isError.restype = ctypes.c_uint
                    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
                    lib.ZSTD_getErrorName.restype = ctypes.c_char_p
                    lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
                    _LIB = lib
                    return _LIB
        raise RuntimeError("The Zstandard library was not found (cadquery_ocp.libs) – the file cannot be unpacked")
    return _LIB


def decompress(src: bytes, size: int) -> bytes:
    """Unpacks a zstd frame of the known size `size` (from the zip header)."""
    lib = _lib()
    dst = ctypes.create_string_buffer(max(size, 1))
    n = lib.ZSTD_decompress(dst, size, src, len(src))
    if lib.ZSTD_isError(n):
        raise RuntimeError("zstd: " + lib.ZSTD_getErrorName(n).decode())
    out = dst.raw[:n]
    if len(out) != size:
        raise RuntimeError(f"zstd: {len(out)} B instead of {size} B")
    return out
