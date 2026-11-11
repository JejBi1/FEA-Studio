"""Native CAD readers: the vectorised byte scans give exactly what the original byte loops gave, and the amplification
limits hold (forged DIFAT, many masked matrices, records × cnt in the MetaStream, scanned segment size)."""
from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from fea_core.native_cad import _scan, cfb, f3d_design, iam


# ---- the original byte-by-byte implementations (reference)
def _texts_ref(d: bytes, nmax: int) -> list[str]:
    out, k, n_all = [], 0, len(d)
    while k <= n_all - 8:
        n = d[k] | d[k + 1] << 8
        if 2 <= n <= nmax and d[k + 2] == 0 and d[k + 3] == 0 and k + 4 + 2 * n <= n_all:
            try:
                t = d[k + 4:k + 4 + 2 * n].decode("utf-16-le")
            except UnicodeDecodeError:
                t = ""
            if t and t.isprintable():
                out.append(t)
                k += 4 + 2 * n
                continue
        k += 1
    return out


def _refs_ref(d: bytes, known) -> list[int]:
    out, seen = [], set()
    k = d.find(b"\x01")
    while 0 <= k <= len(d) - 9:
        v = struct.unpack_from("<Q", d, k + 1)[0]
        if v in known and v not in seen:
            seen.add(v)
            out.append(v)
        k = d.find(b"\x01", k + 1)
    return out


def _matrix_ref(d: bytes):
    for off in range(0, len(d) - 127):
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


def _masks_ref(d: bytes) -> list[int]:
    out = []
    for off in range(len(d) - 4):
        mask = struct.unpack_from("<I", d, off)[0]
        ones, zeros = mask & 0xFFFF, mask >> 16
        if not (ones >> 15) & 1 or (zeros >> 15) & 1 or (zeros >> 12) & 7 != 7 or (ones >> 12) & 7:
            continue
        out.append(off)
    return out


def _sample(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(60):
        r = rng.integers(0, 5)
        if r == 0:
            t = "Part_%d:%d" % (rng.integers(0, 99), rng.integers(1, 9))
            parts.append(struct.pack("<I", len(t)) + t.encode("utf-16-le"))
        elif r == 1:
            parts.append(b"\x01" + struct.pack("<Q", int(rng.integers(0, 40))))
        elif r == 2:
            a = rng.uniform(0, 2 * math.pi)
            c, s = math.cos(a), math.sin(a)
            parts.append(struct.pack("<16d", c, -s, 0, 1.5, s, c, 0, -2, 0, 0, 1, 3, 0, 0, 0, 1))
        elif r == 3:
            parts.append(bytes.fromhex("2184de7b"))
        else:
            parts.append(rng.integers(0, 256, int(rng.integers(1, 40)), dtype=np.uint8).tobytes())
    return b"".join(parts)


@pytest.mark.parametrize("seed", range(12))
def test_scans_equal_byte_loops(seed):
    d = _sample(seed)
    for nmax in (400, 1000):
        assert _scan.utf16_texts(d, nmax) == _texts_ref(d, nmax)
    known = set(range(0, 40, 3))
    obj = f3d_design.Obj(1, "x", d)
    assert obj.refs(known) == _refs_ref(d, known)
    assert obj.texts() == _texts_ref(d, 400)
    assert f3d_design._matrix(d) == _matrix_ref(d)
    assert _scan.mask_candidates(d).tolist() == _masks_ref(d)
    for cut in (0, 3, 7, 8, 9, 127, 128):
        assert _scan.utf16_texts(d[:cut], 400) == _texts_ref(d[:cut], 400)
        assert _scan.mask_candidates(d[:cut]).tolist() == _masks_ref(d[:cut])
        assert f3d_design.Obj(1, "x", d[:cut]).refs(known) == _refs_ref(d[:cut], known)


def test_cfb_forged_difat_does_not_amplify():
    ss, n_dif = 512, 60
    hdr = bytearray(ss)
    hdr[:8] = cfb.SIGNATURE
    struct.pack_into("<HHHHH", hdr, 24, 0x3E, 3, 0xFFFE, 9, 6)
    struct.pack_into("<IIIIIIIII", hdr, 40, 0, 0xFFFFFFFF, 0xFFFFFFFE, 0, 4096, 0xFFFFFFFE, 0, 0, n_dif)
    struct.pack_into("<109I", hdr, 76, *([0xFFFFFFFF] * 109))
    body = bytearray()
    for k in range(n_dif):
        body += struct.pack("<127I", *([0] * 127)) + struct.pack("<I", k + 1 if k + 1 < n_dif else 0xFFFFFFFE)
    data = bytes(hdr + body)
    obj = cfb.CFB.__new__(cfb.CFB)
    try:
        obj.__init__(data)
    except cfb.CFBError:
        pass
    assert len(obj.fat) <= len(data) // 4 + 128                 # at most one entry per 4 bytes of the file


def test_iam_masked_matrix_count_limited(monkeypatch):
    monkeypatch.setattr(iam, "MAX_MATRICES", 100)
    with pytest.raises(iam.IamError):
        iam.masked_matrices(bytes.fromhex("2184de7b") * 500 + b"\0" * 8)
    monkeypatch.setattr(iam, "MAX_SCAN", 1000)
    with pytest.raises(iam.IamError):
        iam.masked_matrices(b"\0" * 2000)
    with pytest.raises(iam.IamError):
        iam._texts(b"\0" * 2000)


def test_f3d_class_ids_total_limited(monkeypatch):
    monkeypatch.setattr(f3d_design, "MAX_IDS", 10_000)
    guid = b"E03784ED-5E19-4E14-B9F2-3B07017018CD"
    cnt = 2000
    rec = b"$\x00\x00\x00" + guid + struct.pack("<I", 36) + b"x" * 36 + b"\0" * 4 + struct.pack("<I", 1) + b"a" \
        + struct.pack("<I", cnt)
    with pytest.raises(f3d_design.DesignError):
        f3d_design.class_ids(rec * 20 + b"\0" * (8 * cnt))
    assert sum(len(v) for v in f3d_design.class_ids(rec * 2 + b"\0" * (8 * cnt)).values()) == 2 * cnt


def test_native_file_size_checked_before_reading(tmp_path, monkeypatch):
    import fea_core.native_cad as nc
    p = tmp_path / "x.ipt"
    p.write_bytes(b"\0" * 4096)
    monkeypatch.setattr(nc, "MAX_NATIVE_FILE", 1000)
    with pytest.raises(nc.NativeError):
        nc._sha256(p)
    monkeypatch.setattr(nc, "MAX_NATIVE_FILE", 1 << 20)
    import hashlib
    assert nc._sha256(p).hexdigest() == hashlib.sha256(b"\0" * 4096).hexdigest()
