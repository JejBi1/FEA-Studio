"""Reading CalculiX results (.frd ASCII, .dat).

.frd is read by FIXED COLUMNS – numbers stick together (1.2E+02-4.5E+01).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ResultBlock:
    name: str                 # DISP, STRESS, ...
    step: int                 # ccx step number (1..)
    increment: int
    time: float
    components: list[str]
    node_ids: np.ndarray      # (n,) int
    values: np.ndarray        # (n, ncomp) float64
    mode: int = 0             # NUMSTP of the 100C header: mode number (*FREQUENCY, *BUCKLE), otherwise increment
    analysis: str = ""        # ANALYS text of the header (MODAL for *FREQUENCY, empty for *BUCKLE)


@dataclass
class FrdFile:
    node_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    blocks: list[ResultBlock] = field(default_factory=list)

    def get(self, name: str, step: int | None = None) -> ResultBlock:
        cand = [b for b in self.blocks if b.name == name and (step is None or b.step == step)]
        if not cand:
            raise KeyError(f"Block {name} is missing in .frd (step {step})")
        return cand[-1]  # last increment of the step

    def steps(self) -> list[int]:
        return sorted({b.step for b in self.blocks})


def _vals(line: str, start: int, n: int, width: int = 12) -> list[float]:
    return [float(line[start + i * width: start + (i + 1) * width]) for i in range(n)]


# exact powers of 10 as double (10^22 is the largest exactly representable one)
_P10 = np.array([float(f"1e{i}") for i in range(23)])


def parse_e_fields(F: np.ndarray, nd: int) -> np.ndarray:
    """Array of numbers in Fortran/C %E form: [space|-]d.<nd digits>E±dd as uint8 (M, nd + 7) → float64 (M,).

    Vectorised in numpy (no Python call per number; ~4× faster than bytes → float64). Bitwise equal to float():
    the mantissa (nd + 1 ≤ 15 digits) and 10^|k| for |k| ≤ 22 are exact in double, the only multiplication / division
    is correctly rounded (Clinger's fast path). Arrays of another form (3-digit exponent, NaN, spaces…) are
    converted as before via bytes → float64 (invalid ones raise ValueError as before).
    """
    w = nd + 7
    F = np.ascontiguousarray(F).reshape(-1, w)
    c = [F[:, j] for j in range(w)]
    z = np.uint8(48)
    d = c[1] - z                                    # uint8: a character < '0' wraps → > 9
    dmax = d.copy()                                 # largest "digit" over the positions (valid ≤ 9)
    mant = d.astype(np.int32 if nd <= 8 else np.int64)
    for j in range(3, 3 + nd):
        d = c[j] - z
        np.maximum(dmax, d, out=dmax)
        mant *= 10
        mant += d
    e1, e2 = c[5 + nd] - z, c[6 + nd] - z
    ok = dmax <= 9
    es, s0 = c[4 + nd], c[0]
    neg_e = es == 45                                # '-'
    ok &= ((e1 <= 9) & (e2 <= 9) & (c[2] == 46) & (c[3 + nd] == 69)          # '.', 'E'
           & (neg_e | (es == 43)) & ((s0 == 32) | (s0 == 45)))                 # '+'; ' ' | '-'
    ex = e1.astype(np.int32) * 10 + e2
    k = np.where(neg_e, -ex, ex) - nd
    a = np.abs(k)
    ok &= a <= 22
    a[~ok] = 0
    m = mant.astype(np.float64)
    p = _P10[a]
    out = np.where(k >= 0, m * p, m / p)
    out *= np.where(s0 == 45, -1.0, 1.0)            # multiplication by −1 keeps −0.0
    if not ok.all():
        bad = np.nonzero(~ok)[0]
        out[bad] = F[bad].copy().view(f"S{w}").ravel().astype(np.float64)
    return out


def _fixed_block(block: bytes, stride: int, nval: int, ids_cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Block of ' -1' lines of the same length → (node ids, values (n, nval)). Node numbers are converted only once
    per file: if the block has the same bytes in the number columns as the previous one (all result blocks), they are reused.
    """
    num_lines = len(block) // stride
    raw = np.frombuffer(block, dtype=np.uint8).reshape(num_lines, stride)
    idb = raw[:, 3:13]
    prev = ids_cache.get("raw")
    if prev is not None and prev.shape == idb.shape and np.array_equal(prev, idb):
        ids = ids_cache["ids"].copy()
    else:
        ids = idb.copy().view("S10").ravel().astype(np.int64)
        ids_cache["raw"], ids_cache["ids"] = idb.copy(), ids.copy()
    vals = parse_e_fields(raw[:, 13:13 + 12 * nval], 5).reshape(num_lines, nval)
    return ids, vals


# result blocks that nothing reads → skipped without conversion (ERROR = gradient estimate of the stress error,
# ccx STR(%), manual 6.12.2 – since 14. 10. 2026 read as the "stress_error" field)
SKIP_BLOCKS: set[str] = set()


def _parse_block(block: list[str], nval: int) -> tuple[np.ndarray, np.ndarray]:
    """' -1' lines of a result block → (node ids, values). Vectorised if the lines have the same length."""
    width = 13 + 12 * nval
    if nval <= 6 and block and all(len(l) == width for l in block):
        raw = np.frombuffer("".join(block).encode("latin-1"), dtype="S1").reshape(len(block), width)
        ids = raw[:, 3:13].copy().view("S10").ravel().astype(np.int64)
        vals = np.empty((len(block), nval))
        for k in range(nval):
            vals[:, k] = raw[:, 13 + 12 * k: 25 + 12 * k].copy().view("S12").ravel().astype(np.float64)
        return ids, vals
    ids, rows = [], []
    for l in block:
        if l.startswith(" -1"):
            ids.append(int(l[3:13]))
            rows.append(_vals(l, 13, min(nval, 6)))
        elif l.startswith(" -2"):  # continuation (> 6 components)
            rows[-1].extend(_vals(l, 13, min(nval - len(rows[-1]), 6)))
    return np.array(ids, dtype=np.int64), np.array(rows, dtype=np.float64).reshape(len(ids), nval)


def read_frd(path: str | Path) -> FrdFile:
    out = FrdFile()
    with open(path, "rb") as f:
        buf = f.read()

    pos = 0
    n = len(buf)
    step, incr = 0, 0
    ids_cache: dict = {}

    while pos < n:
        nl = buf.find(b"\n", pos)
        if nl == -1:
            line = buf[pos:].strip()
            pos = n
        else:
            line = buf[pos:nl].rstrip(b"\r")
            pos = nl + 1

        if not line:
            continue

        sline = line.lstrip()
        if sline.startswith(b"2C"):
            # nodes
            p_end = buf.find(b"\n -3", pos)
            if p_end != -1:
                block = buf[pos:p_end + 1]
                pos = p_end + 1
                stride = block.find(b"\n") + 1
                if stride > 0 and len(block) % stride == 0:
                    out.node_ids, out.coords = _fixed_block(block, stride, 3, ids_cache)
                else:
                    b_lines = block.decode("latin-1").splitlines()
                    ids, xyz = [], []
                    for l in b_lines:
                        if l.startswith(" -1"):
                            ids.append(int(l[3:13]))
                            xyz.append(_vals(l, 13, 3))
                    out.node_ids = np.array(ids, dtype=np.int64)
                    out.coords = np.array(xyz, dtype=np.float64)
        elif sline.startswith(b"3C"):
            # elements – not needed (we have our own mesh), skip to the end of the block
            p_end = buf.find(b"\n -3", pos)
            if p_end != -1:
                pos = p_end + 1
        elif sline.startswith(b"1PSTEP"):
            parts = sline.split()
            incr, step = int(parts[2]), int(parts[3])
        elif sline.startswith(b"100C"):
            time_val = float(line[12:24])
            try:                                       # NUMSTP (I5) a ANALYS (10A1) za ICTYPE
                numstp = int(line[58:63])
            except ValueError:
                numstp = 0
            analys = line[63:73].strip().decode("latin-1")
            try:
                numnod = int(line[24:36])
            except ValueError:
                numnod = 0
            try:
                fmt_flag = int(line.split()[-1])
            except (ValueError, IndexError):
                fmt_flag = 1
            nl4 = buf.find(b"\n", pos)
            head = buf[pos:nl4].rstrip(b"\r")
            pos = nl4 + 1
            name = head[5:13].strip().decode("latin-1")
            comps = []
            while pos < n:
                nl5 = buf.find(b"\n", pos)
                l5 = buf[pos:nl5].rstrip(b"\r")
                if not l5.lstrip().startswith(b"-5"):
                    break
                pos = nl5 + 1
                f5 = l5.split()
                # IEXIST=1 (e.g. ALL) = the component is computed by the post-processor, not written in the output
                if not (len(f5) >= 7 and f5[6].startswith(b"1")):
                    comps.append(l5[5:13].strip().decode("latin-1"))
            nval = len(comps)

            if fmt_flag == 2 and not buf[pos:pos + 4].strip().startswith(b"-") and numnod > 0:
                # Binary ccx block (*NODE OUTPUT / *ELEMENT OUTPUT, 4 B float):
                # numnod records of (id: int32, values: nval × float32)
                block_len = numnod * (1 + nval) * 4
                raw = np.frombuffer(buf[pos:pos + block_len], dtype=np.dtype([("id", "<i4"), ("vals", "<f4", (nval,))]))
                b_ids = raw["id"].astype(np.int64)
                b_vals = raw["vals"].astype(np.float64)
                pos = pos + block_len
                out.blocks.append(ResultBlock(
                    name=name, step=step, increment=incr, time=time_val,
                    components=comps[:nval], node_ids=b_ids, values=b_vals, mode=numstp, analysis=analys,
                ))
                continue

            if buf[pos:pos + 3] == b" -3":
                # empty block (CONTACT without slave nodes in contact, mortar) – otherwise the search for the block end would swallow the next block
                # the following block (spec F8, 4.3: "an empty CONTACT block breaks the reader")
                nl6 = buf.find(b"\n", pos)
                pos = n if nl6 == -1 else nl6 + 1
                out.blocks.append(ResultBlock(name=name, step=step, increment=incr, time=time_val,
                                              components=comps[:nval], node_ids=np.zeros(0, np.int64),
                                              values=np.zeros((0, nval)), mode=numstp, analysis=analys))
                continue
            p_end = buf.find(b"\n -3", pos)
            if p_end != -1 and name in SKIP_BLOCKS:
                pos = p_end + 1
            elif p_end != -1:
                block = buf[pos:p_end + 1]
                pos = p_end + 1
                width = 13 + 12 * nval
                stride = block.find(b"\n") + 1
                if nval <= 6 and stride in (width + 1, width + 2) and len(block) % stride == 0:
                    b_ids, b_vals = _fixed_block(block, stride, nval, ids_cache)
                else:
                    b_lines = block.decode("latin-1").splitlines()
                    b_ids, b_vals = _parse_block(b_lines, nval)
                out.blocks.append(ResultBlock(
                    name=name, step=step, increment=incr, time=time_val,
                    components=comps[:nval], node_ids=b_ids, values=b_vals, mode=numstp, analysis=analys,
                ))

    return out


# Fast .dat reading by bytes takes only b"\n" (b"\r\n") as line end. Line breaks that text reading
# (universal newlines, str.splitlines) takes otherwise – lone \r, \x0b, \x0c, \x1c–\x1e, \x85 – in the lines
# that are read mean the slow original path (ccx does not write them).
_ODD_BYTES = (b"\x0b", b"\x0c", b"\x1c", b"\x1d", b"\x1e", b"\x85")


def odd_line_breaks(seg: bytes) -> bool:
    """Does the section contain line breaks other than \\\\n and \\\\r\\\\n?"""
    return any(c in seg for c in _ODD_BYTES) or seg.count(b"\r") != seg.count(b"\r\n")


def line_end(buf: bytes, pos: int) -> int:
    """Index of the b"\\\\n" line starting at pos (or len(buf) for the last line without an end)."""
    e = buf.find(b"\n", pos)
    return len(buf) if e == -1 else e


def dat_pattern(tmpl: str, setname: str) -> re.Pattern:
    """Precompiled pattern of the block header in .dat for a set (reactions / forces in joints)."""
    return re.compile(tmpl.format(re.escape(setname)).encode("latin-1"), re.IGNORECASE)


def header_lines(buf: bytes, pat: re.Pattern) -> list[int]:
    """Starts of the lines matching the pattern (search over the whole file at once, not line by line)."""
    out = []
    for m in pat.finditer(buf):
        p = m.start()
        prev = buf.rfind(b"\n", 0, p)
        out.append(0 if prev == -1 else prev + 1)
    return out
