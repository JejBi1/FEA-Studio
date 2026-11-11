"""UI strings may only use characters present in Segoe UI (otherwise Windows maps CJK / emoji fallback fonts, +30 MB RAM).

Static check: reads the cmap of segoeui.ttf (skipped where the font is missing, e.g. Linux) and scans the string
literals of fea_gui and the translation catalogue. Nothing is imported or run.
"""
import ast
import json
import pathlib
import struct

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FONT = pathlib.Path(r"C:\Windows\Fonts\segoeui.ttf")
EXEMPT = {"help_web.py"}          # the help browser page (WebEngine has its own font fallback)


def _cmap(path):
    d = path.read_bytes()
    n = struct.unpack(">H", d[4:6])[0]
    co = None
    for i in range(n):
        tag, _, off, _ln = struct.unpack(">4sIII", d[12 + 16 * i:28 + 16 * i])
        if tag == b"cmap":
            co = off
    chars = set()
    for i in range(struct.unpack(">H", d[co + 2:co + 4])[0]):
        _pid, _eid, so = struct.unpack(">HHI", d[co + 4 + 8 * i:co + 12 + 8 * i])
        s = co + so
        fmt = struct.unpack(">H", d[s:s + 2])[0]
        if fmt == 4:
            sc = struct.unpack(">H", d[s + 6:s + 8])[0] // 2
            ends = struct.unpack(">%dH" % sc, d[s + 14:s + 14 + 2 * sc])
            starts = struct.unpack(">%dH" % sc, d[s + 16 + 2 * sc:s + 16 + 4 * sc])
            for a, b in zip(starts, ends):
                chars.update(range(a, b + 1))
        elif fmt == 12:
            ng = struct.unpack(">I", d[s + 12:s + 16])[0]
            for g in range(ng):
                a, b, _ = struct.unpack(">III", d[s + 16 + 12 * g:s + 28 + 12 * g])
                chars.update(range(a, b + 1))
    return chars


def _walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)
    elif isinstance(o, str):
        yield o


@pytest.mark.skipif(not FONT.exists(), reason="Segoe UI is not installed")
def test_ui_strings_use_only_segoe_ui_glyphs():
    cmap = _cmap(FONT)
    bad = {}
    for p in (ROOT / "fea_gui").rglob("*.py"):
        if p.name in EXEMPT:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        docs = {id(b.body[0].value) for b in ast.walk(tree)
                if isinstance(b, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and b.body and isinstance(b.body[0], ast.Expr)}          # docstrings are not shown
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                for ch in n.value:
                    if ord(ch) > 127 and ord(ch) not in cmap and ch not in "\n\t":
                        bad.setdefault(ch, []).append(f"{p.name}:{n.lineno}")
    cs = json.loads((ROOT / "fea_core" / "i18n" / "cs.json").read_text(encoding="utf-8"))
    for s in _walk(cs):
        for ch in s:
            if ord(ch) > 127 and ord(ch) not in cmap:
                bad.setdefault(ch, []).append("cs.json")
    assert not bad, {f"U+{ord(c):04X} {c}": w[:3] for c, w in bad.items()}
