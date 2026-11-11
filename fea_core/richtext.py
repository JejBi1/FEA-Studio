"""Safe handling of names from foreign files in Qt rich text (no Qt import).

Names of parts, load cases, materials, selections… come from .fea / STEP / CAD metadata files of other people. Qt treats a text as
rich (HTML) if a tag-like `<…>` appears anywhere before the first line break (`Qt::mightBeRichText`), and `<img src>` loads the file
DIRECTLY (QTextImageHandler falls back to `QImage::load`) – a name like `x<img src="//host/share/a.png">` makes Windows connect to
the server and send NTLM credentials. Two layers of defence:

1. text built by the program escapes every name with `escape()` (html.escape);
2. a backstop `safe_rich()` is applied to every text that reaches a Qt setter or constructor (`fea_gui/i18n.py`): the value of `src`
   / `source`, `background`, `href`, CSS `url()` (also written with CSS escapes, also with an unterminated quote) and Markdown images
   that is not a harmless resource is replaced; `loadable()` decides what a QTextBrowser may load;
3. in the GUI `safe_rich()` then asks Qt's own parser (`set_checker`, `fea_gui/richtext_qt.py`) which resources the text would
   request – a text that would still load anything else is shown as plain text. Without Qt only the regex layer runs.

Harmless resource = `qrc:`, `data:`, a Qt resource path `:/…` or (only where a folder is allowed) a relative path without `..`.
"""
from __future__ import annotations

import html
import logging
import re
from pathlib import Path

_log = logging.getLogger("fea.richtext")
_reported: set[str] = set()
BLOCKED: list[str] = []                     # urls blocked in this session (tests, diagnostics)

# `source` = Qt's own alias of `src` in <img>; an unterminated quote runs to the end of the text in the Qt parser ("…"? / '…'?)
_ATTR = re.compile(r"""(\b(?:src|source|background|href|poster)\s*=\s*)("[^"]*"?|'[^']*'?|[^\s>"']+)""", re.I)
_CSSURL = re.compile(r"""(\burl\(\s*)("[^"]*"?|'[^']*'?|[^)\s"']*)""", re.I)
# CSS escapes (`\75rl(` = `url(`, `@\69mport`): decoded inside style attributes before the url() / @import check
_CSSESC = re.compile(r"\\(?:([0-9a-fA-F]{1,6})[ \t\r\n\f]?|(.))", re.S)
_STYLEATTR = re.compile(r"""(\bstyle\s*=\s*)("[^"]*"?|'[^']*'?|[^\s>"']+)""", re.I)
_MDIMG = re.compile(r"(!\[[^\]\n]{0,300}\]\(\s*)(<[^>\n]{0,2000}>|[^)\s]*)")     # bounded: linear on many "![" without "]"
_CTRL = re.compile(r"[\x00-\x20\x7f  -‏    　﻿]")
_SCHEME = re.compile(r"([a-z][a-z0-9+.\-]*):")
_MAX_MD = 200_000                           # the Markdown pattern scans to the end of the line: not for huge texts
_LINK_OK = {"fea", "mat", "http", "https", "mailto"}  # own schemes + web; anything else (file, smb, search-ms:, ms-*…) is refused
# <style> blocks (CSS @import "//host/a.css" loads a stylesheet over UNC without url()) and <link> tags – never needed in labels
_STYLE = re.compile(r"<\s*style\b[^>]*>.*?(?:<\s*/\s*style\s*>|$)", re.I | re.S)
_LINKTAG = re.compile(r"<\s*link\b[^>]*>", re.I)
_IMPORT = re.compile(r"@\s*import\b", re.I)


def escape(value) -> str:
    """Name / free text → text for rich text (quotes included)."""
    return html.escape(str(value), quote=True)


def _norm(url: str) -> str:
    """URL as the Qt parser sees it: entities decoded, spaces and control characters removed (browsers also ignore a tab inside the scheme)."""
    return _CTRL.sub("", html.unescape(str(url))).lower()


def _unc_like(u: str) -> bool:
    return u.startswith(("//", "\\\\", "/\\", "\\/"))


def resource_ok(url: str) -> bool:
    """May a `src` / `background` / `url()` point there? Only qrc:, data:, Qt resource paths `:/…`; an empty value and #anchor are harmless."""
    u = _norm(url)
    if not u or u.startswith("#"):
        return True
    if _unc_like(u):
        return False
    m = _SCHEME.match(u)
    if m:
        return m.group(1) in ("qrc", "data")
    return u.startswith(":/")


def link_ok(url: str) -> bool:
    """`href`: only the own schemes `fea:`, `mat:`, http(s), mailto and relative links (no `..`); a file, UNC or drive path and any other
    scheme (`smb:`, `search-ms:`, `ms-msdt:`… – Windows URI handlers) are refused."""
    u = _norm(url)
    if not u or u.startswith("#"):
        return True
    if _unc_like(u) or u.startswith(("/", "\\")):
        return False
    m = _SCHEME.match(u)
    if m:
        return m.group(1) in _LINK_OK
    return ".." not in re.split(r"[\\/]", u)


def _note(url: str) -> None:
    BLOCKED.append(str(url)[:200])
    key = str(url)[:80]
    if key not in _reported and len(_reported) < 50:
        _reported.add(key)
        _log.warning("Blocked an unsafe resource/link in a text: %r", key)


def _fix_attr(m: re.Match) -> str:
    attr = m.group(1).split("=")[0].strip().lower()
    val = _unquote(m.group(2))
    if (link_ok(val) if attr == "href" else resource_ok(val)):
        return m.group(0)
    _note(val)
    return m.group(1) + ('"#blocked"' if attr == "href" else '""')


def _fix_css(m: re.Match) -> str:
    val = _unquote(m.group(2))
    if resource_ok(val):
        return m.group(0)
    _note(val)
    return m.group(1) + '""'


def _unquote(tok: str) -> str:
    """Attribute / url() token without its quotes (an unterminated quote has no closing one)."""
    if tok[:1] in ("'", '"'):
        return tok[1:-1] if len(tok) > 1 and tok[-1] == tok[0] else tok[1:]
    return tok


def _css_unescape(css: str) -> str:
    def one(m: re.Match) -> str:
        if m.group(1):
            try:
                return chr(int(m.group(1), 16))
            except (ValueError, OverflowError):
                return ""
        return m.group(2)
    return _CSSESC.sub(one, css)


def _fix_style(m: re.Match) -> str:
    """style="…" with CSS escapes: decoded (escapes are never needed in our labels); the quote character is kept and closed."""
    tok = m.group(2)
    q = tok[:1] if tok[:1] in ("'", '"') else ""
    val = _css_unescape(_unquote(tok))
    if q:
        val = val.replace(q, "")
    return m.group(1) + q + val + q


def _fix_md(m: re.Match) -> str:
    tok = m.group(2)
    val = tok[1:-1] if tok.startswith("<") and tok.endswith(">") else tok
    if resource_ok(val):
        return m.group(0)
    _note(val)
    return m.group(1) + "#blocked"


def safe_rich(text):
    """Text for a Qt setter with neutralised file / UNC / network resources (`<img src>`, `background`, `href`, `url()`, `<style>`
    and CSS `@import`, `<link>`, Markdown image). A text without `<` and without a Markdown link is returned unchanged (the common case, cheap)."""
    if not isinstance(text, str) or ("<" not in text and "](" not in text):
        return text
    out = text
    if "\\" in out and "style" in out.lower():
        out = _STYLEATTR.sub(_fix_style, out)
    if "style" in out.lower() or "link" in out.lower() or "@" in out:
        out = _STYLE.sub(lambda m: (_note("<style>"), "")[1], out)
        out = _LINKTAG.sub(lambda m: (_note("<link>"), "")[1], out)
        out = _IMPORT.sub("@blocked", out)                          # e.g. in a style="" attribute or after a broken <style>
    out = _ATTR.sub(_fix_attr, out)
    if "url" in out.lower():
        out = _CSSURL.sub(_fix_css, out)
    if "](" in out and len(out) <= _MAX_MD:
        out = _MDIMG.sub(_fix_md, out)
    if _checker is not None and ("<" in out or "](" in out):
        try:
            out = _checker(out)
        except Exception as e:                  # the parser check must never break a label; the regex result stays
            _log.debug("rich text check failed: %s", e)
    return out


_checker = None


def set_checker(fn) -> None:
    """Second layer after the regexes: the GUI installs a check with Qt's own HTML parser (`fea_gui/richtext_qt.py`) that sees every
    resource the text would load, whatever the spelling. `fn(text) -> text`; None removes it."""
    global _checker
    _checker = fn


def loadable(url: str, allowed_dirs=()) -> bool:
    """May a QTextBrowser load this resource? data: / qrc: / `:/` always; a relative path (no `..`, no drive) and a file inside one of
    `allowed_dirs` (the folder of the bundled help) only if such a folder is given; UNC, http(s), ftp… never."""
    u = _norm(url)
    if not u:
        return False
    if u.startswith(":/"):
        return True
    if _unc_like(u):
        return False
    m = _SCHEME.match(u)
    sch = m.group(1) if m else ""
    if sch in ("qrc", "data"):
        return True
    dirs = [Path(d) for d in allowed_dirs if d]
    if not dirs:
        return False
    raw = str(url).strip()
    if sch == "file":
        path = raw[len("file:"):]
        if path.startswith("///"):
            path = path[3:] if re.match(r"///[A-Za-z]:", path) else path[2:]
        elif path.startswith("//"):
            return False                        # file://host/share
    elif sch:
        return False                            # http, ftp, drive letter c:…
    else:
        if u.startswith(("/", "\\")) or ".." in re.split(r"[\\/]", u):
            return False
        path = raw
    try:
        from urllib.parse import unquote
        path = unquote(path)
        cand = Path(path)
        if not cand.is_absolute():
            return any(_inside((d / cand), d) for d in dirs)
        return any(_inside(cand, d) for d in dirs)
    except (OSError, ValueError):
        return False


def _inside(path: Path, folder: Path) -> bool:
    try:
        p, f = path.resolve(), folder.resolve()
    except (OSError, ValueError):
        return False
    return p == f or f in p.parents
