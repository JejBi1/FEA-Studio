"""Rich-text check with Qt's own HTML / Markdown parser (second layer of `fea_core.richtext.safe_rich`).

The regex layer in `fea_core/richtext.py` has to guess how Qt reads a text (`<img source=…>`, CSS escapes `\\75rl(…)`, unterminated
quotes, entities…). Here the text is parsed by a `QTextDocument` whose `loadResource` only records the request (never opens anything);
then the document's formats are inspected (image names, background images, link targets) and the document is laid out (images are
requested at layout). A text that would load anything except `data:` / `qrc:` / Qt resources `:/…`, or that carries a link with a file /
UNC / foreign scheme, is shown as plain text instead (the tags go, the visible text stays – `<` `>` become ‹ ›).

`install()` registers the check in `fea_core.richtext` (done on import of `fea_gui.i18n`). The check runs only in the GUI thread with a
QGuiApplication; otherwise only the regex layer applies (the text is checked again when it reaches a widget setter).
"""
from __future__ import annotations

import re
from collections import OrderedDict

from PySide6.QtCore import QByteArray, QThread, QUrl
from PySide6.QtGui import QColor, QGuiApplication, QImage, QTextDocument, QTextFormat

from fea_core import richtext

_IMAGE = QTextDocument.ResourceType.ImageResource.value
_BG = QTextFormat.Property.BackgroundImageUrl
_CACHE: OrderedDict[str, str] = OrderedDict()
_CACHE_MAX = 4096
_MAX_LEN = 1_000_000                       # longer texts: the document would be expensive → plain text if it contains any tag at all
_busy = False
# A text can only reach a resource / link through one of these spellings: attribute names (src, source, background, href), CSS
# (url, image, @import), <link>, a CSS escape (backslash) or a numeric entity inside an attribute value (&#…). Tag and attribute
# names are not entity-decoded by Qt. A text without any of them is not parsed (the common formatted label: <b>, <br>, colours).
_SUSPECT = re.compile(r"src|source|background|url|href|link|image|import|@|\\|&#", re.I)


def _blank() -> QImage:
    img = QImage(1, 1, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    return img


class _Probe(QTextDocument):
    """Document that records every resource request and never loads one."""

    def __init__(self):
        super().__init__()
        self.asked: list[str] = []

    def loadResource(self, rtype, name):
        url = name.toString() if isinstance(name, QUrl) else str(name)
        self.asked.append(url)
        return _blank() if int(rtype) == _IMAGE else QByteArray()


def _bad_urls(doc: _Probe) -> list[str]:
    bad = [u for u in doc.asked if not richtext.resource_ok(u)]
    for fmt in doc.allFormats():
        if fmt.isImageFormat():
            name = fmt.toImageFormat().name()
            if name and not richtext.resource_ok(name):
                bad.append(name)
        bg = fmt.property(_BG)
        if isinstance(bg, str) and bg and not richtext.resource_ok(bg):
            bad.append(bg)
        if fmt.isCharFormat():
            cf = fmt.toCharFormat()
            if cf.isAnchor():
                href = cf.anchorHref()
                if href and not richtext.link_ok(href):
                    bad.append(href)
    return bad


def resource_requests(text: str, markdown: bool = False) -> list[str]:
    """Resources / links of `text` that are not allowed (empty list = harmless). For tests and diagnostics."""
    doc = _Probe()
    if markdown:
        doc.setMarkdown(text)
    else:
        doc.setHtml(text)
    # no layout needed: stylesheets (<link>, @import) are requested while parsing, images and background images are requested later
    # only by the names stored in the formats – those are inspected directly
    return _bad_urls(doc)


def plain(text: str, markdown: bool = False) -> str:
    """Visible text of `text` without any markup; `<` / `>` replaced so that Qt cannot read it as rich text again."""
    doc = QTextDocument()
    if markdown:
        doc.setMarkdown(text)
    else:
        doc.setHtml(text)
    s = doc.toPlainText().replace("￼", "")
    return s.replace("<", "‹").replace(">", "›")


def _usable() -> bool:
    app = QGuiApplication.instance()
    return app is not None and QThread.currentThread() is app.thread()


def check(text):
    """`text` unchanged if Qt would load nothing foreign from it, otherwise its plain visible text."""
    global _busy
    if not isinstance(text, str) or _busy or not _usable():
        return text
    hit = _CACHE.get(text)
    if hit is not None:
        _CACHE.move_to_end(text)
        return hit
    _busy = True
    try:
        if len(text) > _MAX_LEN:
            out = text.replace("<", "‹").replace(">", "›")
        elif not _SUSPECT.search(text):
            out = text                      # no word that can name a resource / link (incl. CSS escapes and numeric entities)
        else:
            bad = resource_requests(text) if "<" in text else []
            md = "](" in text
            if not bad and md:
                bad = resource_requests(text, markdown=True)
            if bad:
                for u in bad[:3]:
                    richtext._note(u)
                out = plain(text, markdown=md and "<" not in text)
            else:
                out = text
    finally:
        _busy = False
    _CACHE[text] = out
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return out


def install() -> None:
    richtext.set_checker(check)
