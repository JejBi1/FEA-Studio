"""QTextBrowser that does not load foreign resources.

The summaries (results, model info, details of joints…) contain names from foreign .fea / STEP files. QTextBrowser::loadResource opens a
local path, a UNC path (`//host/share/a.png` – Windows then sends NTLM credentials to the server) and files named in `<img src>`; the
default browser also follows links. `SafeTextBrowser` loads only `data:`, `qrc:` and Qt resources; the help browser may in addition
load files from its own folder (`allowed_dirs`). Everything else is replaced by an empty image and logged once
(`fea_core.richtext`). Links are not followed by the widget itself (`anchorClicked` is emitted for whoever needs them).
"""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QUrl
from PySide6.QtGui import QColor, QImage, QTextDocument
from PySide6.QtWidgets import QTextBrowser

from fea_core import richtext

_IMAGE = QTextDocument.ResourceType.ImageResource.value


def _blank() -> QImage:
    img = QImage(1, 1, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    return img


class SafeTextBrowser(QTextBrowser):
    def __init__(self, *args, allowed_dirs=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_dirs = [str(d) for d in allowed_dirs]
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)

    def loadResource(self, rtype, name: QUrl):
        url = name.toString() if isinstance(name, QUrl) else str(name)
        if richtext.loadable(url, self.allowed_dirs):
            return super().loadResource(rtype, name)
        richtext._note(url)
        return _blank() if int(rtype) == _IMAGE else QByteArray()
