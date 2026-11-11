"""Rich text with names from foreign files must not make Qt load a file / UNC resource (NTLM leak) – regression of the audit
proofs `scratch/sec_audit/poc_safe_rich.py`, `poc_qt_label_installed.py`, `poc_qt_label_ctor.py`.

Only a local png in tmp_path is used; a probe document records resource requests instead of loading them. Qt offscreen
(widgets only, no OCCT view).
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtGui import QColor, QImage, QPainter, QTextDocument  # noqa: E402

from fea_core import richtext  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Doc(QTextDocument):
    def __init__(self):
        super().__init__()
        self.asked = []

    def loadResource(self, rtype, name):
        self.asked.append(name.toString())
        img = QImage(1, 1, QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        return img


def _asks(html: str) -> list:
    d = _Doc()
    d.setHtml(html)
    img = QImage(200, 100, QImage.Format.Format_ARGB32)
    p = QPainter(img)
    d.drawContents(p)
    p.end()
    return [u for u in d.asked if u and not richtext.resource_ok(u)]


def _payloads(url: str, local: str) -> dict:
    return {
        "img src": f'x <img src="{url}"> y',
        "img source": f'x <img source="{url}"> y',
        "unterminated end": f'x <img src="{url}',
        "unterminated rich": f"<span style='color:red'>x <img src=\"{url}</span>",
        "css escape": f'<table style="background-image:\\75rl({url})"><tr><td>x</td></tr></table>',
        "css escape 2": f'<table style="background-image:u\\72 l({url})"><tr><td>x</td></tr></table>',
        "css url": f'<table style="background-image:url({url})"><tr><td>x</td></tr></table>',
        "body background": f'<body background="{url}">x</body>',
        "entity": f'x <img src="file&#58;///{local}"> y',
        "unquoted": f"x <img src={url}> y",
        "upper": f'x <IMG SRC = "{url}"> y',
        "unc": 'x <img source="//host/share/a.png"> y',
        "style import": "x<style>@\\69mport '//host/a.css';</style>y",
        "link tag": 'x<link rel="stylesheet" href="//host/a.css">y',
    }


def _url(tmp_path):
    local = (tmp_path / "no_such_marker.png").as_posix()
    return "file:///" + local, local


def test_regex_layer_alone_blocks_audit_payloads(app, tmp_path):
    """fea_core layer without the Qt checker (code without a GUI)."""
    url, local = _url(tmp_path)
    old = richtext._checker
    richtext.set_checker(None)
    try:
        for label, raw in _payloads(url, local).items():
            assert _asks(richtext.safe_rich(raw)) == [], label
    finally:
        richtext.set_checker(old)


def test_qt_checker_blocks_audit_payloads(app, tmp_path):
    from fea_gui import richtext_qt
    richtext_qt.install()
    url, local = _url(tmp_path)
    for label, raw in _payloads(url, local).items():
        assert _asks(richtext.safe_rich(raw)) == [], label
        assert richtext_qt.resource_requests(richtext.safe_rich(raw)) == [], label
    # the Qt layer alone (regex bypassed): what Qt would request is found and the text becomes plain
    for label, raw in _payloads(url, local).items():
        out = richtext_qt.check(raw)
        assert _asks(out) == [], label
    assert richtext_qt.check(f'Part <img source="{url}"> A') == "Part  A"


def test_intended_formatting_is_kept(app):
    from fea_gui import richtext_qt
    richtext_qt.install()
    for ok in ("<b>bold</b> & <i>it</i>", "a<br>b", "<span style='color:#c00'>red</span>",
               "<table><tr><td style='background-color:#fff'>x</td></tr></table>", '<a href="fea:pruvodce1">g</a>',
               '<a href="mat:3">m</a>', '<img src="data:image/png;base64,AAAA">', '<img src=":/icons/a.png">',
               "<p style='margin:0'>x</p><ul><li>a</li></ul>", '<a href="https://www.pragtic.com/vmat.php">p</a>',
               "<font color='#888'>x</font>", "[x](https://example.org) text"):
        assert richtext.safe_rich(ok) == ok, ok


def test_bad_links_become_plain(app):
    from fea_gui import richtext_qt
    assert richtext_qt.check('<a href="file://host/share/x">x</a>') == "x"
    assert richtext_qt.check('<a href="search-ms:query=x">y</a>') == "y"


def _png(tmp_path):
    png = tmp_path / "marker_300x200.png"
    img = QImage(300, 200, QImage.Format.Format_RGB32)
    img.fill(0x336699)
    assert img.save(str(png))
    return "file:///" + png.as_posix()


@pytest.mark.parametrize("lang", ["en", "cs"])
def test_labels_and_message_boxes_do_not_load_images(app, tmp_path, lang):
    """poc_qt_label_installed + poc_qt_label_ctor: setter, constructor (Polish) and QMessageBox in both languages."""
    from fea_gui import i18n
    from fea_core import i18n as core
    i18n.install(app, "en")                     # wrappers once; the language of the texts is switched below
    old = core.language()
    core.set_language(lang)
    try:
        url = _png(tmp_path)
        for name in (f'<img src="{url}">', f'<img source="{url}">', f'<img src="{url}', f"<img src={url}>"):
            lab = QtWidgets.QLabel("")
            lab.setText("Selected body: " + name)
            assert lab.sizeHint().height() < 100, (lang, name)
            lab2 = QtWidgets.QLabel(f"{name}:")          # combinations_dialog: QLabel(f"{lc.name}:")
            lab2.ensurePolished()
            i18n.polish(lab2)
            assert lab2.sizeHint().height() < 100, (lang, name)
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning, "t", f"Pair '{name}' exists.")
            lbl = box.findChild(QtWidgets.QLabel, "qt_msgbox_label")
            assert lbl is None or lbl.sizeHint().height() < 100, (lang, name)
            box.setText(f"Pair '{name}' already exists.")
            assert lbl is None or lbl.sizeHint().height() < 100, (lang, name)
            lab.setToolTip(name)
            rec = lab.property("_i18n_toolTip")             # (original, text given to Qt); toolTip() returns the original
            assert rec is not None and not _asks(rec[1]), (lang, name)
        ok = QtWidgets.QLabel("<b>Part</b> A")
        assert ok.text() == "<b>Part</b> A"
    finally:
        core.set_language(old)
