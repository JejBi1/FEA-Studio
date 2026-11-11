"""Names from foreign .fea / STEP / CAD files in Qt rich text: no HTML injection, no loading of file / UNC / network resources
(`<img src="//host/share/a.png">` makes Windows send NTLM credentials to the server).

The first block is pure Python (fea_core.richtext); the second needs Qt (offscreen) and is skipped without PySide6.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from fea_core import richtext

EVIL_NAMES = [
    'x<img src="file://host/share/a.png">',
    "x<img src='//host/share/a.png'>",
    "x<img src=\\\\host\\share\\a.png>",
    '{0}%s<img src="file:///C:/secret.png">',
    "<table background='//host/b.png'><tr><td>x</td></tr></table>",
    "<td style=\"background-image:url('//host/c.png')\">x</td>",
    "<img src=\"fi\nle://host/a.png\">",
    "<img src=\"&#47;&#47;host/a.png\">",
    "![x](//host/a.png)",
]


# ------------------------------------------------------------------ fea_core.richtext (no Qt)
def test_escape_neutralises_markup():
    s = richtext.escape('a<img src="//h/x.png">&{0}%s')
    assert "<" not in s and ">" not in s and '"' not in s
    assert "{0}%s" in s


@pytest.mark.parametrize("evil", EVIL_NAMES)
def test_safe_rich_blocks_unsafe_resources(evil):
    out = richtext.safe_rich("Part: " + evil + " end")
    low = out.lower().replace("\n", "")
    assert "//host" not in low and "\\\\host" not in low and "file:" not in low and "&#47;" not in low
    assert "Part: " in out and out.endswith(" end")


def test_safe_rich_keeps_harmless_text_and_resources():
    for ok in ("plain text", "<b>bold</b> & <i>it</i>", "a < b and c > d", "{0}%s", '<img src="qrc:/icons/a.png">',
               '<img src="data:image/png;base64,AAAA">', '<img src=":/icons/a.png">', '<a href="mat:3">m</a>',
               '<a href="#top">t</a>', '<a href="https://example.org/x">x</a>', '<a href="fea:pruvodce1">g</a>',
               "<td style=\"background-color:#fff\">x</td>"):
        assert richtext.safe_rich(ok) == ok, ok


def test_safe_rich_blocks_file_links():
    for href in ("file:///C:/x", "file://host/share/x", "//host/x", "\\\\host\\x", "C:/x.exe", "smb://host/x"):
        out = richtext.safe_rich(f'<a href="{href}">x</a>')
        assert href not in out and "#blocked" in out, href


def test_safe_rich_blocks_css_import_and_link_tags():
    """CSS @import "<string>" loads a stylesheet over UNC without url() (QLabel / tooltip documents) – <style>, @import and <link> go."""
    for evil in ('x<style>@import "//evil/s/a.css";</style>y', r"x<STYLE type='text/css'>@\69mport '//evil/a.css'</STYLE>y",
                 'x<style>@import "//evil/a.css"', 'x<link rel="stylesheet" href="//evil/a.css">y'):
        out = richtext.safe_rich(evil)
        assert "//evil" not in out and "@import" not in out.lower(), out
    out = richtext.safe_rich("x<b style='@import \"//e/a\"'>y</b>")         # inline style: the @import keyword is disarmed
    assert "@import" not in out.lower() and "@blocked" in out, out
    assert richtext.safe_rich("<b>Bold</b> part") == "<b>Bold</b> part"


def test_link_ok_allows_only_own_and_web_schemes():
    for ok in ("fea:pruvodce1", "mat:Ocel S235", "https://www.pragtic.com/vmat.php", "mailto:x@example.com", "#a", "page.html"):
        assert richtext.link_ok(ok), ok
    for bad in ("search-ms:query=x&crumb=location:\\evil\s", "ms-msdt:/id", "ms-officecmd:x", "smb://evil/s", "file:///C:/x",
                "javascript:alert(1)", "../x.html", "//evil/s"):
        assert not richtext.link_ok(bad), bad


def test_loadable():
    assert richtext.loadable("data:image/png;base64,AAAA")
    assert richtext.loadable("qrc:/a.png") and richtext.loadable(":/a.png")
    for bad in ("//host/share/a.png", "file://host/share/a.png", "\\\\host\\share\\a.png", "http://example.org/a.png",
                "C:/Windows/win.ini", "ftp://h/a", "file:///etc/passwd", "a.png"):
        assert not richtext.loadable(bad), bad                       # no allowed folder: only data / qrc


def test_loadable_help_folder(tmp_path):
    (tmp_path / "cs").mkdir()
    (tmp_path / "cs" / "a.png").write_bytes(b"x")
    d = [str(tmp_path)]
    assert richtext.loadable("a.png", d) and richtext.loadable("cs/a.png", d)
    assert richtext.loadable((tmp_path / "cs" / "a.png").as_uri(), d)
    assert not richtext.loadable("../a.png", d)
    assert not richtext.loadable("//host/share/a.png", d)
    assert not richtext.loadable("file://host/share/a.png", d)
    assert not richtext.loadable("https://example.org/a.png", d)
    assert not richtext.loadable("C:/Windows/win.ini", d)
    assert not richtext.loadable("/etc/passwd", d)


def test_safe_rich_is_fast_on_adversarial_text():
    import time
    for bad in ("<" * 50000, "<img " * 20000, '<img src="' * 20000, "<" + "url(" * 20000, "<![" * 20000 + "](",
                '<a href="' * 20000, "<img src='" * 20000):
        t = time.perf_counter()
        richtext.safe_rich(bad)
        assert time.perf_counter() - t < 3.0, bad[:10]


# ------------------------------------------------------------------ Qt part (offscreen)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


EVIL = EVIL_NAMES[0] + "{0}%s"


def _clean(html: str) -> bool:
    """Summary built by the program: every name is escaped → no `<img` tag at all."""
    return "<img" not in html.lower()


def _neutral(text: str) -> bool:
    """Text that went through the setter wrapper: markup may stay, but no file / UNC resource."""
    low = text.lower()
    return "file:" not in low and "//host" not in low and "\\\\host" not in low


def test_mode_info_html_escapes_case_name(app):
    from fea_gui.eigen_gui import mode_info_html
    ms = SimpleNamespace(kind="buckling", values=np.array([2.0]), case=EVIL, eff_mass=None, total_mass=None)
    out = mode_info_html(ms, 0)
    assert _clean(out) and "&lt;img" in out


def test_dynamics_info_html_escapes_names(app):
    from fea_gui.dyn_results import info_html
    d = SimpleNamespace(
        K=3, model_mass=1.0, warnings=[EVIL], freq=np.ones(3), zeta=np.ones(3) * 0.02, eff_mass=np.ones((3, 3)),
        harmonic={"f": np.array([1.0]), "labels": [EVIL]}, transient={"t": np.array([0.0]), "labels": [EVIL]},
        summary=lambda: {"harmonic": {"sweep_fatigue": {EVIL: EVIL}}, "random": {"method": EVIL, "note": EVIL},
                         "shock": {"pulse": EVIL, "note": EVIL}, "transient": {"settle_s": {EVIL: 1.0}}})
    vals = np.array([1.0, 2.0])
    for sub in ("harm", "sweep", "tr", "rand", "shock", "mode"):
        out = info_html(d, sub, 0, 0, 0.0, False, vals, EVIL, EVIL, 1.0)
        assert _clean(out), (sub, out)


def test_result_summary_escapes_names(app):
    """MainWindow.fill_result_info / fill_thermal_info on a stub: names of the state, contact, part, reaction, warning."""
    gm = pytest.importorskip("fea_gui.main")

    class Capture:
        html = ""

        def setHtml(self, t):
            self.html = t

    class Stub:
        def __init__(self):
            self.res_info = Capture()
            self.results = SimpleNamespace(
                mesh=SimpleNamespace(nodes=np.zeros((3, 3)), surfaces={}), warnings=[EVIL], info={},
                thermal=[])
            self.study = SimpleNamespace(constraints=[], thermal=SimpleNamespace(reference_temperature=20.0))

        def current_result(self):
            return None

        def result_hotspots(self, *a, **k):
            return None

        def node_re(self):
            return np.array([np.nan])

        hotspot_label = staticmethod(gm.MainWindow.hotspot_label)
        _pt = staticmethod(gm.MainWindow._pt)
        _join_rows = staticmethod(gm.MainWindow._join_rows)

    st = SimpleNamespace(
        name=EVIL, partial=False, contacts={EVIL: {"Fn_N": 1.0, "p_max_MPa": 1.0}}, umag=np.array([0.0, 1.0, 2.0]),
        mises=np.array([1.0, 2.0, 3.0]), fields={},
        printing={"criterion": EVIL, "parts": {EVIL: {"min_safety": 2.0, "region": EVIL, "at": [0, 0, 0],
                                                      "stress_local": [1.0] * 6}}},
        reactions={EVIL: np.array([1.0, 2.0, 3.0])}, remote={}, equilibrium={"imbalance": 1e-9})
    s = Stub()
    gm.MainWindow.fill_result_info(s, st, 1.0, 0, np.zeros(3))
    assert _clean(s.res_info.html), s.res_info.html
    assert "&lt;img" in s.res_info.html

    th = SimpleNamespace(name=EVIL, transient=False, times=[], T=np.array([20.0, 30.0, 25.0]), flux_mag=np.ones(3),
                         heat=[{"name": EVIL, "kind": EVIL, "Q_W": 1.0}], balance={"note": EVIL})
    gm.MainWindow.fill_thermal_info(s, th, None)
    assert _clean(s.res_info.html), s.res_info.html


def test_setter_wrapper_sanitises_labels_and_browsers(app):
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QImage
    from fea_gui import i18n as gi
    from fea_gui.safe_browser import SafeTextBrowser

    class Label(QtWidgets.QLabel):
        pass

    class Browser(SafeTextBrowser):
        pass
    gi._wrap_prop(Label, "text", "setText")
    gi._wrap_text(Browser, "setHtml")
    lab = Label()
    lab.setText("Part: " + EVIL)                                  # a tag in the middle of a label is still rich text for Qt
    stored = lab.property("_i18n_text")
    assert stored is not None and _neutral(stored[1])
    lab2 = Label()
    lab2.setText("{0}%s")                                         # format characters stay literal
    assert lab2.property("_i18n_text") is None

    b = Browser()
    b.setHtml("<p>" + EVIL + "</p>")
    assert _neutral(b.toHtml())
    img = b.loadResource(2, QUrl("file://host/share/a.png"))     # direct call of the loader: blocked, replaced by an empty image
    assert isinstance(img, QImage) and img.width() == 1
    assert any("host/share" in x for x in richtext.BLOCKED)
    assert not b.openLinks()


def test_browser_help_folder_loads_only_inside(app, tmp_path):
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QImage
    from fea_gui.safe_browser import SafeTextBrowser
    img = QImage(2, 2, QImage.Format.Format_ARGB32)
    img.fill(0xFF00FF00)
    assert img.save(str(tmp_path / "ok.png"))
    b = SafeTextBrowser(allowed_dirs=[tmp_path])
    b.setSearchPaths([str(tmp_path)])
    ok = b.loadResource(2, QUrl("ok.png"))
    assert ok is not None and not (isinstance(ok, QImage) and ok.width() == 1)       # the real file (bytes / image), not the blank one
    other = b.loadResource(2, QUrl("../ok.png"))
    assert isinstance(other, QImage) and other.width() == 1
