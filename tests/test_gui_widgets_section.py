"""F11: collapsible section `widgets.Section`, empty tables, two-level labels, TipFilter (Qt offscreen)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def no_remember(monkeypatch):
    monkeypatch.delenv("FEA_UI_REMEMBER", raising=False)


def make(title="Oddíl", **kw):
    from fea_gui.widgets import Section
    s = Section(title, **kw)
    QtWidgets.QFormLayout(s.body).addRow("x", QtWidgets.QDoubleSpinBox())
    return s


def test_open_close_and_signal(app):
    s = make(collapsed=True)
    got = []
    s.toggled.connect(got.append)
    assert not s.is_open() and not s.isChecked() and not s.body.isVisibleTo(s)
    s.show()
    assert not s.body.isVisible()
    s.set_open(True)
    assert s.is_open() and s.isChecked() and s.body.isVisible() and got == [True]
    s.setChecked(False)
    assert not s.body.isVisible() and got == [True, False]
    s.toggle.click()                                  # kompatibilita s print_sweep.Section
    assert s.is_open() and s.header is s.toggle
    s.header.click()
    assert not s.is_open()
    s.set_open(False)                                 # unchanged = no signal
    assert got == [True, False, True, False]
    s.close()


def test_title_badge_header_widget(app):
    s = make("Tlumení", tip="Krátký", help="<b>Podrobně</b>")
    assert s.title() == "Tlumení" and s.header.toolTip() == "Krátký"
    assert s.header.whatsThis() == s.whatsThis() == "<b>Podrobně</b>"
    s.setTitle("Jiný")
    assert s.title() == "Jiný"
    b = QtWidgets.QPushButton("Tlačítko")
    s.add_header_widget(b)
    s.show()
    assert not s._badge.isVisible()
    s.set_badge("3 páry")
    assert s._badge.isVisible() and s._badge.text() == "3 páry"
    s.set_badge("")
    assert not s._badge.isVisible()
    assert b.parentWidget() is s and b.isVisible()
    assert s._info is not None and s._info.objectName() == "infoButton"
    # the ⓘ button is furthest right, the pop-out button before it, the header widget before them
    lay = s._head
    assert lay.itemAt(lay.count() - 1).widget() is s._info and lay.itemAt(lay.count() - 2).widget() is s._pop_btn
    assert lay.itemAt(lay.count() - 3).widget() is b
    s.close()


def test_section_detach_to_window(app):
    from fea_gui.popout import PopoutWindow
    holder = QtWidgets.QWidget()
    v = QtWidgets.QVBoxLayout(holder)
    s = make("Odepínací", key="t.detach")
    v.addWidget(s)
    holder.show()
    s.detach()
    app.processEvents()
    assert s.is_detached() and isinstance(s.body.window(), PopoutWindow) and s.body.isVisible()
    s.set_open(False)
    assert s.body.isVisible() and not s._pop.placeholder.isVisible()
    s.body.window().close()
    for _ in range(3):
        app.processEvents()
    assert not s.is_detached() and s.body.parentWidget() is s and not s.body.isVisible()
    s.set_open(True)
    assert s.body.isVisible()
    assert make("Bez", detachable=False)._pop_btn is None
    holder.close()


def test_check_controls_section(app):
    chk = QtWidgets.QCheckBox("Harmonic response")
    s = make("Harmonická", key="dyn.h", check=chk, collapsed=False)
    s.show()
    assert not s.is_open() and not s.body.isVisible()          # the initial state = the tick
    assert chk.text() == "calculate" and chk.toolTip() == "Harmonic response" and chk.parentWidget() is s
    chk.setChecked(True)
    assert s.is_open() and s.body.isVisible()
    chk.setChecked(False)
    assert not s.is_open()
    on = QtWidgets.QCheckBox("zapnout")
    on.setChecked(True)
    s2 = make("Zapnuto", check=on)
    assert s2.is_open() and s2._key == ""
    s.close()
    s2.close()


def test_reveal(app):
    from fea_gui.widgets import Section
    outer = Section("Vnější", collapsed=True)
    lay = QtWidgets.QVBoxLayout(outer.body)
    inner = Section("Vnitřní", collapsed=True)
    lay.addWidget(inner)
    spin = QtWidgets.QDoubleSpinBox()
    QtWidgets.QFormLayout(inner.body).addRow("p", spin)
    host = QtWidgets.QWidget()
    QtWidgets.QVBoxLayout(host).addWidget(outer)
    host.show()
    app.processEvents()
    assert not spin.isVisible()
    assert Section.reveal(spin) is True
    assert outer.is_open() and inner.is_open() and spin.isVisible() and spin.width() > 0
    assert Section.reveal(spin) is False                       # it is already visible
    inner.set_open(False)
    assert Section.reveal(inner.header) is False and not inner.is_open()   # the header does not expand the section
    assert Section.reveal(inner) is False and not inner.is_open()          # nor the section itself
    host.close()


def test_set_all_in_panel(app):
    from fea_gui.widgets import Section
    stack = QtWidgets.QStackedWidget()
    page = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(page)
    a, b = Section("A", collapsed=True), Section("B", collapsed=False)
    lay.addWidget(a)
    lay.addWidget(b)
    stack.addWidget(page)
    a.set_all(True)
    assert a.is_open() and b.is_open()
    b.set_all(False)
    assert not a.is_open() and not b.is_open()


def test_remember_ui_off_stores_nothing(app, monkeypatch):
    from fea_gui import widgets
    assert widgets.remember_ui() is False

    def boom():
        raise AssertionError("nastavení se nesmí číst ani zapisovat")
    monkeypatch.setattr(widgets, "_ui_settings", boom)
    s = make("A", key="t.a", collapsed=True)
    s.header.click()
    s.set_all(False)
    assert not s.is_open()


def test_remember_ui_persistence(app, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from fea_gui import widgets
    monkeypatch.setenv("FEA_UI_REMEMBER", "1")
    assert widgets.remember_ui() is True
    st = QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat)
    monkeypatch.setitem(widgets._UI_SETTINGS, "s", st)
    s = make("A", key="t.a", collapsed=True)
    s.set_open(True)                                           # not saved from the code
    assert st.value("ui/sections/t.a") is None
    s.header.click()                                           # user click → collapsed, saved
    assert st.value("ui/sections/t.a") in (False, "false")
    s2 = make("A", key="t.a", collapsed=False)                 # restores the saved state
    assert not s2.is_open()
    s2.header.click()
    s3 = make("A", key="t.a", collapsed=True)
    assert s3.is_open()
    chk = QtWidgets.QCheckBox("x")
    s4 = make("B", key="t.b", check=chk)                       # with check= the key is not saved
    s4.header.click()
    assert st.value("ui/sections/t.b") is None


def test_placeholder_and_fit_rows(app):
    from PySide6.QtCore import Qt
    from fea_gui.widgets import fit_rows, set_placeholder, table
    t = table(["A", "B"], placeholder="Nic tu není", rows=(2, 5))
    t.resize(300, 300)
    t.show()
    app.processEvents()
    lab = t.viewport().findChild(QtWidgets.QLabel, "viewPlaceholder")
    assert lab is not None and lab.text() == "Nic tu není" and lab.isVisible()
    assert lab.testAttribute(Qt.WA_TransparentForMouseEvents)
    h0 = t.height()
    for i in range(8):
        t.insertRow(i)
    app.processEvents()
    assert not lab.isVisible()
    h5 = t.height()
    assert h5 > h0                                             # 2 rows → 5 rows (ceiling)
    for i in range(8, 12):
        t.insertRow(i)
    assert t.height() == h5                                    # above hi the height does not change (scroll bar)
    t.setRowCount(0)
    app.processEvents()
    assert lab.isVisible() and t.height() == h0
    lst = QtWidgets.QListWidget()
    set_placeholder(lst, "Prázdný seznam")
    fit_rows(lst, 1, 3)
    lst.addItem("a")
    assert lst.height() > 0
    t.close()


def test_help_tip_and_info_button(app):
    from fea_gui.widgets import help_tip, info_button
    b = QtWidgets.QPushButton("x")
    help_tip(b, "Krátce", "<p>Dlouze</p>")
    assert b.toolTip() == "Krátce" and b.whatsThis() == "<p>Dlouze</p>"
    help_tip(b, "Jinak")                                       # without long the whatsThis does not change
    assert b.toolTip() == "Jinak" and b.whatsThis() == "<p>Dlouze</p>"
    from PySide6.QtGui import QAction
    a = QAction("a")
    help_tip(a, "Akce", "Podrobnosti akce")
    assert a.toolTip() == "Akce" and a.whatsThis() == "Podrobnosti akce"
    i = info_button("<b>Podrobně</b>", "Titulek")
    assert i.objectName() == "infoButton" and i.whatsThis() == "<b>Podrobně</b>" and i.toolTip() == "Details"
    assert i.property("feaIcon") == "info"


def test_tip_filter(app, monkeypatch):
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QHelpEvent
    from fea_gui import guide
    shown = []
    monkeypatch.setattr(guide.QToolTip, "showText", staticmethod(lambda pos, text, w=None, *a: shown.append(text)))
    f = guide.TipFilter()
    ev = QHelpEvent(QEvent.ToolTip, QPoint(1, 1), QPoint(1, 1))
    w = QtWidgets.QPushButton("x")
    w.setToolTip("Krátký popisek")
    assert f.eventFilter(w, ev) is False and not shown          # a short one without whatsThis → passes
    long = ("Tohle je velmi dlouhý popisek, který vysvětluje i podrobnosti a proto přesahuje devadesát znaků, "
            "takže se musí zkrátit na jediný řádek.")
    w.setToolTip(long)
    assert f.eventFilter(w, ev) is True
    assert len(shown) == 1 and "F1 = details" in shown[0] and "zkrátit na jediný" not in shown[0]
    w.setToolTip("Dva\nřádky")                                  # multi-line
    assert f.eventFilter(w, ev) is True and "Dva" in shown[-1] and "řádky" not in shown[-1]
    w.setToolTip("Krátký")
    w.setWhatsThis("Podrobně")                                  # with whatsThis → an F1 line
    assert f.eventFilter(w, ev) is True and "F1 = details" in shown[-1]
    w.setToolTip("")
    n = len(shown)
    assert f.eventFilter(w, ev) is False and len(shown) == n    # without a tooltip nothing
    assert guide.install_tip_filter() is guide.install_tip_filter()


def test_guide_finds_sections(app):
    from fea_gui import guide
    from fea_gui.widgets import Section
    main = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(main)
    sec = Section("Tlumení", collapsed=True, help="<b>Tlumení</b> – dlouhý popis")
    spin = QtWidgets.QDoubleSpinBox()
    QtWidgets.QFormLayout(sec.body).addRow("ζ", spin)
    lay.addWidget(sec)
    main.show()
    app.processEvents()
    assert guide.find_widgets(main, "@Tlumení") == [sec]        # header → section, not the header button
    assert guide._title(sec) == "Tlumení"
    guide._reveal(spin)
    assert sec.is_open() and spin.isVisible()
    main.close()


# ------------------------------------------------------------------ fit_tabs, scroll_to, bublina, F1
def test_fit_tabs_shrinks_to_current_page(app):
    from fea_gui.widgets import fit_tabs
    tabs = QtWidgets.QTabWidget()
    small, big = QtWidgets.QLabel("malá"), QtWidgets.QWidget()
    inner = QtWidgets.QWidget()
    inner.setFixedSize(100, 500)
    QtWidgets.QVBoxLayout(big).addWidget(inner)
    tabs.addTab(small, "A")
    tabs.addTab(big, "B")
    fit_tabs(tabs)
    P = QtWidgets.QSizePolicy
    assert small.sizePolicy().verticalPolicy() == P.Preferred and big.sizePolicy().verticalPolicy() == P.Ignored
    assert tabs.minimumSizeHint().height() < 300       # sizeHint of the QTabWidget Ignored does not take it; the panel controls the minimum
    tabs.setCurrentIndex(1)
    assert small.sizePolicy().verticalPolicy() == P.Ignored and big.sizePolicy().verticalPolicy() == P.Preferred
    assert tabs.minimumSizeHint().height() >= 500


def test_scroll_to_reveals_and_scrolls(app):
    from fea_gui.widgets import scroll_to
    sc = QtWidgets.QScrollArea()
    sc.setWidgetResizable(True)
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    lay.addWidget(QtWidgets.QLabel("x"))
    lay.addSpacing(1500)
    sec = make(collapsed=True)
    lay.addWidget(sec)
    sc.setWidget(host)
    sc.resize(300, 200)
    sc.show()
    app.processEvents()
    target = sec.body.findChild(QtWidgets.QDoubleSpinBox)
    scroll_to(sc, target)
    app.processEvents()
    assert sec.is_open() and sc.verticalScrollBar().value() > 0


def test_as_html_keeps_mixed_markup_and_escapes_plain(app):
    from fea_gui.guide import _as_html
    mixed = "Text s hranou<br><b>tučně</b> a víc"
    assert _as_html(mixed) == "<p>" + mixed + "</p>"
    assert _as_html("<b>Nadpis</b> text") == "<b>Nadpis</b> text"
    assert _as_html("σ < 0 a <neznámé>\ndruhý") == "<p>σ &lt; 0 a &lt;neznámé&gt;<br>druhý</p>"


def test_details_bubble_is_translated(app, monkeypatch):
    from fea_core import i18n
    from fea_gui import guide
    i18n.set_language("en")
    try:
        label = QtWidgets.QLabel("x")
        label.setWhatsThis("Esc = close")
        main = QtWidgets.QWidget()
        main.view = None
        label.setParent(main)
        body = guide.find_explanation(main, label)[1]
        assert "Esc = close" in body and "zavřít" not in body
        assert i18n.tr("F1 = details") == "F1 = details"
    finally:
        i18n.set_language("en")


def test_tip_html_does_not_wrap(app):
    from fea_gui.guide import tip_html
    h = tip_html("Krátký popisek", True)
    assert h.count("white-space:pre") == 2 and "F1 = details" in h


def test_view_helper_disconnects_on_destroy(app):
    from fea_gui.widgets import set_placeholder, _helper
    from PySide6.QtGui import QStandardItemModel
    lst = QtWidgets.QTableView()
    lst.setModel(QStandardItemModel())
    set_placeholder(lst, "prázdno")
    h = _helper(lst)
    n = len(h._links) + len(h._range)
    lst.setModel(QStandardItemModel())                 # replacing the model does not accumulate the connections
    h.rewire()
    assert len(h._links) + len(h._range) == n
    h._drop()
    assert not h._links and not h._range


def test_extractor_collects_tr_literals():
    """tr("F1 = details") looks like a ccx card (F1 = …) – the extractor must collect it explicitly."""
    import ast
    from fea_core.i18n import extract
    c = extract._Collector()
    c.visit(ast.parse('hint = tr("F1 = details")\nx = "F1 = details"\n'))
    assert "F1 = details" in c.forced
    found = extract.collect()
    assert "F1 = details" in found["exact"]
    assert extract.load_catalog()["exact"].get("F1 = details")
