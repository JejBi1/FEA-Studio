"""Translation EN → CS: the core (templates, numbers) and the Qt layer (translation at display, reading returns the original English text)."""
import os

import pytest

from fea_core import i18n

CAT = {
    "exact": {"Force": "Síla", "Constraints": "Vazby", "Save": "Uložit", "Select part": "Vybrat díl",
              "Face": "Plocha", "Analysis step": "Krok výpočtu", "Results": "Výsledky", "Help": "Nápověda"},
    "templates": {"Nodes: {0}, elements: {1}": "Uzlů: {0}, prvků: {1}",
                  "Step “{0}” is not complete": "Krok „{0}“ není hotový",
                  "{0} mm from edge": "{0} mm od hrany",
                  "Value {{x}} = {0}": "Hodnota {{x}} = {0}"},
}


@pytest.fixture
def cs():
    i18n.use_catalog(CAT)
    i18n.set_language("cs")
    yield
    i18n.set_language("en")
    i18n.reload()


def test_default_language_is_english():
    assert i18n.language() == "en" or os.environ.get("FEA_LANG") == "cs"
    assert i18n.tr("Force") == "Force" or i18n.language() == "cs"


def test_core(cs):
    tr = i18n.tr
    assert tr("Force") == "Síla"
    assert tr("  Force ") == "  Síla "
    assert tr("Force:") == "Síla:"
    assert tr("Nodes: 12 345, elements: 6") == "Uzlů: 12 345, prvků: 6"
    assert tr("Step “Constraints” is not complete") == "Krok „Vazby“ není hotový"
    assert tr("1,5 mm from edge") == "1,5 mm od hrany"
    assert tr("Value {x} = 3") == "Hodnota {x} = 3"
    assert tr("Force\nConstraints") == "Síla\nVazby"
    assert tr("1,25") == "1,25" and tr("2,5 MPa") == "2,5 MPa"
    assert tr("Part 1") == "Part 1"                     # unknown → unchanged
    assert tr(None) is None
    i18n.set_language("en")
    assert tr("Force") == "Force" and tr("1.5") == "1.5"


def test_qt(cs):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QListWidget, QMessageBox, QPushButton,
                                   QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
    from PySide6.QtCore import Qt
    from fea_gui import i18n as gi
    app = QApplication.instance() or QApplication([])
    gi.install(app, "cs")
    w = QWidget()
    lay = QVBoxLayout(w)
    lab = QLabel("Force")                              # constructor → translation at Polish
    btn = QPushButton("Save")
    btn.setToolTip("Select part")
    combo = QComboBox()
    combo.addItems(["Force", "Face", "Part A"])
    lst = QListWidget()
    lst.addItems(["Constraints", "Results"])
    tab = QTableWidget(1, 2)
    tab.setHorizontalHeaderLabels(["Force", "Face"])
    tab.setItem(0, 0, QTableWidgetItem("Help"))
    changed = []
    tab.itemChanged.connect(lambda it: changed.append(it))
    for x in (lab, btn, combo, lst, tab):
        lay.addWidget(x)
    w.show()
    app.processEvents()
    assert lab.property("_i18n_text")[1] == "Síla"
    assert lab.text() == "Force"                       # reading returns the original
    assert btn.toolTip() == "Select part"
    assert combo.currentText() == "Force" and combo.itemText(1) == "Face" and combo.itemText(2) == "Part A"
    assert combo.model().item(0).data(0) == "Síla"      # displayed in Czech
    assert combo.findText("Face") == 1
    combo.setCurrentText("Face")
    assert combo.currentIndex() == 1
    assert lst.item(0).text() == "Constraints" and lst.item(0).data(0) == "Vazby"
    assert lst.findItems("Results", Qt.MatchFlag.MatchExactly)
    assert tab.item(0, 0).text() == "Help" and tab.item(0, 0).data(0) == "Nápověda"
    assert tab.horizontalHeaderItem(0).data(0) == "Síla"
    tab.item(0, 0).setText("Face")                     # a later change is translated, itemChanged only once
    app.processEvents()
    assert tab.item(0, 0).data(0) == "Plocha" and tab.item(0, 0).text() == "Face"
    assert len(changed) == 1
    lab.setText("Nodes: 10, elements: 2")
    assert lab.property("_i18n_text")[1] == "Uzlů: 10, prvků: 2"
    lst.addItem("Force")                               # after the display
    assert lst.item(2).data(0) == "Síla"
    QMessageBox.information                           # wrapped, not called (modal)
    w.close()


def test_core_markup(cs):
    tr = i18n.tr
    assert tr("[18:14:27] Force") == "[18:14:27] Síla"
    assert tr("<b>Force</b><br>Constraints") == "<b>Síla</b><br>Vazby"
    assert tr("<b>Face</b>") == "<b>Plocha</b>"
    assert tr("Force, Constraints") == "Síla, Vazby"


def test_core_composed(cs):
    """Composed texts from the GUI: the template has priority over "Name: description", edge spaces in the catalogue, abbreviations and brackets at the
    end, prefixes ▸ / step number, HTML tables, part names not translated, decimal point / comma of the numbers.
    """
    i18n.use_catalog({
        "exact": {"Mesh": "Síť", "Centre of gravity [mm]": "Těžiště [mm]", "Open model": "Otevřít model",
                  "  (built-in)": "  (vestavěná)", "Advice by failure location": "Rada podle místa porušení",
                  "Model": "Model", "loading geometry, parts overview": "načtení geometrie, přehled dílů",
                  "Part": "Díl", "Material": "Materiál", "Plate": "Deska", "Steel S235": "Ocel S235"},
        "templates": {"Mesh: {0} nodes, {1} elements": "Síť: {0} uzlů, {1} prvků",
                      "{0} – component {1}": "{0} – složka {1}", "\n\nDouble-click: step {0}": "\n\nDvojklik: krok {0}",
                      "{0} parts": "{0} dílů"}})
    tr = i18n.tr
    assert tr("Mesh: 24114 nodes, 13518 elements") == "Síť: 24114 uzlů, 13518 prvků"
    assert tr("Centre of gravity [mm] – component x") == "Těžiště [mm] – složka x"
    assert tr("Message\n\nDouble-click: step Mesh") == "Message\n\nDvojklik: krok Síť"
    assert tr("Open model  [Ctrl+O]") == "Otevřít model  [Ctrl+O]"
    assert tr("Model  (built-in)") == "Model  (vestavěná)"
    assert tr("► Advice by failure location") == "► Rada podle místa porušení"
    # the whole sentence from the catalogue, not a template "{0} parts" translated only in the last word
    assert tr("1. Model: loading geometry, parts overview") == "1. Model: načtení geometrie, přehled dílů"
    assert (tr("<table><tr><th>Part</th><th>Material</th></tr><tr><td>Plate</td><td>Steel S235</td></tr></table>")
            == "<table><tr><th>Díl</th><th>Materiál</th></tr><tr><td>Deska</td><td>Ocel S235</td></tr></table>")
    i18n.keep_names(["Plate"])                        # a part name from the CAD is not translated
    try:
        assert tr("Plate") == "Plate" and tr("Plate [Steel S235]") == "Plate [Ocel S235]"
        assert "<td>Plate</td><td>Ocel S235</td>" in tr("<tr><td>Plate</td><td>Steel S235</td></tr>")
    finally:
        i18n.keep_names([])
    assert tr("Plate") == "Deska"
    assert i18n.decimal("1.5") == "1,5"
    i18n.set_language("en")
    assert i18n.decimal("1.5") == "1.5"


def test_qt_czech_buttons():
    """Czech GUI: the standard Qt buttons (QMessageBox, QDialogButtonBox) in Czech through a translator in Python – the package
    PySide6 Essentials does not have qtbase_cs.qm (the question "Save changes?" had the buttons Save / Discard / Cancel).
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox
    from fea_gui import i18n as gi
    app = QApplication.instance() or QApplication([])
    old = gi._qt_cs
    gi._qt_cs = None
    gi._install_qt_czech(app)
    try:
        b = QMessageBox(QMessageBox.Question, "t", "x", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        assert sorted(x.text() for x in b.buttons()) == ["Neukládat", "Uložit", "Zrušit"]
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Yes | QDialogButtonBox.No)
        assert sorted(x.text() for x in bb.buttons()) == ["&Ano", "&Ne", "OK"]
        assert QCoreApplication.translate("X", "Neznámý") == "Neznámý"     # unknown text unchanged
    finally:
        QCoreApplication.removeTranslator(gi._qt_cs)
        gi._qt_cs = old


def test_catalog_complete():
    """Every text from the code has a Czech translation in cs.json or is marked as not translated (skip: names, symbols, data).
    Completion: python -m fea_core.i18n.extract --todo x.json --where → translate → --merge x.json (see CLAUDE.md, Languages).
    """
    import json
    from fea_core.i18n import CATALOG_CS, extract
    found = extract.collect()
    cs = json.loads(CATALOG_CS.read_text(encoding="utf-8"))
    skip = set(cs.get("skip", []))
    missing = [t for kind in ("exact", "templates") for t in found[kind]
               if not cs.get(kind, {}).get(t) and t not in skip]
    assert not missing, f"{len(missing)} texts without a translation, e.g. {missing[:5]}"


def test_help_pages_translated():
    from fea_gui.help_pdf import stale_translations
    assert stale_translations() == [], "outdated English help: python -m fea_gui.help_pdf --stale"
