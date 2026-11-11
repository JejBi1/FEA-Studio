"""Interactive help: the labels of all elements, the simulation setup guide and the "What is this?" mode with snapshots.

  python tests/gui/gui_guide_smoke.py OUT_DIR [project.fea]

1) Every interactive element of the main window (buttons, menus, fields, lists, tables, bar buttons) has a label –
   its own toolTip / whatsThis or the description from the What is this? mode (a parent with a description, the label of a form row).
2) Setup guide: every step finds all the highlighted elements and its check returns the state.
3) What is this?: explanations of the buttons, the fields of the form and the 3D window; switching the mode on / off.
4) F11 (two-level help): elements with a tooltip > 90 characters without whatsThis (with FEA_TIPS_STRICT=1 an error), the headers of the sections
   have a label, `Section.reveal` expands the target of the guide, `TipFilter` shortens a long label (+ "F1 = details"),
   F1 over an element (`context_help`) and a click on ⓘ open the bubble.
The default project = example 1 (without results). Snapshots = the window + the overlay of the highlight.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractSpinBox, QApplication, QComboBox,
                               QHeaderView, QLineEdit, QMessageBox, QScrollBar, QSlider, QTabBar, QToolButton, QWidget)

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PROJECT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "examples" / "example1_bracket.fea"
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)


def main():
    from fea_gui.guide import SETUP_STEPS, explain, find_widgets, spotlight
    from fea_gui.main import MainWindow
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1700, 1000)
    w.show()
    w.open_file(PROJECT)
    app.processEvents()
    fails = []

    # 1) popisky
    kinds = (QAbstractButton, QComboBox, QAbstractSpinBox, QLineEdit, QSlider, QAbstractItemView)
    n = 0
    for c in w.findChildren(QWidget):
        if not isinstance(c, kinds) or isinstance(c, (QScrollBar, QHeaderView)) or c.objectName().startswith("qt_"):
            continue
        if c.window() is not w:                          # drop-down lists of the menus (their own window)
            continue
        par = c.parentWidget()
        if isinstance(par, (QComboBox, QAbstractSpinBox, QTabBar, QLineEdit)) or "ComboBox" in type(par).__name__:
            continue
        n += 1
        own = c.toolTip() or c.whatsThis()
        if isinstance(c, QToolButton) and c.defaultAction() is not None:
            a = c.defaultAction()
            own = a.whatsThis() or (a.toolTip() if a.toolTip().replace("&", "") != a.text().replace("&", "") else "")
        if not own:
            fails.append(f"bez popisku: {type(c).__name__} „{getattr(c, 'text', lambda: '')()}“ "
                         f"{explain(w, c)[1][:60]!r}")
    print(f"prvků {n}, bez popisku {sum(f.startswith('bez') for f in fails)}")

    # 1b) F11: long tooltips without whatsThis; the headers of the sections
    import os
    from fea_gui.guide import TIP_MAX
    from fea_gui.widgets import Section
    long_tips = []
    for c in w.findChildren(QWidget):
        if c.window() is not w or c.objectName().startswith("qt_"):
            continue
        tip = c.toolTip()
        if isinstance(c, QToolButton) and c.defaultAction() is not None:
            tip = c.defaultAction().toolTip()
        if tip and (len(tip) > TIP_MAX or "\n" in tip.strip()) and not (c.whatsThis() or (
                isinstance(c, QToolButton) and c.defaultAction() is not None and c.defaultAction().whatsThis())):
            long_tips.append(f"{type(c).__name__} {c.objectName()!r}: {tip[:50]!r} ({len(tip)})")
    print(f"dlouhé tooltipy bez whatsThis: {len(long_tips)}")
    if long_tips and os.environ.get("FEA_TIPS_STRICT") == "1":
        fails += [f"dlouhý tooltip bez whatsThis: {t}" for t in long_tips[:20]]
    elif long_tips:
        print("  (jen upozornění, FEA_TIPS_STRICT=1 = chyba) např.", long_tips[:3])
    sections = w.findChildren(Section)
    print(f"oddílů Section: {len(sections)}")
    for sec in sections:
        if not sec.header.toolTip():
            fails.append(f"záhlaví oddílu „{sec.title()}“ bez popisku")
        if sec._info is not None and not sec._info.whatsThis():
            fails.append(f"ⓘ oddílu „{sec.title()}“ bez textu")

    # 2) the setup guide
    def shot(name):
        for _ in range(3):
            app.processEvents()
        w.view.redraw()
        app.processEvents()
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QImage, QPainter
        img = w.grab().toImage()
        tmp = OUT / "_view.png"
        w.view.screenshot(str(tmp))
        v = QImage(str(tmp))
        p = QPainter(img)
        p.drawImage(w.view.mapTo(w, w.view.rect().topLeft()), v.scaled(w.view.size()))
        sp = spotlight(w)
        if sp.isVisible():
            sp.repaint()
            p.drawImage(sp.mapToGlobal(QPoint(0, 0)) - w.mapToGlobal(QPoint(0, 0)), sp.grab().toImage())
        b = w.whats_this.bubble
        if b.isVisible():
            p.drawImage(b.mapToGlobal(QPoint(0, 0)) - w.mapToGlobal(QPoint(0, 0)), b.grab().toImage())
        p.end()
        tmp.unlink(missing_ok=True)
        img.save(str(OUT / f"{name}.png"))

    w.help_action("nastaveni")
    dock = w.guide_dock
    assert dock.isVisible() and dock.steps is SETUP_STEPS
    for k, s in enumerate(SETUP_STEPS):
        dock.go(k)
        for act in s["actions"]:
            if act[0] == "spot":
                for spec in act[1:]:
                    ws = find_widgets(w, spec)
                    if not ws:
                        fails.append(f"průvodce krok {k + 1}: prvek „{spec}“ nenalezen")
        if s.get("check"):
            state, text = s["check"](w)
            assert state in (True, False, None) and text
            assert dock.status.isVisible(), "stav kroku průvodce se nezobrazil"
            print(f"  {k + 1:2d} {s['title']}: {state} – {text[:90]}")
        shot(f"guide_{k + 1:02d}")
    w.guide_dock.hide()

    # 3) What is this?
    for target in (w.btn_run, w.btn_fixed, w.ms_size, w.ld_vec[0], w.view, w.tree, w.stage_actions[4]):
        from PySide6.QtGui import QAction
        if isinstance(target, QAction):
            target = find_widgets(w, "stage_actions.4")[0]
        t, text = explain(w, target)
        if t is None or "no separate description" in text:
            fails.append(f"Co je to? bez vysvětlení: {type(target).__name__}")
    w.set_stage(w.STAGE["Loads"])
    w.whats_this.toggle(True)
    assert w.whats_this.active and QApplication.overrideCursor() is not None
    t, text = explain(w, w.btn_add_load)
    from PySide6.QtCore import QPoint
    w.whats_this.bubble.show_at(w.btn_add_load.mapToGlobal(QPoint(10, 10)), text)
    spotlight(w).set_hover(t)
    shot("cojeto_zatizeni")
    w.whats_this.toggle(False)
    assert not w.whats_this.active and not w.whats_this.bubble.isVisible()
    # a real click: in the mode the button does not run and a bubble appears; after Esc the button works
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
    clicks = []
    w.set_stage(w.STAGE["Model"])
    w.btn_open_model.clicked.connect(lambda: clicks.append(1))
    app.processEvents()
    w.whats_this.toggle(True)
    QTest.mouseClick(w.btn_open_model, Qt.LeftButton, Qt.NoModifier, QPoint(10, 10))
    app.processEvents()
    if clicks or not w.whats_this.bubble.isVisible() or not any(x in w.whats_this.bubble.text() for x in ("Open model", "Otevřít model")):
        fails.append(f"Co je to?: klik prošel do tlačítka ({clicks}) nebo chybí bublina")
    QTest.keyClick(w, Qt.Key_Escape)
    app.processEvents()
    QTest.mouseClick(w.btn_open_model, Qt.LeftButton, Qt.NoModifier, QPoint(10, 10))
    app.processEvents()
    if w.whats_this.active or clicks != [1]:
        fails.append("Co je to?: po Esc režim neskončil / tlačítko nereaguje")

    # 4) F11: Section.reveal for the target of the guide, TipFilter, context_help, ⓘ
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QHelpEvent
    from fea_gui import guide
    w.set_stage(w.STAGE["Dynamics"])
    app.processEvents()
    zeta = find_widgets(w, "dynamics_panel.d_zeta")
    sec = None
    if zeta:
        p = zeta[0].parentWidget()
        while p is not None and not isinstance(p, Section):
            p = p.parentWidget()
        sec = p
    if sec is None:
        print("  (Section.reveal: d_zeta zatím není v oddílu Section – přeskočeno)")
    else:
        sec.set_open(False)
        app.processEvents()
        if zeta[0].isVisible():
            fails.append("Section: sbalený oddíl ukazuje d_zeta")
        guide.spot(w, ["dynamics_panel.d_zeta"])
        for _ in range(3):
            app.processEvents()
        if not sec.is_open() or not zeta[0].isVisible() or zeta[0].width() <= 0:
            fails.append("Section.reveal: cíl průvodce ve sbaleném oddílu se nerozbalil / není vidět")
        spotlight(w).clear_spots()

    shown = []
    orig_show = guide.QToolTip.showText
    guide.QToolTip.showText = staticmethod(lambda pos, text, widget=None, *a: shown.append(text))
    try:
        probe = QToolButton(w)
        probe.setToolTip("Dlouhý popisek, který vysvětluje velmi mnoho podrobností a proto překračuje devadesát "
                         "znaků, takže se v bublině zkrátí.")
        ev = QHelpEvent(QEvent.ToolTip, QPoint(1, 1), QPoint(1, 1))
        QApplication.sendEvent(probe, ev)
        if not shown or not any(x in shown[-1] for x in ("F1 = details", "F1 = podrobnosti")) or "zkrátí" in shown[-1]:
            fails.append(f"TipFilter: dlouhý popisek nebyl zkrácen ({shown[-1:]})")
        probe.deleteLater()
    finally:
        guide.QToolTip.showText = orig_show

    w.set_stage(w.STAGE["Solve"])
    app.processEvents()
    orig_at = guide.QApplication.widgetAt
    guide.QApplication.widgetAt = staticmethod(lambda p: w.btn_run)
    try:
        w.whats_this.bubble.hide()
        if not guide.context_help(w) or not w.whats_this.bubble.isVisible():
            fails.append("context_help (F1) nad btn_run neotevřela bublinu")
        else:
            QTest.keyClick(w, Qt.Key_Escape)
            app.processEvents()
            if w.whats_this.bubble.isVisible():
                fails.append("bublina podrobností se nezavřela klávesou Esc")
    finally:
        guide.QApplication.widgetAt = orig_at
    infos = [s for s in sections if s._info is not None]
    if not infos:
        print("  (ⓘ: žádný oddíl s help – přeskočeno)")
    else:
        w.whats_this.bubble.hide()
        infos[0]._info.click()
        app.processEvents()
        if not w.whats_this.bubble.isVisible():
            fails.append("klik na ⓘ neotevřel bublinu")
        shot("info_bubble")
        w.whats_this.bubble.hide()

    # 5) F1 in the whole application and the fixes from the review (fix1): the 3D window / tree / log → False (the page of the step), a dialog → a bubble,
    # the focus with the cursor outside the window, a text with <br> / <b> in the middle is not escaped, ⓘ can be focused
    from PySide6.QtWidgets import QDialog, QDoubleSpinBox, QVBoxLayout
    from fea_gui.widgets import info_button
    for name in ("view", "tree", "log"):
        target = getattr(w, name)
        guide.QApplication.widgetAt = staticmethod(lambda p, t=target: t)
        try:
            if guide.context_help(w):
                fails.append(f"F1 nad {name} otevřela bublinu (má otevřít stránku kroku)")
        finally:
            guide.QApplication.widgetAt = orig_at
    dlg = QDialog()
    lay = QVBoxLayout(dlg)
    spin = QDoubleSpinBox()
    spin.setToolTip("Krátký popisek pole")
    spin.setWhatsThis("Text před značkou<br><b>Tučně</b> a konec")
    lay.addWidget(spin)
    ib = info_button("Podrobnosti <i>mixed</i><br>řádek 2", "Titulek")
    lay.addWidget(ib)
    dlg.show()
    spin.setFocus()
    app.processEvents()
    if ib.focusPolicy() != Qt.StrongFocus:
        fails.append("ⓘ nejde zaostřit klávesnicí")
    guide.QApplication.widgetAt = staticmethod(lambda p: None)         # the cursor over no window of the program
    try:
        w.whats_this.bubble.hide()
        QTest.keyClick(spin, Qt.Key_F1)
        app.processEvents()
        b = w.whats_this.bubble
        if not b.isVisible() or "&lt;" in b.text() or "<b>Tučně</b>" not in b.text():
            fails.append(f"F1 v dialogu (fokus na poli): bublina chybí / escapuje značky ({b.text()[:80]!r})")
        shot("f1_dialog")
        b.hide()
    finally:
        guide.QApplication.widgetAt = orig_at
    dlg.close()

    w.dirty = False
    w.close()
    for f in fails:
        print("CHYBA:", f)
    print("VÝSLEDEK:", "OK" if not fails else f"{len(fails)} chyb")
    sys.exit(1 if fails else 0)


main()
