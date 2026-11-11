"""GUI language (Czech / English) without intervention in the panels.

The texts are in English in the code (still partly Czech during the transition). They are translated where they reach Qt,
in both languages (Czech: English → Czech catalogue, English: the remaining Czech texts → English):
- setter methods (setText, setToolTip, setWindowTitle, addTab, QMessageBox…, appendPlainText, showMessage) are
  wrapped – they translate the text via `fea_core.i18n.tr`,
- texts given in the constructor (QLabel("…"), QAction("…")) are translated by an event filter at QEvent.Polish (before the first
  display of the widget); menu and bar actions together with their widget,
- items of lists, tables, trees and menus (QComboBox) are translated in the model (rowsInserted / dataChanged).
Reading methods (text(), currentText(), itemText(), findText()…) return the ORIGINAL text (the one in the code) → panel logic that
compares texts works in both languages unchanged.

The language is set in QSettings "language" (cs | en); a change applies after a restart (`install` must run before the
windows are created).
"""
from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QEvent, QLocale, QModelIndex, QObject, Qt, QTranslator
from PySide6.QtGui import QAction, QStandardItem
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractSpinBox, QApplication, QComboBox,
                               QDockWidget, QFileDialog, QGroupBox, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QStatusBar,
                               QTabBar, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit, QToolBar, QToolBox,
                               QToolTip, QTreeWidget, QTreeWidgetItem, QWidget)

from fea_core import i18n as core
from fea_core.i18n import tr as _tr
from fea_core.richtext import safe_rich

from . import richtext_qt

richtext_qt.install()                        # safe_rich: second layer = Qt's own parser (in the GUI thread)


def tr(text):
    """Translation + `richtext.safe_rich` (names from foreign files must not load <img src=\"//host/…\"> – NTLM leak)."""
    return safe_rich(_tr(text))


ORIG = int(Qt.ItemDataRole.UserRole) + 0x3A7          # original Czech text of the model item
_ROLES = (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole, Qt.ItemDataRole.WhatsThisRole,
          Qt.ItemDataRole.StatusTipRole)
_installed = False
_filter = None
_chosen = None                                       # language chosen by the first call of install()


def saved_language() -> str:
    """GUI language: environment variable FEA_LANG (tests), the choice from the installation / the language switch (language.txt in the
    user settings), the older choice in QSettings, otherwise by the system (Czech → cs)."""
    import os
    from .user_settings import app_settings
    lang = os.environ.get("FEA_LANG") or core.stored_language() or app_settings().value("language", "")
    if lang not in core.LANGS:
        lang = "en" if QLocale.system().language() != QLocale.Language.Czech else "cs"
    return lang


def save_language(lang: str) -> None:
    from .user_settings import app_settings
    core.store_language(lang)                    # shared with the installer (the last choice wins)
    app_settings().setValue("language", lang)


# ------------------------------------------------------------------ wrapping of widget properties and actions
def _wrap_prop(cls, getter: str, setter: str):
    """setter(text) translates and remembers the original; getter() returns the original if the text has not changed in the meantime."""
    g0, s0 = getattr(cls, getter), getattr(cls, setter)
    if getattr(s0, "_i18n_wrapped", False):   # inherited wrapper: a second layer would see the translated text and drop the original
        return g0, s0
    key = f"_i18n_{getter}"

    def set_(self, text, *a, **k):
        if isinstance(text, str):
            en = tr(text)
            if en != text:                  # a dynamic property restyles the widget: set it only for a translated text
                self.setProperty(key, (text, en))
            elif self.property(key) is not None:
                self.setProperty(key, None)
            return s0(self, en, *a, **k)
        return s0(self, text, *a, **k)

    def get_(self, *a, **k):
        v = g0(self, *a, **k)
        rec = self.property(key)
        if rec and isinstance(rec, (tuple, list)) and rec[1] == v:
            return rec[0]
        return v

    set_._i18n_wrapped = True
    setattr(cls, setter, set_)
    setattr(cls, getter, get_)
    return g0, s0


def _translate_prop(obj, getter: str, setter: str):
    """Translates the text already set (from the constructor) through the wrapped setter."""
    cur = getattr(type(obj), getter)(obj)     # wrapped getter → original
    if isinstance(cur, str) and cur:
        getattr(obj, setter)(cur)


def _wrap_indexed(cls, getter: str, setter: str):
    """tabText(i) / setTabText(i, text) and similar – the originals by index in a property."""
    g0, s0 = getattr(cls, getter), getattr(cls, setter)
    if getattr(s0, "_i18n_wrapped", False):
        return
    key = f"_i18n_{getter}"

    def set_(self, i, text):
        if isinstance(text, str):
            en = tr(text)
            d = dict(self.property(key) or {})
            d[str(i)] = (text, en)
            self.setProperty(key, d)
            return s0(self, i, en)
        return s0(self, i, text)

    def get_(self, i):
        v = g0(self, i)
        rec = (self.property(key) or {}).get(str(i))
        if rec and rec[1] == v:
            return rec[0]
        return v

    set_._i18n_wrapped = True
    setattr(cls, setter, set_)
    setattr(cls, getter, get_)


def _wrap_args(cls, name: str, *pos: int, lists: tuple = ()):
    """Method whose arguments at positions pos (texts) and lists (lists of texts) are translated."""
    f0 = getattr(cls, name)

    def f(*a, **k):
        a = list(a)
        for i in pos:
            if i < len(a) and isinstance(a[i], str):
                a[i] = tr(a[i])
        for i in lists:
            if i < len(a) and isinstance(a[i], (list, tuple)):
                a[i] = [tr(x) for x in a[i]]
        for kk in ("title", "text", "caption", "filter", "label", "informativeText"):
            if isinstance(k.get(kk), str):
                k[kk] = tr(k[kk])
        return f0(*a, **k)

    setattr(cls, name, staticmethod(f) if isinstance(cls.__dict__.get(name), staticmethod) else f)
    return f0


RAW = "_i18n_raw"            # property of a text field: the content is not translated (help, guide – their own English texts)


def raw(widget):
    """The content of a text field (QTextEdit / QTextBrowser) is not translated – the help has its own English pages and the guide
    translates its texts itself. Other text fields (results summary, model info…) are translated as labels.
    """
    widget.setProperty(RAW, True)
    return widget


def _wrap_text(cls, name: str):
    """setHtml / setText / append… of a text field: the text is translated unless the field is marked raw()."""
    f0 = getattr(cls, name)

    def f(self, *a, **k):
        if a and isinstance(a[0], str) and not self.property(RAW):
            a = ((_tr(a[0]) if name in ("setPlainText", "insertPlainText") else tr(a[0])),) + a[1:]
        return f0(self, *a, **k)

    setattr(cls, name, f)
    return f0


def _wrap_static(cls, name: str, *pos: int):
    f0 = getattr(cls, name)

    def f(*a, **k):
        a = list(a)
        for i in pos:
            if i < len(a) and isinstance(a[i], str):
                a[i] = tr(a[i])
        for kk in ("title", "text", "caption", "filter", "label"):
            if isinstance(k.get(kk), str):
                k[kk] = tr(k[kk])
        return f0(*a, **k)

    setattr(cls, name, staticmethod(f))


def _wrap_init(cls, pos: int):
    """Constructor whose text argument at `pos` (or `text=`) goes through `safe_rich` (no translation – Polish translates it later
    and the reading methods keep returning the text from the code)."""
    f0 = cls.__init__
    if getattr(f0, "_i18n_wrapped", False):
        return

    def init(self, *a, **k):
        if pos < len(a) and isinstance(a[pos], str):
            a = a[:pos] + (safe_rich(a[pos]),) + a[pos + 1:]
        if isinstance(k.get("text"), str):
            k["text"] = safe_rich(k["text"])
        f0(self, *a, **k)

    init._i18n_wrapped = True
    cls.__init__ = init


# ------------------------------------------------------------------ item models
def _translate_index(model, idx):
    """Translates the texts of an item. The model signals are blocked meanwhile – the translation is not a data change (itemChanged would
    otherwise report an edit to the panels); the view draws the item anyway from the model.
    """
    for role in _ROLES:
        v = model.data(idx, role)
        if not isinstance(v, str) or not v:
            continue
        en = tr(v) if role != Qt.ItemDataRole.DisplayRole else _tr(v)
        if en != v:
            old = model.blockSignals(True)
            try:
                if role == Qt.ItemDataRole.DisplayRole:
                    model.setData(idx, v, ORIG)
                model.setData(idx, en, role)
            finally:
                model.blockSignals(old)


def _columns(model, parent=QModelIndex()) -> int:
    try:
        return model.columnCount(parent)
    except TypeError:                   # QAbstractListModel: columnCount is private in PySide
        return 1


def _translate_rows(model, parent, first, last):
    if model.property("_i18n_busy"):
        return
    model.setProperty("_i18n_busy", True)
    try:
        cols = _columns(model, parent)
        for r in range(first, last + 1):
            for c in range(cols):
                idx = model.index(r, c, parent)
                _translate_index(model, idx)
                if c == 0:
                    n = model.rowCount(idx)
                    if n:
                        model.setProperty("_i18n_busy", False)
                        _translate_rows(model, idx, 0, n - 1)
                        model.setProperty("_i18n_busy", True)
    finally:
        model.setProperty("_i18n_busy", False)


def _translate_range(model, tl, br):
    if model.property("_i18n_busy") or not tl.isValid():
        return
    model.setProperty("_i18n_busy", True)
    try:
        parent = tl.parent()
        for r in range(tl.row(), br.row() + 1):
            for c in range(tl.column(), br.column() + 1):
                _translate_index(model, model.index(r, c, parent))
    finally:
        model.setProperty("_i18n_busy", False)


def _translate_header(model, orient, first, last):
    if model.property("_i18n_busy"):
        return
    model.setProperty("_i18n_busy", True)
    try:
        for i in range(first, last + 1):
            for role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
                v = model.headerData(i, orient, role)
                if isinstance(v, str) and v:
                    en = _tr(v)
                    if en != v:
                        model.setHeaderData(i, orient, en, role)
    finally:
        model.setProperty("_i18n_busy", False)


def _hook_model(model):
    if model is None or model.property("_i18n_hooked"):
        return
    model.setProperty("_i18n_hooked", True)
    model.rowsInserted.connect(lambda p, a, b, m=model: _translate_rows(m, p, a, b))
    model.dataChanged.connect(lambda tl, br, roles=(), m=model: _translate_range(m, tl, br))
    model.headerDataChanged.connect(lambda o, a, b, m=model: _translate_header(m, o, a, b))

    def reset(m=model):
        n = m.rowCount()
        if n:
            _translate_rows(m, QModelIndex(), 0, n - 1)
        for o, cnt in ((Qt.Orientation.Horizontal, _columns(m)), (Qt.Orientation.Vertical, 0)):
            if cnt:
                _translate_header(m, o, 0, cnt - 1)
    model.modelReset.connect(reset)
    reset()


def _orig_item(get0):
    def get(self, *a):
        o = self.data(*a, ORIG) if a else self.data(ORIG)
        if isinstance(o, str):
            v = get0(self, *a)
            if v == _tr(o):
                return o
        return get0(self, *a)
    return get


# ------------------------------------------------------------------ event filter
_POLISH, _ACTION = QEvent.Type.Polish, QEvent.Type.ActionAdded


class _Polish(QObject):
    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == _POLISH:
            if isinstance(obj, QWidget):
                try:
                    polish(obj)
                except Exception as e:          # translation must not crash the GUI (the object may be destroyed in the meantime, etc.)
                    _report(e)
        elif t == _ACTION and obj.property("_i18n_done"):
            translate_action(ev.action())
        return False


_reported = set()


def _report(e: Exception) -> None:
    key = f"{type(e).__name__}: {e}"
    if key not in _reported:
        _reported.add(key)
        import sys
        print(f"i18n: {key}", file=sys.stderr)


def polish(w: QWidget) -> None:
    """Translates the widget texts given in the constructor (called once before the first display)."""
    if w.property("_i18n_done"):
        return
    w.setProperty("_i18n_done", True)
    if core.english():                      # English source: nothing to translate, but rich texts from constructors are sanitised
        _sanitize(w)
        return
    for g, s in (("toolTip", "setToolTip"), ("whatsThis", "setWhatsThis"), ("statusTip", "setStatusTip")):
        _translate_prop(w, g, s)
    if w.isWindow() or isinstance(w, (QDockWidget, QToolBar)):       # bars: the name in the context menu of the bars
        _translate_prop(w, "windowTitle", "setWindowTitle")
    if isinstance(w, QLabel):
        _translate_prop(w, "text", "setText")
    elif isinstance(w, QAbstractButton):
        _translate_prop(w, "text", "setText")
    elif isinstance(w, QGroupBox):
        _translate_prop(w, "title", "setTitle")
    elif isinstance(w, QLineEdit):
        _translate_prop(w, "placeholderText", "setPlaceholderText")
    elif isinstance(w, QAbstractSpinBox):
        t = w.specialValueText()
        if t:
            w.setSpecialValueText(tr(t))
        for g, s in (("suffix", "setSuffix"), ("prefix", "setPrefix")):
            if hasattr(w, g) and getattr(w, g)():
                getattr(w, s)(_tr(getattr(w, g)()))
    elif isinstance(w, QMenu):
        _translate_prop(w, "title", "setTitle")
    elif isinstance(w, QTabBar):
        for i in range(w.count()):
            w.setTabText(i, w.tabText(i))
            if w.tabToolTip(i):
                w.setTabToolTip(i, tr(w.tabToolTip(i)))
    elif isinstance(w, QToolBox):
        for i in range(w.count()):
            w.setItemText(i, w.itemText(i))
    if isinstance(w, QComboBox):
        _hook_model(w.model())
    elif isinstance(w, QAbstractItemView):
        _hook_model(w.model())
    if isinstance(w, (QPlainTextEdit, QTextEdit)):
        _translate_prop(w, "placeholderText", "setPlaceholderText")
    for a in w.actions():
        translate_action(a)


def _sanitize_prop(obj, getter: str, setter: str) -> None:
    """English UI: a text set in the constructor (not seen by the setter wrapper) goes through `safe_rich`; the setter is called only
    when the text changes (a plain re-set would restyle / re-layout every widget)."""
    cur = getattr(type(obj), getter)(obj)
    if isinstance(cur, str) and cur and ("<" in cur or "](" in cur):
        if safe_rich(cur) != cur:
            getattr(obj, setter)(cur)            # wrapped setter: stores the original, shows the sanitised text


def _sanitize(w: QWidget) -> None:
    """Rich-text capable texts of a widget in the English UI (labels, tooltips, What's This, tab tooltips, actions)."""
    for g, s in (("toolTip", "setToolTip"), ("whatsThis", "setWhatsThis")):
        _sanitize_prop(w, g, s)
    if isinstance(w, QLabel):
        _sanitize_prop(w, "text", "setText")
    elif isinstance(w, QTabBar):
        for i in range(w.count()):
            t = w.tabToolTip(i)
            if t and safe_rich(t) != t:
                w.setTabToolTip(i, t)
    if isinstance(w, (QComboBox, QAbstractItemView)):
        _hook_model(w.model())                   # item tooltips → safe_rich
    for a in w.actions():
        if not a.property("_i18n_done"):
            a.setProperty("_i18n_done", True)
            for g, s in (("toolTip", "setToolTip"), ("whatsThis", "setWhatsThis")):
                _sanitize_prop(a, g, s)


def translate_action(a: QAction) -> None:
    if a.property("_i18n_done"):
        return
    a.setProperty("_i18n_done", True)
    for g, s in (("text", "setText"), ("toolTip", "setToolTip"), ("whatsThis", "setWhatsThis"),
                 ("statusTip", "setStatusTip"), ("iconText", "setIconText")):
        _translate_prop(a, g, s)
    m = a.menu()
    if m is not None:
        _translate_prop(m, "title", "setTitle")


# ------------------------------------------------------------------ Qt texts in Czech
# Standard dialog buttons (QMessageBox, QDialogButtonBox) and the context menus of fields come from Qt's own translations
# (qtbase_cs.qm), which the PySide6_Essentials package does not have (and the installer deletes the translations folder anyway) → without them
# the question "Save changes?" would have the buttons Save / Discard / Cancel. Translation in Python: (context, text) → Czech;
# unknown text → None = Qt uses the original.
_QT_CS = {
    "QPlatformTheme": {"OK": "OK", "Save": "Uložit", "Save All": "Uložit vše", "Open": "Open", "&Yes": "&Ano",
                       "Yes to &All": "Ano &všem", "&No": "&Ne", "N&o to All": "N&e všem", "Abort": "Přerušit",
                       "Retry": "Opakovat", "Ignore": "Ignorovat", "Close": "Close", "Cancel": "Zrušit",
                       "Discard": "Neukládat", "Help": "Nápověda", "Apply": "Použít", "Reset": "Obnovit",
                       "Restore Defaults": "Obnovit výchozí"},
    "QMessageBox": {"Show Details...": "Zobrazit podrobnosti…", "Hide Details...": "Skrýt podrobnosti…",
                    "&OK": "&OK", "OK": "OK"},
    "QLineEdit": {"&Undo": "&Zpět", "&Redo": "&Znovu", "Cu&t": "Vyjmou&t", "&Copy": "&Kopírovat",
                  "&Paste": "&Vložit", "Delete": "Smazat", "Select All": "Select all"},
    "QWidgetTextControl": {"&Undo": "&Zpět", "&Redo": "&Znovu", "Cu&t": "Vyjmou&t", "&Copy": "&Kopírovat",
                           "Copy &Link Location": "Kopírovat &adresu odkazu", "&Paste": "&Vložit",
                           "Delete": "Smazat", "Select All": "Select all"},
    "QAbstractSpinBox": {"&Select All": "&Vybrat vše", "&Step up": "&Zvýšit", "Step &down": "&Snížit"},
}


class _QtCzech(QTranslator):
    def translate(self, context, source, disambiguation=None, n=-1):
        return _QT_CS.get(context, {}).get(source)

    def isEmpty(self):
        return False


_qt_cs = None


def _install_qt_czech(app) -> None:
    global _qt_cs
    if _qt_cs is None and app is not None:
        _qt_cs = _QtCzech(app)
        QCoreApplication.installTranslator(_qt_cs)


# ------------------------------------------------------------------ instalace
def install(app: QApplication, lang: str | None = None) -> str:
    """Sets the language (from QSettings if not given). In English it wraps Qt; returns the language used.
    Without lang and after an earlier choice (test, script) the language does not change – MainWindow calls install at every creation.
    """
    global _installed, _filter, _chosen
    if lang is None and _chosen:
        return _chosen
    lang = lang or saved_language()
    _chosen = lang
    core.set_language(lang)
    if lang == "cs":
        _install_qt_czech(app)
    if _installed:
        return lang
    _installed = True
    if lang == "en":                       # the Czech UI keeps the Czech number format of the spin boxes
        QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

    # vlastnosti s getterem
    for cls in (QLabel, QAbstractButton):
        _wrap_prop(cls, "text", "setText")
    _wrap_prop(QGroupBox, "title", "setTitle")
    _wrap_prop(QMenu, "title", "setTitle")
    _wrap_prop(QLineEdit, "placeholderText", "setPlaceholderText")
    for cls in (QWidget, QAction):
        for g, s in (("toolTip", "setToolTip"), ("whatsThis", "setWhatsThis"), ("statusTip", "setStatusTip")):
            _wrap_prop(cls, g, s)
    _wrap_prop(QWidget, "windowTitle", "setWindowTitle")
    _wrap_prop(QAction, "text", "setText")
    _wrap_prop(QAction, "iconText", "setIconText")
    for cls in (QTabWidget, QTabBar):
        _wrap_indexed(cls, "tabText", "setTabText")
        _wrap_args(cls, "setTabToolTip", 2)
    _wrap_indexed(QToolBox, "itemText", "setItemText")

    # addition with text
    _wrap_args(QTabWidget, "addTab", 2, 3)
    _wrap_args(QTabWidget, "insertTab", 3, 4)
    _wrap_args(QToolBox, "addItem", 2, 3)
    _wrap_args(QMenu, "addMenu", 1, 2)
    _wrap_args(QMenu, "addSection", 1, 2)

    # texts that are only displayed
    for cls in (QPlainTextEdit,):
        for n in ("appendPlainText", "setPlainText", "insertPlainText"):
            _wrap_args(cls, n, 1)
    for n in ("append", "setPlainText", "setHtml", "setText", "setMarkdown"):
        _wrap_text(QTextEdit, n)                # also the QTextBrowser summaries (results, model, print); help see raw()
    _wrap_args(QStatusBar, "showMessage", 1)
    _wrap_args(QProgressBar, "setFormat", 1)
    for n in ("setText", "setInformativeText", "setDetailedText"):
        _wrap_args(QMessageBox, n, 1)
    for n in ("information", "warning", "critical", "question", "about"):
        _wrap_static(QMessageBox, n, 1, 2)
    for n in ("getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory"):
        _wrap_static(QFileDialog, n, 1, 3)
    _wrap_static(QToolTip, "showText", 1)
    _input_dialogs()
    # constructors that parse rich text at once (a QLabel / QMessageBox / QTextEdit document is built in the constructor and
    # <link> / @import stylesheets are fetched while parsing – before the Polish event): the text only sanitised, translated later
    _wrap_init(QLabel, 0)
    _wrap_init(QMessageBox, 2)
    _wrap_init(QTextEdit, 0)

    # items: reading returns the original
    QListWidgetItem.text = _orig_item(QListWidgetItem.text)
    QTableWidgetItem.text = _orig_item(QTableWidgetItem.text)
    QTreeWidgetItem.text = _orig_item(QTreeWidgetItem.text)
    QStandardItem.text = _orig_item(QStandardItem.text)
    _combo()
    for cls in (QListWidget, QTreeWidget):
        f0 = cls.findItems

        def find(self, text, flags, *a, _f0=f0):
            return _f0(self, text, flags, *a) or _f0(self, _tr(text), flags, *a)
        cls.findItems = find
    _wrap_args(QTableWidget, "setHorizontalHeaderLabels", lists=(1,))
    _wrap_args(QTableWidget, "setVerticalHeaderLabels", lists=(1,))
    _wrap_args(QTreeWidget, "setHeaderLabels", lists=(1,))
    _wrap_args(QTreeWidget, "setHeaderLabel", 1)

    _filter = _Polish(app)
    app.installEventFilter(_filter)
    core.patch_matplotlib()
    return lang


def _combo():
    it0, cur0, find0, setcur0 = QComboBox.itemText, QComboBox.currentText, QComboBox.findText, QComboBox.setCurrentText

    def item_text(self, i):
        o = self.itemData(i, ORIG)
        v = it0(self, i)
        return o if isinstance(o, str) and v == _tr(o) else v

    def current_text(self):
        i = self.currentIndex()
        if i >= 0 and not self.isEditable():
            return item_text(self, i)
        if i >= 0 and cur0(self) == it0(self, i):
            return item_text(self, i)
        return cur0(self)

    def find_text(self, text, *a):
        for i in range(self.count()):
            if self.itemData(i, ORIG) == text:
                return i
        return find0(self, text, *a) if find0(self, text, *a) >= 0 else find0(self, _tr(text), *a)

    def set_current(self, text):
        i = find_text(self, text)
        if i >= 0 and not self.isEditable():
            self.setCurrentIndex(i)
        else:
            setcur0(self, _tr(text) if i >= 0 else text)

    QComboBox.itemText = item_text
    QComboBox.currentText = current_text
    QComboBox.findText = find_text
    QComboBox.setCurrentText = set_current


def _input_dialogs():
    gi0 = QInputDialog.getItem

    def get_item(parent, title, label, items, *a, **k):
        items = list(items)
        en = [_tr(x) for x in items]
        res = gi0(parent, _tr(title), tr(label), en, *a, **k)          # the label is an AutoText QLabel: sanitised
        text, ok = res
        if text in en:
            text = items[en.index(text)]
        return text, ok
    QInputDialog.getItem = staticmethod(get_item)
    for n in ("getText", "getInt", "getDouble", "getMultiLineText"):
        _wrap_static(QInputDialog, n, 1, 2)


# ------------------------------------------------------------------ language switch in the bar
_GLOBE = ('<circle cx="12" cy="12" r="8"/><ellipse cx="12" cy="12" rx="3.6" ry="8"/>'
          '<path d="M4.5 9.2h15M4.5 14.8h15"/>')
_CZ_FLAG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 16">'
    '<rect width="24" height="8" fill="#ffffff"/>'
    '<rect y="8" width="24" height="8" fill="#d7141a"/>'
    '<polygon points="0,0 12,8 0,16" fill="#11457e"/>'
    '<rect width="24" height="16" fill="none" stroke="#718096" stroke-width="0.8"/>'
    '</svg>'
)
_EN_FLAG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 16">'
    '<clipPath id="cf"><rect width="24" height="16"/></clipPath>'
    '<g clip-path="url(#cf)">'
    '<rect width="24" height="16" fill="#012169"/>'
    '<path d="M0 0 L24 16 M24 0 L0 16" stroke="#ffffff" stroke-width="3"/>'
    '<path d="M0 0 L24 16 M24 0 L0 16" stroke="#c8102e" stroke-width="1.6"/>'
    '<path d="M12 0 v16 M0 8 h24" stroke="#ffffff" stroke-width="5"/>'
    '<path d="M12 0 v16 M0 8 h24" stroke="#c8102e" stroke-width="3"/>'
    '</g>'
    '<rect width="24" height="16" fill="none" stroke="#718096" stroke-width="0.8"/>'
    '</svg>'
)
FLAGS = {"cs": _CZ_FLAG, "en": _EN_FLAG}
_flag_cache: dict[str, QIcon] = {}


def flag_icon(lang: str) -> QIcon:
    """Colour vector flag of the language (CZ / UK) with a fine border."""
    if lang not in _flag_cache:
        svg_str = FLAGS.get(lang)
        if not svg_str:
            return QIcon()
        from PySide6.QtCore import QByteArray, QRectF
        from PySide6.QtGui import QIcon, QPainter, QPixmap
        from PySide6.QtSvg import QSvgRenderer
        app = QApplication.instance()
        dpr = app.devicePixelRatio() if app is not None else 1.0
        w, h = 21, 14
        pm = QPixmap(int(w * dpr), int(h * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        QSvgRenderer(QByteArray(svg_str.encode())).render(p, QRectF(0, 0, w, h))
        p.end()
        _flag_cache[lang] = QIcon(pm)
    return _flag_cache[lang]


_RESTART = {"cs": ("Jazyk", "The program language changes after a restart.\n\nRestart FEA Studio now?"),
            "en": ("Language", "The program language changes after a restart.\n\nRestart FEA Studio now?")}


def language_button(main):
    """"Language / Jazyk" button with the menu of languages (next to the Appearance button). The choice is saved to QSettings and after
    confirmation the program restarts (an unfinished project offers to save it, as on closing).
    """
    from PySide6.QtGui import QActionGroup
    from PySide6.QtWidgets import QMenu, QToolButton
    b = QToolButton()
    cur_lang = core.language()
    b.setText("Čeština" if cur_lang == "cs" else "English")
    b.setProperty("_i18n_done", True)                   # language name is not translated
    b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    b.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    b.setToolTip("Jazyk programu / Program language")
    b.setIcon(flag_icon(cur_lang))
    m = QMenu(b)
    m.setProperty("_i18n_done", True)
    grp = QActionGroup(m)
    for key, name in core.LANGS.items():
        a = m.addAction(flag_icon(key), name, lambda k=key: _switch(main, k))
        a.setProperty("_i18n_done", True)
        a.setCheckable(True)
        a.setChecked(key == cur_lang)
        grp.addAction(a)
    b.setMenu(m)
    return b


def _switch(main, lang: str) -> None:
    if lang == core.language():
        return
    save_language(lang)
    title, text = _RESTART[lang]
    box = QMessageBox(QMessageBox.Icon.Question, title, text, QMessageBox.StandardButton.Yes |
                      QMessageBox.StandardButton.No, main)
    box.setProperty("_i18n_done", True)
    if box.exec() != QMessageBox.StandardButton.Yes:
        return
    restart(main)


def restart(main) -> None:
    """Closes the main window (with the save question) and starts the program again with the current project."""
    import sys
    from PySide6.QtCore import QProcess
    path = getattr(main, "project_path", None) or getattr(main, "current_file", None)
    if not main.close():
        return
    args = ["-P", "-m", "fea_gui"] + ([str(path)] if path else [])
    from pathlib import Path
    QProcess.startDetached(sys.executable, args, str(Path(__file__).resolve().parents[1]))
    QApplication.instance().quit()
