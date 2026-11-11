"""Common helpers of the GUI panels: numbers with a decimal comma, numeric field, read-only table (with text for an
empty table, height by rows and copying the whole table – Ctrl+C, context menu), check lists,
scrollable panel content, collapsible section `Section`, two-level labels (`help_tip`, `info_button`), a
matplotlib chart loaded only at the first drawing, colour #rrggbb.

Collapsible section (F11, docs/projekt/F11_gui_prehlednost.md):
    sec = Section("Damping", key="dyn.damping", collapsed=False, tip="Vibration damping", help="<b>Details</b>…")
    f = QFormLayout(sec.body); f.addRow("ζ", self.d_zeta)          # content into sec.body
The collapsed state is remembered (QSettings ui/sections/<key>) only after a click by the user and only in a program started by
`python -m fea_gui` (FEA_UI_REMEMBER=1, see `remember_ui`) – tests always see the default state.
"""
import math
import os

from PySide6.QtCore import QEvent, QObject, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractScrollArea, QApplication, QCheckBox,
                               QDoubleSpinBox, QFrame, QHBoxLayout, QHeaderView, QLabel, QListWidgetItem, QMenu,
                               QProgressBar, QScrollArea, QSizePolicy, QStackedWidget, QTabWidget, QTableView, QTableWidget, QToolButton,
                               QTreeView, QVBoxLayout, QWidget)
from fea_core.i18n import decimal

TIP_MAX = 90                  # hard limit on the length of a short label (target ≤ 80 characters); a longer text belongs to whatsThis


def fmt(v, dec=4, inf="∞"):
    """Number with a decimal comma (decimal point in English); None and NaN → "–", infinity → inf."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, float) and math.isinf(v):
        return inf
    return decimal(f"{v:.{dec}g}")


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(v, dec=1):
    """Small / large numbers such as 4·10⁻⁶ (status bar: equilibrium); 0 → "0"."""
    if v is None:
        return "–"
    if v == 0:
        return "0"
    mant, exp = f"{v:.{max(dec - 1, 0)}e}".split("e")
    e = int(exp)
    if -3 <= e <= 3:
        return fmt(v, dec)
    return decimal(mant) + "·10" + str(e).translate(_SUP)


def fmt_dash(v, dec=4):
    """As fmt, but infinity is also "–" (print and heat panels)."""
    return fmt(v, dec, "–")


def dspin(val, lo, hi, dec=3, step=1.0, suffix="", special=None):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    s.setSuffix(suffix)
    s.setGroupSeparatorShown(False)
    if special:
        s.setSpecialValueText(special)
    return s


def panel_header(title: str, subtitle: str = "", step: int | None = None) -> QFrame:
    """Panel step heading: small phase name (Geometry, Assembly, Conditions, Calculation, Evaluation), large name and a grey
    subtitle (appearance in theme.QSS: panelTitle, panelSubtitle). The phase comes from main.PHASE_OF; if it is missing,
    the old "STEP n" is shown (number by main.STAGES). The parameter step is ignored for compatibility.
    """
    label = ""
    try:
        from .main import PHASE_OF, phase_label
        label = phase_label(PHASE_OF[title]) if title in PHASE_OF else ""
    except ImportError:
        try:
            from .main import STAGES
            label = f"STEP {STAGES.index(title) + 1}"
        except (ImportError, ValueError):
            pass
    fr = QFrame()
    fr.setObjectName("panelHeader")
    v = QVBoxLayout(fr)
    v.setContentsMargins(0, 0, 0, 8)
    v.setSpacing(1)
    if label:
        s = QLabel(label)
        s.setObjectName("panelStep")
        if not label.startswith("STEP"):
            f = s.font()
            f.setCapitalization(f.Capitalization.AllUppercase)
            s.setFont(f)
        v.addWidget(s)
    t = QLabel(title)
    t.setObjectName("panelTitle")
    v.addWidget(t)
    if subtitle:
        sub = QLabel(subtitle)
        sub.setObjectName("panelSubtitle")
        sub.setWordWrap(True)
        v.addWidget(sub)
    return fr


def table(cols, min_height=0, stretch_last=False, *, placeholder="", rows=None) -> QTableWidget:
    """Read-only table: selection of whole rows one at a time, column widths by content, no row numbers.
    placeholder = grey text for an empty table (`set_placeholder`), rows=(min, max) = height by the number of rows
    (`fit_rows`).
    """
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(stretch_last)
    t.verticalHeader().hide()
    if min_height:
        t.setMinimumHeight(min_height)
    if placeholder:
        set_placeholder(t, placeholder)
    if rows:
        fit_rows(t, *rows)
    copy_menu(t)
    return t


def check_item(lst, text: str, on: bool, data=None) -> QListWidgetItem:
    """Check item at the end of list lst (data → Qt.UserRole)."""
    it = QListWidgetItem(text)
    if data is not None:
        it.setData(Qt.UserRole, data)
    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
    it.setCheckState(Qt.Checked if on else Qt.Unchecked)
    lst.addItem(it)
    return it


def checked(lst, role=None) -> list:
    """Checked items of the list: texts, or data role."""
    items = [lst.item(i) for i in range(lst.count())]
    return [it.text() if role is None else it.data(role) for it in items if it.checkState() == Qt.Checked]


def set_checks(lst, values, role=None):
    """Checks exactly the items whose text (or data role) is in values."""
    for i in range(lst.count()):
        it = lst.item(i)
        it.setCheckState(Qt.Checked if (it.text() if role is None else it.data(role)) in values else Qt.Unchecked)


def scroll_body(panel: QWidget):
    """Content of the step panel in a scroll area (horizontal scroll bar only when the content does not fit even after narrowing – otherwise
    the right edge would be silently cut); returns (QScrollArea, content layout).
    """
    outer = QVBoxLayout(panel)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    body = QWidget()
    lay = QVBoxLayout(body)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(8)
    scroll.setWidget(body)
    outer.addWidget(scroll)
    return scroll, lay


class TaskProgress(QProgressBar):
    """Green progress bar of the calculation 0–100 % with the phase description and elapsed time (look: theme.QSS #taskProgress).

    report(value, end of phase, text) from the core (fea_core.progress) – the bar moves smoothly between messages
    towards 85 % of the rest of the phase (long phases without progress messages: Gmsh, factorisation); it never goes back.
    """
    TICK_MS = 100
    CREEP = 0.006                 # share of the remainder up to the end of the phase per tick (time constant ~17 s)

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtCore import QElapsedTimer, QTimer
        self.setObjectName("taskProgress")
        self.setRange(0, 1000)
        self.setTextVisible(True)
        help_tip(self, "Calculation progress: finished part, current phase and elapsed time",
                 "Analysis progress: completed part, current phase and elapsed time.\nInside long phases (meshing, solving the system) the progress is estimated.")
        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()
        self._shown = self._target = self._end = 0.0
        self._text = ""
        self.hide()

    def _state(self, state: str):
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)

    def start(self, text: str, end: float = 0.05):
        self._shown = self._target = 0.0
        self._end = end
        self._text = text
        self._state("running")
        self._clock.start()
        self._timer.start()
        self.show()
        self._paint()

    def report(self, value: float, end: float, text: str):
        self._target = max(self._target, min(value, 1.0))
        self._end = max(self._end, self._target, min(end, 1.0))
        self._text = text or self._text
        if not self._timer.isActive():
            self._paint()

    def finish(self, state: str = "done", text: str = "Done"):
        """state: done (100 %) | error | cancelled – the bar stays with the result until the next calculation starts."""
        self._timer.stop()
        if state == "done":
            self._shown = self._target = 1.0
        self._text = text
        self._state(state)
        self._paint()

    def elapsed(self) -> float:
        return self._clock.elapsed() / 1000.0 if self._clock.isValid() else 0.0

    def _tick(self):
        cap = self._target + 0.85 * (self._end - self._target)
        if self._shown < self._target:
            self._shown += max(0.002, 0.35 * (self._target - self._shown))
            self._shown = min(self._shown, self._target)
        elif self._shown < cap:
            self._shown += self.CREEP * (cap - self._shown)
        self._paint()

    def _paint(self):
        t = int(self.elapsed())
        self.setValue(int(round(self._shown * 1000)))
        self.setFormat(f"{int(self._shown * 100)} %  ·  {self._text}  ·  {t // 60}:{t % 60:02d}")


def compact(root: QWidget, combo_chars: int = 10):
    """Narrower panels: menus may be narrowed below the length of the longest item (otherwise they set the width of the whole panel – the text
    is shortened, the full one is in the drop-down and the tooltip), and long form rows put the label above the field.
    """
    from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QFormLayout
    for c in root.findChildren(QComboBox):
        if c.sizeAdjustPolicy() != QComboBox.AdjustToMinimumContentsLengthWithIcon:
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(combo_chars)
    for s in root.findChildren(QAbstractSpinBox):    # a width by the range (±1e9) would be needlessly large
        if s.minimumWidth() == 0:
            s.setMinimumWidth(min(s.minimumSizeHint().width(), 72))
    for f in root.findChildren(QFormLayout):
        f.setRowWrapPolicy(QFormLayout.WrapLongRows)


def C_HEX(c):
    return "#%02x%02x%02x" % c


class LazyPlot(QWidget):
    """Matplotlib chart; matplotlib is loaded only at the first drawing (its import of ~1.9 s slowed the GUI start)."""
    FIGSIZE = (4, 2.6)
    MIN_HEIGHT = 220

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = self.canvas = None
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self.setMinimumHeight(self.MIN_HEIGHT)

    def _ensure(self):
        if self.canvas is None:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=self.FIGSIZE, dpi=90, tight_layout=True)
            self.canvas = FigureCanvasQTAgg(self.fig)
            self._lay.addWidget(self.canvas)

    def axes(self):
        """Empty chart (one set of axes) for a new drawing."""
        self._ensure()
        self.fig.clear()
        return self.fig.add_subplot(111)


# ------------------------------------------------------------------ karty a posuv panelu
def fit_tabs(tabs: QTabWidget) -> None:
    """Tabs adapt to the CURRENT page: the other pages have size policy Ignored (they do not stretch the QTabWidget), the current one Preferred. Call after adding all tabs (a tab added later must be called again).
    The `minimumSizeHint()` is reduced (the scroll panel sizes the content by it); the `sizeHint()` of the QTabWidget always adds up the pages.
    """
    def apply(*_):
        cur = tabs.currentWidget()
        for i in range(tabs.count()):
            p = tabs.widget(i)
            pol = p.sizePolicy()
            pol.setHorizontalPolicy(QSizePolicy.Ignored if p is not cur else QSizePolicy.Preferred)
            pol.setVerticalPolicy(QSizePolicy.Ignored if p is not cur else QSizePolicy.Preferred)
            p.setSizePolicy(pol)
        tabs.updateGeometry()
    if tabs.property("_fit_tabs") is None:
        tabs.setProperty("_fit_tabs", True)
        tabs.currentChanged.connect(apply)
    apply()


def scroll_to(scroll: QScrollArea, w: QWidget, *, deferred: bool = False, margin: int = 40) -> None:
    """Scrolls the panel's scroll area so that element w is visible (guide, advice, test snapshots). Collapsed sections
    above w are expanded. deferred=True scrolls only after returning to the event loop (the layout has not been recomputed yet).
    Panels have `def scroll_to(self, w): widgets.scroll_to(self._scroll, w)`.
    """
    Section.reveal(w)
    if deferred:
        QTimer.singleShot(0, lambda: scroll.ensureWidgetVisible(w, 0, margin))
    else:
        scroll.ensureWidgetVisible(w, 0, margin)


# ------------------------------------------------------------------ two-level labels
def help_tip(obj, short: str, long: str = "") -> None:
    """Two-level label (QWidget | QAction): a short toolTip (one line, ≤ 80 characters) and the details in
    whatsThis (HTML allowed; F1 over the element, ⓘ, Shift+F1). Without long, whatsThis is not changed.
    """
    obj.setToolTip(short)
    if long:
        obj.setWhatsThis(long)


def info_button(long: str, title: str = "") -> QToolButton:
    """Small ⓘ button: a click shows a bubble with the detailed description `long` (HTML allowed). The caller places it."""
    from . import theme
    b = QToolButton()
    b.setObjectName("infoButton")
    b.setAutoRaise(True)
    b.setFocusPolicy(Qt.StrongFocus)           # Tab + Enter / space; F1 with focus on it shows the same bubble
    b.setCursor(Qt.PointingHandCursor)
    theme.set_icon(b, "info")
    b.setToolTip("Details")
    b.setWhatsThis(long)

    def show():
        from .guide import show_details
        show_details(b, long, title)
    b.clicked.connect(show)
    return b


# ------------------------------------------------------------------ collapsible section
_UI_SETTINGS = {}


def remember_ui() -> bool:
    """Remember the GUI state (collapsed sections, pinned steps) between runs? Only in a program started by
    `python -m fea_gui` (it sets FEA_UI_REMEMBER=1); tests and scripts that create MainWindow directly do not.
    """
    return os.environ.get("FEA_UI_REMEMBER") == "1"


def _ui_settings():
    """Shared QSettings instance (created only at the first use)."""
    if "s" not in _UI_SETTINGS:
        from .user_settings import app_settings
        _UI_SETTINGS["s"] = app_settings()
    return _UI_SETTINGS["s"]


def _stored_open(key: str):
    v = _ui_settings().value(f"ui/sections/{key}")
    return None if v is None else str(v).lower() in ("1", "true", "yes")


def _store_open(key: str, on: bool) -> None:
    s = _ui_settings()
    s.setValue(f"ui/sections/{key}", bool(on))
    s.sync()


class _Header(QToolButton):
    """Section header: Enter toggles, the right arrow expands, the left collapses (the space bar is handled by the button)."""

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key_Return, Qt.Key_Enter) or (k == Qt.Key_Right and not self.isChecked()) \
                or (k == Qt.Key_Left and self.isChecked()):
            self.click()
            e.accept()
            return
        super().keyPressEvent(e)


class Section(QFrame):
    """Collapsible panel section (replacement of QGroupBox): header with an arrow, name, grey badge, header widgets and
    ⓘ (only with help), content `body` below it. Look in theme.QSS (#section, #sectionHeader, #sectionBadge).

    key – permanent ASCII key "panel.section" for remembering the state (QSettings ui/sections/<key>, only after a click by the
    user and with remember_ui()); collapsed – default state; tip – short header label; help – detailed description
    (whatsThis of the header and the section + ⓘ button); check – check box switching the function on (it moves into the header, its
    text is shortened to "compute", the original text goes into the label; the section expands / collapses with the tick, and the key
    is then not saved). The check box must not be in another layout (the caller does not insert it there).
    detachable – button in the header that moves the content into a separate window (popout.Popout; closing the window returns it).
    """
    toggled = Signal(bool)                   # True = rozbaleno

    def __init__(self, title: str, content: QWidget | None = None, *, key: str = "", collapsed: bool = False,
                 tip: str = "", help: str = "", check: QAbstractButton | None = None, opened: bool | None = None,
                 detachable: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("section")
        if opened is not None:               # older interface print_sweep.Section(title, content, opened=)
            collapsed = not opened
        self._key = key if check is None else ""
        self._check = check
        self._info = None
        self._pop = None                     # popout.Popout of the content (created at the first detach)
        self._pop_btn = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(4)
        self._head = QHBoxLayout()
        self._head.setSpacing(4)
        self._head.setContentsMargins(0, 0, 0, 0)
        self.header = self.toggle = _Header()
        self.header.setObjectName("sectionHeader")
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setText(title)
        self.header.setToolTip(tip or "Collapse / expand section")
        self.header.setContextMenuPolicy(Qt.CustomContextMenu)
        self.header.customContextMenuRequested.connect(self._menu)
        self._head.addWidget(self.header, 1)
        self._badge = QLabel()
        self._badge.setObjectName("sectionBadge")
        self._badge.hide()
        self._head.addWidget(self._badge)
        lay.addLayout(self._head)
        if content is None:
            content = QWidget()
            content.setObjectName("sectionBody")
        self.body = self.content = content
        lay.addWidget(content)
        if detachable:
            from .popout import popout_button
            self._pop_btn = popout_button("Open the section in a separate window (closing it returns the section)")
            self._pop_btn.clicked.connect(self.toggle_detached)
            self._head.addWidget(self._pop_btn)
        if help:
            self.setWhatsThis(help)
            self.header.setWhatsThis(help)
            self._info = info_button(help, title)
            self._head.addWidget(self._info)
        if check is not None:
            if isinstance(check, QCheckBox) and check.text() and check.text() != "calculate":
                if not check.toolTip():
                    check.setToolTip(check.text())
                check.setText("calculate")
            self.add_header_widget(check)
            collapsed = not check.isChecked()
            check.toggled.connect(self.set_open)
        elif key and remember_ui():
            st = _stored_open(key)
            if st is not None:
                collapsed = not st
        self.header.setChecked(not collapsed)
        self._apply(not collapsed)
        self.header.toggled.connect(self._apply)
        self.header.clicked.connect(self._clicked)

    # --- stav
    def _apply(self, on: bool) -> None:
        from . import theme
        if self.is_detached():               # content in its own window: the header collapses only the placeholder
            self._pop.placeholder.setVisible(on)
        else:
            self.body.setVisible(on)
        theme.set_icon(self.header, "chevron_down" if on else "chevron_right")
        self.toggled.emit(on)

    def _clicked(self, _=False) -> None:
        if self._key and remember_ui():
            _store_open(self._key, self.header.isChecked())

    def set_open(self, on: bool) -> None:
        if self.header.isChecked() != bool(on):
            self.header.setChecked(bool(on))

    def is_open(self) -> bool:
        return self.header.isChecked()

    isChecked = is_open                      # replacement of a checkable QGroupBox only for collapsing (rs_group, el_group)
    setChecked = set_open

    # --- header
    def title(self) -> str:
        return self.header.text()

    def setTitle(self, t: str) -> None:
        self.header.setText(t)

    def set_badge(self, text: str) -> None:
        """Grey summary on the right of the header ("3 pairs", "on"); an empty text hides it."""
        self._badge.setText(text)
        self._badge.setVisible(bool(text))

    def add_header_widget(self, w: QWidget) -> None:
        """Widget on the right of the header (before the pop-out button and ⓘ): a check box "compute", a button…"""
        tail = (self._info is not None) + (self._pop_btn is not None)
        self._head.insertWidget(self._head.count() - tail, w)

    # --- separate window
    def is_detached(self) -> bool:
        return self._pop is not None and self._pop.is_detached()

    def toggle_detached(self) -> None:
        self.attach() if self.is_detached() else self.detach()

    def detach(self) -> None:
        """Content into a separate window (the section stays as a header with a placeholder)."""
        if self._pop is None:
            from .popout import Popout
            self._pop = Popout(self.body, self.title(), f"section.{self._key}" if self._key else "",
                               note="The section is open in a separate window.", default_size=(480, 360))
            self._pop.changed.connect(self._on_detached)
        self.set_open(True)
        self._pop.detach()

    def attach(self) -> None:
        if self._pop is not None:
            self._pop.attach()

    def _on_detached(self, detached: bool) -> None:
        from . import theme
        if detached:
            self._pop.placeholder.setVisible(self.is_open())
        else:
            self.body.setVisible(self.is_open())
        if self._pop_btn is not None:
            theme.set_icon(self._pop_btn, "dock_in" if detached else "popout")
            self._pop_btn.setToolTip("Return the section into the panel" if detached
                                     else "Open the section in a separate window (closing it returns the section)")

    def is_header_part(self, w: QWidget | None) -> bool:
        """Is w part of the section header (button, badge, header widgets)? Such a target does not expand the section."""
        while w is not None and w is not self:
            if w is self.body:
                return False
            if w.parentWidget() is self:
                return True
            w = w.parentWidget()
        return False

    # --- Expand / collapse all
    def _scope(self) -> QWidget:
        """Step panel: the highest QStackedWidget above the section (tabs inside the panel are not counted), otherwise the window."""
        scope, p = None, self
        while p is not None:
            if isinstance(p.parentWidget(), QStackedWidget):
                scope = p
            p = p.parentWidget()
        return scope if scope is not None else self.window()

    def set_all(self, on: bool) -> None:
        scope = self._scope()
        for s in dict.fromkeys([self] + scope.findChildren(Section)):
            s.set_open(on)
            s._clicked()

    def _menu(self, pos) -> None:
        m = QMenu(self.header)
        m.addAction("Expand all", lambda: self.set_all(True))
        m.addAction("Collapse all", lambda: self.set_all(False))
        m.exec(self.header.mapToGlobal(pos))

    @staticmethod
    def reveal(w: QWidget) -> bool:
        """Expands all collapsed sections above w (guide, What is this?, field error) – except the section whose header
        is w itself. Returns True when something was expanded (the layout is recomputed at once, the geometry is valid).
        """
        opened = []
        p, child = w, None
        while p is not None:
            if isinstance(p, Section) and not p.is_open() and p is not w and not p.is_header_part(w):
                opened.append(p)
            p, child = p.parentWidget(), p
        for s in opened:
            s.set_open(True)
        from .popout import raise_for
        raise_for(w)                         # w in a separate window → that window to the front
        if opened:
            p = w
            while p is not None:           # valid geometry immediately (highlight, scroll to the element)
                if p.layout() is not None:
                    p.layout().activate()
                p = p.parentWidget()
        return bool(opened)


# ------------------------------------------------------------------ empty tables, height by rows
class _ViewHelper(QObject):
    """Text for an empty table / list (placeholder) and height by the number of rows (fit_rows) – one helper per view."""

    def __init__(self, view: QAbstractItemView):
        super().__init__(view)
        self.view = view
        self.label = None
        self.rows = None                      # (lo, hi) or None
        self._model = None
        self._busy = False
        self._vp = view.viewport()
        self._vp.installEventFilter(self)
        self._links = []                      # (signal, slot) connected to the model – disconnected when the model is replaced and on destruction
        self._range = []
        view.destroyed.connect(self._drop)
        self.rewire()

    @staticmethod
    def _disconnect(links) -> None:
        for sig, slot in links:
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _drop(self, *_):
        """View is being destroyed: disconnects the model and slider signals (otherwise Qt reports "Slot '_ViewHelper::' not found")."""
        self._disconnect(self._links + self._range)
        self._links, self._range, self._model = [], [], None

    def rewire(self) -> None:
        m = self.view.model()
        if m is self._model or m is None:
            return
        self._disconnect(self._links)          # signals of the earlier model removed
        self._links = []
        self._model = m
        for sig in (m.rowsInserted, m.rowsRemoved, m.modelReset, m.layoutChanged):
            sig.connect(self.refresh)
            self._links.append((sig, self.refresh))
        if not self._range:                    # the slider is the same for the whole life of the view → once
            sb = self.view.horizontalScrollBar()
            sb.rangeChanged.connect(self.refresh)
            self._range.append((sb.rangeChanged, self.refresh))

    def refresh(self, *_):
        try:
            self._refresh()
        except (RuntimeError, AttributeError):   # the view is being destroyed (model signals when the window is closed)
            pass

    def _refresh(self):
        self.rewire()
        m = self.view.model()
        n = m.rowCount() if m is not None else 0
        if self.label is not None:
            self.label.setVisible(n == 0)
            self._place()
        if self.rows is not None:
            self._fit(n)

    def _place(self):
        self.label.setGeometry(self.view.viewport().rect().adjusted(10, 6, -10, -6))

    def _row_h(self, i: int) -> int:
        v = self.view
        n = v.model().rowCount() if v.model() is not None else 0
        if isinstance(v, QTableView):
            return v.rowHeight(i) if i < n else v.verticalHeader().defaultSectionSize()
        h = v.sizeHintForRow(i) if i < n else -1
        return h if h > 0 else v.fontMetrics().height() + 8

    def _fit(self, n: int):
        if self._busy:
            return
        lo, hi = self.rows
        v = self.view
        shown = max(lo, min(n, hi))
        h = sum(self._row_h(i) for i in range(shown)) + 2 * v.frameWidth()
        hh = v.horizontalHeader() if isinstance(v, QTableView) else (
            v.header() if isinstance(v, QTreeView) else None)
        if hh is not None and not hh.isHidden():
            h += hh.height() or hh.sizeHint().height()
        hb = v.horizontalScrollBar()
        if hb.isVisible():
            h += hb.height()
        if v.height() != h or v.minimumHeight() != h:
            self._busy = True
            try:
                v.setFixedHeight(h)
            finally:
                self._busy = False

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Resize and obj is getattr(self, "_vp", None) and getattr(self, "label", None) is not None:
            self.refresh()
        return False


def _helper(view: QAbstractItemView) -> _ViewHelper:
    h = view.findChild(_ViewHelper)
    return h if h is not None else _ViewHelper(view)


def set_placeholder(view: QAbstractItemView, text: str) -> None:
    """Grey text in the middle of an empty table / list (disappears with the first row; the mouse passes through)."""
    h = _helper(view)
    if h.label is None:
        h.label = QLabel(view.viewport())
        h.label.setObjectName("viewPlaceholder")
        h.label.setAlignment(Qt.AlignCenter)
        h.label.setWordWrap(True)
        h.label.setAttribute(Qt.WA_TransparentForMouseEvents)
    h.label.setText(text)
    h.refresh()


def fit_rows(view: QAbstractItemView, lo: int = 2, hi: int = 10) -> None:
    """View height = frame + header + clamp(rows, lo, hi) rows; changes with the number of rows, above hi a scroll bar."""
    h = _helper(view)
    h.rows = (lo, hi)
    view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    h.refresh()


# ------------------------------------------------------------------ copying the whole table
def _paste_safe(s: str) -> str:
    """Cell for pasting into a spreadsheet: without tabs and line breaks; text starting with = + @ (or "-" not followed by a number)
    gets an apostrophe – Excel would otherwise evaluate it as a formula after pasting.
    """
    s = s.replace("\t", " ").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    if s[:1] in ("=", "+", "@") or (s[:1] == "-" and not s[1:2].isdigit() and s[1:2] not in (".", ",")):
        return "'" + s
    return s


def table_text(view: QAbstractItemView, sep: str = "\t") -> str:
    """The whole table as text: header and all visible rows and columns (displayed texts – in English translated
    and with a decimal point), columns separated by a tab → pasting into Excel splits the values into cells.
    """
    m = view.model()
    if m is None:
        return ""
    table_view = isinstance(view, QTableView)
    cols = [c for c in range(m.columnCount()) if not (table_view and view.isColumnHidden(c))]
    rows = [r for r in range(m.rowCount()) if not (table_view and view.isRowHidden(r))]

    def cell(v) -> str:
        return "" if v is None else _paste_safe(str(v))
    lines = [sep.join(cell(m.headerData(c, Qt.Horizontal, Qt.DisplayRole)) for c in cols)]
    lines += [sep.join(cell(m.data(m.index(r, c), Qt.DisplayRole)) for c in cols) for r in rows]
    return "\n".join(lines) + "\n"


class _CopyHelper(QObject):
    """Ctrl+C and the context menu "Copy table": the whole table with the header to the clipboard (the row selection does not change).
    Ctrl+C is caught by an event filter (otherwise the item view would copy only the current cell).
    """

    def __init__(self, view: QAbstractItemView):
        super().__init__(view)
        self.view = view
        act = QAction(view)
        act.setText("Copy table")
        act.setToolTip("Copies the whole table including the header (paste into Excel cell by cell)")
        act.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        act.setShortcutContext(Qt.WidgetShortcut)
        act.setShortcutVisibleInContextMenu(True)
        act.triggered.connect(self.copy)
        view.addAction(act)
        self.action = act
        if view.contextMenuPolicy() == Qt.DefaultContextMenu:     # do not overwrite the caller's own menu
            view.setContextMenuPolicy(Qt.ActionsContextMenu)
        view.installEventFilter(self)

    def copy(self):
        QApplication.clipboard().setText(table_text(self.view))

    def eventFilter(self, obj, ev):
        t = ev.type()
        if obj is self.view and t in (QEvent.ShortcutOverride, QEvent.KeyPress) \
                and ev.matches(QKeySequence.StandardKey.Copy):
            if t == QEvent.KeyPress:
                self.copy()
            ev.accept()
            return True
        return False


def copy_menu(view: QAbstractItemView) -> QAction:
    """Adds copying of the whole table to a table (context menu + Ctrl+C, see table_text); returns the menu action."""
    h = view.findChild(_CopyHelper)
    return (h if h is not None else _CopyHelper(view)).action
