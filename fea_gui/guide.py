"""Interactive help: highlighting of elements (Spotlight), the "What is this?" mode and the simulation setup guide.

- `Spotlight` – transparent window over the main window (tool window, passes the mouse): frames the elements with numbers ①②③, the rest
  of the window is darkened. A separate window, not a child widget – the OCCT 3D window is a native window and a child widget would cover it.
- `WhatsThisMode` – Shift+F1: a cursor with a question mark, the element under the mouse is framed, a click shows a bubble with the explanation
  (whatsThis, otherwise the toolTip of the element or of the nearest parent) and a link to the help page of the step. Clicks are not passed
  to the elements; ends with Esc, the right button or Shift+F1 again.
- `show_details` / `context_help` / `short_of` / `TipFilter` – two-level help (F11): a short tooltip, details in
  whatsThis on F1 over the element, a click on ⓘ and Shift+F1; long tooltips are shortened in `TipFilter` to one line.
- `SETUP_STEPS` – steps of the guide for setting up one's own simulation (`TourDock` dock from tour.py): they switch the step of the
  progress bar, highlight elements and show a live check ("check") of what is done in the step.

Highlight target (spec): path to an attribute of the main window ("btn_fixed", "print_panel.parts", "filter_actions.face",
"ld_vec" – a list = one frame around all of them), "=name" = objectName (bars, docks), "@Text" = button / group by its text;
several targets under one number are joined by "|". A QAction is found as the button on the bar.
"""
from __future__ import annotations

import html
import re
import time

from fea_core.i18n import tr

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QFont, QPainter, QPainterPath, QPen, QTextDocumentFragment
from PySide6.QtWidgets import (QAbstractButton, QAbstractScrollArea, QAbstractSpinBox, QApplication, QComboBox,
                               QDockWidget, QFormLayout, QGroupBox, QLabel, QLineEdit, QScrollArea, QTabBar,
                               QToolBar, QToolButton, QToolTip, QWidget)

from .widgets import TIP_MAX, Section

ACCENT = QColor("#ff8c1a")
HOVER = QColor("#1d6fd1")
SPOT_SECONDS = 9.0
NUMS = "(1)(2)(3)(4)(5)(6)(7)(8)(9)"

# the 3D window has no toolTip (it would pop up over the model) – explanation only for "What is this?"
VIEW_HELP = ("<b>3D window</b><p><b>Left button</b> – selection (face / part / hole / edge / point depending on the step and selection filter), in Results a probe = value at the point. Clicking on empty space clears the selection.<br><b>Right button – drag</b> rotates about the point under the cursor, <b>click</b> opens the context menu.<br><b>Middle button</b> or <b>Ctrl + right</b> – pan. <b>Wheel</b> – zoom toward the point under the cursor. <b>Double-click</b> – zoom to selection. <b>F</b> – fit all.<br>The <b>view cube</b> at the top right rotates the view when you click a face, edge or corner.</p>")


# ------------------------------------------------------------------ search for targets
def _attr(obj, path: str):
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, (list, tuple)) and part.isdigit():
            obj = obj[int(part)] if int(part) < len(obj) else None
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _clean(text: str) -> str:
    return text.replace("&&", "\0").replace("&", "").replace("\0", "&").strip()


def _action_widget(main, action) -> QWidget | None:
    for tb in main.findChildren(QToolBar):
        w = tb.widgetForAction(action)
        if w is not None:
            return w
    return None


def _by_text(main, text: str) -> QWidget | None:
    want = text.lower()
    hits = [w for w in main.findChildren(QAbstractButton)
            if want in _clean(w.text()).lower() and w.objectName() != "sectionHeader"]
    hits += [g for g in main.findChildren(QGroupBox) if want in g.title().lower()]
    hits += [sec for sec in main.findChildren(Section) if want in _clean(sec.title()).lower()]   # header → section
    hits.sort(key=lambda w: not w.isVisibleTo(main))          # visible first
    return hits[0] if hits else None


def find_widgets(main, spec: str) -> list[QWidget]:
    """Widgets of the highlight target (see the module docstring); parts not found are left out."""
    from PySide6.QtGui import QAction
    out: list[QWidget] = []
    for one in spec.split("|"):
        one = one.strip()
        if one.startswith("@"):
            found = [_by_text(main, one[1:])]
        elif one.startswith("="):
            found = [main.findChild(QWidget, one[1:])]
        else:
            obj = _attr(main, one)
            found = list(obj) if isinstance(obj, (list, tuple)) else [obj]
        for o in found:
            if isinstance(o, QAction):
                o = _action_widget(main, o)
            if isinstance(o, QWidget):
                out.append(o)
    return out


def _reveal(w: QWidget) -> None:
    """Expands the collapsed sections above the element and scrolls the parent scroll area so that the element is visible."""
    Section.reveal(w)
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, QScrollArea):
            p.ensureWidgetVisible(w, 10, 10)
        p = p.parentWidget()


def _global_rect(w: QWidget) -> QRect | None:
    if not w.isVisible() or w.width() <= 0:
        return None
    return QRect(w.mapToGlobal(QPoint(0, 0)), w.size())


# ------------------------------------------------------------------ overlay with the highlight
class Spotlight(QWidget):
    """Transparent tool window over the main window: numbers and frames around the targets, the rest darkened; a "hover" frame
    (What is this? mode). The mouse passes through (WindowTransparentForInput).
    """

    def __init__(self, main):
        super().__init__(main, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowTransparentForInput
                         | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.main = main
        self.groups: list[list[QWidget]] = []     # number k → widgets (one frame around all of them)
        self.hover: QWidget | None = None
        self.until = 0.0
        self.t0 = time.monotonic()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)

    def spot(self, groups: list[list[QWidget]], seconds: float = SPOT_SECONDS):
        self.groups = [g for g in groups if g]
        from .popout import attach_for
        for g in self.groups:
            for w in g:
                attach_for(w)                          # the overlay covers the main window only → targets back into it
                _reveal(w)
        self.until = time.monotonic() + seconds
        self.t0 = time.monotonic()
        self._tick()

    def clear_spots(self):
        self.groups = []
        self._tick()

    def set_hover(self, w: QWidget | None):
        if w is not self.hover:
            self.hover = w
            self._tick()

    def _active(self) -> bool:
        return bool(self.groups and time.monotonic() < self.until) or self.hover is not None

    def _tick(self):
        if not self._active() or not self.main.isVisible() or self.main.isMinimized():
            self.groups = [] if time.monotonic() >= self.until else self.groups
            self.timer.stop()
            self.hide()
            return
        geo = QRect(self.main.mapToGlobal(QPoint(0, 0)), self.main.size())
        if self.geometry() != geo:
            self.setGeometry(geo)
        if not self.isVisible():
            self.show()
        if not self.timer.isActive():
            self.timer.start()
        self.update()

    def _rect(self, ws: list[QWidget]) -> QRectF | None:
        r = QRect()
        for w in ws:
            g = _global_rect(w)
            if g is not None:
                r = r.united(g)
        if r.isNull():
            return None
        r.translate(-self.mapToGlobal(QPoint(0, 0)))
        return QRectF(r).adjusted(-4, -4, 4, 4)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rects = [self._rect(g) for g in self.groups] if time.monotonic() < self.until else []
        if any(rects):
            shade = QPainterPath()
            shade.addRect(QRectF(self.rect()))
            for r in rects:
                if r is not None:
                    hole = QPainterPath()
                    hole.addRoundedRect(r, 6, 6)
                    shade = shade.subtracted(hole)
            p.fillPath(shade, QColor(0, 0, 0, 70))
            pulse = 0.5 + 0.5 * abs(((time.monotonic() - self.t0) * 1.6) % 2 - 1)
            col = QColor(ACCENT)
            col.setAlphaF(0.55 + 0.45 * pulse)
            font = QFont(self.font())
            font.setBold(True)
            font.setPointSizeF(10)
            p.setFont(font)
            for k, r in enumerate(rects):
                if r is None:
                    continue
                p.setPen(QPen(col, 3))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(r, 6, 6)
                if len(rects) > 1:                         # numbers only for several targets (links ①②③ in the text of the step)
                    c = QRectF(r.left() - 11, r.top() - 11, 22, 22)
                    c.moveLeft(max(c.left(), 1.0))
                    c.moveTop(max(c.top(), 1.0))
                    p.setPen(Qt.NoPen)
                    p.setBrush(ACCENT)
                    p.drawEllipse(c)
                    p.setPen(QColor("white"))
                    p.drawText(c, Qt.AlignCenter, str(k + 1))
        if self.hover is not None:
            r = self._rect([self.hover])
            if r is not None:
                p.setPen(QPen(HOVER, 2, Qt.DashLine))
                p.setBrush(QColor(29, 111, 209, 25))
                p.drawRoundedRect(r, 4, 4)
        p.end()


def spotlight(main) -> Spotlight:
    sp = getattr(main, "_spotlight", None)
    if sp is None:
        sp = main._spotlight = Spotlight(main)
    return sp


def spot(main, specs, log=None) -> bool:
    """Highlights the targets (list spec, order = numbers ①②③). Not found ones are only written to the log."""
    groups = []
    for s in specs:
        ws = find_widgets(main, s)
        if not ws and log:
            log(f"Guide: element “{s}” not found")
        groups.append(ws)
    spotlight(main).spot(groups)
    return any(groups)


# ------------------------------------------------------------------ explanation of an element
def _form_label(w: QWidget) -> str:
    p = w.parentWidget()
    while p is not None:
        lay = p.layout()
        if lay is not None:
            for fl in [lay] + lay.findChildren(QFormLayout):
                if isinstance(fl, QFormLayout):
                    lbl = fl.labelForField(w)
                    if isinstance(lbl, QLabel):
                        return lbl.text()
        p = p.parentWidget()
    return ""


def _tip(w: QWidget) -> str:
    if isinstance(w, QToolButton) and w.defaultAction() is not None:
        a = w.defaultAction()
        if a.whatsThis():
            return a.whatsThis()
        t = a.toolTip()
        return "" if _clean(t) == _clean(a.text()) else t
    return w.whatsThis() or w.toolTip()


def _title(w: QWidget) -> str:
    if isinstance(w, QToolButton) and w.defaultAction() is not None:
        return _clean(w.defaultAction().text())
    if isinstance(w, QAbstractButton) and w.text():
        return _clean(w.text())
    if isinstance(w, (QGroupBox, Section)):
        return w.title()
    lbl = _form_label(w)
    if lbl:
        return lbl
    if isinstance(w, QLineEdit) and w.placeholderText():
        return w.placeholderText()
    if isinstance(w, QDockWidget):
        return w.windowTitle()
    return ""


_TAG = re.compile(r"</?(?:br|b|i|u|p|ul|ol|li|span|a|div|table|tr|td|th|h[1-6]|code|em|strong|sub|sup|pre)\b[^>]*>",
                  re.I)


def _as_html(text: str) -> str:
    """Text of a label as HTML of a bubble: if it contains tags (also in the middle, e.g. "text<br><b>…</b>"), it is HTML and must not be
    escaped; only plain text is escaped and new lines become <br>.
    """
    if _TAG.search(text):
        return text if text.lstrip().startswith("<") else "<p>" + text + "</p>"
    return "<p>" + html.escape(text).replace("\n", "<br>") + "</p>"


def _stage_of(main, w: QWidget) -> str | None:
    from .main import STAGES
    stack = getattr(main, "panel_stack", None)
    p = w
    while p is not None:
        if stack is not None and p.parentWidget() is stack:
            i = stack.indexOf(p)
            return STAGES[i] if 0 <= i < len(STAGES) else None
        if isinstance(p, QToolButton) and p.defaultAction() in getattr(main, "stage_actions", []):
            return STAGES[main.stage_actions.index(p.defaultAction())]
        p = p.parentWidget()
    return None


def find_explanation(main, w: QWidget | None) -> tuple[QWidget | None, str]:
    """(element, HTML body of the explanation) for the widget under the cursor: the first element with whatsThis / toolTip towards the parents;
    the body is empty when the element has no explanation.
    """
    if w is None:
        return None, ""
    view = getattr(main, "view", None)
    target, body = w, ""
    p = w
    while p is not None:
        if view is not None and p is view:
            target, body = p, tr(VIEW_HELP)
            break
        if isinstance(p.parentWidget(), (QComboBox, QAbstractSpinBox)) or isinstance(p, QTabBar):
            p = p.parentWidget()                   # inner parts of compound elements
            continue
        t = _tip(p)
        field = getattr(getattr(main, "_form_tips", None), "fields", {}).get(p)
        if not t and field is not None:            # label of a form row → description of the field
            t = _tip(field)
        if t:
            target, body = p, _as_html(tr(t))
            break
        if p.isWindow():
            break
        p = p.parentWidget()
    return target, body


def _compose(main, target: QWidget, title: str, body: str, hint: str) -> str:
    """Bubble: step of the progress bar, heading, body and a link to the help page of the step."""
    from .help import STAGE_PAGES, TITLES
    stage = _stage_of(main, target)
    slug = STAGE_PAGES.get(stage, "ovladani") if stage else "ovladani"
    own_head = body.lstrip().startswith("<b>")      # whatsThis starts with its own heading
    title = tr(title) if title else title
    head = f"<b>{html.escape(title)}</b>" if title and not own_head and title.lower() not in body[:120].lower() else ""
    where = tr(f"<span style='color:#6b7785'>Step {stage}</span><br>") if stage else ""
    foot = (f"<p style='margin-top:6px'><a style='color:#1d6fd1' href='fea:{slug}'>Help: {html.escape(TITLES.get(slug, slug))}</a> &nbsp; <span style='color:#6b7785'>{hint}</span></p>")
    return where + head + body + tr(foot)


def explain(main, w: QWidget | None) -> tuple[QWidget | None, str]:
    """(element, HTML explanation) for the widget under the cursor: the first element with whatsThis / toolTip towards the parents."""
    if w is None:
        return None, ""
    view = getattr(main, "view", None)
    target, body = find_explanation(main, w)
    if isinstance(target.parentWidget(), QAbstractScrollArea) and not body:
        target = target.parentWidget()
    title = _title(target) if target is not view else ""
    if not body:
        body = tr("<p>There is no separate description for this element – see the help for the step.</p>")
    return target, _compose(main, target, title, body, "Esc = exit mode")


def _main_of(w: QWidget | None):
    """Main window (it has the attribute whats_this) for the widget; otherwise the first such window of the application."""
    if w is not None and hasattr(w.window(), "whats_this"):
        return w.window()
    for top in QApplication.topLevelWidgets():
        if hasattr(top, "whats_this"):
            return top
    return None


def show_details(anchor: QWidget, html_text: str, title: str = "", at: QPoint | None = None) -> None:
    """Detailed description (level 2) in a bubble below the element `anchor` (or at the point `at`); closed by Esc / a click outside."""
    main = _main_of(anchor)
    if main is None:
        return
    text = _compose(main, anchor, title or _title(anchor), _as_html(tr(html_text)), "Esc = close")
    main.whats_this.bubble.show_closable(text, anchor, at)


def _step_page_widget(main, w: QWidget | None) -> bool:
    """Is w (or its parent) the 3D window, the study tree or the log of the main window? F1 there opens the help page of the step
    (a bubble with the description of the 3D window / tree would cover the work and say nothing more).
    """
    for name in ("view", "tree", "log"):
        v = getattr(main, name, None)
        if isinstance(v, QWidget) and (w is v or v.isAncestorOf(w)):
            return True
    return False


def _help_at(main, w: QWidget | None, bubble, skip_actions):
    """(element, HTML body) for the widget w in any window of the application; (None, "") when no bubble should be shown."""
    if w is None or w is bubble or w.window() is bubble:
        return None, ""
    sp = getattr(main, "_spotlight", None)
    if sp is not None and (w is sp or w.window() is sp):
        return None, ""
    if _step_page_widget(main, w):
        return None, ""
    target, body = find_explanation(main, w)
    if not body:
        return None, ""
    if isinstance(target, QToolButton) and target.defaultAction() in skip_actions:
        return None, ""
    return target, body


def context_help(main, skip_actions=(), focus=None) -> bool:
    """F1: element under the cursor with an explanation → bubble with details (True); if the element under the cursor has no description and the cursor
    is outside the focused window (keyboard work), the description of the focused element; otherwise False (the main window shows the help page
    of the step). Works in all windows of the application (also dialogs); the 3D window, study tree and log return False.
    skip_actions = actions (e.g. the Help button) for which no bubble is made. focus = the widget that received F1 (its window need not be
    the active one – QApplication.focusWidget() would then be an element of the main window).
    """
    pos = QCursor.pos()
    bubble = main.whats_this.bubble
    w = QApplication.widgetAt(pos)
    target, body = _help_at(main, w, bubble, skip_actions)
    anchor = None
    if not body:
        focus = focus if focus is not None else QApplication.focusWidget()
        # cursor over an element of the main window without a description (panel, 3D window, tree…) = help page of the step;
        # focus applies only when the cursor is not over any program window or is in a window other than the focus one
        if focus is not None and (w is None or w.window() is not focus.window()):
            target, body = _help_at(main, focus, bubble, skip_actions)
            anchor = target
    if not body:
        return False
    view = getattr(main, "view", None)
    text = _compose(main, target, _title(target) if target is not view else "", body, "Esc = close")
    bubble.show_closable(text, anchor, None if anchor is not None else pos)
    return True


class HelpKeyFilter(QObject):
    """F1 in other windows of the application (dialogs, non-modal windows): the main window handles it with the Help action (it also shows
    the step page); elsewhere a bubble with the description of the element under the cursor / with the focus. Without a description F1 passes on.
    """

    def __init__(self, main):
        super().__init__(main)
        self.main = main

    def eventFilter(self, obj, ev):
        if ev.type() != QEvent.KeyPress or ev.key() != Qt.Key_F1 or ev.modifiers() != Qt.NoModifier \
                or not isinstance(obj, QWidget):
            return False
        win = obj.window()
        main = self.main
        if win is main or not hasattr(main, "whats_this") or win is main.whats_this.bubble:
            return False
        if getattr(main, "whats_this", None) is not None and main.whats_this.active:
            return False
        return context_help(main, focus=obj)


def install_help_key(main) -> HelpKeyFilter:
    """F1 works in the whole application (calls install_help after main.whats_this is created)."""
    f = getattr(main, "_help_key_filter", None)
    if f is None:
        f = main._help_key_filter = HelpKeyFilter(main)
        QApplication.instance().installEventFilter(f)
    return f


# ------------------------------------------------------------------ short labels (level 1)
_RICH = re.compile(r"\s*<[a-zA-Z!/]")


def _plain(text: str) -> str:
    if _RICH.match(text):
        return QTextDocumentFragment.fromHtml(text).toPlainText().strip()
    return text.strip()


def short_of(text: str, limit: int = TIP_MAX) -> str:
    """Short version of a label: the first line; if longer than the limit, cut at the end of a sentence (". "), otherwise at "; ", ": ",
    " – " (within the range 25…limit), otherwise at the last word; the shortened text ends with "…" (the sentence with a full stop).
    """
    first = _plain(text).split("\n", 1)[0].strip()
    if len(first) <= limit:
        return first
    i = first.rfind(". ", 24, limit)                    # a full stop at position ≥ 24 → the sentence has ≥ 25 characters
    if i >= 0:
        return first[:i + 1]
    best = -1
    for sep in ("; ", ": ", " – "):
        best = max(best, first.rfind(sep, 25, limit))   # the place before the separator; "…" follows it
    if best >= 25:
        return first[:best].rstrip() + "…"
    cut = first[:limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:–-") + "…"


def tip_html(text: str, more: bool) -> str:
    """Tooltip as HTML: short text (short_of) and a grey line "F1 = details", if there is more to show."""
    from . import theme
    # white-space:pre – Qt narrows the rich text of a tooltip and a short label (≤ 90 characters) would wrap into 3–4 lines
    out = "<p style='white-space:pre; margin:0'>" + html.escape(short_of(text)) + "</p>"
    if more:
        hint = tr("F1 = details")
        out += ("<p style='white-space:pre; margin:0; color:" + theme.colors()["muted"] + "'>"
                + html.escape(hint) + "</p>")
    return out


class TipFilter(QObject):
    """Application event filter: a long (> TIP_MAX), multi-line tooltip or the tooltip of an element with whatsThis is shortened to
    one line + "F1 = details". The translation is done before shortening (toolTip() returns the original Czech text). Tooltips
    of table / list items and other tooltips unchanged.
    """

    def eventFilter(self, obj, ev):
        if ev.type() != QEvent.ToolTip or not isinstance(obj, QWidget):
            return False
        raw = obj.toolTip()
        if not raw:
            return False
        t = tr(raw)
        more = bool(obj.whatsThis())
        plain = _plain(t)
        if len(plain) <= TIP_MAX and "\n" not in plain and not more:
            return False
        QToolTip.showText(ev.globalPos(), tip_html(t, True), obj)
        return True


def install_tip_filter(app=None) -> TipFilter | None:
    """Installs the TipFilter on the application (only once, also with several MainWindow instances)."""
    app = app or QApplication.instance()
    if app is None:
        return None
    f = getattr(app, "_fea_tip_filter", None)
    if f is None:
        f = app._fea_tip_filter = TipFilter(app)
        app.installEventFilter(f)
    return f


class Bubble(QLabel):
    """Explanation bubble (white background also in the dark theme, links to the help)."""

    def __init__(self, main):
        super().__init__(main, Qt.ToolTip)
        self.main = main
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self.setOpenExternalLinks(False)
        self.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.setMaximumWidth(460)
        self.setStyleSheet("QLabel { background-color: #ffffff; color: #1f2933; border: 1px solid #1d6fd1; "
                           "border-radius: 4px; padding: 8px; }")
        self.linkActivated.connect(self._link)

    def _link(self, url: str):
        if url.startswith("fea:") and hasattr(self.main, "help_action"):
            self.main.help_action(url[4:])

    def show_closable(self, text: str, anchor: QWidget | None = None, at: QPoint | None = None):
        """Detailed description: below the element anchor (otherwise at the point at); closed by Esc or a click outside the bubble."""
        if anchor is not None and at is None:
            at = anchor.mapToGlobal(QPoint(0, anchor.height() + 4)) - QPoint(16, 20)
        self._attach(anchor)
        self.show_at(at or QCursor.pos(), text)
        if getattr(self, "_closer", None) is None:
            self._closer = _BubbleCloser(self)
        QApplication.instance().installEventFilter(self._closer)

    def _attach(self, anchor: QWidget | None = None):
        """With a modal dialog the bubble is its child – otherwise the modality would leave it without the mouse (links)."""
        modal = QApplication.activeModalWidget()
        win = modal if modal is not None else self.main
        if self.parentWidget() is not win and not self.isVisible():
            self.setParent(win, Qt.ToolTip)

    def hideEvent(self, e):
        c = getattr(self, "_closer", None)
        if c is not None and QApplication.instance() is not None:
            QApplication.instance().removeEventFilter(c)
        super().hideEvent(e)

    def show_at(self, pos: QPoint, text: str):
        self.setText(text)
        self.setFixedWidth(460)
        self.adjustSize()
        scr = (QApplication.screenAt(pos) or QApplication.primaryScreen()).availableGeometry()
        x = min(pos.x() + 16, scr.right() - self.width())
        y = pos.y() + 20
        if y + self.height() > scr.bottom():
            y = max(scr.top(), pos.y() - self.height() - 10)
        self.move(max(scr.left(), x), y)
        self.show()
        self.raise_()


class _BubbleCloser(QObject):
    """Closes the bubble with the details: Esc, a click outside it (clicks still pass through)."""

    def __init__(self, bubble):
        super().__init__(bubble)
        self.bubble = bubble

    def eventFilter(self, obj, ev):
        t = ev.type()
        if not self.bubble.isVisible():
            return False
        if t == QEvent.KeyPress and ev.key() == Qt.Key_Escape:
            self.bubble.hide()
            return True
        if t == QEvent.MouseButtonPress:
            w = QApplication.widgetAt(ev.globalPosition().toPoint()) if hasattr(ev, "globalPosition") else None
            if w is None or not (w is self.bubble or w.window() is self.bubble):
                self.bubble.hide()
        return False


class WhatsThisMode(QObject):
    """"What is this?" mode: application-wide event filter while it is active."""

    def __init__(self, main, action=None):
        super().__init__(main)
        self.main = main
        self.action = action
        self.active = False
        self.bubble = Bubble(main)
        self._last_press = None

    def toggle(self, on: bool | None = None):
        on = (not self.active) if on is None else on
        if on and not self.active:
            self.active = True
            QApplication.instance().installEventFilter(self)
            QApplication.setOverrideCursor(Qt.WhatsThisCursor)
            spotlight(self.main).clear_spots()
            self.main.statusBar().showMessage("What's this?  Click any element of the window – an explanation is shown. Esc or right button = exit.")
        elif not on and self.active:
            self.active = False
            QApplication.instance().removeEventFilter(self)
            QApplication.restoreOverrideCursor()
            spotlight(self.main).set_hover(None)
            self.bubble.hide()
            self.main.statusBar().clearMessage()
        if self.action is not None and self.action.isChecked() != self.active:
            self.action.setChecked(self.active)

    def _ours(self, w: QWidget | None) -> bool:
        return w is not None and (w is self.bubble or w.window() is self.bubble)

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QEvent.KeyPress and ev.key() == Qt.Key_Escape:
            self.toggle(False)
            return True
        if t not in (QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                     QEvent.MouseButtonDblClick, QEvent.Wheel, QEvent.ContextMenu):
            return False
        pos = ev.globalPosition().toPoint() if hasattr(ev, "globalPosition") else QCursor.pos()
        w = QApplication.widgetAt(pos)
        if w is None and isinstance(obj, QWidget):
            w = obj
        if self._ours(w):
            return False                               # links in the bubble
        if t == QEvent.MouseMove:
            target, _ = explain(self.main, w)
            spotlight(self.main).set_hover(target)
            return False
        if t == QEvent.Wheel:
            return False                               # panels can be scrolled in this mode too
        if t == QEvent.MouseButtonPress:
            key = (ev.timestamp(), ev.button())
            if key == self._last_press:                # the same click through QWindow and the widget
                return True
            self._last_press = key
            if ev.button() == Qt.RightButton:
                self.toggle(False)
                return True
            target, text = explain(self.main, w)
            if target is not None:
                spotlight(self.main).set_hover(target)
                self.bubble.show_at(pos, text)
        return True


class FormLabelTips(QObject):
    """The label of a form row without its own toolTip shows the current toolTip of its field (the field changes it according to the load type /
    constraint, so it is not copied once and for all).
    """

    def __init__(self, root: QWidget):
        super().__init__(root)
        self.fields: dict[QLabel, QWidget] = {}
        self.add(root)

    def add(self, root: QWidget) -> None:
        """Form rows of `root` (also a panel built later – the optional step panels are built at their first display)."""
        for fl in root.findChildren(QFormLayout):
            for row in range(fl.rowCount()):
                lbl = fl.itemAt(row, QFormLayout.LabelRole)
                fld = fl.itemAt(row, QFormLayout.FieldRole)
                lw = lbl.widget() if lbl is not None else None
                if not isinstance(lw, QLabel) or lw.toolTip() or fld is None:
                    continue
                fw = fld.widget()
                if fw is None and fld.layout() is not None:      # a row with several fields (x, y, z) → the first with a label
                    lay = fld.layout()
                    ws = [lay.itemAt(i).widget() for i in range(lay.count())]
                    fw = next((w for w in ws if w is not None and w.toolTip()), None)
                if fw is not None:
                    self.fields[lw] = fw
                    lw.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.ToolTip and obj in self.fields and not obj.toolTip():
            fw = self.fields[obj]
            t = fw.toolTip()
            if t:
                t = tr(t)
                more = bool(fw.whatsThis()) or len(_plain(t)) > TIP_MAX or "\n" in _plain(t)
                QToolTip.showText(ev.globalPos(), tip_html(t, more), obj)
                return True
        return False


# ------------------------------------------------------------------ simulation setup guide
def _flow(main):
    """(step colours, errors, warnings) from the last refresh of the progress bar (MainWindow.refresh_flow)."""
    return getattr(main, "flow_state", None) or ({}, [], [])


def _ok(text):
    return True, text


def _todo(text):
    return False, text


def _info(text):
    return None, text


def _check_model(main):
    st = main.study
    if st is None:
        return _todo("No model loaded yet – open a STEP file (or an example from the Tour menu).")
    return _ok(f"Model loaded: {html.escape(st.name)}, number of parts {len(main.parts)}.")


def _check_materials(main):
    st = main.study
    if st is None:
        return _todo("Load a model first.")
    _, errors, _ = _flow(main)
    errs = main.material_errors(errors) if hasattr(main, "material_errors") else []
    if errs:
        return _todo("Missing: " + "; ".join(html.escape(e) for e in errs[:3]))
    used = sorted({st.material_of(p.name).name for p in main.parts if p.name not in st.ignored_parts})
    return _ok("All parts have a material: " + ", ".join(html.escape(n) for n in used) + ".")


def _check_print(main):
    st = main.study
    if st is None:
        return _info("Load a model first.")
    n = [p.name for p in main.parts if st.material_of(p.name).is_print]
    if not n:
        return _info("No part has a printed material – skip the step.")
    return _ok(f"Printed parts: {html.escape(', '.join(n[:6]))} – check the print settings.")


def _check_joints(main):
    st = main.study
    if st is None:
        return _info("Load a model first.")
    if not (st.bolts or st.welds):
        return _info("No bolts and welds (optional). Parts that touch are rigidly connected by the shared mesh.")
    return _ok(f"Bolts: {len(st.bolts)}, welds: {len(st.welds)}.")


def _check_contacts(main):
    st = main.study
    if st is None:
        return _info("Load a model first.")
    if not st.contacts:
        return _info("No contact pairs (optional) – touching parts are rigidly connected by a shared mesh.")
    _, errors, _ = _flow(main)
    errs = [e for e in errors if e.startswith(("Contact", "Sequence", "Step '", "Large deformations"))]
    if errs:
        return _todo(html.escape(errs[0]))
    return _ok(f"Contact pairs: {len(st.contacts)} – the analysis will be nonlinear.")


def _check_dynamics(main):
    st = main.study
    if st is None or not (st.dynamics.enabled or st.masses):
        return _info("No dynamics (optional – vibration, shocks, added mass).")
    _, errors, _ = _flow(main)
    errs = [e for e in errors if e.startswith(("Dynamics", "Harmonic", "Transient", "Random", "Shock",
                                               "Added mass", "Base excitation"))]
    if errs:
        return _todo(html.escape(errs[0]))
    return _ok("Dynamics is defined – check the damping (uncertain input ±50 %).")


def _check_constraints(main):
    st = main.study
    if st is None:
        return _todo("Load a model first.")
    if not st.constraints:
        return _todo("No support yet – the model must be fixed (select faces and add a support).")
    return _ok(f"Supports: {len(st.constraints)}. Whether they prevent all translations and rotations is checked by the Solve step.")


def _check_motion(main):
    st = main.study
    if st is None or not st.mechanism.bodies:
        return _info("No mechanism (optional – motion of bodies, joint reactions, transfer to FEA).")
    if getattr(main, "motion_result", None) is not None:
        return _ok("Motion computed – results and transfer to FEA on the Results tab.")
    m = st.mechanism
    return _todo(f"Bodies {len(m.bodies)}, joints {len(m.joints)}, drivers {len(m.drivers)} – check the mobility and run the motion.")


def _check_thermal(main):
    st = main.study
    if st is None or not st.thermal.cases:
        return _info("No thermal cases (optional – only for heat conduction and thermal stress).")
    return _ok(f"Thermal cases: {len(st.thermal.cases)}.")


def _check_loads(main):
    st = main.study
    if st is None:
        return _todo("Load a model first.")
    empty = [lc.name for lc in st.load_cases if not lc.loads]
    if not st.load_cases or len(empty) == len(st.load_cases):
        return _todo("No load yet – select faces, choose the type and “Add from selection”.")
    _, errors, _ = _flow(main)
    errs = [e for e in errors if e.startswith("State '")]
    if errs:
        return _todo(html.escape(errs[0]))
    if empty:
        return _todo("Empty load case: " + html.escape(", ".join(empty)) + " – complete or delete it.")
    n = sum(len(lc.loads) for lc in st.load_cases)
    return _ok(f"Load cases: {len(st.load_cases)}, loads in total: {n}.")


def _check_mesh(main):
    m = main.mesh
    if m is None:
        return _info("No mesh yet – the analysis will create it, or check it with the “Generate mesh” button.")
    r = getattr(m, "report", None) or {}
    nodes = r.get("nodes")
    return _ok(f"Mesh ready{(f': {nodes:,} nodes'.replace(',', ' ') if nodes else '')}.")


def _check_solve(main):
    if main.study is None:
        return _todo("Load a model first.")
    if main.results is not None:
        return _ok("Analysis finished – continue to the results.")
    _, errors, _ = _flow(main)
    if errors:
        return _todo("Fix before calculation: " + "; ".join(html.escape(e) for e in errors[:3]))
    return _todo("The study is ready – press “Run analysis”.")


def _check_results(main):
    if main.results is None:
        return _todo("No results yet – run the analysis in the Solve step.")
    return _ok("Results are available.")


def _check_fatigue(main):
    fr = getattr(getattr(main, "fat_panel", None), "result", None)
    if fr is not None:
        return _ok("Fatigue analysis done.")
    return _info("Optional – only for variable (repeated) loading.")


def _s(title, text, stage=None, spots=(), check=None, help=None):
    acts = ([("stage", stage)] if stage else []) + ([("spot",) + tuple(spots)] if spots else [])
    return {"title": title, "text": text, "actions": acts, "check": check, "help": help}


SETUP_STEPS = [
    _s("How the program works",
       "<p>The simulation is set up step by step from left to right along the <b>(1)&nbsp;Workflow</b> bar, which is divided into phases: <b>Geometry</b> (model, materials), <b>Assembly</b> (joints, contacts, beams), <b>Conditions</b> (supports, loads, heat, dynamics), <b>Calculation</b> (mesh, solution) and <b>Evaluation</b> (results, fatigue). The phase button opens a menu of all its steps. The colored dot shows the state of the step: <span style='color:#2e9e4f'>●</span> done, <span style='color:#e69a1a'>●</span> warning / optional, <span style='color:#d23b3b'>●</span> error, gray = nothing yet.</p><p>The bar shows only the steps the study uses. <b>Optional steps</b> (3D printing, Contacts, Beams, Motion, Heat, Dynamics, Fatigue) are added with the <b>Analyses</b> menu at the end of the bar; a step also appears by itself as soon as you enter something in it.</p><p>The <b>(2)&nbsp;study tree</b> on the left shows what has been entered (a click switches to the step). The <b>(3)&nbsp;3D window</b> is for selecting faces and displaying results, the <b>(4)&nbsp;step panel</b> on the right for entering values, and the <b>(5)&nbsp;log</b> at the bottom reports progress and errors.</p><p class='tip'><b>Tip:</b> <span class='key'>Shift+F1</span> = <b>What's this?</b> mode – click any button and you will see what it does. <span class='key'>F1</span> over a control shows its detailed description (otherwise the help for the step); controls with the hint “F1 = details” or with the i button have more to show. Panel sections can be collapsed by clicking their heading.</p><p>This guide walks you through setting up <b>your</b> simulation: in each step it highlights the controls (the numbers (1)(2)(3) correspond to the text) and shows below what is already done. “Show in program” repeats the highlighting.</p>",
       None, ["=tb_flow", "tree", "view", "panel_stack", "=dock_log"], None, "index"),
    _s("Model – load geometry",
       "<p>Press <b>(1)&nbsp;Open STEP / project…</b> (or <span class='key'>Ctrl+O</span>) and select a STEP file from your CAD. Each body is one part, units are <b>mm</b>. Part names are taken from the CAD – the program uses them to recognize bolts, nuts, washers and welds.</p><p>In the <b>(2)&nbsp;overview</b> check the number of parts and the volumes; geometry problems (faces without a body, defective bodies) are written to the log.</p><p class='tip'>Want to try the program first? The <b>Tour</b> menu opens a ready-made example.</p>",
       "Model", ["btn_open_model", "model_info"], _check_model, "model"),
    _s("3D window controls and face selection",
       "<p>In the <b>(1)&nbsp;3D window</b>: <b>right button + drag</b> = rotate, <b>middle</b> (or Ctrl + right) = pan, <b>wheel</b> = zoom, <span class='key'>F</span> = fit all, cube at the top right = views.</p><p><b>Left click</b> selects: in the Supports, Heat, Loads and Mesh steps faces (click adds, second click removes, click into empty space clears the selection), in Materials and 3D printing whole parts.</p><p>The <b>(2)&nbsp;selection filter</b> sets what a click selects: <b>Face</b> (P), <b>Body</b> (B), <b>Hole</b> (H – all coaxial cylindrical faces of the hole), <b>Edge</b> (E), <b>Point</b> (V). The status bar at the bottom right shows the number and area of the selected faces.</p><p><b>(3)&nbsp;Section</b> (button above the 3D window) shows the inside of the model (in the results also a coloured section through the mesh).</p>",
       None, ["view", "pick_button", "sec_button"], None, "ovladani"),
    _s("Materials",
       "<p>In the <b>(1)&nbsp;list</b> select parts (Ctrl / Shift = multiple, or click in 3D). In the <b>(2)&nbsp;menu</b> choose a material from the library – the properties (E, ν, ρ, Re, Rm…) are loaded into the form, where you can edit them.</p><p>Don't know the name in the library? Type the grade into the <b>(3)&nbsp;converter</b> (11 373, 1.4301, S355J2, 6061…).</p><p>Then <b>(4)&nbsp;Assign to selected</b> (or Assign to all). A new study has steel as the default material – check it.</p>",
       "Materials", ["mat_parts", "mat_combo", "mat_find", "btn_assign_sel"], _check_materials, "materialy"),
    _s("3D printing (optional – in the Analyses menu)",
       "<p>This step is only for parts with a printed material (“Print PLA…”, “Print PETG…”). In the <b>(1)&nbsp;list</b> select a part and set the perimeters, infill and <b>build direction</b> (printer Z axis) – strength perpendicular to the layers is significantly lower.</p><p>Skip the step if there are no printed parts.</p>",
       "3D printing", ["print_panel.parts"], _check_print, "tisk"),
    _s("Joints – bolts and welds (optional)",
       "<p><b>(1)&nbsp;Detect joints…</b> goes through the parts and suggests bolts (by names and shape) and welds – you confirm the suggestions in a dialog. Bolts are in the <b>(2)&nbsp;bolt list</b>, welds in the <b>(3)&nbsp;weld list</b>; for a bolt set the size, property class and preload.</p><p>Parts that touch in the model are meshed together and are <b>rigidly connected</b> – real contact (separation, friction) is entered in the next step Contacts. Extra parts (magnets, pins) can be omitted from the analysis.</p>",
       "Joints", ["@Detect joints", "joints_panel.bolt_list", "joints_panel.weld_list"], _check_joints, "spoje"),
    _s("Contacts (optional – in the Analyses menu)",
       "<p>Only if parts can separate, slide on each other or are press-fitted. <b>(1)&nbsp;Find touching parts</b> goes through pairs of parts (contact / gap / penetration), add the selected ones as pairs of the chosen type. In the <b>(2)&nbsp;pair table</b> set the type, friction μ (from the library by purpose), interference.</p><p class='warn'>Contact makes the analysis nonlinear: cases are solved incrementally, superposition and fatigue by superposition do not hold and the analysis is slower.</p>",
       "Contacts", ["contact_panel.btn_find", "contact_panel.table"], _check_contacts, "contacts"),
    _s("Supports – fixing the model",
       "<p>In 3D select the faces where the part is fixed (<b>(1)&nbsp;Face</b> filter, for holes <b>Hole</b> H). Selected faces are orange.</p><p><b>(2)&nbsp;Fixed support from selection</b> = the faces do not move. For holes for bolts / pins: <b>(3)&nbsp;Cylindrical support from selection</b>. Symmetry plane: <i>Prescribed displacement</i> in the normal component only. Supports are added to the <b>(4)&nbsp;list</b> and shown in blue in 3D.</p><p class='warn'>The model must not be free to translate or rotate (otherwise the analysis fails) – and the support is ideally rigid: the stress directly at the edge of a fixed support is a singularity.</p>",
       "Constraints", ["pick_button", "btn_fixed", "@Cylindrical support from selection", "con_list"],
       _check_constraints, "constraints"),
    _s("Motion – mechanisms (optional – in the Analyses menu)",
       "<p>Only for mechanisms (crank, lever, guide, lid on hinges). On the <b>Bodies</b> tab assign parts to bodies and mark the ground, on the <b>Joints</b> tab use <b>(1)&nbsp;Suggest from geometry</b> (pin in a hole) or enter them manually, add a driver and run the motion on the Analysis tab. Result: animation, joint reactions, “Transfer to FEA” creates load cases.</p>",
       "Motion", ["motion_panel.tabs", "motion_panel.btn_suggest"], _check_motion, "pohyb"),
    _s("Heat (optional – in the Analyses menu)",
       "<p>Only if you are computing heat conduction or thermal stress. In the <b>(1)&nbsp;menu</b> create a thermal case (steady-state or transient) and add boundary conditions (temperature, flux, convection, radiation, heat generation). You then add thermal stress in Loads with the “Temperature” type.</p><p>Otherwise skip this step.</p>",
       "Heat", ["thermal_panel.case_combo"], _check_thermal, "teplo"),
    _s("Loads",
       "<p><b>(1)&nbsp;Load case</b> = one analysis (+ adds another case, e.g. a different force direction). In 3D select faces (for a force on an edge use the Edge filter, for a point the Point filter), choose the <b>(2)&nbsp;type</b> – force, pressure, moment, bearing load, gravity, rotation, temperature – and fill in the <b>(3)&nbsp;values</b> (the force is total, not per area).</p><p><b>(4)&nbsp;Add from selection</b> – the load is added to the <b>(5)&nbsp;list</b> and shown as arrows in 3D (check the direction!).</p>",
       "Loads", ["lc_combo", "ld_type", "ld_vec|ld_val", "btn_add_load", "load_list"], _check_loads, "zatizeni"),
    _s("Dynamics (optional – in the Analyses menu)",
       "<p>Structural vibration: <b>(1)&nbsp;switch on dynamics</b>, enter <b>(2)&nbsp;damping</b> (estimate or from measurement) and choose a response – harmonic (sine), transient (in time), random vibration (ASD) or shock. Added mass (motor, electronics) is entered there too.</p><p class='warn'>Damping is an uncertain input (±50 %) – the result at resonance is inversely proportional to it.</p>",
       "Dynamics", ["dynamics_panel.on", "dynamics_panel.d_zeta"], _check_dynamics, "dynamika"),
    _s("Mesh",
       "<p><b>(1)&nbsp;Element size</b> determines accuracy: smaller = more accurate but slower. The default is ~1/30 of the model size. For thin walls use at least 2 elements through the thickness.</p><p>At stress concentrations (holes, notches, transitions) use <b>(2)&nbsp;local refinement</b> of selected faces. <b>(3)&nbsp;Generate mesh</b> shows the mesh – it is optional, the analysis creates it by itself.</p><p class='tip'>Verify reliability by convergence: reduce the element size by ~30 % and compare the results.</p>",
       "Mesh", ["ms_size", "ref_size|@Refine selection", "btn_mesh"], _check_mesh, "sit"),
    _s("Solution – running the calculation",
       "<p><b>(1)&nbsp;Check</b> lists errors (red – the analysis cannot be run) and warnings (orange), including a check that the supports prevent all motions. Leave <b>(2)&nbsp;Solver</b> on <i>Automatic</i>.</p><p><b>(3)&nbsp;Run analysis</b> – progress is shown in the log below. At the end the program verifies the equilibrium of reactions and loads and opens the results.</p>",
       "Solve", ["ready_list", "backend_combo", "btn_run"], _check_solve, "reseni"),
    _s("Results – evaluation",
       "<p>Choose the <b>(1)&nbsp;load case</b> and the <b>(2)&nbsp;quantity</b>: compare <i>von Mises</i> with the yield strength, <i>Safety factor to Re</i> shows the margin directly (≥ 1.5 is a common requirement), <i>Displacement</i> the stiffness. <b>(3)&nbsp;Deformation</b> is only a magnification for display.</p><p>A click in the model shows the value at the point (probe). The <b>(4)&nbsp;Highest stresses</b> tab distinguishes real maxima from <b>singularities</b> (fixed support edge, sharp corner, point load) – do not evaluate those by the nodal value. The summary contains the support reactions and the equilibrium check.</p>",
       "Results", ["r_step", "r_qty", "r_scale", "res_tabs"], _check_results, "vysledky"),
    _s("Fatigue (optional – in the Analyses menu)",
       "<p>For repeated loading: in the <b>(1)&nbsp;channels</b> enter how the load cases vary over time (points, sine, CSV), set the method and S-N curves and <b>(2)&nbsp;Run fatigue analysis</b> – the result is damage, life and safety factor.</p>",
       "Fatigue", ["fat_panel.ch_list", "fat_panel.btn"], _check_fatigue, "unava"),
    _s("Report and saving",
       "<p>In the <b>(1)&nbsp;File</b> menu: <b>Save</b> (<span class='key'>Ctrl+S</span>) saves the .fea project (geometry, setup and results), <b>Report…</b> (<span class='key'>Ctrl+P</span>) creates a technical report in Word with the input, result maps, assessments and conclusion.</p><p>Done. You can return to the guide at any time from the <b>Help → How to set up a simulation</b> menu.</p>",
       None, ["menu_file"], None, "protokol"),
]
