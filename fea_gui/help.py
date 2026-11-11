"""FEA Studio help: HTML pages in fea_gui/help/ (one per step of the progress bar + overview, controls, examples).

F1 opens the page of the current step (`STAGE_PAGES`); the window is non-modal and single for the whole application.
The pages are HTML fragments without <style> – the look comes from `STYLE` (QTextBrowser supports only part of CSS 2).
"""
from __future__ import annotations

import re
from html import unescape
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from .safe_browser import SafeTextBrowser
HELP_DIR = Path(__file__).resolve().parent / "help"

# (slug, name in the contents) – order of the contents
PAGES = [
    ("index", "Overview and workflow"),
    ("ukazky", "Examples and tour"),
    ("testovani", "Testing (problems with known results)"),
    ("ovladani", "Controls and keys"),
    ("model", "1. Model"),
    ("cad", "Export from CAD (SolidWorks, Inventor, Fusion…)"),
    ("materialy", "2. Materials"),
    ("tisk", "3. 3D printing"),
    ("spoje", "4. Joints (bolts, welds)"),
    ("contacts", "5. Contacts"),
    ("friction", "Friction and damping (values with sources)"),
    ("members", "6. Members (frames, trusses, beams)"),
    ("constraints", "7. Supports"),
    ("pohyb", "8. Motion (mechanisms)"),
    ("teplo", "9. Heat"),
    ("zatizeni", "10. Loads"),
    ("dynamika", "11. Dynamics"),
    ("sit", "12. Mesh"),
    ("reseni", "13. Solve"),
    ("2d", "2D section analysis"),
    ("shell", "Shell model"),
    ("vysledky", "14. Results"),
    ("voxel", "Print detail (voxels)"),
    ("unava", "15. Fatigue"),
    ("protokol", "Word report"),
    ("automatizace", "Automation (CLI, MCP, REST, CAD)"),
]
TITLES = dict(PAGES)

# step of the progress bar (main.STAGES) → page
STAGE_PAGES = {"Model": "model", "Materials": "materialy", "3D printing": "tisk", "Joints": "spoje", "Contacts": "contacts",
               "Members": "members", "Dynamics": "dynamika",
               "Constraints": "constraints", "Motion": "pohyb",
               "Heat": "teplo", "Loads": "zatizeni", "Mesh": "sit", "Solve": "reseni", "Results": "vysledky",
               "Fatigue": "unava"}

# help is always light (also in the dark theme of the program) – the colours of the style assume a white background
BROWSER_QSS = "QTextBrowser { background-color: #ffffff; color: #1f2933; }"

STYLE = """
body { font-size: 10pt; color: #1f2933; }
h1 { font-size: 17pt; color: #1d4e89; margin-bottom: 4px; }
h2 { font-size: 13pt; color: #1d4e89; margin-top: 16px; border-bottom: 1px solid #c9d3df; }
h3 { font-size: 11pt; color: #33475b; margin-top: 10px; }
p.lead { font-size: 11pt; color: #44515f; }
p.tip { color: #1f2933; background-color: #e8f4ea; border-left: 4px solid #2e9e4f; padding: 6px; }
p.warn { color: #1f2933; background-color: #fdf1e1; border-left: 4px solid #e69a1a; padding: 6px; }
span.key { font-family: Consolas, monospace; background-color: #eef1f5; border: 1px solid #b8c2cc; padding: 1px 4px; }
code, pre { font-family: Consolas, monospace; background-color: #f3f5f8; }
table.grid { border-collapse: collapse; margin: 6px 0px; }
table.grid th { background-color: #e3eaf3; padding: 4px; text-align: left; }
table.grid td { padding: 4px; vertical-align: top; }
a { color: #1d6fd1; }
"""


def page_path(slug: str) -> Path:
    """Page in the language of the program. The source is English (help/<slug>.html), Czech is the translation (help/cs/<slug>.html)."""
    from fea_core import i18n
    if i18n.english() and (HELP_DIR / f"{slug}.html").exists():
        return HELP_DIR / f"{slug}.html"
    return HELP_DIR / "cs" / f"{slug}.html"


def page_html(slug: str) -> str:
    p = page_path(slug)
    if not p.exists():
        return f"<h1>{TITLES.get(slug, slug)}</h1><p>The help page is not available yet.</p>"
    return p.read_text(encoding="utf-8")


def plain_text(html: str) -> str:
    return unescape(re.sub(r"<[^>]+>", " ", html))


def search(text: str) -> list[tuple[str, int]]:
    """Pages containing all words of the query (case-insensitive), sorted by the number of occurrences."""
    words = [w for w in text.lower().split() if w]
    out = []
    for slug, _ in PAGES:
        body = plain_text(page_html(slug)).lower()
        if words and all(w in body for w in words):
            out.append((slug, sum(body.count(w) for w in words)))
    return sorted(out, key=lambda t: -t[1])


class HelpWindow(QDialog):
    """Non-modal help window: contents + search on the left, page on the right, back / forward."""

    def __init__(self, parent=None, on_action=None):
        super().__init__(parent)
        self.setWindowTitle("Help – FEA Studio")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(1050, 760)
        self.on_action = on_action        # links "fea:…" (e.g. fea:ukazka1, fea:pruvodce) → main window

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search help…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._filter)
        self.toc = QListWidget()
        self.toc.itemClicked.connect(lambda it: self.show_page(it.data(Qt.UserRole)))
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.search_edit)
        lv.addWidget(self.toc)

        self.browser = SafeTextBrowser(allowed_dirs=[HELP_DIR])
        from .i18n import raw
        raw(self.browser)                       # the pages are already in the language of the program (help/ or help/cs/)
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._link)
        self.browser.document().setDefaultStyleSheet(STYLE)
        self.browser.setStyleSheet(BROWSER_QSS)
        self.browser.setSearchPaths([str(HELP_DIR), str(HELP_DIR / "cs")])
        self.back_btn = QPushButton("◄ Back")
        self.fwd_btn = QPushButton("Forward ►")
        home = QPushButton("Overview")
        self.back_btn.clicked.connect(self.back)
        self.fwd_btn.clicked.connect(self.forward)
        home.clicked.connect(lambda: self.show_page("index"))
        nav = QHBoxLayout()
        for b in (self.back_btn, self.fwd_btn, home):
            nav.addWidget(b)
        nav.addStretch(1)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addLayout(nav)
        rv.addWidget(self.browser)

        sp = QSplitter()
        sp.addWidget(left)
        sp.addWidget(right)
        sp.setSizes([230, 820])
        lay = QVBoxLayout(self)
        lay.addWidget(sp)
        QShortcut(QKeySequence.Back, self, self.back)
        QShortcut(QKeySequence.Forward, self, self.forward)
        QShortcut(QKeySequence.Find, self, self.search_edit.setFocus)

        self.history: list[str] = []
        self.pos = -1
        self.current = ""
        self._filter("")

    # ------------------------------------------------------------ obsah
    def _filter(self, text: str):
        self.toc.clear()
        hits = search(text) if text.strip() else [(s, 0) for s, _ in PAGES]
        for slug, n in hits:
            it = QListWidgetItem(TITLES[slug] + (f"  ({n}×)" if n else ""))
            it.setData(Qt.UserRole, slug)
            self.toc.addItem(it)
            if slug == self.current:
                self.toc.setCurrentItem(it)
        if text.strip() and self.current:
            self._highlight(text)

    def _highlight(self, text: str):
        word = text.split()[0] if text.split() else ""
        if word:
            self.browser.moveCursor(self.browser.textCursor().MoveOperation.Start)
            self.browser.find(word)

    # ------------------------------------------------------------ pages
    def show_page(self, slug: str, anchor: str = "", record: bool = True):
        if slug not in TITLES:
            slug = "index"
        self.current = slug
        self.browser.setSearchPaths([str(page_path(slug).parent)])   # images from the folder of the page (cs / en)
        self.browser.setHtml(page_html(slug))
        if anchor:
            self.browser.scrollToAnchor(anchor)
        if record:
            del self.history[self.pos + 1:]
            self.history.append(slug)
            self.pos = len(self.history) - 1
        for i in range(self.toc.count()):
            if self.toc.item(i).data(Qt.UserRole) == slug:
                self.toc.setCurrentRow(i)
        if self.search_edit.text().strip():
            self._highlight(self.search_edit.text())
        self.back_btn.setEnabled(self.pos > 0)
        self.fwd_btn.setEnabled(self.pos < len(self.history) - 1)

    def back(self):
        if self.pos > 0:
            self.pos -= 1
            self.show_page(self.history[self.pos], record=False)

    def forward(self):
        if self.pos < len(self.history) - 1:
            self.pos += 1
            self.show_page(self.history[self.pos], record=False)

    def _link(self, url: QUrl):
        s = url.toString()
        if s.startswith("fea:"):                  # actions in the main window (open an example, start the guide…)
            if self.on_action is not None:
                self.on_action(s[4:])
            return
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
            return
        name, _, anchor = s.partition("#")
        slug = Path(name).stem if name else self.current
        self.show_page(slug, anchor)


_window: HelpWindow | None = None


def show_help(parent=None, slug: str = "index", on_action=None) -> HelpWindow:
    """Opens (or brings to the front) the help window on page slug."""
    global _window
    if _window is None or _window.parent() is not parent:
        _window = HelpWindow(parent, on_action)
    elif on_action is not None:
        _window.on_action = on_action
    _window.show_page(slug)
    _window.show()
    _window.raise_()
    _window.activateWindow()
    return _window
