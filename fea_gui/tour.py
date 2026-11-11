"""Example guide: a dock with steps (text + "Show in the program") that goes through the example project step by step.

The steps are data (`fea_gui/tour_steps.py`): name, HTML text, help page and a list of actions. The actions are tuples:
  ("open", "example2_assembly.fea")      opens the example (a copy in the user folder, the original stays)
  ("stage", "Constraints")             switches the step of the progress bar
  ("combo", "r_qty", "von Mises")      in a QComboBox (path to an attribute of the main window, also "print_panel.x") selects
                                       the first item containing the text (case-insensitive; "a|b" = either of the alternatives)
  ("list", "print_panel.parts", "Rameno")  in a QListWidget selects the first row containing the text
  ("tab", "motion_panel.tabs", "Joints")  in a QTabWidget switches to the first tab whose name contains the text
  ("check", "r_edges", True)           ticks / unticks a QCheckBox
  ("call", "view.fit_all")             calls a method without arguments
  ("help", "tisk")                     opens the help on the page
  ("solve",)                           runs the calculation (only when the study has no results yet)
  ("need_mesh",) / ("need_results",)   without a mesh / results writes a request to calculate to the log and skips the other actions of the step
  ("spot", "btn_run", "r_qty", ...)   highlights the elements of the window with numbers 1, 2, 3 (targets see fea_gui.guide.find_widgets)
A step may have a "check": function(main) → (done True / missing False / optional None, HTML) – the live state below the text
(simulation setup guide, fea_gui.guide.SETUP_STEPS), refreshed after every change of the study (main.flowChanged).
A missing attribute or item is only written to the log (the guide must not crash the program after a GUI change).
The actions "combo", "list", "tab", "check" and "spot" first expand the collapsed sections (`widgets.Section.reveal`) in which
the target element lies – otherwise it would be invisible.
"""
from __future__ import annotations

from html import escape
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QHBoxLayout, QLabel, QPushButton, QTextBrowser,
                               QVBoxLayout, QWidget)

from .safe_browser import SafeTextBrowser
from .help import BROWSER_QSS, STYLE, show_help
from .widgets import Section, help_tip

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def user_examples_dir() -> Path:
    """Working copies of the examples (%LOCALAPPDATA%/FEAStudio/ukazky) – edits and Ctrl+S do not overwrite the original."""
    from fea_core.config import user_data_dir
    base = user_data_dir() / "ukazky"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _has_results(path: Path) -> bool:
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return "results.npz" in z.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def example_copy(name: str, parent=None) -> Path:
    """Working copy of an example. If the copy from the past has calculated results (the examples are shipped without results – size),
    it offers to continue with them; otherwise the clean example is copied.
    """
    src = EXAMPLES_DIR / name
    if not src.exists():
        raise FileNotFoundError(f"Example {name} is missing in folder {EXAMPLES_DIR} (create it with `python -m examples.create`)")
    dst = user_examples_dir() / name
    if dst.exists() and _has_results(dst) and not _has_results(src) and dst.stat().st_mtime >= src.stat().st_mtime:
        from PySide6.QtWidgets import QMessageBox
        r = QMessageBox.question(parent, "Example tour",
                                 "The example already has results from last time.\n\nYes = open with the results\nNo = start from the clean example (analysis in the Solve step)")
        if r == QMessageBox.Yes:
            return dst
    shutil.copy2(src, dst)
    return dst


def _resolve(main, path: str):
    obj = main
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _resolve_shown(main, path: str):
    """As _resolve, but a widget in a collapsed section is expanded (so that it is visible and can be operated)."""
    obj = _resolve(main, path)
    if isinstance(obj, QWidget):
        Section.reveal(obj)
    return obj


def run_action(main, act: tuple, log=None) -> bool:
    """Performs one action of the guide. Returns False when it failed (the reason goes to the log)."""
    from .main import STAGE
    log = log or (lambda s: None)
    kind, *args = act
    try:
        if kind == "open":
            main.open_file(example_copy(args[0], main))
            return True
        if kind == "solve":                       # run the calculation, only when there are still no results
            if getattr(main, "results", None) is None and callable(getattr(main, "run_solve", None)):
                main.run_solve()
            return True
        if kind == "need_mesh":                   # mesh step (section through a shell) without a mesh → warn, not a window
            if getattr(main, "mesh", None) is None:
                log("Guide: no mesh yet – create it in the Mesh step (or run the analysis)")
                return False
            return True
        if kind == "need_results":                # results step without results → warn
            if getattr(main, "results", None) is None:
                log("Guide: no results yet – run the analysis in the Solve step (Run button)")
                return False
            return True
        if kind == "stage":
            if args[0] not in STAGE:
                log(f"Guide: unknown step “{args[0]}”")
                return False
            main.set_stage(STAGE[args[0]])
            return True
        if kind == "combo":
            w = _resolve_shown(main, args[0])
            if not isinstance(w, QComboBox):
                log(f"Guide: selection “{args[0]}” not found")
                return False
            wants = [x.strip().lower() for x in args[1].split("|")]         # alternatives: stored name | recomputed name
            for i in range(w.count()):
                if any(x in w.itemText(i).lower() for x in wants):
                    w.setCurrentIndex(i)
                    return True
            log(f"Guide: menu “{args[0]}” is missing “{args[1]}”")
            return False
        if kind == "list":                        # QListWidget: selects the first row containing the text
            w = _resolve_shown(main, args[0])
            if w is None or not hasattr(w, "findItems"):
                log(f"Guide: list “{args[0]}” not found")
                return False
            items = w.findItems(args[1], Qt.MatchContains)
            if not items:
                log(f"Guide: list “{args[0]}” is missing “{args[1]}”")
                return False
            w.clearSelection()
            w.setCurrentItem(items[0])
            items[0].setSelected(True)
            return True
        if kind == "tab":                         # QTabWidget: tab by name (Czech and translated text)
            from PySide6.QtWidgets import QTabWidget
            from fea_core.i18n import tr
            w = _resolve_shown(main, args[0])
            if not isinstance(w, QTabWidget):
                log(f"Guide: tabs “{args[0]}” not found")
                return False
            wants = {args[1].lower(), tr(args[1]).lower()}
            for i in range(w.count()):
                if any(x in w.tabText(i).lower() for x in wants):
                    w.setCurrentIndex(i)
                    return True
            log(f"Guide: tabs “{args[0]}” have no “{args[1]}”")
            return False
        if kind == "check":
            w = _resolve_shown(main, args[0])
            if not isinstance(w, QCheckBox):
                log(f"Guide: option “{args[0]}” not found")
                return False
            w.setChecked(bool(args[1]))
            return True
        if kind == "call":
            f = _resolve(main, args[0])
            if not callable(f):
                log(f"Guide: action “{args[0]}” not found")
                return False
            f()
            return True
        if kind == "help":
            show_help(main, args[0], on_action=getattr(main, "help_action", None))
            return True
        if kind == "spot":
            from .guide import spot
            return spot(main, list(args), log)
    except Exception as e:  # noqa: BLE001 – the guide must not crash the program
        log(f"Guide: action {act} failed: {e}")
        return False
    log(f"Guide: unknown action {kind}")
    return False


class TourDock(QDockWidget):
    """Dock of the guide: heading, counter, text of the step and the buttons Back / Show in the program / Next."""

    def __init__(self, main, tours: dict[str, list[dict]], title: str = "Example tour", name: str = "tour_dock"):
        super().__init__(title, main)
        self.setObjectName(name)
        self.main = main
        self.tours = tours
        self.steps: list[dict] = []
        self.k = 0
        w = QWidget()
        v = QVBoxLayout(w)
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setTextFormat(Qt.RichText)
        self.counter = QLabel()
        self.counter.setStyleSheet("color: #6b7785;")
        self.text = SafeTextBrowser(allowed_dirs=[Path(__file__).resolve().parent / "help"])
        from .i18n import raw
        raw(self.text)                          # the guide translates the step text itself (tr)
        self.text.setOpenExternalLinks(False)
        self.text.setOpenLinks(False)
        self.text.anchorClicked.connect(self._link)
        self.text.document().setDefaultStyleSheet(STYLE)
        self.text.setStyleSheet(BROWSER_QSS)
        self.status = QLabel()                    # live check of the step (check)
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.RichText)
        self.status.setVisible(False)
        self.status.setToolTip("What is done in this step and what is still missing (refreshed after every change of the study)")
        self.prev_btn = QPushButton("◄ Back")
        self.show_btn = QPushButton("Show in app")
        self.next_btn = QPushButton("Next ►")
        self.help_btn = QPushButton("Help for the step")
        help_tip(self.show_btn, "Sets the program to the state of the step and highlights its controls",
                 "Puts the program into the state the step describes (workflow step, view…)\nand highlights the controls the step talks about")
        self.prev_btn.setToolTip("Previous guide step")
        self.next_btn.setToolTip("Next guide step")
        self.help_btn.setToolTip("Open the help page for this step")
        self.prev_btn.clicked.connect(lambda: self.go(self.k - 1))
        self.next_btn.clicked.connect(lambda: self.go(self.k + 1))
        self.show_btn.clicked.connect(self.apply)
        self.help_btn.clicked.connect(self._help)
        h = QHBoxLayout()
        for b in (self.prev_btn, self.show_btn, self.next_btn):
            h.addWidget(b)
        v.addWidget(self.title)
        v.addWidget(self.counter)
        v.addWidget(self.text, 1)
        v.addWidget(self.status)
        v.addLayout(h)
        v.addWidget(self.help_btn)
        self.setWidget(w)
        self.setMinimumWidth(360)
        if hasattr(main, "flowChanged"):
            main.flowChanged.connect(self.update_status)

    def start(self, name: str):
        self.steps = self.tours[name]
        self.setWindowTitle(name if name.startswith("Guide") else f"Guide – {name}")
        self.show()
        self.raise_()
        self.go(0, apply=True)

    def go(self, k: int, apply: bool = True):
        if not self.steps:
            return
        self.k = max(0, min(k, len(self.steps) - 1))
        s = self.steps[self.k]
        self.title.setText(f"<h3>{s['title']}</h3>")
        self.counter.setText(f"Step {self.k + 1} / {len(self.steps)}")
        from fea_core.i18n import tr                # the QTextBrowser wrapper i18n does not translate (help pages) → explicitly
        self.text.setHtml(tr(s["text"]))
        self.prev_btn.setEnabled(self.k > 0)
        self.next_btn.setEnabled(self.k < len(self.steps) - 1)
        self.show_btn.setEnabled(bool(s.get("actions")))
        self.help_btn.setEnabled(bool(s.get("help")))
        self.update_status()
        if apply:
            self.apply()

    def update_status(self):
        s = self.steps[self.k] if self.steps else {}
        check = s.get("check")
        self.status.setVisible(check is not None)
        if check is None or not self.isVisible():
            return
        try:
            state, text = check(self.main)
        except Exception as e:  # noqa: BLE001 – the guide must not crash the program
            state, text = None, f"The state cannot be determined: {e}"
        mark, col, bg = {True: ("√ Done", "#2e9e4f", "#e8f4ea"), False: ("→ Remaining", "#c0392b", "#fdecea")}.get(
            state, ("i Optional", "#6b7785", "#eef1f5"))
        self.status.setStyleSheet(f"QLabel {{ background-color: {bg}; color: #1f2933; border-left: 4px solid {col};"
                                  " padding: 6px; }")
        self.status.setText(f"<b style='color:{col}'>{mark}:</b> {escape(str(text))}")

    def apply(self):
        s = self.steps[self.k] if self.steps else {}
        log = getattr(self.main, "log_msg", None)
        for act in s.get("actions", []):
            if not run_action(self.main, tuple(act), log) and act[0] in ("open", "need_results", "need_mesh"):
                break

    def _help(self):
        s = self.steps[self.k] if self.steps else {}
        if s.get("help"):
            run_action(self.main, ("help", s["help"]))

    def _link(self, url):
        s = url.toString()
        if s.startswith("fea:") and hasattr(self.main, "help_action"):
            self.main.help_action(s[4:])
        elif s.endswith(".html"):
            run_action(self.main, ("help", Path(s).stem))


# ------------------------------------------------------------------ hooking into the main window
def install_help(main) -> None:
    """Help bar (F1 = page of the current step, What is this? Shift+F1, simulation setup guide, example guide,
    manual) + the guide docks + main.help_action. In MainWindow it is enough to call
    install_help(self) at the end of __init__.
    """
    from PySide6.QtGui import QAction, QKeySequence
    from PySide6.QtWidgets import QMenu, QToolBar, QToolButton
    from .guide import (SETUP_STEPS, FormLabelTips, WhatsThisMode, context_help, install_help_key,
                        install_tip_filter)
    from .help import STAGE_PAGES
    from .main import STAGES
    from .tour_steps import TOURS

    main.tour_dock = TourDock(main, TOURS)
    main.addDockWidget(Qt.RightDockWidgetArea, main.tour_dock)
    main.tour_dock.hide()
    setup_name = "Simulation setup guide"
    main.guide_dock = TourDock(main, {setup_name: SETUP_STEPS}, setup_name, "guide_dock")
    main.addDockWidget(Qt.RightDockWidgetArea, main.guide_dock)
    main.guide_dock.hide()
    main._form_tips = FormLabelTips(main)
    install_tip_filter()                        # long tooltips → one line + "F1 = details"
    names = list(TOURS)

    def help_action(name: str) -> None:
        """fea:… links from the help and the guide."""
        if name.startswith("pruvodce") and name[8:].isdigit() and 0 < int(name[8:]) <= len(names):
            main.tour_dock.start(names[int(name[8:]) - 1])
        elif name.startswith("ukazka") and name[6:].isdigit():
            files = [t[0]["actions"][0][1] for t in TOURS.values()]
            k = int(name[6:]) - 1
            if 0 <= k < len(files):
                run_action(main, ("open", files[k]), main.log_msg)
        elif name == "prirucka":
            open_manual(main, "manual")
        elif name == "testovani_pdf":
            open_manual(main, "testing")
        elif name == "prirucka_word":
            export_manual(main)
        elif name == "nastaveni":                  # guide for setting up one's own simulation
            main.tour_dock.hide()
            main.guide_dock.start(setup_name)
        elif name == "cojeto":
            main.whats_this.toggle(True)
        elif name.startswith("spot:"):             # link to an element of the window → highlight
            run_action(main, ("spot", name[5:]), main.log_msg)
        else:
            run_action(main, ("help", name))
    main.help_action = help_action

    def current_help():
        """F1: element under the mouse with a detailed description → bubble; otherwise the help page of the current step."""
        if context_help(main, skip_actions=(help_act,)):
            return
        stage = STAGES[getattr(main, "stage", 0)]
        show_help(main, STAGE_PAGES.get(stage, "index"), on_action=help_action)

    from . import theme
    menu = getattr(main, "help_menu", None)      # Help menu in the header (main._build_toolbars)
    if menu is None:
        btn = QToolButton()
        btn.setText("Help")
        btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(btn)
        btn.setMenu(menu)
        tb = QToolBar("Help", main)
        tb.setObjectName("tb_help")
        tb.addWidget(btn)
        main.addToolBar(tb)
    a = help_act = QAction("Help", main)
    theme.set_icon(a, "help")
    a.setShortcut(QKeySequence("F1"))
    a.setToolTip("Help: the control under the mouse, otherwise the step  [F1]")
    a.triggered.connect(current_help)
    main.addAction(a)
    menu.addAction(a)
    w = QAction("What's this?", main)
    w.setCheckable(True)
    theme.set_icon(w, "whatsthis")
    help_tip(w, "What's this?: click a control to see what it does  [Shift+F1]",
             "What's this? mode: click any element of the window to see what it does and how to use it.\nEsc or right button = exit  [Shift+F1]")
    w.setShortcut(QKeySequence("Shift+F1"))
    main.whats_this = WhatsThisMode(main, w)
    install_help_key(main)                      # F1 = description of the element also in dialogs and other windows
    w.triggered.connect(lambda on: main.whats_this.toggle(on))
    main.addAction(w)
    menu.addAction(w)
    g = QAction("How to set up a simulation", main)
    theme.set_icon(g, "guide")
    help_tip(g, "Step-by-step guide to setting up your simulation",
             "An interactive step-by-step guide to setting up your simulation: highlights controls\nand shows what is done and what is still missing")
    g.triggered.connect(lambda: help_action("nastaveni"))
    menu.addAction(g)
    menu.addSeparator()
    tours = menu.addMenu(theme.icon("tour"), "Example tour")
    tours.setToolTipsVisible(True)
    for k, n in enumerate(names, 1):
        tours.addAction(n, lambda k=k: help_action(f"pruvodce{k}"))
    menu.addSeparator()
    menu.addAction("Manual (PDF)", lambda: help_action("prirucka"))
    menu.addAction("Testing guide (PDF)", lambda: help_action("testovani_pdf"))
    menu.addAction("Manual as Word + HTML…", lambda: help_action("prirucka_word"))
    menu.addSeparator()
    ab = QAction("About FEA Studio…", main)
    ab.setToolTip("Version, licence (GPL-3.0-or-later, no warranty), licence texts, log folder")
    ab.setMenuRole(QAction.AboutRole)

    def about():
        from .about import show_about
        show_about(main)
    ab.triggered.connect(about)
    menu.addAction(ab)
    main.about_action = ab
    main.help_menu = menu


def open_manual(main, kind: str = "manual") -> None:
    """Opens the PDF of the manual / testing guide in the program language (docs/ from the installation). If missing, it creates it
    (fea_gui.help_pdf) in the user data folder (FEAStudio/docs).
    """
    from fea_core.config import user_data_dir
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QApplication, QMessageBox
    from fea_core import i18n
    from . import help_pdf
    name = help_pdf.TEXTS[i18n.language()][f"file_{kind}"]
    path = Path(__file__).resolve().parents[1] / "docs" / name
    if not path.exists():
        path = user_data_dir() / "docs" / name
        if not path.exists():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                help_pdf.export_pdf(path, i18n.language(), kind)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(main, "Manual", f"The manual cannot be generated:\n{e}")
                return
            finally:
                QApplication.restoreOverrideCursor()
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def export_manual(main) -> None:
    """Saves the manual (Word + HTML) to the chosen folder and opens it."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from .help_export import export_docx, export_html
    start = main.settings.value("last_dir", "") if hasattr(main, "settings") else ""
    d = QFileDialog.getExistingDirectory(main, "Folder for the manual", str(start or ""))
    if not d:
        return
    try:
        docx = export_docx(Path(d) / "FEA Studio Manual.docx")
        export_html(Path(d) / "FEA Studio Manual.html")
    except Exception as e:  # noqa: BLE001
        QMessageBox.warning(main, "Manual", f"The manual cannot be generated:\n{e}")
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(docx)))
