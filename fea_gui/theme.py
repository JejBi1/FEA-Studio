"""Appearance of FEA Studio: Fusion style + own palette (light / dark / by system), QSS style and line icons.

- `apply(app, mode)` – mode "system" | "light" | "dark"; can be called repeatedly (switching at run time). Returns True = dark.
- `icon(name)` – icon from `ICONS` (SVG 24×24, lines 1.8 px, drawn for this program) in the colours of the current theme:
  normal = text, on (checked) = highlight colour, unavailable = grey.
- `step_icon(n, color)` – step number in a coloured circle (progress bar: colour = state of the step).
- Panel heading = `widgets.panel_header`, the primary button of a panel = property `primary` (`set_primary`).
The arrows of menus and numeric fields are SVG files in a temporary folder (QSS can use images only from a file / resource).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

MODES = {"system": "Follow system", "light": "Light", "dark": "Dark"}

# Colours of the "Studio × Tables" design (dense engineering program): bg = window background, surface = panels and tables, bar = bars,
# status bar and dock headers, header = section headers, line = dividing lines between areas, border = element frames,
# button = normal button, sel = selected row / item.
DARK = dict(bg="#1e1e1e", surface="#262626", bar="#2b2b2b", header="#313131", line="#161616", input="#1c1c1c",
            alt="#2a2a2a", border="#3c3c3c", hover="#333333", pressed="#3a3a3a", button="#383838",
            button_border="#4a4a4a", text="#d8d8d8", text_hi="#f0f0f0", muted="#9c9c9c", disabled="#666666",
            accent="#4f8ae0", accent_hover="#6a9ce6", accent_soft="#2c3a52", accent_soft_text="#a9c6f5",
            sel="#3d5378", sel_text="#ffffff", on_accent="#ffffff", scroll="#4a4a4a", tip_bg="#333333",
            tip_text="#e4e4e4", ok="#7fcf8a", fail="#ff7a6a", view_top=(74, 77, 82), view_bottom=(42, 44, 47))
LIGHT = dict(bg="#e9e9e9", surface="#f4f4f4", bar="#e4e4e4", header="#e9e9e9", line="#c9c9c9", input="#ffffff",
             alt="#fafafa", border="#c9c9c9", hover="#dcdcdc", pressed="#d0d0d0", button="#ffffff",
             button_border="#c4c4c4", text="#1f1f1f", text_hi="#111111", muted="#5c5c5c", disabled="#9a9a9a",
             accent="#2f6fd0", accent_hover="#2560bb", accent_soft="#e3ecf9", accent_soft_text="#0d2d5c",
             sel="#cddff8", sel_text="#0d2d5c", on_accent="#ffffff", scroll="#b8b8b8", tip_bg="#ffffff",
             tip_text="#1f1f1f", ok="#2f7d45", fail="#c9372c", view_top=(253, 253, 253),
             view_bottom=(201, 206, 212))

_state = {"dark": False, "colors": LIGHT, "dir": None}
_cache: dict = {}

# ------------------------------------------------------------------ ikony (SVG, viewBox 0 0 24 24, barva {c})
_S = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" '
      'stroke-linecap="round" stroke-linejoin="round">')
_CUBE = '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>'
ICONS = {
    "open": '<path d="M3 7.5A1.5 1.5 0 0 1 4.5 6H9l2 2h8.5A1.5 1.5 0 0 1 21 9.5v8a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z"/>',
    "save": '<path d="M5 4h11l4 4v11a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
    "save_as": '<path d="M5 4h11l4 4v4M12 20H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1"/><path d="M8 4v5h7V4"/>'
               '<path d="M15 20l1-3 4.5-4.5 2 2L18 19z"/>',
    "report": '<path d="M7 3h7l5 5v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/><path d="M14 3v5h5"/>'
              '<path d="M9 13h7M9 17h5"/>',
    "undo": '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    "redo": '<path d="M15 14l5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13"/>',
    "fit": '<path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4"/>'
           '<rect x="9" y="9" width="6" height="6" rx="1"/>',
    "iso": _CUBE,
    "front": _CUBE + '<path d="M4 7.5l8 4.5v9l-8-4.5z" fill="{c}" fill-opacity="0.45"/>',
    "top": _CUBE + '<path d="M12 3l8 4.5-8 4.5-8-4.5z" fill="{c}" fill-opacity="0.45"/>',
    "right": _CUBE + '<path d="M20 7.5v9L12 21v-9z" fill="{c}" fill-opacity="0.45"/>',
    "face": '<path d="M4 8l8-4 8 4v8l-8 4-8-4z"/><path d="M4 8l8 4 8-4-8-4z" fill="{c}" fill-opacity="0.45"/>',
    "body": _CUBE.replace('<path d="M12', '<path fill="{c}" fill-opacity="0.3" d="M12', 1),
    "hole": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="12" cy="12" r="5"/>'
            '<circle cx="12" cy="12" r="1" fill="{c}"/>',
    "edge": '<path d="M5 19L19 5" stroke-width="2.6"/><circle cx="5" cy="19" r="1.8" fill="{c}"/>'
            '<circle cx="19" cy="5" r="1.8" fill="{c}"/>',
    "point": '<circle cx="12" cy="12" r="3" fill="{c}"/><path d="M12 2v5M12 17v5M2 12h5M17 12h5"/>',
    "section": '<path d="M4 7.5l8-4.5 8 4.5v9L12 21l-8-4.5z" stroke-opacity="0.45"/>'
               '<path d="M3 13.5l9-4 9 4-9 4z" fill="{c}" fill-opacity="0.35"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .9-1 1.7"/>'
            '<circle cx="12" cy="17" r=".6" fill="{c}"/>',
    "whatsthis": '<path d="M4 3l6.5 16 2-6.5 6.5-2z"/><path d="M15.5 15.5l4.5 4.5"/>',
    "guide": '<circle cx="12" cy="12" r="9"/><path d="M15.5 8.5l-2 5-5 2 2-5z" fill="{c}" fill-opacity="0.35"/>',
    "tour": '<path d="M5 21V4M5 4h11l-2 3.5 2 3.5H5"/>',
    "theme": '<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="{c}"/>',
    "tools": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.8-3.8a6 6 0 0 1-7.9 7.9l-6.9 6.9'
             'a2.1 2.1 0 0 1-3-3l6.9-6.9a6 6 0 0 1 7.9-7.9z"/>',
    "more": '<circle cx="12" cy="4" r="2.6" fill="{c}"/><circle cx="12" cy="12" r="2.6" fill="{c}"/>'
            '<circle cx="12" cy="20" r="2.6" fill="{c}"/>',      # dropdown button of a bar (narrow, ~12 px)
    "play": '<path d="M7 4.5v15l12-7.5z" fill="{c}" fill-opacity="0.25"/>',
    "popout": '<path d="M14 4h6v6M20 4l-8 8"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
    "dock_in": '<path d="M12 12L4 4M12 12V6M12 12H6"/><path d="M14 4h5a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-5"/>',
    "chevron_down": '<path d="M6 9l6 6 6-6"/>',
    "chevron_up": '<path d="M6 15l6-6 6 6"/>',
    "chevron_right": '<path d="M9 6l6 6-6 6"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><circle cx="12" cy="7.6" r=".8" fill="{c}"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5" stroke-width="3"/>',
    "dash": '<path d="M6 12h12" stroke-width="3"/>',
    # tool shelf commands (main.SHELF)
    "fixed": '<path d="M4 16h16M7 16V8h10v8"/><path d="M5 21l2-4M9 21l2-4M13 21l2-4M17 21l2-4"/>',
    "cylinder": '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="2.5"/>',
    "force": '<path d="M12 3v13"/><path d="M7 11.5l5 5.5 5-5.5"/><path d="M5 21h14" stroke-opacity="0.5"/>',
    "pressure": '<path d="M3 5h18M6.5 5v10M12 5v10M17.5 5v10"/><path d="M4.8 13l1.7 2.5L8.2 13M10.3 13l1.7 2.5 1.7-2.5'
                'M15.8 13l1.7 2.5 1.7-2.5"/><path d="M3 20h18" stroke-opacity="0.5"/>',
    "moment": '<path d="M18.5 12a6.5 6.5 0 1 1-2.2-4.9"/><path d="M16.5 3v4.5H12"/>',
    "add": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M12 8.5v7M8.5 12h7"/>',
    "combine": '<path d="M5 6h14M5 12h14M5 18h8"/><path d="M17 16v4M15 18h4"/>',
    "mesh": '<path d="M3 19L12 4l9 15z"/><path d="M7.5 11.5h9M12 4v15M7.5 11.5L12 19l4.5-7.5"/>',
    "run": '<path d="M7 4.5v15l12-7.5z" fill="{c}" fill-opacity="0.35"/>',
    "line": '<path d="M4 20L20 4"/><circle cx="4" cy="20" r="2" fill="{c}"/><circle cx="20" cy="4" r="2" fill="{c}"/>'
            '<path d="M8 20h12v-6" stroke-opacity="0.5"/>',
    "isosurface": '<path d="M4 13c2-7 14-7 16 0-2 5-14 5-16 0z"/><path d="M8 12.5c1.5-2.5 6.5-2.5 8 0" '
                  'stroke-opacity="0.55"/>',
    "cut_force": '<path d="M4 8l8-4 8 4v8l-8 4-8-4z" stroke-opacity="0.45"/><path d="M12 4v16" stroke-dasharray="2 2"/>'
                 '<path d="M14 12h7M18.5 9.5L21 12l-2.5 2.5"/>',
    "animate": '<path d="M4 18c3-9 6-9 8-4s5 5 8-4"/><path d="M4 21h16" stroke-opacity="0.45"/>',
    "export": '<path d="M12 3v12M7.5 10.5L12 15l4.5-4.5"/><path d="M5 19h14"/>',
    "assign": '<circle cx="9" cy="9" r="5"/><path d="M13 13l7 7M16.5 20H20v-3.5"/>',
    "detect": '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5.5 5.5M8 10.5h5M10.5 8v5"/>',
    "hide": '<path d="M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/>'
            '<circle cx="12" cy="12" r="2.8"/><path d="M4 20L20 4"/>',
    "show": '<path d="M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/>'
            '<circle cx="12" cy="12" r="2.8"/>',
    "exclude": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" stroke-opacity="0.5"/><path d="M5 19L19 5"/>',
    "temperature": '<path d="M10 4a2 2 0 0 1 4 0v10a4 4 0 1 1-4 0z"/><circle cx="12" cy="17" r="1.6" fill="{c}"/>'
                   '<path d="M12 15.4V8"/>',
    "heat": '<path d="M4 20h16" stroke-opacity="0.5"/><path d="M8 16c-2-3 2-5 0-9M12 16c-2-3 2-5 0-9M16 16c-2-3 2-5 0-9"/>',
    "convection": '<path d="M3 8c3-3 6 3 9 0s6 3 9 0M3 13c3-3 6 3 9 0s6 3 9 0"/><path d="M4 19h16" stroke-opacity="0.5"/>',
    "radiation": '<circle cx="12" cy="12" r="3.5"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1'
                 'M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/>',
    "profile": '<path d="M5 4h14M5 20h14M12 4v16"/><path d="M5 4v2M19 4v2M5 20v-2M19 20v-2" stroke-opacity="0.6"/>',
    "joint": '<circle cx="12" cy="12" r="2.5"/><path d="M10 10.5L4 6M14 13.5l6 4.5"/><circle cx="4" cy="6" r="1.4" '
             'fill="{c}"/><circle cx="20" cy="18" r="1.4" fill="{c}"/>',
    "mass": '<path d="M8 8h8l3 12H5z"/><circle cx="12" cy="5.5" r="2.5"/>',
    "contact": '<rect x="3" y="4" width="18" height="6" rx="1"/><path d="M6 14h12l-2 6H8z"/><path d="M8 12h8" '
               'stroke-dasharray="2 2"/>',
}


def is_dark() -> bool:
    return _state["dark"]


def colors() -> dict:
    return _state["colors"]


def _svg(name: str, color: str) -> bytes:
    return (_S.format(c=color) + ICONS[name].replace("{c}", color) + "</svg>").encode()


def _pixmap(name: str, color: str, size: int = 20) -> QPixmap:
    app = QApplication.instance()
    dpr = app.devicePixelRatio() if app is not None else 1.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    QSvgRenderer(QByteArray(_svg(name, color))).render(p, QRectF(0, 0, size, size))
    p.end()
    return pm


def icon(name: str) -> QIcon:
    """Icon in the colours of the current theme (call again after switching the theme)."""
    c = colors()
    key = (name, c["text"])
    if key not in _cache:
        ic = QIcon()
        ic.addPixmap(_pixmap(name, c["text"]), QIcon.Normal, QIcon.Off)
        ic.addPixmap(_pixmap(name, c["accent_soft_text"]), QIcon.Normal, QIcon.On)
        ic.addPixmap(_pixmap(name, c["disabled"]), QIcon.Disabled, QIcon.Off)
        _cache[key] = ic
    return _cache[key]


def set_icon(obj, name: str):
    """Theme icon on a QAction / button; the name is remembered (property feaIcon) for refresh_icons."""
    obj.setProperty("feaIcon", name)
    obj.setIcon(icon(name))


def refresh_icons(root):
    """After switching the theme: sets the icons of all actions and buttons under root that have feaIcon again."""
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QAbstractButton
    for obj in root.findChildren(QAction) + root.findChildren(QAbstractButton):
        name = obj.property("feaIcon")
        if name:
            obj.setIcon(icon(name))


def step_icon(n: int, color: str) -> QIcon:
    """Step number in a circle of the state colour (green / orange / red / grey); n = 0 = only a small dot without a number
    (summary state of a collapsed phase).
    """
    app = QApplication.instance()
    dpr = app.devicePixelRatio() if app is not None else 1.0
    s = 20
    pm = QPixmap(int(s * dpr), int(s * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    if n <= 0:
        p.drawEllipse(QRectF(5, 5, s - 10, s - 10))
        p.end()
        return QIcon(pm)
    p.drawEllipse(QRectF(1, 1, s - 2, s - 2))
    f = QFont(p.font())
    f.setBold(True)
    f.setPixelSize(11 if n < 10 else 10)
    p.setFont(f)
    p.setPen(QColor("#ffffff"))
    p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, str(n))
    p.end()
    return QIcon(pm)


def add_menu_section(menu, text: str):
    """Section heading in a menu (QMenu::addSection is not drawn in the QSS style – only the line would remain): an inactive item
    with the label #menuSection (grey bold text). Returns a QWidgetAction.
    """
    from PySide6.QtWidgets import QLabel, QWidgetAction
    lbl = QLabel(text)
    lbl.setObjectName("menuSection")
    act = QWidgetAction(menu)
    act.setDefaultWidget(lbl)
    act.setEnabled(False)
    menu.addAction(act)
    return act


def set_primary(button, on: bool = True):
    """Primary button of a panel (highlight colour)."""
    button.setProperty("primary", on)
    button.style().unpolish(button)
    button.style().polish(button)


# ------------------------------------------------------------------ paleta a styl
def _system_dark(app) -> bool:
    try:
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:                    # Qt < 6.5
        return app.palette().color(QPalette.Window).lightness() < 128


def _palette(c: dict) -> QPalette:
    pal = QPalette()
    col = {k: QColor(v) for k, v in c.items() if isinstance(v, str)}
    for group in (QPalette.Active, QPalette.Inactive):
        pal.setColor(group, QPalette.Window, col["bg"])
        pal.setColor(group, QPalette.WindowText, col["text"])
        pal.setColor(group, QPalette.Base, col["input"])
        pal.setColor(group, QPalette.AlternateBase, col["alt"])
        pal.setColor(group, QPalette.ToolTipBase, col["tip_bg"])
        pal.setColor(group, QPalette.ToolTipText, col["tip_text"])
        pal.setColor(group, QPalette.PlaceholderText, col["muted"])
        pal.setColor(group, QPalette.Text, col["text"])
        pal.setColor(group, QPalette.Button, col["input"])
        pal.setColor(group, QPalette.ButtonText, col["text"])
        pal.setColor(group, QPalette.BrightText, QColor("#ff5555"))
        pal.setColor(group, QPalette.Highlight, col["accent"])
        pal.setColor(group, QPalette.HighlightedText, col["on_accent"])
        pal.setColor(group, QPalette.Link, col["accent"])
        pal.setColor(group, QPalette.LinkVisited, col["accent"])
        pal.setColor(group, QPalette.Light, col["surface"])
        pal.setColor(group, QPalette.Midlight, col["hover"])
        pal.setColor(group, QPalette.Mid, col["border"])
        pal.setColor(group, QPalette.Dark, col["border"])
        pal.setColor(group, QPalette.Shadow, QColor(0, 0, 0, 90))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, col["disabled"])
    for role, key in ((QPalette.Window, "bg"), (QPalette.Base, "surface"), (QPalette.Button, "surface"),
                      (QPalette.Highlight, "border")):
        pal.setColor(QPalette.Disabled, role, col[key])
    return pal


def _arrow_files(c: dict) -> dict:
    d = _state["dir"] or Path(tempfile.mkdtemp(prefix="feastudio_theme_"))
    _state["dir"] = d
    out = {}
    for name in ("chevron_down", "chevron_up", "check", "dash"):
        for tag, color in (("", c["on_accent"] if name in ("check", "dash") else c["muted"]), ("_dis", c["disabled"])):
            p = d / f"{name}{tag}_{color[1:]}.svg"
            if not p.exists():
                p.write_bytes(_svg(name, color))
            out[name + tag] = p.as_posix()
    return out


QSS = """
* {{ outline: 0; }}
QMainWindow, QDialog, QWidget#popoutWindow {{ background: {bg}; }}
QToolTip {{ background: {tip_bg}; color: {tip_text}; border: 1px solid {border}; border-radius: 3px; padding: 5px 7px; }}

QToolBar {{ background: {bar}; border: none; border-bottom: 1px solid {line}; padding: 2px 6px; spacing: 1px; }}
QToolBar::separator {{ background: {border}; width: 1px; margin: 4px 6px; }}
QToolBar QLabel {{ color: {muted}; padding: 0px 2px 0px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 3px; padding: 3px 6px; }}
QToolButton:hover {{ background: {hover}; }}
QToolButton:pressed {{ background: {pressed}; }}
QToolButton:checked {{ background: {sel}; color: {sel_text}; }}
QToolButton:disabled {{ color: {disabled}; }}
QToolButton[popupMode="2"] {{ padding-right: 16px; }}
QToolButton#qt_toolbar_ext_button {{ padding: 0px; margin: 0px; border: none; }}
QToolButton::menu-indicator {{ image: url("{chevron_down}"); width: 10px; height: 10px;
    subcontrol-origin: padding; subcontrol-position: center right; right: 3px; }}
/* hlavička (návrh Studio × Tabulky): nabídky, karty fází, dokument + Spustit výpočet – jeden řádek 32 px */
QToolBar#tb_file, QToolBar#tb_flow, QToolBar#tb_right {{ background: {bar}; padding: 0px 4px; spacing: 0px;
    border-bottom: 1px solid {line}; }}
QToolBar#tb_file QLabel#brandLogo {{ color: {accent}; font-size: 11pt; padding: 0px 2px 0px 6px; }}
QToolBar#tb_file QLabel#brand {{ color: {text_hi}; font-weight: 700; padding: 0px 10px 0px 0px; }}
QToolBar#tb_file QToolButton#menuButton {{ color: {muted}; padding: 4px 7px; border-radius: 3px; }}
QToolBar#tb_file QToolButton#menuButton:hover {{ background: {hover}; color: {text}; }}
QToolBar#tb_file QToolButton#menuButton::menu-indicator {{ image: none; width: 0px; }}
QToolBar#tb_flow {{ padding-left: 18px; }}
QToolBar#tb_flow QToolButton#phaseTab {{ color: {muted}; padding: 5px 12px; border: none; border-radius: 0px;
    border-bottom: 2px solid transparent; }}
QToolBar#tb_flow QToolButton#phaseTab:hover {{ background: {hover}; color: {text}; }}
QToolBar#tb_flow QToolButton#phaseTab:checked {{ background: {header}; color: {text_hi};
    border-bottom: 2px solid {accent}; }}
QToolBar#tb_right QLabel#docLabel {{ color: {disabled}; padding: 0px 10px; }}
QToolBar#tb_right QPushButton {{ margin: 0px 2px; padding: 2px 10px; min-height: 18px; }}
QToolBar#tb_right QPushButton[primary="true"] {{ padding: 2px 12px; }}
/* záložky kroků fáze nad panelem kroku */
QToolBar#tb_steps {{ background: {bar}; border: none; border-bottom: 1px solid {line}; padding: 0px 4px;
    spacing: 0px; }}
QToolBar#tb_steps QToolButton {{ color: {muted}; padding: 4px 10px 4px 6px; border: none; border-radius: 0px;
    border-bottom: 2px solid transparent; }}
QToolBar#tb_steps QToolButton:hover {{ background: {hover}; color: {text}; }}
QToolBar#tb_steps QToolButton:checked {{ background: {header}; color: {text_hi}; font-weight: 600;
    border-bottom: 2px solid {accent}; }}
QToolBar#tb_steps QToolButton#flowMore {{ padding: 4px 6px; }}
QToolBar#tb_steps QToolButton#flowMore::menu-indicator {{ image: none; width: 0px; }}
/* lišta pohledu nad 3D oknem */
QFrame#tb_view {{ background: {surface}; border: none; border-bottom: 1px solid {line}; }}
QFrame#tb_view QLabel#viewHint {{ color: {muted}; padding: 0px 4px; }}
QFrame#tb_view QLabel#viewTitle {{ color: {text_hi}; font-weight: 600; padding: 0px 6px 0px 4px; }}
QFrame#tb_view QLabel#viewLegend {{ color: {text}; padding: 0px 6px; }}
QFrame#tb_view QToolButton {{ background: {button}; border: 1px solid {button_border}; border-radius: 3px;
    padding: 2px 8px; }}
QFrame#tb_view QToolButton[popupMode="2"] {{ padding-right: 18px; }}
QFrame#tb_view QToolButton:hover {{ background: {hover}; }}
QFrame#tb_view QToolButton:checked {{ background: {sel}; color: {sel_text}; }}
QToolBar#tb_shelf {{ background: {bar}; border: none; border-right: 1px solid {line}; padding: 4px 3px; spacing: 1px; }}
QToolBar#tb_shelf QToolButton {{ padding: 2px 1px 2px 1px; border-radius: 3px; font-size: 8pt; min-width: 54px;
    max-width: 60px; }}
QToolBar#tb_shelf QToolButton[popupMode="2"] {{ padding-right: 1px; }}
QToolBar#tb_shelf QToolButton::menu-indicator {{ image: none; width: 0px; }}
QToolBar#tb_shelf QToolButton:checked {{ background: {sel}; color: {sel_text}; }}
QToolBar#tb_shelf QLabel {{ color: {muted}; font-size: 7pt; font-weight: 600; padding: 6px 0px 2px 0px; }}
QToolBar#tb_shelf::separator {{ background: {border}; height: 1px; margin: 4px 8px; }}

QPushButton {{ background: {button}; border: 1px solid {button_border}; border-radius: 3px; padding: 4px 10px; }}
QPushButton:hover {{ background: {hover}; border-color: {accent}; }}
QPushButton:pressed {{ background: {pressed}; }}
QPushButton:disabled {{ color: {disabled}; background: {surface}; border-color: {border}; }}
QPushButton:checked {{ background: {sel}; color: {sel_text}; border-color: {accent}; }}
QPushButton[primary="true"] {{ background: {accent}; color: {on_accent}; border: 1px solid {accent};
    font-weight: 600; padding: 5px 14px; }}
QPushButton[primary="true"]:hover {{ background: {accent_hover}; border-color: {accent_hover}; }}
QPushButton[primary="true"]:disabled {{ background: {accent_soft}; color: {muted}; border-color: {accent_soft}; }}

QLineEdit, QAbstractSpinBox, QComboBox {{ background: {input}; border: 1px solid {border}; border-radius: 3px;
    padding: 1px 5px; min-height: 20px; selection-background-color: {accent}; selection-color: {on_accent}; }}
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover {{ border-color: {muted}; }}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QComboBox:on {{ border-color: {accent}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ color: {disabled}; background: {surface}; }}
QComboBox {{ padding-right: 20px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 20px; border: none; }}
QComboBox::down-arrow {{ image: url("{chevron_down}"); width: 12px; height: 12px; }}
QComboBox::down-arrow:disabled {{ image: url("{chevron_down_dis}"); }}
QComboBox QAbstractItemView {{ background: {surface}; border: 1px solid {border}; border-radius: 3px; padding: 2px;
    selection-background-color: {sel}; selection-color: {sel_text}; outline: 0; }}
QAbstractSpinBox {{ padding-right: 15px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: 14px; border: none;
    background: transparent; border-radius: 4px; margin-right: 2px; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; margin-top: 2px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; margin-bottom: 2px; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {hover}; }}
QAbstractSpinBox::up-arrow {{ image: url("{chevron_up}"); width: 9px; height: 9px; }}
QAbstractSpinBox::down-arrow {{ image: url("{chevron_down}"); width: 9px; height: 9px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{ image: url("{chevron_up_dis}"); }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{ image: url("{chevron_down_dis}"); }}

QGroupBox {{ background: {surface}; border: 1px solid {border}; border-radius: 3px; margin-top: 16px;
    padding: 10px 6px 6px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 6px; top: 2px; padding: 0px 4px;
    color: {muted}; font-weight: 600; }}

QListWidget, QTreeWidget, QTableWidget, QListView, QTreeView, QTableView, QTextBrowser, QPlainTextEdit, QTextEdit {{
    background: {surface}; border: 1px solid {border}; border-radius: 0px; alternate-background-color: {alt};
    selection-background-color: {sel}; selection-color: {sel_text}; }}
QListWidget::item, QTreeWidget::item {{ padding: 2px 4px; border-radius: 0px; }}
QListWidget::item:hover, QTreeWidget::item:hover, QTableWidget::item:hover {{ background: {hover}; }}
QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{ background: {sel};
    color: {sel_text}; }}
QTableWidget {{ gridline-color: {border}; }}
QHeaderView {{ background: transparent; border: none; }}
QHeaderView::section {{ background: {header}; color: {muted}; border: none; border-bottom: 1px solid {border};
    border-right: 1px solid {border}; padding: 3px 6px; font-weight: 600; }}
QTableCornerButton::section {{ background: {header}; border: none; }}

QTabWidget::pane {{ border: 1px solid {border}; border-radius: 0px; top: -1px; background: {surface}; }}
QTabBar::tab {{ background: transparent; color: {muted}; padding: 5px 12px; border: none;
    border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {text_hi}; background: {surface}; border-bottom: 2px solid {accent}; }}
QTabBar::tab:hover {{ color: {text}; }}

QDockWidget {{ color: {text}; font-weight: 600; }}
QDockWidget::title {{ background: {bar}; padding: 4px 8px; border-bottom: 1px solid {line}; text-align: left; }}
QStatusBar {{ background: {bar}; border-top: 1px solid {line}; color: {muted}; }}
QStatusBar QLabel {{ color: {muted}; }}
QSplitter::handle {{ background: {line}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QMainWindow::separator {{ background: {line}; width: 1px; height: 1px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 1px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 1px; }}
QScrollBar::handle {{ background: {scroll}; border-radius: 3px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QSlider::groove:horizontal {{ height: 4px; background: {border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {accent}; width: 14px; height: 14px; margin: -5px 0px; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {accent_hover}; }}
QProgressBar {{ background: {input}; border: 1px solid {border}; border-radius: 3px; text-align: center; min-height: 14px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 2px; }}
QProgressBar#taskProgress {{ min-height: 18px; font-weight: 600; color: {text}; }}
QProgressBar#taskProgress::chunk {{ background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #3fbf66,
    stop: 1 #2e9e4f); border-radius: 2px; }}
QProgressBar#taskProgress[state="error"]::chunk {{ background: #d23b3b; }}
QProgressBar#taskProgress[state="cancelled"]::chunk {{ background: #9aa1ab; }}

QMenu {{ background: {surface}; border: 1px solid {border}; border-radius: 3px; padding: 3px; }}
QMenu::item {{ padding: 5px 28px 5px 12px; border-radius: 2px; }}
QMenu::item:selected {{ background: {sel}; color: {sel_text}; }}
QMenu::item:disabled {{ color: {disabled}; }}
QMenu::separator {{ height: 1px; background: {border}; margin: 4px 8px; }}

QCheckBox, QRadioButton {{ spacing: 6px; }}
QCheckBox::indicator, QGroupBox::indicator, QAbstractItemView::indicator {{ width: 13px; height: 13px;
    border: 1px solid {muted}; border-radius: 2px; background: {input}; }}
QRadioButton::indicator {{ width: 14px; height: 14px; border: 1px solid {muted}; border-radius: 8px; background: {input}; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover, QGroupBox::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked, QAbstractItemView::indicator:checked {{
    background: {accent}; border-color: {accent}; image: url("{check}"); }}
QCheckBox::indicator:indeterminate, QAbstractItemView::indicator:indeterminate {{ background: {accent};
    border-color: {accent}; image: url("{dash}"); }}
QRadioButton::indicator:checked {{ border-color: {accent}; background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5,
    fx: 0.5, fy: 0.5, stop: 0 {on_accent}, stop: 0.32 {on_accent}, stop: 0.42 {accent}, stop: 1 {accent}); }}
QCheckBox::indicator:disabled, QAbstractItemView::indicator:disabled {{ border-color: {disabled}; background: {surface}; }}
QCheckBox::indicator:checked:disabled {{ background: {disabled}; border-color: {disabled}; image: url("{check}"); }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {disabled}; }}

QFrame#panelHeader {{ background: transparent; border: none; border-bottom: 1px solid {border}; }}
QLabel#panelTitle {{ font-size: 11.5pt; font-weight: 600; color: {text_hi}; }}
QLabel#panelStep {{ color: {accent}; font-weight: 600; font-size: 7.5pt; }}
QLabel#panelSubtitle {{ color: {muted}; }}
QLabel[muted="true"] {{ color: {muted}; }}

QFrame#section {{ background: {surface}; border: 1px solid {border}; border-radius: 0px; }}
QToolButton#sectionHeader {{ background: {header}; border: 1px solid transparent; border-radius: 0px;
    padding: 3px 4px; color: {text_hi}; font-weight: 600; text-align: left; }}
QToolButton#sectionHeader:hover, QToolButton#sectionHeader:checked:hover {{ background: {hover}; }}
QToolButton#sectionHeader:checked {{ background: {header}; color: {text_hi}; }}
QToolButton#sectionHeader:focus {{ border: 1px solid {accent}; }}
QLabel#sectionBadge {{ color: {muted}; background: transparent; padding: 0px 4px; }}
QWidget#sectionBody {{ background: transparent; }}
QToolButton#popoutButton {{ background: transparent; border: none; border-radius: 3px; padding: 1px; }}
QToolButton#popoutButton:hover {{ background: {hover}; }}
QFrame#popoutPlaceholder {{ background: transparent; border: 1px dashed {border}; }}
QToolButton#infoButton {{ background: transparent; border: none; border-radius: 9px; padding: 1px; }}
QToolButton#infoButton:hover {{ background: {hover}; }}
QToolButton#infoButton:focus {{ border: 1px solid {accent}; }}
QLabel#menuSection {{ color: {muted}; font-weight: 600; background: transparent; padding: 6px 12px 2px 12px; }}
QLabel#viewPlaceholder {{ color: {muted}; background: transparent; }}
QFrame#overviewCard {{ background: {surface}; border: 1px solid {border}; border-radius: 0px; }}
QLabel#overviewTitle {{ font-weight: 600; color: {text_hi}; }}
QLabel#overviewValue {{ font-size: 20pt; font-weight: 600; color: {text_hi}; padding-right: 8px; }}
QLabel#overviewValue[state="fail"] {{ color: {fail}; }}
QLabel#overviewVerdict {{ font-weight: 600; color: {muted}; }}
QLabel#overviewVerdict[state="ok"] {{ color: {ok}; }}
QLabel#overviewVerdict[state="fail"] {{ color: {fail}; }}
QStatusBar QLabel#modelStatus[state="ok"] {{ color: {ok}; }}
QLabel#propsGroup {{ color: {text_hi}; font-weight: 600; padding: 5px 0px 1px 0px; }}
QWidget#dockHead {{ background: {bar}; border-bottom: 1px solid {line}; }}
QTabBar#dockTabs::tab {{ padding: 5px 12px; color: {muted}; border: none; border-bottom: 2px solid transparent; }}
QTabBar#dockTabs::tab:selected {{ color: {text_hi}; font-weight: 600; border-bottom: 2px solid {accent}; background: transparent; }}
QTabBar#dockTabs::tab:hover {{ color: {text}; }}
QPushButton#dockAction {{ padding: 2px 9px; }}
QLineEdit#dockFilter {{ min-height: 18px; }}
QLabel#dockNote {{ padding: 3px 8px; border-top: 1px solid {line}; }}
QStatusBar QLabel#modelStatus[state="fail"] {{ color: {fail}; }}
QProgressBar#overviewBar {{ background: {input}; border: 1px solid {border}; border-radius: 3px; min-height: 4px; }}
QProgressBar#overviewBar::chunk {{ background: {muted}; border-radius: 2px; }}
QProgressBar#overviewBar[state="ok"]::chunk {{ background: {ok}; }}
QProgressBar#overviewBar[state="fail"]::chunk {{ background: {fail}; }}
"""


def _font() -> QFont:
    fams = set(QFontDatabase.families())
    for fam in ("Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans", "DejaVu Sans"):
        if fam in fams:
            f = QFont(fam)
            f.setPointSizeF(9.0)
            return f
    return QFont()


def mono_font() -> QFont:
    fams = set(QFontDatabase.families())
    for fam in ("Cascadia Mono", "Consolas", "DejaVu Sans Mono"):
        if fam in fams:
            f = QFont(fam)
            f.setPointSizeF(9.0)
            return f
    f = QFont()
    f.setStyleHint(QFont.Monospace)
    return f


def apply(app: QApplication, mode: str = "system", stylesheet: bool = True) -> bool:
    """Sets the style, palette and QSS of the whole application. Returns True if the resulting theme is dark.
    stylesheet=False: the QSS later by `apply_stylesheet` (start of the window: polishing thousands of widgets one by one as they are
    added costs ~1 s more than one pass over the finished window)."""
    if mode not in MODES:
        mode = "system"
    try:                                       # Qt ≥ 6.8: forces a dark / light window title bar (Windows)
        app.styleHints().setColorScheme({"light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark}.get(
            mode, Qt.ColorScheme.Unknown))
    except AttributeError:
        pass
    dark = _system_dark(app) if mode == "system" else mode == "dark"
    c = DARK if dark else LIGHT
    _state.update(dark=dark, colors=c)
    if app.style().name().lower() != "fusion":
        app.setStyle("Fusion")
    if not getattr(app, "_fea_font", False):
        app.setFont(_font())
        app._fea_font = True
    app.setPalette(_palette(c))
    if stylesheet:
        apply_stylesheet(app)
    return dark


def apply_stylesheet(app: QApplication) -> None:
    """QSS of the current theme (colours from the last `apply`)."""
    c = _state["colors"]
    app.setStyleSheet(QSS.format(**{k: v for k, v in c.items() if isinstance(v, str)}, **_arrow_files(c)))
