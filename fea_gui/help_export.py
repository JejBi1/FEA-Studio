"""FEA Studio manual from the help: pages fea_gui/help/*.html → Word document (.docx) and a separate HTML page.

The source is single (the help pages in the GUI), the manual is only assembled. No Qt – it can run without the GUI:
    python -m fea_gui.help_export [-o folder] [--lang cs en]
        → "Příručka FEA Studio.docx/.html", "Návod k testování FEA Studio.docx/.html" (in English "FEA Studio Manual…")
Word: title page with the version, clickable contents (TOC field with the saved list of chapters), bookmarks at headings and anchors,
links between chapters, header / footer with the page number, coloured tables and Tip / Warning frames.
HTML (`help_web.py`): search, expandable sections, contents on the left, dark / light appearance.
Supported HTML subset = the one the help pages use (h1–h3, p with classes lead/tip/warn, ul/ol/li,
table.grid, b/i/code/sub/sup/span.key, a, pre, img, br).
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

HELP_DIR = Path(__file__).resolve().parent / "help"
TITLE = "FEA Studio – uživatelská příručka"

DOCX_UI = {
    "cs": {"toc": "Obsah", "page": "Strana", "toc_hint": "Kapitoly jsou odkazy (Ctrl+klik). Celý obsah s čísly stran a oddíly vznikne "
           "po aktualizaci polí (F9); nadpisy najdete také v Navigačním podokně (Zobrazení → Navigační podokno).",
           "version": "Verze", "edition": "Vydání", "tech": "CalculiX · Gmsh · NGSolve"},
    "en": {"toc": "Contents", "page": "Page", "toc_hint": "Chapters are links (Ctrl+click). The full table of contents with page numbers and "
           "sections is created when fields are updated (F9); headings are also in the Navigation Pane (View → Navigation Pane).",
           "version": "Version", "edition": "Edition", "tech": "CalculiX · Gmsh · NGSolve"},
}

# palette (same as the PDF and HTML): dark blue, blue, sky blue, slate text
PAL = dict(navy="0B1F4D", navy2="123A8A", blue="1D4ED8", sky="38BDF8", ink="0F172A", text="334155", muted="64748B",
           line="CBD5E1", soft="EFF6FF", zebra="F1F5F9", th="1E3A8A", code="BE123C", link="1D63D8",
           tip_bg="ECFDF3", tip_bd="12B76A", tip_bd2="A6F4C5", tip_tx="054F31",
           warn_bg="FFFAEB", warn_bd="F79009", warn_bd2="FEDF89", warn_tx="7A2E0E")


def _version() -> tuple[str, str]:
    """(short "1.3", full "1.3.0")."""
    try:
        from fea_core import __version__ as v
    except Exception:  # noqa: BLE001
        v = "1.3.0"
    return ".".join(v.split(".")[:2]), v


def _edition(lang: str) -> str:
    t = date.today()
    return f"{t.day}. {t.month}. {t.year}" if lang == "cs" else f"{t:%B} {t.day}, {t.year}"


def _kind(only) -> str:
    from .help_pdf import TESTING_PAGES
    return "testing" if only and set(only) <= set(TESTING_PAGES) else "manual"


def _pages(only: list[str] | None = None, lang: str = "cs"):
    from .help_pdf import pages
    return pages(lang, only)


def _page_file(slug: str, lang: str) -> Path:
    from .help_pdf import page_file
    return page_file(slug, lang)


# ------------------------------------------------------------------ HTML → bloky
class _Block:
    def __init__(self, kind, cls="", level=0):
        self.kind = kind          # h1 h2 h3 p li pre img table
        self.cls = cls
        self.level = level        # list indentation
        self.ordered = False
        self.number = 0
        self.runs: list[tuple[str, set]] = []   # (text, {"b","i","code","key","sub","sup","href:…"})
        self.rows: list[list[list[tuple[str, set]]]] = []   # table: rows → cells → runs
        self.header: list[bool] = []
        self.src = ""
        self.anchors: list[str] = []            # <a name="…"> in the block (link targets page.html#anchor)


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[_Block] = []
        self.cur: _Block | None = None
        self.fmt: list[str] = []
        self.lists: list[list] = []          # stack [ordered, counter]
        self.table: _Block | None = None
        self.cell: list | None = None

    def _new(self, kind, cls=""):
        self.cur = _Block(kind, cls, len(self.lists))
        self.blocks.append(self.cur)
        return self.cur

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("h1", "h2", "h3", "p", "pre"):
            self._new(tag, a.get("class", ""))
        elif tag in ("ul", "ol"):
            self.lists.append([tag == "ol", 0])
        elif tag == "li":
            b = self._new("li")
            if self.lists:
                self.lists[-1][1] += 1
                b.ordered, b.number = self.lists[-1][0], self.lists[-1][1]
        elif tag == "table":
            self.table = _Block("table", a.get("class", ""))
            self.blocks.append(self.table)
        elif tag == "tr" and self.table is not None:
            self.table.rows.append([])
            self.table.header.append(False)
        elif tag in ("td", "th") and self.table is not None:
            self.cell = []
            self.table.rows[-1].append(self.cell)
            if tag == "th":
                self.table.header[-1] = True
        elif tag == "img":
            b = self._new("img")
            b.src = a.get("src", "")
            self.cur = None
        elif tag == "br":
            self._text("\n")
        elif tag in ("b", "strong"):
            self.fmt.append("b")
        elif tag in ("i", "em"):
            self.fmt.append("i")
        elif tag in ("code", "sub", "sup"):
            self.fmt.append(tag)
        elif tag == "span" and "key" in a.get("class", ""):
            self.fmt.append("key")
        elif tag == "a":
            if a.get("name") and self.cur is not None:
                self.cur.anchors.append(a["name"])
            self.fmt.append("href:" + a["href"] if a.get("href") else "")
        elif tag == "span":
            self.fmt.append("")

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3", "p", "pre", "li"):
            self.cur = None
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
        elif tag in ("td", "th"):
            self.cell = None
        elif tag == "table":
            self.table = None
        elif tag in ("b", "strong", "i", "em", "code", "sub", "sup", "span", "a") and self.fmt:
            self.fmt.pop()

    def _text(self, text):
        f = set(x for x in self.fmt if x)
        if self.cell is not None:
            self.cell.append((text, f))
            return
        if self.cur is None:
            if not text.strip():
                return
            self._new("p")
        self.cur.runs.append((text, f))

    def handle_data(self, data):
        if self.cur is not None and self.cur.kind == "pre":
            self._text(data)
        else:
            self._text(re.sub(r"\s+", " ", data))


def parse(html: str) -> list[_Block]:
    p = _Parser()
    p.feed(html)
    return p.blocks


def _strip(items):
    """Trims the spaces at the start and the end of a paragraph."""
    items = [list(x) for x in items if x[0]]
    if items:
        items[0][0] = items[0][0].lstrip()
        items[-1][0] = items[-1][0].rstrip()
    return [tuple(x) for x in items]


# ------------------------------------------------------------------ Word: helpers for XML
_PPR = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr", "suppressLineNumbers", "pBdr",
        "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN",
        "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
        "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange"]
_TBLPR = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc",
          "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook"]
_TCPR = ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar", "textDirection", "tcFitText",
         "vAlign", "hideMark"]


def _put(parent, child, order):
    """Inserts a child into the parent at the position given by the order of elements in the OOXML schema (Word is sensitive to the order); replaces the same element."""
    from docx.oxml.ns import qn
    name = child.tag.split("}")[1]
    for old in parent.findall(qn("w:" + name)):
        parent.remove(old)
    rank = order.index(name)
    for i, sib in enumerate(list(parent)):
        sname = sib.tag.split("}")[1]
        if sname in order and order.index(sname) > rank:
            parent.insert(i, child)
            return child
    parent.append(child)
    return child


def _el(tag, **attrs):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    e = OxmlElement(tag if ":" in tag else "w:" + tag)
    for k, v in attrs.items():
        e.set(qn("w:" + k), str(v))
    return e


def _borders(kind, **sides):
    """w:pBdr / w:tblBorders / w:tcBorders; sides = {side: (thickness 1/8 pt, colour, spacing)}."""
    b = _el(kind)
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if side in sides:
            sz, color, space = sides[side]
            b.append(_el(side, val="single" if sz else "nil", sz=sz, space=space, color=color))
    return b


def _shade(parent, fill, order):
    _put(parent, _el("shd", val="clear", color="auto", fill=fill), order)


def _spacing(par, before=None, after=None, line=None, exact=None):
    from docx.shared import Pt
    pf = par.paragraph_format
    if before is not None:
        pf.space_before = Pt(before)
    if after is not None:
        pf.space_after = Pt(after)
    if line is not None:
        pf.line_spacing = line
    if exact is not None:
        from docx.enum.text import WD_LINE_SPACING
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        pf.line_spacing = Pt(exact)


def _field(par, instr, cached="1", size=None, color=None, bold=None):
    """Simple field (PAGE, SECTIONPAGES…) with a stored result."""
    from docx.shared import Pt, RGBColor
    for kind in ("begin", "instr", "separate", "text", "end"):
        r = par.add_run()
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
        if bold:
            r.bold = True
        if kind == "instr":
            it = _el("instrText")
            it.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            it.text = f" {instr} "
            r._r.append(it)
        elif kind == "text":
            r.text = cached
        else:
            r._r.append(_el("fldChar", fldCharType=kind))


class _Marks:
    """Counter of bookmark IDs and the list of existing bookmarks (links to non-existing ones are not created)."""
    def __init__(self):
        self.n = 0
        self.known: set[str] = set()

    def bookmark(self, par, names):
        names = [n for n in dict.fromkeys(names) if n]
        if not names:
            return
        p = par._p
        pos = 1 if p.pPr is not None else 0
        for name in names:
            self.n += 1
            p.insert(pos, _el("bookmarkStart", id=self.n, name=name))
            pos += 1
            p.append(_el("bookmarkEnd", id=self.n))


def _bm(prefix, *parts):
    return re.sub(r"[^A-Za-z0-9_]", "_", prefix + "_" + "_".join(str(x) for x in parts))[:38]


# ------------------------------------------------------------------ Word: styly
def _style_fonts(style, name="Calibri"):
    """Font of the style without a link to the theme (otherwise rFonts would override the font name)."""
    from docx.oxml.ns import qn
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    for k in list(rf.attrib):
        if k.endswith("Theme"):
            del rf.attrib[k]
    rf.set(qn("w:eastAsia"), name)
    rf.set(qn("w:cs"), name)


def _setup_styles(doc, lang):
    from docx.enum.style import WD_STYLE_TYPE
    from docx.shared import Cm, Pt, RGBColor
    st = doc.styles
    rgb = RGBColor.from_string
    n = st["Normal"]
    _style_fonts(n)
    n.font.size = Pt(10.5)
    n.font.color.rgb = rgb(PAL["text"])
    n.paragraph_format.space_after = Pt(6)
    n.paragraph_format.line_spacing = 1.18
    n.element.get_or_add_rPr().append(_el("lang", val="cs-CZ" if lang == "cs" else "en-US"))
    spec = {1: (24, PAL["navy"], 0, 10), 2: (15, PAL["blue"], 20, 6), 3: (12, PAL["ink"], 12, 4)}
    for lvl, (size, color, before, after) in spec.items():
        h = st[f"Heading {lvl}"]
        _style_fonts(h)
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = rgb(color)
        pf = h.paragraph_format
        pf.space_before, pf.space_after = Pt(before), Pt(after)
        pf.keep_with_next = True
        pf.keep_together = True
        ppr = h.element.get_or_add_pPr()
        if lvl == 1:
            pf.page_break_before = True
            _put(ppr, _borders("pBdr", bottom=(18, PAL["sky"], 6)), _PPR)
        elif lvl == 2:
            pf.left_indent = Cm(0.35)
            _put(ppr, _borders("pBdr", left=(32, PAL["sky"], 8)), _PPR)
    for name in ("List Bullet", "List Number"):
        s = st[name]
        _style_fonts(s)
        s.paragraph_format.space_after = Pt(3)
    t = st.add_style("FEA Obsah", WD_STYLE_TYPE.PARAGRAPH)
    t.base_style = n
    _style_fonts(t)
    t.font.size, t.font.bold, t.font.color.rgb = Pt(28), True, rgb(PAL["navy"])
    t.paragraph_format.space_after = Pt(8)
    _put(t.element.get_or_add_pPr(), _borders("pBdr", bottom=(18, PAL["sky"], 6)), _PPR)
    st.add_style("FEA Kod", WD_STYLE_TYPE.PARAGRAPH).base_style = n


# ------------------------------------------------------------------ Word: obsah dokumentu
class _Writer:
    def __init__(self, doc, lang, marks: _Marks, width_dxa: int):
        self.doc, self.lang, self.m, self.width = doc, lang, marks, width_dxa

    # --- text runs
    def runs(self, par, items, color=None, size=None):
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
        from docx.shared import Pt, RGBColor
        for text, f in items:
            if not text:
                continue
            r = par.add_run(text)
            r.bold = "b" in f or None
            r.italic = "i" in f or None
            if size:
                r.font.size = Pt(size)
            if color:
                r.font.color.rgb = RGBColor.from_string(color)
            if "code" in f or "key" in f:
                r.font.name = "Consolas"
                r.font.size = Pt(9.5)
            if "key" in f:
                r.font.color.rgb = RGBColor.from_string(PAL["navy"])
                r.bold = True
            elif "code" in f:
                r.font.color.rgb = RGBColor.from_string(PAL["code"])
            rpr = r._r.get_or_add_rPr()
            if "key" in f or "code" in f:
                if "key" in f:
                    rpr.append(_el("bdr", val="single", sz=4, space=0, color=PAL["line"]))
                rpr.append(_el("shd", val="clear", color="auto", fill="E8EEF7" if "key" in f else PAL["zebra"]))
            if "sub" in f:
                r.font.subscript = True
            if "sup" in f:
                r.font.superscript = True
            href = next((x[5:] for x in f if x.startswith("href:")), None)
            if href:
                target = self.link_target(href)
                if target:
                    kind, val = target
                    h = _el("hyperlink", history=1)
                    if kind == "anchor":
                        h.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}anchor", val)
                    else:
                        h.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id",
                              par.part.relate_to(val, RT.HYPERLINK, is_external=True))
                    par._p.remove(r._r)
                    h.append(r._r)
                    par._p.append(h)
                    r.font.color.rgb = RGBColor.from_string(PAL["link"])
                    r.font.underline = True

    def link_target(self, href):
        m = re.match(r"([a-z0-9_]+)\.html(?:#(.*))?$", href)
        if m:
            if m.group(2) and _bm("a", m.group(1), m.group(2)) in self.m.known:
                return "anchor", _bm("a", m.group(1), m.group(2))
            if _bm("ch", m.group(1)) in self.m.known:
                return "anchor", _bm("ch", m.group(1))
            return None
        if href.startswith(("http://", "https://", "mailto:")):
            return "url", href
        return None                            # fea:… (action in the program) – in the document only text

    # --- odstavce
    def heading(self, b, slug, h2_no):
        level = int(b.kind[1])
        par = self.doc.add_heading(level=level)
        self.runs(par, [(x, set()) for x, _ in b.runs])
        names = [_bm("a", slug, a) for a in b.anchors]
        if level == 1:
            names.append(_bm("ch", slug))
        elif level == 2:
            names.append(_bm("s", slug, h2_no))
        self.m.bookmark(par, names)

    def callout(self, par, kind):
        from docx.shared import Cm
        strong, light, bg = (PAL["tip_bd"], PAL["tip_bd2"], PAL["tip_bg"]) if kind == "tip" else \
            (PAL["warn_bd"], PAL["warn_bd2"], PAL["warn_bg"])
        ppr = par._p.get_or_add_pPr()
        _put(ppr, _borders("pBdr", top=(6, light, 5), left=(40, strong, 8), bottom=(6, light, 5), right=(6, light, 8)), _PPR)
        _shade(ppr, bg, _PPR)
        par.paragraph_format.left_indent = Cm(0.35)
        par.paragraph_format.right_indent = Cm(0.2)
        par.paragraph_format.keep_together = True
        _spacing(par, 8, 10)

    def paragraph(self, b):
        par = self.doc.add_paragraph()
        items = _strip(b.runs)
        if "lead" in b.cls:
            self.runs(par, items, color=PAL["muted"], size=12)
            for r in par.runs:
                r.italic = True
            _spacing(par, 2, 10)
        elif "tip" in b.cls or "warn" in b.cls:
            kind = "tip" if "tip" in b.cls else "warn"
            self.runs(par, items, color=PAL["tip_tx"] if kind == "tip" else PAL["warn_tx"])
            self.callout(par, kind)
        else:
            self.runs(par, items)
        return par

    def list_item(self, b):
        from docx.shared import Cm
        if b.ordered:                      # own numbering – "List Number" lists in Word do not restart
            par = self.doc.add_paragraph()
            ind = 0.9 + 0.63 * max(b.level - 1, 0)
            par.paragraph_format.left_indent = Cm(ind)
            par.paragraph_format.first_line_indent = Cm(-0.6)
            par.paragraph_format.tab_stops.add_tab_stop(Cm(ind))
            from docx.shared import RGBColor
            r = par.add_run(f"{b.number}.\t")
            r.bold = True
            r.font.color.rgb = RGBColor.from_string(PAL["blue"])
        else:
            par = self.doc.add_paragraph(style="List Bullet")
            if b.level > 1:
                par.paragraph_format.left_indent = Cm(0.63 * b.level + 0.2)
        self.runs(par, _strip(b.runs))
        _spacing(par, 0, 3)

    def pre(self, b):
        from docx.shared import Pt
        par = self.doc.add_paragraph(style="FEA Kod")
        self.runs(par, [(x, {"code"}) for x, _ in b.runs])
        for r in par.runs:
            r.font.size = Pt(9)
            r.font.color.rgb = None
            rpr = r._r.get_or_add_rPr()
            for e in rpr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}shd"):
                rpr.remove(e)
        ppr = par._p.get_or_add_pPr()
        box = (4, PAL["line"], 6)
        _put(ppr, _borders("pBdr", top=box, left=box, bottom=box, right=box), _PPR)
        _shade(ppr, PAL["zebra"], _PPR)
        _spacing(par, 4, 8)
        par.paragraph_format.left_indent = par.paragraph_format.right_indent = Pt(6)

    def image(self, b, slug, lang):
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Emu
        for d in (_page_file(slug, lang).parent, HELP_DIR):
            img = d / b.src
            if img.exists() and img.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
                self.doc.add_picture(str(img), width=Emu(int(self.width * 635)))
                self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                return

    # --- tabulky
    @staticmethod
    def weights(rows, ncol):
        w = []
        for c in range(ncol):
            lens, longest = [], 0
            for r in rows:
                if c < len(r):
                    t = "".join(x for x, _ in r[c])
                    lens.append(max((len(s) for s in t.split("\n")), default=0))
                    longest = max([longest] + [len(wd) for wd in t.split()])
            avg = sum(lens) / max(len(lens), 1)
            w.append(max(min(avg, 70), min(longest, 24) + 2, 8))
        return w

    def table(self, b):
        from docx.shared import Pt
        rows = b.rows
        ncol = max(len(r) for r in rows)
        tab = self.doc.add_table(rows=0, cols=ncol)
        tab.autofit = False
        tpr = tab._tbl.tblPr
        _put(tpr, _el("tblW", w=self.width, type="dxa"), _TBLPR)
        _put(tpr, _borders("tblBorders", top=(12, PAL["th"], 0), bottom=(8, PAL["line"], 0), insideH=(4, PAL["line"], 0)), _TBLPR)
        _put(tpr, _el("tblLayout", type="fixed"), _TBLPR)
        mar = _el("tblCellMar")
        for side, v in (("top", 50), ("left", 100), ("bottom", 50), ("right", 100)):
            mar.append(_el(side, w=v, type="dxa"))
        _put(tpr, mar, _TBLPR)
        wt = self.weights(rows, ncol)
        widths = [max(int(self.width * x / sum(wt)), 700) for x in wt]
        scale = self.width / sum(widths)
        widths = [int(x * scale) for x in widths]
        grid = tab._tbl.tblGrid
        for gc, wd in zip(grid.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}gridCol"), widths):
            gc.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}w", str(wd))
        body_i = 0
        for row, head in zip(rows, b.header):
            tr = tab.add_row()
            trpr = tr._tr.get_or_add_trPr()
            trpr.append(_el("cantSplit"))
            if head:
                trpr.append(_el("tblHeader"))
            else:
                body_i += 1
            for c in range(ncol):
                cell = tr.cells[c]
                tcpr = cell._tc.get_or_add_tcPr()
                _put(tcpr, _el("tcW", w=widths[c], type="dxa"), _TCPR)
                if head:
                    _shade(tcpr, PAL["th"], _TCPR)
                elif body_i % 2 == 0:
                    _shade(tcpr, PAL["zebra"], _TCPR)
                items = row[c] if c < len(row) else []
                par = cell.paragraphs[0]
                if head:
                    self.runs(par, [(x, f | {"b"}) for x, f in _strip(items)], color="FFFFFF", size=9.5)
                else:
                    self.runs(par, _strip(items), size=9.5)
                par.paragraph_format.space_after = Pt(0)
                par.paragraph_format.line_spacing = 1.08
        self.spacer(6)

    def spacer(self, pt):
        par = self.doc.add_paragraph()
        _spacing(par, 0, 0, exact=pt)
        return par


def _cover(doc, sec, kind_title, subtitle, ver, edition, author, D):
    """Title page: a saturated blue area over the whole page (a table without page margins), a darker footer with the edition data."""
    from docx.shared import Cm, Pt, RGBColor
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = sec.top_margin = Cm(0)
    sec.bottom_margin = Cm(0.3)
    sec.header_distance = sec.footer_distance = Cm(0)
    tab = doc.add_table(rows=2, cols=1)
    tab.autofit = False
    tpr = tab._tbl.tblPr
    _put(tpr, _el("tblW", w=11906, type="dxa"), _TBLPR)
    _put(tpr, _el("tblInd", w=0, type="dxa"), _TBLPR)
    _put(tpr, _el("tblLayout", type="fixed"), _TBLPR)
    tab._tbl.tblGrid.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}gridCol")[0] \
        .set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}w", "11906")
    for row, (fill, h) in zip(tab.rows, ((PAL["navy"], 13650), (PAL["navy2"], 2900))):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(_el("cantSplit"))
        trpr.append(_el("trHeight", val=h, hRule="exact"))
        tcpr = row.cells[0]._tc.get_or_add_tcPr()
        _put(tcpr, _el("tcW", w=11906, type="dxa"), _TCPR)
        _shade(tcpr, fill, _TCPR)
        mar = _el("tcMar")
        for side, v in (("top", 0), ("left", 1300), ("bottom", 0), ("right", 1000)):
            mar.append(_el(side, w=v, type="dxa"))
        _put(tcpr, mar, _TCPR)

    def line(cell, text, size, color, bold=False, before=0, after=0, spacing=None, first=False, shade=None, right=None):
        par = cell.paragraphs[0] if first else cell.add_paragraph()
        r = par.add_run(text)
        r.font.size, r.bold = Pt(size), bold or None
        r.font.color.rgb = RGBColor.from_string(color)
        if spacing:
            r._r.get_or_add_rPr().append(_el("spacing", val=spacing))
        _spacing(par, before, after)
        if right is not None:
            par.paragraph_format.right_indent = Cm(right)
        if shade:
            _shade(par._p.get_or_add_pPr(), shade, _PPR)
        return par

    c = tab.rows[0].cells[0]
    c.vertical_alignment = None
    line(c, "FEA STUDIO", 13, "93C5FD", True, before=150, spacing=70, first=True)
    line(c, kind_title, 46, "FFFFFF", True, before=12, after=6).paragraph_format.line_spacing = 0.95
    line(c, subtitle, 16, "DBEAFE", before=6, after=22, right=2.5)
    bar = line(c, " ", 4, "38BDF8", after=26, right=15.5)
    _put(bar._p.get_or_add_pPr(), _borders("pBdr", bottom=(48, PAL["sky"], 1)), _PPR)
    line(c, f"  {D['version'].upper()} {ver}  ", 18, "04243D", True, shade=PAL["sky"], right=13.2)
    f = tab.rows[1].cells[0]
    line(f, D["tech"], 11, "BFDBFE", first=True, before=26)
    line(f, f"{D['edition']} {edition}" + (f" · {author}" if author else ""), 11, "DBEAFE", before=4)


def _header_footer(sec, doc_title, ver, ver_full, edition, D):
    from docx.enum.text import WD_TAB_ALIGNMENT
    from docx.shared import Pt, RGBColor
    width = sec.page_width - sec.left_margin - sec.right_margin
    sec.header.is_linked_to_previous = False
    sec.footer.is_linked_to_previous = False
    h = sec.header.paragraphs[0]
    h.paragraph_format.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
    r = h.add_run("FEA STUDIO")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(8.5), RGBColor.from_string(PAL["blue"])
    r = h.add_run(f"   {doc_title}\t{D['version']} {ver}")
    r.font.size, r.font.color.rgb = Pt(8.5), RGBColor.from_string(PAL["muted"])
    _put(h._p.get_or_add_pPr(), _borders("pBdr", bottom=(6, PAL["line"], 4)), _PPR)
    f = sec.footer.paragraphs[0]
    f.paragraph_format.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
    r = f.add_run(f"{ver_full} · {edition}\t{D['page']} ")
    r.font.size, r.font.color.rgb = Pt(8.5), RGBColor.from_string(PAL["muted"])
    _field(f, "PAGE", "1", 8.5, PAL["navy"], True)
    r = f.add_run(" / ")
    r.font.size, r.font.color.rgb = Pt(8.5), RGBColor.from_string(PAL["muted"])
    _field(f, "SECTIONPAGES", "1", 8.5, PAL["muted"])
    _put(f._p.get_or_add_pPr(), _borders("pBdr", top=(6, PAL["line"], 4)), _PPR)


def _toc(doc, W: _Writer, chapters, D):
    """TOC field (Word adds the sections and page numbers when updating) with the stored clickable list of chapters."""
    from docx.shared import Cm, Pt, RGBColor
    doc.add_paragraph(D["toc"], style="FEA Obsah")
    hint = doc.add_paragraph()
    W.runs(hint, [(D["toc_hint"], set())], color=PAL["muted"], size=9)
    _spacing(hint, 0, 10)
    n = len(chapters)
    for i, (slug, name) in enumerate(chapters):
        par = doc.add_paragraph()
        par.paragraph_format.tab_stops.add_tab_stop(Cm(1.1))
        _spacing(par, 3, 3)
        par.paragraph_format.left_indent = Cm(1.1)
        par.paragraph_format.first_line_indent = Cm(-1.1)
        if i == 0:
            for kind, text in (("begin", None), (None, r' TOC \o "1-2" \h \z \u '), ("separate", None)):
                r = par.add_run()
                if kind:
                    fc = _el("fldChar", fldCharType=kind)
                    if kind == "begin":
                        fc.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}dirty", "true")
                    r._r.append(fc)
                else:
                    it = _el("instrText")
                    it.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                    it.text = text
                    r._r.append(it)
        r = par.add_run(f"{i + 1:02d}\t")
        r.bold, r.font.size, r.font.color.rgb = True, Pt(10.5), RGBColor.from_string(PAL["blue"])
        h = _el("hyperlink", anchor=_bm("ch", slug), history=1)
        par._p.append(h)
        r = par.add_run(re.sub(r"^\s*\d+\.\s*", "", name))
        r.font.size, r.font.color.rgb, r.bold = Pt(11), RGBColor.from_string(PAL["ink"]), True
        par._p.remove(r._r)
        h.append(r._r)
        if i == n - 1:
            r = par.add_run()
            r._r.append(_el("fldChar", fldCharType="end"))


def export_docx(path: str | Path, author: str = "", only: list[str] | None = None, title: str | None = None,
                lang: str = "cs", kind: str | None = None) -> Path:
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.shared import Cm
    from .help_pdf import TEXTS

    kind = kind or _kind(only)
    T, D = TEXTS[lang], DOCX_UI[lang]
    doc_title = title or T[kind]
    ver, ver_full = _version()
    edition = _edition(lang)
    items = _pages(only, lang)
    pages = {}
    for slug, _t in items:
        f = _page_file(slug, lang)
        if f.exists():
            pages[slug] = parse(f.read_text(encoding="utf-8"))
    items = [(s, t) for s, t in items if s in pages]

    marks = _Marks()                                    # known bookmarks: chapters, h2 sections and anchors <a name>
    for slug, blocks in pages.items():
        marks.known.add(_bm("ch", slug))
        h2 = 0
        for b in blocks:
            if b.kind == "h2":
                h2 += 1
                marks.known.add(_bm("s", slug, h2))
            for a in b.anchors:
                marks.known.add(_bm("a", slug, a))

    doc = Document()
    _setup_styles(doc, lang)
    sec = doc.sections[0]
    _cover(doc, sec, doc_title, T["sub_" + kind], ver, edition, author, DOCX_UI[lang])
    brk = doc.add_section(WD_SECTION.NEW_PAGE)         # paragraph with the end of a section: must not overflow to the next page
    tiny = doc.paragraphs[-1]
    _spacing(tiny, 0, 0, exact=1)
    brk.page_width, brk.page_height = Cm(21), Cm(29.7)
    brk.left_margin = brk.right_margin = Cm(2.0)
    brk.top_margin, brk.bottom_margin = Cm(2.4), Cm(2.1)
    brk.header_distance = brk.footer_distance = Cm(1.0)
    _put(brk._sectPr, _el("pgNumType", start=1), ["headerReference", "footerReference", "footnotePr", "endnotePr", "type", "pgSz",
                                                    "pgMar", "paperSrc", "pgBorders", "lnNumType", "pgNumType", "cols"])
    _header_footer(brk, doc_title, ver, ver_full, edition, D)
    width_dxa = int((brk.page_width - brk.left_margin - brk.right_margin) / 635)

    W = _Writer(doc, lang, marks, width_dxa)
    _toc(doc, W, items, D)
    for slug, _t in items:
        h2_no = 0
        seen_h1 = False
        for b in pages[slug]:
            if b.kind in ("h1", "h2", "h3"):
                if b.kind == "h2":
                    h2_no += 1
                if b.kind == "h1":
                    if seen_h1:
                        b.kind = "h2"
                    seen_h1 = True
                W.heading(b, slug, h2_no)
            elif b.kind == "img":
                W.image(b, slug, lang)
            elif b.kind == "table":
                if b.rows:
                    W.table(b)
            elif b.kind == "li":
                W.list_item(b)
            elif b.kind == "pre":
                W.pre(b)
            else:
                W.paragraph(b)
    cp = doc.core_properties
    cp.title, cp.subject = f"FEA Studio {ver_full} – {doc_title}", T["sub_" + kind]
    cp.author = author or "FEA Studio"
    cp.keywords = "FEA Studio, MKP, FEM, CalculiX"
    cp.comments = f"FEA Studio {ver_full}"
    path = Path(path)
    doc.save(path)
    return path


# ------------------------------------------------------------------ HTML
def export_html(path: str | Path, only: list[str] | None = None, title: str | None = None, lang: str = "cs",
                kind: str | None = None, bilingual: bool = True) -> Path:
    """One separate HTML file (images base64, links between pages → anchors; search, contents, dark look, CZ/EN switch)."""
    from . import help_web
    ver, _ = _version()
    return help_web.export_html(path=path, lang=lang, only=only, title=title, version=ver, bilingual=bilingual)


def main(argv=None) -> int:
    from .help_pdf import TEXTS
    ap = argparse.ArgumentParser(description="Příručka FEA Studia z nápovědy (Word + HTML)")
    ap.add_argument("-o", "--out", default=".", help="cílová složka")
    ap.add_argument("--author", default="", help="autor / firma na titulní straně")
    ap.add_argument("--lang", nargs="+", default=["cs"], choices=["cs", "en"], help="jazyky (výchozí cs)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for lang in a.lang:
        man = TEXTS[lang]["file_manual"].rsplit(".", 1)[0]
        tst = TEXTS[lang]["file_testing"].rsplit(".", 1)[0]
        print(export_docx(out / f"{man}.docx", a.author, lang=lang))
        print(export_html(out / f"{man}.html", lang=lang))
        print(export_docx(out / f"{tst}.docx", a.author, ["testovani", "ukazky"], lang=lang))
        print(export_html(out / f"{tst}.html", ["testovani", "ukazky"], lang=lang))
    return 0


if __name__ == "__main__":
    sys.exit(main())
