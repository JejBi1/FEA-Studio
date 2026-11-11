"""Manual and testing guide as PDF (Czech and English) from the help pages.

    python -m fea_gui.help_pdf [-o docs] [--lang cs en] [--only manual|testing]

The help pages (English source fea_gui/help/<slug>.html, Czech translation fea_gui/help/cs/<slug>.html) are assembled into one HTML with a print style
(title page, contents with page numbers, chapters from a new page, header and footer with the page number) and printed
by Chromium from Qt WebEngine (QWebEnginePage.printToPdf) – CSS paging (@page, margins with the header) is supported by Chromium 131+.
Page numbers in the contents: first print → chapter pages from invisible markers (pypdfium2) → second print with the numbers.
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
import tempfile
from datetime import date
from html import escape
from pathlib import Path

HELP_DIR = Path(__file__).resolve().parent / "help"

TEXTS = {
    "cs": {"manual": "Uživatelská příručka", "testing": "Návod k testování", "toc": "Obsah", "edition": "Vydání",
           "page": "Strana", "chapter": "Kapitola", "sub_manual": "Analýza metodou konečných prvků pro konstruktéry – "
           "pevnost, únava, šrouby a svary, 3D tisk, teplo",
           "sub_testing": "Úlohy se známým výsledkem: postup v programu a očekávané hodnoty",
           "file_manual": "Příručka FEA Studio.pdf", "file_testing": "Návod k testování FEA Studio.pdf",
           "missing": "Stránka nápovědy zatím chybí."},
    "en": {"manual": "User Manual", "testing": "Testing Guide", "toc": "Contents", "edition": "Edition",
           "page": "Page", "chapter": "Chapter", "sub_manual": "Finite element analysis for design engineers – "
           "strength, fatigue, bolts and welds, 3D printing, heat transfer",
           "sub_testing": "Benchmark problems with known results: workflow in the program and expected values",
           "file_manual": "FEA Studio Manual.pdf", "file_testing": "FEA Studio Testing Guide.pdf",
           "missing": "This help page is not available yet."},
}
TESTING_PAGES = ["testovani", "ukazky"]
MARGINS_MM = (17, 20, 17, 18)          # left, top, right, bottom – Qt passes them to Chromium instead of the margins from @page

CSS = r"""
@page {
  size: A4; margin: 20mm 17mm 18mm 17mm;
  @top-left { content: "FEA Studio @VER@"; font: 600 7.5pt "Segoe UI", sans-serif; color: #2563eb; letter-spacing: .08em; }
  @top-right { content: "@DOC@"; font: 7.5pt "Segoe UI", sans-serif; color: #8a94a6; }
  @bottom-right { content: counter(page); font: 600 8pt "Segoe UI", sans-serif; color: #475467; }
  @bottom-left { content: "@EDITION@"; font: 7.5pt "Segoe UI", sans-serif; color: #98a2b3; }
}
@page cover { margin: 0; @top-left { content: none; } @top-right { content: none; }
              @bottom-right { content: none; } @bottom-left { content: none; } }
@page toc { @bottom-right { content: none; } }
:root { --ink: #101828; --text: #344054; --muted: #667085; --line: #e4e7ec; --accent: #2563eb; --accent2: #0ea5e9;
        --soft: #f5f8ff; --green: #12b76a; --amber: #f79009; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", Arial, sans-serif; font-size: 9.6pt;
       line-height: 1.55; color: var(--text); hyphens: auto; }

/* titulní strana */
.cover { page: cover; height: 296.5mm; position: relative; overflow: hidden; break-after: page;
         background: linear-gradient(160deg, #0b1f4d 0%, #123a8a 45%, #1d63d8 100%); color: #fff; }
.cover .grid { position: absolute; inset: 0; opacity: .13;
               background-image: linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px);
               background-size: 12mm 12mm; }
.cover .mesh { position: absolute; right: -30mm; top: 40mm; width: 170mm; opacity: .5; }
.cover .inner { position: absolute; left: 22mm; right: 22mm; top: 92mm; }
.cover .brand { font-size: 11pt; font-weight: 700; letter-spacing: .32em; text-transform: uppercase; color: #93c5fd; }
.cover h1 { font-size: 40pt; line-height: 1.05; margin: 6mm 0 4mm; color: #fff; font-weight: 750; letter-spacing: -.01em; }
.cover .sub { font-size: 13pt; color: #dbeafe; max-width: 140mm; line-height: 1.45; }
.cover .bar { width: 26mm; height: 2.2mm; border-radius: 2mm; margin: 9mm 0 0;
              background: linear-gradient(90deg, #38bdf8, #a5b4fc); }
.cover .foot { position: absolute; left: 22mm; right: 22mm; bottom: 20mm; display: flex; justify-content: space-between;
               font-size: 9pt; color: #bfdbfe; border-top: 1px solid rgba(255,255,255,.25); padding-top: 4mm; }

/* obsah */
.toc { page: toc; break-after: page; padding-top: 6mm; }
.toc h2 { font-size: 24pt; color: var(--ink); border: 0; margin: 0 0 6mm; font-weight: 700; }
.toc ol { list-style: none; padding: 0; margin: 0; counter-reset: none; }
.toc li { display: flex; align-items: baseline; gap: 3mm; padding: 1.9mm 0; border-bottom: 1px solid var(--line); }
.toc .no { flex: 0 0 9mm; font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
.toc a { color: var(--ink); text-decoration: none; font-size: 10.5pt; font-weight: 550; }
.toc .dots { flex: 1; border-bottom: 1.5px dotted #d0d5dd; transform: translateY(-1mm); }
.toc .pg { flex: 0 0 auto; font-weight: 600; color: var(--muted); font-variant-numeric: tabular-nums; }

/* kapitoly */
section.chapter { break-before: page; }
.chap-head { display: flex; align-items: flex-end; gap: 5mm; margin: 0 0 6mm; padding-bottom: 4mm;
             border-bottom: 2px solid var(--ink); }
.chap-no { font-size: 34pt; line-height: .9; font-weight: 800; color: transparent;
           -webkit-text-stroke: 1.2px var(--accent); font-variant-numeric: tabular-nums; }
.chap-kicker { font-size: 7.5pt; letter-spacing: .2em; text-transform: uppercase; color: var(--accent); font-weight: 700; }
.chap-head h1 { margin: 1mm 0 0; font-size: 21pt; line-height: 1.15; color: var(--ink); font-weight: 700; }
h2 { font-size: 13.5pt; color: var(--ink); margin: 7mm 0 2.5mm; font-weight: 650; break-after: avoid;
     padding-left: 3mm; border-left: 3px solid var(--accent); }
h3 { font-size: 11pt; color: #1d2939; margin: 5mm 0 1.5mm; font-weight: 650; break-after: avoid; }
p { margin: 0 0 2.6mm; orphans: 3; widows: 3; }
p.lead { font-size: 11pt; line-height: 1.6; color: #475467; margin-bottom: 5mm; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin: .8mm 0; }
ul li::marker { color: var(--accent); }
ol li::marker { color: var(--accent); font-weight: 700; }
a { color: var(--accent); text-decoration: none; }
b, strong { color: #1d2939; font-weight: 650; }

p.tip, p.warn { position: relative; border-radius: 2.5mm; padding: 3mm 4mm 3mm 11mm; margin: 3mm 0 4mm;
                break-inside: avoid; }
p.tip { background: #ecfdf3; border: 1px solid #abefc6; color: #05603a; }
p.warn { background: #fffaeb; border: 1px solid #fedf89; color: #7a2e0e; }
p.tip::before, p.warn::before { position: absolute; left: 3.6mm; top: 2.9mm; width: 4.6mm; height: 4.6mm;
                                border-radius: 50%; color: #fff; font-weight: 800; font-size: 8pt; line-height: 4.6mm;
                                text-align: center; }
p.tip::before { content: "i"; background: var(--green); }
p.warn::before { content: "!"; background: var(--amber); }
p.tip b, p.warn b { color: inherit; }

table.grid { width: 100%; border-collapse: separate; border-spacing: 0; margin: 3mm 0 5mm; font-size: 8.6pt;
             border: 1px solid var(--line); border-radius: 2.5mm; overflow: hidden; }
table.grid tr { break-inside: avoid; }
table.grid th { background: #f2f4f7; color: #1d2939; font-weight: 650; text-align: left; padding: 2mm 2.5mm;
                border-bottom: 1px solid var(--line); }
table.grid td { padding: 1.8mm 2.5mm; vertical-align: top; border-bottom: 1px solid #f2f4f7; }
table.grid tr:nth-child(even) td { background: #fcfcfd; }
table.grid tr:last-child td { border-bottom: 0; }
thead { display: table-header-group; }

span.key { display: inline-block; font: 600 7.8pt "Cascadia Mono", Consolas, monospace; color: #1d2939;
           background: linear-gradient(#ffffff, #f2f4f7); border: 1px solid #d0d5dd; border-bottom-width: 2px;
           border-radius: 1.2mm; padding: 0 1.4mm; line-height: 1.5; }
code { font: 8.4pt "Cascadia Mono", Consolas, monospace; background: #f2f4f7; color: #c01048; border-radius: 1mm;
       padding: .2mm 1mm; }
pre { font: 8.2pt/1.5 "Cascadia Mono", Consolas, monospace; background: #0f172a; color: #e2e8f0; border-radius: 2.5mm;
      padding: 3.5mm 4mm; white-space: pre-wrap; break-inside: avoid; margin: 2mm 0 4mm; }
pre code { background: none; color: inherit; padding: 0; }
img { max-width: 100%; border-radius: 2mm; border: 1px solid var(--line); box-shadow: 0 1mm 3mm rgba(16,24,40,.08);
      break-inside: avoid; }
hr { border: 0; border-top: 1px solid var(--line); margin: 6mm 0; }
"""

MESH_SVG = """<svg class="mesh" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="#bfdbfe"
 stroke-width="1.1">{}</svg>"""


def _mesh_art() -> str:
    """Decoration of the title page: a triangle mesh (deterministic)."""
    import math
    pts = []
    for j in range(9):
        for i in range(9):
            x = 20 + i * 45 + (22 if j % 2 else 0) + 9 * math.sin(i * 1.7 + j)
            y = 20 + j * 42 + 8 * math.cos(i + j * 2.1)
            pts.append((x, y))
    lines = []
    for j in range(8):
        for i in range(8):
            a, b = pts[j * 9 + i], pts[j * 9 + i + 1]
            c = pts[(j + 1) * 9 + i] if j % 2 == 0 else pts[(j + 1) * 9 + i + 1]
            lines.append(f'<path d="M{a[0]:.1f} {a[1]:.1f}L{b[0]:.1f} {b[1]:.1f}L{c[0]:.1f} {c[1]:.1f}Z"/>')
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="#93c5fd" stroke="none"/>' for x, y in pts[::3])
    return MESH_SVG.format("".join(lines) + dots)


def pages(lang: str, only: list[str] | None = None) -> list[tuple[str, str]]:
    """[(slug, name)] in the order of the help contents, names in the given language."""
    from fea_core import i18n
    from .help import PAGES
    old = i18n.language()
    i18n.set_language(lang)
    try:
        return [(s, i18n.tr(t)) for s, t in PAGES if only is None or s in only]
    finally:
        i18n.set_language(old)


def page_file(slug: str, lang: str) -> Path:
    """The English source is in help/<slug>.html, the Czech translation in help/cs/<slug>.html."""
    if lang == "en":
        p = HELP_DIR / f"{slug}.html"
        if p.exists():
            return p
    return HELP_DIR / "cs" / f"{slug}.html"


def _chapter(slug: str, title: str, no: int, lang: str) -> str:
    f = page_file(slug, lang)
    html = f.read_text(encoding="utf-8") if f.exists() else f"<h1>{escape(title)}</h1><p>{TEXTS[lang]['missing']}</p>"
    # links between pages → anchors, fea: links (GUI actions) → bold text
    html = re.sub(r'href="([a-z0-9_]+)\.html(#[^"]*)?"',
                  lambda m: f'href="#{m.group(2)[1:] if m.group(2) else m.group(1)}"', html)
    html = re.sub(r'<a href="fea:[^"]*">(.*?)</a>', r"<b>\1</b>", html, flags=re.S)

    def img(m):
        p = (f.parent / m.group(1)) if (f.parent / m.group(1)).exists() else HELP_DIR / m.group(1)
        if not p.exists():
            return '<img alt=""'
        mime = "image/svg+xml" if p.suffix == ".svg" else f"image/{p.suffix[1:].lower()}"
        return f'<img src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"'
    html = re.sub(r'<img src="([^"]+)"', img, html)
    # h1 heading of the page → chapter header (the number from the name "5. Constraints" is replaced by the order of the chapter)
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
    head_title = re.sub(r"^\s*\d+\.\s*", "", m.group(1).strip()) if m else escape(title)
    if m:
        html = html[:m.start()] + html[m.end():]
    head = (f'<div class="chap-head"><div class="chap-no">{no:02d}</div><div>'
            f'<div class="chap-kicker">{TEXTS[lang]["chapter"]} {no}</div><h1>{head_title}</h1></div></div>')
    return f'<section class="chapter" id="{slug}">{head}{html}</section>'


def build_html(lang: str = "cs", kind: str = "manual", page_numbers: dict | None = None, part: str = "body") -> str:
    """HTML of the manual: part = "body" (contents + chapters) or "cover" (title page, printed without margins)."""
    t = TEXTS[lang]
    only = TESTING_PAGES if kind == "testing" else None
    items = pages(lang, only)
    today = date.today()
    edition = f"{t['edition']} {today.day}. {today.month}. {today.year}" if lang == "cs" else \
        f"{t['edition']} {today:%B} {today.day}, {today.year}"
    doc = t[kind]
    toc = []
    for i, (slug, title) in enumerate(items, 1):
        pg = (page_numbers or {}).get(slug, "")
        name = re.sub(r"^\s*\d+\.\s*", "", title)
        toc.append(f'<li><span class="no">{i:02d}</span><a href="#{slug}">{escape(name)}</a>'
                   f'<span class="dots"></span><span class="pg">{pg}</span></li>')
    chapters = "".join(_chapter(s, ti, i, lang) for i, (s, ti) in enumerate(items, 1))
    from fea_core import __version__
    ver = ".".join(__version__.split(".")[:2])          # verze programu (fea_core.__version__)
    ver_label = f"Verze {ver}" if lang == "cs" else f"Version {ver}"
    cover = (f'<div class="cover"><div class="grid"></div>{_mesh_art()}<div class="inner">'
             f'<div class="brand">FEA Studio &middot; {ver_label}</div><h1>{escape(doc)}</h1>'
             f'<div class="sub">{escape(t["sub_" + kind])}</div><div class="bar"></div></div>'
             f'<div class="foot"><span>CalculiX · Gmsh · NGSolve</span><span>{escape(edition)} · FEA Studio {ver}</span></div></div>')
    css = CSS.replace("@DOC@", doc.replace('"', "")).replace("@EDITION@", edition).replace("@VER@", ver)
    body = cover if part == "cover" else f'<div class="toc"><h2>{t["toc"]}</h2><ol>{"".join(toc)}</ol></div>{chapters}'
    return (f'<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8"><title>FEA Studio – {escape(doc)}'
            f"</title><style>{css}</style></head><body>{body}</body></html>")


# ------------------------------------------------------------------ printing through Qt WebEngine
def _app():
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        app = QApplication(sys.argv[:1] + ["--disable-gpu"])
    return app


def print_pdf(html: str, path: Path, margins=MARGINS_MM) -> Path:
    """HTML → PDF (A4; the page margins [mm] are passed by Qt, the content of the margins – header and footer – is in the CSS @page)."""
    from PySide6.QtCore import QEventLoop, QMarginsF, QUrl
    from PySide6.QtGui import QPageLayout, QPageSize
    from PySide6.QtWebEngineCore import QWebEnginePage
    _app()
    page = QWebEnginePage()
    loop = QEventLoop()
    ok = {}
    page.loadFinished.connect(lambda r: (ok.setdefault("load", r), loop.quit()))
    # large HTML (images base64) via a temporary file – setHtml has a 2 MB limit
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "doc.html"
        src.write_text(html, encoding="utf-8")
        page.load(QUrl.fromLocalFile(str(src)))
        loop.exec()
        if not ok.get("load"):
            raise RuntimeError("Qt WebEngine did not load the manual")
        layout = QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Portrait, QMarginsF(*margins),
                             QPageLayout.Unit.Millimeter)
        page.pdfPrintingFinished.connect(lambda p, r: (ok.setdefault("pdf", r), loop.quit()))
        path = Path(path)
        page.printToPdf(str(path), layout)
        loop.exec()
    page.deleteLater()
    if not ok.get("pdf"):
        raise RuntimeError(f"Printing the PDF failed: {path}")
    return path


def chapter_pages(pdf: Path, slugs: list[str], lang: str) -> dict[str, int]:
    """Page (from 1) of each chapter according to the heading "KAPITOLA n" (the Czech word is searched in the text of the PDF)."""
    import pypdfium2 as pdfium
    word = re.sub(r"\s", "", TEXTS[lang]["chapter"]).upper()
    # "K A P I T O L A 1 2" (the Czech heading, as in the PDF text) – letter-spaced text gives spaces even between the digits
    kicker = re.compile(r"\s*".join(map(re.escape, word)) + r"[ \t]*((?:\d[ \t]*)+)")
    doc = pdfium.PdfDocument(str(pdf))
    found = {}
    try:
        for i in range(len(doc)):
            tp = doc[i].get_textpage()
            text = tp.get_text_range().upper()
            tp.close()
            for m in kicker.finditer(text):
                k = int(re.sub(r"\s", "", m.group(1))) - 1
                if 0 <= k < len(slugs):
                    found.setdefault(slugs[k], i + 1)
    finally:
        doc.close()
    return found


def export_pdf(path: str | Path, lang: str = "cs", kind: str = "manual") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pypdfium2 as pdfium
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        first = print_pdf(build_html(lang, kind), tmp / "pass1.pdf")
        slugs = [x for x, _ in pages(lang, TESTING_PAGES if kind == "testing" else None)]
        numbers = chapter_pages(first, slugs, lang)
        body = print_pdf(build_html(lang, kind, numbers), tmp / "body.pdf")
        cover = print_pdf(build_html(lang, kind, part="cover"), tmp / "cover.pdf", (0, 0, 0, 0))
        # title page before the contents; page numbers in the footer and in the contents count from the contents (the title page has none)
        doc, cov = pdfium.PdfDocument(str(body)), pdfium.PdfDocument(str(cover))
        try:
            doc.import_pages(cov, [0], index=0)
            doc.save(str(path))
        finally:
            cov.close()
            doc.close()
    if _has_local_links(path.read_bytes()):
        # an untranslated link between pages → Chromium writes file:/// with the path of the temporary folder (user name)
        path.unlink()
        raise RuntimeError(f"The PDF contains a file:/// link to a local file – fix the link in the help: {path.name}")
    return path


def _has_local_links(data: bytes) -> bool:
    """True when the PDF (also in compressed streams) contains a link file:///."""
    import zlib
    if b"file:///" in data:
        return True
    for m in re.finditer(rb"stream\r?\n", data):
        end = data.find(b"endstream", m.end())
        if end < 0:
            continue
        try:
            if b"file:///" in zlib.decompress(data[m.end():end]):
                return True
        except zlib.error:
            pass
    return False


def export_all(out: str | Path, langs=("cs", "en"), kinds=("manual", "testing")) -> list[Path]:
    out = Path(out)
    res = []
    for lang in langs:
        for kind in kinds:
            res.append(export_pdf(out / TEXTS[lang][f"file_{kind}"], lang, kind))
    return res


# ------------------------------------------------------------------ checking the translations of the help
SOURCES_JSON = HELP_DIR / "cs" / "sources.json"      # page → SHA-1 of the English source from which the Czech translation was made


def _sha(path: Path) -> str:
    import hashlib
    # line endings do not count (git checks the pages out with CRLF on Windows, LF elsewhere)
    return hashlib.sha1(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def stale_translations() -> list[str]:
    """Help pages without a Czech translation or changed since the translation (according to sources.json)."""
    import json
    try:
        known = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    out = []
    for f in sorted(HELP_DIR.glob("*.html")):
        cs = HELP_DIR / "cs" / f.name
        if not cs.exists() or known.get(f.stem) != _sha(f):
            out.append(f.stem)
    return out


def mark_translated(slugs: list[str]) -> None:
    """Writes to sources.json that the Czech translations match the current English version."""
    import json
    try:
        known = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    for s in slugs:
        known[s] = _sha(HELP_DIR / f"{s}.html")
    SOURCES_JSON.write_text(json.dumps(dict(sorted(known.items())), indent=1) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Příručka a návod k testování FEA Studia jako PDF")
    ap.add_argument("-o", "--out", default="docs", help="cílová složka (výchozí docs)")
    ap.add_argument("--lang", nargs="+", default=["cs", "en"], choices=sorted(TEXTS))
    ap.add_argument("--only", choices=["manual", "testing"], help="jen příručka / jen návod k testování")
    ap.add_argument("--stale", action="store_true", help="vypsat stránky, jejichž anglický překlad chybí / je starší")
    ap.add_argument("--mark-translated", nargs="+", metavar="SLUG", help="zapsat, že překlad stránek je aktuální")
    a = ap.parse_args(argv)
    if a.stale or a.mark_translated:
        if a.mark_translated:
            mark_translated(a.mark_translated)
        print("\n".join(stale_translations()) or "the Czech help translations are up to date")
        return 0
    kinds = (a.only,) if a.only else ("manual", "testing")
    for p in export_all(a.out, a.lang, kinds):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
