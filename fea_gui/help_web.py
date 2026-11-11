"""Standalone HTML manual: one file without dependencies (CSS + JS inline), search, expandable contents,
support for merged language variants (CZ / EN) with instant switching without reloading the page.

The input is the fragments of the help pages (fea_gui/help/*.html = English, fea_gui/help/cs/*.html = Czech), the output one page:
  * top bar – language switch (🇨🇿 Čeština / 🇬🇧 English), live search (filter + highlight, `/` or Ctrl+K),
    expand / collapse all, light / dark theme;
  * left panel – the contents as a tree (chapter → sections), follows the reading (scrollspy), slides out in a narrow window;
  * chapters split into collapsible sections (<details> by the h2 headings), links between pages open the target section;
  * chapter cards on the start page, notes .tip / .warn, keys, tables, code with a Copy button;
  * printing (Ctrl+P) expands everything and hides the controls.
No Qt and no extra Python – only the standard library. Interface: `build_site(...)`, `build_bilingual_site(...)`, `prepare_chapter(...)`.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from html import escape, unescape
from pathlib import Path

DEFAULT_OPEN = True          # chapter sections expanded when opened (False = headings only, the reader expands them)

UI = {
    "cs": {
        "search": "Hledat v příručce…", "search_aria": "Hledat v příručce", "only_hits": "Jen shody",
        "expand": "Expand all", "collapse": "Collapse all", "theme": "Světlý / tmavý vzhled",
        "toc": "Obsah", "chapters": "Kapitoly", "edition": "Vydání", "version": "verze",
        "no_hits": "Nic nenalezeno", "hits": "{n} oddílů", "next": "Další shoda (Enter)", "prev": "Předchozí (Shift+Enter)",
        "clear": "Vymazat (Esc)", "copy": "Copy", "copied": "Zkopírováno", "top": "Nahoru",
        "menu": "Obsah", "hint": "Hledání: <span class=\"key\">/</span> nebo <span class=\"key\">Ctrl</span>+<span class=\"key\">K</span>",
        "start": "Kapitoly příručky", "chapter": "Kapitola", "page_title": "Manual", "lang": "Jazyk",
    },
    "en": {
        "search": "Search the manual…", "search_aria": "Search the manual", "only_hits": "Matches only",
        "expand": "Expand all", "collapse": "Collapse all", "theme": "Light / dark theme",
        "toc": "Contents", "chapters": "Chapters", "edition": "Edition", "version": "version",
        "no_hits": "Nothing found", "hits": "{n} sections", "next": "Next match (Enter)", "prev": "Previous (Shift+Enter)",
        "clear": "Clear (Esc)", "copy": "Copy", "copied": "Copied", "top": "Back to top",
        "menu": "Contents", "hint": "Search: <span class=\"key\">/</span> or <span class=\"key\">Ctrl</span>+<span class=\"key\">K</span>",
        "start": "Manual chapters", "chapter": "Chapter", "page_title": "Manual", "lang": "Language",
    },
}

LOGO = ('<svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true"><g fill="none" stroke="#bfdbfe" stroke-width="1.4" '
        'stroke-linejoin="round"><path d="M3 24 14 5l15 9-5 13z"/><path d="M14 5 24 27M3 24l21 3M14 5l15 9"/></g>'
        '<g fill="#38bdf8"><circle cx="3" cy="24" r="2"/><circle cx="14" cy="5" r="2"/><circle cx="29" cy="14" r="2"/>'
        '<circle cx="24" cy="27" r="2"/></g></svg>')


@dataclass
class Chapter:
    slug: str
    title: str                  # name from the help contents (e.g. "5. Contacts")
    h1: str                     # page heading (HTML)
    intro: str                  # HTML before the first h2
    sections: list[tuple[str, str, str]]     # (id, heading as HTML, content HTML)
    lead: str = ""              # plain text of the first lead paragraph (card)


# ------------------------------------------------------------------ preparing the page
def _plain(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()


def _embed_images(html: str, dirs: list[Path]) -> str:
    def img(m):
        for d in dirs:
            p = d / m.group(1)
            if p.exists():
                mime = "image/svg+xml" if p.suffix == ".svg" else f"image/{p.suffix[1:].lower()}"
                return f'<img loading="lazy" alt="" src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"'
        return '<img alt=""'
    return re.sub(r'<img src="([^"]+)"', img, html)


def prepare_chapter(slug: str, title: str, html: str, image_dirs: list[Path] | None = None) -> Chapter:
    """Fragment of a help page → chapter: links to the anchors of the document, images base64, sections by h2."""
    # anchor <a name="x"></a> → id with the page prefix
    html = re.sub(r'<a name="([^"]+)"></a>', lambda m: f'<a class="anchor" id="{slug}--{m.group(1)}"></a>', html)
    # links between pages → anchors of the document; program actions (fea:…) make no sense in the file → bold
    html = re.sub(r'href="([a-z0-9_]+)\.html(?:#([^"]*))?"',
                  lambda m: f'href="#{m.group(1)}--{m.group(2)}"' if m.group(2) else f'href="#{m.group(1)}"', html)
    html = re.sub(r'<a href="fea:[^"]*">(.*?)</a>', r"<b>\1</b>", html, flags=re.S)
    html = _embed_images(html, image_dirs or [])
    html = re.sub(r"(<table\b.*?</table>)", r'<div class="tablewrap">\1</div>', html, flags=re.S)
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
    h1 = m.group(1).strip() if m else escape(title)
    if m:
        html = html[:m.start()] + html[m.end():]
    parts = re.split(r"(?=<h2\b)", html)
    intro, sections = parts[0], []
    for i, part in enumerate(parts[1:], 1):
        hm = re.match(r"<h2[^>]*>(.*?)</h2>", part, re.S)
        if not hm:
            intro += part
            continue
        sections.append((f"{slug}--s{i}", hm.group(1).strip(), part[hm.end():]))
    lm = re.search(r'<p class="lead">(.*?)</p>', intro, re.S)
    return Chapter(slug, title, h1, intro, sections, _plain(lm.group(1)) if lm else "")


# ------------------------------------------------------------------ building the page
def _nav(chapters: list[Chapter]) -> str:
    out = []
    for c in chapters:
        subs = "".join(f'<li><a href="#{sid}" data-spy="{sid}">{escape(_plain(h))}</a></li>' for sid, h, _ in c.sections)
        tw = '<button class="tw" aria-label="+" aria-expanded="false"></button>' if subs else '<span class="tw"></span>'
        out.append(f'<li class="nav-ch" data-ch="{c.slug}">{tw}<a class="chlink" href="#{c.slug}" data-spy="{c.slug}">'
                   f'{escape(c.title)}</a>{"<ul>" + subs + "</ul>" if subs else ""}</li>')
    return "".join(out)


def _nav_bilingual(chapters_cs: list[Chapter], chapters_en: list[Chapter]) -> str:
    out = []
    for c_cs, c_en in zip(chapters_cs, chapters_en):
        subs = []
        for (sid, h_cs, _), (_, h_en, _) in zip(c_cs.sections, c_en.sections):
            subs.append(f'<li><a href="#{sid}" data-spy="{sid}">'
                        f'<span class="lang-pane cs">{escape(_plain(h_cs))}</span>'
                        f'<span class="lang-pane en">{escape(_plain(h_en))}</span></a></li>')
        subs_html = "".join(subs)
        tw = '<button class="tw" aria-label="+" aria-expanded="false"></button>' if subs else '<span class="tw"></span>'
        out.append(f'<li class="nav-ch" data-ch="{c_cs.slug}">{tw}<a class="chlink" href="#{c_cs.slug}" data-spy="{c_cs.slug}">'
                   f'<span class="lang-pane cs">{escape(c_cs.title)}</span>'
                   f'<span class="lang-pane en">{escape(c_en.title)}</span></a>'
                   f'{"<ul>" + subs_html + "</ul>" if subs else ""}</li>')
    return "".join(out)


def _cards(chapters: list[Chapter]) -> str:
    out = []
    for i, c in enumerate(chapters, 1):
        lead = c.lead if len(c.lead) <= 150 else c.lead[:147].rsplit(" ", 1)[0] + "…"
        out.append(f'<a class="card" href="#{c.slug}" data-card="{c.slug}"><span class="card-no">{i:02d}</span>'
                   f'<span class="card-t">{escape(c.title)}</span><span class="card-d">{escape(lead)}</span></a>')
    return "".join(out)


def _cards_bilingual(chapters_cs: list[Chapter], chapters_en: list[Chapter]) -> str:
    out = []
    for i, (c_cs, c_en) in enumerate(zip(chapters_cs, chapters_en), 1):
        lead_cs = c_cs.lead if len(c_cs.lead) <= 150 else c_cs.lead[:147].rsplit(" ", 1)[0] + "…"
        lead_en = c_en.lead if len(c_en.lead) <= 150 else c_en.lead[:147].rsplit(" ", 1)[0] + "…"
        out.append(f'<a class="card" href="#{c_cs.slug}" data-card="{c_cs.slug}"><span class="card-no">{i:02d}</span>'
                   f'<span class="card-t"><span class="lang-pane cs">{escape(c_cs.title)}</span><span class="lang-pane en">{escape(c_en.title)}</span></span>'
                   f'<span class="card-d"><span class="lang-pane cs">{escape(lead_cs)}</span><span class="lang-pane en">{escape(lead_en)}</span></span></a>')
    return "".join(out)


def _chapter_html(c: Chapter, no: int, ui: dict) -> str:
    opened = " open" if DEFAULT_OPEN else ""
    secs = "".join(
        f'<details class="sec" id="{sid}" data-sec data-ch="{c.slug}"{opened}><summary><h2>{h}</h2></summary>'
        f'<div class="sec-body">{body}</div></details>' for sid, h, body in c.sections)
    return (f'<section class="chapter" id="{c.slug}" data-chapter><header class="chap-head">'
            f'<div class="chap-kicker">{ui["chapter"]} {no}</div><h1>{c.h1}</h1></header>'
            f'<div class="chap-intro" data-sec data-ch="{c.slug}">{c.intro}</div>{secs}</section>')


def _chapter_bilingual(c_cs: Chapter, c_en: Chapter, no: int, ui_cs: dict, ui_en: dict) -> str:
    opened = " open" if DEFAULT_OPEN else ""
    secs = []
    for (sid, h_cs, body_cs), (_, h_en, body_en) in zip(c_cs.sections, c_en.sections):
        secs.append(
            f'<details class="sec" id="{sid}" data-sec data-ch="{c_cs.slug}"{opened}>'
            f'<summary><h2 class="lang-pane cs">{h_cs}</h2><h2 class="lang-pane en">{h_en}</h2></summary>'
            f'<div class="sec-body"><div class="lang-pane cs">{body_cs}</div><div class="lang-pane en">{body_en}</div></div></details>'
        )
    secs_html = "".join(secs)
    return (f'<section class="chapter" id="{c_cs.slug}" data-chapter><header class="chap-head">'
            f'<div class="chap-kicker"><span class="lang-pane cs">{ui_cs["chapter"]} {no}</span><span class="lang-pane en">{ui_en["chapter"]} {no}</span></div>'
            f'<h1><span class="lang-pane cs">{c_cs.h1}</span><span class="lang-pane en">{c_en.h1}</span></h1></header>'
            f'<div class="chap-intro" data-sec data-ch="{c_cs.slug}"><div class="lang-pane cs">{c_cs.intro}</div><div class="lang-pane en">{c_en.intro}</div></div>'
            f'{secs_html}</section>')


def build_site(chapters: list[Chapter], *, title: str, subtitle: str, version: str, edition: str, lang: str = "cs") -> str:
    ui = UI.get(lang, UI["cs"])
    body = "".join(_chapter_html(c, i, ui) for i, c in enumerate(chapters, 1))
    js = JS.replace("@UI_ALL@", json.dumps(UI, ensure_ascii=False))
    return (
        f'<!DOCTYPE html><html lang="{lang}" data-lang="{lang}" data-theme="light"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>'
        f'<script>try{{var t=localStorage.getItem("fea-help-theme")||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");'
        f'document.documentElement.dataset.theme=t}}catch(e){{}}</script>'
        f"<style>{CSS}</style></head><body>"
        f'<div id="progress"></div>'
        f'<header class="bar"><button id="menu" class="ibtn menu" aria-label="{ui["menu"]}" title="{ui["menu"]}">'
        f'<svg viewBox="0 0 24 24" width="20" height="20"><path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" fill="none"/></svg></button>'
        f'<a class="brand" href="#top">{LOGO}<span><b>FEA Studio</b><small>{escape(subtitle)} · {ui["version"]} {escape(version)}</small></span></a>'
        f'<div class="search"><svg viewBox="0 0 24 24" width="16" height="16" class="s-ico"><circle cx="11" cy="11" r="6.5" '
        f'stroke="currentColor" stroke-width="2" fill="none"/><path d="m16 16 4.5 4.5" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round"/></svg><input id="q" type="search" autocomplete="off" spellcheck="false" placeholder="{ui["search"]}" '
        f'aria-label="{ui["search_aria"]}"><span id="count" class="count"></span>'
        f'<button id="prev" class="ibtn sm" title="{ui["prev"]}" aria-label="{ui["prev"]}">▲</button>'
        f'<button id="next" class="ibtn sm" title="{ui["next"]}" aria-label="{ui["next"]}">▼</button>'
        f'<button id="clear" class="ibtn sm" title="{ui["clear"]}" aria-label="{ui["clear"]}">✕</button>'
        f'<div id="results" role="listbox" hidden></div></div>'
        f'<label class="chk" title="{ui["only_hits"]}"><input id="onlyhits" type="checkbox" checked><span>{ui["only_hits"]}</span></label>'
        f'<button id="expand" class="btn">{ui["expand"]}</button><button id="collapse" class="btn">{ui["collapse"]}</button>'
        f'<button id="theme" class="ibtn" title="{ui["theme"]}" aria-label="{ui["theme"]}">'
        f'<svg class="i-moon" viewBox="0 0 24 24" width="18" height="18"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" fill="currentColor"/></svg>'
        f'<svg class="i-sun" viewBox="0 0 24 24" width="18" height="18"><circle cx="12" cy="12" r="4.2" fill="currentColor"/>'
        f'<path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button></header>'
        f'<div class="layout"><aside id="side" aria-label="{ui["toc"]}"><div class="side-h">{ui["toc"]}</div>'
        f'<nav><ul class="nav">{_nav(chapters)}</ul></nav></aside><div id="scrim"></div>'
        f'<main id="top"><section class="hero"><div class="hero-in"><div class="hero-brand">FEA Studio</div>'
        f'<h1>{escape(title)}</h1><p>{escape(subtitle)}</p><div class="chips"><span class="chip">{ui["version"]} {escape(version)}</span>'
        f'<span class="chip soft">{escape(edition)}</span><span class="chip soft">{ui["hint"]}</span></div></div></section>'
        f'<h2 class="cards-h">{ui["start"]}</h2><div class="cards">{_cards(chapters)}</div>{body}'
        f'<footer class="foot">FEA Studio {escape(version)} · {escape(edition)}</footer></main></div>'
        f'<button id="totop" class="ibtn totop" title="{ui["top"]}" aria-label="{ui["top"]}">↑</button>'
        f"<script>{js}</script></body></html>")


def build_bilingual_site(chapters_cs: list[Chapter], chapters_en: list[Chapter], *,
                         title_cs: str, title_en: str, subtitle_cs: str, subtitle_en: str,
                         version: str, edition_cs: str, edition_en: str, default_lang: str = "cs") -> str:
    """Builds the merged bilingual HTML manual with an instant language switch."""
    ui_cs, ui_en = UI["cs"], UI["en"]
    cur_ui = UI.get(default_lang, ui_cs)
    cur_title = title_cs if default_lang == "cs" else title_en
    body = "".join(_chapter_bilingual(c_cs, c_en, i, ui_cs, ui_en) for i, (c_cs, c_en) in enumerate(zip(chapters_cs, chapters_en), 1))
    js = JS.replace("@UI_ALL@", json.dumps(UI, ensure_ascii=False))

    return (
        f'<!DOCTYPE html><html lang="{default_lang}" data-lang="{default_lang}" data-theme="light" '
        f'data-title-cs="{escape(title_cs)}" data-title-en="{escape(title_en)}"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(cur_title)}</title>'
        f'<script>try{{var t=localStorage.getItem("fea-help-theme")||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");'
        f'document.documentElement.dataset.theme=t}}catch(e){{}}'
        f'try{{var qm=location.search.match(/[?&]lang=([a-z]{{2}})/);var saved=qm?qm[1]:localStorage.getItem("fea-help-lang");'
        f'if(saved&&(saved==="cs"||saved==="en"))document.documentElement.dataset.lang=saved;}}catch(e){{}}</script>'
        f"<style>{CSS}</style></head><body>"
        f'<div id="progress"></div>'
        f'<header class="bar"><button id="menu" class="ibtn menu" aria-label="{cur_ui["menu"]}" title="{cur_ui["menu"]}">'
        f'<svg viewBox="0 0 24 24" width="20" height="20"><path d="M4 6h16M4 12h16M4 18h16" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" fill="none"/></svg></button>'
        f'<a class="brand" href="#top">{LOGO}<span><b>FEA Studio</b><small>'
        f'<span class="lang-pane cs">{escape(subtitle_cs)}</span><span class="lang-pane en">{escape(subtitle_en)}</span> · '
        f'<span class="lang-pane cs">{ui_cs["version"]}</span><span class="lang-pane en">{ui_en["version"]}</span> {escape(version)}'
        f'</small></span></a>'
        f'<div class="search"><svg viewBox="0 0 24 24" width="16" height="16" class="s-ico"><circle cx="11" cy="11" r="6.5" '
        f'stroke="currentColor" stroke-width="2" fill="none"/><path d="m16 16 4.5 4.5" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round"/></svg><input id="q" type="search" autocomplete="off" spellcheck="false" placeholder="{cur_ui["search"]}" '
        f'aria-label="{cur_ui["search_aria"]}"><span id="count" class="count"></span>'
        f'<button id="prev" class="ibtn sm" title="{cur_ui["prev"]}" aria-label="{cur_ui["prev"]}">▲</button>'
        f'<button id="next" class="ibtn sm" title="{cur_ui["next"]}" aria-label="{cur_ui["next"]}">▼</button>'
        f'<button id="clear" class="ibtn sm" title="{cur_ui["clear"]}" aria-label="{cur_ui["clear"]}">✕</button>'
        f'<div id="results" role="listbox" hidden></div></div>'
        f'<label class="chk" title="{cur_ui["only_hits"]}"><input id="onlyhits" type="checkbox" checked>'
        f'<span class="lang-pane cs">{ui_cs["only_hits"]}</span><span class="lang-pane en">{ui_en["only_hits"]}</span></label>'
        f'<div class="lang-switch" role="group" aria-label="Language">'
        f'<button type="button" class="btn-lang{" active" if default_lang == "cs" else ""}" data-lang="cs" title="Čeština">🇨🇿 Čeština</button>'
        f'<button type="button" class="btn-lang{" active" if default_lang == "en" else ""}" data-lang="en" title="English">🇬🇧 English</button>'
        f'</div>'
        f'<button id="expand" class="btn"><span class="lang-pane cs">{ui_cs["expand"]}</span><span class="lang-pane en">{ui_en["expand"]}</span></button>'
        f'<button id="collapse" class="btn"><span class="lang-pane cs">{ui_cs["collapse"]}</span><span class="lang-pane en">{ui_en["collapse"]}</span></button>'
        f'<button id="theme" class="ibtn" title="{cur_ui["theme"]}" aria-label="{cur_ui["theme"]}">'
        f'<svg class="i-moon" viewBox="0 0 24 24" width="18" height="18"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" fill="currentColor"/></svg>'
        f'<svg class="i-sun" viewBox="0 0 24 24" width="18" height="18"><circle cx="12" cy="12" r="4.2" fill="currentColor"/>'
        f'<path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button></header>'
        f'<div class="layout"><aside id="side" aria-label="{cur_ui["toc"]}"><div class="side-h">'
        f'<span class="lang-pane cs">{ui_cs["toc"]}</span><span class="lang-pane en">{ui_en["toc"]}</span></div>'
        f'<nav><ul class="nav">{_nav_bilingual(chapters_cs, chapters_en)}</ul></nav></aside><div id="scrim"></div>'
        f'<main id="top"><section class="hero"><div class="hero-in"><div class="hero-brand">FEA Studio</div>'
        f'<h1><span class="lang-pane cs">{escape(title_cs)}</span><span class="lang-pane en">{escape(title_en)}</span></h1>'
        f'<p><span class="lang-pane cs">{escape(subtitle_cs)}</span><span class="lang-pane en">{escape(subtitle_en)}</span></p>'
        f'<div class="chips"><span class="chip"><span class="lang-pane cs">{ui_cs["version"]} {escape(version)}</span>'
        f'<span class="lang-pane en">{ui_en["version"]} {escape(version)}</span></span>'
        f'<span class="chip soft"><span class="lang-pane cs">{escape(edition_cs)}</span>'
        f'<span class="lang-pane en">{escape(edition_en)}</span></span>'
        f'<span class="chip soft"><span class="lang-pane cs">{ui_cs["hint"]}</span>'
        f'<span class="lang-pane en">{ui_en["hint"]}</span></span></div></div></section>'
        f'<h2 class="cards-h"><span class="lang-pane cs">{ui_cs["start"]}</span>'
        f'<span class="lang-pane en">{ui_en["start"]}</span></h2>'
        f'<div class="cards">{_cards_bilingual(chapters_cs, chapters_en)}</div>{body}'
        f'<footer class="foot">FEA Studio {escape(version)} · '
        f'<span class="lang-pane cs">{escape(edition_cs)}</span><span class="lang-pane en">{escape(edition_en)}</span></footer></main></div>'
        f'<button id="totop" class="ibtn totop" title="{cur_ui["top"]}" aria-label="{cur_ui["top"]}">↑</button>'
        f"<script>{js}</script></body></html>")


# ------------------------------------------------------------------ vzhled
CSS = r"""
:root{--bar-h:58px;--side-w:312px;--bg:#f4f7fb;--panel:#fff;--ink:#0f172a;--text:#334155;--muted:#64748b;--line:#e2e8f0;
--accent:#2563eb;--accent-ink:#1d4ed8;--soft:#eff6ff;--soft2:#f8fafc;--code-bg:#f1f5f9;--code:#be123c;--pre-bg:#0f172a;--pre:#e2e8f0;
--tip-bg:#ecfdf3;--tip-bd:#a6f4c5;--tip:#05603a;--tip-i:#12b76a;--warn-bg:#fffaeb;--warn-bd:#fedf89;--warn:#7a2e0e;--warn-i:#f79009;
--th:#e8eefb;--zebra:#f8fafc;--hit:#fde68a;--hit-cur:#fb923c;--shadow:0 1px 2px rgba(15,23,42,.06),0 6px 20px rgba(15,23,42,.06);--r:12px}
[data-theme=dark]{--bg:#0b1220;--panel:#111b30;--ink:#e8eefc;--text:#c5d0e4;--muted:#8d9bb8;--line:#22304d;--accent:#60a5fa;
--accent-ink:#93c5fd;--soft:#14213f;--soft2:#0f192e;--code-bg:#18243f;--code:#f9a8c4;--pre-bg:#070d1a;--pre:#d5def0;--tip-bg:#0d2a22;
--tip-bd:#14532d;--tip:#a7f3d0;--warn-bg:#2d2108;--warn-bd:#713f12;--warn:#fde68a;--th:#182645;--zebra:#0f192e;--hit:#854d0e;--hit-cur:#c2410c;
--shadow:0 1px 2px rgba(0,0,0,.4),0 6px 20px rgba(0,0,0,.35)}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:calc(var(--bar-h) + 14px)}
body{margin:0;background:var(--bg);color:var(--text);font:15.5px/1.62 "Segoe UI Variable Text","Segoe UI",Inter,system-ui,-apple-system,Roboto,Arial,sans-serif;
-webkit-font-smoothing:antialiased}
#progress{position:fixed;left:0;top:0;height:3px;width:0;background:linear-gradient(90deg,#38bdf8,#818cf8);z-index:60}
/* horní lišta */
.bar{position:sticky;top:0;z-index:50;height:var(--bar-h);display:flex;align-items:center;gap:10px;padding:0 16px;
background:linear-gradient(100deg,#0b1f4d,#123a8a 55%,#1d63d8);color:#fff;box-shadow:0 2px 14px rgba(11,31,77,.35)}
.brand{display:flex;align-items:center;gap:10px;color:#fff;text-decoration:none;min-width:0}
.brand span{display:flex;flex-direction:column;line-height:1.15}.brand b{font-size:15px;letter-spacing:.04em}
.brand small{font-size:11px;color:#bfdbfe;white-space:nowrap}
.search{position:relative;flex:1;max-width:560px;display:flex;align-items:center;gap:2px;margin-left:10px;background:rgba(255,255,255,.14);
border:1px solid rgba(255,255,255,.25);border-radius:10px;padding:0 6px 0 10px;transition:background .15s,box-shadow .15s}
.search:focus-within{background:rgba(255,255,255,.96);box-shadow:0 0 0 3px rgba(147,197,253,.55);color:#0f172a}
.s-ico{flex:none;opacity:.8}.search input{flex:1;min-width:0;height:36px;border:0;background:transparent;color:inherit;font:inherit;font-size:14px;outline:none;padding:0 8px}
.search input::placeholder{color:#dbeafe}.search:focus-within input::placeholder{color:#64748b}
.search input::-webkit-search-cancel-button{display:none}
.count{font-size:12px;opacity:.85;white-space:nowrap;padding:0 4px}
.ibtn{display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;border:0;border-radius:8px;background:transparent;color:inherit;
cursor:pointer;font-size:13px}.ibtn:hover{background:rgba(255,255,255,.16)}.search:focus-within .ibtn:hover{background:#e2e8f0}
.ibtn.sm{width:26px;height:26px;font-size:11px}.search:not(.active) #prev,.search:not(.active) #next,.search:not(.active) #clear,.search:not(.active) .count{display:none}
.btn{height:34px;padding:0 12px;border:1px solid rgba(255,255,255,.28);border-radius:8px;background:rgba(255,255,255,.1);color:#fff;font:inherit;font-size:13px;cursor:pointer;white-space:nowrap}
.btn:hover{background:rgba(255,255,255,.2)}
.chk{display:flex;align-items:center;gap:6px;font-size:13px;color:#dbeafe;white-space:nowrap;cursor:pointer}.chk input{accent-color:#38bdf8}
/* přepínač jazyků */
.lang-switch{display:inline-flex;align-items:center;background:rgba(0,0,0,.22);border:1px solid rgba(255,255,255,.25);border-radius:20px;padding:2px;gap:2px;flex:none}
.btn-lang{border:0;background:transparent;color:#dbeafe;font:inherit;font-size:12px;font-weight:600;padding:3px 9px;border-radius:14px;cursor:pointer;transition:all .15s;line-height:1.2;display:inline-flex;align-items:center;gap:4px}
.btn-lang:hover{color:#fff;background:rgba(255,255,255,.16)}
.btn-lang.active{background:#38bdf8;color:#04243d;font-weight:700;box-shadow:0 1px 3px rgba(0,0,0,.25)}
/* dvojjazyčné zobrazení */
[data-lang="cs"] .lang-pane.en{display:none!important}
[data-lang="en"] .lang-pane.cs{display:none!important}
.menu{display:none}.i-sun{display:none}[data-theme=dark] .i-sun{display:block}[data-theme=dark] .i-moon{display:none}
#results{position:absolute;left:0;right:0;top:calc(100% + 8px);max-height:min(70vh,560px);overflow:auto;background:var(--panel);color:var(--text);
border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:6px}
#results[hidden]{display:none}
.res{display:block;padding:8px 10px;border-radius:8px;text-decoration:none;color:inherit;cursor:pointer}
.res:hover,.res.sel{background:var(--soft)}.res .rt{font-weight:600;color:var(--ink);font-size:13.5px}.res .rc{font-size:11.5px;color:var(--accent);margin-bottom:1px}
.res .rs{font-size:12.5px;color:var(--muted);line-height:1.45}.res.none{color:var(--muted);cursor:default;text-align:center;padding:16px}
/* rozložení */
.layout{display:grid;grid-template-columns:var(--side-w) minmax(0,1fr);align-items:start}
#side{position:sticky;top:var(--bar-h);height:calc(100vh - var(--bar-h));overflow:auto;background:var(--panel);border-right:1px solid var(--line);padding:14px 10px 40px 12px}
.side-h{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:700;padding:4px 8px 8px}
.nav,.nav ul{list-style:none;margin:0;padding:0}.nav-ch{position:relative;margin:1px 0}
.nav a{display:block;color:var(--text);text-decoration:none;border-radius:8px;padding:5px 8px;font-size:13.5px;line-height:1.35}
.nav a:hover{background:var(--soft);color:var(--ink)}.nav .chlink{margin-left:20px;font-weight:600;color:var(--ink)}
.nav ul{display:none;margin:2px 0 6px 20px;padding-left:10px;border-left:2px solid var(--line)}.nav-ch.open>ul{display:block}
.nav ul a{font-size:12.8px;color:var(--muted);padding:3px 8px}.nav a.act{background:var(--soft);color:var(--accent-ink)}
.nav ul a.act{font-weight:600}.nav-ch.hide,.nav li.hide{display:none}.nav li.dim>a{opacity:.4}
.tw{position:absolute;left:0;top:5px;width:20px;height:24px;border:0;background:transparent;cursor:pointer;padding:0}
.tw::before{content:"";position:absolute;left:6px;top:8px;border:5px solid transparent;border-left:6px solid var(--muted);transition:transform .15s}
.nav-ch.open>.tw::before{transform:rotate(90deg) translate(2px,0)}
#scrim{display:none}
main{padding:28px clamp(16px,4vw,56px) 60px;max-width:1120px;width:100%}
/* úvod */
.hero{border-radius:18px;padding:34px 36px;color:#fff;position:relative;overflow:hidden;margin-bottom:26px;
background:linear-gradient(150deg,#0b1f4d 0%,#123a8a 48%,#1d63d8 100%);box-shadow:var(--shadow)}
.hero::before{content:"";position:absolute;inset:0;opacity:.12;background-image:linear-gradient(#fff 1px,transparent 1px),linear-gradient(90deg,#fff 1px,transparent 1px);background-size:34px 34px}
.hero-in{position:relative}.hero-brand{font-size:12px;letter-spacing:.34em;text-transform:uppercase;color:#93c5fd;font-weight:700}
.hero h1{font-size:clamp(26px,4vw,40px);line-height:1.1;margin:10px 0 8px;color:#fff;letter-spacing:-.01em}
.hero p{margin:0 0 16px;color:#dbeafe;max-width:62ch;font-size:16px}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{display:inline-block;padding:4px 12px;border-radius:99px;background:#38bdf8;color:#04243d;font-weight:700;font-size:12.5px}
.chip.soft{background:rgba(255,255,255,.14);color:#e0ecff;font-weight:500;border:1px solid rgba(255,255,255,.22)}.chip .key{color:#0f172a}
.cards-h{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:26px 0 12px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin-bottom:34px}
.card{display:flex;flex-direction:column;gap:3px;padding:14px 16px;border:1px solid var(--line);border-radius:var(--r);background:var(--panel);
text-decoration:none;color:inherit;transition:transform .12s,box-shadow .12s,border-color .12s}
.card:hover{transform:translateY(-2px);box-shadow:var(--shadow);border-color:var(--accent)}.card.hide{display:none}
.card-no{font-size:11px;font-weight:800;color:var(--accent);letter-spacing:.1em}.card-t{font-weight:700;color:var(--ink);font-size:15px}
.card-d{font-size:12.8px;color:var(--muted);line-height:1.45}
/* kapitoly */
.chapter{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:6px 34px 26px;margin:0 0 26px;box-shadow:var(--shadow)}
.chapter.hide,.sec.hide,.chap-intro.hide{display:none}
.chap-head{margin:0 -34px 18px;padding:26px 34px 16px;border-bottom:2px solid var(--ink);border-radius:16px 16px 0 0}
.chap-kicker{font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);font-weight:800}
.chapter h1{font-size:clamp(24px,3vw,32px);line-height:1.15;margin:4px 0 0;color:var(--ink);letter-spacing:-.01em}
.sec{border-top:1px solid var(--line);margin-top:6px}.sec:first-of-type{border-top:0}
.sec>summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:10px;padding:10px 6px;border-radius:10px;user-select:none}
.sec>summary::-webkit-details-marker{display:none}.sec>summary:hover{background:var(--soft2)}
.sec>summary::before{content:"";flex:none;width:0;height:0;border:6px solid transparent;border-left:8px solid var(--accent);transition:transform .15s;margin-left:2px}
.sec[open]>summary::before{transform:rotate(90deg)}
.sec h2{font-size:19px;margin:0;color:var(--ink);font-weight:700;line-height:1.3}.sec-body{padding:2px 4px 10px 26px}
h3{font-size:16px;color:var(--ink);margin:20px 0 6px;font-weight:650}
p{margin:0 0 10px}p.lead{font-size:17px;line-height:1.6;color:var(--muted);margin:12px 0 14px}
ul,ol{margin:0 0 12px;padding-left:24px}li{margin:3px 0}li::marker{color:var(--accent)}ol li::marker{font-weight:700}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}b,strong{color:var(--ink);font-weight:650}
.anchor{position:relative;top:calc(-1 * var(--bar-h) - 12px);visibility:hidden}
p.tip,p.warn{position:relative;border-radius:10px;padding:11px 14px 11px 46px;margin:12px 0 14px;border:1px solid}
p.tip{background:var(--tip-bg);border-color:var(--tip-bd);color:var(--tip)}p.warn{background:var(--warn-bg);border-color:var(--warn-bd);color:var(--warn)}
p.tip::before,p.warn::before{position:absolute;left:13px;top:11px;width:22px;height:22px;border-radius:50%;color:#fff;font-weight:800;font-size:13px;line-height:22px;text-align:center}
p.tip::before{content:"i";background:var(--tip-i)}p.warn::before{content:"!";background:var(--warn-i)}p.tip b,p.warn b,p.tip a,p.warn a{color:inherit}
.tablewrap{overflow-x:auto;margin:10px 0 16px;border:1px solid var(--line);border-radius:10px}
table.grid{width:100%;border-collapse:collapse;font-size:13.8px}
table.grid th{background:var(--th);color:var(--ink);text-align:left;font-weight:650;padding:8px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
table.grid td{padding:7px 12px;vertical-align:top;border-bottom:1px solid var(--line)}table.grid tr:last-child td{border-bottom:0}
table.grid tr:nth-child(even) td{background:var(--zebra)}table.grid tr:hover td{background:var(--soft)}
span.key{display:inline-block;font:600 12px/1.45 "Cascadia Mono",Consolas,monospace;color:var(--ink);background:linear-gradient(var(--panel),var(--code-bg));
border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:0 6px;white-space:nowrap}
code{font:12.8px "Cascadia Mono",Consolas,monospace;background:var(--code-bg);color:var(--code);border-radius:5px;padding:1px 5px}
pre{position:relative;font:12.8px/1.55 "Cascadia Mono",Consolas,monospace;background:var(--pre-bg);color:var(--pre);border-radius:10px;padding:14px 16px;overflow:auto;margin:10px 0 16px}
pre code{background:none;color:inherit;padding:0}.copy{position:absolute;right:8px;top:8px;padding:2px 9px;font-size:11.5px;border:1px solid rgba(255,255,255,.25);
background:rgba(255,255,255,.08);color:#cbd5e1;border-radius:6px;cursor:pointer;opacity:0;transition:opacity .15s}pre:hover .copy,.copy:focus{opacity:1}
img{max-width:100%;border-radius:8px;border:1px solid var(--line);cursor:zoom-in}img.zoom{position:fixed;inset:4vh 4vw;margin:auto;max-height:92vh;max-width:92vw;z-index:90;
box-shadow:0 0 0 100vmax rgba(2,6,23,.75);cursor:zoom-out;background:#fff}
hr{border:0;border-top:1px solid var(--line);margin:18px 0}
mark.hit{background:var(--hit);color:inherit;border-radius:3px;padding:0 1px}mark.hit.cur{background:var(--hit-cur);color:#fff}
.foot{color:var(--muted);font-size:12.5px;text-align:center;padding:10px 0}
.totop{position:fixed;right:18px;bottom:18px;width:42px;height:42px;border-radius:50%;background:var(--accent);color:#fff;box-shadow:var(--shadow);opacity:0;pointer-events:none;
transition:opacity .2s;font-size:18px}.totop.show{opacity:1;pointer-events:auto}.totop:hover{background:var(--accent-ink)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (max-width:1180px){.btn,.chk{display:none}.brand small{display:none}}
@media (max-width:980px){.layout{grid-template-columns:minmax(0,1fr)}.menu{display:inline-flex}
#side{position:fixed;left:0;top:var(--bar-h);width:min(86vw,340px);z-index:70;transform:translateX(-102%);transition:transform .2s;box-shadow:var(--shadow)}
body.nav-open #side{transform:none}body.nav-open #scrim{display:block;position:fixed;inset:var(--bar-h) 0 0 0;background:rgba(2,6,23,.45);z-index:65}
.chapter{padding:4px 16px 20px}.chap-head{margin:0 -16px 14px;padding:20px 16px 12px}.hero{padding:24px 20px}.sec-body{padding-left:6px}}
@media print{.bar,#side,#scrim,.totop,#progress,.copy,.cards,.cards-h{display:none!important}.layout{display:block}body{background:#fff;color:#000;font-size:11pt}
main{max-width:none;padding:0}.chapter{box-shadow:none;border:0;break-before:page;padding:0}.hero{break-after:page;box-shadow:none}
.sec>summary::before{display:none}.sec-body{padding-left:0}a{color:inherit}}
"""

# ------------------------------------------------------------------ behaviour
JS = r"""
(function(){
"use strict";
var UI_ALL=@UI_ALL@;
var $=function(s,r){return (r||document).querySelector(s)},$$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s))};
var root=document.documentElement,body=document.body;
var curLang=root.dataset.lang||"cs";
var UI=UI_ALL[curLang]||UI_ALL["cs"];
var hitsCount=0;

/* přepínání jazyků */
function setLang(lang,save){
 if(!UI_ALL[lang])lang="cs";
 curLang=lang;
 root.dataset.lang=lang;
 root.setAttribute("lang",lang);
 $$(".btn-lang").forEach(function(b){b.classList.toggle("active",b.dataset.lang===lang)});
 UI=UI_ALL[lang]||UI_ALL["cs"];
 var q=$("#q");
 if(q){q.placeholder=UI.search;q.setAttribute("aria-label",UI.search_aria)}
 var cnt=$("#count");
 if(cnt&&marks.length)cnt.textContent=UI.hits.replace("{n}",hitsCount);
 secs=null;state=null;lastWords="";
 if(q&&q.value.trim().length>=2)run();
 if(save!==false){try{localStorage.setItem("fea-help-lang",lang)}catch(e){}}
 var tit=document.querySelector("title");
 if(tit){
  var dt=lang==="cs"?root.dataset.titleCs:root.dataset.titleEn;
  if(dt)tit.textContent=dt;
 }
}
$$(".btn-lang").forEach(function(b){
 b.addEventListener("click",function(){setLang(b.dataset.lang,true)});
});
try{
 var qm=location.search.match(/[?&]lang=([a-z]{2})/);
 var savedLang=qm?qm[1]:localStorage.getItem("fea-help-lang");
 if(savedLang&&(savedLang==="cs"||savedLang==="en"))setLang(savedLang,false);
}catch(e){}

/* motiv */
$("#theme").addEventListener("click",function(){var t=root.dataset.theme==="dark"?"light":"dark";root.dataset.theme=t;try{localStorage.setItem("fea-help-theme",t)}catch(e){}});
/* oddíly: otevření předků cíle */
function openFor(el){for(var p=el;p;p=p.parentElement){if(p.tagName==="DETAILS")p.open=true}}
function go(id,smooth){var el=document.getElementById(id);if(!el)return false;openFor(el);
 var t=el.classList.contains("anchor")?el:el;t.scrollIntoView({behavior:smooth===false?"auto":"smooth",block:"start"});return true}
document.addEventListener("click",function(e){var a=e.target.closest&&e.target.closest("a[href^='#']");if(!a)return;
 var id=decodeURIComponent(a.getAttribute("href").slice(1));if(!id||id==="top"){if(id==="top"){e.preventDefault();window.scrollTo({top:0,behavior:"smooth"})}return}
 if(go(id)){e.preventDefault();history.replaceState(null,"","#"+id);body.classList.remove("nav-open")}});
if(location.hash){setTimeout(function(){go(decodeURIComponent(location.hash.slice(1)),false)},0)}
$("#expand").addEventListener("click",function(){$$("details.sec").forEach(function(d){d.open=true})});
$("#collapse").addEventListener("click",function(){$$("details.sec").forEach(function(d){d.open=false})});
/* tisk: rozbalit vše */
var printed=null;window.addEventListener("beforeprint",function(){printed=$$("details.sec").map(function(d){return d.open});$$("details.sec").forEach(function(d){d.open=true})});
window.addEventListener("afterprint",function(){if(printed)$$("details.sec").forEach(function(d,i){d.open=printed[i]})});
/* postranní panel */
$("#menu").addEventListener("click",function(){body.classList.toggle("nav-open")});$("#scrim").addEventListener("click",function(){body.classList.remove("nav-open")});
$$(".nav .tw").forEach(function(b){b.addEventListener("click",function(){b.parentElement.classList.toggle("open")})});
var spy=$$("[data-spy]").map(function(a){return {a:a,el:document.getElementById(a.dataset.spy)}}).filter(function(x){return x.el});
var cur=null;
function updateSpy(){var y=window.innerHeight*0.22+58,best=null,ch=null;
 for(var i=0;i<spy.length;i++){var x=spy[i],r=x.el.getBoundingClientRect();if(!r.height&&!r.width)continue;if(r.top<=y){best=x;if(x.a.classList.contains("chlink"))ch=x}}
 if(best===cur)return;cur=best;$$(".nav a.act").forEach(function(a){a.classList.remove("act")});
 if(best){best.a.classList.add("act");var li=best.a.closest(".nav-ch");if(li){li.classList.add("open");}
  var s=$("#side"),r2=best.a.getBoundingClientRect(),sr=s.getBoundingClientRect();if(r2.top<sr.top+40||r2.bottom>sr.bottom-40)s.scrollTop+=r2.top-sr.top-sr.height/3}}
var tick=false;
function onScroll(){if(tick)return;tick=true;requestAnimationFrame(function(){tick=false;updateSpy();
 var h=root.scrollHeight-innerHeight;$("#progress").style.width=(h>0?scrollY/h*100:0)+"%";$("#totop").classList.toggle("show",scrollY>700)})}
addEventListener("scroll",onScroll,{passive:true});addEventListener("resize",onScroll);onScroll();
$("#totop").addEventListener("click",function(){scrollTo({top:0,behavior:"smooth"})});
/* kód: kopírovat, obrázky: zvětšit */
$$("pre").forEach(function(p){var b=document.createElement("button");b.className="copy";b.type="button";b.textContent=UI.copy;
 b.addEventListener("click",function(){var t=p.innerText.replace(UI.copy,"").trim();var done=function(){b.textContent=UI.copied;setTimeout(function(){b.textContent=UI.copy},1200)};
  if(navigator.clipboard)navigator.clipboard.writeText(t).then(done,done);else done()});p.appendChild(b)});
$$("main img").forEach(function(i){i.addEventListener("click",function(){i.classList.toggle("zoom")})});
/* vyhledávání */
function fold(s){var o="";for(var i=0;i<s.length;i++){var c=s.charAt(i).normalize("NFD").charAt(0).toLowerCase().charAt(0);o+=c||" "}return o}
var secs=null,state=null,marks=[],mi=-1,q=$("#q"),box=$(".search"),res=$("#results"),sel=-1,timer=0,lastWords="";
function build(){
 if(secs)return;
 var lang=root.dataset.lang||"cs";
 secs=$$("[data-sec]").map(function(el){
  var h=el.querySelector("h2.lang-pane."+lang)||el.querySelector("h2");
  var ch=document.getElementById(el.dataset.ch);
  var chH1=ch?(ch.querySelector("h1 .lang-pane."+lang)||ch.querySelector("h1")):null;
  var chapTitle=chH1?chH1.textContent.trim():"";
  var pane=el.querySelector(".lang-pane."+lang)||el;
  var text=(pane.textContent||"").replace(/\s+/g," ").trim();
  return {el:el,id:el.id||el.dataset.ch,title:h?h.textContent.trim():chapTitle,chap:chapTitle,
   chEl:ch,text:text,f:fold(text),isSec:!!h};
 });
}
function words(v){return fold(v).split(/\s+/).filter(Boolean)}
function clearMarks(){marks.forEach(function(m){var p=m.parentNode;if(!p)return;p.replaceChild(document.createTextNode(m.textContent),m);p.normalize()});marks=[];mi=-1}
function markIn(el,ws){var re=new RegExp(ws.map(function(w){return w.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")}).join("|"),"g");
 var w=document.createTreeWalker(el,NodeFilter.SHOW_TEXT,{acceptNode:function(n){var p=n.parentNode;
  return (p.closest("mark,script,style,.copy"))?NodeFilter.FILTER_REJECT:NodeFilter.FILTER_ACCEPT}});var nodes=[],n;while((n=w.nextNode()))nodes.push(n);
 nodes.forEach(function(n){if(marks.length>2500)return;var f=fold(n.data),m,last=0,frag=null;re.lastIndex=0;
  while((m=re.exec(f))){if(!m[0].length){re.lastIndex++;continue}if(!frag)frag=document.createDocumentFragment();
   frag.appendChild(document.createTextNode(n.data.slice(last,m.index)));var mk=document.createElement("mark");mk.className="hit";mk.textContent=n.data.slice(m.index,m.index+m[0].length);
   frag.appendChild(mk);marks.push(mk);last=m.index+m[0].length}
  if(frag){frag.appendChild(document.createTextNode(n.data.slice(last)));n.parentNode.replaceChild(frag,n)}})}
function snippet(s,ws){var f=s.f,pos=-1;ws.forEach(function(w){var i=f.indexOf(w);if(i>=0&&(pos<0||i<pos))pos=i});if(pos<0)pos=0;
 var a=Math.max(0,pos-55),b=Math.min(s.text.length,pos+110),t=s.text.slice(a,b),ft=f.slice(a,b),out="",last=0;
 var re=new RegExp(ws.map(function(w){return w.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")}).join("|"),"g"),m;
 while((m=re.exec(ft))){if(!m[0].length){re.lastIndex++;continue}out+=esc(t.slice(last,m.index))+"<mark class=hit>"+esc(t.slice(m.index,m.index+m[0].length))+"</mark>";last=m.index+m[0].length}
 return (a>0?"… ":"")+out+esc(t.slice(last))+(b<s.text.length?" …":"")}
function esc(s){return s.replace(/[&<>]/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;"}[c]})}
function reset(){clearMarks();if(state){secs.forEach(function(s,i){s.el.classList.remove("hide");if(s.isSec&&state[i]!==undefined)s.el.open=state[i];s.chEl.classList.remove("hide")});state=null}
 $$(".nav .hide,.nav .dim,.card.hide").forEach(function(e){e.classList.remove("hide","dim")});res.hidden=true;res.innerHTML="";box.classList.remove("active");$("#count").textContent="";sel=-1;lastWords=""}
function run(){var v=q.value.trim();if(v.length<2){reset();return}build();var ws=words(v);if(ws.join("|")===lastWords)return;lastWords=ws.join("|");
 if(!state){state=secs.map(function(s){return s.isSec?s.el.open:undefined})}
 clearMarks();box.classList.add("active");var only=$("#onlyhits").checked,hits=[];
 secs.forEach(function(s,i){var sc=0,ok=true;ws.forEach(function(w){var n=0,p=-1;while((p=s.f.indexOf(w,p+1))>=0&&n<50)n++;if(!n)ok=false;sc+=n});
  s.hit=ok;s.score=ok?sc+(fold(s.title).indexOf(ws[0])>=0?25:0):0;if(ok)hits.push(s)});
 hitsCount=hits.length;
 var chHit={};hits.forEach(function(s){chHit[s.el.dataset.ch]=true});
 secs.forEach(function(s){var h=s.hit;s.el.classList.toggle("hide",only&&!h);if(s.isSec&&h)s.el.open=true;else if(s.isSec&&only)s.el.open=false;
  s.chEl.classList.toggle("hide",only&&!chHit[s.el.dataset.ch]);
  var a=s.isSec?$('.nav a[data-spy="'+s.id+'"]'):null;if(a)a.parentElement.classList.toggle("dim",!h)});
 $$(".nav-ch").forEach(function(li){li.classList.toggle("hide",!chHit[li.dataset.ch]);if(chHit[li.dataset.ch])li.classList.add("open")});
 $$(".card").forEach(function(c){c.classList.toggle("hide",!chHit[c.dataset.card])});
 var lang=root.dataset.lang||"cs";
 hits.forEach(function(s){var pane=s.el.querySelector(".lang-pane."+lang)||s.el;markIn(pane,ws)});
 $("#count").textContent=marks.length?(UI.hits.replace("{n}",hits.length)):"";
 hits.sort(function(a,b){return b.score-a.score});showResults(hits.slice(0,40),ws);if(marks.length){mi=-1;cycle(1,true)}}
function showResults(list,ws){sel=-1;if(!list.length){res.innerHTML='<div class="res none">'+UI.no_hits+'</div>';res.hidden=false;return}
 res.innerHTML=list.map(function(s,i){return '<a class="res" role="option" data-id="'+(s.id)+'" data-i="'+i+'"><div class="rc">'+esc(s.chap)+'</div><div class="rt">'+esc(s.isSec?s.title:s.chap)+
  '</div><div class="rs">'+snippet(s,ws)+'</div></a>'}).join("");res.hidden=false}
function cycle(d,quiet){if(!marks.length)return;if(mi>=0)marks[mi].classList.remove("cur");mi=(mi+d+marks.length)%marks.length;var m=marks[mi];m.classList.add("cur");openFor(m);
 if(!quiet||true)m.scrollIntoView({block:"center",behavior:"auto"})}
q.addEventListener("input",function(){clearTimeout(timer);timer=setTimeout(run,70)});
q.addEventListener("keydown",function(e){var items=$$(".res[data-id]",res);
 if(e.key==="Escape"){if(q.value){q.value="";reset()}else q.blur();return}
 if(e.key==="ArrowDown"&&items.length){e.preventDefault();sel=Math.min(sel+1,items.length-1)}else if(e.key==="ArrowUp"&&items.length){e.preventDefault();sel=Math.max(sel-1,0)}
 else if(e.key==="Enter"){e.preventDefault();if(sel>=0&&items[sel]){pick(items[sel])}else{res.hidden=true;cycle(e.shiftKey?-1:1)}return}else return;
 items.forEach(function(it,i){it.classList.toggle("sel",i===sel)});if(items[sel])items[sel].scrollIntoView({block:"nearest"})});
function pick(it){res.hidden=true;var el=document.getElementById(it.dataset.id);if(!el)return;openFor(el);el.scrollIntoView({block:"start"});
 var i=marks.findIndex(function(m){return el.contains(m)});if(i>=0){if(mi>=0)marks[mi].classList.remove("cur");mi=i;marks[i].classList.add("cur");marks[i].scrollIntoView({block:"center"})}}
res.addEventListener("click",function(e){var it=e.target.closest(".res[data-id]");if(it)pick(it)});
q.addEventListener("focus",function(){if(q.value.trim().length>=2&&res.innerHTML)res.hidden=false});
document.addEventListener("click",function(e){if(!e.target.closest(".search"))res.hidden=true});
$("#onlyhits").addEventListener("change",function(){lastWords="";run()});
$("#next").addEventListener("click",function(){cycle(1)});$("#prev").addEventListener("click",function(){cycle(-1)});
$("#clear").addEventListener("click",function(){q.value="";reset();q.focus()});
document.addEventListener("keydown",function(e){var t=e.target.tagName;
 if((e.key==="/"&&t!=="INPUT"&&t!=="TEXTAREA")||((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="k")){e.preventDefault();q.focus();q.select()}
 else if(e.key==="F3"&&marks.length){e.preventDefault();cycle(e.shiftKey?-1:1)}});
})();
"""

HELP_DIR = Path(__file__).resolve().parent / "help"


def export_html(path: str | Path, lang: str = "cs", only: list[str] | None = None,
                title: str | None = None, subtitle: str | None = None, version: str = "1.3.0",
                bilingual: bool = True) -> Path:
    """Builds and saves the standalone HTML manual (including images and interactive scripts).
    If bilingual=True, it creates the merged bilingual manual with the CZ / EN switch.
    """
    from datetime import date
    from .help import PAGES
    from .help_pdf import TEXTS
    today = date.today()
    edition_cs = f"{today.day}. {today.month}. {today.year}"
    edition_en = f"{today:%B} {today.day}, {today.year}"
    is_testing = only is not None and set(only) == {"testovani", "ukazky"}

    t_cs = "FEA Studio – Návod k testování" if is_testing else "FEA Studio – Uživatelská příručka"
    t_en = "FEA Studio – Testing Guide" if is_testing else "FEA Studio – User Manual"
    sub_cs = "Úlohy se známým výsledkem a očekávané hodnoty" if is_testing else "Analýza metodou konečných prvků pro konstruktéry"
    sub_en = "Benchmark problems with verified results" if is_testing else "Finite Element Analysis for Design Engineers"

    title_cs = title if (title and lang == "cs") else t_cs
    title_en = title if (title and lang == "en") else t_en
    subtitle_cs = subtitle if (subtitle and lang == "cs") else sub_cs
    subtitle_en = subtitle if (subtitle and lang == "en") else sub_en

    chapters_cs = []
    chapters_en = []

    from .help_pdf import pages
    items_cs = pages("cs", only)
    dict_en = dict(pages("en", only))

    for slug, ptitle in items_cs:
        p_cs = HELP_DIR / "cs" / f"{slug}.html"
        p_en = HELP_DIR / f"{slug}.html"
        if not p_cs.exists() and not p_en.exists():
            continue
        ptitle_en = dict_en.get(slug, ptitle)
        if p_cs.exists():
            chapters_cs.append(prepare_chapter(slug, ptitle, p_cs.read_text(encoding="utf-8"), [HELP_DIR / "cs"]))
        if p_en.exists():
            chapters_en.append(prepare_chapter(slug, ptitle_en, p_en.read_text(encoding="utf-8"), [HELP_DIR, HELP_DIR / "cs"]))

    if bilingual and chapters_cs and chapters_en and len(chapters_cs) == len(chapters_en):
        site_html = build_bilingual_site(
            chapters_cs, chapters_en,
            title_cs=title_cs, title_en=title_en,
            subtitle_cs=subtitle_cs, subtitle_en=subtitle_en,
            version=version,
            edition_cs=edition_cs, edition_en=edition_en,
            default_lang=lang
        )
    else:
        chapters = chapters_cs if lang == "cs" else chapters_en
        t = title_cs if lang == "cs" else title_en
        st = subtitle_cs if lang == "cs" else subtitle_en
        ed = edition_cs if lang == "cs" else edition_en
        site_html = build_site(chapters, title=t, subtitle=st, version=version, edition=ed, lang=lang)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(site_html, encoding="utf-8")
    return path
