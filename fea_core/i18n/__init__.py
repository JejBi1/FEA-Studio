"""Translation of the texts of the program (without Qt).

The source language of the code is English: the literals are written in English. The Czech UI is produced by the catalogue
`cs.json` (English → Czech), in the same format as before:
- `exact`: the whole text → translation,
- `templates`: text with variable parts (f-strings, .format, %) → a template with {0}, {1}… in both languages; the variable
  parts are translated again (e.g. the name of a step inside a sentence).

`tr(text)` returns the text in the current language, an unknown text unchanged (safe for the names of parts, numbers etc.).
Language: `set_language("cs")`, the default from the environment variable FEA_LANG, otherwise English (the GUI chooses it in the
settings or by the system language, `fea_gui.i18n`; the Word report according to `ReportOptions.language`). In English `tr`
returns the text unchanged.

Data that are Czech by nature (material and grade names, standards, vocabularies for recognising names from Czech CAD models) stay Czech
in the code; they are not part of the catalogue.
"""
from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path

LANGS = {"cs": "Čeština", "en": "English"}
_DIR = Path(__file__).resolve().parent
CATALOG_CS = _DIR / "cs.json"          # English (source) → Czech

_lang = os.environ.get("FEA_LANG", "en") if os.environ.get("FEA_LANG", "en") in LANGS else "en"
_local = threading.local()               # language of the thread (the report in another language than the GUI) – see using()
_keep: frozenset[str] = frozenset()       # names that are not translated (parts from the CAD) – see keep_names()

_LETTER = re.compile(r"[^\W\d_]")
_CZ = re.compile(r"[ěščřžýáíéóůúňťďĚŠČŘŽÝÁÍÉÓŮÚŇŤĎ]")     # remaining Czech in a translation into English (choice of the best variant)
_PREFIX = re.compile(r"((?:[•▸▾▴◂▹►◄▼▲▶◀⚠!√×*✓✔✗✘→←↑↓★·–\-]+|\d{1,2}\.)\s+)(\S.*)", re.S)   # „▸ Rada…“, „6. Pruty: …“
_BRACKET = re.compile(r"(.*\S)(\s+)\[([^\[\]\n]{1,40})\]([.:]?)", re.S)          # examples: "Otevřít  [Ctrl+O]", "Deska [Ocel]"
_PAREN = re.compile(r"(.*\S)(\s+)\(([^()\n]{1,120})\)([.:,;]?)", re.S)         # example: "3D tisk standard  (vestavěná)"
_HTML = re.compile(r"(<[^<>]*>|&[A-Za-z#0-9]+;)")
_HOLE = re.compile(r"\{(\d+)\}")
_DEC = re.compile(r"(?<=\d),(?=\d)")
_STAMP = re.compile(r"(\[[\d:.]+\]\s+)(.+)", re.S)
_BR = re.compile(r"(<br\s*/?>)")
_TAGS = re.compile(r"((?:<[^<>]+>)+)([^<>]*)((?:<[^<>]+>)*)", re.S)
_UNIT = re.compile(r"[−\-+]?(?:(?!\d)[\s  .,·⁰¹²³⁴⁵⁶⁷⁸⁹⁻×e])*\d[\d\s  .,·⁰¹²³⁴⁵⁶⁷⁸⁹⁻×e]*"
                   r"(\s?[A-Za-zµ°%‰Ω/·²³⁻¹]{1,8})?")        # the part before the first digit cannot contain a digit → no overlapping stars (ReDoS)
_LINES = re.compile(r"(\n|<br\s*/?>)")
_LONG_MAX = 2000                         # a longer text is only split (lines, <br>, tags); the regex heuristics run on short pieces only
_TEMPLATE_MAX = {0: 2000, 1: 2000, 2: 800, 3: 250}   # max. text length for matching a template by the number of its holes (cost ~ n^holes)
_UNIT_TAIL = re.compile(r"(.+?)( \[[^\[\]\s]{1,12}\])")   # name of a quantity + " [unit]" (the unit without spaces)
_EDGE = re.compile(r"^(?:\s|<br\s*/?>)+|(?:\s|<br\s*/?>)+$")      # edges of the text of the catalogue (spaces, line breaks, <br>)
_LOOSE_MAX = 600                         # templates starting with a variable are searched only in texts up to this length


def language() -> str:
    return getattr(_local, "lang", None) or _lang


def set_language(lang: str) -> None:
    """Switches the language of the texts of the program (cs | en). An unknown language = English."""
    global _lang
    _lang = lang if lang in LANGS else "en"


LANGUAGE_FILE = "language.txt"          # in the user settings: written by the installer (choice at installation) and by the GUI switch


def language_path() -> Path:
    from ..config import user_config_dir
    return user_config_dir() / LANGUAGE_FILE


def stored_language() -> str | None:
    """Program language chosen at installation or by the switch in the GUI (the last choice wins); None = not chosen."""
    try:
        lang = language_path().read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    return lang if lang in LANGS else None


def store_language(lang: str) -> None:
    if lang not in LANGS:
        return
    p = language_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(lang + "\n", encoding="utf-8")
    except OSError:
        pass


def english() -> bool:
    return language() == "en"


@contextmanager
def using(lang: str | None):
    """Language only for the current thread (e.g. the Word report in another language than the GUI); None = no change."""
    old = getattr(_local, "lang", None)
    _local.lang = lang if lang in LANGS else old
    try:
        yield
    finally:
        _local.lang = old


_TAG_HEAD = re.compile(r"(?:<[^<>]*>)+")


def _edge_tags(text: str) -> tuple[str, str, str]:
    """HTML tags at the start and at the end of a text, and the text between them (linear, no backtracking)."""
    m = _TAG_HEAD.match(text)
    head = m.group(0) if m else ""
    rest = text[len(head):]
    j = len(rest)
    while j > 0 and rest[j - 1] == ">":
        k = rest.rfind("<", 0, j)
        if k < 0 or "<" in rest[k + 1:j - 1] or ">" in rest[k:j - 1]:
            break
        j = k
    return head, rest[:j], rest[j:]


class _LazyRx:
    """A template regex compiled at its first use; an invalid pattern never matches."""
    __slots__ = ("src", "rx")

    def __init__(self, src: str):
        self.src, self.rx = src, None

    def match(self, text: str):
        if self.rx is None:
            try:
                self.rx = re.compile(self.src, re.S)
            except re.error:
                self.rx = False
        return self.rx.match(text) if self.rx else None


class _Engine:
    """Catalogue of one direction. `english` = the target is English (catalogue Czech → English, the legacy one) or Czech
    (catalogue English → Czech). The decomposition heuristics (templates, lines, HTML, brackets, composed texts) are shared."""

    def __init__(self, english: bool, path: Path | None):
        self.english = english
        self.path = path
        self.loaded = False
        self.exact: dict[str, str] = {}
        self.buckets: dict[str, list] = {}   # first 3 characters of the fixed start of the template → [(regex, template, order, length, fixed)]
        self.loose: list = []                # templates starting with a variable
        self.loose_by: dict[str, list] = {}  # fixed part → templates (searched by the occurrence of the fixed part in the text)
        self.loose_rx = None
        self.loose_in: dict[str, list] = {}  # key → shorter keys contained in it
        self.cache: dict[str, str] = {}
        self.sub: dict[tuple[str, int], str] = {}
        self.persisted: dict[str, str] | None = None     # translations of the start-up texts from the last start (persist_startup)

    # ---- loading
    def load(self) -> None:
        if self.loaded:
            return
        data = {}
        if self.path is not None:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
        self.use(data)
        if self.persisted:                       # use() empties the cache
            self.cache.update(self.persisted)

    # ---- translations of the start-up texts kept between the starts of the GUI (the catalogue is then not even loaded at the start)
    def persist_file(self) -> Path | None:
        """File of the persisted translations; the key = content of the catalogue + code of the engine (any change = a new file)."""
        if self.path is None or self.english or os.environ.get("FEA_I18N_CACHE") == "0":
            return None
        try:
            import hashlib
            from ..config import user_data_dir
            h = hashlib.sha1(self.path.read_bytes())
            h.update(Path(__file__).read_bytes())
        except (OSError, ImportError):
            return None
        return user_data_dir() / "cache" / f"i18n-cs-{h.hexdigest()[:16]}.json"

    def read_persisted(self) -> None:
        self.persisted = {}
        f = self.persist_file()
        if f is None:
            return
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            self.persisted = {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
            self.cache.update(self.persisted)

    def write_persisted(self, entries: dict[str, str]) -> None:
        f = self.persist_file()
        if f is None or not entries or (self.persisted and len(entries) <= len(self.persisted)):
            return
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            for old in f.parent.glob("i18n-cs-*.json"):          # files of older catalogues / versions
                if old != f:
                    old.unlink(missing_ok=True)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(entries, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            tmp.replace(f)
        except OSError:
            pass

    def use(self, data: dict) -> None:
        """Sets the catalogue {"exact": {…}, "templates": {…}}."""
        self.cache.clear()
        self.sub.clear()
        exact = {k: v for k, v in data.get("exact", {}).items() if v}
        templates = {k: v for k, v in data.get("templates", {}).items() if v}
        # texts with edge spaces / line breaks ("  (vestavěná)", "\n\nDvojklik: krok {0}") and without them – in the code they
        # are appended to other text and after splitting into lines / parts they are searched already trimmed
        for d in (exact, templates):
            for k, v in list(d.items()):
                ks = _EDGE.sub("", k)
                if ks != k and _LETTER.search(ks) and ks not in d:
                    d[ks] = _EDGE.sub("", v)
        self.buckets.clear()
        self.loose.clear()
        for cs, en in templates.items():
            parts = _HOLE.split(cs)          # fixed part, number, fixed part, …
            parts[0::2] = [p.replace("{{", "{").replace("}}", "}") for p in parts[0::2]]
            fixed = parts[0::2]
            rx = "".join(re.escape(p) if i % 2 == 0 else "(.*?)" for i, p in enumerate(parts))   # even an empty part
            pat = _LazyRx(rx + r"\Z")            # compiled at the first use (~2900 templates: 2.5 s at the start of the GUI)
            order = [int(n) for n in parts[1::2]]
            size = sum(len(p) for p in fixed)
            item = (pat, en, order, size, tuple(f for f in fixed if f))
            if len(fixed[0]) >= 3:
                self.buckets.setdefault(fixed[0][:3], []).append(item)
            else:
                self.loose.append((max(fixed, key=len), *item))
        for b in self.buckets.values():
            b.sort(key=lambda x: -x[3])
        self.loose.sort(key=lambda x: -x[4])
        self.loose_by.clear()
        for item in self.loose:
            self.loose_by.setdefault(item[0], []).append(item[1:])
        keys = sorted(self.loose_by, key=len, reverse=True)          # on each position the longest key wins
        self.loose_rx = re.compile("|".join(map(re.escape, keys))) if keys else None
        self.loose_in.clear()
        for k in keys:                                            # shorter keys contained in the found one (also at the same position)
            self.loose_in[k] = [j for j in keys if j != k and j in k]
        self.exact = exact
        self.loaded = True

    # ---- translation
    def tr(self, text: str) -> str:
        hit = self.cache.get(text)
        if hit is not None:
            return hit
        if self.persisted is None and not self.loaded:   # the first text: the persisted start-up translations
            self.read_persisted()
            hit = self.cache.get(text)
            if hit is not None:
                return hit
        self.load()
        out = self.translate(text, 0)
        if len(self.cache) > 50000:
            self.cache.clear()
        self.cache[text] = out
        return out

    def translate(self, text: str, depth: int) -> str:
        if not _LETTER.search(text):
            return _DEC.sub(".", text) if self.english else text
        if text in _keep:
            return text
        hit = self.exact.get(text)
        if hit is not None:
            return hit
        key = (text, depth)
        hit = self.sub.get(key)
        if hit is None:
            hit = self.best(text, depth)
            if len(self.sub) > 200000:
                self.sub.clear()
            self.sub[key] = hit
        return hit

    def best(self, text: str, depth: int) -> str:
        """Translation of a text that is not completely in the catalogue: quick exact adjustments, otherwise the best of the variants of the
        decomposition (templates, lines, HTML, prefixes, brackets, composed text) – the first variant without a Czech remainder (English
        target), otherwise the one with the smallest remainder (on equality the earlier one). Czech target: the first variant that changed the text.
        """
        core = text.strip()
        if core != text:
            if not core:
                return text
            i = text.index(core[0])
            return text[:i] + self.translate(core, depth) + text[i + len(core):]
        if len(text) > _LONG_MAX:                     # no regex heuristics on a long (e.g. foreign) text – ReDoS
            return self.split_long(text, depth)
        if _UNIT.fullmatch(text):                       # „1,5 mm“
            if self.english:
                return _DEC.sub(".", text)
            for cand in self.templates(text, depth):    # "1.3 Frame" (a numbered heading) looks like a quantity with a unit
                if cand != text:
                    return cand
            return text
        m = _UNIT_TAIL.fullmatch(text)                  # "Kontakt – rozevření [mm]" with the catalogue entry "Kontakt – rozevření"
        if m and m.group(1) in self.exact and m.group(1) not in _keep:
            return self.exact[m.group(1)] + m.group(2)
        for tail in (":", " …", "…", ".", " *"):        # "Síla:" with the catalogue entry "Síla"
            if text.endswith(tail) and text[:-len(tail)] in self.exact and text[:-len(tail)] not in _keep:
                return self.exact[text[:-len(tail)]] + tail
        head, mid, tail = _edge_tags(text)              # "</table>Equilibrium check: …": a template of the whole text first,
        if (head or tail) and _LETTER.search(mid) and _LINES.search(text) is None:   # otherwise the text between the end tags
            for cand in (self.templates(text, depth, loose=False) if depth < 4 else ()):
                if cand and cand != text:
                    return cand
            if _HTML.search(mid) is None:
                return head + self.translate(mid, depth) + tail
            return "".join(q if j % 2 or not _LETTER.search(q) else self.translate(q, depth)   # per text between the tags
                           for j, q in enumerate(_HTML.split(text)))
        best, best_bad = None, 0
        for cand in self.variants(text, depth):
            if not cand or cand == text:
                continue
            if not self.english:
                return cand
            bad = len(_CZ.findall(cand))
            if bad == 0:
                return cand
            if best is None or bad < best_bad:
                best, best_bad = cand, bad
        return text if best is None else best

    def split_long(self, text: str, depth: int) -> str:
        """Long text: only per line / `<br>` and per text between tags; a piece that is still longer than _LONG_MAX stays unchanged."""
        out = []
        for i, line in enumerate(_LINES.split(text)):
            if i % 2 or not line:
                out.append(line)
            elif len(line) <= _LONG_MAX:
                out.append(self.translate(line, depth))
            else:
                out.append("".join(p if j % 2 or len(p) > _LONG_MAX or not _LETTER.search(p) else self.translate(p, depth)
                                   for j, p in enumerate(_HTML.split(line))))
        return "".join(out)

    def variants(self, text: str, depth: int):
        """Possible translations by decomposition of the text (generator – further variants are computed only when the previous ones are not enough)."""
        if depth < 4:
            yield from self.templates(text, depth)
        for sep in (": ", " – "):                       # "Otvor sloupku: Q = 15 W" – a known name before the description
            head, _, tail = text.partition(sep)
            if tail and head in self.exact and head not in _keep:
                yield self.exact[head] + sep + self.translate(tail, depth + 1)
        m = _PREFIX.fullmatch(text)                     # „▸ Rada…“, „⚠ …“, „• …“
        if m:
            yield m.group(1) + self.translate(m.group(2), depth)
        if "\n" in text:
            yield "\n".join(self.translate(x, depth) if x else x for x in text.split("\n"))
        m = _STAMP.match(text)                         # log line "[18:14:27] Hotovo"
        if m:
            yield m.group(1) + self.translate(m.group(2), depth)
        if "<br" in text:                               # HTML line by line (summaries of the results)
            parts = _BR.split(text)
            yield "".join(p if i % 2 else self.translate(p, depth) if p else p for i, p in enumerate(parts))
        for rx, a, b in ((_BRACKET, "[", "]"), (_PAREN, "(", ")")):   # "…  [Ctrl+O]", "3D tisk standard  (vestavěná)"
            m = rx.fullmatch(text)
            if m:
                inner = a + m.group(3) + b
                inner = self.exact.get(inner) or a + self.translate(m.group(3), depth + 1) + b
                yield self.translate(m.group(1), depth) + m.group(2) + inner + m.group(4)
        m = _TAGS.fullmatch(text)                       # "<b>Síla</b>"
        if m and m.group(2):
            yield m.group(1) + self.translate(m.group(2), depth) + m.group(3)
        if "<" in text and ">" in text:                 # HTML (tables): each text between the tags separately
            parts = _HTML.split(text)
            if len(parts) > 1:
                yield "".join(p if i % 2 or not _LETTER.search(p) else self.translate(p, depth) for i, p in enumerate(parts))
                return                                  # composed text only in the texts between the tags (not across the tags)
        if depth < 2:                                   # composed text: "radiálně, tečně, osově", "VDI 2230: Tlak …"
            for sep in (" – ", "; ", ": ", ", ", " | ", " · "):
                if sep in text:
                    yield sep.join(self.translate(x, depth + 1) for x in text.split(sep))

    def templates(self, text: str, depth: int, loose: bool = True):
        """Templates that match the text, from the longest fixed part ("{0}: Q = {1} W" before "Otvor {0}"); loose=False = only templates
        starting with a fixed text."""
        cands = list(self.buckets.get(text[:3], ()))
        if loose and self.loose_rx is not None and len(text) <= _LOOSE_MAX:   # a long text is translated in parts (lines, cells)
            hits: set[str] = set()
            m = self.loose_rx.search(text)
            while m is not None:                        # from each found position + 1 → also the overlapping keys
                k = m.group(0)
                if k not in hits:
                    hits.add(k)
                    hits.update(self.loose_in[k])
                m = self.loose_rx.search(text, m.start() + 1)
            multi = _LINES.search(text) is not None     # a template starting with a hole would swallow other lines of the text
            cands += [it for k in hits for it in self.loose_by[k]
                      if not multi or any(_LINES.search(f) for f in it[4])]
        n = len(text)
        cands = [c for c in cands if n <= _TEMPLATE_MAX[min(len(c[2]), 3)]          # cost of the lazy holes grows as n^holes
                 and all(f in text for f in c[4])]      # without all the fixed parts the regex does not match
        cands.sort(key=lambda x: -x[3])
        for item in cands:
            r = self.fill(item, text, depth)
            if r is not None:
                yield r

    def fill(self, item, text: str, depth: int) -> str | None:
        pat, en, order = item[:3]
        m = pat.match(text)
        if m is None:
            return None
        vals = {}
        for n, g in zip(order, m.groups()):
            vals[n] = self.translate(g, depth + 1) if g else g
        try:
            out = _HOLE.sub(lambda h: vals.get(int(h.group(1)), h.group(0)), en)
            return out.replace("{{", "{").replace("}}", "}") if "{{" in en or "}}" in en else out
        except (ValueError, KeyError):
            return None


# The only catalogue: English (source) → Czech.
_ENGINES = {"cs": _Engine(False, CATALOG_CS)}


def use_catalog(data: dict, lang: str = "cs") -> None:
    """Sets the catalogue {"exact": {…}, "templates": {…}} of the language (tests; otherwise the file is loaded)."""
    _ENGINES[lang].use(data)


def reload() -> None:
    """Loads the catalogues again (after editing the files)."""
    for e in _ENGINES.values():
        e.loaded = False
        e.cache.clear()
        e.sub.clear()


def keep_names(names) -> None:
    """Names that are never translated – neither alone nor as a part of a composed text (parts from the CAD: in the Czech and the English
    GUI they must match the model and the CAD, even if they coincide with a text of the catalogue, e.g. "Deska" from the examples).
    """
    global _keep
    new = frozenset(n for n in names if isinstance(n, str) and n)
    if new != _keep:
        _keep = new
        for e in _ENGINES.values():
            e.cache.clear()
            e.sub.clear()


def number(text: str) -> str:
    """Decimal comma → point in the English text (only between digits)."""
    return _DEC.sub(".", text) if language() == "en" else text


def decimal(text: str) -> str:
    """A number written with the decimal point (f"{v:g}") in the language of the program: in Czech a comma, in English a point."""
    return text if language() == "en" else text.replace(".", ",")


def tr(text):
    """Text in the current language; an unknown text (and a non-string) unchanged."""
    lang = language()
    if not isinstance(text, str) or not text:
        return text
    if lang == "cs":
        return _ENGINES["cs"].tr(text)
    return text


def startup_snapshot() -> dict[str, str]:
    """Translations made so far (call right after the main window is built – before a study is opened: no names from files)."""
    eng = _ENGINES.get(language())
    if eng is None:
        return {}
    return {k: v for k, v in eng.cache.items() if len(k) <= 2000}


def persist_startup(snapshot: dict[str, str]) -> None:
    """Saves the translations of the start-up texts for the next start (user data / cache; only when there are new ones)."""
    eng = _ENGINES.get(language())
    if eng is not None and len(snapshot) <= 50000:
        eng.write_persisted(snapshot)


def missing(texts) -> list[str]:
    """Texts that are not translated in the current language – for tests and for completing the catalogue."""
    out = []
    for t in texts:
        if _LETTER.search(t) and tr(t) == t:
            out.append(t)
    return out


# ------------------------------------------------------------------ matplotlib
def _patch_matplotlib(mod) -> None:
    """Labels of charts (axes, titles, legends) – Text.set_text translates them; the constructor of Text calls it too."""
    t0 = mod.Text.set_text
    if getattr(t0, "_i18n", False):
        return

    def set_text(self, s):
        return t0(self, tr(s) if isinstance(s, str) else s)
    set_text._i18n = True
    mod.Text.set_text = set_text


def patch_matplotlib() -> None:
    """The labels of matplotlib charts are translated by tr() (GUI, report). matplotlib is loaded lazily – if it is already
    loaded, it is wrapped at once, otherwise after the import (a hook in sys.meta_path).
    """
    import importlib.abc
    import importlib.machinery
    import sys
    if "matplotlib.text" in sys.modules:
        _patch_matplotlib(sys.modules["matplotlib.text"])
        return

    global _Finder

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "matplotlib.text":
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            ex0 = spec.loader.exec_module

            def exec_module(module):
                ex0(module)
                _patch_matplotlib(module)
            spec.loader.exec_module = exec_module
            return spec
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
