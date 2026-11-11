"""Collection of the English texts from the code for the Czech catalogue cs.json (a development tool).

    python -m fea_core.i18n.extract                 → overview: how many texts are in the catalogue and how many are missing
    python -m fea_core.i18n.extract --todo out.json → missing texts to translate into Czech ({"exact": {english: ""}, "templates": …});
                                                      with --where the value is the place in the code
    python -m fea_core.i18n.extract --merge in.json → adds the Czech translations to cs.json (an empty value = not a text for the user)
    python -m fea_core.i18n.extract --prune         → removes from cs.json the texts that are no longer in the code

The texts are taken from the AST: string literals (and always those passed directly into `tr("…")`), f-strings, "…".format(), "…" % (…) and sums of strings. The variable parts
are replaced by {0}, {1}… (a template). Documentation strings, identifiers, paths and regular expressions are left out.
In addition the texts of the data files that the GUI shows (DATA_TEXTS – the library of friction).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path(__file__).resolve().parent / "cs.json"

# Where the texts that the user sees come from (GUI, messages of the core, the report). MCP/REST/CLI speak to agents and in the console
# – they stay in English, except the modules whose texts the GUI or the report shows.
SOURCES = ["fea_gui", "fea_core", "examples"]          # examples: the names in the example studies
SKIP = ["fea_core/i18n", "fea_core/rest", "fea_core/cli.py", "fea_core/warmup.py", "fea_core/__main__.py",
        "fea_core/mcp/prompts.py", "fea_core/mcp/server.py", "fea_core/mcp/setup.py", "fea_core/mcp/README",
        "fea_core/fatigue/pragtic_fetch.py", "fea_gui/help_export.py", "fea_gui/help_pdf.py", "fea_gui/help_web.py",
        "examples/help_images.py"]
SKIP_MCP_TOOLS = True          # fea_core/mcp/tools_*.py only the functions that the GUI calls (below)
MCP_KEEP = {"tools_mesh.py"}   # rigid_body_check – messages in the step Solve

_DIAC = re.compile(r"[áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]")
_LETTER = re.compile(r"[^\W\d_]")
_IDENT = re.compile(r"^[A-Za-z_][\w.\-:/\\]*$")
_PCT = re.compile(r"%(?:\([^)]*\))?[-+ #0]*(?:\d+|\*)?(?:\.\d+)?[sdifgeExXrcu%]")
_FMT = re.compile(r"\{([^{}]*)\}")
_CARD = re.compile(r"^(\*\*|[A-Z][A-Z0-9_]*\s*(,|=\s)|[A-Z0-9]+\{\d+\})")
_QSS = re.compile(r"[{]{1,2}\s*[a-z-]+\s*:[^{}]*[}]{1,2}")
_CZ_WORDS =re.compile(r"\b(a|v|ve|na|do|od|se|je|jsou|pro|bez|nebo|jen|kde|po|při|za|k|s|z|u|o|už|ani|než|"
                       r"plocha|síla|vazba|díl|stav|tlak|síť|krok|zatížení|výpočet|soubor|teplota)\b", re.I)


def _files():
    for src in SOURCES:
        for f in sorted((ROOT / src).rglob("*.py")):
            rel = f.relative_to(ROOT).as_posix()
            if any(rel.startswith(s) for s in SKIP) or "__pycache__" in rel:
                continue
            if rel.startswith("fea_core/mcp/tools_") and SKIP_MCP_TOOLS and f.name not in MCP_KEEP:
                continue
            yield f, rel


def _esc(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


class _Collector(ast.NodeVisitor):
    def __init__(self):
        self.out: list[tuple[str, bool, int]] = []      # (text, template?, line)
        self.skip: set[int] = set()
        self.forced: set[str] = set()                   # literals passed directly into tr(…) – the filter _ui_text is not used

    # -- parts of a template: [("t", text) | ("h", None)]
    def _pieces(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [("t", node.value)]
        if isinstance(node, ast.JoinedStr):
            res = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    res.append(("t", v.value))
                else:
                    res.append(("h", None))
            return res
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._pieces(node.left), self._pieces(node.right)
            if left is None and right is None:
                return None
            return (left or [("h", None)]) + (right or [("h", None)])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant) \
                and isinstance(node.left.value, str):
            res, pos = [], 0
            s = node.left.value
            for m in _PCT.finditer(s):
                res.append(("t", s[pos:m.start()]))
                res.append(("t", "%") if m.group(0) == "%%" else ("h", None))
                pos = m.end()
            res.append(("t", s[pos:]))
            return res
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format" \
                and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
            res, pos = [], 0
            s = node.func.value.value.replace("{{", "\x01").replace("}}", "\x02")
            for m in _FMT.finditer(s):
                res.append(("t", s[pos:m.start()]))
                res.append(("h", None))
                pos = m.end()
            res.append(("t", s[pos:]))
            return [(k, v.replace("\x01", "{").replace("\x02", "}") if v else v) for k, v in res]
        return None

    def _emit(self, node, pieces):
        text, n, holes = "", 0, False
        for kind, val in pieces:
            if kind == "t":
                text += _esc(val)
            else:
                text += "{%d}" % n
                n += 1
                holes = True
        if not holes:
            text = text.replace("{{", "{").replace("}}", "}")
        self.out.append((text, holes, getattr(node, "lineno", 0)))

    def _mark(self, node):
        for sub in ast.walk(node):
            self.skip.add(id(sub))

    def _direct(self, node) -> list:
        """Nodes that form the fixed parts of a template (literals, f-strings and the items of a sum)."""
        if isinstance(node, ast.Constant):
            return [node]
        if isinstance(node, ast.JoinedStr):
            return [node] + [v for v in node.values if isinstance(v, ast.Constant)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            out = [node]
            for side in (node.left, node.right):
                if self._pieces(side) is not None:
                    out += self._direct(side)
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return [node, node.left]
        if isinstance(node, ast.Call):
            return [node, node.func.value]
        return [node]

    def generic_visit(self, node):
        # documentation strings
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                self._mark(body[0])
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            self._mark(node)
        super().generic_visit(node)

    def visit(self, node):
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)                 and isinstance(node.args[0].value, str):
            fn = node.func
            if (isinstance(fn, ast.Name) and fn.id == "tr") or (isinstance(fn, ast.Attribute) and fn.attr == "tr"):
                self.forced.add(node.args[0].value)
        if id(node) in self.skip:
            return super().visit(node)
        if isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp, ast.Call)):
            p = self._pieces(node)
            if p is not None and (not isinstance(node, ast.Constant) or isinstance(node.value, str)):
                # sums and templates as a whole; their direct parts separately not – texts inside variable parts
                # ("– v pořádku." if ok else …) yes
                for sub in self._direct(node):
                    if sub is not node:
                        self.skip.add(id(sub))
                self._emit(node, p)
        return super().visit(node)


def _ui_text(text: str, template: bool) -> bool:
    """Does the text look like something a person reads (not a key, a path, a regex, a number format)?"""
    fixed = re.sub(r"\{\d+\}", " ", text) if template else text
    if not _LETTER.search(fixed):
        return False
    stripped = fixed.strip()
    if template and len(re.sub(r"[\W\d_]", "", fixed)) < 2:
        return False
    if any(x in text for x in ("(?", "\\d", "\\s", "[^", "\\w")):
        return False
    if _QSS.search(text) and not _DIAC.search(text):      # Qt styles (QSS), not a text for the user
        return False
    if stripped.startswith(("*", "#", "<!--", "http", "%")) and not _DIAC.search(text):
        return False
    if _CARD.match(stripped):                     # lines of the input files of ccx / MYSTRAN
        return False
    if _DIAC.search(fixed):
        return True
    if " " not in stripped:
        if re.search(r"[_./\\]", stripped.rstrip(".…")):       # identifiers, files, modules
            return False
        if re.fullmatch(r"[a-z]+[A-Z]\w*", stripped):           # camelCase
            return False
        if re.fullmatch(r"[A-Z][a-z]:?", stripped):              # "Od", "Do" (labels of fields)
            return True
        return bool(re.search(r"[A-Za-z]{3,}", stripped))       # not Iy, C55 (the translator decides about the rest)
    return bool(re.search(r"[A-Za-z]{2,}", fixed))


# texts from the data files that the GUI shows (file, list, field of an item): the library of friction coefficients
DATA_TEXTS = [("fea_core/data/friction.json", "pairs", ("name", "note"))]


def _data_texts():
    for rel, key, fields in DATA_TEXTS:
        try:
            items = json.loads((ROOT / rel).read_text(encoding="utf-8")).get(key, [])
        except (OSError, ValueError):
            continue
        for i, item in enumerate(items):
            for f in fields:
                t = item.get(f)
                if isinstance(t, str) and t.strip() and _ui_text(t, False):
                    yield t, f"{rel}:{key}[{i}].{f}"


def collect() -> dict[str, dict]:
    """{"exact": {text: [file:line…]}, "templates": {…}}"""
    res = {"exact": {}, "templates": {}}
    for text, where in _data_texts():
        res["exact"].setdefault(text, []).append(where)
    for f, rel in _files():
        try:
            tree = ast.parse(f.read_text(encoding="utf-8-sig"))
        except SyntaxError as e:
            print(f"! {rel}: {e}", file=sys.stderr)
            continue
        c = _Collector()
        c.visit(tree)
        for text, tmpl, line in c.out:
            if text in c.forced and not tmpl and _LETTER.search(text):
                pass                                  # tr("F1 = podrobnosti"): explicitly translated text (also "F1 = …" looks like a key)
            elif not _ui_text(text, tmpl) or (rel == "fea_core/grades.py" and not _DIAC.search(text)):
                continue
            res["templates" if tmpl else "exact"].setdefault(text, []).append(f"{rel}:{line}")
    return res


def load_catalog() -> dict:
    try:
        return json.loads(CATALOG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"exact": {}, "templates": {}}


def save_catalog(cat: dict) -> None:
    out = {"exact": dict(sorted(cat.get("exact", {}).items())),
           "templates": dict(sorted(cat.get("templates", {}).items())),
           "skip": sorted(set(cat.get("skip", [])))}
    CATALOG.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _holes(s: str) -> list[str]:
    return sorted(re.findall(r"\{\d+\}", s))


def merge(path: Path) -> tuple[int, list[str]]:
    cat = load_catalog()
    new = json.loads(Path(path).read_text(encoding="utf-8"))
    n, bad = 0, []
    skip = set(cat.get("skip", []))
    for kind in ("exact", "templates"):
        for en, cs in new.get(kind, {}).items():
            if not isinstance(cs, str) or not cs.strip():
                if isinstance(cs, str):                 # "" = stays (it is not a text for the user)
                    skip.add(en)
                continue
            if kind == "templates" and _holes(cs) != _holes(en):
                bad.append(en)
                continue
            cat.setdefault(kind, {})[en] = cs
            skip.discard(en)
            n += 1
    cat["skip"] = sorted(skip)
    save_catalog(cat)
    return n, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--todo", help="write missing texts to JSON")
    ap.add_argument("--where", action="store_true", help="give the place in the code for missing texts")
    ap.add_argument("--merge", help="add translations from JSON to the catalogue")
    ap.add_argument("--prune", action="store_true", help="remove texts from the catalogue that are no longer in the code")
    a = ap.parse_args(argv)
    if a.merge:
        n, bad = merge(Path(a.merge))
        print(f"{n} translations added")
        for b in bad:
            print(f"  ! variables do not match: {b!r}")
        return 1 if bad else 0
    found = collect()
    cat = load_catalog()
    skip = set(cat.get("skip", []))
    if a.prune:
        for kind in ("exact", "templates"):
            cat[kind] = {k: v for k, v in cat.get(kind, {}).items() if k in found[kind]}
        cat["skip"] = [k for k in cat.get("skip", []) if k in found["exact"] or k in found["templates"]]
        save_catalog(cat)
    todo = {kind: {t: (" | ".join(w[:3]) if a.where else "") for t, w in found[kind].items()
                   if not cat.get(kind, {}).get(t) and t not in skip} for kind in ("exact", "templates")}
    for kind in ("exact", "templates"):
        print(f"{kind}: in code {len(found[kind])}, in catalogue {len(cat.get(kind, {}))}, missing {len(todo[kind])}")
    if a.todo:
        Path(a.todo).write_text(json.dumps(todo, ensure_ascii=False, indent=1), encoding="utf-8")
        print(a.todo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
