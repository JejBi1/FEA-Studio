"""F11: short labels. Without Qt, fast: the source files of the GUI are walked through the AST and every literal in `setToolTip(...)`,
a short text (2nd argument) of `help_tip(...)` and `tip=` of `Section(...)` may have at most `TIP_MAX` (125) characters and must not
contain „\\n“ (an f-string: `{…}` counts as 4 characters). Exception: the comment `# tip-ok` on the line of the call.
A longer explanation belongs to `setWhatsThis` / `help_tip(obj, short, long)` (F1, ⓘ, Shift+F1).

Modules that nobody has rewritten yet (`PENDING`) are `xfail` (they do not block the parallel work); `FEA_TIPS_STRICT=1`
turns them into errors (switch on at the end of F11 – then empty `PENDING`).
"""
import ast
import os
from pathlib import Path

import pytest

GUI = Path(__file__).resolve().parents[1] / "fea_gui"
TIP_MAX = 125            # the English source texts are longer than the Czech ones they replaced (≈ 20 %)

# the checked modules (at the end of F11 all the panels and main.py; the dialogs later)
CHECKED = ["widgets", "guide", "tour", "theme", "main", "eigen_gui", "twod_gui", "shell_gui", "ortho_box",
           "dynamics_panel", "contact_panel", "dyn_results", "print_panel", "print_variants", "print_sweep",
           "thermal_panel", "fatigue_panel", "pragtic_dialog", "frame_panel", "frame_results", "motion_panel",
           "joints_panel", "messages"]
# not rewritten yet (A2–A5 tick them off; whoever is clean, only "XPASS")
PENDING: set[str] = set()          # F11 completed (18. 10. 2026) – all modules strictly


def _length(node) -> int | None:
    """Length of a literal / f-string / sum of literals; None = cannot be determined (variable, call)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return len(node.value)
    if isinstance(node, ast.JoinedStr):
        n = 0
        for v in node.values:
            if isinstance(v, ast.Constant):
                n += len(v.value)
            else:
                n += 4
        return n
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _length(node.left), _length(node.right)
        return None if a is None or b is None else a + b
    return None


def _has_newline(node) -> bool:
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) and "\n" in n.value for n in ast.walk(node))


def _name(call: ast.Call) -> str:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def violations(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        name = _name(node)
        if name == "setToolTip" and node.args:
            tip = node.args[0]
        elif name == "help_tip" and len(node.args) >= 2:
            tip = node.args[1]
        elif name == "help_tip" and any(k.arg == "short" for k in node.keywords):
            tip = next(k.value for k in node.keywords if k.arg == "short")
        elif name == "Section" and any(k.arg == "tip" for k in node.keywords):
            tip = next(k.value for k in node.keywords if k.arg == "tip")
        elif name == "_btn" and len(node.args) >= 2:
            tip = node.args[1]
        elif name == "_btn" and any(k.arg == "tip" for k in node.keywords):
            tip = next(k.value for k in node.keywords if k.arg == "tip")
        elif name == "EditTable" and len(node.args) >= 2:
            tip = node.args[1]
        elif name == "EditTable" and any(k.arg == "tip" for k in node.keywords):
            tip = next(k.value for k in node.keywords if k.arg == "tip")
        elif name == "_vec" and any(k.arg == "tip" for k in node.keywords):
            tip = next(k.value for k in node.keywords if k.arg == "tip")
        elif name == "_vec" and len(node.args) >= 5:
            tip = node.args[4]
        elif name == "vec" and path.name != "twod_gui.py" and len(node.args) >= 2:
            tip = node.args[1]
        elif name == "vec" and path.name != "twod_gui.py" and any(k.arg in ("tip", "tips") for k in node.keywords):
            tip = next(k.value for k in node.keywords if k.arg in ("tip", "tips"))
        else:
            continue
        end = getattr(node, "end_lineno", node.lineno)
        if any("# tip-ok" in lines[i] for i in range(node.lineno - 1, end)):
            continue
        n = _length(tip)
        if (n is not None and n > TIP_MAX) or _has_newline(tip):
            out.append(f"{path.name}:{node.lineno}: popisek {n or '?'} znaků" +
                       (" (s „\\n“)" if _has_newline(tip) else "") + " – krátký text do toolTip, zbytek do whatsThis")
    return out


def _param(m: str):
    strict = os.environ.get("FEA_TIPS_STRICT") == "1"
    marks = [pytest.mark.xfail(reason="modul zatím není přepsaný na krátké popisky (F11)", strict=False)] \
        if m in PENDING and not strict else []
    return pytest.param(m, marks=marks, id=m)


@pytest.mark.parametrize("module", [_param(m) for m in CHECKED])
def test_tooltips_are_short(module):
    path = GUI / f"{module}.py"
    assert path.exists(), path
    bad = violations(path)
    assert not bad, f"{len(bad)} dlouhých popisků:\n" + "\n".join(bad[:15])


def test_checker_detects_long_and_multiline(tmp_path):
    f = tmp_path / "x.py"
    f.write_text('b.setToolTip("%s")\n' % ("a" * 126) +
                 'b.setToolTip("a\\nb")\n'
                 'b.setToolTip(f"{x} ' + "a" * 100 + '")\n'
                 'help_tip(b, "' + "a" * 130 + '", "dlouhý")\n'
                 'b.setToolTip("' + "a" * 135 + '")  # tip-ok\n'
                 'b.setToolTip("krátký")\n'
                 'Section("T", tip="' + "a" * 126 + '")\n'
                 '_btn("X", "' + "a" * 127 + '", None)\n'
                 'EditTable(["A"], "' + "a" * 128 + '")\n'
                 '_vec([1], tip="' + "a" * 129 + '")\n', encoding="utf-8")
    bad = violations(f)
    assert [b.split(":")[1] for b in bad] == ["1", "2", "4", "7", "8", "9", "10"], bad


# ------------------------------------------------------------------ short_of (Qt only because of the import of guide)
def _short_of():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from fea_gui.guide import short_of
    return short_of


def test_short_of():
    short_of = _short_of()
    assert short_of("Krátký popisek") == "Krátký popisek"
    assert short_of("První řádek\ndruhý řádek") == "První řádek"
    long_sentence = "Poměrné tlumení ζ určuje útlum kmitání konstrukce. Typicky 0,01 až 0,05 pro svařované ocelové celky."
    assert short_of(long_sentence) == "Poměrné tlumení ζ určuje útlum kmitání konstrukce."
    semi = "Zadejte hodnotu tlumení pro všechny módy najednou; pokud je neznáte, použijte odhad podle materiálu."
    assert short_of(semi) == "Zadejte hodnotu tlumení pro všechny módy najednou…"
    dash = "Zadejte hodnotu tlumení pro všechny módy najednou – pokud je neznáte, použijte odhad podle materiálu."
    assert short_of(dash) == "Zadejte hodnotu tlumení pro všechny módy najednou…"
    colon = "Zadejte hodnotu tlumení pro všechny módy najednou: pokud je neznáte, použijte odhad podle materiálu."
    assert short_of(colon).endswith("najednou…")
    words = short_of("slovo " * 40)
    assert words.endswith("slovo…") and len(words) <= TIP_MAX
    assert len(short_of("x" * 300)) <= TIP_MAX and short_of("x" * 300).endswith("…")
    assert short_of("<b>Tučně</b> a <i>šikmo</i><br>další řádek") == "Tučně a šikmo"
    for limit in (40, 60, 90):
        assert len(short_of(long_sentence * 2, limit)) <= limit
