"""ReDoS guard of fea_core.i18n.tr: texts from foreign files (names, notes) must not make the translation slow."""
from __future__ import annotations

import time

import pytest

from fea_core import i18n
from fea_core.i18n import _UNIT, _Engine

LIMIT_S = 3.0          # generous bound for a slow CI machine; the quadratic / cubic versions need minutes


@pytest.fixture()
def cs():
    old = i18n.language()
    i18n.set_language("cs")
    yield
    i18n.set_language(old)


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def test_unit_regex_is_linear():
    # adversarial inputs for overlapping stars: digits / spaces / separators followed by a character that breaks the match
    for bad in ("1" * 30000 + "!", " " * 30000 + "!", "1 " * 15000 + "!", "," * 30000 + "!", "e" * 30000 + "!"):
        _, dt = timed(lambda: _UNIT.fullmatch(bad))
        assert dt < LIMIT_S, (bad[:5], dt)


def test_unit_regex_same_results_for_normal_texts():
    for ok in ("1,5 mm", "−12.5 N", "3 × 10⁻⁵ 1/K", "210 000 MPa", "0.3", "1.3 Frame", "1e5", "2 · 3 mm²"):
        assert _UNIT.fullmatch(ok) is not None, ok
    for no in ("mm", "Frame 1.3", "abc", "", "1.3 Frames of x"):
        assert _UNIT.fullmatch(no) is None, no


def test_long_text_is_fast(cs):
    for text in ("1" * 6000 + "x", "1 " * 4000 + "x", "(" * 5000, "x [" * 3000, "<b>" * 3000 + "word",
                 "a: " * 3000 + "b", "line\n" * 3000 + "word", "<br>" * 3000 + "word", " – " * 3000 + "z"):
        _, dt = timed(lambda: i18n.tr(text))
        assert dt < LIMIT_S, (text[:12], dt)


def test_long_text_translated_per_line(cs):
    old = i18n._ENGINES["cs"]
    try:
        i18n.use_catalog({"exact": {"Hello": "Ahoj"}, "templates": {}}, "cs")
        text = "\n".join(["Hello"] * 600) + "\n" + "x" * 1000
        assert len(text) > i18n._LONG_MAX
        out = i18n.tr(text)
        assert out.split("\n")[0] == "Ahoj" and out.split("\n")[599] == "Ahoj"
        assert out.endswith("x" * 1000)
        # a single endless line stays unchanged
        assert i18n.tr("Hello " * 1000) == "Hello " * 1000
    finally:
        i18n.reload()
        i18n._ENGINES["cs"] = old


def test_template_with_holes_is_capped(cs):
    old = i18n._ENGINES["cs"]
    try:
        i18n.use_catalog({"exact": {}, "templates": {"Abc {0} B {1} C {2} D {3} E": "x {0}|{1}|{2}|{3}"}}, "cs")
        assert i18n.tr("Abc 1 B 2 C 3 D 4 E") == "x 1|2|3|4"            # a normal text still matches
        adversarial = "Abc " + " B  C  D  E " * 80 + "!"                 # all fixed parts present, never a match → polynomial blow-up
        out, dt = timed(lambda: i18n.tr(adversarial))
        assert out == adversarial and dt < LIMIT_S, dt
    finally:
        i18n.reload()
        i18n._ENGINES["cs"] = old


def test_engine_limits_are_sane():
    assert i18n._LONG_MAX >= 1000
    assert i18n._TEMPLATE_MAX[3] <= i18n._TEMPLATE_MAX[2] <= i18n._TEMPLATE_MAX[1]
    assert isinstance(_Engine(False, None), _Engine)
