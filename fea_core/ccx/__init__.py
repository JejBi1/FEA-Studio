"""CalculiX: input writing (inp.py), run (runner.py), result reading (frd.py)."""
import re

_CTRL = re.compile(r"[\x00-\x1f\x7f]+")


def deck_text(s) -> str:
    """User text (study, part, state names…) in the ccx / MYSTRAN input file: ASCII only, on ONE line,
    no leading asterisk. A new line in a name (foreign .fea) would otherwise start a new solver card.
    """
    return _CTRL.sub(" ", str(s)).encode("ascii", "replace").decode("ascii").lstrip("*")
