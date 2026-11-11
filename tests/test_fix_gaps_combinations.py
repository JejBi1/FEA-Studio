"""EN 1990 classification of load cases (no single-letter guessing), strict generator inputs, the confirmation table,
contact normal orientation without a silent pass, crash log and Fusion manifest version."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fea_core.combinations import (generate_en1990_combinations, guess_action_class, guess_action_classes,
                                   psi_categories)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["Loading", "Bearing", "Hanging bracket", "Zatěžovací stav 1", "Gear mesh", "LC_g2"])
def test_letter_g_does_not_make_a_case_permanent(name):
    g = guess_action_class(name)
    assert g["action_type"] == "variable" and not g["recognized"]
    assert g["category"] == "cat_E"                     # largest ψ – conservative, to be confirmed by the user


@pytest.mark.parametrize("name,kind,cat", [
    ("Dead load", "permanent", ""), ("Self weight", "permanent", ""), ("Vlastní tíha", "permanent", ""),
    ("Stálé zatížení", "permanent", ""), ("GRAVITY", "permanent", ""),
    ("Snow", "variable", "snow_cz_low"), ("Sníh", "variable", "snow_cz_low"),
    ("Wind +X", "variable", "wind"), ("Vítr", "variable", "wind"),
    ("Imposed load", "variable", "cat_A"), ("Užitné zatížení", "variable", "cat_A"),
    ("Teplota léto", "variable", "temperature"), ("Roof maintenance", "variable", "cat_H"),
    ("Accidental impact", "excluded", ""),
    ("Imposed + self weight", "variable", "cat_A"),   # a mixed name is not taken as permanent
])
def test_word_based_guess(name, kind, cat):
    g = guess_action_class(name)
    assert (g["action_type"], g["category"]) == (kind, cat)
    assert g["recognized"]


def test_preload_is_permanent_favourable():
    g = guess_action_class("Bolt preload")
    assert g["action_type"] == "permanent" and g["favorable"]
    assert [c["name"] for c in guess_action_classes(["A", "Bolt preload"])] == ["A", "Bolt preload"]


def test_guess_categories_exist_in_the_psi_table():
    keys = set(psi_categories())
    for n in ("Snow", "Wind", "Imposed", "Roof", "Temperature", "xyz"):
        c = guess_action_class(n)["category"]
        assert c in keys


def test_generator_rejects_unknown_kinds_and_skips_excluded():
    cases = [{"name": "G", "action_type": "permanent"}, {"name": "Q", "action_type": "variable", "category": "cat_B"},
             {"name": "A", "action_type": "excluded"}]
    combs = generate_en1990_combinations(cases, formula="6.10")
    assert combs and all("A" not in c.factors for c in combs)
    with pytest.raises(ValueError, match="not supported"):
        generate_en1990_combinations([{"name": "X", "action_type": "accidental"}], formula="6.10")
    with pytest.raises(ValueError, match="not supported"):
        generate_en1990_combinations([{"name": "X"}], formula="6.10")
    with pytest.raises(ValueError, match="unknown category"):
        generate_en1990_combinations([{"name": "G", "action_type": "permanent"},
                                      {"name": "Q", "action_type": "variable", "category": "cat_Z"}], formula="6.10")


def test_classification_dialog_rows():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from fea_gui.combinations_dialog import ClassificationDialog
    rows = guess_action_classes(["Self weight", "Loading <b>x</b>", "Snow"])
    dlg = ClassificationDialog(None, rows)
    assert dlg.table.item(1, 0).text() == "Loading <b>x</b>"          # plain text, not interpreted
    assert not dlg._cat[0].isEnabled() and dlg._cat[1].isEnabled()
    dlg._action[1].setCurrentIndex(dlg._action[1].findData("permanent"))
    assert not dlg._cat[1].isEnabled()
    cases = dlg.cases()
    assert [c["action_type"] for c in cases] == ["permanent", "permanent", "variable"]
    assert cases[2]["category"] == "snow_cz_low" and "category" not in cases[0]
    combs = generate_en1990_combinations(cases, formula="all")
    assert combs
    dlg.deleteLater()
    app.processEvents()


def test_contact_orientation_error_is_not_silent():
    from fea_core.contacts.detect import face_normals

    class M:
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [5, 5, 5]], float)
        tets = np.array([[0, 1, 2, 3]])
    good = face_normals(M, np.array([[0, 2, 1]]))         # face 0-1-2 of the tet: outward = −z
    assert good[0] @ np.array([0, 0, 1.0]) < -0.99
    with pytest.raises(ValueError, match="outward normal"):
        face_normals(M, np.array([[0, 1, 4]]))            # not a face of any element


def test_crash_log_is_written_and_anonymised(tmp_path, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    from fea_gui import about
    monkeypatch.setattr(about, "log_dir", lambda: tmp_path / "logs")
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME") or ""
    try:
        raise RuntimeError(f"boom in {home}{os.sep}x.py")
    except RuntimeError as e:
        p = about.write_crash_log(type(e), e, e.__traceback__)
    assert p is not None and p.name.startswith("crash-")
    text = p.read_text(encoding="utf-8")
    from fea_core import __version__
    assert f"FEA Studio {__version__}" in text and "RuntimeError" in text
    if len(home) > 3:
        assert home not in text


def test_convergence_reads_decimal_comma_of_ccx():
    from fea_core.nl_convergence import parse_stdout_residuals
    r = parse_stdout_residuals(" largest residual force= 0,125000 in node 1317 and dof 3\n"
                               " largest correction to disp= 0.1D-02 in node 12 and dof 1\n")
    assert r == [{"worst_force_val": 0.125, "worst_force_node": 1317, "worst_force_dof": 3,
                  "worst_disp_val": 1e-3, "worst_disp_node": 12, "worst_disp_dof": 1}]


def test_fusion_manifest_version():
    from fea_core import __version__
    m = json.loads((ROOT / "adapters" / "fusion" / "FEAStudio" / "FEAStudio.manifest").read_text(encoding="utf-8"))
    assert m["version"] == __version__ and m["supportedOS"] == "windows"
    assert m["description"][""].isascii()
