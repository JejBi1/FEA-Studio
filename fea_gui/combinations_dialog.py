"""Dialog for load-state combinations and envelopes according to ČSN EN 1990 (A2)."""
from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget
)

from .widgets import dspin


def _plain_label(text: str) -> QLabel:
    """Label with a user / CAD name: plain text (no rich-text interpretation), not translated."""
    lb = QLabel()
    lb.setTextFormat(Qt.PlainText)
    lb.setProperty("_i18n_done", True)
    lb.setText(text)
    return lb


ACTION_LABELS = {"permanent": "Permanent (G)", "variable": "Variable (Q)", "excluded": "Not combined"}
CATEGORY_LABELS = {
    "cat_A": "Imposed A: domestic, residential", "cat_B": "Imposed B: offices",
    "cat_C": "Imposed C: congregation areas", "cat_D": "Imposed D: shopping areas",
    "cat_E": "Imposed E: storage areas", "cat_F": "Imposed F: traffic, vehicle ≤ 30 kN",
    "cat_G": "Imposed G: traffic, 30–160 kN", "cat_H": "Imposed H: roofs",
    "snow_cz_low": "Snow, altitude ≤ 1000 m", "snow_cz_high": "Snow, altitude > 1000 m",
    "wind": "Wind", "temperature": "Temperature (non-fire)",
}


class ClassificationDialog(QDialog):
    """Table of the load cases: action type, ψ category and favourable effect, confirmed by the user before
    the EN 1990 combinations are generated (the pre-filled values are only a guess from the case names)."""

    def __init__(self, parent, rows: list[dict]):
        super().__init__(parent)
        from fea_core.combinations import psi_categories
        self.setWindowTitle("Classification of load cases (ČSN EN 1990)")
        self.resize(820, 360)
        self._rows = rows
        self._cats = psi_categories()
        lay = QVBoxLayout(self)
        info = QLabel(
            "Check the classification of every load case. The proposal is only a guess from whole words of the case "
            "names (e.g. dead, weight, snow, wind, imposed); rows marked “check” were not recognised. "
            "Every variable action is taken in turn as the leading one; accidental actions are not combined.")
        info.setWordWrap(True)
        lay.addWidget(info)

        self.table = QTableWidget(len(rows), 5)
        self.table.setToolTip("One row per load case: action type, ψ category of a variable action, favourable effect")
        self.table.setHorizontalHeaderLabels(["Load case", "Action", "Category (ψ)", "Favourable", "Note"])
        hh = self.table.horizontalHeader()
        for c in range(4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self._action: list[QComboBox] = []
        self._cat: list[QComboBox] = []
        self._fav: list[QCheckBox] = []
        for r, row in enumerate(rows):
            it = QTableWidgetItem(row["name"])           # plain text item (names are not interpreted as HTML)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, it)

            act = QComboBox()
            act.setToolTip("Permanent = γ_G, variable = γ_Q with ψ factors, not combined = left out")
            for key, label in ACTION_LABELS.items():
                act.addItem(label, key)
            act.setCurrentIndex(max(0, act.findData(row.get("action_type", "variable"))))
            self.table.setCellWidget(r, 1, act)

            cat = QComboBox()
            cat.setToolTip("ψ factors of the variable action (table A1.1 of ČSN EN 1990 + NA)")
            for key, desc in self._cats.items():
                cat.addItem(CATEGORY_LABELS.get(key, desc), key)
            ci = cat.findData(row.get("category") or "cat_E")
            cat.setCurrentIndex(max(0, ci))
            self.table.setCellWidget(r, 2, cat)

            fav = QCheckBox()
            fav.setToolTip("Favourable action: permanent with γ_G,fav, variable with 0")
            fav.setChecked(bool(row.get("favorable", False)))
            self.table.setCellWidget(r, 3, fav)

            note = row.get("reason", "")
            if not row.get("recognized", True):
                note = "check – " + note
            nit = QTableWidgetItem(note)
            nit.setFlags(nit.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 4, nit)

            act.currentIndexChanged.connect(lambda _i, r=r: self._sync_row(r))
            self._action.append(act)
            self._cat.append(cat)
            self._fav.append(fav)
            self._sync_row(r)
        lay.addWidget(self.table)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Generate combinations")
        bb.button(QDialogButtonBox.Ok).setToolTip("Generate the combinations with this classification")
        bb.button(QDialogButtonBox.Cancel).setToolTip("Close without generating")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _sync_row(self, r: int) -> None:
        kind = self._action[r].currentData()
        self._cat[r].setEnabled(kind == "variable")
        self._fav[r].setEnabled(kind != "excluded")

    def cases(self) -> list[dict]:
        """Rows in the form of `combinations.generate_en1990_combinations` (+ action_type "excluded")."""
        out = []
        for r, row in enumerate(self._rows):
            kind = self._action[r].currentData()
            c = {"name": row["name"], "action_type": kind, "favorable": self._fav[r].isChecked()}
            if kind == "variable":
                c["category"] = self._cat[r].currentData()
            out.append(c)
        return out


class CombinationsDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Load case combinations (ČSN EN 1990)")
        self.resize(750, 520)

        lay = QVBoxLayout(self)

        lbl = QLabel(
            "<b>Load case combinations and envelopes (A2)</b><p>Define linear combinations of results (U, S, RF) or envelopes of extreme values. Code combinations can be generated automatically strictly per ČSN EN 1990 + National Annex. Equivalent von Mises stress and safety factor are computed from the combined stress tensor.</p>"
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        # Table of combinations
        self.table_combs = QTableWidget()
        self.table_combs.setToolTip("Combinations and envelopes of the study (each appears in the case dropdown in Results); click a row to select it for deletion")
        self.table_combs.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_combs.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_combs.setSelectionMode(QTableWidget.SingleSelection)
        self.table_combs.setColumnCount(4)
        self.table_combs.setHorizontalHeaderLabels(["Name", "Type", "Rule / Factors", "Standard / Description"])
        self.table_combs.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_combs.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_combs.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_combs.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        lay.addWidget(self.table_combs)

        # Buttons below the table
        h_ctrl = QHBoxLayout()
        btn_gen = QPushButton("Generate per ČSN EN 1990...")
        btn_gen.setToolTip("Automatically builds the design combinations STR/GEO (6.10, 6.10a/b), EQU and SLS.")
        btn_gen.clicked.connect(self._generate_en1990)
        h_ctrl.addWidget(btn_gen)

        btn_del = QPushButton("Delete selected")
        btn_del.setToolTip("Deletes the combination selected in the table (envelopes based on it must be edited separately)")
        btn_del.clicked.connect(self._delete_selected)
        h_ctrl.addWidget(btn_del)

        btn_clear = QPushButton("Delete all")
        btn_clear.setToolTip("Deletes all combinations and envelopes of the study (after confirmation)")
        btn_clear.clicked.connect(self._clear_all)
        h_ctrl.addWidget(btn_clear)
        lay.addLayout(h_ctrl)

        # Group for adding a linear combination by hand
        g_add = QGroupBox("Add custom combination")
        f_add = QFormLayout(g_add)

        self.txt_name = QLineEdit("Kombinace_1")
        self.txt_name.setToolTip("Name of the new combination; must be unique within the study")
        f_add.addRow("Name:", self.txt_name)

        # Inputs for the individual states
        self.lc_spins: dict[str, QDoubleSpinBox] = {}
        h_spins = QHBoxLayout()
        if win.study and win.study.load_cases:
            for lc in win.study.load_cases:
                sp = dspin(1.0, -100.0, 100.0, 3)
                sp.setToolTip(f"Factor by which the results of case “{html.escape(lc.name)}” are multiplied and added to the combination (0 = the case is not included)")
                self.lc_spins[lc.name] = sp
                h_spins.addWidget(_plain_label(f"{lc.name}:"))
                h_spins.addWidget(sp)
        else:
            h_spins.addWidget(QLabel("No load cases are defined in the study."))
        f_add.addRow("Factors:", h_spins)

        btn_add = QPushButton("Add linear combination")
        btn_add.setToolTip("Creates a combination Σ (factor × case result) from the values entered above")
        btn_add.clicked.connect(self._add_custom)
        f_add.addRow("", btn_add)
        lay.addWidget(g_add)

        # Bottom buttons
        h_btn = QHBoxLayout()
        btn_close = QPushButton("Close")
        btn_close.setToolTip("Close the dialog (combinations stay in the study)")
        btn_close.clicked.connect(self.accept)
        h_btn.addStretch(1)
        h_btn.addWidget(btn_close)
        lay.addLayout(h_btn)

        self._refresh_table()

    def _commit(self):
        """Combinations are only post-processing of the results: the project is marked as changed (undo, save),
        but the calculation results stay.
        """
        self._refresh_table()
        if hasattr(self.win, "changed"):
            self.win.changed(invalidate_results=False)
        if hasattr(self.win, "_sync_result_controls"):
            self.win._sync_result_controls()

    def _refresh_table(self):
        self.table_combs.setRowCount(0)
        study = self.win.study
        if not study or not hasattr(study, "combinations"):
            return

        for row, c in enumerate(study.combinations):
            self.table_combs.insertRow(row)
            self.table_combs.setItem(row, 0, QTableWidgetItem(c.name))
            kind_str = f"Envelope ({c.envelope_type}, {c.envelope_field})" if c.envelope else "Linear"
            self.table_combs.setItem(row, 1, QTableWidgetItem(kind_str))
            if c.envelope:
                f_str = f"Z: {', '.join(c.envelope_combinations)}"
            else:
                f_str = " + ".join(f"{v:g}·{k}" for k, v in c.factors.items())
            self.table_combs.setItem(row, 2, QTableWidgetItem(f_str))
            desc = c.description or c.standard or ""
            self.table_combs.setItem(row, 3, QTableWidgetItem(desc))

    def _add_custom(self):
        name = self.txt_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Enter a combination name.")
            return

        study = self.win.study
        if not study:
            return

        if any(c.name == name for c in study.combinations):
            QMessageBox.warning(self, "Error", f"Combination '{name}' already exists.")
            return

        factors = {k: sp.value() for k, sp in self.lc_spins.items() if abs(sp.value()) > 1e-9}
        if not factors:
            QMessageBox.warning(self, "Error", "At least one factor must be non-zero.")
            return

        from fea_core.combinations import Combination, validate_combination
        comb = Combination(
            name=name,
            factors=factors,
            standard="custom",
            design_situation="custom",
            description="Custom combination",
        )
        try:
            validate_combination(study, comb)
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        study.combinations.append(comb)
        self.txt_name.setText(f"Kombinace_{len(study.combinations) + 1}")
        self._commit()

    def classify_cases(self, study) -> list[dict] | None:
        """Classification of the load cases confirmed by the user (None = cancelled).

        Pre-filled by the previous confirmed choice of this session (kept on the study object, not saved), otherwise by the
        word-based guess `combinations.guess_action_classes`. Tests replace this method.
        """
        from fea_core.combinations import guess_action_class
        prev = getattr(study, "_en1990_classes", None) or {}
        rows = [dict(prev[lc.name], name=lc.name, recognized=True, reason="confirmed earlier in this session")
                if lc.name in prev else guess_action_class(lc.name)
                for lc in study.load_cases]
        dlg = ClassificationDialog(self, rows)
        if not dlg.exec():                         # 0 = Rejected
            return None
        cases = dlg.cases()
        try:
            study._en1990_classes = {c["name"]: dict(c) for c in cases}
        except AttributeError:
            pass
        return cases

    def _generate_en1990(self):
        study = self.win.study
        if not study or not study.load_cases:
            QMessageBox.warning(self, "Error", "First define at least two load cases (e.g. permanent and imposed load).")
            return

        cases = self.classify_cases(study)
        if cases is None:
            return
        cases = [c for c in cases if c["action_type"] != "excluded"]
        if not cases:
            QMessageBox.warning(self, "Error", "All load cases are excluded – nothing to combine.")
            return

        from fea_core.combinations import generate_en1990_combinations, validate_combination
        try:
            combs = generate_en1990_combinations(cases, formula="all")
            for c in combs:
                validate_combination(study, c, others=combs)
        except (ValueError, KeyError) as e:
            QMessageBox.warning(self, "Error", f"The combinations cannot be generated: {e}")
            return
        for c in combs:
            study.combinations = [old for old in study.combinations if old.name != c.name]
            study.combinations.append(c)

        self._commit()
        QMessageBox.information(
            self, "Generation complete",
            f"{len(combs)} code combinations and envelopes (STR/GEO 6.10, 6.10a/b, EQU and SLS) were generated per ČSN EN 1990."
        )

    def _delete_selected(self):
        row = self.table_combs.currentRow()
        study = self.win.study
        if row < 0 or not study or row >= len(study.combinations):
            return
        del study.combinations[row]
        self._commit()

    def _clear_all(self):
        study = self.win.study
        if not study:
            return
        if not study.combinations:
            return
        if QMessageBox.question(
                self, "Delete all combinations",
                f"Really delete all {len(study.combinations)} combinations and envelopes?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        study.combinations.clear()
        self._commit()
