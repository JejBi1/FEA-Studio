"""Dialog for splitting a face (Imprint / Split Face, A10)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QVBoxLayout, QWidget
)

from fea_core.imprint import Imprint
from . import theme
from .widgets import dspin


class ImprintDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Face Imprint (Split Face / Imprint)")
        self.resize(550, 460)

        lay = QVBoxLayout(self)

        lbl = QLabel(
            "<b>Geometric Face Imprint (A10)</b><p>Splits the body surface with a circular or rectangular imprint (OpenCASCADE fragmentation). The resulting face is automatically added to named selections and can receive pressure, force, support, or contact.</p>"
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        # List of the existing imprints
        self.list_imprints = QListWidget()
        self.list_imprints.setToolTip("List of defined geometric imprints on the model")
        self.list_imprints.currentRowChanged.connect(self._on_select_imprint)
        lay.addWidget(QLabel("<b>Existing imprints:</b>"))
        lay.addWidget(self.list_imprints)

        btn_del = QPushButton("Delete selected imprint")
        btn_del.setToolTip("Delete selected geometric imprint from the study")
        btn_del.clicked.connect(self._delete_imprint)
        lay.addWidget(btn_del)

        # Form of a new imprint
        g = QGroupBox("New imprint")
        f = QFormLayout(g)

        self.txt_name = QLineEdit("Otisk_1")
        self.txt_name.setToolTip("Unique imprint name (becomes the selection name)")
        f.addRow("Imprint name:", self.txt_name)

        self.combo_part = QComboBox()
        self.combo_part.setToolTip("Target model part on whose surface the imprint is created")
        parts = getattr(win, "parts", None)
        if parts:
            for p in parts:
                self.combo_part.addItem(p.name, p.name)
        elif win.study and win.study.part_materials:
            for p_name in win.study.part_materials:
                self.combo_part.addItem(p_name, p_name)
        f.addRow("Target part:", self.combo_part)

        self.combo_shape = QComboBox()
        self.combo_shape.setToolTip("Imprint shape (circular for punches and pins, rectangular for pads)")
        self.combo_shape.addItem("Circular (punch / washer / pin)", "circle")
        self.combo_shape.addItem("Rectangular (pad / strip)", "rectangle")
        self.combo_shape.currentIndexChanged.connect(self._on_shape_changed)
        f.addRow("Imprint shape:", self.combo_shape)

        # Centre
        h_c = QHBoxLayout()
        self.sp_cx = dspin(0.0, -1e6, 1e6, 3)
        self.sp_cx.setToolTip("X coordinate of imprint center in global coordinate system [mm]")
        self.sp_cy = dspin(0.0, -1e6, 1e6, 3)
        self.sp_cy.setToolTip("Y coordinate of imprint center in global coordinate system [mm]")
        self.sp_cz = dspin(0.0, -1e6, 1e6, 3)
        self.sp_cz.setToolTip("Z coordinate of imprint center in global coordinate system [mm]")
        h_c.addWidget(QLabel("X:")); h_c.addWidget(self.sp_cx)
        h_c.addWidget(QLabel("Y:")); h_c.addWidget(self.sp_cy)
        h_c.addWidget(QLabel("Z:")); h_c.addWidget(self.sp_cz)
        f.addRow("Center [mm]:", h_c)

        # Normal
        h_n = QHBoxLayout()
        self.sp_nx = dspin(0.0, -1.0, 1.0, 3)
        self.sp_nx.setToolTip("X component of the normal vector of the face hosting the imprint")
        self.sp_ny = dspin(0.0, -1.0, 1.0, 3)
        self.sp_ny.setToolTip("Y component of the normal vector of the face hosting the imprint")
        self.sp_nz = dspin(1.0, -1.0, 1.0, 3)
        self.sp_nz.setToolTip("Z component of the normal vector of the face hosting the imprint")
        h_n.addWidget(QLabel("nx:")); h_n.addWidget(self.sp_nx)
        h_n.addWidget(QLabel("ny:")); h_n.addWidget(self.sp_ny)
        h_n.addWidget(QLabel("nz:")); h_n.addWidget(self.sp_nz)
        f.addRow("Face normal:", h_n)

        # Dimensions
        self.w_circle = QWidget()
        h_cir = QHBoxLayout(self.w_circle)
        h_cir.setContentsMargins(0, 0, 0, 0)
        self.sp_radius = dspin(10.0, 0.01, 1e5, 3)
        self.sp_radius.setToolTip("Circular imprint radius R [mm]")
        h_cir.addWidget(QLabel("Radius R [mm]:"))
        h_cir.addWidget(self.sp_radius)
        f.addRow(self.w_circle)

        self.w_rect = QWidget()
        h_rec = QHBoxLayout(self.w_rect)
        h_rec.setContentsMargins(0, 0, 0, 0)
        self.sp_dx = dspin(20.0, 0.01, 1e5, 3)
        self.sp_dx.setToolTip("Rectangular imprint dimension dx along first axis [mm]")
        self.sp_dy = dspin(20.0, 0.01, 1e5, 3)
        self.sp_dy.setToolTip("Rectangular imprint dimension dy along second axis [mm]")
        h_rec.addWidget(QLabel("Length dx [mm]:")); h_rec.addWidget(self.sp_dx)
        h_rec.addWidget(QLabel("Width dy [mm]:")); h_rec.addWidget(self.sp_dy)
        self.w_rect.setVisible(False)
        f.addRow(self.w_rect)

        lay.addWidget(g)

        # Buttons
        h_btn = QHBoxLayout()
        btn_add = QPushButton("Add imprint")
        theme.set_primary(btn_add)
        btn_add.setToolTip("Create new geometric imprint and add it to the study")
        btn_add.clicked.connect(self._add_imprint)
        btn_close = QPushButton("Close")
        btn_close.setToolTip("Close imprint dialog")
        btn_close.clicked.connect(self.accept)
        h_btn.addStretch(1)
        h_btn.addWidget(btn_add)
        h_btn.addWidget(btn_close)
        lay.addLayout(h_btn)

        self._refresh_list()

    def _on_shape_changed(self):
        shape = self.combo_shape.currentData()
        self.w_circle.setVisible(shape == "circle")
        self.w_rect.setVisible(shape == "rectangle")

    def _refresh_list(self):
        self.list_imprints.clear()
        if not self.win.study:
            return
        for imp in self.win.study.imprints:
            dim_str = f"R = {imp.radius:g} mm" if imp.shape == "circle" else f"{imp.dx:g}×{imp.dy:g} mm"
            self.list_imprints.addItem(f"{imp.name} [{imp.part}] – {imp.shape} ({dim_str})")

    def _on_select_imprint(self, row: int):
        if row < 0 or not self.win.study or row >= len(self.win.study.imprints):
            return

    def _add_imprint(self):
        name = self.txt_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Enter an imprint name.")
            return

        study = self.win.study
        if not study:
            return

        if any(imp.name == name for imp in study.imprints):
            QMessageBox.warning(self, "Error", f"Imprint '{name}' already exists.")
            return

        imp = Imprint(
            name=name,
            part=self.combo_part.currentText(),
            shape=self.combo_shape.currentData(),
            center=[self.sp_cx.value(), self.sp_cy.value(), self.sp_cz.value()],
            normal=[self.sp_nx.value(), self.sp_ny.value(), self.sp_nz.value()],
            radius=self.sp_radius.value(),
            dx=self.sp_dx.value(),
            dy=self.sp_dy.value(),
        )

        study.imprints.append(imp)
        self.win.changed(invalidate_mesh=True)
        self._refresh_list()
        self.txt_name.setText(f"Otisk_{len(study.imprints) + 1}")
        QMessageBox.information(
            self, "Imprint added",
            f"Imprint '{name}' was added.\nOn next meshing / solve the face will be cut and added to selections."
        )

    def _delete_imprint(self):
        row = self.list_imprints.currentRow()
        if row < 0 or not self.win.study or row >= len(self.win.study.imprints):
            return
        imp = self.win.study.imprints.pop(row)
        self.win.changed(invalidate_mesh=True)
        self._refresh_list()
        QMessageBox.information(self, "Deleted", f"Imprint '{imp.name}' was removed.")
