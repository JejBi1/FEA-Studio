"""Export of results and mesh to the VTU format (XML VTK UnstructuredGrid) for ParaView (A6).

Pure Python and numpy (without external VTK / meshio libraries).
Supports quadratic tetrahedra C3D10 (VTK_QUADRATIC_TETRA = 24), linear tetrahedra (VTK_TETRA = 10),
nodal fields (displacements, stresses, von Mises, principal stresses, further fields of the state – temperature, contact, 3D printing …) and cell data.
Numbers as in the solver input (.inp), from 1: PointData NodeId (*NODE), CellData ElementId (*ELEMENT) and PartNumber
(ELSET P<n>); PartId = the same from 0 (for older files), FieldData PartNames = names of the parts (PartNames[PartId]).
ParaView shows its own "Point ID" / "Cell ID" from 0 – for the numbers of the solver use NodeId / ElementId.
Also generates a ParaView collection (.pvd) for browsing all states / combinations; file names `result_file_name`.
"""
from __future__ import annotations

import base64
import re
import struct
import warnings as _warnings
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr

import numpy as np

from .results import StepResult

if TYPE_CHECKING:
    from .mesh import Mesh
    from .results import Results
    from .study import Study

VTK_TETRA = 10
VTK_QUADRATIC_TETRA = 24


def safe_name(name: object) -> str:
    """Name for a file: only letters, digits, `_`, `.` and `-` (nothing that could lead out of the output folder)."""
    s = re.sub(r"[^\w.-]+", "_", str(name)).strip(".")
    return s or "x"


def result_file_name(base: object, kind: str, index: int, name: object, ext: str) -> str:
    """File name of a result (CSV and VTU, GUI and MCP): `<base>_<NN>_<state>.<ext>`, NN = order of the state from 1 (step_index,
    for linear statics the number of the *STEP), combination `<base>_K<NN>_<name>.<ext>` (NN = order of the combination in the study from 1).
    The index is given from 0.
    """
    tag = f"K{index + 1:02d}" if kind == "comb" else f"{index + 1:02d}"
    return f"{safe_name(base)}_{tag}_{safe_name(name)}.{ext}"


def _string_array(name: str, values: list) -> str:
    """<DataArray type="String"> for FieldData in the ascii VTK format: UTF-8 bytes as numbers (as VTK writes them –
    signed char), each string terminated by 0.
    """
    codes: list[str] = []
    for v in values:
        codes += [str(b - 256 if b > 127 else b) for b in str(v).encode("utf-8")]
        codes.append("0")
    return (f'      <DataArray type="String" Name={quoteattr(name)} NumberOfTuples="{len(values)}" format="ascii">\n'
            f'        {" ".join(codes)}\n'
            "      </DataArray>")


def _binary_data_array(name: str, data: np.ndarray, num_components: int = 1) -> str:
    """Creates a <DataArray> element in the binary base64 format of VTK XML (header_type UInt32, no compression).

    According to the VTK specification the header (UInt32 = number of bytes of data) and the data are encoded in base64 SEPARATELY
    (the reader decodes the header as a separate block), otherwise the first two bytes of the data would shift into the header.
    """
    data = np.ascontiguousarray(data)
    dtype = data.dtype
    if dtype == np.float32 or dtype == np.float64:
        vtk_type = "Float32"
        raw_data = data.astype("<f4").tobytes()
    elif dtype == np.int32 or dtype == np.int64:
        vtk_type = "Int32"
        raw_data = data.astype("<i4").tobytes()
    elif dtype == np.uint8:
        vtk_type = "UInt8"
        raw_data = data.astype("u1").tobytes()
    else:
        vtk_type = "Float32"
        raw_data = data.astype("<f4").tobytes()

    header = struct.pack("<I", len(raw_data))
    b64_content = base64.b64encode(header).decode("ascii") + base64.b64encode(raw_data).decode("ascii")

    return (
        f'        <DataArray type="{vtk_type}" Name={quoteattr(name)} '
        f'NumberOfComponents="{num_components}" format="binary">\n'
        f"          {b64_content}\n"
        f"        </DataArray>"
    )


def export_step_vtu(
    mesh: Mesh,
    step: StepResult | None,
    path: Path | str,
    extra_fields: dict[str, np.ndarray] | None = None,
) -> Path:
    """Exports one calculation step to a separate .vtu file (step=None = only `extra_fields`, e.g. an envelope).
    Further nodal fields of the state (StepResult.fields) are written under the CSV column names (`export_csv.field_column`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_nodes = len(mesh.nodes)
    n_tets = len(mesh.tets)
    order = mesh.tets.shape[1] if n_tets > 0 else 10

    # The node order of the ccx C3D10 corresponds to VTK_QUADRATIC_TETRA (corners 0..3, edge midpoints 0-1, 1-2, 2-0, 0-3, 1-3, 2-3)
    cell_type = VTK_QUADRATIC_TETRA if order == 10 else VTK_TETRA
    nodes_per_cell = order

    connectivity = mesh.tets.astype(np.int32).flatten()
    offsets = np.arange(nodes_per_cell, nodes_per_cell * (n_tets + 1), nodes_per_cell, dtype=np.int32)
    types = np.full(n_tets, cell_type, dtype=np.uint8)

    # Preparation of the fields
    points_xml = _binary_data_array("Points", mesh.nodes, num_components=3)
    conn_xml = _binary_data_array("connectivity", connectivity, num_components=1)
    offsets_xml = _binary_data_array("offsets", offsets, num_components=1)
    types_xml = _binary_data_array("types", types, num_components=1)

    # PointData
    point_arrays = [_binary_data_array("NodeId", np.arange(1, n_nodes + 1, dtype=np.int32), num_components=1)]
    if step is not None and step.U is not None and len(step.U) == n_nodes:
        point_arrays.append(_binary_data_array("Displacement", step.U, num_components=3))
        point_arrays.append(_binary_data_array("Displacement_Magnitude", step.umag, num_components=1))

    if step is not None and step.S is not None and len(step.S) == n_nodes:
        point_arrays.append(_binary_data_array("von_Mises_Stress", step.mises, num_components=1))
        point_arrays.append(_binary_data_array("Stress_Components_SXX_SYY_SZZ_SXY_SYZ_SZX", step.S, num_components=6))
        princ = step.principal
        point_arrays.append(_binary_data_array("Principal_Stress_S1", princ[:, 0], num_components=1))
        point_arrays.append(_binary_data_array("Principal_Stress_S2", princ[:, 1], num_components=1))
        point_arrays.append(_binary_data_array("Principal_Stress_S3", princ[:, 2], num_components=1))

    if step is not None:
        from .export_csv import step_fields
        for f_name, f_data in step_fields(step, n_nodes).items():
            if f_name in (extra_fields or {}):
                continue
            comp = 1 if f_data.ndim == 1 else int(np.prod(f_data.shape[1:]))
            point_arrays.append(_binary_data_array(f_name, f_data.reshape(n_nodes, -1) if comp > 1 else f_data,
                                                   num_components=comp))

    if extra_fields:
        for f_name, f_data in extra_fields.items():
            f_data = np.asarray(f_data)
            if len(f_data) != n_nodes:
                _warnings.warn(f"Field '{f_name}' has {len(f_data)} values instead of {n_nodes} nodes – not written to VTU")
                continue
            comp = 1 if f_data.ndim == 1 else int(np.prod(f_data.shape[1:]))
            point_arrays.append(_binary_data_array(f_name, f_data.reshape(n_nodes, -1) if comp > 1 else f_data,
                                                   num_components=comp))

    # CellData
    cell_arrays = [_binary_data_array("ElementId", np.arange(1, n_tets + 1, dtype=np.int32), num_components=1)]
    field_xml = []
    if hasattr(mesh, "tet_part") and mesh.tet_part is not None and len(mesh.tet_part) == n_tets:
        tp = np.asarray(mesh.tet_part, dtype=np.int32)
        cell_arrays.append(_binary_data_array("PartId", tp, num_components=1))
        cell_arrays.append(_binary_data_array("PartNumber", tp + 1, num_components=1))
        names = list(getattr(mesh, "part_names", None) or [])
        if names:
            field_xml = ["    <FieldData>", _string_array("PartNames", names), "    </FieldData>"]

    point_data_str = "\n".join(point_arrays)
    cell_data_str = "\n".join(cell_arrays)

    xml_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">',
        '  <UnstructuredGrid>',
        *field_xml,
        f'    <Piece NumberOfPoints="{n_nodes}" NumberOfCells="{n_tets}">',
        '      <Points>',
        points_xml,
        '      </Points>',
        '      <Cells>',
        conn_xml,
        offsets_xml,
        types_xml,
        '      </Cells>',
        '      <PointData>',
        point_data_str,
        '      </PointData>',
        '      <CellData>',
        cell_data_str,
        '      </CellData>',
        '    </Piece>',
        '  </UnstructuredGrid>',
        '</VTKFile>',
    ]

    path.write_text("\n".join(xml_lines), encoding="utf-8")
    return path


def export_envelope_vtu(mesh: Mesh, env, path: Path | str) -> Path:
    """Envelope (`combinations.EnvelopeResult`) → .vtu: fields of the envelope and the order of the governing combination in the list of the envelope
    per node (Governing_Combination_Index from 0, Governing_Combination_Number from 1 as in the CSV).
    """
    names = list(env.combination_names)
    gov = np.array([names.index(c) for c in env.governing_combinations], dtype=np.int32)
    return export_step_vtu(mesh, None, path, extra_fields={
        f"Envelope_{env.field_name}": np.asarray(env.values, float),
        "Governing_Combination_Index": gov,
        "Governing_Combination_Number": gov + 1,
    })


def export_combination_vtu(mesh: Mesh, result, path: Path | str) -> Path:
    """Result of a combination (StepResult) or an envelope (EnvelopeResult) → .vtu."""
    if isinstance(result, StepResult):
        return export_step_vtu(mesh, result, path)
    return export_envelope_vtu(mesh, result, path)


def export_results_vtu(
    mesh: Mesh,
    results: Results,
    out_dir: Path | str,
    base_name: str = "results",
    study: Study | None = None,
    warnings: list[str] | None = None,
) -> list[Path]:
    """Exports all states from Results to VTU and creates a ParaView collection (.pvd).

    Files `<base>_<NN>_<state>.vtu` (NN = order of the state from 1, timestep NN in the .pvd), combinations and envelopes from the study
    `<base>_K<NN>_<name>.vtu` (timestep = number of states + NN); a combination that cannot be computed is left out and the reason
    is added to `warnings` (if given, otherwise `warnings.warn`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = safe_name(base_name)

    vtu_files: list[Path] = []
    pvd_entries: list[str] = []

    for i, step in enumerate(results.steps):
        vtu_path = out_dir / result_file_name(base, "step", i, step.name, "vtu")
        export_step_vtu(mesh, step, vtu_path)
        vtu_files.append(vtu_path)
        pvd_entries.append(
            f'    <DataSet timestep="{i + 1}" group="" part="0" file={quoteattr(vtu_path.name)}/>'
        )

    # We also export the combinations if they are in the study
    if study is not None and getattr(study, "combinations", None):
        from .combinations import compute_combination
        comb_offset = len(results.steps)
        cache: dict = {}
        for j, comb in enumerate(study.combinations):
            vtu_path = out_dir / result_file_name(base, "comb", j, comb.name, "vtu")
            try:
                export_combination_vtu(mesh, compute_combination(study, results, comb, cache), vtu_path)
            except Exception as e:
                msg = f"Combination '{comb.name}' was not exported: {e}"
                if warnings is not None:
                    warnings.append(msg)
                else:
                    _warnings.warn(msg)
                continue
            vtu_files.append(vtu_path)
            pvd_entries.append(
                f'    <DataSet timestep="{comb_offset + j + 1}" group="combinations" part="0" '
                f'file={quoteattr(vtu_path.name)}/>'
            )

    # Writing the .pvd collection
    pvd_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
        *pvd_entries,
        '  </Collection>',
        '</VTKFile>',
    ]
    pvd_path = out_dir / f"{base}.pvd"
    pvd_path.write_text("\n".join(pvd_lines), encoding="utf-8")
    vtu_files.append(pvd_path)

    return vtu_files
