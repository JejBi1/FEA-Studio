"""The numbers in the results export (CSV, VTU) = the numbers in the solver input: node_id (*NODE), element_id (*ELEMENT), part_id
(ELSET P<n>), step_index (order of the state = *STEP), all from 1; the parts of the nodes at the interface, duplicated nodes, reactions with the row
TOTAL and the moment point, format for Czech Excel, all states in one file, file names without collisions.
Without mesh and ccx (a hand-made mesh of two tet10); the GUI parts offscreen.
"""
import base64
import csv
import io
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from fea_core.combinations import Combination
from fea_core.export_csv import (EXCEL, dialect_of, export_elements_csv, export_nodal_csv, export_nodal_long_csv,
                                 export_reactions_csv, export_results_csv, write_csv)
from fea_core.export_vtu import export_results_vtu, export_step_vtu, result_file_name
from fea_core.mesh import Mesh
from fea_core.results import Results, StepResult
from fea_core.study import Fixed, Study

# two tet10 with a common face (nodes 1, 2, 3 + midpoints of the edges 4, 5, 6 numbered from 0), each a different part
NODES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5],
                  [.5, 0, .5], [0, .5, .5], [1, 1, 1], [1, .5, .5], [.5, 1, .5], [.5, .5, 1]], float)
TETS = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 [10, 2, 1, 3, 12, 5, 11, 13, 9, 8]])


def _model(names=("Deska", "Kotva")):
    mesh = Mesh(nodes=NODES.copy(), tets=TETS.copy(), tet_part=np.array([0, 1]), part_names=list(names))
    n = len(NODES)
    S1 = np.zeros((n, 6))
    S1[:, 0] = 50.0
    s1 = StepResult("Tah", np.full((n, 3), 0.1), S1, reactions={"Vetknuti": np.array([0.0, 0.0, 100.0])},
                    reaction_moments={"Vetknuti": np.array([10.0, 20.0, 0.0])}, moment_point=np.array([0.5, 0.5, 0.5]))
    s1.fields = {"T": np.linspace(20.0, 33.0, n), "CPRESS": np.full(n, np.nan), "po_prvcich": np.zeros(2)}
    s2 = StepResult("Tlak", np.full((n, 3), -0.2), 2 * S1,
                    reactions={"Vetknuti": np.array([0.0, 0.0, -200.0]), "Kotva_A": np.array([5.0, 0.0, 0.0])},
                    reaction_moments={"Vetknuti": np.array([1.0, 0.0, 0.0]), "Kotva_A": np.array([0.0, 2.0, 0.0])},
                    moment_point=np.array([0.5, 0.5, 0.5]))
    res = Results(mesh=mesh, steps=[s1, s2])
    st = Study(name="Zkouska", geometry="")
    st.constraints.append(Fixed("Vetknuti"))
    st.combinations.append(Combination(name="K", factors={"Tah": 1.0, "Tlak": 1.5}))
    st.combinations.append(Combination(name="Obalka", envelope=True, envelope_combinations=["K", "Tlak"]))
    return mesh, res, st


def _rows(path, dialect="standard"):
    d = dialect_of(dialect)
    with open(path, encoding=d.encoding, newline="") as f:
        return list(csv.reader(f, delimiter=d.delimiter))


def _col(rows, name):
    i = rows[0].index(name)
    return [r[i] for r in rows[1:]]


def read_vtu_array(path, name):
    root = ET.parse(path).getroot()
    for da in root.iter("DataArray"):
        if da.attrib["Name"] != name:
            continue
        if da.attrib["type"] == "String":
            codes = [int(v) & 0xFF for v in da.text.split()]
            return [s.decode("utf-8") for s in bytes(codes).split(b"\0")[:-1]]
        txt = "".join(da.text.split())
        n_bytes = struct.unpack("<I", base64.b64decode(txt[:8]))[0]
        raw = base64.b64decode(txt[8:])
        assert len(raw) == n_bytes
        a = np.frombuffer(raw, dtype={"Float32": "<f4", "Int32": "<i4", "UInt8": "u1"}[da.attrib["type"]])
        nc = int(da.attrib["NumberOfComponents"])
        return a.reshape(-1, nc) if nc > 1 else a
    raise KeyError(name)


def _inp_numbers(mesh):
    """Node and element numbers as written by the input of ccx (fea_core.ccx.inp.write_mesh)."""
    from fea_core.ccx.inp import InpMap, write_mesh
    buf = io.StringIO()
    write_mesh(buf, mesh, InpMap())
    nodes, elems, eset = {}, {}, None
    block = None
    for line in buf.getvalue().splitlines():
        if line.startswith("**"):
            continue
        if line.startswith("*"):
            block = line.split(",")[0].upper()
            if block == "*ELEMENT":
                eset = line.split("ELSET=")[1].strip()
            continue
        v = [x.strip() for x in line.split(",") if x.strip()]
        if block == "*NODE":
            nodes[int(v[0])] = [float(x) for x in v[1:4]]
        elif block == "*ELEMENT":
            elems[int(v[0])] = (eset, [int(x) for x in v[1:]])
    return nodes, elems


# ===================================================================== nodes
def test_nodal_ids_parts_match_solver_deck(tmp_path):
    mesh, res, st = _model()
    p = export_nodal_csv(mesh, res.steps[0], tmp_path / "n.csv")
    rows = _rows(p)
    assert rows[0][:7] == ["node_id", "part_id", "part", "n_parts", "x_mm", "y_mm", "z_mm"]
    nodes, elems = _inp_numbers(mesh)
    ids = [int(v) for v in _col(rows, "node_id")]
    assert ids == sorted(nodes) == list(range(1, len(NODES) + 1))          # node_id = number in *NODE
    for r in rows[1:]:
        assert np.allclose([float(v) for v in r[4:7]], nodes[int(r[0])])
    # part of the node: the ELSET of the elements containing the node (the smallest part number), n_parts > 1 = interface
    sets = {}
    for eset, conn in elems.values():
        for nid in conn:
            sets.setdefault(nid, set()).add(int(eset[1:]))
    for r in rows[1:]:
        nid = int(r[0])
        assert int(r[1]) == min(sets[nid]) and int(r[3]) == len(sets[nid])
        assert r[2] == mesh.part_names[int(r[1]) - 1]
    shared = {nid for nid, s in sets.items() if len(s) > 1}
    assert shared == {2, 3, 4, 6, 9, 10}                                   # common face (6 nodes of tri6)


def test_alias_column_only_with_duplicated_nodes(tmp_path):
    mesh, res, st = _model()
    assert "alias_of_node_id" not in _rows(export_nodal_csv(mesh, res.steps[0], tmp_path / "a.csv"))[0]
    mesh.node_alias = np.arange(len(NODES))
    assert "alias_of_node_id" not in _rows(export_nodal_csv(mesh, res.steps[0], tmp_path / "b.csv"))[0]
    mesh.node_alias[11] = 1                                               # node 12 = copy of node 2 (disconnected contact)
    rows = _rows(export_nodal_csv(mesh, res.steps[0], tmp_path / "c.csv"))
    col = _col(rows, "alias_of_node_id")
    assert col[11] == "2" and all(v == "" for k, v in enumerate(col) if k != 11)


def test_missing_results_are_empty_and_fields_named_with_units(tmp_path):
    mesh, res, st = _model()
    rows = _rows(export_nodal_csv(mesh, res.steps[0], tmp_path / "f.csv"))
    h = rows[0]
    assert "temperature_C" in h and "contact_pressure_MPa" in h and "po_prvcich" not in h
    assert float(_col(rows, "temperature_C")[-1]) == pytest.approx(33.0)
    assert set(_col(rows, "contact_pressure_MPa")) == {""}               # NaN = empty cell
    bare = StepResult("bez", None, None)
    rows = _rows(export_nodal_csv(mesh, bare, tmp_path / "e.csv"))
    for name in ("ux_mm", "umag_mm", "sxx_MPa", "von_mises_MPa", "s3_MPa"):
        assert set(_col(rows, name)) == {""}, name                        # missing U / S ≠ zeros
    assert _col(rows, "node_id")[0] == "1"


# ===================================================================== prvky
def test_elements_ids_and_connectivity_match_deck(tmp_path):
    mesh, res, st = _model()
    rows = _rows(export_elements_csv(mesh, tmp_path / "el.csv", step=res.steps[0]))
    h = rows[0]
    assert h[:4] == ["element_id", "part_id", "part", "element_type"]
    assert h[4:14] == [f"n{k}" for k in range(1, 11)]
    assert "von_mises_nodal_avg_MPa" in h and "volume_mm3" in h
    _, elems = _inp_numbers(mesh)
    assert [int(v) for v in _col(rows, "element_id")] == sorted(elems) == [1, 2]
    for r in rows[1:]:
        eset, conn = elems[int(r[0])]
        assert eset == f"P{r[1]}" and r[3] == "C3D10"
        assert [int(v) for v in r[4:14]] == conn                         # = mesh.tets + 1
    assert [int(v) for v in rows[1][4:14]] == (TETS[0] + 1).tolist()
    assert float(_col(rows, "volume_mm3")[0]) == pytest.approx(1 / 6)
    assert float(_col(rows, "von_mises_nodal_avg_MPa")[0]) == pytest.approx(50.0)
    # without a state only the topology
    assert "von_mises_nodal_avg_MPa" not in _rows(export_elements_csv(mesh, tmp_path / "el2.csv"))[0]


# ===================================================================== reakce
def test_reactions_step_index_mref_and_total(tmp_path):
    mesh, res, st = _model()
    k_res = __import__("fea_core.combinations", fromlist=["x"]).compute_combination(st, res, st.combinations[0])
    rows = _rows(export_reactions_csv(res, tmp_path / "r.csv", study=st, combinations=[(1, k_res)]))
    h = rows[0]
    assert h[:4] == ["step_index", "step_name", "selection", "constraint_kind"]
    assert h[-5:] == ["mref_x_mm", "mref_y_mm", "mref_z_mm", "result_kind", "combination_index"]
    by = [(r[0], r[1], r[2]) for r in rows[1:]]
    assert by == [("1", "Tah", "Vetknuti"), ("1", "Tah", "TOTAL"), ("2", "Tlak", "Vetknuti"), ("2", "Tlak", "Kotva_A"),
                  ("2", "Tlak", "TOTAL"), ("", "K", "Vetknuti"), ("", "K", "Kotva_A"), ("", "K", "TOTAL")]
    get = {(r[0], r[2], r[1]): dict(zip(h, r)) for r in rows[1:]}
    assert get[("1", "Vetknuti", "Tah")]["constraint_kind"] == "fixed"
    assert get[("2", "Kotva_A", "Tlak")]["constraint_kind"] == ""
    tot = get[("2", "TOTAL", "Tlak")]
    assert float(tot["Fx_N"]) == pytest.approx(5.0) and float(tot["Fz_N"]) == pytest.approx(-200.0)
    assert float(tot["Mx_Nmm"]) == pytest.approx(1.0) and float(tot["My_Nmm"]) == pytest.approx(2.0)
    assert [float(tot[k]) for k in ("mref_x_mm", "mref_y_mm", "mref_z_mm")] == [0.5, 0.5, 0.5]
    kt = get[("", "TOTAL", "K")]
    assert kt["result_kind"] == "combination" and kt["combination_index"] == "1"
    assert float(kt["Fz_N"]) == pytest.approx(100.0 - 1.5 * 200.0)
    assert get[("1", "TOTAL", "Tah")]["result_kind"] == "load_case"
    # TOTAL from the equilibrium check (each node once): a node on the common edge of two constraints is in both rows
    res.steps[1].equilibrium = {"reaction_total": [4.0, 0.0, -199.0], "reaction_moment_total": [1.0, 2.5, 0.0],
                                "imbalance": 0.0}
    res.steps[0].equilibrium = {"reaction_total": [0.0, 0.0, 100.0], "reaction_moment_total": [0.0, 0.0, 0.0],
                                "imbalance": 0.0}
    k_res = __import__("fea_core.combinations", fromlist=["x"]).compute_combination(st, res, st.combinations[0])
    rows = _rows(export_reactions_csv(res, tmp_path / "r3.csv", study=st, combinations=[(1, k_res)]))
    get = {(r[0], r[2], r[1]): dict(zip(h, r)) for r in rows[1:]}
    tot = get[("2", "TOTAL", "Tlak")]
    assert float(tot["Fx_N"]) == pytest.approx(4.0) and float(tot["Fz_N"]) == pytest.approx(-199.0)
    assert float(tot["My_Nmm"]) == pytest.approx(2.5)
    assert float(get[("", "TOTAL", "K")]["Fz_N"]) == pytest.approx(100.0 - 1.5 * 199.0)   # factors of the combination
    # without a moment point = empty cells
    res.steps[0].moment_point = None
    r1 = _rows(export_reactions_csv(res, tmp_path / "r2.csv", steps=[0]))
    assert len(r1) == 3 and r1[1][h.index("mref_x_mm")] == ""


# ===================================================================== format
def test_excel_dialect_semicolon_decimal_comma_bom(tmp_path):
    mesh, res, st = _model(names=("Deska; horní", "=Kotva"))
    p = export_nodal_csv(mesh, res.steps[0], tmp_path / "x.csv", dialect="excel")
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                               # UTF-8 s BOM
    text = raw.decode("utf-8-sig")
    assert text.splitlines()[0].startswith("node_id;part_id;part;n_parts;x_mm;")
    rows = _rows(p, "excel")
    assert _col(rows, "part")[0] == "Deska; horní"                        # in quotes because of the semicolon
    assert _col(rows, "part")[10] == "'=Kotva"                            # ochrana proti vzorci
    assert _col(rows, "von_mises_MPa")[0] == "50,000000"
    assert _col(rows, "x_mm")[1] == "1,000000"
    assert "." not in "".join(_col(rows, "sxx_MPa"))
    # standard: comma and point, without BOM
    p2 = export_nodal_csv(mesh, res.steps[0], tmp_path / "y.csv")
    assert not p2.read_bytes().startswith(b"\xef\xbb\xbf")
    assert _col(_rows(p2), "von_mises_MPa")[0] == "50.000000"
    with pytest.raises(ValueError, match="CSV"):
        dialect_of("tsv")
    # small table: empty cells, integers, sections
    q = write_csv(tmp_path / "w.csv", ["a", "b", "c"], [[1, 2.5, None], [], ["-x", float("nan"), True]], EXCEL)
    assert q.read_text(encoding="utf-8-sig").splitlines() == ["a;b;c", "1;2,5;", "", "'-x;;1"]


def test_long_format_all_steps_in_one_file(tmp_path):
    mesh, res, st = _model()
    p = export_nodal_long_csv(mesh, res.steps, tmp_path / "all.csv")
    rows = _rows(p)
    n = len(NODES)
    assert rows[0][:3] == ["step_index", "step_name", "node_id"]
    assert len(rows) == 1 + 2 * n
    assert _col(rows, "step_index") == ["1"] * n + ["2"] * n
    assert _col(rows, "step_name") == ["Tah"] * n + ["Tlak"] * n
    assert _col(rows, "node_id") == [str(i) for i in range(1, n + 1)] * 2
    T = _col(rows, "temperature_C")                                       # the field only in state 1
    assert T[0] == "20" and set(T[n:]) == {""}
    assert float(_col(rows, "uz_mm")[n]) == pytest.approx(-0.2)


# ===================================================================== file names, the whole export
def test_file_names_one_based_and_no_collision(tmp_path):
    mesh, res, st = _model()
    res.steps[0].name = "reactions"                                       # previously it overwrote the file of the reactions
    st.combinations[0].factors = {"reactions": 1.0, "Tlak": 1.5}
    w: list[str] = []
    files = export_results_csv(res, tmp_path / "a", "Zkouska", st, elements=True, warnings=w)
    assert not w
    assert [f.name for f in files] == ["Zkouska_01_reactions.csv", "Zkouska_02_Tlak.csv", "Zkouska_K01_K.csv",
                                       "Zkouska_K02_Obalka.csv", "Zkouska_elements.csv", "Zkouska_reactions.csv"]
    assert result_file_name("Zkouska", "step", 9, "a/b", "vtu") == "Zkouska_10_a_b.vtu"
    env = _rows(tmp_path / "a" / "Zkouska_K02_Obalka.csv")
    assert env[0][:3] == ["node_id", "part_id", "part"]
    assert env[0][-3:] == ["governing_combination_number", "envelope_von_mises", "governing_combination"]
    assert {r[-1] for r in env[1:]} == {"K"} and {r[-3] for r in env[1:]} == {"1"}
    rf = _rows(tmp_path / "a" / "Zkouska_reactions.csv")
    assert {r[0] for r in rf[1:]} == {"1", "2", ""}                       # states + combinations K (the envelope without reactions)
    # one file for all states, combinations separately
    files = export_results_csv(res, tmp_path / "b", "Zkouska", st, single_file=True, dialect="excel")
    assert [f.name for f in files] == ["Zkouska_nodal_all.csv", "Zkouska_K01_K.csv", "Zkouska_K02_Obalka.csv",
                                       "Zkouska_reactions.csv"]
    # selected state: the reactions only of it, the element table with the averages of its values
    files = export_results_csv(res, tmp_path / "c", "Zkouska", st, select=("step", 1), elements=True)
    assert [f.name for f in files] == ["Zkouska_02_Tlak.csv", "Zkouska_elements.csv", "Zkouska_reactions.csv"]
    assert {r[1] for r in _rows(files[2])[1:]} == {"Tlak"}
    el = _rows(files[1])
    assert float(_col(el, "von_mises_nodal_avg_MPa")[0]) == pytest.approx(100.0)
    files = export_results_csv(res, tmp_path / "d", "Zkouska", st, select=("comb", 0), reactions=True)
    assert [f.name for f in files] == ["Zkouska_K01_K.csv", "Zkouska_reactions.csv"]
    assert {r[-1] for r in _rows(files[1])[1:]} == {"1"}


def test_vtu_has_solver_ids_and_part_names(tmp_path):
    mesh, res, st = _model(names=("Díl ř", "Kotva"))
    p = export_step_vtu(mesh, res.steps[0], tmp_path / "s.vtu")
    assert np.array_equal(read_vtu_array(p, "NodeId"), np.arange(1, len(NODES) + 1))
    assert np.array_equal(read_vtu_array(p, "ElementId"), [1, 2])
    assert np.array_equal(read_vtu_array(p, "PartNumber"), [1, 2])
    assert np.array_equal(read_vtu_array(p, "PartId"), [0, 1])            # the original fields stay
    assert read_vtu_array(p, "PartNames") == ["Díl ř", "Kotva"]
    assert np.allclose(read_vtu_array(p, "temperature_C"), np.linspace(20.0, 33.0, len(NODES)), atol=1e-4)
    root = ET.parse(p).getroot()
    assert root.find("./UnstructuredGrid/FieldData") is not None          # FieldData before Piece (data set)
    files = export_results_vtu(mesh, res, tmp_path / "v", "Zkouska", st)
    assert [f.name for f in files] == ["Zkouska_01_Tah.vtu", "Zkouska_02_Tlak.vtu", "Zkouska_K01_K.vtu",
                                       "Zkouska_K02_Obalka.vtu", "Zkouska.pvd"]
    pvd = ET.parse(files[-1]).getroot().findall(".//DataSet")
    assert [(d.attrib["timestep"], d.attrib["file"]) for d in pvd] == [(str(k + 1), f.name) for k, f in
                                                                        enumerate(files[:-1])]
    env = files[3]
    assert np.array_equal(read_vtu_array(env, "Governing_Combination_Number"),
                          read_vtu_array(env, "Governing_Combination_Index") + 1)


def test_principal_vectors_have_node_indices_and_unit_directions():
    from fea_core.post_vectors import principal_vector_segments
    mesh, res, st = _model()
    S = np.zeros((len(NODES), 6))
    S[:, 0] = np.linspace(10.0, 100.0, len(NODES))
    S[:, 3] = 5.0
    seg = principal_vector_segments(mesh.nodes, S, which=1, scale=0.1, max_points=100)
    d = seg["directions"]
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    v = seg["ends"] - seg["starts"]
    assert np.allclose(np.abs(np.einsum("ij,ij->i", v / np.linalg.norm(v, axis=1)[:, None], d)), 1.0)
    assert np.allclose(mesh.nodes[seg["indices"]], 0.5 * (seg["starts"] + seg["ends"]))


# ===================================================================== MCP
def test_mcp_export_dialect_elements_single_file(tmp_path):
    pytest.importorskip("mcp")
    import json
    import anyio
    from mcp import Client
    from fea_core.mcp.server import build_server
    mesh, res, st = _model()
    srv = build_server(base_dir=tmp_path / "mcp")
    srv.fea_session.study, srv.fea_session.results = st, res
    names = lambda d: [os.path.basename(f) for f in d["exported_files"]]           # noqa: E731

    async def main():
        async with Client(srv, mode="legacy") as c:
            async def call(**a):
                r = await c.call_tool("fea_results_export", a)
                return r.is_error, r.content[0].text

            async def ok(**a):
                e, t = await call(**a)
                assert not e, t
                return json.loads(t)
            d = await ok(format="csv", output_dir=str(tmp_path / "a"), step="Tlak", dialect="excel", elements=True)
            assert names(d) == ["Zkouska_02_Tlak.csv", "Zkouska_elements.csv", "Zkouska_reactions.csv"]
            assert d["csv_dialect"] == "excel"
            assert (tmp_path / "a" / "Zkouska_02_Tlak.csv").read_bytes().startswith(b"\xef\xbb\xbf")
            assert {r[1] for r in _rows(tmp_path / "a" / "Zkouska_reactions.csv", "excel")[1:]} == {"Tlak"}
            d = await ok(format="all", output_dir=str(tmp_path / "b"), single_file=True)
            assert names(d) == ["Zkouska_01_Tah.vtu", "Zkouska_02_Tlak.vtu", "Zkouska_K01_K.vtu", "Zkouska_K02_Obalka.vtu",
                                "Zkouska.pvd", "Zkouska_nodal_all.csv", "Zkouska_K01_K.csv", "Zkouska_K02_Obalka.csv",
                                "Zkouska_reactions.csv"]
            d = await ok(format="vtu", output_dir=str(tmp_path / "c"), step="Obalka")
            assert names(d) == ["Zkouska_K02_Obalka.vtu"]
            d = await ok(format="vtu", output_dir=str(tmp_path / "c"), step=0)
            assert names(d) == ["Zkouska_01_Tah.vtu"]
            e, t = await call(format="csv", dialect="tsv")
            assert e and "dialect" in t
    anyio.run(main)


# ===================================================================== GUI (offscreen)
@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qw = pytest.importorskip("PySide6.QtWidgets")
    return qw.QApplication.instance() or qw.QApplication([])


def test_gui_table_copy_whole_table(qapp):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication, QTableWidgetItem
    from fea_gui.widgets import table, table_text
    t = table(["Díl", "σ [MPa]"])
    t.setRowCount(2)
    for r, row in enumerate((("Deska", "1,5"), ("=Kotva", "-2"))):
        for c, v in enumerate(row):
            t.setItem(r, c, QTableWidgetItem(v))
    t.selectRow(1)
    assert table_text(t) == "Díl\tσ [MPa]\nDeska\t1,5\n'=Kotva\t-2\n"
    assert t.contextMenuPolicy() == Qt.ActionsContextMenu and t.actions()[0].text() == "Copy table"
    QApplication.clipboard().clear()
    QApplication.sendEvent(t, QKeyEvent(QEvent.KeyPress, Qt.Key_C, Qt.ControlModifier, "c"))
    assert QApplication.clipboard().text() == table_text(t)
    assert [i.row() for i in t.selectionModel().selectedRows()] == [1]   # the selection does not change


def test_gui_vectors_csv_columns(tmp_path, qapp):
    import time
    from PySide6.QtWidgets import QWidget
    from fea_core.combinations import compute_combination
    from fea_gui.vectors_dialog import VectorsDialog
    mesh, res, st = _model()
    win = QWidget()
    win.settings = None                                                    # format by the language of the program
    k = compute_combination(st, res, st.combinations[0])
    dlg = VectorsDialog(win, res, 0, step=k, label="K01: K")
    t0 = time.time()
    while dlg.seg_data is None and time.time() - t0 < 30:
        qapp.processEvents()
        time.sleep(0.01)
    assert dlg.lbl_step.text() == "K01: K"
    p = dlg.write_csv(tmp_path / "v.csv")
    from fea_gui.export_dialog import csv_dialect
    rows = _rows(p, csv_dialect(win))
    assert rows[0][:12] == ["node_id", "part", "x_mm", "y_mm", "z_mm", "principal", "sigma_MPa", "nx", "ny", "nz",
                            "type", "step_name"]
    assert len(rows) == 1 + len(NODES) and set(_col(rows, "step_name")) == {"K"} and set(_col(rows, "principal")) == {"1"}
    r = next(r for r in rows[1:] if r[0] == "11")                         # node 11 only in the part Kotva
    assert r[1] == "Kotva" and float(r[6].replace(",", ".")) == pytest.approx(200.0)
    assert abs(float(r[7].replace(",", "."))) == pytest.approx(1.0)    # σ1 in the direction x
    dlg.accept()


def test_gui_export_dialog_writer(tmp_path, qapp):
    import threading
    from fea_gui.export_dialog import ExportResultsDialog, csv_dialect

    class _T:
        cancel = threading.Event()
    mesh, res, st = _model()
    opts = {"vtu": True, "nodal": True, "reactions": True, "elements": True, "single_file": False, "dialect": "excel"}
    files, warns = ExportResultsDialog._write_files(_T(), res, st, tmp_path, None, opts)
    names = [f.name for f in files]
    assert not warns and "Zkouska_01_Tah.vtu" in names and "Zkouska_01_Tah.csv" in names
    assert "Zkouska_elements.csv" in names and "Zkouska_reactions.csv" in names
    assert (tmp_path / "Zkouska_01_Tah.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    files, _ = ExportResultsDialog._write_files(_T(), res, st, tmp_path / "one", ("comb", 0),
                                                dict(opts, dialect="standard"))
    assert [f.name for f in files] == ["Zkouska_K01_K.vtu", "Zkouska_K01_K.csv", "Zkouska_elements.csv",
                                       "Zkouska_reactions.csv"]

    class _S:
        def __init__(self, v):
            self.v = v

        def value(self, key, default=""):
            return self.v

    class _W:
        settings = _S("standard")
    from fea_core.i18n import language
    assert csv_dialect(_W()) == "standard"
    _W.settings = _S("")
    assert csv_dialect(_W()) == ("excel" if language() == "cs" else "standard")
