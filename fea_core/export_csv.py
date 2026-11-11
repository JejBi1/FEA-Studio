"""Export of results to CSV (A6): nodal fields, mesh elements and reaction tables.

The numbering is the same as in the solver input (.inp of CalculiX), all from 1 – like "Node Number" in ANSYS,
"Node / Element Label" in Abaqus or "Load Case" in HyperView:
  node_id    = number of the node in *NODE (order of the node in the mesh + 1),
  element_id = number of the element in *ELEMENT (order of the element + 1), nodes of the element n1…n10 in ccx order (C3D10),
  part_id    = number of the part (ELSET P<part_id>), part = name of the part,
  step_index = order of the load state + 1 (for linear statics the number of the *STEP in the .inp).
A node on the interface of parts with a common mesh belongs to the part with the lowest number, n_parts = number of parts sharing it;
a duplicated node of a disconnected contact (Mesh.node_alias) has alias_of_node_id = number of the original node.

Format (dialect): "standard" = comma, decimal point, UTF-8 (pandas, ParaView, scripts); "excel" = semicolon,
decimal comma, UTF-8 with BOM (a Czech Excel opens such a file directly in columns). Missing value = empty cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable
import numpy as np

if TYPE_CHECKING:
    from .mesh import Mesh
    from .results import Results, StepResult
    from .study import Study


@dataclass(frozen=True)
class CsvDialect:
    """CSV format: column separator, decimal separator and file encoding."""
    name: str
    delimiter: str
    decimal: str
    encoding: str


STANDARD = CsvDialect("standard", ",", ".", "utf-8")
EXCEL = CsvDialect("excel", ";", ",", "utf-8-sig")
DIALECTS = {d.name: d for d in (STANDARD, EXCEL)}


def dialect_of(dialect: CsvDialect | str | None = None) -> CsvDialect:
    """Dialect by name ("standard" | "excel"); None = standard (core, MCP, scripts)."""
    if dialect is None:
        return STANDARD
    if isinstance(dialect, CsvDialect):
        return dialect
    d = DIALECTS.get(str(dialect).strip().lower())
    if d is None:
        raise ValueError(f"Unknown CSV format '{dialect}' (options: standard | excel)")
    return d


# columns of further nodal fields of StepResult.fields (name with unit); other keys are written unchanged
FIELD_COLUMNS = {
    "T": "temperature_C", "stress_error": "stress_error_pct",
    "CPRESS": "contact_pressure_MPa", "COPEN": "contact_penetration_mm", "CSLIP": "contact_slip_mm",
    "CSHEAR": "contact_shear_MPa", "STATUS": "contact_status", "OPENING": "contact_opening_mm",
    "print_safety": "print_safety", "print_index": "print_index", "peeq": "peeq", "topo_density": "topo_density",
}

_BLOCK = 20_000              # rows per write (memory bounded by the block)


def text_cell(v) -> str:
    """Text in a CSV cell: if it starts with = + - @ or a tab, Excel evaluates it as a formula (name of a state / selection
    from a foreign project) → an apostrophe at the start.
    """
    s = str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def _quote(s: str, d: CsvDialect) -> str:
    """Quotes around text with a separator, a quote or a line end (RFC 4180)."""
    if d.delimiter in s or '"' in s or "\n" in s or "\r" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _text(v, d: CsvDialect) -> str:
    return _quote(text_cell(v), d)


def _num(v, fmt: str, d: CsvDialect) -> str:
    """Number in a cell: None / NaN = empty cell, bool 1 / 0, integer unchanged, real number with the template fmt."""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    f = float(v)
    if f != f:
        return ""
    s = fmt % f
    return s.replace(".", d.decimal) if d.decimal != "." else s


def _cell(v, fmt: str, d: CsvDialect) -> str:
    if isinstance(v, str):
        return _text(v, d)
    try:
        return _num(v, fmt, d)
    except (TypeError, ValueError):
        return _text(v, d)


def _open(path: Path, d: CsvDialect):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", newline="", encoding=d.encoding)


def _header(f, names, d: CsvDialect) -> None:
    f.write(d.delimiter.join(_quote(str(n), d) for n in names) + "\n")


def write_csv(path: Path | str, header: list[str] | None, rows, dialect: CsvDialect | str | None = None,
              fmt: str = "%.8g", fmts: list[str | None] | None = None) -> Path:
    """Small table row by row (progress along a line, vectors, reactions) in the dialect: None / NaN = empty cell, integer
    unchanged, real number with the template fmt (or fmts[k] per column) with the decimal separator of the dialect, text protected
    against formulas (text_cell) and in quotes if it contains the separator. An empty row ([]) separates sections.
    """
    d = dialect_of(dialect)
    path = Path(path)
    with _open(path, d) as f:
        if header:
            _header(f, header, d)
        for row in rows:
            f.write(d.delimiter.join(_cell(v, (fmts[k] if fmts and k < len(fmts) and fmts[k] else fmt), d)
                                     for k, v in enumerate(row)) + "\n")
    return path


# ------------------------------------------------------------------ column writing (large tables)
# column = (name, values, template): values np.ndarray with a `%` template, or list[str] of ready cells (template None)

def _format_block(values, fmt: str | None, lo: int, hi: int, d: CsvDialect) -> list[str]:
    """Cells of the column in rows lo:hi. A numeric column is formatted all at once (template map, replacement of "nan" by an
    empty cell and the decimal point in one string) – faster than formatting cell by cell in Python.
    """
    if fmt is None:
        return values[lo:hi]
    a = values[lo:hi]
    if not len(a):
        return []
    s = "\n".join(map(fmt.__mod__, a.tolist()))
    if "nan" in s:
        s = s.replace("nan", "")
    if d.decimal != "." and "." in s:
        s = s.replace(".", d.decimal)
    return s.split("\n")


def _write_columns(f, columns, d: CsvDialect, n: int) -> None:
    delim = d.delimiter
    for lo in range(0, n, _BLOCK):
        hi = min(n, lo + _BLOCK)
        parts = [_format_block(v, fmt, lo, hi, d) for _, v, fmt in columns]
        f.write("\n".join(map(delim.join, zip(*parts))) + "\n")


def _write_table(path: Path, columns, d: CsvDialect, n: int) -> Path:
    with _open(path, d) as f:
        _header(f, [c[0] for c in columns], d)
        _write_columns(f, columns, d, n)
    return path


# ------------------------------------------------------------------ parts of nodes
def node_parts(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """(part, number of parts) per node: index of the part with the lowest number whose element contains the node (−1 = node without an element),
    and the number of parts sharing the node (> 1 = interface of parts with a common mesh). The arrays must not be changed (mesh cache).
    """
    def compute():
        n = len(mesh.nodes)
        first = np.full(n, -1, np.int64)
        count = np.zeros(n, np.int64)
        tets = np.asarray(mesh.tets, np.int64)
        if len(tets):
            tp = np.asarray(mesh.tet_part, np.int64)
            P = int(max(len(mesh.part_names), int(tp.max()) + 1, 1))
            key = np.unique(tets.ravel() * P + np.repeat(tp, tets.shape[1]))     # sorted: node, then part
            nodes, parts = key // P, key % P
            count = np.bincount(nodes, minlength=n)[:n]
            start = np.r_[True, nodes[1:] != nodes[:-1]]
            first[nodes[start]] = parts[start]
        return first, count
    return mesh.cached("node_parts", ("nodes", "tets", "tet_part"), compute)


def _part_columns(mesh: Mesh, d: CsvDialect, with_count: bool = True) -> list:
    first, count = node_parts(mesh)
    n_names = len(mesh.part_names)
    names = [_text(p, d) for p in mesh.part_names] + [""]          # index −1 (node without an element) → empty cell
    idx = first.tolist()
    if (first < 0).any():
        pid = ("part_id", ["" if k < 0 else str(k + 1) for k in idx], None)
    else:
        pid = ("part_id", first + 1, "%d")
    cols = [pid, ("part", [names[k] if k < n_names else "" for k in idx], None)]
    if with_count:
        cols.append(("n_parts", count, "%d"))
    return cols


def _split(name: str, arr: np.ndarray, n: int, fmt: str) -> list:
    """Nodal field → columns; a vector field is split into `name_0`, `name_1`, …"""
    arr = arr.reshape(n, -1)
    if arr.shape[1] == 1:
        return [(name, arr[:, 0], fmt)]
    return [(f"{name}_{k}", arr[:, k], fmt) for k in range(arr.shape[1])]


def _extra_columns(extra_fields: dict[str, np.ndarray] | None, n_nodes: int) -> list:
    """Added nodal fields (given by the caller) → columns; a field of a length different from the number of nodes = error."""
    cols = []
    for ename, evals in (extra_fields or {}).items():
        arr = np.asarray(evals, dtype=float)
        if arr.ndim == 0 or len(arr) != n_nodes:
            raise ValueError(f"Field '{ename}' has {arr.shape} values, one value per each of the {n_nodes} nodes is expected")
        cols += _split(str(ename), arr, n_nodes, "%.6f")
    return cols


def field_column(key: str) -> str:
    """Name of the CSV column / VTU field for the key StepResult.fields (with unit, see FIELD_COLUMNS)."""
    return FIELD_COLUMNS.get(key, str(key))


def step_fields(step: StepResult | None, n: int) -> dict[str, np.ndarray]:
    """Further numeric nodal fields of the state (temperature, contact, 3D printing, error estimate, topology density …) of length n
    under the column names `field_column`; fields of another length (per element etc.) are left out.
    """
    out: dict[str, np.ndarray] = {}
    if not n:
        return out
    for key, vals in (getattr(step, "fields", None) or {}).items():
        a = np.asarray(vals)
        if a.ndim == 0 or len(a) != n or not (np.issubdtype(a.dtype, np.number) or a.dtype == np.bool_):
            continue
        out[field_column(key)] = a.astype(float)
    return out


def _nodal_columns(mesh: Mesh, step: StepResult | None, extra_fields, d: CsvDialect) -> list:
    n = len(mesh.nodes)
    cols = [("node_id", np.arange(1, n + 1), "%d"), *_part_columns(mesh, d)]
    alias = getattr(mesh, "node_alias", None)
    if alias is not None and len(alias) == n:
        alias = np.asarray(alias, np.int64)
        dup = alias != np.arange(n)
        if dup.any():
            cols.append(("alias_of_node_id", ["%d" % (a + 1) if k else "" for a, k in zip(alias.tolist(), dup.tolist())],
                         None))
    X = np.asarray(mesh.nodes, float)
    cols += [(f"{c}_mm", X[:, k], "%.6f") for k, c in enumerate("xyz")]

    empty = [""] * n
    has_u = step is not None and step.U is not None and len(step.U) == n
    has_s = step is not None and step.S is not None and len(step.S) == n
    u_names = ["ux_mm", "uy_mm", "uz_mm", "umag_mm"]
    if has_u:
        U = np.asarray(step.U, float)
        cols += [(c, v, "%.6e") for c, v in zip(u_names, [*U.T, np.asarray(step.umag, float)])]
    else:                                        # missing displacements = empty cells (not zeros)
        cols += [(c, empty, None) for c in u_names]
    s_names = ["sxx", "syy", "szz", "sxy", "syz", "szx", "von_mises", "s1", "s2", "s3"]
    if has_s:
        S = np.asarray(step.S, float)
        pr = np.asarray(step.principal, float)
        vals = [*S.T, np.asarray(step.mises, float), *pr.T]
        cols += [(f"{c}_MPa", v, "%.6f") for c, v in zip(s_names, vals)]
    else:
        cols += [(f"{c}_MPa", empty, None) for c in s_names]

    taken = {str(k) for k in (extra_fields or {})}
    for name, a in step_fields(step, n).items():
        if name not in taken:
            cols += _split(name, a, n, "%.7g")
    return cols + _extra_columns(extra_fields, n)


# ------------------------------------------------------------------ nodal values
def export_nodal_csv(
    mesh: Mesh,
    step: StepResult,
    path: Path | str,
    extra_fields: dict[str, np.ndarray] | None = None,
    dialect: CsvDialect | str | None = None,
) -> Path:
    """Nodal values of the state to CSV: node_id, part_id, part, n_parts, [alias_of_node_id], x, y, z, displacements, stress components,
    von Mises, principal stresses, further fields of the state (temperature, contact, 3D printing …, see FIELD_COLUMNS) and extra_fields.
    The stresses are nodal (averaged by ccx); missing displacements / stresses = empty cells.
    """
    d = dialect_of(dialect)
    return _write_table(Path(path), _nodal_columns(mesh, step, extra_fields, d), d, len(mesh.nodes))


def export_nodal_long_csv(
    mesh: Mesh,
    steps: list[StepResult],
    path: Path | str,
    indices: list[int] | None = None,
    extra_fields: dict[str, np.ndarray] | None = None,
    dialect: CsvDialect | str | None = None,
) -> Path:
    """All states in one file (long format, row = node × state): step_index (from 1 = number of *STEP; indices
    = own numbers of the states), step_name and then the columns as `export_nodal_csv`; a field that a state does not have = empty cells.
    """
    d = dialect_of(dialect)
    n = len(mesh.nodes)
    idx = list(indices) if indices is not None else list(range(1, len(steps) + 1))
    tables = [_nodal_columns(mesh, s, extra_fields, d) for s in steps]
    order: list[str] = []
    for t in tables:
        order += [c[0] for c in t if c[0] not in order]
    empty = [""] * n
    path = Path(path)
    with _open(path, d) as f:
        _header(f, ["step_index", "step_name", *order], d)
        for k, (s, t) in enumerate(zip(steps, tables)):
            by = {c[0]: c for c in t}
            cols = [("step_index", [str(idx[k])] * n, None), ("step_name", [_text(s.name, d)] * n, None)]
            cols += [by.get(name, (name, empty, None)) for name in order]
            _write_columns(f, cols, d, n)
    return path


def export_envelope_csv(mesh: Mesh, env, path: Path | str, dialect: CsvDialect | str | None = None) -> Path:
    """Envelope (`combinations.EnvelopeResult`) → CSV: node_id, part, coordinates, order of the governing combination in the list of the
    envelope (from 1), value of the envelope and the name of the governing combination per node.
    """
    d = dialect_of(dialect)
    n_nodes = len(mesh.nodes)
    if len(env.values) != n_nodes:
        raise ValueError(f"Envelope '{env.name}' has {len(env.values)} values, the mesh has {n_nodes} nodes")
    X = np.asarray(mesh.nodes, float)
    names = list(env.combination_names)
    pos = {c: i + 1 for i, c in enumerate(names)}
    quoted = {c: _text(c, d) for c in set(env.governing_combinations)}
    gov = list(env.governing_combinations)
    cols = [("node_id", np.arange(1, n_nodes + 1), "%d"), *_part_columns(mesh, d, with_count=False),
            *((f"{c}_mm", X[:, k], "%.6f") for k, c in enumerate("xyz")),
            ("governing_combination_number", [str(pos[c]) if c in pos else "" for c in gov], None),
            (f"envelope_{env.field_name}", np.asarray(env.values, float), "%.6f"),
            ("governing_combination", [quoted[c] for c in gov], None)]
    return _write_table(Path(path), cols, d, n_nodes)


def export_combination_csv(mesh: Mesh, result, path: Path | str, dialect: CsvDialect | str | None = None) -> Path:
    """Result of a combination (StepResult) or an envelope (EnvelopeResult) → CSV."""
    from .results import StepResult as _Step
    if isinstance(result, _Step):
        return export_nodal_csv(mesh, result, path, dialect=dialect)
    return export_envelope_csv(mesh, result, path, dialect=dialect)


# ------------------------------------------------------------------ prvky
def export_elements_csv(mesh: Mesh, path: Path | str, step: StepResult | None = None,
                        dialect: CsvDialect | str | None = None) -> Path:
    """Mesh elements to CSV: element_id (= number of the *ELEMENT in the .inp), part_id (ELSET P<part_id>), part, element_type,
    nodes n1…n10 in ccx order (node_id numbers of the nodal CSV), centroid from the corner nodes, volume (Gauss integration).
    With `step` also the averages of the NODAL (averaged) values over the nodes of the element (*_nodal_avg) – stresses at the integration
    points are not these and are not written to the CSV.
    """
    d = dialect_of(dialect)
    tets = np.asarray(mesh.tets, np.int64)
    m = len(tets)
    npe = tets.shape[1] if tets.ndim == 2 and tets.shape[1] else 10
    tp = np.asarray(mesh.tet_part, np.int64)
    names = [_text(p, d) for p in mesh.part_names]
    cols = [("element_id", np.arange(1, m + 1), "%d"), ("part_id", tp + 1, "%d"),
            ("part", [names[k] if 0 <= k < len(names) else "" for k in tp.tolist()], None),
            ("element_type", [mesh.element_type] * m, None)]
    cols += [(f"n{k + 1}", tets[:, k] + 1 if m else np.zeros(0, np.int64), "%d") for k in range(npe)]
    X = np.asarray(mesh.nodes, float)
    if m:
        from .loads import element_volumes
        c = X[tets[:, :4]].mean(axis=1)
        vol = element_volumes(mesh)
    else:
        c, vol = np.zeros((0, 3)), np.zeros(0)
    cols += [(f"{a}c_mm", c[:, k], "%.6f") for k, a in enumerate("xyz")]
    cols.append(("volume_mm3", np.asarray(vol, float), "%.9g"))
    if step is not None and m:
        n = len(X)
        if step.U is not None and len(step.U) == n:
            cols.append(("umag_nodal_avg_mm", np.asarray(step.umag, float)[tets].mean(axis=1), "%.6e"))
        if step.S is not None and len(step.S) == n:
            pr = np.asarray(step.principal, float)
            cols += [("von_mises_nodal_avg_MPa", np.asarray(step.mises, float)[tets].mean(axis=1), "%.6f"),
                     ("s1_nodal_avg_MPa", pr[:, 0][tets].mean(axis=1), "%.6f"),
                     ("s3_nodal_avg_MPa", pr[:, 2][tets].mean(axis=1), "%.6f")]
    return _write_table(Path(path), cols, d, m)


# ------------------------------------------------------------------ reakce
REACTION_HEADER = ["step_index", "step_name", "selection", "constraint_kind",
                   "Fx_N", "Fy_N", "Fz_N", "F_mag_N", "Mx_Nmm", "My_Nmm", "Mz_Nmm", "M_mag_Nmm",
                   "mref_x_mm", "mref_y_mm", "mref_z_mm", "result_kind", "combination_index"]


def _reaction_rows(index, step: StepResult, kinds: dict, kind: str, comb_index=None) -> list[list]:
    """Reaction rows of one state: selection by selection and a TOTAL row = total reaction (force and moment about the same point).

    TOTAL is taken from the equilibrium check (StepResult.equilibrium – each node once, also rigid remote constraints): a node
    shared by two constraint selections (a common edge) is in the rows of both selections, a plain sum of the rows would count it twice.
    Without equilibrium data (older results) = sum of the rows.
    """
    if not step.reactions:
        return []
    mp = step.moment_point
    mref = [None] * 3 if mp is None else [float(v) for v in np.asarray(mp, float)[:3]]
    rows = []
    F_sum, M_sum = np.zeros(3), np.zeros(3)
    for sel, rf in step.reactions.items():
        rf = np.asarray(rf, float)
        rm = np.asarray(step.reaction_moments.get(sel, np.zeros(3)), float)
        F_sum, M_sum = F_sum + rf, M_sum + rm
        rows.append([index, step.name, sel, kinds.get(sel, ""), *rf, float(np.linalg.norm(rf)),
                     *rm, float(np.linalg.norm(rm)), *mref, kind, comb_index])
    eq = step.equilibrium or {}
    if eq.get("reaction_total") is not None and eq.get("reaction_moment_total") is not None:
        F_sum = np.asarray(eq["reaction_total"], float)[:3]
        M_sum = np.asarray(eq["reaction_moment_total"], float)[:3]
    rows.append([index, step.name, "TOTAL", "", *F_sum, float(np.linalg.norm(F_sum)),
                 *M_sum, float(np.linalg.norm(M_sum)), *mref, kind, comb_index])
    return rows


def constraint_kinds(study: Study | None, step: StepResult | None = None) -> dict[str, str]:
    """Kind of constraint by the name of the selection (fixed, displacement, cylindrical, remote_support, elastic)."""
    kinds = {str(c.selection): str(getattr(c, "kind", "")) for c in (getattr(study, "constraints", None) or [])}
    for sel in (getattr(step, "remote", None) or {}):
        kinds.setdefault(sel, "remote_support")
    return kinds


def export_reactions_csv(
    results: Results,
    path: Path | str,
    study: Study | None = None,
    dialect: CsvDialect | str | None = None,
    steps: list[int] | None = None,
    combinations: list[tuple[int, StepResult]] | None = None,
) -> Path:
    """Reactions of the constraints to CSV: row = state × constraint selection, and at the end of each state a TOTAL row (sum).
    step_index = order of the state from 1 (number of *STEP), constraint_kind = kind of constraint from the study, moments Mx–Mz are about
    mref (StepResult.moment_point = centre of the mesh envelope), result_kind = load_case | combination.
    steps = indices of the states from 0 (None = all), combinations = [(order of the combination from 1, StepResult)] – they are written
    with an empty step_index and the order of the combination in combination_index (= K<NN> in the file name).
    """
    d = dialect_of(dialect)
    sel = range(len(results.steps)) if steps is None else steps
    rows: list[list] = []
    for i in sel:
        st = results.steps[i]
        rows += _reaction_rows(i + 1, st, constraint_kinds(study, st), "load_case")
    for j, st in combinations or []:
        rows += _reaction_rows(None, st, constraint_kinds(study, st), "combination", int(j))
    fmts = [None] * 4 + ["%.4f"] * 8 + ["%.6f"] * 3 + [None, None]
    return write_csv(path, REACTION_HEADER, rows, d, fmts=fmts)


# ------------------------------------------------------------------ whole export (GUI and MCP)
def export_results_csv(
    results: Results,
    out_dir: Path | str,
    base_name: str = "results",
    study: Study | None = None,
    select: tuple[str, int] | None = None,
    dialect: CsvDialect | str | None = None,
    nodal: bool = True,
    reactions: bool = True,
    elements: bool = False,
    single_file: bool = False,
    warnings: list[str] | None = None,
    check: Callable[[], None] | None = None,
    cache: dict | None = None,
) -> list[Path]:
    """Results CSV to the folder out_dir – the same file names in the GUI and MCP:
      <base>_<NN>_<state>.csv     nodal values of the state (NN = step_index from 1, two digits),
      <base>_K<NN>_<name>.csv     combination / envelope (NN = order of the combination in the study from 1),
      <base>_nodal_all.csv        all states in one file (single_file; combinations still separately),
      <base>_elements.csv         elements (element and node numbers, part, centroid, volume; with one state also the averages of values),
      <base>_reactions.csv        reactions (states and combinations; with `select` only the chosen result).
    select = None (all states and combinations) | ("step", i) | ("comb", j) (indices from 0). A combination that cannot be
    computed is left out of an export of everything, with the reason in `warnings` (a chosen combination = exception); check() is called before
    each file (cancelling the export by an exception); cache = cache of combination results (compute_combination).
    """
    from .export_vtu import result_file_name, safe_name
    from .results import StepResult as _Step
    d = dialect_of(dialect)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = safe_name(base_name)
    mesh = results.mesh
    combs = list(getattr(study, "combinations", None) or []) if study is not None else []
    cache = {} if cache is None else cache
    warns = warnings if warnings is not None else []
    tick = check or (lambda: None)
    files: list[Path] = []

    if select is None:
        step_sel, comb_sel = list(range(len(results.steps))), list(range(len(combs)))
    elif select[0] == "step":
        step_sel, comb_sel = [int(select[1])], []
    elif select[0] == "comb":
        step_sel, comb_sel = [], [int(select[1])]
    else:
        raise ValueError(f"Unknown result selection {select!r} (('step', i) | ('comb', j))")

    comb_res: dict[int, object] = {}
    if comb_sel and (nodal or reactions or elements):
        from .combinations import compute_combination
        for j in comb_sel:
            tick()
            try:
                comb_res[j] = compute_combination(study, results, combs[j], cache)
            except (ValueError, KeyError) as e:
                if select is not None:
                    raise
                warns.append(f"Combination '{combs[j].name}' was not exported: {e}")

    if nodal:
        if single_file and select is None and step_sel:
            tick()
            p = out / f"{base}_nodal_all.csv"
            export_nodal_long_csv(mesh, [results.steps[i] for i in step_sel], p, [i + 1 for i in step_sel], dialect=d)
            files.append(p)
        else:
            for i in step_sel:
                tick()
                s = results.steps[i]
                p = out / result_file_name(base, "step", i, s.name, "csv")
                export_nodal_csv(mesh, s, p, dialect=d)
                files.append(p)
        for j, r in comb_res.items():
            tick()
            p = out / result_file_name(base, "comb", j, combs[j].name, "csv")
            export_combination_csv(mesh, r, p, dialect=d)
            files.append(p)

    if elements:
        tick()
        one = None
        if select is not None:
            one = results.steps[step_sel[0]] if step_sel else comb_res.get(comb_sel[0])
            one = one if isinstance(one, _Step) else None
        p = out / f"{base}_elements.csv"
        export_elements_csv(mesh, p, step=one, dialect=d)
        files.append(p)

    if reactions:
        tick()
        cr = [(j + 1, r) for j, r in comb_res.items() if isinstance(r, _Step) and r.reactions]
        p = out / f"{base}_reactions.csv"
        export_reactions_csv(results, p, study=study, dialect=d, steps=step_sel, combinations=cr)
        files.append(p)
    return files
