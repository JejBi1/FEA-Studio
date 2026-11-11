"""Mesh: running Gmsh in a separate process and the Mesh data structure.

Tet10 node order: Gmsh has middle nodes 8=(2,3), 9=(1,3); ccx C3D10 the other way round.
Mesh.tets is ALWAYS in ccx order (generate_mesh does the conversion).
"""
from __future__ import annotations

import json
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from typing import TYPE_CHECKING

from .. import config

if TYPE_CHECKING:  # OCP must not be loaded into the meshing process (see fea_core/__init__)
    from ..geometry import Part

# faces of C3D10 / C3D4 elements in ccx numbering (0-based corners): S1..S4
TET_FACES = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]])
# middle nodes of the corresponding tet10 faces (ccx order) – for tri6 in the order corners + midpoints
TET10_FACE_MIDS = np.array([[4, 5, 6], [7, 8, 4], [8, 9, 5], [9, 7, 6]])


class MeshError(RuntimeError):
    pass


@dataclass
class Mesh:
    nodes: np.ndarray                 # (N, 3) mm
    tets: np.ndarray                  # (M, 10) ccx order, 0-based node indices
    tet_part: np.ndarray              # (M,) part index
    part_names: list[str]
    surfaces: dict[str, np.ndarray] = field(default_factory=dict)  # selection -> (K, 6) tri6
    report: dict = field(default_factory=dict)
    node_alias: np.ndarray | None = None  # duplicated nodes (disconnected contact) -> original number, otherwise itself
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)  # field per element (print_shell – 3D printing)
    edges: dict[str, np.ndarray] = field(default_factory=dict)      # edge selection -> (K, 3) line3 (ends, midpoint) | (K, 2)
    points: dict[str, np.ndarray] = field(default_factory=dict)     # point selection -> (P,) mesh nodes (in the order of the selection points)
    node_data: dict[str, np.ndarray] = field(default_factory=dict)  # pole po uzlech (nominal – tolerance tisku)
    # cache of derived fields (sorted faces, Jacobians, surface) – see cached(); not saved
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def element_type(self) -> str:
        return "C3D10" if self.tets.shape[1] == 10 else "C3D4"

    def canonical(self) -> np.ndarray:
        """Node number without duplicates (to find common faces across a disconnected contact)."""
        if self.node_alias is None or len(self.node_alias) != len(self.nodes):
            return np.arange(len(self.nodes))
        return self.node_alias

    # ------------------------------------------------------------- cache
    def _fingerprint(self, fields: tuple[str, ...]) -> tuple:
        """Fingerprint of mesh fields: shape, dtype and CRC32 of the content. It also catches a change IN PLACE
        (joints.fem.open_weld_roots overwrites mesh.tets[...] = …, the tests too), not only a new array. CRC32 of 40 k tets ~2 ms.
        """
        out = []
        for name in fields:
            a = getattr(self, name)
            if a is None:
                out.append(None)
            else:
                a = np.ascontiguousarray(a)
                out.append((a.shape, a.dtype.str, zlib.crc32(a)))
        return tuple(out)

    def cached(self, key: str, fields: tuple[str, ...], compute):
        """Value from the mesh cache; recomputed when some of the `fields` change (nodes, tets,
        node_alias …). Callers must not modify the returned arrays (the same object is returned – public methods return copies).
        """
        fp = self._fingerprint(fields)
        hit = self._cache.get(key)
        if hit is not None and hit[0] == fp:
            return hit[1]
        val = compute()
        self._cache[key] = (fp, val)
        return val

    def _face_index(self):
        """Sorted keys of all element faces (from tets) for finding the faces of selections – see surface_element_faces."""
        return self.cached("face_index", ("tets",), lambda: face_index(self.tets))

    # ------------------------------------------------------------- faces
    def element_faces(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(element index, face number 1..4) for the triangles of a selection."""
        return surface_element_faces(self.tets, self.surfaces[name], self._face_index())

    def surface_nodes(self, name: str) -> np.ndarray:
        return np.unique(self.surfaces[name])

    def surface_area(self, name: str) -> float:
        return float(tri_areas(self.nodes, self.surfaces[name]).sum())

    def boundary_faces(self) -> tuple[np.ndarray, np.ndarray]:
        """Outer surface of the mesh: (tri6|tri3 with outward normal, element index). Internal interfaces of parts are left out
        (also in a disconnected contact – faces are compared by the original node numbers). Cached (returns copies).
        """
        tri, elem = self.cached("boundary_faces", ("nodes", "tets", "node_alias"), self._boundary_faces)
        return tri.copy(), elem.copy()

    def face_ids(self) -> tuple[np.ndarray, np.ndarray]:
        """Faces of the elements (corners; face j = element j // 4, face j % 4) numbered by the original node numbers
        (Mesh.canonical – the same face even across a disconnected contact): (face number (4M,), count of occurrences of the number).
        """
        faces = self.tets[:, :4][:, TET_FACES].reshape(-1, 3)
        keys = np.sort(self.canonical()[faces], axis=1)
        if len(keys) and keys.max() < (1 << 21):
            k_packed = ((keys[:, 0].astype(np.uint64) << 42)
                        | (keys[:, 1].astype(np.uint64) << 21)
                        | keys[:, 2].astype(np.uint64))
            _, inv, cnt = np.unique(k_packed, return_inverse=True, return_counts=True)
        else:
            _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        return inv.ravel(), cnt

    def _boundary_faces(self) -> tuple[np.ndarray, np.ndarray]:
        inv, cnt = self.face_ids()
        once = np.nonzero(cnt[inv] == 1)[0]
        elem, fno = once // 4, once % 4
        if self.tets.shape[1] == 10:
            corner = self.tets[elem[:, None], TET_FACES[fno]]
            mids = self.tets[elem[:, None], TET10_FACE_MIDS[fno]]
            tri = np.hstack([corner, mids])
        else:
            tri = self.tets[elem[:, None], TET_FACES[fno]]
        # outward orientation: against the opposite corner
        opp = self.nodes[self.tets[elem, np.array([3, 2, 0, 1])[fno]]]
        x = self.nodes[tri[:, :3]]
        n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        flip = np.einsum("ij,ij->i", n, x.mean(axis=1) - opp) < 0
        if np.any(flip):
            tri[flip] = tri[flip][:, [0, 2, 1, 5, 4, 3]] if tri.shape[1] == 6 else tri[flip][:, [0, 2, 1]]
        return tri, elem

    # ------------------------------------------------------------- storage
    def save(self, path: str | Path) -> None:
        """npz WITHOUT compression (float64 hardly shrinks; compression cost ~0.5 s); integer arrays on disk as
        int32 (load returns the original dtype – numba functions are compiled for int64).
        """
        arrays = {"nodes": self.nodes, "tets": self.tets, "tet_part": self.tet_part}
        if self.node_alias is not None:
            arrays["node_alias"] = self.node_alias
        for i, v in enumerate(self.surfaces.values()):
            arrays[f"surf_{i}"] = v
        for k, v in self.cell_data.items():
            arrays[f"cell_{k}"] = v
        for i, v in enumerate(self.edges.values()):
            arrays[f"edge_{i}"] = v
        for i, v in enumerate(self.points.values()):
            arrays[f"pt_{i}"] = v
        for k, v in self.node_data.items():
            arrays[f"nd_{k}"] = v
        dtypes = {}
        for k, v in list(arrays.items()):
            if k.startswith(("cell_", "nd_")):
                continue
            v = np.asarray(v)
            if v.dtype.kind in "iu" and v.dtype.itemsize > 4 and (
                    v.size == 0 or (v.min() >= np.iinfo(np.int32).min and v.max() <= np.iinfo(np.int32).max)):
                dtypes[k] = v.dtype.str
                arrays[k] = v.astype(np.int32)
        meta = {"part_names": self.part_names, "surfaces": list(self.surfaces), "report": self.report,
                "cell_data": list(self.cell_data), "dtypes": dtypes, "edges": list(self.edges),
                "points": list(self.points), "node_data": list(self.node_data)}
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Mesh":
        """Also loads older files (savez_compressed, int64 without the dtypes record)."""
        with np.load(path, allow_pickle=False) as z:                       # close the file at once (on Windows otherwise it cannot be overwritten)
            d = {k: z[k] for k in z.files}
        meta = json.loads(bytes(d["meta"]).decode("utf-8"))
        dtypes = meta.get("dtypes", {})

        def arr(key):
            a = d[key]
            return a.astype(dtypes[key]) if key in dtypes else a
        surfaces = {k: arr(f"surf_{i}") for i, k in enumerate(meta["surfaces"])}
        alias = arr("node_alias") if "node_alias" in d else None
        cells = {k: d[f"cell_{k}"] for k in meta.get("cell_data", [])}
        edges = {k: arr(f"edge_{i}") for i, k in enumerate(meta.get("edges", []))}
        points = {k: arr(f"pt_{i}") for i, k in enumerate(meta.get("points", []))}
        nd = {k: d[f"nd_{k}"] for k in meta.get("node_data", [])}
        mesh = cls(d["nodes"], arr("tets"), arr("tet_part"), meta["part_names"], surfaces, meta["report"], alias,
                   cells, edges, points, nd)
        validate_loaded(mesh)
        return mesh


def _bad(what: str):
    raise MeshError(f"Invalid mesh file: {what}")


def _check_index_array(a, n: int, what: str, ndim: int | None = None, width: tuple[int, ...] = ()) -> None:
    if isinstance(a, np.ndarray) and a.size == 0 and what not in ("tets", "tet_part"):
        return                                        # an empty selection (any dtype / shape)
    if not isinstance(a, np.ndarray) or a.dtype.kind not in "iu":
        _bad(f"{what} is not an integer array")
    if ndim is not None and a.ndim != ndim:
        _bad(f"{what} has a wrong shape {a.shape}")
    if width and a.ndim == 2 and a.shape[1] not in width:
        _bad(f"{what} has a wrong shape {a.shape}")
    if a.size and (int(a.min()) < 0 or int(a.max()) >= n):
        _bad(f"{what} refers to a non-existent node / part")


def _plain_int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
        _bad(f"{what} is not an integer")
    return int(v)


def _validate_report(rep) -> dict:
    """mesh.report from a file: the values that reach the solver input deck or index arrays must have the right type
    (a text in "index" would otherwise be written into the deck as a new card)."""
    if not isinstance(rep, dict):
        _bad("report is not a dictionary")
    contacts = rep.get("contacts")
    if contacts is not None:
        if not isinstance(contacts, list):
            _bad("report.contacts is not a list")
        for r in contacts:
            if not isinstance(r, dict):
                _bad("report.contacts item is not a dictionary")
            r["index"] = _plain_int(r.get("index"), "report.contacts.index")
            for k in ("pair", "slave", "kind"):
                if not isinstance(r.get(k), str):
                    _bad(f"report.contacts.{k} is not a text")
            if r.get("master") is not None and not isinstance(r["master"], str):
                _bad("report.contacts.master is not a text")
            for k, v in r.items():
                if k.endswith(("_mm", "_MPa")) and not (isinstance(v, (int, float)) and not isinstance(v, bool)):
                    _bad(f"report.contacts.{k} is not a number")
    ties = rep.get("ties")
    if ties is not None:
        if not isinstance(ties, list):
            _bad("report.ties is not a list")
        for t in ties:
            if not isinstance(t, dict) or not isinstance(t.get("slave"), str) or not isinstance(t.get("master"), str):
                _bad("report.ties item is not valid")
            v = t.get("tolerance_mm")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                _bad("report.ties.tolerance_mm is not a number")
    return rep


def validate_loaded(mesh: "Mesh") -> None:
    """Shapes, types and index ranges of a mesh loaded from a file (a .fea from outside) – before numba kernels and the
    solver input writer see them (an index out of range would read / write outside the arrays)."""
    nodes = mesh.nodes
    if not isinstance(nodes, np.ndarray) or nodes.dtype.kind != "f" or nodes.ndim != 2 or nodes.shape[1] != 3:
        _bad("nodes are not an (N, 3) array of numbers")
    n = len(nodes)
    if not (isinstance(mesh.part_names, list) and all(isinstance(p, str) for p in mesh.part_names)):
        _bad("part names are not texts")
    _check_index_array(mesh.tets, n, "tets", 2, (4, 10))
    m = len(mesh.tets)
    _check_index_array(mesh.tet_part, max(len(mesh.part_names), 1), "tet_part", 1)
    if len(mesh.tet_part) != m:
        _bad("tet_part does not match the elements")
    for k, v in mesh.surfaces.items():
        _check_index_array(v, n, f"surface {k!r}", 2, (3, 6))
    for k, v in mesh.edges.items():
        _check_index_array(v, n, f"edge selection {k!r}", 2, (2, 3))
    for k, v in mesh.points.items():
        _check_index_array(v, n, f"point selection {k!r}", 1)
    if mesh.node_alias is not None:
        _check_index_array(mesh.node_alias, n, "node_alias", 1)
        if len(mesh.node_alias) != n:
            _bad("node_alias does not match the nodes")
    for k, v in mesh.cell_data.items():
        if not isinstance(v, np.ndarray) or v.dtype.kind not in "iuf" or len(v) != m:
            _bad(f"element field {k!r} does not match the elements")
    for k, v in mesh.node_data.items():
        if not isinstance(v, np.ndarray) or v.dtype.kind not in "iuf" or len(v) != n:
            _bad(f"node field {k!r} does not match the nodes")
    mesh.report = _validate_report(mesh.report)


def part_union(mesh: Mesh):
    """Parts joined by common nodes (conforming mesh) – union-find over part indices.
    Returns (owner, find, union): owner = part of the node (the last one by part order, −1 = node without element).
    """
    parent = list(range(len(mesh.part_names)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    owner = np.full(len(mesh.nodes), -1, dtype=np.int64)
    for pi in range(len(mesh.part_names)):
        nodes = np.unique(mesh.tets[mesh.tet_part == pi])
        prev = owner[nodes]
        for q in np.unique(prev[prev >= 0]):
            union(pi, int(q))
        owner[nodes] = pi
    return owner, find, union


def tri_areas(nodes: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = nodes[tri[:, 0]], nodes[tri[:, 1]], nodes[tri[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _row_view(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.int64)
    return a.view([("a", np.int64), ("b", np.int64), ("c", np.int64)]).ravel()


def _pack(f: np.ndarray) -> np.ndarray:
    """Sorted triples of nodes (K, 3) < 2^21 → one uint64 key."""
    return (f[:, 0].astype(np.uint64) << 42) | (f[:, 1].astype(np.uint64) << 21) | f[:, 2].astype(np.uint64)


def face_index(tets: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray, int]:
    """Keys of all element faces, sorted stably: (packed keys?, order, sorted keys, max. node)."""
    faces = np.sort(tets[:, :4][:, TET_FACES].reshape(-1, 3), axis=1)
    fmax = int(faces.max()) if len(faces) else 0
    packed = fmax < (1 << 21)
    keys = _pack(faces) if packed else _row_view(faces)
    order = np.argsort(keys, kind="stable")
    return packed, order, keys[order], fmax


def surface_element_faces(tets: np.ndarray, tri: np.ndarray, index=None) -> tuple[np.ndarray, np.ndarray]:
    """(element, face number 1..4) of the triangles tri; index = face_index(tets) from the mesh cache (otherwise computed)."""
    packed, order, skeys, _ = face_index(tets) if index is None else index
    q_tri = np.sort(tri[:, :3], axis=1)
    if packed:
        if len(skeys) and len(q_tri) and int(q_tri.max()) >= (1 << 21):
            # node outside all elements (number ≥ 2^21 > max. element node) – it would not match anyway
            raise MeshError("Selection face does not match a face of any element")
        q = _pack(q_tri)
    else:
        q = _row_view(q_tri)
    pos = np.searchsorted(skeys, q)
    pos = np.clip(pos, 0, len(skeys) - 1)
    if np.any(skeys[pos] != q):
        raise MeshError("Selection face does not match a face of any element")
    fid = order[pos]
    return fid // 4, fid % 4 + 1


def gmsh_to_ccx_tet10(tets: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Swaps the middle nodes 8 and 9 (Gmsh → ccx) and checks it geometrically."""
    if tets.shape[1] != 10:
        return tets
    out = tets.copy()
    out[:, [8, 9]] = tets[:, [9, 8]]
    k = out[: min(len(out), 200)]
    x = nodes[k]
    # ccx: 9th node (index 8) between corners 2 and 4 (index 1, 3), 10th (index 9) between 3 and 4
    d_ok = np.linalg.norm(x[:, 8] - 0.5 * (x[:, 1] + x[:, 3]), axis=1)
    d_bad = np.linalg.norm(x[:, 8] - 0.5 * (x[:, 2] + x[:, 3]), axis=1)
    if not np.all(d_ok < d_bad):
        raise MeshError("Unexpected tet10 node order from Gmsh")
    return out


def generate_mesh(parts: list["Part"], selections: dict[str, list[dict]], settings: dict,
                  workdir: str | Path, timeout: float | None = None, imprints: list[dict] | None = None,
                  volume_sizes: list[dict] | None = None, cancel=None, edges: dict[str, list[dict]] | None = None,
                  points: dict[str, list[dict]] | None = None) -> Mesh:
    """Creates the mesh in a separate Gmsh process. selections: name -> [selector dict] (faces);
    edges: name -> [EdgeSelector dict] → Mesh.edges, points: name -> [PointSelector dict] → Mesh.points.

    imprints: circular imprints (bolt bearing faces) → selections named imprint["name"];
    volume_sizes: local element size in the whole part (welds). See mesh.worker.
    cancel: threading.Event – setting it ends the Gmsh process (MeshError "cancelled").
    """
    from ..geometry import write_brep
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    plist = []
    for i, p in enumerate(parts):
        bp = workdir / f"part_{i:03d}.brep"
        write_brep(p.shape, bp)
        plist.append({"name": p.name, "brep": str(bp)})
    job = {
        "parts": plist, "selections": selections, "settings": settings,
        "output": str(workdir / "mesh_raw.npz"), "report": str(workdir / "mesh_report.json"),
        "imprints": imprints or [], "volume_sizes": volume_sizes or [],
        "edges": edges or {}, "points": points or {},
    }
    job_path = workdir / "mesh_job.json"
    job_path.write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    for old in (job["output"], job["report"]):          # old outputs must not survive a crash of the process
        Path(old).unlink(missing_ok=True)
    out_path = workdir / "mesh_worker.log"
    rc = config.run_logged([sys.executable, "-m", "fea_core.mesh.worker", str(job_path)], out_path, cancel,
                           timeout=timeout)
    if rc is None:
        raise MeshError("Meshing cancelled by user")
    rep_path = Path(job["report"])
    if not rep_path.exists():
        tail = out_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise MeshError(f"The meshing process crashed (code {rc}) without a message:\n{tail}")
    report = json.loads(rep_path.read_text(encoding="utf-8"))
    if not report.get("ok"):
        raise MeshError(f"Meshing failed: {report.get('error')}\n{report.get('traceback', '')}")
    d = np.load(job["output"], allow_pickle=False)
    nodes = d["nodes"]
    tets = gmsh_to_ccx_tet10(d["tets"], nodes)
    surfaces = {name: d[info["key"]] for name, info in report["selections"].items()}
    edge_sets = {name: d[info["key"]] for name, info in report.get("edges", {}).items()}
    point_sets = {name: d[info["key"]] for name, info in report.get("points", {}).items()}
    return Mesh(nodes, tets, d["tet_part"], report["part_names"], surfaces, report, edges=edge_sets,
                points=point_sets)
