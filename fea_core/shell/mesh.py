"""Mesh of the shells: starting Gmsh (separate process fea_core.shell.mesh_worker) and the structure ShellMesh.

Elements tri6 in the order of ccx S6 (corners 0–2, midpoints 01, 12, 20); normal of the element n = (x1 − x0) × (x2 − x0) (right-handed)
= direction of the "upper" face of the shell (ccx: the expanded C3D15 has the nodes 1–3 at −t/2, 4–6 at +t/2 – verified by the trial
scratch/shell/exp1.py). The orientation is unified per part (neighbouring elements through a common edge of two elements).
"""
from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config
from .model import ShellError

TRI_EDGES = np.array([[0, 1, 3], [1, 2, 4], [2, 0, 5]])    # edges of tri6: end, end, midpoint


@dataclass
class ShellMesh:
    nodes: np.ndarray                 # (N, 3) mm – nodes of the mid (reference) face
    tri: np.ndarray                   # (M, 6) tri6 in the order of ccx S6, 0-based
    tri_part: np.ndarray              # (M,) index of the part
    part_names: list[str]
    thickness: np.ndarray             # (P,) thickness of the part [mm]
    offset: np.ndarray                # (P,) ccx OFFSET of the part (0 = mid face)
    report: dict = field(default_factory=dict)
    tri_surf: np.ndarray | None = None   # (M,) Gmsh area (after fragmentation)

    @property
    def element_thickness(self) -> np.ndarray:
        return self.thickness[self.tri_part]

    def normals(self) -> np.ndarray:
        """Unit normals of the elements (corners) (M, 3)."""
        x = self.nodes[self.tri[:, :3]]
        n = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def areas(self) -> np.ndarray:
        """Area of the elements (curved tri6 – Gauss 6 points)."""
        from .loads import tri6_integrals
        return tri6_integrals(self.nodes, self.tri)[1]

    def node_normals(self) -> np.ndarray:
        """Normals in the nodes – average of the normals of the elements weighted by area (corners)."""
        n = self.normals()
        x = self.nodes[self.tri[:, :3]]
        a = 0.5 * np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1)
        out = np.zeros_like(self.nodes)
        for k in range(6):
            np.add.at(out, self.tri[:, k], n * a[:, None])
        ln = np.linalg.norm(out, axis=1)
        ln[ln == 0] = 1.0
        return out / ln[:, None]

    def edge_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Edges of the elements: (line3 (K, 3) [end, end, midpoint], number of elements on the edge (K,), elements (3M,) → edge)."""
        e = self.tri[:, TRI_EDGES].reshape(-1, 3)
        key = np.sort(e[:, :2], axis=1)
        _, first, inv, cnt = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
        return e[first], cnt, inv.ravel()

    def boundary_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Boundary edges (belonging to a single element): (line3 (K, 3), element (K,))."""
        e = self.tri[:, TRI_EDGES].reshape(-1, 3)
        _, _, inv = self.edge_table()
        cnt = np.bincount(inv)
        once = np.nonzero(cnt[inv] == 1)[0]
        return e[once], once // 3

    def save(self, path) -> None:
        meta = {"part_names": self.part_names, "report": self.report}
        arr = {"nodes": self.nodes, "tri": self.tri, "tri_part": self.tri_part, "thickness": self.thickness,
               "offset": self.offset,
               "meta": np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)}
        if self.tri_surf is not None:
            arr["tri_surf"] = self.tri_surf
        np.savez(path, **arr)

    @classmethod
    def load(cls, path) -> "ShellMesh":
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta = json.loads(bytes(d["meta"]).decode("utf-8"))
        return cls(d["nodes"], d["tri"].astype(np.int64), d["tri_part"].astype(np.int64), meta["part_names"],
                   d["thickness"], d["offset"], meta["report"], d.get("tri_surf"))


def orient(mesh: ShellMesh, ref_normals: dict[int, np.ndarray] | None = None) -> int:
    """Unifies the orientation of the elements in each part (a pass over the edges shared by exactly two elements of the same part).
    Sign of a connected part: ref_normals[part] (planar plate – normal of the upper face), otherwise the majority of the area
    according to the current orientation (normal of the Gmsh face). Returns the number of the flipped elements.
    """
    tri = mesh.tri
    M = len(tri)
    e = tri[:, TRI_EDGES[:, :2]].reshape(-1, 2)            # oriented edges (3M, 2)
    key = np.sort(e, axis=1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    sinv = inv[order]
    starts = np.r_[0, np.nonzero(np.diff(sinv))[0] + 1]
    ends = np.r_[starts[1:], len(sinv)]
    nbr = [[] for _ in range(M)]
    for a, b in zip(starts, ends):
        if b - a != 2:
            continue
        i, j = order[a], order[a + 1]
        ei, ej = i // 3, j // 3
        if mesh.tri_part[ei] != mesh.tri_part[ej]:
            continue
        same = bool(e[i, 0] == e[j, 0])                    # the same direction of an edge = opposite orientation of the elements
        nbr[ei].append((ej, same))
        nbr[ej].append((ei, same))
    flip = np.zeros(M, dtype=bool)
    seen = np.zeros(M, dtype=bool)
    n = mesh.normals()
    x = mesh.nodes[tri[:, :3]]
    area = 0.5 * np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1)
    for s in range(M):
        if seen[s]:
            continue
        comp = [s]
        seen[s] = True
        q = deque([s])
        while q:
            i = q.popleft()
            for j, same in nbr[i]:
                if not seen[j]:
                    seen[j] = True
                    flip[j] = flip[i] ^ same
                    comp.append(j)
                    q.append(j)
        comp = np.asarray(comp)
        sgn = np.where(flip[comp], -1.0, 1.0)
        ref = None if ref_normals is None else ref_normals.get(int(mesh.tri_part[s]))
        if ref is not None:
            vote = float(np.sum(area[comp] * sgn * (n[comp] @ ref)))
        else:
            vote = float(np.sum(area[comp] * sgn))
        if vote < 0:
            flip[comp] = ~flip[comp]
    if flip.any():
        mesh.tri[flip] = mesh.tri[flip][:, [0, 2, 1, 5, 4, 3]]
    return int(flip.sum())


def generate_shell_mesh(sheets, size: float, workdir, curvature_points: int = 12, threads: int = 0,
                        cancel=None, timeout: float | None = None) -> ShellMesh:
    """Mesh of the mid faces (SheetInfo) in a separate Gmsh process."""
    from ..geometry import write_brep
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    plist = []
    for i, sh in enumerate(sheets):
        bp = workdir / f"shell_{i:03d}.brep"
        write_brep(sh.mid_shape(), bp)
        plist.append({"name": sh.part, "brep": str(bp)})
    job = {"parts": plist, "settings": {"size": float(size), "curvature_points": int(curvature_points),
                                        "threads": int(threads)},
           "output": str(workdir / "shell_mesh_raw.npz"), "report": str(workdir / "shell_mesh_report.json")}
    jp = workdir / "shell_mesh_job.json"
    jp.write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    for old in (job["output"], job["report"]):
        Path(old).unlink(missing_ok=True)
    log = workdir / "shell_mesh_worker.log"
    rc = config.run_logged([sys.executable, "-m", "fea_core.shell.mesh_worker", str(jp)], log, cancel,
                           timeout=timeout)
    if rc is None:
        raise ShellError("Shell meshing cancelled by user")
    rp = Path(job["report"])
    if not rp.exists():
        tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise ShellError(f"The shell meshing process crashed (code {rc}) without a message:\n{tail}")
    rep = json.loads(rp.read_text(encoding="utf-8"))
    if not rep.get("ok"):
        raise ShellError(f"Shell meshing failed: {rep.get('error')}\n{rep.get('traceback', '')}")
    with np.load(job["output"], allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    mesh = ShellMesh(d["nodes"], d["tri"].astype(np.int64), d["tri_part"].astype(np.int64),
                     [s.part for s in sheets], np.array([s.thickness for s in sheets], float),
                     np.array([s.offset for s in sheets], float), rep, d["tri_surf"].astype(np.int64))
    refs = {i: np.asarray(s.normal, float) for i, s in enumerate(sheets) if s.normal is not None}
    rep["reoriented"] = orient(mesh, refs)
    return mesh
