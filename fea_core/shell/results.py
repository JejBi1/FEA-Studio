"""Results of the shell model: reading the expanded nodes of ccx (.frd, OUTPUT=3D), stresses on the upper / mid / lower
face, reactions, equilibrium, storage (npz + JSON), probes and a PNG picture (matplotlib Agg without pyplot).

Stress components in the order of the .frd: xx, yy, zz, xy, yz, zx [MPa], global axes. The "upper" face = the side of the normal of the element
(for a plate from a body the outward normal of the upper face, see ShellMesh); membrane stress = mid face
= (upper + lower)/2, bending = (upper − lower)/2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mesh import ShellMesh
from .model import ShellError

LAYERS = ("top", "mid", "bot", "max")
COMP = {"sxx": 0, "syy": 1, "szz": 2, "sxy": 3, "syz": 4, "szx": 5}
FIELD_LABEL = {"u": "Displacement |u| [mm]", "ux": "Displacement ux [mm]", "uy": "Displacement uy [mm]", "uz": "Displacement uz [mm]",
               "mises": "von Mises stress [MPa]", "membrane": "Membrane stress (von Mises) [MPa]",
               "bending": "Bending stress (von Mises) [MPa]", "s1": "Principal stress σ1 [MPa]",
               "s3": "Principal stress σ3 [MPa]", "thickness": "Thickness [mm]",
               **{k: f"Stress σ{k[1:]} [MPa]" for k in COMP}}
LAYER_LABEL = {"top": "top surface", "mid": "mid surface", "bot": "bottom surface", "max": "max. of surfaces"}


def von_mises(S: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, yz, zx = (S[..., k] for k in range(6))
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy ** 2 + yz ** 2 + zx ** 2))


def principal(S: np.ndarray) -> np.ndarray:
    """Principal stresses in descending order (N, 3)."""
    T = np.empty(S.shape[:-1] + (3, 3))
    T[..., 0, 0], T[..., 1, 1], T[..., 2, 2] = S[..., 0], S[..., 1], S[..., 2]
    T[..., 0, 1] = T[..., 1, 0] = S[..., 3]
    T[..., 1, 2] = T[..., 2, 1] = S[..., 4]
    T[..., 0, 2] = T[..., 2, 0] = S[..., 5]
    return np.linalg.eigvalsh(T)[..., ::-1]


# ------------------------------------------------------------------ .frd: elements and expanded nodes

def read_frd_elements(path) -> dict[int, np.ndarray]:
    """Elements of the block 3C of the .frd: element number → nodes (frd numbers) in the order of the frd."""
    els: dict[int, list] = {}
    cur = None
    inblk = False
    with open(path, encoding="latin-1") as f:
        for line in f:
            if not inblk:
                if line.lstrip().startswith("3C"):
                    inblk = True
                continue
            if line.startswith(" -3"):
                break
            if line.startswith(" -1"):
                cur = []
                els[int(line[3:13])] = cur
            elif line.startswith(" -2") and cur is not None:
                cur.extend(int(v) for v in line[3:].split())
    return {k: np.asarray(v, np.int64) for k, v in els.items()}


@dataclass
class Expansion:
    """Expanded nodes of ccx by elements: lower (M, 6), upper (M, 6) frd numbers (order of the nodes of tri6)."""
    bot: np.ndarray
    top: np.ndarray

    @classmethod
    def from_frd(cls, mesh: ShellMesh, path, coords: dict | None = None) -> "Expansion":
        els = read_frd_elements(path)
        M = len(mesh.tri)
        bot = np.zeros((M, 6), np.int64)
        top = np.zeros((M, 6), np.int64)
        for e in range(M):
            c = els.get(e + 1)
            if c is None or len(c) != 15:
                raise ShellError(f"The .frd file is missing expanded element {e + 1} (C3D15)")
            bot[e, :3], top[e, :3] = c[0:3], c[3:6]
            bot[e, 3:], top[e, 3:] = c[6:9], c[12:15]
        return cls(bot, top)


def _gather(ids: np.ndarray, vals: np.ndarray, frd_ids: np.ndarray) -> np.ndarray:
    """Values of a block of the .frd (frd_ids, vals) for the node numbers ids (any shape) → ids.shape + (k,)."""
    lut = np.full(int(max(frd_ids.max(), ids.max())) + 1, -1, np.int64)
    lut[frd_ids] = np.arange(len(frd_ids))
    pos = lut[ids]
    if np.any(pos < 0):
        raise ShellError("The .frd file is missing results for some expanded nodes")
    return vals[pos]


def _node_mean(mesh: ShellMesh, per_elem: np.ndarray) -> np.ndarray:
    """Average of the values (M, 6, k) of the elements at the nodes of the shell → (N, k)."""
    N = len(mesh.nodes)
    k = per_elem.shape[-1]
    cnt = np.bincount(mesh.tri.ravel(), minlength=N).astype(float)
    cnt[cnt == 0] = 1.0
    out = np.empty((N, k))
    for c in range(k):
        out[:, c] = np.bincount(mesh.tri.ravel(), per_elem[..., c].ravel(), minlength=N) / cnt
    return out


# ------------------------------------------------------------------ results

@dataclass
class ShellStep:
    name: str
    U: np.ndarray                     # (N, 3) displacement of the reference face [mm]
    S_top: np.ndarray                 # (N, 6) [MPa]
    S_mid: np.ndarray
    S_bot: np.ndarray
    reactions: dict[str, np.ndarray] = field(default_factory=dict)   # selection → [Fx Fy Fz Mx My Mz] (M about the ref. point)
    equilibrium: dict = field(default_factory=dict)
    applied: np.ndarray = field(default_factory=lambda: np.zeros(6))  # sum of the loads [F, M about the ref. point]

    def stress(self, layer: str) -> np.ndarray:
        if layer == "top":
            return self.S_top
        if layer == "bot":
            return self.S_bot
        if layer == "mid":
            return self.S_mid
        raise ShellError(f"Unknown layer '{layer}' (top | mid | bot)")

    @property
    def membrane(self) -> np.ndarray:
        return self.S_mid

    @property
    def bending(self) -> np.ndarray:
        return 0.5 * (self.S_top - self.S_bot)


@dataclass
class ShellResults:
    mesh: ShellMesh
    steps: list[ShellStep]
    ref_point: np.ndarray
    selections: dict[str, np.ndarray] = field(default_factory=dict)  # selection of a constraint → nodes
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    # -------------------------------------------------------------- pole
    def step(self, i: int | str = 0) -> ShellStep:
        if isinstance(i, str):
            for s in self.steps:
                if s.name == i:
                    return s
            raise KeyError(f"Case '{i}' is not in the results")
        return self.steps[i]

    @property
    def nodes(self) -> np.ndarray:
        return self.mesh.nodes

    @property
    def tri(self) -> np.ndarray:
        return self.mesh.tri

    def node_thickness(self) -> np.ndarray:
        t = np.zeros(len(self.mesh.nodes))
        for k in range(6):
            np.maximum.at(t, self.mesh.tri[:, k], self.mesh.element_thickness)
        return t

    def field(self, name: str = "mises", step: int | str = 0, layer: str = "max") -> np.ndarray:
        """Nodal field (N,): u, ux, uy, uz, mises, sxx … szx, s1, s3, membrane, bending, thickness."""
        st = self.step(step)
        if name == "u":
            return np.linalg.norm(st.U, axis=1)
        if name in ("ux", "uy", "uz"):
            return st.U[:, "xyz".index(name[1])].copy()
        if name == "thickness":
            return self.node_thickness()
        if name == "membrane":
            return von_mises(st.S_mid)
        if name == "bending":
            return von_mises(st.bending)
        if name not in ("mises", "s1", "s3") and name not in COMP:
            raise ShellError(f"Unknown field '{name}'")

        def one(S):
            if name == "mises":
                return von_mises(S)
            if name == "s1":
                return principal(S)[:, 0]
            if name == "s3":
                return principal(S)[:, 2]
            return S[:, COMP[name]].copy()
        if layer != "max":
            return one(st.stress(layer))
        a, b = one(st.S_top), one(st.S_bot)
        if name in ("mises", "s1"):
            return np.maximum(a, b)
        if name == "s3":
            return np.minimum(a, b)
        return np.where(np.abs(a) >= np.abs(b), a, b)

    def probe(self, point, name: str = "mises", step: int | str = 0, layer: str = "top") -> dict:
        """Value of a field at the point of the mid face nearest to `point` (interpolation with the shape functions of tri6)."""
        from .loads import tri6_shape
        p = np.asarray(point, float)
        f = self.field(name, step, layer)
        x = self.mesh.nodes[self.mesh.tri[:, :3]]
        c = x.mean(axis=1)
        k = min(len(c), 16)
        cand = np.argsort(np.linalg.norm(c - p, axis=1))[:k]
        best = None
        for e in cand:
            a, b, d = x[e]
            u, v = b - a, d - a
            G = np.array([[u @ u, u @ v], [u @ v, v @ v]])
            r = np.linalg.solve(G, [u @ (p - a), v @ (p - a)])
            xi, eta = np.clip(r, 0, 1)
            if xi + eta > 1:
                s = xi + eta
                xi, eta = xi / s, eta / s
            N, _ = tri6_shape(xi, eta)
            q = N @ self.mesh.nodes[self.mesh.tri[e]]
            dist = float(np.linalg.norm(q - p))
            if best is None or dist < best[0]:
                best = (dist, e, N, q)
        dist, e, N, q = best
        return {"value": float(N @ f[self.mesh.tri[e]]), "point": q.tolist(), "distance": dist,
                "element": int(e), "part": self.mesh.part_names[int(self.mesh.tri_part[e])], "field": name,
                "layer": layer}

    # -------------------------------------------------------------- souhrn
    def summary(self) -> dict:
        out = {"nodes": int(len(self.mesh.nodes)), "elements": int(len(self.mesh.tri)), "element": "S6",
               "parts": {n: {"thickness": float(self.mesh.thickness[i]), "offset": float(self.mesh.offset[i]),
                             "elements": int(np.sum(self.mesh.tri_part == i))}
                         for i, n in enumerate(self.mesh.part_names)},
               "steps": [], "warnings": list(self.warnings)}
        for k, st in enumerate(self.steps):
            u = np.linalg.norm(st.U, axis=1)
            vt, vb = von_mises(st.S_top), von_mises(st.S_bot)
            i = int(np.argmax(np.maximum(vt, vb)))
            out["steps"].append({
                "name": st.name, "u_max": float(u.max()), "u_max_at": self.mesh.nodes[int(np.argmax(u))].tolist(),
                "mises_top_max": float(vt.max()), "mises_bot_max": float(vb.max()),
                "mises_max": float(max(vt.max(), vb.max())), "mises_max_at": self.mesh.nodes[i].tolist(),
                "membrane_max": float(von_mises(st.S_mid).max()),
                "bending_max": float(von_mises(st.bending).max()),
                "reactions": {k2: [float(v) for v in r] for k2, r in st.reactions.items()},
                "imbalance": float(st.equilibrium.get("imbalance", 0.0))})
        return out

    # -------------------------------------------------------------- storage
    def save(self, path) -> Path:
        """Writes into <path>.npz (fields) + <path>.json (description). Returns the path to the .npz."""
        path = Path(path)
        base = path.with_suffix("")
        arr = {"nodes": self.mesh.nodes, "tri": self.mesh.tri, "tri_part": self.mesh.tri_part,
               "thickness": self.mesh.thickness, "offset": self.mesh.offset, "ref_point": self.ref_point}
        for k, st in enumerate(self.steps):
            arr[f"s{k}_U"], arr[f"s{k}_St"] = st.U, st.S_top
            arr[f"s{k}_Sm"], arr[f"s{k}_Sb"] = st.S_mid, st.S_bot
        for i, (k, v) in enumerate(self.selections.items()):
            arr[f"sel_{i}"] = v
        np.savez(base.with_suffix(".npz"), **arr)
        meta = {"format": "fea-studio-shell-results", "version": 1, "part_names": self.mesh.part_names,
                "mesh_report": self.mesh.report, "selections": list(self.selections), "warnings": self.warnings,
                "info": self.info,
                "steps": [{"name": st.name, "reactions": {k: list(map(float, r)) for k, r in st.reactions.items()},
                           "equilibrium": st.equilibrium, "applied": list(map(float, st.applied))}
                          for st in self.steps]}
        base.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        return base.with_suffix(".npz")

    @classmethod
    def load(cls, path) -> "ShellResults":
        base = Path(path).with_suffix("")
        meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        mesh = ShellMesh(d["nodes"], d["tri"].astype(np.int64), d["tri_part"].astype(np.int64), meta["part_names"],
                         d["thickness"], d["offset"], meta.get("mesh_report", {}))
        steps = []
        for k, sm in enumerate(meta["steps"]):
            steps.append(ShellStep(sm["name"], d[f"s{k}_U"], d[f"s{k}_St"], d[f"s{k}_Sm"], d[f"s{k}_Sb"],
                                   {n: np.asarray(r) for n, r in sm["reactions"].items()}, sm["equilibrium"],
                                   np.asarray(sm["applied"])))
        sels = {n: d[f"sel_{i}"] for i, n in enumerate(meta.get("selections", []))}
        return cls(mesh, steps, d["ref_point"], sels, meta.get("warnings", []), meta.get("info", {}))

    # -------------------------------------------------------------- picture
    def render_png(self, path, field: str = "mises", step: int | str = 0, layer: str = "max",
                   deform: float | bool | None = None, size=(1000, 750), view=(30.0, -60.0),
                   title: str | None = None, cmap: str = "turbo") -> Path:
        """Colour map of a field on the mid face (matplotlib Agg, without pyplot). deform: None/False = undeformed
        shape, True = automatic scale (5 % of the size), a number = scale of the displacements.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        st = self.step(step)
        f = self.field(field, step, layer)
        x = self.mesh.nodes.copy()
        size_m = float(np.linalg.norm(x.max(axis=0) - x.min(axis=0))) or 1.0
        scale = 0.0
        if deform is True:
            um = float(np.linalg.norm(st.U, axis=1).max())
            scale = 0.05 * size_m / um if um > 0 else 0.0
        elif deform:
            scale = float(deform)
        x = x + scale * st.U
        sub = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])
        tris = self.mesh.tri[:, sub].reshape(-1, 3)
        verts = x[tris]
        vals = f[tris].mean(axis=1)
        lo, hi = float(f.min()), float(f.max())
        if hi <= lo:
            hi = lo + 1e-12
        norm = Normalize(lo, hi)
        sm = ScalarMappable(norm, cmap)
        cols = sm.to_rgba(vals)
        # simple shading according to the normal (light from the observer)
        n = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        el, az = np.radians(view[0]), np.radians(view[1])
        eye = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        shade = 0.65 + 0.35 * np.abs(n @ eye)
        cols[:, :3] *= shade[:, None]
        fig = Figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection="3d")
        pc = Poly3DCollection(verts, facecolors=cols, edgecolors="none", linewidths=0)
        ax.add_collection3d(pc)
        lo3, hi3 = x.min(axis=0), x.max(axis=0)
        mid = 0.5 * (lo3 + hi3)
        half = 0.5 * float((hi3 - lo3).max()) or 1.0
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(*view)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        lab = FIELD_LABEL.get(field, field)
        if field not in ("u", "ux", "uy", "uz", "thickness", "membrane", "bending"):
            lab += f" – {LAYER_LABEL.get(layer, layer)}"
        fig.colorbar(sm, ax=ax, shrink=0.7, label=lab)
        ttl = title or f"{st.name}: {lab}"
        if scale:
            ttl += f" (deformation × {scale:.3g})"
        ax.set_title(ttl)
        fig.text(0.01, 0.01, f"max {hi:.4g}, min {lo:.4g}", fontsize=9)
        path = Path(path)
        fig.savefig(path)
        return path
