"""Results of the 2D calculation: displacements, stresses, reactions, probes, segments, linearisation, storage and a PNG picture.

Stresses S (N, 6) in the order of ccx xx yy zz xy yz zx (2D axes): axisym SXX = σr, SYY = axial σ, SZZ = σθ (circumferential), SXY = τrz;
plane stress SZZ = 0, plane strain SZZ = ν(σx + σy). Nodal values = average over the elements (ccx, OUTPUT=2D).
Reactions [R1, R2] in the axes of the section (axisym: the radial and axial component OVER THE WHOLE CIRCUMFERENCE – a radial sum has no meaning as a resultant).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..results import principal, von_mises
from .mesh2d import SUBTRI, Mesh2D, tri6_shape
from .model import AXIS_LABELS, KIND_LABEL, Frame2D

COMP = {"sx": 0, "sy": 1, "sz": 2, "sxy": 3, "syz": 4, "szx": 5}
AXI_ALIAS = {"sr": "sx", "sa": "sy", "st": "sz", "srz": "sxy", "ur": "ux", "ua": "uy"}
FIELD_LABEL = {"mises": "von Mises", "s1": "σ1", "s2": "σ2", "s3": "σ3", "tresca": "Tresca", "umag": "|u|",
               "ux": "ux", "uy": "uy", "sx": "σx", "sy": "σy", "sz": "σz", "sxy": "τxy"}
AXI_LABEL = {"ux": "ur (radial)", "uy": "uz (axial)", "sx": "σr (radial)", "sy": "σz (axial)",
             "sz": "σθ (hoop)", "sxy": "τrz"}


@dataclass
class Results2D:
    kind: str
    nodes: np.ndarray                         # (N, 2)
    tri: np.ndarray                           # (M, 6)
    tri_part: np.ndarray                      # (M,)
    part_names: list[str]
    steps: list[dict]                         # {name, U (N, 2), S (N, 6), RF (N, 2), reactions, equilibrium}
    edges: dict[str, np.ndarray] = field(default_factory=dict)     # selection → boundary segments line3 (K, 3)
    warnings: list[str] = field(default_factory=list)
    frame: dict = field(default_factory=dict)  # origin, e1, e2, n (for the conversion to 3D)
    thickness: float = 1.0
    info: dict = field(default_factory=dict)
    mesh: Mesh2D | None = None
    _last: int = field(default=0, repr=False)

    # ------------------------------------------------------------- basic
    @property
    def axisym(self) -> bool:
        return self.kind == "axisym"

    def frame2d(self) -> Frame2D:
        f = self.frame
        return Frame2D(self.kind, *(np.asarray(f[k], float) for k in ("origin", "e1", "e2", "n")))

    def step(self, step: int | str = 0) -> dict:
        if isinstance(step, str):
            for s in self.steps:
                if s["name"] == step:
                    return s
            raise KeyError(f"State '{step}' does not exist")
        return self.steps[step]

    def mises(self, step: int | str = 0) -> np.ndarray:
        return von_mises(self.step(step)["S"])

    def field(self, name: str, step: int | str = 0) -> np.ndarray:
        """Nodal field: mises | tresca | s1 s2 s3 | sx sy sz sxy syz szx | ux uy umag | U (N, 2) | S (N, 6);
        axisym also sr, sa (axial), st (θ), srz, ur, ua.
        """
        name = AXI_ALIAS.get(name, name)
        s = self.step(step)
        if name == "U":
            return s["U"]
        if name == "S":
            return s["S"]
        if name == "mises":
            return von_mises(s["S"])
        if name in ("s1", "s2", "s3"):
            return principal(s["S"])[:, int(name[1]) - 1]
        if name == "tresca":
            p = principal(s["S"])
            return p[:, 0] - p[:, 2]
        if name in COMP:
            return s["S"][:, COMP[name]]
        if name in ("ux", "uy"):
            return s["U"][:, "xy".index(name[1])]
        if name == "umag":
            return np.linalg.norm(s["U"], axis=1)
        raise KeyError(f"Unknown field '{name}'")

    # ------------------------------------------------------------- search of a point
    def _linear_bary(self, p: np.ndarray):
        X = self.nodes[self.tri[:, :3]]
        a, b, c = X[:, 0], X[:, 1], X[:, 2]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        l2 = ((p[0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (p[1] - a[:, 1])) / det
        l3 = ((b[:, 0] - a[:, 0]) * (p[1] - a[:, 1]) - (p[0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
        return l2, l3

    def locate(self, p) -> tuple[int, np.ndarray]:
        """(element, (L2, L3)) of the point p (2D or 3D with z = 0 – the interface of paths.py). Outside the mesh = the nearest element
        (coordinates outside ⟨0, 1⟩). Curved elements: Newton on the quadratic mapping of tri6.
        """
        p = np.asarray(p, float)[:2]
        l2, l3 = self._linear_bary(p)
        m = np.minimum(np.minimum(l2, l3), 1 - l2 - l3)
        cand = np.argsort(-m)[:6]
        best, best_m, best_xi = int(cand[0]), -np.inf, np.array([l2[cand[0]], l3[cand[0]]])
        for e in cand:
            xi = np.array([l2[e], l3[e]])
            X = self.nodes[self.tri[e]]
            for _ in range(8):
                N, dN = tri6_shape(xi[0], xi[1])
                r = N @ X - p
                J = dN @ X                                  # (2, 2): d x / d (L2, L3)
                try:
                    dxi = np.linalg.solve(J.T, -r)
                except np.linalg.LinAlgError:
                    break
                xi = xi + dxi
                if np.linalg.norm(dxi) < 1e-12:
                    break
            mm = min(xi.min(), 1 - xi.sum())
            if mm > best_m:
                best, best_m, best_xi = int(e), mm, xi
            if mm >= -1e-9:
                break
        self._last = best
        return best, best_xi

    def probe(self, point, field: str = "S", step: int | str = 0):
        """Value of a field at a point (interpolation with the shape functions of tri6). field = U | S | mises | … (see field())."""
        e, xi = self.locate(point)
        N, _ = tri6_shape(xi[0], xi[1])
        vals = self.field(field, step)[self.tri[e]]
        return N @ vals

    # ------------------------------------------------------------- segments
    def _paths_view(self):
        """Interface for fea_core.paths (locate / probe with 3D points, U with three components)."""
        res = self

        class _View:
            mesh = None

            def locate(self, p):
                return res.locate(p)

            def probe(self, p, field, step=0):
                if field == "U":
                    u = res.probe(p, "U", step)
                    return np.array([u[0], u[1], 0.0])
                return res.probe(p, field, step)
        return _View()

    def line_values(self, p0, p1, field: str = "mises", step: int | str = 0, n: int = 51) -> dict:
        from ..paths import line_values
        p0, p1 = (np.r_[np.asarray(q, float)[:2], 0.0] for q in (p0, p1))
        field = AXI_ALIAS.get(field, field)
        out = line_values(self._paths_view(), p0, p1, field, step, n)
        out["points"] = [q[:2] for q in out["points"]]
        return out

    def linearize(self, p0, p1, step: int | str = 0, n: int = 41, hint=None) -> dict:
        """Linearisation of the stress on the SCL p0 → p1 (ASME VIII-2 5-A, the same method as fea_core.paths.linearize).
        The axis b of the local axes = z (for axisym the circumferential direction θ, for the planar problems perpendicular to the plane). For axisym without correction
        for the curvature (5-A.4.1.2 allows for the rotationally symetric problems also surface integrals with the weight r – we do not do it).
        """
        from ..paths import linearize

        class _M:
            tets = np.array([[0, 1, 0, 1]])           # only so that paths.mean_edge does not compute the mean of nothing
            nodes = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        view = self._paths_view()
        view.mesh = _M()
        p0, p1 = (np.r_[np.asarray(q, float)[:2], 0.0] for q in (p0, p1))
        out = linearize(view, p0, p1, step, n, hint)
        out.pop("warning", None)                  # mean_edge of an empty "mesh" – a warning according to the 2D mesh below
        h = self.mesh_mean_edge()
        t = out["length_mm"]
        if t / h < 3:
            out["warning"] = (f"Across the {t:.3g} mm thickness there are only ~{t / h:.1f} elements (mean edge {h:.3g} mm) – refine the mesh for linearization (≥ 3–4 elements)")
        out["points"] = [p0[:2].tolist(), p1[:2].tolist()]
        return out

    def mesh_mean_edge(self) -> float:
        X = self.nodes[self.tri[:, :3]]
        return float(np.mean(np.linalg.norm(X - X[:, [1, 2, 0]], axis=2)))

    # ------------------------------------------------------------- souhrn
    def summary(self) -> dict:
        out = {"kind": self.kind, "kind_label": KIND_LABEL[self.kind], "nodes": int(len(self.nodes)),
               "elements": int(len(self.tri)), "parts": list(self.part_names),
               "thickness_mm": None if self.axisym else self.thickness, "steps": [], "warnings": list(self.warnings)}
        for s in self.steps:
            vm = von_mises(s["S"])
            um = np.linalg.norm(s["U"], axis=1)
            i, j = int(np.argmax(vm)), int(np.argmax(um))
            out["steps"].append({
                "name": s["name"], "max_mises": float(vm[i]), "max_mises_at": self.nodes[i].tolist(),
                "max_disp": float(um[j]), "max_disp_at": self.nodes[j].tolist(),
                "reactions": {k: list(map(float, v)) for k, v in s["reactions"].items()},
                "imbalance": float(s["equilibrium"]["imbalance"])})
        return out

    # ------------------------------------------------------------- storage
    def save(self, path: str | Path) -> None:
        arrays = {"nodes": self.nodes, "tri": self.tri, "tri_part": self.tri_part}
        for i, s in enumerate(self.steps):
            for k in ("U", "S", "RF"):
                arrays[f"s{i}_{k}"] = s[k]
        for i, v in enumerate(self.edges.values()):
            arrays[f"edge_{i}"] = v
        meta = {"format": "fea-studio-2d", "version": 1, "kind": self.kind, "part_names": self.part_names,
                "edges": list(self.edges), "warnings": self.warnings, "frame": self.frame,
                "thickness": self.thickness, "info": self.info,
                "steps": [{"name": s["name"], "reactions": {k: list(map(float, v)) for k, v in s["reactions"].items()},
                           "equilibrium": s["equilibrium"]} for s in self.steps]}
        if self.mesh is not None:
            arrays.update({k: v for k, v in self.mesh.arrays().items() if k not in ("mesh_nodes", "mesh_tri",
                                                                                   "mesh_tri_part")})
            meta["mesh_report"] = self.mesh.report
        arrays["meta"] = np.frombuffer(json.dumps(meta, ensure_ascii=False, default=float).encode("utf-8"), np.uint8)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Results2D":
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        meta = json.loads(bytes(d["meta"]).decode("utf-8"))
        steps = []
        for i, s in enumerate(meta["steps"]):
            steps.append({"name": s["name"], "U": d[f"s{i}_U"], "S": d[f"s{i}_S"], "RF": d[f"s{i}_RF"],
                          "reactions": {k: np.asarray(v) for k, v in s["reactions"].items()},
                          "equilibrium": s["equilibrium"]})
        mesh = None
        if "mesh_seg" in d:
            dd = dict(d)
            dd.update(mesh_nodes=d["nodes"], mesh_tri=d["tri"], mesh_tri_part=d["tri_part"])
            mesh = Mesh2D.from_arrays(dd, meta["part_names"], meta.get("mesh_report", {}))
        return cls(meta["kind"], d["nodes"], d["tri"], d["tri_part"], meta["part_names"], steps,
                   {k: d[f"edge_{i}"] for i, k in enumerate(meta["edges"])}, meta["warnings"], meta["frame"],
                   float(meta["thickness"]), meta.get("info", {}), mesh)

    # ------------------------------------------------------------- picture
    def render_png(self, path: str | Path, field: str = "mises", step: int | str = 0, mirror: bool | None = None,
                   scale: float = 0.0, bands: int = 12, size: tuple[float, float] | None = None,
                   title: str | None = None, dpi: int = 100, mesh_lines: bool = False) -> Path:
        """Colour band of a field (matplotlib Agg without pyplot). mirror (axisym) = also the mirrored half (None = yes for axisym);
        scale = magnification of the deformation (0 = undeformed shape); size = (width, height) in inches (None = according to the shape of the model);
        mesh_lines = edges of the elements.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.figure import Figure
        from matplotlib.tri import Triangulation
        from ..contour import band_colors, band_edges
        key = AXI_ALIAS.get(field, field)
        vals = self.field(key, step)
        st = self.step(step)
        X = self.nodes + scale * st["U"] if scale else self.nodes
        lt = self.tri[:, SUBTRI].reshape(-1, 3)
        edges = band_edges(float(np.nanmin(vals)), float(np.nanmax(vals)), bands)
        cmap = ListedColormap(np.asarray(band_colors(bands), float) / 255.0)
        norm = BoundaryNorm(edges, cmap.N)
        mirror = self.axisym if mirror is None else (mirror and self.axisym)
        sides = (1.0, -1.0) if mirror else (1.0,)
        w_data = 2 * float(np.abs(X[:, 0]).max()) if mirror else float(np.ptp(X[:, 0]))
        h_data = float(np.ptp(X[:, 1]))
        if size is None:                              # height according to the aspect ratio of the model (legend on the right)
            size = (9.6, float(np.clip(7.6 * h_data / max(w_data, 1e-9) + 1.4, 3.6, 8.0)))
        fig = Figure(figsize=size, dpi=dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        cs = None
        for sx in sides:
            T = Triangulation(sx * X[:, 0], X[:, 1], lt)
            cs = ax.tricontourf(T, vals, levels=edges, cmap=cmap, norm=norm, extend="neither")
        # outline (boundary segments of the corner triangles)
        e = np.sort(np.vstack([lt[:, [0, 1]], lt[:, [1, 2]], lt[:, [2, 0]]]), axis=1)
        u, cnt = np.unique(e, axis=0, return_counts=True)
        bnd = u[cnt == 1]
        from matplotlib.collections import LineCollection
        for sx in sides:
            segs = np.stack([X[bnd[:, 0]], X[bnd[:, 1]]], axis=1) * np.array([sx, 1.0])
            ax.add_collection(LineCollection(segs, colors="0.15", linewidths=0.6))
        if mesh_lines:
            ue = np.unique(np.sort(np.vstack([self.tri[:, [0, 3]], self.tri[:, [3, 1]], self.tri[:, [1, 4]],
                                               self.tri[:, [4, 2]], self.tri[:, [2, 5]], self.tri[:, [5, 0]]]), axis=1),
                           axis=0)
            for sx in sides:
                segs = np.stack([X[ue[:, 0]], X[ue[:, 1]]], axis=1) * np.array([sx, 1.0])
                ax.add_collection(LineCollection(segs, colors="0.25", linewidths=0.25, alpha=0.6))
        if mirror:
            ax.axvline(0.0, color="0.3", lw=0.8, ls=(0, (6, 3)))
        i_max, i_min = int(np.nanargmax(vals)), int(np.nanargmin(vals))
        unit = "mm" if key in ("ux", "uy", "umag") else "MPa"
        xmid = 0.5 * (X[:, 0].min() + X[:, 0].max()) if not mirror else 0.0
        ymid = 0.5 * (X[:, 1].min() + X[:, 1].max())
        for k, (i, lab) in enumerate(((i_max, "max"), (i_min, "min"))):
            ax.plot(X[i, 0], X[i, 1], marker="v" if lab == "max" else "^", color="k", ms=7)
            dx = -1 if X[i, 0] > xmid else 1                 # label towards the centre of the picture
            dy = -1 if X[i, 1] > ymid else 1
            ax.annotate(f"{lab} {vals[i]:.4g} {unit}", (X[i, 0], X[i, 1]), xytext=(dx * 14, dy * (14 + 16 * k)),
                        textcoords="offset points", ha="right" if dx < 0 else "left", va="center", fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "0.3", "lw": 0.6},
                        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.5", "alpha": 0.9})
        ax.set_aspect("equal")
        ax.autoscale_view()
        xl, yl = AXIS_LABELS[self.kind]
        ax.set_xlabel(f"{xl} [mm]")
        ax.set_ylabel(f"{yl} [mm]")
        lab = (AXI_LABEL if self.axisym else {}).get(key, FIELD_LABEL.get(key, key))
        cb = fig.colorbar(cs, ax=ax, ticks=edges, shrink=0.9)
        cb.set_label(f"{lab} [{unit}]")
        cb.ax.set_yticklabels([f"{v:.4g}" for v in edges])
        extra = "" if self.axisym else f", t = {self.thickness:g} mm"
        def_txt = f", deformation ×{scale:g}" if scale else ""
        ax.set_title(title or f"{KIND_LABEL[self.kind].capitalize()}{extra} – {lab} – {st['name']}{def_txt}",
                     fontsize=11)
        fig.tight_layout()
        path = Path(path)
        fig.savefig(path, dpi=dpi)
        return path
