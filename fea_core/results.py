"""Results: displacements, stresses at the nodes, derived quantities, probes, reactions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mesh import Mesh

# order of the stress components as in ccx: SXX SYY SZZ SXY SYZ SZX


def von_mises(s: np.ndarray) -> np.ndarray:
    sx, sy, sz, txy, tyz, tzx = (s[..., i] for i in range(6))
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                   + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))


def tensor(s: np.ndarray) -> np.ndarray:
    t = np.empty(s.shape[:-1] + (3, 3))
    t[..., 0, 0], t[..., 1, 1], t[..., 2, 2] = s[..., 0], s[..., 1], s[..., 2]
    t[..., 0, 1] = t[..., 1, 0] = s[..., 3]
    t[..., 1, 2] = t[..., 2, 1] = s[..., 4]
    t[..., 2, 0] = t[..., 0, 2] = s[..., 5]
    return t


def principal(s: np.ndarray) -> np.ndarray:
    """Principal stresses sorted S1 ≥ S2 ≥ S3."""
    return np.linalg.eigvalsh(tensor(s))[..., ::-1]


def tet10_shape(xi: np.ndarray) -> np.ndarray:
    """Shape functions of C3D10 (ccx order) in the natural coordinates (r, s, t)."""
    r, s, t = xi[..., 0], xi[..., 1], xi[..., 2]
    u = 1.0 - r - s - t
    return np.stack([
        u * (2 * u - 1), r * (2 * r - 1), s * (2 * s - 1), t * (2 * t - 1),
        4 * u * r, 4 * r * s, 4 * s * u, 4 * u * t, 4 * r * t, 4 * s * t,
    ], axis=-1)


@dataclass
class StepResult:
    name: str
    U: np.ndarray                     # (N, 3) mm
    S: np.ndarray                     # (N, 6) MPa (nodal, averaged by ccx)
    reactions: dict[str, np.ndarray] = field(default_factory=dict)   # selection -> (3,) N
    ref_displacements: dict[int, np.ndarray] = field(default_factory=dict)
    equilibrium: dict = field(default_factory=dict)   # residuals of forces / moments, imbalance
    reaction_moments: dict[str, np.ndarray] = field(default_factory=dict)  # k moment_point, N·mm
    moment_point: np.ndarray | None = None
    joints: dict = field(default_factory=dict)        # forces in bolts and welds (joints.evaluate, JSON)
    fields: dict[str, np.ndarray] = field(default_factory=dict)   # further nodal fields (print_safety, print_index)
    printing: dict = field(default_factory=dict)      # 3D printing: summary per part (printing.failure, JSON)
    # remote constraints: selection -> {"point", "coupling", "dofs", "force" [N], "moment" [N·mm] ABOUT THE POINT, "u" [mm], "rot" [rad]}
    remote: dict = field(default_factory=dict)
    # frame (fea_core.beams.fem.recover, JSON): {"members": {member: {s, N, Vy, Vz, T, My, Mz, points, u,
    # sigma_max, tau_est, sigma_eq}}, "reactions": {node: [6]}, "nodes": {node: [ux uy uz rx ry rz]}, …}
    frame: dict = field(default_factory=dict)
    # contact (F8.5): {pair: {"Fn_N", "Ft_N", "area_contact_mm2", "p_max_MPa", "open_max_mm", "slip_max_mm", …}}
    contacts: dict = field(default_factory=dict)
    partial: bool = False                             # partial result (non-convergence – last increment)
    curves: dict = field(default_factory=dict)        # force–displacement of prescribed motions {target: {"t", "u", "F"}}
    _mises: np.ndarray | None = None
    _principal: np.ndarray | None = None
    _umag: np.ndarray | None = None

    @property
    def mises(self) -> np.ndarray:
        if self._mises is None:
            self._mises = von_mises(self.S)
        return self._mises

    @property
    def principal(self) -> np.ndarray:
        if self._principal is None:
            self._principal = principal(self.S)
        return self._principal

    @property
    def umag(self) -> np.ndarray:
        if self._umag is None:
            self._umag = np.linalg.norm(self.U, axis=1)
        return self._umag


@dataclass
class ThermalResult:
    """Result of a thermal state (fea_core.thermal): temperature [°C], heat flux [W/m²], fluxes of the boundary
    conditions [W] (+ = heat into the body) and the heat balance (steady state only).
    """
    name: str
    T: np.ndarray                     # (N,) °C – end of the state
    flux: np.ndarray                  # (N, 3) W/m² (nodal, averaged by ccx)
    transient: bool = False
    times: list[float] = field(default_factory=list)       # transient state: times of the frames [s]
    frames: np.ndarray | None = None                        # (F, N) temperatures of the frames
    heat: list[dict] = field(default_factory=list)          # [{"kind", "name", "Q_W"}]
    balance: dict = field(default_factory=dict)

    @property
    def flux_mag(self) -> np.ndarray:
        return np.linalg.norm(self.flux, axis=1)

    def at_time(self, t: float | None = None) -> np.ndarray:
        """Temperature at time t (linearly between the frames; None or steady state = end)."""
        if t is None or not self.transient or self.frames is None or not len(self.times):
            return self.T
        times = np.asarray(self.times, float)
        if t <= times[0]:
            return self.frames[0]
        if t >= times[-1]:
            return self.frames[-1]
        k = int(np.searchsorted(times, t)) - 1
        a = (t - times[k]) / (times[k + 1] - times[k])
        return (1 - a) * self.frames[k] + a * self.frames[k + 1]

    def summary(self, nodes: np.ndarray | None = None) -> dict:
        d = {"name": self.name, "transient": self.transient, "T_min_C": float(self.T.min()),
             "T_max_C": float(self.T.max()), "max_heat_flux_W_m2": float(self.flux_mag.max()),
             "heat_W": self.heat, "balance": self.balance}
        if nodes is not None:
            d["T_max_at"] = nodes[int(self.T.argmax())].tolist()
            d["T_min_at"] = nodes[int(self.T.argmin())].tolist()
        if self.transient:
            d["times_s"] = list(self.times)
        return d


def _validate_loaded(mesh: Mesh, steps: list, thermal: list) -> None:
    """Arrays of a result file (a .fea from outside): numeric, and nodal arrays at least as long as the mesh has nodes
    (the numba kernels index them by the node numbers of the elements without bounds checks)."""
    from .mesh import MeshError
    n = len(mesh.nodes)

    def num(a, what, width=None, nodal=True):
        if not isinstance(a, np.ndarray) or a.dtype.kind not in "iuf":
            raise MeshError(f"Invalid result file: {what} is not a numeric array")
        if width is not None and a.size and (a.ndim != 2 or a.shape[1] != width):
            raise MeshError(f"Invalid result file: {what} has a wrong shape {a.shape}")
        if nodal and n and len(a) < n:
            raise MeshError(f"Invalid result file: {what} has fewer rows than the mesh has nodes")
    for i, st in enumerate(steps):
        num(st.U, f"U of state {i + 1}", 3)
        num(st.S, f"S of state {i + 1}", None, nodal=st.S.size > 0)
        for k, v in st.fields.items():
            num(v, f"field {k!r} of state {i + 1}", nodal=False)
    for i, tr in enumerate(thermal):
        num(tr.T, f"temperature of thermal state {i + 1}")


@dataclass
class Results:
    mesh: Mesh
    steps: list[StepResult]
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    thermal: list[ThermalResult] = field(default_factory=list)     # thermal states (fea_core.thermal)
    modes: list = field(default_factory=list)                       # natural frequencies and buckling (eigen.ModeSet)
    twod: object = None                                             # 2D calculation by a section (twod.Results2D)
    shell: object = None                                            # shell model (shell.ShellResults)
    dynamics: object = None                                         # dynamika (dynamics.DynamicsResults)
    _tree: object = None
    _last_locate: tuple | None = None     # (point, element, xi) – the probe U, S of the same point does not look for the element again

    def thermal_case(self, key: int | str = 0) -> ThermalResult:
        if isinstance(key, int):
            return self.thermal[key]
        for t in self.thermal:
            if t.name == key:
                return t
        raise KeyError(key)

    def step(self, key: int | str = 0) -> StepResult:
        if isinstance(key, int):
            return self.steps[key]
        for s in self.steps:
            if s.name == key:
                return s
        raise KeyError(key)

    # ------------------------------------------------------------- sondy
    def locate(self, point) -> tuple[int, np.ndarray]:
        """Element containing the point and its natural coordinates (from the corner nodes).

        Candidates = the 64 elements with the nearest centroid; the first (by distance) with the point inside (deviation ≤ 1e-9),
        otherwise the one with the smallest deviation. All candidates are solved at once (batch np.linalg.solve – the same
        LAPACK as one by one). The last point is remembered (fea_results_probe calls locate + probe U + probe S).
        """
        p = np.asarray(point, float)
        last = self._last_locate
        if last is not None and last[0].shape == p.shape and np.array_equal(last[0], p):
            return last[1], last[2].copy()
        if self._tree is None:
            from scipy.spatial import cKDTree
            t4 = self.mesh.tets[:, :4]
            centers = self.mesh.nodes[t4].mean(axis=1)
            self._tree = cKDTree(centers)
        _, cand = self._tree.query(p, k=min(64, len(self.mesh.tets)))
        cand = np.atleast_1d(cand)
        pts = self.mesh.nodes[self.mesh.tets[cand, :4]]          # (k, 4, 3)
        a = pts[:, 0]
        J = (pts[:, 1:] - a[:, None, :]).transpose(0, 2, 1)
        try:
            xi = np.linalg.solve(J, (p - a)[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:                # degenerate candidate → one by one as before
            return self._locate_loop(p, cand)
        err = np.maximum(np.maximum(-xi.min(axis=1), xi.sum(axis=1) - 1.0), 0.0)
        inside = np.nonzero(err <= 1e-9)[0]
        j = int(inside[0]) if len(inside) else int(np.argmin(err))
        best, best_xi = int(cand[j]), xi[j].copy()
        self._last_locate = (p.copy(), best, best_xi.copy())
        return best, best_xi

    def _locate_loop(self, p: np.ndarray, cand: np.ndarray) -> tuple[int, np.ndarray]:
        best, best_xi, best_err = -1, None, np.inf
        for e in cand:
            pts = self.mesh.nodes[self.mesh.tets[e, :4]]
            a = pts[0]
            xi = np.linalg.solve((pts[1:] - a).T, p - a)
            err = max(-xi.min(), xi.sum() - 1.0, 0.0)
            if err < best_err:
                best, best_xi, best_err = int(e), xi, err
            if err <= 1e-9:
                break
        return best, best_xi

    def probe(self, point, field_name: str = "U", step: int | str = 0) -> np.ndarray:
        """Value of a field at a point (interpolation with the shape functions of the element)."""
        if field_name in ("T", "HFL"):
            th = self.thermal_case(step)
            data = th.T[:, None] if field_name == "T" else th.flux
        else:
            st = self.step(step)
            data = {"U": st.U, "S": st.S}[field_name] if field_name in ("U", "S") else st.fields[field_name]
        e, xi = self.locate(point)
        conn = self.mesh.tets[e]
        if len(conn) == 10:
            N = tet10_shape(xi)
        else:
            N = np.array([1 - xi.sum(), *xi])
        return N @ data[conn]

    # ------------------------------------------------------------- storage
    def summary(self) -> dict:
        out = {"nodes": int(len(self.mesh.nodes)), "elements": int(len(self.mesh.tets)),
               "warnings": self.warnings, "steps": []}
        for s in self.steps:
            vm = s.mises
            d = {"name": s.name, "reactions_N": {k: v.tolist() for k, v in s.reactions.items()}}
            if len(vm):
                d.update(max_displacement_mm=float(s.umag.max()), max_mises_MPa=float(vm.max()),
                         max_mises_at=self.mesh.nodes[int(vm.argmax())].tolist())
            if s.frame:
                from .beams.results import frame_summary
                d["frame"] = frame_summary(s.frame)
            if s.printing:
                d["printing"] = s.printing
            out["steps"].append(d)
        if self.thermal:
            out["thermal"] = [t.summary(self.mesh.nodes) for t in self.thermal]
        if self.modes:
            out["modes"] = [m.summary() for m in self.modes]
        if self.dynamics is not None:
            out["dynamics"] = self.dynamics.summary()
        return out

    def save(self, path: str | Path) -> None:
        """Results including the mesh (the mesh is saved next to it as <path>_mesh.npz). Without compression: the float64 of the results
        shrinks only by ~12 % and the compression cost ~1 s (197 k DOF, 3 states); load also reads older compressed files.
        """
        path = Path(path)
        self.mesh.save(path.with_name(path.stem + "_mesh.npz"))
        arrays = {}
        steps_meta = []
        for i, s in enumerate(self.steps):
            arrays[f"U_{i}"] = s.U
            arrays[f"S_{i}"] = s.S
            for k, v in s.fields.items():
                arrays[f"F_{i}_{k}"] = v
            steps_meta.append({
                "fields": list(s.fields), "printing": s.printing,
                "name": s.name,
                "reactions": {k: np.asarray(v).tolist() for k, v in s.reactions.items()},
                "reaction_moments": {k: np.asarray(v).tolist() for k, v in s.reaction_moments.items()},
                "moment_point": None if s.moment_point is None else np.asarray(s.moment_point).tolist(),
                "equilibrium": s.equilibrium,
                "joints": s.joints,
                "ref_displacements": {str(k): np.asarray(v).tolist() for k, v in s.ref_displacements.items()},
                **({"remote": s.remote} if s.remote else {}),
                **({"frame": s.frame} if s.frame else {}),
                **({"contacts": s.contacts} if s.contacts else {}),
                **({"partial": True} if s.partial else {}),
                **({"curves": s.curves} if s.curves else {}),
            })
        thermal_meta = []
        for i, t in enumerate(self.thermal):
            arrays[f"T_{i}"] = t.T
            arrays[f"HFL_{i}"] = t.flux
            if t.frames is not None:
                arrays[f"TF_{i}"] = t.frames
            thermal_meta.append({"name": t.name, "transient": t.transient, "times": list(t.times),
                                 "frames": t.frames is not None, "heat": t.heat, "balance": t.balance})
        from .eigen import save_arrays
        modes_meta = save_arrays(self.modes, arrays)
        p2d = path.with_name(path.stem + "_2d.npz")
        if self.twod is not None:
            self.twod.save(p2d)
        elif p2d.exists():
            p2d.unlink()
        psh = path.with_name(path.stem + "_shell.npz")
        if self.shell is not None:
            self.shell.save(psh)                       # + <stem>_shell.json
        else:
            for q in (psh, psh.with_suffix(".json")):
                if q.exists():
                    q.unlink()
        pdyn = path.with_name(path.stem + "_dyn.npz")
        if self.dynamics is not None:
            self.dynamics.save(pdyn)
        elif pdyn.exists():
            pdyn.unlink()
        meta = {"steps": steps_meta, "thermal": thermal_meta, "modes": modes_meta, "warnings": self.warnings, "summary": self.summary(),
                "joints": self.info.get("joints"), "printing": self.info.get("printing"),
                "backend": self.info.get("backend"), "frame": self.info.get("frame"),
                "nonlinear": self.info.get("nonlinear"), "topology": self.info.get("topology"),
                "ccx_solver": self.info.get("ccx_solver"), "timings": self.info.get("timings"),
                "twod": self.twod is not None, "shell": self.shell is not None, "dynamics": self.dynamics is not None}
        arrays["meta"] = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Results":
        path = Path(path)
        mesh = Mesh.load(path.with_name(path.stem + "_mesh.npz"))
        with np.load(path, allow_pickle=False) as z:                       # close the file at once (Windows: otherwise it cannot be overwritten)
            d = {k: z[k] for k in z.files}
        meta = json.loads(bytes(d["meta"]).decode("utf-8"))
        steps = []
        for i, sm in enumerate(meta["steps"]):
            st = StepResult(sm["name"], d[f"U_{i}"], d[f"S_{i}"],
                            {k: np.array(v) for k, v in sm["reactions"].items()})
            st.reaction_moments = {k: np.array(v) for k, v in sm.get("reaction_moments", {}).items()}
            mp = sm.get("moment_point")
            st.moment_point = None if mp is None else np.array(mp)
            st.equilibrium = sm.get("equilibrium", {})
            st.joints = sm.get("joints") or {}
            st.fields = {k: d[f"F_{i}_{k}"] for k in sm.get("fields", [])}
            st.printing = sm.get("printing") or {}
            st.ref_displacements = {int(k): np.array(v) for k, v in (sm.get("ref_displacements") or {}).items()}
            st.remote = sm.get("remote") or {}
            st.frame = sm.get("frame") or {}
            st.contacts = sm.get("contacts") or {}
            st.partial = bool(sm.get("partial", False))
            st.curves = sm.get("curves") or {}
            steps.append(st)
        info = {k: meta[k] for k in ("joints", "printing", "backend", "frame", "nonlinear", "topology", "ccx_solver",
                                         "timings")
                if meta.get(k)}
        thermal = [ThermalResult(tm["name"], d[f"T_{i}"], d[f"HFL_{i}"], tm.get("transient", False),
                                 tm.get("times", []), d[f"TF_{i}"] if tm.get("frames") else None,
                                 tm.get("heat", []), tm.get("balance", {}))
                   for i, tm in enumerate(meta.get("thermal", []))]
        from .eigen import load_arrays
        twod = None
        if meta.get("twod"):
            from .twod.results import Results2D
            twod = Results2D.load(path.with_name(path.stem + "_2d.npz"))
        shell = None
        if meta.get("shell"):
            from .shell.results import ShellResults
            shell = ShellResults.load(path.with_name(path.stem + "_shell.npz"))
        dyn = None
        if meta.get("dynamics"):
            from .dynamics.results import DynamicsResults
            dyn = DynamicsResults.load(path.with_name(path.stem + "_dyn.npz"))
        _validate_loaded(mesh, steps, thermal)
        return cls(mesh, steps, meta.get("warnings", []), info, thermal, load_arrays(meta.get("modes"), d), twod,
                   shell, dyn)
