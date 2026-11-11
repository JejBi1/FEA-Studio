"""State of the MCP server: current study, geometry, mesh, results (one client = one session).

The tools run in working threads (the SDK calls the synchronous tools through anyio.to_thread) →
access to the state is guarded by a lock; long operations (mesh, calculation) hold the lock and the other tools
after a short wait return a comprehensible error instead of blocking.
"""
from __future__ import annotations

import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class FeaError(RuntimeError):
    """Error of a tool with a comprehensible Czech message (returned to the LLM as is_error)."""


def default_base_dir() -> Path:
    env = os.environ.get("FEA_MCP_WORKDIR")
    if env:
        return Path(env)
    from ..config import user_data_dir
    return user_data_dir() / "mcp"


def user_path(raw: str | Path) -> Path:
    """A path given by the MCP / REST client (expands ~). Network paths UNC and device paths (they start with two
    slashes) are refused: Windows would send the login hash (NTLM) to a foreign server when accessing them.
    """
    from ..pathcheck import is_unc
    s = str(raw).strip()
    if is_unc(s):
        raise FeaError(f"Network paths (UNC) are not allowed: {s}")
    return Path(s).expanduser()


def user_file(raw: str | Path, max_bytes: int = 64 << 20, what: str = "File") -> Path:
    """An existing input file given by the client (user_path), at most max_bytes large."""
    p = user_path(raw)
    if not p.is_file():
        raise FeaError(f"{what} does not exist: {p}")
    if p.stat().st_size > max_bytes:
        raise FeaError(f"{what} is too large ({p.stat().st_size >> 20} MB, at most {max_bytes >> 20} MB): {p}")
    return p


_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_name(s: str) -> str:
    s = re.sub(r"[^\w.-]+", "_", s, flags=re.UNICODE).strip("_")
    s = s[:40] or "studie"
    return f"_{s}" if s.split(".")[0].upper() in _DEVICE_NAMES else s      # CON, NUL… = Windows devices


WORKDIR_MARK = ".fea_mcp_workdir"       # automatically created working folder (may be cleaned up)
KEEP_DAYS = 30                          # default age for the cleaning (FEA_MCP_KEEP_DAYS, 0 = do not delete)
DELIVERABLES = (".docx", ".fea")        # a folder with the report / project is not deleted (outputs for the user)


def cleanup_workdirs(base_dir: str | Path | None = None, days: float | None = None) -> list[str]:
    """Deletes the automatically created working folders (mark WORKDIR_MARK) in base_dir that have not changed for days days.
    Folders given by the user (workdir=…) have no mark and are never deleted; folders with a report .docx or a project
    .fea at the top level stay. Returns the deleted folders.
    """
    import shutil
    if days is None:
        try:
            days = float(os.environ.get("FEA_MCP_KEEP_DAYS", KEEP_DAYS))
        except ValueError:
            days = KEEP_DAYS
    if not days or days <= 0:
        return []
    base = Path(base_dir) if base_dir else default_base_dir()
    limit = time.time() - days * 86400.0
    removed = []
    try:
        dirs = [d for d in base.iterdir() if d.is_dir() and (d / WORKDIR_MARK).is_file()]
    except OSError:
        return []
    for d in dirs:
        try:
            entries = list(os.scandir(d))
            if any(e.is_file() and e.name.lower().endswith(DELIVERABLES) for e in entries):
                continue
            newest = max([d.stat().st_mtime] + [e.stat().st_mtime for e in entries])
            if newest < limit:
                shutil.rmtree(d, ignore_errors=True)
                removed.append(str(d))
        except OSError:
            continue
    return removed


class Session:
    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else default_base_dir()
        self.lock = threading.RLock()
        self.busy: str | None = None          # description of the running long operation
        self.reset()

    def reset(self) -> None:
        self.study = None
        self.study_path: Path | None = None   # where the study was saved last / from where it was loaded
        self.step_path: Path | None = None
        self.workdir: Path | None = None
        self._parts = None
        self.geometry_warnings: list[str] = []   # check of the geometry when loading a STEP (read_step report)
        self._face_props: dict[int, list[dict]] = {}
        self._edge_props: dict[int, list[dict]] = {}
        self._tess = None
        self.mesh = None
        self.mesh_sig: str | None = None
        self.mesh_time: float | None = None
        self.results = None
        self.results_info: dict = {}
        self.fatigue = None
        self.joints = None
        self.variants = None              # printing.variants.VariantsRun (results of the variants of the printing)
        self.frame_check = None           # beams.assess.assess – assessment of the bars EN 1993-1-1
        self.voxel = None                 # printing.voxel.VoxelResults – voxel detail of the printing
        self.voxel_spec: dict | None = None
        self.motion = None                # motion.results.MotionResult – result of the mechanism (fea_motion_run)
        self.param_study = None           # sweep.ParametricStudyResult of the last parametric study (fea_param_study_run)
        self.log: list[str] = []

    # ---------------------------------------------------------------- lock
    @contextmanager
    def use(self, wait: float = 10.0, op: str | None = None):
        """Exclusive access to the state; op = description of the long operation (for the message to the others)."""
        if not self.lock.acquire(timeout=wait):
            raise FeaError(f"The server is busy ({self.busy or 'another operation'}) – try again in a moment.")
        prev = self.busy
        if op:
            self.busy = op
        try:
            yield self
        finally:
            self.busy = prev
            self.lock.release()

    # ---------------------------------------------------------------- studie
    def require_study(self):
        if self.study is None:
            raise FeaError("No study is open – use fea_study_new(step_path) or fea_study_open(path).")
        return self.study

    def new_workdir(self, name: str, workdir: str | Path | None = None) -> Path:
        if workdir:
            wd = user_path(workdir)
            wd.mkdir(parents=True, exist_ok=True)
            return wd
        wd = self.base_dir / f"{safe_name(name)}_{time.strftime('%Y%m%d_%H%M%S')}"
        wd.mkdir(parents=True, exist_ok=True)
        try:                                   # mark: the folder was created by the server → cleaning of the old ones (cleanup_workdirs)
            (wd / WORKDIR_MARK).write_text("FEA Studio MCP server working folder – deleted after FEA_MCP_KEEP_DAYS days without changes (not if it contains a .docx / .fea file).\n", encoding="utf-8")
        except OSError:
            pass
        return wd

    @property
    def parts(self):
        """Parts of the geometry (OCP), loaded lazily from the STEP of the study."""
        if self._parts is None:
            self.require_study()
            if self.frame_only:                    # frame system only – without the STEP geometry
                return []
            from ..geometry import legacy_step_names, read_step
            if self.step_path is None or not Path(self.step_path).exists():
                raise FeaError(f"Geometry file not found: {self.step_path}")
            report: list[str] = []
            self._parts = read_step(self.step_path, report=report, legacy_names=legacy_step_names(self.study))
            self.geometry_warnings = report
            if not self._parts:
                raise FeaError(f"The STEP file contains no body: {self.step_path}")
        return self._parts

    @property
    def frame_only(self) -> bool:
        """Study without the geometry STEP (Study.geometry == "") – the frame system only."""
        return self.study is not None and not str(self.study.geometry or "").strip()

    def part_names(self) -> list[str]:
        return [p.name for p in self.parts]

    def part_index(self, name: str) -> int:
        for i, p in enumerate(self.parts):
            if p.name == name:
                return i
        raise FeaError(f"Part '{name}' does not exist. Parts: {', '.join(self.part_names())}")

    def face_props(self, pi: int) -> list[dict]:
        """Properties of the faces of the part (cache; face_props by triangulation is slow for large assemblies)."""
        if pi not in self._face_props:
            from ..geometry import face_props, cylinder_axis
            out = []
            for f in self.parts[pi].faces():
                d = face_props(f)
                cyl = cylinder_axis(f)
                if cyl is not None:
                    d["axis_point"], d["axis_dir"], d["radius"] = cyl[0], cyl[1], float(cyl[2])
                out.append(d)
            self._face_props[pi] = out
        return self._face_props[pi]

    def faces_of(self, pi: int) -> list:
        return self.parts[pi].faces()

    def edge_props(self, pi: int) -> list[dict]:
        """Properties of the edges of the part (geometry.edge_props, the order of geometry.list_edges; cache)."""
        if pi not in self._edge_props:
            from ..geometry import edge_props, list_edges
            self._edge_props[pi] = [edge_props(e) for e in list_edges(self.parts[pi].shape)]
        return self._edge_props[pi]

    def changed(self, mesh: bool = False) -> None:
        """The study has changed → the results do not hold (as the GUI changed()); the mesh is guarded by the signature."""
        self.results = None
        self.results_info = {}
        self.fatigue = None
        self.joints = None
        self.variants = None
        self.frame_check = None
        self.voxel = None
        self.voxel_spec = None
        if mesh:
            self.mesh = None
            self.mesh_sig = None

    # ---------------------------------------------------------------- mesh
    def mesh_signature(self) -> str:
        """Signature of the inputs of the meshing (Gmsh) – a match = the stored mesh can be used for the calculation.

        The thermal states, the boundary conditions (only their selections of faces – they are in "selections"), the materials and the settings of the
        3D printing do not affect the mesh: the assignment of the shell and the infill to the elements is recomputed on the finished mesh
        (printing.region.ensure) → a change of the perimeters or the infill does not re-mesh.
        """
        from ..signature import mesh_signature
        return mesh_signature(self.study)

    def valid_mesh(self):
        if self.mesh is not None and self.study is not None and self.mesh_sig == self.mesh_signature():
            return self.mesh
        return None

    def require_results(self):
        if self.results is None:
            raise FeaError("No results available – run fea_solve_run() (results are discarded when the study changes).")
        return self.results

    def step_index(self, step) -> int:
        res = self.require_results()
        if not res.steps:
            th = ", ".join(f"'{t.name}'" for t in res.thermal)
            modes = getattr(res, "modes", None)
            if getattr(res, "twod", None) is not None or getattr(res, "shell", None) is not None:
                from .diagnostics import no_solid_hint
                raise FeaError(no_solid_hint(res))
            raise FeaError("The results contain no mechanical load case"
                           + (f" (thermal cases only {th}) – for temperature and heat flux specify thermal_case, or field='temperature' / 'heat_flux'" if th else "")
                           + (" – natural frequency / buckling: fea_results_modes, mode shape image fea_results_render_mode"
                              if modes else ""))
        if isinstance(step, str):
            if step.strip().lstrip("-").isdigit():
                step = int(step)
            else:
                for i, s in enumerate(res.steps):
                    if s.name == step:
                        return i
                raise FeaError(f"Load case '{step}' is not in the results. Cases: "
                               + ", ".join(f"{i}: {s.name}" for i, s in enumerate(res.steps)))
        i = int(step)
        if not (-len(res.steps) <= i < len(res.steps)):
            raise FeaError(f"State index {i} out of range 0..{len(res.steps) - 1}")
        return i % len(res.steps)

    def thermal_index(self, case) -> int:
        """Index of the thermal state in the results (name or number)."""
        res = self.require_results()
        names = [t.name for t in res.thermal]
        if not names:
            raise FeaError("Results contain no thermal case (fea_thermal_case_create + boundary conditions, then fea_solve_run)")
        if isinstance(case, str):
            if case in names:
                return names.index(case)
            if not case.strip().lstrip("-").isdigit():
                raise FeaError(f"Thermal state '{case}' is not in the results. Thermal states: "
                               + ", ".join(f"{i}: {n}" for i, n in enumerate(names)))
        i = int(case)
        if not (-len(names) <= i < len(names)):
            raise FeaError(f"Thermal case index {i} out of range 0..{len(names) - 1}")
        return i % len(names)

    def node_re(self) -> np.ndarray:
        """Yield strength at the nodes (a node shared by parts takes the smaller), inf = a material without Re."""
        m = self.results.mesh
        re_node = np.full(len(m.nodes), np.inf)
        for pi, pn in enumerate(m.part_names):
            try:
                mat = self.study.material_of(pn)
                re = np.inf if getattr(mat, "is_ortho", False) else (mat.Re or np.inf)   # orthotropic: Re does not hold
            except KeyError:
                re = np.inf
            np.minimum.at(re_node, m.tets[m.tet_part == pi].ravel(), re)
        return re_node

    def add_log(self, line: str) -> None:
        self.log.append(line)
        del self.log[:-200]
