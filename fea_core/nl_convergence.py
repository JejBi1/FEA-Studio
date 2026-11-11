"""Convergence monitor of a nonlinear ccx calculation (A7): reading .cvg, .sta and the ccx output – progress of iterations and residuals."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh import Mesh


@dataclass
class ConvergenceIteration:
    step: int
    inc: int
    att: int
    iteration: int
    contact_elements: int = 0
    resid_force_pct: float = 0.0
    corr_disp_pct: float = 0.0
    resid_flux_pct: float = 0.0
    corr_temp_pct: float = 0.0
    tot_time: float = 0.0
    step_time: float = 0.0
    inc_time: float = 0.0
    worst_force_node: int | None = None
    worst_force_dof: int | None = None
    worst_force_val: float | None = None
    worst_disp_node: int | None = None
    worst_disp_dof: int | None = None
    worst_disp_val: float | None = None


@dataclass
class ConvergenceHistory:
    iterations: list[ConvergenceIteration] = field(default_factory=list)
    increments: list[dict] = field(default_factory=list)
    converged: bool = True
    cutbacks: int = 0
    worst_node: int | None = None
    worst_node_coords: list[float] | None = None
    worst_residual_pct: float = 0.0
    status_message: str = ""

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "cutbacks": self.cutbacks,
            "worst_node": self.worst_node,
            "worst_node_coords": self.worst_node_coords,
            "worst_residual_pct": self.worst_residual_pct,
            "status_message": self.status_message,
            "total_iterations": len(self.iterations),
            "increments": self.increments,
            "iterations": [asdict(it) for it in self.iterations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConvergenceHistory":
        its = [ConvergenceIteration(**it) for it in d.get("iterations", [])]
        return cls(
            iterations=its,
            increments=list(d.get("increments", [])),
            converged=d.get("converged", True),
            cutbacks=d.get("cutbacks", 0),
            worst_node=d.get("worst_node"),
            worst_node_coords=d.get("worst_node_coords"),
            worst_residual_pct=d.get("worst_residual_pct", 0.0),
            status_message=d.get("status_message", ""),
        )


_RE_CVG_LINE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.+-EedD]+)\s+([0-9.+-EedD]+)\s+([0-9.+-EedD]+)\s+([0-9.+-EedD]+)"
)
_RE_STA_LINE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.+-EedD]+)\s+([0-9.+-EedD]+)\s+([0-9.+-EedD]+)"
)

_RE_WORST_FORCE = re.compile(
    r"largest residual force=\s*([0-9.+-EedD]+)\s+in node\s+(\d+)\s+and dof\s+(\d+)",
    re.IGNORECASE,
)
_RE_WORST_DISP = re.compile(
    r"largest correction to disp=\s*([0-9.+-EedD]+)\s+in node\s+(\d+)\s+and dof\s+(\d+)",
    re.IGNORECASE,
)


def _num(s: str) -> float:
    """Number from ccx output: decimal comma (ccx on a Windows locale prints "0,000000") and Fortran exponent D."""
    return float(s.replace(",", ".").replace("D", "E").replace("d", "e"))


def parse_cvg(text: str) -> list[ConvergenceIteration]:
    """Zpracuje obsah souboru .cvg z CalculiXu."""
    out = []
    for line in text.splitlines():
        m = _RE_CVG_LINE.match(line)
        if m:
            step, inc, att, itr, cont_el, rf, cd, rflux, ctemp = m.groups()
            out.append(
                ConvergenceIteration(
                    step=int(step),
                    inc=int(inc),
                    att=int(att),
                    iteration=int(itr),
                    contact_elements=int(cont_el),
                    resid_force_pct=_num(rf),
                    corr_disp_pct=_num(cd),
                    resid_flux_pct=_num(rflux),
                    corr_temp_pct=_num(ctemp),
                )
            )
    return out


def parse_sta(text: str) -> list[dict]:
    """Zpracuje obsah souboru .sta z CalculiXu."""
    out = []
    for line in text.splitlines():
        m = _RE_STA_LINE.match(line)
        if m:
            step, inc, att, itrs, tot_t, step_t, inc_t = m.groups()
            out.append({
                "step": int(step),
                "inc": int(inc),
                "att": int(att),
                "iterations": int(itrs),
                "tot_time": _num(tot_t),
                "step_time": _num(step_t),
                "inc_time": _num(inc_t),
            })
    return out


def parse_stdout_residuals(text: str) -> list[dict]:
    """Finds the nodes of the largest residuals in the ccx log / output."""
    residuals = []
    curr: dict = {}
    for line in text.splitlines():
        mf = _RE_WORST_FORCE.search(line)
        if mf:
            curr["worst_force_val"] = _num(mf.group(1))
            curr["worst_force_node"] = int(mf.group(2))
            curr["worst_force_dof"] = int(mf.group(3))
        md = _RE_WORST_DISP.search(line)
        if md:
            curr["worst_disp_val"] = _num(md.group(1))
            curr["worst_disp_node"] = int(md.group(2))
            curr["worst_disp_dof"] = int(md.group(3))
            residuals.append(dict(curr))
            curr = {}
    return residuals


def read_convergence_job(
    workdir: Path | str,
    job_name: str,
    mesh: Mesh | None = None,
) -> ConvergenceHistory:
    """Complete reading of the convergence history for the given ccx job."""
    wd = Path(workdir)
    if not re.fullmatch(r"[\w.-]+", str(job_name)) or str(job_name).startswith("."):
        raise ValueError(f"Invalid job name: {job_name!r}")
    cvg_path = wd / f"{job_name}.cvg"
    sta_path = wd / f"{job_name}.sta"
    log_path = wd / f"{job_name}.log"
    out_path = wd / "stdout.txt"

    iterations = []
    if cvg_path.exists():
        try:
            iterations = parse_cvg(cvg_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass

    increments = []
    if sta_path.exists():
        try:
            increments = parse_sta(sta_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass

    # Matching of times from .sta to .cvg
    sta_map = {(inc["step"], inc["inc"], inc["att"]): inc for inc in increments}
    for it in iterations:
        key = (it.step, it.inc, it.att)
        if key in sta_map:
            it.tot_time = sta_map[key]["tot_time"]
            it.step_time = sta_map[key]["step_time"]
            it.inc_time = sta_map[key]["inc_time"]

    # Matching of nodes from the log / stdout
    stdout_text = ""
    for p in (out_path, log_path):
        if p.exists():
            try:
                stdout_text += p.read_text(encoding="utf-8", errors="replace") + "\n"
            except Exception:
                pass

    if stdout_text:
        res_list = parse_stdout_residuals(stdout_text)
        for i, res in enumerate(res_list):
            if i < len(iterations):
                iterations[i].worst_force_node = res.get("worst_force_node")
                iterations[i].worst_force_dof = res.get("worst_force_dof")
                iterations[i].worst_force_val = res.get("worst_force_val")
                iterations[i].worst_disp_node = res.get("worst_disp_node")
                iterations[i].worst_disp_dof = res.get("worst_disp_dof")
                iterations[i].worst_disp_val = res.get("worst_disp_val")

    # Analysis of cutbacks and convergence
    cutbacks = sum(1 for it in iterations if it.att > 1 and it.iteration == 1)
    converged = True
    status_msg = "Analysis converged"

    # Last iteration
    worst_node = None
    worst_coords = None
    worst_res_pct = 0.0

    if iterations:
        last = iterations[-1]
        worst_res_pct = last.resid_force_pct
        worst_node = last.worst_force_node or last.worst_disp_node

        if mesh is not None and worst_node is not None:
            # Conversion of a 1-based ccx node number to coordinates
            node_idx = worst_node - 1
            if 0 <= node_idx < len(mesh.nodes):
                worst_coords = [float(x) for x in mesh.nodes[node_idx]]

    # Checking error messages in the log
    if "too many attempts" in stdout_text.lower() or "no convergence" in stdout_text.lower() and not increments:
        converged = False
        status_msg = "Analysis failed – maximum number of attempts (cutbacks) exceeded"
    elif iterations and iterations[-1].resid_force_pct > 1.0 and len(increments) == 0:
        converged = False
        status_msg = "Analysis did not converge"

    return ConvergenceHistory(
        iterations=iterations,
        increments=increments,
        converged=converged,
        cutbacks=cutbacks,
        worst_node=worst_node,
        worst_node_coords=worst_coords,
        worst_residual_pct=worst_res_pct,
        status_message=status_msg,
    )
