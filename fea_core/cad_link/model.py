"""Link of the study to the CAD and the history of the iterations of the geometry (Study.cad_link, written only when it is used).

The identity of the faces and parts lives in the CAD (Fusion tokens in the table of faces of the metadata), the settings of the calculation in FEA Studio.
The geometry update = new selectors of the selections (fea_core.cad_link.update) – everything else (mesh, solver, MCP, GUI,
report) stays unchanged. Iteration = version of the geometry; for each one the summary of the results of the last calculation is stored
(KPI), not the whole results.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields


def _known(klass, d: dict | None):
    names = {f.name for f in fields(klass)}
    return klass(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class Iteration:
    """One version of the geometry: where it came from, how the selections and the results of the last calculation were remapped onto it."""
    n: int
    time: str = ""                        # ISO, local time
    trigger: str = "step"                 # cad (CAD add-in) | step (manual replacement of the STEP) | initial
    source: str = ""                      # CAD / file
    geometry: str = ""                    # name of the STEP file (without the path – paths are not written into the .fea)
    digest: str = ""                      # otisk geometrie (digest.step_digest)
    params: dict = field(default_factory=dict)      # parameters of the CAD {name: {"expression", "value", "unit"}}
    remap: dict = field(default_factory=dict)       # numbers of the selections by state {"same": n, "cad": n, …}
    changes: list = field(default_factory=list)     # short descriptions of the changes of parts / selections (for the table of iterations)
    kpi: dict = field(default_factory=dict)         # summary of the results (history.kpis), empty = no calculation
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Iteration":
        return _known(cls, d)


@dataclass
class CadLink:
    """Where the geometry of the study comes from and how it evolved.

    document_id = permanent id of the document in the CAD (Fusion dataFile.id); used to verify that the update belongs to this
    study, and to find the project .fea (local registry, `registry.py`). Absolute paths are not written.
    """
    source: str = ""                      # "Autodesk Fusion" | "STEP" | the name of the CAD from the metadata
    document: str = ""                    # name of the document in the CAD
    document_id: str = ""
    revision: int = 0                     # number of the geometry updates
    digest: str = ""                      # fingerprint of the current geometry
    params: dict = field(default_factory=dict)      # parameters of the CAD of the current geometry
    iterations: list[Iteration] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["iterations"] = [it.to_dict() for it in self.iterations]
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "CadLink | None":
        if not d:
            return None
        link = _known(cls, {k: v for k, v in d.items() if k != "iterations"})
        link.iterations = [Iteration.from_dict(x) for x in d.get("iterations") or [] if isinstance(x, dict)]
        return link

    def current(self) -> Iteration | None:
        return self.iterations[-1] if self.iterations else None
