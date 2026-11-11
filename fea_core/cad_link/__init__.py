"""Geometry update that keeps the settings of the study and the history of the iterations (link to CAD).

Procedure: `update.plan_update` (proposal of the remapping of parts, selections and points) → check by the user → `update.apply_report`
(new study) → calculation → `history.record` (KPI of the iteration). It works for any STEP (exported again from the CAD);
with the CAD add-in (table of faces with Fusion tokens) the faces are matched by their identity in the CAD.

Modules: model (CadLink, Iteration), faces (properties of faces), facetable (table of faces from the CAD), parts (matching of
parts), remap (selections), refs (renaming of parts in the study), update (proposal and use), history (KPI), digest
(fingerprint of the geometry), registry (CAD document → project .fea).
"""
from .model import CadLink, Iteration  # noqa: F401
