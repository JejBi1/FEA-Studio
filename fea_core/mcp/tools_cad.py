"""Tools: update of the geometry keeping the settings of the study (a new STEP from the CAD) and the iterations of the geometry."""
from __future__ import annotations


from .common import out
from .session import FeaError, Session, user_path

ACCEPT = {"certain": frozenset({"same", "cad"}), "auto": frozenset({"same", "cad", "probable", "split"})}


def replace_geometry(sess: Session, step_path: str, accept: str = "auto", dry_run: bool = False,
                     trigger: str = "step") -> dict:
    """Body of fea_geometry_replace (the caller holds the lock of the session): proposal of the remapping and its use."""
    from .. import cad_import
    from ..cad_link.update import apply_report, plan_update
    from ..geometry import read_step
    st = sess.require_study()
    if sess.frame_only:
        raise FeaError("The study has no STEP geometry (a frame system only) – there is nothing to replace")
    if accept not in ACCEPT:
        raise FeaError("accept must be 'auto' (certain + probable matches and split faces) or 'certain' (only unchanged ones, by CAD identity)")
    p = user_path(step_path)
    if not p.is_file():
        raise FeaError(f"File does not exist: {p}")
    report: list[str] = []
    if p.suffix.lower() not in cad_import.STEP_SUFFIXES:
        from ..native_cad import is_native
        if not is_native(p):
            raise FeaError("Expected a STEP file (.step / .stp), an Inventor part or assembly (.ipt, .iam) or a Fusion design (.f3d, .f3z)")
        from .tools_study import native_step
        p = native_step(p, report)
    try:
        new_parts = read_step(p.resolve(), report=report)
    except Exception as e:
        raise FeaError(f"Cannot read STEP: {e}") from e
    if not new_parts:
        raise FeaError(f"The STEP file contains no body: {p}")
    warn: list[str] = []
    old_meta = cad_import.load_sidecar(sess.step_path, warn) if sess.step_path else None
    new_meta = cad_import.load_sidecar(p, warn)
    rep = plan_update(st, sess.parts, new_parts, old_meta, new_meta)
    d = {"report": rep.to_dict(), "accept": sorted(ACCEPT[accept]),
         "not_remapped": rep.blocking(ACCEPT[accept]), "warnings": warn + report}
    if dry_run:
        d["applied"] = False
        d["next"] = "fea_geometry_replace(dry_run=False) applies the proposal"
        return d
    new = apply_report(st, rep, ACCEPT[accept], new_step=str(p.resolve()), new_parts=new_parts, new_meta=new_meta,
                       trigger=trigger, results=sess.results, old_parts=sess.parts)
    sess.study, sess.step_path = new, p.resolve()
    sess._parts, sess.geometry_warnings = new_parts, report
    sess._face_props, sess._edge_props, sess._tess = {}, {}, None
    sess.changed(mesh=True)
    d["applied"] = True
    d["notes"] = list(rep.notes)
    d["iteration"] = new.cad_link.iterations[-1].n if new.cad_link else None
    d["next"] = ("fix the unmapped selections (fea_selection_create) – otherwise fea_check_readiness reports them; then fea_solve_run (the results summary is written into the iteration, fea_iterations)"
                 if d["not_remapped"] else "fea_solve_run – the results summary is written into the iteration (fea_iterations)")
    return d


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_geometry_replace(step_path: str, accept: str = "auto", dry_run: bool = False) -> str:
        """Replaces the geometry with a new STEP (edited CAD model) KEEPING settings: face/edge/point selections remapped, parts matched (also renamed), remote force points and imprints moved with their face, bolts re-detected. Face matching: CAD identity (Fusion add-in table <STEP>.json) -> exact -> unambiguous similar (probable) -> split face -> not found. accept 'auto' (certain+probable+split) | 'certain'; non-accepted selections keep old faces and the readiness check reports them. dry_run=True only proposes. Mesh and results are discarded; the previous results are saved as an iteration (fea_iterations)."""
        with sess.use(op="geometry update"):
            return out(replace_geometry(sess, step_path, accept, dry_run))

    @mcp.tool(structured_output=False)
    def fea_iterations(case: str | None = None) -> str:
        """Geometry iterations (CAD versions): part/parameter changes, remapping, and per version the last results (max stress outside singularities, displacement, safety, mass, 1st frequency) with % change vs the previous. case = load state (default first)."""
        from ..cad_link.history import cases_of, mesh_size_changed, table
        st = sess.require_study()
        link = st.cad_link
        if link is None or not link.iterations:
            return out({"iterations": [], "note": "The study has no iterations yet – they arise with the first geometry update (fea_geometry_replace, CAD add-in “Update in FEA Studio”)"})
        d = {"source": link.source, "document": link.document, "revision": link.revision,
             "cases": cases_of(link), "iterations": table(link, case)}
        if mesh_size_changed(link):
            d["warning"] = "The mesh element size changed between iterations – the stress differences include the effect of the mesh"
        return out(d)
