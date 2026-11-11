"""Technical report on the FE calculation as a .docx.

Content: summary with the assessment → setup (geometry, materials, constraints, load states, thermal states, pictures of the model)
→ mesh and solver → results (thermal states: temperatures, fluxes of the boundary conditions, balance, map of the temperature; mechanical
states: values, reactions and equilibrium, safety by parts, places of the maxima with the classification of the singularities, pictures) → bolts
and welds → fatigue → 3D printing (always a warning about the creep) → assumptions and limits. A study with heat only has no
mechanical sections. Numbers in the Czech style (decimal comma), units for all values.

The pictures are drawn into a temporary folder; with keep_images they are moved to <name>_obrazky – a foreign existing folder is not
overwritten (suffix _2, _3 …), an own folder from an earlier report (marker MARKER) is refreshed only in the files
that the report itself wrote.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__ as FEA_VERSION
from .. import i18n
from ..i18n import tr as _tr
from . import images as IM

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
MARKER = ".fea_protokol_obrazky.json"         # list of the pictures that the report wrote into the folder


@dataclass
class ReportOptions:
    title: str | None = None                  # None = "Strength / Thermal FE calculation – <study>"
    author: str = ""
    company: str = ""
    subject: str = ""                         # subject of the calculation / job (paragraph in the introduction)
    notes: str = ""                           # conclusion / notes of the author (inserted into the conclusion)
    required_safety: float = 1.0              # required safety against Re (assessment of the statics)
    images: bool = True
    result_fields: list[str] = field(default_factory=lambda: ["mises", "displacement"])
    safety_image: bool = True                 # map of the safety against Re (if the material has Re)
    image_view: str = "iso"
    max_hotspots: int = 5
    joint_details: bool = True                # tables of the individual assessments of each joint
    fatigue_top: int = 10
    keep_images: bool = False                 # keep the PNGs next to the report (<name>_obrazky), otherwise delete them
    assess_frame: bool = True                 # assessment of the bars EN 1993-1-1 (beams.assess), if there is a frame system
    voxel: bool = True                        # section "Voxel detail of the printing", if the session has the result (sess.voxel)
    language: str = ""                        # cs | en; "" = jazyk programu (fea_core.i18n)
    _tiny_force: float = 0.0                  # internal: threshold of the numerical noise of the forces [N]
    _size: float = 1.0                        # internal: diagonal of the model [mm]


# ===================================================================== numbers

def num(v, sig: int = 4) -> str:
    """Number: 4 valid digits, decimal comma (in English the point), exponent as ·10ⁿ."""
    if v is None:
        return "–"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(v):
        return "–"
    if math.isinf(v):
        return "∞" if v > 0 else "−∞"
    s = f"{v:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        s = f"{m}·10{str(int(e)).translate(SUP)}"
    return (s if i18n.english() else s.replace(".", ",")).replace("-", "−")


def _sub(i: int) -> str:
    """Subscript with a number (f₁, λ₂)."""
    return str(i).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def vec(v, sig: int = 4, tiny: float = 0.0) -> str:
    return "(" + "; ".join(num(clean(x, tiny), sig) for x in v) + ")"


def clean(x, tiny: float):
    """Numerical noise (|x| < tiny) → 0."""
    try:
        return 0.0 if abs(float(x)) < tiny else x
    except (TypeError, ValueError):
        return x


# ===================================================================== relace

def session_from(study, step_path, results=None, parts=None, fatigue=None, joints=None, workdir=None, motion=None):
    """Session for the report from a finished state (GUI, CLI): study, STEP, parts of OCP (optional), results, result
    of the mechanism (motion = MotionResult → section Motion)…
    """
    from ..mcp.session import Session
    sess = Session(workdir)
    sess.study = study
    sess.step_path = Path(step_path) if step_path else None
    sess._parts = parts
    sess.workdir = Path(workdir) if workdir else None
    sess.results = results
    sess.fatigue = fatigue
    sess.joints = joints
    sess.motion = motion
    if results is not None:
        sess.mesh = results.mesh
    return sess


# ===================================================================== docx helpers

class _Doc:
    def __init__(self):
        from docx import Document
        from docx.shared import Pt, Cm
        from docx.oxml.ns import qn
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        sec.left_margin = sec.right_margin = Cm(2.0)
        sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)
        st = self.doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(10)
        rpr = st.element.get_or_add_rPr()
        lang = rpr.find(qn("w:lang"))
        if lang is None:
            from docx.oxml import OxmlElement
            lang = OxmlElement("w:lang")
            rpr.append(lang)
        lang.set(qn("w:val"), "en-US" if i18n.english() else "cs-CZ")
        st.paragraph_format.space_after = Pt(4)
        self.fig = 0
        self.tab = 0
        self.width = sec.page_width - sec.left_margin - sec.right_margin

    def h(self, text: str, level: int = 1):
        return self.doc.add_heading(_tr(text), level)

    def p(self, text: str = "", bold: bool = False, italic: bool = False, size: float | None = None,
          color: tuple | None = None):
        from docx.shared import Pt, RGBColor
        par = self.doc.add_paragraph()
        if text:
            r = par.add_run(_tr(text))
            r.bold, r.italic = bold, italic
            if size:
                r.font.size = Pt(size)
            if color:
                r.font.color.rgb = RGBColor(*color)
        return par

    def bullet(self, text: str):
        return self.doc.add_paragraph(_tr(text), style="List Bullet")

    def kv(self, rows: list[tuple[str, str]]):
        """Two-column table "item: value" without a heading."""
        self.table(None, rows, widths=(0.35, 0.65))

    def table(self, header: list[str] | None, rows: list[list], widths=None, caption: str | None = None,
              status_col: int | None = None, font: float = 9):
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        if caption:
            self.tab += 1
            self.p(f"Table {self.tab}: {_tr(caption)}", italic=True, size=9)
        ncol = len(header) if header else max(len(r) for r in rows)
        t = self.doc.add_table(rows=0, cols=ncol)
        t.style = "Table Grid"
        if header:
            cells = t.add_row().cells
            for c, txt in zip(cells, header):
                c.text = ""
                r = c.paragraphs[0].add_run(_tr(str(txt)))
                r.bold = True
                r.font.size = Pt(font)
                _shade(c, "D9E2F3")
            _repeat_header(t.rows[0])
        for row in rows:
            cells = t.add_row().cells
            for j, (c, txt) in enumerate(zip(cells, row)):
                txt = "" if txt is None else str(txt)
                c.text = ""
                r = c.paragraphs[0].add_run(_tr(txt))
                r.font.size = Pt(font)
                if _numeric(txt):
                    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
                if status_col is not None and j == status_col:
                    col = {"PASS": "E2EFDA", "FAIL": "F8CBAD"}.get(txt)
                    if col:
                        _shade(c, col)
                        r.bold = True
        if widths:
            for j, w in enumerate(widths):
                for row in t.rows:
                    row.cells[j].width = int(self.width * w)
        self.doc.add_paragraph().paragraph_format.space_after = 0
        return t

    def image(self, path: Path, caption: str):
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from .images import current
        pool = current()
        if pool is not None and pool.pending(path):      # still being drawn: the picture goes into this run later
            pool.slots.append((self.doc.add_paragraph().add_run(), path, self.width))
        else:
            self.doc.add_picture(str(path), width=self.width)
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.fig += 1
        par = self.p(f"Figure {self.fig}: {_tr(caption)}", italic=True, size=9)
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(10)

    def footer(self, text: str):
        """Footer on the right: text and "page X / Y" (fields PAGE / NUMPAGES)."""
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        par = self.doc.sections[0].footer.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = par.add_run(_tr(text) + _tr(" · page "))
        r.font.size = Pt(8)
        _field(par, "PAGE")
        r = par.add_run(" / ")
        r.font.size = Pt(8)
        _field(par, "NUMPAGES")


def _numeric(s: str) -> bool:
    s = s.strip()
    return bool(s) and (s[0].isdigit() or (s[0] in "−-" and len(s) > 1 and s[1].isdigit())) and \
        all(ch.isdigit() or ch in ",.−-·⁻⁰¹²³⁴⁵⁶⁷⁸⁹ %" for ch in s.replace("10", ""))


def _shade(cell, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def _repeat_header(row) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def _field(par, code: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt
    for kind, text in (("begin", None), (None, code), ("separate", None), (None, "1"), ("end", None)):
        r = par.add_run()
        r.font.size = Pt(8)
        if kind:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), kind)
            r._r.append(fc)
        elif text == code:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = f" {code} "
            r._r.append(it)
        else:
            r.text = text


# ===================================================================== descriptions of the setup

def constraint_text(c) -> str:
    from ..study import Fixed, Displacement, CylindricalSupport
    if isinstance(c, Fixed):
        return "fixed support – all displacements zero"
    if isinstance(c, Displacement):
        comp = [f"{k} = {num(v)} mm" for k, v in (("ux", c.ux), ("uy", c.uy), ("uz", c.uz)) if v is not None]
        free = [k for k, v in (("ux", c.ux), ("uy", c.uy), ("uz", c.uz)) if v is None]
        return "prescribed displacement: " + ", ".join(comp) + (f"; free {', '.join(free)}" if free else "")
    if isinstance(c, CylindricalSupport):
        st = lambda b: "constrained" if b else "free"
        return (f"cylindrical support (cylinder axes): radial {st(c.radial)}, tangential {st(c.tangential)}, axial {st(c.axial)}")
    if getattr(c, "kind", None) == "remote_support":
        from ..remote import dofs_text, DOF_NAMES
        cpl = "rigid" if c.coupling == "rigid" else "distributing (RBE3)"
        free = [n for n, on in zip(DOF_NAMES, c.dofs) if not on]
        return (f"remote support, {cpl}, point {vec(c.point)} mm: zero {dofs_text(c.dofs)}"
                + (f"; free {', '.join(free)}" if free else "") + " (u = displacement, r = rotation about the point)")
    if getattr(c, "kind", None) == "elastic":
        from ..loads_text import constraint_text as ct
        return ct(c, None, num) + " (Winkler foundation, linear – transmits tension and compression)"
    return getattr(c, "kind", type(c).__name__)


def load_text(l, study=None) -> tuple[str, str, str]:
    """(kind, selection, value); study = kind of selection (faces / edges / points) and the local coordinate systems."""
    from ..study import Force, Pressure, Gravity, RemoteLoad, BearingLoad, Rotation
    from ..loads_text import force_kind, csys_suffix, pressure_text, rotation_text, entity_of
    if isinstance(l, Force):
        val = f"F = {vec(l.vector)} N, |F| = {num(np.linalg.norm(l.vector))} N{csys_suffix(l, study)}"
        if entity_of(l, study) == "points" and len(study.selections[l.selection].points) > 1:
            n = len(study.selections[l.selection].points)
            val += f"; {n} points per F/{n}"
        if entity_of(l, study) in ("edges", "points"):
            val += " (singular – stress in the vicinity is not valid)"
        return force_kind(l, study), l.selection, val
    if isinstance(l, Pressure):
        kind = "pressure (positive into material)" if not l.variable else \
            ("hydrostatic pressure" if l.clip else "variable (linear) pressure")
        return kind, l.selection, pressure_text(l, num)
    if isinstance(l, Gravity):
        return "self-weight", "entire model", f"g = {vec(l.vector)} mm/s²{csys_suffix(l, study)}"
    if isinstance(l, Rotation):
        return ("rotation (centrifugal and inertial force)", ", ".join(l.parts) if l.parts else "entire model",
                rotation_text(l, num))
    if getattr(l, "kind", None) == "inertia":
        from ..loads_text import inertia_text
        return ("inertia forces (transfer from kinematics)", ", ".join(l.parts) if l.parts else "entire model",
                inertia_text(l, num))
    if isinstance(l, RemoteLoad):
        cpl = "distributing (RBE3)" if l.coupling == "distributing" else "rigid"
        return (f"remote load, coupling {cpl}", l.selection,
                f"point {vec(l.point)} mm; F = {vec(l.force)} N; M = {vec(l.moment)} N·mm{csys_suffix(l, study)}")
    if isinstance(l, BearingLoad):
        return ("bearing load (pressure ~ cos θ on half of the cylinder)", l.selection,
                f"F = {vec(l.vector)} N, |F| = {num(np.linalg.norm(l.vector))} N{csys_suffix(l, study)}")
    if getattr(l, "kind", None) in ("frame_node", "frame_member", "frame_temperature"):
        from .frame import load_text as frame_load
        return frame_load(l)
    if getattr(l, "kind", None) == "thermal":
        return ("thermal load (strain α·(T − T_ref))", "parts " + ", ".join(l.parts) if l.parts else "entire model",
                thermal_load_cz(l))
    return getattr(l, "kind", type(l).__name__), getattr(l, "selection", ""), ""


def thermal_load_cz(l) -> str:
    if l.case:
        return f"temperature field from thermal state “{l.case}”" + ("" if l.time is None else f" at time {num(l.time)} s")
    return f"uniform temperature {num(l.value)} °C"


def bc_text(bc) -> str:
    """Value of a thermal boundary condition in Czech with units."""
    k = bc.kind
    if k == "fixed_temperature":
        return f"T = {num(bc.value)} °C"
    if k == "heat_flux":
        return f"q = {num(bc.value)} W/m² (+ into body)"
    if k == "heat_flow":
        return f"Q = {num(bc.value)} W uniformly over the area (+ into the body)"
    if k == "convection":
        return f"h = {num(bc.h)} W/(m²·K), ambient {num(bc.ambient)} °C"
    if k == "radiation" and getattr(bc, "cavity", False):
        k = "cavity_radiation"
    if k == "radiation":
        return f"ε = {num(bc.emissivity, 3)}, ambient {num(bc.ambient)} °C"
    if k == "cavity_radiation":
        name = getattr(bc, "cavity_name", None) or "CAV"
        return (f"ε = {num(bc.emissivity, 3)}, cavity “{name}”, closed (view factors normalised to 1)" if bc.closed
                else f"ε = {num(bc.emissivity, 3)}, cavity “{name}”, open to ambient {num(bc.ambient)} °C")
    if k == "contact_conductance":
        return f"h_c = {num(bc.conductance)} W/(m²·K) between “{bc.selection}” and “{bc.master}”"
    if k == "heat_generation":
        return f"{num(bc.value)} W uniformly over the part volume"
    return k


def _selection_area(mesh, name: str) -> float | None:
    if mesh is None or name not in getattr(mesh, "surfaces", {}):
        return None
    from ..mesh import tri_areas
    return float(tri_areas(mesh.nodes, mesh.surfaces[name]).sum())


# ===================================================================== protokol

def build_report(sess, path: str | Path, options: ReportOptions | None = None, on_log=None) -> dict:
    """Creates the report .docx from the session. Returns {"path", "images", "pages_hint", "warnings"}."""
    o = options or ReportOptions()
    log = on_log or (lambda s: None)
    study = sess.require_study()
    res = sess.results
    path = Path(path)
    if path.suffix.lower() != ".docx":
        path = path.with_suffix(".docx")
    path.parent.mkdir(parents=True, exist_ok=True)
    img_dir = Path(tempfile.mkdtemp(prefix="fea_protokol_"))      # own temporary folder of the pictures
    i18n.patch_matplotlib()                                         # captions of the pictures in the language of the report
    try:
        with i18n.using(o.language or None), IM.Pool():             # this thread only (the GUI may have another language)
            return _build(sess, study, res, path, img_dir, o, log)   # pictures drawn in parallel (report/images.py)
    finally:
        shutil.rmtree(img_dir, ignore_errors=True)


def _keep_images(images: list[str], target: Path) -> list[str]:
    """Moves the pictures of the report to target; does not overwrite a foreign existing folder (suffix _2, _3 …); in an own
    folder from an earlier time (MARKER) it deletes only the files that the report wrote there before.
    """
    def ours(d: Path) -> list[str] | None:
        try:
            return list(json.loads((d / MARKER).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None
    cand, i = target, 1
    while cand.exists() and ours(cand) is None:
        i += 1
        cand = target.with_name(f"{target.name}_{i}")
    if cand.exists():
        for name in ours(cand) or []:
            p = cand / Path(name).name
            if p.is_file():
                p.unlink()
    cand.mkdir(parents=True, exist_ok=True)
    out = []
    for p in images:
        dst = cand / Path(p).name
        shutil.move(p, dst)
        out.append(str(dst))
    (cand / MARKER).write_text(json.dumps([Path(p).name for p in out], ensure_ascii=False), encoding="utf-8")
    return out


def _build(sess, study, res, path: Path, img_dir: Path, o: ReportOptions, log) -> dict:
    warnings: list[str] = []
    images: list[str] = []
    mech = bool(study.load_cases) and (res is None or bool(res.steps))
    therm = bool(study.thermal.cases)
    D = _Doc()
    kind = "FE thermal analysis" if therm and not mech else ("FE structural and thermal analysis" if therm
                                                              else "FE structural analysis")
    if not study.load_cases and not therm and study.modal.enabled:
        kind = "FEA modal analysis"
    from . import f8 as F8
    kind = F8.title_kind(study, therm) or kind
    from . import frame as FR
    from . import reduced as RD
    rmode = RD.mode(study)
    if rmode:
        kind = RD.title_kind(study)
    fonly = FR.frame_only(study)
    if fonly:
        kind = "Frame analysis" if study.load_cases else "Modal analysis of a frame"
    title = o.title or f"{kind} – {study.name}"
    D.footer(f"{title} · FEA Studio {FEA_VERSION}")

    # ------------------------------------------------------------ titul
    D.doc.add_heading(_tr(title), 0)
    meta = [("Study", study.name), ("Geometry", Path(sess.step_path).name if sess.step_path else
                                     (study.geometry or "no body geometry – frame only")),
            ("Date", time.strftime("%d. %m. %Y")), ("Software", _program_row(study, res))]
    if o.author:
        meta.insert(2, ("Prepared by", o.author))
    if o.company:
        meta.insert(2, ("Company", o.company))
    D.kv(meta)
    if o.subject:
        D.p(o.subject)

    from ..mcp.diagnostics import results_summary
    summ = None
    if res is not None:
        try:
            summ = results_summary(sess)
        except Exception as e:  # noqa: BLE001 – the report is created even without the heuristics
            warnings.append(f"Results summary: {e}")
    if summ is not None:
        scale = [1e-300]
        for s in summ["steps"]:
            eq = s.get("equilibrium") or {}
            scale.append(float(np.linalg.norm(eq.get("applied_load_N") or (0, 0, 0))))
            scale += [r["magnitude_N"] for r in s.get("reactions", {}).values()]
        o._tiny_force = 1e-5 * max(scale)          # .frd has 6 valid digits
        pts = res.mesh.nodes if len(res.mesh.nodes) else np.asarray((res.info.get("frame") or {}).get("xyz") or
                                                                     np.zeros((1, 3)), float)
        o._size = float(np.linalg.norm(np.ptp(pts, axis=0))) or 1.0
    _summary_section(D, sess, summ, o)
    if summ is not None:
        F8.summary_bullets(D, sess)                     # contact, partial results, dynamics

    # ------------------------------------------------------------ setup
    D.h("1 Analysis setup", 1)
    intro = []
    if rmode:
        intro.append(RD.intro(study))
    elif fonly:
        intro.append("Linear static analysis of a frame by the finite element method (beam elements, Timoshenko): linear elastic material, small deformations; member check per EN 1993-1-1. Units: mm, N, MPa, t, s.")
    elif study.is_nonlinear() and study.load_cases:
        pass                                             # nonlinear introduction (contact) below – F8.intro
    elif study.load_cases or not (therm or study.modal.enabled or study.dynamics.enabled):
        intro.append("Linear static finite element analysis: linear elastic material, small deformations, assembly parts rigidly connected by a shared mesh (no contact). Units: mm, N, MPa, t, s."
                     + (" The model combines solid bodies (tet10) and a member system (beam elements)."
                        if study.frame.active else ""))
    intro += F8.intro(study)
    if study.modal.enabled:
        intro.append("Modal analysis (CalculiX *FREQUENCY): natural frequencies and mode shapes of the undamped vibration of the structure with the study's supports, mass from material densities, without damping and without the effect of loading (preload). Units: mm, t, s, Hz.")
    if study.buckling.enabled:
        intro.append("Linear buckling (CalculiX *BUCKLE): critical load factor λ = multiple of the selected load case at which stability is lost (eigenvalues with the stress stiffness matrix) – without imperfections, plasticity and geometric nonlinearity.")
    if therm:
        intro.append("Finite element heat conduction (steady / transient, CalculiX *HEAT TRANSFER): linear conduction, material properties independent of temperature, "
                     + _thermal_contact_sentence(study)
                     + "Units: °C, W, W/m², W/(m²·K)."
                     + (" The temperature field serves as a static load (thermal stress)." if study.has_thermal_loads()
                        else ""))
    for t in intro:
        D.p(t)
    if fonly:
        D.h("1.1 Geometry", 2)
        D.p("The model has no body geometry – only a frame (nodes and members, chapter 1.3).")
    else:
        _geometry_section(D, sess, res)
    _materials_section(D, study, sess)
    sub = 3
    if study.frame.active:
        FR.input_section(D, sess, f"1.{sub}")
        sub += 1
    if rmode:
        RD.input_section(D, sess, f"1.{sub}")
        sub += 1
    if study.load_cases or study.constraints or not therm:
        _bc_section(D, study, res, f"1.{sub}", f"1.{sub + 1}", fonly)
        sub += 2
    if therm:
        _thermal_input_section(D, study, res, f"1.{sub}")
    if o.images:
        for lc in study.load_cases:
            p = _render_model(sess, img_dir, lc.name, warnings)
            if p:
                images.append(str(p))
                D.image(p, f"Model with the load case “{lc.name}” – " + (
                    "members by cross-section, supports blue, loads red" if fonly else
                    "supports blue, loads red") + " (4 views)")
        for tc in study.thermal.cases:
            p = _render_model(sess, img_dir, None, warnings, thermal_case=tc.name)
            if p:
                images.append(str(p))
                D.image(p, f"Thermal boundary conditions of case “{tc.name}” (temperature in purple, flux in orange, convection in turquoise, radiation in yellow, heat generation in pink; 4 views)")
        if not study.load_cases and not therm:
            p = _render_model(sess, img_dir, None, warnings)
            if p:
                images.append(str(p))
                D.image(p, "Model with supports (4 views)")

    # ------------------------------------------------------------ mesh
    D.h("2 Mesh and solver", 1)
    _mesh_section(D, sess, study, res)

    # ------------------------------------------------------------ results
    D.h("3 Results", 1)
    if res is None:
        D.p("Results are not available – the report contains the setup only.", italic=True)
    else:
        n = 0
        for k, tr in enumerate(res.thermal):
            n += 1
            log(f"Report: thermal case {tr.name}")
            ts = next((t for t in (summ or {}).get("thermal", []) if t["index"] == k), None)
            _thermal_section(D, sess, k, ts, f"3.{n}", o, img_dir, images, warnings)
        for i, st in enumerate(res.steps):
            n += 1
            log(f"Report: state {st.name}")
            if not len(res.mesh.nodes) and st.frame:    # frame system only
                FR.step_section(D, sess, i, f"3.{n}", o, img_dir, images, warnings)
                continue
            _step_section(D, sess, i, summ["steps"][i] if summ else None, o, img_dir, images, warnings, f"3.{n}")
            if st.frame:
                FR.step_section(D, sess, i, None, o, img_dir, images, warnings)
        if getattr(res, "twod", None) is not None or getattr(res, "shell", None) is not None:
            log("Report: 2D / shell model")
            n = RD.results_sections(D, sess, o, img_dir, images, warnings, n)
        for j, ms in enumerate(getattr(res, "modes", None) or []):
            n += 1
            log(f"Report: {ms.name}")
            _modes_section(D, sess, j, f"3.{n}", o, img_dir, images, warnings)
        if getattr(res, "submodels", None):
            from . import submodel as SMR
            n = SMR.results_sections(D, sess, o, img_dir, images, warnings, n)

    sec = 4
    if F8.has_contacts(study, res):
        D.h(f"{sec} Contacts and nonlinear analysis", 1)
        log("Report: contacts")
        F8.contacts_section(D, sess, o, img_dir, images, warnings, str(sec))
        sec += 1
    from . import plasticity as PLR
    if PLR.has_plasticity(study, res):
        D.h(f"{sec} Plasticity", 1)
        log("Report: plasticity")
        PLR.section(D, sess, o, img_dir, images, warnings, str(sec), num, _render_field)
        sec += 1
    if F8.has_dynamics(study, res):
        D.h(f"{sec} Dynamics", 1)
        log("Report: dynamics")
        F8.dynamics_section(D, sess, o, img_dir, images, warnings, str(sec))
        sec += 1
    if F8.has_motion(sess):
        D.h(f"{sec} Motion (mechanism)", 1)
        log("Report: mechanism")
        F8.motion_section(D, sess, o, img_dir, images, warnings, str(sec))
        sec += 1
    from . import cyclic as CY
    if CY.has_cyclic(study, res):
        D.h(f"{sec} Cyclic symmetry", 1)
        log("Report: cyclic symmetry")
        CY.section(D, sess, o, img_dir, images, warnings, str(sec))
        sec += 1
    if res is not None and study.frame.active and any(s_.frame for s_ in res.steps) and o.assess_frame:
        D.h(f"{sec} Member check per EN 1993-1-1", 1)
        log("Report: member check")
        FR.assess_section(D, sess, o, img_dir, images, warnings)
        sec += 1
    if res is not None and sess.joints is not None:
        D.h(f"{sec} Bolted and welded joints", 1)
        _joints_section(D, sess.joints, o)
        sec += 1
    elif res is not None and (study.active_bolts() or study.active_welds()):
        D.h(f"{sec} Bolted and welded joints", 1)
        D.p("Joints are in the model but the assessment was not run – for joint forces see the case results.", italic=True)
        sec += 1
    if res is not None and sess.fatigue is not None:
        D.h(f"{sec} Fatigue", 1)
        _fatigue_section(D, sess, o, img_dir, images, warnings)
        sec += 1
    printed = _printed_parts(study, res, _part_names(sess, res))
    if printed:
        D.h(f"{sec} 3D printing", 1)
        _printing_section(D, sess, printed, o, img_dir, images, warnings)
        sec += 1
    from . import param as PS
    if PS.has_param(sess):
        D.h(f"{sec} Parametric study", 1)
        PS.param_section(D, sess)
        sec += 1

    D.h(f"{sec} Assumptions, limitations, and warnings", 1)
    _assumptions(D, study, res, printed, o)
    if o.notes:
        D.h(f"{sec + 1} Conclusion", 1)
        for par in o.notes.split("\n\n"):
            D.p(par.strip())

    D.doc.core_properties.title = title
    D.doc.core_properties.author = o.author or "FEA Studio"
    D.doc.core_properties.subject = study.name
    if IM.current() is not None:
        IM.current().finish(warnings)                  # pictures drawn in parallel → into their places
    D.doc.save(str(path))
    images = _keep_images(images, path.with_name(path.stem + "_obrazky")) if o.keep_images and images else []
    log(f"Report saved: {path}")
    return {"path": str(path), "images": images, "figures": D.fig, "tables": D.tab, "warnings": warnings}


def _part_names(sess, res) -> list[str]:
    """Parts of the study: from the mesh of the result, otherwise from the geometry, otherwise from the assignment of the materials."""
    if res is not None and len(res.mesh.part_names):
        return list(res.mesh.part_names)
    try:
        return sess.part_names()
    except Exception:  # noqa: BLE001 – the STEP need not be available
        return list(sess.study.part_materials)


# --------------------------------------------------------------------- souhrn

def _summary_section(D: _Doc, sess, summ, o: ReportOptions) -> None:
    D.h("Summary", 1)
    if summ is None:
        D.p("The analysis was not run – the report contains the setup only.", italic=True)
        return
    if summ.get("thermal"):
        rows = []
        for t in summ["thermal"]:
            bal = t.get("balance") or {}
            rows.append([t["thermal_case"], "transient" if t["transient"] else "steady-state",
                         num(t["T_min_C"]["value"]), num(t["T_max_C"]["value"]),
                         num(t["max_heat_flux_W_m2"]["value"]),
                         num(bal.get("in_W")) if "in_W" in bal else "–",
                         num(bal.get("imbalance"), 2) if "imbalance" in bal else "–"])
        D.table(["Thermal case", "type", "T min [°C]", "T max [°C]", "max. flux [W/m²]", "heat input [W]",
                 "balance imbalance [–]"], rows, caption="Heat conduction results summary")
    from . import frame as FR
    from . import reduced as RD
    rsum = RD.summary_rows(sess.results)
    if rsum:
        D.table(rsum[0], rsum[1], caption="Results summary – " + (RD.title_kind(sess.study) or "simplified model"))
    frows = FR.summary_rows(summ)
    if frows:
        D.table(["Load case", "max. node displacement [mm]", "max. σ_eq [MPa]", "member", "max. |M| [N·mm]",
                 "imbalance [–]"], frows, caption="Frame results overview")
    if sess.study.frame.active and sess.results is not None and any(st.frame for st in sess.results.steps):
        try:
            if sess.frame_check is None:
                from ..beams.assess import assess
                sess.frame_check = assess(sess.study, sess.results)
            umax, mmax = sess.frame_check.get("_max", (0.0, None))
            if mmax is not None:
                D.bullet(f"Members (EN 1993-1-1): max. utilisation {num(umax, 3)} – member “{mmax}” – "
                         + ("PASS" if umax <= 1 else "FAIL") + " (loads taken as design loads)")
        except Exception as e:  # noqa: BLE001 – stated in the summary (a missing assessment must not look like a pass)
            D.bullet(f"Members (EN 1993-1-1): the assessment could not be computed ({type(e).__name__}: {e}) – "
                     f"the members are NOT assessed in this report")
    rows = []
    for s in summ["steps"]:
        if "max_von_mises_MPa" not in s:               # frame system only
            continue
        smin = s.get("min_safety_Re")
        ok = None if smin is None else smin >= o.required_safety
        eq = s.get("equilibrium") or {}
        rows.append([s["step"], num(s["max_displacement_mm"]["value"]), num(s["max_von_mises_MPa"]["value"]),
                     num(s.get("max_mises_without_singularities_MPa")) if s.get("note") else "–",
                     num(smin, 3), num(eq.get("imbalance_relative"), 2) if eq else "–",
                     "–" if ok is None else ("PASS" if ok else "FAIL")])
    if rows:
        D.table(["Load case", "max. displacement [mm]", "max. σ_vM [MPa]", "σ_vM excl. singularities [MPa]",
                 "min. safety factor vs. Re [–]", "imbalance [–]", f"rating (k ≥ {num(o.required_safety, 3)})"],
                rows, caption="Static results summary", status_col=6)
    if any(s.get("note") for s in summ["steps"]):
        D.p("In some cases the maximum stress lies in a singularity (support edge, sharp corner, point load) – the safety factor at the node is therefore underestimated; the assessment of locations in Chapter 3 is decisive.", italic=True, size=9)
    extra = []
    if sess.joints is not None:
        a = sess.joints
        allc = a.all()
        if allc:
            u = max((c.utilization for c in allc if math.isfinite(c.utilization)), default=float("nan"))
            sts = [_status(c) for c in allc]               # the same evaluation as the tables of the joints
            verdict = ("FAIL" if "FAIL" in sts else
                       "NOT ASSESSED (some joints cannot be assessed)" if "NOT ASSESSED" in sts else "PASS")
            extra.append(f"Joints: {len(a.bolts)} bolts, {len(a.welds)} welds, max. utilisation {num(u, 3)} – " + verdict)
    if sess.fatigue is not None:
        fr = sess.fatigue
        dmax = float(fr.total_damage.max())
        extra.append(f"Fatigue: max. damage D = {num(dmax, 3)} at {num(fr.repeats)} blocks, min. safety factor {num(float(np.min(fr.safety)), 3)} – " + ("PASS" if dmax < 1 else "FAIL"))
    res = sess.results
    pr = [d["min_safety"] for st in res.steps if "creep" not in (st.printing or {})       # without the creep steps
          for d in ((st.printing or {}).get("parts") or {}).values()]
    if pr:
        req = _print_required(sess.study)
        extra.append(f"3D printing: min. safety factor (Tsai-Wu/Hill) {num(min(pr), 3)} – "
                     + ("PASS" if min(pr) >= req else "FAIL")
                     + f" (required k ≥ {num(req, 3)}, without creep, see chapter 3D printing)")
    for ms in summ.get("modes") or []:
        if ms["kind"] == "modal":
            f = ms["frequencies_Hz"]
            extra.append(f"Natural frequencies (number of modes {len(f)}): " + ", ".join(f"f{_sub(i + 1)} = {num(v)} Hz"
                                                                          for i, v in enumerate(f[:4]))
                         + (" …" if len(f) > 4 else "") + " (ch. 3)")
        elif ms["kind"] == "nlbuckle":                # nonlinear buckling with imperfection (fea_core.imperfect)
            from .nlbuckling import summary_line
            extra.append(summary_line(sess))
        else:
            lam = ms["buckling_factors"]
            l1 = min(lam, key=abs) if lam else None
            extra.append(f"Buckling (linear, state “{ms['case']}”): λ₁ = {num(l1, 3)} – critical load = λ × load of the state; without imperfections and plasticity (actual capacity lower)"
                         + (" – LOSS OF STABILITY ALREADY AT THE APPLIED LOAD" if l1 is not None and abs(l1) < 1 else ""))
    for e in extra:
        D.bullet(e)
    if summ.get("warnings"):
        D.p(f"Analysis warnings: {len(summ['warnings'])} (see the final chapter).", italic=True, size=9)


# --------------------------------------------------------------------- setup

def _geometry_section(D: _Doc, sess, res) -> None:
    D.h("1.1 Geometry", 2)
    study = sess.study
    excl = study.excluded_parts()
    names = res.mesh.part_names if res is not None and len(res.mesh.part_names) else None
    if names is None:
        try:
            names = sess.part_names()
        except Exception:  # noqa: BLE001 – the STEP need not be available
            names = list(study.part_materials)
    vol = None
    if res is not None and len(res.mesh.nodes):
        from ..loads import element_volumes
        from ..printing import element_density
        ev = element_volumes(res.mesh)
        rho = element_density(study, res.mesh)
        vol = {pn: (float(ev[res.mesh.tet_part == i].sum()), float((ev * rho)[res.mesh.tet_part == i].sum()))
               for i, pn in enumerate(res.mesh.part_names)}
    rows = []
    for pn in names:
        try:
            mat = study.material_of(pn).name
        except KeyError:
            mat = "–"
        v = vol.get(pn) if vol else None
        rows.append([pn, mat, num(v[0] / 1e3) if v else "–", num(v[1] * 1e6) if v else "–"])
    hdr = ["Part", "Material", "volume [cm³]", "mass [g]"]
    D.table(hdr, rows, caption="Assembly parts (volume and mass from mesh)", widths=(0.4, 0.3, 0.15, 0.15))
    if excl:
        D.p("Parts left out of the calculation (bolts replaced by a beam, nuts, washers, parts excluded by the user): "
            + ", ".join(sorted(excl)) + ".", size=9)


def _materials_section(D: _Doc, study, sess) -> None:
    D.h("1.2 Materials", 2)
    used = set()
    excl = study.excluded_parts()
    for pn in _part_names(sess, sess.results):            # also the parts with the default material of the study
        if pn in excl:
            continue
        try:
            used.add(study.material_of(pn).name)
        except KeyError:
            pass
    mats = [m for m in study.materials.values() if m.name in used] or list(study.materials.values())
    rows = [[m.name, num(m.E), num(m.nu, 3), num(m.rho * 1e12) if m.rho else "–", num(m.Re), num(m.Rm)]
            for m in mats]
    hdr = ["Material", "E [MPa]", "ν [–]", "ρ [kg/m³]", "Re [MPa]", "Rm [MPa]"]
    widths = (0.4, 0.14, 0.1, 0.12, 0.12, 0.12)
    caption = "Material properties (linear elastic)"
    if study.thermal.cases or study.has_thermal_loads():
        for r, m in zip(rows, mats):
            r += [num(m.k), num(m.alpha * 1e6, 3) if m.alpha is not None else "–", num(m.cp)]
        hdr += ["λ [W/(m·K)]", "α [10⁻⁶/K]", "c_p [J/(kg·K)]"]
        widths = (0.25, 0.1, 0.07, 0.09, 0.08, 0.08, 0.11, 0.1, 0.12)
        caption += " and thermal properties (temperature-independent)"
    D.table(hdr, rows, caption=caption, widths=widths)
    pr = [m for m in mats if m.is_print]
    if pr:
        rows = [[m.name, num(m.E_z), num(m.Rm_z), num(m.Rc), num(m.Rc_z), num(m.S_xy), num(m.S_z)] for m in pr]
        D.table(["Printed material (solid extrusion)", "E_z [MPa]", "Rm_z [MPa]", "Rc [MPa]", "Rc_z [MPa]",
                 "S_xy [MPa]", "S_z [MPa]"], rows, caption="Properties of printed materials perpendicular to the layers (Z)")
    orth = [m for m in mats if getattr(m, "ortho", None)]
    if orth:
        from ..csys import describe
        rows = []
        for m in orth:
            o = m.ortho
            cs = study.coordinate_systems.get(m.ortho_csys or "")
            axes = (describe(cs) + (" (1 = r, 2 = θ, 3 = z)" if cs.kind == "cylindrical" else " (1, 2, 3 = x, y, z)")
                    if cs is not None else "global axes (1, 2, 3 = x, y, z)")
            alpha = [o.get(f"alpha{i}") for i in (1, 2, 3)]
            rows.append([m.name, f"{num(o['E1'])} / {num(o['E2'])} / {num(o['E3'])}",
                         f"{num(o['nu12'], 3)} / {num(o['nu13'], 3)} / {num(o['nu23'], 3)}",
                         f"{num(o['G12'])} / {num(o['G13'])} / {num(o['G23'])}",
                         " / ".join(num(a * 1e6, 3) for a in alpha) if all(a is not None for a in alpha) else "–",
                         axes])
        D.table(["Orthotropic material", "E1 / E2 / E3 [MPa]", "ν12 / ν13 / ν23 [–]", "G12 / G13 / G23 [MPa]",
                 "α1 / α2 / α3 [10⁻⁶/K]", "material axes"], rows,
                caption="Orthotropic materials – engineering constants in the material axes (ε_j = −ν_ij·σ_i/E_i)",
                widths=(0.18, 0.17, 0.15, 0.17, 0.13, 0.2), font=8)
        D.p("WARNING: for orthotropic materials the safety assessment against Re based on equivalent von Mises stress is not valid (safety tables omit these parts) – stress must be assessed in the material axes against its strengths (e.g. Tsai-Wu, Hill). Stresses in the results are in global axes.",
            bold=True, size=9, color=(0xC0, 0x00, 0x00))
    for m in mats:
        if m.source:
            D.p(f"Source – {m.name}: {m.source}", size=8)


def _bc_section(D: _Doc, study, res, lab_bc: str = "1.3", lab_lc: str = "1.4", frame_only: bool = False) -> None:
    mesh = res.mesh if res is not None else None
    D.h(f"{lab_bc} Constraints", 2)
    rows = []
    for c in study.constraints:
        sel = study.selections.get(c.selection)
        a = _selection_area(mesh, c.selection)
        rows.append([c.selection, constraint_text(c), len(sel.faces) if sel else "–", num(a) if a else "–"])
    if rows:
        D.table(["Selection", "Constraint", "faces", "area [mm²]"], rows, caption="Supports (boundary conditions)",
                widths=(0.25, 0.5, 0.1, 0.15))
    else:
        D.p("No body supports – for member supports see the frame." if study.frame.supports else "No constraints.",
            italic=True)
    D.h(f"{lab_lc} Load cases", 2)
    if not study.load_cases:
        D.p("No mechanical load cases.", italic=True)
    elif study.is_nonlinear():
        D.p("The analysis is nonlinear (contact) – load cases cannot be superposed; for the load history see the Contacts chapter (branches or step sequence).")
    else:
        D.p("Each load case is a separate linear analysis; cases can be superposed (fatigue).")
    if study.has_thermal_loads():
        D.p(f"Thermal load: strain α·(T − T_ref), reference (assembly) temperature T_ref = {num(study.thermal.reference_temperature)} °C; the temperature field from the thermal case is taken at the mesh nodes.",
            size=9)
    if study.coordinate_systems:
        from ..csys import describe
        D.p("Local coordinate systems of the loads: " + "; ".join(describe(c) for c in study.coordinate_systems.values())
            + ". In the cylindrical system the components are (r, θ, z) and the direction changes over the area.", size=9)
    for lc in study.load_cases:
        rows = []
        for l in lc.loads:
            kind, sel, val = load_text(l, study)
            a = _selection_area(mesh, sel) if sel and sel in study.selections and study.selections[sel].faces else None
            rows.append([kind, sel, val + (f"; A = {num(a)} mm²" if a else "")])
        D.table(["Loads", "Selection", "Value"], rows or [["no loads", "", ""]],
                caption=f"Load case “{lc.name}”", widths=(0.3, 0.2, 0.5))
    if study.active_bolts():
        rows = [[b.name, b.designation, b.strength, num(b.grip), b.preload_mode,
                 num(b.preload) if b.preload_mode != "auto" else "auto"] for b in study.active_bolts()]
        D.table(["Bolt", "Designation", "Class", "clamping length [mm]", "preload", "value [N / N·m]"], rows,
                caption="Bolts (FEA: beam between bearing faces, preload per VDI 2230)")
    if study.active_welds():
        rows = [[w.name, w.part, num(w.a), num(w.length)] for w in study.active_welds()]
        D.table(["Weld", "Part (weld body)", "a [mm]", "length [mm]"], rows, caption="Welds")

    if hasattr(study, "combinations") and study.combinations:
        D.h(f"{lab_lc}.1 Load case combinations (ČSN EN 1990)", 3)
        D.p("Linear combinations and envelopes per ČSN EN 1990. Superposition of stress tensors and displacements; the equivalent von Mises stress is computed from the combined tensor.")
        c_rows = []
        for c in study.combinations:
            if c.envelope:
                f_str = f"{c.envelope_type} ({', '.join(c.envelope_combinations)})"
                c_rows.append([c.name, f"Envelope ({c.envelope_field})", f_str, c.description or "ČSN EN 1990"])
            else:
                f_str = " + ".join(f"{v:g}·{k}" for k, v in c.factors.items())
                c_rows.append([c.name, "Linear", f_str, c.description or "ČSN EN 1990"])
        D.table(["Combination", "Type", "Basis", "Description / Standard"], c_rows,
                caption="Load case combinations", widths=(0.25, 0.2, 0.35, 0.2))



def _thermal_contact_sentence(study) -> str:
    from .thermal_extras import contact_sentence
    return contact_sentence(study)


def _thermal_input_section(D: _Doc, study, res, num_label: str) -> None:
    """Setup of the heat: reference / initial temperature, thermal states and their boundary conditions."""
    from ..mcp.diagnostics import THERMAL_LABELS
    mesh = res.mesh if res is not None else None
    th = study.thermal
    D.h(f"{num_label} Thermal cases", 2)
    rows = [("Reference (assembly) temperature", f"{num(th.reference_temperature)} °C – no thermal strain")]
    if any(c.transient for c in th.cases):
        rows.append(("Initial temperature of transient cases", f"{num(th.t_initial)} °C"))
    D.kv(rows)
    D.p("Faces without a boundary condition are thermally insulated (adiabatic). Positive flux / power = heat enters the body. Convection q = h·(T_ambient − T), radiation q = ε·σ·(T_ambient⁴ − T⁴) in absolute temperatures.", size=9)
    for c in th.cases:
        rows = []
        for bc in c.loads:
            sel = getattr(bc, "selection", None)
            a = _selection_area(mesh, sel) if sel else None
            rows.append([THERMAL_LABELS.get(bc.kind, bc.kind), sel or f"part {bc.part}", bc_text(bc),
                         num(a) if a else "–"])
        if c.transient:
            from ..thermal.inp import time_stepping
            n, freq, dt = time_stepping(c)
            typ = (f"transient case: duration {num(c.time)} s, {n} steps of {num(dt)} s, {c.outputs} saved frames")
        else:
            typ = "steady state"
        D.table(["Boundary condition", "Selection / part", "Value", "area [mm²]"], rows or [["no conditions", "", "", ""]],
                caption=f"Thermal case “{c.name}” – {typ}", widths=(0.25, 0.2, 0.4, 0.15))
    from .thermal_extras import input_tables
    input_tables(D, study)                               # contact conductances of interfaces (all thermal cases)


def _render_model(sess, img_dir: Path, lc, warnings, thermal_case: str | None = None) -> Path | None:
    from .images import numbered, render
    from ..mcp.scenes import model_scene
    try:
        scene = model_scene(sess, lc, None, False, None, thermal_case)
        img_dir.mkdir(parents=True, exist_ok=True)
        p = numbered(img_dir, "model_")
        render(scene, p, "multi", 1600, 1000)
        return p
    except Exception as e:  # noqa: BLE001 – without geometry (the STEP is missing) the report continues without the picture
        warnings.append(f"Model figure ({lc or thermal_case}): {e}")
        return None


# --------------------------------------------------------------------- mesh

def _ccx_solver(study, res) -> str:
    """The linear solver with which CalculiX actually computed (Results.info["ccx_solver"]); for results without a record
    (loaded project) according to the installation (config.solver_name – PARDISO without MKL = PaStiX / SPOOLES).
    """
    from .. import config
    return (res.info.get("ccx_solver") if res is not None else None) or config.solver_name(study.solver.solver)


def _program_row(study, res) -> str:
    """Title page: the solvers actually used in the calculation (without results according to the setup)."""
    from . import frame as FR
    from ..beams.fem import solver_of
    be = res.info.get("backend") if res is not None else None
    if FR.frame_only(study) and solver_of(study) == "timoshenko":
        return f"FEA Studio {FEA_VERSION} (own beam solver)"
    used = {"ngs_hypre": ["NGSolve + hypre"], "ngs_pardiso": ["NGSolve + PARDISO"]}.get(be, [])
    if be not in ("ngs_hypre", "ngs_pardiso") or study.thermal.cases or study.modal.enabled \
            or study.buckling.enabled or study.dynamics.enabled:
        used.insert(0, "CalculiX")
    if not FR.frame_only(study):
        used.append("Gmsh")
    return f"FEA Studio {FEA_VERSION} ({', '.join(used)})"


def _mesh_section(D: _Doc, sess, study, res) -> None:
    ms = study.mesh
    from . import frame as FR
    from . import reduced as RD
    if RD.mode(study):
        rows = RD.mesh_rows(sess, res)
        D.kv(rows)
        return
    if FR.frame_only(study):
        rows = FR.mesh_rows(sess, res)
        from ..beams.fem import solver_of
        tim = solver_of(study) == "timoshenko"           # default: the own solver of the bars (without CalculiX and MYSTRAN)
        if study.load_cases:
            rows.append(("Solver", "own beam solver – direct, sparse matrices (scipy), one factorization for all cases"
                         if tim else f"CalculiX 2.23, direct solver {_ccx_solver(study, res)}"))
            rows.append(("Equilibrium check", "external load vs. support reactions (forces and moments), tolerance 1·10⁻³"))
        if study.modal.enabled:
            md = study.modal
            rows.append(("Natural frequencies", f"own beam solver (consistent mass), number of modes {md.modes}"
                         if tim else f"CalculiX *FREQUENCY, {md.modes} modes"))
        if study.buckling.enabled:
            rows.append(("Linear buckling", f"own beam solver (geometric stiffness matrix), number of shapes {study.buckling.modes}" if tim else
                         f"MYSTRAN SOL 105 (CBAR), {study.buckling.modes} mode shapes"))
        D.kv(rows)
        return
    mech = bool(study.load_cases)
    therm = bool(study.thermal.cases)
    rows = [("Elements", "quadratic tetrahedra C3D10 (tet10)" if ms.order == 2 else "linear tetrahedra C3D4"),
            ("Global element size", f"{num(ms.size)} mm" + (f", min. {num(ms.min_size)} mm" if ms.min_size else "")),
            ("Curvature-based refinement", f"{ms.curvature_points} elements per 2π" if ms.curvature_points else "off")]
    for loc in ms.local or []:
        if "center" in loc:
            rows.append(("Local refinement (sphere)", f"center {vec(loc['center'], 5)} mm, radius {num(loc.get('radius'))} mm: {num(loc.get('size'))} mm"))
        else:
            rows.append(("Local refinement", f"{loc.get('selection')}: {num(loc.get('size'))} mm"))
    if ms.separate:
        rows.append(("Separately meshed parts", ", ".join(ms.separate) + " – connected by *TIE (tolerance "
                     + (f"{num(ms.tie_tolerance)} mm" if ms.tie_tolerance else "5 % of the element size") + ")"))
    if study.modal.enabled:
        md = study.modal
        rows.append(("Natural frequencies", f"CalculiX *FREQUENCY, number of modes {md.modes}, range {num(md.f_min)} – "
                                          + (f"{num(md.f_max)} Hz" if md.f_max else "∞ Hz") + ", same mesh"))
    if study.buckling.enabled:
        bs = study.buckling
        case = bs.case or (study.load_cases[0].name if study.load_cases else "–")
        rows.append(("Linear buckling", f"CalculiX *BUCKLE, {bs.modes} modes, loads of case “{case}”"))
    if study.nlbuckling.enabled:
        rows.append(("Nonlinear buckling", "CalculiX *STATIC, NLGEOM on the imperfect mesh, load control with automatic increments"))
    mesh = res.mesh if res is not None else sess.mesh
    if res is not None:
        m = res.mesh
        rep = m.report or {}
        dof = [f"{3 * len(m.nodes)} degrees of freedom " + ("statics" if mech else "")] \
            if mech or study.modal.enabled else []
        dof = [x.rstrip() for x in dof]
        if therm:
            dof.append(f"{len(m.nodes)} temperatures")
        rows += [("Nodes / elements", f"{len(m.nodes)} / {len(m.tets)} ({', '.join(dof)})")]
        if rep.get("min_scaled_jacobian") is not None:
            sj = rep["min_scaled_jacobian"]
            rows.append(("Min. scaled Jacobian", num(sj, 3) + (" (OK)" if sj >= 0.2 else " (low quality!)")))
        if mech:
            be = res.info.get("backend") or study.solver.backend
            rows.append(("Solver", {"ccx": f"CalculiX 2.23, direct solver {_ccx_solver(study, res)}",
                                   "ngs_hypre": "NGSolve (matrix) + hypre BoomerAMG (PCG, MPI)",
                                   "ngs_pardiso": "NGSolve (matrix) + direct MKL PARDISO solver (single factorization)"
                                   }.get(be, str(be))))
    if therm:
        rows.append(("Heat conduction", f"CalculiX 2.23 *HEAT TRANSFER (steady / transient state with a fixed step), direct solver {_ccx_solver(study, res)}" + (", same mesh as statics" if mech else "")))
        rows.append(("Heat balance check", "steady state: sum of boundary condition heat = 0, tolerance 1·10⁻³ of the fluxes (prescribed temperatures from RFL reactions)"))
    if mech:
        rows.append(("Equilibrium check", "external load vs. reactions (forces and moments), tolerance 1·10⁻³"))
    if study.frame.active:
        rows += FR.mesh_rows(sess, res)
    D.kv(rows)
    if mesh is not None and len(mesh.part_names) > 1:
        try:
            from ..contacts import bonded_pairs, edge_only
            pairs = bonded_pairs(mesh)
            edges = edge_only(mesh)
        except Exception:  # noqa: BLE001 – the overview of the connections is only informative
            pairs, edges = [], []
        if pairs:
            D.table(["Part A", "Part B", "contact area [mm²]", "element faces"],
                    [[p["a"], p["b"], num(p["area"]), p["faces"]] for p in pairs],
                    caption=("Connected parts – rigidly connected by a shared mesh (other part interfaces are contact pairs, see the Contacts chapter)" if study.active_contacts() else
                             "Connected parts – rigidly connected by a shared mesh (no contact; also perfect thermal contact)"),
                    widths=(0.32, 0.32, 0.2, 0.16), font=8)
        for a, b in edges:
            D.p(f"Warning: parts “{a}” and “{b}” touch only along an edge or at a point – connection through a few nodes (singularity).", italic=True, size=9)
    ties = ((mesh.report or {}).get("ties") or []) if mesh is not None else []
    if ties:
        D.table(["Separately meshed part", "*TIE area [mm²]", "tolerance [mm]"],
                [[t.get("part"), num(t.get("area_mm2")), num(t.get("tolerance_mm"), 3)] for t in ties],
                caption="*TIE connection – separately meshed parts (non-conforming mesh) tied to surrounding parts (slave face = separate part)", widths=(0.5, 0.25, 0.25), font=8)
    elif ms.separate and mesh is not None:
        D.p("Warning: separately meshed parts have no *TIE face (they do not touch another part within tolerance).",
            italic=True, size=9)


# --------------------------------------------------------------------- results of the state

def _render_field(sess, i, key, img_dir: Path, view: str, warnings, deformed: bool = True,
                  thermal: int | None = None) -> tuple[Path, dict] | None:
    """Map of a field: the mechanical state i (deformed), or the thermal state thermal (always the undeformed shape)."""
    from ..mcp.diagnostics import FIELDS, REVERSED, SIGNED, field_values, thermal_values
    from .images import render
    from ..mcp.scenes import result_scene
    try:
        vals = thermal_values(sess, thermal, key) if thermal is not None else field_values(sess, i, key)
        label, unit = FIELDS[key]
        scene, meta = result_scene(sess, i, key, vals, label, unit, key in REVERSED, deformed, None, 12, None, None,
                                   False, key in SIGNED, thermal=thermal)
        img_dir.mkdir(parents=True, exist_ok=True)
        p = img_dir / (f"{key}_T{thermal}.png" if thermal is not None else f"{key}_{i}.png")
        render(scene, p, view, 1600, 900)
        return p, meta
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Figure {key} ({(i if thermal is None else 'thermal case ' + str(thermal))}): {e}")
        return None


FIELD_CAPTION = {"mises": "Equivalent von Mises stress", "displacement": "Total displacement",
                 "safety": "Safety factor to yield strength Re/σ_vM (capped at 10)", "s1": "1st principal stress",
                 "s3": "3rd principal stress", "print_safety": "3D printing – safety factor (capped at 10)",
                 "fatigue_damage": "Fatigue – log10 damage D", "fatigue_safety": "Fatigue – safety factor (capped at 10)",
                 "fatigue_life": "Fatigue – log10 life in blocks", "temperature": "Temperature",
                 "heat_flux": "Heat flux magnitude"}


def _thermal_section(D: _Doc, sess, k: int, t: dict | None, num_label: str, o: ReportOptions, img_dir, images,
                     warnings) -> None:
    """Results of the thermal state: temperatures, flux, temperatures by parts, fluxes of the boundary conditions, balance, map of the temperature."""
    from ..mcp.diagnostics import THERMAL_LABELS
    tr = sess.results.thermal[k]
    D.h(f"{num_label} Thermal case “{tr.name}”", 2)
    if t is None:
        D.p("Thermal case summary is not available.", italic=True)
        return
    end = f" (end of the state, t = {num(tr.times[-1])} s)" if tr.transient and tr.times else ""
    D.kv([("Type", "transient" if tr.transient else "steady-state"),
          ("T max" + end, f"{num(t['T_max_C']['value'])} °C at point {vec(t['T_max_C']['at_mm'], 5)} mm (part {t['T_max_C']['part']})"),
          ("T min" + end, f"{num(t['T_min_C']['value'])} °C at point {vec(t['T_min_C']['at_mm'], 5)} mm (part {t['T_min_C']['part']})"),
          ("Max. heat flux", f"{num(t['max_heat_flux_W_m2']['value'])} W/m² at point {vec(t['max_heat_flux_W_m2']['at_mm'], 5)} mm (part {t['max_heat_flux_W_m2']['part']})")])
    if len(t.get("parts", [])) > 1:
        D.table(["Part", "T min [°C]", "T max [°C]"], [[p["part"], num(p["T_min_C"]), num(p["T_max_C"])]
                                                       for p in t["parts"]],
                caption=f"Temperatures by part – thermal case “{tr.name}”", widths=(0.5, 0.25, 0.25), font=8)
    case = next((c for c in sess.study.thermal.cases if c.name == tr.name), None)
    given = {}
    for bc in (case.loads if case is not None else []):
        key = getattr(bc, "selection", None) or getattr(bc, "part", "")
        given[(bc.kind, ", ".join(key) if isinstance(key, list) else key)] = bc_text(bc)
    rows = []
    for f in t.get("heat_flows_W", []):
        rows.append([THERMAL_LABELS.get(f["kind"], f["kind"]), f["name"], given.get((f["kind"], f["name"]), "–"),
                     num(f["Q_W"])])
    if rows:
        D.table(["Boundary condition", "Selection / part", "setup", "heat Q [W]"], rows,
                caption=f"Heat of the boundary conditions – thermal state “{tr.name}” (positive = into the body, negative = removed"
                        + (", at the end of the state" if tr.transient else "") + ")",
                widths=(0.25, 0.2, 0.4, 0.15), font=8)
    from .thermal_extras import result_tables
    result_tables(D, tr)                                 # heat through conductance interfaces and cavity faces
    bal = t.get("balance") or {}
    if "imbalance" in bal:
        D.p(f"Heat balance: supply {num(bal.get('in_W'))} W, removal {num(bal.get('out_W'))} W, residual {num(bal.get('residual_W'), 2)} W, relative imbalance {num(bal['imbalance'], 2)} "
            + ("– OK." if bal.get("ok", True) else "– tolerance EXCEEDED!"), size=9)
    elif bal.get("note"):
        D.p(f"Heat balance: {bal['note']}.", size=9)
    if tr.transient and t.get("frames"):
        D.table(["time [s]", "T min [°C]", "T max [°C]"],
                [[num(f["t_s"]), num(f["T_min_C"]), num(f["T_max_C"])] for f in t["frames"]],
                caption=f"Temperature history – thermal case “{tr.name}” (saved frames)",
                widths=(0.34, 0.33, 0.33), font=8)
    if o.images:
        r = _render_field(sess, 0, "temperature", img_dir, o.image_view, warnings, deformed=False, thermal=k)
        if r:
            images.append(str(r[0]))
            D.image(r[0], f"Temperature – thermal state “{tr.name}”" + end)


def _step_section(D: _Doc, sess, i: int, s: dict | None, o: ReportOptions, img_dir, images, warnings,
                  num_label: str | None = None) -> None:
    res = sess.results
    st = res.steps[i]
    D.h(f"{num_label or f'3.{i + 1}'} State “{st.name}“", 2)
    if st.partial:
        D.p("PARTIAL RESULT: the nonlinear analysis of this case did not converge – the values are from the last converged increment (the full load was not carried).", bold=True, size=9, color=(0xC0, 0x00, 0x00))
    if s is None:
        D.p("Case summary is not available.", italic=True)
        return
    rows = [("Max. displacement", f"{num(s['max_displacement_mm']['value'])} mm at point {vec(s['max_displacement_mm']['at_mm'], 5)} mm"),
            ("Max. σ_vM", f"{num(s['max_von_mises_MPa']['value'])} MPa at point {vec(s['max_von_mises_MPa']['at_mm'], 5)} mm (part {s['max_von_mises_MPa']['part']})"),
            ("σ1 max / σ3 min", f"{num(s['max_principal_s1_MPa']['value'])} / {num(s['min_principal_s3_MPa']['value'])} MPa")]
    for c, (lo, hi) in s["displacement_range_mm"].items():
        rows.append((f"Range {c}", f"{num(lo)} … {num(hi)} mm"))
    tc = s.get("temperature_C")
    if tc:
        lc = next((x for x in sess.study.load_cases if x.name == st.name), None)
        src = "; ".join(thermal_load_cz(l) for l in (lc.loads if lc else []) if l.kind == "thermal") or "–"
        rows.append(("Thermal load", f"{src}; temperature in the model {num(tc['min'])} … {num(tc['max'])} °C, reference {num(tc['reference'])} °C"))
    D.kv(rows)
    if s.get("note"):
        D.p(s["note"].replace("see fea_results_hotspots", "see table of maximum locations"), italic=True, size=9)

    # reactions and equilibrium
    tf, tm = o._tiny_force, o._tiny_force * o._size
    rr = []
    for k, r in s["reactions"].items():
        f = np.asarray(r["force_N"], float)
        m = np.asarray(r["moment_Nmm"], float) if r.get("moment_Nmm") is not None else np.full(3, np.nan)
        rr.append([k, *[num(clean(x, tf)) for x in f], num(clean(r["magnitude_N"], tf)), *[num(clean(x, tm)) for x in m]])
    if rr:
        D.table(["Constraint", "Rx [N]", "Ry [N]", "Rz [N]", "|R| [N]", "Mx [N·mm]", "My [N·mm]", "Mz [N·mm]"], rr,
                caption=f"Support reactions – case “{st.name}” (moments about point {vec(s.get('reaction_moment_point_mm', (0, 0, 0)), 4)} mm)",
                font=8)
    rs = s.get("remote_supports") or {}
    if rs:                                       # remote constraints: force and moment ABOUT THE POINT of the constraint, motion of the point
        tu = 1e-6 * float(s["max_displacement_mm"]["value"])          # noise of the constrained components → 0
        tr = tu / max(o._size, 1e-9)
        rows_r = [[k, vec(r["point_mm"], 5), "rigid" if r["coupling"] == "rigid" else "distributing",
                   ", ".join(r["fixed_dofs"]) or "–", vec(r["force_N"], tiny=tf), vec(r["moment_about_point_Nmm"], tiny=tm),
                   vec(r["point_displacement_mm"], tiny=tu), vec(r["point_rotation_rad"], tiny=tr)] for k, r in rs.items()]
        D.table(["Remote support", "point [mm]", "constraint", "constrained", "reaction F [N]", "moment about point [N·mm]",
                 "point displacement [mm]", "point rotation [rad]"], rows_r,
                caption=f"Remote supports – reactions at the support point, case “{st.name}”", font=8)
    eq = s.get("equilibrium")
    if eq:
        D.p(f"Equilibrium check: external load {vec(eq['applied_load_N'] or (0, 0, 0), tiny=tf)} N, sum of reactions {vec(eq['reaction_total_N'] or (0, 0, 0), tiny=tf)} N, relative imbalance {num(eq['imbalance_relative'], 2)} "
            + ("– OK." if eq.get("ok") else "– tolerance EXCEEDED!"), size=9)

    # safety by parts
    pr = [[p["part"], p.get("material") or "–", num(p["max_mises_MPa"]), num(p.get("Re_MPa")),
           num(p.get("safety_Re"), 3),
           "–" if p.get("safety_Re") is None else ("PASS" if p["safety_Re"] >= o.required_safety else "FAIL")]
          for p in s.get("parts", [])]
    if pr:
        D.table(["Part", "Material", "max. σ_vM [MPa]", "Re [MPa]", "safety factor Re/σ_vM [–]",
                 f"rating (k ≥ {num(o.required_safety, 3)})"], pr,
                caption=f"Safety factor to yield strength by part – case “{st.name}” (maximum at a node, incl. any singularities)",
                status_col=5, font=8)
        ortho = [p["part"] for p in s.get("parts", []) if p.get("note")]
        if ortho:
            D.p(f"Parts made of orthotropic material ({', '.join(ortho)}): the safety factor to Re based on von Mises stress does not apply – assess stress in the material axes.", italic=True, size=9)

    # places of the maxima
    try:
        from ..mcp.diagnostics import hotspots
        hs = hotspots(sess, i, o.max_hotspots)["hotspots"]
    except Exception as e:  # noqa: BLE001
        hs = []
        warnings.append(f"Locations of maxima ({st.name}): {e}")
    if hs:
        rows = [[h["rank"], vec(h["xyz_mm"], 4), num(h["mises_MPa"]), ", ".join(h["parts"]), h["classification"],
                 "; ".join(h["reasons"]) or "–"] for h in hs]
        D.table(["#", "location [mm]", "σ_vM [MPa]", "part", "classification", "reason"], rows,
                caption=f"Highest-stress locations – case “{st.name}” (automatic singularity classification)",
                widths=(0.04, 0.2, 0.1, 0.14, 0.17, 0.35), font=8)
        recs = []
        for h in hs:
            if h["recommendation"] not in recs:
                recs.append(h["recommendation"])
        for r in recs:
            D.bullet(r)

    if o.images:
        keys = list(o.result_fields)
        if o.safety_image and "safety" not in keys and np.isfinite(sess.node_re()).any():
            if (s.get("min_safety_Re") or np.inf) < 10:
                keys.append("safety")
            else:
                D.p("The safety factor with respect to Re is greater than 10 everywhere – the map is not shown.", italic=True, size=9)
        if st.fields.get("print_safety") is not None and "print_safety" not in keys:
            keys.append("print_safety")
        for key in keys:
            r = _render_field(sess, i, key, img_dir, o.image_view, warnings)
            if r:
                p, meta = r
                images.append(str(p))
                k = meta.get("deformation_scale") or 0
                D.image(p, f"{FIELD_CAPTION.get(key, key)} – state “{st.name}“"
                        + (f", deformation scaled {num(k, 3)}×" if k > 0 else ""))
    pj = st.joints or {}
    if pj and sess.joints is None:
        D.p("Joint forces are stored in the result; the joint assessment was not run.", italic=True, size=9)


# --------------------------------------------------------------------- natural frequencies and buckling

def _render_mode(sess, j: int, k: int, img_dir: Path, view: str, warnings) -> Path | None:
    """Picture of the shape of the mode k of the set Results.modes[j] (|u| normalised, enlarged shape)."""
    from .images import render
    from ..mcp.scenes import result_scene
    try:
        ms = sess.results.modes[j]
        if not len(sess.results.mesh.nodes) or sess.study.frame.members:      # frame system
            from ..mcp.frame_scenes import mode_frame_scene
            scene, _ = mode_frame_scene(sess, j, k)
            img_dir.mkdir(parents=True, exist_ok=True)
            p = img_dir / f"mode_{ms.kind}_{k + 1}.png"
            render(scene, p, view, 1600, 900)
            return p
        vals = np.linalg.norm(np.asarray(ms.shapes[k], float), axis=1)
        nl = ms.kind == "nlbuckle"                    # nonlinear buckling: real displacement [mm]
        scene, _ = result_scene(sess, 0, "displacement", vals, "|u|" if nl else "Mode |u|", "mm" if nl else "–", False,
                                True, None, 12, None, None,
                                False, False, mode=(j, k))
        img_dir.mkdir(parents=True, exist_ok=True)
        p = img_dir / f"mode_{ms.kind}_{k + 1}.png"
        render(scene, p, view, 1600, 900)
        return p
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Mode shape figure {k + 1}: {e}")
        return None


def _modes_section(D: _Doc, sess, j: int, num_label: str, o: ReportOptions, img_dir, images, warnings) -> None:
    """Natural frequencies (table f, shares of the effective mass, shapes 1–3) or linear buckling (λ, critical load,
    1st shape, warning).
    """
    from ..mcp.diagnostics import modes_summary
    ms = sess.results.modes[j]
    if ms.kind == "nlbuckle":                       # nonlinear buckling with imperfection (F9 C2)
        from .nlbuckling import nlbuckling_section
        return nlbuckling_section(D, sess, j, num_label, o, img_dir, images, warnings, _render_mode)
    d = modes_summary(sess, j)
    if ms.kind == "modal":
        D.h(f"{num_label} Natural frequencies", 2)
        D.p("Natural frequencies and mode shapes of undamped vibration with the study's supports. Effective mass fraction = effective mass of the mode in the x / y / z direction relative to the total effective mass (mass of free nodes); a sum ≥ 90 % = the modes capture the response to excitation in that direction.", size=9)
        rows = []
        pct = lambda v: num(clean(100 * float(v), 1e-4), 3)          # noqa: E731 – noise of ccx (1e-10 %) → 0
        for m in d.get("modes", []):
            fr, cu = m.get("eff_mass_fraction_xyz"), m.get("cumulative_xyz")
            rows.append([m["mode"], num(m["frequency_Hz"]), num(m.get("period_s")),
                         *([pct(x) for x in fr] if fr is not None else ["–"] * 3),
                         *([pct(x) for x in cu] if cu is not None else ["–"] * 3)])
        D.table(["mode", "f [Hz]", "T [s]", "m_ef,x [%]", "m_ef,y [%]", "m_ef,z [%]", "Σ x [%]", "Σ y [%]", "Σ z [%]"],
                rows, caption="Natural frequencies and effective mass fractions", font=8)
        cum = d.get("cumulative_mass_fraction_xyz")
        if cum is not None:
            D.p(f"Sum of effective mass fractions of all computed modes: x {pct(cum[0])} %, y {pct(cum[1])} %, z {pct(cum[2])} %; total effective mass {num(d.get('total_mass_kg'))} kg.", size=9)
        for n in d.get("notes", [])[2:]:              # the first two notes are for an LLM (pictures, definitions)
            D.p(n, italic=True, size=9)
        if o.images:
            for k in range(min(3, len(ms.values))):
                p = _render_mode(sess, j, k, img_dir, o.image_view, warnings)
                if p:
                    images.append(str(p))
                    D.image(p, f"Mode shape {k + 1}: f = {num(ms.values[k])} Hz (shape normalized, scaled up)")
        return
    D.h(f"{num_label} Buckling (linear) – case “{ms.case}”", 2)
    D.p("Critical load factor λ: stability is lost at λ times the load case loading (all loads of the case are scaled together). Negative λ = loss of stability under reversed loading.", size=9)
    crit = d.get("critical_load_N")
    rows = [[i + 1, num(v, 4), num(crit[i]) if crit else "–"] for i, v in enumerate(d["buckling_factors"])]
    D.table(["shape", "λ [–]", "critical resultant λ·|F| [N]"], rows,
            caption=f"Linear buckling – critical load factors (case “{ms.case}”)", widths=(0.15, 0.35, 0.5),
            font=8)
    lc = next((x for x in sess.study.load_cases if x.name == ms.case), None)
    if lc is not None and lc.loads:
        D.p("Load of the state: " + "; ".join(f"{k} {sel}: {v}".replace("  ", " ") for k, sel, v in
                                          (load_text(l, sess.study) for l in lc.loads)) + ".", size=9)
    for n in d.get("notes", [])[2:]:
        D.p(n, bold=True, size=9)
    if o.images and ms.values:
        p = _render_mode(sess, j, 0, img_dir, o.image_view, warnings)
        if p:
            images.append(str(p))
            D.image(p, f"1st buckling mode: λ = {num(ms.values[0], 4)} (shape normalized, scaled up)")
    D.p("WARNING – LINEAR BUCKLING: the analysis assumes ideal geometry without imperfections (curvature, eccentricity), without plasticity and residual stresses. The actual resistance is usually lower than λ × load (significantly so for slender members and thin shells). For design use the code procedure (EN 1993-1-1, 6.3 – buckling curves) or a geometrically and materially nonlinear analysis with imperfections.", bold=True, size=9,
        color=(0xC0, 0x00, 0x00))


# --------------------------------------------------------------------- spoje

def _status(c) -> str:
    return "PASS" if c.ok else ("FAIL" if math.isfinite(c.utilization) else "NOT ASSESSED")


def _joints_section(D: _Doc, a, o: ReportOptions) -> None:
    D.p("Bolts: VDI 2230 (preload, working stress, bearing pressure under head, thread stripping, fatigue σ_ASV) and EN 1993-1-8 (shear, tension, combination, punching, bearing, slip). Welds: EN 1993-1-8 directional method, fatigue EN 1993-1-9 / IIW (weld root and toe). Loads from load cases are taken as design loads. Utilization ≤ 1 passes; rows marked “(informative)” are not counted in utilization.")
    if a.cases:
        D.p("Assessed states: " + ", ".join(a.cases) + (" (incl. fatigue)" if a.fatigue else "") + ".", size=9)
    for title, lst in (("Bolts", a.bolts), ("Welds", a.welds)):
        if not lst:
            continue
        rows = [[c.name, c.label, num(c.utilization, 3), _status(c), c.governing] for c in lst]
        D.table(["Joint", "Description", "utilization [–]", "rating", "governing check"], rows,
                caption=f"{title} – utilization overview", status_col=3, widths=(0.15, 0.25, 0.1, 0.13, 0.37), font=8)
    if o.joint_details:
        for c in a.all():
            kind = "Bolt" if c.kind == "bolt" else "Weld"
            D.h(c.name if c.name.lower().startswith(kind.lower()) else f"{kind} {c.name}", 3)
            rows = [[n, num(u, 3) if math.isfinite(u) else "–", t] for n, u, t in c.rows]
            D.table(["Assessment", "utilization [–]", "details"], rows, widths=(0.38, 0.1, 0.52), font=8)
            for w in c.warnings:
                D.p(f"Warning: {w}", italic=True, size=8)
    for w in a.warnings:
        D.p(f"Warning: {w}", italic=True, size=9)


# --------------------------------------------------------------------- fatigue

def _fatigue_section(D: _Doc, sess, o: ReportOptions, img_dir, images, warnings) -> None:
    fr = sess.fatigue
    res = sess.results
    D.p("Stress history by superposition of unit load cases σ(t) = Σ f_k(t)·σ_k, rainflow cycle counting (ASTM E1049), linear damage accumulation per Palmgren–Miner. Safety factor = stress multiple at which D = 1 for the specified number of blocks.")
    for line in fr.method_lines():
        D.bullet(line)
    D.kv([("Load cases in history", ", ".join(fr.load_cases)), ("Block time points", str(len(fr.t))),
          ("Block repetitions", num(fr.repeats))])
    rows = []
    for name, info in fr.sn_used.items():
        sn, m = info["sn"], info.get("marin") or {}
        rows.append([name, num(sn["SD"]), num(sn["ND"]), num(sn["k"], 3), num(sn["k2"], 3) if sn.get("k2") else "endurance limit",
                     num(m.get("k"), 3), sn.get("source", "")])
    if rows:
        D.table(["Material", "SD [MPa]", "ND [cycles]", "k", "k₂", "Marin k [–]", "source"], rows,
                caption="S-N curves (amplitude, after corrections)", widths=(0.2, 0.1, 0.1, 0.07, 0.1, 0.1, 0.33), font=8)
        if any("PragTic" in r[-1] for r in rows):
            from ..fatigue import pragtic
            D.p(f"Source of fatigue data: {pragtic.citation()}. {pragtic.NOTICE}", size=8)
    dmax = float(fr.total_damage.max())
    D.kv([("Max. damage D", f"{num(dmax, 3)} – " + ("PASSES (D < 1)" if dmax < 1 else "FAILS (D ≥ 1)")),
          ("Min. life", f"{num(float(fr.life_blocks.min()), 3)} blocks"),
          ("Min. safety factor", num(float(np.min(fr.safety)), 3))])
    if dmax > 0:
        worst = fr.worst(res.mesh, o.fatigue_top)
        rows = [[w["node"] + 1, vec(w["xyz"], 4), num(w["amplitude"]), num(w["total_damage"], 3),
                 num(w["life_blocks"], 3), num(w["safety"], 3)] for w in worst]
        D.table(["node", "location [mm]", "amplitude [MPa]", "Total D", "life [blocks]", "safety factor"], rows,
                caption="Most damaged locations", font=8)
    for w in fr.warnings:
        D.p(f"Warning: {w}", italic=True, size=9)
    if o.images:
        for key in ("fatigue_damage", "fatigue_safety"):          # fatigue does not belong to one state → undeformed
            r = _render_field(sess, 0, key, img_dir, o.image_view, warnings, deformed=False)
            if r:
                images.append(str(r[0]))
                D.image(r[0], FIELD_CAPTION[key])


# --------------------------------------------------------------------- 3D tisk

def _printed_parts(study, res, names: list[str] | None = None) -> list[str]:
    """Parts with printed material (also with the default material of the study)."""
    out = []
    if names is None:
        names = res.mesh.part_names if res is not None else list(study.part_materials)
    for pn in names:
        try:
            if study.material_of(pn).is_print and pn not in study.excluded_parts():
                out.append(pn)
        except KeyError:
            pass
    return out


def _print_required(study) -> float:
    """Required safety of the printing (PrintSettings.required_safety, default 1.5; older studies without the field → 1.5)."""
    v = getattr(study.printing, "required_safety", None)
    return float(v) if v else 1.5


def _printing_section(D: _Doc, sess, printed: list[str], o: ReportOptions, img_dir, images, warnings) -> None:
    from ..printing.infill import HOLLOW, core
    from ..printing.props import family_of, props_of
    study, res = sess.study, sess.results
    req = _print_required(study)
    D.p(f"Printed parts: homogenized model – shell (perimeters, top/bottom layers) and core with effective infill properties, transversely isotropic material with axis 3 along the build direction, anisotropy along/across the perimeter lines; failure criterion {study.printing.criterion.replace('_', '-').title()}. Infill and line properties are INDICATIVE (literature) – calibration by ISO 527 tests is recommended.")
    rows = []
    for pn in printed:
        s = study.printing.spec(pn)
        tol = float(getattr(s, "tolerance", 0.0) or 0.0)
        rows.append([pn, study.material_of(pn).name, vec(s.build_dir, 3), num(s.layer_height, 3), num(s.line_width, 3),
                     s.walls, f"{s.top_layers}/{s.bottom_layers}", f"{num(100 * s.infill, 3)} % {s.pattern}",
                     num(tol, 3) if tol > 0 else "–"])
    D.table(["Part", "Material", "build direction", "layer [mm]", "line [mm]", "perimeters", "top/bottom", "infill",
             "tolerance [mm]"], rows, caption="Print settings", font=8)
    if any(float(getattr(study.printing.spec(pn), "tolerance", 0.0) or 0.0) > 0 for pn in printed):
        D.p("Print tolerance: the part is analyzed smaller by the specified amount on every free face (mesh surface offset inward – undersized walls, rounded bead edges); support and load faces remain at the nominal position.",
            size=8, italic=True)
    # effective properties of the core (infill) and of the shell thickness – what the model actually used
    rows = []
    for pn in printed:
        s, m = study.printing.spec(pn), study.material_of(pn)
        if not s.enabled:
            rows.append([pn, "entire part solid", "–", "–", "–", "–"])
            continue
        if s.infill < HOLLOW:
            rows.append([pn, num(s.wall_thickness, 3), "hollow (0 %)", "–", "–", "–"])
            continue
        c = core(m, s.pattern, s.infill)
        rows.append([pn, num(s.wall_thickness, 3), f"{num(s.top_layers * s.layer_height, 3)} / "
                     f"{num(s.bottom_layers * s.layer_height, 3)}", f"{num(c.E1)} / {num(c.E3)}",
                     f"{num(c.Xt, 3)} / {num(c.Zt, 3)}", num(100 * c.E1 / m.E, 3)])
    D.table(["Part", "wall [mm]", "top / bottom [mm]", "core E_xy / E_z [MPa]", "core Rm_xy / Rm_z [MPa]",
             "core E / solid extrusion [%]"], rows, caption="Shell and effective infill properties (indicative)", font=8)
    # properties of the printing by materials (line, creep, S-N) with the sources
    rows, srcs = [], []
    for mname in sorted({study.material_of(pn).name for pn in printed}):
        m = study.materials.get(mname) or study.material_of(next(p for p in printed if study.material_of(p).name == mname))
        pp = props_of(study, m)
        own = "overridden in study" if study.printing.materials.get(mname) else f"default {family_of(mname)}"
        rows.append([mname, own, f"{num(pp.line_E_ratio, 3)} / {num(pp.line_R_ratio, 3)}",
                     f"{num(pp.creep_a, 3)} / {num(pp.creep_n, 3)}", num(pp.rupture_beta, 3),
                     f"k = {num(pp.sn_k, 3)}, k₂ = {num(pp.sn_k2, 3)}"])
        if pp.source:
            srcs.append(f"{mname}: {pp.source}")
    D.table(["Material", "values", "line across/along E / Rm", "Findley creep a / n", "strength reduction β", "S-N"],
            rows, caption="Print properties (printing.props)", font=8)
    for t in srcs:
        D.p(t, italic=True, size=8)
    tests = study.printing.tests or []
    if tests:
        rows = [[t.get("name", ""), "XY" if t.get("direction") != "z" else "Z", num(t.get("E")), num(t.get("Rm"), 3),
                 f"{num(t.get('width'), 3)} × {num(t.get('thickness'), 3)}", t.get("walls"),
                 f"{num(100 * float(t.get('infill', 1)), 3)} %"] for t in tests if t.get("enabled", True)]
        D.table(["Body", "direction", "E [MPa]", "Rm [MPa]", "cross-section [mm]", "perimeters", "infill"], rows,
                caption="ISO 527 calibration tests (entered in the study)", font=8)
    info = (res.info.get("printing") or {}) if res is not None else {}
    ps = info.get("parts") or {}
    if ps:
        rows = [[pn, num(d["volume_mm3"] / 1e3), num(100 * d["shell_fraction"], 3), num(d["mass_g"])]
                for pn, d in ps.items()]
        D.table(["Part", "volume [cm³]", "shell fraction [%]", "mass [g]"], rows, caption="Printed parts – volume and mass")
    if res is not None:
        rows = []
        for st in res.steps:
            for pn, d in ((st.printing or {}).get("parts") or {}).items():
                sig = d.get("stress_local")
                rows.append([st.name, pn, num(d["min_safety"], 3), d["region"], vec(d["at"], 4),
                             "–" if sig is None else vec(sig, 3),
                             "PASS" if d["min_safety"] >= req else "FAIL"])
        if rows:
            D.table(["State", "Part", "min. safety factor [–]", "region", "location [mm]",
                     "σ11 σ22 σ33 τ12 τ13 τ23 [MPa] (print / line axes)", f"rating (k ≥ {num(req, 3)})"], rows,
                    caption="3D printing – safety factor (load multiple to failure)", status_col=6, font=8)
            D.p("Axes: 1 = along the line (for perimeters) or print axis 1, 2 = transverse in the layer plane, 3 = build direction (layer bond). Large σ33, τ13, τ23 = loading of the layer bond → consider a different build direction.", size=8,
                italic=True)
    _print_variants_section(D, sess, req, warnings)
    _print_advice_section(D, study, res, req, warnings)
    if o.voxel and getattr(sess, "voxel", None) is not None:
        _voxel_section(D, sess, o, img_dir, images, warnings)
    creep = info.get("creep")
    if creep:
        D.p(f"Creep: {num(creep['hours'])} h at {num(creep['temperature'])} °C – {creep.get('method', '')}.")
        rows = [[c["case"], num(c.get("u_short")), num(c.get("u_long")), num(c.get("ratio"), 3),
                 num(c.get("utilization_short"), 3), num(c.get("min_safety_long"), 3)] for c in creep["cases"]]
        D.table(["State", "displacement short-term [mm]", "displacement long-term [mm]", "ratio", "short-term utilization",
                 "min. safety factor long-term"], rows, caption="Long-term loading (creep)", font=8)
    D.p("WARNING – CREEP: the linear model does not capture creep of plastic. For permanently loaded printed parts (especially PLA above ~40 °C) expect creep and a decrease of strength with time"
        + (" – the isochronous-modulus estimate above is indicative only." if creep else
           "; long-term cases were not computed (step 3D printing → Creep)."), bold=True, size=9, color=(0xC0, 0x00, 0x00))


def _variant_settings(o) -> str:
    """Short description of the overrides of the variant (common for all parts once, otherwise by parts)."""
    names = {"walls": "perimeters", "infill": "infill", "pattern": "pattern", "build_dir": "build",
             "top_layers": "top", "bottom_layers": "bottom", "line_width": "line", "layer_height": "layer",
             "enabled": "solid part", "material": "material", "solid_thin": "solid infill of thin regions",
             "thin_width": "thin-region limit", "tolerance": "tolerance [mm]"}

    def one(ov: dict) -> str:
        bits = []
        for k, v in ov.items():
            if k == "infill":
                v = f"{num(100 * float(v), 3)} %"
            elif k == "build_dir":
                v = vec(v, 3)
            elif k == "enabled":
                if v:
                    continue
                bits.append("solid part")
                continue
            bits.append(f"{names.get(k, k)} {v}")
        return ", ".join(bits)
    ov = o.overrides or {}
    if not ov:
        return "study settings"
    texts = {p: one(d) for p, d in ov.items()}
    if len(set(texts.values())) == 1:
        return next(iter(texts.values())) or "–"
    return "; ".join(f"{p}: {t}" for p, t in texts.items())


def _print_variants_section(D: _Doc, sess, req: float, warnings) -> None:
    """Variants of the internal structure / sweep (sess.variants = printing.variants.VariantsRun) – table, Pareto,
    recommendation.
    """
    vr = getattr(sess, "variants", None)
    if vr is None or not getattr(vr, "outcomes", None):
        return
    sweep = getattr(vr, "sweep", None)
    front = set((sweep or {}).get("pareto") or [])
    if not sweep:
        from ..printing.variants import pareto
        front = set(pareto(vr.outcomes))
    kind = "print settings sweep" if sweep else "print internal structure variants"
    D.p(f"Comparison: {kind} – {len(vr.outcomes)} analyses on one mesh ({vr.mesh_info.get('nodes', '–')} nodes); only the shell and infill assignment differs, the rest of the setup is identical. Mass = printed parts (shell + infill). Pareto = no other variant is both lighter and safer.")
    rows = []
    for o in vr.outcomes:
        g = o.governing or {}
        if not o.ok:
            rows.append([o.name, _variant_settings(o), "–", "–", "–", "–", "", "ERROR: " + (o.error or "")[:80]])
            continue
        rows.append([o.name + (" *" if o.name == vr.recommended else ""), _variant_settings(o),
                     num(o.mass_g) if o.mass_g is not None else "–", num(o.min_safety, 3),
                     f"{g.get('part', '')}: {g.get('region', '')}", num(o.max_disp),
                     "yes" if o.name in front else "", "PASS" if o.min_safety >= req else "FAIL"])
    D.table(["Variant", "settings", "mass [g]", "min. safety factor", "governing", "max. displacement [mm]", "Pareto",
             f"k ≥ {num(req, 3)}"], rows, caption="3D printing – variant comparison (* = recommended)", status_col=7, font=7)
    if sweep:
        d = sweep.get("definition") or {}
        bits = []
        if d.get("walls"):
            bits.append("perimetry " + ", ".join(str(w) for w in d["walls"]))
        if d.get("infill"):
            bits.append("infill " + ", ".join(f"{num(100 * x, 3)} %" for x in d["infill"]))
        if d.get("patterns"):
            bits.append("vzory " + ", ".join(d["patterns"]))
        if d.get("build_dirs"):
            bits.append("build directions " + ", ".join(d["build_dirs"]))
        D.p(f"Sweep ({', '.join(sweep.get('parts') or [])}): {'; '.join(bits)}.", size=9)
    if vr.reason:
        D.p("Recommendation: " + vr.reason, bold=True, size=9)
    D.p("The variant choice is INDICATIVE (homogenized infill model) – verify the recommended settings by testing.",
        italic=True, size=8)


def _voxel_section(D: _Doc, sess, o: ReportOptions, img_dir: Path, images, warnings) -> None:
    """Voxel detail of the printing (printing.voxel): setting of the region, voxels by kinds, safety of the detail against the homogenisation, difference of the average stress, sections by layers, warnings (indicative, stair-step singularities)."""
    from ..mcp.tools_voxel import critical_layer, voxel_summary
    from ..printing.voxel import render_png
    vr = sess.voxel
    d = voxel_summary(vr)
    spec = getattr(sess, "voxel_spec", None) or {}
    D.h("Print voxel detail", 3)
    D.p(f"Local submodel made of beads around the critical location: toolpaths of perimeters, top/bottom layers (±45°), infill and solid thin regions generated from the print settings in voxels, C3D8I elements, full-extrusion material in bead axes (1 = line, 2 = transverse, 3 = build); boundary conditions = displacements of the homogenized solution on the cut faces of the region and at supports / contacts. Criterion {str(d.get('criterion') or '').replace('_', '-').title()} at integration points.", size=9)
    auto = d.get("auto") or {}
    rows = [("Part", str(d.get("part"))),
            ("Region center", f"{vec(d.get('center_mm') or (0, 0, 0), 5)} mm"
             + (f" (automatic – min. print safety factor {num(auto.get('safety'), 3)}, case “{auto.get('step')}”)"
                if auto else "")),
            ("Evaluation region (print axes)", f"{vec(d.get('size_mm') or (0, 0, 0), 3)} mm, margin {num(spec.get('margin', 1.0), 3)} mm"),
            ("Voxel", f"{vec(d.get('voxel_mm') or (0, 0, 0), 3)} mm (resolution {spec.get('resolution', 'w/2')}), grid {' × '.join((str(v) for v in d.get('shape') or []))}"),
            ("Model", f"{d.get('voxels')} voxels, {d.get('dof')} equations, element {d.get('element')}, {d.get('bc_nodes')} nodes with boundary condition, {num(d.get('time_s'))} s")]
    D.kv(rows)
    kinds = d.get("kinds") or {}
    if kinds:
        tot = max(sum(kinds.values()), 1)
        D.table(["Voxel type", "count", "fraction [%]"],
                [[k, v, num(100 * v / tot, 3)] for k, v in kinds.items()],
                caption="Voxels by type (region including margin)", widths=(0.5, 0.25, 0.25), font=8)
    rows = []
    for r in d["steps"]:
        rows.append([r["step"], num(r.get("safety_avg"), 3), num(r.get("safety_p1"), 3), num(r.get("min_safety_ip"), 3),
                     num(r.get("homogenized_min_safety"), 3), num(r.get("ratio_avg_to_homogenized"), 3),
                     num(100 * r["mean_stress_difference"], 3) if r.get("mean_stress_difference") is not None else "–"])
    D.table(["State", "safety_avg [–]", "safety_p1 [–]", "min. at int. point [–]", "homogenization [–]",
             "avg / homog. ratio", "avg. stress difference [%]"], rows,
            caption="Voxel detail – safety factor vs. the homogenized model in the same region", font=8)
    D.p("safety_avg = minimum safety factor from the element-average stress, safety_p1 = 1st percentile of the safety factor at integration points of the evaluation region – the key values. Average stress difference = |σ̄_voxel − σ̄_homog| / |σ̄_homog| in the evaluation region: a small difference (<~ 10 %) means the homogenization (infill and perimeter calibration) matches the detail.", size=8, italic=True)
    worst = min(d["steps"], key=lambda r: r.get("safety_avg") or float("inf"), default=None)
    if worst and worst.get("stress_local_MPa"):
        D.p(f"Location of the minimum at an integration point (load case “{worst['step']}”): {vec(worst.get('at_mm') or (0, 0, 0), 4)} mm, {worst.get('kind')}, line angle {num(worst.get('angle_deg'), 3)}°, σ11 σ22 σ33 τ12 τ13 τ23 = {vec(worst['stress_local_MPa'], 3)} MPa.", size=8)
    for w in vr.warnings:
        D.p(f"Warning: {w}", italic=True, size=8)
    if o.images:
        try:
            z = critical_layer(vr)
            img_dir.mkdir(parents=True, exist_ok=True)
            for field, cap in (("kind", "voxel types (perimeter, layer, infill, thin region)"),
                               ("safety", "safety factor in elements (minimum over integration points)")):
                pth = img_dir / f"voxel_{field}.png"
                render_png(vr, pth, field, z, 0, lines=field == "kind")
                images.append(str(pth))
                D.image(pth, f"Voxel detail – {cap}, section through critical layer {z}")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Voxel detail plot: {e}")
    D.p("WARNING – THE VOXEL DETAIL IS INDICATIVE ONLY: toolpaths are generated from the print settings (not from a slicer), bead bonds are perfect, the material is full extrusion. The minimum safety factor at an integration point typically lies at a voxel step or a bead corner – a local (stair-step) singularity that keeps decreasing with refinement; use safety_avg and safety_p1 for evaluation, not the minimum itself.", bold=True, size=9, color=(0xC0, 0x00, 0x00))


def _print_advice_section(D: _Doc, study, res, req: float, warnings) -> None:
    """Advice according to the place of failure (printing.advice) – indicative."""
    if res is None or not any((s.printing or {}).get("parts") for s in res.steps):
        return
    try:
        from ..printing.advice import advise
        items = advise(study, res, req)
    except Exception as e:  # noqa: BLE001 – the advice must not bring down the report
        warnings.append(f"Print advice was not generated: {e}")
        return
    items = [i for i in items if i["kind"] not in ("ok",)]
    if not items:
        return
    D.p("Advice based on the failure location (INDICATIVE, to be verified by analyzing a variant and by testing):", bold=True, size=9)
    for it in items:
        D.bullet(f"{it['part']} – {it['title']}: {it['text']} (Rationale: {it['reason']}.)")


# --------------------------------------------------------------------- assumptions

def _assumptions(D: _Doc, study, res, printed, o: ReportOptions) -> None:
    items = []
    from . import frame as FR
    from . import reduced as RD
    if study.frame.active:
        items += FR.assumptions(study)
    if RD.mode(study):
        items += RD.assumptions(study)
    if FR.frame_only(study) or RD.mode(study):
        pass
    elif study.load_cases and study.is_nonlinear():
        has_plastic = any(getattr(m, "is_plastic", False) for m in study.materials.values())
        items += [
            ("Nonlinear statics: elastoplastic material (von Mises yield condition, hardening curve – see the Plasticity chapter), "
             if has_plastic else "Statics with contact: linear elastic material (Hooke's law), ")
            + ("large displacements and rotations (NLGEOM)" if study.contact_settings.nlgeom else "small displacements and strains")
            + ("; material plasticity is considered." if has_plastic else "; plasticity is not considered."),
            "Parts in contact pairs interact through contact surfaces (opening, friction); the other assembly parts are connected by shared nodes on the mating faces.",
            "Nodal stresses are extrapolated from integration points and averaged; at sharp internal corners, support edges and under point forces they are theoretically infinite (singularities) – evaluate away from these locations.",
            f"Statics is assessed by the safety factor to yield strength Re/σ_vM, required safety factor {num(o.required_safety, 3)}.",
        ]
    elif study.load_cases or not (study.thermal.cases or study.modal.enabled or study.dynamics.enabled):
        items += [
            "Linear statics: linear elastic material (Hooke's law), small displacements and strains; plasticity, contact" + (", nonlinear loss of stability" if study.buckling.enabled else ", loss of stability")
            + (" and forced vibration" if study.modal.enabled and not study.dynamics.enabled else
               "" if study.dynamics.enabled else " and dynamics") + " are not considered.",
            "Assembly parts are connected by shared nodes on mating faces (no contact, no friction, no separation).",
            "Nodal stresses are extrapolated from integration points and averaged; at sharp internal corners, support edges and under point forces they are theoretically infinite (singularities) – evaluate away from these locations.",
            f"Statics is assessed by the safety factor to yield strength Re/σ_vM, required safety factor {num(o.required_safety, 3)}.",
        ]
    if study.thermal.cases:
        items += [
            "Heat conduction is linear (Fourier's law); thermal conductivity, density and specific heat capacity are independent of temperature; latent heat (phase change) is not considered.",
            ("Parts with a shared mesh are in perfect thermal contact (no contact resistance); contact pair interfaces do not transfer heat (insulated); bolts replaced by a beam do not conduct heat." if study.active_contacts()
             else "Assembly parts are in perfect thermal contact (shared nodes, no thermal contact resistance); bolts replaced by a beam do not conduct heat."),
            "Faces without a boundary condition are perfectly insulated. Heat transfer uses a specified constant coefficient h; radiation only to the surroundings (no surface-to-surface radiation, no shadowing). The coefficients h and emissivities are estimates – the main source of result uncertainty.",
        ]
        if any(c.transient for c in study.thermal.cases):
            items.append("Transient case: implicit time integration with a fixed step, results in saved frames; heat flux is stored only at the end of the case.")
    if study.has_thermal_loads():
        items.append(f"Thermal stress: strain α·(T − T_ref), T_ref = {num(study.thermal.reference_temperature)} °C, α and Young's modulus independent of temperature; the temperature field is taken from the thermal case without feedback (deformation does not affect temperature).")
    if study.active_bolts():
        items.append("Bolts are replaced by beams with stiffness per VDI 2230, rigidly connected to the bearing faces; the thread and head contact are not modeled.")
    if printed:
        items.append("Printed parts: homogenized orthotropic model, indicative properties; creep – see the 3D printing chapter.")
    if study.modal.enabled:
        items.append("Natural frequencies: undamped vibration, linear, without the effect of prestress from loads; idealised stiffness of the supports (a fixed support is stiffer than the real support → higher frequencies), mass "
                     + ("from the model parts and added masses" if study.masses else "only from the model parts (no added masses)")
                     + ". Mode shapes are normalised – the displacements in them have no physical magnitude.")
    if study.buckling.enabled:
        items.append("Buckling is LINEAR (eigenvalue problem): ideal geometry without imperfections, without plasticity and without residual stresses – actual capacity is typically lower than λ × load; for design use standard code procedure (EN 1993-1-1, 6.3) or nonlinear analysis with imperfections. Solid-element linear buckling includes shear flexibility (closer to Engesser's theory); differences of ~1–2% between codes are expected.")
    has_kinematic = any(
        isinstance(getattr(m, "plastic", None), dict)
        and getattr(m, "plastic", {}).get("hardening", "").lower() == "kinematic"
        for m in study.materials.values()
    )
    if has_kinematic:
        items.append("Kinematic hardening: large deformations (NLGEOM) were enabled for correct directional hardening formulation (Bauschinger effect) in CalculiX; without NLGEOM the calculation behaves isotropically or diverges.")
    if any(getattr(c, "kind", None) == "elastic" for c in study.constraints):
        items.append("Elastic support (Winkler foundation): independent linear springs at the face nodes, carrying both tension and compression (the foundation does not lift off), without shear coupling between adjacent foundation points.")
    if any(getattr(m, "ortho", None) for m in study.materials.values()):
        items.append("Orthotropic material: linear elastic in the material axes; the safety factor to Re based on von Mises stress does not apply to it.")
    if study.mesh.separate:
        items.append("Separately meshed parts are connected by a *TIE constraint (slave face displacements interpolated from the master); stress immediately at the non-conforming mesh interface is less accurate.")
    from . import f8 as F8
    items += F8.assumptions(study, res)
    for it in items:
        D.bullet(it)
    if res is not None and res.warnings:
        D.p("Warnings from the analysis:", bold=True)
        for w in res.warnings:
            D.bullet(w)
