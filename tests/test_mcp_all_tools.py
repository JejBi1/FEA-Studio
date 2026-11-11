"""Automatic test of ALL MCP tools (F10 B2): no tool may fail with an internal error.

The F9 review showed that the errors of the glue tool ↔ core (wrong arguments, missing attributes) were silently swallowed –
`_Registrar` converts every exception into a ToolError with the text "TypeError: …", so the LLM sees it only as an "error".
The test therefore classifies every call:

  * success (the answer must serialise to JSON, images are PNG),
  * an understandable error `FeaError` (a Czech sentence without the name of the exception class) – fine for an invalid input,
  * INTERNAL ERROR (the text starts with the name of the exception class: TypeError, AttributeError, KeyError, NameError…) = FAILURE of the test.

Coverage:
  1. `test_every_tool_is_covered` (fast) – every registered tool is either in some scenario, or in SKIP
     with a reason (a new tool cannot be forgotten).
  2. `test_fuzz_empty_session` (fast) – every tool with minimal arguments derived from the JSON schema
     (required parameters: empty string / 1 / empty list…) without a study; it must return an understandable error or success.
  3. `test_fuzz_loaded_study` – the same over a loaded study (without mesh; heavy tools in FUZZ_HEAVY left out).
  4. Scenarios with verified, meaningful arguments (the table of tools in the scenarios themselves): without calculation (fast),
     with calculation (slow): statics, frame system, 2D, shells, heat, printing, voxels, contacts, dynamics, mechanisms.

Run:  python -m pytest tests/test_mcp_all_tools.py -q -p no:cacheprovider            (all)
           python -m pytest tests/test_mcp_all_tools.py -q -m "not slow"                  (fast part)
           python -m pytest tests/test_mcp_all_tools.py -q -m slow                        (calculations)
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from fea_core.mcp.server import build_server  # noqa: E402
from fea_core.mcp.session import Session  # noqa: E402

HERE = Path(__file__).resolve()

# ===================================================================== tools that are not called (with the reason)
# A tool is in SKIP only when it cannot reasonably be called on a small model in the test. These tools still go through the
# fuzz test (empty session / loaded study with minimal arguments), so a wrong wiring is caught.
SKIP: dict[str, str] = {
}

# tools left out of the fuzz test over a loaded study (long calculations / writing files outside the working folder)
FUZZ_HEAVY = {
    "fea_solve_run", "fea_job_start", "fea_mesh_generate", "fea_mesh_convergence", "fea_print_sweep",
    "fea_print_variants_run", "fea_print_rank_build_direction", "fea_print_refine", "fea_print_voxel_run",
    "fea_topo_opt_run", "fea_motion_run", "fea_fatigue_run", "fea_report_word", "fea_motion_check_dof",
    "fea_results_export", "fea_param_study_run", "fea_submodel_run",
}

# KNOWN BUGS in files that could not be fixed while the test was prepared (another agent is editing them): (tool, regular expression of the
# error text) → description and place of the fix. The corresponding call is not counted as a failure, but it is recorded in ToolTester.known
# (after the fix delete the record from the table; pytest -rA prints how many times it applied).
KNOWN_BUGS: list[tuple[str, str, str]] = [
]

# internal error = the text of the ToolError starts with the name of the exception class (FeaError carries only a Czech sentence)
INTERNAL_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning|Exit|Interrupt)\b[:(]?")
# in the fuzz test with nonsensical arguments a ValueError / OSError from the libraries is also tolerated (number from the text, file missing)
FUZZ_TOLERATED = ("ValueError", "FileNotFoundError", "NotADirectoryError", "IsADirectoryError", "PermissionError",
                  "OSError")
SDK_VALIDATION_RE = re.compile(r"validation errors? for|Input should be|Field required", re.I)
SDK_PREFIX_RE = re.compile(r"^Error executing tool \S+?: ")      # the SDK puts a prefix before the text of the ToolError


# ===================================================================== helpers

class Out:
    """Result of one tool call."""

    def __init__(self, tool: str, args: dict, res):
        self.tool = tool
        self.args = args
        self.is_error = bool(res.is_error)
        self.contents = res.content
        self.text = "\n".join(getattr(c, "text", "") or "" for c in res.content if getattr(c, "type", "") == "text")
        self.images = [c for c in res.content if getattr(c, "type", "") == "image"]
        if self.is_error:
            self.text = SDK_PREFIX_RE.sub("", self.text)
        self.structured = getattr(res, "structured_content", None)
        self.data = None
        if not self.is_error:
            try:
                self.data = json.loads(self.text) if self.text.strip() else {}
            except json.JSONDecodeError:
                self.data = None

    @property
    def internal(self) -> bool:
        return self.is_error and bool(INTERNAL_RE.match(self.text))

    @property
    def schema_reject(self) -> bool:
        return self.is_error and bool(SDK_VALIDATION_RE.search(self.text)) and not self.internal


def is_known_bug(o: "Out") -> bool:
    return any(re.fullmatch(tools, o.tool) and re.search(rx, o.text or "") for tools, rx, _ in KNOWN_BUGS)


class ToolTester:
    """Calls the tools through the MCP protocol (SDK client in memory) and collects the failures (the test continues so that everything is seen)."""

    def __init__(self, client: Client, called: set[str]):
        self.c = client
        self.called = called
        self.failures: list[str] = []
        self.notes: list[str] = []
        self.known: list[str] = []

    async def call(self, tool: str, args: dict) -> Out:
        self.called.add(tool)
        res = await self.c.call_tool(tool, args)
        return Out(tool, args, res)

    def _fail(self, o: Out, why: str) -> None:
        for tools, rx, desc in KNOWN_BUGS:
            if re.fullmatch(tools, o.tool) and re.search(rx, o.text or ""):
                self.known.append(f"{o.tool}: {desc}")
                return
        a = json.dumps(o.args, ensure_ascii=False, default=str)
        self.failures.append(f"{o.tool}({a[:300]}): {why}: {(o.text or '')[:400]}")

    def _check_payload(self, o: Out) -> None:
        """A successful answer must be serialisable (text = JSON or plain text, image = valid PNG)."""
        if o.text.strip() and o.data is None and (o.text.lstrip()[:1] in "{["):
            self._fail(o, "odpověď vypadá jako JSON, ale nejde načíst")
        for im in o.images:
            import base64
            raw = base64.b64decode(im.data)
            if im.mime_type == "image/png" and raw[:8] != b"\x89PNG\r\n\x1a\n":
                self._fail(o, "obrázek není PNG")
        if o.structured is not None:
            try:
                json.dumps(o.structured)
            except TypeError as e:
                self._fail(o, f"structured_content nejde serializovat ({e})")

    async def ok(self, tool: str, **args):
        """The call MUST succeed. Returns the loaded JSON ({} on failure, so that the scenario can continue)."""
        o = await self.call(tool, args)
        if o.is_error:
            self._fail(o, "VNITŘNÍ CHYBA" if o.internal else ("odmítnutí schématem" if o.schema_reject else "čekal se úspěch"))
            return {}
        self._check_payload(o)
        return o.data if isinstance(o.data, dict) else ({"__list__": o.data} if o.data is not None else {"__text__": o.text})

    async def raw(self, tool: str, **args) -> Out:
        """As ok(), but returns the whole Out (images)."""
        o = await self.call(tool, args)
        if o.is_error:
            self._fail(o, "VNITŘNÍ CHYBA" if o.internal else "čekal se úspěch")
        else:
            self._check_payload(o)
        return o

    async def fea(self, tool: str, expect: str = "", **args) -> str:
        """The call MUST end with an understandable error FeaError (optionally containing `expect`)."""
        o = await self.call(tool, args)
        if not o.is_error:
            self._fail(o, "čekala se chyba, ale volání uspělo")
            return ""
        if o.internal:
            self._fail(o, "VNITŘNÍ CHYBA místo srozumitelné chyby")
        elif expect and expect.lower() not in o.text.lower():
            self._fail(o, f"chyba neobsahuje „{expect}“")
        return o.text

    async def maybe(self, tool: str, **args):
        """Success OR an understandable error (it may depend on the environment); an internal error is a failure. → (ok, data/text)."""
        o = await self.call(tool, args)
        if o.internal or o.schema_reject:
            self._fail(o, "VNITŘNÍ CHYBA" if o.internal else "odmítnutí schématem")
            return False, o.text
        if o.is_error:
            return False, o.text
        self._check_payload(o)
        return True, (o.data if o.data is not None else o.text)

    def done(self) -> None:
        if self.known:
            known = "; ".join(sorted(set(self.known)))
            print("ZNÁMÉ CHYBY uplatněné: " + known)
        assert not self.failures, "\n".join(["", *self.failures])


CALLED: set[str] = set()                  # all tools called during the whole run (coverage check)
RAN: set[str] = set()                     # which scenarios finished


def run_scenario(name, tmp_path, body, session: Session | None = None):
    """Runs the scenario `body(T, sess)` over a fresh server (and session); a failure is reported only after the scenario ends."""
    sess = session or Session(tmp_path / "mcp")
    holder: list[ToolTester] = []

    async def main():
        srv = build_server(sess)
        async with Client(srv, mode="legacy") as c:
            T = ToolTester(c, CALLED)
            holder.append(T)
            await body(T, sess)
    anyio.run(main)
    holder[0].done()
    RAN.add(name)
    return sess


@pytest.fixture(scope="module", autouse=True)
def _no_gui(tmp_path_factory):
    """The tools must not be redirected to a possibly running GUI window (Dynamic IPC Bridge)."""
    mp = pytest.MonkeyPatch()
    mp.setattr("fea_core.ipc.is_gui_running", lambda *a, **k: False)
    yield
    mp.undo()


# ===================================================================== geometrie

@pytest.fixture(scope="module")
def geo(tmp_path_factory):
    """STEP files: plate with a hole, assembly of three blocks, cantilever V1, wire model + bar, two-part beam."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.GC import GC_MakeArcOfCircle
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS_Compound
    from OCP.gp import gp_Pnt
    from verification.cases import box, cyl, solid
    from fea_core.geometry import Part, write_step
    d = tmp_path_factory.mktemp("all_tools_geo")
    g = {}
    plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 60, 40, 5), cyl((30, 20, -1), (0, 0, 1), 5, 7)).Shape())
    write_step([Part("Deska", plate)], d / "plate.step")
    g["plate"] = d / "plate.step"
    plate2 = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 60, 40, 6), cyl((30, 20, -1), (0, 0, 1), 6, 8)).Shape())
    write_step([Part("Deska", plate2)], d / "plate2.step")       # modified model (hole Ø12, thickness 6)
    g["plate2"] = d / "plate2.step"
    write_step([Part("A", box(0, 0, 0, 20, 20, 10)), Part("B", box(0, 0, 10, 20, 20, 10)),
                Part("C", box(0, 0, 20.5, 20, 20, 5))], d / "blocks.step")
    g["blocks"] = d / "blocks.step"
    write_step([Part("Konzola", box(0, 0, 0, 200, 20, 10))], d / "beam.step")
    g["beam"] = d / "beam.step"
    write_step([Part("Tyc", box(0, 0, 0, 100, 10, 10))], d / "bar.step")
    g["bar"] = d / "bar.step"
    write_step([Part("A", box(0, 0, 0, 100, 10, 10)), Part("B", box(100, 0, 0, 100, 10, 10))], d / "split.step")
    g["split"] = d / "split.step"
    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    for p, q in (((0, 0, 0), (0, 0, 1000)), ((0, 0, 1000), (1000, 0, 1000))):
        b.Add(comp, BRepBuilderAPI_MakeEdge(gp_Pnt(*p), gp_Pnt(*q)).Edge())
    arc = GC_MakeArcOfCircle(gp_Pnt(1000, 0, 1000), gp_Pnt(1500, 0, 1500), gp_Pnt(2000, 0, 1000)).Value()
    b.Add(comp, BRepBuilderAPI_MakeEdge(arc).Edge())
    w = STEPControl_Writer()
    w.Transfer(comp, STEPControl_AsIs)
    w.Write(str(d / "wires.step"))
    g["wires"] = d / "wires.step"
    write_step([Part("Kostka", box(0, 0, 0, 100, 100, 20)), Part("Tyc", box(40, 45, 20, 20, 10, 500))], d / "rod.step")
    g["rod"] = d / "rod.step"
    from verification.cyclic_cases import ring_sector       # 45° sector of a ring (cyclic symmetry, N = 8)
    import math as _m
    write_step([Part("Sektor", solid(ring_sector(20.0, 60.0, 0.0, 8.0, _m.pi / 4)))], d / "sector.step")
    g["sector"] = d / "sector.step"
    return g


# ===================================================================== automatic arguments from the schema

def _first_branch(sch: dict) -> dict:
    if "anyOf" in sch:
        for b in sch["anyOf"]:
            if b.get("type") != "null":
                return b
        return {"type": "null"}
    return sch


def auto_value(sch: dict):
    """Minimal value according to the JSON schema: string "x", number 1, empty list / object."""
    sch = _first_branch(sch)
    if "enum" in sch:
        return sch["enum"][0]
    t = sch.get("type")
    return {"string": "x", "number": 1.0, "integer": 1, "boolean": False, "array": [], "object": {}, "null": None}.get(t, "x")


def auto_args(tool) -> dict:
    props = tool.input_schema.get("properties", {}) if hasattr(tool, "input_schema") else {}
    return {k: auto_value(props[k]) for k in tool.input_schema.get("required", [])}


async def list_tools(c):
    """All tools except the tool-list management (fea_tools_*, fea_tool_help): those are tested in test_mcp_groups.py."""
    from fea_core.mcp.groups import META_TOOLS
    return [t for t in (await c.list_tools()).tools if t.name not in META_TOOLS]


# ===================================================================== meta test of the coverage

def _string_constants(node) -> set[str]:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and re.fullmatch(r"fea_[a-z0-9_]+", n.value)}


def exercised_tools() -> set[str]:
    """Tools that the test explicitly calls: string constants `fea_*` anywhere outside the tables SKIP, FUZZ_HEAVY and
    PARAM_FUZZ_SKIP (those only enumerate the tools that are not called / left out of the fuzz pass).
    """
    tree = ast.parse(HERE.read_text(encoding="utf-8"))
    tables = {"SKIP", "FUZZ_HEAVY", "PARAM_FUZZ_SKIP"}

    def is_table(n) -> bool:
        if isinstance(n, ast.AnnAssign):
            return getattr(n.target, "id", "") in tables
        return isinstance(n, ast.Assign) and any(getattr(t, "id", "") in tables for t in n.targets)
    found: set[str] = set()
    for n in tree.body:
        if not is_table(n):
            found |= _string_constants(n)
    return found


def registered_tools() -> set[str]:
    async def main():
        srv = build_server(Session())
        async with Client(srv, mode="legacy") as c:
            return {t.name for t in await list_tools(c)}
    return anyio.run(main)


def test_every_tool_is_covered():
    """Every registered tool is in a scenario, or in SKIP with a reason; a tool that is called must not be in SKIP."""
    reg = registered_tools()
    used = exercised_tools()
    missing = sorted(reg - used - set(SKIP))
    assert not missing, f"nástroje bez testu (přidat do scénáře, nebo do SKIP s důvodem): {missing}"
    stale = sorted(set(SKIP) - reg)
    assert not stale, f"SKIP obsahuje neexistující nástroje: {stale}"
    both = sorted(set(SKIP) & used)
    assert not both, f"nástroj je současně ve SKIP a ve scénáři: {both}"
    for k, why in SKIP.items():
        assert len(why) > 15, f"SKIP[{k}] nemá důvod"
    unknown = sorted(used - reg)
    assert not unknown, f"scénáře volají neexistující nástroje: {unknown}"
    assert len(reg) >= 180


# ===================================================================== fuzz: minimal arguments from the schema

def test_fuzz_empty_session(tmp_path, monkeypatch):
    """Every tool without a study with minimal arguments: an understandable error, or success; never an internal error."""
    monkeypatch.chdir(tmp_path)                                   # a possible "x" written ends in the temporary folder

    async def body(T, sess):
        bad = []
        for t in sorted(await list_tools(T.c), key=lambda t: t.name):
            o = await T.call(t.name, auto_args(t))
            if o.is_error and (o.internal and not o.text.startswith(FUZZ_TOLERATED) or o.schema_reject)                     and not is_known_bug(o):
                bad.append(f"{t.name}{auto_args(t)}: {o.text[:300]}")
            elif not o.is_error:
                T._check_payload(o)
        T.failures += bad
    run_scenario("fuzz_empty", tmp_path, body)


def test_fuzz_loaded_study(tmp_path, monkeypatch, geo):
    """The same over a loaded study without mesh and results (the tools get past the check "no study")."""
    monkeypatch.chdir(tmp_path)

    async def body(T, sess):
        await T.ok("fea_study_new", step_path=str(geo["plate"]), workdir=str(tmp_path / "wd"), material="Ocel S235")
        await T.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
        await T.ok("fea_load_add_force", case_name="F", name="konec", selection={"x": 60}, force=[100, 0, 0])
        bad = []
        for t in sorted(await list_tools(T.c), key=lambda t: t.name):
            if t.name in FUZZ_HEAVY or t.name in ("fea_study_new", "fea_study_open"):
                continue
            o = await T.call(t.name, auto_args(t))
            if o.is_error and (o.internal and not o.text.startswith(FUZZ_TOLERATED) or o.schema_reject)                     and not is_known_bug(o):
                bad.append(f"{t.name}{auto_args(t)}: {o.text[:300]}")
            elif not o.is_error:
                T._check_payload(o)
        T.failures += bad
    run_scenario("fuzz_loaded", tmp_path, body)


# ===================================================================== scenario: tools without a study
# (calculators, sections, analytic beams, GUI control in headless mode)

def calc_params(fn) -> dict:
    """Minimal parameters of the calculator: required (without a default value) = 10.0 (a positive number)."""
    from fea_core.calc.common import parameters
    import inspect
    return {p.name: 10.0 for p in parameters(fn) if p.default is inspect.Parameter.empty}


async def scen_no_study(T: ToolTester, sess: Session, tmp_path: Path):
    from fea_core.calc.common import kinds
    # --- calculators: every kind of every group at least once (success, or an understandable error of a physical input)
    for tool, group in (("fea_calc_hertz", "hertz"), ("fea_calc_friction", "friction"), ("fea_calc_mechanism", "mechanism"),
                        ("fea_calc_convection", "convection"), ("fea_calc_damping", "damping"), ("fea_calc_drop", "pulse")):
        for kind, (fn, _) in kinds(group).items():
            await T.maybe(tool, kind=kind, params=calc_params(fn))
    await T.ok("fea_calc_hertz", kind="sphere", params={"F": 100, "R1": 10})
    await T.ok("fea_calc_friction", kind="journal", params={"F": 100, "r": 10, "mu": 0.1})
    await T.ok("fea_calc_mechanism", kind="pendulum", params={"L": 500, "theta0_deg": 30})
    await T.ok("fea_calc_convection", kind="vertical_plate", params={"L_mm": 200})
    await T.ok("fea_calc_damping", kind="convert", params={"zeta": 0.02})
    await T.ok("fea_calc_drop", params={"h": 500, "tau": 0.005})
    await T.fea("fea_calc_hertz", kind="neexistuje")
    await T.ok("fea_calc_snapfit", l_mm=10, h_mm=1.5, b_mm=5, E_MPa=2500, y_mm=1.0)
    await T.ok("fea_calc_pressfit", d_mm=20, D_mm=40, L_mm=30, delta_mm=0.03)
    lib = await T.ok("fea_calc_friction_library")
    name = lib["pairs"][0]["name"]
    await T.ok("fea_calc_friction_library", action="get", name=name)
    await T.ok("fea_calc_friction_library", action="mu", name=name, purpose="resistance")
    await T.ok("fea_calc_friction_library", action="suggest", material_a="steel", material_b="PLA")
    await T.fea("fea_calc_friction_library", action="get", name="neexistuje")

    # --- sections, analytic beams, EN 1993
    await T.ok("fea_section_search", text="IPE 2")
    await T.ok("fea_section_get", section="IPE 200")
    await T.ok("fea_section_get", section={"shape": "RECT", "b": 20, "h": 40})
    await T.ok("fea_beam_analytic", length_mm=1000, supports="simple", section="IPE 200",
               loads=[{"type": "force", "P_N": 1000, "x_mm": 500}])
    await T.ok("fea_beam_analytic", spans_mm=[1000, 1000], supports=["pinned", "pinned"], I_mm4=1e6,
               loads=[{"type": "distributed", "q_N_mm": 1.0, "from_mm": 0, "to_mm": 2000}])
    await T.ok("fea_beam_buckling", length_mm=2000, section="IPE 200", case="pinned-pinned", fy_MPa=235)
    await T.ok("fea_curved_beam", profile={"shape": "rect", "b": 10, "h": 20}, M_Nmm=1e5, r_inner_mm=20)
    await T.ok("fea_curved_beam", kind="ring_pinch", F_N=100, R_mm=50, I_mm4=1000)
    await T.ok("fea_curved_beam", kind="arc_cantilever", R_mm=100, theta_deg=90, Ft_N=10, I_mm4=1000)
    await T.ok("fea_ec3_member_check", section="IPE 200", length_mm=3000, grade="S235",
               forces={"N_N": -50000, "My_Nmm": 2e7, "Vz_N": 5000})

    # --- GUI in headless mode: information without an error
    st = await T.ok("fea_gui_status")
    assert st["gui_running"] is False
    await T.ok("fea_gui_camera_focus", part="*")
    await T.ok("fea_gui_highlight", entities=[])
    await T.ok("fea_gui_show_mode", mode_number=1)
    await T.ok("fea_gui_toast", message="test")
    await T.ok("fea_gui_stage_switch", stage="Síť")

    # --- catalogues without a study
    await T.ok("fea_materials_list")
    await T.ok("fea_dyn_profile_presets")
    await T.ok("fea_print_presets")


def test_scenario_no_study(tmp_path):
    async def body(T, sess):
        await scen_no_study(T, sess, tmp_path)
    run_scenario("no_study", tmp_path, body)


# ===================================================================== scenario: model without calculation
# geometry, selections, materials, constraints, loads, coordinate systems, combinations, heat (specification), mesh (without generation)

ORTHO = {"E1": 140000, "E2": 10000, "E3": 10000, "nu12": 0.3, "nu13": 0.3, "nu23": 0.4,
         "G12": 5000, "G13": 5000, "G23": 3500}


async def scen_model(T: ToolTester, sess: Session, geo, tmp_path: Path):
    wd = str(tmp_path / "wd")
    await T.ok("fea_study_new", step_path=str(geo["plate"]), name="Deska s otvorem", workdir=wd)
    await T.ok("fea_study_status")
    await T.ok("fea_geometry_list_parts")
    left = (await T.ok("fea_geometry_find_faces", x=0))["faces"][0]["id"]
    right = (await T.ok("fea_geometry_find_faces", x=60))["faces"][0]["id"]
    top = (await T.ok("fea_geometry_find_faces", z=5))["faces"][0]["id"]
    hole = (await T.ok("fea_geometry_find_faces", radius=5))["faces"][0]["id"]
    info = await T.ok("fea_geometry_get_face_info", selector=hole)
    await T.ok("fea_selection_create", name="levy", faces=[left])
    await T.ok("fea_selection_create", name="pravy", faces=right)
    await T.ok("fea_selection_create", name="otvor", faces=info["selector"])
    await T.ok("fea_selection_create", name="horni", faces=[top])
    await T.ok("fea_selection_create", name="spodni", faces={"z": 0})
    await T.ok("fea_selection_create", name="predni", faces={"y": 0})
    edges = await T.ok("fea_geometry_find_edges", x=60, z=5)
    await T.ok("fea_selection_create_edges", name="hrana", edges=[edges["edges"][0]["id"]])
    await T.ok("fea_selection_create_points", name="bod", points=[60, 20, 5])
    await T.ok("fea_selection_list")
    await T.ok("fea_selection_create", name="docasny", faces=[left])
    await T.ok("fea_selection_delete", name="docasny")
    await T.fea("fea_selection_delete", name="docasny")

    # --- materials
    await T.ok("fea_materials_list", query="S355")
    await T.ok("fea_materials_assign", part="*", material_name="Ocel S235")
    await T.ok("fea_materials_assign", part="Deska", material_name="Moje ocel", custom_props={
        "E": 200000, "nu": 0.3, "rho": 7.85e-9, "Re": 300, "Rm": 450, "k": 45, "alpha": 12e-6, "cp": 470})
    await T.ok("fea_materials_assign", part="Deska", material_name="Ocel S235")
    await T.ok("fea_material_set_orthotropic", name="CFRP", constants=ORTHO, rho=1.6e-9, Re=600, Rm=800)

    # --- coordinate systems
    await T.ok("fea_csys_create", name="kart", kind="cartesian", origin=[0, 0, 0], z_axis=[0, 0, 1], x_axis=[1, 0, 0])
    await T.ok("fea_csys_create", name="valec", kind="cylindrical", from_face=hole)
    await T.ok("fea_csys_list")

    # --- constraints
    await T.ok("fea_constraint_add_fixed", name="levy", selection="levy")
    await T.ok("fea_constraint_add_displacement", name="sym_y", selection="predni", uy=0.0)
    await T.ok("fea_constraint_add_cylindrical", name="otvor", selection="otvor", radial=True, tangential=False, axial=True)
    await T.ok("fea_constraint_add_remote", name="vzdalena", selection="horni", point=[30, 20, 40], dofs="ux,uy,uz",
               coupling="distributing")
    await T.ok("fea_constraint_add_elastic_support", name="podlozi", selection="spodni", stiffness=10.0)
    await T.ok("fea_constraint_list")
    await T.ok("fea_constraint_remove", name="vzdalena")
    await T.ok("fea_constraint_remove", name="podlozi")
    await T.ok("fea_constraint_remove", name="otvor")
    await T.ok("fea_constraint_remove", name="sym_y")

    # --- loads
    await T.ok("fea_loadcase_create", name="Tah")
    await T.ok("fea_load_add_force", case_name="Tah", name="F_pravy", selection="pravy", force=[1000, 0, 0])
    await T.ok("fea_load_add_force", case_name="Tah", name="F_kart", selection="pravy", force=[1000, 0, 0], csys="kart")
    await T.ok("fea_load_add_force", case_name="Tah", name="F_valec", selection="otvor", force=[0, 100, 0], csys="valec")
    await T.ok("fea_load_add_edge_force", case_name="Tah", name="F_hrana", edges="hrana", force=[0, 0, -50])
    await T.ok("fea_load_add_point_force", case_name="Tah", name="F_bod", points="bod", force=[0, 0, -10])
    await T.ok("fea_load_add_pressure", case_name="Tah", name="p", selection="horni", pressure=1.0,
               gradient=[0.01, 0, 0], origin=[0, 0, 0], clip=True)
    await T.ok("fea_load_add_hydrostatic", case_name="Tah", name="hydro", selection="predni", surface_point=[0, 0, 20])
    await T.ok("fea_load_add_gravity", case_name="Tah", direction=[0, 0, -1])
    await T.ok("fea_load_add_remote", case_name="Tah", name="vzd", selection="horni", point=[30, 20, 40],
               force=[0, 0, -100], moment=[1000, 0, 0])
    await T.ok("fea_load_add_bearing", case_name="Tah", name="lozisko", selection="otvor", force=[0, 500, 0])
    await T.ok("fea_load_add_rotation", case_name="Tah", axis_point=[0, 0, 0], axis_direction=[0, 0, 1], rpm=100)
    await T.ok("fea_load_add_temperature", case_name="Tah", value_C=40.0)
    await T.ok("fea_load_list")
    await T.ok("fea_load_list", case_name="Tah")
    # a non-existing name must not delete anything (it used to delete the self-weight) and ends with an understandable error
    n_before = len((await T.ok("fea_load_list", case_name="Tah"))["load_cases"][0]["loads"])
    await T.fea("fea_load_remove", "does not exist", case_name="Tah", name="neexistujici_zatizeni")
    loads = (await T.ok("fea_load_list", case_name="Tah"))["load_cases"][0]["loads"]
    assert len(loads) == n_before and any(l["kind"] == "gravity" for l in loads)
    for l in loads:
        await T.ok("fea_load_remove", case_name="Tah", name=l.get("name") or l.get("selection") or l["kind"])
    assert (await T.ok("fea_load_list", case_name="Tah"))["load_cases"][0]["loads"] == []
    await T.ok("fea_csys_delete", name="kart")
    await T.ok("fea_csys_delete", name="valec")
    await T.ok("fea_loadcase_create", name="Dalsi")
    await T.ok("fea_load_add_gravity", case_name="Dalsi")
    await T.ok("fea_loadcase_delete", name="Dalsi")

    # --- kombinace EN 1990
    await T.ok("fea_loadcase_create", name="Stale")
    await T.ok("fea_load_add_gravity", case_name="Stale")
    await T.ok("fea_combination_add", name="K1", factors={"Tah": 1.5, "Stale": 1.35})
    await T.ok("fea_combination_generate_en1990", formula="6.10", cases=[
        {"name": "Stale", "action_type": "permanent", "subtype": "self_weight"},
        {"name": "Tah", "action_type": "variable", "category": "B", "leading": True}])
    await T.ok("fea_combination_list")
    await T.ok("fea_combination_delete", name="K1")
    await T.ok("fea_loadcase_delete", name="Stale")

    # --- heat (specification)
    await T.ok("fea_thermal_settings", reference_C=20, initial_C=20)
    await T.ok("fea_thermal_case_create", name="Ohrev")
    await T.ok("fea_thermal_add_temperature", case="Ohrev", selection="levy", value_C=100)
    await T.ok("fea_thermal_add_convection", case="Ohrev", selection="pravy", h_W_m2K=20, ambient_C=20)
    await T.ok("fea_thermal_add_radiation", case="Ohrev", selection="horni", emissivity=0.8)
    await T.ok("fea_thermal_add_heat_flux", case="Ohrev", selection="spodni", flux_W_m2=100)
    await T.ok("fea_thermal_add_heat_generation", case="Ohrev", part="Deska", power_W=1.0)
    await T.ok("fea_thermal_add_cavity", case="Ohrev", selection="spodni", emissivity=0.7, cavity="dutina", closed=False)
    await T.ok("fea_thermal_add_contact_conductance", side_a="levy", side_b="pravy", h_W_m2K=2000.0, name="styk")
    await T.ok("fea_thermal_list")
    await T.ok("fea_thermal_remove", case="", name="styk", kind="contact_conductance")
    await T.ok("fea_thermal_remove", case="Ohrev", name="spodni", kind="cavity_radiation")
    await T.ok("fea_load_add_temperature", case_name="Tah", thermal_case="Ohrev")
    await T.ok("fea_load_remove", case_name="Tah", name="teplota")
    await T.ok("fea_thermal_remove", case="Ohrev", name="spodni")
    await T.ok("fea_thermal_case_create", name="Prechod", transient=True, time_s=10, dt_s=1, outputs=2)
    await T.ok("fea_thermal_case_delete", name="Prechod")
    await T.ok("fea_thermal_case_delete", name="Ohrev")

    # --- imprint, refinement, separately meshed parts (without mesh generation)
    await T.ok("fea_imprint_add", name="otisk", part="Deska", shape="circle", center=[45, 20, 5], normal=[0, 0, 1], radius=4)
    await T.ok("fea_imprint_list")
    await T.ok("fea_imprint_delete", name="otisk")
    await T.ok("fea_mesh_refine_point", center=[30, 20, 2], radius_mm=8, size_mm=1.5)
    await T.ok("fea_mesh_refine_point", remove_all=True)
    await T.ok("fea_mesh_separate_parts", parts=["Deska"], separate=False)
    await T.maybe("fea_mesh_info")                     # without a mesh: an understandable error or an empty overview

    # --- check, image, saving, opening
    await T.ok("fea_check_readiness")
    await T.raw("fea_model_render_image", output_path=str(tmp_path / "model.png"), view="iso", load_case="Tah",
                highlight=["levy"], show_face_ids=True, width=640, height=360)
    assert (tmp_path / "model.png").exists()
    await T.ok("fea_study_set_parts_ignored", parts=["Deska"], ignored=True)
    await T.ok("fea_study_set_parts_ignored", parts=["Deska"], ignored=False)
    await T.ok("fea_study_set_parts_hidden", parts=["Deska"], hidden=True)
    await T.ok("fea_study_set_parts_hidden", parts=["Deska"], hidden=False)
    await T.ok("fea_study_save", path=str(tmp_path / "deska.fea"))
    await T.ok("fea_study_save", path=str(tmp_path / "deska.json"))
    await T.ok("fea_study_open", path=str(tmp_path / "deska.fea"), workdir=str(tmp_path / "wd2"))
    await T.ok("fea_study_open", path=str(tmp_path / "deska.json"), workdir=str(tmp_path / "wd3"))

    # --- geometry update keeping the settings (iteration)
    await T.ok("fea_iterations")                       # none yet
    plan = await T.ok("fea_geometry_replace", step_path=str(geo["plate2"]), dry_run=True)
    assert plan["applied"] is False and plan["report"]["selections"], plan
    await T.fea("fea_geometry_replace", step_path=str(geo["plate2"]), accept="vse")
    done = await T.ok("fea_geometry_replace", step_path=str(geo["plate2"]))
    assert done["applied"] and done["iteration"] == 2, done
    it = await T.ok("fea_iterations")
    assert [r["n"] for r in it["iterations"]] == [1, 2], it


def test_scenario_model(tmp_path, geo):
    async def body(T, sess):
        await scen_model(T, sess, geo, tmp_path)
    run_scenario("model", tmp_path, body)


# ===================================================================== scenario: statics, modes, results (slow)

async def poll_job(T: ToolTester, job_id: str, limit: float = 600.0) -> dict:
    import time
    t0 = time.time()
    while time.time() - t0 < limit:
        st = await T.ok("fea_job_status", job_id=job_id, log_lines=5)
        if st.get("status") not in ("starting", "running"):
            if st.get("status") != "done":
                T.failures.append(f"úloha {job_id} skončila stavem {st.get('status')}: {str(st.get('error'))[:600]}")
            return st
        await anyio.sleep(0.5)
    raise AssertionError(f"úloha {job_id} nedoběhla do {limit} s")


async def scen_static(T: ToolTester, sess: Session, geo, tmp_path: Path):
    await T.ok("fea_study_new", step_path=str(geo["plate"]), name="Deska", workdir=str(tmp_path / "wd"),
               material="Ocel S235")
    await T.ok("fea_selection_create", name="levy", faces={"x": 0})
    await T.ok("fea_constraint_add_fixed", name="levy", selection="levy")
    await T.ok("fea_load_add_force", case_name="Tah", name="pravy", selection={"x": 60}, force=[2000, 0, 0])
    await T.ok("fea_load_add_pressure", case_name="Tlak", name="horni", selection={"z": 5}, pressure=2.0)
    await T.ok("fea_load_add_force", case_name="Tlak2", name="pravy2", selection={"x": 60}, force=[-500, 0, 0])
    await T.ok("fea_modal_set", modes=3, f_max_Hz=50000)
    await T.ok("fea_buckling_set", case_name="Tlak2", modes=2)
    await T.fea("fea_nlbuckling_set", case_name="neni")                 # unknown case → understandable error
    await T.ok("fea_nlbuckling_set", case_name="Tlak2", amplitude_ratio=500, max_factor=2.0, increments=4)
    await T.ok("fea_mesh_generate", global_size=5.0, timeout_s=300)
    await T.ok("fea_mesh_info")
    await T.ok("fea_check_readiness")
    r = await T.ok("fea_solve_run", timeout_s=900)
    assert r.get("ok", True)

    # --- overview, probes, hotspots
    await T.ok("fea_results_summary")
    await T.ok("fea_results_summary", step="Tah")
    await T.ok("fea_results_summary", step=None)
    await T.ok("fea_results_probe", point=[30, 5, 2.5])
    await T.ok("fea_results_probe", point=[[10, 5, 2.5], [50, 35, 2.5]], step="Tah", criterion="tresca", limit_MPa=235)
    await T.ok("fea_results_hotspots", top_n=3, step="Tah")
    await T.ok("fea_results_hotspots", field="displacement", step="Tlak")
    await T.ok("fea_results_section_force", point=[30, 20, 2.5], normal=[1, 0, 0], step=0)
    await T.ok("fea_results_line", p0=[0, 5, 2.5], p1=[60, 5, 2.5], field="mises", step="Tah", n=11)
    await T.ok("fea_results_line", p0=[0, 5, 2.5], p1=[60, 5, 2.5], field="ux", plot=True,
               output_path=str(tmp_path / "line.png"), return_image=False)
    await T.ok("fea_results_linearize", p0=[10, 5, 0], p1=[10, 5, 5], step="Tah", n=11, hint=[1, 0, 0],
               allowable_S_MPa=160)
    await T.fea("fea_plasticity_assess")                          # material without plasticity → understandable error

    # --- images
    await T.raw("fea_results_render_image", output_path=str(tmp_path / "vm.png"), field="mises", step="Tah", width=640, height=360)
    await T.raw("fea_results_render_image", output_path=str(tmp_path / "sec.png"), field="displacement", step="Tlak",
                section_point=[30, 20, 2.5], section_normal=[0, 1, 0], show_mesh=True, bands=8, width=640, height=360)
    await T.raw("fea_results_render_image", field="safety", step="Tah", view="multi", return_image=False,
                output_path=str(tmp_path / "safety.png"), width=640, height=360)

    # --- submodel around the hole (pressure on the top face is transferred into the region)
    await T.ok("fea_submodel_results")                              # none yet
    await T.raw("fea_submodel_run", center=[30, 20, 2.5], size_mm=[20, 20, 7], mesh_size_mm=1.5, step="Tlak",
                     return_image=False)
    await T.ok("fea_submodel_run", center="max", shape="sphere", radius_mm=8, mesh_size_mm=1.5, step="Tah",
               return_image=False)
    await T.ok("fea_submodel_results", index=0)
    await T.raw("fea_submodel_render", field="s1", output_path=str(tmp_path / "sub.png"), width=640, height=360)
    await T.fea("fea_submodel_render", field="temperature")
    await T.ok("fea_submodel_delete", index=1)

    # --- modes and buckling
    modes = await T.ok("fea_results_modes")
    assert modes["mode_sets"], modes
    await T.ok("fea_results_modes", kind="buckling")
    await T.raw("fea_results_render_mode", mode=1, kind="modal", output_path=str(tmp_path / "m1.png"), width=640, height=360)
    await T.raw("fea_results_render_mode", mode=1, kind="buckling", field="uz", output_path=str(tmp_path / "b1.png"),
                width=640, height=360)
    nlb = await T.ok("fea_results_nlbuckling")                       # nonlinear buckling with imperfection (F9 C2)
    assert nlb["ultimate_load_factor"] and nlb["curve"]["lambda"], nlb
    await T.raw("fea_results_nlbuckling", plot=True, output_path=str(tmp_path / "nlb.png"))
    await T.raw("fea_results_render_mode", mode=2, kind="nonlinear", output_path=str(tmp_path / "nl2.png"),
                width=640, height=360)

    # --- kombinace + export
    await T.ok("fea_combination_add", name="K1", factors={"Tah": 1.0, "Tlak": 1.0})
    await T.ok("fea_combination_add", name="Obalka", factors={}, envelope=True, envelope_type="max",
               envelope_combinations=["K1"], envelope_field="von_mises")
    await T.ok("fea_results_export", format="all", output_dir=str(tmp_path / "export"))
    await T.ok("fea_results_export", format="csv", output_dir=str(tmp_path / "export2"), step="Tah")

    # --- fatigue, joints, report, other
    await T.ok("fea_fatigue_run", history_curve={"load_case": "Tah", "kind": "sine", "amplitude": 1.0, "mean": 0.0,
                                                 "cycles": 1}, repeats=1e6, top_n=3)
    await T.maybe("fea_convergence_history")                       # linear calculation: no convergence data
    await T.ok("fea_joints_list")
    await T.ok("fea_joints_detect")
    await T.fea("fea_joints_assess", "no active bolts or welds")           # study without joints
    await T.fea("fea_print_results")
    await T.ok("fea_report_word", output_path=str(tmp_path / "zprava.docx"), images=False, title="Test", author="Test")
    await T.ok("fea_study_save", path=str(tmp_path / "deska_vysl.fea"))

    # --- asynchronous job
    await T.fea("fea_job_cancel", "No job")                   # no job has been started yet
    j = await T.ok("fea_job_start", kind="solve")
    st = await poll_job(T, j["job_id"])
    if st.get("status") != "done":
        return
    await T.ok("fea_job_result", job_id=j["job_id"])
    await T.ok("fea_job_cancel", job_id=j["job_id"])

    # --- mesh refinement, convergence, topology optimisation
    await T.ok("fea_mesh_convergence", quantity="max_disp", levels=2, ratio=1.6, timeout_s=900)
    await T.ok("fea_mesh_generate", global_size=5.0)
    await T.ok("fea_topo_opt_run", target_volume_fraction=0.8, filter_radius=4.0, max_iterations=3, timeout_s=900,
               output_stl=str(tmp_path / "topo.stl"))

    # --- parametric study (buckling is enabled → no linear shortcut: one analysis per point; E sweep likewise)
    p = await T.ok("fea_param_study_run", parameters=[{"kind": "load_factor", "values": [1, 2]}], timeout_s=900)
    assert len(p["points"]) == 2 and abs(p["points"][1]["max_disp"] / p["points"][0]["max_disp"] - 2) < 1e-3, p
    p = await T.ok("fea_param_study_run", parameters=[{"kind": "material_prop", "target": "Ocel S235", "field": "E",
                                                       "values": "105000:210000:2"}], timeout_s=900)
    assert p["points"][0]["max_disp"] > 1.9 * p["points"][1]["max_disp"], p
    await T.fea("fea_param_study_run", "does not exist", parameters=[{"kind": "case_factor", "target": "nope", "values": [2]}])
    await T.ok("fea_param_study_results")
    await T.ok("fea_param_study_apply", index=0)


@pytest.mark.slow
def test_scenario_static(tmp_path, geo):
    async def body(T, sess):
        await scen_static(T, sess, geo, tmp_path)
    run_scenario("static", tmp_path, body)


# ===================================================================== scenario: cyclic symmetry (sector of a ring)

def test_scenario_cyclic(tmp_path, geo):
    import math as _m

    async def body(T, sess):
        await T.ok("fea_study_new", step_path=str(geo["sector"]), workdir=str(tmp_path / "wd"), material="Ocel S235")
        await T.fea("fea_cyclic_results")                                          # no results yet
        await T.ok("fea_constraint_add_fixed", name="otvor", selection={"radius": 20})
        await T.ok("fea_load_add_rotation", case_name="w", axis_point=[0, 0, 0], axis_direction=[0, 0, 1], omega=500.0)
        right = {"normal": [-_m.sin(_m.pi / 4), _m.cos(_m.pi / 4), 0]}
        await T.fea("fea_cyclic_set", left={"y": 0}, right={"y": 0})               # the same face twice
        d = await T.ok("fea_cyclic_set", left={"y": 0}, right=right)
        assert d["cyclic"]["sectors"] == 8
        await T.ok("fea_cyclic_get")
        await T.ok("fea_mesh_generate", global_size=8.0)
        await T.ok("fea_solve_run", timeout_s=600)
        r = await T.ok("fea_cyclic_results")
        assert r["states"][0]["cut_force_N"] is not None
        await T.raw("fea_cyclic_render_full", field="mises", output_path=str(tmp_path / "full.png"), width=480, height=320)
        await T.ok("fea_modal_set", modes=2)
        await T.ok("fea_cyclic_set", left="cyclic_left", right="cyclic_right", nodal_diameters=[0, 2])
        await T.ok("fea_solve_run", timeout_s=600)
        r = await T.ok("fea_cyclic_results")
        assert {row["nodal_diameter"] for row in r["frequencies"]} == {0, 2}
        await T.raw("fea_cyclic_render_full", mode=1, output_path=str(tmp_path / "mode.png"), width=480, height=320)
        await T.ok("fea_cyclic_remove")
    run_scenario("cyclic", tmp_path, body)


# ===================================================================== scenario: frame system (without geometry)

async def scen_frame(T: ToolTester, sess: Session, geo, tmp_path: Path, solve: bool):
    await T.ok("fea_study_new", name="Ram", workdir=str(tmp_path / "wd"), material="Ocel S235")
    await T.ok("fea_frame_section_set", name="IPE 200", library="IPE 200")
    await T.ok("fea_frame_section_set", name="R", shape="RECT", dims={"b": 20, "h": 40})
    await T.ok("fea_frame_section_set", name="docasny", library="CHS 76.1x3.6")
    await T.ok("fea_frame_node_set", nodes={"A": [0, 0, 0], "B": [0, 0, 3000], "C": [4000, 0, 3000], "D": [4000, 0, 0],
                                            "E": [2000, 0, 4000]})
    await T.ok("fea_frame_node_set", nodes=[{"name": "T1", "xyz": [6000, 0, 0]}, {"name": "T2", "xyz": [8000, 0, 0]},
                                            {"name": "T3", "xyz": [7000, 0, 1500]}, {"name": "X", "xyz": [9000, 0, 0]}])
    await T.ok("fea_frame_member_set", name="m1", start="A", end="B", section="IPE 200")
    await T.ok("fea_frame_member_set", name="m2", start="D", end="C", section="IPE 200", roll_deg=0, up=[0, 1, 0])
    await T.ok("fea_frame_member_set", name="ob", start="B", end="C", section="IPE 200", through=[2000, 0, 4000],
               releases_start="hinge", releases_end="none")
    await T.ok("fea_frame_member_set", name="m1", segments=0)                         # change of an existing member
    for n, a, b in (("t1", "T1", "T2"), ("t2", "T2", "T3"), ("t3", "T3", "T1")):
        await T.ok("fea_frame_member_set", name=n, start=a, end=b, section="R", kind="truss")
    await T.ok("fea_frame_member_set", name="tmp", start="T2", end="X", section="docasny")
    await T.ok("fea_frame_support_set", node="A", dofs="fixed")
    await T.ok("fea_frame_support_set", node="D", dofs="pinned")
    await T.ok("fea_frame_support_set", node="T1", dofs="pinned")
    await T.ok("fea_frame_support_set", node="T2", dofs=["uy", "uz"])
    await T.ok("fea_frame_support_set", node="T3", dofs=["uy"])
    await T.ok("fea_frame_support_set", node="X", dofs="pinned", stiffness={"uz": 1000.0})
    await T.ok("fea_frame_load_node", case_name="Q", node="E", force=[0, 0, -10.0])        # node without a member (the arc only passes through it)
    await T.ok("fea_frame_remove", kind="node_load", names=["E"], case_name="Q")
    await T.ok("fea_frame_load_node", case_name="Q", node="B", force=[0, 0, -10000.0], moment=[0, 0, 0])
    await T.ok("fea_frame_load_node", case_name="Q", node="T3", force=[0, 0, -1000.0])
    await T.ok("fea_frame_load_member", case_name="Q", member="m2", q=[0, 0, -2.0], q_end=[0, 0, -4.0])
    await T.ok("fea_frame_load_member", case_name="Q", member="ob", q=[0, 0, -3.0], axes="global")
    await T.ok("fea_frame_load_temperature", case_name="Teplota", dT=20.0, grad_z=5.0)
    await T.ok("fea_load_add_gravity", case_name="Tiha")
    await T.ok("fea_frame_settings")
    await T.ok("fea_frame_settings", shear_deformation=True, steel_grade="S355", gamma_M0=1.0, gamma_M1=1.0,
               segments_modal=4, segments_static=1)
    await T.ok("fea_frame_checks_set", member="m1", settings={"Lcr_y": 3000, "Lcr_z": 3000})
    await T.ok("fea_frame_list")
    await T.raw("fea_frame_render", quantity="model", output_path=str(tmp_path / "model.png"), width=640, height=360)
    # removal of elements
    await T.ok("fea_frame_remove", kind="support", names=["X"])
    await T.ok("fea_frame_remove", kind="member", names=["tmp"])
    await T.ok("fea_frame_remove", kind="section", names=["docasny"])
    await T.ok("fea_frame_remove", kind="node", names=["X"])
    await T.fea("fea_frame_remove", kind="section", names=["IPE 200"])            # used by a member
    await T.ok("fea_frame_remove", kind="node_load", names=["T3"], case_name="Q")
    await T.ok("fea_frame_load_node", case_name="Q", node="T3", force=[0, 0, -1000.0])
    await T.ok("fea_check_readiness")
    if not solve:
        return
    await T.ok("fea_modal_set", modes=4)
    await T.ok("fea_buckling_set", case_name="Q", modes=2)
    await T.ok("fea_solve_run", timeout_s=600)
    await T.ok("fea_frame_results")
    await T.ok("fea_frame_results", step="Q", member="m2", points=11)
    await T.ok("fea_frame_assess")
    await T.ok("fea_frame_assess", member="m1", rows=False)
    for q in ("deformed", "My", "N", "Vz", "sigma", "utilization"):
        await T.raw("fea_frame_render", quantity=q, step="Q", output_path=str(tmp_path / f"{q}.png"), width=640, height=360)
    await T.ok("fea_results_modes")
    await T.raw("fea_results_render_mode", mode=1, kind="modal", output_path=str(tmp_path / "mode.png"), width=640, height=360)
    await T.ok("fea_results_summary")
    await T.ok("fea_report_word", output_path=str(tmp_path / "ram.docx"), images=False)
    await T.ok("fea_study_save", path=str(tmp_path / "ram.fea"))


def test_scenario_frame_definition(tmp_path, geo):
    async def body(T, sess):
        await scen_frame(T, sess, geo, tmp_path, solve=False)
    run_scenario("frame_def", tmp_path, body)


@pytest.mark.slow
def test_scenario_frame_solve(tmp_path, geo):
    async def body(T, sess):
        await scen_frame(T, sess, geo, tmp_path, solve=True)
    run_scenario("frame_solve", tmp_path, body)


async def scen_frame_cad(T: ToolTester, sess: Session, geo, tmp_path: Path, solve: bool):
    """Bars from CAD: wire model of a STEP and a prismatic part (a bar welded to a cube) with a connection to the body."""
    await T.ok("fea_study_new", name="Ram z drátů", workdir=str(tmp_path / "wd1"), material="Ocel S235")
    await T.ok("fea_frame_section_set", name="IPE 200", library="IPE 200")
    r = await T.ok("fea_frame_from_wires", section="IPE 200", step_path=str(geo["wires"]), tol_mm=0.01, prefix="d_")
    assert r.get("members") or r.get("n_members") or r, r
    await T.ok("fea_frame_list")
    # mixed model: replace a bar with a member
    await T.ok("fea_study_new", step_path=str(geo["rod"]), name="Tyc na kostce", workdir=str(tmp_path / "wd2"),
               material="Ocel S235")
    await T.ok("fea_frame_from_part", part="Tyc", use_library=False, connect=True)
    await T.ok("fea_frame_list")
    await T.ok("fea_selection_create", name="kostka_dole", faces={"z": 0})
    await T.ok("fea_constraint_add_fixed", name="kostka_dole", selection="kostka_dole")
    nodes = (await T.ok("fea_frame_list"))["nodes"]
    free = max(nodes.items(), key=lambda kv: kv[1][2] if isinstance(kv[1], list) else kv[1]["xyz"][2])[0]
    await T.ok("fea_selection_create", name="kostka_strana", faces={"x": 0, "part": "Kostka"})
    await T.ok("fea_frame_connection_set", node=free, selection="kostka_strana", name="spoj_kostka")
    await T.ok("fea_frame_load_node", case_name="Q", node=free, force=[100.0, 0, 0])
    if solve:
        await T.ok("fea_mesh_generate", global_size=15.0, timeout_s=300)
        await T.ok("fea_solve_run", timeout_s=600)
        await T.ok("fea_frame_results")
        await T.ok("fea_frame_assess")


@pytest.mark.slow
def test_scenario_frame_cad(tmp_path, geo):
    async def body(T, sess):
        await scen_frame_cad(T, sess, geo, tmp_path, solve=False)
    run_scenario("frame_cad", tmp_path, body)


# ===================================================================== scenario: 3D printing (settings, variants, voxels)

PLA = "Tisk PLA Bambu Basic"


async def scen_print(T: ToolTester, sess: Session, geo, tmp_path: Path, solve: bool):
    await T.ok("fea_study_new", step_path=str(geo["bar"]), name="Tyc PLA", workdir=str(tmp_path / "wd"))
    await T.ok("fea_materials_assign", part="*", material_name=PLA)
    await T.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
    await T.ok("fea_load_add_force", case_name="Zatizeni", name="konec", selection={"x": 100}, force=[0, 0, -20])
    await T.ok("fea_print_get")
    await T.ok("fea_print_get", part="Tyc")
    await T.ok("fea_print_set", part="Tyc", settings={"walls": 3, "infill_percent": 30, "pattern": "gyroid",
                                                      "build_dir": [0, 0, 1], "tolerance": 0.0})
    await T.ok("fea_print_set", criterion="tsai_hill")
    await T.ok("fea_print_props", material=PLA, overrides={"line_E_ratio": 0.7, "creep_a": 0.05})
    await T.ok("fea_print_props", material=PLA, reset=True)
    await T.ok("fea_print_creep", enabled=True, hours=1000, temperature=23, cases=["Zatizeni"])
    tests = [{"name": "XY-1", "direction": "xy", "E": 2900, "Rm": 44, "infill": 100},
             {"name": "Z-1", "direction": "z", "E": 2500, "Rm": 30}]
    await T.ok("fea_print_calibrate", tests=tests, base_material=PLA, name="PLA kalibrovane", assign_parts=["*"])
    await T.ok("fea_materials_assign", part="*", material_name=PLA)
    await T.ok("fea_print_creep", enabled=False)
    await T.ok("fea_print_presets")
    await T.ok("fea_print_presets", action="save", name="test_predvolba", from_part="Tyc", note="test")
    await T.ok("fea_print_presets", action="apply", name="test_predvolba", parts=["Tyc"])
    await T.ok("fea_print_presets", action="delete", name="test_predvolba")
    await T.ok("fea_print_variant_set", name="v_walls4", settings={"walls": 4}, note="test")
    await T.ok("fea_print_variant_set", name="v_kopie", copy_from="v_walls4", part_settings={"Tyc": {"infill": 0.5}})
    await T.ok("fea_print_variant_list")
    await T.ok("fea_print_variant_remove", name="v_kopie")
    await T.ok("fea_print_sweep", walls=[2, 3], infill=[0.2], run=False)            # estimate only
    await T.ok("fea_check_readiness")
    if not solve:
        return
    await T.ok("fea_mesh_generate", global_size=4.0, timeout_s=300)
    await T.raw("fea_print_render_shell", axis="y", output_path=str(tmp_path / "shell.png"), width=640, height=360,
                timeout_s=300)
    await T.ok("fea_solve_run", timeout_s=900)
    await T.ok("fea_print_results")
    await T.ok("fea_print_advice")
    await T.raw("fea_results_render_image", field="print_safety", output_path=str(tmp_path / "ps.png"), width=640, height=360)
    await T.ok("fea_print_variants_run", names=["v_walls4"], timeout_s=900)
    await T.ok("fea_print_variant_apply", name="v_walls4")
    await T.ok("fea_print_sweep", walls=[2, 3], infill=[0.2], timeout_s=900, add_recommended=True)
    await T.ok("fea_print_rank_build_direction", directions=["+Z", "+X"], timeout_s=900)
    await T.ok("fea_solve_run", timeout_s=900)
    # voxel detail
    await T.ok("fea_print_voxel_estimate", center=[50, 5, 5], size_mm=[3, 3, 2], margin_mm=0.5)
    await T.raw("fea_print_voxel_run", center=[50, 5, 5], size_mm=[3, 3, 2], margin_mm=0.5, timeout_s=900,
                output_path=str(tmp_path / "vox.png"), return_image=False)
    await T.raw("fea_print_voxel_render", field="safety", output_path=str(tmp_path / "vox2.png"), return_image=False)
    await T.ok("fea_print_refine", n_spots=1, timeout_s=900)
    await T.ok("fea_report_word", output_path=str(tmp_path / "tisk.docx"), images=False)


def test_scenario_print_definition(tmp_path, geo):
    async def body(T, sess):
        await scen_print(T, sess, geo, tmp_path, solve=False)
    run_scenario("print_def", tmp_path, body)


@pytest.mark.slow
def test_scenario_print_solve(tmp_path, geo):
    async def body(T, sess):
        await scen_print(T, sess, geo, tmp_path, solve=True)
    run_scenario("print_solve", tmp_path, body)


# ===================================================================== scenario: 2D section and shells

@pytest.fixture(scope="module")
def reduced_models(tmp_path_factory):
    """Study JSON: A1 (quarter of a tube, internal pressure) and Sh1 (plate 200×40×5 clamped, force at the end)."""
    import verification.cases  # noqa: F401 – order of the imports (twod_cases ↔ cases)
    from verification.shell_cases import sh1_build
    from verification.twod_cases import a1_build
    d = tmp_path_factory.mktemp("all_tools_reduced")
    (d / "a1").mkdir()
    (d / "sh1").mkdir()
    st, _ = a1_build(d / "a1")
    st.twod.enabled = False
    st.save(d / "a1" / "study.json")
    st, _ = sh1_build(d / "sh1")
    st.shell.enabled = False
    st.save(d / "sh1" / "study.json")
    return {"a1": d / "a1" / "study.json", "sh1": d / "sh1" / "study.json"}


async def scen_twod(T: ToolTester, sess: Session, models, tmp_path: Path):
    await T.ok("fea_study_open", path=str(models["a1"]), workdir=str(tmp_path / "wd"))
    await T.ok("fea_twod_set", kind="axisym", origin=[0, 0, 0], axis=[0, 0, 1], xdir=[1, 1, 0], size=1.0)
    await T.ok("fea_solve_run", timeout_s=600)
    await T.ok("fea_twod_results", field="st")
    await T.ok("fea_twod_probe", point=[[20, 10], [14.14, 14.14, 10]])
    await T.ok("fea_twod_line", p0=[20, 10], p1=[40, 10], field="st", n=11)
    await T.ok("fea_twod_linearize", p0=[20, 10], p1=[40, 10])
    await T.raw("fea_twod_render", field="mises", output_path=str(tmp_path / "t.png"), mesh_lines=True)
    await T.raw("fea_twod_render", field="mises", scale=10.0, mirror=False, return_image=False,
                output_path=str(tmp_path / "t2.png"))
    await T.fea("fea_results_render_image", "2D section analysis")
    await T.ok("fea_results_summary", step=None)
    await T.ok("fea_report_word", output_path=str(tmp_path / "a1.docx"), images=False)
    await T.ok("fea_twod_set", enabled=False)


async def scen_shell(T: ToolTester, sess: Session, models, tmp_path: Path):
    await T.ok("fea_study_open", path=str(models["sh1"]), workdir=str(tmp_path / "wd"))
    await T.ok("fea_shell_detect")
    await T.ok("fea_shell_detect", part="Deska")
    await T.ok("fea_shell_set", parts={"Deska": None}, offset="mid")
    await T.ok("fea_check_readiness")
    await T.ok("fea_solve_run", timeout_s=600)
    await T.ok("fea_shell_results", step=0, field="sxx", layer="top")
    await T.ok("fea_shell_probe", point=[100, 20, 2.5], field="sxx", layer="bot")
    await T.raw("fea_shell_render", field="mises", layer="top", deform=True, output_path=str(tmp_path / "s.png"))
    await T.fea("fea_results_render_image", "shell model")
    await T.ok("fea_report_word", output_path=str(tmp_path / "sh1.docx"), images=False)
    await T.ok("fea_shell_set", enabled=False)


def test_scenario_reduced_definition(tmp_path, reduced_models):
    """Settings of 2D and shells without a calculation."""
    async def body(T, sess):
        await T.ok("fea_study_open", path=str(reduced_models["a1"]), workdir=str(tmp_path / "wd"))
        await T.ok("fea_twod_set", kind="plane_stress", origin=[0, 0, 0], axis=[0, 1, 0], xdir=[1, 0, 0], thickness=2.0)
        await T.ok("fea_twod_set", enabled=False)
        await T.ok("fea_study_open", path=str(reduced_models["sh1"]), workdir=str(tmp_path / "wd2"))
        await T.ok("fea_shell_detect")
        await T.ok("fea_shell_set", parts={"Deska": 5.0}, size=5.0, offset="top")
        await T.ok("fea_shell_set", enabled=False)
    run_scenario("reduced_def", tmp_path, body)


@pytest.mark.slow
def test_scenario_twod(tmp_path, reduced_models):
    async def body(T, sess):
        await scen_twod(T, sess, reduced_models, tmp_path)
    run_scenario("twod", tmp_path, body)


@pytest.mark.slow
def test_scenario_shell(tmp_path, reduced_models):
    async def body(T, sess):
        await scen_shell(T, sess, reduced_models, tmp_path)
    run_scenario("shell", tmp_path, body)


# ===================================================================== scenario: contacts (nonlinear), dynamics, mechanisms

async def scen_contact(T: ToolTester, sess: Session, geo, tmp_path: Path, solve: bool):
    await T.ok("fea_study_new", step_path=str(geo["blocks"]), name="Styk", workdir=str(tmp_path / "wd"),
               material="Ocel S235")
    await T.ok("fea_study_set_parts_ignored", parts=["C"], ignored=True)
    await T.ok("fea_constraint_add_fixed", name="dno", selection={"z": 0})
    await T.ok("fea_load_add_pressure", case_name="p", name="vrch", selection={"z": 20}, pressure=10.0)
    d = await T.ok("fea_contact_detect", tol_mm=0.1)
    assert d["pairs"], d
    await T.ok("fea_contact_detect", tol_mm=0.1, apply=True, type="frictional")
    lst = await T.ok("fea_contact_list")
    pair = lst["pairs"][0]["name"]
    await T.ok("fea_contact_set", pair=pair, mu=0.2, stiffness=1e6, purpose="slip")
    await T.ok("fea_contact_set", pair=pair, reset=["mu"], purpose="resistance")
    await T.ok("fea_contact_set", pair="podpora", type="compression_only", a={"part": "A", "z": 0})
    await T.ok("fea_contact_remove", pair="podpora")
    await T.ok("fea_contact_settings")
    await T.ok("fea_contact_settings", method="surface_to_surface", nlgeom=False, stabilize="none")
    await T.ok("fea_analysis_set_sequence", kind="nonlinear", branches=False,
               steps=[{"name": "zatížení", "case": "p", "scale": 1.0}])
    await T.ok("fea_contact_check")
    await T.raw("fea_model_render_image", output_path=str(tmp_path / "model.png"), view="iso", width=640, height=360)
    await T.fea("fea_contact_results", "results available")
    await T.fea("fea_results_history", "results available")
    if not solve:
        await T.ok("fea_contact_remove", pair="*")
        await T.ok("fea_analysis_set_sequence", kind="linear")
        return
    await T.ok("fea_mesh_generate", global_size=5.0, timeout_s=300)
    await T.ok("fea_contact_check")
    j = await T.ok("fea_job_start", kind="solve")
    st = await poll_job(T, j["job_id"], 900)
    if st.get("status") != "done":
        return
    await T.ok("fea_job_result", job_id=j["job_id"])
    await T.ok("fea_contact_results")
    await T.ok("fea_contact_results", step=0, pair=pair)
    for f in ("tlak", "stav", "mezera", "prokluz", "treni"):
        await T.raw("fea_contact_render", field=f, output_path=str(tmp_path / f"{f}.png"), width=640, height=360)
    await T.raw("fea_contact_render", field="tlak", side="master", deformed=True, show_mesh=True, bands=6,
                output_path=str(tmp_path / "t2.png"), width=640, height=360)
    await T.ok("fea_results_history")
    await T.raw("fea_results_history", plot=True, output_path=str(tmp_path / "hist.png"), return_image=False)
    await T.maybe("fea_convergence_history")
    await T.ok("fea_results_summary")
    await T.ok("fea_report_word", output_path=str(tmp_path / "styk.docx"), images=False)


def test_scenario_contact_definition(tmp_path, geo):
    async def body(T, sess):
        await scen_contact(T, sess, geo, tmp_path, solve=False)
    run_scenario("contact_def", tmp_path, body)


@pytest.mark.slow
def test_scenario_contact_solve(tmp_path, geo):
    async def body(T, sess):
        await scen_contact(T, sess, geo, tmp_path, solve=True)
    run_scenario("contact_solve", tmp_path, body)


async def scen_dynamics(T: ToolTester, sess: Session, geo, tmp_path: Path, solve: bool):
    await T.ok("fea_study_new", step_path=str(geo["bar"]), name="Dynamics", workdir=str(tmp_path / "wd"),
               material="Ocel S235")
    await T.ok("fea_constraint_add_fixed", name="vetknuti", selection={"x": 0})
    await T.ok("fea_load_add_force", case_name="F", name="konec", selection={"x": 100}, force=[0, 0, 100])
    await T.ok("fea_dyn_settings")
    await T.ok("fea_dyn_settings", enabled=True, modes=8, f_max_Hz=30000, verify_ccx=False, max_frames=20)
    await T.ok("fea_dyn_damping_set", kind="modal", zeta=0.02)
    await T.ok("fea_dyn_damping_set", kind="rayleigh", zeta=0.03, f1_Hz=50, f2_Hz=500)
    await T.ok("fea_dyn_damping_set", kind="structural", eta=0.04)
    await T.ok("fea_dyn_damping_set", kind="modal", zeta=0.02, table=[[10, 0.01], [10000, 0.03]])
    await T.ok("fea_dyn_damping_set", kind="modal", zeta=0.02, per_material={"Ocel S235": 0.01})
    await T.ok("fea_dyn_damping_set", kind="modal", zeta=0.02)
    await T.ok("fea_dyn_probe_add", name="konec", point=[100, 5, 5])
    await T.ok("fea_dyn_probe_add", name="celo", selection={"x": 100}, quantity="A")
    await T.ok("fea_dyn_probe_list")
    await T.ok("fea_dyn_probe_remove", name="celo")
    await T.ok("fea_dyn_mass_add", name="motor", selection={"x": 100}, mass_kg=0.05, inertia_kg_mm2=[1, 1, 1])
    await T.ok("fea_dyn_mass_add", name="DPS", selection={"y": 0}, mass_per_area_kg_m2=1.0)
    await T.ok("fea_dyn_mass_list")
    await T.ok("fea_dyn_mass_remove", name="DPS")
    await T.ok("fea_dyn_harmonic_set", f_min_Hz=10, f_max_Hz=3000, points=300, base_acc_g=1.0, dof="z")
    await T.ok("fea_dyn_harmonic_set", sine_profile={"displacement_mm": 0.075, "acceleration_g": 1.0}, dof="z",
               sweep_rate_oct_min=1.0, sweeps=1, static_correction=True)
    await T.ok("fea_dyn_transient_set", t_end_s=0.004, dt_s=5e-6, base_acc_g=10, dof=3,
               pulse={"shape": "halfsine", "tau_s": 0.001}, settle_tol_mm=0.001)
    await T.ok("fea_dyn_random_set", points=[[20, None], [80, 0.04], [350, 0.04], [2000, None]],
               slopes_db_oct=[3, 0, -6], dof="z", duration_h=0.5, method="dirlik")
    await T.ok("fea_dyn_shock_set", pulse="halfsine", A_g=50, tau_s=0.011, dof=3, combine="srss", missing_mass=True,
               transient=True)
    await T.ok("fea_dyn_prestress_set", enabled=False)
    await T.fea("fea_dyn_prestress_set", "does not exist", case="neni")
    await T.ok("fea_dyn_profile_presets", name="IEC 60068-2-64")
    await T.raw("fea_model_render_image", output_path=str(tmp_path / "model.png"), view="iso", width=640, height=360)
    await T.ok("fea_check_readiness")
    await T.fea("fea_dyn_results", "")
    await T.fea("fea_dyn_frf_get", "")
    await T.fea("fea_dyn_plot", "")
    await T.fea("fea_dyn_render", "")
    if not solve:
        return
    await T.ok("fea_mesh_generate", global_size=5.0, timeout_s=300)
    j = await T.ok("fea_job_start")
    st = await poll_job(T, j["job_id"], 1200)
    if st.get("status") != "done":
        return
    await T.ok("fea_job_result")
    await T.ok("fea_dyn_results")
    for kind in ("harmonic", "transient", "random", "shock"):
        await T.ok("fea_dyn_frf_get", probe="konec", kind=kind, quantity="U", max_points=100)
    await T.ok("fea_dyn_frf_get", kind="harmonic", quantity="A", reaction=True)
    await T.ok("fea_dyn_frf_get", kind="harmonic", quantity="S")
    for kind in ("frf", "transient", "psd", "asd", "srs", "pulse", "shock_time", "reaction"):
        await T.raw("fea_dyn_plot", kind=kind, probe="konec", output_path=str(tmp_path / f"{kind}.png"), return_image=False)
    await T.raw("fea_dyn_plot", kind="frf", probe="konec", quantity="A", component="mag", output_path=str(tmp_path / "frf2.png"))
    for field in ("harmonic_envelope", "harmonic_at", "transient_envelope", "transient_at", "rms", "3sigma", "damage",
                  "life", "sweep_damage", "shock", "mode"):
        await T.raw("fea_dyn_render", field=field, output_path=str(tmp_path / f"r_{field}.png"), width=480, height=270,
                    return_image=False)
    await T.raw("fea_dyn_render", field="harmonic_at", quantity="displacement", frequency_Hz=500.0,
                output_path=str(tmp_path / "r_h2.png"), width=480, height=270)
    await T.ok("fea_report_word", output_path=str(tmp_path / "dyn.docx"), images=False)
    await T.ok("fea_dyn_probe_remove", name="*")
    await T.ok("fea_dyn_mass_remove", name="*")


def test_scenario_dynamics_definition(tmp_path, geo):
    async def body(T, sess):
        await scen_dynamics(T, sess, geo, tmp_path, solve=False)
    run_scenario("dynamics_def", tmp_path, body)


@pytest.mark.slow
def test_scenario_dynamics_solve(tmp_path, geo):
    async def body(T, sess):
        await scen_dynamics(T, sess, geo, tmp_path, solve=True)
    run_scenario("dynamics_solve", tmp_path, body)


@pytest.fixture(scope="module")
def lever_step(tmp_path_factory):
    pytest.importorskip("exudyn")
    from fea_core.geometry import write_step
    from tests.test_motion import _pin_parts
    p = tmp_path_factory.mktemp("all_tools_motion") / "paka.step"
    write_step(_pin_parts(), p)
    return p


async def scen_motion(T: ToolTester, sess: Session, lever_step, tmp_path: Path):
    await T.ok("fea_study_new", step_path=str(lever_step), name="Paka", workdir=str(tmp_path / "wd"),
               material="Ocel S235")
    await T.ok("fea_motion_body_add", name="rám", parts=["Deska"], ground=True)
    await T.ok("fea_motion_body_add", name="páka", parts=["Páka"])
    await T.ok("fea_motion_body_list")
    await T.ok("fea_motion_joints_from_cad", accept=False)             # proposal only
    await T.ok("fea_motion_joint_add", name="čep", kind="revolute", body_a="rám", body_b="páka",
               faces_b={"part": "Páka", "type": "cylinder", "radius": 5})
    await T.ok("fea_motion_joint_remove", name="čep")
    await T.ok("fea_motion_joints_from_cad", accept=True)
    await T.ok("fea_motion_joint_list")
    await T.ok("fea_motion_force_add", name="pruzina", kind="spring", body_a="rám", body_b="páka", point_a_mm=[0, 0, 0],
               point_b_mm=[50, 0, 0], k=1.0, c=0.0, f0=0.0)
    await T.ok("fea_motion_load_add", body="páka", point_mm=[50, 0, 0], force_N=[0, -1.0, 0], name="zatizeni")
    await T.ok("fea_motion_check_dof")
    await T.ok("fea_motion_settings", kind="dynamic", t_end_s=0.1, outputs=20, gravity_mm_s2=[0, -9810, 0])
    await T.fea("fea_motion_results", "")                              # without a calculation
    r = await T.ok("fea_motion_run", timeout_s=300)
    jn = r["joints"][0]["joint"]
    await T.ok("fea_motion_results", series=[f"reakce:{jn}", "energie"], find_max=f"reakce:{jn}", time_s=0.05)
    await T.raw("fea_motion_render", quantity=f"reakce:{jn}", output_path=str(tmp_path / "r.png"), width=640, height=360)
    await T.raw("fea_motion_render", quantity="poloha", time_s=0.1, output_path=str(tmp_path / "p.png"), return_image=False)
    await T.fea("fea_motion_export_loads", "has no face", mode="time", times_s=[0.05], body="páka")   # spring without a face of the attachment
    await T.ok("fea_motion_remove", kind="force", name="pruzina")
    await T.ok("fea_motion_remove", kind="load", name="zatizeni")
    await T.ok("fea_motion_run", timeout_s=300)                      # a change of the mechanism discarded the result
    await T.ok("fea_motion_export_loads", mode="time", times_s=[0.05], body="páka")
    await T.ok("fea_motion_export_loads", mode="max", quantity=f"reakce:{jn}", body="páka")
    await T.ok("fea_report_word", output_path=str(tmp_path / "paka.docx"), images=False)
    # drive, contact (definition only), removal
    await T.ok("fea_motion_driver_add", joint=jn, quantity="angular_velocity", law="constant", data=[2.0], name="motor")
    await T.ok("fea_motion_settings", kind="kinematic", t_end_s=0.5, outputs=10)
    await T.ok("fea_motion_run", timeout_s=300)
    await T.maybe("fea_motion_contact_add", body_a="rám", body_b="páka", mu=0.2, deflection_mm=0.05, name="contact")
    await T.maybe("fea_motion_remove", kind="contact", name="contact")
    await T.ok("fea_motion_remove", kind="driver", name="motor")
    await T.ok("fea_motion_joint_remove", name=jn)
    await T.ok("fea_motion_body_remove", name="páka", cascade=True)


@pytest.mark.slow
def test_scenario_motion(tmp_path, lever_step):
    async def body(T, sess):
        await scen_motion(T, sess, lever_step, tmp_path)
    run_scenario("motion", tmp_path, body)


# ===================================================================== scenario: heat (calculation), joints with bolts

async def scen_thermal(T: ToolTester, sess: Session, geo, tmp_path: Path):
    steel = {"E": 210000, "nu": 0.3, "rho": 7.85e-9, "Re": 235, "Rm": 360, "k": 50.0, "alpha": 12e-6, "cp": 470.0}
    await T.ok("fea_study_new", step_path=str(geo["bar"]), name="Ohrev tyce", workdir=str(tmp_path / "wd"))
    await T.ok("fea_materials_assign", part="*", material_name="Ocel (teplo)", custom_props=steel)
    await T.ok("fea_thermal_settings", reference_C=20)
    await T.ok("fea_thermal_case_create", name="Ohrev")
    await T.ok("fea_thermal_add_temperature", case="Ohrev", selection={"x": 0}, value_C=100, name="horke_celo")
    await T.ok("fea_thermal_add_convection", case="Ohrev", selection={"x": 100}, h_W_m2K=500, ambient_C=20, name="chlazeni")
    await T.ok("fea_thermal_case_create", name="Prechod", transient=True, time_s=20, dt_s=2, outputs=2)
    await T.ok("fea_thermal_add_temperature", case="Prechod", selection="horke_celo", value_C=100)
    await T.ok("fea_thermal_add_heat_generation", case="Prechod", part="Tyc", power_W=1.0)
    await T.ok("fea_thermal_add_heat_flux", case="Prechod", selection="chlazeni", power_W=-0.5)
    await T.ok("fea_constraint_add_displacement", name="sym_x", selection="horke_celo", ux=0.0)
    await T.ok("fea_constraint_add_displacement", name="sym_y", selection={"y": 0}, uy=0.0)
    await T.ok("fea_constraint_add_displacement", name="sym_z", selection={"z": 0}, uz=0.0)
    await T.ok("fea_load_add_temperature", case_name="Teplotni napeti", thermal_case="Ohrev")
    await T.ok("fea_check_readiness")
    await T.raw("fea_model_render_image", output_path=str(tmp_path / "model.png"), thermal_case="Ohrev", width=640, height=360)
    await T.ok("fea_mesh_generate", global_size=5.0, timeout_s=300)
    await T.ok("fea_solve_run", timeout_s=900)
    await T.ok("fea_results_summary", thermal_case="Ohrev")
    await T.ok("fea_results_summary", thermal_case=None, step=None)
    await T.ok("fea_results_probe", point=[[50, 5, 5], [90, 5, 5]], thermal_case="Ohrev")
    await T.ok("fea_results_probe", point=[50, 5, 5], thermal_case="Prechod", time_s=10.0)
    await T.ok("fea_results_hotspots", field="temperature", thermal_case="Ohrev")
    await T.ok("fea_results_hotspots", field="heat_flux", thermal_case="Ohrev")
    await T.raw("fea_results_render_image", field="temperature", thermal_case="Ohrev", output_path=str(tmp_path / "T.png"),
                width=640, height=360)
    await T.raw("fea_results_render_image", field="heat_flux", thermal_case="Prechod", time_s=20.0,
                output_path=str(tmp_path / "q.png"), width=640, height=360)
    await T.raw("fea_results_render_image", field="temperature", step="Teplotni napeti", output_path=str(tmp_path / "Tm.png"),
                width=640, height=360)
    await T.ok("fea_results_line", p0=[0, 5, 5], p1=[100, 5, 5], field="temperature", thermal_case="Ohrev", n=11)
    await T.ok("fea_report_word", output_path=str(tmp_path / "teplo.docx"), images=False)
    await T.ok("fea_thermal_remove", case="Prechod", name="horke_celo", kind="fixed_temperature")
    await T.fea("fea_thermal_case_delete", "temperature load", name="Ohrev")             # the mechanical state takes the temperature from it


@pytest.mark.slow
def test_scenario_thermal(tmp_path, geo):
    async def body(T, sess):
        await scen_thermal(T, sess, geo, tmp_path)
    run_scenario("thermal", tmp_path, body)


async def scen_joints(T: ToolTester, sess: Session, tmp_path: Path):
    from fea_core.geometry import write_step
    from verification.joint_models import lap_joint
    step = tmp_path / "lap.step"
    write_step(lap_joint(), step)
    await T.ok("fea_study_new", step_path=str(step), workdir=str(tmp_path / "wd"), material="Ocel S235")
    d = await T.ok("fea_joints_detect", apply=False)
    d = await T.ok("fea_joints_detect", accept_shape=False)
    assert len(d["bolts"]) == 1, d
    name = d["bolts"][0]["name"]
    await T.ok("fea_joints_list")
    await T.ok("fea_joints_update", name=name, changes={"preload_mode": "force", "preload": 15000})
    await T.ok("fea_joints_update", name="*settings*", changes={"category": "B", "gamma_M2": 1.25})
    await T.ok("fea_constraint_add_fixed", name="B_konec", selection={"part": "Deska B", "x": 160})
    await T.ok("fea_load_add_force", case_name="Tah", name="A_konec", selection={"part": "Deska A", "x": 0},
               force=[-5000, 0, 0])
    await T.ok("fea_mesh_generate", global_size=4.0, timeout_s=300)
    await T.ok("fea_solve_run", timeout_s=900)
    await T.ok("fea_joints_assess")
    await T.ok("fea_results_summary")
    await T.ok("fea_report_word", output_path=str(tmp_path / "spoj.docx"), images=False)


@pytest.mark.slow
def test_scenario_joints(tmp_path):
    async def body(T, sess):
        await scen_joints(T, sess, tmp_path)
    run_scenario("joints", tmp_path, body)


# ===================================================================== fuzz: each optional parameter separately over a calculated study

# tools that are not called in this pass (long calculations, starting processes, writing files outside the working folder)
PARAM_FUZZ_SKIP = FUZZ_HEAVY | {"fea_study_new", "fea_study_open", "fea_study_save", "fea_job_start", "fea_job_cancel",
                                "fea_print_voxel_estimate"}
# tools with an image: only minimal calls and a few parameters (rasterisation is slow)
RENDER_WORDS = ("render", "fea_dyn_plot")


def snapshot(sess: Session) -> dict:
    """State of the session before the call (the study by a deep copy – the tools change it in place)."""
    import copy
    snap = dict(sess.__dict__)
    snap["study"] = copy.deepcopy(sess.study)
    return snap


def restore(sess: Session, snap: dict) -> None:
    import copy
    sess.__dict__.update(snap)
    sess.study = copy.deepcopy(snap["study"])


async def solved_plate(T: ToolTester, geo, tmp_path: Path) -> None:
    await T.ok("fea_study_new", step_path=str(geo["plate"]), name="Deska", workdir=str(tmp_path / "wd"),
               material="Ocel S235")
    await T.ok("fea_selection_create", name="levy", faces={"x": 0})
    await T.ok("fea_constraint_add_fixed", name="levy", selection="levy")
    await T.ok("fea_load_add_force", case_name="Tah", name="pravy", selection={"x": 60}, force=[2000, 0, 0])
    await T.ok("fea_load_add_force", case_name="Tlak", name="pravy2", selection={"x": 60}, force=[-500, 0, 0])
    await T.ok("fea_modal_set", modes=3)
    await T.ok("fea_buckling_set", case_name="Tlak", modes=2)
    await T.ok("fea_mesh_generate", global_size=8.0, timeout_s=300)
    await T.ok("fea_solve_run", timeout_s=900)


async def scen_fuzz_params(T: ToolTester, sess: Session, geo, tmp_path: Path):
    await solved_plate(T, geo, tmp_path)
    bad: list[str] = []
    n = n_ok = 0
    tools_ok: set[str] = set()
    for t in sorted(await list_tools(T.c), key=lambda t: t.name):
        if t.name in PARAM_FUZZ_SKIP:
            continue
        props = t.input_schema.get("properties", {})
        req = set(t.input_schema.get("required", []))
        base = auto_args(t)
        opt = [p for p in props if p not in req]
        if any(w in t.name for w in RENDER_WORDS):
            opt = opt[:3]
        for args in [base] + [{**base, p: auto_value(props[p])} for p in opt]:
            snap = snapshot(sess)
            o = await T.call(t.name, args)
            restore(sess, snap)
            n += 1
            n_ok += not o.is_error
            tools_ok.add(t.name) if not o.is_error else None
            if o.is_error and (o.internal and not o.text.startswith(FUZZ_TOLERATED) or o.schema_reject)                     and not is_known_bug(o):
                bad.append(f"{t.name}({json.dumps(args, ensure_ascii=False)}): {o.text[:300]}")
            elif not o.is_error:
                T._check_payload(o)
    print(f"fuzz parametrů: {n} volání, z toho {n_ok} úspěšných ({len(tools_ok)} nástrojů aspoň jednou)")
    assert n_ok > n // 5, "většina volání skončila chybou – stav relace se asi nevrací"
    T.failures += bad


@pytest.mark.slow
def test_fuzz_params_on_solved_study(tmp_path, monkeypatch, geo):
    """Every tool with every optional parameter separately (value from the schema) over a study with results: apart from an input error
    an internal error must not occur. The state of the session is returned after each call (the tools change the study in place).
    """
    monkeypatch.chdir(tmp_path)

    async def body(T, sess):
        await scen_fuzz_params(T, sess, geo, tmp_path)
    run_scenario("fuzz_params", tmp_path, body)


# ===================================================================== final check: what was really called

ALL_SCENARIOS = {"fuzz_empty", "fuzz_loaded", "no_study", "model", "static", "frame_def", "frame_solve", "frame_cad",
                 "print_def", "print_solve", "reduced_def", "twod", "shell", "contact_def", "contact_solve",
                 "dynamics_def", "dynamics_solve", "motion", "thermal", "joints", "fuzz_params"}


def test_zz_every_declared_tool_was_really_called():
    """After ALL scenarios have finished (the whole file), every tool named in the scenarios really was called – a scenario
    cannot merely name a tool (e.g. behind a condition that is not met). When a part of the tests is selected, this is skipped.
    """
    if not ALL_SCENARIOS <= RAN:
        pytest.skip(f"neběžely všechny scénáře (chybí {sorted(ALL_SCENARIOS - RAN)})")
    not_called = sorted(exercised_tools() - CALLED)
    assert not not_called, f"nástroje uvedené v testu, které se nikdy nezavolaly: {not_called}"
