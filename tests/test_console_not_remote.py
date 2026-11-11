"""The Python console of the GUI (fea_gui.console_panel) runs code in the program – it must not be reachable from IPC / MCP / REST.
Static checks of the sources (no window): the IPC server only dispatches to MainWindow.execute_mcp_action, which neither knows the
console nor evaluates code, and no tool module or the REST layer refers to the console."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("console", "ConsolePanel", "run_code", "exec(", "eval(", "compile(")


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_ipc_server_does_not_reach_console():
    src = _src("fea_gui/ipc_server.py")
    for word in FORBIDDEN:
        assert word not in src, f"fea_gui/ipc_server.py refers to {word!r}"
    tree = ast.parse(src)
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Attribute) and getattr(n.func.value, "attr", "") == "main_window"}
    assert calls <= {"execute_mcp_action"}, calls                  # the only entry into the window


def test_execute_mcp_action_has_no_console_or_eval():
    tree = ast.parse(_src("fea_gui/main.py"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute_mcp_action")
    seg = ast.get_source_segment(_src("fea_gui/main.py"), fn)
    for word in FORBIDDEN:
        assert word not in seg, f"execute_mcp_action refers to {word!r}"
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert not names & {"console_panel", "dock_console", "exec", "eval", "show_console"}


def test_no_tool_module_refers_to_console():
    for d in ("fea_core/mcp", "fea_core/rest"):
        for p in (ROOT / d).glob("*.py"):
            txt = p.read_text(encoding="utf-8")
            assert "console_panel" not in txt and "ConsolePanel" not in txt and "show_console" not in txt, p
