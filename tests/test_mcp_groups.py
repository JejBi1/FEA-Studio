"""Tool groups of the MCP server: the default list is small, the groups are enabled by env / --groups / fea_tools_enable,
REST always exposes everything, the listed descriptions stay short (context tokens)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from fea_core.mcp import groups  # noqa: E402
from fea_core.mcp.server import build_server, split_doc  # noqa: E402
from fea_core.mcp.session import Session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
META = set(groups.META_TOOLS)


def _tools(spec, tmp_path):
    async def main():
        srv = build_server(Session(tmp_path / "mcp"), groups_spec=spec)
        async with Client(srv, mode="legacy") as c:
            return (await c.list_tools()).tools
    return anyio.run(main)


def _size(tools) -> int:
    return sum(len(json.dumps(t.model_dump(mode="json", exclude_none=True, by_alias=True),
                              ensure_ascii=False, separators=(",", ":"))) for t in tools)


def test_default_list_is_core_only(tmp_path):
    tools = _tools(None if "FEA_MCP_GROUPS" not in os.environ else "core", tmp_path)
    names = {t.name for t in tools}
    assert META <= names
    assert 35 <= len(tools) <= 55, len(tools)
    assert "fea_solve_run" in names and "fea_job_start" in names and "fea_report_word" in names
    assert not any(n.startswith(("fea_print_", "fea_frame_", "fea_motion_", "fea_dyn_", "fea_thermal_")) for n in names)
    assert _size(tools) <= 32_000, _size(tools)          # ~8 000 tokens


def test_all_groups_budget(tmp_path):
    tools = _tools("all", tmp_path)
    assert len(tools) >= 205
    assert _size(tools) <= 110_000, _size(tools)         # ~27 000 tokens (was ~60 000)
    for t in tools:                                      # no auto titles / null defaults in the listed schemas
        s = json.dumps(t.input_schema)
        assert '"title": "' not in s, t.name             # (a parameter may itself be called "title")
        assert '"default": null' not in s, t.name


def test_group_tables_are_consistent():
    from fea_core.mcp import server
    from mcp.server.mcpserver import MCPServer
    sess = Session()
    seen = {}
    declared = {m: {d, *o} for m, d, o in groups.MODULES}
    for name, default, others in groups.MODULES:
        mcp = MCPServer(name="x", log_level="WARNING")
        import importlib
        reg = server._Registrar(mcp)
        reg.module_group = default
        importlib.import_module(f"fea_core.mcp.{name}").register(reg, sess)
        for t in mcp._tool_manager.list_tools():
            g = groups.group_of(t.name, default)
            assert g in declared[name], f"{t.name}: group {g} is not declared for module {name}"
            seen[t.name] = g
    assert set(groups.OVERRIDES) <= set(seen), set(groups.OVERRIDES) - set(seen)
    assert set(seen.values()) <= set(groups.GROUPS)
    assert set(seen.values()) == set(groups.GROUPS)       # every group has tools


def test_parse():
    assert groups.parse(None) == {"core"}
    assert groups.parse("thermal, print") == {"core", "thermal", "print"}
    assert groups.parse("all") == set(groups.GROUPS)
    assert groups.parse(["frame"]) == {"core", "frame"}
    with pytest.raises(ValueError):
        groups.parse("thermall")


def test_split_doc():
    short = "Short text. Two sentences."
    assert split_doc(short) == (short, "")
    long = ("Creates a thing (e.g. a box. With brackets). " + "More detail here and there. " * 12).strip()
    summary, full = split_doc(long)
    assert full == long and summary.startswith("Creates a thing (e.g. a box. With brackets).")
    assert len(summary) < len(long) // 2


def test_enable_at_run_time_and_help(tmp_path):
    seen = []

    async def handler(msg):
        seen.append(repr(msg))

    async def main():
        srv = build_server(Session(tmp_path / "mcp"), groups_spec="core")
        async with Client(srv, mode="legacy", message_handler=handler) as c:
            before = {t.name for t in (await c.list_tools()).tools}
            assert "fea_thermal_case_create" not in before
            g = json.loads((await c.call_tool("fea_tools_groups", {})).content[0].text)
            assert {x["group"] for x in g["groups"]} == set(groups.GROUPS)
            assert [x["enabled"] for x in g["groups"] if x["group"] == "thermal"] == [False]
            r = json.loads((await c.call_tool("fea_tools_enable", {"groups": "thermal"})).content[0].text)
            assert "fea_thermal_case_create" in r["added"] and "thermal" in r["enabled"]
            after = {t.name: t for t in (await c.list_tools()).tools}
            assert "fea_thermal_case_create" in after and "fea_load_add_temperature" in after
            # second call: nothing new
            r2 = json.loads((await c.call_tool("fea_tools_enable", {"groups": "thermal"})).content[0].text)
            assert r2["added"] == []
            # a bad group = error, the list stays
            bad = await c.call_tool("fea_tools_enable", {"groups": "nonsense"})
            assert bad.is_error
            # help returns the full text of a shortened description
            short = [t for t in after.values() if "[more: fea_tool_help]" in t.description]
            assert short
            full = (await c.call_tool("fea_tool_help", {"tool": short[0].name})).content[0].text
            assert len(full) > len(short[0].description)
            # a late-enabled tool works
            res = await c.call_tool("fea_thermal_list", {})          # reached the tool (no study is open)
            assert "No study is open" in res.content[0].text, res
    anyio.run(main)
    assert any("tools/list_changed" in s for s in seen), seen


def test_server_announces_list_changed(tmp_path):
    async def main():
        srv = build_server(Session(tmp_path / "mcp"), groups_spec="core")
        opts = srv._lowlevel_server.create_initialization_options()
        assert opts.capabilities.tools.list_changed is True
    anyio.run(main)


def test_core_start_does_not_import_other_groups():
    code = ("import sys; from fea_core.mcp.server import build_server; build_server(groups_spec='core'); "
            "mods = [m for m in ('tools_print', 'tools_frame', 'tools_motion', 'tools_dynamics', 'tools_thermal', 'tools_beams', "
            "'tools_contact', 'tools_voxel', 'tools_reduced', 'tools_calc') if 'fea_core.mcp.' + m in sys.modules]; "
            "print('LOADED', ' '.join(mods))")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-600:]
    assert r.stdout.strip().endswith("LOADED"), r.stdout


def test_rest_exposes_all_tools_regardless_of_groups(tmp_path):
    from fea_core.rest.registry import collect_tools
    rest = collect_tools(Session(tmp_path / "mcp"))
    mcp_all = {t.name for t in _tools("all", tmp_path)} - META
    assert mcp_all <= set(rest), mcp_all - set(rest)
    assert "fea_print_sweep" in rest and "fea_motion_run" in rest


def test_stdio_server_with_groups_argument(tmp_path):
    async def main(args):
        from mcp import StdioServerParameters
        params = StdioServerParameters(command=sys.executable, args=["-m", "fea_core.mcp", "--workdir", str(tmp_path / "w"), *args],
                                       env={**os.environ, "PYTHONPATH": str(ROOT), "FEA_MCP_GROUPS": ""})
        async with Client(params, mode="legacy") as c:
            return {t.name for t in (await c.list_tools()).tools}
    core = anyio.run(main, [])
    assert "fea_solve_run" in core and "fea_print_sweep" not in core and "fea_tools_enable" in core
    more = anyio.run(main, ["--groups", "print,frame"])
    assert "fea_print_sweep" in more and "fea_frame_node_set" in more and "fea_motion_run" not in more
    allg = anyio.run(main, ["--groups", "all"])
    assert len(allg) > len(more) > len(core)
