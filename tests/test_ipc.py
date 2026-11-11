"""Tests of the Dynamic IPC Bridge (control from an LLM / MCP and detection of the GUI)."""

import pytest

pytest.importorskip("mcp")                       # module mcp (the IPC bridge is exercised through the MCP server)
import anyio  # noqa: E402

import fea_core.ipc as ipc_mod
from fea_core.ipc import (
    GuiNotRunningError,

    get_pipe_path,
    is_gui_running,
    load_active_session_info,
    save_active_session_info,
    clear_active_session_info,
    send_command,
)
from fea_core.mcp.session import Session
from fea_core.mcp.server import build_server


def test_gui_detection_when_not_running():
    """When the GUI is not running, is_gui_running() returns False and send_command raises GuiNotRunningError."""
    fake_pipe = get_pipe_path("non_existent_pipe_for_test_12345")
    assert not is_gui_running(fake_pipe)
    with pytest.raises(GuiNotRunningError):
        send_command("fea_gui_status", pipe_path=fake_pipe, timeout=1.0)


def test_active_session_handoff_info():
    """Writing and reading of the metadata about the active session of the assistant."""
    clear_active_session_info()
    try:
        assert load_active_session_info() is None
        save_active_session_info(
            study_name="Nosnik_L",
            step_path="/path/to/step.stp",
            study_path="/path/to/study.fea",
            workdir="/path/to/workdir",
        )
        info = load_active_session_info()
        assert info is not None
        assert info["study_name"] == "Nosnik_L"
        assert info["step_path"] == "/path/to/step.stp"
        assert info["study_path"] == "/path/to/study.fea"
    finally:
        clear_active_session_info()


def test_headless_tools_gui():
    """The fea_gui_* tools in headless mode safely return gui_running=False without an error."""
    srv = build_server(Session())
    tool_mgr = srv._tool_manager

    # fea_gui_status
    status_tool = tool_mgr.get_tool("fea_gui_status")
    assert status_tool is not None

    # fea_gui_toast
    toast_tool = tool_mgr.get_tool("fea_gui_toast")
    assert toast_tool is not None

    # fea_gui_camera_focus
    cam_tool = tool_mgr.get_tool("fea_gui_camera_focus")
    assert cam_tool is not None


def test_mcp_routing_through_ipc(monkeypatch):
    """When the GUI is active, a call of an MCP tool is automatically redirected to the GUI IPC."""
    monkeypatch.setattr(ipc_mod, "is_gui_running", lambda *_: True)

    calls = []
    def mock_send(tool, args, **kw):
        calls.append((tool, args))
        return {"status": "ok", "result": {"routed_to_gui": True, "tool": tool}}

    monkeypatch.setattr(ipc_mod, "send_command", mock_send)

    srv = build_server(Session())
    tool = srv._tool_manager.get_tool("fea_gui_toast")
    assert tool is not None

    async def call_it():
        return await tool.fn(message="Test IPC bridge")

    res = anyio.run(call_it)
    assert len(calls) == 1
    assert calls[0][0] == "fea_gui_toast"
    assert calls[0][1]["message"] == "Test IPC bridge"
    assert res == {"routed_to_gui": True, "tool": "fea_gui_toast"}


def _run_case(name: str) -> None:
    import subprocess
    import sys
    from pathlib import Path
    p = subprocess.run([sys.executable, "-m", "tests.ipc_server_cases", name], capture_output=True, text=True,
                       cwd=str(Path(__file__).resolve().parents[1]), timeout=120)
    assert p.returncode == 0 and "OK" in p.stdout, (p.stdout + p.stderr)[-3000:]


def test_ipc_server_communication():
    """Two-way communication through QLocalServer and an IPC client (a separate process)."""
    _run_case("test_ipc_server_communication")


def test_ipc_rejects_unsigned_and_forged_server():
    """A request without a valid HMAC is refused by the GUI and the tool is not run (a separate process)."""
    _run_case("test_ipc_rejects_unsigned_and_forged_server")
