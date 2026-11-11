"""Tests of the IPC server (Qt) run in a SEPARATE process from tests/test_ipc.py.

In the full pytest set processEvents would process the leftovers of objects from the earlier GUI tests in the same process (an occasional
crash access violation outside the IPC). Run: python -m tests.ipc_server_cases <function name>.
"""
import os
import sys
import threading
import time
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import fea_core.ipc as ipc_mod  # noqa: E402
from fea_core.ipc import get_pipe_path, is_gui_running, send_command  # noqa: E402


def test_ipc_server_communication():
    """Check of the two-way communication through QLocalServer and an IPC client."""
    from PySide6.QtWidgets import QApplication
    from fea_gui.ipc_server import IpcServer

    app = QApplication.instance() or QApplication([])

    test_pipe = "test_feastudio_ipc_unit"
    pipe_path = get_pipe_path(test_pipe)

    # Mock MainWindow
    mock_win = MagicMock()
    mock_win.execute_mcp_action.side_effect = lambda tool, args: {
        "executed_tool": tool,
        "received_args": args,
        "gui_ok": True,
    }

    server = IpcServer(mock_win, pipe_name=test_pipe)
    try:
        assert is_gui_running(pipe_path)

        res_container = []
        err_container = []

        def client_thread():
            try:
                resp = send_command(
                    "fea_gui_toast",
                    {"message": "Ahoj z testu", "timeout_ms": 3000},
                    pipe_path=pipe_path,
                    timeout=5.0,
                )
                res_container.append(resp)
            except Exception as e:
                err_container.append(e)

        t = threading.Thread(target=client_thread)
        t.start()

        start = time.time()
        while t.is_alive() and (time.time() - start) < 5.0:
            app.processEvents()
            time.sleep(0.01)

        t.join(timeout=1.0)
        assert not err_container, f"Client error: {err_container}"
        assert res_container, "No response received"

        resp = res_container[0]
        assert resp["status"] == "ok"
        res = resp["result"]
        assert res["executed_tool"] == "fea_gui_toast"
        assert res["received_args"]["message"] == "Ahoj z testu"
        assert res["gui_ok"] is True
        assert mock_win.execute_mcp_action.called
    finally:
        server.close()
        assert not is_gui_running(pipe_path)


def test_ipc_rejects_unsigned_and_forged_server():
    """A request without a valid HMAC is refused by the GUI; a server without the key (a planted pipe) is not accepted by the client."""
    from PySide6.QtWidgets import QApplication
    from fea_gui.ipc_server import IpcServer

    app = QApplication.instance() or QApplication([])
    test_pipe = "test_feastudio_ipc_auth"
    pipe_path = get_pipe_path(test_pipe)
    mock_win = MagicMock()
    mock_win.execute_mcp_action.side_effect = lambda tool, args: {"ok": True}
    server = IpcServer(mock_win, pipe_name=test_pipe)
    try:
        good = ipc_mod.read_token(test_pipe)
        assert good
        # a client with a wrong key (the key file overwritten) → rejected, the tool is not run
        ipc_mod.token_path(test_pipe).write_text("0" * 64, encoding="ascii")
        errs = []

        def client():
            try:
                send_command("fea_gui_toast", {"message": "x"}, pipe_path=pipe_path, timeout=5.0)
            except Exception as e:  # noqa: BLE001
                errs.append(e)

        t = threading.Thread(target=client)
        t.start()
        start = time.time()
        while t.is_alive() and time.time() - start < 6.0:
            app.processEvents()
            time.sleep(0.01)
        t.join(timeout=1.0)
        assert errs and isinstance(errs[0], ipc_mod.IpcError)
        assert not mock_win.execute_mcp_action.called
    finally:
        ipc_mod.token_path(test_pipe).write_text(good, encoding="ascii")
        server.close()
    assert not ipc_mod.token_path(test_pipe).exists()


if __name__ == "__main__":
    globals()[sys.argv[1]]()
    print("OK")
