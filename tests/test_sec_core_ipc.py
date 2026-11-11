"""IPC client: a process on the pipe that does not prove the session key (stale key file, foreign process) is not tried
again – the MCP server goes headless at once instead of failing / waiting ACK_TIMEOUT on every call."""
from __future__ import annotations

import io
import json
import time

import pytest

from fea_core import ipc


@pytest.fixture
def fake_pipe(tmp_path, monkeypatch):
    pipe = tmp_path / "feastudio_test_pipe"
    pipe.write_text("")                                           # "the pipe exists"
    monkeypatch.setattr(ipc, "token_path", lambda name=None: tmp_path / f"{name or 'x'}.token")
    ipc.write_token(pipe.name)
    ipc._UNVERIFIED.clear()
    calls = []

    def answer(lines):
        def _open(path):
            calls.append(path)
            buf = io.BytesIO(b"".join(json.dumps(x).encode() + b"\n" for x in lines))
            return buf, lambda data: None, lambda: None
        monkeypatch.setattr(ipc, "_open", _open)
    yield str(pipe), answer, calls
    ipc._UNVERIFIED.clear()


def test_unproven_server_falls_back_and_is_not_retried(fake_pipe):
    path, answer, calls = fake_pipe
    answer([{"ack": True, "proof": "0" * 64}])                    # a server without the key
    assert ipc.is_gui_running(path)
    with pytest.raises(ipc.GuiUnverifiedError):
        ipc.send_command("fea_gui_status", pipe_path=path, timeout=2.0)
    assert issubclass(ipc.GuiUnverifiedError, ipc.GuiNotRunningError)
    assert not ipc.is_gui_running(path)                           # the next call goes headless at once
    t = time.monotonic()
    with pytest.raises(ipc.GuiUnverifiedError):
        ipc.send_command("fea_gui_status", pipe_path=path, timeout=2.0)
    assert time.monotonic() - t < 0.5 and len(calls) == 1        # the pipe was not opened again
    time.sleep(0.02)
    ipc.write_token(path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1])   # a new GUI session = a new key file
    assert ipc.is_gui_running(path)


def test_rejected_signature_falls_back(fake_pipe):
    path, answer, _ = fake_pipe
    answer([{"status": "error", "error": "invalid signature"}])     # the GUI has another key (stale key file)
    with pytest.raises(ipc.GuiNotRunningError):
        ipc.send_command("fea_gui_status", pipe_path=path, timeout=2.0)
    assert not ipc.is_gui_running(path)


def test_valid_server_still_works(fake_pipe, monkeypatch):
    path, _, _ = fake_pipe
    name = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    token = ipc.read_token(name)
    sent = {}

    def _open(p):
        def write(data):
            sent["req"] = json.loads(data)
        class R:
            def readline(self, n=-1):
                nonce = sent["req"]["nonce"]
                if "ack" not in sent:
                    sent["ack"] = 1
                    return json.dumps({"ack": True, "proof": ipc.mac(token, b"ack", nonce.encode())}).encode() + b"\n"
                res = {"status": "ok", "result": "fine"}
                res["rmac"] = ipc.mac(token, b"res", nonce.encode(), ipc.result_body(res))
                return json.dumps(res).encode() + b"\n"
        return R(), write, lambda: None
    monkeypatch.setattr(ipc, "_open", _open)
    assert ipc.send_command("fea_gui_status", pipe_path=path, timeout=5.0) == {"status": "ok", "result": "fine"}
    assert ipc.is_gui_running(path)
