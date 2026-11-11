"""Fusion add-in: the client copy of the IPC protocol stays in sync with fea_core/ipc.py, and the REST client sends nothing to a
port owner it cannot trust. Pure Python (the add-in is imported without Fusion's `adsk`)."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from fea_core import ipc

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def addin():
    p = ROOT / "adapters" / "fusion" / "FEAStudio" / "FEAStudio.py"
    spec = importlib.util.spec_from_file_location("fea_fusion_addin_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.adsk is None
    return mod


def test_protocol_primitives_match_core(addin, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    tok = "ab" * 32
    nonce = "0123456789abcdef" * 2
    assert addin._pipe_name() == ipc.default_pipe_name()
    if sys.platform == "win32":
        assert Path(addin._ipc_token_path()) == ipc.token_path()
        assert addin.PIPE_PREFIX + addin._pipe_name() == ipc.get_pipe_path()
    for parts in ((b"req", nonce.encode(), "č{}".encode()), (b"ack", nonce.encode()), (b"res",)):
        assert addin._ipc_mac(tok, *parts) == ipc.mac(tok, *parts)
    res = {"status": "ok", "result": {"b": [1, 2.5, "ž"], "a": None}, "rmac": "x"}
    assert addin._ipc_result_body(res) == ipc.result_body(res)
    line = addin._ipc_request(tok, "fea_cad_open", {"step_path": "C:/a b/č.step"}, nonce)
    req = json.loads(line)
    body = json.dumps({"tool": "fea_cad_open", "args": {"step_path": "C:/a b/č.step"}}, ensure_ascii=False)
    assert line.endswith(b"\n") and req == {"nonce": nonce, "body": body,
                                            "mac": ipc.mac(tok, b"req", nonce.encode(), body.encode("utf-8"))}


class _FakePipe(io.RawIOBase):
    """The GUI side of one exchange, built only from fea_core.ipc primitives (as fea_gui/ipc_server.py checks a request)."""

    def __init__(self, token, result, sign=True):
        self.token, self.result, self.sign = token, result, sign
        self.inbuf, self.out = b"", io.BytesIO()

    def readable(self):
        return True

    def writable(self):
        return True

    def write(self, b):
        self.inbuf += bytes(b)
        if self.inbuf.endswith(b"\n"):
            req = json.loads(self.inbuf)
            assert ipc.mac(self.token, b"req", req["nonce"].encode(), req["body"].encode("utf-8")) == req["mac"]
            n = req["nonce"].encode()
            ack = {"ack": True, "proof": ipc.mac(self.token, b"ack", n) if self.sign else "0" * 64}
            res = {"status": "ok", "result": self.result}
            res["rmac"] = ipc.mac(self.token, b"res", n, ipc.result_body(res))
            self.out = io.BytesIO((json.dumps(ack) + "\n" + json.dumps(res, ensure_ascii=False) + "\n").encode("utf-8"))
        return len(b)

    def readinto(self, b):
        data = self.out.read(len(b))
        b[:len(data)] = data
        return len(data)


def test_ipc_call_against_core_protocol(addin, monkeypatch):
    tok = "cd" * 32
    monkeypatch.setattr(addin, "_ipc_token", lambda: tok)
    monkeypatch.setattr(addin, "_ipc_open", lambda: _FakePipe(tok, {"mode": "update", "x": "ř"}))
    assert addin.ipc_call("fea_cad_update", {"step_path": "a.step"}, timeout=10) == {"mode": "update", "x": "ř"}
    monkeypatch.setattr(addin, "_ipc_open", lambda: _FakePipe(tok, {}, sign=False))
    with pytest.raises(RuntimeError, match="did not prove"):
        addin.ipc_call("fea_cad_update", {}, timeout=10)


def test_rest_check_accepts_own_listener(addin):
    """A listener of this process (= our user) passes; a remote server is not checked; no listener = nothing to refuse."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        port = s.getsockname()[1]
        settings = {"server": f"http://127.0.0.1:{port}"}
        addin.check_server(settings)                         # no exception
        if os.name == "nt":
            assert os.getpid() in addin._listener_pids(port)
    finally:
        s.close()
    addin.check_server({"server": "http://192.0.2.1:8765"})


@pytest.mark.skipif(os.name != "nt", reason="Windows owner check")
def test_rest_check_refuses_foreign_owner(addin, monkeypatch):
    sent = []
    monkeypatch.setattr(addin, "_listener_pids", lambda port: {4})            # PID 4 = System: not this user
    monkeypatch.setattr(addin.urllib.request, "urlopen", lambda *a, **k: sent.append(a))
    st = addin.server_status({"server": "http://127.0.0.1:8765", "token": "secret"})
    assert st is None and not sent                                           # neither the token nor a request went out
    assert any("another user" in n for n in addin.NOTES)
    with pytest.raises(addin.UntrustedServer):
        addin._call({"server": "http://localhost:8765", "token": "secret"}, "POST", "/api/import", b"x")
    assert not sent
    assert addin._process_is_mine(os.getpid())
    assert not addin._process_is_mine(4)
