"""REST token proof (POST /api/hello, challenge–response) and its use in the Fusion add-in: the token and the design go only to
a server that proves it knows the token; an older server without the endpoint / a server without a token / a wrong proof =
nothing is sent, the add-in falls back to the export folder."""
from __future__ import annotations

import importlib.util
import json
import secrets
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fea_core.rest import create_app  # noqa: E402
from fea_core.rest.app import hello_proof  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def addin():
    p = ROOT / "adapters" / "fusion" / "FEAStudio" / "FEAStudio.py"
    spec = importlib.util.spec_from_file_location("fea_fusion_addin_hello", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    def __init__(self, app):
        self.port = _free_port()
        self.srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.th = threading.Thread(target=self.srv.run, daemon=True)

    def __enter__(self):
        self.th.start()
        t0 = time.time()
        while not self.srv.started:
            assert time.time() - t0 < 20, "server did not start"
            time.sleep(0.05)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc):
        self.srv.should_exit = True
        self.th.join(10)


# ===================================================================== server side (TestClient)

def test_hello_proves_the_token_without_it(tmp_path):
    c = TestClient(create_app(base_dir=tmp_path / "r", token="tajne-123", gui_launcher=lambda p: 0),
                   base_url="http://127.0.0.1:8765")
    nonce = secrets.token_hex(16)
    r = c.post("/api/hello", json={"nonce": nonce})                 # no Authorization header
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["token"] is True and d["proof"] == hello_proof("tajne-123", nonce, 8765)
    assert "tajne" not in r.text
    assert d["proof"] != hello_proof("tajne-123", nonce, 8766)       # bound to the port (a relay from another port fails)
    assert d["proof"] != hello_proof("jine", nonce, 8765)
    assert c.get("/api/status").status_code == 401                   # everything else still needs the token
    for bad in ({}, {"nonce": "abc"}, {"nonce": "zz" * 16}, {"nonce": "a" * 129}):
        assert c.post("/api/hello", json=bad).status_code == 400, bad
    # a foreign web page cannot use the endpoint either (Origin check of unsafe methods)
    assert c.post("/api/hello", json={"nonce": nonce}, headers={"Origin": "https://evil.example"}).status_code == 403


def test_hello_without_token(tmp_path):
    c = TestClient(create_app(base_dir=tmp_path / "r", gui_launcher=lambda p: 0), base_url="http://127.0.0.1:8765")
    d = c.post("/api/hello", json={"nonce": "0" * 32}).json()
    assert d == {"app": "FEA Studio", "token": False}


# ===================================================================== Fusion add-in against real servers

def _step(tmp_path: Path) -> Path:
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    p = tmp_path / "plate.step"
    write_step([Part("Plate", box(0, 0, 0, 50, 20, 5))], p)
    return p


def test_addin_sends_after_a_valid_proof(addin, tmp_path):
    app = create_app(base_dir=tmp_path / "srv", token="tajne-123", gui_launcher=lambda p: 1)
    with _Server(app) as url:
        s = dict(addin.DEFAULTS, server=url, token="tajne-123", export_dir=str(tmp_path))
        addin.prove_server(s)                                         # no exception
        st = addin.server_status(s)
        assert st is not None and st["app"] == "FEA Studio"
        step = _step(tmp_path)
        msg = addin.deliver(s, str(step), {"source": "Test", "document": "Plate", "parts": []}, open_gui=False)
        assert "Sent to FEA Studio" in msg, msg
        assert app.state.session.study is not None


def test_addin_refuses_wrong_or_missing_proof(addin, tmp_path):
    """Wrong token in the add-in, a server without a token: neither the token nor the design is sent, the export folder is used."""
    seen: list[dict] = []
    for srv_token, cli_token, why in (("tajne-123", "jine", "could not prove"), (None, "tajne-123", "does not use a token")):
        app = create_app(base_dir=tmp_path / f"srv_{why[:5]}", token=srv_token, gui_launcher=lambda p: 1)

        @app.middleware("http")
        async def record(request: Request, call_next, _seen=seen):
            _seen.append({"path": request.url.path, "auth": request.headers.get("authorization")})
            return await call_next(request)
        with _Server(app) as url:
            s = dict(addin.DEFAULTS, server=url, token=cli_token, export_dir=str(tmp_path), fea_studio_home="")
            with pytest.raises(addin.UntrustedServer, match=why):
                addin.prove_server(s)
            n = len(addin.NOTES)
            assert addin.server_status(s) is None and any(why in x for x in addin.NOTES[max(0, n - 1):]), addin.NOTES
            step = _step(tmp_path)
            msg = addin.deliver(s, str(step), {"source": "Test", "parts": []}, open_gui=False)
            assert "saved" in msg and why in msg, msg
            assert app.state.session.study is None
    assert seen and all(x["path"] == "/api/hello" and x["auth"] is None for x in seen), seen


def test_addin_against_an_older_server_without_hello(addin, tmp_path):
    """A server of an older version (token everywhere, no /api/hello) never gets the token: the add-in falls back."""
    seen: list[dict] = []
    old = FastAPI()

    @old.middleware("http")
    async def auth(request: Request, call_next):
        seen.append({"path": request.url.path, "auth": request.headers.get("authorization")})
        if request.headers.get("authorization") != "Bearer tajne-123":
            return JSONResponse({"error": "Token missing or wrong"}, status_code=401)
        return await call_next(request)

    @old.get("/api/status")
    def status():
        return {"app": "FEA Studio"}
    with _Server(old) as url:
        s = dict(addin.DEFAULTS, server=url, token="tajne-123", export_dir=str(tmp_path), fea_studio_home="")
        assert addin.server_status(s) is None
        assert any("older version" in x for x in addin.NOTES)
        msg = addin.deliver(s, str(_step(tmp_path)), {"source": "Test", "parts": []}, open_gui=False)
        assert "saved" in msg, msg
    assert seen and all(x["auth"] is None for x in seen), seen
    assert json.loads((tmp_path / "plate.json").read_text(encoding="utf-8"))["source"] == "Test"
