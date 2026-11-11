"""CAD adapter (F6): the REST client from the Fusion add-in against a real server (uvicorn in a thread)."""
import importlib.machinery
import importlib.util
import json
import py_compile
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

ROOT = Path(__file__).resolve().parent.parent
FUSION = ROOT / "adapters" / "fusion" / "FEAStudio" / "FEAStudio.py"


def load(path: Path, name: str):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mods():
    return {"fusion": load(FUSION, "fea_fusion_addin")}


def test_compile(tmp_path):
    for p in (FUSION,):
        py_compile.compile(str(p), cfile=str(tmp_path / (p.stem + ".pyc")), doraise=True)
    assert json.loads((FUSION.parent / "FEAStudio.manifest").read_text(encoding="utf-8"))["type"] == "addin"


# ===================================================================== real server

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from fea_core.rest import create_app
    calls = []
    app = create_app(base_dir=tmp_path_factory.mktemp("rest_srv"), gui_launcher=lambda p: calls.append(p) or 1)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    t0 = time.time()
    while not srv.started:
        assert time.time() - t0 < 20, "server nenaběhl"
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}", "app": app, "gui": calls}
    srv.should_exit = True
    th.join(10)


@pytest.fixture(scope="module")
def step(tmp_path_factory):
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    p = tmp_path_factory.mktemp("adapt_geo") / "díl.step"
    write_step([Part("Rám:1_1", box(0, 0, 0, 50, 20, 5)), Part("Rám:1_2", box(0, 0, 5, 50, 5, 20)),
                Part("Šroub M4:1", box(60, 0, 0, 4, 4, 20))], p)
    return p


def _meta():
    return {"source": "Test", "document": "Držák", "units": "mm",
            "parts": [{"name": "Rám:1_1", "names": ["Rám:1", "Body1"], "material": "Aluminum 6061"},
                      {"name": "Rám:1_2", "names": ["Rám:1", "Body2"], "material": "S355"},
                      {"name": "Šroub M4:1", "material": "Stainless Steel AISI 304", "attributes": {"a.b": "c"}}]}


@pytest.mark.parametrize("which", ["fusion"])
def test_deliver_via_rest(mods, server, step, tmp_path, which):
    m = mods[which]
    settings = dict(m.DEFAULTS, server=server["url"], export_dir=str(tmp_path))
    local = tmp_path / step.name
    local.write_bytes(step.read_bytes())
    n_gui = len(server["gui"])
    msg = m.deliver(settings, str(local), _meta(), open_gui=True)
    # all three materials are assigned by the designation converter (a match of the group, not an approximate one): S355 → Ocel S355,
    # AISI 304 → 1.4301 → Nerez 1.4301 and since the extension of the library (15. 10. 2026) also "Aluminum 6061" → EN AW-6061 →
    # Hliník EN AW-6061 T6 (previously it was not in the library → only a hint, "material assigned 2")
    assert "Sent to FEA Studio" in msg and "3 parts, material assigned to 3" in msg, msg
    assert json.loads(local.with_suffix(".json").read_text(encoding="utf-8"))["document"] == "Držák"
    sess = server["app"].state.session
    assert sess.study.name == "Držák" and sess.part_names() == ["Rám:1_1", "Rám:1_2", "Šroub M4:1"]
    assert sess.study.part_materials == {"Rám:1_1": "Hliník EN AW-6061 T6", "Rám:1_2": "Ocel S355",
                                         "Šroub M4:1": "Nerez 1.4301"}
    assert len(server["gui"]) == n_gui + 1 and server["gui"][-1].suffix == ".step"


def test_fallback_without_server(mods, tmp_path, step):
    for m in (mods["fusion"],):
        s = dict(m.DEFAULTS, server=f"http://127.0.0.1:{_free_port()}", export_dir=str(tmp_path), fea_studio_home="")
        assert m.server_status(s) is None
        local = tmp_path / "x.step"
        local.write_bytes(step.read_bytes())
        assert "not running" in m.deliver(s, str(local), _meta(), open_gui=False)
        with pytest.raises(RuntimeError, match="FEA_STUDIO_HOME"):
            m.deliver(s, str(local), _meta(), open_gui=True)
        s["fea_studio_home"] = str(tmp_path)
        with pytest.raises(RuntimeError, match="FEA Studio.bat"):
            m.launch_gui_local(s, str(local))


def test_settings_env(mods, monkeypatch, tmp_path):
    monkeypatch.setenv("FEA_REST_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("FEA_STUDIO_HOME", str(tmp_path))
    for m in (mods["fusion"],):
        s = m.load_settings()
        assert s["server"] == "http://127.0.0.1:9999" and s["fea_studio_home"] == str(tmp_path) and s["export_dir"]
    cfg = tmp_path / "settings.json"
    cfg.write_text(json.dumps({"token": "abc", "open_gui": False}), encoding="utf-8")
    s = mods["fusion"].load_settings(str(cfg))
    assert s["token"] == "abc" and s["open_gui"] is False
