"""REST API (F6): tools = MCP server, import of a STEP from a CAD adapter, files, token, GUI; e2e console V1.

Fast tests through fastapi TestClient without Gmsh/ccx; e2e (mesh + ccx) is slow.
"""
import json
import struct
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mcp")

from fastapi.testclient import TestClient  # noqa: E402

from fea_core.rest import create_app  # noqa: E402


# ===================================================================== helpers

@pytest.fixture(scope="module")
def steps(tmp_path_factory):
    """STEP: console V1 (200×20×10) and an assembly of two bodies with diacritics."""
    from verification.cases import box
    from fea_core.geometry import Part, write_step
    d = tmp_path_factory.mktemp("rest_geo")
    beam = d / "v1.step"
    write_step([Part("Konzola", box(0, 0, 0, 200, 20, 10))], beam)
    asm = d / "sestava.step"
    write_step([Part("Deska nosná", box(0, 0, 0, 60, 40, 5), color=(0.2, 0.4, 0.8)),
                Part("Žebro", box(0, 0, 5, 60, 5, 20))], asm)
    return {"beam": beam, "asm": asm}


class Launcher:
    def __init__(self):
        self.calls = []

    def __call__(self, path):
        self.calls.append(Path(path))
        return 4242


@pytest.fixture
def client(tmp_path):
    gui = Launcher()
    app = create_app(base_dir=tmp_path / "rest", gui_launcher=gui)
    c = TestClient(app, base_url="http://127.0.0.1:8765")      # Host = loopback (ochrana DNS rebinding)
    c.gui = gui
    c.sess = app.state.session
    return c


def png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "není PNG"
    return struct.unpack(">II", data[16:24])


def call(c, tool, status=200, **args):
    r = c.post(f"/api/tools/{tool}", json=args)
    assert r.status_code == status, (tool, r.status_code, r.text)
    return r.json()


def upload(c, path: Path, meta=None, **form):
    data = {k: (json.dumps(v) if isinstance(v, bool) else str(v)) for k, v in form.items()}
    if meta is not None:
        data["metadata"] = json.dumps(meta, ensure_ascii=False)
    return c.post("/api/import", files={"file": (path.name, path.read_bytes(), "application/step")}, data=data)


# ===================================================================== fast tests

def test_tools_same_as_mcp(client):
    """List of the tools = MCP server (names, descriptions, parameter schemas)."""
    from fea_core.mcp.server import build_server
    from fea_core.mcp.groups import META_TOOLS      # fea_tools_* manage the MCP tool list only; REST has every tool
    mcp_tools = {t.name: t for t in build_server(base_dir=client.sess.base_dir, groups_spec="all")._tool_manager.list_tools()
                 if t.name not in META_TOOLS}
    rest = {t["name"]: t for t in client.get("/api/tools").json()["tools"]}
    assert set(rest) == set(mcp_tools)
    for name, t in mcp_tools.items():
        assert rest[name]["parameters"] == t.parameters, name
        assert rest[name]["description"].split()[:2] == t.description.split()[:2], name
    one = client.get("/api/tools/fea_study_new").json()
    assert "step_path" in one["parameters"]["properties"]          # optional: without STEP = only the frame system
    assert client.get("/api/tools/neni").status_code == 404


def test_status_and_errors(client):
    st = client.get("/api/status").json()
    assert st["app"] == "FEA Studio" and st["study"] is None and st["busy"] is None and not st["results"]
    assert st["tools"] >= 42
    e = call(client, "fea_study_status", 400)
    assert "study" in e["error"].lower()
    e = call(client, "fea_study_new", 400, step_path="C:/neexistuje/x.step")
    assert "does not exist" in e["error"]
    e = call(client, "fea_materials_list", 400, qury="S355")
    assert "qury" in e["error"] and "query" in e["error"]
    e = call(client, "fea_mesh_generate", 400, global_size="hodně")
    assert "Invalid arguments" in e["error"]
    r = call(client, "fea_materials_list", query="S355")
    assert r["tool"] == "fea_materials_list" and r["result"]["materials"][0]["name"] == "Ocel S355"
    assert client.post("/api/tools/neni", json={}).status_code == 404
    assert client.post("/api/tools/fea_materials_list?images=x", json={}).status_code == 400


def test_busy_409(client):
    client.sess.busy = "výpočet"
    try:
        e = call(client, "fea_materials_list", 409)
        assert "busy" in e["error"] and e["busy"] == "výpočet"
        assert client.get("/api/status").json()["busy"] == "výpočet"     # the status also matches during a calculation
    finally:
        client.sess.busy = None


def test_job_tools_answer_while_busy(client):
    """Asynchronous jobs (F8): status / result / cancellation of a job are excluded from the "busy" check (the calculation holds the lock)."""
    client.sess.busy = "úloha job-1 (výpočet studie)"
    try:
        e = call(client, "fea_job_status", 400)                      # without a job an error of the tool, not 409
        assert "No job" in e["error"]
        call(client, "fea_contact_list", 409)
        call(client, "fea_dyn_settings", 409)
    finally:
        client.sess.busy = None


def test_import_with_metadata(client, steps):
    meta = {"source": "Test CAD", "document": "Sestava držáku", "units": "mm",
            "parts": [{"name": "Deska nosná", "material": "S235", "color": [0.2, 0.4, 0.8],
                       "attributes": {"vyroba.svar": "ne"}},
                      {"name": "Žebro:1", "material": "Steel", "material_props": {"E_MPa": 210000, "nu": 0.3}},
                      {"name": "Duch", "material": "1.4301"}]}
    r = upload(client, steps["asm"], meta)
    assert r.status_code == 200, r.text
    d = r.json()
    step = Path(d["step_path"])
    assert step.parent.name == "imports" and step.read_bytes() == steps["asm"].read_bytes()
    assert json.loads(Path(d["metadata_path"]).read_text(encoding="utf-8"))["document"] == "Sestava držáku"
    assert d["study"]["study"] == "Sestava držáku"
    parts = {p["name"]: p for p in d["parts"]}
    assert set(parts) == {"Deska nosná", "Žebro"}                      # diacritics preserved through the STEP
    assert parts["Deska nosná"]["material"] == "Ocel S235" and parts["Deska nosná"]["attributes"]
    assert "material" not in parts["Žebro"] and parts["Žebro"]["metadata"] == "Žebro:1"
    assert parts["Žebro"]["cad_material_props"]["E_MPa"] == 210000
    assert any("Duch" in w for w in d["warnings"]) and any("Steel" in w for w in d["warnings"])
    # the study is open in the session and the material assigned as through fea_materials_assign (the first = also the default)
    assert parts["Žebro"]["material_effective"] == "Ocel S235"
    assert any("default material" in w and "Žebro" in w for w in d["warnings"])
    s = call(client, "fea_study_status")["result"]
    assert {p["name"]: p["material"] for p in s["parts"]} == {"Deska nosná": "Ocel S235", "Žebro": "Ocel S235"}
    st = client.get("/api/status").json()
    assert st["study"] == "Sestava držáku" and st["last_import"] == str(step)
    # download of the imported file
    f = client.get(d["step_url"])
    assert f.status_code == 200 and f.content == steps["asm"].read_bytes()
    # the GUI on the last import
    g = client.post("/api/open-gui", json={}).json()
    assert g["pid"] == 4242 and client.gui.calls == [step.resolve()]


def test_import_without_assigning_materials(client, steps):
    """assign_materials=false: the material from the library is not assigned, but it is not reported as "not in the library"
    (previously a false warning) – the found material is in material_suggested; an unknown material is still reported.
    """
    meta = {"source": "Test", "parts": [{"name": "Deska nosná", "material": "Ocel S235"},
                                        {"name": "Žebro", "material": "Unobtainium"}]}
    d = upload(client, steps["asm"], meta, assign_materials=False).json()
    parts = {p["name"]: p for p in d["parts"]}
    assert "material" not in parts["Deska nosná"] and parts["Deska nosná"]["material_suggested"] == "Ocel S235"
    bad = [w for w in d["warnings"] if "is not in the library" in w]
    assert len(bad) == 1 and "Unobtainium" in bad[0] and "S235" not in bad[0]


def test_import_without_open_and_errors(client, steps, tmp_path):
    r = upload(client, steps["beam"], open=False)
    assert r.status_code == 200 and "study" not in r.json() and r.json()["metadata_path"] is None
    assert client.get("/api/status").json()["study"] is None
    bad = tmp_path / "x.step"
    bad.write_text("nic")
    assert "ISO-10303-21" in upload(client, bad).json()["error"]
    txt = tmp_path / "x.txt"
    txt.write_text("ISO-10303-21;")
    assert upload(client, txt).status_code == 400
    assert "JSON" in upload(client, steps["beam"], open=False, metadata="{nic").json()["error"]


def test_files_traversal(client, tmp_path):
    (client.sess.base_dir / "a").mkdir(parents=True)
    (client.sess.base_dir / "a" / ".fea_mcp_workdir").write_text("")     # a server work folder (allow-list of /api/files)
    (client.sess.base_dir / "a" / "ok.txt").write_text("ok")
    secret = tmp_path / "tajne.txt"
    secret.write_text("x")
    assert client.get("/api/files/a/ok.txt").text == "ok"
    for p in ("..%2Ftajne.txt", "a/..%2F..%2Ftajne.txt", "a%2F..%2F..%2Ftajne.txt", "..%5Ctajne.txt",
              "C:%2FWindows%2Fwin.ini", "%2F%2Fetc/passwd"):
        r = client.get("/api/files/" + p)
        assert r.status_code in (403, 404), (p, r.status_code)
        assert r.text != "x"
    assert client.get("/api/files/a/neni.txt").status_code == 404


def test_open_gui_checks(client, steps, tmp_path):
    assert client.post("/api/open-gui", json={}).status_code == 400            # nothing to open
    assert client.post("/api/open-gui", json={"path": str(tmp_path / "neni.fea")}).status_code == 404
    other = tmp_path / "x.exe"
    other.write_text("")
    assert client.post("/api/open-gui", json={"path": str(other)}).status_code == 400
    r = client.post("/api/open-gui", json={"path": str(steps["beam"])}).json()
    assert r["opened"] == str(steps["beam"].resolve()) and client.gui.calls == [steps["beam"].resolve()]


def test_open_gui_json(client, steps, tmp_path):
    """.json opens in the GUI only as a study (fea_study_save) – metadata from the CAD and a study without STEP → 400 immediately in REST."""
    import os
    from fea_core.project import load_study_json
    from fea_core.study import Study
    st = Study(name="Konzola", geometry=os.path.relpath(steps["beam"], tmp_path))
    st.save(tmp_path / "studie.json")
    loaded, geo = load_study_json(tmp_path / "studie.json")            # GUI and MCP: the geometry relative to the folder of the file
    assert geo == steps["beam"].resolve() and loaded.geometry == str(geo)
    r = client.post("/api/open-gui", json={"path": str(tmp_path / "studie.json")})
    assert r.status_code == 200 and client.gui.calls == [(tmp_path / "studie.json").resolve()]
    meta = tmp_path / "v1.json"
    meta.write_text(json.dumps({"source": "Test", "parts": [{"name": "Konzola", "material": "S235"}]}),
                    encoding="utf-8")
    r = client.post("/api/open-gui", json={"path": str(meta)})
    assert r.status_code == 400 and "CAD metadata" in r.json()["error"]
    Study(name="Bez STEP", geometry="neni.step").save(tmp_path / "bez.json")
    r = client.post("/api/open-gui", json={"path": str(tmp_path / "bez.json")})
    assert r.status_code == 400 and "Study geometry not found" in r.json()["error"]
    Study(name="Members", geometry="").save(tmp_path / "pruty.json")    # frame system only – without STEP
    assert load_study_json(tmp_path / "pruty.json")[1] is None
    assert client.post("/api/open-gui", json={"path": str(tmp_path / "pruty.json")}).status_code == 200
    assert len(client.gui.calls) == 2


def test_token(tmp_path):
    c = TestClient(create_app(base_dir=tmp_path / "r", token="tajne-123", gui_launcher=lambda p: 0))
    assert c.get("/api/status").status_code == 401
    assert c.get("/api/status", headers={"Authorization": "Bearer spatne"}).status_code == 401
    r = c.get("/api/status", headers={"Authorization": "Bearer tajne-123"})
    assert r.status_code == 200 and r.json()["app"] == "FEA Studio"
    assert c.post("/api/tools/fea_materials_list", json={}).status_code == 401
    # the documentation and the API schema are also behind the authentication with the token
    assert c.get("/docs").status_code == 401 and c.get("/openapi.json").status_code == 401
    assert c.get("/openapi.json", headers={"Authorization": "Bearer tajne-123"}).status_code == 200


def test_host_and_origin_checks(tmp_path, steps):
    """Without a token only Host = loopback (DNS rebinding); a POST with the Origin of a foreign page (CSRF) is always rejected."""
    app = create_app(base_dir=tmp_path / "r", gui_launcher=lambda p: 0)
    for base in ("http://127.0.0.1:8765", "http://localhost", "http://[::1]:8765"):
        assert TestClient(app, base_url=base).get("/api/status").status_code == 200, base
    evil = TestClient(app, base_url="http://utocnik.example:8765")
    r = evil.get("/api/status")
    assert r.status_code == 403 and "Host" in r.json()["error"]
    assert evil.get("/docs").status_code == 403
    c = TestClient(app, base_url="http://127.0.0.1:8765")
    r = c.post("/api/tools/fea_materials_list", json={}, headers={"Origin": "https://utocnik.example"})
    assert r.status_code == 403 and "foreign" in r.json()["error"]
    r = c.post("/api/import", files={"file": ("x.step", steps["beam"].read_bytes(), "application/step")},
               data={"open": "false"}, headers={"Origin": "null"})
    assert r.status_code == 403
    assert c.get("/api/status", headers={"Origin": "https://utocnik.example"}).status_code == 200   # read-only
    for origin in ("http://127.0.0.1:8765", "http://localhost:3000"):           # /docs, local tools
        assert c.post("/api/tools/fea_materials_list", json={}, headers={"Origin": origin}).status_code == 200
    # with the token (remote access) Host is not checked, Origin = the same server is fine
    t = TestClient(create_app(base_dir=tmp_path / "t", token="x", gui_launcher=lambda p: 0),
                   base_url="http://fea-server:8765", headers={"Authorization": "Bearer x"})
    assert t.get("/api/status").status_code == 200
    assert t.post("/api/tools/fea_materials_list", json={},
                  headers={"Origin": "http://fea-server:8765"}).status_code == 200
    assert t.post("/api/tools/fea_materials_list", json={},
                  headers={"Origin": "http://jiny-server"}).status_code == 403


def test_gui_pid_protected_from_kill_children():
    """A GUI started through /api/open-gui (and its descendants) is not terminated by kill_children when the time is exceeded."""
    import subprocess
    import sys
    from fea_core.mcp.common import kill_children, protect_pid, _PROTECTED_PIDS
    code = "import time; time.sleep(30)  # ccx"
    p = subprocess.Popen([sys.executable, "-c", code])
    try:
        protect_pid(p.pid)
        kill_children(patterns=("ccx",))
        assert p.poll() is None                         # the protected process runs
        _PROTECTED_PIDS.discard(p.pid)
        kill_children(patterns=("ccx",))
        p.wait(10)
        assert p.returncode is not None                 # the unprotected one is terminated
    finally:
        _PROTECTED_PIDS.discard(p.pid)
        if p.poll() is None:
            p.kill()


def test_headless_import_without_qt():
    import subprocess
    import sys
    code = ("import sys; from fea_core.rest import create_app; create_app(base_dir='.'); "
            "print('QT' if any(m.startswith(('PySide6', 'fea_gui')) for m in sys.modules) else 'OK')")
    r = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent,
                       capture_output=True, text=True, timeout=120)
    assert r.stdout.strip().endswith("OK"), r.stdout + r.stderr


def test_cli_refuses_remote_without_token():
    from fea_core.rest.__main__ import main
    assert main(["--host", "0.0.0.0"]) == 2


def test_render_image_url(client, steps):
    """An image of a tool → a URL for download (PNG in the working folder) and optionally base64."""
    assert upload(client, steps["beam"]).status_code == 200
    r = call(client, "fea_model_render_image", view="iso", width=400, height=300)
    img = r["images"][0]
    assert img["url"].startswith("/api/files/") and "base64" not in img
    assert png_size(client.get(img["url"]).content) == (400, 300)
    r = client.post("/api/tools/fea_model_render_image?images=base64", json={"width": 320, "height": 200}).json()
    import base64
    assert png_size(base64.b64decode(r["images"][0]["base64"])) == (320, 200)


# ===================================================================== e2e

@pytest.mark.slow
def test_e2e_v1_cantilever(client, steps):
    """Console V1 through REST: import of the STEP → material → clamping → 100 N → mesh → calculation → summary."""
    meta = {"source": "Test", "document": "V1 konzola", "parts": [{"name": "Konzola", "material": "Ocel S235"}]}
    d = upload(client, steps["beam"], meta).json()
    assert d["parts"][0]["material"] == "Ocel S235"
    call(client, "fea_materials_assign", part="*", material_name="Ocel S235", custom_props={"E": 210000, "nu": 0.3})
    faces = call(client, "fea_geometry_find_faces", x=0)["result"]
    call(client, "fea_constraint_add_fixed", name="vetknuti", selection=[faces["faces"][0]["id"]])
    call(client, "fea_loadcase_create", name="Sila 100 N")
    call(client, "fea_load_add_force", case_name="Sila 100 N", name="konec", selection={"x": 200},
         force=[0, 0, -100])
    assert call(client, "fea_check_readiness")["result"]["ready"]
    m = call(client, "fea_mesh_generate", global_size=2.5)["result"]
    assert m["element_type"] == "C3D10" and m["dof"] > 30000
    r = call(client, "fea_solve_run", timeout_s=600)["result"]
    st = r["steps"][0]
    assert r["backend"] == "ccx" and st["equilibrium"]["ok"]
    assert abs(st["reactions"]["vetknuti"]["force_N"][2] - 100) < 0.01
    p = call(client, "fea_results_probe", point=[200, 10, 5])["result"]
    uz = -p["probes"][0]["u_mm"][2]
    assert abs(uz - 0.7634) / 0.7634 < 0.03, uz
    s0 = call(client, "fea_results_summary")["result"]["steps"][0]
    assert abs(s0["max_displacement_mm"]["value"] - 0.76) < 0.03 and s0["min_safety_Re"] > 3
    assert client.get("/api/status").json()["results"]
    img = call(client, "fea_results_render_image", field="mises", view="iso", width=640, height=400)["images"][0]
    assert png_size(client.get(img["url"]).content) == (640, 400)
    fea = Path(client.sess.workdir) / "v1.fea"
    assert call(client, "fea_study_save", path=str(fea))["result"]["with_results"]
    assert client.get(img["url"].rsplit("/images/", 1)[0] + "/v1.fea").status_code == 200
