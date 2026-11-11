"""REST API of FEA Studio (FastAPI) – a thin layer over the tools of the MCP server.

The tools (`/api/tools/{name}`) are the same functions as in `python -m fea_core.mcp` (the same names, descriptions,
parameter schemas and error messages), one session `Session` per process. In addition convenient endpoints for the CAD
adapters: import of a STEP with metadata, download of files from the working folder, starting the GUI.

Synchronous endpoints (def) run in threads of FastAPI → a long calculation does not block /api/status.
"""
from __future__ import annotations

import base64
import hmac
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..mcp.session import FeaError, Session
from ..cad_import import apply_metadata
from .imports import MAX_STEP_BYTES, STEP_SUFFIXES, library_material, parse_metadata, save_import
from .registry import collect_tools

log = logging.getLogger("fea_core.rest")

ROOT = Path(__file__).resolve().parent.parent.parent       # project root (cwd for the GUI)
GUI_SUFFIXES = STEP_SUFFIXES + (".fea", ".json")
BUSY_PREFIX = "Server is busy"                     # text FeaError ze Session.use


class ApiError(Exception):
    def __init__(self, status: int, message: str, **extra):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


def launch_gui(path: Path) -> int:
    """Starts the GUI (`pythonw -m fea_gui <file>`) as a separate process independent of the server."""
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    if pyw.exists():
        exe = pyw
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen([str(exe), "-m", "fea_gui", str(path)], cwd=str(ROOT), creationflags=flags,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         close_fds=True, start_new_session=os.name != "nt")
    from ..mcp.common import protect_pid
    protect_pid(p.pid)                   # time limit of the tools (kill_children) does not terminate the GUI nor its ccx
    return p.pid


def _is_within(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
PROTECTED_PREFIXES = ("/api", "/docs", "/redoc", "/openapi.json")       # after the token (if set)
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
HELLO_PATH = "/api/hello"           # token proof (challenge–response) – reachable without the token
NONCE_CHARS = frozenset("0123456789abcdef")


def hello_proof(token: str, nonce: str, port: int | None) -> str:
    """Proof that the server knows the token, without sending it: HMAC-SHA256(token, "rest" + nonce + ":" + port) in hex.
    The port is the one the server listens on (the client uses the port of its URL) – a foreign process on another port cannot
    relay the client's challenge to the real server and pass the answer on as its own.
    """
    import hashlib
    msg = ("rest" + nonce + ":" + str(port or "")).encode("ascii")
    return hmac.new(token.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _host_name(value: str) -> str:
    """Host header → name without the port ('127.0.0.1:8765' → '127.0.0.1', '[::1]:8765' → '::1')."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        return v[1:v.find("]")] if "]" in v else v[1:]
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def create_app(session: Session | None = None, base_dir: str | Path | None = None, token: str | None = None,
               gui_launcher: Callable[[Path], Any] | None = None,
               allowed_hosts: set[str] | None = None) -> FastAPI:
    """FastAPI application. token = require the header `Authorization: Bearer <token>` (None = without verification);
    gui_launcher = replacement of the start of the GUI (tests).

    Without a token the server runs only on the loopback → the Host header must be 127.0.0.1 / localhost / [::1] (protection against
    DNS rebinding: a foreign page with a domain redirected to 127.0.0.1); allowed_hosts = other allowed names.
    POST/PUT/PATCH/DELETE with an Origin header of a foreign domain (the browser of a web page – CSRF, e.g. multipart
    /api/import) are always refused; an Origin = loopback or the same Host (the documentation /docs) is fine.
    """
    from .. import __version__
    sess = session or Session(Path(base_dir).resolve() if base_dir else None)
    tools = collect_tools(sess)
    launcher = gui_launcher or launch_gui
    state: dict = {"last_import": None}
    issued: set[str] = set()    # files outside base_dir whose URL the server returned (only those can be downloaded through @workdir)
    hosts = None if (token and allowed_hosts is None) else (LOOPBACK_HOSTS | {h.lower() for h in allowed_hosts or ()})

    docs = bool(token)          # without a token no /docs and no /redoc (scripts from a CDN could call the tools)
    app = FastAPI(title="FEA Studio REST API", version=__version__,
                  docs_url="/docs" if docs else None, redoc_url="/redoc" if docs else None,
                  description="Linear FEM statics (STEP → Gmsh → CalculiX/NGSolve). Tools = MCP server of FEA Studio; units mm, N, MPa, t, s.")
    app.state.session = sess
    app.state.tools = tools

    # ------------------------------------------------------------ verification and errors
    @app.middleware("http")
    async def auth(request: Request, call_next):
        host = request.headers.get("host", "")
        if hosts is not None and _host_name(host) not in hosts:
            return JSONResponse({"error": f"Host header '{host}' not allowed – the server without a token accepts only 127.0.0.1 / localhost"}, status_code=403)
        origin = request.headers.get("origin")
        if origin is not None and request.method in UNSAFE_METHODS:
            from urllib.parse import urlsplit
            o = urlsplit(origin.strip())
            same = o.netloc.lower() == host.strip().lower()
            if not (same or (o.scheme in ("http", "https") and (o.hostname or "") in LOOPBACK_HOSTS)):
                return JSONResponse({"error": f"Request from a foreign web page ({origin}) rejected"}, status_code=403)
        if token and request.url.path.startswith(PROTECTED_PREFIXES) and request.url.path != HELLO_PATH:
            h = request.headers.get("authorization", "")
            if not (h.startswith("Bearer ") and hmac.compare_digest(h[7:].strip().encode(), token.encode())):
                return JSONResponse({"error": "Token missing or wrong (header Authorization: Bearer …)"},
                                    status_code=401, headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    @app.exception_handler(ApiError)
    async def _api_error(request, e: ApiError):
        return JSONResponse({"error": e.message, **e.extra}, status_code=e.status)

    @app.exception_handler(FeaError)
    async def _fea_error(request, e: FeaError):
        msg = str(e)
        return JSONResponse({"error": msg}, status_code=409 if msg.startswith(BUSY_PREFIX) else 400)

    def check_busy():
        if sess.busy:
            raise ApiError(409, f"{BUSY_PREFIX} ({sess.busy}) – try again in a moment.", busy=sess.busy)

    # ------------------------------------------------------------ soubory
    def roots() -> list[tuple[str, Path]]:
        r = [("", sess.base_dir)]
        if sess.workdir is not None and not _is_within(sess.workdir, sess.base_dir):
            r.append(("@workdir/", Path(sess.workdir)))
        return r

    def _servable(p: Path) -> bool:
        """Allow-list below base_dir: the imports folder, work folders created by the server (mark), the current work
        folder of the session and files whose URL the server returned. The rest of the base folder (IPC session key in
        ipc/, active_session.json …) is not served."""
        from ..ipc import token_path
        from ..mcp.session import WORKDIR_MARK
        rp = p.resolve()
        if rp.suffix.lower() == ".token" or _is_within(rp, token_path().parent):
            return False
        if str(rp).lower() in issued:
            return True
        if sess.workdir is not None and _is_within(rp, Path(sess.workdir)) and Path(sess.workdir).resolve() != \
                Path(sess.base_dir).resolve():
            return True
        try:
            rel = rp.relative_to(Path(sess.base_dir).resolve())
        except ValueError:
            return False
        if len(rel.parts) < 2:
            return False                                  # files directly in the base folder
        top = Path(sess.base_dir).resolve() / rel.parts[0]
        return rel.parts[0].lower() == "imports" or (top / WORKDIR_MARK).is_file()

    def file_url(p: Path) -> str | None:
        for prefix, root in roots():
            if _is_within(p, root):
                rel = p.resolve().relative_to(Path(root).resolve()).as_posix()
                if prefix:
                    issued.add(str(p.resolve()).lower())
                return "/api/files/" + quote(prefix + rel)
        return None

    def to_payload(value, images: str) -> dict:
        """Result of a tool (str JSON | [str, Image]) → JSON response; pictures as a URL and/or base64 PNG."""
        import json
        items = value if isinstance(value, (list, tuple)) else [value]
        texts, imgs = [], []
        for it in items:
            if isinstance(it, str):
                try:
                    texts.append(json.loads(it))
                except ValueError:
                    texts.append(it)
            elif hasattr(it, "to_image_content"):           # mcp Image
                d: dict = {"mime_type": getattr(it, "_mime_type", "image/png")}
                path = getattr(it, "path", None)
                url = file_url(Path(path)) if path else None
                if path:
                    d["path"] = str(path)
                if url:
                    d["url"] = url
                if images in ("base64", "both") or not url:
                    data = Path(path).read_bytes() if path else it.data
                    d["base64"] = base64.b64encode(data).decode("ascii")
                imgs.append(d)
            else:
                texts.append(it)
        out = {"result": texts[0] if len(texts) == 1 else texts}
        if imgs:
            out["images"] = imgs
        return out

    # ------------------------------------------------------------ endpointy
    @app.post(HELLO_PATH)
    def hello(request: Request, body: dict | None = Body(default=None)) -> dict:
        """Token proof for clients (CAD add-ins) BEFORE they send the token or a design: {"nonce": 32–128 hex characters} →
        {"app", "token": true, "proof": HMAC-SHA256(token, "rest" + nonce + ":" + port)}; a server without a token answers
        {"app", "token": false} (the client then must not send its token). Needs no token itself."""
        nonce = str((body or {}).get("nonce", "")).lower()
        if not (32 <= len(nonce) <= 128 and set(nonce) <= NONCE_CHARS):
            raise ApiError(400, "nonce must be 32–128 hexadecimal characters")
        out: dict = {"app": "FEA Studio", "token": bool(token)}
        if token:
            server = request.scope.get("server") or (None, None)
            out["proof"] = hello_proof(token, nonce, server[1])
        return out

    @app.get("/api/status")
    def status() -> dict:
        from ..solvers import summary as solvers_summary
        st = sess.study                                   # without a lock – also during a calculation
        return {"app": "FEA Studio", "version": __version__, "busy": sess.busy,
                "study": st.name if st is not None else None,
                "step_file": str(sess.step_path) if sess.step_path else None,
                "workdir": str(sess.workdir) if sess.workdir else None,
                "results": sess.results is not None, "base_dir": str(sess.base_dir),
                "last_import": state["last_import"], "tools": len(tools),
                "solvers": solvers_summary()}             # availability of the solvers (what is missing and how to install it)

    @app.get("/api/tools")
    def list_tools() -> dict:
        return {"tools": [t.info() for t in tools.values()]}

    @app.get("/api/tools/{name}")
    def tool_info(name: str) -> dict:
        if name not in tools:
            raise ApiError(404, f"Unknown tool '{name}' (list: GET /api/tools)")
        return tools[name].info()

    @app.post("/api/tools/{name}")
    def call_tool(name: str, args: dict | None = Body(default=None), images: str = "url") -> dict:
        """Calls a tool; body = JSON object of the arguments. images = url | base64 | both (pictures of the render tools)."""
        if name not in tools:
            raise ApiError(404, f"Unknown tool '{name}' (list: GET /api/tools)")
        if images not in ("url", "base64", "both"):
            raise ApiError(400, "images must be url | base64 | both")
        t = tools[name]
        try:
            kw = t.validate(args or {})
        except ValueError as e:                            # pydantic ValidationError is a subclass of ValueError
            raise ApiError(400, f"Invalid arguments of tool {name}: {e}") from None
        from ..mcp.tools_jobs import NO_LOCK_TOOLS
        if name not in NO_LOCK_TOOLS:                      # state / cancellation of an asynchronous job also during a calculation
            check_busy()
        t0 = time.time()
        try:
            value = t.fn(**kw)
        except (FeaError, ApiError):
            raise
        except Exception as e:  # noqa: BLE001 – return the errors of the core (mesh, ccx, OCP) as text
            log.exception("Tool %s failed", name)
            raise ApiError(500, f"{type(e).__name__}: {e}") from e
        return {"tool": name, **to_payload(value, images), "time_s": round(time.time() - t0, 3)}

    @app.get("/api/study")
    def study_json() -> dict:
        """Current definition of the study (JSON of the model Study)."""
        with sess.use():
            if sess.study is None:
                return {"study": None}
            return sess.study.to_dict()

    @app.post("/api/import")
    def import_step(file: UploadFile = File(...), metadata: str | None = Form(None), open: bool = Form(True),
                    name: str | None = Form(None), assign_materials: bool = Form(True)) -> dict:
        """STEP from a CAD adapter (+ metadata JSON) → imports/; open = directly a new study (as fea_study_new)."""
        meta = parse_metadata(metadata)
        step, meta_path = save_import(sess.base_dir, file.filename, file.file, meta, MAX_STEP_BYTES)   # streamed
        state["last_import"] = str(step)
        out: dict = {"step_path": str(step), "metadata_path": str(meta_path) if meta_path else None,
                     "step_url": file_url(step), "size_bytes": step.stat().st_size}
        if meta:
            out["source"] = meta.get("source")
            out["document"] = meta.get("document")
            if meta.get("units") and str(meta["units"]).lower() not in ("mm", "millimeter", "millimeters"):
                out["note_units"] = (f"CAD units '{meta['units']}' – the STEP carries its own units, converted to mm on loading.")
        if not open:
            return out
        check_busy()
        doc_name = name or (meta or {}).get("document") or None
        import json
        # the metadata lie next to the STEP – fea_study_new would assign them itself; here they are assigned according to assign_materials
        res = json.loads(tools["fea_study_new"].fn(step_path=str(step), name=doc_name, apply_cad_metadata=False))
        out["study"] = res
        part_names = sess.part_names()
        found: dict[str, str | None] = {}            # material from the CAD → library material (also without assignment)

        def lookup(cad, props=None):
            lib, info = library_material(cad, props)
            found[cad] = lib
            return lib, info
        # matching and materials as the GUI when a STEP with metadata next to it is opened (fea_core.cad_import)
        matched = apply_metadata(sess.study, part_names, meta, assign=assign_materials, lookup=lookup,
                                 assign_fn=lambda pn, lib: tools["fea_materials_assign"].fn(part=pn, material_name=lib))
        rows, warnings = matched.rows, [f"Geometrie: {w}" for w in res.get("geometry_warnings", [])]

        def listing(names: list[str], n: int = 8) -> str:
            return ", ".join(names[:n]) + (f" … ({len(names)} in total)" if len(names) > n else "")
        if matched.unmatched:
            warnings.append("Metadata records without a part in the STEP: " + listing(matched.unmatched))
        st = sess.study                     # the first assignment also becomes the default material (fea_materials_assign)
        for r in rows:
            try:
                r["material_effective"] = st.material_of(r["name"]).name
            except KeyError:
                r["material_effective"] = None
        no_mat = [r["name"] for r in rows if r["material_effective"] is None]
        if no_mat:
            warnings.append("Parts without material (fea_materials_assign): " + listing(no_mat))
        for r in rows:                      # assign_materials=false: the material is in the library, only it was not assigned
            cad = (r.get("material_match") or {}).get("cad_material")
            if "material" not in r and cad and found.get(cad):
                r["material_suggested"] = found[cad]
        if matched.not_in_library:          # only the materials that really are not in the library
            by_name = {r["name"]: r for r in rows}
            dflt = [pn for parts in matched.not_in_library.values() for pn in parts
                    if by_name.get(pn, {}).get("material_effective")]
            hints = [f"{c}: {h}" for c, h in matched.hints.items()]
            warnings.append(f"Material from CAD is not in the library: {', '.join(sorted(matched.not_in_library))}"
                            + (f" – parts {listing(dflt)} got the default material of the study '{st.default_material}', check fea_materials_assign" if dflt else "")
                            + (f" ({'; '.join(hints)})" if hints else ""))
        out["parts"] = rows
        out["warnings"] = warnings
        return out

    @app.get("/api/files/{path:path}")
    def get_file(path: str):
        """Download of a file from the working folder of the session (PNG, .docx, .fea, .step …). Outside it → 403."""
        prefix, root = "", sess.base_dir
        if path.startswith("@workdir/") and sess.workdir is not None:
            prefix, root, path = "@workdir/", Path(sess.workdir), path[len("@workdir/"):]
        if Path(path).is_absolute() or ":" in path:
            raise ApiError(403, "Absolute paths are not allowed – the path is relative to the work folder")
        p = (Path(root) / path)
        if not _is_within(p, Path(root)):
            raise ApiError(403, "Path outside the work folder")
        if prefix and str(p.resolve()).lower() not in issued:
            raise ApiError(403, "Only files created by the server itself can be downloaded from a work folder outside the default one")
        if not prefix and not _servable(p):
            raise ApiError(403, "Only files of the session work folders and of the imports can be downloaded")
        if not p.is_file():
            raise ApiError(404, f"File not found: {prefix}{path}")
        return FileResponse(p, filename=p.name)

    @app.post("/api/open-gui")
    def open_gui(body: dict | None = Body(default=None)) -> dict:
        """Starts the GUI of FEA Studio on a file: {"path": ".step|.fea|.json"} (relative to the working folder
        or absolute; .json = study from fea_study_save, not metadata from the CAD); without a path the last import, otherwise the saved / opened study of the session (.fea, .json) or its STEP.
        """
        body = body or {}
        raw = body.get("path") or state["last_import"] or (str(sess.study_path or sess.step_path or "") or None)
        if not raw:
            raise ApiError(400, "Nothing to open – give {\"path\": …} or first POST /api/import")
        from ..pathcheck import is_unc
        if is_unc(raw):
            raise ApiError(400, "Network paths (UNC) are not allowed")
        p = Path(raw)
        if not p.is_absolute():
            p = sess.base_dir / p
        if p.suffix.lower() not in GUI_SUFFIXES:
            raise ApiError(400, f"The GUI opens only {', '.join(GUI_SUFFIXES)}")
        if not p.is_file():
            raise ApiError(404, f"File not found: {p}")
        if p.suffix.lower() == ".json":            # the GUI opens a study JSON, not the metadata from the CAD next to the STEP
            from ..project import load_study_json
            try:
                load_study_json(p)
            except (OSError, ValueError) as e:     # it is not a study, invalid JSON, its STEP is missing
                raise ApiError(400, str(e)) from None
        pid = launcher(p.resolve())
        return {"opened": str(p.resolve()), "pid": pid}

    return app
