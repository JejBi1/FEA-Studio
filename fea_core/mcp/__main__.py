"""Starting the MCP server: python -m fea_core.mcp [--transport stdio|sse|streamable-http] [--port 8766].

Default port of HTTP 8766 – 8765 belongs to the REST API (fea_core.rest, CAD adapters), both servers thus run at the same time.
HTTP without a token only on the address of this computer (127.0.0.1 / localhost / ::1) with the check of the Host and Origin headers
(DNS rebinding); another address requires --token (FEA_MCP_TOKEN) – as the REST API. The tools read and write
files at arbitrary paths and start calculations.

When the server ends (the client closed stdin, Ctrl+C) the running calculations are cancelled and the child processes (Gmsh, ccx, MPI)
end; in Windows the server is moreover in a Job Object, so the children end even on a hard termination of the server.
"""
from __future__ import annotations

import argparse
import hmac
import os
import sys

LOCAL = {"127.0.0.1", "::1", "localhost"}


def _is_loopback(host: str) -> bool:
    if host in LOCAL:
        return True
    import ipaddress
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _host_pattern(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def security_settings(host: str, token: str | None):
    """Check of Host / Origin (DNS rebinding) – the SDK turns it on by itself only for exactly 127.0.0.1 / localhost / ::1.
    Outside the loopback (only with a token) the token protects: the browser of a foreign page does not know it.
    """
    from mcp.server.transport_security import TransportSecuritySettings
    if not _is_loopback(host):
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    hosts = ["127.0.0.1", "localhost", "[::1]"]
    h = _host_pattern(host)
    if h not in hosts:
        hosts.append(h)
    return TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                     allowed_hosts=[f"{x}:*" for x in hosts],
                                     allowed_origins=[f"http://{x}:*" for x in hosts])


class BearerAuth:
    """ASGI middleware: HTTP requests only with the header Authorization: Bearer <token> (otherwise 401)."""

    def __init__(self, app, token: str):
        self.app = app
        self.expected = f"Bearer {token}".encode("utf-8")

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            got = b""
            for k, v in scope.get("headers") or []:
                if k.lower() == b"authorization":
                    got = v
                    break
            if not hmac.compare_digest(got, self.expected):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                body = "Missing or invalid token (header Authorization: Bearer <token>)".encode("utf-8")
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                                        (b"www-authenticate", b"Bearer"),
                                        (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def http_app(srv, transport: str, host: str, token: str | None):
    """Starlette application for sse / streamable-http with the check of Host/Origin and optionally a token."""
    sec = security_settings(host, token)
    if transport == "sse":
        app = srv.sse_app(transport_security=sec, host=host)
    else:
        app = srv.streamable_http_app(transport_security=sec, host=host)
    return BearerAuth(app, token) if token else app


def _kill_children_on_exit():
    """Windows: a Job Object with KILL_ON_JOB_CLOSE, to which the server and its children belong – after the end of the server (also a hard termination by the client)
    the system ends Gmsh, ccx, mpiexec… (without it they ran on as orphans). Returns the descriptor (must
    stay open for the whole run), elsewhere / on an error None.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(job)
            return None
        # the server passes the job to the children; but the Python from the Microsoft Store / the venv launcher detaches the children from the job
        # (verified 15. 10. 2026) → every process started through subprocess is assigned to the job immediately after its
        # start (wrapper of Popen, only in the process of the server) and the watchdog thread assigns the other children too (within 0.5 s)
        k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())
        _assign_on_popen(k32, job)
        _adopt_children(k32, job)
        return job
    except Exception:  # noqa: BLE001 – only a safeguard, the server runs even without it
        return None


def _assign_on_popen(k32, job) -> None:
    """subprocess.Popen in the process of the server: a new process (ccx, the Gmsh worker, mpiexec…) goes into the job at once –
    its children then inherit the job (the job's breakaway is not allowed).
    """
    import subprocess
    from ctypes import wintypes
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    orig = subprocess.Popen.__init__
    if getattr(orig, "_fea_job", False):
        return

    def init(self, *a, **k):
        orig(self, *a, **k)
        try:
            h = getattr(self, "_handle", None)
            if h:
                k32.AssignProcessToJobObject(job, int(h))
        except Exception:  # noqa: BLE001 – only a safeguard
            pass
    init._fea_job = True
    subprocess.Popen.__init__ = init


def _adopt_children(k32, job) -> None:
    """Thread: every 0.5 s assigns the new children of the server (Gmsh, ccx, mpiexec…) to the Job Object. The protected processes
    (common.protect_pid) stay outside.
    """
    import threading
    import time
    from ctypes import wintypes
    try:
        import psutil
    except ImportError:
        return
    from fea_core.mcp.common import _PROTECTED_PIDS
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    seen: set[tuple[int, float]] = set()

    def loop():
        me = psutil.Process()
        while True:
            try:
                protected = set(_PROTECTED_PIDS)
                for ch in me.children(recursive=True):
                    try:
                        key = (ch.pid, ch.create_time())
                        if key in seen or ch.pid in protected or ch.ppid() in protected:
                            continue
                        seen.add(key)
                        h = k32.OpenProcess(0x0100 | 0x0001, False, ch.pid)   # SET_QUOTA | TERMINATE
                        if h:
                            k32.AssignProcessToJobObject(job, h)
                            k32.CloseHandle(h)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except Exception:  # noqa: BLE001 – the watchdog must not bring down the server
                pass
            time.sleep(0.5)
    threading.Thread(target=loop, daemon=True, name="fea-mcp child job").start()


def shutdown(srv) -> None:
    """Cancels the running tool calls and the asynchronous jobs and ends the child processes (Gmsh, ccx, MPI)."""
    from fea_core.mcp.common import ACTIVE_CALLS, kill_children
    from fea_core.mcp.tools_jobs import jobs
    for ev in ACTIVE_CALLS.copy():            # copy() is atomic (the calling threads change the set)
        ev.set()
    sess = getattr(srv, "fea_session", None)
    threads = []
    if sess is not None:
        for j in jobs(sess).values():
            if j.active:
                j.cancel.set()
                if j.thread is not None:
                    threads.append(j.thread)
    kill_children()
    for th in threads:
        th.join(5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fea_core.mcp", description="FEA Studio MCP server (headless)")
    ap.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio",
                    help="stdio (default, local LLM clients) | sse | streamable-http")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address for sse / streamable-http (default 127.0.0.1 = this computer only; any other address requires --token)")
    ap.add_argument("--port", type=int, default=8766,
                    help="port for sse / streamable-http (default 8766; REST API uses 8765)")
    ap.add_argument("--token", default=os.environ.get("FEA_MCP_TOKEN") or None,
                    help="HTTP: require the header Authorization: Bearer <token> (default: variable FEA_MCP_TOKEN)")
    ap.add_argument("--groups", default=os.environ.get("FEA_MCP_GROUPS") or "core",
                    help="tool groups exposed at start: comma list or 'all' (default core; variable FEA_MCP_GROUPS). "
                         "More at run time: tool fea_tools_enable")
    ap.add_argument("--workdir", help="root of the study working folders (default: user data folder, FEAStudio/mcp)")
    a = ap.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if any(x == "--token" or x.startswith("--token=") for x in (argv if argv is not None else sys.argv[1:])):
        print("Warning: the token on the command line is visible to other processes (process list) – preferably use the FEA_MCP_TOKEN variable.", file=sys.stderr)
    if a.transport != "stdio" and not _is_loopback(a.host) and not a.token:
        print(f"Refused: address {a.host} is reachable from other computers – specify --token (or FEA_MCP_TOKEN). The MCP tools read and write files and run analyses; without a token the server may run only on 127.0.0.1.", file=sys.stderr)
        return 2
    # the project on sys.path (start from elsewhere: python -m fea_core.mcp with cwd outside the project and PYTHONPATH)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from fea_core import modules
    try:
        modules.require("mcp")
    except modules.ModuleMissing as e:
        print(e, file=sys.stderr)
        return 3
    job = _kill_children_on_exit()        # noqa: F841 – the descriptor keeps the object of the job for the whole run
    from fea_core.mcp.server import build_server
    from fea_core.mcp.session import cleanup_workdirs
    try:
        srv = build_server(base_dir=a.workdir, groups_spec=a.groups)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    import threading                      # cleaning of old automatic working folders (FEA_MCP_KEEP_DAYS)
    threading.Thread(target=cleanup_workdirs, args=(srv.fea_session.base_dir,), daemon=True,
                     name="fea-mcp cleanup").start()
    try:
        if a.transport == "stdio":
            srv.run("stdio")
        else:
            import uvicorn
            uvicorn.run(http_app(srv, a.transport, a.host, a.token), host=a.host, port=a.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(srv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
