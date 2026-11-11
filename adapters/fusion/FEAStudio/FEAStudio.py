"""FEA Studio – add-in for Autodesk Fusion: the command "Send to FEA Studio".

Exports the active design (the whole one, or only the SELECTED parts / subassemblies – the others are hidden for the time of the export, their position
in the assembly stays) to STEP, collects the names of the bodies and occurrences, the materials (name + mechanical properties), the colours
of the appearance and the attributes and passes them to FEA Studio:
  1. if the REST server of FEA Studio runs (`python -m fea_core.rest`), it sends the STEP + metadata to POST /api/import
     (opens a study) and optionally starts the GUI through POST /api/open-gui;
  2. otherwise it saves the STEP + JSON into the export folder and starts the GUI of FEA Studio (FEA_STUDIO_HOME) with the STEP.
With a token set, the server must first prove it knows the token (POST /api/hello) – the token and the design go only to such a
server; on this computer every process listening on the port must also belong to the current user (Windows).

Only the standard library of Python (Fusion has its own Python without packages). Settings: settings.json next to
this file (the pattern settings.example.json), the environment variables FEA_REST_URL, FEA_REST_TOKEN, FEA_STUDIO_HOME.
"""
import contextlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import uuid

try:
    import adsk.core
    import adsk.fusion
except ImportError:          # outside Fusion (tests) – only the helper functions without a command
    adsk = None

CMD_ID = "FEAStudio_SendToFEA"
UPD_ID = "FEAStudio_UpdateFEA"
UPD_NAME = "Update in FEA Studio"
UPD_TIP = ("Edited model → the open FEA Studio study: new geometry, the settings stay (face selections are "
           "remapped by the face identity in Fusion) and the next iteration is computed")
CMD_NAME = "Send to FEA Studio"
CMD_TIP = ("Export of the whole design or only the selected parts / subassemblies to STEP + names, materials, colors "
           "and attributes → FEA Studio")
PANEL_ID = "SolidScriptsAddinsPanel"          # Tools (Utilities) → ADD-INS
# an own panel with a large button between INSERT and SELECT in all tabs of the Design that have INSERT
# (SOLID, SURFACE, MESH, FORM, SHEET METAL, PLASTIC, SKETCH…); the id of the panel is per tab (FEAStudioPanel_<tab>)
OWN_PANEL = "FEAStudioPanel"
OWN_PANEL_NAME = "FEA STUDIO"
WORKSPACE_ID = "FusionSolidEnvironment"
INSERT_PANELS = ("InsertPanel", "InsertAssemblePanel")
SELECT_PANELS = ("SelectPanel", "ParaMeshSelectPanel")
HERE = os.path.dirname(os.path.abspath(__file__))
ICONS = os.path.join(HERE, "resources", "SendToFEA")        # 16x16, 32x32 (+ @2x)
SOURCE = "Autodesk Fusion"

_handlers = []

# ===================================================================== settings

DEFAULTS = {
    "server": "http://127.0.0.1:8765",       # REST API FEA Studia
    "token": "",                              # Authorization: Bearer (if the server is started with --token)
    "fea_studio_home": "",                    # root of the installation of FEA Studio (to start the GUI without the server)
    "export_dir": "",                         # folder for the STEP + JSON (default %TEMP%/FEAStudio/export)
    "open_gui": True,                         # open the GUI after sending
    "open_study": True,                       # create a study on the server at once (fea_study_new)
    "timeout_s": 300,
}


def _home_from_location():
    """The add-in started directly from the folder of FEA Studio (adapters/fusion/FEAStudio) → root of the installation, otherwise ""."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    return root if os.path.isdir(os.path.join(root, "fea_gui")) else ""


def load_settings(path=None):
    s = dict(DEFAULTS)
    s["fea_studio_home"] = _home_from_location()
    path = path or os.path.join(HERE, "settings.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            s.update({k: v for k, v in json.load(f).items() if v not in (None, "")})
    env = {"server": "FEA_REST_URL", "token": "FEA_REST_TOKEN", "fea_studio_home": "FEA_STUDIO_HOME"}
    for k, e in env.items():
        if os.environ.get(e):
            s[k] = os.environ[e]
    if not s["export_dir"]:
        s["export_dir"] = os.path.join(tempfile.gettempdir(), "FEAStudio", "export")
    return s


def safe_name(text):
    t = re.sub(r"[^\w.-]+", "_", str(text or ""), flags=re.UNICODE).strip("_")
    return t[:60] or "model"


# ===================================================================== REST klient (stdlib)

def _headers(settings, extra=None):
    h = {"Accept": "application/json"}
    if settings.get("token"):
        h["Authorization"] = "Bearer " + settings["token"]
    h.update(extra or {})
    return h


class UntrustedServer(RuntimeError):
    """The process listening on the loopback port of the REST API does not belong to the current user."""


_LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def _server_host_port(settings):
    from urllib.parse import urlsplit
    u = urlsplit(settings["server"])
    try:
        port = u.port or (443 if u.scheme == "https" else 80)
    except ValueError:
        port = None
    return (u.hostname or "").lower(), port


def _listener_pids(port):
    """PIDs of the processes listening on the TCP port (IPv4 and IPv6) – Windows, iphlpapi.GetExtendedTcpTable."""
    import ctypes
    from ctypes import wintypes
    iphlp = ctypes.WinDLL("iphlpapi")
    pids = set()
    # (address family, row size in DWORDs, index of the local port, index of the owning PID) of MIB_TCP(6)ROW_OWNER_PID
    for af, row_dw, port_i, pid_i in ((2, 6, 2, 5), (23, 14, 5, 13)):
        size = wintypes.DWORD(0)
        iphlp.GetExtendedTcpTable(None, ctypes.byref(size), False, af, 3, 0)    # 3 = TCP_TABLE_OWNER_PID_LISTENER
        for _ in range(4):
            buf = (ctypes.c_ubyte * max(size.value, 4))()
            rc = iphlp.GetExtendedTcpTable(buf, ctypes.byref(size), False, af, 3, 0)
            if rc != 122:                                                        # ERROR_INSUFFICIENT_BUFFER: the table grew
                break
        if rc != 0:
            if af == 23:
                continue                                                         # no IPv6 stack
            raise UntrustedServer("Cannot read the TCP table (error {})".format(rc))
        n = ctypes.cast(buf, ctypes.POINTER(wintypes.DWORD))[0]
        rows = ctypes.cast(ctypes.byref(buf, 4), ctypes.POINTER(wintypes.DWORD * row_dw))
        for i in range(n):
            r = rows[i]
            p = r[port_i] & 0xFFFF
            if ((p & 0xFF) << 8 | p >> 8) == port:                               # network byte order
                pids.add(int(r[pid_i]))
    return pids


def _process_is_mine(pid):
    """Does the process run under the same Windows account as this one (owner SID of the process token)?"""
    import ctypes
    from ctypes import wintypes
    k32, adv = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("advapi32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    adv.OpenProcessToken.argtypes = (wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE))
    adv.GetTokenInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD))
    adv.GetLengthSid.argtypes = (ctypes.c_void_p,)
    adv.GetLengthSid.restype = wintypes.DWORD

    def user_sid(hproc):
        tok = wintypes.HANDLE()
        if not adv.OpenProcessToken(hproc, 0x0008, ctypes.byref(tok)):          # TOKEN_QUERY
            return None
        try:
            need = wintypes.DWORD(0)
            adv.GetTokenInformation(tok, 1, None, 0, ctypes.byref(need))         # TokenUser
            buf = (ctypes.c_ubyte * max(need.value, 1))()
            if not adv.GetTokenInformation(tok, 1, buf, need, ctypes.byref(need)):
                return None
            sid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]          # TOKEN_USER.User.Sid
            ln = adv.GetLengthSid(ctypes.c_void_p(sid))
            return bytes((ctypes.c_ubyte * ln).from_address(sid))
        finally:
            k32.CloseHandle(tok)

    mine = user_sid(k32.GetCurrentProcess())
    h = k32.OpenProcess(0x1000, False, int(pid))                                # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False                                                             # another user's process: no access
    try:
        return mine is not None and user_sid(h) == mine
    finally:
        k32.CloseHandle(h)


def check_server(settings):
    """Before anything (the bearer token, the design) is sent to a REST server on this computer: every process listening on its
    port must belong to the current user – another user's / a foreign service's process that took the port first would otherwise
    receive the token and the model. A server on another computer is the user's explicit choice (not checked); outside Windows
    the check is not available. Raises UntrustedServer."""
    host, port = _server_host_port(settings)
    if os.name != "nt" or host not in _LOOPBACK or not port:
        return
    pids = _listener_pids(port)
    foreign = [p for p in pids if p != os.getpid() and not _process_is_mine(p)]
    if foreign:
        raise UntrustedServer("Port {} is held by a process of another user (PID {}) – nothing was sent to it".format(
            port, ", ".join(map(str, sorted(foreign)))))


def _hello_proof(token, nonce, port):
    """The same formula as fea_core.rest.app.hello_proof: HMAC-SHA256(token, "rest" + nonce + ":" + port)."""
    import hashlib
    import hmac
    msg = ("rest" + nonce + ":" + str(port or "")).encode("ascii")
    return hmac.new(token.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def prove_server(settings, timeout=10):
    """With a token in the settings the server must first prove that it knows the token (POST /api/hello, challenge–response)
    – only then the token and the design are sent. A server that cannot (an older FEA Studio without the endpoint, a server
    started without a token, a wrong proof) is not trusted: UntrustedServer, the caller falls back to the export folder.
    Without a token nothing is checked here (the port-owner check stays the protection). A refused connection propagates
    as URLError (= the server is not running)."""
    import hmac
    import secrets
    token = settings.get("token")
    if not token:
        return
    _host, port = _server_host_port(settings)
    nonce = secrets.token_hex(16)
    req = urllib.request.Request(settings["server"].rstrip("/") + "/api/hello", method="POST",
                                 data=json.dumps({"nonce": nonce}).encode("ascii"),
                                 headers={"Accept": "application/json", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ans = json.loads(r.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:              # 401 / 404: a server of an older version (no /api/hello)
        raise UntrustedServer("The FEA Studio server on port {} cannot prove the token (HTTP {}; an older version?) – "
                              "the token and the design were not sent".format(port, e.code)) from None
    except ValueError:
        raise UntrustedServer("The server on port {} gave an invalid answer to the token proof – nothing was sent "
                              "to it".format(port)) from None
    if not isinstance(ans, dict) or ans.get("token") is not True:
        raise UntrustedServer("The server on port {} does not use a token although one is set – the token and the design "
                              "were not sent".format(port))
    if not hmac.compare_digest(str(ans.get("proof", "")), _hello_proof(token, nonce, port)):
        raise UntrustedServer("The server on port {} could not prove the token – nothing was sent to it".format(port))


def _call(settings, method, path, body=None, ctype=None, timeout=None):
    check_server(settings)
    prove_server(settings, timeout=min(10, timeout or 10))
    url = settings["server"].rstrip("/") + path
    req = urllib.request.Request(url, data=body, method=method,
                                 headers=_headers(settings, {"Content-Type": ctype} if ctype else None))
    try:
        with urllib.request.urlopen(req, timeout=timeout or settings.get("timeout_s", 300)) as r:
            return json.loads(r.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error")
        except Exception:  # noqa: BLE001
            msg = None
        raise RuntimeError("FEA Studio (HTTP {}): {}".format(e.code, msg or e.reason)) from None


def server_status(settings):
    """State of the REST server or None when it does not run."""
    try:
        return _call(settings, "GET", "/api/status", timeout=3)
    except UntrustedServer as e:                     # treated as "no server": the STEP goes to the export folder + the local GUI
        _log_note(str(e))
        return None
    except (urllib.error.URLError, OSError, ValueError):
        return None


NOTES = []                                           # notes for the user (an untrusted server on the port)
_NOTE_SEQ = [0]                                      # number of notes logged (also repeated ones – deliver compares it)


def _log_note(text):
    _NOTE_SEQ[0] += 1
    if text in NOTES:
        NOTES.remove(text)                           # the latest note is always the last one
    NOTES.append(text)


def multipart(fields, files):
    """Body of multipart/form-data: fields {name: text}, files [(field, file name, bytes, type)]."""
    b = "----FEAStudio" + uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out += [("--" + b).encode(), 'Content-Disposition: form-data; name="{}"'.format(k).encode(), b"",
                str(v).encode("utf-8")]
    for field, fname, data, ctype in files:
        out += [("--" + b).encode(),
                'Content-Disposition: form-data; name="{}"; filename="{}"'.format(field, fname).encode("utf-8"),
                "Content-Type: {}".format(ctype).encode(), b"", data]
    out += [("--" + b + "--").encode(), b""]
    return b"\r\n".join(out), "multipart/form-data; boundary=" + b


def send_import(settings, step_path, meta, open_study=True):
    with open(step_path, "rb") as f:
        data = f.read()
    fields = {"metadata": json.dumps(meta, ensure_ascii=False), "open": "true" if open_study else "false"}
    if meta.get("document"):
        fields["name"] = meta["document"]
    body, ctype = multipart(fields, [("file", os.path.basename(step_path), data, "application/step")])
    return _call(settings, "POST", "/api/import", body, ctype)


def open_gui_rest(settings, path=None):
    body = json.dumps({"path": path} if path else {}).encode("utf-8")
    return _call(settings, "POST", "/api/open-gui", body, "application/json", timeout=30)


def launch_gui_local(settings, step_path):
    """GUI without the server: <FEA_STUDIO_HOME>/.venv/Scripts/pythonw.exe -m fea_gui <step> (a backup of FEA Studio.bat)."""
    home = settings.get("fea_studio_home")
    if not home or not os.path.isdir(home):
        raise RuntimeError("The FEA Studio server is not running and no FEA Studio installation is set "
                           "(the FEA_STUDIO_HOME variable or 'fea_studio_home' in settings.json).\n"
                           "The STEP and metadata are saved in:\n" + os.path.dirname(step_path))
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    for exe in (os.path.join(home, ".venv", "Scripts", "pythonw.exe"), os.path.join(home, ".venv", "bin", "python")):
        if os.path.isfile(exe):
            subprocess.Popen([exe, "-m", "fea_gui", step_path], cwd=home, creationflags=flags, close_fds=True)
            return exe
    bat = os.path.join(home, "FEA Studio.bat")
    if os.path.isfile(bat):
        subprocess.Popen([bat, step_path], cwd=home, creationflags=flags, close_fds=True)
        return bat
    raise RuntimeError("There is no .venv or 'FEA Studio.bat' in {}".format(home))


def deliver(settings, step_path, meta, open_gui=None):
    """Passes the STEP + metadata to FEA Studio (REST, otherwise a folder + the GUI). Returns the text for the user."""
    open_gui = settings.get("open_gui", True) if open_gui is None else open_gui
    meta_path = os.path.splitext(step_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    n0 = _NOTE_SEQ[0]
    st = server_status(settings)
    if st is None and _NOTE_SEQ[0] > n0:
        warn = "Warning: " + NOTES[-1] + "\n"
        if not open_gui:
            return warn + "The STEP and metadata were saved:\n" + step_path
        launch_gui_local(settings, step_path)
        return warn + "The STEP was saved and is opening in the FEA Studio GUI:\n" + step_path
    if st is not None:
        r = send_import(settings, step_path, meta, settings.get("open_study", True))
        lines = ["Sent to FEA Studio ({}).".format(settings["server"]),
                 "STEP: " + str(r.get("step_path"))]
        if r.get("parts") is not None:
            with_mat = sum(1 for p in r["parts"] if p.get("material"))
            lines.append("Study: {} parts, material assigned to {}.".format(len(r["parts"]), with_mat))
        lines += ["Warning: " + w for w in r.get("warnings", [])]
        if open_gui:
            open_gui_rest(settings)
            lines.append("The FEA Studio GUI is starting.")
        return "\n".join(lines)
    if not open_gui:
        return "The FEA Studio server is not running – the STEP and metadata were saved:\n" + step_path
    launch_gui_local(settings, step_path)
    return "The FEA Studio server is not running – the STEP was saved and is opening in the FEA Studio GUI:\n" + step_path


# ===================================================================== connection with the running GUI (IPC, standard library)
# a copy of the client part of fea_core/ipc.py: the pipe per user, the session key of the GUI in the user data, HMAC

PIPE_PREFIX = "\\\\.\\pipe\\"      # \\.\pipe\
IPC_ACK_TIMEOUT = 30.0             # confirmation of the server (as fea_core/ipc.py)
IPC_MAX_LINE = 64 << 20            # the longest line of the answer


def _pipe_name():
    import getpass
    import hashlib
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = "user"
    return "feastudio_live_ipc_" + hashlib.sha1(user.encode("utf-8", "replace")).hexdigest()[:8]


def _ipc_token_path():
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "FEAStudio", "mcp", "ipc", _pipe_name() + ".token")


def _ipc_token():
    try:
        with open(_ipc_token_path(), encoding="ascii") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _ipc_mac(token, *parts):
    """= fea_core.ipc.mac"""
    import hashlib
    import hmac
    return hmac.new(token.encode("ascii"), b"\n".join(parts), hashlib.sha256).hexdigest()


def _ipc_result_body(res):
    """= fea_core.ipc.result_body"""
    return json.dumps({k: v for k, v in res.items() if k != "rmac"}, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _ipc_request(token, tool, args, nonce):
    """Request line (bytes) as fea_core.ipc.send_command writes it."""
    body = json.dumps({"tool": tool, "args": args or {}}, ensure_ascii=False)
    req = {"nonce": nonce, "body": body, "mac": _ipc_mac(token, b"req", nonce.encode(), body.encode("utf-8"))}
    return json.dumps(req, ensure_ascii=False).encode("utf-8") + b"\n"


def _ipc_open():
    return open(PIPE_PREFIX + _pipe_name(), "r+b", buffering=0)


def gui_running():
    return os.name == "nt" and _ipc_token() is not None and os.path.exists(PIPE_PREFIX + _pipe_name())


def ipc_call(tool, args=None, timeout=60.0):
    """Command to the running GUI of FEA Studio → result (a dictionary); an error = RuntimeError."""
    import hmac
    import io
    import secrets
    import threading
    token = _ipc_token()
    if token is None:
        raise RuntimeError("The FEA Studio GUI is not running")
    nonce = secrets.token_hex(16)
    data = _ipc_request(token, tool, args, nonce)
    box, acked = {}, threading.Event()

    def exchange():                     # in a thread: a hung GUI / a foreign process on the pipe must not freeze Fusion
        f = None
        try:
            f = _ipc_open()
            view = memoryview(data)
            while view:
                view = view[f.write(view):]
            rd = io.BufferedReader(f, 1 << 16)
            ack = json.loads(rd.readline(IPC_MAX_LINE).decode("utf-8") or "{}")
            if not hmac.compare_digest(str(ack.get("proof", "")), _ipc_mac(token, b"ack", nonce.encode())):
                raise RuntimeError("The process on the pipe did not prove the FEA Studio GUI key: " + str(ack.get("error", "")))
            acked.set()
            res = json.loads(rd.readline(IPC_MAX_LINE).decode("utf-8") or "{}")
            if not isinstance(res, dict) or not hmac.compare_digest(
                    str(res.get("rmac", "")), _ipc_mac(token, b"res", nonce.encode(), _ipc_result_body(res))):
                raise RuntimeError("The result from the pipe is not signed with the FEA Studio GUI key")
            res.pop("rmac", None)
            box["res"] = res
        except Exception as e:  # noqa: BLE001
            box["err"] = e
        finally:
            if f is not None:
                f.close()

    t = threading.Thread(target=exchange, daemon=True)
    t.start()
    first = min(IPC_ACK_TIMEOUT, timeout)
    t.join(first)
    if t.is_alive() and not acked.is_set():
        raise RuntimeError("The FEA Studio GUI is not responding (no confirmation within %d s)" % int(first))
    t.join(max(0.0, timeout - first))
    if t.is_alive():
        raise RuntimeError("The FEA Studio GUI did not finish the command within %d s" % int(timeout))
    if "err" in box:
        raise RuntimeError(str(box["err"]))
    res = box.get("res") or {}
    if res.get("status") != "ok":
        raise RuntimeError("FEA Studio: " + str(res.get("error")))
    return res.get("result")


# ===================================================================== table of faces (identity of the faces for the iterations)

_SURF = {0: "plane", 1: "cylinder", 2: "cone", 3: "sphere", 4: "torus"}


def _pt(p, k=10.0):
    return [round(p.x * k, 6), round(p.y * k, 6), round(p.z * k, 6)]


def face_geometry(f):
    """Geometry of the face in mm (the global position – the face in the context of the assembly)."""
    pt = f.pointOnFace
    d = {"type": _SURF.get(int(f.geometry.surfaceType), "other"), "area": round(f.area * 100.0, 6),
         "centroid": _pt(f.centroid), "point": _pt(pt)}
    try:
        ok, n = f.evaluator.getNormalAtPoint(pt)
        if ok:
            d["normal"] = [round(n.x, 9), round(n.y, 9), round(n.z, 9)]
    except Exception:  # noqa: BLE001
        pass
    return d


def body_geometry(b):
    bb = b.boundingBox
    lo, hi = _pt(bb.minPoint), _pt(bb.maxPoint)
    return {"volume": round(b.volume * 1e3, 6), "center": [(a + c) / 2 for a, c in zip(lo, hi)], "bbox": lo + hi}


def face_table(bodies):
    """bodies = [(index of the row of parts, body in the context of the assembly)] → rows of the table of faces."""
    rows = []
    for ri, b in bodies:
        for f in b.faces:
            try:
                rows.append(dict(face_geometry(f), token=f.entityToken, part=ri))
            except Exception:  # noqa: BLE001 – one faulty face must not stop the export
                pass
    return rows


def resolve_tokens(design, tokens, body_tokens):
    """Tokens of the last export → geometry of the faces / bodies of the CURRENT version (Fusion translates them itself; the token strings
    are not compared). A face split in the CAD may return more pieces, a vanished one none.
    """
    faces, bodies = [], []
    for t in tokens:
        try:
            ents = list(design.findEntityByToken(t) or [])
        except Exception:  # noqa: BLE001
            ents = []
        faces.append({"token": t, "faces": [face_geometry(e) for e in ents if _kind(e) == "face"]})
    for t in body_tokens:
        try:
            ents = list(design.findEntityByToken(t) or [])
        except Exception:  # noqa: BLE001
            ents = []
        bodies.append({"token": t, "bodies": [body_geometry(e) for e in ents if _kind(e) == "body"]})
    return faces, bodies


def user_parameters(design):
    out = {}
    try:
        for prm in design.userParameters:
            out[prm.name] = {"expression": prm.expression, "value": prm.value, "unit": prm.unit,
                             "comment": prm.comment}
    except Exception:  # noqa: BLE001
        pass
    return out


def document_link(design):
    doc = design.parentDocument
    try:
        df = doc.dataFile if doc is not None else None
        return {"document_id": df.id if df is not None else "", "version": df.versionNumber if df is not None else 0}
    except Exception:  # noqa: BLE001 – an unsaved document
        return {"document_id": "", "version": 0}


def _cache_path(settings, doc_id):
    return os.path.join(settings["export_dir"], "posledni_{}.json".format(safe_name(doc_id or "neulozeny")))


# ===================================================================== metadata z Fusionu

# conversion of the units of the material properties of Fusion → MPa, t/mm³
_PRESSURE = {"Pascal": 1e-6, "Kilopascal": 1e-3, "Megapascal": 1.0, "Gigapascal": 1e3,
             "NewtonPerSquareMillimeter": 1.0, "PoundForcePerSquareInch": 0.00689476,
             "KilopoundForcePerSquareInch": 6.89476}
_DENSITY = {"KilogramPerCubicMeter": 1e-12, "GramPerCubicCentimeter": 1e-9, "KilogramPerCubicCentimeter": 1e-6,
            "PoundPerCubicInch": 2.76799e-8}
_PROPS = {"structural_Young_modulus": ("E_MPa", _PRESSURE), "structural_Poisson_ratio": ("nu", None),
          "structural_Density": ("rho_t_mm3", _DENSITY), "structural_Minimum_yield_stress": ("Re_MPa", _PRESSURE),
          "structural_Minimum_tensile_strength": ("Rm_MPa", _PRESSURE)}


def material_props(material):
    """Mechanical properties of the material of Fusion in the units of FEA Studio (mm, N, MPa, t)."""
    out = {}
    if material is None:
        return out
    mp = material.materialProperties
    for pid, (key, conv) in _PROPS.items():
        p = mp.itemById(pid)
        if p is None:
            continue
        v = getattr(p, "value", None)
        if not isinstance(v, (int, float)):
            continue
        if conv is not None:
            unit = getattr(p, "units", None)
            if unit not in conv:
                out[key + "_raw"] = "{} {}".format(v, unit)
                continue
            v *= conv[unit]
        out[key] = float("{:.6g}".format(v))
    return out


_PROPS_CACHE = {}


def _cached_props(material):
    if material is None:
        return {}
    key = (material.id, material.name)
    if key not in _PROPS_CACHE:
        _PROPS_CACHE[key] = material_props(material)
    return dict(_PROPS_CACHE[key])


def appearance_color(appearance):
    """Colour of the appearance as sRGB 0–1 (a STEP from Fusion carries the colour linearly – other numbers, the same colour)."""
    if appearance is None:
        return None
    props = appearance.appearanceProperties
    for pid in ("opaque_albedo", "generic_diffuse", "metal_f0", "layered_diffuse", "transparent_color"):
        p = props.itemById(pid)
        if p is not None and getattr(p, "value", None) is not None:
            c = p.value
            return [round(c.red / 255.0, 4), round(c.green / 255.0, 4), round(c.blue / 255.0, 4)]
    for i in range(props.count):
        p = props.item(i)
        if p.objectType == adsk.core.ColorProperty.classType() and p.value is not None:
            c = p.value
            return [round(c.red / 255.0, 4), round(c.green / 255.0, 4), round(c.blue / 255.0, 4)]
    return None


def attributes_of(*entities):
    d = {}
    for e in entities:
        if e is None:
            continue
        try:
            for a in e.attributes:
                d["{}.{}".format(a.groupName, a.name)] = a.value
        except Exception:  # noqa: BLE001 – some entities do not have attributes
            pass
    return d


def _body_row(body, occ, step_names):
    comp = occ.component if occ is not None else body.parentComponent
    mat = body.material
    app = body.appearance
    row = {"name": step_names[0], "names": step_names[1:], "body": body.name,
           "component": comp.name, "occurrence": occ.name if occ is not None else None,
           "path": occ.fullPathName if occ is not None else None,
           "material": mat.name if mat else None, "material_props": _cached_props(mat),
           "appearance": app.name if app else None, "color": appearance_color(app),
           "attributes": attributes_of(body, occ, comp)}
    # Component.partNumber / description are NOT read: ~0.5 s per occurrence (verified, an assembly with 115 occurrences → minutes)
    try:                       # BRepBody.volume [cm³] – physicalProperties is slow for large assemblies
        row["volume_mm3"] = round(body.volume * 1e3, 4)
    except Exception:  # noqa: BLE001
        pass
    return {k: v for k, v in row.items() if v not in (None, "", {}, [])}


# Fusion JointTypes → typ v metadatech (fea_core.motion.joints_cad.FUSION_TYPES)
_JOINT_TYPES = {0: "rigid", 1: "revolute", 2: "slider", 3: "cylindrical", 4: "pin_slot", 5: "planar", 6: "ball"}


def _vec(v, scale=1.0):
    return [v.x * scale, v.y * scale, v.z * scale] if v is not None else None


def _limits(lim, scale):
    """Limits of the motion of a joint (JointLimits) in mm / ° (None = no limits)."""
    if lim is None or not (lim.isMinimumValueEnabled or lim.isMaximumValueEnabled):
        return None
    return {"min": lim.minimumValue * scale if lim.isMinimumValueEnabled else None,
            "max": lim.maximumValue * scale if lim.isMaximumValueEnabled else None}


def collect_joints(design):
    """Joints of the assembly (Joint, AsBuiltJoint) for the proposal of the joints of the mechanism in FEA Studio (step Motion) – the format
    see fea_core.motion.joints_cad. Fusion works internally in cm and rad → conversion to mm and °.
    """
    out = []
    root = design.rootComponent
    joints = []
    for attr in ("allJoints", "allAsBuiltJoints"):
        try:
            joints += list(getattr(root, attr))
        except Exception:  # noqa: BLE001 – older API
            pass
    for j in joints:
        try:
            mot = j.jointMotion
            typ = _JOINT_TYPES.get(int(mot.jointType), "rigid")
            geo = getattr(j, "geometryOrOriginOne", None) or getattr(j, "geometry", None)
            origin = _vec(geo.origin, 10.0) if geo is not None and geo.origin is not None else None
            axis = axis2 = None
            lim = None
            if typ in ("revolute", "cylindrical", "pin_slot"):
                axis = _vec(getattr(mot, "rotationAxisVector", None))
                lim = _limits(getattr(mot, "rotationLimits", None), 180.0 / math.pi)
            if typ == "slider":
                axis = _vec(getattr(mot, "slideDirectionVector", None))
                lim = _limits(getattr(mot, "slideLimits", None), 10.0)
            if typ == "pin_slot":
                axis2 = _vec(getattr(mot, "slideDirectionVector", None))
            if typ == "planar":
                axis = _vec(getattr(mot, "normalDirectionVector", None))
            o1, o2 = getattr(j, "occurrenceOne", None), getattr(j, "occurrenceTwo", None)
            out.append({"name": j.name, "type": typ,
                        "occurrence_one": o1.name if o1 is not None else None,
                        "occurrence_two": o2.name if o2 is not None else None,
                        "origin": origin, "axis": axis, "axis2": axis2, "limits": lim,
                        "suppressed": bool(getattr(j, "isSuppressed", False)),
                        "units": {"length": "mm", "angle": "deg"}})
        except Exception as e:  # noqa: BLE001 – one faulty joint must not stop the export
            out.append({"name": getattr(j, "name", "?"), "error": str(e)})
    return out


def collect_metadata(design, faces=True):
    """Metadata of the visible bodies (the STEP from Fusion does not contain the hidden bodies) + the table of faces with tokens (faces).

    The name of a part in FEA Studio (fea_core.geometry.read_step) = the name of the occurrence ("Deska:1"); for a component with several
    bodies "Deska:1_1", "Deska:1_2" (the order of the bodies); the bodies of the root component = the name of the body.
    """
    root = design.rootComponent
    parts, hidden, ctx_bodies = [], [], []
    _PROPS_CACHE.clear()

    def add_bodies(bodies, occ):
        vis = []
        for b in bodies:
            if not b.isSolid:
                continue
            if b.isLightBulbOn and (occ is None or occ.isVisible):
                vis.append(b)
            else:
                hidden.append((occ.name + "/" if occ is not None else "") + b.name)
        base = occ.name if occ is not None else None
        for k, b in enumerate(vis, 1):
            if base is None:
                names = [b.name]
            elif len(vis) == 1:
                names = [base, "{}_{}".format(base, k)]
            else:
                names = ["{}_{}".format(base, k), base]
            names.append(b.name)
            row = _body_row(b, occ, names)
            ctx = b.createForAssemblyContext(occ) if occ is not None else b
            try:
                row["token"] = ctx.entityToken
            except Exception:  # noqa: BLE001
                pass
            parts.append(row)
            ctx_bodies.append((len(parts) - 1, ctx))

    add_bodies(list(root.bRepBodies), None)
    for occ in root.allOccurrences:
        add_bodies(list(occ.component.bRepBodies), occ)
    doc = design.parentDocument
    return {"source": SOURCE, "document": doc.name if doc else root.name, "root_component": root.name,
            "units": "mm", "design_units": design.unitsManager.defaultLengthUnits,
            "exported": time.strftime("%Y-%m-%dT%H:%M:%S"), "parts": parts, "hidden_bodies": hidden,
            "joints": collect_joints(design), "link": document_link(design), "params": user_parameters(design),
            "faces": face_table(ctx_bodies) if faces else [],
            "notes": ["The STEP carries occurrence/body names and colors (linear); materials and attributes only in this JSON.",
                      "Barvy v JSON jsou sRGB 0–1.", "faces = plochy v mm s tokeny Fusionu (iterace geometrie)."]}


# ===================================================================== scope: the whole design / selection

def _tok(e):
    return getattr(e, "entityToken", None) or id(e)


def _kind(e):
    """occ | body | face | edge | comp | '' according to the objectType of Fusion (adsk::fusion::Occurrence …)."""
    t = str(getattr(e, "objectType", ""))
    for k, name in (("occ", "Occurrence"), ("body", "BRepBody"), ("face", "BRepFace"), ("edge", "BRepEdge"),
                    ("comp", "Component")):
        if t == name or t.endswith("::" + name):
            return k
    return ""


def plan_isolation(root, selected):
    """What to hide so that the STEP contains only the selected parts (a body) and subassemblies (an occurrence with the whole subtree).

    Returns (occurrences to hide, bodies to hide, names of the selected) or None = the whole design (the root component is selected).
    It hides as high as possible in the tree (a hidden occurrence hides the whole subtree); bodies are hidden only in the components that
    remain visible because of the selected body (a body of another occurrence of the same component would otherwise be hidden together
    with the selected one). The position of the parts in the assembly does not change – the STEP has the same coordinates as the whole design.
    """
    sel_occ, sel_body = {}, {}
    for e in selected:
        k = _kind(e)
        if k in ("face", "edge"):
            e, k = e.body, "body"
        if k == "comp":
            if _tok(e) == _tok(root):
                return None
            raise RuntimeError("Select an occurrence (subassembly) or a body, not the component definition “{}”.".format(e.name))
        if k == "occ":
            sel_occ[_tok(e)] = e
        elif k == "body":
            sel_body[_tok(e)] = e
    if not sel_occ and not sel_body:
        raise RuntimeError("The selection contains no part (body) or subassembly (occurrence).")
    full, context = set(), set()

    def mark_full(o):
        full.add(_tok(o))
        context.add(_tok(o))
        for c in o.childOccurrences:
            mark_full(c)

    def mark_ancestors(o):
        while o is not None:
            context.add(_tok(o))
            o = o.assemblyContext

    for o in sel_occ.values():
        mark_full(o)
        mark_ancestors(o.assemblyContext)
    for b in sel_body.values():
        mark_ancestors(b.assemblyContext)
    hide_occ, hide_body = [], []
    hide_body += [b for b in root.bRepBodies if _tok(b) not in sel_body]

    def visit(o):
        if _tok(o) not in full:
            hide_body.extend(b for b in o.bRepBodies if _tok(b) not in sel_body)
        for c in o.childOccurrences:
            (visit if _tok(c) in context else hide_occ.append)(c)

    for o in root.occurrences:
        (visit if _tok(o) in context else hide_occ.append)(o)
    names = [o.name for o in sel_occ.values()] + [b.name for b in sel_body.values()]
    return hide_occ, hide_body, names


@contextlib.contextmanager
def isolate(design, selected):
    """For the time of the export hides everything outside the selection (isLightBulbOn), at the end returns the original visibility (also on an error).
    No selection / the root component = the whole design, nothing changes. Returns the description of the scope or None.
    """
    plan = plan_isolation(design.rootComponent, selected) if selected else None
    if plan is None:
        yield None
        return
    hide_occ, hide_body, names = plan
    changed = []
    try:
        for o in list(hide_occ) + list(hide_body):          # occurrences first: hiding a subtree is cheaper
            if o.isLightBulbOn:
                o.isLightBulbOn = False
                changed.append(o)
        yield {"mode": "selection", "items": names}
    finally:
        for o in reversed(changed):
            try:
                o.isLightBulbOn = True
            except Exception:  # noqa: BLE001 – the restoration must not hide the original error
                pass


def scope_title(doc_name, scope):
    """Name of the study: document + selection ("Držák – Deska:1 + 2 more")."""
    items = scope["items"]
    more = " + {} more".format(len(items) - 1) if len(items) > 1 else ""
    return "{} – {}{}".format(doc_name, items[0], more)


def export_step(design, path):
    em = design.exportManager
    opts = em.createSTEPExportOptions(path, design.rootComponent)
    if not em.execute(opts):
        raise RuntimeError("Export STEP selhal: " + path)
    return path


def send_active_design(settings=None, open_gui=None, selection=None):
    """The whole procedure for the active design → the text of the result (calls the command; also possible from a script).
    selection = selected occurrences / bodies (None or empty = the whole design).
    """
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    if design is None:
        raise RuntimeError("The active document is not a Fusion design (switch to the Design workspace).")
    settings = settings or load_settings()
    os.makedirs(settings["export_dir"], exist_ok=True)
    doc = design.parentDocument.name if design.parentDocument else design.rootComponent.name
    with isolate(design, selection) as scope:
        meta = collect_metadata(design)                    # after hiding: the list of parts = the content of the STEP
        if not meta["parts"]:
            raise RuntimeError("The selected scope contains no visible solid body.")
        label = safe_name(doc if scope is None else "{}_{}".format(doc, scope["items"][0]))
        step = os.path.join(settings["export_dir"], "{}_{}.step".format(label, time.strftime("%Y%m%d_%H%M%S")))
        export_step(design, step)
    if scope is not None:
        meta["scope"] = scope
        meta["document"] = scope_title(doc, scope)
        meta["notes"].append("Only the selection was sent: " + ", ".join(scope["items"]))
    _save_cache(settings, meta)
    if gui_running():                                # a running GUI: a new study in it at once
        meta_path = os.path.splitext(step)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        ipc_call("fea_cad_open", {"step_path": step})
        text = "Sent to the running FEA Studio (new study):\n" + step
    else:
        text = deliver(settings, step, meta, open_gui)
    if scope is not None:
        text = "Scope: selection ({} parts in the STEP) – {}\n".format(len(meta["parts"]), ", ".join(scope["items"])) + text
    return text


def _save_cache(settings, meta):
    """Tokens of the last export of the document (an update without a running GUI translates all of them)."""
    try:
        doc_id = (meta.get("link") or {}).get("document_id", "")
        with open(_cache_path(settings, doc_id), "w", encoding="utf-8") as f:
            json.dump({"tokens": [r["token"] for r in meta.get("faces", []) if r.get("token")],
                       "body_tokens": [p["token"] for p in meta.get("parts", []) if p.get("token")]}, f)
    except Exception:  # noqa: BLE001 – the cache is only a help
        pass


def update_active_design(settings=None, solve=True):
    """"Update in FEA Studio": recalculation of the model, STEP + table of faces with the tokens of the previous version translated →
    an opened study of FEA Studio (the settings stay, the selections are remapped). If the GUI does not run, it starts with the project
    linked to the document (the local registry of FEA Studio).
    """
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    if design is None:
        raise RuntimeError("The active document is not a Fusion design (switch to the Design workspace).")
    settings = settings or load_settings()
    os.makedirs(settings["export_dir"], exist_ok=True)
    try:
        design.computeAll()
    except Exception:  # noqa: BLE001 – direct modelling
        pass
    info = ipc_call("fea_cad_link_info", timeout=30) if gui_running() else None
    if info is not None and info.get("busy"):
        raise RuntimeError("FEA Studio is meshing or calculating right now – try again when it finishes.")
    meta = collect_metadata(design)
    if not meta["parts"]:
        raise RuntimeError("The design contains no visible solid body.")
    link = meta["link"]
    if info is not None and info.get("study") and info.get("tokens"):
        tokens, btokens = info["tokens"], info.get("body_tokens", [])
    else:                                            # tokens of the last export from the local cache
        try:
            with open(_cache_path(settings, link.get("document_id")), encoding="utf-8") as f:
                c = json.load(f)
            tokens, btokens = c.get("tokens", [])[:20000], c.get("body_tokens", [])
        except (OSError, ValueError):
            tokens, btokens = [], []
    meta["resolved"], meta["resolved_bodies"] = resolve_tokens(design, tokens, btokens)
    doc = design.parentDocument.name if design.parentDocument else design.rootComponent.name
    step = os.path.join(settings["export_dir"], "{}_{}.step".format(safe_name(doc), time.strftime("%Y%m%d_%H%M%S")))
    export_step(design, step)
    with open(os.path.splitext(step)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    _save_cache(settings, meta)
    lost = sum(1 for r in meta["resolved"] if not r["faces"])
    split = sum(1 for r in meta["resolved"] if len(r["faces"]) > 1)
    head = "Study faces translated: {} (lost {}, split {}).".format(len(tokens), lost, split)
    if info is not None:
        r = ipc_call("fea_cad_update", {"step_path": step, "solve": bool(solve)})
        return head + "\nFEA Studio: the new geometry was passed on ({}) – check the remapping in FEA Studio.".format(
            r.get("mode"))
    home = settings.get("fea_studio_home")
    exe = os.path.join(home or "", ".venv", "Scripts", "pythonw.exe")
    if not home or not os.path.isfile(exe):
        raise RuntimeError("FEA Studio is not running and no installation is set (fea_studio_home). STEP: " + step)
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    subprocess.Popen([exe, "-m", "fea_gui", "--cad-update", step], cwd=home, creationflags=flags, close_fds=True)
    return head + "\nFEA Studio is starting with the project linked to this document."


# ===================================================================== command in the UI

if adsk is not None:

    SCOPE_ALL = "Whole design"
    SCOPE_SEL = "Selected parts / subassemblies"

    def _preselected(ui):
        """The current selection in Fusion (a body, an occurrence; a face / edge → its body) – pre-fills the selection field."""
        out, seen = [], set()
        for i in range(ui.activeSelections.count):
            e = ui.activeSelections.item(i).entity
            k = _kind(e)
            if k in ("face", "edge"):
                e, k = e.body, "body"
            if k in ("occ", "body") and _tok(e) not in seen:
                seen.add(_tok(e))
                out.append(e)
        return out

    class _Execute(adsk.core.CommandEventHandler):
        def notify(self, args):
            ui = adsk.core.Application.get().userInterface
            try:
                inputs = args.command.commandInputs
                s = load_settings()
                s["server"] = inputs.itemById("server").value.strip() or s["server"]
                open_gui = inputs.itemById("open_gui").value
                selection = None
                if inputs.itemById("scope").selectedItem.name == SCOPE_SEL:
                    sel = inputs.itemById("selection")
                    selection = [sel.selection(i).entity for i in range(sel.selectionCount)]
                    if not selection:
                        raise RuntimeError("The scope “{}” is selected, but nothing is selected.".format(SCOPE_SEL))
                ui.messageBox(send_active_design(s, open_gui, selection), CMD_NAME)
            except Exception as e:  # noqa: BLE001
                ui.messageBox("Sending failed:\n{}\n\n{}".format(e, traceback.format_exc()[-1500:]), CMD_NAME)

    class _UpdExecute(adsk.core.CommandEventHandler):
        def notify(self, args):
            ui = adsk.core.Application.get().userInterface
            try:
                ui.messageBox(update_active_design(), UPD_NAME)
            except Exception as e:  # noqa: BLE001
                ui.messageBox("Update failed:\n{}\n\n{}".format(e, traceback.format_exc()[-1500:]), UPD_NAME)

    class _UpdCreated(adsk.core.CommandCreatedEventHandler):
        def notify(self, args):
            h = _UpdExecute()
            args.command.execute.add(h)
            _handlers.append(h)
            args.command.isExecutedWhenPreEmpted = False

    class _Changed(adsk.core.InputChangedEventHandler):
        def notify(self, args):
            if args.input.id == "scope":
                sel = args.inputs.itemById("selection")
                sel.isVisible = args.input.selectedItem.name == SCOPE_SEL
                if sel.isVisible:
                    sel.hasFocus = True

    class _Created(adsk.core.CommandCreatedEventHandler):
        def notify(self, args):
            ui = adsk.core.Application.get().userInterface
            s = load_settings()
            cmd = args.command
            cmd.okButtonText = "Send"
            inputs = cmd.commandInputs
            st = server_status(s)
            inputs.addTextBoxCommandInput(
                "info", "", "Server FEA Studia: " + ("running – the STEP is sent through the REST API" if st else
                                                    "not running – the STEP is saved and opened in the GUI"), 2, True)
            pre = _preselected(ui)
            dd = inputs.addDropDownCommandInput("scope", "Co odeslat", adsk.core.DropDownStyles.TextListDropDownStyle)
            dd.listItems.add(SCOPE_ALL, not pre)
            dd.listItems.add(SCOPE_SEL, bool(pre))
            dd.tooltip = ("The whole design, or only the selected bodies / occurrences (subassemblies with the whole subtree). The others are hidden "
                          "for the time of the export, the position of parts in the assembly stays; the visibility is then restored.")
            sel = inputs.addSelectionInput("selection", "Selection",
                                           "Select a part (body) or a subassembly (occurrence) in the model or in the browser")
            sel.addSelectionFilter("Occurrences")
            sel.addSelectionFilter("SolidBodies")
            sel.setSelectionLimits(0)
            for e in pre:
                sel.addSelection(e)
            sel.isVisible = bool(pre)
            inputs.addStringValueInput("server", "Adresa REST API", s["server"])
            inputs.addBoolValueInput("open_gui", "Open the FEA Studio GUI", True, "", bool(s.get("open_gui", True)))
            for h, ev in ((_Execute(), cmd.execute), (_Changed(), cmd.inputChanged)):
                ev.add(h)
                _handlers.append(h)


def _own_panels(ui):
    """(tab, panel) of the own panel in all tabs of the Design workspace with the INSERT panel."""
    ws = ui.workspaces.itemById(WORKSPACE_ID)
    if ws is None:
        return []
    out = []
    for tab in ws.toolbarTabs:
        ids = [p.id for p in tab.toolbarPanels]
        if not any(i in ids for i in INSERT_PANELS):
            continue
        pid = "{}_{}".format(OWN_PANEL, tab.id)
        panel = tab.toolbarPanels.itemById(pid)
        if panel is None:
            sel = next((i for i in SELECT_PANELS if i in ids), None)
            if sel is not None:                      # right before SELECT (= after INSERT)
                panel = tab.toolbarPanels.add(pid, OWN_PANEL_NAME, sel, True)
            else:
                ins = next(i for i in INSERT_PANELS if i in ids)
                panel = tab.toolbarPanels.add(pid, OWN_PANEL_NAME, ins, False)
        out.append((tab, panel))
    return out


def _remove_ui(ui):
    """Deletes the button, the own panels and the definition of the command (also after a previous run of the add-in that did not end with stop())."""
    panel = ui.allToolbarPanels.itemById(PANEL_ID)
    if panel is not None:
        for cid in (CMD_ID, UPD_ID):
            c = panel.controls.itemById(cid)
            if c:
                c.deleteMe()
    ws = ui.workspaces.itemById(WORKSPACE_ID)
    for tab in (ws.toolbarTabs if ws is not None else []):
        own = tab.toolbarPanels.itemById("{}_{}".format(OWN_PANEL, tab.id))
        if own is not None:
            own.deleteMe()
    for cid in (CMD_ID, UPD_ID):
        cd = ui.commandDefinitions.itemById(cid)
        if cd:
            cd.deleteMe()


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        # a repeated run (another copy of the add-in, restart from the Add-Ins dialog): the old definition would have two handlers
        # and the dialog would be assembled twice → always again from the start
        _remove_ui(ui)
        cd = ui.commandDefinitions.addButtonDefinition(CMD_ID, CMD_NAME, CMD_TIP, ICONS)
        h = _Created()
        cd.commandCreated.add(h)
        _handlers.append(h)
        panel = ui.allToolbarPanels.itemById(PANEL_ID)
        if panel is not None and panel.controls.itemById(CMD_ID) is None:
            panel.controls.addCommand(cd)
        ud = ui.commandDefinitions.addButtonDefinition(UPD_ID, UPD_NAME, UPD_TIP, ICONS)
        hu = _UpdCreated()
        ud.commandCreated.add(hu)
        _handlers.append(hu)
        if panel is not None and panel.controls.itemById(UPD_ID) is None:
            panel.controls.addCommand(ud)
        for _tab, own in _own_panels(ui):
            for d in (cd, ud):
                ctrl = own.controls.itemById(d.id) or own.controls.addCommand(d)
                ctrl.isPromoted = True                  # a large icon directly on the bar
                ctrl.isPromotedByDefault = True
    except Exception:  # noqa: BLE001
        if ui:
            ui.messageBox("FEA Studio – the add-in failed to start:\n" + traceback.format_exc())


def stop(context):
    try:
        _remove_ui(adsk.core.Application.get().userInterface)
    except Exception:  # noqa: BLE001
        pass
    _handlers.clear()


if __name__ == "__main__" and len(sys.argv) > 1:      # debugging the client outside Fusion: python FEAStudio.py status
    print(json.dumps(server_status(load_settings()), ensure_ascii=False, indent=1))
