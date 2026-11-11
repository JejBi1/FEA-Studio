"""Ultralight IPC client and protocol for the Dynamic IPC Bridge (FEA Studio).

Pure Python (standard library) – NO import of PySide6 or other heavy libraries!
Allows the MCP server and other clients to find out whether the FEA Studio GUI is running,
and to redirect commands transparently into it for live manipulation of the model, mesh and results.

On Windows: named pipe r"\\\\\\\\.\\\\pipe\\\\feastudio_live_ipc_<user>"
On Linux / macOS: UNIX domain socket in $XDG_RUNTIME_DIR, otherwise in a private folder (0700) in the temporary directory.

Security (17. 10. 2026): at start the GUI creates a secret session key (file `<name>.token` in the user data
folder, readable only by the owner). A request carries an HMAC of the key over a one-time nonce and the request body (a foreign process of the same
user without access to the key, or a fake server with the pipe name taken, cannot do anything and does not learn the key);
the server answers with a confirmation with the HMAC of the nonce (so the client verifies that it talks to the real GUI), and then with the result.
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import io
import json
import logging
import os
import queue
import secrets
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("fea_core.ipc")

PIPE_NAME = "feastudio_live_ipc"
ACTIVE_SESSION_FILENAME = "active_session.json"
MAX_LINE = 256 << 20            # largest protocol line (a reply with images in base64)
ACK_TIMEOUT = 30.0              # the GUI must confirm the request (prove the key) within this time


def _user_tag() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 – without the user's environment variables
        user = str(os.getuid()) if hasattr(os, "getuid") else "user"
    return hashlib.sha1(user.encode("utf-8", "replace")).hexdigest()[:8]


def default_pipe_name() -> str:
    """Name of the pipe / socket for the current user (another user of the same PC has another one)."""
    return f"{PIPE_NAME}_{_user_tag()}"


def _private_runtime_dir() -> Path:
    """Private folder for the unix socket: $XDG_RUNTIME_DIR, otherwise <tmp>/feastudio-<uid> with rights 0700."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and os.path.isdir(xdg):
        return Path(xdg)
    d = Path(tempfile.gettempdir()) / f"feastudio-{os.getuid() if hasattr(os, 'getuid') else _user_tag()}"
    d.mkdir(mode=0o700, exist_ok=True)
    st = d.stat()
    if hasattr(os, "getuid") and (st.st_uid != os.getuid() or st.st_mode & 0o077):
        raise PermissionError(f"The folder for IPC socket {d} does not belong to the user or is not private")
    return d


def get_pipe_path(name: str | None = None) -> str:
    """Returns the system path of the named pipe (Windows) or of the unix socket (Linux)."""
    name = name or default_pipe_name()
    if sys.platform == "win32":
        return rf"\\.\pipe\{name}"
    return str(_private_runtime_dir() / name)


def server_name(name: str | None = None) -> str:
    """What to pass to QLocalServer.listen: Windows the pipe name, unix the full path of the socket."""
    name = name or default_pipe_name()
    return name if sys.platform == "win32" else get_pipe_path(name)


def _name_of(pipe_path: str) -> str:
    return pipe_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def token_path(name: str | None = None) -> Path:
    """File with the secret session key of the GUI (readable only by the owner)."""
    from .config import user_data_dir
    return user_data_dir() / "mcp" / "ipc" / f"{name or default_pipe_name()}.token"


def write_token(name: str | None = None) -> str:
    """Creates a new session key and writes it (the GUI calls it after a successful listen)."""
    tok = secrets.token_hex(32)
    p = token_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(tok)
    os.replace(tmp, p)
    return tok


def read_token(name: str | None = None) -> str | None:
    try:
        tok = token_path(name).read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        return None
    return tok or None


def remove_token(name: str | None = None, token: str | None = None) -> None:
    """Deletes the key file (only if it contains our `token` – a second window does not delete the key of the first one)."""
    try:
        if token is None or read_token(name) == token:
            token_path(name).unlink()
    except OSError:
        pass


def result_body(resp: dict) -> bytes:
    """Canonical JSON of the result for signing (without the field rmac; json dumps→loads→dumps is stable in Python)."""
    return json.dumps({k: v for k, v in resp.items() if k != "rmac"}, ensure_ascii=False, sort_keys=True).encode("utf-8")


def mac(token: str, *parts: bytes) -> str:
    return hmac.new(token.encode("ascii"), b"\n".join(parts), hashlib.sha256).hexdigest()


def _key_sig(name: str):
    """Identity of the key file (time of change, size) – a new GUI session writes a new file."""
    try:
        st = token_path(name).stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


# (pipe, key file identity) → time until which the pipe is not tried: the process did not prove the key or rejected our
# signature (stale key, foreign process on the pipe – until the GUI writes a new key), or it did not confirm within
# ACK_TIMEOUT (UNANSWERED_RETRY s) – further calls go headless at once instead of waiting again
_UNVERIFIED: dict = {}
UNANSWERED_RETRY = 60.0


def _skipped(key) -> bool:
    until = _UNVERIFIED.get(key)
    if until is None:
        return False
    if time.monotonic() < until:
        return True
    _UNVERIFIED.pop(key, None)
    return False


def is_gui_running(pipe_path: str | None = None) -> bool:
    """Checks whether the GUI application is open and listening on the IPC pipe / socket (< 0.1 ms)."""
    try:
        p = pipe_path or get_pipe_path()
        if not (os.path.exists(p) and token_path(_name_of(p)).is_file()):
            return False
        return not _skipped((p, _key_sig(_name_of(p))))
    except OSError:
        return False


class IpcError(RuntimeError):
    """General error of the IPC communication."""


class GuiNotRunningError(IpcError):
    """The GUI window is not running or does not answer on IPC."""


class GuiUnverifiedError(GuiNotRunningError):
    """The process on the pipe did not prove the session key (stale key file, or not our GUI) – the caller works
    without the GUI; the pipe is not tried again until a new key file appears."""


def _open(path: str):
    """Opens the connection → (file for writing and reading line by line, closing function)."""
    if sys.platform == "win32":
        try:
            f = open(path, "r+b", buffering=0)
        except FileNotFoundError:
            raise GuiNotRunningError(f"GUI pipe not found: {path}") from None
        except OSError as e:
            raise GuiNotRunningError(f"Cannot connect to the GUI: {e}") from e

        def write(data: bytes) -> None:
            view = memoryview(data)
            while view:
                view = view[f.write(view):]
        # reading through a buffer (readline on raw FileIO reads byte by byte = a system call for every byte)
        return io.BufferedReader(f, 1 << 16), write, f.close
    if not os.path.exists(path):
        raise GuiNotRunningError(f"GUI socket not found: {path}")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(10.0)
        s.connect(path)
        s.settimeout(None)
    except (FileNotFoundError, ConnectionRefusedError, socket.timeout) as e:
        s.close()
        raise GuiNotRunningError(f"Cannot connect to the GUI: {e}") from e
    rf = s.makefile("rb")

    def close():
        for c in (rf.close, s.close):
            try:
                c()
            except OSError:
                pass
    return rf, s.sendall, close


def send_command(
    tool: str,
    args: dict[str, Any] | None = None,
    timeout: float | None = None,
    pipe_path: str | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Sends a command over IPC to the running GUI window and waits for the result (JSON lines).

    `timeout` = the longest time to wait for the result (None = no limit – a calculation in the GUI may take long);
    the confirmation from the GUI must arrive within ACK_TIMEOUT. `cancel` (a set event) ends the waiting.

    Returns a dictionary of the answer:
      - on success: {"status": "ok", "result": ...}
      - on error:   {"status": "error", "error": "error description"}
    """
    path = pipe_path or get_pipe_path()
    name = _name_of(path)
    sig = _key_sig(name)
    token = read_token(name)
    if token is None:
        raise GuiNotRunningError(f"GUI is not running (session key {token_path(name)} missing)")
    if _skipped((path, sig)):
        raise GuiUnverifiedError("The process on the IPC pipe did not prove the GUI session key earlier – working without the GUI")
    body = json.dumps({"tool": tool, "args": args or {}}, ensure_ascii=False)
    nonce = secrets.token_hex(16)
    req = {"nonce": nonce, "body": body, "mac": mac(token, b"req", nonce.encode(), body.encode("utf-8"))}
    payload = json.dumps(req, ensure_ascii=False).encode("utf-8") + b"\n"

    f, write, close = _open(path)
    lines: queue.Queue = queue.Queue()
    rt: threading.Thread | None = None

    def reader():                       # confirmation + result, then the end (do not block in another read)
        try:
            for _ in range(2):
                line = f.readline(MAX_LINE)
                lines.put(line)
                if not line or not line.endswith(b"\n"):
                    return
        except Exception as e:  # noqa: BLE001 – pass to the waiting thread
            lines.put(e)

    try:
        write(payload)
        rt = threading.Thread(target=reader, daemon=True, name="fea-ipc-reader")
        rt.start()

        def next_line(limit: float | None) -> bytes:
            end = None if limit is None else time.monotonic() + limit
            while True:
                if cancel is not None and cancel.is_set():
                    raise IpcError("Call cancelled")
                wait = 0.25 if end is None else min(0.25, end - time.monotonic())
                if wait <= 0:
                    raise IpcError(f"GUI response timed out ({limit} s)")
                try:
                    item = lines.get(timeout=wait)
                except queue.Empty:
                    continue
                if isinstance(item, Exception):
                    raise IpcError(f"GUI connection error: {item}") from item
                if not item:
                    raise IpcError("GUI closed the connection without a response")
                if not item.endswith(b"\n"):
                    raise IpcError("GUI response is too long or incomplete")
                return item

        ack_limit = ACK_TIMEOUT if timeout is None else min(ACK_TIMEOUT, timeout)
        try:
            ack_line = next_line(ack_limit)
        except IpcError as e:
            if ack_limit == ACK_TIMEOUT and "timed out" in str(e):
                _UNVERIFIED[(path, sig)] = time.monotonic() + UNANSWERED_RETRY
            raise
        ack = _parse(ack_line)
        if ack.get("status") == "error" and "proof" not in ack:      # our key is not the key of that GUI (stale file)
            _UNVERIFIED[(path, sig)] = float("inf")
            raise GuiUnverifiedError(f"GUI rejected the request: {ack.get('error')}")
        if not hmac.compare_digest(str(ack.get("proof", "")), mac(token, b"ack", nonce.encode())):
            _UNVERIFIED[(path, sig)] = float("inf")
            raise GuiUnverifiedError("The process on the IPC pipe did not prove the GUI session key – the response is discarded")
        res = _parse(next_line(timeout))
        if not hmac.compare_digest(str(res.get("rmac", "")), mac(token, b"res", nonce.encode(), result_body(res))):
            raise IpcError("The result from the IPC pipe is not signed with the GUI session key – discarded")
        res.pop("rmac", None)
        return res
    finally:
        if rt is not None and rt.is_alive():
            # the reading thread hangs in a read (cancel, time limit): closing the pipe on Windows may block
            threading.Thread(target=close, daemon=True).start()
        else:
            close()


def _parse(line: bytes) -> dict[str, Any]:
    try:
        data = json.loads(line.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise IpcError(f"Invalid JSON response from the GUI: {e}") from e
    if not isinstance(data, dict):
        raise IpcError("GUI response is not a JSON object")
    return data


def active_session_path() -> Path:
    """Path to the file with information about the active background session."""
    from .config import user_data_dir
    return user_data_dir() / "mcp" / ACTIVE_SESSION_FILENAME


def save_active_session_info(
    study_name: str,
    step_path: str | Path | None = None,
    study_path: str | Path | None = None,
    workdir: str | Path | None = None,
) -> None:
    """Saves metadata of a study in progress in headless mode for later takeover in the GUI."""
    try:
        p = active_session_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "study_name": study_name,
            "step_path": str(step_path) if step_path else None,
            "study_path": str(study_path) if study_path else None,
            "workdir": str(workdir) if workdir else None,
            "timestamp": time.time(),
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as e:  # noqa: BLE001
        log.warning("Cannot write active session metadata: %s", e)


def load_active_session_info(max_age_seconds: float = 7200.0) -> dict[str, Any] | None:
    """Loads metadata of a recent running assistant session (if it is not older than max_age_seconds)."""
    p = active_session_path()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        age = time.time() - float(data.get("timestamp", 0))
        if age <= max_age_seconds:
            return data
    except Exception as e:  # noqa: BLE001
        log.warning("Cannot read active session metadata: %s", e)
    return None


def clear_active_session_info() -> None:
    """Deletes the record of the active session."""
    p = active_session_path()
    try:
        if p.is_file():
            p.unlink()
    except OSError:
        pass
