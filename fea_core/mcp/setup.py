"""Registration of the MCP server of FEA Studio in the LLM clients on this computer: `python -m fea_core.mcp.setup`.

    python -m fea_core.mcp.setup              # write into all found clients (with a backup of the original files)
    python -m fea_core.mcp.setup --dry-run    # only print what would change
    python -m fea_core.mcp.setup --remove     # remove the record "fea-studio"
    python -m fea_core.mcp.setup --only claude-desktop,cursor

Client = configuration file (JSON "mcpServers", VS Code "servers", Codex TOML). It is written only where the client is installed
(its folder exists). Trailing commas (JSONC – VS Code) are tolerated; a file that cannot be read
(JSON with comments, damaged) is NOT overwritten – the manual procedure is printed. Before the first change a backup
`<file>.fea-zaloha` is saved. Claude Code: preferably the command `claude mcp add --scope user fea-studio -e … -- …` (the name
before -e), otherwise `mcpServers` in `.claude.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

NAME = "fea-studio"
ROOT = Path(__file__).resolve().parents[2]


def server_spec(root: Path = ROOT, python: str | None = None) -> dict:
    """Start of the server stdio independently of the working folder of the client (PYTHONPATH = root of the program)."""
    py = python or sys.executable
    if os.name == "nt" and Path(py).name.lower() == "pythonw.exe":     # from the GUI: stdio needs a console python
        py = str(Path(py).with_name("python.exe"))
    return {"command": py, "args": ["-P", "-m", "fea_core.mcp"],
            "env": {"PYTHONPATH": str(root), "PYTHONIOENCODING": "utf-8"}}


@dataclass
class Client:
    key: str
    label: str
    path: Path                 # configuration file
    marker: Path               # folder whose existence = the client is installed
    kind: str = "mcpServers"   # mcpServers | vscode | codex


def clients(env: dict | None = None) -> list[Client]:
    e = env if env is not None else os.environ
    home = Path(e.get("USERPROFILE") or e.get("HOME") or ".")
    if os.name == "nt":
        appdata = Path(e.get("APPDATA") or home / "AppData" / "Roaming")
        local = Path(e.get("LOCALAPPDATA") or home / "AppData" / "Local")
    else:                                   # Linux: configuration of the clients in ~/.config (XDG)
        appdata = Path(e.get("XDG_CONFIG_HOME") or home / ".config")
        local = Path(e.get("XDG_DATA_HOME") or home / ".local" / "share")
    out = [Client("claude-desktop", "Claude Desktop", appdata / "Claude" / "claude_desktop_config.json",
                  appdata / "Claude")]
    # Claude Desktop from the Microsoft Store (MSIX) has its configuration in the virtualised folder of the package
    pk = local / "Packages"
    if os.name == "nt" and pk.is_dir():
        for d in sorted(pk.glob("Claude_*")):
            cfg = d / "LocalCache" / "Roaming" / "Claude"
            out.append(Client("claude-desktop-store", "Claude Desktop (Store)", cfg / "claude_desktop_config.json", cfg))
    out += [
        Client("claude-code", "Claude Code", home / ".claude.json", home / ".claude"),
        Client("cursor", "Cursor", home / ".cursor" / "mcp.json", home / ".cursor"),
        Client("windsurf", "Windsurf", home / ".codeium" / "windsurf" / "mcp_config.json", home / ".codeium" / "windsurf"),
        Client("antigravity", "Antigravity", home / ".gemini" / "antigravity" / "mcp_config.json",
               home / ".gemini" / "antigravity"),
        Client("gemini-cli", "Gemini CLI", home / ".gemini" / "settings.json", home / ".gemini"),
        Client("lmstudio", "LM Studio", home / ".lmstudio" / "mcp.json", home / ".lmstudio"),
        Client("vscode", "VS Code (Copilot)", appdata / "Code" / "User" / "mcp.json", appdata / "Code" / "User", "vscode"),
        Client("vscode-insiders", "VS Code Insiders", appdata / "Code - Insiders" / "User" / "mcp.json",
               appdata / "Code - Insiders" / "User", "vscode"),
        Client("codex", "Codex CLI", home / ".codex" / "config.toml", home / ".codex", "codex"),
    ]
    return out


# ------------------------------------------------------------------ editing of the files
def _backup(p: Path) -> None:
    b = p.with_name(p.name + ".fea-zaloha")
    if p.exists() and not b.exists():
        shutil.copy2(p, b)


def _jsonc_scan(raw: str) -> tuple[str, bool]:
    """JSONC (VS Code, Cursor…): text without trailing commas before } / ] (outside the strings) and a flag whether it contains
    comments // or /* */ – those would be lost by the rewrite, such a file is not rewritten.
    """
    out, i, n, comments = [], 0, len(raw), False
    while i < n:
        ch = raw[i]
        if ch == '"':                                   # the string including escaping
            j = i + 1
            while j < n and raw[j] != '"':
                j += 2 if raw[j] == "\\" else 1
            out.append(raw[i:j + 1])
            i = j + 1
            continue
        if ch == "/" and i + 1 < n and raw[i + 1] in "/*":
            comments = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and raw[j] in " \t\r\n":
                j += 1
            if j < n and raw[j] in "}]":               # trailing comma – leave out
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out), comments


def _json_update(c: Client, spec: dict, remove: bool) -> tuple[str, str | None]:
    """(state, new text | None). State: added / updated / unchanged / removed / not found / ERROR …"""
    data = {}
    if c.path.exists():
        raw = c.path.read_text(encoding="utf-8-sig")
        if raw.strip():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                cleaned, comments = _jsonc_scan(raw)    # trailing commas (JSONC) – the rewrite loses nothing
                try:
                    if comments:
                        raise e
                    data = json.loads(cleaned)
                except json.JSONDecodeError:
                    why = "contains comments" if comments else f"{e.msg}, line {e.lineno}"
                    return f"ERROR: the file cannot be read as JSON ({why}) – not overwriting", None
        if not isinstance(data, dict):
            return "ERROR: the file root is not a JSON object – not overwriting", None
    key = "servers" if c.kind == "vscode" else "mcpServers"
    entry = dict(spec, type="stdio") if c.kind == "vscode" else dict(spec)
    servers = data.get(key)
    if servers is None:
        servers = {}
    if not isinstance(servers, dict):
        return f"ERROR: “{key}” is not an object – not overwriting", None
    old = servers.get(NAME)
    if remove:
        if old is None:
            return "not written", None
        servers.pop(NAME)
        state = "removed"
    else:
        if old == entry:
            return "no changes", None
        servers[NAME] = entry
        state = "updated" if old is not None else "added"
    data[key] = servers
    return state, json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _toml_str(s: str) -> str:
    return "'" + s + "'" if "'" not in s else json.dumps(s)        # literal string without escaping of the backslashes


_OURS = re.compile(r'^\s*\[\s*mcp_servers\.("fea-studio"|fea-studio)(\.env)?\s*\]\s*$')
_HEADER = re.compile(r"^\s*\[")


def _codex_update(c: Client, spec: dict, remove: bool) -> tuple[str, str | None]:
    """Codex CLI (TOML): the section [mcp_servers.fea-studio] (+ [….env]) is replaced line by line, the rest of the file unchanged."""
    text = c.path.read_text(encoding="utf-8-sig") if c.path.exists() else ""
    keep, ours, inside = [], [], False
    for line in text.splitlines():
        if _HEADER.match(line):
            inside = bool(_OURS.match(line))
        (ours if inside else keep).append(line)
    rest = "\n".join(keep).rstrip()
    if remove:
        return ("removed", (rest + "\n") if rest else "") if ours else ("not written", None)
    env = ", ".join(f"{k} = {_toml_str(v)}" for k, v in spec["env"].items())
    block = (f"[mcp_servers.{NAME}]\ncommand = {_toml_str(spec['command'])}\n"
             f"args = [{', '.join(_toml_str(a) for a in spec['args'])}]\nenv = {{ {env} }}")
    if "\n".join(ours).strip() == block:
        return "no changes", None
    return ("updated" if ours else "added"), (rest + "\n\n" if rest else "") + block + "\n"


def _claude_add_args(exe: str, name: str, command: str, args: list[str], env: dict) -> list[str]:
    """`claude mcp add` – the name of the server MUST stand before -e: the option -e/--env is variadic and a name after it would be taken
    by the CLI as another variable ("Invalid environment variable format: fea-studio").
    """
    out = [exe, "mcp", "add", "--scope", "user", name]
    for k, v in env.items():
        out += ["-e", f"{k}={v}"]
    return out + ["--", command, *args]


def _run_claude(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)


def _claude_get(exe: str) -> dict | None:
    """Record fea-studio in the scope user from `claude mcp get` ({command, args, env}) or None."""
    try:
        r = _run_claude([exe, "mcp", "get", NAME])
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or "Scope: User" not in (r.stdout or ""):
        return None
    d: dict = {"command": None, "args": [], "env": {}}
    in_env = False
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith("Command:"):
            d["command"], in_env = s[len("Command:"):].strip(), False
        elif s.startswith("Args:"):
            d["args"], in_env = s[len("Args:"):].split(), False
        elif s.startswith("Environment:"):
            in_env = True
        elif in_env and "=" in s:
            k, v = s.split("=", 1)
            d["env"][k] = v
        elif s:
            in_env = False
    return d


def _claude_cli(spec: dict, remove: bool, dry: bool) -> str | None:
    """Claude Code through its own CLI (safer than editing .claude.json at run time). None = the CLI is not there.
    An existing identical record is not changed; a different one is replaced (if adding fails, the original is restored).
    """
    exe = shutil.which("claude")
    if not exe:
        return None
    add = _claude_add_args(exe, NAME, spec["command"], spec["args"], spec["env"])
    if dry:
        shown = subprocess.list2cmdline(["claude", *add[1:]]) if os.name == "nt" else " ".join(["claude", *add[1:]])
        return f"via `{shown}` (dry-run – nothing executed)"
    cur = _claude_get(exe)
    if remove:
        if cur is None:
            return "not written"
        r = _run_claude([exe, "mcp", "remove", "--scope", "user", NAME])
        return "removed (claude mcp remove)" if r.returncode == 0 else f"ERROR claude mcp remove: {r.stderr.strip()[:200]}"
    if cur is not None and cur["command"] == spec["command"] and cur["args"] == list(spec["args"]) \
            and all(cur["env"].get(k) == v for k, v in spec["env"].items()):
        return "no changes"
    if cur is not None:
        _run_claude([exe, "mcp", "remove", "--scope", "user", NAME])
    r = _run_claude(add)
    if r.returncode == 0:
        return "updated (claude mcp add --scope user)" if cur else "added (claude mcp add --scope user)"
    err = (r.stderr or r.stdout or "").strip()[:200]
    if cur is not None and cur["command"]:                     # return the original record
        back = _run_claude(_claude_add_args(exe, NAME, cur["command"], cur["args"], cur["env"]))
        err += " – original entry " + ("obnoven" if back.returncode == 0 else "FAILED to restore")
    return f"ERROR claude mcp add: {err}"


def apply(only: list[str] | None = None, remove: bool = False, dry: bool = False, env: dict | None = None,
          spec: dict | None = None, use_cli: bool = True) -> list[tuple[str, str, str]]:
    """Writes / removes the record in all the found clients. Returns [(client, file, state)]."""
    spec = spec or server_spec()
    rows = []
    for c in clients(env):
        if only and c.key not in only:
            continue
        if not c.marker.is_dir():
            continue
        if c.key == "claude-code" and use_cli:
            st = _claude_cli(spec, remove, dry)
            if st is not None:
                rows.append((c.label, "claude mcp (user)", st))
                continue
        state, new = (_codex_update if c.kind == "codex" else _json_update)(c, spec, remove)
        if new is not None and not dry:
            c.path.parent.mkdir(parents=True, exist_ok=True)
            _backup(c.path)
            tmp = c.path.with_name(c.path.name + ".tmp")
            tmp.write_text(new, encoding="utf-8")
            tmp.replace(c.path)
        rows.append((c.label, str(c.path), state + (" (dry-run)" if dry and new is not None else "")))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Registration of the FEA Studio MCP server with LLM clients")
    ap.add_argument("--dry-run", action="store_true", help="only print, write nothing")
    ap.add_argument("--remove", action="store_true", help="remove the fea-studio entry")
    ap.add_argument("--only", default="", help="only selected clients (comma-separated): " + ", ".join(c.key for c in clients()))
    ap.add_argument("--no-cli", action="store_true", help="Claude Code via .claude.json instead of the claude command")
    a = ap.parse_args(argv)
    only = [s.strip() for s in a.only.split(",") if s.strip()] or None
    rows = apply(only, a.remove, a.dry_run, use_cli=not a.no_cli)
    spec = server_spec()
    print(f"MCP server „{NAME}“: {spec['command']} {' '.join(spec['args'])}")
    if not rows:
        print("No supported client found (Claude Desktop/Code, Cursor, Windsurf, Antigravity, Gemini CLI, "
              "LM Studio, VS Code, Codex). Manual configuration: fea_core/mcp/README.md")
    for label, path, state in rows:
        print(f"  {label:24s} {state:34s} {path}")
    if any(s.startswith("ERROR") for _, _, s in rows):
        print("For clients with an error, add the entry by hand following fea_core/mcp/README.md.")
    if rows and not a.dry_run and not a.remove:
        print("Restart the clients (Claude Desktop, Cursor, VS Code…) so they load the server.")
    return 1 if any(s.startswith("ERROR") for _, _, s in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
