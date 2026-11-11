"""Paths given from outside (MCP, REST, study file): rejection of UNC network paths.

Access to \\\\\\\\server\\\\share (also written with mixed slashes "/\\\\server", "\\\\/server" or as \\\\\\\\?\\\\UNC\\\\…)
would make Windows log in to a foreign server (leak of the NTLM hash); for a study file also reading from a foreign server.
"""
from __future__ import annotations

from pathlib import Path


# namespace prefixes NT – Win32 passes them unchanged (\??\UNC\server\… = network path)
_NT_PREFIXES = ("\\??\\", "\\GLOBAL??\\", "\\DEVICE\\", "\\DOSDEVICES\\")


def is_unc(raw) -> bool:
    s = str(raw).strip().replace("/", "\\")
    return s.startswith("\\\\") or s.upper().startswith(_NT_PREFIXES)


def reject_unc(raw, what: str = "Path") -> Path:
    """Path from text; a network path (UNC) or a device path = ValueError."""
    if is_unc(raw):
        raise ValueError(f"{what}: network (UNC) paths are not allowed: {raw}")
    return Path(str(raw).strip())
