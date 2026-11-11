"""Local registry "CAD document → project .fea" (only on this computer, user data of FEA Studio).

The CAD add-in knows only the id of the document; when the GUI is not running, it starts it with the update (`python -m fea_gui --cad-update STEP`)
and the GUI opens the project linked with the document according to the registry. Paths are not written into the CAD document or into the .fea.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

MAX_ENTRIES = 500


def registry_path() -> Path:
    from ..config import user_data_dir
    return user_data_dir() / "cad_links.json"


def _load() -> dict:
    try:
        d = json.loads(registry_path().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def remember(document_id: str, fea_path: str | Path, document: str = "") -> None:
    if not document_id:
        return
    d = _load()
    d[str(document_id)[:256]] = {"fea": str(Path(fea_path).resolve()), "document": str(document)[:200],
                                 "time": time.time()}
    if len(d) > MAX_ENTRIES:
        for k in sorted(d, key=lambda k: d[k].get("time", 0))[:len(d) - MAX_ENTRIES]:
            d.pop(k, None)
    p = registry_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


def lookup(document_id: str) -> Path | None:
    """Project .fea linked with the CAD document (an existing file), otherwise None."""
    e = _load().get(str(document_id or ""))
    if not isinstance(e, dict) or not e.get("fea"):
        return None
    p = Path(e["fea"])
    return p if p.is_file() else None
