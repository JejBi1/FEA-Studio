"""Import of a STEP from a CAD adapter: saving of the file + metadata, matching of the parts and materials with the library.

Metadata (JSON from the adapter, all optional):
    {"source": "Fusion 360", "document": "Držák", "units": "mm",
     "parts": [{"name": "Deska", "component": "Deska:1", "names": ["…"], "color": [r, g, b] | "#rrggbb",
                "material": "Steel", "material_props": {...}, "attributes": {"skupina.klíč": "hodnota"},
                "layers": ["…"]}]}
RGB colours 0–1 (or #rrggbb). Parts are matched by name (a STEP from the CAD carries the names of the bodies / components –
`fea_core.geometry.read_step`); a material is assigned only if it matches the library (exact name or the same
group of the designation converter `grades`), otherwise it is only returned in the answer with a hint.

Matching and materials are shared with the GUI (opening a STEP with metadata next to it) – `fea_core.cad_import`. The metadata
are saved next to the STEP with the same name (.json), so a GUI started on the import (`/api/open-gui`) takes them over too.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import cad_import
from ..cad_import import STEP_SUFFIXES, MetadataError, match_parts, name_keys, norm  # noqa: F401 – API modulu
from ..mcp.session import FeaError, safe_name


MAX_STEP_BYTES = 2 << 30                # largest accepted STEP (2 GB)


def read_limited(f, limit: int = MAX_STEP_BYTES) -> bytes:
    """Reads the uploaded file in blocks; larger than `limit` = error (not exhaustion of the memory)."""
    buf = bytearray()
    while chunk := f.read(1 << 20):
        buf.extend(chunk)
        if len(buf) > limit:
            raise FeaError(f"The file is larger than {limit >> 20} MB")
    return buf                                            # bytearray – without another copy (up to 2 GB)


def save_import(base_dir: Path, filename: str, data, metadata: dict | None,
                limit: int = MAX_STEP_BYTES) -> tuple[Path, Path | None]:
    """Saves the STEP (and the metadata) to <base_dir>/imports/<time>_<name>.step|.json.
    data = bytes or a file object (an upload): a file object is copied to the disk in blocks (not held in memory);
    larger than `limit` = error and the partial file is deleted. The saved size is in step.stat().st_size."""
    name = Path(filename or "model.step").name
    suffix = Path(name).suffix.lower()
    if suffix not in STEP_SUFFIXES:
        raise FeaError(f"A STEP file (.step / .stp) was expected, got '{name}'")
    stream = hasattr(data, "read")
    head = data.read(4096) if stream else data[:4096]
    if not bytes(head).lstrip()[:20].upper().startswith(b"ISO-10303-21"):      # without a copy of the whole file
        raise FeaError(f"File '{name}' is not STEP (the ISO-10303-21 header is missing)")
    d = Path(base_dir) / "imports"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name(Path(name).stem)}"
    i = 1
    while True:                                           # two imports within one second → _2, _3…
        step = d / (f"{stem}{suffix}" if i == 1 else f"{stem}_{i}{suffix}")
        try:
            f = open(step, "xb")                          # exclusively a new file (without a race of exists → write)
            break
        except FileExistsError:
            i += 1
    try:
        with f:
            if stream:
                n = len(head)
                f.write(head)
                while chunk := data.read(1 << 20):
                    n += len(chunk)
                    if n > limit:
                        raise FeaError(f"The file is larger than {limit >> 20} MB")
                    f.write(chunk)
            else:
                if len(data) > limit:
                    raise FeaError(f"The file is larger than {limit >> 20} MB")
                f.write(data)
    except BaseException:
        step.unlink(missing_ok=True)
        raise
    meta_path = None
    if metadata is not None:
        meta_path = cad_import.sidecar_path(step)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=1), encoding="utf-8")
    return step, meta_path


def parse_metadata(text: str | bytes | None) -> dict | None:
    try:
        return cad_import.parse_metadata(text)
    except MetadataError as e:
        raise FeaError(str(e)) from None


def library_material(cad_name: str, props: dict | None = None) -> tuple[str | None, dict]:
    """Library material for a name from the CAD: (name or None, information for the answer)."""
    lib, info = cad_import.library_material(cad_name, props)
    if cad_name and lib is None and "hint" not in info:
        info["hint"] = "material is not in the library – assign with fea_materials_assign (or custom_props)"
    return lib, info
