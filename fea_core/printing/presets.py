"""Presets of print settings (F5): name → fields of PrintSpec (layer, line, perimeters, upper/lower layers, infill, pattern,
thin places). Only "3D print standard" (the default PrintSpec) is built in; own presets are entered and confirmed by the
user (values are not written from memory) and are stored next to the user library of materials
(%APPDATA%/FEAStudio/print_presets.json). The build direction and "enabled" belong to the part, not to the preset.

The file is read as robustly as the material library: a bad record is skipped (it stays in the file), an unreadable
file → PresetError when saving (it is not overwritten).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .model import PrintSpec

BUILTIN_NAME = "3D print standard"
LEGACY_BUILTIN_NAMES = ("3D print standard",)
PRESET_KEYS = ("layer_height", "line_width", "walls", "top_layers", "bottom_layers", "infill", "pattern",
               "solid_thin", "thin_width")


class PresetError(ValueError):
    pass


@dataclass
class PrintPreset:
    name: str
    settings: dict = field(default_factory=dict)      # pole PRESET_KEYS
    note: str = ""
    builtin: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "settings": dict(self.settings), "note": self.note}


def _builtin() -> list[PrintPreset]:
    d = PrintSpec("_").to_dict()
    return [PrintPreset(BUILTIN_NAME, {k: d[k] for k in PRESET_KEYS},
                        "Standard 3D print profile 0.20 mm (FEA Studio default settings)", builtin=True)]


def presets_path():
    from ..materials import user_library_path
    return user_library_path().parent / "print_presets.json"


def check_settings(settings: dict) -> dict:
    """Verified field of a preset (types, ranges; infill in % → 0–1). An unknown key = error."""
    from .infill import PATTERNS
    if not isinstance(settings, dict) or not settings:
        raise PresetError("A preset must contain settings {key: value}")
    bad = set(settings) - set(PRESET_KEYS)
    if bad:
        raise PresetError(f"Unknown preset items {', '.join(sorted(bad))} (allowed: {', '.join(PRESET_KEYS)})")
    out = {}
    for k, v in settings.items():
        try:
            if k in ("walls", "top_layers", "bottom_layers"):
                x = float(v)
                if isinstance(v, bool) or x != int(x) or x < 0:
                    raise ValueError
                out[k] = int(x)
            elif k in ("layer_height", "line_width"):
                out[k] = float(v)
                if out[k] <= 0:
                    raise ValueError
            elif k == "thin_width":
                out[k] = float(v)
                if out[k] < 0:
                    raise ValueError
            elif k == "infill":
                x = float(v)
                x = x / 100.0 if x > 1.0 else x
                if not 0.0 <= x <= 1.0:
                    raise ValueError
                out[k] = x
            elif k == "pattern":
                if v not in PATTERNS:
                    raise PresetError(f"Unknown infill pattern '{v}' ({' | '.join(PATTERNS)})")
                out[k] = str(v)
            elif k == "solid_thin":
                if not isinstance(v, bool):
                    raise ValueError
                out[k] = v
        except (TypeError, ValueError):
            raise PresetError(f"Preset: invalid value {k} = {v!r}") from None
    return out


def _raw(strict: bool) -> list:
    p = presets_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        if strict:
            raise PresetError(f"The print presets file {p} cannot be read ({e}) – fix or rename it, otherwise saving would overwrite it") from None
        return []
    return data if isinstance(data, list) else []


def _load(d) -> PrintPreset | None:
    try:
        if not isinstance(d, dict) or not str(d.get("name", "")).strip():
            return None
        return PrintPreset(str(d["name"]), check_settings(d.get("settings") or {}), str(d.get("note", "")))
    except (PresetError, TypeError, ValueError):
        return None


def list_presets() -> list[PrintPreset]:
    """Built-in + user presets (bad records are skipped, the user preset does not overwrite the built-in one)."""
    out = _builtin()
    names = {p.name for p in out}
    for p in map(_load, _raw(strict=False)):
        if p is not None and p.name not in names:
            out.append(p)
            names.add(p.name)
    return out


def get_preset(name: str) -> PrintPreset:
    if name in LEGACY_BUILTIN_NAMES:
        name = BUILTIN_NAME
    for p in list_presets():
        if p.name == name:
            return p
    raise PresetError(f"Preset '{name}' does not exist (presets: {', '.join((p.name for p in list_presets()))})")


def _write(raw: list) -> None:
    p = presets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def save_preset(name: str, settings: dict, note: str = "") -> PrintPreset:
    """Saves / replaces a user preset. The built-in one cannot be overwritten; records that this version cannot load
    stay in the file unchanged.
    """
    name = str(name).strip()
    if not name:
        raise PresetError("A preset must have a name")
    if name in {p.name for p in _builtin()} or name in LEGACY_BUILTIN_NAMES:
        raise PresetError(f"'{name}' is a built-in preset – choose a different name")
    pr = PrintPreset(name, check_settings(settings), str(note))
    raw = [d for d in _raw(strict=True) if not (isinstance(d, dict) and d.get("name") == name)]
    _write(raw + [pr.to_dict()])
    return pr


def delete_preset(name: str) -> None:
    if name in {p.name for p in _builtin()} or name in LEGACY_BUILTIN_NAMES:
        raise PresetError(f"'{name}' is a built-in preset – it cannot be deleted")
    raw = _raw(strict=True)
    keep = [d for d in raw if not (isinstance(d, dict) and d.get("name") == name)]
    if len(keep) == len(raw):
        raise PresetError(f"User preset '{name}' does not exist")
    _write(keep)


def preset_from_spec(spec: PrintSpec) -> dict:
    """Fields of the preset from the settings of a part (for "Save as preset")."""
    d = spec.to_dict()
    return {k: d[k] for k in PRESET_KEYS}


def apply_preset(study, name: str, parts: list[str]) -> list[str]:
    """Applies the preset to parts (build direction, enabled and the note of the part stay). Returns the changed parts."""
    pr = get_preset(name)
    for p in parts:
        d = study.printing.spec(p).to_dict()
        d.update(pr.settings)
        d["part"] = p
        study.printing.parts[p] = PrintSpec.from_dict(d)
    return list(parts)
