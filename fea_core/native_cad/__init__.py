"""Reading native files of Autodesk Inventor (.ipt) and Fusion (.f3d, .f3z) without the CAD and without an export to STEP
(optional module "native", fea_core/modules.json).

Both programs store the exact geometry in the ShapeManager kernel (stream "ASM BinaryFile"); our own reader (`brep.py`)
converts it to OCC bodies in mm. Verified against the STEP export from Fusion (volume and area 1e-14 for parts of planes, cylinders,
cones and NURBS). For the rest of the program the model is saved as STEP into a cache (`to_step`) – the study, the mesh, the project .fea
and the geometry update then work without change.

Check of the result (always): every body must be topologically valid (BRepCheck) and have a positive volume; a body that cannot be converted
(a helix of a thread without a stored approximation, an elliptic cylinder) stops the import with an explanation – an error is better than
a silently wrong geometry. Assemblies of Fusion (.f3d with components, .f3z with nested documents) are assembled by `f3d_design` from the design data:
positions of the occurrences, visibility of the bodies and occurrences (hidden bodies → `hidden`, in the metadata "hidden": true), names
as in the STEP export ("Deska:1", "Deska:1_2"). Inventor assemblies .iam are assembled by `iam` from the referenced parts .ipt (they must
lie next to the assembly). Not supported: materials from the CAD, suppressed / renamed occurrences in .iam.
Unofficial reading of the Autodesk formats – check the results (by comparison with the STEP export) for important models.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

SUFFIXES = (".ipt", ".iam", ".f3d", ".f3z")
FORMAT_VERSION = 3                       # version of the conversion (if the reader changes, the cache is renewed)


class NativeError(ValueError):
    """The file cannot be read or converted (text in Czech)."""


def is_native(path) -> bool:
    return Path(str(path)).suffix.lower() in SUFFIXES


def require_module() -> None:
    from .. import modules
    if not modules.available("native"):
        raise modules.missing("native")


def _check(solid, label: str) -> str | None:
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    if not BRepCheck_Analyzer(solid).IsValid():
        return f"{label}: the solid is not topologically valid"
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, g)
    if not g.Mass() > 0:
        return f"{label}: non-positive volume ({g.Mass():.4g} mm³) – faces converted with wrong orientation"
    return None


def _transform(solid, m):
    if m is None:
        return solid
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    t = gp_Trsf()
    t.SetValues(m[0], m[1], m[2], m[3] * 10.0, m[4], m[5], m[6], m[7] * 10.0, m[8], m[9], m[10], m[11] * 10.0)
    return BRepBuilderAPI_Transform(solid, t, True).Shape()


def _fusion_assembly(p: Path, report: list[str] | None, hidden: list[str] | None):
    """Fusion assembly from the design data (positions of occurrences, visibility, XREF) → (parts, problems); DesignError = not possible."""
    from ..geometry import Part
    from . import brep
    from .f3d import design_documents
    from .f3d_design import DesignError, instances, redirection_roles
    docs, root = design_documents(p)
    if root is None:
        raise DesignError("root document not found")
    roles = redirection_roles(docs[root]["redir"], {k: k for k in docs})
    inst = instances({k: (d["bulk"], d["meta"]) for k, d in docs.items()}, root, roles,
                     {k: d["label"] for k, d in docs.items()})
    if not inst or not any(i.visible for i in inst):
        raise DesignError("no visible body found in the design")
    cache: dict[tuple[str, str], object] = {}
    notes: list[str] = []
    parts, problems = [], []
    totals: dict[tuple[str, bool], int] = {}
    for i in inst:
        totals[(i.name, i.visible)] = totals.get((i.name, i.visible), 0) + 1
    counts: dict[tuple[str, bool], int] = {}
    for i in inst:
        streams = docs[i.document]["streams"]
        if i.stream not in streams:
            raise DesignError(f"stream {i.stream} is missing")
        key = (i.document, i.stream)
        if key not in cache:
            try:
                cache[key] = brep.bodies(streams[i.stream])
            except Exception as e:  # noqa: BLE001 – an unconverted stream is reported as a problem
                cache[key] = e
        got = cache[key]
        label = i.name
        if isinstance(got, Exception) or i.body >= len(got):
            why = got if isinstance(got, Exception) else "the body is missing in the stream"
            (problems if i.visible else notes).append(f"{label}: the geometry stream cannot be converted ({why})")
            continue
        _, sols, err = got[i.body]
        if err:
            (problems if i.visible else notes).append(f"{label}: {err}")
            continue
        for s in sols:
            c = counts[(i.name, i.visible)] = counts.get((i.name, i.visible), 0) + 1
            many = totals[(i.name, i.visible)] > 1 or len(sols) > 1
            if i.visible:
                name = f"{label}_{c}" if many else label
            else:
                name = f"{label} (hidden {c})" if many else f"{label} (hidden)"
            s = _transform(s, i.matrix)
            bad = _check(s, name)
            if bad:
                (problems if i.visible else notes).append(bad)
                continue
            parts.append(Part(name, s))
            if not i.visible and hidden is not None:
                hidden.append(name)
    if report is not None:
        n_occ = len({i.name for i in inst})
        if n_occ > 1 or len(docs) > 1:
            report.append(f"Fusion assembly: {n_occ} occurrences, {len(docs)} document(s) – component positions from the design data")
        n_hidden = sum(1 for x in parts if hidden is not None and x.name in hidden)
        if n_hidden:
            report.append(f"Bodies hidden in Fusion: {n_hidden} – loaded as hidden and excluded from the analysis (Model step, Show / Analyze columns)")
        report += [f"Hidden body skipped: {x}" for x in notes]
        if n_occ > 1:
            report.append("Visibility of occurrences and bodies is taken from Fusion (unofficial reading) – check in the Model step which bodies are in the analysis")
    return parts, problems


def _inventor_assembly(p: Path, report: list[str] | None, fasteners: list[dict] | None = None):
    """Inventor assembly: the .ipt parts at the positions of the occurrences → (parts, problems)."""
    from ..geometry import Part
    from . import brep
    from .iam import instances
    from .ipt import brep_streams
    inst, _ = instances(p, report, fasteners)
    cache: dict = {}
    parts, problems = [], []
    for i in inst:
        if i.path not in cache:
            sols, errs = [], []
            try:
                for data in brep_streams(i.path):
                    got, e = brep.solids(data)
                    sols += [s for _, _, s in got]
                    errs += e
            except ValueError as e:                          # IptError
                errs.append(str(e))
            except Exception as e:  # noqa: BLE001 – a damaged part
                errs.append(f"the part cannot be read: {e}")
            cache[i.path] = (sols, errs)
        sols, errs = cache[i.path]
        if errs:
            problems += [f"{i.name} ({i.path.name}): {e}" for e in errs]
            continue
        for k, s in enumerate(sols):
            name = i.name if len(sols) == 1 else f"{i.name}_{k + 1}"
            s = _transform(s, i.matrix)
            bad = _check(s, name)
            if bad:
                problems.append(bad)
            parts.append(Part(name, s))
    return parts, problems


def read_native(path: str | Path, report: list[str] | None = None, hidden: list[str] | None = None,
                fasteners: list[dict] | None = None) -> list:
    """Bodies of the native file as parts (geometry.Part) in mm, position in the assembly. hidden = a list to which the names
    of the bodies hidden in the CAD are added (they are read; the caller hides them and leaves them out of the calculation).
    """
    from ..geometry import Part
    from . import brep
    p = Path(path)
    suf = p.suffix.lower()
    if suf not in SUFFIXES:
        raise NativeError(f"Unsupported file {p.name} (supported: {', '.join(SUFFIXES)})")
    parts, problems, why = None, [], ""
    if suf == ".iam":
        try:
            parts, problems = _inventor_assembly(p, report, fasteners)
        except NativeError:
            raise
        except ValueError as e:                              # IamError, IptError – a clear text
            raise NativeError(str(e)) from None
        except Exception as e:  # noqa: BLE001
            raise NativeError(f"The assembly {p.name} cannot be read: {e}") from None
    if suf in (".f3d", ".f3z"):
        try:
            parts, problems = _fusion_assembly(p, report, hidden)
        except (ValueError, KeyError, IndexError, struct.error) as e:   # DesignError, F3dError – try without the assembly
            parts, why = None, str(e)
            if hidden is not None:
                hidden.clear()
    if parts is None:
        groups: list[tuple[str, list[bytes]]] = []
        try:
            if suf == ".ipt":
                from .ipt import brep_streams
                groups.append((p.stem, brep_streams(p)))
            else:
                from .f3d import documents, model_streams
                docs = documents(p)
                for doc, streams in docs:
                    groups.append((doc, [d for _, d in model_streams(streams)]))
                if report is not None and (len(docs) > 1 or any(len(model_streams(s)) > 1 for _, s in docs)):
                    report.append(f"Fusion assembly: the component tree could not be read ({why}) – bodies are in the coordinates of their components and hidden bodies are not distinguished; check the part positions (otherwise export STEP)")
        except NativeError:
            raise
        except ValueError as e:                              # IptError, F3dError, CFBError – a clear text
            raise NativeError(str(e)) from None
        except Exception as e:  # noqa: BLE001 – a damaged / unknown file
            raise NativeError(f"The file {p.name} cannot be read: {e}") from None
        if not groups:
            raise NativeError(f"The file {p.name} contains no solid (B-rep)")
        parts = []
        for doc, streams in groups:
            solids = []
            for data in streams:
                try:
                    got, errors = brep.solids(data)
                except Exception as e:  # noqa: BLE001
                    problems.append(f"{doc}: the geometry stream cannot be converted ({e})")
                    continue
                problems += [f"{doc}: {e}" for e in errors]
                solids += got
            for k, (_, _, s) in enumerate(solids):
                name = doc if len(solids) == 1 else f"{doc}_{k + 1}"
                bad = _check(s, name)
                if bad:
                    problems.append(bad)
                parts.append(Part(name, s))
    if problems:
        raise NativeError(f"Model {p.name} cannot be converted reliably:\n  " + "\n  ".join(problems[:12])
                          + ("\n  …" if len(problems) > 12 else "")
                          + "\nExport it from CAD to STEP (Fusion add-in / Inventor macro).")
    if not parts:
        raise NativeError(f"The file {p.name} contains no solid")
    used: dict[str, int] = {}
    for part in parts:                                   # the same names (occurrences, .f3z) – as read_step
        if part.name in used:
            used[part.name] += 1
            old, part.name = part.name, f"{part.name} ({used[part.name]})"
            if hidden is not None and old in hidden:
                hidden.append(part.name)
        else:
            used[part.name] = 1
    return parts


def cache_dir() -> Path:
    from ..config import user_data_dir
    return user_data_dir() / "native_cache"


MAX_NATIVE_FILE = 3 << 30           # larger native files are not read at all (the readers have their own, lower limits)


def _sha256(p: Path):
    """SHA-256 of a file read in blocks; a file over MAX_NATIVE_FILE = NativeError (checked before anything is read)."""
    if p.stat().st_size > MAX_NATIVE_FILE:
        raise NativeError(f"The file {p.name} is too large ({p.stat().st_size >> 20} MB)")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h


def to_step(path: str | Path, report: list[str] | None = None) -> Path:
    """Converts the native file to STEP in the cache (according to the content of the file) and returns the path; next to the STEP the metadata
    <name>.json (source, document, parts – fea_core.cad_import). The same file is not converted a second time.
    """
    require_module()
    p = Path(path)
    h = _sha256(p)                                       # same key as sha256(content + "|v…") – read in blocks
    h.update(f"|v{FORMAT_VERSION}".encode())
    if p.suffix.lower() == ".iam":                       # assembly: the key also from the referenced parts
        from .iam import instances
        try:
            _, used = instances(p)
        except ValueError as e:
            raise NativeError(str(e)) from None
        for f in used[1:]:
            h.update(f.name.encode("utf-8", "replace") + _sha256(f).digest())
    key = h.hexdigest()[:20]
    d = cache_dir() / key
    step = d / (p.stem[:80] + ".step")
    info = d / "native.json"
    if step.is_file() and info.is_file():
        try:
            if report is not None:
                report += json.loads(info.read_text(encoding="utf-8")).get("warnings", [])
        except (OSError, ValueError):
            pass
        return step
    warn: list[str] = []
    hidden: list[str] = []
    fasteners: list[dict] = []
    parts = read_native(p, warn, hidden, fasteners)
    from ..geometry import write_step
    d.mkdir(parents=True, exist_ok=True)
    tmp = step.with_suffix(f".{os.getpid()}.tmp")
    write_step(parts, tmp)
    os.replace(tmp, step)
    src = "Autodesk Inventor" if p.suffix.lower() in (".ipt", ".iam") else "Autodesk Fusion"
    meta = {"source": src, "document": p.stem, "units": "mm", "native_file": p.name,
            "parts": [{"name": x.name, **({"hidden": True} if x.name in hidden else {})} for x in parts],
            "notes": ["Converted from the native file by the FEA Studio reader (module native), without the CAD."]}
    if fasteners:
        meta["fasteners"] = fasteners
    step.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    info.write_text(json.dumps({"source": p.name, "warnings": warn}, ensure_ascii=False), encoding="utf-8")
    if report is not None:
        report += warn
    return step
