"""Unpacking of a conda package (.conda / .tar.bz2) into the environment without long paths (Windows MAX_PATH 260).

micromamba cannot unpack the package `chrono` (C++ header files in Library/include/chrono_thirdparty… have paths
longer than 260 characters when the project lies in a deeper folder) → the environment is created without chrono / pychrono and these two packages
are unpacked here: the headers, tests and sample data are left out (not needed for running), text files with the placeholder of the prefix
(info/paths.json, prefix_placeholder) are rewritten to the real path of the environment. It runs with the interpreter of the conda
environment (the package zstandard) – it is called by the installer and by the manual preparation. Usage:
    python -m fea_core.motion.conda_extract PACKAGE.conda ENVIRONMENT
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

SKIP = ("Library/include/", "include/", "Library/data/", "share/chrono/data/", "Library/share/chrono/data/",
        "Library/lib/", "lib/cmake/", "lib/libChrono_core.a", "Library/template_project/", "share/chrono/template_project/",
        "Library/share/chrono/demos", "share/chrono/demos", "data/")      # headers, data of the samples (1.2 GB), import. libraries


def _members(archive: Path):
    """(info tar, pkg tar) as TarFile objects."""
    if archive.suffix == ".conda":
        import zstandard
        z = zipfile.ZipFile(archive)
        out = []
        for prefix in ("info-", "pkg-"):
            name = next(n for n in z.namelist() if n.startswith(prefix) and n.endswith(".tar.zst"))
            data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(z.read(name)))
            out.append(tarfile.open(fileobj=io.BytesIO(data.read()), mode="r:"))
        return out
    t = tarfile.open(archive, "r:bz2")
    return [t, t]


def extract(archive: Path, env: Path, skip=SKIP) -> int:
    env = Path(env)
    info, pkg = _members(Path(archive))
    paths = {}
    try:
        pj = info.extractfile("info/paths.json")
        if pj is not None:
            for p in json.loads(pj.read().decode("utf-8")).get("paths", []):
                if p.get("prefix_placeholder"):
                    paths[p["_path"]] = (p["prefix_placeholder"], p.get("file_mode", "text"))
    except KeyError:
        pass
    n = 0
    for m in pkg.getmembers():
        if not m.isfile() or m.name.startswith("info/") or m.name.startswith(skip):
            continue
        target = env / m.name
        try:                                        # a member of the package must not point outside the environment (../, absolute path)
            target.resolve().relative_to(env.resolve())
        except ValueError:
            continue
        if len(str(target)) >= 255:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        data = pkg.extractfile(m).read()
        if m.name in paths and paths[m.name][1] == "text":
            ph = paths[m.name][0].encode("utf-8")
            data = data.replace(ph, str(env).replace("\\", "/").encode("utf-8") if b"/" in ph
                                else str(env).encode("utf-8"))
        target.write_bytes(data)
        n += 1
    return n


if __name__ == "__main__":
    print(extract(Path(sys.argv[1]), Path(sys.argv[2])), "files")
