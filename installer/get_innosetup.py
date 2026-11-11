#!/usr/bin/env python3
"""Downloads the portable Inno Setup 6 compiler (ISCC.exe) into tools/innosetup – for building setup.exe (make_package.py).

    python installer/get_innosetup.py

Source: the NuGet package "Tools.InnoSetup" (https://www.nuget.org/packages/Tools.InnoSetup – an unofficial packing of the official
Inno Setup 6 including translations, ISCC.exe signed by the publisher of Inno Setup). The nupkg is a zip; the SHA-256 digest is pinned.
Nothing is installed into the system; tools/innosetup is not in git. Alternative: install Inno Setup 6 from jrsoftware.org
and pass the path `make_package.py --iscc "…\\\\ISCC.exe"`.
"""
from __future__ import annotations

import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "6.7.1"
URL = f"https://api.nuget.org/v3-flatcontainer/tools.innosetup/{VERSION}/tools.innosetup.{VERSION}.nupkg"
SHA256 = "00a5a07e8a4bd3a30364cb579d744677b37adfc22e365099cb01b8bd1e651828"
DEST = ROOT / "tools" / "innosetup"


def main() -> int:
    if (DEST / "ISCC.exe").exists():
        print(f"ISCC.exe is already there: {DEST}")
        return 0
    print(f"stahuji Inno Setup {VERSION} (NuGet Tools.InnoSetup)…")
    req = urllib.request.Request(URL, headers={"User-Agent": "FEAStudio-build"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        print(f"ERROR: the package digest does not match ({got}) – download aborted", file=sys.stderr)
        return 1
    tmp = DEST.with_name("innosetup.part")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for m in z.namelist():
            if m.startswith("tools/") and not m.endswith("/"):
                out = tmp / m[len("tools/"):]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(m))
    if not (tmp / "ISCC.exe").exists():
        print("ERROR: tools/ISCC.exe is missing in the package", file=sys.stderr)
        return 1
    shutil.rmtree(DEST, ignore_errors=True)
    tmp.rename(DEST)
    print(f"OK: {DEST / 'ISCC.exe'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
