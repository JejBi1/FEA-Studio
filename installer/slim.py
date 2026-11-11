"""Pruning of the installed packages: deletes the files the program never uses (stdlib, without dependencies).

Measured on Windows (`python installer/slim.py --report`): vtk.libs and vtkmodules (OCP links them, but uses only
~50 libraries), MKL architectures for old processors and threading layers that ccx does not use, tests and sample data
of the packages, Qt translations / QML, conda cache. The functions are idempotent. After pruning MKL / vtk the functionality is always checked
(`verify_*`); if the check fails, the installer reports it (the package can be restored with `fea_setup.py install --force`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setup_lang import T

IS_WIN = sys.platform == "win32"


def long_path(p) -> str:
    """Path for deleting on Windows also above 260 characters (prefix \\\\\\\\?\\\\; without it rmtree with
    ignore_errors would silently leave the deep files – the uv cache, the PySide6 QML – in a longer installation folder).
    """
    s = os.path.abspath(str(p))
    if os.name != "nt" or s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):                                  # network path (the installation does not use it)
        return s
    return "\\\\?\\" + s


def rmtree(p) -> None:
    """shutil.rmtree(ignore_errors=True) also for paths longer than 260 characters."""
    shutil.rmtree(long_path(p), ignore_errors=True)


def dir_size(p: Path) -> int:
    n = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n


def _rm(p: Path) -> int:
    """Deletes a file / folder, returns the freed bytes."""
    try:
        if p.is_dir():
            n = dir_size(p)
            rmtree(p)
            return n
        n = p.stat().st_size
        p.unlink()
        return n
    except OSError:
        return 0


def _say(msg: str) -> None:
    print("       " + msg, flush=True)


# ------------------------------------------------------------------ vtk (only a dependency of OCP)
_VTK_PROBE = r"""
import importlib, os, sys
import OCP
for m in ("OCP.STEPControl", "OCP.XCAFDoc", "OCP.BRepMesh", "OCP.BRepAlgoAPI", "OCP.AIS", "OCP.OpenGl", "OCP.V3d",
          "OCP.WNT" if sys.platform == "win32" else "OCP.Xw", "OCP.TopoDS", "OCP.BRepGProp", "OCP.GeomAPI"):
    try:
        importlib.import_module(m)
    except Exception:
        pass
keep = set()
if os.path.exists("/proc/self/maps"):
    for line in open("/proc/self/maps"):
        parts = line.split()
        if len(parts) >= 6:
            keep.add(os.path.normcase(os.path.realpath(parts[5])))
else:
    import psutil
    keep = {os.path.normcase(os.path.realpath(m.path)) for m in psutil.Process().memory_maps()}
sp = os.path.dirname(os.path.dirname(OCP.__file__))
print(sp)
for k in sorted(keep):
    if "vtk" in os.path.basename(k).lower() or "vtkmodules" in k or "vtk.libs" in k:
        print(k)
"""


def prune_vtk(venv_python: Path, site: Path, say=_say) -> int:
    """OCP (OCCT) links about 50 VTK libraries from the vtk package; the remaining ~150 + the Python modules vtkmodules do not need them.
    Windows: the libraries in vtk.libs; Linux: libvtk*.so directly in vtkmodules (+ vtk.libs for foreign dependencies).
    """
    dirs = [d for d in (site / "vtk.libs", site / "vtkmodules") if d.is_dir()]
    if not dirs:
        return 0
    r = subprocess.run([str(venv_python), "-c", _VTK_PROBE], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        say(T("vtk: ověření načtených knihoven selhalo – ponecháno celé: ",
              "vtk: checking the loaded libraries failed – kept complete: ") + r.stderr[-300:])
        return 0
    keep = {os.path.normcase(os.path.realpath(l)) for l in r.stdout.splitlines()[1:] if l.strip()}
    if len(keep) < 10:
        say(T("vtk: podezřele málo načtených knihoven – ponecháno celé", "vtk: suspiciously few libraries loaded – kept complete"))
        return 0
    freed = 0
    for d in dirs:
        for root, _dirs, files in os.walk(d, topdown=False):
            for f in files:
                fp = Path(root) / f
                if os.path.normcase(os.path.realpath(fp)) not in keep:
                    freed += _rm(fp)
            try:
                Path(root).rmdir()                       # empty folder
            except OSError:
                pass
    freed += _rm(site / "vtk.py")
    say(T(f"vtk: uvolněno {freed / 1e6:.0f} MB", f"vtk: freed {freed / 1e6:.0f} MB"))
    return freed


def strip_binaries(site: Path, venv: Path, say=_say, min_mb: float = 3.0) -> int:
    """Linux: `strip --strip-unneeded` of large shared libraries (vtk, gmsh, …; debug symbols make up to 40 %).
    Only if strip is in the system; it does not reduce the function of the package libraries in any way (dynamic symbols stay).
    """
    import shutil as _sh
    tool = _sh.which("strip")
    if IS_WIN or not tool:
        return 0
    before = 0
    after = 0
    cands = [f for d in (site / "vtkmodules", site / "vtk.libs", venv / "lib", site / "numpy.libs", site / "scipy.libs")
             if d.is_dir() for f in d.glob("*.so*") if f.is_file() and not f.is_symlink() and f.stat().st_size > min_mb * 1e6]
    for f in cands:
        n0 = f.stat().st_size
        r = subprocess.run([tool, "--strip-unneeded", str(f)], capture_output=True)
        if r.returncode == 0:
            before += n0
            after += f.stat().st_size
    if before:
        say(T(f"strip: {len(cands)} knihoven, uvolněno {(before - after) / 1e6:.0f} MB",
              f"strip: {len(cands)} libraries, freed {(before - after) / 1e6:.0f} MB"))
    return before - after


def verify_ocp(venv_python: Path) -> bool:
    r = subprocess.run([str(venv_python), "-c", "from OCP.STEPControl import STEPControl_Reader; import OCP.OpenGl"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0


# ------------------------------------------------------------------ Python od uv
def prune_python(venv: Path, say=_say) -> int:
    """Python downloaded by uv (python-build-standalone): without tests, IDLE, tkinter / Tcl, ensurepip and the bundled pip."""
    home = None
    try:
        for line in (venv / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            if line.startswith("home"):
                home = Path(line.split("=", 1)[1].strip())
    except OSError:
        return 0
    if home is not None and home.name == "bin":                # Linux: pyvenv.cfg ukazuje na <python>/bin
        home = home.parent
    ours = (venv.parent / "tools" / "python").resolve()
    if home is None or ours not in home.resolve().parents:
        return 0                                    # a foreign (system) Python is never deleted
    freed = prune_python_home(home)
    say(T(f"Python: uvolněno {freed / 1e6:.0f} MB", f"Python: freed {freed / 1e6:.0f} MB"))
    return freed


def prune_python_home(home: Path) -> int:
    """Deletes from the unpacked python-build-standalone (`home`) the parts the program does not need (also make_package for setup.exe)."""
    freed = 0
    rels = ["tcl", "include", "share", "libs", "Lib/idlelib", "Lib/ensurepip", "Lib/turtledemo", "Lib/test", "Lib/tkinter",
            "Lib/lib2to3", "Lib/site-packages", "DLLs/_tkinter.pyd", "DLLs/tcl86t.dll", "DLLs/tk86t.dll", "DLLs/_testcapi.pyd",
            "lib/tcl8.6", "lib/tk8.6", "lib/python3.11/idlelib", "lib/python3.11/ensurepip", "lib/python3.11/turtledemo",
            "lib/python3.11/test", "lib/python3.11/tkinter", "lib/python3.11/lib2to3", "lib/python3.11/site-packages/pip",
            "lib/python3.11/site-packages/setuptools", "lib/python3.11/lib-dynload/_tkinter.cpython-311-x86_64-linux-gnu.so",
            "lib/python3.11/config-3.11-x86_64-linux-gnu", "lib/libtcl8.6.so", "lib/libtk8.6.so"]
    for rel in rels:
        t = home / rel
        if t.exists():
            freed += _rm(t)
    return freed


# ------------------------------------------------------------------ gmsh and empty leftovers
def prune_gmsh(venv: Path, site: Path, say=_say) -> int:
    """gmsh from pip: the import library .lib (42 MB), examples and headers are not needed."""
    freed = 0
    for p in list((venv / "Lib").glob("gmsh*.lib")) + list(site.glob("*.whl")):
        freed += _rm(p)
    for d in (venv / "share" / "doc", venv / "share" / "gmsh", venv / "include"):
        if d.is_dir():
            freed += _rm(d)
    say(T(f"gmsh: uvolněno {freed / 1e6:.0f} MB", f"gmsh: freed {freed / 1e6:.0f} MB"))
    return freed


# ------------------------------------------------------------------ MKL
# Kept: mkl_rt, mkl_core, mkl_intel_thread (+ libiomp5), mkl_def, mkl_avx2, mkl_avx512, mkl_vml_{def,avx2,avx512},
# mkl_msg / mkl_intel_lp64 / mkl_gnu_thread (Linux ccx_i4 links through mkl_rt). The remaining architectures (avx, mc, mc3)
# use mkl_def (the generic path), so a processor without AVX2 works too.
_MKL_DROP = ("mkl_pgi_thread", "mkl_tbb_thread", "mkl_sequential", "mkl_scalapack", "mkl_blacs", "mkl_cdft", "mkl_avx.",
             "mkl_mc.", "mkl_mc3.", "mkl_vml_avx.", "mkl_vml_mc.", "mkl_vml_mc3.", "mkl_vml_cmpt", "mkl_sycl", "mkl_gf_",
             "mkl_intel_ilp64", "omptarget", "libiomp5md.pdb", "libiomp5md.lib", "tbbmalloc", "tbbbind",
             "tbb12", "tbb.", "libtbb", "mkl_lapack", "mkl_pardiso_mpi")


def prune_mkl(venv: Path, say=_say) -> int:
    libdir = venv / ("Library/bin" if IS_WIN else "lib")
    if not libdir.is_dir():
        return 0
    freed = 0
    for f in libdir.iterdir():
        n = f.name.lower()
        if (n.startswith(("mkl_", "libmkl_", "omptarget", "libiomp5", "tbb", "libtbb")) and any(d in n for d in _MKL_DROP)):
            if not n.startswith(("mkl_rt", "libmkl_rt")):
                freed += _rm(f)
    for d in (venv / "Library" / "lib", venv / "Library" / "include", venv / "Library" / "share", venv / "include",
              venv / "share" / "doc", venv / "share" / "man"):
        if d.is_dir():
            freed += _rm(d)
    say(T(f"MKL: uvolněno {freed / 1e6:.0f} MB", f"MKL: freed {freed / 1e6:.0f} MB"))
    return freed


def verify_mkl(venv_python: Path, mkl_rt: Path) -> bool:
    code = ("import ctypes,sys; l=ctypes.CDLL(sys.argv[1]); v=ctypes.create_string_buffer(256); "
            "l.mkl_get_version_string(v,256); print(v.value.decode())")
    r = subprocess.run([str(venv_python), "-c", code, str(mkl_rt)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0 and "Intel" in r.stdout


# ------------------------------------------------------------------ general surplus
_TEST_DIRS = ("tests", "test", "testing")
_KEEP_TEST = {"numpy": ("testing",), "scipy": (), "pytest": ("*",), "_pytest": ("*",), "numba": (), "matplotlib": ()}


def prune_generic(site: Path, keys: list[str], say=_say) -> int:
    """Tests of the packages numpy / scipy / matplotlib / numba / ngsolve…, .pyi stubs, bundled data and documentation."""
    freed = 0
    # test packages inside large libraries (not `numpy/testing`, `pytest`, which the code or pytest uses)
    for pkg in ("numpy", "scipy", "matplotlib", "numba", "llvmlite", "pandas", "PIL", "fontTools", "lxml", "shapely",
                "pydantic", "docx", "sectionproperties", "mpl_toolkits", "psutil", "cffi", "mpi4py", "ngsolve", "netgen"):
        base = site / pkg
        if not base.is_dir():
            continue
        for d in list(base.rglob("tests")):
            if d.is_dir():
                freed += _rm(d)
    # matplotlib: sample data and images (sample_data ~20 MB), unnecessary styles
    mpl = site / "matplotlib" / "mpl-data"
    for sub in ("sample_data", "images"):
        if (mpl / sub).is_dir() and sub == "sample_data":
            freed += _rm(mpl / sub)
    # type stubs (.pyi) – only for IDEs
    for pkg in ("PySide6", "shiboken6", "shiboken6_generator"):
        for p in (site / pkg).rglob("*.pyi") if (site / pkg).is_dir() else []:
            freed += _rm(p)
    # pip cache, documentation, headers
    for pkg in ("PySide6", "shiboken6", "PySide6_Essentials", "PySide6_Addons"):
        base = site / pkg
        for sub in ("include", "typesystems", "glue", "scripts", "examples", "doc", "support"):
            if (base / sub).is_dir() and sub in ("include", "typesystems", "glue", "scripts", "examples", "doc"):
                freed += _rm(base / sub)
    if freed:
        say(T(f"obecné: uvolněno {freed / 1e6:.0f} MB", f"generic: freed {freed / 1e6:.0f} MB"))
    return freed


# ------------------------------------------------------------------ Qt
# Qt: the GUI is pure QtWidgets (+ Svg icons, Test in the tests). QML / Quick, translations, software OpenGL, tools,
# plugins for Designer / QML / SQL and the import libraries go. Package layout: Windows PySide6/*.dll, Linux PySide6/Qt/lib.
_QT_DROP_DIRS = ("translations", "qml", "metatypes", "lib", "support", "libexec", "plugins/designer", "plugins/qmllint",
                 "plugins/qmltooling", "plugins/sqldrivers", "plugins/networkinformation", "plugins/tls")
_QT_DROP_FILES = ("opengl32sw.dll", "d3dcompiler_47.dll", "pyside6qml", "icudtl.dat", "pyside6.abi3.lib", "Qt6UiTools",
                  "libQt6UiTools", "QtUiTools", "Qt6Quick", "Qt6Qml", "libQt6Quick", "libQt6Qml", "QtQuick", "QtQml",
                  "libpyside6qml", "designer", "assistant", "linguist", "lupdate", "lrelease", "qmllint", "qmlformat",
                  "qmlls", "qmlcachegen", "uic", "rcc", "svgtoqml", "pyside6-", "qtpaths", "balsam", "qsb", "qml")


def prune_qt(site: Path, keys: list[str], say=_say) -> int:
    """PySide6_Essentials (~280 MB) → only what the GUI uses (~120 MB)."""
    base = site / "PySide6"
    if "docs" in keys:           # Qt WebEngine (PDF manuals) needs Qml, Quick, translations and icudtl.dat → do not prune
        say(T("Qt: modul docs (WebEngine) – bez prořezání", "Qt: module docs (WebEngine) – not slimmed"))
        return 0
    if not base.is_dir():
        return 0
    freed = 0
    for root in (base, base / "Qt"):
        if not root.is_dir():
            continue
        for rel in _QT_DROP_DIRS:
            p = root / rel
            if p.exists() and not (rel == "lib" and root.name == "Qt" and any(root.joinpath(rel).glob("libQt6Core*"))):
                freed += _rm(p)
        for f in list(root.iterdir()) + (list((root / "lib").iterdir()) if (root / "lib").is_dir() else []):
            n = f.name
            if f.is_file() and any(n.startswith(d) for d in _QT_DROP_FILES) and not n.startswith(("Qt6Core", "Qt6Gui", "Qt6Widgets")):
                freed += _rm(f)
        if (root / "resources").is_dir():
            freed += _rm(root / "resources" / "icudtl.dat")
    # Linux: the shared libraries are in Qt/lib (libQt6*.so.6) – the QML libraries are removed above by name
    say(T(f"Qt: uvolněno {freed / 1e6:.0f} MB", f"Qt: freed {freed / 1e6:.0f} MB"))
    return freed


def prune_conda_env(env: Path, say=_say) -> int:
    """The hypre environment from conda-forge: without headers, documentation, package caches and unused tools."""
    freed = 0
    for rel in ("include", "share", "man", "conda-meta/history", "pkgs", "Library/include", "Library/share", "Library/lib",
                "Library/man", "Library/doc", "Scripts", "etc", "compiler_compat", "Lib/site-packages", "lib/cmake",
                "lib/pkgconfig", "lib/python3.11", "lib/python3.12", "lib/python3.13"):
        p = env / rel
        if p.exists():
            freed += _rm(p)
    for f in list(env.rglob("*.lib")) + list(env.rglob("*.a")) + list(env.rglob("*.pdb")):
        freed += _rm(f)
    say(T(f"hypre prostředí: uvolněno {freed / 1e6:.0f} MB, zbývá {dir_size(env) / 1e6:.0f} MB",
          f"hypre environment: freed {freed / 1e6:.0f} MB, {dir_size(env) / 1e6:.0f} MB left"))
    return freed


# ------------------------------------------------------------------ everything
def slim_all(venv: Path, site: Path, keys: list[str], say=_say) -> None:
    py = venv / ("Scripts/python.exe" if IS_WIN else "bin/python")
    before = dir_size(venv)
    prune_python(venv, say)
    prune_vtk(py, site, say)
    prune_gmsh(venv, site, say)
    strip_binaries(site, venv, say)
    if "gui" in keys:
        prune_qt(site, keys, say)
    prune_generic(site, keys, say)
    if not verify_ocp(py):
        say(T("!! OCP po prořezání nejde importovat – spusťte fea_setup.py install --force --no-slim",
              "!! OCP cannot be imported after slimming – run fea_setup.py install --force --no-slim"))
    say(f".venv: {before / 1e6:.0f} → {dir_size(venv) / 1e6:.0f} MB")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    v = root / ".venv"
    if "--report" in sys.argv:
        sp = next(v.glob("Lib/site-packages" if IS_WIN else "lib/python3*/site-packages"))
        rows = sorted(((dir_size(d), d.name) for d in sp.iterdir() if d.is_dir()), reverse=True)[:25]
        for n, name in rows:
            print(f"{n / 1e6:8.1f} MB  {name}")
        print(f"{dir_size(v) / 1e6:8.1f} MB  " + T("celé .venv", "whole .venv"))
