"""RAM of the idle program: the MCP server and the GUI start without the heavy libraries that they need only for results.

The solver (ccx, PARDISO, hypre) gets the memory the program does not hold (see CLAUDE.md, "RAM of the program").
Each check runs in a fresh interpreter, so the import set is not influenced by other tests.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# libraries that cost > 20 MB each and are needed only for pictures, plots, optional modules and results
NOT_AT_IDLE = ("numba", "llvmlite", "scipy", "matplotlib", "pandas", "gmsh", "ngsolve", "mpi4py", "exudyn", "docx",
               "pyvista", "vtkmodules", "win32com", "pythoncom")


def _loaded_after(code: str) -> list[str]:
    code = code + (
        "\nimport sys\n"
        f"heavy = [m for m in {NOT_AT_IDLE!r} if any(k == m or k.startswith(m + '.') for k in sys.modules)]\n"
        "print('HEAVY', ' '.join(heavy))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-800:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("HEAVY")][-1]
    return line.split()[1:]


def test_mcp_server_idle_does_not_import_numba_or_scipy():
    """build_server (all tools registered) without the solver libraries; the raster kernels compile on the first picture."""
    heavy = _loaded_after("import fea_core.mcp.server as srv\nsrv.build_server()\n")
    assert heavy == [], f"MCP server imports at idle: {heavy}"


def test_raster_kernels_compile_on_first_call():
    """The rasteriser module imports nothing heavy; the kernel itself is compiled (and numba imported) on the first call."""
    code = ("import sys, numpy as np\n"
            "import fea_core.mcp.raster as r\n"
            "assert 'numba' not in sys.modules\n"
            "z = np.full((4, 4), -1e30); c = np.zeros((4, 4, 3)); i = np.full((4, 4), -1, dtype=np.int64)\n"
            "px = np.array([[0.0, 3.0, 0.0]]); py = np.array([[0.0, 0.0, 3.0]]); pz = np.ones((1, 3))\n"
            "r.raster_triangles(px, py, pz, np.zeros((1, 3)), np.ones((1, 3)), np.ones(1), np.zeros(1, dtype=np.int64),\n"
            "                   1, np.zeros(2), np.zeros((1, 3)), z, c, i)\n"
            "assert 'numba' in sys.modules and i[0, 0] == 0 and i[3, 3] == -1, i\n"
            "print('OK')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-800:]


def test_release_unused_switch(monkeypatch):
    """FEA_RELEASE_MEMORY=0 switches the release off (A/B measurements); nothing is released then."""
    from fea_core import memory
    monkeypatch.setenv("FEA_RELEASE_MEMORY", "0")
    assert memory.release_unused() is False


def test_release_unused_gives_working_set_back(monkeypatch):
    """Windows: the touched pages leave the working set; the data is still there when it is read again."""
    import sys
    import numpy as np
    import psutil
    import pytest
    from fea_core import memory
    if sys.platform != "win32":
        pytest.skip("EmptyWorkingSet is Windows only")
    monkeypatch.delenv("FEA_RELEASE_MEMORY", raising=False)
    a = np.ones(100 * 2**20 // 8)                  # 100 MB, touched
    rss0 = psutil.Process().memory_info().rss
    assert memory.release_unused() is True
    rss1 = psutil.Process().memory_info().rss
    assert rss0 - rss1 > 50 * 2**20, (rss0 / 2**20, rss1 / 2**20)
    assert a.sum() == a.size                       # the values come back on access
    del a


def test_gui_import_does_not_import_solver_libraries():
    """The GUI module is imported without numba, scipy, matplotlib, the MCP SDK and the REST stack (they come on use)."""
    heavy = _loaded_after("import fea_gui.main\n")
    assert heavy == [], f"GUI imports at start: {heavy}"
    r = subprocess.run([sys.executable, "-c", "import fea_gui.main, sys; print('MCP' if 'mcp' in sys.modules else 'no-mcp')"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert "no-mcp" in r.stdout, r.stdout + r.stderr[-400:]
