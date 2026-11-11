"""Scaling of the whole chain with the size of the model: times of the phases + peak memory of the processes.

  python -m verification.bench_scaling [element size ...] [--solver PARDISO]

Model: plate 200×60×8 with a hole Ø20 + foot 20×60×30 (2 parts), clamping of the foot, force at the end.
The memory watchdog ends the calculation when the free RAM drops below MIN_FREE_GB (the computer must not be overloaded).
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import psutil

from fea_core import materials
from fea_core.geometry import Part, read_step, write_step
from fea_core.selectors import find_faces
from fea_core.study import Study, Fixed, Force, LoadCase

OUT = Path(__file__).resolve().parent / "out_bench"
MIN_FREE_GB = 2.0


def model(wd: Path) -> Path:
    from verification.cases import box, cyl, solid
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    step = wd / "deska.step"
    if not step.exists():
        plate = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 200, 60, 8), cyl((100, 30, -1), (0, 0, 1), 10, 10)).Shape())
        write_step([Part("Deska", plate), Part("Patka", box(0, 0, 8, 20, 60, 30))], step)
    return step


class Watch:
    """Peak RSS of the children (ccx, gmsh worker) and the watch of the free memory."""

    def __init__(self):
        self.peak: dict[str, float] = {}
        self.min_free = 1e9
        self.killed = False
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        me = psutil.Process()
        while not self._stop.wait(0.25):
            free = psutil.virtual_memory().available / 2**30
            self.min_free = min(self.min_free, free)
            for ch in me.children(recursive=True):
                try:
                    name = ch.name().lower()
                    key = "ccx" if "ccx" in name else ("gmsh" if "python" in name else name)
                    self.peak[key] = max(self.peak.get(key, 0.0), ch.memory_info().rss / 2**30)
                    if free < MIN_FREE_GB:
                        ch.kill()
                        self.killed = True
                except psutil.Error:
                    pass
            self.peak["python"] = max(self.peak.get("python", 0.0), me.memory_info().rss / 2**30)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def run(sizes: list[float], solver: str, backend: str = "ccx") -> list[dict]:
    from fea_core.solve import run_study
    OUT.mkdir(parents=True, exist_ok=True)
    step = model(OUT)
    parts = read_step(step)
    rows = []
    for h in sizes:
        st = Study(f"bench h={h}", str(step))
        st.add_material(materials.default())
        st.add_selection("fix", find_faces(parts, part="Patka", x=0))
        st.add_selection("F", find_faces(parts, part="Deska", x=200))
        st.constraints.append(Fixed("fix"))
        st.load_cases.append(LoadCase("LC", [Force("F", [0, 0, -500.0])]))
        st.mesh.size = h
        st.solver.solver = solver
        st.solver.backend = backend
        marks = []
        t0 = time.time()

        def log(s):
            marks.append((time.time() - t0, s))
        tag = solver if backend == "ccx" else backend
        wd = OUT / f"h{h}_{tag}"
        err = None
        with Watch() as w:
            try:
                res = run_study(st, wd, on_log=log)
            except Exception as e:  # noqa: BLE001
                err = str(e)[:300]
                res = None
        total = time.time() - t0
        ph = {}
        names = {"Síťuji": "sit", "Počítám": "ccx", "Sestavuji": "sestaveni", "Řeším": "reseni",
                 "Počítám napětí": "napeti", "Hotovo": "konec"}
        for (ta, sa), (tb, _) in zip(marks, marks[1:] + [(total, "")]):
            for k, v in names.items():
                if sa.startswith(k):
                    ph[v] = round(tb - ta, 2)
        row = {"h": h, "solver": tag, "total_s": round(total, 1), "phases_s": ph,
               "peak_gb": {k: round(v, 2) for k, v in w.peak.items()}, "min_free_gb": round(w.min_free, 1),
               "killed": w.killed, "error": err}
        if res is not None:
            n = len(res.mesh.nodes)
            row.update({"nodes": n, "elements": len(res.mesh.tets), "dof": 3 * n,
                        "max_u": float(res.step(0).umag.max()), "max_vm": float(res.step(0).mises.max())})
            hi = res.info.get("hypre")
            if hi:
                row["hypre"] = {"iter": hi["cases"][0]["iterations"], "t_setup": round(hi["t_setup"], 1),
                                "t_solve": round(hi["cases"][0]["t_solve"], 1), "peak_gb": round(hi["peak_gb_total"], 2),
                                "t_assemble": round(hi["assemble"]["t_total"], 1)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (OUT / f"bench_{tag}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        if err:
            break
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    solver, backend = "PARDISO", "ccx"
    if "--backend" in args:
        i = args.index("--backend")
        backend = args[i + 1]
        del args[i:i + 2]
    if "--solver" in args:
        i = args.index("--solver")
        solver = args[i + 1]
        del args[i:i + 2]
    sizes = [float(a) for a in args] or [2.5, 1.6, 1.25, 1.0]
    run(sizes, solver, backend)
