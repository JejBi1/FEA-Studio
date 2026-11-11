"""The green bar of the progress of the calculation: the values only grow, it ends at 100 % "Done", snapshots during the calculation.

  python tests/gui/gui_progress_smoke.py OUT_DIR [project.fea]

Default project = example 1 (the mesh is created again, 2 load states). It prints the time course of the bar.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PROJECT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "examples" / "example1_bracket.fea"
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)


def main():
    from fea_gui.main import MainWindow
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1700, 1000)
    w.show()
    w.open_file(PROJECT)
    w.mesh = None                                      # the mesh again (the Gmsh phase)
    w.results = None
    w.set_stage(w.STAGE["Solve"])
    app.processEvents()
    bar, sbar = w.solve_progress, w.status_progress
    seen, fails = [], []
    t0 = time.time()
    shots = {5: False, 30: False, 60: False, 85: False}

    def sample():
        v = bar.value() / 10
        seen.append(v)
        if len(seen) % 20 == 1:
            print(f"{time.time() - t0:6.1f} s  {bar.text()}", flush=True)
        for p in shots:
            if not shots[p] and v >= p:
                shots[p] = True
                w.grab().save(str(OUT / f"pruh_{p:02d}.png"))
    timer = QTimer()
    timer.timeout.connect(sample)
    timer.start(250)
    w.run_solve()
    if not (bar.isVisible() and sbar.isVisible()):
        fails.append("pruh se při spuštění výpočtu neukázal")
    while w._thread is not None:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()
    timer.stop()
    sample()
    print(f"{time.time() - t0:6.1f} s  {bar.text()}")
    w.set_stage(w.STAGE["Solve"])
    app.processEvents()
    w.grab().save(str(OUT / "pruh_hotovo.png"))
    if any(b < a for a, b in zip(seen, seen[1:])):
        fails.append("pruh couvl")
    if bar.value() != 1000 or not any(x in bar.text() for x in ("Done", "Hotovo")) or bar.property("state") != "done":
        fails.append(f"konec: {bar.value()} {bar.text()!r} {bar.property('state')}")
    if w.results is None:
        fails.append("výpočet nevrátil výsledky")
    w.dirty = False
    w.close()
    for f in fails:
        print("CHYBA:", f)
    print("VÝSLEDEK:", "OK" if not fails else f"{len(fails)} chyb")
    sys.exit(1 if fails else 0)


main()
