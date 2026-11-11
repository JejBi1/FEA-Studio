"""A pass through both example guides in the GUI with snapshots + the help (search).

  python tests/gui/gui_tour_smoke.py OUT_DIR

The examples from examples/ (without results: the results steps only ask for a calculation; with results – `python -m examples.create
--solve` – the results are gone through too). A question about the results from the past is answered No (always a clean example).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PySide6.QtWidgets import QApplication, QMessageBox

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)   # always a clean example


def main():
    from fea_gui.main import MainWindow
    from fea_gui.help import show_help
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1700, 1000)
    w.show()
    assert hasattr(w, "tour_dock"), "main.py nevolá install_help"
    w.run_solve = lambda: msgs.append("Průvodce: výpočet by se spustil (v testu vypnuto)")
    msgs = []
    orig = w.log_msg
    w.log_msg = lambda t: (msgs.append(t), orig(t))

    def shot(name):
        for _ in range(3):
            app.processEvents()
        w.view.redraw()
        app.processEvents()
        from PySide6.QtGui import QImage, QPainter
        img = w.grab().toImage()
        tmp = OUT / "_view.png"
        w.view.screenshot(str(tmp))
        v = QImage(str(tmp))
        pos = w.view.mapTo(w, w.view.rect().topLeft())
        p = QPainter(img); p.drawImage(pos, v.scaled(w.view.size())); p.end()
        tmp.unlink(missing_ok=True)
        img.save(str(OUT / f"{name}.png"))

    for t, name in enumerate(w.tour_dock.tours, 1):
        w.help_action(f"pruvodce{t}")
        dock = w.tour_dock
        for k in range(len(dock.steps)):
            dock.go(k)
            shot(f"t{t}_{k:02d}")
            print(t, k, dock.steps[k]["title"], "| ukázka má výsledky:", w.results is not None, flush=True)
    hw = show_help(w, "tisk", on_action=w.help_action)
    app.processEvents()
    hw.grab().save(str(OUT / "help_tisk.png"))
    hw.search_edit.setText("creep")
    app.processEvents()
    hw.grab().save(str(OUT / "help_search.png"))
    print("LOG průvodce:", [m for m in msgs if "Guide" in m])
    w.dirty = False
    w.close()


main()
