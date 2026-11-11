"""Dropping a file onto the window and the metadata from the CAD next to the STEP (snapshots for a manual check).

  python tests/gui/gui_cad_drop_smoke.py OUT_DIR [--ole]

Runs WITH A WINDOW (OCCT needs OpenGL). The model examples/user_tests/T5_joint_M10.step is copied to OUT_DIR and next to
it <the same name>.json is written in the format of the SolidWorks macro (occurrences "Deska A-1", UTF-8 with BOM as from VBA).
Steps:
  1. The OCCT 3D window is a native window without its own OLE target – Windows passes the drop to the nearest ancestor with a target,
     which must be the main window (the property "OleDropTargetInterface" on the HWND).
  2. Dropping a STEP over the 3D window: the events go to the Qt window of the main window (as the OLE target sends them), Qt sends them
     by position → dragEnterEvent / dropEvent of the main window → open_file. Check of the parts, of the materials from the metadata,
     of the messages (source "import from CAD") and of the log.
  3. Dropping onto a text field (QLineEdit) of the Materials panel, unsaved changes → the question "Save changes?" (replaced),
     a .txt file is refused, dropping a project .fea opens it (the metadata are not read for a project).
  --ole: additionally a real drag over OLE (DoDragDrop, the mouse via SendInput) over the 3D window – it moves the cursor
     of the mouse, the window comes to the top for a while; it runs only when the window is visible at the points of the drag.
"""
import json
import shutil
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(ARGS[0] if ARGS else "gui_cad_drop_out").resolve()
OUT.mkdir(parents=True, exist_ok=True)
OLE = "--ole" in sys.argv
FAIL = []


def check(cond, text):
    print(("OK   " if cond else "CHYBA ") + text)
    if not cond:
        FAIL.append(text)


def make_files():
    """STEP from examples/user_tests + metadata as from a SolidWorks macro (UTF-8 with BOM)."""
    step = OUT / "Spoj_M10_20261015_101500.step"
    shutil.copyfile(ROOT / "examples" / "user_tests" / "T5_joint_M10.step", step)
    meta = {"source": "SolidWorks", "document": "Spoj M10", "units": "mm",
            "parts": [{"name": "Deska A-1", "names": ["Deska A"], "material": "S355"},
                      {"name": "Deska B-1", "material": "AISI 304"},                       # matching only without "-1"
                      {"name": "ISO 4017 - M10 x 40 - 8.8-1", "names": ["ISO 4017 - M10 x 40 - 8.8"],
                       "material": "Plain Carbon Steel"},                                  # not in the library
                      {"name": "ISO 7089 - 10 - 200 HV-1", "material": "AISI 316L"},
                      {"name": "ISO 7089 - 10 - 200 HV-2", "material": "AISI 316L"},
                      {"name": "ISO 4032 - M10 - 8-1", "material": ""},                    # without a material
                      {"name": "Skryte teleso-1", "material": "S235"}]}                    # not in the STEP
    step.with_suffix(".json").write_bytes(b"\xef\xbb\xbf" + json.dumps(meta, ensure_ascii=False, indent=1)
                                          .encode("utf-8"))
    return step


def mime_for(*paths):
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return m


def qt_drop(w, widget, paths):
    """A drag as delivered by the OLE target: events into the Qt window (QWidgetWindow) of the main window at the point above `widget`;
    Qt finds the target widget by the position. Returns (dragEnter accepted, drop accepted).
    """
    win = w.windowHandle()
    pos = QPointF(widget.mapTo(w, widget.rect().center()))
    mime = mime_for(*paths)
    act = Qt.CopyAction | Qt.MoveAction | Qt.LinkAction
    ev = QDragEnterEvent(pos.toPoint(), act, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(win, ev)
    entered = ev.isAccepted()
    ev = QDragMoveEvent(pos.toPoint(), act, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(win, ev)
    ev = QDropEvent(pos, act, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(win, ev)
    dropped = ev.isAccepted()
    for _ in range(5):                                  # open_file only after returning from the drop (QTimer 0)
        QApplication.processEvents()
    return entered, dropped


def ole_target_hwnd(hwnd: int) -> int | None:
    """The HWND that receives the drag over `hwnd`: the first ancestor with a registered OLE target (as DoDragDrop)."""
    import ctypes
    u = ctypes.windll.user32
    u.GetPropW.restype = ctypes.c_void_p
    u.GetPropW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    u.GetParent.restype = ctypes.c_void_p
    u.GetParent.argtypes = [ctypes.c_void_p]
    h = hwnd
    while h:
        if u.GetPropW(h, "OleDropTargetInterface"):
            return int(h)
        h = u.GetParent(h)
    return None


def main():
    import fea_gui.main as gm
    from fea_gui.main import CAD_SOURCE, MainWindow
    shown = []
    gm.msg = lambda parent, title, text, icon=None: shown.append(text) or print(f"[{title}] {text[:200]}")
    asked = []

    def question(*a, **k):
        asked.append(a[2] if len(a) > 2 else "")
        return QMessageBox.Discard
    QMessageBox.question = staticmethod(question)
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.resize(1500, 950)
    w.show()
    for _ in range(10):
        app.processEvents()
    step = make_files()

    def shot(name):
        from PySide6.QtGui import QImage, QPainter
        app.processEvents()
        w.view.redraw()
        app.processEvents()
        img = w.grab().toImage()
        tmp = OUT / "_view.png"
        w.view.screenshot(str(tmp))
        v = QImage(str(tmp))
        p = QPainter(img)
        p.drawImage(w.view.mapTo(w, w.view.rect().topLeft()), v.scaled(w.view.size()))
        p.end()
        tmp.unlink(missing_ok=True)
        img.save(str(OUT / f"{name}.png"))

    # 1. the OLE target above the native 3D window
    check(w.acceptDrops(), "hlavní okno přijímá přetažení (setAcceptDrops)")
    if sys.platform == "win32":
        view_hwnd, main_hwnd = int(w.view.winId()), int(w.winId())
        check(view_hwnd != main_hwnd, "3D okno je nativní okno (vlastní HWND)")
        tgt = ole_target_hwnd(view_hwnd)
        check(tgt == main_hwnd, f"přetažení nad 3D oknem jde na cíl OLE hlavního okna (HWND {tgt} = {main_hwnd})")

    # 2. drop of a STEP over the 3D window → opening + metadata from the CAD
    entered, dropped = qt_drop(w, w.view, [step])
    check(entered and dropped, f"STEP nad 3D oknem: dragEnter {entered}, drop {dropped}")
    check(w.step_path == step and len(w.parts) == 6, f"otevřen {w.step_path}, dílů {len(w.parts)}")
    check(w.STAGES[w.stage] == "Materials", f"krok po otevření: {w.STAGES[w.stage]}")
    pm = w.study.part_materials
    want = {"Deska A": "Ocel S355", "Deska B": "Nerez 1.4301", "ISO 7089 - 10 - 200 HV": "Nerez 1.4404",
            "ISO 7089 - 10 - 200 HV (2)": "Nerez 1.4404"}
    check(pm == want, f"materiály z metadat: {pm}")
    check(w.study.material_of("ISO 4017 - M10 x 40 - 8.8").name == "Ocel S235", "šroub (materiál mimo knihovnu) "
          "má výchozí Ocel S235")
    msgs = w.messages.messages(source=CAD_SOURCE)
    texts = [m["text"] for m in msgs]
    print("  zprávy:", *texts, sep="\n    ")
    check(any(m["level"] == "info" and "6 of 6 parts matched" in m["text"] and "assigned to 4 parts" in m["text"]
              for m in msgs), "zpráva se souhrnem (spárováno 6 z 6, přiřazen 4)")
    check(any(m["level"] == "warning" and "Plain Carbon Steel" in m["text"] and "Ocel S235" in m["text"]
              for m in msgs), "varování: materiál z CAD není v knihovně, díl má výchozí materiál")
    check(any("ISO 4032 - M10 - 8" in t and "without a material" in t for t in texts), "varování: díl bez materiálu v CAD")
    check(any("Skryte teleso-1" in t for t in texts), "informace: záznam metadat bez dílu")
    check(all(m["stage"] == "Materials" for m in msgs), "dvojklik na zprávu → krok Materiály")
    log = w.log.toPlainText()
    check(any(x in log for x in ("CAD metadata (SolidWorks, Spoj M10)", "Metadata z CAD (SolidWorks, Spoj M10)"))
          and any(x in log for x in ("Material from CAD: Deska B", "Materiál z CAD: Deska B")), "shrnutí v logu")
    check(not w.dirty and not asked, "po otevření bez neuložených změn, bez dotazu")
    check(not shown, f"žádné modální okno (geometrie v pořádku): {shown}")
    shot("d1_step_s_metadaty")

    # 3a. unsaved changes + drop onto the text field → a question, then opening
    w.changed()
    check(w.dirty, "změna studie → neuložené změny")
    w.set_stage(w.STAGE["Materials"])
    entered, dropped = qt_drop(w, w.m_name, [step])
    check(entered and dropped, f"STEP nad textovým polem panelu Materiály: dragEnter {entered}, drop {dropped}")
    check(len(asked) == 1 and any(x in asked[0] for x in ("Save", "Uložit")), f"dotaz na neuložené změny: {asked}")
    check(not w.dirty and w.study.part_materials == want, "znovu otevřeno se stejnými materiály")
    # 3b. another file is refused
    txt = OUT / "poznamka.txt"
    txt.write_text("x", encoding="utf-8")
    entered, dropped = qt_drop(w, w.view, [txt])
    check(not entered and not dropped, f".txt se odmítne (dragEnter {entered}, drop {dropped})")
    # 3b'. Shift during the drop suggests a move – the program only reads the file, it accepts it as a copy
    # (otherwise Explorer would delete the original STEP after the "move")
    win = w.windowHandle()
    pos = QPointF(w.view.mapTo(w, w.view.rect().center())).toPoint()
    mime = mime_for(step)                               # keep a reference – the event points to QMimeData
    ev = QDragEnterEvent(pos, Qt.CopyAction | Qt.MoveAction, mime, Qt.LeftButton, Qt.ShiftModifier)
    proposed = ev.proposedAction()
    QApplication.sendEvent(win, ev)
    check(ev.isAccepted() and ev.dropAction() == Qt.CopyAction,
          f"Shift (navržen {proposed}) → přijato jako kopírování: {ev.dropAction()}")
    QApplication.sendEvent(win, QDragLeaveEvent())
    # 3c. a project .fea (the metadata are not read for a project – the materials are in the project)
    fea = OUT / "spoj_m10.fea"
    check(w._save(fea), "uložení projektu")
    w.study.part_materials.clear()
    w.changed()
    asked.clear()
    entered, dropped = qt_drop(w, w.tree, [fea, step])        # several files → only the first
    check(entered and dropped and w.project_path == fea, f"projekt .fea přetažením: {w.project_path}")
    check(len(asked) == 1, "dotaz na neuložené změny před otevřením projektu")
    check(w.study.part_materials == want, "materiály z projektu")
    check(not w.messages.messages(source=CAD_SOURCE), "u projektu žádné zprávy importu z CAD")
    check(any(x in w.log.toPlainText() for x in ("only the first one is opened", "otevře se jen první")), "víc souborů → log „otevře se jen první“")
    shot("d2_projekt")

    if OLE and sys.platform == "win32":
        ole_test(app, w, step)

    w.dirty = False
    w.close()
    print("\nVÝSLEDEK:", "OK" if not FAIL else f"{len(FAIL)} chyb")
    for f in FAIL:
        print("  -", f)
    sys.exit(1 if FAIL else 0)


def ole_test(app, w, step):
    """A real OLE drag (DoDragDrop) from a window over the 3D window: the mouse button via mouse_event, the movement SetCursorPos
    in a thread (the Qt loop during QDrag.exec stands on Windows). Only when both points are visible in the window of the program.
    """
    import ctypes
    from ctypes import wintypes
    from PySide6.QtGui import QDrag
    u = ctypes.windll.user32
    u.WindowFromPoint.restype = ctypes.c_void_p
    u.WindowFromPoint.argtypes = [wintypes.POINT]
    u.GetAncestor.restype = ctypes.c_void_p
    u.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    u.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
    u.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_uint]
    main_hwnd, view_hwnd = int(w.winId()), int(w.view.winId())
    LEFTDOWN, LEFTUP = 0x0002, 0x0004
    w.showNormal()                                     # the whole window on the screen above the Windows taskbar
    w.setGeometry(w.screen().availableGeometry().adjusted(60, 60, -60, -60))
    for _ in range(10):
        app.processEvents()
        time.sleep(0.02)

    def rect(h):
        r = wintypes.RECT()
        u.GetWindowRect(h, ctypes.byref(r))
        return r

    u.ClientToScreen.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.POINT)]
    rv = rect(view_hwnd)
    dst = ((rv.left + rv.right) // 2, (rv.top + rv.bottom) // 2)
    origin = wintypes.POINT(0, 0)                      # the upper left corner of the client area of the main window [px]
    u.ClientToScreen(main_hwnd, ctypes.byref(origin))
    dpr = w.devicePixelRatioF()
    sb = w.statusBar()                                 # start of the drag: a label in the status bar (a click does nothing)
    c = sb.mapTo(w, QPoint(60, sb.height() // 2))
    src = (origin.x + int(c.x() * dpr), origin.y + int(c.y() * dpr))

    def ours(pt):
        h = u.WindowFromPoint(wintypes.POINT(*pt))
        return bool(h) and int(u.GetAncestor(h, 2)) == main_hwnd          # GA_ROOT
    old = wintypes.POINT()
    u.GetCursorPos(ctypes.byref(old))
    u.SetWindowPos(main_hwnd, ctypes.c_void_p(-1), 0, 0, 0, 0, 0x0001 | 0x0002)    # HWND_TOPMOST, without changing the position
    app.processEvents()
    from PySide6.QtCore import QEvent, QObject

    class DropSpy(QObject):                            # records where Qt delivered the drop (main window, position)
        def __init__(self):
            super().__init__()
            self.hits = []

        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Drop:
                self.hits.append((obj, ev.position().toPoint()))
            return False
    spy = DropSpy()
    try:
        if not (ours(src) and ours(dst)):
            print(f"  (OLE: okno programu není v bodech přetažení {src} → {dst} vidět – test přeskočen)")
            return
        w.installEventFilter(spy)
        w.dirty = False
        done = threading.Event()

        def mover():
            time.sleep(0.4)
            for k in range(1, 11):                     # movement over the 3D window in steps (OLE processes WM_MOUSEMOVE)
                u.SetCursorPos(src[0] + (dst[0] - src[0]) * k // 10, src[1] + (dst[1] - src[1]) * k // 10)
                time.sleep(0.05)
            time.sleep(0.3)
            u.mouse_event(LEFTUP, 0, 0, 0, 0)
            if not done.wait(3):                       # a safety net: the drag does not end → Esc (cancel)
                u.keybd_event(0x1B, 0, 0, 0)
                u.keybd_event(0x1B, 0, 2, 0)
        u.SetCursorPos(*src)
        u.mouse_event(LEFTDOWN, 0, 0, 0, 0)
        th = threading.Thread(target=mover, daemon=True)
        th.start()
        drag = QDrag(sb)
        drag.setMimeData(mime_for(step))
        t0 = time.time()
        result = drag.exec(Qt.CopyAction)
        done.set()
        th.join(5)
        for _ in range(10):
            app.processEvents()
        hits = spy.hits
        print(f"  OLE: výsledek {result}, za {time.time() - t0:.1f} s, drop {[(type(o).__name__, p) for o, p in hits]}")
        check(any(o is w for o, _ in hits), "skutečné přetažení OLE nad 3D okno OCCT doručeno hlavnímu oknu")
        if hits:
            child = w.childAt(hits[0][1])
            check(child is w.view or (child is not None and w.view.isAncestorOf(child)),
                  f"bod přetažení leží nad 3D oknem ({type(child).__name__ if child else None})")
        check(w.step_path == step and len(w.parts) == 6, "STEP otevřen skutečným přetažením")
    finally:
        w.removeEventFilter(spy)
        u.SetWindowPos(main_hwnd, ctypes.c_void_p(-2), 0, 0, 0, 0, 0x0001 | 0x0002)   # HWND_NOTOPMOST
        u.SetCursorPos(old.x, old.y)


if __name__ == "__main__":
    main()
