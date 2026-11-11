"""Test models for manual testing in the GUI (the same geometry as the verification cases verification/, known results).
T1–T7 statics, heat, joints, printing; T8 dynamics (Y2), T9 / T9b contact (K1, K4), T10 mechanism (MB1).

    python -m examples.user_tests            → examples/user_tests/*.step

Procedure and expected values: help → "Testing" (fea_gui/help/testovani.html), the Word manual.
"""
from __future__ import annotations

import sys
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut

from fea_core.geometry import Part, write_step

from .geometry import OCEL, SVAR, SROUB, TISK, box, cyl, klikovy_mechanismus, solid

OUT = Path(__file__).resolve().parent / "testy"


def _colored(parts, color):
    for p in parts:
        p.color = color
    return parts


def modely() -> dict[str, list[Part]]:
    from verification.joint_models import lap_joint, t_joint_fillet
    ring = BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), 40, 10), cyl((0, 0, -1), (0, 0, 1), 20, 12)).Shape()
    ctvrt = solid(BRepAlgoAPI_Common(ring, box(0, 0, -1, 41, 41, 12)).Shape())
    deska = solid(BRepAlgoAPI_Cut(box(0, 0, 0, 150, 50, 2.5), cyl((0, 0, -1), (0, 0, 1), 5, 4.5)).Shape())
    spoj = lap_joint()
    for p in spoj:
        p.color = SROUB if p.name.startswith("ISO") else OCEL
    # T9b press fit: an eighth of the shaft Ø20 / hub Ø20/40 (length 2 × 10, plane z = 0 = middle of the length)
    q = box(0, 0, 0, 30, 30, 10)
    hridel = solid(BRepAlgoAPI_Common(cyl((0, 0, 0), (0, 0, 1), 10, 10), q).Shape())
    naboj = solid(BRepAlgoAPI_Common(BRepAlgoAPI_Cut(cyl((0, 0, 0), (0, 0, 1), 20, 10),
                                                     cyl((0, 0, 0), (0, 0, 1), 10, 10)).Shape(), q).Shape())
    tspoj = t_joint_fillet()
    for p in tspoj:
        p.color = SVAR if p.name.startswith("Svar") else OCEL
    return {
        "T1_cantilever": _colored([Part("Konzola", solid(box(0, 0, 0, 200, 20, 10)))], OCEL),
        "T2_plate_with_hole_eighth": _colored([Part("Deska", deska)], OCEL),
        "T3_pipe_quarter": _colored([Part("Trubka", ctvrt)], OCEL),
        "T4_shaft": _colored([Part("Hridel", solid(cyl((0, 0, 0), (1, 0, 0), 10, 200)))], OCEL),
        "T5_joint_M10": spoj,
        "T6_t_joint_welds_a5": tspoj,
        "T7_printed_bars": _colored([Part("Tyc nastojato", solid(box(0, 0, 0, 100, 10, 10))),
                                  Part("Tyc nalezato", solid(box(0, 20, 0, 100, 10, 10)))], TISK),
        # F8: dynamika (Y2), kontakt (K1, K4), mechanismus (MB1)
        "T8_cantilever_10x10x100": _colored([Part("Konzola", solid(box(0, 0, 0, 100, 10, 10)))], OCEL),
        "T9_contact_blocks": _colored([Part("Kvadr A", solid(box(0, 0, 0, 20, 20, 10))),
                                       Part("Kvadr B", solid(box(0, 0, 10, 20, 20, 10)))], OCEL),
        "T9b_press_fit_eighth": _colored([Part("Hridel", hridel), Part("Naboj", naboj)], OCEL),
        "T10_crank_r20_l80": klikovy_mechanismus(20.0, 80.0),
    }


def main(argv=None) -> int:
    OUT.mkdir(exist_ok=True)
    for name, parts in modely().items():
        write_step(parts, OUT / f"{name}.step")
        print(f"{name}.step ({len(parts)} parts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
