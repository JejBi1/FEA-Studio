"""Example 6 – a steel hall frame of members (without STEP geometry): two frames IPE 300 / HEA 260 with struts, purlins
UPE 160, a bracing of tubes CHS 76.1×3.6 (truss members), snow, wind, self-weight; natural frequencies and the assessment
EN 1993-1-1.  python -m examples.frames [--solve] [--out FOLDER]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fea_core.beams.model import FrameMember, FrameSupport, SectionDef, FrameNodeLoad, FrameMemberLoad
from fea_core.study import Study, Material, LoadCase, Gravity

HERE = Path(__file__).resolve().parent
L, H, B = 12000.0, 5000.0, 6000.0           # span, height of the column, distance of the frames [mm]
RIDGE = 1000.0                              # rise of the ridge


def build() -> Study:
    st = Study("Ukázka 6 – rám haly z prutů", "")
    st.add_material(Material("Ocel S355", 210000.0, 0.3, 7.85e-9, Re=355.0, Rm=490.0, source="EN 10025-2"), default=True)
    fr = st.frame
    fr.sections["IPE 300"] = SectionDef("IPE 300", library="IPE 300")
    fr.sections["HEA 260"] = SectionDef("HEA 260", library="HEA 260")
    fr.sections["UPE 160"] = SectionDef("UPE 160", library="UPE 160")
    fr.sections["CHS 76.1x3.6"] = SectionDef("CHS 76.1x3.6", library="CHS 76.1x3.6")
    for k, y in enumerate((0.0, B)):
        s = "AB"[k]
        fr.add_node(f"{s}1", [0, y, 0])
        fr.add_node(f"{s}2", [0, y, H])
        fr.add_node(f"{s}3", [L / 2, y, H + RIDGE])
        fr.add_node(f"{s}4", [L, y, H])
        fr.add_node(f"{s}5", [L, y, 0])
        fr.members[f"{s} sloup L"] = FrameMember(f"{s} sloup L", f"{s}1", f"{s}2", "HEA 260", up=[1, 0, 0])
        fr.members[f"{s} sloup P"] = FrameMember(f"{s} sloup P", f"{s}5", f"{s}4", "HEA 260", up=[1, 0, 0])
        fr.members[f"{s} příčel L"] = FrameMember(f"{s} příčel L", f"{s}2", f"{s}3", "IPE 300", up=[0, 0, 1])
        fr.members[f"{s} příčel P"] = FrameMember(f"{s} příčel P", f"{s}3", f"{s}4", "IPE 300", up=[0, 0, 1])
        fr.supports += [FrameSupport(f"{s}1"), FrameSupport(f"{s}5")]   # clamped feet
    for n in ("2", "3", "4"):                      # purlins (pinned to the frames)
        fr.members[f"vaznice {n}"] = FrameMember(f"vaznice {n}", f"A{n}", f"B{n}", "UPE 160", up=[0, 0, 1],
                                                releases_start=[False] * 4 + [True, True],
                                                releases_end=[False] * 3 + [True, True, True])
    # bracings: both gable walls and both roof planes (without the roof bracing the ridge would buckle out of the plane of the frames – λ ≈ 0.9)
    for k, (a, b) in enumerate((("A1", "B2"), ("B1", "A2"), ("A5", "B4"), ("B5", "A4"),
                                ("A2", "B3"), ("B2", "A3"), ("A3", "B4"), ("B3", "A4")), 1):
        fr.members[f"ztužidlo {k}"] = FrameMember(f"ztužidlo {k}", a, b, "CHS 76.1x3.6", kind="truss")
    fr.checks = {f"{s} sloup {p}": {"Lcr_y": 2.0 * H, "Lcr_z": H} for s in "AB" for p in "LP"}
    snow = -0.8e-3 * B                             # 0.8 kN/m² × the loaded width [N/mm]
    wind = 0.5e-3 * B
    st.load_cases.append(LoadCase("Vlastní tíha", [Gravity()]))
    st.load_cases.append(LoadCase("Sníh", [FrameMemberLoad(f"{s} příčel {p}", [0, 0, snow / 2])
                                           for s in "AB" for p in "LP"]))
    st.load_cases.append(LoadCase("Vítr zleva", [FrameMemberLoad(f"{s} sloup L", [wind / 2, 0, 0]) for s in "AB"]
                                  + [FrameNodeLoad("A2", [0, 2000.0, 0])]))
    st.modal.enabled, st.modal.modes = True, 6
    # buckling of the spatial frame by MYSTRAN (CBAR) does not converge with the subdivision of the members (λ 3.0 → 1.4 for 2 → 8 elements – bending-
    # torsional mode of the crossbars) → switched off in the example; the stability of the members is assessed by EN 1993-1-1 (buckling, lateral-torsional buckling)
    st.buckling.enabled, st.buckling.case, st.buckling.modes = False, "Sníh", 3
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    from fea_core.project import save_project
    st = build()
    out = Path(a.out)
    res = None
    if a.solve:
        from fea_core.solve import run_study
        from fea_core.beams.assess import assess
        res = run_study(st, out / "_example6_run")
        print("frekvence [Hz]:", [round(v, 2) for v in res.modes[0].values])
        if len(res.modes) > 1:
            print("buckling λ:", [round(v, 2) for v in res.modes[1].values])
        ev = assess(st, res)
        for k, v in ev.items():
            if k != "_max":
                print(f"{k:14s} utilization {v.get('utilization', 0):.3f}  {v.get('governing', '')}  ({v.get('case', '')})")
    save_project(out / "example6_hall_frame.fea", st, "", res if False else None)
    print("saved", out / "example6_hall_frame.fea")


if __name__ == "__main__":
    main()
