"""Creates the example projects of FEA Studio in the folder examples/:

  example1_bracket.fea    – one printed part of PLA (3D printing, cylindrical constraint, bearing force, self-weight, fatigue, creep)
  example2_assembly.fea  – assembly steel + printing: welds, bolts with preload, remote force, heat and thermal
                         stress, fatigue, 3D printing with creep, local refinement of the mesh
  example3_flange.fea  – flange joint of a pipe with a cover: 4× bolts M10 8.8 with preload, internal pressure 2.5 MPa,
                         axial tension of the pipe, assessment of the bolts VDI 2230
  example4_jib.fea – welded crane jib: steel S355, 5 fillet welds a4, load on the eye 5000 N,
                         transverse force, fatigue of the welds according to EN 1993-1-8 and IIW
  example5_heat_sink.fea  – electronics heat sink: finned aluminium heat sink + stainless power module, loss power 40 W,
                         convection to the surroundings, steady and transient ramp-up, thermal expansion and stress
  example7_pcb_dynamics.fea – vibration fixture with a PCB: point mass (transformer) and surface mass (the equipment), natural
                         frequencies, harmonic response to the excitation of the base 1 g, random vibration (ASD = EXAMPLE, not a standard)
  example8_crank.fea    – crank mechanism: bodies from the CAD, joints from the geometry, drive 3000 rpm, force on the piston, animation,
                         reactions; the instant of the maximum force in the pin of the piston → load case of the connecting rod in the FEM (inertia relief)
  example9_base_contact.fea – bolted column base: 2× M10 8.8, the joint as a contact with friction (nonlinear),
                         sequence of steps preload → pull 20 kN, opening of the joint
  example10_bracket_topology.fea – wall bracket: a solid design space, cylindrical constraint of the screws, load 4 kN
                         on the eye, topology optimisation BESO to 40 % of the volume (density field in the results, STL)
(example 6 = frame system, examples/frames.py)

Run:  python -m examples.create [--solve]   (--solve = calculate and save also with the results at once)
The step-by-step procedure is described by the help in the GUI (F1, Help → Example guide) and by examples/README.md.
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

# UTF-8 output for the Windows console (prevents UnicodeEncodeError for the arrows and the like)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fea_core import materials as matlib
from fea_core.fatigue.history import Channel, LoadHistory
from fea_core.fatigue.settings import FatigueSettings
from fea_core.geometry import read_step, write_step
from fea_core.joints.detect import detect
from fea_core.printing import PrintSpec
from fea_core.project import save_project
from fea_core.selectors import find_faces
from fea_core.contacts.model import LoadStep
from fea_core.dynamics.model import Excitation, PointMass, Probe, SurfaceMass
from fea_core.study import (BearingLoad, CylindricalSupport, Displacement, Fixed, Force, Gravity,
                            LoadCase, Material, MeshSettings, Pressure, RemoteLoad, Study, ThermalLoad)
from fea_core.thermal.model import Convection, FixedTemperature, HeatFlow, ThermalCase

from .geometry import (CHLADIC_MONT_OTVORY, D_KONZOLA_DIRA, D_KONZOLA_OKO, D_KOTVA, KONZOLA_DIRY, D_MONT_OTVOR, DPS, H_SLOUPEK, KOTVY, KOTVY_VYLOZNIK,
                        L_PIPE, M_PRIRUBA, PATICE, PATKA, R_PIPE_IN, T_FLANGE, T_VIKO, X_ZAKLAD,
                        chladic, drzak, klikovy_mechanismus, konzola_navrh, patka_sroubovana, pripravek_dps, priruba, sestava,
                        vyloznik)

HERE = Path(__file__).resolve().parent
PLA = "Tisk PLA Bambu Basic"


def _not_on_plane(axis: int, value: float):
    """where for find_faces: a face that does not lie entirely in the plane of the coordinate axis = value (contact faces with another part)."""
    def f(p):
        bb = p["bbox"]
        return not (abs(bb[axis] - value) < 1e-6 and abs(bb[axis + 3] - value) < 1e-6)
    return f


def _near_axis(x: float, y: float, tol: float = 1.0):
    """where for find_faces: a cylindrical face with a vertical axis at the point (x, y)."""
    def f(p):
        bb = p["bbox"]
        return abs((bb[0] + bb[3]) / 2 - x) < tol and abs((bb[1] + bb[4]) / 2 - y) < tol
    return f


# ------------------------------------------------------------------ example 1
def example1(step: Path) -> Study:
    write_step(drzak(), step)
    parts = read_step(step)
    st = Study("Ukázka 1 – tištěný držák", str(step))
    st.add_material(matlib.by_name(PLA), default=True)

    # cylindrical constraint = one axis per selection → each hole separately
    st.add_selection("Horní vrut", find_faces(parts, radius=2.5, where=lambda p: p["bbox"][2] > 20))
    st.add_selection("Dolní vrut", find_faces(parts, radius=2.5, where=lambda p: p["bbox"][5] < 20))
    st.add_selection("Zadní plocha", find_faces(parts, x=0))
    st.add_selection("Oko", find_faces(parts, radius=4.0))
    st.constraints += [CylindricalSupport("Horní vrut"), CylindricalSupport("Dolní vrut"),   # the screws hold the part
                       Displacement("Zadní plocha", ux=0.0)]            # resting on the wall (without contact)
    st.load_cases += [LoadCase("Zatížení oka 200 N", [BearingLoad("Oko", [0.0, 0.0, -200.0])]),
                      LoadCase("Vlastní tíha", [Gravity()])]

    st.mesh = MeshSettings(size=2.5)
    st.printing.parts[parts[0].name] = PrintSpec(parts[0].name, build_dir=[0.0, 0.0, 1.0], walls=3, infill=0.20,
                                                 pattern="gyroid")
    st.printing.criterion = "tsai_wu"
    st.printing.creep.enabled = True                 # a permanently hanging load → creep of the PLA
    st.printing.creep.hours = 1000.0
    st.printing.creep.temperature = 30.0
    st.printing.creep.cases = ["Zatížení oka 200 N"]

    st.fatigue = FatigueSettings(
        history=LoadHistory([Channel("Zatížení oka 200 N", kind="sine", amplitude=0.5, mean=0.5, cycles=1)]),
        repeats=1e5)                                     # alternating load 0–200 N, 100 000 cycles
    return st


# ------------------------------------------------------------------ example 2
def example2(step: Path) -> Study:
    write_step(sestava(), step)
    parts = read_step(step)
    st = Study("Ukázka 2 – sestava ocel + tisk", str(step))
    st.add_material(matlib.by_name("Ocel S235"), default=True)
    st.add_material(matlib.by_name(PLA))
    st.part_materials["Rameno (tisk PLA)"] = PLA

    # screws and welds by the names from the CAD (in the GUI: step Joints → Recognise joints…)
    rep = detect(parts, [], st.joints, shapes=False)
    st.bolts, st.welds = rep.bolts, rep.welds

    for k, (x, y) in enumerate(KOTVY, 1):                       # cylindrical constraint = one axis per selection
        st.add_selection(f"Kotva {k}", find_faces(parts, part="Patka", radius=4.5, where=_near_axis(x, y))
                         )
    st.add_selection("Oko ramena", find_faces(parts, part="Rameno (tisk PLA)", radius=5.0))
    st.add_selection("Hlava sloupku", find_faces(parts, part="Sloupek", z=90))
    st.add_selection("Otvor sloupku", find_faces(parts, part="Sloupek", radius=5.0))
    st.add_selection("Spodek patky", find_faces(parts, part="Patka", z=PATKA[2]))
    st.add_selection("Povrch ramena", find_faces(parts, part="Rameno (tisk PLA)", where=_not_on_plane(2, 0.0)))
    st.constraints += [CylindricalSupport(f"Kotva {k}") for k in range(1, len(KOTVY) + 1)]   # kotvy do podlahy

    st.load_cases += [
        LoadCase("Síla na oko ramena", [BearingLoad("Oko ramena", [0.0, 0.0, -300.0])]),
        LoadCase("Boční síla na sloupek", [RemoteLoad("Hlava sloupku", [35.0, 35.0, 150.0], force=[250.0, 0.0, 0.0])]),
        LoadCase("Vlastní tíha", [Gravity()]),
        LoadCase("Ohřev za provozu", [ThermalLoad(case="Provoz – ustálený stav")]),
    ]

    # heat: a heating element in the hole of the column 15 W, the arm cooled by air, the floor 20 °C
    st.thermal.reference_temperature = 20.0
    tl = [HeatFlow("Otvor sloupku", 15.0), Convection("Povrch ramena", h=10.0, ambient=25.0),
          FixedTemperature("Spodek patky", 20.0)]
    st.thermal.cases += [ThermalCase("Provoz – ustálený stav", list(tl)),
                         ThermalCase("Náběh – 1 hodina", list(tl), transient=True, time=3600.0, dt=120.0, outputs=6)]

    st.mesh = MeshSettings(size=4.0, local=[{"selection": "Oko ramena", "size": 1.5}])

    st.printing.parts["Rameno (tisk PLA)"] = PrintSpec("Rameno (tisk PLA)", build_dir=[0.0, 0.0, 1.0], walls=3,
                                                       infill=0.25, pattern="gyroid")
    st.printing.creep.enabled = True
    st.printing.creep.hours = 2000.0
    st.printing.creep.temperature = 30.0
    st.printing.creep.cases = ["Síla na oko ramena"]

    st.fatigue = FatigueSettings(
        history=LoadHistory([Channel("Síla na oko ramena", kind="sine", amplitude=0.5, mean=0.5, cycles=1),
                             Channel("Boční síla na sloupek", kind="sine", amplitude=1.0, cycles=1, phase_deg=90)]),
        repeats=2e5)
    return st


# ------------------------------------------------------------------ example 3: Flange
def example3(step: Path) -> Study:
    write_step(priruba(), step)
    parts = read_step(step)
    st = Study("Ukázka 3 – přírubový spoj", str(step))
    st.add_material(matlib.by_name("Ocel S355"), default=True)

    rep = detect(parts, [], st.joints, shapes=False)
    st.bolts = rep.bolts

    # Selections
    st.add_selection("Spodek trubky", find_faces(parts, part="Priruba potrubi", z=-L_PIPE))
    st.add_selection("Vnitrek trubky", find_faces(parts, part="Priruba potrubi", radius=R_PIPE_IN))
    st.add_selection("Horni celo vika", find_faces(parts, part="Viko priruby", z=T_VIKO))

    st.constraints += [Fixed("Spodek trubky")]

    st.load_cases += [
        LoadCase("Vnitřní tlak 2,5 MPa", [Pressure("Vnitrek trubky", 2.5)]),
        LoadCase("Osový tah potrubí 10 kN", [Force("Horni celo vika", [0.0, 0.0, 10000.0])]),
        LoadCase("Vlastní tíha", [Gravity()]),
    ]

    st.mesh = MeshSettings(size=5.0)
    return st


# ------------------------------------------------------------------ example 4: Jib
def example4(step: Path) -> Study:
    write_step(vyloznik(), step)
    parts = read_step(step)
    st = Study("Ukázka 4 – svařovaný výložník", str(step))
    st.add_material(matlib.by_name("Ocel S355"), default=True)

    rep = detect(parts, [], st.joints, shapes=False)
    st.welds = rep.welds

    for k, (y, z) in enumerate(KOTVY_VYLOZNIK, 1):
        st.add_selection(
            f"Kotva {k}",
            find_faces(parts, part="Kotevni deska", radius=6.5,
                       where=lambda p, cy=y, cz=z: abs((p["bbox"][1] + p["bbox"][4]) / 2 - cy) < 2 and abs((p["bbox"][2] + p["bbox"][5]) / 2 - cz) < 2))
        st.constraints.append(CylindricalSupport(f"Kotva {k}"))

    st.add_selection("Zadni stena desky", find_faces(parts, part="Kotevni deska", x=-15.0))
    st.constraints.append(Displacement("Zadni stena desky", ux=0.0))

    st.add_selection("Oko vylozniku", find_faces(parts, part="Rameno vylozniku", radius=10.0))
    st.add_selection("Celo vylozniku", find_faces(parts, part="Rameno vylozniku", x=260.0))

    st.load_cases += [
        LoadCase("Břemeno 5000 N", [BearingLoad("Oko vylozniku", [0.0, 0.0, -5000.0])]),
        LoadCase("Boční síla 800 N", [Force("Celo vylozniku", [0.0, 800.0, 0.0])]),
        LoadCase("Vlastní tíha", [Gravity()]),
    ]

    st.mesh = MeshSettings(size=6.0)

    st.fatigue = FatigueSettings(
        history=LoadHistory([
            Channel("Břemeno 5000 N", kind="sine", amplitude=0.5, mean=0.5, cycles=1),
            Channel("Boční síla 800 N", kind="sine", amplitude=0.5, mean=0.0, cycles=1, phase_deg=90),
        ]),
        repeats=2e5,
    )
    return st


# ------------------------------------------------------------------ example 5: Heat sink
def example5(step: Path) -> Study:
    write_step(chladic(), step)
    parts = read_step(step)
    st = Study("Ukázka 5 – chladič elektroniky", str(step))
    st.add_material(matlib.by_name("Hliník EN AW-6082 T6"), default=True)
    st.add_material(matlib.by_name("Nerez 1.4301"))
    st.part_materials["Vykonovy modul"] = "Nerez 1.4301"

    for k, (x, y) in enumerate(CHLADIC_MONT_OTVORY, 1):
        st.add_selection(
            f"Otvor {k}",
            find_faces(parts, part="Chladic (Al)", radius=D_MONT_OTVOR / 2,
                       where=lambda p, cx=x, cy=y: abs((p["bbox"][0] + p["bbox"][3]) / 2 - cx) < 2 and abs((p["bbox"][1] + p["bbox"][4]) / 2 - cy) < 2))
        st.constraints.append(CylindricalSupport(f"Otvor {k}"))

    st.add_selection("Spodek modulu", find_faces(parts, part="Vykonovy modul", z=-4.0))
    st.add_selection("Ochlazovany povrch", find_faces(
        parts, part="Chladic (Al)",
        where=lambda p: not (abs(p["bbox"][2]) < 1e-5 and abs(p["bbox"][5]) < 1e-5 and 30 <= p["bbox"][0] <= 60 and 25 <= p["bbox"][1] <= 55)))

    st.thermal.reference_temperature = 20.0
    tl = [
        HeatFlow("Spodek modulu", 40.0),
        Convection("Ochlazovany povrch", h=15.0, ambient=25.0),
    ]
    st.thermal.cases += [
        ThermalCase("Provoz – ustálený stav", list(tl)),
        ThermalCase("Náběh – 10 minut", list(tl), transient=True, time=600.0, dt=60.0, outputs=6),
    ]

    st.load_cases += [
        LoadCase("Ohřev za provozu", [ThermalLoad(case="Provoz – ustálený stav")]),
        LoadCase("Vlastní tíha", [Gravity()]),
    ]

    st.mesh = MeshSettings(size=4.0)
    return st


# ------------------------------------------------------------------ example 7: dynamics of the fixture with a PCB
FR4 = Material("DPS FR-4 (orientačně)", 22000.0, 0.15, 1.9e-9, Re=None, Rm=300.0,
               source="typické hodnoty laminátu FR-4 v rovině desky (IPC-4101) – orientačně")
# ASD = an EXAMPLE in the structure of IEC 60068-2-64 (rise +3 dB/oct, plateau, fall −3 dB/oct) – the numbers are NOT from the standard
ASD_PRIKLAD = [[20.0, 0.008], [50.0, 0.02], [500.0, 0.02], [2000.0, 0.005]]


def example7(step: Path) -> Study:
    write_step(pripravek_dps(), step)
    parts = read_step(step)
    st = Study("Ukázka 7 – dynamika přípravku s DPS", str(step))
    st.add_material(matlib.by_name("Hliník EN AW-6082 T6"), default=True)
    st.add_material(FR4)
    st.add_material(matlib.by_name("Mosaz CuZn39Pb3"))
    st.part_materials["DPS (FR-4)"] = FR4.name
    st.part_materials["Patice transformatoru"] = FR4.name
    for k in range(1, 5):
        st.part_materials[f"Sloupek {k}"] = "Mosaz CuZn39Pb3"
    x0, yb, z0, dx, t, dz = DPS
    yf = yb - t                                         # front (populated) side of the PCB
    px, pz, a, h = PATICE
    st.add_selection("Upnutí na stůl vibrátoru", find_faces(parts, part="Konzola (Al)", z=0))
    st.add_selection("Patice transformátoru", find_faces(parts, part="Patice transformatoru", y=yf - h))
    st.add_selection("Osazení DPS", find_faces(parts, part="DPS (FR-4)", y=yf))
    st.constraints.append(Fixed("Upnutí na stůl vibrátoru"))     # the base screwed to the table = the excited base

    m = 0.080e-3                                        # transformer 80 g, cube 24 mm, centre of mass 12 mm above the PCB
    J = m * (24.0 ** 2 + 24.0 ** 2) / 12
    st.masses += [PointMass("Patice transformátoru", m, point=[px + a / 2, yf - 12.0, pz + a / 2],
                            inertia=[J, J, J], coupling="rigid", name="Transformátor 80 g"),
                  SurfaceMass("Osazení DPS", 1.5, name="Osazení DPS 1,5 kg/m²")]
    st.load_cases.append(LoadCase("Vlastní tíha", [Gravity()]))

    ds = st.dynamics
    ds.enabled = True
    ds.modes = 20
    ds.damping.kind, ds.damping.zeta = "modal", 0.03
    hs = ds.harmonic
    hs.enabled = True
    hs.f_min, hs.f_max, hs.points = 10.0, 2000.0, 20
    hs.excitation = [Excitation(kind="base_acc", dof=2, unit="g", scale=1.0)]      # sine 1 g perpendicular to the PCB (axis Y)
    r = ds.random
    r.enabled = True
    r.asd = [list(x) for x in ASD_PRIKLAD]
    r.dof, r.duration_h, r.method = 2, 0.5, "dirlik"
    ds.probes = [Probe("Transformátor (patice)", point=[px + a / 2, yf - h, pz + a / 2]),
                 Probe("DPS vlevo od transformátoru", point=[30.0, yf, pz + a / 2]),
                 Probe("Horní hrana konzoly", point=[60.0, 57.5, 86.0])]
    st.mesh = MeshSettings(size=6.0, local=[{"selection": "Osazení DPS", "size": 3.0}])   # fine only at the plate
    return st


# ------------------------------------------------------------------ example 8: crank mechanism → FEM of the connecting rod
R_KLIKA, L_OJNICE = 25.0, 100.0
OMEGA8 = 3000.0 * 2 * math.pi / 60                    # 3000 ot/min [rad/s]
F_PIST = 3000.0                                       # gas force on the piston [N] (against the positive direction of x)


def mechanism8(st: Study, parts, r: float = R_KLIKA, l: float = L_OJNICE, omega: float = OMEGA8, force: float = F_PIST,
               outputs: int = 120, non_redundant: bool = True) -> None:
    """Bodies, joints from the geometry (coaxial pins and holes), guide of the piston, drive and the force on the piston."""
    from fea_core.motion.joints_cad import apply_suggestion, suggest_from_geometry
    from fea_core.motion.model import Body, Driver, MotionJoint, MotionLoad, MotionRun, ToFea
    M = st.mechanism
    M.bodies = [Body("Rám", parts=["Ram"], ground=True), Body("Klika", parts=["Klika"]),
                Body("Ojnice", parts=["Ojnice"]), Body("Píst", parts=["Pist"])]
    names = {frozenset(("Rám", "Klika")): "Čep kliky", frozenset(("Klika", "Ojnice")): "Čep ojnice",
             frozenset(("Ojnice", "Píst")): "Čep pístu"}
    for sug in suggest_from_geometry(st, parts):        # in the GUI: step Motion → Joints → Propose from geometry
        nm = names.get(frozenset((sug.joint.body_a, sug.joint.body_b)))
        if nm and nm not in [j.name for j in M.joints]:
            apply_suggestion(st, parts, sug, nm)
    found = sorted((j.name, j.kind) for j in M.joints)
    if found != sorted((n, "revolute") for n in names.values()):
        raise RuntimeError(f"Klouby z geometrie neodpovídají (čekány 3 rotační): {found}")
    order = list(names.values())
    M.joints.sort(key=lambda j: order.index(j.name))
    xb = r + l
    if non_redundant:
        # a planar mechanism made only of rotary joints is over-constrained in space (3 redundant constraints – Mobility
        # check) → the moments perpendicular to the plane would be shared arbitrarily between the joints and the connecting rod would get an absurd
        # bending out of the plane. A non-redundant set (RSUP): the pin of the connecting rod spherical, the pin of the piston a universal joint (rotation about the
        # axis of the connecting rod blocked) → the reactions are unique.
        M.joint("Čep ojnice").kind = "spherical"
        j = M.joint("Čep pístu")
        j.kind, j.axis, j.axis2 = "universal", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    M.joints.append(MotionJoint("Vedení pístu", "prismatic", "Rám", "Píst", [xb, 0.0, 13.0], [1.0, 0.0, 0.0]))
    M.drivers = [Driver("Čep kliky", "angular_velocity", "constant", [omega], name="Motor")]
    M.loads = [MotionLoad("Píst", [xb, 0.0, 13.0], [-force, 0.0, 0.0], name="Síla plynu na píst")]
    M.run = MotionRun("kinematic", t_end=2 * math.pi / omega, outputs=outputs,
                      to_fea=ToFea(mode="max", quantity="reakce:Čep pístu", parts=["Ojnice"]))


def example8(step: Path) -> Study:
    write_step(klikovy_mechanismus(R_KLIKA, L_OJNICE), step)
    parts = read_step(step)
    st = Study("Ukázka 8 – klikový mechanismus", str(step))
    st.add_material(matlib.by_name("Ocel C45 +N"), default=True)
    st.add_material(matlib.by_name("Ocel 42CrMo4 +QT"))
    st.part_materials["Ojnice"] = "Ocel 42CrMo4 +QT"
    mechanism8(st, parts)
    st.ignored_parts = ["Ram", "Klika", "Pist"]   # only the connecting rod goes into the FEM (the mass of the other moving parts stays)
    for j in st.mechanism.joints:                 # the joint faces on the parts outside the FEM are not needed (the mesh does not find them)
        for side in ("selection_a", "selection_b"):
            sel = getattr(j, side)
            if sel and all(f.part in st.ignored_parts for f in st.selections[sel].faces):
                del st.selections[sel]
                setattr(j, side, None)
    st.mesh = MeshSettings(size=2.0)
    return st


def solve8(st: Study, step: Path, tmp: Path):
    """Motion (Exudyn) → the instant of the maximum reaction in the pin of the piston → load state of the connecting rod → FEM."""
    from fea_core.motion.run import run_motion
    from fea_core.motion.to_fea import check_relief, export_loads
    from fea_core.solve import run_study
    parts = read_step(step)
    motion = run_motion(st, tmp / "mbd", parts=parts, on_log=_safe_log)
    info = export_loads(st, motion)
    mx = info["maximum"]
    print(f"   max. reaction in the piston pin {mx['value']:.1f} N at {mx['t'] * 1e3:.3f} ms")
    results = run_study(st, tmp / "mkp", parts=parts, on_log=_safe_log)
    rel = check_relief(info["cases"][0], results.step(info["load_cases"][0]))
    print(f"   inertia relief: reaction FEA − MBD {rel['max_rel']:.2e} (relative)")
    return results, motion


# ------------------------------------------------------------------ example 9: bolted base – contact in the joint
F9_TAH = 20000.0                                      # pull of the column at the head of the column [N] (2 × F_V ≈ 40 kN)
K9 = 10 * 210000.0                                    # penalty stiffness of the joint 10·E [N/mm³] (the default 50·E converges
                                                      # more slowly for a large planar joint – ~1.4× more iterations, the same results)


def example9(step: Path) -> Study:
    from fea_core.contacts.model import ContactPair
    write_step(patka_sroubovana(), step)
    parts = read_step(step)
    st = Study("Ukázka 9 – šroubovaná patka, spára jako kontakt", str(step))
    st.add_material(matlib.by_name("Ocel S235"), default=True)
    st.bolts = detect(parts, [], st.joints, shapes=False).bolts          # v GUI: Spoje → Rozpoznat spoje…
    st.add_selection("Konce základové desky", find_faces(parts, part="Zakladova deska", x=-X_ZAKLAD)
                     + find_faces(parts, part="Zakladova deska", x=X_ZAKLAD))
    st.add_selection("Hlava sloupku", find_faces(parts, part="Patka sloupku", z=H_SLOUPEK))
    st.constraints.append(Fixed("Konce základové desky"))     # the plate clamped at the ends (the bottom carries the nuts of the screws)
    tah = f"Tah {F9_TAH / 1000:g} kN"
    st.load_cases.append(LoadCase(tah, [Force("Hlava sloupku", [0.0, 0.0, F9_TAH])]))
    # the joint base–plate = a contact with friction μ_T (step Contacts: "joints of the bolted connections as contacts"); a pair of the
    # same name replaces its default setting automatically (here only the softer penalty K)
    st.joints.contact_joint = True
    st.contacts.append(ContactPair("Spára Patka sloupku–Zakladova deska", "Patka sloupku", "Zakladova deska",
                                   kind="frictional", mu=0.15, stiffness=K9))
    # nonlinear sequence of steps (one CalculiX job): preload of the bolts → pull of the column
    an = st.analysis
    an.kind, an.branches = "nonlinear", False
    an.sequence = [LoadStep("Předpětí šroubů", case=None, bolts="preload", dt0=0.5, dt_max=1.0),
                   LoadStep(tah, case=tah, dt0=0.5, dt_max=1.0)]
    st.mesh = MeshSettings(size=6.0)
    return st


# ------------------------------------------------------------------ example 10: bracket – topology optimisation
F10_BREMENO = 4000.0                                  # load on the eye of the pin [N]
TOPO10 = dict(target_volume_fraction=0.40, evolution_rate=0.04, filter_radius=8.0, max_iterations=60)


def example10(step: Path) -> Study:
    write_step(konzola_navrh(), step)
    parts = read_step(step)
    st = Study("Ukázka 10 – konzola, topologická optimalizace", str(step))
    st.add_material(matlib.by_name("Ocel S235"), default=True)
    r_dira, r_oko = D_KONZOLA_DIRA / 2, D_KONZOLA_OKO / 2
    for k, (x, z) in enumerate(KONZOLA_DIRY, 1):          # cylindrical constraint = one axis per selection → each hole separately
        st.add_selection(f"Šroub {k}", find_faces(parts, radius=r_dira,
                                                  where=lambda p, cz=z: abs((p["bbox"][2] + p["bbox"][5]) / 2 - cz) < 1))
        st.constraints.append(CylindricalSupport(f"Šroub {k}"))
    st.add_selection("Opření o stěnu", find_faces(parts, x=0))
    st.constraints.append(Displacement("Opření o stěnu", ux=0.0))
    st.add_selection("Oko", find_faces(parts, radius=r_oko))
    st.load_cases.append(LoadCase(f"Břemeno {F10_BREMENO / 1000:g} kN", [BearingLoad("Oko", [0.0, 0.0, -F10_BREMENO])]))
    st.mesh = MeshSettings(size=4.0)
    return st


def solve10(st: Study, step: Path, tmp: Path):
    """Static calculation of the full design space → topology optimisation BESO on the same mesh → the result attached
    to the results (the field "Topology – density", the summary in Results.info["topology"]).
    """
    from fea_core.solve import run_study
    from fea_core.topo import TopologyOptSpec, TopologyOptimizer
    from fea_core.topo.results import attach
    results = run_study(st, tmp / "run", on_log=_safe_log)
    spec = TopologyOptSpec(**TOPO10, non_design_selections=list(st.selections))
    opt = TopologyOptimizer(st, results.mesh, spec)
    opt.run(callback=lambda it, n, vf, c: _safe_log(f"BESO iterace {it}: objem {vf * 100:.1f} %, poddajnost {c:.4g} N·mm"),
            workdir=tmp / "topo")
    info = attach(results, opt)
    _safe_log(f"BESO: {info['iterations']} iterací, objem {info['final_volume_fraction'] * 100:.1f} %, poddajnost "
              f"{info['initial_compliance_Nmm']:.4g} → {info['final_compliance_Nmm']:.4g} N·mm")
    return results, None


def _safe_log(s: str) -> None:
    try:
        print("  ", s)
    except UnicodeEncodeError:
        print("  ", s.encode("ascii", errors="replace").decode("ascii"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solve", action="store_true", help="solve and save with results")
    ap.add_argument("--out", default=str(HERE), help="output folder (default examples/)")
    ap.add_argument("--only", "--jen", dest="only", type=int, choices=(1, 2, 3, 4, 5, 7, 8, 9, 10),
                    help="create only this example (1–5, 7–10; example 6 is created by examples.frames)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_examples = [
        (example1, "example1_bracket"),
        (example2, "example2_assembly"),
        (example3, "example3_flange"),
        (example4, "example4_jib"),
        (example5, "example5_heat_sink"),
        (example7, "example7_pcb_dynamics"),
        (example8, "example8_crank"),
        (example9, "example9_base_contact"),
        (example10, "example10_bracket_topology"),
    ]
    solvers = {"example8_crank": solve8, "example10_bracket_topology": solve10}                 # mechanism: motion → transfer to the FEM → FEM
    for fn, name in all_examples:
        k = int(name[6:].split("_")[0])
        if a.only and a.only != k:
            continue
        with tempfile.TemporaryDirectory(prefix="fea_ukazka_") as tmp:
            step = out / f"{name}.step"
            t0 = time.time()
            st = fn(step)
            results = motion = None
            if a.solve:
                from fea_core.solve import run_study
                if name in solvers:
                    results, motion = solvers[name](st, step, Path(tmp))
                else:
                    results = run_study(st, Path(tmp) / "run", on_log=_safe_log)
            save_project(out / f"{name}.fea", st, step, results, motion=motion)
            print(f"{name}.fea saved" + (" with results" if results is not None else "")
                  + f" ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
