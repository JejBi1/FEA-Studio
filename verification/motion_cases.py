"""Verification of mechanisms (stage F8.3, F8.8 – fea_core.motion, Exudyn): python -m verification.run_all MB1 MB2 …

MB1  crank r 20, l 80, ω 100 rad/s, φ = 60°: position, velocity, acceleration of the piston, moment M = F·v/ω (bodies almost massless)
MB2  four-bar linkage a 40, b 120, c 80, d 100: Freudenstein θ₄ of both branches (assembly from an inexact CAD position), a full revolution
MB3  pendulum L 0.5 m, θ₀ = 60°: period (elliptic integral, physical pendulum), force in the pin at the bottom, energy
MB3b pendulum with pin friction μ·|R|·r (decision 2): amplitude against the integration of the same equation in numpy
MB4  rolling of a solid cylinder on a plane at 30° (kinematic rolling constraint = pinion–rack): a = 2/3·g·sin α, friction force;
     slip at μ 0.1 (a = 4.055 m/s²) without contact NOT verifiable – only a check that rolling needs μ ≥ tan α/3
MB5  run-out of a block v₀ 1 m/s, μ 0.2 (sliding joint with Coulomb friction): s = v₀²/(2μg), t = v₀/(μg), then it stands
MB6  stick-slip m 1 kg on a belt 0.1 m/s, k 1000 N/m, μs 0.4, μk 0.3 (Stribeck, regularisation)
MB7  MB1 with pins r 5, μ 0.1, massive bodies: power balance 1e-4, equilibrium of the bodies
MB8  spatial 4R with parallel axes: Grübler −2, actual mobility 1, redundant 3
MB9  gear drive (exact ratio) + screw joint (lift = pitch × revolutions, M = F·P/2π)
MB15 start-up of a flywheel by a moment and braking by an S-curve (D6): ω = M·t/J, drive moment = J·ε
MB16 stop with restitution (D6): pendulum on a stop, ratio of the velocities after / before the impact ≈ e
MB11–MB14 (transfer to the FEM, collision) – geometry of OCP, see below.
MB10 flexible arm (Craig–Bampton, Exudyn FFRF) against ccx *FREQUENCY and Euler–Bernoulli
MB4C, MB4R, KK5 – contact of triangle meshes in Chrono (F8.9 C1, module chrono): slip μ 0.1 (4.055 m/s²), rolling μ 0.3
     (3.270 m/s²), eccentric cam e 5, R_c 30, roller 10: s(60°) 7.265, s(90°) 4.686 mm
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from fea_core.motion.model import (Body, MotionJoint, JointFriction, ForceElement, Driver, MotionLoad, MotionRun,
                                   ToFea, MotionContact)
from fea_core.study import Study, Material
from verification.cases import AtLeast, AtMost, Case, Check, E, NU, RHO

G = 9810.0                       # mm/s²


class Zero(Check):
    """Check "≈ 0" (reference 0): error = |value| (already relative / dimensionless)."""

    @property
    def error(self) -> float:
        return abs(self.value)


def _zero(name, value, tol, unit=""):
    return Zero(name, abs(float(value)), 0.0, tol, unit)


def _study(name: str) -> Study:
    st = Study(name, "")
    st.add_material(Material("Ocel S235", E, NU, RHO, Re=235.0, Rm=360.0), default=True)
    return st


def _run(study, wd, parts):
    from fea_core.motion.run import run_motion
    return run_motion(study, wd, parts=parts or None)


def _common(res) -> list[Check]:
    out = [_zero("rovnováha těles Σ F = m·a, Σ M = dH/dt (rel.)", res.meta["equilibrium"]["max_rel"], 1e-6)]
    pb = res.meta.get("power_balance")
    if pb:
        out.append(_zero("výkonová bilance P = dE/dt (rel.)", pb["max_rel"], 1e-4))
    return out


def _I(m, r=1.0):
    return (np.eye(3) * 0.4 * m * r * r).tolist()


# ------------------------------------------------------------------ MB1 klika
R1, L1, W1, F1 = 20.0, 80.0, 100.0, -1000.0


def _crank(st, masses=(1e-9, 1e-9, 1e-9), friction=None):
    M = st.mechanism
    M.gravity = None
    mc, mr, mp = masses
    M.bodies = [Body("rám", ground=True),
                Body("klika", mass=mc, com=[R1 / 2, 0, 0], inertia=(np.diag([1, 1, 1]) * mc * R1 * R1 / 12).tolist()),
                Body("ojnice", mass=mr, com=[R1 + L1 / 2, 0, 0],
                     inertia=(np.diag([1e-3, 1, 1]) * mr * L1 * L1 / 12).tolist()),
                Body("píst", mass=mp, com=[R1 + L1, 0, 0], inertia=_I(mp, 10.0))]
    fr = friction
    M.joints = [MotionJoint("čep klika", "revolute", "rám", "klika", [0, 0, 0], [0, 0, 1], friction=fr),
                MotionJoint("čep A", "revolute", "klika", "ojnice", [R1, 0, 0], [0, 0, 1], friction=fr),
                MotionJoint("čep B", "revolute", "ojnice", "píst", [R1 + L1, 0, 0], [0, 0, 1], friction=fr),
                MotionJoint("vedení", "prismatic", "rám", "píst", [R1 + L1, 0, 0], [1, 0, 0])]
    M.drivers = [Driver("čep klika", "angular_velocity", "constant", [W1], name="motor")]
    M.loads = [MotionLoad("píst", [R1 + L1, 0, 0], [F1, 0, 0], name="síla na píst")]
    M.run = MotionRun("kinematic", t_end=2 * math.pi / W1, outputs=360)


def _slider(phi):
    """x(φ), v, a of the piston (analytic, ω constant)."""
    s, c = math.sin(phi), math.cos(phi)
    q = math.sqrt(L1 * L1 - (R1 * s) ** 2)
    x = R1 * c + q
    dx = -R1 * s - R1 * R1 * s * c / q                     # dx/dφ
    ddx = -R1 * c - R1 * R1 * (c * c - s * s) / q - R1 ** 4 * s * s * c * c / q ** 3
    return x, dx * W1, ddx * W1 * W1


def mb1_build(wd: Path):
    st = _study("MB1 klika")
    _crank(st)
    return st, []


def mb1_eval(res, st):
    k = 60                                                  # 360 outputs per revolution → 60°
    x, v, a = _slider(math.radians(60))
    A = res.arrays
    M = A["dQ"][k, 0]
    return [
        Check("čas výstupu = 60° / ω", A["t"][k], math.radians(60) / W1, 1e-12, "s"),
        Check("poloha pístu x(60°)", A["pos"][k, 3, 0], x, 1e-6, "mm"),
        Check("rychlost pístu v(60°)", A["vel"][k, 3, 0], v, 1e-4, "mm/s"),
        Check("zrychlení pístu a(60°)", A["acc"][k, 3, 0], a, 1e-4, "mm/s²"),
        Check("moment pohonu M = F·v/ω", M, F1 * v / W1 * -1.0, 1e-4, "N·mm"),
        Check("vzorec zadání a = −75 022 mm/s²", A["acc"][k, 3, 0], -75022.0, 1e-4, "mm/s²"),
        *_common(res),
    ]


# ------------------------------------------------------------------ MB2 four-bar linkage
A2, B2, C2, D2 = 40.0, 120.0, 80.0, 100.0


def freudenstein(th2: float, branch: int) -> float:
    """θ₄ [rad] of the four-bar linkage (link a = crank, b = coupler, c = rocker, d = frame), branch +1 open / −1 crossed."""
    K1, K2, K3 = D2 / A2, D2 / C2, (A2 ** 2 - B2 ** 2 + C2 ** 2 + D2 ** 2) / (2 * A2 * C2)
    Aq = math.cos(th2) - K1 - K2 * math.cos(th2) + K3
    Bq = -2 * math.sin(th2)
    Cq = K1 - (K2 + 1) * math.cos(th2) + K3
    return 2 * math.atan2(-Bq + branch * -math.sqrt(Bq * Bq - 4 * Aq * Cq), 2 * Aq) % (2 * math.pi)


def _fourbar(st, th4_cad, drive=True, t_end=2 * math.pi):
    """Four-bar linkage in the CAD position θ₂ = 60°, the rocker turned by θ₄_CAD (inexactly) – the joint point of the coupler on the rocker is elsewhere
    than on the coupler (point_b) → the assembly.
    """
    M = st.mechanism
    M.gravity = None
    th2 = math.radians(60)
    O2, O4 = np.array([0.0, 0, 0]), np.array([D2, 0, 0])
    A = O2 + A2 * np.array([math.cos(th2), math.sin(th2), 0])
    Br = O4 + C2 * np.array([math.cos(th4_cad), math.sin(th4_cad), 0])     # end of the rocker (CAD, approximate)
    u = (Br - A) / np.linalg.norm(Br - A)
    Bc = A + B2 * u                                                       # end of the coupler (CAD)
    m = 1e-6
    M.bodies = [Body("rám", ground=True), Body("klika", mass=m, com=list((O2 + A) / 2), inertia=_I(m, 10)),
                Body("táhlo", mass=m, com=list((A + Bc) / 2), inertia=_I(m, 10)),
                Body("vahadlo", mass=m, com=list((O4 + Br) / 2), inertia=_I(m, 10))]
    M.joints = [MotionJoint("O2", "revolute", "rám", "klika", list(O2), [0, 0, 1]),
                MotionJoint("A", "revolute", "klika", "táhlo", list(A), [0, 0, 1]),
                MotionJoint("B", "revolute", "táhlo", "vahadlo", list(Bc), [0, 0, 1], point_b=list(Br)),
                MotionJoint("O4", "revolute", "rám", "vahadlo", list(O4), [0, 0, 1])]
    M.drivers = [Driver("O2", "angular_velocity", "constant", [1.0], name="klika")] if drive else []
    M.run = MotionRun("kinematic", t_end=t_end, outputs=360)


def mb2_build(wd: Path):
    st = _study("MB2 čtyřkloub")
    _fourbar(st, math.radians(66.0))
    return st, []


def mb2_run(st, wd, parts):
    out = {}
    for br, th4 in ((1, math.radians(66.0)), (-1, math.radians(246.0))):
        _fourbar(st, th4)
        out[br] = (_run(st, Path(wd) / f"vetev{br}", parts), th4)
    return _Multi(out)


class _Multi(dict):
    """More results of one case (run_all reads the warnings)."""

    @property
    def warnings(self):
        w = []
        for v in self.values():
            r = v[0] if isinstance(v, tuple) else v
            w += [x for x in getattr(r, "warnings", []) if x not in w]
        return w


def mb2_eval(res, st):
    out = []
    for br, (r, th4_cad) in res.items():
        A = r.arrays
        th4 = th4_cad + A["jq"][:, 3, 0]                      # angle of the rocker at the joint O4 + CAD position
        th2 = math.radians(60) + A["t"]
        ref = np.array([freudenstein(x, br) for x in th2])
        err = np.abs((th4 - ref + math.pi) % (2 * math.pi) - math.pi)
        nm = "otevřená" if br == 1 else "překřížená"
        out += [Check(f"θ₄(60°) {nm} větev", math.degrees(th4[0]) % 360, 64.943 if br == 1 else 248.230, 2e-5, "°"),
                Check(f"θ₄(60°) {nm} = Freudenstein přesně", th4[0] % (2 * math.pi), ref[0], 1e-8, "rad"),
                _zero(f"max |θ₄ − Freudenstein| za otáčku ({nm})", err.max(), 1e-7, "rad"),
                *_common(r)]
    return out


# ------------------------------------------------------------------ MB3 kyvadlo
M3, L3, RB3, TH3 = 2e-3, 500.0, 10.0, math.radians(60)


def _pendulum(st, friction=None, t_end=3.0, outputs=3000, dt=1e-4):
    M = st.mechanism
    M.gravity = [0, -G, 0]
    J = 0.4 * M3 * RB3 ** 2
    M.bodies = [Body("rám", ground=True),
                Body("kyvadlo", mass=M3, com=[L3 * math.sin(TH3), -L3 * math.cos(TH3), 0], inertia=(np.eye(3) * J).tolist())]
    M.joints = [MotionJoint("čep", "revolute", "rám", "kyvadlo", [0, 0, 0], [0, 0, 1], friction=friction)]
    M.run = MotionRun("dynamic", t_end=t_end, outputs=outputs, dt=dt)


def mb3_build(wd: Path):
    st = _study("MB3 kyvadlo")
    _pendulum(st)
    return st, []


def _crossings(t, x):
    return np.array([t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i]) for i in range(len(x) - 1)
                     if x[i] > 0 >= x[i + 1]])


def mb3_eval(res, st):
    from scipy.special import ellipk
    A = res.arrays
    J = 0.4 * M3 * RB3 ** 2
    Leq = (J + M3 * L3 ** 2) / (M3 * L3)
    T = 4 * math.sqrt(Leq / G) * ellipk(math.sin(TH3 / 2) ** 2)
    zc = _crossings(A["t"], A["pos"][:, 1, 0])
    w2 = 2 * M3 * G * L3 * (1 - math.cos(TH3)) / (J + M3 * L3 ** 2)
    Rb = M3 * G + M3 * w2 * L3
    F = np.linalg.norm(A["jF"][:, 0], axis=1)
    k = int(np.argmax(F))
    if 0 < k < len(F) - 1:                                     # vertex of the parabola from three outputs
        y0, y1, y2 = F[k - 1], F[k], F[k + 1]
        Fmax = y1 - 0.125 * (y2 - y0) ** 2 / (y2 - 2 * y1 + y0)
    else:
        Fmax = F[k]
    E = A["ekin"] + A["epot"]
    return [
        Check("perioda T (fyzikální kyvadlo, K(sin²30°))", float(np.diff(zc).mean()), T, 1e-4, "s"),
        Check("perioda proti zadání 1,5223 s (hmotný bod)", float(np.diff(zc).mean()), 1.5223, 2e-4, "s"),
        Check("síla v čepu dole = mg(3 − 2cos θ₀) = 2mg", Fmax, Rb, 1e-4, "N"),
        _zero("zachování energie (max − min)/E", (E.max() - E.min()) / (M3 * G * L3), 1e-5),
        *_common(res),
    ]


MU3B, RF3B = 0.2, 10.0


def mb3b_build(wd: Path):
    st = _study("MB3b kyvadlo s třením v čepu")
    _pendulum(st, JointFriction("coulomb", mu_s=MU3B, mu_k=MU3B, r_eff=RF3B, v_reg=0.5), t_end=3.0)
    return st, []


def pendulum_friction_reference(t_end=3.0, v_reg=0.5):
    """Integration (J + mL²)θ̈ = −mgL·sin θ − μ'·|R|·r·tanh(θ̇·r/v_reg), R = m·a_c − m·g (implicitly in θ̈)."""
    from scipy.integrate import solve_ivp
    J = 0.4 * M3 * RB3 ** 2
    Jo = J + M3 * L3 ** 2
    mu = MU3B / math.sqrt(1 + MU3B ** 2)

    def rhs(t, y):
        th, w = y
        er = np.array([math.sin(th), -math.cos(th)])
        et = np.array([math.cos(th), math.sin(th)])
        acc = -M3 * G * L3 * math.sin(th) / Jo
        for _ in range(50):
            ac = L3 * (acc * et - w * w * er)
            R = np.linalg.norm(M3 * ac - M3 * np.array([0, -G]))
            new = (-M3 * G * L3 * math.sin(th) - mu * R * RF3B * math.tanh(w * RF3B / v_reg)) / Jo
            if abs(new - acc) < 1e-14 * max(abs(acc), 1.0):
                acc = new
                break
            acc = new
        return [w, acc]
    th0 = -TH3                                              # θ od svislice: poloha CAD x = L·sin θ₀ > 0
    sol = solve_ivp(rhs, (0, t_end), [TH3, 0.0], method="Radau", rtol=1e-11, atol=1e-13, dense_output=True,
                    max_step=1e-3)
    return sol


def mb3b_eval(res, st):
    A = res.arrays
    th = np.arctan2(A["pos"][:, 1, 0], -A["pos"][:, 1, 1])  # angle from the vertical downwards
    sol = pendulum_friction_reference()
    ref = sol.sol(A["t"])[0]

    def turning(t, th_, w):
        """Angles of the turning points (ω changes sign; linear interpolation of the position in time at ω = 0) – without t = 0."""
        out = []
        for i in range(1, len(w) - 1):
            if w[i] != 0 and w[i] * w[i + 1] < 0:
                s = w[i] / (w[i] - w[i + 1])
                out.append(th_[i] + s * (th_[i + 1] - th_[i]))
        return np.array([TH3] + out)
    tt = np.linspace(0, A["t"][-1], 200001)
    yr = sol.sol(tt)
    pk = turning(A["t"], th, A["omega"][:, 1, 2])
    pr = turning(tt, yr[0], yr[1])
    n = min(len(pk), len(pr), 4)
    out = [Check(f"amplituda po {k} půlkmitech (proti numpy)", abs(pk[k]), abs(pr[k]), 2e-3, "rad") for k in range(1, n)]
    out += [Check("pokles amplitudy za první půlkmit", TH3 - abs(pk[1]), TH3 - abs(pr[1]), 2e-2, "rad"),
            _zero("max |θ − θ_numpy| / θ₀", np.abs(th - ref).max() / TH3, 3e-3),
            Check("práce tření = úbytek energie", -A["wfr"][-1], (A["ekin"][0] + A["epot"][0]) - (A["ekin"][-1] + A["epot"][-1]),
                  1e-3, "mJ"),
            *_common(res)]
    return out


# ------------------------------------------------------------------ MB4 rolling
ALPHA4, R4, M4 = math.radians(30), 50.0, 1e-3


def mb4_build(wd: Path):
    st = _study("MB4 valení válce po nakloněné rovině")
    M = st.mechanism
    M.gravity = [0, 0, -G]
    d = np.array([math.cos(ALPHA4), 0, -math.sin(ALPHA4)])          # slope of the plane (x axis down the slope)
    n = np.array([math.sin(ALPHA4), 0, math.cos(ALPHA4)])           # normal of the plane
    c = R4 * n
    J = M4 * R4 ** 2 / 2
    M.bodies = [Body("rám", ground=True), Body("vozík", mass=1e-9, com=list(c), inertia=_I(1e-9, 1)),
                Body("válec", mass=M4, com=list(c), inertia=np.diag([J / 2 + M4 * 10 ** 2 / 12, J, J / 2]).tolist())]
    # axis of the cylinder = y (perpendicular to the slope in the plane); rolling: displacement of the carriage s = −R·φ (φ about +y – rolling down = −φ? see the sign)
    M.joints = [MotionJoint("svah", "prismatic", "rám", "vozík", list(c), list(d)),
                MotionJoint("osa", "revolute", "vozík", "válec", list(c), [0, 1, 0]),
                MotionJoint("valení", "rack_pinion", joint_a="osa", joint_b="svah", ratio=R4)]
    M.run = MotionRun("dynamic", t_end=0.2, outputs=200, dt=1e-4)
    return st, []


def mb4_eval(res, st):
    A = res.arrays
    d = np.array([math.cos(ALPHA4), 0, -math.sin(ALPHA4)])
    a = A["acc"][:, 2] @ d
    k = len(a) // 2
    g = G * 1e-3
    a_roll = 2 / 3 * g * math.sin(ALPHA4)
    Ft = M4 * G * math.sin(ALPHA4) / 3                      # friction force needed for rolling
    mu_min = math.tan(ALPHA4) / 3
    n = np.array([math.sin(ALPHA4), 0, math.cos(ALPHA4)])
    # normal force in the joint "slope" (the carriage is massless → it carries the whole normal pressure of the cylinder onto the plane)
    N_solver = abs(float(A["jF"][k, 0] @ n))
    # friction force = force of the rolling constraint on the carriage in the direction of the slope: = moment in the axis / R
    Mosa = A["jMl"][k, 1, 2]                                # moment in the frame of the joint "axis" about the axis (on the cylinder)
    return [
        Check("zrychlení valení a = 2/3·g·sin α (μ 0,3 ≥ tan α/3 = 0,192)", a[k] * 1e-3, a_roll, 1e-4, "m/s²"),
        Check("zadání 3,270 m/s²", a[k] * 1e-3, 3.270, 2e-4, "m/s²"),
        Check("třecí síla valení F = m·g·sin α/3 (moment vazby / R)", abs(Mosa) / R4, Ft, 1e-4, "N"),
        Check("potřebné μ řešiče = |třecí síla (moment vazby/R)| / normálová síla kloubu svah = tan α/3 "
              "(μ 0,1 < μ_min → skluz 4,055 m/s² jen s kontaktem C1 – NEOVĚŘITELNÉ)",
              abs(Mosa) / R4 / N_solver, mu_min, 1e-3, ""),
        *_common(res),
    ]


# ------------------------------------------------------------------ MB5 run-out of a block
MU5, V5, M5 = 0.2, 1000.0, 1e-3


def mb5_build(wd: Path):
    st = _study("MB5 doběh kvádru s Coulombovým třením")
    M = st.mechanism
    M.gravity = [0, 0, -G]
    M.bodies = [Body("rám", ground=True), Body("kvádr", mass=M5, com=[0, 0, 10], inertia=_I(M5, 10), v0=[V5, 0, 0])]
    M.joints = [MotionJoint("vedení", "prismatic", "rám", "kvádr", [0, 0, 0], [1, 0, 0],
                            friction=JointFriction("coulomb", mu_s=MU5, mu_k=MU5, v_reg=0.05))]
    M.run = MotionRun("dynamic", t_end=1.0, outputs=1000, dt=2e-4)
    return st, []


def mb5_eval(res, st):
    A = res.arrays
    t, x, v = A["t"], A["pos"][:, 1, 0], A["vel"][:, 1, 0]
    s_ref = V5 ** 2 / (2 * MU5 * G)
    t_ref = V5 / (MU5 * G)
    sel = (v > 0.1 * V5) & (v < 0.9 * V5)
    p = np.polyfit(t[sel], v[sel], 1)                       # linear run-out → intersection with zero
    return [
        Check("zpomalení a = μ·g", -p[0], MU5 * G, 1e-4, "mm/s²"),
        Check("doba doběhu t = v₀/(μg) = 0,5097 s", -p[1] / p[0], t_ref, 1e-4, "s"),
        Check("dráha s = v₀²/(2μg) = 254,8 mm", x[-1] - x[0], s_ref, 1e-3, "mm"),
        _zero("rychlost po zastavení (t > 0,6 s) / v₀", np.abs(v[t > 0.6]).max() / V5, 1e-3),
        *_common(res),
    ]


# ------------------------------------------------------------------ MB6 stick-slip
M6, VB6, K6, MUS6, MUK6 = 1e-3, 100.0, 1.0, 0.4, 0.3


def mb6_build(wd: Path):
    st = _study("MB6 stick-slip na pásu")
    M = st.mechanism
    M.gravity = [0, 0, -G]
    M.bodies = [Body("rám", ground=True),
                Body("pás", mass=1e-3, com=[0, 0, -5], inertia=_I(1e-3, 10), v0=[VB6, 0, 0]),
                Body("hmota", mass=M6, com=[0, 0, 5], inertia=_I(M6, 10), v0=[VB6, 0, 0])]
    M.joints = [MotionJoint("pás", "prismatic", "rám", "pás", [0, 0, -5], [1, 0, 0]),
                MotionJoint("styk", "prismatic", "pás", "hmota", [0, 0, 0], [1, 0, 0],
                            friction=JointFriction("stribeck", mu_s=MUS6, mu_k=MUK6, v_s=0.2, delta=2.0, v_reg=0.002))]
    M.drivers = [Driver("pás", "velocity", "constant", [VB6], name="pás")]
    M.forces = [ForceElement("pružina", "spring", "rám", "hmota", [-100, 0, 5], [0, 0, 5], k=K6)]
    M.run = MotionRun("dynamic", t_end=0.7, outputs=2800, dt=2e-5)
    return st, []


def mb6_eval(res, st):
    A = res.arrays
    t, x = A["t"], A["pos"][:, 2, 0]
    xs, xk = MUS6 * M6 * G / K6, MUK6 * M6 * G / K6
    w = math.sqrt(K6 / M6)
    amp = math.hypot(xs - xk, VB6 / w)
    phi = math.asin((xs - xk) / amp)
    T = (2 * math.pi - 2 * phi) / w + 2 * (xs - xk) / VB6
    sel = t > 0.25
    tt, xx = t[sel], x[sel]
    mx = [i for i in range(1, len(xx) - 1) if xx[i] >= xx[i - 1] and xx[i] > xx[i + 1]]
    per = np.diff(tt[mx]).mean() if len(mx) > 1 else float("nan")
    fpk = np.abs(A["jfr"][:, 1, 1]).max()                  # peak of the static friction = the spring force at breakaway
    return [
        Check("x_s = μs·m·g/k (odtržení: vrchol tření / k)", fpk / K6, xs, 5e-3, "mm"),
        Check("x_k = μk·m·g/k (střed kmitu při prokluzu)", 0.5 * (xx.max() + xx.min()), xk, 5e-3, "mm"),
        Check("největší výchylka x_k + √((x_s − x_k)² + (v/ω)²)", xx.max(), xk + amp, 1e-2, "mm"),
        Check("perioda stick-slip T", per, T, 5e-3, "s"),
        Check("zadání T = 0,1993 s", per, 0.1993, 5e-3, "s"),
        *_common(res),
    ]


# ------------------------------------------------------------------ MB7 crank with friction in the pins
def mb7_build(wd: Path):
    st = _study("MB7 klika s čepovým třením")
    _crank(st, masses=(0.2e-3, 0.4e-3, 0.5e-3), friction=JointFriction("coulomb", mu_s=0.1, mu_k=0.1, r_eff=5.0))
    st.mechanism.run.outputs = 720
    return st, []


def mb7_eval(res, st):
    A = res.arrays
    pb = res.meta["power_balance"]
    eta = pb.get("efficiency")
    return [
        _zero("výkonová bilance M·ω = F·v + dE_k/dt + Σ M_tř·ω_rel (max rel.)", pb["max_rel"], 1e-4),
        _zero("integrálně: ΔE = Σ práce (rel.)", pb["integral_rel"], 1e-4),
        _zero("práce tření ≤ 0 (ztráta)", max(A["wfr"][-1], 0.0), 1e-12, "mJ"),
        Check("tření je ztráta: |W_tř| > 0", 1.0 + (A["wfr"][-1] < 0), 2.0, 1e-12, ""),
        *_common(res),
    ]


# ------------------------------------------------------------------ MB8 mobility check
def mb8_build(wd: Path):
    st = _study("MB8 prostorový 4R s rovnoběžnými osami")
    _fourbar(st, math.radians(66.0), drive=False)
    return st, []


def mb8_run(st, wd, parts):
    from fea_core.motion.run import check_dof
    d = check_dof(st, None, workdir=wd)
    return _DofRes(d)


class _DofRes(dict):
    warnings: list = []


def mb8_eval(res, st):
    return [Check("Grübler M = 6(4 − 1) − 4·5", res["grubler"], -2, 1e-12, ""),
            Check("skutečná volnost (hodnost Jacobiánu)", res["dof_free"], 1, 1e-12, ""),
            Check("redundantní vazby", res["redundant"], 3, 1e-12, ""),
            Check("hlášena nejednoznačnost reakcí", 1.0 + any("AMBIGUOUS" in m for m in res["messages"]), 2.0,
                  1e-12, "")]


# ------------------------------------------------------------------ MB9 gear and screw
RATIO9, LEAD9, F9 = -2.5, 4.0, 1000.0


def mb9_build(wd: Path):
    st = _study("MB9 ozubený převod a šroubový kloub")
    M = st.mechanism
    M.gravity = None
    m = 1e-3
    M.bodies = [Body("rám", ground=True), Body("kolo 1", mass=m, com=[0, 0, 0], inertia=_I(m, 20)),
                Body("kolo 2", mass=m, com=[75, 0, 0], inertia=_I(m, 50)),
                Body("matice", mass=m, com=[0, 200, 0], inertia=_I(m, 10))]
    M.joints = [MotionJoint("hřídel 1", "revolute", "rám", "kolo 1", [0, 0, 0], [0, 0, 1]),
                MotionJoint("hřídel 2", "revolute", "rám", "kolo 2", [75, 0, 0], [0, 0, 1]),
                MotionJoint("převod", "gear", joint_a="hřídel 1", joint_b="hřídel 2", ratio=RATIO9),
                MotionJoint("šroub", "screw", "rám", "matice", [0, 200, 0], [0, 0, 1], lead=LEAD9)]
    M.drivers = [Driver("hřídel 1", "angular_velocity", "constant", [2 * math.pi], name="motor"),
                 Driver("šroub", "angular_velocity", "constant", [4 * math.pi], name="šroub")]
    M.loads = [MotionLoad("matice", [0, 200, 0], [0, 0, -F9], name="osová síla")]
    M.run = MotionRun("kinematic", t_end=1.0, outputs=100)
    return st, []


def mb9_eval(res, st):
    A = res.arrays
    w1 = A["omega"][:, 1, 2]
    w2 = A["omega"][:, 2, 2]
    k = 50
    z = A["pos"][:, 3, 2] - A["pos"][0, 3, 2]
    turns = A["jq"][-1, 3, 0] / (2 * math.pi)
    Mscrew = A["dQ"][k, 1]
    return [
        Check("převod ω₂/ω₁", w2[k] / w1[k], RATIO9, 1e-9, ""),
        Check("úhel kola 2 = poměr × úhel kola 1 (konec)", A["jq"][-1, 1, 0], RATIO9 * A["jq"][-1, 0, 0], 1e-9, "rad"),
        Check("zdvih matice = stoupání × otáčky (2 otáčky)", z[-1], LEAD9 * turns, 1e-9, "mm"),
        Check("zdvih = 8 mm", z[-1], 8.0, 1e-9, "mm"),
        Check("moment šroubu M = F·P/(2π)", abs(Mscrew), F9 * LEAD9 / (2 * math.pi), 1e-6, "N·mm"),
        *_common(res),
    ]


# ------------------------------------------------------------------ MB15 start-up and braking (D6)
J15, M15 = 2.0, 500.0           # t·mm², N·mm


def mb15_build(wd: Path):
    st = _study("MB15 rozběh momentem a brzdění S-křivkou")
    M = st.mechanism
    M.gravity = None
    M.bodies = [Body("rám", ground=True), Body("setrvačník", mass=1e-3, com=[0, 0, 0],
                                              inertia=np.diag([J15 / 2, J15 / 2, J15]).tolist())]
    M.joints = [MotionJoint("hřídel", "revolute", "rám", "setrvačník", [0, 0, 0], [0, 0, 1])]
    M.drivers = [Driver("hřídel", "torque", "table", [[0, M15], [0.5, M15], [0.5000001, 0], [1, 0]], name="motor")]
    M.run = MotionRun("dynamic", t_end=1.0, outputs=200, dt=1e-4)    # ω·h ≤ 0.0125 rad (large rotations per step)
    return st, []


def mb15_run(st, wd, parts):
    r1 = _run(st, Path(wd) / "rozbeh", parts)
    # braking: kinematic drive with an S-curve (path 10 rad, v 30 rad/s, a 100, jerk 2000) – moment = J·ε
    st.mechanism.drivers = [Driver("hřídel", "angle", "scurve", [10.0, 30.0, 100.0, 2000.0, 0.0], name="osa")]
    st.mechanism.run = MotionRun("kinematic", t_end=0.8, outputs=400)
    r2 = _run(st, Path(wd) / "scurve", parts)
    return _Multi({"rozběh": r1, "S-křivka": r2})


def mb15_eval(res, st):
    from fea_core.motion.laws import Law
    r1, r2 = res["rozběh"], res["S-křivka"]
    A = r1.arrays
    k = int(np.argmin(np.abs(A["t"] - 0.4)))
    law = Law("scurve", [10.0, 30.0, 100.0, 2000.0, 0.0])
    B = r2.arrays
    ref_M = np.array([J15 * law.d2(t) for t in B["t"]])
    ref_phi = np.array([law.value(t) for t in B["t"]])
    return [
        Check("rozběh: ω(0,4 s) = M·t/J", A["omega"][k, 1, 2], M15 * A["t"][k] / J15, 1e-4, "rad/s"),
        Check("po vypnutí ω konstantní (t = 1 s)", A["omega"][-1, 1, 2], M15 * 0.5 / J15, 1e-3, "rad/s"),
        _zero("S-křivka: max |φ − profil| / zdvih", np.abs(B["jq"][:, 0, 0] - ref_phi).max() / 10.0, 1e-8),
        # the acceleration from finite differences has an error of order j·δ at the kinks of the jerk of the S-curve (δ = step of the differences)
        _zero("S-křivka: max |M − J·ε| / max|J·ε| (zlomy ryvu)", np.abs(B["dQ"][:, 0] - ref_M).max() / np.abs(ref_M).max(),
              1e-3),
        Check("S-křivka: max moment = J·a_max", np.abs(B["dQ"][:, 0]).max(), J15 * 100.0, 1e-3, "N·mm"),
        *_common(r1), *_common(r2),
    ]


# ------------------------------------------------------------------ MB16 stop with restitution (D6)
E16 = 0.6


def mb16_build(wd: Path):
    st = _study("MB16 kyvadlo na dorazu s restitucí")
    _pendulum(st, t_end=1.0, outputs=4000, dt=2.5e-5)
    # stop at −50° from the vertical; the coordinate of the joint is measured from the position in the CAD (+60°) → −110°
    st.mechanism.joints[0].limits = [math.radians(-50) - TH3, None]
    st.mechanism.joints[0].restitution = E16
    return st, []


def mb16_eval(res, st):
    A = res.arrays
    w = A["omega"][:, 1, 2]
    q = A["jq"][:, 0, 0]
    lo = math.radians(-50) - TH3                         # limit in the coordinate of the joint (from the position in the CAD)
    inside = q < lo
    k0 = int(np.argmax(inside))                          # first penetration into the stop
    k1 = k0 + int(np.argmax(~inside[k0:]))               # leaving the stop
    w_in, w_out = w[k0 - 1], w[k1]
    J = 0.4 * M3 * RB3 ** 2
    Jo = J + M3 * L3 ** 2
    th_hit = math.radians(-50)
    w_ref = -math.sqrt(2 * M3 * G * L3 * (math.cos(th_hit) - math.cos(TH3)) / Jo)
    return [
        Check("rychlost před dorazem (energie)", w_in, w_ref, 2e-3, "rad/s"),
        Check("součinitel restituce −ω_po/ω_před = e (lineární doraz, síla jen tlačí)", -w_out / w_in, E16, 0.02, ""),
        AtMost("největší průnik do dorazu [°] (≤ 1°, tuhost z doby rázu 1 ms; větší průnik = chyba)",
               float(np.degrees(lo - q.min())), 1.0, 1e-12, "°"),
        *_common(res)[:1],
    ]


# ------------------------------------------------------------------ geometry (OCP) for MB11–MB14
ROD_A, ROD_B, ROD_RE, ROD_RH, ROD_T, ROD_W = 20.0, 100.0, 10.0, 5.0, 10.0, 8.0


def _rod_shape():
    """Connecting rod: eyes R10 with holes Ø10 at the points A (x 20) and B (x 100), shank 8 × 10, thickness 10 (z ±5)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from verification.cases import box, cyl, solid
    s = BRepAlgoAPI_Fuse(box(ROD_A, -ROD_W / 2, -ROD_T / 2, ROD_B - ROD_A, ROD_W, ROD_T),
                         cyl((ROD_A, 0, -ROD_T / 2), (0, 0, 1), ROD_RE, ROD_T)).Shape()
    s = BRepAlgoAPI_Fuse(s, cyl((ROD_B, 0, -ROD_T / 2), (0, 0, 1), ROD_RE, ROD_T)).Shape()
    for x in (ROD_A, ROD_B):
        s = BRepAlgoAPI_Cut(s, cyl((x, 0, -ROD_T), (0, 0, 1), ROD_RH, 2 * ROD_T)).Shape()
    return solid(s)


def _geo_study(name, wd, shapes: dict, size=2.5):
    from fea_core.geometry import Part, read_step, write_step
    from fea_core.study import MeshSettings
    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    step = wd / "model.step"
    write_step([Part(n, s) for n, s in shapes.items()], step)
    parts = read_step(step)
    st = _study(name)
    st.geometry = str(step)
    st.mesh = MeshSettings(size=size)
    return st, parts


def _hole(parts, part, x):
    from fea_core.selectors import find_faces
    return find_faces(parts, part=part, type="cylinder", radius=ROD_RH,
                      where=lambda p: abs(p["centroid"][0] - x) < 1.0)


def _crank_rod(st, parts, outputs=72, revolutions=1.0):
    """MB1 with the connecting rod from CAD (part "Ojnice", steel density) – crank and piston without geometry (given mass)."""
    _crank(st, masses=(0.2e-3, 1.0, 0.3e-3))
    M = st.mechanism
    M.bodies[2] = Body("ojnice", parts=["Ojnice"])
    st.add_selection("oko A", _hole(parts, "Ojnice", ROD_A))
    st.add_selection("oko B", _hole(parts, "Ojnice", ROD_B))
    M.joints[1].selection_b = "oko A"                      # pin A: crank → connecting rod (the connecting rod = body_b)
    M.joints[2].selection_a = "oko B"                      # pin B: connecting rod → piston (the connecting rod = body_a)
    M.run = MotionRun("kinematic", t_end=revolutions * 2 * math.pi / W1, outputs=outputs)


def mb11_build(wd: Path):
    st, parts = _geo_study("MB11 klika → MKP (inertia relief)", wd, {"Ojnice": _rod_shape()})
    _crank_rod(st, parts)
    return st, parts


def mb11_run(st, wd, parts):
    from fea_core.motion.to_fea import export_loads
    from fea_core.solve import run_study
    res = _run(st, Path(wd) / "mbd", parts)
    info = export_loads(st, res, mode="max", quantity="reakce:čep B", parts=["Ojnice"])
    fe = run_study(st, Path(wd) / "mkp", parts=parts)
    return _Multi({"mbd": res, "fe": fe, "info": info})


def mb11_eval(res, st):
    from fea_core.motion.to_fea import check_relief
    r, fe, info = res["mbd"], res["fe"], res["info"]
    case = info["cases"][0]
    step = fe.step(case["load_case"])
    rel = check_relief(case, step)
    k = case["index"]
    A = r.arrays
    R = A["rot"][k, 2]
    ex = R[:, 0]                                           # axis of the connecting rod at the time k (CAD axis x)
    # axial force in the shank: average of the axial components of the forces in the eyes (the inertia of the connecting rod ~ linear along the length)
    FA = A["jF"][k, 1] @ ex                                # force of the crank on the connecting rod (eye A) in the direction of the axis
    FB = -A["jF"][k, 2] @ ex                               # force of the piston on the connecting rod (eye B)
    N = 0.5 * (FB - FA)                                    # tah > 0
    sig = fe.probe([0.5 * (ROD_A + ROD_B), 0.0, 0.0], "S", step=case["load_case"])[0]
    mx = info["maximum"]
    sumF = sum(np.linalg.norm(v) for v in step.reactions.values()) + np.linalg.norm(A["jF"][k, 1]) + np.linalg.norm(A["jF"][k, 2])
    mom = np.linalg.norm(np.asarray(step.remote["oko A"]["moment"]) - np.asarray(case["expected"]["oko A"]["moment"]))
    return [
        Check("okamžik max. reakce v čepu B = max |F_B| za otáčku", mx["value"],
              float(np.linalg.norm(A["jF"][:, 2], axis=1).max()), 1e-12, "N"),
        _zero("inertia relief: |R_MKP − R_MBD| v uložení / Σ|F|", rel["max_rel"], 1e-3),
        _zero("moment v uložení MKP − MBD / (Σ|F|·L)", mom / (sumF * (ROD_B - ROD_A)), 1e-3),
        Check("napětí v dříku σ = N/A (uprostřed, těžiště průřezu)", sig, N / (ROD_W * ROD_T), 2e-2, "MPa"),
        *_common(r),
    ]


T12 = 8


def mb12_build(wd: Path):
    st, parts = _geo_study("MB12 otáčející se reakce v oku ojnice – superpozice", wd, {"Ojnice": _rod_shape()})
    _crank_rod(st, parts, outputs=72)
    return st, parts


def mb12_run(st, wd, parts):
    from fea_core.motion.to_fea import history_loads, instant_loads
    from fea_core.solve import run_study
    from fea_core.study import BearingLoad, LoadCase
    res = _run(st, Path(wd) / "mbd", parts)
    hist = history_loads(st, res, parts=["Ojnice"])
    ks = [int(round(i * 72 / T12)) for i in range(T12)]
    direct = [instant_loads(st, res, k, parts=["Ojnice"]) for k in ks]
    # BearingLoad: unit states Fx, Fy (axes of the connecting rod) and the direct calculation at the moment with the oblique reaction
    A = res.arrays
    Fc = np.einsum("kji,kj->ki", A["rot"][:, 2], -A["jF"][:, 2])     # force of the piston on the connecting rod in the axes of the connecting rod
    ang = np.abs(np.degrees(np.arctan2(Fc[:, 1], Fc[:, 0])) % 180 - 45)
    kb = int(np.argmin(ang + 1e3 * (np.linalg.norm(Fc, axis=1) < 0.3 * np.linalg.norm(Fc, axis=1).max())))
    st.load_cases += [LoadCase("BL Fx", [BearingLoad("oko B", [1.0, 0.0, 0.0])]),
                      LoadCase("BL Fy", [BearingLoad("oko B", [0.0, 1.0, 0.0])]),
                      LoadCase("BL přímo", [BearingLoad("oko B", Fc[kb].tolist())])]
    fe = run_study(st, Path(wd) / "mkp", parts=parts)
    return _Multi({"mbd": res, "fe": fe, "hist": hist, "direct": direct, "kb": kb, "Fb": Fc[kb]})


# Expected number of unit states of MB12 derived from the DEFINITION of the model (not from the output of the program): planar motion in the plane
# xy (rotation axis z). The connecting rod has 2 pins, one (pin A) is the bearing → one joint goes into the states; its reaction has in the plane
# only Fx, Fy (Fz, Mx, My zero, Mz = the axis of the rotary joint = 0) → 2 states. Inertia: a0x, a0y (2), αz (1), and from the matrix
# W = ωωᵀ − |ω|²I with ω = (0, 0, ω) only Wxx = Wyy = −ω² (2; Wzz = 0, off-diagonal 0) → 5. Total 2 + 5.
N_STATES12 = (2 - 1) * 2 + (2 + 1 + 2)


def mb12_eval(res, st):
    from fea_core.motion.to_fea import superpose
    r, fe = res["mbd"], res["fe"]
    worst = 0.0
    for info in res["direct"]:
        Sd = fe.step(info["load_case"]).S
        Ss = superpose(fe, st, info["t"])
        worst = max(worst, float(np.abs(Ss - Sd).max() / np.abs(Sd).max()))
    Fb = res["Fb"]
    Sb = fe.step("BL přímo").S
    Sbs = Fb[0] * fe.step("BL Fx").S + Fb[1] * fe.step("BL Fy").S
    diff = float(np.abs(Sbs - Sb).max() / np.abs(Sb).max())
    return [
        _zero(f"superpozice jednotkových stavů (RemoteLoad RBE3) = přímý výpočet v {T12} polohách", worst, 1e-3),
        Check("počet jednotkových stavů = nenulové složky rovinného pohybu z definice modelu", res["hist"]["n_states"],
              N_STATES12, 1e-12, ""),
        AtLeast("BearingLoad: superpozice Fx, Fy ≠ přímý výpočet (není lineární) – rozdíl ≥ 5 %", diff, 0.05, 1e-12,
                ""),
        Check("kanály únavy = počet očekávaných jednotkových stavů", len(st.fatigue.history.channels), N_STATES12,
              1e-12, ""),
        *_common(r),
    ]


# MB13 lid on two coaxial hinges
LID_L, LID_W, LID_T, HX1, HX2, HR_O, HR_I, HL = 200.0, 100.0, 5.0, 25.0, 175.0, 7.0, 4.0, 12.0


def _lid_shape():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from verification.cases import box, cyl, solid
    s = box(0, 3, -LID_T / 2, LID_L, LID_W, LID_T)
    for x in (HX1, HX2):
        s = BRepAlgoAPI_Fuse(s, cyl((x - HL / 2, 0, 0), (1, 0, 0), HR_O, HL)).Shape()
    for x in (HX1, HX2):
        s = BRepAlgoAPI_Cut(s, cyl((x - 20, 0, 0), (1, 0, 0), HR_I, 40)).Shape()
    return solid(s)


def mb13_build(wd: Path):
    from fea_core.selectors import find_faces
    st, parts = _geo_study("MB13 víko na dvou pantech", wd, {"Víko": _lid_shape()}, size=5.0)
    st.mesh.curvature_points = 8          # few nodes at the holes of the hinges (dense equation of the distributing constraint in ccx)
    for k, x in enumerate((HX1, HX2), 1):
        st.add_selection(f"pant {k}", find_faces(parts, part="Víko", type="cylinder", radius=HR_I,
                                                 where=lambda p, x=x: abs(p["centroid"][0] - x) < 1.0))
    M = st.mechanism
    M.gravity = [0, 0, -G]
    M.bodies = [Body("rám", ground=True), Body("víko", parts=["Víko"])]
    M.joints = [MotionJoint("pant 1", "revolute", "rám", "víko", [HX1, 0, 0], [1, 0, 0], selection_b="pant 1"),
                MotionJoint("pant 2", "revolute", "rám", "víko", [HX2, 0, 0], [1, 0, 0], selection_b="pant 2")]
    M.drivers = [Driver("pant 1", "angle", "sine", [0.5, 0.5, 0.0, 0.0], name="otevírání")]
    M.run = MotionRun("kinematic", t_end=1.0, outputs=20, to_fea=ToFea(support_joints=["pant 1", "pant 2"]))
    return st, parts


def mb13_run(st, wd, parts):
    from fea_core.motion.run import check_dof
    from fea_core.motion.to_fea import instant_loads
    from fea_core.solve import run_study
    dof = check_dof(st, parts, workdir=wd)
    res = _run(st, Path(wd) / "mbd", parts)
    info = instant_loads(st, res, 10, parts=["Víko"], support_joints=["pant 1", "pant 2"])
    fe = run_study(st, Path(wd) / "mkp", parts=parts)
    return _Multi({"dof": dof, "mbd": res, "fe": fe, "info": info})


def mb13_eval(res, st):
    r, fe, info, dof = res["mbd"], res["fe"], res["info"], res["dof"]
    A = r.arrays
    k = info["index"]
    B = r.meta["bodies"][1]
    g = np.array([0, 0, -G])
    need = B["mass"] * (A["acc"][k, 1] - g)               # Σ of the reactions of the hinges on the lid = m·(a − g)
    sum_mbd = A["jF"][k, 0] + A["jF"][k, 1]
    step = fe.step(info["load_case"])
    R = A["rot"][k, 1]
    f1 = np.asarray(step.remote["pant 1"]["force"])
    f2 = np.asarray(step.remote["pant 2"]["force"])
    sum_fe = R @ (f1 + f2)                                  # into the global axes
    # distribution of the vertical force hinge 1 / total is statically indeterminate (it depends on the stiffness) → informative only, not a check
    print(f"  [info] MB13 podíl svislé síly pantu 1 z celku: {(R @ f1)[2] / sum_mbd[2]:.3f} (orientačně 0,5)")
    return [
        Check("Grübler 6·1 − 2·5", dof["grubler"], -4, 1e-12, ""),
        Check("skutečná volnost", dof["dof_free"], 1, 1e-12, ""),
        Check("redundantní vazby (2 souosé rotační klouby)", dof["redundant"], 5, 1e-12, ""),
        Check("hlášeno „reakce NEJEDNOZNAČNÉ“", 1.0 + any("AMBIGUOUS" in m for m in dof["messages"]), 2.0, 1e-12, ""),
        _zero("MBD: součet reakcí pantů − m·(a − g) (rel.)", np.linalg.norm(sum_mbd - need) / np.linalg.norm(need), 1e-6),
        _zero("MKP: součet reakcí obou pantů = součet MBD (rel.)", np.linalg.norm(sum_fe - sum_mbd) / np.linalg.norm(sum_mbd),
              1e-3),
        # hinge 2 has the rotation about the axis free (a rotary joint) → the moment about the axis is carried only by hinge 1 = the drive moment from MBD
        Check("MKP: moment k ose pantů nese pant 1 = moment pohonu MBD",
              (R @ np.asarray(step.remote["pant 1"]["moment"]))[0], (A["jM"][k, 0] + A["jM"][k, 1])[0], 1e-3, "N·mm"),
        *_common(r),
    ]


# MB14 crank against a stop – collision check
ARM_L, ARM_H, W14 = 100.0, 5.0, 1.0
STOP = (50.0, 30.0, 60.0, 60.0)                           # x0, y0, x1, y1 dorazu


def mb14_build(wd: Path):
    from verification.cases import box
    x0, y0, x1, y1 = STOP
    st, parts = _geo_study("MB14 klika proti dorazu – kolize", wd,
                           {"Klika": box(-10, -ARM_H, -5, ARM_L + 10, 2 * ARM_H, 10),
                            "Doraz": box(x0, y0, -20, x1 - x0, y1 - y0, 40)})
    M = st.mechanism
    M.gravity = None
    M.bodies = [Body("doraz", ground=True, parts=["Doraz"]), Body("klika", parts=["Klika"])]
    M.joints = [MotionJoint("čep", "revolute", "doraz", "klika", [0, 0, 0], [0, 0, 1])]
    M.drivers = [Driver("čep", "angular_velocity", "constant", [W14], name="motor")]
    M.run = MotionRun("kinematic", t_end=0.6, outputs=60, collisions=[["klika", "doraz"]])
    return st, parts


def mb14_eval(res, st):
    x0, y0, x1, y1 = STOP
    phis = []
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        rho = math.hypot(x, y)
        phis.append(math.atan2(y, x) - math.asin(ARM_H / rho))
    tc = min(phis) / W14
    col = res.meta["collisions"]
    p = col["pairs"][0]
    d0 = p["distance"][0]
    # distance at t = 0: the corner of the stop (x1, y0) from the upper face of the crank y = 5
    return [
        Check("čas první kolize φ_c/ω (roh dorazu na boku kliky)", col["first"]["t"], tc, 1e-4, "s"),
        Check("vzdálenost v t = 0 (roh 60, 30 od boku y = 5)", d0, y0 - ARM_H, 1e-6, "mm"),
        _zero("po kolizi vzdálenost 0", p["distance"][-1], 1e-9, "mm"),
        Check("kolize nahlášena ve varováních", 1.0 + any("Collision" in w for w in res.warnings), 2.0, 1e-12, ""),
    ]


# MB10 flexible arm (Craig–Bampton, Exudyn FFRF): cantilever 200 × 20 × 10 clamped by a rigid joint to the frame
ARM10 = (200.0, 20.0, 10.0)


def mb10_build(wd: Path):
    from verification.cases import box
    from fea_core.selectors import find_faces
    from fea_core.study import Fixed, Gravity, LoadCase
    L, B, H = ARM10
    st, parts = _geo_study("MB10 pružné rameno", wd, {"Rameno": box(0, 0, 0, L, B, H)}, size=5.0)
    st.add_selection("čelo", find_faces(parts, x=0))
    M = st.mechanism
    M.gravity = [0, 0, -G]
    M.bodies = [Body("rám", ground=True), Body("rameno", parts=["Rameno"], flexible=True, modes=8)]
    M.joints = [MotionJoint("vetknutí", "fixed", "rám", "rameno", [0, B / 2, H / 2], [1, 0, 0], selection_b="čelo")]
    M.run = MotionRun("dynamic", t_end=0.048, outputs=960, dt=2.5e-5, spectral_radius=1.0)
    # FEM of the same geometry: clamping of the end face, self-weight and natural frequencies (ccx)
    st.constraints = [Fixed("čelo")]
    st.load_cases = [LoadCase("tíha", [Gravity([0, 0, -G])])]
    st.modal.enabled = True
    st.modal.modes = 4
    return st, parts


def mb10_run(st, wd, parts):
    from fea_core.solve import run_study
    res = _run(st, Path(wd) / "mbd", parts)
    fe = run_study(st, Path(wd) / "mkp", parts=parts)
    return _Multi({"mbd": res, "fe": fe})


def mb10_eval(res, st):
    from fea_core.motion import flex
    r, fe = res["mbd"], res["fe"]
    L, B, H = ARM10
    f_mbd = r.meta["flex"][0]["freq_hz"]
    f_ccx = fe.modes[0].values
    tip = np.array([L, B / 2, H / 2])
    nodes = r.arrays["flex1_nodes"]
    kt = int(np.argmin(np.linalg.norm(nodes - tip, axis=1)))
    uz = np.array([flex.displacement(r, "rameno", k)[kt, 2] for k in range(len(r.arrays["t"]))])
    t = r.arrays["t"]
    # centre of the oscillation = static deflection (without damping: average over whole periods), frequency from the passages through the centre
    T1 = 1.0 / f_mbd[0]
    n_per = int(t[-1] / T1)
    sel = t <= n_per * T1
    mean = np.trapz(uz[sel], t[sel]) / (n_per * T1)
    x = uz - mean
    zc = [t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i]) for i in range(len(x) - 1) if x[i] > 0 >= x[i + 1]]
    f_dyn = 1.0 / float(np.diff(zc).mean())
    st_fe = fe.step(0)
    uz_ccx = float(fe.probe(tip, "U", step=0)[2])
    I = B * H ** 3 / 12
    w = RHO * B * H * G                                   # N/mm
    uz_eb = -w * L ** 4 / (8 * E * I)
    pr = np.array([20.0, B / 2, H])                       # upper face near the clamping (away from the singular corners)
    kr = int(np.argmin(np.linalg.norm(nodes - pr, axis=1)))
    sxx = np.array([flex.stress(r, "rameno", k)[kr, 0] for k in range(len(t))])
    sxx_mean = np.trapz(sxx[sel], t[sel]) / (n_per * T1)
    sxx_ccx = float(fe.probe(nodes[kr], "S", step=0)[0])
    del st_fe
    return [
        Check("f₁ Craig–Bampton (rozhraní vetknuté) = ccx *FREQUENCY na téže geometrii", f_mbd[0], f_ccx[0], 5e-3, "Hz"),
        Check("f₂ Craig–Bampton = ccx", f_mbd[1], f_ccx[1], 5e-3, "Hz"),
        Check("f₁ proti Euler–Bernoulli 1,8751²/(2πL²)·√(EI/ρA)", f_mbd[0],
              1.8751 ** 2 / (2 * math.pi * L ** 2) * math.sqrt(E * I / (RHO * B * H)), 1e-2, "Hz"),
        Check("dynamika: kmit konce pod vlastní tíhou má frekvenci f₁", f_dyn, f_mbd[0], 5e-3, "Hz"),
        Check("dynamika: střed kmitu konce = statický průhyb ccx (téže síť, tíha)", mean, uz_ccx, 2e-2, "mm"),
        Check("statický průhyb ccx proti Euler–Bernoulli qL⁴/8EI", uz_ccx, uz_eb, 2e-2, "mm"),
        Check("max. průhyb konce ≈ 2 × statický (náhlé zatížení tíhou)", uz.min(), 2 * uz_ccx, 3e-2, "mm"),
        Check("napětí z tvarů: střed kmitu σ_xx u vetknutí (x 20, horní plocha) = ccx statické", sxx_mean, sxx_ccx,
              3e-2, "MPa"),
    ]


# ------------------------------------------------------------------ contact in MBD – Chrono solver (F8.9 C1)
def _chrono_run(st, wd, parts):
    from fea_core.motion.chrono_run import available, INSTALL_HINT
    if not available():
        raise RuntimeError(INSTALL_HINT)
    return _run(st, wd, parts)


def _mb4c_build(wd: Path, mu: float, deflection: float = 0.02, dt: float = 2.5e-4):
    """Cylinder R 50 × 20 (a part from CAD, steel) on a plate, the weight inclined by 30° (= inclined plane), contact
    of the triangle meshes in Chrono (NSC, hard contact with Coulomb friction) – without a rolling constraint.
    """
    from verification.cases import box, cyl
    st, parts = _geo_study(f"MB4C valení / skluz válce – kontakt Chrono, μ {mu:g}", wd,
                           {"Válec": cyl((0, -10, R4), (0, 1, 0), R4, 20), "Deska": box(-100, -40, -10, 500, 80, 10)})
    M = st.mechanism
    M.gravity = [G * math.sin(ALPHA4), 0, -G * math.cos(ALPHA4)]
    M.bodies = [Body("deska", parts=["Deska"], ground=True), Body("válec", parts=["Válec"])]
    M.contacts = [MotionContact("válec", "deska", mu=mu, deflection=deflection)]
    M.run = MotionRun("dynamic", t_end=0.2, outputs=100, dt=dt, solver="chrono")
    return st, parts


def _mb4c_eval(res, st, mu):
    A = res.arrays
    t, x = A["t"], A["pos"][:, 1, 0]
    sel = t > 0.05
    a = 2 * np.polyfit(t[sel], x[sel], 2)[0] * 1e-3                     # m/s²
    g = G * 1e-3
    om = A["omega"][:, 1, 1]
    al = 2 * np.polyfit(t[sel], np.polyfit(t[sel], om[sel], 1)[0] * t[sel] ** 2 / 2, 2)[0]
    out = [_zero("rovnováha válce s kontaktní silou Chrono (Euler implicitní – chyba úměrná h, rel.)",
                 res.meta["equilibrium"]["max_rel"], 5e-2)]
    if mu < math.tan(ALPHA4) / 3:
        a_ref = g * (math.sin(ALPHA4) - mu * math.cos(ALPHA4))
        al_ref = 2 * mu * g * math.cos(ALPHA4) / (R4 * 1e-3)          # μ·m·g·cos α·R / (m R²/2)
        out += [Check(f"skluz μ {mu:g}: a = g(sin α − μ cos α) (zadání 4,055)", a, a_ref, 1e-3, "m/s²"),
                Check("skluz: úhlové zrychlení ε = 2μg cos α / R (mnohoúhelník triangulace, Euler)", al, al_ref, 5e-2,
                      "rad/s²")]
    else:
        a_ref = 2 / 3 * g * math.sin(ALPHA4)
        v, w = A["vel"][:, 1, 0][-1], A["omega"][:, 1, 1][-1]
        out += [Check(f"valení μ {mu:g}: a = 2/3·g·sin α (zadání 3,270; Euler implicitní tlumí ~h)", a, a_ref, 1e-2,
                      "m/s²"),
                _zero("valení bez skluzu: |v − ω·R| / v na konci", abs(v - w * R4) / abs(v), 1e-2)]
    return out


def mb4c_build(wd: Path):
    return _mb4c_build(wd, 0.1)


def mb4c_eval(res, st):
    return _mb4c_eval(res, st, 0.1)


def mb4r_build(wd: Path):
    return _mb4c_build(wd, 0.3, deflection=0.005, dt=1e-4)


def mb4r_eval(res, st):
    return _mb4c_eval(res, st, 0.3)


E5, RC5, RF5, W5, F05 = 5.0, 30.0, 10.0, 2 * math.pi, 20.0


def kk5_s(theta):
    """Lift of the roller of the eccentric cam from the lowest position (KK1, = crank mechanism e, R_c + r_f)."""
    L = RC5 + RF5
    return E5 * math.cos(theta) + math.sqrt(L * L - (E5 * math.sin(theta)) ** 2) - (L - E5)


def kk5_build(wd: Path):
    """Eccentric cam (disc R 30, centre 5 mm from the axis, from CAD) turned at 1 rev/s, roller R 10 (fixed on the lifter,
    μ = 0) in a straight guide pressed by a force of 20 N; contact of the triangle meshes in Chrono.
    """
    from verification.cases import cyl
    st, parts = _geo_study("KK5 vačka z CAD v MBD – kontakt Chrono", wd,
                           {"Vačka": cyl((0, E5, 0), (0, 0, 1), RC5, 10),
                            "Kladka": cyl((0, E5 + RC5 + RF5, 2), (0, 0, 1), RF5, 6)})
    M = st.mechanism
    M.gravity = None
    y0 = E5 + RC5 + RF5
    M.bodies = [Body("rám", ground=True), Body("vačka", parts=["Vačka"], w0=[0, 0, W5]),
                Body("zvedák", parts=["Kladka"])]
    M.joints = [MotionJoint("čep vačky", "revolute", "rám", "vačka", [0, 0, 5], [0, 0, 1]),
                MotionJoint("vedení", "prismatic", "rám", "zvedák", [0, y0, 5], [0, 1, 0])]
    M.drivers = [Driver("čep vačky", "angular_velocity", "constant", [W5])]
    M.loads = [MotionLoad("zvedák", [0, y0, 5], [0, -F05, 0], name="přítlak")]
    M.contacts = [MotionContact("vačka", "zvedák", mu=0.0, deflection=0.002)]
    M.run = MotionRun("dynamic", t_end=0.25, outputs=90, dt=2.5e-4, solver="chrono")
    return st, parts


def kk5_eval(res, st):
    A = res.arrays
    t = A["t"]
    y = A["pos"][:, 2, 1]
    y0 = E5 + RC5 + RF5
    out = []
    for deg in (60, 90):
        k = int(np.argmin(np.abs(t - math.radians(deg) / W5)))
        th = W5 * t[k]
        out.append(Check(f"zdvih s({deg}°) = e cos θ + √((R_c + r_f)² − e² sin² θ) − (R_c + r_f − e)",
                         y[k] - (y0 - 2 * E5), kk5_s(th), 1e-3, "mm"))
    # normal force: centre over 60–90° (triangulation polygon – the normal jumps by the angle of the facet)
    sel = (t >= math.radians(60) / W5 - 1e-9) & (t <= math.radians(90) / W5 + 1e-9)
    phi = np.arcsin(E5 * np.sin(W5 * t[sel]) / (RC5 + RF5))
    m = res.meta["bodies"][2]["mass"]
    N_ref = (F05 + m * A["acc"][sel, 2, 1]) / np.cos(phi)              # t·mm/s² = N
    N = np.linalg.norm(A["cF"][sel, 2], axis=1)
    out += [Check("normálová síla 60–90° (průměr): N = (F₀ + m·a)/cos φ", float(N.mean()), float(N_ref.mean()), 1e-2,
                  "N"),
            _zero("rovnováha těles s kontaktní silou Chrono (Euler implicitní – chyba úměrná h, rel.)",
                  res.meta["equilibrium"]["max_rel"], 5e-2)]
    return out


MOTION_CASES: dict[str, Case] = {
    "MB1": Case("MB1", "Klika r 20, l 80, ω 100 rad/s – kinematika a moment pohonu", mb1_build, mb1_eval, run=_run),
    "MB2": Case("MB2", "Čtyřkloub 40/120/80/100 – Freudenstein, obě větve, sestavení", mb2_build, mb2_eval, run=mb2_run),
    "MB3": Case("MB3", "Kyvadlo L 0,5 m, θ₀ 60° – perioda, síla v čepu, energie", mb3_build, mb3_eval, run=_run),
    "MB3b": Case("MB3b", "Kyvadlo s čepovým třením μ·|R|·r", mb3b_build, mb3b_eval, run=_run),
    "MB4": Case("MB4", "Valení válce po rovině 30° (vazba valení)", mb4_build, mb4_eval, run=_run),
    "MB5": Case("MB5", "Doběh kvádru v₀ 1 m/s, μ 0,2", mb5_build, mb5_eval, run=_run),
    "MB6": Case("MB6", "Stick-slip na pásu (μs 0,4, μk 0,3)", mb6_build, mb6_eval, run=_run),
    "MB7": Case("MB7", "Klika s čepovým třením – výkonová bilance", mb7_build, mb7_eval, run=_run),
    "MB8": Case("MB8", "Prostorový 4R – volnost a redundantní vazby", mb8_build, mb8_eval, run=mb8_run),
    "MB9": Case("MB9", "Ozubený převod a šroubový kloub", mb9_build, mb9_eval, run=_run),
    "MB10": Case("MB10", "Pružné rameno Craig–Bampton (Exudyn FFRF) proti ccx a analytice", mb10_build, mb10_eval,
                 run=mb10_run),
    "MB11": Case("MB11", "Klika → MKP v okamžiku max. reakce (inertia relief)", mb11_build, mb11_eval, run=mb11_run),
    "MB12": Case("MB12", "Otáčející se reakce v oku ojnice – superpozice jednotkových stavů", mb12_build, mb12_eval,
                 run=mb12_run),
    "MB13": Case("MB13", "Víko na dvou souosých pantech – redundance, rozdělení v MKP", mb13_build, mb13_eval,
                 run=mb13_run),
    "MB14": Case("MB14", "Klika proti dorazu – kontrola kolizí bez kontaktu", mb14_build, mb14_eval, run=_run),
    "MB15": Case("MB15", "Rozběh momentem a S-křivka (D6)", mb15_build, mb15_eval, run=mb15_run),
    "MB16": Case("MB16", "Kyvadlo na dorazu s restitucí (D6)", mb16_build, mb16_eval, run=_run),
    "MB4C": Case("MB4C", "Skluz válce po rovině 30°, μ 0,1 – kontakt sítí v Chrono (C1)", mb4c_build, mb4c_eval,
                 run=_chrono_run),
    "MB4R": Case("MB4R", "Valení válce po rovině 30°, μ 0,3 – kontakt sítí v Chrono (C1)", mb4r_build, mb4r_eval,
                 run=_chrono_run),
    "KK5": Case("KK5", "Excentrická vačka z CAD v MBD – kontakt sítí v Chrono (C1)", kk5_build, kk5_eval,
                run=_chrono_run),
}
