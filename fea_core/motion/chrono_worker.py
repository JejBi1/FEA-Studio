"""Working process of Chrono: <tools/chrono_env/python> -m fea_core.motion.chrono_worker JOB  (PYTHONPATH = project root)

The same input (JOB/job.json in SI) and output (JOB/result.npz with the same fields as fea_core.motion.worker, result.json) as the
worker of Exudyn, in addition the contact of the triangular meshes (job["contacts"], shapes JOB/tri_<i>.npz) and the fields cF, cM = the resultant
contact force on the body and the moment about the centroid (check of the equilibrium). The conda environment has neither scipy nor OCP – only numpy,
pychrono and fea_core.motion.laws.

Verified conventions (scratch/f8_motion/ch_proto2.py, ch_api*.py, dbg_kk5.py, Chrono 10.0):
  - link.Initialize(body_b, body_a, frame) → body1 = body_b; GetReaction2() = force and moment on body2 (body_a) in frame 2
    with the point of application at the ORIGIN OF FRAME 2 → on body_b acts −R₂·force, −R₂·torque about the origin of frame 2 (sliding joint: frame 1
    moves away with the body – the moment about it did not agree by F × displacement);
  - motor ChLinkMotorRotation*: angle / moment = rotation of body1 with respect to body2 about the z of the frame (= the coordinate of the joint of FEA);
    the linear motor ChLinkMotorLinear* moves along the z of the frame; the force motor is not a constraint → missing in the reaction (it is added);
  - ChLinkTSDA: GetForce > 0 = pressure (apart), SetActuatorForce likewise; GetContactableForce/Torque = global axes,
    moment about the centroid (= the origin of the body);
  - the time step Euler implicit linearised: the reactions and the accelerations consistent only up to the order of h (forces in the position at the start
    of the step; pendulum dt 2e-5 → 8e-5, contact of a polygon dt 2.5e-4 → 2 %); HHT the reaction shifts by α and with the contact NSC
    it does not work. Without contact the direct solver SparseQR (PSOR is inaccurate for joints), with contact NSC Barzilai–Borwein;
  - contact: a triangle mesh with the envelope r (sphere-swept, vertices shifted inward by r) – without the envelope (GIMPACT) a
    cam stuck after ~50° and penetrated; a convex hull with more than ~200 points does not detect the contact.
The kinematic task = dynamics with motors on all the degrees of freedom (Chrono DoStepDynamics; the reactions including the inertia).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .laws import Law, DriverMotion, KINEMATIC_QUANTITIES

ARRAY_KEYS = ("pos", "rot", "vel", "omega", "acc", "alpha", "jF", "jM", "jP", "jPa", "jFl", "jMl", "jq", "jv", "jfr",
              "jlim", "jQ", "dQ", "dq", "dv", "fval", "flen", "fF", "fM", "fPa", "fPb", "lF", "lM", "lP", "cF", "cM")
WORKS = ("wdrv", "wfr", "wlim", "wdamp", "wbush", "wload")
POWER_KEYS = ("pdrv", "pfr", "plim", "pdamp", "pbush", "pload", "pelem")
DYNAMIC_KEYS = ("acc", "alpha", "jF", "jM", "jFl", "jMl", "jQ", "dQ", "cF", "cM")
UNSUPPORTED = {"gear": "gear pair", "rack_pinion": "rack and pinion", "pin_slot": "pin in slot"}


class ChronoError(RuntimeError):
    pass


def log(*a):
    print(*a, flush=True)


def _v(x) -> np.ndarray:
    return np.array([x.x, x.y, x.z], float)


def _R(m) -> np.ndarray:
    """ChMatrix33d → numpy (columns = axes)."""
    return np.column_stack([_v(m.GetAxisX()), _v(m.GetAxisY()), _v(m.GetAxisZ())])


def _quat(c, A: np.ndarray):
    m = c.ChMatrix33d()
    m.SetFromDirectionAxes(c.ChVector3d(*A[:, 0]), c.ChVector3d(*A[:, 1]), c.ChVector3d(*A[:, 2]))
    return m.GetQuaternion()


def joint_frame(axis, axis2=None) -> np.ndarray:
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    x = None
    if axis2 is not None:
        x = np.asarray(axis2, float) - np.dot(axis2, z) * z
        if np.linalg.norm(x) < 1e-9:
            x = None
    if x is None:
        x = np.cross([0.0, 1.0, 0.0], z) if abs(z[1]) < 0.9 else np.cross([1.0, 0.0, 0.0], z)
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class ChronoModel:
    def __init__(self, job: dict):
        import pychrono as c
        self.c, self.job = c, job
        self.kind = job.get("kind", "dynamic")
        contacts = job.get("contacts") or []
        smc = any(ct.get("method") == "smc" for ct in contacts)
        self.sys = c.ChSystemSMC() if smc else c.ChSystemNSC()
        self.smc = smc
        self.sys.SetCollisionSystemType(c.ChCollisionSystem.Type_BULLET)
        g = job.get("gravity")
        self.g = np.asarray(g, float) if g is not None else np.zeros(3)
        self.sys.SetGravitationalAcceleration(c.ChVector3d(*self.g))
        ch = job.get("chrono", {})
        env = float(os.environ.get("FEA_CHRONO_ENVELOPE") or ch.get("envelope", 1e-3))
        margin = float(os.environ.get("FEA_CHRONO_MARGIN") or ch.get("margin", env * 0.5))
        c.ChCollisionModel.SetDefaultSuggestedEnvelope(env)
        c.ChCollisionModel.SetDefaultSuggestedMargin(margin)
        if contacts and not smc:
            # complementarity of the contact NSC → iterative solver (Barzilai–Borwein)
            self.sys.SetSolverType(c.ChSolver.Type_BARZILAIBORWEIN)
            self.sys.GetSolver().AsIterative().SetMaxIterations(int(job.get("chrono", {}).get("iterations", 300)))
        else:
            # only constraints (+ penalty contact SMC) → direct solver (the default PSOR is inaccurate for joints)
            self.sys.SetSolver(c.ChSolverSparseQR())
        self.t_shift = 0.0
        self.notes: list[str] = []
        if self.kind == "static":
            raise ChronoError("The Chrono solver computes only dynamic (and kinematic) analyses – quasi-static runs use Exudyn")
        self._bodies()
        self._contacts(contacts)
        self._joints()
        self._forces()
        self._loads()
        has_v0 = any(b.get("v0") or b.get("w0") for b in self.job["bodies"])
        for d in self.drivers:
            if d["kin"] and not has_v0 and abs(DriverMotion(d["quantity"], d["law"]).rate(0.0)) > 1e-12:
                self.notes.append("The drive has a non-zero velocity at time 0 – Chrono matches it in the first step (impact); set the initial body velocities (Body.v0, w0) or a ramp-up (S-curve)")
                break

    # ---------------------------------------------------------------- bodies
    def _bodies(self):
        c = self.c
        self.ground = c.ChBody()
        self.ground.SetFixed(True)
        self.sys.Add(self.ground)
        self.B = []
        for b in self.job["bodies"]:
            com = np.asarray(b.get("com") or [0, 0, 0], float)
            if b.get("flexible"):
                raise ChronoError(f"Flexible body '{b['name']}' is supported only by the Exudyn solver")
            if b.get("ground"):
                body = c.ChBody()
                body.SetFixed(True)
                body.SetPos(c.ChVector3d(*com))
                self.sys.Add(body)
                self.B.append({"name": b["name"], "ground": True, "body": body, "com": com, "mass": 0.0,
                               "J": np.zeros((3, 3))})
                continue
            m = float(b["mass"])
            J = np.asarray(b["inertia"], float).reshape(3, 3)
            body = c.ChBody()
            body.SetMass(m)
            body.SetInertiaXX(c.ChVector3d(J[0, 0], J[1, 1], J[2, 2]))
            body.SetInertiaXY(c.ChVector3d(J[0, 1], J[0, 2], J[1, 2]))
            body.SetPos(c.ChVector3d(*com))
            if b.get("v0"):
                body.SetPosDt(c.ChVector3d(*map(float, b["v0"])))
            if b.get("w0"):
                body.SetAngVelParent(c.ChVector3d(*map(float, b["w0"])))
            self.sys.Add(body)
            self.B.append({"name": b["name"], "ground": False, "body": body, "com": com, "mass": m, "J": J})

    def body(self, i):
        return self.ground if i is None or i < 0 else self.B[i]["body"]

    def body_ground(self, i):
        return i is None or i < 0 or self.B[i]["ground"]

    # ---------------------------------------------------------------- kontakty
    def _contacts(self, contacts):
        c = self.c
        mats: dict[int, object] = {}
        for ct in contacts:
            for i in (ct["a"], ct["b"]):
                if i in mats:
                    continue
                if self.smc:
                    mt = c.ChContactMaterialSMC()
                    mt.SetYoungModulus(float(ct.get("young", 2.1e11)))
                    mt.SetPoissonRatio(0.3)
                else:
                    mt = c.ChContactMaterialNSC()
                    if ct.get("rolling"):
                        mt.SetRollingFriction(float(ct["rolling"]))
                mt.SetFriction(float(ct.get("mu", 0.3)))
                mt.SetRestitution(float(ct.get("restitution", 0.0)))
                mats[i] = mt
                d = np.load(self.job["tri"][str(i)])
                V, F = d["v"], d["f"]
                body = self.body(i)
                fixed = self.body_ground(i)
                # triangle mesh (the convex hull ChCollisionShapeConvexHull with more than ~200 points the contact in Chrono 10
                # does not detect – scratch/f8_motion/ch_hull*.py)
                r = float(os.environ.get("FEA_CHRONO_RADIUS") or self.job.get("chrono", {}).get("radius", 0.0))
                V, F = weld(V, F)
                if r > 0:
                    V = V - r * vertex_normals(V, F)          # envelope r around the mesh → the surface back to the shape from the CAD
                tm = c.ChTriangleMeshConnected()
                for t in F:
                    p = [c.ChVector3d(*map(float, V[k])) for k in t]
                    tm.AddTriangle(p[0], p[1], p[2])
                body.AddCollisionShape(c.ChCollisionShapeTriangleMesh(mt, tm, bool(fixed), False, r))
                body.EnableCollision(True)
        self.contact_bodies = sorted(mats)
        # only the given pairs: families of collisions (Chrono has 15) – a body = a family, the pairs not given do not collide
        pairs = {frozenset((ct["a"], ct["b"])) for ct in contacts}
        if len(self.contact_bodies) <= 15:
            fam = {i: k for k, i in enumerate(self.contact_bodies)}
            for i in self.contact_bodies:
                cm = self.body(i).GetCollisionModel()
                cm.SetFamily(fam[i])
                for j in self.contact_bodies:
                    if j != i and frozenset((i, j)) not in pairs:
                        cm.DisallowCollisionsWith(fam[j])
        elif len(pairs) < len(self.contact_bodies) * (len(self.contact_bodies) - 1) // 2:
            self.notes.append("More than 15 bodies with contact – contact is searched between all their pairs")
        self.materials_note(contacts)

    def materials_note(self, contacts):
        """The material of the contact is a property of the body in Chrono (a pair: friction = minimum) → a warning of the inconsistency."""
        seen = {}
        for ct in contacts:
            for i in (ct["a"], ct["b"]):
                key = (ct.get("mu"), ct.get("restitution"), ct.get("rolling"), ct.get("young"))
                if i in seen and seen[i] != key:
                    self.notes.append(f"Body '{(self.B[i]['name'] if i >= 0 else 'ground')}' is in several contacts with different parameters – Chrono uses the first one (the material belongs to the body, a pair takes the lower friction)")
                seen.setdefault(i, key)

    # ---------------------------------------------------------------- joints and drives
    def _joints(self):
        c = self.c
        self.J = []
        drv = {}
        for d in self.job.get("drivers", []):
            drv.setdefault(d["joint"], []).append(d)
        self.drivers = []
        for k, j in enumerate(self.job["joints"]):
            kind = j["kind"]
            if kind in UNSUPPORTED:
                raise ChronoError(f"Joint '{j['name']}' ({UNSUPPORTED[kind]}) is supported only by the Exudyn solver")
            if j.get("friction") or j.get("limits") is not None:
                raise ChronoError(f"Joint '{j['name']}': joint friction and stops are supported only by Exudyn (in Chrono use shape contact)")
            a, b = j.get("a", -1), j.get("b", -1)
            ba, bb = self.body(a), self.body(b)
            rec = {"name": j["name"], "kind": kind, "a": a, "b": b, "motor": None, "coord": None}
            A = joint_frame(j.get("axis") or [0, 0, 1], j.get("axis2"))
            p = np.asarray(j.get("origin") or [0, 0, 0], float)
            ds = drv.get(k, [])
            if len(ds) > 1:
                raise ChronoError(f"Joint '{j['name']}': only one drive per joint in Chrono")
            dd = ds[0] if ds else None
            q = dd["quantity"] if dd else None
            if kind == "distance":
                pa, pb = np.asarray(j["point_a"], float), np.asarray(j["point_b"], float)
                link = c.ChLinkDistance()
                link.Initialize(bb, ba, False, c.ChVector3d(*pb), c.ChVector3d(*pa))
                A = joint_frame(pb - pa)
                p = pb
            elif kind in ("revolute", "cylindrical", "screw") and dd and q in ("angle", "angular_velocity", "torque"):
                if kind != "revolute":
                    raise ChronoError(f"Joint '{j['name']}': a rotation drive in Chrono only on a revolute joint")
                # kinematic driver = prescribed rotation (the velocity is integrated – DriverMotion, as Exudyn)
                link = c.ChLinkMotorRotationTorque() if q == "torque" else c.ChLinkMotorRotationAngle()
                rec["coord"] = "rot"
            elif kind == "prismatic" and dd and q in ("disp", "velocity", "force"):
                # linear motor of Chrono 10 moves along the axis z of the frame (verified scratch/f8_motion/ch_api.py)
                link = c.ChLinkMotorLinearForce() if q == "force" else c.ChLinkMotorLinearPosition()
                rec["coord"] = "trans"
            elif dd:
                raise ChronoError(f"Joint drive '{j['name']}' ({kind}, {q}) is not possible in Chrono")
            else:
                link = {"fixed": c.ChLinkLockLock, "revolute": c.ChLinkLockRevolute,
                        "prismatic": c.ChLinkLockPrismatic, "cylindrical": c.ChLinkLockCylindrical,
                        "spherical": c.ChLinkLockSpherical, "planar": c.ChLinkLockPlanar,
                        "screw": c.ChLinkLockScrew, "universal": c.ChLinkUniversal}[kind]()
                if kind == "screw":
                    link.SetThread(float(j["lead"]))
            if kind != "distance":
                link.Initialize(bb, ba, c.ChFramed(c.ChVector3d(*p), _quat(c, A)))
            self.sys.Add(link)
            rec.update(link=link, frame=A, p=p)
            if dd:
                law = Law(dd["law"], dd["data"], dd.get("scale", 1.0))
                fn = _make_function(c, self, q, law)
                rec["fn"] = fn
                if q == "force":
                    link.SetForceFunction(fn)
                elif q == "torque":
                    link.SetTorqueFunction(fn)
                elif rec["coord"] == "trans":
                    link.SetMotionFunction(fn)
                else:
                    link.SetAngleFunction(fn)
                rec["motor"] = q
                self.drivers.append({"joint": k, "coord": rec["coord"], "quantity": q, "law": law,
                                     "kin": q in KINEMATIC_QUANTITIES})
            self.J.append(rec)
        # order of the drivers as in the worker of Exudyn: kinematic, then force
        self.drivers.sort(key=lambda d: not d["kin"])

    # ---------------------------------------------------------------- force elements and loads
    def _forces(self):
        c = self.c
        self.F = []
        for f in self.job.get("forces", []):
            if f["kind"] not in ("spring", "damper", "spring_damper"):
                raise ChronoError(f"Force element '{f['name']}' ({f['kind']}) is supported only by the Exudyn solver")
            a, b = f.get("a", -1), f.get("b", -1)
            pa, pb = np.asarray(f["point_a"], float), np.asarray(f["point_b"], float)
            L0 = float(np.linalg.norm(pb - pa)) if f.get("l0") is None else float(f["l0"])
            t = c.ChLinkTSDA()
            t.Initialize(self.body(b), self.body(a), False, c.ChVector3d(*pb), c.ChVector3d(*pa))
            t.SetRestLength(L0)
            t.SetSpringCoefficient(float(f.get("k", 0.0)) if f["kind"] != "damper" else 0.0)
            t.SetDampingCoefficient(float(f.get("c", 0.0)) if f["kind"] != "spring" else 0.0)
            t.SetActuatorForce(-float(f.get("f0", 0.0)))      # Chrono: positive force = pushes apart
            self.sys.Add(t)
            self.F.append({**f, "tsda": t, "lref": L0})

    def _loads(self):
        self.L = []
        for ld in self.job.get("loads", []):
            b = ld["body"]
            tab = ld.get("table")
            self.L.append({**ld, "law": Law("table", tab) if tab else None,
                           "F0": np.asarray(ld.get("force") or [0, 0, 0], float),
                           "M0": np.asarray(ld.get("moment") or [0, 0, 0], float),
                           "p": np.asarray(ld["point"], float), "bi": b})
        self.acc = {}
        for ld in self.L:
            if ld["bi"] not in self.acc:
                self.acc[ld["bi"]] = self.B[ld["bi"]]["body"].AddAccumulator()

    def apply_loads(self, t):
        """External loads into the accumulators of the bodies (again before each step – explicit in the step)."""
        c = self.c
        for i in self.acc:
            self.B[i]["body"].EmptyAccumulator(self.acc[i])
        for ld in self.L:
            k = ld["law"].value(t) if ld["law"] is not None else 1.0
            i = ld["bi"]
            body = self.B[i]["body"]
            R = _R(body.GetRotMat())
            p_abs = _v(body.GetPos()) + R @ (ld["p"] - self.B[i]["com"])
            body.AccumulateForce(self.acc[i], c.ChVector3d(*(k * ld["F0"])), c.ChVector3d(*p_abs), False)
            body.AccumulateTorque(self.acc[i], c.ChVector3d(*(k * ld["M0"])), False)

    # ---------------------------------------------------------------- snapshot
    def snapshot(self, t) -> dict:
        nb, nj = len(self.B), len(self.J)
        S = {"t": t}
        pos, rot, vel, om, acc, al = (np.zeros((nb, 3)) for _ in range(6))
        rot = np.zeros((nb, 3, 3))
        for i, B in enumerate(self.B):
            b = B["body"]
            pos[i], rot[i] = _v(b.GetPos()), _R(b.GetRotMat())
            if not B["ground"]:
                vel[i], om[i] = _v(b.GetPosDt()), _v(b.GetAngVelParent())
                acc[i], al[i] = _v(b.GetPosDt2()), _v(b.GetAngAccParent())
        S.update(pos=pos, rot=rot, vel=vel, omega=om, acc=acc, alpha=al)
        jF, jM, jP, jPa, jFl, jMl = (np.zeros((nj, 3)) for _ in range(6))
        jq, jv, jfr, jQ = np.zeros((nj, 2)), np.zeros((nj, 2)), np.zeros((nj, 2)), np.zeros((nj, 2))
        for k, J in enumerate(self.J):
            L = J["link"]
            r2 = L.GetReaction2()
            f2 = L.GetFrame2Abs()
            R2 = _R(f2.GetRotMat())
            jF[k] = -R2 @ _v(r2.force)
            jM[k] = -R2 @ _v(r2.torque)
            # reaction of Chrono = force couple at the origin of frame 2 (sliding joint: frame 1 moves away with body b – the moment about the origin of frame 1 would
            # not agree, KK5) → the point of application on both bodies = origin of frame 2
            jP[k] = _v(f2.GetPos())
            jPa[k] = _v(f2.GetPos())
            if J["motor"]:
                # moment / force of the motor on body1 (= body b) about / along the axis z of the frame; the force motor is not a constraint →
                # missing in the reaction, it is added
                if J["coord"] == "rot":
                    jq[k, 0], jv[k, 0] = L.GetMotorAngle(), L.GetMotorAngleDt()
                    jQ[k, 0] = L.GetMotorTorque()
                    if J["motor"] == "torque":
                        jM[k] += jQ[k, 0] * R2[:, 2]
                else:
                    jq[k, 1], jv[k, 1] = L.GetMotorPos(), L.GetMotorPosDt()
                    jQ[k, 1] = L.GetMotorForce()
                    if J["motor"] == "force":
                        jF[k] += jQ[k, 1] * R2[:, 2]
            Ra = rot[J["a"]] if not self.body_ground(J["a"]) else np.eye(3)
            Rm0 = Ra @ J["frame"]
            jFl[k], jMl[k] = Rm0.T @ jF[k], Rm0.T @ jM[k]
        dQ = np.array([jQ[d["joint"], 0 if d["coord"] == "rot" else 1] for d in self.drivers])
        dq = np.array([jq[d["joint"], 0 if d["coord"] == "rot" else 1] for d in self.drivers])
        dv = np.array([jv[d["joint"], 0 if d["coord"] == "rot" else 1] for d in self.drivers])
        S.update(jF=jF, jM=jM, jP=jP, jPa=jPa, jFl=jFl, jMl=jMl, jq=jq, jv=jv, jfr=jfr, jlim=np.zeros(nj), jQ=jQ,
                 dQ=dQ, dq=dq, dv=dv)
        nf = len(self.F)
        fval, flen = np.zeros(nf), np.zeros(nf)
        fF, fM, fPa, fPb = (np.zeros((nf, 3)) for _ in range(4))
        pelem = pdamp = epot_s = 0.0
        for k, f in enumerate(self.F):
            T = f["tsda"]
            pb_, pa_ = _v(T.GetPoint1Abs()), _v(T.GetPoint2Abs())
            Lc = float(T.GetLength())
            Ld = float(T.GetVelocity())
            ften = -float(T.GetForce())                     # Chrono: positive = pressure (scratch/f8_motion/ch_api2.py)
            e = (pb_ - pa_) / max(np.linalg.norm(pb_ - pa_), 1e-300)
            fval[k], flen[k] = ften, Lc
            fF[k] = -ften * e
            fPa[k], fPb[k] = pa_, pb_
            kk = float(T.GetSpringCoefficient())
            dl = Lc - f["lref"]
            epot_s += f.get("f0", 0.0) * dl + 0.5 * kk * dl * dl
            pelem -= ften * Ld
            pdamp -= float(T.GetDampingCoefficient()) * Ld * Ld
        S.update(fval=fval, flen=flen, fF=fF, fM=fM, fPa=fPa, fPb=fPb)
        nl = len(self.L)
        lF, lM, lP = (np.zeros((nl, 3)) for _ in range(3))
        pload = 0.0
        for k, ld in enumerate(self.L):
            kf = ld["law"].value(t) if ld["law"] is not None else 1.0
            i = ld["bi"]
            lF[k], lM[k] = kf * ld["F0"], kf * ld["M0"]
            lP[k] = pos[i] + rot[i] @ (ld["p"] - self.B[i]["com"])
            vp = vel[i] + np.cross(om[i], lP[k] - pos[i])
            pload += lF[k] @ vp + lM[k] @ om[i]
        S.update(lF=lF, lM=lM, lP=lP)
        cc = self.sys.GetContactContainer()
        cF, cM = np.zeros((nb, 3)), np.zeros((nb, 3))
        for i in getattr(self, "contact_bodies", []):
            if self.B[i]["ground"]:
                continue
            b = self.B[i]["body"]
            cF[i] = _v(cc.GetContactableForce(b))
            cM[i] = _v(cc.GetContactableTorque(b))
        S.update(cF=cF, cM=cM)
        ek, ep = 0.0, epot_s
        for i, B in enumerate(self.B):
            if B["ground"]:
                continue
            Jg = rot[i] @ B["J"] @ rot[i].T
            ek += 0.5 * B["mass"] * vel[i] @ vel[i] + 0.5 * om[i] @ Jg @ om[i]
            ep -= B["mass"] * self.g @ pos[i]
        pdrv = float(dQ @ dv) if len(dQ) else 0.0
        S.update(ekin=ek, epot=ep, pdrv=pdrv, pfr=0.0, plim=0.0, pdamp=pdamp, pbush=0.0, pload=pload, pelem=pelem)
        return S


def weld(V: np.ndarray, F: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Merges identical vertices (the triangulation of OCP has vertices per face) → a continuous mesh for the adjacency of the edges."""
    key = np.round(V / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    F2 = inv.reshape(-1)[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return V[idx], F2[ok]


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Normals of the vertices (angle-weighted average of the normals of the triangles), length 1/cos – the shift by r keeps the planar faces."""
    N = np.zeros_like(V)
    a, b, cc = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, cc - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    for k, (p, q, w) in enumerate(((a, b, cc), (b, cc, a), (cc, a, b))):
        u, v = q - p, w - p
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", u, v) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300), -1, 1))
        np.add.at(N, F[:, k], n * ang[:, None])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
    # at an edge / in a vertex: the length so that the faces shift by r (projection on the normals of the faces = 1)
    proj = np.ones(len(V))
    for k in range(3):
        np.minimum.at(proj, F[:, k], np.einsum("ij,ij->i", N[F[:, k]], n))
    return N / np.clip(proj, 0.3, 1.0)[:, None]


def _make_function(c, model, q, law):
    """ChFunction with the law of the driver in the physical time: the kinematic driver = coordinate of the joint (DriverMotion –
    the velocity is integrated), the force one = the value of the law. GetDer/GetDer2 exactly (Chrono takes them into the constraint of the motor).
    """
    motion = DriverMotion(q, law) if q in KINEMATIC_QUANTITIES else None

    class F(c.ChFunction):
        def __init__(self):
            c.ChFunction.__init__(self)

        def GetVal(self, x):
            t = x - model.t_shift
            return law.value(t) if motion is None else motion.coord(t)

        def GetDer(self, x):
            t = x - model.t_shift
            return law.d1(t) if motion is None else motion.rate(t)

        def GetDer2(self, x):
            t = x - model.t_shift
            return law.d2(t) if motion is None else motion.accel(t)

        def Clone(self):
            return self
    return F()


def run(job: dict) -> tuple[dict, dict]:
    m = ChronoModel(job)
    N = max(int(job.get("outputs", 200)), 1)
    t_end = float(job["t_end"])
    dt_out = t_end / N
    h = job.get("dt") or min(dt_out, t_end / 2000.0, 1e-3)
    nsub = max(int(math.ceil(dt_out / h - 1e-9)), 1)
    h = dt_out / nsub
    log(f"Chrono {('SMC' if m.smc else 'NSC')}: step {h:.4g} s, {N + 1} outputs, {len(job.get('contacts') or [])} contacts")
    frames, works = [], []
    work = {k: 0.0 for k in WORKS}
    m.apply_loads(0.0)
    S = m.snapshot(0.0)
    frames.append(S)
    works.append(dict(work))
    prev = S
    t0 = time.time()
    last = 0.0
    ncmax = 0
    for k in range(1, N + 1):
        for s_ in range(nsub):
            tt = m.sys.GetChTime()
            m.apply_loads(tt)
            m.sys.DoStepDynamics(h)
            ncmax = max(ncmax, m.sys.GetNumContacts())
            if k == 1 and s_ == 0:
                # Chrono does not compute the accelerations and reactions at t = 0 → frame 0 gets the values from the first step (h)
                S1 = m.snapshot(m.sys.GetChTime())
                for key in DYNAMIC_KEYS:
                    frames[0][key] = S1[key]
        S = m.snapshot(m.sys.GetChTime())
        dt = S["t"] - prev["t"]
        for wk, pk in zip(WORKS, ("pdrv", "pfr", "plim", "pdamp", "pbush", "pload")):
            work[wk] += 0.5 * (prev[pk] + S[pk]) * dt
        frames.append(S)
        works.append(dict(work))
        prev = S
        if time.time() - last > 2.0:
            last = time.time()
            log(f"progress {100.0 * k / N:5.1f} % (t = {S['t']:.6g} s)")
    out = {"t": np.array([f["t"] for f in frames])}
    for key in ARRAY_KEYS:
        out[key] = np.array([f[key] for f in frames])
    for key in ("ekin", "epot") + POWER_KEYS:
        out[key] = np.array([f[key] for f in frames])
    for key in WORKS:
        out[key] = np.array([w[key] for w in works])
    info = {"h": h, "time_s": time.time() - t0, "contacts_max": int(ncmax), "notes": m.notes}
    return out, info


def main(job_dir: str) -> int:
    jd = Path(job_dir)
    job = json.loads((jd / "job.json").read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        arr, info = run(job)
        np.savez(jd / "result.npz", **arr)
        meta = {"ok": True, "notes": info["notes"], "time_s": time.time() - t0, "h": info["h"],
                "frames": len(arr["t"]), "solver": "chrono", "contacts_max": info["contacts_max"]}
        (jd / "result.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        log(f"finished in {time.time() - t0:.2f} s")
        return 0
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        msg = str(e) if type(e).__name__ in ("ChronoError", "LawError") else f"{type(e).__name__}: {e}"
        (jd / "result.json").write_text(json.dumps({"ok": False, "error": msg}, ensure_ascii=False), encoding="utf-8")
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
