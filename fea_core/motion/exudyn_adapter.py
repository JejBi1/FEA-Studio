"""Model of the mechanism in Exudyn (runs only in the working process fea_core.motion.worker, all in SI).

Bodies: NodeRigidBodyEP at the centroid, the reference rotation = identity matrix (the axes of the body = global axes in the position
of the CAD), ObjectRigidBody (mass, tensor of inertia about the centroid). Frame bodies (ground) = ObjectGround.
Joints: ObjectJointGeneric with the mask of the constrained axes in the frame of the joint (z = axis of the joint, x = axis2), markers on both
bodies in the same global frame (in the position of the CAD). The coordinate of the joint (rotation / displacement) = MarkerBodiesRelative-
Rotation/TranslationCoordinate → kinematic driver (ObjectConnectorCoordinate with offsetUserFunction), gear, rack-pinion, screw (coordinate
constraint with the ratio) and forces on the coordinate (force driver, torsion spring, friction, stops) once ObjectConnectorCoordinateSpringDamper with its own
force function (the load – the calculation of the force is ours).

Verified conventions (scratch/f8_motion/proto3.py):
  - ObjectConnectorCoordinate [m0, m1], factor1 f: equation f·q1 − q0 − offset = 0; generalised forces Q_m0 = +λ,
    Q_m1 = −f·λ (λ = the output Force).
  - ObjectConnectorCoordinateSpringDamper [m_ground, m_q] with a user function: generalised force on q = −(the returned value).
  - ObjectJointGeneric: ForceLocal / TorqueLocal = force / moment in the frame of marker 0 acting on the body of marker 0
    (body_a) → on body_b acts −R_m0·ForceLocal, −R_m0·TorqueLocal.
  - ObjectConnectorSpringDamper: returned force > 0 = tension.
Friction (decision 2): the normal force = the reaction of the joint from the previous (converged) step.
"""
from __future__ import annotations

import math

import numpy as np

from .laws import Law, DriverMotion, KINEMATIC_QUANTITIES, FORCE_QUANTITIES

# mask of the constrained axes of the frame of the joint [x y z | rx ry rz] (the axis of the joint = z, the second direction = x)
MASKS = {"fixed": [1, 1, 1, 1, 1, 1], "revolute": [1, 1, 1, 1, 1, 0], "prismatic": [1, 1, 0, 1, 1, 1],
         "cylindrical": [1, 1, 0, 1, 1, 0], "spherical": [1, 1, 1, 0, 0, 0], "universal": [1, 1, 1, 0, 0, 1],
         "planar": [0, 0, 1, 1, 1, 0], "pin_slot": [0, 1, 1, 1, 1, 0], "screw": [1, 1, 0, 1, 1, 0]}
ROT_FREE = ("revolute", "cylindrical", "screw", "pin_slot")      # coordinate of the rotation about z
TRANS_FREE = {"prismatic": 2, "cylindrical": 2, "screw": 2, "pin_slot": 0}   # displacement along the axis of the frame (index)
FIT = {"clearance": 1.0, "worn": 4.0 / math.pi, "fitted": math.pi / 2.0, "uniform": math.pi / 2.0}


class MotionError(RuntimeError):
    """Comprehensible error of the model / calculation of the mechanism."""


def joint_frame(axis, axis2=None) -> np.ndarray:
    """Frame of the joint (columns x, y, z): z = axis, x = axis2 projected perpendicular to the axis (otherwise any perpendicular)."""
    z = np.asarray(axis, float)
    nz = np.linalg.norm(z)
    if nz == 0:
        raise MotionError("zero joint axis")
    z = z / nz
    x = None
    if axis2 is not None:
        x = np.asarray(axis2, float) - np.dot(axis2, z) * z
        if np.linalg.norm(x) < 1e-9:
            x = None
    if x is None:
        x = np.cross([0.0, 1.0, 0.0], z) if abs(z[1]) < 0.9 else np.cross([1.0, 0.0, 0.0], z)
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _skew_vec(W):
    return np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]]) * 0.5


class Model:
    """Assembled model of Exudyn + writer of the outputs. with_drivers=False → without the kinematic drivers (the check of the freedom of the mechanism without drivers)."""

    def __init__(self, job: dict, with_drivers: bool = True, log=print):
        import exudyn as exu
        from exudyn import itemInterface as ii
        self.exu, self.ii, self.job, self.log = exu, ii, job, log
        self.kind = job.get("kind", "dynamic")
        self.SC = exu.SystemContainer()
        mbs = self.mbs = self.SC.AddSystem()
        self.t_override = None             # time for the static solution at another time (assembly, finite differences)
        self.t_shift = 0.0                 # Exudyn must not start at a negative time → the lead shifts the time: t = τ − t_shift
        self.assembly = False              # assembly: loads switched off
        self.lam_sign = 1.0                # sign of the Lagrange multipliers: dynamics +1, statics −1
        self.kin_rates = None              # velocities of the coordinates and of the elements from finite differences (a series of statics)
        self.ground = mbs.AddObject(ii.ObjectGround())
        self.ng = mbs.AddNode(ii.NodePointGround())
        self.mg = mbs.AddMarker(ii.MarkerNodeCoordinate(nodeNumber=self.ng, coordinate=0))
        g = job.get("gravity")
        self.g = np.asarray(g, float) if g is not None else np.zeros(3)
        self._bodies()
        self._joints()
        self._couplings()
        self.drivers_kin, self.drivers_force = [], []
        self._drivers(with_drivers)
        self._forces()
        self._loads()
        self._coord_forces()
        mbs.Assemble()

    def time(self, t: float) -> float:
        """Physical time from the time of Exudyn τ (the lead of the kinematic task) or the imposed time of the static solution."""
        return self.t_override if self.t_override is not None else t - self.t_shift

    # ---------------------------------------------------------------- bodies
    def _bodies(self):
        mbs, ii = self.mbs, self.ii
        self.B = []                        # by bodies: dict(obj, node, mass, com, J, ground)
        for b in self.job["bodies"]:
            com = np.asarray(b.get("com") or [0, 0, 0], float)
            if b.get("flexible") and not b.get("ground"):
                self.B.append(self._flex_body(b, len(self.B)))
                continue
            if b.get("ground"):
                self.B.append({"name": b["name"], "ground": True, "obj": self.ground, "node": None, "com": com,
                               "mass": 0.0, "J": np.zeros((3, 3))})
                continue
            m = float(b["mass"])
            J = np.asarray(b["inertia"], float).reshape(3, 3)
            if not m > 0:
                raise MotionError(f"Body '{b['name']}' has zero mass")
            v0 = np.asarray(b.get("v0") or [0, 0, 0], float)
            w0 = np.asarray(b.get("w0") or [0, 0, 0], float)
            ep_t = 0.5 * np.array([-0.0, w0[0], w0[1], w0[2]])      # ε̇ = ½·[0, ω] for ε = [1, 0, 0, 0]
            n = mbs.AddNode(ii.NodeRigidBodyEP(referenceCoordinates=list(com) + [1.0, 0, 0, 0],
                                               initialVelocities=list(v0) + list(ep_t)))
            o = mbs.AddObject(ii.ObjectRigidBody(physicsMass=m, nodeNumber=n,
                                                 inertia=[J[0, 0], J[1, 1], J[2, 2], J[1, 2], J[0, 2], J[0, 1]]))
            rec = {"name": b["name"], "ground": False, "obj": o, "node": n, "com": com, "mass": m, "J": J}
            if self.g.any():
                mm = mbs.AddMarker(ii.MarkerBodyMass(bodyNumber=o))
                rec["gload"] = mbs.AddLoad(ii.LoadMassProportional(markerNumber=mm, loadVector=list(self.g)))
            # inertia forces d'Alembert (the kinematic task solved by a series of static equilibria): −m·a, −(J·α + ω × J·ω)
            mc = mbs.AddMarker(ii.MarkerBodyRigid(bodyNumber=o, localPosition=[0.0, 0.0, 0.0]))
            ib = len(self.B)
            rec["iF"], rec["iM"] = np.zeros(3), np.zeros(3)
            mbs.AddLoad(ii.LoadForceVector(markerNumber=mc, loadVector=[0.0, 0.0, 0.0],
                                           loadVectorUserFunction=lambda m_, t, v, _i=ib: list(self.B[_i]["iF"])))
            mbs.AddLoad(ii.LoadTorqueVector(markerNumber=mc, loadVector=[0.0, 0.0, 0.0],
                                            loadVectorUserFunction=lambda m_, t, v, _i=ib: list(self.B[_i]["iM"])))
            self.B.append(rec)

    def _flex_body(self, b: dict, bi: int) -> dict:
        """Flexible body: M, K from NGSolve on our tet10 mesh (nodal basis, as the backend ngs) → Exudyn FEMinterface →
        Craig–Bampton (RBE2 / RBE3 on the faces of the joints) → ObjectFFRFreducedOrder with the markers at the joints.
        """
        from pathlib import Path
        fd = Path(b["flex_dir"])
        key = (str(fd), int(b.get("modes", 10)), b.get("interface"))
        if key not in _FLEX_CACHE:                     # the check of the freedom and the calculation build the model several times
            _FLEX_CACHE[key] = self._flex_fem(b, fd)
        fem, X, ref, mass, lumped, nodes, weights, ifs, freq = _FLEX_CACHE[key]
        N = len(X)
        # markers at the points of the joints: displacement from the weighted centre of the nodes of the interface (the axes of the body)
        offs = []
        for k, it in enumerate(ifs):
            jj = self.job["joints"][it["joint"]]
            p = np.asarray(jj["origin"], float)
            if it["side"] == "b" and jj.get("origin_b") is not None:
                p = np.asarray(jj["origin_b"], float)
            offs.append(p - ref - weights[k] @ (X[nodes[k]] - ref))
        beta = 0.0
        if b.get("zeta") and len(freq):
            beta = 2.0 * float(b["zeta"]) / (2 * math.pi * float(freq[0]))
        d = self.mbs.CreateFFRFReducedOrderObject(name=f"pruzne{bi}", femInterface=fem, referencePosition=list(ref),
                                                  gravity=list(self.g), stiffnessProportionalDamping=beta,
                                                  superElementRigidMarkersOffsets=np.array(offs), showMarkers=False)
        od = d["FFRFReducedOrderObjectDict"]
        phi = np.asarray(fem.modeBasis["matrix"], float).reshape(N, 3, -1)
        np.savez(fd / "modes.npz", phi=phi, freq=freq, ref=ref, mass=mass)
        iface = {it["joint"]: d[f"pruzne{bi}:J{it['joint']}"] for it in ifs}
        J = np.zeros((3, 3))                                        # inertia about the centroid (informative only)
        r = X - ref
        for c in range(3):
            J[c, c] = float((lumped * (r[:, (c + 1) % 3] ** 2 + r[:, (c + 2) % 3] ** 2)).sum())
        return {"name": b["name"], "ground": False, "flex": True, "obj": od["oFFRFreducedOrder"],
                "node": od["nRigidBody"], "modal_node": od["nGenericODE2"], "com": ref, "mass": mass, "J": J,
                "iface": iface, "nmodes": phi.shape[2], "freq": freq}

    def _flex_fem(self, b, fd):
        """M, K (NGSolve, nodal basis of our mesh) and the Craig–Bampton shapes of the flexible body – once per process."""
        import json
        import scipy.sparse as sp
        from exudyn.FEM import FEMinterface, HCBstaticModeSelection
        from ..ngs.worker import build, transform, node_to_ngs, bilinear
        ngs, msh, mesh, order, nv, ev, edge_mid = build(fd)
        fes, a, _ = bilinear(ngs, mesh, fd, msh)
        mats = json.loads((fd / "materials.json").read_text(encoding="utf-8"))
        rho = mesh.MaterialCF({f"p{i}": m["rho"] for i, m in enumerate(mats)}, default=mats[0]["rho"])
        u, v = fes.TnT()
        bm = ngs.BilinearForm(rho * u * v * ngs.dx)
        with ngs.TaskManager():
            a.Assemble()
            bm.Assemble()

        def csr(bf):
            vals, cols, ptr = bf.mat.CSR()
            return sp.csr_matrix((np.array(vals), np.array(cols), np.array(ptr)), shape=(fes.ndof, fes.ndof))
        N = len(msh.nodes)
        Q = node_to_ngs(transform(nv, ev, edge_mid, N), order, N)
        K = (Q.T @ csr(a) @ Q).tocsr()
        M = (Q.T @ csr(bm) @ Q).tocsr()
        lumped = np.asarray(M.sum(axis=1)).ravel()[0::3]          # mass per node (row sums, component x)
        mass = float(lumped.sum())
        X = msh.nodes
        ref = (lumped[:, None] * X).sum(axis=0) / mass             # the centroid of the mesh = origin of the floating frame
        ifs = json.loads((fd / "interfaces.json").read_text(encoding="utf-8"))
        fem = FEMinterface()
        fem.nodes = {"Position": X - ref}
        fem.elements = [{"Name": "tet10", "Tet10": msh.tets}]
        fem.massMatrix = M
        fem.stiffnessMatrix = K
        tri, _ = msh.boundary_faces()
        fem.surface = [{"Name": "surface", "Trigs": np.asarray(tri[:, :3], int)}]
        nodes = [np.asarray(it["nodes"], int) for it in ifs]
        weights = [np.asarray(it["weights"], float) for it in ifs]
        fem.nodeSets = [{"Name": f"J{it['joint']}", "NodeNumbers": nodes[k], "NodeWeights": weights[k],
                         "type": "boundary"} for k, it in enumerate(ifs)]
        mode = HCBstaticModeSelection.RBE3 if b.get("interface") == "rbe3" else HCBstaticModeSelection.RBE2
        self.log(f"flexible body {b['name']}: {N} nodes, {len(msh.tets)} tet10 elements, Craig–Bampton {b.get('modes', 10)} modes + {len(ifs)} interfaces")
        fem.ComputeHurtyCraigBamptonModes(boundaryNodesList=nodes, nEigenModes=int(b.get("modes", 10)),
                                          useSparseSolver=True, computationMode=mode, boundaryNodesWeights=weights)
        freq = np.sqrt(np.abs(np.asarray(fem.eigenValues, float))) / (2 * math.pi)
        return fem, X, ref, mass, lumped, nodes, weights, ifs, freq

    def body_flex(self, i):
        return i is not None and i >= 0 and bool(self.B[i].get("flex"))

    def body_obj(self, i):
        return self.ground if i is None or i < 0 else self.B[i]["obj"]

    def body_com(self, i):
        return np.zeros(3) if i is None or i < 0 or self.B[i]["ground"] else self.B[i]["com"]

    def body_ground(self, i):
        return i is None or i < 0 or self.B[i]["ground"]

    # ---------------------------------------------------------------- joints
    def _joints(self):
        mbs, ii, exu = self.mbs, self.ii, self.exu
        self.J = []
        for jk, j in enumerate(self.job["joints"]):
            kind = j["kind"]
            rec = {"name": j["name"], "kind": kind, "a": j.get("a", -1), "b": j.get("b", -1), "generic": None,
                   "dist": None, "rot": None, "trans": None, "Q": {}, "fric": j.get("friction"),
                   "limits": j.get("limits"), "limit_k": j.get("limit_k"), "limit_c": j.get("limit_c", 0.0),
                   "torsion": [], "fdrv": []}
            self.J.append(rec)
            if kind in ("gear", "rack_pinion"):
                rec["frame"] = np.eye(3)
                rec["pa"] = rec["pb"] = np.asarray(j.get("origin") or [0, 0, 0], float)
                continue
            a, b = rec["a"], rec["b"]
            if a == b or (self.body_ground(a) and self.body_ground(b)):
                raise MotionError(f"Joint '{j['name']}' connects the same body (or two ground bodies)")
            oa, ob = self.body_obj(a), self.body_obj(b)
            if kind == "distance":
                pa = np.asarray(j.get("point_a") or j["origin"], float)
                pb = np.asarray(j.get("point_b") or j["origin"], float)
                d = float(np.linalg.norm(pb - pa))
                if d <= 0:
                    raise MotionError(f"Link '{j['name']}' has zero length (point_a = point_b)")
                ma = mbs.AddMarker(ii.MarkerBodyPosition(bodyNumber=oa, localPosition=list(pa - self.body_com(a))))
                mb = mbs.AddMarker(ii.MarkerBodyPosition(bodyNumber=ob, localPosition=list(pb - self.body_com(b))))
                rec["dist"] = mbs.AddObject(ii.ObjectConnectorDistance(markerNumbers=[ma, mb], distance=d))
                rec.update(pa=pa, pb=pb, frame=joint_frame(pb - pa), ma=ma, mb=mb, length=d)
                continue
            if kind not in MASKS:
                raise MotionError(f"Joint '{j['name']}': unknown type '{kind}'")
            A = joint_frame(j["axis"], j.get("axis2"))
            pa = np.asarray(j["origin"], float)
            pb = np.asarray(j.get("origin_b") or j["origin"], float)
            la, lb = pa - self.body_com(a), pb - self.body_com(b)
            fa, fb = self.body_flex(a), self.body_flex(b)
            # flexible body: marker of the Craig–Bampton interface (MarkerSuperElementRigid) rotated by the parameter of the joint
            if fa:
                ma = self.B[a]["iface"][jk]
            else:
                ma = mbs.AddMarker(ii.MarkerBodyRigid(bodyNumber=oa, localHT=exu.HT(rotation=A, translation=list(la))))
            if fb:
                mb = self.B[b]["iface"][jk]
            else:
                mb = mbs.AddMarker(ii.MarkerBodyRigid(bodyNumber=ob, localHT=exu.HT(rotation=A, translation=list(lb))))
            kw = {}
            if fa or fb:
                kw = {"rotationMarker0": A if fa else np.eye(3), "rotationMarker1": A if fb else np.eye(3)}
            rec["generic"] = mbs.AddObject(ii.ObjectJointGeneric(markerNumbers=[ma, mb], constrainedAxes=MASKS[kind],
                                                                 **kw))
            rec.update(pa=pa, pb=pb, frame=A, ma=ma, mb=mb, flex=fa or fb)
            if fa or fb:
                if j.get("friction") or j.get("limits") is not None:
                    raise MotionError(f"Joint '{j['name']}' on a flexible body: friction and stops are not supported yet")
                continue                       # coordinate markers cannot be attached to a flexible body in Exudyn
            if kind in ROT_FREE:
                nd = mbs.AddNode(ii.NodeGenericData(numberOfDataCoordinates=1, initialCoordinates=[0.0]))
                rec["rot"] = mbs.AddMarker(ii.MarkerBodiesRelativeRotationCoordinate(
                    bodyNumbers=[oa, ob], nodeNumber=nd, axis0=list(A[:, 2]), localPosition0=list(la),
                    localPosition1=list(lb)))
                rec["rot_axis"] = A[:, 2]
            if kind in TRANS_FREE:
                ax = A[:, TRANS_FREE[kind]]
                rec["trans"] = mbs.AddMarker(ii.MarkerBodiesRelativeTranslationCoordinate(
                    bodyNumbers=[oa, ob], axis0=list(ax), localPosition0=list(la), localPosition1=list(lb)))
                rec["trans_axis"] = ax
            if kind == "screw":
                lead = j.get("lead")
                if not lead:
                    raise MotionError(f"Screw joint '{j['name']}' has no lead")
                rec["screw"] = mbs.AddObject(ii.ObjectConnectorCoordinate(
                    markerNumbers=[rec["trans"], rec["rot"]], factor1=float(lead) / (2 * math.pi)))
                rec["screw_f"] = float(lead) / (2 * math.pi)

    def _couplings(self):
        """Gear (φ_b = ratio·φ_a) and rack-pinion (s_b = ratio·φ_a) between the coordinates of two joints."""
        mbs, ii = self.mbs, self.ii
        for k, j in enumerate(self.job["joints"]):
            if j["kind"] not in ("gear", "rack_pinion"):
                continue
            ja, jb = j.get("joint_a"), j.get("joint_b")
            if ja is None or jb is None or not j.get("ratio"):
                raise MotionError(f"'{j['name']}': gearing needs coupled joints joint_a, joint_b and the ratio")
            A, Bj = self.J[ja], self.J[jb]
            if A["rot"] is None:
                raise MotionError(f"'{j['name']}': joint '{A['name']}' has no rotation (gearing couples revolute joints)")
            if j["kind"] == "gear":
                if Bj["rot"] is None:
                    raise MotionError(f"'{j['name']}': joint '{Bj['name']}' has no rotation")
                mb, coord_b = Bj["rot"], "rot"
            else:
                if Bj["trans"] is None:
                    raise MotionError(f"'{j['name']}': rack '{Bj['name']}' must be a slider joint")
                mb, coord_b = Bj["trans"], "trans"
            r = float(j["ratio"])
            o = mbs.AddObject(ii.ObjectConnectorCoordinate(markerNumbers=[mb, A["rot"]], factor1=r))
            self.J[k]["coupling"] = (o, jb, coord_b, ja, r)

    # ---------------------------------------------------------------- pohony
    def _drivers(self, with_drivers: bool):
        """Kinematic driver = the constrained axis of the generic joint with the prescribed displacement (offsetUserFunction) –
        the rotation over ±π without a jump (the constraint on MarkerBodiesRelativeRotationCoordinate had peaks of the acceleration at ±π,
        scratch/f8_motion/proto4.py); force driver = force on the coordinate of the joint (coord_force).
        """
        mbs = self.mbs
        kin_by_joint: dict[int, dict[int, object]] = {}
        for d in self.job.get("drivers", []):
            J = self.J[d["joint"]]
            q = d["quantity"]
            law = Law(d["law"], d["data"], d.get("scale", 1.0))
            coord = "rot" if q in ("angle", "angular_velocity", "torque") else "trans"
            if J[coord] is None or J["generic"] is None:
                raise MotionError(f"Drive of joint '{J['name']}': the joint has no {('rotation' if coord == 'rot' else 'displacement')} for quantity '{q}'")
            rec = {"name": d.get("name") or J["name"], "joint": d["joint"], "coord": coord, "quantity": q,
                   "law": law}
            if q in KINEMATIC_QUANTITIES:
                rec["motion"] = DriverMotion(q, law)
                rec["axis"] = 5 if coord == "rot" else TRANS_FREE[J["kind"]]
                if not with_drivers:
                    continue
                if rec["axis"] in kin_by_joint.get(d["joint"], {}):
                    raise MotionError(f"Joint '{J['name']}' has two kinematic drives of the same coordinate")
                kin_by_joint.setdefault(d["joint"], {})[rec["axis"]] = rec["motion"]
                rec["obj"] = J["generic"]
                self.drivers_kin.append(rec)
            elif q in FORCE_QUANTITIES:
                J["fdrv"].append(rec)
                self.drivers_force.append(rec)
            else:
                raise MotionError(f"Drive '{rec['name']}': unknown quantity '{q}'")
        for k, axes in kin_by_joint.items():
            J = self.J[k]
            mask = list(MASKS[J["kind"]])
            for ax in axes:
                mask[ax] = 1

            # verified (scratch/f8_motion/proto8.py, proto9.py): the rotational components of the displacement are in the frame of the marker (the axes
            # of the joint), but the translational ones in the coordinates of body 0 WITHOUT the rotation of the marker → displacement s along the axis of the frame = s·A[:, ax]
            def off(mbs_, t, itemNumber, par, _axes=axes, _A=J["frame"]):
                tt = self.time(t)
                v = [0.0] * 6
                for ax, mo in _axes.items():
                    if ax >= 3:
                        v[ax] = mo.coord(tt)
                    else:
                        d = mo.coord(tt) * _A[:, ax]
                        v[0] += d[0]
                        v[1] += d[1]
                        v[2] += d[2]
                return v
            mbs.SetObjectParameter(J["generic"], "constrainedAxes", mask)
            mbs.SetObjectParameter(J["generic"], "offsetUserFunction", off)
            J["driven_axes"] = sorted(axes)

    # ---------------------------------------------------------------- force elements
    def _forces(self):
        mbs, ii, exu = self.mbs, self.ii, self.exu
        self.F = []
        for f in self.job.get("forces", []):
            kind = f["kind"]
            rec = dict(f)
            self.F.append(rec)
            if kind == "torsion":
                J = self.J[f["joint"]] if f.get("joint") is not None else None
                if J is None or J["rot"] is None:
                    raise MotionError(f"Torsion spring '{f['name']}' needs a revolute joint (joint)")
                J["torsion"].append(rec)
                continue
            a, b = f.get("a", -1), f.get("b", -1)
            oa, ob = self.body_obj(a), self.body_obj(b)
            pa, pb = np.asarray(f["point_a"], float), np.asarray(f["point_b"], float)
            rec["pa"], rec["pb"] = pa, pb
            if kind == "bushing":
                ma = mbs.AddMarker(ii.MarkerBodyRigid(bodyNumber=oa, localPosition=list(pa - self.body_com(a))))
                mb = mbs.AddMarker(ii.MarkerBodyRigid(bodyNumber=ob, localPosition=list(pa - self.body_com(b))))
                k6 = np.asarray(f.get("k6") or [0] * 6, float)
                c6 = np.asarray(f.get("c6") or [0] * 6, float)
                rec["obj"] = mbs.AddObject(ii.ObjectConnectorRigidBodySpringDamper(
                    markerNumbers=[ma, mb], stiffness=np.diag(k6), damping=np.diag(c6)))
                rec["ma"], rec["mb"] = ma, mb
                continue
            if kind not in ("spring", "damper", "spring_damper", "gas_spring"):
                raise MotionError(f"Force element '{f['name']}': unknown type '{kind}'")
            L0 = float(np.linalg.norm(pb - pa))
            if L0 <= 0:
                raise MotionError(f"Force element '{f['name']}': the attachment points coincide")
            rec["lref"] = L0 if f.get("l0") is None else float(f["l0"])
            if kind == "gas_spring":
                tab = np.asarray(f.get("table") or [], float).reshape(-1, 2)
                if len(tab) < 2:
                    raise MotionError(f"Gas spring '{f['name']}' needs an F(s) table with at least 2 points")
                tab = tab[np.argsort(tab[:, 0])]
                rec["tab"] = tab
                # potential ∫F ds (trapezium along the table)
                rec["tab_cum"] = np.concatenate([[0.0], np.cumsum(0.5 * (tab[1:, 1] + tab[:-1, 1]) * np.diff(tab[:, 0]))])
            ma = mbs.AddMarker(ii.MarkerBodyPosition(bodyNumber=oa, localPosition=list(pa - self.body_com(a))))
            mb = mbs.AddMarker(ii.MarkerBodyPosition(bodyNumber=ob, localPosition=list(pb - self.body_com(b))))
            rec["ma"], rec["mb"] = ma, mb

            def uf(mbs_, t, itemNumber, deltaL, deltaL_t, stiffness, damping, force, _r=rec):
                if self.kin_rates is not None:          # series of static solutions: velocity from finite differences
                    deltaL_t = self.kin_rates.get(("F", id(_r)), 0.0)
                return self.spring_force(_r, deltaL + _r["lref"], deltaL_t)
            rec["obj"] = mbs.AddObject(ii.ObjectConnectorSpringDamper(
                markerNumbers=[ma, mb], referenceLength=rec["lref"], springForceUserFunction=uf))

    def spring_force(self, r, L, Ld) -> float:
        """Force of the element [N], positive = tension: f0 + k·(L − l0) + c·L̇; gas spring −F(l0 − L) (pushes)."""
        if r["kind"] == "gas_spring":
            s = r["lref"] - L
            return -float(np.interp(s, r["tab"][:, 0], r["tab"][:, 1])) + r.get("c", 0.0) * Ld
        k = r.get("k", 0.0) if r["kind"] != "damper" else 0.0
        c = r.get("c", 0.0) if r["kind"] != "spring" else 0.0
        return r.get("f0", 0.0) + k * (L - r["lref"]) + c * Ld

    def spring_energy(self, r, L) -> float:
        if r["kind"] == "gas_spring":
            s = r["lref"] - L
            tab, cum = r["tab"], r["tab_cum"]
            if s <= tab[0, 0]:
                return float(cum[0] + tab[0, 1] * (s - tab[0, 0]))
            if s >= tab[-1, 0]:
                return float(cum[-1] + tab[-1, 1] * (s - tab[-1, 0]))
            i = int(np.searchsorted(tab[:, 0], s) - 1)
            d = s - tab[i, 0]
            sl = (tab[i + 1, 1] - tab[i, 1]) / (tab[i + 1, 0] - tab[i, 0])
            return float(cum[i] + tab[i, 1] * d + 0.5 * sl * d * d)
        if r["kind"] == "damper":
            return 0.0
        dl = L - r["lref"]
        return r.get("f0", 0.0) * dl + 0.5 * r.get("k", 0.0) * dl * dl

    # ---------------------------------------------------------------- external loads
    def _loads(self):
        mbs, ii = self.mbs, self.ii
        self.L = []
        for ld in self.job.get("loads", []):
            b = ld["body"]
            if self.body_ground(b):
                continue
            rec = dict(ld)
            p = np.asarray(ld["point"], float)
            rec["p"] = p
            m = mbs.AddMarker(ii.MarkerBodyRigid(bodyNumber=self.body_obj(b), localPosition=list(p - self.body_com(b))))
            rec["m"] = m
            tab = ld.get("table")
            rec["law"] = Law("table", tab) if tab else None
            F0, M0 = np.asarray(ld.get("force") or [0, 0, 0], float), np.asarray(ld.get("moment") or [0, 0, 0], float)
            rec["F0"], rec["M0"] = F0, M0

            def fF(mbs_, t, v, _r=rec):
                return list(self.load_factor(_r, t) * _r["F0"])

            def fM(mbs_, t, v, _r=rec):
                return list(self.load_factor(_r, t) * _r["M0"])
            if F0.any():
                mbs.AddLoad(ii.LoadForceVector(markerNumber=m, loadVector=list(F0), loadVectorUserFunction=fF))
            if M0.any():
                mbs.AddLoad(ii.LoadTorqueVector(markerNumber=m, loadVector=list(M0), loadVectorUserFunction=fM))
            self.L.append(rec)

    def load_factor(self, r, t) -> float:
        if self.assembly:
            return 0.0
        tt = self.time(t)
        return r["law"].value(tt) if r["law"] is not None else 1.0

    # ---------------------------------------------------------------- forces on the coordinate of the joint
    def _coord_forces(self):
        mbs, ii = self.mbs, self.ii
        for J in self.J:
            fr = J["fric"]
            for coord in ("rot", "trans"):
                if J[coord] is None:
                    continue
                need = bool(J["fdrv"] and any(d["coord"] == coord for d in J["fdrv"]))
                need |= bool(coord == "rot" and J["torsion"])
                need |= bool(J["limits"] is not None and coord == self.primary(J))
                need |= bool(fr and coord in self.friction_coords(J))
                if not need:
                    continue

                def uf(mbs_, t, itemNumber, u, v, k, d, off, _J=J, _c=coord):
                    if self.kin_rates is not None:
                        v = self.kin_rates.get((id(_J), _c), 0.0)
                    return -self.coord_force(_J, _c, t, u, v)[0]
                J.setdefault("cf", {})[coord] = mbs.AddObject(ii.ObjectConnectorCoordinateSpringDamper(
                    markerNumbers=[self.mg, J[coord]], springForceUserFunction=uf))
            J["Rprev"] = np.zeros(3)          # force of the joint from the previous step (frame of the joint) – normal force of the friction

    @staticmethod
    def primary(J) -> str:
        return "trans" if J["kind"] in ("prismatic", "pin_slot") else "rot" if J["rot"] is not None else "trans"

    @staticmethod
    def friction_coords(J) -> tuple:
        if J["kind"] == "cylindrical":
            return ("rot", "trans")
        return (Model.primary(J),)

    def friction_force(self, J, coord, v) -> float:
        """Friction force / moment on the coordinate (against the velocity v), normal force from the reaction of the previous step.
        The parameters depending on the kind of the coordinate (c_v, v_s, σ2) are dictionaries {"rot": …, "trans": …} in SI.
        """
        f = J["fric"]
        if not f:
            return 0.0
        model = f.get("model", "coulomb")
        cv = f.get("c_v", {}).get(coord, 0.0)
        if model == "viscous":
            return -cv * v
        R = J["Rprev"]
        if coord == "rot":
            if J["kind"] == "spherical":
                N = float(np.linalg.norm(R))
            elif model == "thread":
                N = abs(float(R[2]))
            else:
                N = float(math.hypot(R[0], R[1]))
            r = f.get("r_eff", 0.005)
            vs = v * r                                   # slip velocity at the radius r_eff
        else:
            N = float(math.hypot(R[0], R[1])) if J["kind"] != "pin_slot" else float(math.hypot(R[1], R[2]))
            r = 1.0
            vs = v
        N += f.get("preload", 0.0)
        mu_k = f.get("mu_k", 0.1)
        mu_s = f.get("mu_s", mu_k)
        mu = mu_k
        if model == "stribeck":
            vsx = f.get("v_s", {}).get(coord, 0.0)
            if vsx > 0:
                mu = mu_k + (mu_s - mu_k) * math.exp(-abs(v / vsx) ** f.get("delta", 2.0))
        if coord == "rot" and model in ("coulomb", "stribeck"):
            mu = FIT.get(f.get("fit", "clearance"), 1.0) * mu / math.sqrt(1.0 + mu * mu)
        elif model == "thread":
            mu = mu / math.cos(math.radians(30.0))       # metric thread, apex angle 60°
        vreg = max(f.get("v_reg", 5e-4), 1e-12)
        return -mu * N * r * math.tanh(vs / vreg) - (f.get("sigma2", {}).get(coord, 0.0) + cv) * v

    def limit_force(self, J, u, v) -> float:
        lim = J["limits"]
        if lim is None:
            return 0.0
        lo, hi = lim
        k, c = J["limit_k"], J["limit_c"]
        if hi is not None and u > hi:
            return -max(0.0, k * (u - hi) + c * v)
        if lo is not None and u < lo:
            return max(0.0, k * (lo - u) - c * v)
        return 0.0

    def coord_force(self, J, coord, t, u, v):
        """(total generalised force on the coordinate, friction, stop, force driver, torsion springs)."""
        tt = self.time(t)
        fd = 0.0 if self.assembly else sum(d["law"].value(tt) for d in J["fdrv"] if d["coord"] == coord)
        ft = 0.0
        if coord == "rot":
            for s in J["torsion"]:
                ft -= s.get("f0", 0.0) + s.get("k", 0.0) * (u - s.get("phi0", 0.0)) + s.get("c", 0.0) * v
        ff = self.friction_force(J, coord, v) if (J["fric"] and coord in self.friction_coords(J)) else 0.0
        fl = self.limit_force(J, u, v) if coord == self.primary(J) else 0.0
        return fd + ft + ff + fl, ff, fl, fd, ft

    # ---------------------------------------------------------------- stav
    def body_state(self, i):
        """(position of the centroid, R, v, ω, a, α) – globally."""
        exu = self.exu
        B = self.B[i]
        if B["ground"]:
            z = np.zeros(3)
            return B["com"], np.eye(3), z, z, z, z
        n, ov = B["node"], exu.OutputVariableType
        g = self.mbs.GetNodeOutput
        return (np.array(g(n, ov.Position)), np.array(g(n, ov.RotationMatrix)).reshape(3, 3),
                np.array(g(n, ov.Velocity)), np.array(g(n, ov.AngularVelocity)),
                np.array(g(n, ov.Acceleration)), np.array(g(n, ov.AngularAcceleration)))

    def marker_coord(self, m):
        ov = self.exu.OutputVariableType
        return (float(np.ravel(self.mbs.GetMarkerOutput(m, ov.Coordinates))[0]),
                float(np.ravel(self.mbs.GetMarkerOutput(m, ov.Coordinates_t))[0]))

    # ---------------------------------------------------------------- snapshot of the state
    def snapshot(self, t: float) -> dict:
        """All output quantities in the current (converged) state – SI, global axes."""
        exu, mbs = self.exu, self.mbs
        ov = exu.OutputVariableType
        nb, nj = len(self.B), len(self.J)
        S = {"t": self.time(t)}
        st = [self.body_state(i) for i in range(nb)]
        for k, key in enumerate(("pos", "rot", "vel", "omega", "acc", "alpha")):
            S[key] = np.array([s[k] for s in st]) if nb else np.zeros((0, 3, 3) if key == "rot" else (0, 3))
        S["flexq"] = {i: np.array(mbs.GetNodeOutput(B["modal_node"], ov.Coordinates), float)
                      for i, B in enumerate(self.B) if B.get("flex")}       # modal coordinates of the flexible bodies

        def rot(i):
            return np.eye(3) if self.body_ground(i) else S["rot"][i]

        def omega(i):
            return np.zeros(3) if self.body_ground(i) else S["omega"][i]
        # generalised forces on the coordinates of the joints (drivers, constraints of the coordinates, forces on the coordinate)
        Q = [{"rot": 0.0, "trans": 0.0} for _ in range(nj)]
        q = np.zeros((nj, 2))
        qd = np.zeros((nj, 2))
        fr = np.zeros((nj, 2))
        lim = np.zeros(nj)
        for k, J in enumerate(self.J):
            for c, ci in (("rot", 0), ("trans", 1)):
                if J[c] is not None:
                    q[k, ci], qd[k, ci] = self.marker_coord(J[c])
            for c in J.get("cf", {}):
                ci = 0 if c == "rot" else 1
                tot, ff, fl, fd, ft = self.coord_force(J, c, t, q[k, ci], qd[k, ci])
                Q[k][c] += tot
                fr[k, ci] += ff
                lim[k] += fl
            if J.get("screw") is not None:
                lam = self.lam_sign * float(np.ravel(mbs.GetObjectOutput(J["screw"], ov.Force))[0])
                Q[k]["trans"] += lam
                Q[k]["rot"] += -J["screw_f"] * lam
            if J.get("coupling") is not None:
                o, jb, cb, ja, r = J["coupling"]
                lam = self.lam_sign * float(np.ravel(mbs.GetObjectOutput(o, ov.Force))[0])
                Q[jb][cb] += lam
                Q[ja]["rot"] += -r * lam
        dQ, dq, dv = [], [], []
        for d in self.drivers_kin:                          # force of the driver = component of the reaction of the generic joint
            dQ.append(self.driver_force(d))
        for d in self.drivers_force:
            dQ.append(0.0 if self.assembly else d["law"].value(self.time(t)))
        for d in self.drivers_kin + self.drivers_force:
            ci = 0 if d["coord"] == "rot" else 1
            dq.append(q[d["joint"], ci])
            dv.append(qd[d["joint"], ci])
        jF, jM, jP, jPa, jFl, jMl = (np.zeros((nj, 3)) for _ in range(6))
        for k, J in enumerate(self.J):
            Ra = rot(J["a"])
            if J["generic"] is not None:
                FL = np.array(mbs.GetObjectOutput(J["generic"], ov.ForceLocal))
                TL = np.array(mbs.GetObjectOutput(J["generic"], ov.TorqueLocal))
                Rm0 = Ra @ J["frame"]
                Rm1 = rot(J["b"]) @ J["frame"]
                jF[k] = -self.lam_sign * (Rm0 @ FL)
                jM[k] = -self.lam_sign * (Rm1 @ (self.torque_map(J) @ TL))
                jP[k] = np.array(mbs.GetMarkerOutput(J["mb"], ov.Position))
                jPa[k] = np.array(mbs.GetMarkerOutput(J["ma"], ov.Position))
            elif J["dist"] is not None:
                lam = self.lam_sign * float(np.ravel(mbs.GetObjectOutput(J["dist"], ov.Force))[0])
                pa = np.array(mbs.GetMarkerOutput(J["ma"], ov.Position))
                pb = np.array(mbs.GetMarkerOutput(J["mb"], ov.Position))
                e = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-300)
                jF[k] = DIST_SIGN * lam * e
                jP[k], jPa[k] = pb, pa
            else:                                           # gear – coordinates of the coupled joints
                jP[k] = jPa[k] = J["pa"]
            if J["rot"] is not None:
                jM[k] += Q[k]["rot"] * (Ra @ J["rot_axis"])
            if J["trans"] is not None:
                jF[k] += Q[k]["trans"] * (Ra @ J["trans_axis"])
            Rm0 = Ra @ J["frame"]
            jFl[k], jMl[k] = Rm0.T @ jF[k], Rm0.T @ jM[k]
        S.update(jF=jF, jM=jM, jP=jP, jPa=jPa, jFl=jFl, jMl=jMl, jq=q, jv=qd, jfr=fr, jlim=lim,
                 jQ=np.array([[Q[k]["rot"], Q[k]["trans"]] for k in range(nj)]).reshape(nj, 2),
                 dQ=np.array(dQ, float), dq=np.array(dq, float), dv=np.array(dv, float))
        # force elements
        nf = len(self.F)
        fval, flen = np.zeros(nf), np.zeros(nf)
        fF, fM, fPa, fPb = (np.zeros((nf, 3)) for _ in range(4))
        pdamp = pbush = epot_s = pelem = 0.0
        for k, f in enumerate(self.F):
            if f["kind"] == "torsion":
                jk = f["joint"]
                J = self.J[jk]
                u, v = q[jk, 0], qd[jk, 0]
                fval[k] = -(f.get("f0", 0.0) + f.get("k", 0.0) * (u - f.get("phi0", 0.0)) + f.get("c", 0.0) * v)
                flen[k] = u
                fM[k] = fval[k] * (rot(J["a"]) @ J["rot_axis"])
                fPa[k] = fPb[k] = jP[jk]
                epot_s += f.get("f0", 0.0) * u + 0.5 * f.get("k", 0.0) * (u - f.get("phi0", 0.0)) ** 2
                pdamp -= f.get("c", 0.0) * v * v
                pelem += fval[k] * v
                continue
            pa = np.array(mbs.GetMarkerOutput(f["ma"], ov.Position))
            pb = np.array(mbs.GetMarkerOutput(f["mb"], ov.Position))
            va = np.array(mbs.GetMarkerOutput(f["ma"], ov.Velocity))
            vb = np.array(mbs.GetMarkerOutput(f["mb"], ov.Velocity))
            fPa[k], fPb[k] = pa, pb
            if f["kind"] == "bushing":
                FL = np.array(mbs.GetObjectOutput(f["obj"], ov.ForceLocal))
                TL = np.array(mbs.GetObjectOutput(f["obj"], ov.TorqueLocal))
                Ra = rot(f.get("a", -1))
                fF[k], fM[k] = BUSH_SIGN * (Ra @ FL), BUSH_SIGN * (Ra @ TL)
                fval[k] = float(np.linalg.norm(FL))
                wa, wb = omega(f.get("a", -1)), omega(f.get("b", -1))
                pb_k = fF[k] @ (vb - va) + fM[k] @ (wb - wa) - np.cross(pb - pa, fF[k]) @ wa
                pbush += pb_k
                pelem += pb_k
                continue
            d = pb - pa
            L = float(np.linalg.norm(d))
            e = d / max(L, 1e-300)
            Ld = float(e @ (vb - va))
            ften = self.spring_force(f, L, Ld)
            fval[k], flen[k] = ften, L
            fF[k] = -ften * e                               # the tension pulls the point b towards the point a
            epot_s += self.spring_energy(f, L)
            cc = f.get("c", 0.0) if f["kind"] != "spring" else 0.0
            pdamp -= cc * Ld * Ld
            pelem -= ften * Ld
        S.update(fval=fval, flen=flen, fF=fF, fM=fM, fPa=fPa, fPb=fPb)
        # external loads
        nl = len(self.L)
        lF, lM, lP = (np.zeros((nl, 3)) for _ in range(3))
        pload = 0.0
        for k, ld in enumerate(self.L):
            c = self.load_factor(ld, t)
            lF[k], lM[k] = c * ld["F0"], c * ld["M0"]
            lP[k] = np.array(mbs.GetMarkerOutput(ld["m"], ov.Position))
            vp = np.array(mbs.GetMarkerOutput(ld["m"], ov.Velocity))
            pload += lF[k] @ vp + lM[k] @ omega(ld["body"])
        S.update(lF=lF, lM=lM, lP=lP)
        # energy and powers
        ek, ep = 0.0, epot_s
        for i, B in enumerate(self.B):
            if B["ground"]:
                continue
            R = S["rot"][i]
            Jg = R @ B["J"] @ R.T
            w = S["omega"][i]
            ek += 0.5 * B["mass"] * S["vel"][i] @ S["vel"][i] + 0.5 * w @ Jg @ w
            ep -= B["mass"] * self.g @ S["pos"][i]
        pdrv = float(np.dot(S["dQ"], S["dv"])) if len(dQ) else 0.0
        pfr = float(np.sum(fr * qd))
        plim = float(sum(lim[k] * qd[k, 0 if self.primary(J) == "rot" else 1] for k, J in enumerate(self.J)))
        S.update(ekin=ek, epot=ep, pdrv=pdrv, pfr=pfr, plim=plim, pdamp=pdamp, pbush=pbush, pload=pload,
                 pelem=pelem)
        return S

    def powers(self, t: float) -> dict:
        """Only the powers of the non-conservative forces (each integration step – the work is summed by the trapezium); the same formulas
        as the snapshot, without the other outputs.
        """
        mbs = self.mbs
        ov = self.exu.OutputVariableType
        P = {"wdrv": 0.0, "wfr": 0.0, "wlim": 0.0, "wdamp": 0.0, "wbush": 0.0, "wload": 0.0}
        rates = {}

        def rate(k, c):
            key = (k, c)
            if key not in rates:
                rates[key] = self.marker_coord(self.J[k][c])
            return rates[key]
        for d in self.drivers_kin:
            P["wdrv"] += self.driver_force(d) * rate(d["joint"], d["coord"])[1]
        for d in self.drivers_force:
            if not self.assembly:
                P["wdrv"] += d["law"].value(self.time(t)) * rate(d["joint"], d["coord"])[1]
        for k, J in enumerate(self.J):
            for c in J.get("cf", {}):
                u, v = rate(k, c)
                tot, ff, fl, fd, ft = self.coord_force(J, c, t, u, v)
                P["wfr"] += ff * v
                P["wlim"] += fl * v
        for f in self.F:
            if f["kind"] == "torsion":
                u, v = rate(f["joint"], "rot")
                P["wdamp"] -= f.get("c", 0.0) * v * v
            elif f["kind"] in ("damper", "spring_damper", "gas_spring") and f.get("c", 0.0):
                pa = np.array(mbs.GetMarkerOutput(f["ma"], ov.Position))
                pb = np.array(mbs.GetMarkerOutput(f["mb"], ov.Position))
                dv = np.array(mbs.GetMarkerOutput(f["mb"], ov.Velocity)) - np.array(mbs.GetMarkerOutput(f["ma"], ov.Velocity))
                e = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-300)
                P["wdamp"] -= f["c"] * float(e @ dv) ** 2
            elif f["kind"] == "bushing":
                P["wbush"] += self._bush_power(f)
        for ld in self.L:
            c = self.load_factor(ld, t)
            if c == 0.0:
                continue
            vp = np.array(mbs.GetMarkerOutput(ld["m"], ov.Velocity))
            n = self.B[ld["body"]]["node"]
            w = np.array(mbs.GetNodeOutput(n, ov.AngularVelocity))
            P["wload"] += c * (ld["F0"] @ vp + ld["M0"] @ w)
        return P

    def torque_map(self, J) -> np.ndarray:
        """Matrix T of the joint (according to the mask of the constrained axes including the drivers) – see calibrate_torque_map."""
        if "tmap" not in J:
            mask = list(MASKS.get(J["kind"], [1] * 6))
            for ax in J.get("driven_axes", []):
                mask[ax] = 1
            J["tmap"] = calibrate_torque_map(mask)
        return J["tmap"]

    def driver_force(self, d) -> float:
        """Moment / force of the kinematic driver on body_b in the direction of the coordinate (component of the reaction of the generic joint)."""
        ov = self.exu.OutputVariableType
        ax = d["axis"]
        if ax >= 3:
            TL = np.ravel(self.mbs.GetObjectOutput(d["obj"], ov.TorqueLocal))
            return -self.lam_sign * float((self.torque_map(self.J[d["joint"]]) @ TL)[ax - 3])
        return -self.lam_sign * float(np.ravel(self.mbs.GetObjectOutput(d["obj"], ov.ForceLocal))[ax])

    def _bush_power(self, f) -> float:
        mbs = self.mbs
        ov = self.exu.OutputVariableType
        FL = np.array(mbs.GetObjectOutput(f["obj"], ov.ForceLocal))
        TL = np.array(mbs.GetObjectOutput(f["obj"], ov.TorqueLocal))
        a, b = f.get("a", -1), f.get("b", -1)
        Ra = np.eye(3) if self.body_ground(a) else self.body_state(a)[1]
        F, M = BUSH_SIGN * (Ra @ FL), BUSH_SIGN * (Ra @ TL)
        pa = np.array(mbs.GetMarkerOutput(f["ma"], ov.Position))
        pb = np.array(mbs.GetMarkerOutput(f["mb"], ov.Position))
        va = np.array(mbs.GetMarkerOutput(f["ma"], ov.Velocity))
        vb = np.array(mbs.GetMarkerOutput(f["mb"], ov.Velocity))
        wa = np.zeros(3) if self.body_ground(a) else self.body_state(a)[3]
        wb = np.zeros(3) if self.body_ground(b) else self.body_state(b)[3]
        return float(F @ (vb - va) + M @ (wb - wa) - np.cross(pb - pa, F) @ wa)

    def current_rates(self) -> dict:
        """Velocities of the coordinates of the joints and the lengths of the elements from the current state (ODE2_t) – for the static solutions of the series."""
        mbs = self.mbs
        ov = self.exu.OutputVariableType
        r = {}
        for J in self.J:
            for c in ("rot", "trans"):
                if J[c] is not None:
                    r[(id(J), c)] = self.marker_coord(J[c])[1]
        for f in self.F:
            if f["kind"] in ("spring", "damper", "spring_damper", "gas_spring"):
                pa = np.array(mbs.GetMarkerOutput(f["ma"], ov.Position))
                pb = np.array(mbs.GetMarkerOutput(f["mb"], ov.Position))
                dv = np.array(mbs.GetMarkerOutput(f["mb"], ov.Velocity)) - np.array(mbs.GetMarkerOutput(f["ma"], ov.Velocity))
                r[("F", id(f))] = float((pb - pa) @ dv / max(np.linalg.norm(pb - pa), 1e-300))
        return r

    def set_inertia(self, on: bool) -> None:
        """Inertia forces of the bodies from the current state (accelerations in ODE2_tt) as loads – or switch off."""
        for i, B in enumerate(self.B):
            if B["ground"] or B.get("flex"):
                continue
            if not on:
                B["iF"], B["iM"] = np.zeros(3), np.zeros(3)
                continue
            c, R, v, w, a, al = self.body_state(i)
            Jg = R @ B["J"] @ R.T
            B["iF"] = -B["mass"] * a
            B["iM"] = -(Jg @ al + np.cross(w, Jg @ w))

    def update_states(self, t_prev=None, t=None):
        """After a converged step: the reactions of the joints for the friction in the next step (decision 2)."""
        ov = self.exu.OutputVariableType
        sg = self.lam_sign
        for J in self.J:
            if J["generic"] is not None and J["fric"]:
                # force on body_b in the frame of the joint (marker 0) – for the normal force of the friction the magnitudes of the components suffice
                J["Rprev"] = -sg * np.array(self.mbs.GetObjectOutput(J["generic"], ov.ForceLocal))


DIST_SIGN = -1.0        # ObjectConnectorDistance: force on body b = DIST_SIGN·λ·e (e from a to b)
# ObjectJointGeneric (verified scratch/f8_motion/proto6.py, proto7.py, proto11.py): ForceLocal in the frame of marker 0, in the dynamics
# force on body_b = −R_m0·FL; TorqueLocal in the frame of marker 1, but the assignment of the components to the axes of the moment depends on the mask of the constrained
# rotations (all 6: diag(−1, 1, 1), rotational joint: (M_x, M_y) = (−TL_y, TL_x)…) → the matrix T is found by calibration
# (Model.torque_map); the static solver has the Lagrange multipliers with the opposite sign (lam_sign = −1)
_TORQUE_MAPS: dict = {}
_FLEX_CACHE: dict = {}          # (component, shapes, interface) → FEMinterface with the shapes Craig–Bampton (once per process)


def calibrate_torque_map(mask) -> np.ndarray:
    """T (3×3): moment of the joint on body_b in the frame of marker 1 = T·TorqueLocal in the STATIC solution (in the dynamics −T).
    Calibration: a body at the origin, a joint with the given mask of the rotation (the displacements constrained), unit moments about the constrained
    axes → TorqueLocal; T = M·pinv(TL).
    """
    key = tuple(int(x) for x in mask[3:6])
    if key in _TORQUE_MAPS:
        return _TORQUE_MAPS[key]
    import exudyn as exu
    from exudyn import itemInterface as ii
    rot = [i for i in range(3) if key[i]]
    Ms, TLs = [], []
    for i in rot:
        SC = exu.SystemContainer()
        mbs = SC.AddSystem()
        g = mbs.AddObject(ii.ObjectGround())
        n = mbs.AddNode(ii.NodeRigidBodyEP(referenceCoordinates=[0, 0, 0, 1, 0, 0, 0]))
        b = mbs.AddObject(ii.ObjectRigidBody(physicsMass=1.0, inertia=[1, 1, 1, 0, 0, 0], nodeNumber=n))
        ma = mbs.AddMarker(ii.MarkerBodyRigid(bodyNumber=g, localPosition=[0, 0, 0]))
        mb = mbs.AddMarker(ii.MarkerBodyRigid(bodyNumber=b, localPosition=[0, 0, 0]))
        jt = mbs.AddObject(ii.ObjectJointGeneric(markerNumbers=[ma, mb], constrainedAxes=[1, 1, 1, *key]))
        e = [0.0, 0.0, 0.0]
        e[i] = 1.0
        mbs.AddLoad(ii.LoadTorqueVector(markerNumber=mb, loadVector=e))
        mbs.Assemble()
        ss = exu.SimulationSettings()
        ss.solution.file.write = False
        ss.staticSolver.verboseMode = 0
        ss.linearSolver.solverType = exu.LinearSolverType.EigenDense
        ss.linearSolver.ignoreSingularJacobian = True
        mbs.SolveStatic(ss)
        TLs.append(np.array(mbs.GetObjectOutput(jt, exu.OutputVariableType.TorqueLocal), float))
        Ms.append(-np.array(e))                         # the moment of the joint on the body is held by the load
    T = np.zeros((3, 3))
    if rot:
        T = np.array(Ms).T @ np.linalg.pinv(np.array(TLs).T)
        T[np.abs(T) < 1e-9] = 0.0
    _TORQUE_MAPS[key] = T
    return T
BUSH_SIGN = -1.0        # ObjectConnectorRigidBodySpringDamper: ForceLocal/TorqueLocal acts on the body and
