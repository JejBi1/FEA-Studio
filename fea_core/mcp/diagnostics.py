"""Evaluation of the results for the LLM: fields to display, summary, places of the maxima with the classification of the singularities.

Classification of the maximum (heuristics over the mesh, with a justification):
  - singularity: an edge of a clamped / constrained face, a sharp inner (concave) edge or a corner without a fillet,
    a small (almost point) loaded face, connection of a bolt (rigid constraint of the beam);
  - probably influenced by the boundary condition: inside a face of a constraint / at a constraint, an edge of a loaded face;
  - real concentration: fillet, hole, smooth surface → check by the convergence of the mesh.
A steep gradient (node ≫ surroundings at the distance of ~1 element) is stated as a sign of the dependence on the mesh.

Heat (fea_core.thermal): fields temperature / heat_flux of the thermal states (Results.thermal), summary with the fluxes of the boundary
conditions and the heat balance, the hottest places. The topology of the mesh (KD tree, surface, selections, node → elements) is computed
once per object Results (MeshCache) – the repeated calls of the summary, of the places of the maxima and of the render share it.
"""
from __future__ import annotations

import math

import numpy as np

from .session import FeaError, Session

# key: (description, unit)
FIELDS = {
    "mises": ("von Mises", "MPa"), "s1": ("1st principal stress σ1", "MPa"), "s2": ("2nd principal stress σ2", "MPa"),
    "s3": ("3rd principal stress σ3", "MPa"), "tresca": ("Tresca (σ1 − σ3)", "MPa"),
    "sxx": ("σx", "MPa"), "syy": ("σy", "MPa"), "szz": ("σz", "MPa"), "sxy": ("τxy", "MPa"),
    "syz": ("τyz", "MPa"), "szx": ("τzx", "MPa"),
    "displacement": ("Total displacement |u|", "mm"), "ux": ("Displacement ux", "mm"), "uy": ("Displacement uy", "mm"),
    "uz": ("Displacement uz", "mm"),
    "safety": ("Safety factor to Re (Re/σ_vM, max 10)", "–"),
    "print_safety": ("3D printing – safety factor (max 10)", "–"), "print_index": ("3D printing – failure index", "–"),
    "fatigue_damage": ("Fatigue – damage D (log10)", "log10 D"), "fatigue_life": ("Fatigue – life (log10 blocks)", "log10 blocks"),
    "fatigue_safety": ("Fatigue – safety factor (max 10)", "–"), "fatigue_amplitude": ("Fatigue – equivalent stress amplitude", "MPa"),
    "temperature": ("Temperature", "°C"), "heat_flux": ("Heat flux |q|", "W/m²"),
    "hfx": ("Heat flux q_x", "W/m²"), "hfy": ("Heat flux q_y", "W/m²"), "hfz": ("Heat flux q_z", "W/m²"),
    "stress_error": ("ccx stress error estimate (STR)", "%"),
    "topo_density": ("Topology – density (1 = material, 0 = removed)", "–"),
}
ALIASES = {"von_mises": "mises", "vm": "mises", "seqv": "mises", "u": "displacement", "umag": "displacement",
           "disp": "displacement", "usum": "displacement", "sf": "safety", "s_1": "s1", "s_3": "s3",
           "t": "temperature", "temp": "temperature", "temperature": "temperature", "nt": "temperature",
           "flux": "heat_flux", "hfl": "heat_flux", "tok": "heat_flux", "q": "heat_flux", "heatflux": "heat_flux",
           "error": "stress_error", "str": "stress_error", "err": "stress_error", "error": "stress_error", "teplota": "temperature", "chyba": "stress_error"}
REVERSED = {"safety", "print_safety", "fatigue_safety", "fatigue_life"}
SIGNED = {"s1", "s2", "s3", "sxx", "syy", "szz", "sxy", "syz", "szx", "ux", "uy", "uz", "hfx", "hfy", "hfz"}
THERMAL_KEYS = {"temperature", "heat_flux", "hfx", "hfy", "hfz"}
FLUX_KEYS = THERMAL_KEYS - {"temperature"}

# thermal boundary conditions (fea_core.thermal.model.THERMAL_KINDS) – descriptions for the LLM, pictures and the report
THERMAL_LABELS = {"fixed_temperature": "prescribed temperature", "heat_flux": "heat flux", "heat_flow": "heat power",
                  "convection": "heat transfer (convection)", "radiation": "radiation to ambient",
                  "heat_generation": "internal heat generation"}


def no_solid_hint(res) -> str:
    """Message for the tools of the 3D results when the result has no 3D mesh (bars, 2D section, shells)."""
    if getattr(res, "twod", None) is not None:
        return ("The result is a 2D section analysis (no 3D mesh) – fea_twod_results, fea_twod_probe, fea_twod_line, fea_twod_linearize, fea_twod_render")
    if getattr(res, "shell", None) is not None:
        return "The result is a shell model (no 3D mesh) – fea_shell_results, fea_shell_probe, fea_shell_render"
    return "The study has no solid bodies (beam model only) – member results: fea_frame_results, image fea_frame_render"


def field_key(field: str) -> str:
    k = ALIASES.get(field.strip().lower(), field.strip().lower())
    if k not in FIELDS:
        raise FeaError(f"Unknown quantity '{field}'. Available: {', '.join(FIELDS)}")
    return k


def bc_value_text(bc) -> str:
    """Value of a thermal boundary condition with units (decimal point – for the LLM and the pictures)."""
    k = bc.kind
    if k == "fixed_temperature":
        return f"T = {bc.value:g} °C"
    if k == "heat_flux":
        return f"q = {bc.value:g} W/m²"
    if k == "heat_flow":
        return f"Q = {bc.value:g} W"
    if k == "convection":
        return f"h = {bc.h:g} W/(m²·K), ambient {bc.ambient:g} °C"
    if k == "radiation":
        return f"ε = {bc.emissivity:g}, ambient {bc.ambient:g} °C"
    if k == "heat_generation":
        return f"{bc.value:g} W in the volume of part {bc.part}"
    return k


def thermal_load_text(l) -> str:
    """Description of the mechanical thermal load (study.ThermalLoad)."""
    if l.case:
        return f"temperature field from thermal state '{l.case}'" + ("" if l.time is None else f" at time {l.time:g} s")
    parts = ", ".join(l.parts) if l.parts else "of all parts"
    return f"uniform temperature {l.value:g} °C ({('parts ' + parts if l.parts else parts)})"


def field_values(sess: Session, step: int, key: str) -> np.ndarray:
    res = sess.require_results()
    st = res.steps[step]
    if key == "mises":
        return st.mises
    if key in ("s1", "s2", "s3", "tresca"):
        p = st.principal
        return p[:, 0] - p[:, 2] if key == "tresca" else p[:, int(key[1]) - 1]
    comp = {"sxx": 0, "syy": 1, "szz": 2, "sxy": 3, "syz": 4, "szx": 5}
    if key in comp:
        return st.S[:, comp[key]]
    if key == "displacement":
        return st.umag
    if key in ("ux", "uy", "uz"):
        return st.U[:, "xyz".index(key[1])]
    if key == "safety":
        re = sess.node_re()
        if not np.isfinite(re).any():
            raise FeaError("No material has Re – safety factor cannot be computed")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.minimum(np.where(np.isfinite(re), re / np.maximum(st.mises, 1e-12), 10.0), 10.0)
    if key in ("print_safety", "print_index"):
        v = st.fields.get(key)
        if v is None:
            raise FeaError("The 3D printing quantity is not in the results (no printed part)")
        return np.minimum(np.nan_to_num(v, nan=10.0, posinf=10.0), 10.0) if key == "print_safety" else np.nan_to_num(v)
    if key.startswith("fatigue_"):
        fr = sess.fatigue
        if fr is None:
            raise FeaError("Fatigue has not been computed – run fea_fatigue_run")
        if key == "fatigue_damage":
            return np.log10(np.maximum(fr.total_damage, 1e-12))
        if key == "fatigue_life":
            return np.log10(np.minimum(np.maximum(fr.life_blocks, 1.0), 1e12))
        if key == "fatigue_safety":
            return np.minimum(np.nan_to_num(fr.safety, nan=10.0, posinf=10.0), 10.0)
        return fr.amplitude
    if key == "temperature":
        T = st.fields.get("T")
        if T is None:
            raise FeaError(f"Case '{st.name}' has no thermal load – temperature is in the thermal case result (parameter thermal_case)")
        return T
    if key in FLUX_KEYS:
        raise FeaError("Heat flux is available only in thermal case results – specify thermal_case")
    if key == "stress_error":
        v = st.fields.get("stress_error")
        if v is None:
            raise FeaError("The stress error estimate (STR) is available only in CalculiX results (ccx backend) – not in NGSolve")
        return v
    if key == "topo_density":
        v = st.fields.get("topo_density")
        if v is None:
            raise FeaError("The result has no topology optimization – run fea_topo_opt_run")
        return v
    raise FeaError(f"Quantity '{key}' is not available")


def thermal_values(sess: Session, k: int, key: str, time_s: float | None = None) -> np.ndarray:
    """Field of the thermal state k: temperature (for a transient state at the time time_s) or heat flux (the end of the state)."""
    res = sess.require_results()
    tr = res.thermal[k]
    if key == "temperature":
        return tr.at_time(time_s)
    if key not in FLUX_KEYS:
        raise FeaError(f"Quantity '{key}' belongs to a mechanical case – a thermal case has only temperature, heat_flux, hfx, hfy, hfz")
    if time_s is not None and tr.transient and tr.times and \
            abs(float(time_s) - tr.times[-1]) > 1e-9 * max(1.0, abs(tr.times[-1])):
        raise FeaError(f"Heat flux of transient case '{tr.name}' is stored only at the end of the case (t = {tr.times[-1]:g} s) – omit time_s")
    if key == "heat_flux":
        return tr.flux_mag
    return tr.flux[:, "xyz".index(key[2])]


def resolve_source(sess: Session, key: str, step=0, thermal_case=None) -> tuple[str, int]:
    """Where to take the field: ("thermal", index of the thermal state) or ("step", index of the mechanical state).

    thermal_case given → thermal state; heat flux → the first thermal state; temperature → the mechanical state with a thermal load (step), otherwise the first thermal state; other quantities → the mechanical state step.
    """
    res = sess.require_results()
    if not len(res.mesh.nodes):
        raise FeaError(no_solid_hint(res))
    if thermal_case is not None:
        return "thermal", sess.thermal_index(thermal_case)
    if key in FLUX_KEYS:
        return "thermal", sess.thermal_index(0)
    if key == "temperature":
        if res.steps:
            i = sess.step_index(step)
            if "T" in res.steps[i].fields:
                return "step", i
        if res.thermal:
            return "thermal", 0
        raise FeaError("Results contain no temperature (no thermal case or thermal load in the case)")
    return "step", sess.step_index(step)


# ===================================================================== geometry of the surface of the mesh

class SurfaceInfo:
    """Surface of the mesh for the classification: edges of tri6 (corners a, b + midpoint m), sharp / concave edges, nodes of the faces."""

    def __init__(self, mesh, boundary=None):
        tri, elem = boundary if boundary is not None else mesh.boundary_faces()
        self.tri = tri
        self.surface_nodes = np.zeros(len(mesh.nodes), bool)
        self.surface_nodes[np.unique(tri)] = True
        X = mesh.nodes
        c = tri[:, :3]
        n = np.cross(X[c[:, 1]] - X[c[:, 0]], X[c[:, 2]] - X[c[:, 0]])
        self.normals = n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
        self.centroids = X[c].mean(axis=1)
        mids = tri[:, 3:6] if tri.shape[1] == 6 else np.full((len(tri), 3), -1)
        a = np.concatenate([c[:, 0], c[:, 1], c[:, 2]])
        b = np.concatenate([c[:, 1], c[:, 2], c[:, 0]])
        m = np.concatenate([mids[:, 0], mids[:, 1], mids[:, 2]])
        t = np.tile(np.arange(len(tri)), 3)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo.astype(np.int64) * (len(X) + 1) + hi
        order = np.argsort(key, kind="stable")
        k = key[order]
        start = np.r_[True, k[1:] != k[:-1]]
        first = order[start]
        second = np.full(len(first), -1)
        nxt = np.nonzero(~start)[0]
        gid = np.cumsum(start) - 1
        if len(nxt):
            g2 = gid[nxt]
            _, f2 = np.unique(g2, return_index=True)
            second[g2[f2]] = order[nxt[f2]]
        ok = second >= 0
        t1, t2 = t[first[ok]], t[second[ok]]
        n1, n2 = self.normals[t1], self.normals[t2]
        flat_bend = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n1, n2), -1, 1)))
        if tri.shape[1] == 6:
            # normals of the quadratic face tri6 at the midpoint of a common edge: on a smooth (also coarsely divided) cylindrical face they agree,
            # on a sharp edge of the CAD they do not → a hole / a fillet is not mistaken for a sharp corner
            from ..loads import _tri6_dN
            en = np.zeros((len(tri), 3, 3))
            for kk, (xi, eta) in enumerate(((0.5, 0.0), (0.5, 0.5), (0.0, 0.5))):
                dxi, deta = _tri6_dN(xi, eta)
                P = X[tri]
                v = np.cross(np.einsum("i,kij->kj", dxi, P), np.einsum("i,kij->kj", deta, P))
                en[:, kk] = v / np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]
            kidx = np.repeat([0, 1, 2], len(tri))
            q1, q2 = en[t1, kidx[first[ok]]], en[t2, kidx[second[ok]]]
            bend = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", q1, q2), -1, 1)))
        else:
            bend = flat_bend
        concave = np.einsum("ij,ij->i", self.centroids[t2] - self.centroids[t1], n1) > 0
        ea, eb, em = a[first[ok]], b[first[ok]], m[first[ok]]
        self.edge_nodes = np.stack([ea, eb, em], axis=1)
        self.edge_bend = bend
        self.edge_concave = concave
        # node → max. bending of the sharp concave / convex edge and the curvature (small bending)
        N = len(X)
        self.reentrant = np.zeros(N)
        self.convex = np.zeros(N)
        self.curved = np.zeros(N)
        for col in range(3):
            nodes = self.edge_nodes[:, col]
            okn = nodes >= 0
            sharp = bend >= 25
            np.maximum.at(self.reentrant, nodes[okn & sharp & concave], bend[okn & sharp & concave])
            np.maximum.at(self.convex, nodes[okn & sharp & ~concave], bend[okn & sharp & ~concave])
            soft = (flat_bend > 1.0) & (bend < 25)
            np.maximum.at(self.curved, nodes[okn & soft], flat_bend[okn & soft])


def _tri_keys(tri: np.ndarray, n: int) -> np.ndarray:
    """Key of a triangle by the sorted corner nodes (for comparing sets of triangles)."""
    s = np.sort(tri[:, :3], axis=1).astype(np.int64)
    if n < 2_000_000:                                   # n³ < 2⁶³
        return (s[:, 0] * n + s[:, 1]) * n + s[:, 2]
    return np.ascontiguousarray(s).view(np.dtype((np.void, 24))).ravel()


class MeshCache:
    """Cache of the topology of the mesh of the result for the diagnostics and the render (the mesh of the result does not change).

    Stored on the object Results (`mesh_cache(res)`) → new results = new cache. The content is computed lazily:
    the outer surface (boundary_faces), SurfaceInfo, KD tree of the nodes, node → elements (CSR), data of the selections of faces.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._boundary = None
        self._surface = None
        self._tree = None
        self._n2e = None
        self._sel: dict[str, tuple] = {}
        self._keys = None
        self._re = None
        self._h = None

    @property
    def boundary(self) -> tuple[np.ndarray, np.ndarray]:
        if self._boundary is None:
            self._boundary = self.mesh.boundary_faces()
        return self._boundary

    @property
    def surface(self) -> SurfaceInfo:
        if self._surface is None:
            self._surface = SurfaceInfo(self.mesh, self.boundary)
        return self._surface

    @property
    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.mesh.nodes)
        return self._tree

    @property
    def mean_edge(self) -> float:
        if self._h is None:
            self._h = _mean_edge(self.mesh)
        return self._h

    @property
    def reentrant(self):
        """(nodes on the sharp concave edges, their KD tree | None)."""
        if self._re is None:
            from scipy.spatial import cKDTree
            idx = np.nonzero(self.surface.reentrant > 0)[0]
            self._re = (idx, cKDTree(self.mesh.nodes[idx]) if len(idx) else None)
        return self._re

    def node_elements(self, n: int) -> np.ndarray:
        """Elements containing the node n (ascending)."""
        if self._n2e is None:
            t = self.mesh.tets
            flat = t.ravel()
            order = np.argsort(flat, kind="stable")
            start = np.zeros(len(self.mesh.nodes) + 1, np.int64)
            np.cumsum(np.bincount(flat, minlength=len(self.mesh.nodes)), out=start[1:])
            self._n2e = (order // t.shape[1], start)
        el, start = self._n2e
        return el[start[n]:start[n + 1]]

    def node_part(self, n: int) -> str:
        """Part of the first element with the node n."""
        return self.mesh.part_names[int(self.mesh.tet_part[self.node_elements(n)[0]])]

    def node_parts(self, n: int) -> list[str]:
        return sorted({self.mesh.part_names[int(p)] for p in np.unique(self.mesh.tet_part[self.node_elements(n)])})

    def part_extrema(self, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(min, max, number of elements) of a nodal field by parts (over all nodes of the elements of the part)."""
        m = self.mesh
        ev = vals[m.tets]
        P = len(m.part_names)
        pmax = np.full(P, -np.inf)
        pmin = np.full(P, np.inf)
        np.maximum.at(pmax, m.tet_part, ev.max(axis=1))
        np.minimum.at(pmin, m.tet_part, ev.min(axis=1))
        return pmin, pmax, np.bincount(m.tet_part, minlength=P)

    def selection(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """(mask of the nodes of the selection, mask of the nodes on the edge of the selection, indices of the edge nodes, area of the face)."""
        if name not in self._sel:
            m = self.mesh
            tri_s = m.surfaces[name]
            mask = np.zeros(len(m.nodes), bool)
            mask[np.unique(tri_s)] = True
            edge = self._sel_boundary(tri_s, mask)
            self._sel[name] = (mask, edge, np.nonzero(edge)[0], m.surface_area(name))
        return self._sel[name]

    def _sel_boundary(self, tri_sel: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Nodes on the edge of the selection (shared with a surface triangle outside the selection)."""
        n = len(self.mesh.nodes)
        tri = self.surface.tri
        if self._keys is None:
            self._keys = _tri_keys(tri, n)
        touching = mask[tri].any(axis=1)
        outside = touching & ~np.isin(self._keys, _tri_keys(tri_sel, n))
        edge = np.zeros(n, bool)
        if outside.any():
            nodes = np.unique(tri[outside])
            edge[nodes[mask[nodes]]] = True
        return edge


def mesh_cache(res) -> MeshCache:
    c = getattr(res, "_mcp_cache", None)
    if c is None or c.mesh is not res.mesh:
        c = MeshCache(res.mesh)
        res._mcp_cache = c
    return c


def _is_sliding(mesh, c) -> bool:
    """A prescribed ZERO displacement only perpendicular to a planar face (symmetry, sliding support) – it does not prevent the
    lateral contraction, so the edge of such a face is not a singularity (unlike a clamping).
    """
    if c.kind != "displacement" or c.selection not in mesh.surfaces:
        return False
    vals = [(k, v) for k, v in enumerate((c.ux, c.uy, c.uz)) if v is not None]
    if not vals or any(abs(v) > 0 for _, v in vals):
        return False
    tri = mesh.surfaces[c.selection][:, :3]
    if not len(tri):
        return False
    X = mesh.nodes
    n = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    n0 = n[0]
    if np.min(np.abs(n @ n0)) < 0.999:                       # not a plane
        return False
    return all(abs(n0[k]) > 0.999 for k, _ in vals)


def _mean_edge(mesh) -> float:
    t = mesh.tets[: min(len(mesh.tets), 5000), :4]
    X = mesh.nodes
    e = [np.linalg.norm(X[t[:, i]] - X[t[:, j]], axis=1) for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
    return float(np.mean(e))


def _pick(X: np.ndarray, vals: np.ndarray, top_n: int, r_cluster: float) -> list[int]:
    """Nodes with the highest values in separate places (farther from each other than r_cluster)."""
    order = np.argsort(-vals)
    cand = order[: max(2000, 50 * top_n)]
    P = X[cand]
    alive = np.ones(len(cand), bool)
    picked: list[int] = []
    while len(picked) < top_n:
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        j = idx[0]
        picked.append(int(cand[j]))
        alive &= np.linalg.norm(P - P[j], axis=1) > r_cluster
    return picked


def hotspots(sess: Session, step: int, top_n: int = 5, field: str = "mises", tags: bool = False) -> dict:
    """Places of the maxima of the field with the classification (see the module). tags=True adds to the places "tags" = short labels of the kind of
    the singularity / influence of the boundary condition for the GUI (e.g. "clamping edge", "sharp inner edge", "point
    load"); the output of MCP is without them.
    """
    res = sess.require_results()
    st_res = res.steps[step]
    study = sess.study
    mesh = res.mesh
    key = field_key(field)
    vals = field_values(sess, step, key)
    if key in SIGNED:
        vals = np.abs(vals)
    c = mesh_cache(res)
    surf = c.surface
    h = c.mean_edge
    diag = float(np.linalg.norm(mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0)))
    r_cluster = max(3.0 * h, 0.02 * diag)
    # role of the selections
    roles: dict[str, tuple[str, str]] = {}
    for con in study.constraints:
        roles[con.selection] = ("sliding" if _is_sliding(mesh, con) else "constraint", con.kind)
    for lc in study.load_cases:
        for l in lc.loads:
            if getattr(l, "selection", None):
                roles.setdefault(l.selection, ("load", l.kind))
    for name in mesh.surfaces:
        if name.startswith("__b"):
            roles[name] = ("bolt", "bolt")
    line_loads = {name: np.unique(mesh.edges[name]) if name in mesh.edges else np.unique(mesh.points[name])
                  for name in {getattr(l, "selection", None) for lc in study.load_cases for l in lc.loads}
                  if name in mesh.edges or name in mesh.points}

    picked = _pick(mesh.nodes, vals, top_n, r_cluster)
    err_field = st_res.fields.get("stress_error")          # estimate of the error of ccx (STR %, informative)
    tree = c.tree
    p_all = getattr(st_res, "_principal", None)          # principal stresses: all nodes only when already cached,
    if p_all is None:                                       # otherwise just the picked ones (eigvalsh per matrix = the same values)
        from ..results import principal
        p_all = dict(zip(picked, principal(st_res.S[np.asarray(picked, int)]))) if picked else {}
    re_idx, re_tree = c.reentrant

    def near_set(idx: np.ndarray, x) -> float:
        """Distance of a point from the nearest node of the set (inf = empty)."""
        if not len(idx):
            return math.inf
        return float(np.min(np.linalg.norm(mesh.nodes[idx] - x, axis=1)))
    out_list = []
    for rank, n in enumerate(picked, 1):
        x = mesh.nodes[n]
        reasons, sing, bc = [], False, False
        tag_s, tag_b = [], []                        # short labels of the singularities / influences of the boundary conditions (tags=True)
        for name, (role, kind) in roles.items():
            if name not in mesh.surfaces or len(mesh.surfaces[name]) == 0:
                continue
            mask, edge, edge_idx, area = c.selection(name)
            if role == "sliding":
                if mask[n]:
                    reasons.append(f"on symmetry plane / sliding support '{name}' (displacement normal to the face only) – no singularity")
            elif role in ("constraint", "bolt"):
                de = near_set(edge_idx, x)
                if de <= 1.0 * h:
                    where = "on edge" if de < 1e-9 else f"{de:.2g} mm (≤ 1 element) from the edge"
                    reasons.append(f"{where} of {('constraint' if role == 'constraint' else 'bolt connection')} face '{name}' ({kind}) – stiffness discontinuity, stress grows with mesh refinement")
                    sing = True
                    tag_s.append("bolt connection" if role == "bolt" else
                                 "fixed support edge" if kind == "fixed" else "constraint edge")
                elif mask[n]:
                    reasons.append(f"on support face '{name}' ({kind}) – the support restrains lateral contraction")
                    bc = True
                    tag_b.append("on support face")
                else:
                    near = tree.query_ball_point(x, 1.5 * h)
                    if any(mask[j] for j in near):
                        reasons.append(f"within 1.5 elements of support '{name}' – affected by the boundary condition")
                        bc = True
                        tag_b.append("at the constraint")
            elif role == "load" and mask[n]:
                if area < (3 * h) ** 2:
                    reasons.append(f"small loaded face '{name}' ({area:.3g} mm² < (3·element)²) – close to a point force")
                    sing = True
                    tag_s.append("point load")
                elif edge[n]:
                    reasons.append(f"on the edge of loaded face '{name}' – load discontinuity")
                    bc = True
                    tag_b.append("edge of loaded face")
        for name, idx in line_loads.items():           # force on an edge / at a point: singularity in its surroundings
            d = near_set(idx, x)
            if d <= 2.0 * h:
                what = "an edge" if name in mesh.edges else "a point"
                reasons.append(f"{('at the location' if d < 1e-09 else f'{d:.2g} mm (≤ 2 elements) from')} of the force into {what} '{name}' – a point / line force is singular, stress grows with mesh refinement")
                sing = True
                tag_s.append("edge force" if what == "an edge" else "point force")
        dre, jre = (re_tree.query(x) if re_tree is not None else (math.inf, -1))
        if dre <= 1.0 * h:
            bend = surf.reentrant[re_idx[jre]]
            where = ("on a sharp inner (concave) edge/corner" if dre < 1e-9 else
                     f"{dre:.2g} mm (≤ 1 element) from a sharp inner (concave) edge/corner")
            reasons.append(f"{where}, angle change {bend:.0f}° without a fillet – theoretically infinite stress, the value depends on the mesh")
            sing = True
            tag_s.append("sharp inner edge")
        elif surf.convex[n] > 0 and not reasons:
            reasons.append(f"outer (convex) edge, angle change {surf.convex[n]:.0f}° – not a singularity by itself")
        if surf.curved[n] > 0 and not sing:
            reasons.append("curved surface (fillet/hole) – typical location of a real stress concentration")
        parts = c.node_parts(n)
        if len(parts) > 1:
            reasons.append(f"at the interface of parts {', '.join(parts)}")
        if not surf.surface_nodes[n]:
            reasons.append("inside the material")
        near = [j for j in tree.query_ball_point(x, 1.2 * h) if j != n]
        grad = float(vals[n] / max(np.mean(vals[near]), 1e-30)) if near else 1.0
        if grad > 1.4:
            reasons.append(f"steep gradient: {grad:.2f}× the neighborhood average within 1.2 elements – mesh dependent")
        if sing:
            cls = "singularity"
            rec = ("Do not evaluate from the nodal value: evaluate the stress farther from the location (≥ 2–3 elements), in a section, or the nominal stress; for a sharp corner add a fillet in CAD, for a support/force model the real attachment.")
        elif bc:
            cls = "affected by boundary condition"
            rec = "Verify that the support/load matches reality (support stiffness, contact area); evaluate away from the support."
        else:
            cls = "probably real concentration"
            rec = "Verify by mesh convergence (refine locally 2× – change < 5 %); assess against Re / fatigue."
        row = {
            "rank": rank, "node": n + 1, "xyz_mm": x, "value": float(vals[n]),
            "mises_MPa": float(st_res.mises[n]), "s1_MPa": float(p_all[n][0]), "s3_MPa": float(p_all[n][2]),
            "parts": parts, "on_surface": bool(surf.surface_nodes[n]),
            "classification": cls, "reasons": reasons, "recommendation": rec,
        }
        if "T" in st_res.fields:
            row["T_C"] = float(st_res.fields["T"][n])
        if err_field is not None:
            row["stress_error_pct"] = float(err_field[n])
        if tags:
            row["tags"] = list(dict.fromkeys(tag_s if sing else tag_b if bc else []))
        out_list.append(row)
    return {"field": key, "unit": FIELDS[key][1], "step": st_res.name, "mean_element_edge_mm": h,
            "cluster_radius_mm": r_cluster, "hotspots": out_list}


def thermal_hotspots(sess: Session, k: int, top_n: int = 5, field: str = "temperature",
                     time_s: float | None = None) -> dict:
    """The hottest places (temperature) or the places of the highest heat flux of the thermal state k (separate places),
    with the boundary conditions at the place; for the flux a sharp inner edge = singularity. In addition the coldest place.
    """
    res = sess.require_results()
    mesh = res.mesh
    tr = res.thermal[k]
    key = field_key(field)
    vals = thermal_values(sess, k, key, time_s)
    rank_vals = np.abs(vals) if key in SIGNED else vals
    T = tr.at_time(time_s)
    q = tr.flux_mag
    c = mesh_cache(res)
    surf = c.surface
    h = c.mean_edge
    diag = float(np.linalg.norm(mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0)))
    r_cluster = max(3.0 * h, 0.02 * diag)
    case = next((cs for cs in sess.study.thermal.cases if cs.name == tr.name), None) if sess.study else None
    bcs = list(case.loads) if case is not None else []
    re_idx, re_tree = c.reentrant
    out_list = []
    for rank, n in enumerate(_pick(mesh.nodes, rank_vals, top_n, r_cluster), 1):
        x = mesh.nodes[n]
        reasons, cls = [], None
        parts = c.node_parts(n)
        for bc in bcs:
            sel = getattr(bc, "selection", None)
            if sel is not None:
                if sel in mesh.surfaces and len(mesh.surfaces[sel]) and c.selection(sel)[0][n]:
                    reasons.append(f"on face '{sel}': {THERMAL_LABELS.get(bc.kind, bc.kind)} {bc_value_text(bc)}")
            elif bc.kind == "heat_generation" and bc.part in parts:
                reasons.append(f"in part '{bc.part}' with heat generation {bc.value:g} W")
        if len(parts) > 1:
            reasons.append(f"at the interface of parts {', '.join(parts)} (perfect thermal contact)")
        if not surf.surface_nodes[n]:
            reasons.append("inside the material")
        if key in FLUX_KEYS:
            dre = re_tree.query(x)[0] if re_tree is not None else math.inf
            if dre <= h:
                reasons.append("at a sharp inner (concave) edge/corner – heat flux is theoretically infinite there, the value depends on the mesh")
                cls = "singularity"
            else:
                cls = "probably the true flux maximum"
        row = {"rank": rank, "node": n + 1, "xyz_mm": x, "value": float(vals[n]), "T_C": float(T[n]),
               "heat_flux_W_m2": float(q[n]), "parts": parts, "on_surface": bool(surf.surface_nodes[n]),
               "reasons": reasons}
        if cls:
            row["classification"] = cls
        out_list.append(row)
    d = {"field": key, "unit": FIELDS[key][1], "thermal_case": tr.name, "transient": tr.transient,
         "mean_element_edge_mm": h, "cluster_radius_mm": r_cluster, "hotspots": out_list}
    if tr.transient:
        d["time_s"] = float(tr.times[-1]) if time_s is None and tr.times else time_s
    if key == "temperature":
        i = int(np.argmin(T))
        d["coldest"] = {"node": i + 1, "xyz_mm": mesh.nodes[i], "T_C": float(T[i]), "parts": c.node_parts(i)}
    return d


# ===================================================================== souhrn

def frame_step_summary(st) -> dict:
    """Frame system in the state: extremes by bars (N, V, M, T, σ), reactions of the nodes, max. displacement (beams.results)."""
    from ..beams.results import frame_summary
    fs = frame_summary(st.frame)
    sig = [(v.get("sigma_eq_max_MPa", 0.0), k) for k, v in fs["members"].items()]
    mom = [(max(v.get("My", {}).get("max_abs", 0.0), v.get("Mz", {}).get("max_abs", 0.0)), k)
           for k, v in fs["members"].items()]
    if sig:
        fs["max_sigma_eq_MPa"] = {"value": max(sig)[0], "member": max(sig)[1]}
    if mom:
        fs["max_abs_moment_Nmm"] = {"value": max(mom)[0], "member": max(mom)[1]}
    return fs


def step_summary(sess: Session, i: int, with_hotspots: bool = True) -> dict:
    res = sess.require_results()
    st = res.steps[i]
    m = res.mesh
    if not len(m.nodes):                                  # frame system only
        d = {"step": st.name, "index": i, "frame": frame_step_summary(st)}
        eq = st.equilibrium or {}
        if eq:
            d["equilibrium"] = {"applied_load_N": eq.get("applied_exact"), "reaction_total_N": eq.get("reaction_total"),
                                "force_residual_N": eq.get("force_residual"), "imbalance_relative": eq.get("imbalance"),
                                "ok": (eq.get("imbalance") or 0) <= 1e-3}
        d["max_displacement_mm"] = {"value": d["frame"].get("max_node_displacement_mm", 0.0)}
        return d
    c = mesh_cache(res)
    study = sess.study
    vm = st.mises
    um = st.umag
    pr = st.principal
    iu, iv = int(np.argmax(um)), int(np.argmax(vm))
    i1, i3 = int(np.argmax(pr[:, 0])), int(np.argmin(pr[:, 2]))
    d = {
        "step": st.name, "index": i,
        "max_displacement_mm": {"value": float(um[iu]), "at_mm": m.nodes[iu], "vector_mm": st.U[iu]},
        "displacement_range_mm": {c_: [float(st.U[:, k].min()), float(st.U[:, k].max())] for k, c_ in enumerate(("ux", "uy", "uz"))},
        "max_von_mises_MPa": {"value": float(vm[iv]), "at_mm": m.nodes[iv], "part": c.node_part(iv)},
        "max_principal_s1_MPa": {"value": float(pr[i1, 0]), "at_mm": m.nodes[i1]},
        "min_principal_s3_MPa": {"value": float(pr[i3, 2]), "at_mm": m.nodes[i3]},
        "reactions": {k: {"force_N": v, "magnitude_N": float(np.linalg.norm(v)),
                          "moment_Nmm": st.reaction_moments.get(k)} for k, v in st.reactions.items()},
    }
    if st.moment_point is not None:
        d["reaction_moment_point_mm"] = st.moment_point
    if st.remote:                                         # remote constraints: reaction ABOUT THE POINT of the constraint, displacement / rotation of the point
        d["remote_supports"] = {k: {"point_mm": r["point"], "coupling": r["coupling"],
                                    "fixed_dofs": [n for n, on in zip(("ux", "uy", "uz", "rx", "ry", "rz"), r["dofs"]) if on],
                                    "force_N": r["force"], "moment_about_point_Nmm": r["moment"],
                                    "point_displacement_mm": r["u"], "point_rotation_rad": r["rot"]}
                                for k, r in st.remote.items()}
    eq = st.equilibrium or {}
    if eq:
        d["equilibrium"] = {"applied_load_N": eq.get("applied_exact"), "reaction_total_N": eq.get("reaction_total"),
                            "force_residual_N": eq.get("force_residual"), "imbalance_relative": eq.get("imbalance"),
                            "ok": (eq.get("imbalance") or 0) <= 1e-3}
    T = st.fields.get("T")
    if T is not None:
        lc = next((x for x in study.load_cases if x.name == st.name), None)
        src = [thermal_load_text(l) for l in lc.loads if l.kind == "thermal"] if lc is not None else []
        d["temperature_C"] = {"min": float(T.min()), "max": float(T.max()),
                              "reference": study.thermal.reference_temperature, "source": "; ".join(src) or None}
    # safety against Re by parts
    parts = []
    re_min = math.inf
    _, pmax, cnt = c.part_extrema(vm)
    for pi, pn in enumerate(m.part_names):
        if not cnt[pi]:
            continue
        vmax = float(pmax[pi])
        try:
            mat = study.material_of(pn)
        except KeyError:
            mat = None
        row = {"part": pn, "max_mises_MPa": vmax, "material": mat.name if mat else None}
        if mat is not None and getattr(mat, "is_ortho", False):
            row["note"] = "orthotropic material – safety factor to Re (von Mises) is not valid, evaluate stress in the material axes"
        elif mat is not None and mat.Re:
            row["Re_MPa"] = mat.Re
            row["safety_Re"] = mat.Re / max(vmax, 1e-12)
            re_min = min(re_min, row["safety_Re"])
        parts.append(row)
    d["parts"] = parts
    if math.isfinite(re_min):
        d["min_safety_Re"] = re_min
    try:
        from ..assessment import assessment_summary
        ass = assessment_summary(study, m, st)
        if ass:
            d["assessment"] = ass
    except Exception as e:  # noqa: BLE001 – the summary must not fall because of the assessment, but the error is shown
        d["assessment_error"] = str(e)
    if with_hotspots:
        try:
            hs = hotspots(sess, i, 5)["hotspots"]
            d["hotspots_top"] = [{"xyz_mm": h["xyz_mm"], "mises_MPa": h["mises_MPa"], "classification": h["classification"],
                                  **({"stress_error_pct": h["stress_error_pct"]} if "stress_error_pct" in h else {})}
                                 for h in hs]
            errs = [h["stress_error_pct"] for h in hs if "stress_error_pct" in h]
            if errs:
                d["max_stress_error_hotspots_pct"] = max(errs)
                d["stress_error_note"] = ("ccx stress error estimate (STR, principal stress gradient in neighboring elements) at the maxima – informative; > 10–20 % = refine the mesh there")
            ns = [h for h in hs if h["classification"] != "singularity"]
            if hs and hs[0]["classification"] != "probably real concentration":
                d["note"] = (f"The von Mises maximum is probably a singularity/edge effect ({hs[0]['classification']}) – see fea_results_hotspots.")
                if ns:
                    d["max_mises_without_singularities_MPa"] = ns[0]["mises_MPa"]
        except Exception as e:  # noqa: BLE001 – the summary must not fall because of the heuristic
            d["hotspots_error"] = str(e)
    if st.printing:
        d["printing"] = st.printing
    if getattr(st, "frame", None):
        d["frame"] = frame_step_summary(st)
    return d


def thermal_summary(sess: Session, k: int) -> dict:
    """Summary of the thermal state: T min/max (place, part), max. heat flux, temperatures by parts, fluxes of the boundary
    conditions [W] (+ = heat into the body) and the heat balance; for a transient state the frames in time.
    """
    res = sess.require_results()
    tr = res.thermal[k]
    m = res.mesh
    c = mesh_cache(res)
    T, q = tr.T, tr.flux_mag
    imax, imin, iq = int(np.argmax(T)), int(np.argmin(T)), int(np.argmax(q))
    d = {"thermal_case": tr.name, "index": k, "transient": tr.transient,
         "T_max_C": {"value": float(T[imax]), "at_mm": m.nodes[imax], "part": c.node_part(imax)},
         "T_min_C": {"value": float(T[imin]), "at_mm": m.nodes[imin], "part": c.node_part(imin)},
         "max_heat_flux_W_m2": {"value": float(q[iq]), "at_mm": m.nodes[iq], "part": c.node_part(iq)}}
    pmin, pmax, cnt = c.part_extrema(T)
    d["parts"] = [{"part": pn, "T_min_C": float(pmin[pi]), "T_max_C": float(pmax[pi])}
                  for pi, pn in enumerate(m.part_names) if cnt[pi]]
    flows = [{"kind": h["kind"], "label": THERMAL_LABELS.get(h["kind"], h["kind"]), "name": h["name"],
              "Q_W": float(h["Q_W"])} for h in tr.heat]
    d["heat_flows_W"] = flows
    d["heat_sign"] = "Q_W > 0 = heat into the body, < 0 = out of the body"
    bal = dict(tr.balance or {})
    if "imbalance" in bal:
        from .. import config
        bal["ok"] = bal["imbalance"] <= config.EQUILIBRIUM_TOL
    d["balance"] = bal
    if tr.transient:
        d["times_s"] = list(tr.times)
        if tr.frames is not None:
            d["frames"] = [{"t_s": float(t), "T_min_C": float(f.min()), "T_max_C": float(f.max())}
                           for t, f in zip(tr.times, tr.frames)]
        d["note"] = "Temperature and flux are valid at the end of the case; temperature at another time is given by time_s (probe, image)."
    return d


def results_summary(sess: Session, step=None, thermal_case=None) -> dict:
    """Summary of the results: the mechanical states (step = index/name, None = all) and the thermal states (thermal_case,
    None = all). A study with heat only has steps = [].
    """
    res = sess.require_results()
    if not res.steps:
        idx = []
    else:
        idx = range(len(res.steps)) if step is None else [sess.step_index(step)]
    d = {"study": sess.study.name, "backend": res.info.get("backend"),
         "mesh": {"nodes": len(res.mesh.nodes), "elements": len(res.mesh.tets), "dof": 3 * len(res.mesh.nodes)},
         "units": "mm, N, MPa" + (", °C, W, W/m²" if res.thermal else ""),
         "steps": [step_summary(sess, i) for i in idx], "warnings": res.warnings}
    if res.thermal:
        tidx = range(len(res.thermal)) if thermal_case is None else [sess.thermal_index(thermal_case)]
        d["thermal"] = [thermal_summary(sess, k) for k in tidx]
    if getattr(res, "modes", None):
        d["modes"] = [modes_summary(sess, j, brief=True) for j in range(len(res.modes))]
    if getattr(res, "twod", None) is not None:
        t = res.twod.summary()
        d["twod"] = {"kind": t["kind_label"], "nodes": t["nodes"], "elements": t["elements"],
                     "steps": [{k: s[k] for k in ("name", "max_mises", "max_disp", "reactions", "imbalance")}
                               for s in t["steps"]], "detail": "fea_twod_results"}
        d["mesh"]["note"] = "2D mesh (tri6) – see twod"
    if getattr(res, "shell", None) is not None:
        t = res.shell.summary()
        d["shell"] = {"elements": t["elements"], "nodes": t["nodes"], "parts": t["parts"],
                      "steps": [{k: s[k] for k in ("name", "u_max", "mises_max", "membrane_max", "bending_max",
                                                   "imbalance")} for s in t["steps"]], "detail": "fea_shell_results"}
        d["mesh"]["note"] = "S6 shell mesh – see shell"
    return d


def modes_summary(sess: Session, j: int, brief: bool = False) -> dict:
    """Natural frequencies (frequencies, shares of the effective mass by the directions x, y, z and cumulatively) or buckling
    (factors λ, critical load = λ × the load of the state) of the set Results.modes[j]. brief = only numbers (summary).
    """
    res = sess.require_results()
    ms = res.modes[j]
    d = {"name": ms.name, "kind": ms.kind, "index": j}
    vals = [float(v) for v in ms.values]
    if ms.kind == "modal":
        d["frequencies_Hz"] = vals
        tot = np.asarray(ms.total_mass or [], float)
        em = np.asarray(ms.eff_mass or [], float)
        frac = None
        if em.size and tot.size >= 3:
            frac = em[:, :3] / np.where(tot[:3] > 0, tot[:3], 1.0)
            d["cumulative_mass_fraction_xyz"] = frac.sum(axis=0)
            d["total_mass_kg"] = float(tot[0] * 1000.0)
        fmax = max([abs(v) for v in vals], default=0.0)
        rigid = [i + 1 for i, v in enumerate(vals) if abs(v) <= max(1e-3 * fmax, 1e-6)]
        if rigid:
            d["rigid_body_modes"] = rigid
        if brief:
            return d
        rows = []
        cum = np.zeros(3)
        for i, f in enumerate(vals):
            r = {"mode": i + 1, "frequency_Hz": f, "period_s": (1.0 / f) if f > 0 else None}
            if frac is not None and i < len(frac):
                cum = cum + frac[i]
                r["eff_mass_fraction_xyz"] = frac[i]
                r["cumulative_xyz"] = cum.copy()
            if ms.participation and i < len(ms.participation):
                r["participation_xyz"] = list(ms.participation[i][:3])
            rows.append(r)
        d["modes"] = rows
        notes = ["Mode shapes are normalized to max |u| = 1 (displacement in a mode has no physical magnitude) – image fea_results_render_mode(mode=…)",
                 "Effective mass fraction = relative to the total effective mass in ccx (mass of free nodes; constrained nodes are not counted)"]
        if frac is not None and np.min(frac.sum(axis=0)) < 0.9:
            notes.append("Sum of effective mass fractions in some direction < 90 % – for dynamic response (seismic, spectrum) increase the number of modes (fea_modal_set modes)")
        if rigid:
            notes.append(f"Modes {', '.join(map(str, rigid))} have ≈ 0 Hz – rigid-body motions (free body / insufficient supports)")
        d["notes"] = notes
        return d
    d["case"] = ms.case
    d["buckling_factors"] = vals
    st = next((s for s in res.steps if s.name == ms.case), None)
    applied = None
    if st is not None and st.equilibrium:
        applied = st.equilibrium.get("applied_exact")
    if applied is not None:
        F = float(np.linalg.norm(applied))
        d["applied_load_N"] = applied
        if not brief:
            d["critical_load_N"] = [v * F for v in vals]
    if brief:
        return d
    lc = next((x for x in sess.study.load_cases if x.name == ms.case), None) if sess.study else None
    if lc is not None:
        from ..loads_text import load_text
        d["case_loads"] = [load_text(l, sess.study) for l in lc.loads]
    notes = ["λ = load case multiplier at loss of stability: critical load = λ × case load (all loads in the case are scaled together)",
             "LINEAR BUCKLING without imperfections, plasticity and nonlinearities – actual capacity is usually considerably lower (by tens of % for slender members and shells); for design use the code procedure (EN 1993-1-1, 6.3 – fea_ec3_member_check) or a nonlinear analysis with imperfections"]
    if any(v < 0 for v in vals):
        notes.append("Negative λ = loss of stability under REVERSED load (e.g. tension ↔ compression)")
    if vals and 0 < min(abs(v) for v in vals) < 1:
        notes.append("|λ| < 1 – loss of stability already at the applied load!")
    d["notes"] = notes
    return d


def summary_text(sess: Session) -> str:
    """Text summary of the last results (source fea://results/summary)."""
    if sess.study is None:
        return "No study is open."
    if sess.results is None:
        return f"Study '{sess.study.name}': results are not available (fea_solve_run)."
    s = results_summary(sess)
    lines = [f"Study '{s['study']}', solver {s['backend']}, mesh {s['mesh']['nodes']} nodes / {s['mesh']['elements']} elements ({s['mesh']['dof']} DOF). Units {s['units']}."]
    for th in s.get("thermal", []):
        kind = "transient" if th["transient"] else "steady-state"
        lines.append(f"\nThermal case '{th['thermal_case']}' ({kind}):")
        lines.append(f"  T max {th['T_max_C']['value']:.4g} °C at {np.round(th['T_max_C']['at_mm'], 3).tolist()} (part {th['T_max_C']['part']}), T min {th['T_min_C']['value']:.4g} °C")
        lines.append(f"  max. heat flux {th['max_heat_flux_W_m2']['value']:.4g} W/m²")
        for f in th["heat_flows_W"]:
            lines.append(f"  heat '{f['name']}' ({f['label']}): {f['Q_W']:+.4g} W")
        bal = th["balance"]
        if "imbalance" in bal:
            lines.append(f"  balance: input {bal.get('in_W', 0):.4g} W, output {bal.get('out_W', 0):.4g} W, imbalance {bal['imbalance']:.2e}")
        elif bal.get("note"):
            lines.append("  " + bal["note"])
    for ms in s.get("modes", []):
        if ms["kind"] == "modal":
            fr = ", ".join(f"{v:.4g}" for v in ms["frequencies_Hz"][:10])
            lines.append(f"\nNatural frequencies [Hz]: {fr}" + (" …" if len(ms["frequencies_Hz"]) > 10 else ""))
        else:
            lines.append(f"\nBuckling (linear, state '{ms['case']}'): λ = "
                         + ", ".join(f"{v:.4g}" for v in ms["buckling_factors"]))
    for st in s["steps"]:
        lines.append(f"\nState '{st['step']}':")
        fr = st.get("frame")
        if fr:
            lines.append(f"  members: max. node displacement {fr.get('max_node_displacement_mm', 0):.4g} mm"
                         + (f", max. σ_eq {fr['max_sigma_eq_MPa']['value']:.4g} MPa (member {fr['max_sigma_eq_MPa']['member']})"
                            if fr.get("max_sigma_eq_MPa") else ""))
            for k, r in fr.get("reactions", {}).items():
                lines.append(f"  node reaction '{k}': F = {np.round(np.asarray(r[:3], float), 4).tolist()} N, "
                             f"M = {np.round(np.asarray(r[3:], float), 1).tolist()} N·mm")
        if "max_von_mises_MPa" not in st:
            if "equilibrium" in st:
                lines.append(f"  equilibrium: imbalance {st['equilibrium']['imbalance_relative']:.2e}")
            continue
        lines.append(f"  max. displacement {st['max_displacement_mm']['value']:.4g} mm at {np.round(st['max_displacement_mm']['at_mm'], 3).tolist()}")
        lines.append(f"  max. von Mises {st['max_von_mises_MPa']['value']:.4g} MPa at {np.round(st['max_von_mises_MPa']['at_mm'], 3).tolist()} (part {st['max_von_mises_MPa']['part']})")
        lines.append(f"  σ1 max {st['max_principal_s1_MPa']['value']:.4g} MPa, σ3 min {st['min_principal_s3_MPa']['value']:.4g} MPa")
        if "temperature_C" in st:
            t = st["temperature_C"]
            lines.append(f"  temperature {t['min']:.4g} … {t['max']:.4g} °C (reference {t['reference']:g} °C)"
                         + (f": {t['source']}" if t.get("source") else ""))
        for k, r in st["reactions"].items():
            f = np.round(np.asarray(r["force_N"], float), 4).tolist()
            lines.append(f"  reaction '{k}': {f} N (|R| = {r['magnitude_N']:.4g} N)")
        if "equilibrium" in st:
            lines.append(f"  equilibrium: imbalance {st['equilibrium']['imbalance_relative']:.2e}")
        if "min_safety_Re" in st:
            lines.append(f"  min. safety factor to Re {st['min_safety_Re']:.3g}")
        if st.get("note"):
            lines.append("  " + st["note"])
    for w in s["warnings"]:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
