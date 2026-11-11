"""Result of the topology optimisation in the results of a study (fea_core/topo/results.py): attaching to the results, saving
and loading of the project, the densities of the elements from the saved summary and the surface of the resulting shape. Fast – a synthetic tet10 mesh,
a fake optimiser, without Gmsh and ccx.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fea_core.mesh import Mesh
from fea_core.results import Results, StepResult
from fea_core.topo.export import densities_to_nodes
from fea_core.topo.results import FIELD, attach, element_densities, shape_triangles

NX, NY, NZ = 4, 2, 2                                   # krychle 1×1×1 mm
SOLID_NX = 3                                           # cubes with ix >= 3 are removed


def box_mesh(nx=NX, ny=NY, nz=NZ) -> Mesh:
    """A block of nx×ny×nz cubes, each divided into 6 tetrahedra (Kuhn), tet10 in the order of ccx (corners, midpoints of the edges
    01 12 20 03 13 23).
    """
    nodes: list[tuple[float, float, float]] = []
    index: dict[tuple[float, float, float], int] = {}

    def node(p) -> int:
        key = tuple(round(float(c), 9) for c in p)
        if key not in index:
            index[key] = len(nodes)
            nodes.append(key)
        return index[key]

    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    tets = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for perm in perms:
                    p = np.array([ix, iy, iz], float)
                    corners = [p.copy()]
                    for a in perm:
                        p = p.copy()
                        p[a] += 1.0
                        corners.append(p)
                    c = np.array(corners)
                    if np.linalg.det(c[1:] - c[0]) < 0:             # positive orientation
                        c[[1, 2]] = c[[2, 1]]
                    mids = [(c[0] + c[1]) / 2, (c[1] + c[2]) / 2, (c[2] + c[0]) / 2,
                            (c[0] + c[3]) / 2, (c[1] + c[3]) / 2, (c[2] + c[3]) / 2]
                    tets.append([node(q) for q in list(c) + mids])
    return Mesh(nodes=np.array(nodes, float), tets=np.array(tets, np.int32),
                tet_part=np.zeros(len(tets), np.int32), part_names=["Konzola"])


def make_results(mesh: Mesh, n_steps: int = 2) -> Results:
    n = len(mesh.nodes)
    steps = [StepResult(name=f"Stav {k + 1}", U=np.zeros((n, 3)), S=np.zeros((n, 6)),
                        reactions={"vazba": np.array([0.0, 0.0, 1.0])}) for k in range(n_steps)]
    return Results(mesh=mesh, steps=steps)


def fake_optimizer(mesh: Mesh) -> SimpleNamespace:
    """An object with the attributes that `attach` reads from TopologyOptimizer; the elements in the last column of cubes are removed."""
    centers = mesh.nodes[mesh.tets[:, :4]].mean(axis=1)
    rho = np.where(centers[:, 0] > SOLID_NX, 0.0, 1.0)
    hist = [SimpleNamespace(converged=False, volume_fraction=0.9, total_compliance=1.0),
            SimpleNamespace(converged=True, volume_fraction=0.75, total_compliance=1.4)]
    spec = SimpleNamespace(target_volume_fraction=0.75, evolution_rate=0.04, filter_radius=1.5, penalty=3.0,
                           non_design_selections=["Vazba", "Oko"])
    return SimpleNamespace(rho=rho, spec=spec, history=hist, current_volume_fraction=float(rho.mean()),
                           initial_compliance=1.0, final_compliance=1.4)


def test_attach_adds_node_density_to_every_step_and_info():
    mesh = box_mesh()
    res = make_results(mesh)
    opt = fake_optimizer(mesh)
    info = attach(res, opt)
    expected = densities_to_nodes(mesh, opt.rho)
    for st in res.steps:
        v = st.fields[FIELD]
        assert v.shape == (len(mesh.nodes),)
        assert np.allclose(v, expected)
        assert v.min() >= 0.0 and v.max() <= 1.0 + 1e-12
    assert res.info["topology"] is info
    assert info["iterations"] == 2 and info["converged"] is True
    assert info["final_volume_fraction"] == pytest.approx(opt.current_volume_fraction)
    assert info["initial_compliance_Nmm"] == 1.0 and info["final_compliance_Nmm"] == 1.4
    assert info["non_design_selections"] == ["Vazba", "Oko"]
    assert info["volume_history"] == [0.9, 0.75]
    assert len(info["void_elements"]) == int(np.count_nonzero(opt.rho == 0.0)) > 0


def test_attach_rejects_other_mesh():
    mesh = box_mesh()
    res = make_results(mesh)
    opt = fake_optimizer(mesh)
    opt.rho = opt.rho[:-1]
    with pytest.raises(ValueError):
        attach(res, opt)
    assert "topology" not in res.info


def test_element_densities_and_shape_none_without_topology():
    res = make_results(box_mesh())
    assert element_densities(res) is None
    assert shape_triangles(res) is None


def test_element_densities_are_zero_one():
    mesh = box_mesh()
    res = make_results(mesh)
    opt = fake_optimizer(mesh)
    attach(res, opt)
    rho = element_densities(res)
    assert rho.shape == (len(mesh.tets),)
    assert set(np.unique(rho)) == {0.0, 1.0}
    assert np.array_equal(rho, opt.rho)


def test_save_load_roundtrip_keeps_field_and_info(tmp_path):
    mesh = box_mesh()
    res = make_results(mesh)
    opt = fake_optimizer(mesh)
    attach(res, opt)
    path = tmp_path / "vysledky.npz"
    res.save(path)
    back = Results.load(path)
    assert back.info["topology"]["iterations"] == 2
    assert back.info["topology"]["target_volume_fraction"] == pytest.approx(0.75)
    assert back.info["topology"]["compliance_history_Nmm"] == [1.0, 1.4]
    for a, b in zip(res.steps, back.steps):
        assert np.allclose(a.fields[FIELD], b.fields[FIELD])
    assert np.array_equal(element_densities(back), opt.rho)


def test_shape_triangles_closed_nonempty_surface():
    mesh = box_mesh()
    res = make_results(mesh)
    attach(res, fake_optimizer(mesh))
    P, V = shape_triangles(res)
    assert P.ndim == 3 and P.shape[1:] == (3, 3) and len(P) > 0
    assert V.shape == (len(P), 3)
    assert np.isfinite(P).all() and np.isfinite(V).all()
    assert V.min() >= 0.5 - 1e-6 and V.max() <= 1.0 + 1e-9        # hustota ve vrcholech: od izoplochy 0,5 do 1
    assert P[..., 0].min() == pytest.approx(0.0, abs=1e-9)         # the surface starts at the face with the constraint
    assert P[..., 0].max() <= NX + 1e-9                            # and does not overhang the block
    # the surface of the resulting shape does not intrude into the removed end of the block
    assert np.mean(P[..., 0]) < SOLID_NX


def test_shape_triangles_all_void_gives_empty():
    mesh = box_mesh(1, 1, 1)
    res = make_results(mesh)
    opt = fake_optimizer(mesh)
    opt.rho = np.zeros(len(mesh.tets))
    opt.current_volume_fraction = 0.0
    attach(res, opt)
    P, V = shape_triangles(res)
    assert P.shape == (0, 3, 3) and V.shape == (0, 3)
