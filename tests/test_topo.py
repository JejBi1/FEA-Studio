"""Unit tests for the topology optimisation module (fea_core.topo)."""
import numpy as np
import pytest

from fea_core.mesh import Mesh
from fea_core.results import StepResult
from fea_core.study import Study, Material
from fea_core.topo import (
    TopologyOptSpec,
    SensitivityFilter,
    filter_sensitivities,
    TopologyOptimizer,
    compute_tet_volumes,
    compute_tet_centroids,
    compute_strain_energy_density,
    compute_element_strain_energies,
    find_threshold_bisection,
    identify_non_design_elements,
)
from fea_core.topo.export import export_topo_stl


def _make_simple_mesh():
    """Creates a simple mesh of two tetrahedra sharing a face."""
    nodes = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [1.0, 1.0, 1.0],  # 4
    ], dtype=float)
    tets = np.array([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ], dtype=int)
    tet_part = np.array([0, 0], dtype=int)
    surfaces = {
        "fixed_face": np.array([[0, 1, 2]]),
        "load_face": np.array([[1, 2, 4]]),
    }
    return Mesh(nodes=nodes, tets=tets, tet_part=tet_part, part_names=["part1"], surfaces=surfaces)


def _make_grid_mesh(nx=5, ny=5, nz=2, size=10.0):
    """Creates a block mesh of tetrahedra by dividing cubes into tets."""
    x = np.linspace(0, size, nx)
    y = np.linspace(0, size, ny)
    z = np.linspace(0, size, nz)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)

    tets_list = []
    def node_id(i, j, k):
        return i * (ny * nz) + j * nz + k

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # 8 corner nodes of the cell
                n000 = node_id(i, j, k)
                n100 = node_id(i + 1, j, k)
                n010 = node_id(i, j + 1, k)
                n110 = node_id(i + 1, j + 1, k)
                n001 = node_id(i, j, k + 1)
                n101 = node_id(i + 1, j, k + 1)
                n011 = node_id(i, j + 1, k + 1)
                n111 = node_id(i + 1, j + 1, k + 1)

                # Division into 5 tetrahedra
                tets_list.append([n000, n100, n010, n001])
                tets_list.append([n100, n110, n010, n111])
                tets_list.append([n100, n001, n101, n111])
                tets_list.append([n010, n001, n111, n011])
                tets_list.append([n100, n010, n001, n111])

    tets = np.array(tets_list, dtype=int)
    tet_part = np.zeros(len(tets), dtype=int)
    return Mesh(nodes=nodes, tets=tets, tet_part=tet_part, part_names=["part_block"])


# =========================================================================
# 1. Testy TopologyOptSpec
# =========================================================================

def test_spec_defaults():
    spec = TopologyOptSpec()
    assert spec.target_volume_fraction == 0.5
    assert spec.evolution_rate == 0.02
    assert spec.filter_radius == 6.0
    assert spec.penalty == 3.0
    assert spec.max_iterations == 50
    assert spec.non_design_selections == []
    assert spec.min_density == 1e-3
    assert spec.target_step == 0
    assert spec.method == "beso"
    assert spec.tol == 1e-3


def test_spec_validation():
    with pytest.raises(ValueError, match="target_volume_fraction"):
        TopologyOptSpec(target_volume_fraction=0.0)

    with pytest.raises(ValueError, match="target_volume_fraction"):
        TopologyOptSpec(target_volume_fraction=1.2)

    with pytest.raises(ValueError, match="evolution_rate"):
        TopologyOptSpec(evolution_rate=-0.05)

    with pytest.raises(ValueError, match="filter_radius"):
        TopologyOptSpec(filter_radius=0.0)

    with pytest.raises(ValueError, match="penalty"):
        TopologyOptSpec(penalty=0.5)

    with pytest.raises(ValueError, match="max_iterations"):
        TopologyOptSpec(max_iterations=0)

    with pytest.raises(ValueError, match="min_density"):
        TopologyOptSpec(min_density=-0.01)


def test_spec_dict_roundtrip():
    spec1 = TopologyOptSpec(
        target_volume_fraction=0.3,
        evolution_rate=0.04,
        filter_radius=8.5,
        penalty=4.0,
        max_iterations=80,
        non_design_selections=["support", "load"],
        min_density=1e-4,
        target_step=1,
    )
    d = spec1.to_dict()
    assert d["target_volume_fraction"] == 0.3
    assert d["non_design_selections"] == ["support", "load"]

    spec2 = TopologyOptSpec.from_dict(d)
    assert spec1 == spec2


# =========================================================================
# 2. Testy SensitivityFilter
# =========================================================================

def test_filter_constant_field():
    """A constant field of sensitivities must stay constant after the spatial filtering."""
    centroids = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ])
    radius = 5.0
    filt = SensitivityFilter(centroids, filter_radius=radius)

    alpha_const = np.full(4, 42.0)
    filtered = filt.filter(alpha_const, use_history=False)
    assert np.allclose(filtered, 42.0)


def test_filter_distance_weights():
    """The nearer point must have a higher weight than the more distant point."""
    centroids = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1 (distance 1.0)
        [3.0, 0.0, 0.0],  # 2 (distance 3.0)
        [10.0, 0.0, 0.0], # 3 (distance 10.0 > r=4.0)
    ])
    radius = 4.0
    filt = SensitivityFilter(centroids, filter_radius=radius)

    # Sensitivity only at points 1 and 2, point 0 has 0
    alpha = np.array([0.0, 10.0, 10.0, 100.0])
    filtered = filt.filter(alpha, use_history=False)

    # Point 3 is outside the filter radius for point 0, it must not influence point 0
    # For point 0: d0=0 (w=4), d1=1 (w=3), d2=3 (w=1), d3=10 (w=0)
    # Sum of the weights: 4 + 3 + 1 = 8
    # alpha_tilde_0 = (4*0 + 3*10 + 1*10) / 8 = 40 / 8 = 5.0
    assert np.isclose(filtered[0], 5.0)


def test_filter_historical_averaging():
    """Historical averaging: alpha_curr = 0.5 * (alpha_tilde + alpha_prev)."""
    centroids = np.array([[0.0, 0.0, 0.0]])
    filt = SensitivityFilter(centroids, filter_radius=2.0)

    # Iteration 1: alpha = 100.0 -> no history, the result 100.0
    a1 = filt.filter(np.array([100.0]), use_history=True)
    assert np.isclose(a1[0], 100.0)

    # Iterace 2: alpha = 50.0 -> alpha_curr = 0.5 * (50.0 + 100.0) = 75.0
    a2 = filt.filter(np.array([50.0]), use_history=True)
    assert np.isclose(a2[0], 75.0)

    # Iterace 3: alpha = 50.0 -> alpha_curr = 0.5 * (50.0 + 75.0) = 62.5
    a3 = filt.filter(np.array([50.0]), use_history=True)
    assert np.isclose(a3[0], 62.5)

    # Reset historie
    filt.reset()
    assert filt.alpha_prev is None
    a4 = filt.filter(np.array([50.0]), use_history=True)
    assert np.isclose(a4[0], 50.0)


def test_filter_sensitivities_helper():
    centroids = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    alpha = np.array([10.0, 20.0])
    res, prev = filter_sensitivities(centroids, alpha, filter_radius=2.0)
    assert len(res) == 2
    assert prev is not None
    assert np.allclose(res, prev)


# =========================================================================
# 3. Testy BESO bisekce (find_threshold_bisection)
# =========================================================================

def test_bisection_uniform_volumes():
    """Bisection for uniform element volumes."""
    sens = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    vols = np.ones(5)
    target_vol = 3.0  # chceme 3 solid prvky ze 5

    th, rho = find_threshold_bisection(sens, vols, target_volume=target_vol, min_density=0.0)
    assert np.array_equal(rho, [0.0, 0.0, 1.0, 1.0, 1.0])
    assert np.isclose(np.sum(vols * rho), 3.0)
    assert 20.0 <= th <= 30.0


def test_bisection_with_min_density():
    sens = np.array([5.0, 15.0, 25.0, 35.0])
    vols = np.ones(4)
    target_vol = 2.0
    min_d = 1e-3

    th, rho = find_threshold_bisection(sens, vols, target_volume=target_vol, min_density=min_d)
    assert np.allclose(rho, [min_d, min_d, 1.0, 1.0])
    assert abs(np.sum(vols * rho) - target_vol) < 0.01


def test_bisection_non_design_elements():
    """Non-design elements with a low sensitivity must not be removed."""
    sens = np.array([1.0, 10.0, 20.0, 30.0, 40.0])
    vols = np.ones(5)
    target_vol = 3.0

    nd_mask = np.array([True, False, False, False, False])
    th, rho = find_threshold_bisection(sens, vols, target_volume=target_vol, non_design_mask=nd_mask, min_density=0.0)

    assert rho[0] == 1.0  # non-design trvale solid
    assert np.array_equal(rho, [1.0, 0.0, 0.0, 1.0, 1.0])
    assert np.isclose(np.sum(vols * rho), 3.0)


def test_bisection_extreme_targets():
    sens = np.array([10.0, 20.0, 30.0])
    vols = np.ones(3)

    _, rho_all = find_threshold_bisection(sens, vols, target_volume=3.0)
    assert np.allclose(rho_all, 1.0)

    _, rho_none = find_threshold_bisection(sens, vols, target_volume=0.0, min_density=1e-3)
    assert np.allclose(rho_none, 1e-3)


# =========================================================================
# 4. Tests of the element geometry and of the calculation of the strain energy
# =========================================================================

def test_tet_volumes_and_centroids():
    mesh = _make_simple_mesh()
    vols = compute_tet_volumes(mesh.nodes, mesh.tets)
    cents = compute_tet_centroids(mesh.nodes, mesh.tets)

    assert len(vols) == 2
    assert len(cents) == 2
    assert np.isclose(vols[0], 1.0 / 6.0)
    assert np.allclose(cents[0], [0.25, 0.25, 0.25])


def test_strain_energy_density_uniaxial():
    """Check of the calculation of the strain energy density u0 = sigma^2 / (2E) for uniaxial tension."""
    E = 210000.0  # MPa
    nu = 0.3
    sigma_x = 210.0  # MPa

    S = np.zeros((1, 6))
    S[0, 0] = sigma_x

    u0 = compute_strain_energy_density(S, E=E, nu=nu)
    expected_u0 = (sigma_x ** 2) / (2.0 * E)
    assert np.isclose(u0[0], expected_u0)


def test_strain_energy_density_pure_shear():
    """Check of u0 for pure shear u0 = tau^2 / (2G) = tau^2 * (1+nu) / E."""
    E = 200000.0
    nu = 0.3
    tau_xy = 100.0

    S = np.zeros((1, 6))
    S[0, 3] = tau_xy

    u0 = compute_strain_energy_density(S, E=E, nu=nu)
    expected_u0 = (tau_xy ** 2) * (1.0 + nu) / E
    assert np.isclose(u0[0], expected_u0)


def test_element_strain_energies_from_step_result():
    mesh = _make_simple_mesh()
    S = np.zeros((len(mesh.nodes), 6))
    S[:, 0] = 100.0
    U = np.zeros((len(mesh.nodes), 3))

    step_res = StepResult(name="Step 1", U=U, S=S)
    elem_u = compute_element_strain_energies(mesh, step_res, E=200000.0, nu=0.3)

    assert len(elem_u) == 2
    vols = compute_tet_volumes(mesh.nodes, mesh.tets)
    u0 = (100.0 ** 2) / (2.0 * 200000.0)
    assert np.allclose(elem_u, u0 * vols)


# =========================================================================
# 5. Tests of the identification of the non-design elements
# =========================================================================

def test_identify_non_design_elements():
    mesh = _make_simple_mesh()
    nd_mask = identify_non_design_elements(mesh, ["fixed_face"])
    assert nd_mask[0] == True
    assert nd_mask[1] == True

    empty_mask = identify_non_design_elements(mesh, ["non_existent"])
    assert np.all(~empty_mask)


# =========================================================================
# 6. Tests of TopologyOptimizer and of the iteration loop
# =========================================================================

def test_optimizer_initialization():
    mesh = _make_grid_mesh(nx=3, ny=3, nz=2, size=10.0)
    spec = TopologyOptSpec(target_volume_fraction=0.4, evolution_rate=0.05, filter_radius=5.0)
    opt = TopologyOptimizer(mesh, spec=spec, base_E=210000.0, nu=0.3)

    assert opt.iteration == 0
    assert opt.num_elements == len(mesh.tets)
    assert opt.current_volume_fraction == 1.0
    assert np.allclose(opt.rho, 1.0)
    assert np.allclose(opt.densities, 1.0)
    assert opt.solid_elements_count == opt.num_elements
    assert opt.void_elements_count == 0


def test_optimizer_initialization_with_study():
    mesh = _make_simple_mesh()
    mat = Material(name="Steel", E=210000, nu=0.3)
    study = Study(name="TestStudy", geometry="dummy.step", default_material="Steel", materials={"Steel": mat})
    spec = TopologyOptSpec(target_volume_fraction=0.4)

    # Call as (study, mesh, spec)
    opt = TopologyOptimizer(study, mesh, spec)
    assert opt.study is study
    assert opt.mesh is mesh
    assert opt.base_E == 210000
    assert opt.nu == 0.3


def test_optimizer_densities_property():
    mesh = _make_simple_mesh()
    opt = TopologyOptimizer(mesh)
    assert np.allclose(opt.densities, 1.0)

    opt.densities = np.array([0.5, 0.8])
    assert np.allclose(opt.rho, [0.5, 0.8])


def test_optimizer_young_moduli():
    mesh = _make_simple_mesh()
    spec = TopologyOptSpec(penalty=3.0, min_density=1e-2)
    opt = TopologyOptimizer(mesh, spec=spec, base_E=1000.0)

    E = opt.get_element_young_moduli()
    assert np.allclose(E, 1000.0)

    opt.rho[0] = 0.0
    E_void = opt.get_element_young_moduli()
    assert np.isclose(E_void[0], 1000.0 * (1e-2 ** 3))
    assert np.isclose(E_void[1], 1000.0)


def test_optimizer_volume_progression():
    mesh = _make_simple_mesh()
    spec = TopologyOptSpec(target_volume_fraction=0.5, evolution_rate=0.1)
    opt = TopologyOptimizer(mesh, spec=spec)

    v1 = opt.compute_next_target_volume_fraction(1.0)
    assert np.isclose(v1, 0.9)

    v2 = opt.compute_next_target_volume_fraction(0.52)
    assert np.isclose(v2, 0.5)


def test_optimizer_step_iterations():
    """Multi-step simulation of the optimisation with artificial sensitivities."""
    mesh = _make_grid_mesh(nx=4, ny=4, nz=2, size=15.0)
    spec = TopologyOptSpec(
        target_volume_fraction=0.6,
        evolution_rate=0.1,  # 10 % za krok
        filter_radius=4.0,
        max_iterations=10,
    )
    opt = TopologyOptimizer(mesh, spec=spec)

    cents = opt.centroids
    synthetic_sens = 10.0 + cents[:, 0] * 5.0 + cents[:, 1] * 2.0

    for i in range(5):
        it = opt.step(sensitivities=synthetic_sens)
        assert it.iteration == i + 1
        assert len(opt.history) == i + 1
        assert it.volume_fraction <= 1.0
        # Check of the access through dictionary indexing it["volume_fraction"]
        assert it["volume_fraction"] == it.volume_fraction
        assert it["total_compliance"] == it.total_compliance

    final_rec = opt.history[-1]
    assert final_rec.volume_fraction < 0.8
    assert final_rec.elements_removed > 0
    assert opt.solid_elements_count > 0
    assert opt.void_elements_count > 0


def test_optimizer_run_without_study_is_error():
    """Without a study / load state there is nothing to compute the sensitivities from – no invented values, a clear error."""
    mesh = _make_grid_mesh(nx=3, ny=3, nz=2, size=10.0)
    spec = TopologyOptSpec(target_volume_fraction=0.7, evolution_rate=0.15, max_iterations=3)
    opt = TopologyOptimizer(mesh, spec=spec)
    with pytest.raises(ValueError, match="study"):
        opt.run(callback=lambda *a: None)
    assert opt.iteration == 0


def test_optimizer_create_solid_mesh():
    """Check of the creation of a sub-mesh containing only solid elements."""
    mesh = _make_simple_mesh()
    opt = TopologyOptimizer(mesh)

    opt.rho = np.array([1.0, 0.0])

    solid_mesh = opt.create_solid_mesh()
    assert len(solid_mesh.tets) == 1
    assert len(solid_mesh.nodes) == 4
    assert solid_mesh.tets.max() < len(solid_mesh.nodes)
    assert solid_mesh.part_names == mesh.part_names


def test_export_topo_stl(tmp_path):
    """Check of the export to STL through export_topo_stl."""
    mesh = _make_grid_mesh(nx=3, ny=3, nz=3, size=10.0)
    opt = TopologyOptimizer(mesh, TopologyOptSpec(target_volume_fraction=0.5))

    # We set half of the elements as solid
    dens = np.ones(opt.num_elements)
    dens[opt.centroids[:, 0] > 5.0] = 0.0
    opt.rho = dens

    out_file = tmp_path / "topo_test.stl"
    res_path = export_topo_stl(mesh, opt.densities, out_file)
    assert res_path.exists()
    assert res_path.stat().st_size > 84  # header + at least some triangles


def test_simp_mode():
    """Check of the SIMP method with a smooth optimisation of the densities."""
    mesh = _make_grid_mesh(nx=3, ny=3, nz=2, size=10.0)
    spec = TopologyOptSpec(
        target_volume_fraction=0.7,
        evolution_rate=0.1,
        method="simp",
    )
    opt = TopologyOptimizer(mesh, spec=spec)

    sens = 20.0 + opt.centroids[:, 0] * 3.0
    it = opt.step(sensitivities=sens)
    assert it.iteration == 1
    assert it.volume_fraction < 1.0
    assert np.any((opt.rho > spec.min_density) & (opt.rho < 1.0))
