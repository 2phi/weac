"""
This module contains tests for the Analyzer class.
"""

# Third party imports
import numpy as np
import pytest

from weac.analysis import Analyzer
from weac.analysis.analyzer import local_segment_grid
from weac.components import (
    Config,
    Layer,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac.components.model_input import ModelInput
from weac.core.system_model import SystemModel


@pytest.fixture
def sm_ski():
    """Basic skier system model."""
    model_input_ski = ModelInput(
        scenario_config=ScenarioConfig(phi=15.0, system_type="skier"),
        layers=[Layer()],
        weak_layer=WeakLayer(),
        segments=[Segment(), Segment()],
    )
    return SystemModel(model_input=model_input_ski, config=Config())


@pytest.fixture
def an_ski(sm_ski):
    """Analyzer for the basic skier system."""
    return Analyzer(system_model=sm_ski, printing_enabled=False)


@pytest.fixture
def sm_pst():
    """PST system model for potential-energy related methods."""
    model_input_pst = ModelInput(
        scenario_config=ScenarioConfig(phi=10.0, system_type="pst-"),
        layers=[Layer()],
        weak_layer=WeakLayer(),
        segments=[Segment(), Segment()],
    )
    return SystemModel(model_input=model_input_pst, config=Config())


@pytest.fixture
def an_pst(sm_pst):
    """Analyzer for the PST system."""
    return Analyzer(system_model=sm_pst, printing_enabled=False)



class TestLocalSegmentGrid:
    """Unit tests for piecewise boundary-refined local grids."""

    def test_uniform_matches_linspace(self):
        """Omitting boundary knobs preserves historical linspace behavior."""
        for endpoint in (True, False):
            got = local_segment_grid(
                1000.0, 11, include_right_endpoint=endpoint
            )
            expected = np.linspace(0.0, 1000.0, num=11, endpoint=endpoint)
            np.testing.assert_allclose(got, expected)

    def test_boundary_windows_have_fine_spacing(self):
        """Fine windows respect boundary_dx near both ends."""
        length = 1000.0
        window = 15.0
        dx = 0.5
        xi = local_segment_grid(
            length,
            n_budget=80,
            include_right_endpoint=True,
            boundary_window=window,
            boundary_dx=dx,
        )
        assert xi[0] == 0.0
        assert xi[-1] == length
        left = xi[xi <= window + 1e-9]
        right = xi[xi >= length - window - 1e-9]
        assert left.size > 2
        assert right.size > 2
        assert np.max(np.diff(left)) <= dx + 1e-9
        assert np.max(np.diff(right)) <= dx + 1e-9
        # Interior should be coarser than the fine window.
        interior = xi[(xi > window) & (xi < length - window)]
        if interior.size > 1:
            assert np.median(np.diff(interior)) > dx

    def test_non_last_segment_omits_right_endpoint(self):
        """Joints are owned by the next segment's local x=0."""
        xi = local_segment_grid(
            500.0,
            n_budget=40,
            include_right_endpoint=False,
            boundary_window=15.0,
            boundary_dx=0.5,
        )
        assert xi[0] == 0.0
        assert xi[-1] < 500.0
        # Still refined approaching the right joint.
        near_right = xi[xi >= 500.0 - 15.0]
        assert near_right.size > 5
        assert np.max(np.diff(near_right)) <= 0.5 + 1e-9

    def test_boundary_args_must_be_paired(self):
        """Setting only one boundary kwarg is an error."""
        with pytest.raises(ValueError):
            local_segment_grid(
                100.0, 10, include_right_endpoint=True, boundary_window=15.0
            )
        with pytest.raises(ValueError):
            local_segment_grid(
                100.0, 10, include_right_endpoint=True, boundary_dx=0.5
            )


class TestAnalyzer:
    """Test suite for the Analyzer."""

    def test_rasterize_solution_runs_and_shapes(self, an_ski):
        """Test rasterize_solution runs and shapes."""
        for mode in ("cracked", "uncracked"):
            xs, Z, xs_supported = an_ski.rasterize_solution(mode=mode, num=200)
            assert Z.shape[0] == 6
            assert xs.shape[0] == Z.shape[1]
            assert xs_supported.shape[0] == xs.shape[0]
            assert np.all(np.diff(xs[~np.isnan(xs)]) >= 0)

    def test_rasterize_solution_boundary_refinement(self, an_pst, sm_pst):
        """Boundary mode densifies domain ends and the segment joint."""
        xs, Z, _ = an_pst.rasterize_solution(
            mode="cracked",
            num=200,
            boundary_window=15.0,
            boundary_dx=0.5,
        )
        assert xs.shape[0] == Z.shape[1]
        assert np.all(np.diff(xs) >= -1e-12)
        # Domain ends present.
        assert float(xs[0]) == pytest.approx(0.0, abs=0.5 * 10 ** (-9))
        assert float(xs[-1]) == pytest.approx(
            float(np.sum(sm_pst.scenario.li)), abs=0.5 * 10 ** (-6)
        )
        # Joint between the two PST segments is present.
        joint = float(abs(sm_pst.scenario.li[0]))
        assert np.any(np.isclose(xs, joint, atol=1e-9))
        # Local spacing near the joint is fine.
        near = xs[np.abs(xs - joint) <= 15.0]
        assert near.size > 10
        assert float(np.max(np.diff(near))) <= 0.5 + 1e-6

    def test_get_zmesh_contains_expected_keys(self, an_ski):
        """Test get_zmesh contains expected keys."""
        zmesh = an_ski.get_zmesh(dz=5)
        for key in ("z", "E", "nu", "rho", "tensile_strength"):
            assert key in zmesh
        # Non-empty mesh
        assert len(zmesh["z"]) > 1
        z = np.asarray(zmesh["z"])
        assert np.all(np.diff(z) > 0)

    def test_stress_fields_shapes_and_finite(self, an_ski, sm_ski):
        """Test stress fields shapes and finite values."""
        _, Z, _ = an_ski.rasterize_solution(num=150)
        phi = sm_ski.scenario.phi
        Sxx = an_ski.Sxx(Z=Z, phi=phi, dz=5)
        Txz = an_ski.Txz(Z=Z, phi=phi, dz=5)
        Szz = an_ski.Szz(Z=Z, phi=phi, dz=5)
        # Consistent shapes
        assert Sxx.shape == Txz.shape
        assert Sxx.shape == Szz.shape
        # Finite values
        assert np.isfinite(Sxx).all()
        assert np.isfinite(Txz).all()
        assert np.isfinite(Szz).all()

    def test_stress_fields_unit_conversion(self, an_ski, sm_ski):
        """Test stress fields unit conversion."""
        _, Z, _ = an_ski.rasterize_solution(num=150)
        phi = sm_ski.scenario.phi
        Sxx_kPa = an_ski.Sxx(Z=Z, phi=phi, dz=5, unit="kPa")
        Sxx_MPa = an_ski.Sxx(Z=Z, phi=phi, dz=5, unit="MPa")
        assert Sxx_kPa.shape == Sxx_MPa.shape
        np.testing.assert_array_almost_equal(Sxx_kPa, Sxx_MPa * 1e3, decimal=8)
        principal_stress_MPa = an_ski.principal_stress_slab(
            Z=Z, phi=phi, dz=5, unit="MPa"
        )
        principal_stress_kPa = an_ski.principal_stress_slab(
            Z=Z, phi=phi, dz=5, unit="kPa"
        )
        assert principal_stress_MPa.shape == principal_stress_kPa.shape
        np.testing.assert_array_almost_equal(
            principal_stress_MPa * 1e3, principal_stress_kPa, decimal=8
        )
        # Test normalized is the same irrespective of unit
        Sxx_kPa_norm = an_ski.Sxx(Z=Z, phi=phi, dz=5, unit="kPa", normalize=True)
        Sxx_MPa_norm = an_ski.Sxx(Z=Z, phi=phi, dz=5, unit="MPa", normalize=True)
        assert Sxx_kPa_norm.shape == Sxx_MPa_norm.shape
        np.testing.assert_array_almost_equal(Sxx_kPa_norm, Sxx_MPa_norm, decimal=8)
        principal_stress_MPa_norm = an_ski.principal_stress_slab(
            Z=Z, phi=phi, dz=5, unit="MPa", normalize=True
        )
        principal_stress_kPa_norm = an_ski.principal_stress_slab(
            Z=Z, phi=phi, dz=5, unit="kPa", normalize=True
        )
        assert principal_stress_MPa_norm.shape == principal_stress_kPa_norm.shape
        np.testing.assert_array_almost_equal(
            principal_stress_MPa_norm, principal_stress_kPa_norm, decimal=8
        )

    def test_principal_stress_slab_variants(self, an_ski, sm_ski):
        """Test principal stress slab variants."""
        _, Z, _ = an_ski.rasterize_solution(num=120)
        phi = sm_ski.scenario.phi
        for val in ("max", "min"):
            Ps = an_ski.principal_stress_slab(Z=Z, phi=phi, dz=5, val=val)
            assert np.isfinite(Ps).all()
        # Normalized tensile principal stress
        Ps_norm = an_ski.principal_stress_slab(
            Z=Z, phi=phi, dz=5, val="max", normalize=True
        )
        assert np.isfinite(Ps_norm).all()
        # Normalizing compressive should error
        with pytest.raises(ValueError):
            _ = an_ski.principal_stress_slab(
                Z=Z, phi=phi, dz=5, val="min", normalize=True
            )

    def test_principal_stress_weaklayer_variants(self, an_ski):
        """Test principal stress weaklayer variants."""
        _, Z, _ = an_ski.rasterize_solution(num=120)
        for val in ("max", "min"):
            ps = an_ski.principal_stress_weaklayer(Z=Z, val=val)
            assert np.isfinite(ps).all()
        # Normalized compressive principal stress in weak layer
        psn = an_ski.principal_stress_weaklayer(Z=Z, val="min", normalize=True)
        assert np.isfinite(psn).all()
        # Normalizing tensile should error
        with pytest.raises(ValueError):
            _ = an_ski.principal_stress_weaklayer(Z=Z, val="max", normalize=True)

    def test_energy_release_rates_shapes(self, an_ski):
        """Test energy release rates shapes."""
        Ginc = an_ski.incremental_ERR()
        assert Ginc.shape == (4,)
        assert np.isfinite(Ginc).all()

        Gdif = an_ski.differential_ERR()
        assert Gdif.shape == (4,)
        assert np.isfinite(Gdif).all()

    def test_energy_release_rate_integrands_non_negative(self):
        """Test that ERR integrands are non-negative for matching stress/strain."""
        slope_angle = 20.0
        system = SystemModel(
            model_input=ModelInput(
                scenario_config=ScenarioConfig(phi=slope_angle, system_type="skier"),
                layers=[Layer()],
                weak_layer=WeakLayer(),
                segments=[Segment(), Segment()],
            ),
            config=Config(),
        )
        analyzer = Analyzer(system_model=system, printing_enabled=False)

        z_uncracked = np.array([[0.0], [0.0], [1.0], [0.2], [0.0], [0.0]])

        def constant_solution(x):
            return np.repeat(z_uncracked, np.size(np.atleast_1d(x)), axis=1)

        mode_i = analyzer._integrand_GI(  # pylint: disable=protected-access
            np.array([0.0, 1.0]), constant_solution, constant_solution
        )
        mode_ii = analyzer._integrand_GII(  # pylint: disable=protected-access
            np.array([0.0, 1.0]), constant_solution, constant_solution
        )

        assert np.all(mode_i >= 0), "Mode I integrand should be non-negative"
        assert np.all(mode_ii >= 0), "Mode II integrand should be non-negative"

    def test_internal_and_external_potentials_pst(self, an_pst):
        """Test internal and external potentials for PST."""
        # Ensure PST-specific methods run
        Pi_total = an_pst.total_potential()
        assert np.isfinite(Pi_total)

        Pi_ext = an_pst._external_potential()  # pylint: disable=protected-access

        assert np.isfinite(Pi_ext)

        Pi_int = an_pst._internal_potential()  # pylint: disable=protected-access

        assert np.isfinite(Pi_int)
        # Consistency: total ≈ int + ext
        assert Pi_total == pytest.approx(Pi_int + Pi_ext, abs=0.5 * 10 ** (-6))
