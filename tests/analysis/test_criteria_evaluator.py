"""
This module contains tests for the CriteriaEvaluator class.
"""

# Standard library imports
from types import SimpleNamespace
from unittest.mock import patch

# Third party imports
import numpy as np
import pytest

# weac imports
from weac.analysis.analyzer import Analyzer
from weac.analysis.criteria_evaluator import (
    CoupledCriterionResult,
    CriteriaEvaluator,
    FindMinimumForceResult,
    SteadyStateResult,
)
from weac.components import (
    Config,
    CriteriaConfig,
    Layer,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac.components.model_input import ModelInput
from weac.core.system_model import SystemModel


@pytest.fixture
def config():
    """Default WEAC config."""
    return Config()


@pytest.fixture
def criteria_config():
    """Default criteria config."""
    return CriteriaConfig()


@pytest.fixture
def evaluator(criteria_config):
    """CriteriaEvaluator bound to the criteria config fixture."""
    return CriteriaEvaluator(criteria_config)


@pytest.fixture
def layers():
    """Standard multi-layer slab profile used across evaluator tests."""
    return [
        Layer(rho=170, h=100, tensile_strength_method="hybrid"),
        Layer(rho=190, h=40, tensile_strength_method="hybrid"),
        Layer(rho=230, h=130, tensile_strength_method="hybrid"),
        Layer(rho=250, h=20, tensile_strength_method="hybrid"),
        Layer(rho=210, h=70, tensile_strength_method="hybrid"),
        Layer(rho=380, h=20, tensile_strength_method="hybrid"),
        Layer(rho=280, h=100, tensile_strength_method="hybrid"),
    ]


@pytest.fixture
def weak_layer():
    """Default weak layer for evaluator tests."""
    return WeakLayer(rho=180, h=10, G_Ic=0.5, G_IIc=0.8, kn=100, kt=100)


@pytest.fixture
def phi():
    """Default slope angle."""
    return 30.0


@pytest.fixture
def segments_length():
    """Default segment length used in evaluator setups."""
    return 10000


def _make_ss_system(
    layers,
    weak_layer,
    segments_length,
    *,
    phi=30.0,
    touchdown=True,
    override_weak_layer=None,
):
    """Build a system for hybrid steady-state tests."""
    segments = [
        Segment(length=segments_length, has_foundation=True, m=0),
        Segment(length=segments_length, has_foundation=True, m=0),
    ]
    return SystemModel(
        model_input=ModelInput(
            layers=layers,
            weak_layer=override_weak_layer or weak_layer,
            segments=segments,
            scenario_config=ScenarioConfig(phi=phi),
        ),
        config=Config(touchdown=touchdown),
    )


class TestCriteriaEvaluator:
    """Test suite for the CriteriaEvaluator."""

    def test_fracture_toughness_criterion(self, evaluator, weak_layer):
        """Test the fracture toughness criterion calculation."""
        g_delta = evaluator.fracture_toughness_envelope(
            G_I=0.25, G_II=0.4, weak_layer=weak_layer
        )
        # Expected: (|0.25| / 0.5)^5.0 + (|0.4| / 0.8)^2.22
        # = (0.5)^5 + (0.5)^2.22 = 0.03125 + 0.2146...
        np.testing.assert_almost_equal(g_delta, 0.2455609957, decimal=5)

    def test_stress_envelope_adam_unpublished(
        self, criteria_config, evaluator, weak_layer
    ):
        """Test the 'adam_unpublished' stress envelope."""
        criteria_config.stress_envelope_method = "adam_unpublished"
        sigma, tau = np.array([2.0]), np.array([1.5])
        result = evaluator.stress_envelope(sigma, tau, weak_layer)
        assert result[0] > 0

    @patch("weac.analysis.coupled_criterion.Analyzer")
    def test_calculate_maximal_stresses_applies_directional_low_density_exclusion(
        self, mock_analyzer_cls
    ):  # pylint: disable=protected-access
        """Test that weak snow is excluded only after downward failure growth."""
        sxx_kpa = np.zeros((4, 1))
        principal_stress_kPa = np.zeros((4, 1))
        sxx_norm = np.array([[1.5], [0.5], [0.5], [0.5]])
        principal_stress_norm = np.full((4, 1), 0.5)

        mock_analyzer = mock_analyzer_cls.return_value
        mock_analyzer.rasterize_solution.return_value = (
            None,
            np.array([0, 1, 2, 3]),
            None,
        )
        mock_analyzer.Sxx.side_effect = (
            lambda *args, normalize=False, **kwargs: sxx_norm if normalize else sxx_kpa
        )
        mock_analyzer.principal_stress_slab.side_effect = (
            lambda *args, normalize=False, **kwargs: (
                principal_stress_norm if normalize else principal_stress_kPa
            )
        )
        mock_analyzer.get_zmesh.return_value = {
            "rho": np.array([130.0, 90.0, 90.0, 130.0]) * 1e-12
        }
        system = SimpleNamespace(scenario=SimpleNamespace(phi=30.0))

        # Access the helper directly so the test can isolate the density-threshold logic.
        top_broken_result = CriteriaEvaluator(
            CriteriaConfig()
        )._calculate_maximal_stresses(system=system)  # pylint: disable=protected-access

        sxx_norm = np.array([[0.5], [0.5], [0.5], [1.5]])
        top_unbroken_result = CriteriaEvaluator(
            CriteriaConfig()
        )._calculate_maximal_stresses(system=system)  # pylint: disable=protected-access

        assert top_broken_result.slab_tensile_criterion == pytest.approx(
            1 / 2, abs=0.5 * 10 ** (-7)
        )
        assert top_unbroken_result.slab_tensile_criterion == pytest.approx(
            1 / 4, abs=0.5 * 10 ** (-7)
        )

    def test_find_minimum_force_convergence(
        self, evaluator, layers, weak_layer, phi, segments_length, config
    ):
        """Test the convergence of find_minimum_force."""
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi),
            ),
            config=config,
        )
        results: FindMinimumForceResult = evaluator.find_minimum_force(system=system)
        skier_weight = results.critical_skier_weight
        new_segments = results.new_segments
        assert skier_weight > 0
        assert new_segments is not None

    def test_find_crack_length_for_weight(
        self, evaluator, layers, weak_layer, phi, segments_length, config
    ):
        """Test the find_crack_length_for_weight method."""
        skier_weight = 100  # A substantial weight
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=0, has_foundation=False, m=skier_weight),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi, cut_length=0),
            ),
            config=config,
        )
        crack_len, segments = evaluator.find_crack_length_for_weight(
            system, skier_weight
        )
        assert crack_len >= 0
        assert isinstance(segments, list)
        assert all(isinstance(s, Segment) for s in segments)

    def test_find_crack_length_for_weight_single_rasterize(
        self, evaluator, layers, weak_layer, phi, segments_length, config
    ):
        """Crack-length search rasterizes the cracked field once per weight."""
        skier_weight = 100
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=0, has_foundation=False, m=skier_weight),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi, cut_length=0),
            ),
            config=config,
        )
        original_rasterize = Analyzer.rasterize_solution
        rasterize_calls = []

        def counting_rasterize(self, *args, **kwargs):
            rasterize_calls.append(kwargs.get("mode", args[0] if args else None))
            return original_rasterize(self, *args, **kwargs)

        with patch.object(Analyzer, "rasterize_solution", counting_rasterize):
            crack_len, new_segments = evaluator.find_crack_length_for_weight(
                system, skier_weight
            )

        assert len(rasterize_calls) == 1
        assert rasterize_calls[0] == "cracked"
        assert crack_len >= 0
        assert all(isinstance(s, Segment) for s in new_segments)

    def test_check_crack_propagation_stable(
        self, evaluator, layers, weak_layer, phi, segments_length, config
    ):
        """Test check_crack_propagation for a stable scenario (no crack)."""
        segments = [Segment(length=segments_length, has_foundation=True, m=0)]
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi),
            ),
            config=config,
        )
        g_delta, can_propagate = evaluator.check_crack_self_propagation(system)
        assert not can_propagate
        assert g_delta < 1.0, "Stable scenario should be below the fracture envelope"

    def test_check_crack_propagation_unstable(
        self, evaluator, layers, phi, segments_length, config
    ):
        """Test check_crack_propagation for an unstable scenario (pre-cracked)."""
        # A configuration with a very weak layer and a large crack that should
        # be unstable under its own weight.
        unstable_weak_layer = WeakLayer(
            rho=180, h=10, G_Ic=0.01, G_IIc=0.01, kn=100, kt=100
        )
        crack_length = 4000  # 4m crack
        side_length = (segments_length - crack_length) / 2
        segments = [
            Segment(length=side_length, has_foundation=True, m=0),
            Segment(length=crack_length, has_foundation=False, m=0),
            Segment(length=side_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=unstable_weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi),
            ),
            config=config,
        )
        g_delta, can_propagate = evaluator.check_crack_self_propagation(system)
        assert g_delta > 1
        assert can_propagate

    def test_evaluate_coupled_criterion_full_run(
        self, evaluator, layers, weak_layer, phi, segments_length, config
    ):
        """Test the main evaluate_coupled_criterion workflow."""
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi),
            ),
            config=config,
        )
        results: CoupledCriterionResult = evaluator.evaluate_coupled_criterion(
            system=system
        )
        assert isinstance(results, CoupledCriterionResult)
        assert results.critical_skier_weight > 0
        assert results.history is not None
        history = results.history
        assert history is not None
        assert len(history.sigma_maxs) == len(history.skier_weights)
        assert len(history.sigma_maxs) > 0
        assert len(history.tau_maxs) == len(history.skier_weights)
        assert len(history.tau_maxs) > 0

    def test_evaluate_coupled_criterion_finds_force_once(self):
        """Force-finding must run once even when max_iterations triggers damping."""
        layers = [Layer(rho=170, h=100), Layer(rho=230, h=130)]
        wl = WeakLayer(rho=180, h=20)
        segs = [Segment(length=10000, has_foundation=True, m=0)]
        sc = ScenarioConfig(phi=30.0, system_type="skier", cut_length=0.0)
        mi = ModelInput(layers=layers, weak_layer=wl, segments=segs, scenario_config=sc)
        sm = SystemModel(model_input=mi, config=Config(touchdown=True))
        evaluator = CriteriaEvaluator(CriteriaConfig())
        engine = evaluator._coupled  # pylint: disable=protected-access

        with patch.object(
            engine,
            "find_minimum_force",
            wraps=engine.find_minimum_force,
        ) as mock_fm:
            results = evaluator.evaluate_coupled_criterion(
                system=sm, max_iterations=10
            )

        assert mock_fm.call_count == 1
        assert results.converged
        assert results.critical_skier_weight > 0

    def test_evaluate_coupled_criterion_skips_uncracked_rasterize_after_iter1(self):
        """Main-loop uncracked rasterize runs once; force-find adds one more."""
        layers = [Layer(rho=170, h=100), Layer(rho=230, h=130)]
        wl = WeakLayer(rho=180, h=20)
        segs = [Segment(length=10000, has_foundation=True, m=0)]
        sc = ScenarioConfig(phi=30.0, system_type="skier", cut_length=0.0)
        mi = ModelInput(layers=layers, weak_layer=wl, segments=segs, scenario_config=sc)
        sm = SystemModel(model_input=mi, config=Config(touchdown=True))
        evaluator = CriteriaEvaluator(CriteriaConfig())

        original_rasterize = Analyzer.rasterize_solution
        rasterize_modes = []

        def counting_rasterize(self, *args, **kwargs):
            mode = kwargs.get("mode", args[0] if args else "cracked")
            rasterize_modes.append(mode)
            return original_rasterize(self, *args, **kwargs)

        with patch.object(Analyzer, "rasterize_solution", counting_rasterize):
            results = evaluator.evaluate_coupled_criterion(
                system=sm, max_iterations=10
            )

        assert results.converged
        assert results.iterations > 1
        assert not results.pure_stress_criteria
        uncracked_count = sum(1 for m in rasterize_modes if m == "uncracked")
        # 1x find_minimum_force + 1x main-loop iteration 1
        assert uncracked_count == 2
        history = results.history
        assert history is not None
        assert len(history.g_deltas) == results.iterations
        assert len(history.sigma_maxs) == results.iterations
        # Post-iter-1 stress history reuses the iteration-1 sample
        assert all(s == history.sigma_maxs[0] for s in history.sigma_maxs[1:])

    def test_evaluate_SteadyState(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Hybrid SS returns structured tensile / ERR blocks."""
        system = _make_ss_system(layers, weak_layer, segments_length, phi=phi)
        results: SteadyStateResult = evaluator.evaluate_SteadyState(system)
        assert results.converged
        assert results.tensile.critical_cut_length > 0
        assert results.err.energy_release_rate > 0
        assert results.phi == phi
        assert results.tensile.cut_direction_winner in ("upslope", "downslope")
        assert results.err.cut_direction_winner in ("upslope", "downslope")
        stress = results.tensile.maximal_stress_result
        assert stress is not None
        assert stress.max_Sxx_norm > 0

    def test_evaluate_SteadyState_without_touchdown_in_config(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Hybrid SS does not require touchdown=True on the input system."""
        system = _make_ss_system(
            layers, weak_layer, segments_length, phi=phi, touchdown=False
        )
        results: SteadyStateResult = evaluator.evaluate_SteadyState(system)
        assert results.converged
        assert results.tensile.critical_cut_length > 0
        assert results.err.energy_release_rate > 0
        assert not system.config.touchdown

    def test_evaluate_SteadyState_rejects_mode_kwarg(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """TouchdownMode / mode= is no longer part of the public SS API."""
        system = _make_ss_system(layers, weak_layer, segments_length, phi=phi)
        with pytest.raises(TypeError):
            evaluator.evaluate_SteadyState(system, mode="B_point_contact")

    def test_steady_state_maximal_stress_structure(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Tensile-leg maximal stress has valid structure."""
        result = evaluator.evaluate_SteadyState(
            _make_ss_system(layers, weak_layer, segments_length, phi=phi)
        )
        maximal_stress = result.tensile.maximal_stress_result
        assert maximal_stress is not None
        assert (
            maximal_stress.principal_stress_kPa.shape == maximal_stress.Sxx_kPa.shape
        )
        assert maximal_stress.principal_stress_kPa.size > 0
        assert maximal_stress.max_Sxx_norm > 0
        assert 0 <= maximal_stress.slab_tensile_criterion <= 1

    def test_steady_state_energy_release_rate_positive(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """ERR-leg energy release rate is positive."""
        result = evaluator.evaluate_SteadyState(
            _make_ss_system(layers, weak_layer, segments_length, phi=phi)
        )
        assert result.err.energy_release_rate > 0

    def test_steady_state_with_different_weak_layers(
        self, evaluator, layers, phi, segments_length
    ):
        """Hybrid SS converges across weak-layer property variations."""
        weak_layers = [
            WeakLayer(rho=150, h=10, G_Ic=0.3, G_IIc=0.6, kn=50, kt=50),
            WeakLayer(rho=200, h=15, G_Ic=0.8, G_IIc=1.2, kn=150, kt=150),
            WeakLayer(rho=180, h=10, G_Ic=0.5, G_IIc=0.8, kn=100, kt=100),
        ]
        for weak_layer in weak_layers:
            result = evaluator.evaluate_SteadyState(
                _make_ss_system(
                    layers,
                    weak_layer,
                    segments_length,
                    phi=phi,
                    override_weak_layer=weak_layer,
                )
            )
            assert result.converged
            assert result.err.energy_release_rate > 0
            assert result.tensile.critical_cut_length > 0

    @pytest.mark.parametrize("slope_angle", [20.0, 30.0, 40.0, 45.0])
    def test_steady_state_with_different_slope_angles(
        self, evaluator, layers, weak_layer, segments_length, slope_angle
    ):
        """Hybrid SS uses system φ (no flat-slab override)."""
        result = evaluator.evaluate_SteadyState(
            _make_ss_system(layers, weak_layer, segments_length, phi=slope_angle)
        )
        assert result.converged
        assert result.phi == slope_angle
        assert result.err.energy_release_rate > 0
        assert result.tensile.critical_cut_length > 0

    def test_steady_state_system_isolation(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """evaluate_SteadyState does not mutate the caller's system."""
        system = _make_ss_system(layers, weak_layer, segments_length, phi=phi)
        original_phi = system.scenario.phi
        original_L = system.scenario.L
        original_nseg = len(system.scenario.segments)
        result = evaluator.evaluate_SteadyState(system)
        assert system.scenario.phi == original_phi
        assert system.scenario.L == original_L
        assert len(system.scenario.segments) == original_nseg
        assert result.phi == original_phi

    def test_steady_state_message_format(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Hybrid result message mentions both legs."""
        result = evaluator.evaluate_SteadyState(
            _make_ss_system(layers, weak_layer, segments_length, phi=phi)
        )
        assert isinstance(result.message, str)
        assert "tensile[" in result.message
        assert "err[" in result.message

    def test_steady_state_normalized_stresses_consistency(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Max normalized stresses match array maxima on the tensile leg."""
        result = evaluator.evaluate_SteadyState(
            _make_ss_system(layers, weak_layer, segments_length, phi=phi)
        )
        maximal_stress = result.tensile.maximal_stress_result
        assert maximal_stress is not None
        np.testing.assert_almost_equal(
            np.max(maximal_stress.principal_stress_norm),
            maximal_stress.max_principal_stress_norm,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            np.max(maximal_stress.Sxx_norm),
            maximal_stress.max_Sxx_norm,
            decimal=5,
        )

    def test_steady_state_preserves_phi(
        self, evaluator, layers, weak_layer, segments_length
    ):
        """Hybrid SS must not force φ→0."""
        original_phi = 35.0
        system = _make_ss_system(
            layers, weak_layer, segments_length, phi=original_phi
        )
        result = evaluator.evaluate_SteadyState(system)
        assert result.phi == original_phi
        assert system.scenario.phi == original_phi
        if result.tensile.system is not None:
            assert result.tensile.system.scenario.phi == original_phi

    def test_steady_state_critical_cut_length_bounds(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Tensile critical cut length is within the search window."""
        result = evaluator.evaluate_SteadyState(
            _make_ss_system(layers, weak_layer, segments_length, phi=phi)
        )
        assert result.tensile.critical_cut_length > 0
        assert result.tensile.critical_cut_length <= 5000.0

    def test_steady_state_with_thin_weak_layer(
        self, evaluator, layers, phi, segments_length
    ):
        """Hybrid SS converges with a thin weak layer."""
        thin_weak_layer = WeakLayer(rho=180, h=5, G_Ic=0.5, G_IIc=0.8, kn=100, kt=100)
        result = evaluator.evaluate_SteadyState(
            _make_ss_system(
                layers,
                thin_weak_layer,
                segments_length,
                phi=phi,
                override_weak_layer=thin_weak_layer,
            )
        )
        assert result.converged
        assert result.tensile.critical_cut_length > 0
        assert result.err.energy_release_rate > 0

    def test_steady_state_with_thick_weak_layer(
        self, evaluator, layers, phi, segments_length
    ):
        """Hybrid SS converges with a thick weak layer."""
        thick_weak_layer = WeakLayer(rho=180, h=20, G_Ic=0.5, G_IIc=0.8, kn=100, kt=100)
        result = evaluator.evaluate_SteadyState(
            _make_ss_system(
                layers,
                thick_weak_layer,
                segments_length,
                phi=phi,
                override_weak_layer=thick_weak_layer,
            )
        )
        assert result.converged
        assert result.tensile.critical_cut_length > 0
        assert result.err.energy_release_rate > 0

    def test_find_minimum_crack_length(
        self, evaluator, layers, weak_layer, phi, segments_length, config
    ):
        """Test the find_minimum_crack_length method."""
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi),
            ),
            config=config,
        )
        crack_length, new_segments = evaluator.find_minimum_crack_length(system)
        assert crack_length > 0
        assert isinstance(new_segments, list)
        assert all(isinstance(s, Segment) for s in new_segments)
