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
        Layer(rho=170, h=100),
        Layer(rho=190, h=40),
        Layer(rho=230, h=130),
        Layer(rho=250, h=20),
        Layer(rho=210, h=70),
        Layer(rho=380, h=20),
        Layer(rho=280, h=100),
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

    @patch("weac.analysis.criteria_evaluator.Analyzer")
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

    def test_evaluate_SteadyState(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test the evaluate_SteadyState method."""
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
            config=Config(touchdown=True),
        )
        results: SteadyStateResult = evaluator.evaluate_SteadyState(system)
        assert results.converged
        assert results.energy_release_rate > 0
        assert results.touchdown_distance > 0
        assert results.touchdown_distance < system.scenario.L
        max_principal_stress_norm = (
            results.maximal_stress_result.max_principal_stress_norm
        )
        max_Sxx_norm = results.maximal_stress_result.max_Sxx_norm
        assert max_principal_stress_norm > 0
        assert max_Sxx_norm > 0

    def test_evaluate_SteadyState_without_touchdown_in_config(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """
        Test evaluate_SteadyState when SystemModel is initialized without touchdown=True.

        This is a regression test for bug #37: SteadyState evaluation should work
        even if the SystemModel is not initialized with touchdown=True in Config.
        The evaluate_SteadyState method should internally enable touchdown mode
        using toggle_touchdown() to properly invalidate cached properties.
        """
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]
        # Initialize system WITHOUT touchdown=True (default is False)
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi),
            ),
            config=Config(),  # touchdown defaults to False
        )

        # This should not raise AttributeError: 'NoneType' object has no attribute 'l_BC'
        results: SteadyStateResult = evaluator.evaluate_SteadyState(system)

        # Verify results are valid
        assert results.converged
        assert results.energy_release_rate > 0
        assert results.touchdown_distance > 0
        assert results.touchdown_distance < results.system.scenario.L
        max_principal_stress_norm = (
            results.maximal_stress_result.max_principal_stress_norm
        )
        max_Sxx_norm = results.maximal_stress_result.max_Sxx_norm
        assert max_principal_stress_norm > 0
        assert max_Sxx_norm > 0

        # Verify the original system's touchdown state was not modified
        assert not system.config.touchdown

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

    def test_evaluate_SteadyState_modes(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test evaluate_SteadyState with various modes."""
        test_cases = [
            ("C_in_contact", "C_in_contact"),
            ("B_point_contact", "B_point_contact"),
            ("A_free_hanging", "A_free_hanging"),
            (None, "C_in_contact"),  # default mode
        ]

        for mode_param, expected_mode in test_cases:
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
                config=Config(touchdown=True),
            )

            if mode_param is None:
                results: SteadyStateResult = evaluator.evaluate_SteadyState(system)
            else:
                results = evaluator.evaluate_SteadyState(system, mode=mode_param)

            assert results.converged
            assert results.system.slab_touchdown.touchdown_mode == expected_mode

    def test_steady_state_maximal_stress_structure(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test that maximal stress result has correct structure and valid values."""
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
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system)
        maximal_stress = result.maximal_stress_result

        # Check that all arrays have correct shape
        assert (
            maximal_stress.principal_stress_kPa.shape == maximal_stress.Sxx_kPa.shape
        )
        assert (
            maximal_stress.principal_stress_norm.shape == maximal_stress.Sxx_norm.shape
        )

        # Check that arrays are not empty
        assert maximal_stress.principal_stress_kPa.size > 0
        assert maximal_stress.Sxx_kPa.size > 0

        # Check that maximum values are positive
        assert maximal_stress.max_principal_stress_norm > 0
        assert maximal_stress.max_Sxx_norm > 0

        # Check that slab_tensile_criterion is between 0 and 1
        assert maximal_stress.slab_tensile_criterion >= 0
        assert maximal_stress.slab_tensile_criterion <= 1

    def test_steady_state_energy_release_rate_positive(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test that steady state ERR is always positive."""
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
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system)
        assert result.energy_release_rate > 0, "Steady state ERR should be positive"

    def test_steady_state_with_different_weak_layers(
        self, evaluator, layers, phi, segments_length
    ):
        """Test steady state evaluation with different weak layer properties."""
        weak_layers = [
            WeakLayer(rho=150, h=10, G_Ic=0.3, G_IIc=0.6, kn=50, kt=50),
            WeakLayer(rho=200, h=15, G_Ic=0.8, G_IIc=1.2, kn=150, kt=150),
            WeakLayer(rho=180, h=10, G_Ic=0.5, G_IIc=0.8, kn=100, kt=100),
        ]

        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]

        for weak_layer in weak_layers:
            system = SystemModel(
                model_input=ModelInput(
                    layers=layers,
                    weak_layer=weak_layer,
                    segments=segments,
                    scenario_config=ScenarioConfig(phi=phi),
                ),
                config=Config(touchdown=True),
            )

            result = evaluator.evaluate_SteadyState(system)
            assert result.converged
            assert result.energy_release_rate > 0
            assert result.touchdown_distance > 0

    def test_steady_state_with_different_slope_angles(
        self, evaluator, layers, weak_layer, segments_length
    ):
        """Test steady state evaluation at different slope angles."""
        slope_angles = [20.0, 30.0, 40.0, 45.0]

        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]

        for phi in slope_angles:
            system = SystemModel(
                model_input=ModelInput(
                    layers=layers,
                    weak_layer=weak_layer,
                    segments=segments,
                    scenario_config=ScenarioConfig(phi=phi),
                ),
                config=Config(touchdown=True),
            )

            result = evaluator.evaluate_SteadyState(system)
            assert result.converged
            assert result.energy_release_rate > 0
            assert result.touchdown_distance > 0
            assert result.maximal_stress_result is not None

    def test_steady_state_system_isolation(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test that evaluate_SteadyState doesn't modify original system."""
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
            config=Config(touchdown=True),
        )

        original_segments = system.scenario.segments.copy()
        original_phi = system.scenario.phi
        original_L = system.scenario.L

        result = evaluator.evaluate_SteadyState(system)

        # Verify original system is unchanged
        assert len(system.scenario.segments) == len(original_segments)
        assert system.scenario.phi == original_phi
        assert system.scenario.L == original_L

        # Verify result system is different
        assert result.system.scenario.phi == 0.0

    def test_steady_state_message_format(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test that steady state result message is correctly formatted."""
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
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system)
        assert isinstance(result.message, str)
        assert len(result.message) > 0
        assert result.message == "Steady State evaluation successful."

    def test_steady_state_normalized_stresses_consistency(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test consistency between absolute and normalized stress values."""
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
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system)
        maximal_stress = result.maximal_stress_result

        # Verify that max normalized values match the max of the arrays
        computed_max_principal = np.max(maximal_stress.principal_stress_norm)
        computed_max_Sxx = np.max(maximal_stress.Sxx_norm)

        np.testing.assert_almost_equal(
            computed_max_principal,
            maximal_stress.max_principal_stress_norm,
            decimal=5,
        )
        np.testing.assert_almost_equal(
            computed_max_Sxx,
            maximal_stress.max_Sxx_norm,
            decimal=5,
        )

    def test_steady_state_with_thin_weak_layer(
        self, evaluator, layers, phi, segments_length
    ):
        """Test steady state evaluation with a thin weak layer."""
        thin_weak_layer = WeakLayer(rho=180, h=5, G_Ic=0.5, G_IIc=0.8, kn=100, kt=100)
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=thin_weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi),
            ),
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system)
        assert result.converged
        assert result.touchdown_distance > 0
        assert result.energy_release_rate > 0

    def test_steady_state_with_thick_weak_layer(
        self, evaluator, layers, phi, segments_length
    ):
        """Test steady state evaluation with a thick weak layer."""
        thick_weak_layer = WeakLayer(rho=180, h=20, G_Ic=0.5, G_IIc=0.8, kn=100, kt=100)
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=thick_weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi),
            ),
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system)
        assert result.converged
        assert result.touchdown_distance > 0
        assert result.energy_release_rate > 0

    def test_steady_state_vertical_mode_warning(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test that vertical mode raises a warning."""
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
            config=Config(touchdown=True),
        )

        with pytest.warns(UserWarning):
            result = evaluator.evaluate_SteadyState(system, vertical=True)
            assert result.converged

    def test_steady_state_slab_tensile_criterion_calculation(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test the slab tensile criterion calculation in steady state."""
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
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system)
        slab_tensile_criterion = result.maximal_stress_result.slab_tensile_criterion

        # Verify it's within valid range
        assert slab_tensile_criterion >= 0.0
        assert slab_tensile_criterion <= 1.0

    def test_steady_state_stress_arrays_shapes(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test that stress arrays in maximal stress result have consistent shapes."""
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
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system)
        maximal_stress = result.maximal_stress_result

        # All stress arrays should have the same shape
        assert (
            maximal_stress.principal_stress_kPa.shape
            == maximal_stress.principal_stress_norm.shape
        )
        assert maximal_stress.Sxx_kPa.shape == maximal_stress.Sxx_norm.shape
        assert (
            maximal_stress.principal_stress_kPa.shape == maximal_stress.Sxx_kPa.shape
        )

        # Arrays should be 2D (spatial dimensions)
        assert len(maximal_stress.principal_stress_kPa.shape) == 2
        assert len(maximal_stress.Sxx_kPa.shape) == 2

    def test_steady_state_with_varying_stiffness(
        self, evaluator, layers, phi, segments_length
    ):
        """Test steady state evaluation with different weak layer stiffnesses."""
        stiffness_values = [(50, 50), (100, 100), (200, 200)]

        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]

        results_list = []
        for kn, kt in stiffness_values:
            weak_layer = WeakLayer(rho=180, h=10, G_Ic=0.5, G_IIc=0.8, kn=kn, kt=kt)
            system = SystemModel(
                model_input=ModelInput(
                    layers=layers,
                    weak_layer=weak_layer,
                    segments=segments,
                    scenario_config=ScenarioConfig(phi=phi),
                ),
                config=Config(touchdown=True),
            )

            result = evaluator.evaluate_SteadyState(system)
            assert result.converged
            results_list.append(result)

        # Verify that all results are valid
        for result in results_list:
            assert result.touchdown_distance > 0
            assert result.energy_release_rate > 0

    def test_steady_state_scenario_config_phi_reset(
        self, evaluator, layers, weak_layer, segments_length
    ):
        """Test that steady state evaluation sets phi to 0.0 internally."""
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]
        original_phi = 35.0
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=original_phi),
            ),
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system)

        # The result system should have phi=0 for steady state evaluation
        assert result.system.scenario.phi == 0.0

        # The original system should be unchanged
        assert system.scenario.phi == original_phi

    def test_steady_state_touchdown_distance_bounds(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test that touchdown distance is within reasonable bounds."""
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
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system)

        # Touchdown distance should be positive
        assert result.touchdown_distance > 0

        # Touchdown distance should be less than the cut distance (5e3)
        # which is the length of the hanging segment in the steady state setup
        assert result.touchdown_distance < 5e3

    def test_steady_state_mode_forced_correctly(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """Test that the forced touchdown mode is correctly applied."""
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]

        for mode in ["C_in_contact", "B_point_contact", "A_free_hanging"]:
            system = SystemModel(
                model_input=ModelInput(
                    layers=layers,
                    weak_layer=weak_layer,
                    segments=segments,
                    scenario_config=ScenarioConfig(phi=phi),
                ),
                config=Config(touchdown=True),
            )

            result = evaluator.evaluate_SteadyState(system, mode=mode)

            # Verify the result system has the correct forced mode
            assert result.system.slab_touchdown.touchdown_mode == mode

    def test_steady_state_regression_c_in_contact_values(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """
        Regression test: Check specific numerical values for C_in_contact mode.

        These values are baseline references to catch breaking changes.
        Update these values if intentional changes are made to the calculation.
        """
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
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system, mode="C_in_contact")

        expected_touchdown_distance = 1913.1270
        expected_err = 5.7857
        expected_max_principal_stress_norm = 443161.8028
        expected_max_Sxx_norm = 2.2260
        expected_slab_tensile_criterion = 0.51129

        # Allow small tolerances for numerical precision
        np.testing.assert_allclose(
            result.touchdown_distance,
            expected_touchdown_distance,
            rtol=1e-4,
            err_msg="Touchdown distance changed unexpectedly",
        )
        np.testing.assert_allclose(
            result.energy_release_rate,
            expected_err,
            rtol=1e-4,
            err_msg="Energy release rate changed unexpectedly",
        )
        np.testing.assert_allclose(
            result.maximal_stress_result.max_principal_stress_norm,
            expected_max_principal_stress_norm,
            rtol=1e-4,
            err_msg="Max principal stress norm changed unexpectedly",
        )
        np.testing.assert_allclose(
            result.maximal_stress_result.max_Sxx_norm,
            expected_max_Sxx_norm,
            rtol=1e-4,
            err_msg="Max Sxx norm changed unexpectedly",
        )
        np.testing.assert_allclose(
            result.maximal_stress_result.slab_tensile_criterion,
            expected_slab_tensile_criterion,
            rtol=1e-4,
            err_msg="Slab tensile criterion changed unexpectedly",
        )

    def test_steady_state_regression_b_point_contact_values(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """
        Regression test: Check specific numerical values for B_point_contact mode.

        These values are baseline references to catch breaking changes.
        Update these values if intentional changes are made to the calculation.
        """
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
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system, mode="B_point_contact")

        expected_touchdown_distance = 2111.1553
        expected_err = 5.5184
        expected_max_Sxx_norm = 2.1727

        np.testing.assert_allclose(
            result.touchdown_distance,
            expected_touchdown_distance,
            rtol=1e-4,
            err_msg="Touchdown distance changed unexpectedly",
        )
        np.testing.assert_allclose(
            result.energy_release_rate,
            expected_err,
            rtol=1e-4,
            err_msg="Energy release rate changed unexpectedly",
        )
        np.testing.assert_allclose(
            result.maximal_stress_result.max_Sxx_norm,
            expected_max_Sxx_norm,
            rtol=1e-4,
            err_msg="Max Sxx norm changed unexpectedly",
        )

    def test_steady_state_regression_a_free_hanging_values(
        self, evaluator, layers, weak_layer, phi, segments_length
    ):
        """
        Regression test: Check specific numerical values for A_free_hanging mode.

        These values are baseline references to catch breaking changes.
        Update these values if intentional changes are made to the calculation.
        """
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
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system, mode="A_free_hanging")

        expected_touchdown_distance = 1207.9559
        expected_err = 5.6629
        expected_max_Sxx_norm = 2.2981

        np.testing.assert_allclose(
            result.touchdown_distance,
            expected_touchdown_distance,
            rtol=1e-4,
            err_msg="Touchdown distance changed unexpectedly",
        )
        np.testing.assert_allclose(
            result.energy_release_rate,
            expected_err,
            rtol=1e-4,
            err_msg="Energy release rate changed unexpectedly",
        )
        np.testing.assert_allclose(
            result.maximal_stress_result.max_Sxx_norm,
            expected_max_Sxx_norm,
            rtol=1e-4,
            err_msg="Max Sxx norm changed unexpectedly",
        )

    def test_steady_state_regression_different_weak_layer(
        self, evaluator, layers, phi, segments_length
    ):
        """
        Regression test: Check specific numerical values with different weak layer properties.

        Tests with a weaker layer to ensure parameter variations are handled consistently.
        """
        weak_weak_layer = WeakLayer(rho=150, h=10, G_Ic=0.3, G_IIc=0.6, kn=50, kt=50)
        segments = [
            Segment(length=segments_length, has_foundation=True, m=0),
            Segment(length=segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=layers,
                weak_layer=weak_weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=phi),
            ),
            config=Config(touchdown=True),
        )

        result = evaluator.evaluate_SteadyState(system, mode="C_in_contact")

        expected_touchdown_distance = 1911.4747
        expected_err = 5.0860
        expected_slab_tensile_criterion = 0.5092

        np.testing.assert_allclose(
            result.touchdown_distance,
            expected_touchdown_distance,
            rtol=1e-4,
            err_msg="Touchdown distance changed unexpectedly for weak layer",
        )
        np.testing.assert_allclose(
            result.energy_release_rate,
            expected_err,
            rtol=1e-4,
            err_msg="Energy release rate changed unexpectedly for weak layer",
        )
        np.testing.assert_allclose(
            result.maximal_stress_result.slab_tensile_criterion,
            expected_slab_tensile_criterion,
            rtol=1e-4,
            err_msg="Slab tensile criterion changed unexpectedly for weak layer",
        )
