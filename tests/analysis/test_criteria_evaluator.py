"""
This module contains tests for the CriteriaEvaluator class.
"""

# Standard library imports
import unittest

# Third party imports
import numpy as np

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


class TestCriteriaEvaluator(unittest.TestCase):
    """Test suite for the CriteriaEvaluator."""

    def setUp(self):
        """Set up common objects for testing."""
        self.config = Config()
        self.criteria_config = CriteriaConfig()
        self.evaluator = CriteriaEvaluator(self.criteria_config)

        self.layers = [
            Layer(rho=170, h=100),
            Layer(rho=190, h=40),
            Layer(rho=230, h=130),
            Layer(rho=250, h=20),
            Layer(rho=210, h=70),
            Layer(rho=380, h=20),
            Layer(rho=280, h=100),
        ]
        self.weak_layer = WeakLayer(rho=180, h=10, G_Ic=0.5, G_IIc=0.8, kn=100, kt=100)
        self.phi = 30.0
        self.segments_length = 10000

    def test_fracture_toughness_criterion(self):
        """Test the fracture toughness criterion calculation."""
        g_delta = self.evaluator.fracture_toughness_envelope(
            G_I=0.25, G_II=0.4, weak_layer=self.weak_layer
        )
        # Expected: (|0.25| / 0.5)^5.0 + (|0.4| / 0.8)^2.22
        # = (0.5)^5 + (0.5)^2.22 = 0.03125 + 0.2146...
        np.testing.assert_almost_equal(g_delta, 0.2455609957, decimal=5)

    def test_stress_envelope_adam_unpublished(self):
        """Test the 'adam_unpublished' stress envelope."""
        self.criteria_config.stress_envelope_method = "adam_unpublished"
        sigma, tau = np.array([2.0]), np.array([1.5])
        result = self.evaluator.stress_envelope(sigma, tau, self.weak_layer)
        self.assertGreater(result[0], 0)

    def test_find_minimum_force_convergence(self):
        """Test the convergence of find_minimum_force."""
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=self.config,
        )
        results: FindMinimumForceResult = self.evaluator.find_minimum_force(
            system=system
        )
        skier_weight = results.critical_skier_weight
        new_segments = results.new_segments
        self.assertGreater(skier_weight, 0)
        self.assertIsNotNone(new_segments)

    def test_find_crack_length_for_weight(self):
        """Test the find_crack_length_for_weight method."""
        skier_weight = 100  # A substantial weight
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=0, has_foundation=False, m=skier_weight),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi, cut_length=0),
            ),
            config=self.config,
        )
        crack_len, segments = self.evaluator.find_crack_length_for_weight(
            system, skier_weight
        )
        self.assertGreaterEqual(crack_len, 0)
        self.assertIsInstance(segments, list)
        self.assertTrue(all(isinstance(s, Segment) for s in segments))

    def test_check_crack_propagation_stable(self):
        """Test check_crack_propagation for a stable scenario (no crack)."""
        segments = [Segment(length=self.segments_length, has_foundation=True, m=0)]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=self.config,
        )
        g_delta, can_propagate = self.evaluator.check_crack_self_propagation(system)
        self.assertFalse(can_propagate)
        self.assertLess(
            g_delta, 1.0, "Stable scenario should be below the fracture envelope"
        )

    def test_check_crack_propagation_unstable(self):
        """Test check_crack_propagation for an unstable scenario (pre-cracked)."""
        # A configuration with a very weak layer and a large crack that should
        # be unstable under its own weight.
        unstable_weak_layer = WeakLayer(
            rho=180, h=10, G_Ic=0.01, G_IIc=0.01, kn=100, kt=100
        )
        crack_length = 4000  # 4m crack
        side_length = (self.segments_length - crack_length) / 2
        segments = [
            Segment(length=side_length, has_foundation=True, m=0),
            Segment(length=crack_length, has_foundation=False, m=0),
            Segment(length=side_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=unstable_weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=self.config,
        )
        g_delta, can_propagate = self.evaluator.check_crack_self_propagation(system)
        self.assertGreater(g_delta, 1)
        self.assertTrue(can_propagate)

    def test_evaluate_coupled_criterion_full_run(self):
        """Test the main evaluate_coupled_criterion workflow."""
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=self.config,
        )
        results: CoupledCriterionResult = self.evaluator.evaluate_coupled_criterion(
            system=system
        )
        self.assertIsInstance(results, CoupledCriterionResult)
        self.assertGreater(results.critical_skier_weight, 0)

    def test_evaluate_SteadyState(self):
        """Test the evaluate_SteadyState method."""
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=Config(touchdown=True),
        )
        results: SteadyStateResult = self.evaluator.evaluate_SteadyState(system)
        self.assertTrue(results.converged)
        self.assertGreater(results.energy_release_rate, 0)
        self.assertGreater(results.touchdown_distance, 0)
        self.assertLess(results.touchdown_distance, system.scenario.L)
        max_principal_stress_norm = (
            results.maximal_stress_result.max_principal_stress_norm
        )
        max_Sxx_norm = results.maximal_stress_result.max_Sxx_norm
        self.assertGreater(max_principal_stress_norm, 0)
        self.assertGreater(max_Sxx_norm, 0)

    def test_evaluate_SteadyState_without_touchdown_in_config(self):
        """
        Test evaluate_SteadyState when SystemModel is initialized without touchdown=True.

        This is a regression test for bug #37: SteadyState evaluation should work
        even if the SystemModel is not initialized with touchdown=True in Config.
        The evaluate_SteadyState method should internally enable touchdown mode
        using toggle_touchdown() to properly invalidate cached properties.
        """
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        # Initialize system WITHOUT touchdown=True (default is False)
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=Config(),  # touchdown defaults to False
        )

        # This should not raise AttributeError: 'NoneType' object has no attribute 'l_BC'
        results: SteadyStateResult = self.evaluator.evaluate_SteadyState(system)

        # Verify results are valid
        self.assertTrue(results.converged)
        self.assertGreater(results.energy_release_rate, 0)
        self.assertGreater(results.touchdown_distance, 0)
        self.assertLess(results.touchdown_distance, results.system.scenario.L)
        max_principal_stress_norm = (
            results.maximal_stress_result.max_principal_stress_norm
        )
        max_Sxx_norm = results.maximal_stress_result.max_Sxx_norm
        self.assertGreater(max_principal_stress_norm, 0)
        self.assertGreater(max_Sxx_norm, 0)

        # Verify the original system's touchdown state was not modified
        self.assertFalse(system.config.touchdown)

    def test_find_minimum_crack_length(self):
        """Test the find_minimum_crack_length method."""
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=self.config,
        )
        crack_length, new_segments = self.evaluator.find_minimum_crack_length(system)
        self.assertGreater(crack_length, 0)
        self.assertIsInstance(new_segments, list)
        self.assertTrue(all(isinstance(s, Segment) for s in new_segments))

    def test_evaluate_SteadyState_mode_C_in_contact(self):
        """Test evaluate_SteadyState with mode='C_in_contact'."""
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=Config(touchdown=True),
        )
        results: SteadyStateResult = self.evaluator.evaluate_SteadyState(
            system, mode="C_in_contact"
        )
        self.assertTrue(results.converged)
        self.assertEqual(
            results.system.slab_touchdown.touchdown_mode,
            "C_in_contact",
            "Touchdown mode should match the mode parameter passed to evaluate_SteadyState",
        )

    def test_evaluate_SteadyState_mode_B_point_contact(self):
        """Test evaluate_SteadyState with mode='B_point_contact'."""
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=Config(touchdown=True),
        )
        results: SteadyStateResult = self.evaluator.evaluate_SteadyState(
            system, mode="B_point_contact"
        )
        self.assertTrue(results.converged)
        self.assertEqual(
            results.system.slab_touchdown.touchdown_mode,
            "B_point_contact",
            "Touchdown mode should match the mode parameter passed to evaluate_SteadyState",
        )

    def test_evaluate_SteadyState_mode_A_free_hanging(self):
        """Test evaluate_SteadyState with mode='A_free_hanging'."""
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=Config(touchdown=True),
        )
        results: SteadyStateResult = self.evaluator.evaluate_SteadyState(
            system, mode="A_free_hanging"
        )
        self.assertTrue(results.converged)
        self.assertEqual(
            results.system.slab_touchdown.touchdown_mode,
            "A_free_hanging",
            "Touchdown mode should match the mode parameter passed to evaluate_SteadyState",
        )

    def test_evaluate_SteadyState_default_mode(self):
        """Test evaluate_SteadyState with default mode (should be 'C_in_contact')."""
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        system = SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi),
            ),
            config=Config(touchdown=True),
        )
        # Call without specifying mode - should default to 'C_in_contact'
        results: SteadyStateResult = self.evaluator.evaluate_SteadyState(system)
        self.assertTrue(results.converged)
        self.assertEqual(
            results.system.slab_touchdown.touchdown_mode,
            "C_in_contact",
            "Default touchdown mode should be 'C_in_contact'",
        )


if __name__ == "__main__":
    unittest.main()
