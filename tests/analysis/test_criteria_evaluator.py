# Standard library imports
import unittest

# Third party imports
import numpy as np

# weac imports
from weac.analysis.criteria_evaluator import (
    CoupledCriterionResult,
    CriteriaEvaluator,
    FindMinimumForceResult,
    SSERRResult,
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
        self.assertAlmostEqual(g_delta, 0.2455609957, places=5)

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

    def test_find_new_anticrack_length(self):
        """Test the find_new_anticrack_length method."""
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
                scenario_config=ScenarioConfig(phi=self.phi, crack_length=0),
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
        self.assertAlmostEqual(g_delta, 0, places=4)

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

    def test_evaluate_SSERR(self):
        """Test the evaluate_SSERR method."""
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
        results: SSERRResult = self.evaluator.evaluate_SSERR(system)
        self.assertTrue(results.converged)
        self.assertGreater(results.SSERR, 0)
        self.assertGreater(results.touchdown_distance, 0)
        self.assertLess(results.touchdown_distance, system.scenario.L)

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


if __name__ == "__main__":
    unittest.main()
