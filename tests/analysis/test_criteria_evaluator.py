"""
This module contains tests for the CriteriaEvaluator class.
"""

# Standard library imports
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Third party imports
import numpy as np

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


class TestCriteriaEvaluator(unittest.TestCase):
    """Test suite for the CriteriaEvaluator."""

    def setUp(self):
        """Set up common objects for testing."""
        self.config = Config()
        self.criteria_config = CriteriaConfig()
        self.evaluator = CriteriaEvaluator(self.criteria_config)

        self.layers = [
            Layer(rho=170, h=100, tensile_strength_method="hybrid"),
            Layer(rho=190, h=40, tensile_strength_method="hybrid"),
            Layer(rho=230, h=130, tensile_strength_method="hybrid"),
            Layer(rho=250, h=20, tensile_strength_method="hybrid"),
            Layer(rho=210, h=70, tensile_strength_method="hybrid"),
            Layer(rho=380, h=20, tensile_strength_method="hybrid"),
            Layer(rho=280, h=100, tensile_strength_method="hybrid"),
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

        self.assertAlmostEqual(top_broken_result.slab_tensile_criterion, 1 / 2)
        self.assertAlmostEqual(top_unbroken_result.slab_tensile_criterion, 1 / 4)

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

    def test_find_crack_length_for_weight_single_rasterize(self):
        """Crack-length search rasterizes the cracked field once per weight."""
        skier_weight = 100
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
        original_rasterize = Analyzer.rasterize_solution
        rasterize_calls = []

        def counting_rasterize(self, *args, **kwargs):
            rasterize_calls.append(kwargs.get("mode", args[0] if args else None))
            return original_rasterize(self, *args, **kwargs)

        with patch.object(Analyzer, "rasterize_solution", counting_rasterize):
            crack_len, new_segments = self.evaluator.find_crack_length_for_weight(
                system, skier_weight
            )

        self.assertEqual(len(rasterize_calls), 1)
        self.assertEqual(rasterize_calls[0], "cracked")
        self.assertGreaterEqual(crack_len, 0)
        self.assertTrue(all(isinstance(s, Segment) for s in new_segments))

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
        self.assertIsNotNone(results.history)
        history = results.history
        assert history is not None
        self.assertEqual(len(history.sigma_maxs), len(history.skier_weights))
        self.assertGreater(len(history.sigma_maxs), 0)
        self.assertEqual(len(history.tau_maxs), len(history.skier_weights))
        self.assertGreater(len(history.tau_maxs), 0)

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

        self.assertEqual(mock_fm.call_count, 1)
        self.assertTrue(results.converged)
        self.assertGreater(results.critical_skier_weight, 0)

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

        self.assertTrue(results.converged)
        self.assertGreater(results.iterations, 1)
        self.assertFalse(results.pure_stress_criteria)
        uncracked_count = sum(1 for m in rasterize_modes if m == "uncracked")
        # 1x find_minimum_force + 1x main-loop iteration 1
        self.assertEqual(uncracked_count, 2)
        history = results.history
        assert history is not None
        self.assertEqual(len(history.g_deltas), results.iterations)
        self.assertEqual(len(history.sigma_maxs), results.iterations)
        # Post-iter-1 stress history reuses the iteration-1 sample
        self.assertTrue(
            all(s == history.sigma_maxs[0] for s in history.sigma_maxs[1:])
        )


    def _make_ss_system(self, *, phi=None, touchdown=True, weak_layer=None):
        """Build a system for hybrid steady-state tests."""
        segments = [
            Segment(length=self.segments_length, has_foundation=True, m=0),
            Segment(length=self.segments_length, has_foundation=True, m=0),
        ]
        return SystemModel(
            model_input=ModelInput(
                layers=self.layers,
                weak_layer=weak_layer or self.weak_layer,
                segments=segments,
                scenario_config=ScenarioConfig(phi=self.phi if phi is None else phi),
            ),
            config=Config(touchdown=touchdown),
        )

    def test_evaluate_SteadyState(self):
        """Hybrid SS returns structured tensile / ERR blocks."""
        system = self._make_ss_system()
        results: SteadyStateResult = self.evaluator.evaluate_SteadyState(system)
        self.assertTrue(results.converged)
        self.assertGreater(results.tensile.critical_cut_length, 0)
        self.assertGreater(results.err.energy_release_rate, 0)
        self.assertEqual(results.phi, self.phi)
        self.assertIn(results.tensile.cut_direction_winner, ("upslope", "downslope"))
        self.assertIn(results.err.cut_direction_winner, ("upslope", "downslope"))
        stress = results.tensile.maximal_stress_result
        self.assertIsNotNone(stress)
        self.assertGreater(stress.max_Sxx_norm, 0)

    def test_evaluate_SteadyState_without_touchdown_in_config(self):
        """Hybrid SS does not require touchdown=True on the input system."""
        system = self._make_ss_system(touchdown=False)
        results: SteadyStateResult = self.evaluator.evaluate_SteadyState(system)
        self.assertTrue(results.converged)
        self.assertGreater(results.tensile.critical_cut_length, 0)
        self.assertGreater(results.err.energy_release_rate, 0)
        self.assertFalse(system.config.touchdown)

    def test_evaluate_SteadyState_rejects_mode_kwarg(self):
        """TouchdownMode / mode= is no longer part of the public SS API."""
        system = self._make_ss_system()
        with self.assertRaises(TypeError):
            self.evaluator.evaluate_SteadyState(system, mode="B_point_contact")

    def test_steady_state_maximal_stress_structure(self):
        """Tensile-leg maximal stress has valid structure."""
        result = self.evaluator.evaluate_SteadyState(self._make_ss_system())
        maximal_stress = result.tensile.maximal_stress_result
        self.assertIsNotNone(maximal_stress)
        self.assertEqual(
            maximal_stress.principal_stress_kPa.shape,
            maximal_stress.Sxx_kPa.shape,
        )
        self.assertGreater(maximal_stress.principal_stress_kPa.size, 0)
        self.assertGreater(maximal_stress.max_Sxx_norm, 0)
        self.assertGreaterEqual(maximal_stress.slab_tensile_criterion, 0)
        self.assertLessEqual(maximal_stress.slab_tensile_criterion, 1)

    def test_steady_state_energy_release_rate_positive(self):
        """ERR-leg energy release rate is positive."""
        result = self.evaluator.evaluate_SteadyState(self._make_ss_system())
        self.assertGreater(result.err.energy_release_rate, 0)

    def test_steady_state_with_different_weak_layers(self):
        """Hybrid SS converges across weak-layer property variations."""
        weak_layers = [
            WeakLayer(rho=150, h=10, G_Ic=0.3, G_IIc=0.6, kn=50, kt=50),
            WeakLayer(rho=200, h=15, G_Ic=0.8, G_IIc=1.2, kn=150, kt=150),
            WeakLayer(rho=180, h=10, G_Ic=0.5, G_IIc=0.8, kn=100, kt=100),
        ]
        for weak_layer in weak_layers:
            with self.subTest(weak_layer=weak_layer):
                result = self.evaluator.evaluate_SteadyState(
                    self._make_ss_system(weak_layer=weak_layer)
                )
                self.assertTrue(result.converged)
                self.assertGreater(result.err.energy_release_rate, 0)
                self.assertGreater(result.tensile.critical_cut_length, 0)

    def test_steady_state_with_different_slope_angles(self):
        """Hybrid SS uses system φ (no flat-slab override)."""
        for phi in [20.0, 30.0, 40.0, 45.0]:
            with self.subTest(phi=phi):
                result = self.evaluator.evaluate_SteadyState(
                    self._make_ss_system(phi=phi)
                )
                self.assertTrue(result.converged)
                self.assertEqual(result.phi, phi)
                self.assertGreater(result.err.energy_release_rate, 0)
                self.assertGreater(result.tensile.critical_cut_length, 0)

    def test_steady_state_system_isolation(self):
        """evaluate_SteadyState does not mutate the caller's system."""
        system = self._make_ss_system()
        original_phi = system.scenario.phi
        original_L = system.scenario.L
        original_nseg = len(system.scenario.segments)
        result = self.evaluator.evaluate_SteadyState(system)
        self.assertEqual(system.scenario.phi, original_phi)
        self.assertEqual(system.scenario.L, original_L)
        self.assertEqual(len(system.scenario.segments), original_nseg)
        self.assertEqual(result.phi, original_phi)

    def test_steady_state_message_format(self):
        """Hybrid result message mentions both legs."""
        result = self.evaluator.evaluate_SteadyState(self._make_ss_system())
        self.assertIsInstance(result.message, str)
        self.assertIn("tensile[", result.message)
        self.assertIn("err[", result.message)

    def test_steady_state_normalized_stresses_consistency(self):
        """Max normalized stresses match array maxima on the tensile leg."""
        result = self.evaluator.evaluate_SteadyState(self._make_ss_system())
        maximal_stress = result.tensile.maximal_stress_result
        self.assertIsNotNone(maximal_stress)
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

    def test_steady_state_preserves_phi(self):
        """Hybrid SS must not force φ→0."""
        original_phi = 35.0
        system = self._make_ss_system(phi=original_phi)
        result = self.evaluator.evaluate_SteadyState(system)
        self.assertEqual(result.phi, original_phi)
        self.assertEqual(system.scenario.phi, original_phi)
        if result.tensile.system is not None:
            self.assertEqual(result.tensile.system.scenario.phi, original_phi)

    def test_steady_state_critical_cut_length_bounds(self):
        """Tensile critical cut length is within the search window."""
        result = self.evaluator.evaluate_SteadyState(self._make_ss_system())
        self.assertGreater(result.tensile.critical_cut_length, 0)
        self.assertLessEqual(result.tensile.critical_cut_length, 5000.0)


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
