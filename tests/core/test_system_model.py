import unittest
from unittest.mock import patch, MagicMock

from weac.components import (
    Config,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac.core.system_model import SystemModel
import numpy as np


class TestSystemModelCaching(unittest.TestCase):
    """Test caching mechanisms in the SystemModel."""

    def setUp(self):
        """Set up common components for tests."""
        self.config = Config()
        self.layers = [Layer(rho=200, h=500)]
        self.weak_layer = WeakLayer(rho=150, h=10)
        self.segments = [Segment(length=10000, has_foundation=True, m=0)]
        self.scenario_config = ScenarioConfig(phi=30, system_type="skiers")

    @patch("weac.core.eigensystem.Eigensystem.calc_eigensystem")
    def test_eigensystem_calculation_called_once(self, mock_calc):
        """Test that eigensystem calculation is called only once when cached."""
        model_input = ModelInput(
            layers=self.layers,
            weak_layer=self.weak_layer,
            segments=self.segments,
            scenario_config=self.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=self.config)

        # Access eigensystem multiple times
        _ = system.eigensystem
        _ = system.eigensystem
        _ = system.eigensystem

        # calc_eigensystem should only be called once due to caching
        self.assertEqual(
            mock_calc.call_count,
            1,
            "Eigensystem calculation should only be called once",
        )

    def test_eigensystem_caching(self):
        """Test that eigensystem is cached and reused."""
        model_input = ModelInput(
            layers=self.layers,
            weak_layer=self.weak_layer,
            segments=self.segments,
            scenario_config=self.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=self.config)
        eigensystem1 = system.eigensystem
        eigensystem2 = system.eigensystem
        self.assertIs(
            eigensystem1, eigensystem2, "Cached eigensystem should be the same object"
        )

    def test_unknown_constants_caching(self):
        """Test that unknown constants are cached and reused."""
        model_input = ModelInput(
            layers=self.layers,
            weak_layer=self.weak_layer,
            segments=self.segments,
            scenario_config=self.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=self.config)
        constants1 = system.unknown_constants
        constants2 = system.unknown_constants
        self.assertIs(
            constants1, constants2, "Cached constants should be the same object"
        )

    def test_slab_update_invalidates_all_caches(self):
        """Test that slab updates invalidate both eigensystem and unknown constants."""
        model_input = ModelInput(
            layers=self.layers,
            weak_layer=self.weak_layer,
            segments=self.segments,
            scenario_config=self.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=self.config)
        eigensystem_before = system.eigensystem
        constants_before = system.unknown_constants

        # Update the slab layers
        system.update_layers(new_layers=[Layer(rho=250, h=600)])

        eigensystem_after = system.eigensystem
        constants_after = system.unknown_constants

        self.assertIsNot(eigensystem_before, eigensystem_after)
        self.assertIsNot(constants_before, constants_after)

    def test_weak_layer_update_invalidates_all_caches(self):
        """Test that weak layer updates invalidate both caches."""
        model_input = ModelInput(
            layers=self.layers,
            weak_layer=self.weak_layer,
            segments=self.segments,
            scenario_config=self.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=self.config)
        eigensystem_before = system.eigensystem
        constants_before = system.unknown_constants

        # Update the weak layer
        system.update_weak_layer(WeakLayer(rho=160, h=12))

        eigensystem_after = system.eigensystem
        constants_after = system.unknown_constants

        self.assertIsNot(eigensystem_before, eigensystem_after)
        self.assertIsNot(constants_before, constants_after)

    def test_scenario_update_invalidates_constants_only(self):
        """Test that scenario updates only invalidate unknown constants, not eigensystem."""
        model_input = ModelInput(
            layers=self.layers,
            weak_layer=self.weak_layer,
            segments=self.segments,
            scenario_config=self.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=self.config)
        eigensystem_before = system.eigensystem
        constants_before = system.unknown_constants

        # Update the scenario
        scenario_config = system.scenario.scenario_config
        scenario_config.phi = 45.0
        system.update_scenario(scenario_config=scenario_config)

        eigensystem_after = system.eigensystem
        constants_after = system.unknown_constants

        self.assertIs(eigensystem_before, eigensystem_after)
        self.assertIsNot(constants_before, constants_after)


class TestSystemModelBehavior(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.layers = [Layer(rho=200, h=500)]
        self.weak_layer = WeakLayer(rho=150, h=10)
        self.segments = [
            Segment(length=10000, has_foundation=True, m=80),
            Segment(length=4000, has_foundation=False, m=0),
        ]
        self.scenario_config = ScenarioConfig(
            phi=10.0, system_type="skiers", cut_length=3000.0
        )

    def _build_model(
        self, touchdown: bool = False, system_type: str = "skiers"
    ) -> SystemModel:
        config = Config(touchdown=touchdown)
        sc = ScenarioConfig(phi=10.0, system_type=system_type, cut_length=3000.0)
        model_input = ModelInput(
            layers=self.layers,
            weak_layer=self.weak_layer,
            segments=self.segments,
            scenario_config=sc,
        )
        return SystemModel(model_input=model_input, config=config)

    @patch("weac.core.system_model.SlabTouchdown")
    def test_touchdown_updates_segments_for_pst_minus(self, mock_td):
        mock_inst = MagicMock()
        mock_inst.touchdown_distance = 1234.0
        mock_inst.touchdown_mode = "B_point_contact"
        mock_inst.collapsed_weak_layer_kR = 42.0
        mock_td.return_value = mock_inst

        system = self._build_model(touchdown=True, system_type="pst-")
        _ = system.slab_touchdown  # trigger

        self.assertEqual(system.scenario.segments[-1].length, 1234.0)

    @patch("weac.core.system_model.SlabTouchdown")
    def test_touchdown_updates_segments_for_minus_pst(self, mock_td):
        mock_inst = MagicMock()
        mock_inst.touchdown_distance = 2222.0
        mock_inst.touchdown_mode = "B_point_contact"
        mock_inst.collapsed_weak_layer_kR = 11.0
        mock_td.return_value = mock_inst

        system = self._build_model(touchdown=True, system_type="-pst")
        _ = system.slab_touchdown  # trigger

        self.assertEqual(system.scenario.segments[0].length, 2222.0)

    @patch("weac.core.system_model.UnknownConstantsSolver.solve_for_unknown_constants")
    @patch("weac.core.system_model.SlabTouchdown")
    def test_unknown_constants_uses_touchdown_params_when_enabled(
        self, mock_td, mock_solve
    ):
        mock_inst = MagicMock()
        mock_inst.touchdown_distance = 1500.0
        mock_inst.touchdown_mode = "C_in_contact"
        mock_inst.collapsed_weak_layer_kR = 7.5
        mock_td.return_value = mock_inst

        def solver_side_effect(
            scenario,
            eigensystem,
            system_type,
            touchdown_distance,
            touchdown_mode,
            collapsed_weak_layer_kR,
        ):
            n = len(scenario.segments)
            return np.zeros((6, n))

        mock_solve.side_effect = solver_side_effect

        system = self._build_model(touchdown=True, system_type="pst-")
        _ = system.unknown_constants

        mock_solve.assert_called_once()
        _, kwargs = mock_solve.call_args
        self.assertEqual(kwargs["touchdown_distance"], 1500.0)
        self.assertEqual(kwargs["touchdown_mode"], "C_in_contact")
        self.assertEqual(kwargs["collapsed_weak_layer_kR"], 7.5)

    @patch("weac.core.system_model.UnknownConstantsSolver.solve_for_unknown_constants")
    def test_unknown_constants_without_touchdown_passes_none(self, mock_solve):
        def solver_side_effect(
            scenario,
            eigensystem,
            system_type,
            touchdown_distance,
            touchdown_mode,
            collapsed_weak_layer_kR,
        ):
            n = len(scenario.segments)
            self.assertIsNone(touchdown_distance)
            self.assertIsNone(touchdown_mode)
            self.assertIsNone(collapsed_weak_layer_kR)
            return np.zeros((6, n))

        mock_solve.side_effect = solver_side_effect

        system = self._build_model(touchdown=False, system_type="skiers")
        _ = system.unknown_constants
        mock_solve.assert_called_once()

    @patch("weac.core.system_model.UnknownConstantsSolver.solve_for_unknown_constants")
    def test_uncracked_unknown_constants_sets_all_foundation(self, mock_solve):
        captured_scenarios = []

        def solver_side_effect(
            scenario,
            eigensystem,
            system_type,
            touchdown_distance,
            touchdown_mode,
            collapsed_weak_layer_kR,
        ):
            captured_scenarios.append(scenario)
            n = len(scenario.segments)
            return np.zeros((6, n))

        mock_solve.side_effect = solver_side_effect

        system = self._build_model(touchdown=False, system_type="skiers")
        _ = system.uncracked_unknown_constants

        self.assertIsNotNone(system.uncracked_scenario)
        self.assertTrue(
            all(seg.has_foundation for seg in system.uncracked_scenario.segments)
        )
        self.assertGreater(len(captured_scenarios), 0)
        self.assertTrue(
            all(seg.has_foundation for seg in captured_scenarios[-1].segments)
        )

    @patch("weac.core.system_model.SlabTouchdown")
    @patch("weac.core.system_model.UnknownConstantsSolver.solve_for_unknown_constants")
    def test_update_scenario_invalidates_touchdown_and_constants(
        self, mock_solve, mock_td
    ):
        mock_inst = MagicMock()
        mock_inst.touchdown_distance = 1800.0
        mock_inst.touchdown_mode = "B_point_contact"
        mock_inst.collapsed_weak_layer_kR = 3.14
        mock_td.return_value = mock_inst

        def solver_side_effect(
            scenario,
            eigensystem,
            system_type,
            touchdown_distance,
            touchdown_mode,
            collapsed_weak_layer_kR,
        ):
            n = len(scenario.segments)
            return np.zeros((6, n))

        mock_solve.side_effect = solver_side_effect

        system = self._build_model(touchdown=True, system_type="pst-")
        _ = system.slab_touchdown
        first_td_calls = mock_td.call_count
        _ = system.unknown_constants

        # Update scenario (e.g., change phi)
        new_cfg = system.scenario.scenario_config
        new_cfg.phi = 20.0
        system.update_scenario(scenario_config=new_cfg)

        # Access again to trigger recompute
        _ = system.slab_touchdown
        _ = system.unknown_constants

        self.assertGreater(mock_td.call_count, first_td_calls)
        self.assertGreaterEqual(mock_solve.call_count, 2)

    @patch("weac.core.system_model.UnknownConstantsSolver.solve_for_unknown_constants")
    def test_toggle_touchdown_switches_solver_arguments(self, mock_solve):
        calls = []

        def solver_side_effect(
            scenario,
            eigensystem,
            system_type,
            touchdown_distance,
            touchdown_mode,
            collapsed_weak_layer_kR,
        ):
            calls.append((touchdown_distance, touchdown_mode, collapsed_weak_layer_kR))
            n = len(scenario.segments)
            return np.zeros((6, n))

        mock_solve.side_effect = solver_side_effect

        system = self._build_model(touchdown=False, system_type="skiers")
        _ = system.unknown_constants  # first call without TD

        with patch("weac.core.system_model.SlabTouchdown") as mock_td:
            mock_inst = MagicMock()
            mock_inst.touchdown_distance = 900.0
            mock_inst.touchdown_mode = "A_free_hanging"
            mock_inst.collapsed_weak_layer_kR = None
            mock_td.return_value = mock_inst

            system.toggle_touchdown(True)
            _ = system.unknown_constants  # second call with TD

        self.assertEqual(len(calls), 2)
        # First without touchdown
        self.assertEqual(calls[0], (None, None, None))
        # Second with touchdown
        self.assertEqual(calls[1], (900.0, "A_free_hanging", None))

    def test_z_function_scalar_and_array(self):
        system = self._build_model(touchdown=False, system_type="skiers")

        # Patch eigensystem methods on the instance to simple deterministic outputs
        I6 = np.eye(6)

        def fake_zh(x, length, has_foundation):
            return 2.0 * I6

        def fake_zp(x, phi, has_foundation, qs):
            return np.ones((6, 1))

        with (
            patch.object(system.eigensystem, "zh", side_effect=fake_zh),
            patch.object(system.eigensystem, "zp", side_effect=fake_zp),
        ):
            C = np.eye(6)
            # Scalar x
            z_scalar = system.z(
                x=100.0, C=C, length=1000.0, phi=10.0, has_foundation=True, qs=0.0
            )
            self.assertEqual(z_scalar.shape, (6, 6))
            expected = 2.0 * I6 + np.ones((6, 1)) @ np.ones(
                (1, 6)
            )  # Broadcast to (6, 6)
            np.testing.assert_allclose(z_scalar, expected)
            # Array x of length 3 -> concatenation along axis=1
            z_array = system.z(
                x=[0.0, 50.0, 100.0],
                C=C,
                length=1000.0,
                phi=10.0,
                has_foundation=True,
                qs=0.0,
            )
            self.assertEqual(z_array.shape, (6, 18))


if __name__ == "__main__":
    unittest.main(verbosity=2)
