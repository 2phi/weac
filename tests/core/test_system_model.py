"""
This module contains tests for the SystemModel class.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from weac.components import (
    Config,
    Layer,
    ModelInput,
    ScenarioConfig,
    SystemType,
    Segment,
    WeakLayer,
)
from weac.core.system_model import SystemModel


@pytest.fixture
def caching_parts():
    """Shared components for SystemModel caching tests."""
    return SimpleNamespace(
        config=Config(),
        layers=[Layer(rho=200, h=500)],
        weak_layer=WeakLayer(rho=150, h=10),
        segments=[Segment(length=10000, has_foundation=True, m=0)],
        scenario_config=ScenarioConfig(phi=30, system_type="skiers"),
    )


@pytest.fixture
def behavior_parts():
    """Shared components for SystemModel behavior tests."""
    return SimpleNamespace(
        config=Config(),
        layers=[Layer(rho=200, h=500)],
        weak_layer=WeakLayer(rho=150, h=10),
        segments=[
            Segment(length=10000, has_foundation=True, m=80),
            Segment(length=4000, has_foundation=False, m=0),
        ],
        scenario_config=ScenarioConfig(
            phi=10.0, system_type="skiers", cut_length=3000.0
        ),
    )


def _build_model(
    parts,
    touchdown: bool = False,
    system_type: SystemType = "skiers",
) -> SystemModel:
    """Build a SystemModel from shared behavior parts."""
    config = Config(touchdown=touchdown)
    sc = ScenarioConfig(phi=10.0, system_type=system_type, cut_length=3000.0)
    model_input = ModelInput(
        layers=parts.layers,
        weak_layer=parts.weak_layer,
        segments=parts.segments,
        scenario_config=sc,
    )
    return SystemModel(model_input=model_input, config=config)


class TestSystemModelCaching:
    """Test caching mechanisms in the SystemModel."""

    @patch("weac.core.eigensystem.Eigensystem.calc_eigensystem")
    def test_eigensystem_calculation_called_once(self, mock_calc, caching_parts):
        """Test that eigensystem calculation is called only once when cached."""
        model_input = ModelInput(
            layers=caching_parts.layers,
            weak_layer=caching_parts.weak_layer,
            segments=caching_parts.segments,
            scenario_config=caching_parts.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=caching_parts.config)

        # Access eigensystem multiple times
        _ = system.eigensystem
        _ = system.eigensystem
        _ = system.eigensystem

        # calc_eigensystem should only be called once due to caching
        assert mock_calc.call_count == 1, (
            "Eigensystem calculation should only be called once"
        )

    def test_eigensystem_caching(self, caching_parts):
        """Test that eigensystem is cached and reused."""
        model_input = ModelInput(
            layers=caching_parts.layers,
            weak_layer=caching_parts.weak_layer,
            segments=caching_parts.segments,
            scenario_config=caching_parts.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=caching_parts.config)
        eigensystem1 = system.eigensystem
        eigensystem2 = system.eigensystem
        assert eigensystem1 is eigensystem2, (
            "Cached eigensystem should be the same object"
        )

    def test_unknown_constants_caching(self, caching_parts):
        """Test that unknown constants are cached and reused."""
        model_input = ModelInput(
            layers=caching_parts.layers,
            weak_layer=caching_parts.weak_layer,
            segments=caching_parts.segments,
            scenario_config=caching_parts.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=caching_parts.config)
        constants1 = system.unknown_constants
        constants2 = system.unknown_constants
        assert constants1 is constants2, "Cached constants should be the same object"

    def test_slab_update_invalidates_all_caches(self, caching_parts):
        """Test that slab updates invalidate both eigensystem and unknown constants."""
        model_input = ModelInput(
            layers=caching_parts.layers,
            weak_layer=caching_parts.weak_layer,
            segments=caching_parts.segments,
            scenario_config=caching_parts.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=caching_parts.config)
        eigensystem_before = system.eigensystem
        constants_before = system.unknown_constants

        # Update the slab layers
        system.update_layers(new_layers=[Layer(rho=250, h=600)])

        eigensystem_after = system.eigensystem
        constants_after = system.unknown_constants

        assert eigensystem_before is not eigensystem_after
        assert constants_before is not constants_after

    def test_weak_layer_update_invalidates_all_caches(self, caching_parts):
        """Test that weak layer updates invalidate both caches."""
        model_input = ModelInput(
            layers=caching_parts.layers,
            weak_layer=caching_parts.weak_layer,
            segments=caching_parts.segments,
            scenario_config=caching_parts.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=caching_parts.config)
        eigensystem_before = system.eigensystem
        constants_before = system.unknown_constants

        # Update the weak layer
        system.update_weak_layer(WeakLayer(rho=160, h=12))

        eigensystem_after = system.eigensystem
        constants_after = system.unknown_constants

        assert eigensystem_before is not eigensystem_after
        assert constants_before is not constants_after

    def test_scenario_update_invalidates_constants_only(self, caching_parts):
        """Test that scenario updates only invalidate unknown constants, not eigensystem."""
        model_input = ModelInput(
            layers=caching_parts.layers,
            weak_layer=caching_parts.weak_layer,
            segments=caching_parts.segments,
            scenario_config=caching_parts.scenario_config,
        )
        system = SystemModel(model_input=model_input, config=caching_parts.config)
        eigensystem_before = system.eigensystem
        constants_before = system.unknown_constants

        # Update the scenario
        new_cfg = system.scenario.scenario_config.model_copy()
        new_cfg.phi = 45.0
        system.update_scenario(scenario_config=new_cfg)

        eigensystem_after = system.eigensystem
        constants_after = system.unknown_constants

        assert eigensystem_before is eigensystem_after
        assert constants_before is not constants_after


class TestSystemModelBehavior:
    """Test the behavior of the SystemModel class."""

    @patch("weac.core.system_model.SlabTouchdown")
    def test_touchdown_updates_segments_for_pst_minus(self, mock_td, behavior_parts):
        """Test that touchdown updates segments for pst-."""
        mock_inst = MagicMock()
        mock_inst.touchdown_distance = 1234.0
        mock_inst.touchdown_mode = "B_point_contact"
        mock_inst.collapsed_weak_layer_kR = 42.0
        mock_td.return_value = mock_inst

        system = _build_model(behavior_parts, touchdown=True, system_type="pst-")
        _ = system.slab_touchdown  # trigger

        assert system.scenario.segments[-1].length == 1234.0

    @patch("weac.core.system_model.SlabTouchdown")
    def test_touchdown_updates_segments_for_minus_pst(self, mock_td, behavior_parts):
        """Test that touchdown updates segments for -pst."""
        mock_inst = MagicMock()
        mock_inst.touchdown_distance = 2222.0
        mock_inst.touchdown_mode = "B_point_contact"
        mock_inst.collapsed_weak_layer_kR = 11.0
        mock_td.return_value = mock_inst

        system = _build_model(behavior_parts, touchdown=True, system_type="-pst")
        _ = system.slab_touchdown  # trigger

        assert system.scenario.segments[0].length == 2222.0

    @patch("weac.core.system_model.UnknownConstantsSolver.solve_for_unknown_constants")
    @patch("weac.core.system_model.SlabTouchdown")
    def test_unknown_constants_uses_touchdown_params_when_enabled(
        self, mock_td, mock_solve, behavior_parts
    ):
        """Test that unknown constants uses touchdown params when enabled."""
        mock_inst = MagicMock()
        mock_inst.touchdown_distance = 1500.0
        mock_inst.touchdown_mode = "C_in_contact"
        mock_inst.collapsed_weak_layer_kR = 7.5
        mock_td.return_value = mock_inst

        def solver_side_effect(
            scenario,
            eigensystem,  # pylint: disable=unused-argument
            system_type,  # pylint: disable=unused-argument
            touchdown_distance,  # pylint: disable=unused-argument
            touchdown_mode,  # pylint: disable=unused-argument
            collapsed_weak_layer_kR,  # pylint: disable=unused-argument
        ):
            n = len(scenario.segments)
            return np.zeros((6, n))

        mock_solve.side_effect = solver_side_effect

        system = _build_model(behavior_parts, touchdown=True, system_type="pst-")
        _ = system.unknown_constants

        mock_solve.assert_called_once()
        _, kwargs = mock_solve.call_args
        assert kwargs["touchdown_distance"] == 1500.0
        assert kwargs["touchdown_mode"] == "C_in_contact"
        assert kwargs["collapsed_weak_layer_kR"] == 7.5

    @patch("weac.core.system_model.UnknownConstantsSolver.solve_for_unknown_constants")
    def test_unknown_constants_without_touchdown_passes_none(
        self, mock_solve, behavior_parts
    ):
        """Test that unknown constants without touchdown passes None."""

        def solver_side_effect(
            scenario,
            eigensystem,  # pylint: disable=unused-argument
            system_type,  # pylint: disable=unused-argument
            touchdown_distance,
            touchdown_mode,
            collapsed_weak_layer_kR,
        ):
            n = len(scenario.segments)
            assert touchdown_distance is None
            assert touchdown_mode is None
            assert collapsed_weak_layer_kR is None
            return np.zeros((6, n))

        mock_solve.side_effect = solver_side_effect

        system = _build_model(behavior_parts, touchdown=False, system_type="skiers")
        _ = system.unknown_constants
        mock_solve.assert_called_once()

    @patch("weac.core.system_model.UnknownConstantsSolver.solve_for_unknown_constants")
    def test_uncracked_unknown_constants_sets_all_foundation(
        self, mock_solve, behavior_parts
    ):
        """Test that uncracked_unknown_constants sets all foundation."""
        captured_scenarios = []

        def solver_side_effect(
            scenario,
            eigensystem,  # pylint: disable=unused-argument
            system_type,  # pylint: disable=unused-argument
            touchdown_distance,  # pylint: disable=unused-argument
            touchdown_mode,  # pylint: disable=unused-argument
            collapsed_weak_layer_kR,  # pylint: disable=unused-argument
        ):
            captured_scenarios.append(scenario)
            n = len(scenario.segments)
            return np.zeros((6, n))

        mock_solve.side_effect = solver_side_effect

        system = _build_model(behavior_parts, touchdown=False, system_type="skiers")
        _ = system.uncracked_unknown_constants

        assert len(captured_scenarios) > 0
        assert all(seg.has_foundation for seg in captured_scenarios[-1].segments)

    @patch("weac.core.system_model.SlabTouchdown")
    @patch("weac.core.system_model.UnknownConstantsSolver.solve_for_unknown_constants")
    def test_update_scenario_invalidates_touchdown_and_constants(
        self, mock_solve, mock_td, behavior_parts
    ):
        """Test that update_scenario invalidates touchdown and constants."""
        mock_inst = MagicMock()
        mock_inst.touchdown_distance = 1800.0
        mock_inst.touchdown_mode = "B_point_contact"
        mock_inst.collapsed_weak_layer_kR = 3.14
        mock_td.return_value = mock_inst

        def solver_side_effect(
            scenario,
            eigensystem,  # pylint: disable=unused-argument
            system_type,  # pylint: disable=unused-argument
            touchdown_distance,  # pylint: disable=unused-argument
            touchdown_mode,  # pylint: disable=unused-argument
            collapsed_weak_layer_kR,  # pylint: disable=unused-argument
        ):
            n = len(scenario.segments)
            return np.zeros((6, n))

        mock_solve.side_effect = solver_side_effect

        system = _build_model(behavior_parts, touchdown=True, system_type="pst-")
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

        assert mock_td.call_count > first_td_calls
        assert mock_solve.call_count >= 2

    @patch("weac.core.system_model.UnknownConstantsSolver.solve_for_unknown_constants")
    def test_toggle_touchdown_switches_solver_arguments(
        self, mock_solve, behavior_parts
    ):
        """Test that toggle_touchdown switches the solver arguments."""
        calls = []

        def solver_side_effect(
            scenario,
            eigensystem,  # pylint: disable=unused-argument
            system_type,  # pylint: disable=unused-argument
            touchdown_distance,  # pylint: disable=unused-argument
            touchdown_mode,  # pylint: disable=unused-argument
            collapsed_weak_layer_kR,  # pylint: disable=unused-argument
        ):
            calls.append((touchdown_distance, touchdown_mode, collapsed_weak_layer_kR))
            n = len(scenario.segments)
            return np.zeros((6, n))

        mock_solve.side_effect = solver_side_effect

        system = _build_model(behavior_parts, touchdown=False, system_type="skiers")
        _ = system.unknown_constants  # first call without TD

        with patch("weac.core.system_model.SlabTouchdown") as mock_td:
            mock_inst = MagicMock()
            mock_inst.touchdown_distance = 900.0
            mock_inst.touchdown_mode = "A_free_hanging"
            mock_inst.collapsed_weak_layer_kR = None
            mock_td.return_value = mock_inst

            system.toggle_touchdown(True)
            _ = system.unknown_constants  # second call with TD

        assert len(calls) == 2
        # First without touchdown
        assert calls[0] == (None, None, None)
        # Second with touchdown
        assert calls[1] == (900.0, "A_free_hanging", None)

    def test_z_function_scalar_and_array(self, behavior_parts):
        """Test the z function with scalar and array inputs."""
        system = _build_model(behavior_parts, touchdown=False, system_type="skiers")

        # Patch eigensystem methods on the instance to simple deterministic outputs
        I6 = np.eye(6)

        def fake_zh(x, length, has_foundation):  # pylint: disable=unused-argument
            return 2.0 * I6

        def fake_zp(x, phi, has_foundation, qs):  # pylint: disable=unused-argument
            return np.ones((6, 1))

        with (
            patch.object(system.eigensystem, "zh", side_effect=fake_zh),
            patch.object(system.eigensystem, "zp", side_effect=fake_zp),
        ):
            C = np.eye(6)
            # Scalar x
            z_scalar = system.z(
                x=100.0,
                C=C,
                length=1000.0,
                phi=10.0,
                theta=0.0,
                has_foundation=True,
                qs=0.0,
            )
            assert z_scalar.shape == (6, 6)
            expected = 2.0 * I6 + np.ones((6, 1)) @ np.ones(
                (1, 6)
            )  # Broadcast to (6, 6)
            np.testing.assert_allclose(z_scalar, expected)
            # Array x of length 3 -> concatenation along axis=1
            x = np.array([0.0, 50.0, 100.0])
            z_array = system.z(
                x=x,
                C=C,
                length=1000.0,
                phi=10.0,
                theta=0.0,
                has_foundation=True,
                qs=0.0,
            )
            expected_cols = z_scalar.shape[1] * len(x)
            assert z_array.shape == (6, expected_cols)
