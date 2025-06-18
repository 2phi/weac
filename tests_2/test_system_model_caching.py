import unittest
from unittest.mock import patch

from weac_2.components import (
    Config,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac_2.core.system_model import SystemModel


class TestSystemModelCaching(unittest.TestCase):
    """Test caching mechanisms in the SystemModel."""

    def setUp(self):
        """Set up common components for tests."""
        self.config = Config()
        self.layers = [Layer(rho=200, h=500)]
        self.weak_layer = WeakLayer(rho=150, h=10)
        self.segments = [Segment(length=10000, has_foundation=True, m=0)]
        self.scenario_config = ScenarioConfig(phi=30, system_type="skiers")

    @patch("weac_2.core.eigensystem.Eigensystem.calc_eigensystem")
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
        system.update_slab_layers(new_layers=[Layer(rho=250, h=600)])

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
        system.update_weak_layer(rho=160, h=12)

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
        system.update_scenario(phi=45.0)

        eigensystem_after = system.eigensystem
        constants_after = system.unknown_constants

        self.assertIs(eigensystem_before, eigensystem_after)
        self.assertIsNot(constants_before, constants_after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
