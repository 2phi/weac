"""
Unit tests for configuration components.

Tests Config, ScenarioConfig, CriteriaConfig, Segment, and ModelInput validation.
"""

import json
import unittest

from pydantic import ValidationError

from weac.components import (
    Config,
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)


class TestConfig(unittest.TestCase):
    """Test the Config class for runtime configuration."""

    def test_config_default_creation(self):
        """Test creating Config with default values."""
        config = Config()

        # Check default values
        self.assertEqual(config.touchdown, False)


class TestScenarioConfig(unittest.TestCase):
    """Test the ScenarioConfig class."""

    def test_scenario_config_defaults(self):
        """Test ScenarioConfig with default values."""
        scenario = ScenarioConfig()

        self.assertEqual(scenario.phi, 0)
        self.assertEqual(scenario.system_type, "skiers")
        self.assertEqual(scenario.cut_length, 0.0)
        self.assertEqual(scenario.stiffness_ratio, 1000)
        self.assertEqual(scenario.surface_load, 0.0)

    def test_scenario_config_custom_values(self):
        """Test ScenarioConfig with custom values."""
        scenario = ScenarioConfig(
            phi=30.0,
            system_type="skier",
            cut_length=150.0,
            stiffness_ratio=500.0,
            surface_load=10.0,
        )

        self.assertEqual(scenario.phi, 30.0)
        self.assertEqual(scenario.system_type, "skier")
        self.assertEqual(scenario.cut_length, 150.0)
        self.assertEqual(scenario.stiffness_ratio, 500.0)
        self.assertEqual(scenario.surface_load, 10.0)

    def test_scenario_config_validation(self):
        """Test ScenarioConfig validation."""
        # Negative crack length
        with self.assertRaises(ValidationError):
            ScenarioConfig(cut_length=-10.0)

        # Invalid stiffness ratio (<= 0)
        with self.assertRaises(ValidationError):
            ScenarioConfig(stiffness_ratio=0.0)

        # Negative surface load
        with self.assertRaises(ValidationError):
            ScenarioConfig(surface_load=-5.0)

        # Invalid system type
        with self.assertRaises(ValidationError):
            ScenarioConfig(system_type="invalid_system")


class TestCriteriaConfig(unittest.TestCase):
    """Test the CriteriaConfig class."""

    def test_criteria_config_defaults(self):
        """Test CriteriaConfig with default values."""
        criteria = CriteriaConfig()

        self.assertEqual(criteria.fn, 2.0)
        self.assertEqual(criteria.fm, 2.0)
        self.assertEqual(criteria.gn, 5.0)
        self.assertEqual(criteria.gm, 1 / 0.45)

    def test_criteria_config_custom_values(self):
        """Test CriteriaConfig with custom values."""
        criteria = CriteriaConfig(fn=1.5, fm=2.0, gn=0.8, gm=1.2)

        self.assertEqual(criteria.fn, 1.5)
        self.assertEqual(criteria.fm, 2.0)
        self.assertEqual(criteria.gn, 0.8)
        self.assertEqual(criteria.gm, 1.2)

    def test_criteria_config_validation(self):
        """Test CriteriaConfig validation."""
        # All parameters must be positive
        with self.assertRaises(ValidationError):
            CriteriaConfig(fn=0.0)

        with self.assertRaises(ValidationError):
            CriteriaConfig(fm=-0.5)

        with self.assertRaises(ValidationError):
            CriteriaConfig(gn=-1.0)

        with self.assertRaises(ValidationError):
            CriteriaConfig(gm=0.0)


class TestSegment(unittest.TestCase):
    """Test the Segment class."""

    def test_segment_creation(self):
        """Test creating segments with various parameters."""
        # Basic segment
        seg1 = Segment(length=1000.0, has_foundation=True, m=0.0)
        self.assertEqual(seg1.length, 1000.0)
        self.assertEqual(seg1.has_foundation, True)
        self.assertEqual(seg1.m, 0.0)

        # Segment with skier load
        seg2 = Segment(length=2000.0, has_foundation=False, m=75.0)
        self.assertEqual(seg2.length, 2000.0)
        self.assertEqual(seg2.has_foundation, False)
        self.assertEqual(seg2.m, 75.0)

    def test_segment_default_mass(self):
        """Test that segment mass defaults to 0."""
        seg = Segment(length=1500.0, has_foundation=True)
        self.assertEqual(seg.m, 0.0)

    def test_segment_validation(self):
        """Test segment validation."""
        # Negative length
        with self.assertRaises(ValidationError):
            Segment(length=-100.0, has_foundation=True)

        # Negative mass
        with self.assertRaises(ValidationError):
            Segment(length=1000.0, has_foundation=True, m=-10.0)


class TestModelInput(unittest.TestCase):
    """Test the ModelInput class for complete model validation."""

    def setUp(self):
        """Set up common test data."""
        self.scenario_config = ScenarioConfig(phi=25, system_type="skier")
        self.weak_layer = WeakLayer(rho=50, h=30, E=0.25, G_Ic=1)
        self.layers = [Layer(rho=200, h=100), Layer(rho=300, h=150)]
        self.segments = [
            Segment(length=3000, has_foundation=True, m=70),
            Segment(length=4000, has_foundation=True, m=0),
        ]

    def test_model_input_complete(self):
        """Test creating complete ModelInput."""
        model = ModelInput(
            scenario_config=self.scenario_config,
            weak_layer=self.weak_layer,
            layers=self.layers,
            segments=self.segments,
        )

        self.assertEqual(model.scenario_config, self.scenario_config)
        self.assertEqual(model.weak_layer, self.weak_layer)
        self.assertEqual(model.layers, self.layers)
        self.assertEqual(model.segments, self.segments)

    def test_model_input_empty_collections(self):
        """Test validation with empty layers or segments."""
        # Empty layers list
        with self.assertRaises(ValidationError):
            ModelInput(
                scenario_config=self.scenario_config,
                weak_layer=self.weak_layer,
                layers=[],
                segments=self.segments,
            )

        # Empty segments list
        with self.assertRaises(ValidationError):
            ModelInput(
                scenario_config=self.scenario_config,
                weak_layer=self.weak_layer,
                layers=self.layers,
                segments=[],
            )

    def test_model_input_json_serialization(self):
        """Test JSON serialization and schema generation."""
        model = ModelInput(
            scenario_config=self.scenario_config,
            weak_layer=self.weak_layer,
            layers=self.layers,
            segments=self.segments,
        )

        # Test JSON serialization
        json_str = model.model_dump_json()
        self.assertIsInstance(json_str, str)

        # Test that it can be parsed back
        parsed_data = json.loads(json_str)
        self.assertIsInstance(parsed_data, dict)

        # Test schema generation
        schema = ModelInput.model_json_schema()
        self.assertIsInstance(schema, dict)
        self.assertIn("properties", schema)
        self.assertIn("scenario_config", schema["properties"])
        self.assertIn("weak_layer", schema["properties"])
        self.assertIn("layers", schema["properties"])
        self.assertIn("segments", schema["properties"])


class TestModelInputPhysicalConsistency(unittest.TestCase):
    """Test physical consistency checks for ModelInput."""

    def test_layer_ordering_makes_sense(self):
        """Test that layer ordering is physically reasonable."""
        # This is more of a documentation test - the model doesn't enforce
        # physical layer ordering, but we can test that our test data makes sense
        layers = [
            Layer(rho=150, h=50),  # Light surface layer
            Layer(rho=200, h=100),  # Medium density
            Layer(rho=350, h=75),  # Denser bottom layer
        ]

        weak_layer = WeakLayer(rho=80, h=20)  # Weak layer should be less dense

        # Check that weak layer is less dense than slab layers
        for layer in layers:
            self.assertLess(
                weak_layer.rho,
                layer.rho,
                "Weak layer should typically be less dense than slab layers",
            )

    def test_segment_length_consistency(self):
        """Test that segment lengths are reasonable."""
        segments = [
            Segment(length=1000, has_foundation=True, m=0),  # 1m segment
            Segment(
                length=2000, has_foundation=False, m=75
            ),  # 2m free segment with skier
            Segment(length=1500, has_foundation=True, m=0),  # 1.5m segment
        ]

        total_length = sum(seg.length for seg in segments)
        self.assertGreater(total_length, 0, "Total length should be positive")
        self.assertLess(
            total_length, 100000, "Total length should be reasonable (< 100m)"
        )

        # Check that at least one segment is supported
        has_support = any(seg.has_foundation for seg in segments)
        self.assertTrue(
            has_support, "At least one segment should have foundation support"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
