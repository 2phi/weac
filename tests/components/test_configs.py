"""
Unit tests for configuration components.

Tests Config, ScenarioConfig, CriteriaConfig, Segment, and ModelInput validation.
"""

import json

import pytest
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


class TestConfig:
    """Test the Config class for runtime configuration."""

    def test_config_default_creation(self):
        """Test creating Config with default values."""
        config = Config()

        # Check default values
        assert config.touchdown is False

    def test_config_backend_touchdown_compatibility_at_construction(self):
        """Test that generalized backend + touchdown=True is rejected at construction."""
        # Valid combinations should work
        Config(backend="classic", touchdown=False)  # OK
        Config(backend="classic", touchdown=True)  # OK
        Config(backend="generalized", touchdown=False)  # OK

        # Invalid combination should raise ValidationError
        with pytest.raises(
            ValidationError,
            match="Slab touchdown is only available for the classic backend",
        ):
            Config(backend="generalized", touchdown=True)

    def test_config_backend_touchdown_compatibility_via_direct_assignment(self):
        """Test that validation runs when fields are directly assigned."""
        # Start with a valid configuration
        config = Config(backend="generalized", touchdown=False)
        assert config.touchdown is False
        assert config.backend == "generalized"

        # Direct assignment of touchdown should trigger validation and fail
        with pytest.raises(
            ValidationError,
            match="Slab touchdown is only available for the classic backend",
        ):
            config.touchdown = True

        # Verify config state hasn't changed
        assert config.touchdown is False
        assert config.backend == "generalized"

    def test_config_backend_touchdown_compatibility_via_backend_assignment(self):
        """Test that validation runs when backend is changed to incompatible value."""
        # Start with touchdown enabled on classic backend
        config = Config(backend="classic", touchdown=True)
        assert config.touchdown is True
        assert config.backend == "classic"

        # Changing backend to generalized should trigger validation and fail
        with pytest.raises(
            ValidationError,
            match="Slab touchdown is only available for the classic backend",
        ):
            config.backend = "generalized"

        # Verify config state hasn't changed
        assert config.touchdown is True
        assert config.backend == "classic"

    def test_config_valid_assignment_transitions(self):
        """Test that valid field assignments work correctly."""
        config = Config(backend="generalized", touchdown=False)

        # Switch to classic backend first, then enable touchdown - should work
        config.backend = "classic"
        config.touchdown = True

        assert config.backend == "classic"
        assert config.touchdown is True

        # Disable touchdown, then switch to generalized - should work
        config.touchdown = False
        config.backend = "generalized"

        assert config.backend == "generalized"
        assert config.touchdown is False


class TestScenarioConfig:
    """Test the ScenarioConfig class."""

    def test_scenario_config_defaults(self):
        """Test ScenarioConfig with default values."""
        scenario = ScenarioConfig()

        assert scenario.phi == 0
        assert scenario.system_type == "skiers"
        assert scenario.cut_length == 0.0
        assert scenario.stiffness_ratio == 1000
        assert scenario.surface_load == 0.0

    def test_scenario_config_custom_values(self):
        """Test ScenarioConfig with custom values."""
        scenario = ScenarioConfig(
            phi=30.0,
            system_type="skier",
            cut_length=150.0,
            stiffness_ratio=500.0,
            surface_load=0.1,
        )

        assert scenario.phi == 30.0
        assert scenario.system_type == "skier"
        assert scenario.cut_length == 150.0
        assert scenario.stiffness_ratio == 500.0
        assert scenario.surface_load == 0.1

    def test_scenario_config_validation(self):
        """Test ScenarioConfig validation."""
        # Negative crack length
        with pytest.raises(ValidationError):
            ScenarioConfig(cut_length=-10.0)

        # Invalid stiffness ratio (<= 0)
        with pytest.raises(ValidationError):
            ScenarioConfig(stiffness_ratio=0.0)

        # Negative surface load
        with pytest.raises(ValidationError):
            ScenarioConfig(surface_load=-5.0)

        # Invalid system type
        with pytest.raises(ValidationError):
            ScenarioConfig(system_type="invalid_system")


class TestCriteriaConfig:
    """Test the CriteriaConfig class."""

    def test_criteria_config_defaults(self):
        """Test CriteriaConfig with default values."""
        criteria = CriteriaConfig()

        assert criteria.fn == 2.0
        assert criteria.fm == 2.0
        assert criteria.gn == 5.0
        assert criteria.gm == pytest.approx(1 / 0.45, abs=0.5 * 10 ** (-10))
        assert criteria.low_density_threshold_kg_m3 == 100

    def test_criteria_config_custom_values(self):
        """Test CriteriaConfig with custom values."""
        criteria = CriteriaConfig(
            fn=1.5,
            fm=2.0,
            gn=0.8,
            gm=1.2,
            low_density_threshold_kg_m3=120,
        )

        assert criteria.fn == 1.5
        assert criteria.fm == 2.0
        assert criteria.gn == 0.8
        assert criteria.gm == 1.2
        assert criteria.low_density_threshold_kg_m3 == 120

    def test_criteria_config_validation(self):
        """Test CriteriaConfig validation."""
        # All parameters must be positive
        with pytest.raises(ValidationError):
            CriteriaConfig(fn=0.0)

        with pytest.raises(ValidationError):
            CriteriaConfig(fm=-0.5)

        with pytest.raises(ValidationError):
            CriteriaConfig(gn=-1.0)

        with pytest.raises(ValidationError):
            CriteriaConfig(gm=0.0)

        with pytest.raises(ValidationError):
            CriteriaConfig(low_density_threshold_kg_m3=0.0)


class TestSegment:
    """Test the Segment class."""

    def test_segment_creation(self):
        """Test creating segments with various parameters."""
        # Basic segment
        seg1 = Segment(length=1000.0, has_foundation=True, m=0.0)
        assert seg1.length == 1000.0
        assert seg1.has_foundation is True
        assert seg1.m == 0.0

        # Segment with skier load
        seg2 = Segment(length=2000.0, has_foundation=False, m=75.0)
        assert seg2.length == 2000.0
        assert seg2.has_foundation is False
        assert seg2.m == 75.0

    def test_segment_default_mass(self):
        """Test that segment mass defaults to 0."""
        seg = Segment(length=1500.0, has_foundation=True)
        assert seg.m == 0.0

    def test_segment_validation(self):
        """Test segment validation."""
        # Negative length
        with pytest.raises(ValidationError):
            Segment(length=-100.0, has_foundation=True)

        # Negative mass
        with pytest.raises(ValidationError):
            Segment(length=1000.0, has_foundation=True, m=-10.0)


@pytest.fixture
def model_input_parts():
    """Shared builders for ModelInput tests."""
    return {
        "scenario_config": ScenarioConfig(phi=25, system_type="skier"),
        "weak_layer": WeakLayer(rho=50, h=30, E=0.25, G_Ic=1),
        "layers": [Layer(rho=200, h=100), Layer(rho=300, h=150)],
        "segments": [
            Segment(length=3000, has_foundation=True, m=70),
            Segment(length=4000, has_foundation=True, m=0),
        ],
    }


class TestModelInput:
    """Test the ModelInput class for complete model validation."""

    def test_model_input_complete(self, model_input_parts):
        """Test creating complete ModelInput."""
        model = ModelInput(
            scenario_config=model_input_parts["scenario_config"],
            weak_layer=model_input_parts["weak_layer"],
            layers=model_input_parts["layers"],
            segments=model_input_parts["segments"],
        )

        assert model.scenario_config == model_input_parts["scenario_config"]
        assert model.weak_layer == model_input_parts["weak_layer"]
        assert model.layers == model_input_parts["layers"]
        assert model.segments == model_input_parts["segments"]

    def test_model_input_empty_collections(self, model_input_parts):
        """Test validation with empty layers or segments."""
        # Empty layers list
        with pytest.raises(ValidationError):
            ModelInput(
                scenario_config=model_input_parts["scenario_config"],
                weak_layer=model_input_parts["weak_layer"],
                layers=[],
                segments=model_input_parts["segments"],
            )

        # Empty segments list
        with pytest.raises(ValidationError):
            ModelInput(
                scenario_config=model_input_parts["scenario_config"],
                weak_layer=model_input_parts["weak_layer"],
                layers=model_input_parts["layers"],
                segments=[],
            )

    def test_model_input_json_serialization(self, model_input_parts):
        """Test JSON serialization and schema generation."""
        model = ModelInput(
            scenario_config=model_input_parts["scenario_config"],
            weak_layer=model_input_parts["weak_layer"],
            layers=model_input_parts["layers"],
            segments=model_input_parts["segments"],
        )

        # Test JSON serialization
        json_str = model.model_dump_json()
        assert isinstance(json_str, str)

        # Test that it can be parsed back
        parsed_data = json.loads(json_str)
        assert isinstance(parsed_data, dict)

        # Test schema generation
        schema = ModelInput.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "scenario_config" in schema["properties"]
        assert "weak_layer" in schema["properties"]
        assert "layers" in schema["properties"]
        assert "segments" in schema["properties"]


class TestModelInputPhysicalConsistency:
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
            assert (
                weak_layer.rho < layer.rho
            ), "Weak layer should typically be less dense than slab layers"

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
        assert total_length > 0, "Total length should be positive"
        assert total_length < 100000, "Total length should be reasonable (< 100m)"

        # Check that at least one segment is supported
        has_support = any(seg.has_foundation for seg in segments)
        assert has_support, "At least one segment should have foundation support"
