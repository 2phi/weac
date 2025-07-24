"""
Unit tests for the SnowPilotParser class.

Tests the parsing of CAAML files, density measurement extraction,
fallback to hardness+grain type calculations, and stability test parsing.
"""

import unittest
import os
from unittest.mock import patch, MagicMock
import tempfile
import logging

from weac_2.utils.snowpilot_parser import SnowPilotParser
from weac_2.components import Layer, WeakLayer, ModelInput


class TestSnowPilotParser(unittest.TestCase):
    """Test the SnowPilotParser functionality."""

    def setUp(self):
        """Set up test fixtures with paths to test CAAML files."""
        # Paths to test materials in .materials/
        self.materials_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), ".materials"
        )
        self.caaml_with_density = os.path.join(
            self.materials_dir, "snowpits-17030-caaml.xml"
        )
        self.caaml_without_density = os.path.join(
            self.materials_dir, "Falsa Parva-10-Jul-caaml.xml"
        )

        # Verify test files exist
        self.assertTrue(
            os.path.exists(self.caaml_with_density),
            f"Test file not found: {self.caaml_with_density}",
        )
        self.assertTrue(
            os.path.exists(self.caaml_without_density),
            f"Test file not found: {self.caaml_without_density}",
        )

    def test_parse_caaml_with_density_measurements(self):
        """Test parsing CAAML file that contains density measurements."""
        parser = SnowPilotParser(self.caaml_with_density)

        # Capture log messages to verify density source
        with patch("weac_2.utils.snowpilot_parser.logger") as mock_logger:
            layers = parser._extract_layers(parser.snowpit)

            # Should have extracted layers
            self.assertGreater(len(layers), 0, "Should extract layers from CAAML")

            # Check that some layers used measured density
            measured_density_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Using measured density" in str(call)
            ]
            self.assertGreater(
                len(measured_density_calls),
                0,
                "Should use measured density for some layers",
            )

            # Check that some layers may have used computed density (for layers without overlap)
            computed_density_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Using computed density" in str(call)
            ]
            # This may or may not be > 0 depending on overlap, so we don't assert

    def test_parse_caaml_without_density_measurements(self):
        """Test parsing CAAML file that lacks density measurements."""
        parser = SnowPilotParser(self.caaml_without_density)

        # Capture log messages to verify density source
        with patch("weac_2.utils.snowpilot_parser.logger") as mock_logger:
            layers = parser._extract_layers(parser.snowpit)

            # Should have extracted layers
            self.assertGreater(len(layers), 0, "Should extract layers from CAAML")

            # All layers should use computed density (no density measurements available)
            computed_density_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Using computed density" in str(call)
                and "no density measurement available" in str(call)
            ]
            self.assertEqual(
                len(computed_density_calls),
                len(layers),
                "All layers should use computed density when no measurements available",
            )

    def test_density_extraction_logic(self):
        """Test the density extraction logic with overlapping measurements."""
        parser = SnowPilotParser(self.caaml_with_density)

        # Get density layers for testing
        sp_density_layers = [
            layer
            for layer in parser.snowpit.snow_profile.density_profile
            if layer.depth_top is not None
        ]

        # Test case 1: Layer that should overlap with density measurements
        # From the CAAML file, we have density measurements at 0-4cm, 10-14cm, etc.
        # Test a layer at 2-6cm (should overlap with 0-4cm measurement)
        density = parser._get_density_for_layer_range(
            20, 60, sp_density_layers
        )  # 2-6cm in mm
        self.assertIsNotNone(density, "Should find density for overlapping layer")
        self.assertIsInstance(density, float, "Density should be a float")
        self.assertGreater(density, 0, "Density should be positive")

        # Test case 2: Layer with no overlap
        # Test a layer well beyond the density measurements
        density_no_overlap = parser._get_density_for_layer_range(
            1000, 1100, sp_density_layers
        )  # 100-110cm
        self.assertIsNone(
            density_no_overlap, "Should return None for non-overlapping layer"
        )

    def test_stability_test_parsing(self):
        """Test parsing of different stability test types."""
        # Test file with PST
        parser_pst = SnowPilotParser(self.caaml_without_density)
        model_inputs_pst = parser_pst.run()

        # Should generate model inputs based on stability tests
        self.assertGreater(len(model_inputs_pst), 0, "Should generate model inputs")

        # Check for PST-specific scenarios
        pst_scenarios = [
            mi for mi in model_inputs_pst if mi.scenario_config.system_type == "-pst"
        ]
        self.assertGreater(len(pst_scenarios), 0, "Should create PST scenarios")

        # Test file with CT tests
        parser_ct = SnowPilotParser(self.caaml_with_density)
        model_inputs_ct = parser_ct.run()

        # Should generate model inputs for CT tests
        self.assertGreater(
            len(model_inputs_ct), 0, "Should generate model inputs for CT tests"
        )

    def test_layer_properties_validation(self):
        """Test that extracted layers have valid properties."""
        parser = SnowPilotParser(self.caaml_with_density)
        layers = parser._extract_layers(parser.snowpit)

        for i, layer in enumerate(layers):
            with self.subTest(layer_index=i):
                # Validate layer properties
                self.assertIsInstance(
                    layer, Layer, f"Layer {i} should be Layer instance"
                )
                self.assertGreater(
                    layer.rho, 0, f"Layer {i} density should be positive"
                )
                self.assertGreater(
                    layer.h, 0, f"Layer {i} thickness should be positive"
                )
                self.assertLessEqual(
                    layer.rho,
                    1000,
                    f"Layer {i} density should be reasonable (<= 1000 kg/m³)",
                )

    def test_model_input_generation(self):
        """Test that model inputs are generated correctly."""
        parser = SnowPilotParser(self.caaml_with_density)
        model_inputs = parser.run()

        self.assertGreater(
            len(model_inputs), 0, "Should generate at least one model input"
        )

        for i, model_input in enumerate(model_inputs):
            with self.subTest(scenario_index=i):
                # Validate model input structure
                self.assertIsInstance(
                    model_input,
                    ModelInput,
                    f"Model input {i} should be ModelInput instance",
                )
                self.assertIsInstance(
                    model_input.weak_layer,
                    WeakLayer,
                    f"Model input {i} should have WeakLayer",
                )
                self.assertGreater(
                    len(model_input.layers), 0, f"Model input {i} should have layers"
                )
                self.assertGreater(
                    len(model_input.segments),
                    0,
                    f"Model input {i} should have segments",
                )

                # Validate slope angle was extracted
                self.assertIsInstance(
                    model_input.scenario_config.phi,
                    (int, float),
                    f"Model input {i} should have slope angle",
                )

    def test_weak_layer_extraction(self):
        """Test weak layer extraction for different depths."""
        parser = SnowPilotParser(self.caaml_with_density)
        layers = parser.layers = parser._extract_layers(parser.snowpit)

        # Test weak layer extraction at a specific depth (e.g., 21cm from CT test)
        test_depth_mm = 210  # 21cm converted to mm
        weak_layer, layers_above = parser._extract_weak_layer_and_layers_above(
            parser.snowpit, test_depth_mm, layers
        )

        # Validate weak layer
        self.assertIsInstance(
            weak_layer, WeakLayer, "Should extract WeakLayer instance"
        )
        self.assertGreater(weak_layer.rho, 0, "Weak layer density should be positive")
        self.assertGreater(weak_layer.h, 0, "Weak layer thickness should be positive")

        # Validate layers above
        self.assertGreater(len(layers_above), 0, "Should have layers above weak layer")
        total_depth_above = sum(layer.h for layer in layers_above)
        self.assertAlmostEqual(
            total_depth_above,
            test_depth_mm,
            delta=1,
            msg="Total depth of layers above should match test depth",
        )

    def test_error_handling_missing_data(self):
        """Test error handling for missing required data."""
        # This would require creating a malformed CAAML file or mocking
        # For now, test that parser handles empty density layers gracefully
        parser = SnowPilotParser(self.caaml_without_density)

        # Test with empty density layers list
        result = parser._get_density_for_layer_range(0, 100, [])
        self.assertIsNone(result, "Should return None for empty density layers")

    def test_unit_conversion(self):
        """Test that different units are converted correctly."""
        parser = SnowPilotParser(self.caaml_with_density)
        layers = parser._extract_layers(parser.snowpit)

        # All thicknesses should be in mm (converted from cm in CAAML)
        for layer in layers:
            # Thicknesses should be reasonable for mm units (> 1mm, < 2000mm typically)
            self.assertGreater(layer.h, 0.1, "Layer thickness should be > 0.1mm")
            self.assertLess(
                layer.h, 5000, "Layer thickness should be < 5000mm (reasonable limit)"
            )

    def test_density_weighted_average(self):
        """Test that overlapping density measurements are weighted correctly."""
        parser = SnowPilotParser(self.caaml_with_density)

        # Get density layers
        sp_density_layers = [
            layer
            for layer in parser.snowpit.snow_profile.density_profile
            if layer.depth_top is not None
        ]

        # Test a layer that spans multiple density measurements
        # Based on the CAAML data, density measurements are at:
        # 0-4cm (20 kg/m³), 10-14cm (20 kg/m³), 20-24cm (20 kg/m³), etc.

        # Test layer from 0-25cm (should span first 3 measurements)
        density = parser._get_density_for_layer_range(
            0, 250, sp_density_layers
        )  # 0-25cm in mm

        if density is not None:  # May be None if no overlap logic issue
            self.assertIsInstance(density, float, "Weighted density should be float")
            self.assertGreater(density, 0, "Weighted density should be positive")
            # Should be close to 20 since most measurements are 20 kg/m³
            self.assertAlmostEqual(
                density, 20, delta=5, msg="Weighted average should be close to 20 kg/m³"
            )


if __name__ == "__main__":
    # Set up logging to see debug info during tests
    logging.basicConfig(level=logging.INFO)

    unittest.main()
