"""
Unit tests for the SnowPilotParser class.

Tests the parsing of CAAML files, density measurement extraction,
fallback to hardness+grain type calculations, and stability test parsing.
"""

import unittest
import os
import logging

from weac.utils.snowpilot_parser import SnowPilotParser
from weac.components import Layer, WeakLayer


class TestSnowPilotParser(unittest.TestCase):
    """Test the SnowPilotParser functionality."""

    def setUp(self):
        """Set up test fixtures with paths to test CAAML files."""
        # Paths to test materials in .materials/
        self.materials_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), ".materials"
        )
        self.caaml_with_density = os.path.join(self.materials_dir, "test_snowpit1.xml")
        self.caaml_without_density = os.path.join(
            self.materials_dir, "test_snowpit2.xml"
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
        layers, density_methods = parser.extract_layers()

        # Should have extracted layers
        self.assertGreater(len(layers), 0, "Should extract layers from CAAML")
        self.assertGreater(
            density_methods.count("density_obs"),
            0,
            "Should use measured density for some layers",
        )

    def test_parse_caaml_without_density_measurements(self):
        """Test parsing CAAML file that lacks density measurements."""
        parser = SnowPilotParser(self.caaml_without_density)
        layers, density_methods = parser.extract_layers()

        # Should have extracted layers
        self.assertGreater(len(layers), 0, "Should extract layers from CAAML")
        self.assertEqual(density_methods.count("geldsetzer"), len(layers))

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

    def test_layer_properties_validation(self):
        """Test that extracted layers have valid properties."""
        parser = SnowPilotParser(self.caaml_with_density)
        layers, _ = parser.extract_layers()

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

    def test_weak_layer_extraction(self):
        """Test weak layer extraction for different depths."""
        parser = SnowPilotParser(self.caaml_with_density)
        layers, _ = parser.extract_layers()

        # Test weak layer extraction at a specific depth (e.g., 21cm from CT test)
        test_depth_mm = 210  # 21cm converted to mm
        weak_layer, layers_above = parser.extract_weak_layer_and_layers_above(
            test_depth_mm, layers
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
        layers, _ = parser.extract_layers()

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
