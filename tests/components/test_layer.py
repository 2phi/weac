"""
Unit tests for Layer and WeakLayer components.

Tests validation, automatic property calculations, and edge cases.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from weac.components.layer import (
    Layer,
    _adam_tensile_strength,
    _bergfeld_youngs_modulus,
    _gerling_youngs_modulus,
    _scapozza_youngs_modulus,
    _sigrist_tensile_strength,
)
from weac.components.presets import (
    LESS_WEAK_LAYER,
    VERY_WEAK_LAYER,
    WEAK_LAYER,
)
from weac.components.weak_layer import WeakLayer, _schottner_fc_dh_youngs_modulus
from weac.constants import CS0, CS1, NU, RHO_ICE


class TestLayerPropertyCalculations:
    """Test the layer property calculation functions."""

    def test_bergfeld_calculation(self):
        """Test Bergfeld Young's modulus calculation."""
        # Test with standard ice density
        E = _bergfeld_youngs_modulus(rho=917.0)  # Ice density
        assert E > 0, "Young's modulus should be positive"
        assert np.isscalar(E), "Result should be a scalar"

        # Test with typical snow densities
        E_light = _bergfeld_youngs_modulus(rho=100.0)
        E_heavy = _bergfeld_youngs_modulus(rho=400.0)
        assert E_light < E_heavy, "Heavier snow should have higher modulus"

    def test_scapozza_calculation(self):
        """Test Scapozza Young's modulus calculation."""
        E = _scapozza_youngs_modulus(rho=200.0)
        assert E > 0, "Young's modulus should be positive"

    def test_gerling_calculation(self):
        """Test Gerling Young's modulus calculation."""
        E = _gerling_youngs_modulus(rho=250.0)
        assert E > 0, "Young's modulus should be positive"

    def test_schottner_fc_dh_calculation(self):
        """Test Schöttner FC&DH Young's modulus at paper benchmark densities."""
        E_150 = _schottner_fc_dh_youngs_modulus(rho=150.0)
        E_250 = _schottner_fc_dh_youngs_modulus(rho=250.0)
        expected_150 = CS0 * (150.0 / RHO_ICE) ** CS1
        expected_250 = CS0 * (250.0 / RHO_ICE) ** CS1
        assert E_150 == pytest.approx(expected_150, abs=0.5 * 10 ** (-10))
        assert E_250 == pytest.approx(expected_250, abs=0.5 * 10 ** (-10))
        # Paper window: ~1.5 MPa at 150 kg/m³, ~24 MPa at 250 kg/m³
        assert E_150 == pytest.approx(1.5, abs=0.1)
        assert E_250 == pytest.approx(24.0, abs=1.0)
        assert E_150 < E_250


class TestTensileStrengthCalculations:
    """Test tensile strength calculation functions."""

    def test_sigrist_calculation_kPa(self):
        """Test Sigrist tensile strength calculation in kPa."""
        # Test with typical snow density
        ts = _sigrist_tensile_strength(rho=200.0, unit="kPa")
        assert ts > 0, "Tensile strength should be positive"
        assert np.isscalar(ts), "Result should be a scalar"

        # Test with different densities
        ts_light = _sigrist_tensile_strength(rho=100.0, unit="kPa")
        ts_heavy = _sigrist_tensile_strength(rho=400.0, unit="kPa")
        assert ts_light < ts_heavy, "Heavier snow should have higher strength"

    def test_sigrist_calculation_MPa(self):
        """Test Sigrist tensile strength calculation in MPa."""
        ts_kPa = _sigrist_tensile_strength(rho=200.0, unit="kPa")
        ts_MPa = _sigrist_tensile_strength(rho=200.0, unit="MPa")
        assert ts_kPa == pytest.approx(
            ts_MPa * 1000, abs=0.5 * 10 ** (-5)
        ), "Unit conversion should be correct"

    def test_adam_calculation_kPa(self):
        """Test Adam tensile strength calculation in kPa."""
        # Test with typical snow density
        ts = _adam_tensile_strength(rho=300.0, unit="kPa")
        assert ts > 0, "Tensile strength should be positive"
        assert np.isscalar(ts), "Result should be a scalar"

        # Test with different densities
        ts_light = _adam_tensile_strength(rho=150.0, unit="kPa")
        ts_heavy = _adam_tensile_strength(rho=450.0, unit="kPa")
        assert ts_light < ts_heavy, "Heavier snow should have higher strength"

    def test_adam_calculation_MPa(self):
        """Test Adam tensile strength calculation in MPa."""
        ts_kPa = _adam_tensile_strength(rho=300.0, unit="kPa")
        ts_MPa = _adam_tensile_strength(rho=300.0, unit="MPa")
        assert ts_kPa == pytest.approx(
            ts_MPa * 1000, abs=0.5 * 10 ** (-5)
        ), "Unit conversion should be correct"

    def test_sigrist_vs_adam_comparison(self):
        """Compare Sigrist and Adam formulations at different densities."""
        # At low densities, compare the formulations
        rho_low = 150.0
        ts_sigrist = _sigrist_tensile_strength(rho=rho_low, unit="kPa")
        ts_adam = _adam_tensile_strength(rho=rho_low, unit="kPa")
        # Both should give positive values
        assert ts_sigrist > 0
        assert ts_adam > 0

        # At high densities
        rho_high = 400.0
        ts_sigrist_high = _sigrist_tensile_strength(rho=rho_high, unit="kPa")
        ts_adam_high = _adam_tensile_strength(rho=rho_high, unit="kPa")
        assert ts_sigrist_high > 0
        assert ts_adam_high > 0


class TestLayerTensileStrength:
    """Test Layer class tensile strength functionality."""

    def test_layer_default_tensile_strength_method(self):
        """Test that default method is 'hybrid'."""
        layer = Layer(rho=200.0, h=100.0)
        assert (
            layer.tensile_strength_method == "hybrid"
        ), "Default method should be 'hybrid'"
        assert layer.tensile_strength > 0, "Tensile strength should be calculated"

    def test_layer_sigrist_method(self):
        """Test Layer with explicit Sigrist method."""
        layer = Layer(rho=200.0, h=100.0, tensile_strength_method="sigrist")
        expected_ts = _sigrist_tensile_strength(rho=200.0, unit="kPa")
        assert layer.tensile_strength == pytest.approx(
            expected_ts, abs=0.5 * 10 ** (-5)
        ), "Tensile strength should match Sigrist calculation"

    def test_layer_adam_method(self):
        """Test Layer with explicit Adam method."""
        layer = Layer(rho=300.0, h=100.0, tensile_strength_method="adam")
        expected_ts = _adam_tensile_strength(rho=300.0, unit="kPa")
        assert layer.tensile_strength == pytest.approx(
            expected_ts, abs=0.5 * 10 ** (-5)
        ), "Tensile strength should match Adam calculation"

    def test_layer_hybrid_method_low_density(self):
        """Test hybrid method uses Sigrist for density < 250."""
        rho = 200.0  # Below 250 threshold
        layer = Layer(rho=rho, h=100.0, tensile_strength_method="hybrid")
        expected_ts = _sigrist_tensile_strength(rho=rho, unit="kPa")
        assert layer.tensile_strength == pytest.approx(
            expected_ts, abs=0.5 * 10 ** (-5)
        ), "Hybrid should use Sigrist for rho < 250"

    def test_layer_hybrid_method_high_density(self):
        """Test hybrid method uses Adam for density >= 250."""
        rho = 300.0  # Above 250 threshold
        layer = Layer(rho=rho, h=100.0, tensile_strength_method="hybrid")
        expected_ts = _adam_tensile_strength(rho=rho, unit="kPa")
        assert layer.tensile_strength == pytest.approx(
            expected_ts, abs=0.5 * 10 ** (-5)
        ), "Hybrid should use Adam for rho >= 250"

    def test_layer_hybrid_method_at_threshold(self):
        """Test hybrid method behavior exactly at 250 kg/m³."""
        rho = 250.0  # Exactly at threshold
        layer = Layer(rho=rho, h=100.0, tensile_strength_method="hybrid")
        expected_ts = _adam_tensile_strength(rho=rho, unit="kPa")
        assert layer.tensile_strength == pytest.approx(
            expected_ts, abs=0.5 * 10 ** (-5)
        ), "Hybrid should use Adam for rho = 250"

    def test_layer_custom_tensile_strength(self):
        """Test that custom tensile strength overrides calculation."""
        custom_ts = 50.0
        layer = Layer(
            rho=200.0,
            h=100.0,
            tensile_strength=custom_ts,
            tensile_strength_method="sigrist",
        )
        assert (
            layer.tensile_strength == custom_ts
        ), "Custom tensile strength should override calculation"


class TestTensileStrengthPhysicalConsistency:
    """Test physical consistency of tensile strength calculations."""

    def test_density_strength_relationship(self):
        """Test that higher density leads to higher tensile strength."""
        layer_light = Layer(rho=150.0, h=100.0)
        layer_heavy = Layer(rho=350.0, h=100.0)

        assert (
            layer_light.tensile_strength < layer_heavy.tensile_strength
        ), "Heavier snow should have higher tensile strength"

    def test_hybrid_continuity_around_threshold(self):
        """Test continuity of hybrid method around 250 kg/m³ threshold."""
        # Test just below threshold
        layer_below = Layer(rho=249.0, h=100.0, tensile_strength_method="hybrid")
        # Test just above threshold
        layer_above = Layer(rho=251.0, h=100.0, tensile_strength_method="hybrid")

        # Both should have positive strength
        assert layer_below.tensile_strength > 0
        assert layer_above.tensile_strength > 0

        # Values should be reasonably close (within an order of magnitude)
        # This is a loose check since the formulations differ
        ratio = layer_above.tensile_strength / layer_below.tensile_strength
        assert ratio < 10.0, "Strength shouldn't jump by more than 10x at threshold"
        assert ratio > 0.1, "Strength shouldn't drop by more than 10x at threshold"

    def test_all_methods_give_positive_strength(self):
        """Test that all methods produce positive tensile strength."""
        rho_values = [100.0, 200.0, 300.0, 400.0]
        methods = ["sigrist", "adam", "hybrid"]

        for rho in rho_values:
            for method in methods:
                layer = Layer(rho=rho, h=100.0, tensile_strength_method=method)
                assert (
                    layer.tensile_strength > 0
                ), f"Method {method} with rho={rho} should give positive strength"

    def test_tensile_strength_density_monotonicity(self):
        """Test that tensile strength increases monotonically with density."""
        densities = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
        methods = ["sigrist", "adam", "hybrid"]

        for method in methods:
            strengths = [
                Layer(rho=rho, h=100.0, tensile_strength_method=method).tensile_strength
                for rho in densities
            ]
            # Check that each strength is greater than the previous
            for i in range(1, len(strengths)):
                assert (
                    strengths[i] > strengths[i - 1]
                ), f"Strength should increase with density for {method} method"


class TestLayer:
    """Test the Layer class functionality."""

    def test_layer_creation_with_required_fields(self):
        """Test creating a layer with only required fields."""
        layer = Layer(rho=200.0, h=100.0)

        # Check required fields
        assert layer.rho == 200.0
        assert layer.h == 100.0

        # Check auto-calculated fields
        assert layer.E is not None, "Young's modulus should be auto-calculated"
        assert layer.G is not None, "Shear modulus should be auto-calculated"
        assert layer.E > 0, "Young's modulus should be positive"
        assert layer.G > 0, "Shear modulus should be positive"

        # Check default Poisson's ratio
        assert layer.nu == NU, "Default Poisson's ratio should be 0.25"

    def test_layer_creation_with_all_fields(self):
        """Test creating a layer with all fields specified."""
        layer = Layer(rho=250.0, h=150.0, nu=0.3, E=50.0, G=20.0)

        assert layer.rho == 250.0
        assert layer.h == 150.0
        assert layer.nu == 0.3
        assert layer.E == 50.0, "Specified E should override auto-calculation"
        assert layer.G == 20.0, "Specified G should override auto-calculation"

    def test_layer_validation_errors(self):
        """Test that invalid layer parameters raise ValidationError."""
        # Negative density
        with pytest.raises(ValidationError):
            Layer(rho=-100.0, h=100.0)

        # Zero thickness
        with pytest.raises(ValidationError):
            Layer(rho=200.0, h=0.0)

        # Invalid Poisson's ratio (>= 0.5)
        with pytest.raises(ValidationError):
            Layer(rho=200.0, h=100.0, nu=0.5)

        # Negative Young's modulus
        with pytest.raises(ValidationError):
            Layer(rho=200.0, h=100.0, E=-10.0)

    def test_shear_modulus_calculation(self):
        """Test automatic shear modulus calculation from E and nu."""
        layer = Layer(rho=200.0, h=100.0, nu=0.25, E=100.0)

        # G = E / (2 * (1 + nu))
        expected_G = 100.0 / (2 * (1 + 0.25))
        assert layer.G == pytest.approx(expected_G, abs=0.5 * 10 ** (-5))


class TestWeakLayer:
    """Test the WeakLayer class functionality."""

    def test_weak_layer_defaults_schottner(self):
        """Bare WeakLayer uses Schöttner FC&DH at rho=150."""
        wl = WeakLayer()
        assert wl.rho == 150.0
        assert wl.E_method == "schottner_fc_dh"
        expected = CS0 * (150.0 / RHO_ICE) ** CS1
        assert wl.E == pytest.approx(expected, abs=0.5 * 10 ** (-10))
        assert wl.E == pytest.approx(1.55, abs=0.02)

    def test_weak_layer_explicit_E_skips_density_law(self):
        """Explicit E > 0 overrides the density law (PlaneStrain keeps E)."""
        wl = WeakLayer(rho=200.0, E=5.0)
        assert wl.E == pytest.approx(5.0, abs=0.5 * 10 ** (-10))

    def test_weak_layer_bergfeld_selection(self):
        """Explicit E_method=bergfeld uses Bergfeld, not Schottner."""
        wl = WeakLayer(rho=200.0, E_method="bergfeld")
        expected = _bergfeld_youngs_modulus(200.0)
        assert wl.E == pytest.approx(expected, abs=0.5 * 10 ** (-10))
        assert wl.E != pytest.approx(
            _schottner_fc_dh_youngs_modulus(200.0), abs=0.5 * 10 ** (-5)
        )

    def test_weak_layer_presets_density_to_E(self):
        """Presets derive E from Schottner at their densities."""
        assert VERY_WEAK_LAYER.rho == 100
        assert WEAK_LAYER.rho == 150
        assert LESS_WEAK_LAYER.rho == 200
        assert VERY_WEAK_LAYER.E == pytest.approx(
            _schottner_fc_dh_youngs_modulus(100), abs=0.5 * 10 ** (-10)
        )
        assert WEAK_LAYER.E == pytest.approx(
            _schottner_fc_dh_youngs_modulus(150), abs=0.5 * 10 ** (-10)
        )
        assert LESS_WEAK_LAYER.E == pytest.approx(
            _schottner_fc_dh_youngs_modulus(200), abs=0.5 * 10 ** (-10)
        )

    def test_weak_layer_creation_minimal(self):
        """Test creating a weak layer with minimal required fields."""
        wl = WeakLayer(rho=50.0, h=10.0)

        # Check required fields
        assert wl.rho == 50.0
        assert wl.h == 10.0

        # Check auto-calculated fields
        assert wl.E is not None, "Young's modulus should be auto-calculated"
        assert wl.G is not None, "Shear modulus should be auto-calculated"
        assert wl.kn is not None, "Normal stiffness should be auto-calculated"
        assert wl.kt is not None, "Shear stiffness should be auto-calculated"
        assert wl.E > 0, "Young's modulus should be positive"
        assert wl.G > 0, "Shear modulus should be positive"
        assert wl.kn > 0, "Normal stiffness should be positive"
        assert wl.kt > 0, "Shear stiffness should be positive"

        # Check default fracture properties
        assert wl.G_Ic == 0.56
        assert wl.G_IIc == 0.79

    def test_weak_layer_stiffness_calculations(self):
        """Test weak layer stiffness calculations."""
        wl = WeakLayer(rho=100.0, h=20.0, E=10.0, nu=0.2)

        # kn = E_plane / h = E / (1 - nu²) / h
        E_plane = 10.0 / (1 - 0.2**2)
        expected_kn = E_plane / 20.0
        assert wl.kn == pytest.approx(expected_kn, abs=0.5 * 10 ** (-5))

        # kt = G / h
        expected_G = 10.0 / (2 * (1 + 0.2))
        expected_kt = expected_G / 20.0
        assert wl.kt == pytest.approx(expected_kt, abs=0.5 * 10 ** (-5))

    def test_weak_layer_custom_stiffnesses(self):
        """Test weak layer with custom stiffness values."""
        wl = WeakLayer(rho=80.0, h=15.0, kn=5.0, kt=3.0)

        assert wl.kn == 5.0, "Custom kn should override calculation"
        assert wl.kt == 3.0, "Custom kt should override calculation"

    def test_weak_layer_fracture_properties(self):
        """Test weak layer fracture property validation."""
        wl = WeakLayer(rho=90.0, h=25.0, G_Ic=1.5, G_IIc=1.8)

        assert wl.G_Ic == 1.5
        assert wl.G_IIc == 1.8

    def test_weak_layer_validation_errors(self):
        """Test weak layer validation errors."""
        # Zero thickness
        with pytest.raises(ValidationError):
            WeakLayer(rho=100.0, h=0.0)


class TestLayerPhysicalConsistency:
    """Test physical consistency of layer calculations."""

    def test_layer_density_modulus_relationship(self):
        """Test that higher density leads to higher modulus."""
        layer_light = Layer(rho=150.0, h=100.0)
        layer_heavy = Layer(rho=350.0, h=100.0)

        assert (
            layer_light.E < layer_heavy.E
        ), "Heavier snow should have higher Young's modulus"
        assert (
            layer_light.G < layer_heavy.G
        ), "Heavier snow should have higher shear modulus"

    def test_weak_layer_thickness_stiffness_relationship(self):
        """Test that thicker weak layers have lower stiffness."""
        wl_thin = WeakLayer(rho=100.0, h=10.0)
        wl_thick = WeakLayer(rho=100.0, h=30.0)

        assert (
            wl_thin.kn > wl_thick.kn
        ), "Thinner weak layer should have higher normal stiffness"
        assert (
            wl_thin.kt > wl_thick.kt
        ), "Thinner weak layer should have higher shear stiffness"

    def test_poisson_ratio_bounds(self):
        """Test Poisson's ratio physical bounds."""
        # Test upper bound (must be < 0.5 for positive definite stiffness)
        with pytest.raises(ValidationError):
            Layer(rho=200.0, h=100.0, nu=0.5)

        with pytest.raises(ValidationError):
            Layer(rho=200.0, h=100.0, nu=0.6)

        # Test lower bound (must be >= 0)
        with pytest.raises(ValidationError):
            Layer(rho=200.0, h=100.0, nu=-0.1)
