"""
Unit tests for the Eigensystem class.

Tests system matrix assembly, eigenvalue/eigenvector calculations,
complementary and particular solutions.
"""

import numpy as np
import pytest

from weac.components import Layer, WeakLayer
from weac.core.eigensystem import Eigensystem
from weac.core.slab import Slab


@pytest.fixture
def multi_layer_eigensystem():
    """Eigensystem with a two-layer slab."""
    layers = [Layer(rho=200, h=100), Layer(rho=300, h=150)]
    weak_layer = WeakLayer(rho=50, h=20, E=0.5, G_Ic=1.0)
    slab = Slab(layers)
    return Eigensystem(weak_layer, slab)


@pytest.fixture
def single_layer_eigensystem():
    """Eigensystem with a single-layer slab (eigenvalue tests)."""
    layers = [Layer(rho=250, h=120)]
    weak_layer = WeakLayer(rho=80, h=25, E=0.3)
    slab = Slab(layers)
    return Eigensystem(weak_layer, slab)


@pytest.fixture
def solution_eigensystem():
    """Eigensystem for complementary/particular solution tests."""
    layers = [Layer(rho=200, h=100)]
    weak_layer = WeakLayer(rho=60, h=15)
    slab = Slab(layers)
    return Eigensystem(weak_layer, slab)


class TestEigensystemBasicProperties:
    """Test basic eigensystem setup and property calculations."""

    def test_eigensystem_initialization(self, multi_layer_eigensystem):
        """Test that eigensystem initializes correctly."""
        eigensystem = multi_layer_eigensystem
        assert eigensystem.weak_layer is not None
        assert eigensystem.slab is not None

        # Check that eigenvalue calculation was performed
        assert eigensystem.ewC is not None, "Complex eigenvalues should be calculated"
        assert eigensystem.ewR is not None, "Real eigenvalues should be calculated"
        assert eigensystem.evC is not None, "Complex eigenvectors should be calculated"
        assert eigensystem.evR is not None, "Real eigenvectors should be calculated"

    def test_laminate_stiffness_parameters(self, multi_layer_eigensystem):
        """Test calculation of laminate stiffness parameters."""
        eigensystem = multi_layer_eigensystem
        # Check that stiffness parameters are positive
        assert eigensystem.A11 > 0, "Extensional stiffness should be positive"
        assert eigensystem.D11 > 0, "Bending stiffness should be positive"
        assert eigensystem.kA55 > 0, "Shear stiffness should be positive"

        # K0 can be negative depending on coupling
        assert isinstance(eigensystem.K0, float)

    def test_system_matrix_properties(self, multi_layer_eigensystem):
        """Test properties of the system matrix."""
        K = multi_layer_eigensystem.K

        # Check matrix dimensions
        assert K.shape == (6, 6), "System matrix should be 6x6"

        # Check that it's a real matrix
        assert np.all(np.isreal(K)), "System matrix should be real"

        # Check specific structure (first row should be [0, 1, 0, 0, 0, 0])
        expected_first_row = [0, 1, 0, 0, 0, 0]
        np.testing.assert_array_equal(
            K[0, :],
            expected_first_row,
            "First row of system matrix has known structure",
        )

        # Check third row should be [0, 0, 0, 1, 0, 0]
        expected_third_row = [0, 0, 0, 1, 0, 0]
        np.testing.assert_array_equal(
            K[2, :],
            expected_third_row,
            "Third row of system matrix has known structure",
        )

        # Check fifth row should be [0, 0, 0, 0, 0, 1]
        expected_fifth_row = [0, 0, 0, 0, 0, 1]
        np.testing.assert_array_equal(
            K[4, :],
            expected_fifth_row,
            "Fifth row of system matrix has known structure",
        )


class TestEigensystemEigenvalueCalculations:
    """Test eigenvalue and eigenvector calculations."""

    def test_eigenvalue_classification(self, single_layer_eigensystem):
        """Test that eigenvalues are correctly classified."""
        eigensystem = single_layer_eigensystem
        # Real eigenvalues should be real
        assert np.all(np.isreal(eigensystem.ewR)), (
            "Real eigenvalues should be real numbers"
        )

        # Complex eigenvalues should have positive imaginary parts
        if len(eigensystem.ewC) > 0:
            assert np.all(eigensystem.ewC.imag > 0), (
                "Complex eigenvalues should have positive imaginary parts"
            )

    def test_eigenvector_dimensions(self, single_layer_eigensystem):
        """Test that eigenvectors have correct dimensions."""
        eigensystem = single_layer_eigensystem
        # Real eigenvectors
        if len(eigensystem.ewR) > 0:
            assert eigensystem.evR.shape[0] == 6, (
                "Real eigenvectors should be 6-dimensional"
            )
            assert eigensystem.evR.shape[1] == len(eigensystem.ewR), (
                "Number of real eigenvectors should match number of real eigenvalues"
            )

        # Complex eigenvectors
        if len(eigensystem.ewC) > 0:
            assert eigensystem.evC.shape[0] == 6, (
                "Complex eigenvectors should be 6-dimensional"
            )
            assert eigensystem.evC.shape[1] == len(eigensystem.ewC), (
                "Number of complex eigenvectors should match number of complex eigenvalues"
            )

    def test_eigenvalue_shifts(self, single_layer_eigensystem):
        """Test eigenvalue shift arrays."""
        eigensystem = single_layer_eigensystem
        # Shifts should have same length as eigenvalues
        assert len(eigensystem.sR) == len(eigensystem.ewR), (
            "Real shifts should match real eigenvalues"
        )
        assert len(eigensystem.sC) == len(eigensystem.ewC), (
            "Complex shifts should match complex eigenvalues"
        )

        # Shifts should be -1 or 0
        assert np.all(np.isin(eigensystem.sR, [-1, 0])), "Real shifts should be -1 or 0"
        assert np.all(np.isin(eigensystem.sC, [-1, 0])), (
            "Complex shifts should be -1 or 0"
        )


class TestEigensystemSolutionMethods:
    """Test complementary and particular solution methods."""

    def test_complementary_solution_bedded(self, solution_eigensystem):
        """Test complementary solution for bedded segment."""
        x = 100.0  # Position
        length = 1000.0  # Segment length
        has_foundation = True  # Bedded

        zh = solution_eigensystem.zh(x, length, has_foundation)

        # Should return 6x6 matrix
        assert zh.shape == (6, 6), "Complementary solution should be 6x6 matrix"

        # Should be real for bedded segments
        assert np.allclose(np.imag(zh), 0.0, atol=1e-12), (
            "Bedded complementary solution should be (numerically) real"
        )

    def test_complementary_solution_free(self, solution_eigensystem):
        """Test complementary solution for free segment."""
        x = 50.0  # Position
        length = 500.0  # Segment length
        has_foundation = False  # Free

        zh = solution_eigensystem.zh(x, length, has_foundation)

        # Should return 6x6 matrix
        assert zh.shape == (6, 6), "Complementary solution should be 6x6 matrix"

        assert np.allclose(np.imag(zh), 0.0, atol=1e-12), (
            "Free complementary solution should be (numerically) real"
        )

    def test_complementary_solution_at_origin(self, solution_eigensystem):
        """Test complementary solution at x=0."""
        zh_bedded = solution_eigensystem.zh(0.0, 1000.0, True)
        zh_free = solution_eigensystem.zh(0.0, 1000.0, False)

        # At x=0, certain columns should have specific values
        # For free segments, the polynomial form gives specific patterns
        assert np.isfinite(zh_bedded).all(), "Bedded solution should be finite at origin"
        assert np.isfinite(zh_free).all(), "Free solution should be finite at origin"

    def test_particular_solution_bedded(self, solution_eigensystem):
        """Test particular solution for bedded segment."""
        x = 200.0  # Position
        phi = 30.0  # Inclination
        has_foundation = True  # Bedded
        qs = 5.0  # Surface load

        zp = solution_eigensystem.zp(x, phi, has_foundation, qs)

        # Should return 6x1 vector
        assert zp.shape == (6, 1), "Particular solution should be 6x1 vector"
        # Should be real
        assert np.allclose(np.imag(zp), 0.0, atol=1e-12), (
            "Particular solution should be (numerically) real"
        )

    def test_particular_solution_free(self, solution_eigensystem):
        """Test particular solution for free segment."""
        x = 150.0  # Position
        phi = 25.0  # Inclination
        has_foundation = False  # Free
        qs = 0.0  # No additional surface load

        zp = solution_eigensystem.zp(x, phi, has_foundation, qs)

        # Should be real
        assert np.allclose(np.imag(zp), 0.0, atol=1e-12), (
            "Particular solution should be (numerically) real"
        )

    def test_load_vector_calculation(self, solution_eigensystem):
        """Test system load vector calculation."""
        phi = 20.0  # Inclination
        qs = 10.0  # Surface load

        q = solution_eigensystem.get_load_vector(phi, qs)

        # Should return 6x1 vector
        assert q.shape == (6, 1), "Load vector should be 6x1"

        # Should be real
        assert np.allclose(np.imag(q), 0.0, atol=1e-12), (
            "Load vector should be (numerically) real"
        )


class TestEigensystemPhysicalConsistency:
    """Test physical consistency of eigensystem calculations."""

    def test_stiffness_scaling_with_properties(self):
        """Test that stiffness parameters scale correctly with material properties."""
        # Create two systems with different Young's moduli
        layers1 = [Layer(rho=200, h=100, E=50)]
        layers2 = [Layer(rho=200, h=100, E=100)]  # Double the modulus

        weak_layer = WeakLayer(rho=50, h=20)
        slab1 = Slab(layers1)
        slab2 = Slab(layers2)

        eig1 = Eigensystem(weak_layer, slab1)
        eig2 = Eigensystem(weak_layer, slab2)

        # Higher Young's modulus should lead to higher stiffnesses
        assert eig2.A11 > eig1.A11, "Higher E should increase extensional stiffness"
        assert eig2.D11 > eig1.D11, "Higher E should increase bending stiffness"

    def test_weak_layer_stiffness_influence(self):
        """Test that weak layer properties affect system behavior."""
        layers = [Layer(rho=250, h=120)]

        # Soft weak layer
        wl_soft = WeakLayer(rho=50, h=25, E=0.1)
        # Stiff weak layer
        wl_stiff = WeakLayer(rho=120, h=25, E=1.0)

        slab = Slab(layers)
        eig_soft = Eigensystem(wl_soft, slab)
        eig_stiff = Eigensystem(wl_stiff, slab)

        # Stiffness values should be different
        assert eig_soft.K[1, 0] != pytest.approx(
            eig_stiff.K[1, 0], abs=0.5 * 10 ** (-7)
        ), "Different weak layer properties should affect system matrix"

    def test_inclination_effect_on_loads(self):
        """Test that inclination affects load vectors correctly."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=50, h=20)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)

        # Compare load vectors for different inclinations
        q_flat = eigensystem.get_load_vector(phi=0.0, qs=0.0)
        q_inclined = eigensystem.get_load_vector(phi=30.0, qs=0.0)

        # Should be different for non-zero inclination
        assert not np.allclose(q_flat, q_inclined), (
            "Load vectors should differ for different inclinations"
        )

    def test_complementary_solution_continuity(self):
        """Test continuity of complementary solutions."""
        layers = [Layer(rho=200, h=100)]
        weak_layer = WeakLayer(rho=50, h=20)
        slab = Slab(layers)
        eigensystem = Eigensystem(weak_layer, slab)

        # Test continuity for bedded segments
        x1, x2 = 100.0, 100.000001  # Very close points
        length = 1000.0

        zh1 = eigensystem.zh(x1, length, True)
        zh2 = eigensystem.zh(x2, length, True)

        # Solutions should be very close for nearby points
        assert np.allclose(zh1, zh2, atol=1e-6), (
            "Complementary solutions should be continuous"
        )
