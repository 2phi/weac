"""
Unit tests for utility functions.

Tests force decomposition, skier load calculations, and other utility functions.
"""
import unittest
import numpy as np

from weac_2.utils import decompose_to_normal_tangential, get_skier_point_load
from weac_2.constants import G_MM_S2, LSKI_MM


class TestForceDecomposition(unittest.TestCase):
    """Test the decompose_to_normal_tangential function."""
    
    def test_flat_surface_decomposition(self):
        """Test force decomposition on flat surface (phi=0)."""
        f = 100.0  # Vertical force
        phi = 0.0  # Flat surface
        
        f_norm, f_tan = decompose_to_normal_tangential(f, phi)
        
        # On flat surface, normal component equals original force, tangential is zero
        self.assertAlmostEqual(f_norm, f, places=10, 
                              msg="Normal component should equal original force on flat surface")
        self.assertAlmostEqual(f_tan, 0.0, places=10,
                              msg="Tangential component should be zero on flat surface")
        
    def test_vertical_surface_decomposition(self):
        """Test force decomposition on vertical surface (phi=90)."""
        f = 100.0  # Vertical force
        phi = 90.0  # Vertical surface
        
        f_norm, f_tan = decompose_to_normal_tangential(f, phi)
        
        # On vertical surface, normal component is zero, tangential equals original force
        self.assertAlmostEqual(f_norm, 0.0, places=10,
                              msg="Normal component should be zero on vertical surface")
        self.assertAlmostEqual(f_tan, -f, places=10,
                              msg="Tangential component should equal negative original force")
        
    def test_45_degree_decomposition(self):
        """Test force decomposition on 45-degree surface."""
        f = 100.0  # Vertical force
        phi = 45.0  # 45-degree surface
        
        f_norm, f_tan = decompose_to_normal_tangential(f, phi)
        
        # On 45-degree surface, both components should be equal in magnitude
        expected_component = f / np.sqrt(2)
        self.assertAlmostEqual(abs(f_norm), expected_component, places=8,
                              msg="Normal component magnitude should be f/√2 for 45° surface")
        self.assertAlmostEqual(abs(f_tan), expected_component, places=8,
                              msg="Tangential component magnitude should be f/√2 for 45° surface")
        
        # Check signs: normal should be positive (into slope), tangential negative (downslope)
        self.assertGreater(f_norm, 0, "Normal component should be positive (into slope)")
        self.assertLess(f_tan, 0, "Tangential component should be negative (downslope)")
        
    def test_30_degree_decomposition(self):
        """Test force decomposition on 30-degree surface."""
        f = 100.0  # Vertical force
        phi = 30.0  # 30-degree surface
        
        f_norm, f_tan = decompose_to_normal_tangential(f, phi)
        
        # Known analytical values for 30 degrees
        expected_norm = f * np.cos(np.deg2rad(30))  # f * cos(30°) = f * √3/2
        expected_tan = -f * np.sin(np.deg2rad(30))  # -f * sin(30°) = -f/2
        
        self.assertAlmostEqual(f_norm, expected_norm, places=10)
        self.assertAlmostEqual(f_tan, expected_tan, places=10)
        
    def test_negative_angles(self):
        """Test force decomposition with negative angles."""
        f = 100.0  # Vertical force
        phi = -30.0  # Negative angle (surface slopes down in +x direction)
        
        f_norm, f_tan = decompose_to_normal_tangential(f, phi)
        
        # Normal component should still be positive (into slope)
        # Tangential component should be positive (upslope for negative angle)
        self.assertGreater(f_norm, 0, "Normal component should be positive")
        self.assertGreater(f_tan, 0, "Tangential component should be positive for negative angle")
        
    def test_zero_force(self):
        """Test force decomposition with zero force."""
        f = 0.0
        phi = 30.0
        
        f_norm, f_tan = decompose_to_normal_tangential(f, phi)
        
        self.assertEqual(f_norm, 0.0, "Zero force should give zero normal component")
        self.assertEqual(f_tan, 0.0, "Zero force should give zero tangential component")
        
    def test_energy_conservation(self):
        """Test that force decomposition conserves energy (magnitude)."""
        f = 150.0
        phi = 37.0  # Arbitrary angle
        
        f_norm, f_tan = decompose_to_normal_tangential(f, phi)
        
        # Total magnitude should be conserved: f² = f_norm² + f_tan²
        original_magnitude_squared = f**2
        decomposed_magnitude_squared = f_norm**2 + f_tan**2
        
        self.assertAlmostEqual(original_magnitude_squared, decomposed_magnitude_squared, places=10,
                              msg="Force magnitude should be conserved in decomposition")


class TestSkierPointLoad(unittest.TestCase):
    """Test the get_skier_point_load function."""
    
    def test_skier_load_calculation(self):
        """Test basic skier load calculation."""
        m = 70.0  # 70 kg skier
        
        F = get_skier_point_load(m)
        
        # Expected calculation: F = 1e-3 * m * G_MM_S2 / LSKI_MM
        expected_F = 1e-3 * m * G_MM_S2 / LSKI_MM
        
        self.assertAlmostEqual(F, expected_F, places=10,
                              msg="Skier load should match expected calculation")
        
    def test_skier_load_units(self):
        """Test that skier load has correct units."""
        m = 80.0  # kg
        F = get_skier_point_load(m)
        
        # Result should be in N/mm (force per unit length)
        # For typical values, this should be a small positive number
        self.assertGreater(F, 0, "Skier load should be positive")
        self.assertLess(F, 1, "Skier load should be reasonable magnitude (< 1 N/mm)")
        
    def test_zero_mass_skier(self):
        """Test skier load calculation with zero mass."""
        m = 0.0
        F = get_skier_point_load(m)
        
        self.assertEqual(F, 0.0, "Zero mass should give zero load")
        
    def test_heavy_skier(self):
        """Test skier load calculation with heavy skier."""
        m = 120.0  # Heavy skier
        F = get_skier_point_load(m)
        
        # Should be positive and larger than for lighter skier
        m_light = 60.0
        F_light = get_skier_point_load(m_light)
        
        self.assertGreater(F, F_light, "Heavier skier should produce larger load")
        self.assertAlmostEqual(F / F_light, m / m_light, places=10,
                              msg="Load should scale linearly with mass")
        
    def test_skier_load_scaling(self):
        """Test that skier load scales linearly with mass."""
        masses = [50, 75, 100, 125]  # Different skier masses
        loads = [get_skier_point_load(m) for m in masses]
        
        # Check linear scaling
        for i in range(1, len(masses)):
            ratio_mass = masses[i] / masses[0]
            ratio_load = loads[i] / loads[0]
            self.assertAlmostEqual(ratio_mass, ratio_load, places=10,
                                  msg=f"Load should scale linearly: mass ratio {ratio_mass}, load ratio {ratio_load}")


class TestUtilityFunctionConsistency(unittest.TestCase):
    """Test consistency and edge cases for utility functions."""
    
    def test_decomposition_symmetry(self):
        """Test that force decomposition is symmetric for opposite angles."""
        f = 100.0
        phi = 25.0
        
        f_norm_pos, f_tan_pos = decompose_to_normal_tangential(f, phi)
        f_norm_neg, f_tan_neg = decompose_to_normal_tangential(f, -phi)
        
        # Normal components should be equal
        self.assertAlmostEqual(f_norm_pos, f_norm_neg, places=10,
                              msg="Normal components should be equal for ±φ")
        
        # Tangential components should be opposite
        self.assertAlmostEqual(f_tan_pos, -f_tan_neg, places=10,
                              msg="Tangential components should be opposite for ±φ")
        
    def test_large_angles(self):
        """Test force decomposition for large angles."""
        f = 100.0
        
        # Test beyond 90 degrees
        phi = 120.0
        f_norm, f_tan = decompose_to_normal_tangential(f, phi)
        
        # At 120°, normal component should be negative (surface leans over)
        # and tangential component should be negative (large downslope)
        self.assertLess(f_norm, 0, "Normal component should be negative for obtuse angles")
        self.assertLess(f_tan, 0, "Tangential component should be negative")
        
    def test_angle_bounds(self):
        """Test force decomposition at angle boundaries."""
        f = 100.0
        
        # Test at exactly 0°
        f_norm, f_tan = decompose_to_normal_tangential(f, 0.0)
        self.assertAlmostEqual(f_norm, f, places=15)
        self.assertAlmostEqual(f_tan, 0.0, places=15)
        
        # Test at exactly 90° (expect some floating-point precision issues)
        f_norm, f_tan = decompose_to_normal_tangential(f, 90.0)
        self.assertAlmostEqual(f_norm, 0.0, places=10)  # Reduced precision for 90° case
        self.assertAlmostEqual(f_tan, -f, places=15)
        
    def test_force_decomposition_with_arrays(self):
        """Test that functions work with array inputs (if applicable)."""
        # This tests if the functions can handle numpy arrays
        masses = np.array([60.0, 70.0, 80.0])
        
        # Should work with array input
        try:
            loads = get_skier_point_load(masses)
            self.assertEqual(len(loads), len(masses), "Should handle array input")
            
            # Check that each element is calculated correctly
            for i, m in enumerate(masses):
                expected = get_skier_point_load(m)
                self.assertAlmostEqual(loads[i], expected, places=10)
                
        except (TypeError, AttributeError):
            # If function doesn't support arrays, that's fine too
            pass


class TestPhysicalReasonableness(unittest.TestCase):
    """Test that utility functions produce physically reasonable results."""
    
    def test_typical_skier_loads(self):
        """Test that typical skier loads are in reasonable ranges."""
        # Typical skier masses
        typical_masses = [50, 70, 90, 110]  # kg
        
        for m in typical_masses:
            F = get_skier_point_load(m)
            
            # Load should be positive but not huge
            self.assertGreater(F, 0, f"Load should be positive for {m} kg skier")
            self.assertLess(F, 10, f"Load should be reasonable for {m} kg skier")
            
            # Rough sanity check: load should be on order of mg/length
            # where length is ski contact length
            rough_estimate = m * 9.81 / 1000  # Very rough estimate in N/mm
            self.assertLess(F, 10 * rough_estimate, "Load should be reasonable compared to weight")
            
    def test_typical_force_decompositions(self):
        """Test force decomposition for typical avalanche slopes."""
        f = 100.0  # Typical force
        typical_angles = [25, 30, 35, 40, 45]  # Typical avalanche slope angles
        
        for phi in typical_angles:
            f_norm, f_tan = decompose_to_normal_tangential(f, phi)
            
            # Both components should be significant but less than original force
            self.assertGreater(abs(f_norm), 0, f"Normal component should be non-zero at {phi}°")
            self.assertGreater(abs(f_tan), 0, f"Tangential component should be non-zero at {phi}°")
            self.assertLess(abs(f_norm), f, f"Normal component should be less than total at {phi}°")
            self.assertLess(abs(f_tan), f, f"Tangential component should be less than total at {phi}°")


if __name__ == "__main__":
    unittest.main(verbosity=2) 