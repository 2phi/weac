"""
This module contains tests for the SlabTouchdown class.
"""

import unittest
from unittest.mock import patch

import numpy as np

from weac.components import Layer, ScenarioConfig, Segment, WeakLayer
from weac.constants import STIFFNESS_COLLAPSE_FACTOR
from weac.core.eigensystem import Eigensystem
from weac.core.scenario import Scenario
from weac.core.slab import Slab
from weac.core.slab_touchdown import SlabTouchdown


class SlabTouchdownTestBase(unittest.TestCase):
    """Base class for SlabTouchdown tests, providing common setup."""

    def make_base_objects(self):
        """Make base objects for testing."""
        layers = [Layer(rho=220, h=120)]
        slab = Slab(layers)
        weak_layer = WeakLayer(rho=120, h=25)
        # Two segments: supported then unsupported, typical PST layout
        segments = [
            Segment(length=5e3, has_foundation=True, m=0.0),
            Segment(length=200.0, has_foundation=False, m=0.0),
        ]
        cfg = ScenarioConfig(
            phi=10.0, system_type="pst-", cut_length=200.0, surface_load=0.0
        )
        scenario = Scenario(cfg, segments, weak_layer, slab)
        eig = Eigensystem(weak_layer, slab)
        return scenario, eig


class TestSlabTouchdownInitialization(SlabTouchdownTestBase):
    """Test the initialization of the SlabTouchdown class."""

    def test_init_sets_flat_config_and_collapsed_eigensystem(self):
        """Test the initialization of the SlabTouchdown class."""
        scenario, eig = self.make_base_objects()
        with patch.object(SlabTouchdown, "_setup_touchdown_system", return_value=None):
            td = SlabTouchdown(scenario, eig)
        # flat_config has phi=0 and preserves other fields
        self.assertEqual(td.flat_config.phi, 0.0)
        self.assertEqual(
            td.flat_config.system_type, scenario.scenario_config.system_type
        )
        self.assertEqual(td.flat_config.cut_length, scenario.scenario_config.cut_length)
        self.assertEqual(
            td.flat_config.surface_load, scenario.scenario_config.surface_load
        )
        # collapsed weak layer stiffness scaled
        self.assertAlmostEqual(
            td.collapsed_weak_layer.kn,
            scenario.weak_layer.kn * STIFFNESS_COLLAPSE_FACTOR,
        )
        self.assertAlmostEqual(
            td.collapsed_weak_layer.kt,
            scenario.weak_layer.kt * STIFFNESS_COLLAPSE_FACTOR,
        )
        # collapsed eigensystem uses collapsed weak layer and same slab
        self.assertIs(td.collapsed_eigensystem.weak_layer, td.collapsed_weak_layer)
        self.assertIs(td.collapsed_eigensystem.slab, scenario.slab)


class TestSlabTouchdownBoundaries(SlabTouchdownTestBase):
    """Test the calculation of touchdown mode boundaries."""

    def test_calc_l_AB_root_exists_and_within_bounds(self):
        """Test the calculation of touchdown mode boundaries."""
        scenario, eig = self.make_base_objects()
        # Avoid heavy setup
        with patch.object(SlabTouchdown, "_setup_touchdown_system", return_value=None):
            td = SlabTouchdown(scenario, eig)
        # Make bs positive and control substitute stiffness to constants
        td.eigensystem.A11 = 100.0
        td.eigensystem.B11 = 1.0
        td.eigensystem.D11 = 100.0
        td.eigensystem.kA55 = 10.0
        with patch.object(td, "_substitute_stiffness", return_value=2.0):
            l_ab = td._calc_l_AB()  # pylint: disable=protected-access
        self.assertGreater(l_ab, 0.0)
        self.assertLess(l_ab, td.scenario.L)

    def test_calc_l_BC_root_exists_and_within_bounds(self):
        """Test the calculation of touchdown mode boundaries."""
        scenario, eig = self.make_base_objects()
        with patch.object(SlabTouchdown, "_setup_touchdown_system", return_value=None):
            td = SlabTouchdown(scenario, eig)
        # Make bs positive and control substitute stiffness to constants
        td.eigensystem.A11 = 100.0
        td.eigensystem.B11 = 1.0
        td.eigensystem.D11 = 100.0
        td.eigensystem.kA55 = 10.0
        with patch.object(td, "_substitute_stiffness", return_value=3.0):
            l_bc = td._calc_l_BC()  # pylint: disable=protected-access
        self.assertGreater(l_bc, 0.0)
        self.assertLess(l_bc, td.scenario.L)


class TestSlabTouchdownModeAndDistance(SlabTouchdownTestBase):
    """Test the calculation of touchdown mode and distance."""

    def test_calc_touchdown_mode_assigns_correct_mode(self):
        """Test the calculation of touchdown mode and distance."""
        scenario, eig = self.make_base_objects()
        with patch.object(SlabTouchdown, "_setup_touchdown_system", return_value=None):
            td = SlabTouchdown(scenario, eig)
        with (
            patch.object(td, "_calc_l_AB", return_value=300.0),
            patch.object(td, "_calc_l_BC", return_value=600.0),
        ):
            # Mode A: cut_length <= l_AB
            td.scenario.scenario_config.cut_length = 200.0
            td.scenario.cut_length = 200.0
            td._calc_touchdown_mode()  # pylint: disable=protected-access
            self.assertEqual(td.touchdown_mode, "A_free_hanging")
            # Mode B: l_AB < cut_length <= l_BC
            td.scenario.scenario_config.cut_length = 400.0
            td.scenario.cut_length = 400.0
            td._calc_touchdown_mode()  # pylint: disable=protected-access
            self.assertEqual(td.touchdown_mode, "B_point_contact")
            # Mode C: cut_length > l_BC
            td.scenario.scenario_config.cut_length = 800.0
            td.scenario.cut_length = 800.0
            td._calc_touchdown_mode()  # pylint: disable=protected-access
            self.assertEqual(td.touchdown_mode, "C_in_contact")

    def test_calc_touchdown_distance_sets_expected_values(self):
        """Test the calculation of touchdown mode and distance."""
        scenario, eig = self.make_base_objects()
        with patch.object(SlabTouchdown, "_setup_touchdown_system", return_value=None):
            td = SlabTouchdown(scenario, eig)
        # Mode A/B: equals cut_length
        td.touchdown_mode = "A_free_hanging"
        td.scenario.cut_length = 123.0
        td._calc_touchdown_distance()  # pylint: disable=protected-access
        self.assertEqual(td.touchdown_distance, 123.0)

        td.touchdown_mode = "B_point_contact"
        td.scenario.cut_length = 321.0
        td._calc_touchdown_distance()  # pylint: disable=protected-access
        self.assertEqual(td.touchdown_distance, 321.0)

        # Mode C: uses helper methods
        td.touchdown_mode = "C_in_contact"
        with (
            patch.object(td, "_calc_touchdown_distance_in_mode_C", return_value=111.0),
            patch.object(td, "_calc_collapsed_weak_layer_kR", return_value=222.0),
        ):
            td._calc_touchdown_distance()  # pylint: disable=protected-access
        self.assertEqual(td.touchdown_distance, 111.0)
        self.assertEqual(td.collapsed_weak_layer_kR, 222.0)


class TestSlabTouchdownHelpers(SlabTouchdownTestBase):
    """Test helper methods for the SlabTouchdown class."""

    def test_generate_straight_scenario(self):
        """Test the generation of a straight scenario."""
        scenario, eig = self.make_base_objects()
        with patch.object(SlabTouchdown, "_setup_touchdown_system", return_value=None):
            td = SlabTouchdown(scenario, eig)
        L = 555.5
        straight = td._generate_straight_scenario(L)  # pylint: disable=protected-access
        self.assertAlmostEqual(straight.L, L)
        self.assertEqual(straight.phi, 0.0)
        # First segment should be the provided one, dummy appended internally
        self.assertGreaterEqual(len(straight.li), 1)
        self.assertTrue(bool(straight.ki[0]))

    def test_create_collapsed_eigensystem_scales_weak_layer(self):
        """Test the creation of a collapsed eigensystem."""
        scenario, eig = self.make_base_objects()
        with patch.object(SlabTouchdown, "_setup_touchdown_system", return_value=None):
            td = SlabTouchdown(scenario, eig)
        # Recreate to test method in isolation
        collapsed = td._create_collapsed_eigensystem()  # pylint: disable=protected-access
        self.assertAlmostEqual(
            collapsed.weak_layer.kn, scenario.weak_layer.kn * STIFFNESS_COLLAPSE_FACTOR
        )
        self.assertAlmostEqual(
            collapsed.weak_layer.kt, scenario.weak_layer.kt * STIFFNESS_COLLAPSE_FACTOR
        )

    def test_calc_touchdown_distance_in_mode_C_root_in_range(self):
        """Test the calculation of touchdown mode and distance."""
        scenario, eig = self.make_base_objects()
        scenario.scenario_config.cut_length = 300.0
        scenario.cut_length = 300.0
        with patch.object(SlabTouchdown, "_setup_touchdown_system", return_value=None):
            td = SlabTouchdown(scenario, eig)
        # Make bs positive and control substitute stiffness values by inspecting args
        td.eigensystem.A11 = 100.0
        td.eigensystem.B11 = 1.0
        td.eigensystem.D11 = 100.0
        td.eigensystem.kA55 = 10.0

        def fake_subst(straight_scenario, es, dof):  # pylint: disable=unused-argument
            """Fake substitute stiffness."""
            # Return different constants for original vs collapsed eigensystem
            if es is td.eigensystem:
                return 2.0  # kRl or kNl
            if es is td.collapsed_eigensystem:
                return 5.0  # kRr
            return 3.0

        with patch.object(td, "_substitute_stiffness", side_effect=fake_subst):
            d = td._calc_touchdown_distance_in_mode_C()  # pylint: disable=protected-access

        self.assertGreater(d, 0.0)
        self.assertLess(d, scenario.cut_length)

    def test_calc_collapsed_weak_layer_kR_returns_positive(self):
        """Test the calculation of collapsed weak layer stiffness."""
        scenario, eig = self.make_base_objects()
        with patch.object(SlabTouchdown, "_setup_touchdown_system", return_value=None):
            td = SlabTouchdown(scenario, eig)
        td.touchdown_mode = "A_free_hanging"
        td.touchdown_distance = 100.0
        with patch.object(td, "_substitute_stiffness", return_value=7.5):
            kR = td._calc_collapsed_weak_layer_kR()  # pylint: disable=protected-access
        self.assertGreater(kR, 0.0)
        self.assertAlmostEqual(kR, 7.5)

    def test_substitute_stiffness_rot_and_trans_are_finite(self):
        """Test the calculation of substitute stiffness."""
        scenario, eig = self.make_base_objects()
        # Avoid running setup (roots) and use method directly
        with patch.object(SlabTouchdown, "_setup_touchdown_system", return_value=None):
            td = SlabTouchdown(scenario, eig)
        # Use a small, straight scenario to compute substitute stiffness
        straight = td._generate_straight_scenario(L=400.0)  # pylint: disable=protected-access
        kR = td._substitute_stiffness(straight, td.eigensystem, dof="rot")  # pylint: disable=protected-access
        kN = td._substitute_stiffness(straight, td.eigensystem, dof="trans")  # pylint: disable=protected-access
        self.assertTrue(np.isfinite(kR))
        self.assertTrue(np.isfinite(kN))
        self.assertGreater(kR, 0.0)
        self.assertGreater(kN, 0.0)

    def test_setup_touchdown_system_calls_subroutines(self):
        """Test the setup of the touchdown system."""
        scenario, eig = self.make_base_objects()
        with (
            patch.object(
                SlabTouchdown, "_calc_touchdown_mode", return_value=None
            ) as m1,
            patch.object(
                SlabTouchdown, "_calc_touchdown_distance", return_value=None
            ) as m2,
        ):
            SlabTouchdown(scenario, eig)
            # The constructor calls _setup_touchdown_system which should call both
            self.assertTrue(m1.called)
            self.assertTrue(m2.called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
