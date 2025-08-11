import unittest
import numpy as np

from weac_2.components import ScenarioConfig, Segment, WeakLayer, Layer
from weac_2.core.slab import Slab
from weac_2.core.scenario import Scenario
from weac_2.utils.misc import decompose_to_normal_tangential


class TestScenario(unittest.TestCase):
    def setUp(self):
        # Simple slab with a single layer
        self.layer = Layer(rho=200, h=100)
        self.slab = Slab([self.layer])
        # Weak layer with defaults (kn derived from properties)
        self.weak_layer = WeakLayer(rho=150, h=30)
        # Default two segments to test typical case
        self.segments_two = [
            Segment(length=400.0, has_foundation=True, m=75.0),
            Segment(length=600.0, has_foundation=True, m=0.0),
        ]
        # Config with non-zero angle and surface load to exercise load decomposition
        self.cfg = ScenarioConfig(
            phi=10.0, system_type="skiers", surface_load=2.5, crack_length=123.0
        )

    def test_init_sets_core_attributes(self):
        s = Scenario(self.cfg, self.segments_two, self.weak_layer, self.slab)
        self.assertEqual(s.system_type, self.cfg.system_type)
        self.assertAlmostEqual(s.phi, self.cfg.phi)
        self.assertAlmostEqual(s.surface_load, self.cfg.surface_load)
        # L is total length
        self.assertAlmostEqual(s.L, sum(seg.length for seg in self.segments_two))
        # crack_length is propagated
        self.assertAlmostEqual(s.crack_length, self.cfg.crack_length)

    def test_setup_scenario_multiple_segments(self):
        s = Scenario(self.cfg, self.segments_two, self.weak_layer, self.slab)
        # li is segment lengths
        np.testing.assert_allclose(s.li, np.array([400.0, 600.0]))
        # ki reflects foundation flags
        np.testing.assert_array_equal(s.ki, np.array([True, True]))
        # mi are masses at internal boundaries (all but last segment)
        np.testing.assert_allclose(s.mi, np.array([75.0]))
        # cumulative length
        np.testing.assert_allclose(s.cum_sum_li, np.array([400.0, 1000.0]))
        # get_segment_idx mapping across domains
        self.assertEqual(s.get_segment_idx(0.0), 0)
        self.assertEqual(s.get_segment_idx(399.9999), 0)
        # exactly on boundary goes to next bin
        self.assertEqual(s.get_segment_idx(400.0), 1)
        self.assertEqual(s.get_segment_idx(999.9999), 1)
        # vectorized
        np.testing.assert_array_equal(
            s.get_segment_idx(np.array([0.0, 100.0, 400.0, 500.0, 999.0])),
            np.array([0, 0, 1, 1, 1]),
        )
        # out of bounds (> L) raises
        with self.assertRaises(ValueError):
            s.get_segment_idx(1000.0001)

    def test_setup_scenario_single_segment_adds_dummy(self):
        segments_one = [Segment(length=750.0, has_foundation=True, m=0.0)]
        s = Scenario(self.cfg, segments_one, self.weak_layer, self.slab)
        # Dummy segment appended
        self.assertEqual(len(s.li), 2)
        self.assertAlmostEqual(s.li[0], 750.0)
        self.assertAlmostEqual(s.li[1], 0.0)
        self.assertTrue(bool(s.ki[1]))
        self.assertAlmostEqual(s.mi[-1], 0.0)
        # L equals the actual provided length
        self.assertAlmostEqual(s.L, 750.0)
        # get_segment_idx behavior at end
        self.assertEqual(s.get_segment_idx(749.9999), 0)
        # x == L is allowed and maps to bin 1
        self.assertEqual(s.get_segment_idx(750.0), 1)
        with self.assertRaises(ValueError):
            s.get_segment_idx(750.0001)

    def test_calc_normal_and_tangential_loads(self):
        s = Scenario(self.cfg, self.segments_two, self.weak_layer, self.slab)
        # Expected from decomposition of slab weight and surface load
        qwn, qwt = decompose_to_normal_tangential(self.slab.qw, self.cfg.phi)
        qsn, qst = decompose_to_normal_tangential(self.cfg.surface_load, self.cfg.phi)
        np.testing.assert_allclose(s.qn, qwn + qsn, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s.qt, qwt + qst, rtol=1e-12, atol=1e-12)
        # Sanity signs: qn positive (into slope), qt negative (downslope)
        self.assertGreater(s.qn, 0.0)
        self.assertLessEqual(s.qt, 0.0)

    def test_calc_crack_height(self):
        s = Scenario(self.cfg, self.segments_two, self.weak_layer, self.slab)
        expected_crack_h = self.weak_layer.collapse_height - s.qn / self.weak_layer.kn
        self.assertTrue(np.isfinite(expected_crack_h))
        self.assertAlmostEqual(s.crack_h, expected_crack_h)

    def test_refresh_from_config_updates_attributes_and_recomputes_crack_height_only(
        self,
    ):
        s = Scenario(self.cfg, self.segments_two, self.weak_layer, self.slab)
        old_qn = s.qn
        old_qt = s.qt
        old_crack_h = s.crack_h
        # Change config values
        s.scenario_config.phi = 25.0
        s.scenario_config.surface_load = 10.0
        s.scenario_config.system_type = "pst-"
        s.refresh_from_config()
        # Attributes copied from config
        self.assertEqual(s.system_type, "pst-")
        self.assertAlmostEqual(s.phi, 25.0)
        self.assertAlmostEqual(s.surface_load, 10.0)
        # Current implementation does not recalc qn/qt on refresh
        self.assertAlmostEqual(s.qn, old_qn)
        self.assertAlmostEqual(s.qt, old_qt)
        # Crack height recomputed using existing qn -> unchanged
        self.assertAlmostEqual(s.crack_h, old_crack_h)

    def test_refresh_recomputes_setup_when_segments_change(self):
        s = Scenario(self.cfg, self.segments_two, self.weak_layer, self.slab)
        # Mutate segments: change lengths and foundation flags
        new_segments = [
            Segment(length=100.0, has_foundation=True, m=0.0),
            Segment(length=200.0, has_foundation=False, m=0.0),
            Segment(length=300.0, has_foundation=True, m=0.0),
        ]
        s.segments = new_segments
        # refresh_from_config should call _setup_scenario and _calc_crack_height
        s.refresh_from_config()
        np.testing.assert_allclose(s.li, np.array([100.0, 200.0, 300.0]))
        np.testing.assert_array_equal(s.ki, np.array([True, False, True]))
        np.testing.assert_allclose(s.mi, np.array([0.0, 0.0]))
        np.testing.assert_allclose(s.cum_sum_li, np.array([100.0, 300.0, 600.0]))
        self.assertAlmostEqual(s.L, 600.0)


if __name__ == "__main__":
    unittest.main()
