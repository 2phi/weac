"""
This module contains tests for the Scenario class.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from weac.components import Layer, ScenarioConfig, Segment, WeakLayer
from weac.core.scenario import Scenario
from weac.core.slab import Slab
from weac.utils.misc import decompose_to_xyz


@pytest.fixture
def scenario_parts():
    """Shared building blocks for Scenario tests."""
    layer = Layer(rho=200, h=100)
    slab = Slab([layer])
    weak_layer = WeakLayer(rho=150, h=30)
    segments_two = [
        Segment(length=400.0, has_foundation=True, m=75.0),
        Segment(length=600.0, has_foundation=True, m=0.0),
    ]
    cfg = ScenarioConfig(
        phi=10.0, system_type="skiers", surface_load=0.2, cut_length=123.0
    )
    return SimpleNamespace(
        layer=layer,
        slab=slab,
        weak_layer=weak_layer,
        segments_two=segments_two,
        cfg=cfg,
    )


class TestScenario:
    """Test the Scenario class."""

    def test_init_sets_core_attributes(self, scenario_parts):
        """Test that init sets core attributes correctly."""
        s = Scenario(
            scenario_parts.cfg,
            scenario_parts.segments_two,
            scenario_parts.weak_layer,
            scenario_parts.slab,
        )
        assert s.system_type == scenario_parts.cfg.system_type
        assert s.phi == pytest.approx(
            scenario_parts.cfg.phi, abs=0.5 * 10 ** (-7)
        )
        assert s.surface_load == pytest.approx(
            scenario_parts.cfg.surface_load, abs=0.5 * 10 ** (-7)
        )
        # L is total length
        assert s.L == pytest.approx(
            sum(seg.length for seg in scenario_parts.segments_two),
            abs=0.5 * 10 ** (-7),
        )
        # cut_length is propagated
        assert s.cut_length == pytest.approx(
            scenario_parts.cfg.cut_length, abs=0.5 * 10 ** (-7)
        )

    def test_setup_scenario_multiple_segments(self, scenario_parts):
        """Test that setup_scenario sets up correctly for multiple segments."""
        s = Scenario(
            scenario_parts.cfg,
            scenario_parts.segments_two,
            scenario_parts.weak_layer,
            scenario_parts.slab,
        )
        # li is segment lengths
        np.testing.assert_allclose(s.li, np.array([400.0, 600.0]))
        # ki reflects foundation flags
        np.testing.assert_array_equal(s.ki, np.array([True, True]))
        # mi are masses at internal boundaries (all but last segment)
        np.testing.assert_allclose(s.mi, np.array([75.0]))
        # cumulative length
        np.testing.assert_allclose(s.cum_sum_li, np.array([400.0, 1000.0]))
        # get_segment_idx mapping across domains
        assert s.get_segment_idx(0.0) == 0
        assert s.get_segment_idx(399.9999) == 0
        # exactly on boundary goes to next bin
        assert s.get_segment_idx(400.0) == 1
        assert s.get_segment_idx(999.9999) == 1
        # vectorized
        np.testing.assert_array_equal(
            s.get_segment_idx(np.array([0.0, 100.0, 400.0, 500.0, 999.0])),
            np.array([0, 0, 1, 1, 1]),
        )
        # out of bounds (> L) raises
        with pytest.raises(ValueError, match=r"out of bounds|exceeds|beyond"):
            s.get_segment_idx(1000.0001)

    def test_setup_scenario_single_segment_adds_dummy(self, scenario_parts):
        """Test that setup_scenario adds a dummy segment for single segment case."""
        segments_one = [Segment(length=750.0, has_foundation=True, m=0.0)]
        s = Scenario(
            scenario_parts.cfg,
            segments_one,
            scenario_parts.weak_layer,
            scenario_parts.slab,
        )
        # Dummy segment appended
        assert len(s.li) == 2
        assert s.li[0] == pytest.approx(750.0, abs=0.5 * 10 ** (-7))
        assert s.li[1] == pytest.approx(0.0, abs=0.5 * 10 ** (-7))
        assert bool(s.ki[1])
        assert s.mi[-1] == pytest.approx(0.0, abs=0.5 * 10 ** (-7))
        # L equals the actual provided length
        assert s.L == pytest.approx(750.0, abs=0.5 * 10 ** (-7))
        # get_segment_idx behavior at end
        assert s.get_segment_idx(749.9999) == 0
        # x == L is allowed and maps to bin 1
        assert s.get_segment_idx(750.0) == 1
        with pytest.raises(ValueError, match=r"out of bounds|exceeds|beyond"):
            s.get_segment_idx(750.0001)

    def test_calc_normal_and_tangential_loads(self, scenario_parts):
        """Test that calc_normal_and_tangential_loads computes expected loads."""
        s = Scenario(
            scenario_parts.cfg,
            scenario_parts.segments_two,
            scenario_parts.weak_layer,
            scenario_parts.slab,
        )
        # Expected from decomposition of slab weight and surface load
        qwt, _, qwn = decompose_to_xyz(scenario_parts.slab.qw, scenario_parts.cfg.phi)
        qst, _, qsn = decompose_to_xyz(
            scenario_parts.cfg.surface_load, scenario_parts.cfg.phi
        )
        np.testing.assert_allclose(s.qz, qwn + qsn, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s.qx, qwt + qst, rtol=1e-12, atol=1e-12)
        # Sanity signs: qz positive (into slope), qx negative (downslope)
        assert s.qz > 0.0
        assert s.qx <= 0.0

    def test_calc_crack_height(self, scenario_parts):
        """Test that calc_crack_height computes expected crack height."""
        s = Scenario(
            scenario_parts.cfg,
            scenario_parts.segments_two,
            scenario_parts.weak_layer,
            scenario_parts.slab,
        )
        expected_crack_h = (
            scenario_parts.weak_layer.collapse_height
            - s.qz / scenario_parts.weak_layer.kn
        )
        assert np.isfinite(expected_crack_h)
        assert s.crack_h == pytest.approx(expected_crack_h, abs=0.5 * 10 ** (-7))

    def test_refresh_from_config_updates_attributes(self, scenario_parts):
        """Test that refresh_from_config updates attributes."""
        s = Scenario(
            scenario_parts.cfg,
            scenario_parts.segments_two,
            scenario_parts.weak_layer,
            scenario_parts.slab,
        )
        # Change config values
        s.scenario_config.phi = 25.0
        s.scenario_config.surface_load = 0.2
        s.scenario_config.system_type = "pst-"
        s.refresh_from_config()
        # Attributes copied from config
        assert s.system_type == "pst-"
        assert s.phi == pytest.approx(25.0, abs=0.5 * 10 ** (-7))
        assert s.surface_load == pytest.approx(0.2, abs=0.5 * 10 ** (-7))

    def test_refresh_recomputes_setup_when_segments_change(self, scenario_parts):
        """Test that refresh_from_config recomputes setup when segments change."""
        s = Scenario(
            scenario_parts.cfg,
            scenario_parts.segments_two,
            scenario_parts.weak_layer,
            scenario_parts.slab,
        )
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
        assert s.L == pytest.approx(600.0, abs=0.5 * 10 ** (-7))
