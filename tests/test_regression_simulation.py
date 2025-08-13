import unittest

import numpy as np

from weac.analysis import CriteriaEvaluator
from weac.components import (
    Config,
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac.core.system_model import SystemModel


class TestRegressionSimulation(unittest.TestCase):
    """Regression tests asserting stable outputs for key scenarios."""

    def test_skier_baseline(self):
        layers = [Layer(rho=200, h=150)]
        wl = WeakLayer(rho=150, h=10)
        segs = [
            Segment(length=10000, has_foundation=True, m=80),
            Segment(length=4000, has_foundation=True, m=0),
        ]
        sc = ScenarioConfig(phi=10.0, system_type="skier", cut_length=0)
        mi = ModelInput(layers=layers, weak_layer=wl, segments=segs, scenario_config=sc)
        sm = SystemModel(model_input=mi, config=Config(touchdown=False))

        C = sm.unknown_constants

        # Baseline captured values (shape 6x2)
        expected = np.array(
            [
                [1.077301285647e-02, -1.278718341225e-11],
                [1.306660341145e-25, -1.860324883076e-02],
                [-1.949176767846e-26, 4.302301809624e-02],
                [-1.975734506280e-02, 1.802664410514e-12],
                [5.557284761724e-27, -1.898878164007e-02],
                [3.605266766554e-02, 8.274691619617e-13],
            ]
        )

        self.assertEqual(C.shape, expected.shape)
        np.testing.assert_allclose(C, expected, rtol=1e-6, atol=1e-8)

    def test_skiers_baseline(self):
        layers = [Layer(rho=200, h=150)]
        wl = WeakLayer()
        segs = [
            Segment(length=5e3, has_foundation=True, m=30.0),
            Segment(length=2000, has_foundation=True, m=35.0),
            Segment(length=5e3, has_foundation=True, m=0.0),
        ]
        sc = ScenarioConfig(phi=10.0, system_type="skiers", cut_length=0.0)
        mi = ModelInput(layers=layers, weak_layer=wl, segments=segs, scenario_config=sc)
        sm = SystemModel(model_input=mi, config=Config(touchdown=False))
        C = sm.unknown_constants

        expected = np.array(
            [
                [-4.088162010358e-03, -4.764174602231e-03, 3.408538076878e-10],
                [1.191472990454e-10, -1.001629823457e-02, -1.169531830633e-02],
                [-1.010395028771e-02, 2.526460884175e-02, -8.035562290509e-12],
                [-2.139647386757e-11, 3.668451190769e-02, 4.279859722781e-02],
                [-3.695151762335e-02, -3.686646408552e-02, -6.269554006981e-11],
                [-5.511146253945e-12, 3.950748621493e-03, 4.609206726858e-03],
            ]
        )

        self.assertEqual(C.shape, expected.shape)
        np.testing.assert_allclose(C, expected, rtol=1e-10, atol=1e-12)

    def test_pst_without_touchdown_baseline(self):
        layers = [Layer(rho=200, h=150), Layer(rho=300, h=100)]
        wl = WeakLayer(rho=170, h=20)
        segs = [
            Segment(length=10000, has_foundation=True, m=0),
            Segment(length=4000, has_foundation=False, m=0),
        ]
        sc = ScenarioConfig(phi=30.0, system_type="pst-", cut_length=4000)
        mi = ModelInput(layers=layers, weak_layer=wl, segments=segs, scenario_config=sc)
        sm = SystemModel(model_input=mi, config=Config(touchdown=False))

        C = sm.unknown_constants

        expected = np.array(
            [
                [-1.048702730641e00, 1.712797455469e00],
                [9.314583991285e-04, 2.931185753374e-02],
                [2.660951120765e00, 8.896908397628e-05],
                [3.091099845912e-03, -1.493044031727e-08],
                [-2.476037598677e00, 2.077316283914e00],
                [-1.326212845668e-03, 8.697324037316e-03],
            ]
        )

        self.assertEqual(C.shape, expected.shape)
        np.testing.assert_allclose(C, expected, rtol=1e-10, atol=1e-12)

    def test_pst_with_touchdown_baseline(self):
        layers = [Layer(rho=200, h=150), Layer(rho=300, h=100)]
        wl = WeakLayer(rho=50, h=20, E=0.35, nu=0.1)
        segs = [
            Segment(length=10000, has_foundation=True, m=0),
            Segment(length=4000, has_foundation=False, m=0),
        ]
        sc = ScenarioConfig(phi=30.0, system_type="pst-", cut_length=4000)
        mi = ModelInput(layers=layers, weak_layer=wl, segments=segs, scenario_config=sc)
        sm = SystemModel(model_input=mi, config=Config(touchdown=True))

        td = sm.slab_touchdown
        C = sm.unknown_constants

        # Touchdown mode and distance baselines
        self.assertEqual(td.touchdown_mode, "C_in_contact")
        self.assertAlmostEqual(td.touchdown_distance, 1577.2698088929287, places=6)

        # Scenario segments updated by touchdown length
        seg_lengths = np.array([seg.length for seg in sm.scenario.segments])
        np.testing.assert_allclose(
            seg_lengths, np.array([10000.0, 1577.269808892929]), rtol=1e-12, atol=1e-12
        )

        expected = np.array(
            [
                [-1.530083342282e-03, 4.529393405710e-01],
                [-1.232210460299e-01, 2.790068096799e-03],
                [5.074156205051e-01, 3.550123902347e-06],
                [1.634883713190e-02, -3.868724171529e-09],
                [-1.895302012103e-01, -3.887063412519e-02],
                [-1.845836424067e-03, 1.818424547898e-04],
            ]
        )

        self.assertEqual(C.shape, expected.shape)
        np.testing.assert_allclose(C, expected, rtol=1e-10, atol=1e-12)

    def test_criteria_evaluator_regressions(self):
        layers = [Layer(rho=170, h=100), Layer(rho=230, h=130)]
        wl = WeakLayer(rho=180, h=20)
        segs = [Segment(length=10000, has_foundation=True, m=0)]
        sc = ScenarioConfig(phi=30.0, system_type="skier", cut_length=0.0)
        mi = ModelInput(layers=layers, weak_layer=wl, segments=segs, scenario_config=sc)
        sm = SystemModel(model_input=mi, config=Config(touchdown=False))

        evaluator = CriteriaEvaluator(CriteriaConfig())

        # find_minimum_force baseline
        fm = evaluator.find_minimum_force(system=sm, tolerance_stress=0.005)
        self.assertTrue(fm.success)
        self.assertGreater(fm.critical_skier_weight, 0)
        # Baseline values recorded
        self.assertAlmostEqual(fm.critical_skier_weight, 68.504569930, places=6)
        self.assertAlmostEqual(fm.max_dist_stress, 1.0000189267255666, places=6)
        self.assertLess(fm.min_dist_stress, 1.0)

        # evaluate_SSERR baseline
        ss = evaluator.evaluate_SSERR(system=sm, vertical=False)
        self.assertTrue(ss.converged)
        self.assertGreater(ss.touchdown_distance, 0)
        # Baseline values recorded
        self.assertAlmostEqual(ss.touchdown_distance, 1320.108936137, places=6)
        np.testing.assert_allclose(ss.SSERR, 2.168112101045914, rtol=1e-8, atol=0)

        # evaluate_coupled_criterion baseline
        cc = evaluator.evaluate_coupled_criterion(system=sm, max_iterations=10)
        self.assertIsNotNone(cc)
        self.assertIsInstance(cc.critical_skier_weight, float)
        self.assertIsInstance(cc.crack_length, float)
        # Baseline values recorded
        self.assertTrue(cc.converged)
        np.testing.assert_allclose(
            cc.critical_skier_weight, 183.40853553646807, rtol=1e-2
        )
        np.testing.assert_allclose(cc.crack_length, 119.58600407185531, rtol=1e-2)
        np.testing.assert_allclose(cc.g_delta, 1.0, rtol=1e-2)
        np.testing.assert_allclose(cc.dist_ERR_envelope, 0.0, atol=1e-2)

        # find_minimum_crack_length baseline (returns crack length > 0)
        crack_len, new_segments = evaluator.find_minimum_crack_length(system=sm)
        self.assertGreater(crack_len, 0)
        self.assertTrue(all(isinstance(s, Segment) for s in new_segments))
        # Baseline value recorded
        np.testing.assert_allclose(crack_len, 1582.87791111003, rtol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
