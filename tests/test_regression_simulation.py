"""
This module contains regression tests for the WEAC model.
"""

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

GT_skier_baseline = np.array(
    [
        [
            -1.3311587133616033e-03,
            -1.3311587133987555e-03,
            -1.4922878538805329e-02,
            -1.4922878538805305e-02,
            -1.3316416781406679e-03,
            -1.3311587133616033e-03,
        ],
        [
            -1.3400532113402682e-27,
            -1.9609062333698352e-16,
            -8.8088543943750638e-05,
            1.8243392275606253e-05,
            2.5491108889889770e-09,
            1.3113971286963517e-13,
        ],
        [
            1.2028124616334202e-03,
            1.2028124616361854e-03,
            4.2336109897242152e-02,
            4.2336109897242159e-02,
            1.2027765147493792e-03,
            1.2028124616334202e-03,
        ],
        [
            4.4863892018696710e-28,
            1.4594950179586782e-17,
            9.0840725538762377e-04,
            -1.0213155501342633e-03,
            1.8972934933226463e-10,
            4.3904509669894562e-14,
        ],
        [
            1.0207865877058275e-05,
            1.0207865877358878e-05,
            2.0858241860062231e-04,
            2.0858241860062263e-04,
            1.0211773622223890e-05,
            1.0207865877058275e-05,
        ],
        [
            9.3082770992463219e-30,
            1.5866005526363208e-18,
            5.7089479049104315e-06,
            1.4556704561361483e-06,
            -2.0625263341890901e-11,
            -9.1092262290486623e-16,
        ],
    ]
)

GT_skiers_baseline = np.array(
    [
        [
            -3.3364140411700502e-03,
            -3.3371039610692352e-03,
            -1.0211953916849679e-02,
            -1.0211953916849772e-02,
            -3.7930081429868277e-03,
            -1.1362149028450508e-02,
            -1.1362149028450560e-02,
            -3.3383877478897019e-03,
            -3.3364140411700502e-03,
        ],
        [
            -8.0289180784556896e-13,
            -2.3962146278368322e-09,
            -3.5438765390651617e-05,
            4.4357106916068844e-06,
            5.6324248362093287e-07,
            -4.1293393719283317e-05,
            5.2268283766852303e-06,
            6.8550347830960343e-09,
            2.2968941139093848e-12,
        ],
        [
            5.3656877671703247e-03,
            5.3657501774765836e-03,
            4.0999377862256728e-02,
            4.0999377862256728e-02,
            5.3516089951323098e-03,
            4.6936976212589937e-02,
            4.6936976212589923e-02,
            5.3655092252078438e-03,
            5.3656877671703247e-03,
        ],
        [
            2.1476913299692529e-13,
            2.1676209355807275e-10,
            3.2479067052872183e-04,
            -3.9885538154198533e-04,
            1.4280212737807196e-07,
            3.7892379106187288e-04,
            -4.6532993635395239e-04,
            6.2010795683549551e-10,
            6.1440651481267745e-13,
        ],
        [
            1.0845857494374418e-05,
            1.0848064160073291e-05,
            9.1766771806106752e-05,
            9.1766771806106942e-05,
            1.2307527822099668e-05,
            1.0526725389866414e-04,
            1.0526725389866431e-04,
            1.0852170272287989e-05,
            1.0845857494374418e-05,
        ],
        [
            1.1022164968036404e-15,
            7.6641418314537503e-12,
            3.3164846239000650e-06,
            1.7215055806097094e-06,
            -1.7298852781935918e-09,
            3.8690662777605046e-06,
            2.0082573939217563e-06,
            -2.1925402470511338e-11,
            -3.1531951864791889e-15,
        ],
    ]
)

GT_pst_without_touchdown = np.array(
    [
        [
            -7.2487996383562396e-03,
            -6.0196423568498235e-03,
            2.0773162839138180e00,
            2.0773162839138175e00,
            1.2130315043983948e01,
            1.3485989766738559e01,
        ],
        [
            -8.4703294725430034e-22,
            5.0708000603491068e-10,
            8.6973240373155250e-03,
            8.6973240373155267e-03,
            2.1039215467303948e-03,
            1.7347234759768071e-18,
        ],
        [
            5.2190784110483475e-03,
            2.4392769285311888e-03,
            1.7127974554689163e00,
            1.7127974554689156e00,
            3.1178068254972919e02,
            8.2709909746257256e02,
        ],
        [
            -3.1911617258468120e-05,
            -3.9755915991866683e-11,
            2.9311857533740264e-02,
            2.9311857533740261e-02,
            2.3604562295124668e-01,
            2.6458510067192831e-01,
        ],
        [
            3.1911617258468134e-05,
            1.8113788874495151e-05,
            -2.8287378556700056e-02,
            -2.8287378556700049e-02,
            -2.3553338346272659e-01,
            -2.6458510067192831e-01,
        ],
        [
            5.0398620458123951e-24,
            -1.2082657686176822e-12,
            -1.7819428769682468e-04,
            -1.7819428769682468e-04,
            -4.4063073869004441e-05,
            0.0000000000000000e00,
        ],
    ]
)

GT_pst_with_touchdown = np.array(
    [
        [
            -4.3146866755634006e-02,
            -3.9757397730484006e-02,
            -3.8870634125188548e-02,
            -3.8870634125188416e-02,
            -4.0032928708301152e-01,
            3.7738995266905739e00,
        ],
        [
            4.2351647362715017e-22,
            -5.3427584324835562e-07,
            1.8184245478981639e-04,
            1.8184245478981668e-04,
            2.0494571622815035e-04,
            4.7175299215212229e-03,
        ],
        [
            4.4598339301043052e-02,
            2.8856853343279535e-02,
            4.5293934057096763e-01,
            4.5293934057096763e-01,
            4.2951344311263497e00,
            6.0998553744300381e01,
        ],
        [
            -7.1148137410428485e-05,
            2.2653209597744274e-08,
            2.7900680967986886e-03,
            2.7900680967986920e-03,
            5.8858696744321093e-04,
            8.5674005639022610e-02,
        ],
        [
            7.1148137410428485e-05,
            1.8256141574911238e-05,
            -2.5205172650368105e-03,
            -2.5205172650368144e-03,
            -8.3127562420141909e-04,
            -8.6428933784300915e-02,
        ],
        [
            -6.6672444826954921e-24,
            1.5311948547352858e-10,
            -7.3563675489538430e-06,
            -7.3563675489538447e-06,
            -5.9657474700133831e-06,
            -9.4643267349888723e-05,
        ],
    ]
)


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

        z1 = sm.z(
            x=[0, 5000, 10000],
            C=C[:, [0]],
            length=10000,
            phi=10.0,
            has_foundation=True,
        )
        z2 = sm.z(
            x=[0, 2000, 4000],
            C=C[:, [1]],
            length=4000,
            phi=10.0,
            has_foundation=True,
        )

        zz = np.hstack([z1, z2])
        np.testing.assert_allclose(GT_skier_baseline, zz, rtol=1e-10, atol=1e-12)

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

        z1 = sm.z(
            x=[0, 2500, 5000],
            C=C[:, [0]],
            length=5000,
            phi=10.0,
            has_foundation=True,
        )
        z2 = sm.z(
            x=[0, 1000, 2000],
            C=C[:, [1]],
            length=2000,
            phi=10.0,
            has_foundation=True,
        )
        z3 = sm.z(
            x=[0, 2500, 5000],
            C=C[:, [2]],
            length=5000,
            phi=10.0,
            has_foundation=True,
        )

        zz = np.hstack([z1, z2, z3])
        np.testing.assert_allclose(GT_skiers_baseline, zz, rtol=1e-10, atol=1e-12)

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

        z1 = sm.z(
            x=[0, 5000, 10000],
            C=C[:, [0]],
            length=10000,
            phi=30.0,
            has_foundation=True,
        )
        z2 = sm.z(
            x=[0, 2000, 4000],
            C=C[:, [1]],
            length=4000,
            phi=30.0,
            has_foundation=False,
        )

        zz = np.hstack([z1, z2])
        np.testing.assert_allclose(GT_pst_without_touchdown, zz, rtol=1e-10, atol=1e-12)

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

        z1 = sm.z(
            x=[0, 5000, 10000],
            C=C[:, [0]],
            length=10000,
            phi=30.0,
            has_foundation=True,
        )
        z2 = sm.z(
            x=[0, 2000, 4000],
            C=C[:, [1]],
            length=4000,
            phi=30.0,
            has_foundation=False,
        )

        zz = np.hstack([z1, z2])
        np.testing.assert_allclose(GT_pst_with_touchdown, zz, rtol=1e-10, atol=1e-12)

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
