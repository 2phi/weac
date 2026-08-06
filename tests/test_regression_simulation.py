"""
This module contains regression tests for the WEAC model.
"""

import numpy as np
import pytest

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
            -1.5915629333945330e-03,
            -1.5915629336245172e-03,
            -1.5881063073921575e-02,
            -1.5881063073921602e-02,
            -1.5926744799202215e-03,
            -1.5915629333945330e-03,
        ],
        [
            -4.9580332155970320e-26,
            -1.1282559163294969e-15,
            -8.8727607933936081e-05,
            1.7604328285420965e-05,
            5.4530238965214982e-09,
            5.9050496161149038e-13,
        ],
        [
            1.7566221061926016e-03,
            1.7566221062121309e-03,
            5.2305890327102263e-02,
            5.2305890327102263e-02,
            1.7565277179893842e-03,
            1.7566221061926016e-03,
        ],
        [
            1.6582910707734481e-26,
            9.5807099034229853e-17,
            8.9919447058825388e-04,
            -1.0305283349336320e-03,
            4.6304955252766267e-10,
            1.9750353870306439e-13,
        ],
        [
            1.0207865877058275e-05,
            1.0207865878602401e-05,
            2.1212597559624593e-04,
            2.1212597559624601e-04,
            1.0215328859423427e-05,
            1.0207865877058275e-05,
        ],
        [
            2.6286547599000873e-28,
            7.5751701007567834e-18,
            6.4394219643531282e-06,
            2.1861445155788474e-06,
            -3.6611891841525330e-11,
            -3.1307448147011621e-15,
        ],
    ]
)

GT_skiers_baseline = np.array(
    [
        [
            -2.4653852973084063e-03,
            -2.4655484885915658e-03,
            -8.6351058719333351e-03,
            -8.6351058719333403e-03,
            -2.7068559840275818e-03,
            -9.6646423216041888e-03,
            -9.6646423216041888e-03,
            -2.4658396871515922e-03,
            -2.4653852973084063e-03,
        ],
        [
            -5.5742993132157257e-14,
            -6.5624814220476531e-10,
            -3.5669712207617548e-05,
            4.2047638746413479e-06,
            3.3339662507053073e-07,
            -4.1596301998286905e-05,
            4.9239200976819496e-06,
            1.8272574652797735e-09,
            1.5521080181112037e-13,
        ],
        [
            3.5132442123852032e-03,
            3.5132596023747385e-03,
            3.1362176114134076e-02,
            3.1362176114134097e-02,
            3.5054833580758235e-03,
            3.6003234142644044e-02,
            3.6003234142644051e-02,
            3.5132013604950180e-03,
            3.5132442123852032e-03,
        ],
        [
            1.7212354353603082e-14,
            6.1888427880333399e-11,
            3.2999742490243448e-04,
            -3.9364862716827253e-04,
            9.1521517730917898e-08,
            3.8499746964858417e-04,
            -4.5925625776724132e-04,
            1.7232215222682087e-10,
            4.7926083085385896e-14,
        ],
        [
            1.0845857494374418e-05,
            1.0846583894195900e-05,
            8.9675432515149128e-05,
            8.9675432515149019e-05,
            1.1920821852005561e-05,
            1.0281925764731007e-04,
            1.0281925764731007e-04,
            1.0847880082191190e-05,
            1.0845857494374418e-05,
        ],
        [
            1.5154752883604485e-16,
            2.9211029133588500e-12,
            2.8976674951365774e-06,
            1.3026884518462219e-06,
            -1.4769670776969376e-09,
            3.3805303433233482e-06,
            1.5197214594845995e-06,
            -8.1335195861421427e-12,
            -4.2196897119194863e-16,
        ],
    ]
)

GT_pst_without_touchdown = np.array(
    [
        [
            -8.4871937355689708e-03,
            -7.1030228169780734e-03,
            2.1468411930446529e00,
            2.1468411930446507e00,
            1.2199839953114783e01,
            1.3555514675869393e01,
        ],
        [
            0.0000000000000000e00,
            1.9281153001886413e-09,
            8.6973240373155267e-03,
            8.6973240373155267e-03,
            2.1039215467303948e-03,
            1.7347234759768071e-18,
        ],
        [
            6.4429390610499186e-03,
            3.1432512066047066e-03,
            1.9796843832394049e00,
            1.9796843832394051e00,
            3.1544987306063848e02,
            8.3417059155662048e02,
        ],
        [
            -3.4064303328982494e-05,
            -1.4409831819984820e-10,
            3.1013009325309624e-02,
            3.1013009325309628e-02,
            2.3774677474281602e-01,
            2.6628625246349769e-01,
        ],
        [
            3.4064303328982487e-05,
            1.8112855093008230e-05,
            -2.9988530348269416e-02,
            -2.9988530348269416e-02,
            -2.3723453525429594e-01,
            -2.6628625246349769e-01,
        ],
        [
            -4.1030402786211213e-24,
            -3.7428407015253508e-12,
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


class TestRegressionSimulation:
    """Regression tests asserting stable outputs for key scenarios."""

    def test_skier_baseline(self):
        """Test the skier baseline."""
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
            theta=0.0,
            has_foundation=True,
        )
        z2 = sm.z(
            x=[0, 2000, 4000],
            C=C[:, [1]],
            length=4000,
            phi=10.0,
            theta=0.0,
            has_foundation=True,
        )

        zz = np.hstack([z1, z2])
        np.testing.assert_allclose(GT_skier_baseline, zz, rtol=1e-10, atol=1e-12)

    def test_skiers_baseline(self):
        """Test the skiers baseline."""
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
            theta=0.0,
            has_foundation=True,
        )
        z2 = sm.z(
            x=[0, 1000, 2000],
            C=C[:, [1]],
            length=2000,
            phi=10.0,
            theta=0.0,
            has_foundation=True,
        )
        z3 = sm.z(
            x=[0, 2500, 5000],
            C=C[:, [2]],
            length=5000,
            phi=10.0,
            theta=0.0,
            has_foundation=True,
        )

        zz = np.hstack([z1, z2, z3])
        np.testing.assert_allclose(GT_skiers_baseline, zz, rtol=1e-10, atol=1e-12)

    def test_pst_without_touchdown_baseline(self):
        """Test the pst without touchdown baseline."""
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
            theta=0.0,
            has_foundation=True,
        )
        z2 = sm.z(
            x=[0, 2000, 4000],
            C=C[:, [1]],
            length=4000,
            phi=30.0,
            theta=0.0,
            has_foundation=False,
        )

        zz = np.hstack([z1, z2])
        np.testing.assert_allclose(GT_pst_without_touchdown, zz, rtol=1e-10, atol=1e-12)

    def test_pst_with_touchdown_baseline(self):
        """Test the pst with touchdown baseline."""
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
        assert td.touchdown_mode == "C_in_contact"
        assert td.touchdown_distance == pytest.approx(
            1577.2698088929287, abs=0.5 * 10 ** (-6)
        )

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
            theta=0.0,
            has_foundation=True,
        )
        z2 = sm.z(
            x=[0, 2000, 4000],
            C=C[:, [1]],
            length=4000,
            phi=30.0,
            theta=0.0,
            has_foundation=False,
        )

        zz = np.hstack([z1, z2])
        np.testing.assert_allclose(GT_pst_with_touchdown, zz, rtol=1e-10, atol=1e-12)

    def test_criteria_evaluator_regressions(self):
        """Test the criteria evaluator regressions."""
        layers = [Layer(rho=170, h=100), Layer(rho=230, h=130)]
        wl = WeakLayer(rho=180, h=20)
        segs = [Segment(length=10000, has_foundation=True, m=0)]
        sc = ScenarioConfig(phi=30.0, system_type="skier", cut_length=0.0)
        mi = ModelInput(layers=layers, weak_layer=wl, segments=segs, scenario_config=sc)
        sm = SystemModel(model_input=mi, config=Config(touchdown=True))

        evaluator = CriteriaEvaluator(CriteriaConfig())

        # find_minimum_force baseline
        fm = evaluator.find_minimum_force(system=sm, tolerance_stress=0.005)
        assert fm.success
        assert fm.critical_skier_weight > 0
        # Baseline values recorded
        assert fm.critical_skier_weight == pytest.approx(
            75.17870187198098, abs=0.5 * 10 ** (-6)
        )
        assert fm.max_dist_stress == pytest.approx(
            1.0000048176337313, abs=0.5 * 10 ** (-6)
        )
        assert fm.min_dist_stress < 1.0

        # evaluate_SteadyState baseline (hybrid structured result)
        ss = evaluator.evaluate_SteadyState(system=sm)
        assert ss.converged
        assert ss.tensile.critical_cut_length > 0
        assert ss.err.energy_release_rate > 0
        assert ss.phi == 30.0
        # Baseline values recorded from a green hybrid SS run
        assert ss.tensile.critical_cut_length == pytest.approx(
            276.88720225, abs=0.5 * 10 ** (-6)
        )
        assert ss.err.energy_release_rate == pytest.approx(
            1.8587470190926725, abs=0.5 * 10 ** (-6)
        )
        assert ss.tensile.cut_direction_winner == "upslope"
        assert ss.err.cut_direction_winner == "upslope"

        # evaluate_coupled_criterion baseline
        cc = evaluator.evaluate_coupled_criterion(system=sm, max_iterations=10)
        assert cc is not None
        assert isinstance(cc.critical_skier_weight, float)
        assert isinstance(cc.crack_length, float)
        # Baseline values recorded
        assert cc.converged
        np.testing.assert_allclose(
            cc.critical_skier_weight, 180.87597195071328, rtol=1e-2
        )
        np.testing.assert_allclose(cc.crack_length, 118.82324435526425, rtol=1e-2)
        np.testing.assert_allclose(cc.g_delta, 1.0, rtol=1e-2)
        np.testing.assert_allclose(cc.dist_ERR_envelope, 0.0, atol=1e-2)

        # find_minimum_crack_length baseline (returns crack length > 0)
        crack_len, new_segments = evaluator.find_minimum_crack_length(system=sm)
        assert crack_len > 0
        assert all(isinstance(s, Segment) for s in new_segments)
        # Baseline value recorded
        np.testing.assert_allclose(crack_len, 1564.671141349807, rtol=1e-2)
