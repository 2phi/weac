# tests/test_system_model.py
import unittest
import numpy as np
from functools import cached_property

from weac_2.components import (
    ModelInput, Layer, Segment, CriteriaConfig,
    WeakLayer, ScenarioConfig
)
from weac_2.components.config import Config
from weac_2.core.system_model import SystemModel

class DummyEigensystem:
    calls = 0

    def __init__(self, weak_layer, slab):
        DummyEigensystem.calls += 1
        self.tag = f"EIG#{DummyEigensystem.calls}"


class DummySystemModel(SystemModel):
    """SystemModel that swaps in DummyEigensystem and
       a trivial _solve_for_unknown_constants()."""
    _const_calls = 0

    @cached_property
    def eigensystem(self):              # replaces the heavy one
        return DummyEigensystem(self.weak_layer, self.slab)
    
    def _solve_for_unknown_constants(self):
        DummySystemModel._const_calls += 1   # <-- NEW
        return np.array([DummySystemModel._const_calls])
# ----------------------------------------------------------------------
# 2.  The actual tests
# ----------------------------------------------------------------------
class TestSystemModelCaching(unittest.TestCase):

    def setUp(self):
        # reset static counter between test methods
        DummyEigensystem.calls = 0

        model_input = ModelInput(
            scenario_config=ScenarioConfig(phi=5, touchdown=True, system='skier'),
            weak_layer=WeakLayer(rho=10, h=30, E=0.25, G_Ic=1),
            layers=[Layer(rho=170, h=100), Layer(rho=280, h=100)],
            segments=[Segment(l=3000, k=True, m=70), Segment(l=4000, k=True, m=0)],
            criteria_config=CriteriaConfig(fn=1, fm=1, gn=1, gm=1),
        )
        cfg = Config(youngs_modulus_method='bergfeld',
                     stress_failure_envelope_method='adam_unpublished')

        self.system = DummySystemModel(model_input, cfg)

    # ------------------------------------------------------------------
    def test_caching(self):
        # first access builds both heavy objects
        eig1 = self.system.eigensystem
        C1   = self.system.unknown_constants
        self.assertEqual(DummyEigensystem.calls, 1)

        # second access without changes must reuse the cache
        eig1_again = self.system.eigensystem
        C1_again   = self.system.unknown_constants
        self.assertIs(eig1_again, eig1)
        self.assertIs(C1_again, C1)
        self.assertEqual(DummyEigensystem.calls, 1)

    # ----------------------------------------------------------------
    def test_scenario_update_only_rebuilds_constants(self):
        _ = self.system.eigensystem      # build once
        C_before = self.system.unknown_constants.copy()
        print(C_before)

        # Change a value that the solver actually uses (phi in degrees)
        self.system.update_scenario(phi=15)
        C_after = self.system.unknown_constants
        print(C_after)
        # eigensystem must still be cached
        self.assertEqual(DummyEigensystem.calls, 1)
        # constants must have changed
        self.assertFalse(np.array_equal(C_after, C_before))
    # ------------------------------------------------------------------
    def test_slab_update_rebuilds_both(self):
        eig_before = self.system.eigensystem
        C_before   = self.system.unknown_constants.copy()

        self.system.update_slab_layers([
            Layer(rho=200, h=50),
            Layer(rho=280, h=150)
        ])

        eig_after = self.system.eigensystem
        C_after   = self.system.unknown_constants

        self.assertEqual(DummyEigensystem.calls, 2)
        self.assertIsNot(eig_after, eig_before)
        self.assertFalse(np.array_equal(C_after, C_before))


# Run the tests when the file is executed directly
if __name__ == "__main__":
    unittest.main(verbosity=2)
