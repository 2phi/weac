"""
This module contains tests for the Analyzer class.
"""

# Standard library imports
import unittest

# Third party imports
import numpy as np

from weac.analysis.analyzer import Analyzer
from weac.components import (
    Config,
    Layer,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac.components.model_input import ModelInput
from weac.core.system_model import SystemModel


class TestAnalyzer(unittest.TestCase):
    """Test suite for the Analyzer."""

    def setUp(self):
        """Set up systems for tests: a generic skier system and a PST system."""
        # Basic "skier" system
        self.model_input_ski = ModelInput(
            scenario_config=ScenarioConfig(phi=15.0, system_type="skier"),
            layers=[Layer()],
            weak_layer=WeakLayer(),
            segments=[Segment(), Segment()],
        )
        self.sm_ski = SystemModel(model_input=self.model_input_ski, config=Config())
        self.an_ski = Analyzer(system_model=self.sm_ski, printing_enabled=False)

        # PST system for potential energy related methods
        self.model_input_pst = ModelInput(
            scenario_config=ScenarioConfig(phi=10.0, system_type="pst-"),
            layers=[Layer()],
            weak_layer=WeakLayer(),
            segments=[Segment(), Segment()],
        )
        self.sm_pst = SystemModel(model_input=self.model_input_pst, config=Config())
        self.an_pst = Analyzer(system_model=self.sm_pst, printing_enabled=False)

    def test_rasterize_solution_runs_and_shapes(self):
        for mode in ("cracked", "uncracked"):
            xs, Z, xs_supported = self.an_ski.rasterize_solution(mode=mode, num=200)
            self.assertEqual(Z.shape[0], 6)
            self.assertEqual(xs.shape[0], Z.shape[1])
            self.assertEqual(xs_supported.shape[0], xs.shape[0])
            self.assertTrue(np.all(np.diff(xs[~np.isnan(xs)]) >= 0))

    def test_get_zmesh_contains_expected_keys(self):
        zmesh = self.an_ski.get_zmesh(dz=5)
        for key in ("z", "E", "nu", "rho", "tensile_strength"):
            self.assertIn(key, zmesh)
        # Non-empty mesh
        self.assertGreater(len(zmesh["z"]), 1)

    def test_stress_fields_shapes_and_finite(self):
        _, Z, _ = self.an_ski.rasterize_solution(num=150)
        phi = self.sm_ski.scenario.phi
        Sxx = self.an_ski.Sxx(Z=Z, phi=phi, dz=5)
        Txz = self.an_ski.Txz(Z=Z, phi=phi, dz=5)
        Szz = self.an_ski.Szz(Z=Z, phi=phi, dz=5)
        # Consistent shapes
        self.assertEqual(Sxx.shape, Txz.shape)
        self.assertEqual(Sxx.shape, Szz.shape)
        # Finite values
        self.assertTrue(np.isfinite(Sxx).all())
        self.assertTrue(np.isfinite(Txz).all())
        self.assertTrue(np.isfinite(Szz).all())

    def test_principal_stress_slab_variants(self):
        _, Z, _ = self.an_ski.rasterize_solution(num=120)
        phi = self.sm_ski.scenario.phi
        for val in ("max", "min"):
            Ps = self.an_ski.principal_stress_slab(Z=Z, phi=phi, dz=5, val=val)
            self.assertTrue(np.isfinite(Ps).all())
        # Normalized tensile principal stress
        Ps_norm = self.an_ski.principal_stress_slab(
            Z=Z, phi=phi, dz=5, val="max", normalize=True
        )
        self.assertTrue(np.isfinite(Ps_norm).all())
        # Normalizing compressive should error
        with self.assertRaises(ValueError):
            _ = self.an_ski.principal_stress_slab(
                Z=Z, phi=phi, dz=5, val="min", normalize=True
            )

    def test_principal_stress_weaklayer_variants(self):
        _, Z, _ = self.an_ski.rasterize_solution(num=120)
        for val in ("max", "min"):
            ps = self.an_ski.principal_stress_weaklayer(Z=Z, val=val)
            self.assertTrue(np.isfinite(ps).all())
        # Normalized compressive principal stress in weak layer
        psn = self.an_ski.principal_stress_weaklayer(Z=Z, val="min", normalize=True)
        self.assertTrue(np.isfinite(psn).all())
        # Normalizing tensile should error
        with self.assertRaises(ValueError):
            _ = self.an_ski.principal_stress_weaklayer(Z=Z, val="max", normalize=True)

    def test_energy_release_rates_shapes(self):
        Ginc = self.an_ski.incremental_ERR()
        self.assertEqual(Ginc.shape, (3,))
        self.assertTrue(np.isfinite(Ginc).all())

        Gdif = self.an_ski.differential_ERR()
        self.assertEqual(Gdif.shape, (3,))
        self.assertTrue(np.isfinite(Gdif).all())

    def test_internal_and_external_potentials_pst(self):
        # Ensure PST-specific methods run
        Pi_total = self.an_pst.total_potential()
        self.assertTrue(np.isfinite(Pi_total))

        Pi_ext = self.an_pst._external_potential()
        self.assertTrue(np.isfinite(Pi_ext))

        Pi_int = self.an_pst._internal_potential()
        self.assertTrue(np.isfinite(Pi_int))
        # Consistency: total ≈ int + ext
        self.assertAlmostEqual(Pi_total, Pi_int + Pi_ext, places=6)
