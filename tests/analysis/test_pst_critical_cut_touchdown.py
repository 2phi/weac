"""Unit tests for tip-touch check at the tensile root (no second Brent)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from weac.analysis.steady_state import (
    TIP_CONTACT_EPS_MM,
    CutSearchResult,
    search_critical_cut_length,
)
from weac.components import Layer, WeakLayer


class TestTipTouchAtTensileRoot(unittest.TestCase):
    def setUp(self) -> None:
        self.layers = [Layer(rho=200.0, h=200.0)]
        self.weak_layer = WeakLayer(rho=100.0, h=20.0)

    def _stress_sample(self, max_sxx: float = 1.0):
        stress = MagicMock()
        stress.max_Sxx_norm = max_sxx
        stress.slab_tensile_criterion = 0.1
        return MagicMock(), stress, 1.0

    def test_touching_at_crack_reports_no_crack_cut_max(self) -> None:
        tensile = CutSearchResult(
            cut_length=400.0,
            already_at_min=False,
            never_reached=False,
            converged=True,
            sample=self._stress_sample(1.0),
        )
        cut_max_sample = self._stress_sample(0.6)

        with (
            patch(
                "weac.analysis.steady_state.search_cut_by_residual",
                return_value=tensile,
            ),
            patch(
                "weac.analysis.steady_state.free_tip_deflection",
                return_value=(12.0, 10.0),
            ),
            patch(
                "weac.analysis.steady_state.evaluate_pst_at_cut",
                return_value=cut_max_sample,
            ) as eval_at_cut,
        ):
            result = search_critical_cut_length(
                layers=self.layers,
                weak_layer=self.weak_layer,
                system_type="pst-",
                phi=30.0,
                cut_max=5000.0,
            )

        self.assertTrue(result.no_crack)
        self.assertFalse(result.converged)
        self.assertEqual(result.critical_cut_length, 5000.0)
        self.assertIn("tip already touching", result.message)
        self.assertIn("w_tip=12", result.message)
        eval_at_cut.assert_called_once()
        self.assertEqual(eval_at_cut.call_args.kwargs["cut_length"], 5000.0)

    def test_not_touching_keeps_critical_cut(self) -> None:
        tensile = CutSearchResult(
            cut_length=400.0,
            already_at_min=False,
            never_reached=False,
            converged=True,
            sample=self._stress_sample(1.0),
        )

        with (
            patch(
                "weac.analysis.steady_state.search_cut_by_residual",
                return_value=tensile,
            ),
            patch(
                "weac.analysis.steady_state.free_tip_deflection",
                return_value=(8.0, 10.0),
            ),
        ):
            result = search_critical_cut_length(
                layers=self.layers,
                weak_layer=self.weak_layer,
                system_type="pst-",
                phi=30.0,
            )

        self.assertFalse(result.no_crack)
        self.assertTrue(result.converged)
        self.assertEqual(result.critical_cut_length, 400.0)
        self.assertIn("critical cut=400", result.message)

    def test_touching_within_eps_reports_no_crack(self) -> None:
        """Near-equality (within TIP_CONTACT_EPS_MM) counts as tip contact."""
        crack_h = 10.0
        w_tip = crack_h - 0.5 * TIP_CONTACT_EPS_MM
        tensile = CutSearchResult(
            cut_length=400.0,
            already_at_min=False,
            never_reached=False,
            converged=True,
            sample=self._stress_sample(1.0),
        )
        cut_max_sample = self._stress_sample(0.6)

        with (
            patch(
                "weac.analysis.steady_state.search_cut_by_residual",
                return_value=tensile,
            ),
            patch(
                "weac.analysis.steady_state.free_tip_deflection",
                return_value=(w_tip, crack_h),
            ),
            patch(
                "weac.analysis.steady_state.evaluate_pst_at_cut",
                return_value=cut_max_sample,
            ),
        ):
            result = search_critical_cut_length(
                layers=self.layers,
                weak_layer=self.weak_layer,
                system_type="pst-",
                phi=30.0,
                cut_max=5000.0,
            )

        self.assertTrue(result.no_crack)
        self.assertFalse(result.converged)
        self.assertEqual(result.critical_cut_length, 5000.0)
        self.assertIn("tip already touching", result.message)

    def test_never_reached_skips_tip_check(self) -> None:
        tensile = CutSearchResult(
            cut_length=5000.0,
            already_at_min=False,
            never_reached=True,
            converged=False,
            sample=self._stress_sample(0.4),
        )

        with (
            patch(
                "weac.analysis.steady_state.search_cut_by_residual",
                return_value=tensile,
            ),
            patch(
                "weac.analysis.steady_state.free_tip_deflection",
            ) as tip,
        ):
            result = search_critical_cut_length(
                layers=self.layers,
                weak_layer=self.weak_layer,
                system_type="pst-",
                phi=30.0,
                cut_max=5000.0,
            )

        tip.assert_not_called()
        self.assertTrue(result.no_crack)
        self.assertEqual(result.critical_cut_length, 5000.0)
        self.assertIn("no_crack up to cut_max", result.message)


if __name__ == "__main__":
    unittest.main()
