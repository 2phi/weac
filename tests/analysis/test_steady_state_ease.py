"""Slim unit tests for steady-state ease orientation selection."""

from __future__ import annotations

import math
import unittest
from typing import Any

from weac.analysis.steady_state import is_usable_orientation, select_ease_orientation


def _side(
    *,
    ease: float,
    err: float,
    converged: bool = True,
    never_cracked: bool = False,
    no_crack: bool = False,
    ease_key: str = "max_Sxx_norm",
) -> dict[str, Any]:
    return {
        "converged": converged,
        "never_cracked": never_cracked,
        "no_crack": no_crack,
        ease_key: ease,
        "energy_release_rate": err,
    }


class TestEaseSelection(unittest.TestCase):
    """Edge cases for select_ease_orientation."""

    def test_higher_is_easier_picks_larger_metric(self) -> None:
        result = select_ease_orientation(
            _side(ease=1.2, err=1.0),
            _side(ease=1.5, err=2.0),
            ease_key="max_Sxx_norm",
            higher_is_easier=True,
        )
        self.assertEqual(result.winner, "downslope")
        self.assertEqual(result.err_winner, "downslope")
        self.assertEqual(result.selection_rule, "ease:max_Sxx_norm")

    def test_lower_is_easier_picks_smaller_metric(self) -> None:
        result = select_ease_orientation(
            _side(ease=80.0, err=3.0, ease_key="critical_cut_length"),
            _side(ease=40.0, err=1.0, ease_key="critical_cut_length"),
            ease_key="critical_cut_length",
            higher_is_easier=False,
        )
        self.assertEqual(result.winner, "downslope")
        self.assertEqual(result.err_winner, "upslope")
        self.assertEqual(result.selection_rule, "ease:critical_cut_length")

    def test_one_unusable_side_selects_usable(self) -> None:
        result = select_ease_orientation(
            _side(
                ease=10.0, err=5.0, never_cracked=True, ease_key="critical_cut_length"
            ),
            _side(ease=50.0, err=1.0, ease_key="critical_cut_length"),
            ease_key="critical_cut_length",
            higher_is_easier=False,
        )
        self.assertEqual(result.winner, "downslope")
        self.assertEqual(result.err_winner, "upslope")

    def test_unusable_via_converged_false(self) -> None:
        self.assertFalse(
            is_usable_orientation({"converged": False, "critical_cut_length": 1.0})
        )
        result = select_ease_orientation(
            _side(ease=1.0, err=9.0, converged=False, ease_key="critical_cut_length"),
            _side(ease=100.0, err=1.0, ease_key="critical_cut_length"),
            ease_key="critical_cut_length",
            higher_is_easier=False,
        )
        self.assertEqual(result.winner, "downslope")

    def test_ease_tie_uses_err_tiebreak(self) -> None:
        result = select_ease_orientation(
            _side(ease=1.0, err=2.0),
            _side(ease=1.0, err=3.0),
            ease_key="max_Sxx_norm",
            higher_is_easier=True,
        )
        self.assertEqual(result.winner, "downslope")
        self.assertEqual(result.err_winner, "downslope")

    def test_both_equal_defaults_to_upslope(self) -> None:
        result = select_ease_orientation(
            _side(ease=1.0, err=2.0),
            _side(ease=1.0, err=2.0),
            ease_key="max_Sxx_norm",
            higher_is_easier=True,
        )
        self.assertEqual(result.winner, "upslope")
        self.assertEqual(result.err_winner, "upslope")

    def test_none_usable_falls_back_to_err_then_upslope(self) -> None:
        result = select_ease_orientation(
            _side(ease=1.0, err=1.0, never_cracked=True),
            _side(ease=2.0, err=1.0, never_cracked=True),
            ease_key="max_Sxx_norm",
            higher_is_easier=True,
        )
        self.assertEqual(result.winner, "upslope")
        self.assertEqual(result.err_winner, "upslope")

    def test_nan_ease_excluded_from_usable_set(self) -> None:
        # Finite upslope must win; pre-fix NaN compare always biased to downslope.
        result = select_ease_orientation(
            _side(ease=2.0, err=1.0),
            _side(ease=math.nan, err=9.0),
            ease_key="max_Sxx_norm",
            higher_is_easier=True,
        )
        self.assertEqual(result.winner, "upslope")
        self.assertEqual(result.err_winner, "downslope")

    def test_both_nan_ease_falls_back_to_err(self) -> None:
        result = select_ease_orientation(
            _side(ease=math.nan, err=1.0),
            _side(ease=math.nan, err=3.0),
            ease_key="max_Sxx_norm",
            higher_is_easier=True,
        )
        self.assertEqual(result.winner, "downslope")
        self.assertEqual(result.err_winner, "downslope")

    def test_already_cracked_is_usable(self) -> None:
        self.assertTrue(
            is_usable_orientation(
                {
                    "converged": True,
                    "already_cracked": True,
                    "never_cracked": False,
                    "critical_cut_length": 0.0,
                }
            )
        )

    def test_never_touches_custom_unusable_flag(self) -> None:
        result = select_ease_orientation(
            {
                "converged": True,
                "never_touches": True,
                "thickness_fraction_without_density_gate": 0.9,
                "energy_release_rate": 5.0,
            },
            {
                "converged": True,
                "never_touches": False,
                "thickness_fraction_without_density_gate": 0.2,
                "energy_release_rate": 1.0,
            },
            ease_key="thickness_fraction_without_density_gate",
            higher_is_easier=True,
            unusable_if_true=("never_touches",),
        )
        self.assertEqual(result.winner, "downslope")


if __name__ == "__main__":
    unittest.main()
