"""Unit tests for experimental util ease selection and UD compare helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from weac.analysis.experimental.util.compare import (
    ApproachCompareConfig,
    MetricSpec,
    RatioMetricSpec,
    build_comparison_ud_rows,
    metric_value,
    orientation_metric_value,
    write_approach_plots,
    write_comparison_csv,
    write_comparison_ud_md,
)
from weac.analysis.experimental.util.ease import (
    is_usable_orientation,
    select_ease_orientation,
)
from weac.analysis.experimental.util.plot import plot_ud_grouped_bars


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
            _side(ease=80.0, err=3.0, ease_key="critical_mass_kg"),
            _side(ease=40.0, err=1.0, ease_key="critical_mass_kg"),
            ease_key="critical_mass_kg",
            higher_is_easier=False,
        )
        self.assertEqual(result.winner, "downslope")
        self.assertEqual(result.err_winner, "upslope")
        self.assertEqual(result.selection_rule, "ease:critical_mass_kg")

    def test_one_unusable_side_selects_usable(self) -> None:
        result = select_ease_orientation(
            _side(ease=10.0, err=5.0, never_cracked=True, ease_key="critical_mass_kg"),
            _side(ease=50.0, err=1.0, ease_key="critical_mass_kg"),
            ease_key="critical_mass_kg",
            higher_is_easier=False,
        )
        self.assertEqual(result.winner, "downslope")
        self.assertEqual(result.err_winner, "upslope")

    def test_unusable_via_converged_false(self) -> None:
        self.assertFalse(
            is_usable_orientation({"converged": False, "critical_mass_kg": 1.0})
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

    def test_already_cracked_is_usable(self) -> None:
        self.assertTrue(
            is_usable_orientation(
                {
                    "converged": True,
                    "already_cracked": True,
                    "never_cracked": False,
                    "critical_mass_kg": 0.0,
                }
            )
        )


class TestOrientationMetricExtraction(unittest.TestCase):
    """Nested diagnostic metric reads for UD compare."""

    def test_orientation_metric_and_thickness_alias(self) -> None:
        payload = {
            "ok": True,
            "diagnostics": {
                "upslope": {
                    "energy_release_rate": 1.5,
                    "thickness_fraction_without_density_gate": 0.25,
                },
                "downslope": {
                    "energy_release_rate": 0.5,
                    "thickness_fraction_without_density_gate": 0.1,
                },
            },
        }
        self.assertEqual(
            orientation_metric_value(payload, "upslope", "energy_release_rate"),
            1.5,
        )
        self.assertEqual(
            orientation_metric_value(payload, "downslope", "thickness_fraction"),
            0.1,
        )
        self.assertEqual(
            metric_value(payload, "diagnostics.upslope.energy_release_rate"),
            1.5,
        )


class TestUdCompareArtifacts(unittest.TestCase):
    """UD plot/table writers on synthetic nested results."""

    def _synthetic_approach(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {
            "case_1": {
                "a": {
                    "ok": True,
                    "energy_release_rate": 2.0,
                    "max_Sxx_norm": 1.2,
                    "diagnostics": {
                        "winner": "downslope",
                        "err_winner": "upslope",
                        "selection_rule": "ease:max_Sxx_norm",
                        "upslope": {
                            "energy_release_rate": 2.0,
                            "max_Sxx_norm": 1.0,
                        },
                        "downslope": {
                            "energy_release_rate": 1.0,
                            "max_Sxx_norm": 1.5,
                        },
                    },
                },
                "b": {
                    "ok": True,
                    "energy_release_rate": 3.0,
                    "max_Sxx_norm": 0.8,
                    "diagnostics": {
                        "winner": "upslope",
                        "err_winner": "upslope",
                        "selection_rule": "ease:max_Sxx_norm",
                        "upslope": {
                            "energy_release_rate": 3.0,
                            "max_Sxx_norm": 0.9,
                        },
                        "downslope": {
                            "energy_release_rate": 2.5,
                            "max_Sxx_norm": 0.7,
                        },
                    },
                },
            }
        }

    def test_build_comparison_ud_rows_delta_sign(self) -> None:
        approach = self._synthetic_approach()
        metrics = (
            MetricSpec("energy_release_rate", "ERR"),
            MetricSpec("max_Sxx_norm", "max_Sxx_norm"),
        )
        fieldnames, rows = build_comparison_ud_rows(
            approach=approach, ud_metrics=metrics
        )
        self.assertIn("ease_winner", fieldnames)
        self.assertIn("ERR_Δ", fieldnames)
        row_a = next(r for r in rows if r["setup"] == "a")
        self.assertEqual(row_a["ease_winner"], "downslope")
        self.assertEqual(row_a["err_winner"], "upslope")
        # Δ = upslope − downslope = 2.0 − 1.0
        self.assertEqual(row_a["ERR_Δ"], "1")
        self.assertEqual(row_a["max_Sxx_norm_Δ"], "-0.5")

    def test_plot_and_table_writers_create_files(self) -> None:
        approach = self._synthetic_approach()
        baseline = {
            "case_1": {
                "a": {"ok": True, "energy_release_rate": 1.0, "max_Sxx_norm": 0.5},
                "b": {"ok": True, "energy_release_rate": 1.1, "max_Sxx_norm": 0.6},
            }
        }
        ud_metrics = (
            MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
            MetricSpec("max_Sxx_norm", "max_Sxx_norm", ylabel="max Sxx_norm [-]"),
        )
        config = ApproachCompareConfig(
            method_id="pst_fixed_cut",
            evaluate=MagicMock(),
            vs_baseline_metrics=(MetricSpec("energy_release_rate", "ERR"),),
            ab_metrics=(MetricSpec("max_Sxx_norm", "max_Sxx_norm"),),
            ud_metrics=ud_metrics,
            ratio_metrics=(
                RatioMetricSpec(
                    approach_key="max_Sxx_norm",
                    baseline_key="max_Sxx_norm",
                    label="Sxx_over_baseline",
                    ylabel="ratio [-]",
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plots_dir = out / "plots"
            written = write_approach_plots(
                config=config,
                baseline=baseline,
                approach=approach,
                plots_dir=plots_dir,
            )
            self.assertTrue((plots_dir / "vs_baseline_ERR.png").is_file())
            self.assertTrue((plots_dir / "ab_max_Sxx_norm.png").is_file())
            self.assertTrue((plots_dir / "ab_Sxx_over_baseline.png").is_file())
            self.assertTrue((plots_dir / "ud_ERR.png").is_file())
            self.assertTrue((plots_dir / "ud_max_Sxx_norm.png").is_file())
            self.assertTrue(all(p.is_file() for p in written))

            fieldnames, rows = build_comparison_ud_rows(
                approach=approach, ud_metrics=ud_metrics
            )
            ud_csv = out / "comparison_ud.csv"
            ud_md = out / "comparison_ud.md"
            write_comparison_csv(fieldnames, rows, ud_csv)
            write_comparison_ud_md(fieldnames, rows, ud_md, method_id="pst_fixed_cut")
            self.assertTrue(ud_csv.is_file())
            self.assertTrue(ud_md.is_file())
            self.assertIn("Δ = upslope − downslope", ud_md.read_text(encoding="utf-8"))

    def test_empty_ud_metrics_skips_ud_plots(self) -> None:
        approach = self._synthetic_approach()
        baseline = {
            "case_1": {
                "a": {"ok": True, "energy_release_rate": 1.0},
                "b": {"ok": True, "energy_release_rate": 1.1},
            }
        }
        config = ApproachCompareConfig(
            method_id="pst_fixed_cut",
            evaluate=MagicMock(),
            vs_baseline_metrics=(MetricSpec("energy_release_rate", "ERR"),),
            ab_metrics=(),
            ud_metrics=(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            plots_dir = Path(tmp) / "plots"
            written = write_approach_plots(
                config=config,
                baseline=baseline,
                approach=approach,
                plots_dir=plots_dir,
            )
            names = [p.name for p in written]
            self.assertTrue(any(n.startswith("vs_baseline_") for n in names))
            self.assertFalse(any(n.startswith("ud_") for n in names))

    def test_plot_ud_grouped_bars_direct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ud_ERR.png"
            out = plot_ud_grouped_bars(
                ["case_1/a", "case_1/b"],
                [2.0, 3.0],
                [1.0, 2.5],
                ease_winners=["downslope", "upslope"],
                ylabel="ERR",
                title="ud test",
                path=path,
            )
            self.assertEqual(out, path)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
