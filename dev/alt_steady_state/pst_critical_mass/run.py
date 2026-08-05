"""
Approach 2 runner — PST fixed cut with critical right-edge end mass.

Run from the repository root:

    uv run python dev/alt_steady_state/pst_critical_mass/run.py

Loads the saved ``baseline_ss/results.json`` snapshot (hard-fail if missing),
evaluates ``pst_critical_mass`` over COMPARISON_CASES, and writes:

- ``results.json``
- ``comparison_ab.{csv,md}``
- ``comparison_ud.{csv,md}``
- ``plots/`` — vs-baseline, A/B, and UD (ERR, critical_mass)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Script lives under gitignored ``dev/``; ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.analysis.test_slab_tensile_comparisons import (  # noqa: E402
    COMPARISON_CASES,
    DEFAULT_SCENARIO_KWARGS,
    SetupDefinition,
    _build_layers,
)
from weac.analysis.experimental.pst_critical_mass import (  # noqa: E402
    evaluate_pst_critical_mass,
)
from weac.analysis.experimental.util.compare import (  # noqa: E402
    ApproachCompareConfig,
    MetricSpec,
    run_approach_comparison,
)
from weac.analysis.experimental.util.result import (  # noqa: E402
    ExperimentalSteadyStateResult,
)
from weac.components.presets import WEAK_LAYER  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_PHI = float(DEFAULT_SCENARIO_KWARGS["phi"])

VS_BASELINE_METRICS = (
    MetricSpec(
        key="characteristic_length",
        label="characteristic_length",
        ylabel="characteristic_length [mm]",
    ),
    MetricSpec(
        key="energy_release_rate",
        label="ERR",
        ylabel="Energy release rate [J/m²]",
    ),
)
AB_METRICS = (
    MetricSpec(
        key="energy_release_rate",
        label="ERR",
        ylabel="Energy release rate [J/m²]",
    ),
    MetricSpec(
        key="critical_mass_kg",
        label="critical_mass",
        ylabel="critical mass [kg]",
    ),
)
UD_METRICS = (
    MetricSpec(
        key="energy_release_rate",
        label="ERR",
        ylabel="Energy release rate [J/m²]",
    ),
    MetricSpec(
        key="critical_mass_kg",
        label="critical_mass",
        ylabel="critical mass [kg]",
    ),
)


def _phi(setup: SetupDefinition) -> float:
    return float(setup.scenario_kwargs.get("phi", DEFAULT_PHI))


def evaluate_setup(setup: SetupDefinition) -> ExperimentalSteadyStateResult:
    """Evaluate Approach 2 for one COMPARISON_CASES setup."""
    layers = _build_layers(setup.layers)
    weak_layer = WEAK_LAYER.model_copy(update=setup.weak_layer_kwargs)
    return evaluate_pst_critical_mass(
        layers=layers,
        weak_layer=weak_layer,
        phi=_phi(setup),
    )


COMPARE_CONFIG = ApproachCompareConfig(
    method_id="pst_critical_mass",
    evaluate=evaluate_setup,
    vs_baseline_metrics=VS_BASELINE_METRICS,
    ab_metrics=AB_METRICS,
    ud_metrics=UD_METRICS,
    title="Paired A/B judgment — pst_critical_mass (Approach 2)",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pst_critical_mass over COMPARISON_CASES against the "
            "saved baseline_ss snapshot."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional subset of case names (default: all COMPARISON_CASES).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR}).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_approach_comparison(
        COMPARE_CONFIG,
        cases=COMPARISON_CASES,
        out_dir=args.out_dir,
        case_names=args.cases,
        skip_plots=args.skip_plots,
    )
    results = summary["results"]
    n_ok = sum(
        1
        for setups in results.values()
        for payload in setups.values()
        if payload.get("ok")
    )
    n_total = sum(len(setups) for setups in results.values())
    print(f"Setups: {n_total} total, {n_ok} ok, {n_total - n_ok} errors", flush=True)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
