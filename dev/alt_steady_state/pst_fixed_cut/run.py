"""
Approach 1 comparison runner: fixed-cut PST vs saved ``baseline_ss``.

Run from the repository root:

    uv run python dev/alt_steady_state/pst_fixed_cut/run.py

Loads ``baseline_ss/results.json`` (hard-fail if missing). Writes
``results.json``, ``comparison_ab.{csv,md}``, ``comparison_ud.{csv,md}``,
and plots (including ``ud_*.png``) under this folder.
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
from weac.analysis.experimental.pst_fixed_cut import (  # noqa: E402
    evaluate_pst_fixed_cut,
)
from weac.analysis.experimental.util.compare import (  # noqa: E402
    ApproachCompareConfig,
    MetricSpec,
    MissingBaselineError,
    run_approach_comparison,
)
from weac.analysis.experimental.util.result import (  # noqa: E402
    ExperimentalSteadyStateResult,
)
from weac.components.presets import WEAK_LAYER  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_PHI = float(DEFAULT_SCENARIO_KWARGS["phi"])

VS_BASELINE_METRICS = (
    MetricSpec("characteristic_length", "characteristic_length", ylabel="L [mm]"),
    MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
    MetricSpec("max_Sxx_norm", "max_Sxx_norm", ylabel="max Sxx_norm [-]"),
)
AB_METRICS = (
    MetricSpec("max_Sxx_norm", "max_Sxx_norm", ylabel="max Sxx_norm [-]"),
    MetricSpec(
        "thickness_fraction",
        "thickness_fraction",
        ylabel="thickness fraction [-]",
    ),
    MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
)
UD_METRICS = (
    MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
    MetricSpec("max_Sxx_norm", "max_Sxx_norm", ylabel="max Sxx_norm [-]"),
    MetricSpec(
        "thickness_fraction",
        "thickness_fraction",
        ylabel="thickness fraction [-]",
    ),
)


def evaluate(setup: SetupDefinition) -> ExperimentalSteadyStateResult:
    """Evaluate fixed-cut PST for one comparison setup."""
    layers = _build_layers(setup.layers)
    weak_layer = WEAK_LAYER.model_copy(update=setup.weak_layer_kwargs)
    phi = float(setup.scenario_kwargs.get("phi", DEFAULT_PHI))
    return evaluate_pst_fixed_cut(layers=layers, weak_layer=weak_layer, phi=phi)


CONFIG = ApproachCompareConfig(
    method_id="pst_fixed_cut",
    evaluate=evaluate,
    vs_baseline_metrics=VS_BASELINE_METRICS,
    ab_metrics=AB_METRICS,
    ud_metrics=UD_METRICS,
    title="Paired A/B judgment — pst_fixed_cut",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Approach 1 (pst_fixed_cut) over COMPARISON_CASES, "
            "compare to saved baseline_ss, and write tables + plots."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional subset of case names (default: all COMPARISON_CASES).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR}).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip writing grouped-bar plots.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_approach_comparison(
            CONFIG,
            cases=COMPARISON_CASES,
            out_dir=args.out,
            case_names=args.cases,
            skip_plots=args.skip_plots,
        )
    except MissingBaselineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

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
