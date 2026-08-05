"""
Approach 4 runner — deformation-based first tip contact vs saved baseline_ss.

Searches the tilted PST cut where ``w_tip = crack_h`` (not flat ``l_AB``).

Run from the repository root:

    uv run python dev/alt_steady_state/pst_touchdown_cut/run.py

Hard-fails if ``dev/alt_steady_state/baseline_ss/results.json`` is missing.
Writes ``results.json``, ``comparison_ab.{csv,md}``, ``comparison_ud.{csv,md}``,
and plots under this directory.
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
    SetupDefinition,
    _build_layers,
)
from weac.analysis.experimental.pst_touchdown_cut import (  # noqa: E402
    evaluate_pst_touchdown_cut,
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
DEFAULT_PHI = 35.0

VS_BASELINE_METRICS = (
    MetricSpec(
        key="characteristic_length",
        label="characteristic_length",
        ylabel="characteristic_length [mm]",
    ),
    MetricSpec(
        key="energy_release_rate",
        label="ERR",
        ylabel="ERR [J/m²]",
    ),
    MetricSpec(
        key="thickness_fraction",
        label="thickness_fraction",
        ylabel="thickness fraction [-]",
    ),
)
AB_METRICS = (
    MetricSpec(
        key="energy_release_rate",
        label="ERR",
        ylabel="ERR [J/m²]",
    ),
    MetricSpec(
        key="thickness_fraction",
        label="thickness_fraction",
        ylabel="thickness fraction [-]",
    ),
    MetricSpec(
        key="characteristic_length",
        label="cut_length",
        ylabel="cut length [mm]",
    ),
)
UD_METRICS = (
    MetricSpec(
        key="energy_release_rate",
        label="ERR",
        ylabel="ERR [J/m²]",
    ),
    MetricSpec(
        key="thickness_fraction",
        label="thickness_fraction",
        ylabel="thickness fraction [-]",
    ),
    MetricSpec(
        key="cut_length",
        label="cut_length",
        ylabel="cut length [mm]",
    ),
)


def evaluate_setup(setup: SetupDefinition) -> ExperimentalSteadyStateResult:
    layers = _build_layers(setup.layers)
    weak_layer = WEAK_LAYER.model_copy(update=setup.weak_layer_kwargs)
    phi = float(setup.scenario_kwargs.get("phi", DEFAULT_PHI))
    return evaluate_pst_touchdown_cut(
        layers=layers,
        weak_layer=weak_layer,
        phi=phi,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pst_touchdown_cut over COMPARISON_CASES against the saved "
            "baseline_ss snapshot; write results, comparison tables, and plots."
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
        help="Skip writing vs-baseline / A/B / UD plots.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ApproachCompareConfig(
        method_id="pst_touchdown_cut",
        evaluate=evaluate_setup,
        vs_baseline_metrics=VS_BASELINE_METRICS,
        ab_metrics=AB_METRICS,
        ud_metrics=UD_METRICS,
        title=(
            "Paired A/B judgment — pst_touchdown_cut "
            "(ERR + thickness_fraction + cut_length)"
        ),
    )
    try:
        run_approach_comparison(
            config,
            cases=COMPARISON_CASES,
            out_dir=args.out,
            case_names=args.cases,
            skip_plots=args.skip_plots,
        )
    except MissingBaselineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
