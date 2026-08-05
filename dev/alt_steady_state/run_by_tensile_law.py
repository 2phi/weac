"""
Re-run alt steady-state approaches under a chosen slab tensile-strength law.

Writes a self-contained tree (does not overwrite the default JJ artifacts)::

    dev/alt_steady_state/ts_<method>/
      baseline_ss/results.json
      pst_fixed_cut/
      pst_critical_mass/
      pst_critical_cut/
      pst_touchdown_cut/

Examples::

    uv run python dev/alt_steady_state/run_by_tensile_law.py --method hybrid
    uv run python dev/alt_steady_state/run_by_tensile_law.py --method adam --skip-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.analysis.test_slab_tensile_comparisons import (  # noqa: E402
    COMPARISON_CASES,
    DEFAULT_SCENARIO_KWARGS,
    LayerDefinition,
    SetupDefinition,
    _build_system,
)
from weac.analysis.experimental.pst_critical_cut import (  # noqa: E402
    evaluate_pst_critical_cut,
)
from weac.analysis.experimental.pst_critical_mass import (  # noqa: E402
    evaluate_pst_critical_mass,
)
from weac.analysis.experimental.pst_fixed_cut import (  # noqa: E402
    evaluate_pst_fixed_cut,
)
from weac.analysis.experimental.pst_touchdown_cut import (  # noqa: E402
    evaluate_pst_touchdown_cut,
)
from weac.analysis.experimental.util.baseline import (  # noqa: E402
    evaluate_baseline_steady_state,
)
from weac.analysis.experimental.util.compare import (  # noqa: E402
    ApproachCompareConfig,
    MetricSpec,
    RatioMetricSpec,
    iterate_comparison_cases,
    run_approach_comparison,
    serialize_result,
    write_results_json,
)
from weac.analysis.experimental.util.result import (  # noqa: E402
    ExperimentalSteadyStateResult,
)
from weac.components import Layer  # noqa: E402
from weac.components.presets import WEAK_LAYER  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_PHI = float(DEFAULT_SCENARIO_KWARGS["phi"])
TensileMethod = Literal["sigrist", "adam", "hybrid", "jamieson_johnson"]

APPROACHES = (
    "pst_fixed_cut",
    "pst_critical_mass",
    "pst_critical_cut",
    "pst_touchdown_cut",
)


def _build_layers(
    layer_defs: tuple[LayerDefinition, ...],
    *,
    method: TensileMethod,
) -> list[Layer]:
    if not 1 <= len(layer_defs) <= 2:
        raise ValueError("Each setup must define one or two slab layers.")
    return [
        Layer(rho=d.rho, h=d.h, tensile_strength_method=method) for d in layer_defs
    ]


def _weak_layer(setup: SetupDefinition):
    return WEAK_LAYER.model_copy(update=setup.weak_layer_kwargs)


def _phi(setup: SetupDefinition) -> float:
    return float(setup.scenario_kwargs.get("phi", DEFAULT_PHI))


def _make_evaluate(approach: str, method: TensileMethod):
    def evaluate(setup: SetupDefinition) -> ExperimentalSteadyStateResult:
        layers = _build_layers(setup.layers, method=method)
        weak_layer = _weak_layer(setup)
        phi = _phi(setup)
        if approach == "pst_fixed_cut":
            return evaluate_pst_fixed_cut(
                layers=layers, weak_layer=weak_layer, phi=phi
            )
        if approach == "pst_critical_mass":
            return evaluate_pst_critical_mass(
                layers=layers, weak_layer=weak_layer, phi=phi
            )
        if approach == "pst_critical_cut":
            return evaluate_pst_critical_cut(
                layers=layers, weak_layer=weak_layer, phi=phi
            )
        if approach == "pst_touchdown_cut":
            return evaluate_pst_touchdown_cut(
                layers=layers, weak_layer=weak_layer, phi=phi
            )
        raise ValueError(f"Unknown approach: {approach}")

    return evaluate


def _approach_config(approach: str, method: TensileMethod) -> ApproachCompareConfig:
    evaluate = _make_evaluate(approach, method)
    if approach == "pst_fixed_cut":
        return ApproachCompareConfig(
            method_id=approach,
            evaluate=evaluate,
            vs_baseline_metrics=(
                MetricSpec(
                    "characteristic_length", "characteristic_length", ylabel="L [mm]"
                ),
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
                MetricSpec("max_Sxx_norm", "max_Sxx_norm", ylabel="max Sxx_norm [-]"),
            ),
            ab_metrics=(
                MetricSpec("max_Sxx_norm", "max_Sxx_norm", ylabel="max Sxx_norm [-]"),
                MetricSpec(
                    "thickness_fraction",
                    "thickness_fraction",
                    ylabel="thickness fraction [-]",
                ),
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
            ),
            ud_metrics=(
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
                MetricSpec("max_Sxx_norm", "max_Sxx_norm", ylabel="max Sxx_norm [-]"),
                MetricSpec(
                    "thickness_fraction",
                    "thickness_fraction",
                    ylabel="thickness fraction [-]",
                ),
            ),
            title=f"Paired A/B — {approach} ({method})",
        )
    if approach == "pst_critical_mass":
        return ApproachCompareConfig(
            method_id=approach,
            evaluate=evaluate,
            vs_baseline_metrics=(
                MetricSpec(
                    "characteristic_length",
                    "characteristic_length",
                    ylabel="L [mm]",
                ),
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
            ),
            ab_metrics=(
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
                MetricSpec(
                    "critical_mass_kg",
                    "critical_mass",
                    ylabel="critical mass [kg]",
                ),
            ),
            ud_metrics=(
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
                MetricSpec(
                    "critical_mass_kg",
                    "critical_mass",
                    ylabel="critical mass [kg]",
                ),
            ),
            title=f"Paired A/B — {approach} ({method})",
        )
    if approach == "pst_critical_cut":
        return ApproachCompareConfig(
            method_id=approach,
            evaluate=evaluate,
            vs_baseline_metrics=(
                MetricSpec(
                    "characteristic_length",
                    "characteristic_length",
                    ylabel="characteristic_length [mm]",
                ),
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
            ),
            ab_metrics=(
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
                MetricSpec(
                    "characteristic_length",
                    "L_crit",
                    ylabel="L_crit [mm]",
                ),
            ),
            ud_metrics=(
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
                MetricSpec(
                    "critical_cut_length",
                    "L_crit",
                    ylabel="L_crit [mm]",
                ),
            ),
            ratio_metrics=(
                RatioMetricSpec(
                    approach_key="characteristic_length",
                    baseline_key="characteristic_length",
                    label="L_crit_over_L_B",
                    ylabel="L_crit / L_B [-]",
                ),
            ),
            title=f"Paired A/B — {approach} ({method})",
        )
    if approach == "pst_touchdown_cut":
        return ApproachCompareConfig(
            method_id=approach,
            evaluate=evaluate,
            vs_baseline_metrics=(
                MetricSpec(
                    "characteristic_length",
                    "characteristic_length",
                    ylabel="L [mm]",
                ),
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
                MetricSpec(
                    "thickness_fraction",
                    "thickness_fraction",
                    ylabel="thickness fraction [-]",
                ),
            ),
            ab_metrics=(
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
                MetricSpec(
                    "thickness_fraction",
                    "thickness_fraction",
                    ylabel="thickness fraction [-]",
                ),
                MetricSpec(
                    "characteristic_length",
                    "cut_length",
                    ylabel="cut length [mm]",
                ),
            ),
            ud_metrics=(
                MetricSpec("energy_release_rate", "ERR", ylabel="ERR [J/m²]"),
                MetricSpec(
                    "thickness_fraction",
                    "thickness_fraction",
                    ylabel="thickness fraction [-]",
                ),
                MetricSpec(
                    "characteristic_length",
                    "cut_length",
                    ylabel="cut length [mm]",
                ),
            ),
            title=f"Paired A/B — {approach} ({method})",
        )
    raise ValueError(f"Unknown approach: {approach}")


def run_baseline(*, method: TensileMethod, out_root: Path, case_names) -> Path:
    """Run baseline with patched layer tensile law; return results.json path."""
    import tests.analysis.test_slab_tensile_comparisons as tsc

    original = tsc._build_layers
    tsc._build_layers = lambda defs: _build_layers(defs, method=method)
    try:
        out_path = out_root / "baseline_ss" / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cells = iterate_comparison_cases(COMPARISON_CASES, case_names=case_names)
        nested: dict = {}
        total = len(cells)
        n_ok = 0
        for i, (case_name, setup_key, setup) in enumerate(cells, start=1):
            print(
                f"[{i}/{total}] {case_name}/{setup_key}/baseline_ss "
                f"[{method}] ...",
                flush=True,
            )
            try:
                result = evaluate_baseline_steady_state(_build_system(setup))
                payload = serialize_result(result)
                payload.update(
                    {
                        "case": case_name,
                        "setup": setup_key,
                        "method": "baseline_ss",
                        "tensile_strength_method": method,
                        "ok": True,
                        "error": None,
                    }
                )
                print("    -> ok", flush=True)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                print(f"    -> ERROR: {err}", flush=True)
                payload = {
                    "case": case_name,
                    "setup": setup_key,
                    "method": "baseline_ss",
                    "tensile_strength_method": method,
                    "ok": False,
                    "error": err,
                    "converged": False,
                    "message": err,
                    "characteristic_length": None,
                    "energy_release_rate": None,
                    "max_Sxx_norm": None,
                    "thickness_fraction_without_density_gate": None,
                    "diagnostics": {},
                }
            nested.setdefault(case_name, {})[setup_key] = payload
        write_results_json(nested, out_path)
        print(f"Wrote {out_path} ({n_ok}/{total} ok)", flush=True)
        return out_path
    finally:
        tsc._build_layers = original


def run_approaches(
    *,
    method: TensileMethod,
    out_root: Path,
    baseline_path: Path,
    case_names,
    skip_plots: bool,
    approaches: tuple[str, ...],
) -> None:
    for approach in approaches:
        out_dir = out_root / approach
        print(f"\n=== {approach} [{method}] → {out_dir} ===", flush=True)
        run_approach_comparison(
            _approach_config(approach, method),
            cases=COMPARISON_CASES,
            out_dir=out_dir,
            baseline_path=baseline_path,
            case_names=case_names,
            skip_plots=skip_plots,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline + PST approaches under a chosen tensile-strength law "
            "into ts_<method>/ (leaves default JJ tree untouched)."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["sigrist", "adam", "hybrid", "jamieson_johnson"],
        help="Layer.tensile_strength_method to use for all setups.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional subset of case names.",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=list(APPROACHES),
        choices=list(APPROACHES),
        help="Approaches to run (default: all four).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip grouped-bar plots.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Reuse existing ts_<method>/baseline_ss/results.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    method: TensileMethod = args.method
    out_root = ROOT / f"ts_{method}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}", flush=True)

    if args.skip_baseline:
        baseline_path = out_root / "baseline_ss" / "results.json"
        if not baseline_path.is_file():
            print(f"ERROR: missing baseline at {baseline_path}", file=sys.stderr)
            return 1
    else:
        baseline_path = run_baseline(
            method=method, out_root=out_root, case_names=args.cases
        )

    run_approaches(
        method=method,
        out_root=out_root,
        baseline_path=baseline_path,
        case_names=args.cases,
        skip_plots=args.skip_plots,
        approaches=tuple(args.approaches),
    )
    print(f"\nDone [{method}] → {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
