"""
Re-run baseline + PST approaches at a chosen inclination φ.

Writes a self-contained tree (does not overwrite the default φ=35° artifacts)::

    dev/alt_steady_state/phi_<deg>/
      baseline_ss/results.json
      pst_fixed_cut/
      pst_critical_mass/
      pst_critical_cut/
      pst_touchdown_cut/

Examples::

    uv run python dev/alt_steady_state/run_at_phi.py --phi 15
    uv run python dev/alt_steady_state/run_at_phi.py --phi 15 --skip-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.analysis.test_slab_tensile_comparisons import (  # noqa: E402
    COMPARISON_CASES,
    ComparisonCase,
    SetupDefinition,
    _build_system,
)
from weac.analysis.experimental.util.baseline import (  # noqa: E402
    evaluate_baseline_steady_state,
)
from weac.analysis.experimental.util.compare import (  # noqa: E402
    iterate_comparison_cases,
    run_approach_comparison,
    serialize_result,
    write_results_json,
)

# Reuse approach configs from the tensile-law runner.
from run_by_tensile_law import APPROACHES, _approach_config  # noqa: E402


def _with_phi(setup: SetupDefinition, phi: float) -> SetupDefinition:
    scenario = {**setup.scenario_kwargs, "phi": float(phi)}
    return SetupDefinition(
        layers=setup.layers,
        weak_layer_kwargs=dict(setup.weak_layer_kwargs),
        scenario_kwargs=scenario,
        config_kwargs=dict(setup.config_kwargs),
    )


def cases_at_phi(phi: float) -> tuple[ComparisonCase, ...]:
    return tuple(
        ComparisonCase(
            name=case.name,
            setup_a=_with_phi(case.setup_a, phi),
            setup_b=_with_phi(case.setup_b, phi),
        )
        for case in COMPARISON_CASES
    )


def run_baseline(*, cases: tuple[ComparisonCase, ...], out_root: Path, case_names) -> Path:
    out_path = out_root / "baseline_ss" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cells = iterate_comparison_cases(cases, case_names=case_names)
    nested: dict = {}
    total = len(cells)
    n_ok = 0
    for i, (case_name, setup_key, setup) in enumerate(cells, start=1):
        print(f"[{i}/{total}] {case_name}/{setup_key}/baseline_ss ...", flush=True)
        try:
            result = evaluate_baseline_steady_state(_build_system(setup))
            payload = serialize_result(result)
            payload.update(
                {
                    "case": case_name,
                    "setup": setup_key,
                    "method": "baseline_ss",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phi", type=float, required=True, help="Inclination [deg].")
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument(
        "--approaches",
        nargs="+",
        default=list(APPROACHES),
        choices=list(APPROACHES),
    )
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Reuse existing phi_<deg>/baseline_ss/results.json.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    phi = float(args.phi)
    # Folder tag: 15 -> phi_15, 15.5 -> phi_15p5
    tag = f"{phi:g}".replace(".", "p")
    out_root = ROOT / f"phi_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    cases = cases_at_phi(phi)
    print(f"Output root: {out_root} (φ={phi}°)", flush=True)

    if args.skip_baseline:
        baseline_path = out_root / "baseline_ss" / "results.json"
        if not baseline_path.is_file():
            print(f"ERROR: missing baseline at {baseline_path}", file=sys.stderr)
            return 1
    else:
        baseline_path = run_baseline(
            cases=cases, out_root=out_root, case_names=args.cases
        )

    # Default JJ tensile law — same as root tree.
    method = "jamieson_johnson"
    for approach in args.approaches:
        out_dir = out_root / approach
        print(f"\n=== {approach} [φ={phi}] → {out_dir} ===", flush=True)
        # Patch evaluate to use cases' scenario phi (already injected).
        config = _approach_config(approach, method)
        run_approach_comparison(
            config,
            cases=cases,
            out_dir=out_dir,
            baseline_path=baseline_path,
            case_names=args.cases,
            skip_plots=args.skip_plots,
        )

    print(f"\nDone [φ={phi}] → {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
