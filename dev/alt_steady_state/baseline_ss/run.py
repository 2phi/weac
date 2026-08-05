"""
One-shot baseline steady-state snapshot over COMPARISON_CASES.

Run from the repository root:

    uv run python dev/alt_steady_state/baseline_ss/run.py

Writes ``dev/alt_steady_state/baseline_ss/results.json`` nested by
case → setup with core scalars. Approach runners must load this file and
must not recompute baseline (see ``util.compare.load_baseline_results``).
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
    _build_system,
)
from weac.analysis.experimental.util.baseline import (  # noqa: E402
    evaluate_baseline_steady_state,
)
from weac.analysis.experimental.util.compare import (  # noqa: E402
    iterate_comparison_cases,
    serialize_result,
    write_results_json,
)

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate production B_point_contact baseline over COMPARISON_CASES "
            "and write baseline_ss/results.json."
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
        default=RESULTS_PATH,
        help=f"Output JSON path (default: {RESULTS_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells = iterate_comparison_cases(COMPARISON_CASES, case_names=args.cases)
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
        except Exception as exc:  # noqa: BLE001 — record cell failure
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
    print(f"Wrote {out_path}", flush=True)
    print(f"Setups: {total} total, {n_ok} ok, {total - n_ok} errors", flush=True)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
