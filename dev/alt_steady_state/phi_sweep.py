"""
Sweep inclination φ for experimental PST approaches.

Checks whether higher φ makes tensile failure easier and increases ERR.

Run from repo root:

    uv run python dev/alt_steady_state/phi_sweep.py
    uv run python dev/alt_steady_state/phi_sweep.py --cases case_1 case_5 --phis 0 15 25 35 45
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.analysis.test_slab_tensile_comparisons import (  # noqa: E402
    COMPARISON_CASES,
    SetupDefinition,
    _build_layers,
)
from weac.analysis.experimental import (  # noqa: E402
    evaluate_pst_critical_cut,
    evaluate_pst_critical_mass,
    evaluate_pst_fixed_cut,
    evaluate_pst_touchdown_cut,
)
from weac.analysis.experimental.util.result import (  # noqa: E402
    ExperimentalSteadyStateResult,
)
from weac.components.presets import WEAK_LAYER  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "phi_sweep"

# Ease: higher φ should make failure easier.
# ERR: higher φ should increase ERR.
EASE_EXPECTATIONS: dict[str, dict[str, str]] = {
    "pst_fixed_cut": {
        "max_Sxx_norm": "increase",
        "thickness_fraction_without_density_gate": "increase",
    },
    "pst_critical_cut": {
        "characteristic_length": "decrease",  # L_crit
    },
    "pst_critical_mass": {
        "critical_mass_kg": "decrease",
    },
    "pst_touchdown_cut": {
        "thickness_fraction_without_density_gate": "increase",
    },
}


def _evaluate(
    method: str,
    setup: SetupDefinition,
    phi: float,
) -> ExperimentalSteadyStateResult:
    layers = _build_layers(setup.layers)
    weak_layer = WEAK_LAYER.model_copy(update=setup.weak_layer_kwargs)
    kwargs: dict[str, Callable[..., ExperimentalSteadyStateResult]] = {
        "pst_fixed_cut": evaluate_pst_fixed_cut,
        "pst_critical_cut": evaluate_pst_critical_cut,
        "pst_critical_mass": evaluate_pst_critical_mass,
        "pst_touchdown_cut": evaluate_pst_touchdown_cut,
    }
    return kwargs[method](layers=layers, weak_layer=weak_layer, phi=phi)


def _winner_critical_mass(diag: dict[str, Any]) -> float | None:
    masses = diag.get("critical_mass_kg")
    if isinstance(masses, dict) and masses.get("winner") is not None:
        return float(masses["winner"])
    if isinstance(masses, (int, float)):
        return float(masses)
    return None


def _row_from_result(
    *,
    method: str,
    case: str,
    setup_key: str,
    phi: float,
    result: ExperimentalSteadyStateResult | None,
    error: str | None,
) -> dict[str, Any]:
    diag = (result.diagnostics if result else {}) or {}
    stress = result.maximal_stress_result if result else None
    return {
        "method": method,
        "case": case,
        "setup": setup_key,
        "phi": phi,
        "ok": error is None and bool(result and result.converged),
        "error": error,
        "winner": diag.get("winner"),
        "err_winner": diag.get("err_winner"),
        "ERR": None if result is None else result.energy_release_rate,
        "characteristic_length": (
            None if result is None else result.characteristic_length
        ),
        "max_Sxx_norm": None if stress is None else stress.max_Sxx_norm,
        "thickness_fraction_without_density_gate": (
            None if stress is None else stress.slab_tensile_criterion
        ),
        "critical_mass_kg": _winner_critical_mass(diag),
    }


def _trend(values: list[float | None], direction: str) -> str:
    """Return pass/fail/mixed/insufficient for monotonic expectation."""
    nums = [v for v in values if v is not None]
    if len(nums) < 2:
        return "insufficient"
    diffs = [b - a for a, b in zip(nums, nums[1:])]
    if direction == "increase":
        ups = sum(d > 1e-9 for d in diffs)
        downs = sum(d < -1e-9 for d in diffs)
        if downs == 0 and ups > 0:
            return "pass"
        if ups == 0 and downs > 0:
            return "fail"
        if ups == 0 and downs == 0:
            return "flat"
        return "mixed"
    # decrease
    downs = sum(d < -1e-9 for d in diffs)
    ups = sum(d > 1e-9 for d in diffs)
    if ups == 0 and downs > 0:
        return "pass"
    if downs == 0 and ups > 0:
        return "fail"
    if ups == 0 and downs == 0:
        return "flat"
    return "mixed"


def iterate_cells(
    cases: tuple[Any, ...],
    case_names: list[str] | None,
) -> list[tuple[str, str, SetupDefinition]]:
    selected = cases
    if case_names:
        wanted = set(case_names)
        selected = tuple(c for c in cases if c.name in wanted)
        missing = wanted - {c.name for c in selected}
        if missing:
            raise ValueError(f"Unknown case name(s): {sorted(missing)}")
    out: list[tuple[str, str, SetupDefinition]] = []
    for case in selected:
        out.append((case.name, "a", case.setup_a))
        out.append((case.name, "b", case.setup_b))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases",
        nargs="+",
        default=["case_1", "case_5", "case_12", "case_21"],
        help="Cases to sweep (default: case_1 case_5 case_12 case_21).",
    )
    p.add_argument(
        "--phis",
        nargs="+",
        type=float,
        default=[0.0, 15.0, 25.0, 35.0, 45.0],
        help="Inclination angles in degrees.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=[
            "pst_fixed_cut",
            "pst_critical_cut",
            "pst_critical_mass",
            "pst_touchdown_cut",
        ],
        choices=[
            "pst_fixed_cut",
            "pst_critical_cut",
            "pst_critical_mass",
            "pst_touchdown_cut",
        ],
    )
    p.add_argument("--out", type=Path, default=OUT_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = iterate_cells(COMPARISON_CASES, args.cases)
    rows: list[dict[str, Any]] = []
    total = len(args.methods) * len(cells) * len(args.phis)
    i = 0
    for method in args.methods:
        for case, setup_key, setup in cells:
            for phi in args.phis:
                i += 1
                print(
                    f"[{i}/{total}] {method} {case}/{setup_key} φ={phi} ...",
                    flush=True,
                )
                try:
                    result = _evaluate(method, setup, phi)
                    rows.append(
                        _row_from_result(
                            method=method,
                            case=case,
                            setup_key=setup_key,
                            phi=phi,
                            result=result,
                            error=None,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        _row_from_result(
                            method=method,
                            case=case,
                            setup_key=setup_key,
                            phi=phi,
                            result=None,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )

    fieldnames = list(rows[0].keys()) if rows else []
    csv_path = out_dir / "phi_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}", flush=True)

    # Summarize monotonic trends per method/case/setup
    summary_lines = [
        "# Inclination (φ) sweep — tensile ease & ERR",
        "",
        f"Phis: {args.phis}",
        f"Cases: {args.cases}",
        "",
        "Expectation: higher φ → easier tensile failure + higher ERR.",
        "",
    ]
    methods = args.methods
    for method in methods:
        summary_lines.append(f"## {method}")
        summary_lines.append("")
        ease_keys = EASE_EXPECTATIONS[method]
        header = (
            "| Case/setup | "
            + " | ".join(f"{k} ({d})" for k, d in ease_keys.items())
            + " | ERR (increase) |"
        )
        summary_lines.append(header)
        summary_lines.append(
            "|---|---" + "|---" * len(ease_keys) + "|"
        )
        for case, setup_key, _ in cells:
            series = [
                r
                for r in rows
                if r["method"] == method
                and r["case"] == case
                and r["setup"] == setup_key
            ]
            series = sorted(series, key=lambda r: r["phi"])
            cells_md: list[str] = []
            for key, direction in ease_keys.items():
                vals = [
                    float(v) if (v := r.get(key)) is not None else None for r in series
                ]
                cells_md.append(_trend(vals, direction))
            err_vals = [
                float(r["ERR"]) if r["ERR"] is not None else None for r in series
            ]
            cells_md.append(_trend(err_vals, "increase"))
            summary_lines.append(
                f"| {case}/{setup_key} | " + " | ".join(cells_md) + " |"
            )
        summary_lines.append("")

    # Tallies
    summary_lines.append("## Tallies")
    summary_lines.append("")
    summary_lines.append(
        "| Method | Ease metric | pass | fail | mixed | flat | insuff |"
    )
    summary_lines.append("|---|---|---|---|---|---|---|")
    for method in methods:
        ease_keys = EASE_EXPECTATIONS[method]
        for key, direction in list(ease_keys.items()) + [("ERR", "increase")]:
            counts = {"pass": 0, "fail": 0, "mixed": 0, "flat": 0, "insufficient": 0}
            for case, setup_key, _ in cells:
                series = sorted(
                    [
                        r
                        for r in rows
                        if r["method"] == method
                        and r["case"] == case
                        and r["setup"] == setup_key
                    ],
                    key=lambda r: r["phi"],
                )
                lookup = "ERR" if key == "ERR" else key
                vals = [
                    float(r[lookup]) if r.get(lookup) is not None else None
                    for r in series
                ]
                counts[_trend(vals, direction)] += 1
            summary_lines.append(
                f"| {method} | {key} ({direction}) | {counts['pass']} | "
                f"{counts['fail']} | {counts['mixed']} | {counts['flat']} | "
                f"{counts['insufficient']} |"
            )
    summary_lines.append("")

    # Detailed numeric table
    summary_lines.append("## Numeric series")
    summary_lines.append("")
    for method in methods:
        summary_lines.append(f"### {method}")
        summary_lines.append("")
        ease_keys = list(EASE_EXPECTATIONS[method].keys())
        cols = ["φ"] + ease_keys + ["ERR", "winner"]
        for case, setup_key, _ in cells:
            summary_lines.append(f"**{case}/{setup_key}**")
            summary_lines.append("")
            summary_lines.append("| " + " | ".join(cols) + " |")
            summary_lines.append("|" + "---|" * len(cols))
            series = sorted(
                [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["case"] == case
                    and r["setup"] == setup_key
                ],
                key=lambda r: r["phi"],
            )
            for r in series:
                if not r["ok"]:
                    summary_lines.append(
                        f"| {r['phi']:g} | error: {r['error']} |"
                    )
                    continue
                vals = [f"{r['phi']:g}"]
                for k in ease_keys:
                    vals.append(_fmt(r.get(k)))
                vals.append(_fmt(r.get("ERR")))
                vals.append(str(r.get("winner") or ""))
                summary_lines.append("| " + " | ".join(vals) + " |")
            summary_lines.append("")

    md_path = out_dir / "phi_sweep.md"
    md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}", flush=True)
    return 0


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
