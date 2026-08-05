"""
Comparison orchestration for approach runners.

Approach runners must **load** the saved ``baseline_ss`` snapshot and must not
recompute baseline. Missing baseline is a hard error.
"""

from __future__ import annotations

import csv
import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from weac.analysis.experimental.util.plot import (
    plot_ab_grouped_bars,
    plot_ud_grouped_bars,
    plot_vs_baseline_grouped_bars,
)
from weac.analysis.experimental.util.result import ExperimentalSteadyStateResult

# Default location relative to the repository root.
DEFAULT_BASELINE_RESULTS_PATH = Path("dev/alt_steady_state/baseline_ss/results.json")

_CASE_NUM_RE = re.compile(r"(\d+)")


class MissingBaselineError(FileNotFoundError):
    """Raised when ``baseline_ss/results.json`` is absent (no silent recompute)."""


@dataclass(frozen=True)
class MetricSpec:
    """
    One metric extracted from a serialized result payload.

    ``key`` is a top-level payload field, a dotted nested path (e.g.
    ``diagnostics.upslope.energy_release_rate``), or ``critical_mass_kg``
    (resolved from ``diagnostics.critical_mass_kg.winner``).
    ``thickness_fraction`` is accepted as an alias for
    ``thickness_fraction_without_density_gate`` (top-level or within an
    orientation block for UD metrics).
    """

    key: str
    label: str
    ylabel: str | None = None


@dataclass(frozen=True)
class RatioMetricSpec:
    """
    Approach metric divided by a baseline metric at the same case/setup.

    Used for dimensionless comparisons such as
    ``L_crit / L_B`` (critical cut over baseline mode-B touchdown length).
    """

    approach_key: str
    baseline_key: str
    label: str
    ylabel: str | None = None


@dataclass(frozen=True)
class ApproachCompareConfig:
    """Method-specific wiring for a thin ``dev/alt_steady_state/<method>/run.py``."""

    method_id: str
    evaluate: Callable[[Any], ExperimentalSteadyStateResult]
    vs_baseline_metrics: tuple[MetricSpec, ...]
    ab_metrics: tuple[MetricSpec, ...]
    title: str | None = None
    ud_metrics: tuple[MetricSpec, ...] = ()
    ratio_metrics: tuple[RatioMetricSpec, ...] = ()


def natural_case_key(name: str) -> tuple[Any, ...]:
    """Sort ``case_2`` before ``case_10``."""
    parts = _CASE_NUM_RE.split(name)
    key: list[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def json_safe(obj: Any) -> Any:
    """Recursively convert values to JSON-serializable forms."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def serialize_result(result: ExperimentalSteadyStateResult) -> dict[str, Any]:
    """JSON-friendly experimental payload (scalars + diagnostics; no system)."""
    payload = {
        **result.core_scalars(),
        "diagnostics": json_safe(result.diagnostics),
    }
    prod = result.diagnostics.get("production_slab_tensile_criterion")
    if prod is not None:
        payload["production_slab_tensile_criterion"] = float(prod)
    return payload


def load_baseline_results(
    path: Path | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Load the canonical ``baseline_ss/results.json`` snapshot.

    Raises ``MissingBaselineError`` if the file is missing. Callers must not
    fall back to recomputing baseline.
    """
    results_path = Path(path) if path is not None else DEFAULT_BASELINE_RESULTS_PATH
    if not results_path.is_file():
        raise MissingBaselineError(
            f"Baseline snapshot not found at {results_path}. "
            "Run `uv run python dev/alt_steady_state/baseline_ss/run.py` first; "
            "approach runners must not recompute baseline."
        )
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Baseline results must be a JSON object: {results_path}")
    return data


def _resolve_metric_key(key: str) -> str:
    if key == "thickness_fraction":
        return "thickness_fraction_without_density_gate"
    if key.endswith(".thickness_fraction"):
        return key[: -len("thickness_fraction")] + (
            "thickness_fraction_without_density_gate"
        )
    return key


def _nested_get(payload: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_value(payload: dict[str, Any] | None, key: str) -> float | None:
    """Extract a numeric metric from a serialized payload (or None)."""
    if payload is None:
        return None
    # Missing ``ok`` (baseline snapshots) is treated as success.
    if payload.get("ok") is False:
        return None

    resolved = _resolve_metric_key(key)
    if resolved == "critical_mass_kg":
        diagnostics = payload.get("diagnostics") or {}
        masses = diagnostics.get("critical_mass_kg")
        if isinstance(masses, dict):
            value = masses.get("winner")
        else:
            value = masses
    elif "." in resolved:
        value = _nested_get(payload, resolved)
    else:
        value = payload.get(resolved)

    return _as_optional_float(value)


def ratio_metric_value(
    approach_payload: dict[str, Any] | None,
    baseline_payload: dict[str, Any] | None,
    metric: RatioMetricSpec,
) -> float | None:
    """Return ``approach[key] / baseline[key]``, or None if either side is unusable."""
    numerator = metric_value(approach_payload, metric.approach_key)
    denominator = metric_value(baseline_payload, metric.baseline_key)
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def orientation_metric_value(
    payload: dict[str, Any] | None,
    orientation: str,
    key: str,
) -> float | None:
    """
    Extract a numeric metric from ``diagnostics.<orientation>``.

    ``key`` is relative to the orientation block (supports dotted paths and
    the ``thickness_fraction`` alias).
    """
    if payload is None:
        return None
    if payload.get("ok") is False:
        return None
    diagnostics = payload.get("diagnostics") or {}
    if not isinstance(diagnostics, dict):
        return None
    block = diagnostics.get(orientation)
    if not isinstance(block, dict):
        return None
    resolved = _resolve_metric_key(key)
    if "." in resolved:
        value = _nested_get(block, resolved)
    else:
        value = block.get(resolved)
    return _as_optional_float(value)


def write_results_json(nested: dict[str, Any], path: Path) -> None:
    """Write nested results.json (case → setup → …)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(nested), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scalar_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _delta_str(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return ""
    return _scalar_or_empty(b - a)


def _ud_delta_str(upslope: float | None, downslope: float | None) -> str:
    """Δ = upslope − downslope."""
    if upslope is None or downslope is None:
        return ""
    return _scalar_or_empty(upslope - downslope)


def build_comparison_ab_rows(
    *,
    cases: Sequence[str],
    baseline: dict[str, dict[str, dict[str, Any]]],
    approach: dict[str, dict[str, dict[str, Any]]],
    method_id: str,
    baseline_metrics: Sequence[MetricSpec],
    approach_metrics: Sequence[MetricSpec],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Paired A/B table: baseline mirror columns + approach A/B metrics.

    Columns are ``baseline_ss.{label}_{a|b|Δ}`` and ``{method}.{label}_{a|b|Δ}``.
    """
    fieldnames = ["case"]
    for metric in baseline_metrics:
        for suffix in ("a", "b", "Δ"):
            fieldnames.append(f"baseline_ss.{metric.label}_{suffix}")
    for metric in approach_metrics:
        for suffix in ("a", "b", "Δ"):
            fieldnames.append(f"{method_id}.{metric.label}_{suffix}")

    rows: list[dict[str, str]] = []
    for case in cases:
        row: dict[str, str] = {"case": case}
        for metric in baseline_metrics:
            val_a = metric_value((baseline.get(case) or {}).get("a"), metric.key)
            val_b = metric_value((baseline.get(case) or {}).get("b"), metric.key)
            row[f"baseline_ss.{metric.label}_a"] = _scalar_or_empty(val_a)
            row[f"baseline_ss.{metric.label}_b"] = _scalar_or_empty(val_b)
            row[f"baseline_ss.{metric.label}_Δ"] = _delta_str(val_a, val_b)
        for metric in approach_metrics:
            val_a = metric_value((approach.get(case) or {}).get("a"), metric.key)
            val_b = metric_value((approach.get(case) or {}).get("b"), metric.key)
            row[f"{method_id}.{metric.label}_a"] = _scalar_or_empty(val_a)
            row[f"{method_id}.{metric.label}_b"] = _scalar_or_empty(val_b)
            row[f"{method_id}.{metric.label}_Δ"] = _delta_str(val_a, val_b)
        rows.append(row)
    return fieldnames, rows


def write_comparison_csv(
    fieldnames: list[str], rows: list[dict[str, str]], path: Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_ab_md(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    path: Path,
    *,
    method_id: str,
    title: str | None = None,
) -> None:
    """Markdown paired A/B judgment table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    heading = title or f"Paired A/B judgment — {method_id}"
    lines = [
        f"# {heading}",
        "",
        "Per-case A vs B (and Δ = b−a). Baseline columns mirror vs-baseline "
        "plots; approach columns match the A/B plot metric set.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(c, "") for c in fieldnames) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_comparison_ud_rows(
    *,
    approach: dict[str, dict[str, dict[str, Any]]],
    ud_metrics: Sequence[MetricSpec],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Per ``case/setup`` up/down table with Δ, ease_winner, and err_winner.

    Columns are ``case``, ``setup``, ``ease_winner``, ``err_winner``, then
    ``{label}_{upslope|downslope|Δ}`` for each UD metric.
    ``Δ = upslope − downslope``.
    """
    fieldnames = ["case", "setup", "ease_winner", "err_winner"]
    for metric in ud_metrics:
        for suffix in ("upslope", "downslope", "Δ"):
            fieldnames.append(f"{metric.label}_{suffix}")

    rows: list[dict[str, str]] = []
    for case in sorted(approach, key=natural_case_key):
        setups = approach.get(case) or {}
        for setup in ("a", "b"):
            payload = setups.get(setup)
            diagnostics = (payload or {}).get("diagnostics") or {}
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            row: dict[str, str] = {
                "case": case,
                "setup": setup,
                "ease_winner": str(diagnostics.get("winner") or ""),
                "err_winner": str(diagnostics.get("err_winner") or ""),
            }
            for metric in ud_metrics:
                val_up = orientation_metric_value(payload, "upslope", metric.key)
                val_down = orientation_metric_value(payload, "downslope", metric.key)
                row[f"{metric.label}_upslope"] = _scalar_or_empty(val_up)
                row[f"{metric.label}_downslope"] = _scalar_or_empty(val_down)
                row[f"{metric.label}_Δ"] = _ud_delta_str(val_up, val_down)
            rows.append(row)
    return fieldnames, rows


def write_comparison_ud_md(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    path: Path,
    *,
    method_id: str,
    title: str | None = None,
) -> None:
    """Markdown upslope vs downslope comparison table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    heading = title or f"Upslope / downslope — {method_id}"
    lines = [
        f"# {heading}",
        "",
        "Per case/setup orientation metrics (Δ = upslope − downslope). "
        "ease_winner and err_winner come from diagnostics.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(c, "") for c in fieldnames) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_approach_plots(
    *,
    config: ApproachCompareConfig,
    baseline: dict[str, dict[str, dict[str, Any]]],
    approach: dict[str, dict[str, dict[str, Any]]],
    plots_dir: Path,
) -> list[Path]:
    """Write vs-baseline and A/B grouped-bar plots for the method config."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    case_names = sorted(
        set(baseline) | set(approach),
        key=natural_case_key,
    )
    labels = [f"{case}/{setup}" for case in case_names for setup in ("a", "b")]

    for metric in config.vs_baseline_metrics:
        ylabel = metric.ylabel or metric.label
        baseline_vals = [
            metric_value((baseline.get(case) or {}).get(setup), metric.key)
            for case in case_names
            for setup in ("a", "b")
        ]
        approach_vals = [
            metric_value((approach.get(case) or {}).get(setup), metric.key)
            for case in case_names
            for setup in ("a", "b")
        ]
        path = plots_dir / f"vs_baseline_{metric.label}.png"
        plot_vs_baseline_grouped_bars(
            labels,
            baseline_vals,
            approach_vals,
            ylabel=ylabel,
            title=f"{config.method_id} — {metric.label} vs baseline",
            path=path,
            approach_label=config.method_id,
        )
        written.append(path)

    for metric in config.ab_metrics:
        ylabel = metric.ylabel or metric.label
        vals_a = [
            metric_value((approach.get(case) or {}).get("a"), metric.key)
            for case in case_names
        ]
        vals_b = [
            metric_value((approach.get(case) or {}).get("b"), metric.key)
            for case in case_names
        ]
        path = plots_dir / f"ab_{metric.label}.png"
        plot_ab_grouped_bars(
            case_names,
            vals_a,
            vals_b,
            ylabel=ylabel,
            title=f"{config.method_id} — paired A/B {metric.label}",
            path=path,
        )
        written.append(path)

    if config.ud_metrics:
        ease_winners: list[str | None] = []
        for case in case_names:
            for setup in ("a", "b"):
                payload = (approach.get(case) or {}).get(setup) or {}
                diagnostics = payload.get("diagnostics") or {}
                winner = (
                    diagnostics.get("winner") if isinstance(diagnostics, dict) else None
                )
                ease_winners.append(str(winner) if winner else None)

        for metric in config.ud_metrics:
            ylabel = metric.ylabel or metric.label
            vals_up = [
                orientation_metric_value(
                    (approach.get(case) or {}).get(setup), "upslope", metric.key
                )
                for case in case_names
                for setup in ("a", "b")
            ]
            vals_down = [
                orientation_metric_value(
                    (approach.get(case) or {}).get(setup), "downslope", metric.key
                )
                for case in case_names
                for setup in ("a", "b")
            ]
            path = plots_dir / f"ud_{metric.label}.png"
            plot_ud_grouped_bars(
                labels,
                vals_up,
                vals_down,
                ease_winners=ease_winners,
                ylabel=ylabel,
                title=f"{config.method_id} — up/down {metric.label}",
                path=path,
            )
            written.append(path)

    for metric in config.ratio_metrics:
        ylabel = metric.ylabel or metric.label
        vals_a = [
            ratio_metric_value(
                (approach.get(case) or {}).get("a"),
                (baseline.get(case) or {}).get("a"),
                metric,
            )
            for case in case_names
        ]
        vals_b = [
            ratio_metric_value(
                (approach.get(case) or {}).get("b"),
                (baseline.get(case) or {}).get("b"),
                metric,
            )
            for case in case_names
        ]
        path = plots_dir / f"ab_{metric.label}.png"
        plot_ab_grouped_bars(
            case_names,
            vals_a,
            vals_b,
            ylabel=ylabel,
            title=f"{config.method_id} — paired A/B {metric.label}",
            path=path,
        )
        written.append(path)

    return written


def iterate_comparison_cases(
    cases: Sequence[Any],
    *,
    case_names: Sequence[str] | None = None,
) -> list[tuple[str, str, Any]]:
    """
    Expand comparison cases into ``(case_name, setup_key, setup)`` triples.

    ``cases`` is typically ``COMPARISON_CASES`` from the slab tensile tests.
    """
    selected = tuple(cases)
    if case_names:
        wanted = set(case_names)
        selected = tuple(c for c in cases if c.name in wanted)
        missing = wanted - {c.name for c in selected}
        if missing:
            raise ValueError(f"Unknown case name(s): {sorted(missing)}")

    out: list[tuple[str, str, Any]] = []
    for case in selected:
        out.append((case.name, "a", case.setup_a))
        out.append((case.name, "b", case.setup_b))
    return out


def evaluate_setup_soft(
    evaluate: Callable[[Any], ExperimentalSteadyStateResult],
    setup: Any,
    *,
    case: str,
    setup_key: str,
    method: str,
) -> dict[str, Any]:
    """Evaluate one setup; soft-fail into an error payload on exception."""
    try:
        result = evaluate(setup)
        payload = serialize_result(result)
        payload.update(
            {
                "case": case,
                "setup": setup_key,
                "method": method,
                "ok": True,
                "error": None,
            }
        )
        return payload
    except Exception as exc:  # noqa: BLE001 — soft-fail matrix cell
        err = f"{type(exc).__name__}: {exc}"
        return {
            "case": case,
            "setup": setup_key,
            "method": method,
            "ok": False,
            "error": err,
            "traceback": traceback.format_exc(),
            "converged": False,
            "message": err,
            "characteristic_length": None,
            "energy_release_rate": None,
            "max_Sxx_norm": None,
            "thickness_fraction_without_density_gate": None,
            "diagnostics": {},
        }


def run_approach_comparison(
    config: ApproachCompareConfig,
    *,
    cases: Sequence[Any],
    out_dir: Path,
    baseline_path: Path | None = None,
    case_names: Sequence[str] | None = None,
    skip_plots: bool = False,
) -> dict[str, Any]:
    """
    Load baseline, evaluate the approach over cases, write artifacts.

    Writes ``results.json``, ``comparison_ab.{csv,md}``, and plots under
    ``out_dir``. Raises ``MissingBaselineError`` if the baseline snapshot is
    absent — never recomputes baseline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    baseline = load_baseline_results(baseline_path)

    approach_nested: dict[str, dict[str, dict[str, Any]]] = {}
    cells = iterate_comparison_cases(cases, case_names=case_names)
    total = len(cells)
    for i, (case_name, setup_key, setup) in enumerate(cells, start=1):
        print(
            f"[{i}/{total}] {case_name}/{setup_key}/{config.method_id} ...",
            flush=True,
        )
        payload = evaluate_setup_soft(
            config.evaluate,
            setup,
            case=case_name,
            setup_key=setup_key,
            method=config.method_id,
        )
        status = "ok" if payload.get("ok") else f"ERROR: {payload.get('error')}"
        print(f"    -> {status}", flush=True)
        approach_nested.setdefault(case_name, {})[setup_key] = payload

    results_path = out_dir / "results.json"
    write_results_json(approach_nested, results_path)
    print(f"Wrote {results_path}", flush=True)

    ordered_cases = sorted(approach_nested, key=natural_case_key)
    fieldnames, rows = build_comparison_ab_rows(
        cases=ordered_cases,
        baseline=baseline,
        approach=approach_nested,
        method_id=config.method_id,
        baseline_metrics=config.vs_baseline_metrics,
        approach_metrics=config.ab_metrics,
    )
    ab_csv = out_dir / "comparison_ab.csv"
    ab_md = out_dir / "comparison_ab.md"
    write_comparison_csv(fieldnames, rows, ab_csv)
    write_comparison_ab_md(
        fieldnames,
        rows,
        ab_md,
        method_id=config.method_id,
        title=config.title,
    )
    print(f"Wrote {ab_csv}", flush=True)
    print(f"Wrote {ab_md}", flush=True)

    ud_csv: Path | None = None
    ud_md: Path | None = None
    if config.ud_metrics:
        ud_fieldnames, ud_rows = build_comparison_ud_rows(
            approach=approach_nested,
            ud_metrics=config.ud_metrics,
        )
        ud_csv = out_dir / "comparison_ud.csv"
        ud_md = out_dir / "comparison_ud.md"
        write_comparison_csv(ud_fieldnames, ud_rows, ud_csv)
        write_comparison_ud_md(
            ud_fieldnames,
            ud_rows,
            ud_md,
            method_id=config.method_id,
            title=f"Upslope / downslope — {config.method_id}",
        )
        print(f"Wrote {ud_csv}", flush=True)
        print(f"Wrote {ud_md}", flush=True)

    plot_paths: list[Path] = []
    if not skip_plots:
        plot_paths = write_approach_plots(
            config=config,
            baseline=baseline,
            approach=approach_nested,
            plots_dir=plots_dir,
        )
        for path in plot_paths:
            print(f"  wrote {path}", flush=True)

    return {
        "results_path": results_path,
        "comparison_ab_csv": ab_csv,
        "comparison_ab_md": ab_md,
        "comparison_ud_csv": ud_csv,
        "comparison_ud_md": ud_md,
        "plots": plot_paths,
        "results": approach_nested,
    }
