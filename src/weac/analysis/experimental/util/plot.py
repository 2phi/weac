"""Grouped-bar plot helpers for approach vs baseline, A/B, and up/down."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _as_float_or_nan(value: float | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _outline_ease_winner(
    bars_up,
    bars_down,
    ease_winners: Sequence[str | None],
) -> None:
    """Draw a black outline on the ease-winning bar in each group."""
    for i, winner in enumerate(ease_winners):
        if winner == "upslope":
            bars_up[i].set_edgecolor("black")
            bars_up[i].set_linewidth(1.8)
            bars_up[i].set_zorder(3)
        elif winner == "downslope":
            bars_down[i].set_edgecolor("black")
            bars_down[i].set_linewidth(1.8)
            bars_down[i].set_zorder(3)


def plot_vs_baseline_grouped_bars(
    labels: Sequence[str],
    baseline_values: Sequence[float | None],
    approach_values: Sequence[float | None],
    *,
    ylabel: str,
    title: str,
    path: Path,
    baseline_label: str = "baseline_ss",
    approach_label: str,
) -> Path:
    """
    Grouped bars: baseline | approach for each ``case/setup`` label.

    Used by approach runners for vs-baseline tracking plots.
    """
    if not (len(labels) == len(baseline_values) == len(approach_values)):
        raise ValueError("labels and value sequences must have equal length")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.45), 6))
    ax.bar(
        x - width / 2,
        [_as_float_or_nan(v) for v in baseline_values],
        width=width,
        label=baseline_label,
    )
    ax.bar(
        x + width / 2,
        [_as_float_or_nan(v) for v in approach_values],
        width=width,
        label=approach_label,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ab_grouped_bars(
    cases: Sequence[str],
    values_a: Sequence[float | None],
    values_b: Sequence[float | None],
    *,
    ylabel: str,
    title: str,
    path: Path,
    label_a: str = "setup a",
    label_b: str = "setup b",
) -> Path:
    """
    Grouped bars: setup a | b per case (approach only).

    Used by approach runners for A/B judgment plots.
    """
    if not (len(cases) == len(values_a) == len(values_b)):
        raise ValueError("cases and value sequences must have equal length")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(cases))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(cases) * 0.55), 5))
    ax.bar(
        x - width / 2,
        [_as_float_or_nan(v) for v in values_a],
        width=width,
        label=label_a,
    )
    ax.bar(
        x + width / 2,
        [_as_float_or_nan(v) for v in values_b],
        width=width,
        label=label_b,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(cases), rotation=90, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ud_grouped_bars(
    labels: Sequence[str],
    values_upslope: Sequence[float | None],
    values_downslope: Sequence[float | None],
    *,
    ease_winners: Sequence[str | None],
    ylabel: str,
    title: str,
    path: Path,
    label_upslope: str = "upslope",
    label_downslope: str = "downslope",
) -> Path:
    """
    Grouped bars: upslope | downslope per ``case/setup`` label.

    The ease-winning bar in each group gets a black outline.
    """
    n = len(labels)
    if not (n == len(values_upslope) == len(values_downslope) == len(ease_winners)):
        raise ValueError(
            "labels, value sequences, and ease_winners must have equal length"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 6))
    bars_up = ax.bar(
        x - width / 2,
        [_as_float_or_nan(v) for v in values_upslope],
        width=width,
        label=label_upslope,
    )
    bars_down = ax.bar(
        x + width / 2,
        [_as_float_or_nan(v) for v in values_downslope],
        width=width,
        label=label_downslope,
    )
    _outline_ease_winner(bars_up, bars_down, ease_winners)
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), rotation=90, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
