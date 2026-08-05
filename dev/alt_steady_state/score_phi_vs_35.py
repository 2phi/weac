"""
Score φ=15° A/B tree vs Steph and vs existing φ=35° JJ rankings.

    uv run python dev/alt_steady_state/score_phi_vs_35.py
    uv run python dev/alt_steady_state/score_phi_vs_35.py --phi-dir phi_15
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

from score_vs_steph import (  # noqa: E402
    APPROACHES,
    STEPH_EASE,
    STEPH_ERR,
    score_law,
)

OUT_MD = _REPO_ROOT / "todo" / "slab-tensile-comparison-phi15.md"


def render(
    scores_35: dict[str, dict[int, dict[str, str]]],
    scores_phi: dict[str, dict[int, dict[str, str]]],
    *,
    phi_label: str,
) -> str:
    lines = [
        f"# PST A/B comparison — φ=35° vs {phi_label} (JJ)",
        "",
        "Side-by-side marks are **35° / "
        f"{phi_label}** vs Steph.",
        "",
        "- **35°** = `dev/alt_steady_state/<approach>/`",
        f"- **{phi_label}** = `dev/alt_steady_state/phi_"
        f"{phi_label.replace('°', '')}/`",
        "",
        "Method cells: ✅ = matches Steph · ❌ = mismatches · `=` = tie · `-` = missing.",
        "",
        "---",
        "",
        "## 1 — Tensile ease",
        "",
        "| Case | Steph |  | pst_critical_cut<br>35/"
        f"{phi_label} | pst_critical_mass<br>35/{phi_label} | "
        f"pst_fixed_cut<br>35/{phi_label} | pst_touchdown_cut<br>35/{phi_label} |",
        "|:----:|:-----:|:-:|:---:|:---:|:---:|:---:|",
    ]
    for case in range(1, 24):
        cells = []
        for approach in APPROACHES:
            m35 = scores_35["ease"][case][approach]
            m15 = scores_phi["ease"][case][approach]
            cells.append(f"{m35}/{m15}")
        lines.append(
            f"| {case} | {STEPH_EASE[case]} |  | {cells[0]} | {cells[1]} | "
            f"{cells[2]} | {cells[3]} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2 — ERR",
        "",
        "| Case | Steph |  | pst_critical_cut<br>35/"
        f"{phi_label} | pst_critical_mass<br>35/{phi_label} | "
        f"pst_fixed_cut<br>35/{phi_label} | pst_touchdown_cut<br>35/{phi_label} |",
        "|:----:|:-----:|:-:|:---:|:---:|:---:|:---:|",
    ]
    for case in range(1, 24):
        cells = []
        for approach in APPROACHES:
            m35 = scores_35["err"][case][approach]
            m15 = scores_phi["err"][case][approach]
            cells.append(f"{m35}/{m15}")
        lines.append(
            f"| {case} | {STEPH_ERR[case]} |  | {cells[0]} | {cells[1]} | "
            f"{cells[2]} | {cells[3]} |"
        )

    # Agreement between 35 and phi (same A/B winner, ignoring Steph)
    lines += [
        "",
        "---",
        "",
        f"## 3 — A/B winner agreement (35° vs {phi_label})",
        "",
        "Same A/B side chosen (✅/❌/='/'- treated as the mark string match).",
        "",
        "| Table | " + " | ".join(APPROACHES) + " |",
        "|:-----:|" + "|".join([":---:"] * len(APPROACHES)) + "|",
    ]
    for table, label in (("ease", "tensile ease"), ("err", "ERR")):
        counts = []
        for approach in APPROACHES:
            n = sum(
                1
                for case in range(1, 24)
                if scores_35[table][case][approach]
                == scores_phi[table][case][approach]
                and scores_35[table][case][approach] not in ("-",)
            )
            # also count when both missing? skip. Count agreement including ties.
            n_agree = sum(
                1
                for case in range(1, 24)
                if scores_35[table][case][approach]
                == scores_phi[table][case][approach]
            )
            counts.append(f"{n_agree}/23")
        lines.append(f"| {label} | " + " | ".join(counts) + " |")

    lines += [
        "",
        "## Tallies vs Steph (✅ / 23)",
        "",
        "| φ | Table | " + " | ".join(APPROACHES) + " |",
        "|:-:|:-----:|" + "|".join([":---:"] * len(APPROACHES)) + "|",
    ]
    for label, scores in (("35°", scores_35), (phi_label, scores_phi)):
        for table, tlabel in (("ease", "tensile ease"), ("err", "ERR")):
            counts = [
                str(
                    sum(
                        1
                        for case in range(1, 24)
                        if scores[table][case][approach] == "✅"
                    )
                )
                for approach in APPROACHES
            ]
            lines.append(f"| {label} | {tlabel} | " + " | ".join(counts) + " |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phi-dir",
        default="phi_15",
        help="Directory under alt_steady_state (default: phi_15).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_MD,
        help=f"Markdown output (default: {OUT_MD}).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dir_35 = ROOT
    dir_phi = ROOT / args.phi_dir
    if not dir_phi.is_dir():
        print(f"ERROR: missing {dir_phi}", file=sys.stderr)
        return 1

    scores_35 = score_law(dir_35)
    scores_phi = score_law(dir_phi)
    # Derive label from folder name
    phi_label = args.phi_dir.replace("phi_", "") + "°"
    text = render(scores_35, scores_phi, phi_label=phi_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)

    # Print compact agreement summary to stdout
    print("\nAgreement 35° vs " + phi_label + " (identical Steph marks):", flush=True)
    for table, label in (("ease", "tensile ease"), ("err", "ERR")):
        for approach in APPROACHES:
            n = sum(
                1
                for case in range(1, 24)
                if scores_35[table][case][approach]
                == scores_phi[table][case][approach]
            )
            n_ok35 = sum(
                1
                for case in range(1, 24)
                if scores_35[table][case][approach] == "✅"
            )
            n_ok15 = sum(
                1
                for case in range(1, 24)
                if scores_phi[table][case][approach] == "✅"
            )
            print(
                f"  {label:14s} {approach:20s} agree {n}/23  "
                f"Steph✅ 35°={n_ok35} {phi_label}={n_ok15}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
