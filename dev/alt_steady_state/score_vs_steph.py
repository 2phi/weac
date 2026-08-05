"""
Score comparison_ab.csv trees vs Steph ground truth; write side-by-side markdown.

    uv run python dev/alt_steady_state/score_vs_steph.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ROOT = Path(__file__).resolve().parent
OUT_MD = _REPO_ROOT / "todo" / "slab-tensile-comparison-results.md"

APPROACHES = (
    "pst_critical_cut",
    "pst_critical_mass",
    "pst_fixed_cut",
    "pst_touchdown_cut",
)

# Tensile-ease ranking metric per approach (from slab-tensile-comparison-results.md).
EASE_SPEC: dict[str, tuple[str, str]] = {
    "pst_critical_cut": ("L_crit", "lower"),
    "pst_critical_mass": ("critical_mass", "lower"),
    "pst_fixed_cut": ("thickness_fraction", "higher"),
    "pst_touchdown_cut": ("thickness_fraction", "higher"),
}

# Steph yardstick (same as existing comparison-results table).
STEPH_EASE: dict[int, str] = {
    **{n: "A" for n in range(1, 12)},
    **{n: "B" for n in range(12, 16)},
    **{n: "A" for n in range(16, 24)},
}
STEPH_ERR: dict[int, str] = {
    **{n: "B" for n in range(1, 12)},
    **{n: "A" for n in range(12, 16)},
    **{n: "B" for n in range(16, 21)},
    **{n: "A" for n in range(21, 24)},
}

# Law label → directory with approach comparison_ab.csv files.
LAW_DIRS: dict[str, Path] = {
    "JJ": ROOT,  # current default artifacts at tree root
    "hybrid": ROOT / "ts_hybrid",
    "adam": ROOT / "ts_adam",
}


def _parse_float(raw: str) -> float | None:
    s = (raw or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _winner(a: float | None, b: float | None, *, preference: str) -> str:
    if a is None or b is None:
        return "-"
    if a == b:
        return "="
    if preference == "higher":
        return "A" if a > b else "B"
    if preference == "lower":
        return "A" if a < b else "B"
    raise ValueError(preference)


def _mark(winner: str, steph: str) -> str:
    if winner == "-":
        return "-"
    if winner == "=":
        return "="
    return "✅" if winner == steph else "❌"


def load_ab_table(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        case = row.get("case") or ""
        if not case.startswith("case_"):
            continue
        out[int(case.split("_", 1)[1])] = row
    return out


def score_cell(
    row: dict[str, str] | None,
    *,
    approach: str,
    metric: str,
    preference: str,
    steph: str,
) -> str:
    if not row:
        return "-"
    a = _parse_float(row.get(f"{approach}.{metric}_a", ""))
    b = _parse_float(row.get(f"{approach}.{metric}_b", ""))
    return _mark(_winner(a, b, preference=preference), steph)


def score_law(law_dir: Path) -> dict[str, dict[int, dict[str, str]]]:
    """Return {table: {case: {approach: mark}}} for ease and ERR."""
    ease: dict[int, dict[str, str]] = {}
    err: dict[int, dict[str, str]] = {}
    tables = {
        approach: load_ab_table(law_dir / approach / "comparison_ab.csv")
        for approach in APPROACHES
    }
    for case in range(1, 24):
        ease[case] = {}
        err[case] = {}
        for approach in APPROACHES:
            metric, pref = EASE_SPEC[approach]
            ease[case][approach] = score_cell(
                tables[approach].get(case),
                approach=approach,
                metric=metric,
                preference=pref,
                steph=STEPH_EASE[case],
            )
            err[case][approach] = score_cell(
                tables[approach].get(case),
                approach=approach,
                metric="ERR",
                preference="higher",
                steph=STEPH_ERR[case],
            )
    return {"ease": ease, "err": err}


def _fmt_side_by_side(marks: dict[str, str]) -> str:
    """JJ / hybrid / adam in one cell."""
    parts = [marks.get(law, "-") for law in ("JJ", "hybrid", "adam")]
    return "/".join(parts)


def render_markdown(scores: dict[str, dict[str, dict[int, dict[str, str]]]]) -> str:
    laws = ("JJ", "hybrid", "adam")
    lines = [
        "# PST A/B comparison results (tensile-strength law sweep)",
        "",
        "Side-by-side marks are **JJ / hybrid / adam**.",
        "",
        "- **JJ** = current default (`jamieson_johnson`) at `dev/alt_steady_state/<approach>/`",
        "- **hybrid** / **adam** = `dev/alt_steady_state/ts_<law>/<approach>/`",
        "",
        "Method cells: ✅ = matches Steph · ❌ = mismatches · `=` = A/B tie · `-` = missing/error.",
        "",
        "**Steph** = expected winning side (letter).",
        "",
        "---",
        "",
        "## 1 — Tensile ease",
        "",
        "Steph = easier side. Scalars: `L_crit` ↓ · `critical_mass` ↓ · "
        "`thickness_fraction` ↑ "
        "(`pst_fixed_cut` ranked on thickness fraction).",
        "",
        "| Case | Steph |  | pst_critical_cut<br>JJ/hy/ad | pst_critical_mass<br>JJ/hy/ad | "
        "pst_fixed_cut<br>JJ/hy/ad | pst_touchdown_cut<br>JJ/hy/ad |",
        "|:----:|:-----:|:-:|:---------------------------:|:----------------------------:|"
        ":------------------------:|:----------------------------:|",
    ]
    for case in range(1, 24):
        cells = [
            _fmt_side_by_side(
                {law: scores[law]["ease"][case][approach] for law in laws}
            )
            for approach in APPROACHES
        ]
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
        "Steph = side with higher ERR (intuition). Scalar: production `ERR` (higher wins).",
        "",
        "| Case | Steph |  | pst_critical_cut<br>JJ/hy/ad | pst_critical_mass<br>JJ/hy/ad | "
        "pst_fixed_cut<br>JJ/hy/ad | pst_touchdown_cut<br>JJ/hy/ad |",
        "|:----:|:-----:|:-:|:---------------------------:|:----------------------------:|"
        ":------------------------:|:----------------------------:|",
    ]
    for case in range(1, 24):
        cells = [
            _fmt_side_by_side(
                {law: scores[law]["err"][case][approach] for law in laws}
            )
            for approach in APPROACHES
        ]
        lines.append(
            f"| {case} | {STEPH_ERR[case]} |  | {cells[0]} | {cells[1]} | "
            f"{cells[2]} | {cells[3]} |"
        )

    # Compact tallies
    lines += ["", "---", "", "## Tallies (✅ count / 23)", ""]
    lines.append("| Law | Table | " + " | ".join(APPROACHES) + " |")
    lines.append("|:---:|:-----:|" + "|".join([":---:" ] * len(APPROACHES)) + "|")
    for law in laws:
        for table, label in (("ease", "tensile ease"), ("err", "ERR")):
            counts = []
            for approach in APPROACHES:
                n = sum(
                    1
                    for case in range(1, 24)
                    if scores[law][table][case][approach] == "✅"
                )
                counts.append(str(n))
            lines.append(f"| {law} | {label} | " + " | ".join(counts) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    scores = {law: score_law(path) for law, path in LAW_DIRS.items()}
    missing = [
        law
        for law, path in LAW_DIRS.items()
        if law != "JJ"
        and not all((path / a / "comparison_ab.csv").is_file() for a in APPROACHES)
    ]
    text = render_markdown(scores)
    OUT_MD.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT_MD}", flush=True)
    if missing:
        print(f"WARNING: incomplete law dirs: {missing}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
