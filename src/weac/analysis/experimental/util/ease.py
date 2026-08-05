"""Shared ease-based orientation selection for dual-orientation PST methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

OrientationName = Literal["upslope", "downslope"]

_DEFAULT_UNUSABLE_FLAGS: tuple[str, ...] = ("never_cracked", "no_crack")


@dataclass(frozen=True)
class EaseSelection:
    """Result of comparing upslope vs downslope on an ease metric."""

    winner: OrientationName
    err_winner: OrientationName
    selection_rule: str


def is_usable_orientation(
    block: Mapping[str, Any],
    *,
    unusable_if_true: Sequence[str] = _DEFAULT_UNUSABLE_FLAGS,
) -> bool:
    """
    Return whether an orientation block may enter the ease comparison.

    Missing ``converged`` is treated as usable (e.g. fixed-cut diagnostics).
    Truthy values of any name in ``unusable_if_true`` exclude the side
    (``never_cracked`` / ``no_crack``). ``already_cracked`` remains usable.
    """
    if block.get("converged") is False:
        return False
    for flag in unusable_if_true:
        if block.get(flag) is True:
            return False
    return True


def _numeric(block: Mapping[str, Any], key: str) -> float | None:
    value = block.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _higher_err_winner(
    upslope: Mapping[str, Any],
    downslope: Mapping[str, Any],
    *,
    err_key: str,
) -> OrientationName:
    err_up = _numeric(upslope, err_key)
    err_down = _numeric(downslope, err_key)
    if err_up is None and err_down is None:
        return "upslope"
    if err_up is None:
        return "downslope"
    if err_down is None:
        return "upslope"
    if err_up >= err_down:
        return "upslope"
    return "downslope"


def select_ease_orientation(
    upslope: Mapping[str, Any],
    downslope: Mapping[str, Any],
    *,
    ease_key: str,
    higher_is_easier: bool,
    err_key: str = "energy_release_rate",
    unusable_if_true: Sequence[str] = _DEFAULT_UNUSABLE_FLAGS,
) -> EaseSelection:
    """
    Pick the production orientation by ease among usable sides.

    Order: usable-side filter → ease compare → ERR tie-break → upslope default.
    ``selection_rule`` is always ``ease:<ease_key>``.
    """
    selection_rule = f"ease:{ease_key}"
    err_winner = _higher_err_winner(upslope, downslope, err_key=err_key)

    sides: dict[OrientationName, Mapping[str, Any]] = {
        "upslope": upslope,
        "downslope": downslope,
    }
    usable: list[OrientationName] = [
        name
        for name, block in sides.items()
        if is_usable_orientation(block, unusable_if_true=unusable_if_true)
        and _numeric(block, ease_key) is not None
    ]

    if len(usable) == 0:
        return EaseSelection(
            winner=err_winner,
            err_winner=err_winner,
            selection_rule=selection_rule,
        )
    if len(usable) == 1:
        return EaseSelection(
            winner=usable[0],
            err_winner=err_winner,
            selection_rule=selection_rule,
        )

    ease_up = _numeric(upslope, ease_key)
    ease_down = _numeric(downslope, ease_key)
    assert ease_up is not None and ease_down is not None

    if ease_up == ease_down:
        winner: OrientationName = err_winner
    elif higher_is_easier:
        winner = "upslope" if ease_up > ease_down else "downslope"
    else:
        winner = "upslope" if ease_up < ease_down else "downslope"

    return EaseSelection(
        winner=winner,
        err_winner=err_winner,
        selection_rule=selection_rule,
    )
