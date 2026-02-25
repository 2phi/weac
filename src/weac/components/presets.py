"""Named presets for WEAC components."""

from weac.components.layer import WeakLayer

_WEAK_LAYER_PARAMS: dict[str, dict] = {
    "very_weak": {
        "rho": 125,
        "h": 10,
        "sigma_c": 5.16,
        "tau_c": 4.09,
        "E": 2.0,
    },
    "weak": {
        "rho": 125,
        "h": 10,
        "sigma_c": 6.16,
        "tau_c": 5.09,
        "E": 2.0,
    },
    "less_weak": {
        "rho": 125,
        "h": 10,
        "sigma_c": 7.16,
        "tau_c": 6.09,
        "E": 2.0,
    },
}

VERY_WEAK_LAYER = WeakLayer(**_WEAK_LAYER_PARAMS["very_weak"])
WEAK_LAYER = WeakLayer(**_WEAK_LAYER_PARAMS["weak"])
LESS_WEAK_LAYER = WeakLayer(**_WEAK_LAYER_PARAMS["less_weak"])

WEAK_LAYER_PRESETS: dict[str, WeakLayer] = {
    "very_weak": VERY_WEAK_LAYER,
    "weak": WEAK_LAYER,
    "less_weak": LESS_WEAK_LAYER,
}


def weak_layer_from_preset(name: str, **overrides) -> WeakLayer:
    """Create a WeakLayer from a named preset, with optional overrides.

    Without overrides, returns the shared frozen instance.
    With overrides, returns a new instance.
    """
    if name not in _WEAK_LAYER_PARAMS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(_WEAK_LAYER_PARAMS)}"
        )
    if not overrides:
        return WEAK_LAYER_PRESETS[name]
    params = {**_WEAK_LAYER_PARAMS[name], **overrides}
    return WeakLayer(**params)
