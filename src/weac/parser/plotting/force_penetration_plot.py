"""SMP and SnowScope penetration-resistance debug plotting (unified entry points).

Penetration parsers record **penetration resistance first**, then derive sample
density (SnowMicroPen via snowmicropyn; SnowScope via a semilog hardness model).
SnowPilot pits are layer lists only — they do not expose this resistance–depth
signal, so this package does not apply to
:class:`~weac.parser.snowpilot_parser.SnowPilotParser`.

Four depth-aligned panels (see :func:`plot_force_penetration_layers`):

1. Penetration resistance vs plumb depth.
2. Sample density vs plumb depth.
3. Sample density vs slope-normal depth (plumb depth compressed by ``cos(phi)``).
4. Segmented WEAC layers (``bin`` or ``gradient``) vs slope-normal depth, with
   the slope-normal density faintly overlaid.

Parser-specific extract-and-plot helpers live in
:mod:`weac.parser.plotting.smp_penetration_plot` and
:mod:`weac.parser.plotting.snowscope_penetration_plot`.
"""

from __future__ import annotations

from typing import Literal, Protocol

from matplotlib.figure import Figure

from weac.components import Layer
from weac.parser.plotting.force_penetration_layers import (
    ForcePenetrationProfile,
    plot_force_penetration_layers,
)
from weac.parser.plotting.smp_penetration_plot import plot_smp_penetration_parser
from weac.parser.plotting.snowscope_penetration_plot import (
    plot_snowscope_penetration_parser,
)
from weac.parser.smp_parser import SMPParser
from weac.parser.snowscope_parser import SnowScopeParser

__all__ = [
    "ForcePenetrationParser",
    "ForcePenetrationProfile",
    "plot_force_penetration_layers",
    "plot_force_penetration_parser",
    "plot_smp_penetration_parser",
    "plot_snowscope_penetration_parser",
]


class ForcePenetrationParser(Protocol):
    """SMP or SnowScope parser with ``extract_profile`` / ``extract_layers``."""

    def extract_profile(self, **kwargs) -> ForcePenetrationProfile: ...

    def extract_layers(self, *args, **kwargs) -> tuple[list[Layer], list[str]]: ...


def plot_force_penetration_parser(
    parser: ForcePenetrationParser,
    *,
    method: Literal["bin", "gradient"] = "bin",
    slope_angle_deg: float = 0.0,
    density_method: str | None = None,
    semilog_slope: float | None = None,
    semilog_intercept: float | None = None,
    layer_thickness_mm: float | None = None,
    gradient_threshold: float = 12.0,
    **plot_kwargs,
) -> Figure:
    """Extract from an SMP or SnowScope parser and plot the debug view.

    Dispatches to :func:`plot_smp_penetration_parser` or
    :func:`plot_snowscope_penetration_parser`. ``semilog_slope`` /
    ``semilog_intercept`` apply only to SnowScope. ``**plot_kwargs`` (e.g.
    ``title``, ``save_path``, ``show``) pass through to
    :func:`plot_force_penetration_layers`.
    """
    if isinstance(parser, SnowScopeParser):
        return plot_snowscope_penetration_parser(
            parser,
            method=method,
            slope_angle_deg=slope_angle_deg,
            density_method=density_method,  # type: ignore[arg-type]
            semilog_slope=semilog_slope,
            semilog_intercept=semilog_intercept,
            layer_thickness_mm=layer_thickness_mm,
            gradient_threshold=gradient_threshold,
            **plot_kwargs,
        )
    if isinstance(parser, SMPParser):
        if semilog_slope is not None or semilog_intercept is not None:
            raise TypeError(
                "semilog_slope and semilog_intercept apply only to SnowScopeParser"
            )
        return plot_smp_penetration_parser(
            parser,
            method=method,
            slope_angle_deg=slope_angle_deg,
            density_method=density_method,  # type: ignore[arg-type]
            layer_thickness_mm=layer_thickness_mm,
            gradient_threshold=gradient_threshold,
            **plot_kwargs,
        )
    raise TypeError(
        "plot_force_penetration_parser expects SMPParser or SnowScopeParser, "
        f"got {type(parser).__name__}"
    )
