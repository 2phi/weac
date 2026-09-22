"""Deprecated import path for :class:`~weac.parser.snowpilot_parser.SnowPilotParser`."""

from __future__ import annotations

import warnings

from weac.parser.snowpilot_parser import SnowPilotParser

warnings.warn(
    "Importing from weac.utils.snowpilot_parser is deprecated; "
    "use weac.parser or weac.parser.snowpilot_parser instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SnowPilotParser"]
