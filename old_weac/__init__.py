"""
WEak Layer AntiCrack nucleation model.

Implementation of closed-form analytical models for the analysis of
dry-snow slab avalanche release.
"""

# Module imports
from old_weac.layered import Layered
from old_weac.inverse import Inverse
from old_weac import plot

# Version
__version__ = "2.6.1"

# Public names
__all__ = ["Layered", "Inverse", "plot"]
