"""
WEak Layer AntiCrack nucleation model.

Implementation of closed-form analytical models for the analysis of
dry-snow slab avalanche release.
"""

# Module imports
from weac import plot
from weac.inverse import Inverse
from weac.layered import Layered

# Version
__version__ = "2.6.4"

# Public names
__all__ = ["Layered", "Inverse", "plot"]
