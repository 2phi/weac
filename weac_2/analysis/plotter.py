# Standard library imports
from functools import partial
# Third party imports
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import brentq
# Module imports

from weac_2.core.system_model import SystemModel

class Plotter:
    """
    Provides methods for the analysis of layered slabs on compliant
    elastic foundations.
    """
    system: SystemModel
    
    def __init__(self, system: SystemModel):
        self.system = system
