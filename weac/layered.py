"""Class for the elastic analysis of layered snow slabs."""

# Project imports

from weac.eigensystem import Eigensystem
from weac.mixins import (
    AnalysisMixin,
    FieldQuantitiesMixin,
    OutputMixin,
    SlabContactMixin,
    SolutionMixin,
)


class Layered(
    FieldQuantitiesMixin,
    SlabContactMixin,
    SolutionMixin,
    AnalysisMixin,
    OutputMixin,
    Eigensystem,
):
    """
    Layered beam on elastic foundation model application interface.

    Inherits methods for the eigensystem calculation from the base
    class Eigensystem(), methods for the calculation of field
    quantities from FieldQuantitiesMixin(), methods for the solution
    of the system from SolutionMixin() and methods for the output
    analysis from AnalysisMixin().
    """

    def __init__(self, system='pst-', layers=None, phi=0, theta = 0, b = 290, touchdown = False):
        """
        Initializes a layered snow slab model for elastic analysis.
        
        Args:
            system: Specifies the type of system to analyze. Options are 'pst-' (PST cut from the right), '-pst' (PST cut from the left), 'skier' (one skier on an infinite slab), or 'skiers' (multiple skiers on an infinite slab). Default is 'pst-'.
            layers: Optional 2D list specifying layer densities (kg/m³) and thicknesses (mm), where each row represents a layer. Defaults to [[240, 200]].
            phi: Optional rotation angle of the beam in degrees. Default is 0.
            theta: Optional tilt angle of the beam in degrees. Default is 0.
            b: Optional beam width in millimeters. Default is 290.
        
        Sets up the beam and foundation properties and computes the fundamental system based on the provided parameters.
        """
        # Call parent __init__
        super().__init__(system=system)

        # Set material properties and set up model
        self.set_beam_properties(layers if layers else [[240, 200], ], phi = phi, theta=theta, b=b)
        self.set_foundation_properties()
        self.calc_fundamental_system()
