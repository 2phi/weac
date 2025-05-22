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

    def __init__(
        self, system="pst-", layers=None, phi=0, theta=0, b=290, touchdown=False
    ):
        """
        Initialize model with user input.

        Arguments
        ---------
        system : {'pst-', '-pst', 'skier', 'skiers'}, optional
            Type of system to analyse: PST cut from the right (pst-),
            PST cut form the left (-pst), one skier on infinite
            slab (skier) or multiple skiers on infinite slab (skiers).
            Default is 'pst-'.
        layers : list, optional
            2D list of layer densities and thicknesses. Columns are
            density(kg/m ^ 3) and thickness(mm). One row corresponds
            to one layer. Default is [[240, 200], ].
        phi : float
            Inclination of snow slab. Default is 0.
        theta : float
            Rotation around the x-axis of the snow slab. Default is 0.
        b : float
            Out-of-plane (y) width of the snow pack. Default is 290.
        touchdown : bool
            Flag whether touchdown analysis is conducted. Default is False.
        """
        # Call parent __init__
        super().__init__(system=system)

        # Set material properties and set up model
        self.set_beam_properties(
            layers
            if layers
            else [
                [240, 200],
            ],
            phi=phi,
            theta=theta,
            b=b,
        )
        self.set_foundation_properties()
        self.calc_fundamental_system()
