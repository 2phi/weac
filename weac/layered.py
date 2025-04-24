"""Class for the elastic analysis of layered snow slabs."""

# Project imports
from weac.mixins import FieldQuantitiesMixin
from weac.mixins import SolutionMixin
from weac.mixins import AnalysisMixin
from weac.mixins import OutputMixin
from weac.eigensystem import Eigensystem


class Layered(FieldQuantitiesMixin, SolutionMixin, AnalysisMixin,
              OutputMixin, Eigensystem):
    """
    Layered beam on elastic foundation model application interface.

    Inherits methods for the eigensystem calculation from the base
    class Eigensystem(), methods for the calculation of field
    quantities from FieldQuantitiesMixin(), methods for the solution
    of the system from SolutionMixin() and methods for the output
    analysis from AnalysisMixin().
    """

    def __init__(self, system='pst-', layers=None, phi=0, theta=0, b=290):
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
        """
        # Call parent __init__
        super().__init__(system=system)

        # Set material properties and set up model
        self.set_beam_properties(layers if layers else [[240, 200], ], phi=phi, theta=theta, b=b)
        self.calc_fundamental_system()
        #self.calc_touchdown_system()
