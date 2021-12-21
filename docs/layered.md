Module weac.layered
===================
Class for the elastic analysis of layered snow slabs.

Classes
-------

`Layered(system='pst-', layers=None)`
:   Layered beam on elastic foundation model application interface.
    
    Inherits methods for the eigensystem calculation from the base
    class Eigensystem(), methods for the calculation of field
    quantities from FieldQuantitiesMixin(), methods for the solution
    of the system from SolutionMixin() and methods for the output
    analysis from AnalysisMixin().
    
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

    ### Ancestors (in MRO)

    * weac.mixins.FieldQuantitiesMixin
    * weac.mixins.SolutionMixin
    * weac.mixins.AnalysisMixin
    * weac.eigensystem.Eigensystem