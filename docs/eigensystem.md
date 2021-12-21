Module weac.eigensystem
=======================
Base class for the elastic analysis of layered snow slabs.

Classes
-------

`Eigensystem(system='pst-')`
:   Base class for a layered beam on an elastic foundation.
    
    Provides geometry, material and loading attributes, and methods
    for the assembly of a fundamental system.
    
    Attributes
    ----------
    g : float
        Gravitational constant (mm/s^2). Default is 9180.
    lski : float
        Effective out-of-plance length of skis (mm). Default is 1000.
    tol : float
        Relative Romberg integration toleranc. Default is 1e-3.
    system : str
        Type of boundary value problem. Default is 'pst-'.
    weak : dict
        Dictionary that holds the weak layer properties Young's
        modulus (MPa) and Poisson's raito. Defaults are 0.25
        and 0.25, respectively.
    t : float
        Weak-layer thickness (mm). Default is 30.
    kn : float
        Compressive foundation (weak-layer) stiffness (N/mm^3).
    kt : float
        Shear foundation (weak-layer) stiffness (N/mm^3).
    slab : ndarray
        Matrix that holds the elastic properties of all slab layers.
        Columns are density (kg/m^3), layer heigth (mm), Young's
        modulus (MPa), shear modulus (MPa), and Poisson's ratio.
    k : float
        Shear correction factor of the slab. Default is 5/6.
    h : float
        Slab thickness (mm). Default is 300.
    zs : float
        Z-coordinate of the center of gravity of the slab (mm).
    A11 : float
        Extensional stiffness of the slab (N/mm).
    B11 : float
        Bending-extension coupling stiffness of the slab (N).
    D11 : float
        Bending stiffness of the slab (Nmm).
    kA55 : float
        Shear stiffness of the slab (N/mm).
    E0 : float
        Characteristic stiffness value (N).
    ewC : ndarray
        List of complex eigenvalues.
    ewR : ndarray
        List of real eigenvalues.
    evC : ndarray
        Matrix with eigenvectors corresponding to complex
        eigenvalues as columns.
    evR : ndarray
        Matrix with eigenvectors corresponding to real
        eigenvalues as columns.
    sC : float
        X-coordinate shift (mm) of complex parts of the solution.
        Used for numerical stability.
    sR : float
        X-coordinate shift (mm) of real parts of the solution.
        Used for numerical stability.
    sysmat : ndarray
        System matrix.
    
    Initialize eigensystem with user input.
    
    Arguments
    ---------
    system : {'pst-', '-pst', 'skier', 'skiers'}, optional
        Type of system to analyse: PST cut from the right (pst-),
        PST cut form the left (-pst), one skier on infinite
        slab (skier) or multiple skiers on infinite slab (skeirs).
        Default is 'pst-'.
    layers : list, optional
        2D list of layer densities and thicknesses. Columns are
        density (kg/m^3) and thickness (mm). One row corresponds
        to one layer. Default is [[240, 200], ].

    ### Descendants

    * weac.inverse.Inverse
    * weac.layered.Layered

    ### Methods

    `calc_eigensystem(self)`
    :   Run eigenvalue analysis of the system matrix.

    `calc_foundation_stiffness(self)`
    :   Compute foundation normal and shear stiffness.

    `calc_laminate_stiffness_matrix(self)`
    :   Provide ABD matrix.
        
        Return plane-strain laminate stiffness matrix (ABD matrix).

    `calc_system_matrix(self)`
    :   Assemble first-order ODE system matrix.
        
        Using the solution vector z = [u, u', w, w', psi, psi']
        the ODE system is written in the form Az' + Bz = d
        and rearranged to z' = -(A ^ -1)Bz + (A ^ -1)d = Ez + F

    `get_skier_load(self, m, phi)`
    :   Calculate skier point load.
        
        Arguments
        ---------
        m : float
            Skier weight (kg).
        phi : float
            Inclination (degrees).
        
        Returns
        -------
        Fn : float
            Skier load (N) in normal direction.
        Ft : float
            Skier load (N) in tangential direction.

    `get_weight_load(self, phi)`
    :   Calculate line loads.
        
        Arguments
        ---------
        phi : float
            Inclination (degrees).
        
        Returns
        -------
        qn : float
            Line load (N/mm) in normal direction.
        qt : float
            Line load (N/mm) in tangential direction.

    `set_beam_properties(self, layers, C0=6.0, C1=4.6, nu=0.25)`
    :   Set material and properties geometry of beam (slab).
        
        Arguments
        ---------
        layers : list or str
            2D list of layer densities and thicknesses. Columns are
            density (kg/m^3) and thickness (mm). One row corresponds
            to one layer. If entered as str, last split must be
            available in database.
        C0 : float, optional
            Multiplicative constant of Young modulus parametrization
            according to Gerling et al. (2017). Default is 6.0.
        C1 : float, optional
            Exponent of Young modulus parameterization according to
            Gerling et al. (2017). Default is 4.6.
        nu : float, optional
            Possion's ratio. Default is 0.25

    `set_foundation_properties(self, t=30, E=0.25, nu=0.25)`
    :   Set material properties and geometry of foundation (weak layer).
        
        Arguments
        ---------
        t : float, optional
            Weak-layer thickness (mm). Default is 30.
        E : float, optional
            Weak-layer Young modulus (MPa). Default is 0.25.
        nu : float, optional
            Weak-layer Poisson ratio. Default is 0.25.

    `z(self, x, C, l, phi, bed=True)`
    :   Assemble solution vector at positions x.
        
        Arguments
        ---------
        x : float or squence
            Horizontal coordinate (mm). Can be sequence of length N.
        C : ndarray
            Vector of constants (6xN) at positions x.
        l : float
            Segment length (mm).
        phi : float
            Inclination (degrees).
        bed : bool
            Indicates whether segment has foundation (True) or not
            (False). Default is True.
        
        Returns
        -------
        z : ndarray
            Solution vector (6xN) at position x.

    `zh(self, x, l=0, bed=True)`
    :   Compute bedded or free complementary solution at position x.
        
        Arguments
        ---------
        x : float
            Horizontal coordinate (mm).
        l : float, optional
            Segment length (mm). Default is 0.
        bed : bool
            Indicates whether segment has foundation or not. Default
            is True.
        
        Returns
        -------
        zh : ndarray
            Complementary solution matrix (6x6) at position x.

    `zp(self, x, phi, bed=True)`
    :   Compute bedded or free particular integrals at position x.
        
        Arguments
        ---------
        x : float
            Horizontal coordinate (mm).
        phi : float
            Inclination (degrees).
        bed : bool
            Indicates whether segment has foundation (True) or not
            (False). Default is True.
        
        Returns
        -------
        zp : ndarray
            Particular integral vector (6x1) at position x.