"""Base class for the elastic analysis of layered snow slabs."""
# pylint: disable=invalid-name,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-locals

# Third party imports
import numpy as np
from numpy.linalg import inv

# Project imports
from weac.tools import gerling, calc_center_of_gravity, load_dummy_profile


class Eigensystem:
    """
    Base class for a layered beam on an elastic foundation.

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
    tc : float
        Weak-layer thickness after collapse (mm).
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
    lC : float
        Cracklength whose maximum deflection equals the
        weak-layer thickness (mm).
    lS : float
        Cracklength when touchdown exerts maximum support
        on the slab (mm). Corresponds to the longest possible
        unbedded length.
    ratio : float
        Increment factor for the weak-layer stiffness from intact
        to collapsed state.
    beta : float
        Describes the stiffnesses of weak-layer and slab.
    """

    def __init__(self, system='pst-'):
        """
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
        """
        # Assign global attributes
        self.g = 9810           # Gravitaiton (mm/s^2)
        self.lski = 1000        # Effective out-of-plane length of skis (mm)
        self.tol = 1e-3         # Relative Romberg integration tolerance
        self.system = system    # 'pst-', '-pst', 'skier', 'skiers'

        # Initialize weak-layer attributes that will be filled later
        self.weak = False       # Weak-layer properties dictionary
        self.t = False          # Weak-layer thickness (mm)
        self.kn = False         # Weak-layer compressive stiffness
        self.kt = False         # Weak-layer shear stiffness
        self.tc = False         # Weak-layer collapse height (mm)

        # Initialize slab attributes
        self.p = 0              # Surface line load (N/mm)
        self.slab = False       # Slab properties dictionary
        self.k = False          # Slab shear correction factor
        self.h = False          # Total slab height (mm)
        self.zs = False         # Z-coordinate of slab center of gravity (mm)
        self.phi = False        # Slab inclination (°)
        self.A = False          # Constant stiffness value
        self.B = False          # Linear stiffness value
        self.C = False          # Quadratic stiffness value
        self.D = False          # Constant shear modulus
        self.A11 = False        # Slab extensional stiffness
        self.B11 = False        # Slab bending-extension coupling stiffness
        self.D11 = False        # Slab bending stiffness
        self.kA55 = False       # Slab shear stiffness
        self.E0 = False         # Stiffness determinant
        self.m = False          # Integral over densities
        self.mz = False         # First static moment of densities
        self.zz = False

        # Inizialize eigensystem attributes
        self.ewC = False        # Complex eigenvalues
        self.ewR = False        # Real eigenvalues
        self.evC = False        # Complex eigenvectors
        self.evR = False        # Real eigenvectors
        self.sC = False         # Stability shift of complex eigenvalues
        self.sR = False         # Stability shift of real eigenvalues
        self.sysmat = False     # System matrix
        self.sysMatA = False
        self.sysMatB = False
        # Initialize touchdown attributes
        self.lC = False         # Minimum length of substratum contact (mm)
        self.lS = False         # Maximum length of span between
                                # between bedded and touchdowned boundary (mm)
        self.ratio = False      # Stiffness ratio of collalpsed to uncollapsed weak-layer
        self.beta = False       # Ratio of slab to bedding stiffness

    def set_foundation_properties(self, t=30, cf=0.5, E=0.25, nu=0.25, rhoweak = 100, update=False):
        """
        Set material properties and geometry of foundation (weak layer).

        Arguments
        ---------
        t : float, optional
            Weak-layer thickness (mm). Default is 30.
        cf : float
            Fraction by which the weak-layer thickness is reduced
            due to collapse. Default is 0.5.
        E : float, optional
            Weak-layer Young modulus (MPa). Default is 0.25.
        nu : float, optional
            Weak-layer Poisson ratio. Default is 0.25.
        update : bool, optional
            If true, recalculate the fundamental system after
            foundation properties have changed.
        """
        # Geometry
        self.t = t              # Weak-layer thickness (mm)
        self.tc = cf*self.t     # Weak-layer collapse height (mm)

        # Material properties
        self.weak = {
            'nu': nu,            # Poisson's ratio (-)
            'E': E,              # Young's modulus (MPa)
            'rho': rhoweak*1e-12 # Density (t/mm^3)       
        }

        # Recalculate the fundamental system after properties have changed
        if update:
            self.calc_fundamental_system()

    def set_beam_properties(self, layers, phi, C0=6.0, C1=4.60,
                            nu=0.25, update=False):
        """
        Set material and properties geometry of beam (slab).

        Arguments
        ---------
        layers : list or str
            2D list of top-to-bottom layer densities and thicknesses.
            Columns are density (kg/m^3) and thickness (mm). One row
            corresponds to one layer. If entered as str, last split
            must be available in database.
        phi : float
            Inclination of the slab (degrees).
        C0 : float, optional
            Multiplicative constant of Young modulus parametrization
            according to Gerling et al. (2017). Default is 6.0.
        C1 : float, optional
            Exponent of Young modulus parameterization according to
            Gerling et al. (2017). Default is 4.6.
        nu : float, optional
            Possion's ratio. Default is 0.25
        update : bool, optional
            If true, recalculate the fundamental system after
            foundation properties have changed.
        """
        if isinstance(layers, str):
            # Read layering and Young's modulus from database
            layers, E = load_dummy_profile(layers.split()[-1])
        else:
            # Compute Young's modulus from density parametrization
            layers = np.array(layers)
            E = gerling(layers[:, 0], C0=C0, C1=C1)  # Young's modulus

        # Derive other elastic properties
        nu = nu*np.ones(layers.shape[0])         # Global poisson's ratio
        G = E/(2*(1 + nu))                       # Shear modulus
        self.k = 5/6                             # Shear correction factor

        # Compute total slab thickness and center of gravity
        self.h, self.zs = calc_center_of_gravity(layers)

        # Assemble layering into matrix (top to bottom)
        # Columns are density (kg/m^3), layer thickness (mm)
        # Young's modulus (MPa), shear modulus (MPa), and
        # Poisson's ratio
        self.slab = np.vstack([layers.T, E, G, nu]).T

        # Set beam inclination
        self.phi = phi

        # Recalculate the fundamental system after properties have changed
        if update:
            self.calc_fundamental_system()

    def set_surface_load(self, p):
        """
        Set surface line load.

        Define a distributed surface load (N/mm) that acts
        in vertical (gravity) direction on the top surface
        of the slab.

        Arguments
        ---------
        p : float
            Surface line load (N/mm) that acts in vertical
            (gravity) direction onm the top surface of the
            slab.
        """
        self.p = p

    def calc_foundation_stiffness(self, ratio=16):
        """Compute foundation normal and shear stiffness."""
        # Elastic moduli (MPa) under plane-strain conditions
        G = self.weak['E']/(2*(1 + self.weak['nu']))    # Shear modulus
        E = self.weak['E']/(1 - self.weak['nu']**2)     # Young's modulus

        # Foundation (weak layer) stiffnesses (N/mm^3)
        self.kn = E/self.t                              # Normal stiffness
        self.kt = G/self.t                              # Shear stiffness

        # Weak-layer stiffness increment factor for collapse
        self.ratio = ratio


    def calc_laminate_stiffness_matrix(self):
        """
        Provide ABD matrix.

        Return plane-strain laminate stiffness matrix (ABD matrix).
        """
        # Number of plies and ply thicknesses (top to bottom)
        n = self.slab.shape[0]
        t = self.slab[:, 1]
        # Calculate ply coordinates (top to bottom) in coordinate system
        # with downward pointing z-axis (z-list will be negative to positive)
        z = np.zeros(n + 1)
        for j in range(n + 1):
            z[j] = -self.h/2 + sum(t[0:j])
        # Initialize stiffness components
        A11, B11, D11, kA55 = 0, 0, 0, 0
        # Add layerwise contributions
        for i in range(n):
            E, G, nu = self.slab[i, 2:5]
            A11 = A11 + E/(1 - nu**2)*(z[i+1] - z[i])
            B11 = B11 + 1/2*E/(1 - nu**2)*(z[i+1]**2 - z[i]**2)
            D11 = D11 + 1/3*E/(1 - nu**2)*(z[i+1]**3 - z[i]**3)
            kA55 = kA55 + self.k*G*(z[i+1] - z[i])

        self.A11 = A11
        self.B11 = B11
        self.D11 = D11
        self.kA55 = kA55
        self.E0 = B11**2 - A11*D11

    def calc_stiffness_properties(self):
        """
        Provide layering dependent variables.

        Return plane-strain laminate stiffness matrix (ABD matrix).
        """
        # Number of plies and ply thicknesses (top to bottom)
        n = self.slab.shape[0]
        t = self.slab[:, 1]
        # Calculate ply coordinates (top to bottom) in coordinate system
        # with downward pointing z-axis (z-list will be negative to positive)
        z = np.zeros(n + 1)
        for j in range(n + 1):
            z[j] = -self.h - self.t/2 + sum(t[0:j])

        # Initialize stiffness components
        A, B, C, D = 0, 0, 0, 0
        # Inititalize densities
        m , mz = 0 , 0
        # Add layerwise contributions
        for i in range(n):
            E, G, nu = self.slab[i, 2:5]
            A = A + E/((1-2*nu)*(1+nu))*(1-nu)*(z[i+1] - z[i])
            B = B + 1/2*E/((1-2*nu)*(1+nu))*(1-nu)*(z[i+1]**2 - z[i]**2)
            C = C + 1/3*E/((1-2*nu)*(1+nu))*(1-nu)*(z[i+1]**3 - z[i]**3)
            D = D + G*(z[i+1] - z[i])
            m = m + self.slab[i,0]*1e-12*(z[i+1]-z[i])
            mz = mz + 1/2*self.slab[i,0]*1e-12*(z[i+1]**2 - z[i]**2)
        self.zz = z
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.m = m
        self.mz = mz 


    def calc_system_matrix(self):
        """
        Assemble first-order ODE system matrix.

        Using the solution vector z = [u, u', w, w', psi, psi',phiU, phiU', phiW, phiW']
        the ODE system is written in the form Az' + Bz = d
        and rearranged to z' = -(A ^ -1) B z + (A ^ -1) d = E z + F
        """
        Ew = self.weak['E']
        nuw = self.weak['nu']

        t = self.t
        h = self.h
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        Pi = np.pi


        # a22 = -A - (Ew * t * (1. - nuw)) / (3. * (1. + nuw) * (1. - 2. * nuw))
        # a26 = -B - 0.5 * A *(h + t)
        # a28 = -(t * Ew * (1 - nuw)) / (Pi * (1. + nuw) * (1. - 2. * nuw))

        # a44 = - D - t * Ew / (6. * (1. + nuw))
        # a4X = - t * Ew / (2. * Pi * (1. + nuw))

        # a62 = - B - 0.5 * A * (h + t)
        # a66 = - C -0.25 * (h + t) * (4. * B + A * (h + t))

        # a82 = - t * Ew * (1. - nuw) / (Pi * (1. + nuw) * (1. - 2. * nuw))
        # a88 = - t * Ew * (1. - nuw) / (2. * (1. + nuw) * (1. - 2. * nuw))

        # aX4 = - t * Ew / (2. * Pi * (1. + nuw))
        # aXX = - t * Ew / (4. * (1. + nuw))

        # SystemMatrixA = [[1,   0, 0,   0, 0,   0, 0,   0, 0,   0],
        #                  [0, a22, 0,   0, 0, a26, 0, a28, 0,   0],
        #                  [0,   0, 1,   0, 0,   0, 0,   0, 0,   0],
        #                  [0,   0, 0, a44, 0,   0, 0,   0, 0, a4X],
        #                  [0,   0, 0,   0, 1,   0, 0,   0, 0,   0],
        #                  [0, a62, 0,   0, 0, a66, 0,   0, 0,   0],
        #                  [0,   0, 0,   0, 0,   0, 1,   0, 0,   0],
        #                  [0, a82, 0,   0, 0,   0, 0, a88, 0,   0],
        #                  [0,   0, 0,   0, 0,   0, 0,   0, 1,   0],
        #                  [0,   0, 0, aX4, 0,   0, 0,   0, 0, aXX]]
        # self.sysMatA = SystemMatrixA
        # b21 = Ew / (t * 2. * (1. + nuw))
        # b24 = - Ew * (1. - 4. * nuw) / (4. * (1. - 2. * nuw) * (1. - nuw))
        # b2X = - Ew / (Pi * (1. - 2. * nuw) * (1. + nuw))

        # b42 = Ew * (1. - 4. * nuw) / (4. * (1. - 2. * nuw) * (1. + nuw))
        # b43 = Ew * (1. - nuw) / (t * (1. - 2. * nuw) * (1. + nuw))
        # b46 = - D
        # b48 = - Ew / (Pi * (1. - 2. * nuw) * (1. + nuw))
        
        # b64 = D
        # b65 = D

        # b84 = Ew / (Pi * (1. - 2. * nuw) * (1. + nuw))
        # b87 = Ew * Pi**2 / (4. * t * (1. + nuw))

        # bX2 = Ew / (Pi * (1. - 2. * nuw) * (1. + nuw))
        # bX9 = Ew * Pi**2 * (1. - nuw) / (2. * t * (1. + nuw) * (1. - 2. * nuw))


        # SystemMatrixB = [[   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0],
        #                  [ b21,   0, b24,   0,   0,   0,   0,   0,   0, b2X],
        #                  [   0,   0,   0,  -1,   0,   0,   0,   0,   0,   0],
        #                  [   0, b42, b43,   0,   0, b46,   0, b48,   0,   0],
        #                  [   0,   0,   0,   0,   0,  -1,   0,   0,   0,   0],
        #                  [   0,   0,   0, b64, b65,   0,   0,   0,   0,   0],
        #                  [   0,   0,   0,   0,   0,   0,   0,  -1,   0,   0],
        #                  [   0,   0,   0, b84,   0,   0, b87,   0,   0,   0],
        #                  [   0,   0,   0,   0,   0,   0,   0,   0,   0,  -1],
        #                  [   0, bX2,   0,   0,   0,   0,   0,   0, bX9,   0]]
        
        
        # self.sysMatB = SystemMatrixB
        # Assemble (10x10) system matrix in accordance to Mathemtica script.
        c21 = 3. * Pi**2 * (4. * C + (  t)*( A * (  t) + 4. * B ) )* Ew * (-1. + 2 * nuw) \
                / (2. * t * ((Pi**2 - 6.) * t *(4. * C + (  t)*( A * (  t) + 4. * B ) ) * Ew * (-1. + nuw) - 12. * (B**2 - A * C) * Pi**2 * (-1. + nuw + 2. * nuw**2)))
        c24 = - 3. * ((4 * C + (  t) * (4 * B + A * (  t))) * Ew * (-8. - Pi**2 + 4. * Pi**2 * nuw) + 8. * D * Pi**2 * (2. * B + A* (  t) ) * (-1. + nuw + 2. * nuw**2)) \
                / (4. * (-6. + Pi**2) * t * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * (-1. + nuw) - 48. * (B**2 - A * C) * Pi**2 * (-1. + nuw + 2. * nuw**2))
        
        
        c25 = 6. * D * Pi**2 * (2. * B + A * (  t)) * (1 + nuw) * (-1. + 2. * nuw) \
                / (-((-6. + Pi**2) * t * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * (-1. +  nuw)) + 12. * (B**2 - A * C) * Pi**2 * (-1. + nuw + 2. * nuw**2))
        
        
        
        c27 = - 3. * Pi**3 * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * (-1. + 2. * nuw) \
                / (2. * t * ((-6. + Pi**2) * t * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * (-1. + nuw) - 12. * (B**2 - A * C) * Pi**2 * (-1. + nuw +  2. * nuw**2)))
        c2X =  3. * Pi * (4. * C + (  t) * (4. * B + A * (  t))) * Ew \
                / ((-6. + Pi**2) * t * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * (-1. + nuw) - 12. * (B**2 - A * C) * Pi**2 * (-1. + nuw + 2. * nuw**2))

        c42 = 3. * Ew * (8. - Pi**2 + 4. * Pi**2 * nuw) / (2. * (-1. + 2. * nuw) * ((-6. + Pi**2) * t * Ew + 6. * D * Pi**2 * (1. + nuw)))
        c43 = 6. * Pi**2  * (-1 + nuw) * Ew / (t * (-1. + 2. * nuw) * ((-6 + Pi**2) * t * Ew + 6. * D * Pi**2 * (1. + nuw)))

        c46 = - 6. * D * Pi**2 * (1. + nuw) / ((-6. + Pi**2) * t * Ew + 6. * D * Pi**2 * (1. + nuw))
        
        c48 = 6. * Pi * Ew / ((-1. + 2. * nuw ) * ((-6 + Pi**2) * t * Ew + 6. * D * Pi**2 * (1. + nuw)))

        c49 = - 6. * Pi**3 * (-1. + nuw) * Ew / (t * (-1. + 2. * nuw ) * ((-6 + Pi**2) * t * Ew + 6. * D * Pi**2 * (1. + nuw)))

        c61 = - 3. * Pi**2 * (2 * B + A * (  t)) * Ew * (-1. + 2. * nuw) \
                / (t * ((-6 + Pi**2) * t * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * (-1. + nuw) - 12. * (B**2 - A * C) * Pi**2 * (-1. + nuw + 2. * nuw**2)))

                
        c64 = (Ew * ( (-6) *B* ( 8 + Pi**2) - 8 * D * ( -6 + Pi**2) * t - 3 * A * ( 8 + Pi**2) * t + 4 * ( 6 * B * Pi**2 + 3 * A * Pi**2 * t + 2 * D * (-6 + Pi**2) * t) * nuw) + 24 * A * D * Pi**2 * (-1 + nuw + 2 * nuw**2)) \
                / (2 * (-6 + Pi**2) * t * ( 4*C + t*(4 * B + A * t)) * Ew * (-1 + nuw) - 24 * ( B**2 - A * C) * Pi**2 * (-1 + nuw + 2*nuw**2))

        c65 = (4. * D * ((-6. + Pi**2) * t * Ew * (-1 + nuw) + 3. * A * Pi**2 * (-1. + nuw + 2. * nuw**2))) \
                / ((-6. + Pi**2) * t * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * (-1. + nuw) - 12. * (B**2 - A * C) * Pi**2 * (-1. + nuw + 2. * nuw**2)) 



        c67 = 3. * Pi**3 * (2. * B + A * (  t) )* Ew * (-1 + 2. * nuw) \
                / (t * ((-6. + Pi**2) * t * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * ( -1. + nuw) - 12. * (B**2 - A * C) * Pi**2 * ( -1. + nuw + 2. * nuw**2)))
        


        
        c6X = 6. * Pi * (2. * B + A * (  t)) * Ew \
                / (-((-6. + Pi**2) * t * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * (-1. + nuw)) + 12. * (B**2 - A * C) * Pi**2 * (-1. + nuw + 2. * nuw**2))

        c81 = - 3. * Pi * (4. * C +  (  t) * (4. * B + A * (  t))) * Ew * (-1. + 2. * nuw) \
                / (t * ((-6. + Pi**2) * t * ( 4. * C + (  t) * (4. * B + A * (  t))) * Ew * (-1. + nuw) - 12. * (B**2 - A * C) * Pi**2 * (-1 + nuw + 2. * nuw**2)))

        c84 = (Pi * (24. * (2. * B**2 - 2. * B * D * t - A * (2. * C + D * t * (  t)) + D * t * (2. * B + A * (  t)) * nuw)* (-1. + nuw + 2. * nuw**2) + \
                t*(4. * C + (  t) * (4. * B + A * (  t)))* Ew * ( 7. - 19. * nuw + 12. * nuw**2) )) \
                / (2. * t * (-1. + nuw) * ((-6. + Pi**2) * t * (4. * C + (  t) * ( 4. * B + A * (  t))) * Ew * (-1. + nuw) - 12.* ( B**2 - A * C) * Pi**2 * (-1. + nuw +2. * nuw**2)))
        c85 = -12. * D * Pi * ( 2. * B + A * (  t)) * (1. + nuw) * (-1. + 2. * nuw) \
                / (-((-6. + Pi**2) * t * (4. * C + (  t) * ( 4. * B + A * (  t))) * Ew * (-1. + nuw) ) + 12. * ( B**2 - A * C) * Pi**2 * (-1. + nuw + 2. * nuw**2))
        
        
        c87 = Pi**4 * (-1. + 2. * nuw) * (t * ( 4. * C + (  t) * (4. * B + A * (  t))) * Ew * (-1. + nuw) - 12 * (B**2 - A * C) * (-1. + nuw + 2. * nuw**2)) \
                / (2. * t**2  * (-1 + nuw) * ((-6. + Pi**2) * t * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * ( -1. + nuw) - 12. * (B**2 - A * C) * Pi**2 * ( -1. + nuw + 2. * nuw**2)))
        
        
        
        c8X =- 6. * (4. * C + (  t) * (4. * B + A * (  t))) * Ew \
                / ((-6. + Pi**2) * t * (4. * C + (  t) * (4. * B + A * (  t))) * Ew * ( -1. + nuw) - 12. * (B**2 - A * C) * Pi**2 * ( -1. + nuw + 2. * nuw**2))

        cX2 = - Pi * (24. * D * (1. + nuw) + Ew * (t + 12. * t * nuw))\
                / (t * (-1. + 2. * nuw) * ((-6 + Pi**2) * t* Ew + 6. * D * Pi**2 * (1 + nuw)))
        cX3 = - 12. * Pi * Ew * (-1 + nuw) / (t * (-1. + 2. * nuw) * ((-6 + Pi**2) * t* Ew + 6. * D * Pi**2 * (1 + nuw)))
        cX6 = 12. * D * Pi * (1. + nuw) / ((-6 + Pi**2)*t * Ew + 6. * D * Pi**2 * (1. + nuw))
        cX8 = -12. * Ew \
                / ((-1. + 2. * nuw) * ((-6 + Pi**2) * t* Ew + 6. * D * Pi**2 * (1 + nuw)))
        cX9 = 2. * Pi**4 *(-1. + nuw) * (t*Ew + 6. * D * (1+nuw)) \
                / (t**2 * (-1. + 2. * nuw) * ((-6 + Pi**2) * t* Ew + 6. * D * Pi**2 * (1 + nuw)))

        SystemMatrixC = [[  0,   1,   0,   0,   0,   0,   0,   0,   0,   0],
                         [c21,   0,   0, c24, c25,   0, c27,   0,   0, c2X],
                         [  0,   0,   0,   1,   0,   0,   0,   0,   0,   0],
                         [  0, c42, c43,   0,   0, c46,   0, c48, c49,   0],
                         [  0,   0,   0,   0,   0,   1,   0,   0,   0,   0],
                         [c61,   0,   0, c64, c65,   0, c67,   0,   0, c6X],
                         [  0,   0,   0,   0,   0,   0,   0,   1,   0,   0],
                         [c81,   0,   0, c84, c85,   0, c87,   0,   0, c8X],
                         [  0,   0,   0,   0,   0,   0,   0,   0,   0,   1],
                         [  0, cX2, cX3,   0,   0, cX6,   0, cX8, cX9,   0]]
        


        self.sysmat = np.array(SystemMatrixC)

    def calc_eigensystem(self):
        """Calculate eigenvalues and eigenvectors of the system matrix."""
        # Calculate eigenvalues (ew) and eigenvectors (ev)
        ew, ev = np.linalg.eig(self.sysmat)
        # Classify real and complex eigenvalues
        real = (ew.imag == 0) & (ew.real != 0)  # real eigenvalues
        cmplx = ew.imag > 0                   # positive complex conjugates
        # Eigenvalues
        self.ewC = ew[cmplx]
        self.ewR = ew[real].real
        # Eigenvectors
        self.evC = ev[:, cmplx]
        self.evR = ev[:, real].real
        # Prepare positive eigenvalue shifts for numerical robustness
        self.sR, self.sC = np.zeros(self.ewR.shape), np.zeros(self.ewC.shape)
        self.sR[self.ewR > 0], self.sC[self.ewC > 0] = -1, -1

    def calc_fundamental_system(self):
        """Calculate the fundamental system of the problem."""
        self.calc_foundation_stiffness()
        self.calc_laminate_stiffness_matrix()
        self.calc_stiffness_properties()
        self.calc_system_matrix()
        self.calc_eigensystem()

    def get_weight_load(self, phi):
        """
        Calculate line loads from slab mass.

        Arguments
        ---------
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        qn : float
            Line load (N/mm) at center of gravity in normal direction.
        qt : float
            Line load (N/mm) at center of gravity in tangential direction.
        """
        # Convert units
        phi = np.deg2rad(phi)                   # Convert inclination to rad
        rho = self.slab[:, 0]*1e-12             # Convert density to t/mm^3
        # Sum up layer weight loads
        q = sum(rho*self.g*self.slab[:, 1])     # Line load (N/mm)
        # Split into components
        qn = q*np.cos(phi)                      # Normal direction
        qt = -q*np.sin(phi)                     # Tangential direction

        return qn, qt

    def get_surface_load(self, phi):
        """
        Calculate surface line loads.

        Arguments
        ---------
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        pn : float
            Surface line load (N/mm) in normal direction.
        pt : float
            Surface line load (N/mm) in tangential direction.
        """
        # Convert units
        phi = np.deg2rad(phi)                   # Convert inclination to rad
        # Split into components
        pn = self.p*np.cos(phi)                 # Normal direction
        pt = -self.p*np.sin(phi)                # Tangential direction

        return pn, pt

    def get_skier_load(self, m, phi):
        """
        Calculate skier point load.

        Arguments
        ---------
        m : float
            Skier weight (kg).
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        Fn : float
            Skier load (N) in normal direction.
        Ft : float
            Skier load (N) in tangential direction.
        """
        phi = np.deg2rad(phi)                   # Convert inclination to rad
        F = 1e-3*np.array(m)*self.g/self.lski   # Total skier load (N)
        Fn = F*np.cos(phi)                      # Normal skier load (N)
        Ft = -F*np.sin(phi)                     # Tangential skier load (N)

        return Fn, Ft

    def calc_beta(self):
        """
        Calculate beta.

        Returns
        -------
        beta : float
            Weak-layer to slab stiffness relation factor.
        """
        # (Intact) weak-layer to slab stiffness relation factor
        self.beta = (self.kn/(4*self.D11))**(1/4)

    def calc_span_length(self):
        """
        Calculate span from layer and weak layer properties and load situation.

        Returns
        -------
        lS : float
            Span of the element between bedded element and touchdown for full touchdown.
        """
        def polynomial():
            """
            Calculate the coefficients of a sixth order polynomial equation.

            Returns
            -------
            list
                First coefficient for sixth order term,
                second coefficient for fith order term and so on.
            """
            a1 = self.kA55**2*kR1*kN1*q0
            a2 = 6*self.kA55*(self.D11*self.kA55 + kR1*kR2)*kN1*q0
            a3 = 30*self.D11*self.kA55*(kR1 + kR2)*kN1*q0
            a4 = 24*self.D11*(2*self.kA55**2*kR1 + 3*self.D11*self.kA55*kN1 + 3*kR1*kR2*kN1)*q0
            a5 = 72*self.D11*(self.D11*(self.kA55**2 + (kR1 + kR2)*kN1)*q0 \
                + self.kA55*kR1*(2*kR2*q0 - self.kA55*kN1*self.tc))
            a6 = 144*self.D11*self.kA55*(self.D11*(kR1 + kR2)*q0 \
                - (self.D11*self.kA55 + kR1*kR2)*kN1*self.tc)
            a7 = - 144*self.D11**2*self.kA55*(kR1 + kR2)*kN1*self.tc
            return [a1,a2,a3,a4,a5,a6,a7]

        # Get spring stiffnesses for adjacent segment with intact weak-layer
        kR1 = self.calc_rot_spring(collapse=False)
        kN1 = self.calc_trans_spring()
        # Get spring stiffnesses for adjacent segment with collapsed weak-layer
        kR2 = self.calc_rot_spring(collapse=True)
        # Get surface normal load components
        qn = self.get_weight_load(self.phi)[0]
        pn = self.get_surface_load(self.phi)[0]
        q0 = qn + pn
        # Calculate positive real roots
        pos = (np.roots(polynomial()).imag == 0) & (np.roots(polynomial()).real > 0)
        self.lS = np.roots(polynomial())[pos].real[0]

    def calc_contact_length(self):
        """
        Calculate segment length where max slab deflection equals tc.

        Returns
        -------
        lC : float
            Maximum length without substratum contact.
        """
        def polynomial():
            """
            Calculate the coefficients of a fourth order polynomial equation.

            Returns
            -------
            list
                First coefficient for fourth order term,
                second coefficient for third order term and so on.
            """
            a1 = 1/(8*self.D11)*q0
            a2 = 1/(2*kR1)*q0
            a3 = 1/(2*self.kA55)*q0
            a4 = 1/kN1*q0
            a5 = -self.tc
            return [a1,a2,a3,a4,a5]

        # Get spring stiffnesses for adjacent segment intact intact weak-layer
        kR1 = self.calc_rot_spring(collapse=False)
        kN1 = self.calc_trans_spring()
        # Get surface normal load components
        qn = self.get_weight_load(self.phi)[0]
        pn = self.get_surface_load(self.phi)[0]
        q0 = qn + pn
        # Calculate positive real roots
        pos = (np.roots(polynomial()).imag == 0) & (np.roots(polynomial()).real > 0)
        self.lC = np.roots(polynomial())[pos].real[0]

    def calc_rot_spring(self, collapse=True):
        """
        Calculate rotational spring stiffness from layer properties.

        Arguments
        ---------
        collapse : boolean
            Indicates whether weak-layer is collapsed.

        Returns
        -------
        kR : float
            Rotational spring stiffness (Nmm/mm/rad).
        """
        # get ratio for foundation stiffness after collapse
        if collapse:
            ratio = self.ratio
        else:
            ratio = 1
        # calc spring stiffness
        kR = self.D11*self.beta*ratio**(1/4)

        return kR

    def calc_trans_spring(self):
        """
        Calculate translational spring stiffness from layer properties.

        Returns
        -------
        kN : float
            Translational spring stiffness (N/mm^2).
        """
        # calc translational spring stiffness for bedded euler-bernoulli-beam
        kN = 2*self.D11*self.beta**3

        return kN

    def calc_touchdown_system(self):
        """Calculate the lenghts for touchdown evaluation"""
        self.calc_beta()
        self.calc_span_length()
        self.calc_contact_length()

    def zh(self, x, l=0, bed=True):
        """
        Compute bedded or free complementary solution at position x.

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
        """
        if bed:
            zh = np.concatenate([
                # Real
                self.evR*np.exp(self.ewR*(x + l*self.sR)),
                # Complex
                np.exp(self.ewC.real*(x + l*self.sC))*(
                       self.evC.real*np.cos(self.ewC.imag*x)
                       - self.evC.imag*np.sin(self.ewC.imag*x)),
                # Complex
                np.exp(self.ewC.real*(x + l*self.sC))*(
                       self.evC.imag*np.cos(self.ewC.imag*x)
                       + self.evC.real*np.sin(self.ewC.imag*x))], axis=1)
        else:
            # Abbreviations for the unbedded segemnts in accordance with Mathematica script
            H12 = 0.5 * self.kA55 *self.B11/self.E0*x**2
            H13 = 0.5 * self.kA55 *self.B11/self.E0*x**2


            H22 = self.kA55 * self.B11 / self.E0 * x
            H23 = self.kA55 * self.B11 / self.E0 * x


            H32 = x + 1 / 6 * self.A11 * self.kA55 / self.E0 *x**3
            H33 =     1 / 6 * self.A11 * self.kA55 / self.E0 *x**3
            H34 = -0.5 * x**2
            
            
            H42 = 1 + 1 / 2 * self.A11 * self.kA55 / self.E0 * x**2
            H43 = 1 / 2 * self.A11 * self.kA55 / self.E0 * x**2
            H44 = - x


            H52 = -1/2 * self.A11 * self.kA55 / self.E0 * x**2
            H53 = 1 - 1/2 * self.A11 * self.kA55 / self.E0 * x**2
            H54 = x


            H62 = -self.A11 * self.kA55 / self.E0 * x
            H63 = -self.A11 * self.kA55 / self.E0 * x
            H64 = 1


            # Complementary solution matrix of free segments
            zh = np.array(
                [[0,    H12,    H13,      0,      1,      x],
                 [0,    H22,    H23,      0,      0,      1],
                 [1,    H32,    H33,    H34,      0,      0],
                 [0,    H42,    H43,    H44,      0,      0],
                 [0,    H52,    H53,    H54,      0,      0],
                 [0,    H62,    H63,    H64,      0,      0]])

        return zh

    def zp(self, x, phi, bed=True):
        """
        Compute bedded or free particular integrals at position x.

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
        """
        # Get weight and surface loads
        qn, qt = self.get_weight_load(phi)
        pn, pt = self.get_surface_load(phi)

        # Set foundation stiffnesses
        kn = self.kn
        kt = self.kt

        # Unpack laminate stiffnesses
        A11 = self.A11
        B11 = self.B11
        kA55 = self.kA55
        D11 = self.D11
        E0 = self.E0
        D = self.D

        # Unpack weak layer properties
        Ew = self.weak['E']
        nuw = self.weak['nu']
        rhoweak = self.weak['rho']

        # Unpack layering  information
        m = self.m
        mz = self.mz

        # Unpack general variables
        g = self.g
        Pi = np.pi


        # Unpack geometric properties
        h = self.h
        t = self.t
        zs = self.zs

        # Assemble particular integral vectors in accordance with Mathematica scripts
        if bed:
            zp = np.array([
                [- t * (2 * pt + (2 * g * m + g * t * rhoweak) * np.sin(np.deg2rad(phi))) * (1 + nuw) / (Ew) ],
                [0],
                [t * (2 * (-pn  + g * m  * np.cos(np.deg2rad(phi)))  + g * t *np.cos(np.deg2rad(phi)) * rhoweak )* ( 1+nuw) * (-1 + 2 * nuw) / (2 * Ew * (-1 + nuw))],
                [0],
                [(2 * h * pt - g * (2 * mz + m *  t) *np.sin(np.deg2rad(phi))) / (2 * D)],
                [0],
                [-(8 * g * t**2 * np.sin(np.deg2rad(phi)) * rhoweak * ( 1 + nuw)) / (Pi**3 * Ew)],
                [0],
                [(4 * g * t**2 * np.cos(np.deg2rad(phi)) * rhoweak * (1 + nuw) * (-1 + 2 * nuw)) / (Pi**3 * Ew * (-1 + nuw))],
                [0]])
        else:
            zp = np.array([
                [D11/(2*E0) * (pt + qt) * x**2 + B11/(2*E0) * (h/2 * pt - zs * qt) * x**2 - B11/(6*E0) * (pn + qn) * x**3],
                [D11/(  E0) * (pt + qt) * x    + B11/(  E0) * (h/2 * pt - zs * qt) * x    - B11/(2*E0) * (pn + qn) * x**2],
                [-(pn+qn)/(2*kA55) * x**2 + B11/(6*E0) * (pt + qt) * x**3 + A11/(6 * E0)*(h/2 * pt - zs * qt) * x**3 - A11/(24*E0)*(pn+qn)*x**4],
                [-(pn+qn)/(  kA55) * x    + B11/(2*E0) * (pt + qt) * x**2 + A11/(2 * E0)*(h/2 * pt - zs * qt) * x**2 - A11/( 6*E0)*(pn+qn)*x**3],
                [- B11/(2*E0) * (pt+qt) * x**2 - A11/(2*E0) * (h/2 * pt - zs * qt) * x**2 + A11/(6*E0) * (pn + qn) * x**3],
                [- B11/(  E0) * (pt+qt) * x    - A11/(  E0) * (h/2 * pt - zs * qt) * x    + A11/(2*E0) * (pn + qn) * x**2]])

        return zp

    def z(self, x, C, l, phi, bed=True):
        """
        Assemble solution vector at positions x.

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
            Solution vector (6xN) or (10xN) at position x.
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            z = np.concatenate([
                np.dot(self.zh(xi, l, bed), C)
                + self.zp(xi, phi, bed) for xi in x], axis=1)
        else:
            z = np.dot(self.zh(x, l, bed), C) + self.zp(x, phi, bed)

        return z
