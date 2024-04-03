"""Base class for the elastic analysis of layered snow slabs."""
# pylint: disable=invalid-name,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-locals

# Third party imports
import numpy as np

# Project imports
from weac.tools import bergfeld, calc_center_of_gravity, load_dummy_profile


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
        modulus (MPa) and Poisson's ratio. Defaults are 0.25
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
    K0 : float
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

    def __init__(self, system='pst-', touchdown=False):
        """
        Initialize eigensystem with user input.

        Arguments
        ---------
        system : {'pst-', '-pst', 'vpst-', '-vpst', 'skier', 'skiers'}, optional
            Type of system to analyse: PST cut from the right (pst-),
            PST cut form the left (-pst), PST with vertical faces cut
            from the right (vpst-), PST with vertical faces cut from the
            left (-vpst), one skier on infinite slab (skier) or multiple
            skiers on infinite slab (skiers). Default is 'pst-'.
        layers : list, optional
            2D list of layer densities and thicknesses. Columns are
            density (kg/m^3) and thickness (mm). One row corresponds
            to one layer. Default is [[240, 200], ].
        """
        # Assign global attributes
        self.g = 9810           # Gravitaiton (mm/s^2)
        self.lski = 1000        # Effective out-of-plane length of skis (mm)
        self.tol = 1e-3         # Relative Romberg integration tolerance
        self.system = system    # 'pst-', '-pst', 'vpst-', '-vpst', 'skier', 'skiers'

        # Initialize weak-layer attributes that will be filled later
        self.weak = False           # Weak-layer properties dictionary
        self.t = False              # Weak-layer thickness (mm)
        self.kn = False             # Weak-layer compressive stiffness
        self.kt = False             # Weak-layer shear stiffness

        # Initialize slab attributes
        self.p = 0                  # Surface line load (N/mm)
        self.slab = False           # Slab properties dictionary
        self.k = False              # Slab shear correction factor
        self.h = False              # Total slab height (mm)
        self.zs = False             # Z-coordinate of slab center of gravity (mm)
        self.phi = False            # Slab inclination (°)
        self.A11 = False            # Slab extensional stiffness
        self.B11 = False            # Slab bending-extension coupling stiffness
        self.D11 = False            # Slab bending stiffness
        self.kA55 = False           # Slab shear stiffness
        self.K0 = False             # Stiffness determinant

        # Inizialize eigensystem attributes
        self.ewC = False            # Complex eigenvalues
        self.ewR = False            # Real eigenvalues
        self.evC = False            # Complex eigenvectors
        self.evR = False            # Real eigenvectors
        self.sC = False             # Stability shift of complex eigenvalues
        self.sR = False             # Stability shift of real eigenvalues

        # Initialize touchdown attributes
        self.touchdown = touchdown  # Flag whether touchdown is possible
        self.a = False              # Cracklength
        self.tc = False             # Weak-layer collapse height (mm)
        self.ratio = False          # Stiffness ratio of collapsed to uncollapsed weak-layer
        self.betaU = False          # Ratio of slab to bedding stiffness (uncollapsed)
        self.betaC = False          # Ratio of slab to bedding stiffness (collapsed)
        self.mode = False           # Touchdown-mode can be either A, B, C or D
        self.td = False             # Touchdown length

    def set_foundation_properties(
            self,
            t: float = 30.0,
            E: float = 0.25,
            nu: float = 0.25,
            update: bool = False):
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

        # Material properties
        self.weak = {
            'nu': nu,           # Poisson's ratio (-)
            'E': E              # Young's modulus (MPa)
        }

        # Recalculate the fundamental system after properties have changed
        if update:
            self.calc_fundamental_system()

    def set_beam_properties(self, layers, C0=6.5, C1=4.4,
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
        C0 : float, optional
            Multiplicative constant of Young modulus parametrization
            according to Bergfeld et al. (2023). Default is 6.5.
        C1 : float, optional
            Exponent of Young modulus parameterization according to
            Bergfeld et al. (2023). Default is 4.6.
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
            E = bergfeld(layers[:, 0], C0=C0, C1=C1)  # Young's modulus

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

    def calc_foundation_stiffness(self):
        """Compute foundation normal and shear stiffness."""
        # Elastic moduli (MPa) under plane-strain conditions
        G = self.weak['E']/(2*(1 + self.weak['nu']))    # Shear modulus
        E = self.weak['E']/(1 - self.weak['nu']**2)     # Young's modulus

        # Foundation (weak layer) stiffnesses (N/mm^3)
        self.kn = E/self.t                              # Normal stiffness
        self.kt = G/self.t                              # Shear stiffness

    def get_ply_coordinates(self):
        """
        Calculate ply (layer) z-coordinates.

        Returns
        -------
        ndarray
            Ply (layer) z-coordinates (top to bottom) in coordinate system with
            downward pointing z-axis (z-list will be negative to positive).

        """
        # Get list of ply (layer) thicknesses and prepend 0
        t = np.concatenate(([0], self.slab[:, 1]))
        # Calculate and return ply z-coordiantes
        return np.cumsum(t) - self.h/2

    def calc_laminate_stiffness_matrix(self):
        """
        Provide ABD matrix.

        Return plane-strain laminate stiffness matrix (ABD matrix).
        """
        # Get ply coordinates (z-list is top to bottom, negative to positive)
        z = self.get_ply_coordinates()
        # Initialize stiffness components
        A11, B11, D11, kA55 = 0, 0, 0, 0
        # Add layerwise contributions
        for i in range(len(z) - 1):
            E, G, nu = self.slab[i, 2:5]
            A11 = A11 + E/(1 - nu**2)*(z[i+1] - z[i])
            B11 = B11 + 1/2*E/(1 - nu**2)*(z[i+1]**2 - z[i]**2)
            D11 = D11 + 1/3*E/(1 - nu**2)*(z[i+1]**3 - z[i]**3)
            kA55 = kA55 + self.k*G*(z[i+1] - z[i])

        self.A11 = A11
        self.B11 = B11
        self.D11 = D11
        self.kA55 = kA55
        self.K0 = B11**2 - A11*D11

    def calc_system_matrix(self):
        """
        Assemble first-order ODE system matrix K.

        Using the solution vector z = [u, u', w, w', psi, psi']
        the ODE system is written in the form Az' + Bz = d
        and rearranged to z' = -(A ^ -1)Bz + (A ^ -1)d = Kz + q

        Returns
        -------
        ndarray
            System matrix K (6x6).
        """
        kn = self.kn
        kt = self.kt

        # Abbreviations (MIT t/2 im GGW, MIT w' in Kinematik)
        K21 = kt*(-2*self.D11 + self.B11*(self.h + self.t))/(2*self.K0)
        K24 = (2*self.D11*kt*self.t
               - self.B11*kt*self.t*(self.h + self.t)
               + 4*self.B11*self.kA55)/(4*self.K0)
        K25 = (-2*self.D11*self.h*kt
               + self.B11*self.h*kt*(self.h + self.t)
               + 4*self.B11*self.kA55)/(4*self.K0)
        K43 = kn/self.kA55
        K61 = kt*(2*self.B11 - self.A11*(self.h + self.t))/(2*self.K0)
        K64 = (-2*self.B11*kt*self.t
               + self.A11*kt*self.t*(self.h+self.t)
               - 4*self.A11*self.kA55)/(4*self.K0)
        K65 = (2*self.B11*self.h*kt
               - self.A11*self.h*kt*(self.h+self.t)
               - 4*self.A11*self.kA55)/(4*self.K0)

        # System matrix
        K = [[0,    1,    0,    0,    0,    0],
             [K21,  0,    0,  K24,  K25,    0],
             [0,    0,    0,    1,    0,    0],
             [0,    0,  K43,    0,    0,   -1],
             [0,    0,    0,    0,    0,    1],
             [K61,  0,    0,  K64,  K65,    0]]

        return np.array(K)

    def get_load_vector(self, phi):
        """
        Compute sytem load vector q.

        Using the solution vector z = [u, u', w, w', psi, psi']
        the ODE system is written in the form Az' + Bz = d
        and rearranged to z' = -(A ^ -1)Bz + (A ^ -1)d = Kz + q

        Arguments
        ---------
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray
            System load vector q (6x1).
        """
        qn, qt = self.get_weight_load(phi)
        pn, pt = self.get_surface_load(phi)
        return np.array([
            [0],
            [(self.B11*(self.h*pt - 2*qt*self.zs)
              + 2*self.D11*(qt + pt))/(2*self.K0)],
            [0],
            [-(qn + pn)/self.kA55],
            [0],
            [-(self.A11*(self.h*pt - 2*qt*self.zs)
               + 2*self.B11*(qt + pt))/(2*self.K0)]
        ])

    def calc_eigensystem(self):
        """Calculate eigenvalues and eigenvectors of the system matrix."""
        # Calculate eigenvalues (ew) and eigenvectors (ev)
        ew, ev = np.linalg.eig(self.calc_system_matrix())
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
            # Abbreviations
            H14 = 3*self.B11/self.A11*x**2
            H24 = 6*self.B11/self.A11*x
            H54 = -3*x**2 + 6*self.K0/(self.A11*self.kA55)
            # Complementary solution matrix of free segments
            zh = np.array(
                [[0,      0,      0,    H14,      1,      x],
                 [0,      0,      0,    H24,      0,      1],
                 [1,      x,   x**2,   x**3,      0,      0],
                 [0,      1,    2*x, 3*x**2,      0,      0],
                 [0,     -1,   -2*x,    H54,      0,      0],
                 [0,      0,     -2,   -6*x,      0,      0]])

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
        K0 = self.K0

        # Unpack geometric properties
        h = self.h
        t = self.t
        zs = self.zs

        # Assemble particular integral vectors
        if bed:
            zp = np.array([
                [(qt + pt)/kt + h*qt*(h + t - 2*zs)/(4*kA55)
                    + h*pt*(2*h + t)/(4*kA55)],
                [0],
                [(qn + pn)/kn],
                [0],
                [-(qt*(h + t - 2*zs) + pt*(2*h + t))/(2*kA55)],
                [0]])
        else:
            zp = np.array([
                [(-3*(qt + pt)/A11 - B11*(qn + pn)*x/K0)/6*x**2],
                [(-2*(qt + pt)/A11 - B11*(qn + pn)*x/K0)/2*x],
                [-A11*(qn + pn)*x**4/(24*K0)],
                [-A11*(qn + pn)*x**3/(6*K0)],
                [A11*(qn + pn)*x**3/(6*K0)
                 + ((zs - B11/A11)*qt - h*pt/2 - (qn + pn)*x)/kA55],
                [(qn + pn)*(A11*x**2/(2*K0) - 1/kA55)]])

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
            Solution vector (6xN) at position x.
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            z = np.concatenate([
                np.dot(self.zh(xi, l, bed), C)
                + self.zp(xi, phi, bed) for xi in x], axis=1)
        else:
            z = np.dot(self.zh(x, l, bed), C) + self.zp(x, phi, bed)

        return z
