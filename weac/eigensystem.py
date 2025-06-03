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

    This class provides a comprehensive framework for analyzing the mechanical behavior
    of layered snow slabs on an elastic foundation. It includes methods for setting up
    geometry, material properties, and loading conditions, as well as calculating the
    fundamental system and eigenvalues.

    Attributes
    ----------
    g : float
        Gravitational constant (mm/s^2). Default is 9810.
    lski : float
        Effective out-of-plane length of skis (mm). Default is 1000.
    tol : float
        Relative Romberg integration tolerance. Default is 1e-3.
    system : str
        Type of boundary value problem. Default is 'pst-'.
    weak : dict
        Dictionary containing weak layer properties:
        - E: Young's modulus (MPa)
        - nu: Poisson's ratio
        - rho: Density (t/mm^3)
    t : float
        Weak-layer thickness (mm). Default is 30.
    kn : float
        Compressive foundation (weak-layer) stiffness (N/mm^3).
    kt : float
        Shear foundation (weak-layer) stiffness (N/mm^3).
    tc : float
        Weak-layer thickness after collapse (mm).
    slab : ndarray
        Matrix containing elastic properties of all slab layers.
        Columns are:
        - density (kg/m^3)
        - layer height (mm)
        - Young's modulus (MPa)
        - shear modulus (MPa)
        - Poisson's ratio
    k : float
        Shear correction factor of the slab. Default is 1.
    h : float
        Slab thickness (mm). Default is 300.
    zs : float
        Z-coordinate of the center of gravity of the slab (mm).
    zA : float
        Z-coordinate of the center of gravity of surface loads (mm).
    yA : float
        Y-coordinate of the center of gravity of surface loads (mm).
    A11 : float
        Extensional stiffness of the slab (N/mm).
    B11 : float
        Bending-extension coupling stiffness of the slab (N).
    D11 : float
        Bending stiffness of the slab (Nmm).
    kA55 : float
        Shear stiffness of the slab (N/mm).
    kB55 : float
        Shear coupling stiffness of the slab (N).
    kD55 : float
        Bending shear stiffness of the slab (Nmm).
    E0 : float
        Characteristic stiffness value (N).
    ewC : ndarray
        List of complex eigenvalues.
    ewR : ndarray
        List of real eigenvalues.
    evC : ndarray
        Matrix with eigenvectors corresponding to complex eigenvalues as columns.
    evR : ndarray
        Matrix with eigenvectors corresponding to real eigenvalues as columns.
    sC : float
        X-coordinate shift (mm) of complex parts of the solution.
        Used for numerical stability.
    sR : float
        X-coordinate shift (mm) of real parts of the solution.
        Used for numerical stability.
    sysmat : ndarray
        System matrix.
    lC : float
        Crack length whose maximum deflection equals the weak-layer thickness (mm).
    lS : float
        Crack length when touchdown exerts maximum support on the slab (mm).
        Corresponds to the longest possible unbedded length.
    ratio : float
        Increment factor for the weak-layer stiffness from intact to collapsed state.
    beta : float
        Describes the stiffnesses of weak-layer and slab.
    """

    def __init__(self, system='pst-', touchdown = False):
        """
        Initializes an Eigensystem instance for layered snow slab analysis.
        
        Sets up default physical constants, system type, and initializes all attributes related to weak-layer properties, slab properties, eigenvalues, eigenvectors, system matrices, and touchdown analysis. The system type determines the loading and boundary conditions to be analyzed, such as PST cuts or skier loading scenarios.
        """
        # Assign global attributes
        self.g = 9810           # Gravitation (mm/s^2)
        self.lski = 1000        # Effective out-of-plane length of skis (mm)
        self.tol = 1e-3         # Relative Romberg integration tolerance
        self.system = system    # System type: 'pst-', '-pst', 'skier', 'skiers'

        # Initialize weak-layer attributes
        self.weak = False       # Weak-layer properties dictionary
        self.t = False          # Weak-layer thickness (mm)
        self.tc = False         # Weak-layer collapse height (mm)

        # Initialize slab attributes
        self.p = 0          # Surface line load (N/mm)
        self.slab = False       # Slab properties matrix
        self.k = False          # Slab shear correction factor
        self.h = False          # Total slab height (mm)
        self.b = False          # Total snowpack thickness (mm)
        self.zs = False         # Z-coordinate of slab center of gravity (mm)
        self.zA = False         # Z-coordinate of weights center of gravity (mm)
        self.yA = False         # Y-coordinate of weights center of gravity (mm)
        self.phi = False        # Slab inclination (°)
        self.theta = False      # Slab rotation (°)         
        self.A11 = False        # Slab extensional stiffness (N/mm)
        self.B11 = False        # Slab bending-extension coupling stiffness (N)
        self.D11 = False        # Slab bending stiffness (Nmm)
        self.kA55 = False       # Slab shear stiffness (N/mm)
        self.kB55 = False       # Higher-order slab shear stiffness (N)
        self.kD55 = False       # Higher-order slab shear stiffness (Nmm)
        self.K0 = False         # Stiffness determinant (N^2)

        # Initialize eigensystem attributes
        self.ewC = False        # Complex eigenvalues
        self.ewR = False        # Real eigenvalues
        self.evC = False        # Complex eigenvectors
        self.evR = False        # Real eigenvectors
        self.sC = False         # Stability shift of complex eigenvalues (mm)
        self.sR = False         # Stability shift of real eigenvalues (mm)
        self.sysmat = False     # System matrix


        # Initialize touchdown attributes
        self.touchdown = touchdown
        self.lC = False         # Minimum length of substratum contact (mm)
        self.lS = False         # Maximum length of span between bedded and touchdowned boundary (mm)
        self.ratio = False      # Stiffness ratio of collapsed to uncollapsed weak-layer
        self.beta = False       # Ratio of slab to bedding stiffness

    def set_foundation_properties(self, t=30,  E=0.25, nu=0.25, rhoweak=100, constitutive='plane strain', update=False):
        """
        Sets the material properties and geometry of the weak-layer foundation.
        
        Configures weak-layer thickness, Young's modulus, Poisson's ratio, and density, adjusting for the specified constitutive behavior ('plane strain', 'plane stress', or 'uniaxial'). Optionally updates the fundamental system if properties are changed and `update` is True.
        
        Args:
            t: Weak-layer thickness in mm.
            E: Weak-layer Young's modulus in MPa.
            nu: Weak-layer Poisson's ratio.
            rhoweak: Weak-layer density in kg/m³.
            constitutive: Constitutive behavior for out-of-plane direction ('plane strain', 'plane stress', or 'uniaxial').
            update: If True, recalculates the fundamental system after updating properties.
        """
        # Geometry
        self.t = t              # Weak-layer thickness (mm)
        if constitutive =='plane strain':
            nuUpdate = nu
            EUpdate = E
        elif constitutive == 'plane stress':
            nuUpdate = nu/(1+nu)
            EUpdate = E*(1+2*nu)/((1+nu)**2)
        elif constitutive == 'uniaxial':
            nuUpdate = 0
            EUpdate = E

        # Material properties
        self.weak = {
            'nu': nuUpdate,            # Poisson's ratio (-)
            'E': EUpdate,              # Young's modulus (MPa)
            'rho': rhoweak*1e-12 # Density (t/mm^3)       
        }

        # Recalculate the fundamental system after properties have changed
        if update:
            self.calc_fundamental_system()

    def set_beam_properties(self, layers, phi, theta=0, C0=6.5, C1=4.40,
                           nu=0.25, b=290, k=5/6, update=False):
        """
                           Sets the material and geometric properties of the slab (beam), including layering, elastic moduli, inclination, and rotation.
                           
                           Args:
                               layers: 2D list or string specifying slab layering. If a list, each row contains density (kg/m³) and thickness (mm) for a layer. If a string, loads a predefined profile from the database.
                               phi: Inclination angle of the slab in degrees.
                               theta: Rotation angle of the slab in degrees (default 0).
                               C0: Multiplicative constant for Young's modulus parameterization (default 6.5).
                               C1: Exponent for Young's modulus parameterization (default 4.40).
                               nu: Poisson's ratio for all layers (default 0.25).
                               b: Total snowpack thickness in mm (default 290).
                               k: Shear correction factor (default 5/6).
                               update: If True, recalculates the fundamental system after updating properties.
                           
                           If `layers` is a string, loads density, thickness, and Young's modulus from a database profile. Otherwise, computes Young's modulus for each layer using the Bergfeld parameterization. Updates slab attributes including thickness, center of gravity, and elastic properties. Optionally updates the system matrices if `update` is True.
                           """
        if isinstance(layers, str):
            # Read layering and Young's modulus from database
            layers, E = load_dummy_profile(layers.split()[-1])
        else:
            # Compute Young's modulus from density parametrization
            layers = np.array(layers)
            E = bergfeld(layers[:, 0], C0=C0, C1=C1)  # Young's modulus (MPa)

        # Derive other elastic properties
        nu = nu*np.ones(layers.shape[0])         # Global Poisson's ratio
        G = E/(2*(1 + nu))                       # Shear modulus (MPa)
        self.k = k                              # Shear correction factor

        # Compute total slab thickness and center of gravity
        self.h, self.zs = calc_center_of_gravity(layers)
        self.b = b

        # Assemble layering into matrix (top to bottom)
        # Columns are:
        # - density (kg/m^3)
        # - layer thickness (mm)
        # - Young's modulus (MPa)
        # - shear modulus (MPa)
        # - Poisson's ratio
        self.slab = np.vstack([layers.T, E, G, nu]).T


        # Recalculate the fundamental system after properties have changed
        if update:
            self.calc_fundamental_system()

    def set_surface_load(self, p, yA = 0, zA = 0, width = 1.):
        """
        Sets a distributed surface line load acting in the vertical direction.
        
        Defines the magnitude and location of an external surface load applied to the slab, including optional out-of-plane and lateral coordinates and load width.
        
        Args:
            p: Magnitude of the distributed surface load (N/mm).
            yA: y-coordinate of the load application point (mm), optional.
            zA: z-coordinate of the load application point (mm), optional.
            width: Width over which the load is distributed (mm), optional.
        """
        self.p = p        # Total line load from the additional weigths
        self.zA = zA                            # z-coordiante of the additional weights 
        self.yA = yA                            # y-coordinate of the additional weigths 


    def calc_laminate_stiffness_matrix(self):
        """
        Calculates the plane-strain laminate stiffness matrix (ABD matrix) for the layered slab.
        
        Computes extensional (A11), bending-extension coupling (B11), bending (D11), and shear (kA55, kB55, kD55) stiffness components by summing contributions from each layer based on their elastic properties and thicknesses. Updates the corresponding class attributes.
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
        A11, B11, D11, kA55, kB55, kD55 = 0, 0, 0, 0, 0, 0
        # Add layerwise contributions
        for i in range(n):
            E, G, nu = self.slab[i, 2:5]
            A11 = A11 + E/(1-nu**2)*(z[i+1] - z[i])
            B11 = B11 + 1/2*E/(1-nu**2)*(z[i+1]**2 - z[i]**2)
            D11 = D11 + 1/3*E/(1-nu**2)*(z[i+1]**3 - z[i]**3)
            kA55 = kA55 + self.k*G*(z[i+1] - z[i])
            kB55 = kB55 + 1/2 * self.k*G*(z[i+1]**2 - z[i]**2)
            kD55 = kD55 + 1/3 * self.k*G*(z[i+1]**3 - z[i]**3)
        self.A11 = A11
        self.B11 = B11
        self.D11 = D11
        self.kA55 = kA55
        self.kB55 = kB55
        self.kD55 = kD55
        self.K0 = B11**2 - A11*D11

    

  

    def calc_system_matrix(self):
        """
        Assembles the 24x24 first-order ODE system matrix for the layered beam on an elastic foundation.
        
        The matrix incorporates slab and weak-layer stiffness, geometric, and material properties, and is used to represent the coupled equations of motion for the layered snow slab system. The resulting matrix is stored in `self.sysmat` and is fundamental for subsequent eigenvalue and solution computations.
        """
        Ew = self.weak['E']
        nuw = self.weak['nu']

        t = self.t
        h = self.h
        b = self.b
        A11 = self.A11
        B11 = self.B11
        D11 = self.D11
        kA55 = self.kA55
        kB55 = self.kB55
        kD55 = self.kD55
        Pi = np.pi

        c0201 = (-3*(2*D11 - B11*h)*Pi**2*Ew*(-1 + 2*nuw))/ \
            (t*(-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - 12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c0206 =-((4*h*kA55*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 6*D11*Ew*(-8 + Pi**2*(-1 + 4*nuw)) + 3*B11*(h*Ew*(8 + Pi**2 - 4*Pi**2*nuw) + 8*kA55*Pi**2*(-1 + nuw + 2*nuw**2)))/ \
            (8*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) - 8*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 2*A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) - 24*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) + \
            24*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))
        
        
        c0209 = -1/2*(-3*B11*Pi**2*(-1 + 2*nuw)*(h**2*Ew + 8*kA55*t*(1 + nuw)) + 2*h*Ew*(-2*kA55*(-6 + Pi**2)*t**2*(-1 + nuw) + 3*D11*Pi**2*(-1 + 2*nuw)))/ \
            (t*(-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c0213 = (3*(2*D11 - B11*h)*Pi**3*Ew*(-1 + 2*nuw))/ \
            (t*(-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))
        
        c0222 = (6*(2*D11 - B11*h)*Pi*Ew)/ \
            (4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) - 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) + A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) - 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) + \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2))

        c0403 = (3*Ew*(36*Pi**2*(2*kD55*Pi**2 - kB55*(-8 + Pi**2)*t)*(1 + nuw) + b**2*(Pi**2*(-6 + Pi**2)*t*Ew + 6*kA55*Pi**4*(1 + nuw)) - \
            3*h*((48 - 14*Pi**2 + Pi**4)*t**2*Ew + 12*kB55*Pi**4*(1 + nuw))))/ \
            (t*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))


        c0407 = (-3*Ew*(-((-1 + 2*nuw)*(24*Pi**2*t*(-3*kD55*(8 + Pi**2) + 2*kB55*(-6 + Pi**2)*t)*(1 + nuw) + 3*h**2*((48 - 14*Pi**2 + Pi**4)*t**2*Ew + 12*kB55*Pi**4*(1 + nuw)) + \
            4*h*((-6 + Pi**2)**2*t**3*Ew - 18*kD55*Pi**4*(1 + nuw) + 18*kB55*Pi**4*t*(1 + nuw)))) + b**2*(-24*kB55*Pi**4*(-1 + nuw**2) + (8 + Pi**2)*t*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + \
            6*kA55*Pi**2*(1 + nuw)) + h*(Pi**2*(-6 + Pi**2)*t*Ew + 6*kA55*Pi**4*(-1 + nuw + 2*nuw**2)))))/ \
            (2*t*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))


        c0412 = -1/4*(72*Pi**2*(-1 + nuw + 2*nuw**2)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw))) + \
            b**2*(-144*kA55**2*Pi**4*(1 + nuw)**2*(-1 + 2*nuw) - 48*kA55*Pi**2*(-6 + Pi**2)*t*Ew*(-1 + nuw + 2*nuw**2) + Ew*(36*kB55*Pi**2*(1 + nuw)*(8 + Pi**2*(-1 + 4*nuw)) + \
            (-6 + Pi**2)*t*Ew*(-4*(-6 + Pi**2)*t*(-1 + 2*nuw) + 3*h*(8 + Pi**2*(-1 + 4*nuw))))))/ \
            ((-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))
        
        c0416 =  (6*b*Pi*Ew*(h*(-6 + Pi**2)*t*Ew + 12*kB55*Pi**2*(1 + nuw)))/ \
            ((-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0417 = (-3*Pi*Ew*(b**2*(Pi**2*(-6 + Pi**2)*t*Ew + 6*kA55*Pi**4*(1 + nuw)) - 12*(h*(-6 + Pi**2)*t**2*Ew + 3*h*kB55*Pi**4*(1 + nuw) - 6*Pi**2*(kD55*Pi**2 - 2*kB55*t)*(1 + nuw))))/ \
            (t*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0423 = (-6*Pi*Ew*(-72*(h*kB55 - 2*kD55)*Pi**2*t*(-1 + nuw + 2*nuw**2) + b**2*(12*kB55*Pi**4*(-1 + nuw**2) + 12*kA55*Pi**2*t*(-1 + nuw + 2*nuw**2) + (-6 + Pi**2)*t*Ew*(h*Pi**2*(-1 + nuw) + \
            2*t*(-1 + 2*nuw)))))/ \
            (b*t*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0602 = (3*Ew*(8 + Pi**2*(-1 + 4*nuw)))/(2*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c0605 = (6*Pi**2*Ew*(-1 + nuw))/(t*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c0610 = (-3*(h*Ew*(-8 + Pi**2*(1 - 4*nuw)) + 8*kA55*Pi**2*(-1 + nuw + 2*nuw**2)))/(4*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c0614 = (6*Pi*Ew)/((-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c0619 = (24*Pi*Ew*nuw)/(b*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c0621 = (-6*Pi**3*Ew*(-1 + nuw))/(t*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c0803 = (-18*Ew*(6*h*kA55*Pi**4*(1 + nuw) - 12*kB55*Pi**4*(1 + nuw) + (-8 + Pi**2)*t*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))))/ \
            (t*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0807 = (3*Ew*(2*b**2*Pi**2*(-1 + nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)) + (-1 + 2*nuw)*(18*h**2*kA55*Pi**4*(1 + nuw) + 4*t*(-9*kB55*Pi**2*(8 + Pi**2)*(1 + nuw) + \
            (-6 + Pi**2)*t*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))) + h*(-36*kB55*Pi**4*(1 + nuw) + 3*t*((48 - 14*Pi**2 + Pi**4)*t*Ew + 12*kA55*Pi**4*(1 + nuw))))))/ \
            (t*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0812 = (-3*b**2*Ew*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))*(8 + Pi**2*(-1 + 4*nuw)))/ \
            (2*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0816 = (12*b*Pi*Ew*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))/ \
            ((-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0817 = (36*Pi*Ew*(3*h*kA55*Pi**4*(1 + nuw) - 6*kB55*Pi**4*(1 + nuw) + 2*t*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))))/ \
            (t*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c0823 = (-12*Pi**3*Ew*(-36*(h*kA55 - 2*kB55)*t*(-1 + nuw + 2*nuw**2) + b**2*(-1 + nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))))/ \
            (b*t*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c1001 = (3*(2*B11 - A11*h)*Pi**2*Ew*(-1 + 2*nuw))/ \
            (t*(-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1006 = -((-8*kA55*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 6*B11*Ew*(8 + Pi**2 - 4*Pi**2*nuw) - 3*A11*(h*Ew*(8 + Pi**2 - 4*Pi**2*nuw) + 8*kA55*Pi**2*(-1 + nuw + 2*nuw**2)))/ \
            (8*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) - 8*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 2*A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) - 24*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) +  \
            24*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))


        c1009 = -1/2*(6*B11*h*Pi**2*Ew*(1 - 2*nuw) + 8*kA55*(-6 + Pi**2)*t**2*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + 2*nuw)*(h**2*Ew + 8*kA55*t*(1 + nuw)))/ \
            (t*(-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))


        c1013 = (-3*(2*B11 - A11*h)*Pi**3*Ew*(-1 + 2*nuw))/ \
            (t*(-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))


        c1022 = (6*(2*B11 - A11*h)*Pi*Ew)/ \
            (-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2))

        c1204 = (-6*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))/(b**2*((-6 + Pi**2)*t*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1208 = (3*(4*h*(-6 + Pi**2)*t*Ew*(-1 + 2*nuw) + 48*kB55*Pi**2*(-1 + nuw + 2*nuw**2) + b**2*Ew*(-8 + Pi**2*(-1 + 4*nuw))))/ \
            (4*b**2*((-6 + Pi**2)*t*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1211 =  (3*(-1 + 2*nuw)*(b**2*Pi**2*Ew + 4*(-6 + Pi**2)*t**2*Ew + 24*kA55*Pi**2*t*(1 + nuw)))/(2*b**2*t*((-6 + Pi**2)*t*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1215 = (3*Pi**3*Ew*(-1 + 2*nuw))/(b*t*((-6 + Pi**2)*t*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1224= (-6*Pi*Ew)/(b*((-6 + Pi**2)*t*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1401 = (3*(4*D11 + h*(-4*B11 + A11*h))*Pi*Ew*(-1 + 2*nuw))/ \
            (t*(-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))


        c1406 = -1/2*(Pi*(48*B11**2*(-1 + nuw + 2*nuw**2) - 48*A11*D11*(-1 + nuw + 2*nuw**2) + 4*D11*t*Ew*(7 - 19*nuw + 12*nuw**2) + A11*h*t*(-1 + nuw)*(h*Ew*(-7 + 12*nuw) - \
            24*kA55*(-1 + nuw + 2*nuw**2)) + 4*B11*t*(-1 + nuw)*(h*Ew*(7 - 12*nuw) + 12*kA55*(-1 + nuw + 2*nuw**2))))/ \
            (t*(-1 + nuw)*(-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1409 = (-3*Pi*(-1 + 2*nuw)*(4*B11*(h**2*Ew + 4*kA55*t*(1 + nuw)) - h*(4*D11*Ew + A11*h**2*Ew + 8*A11*kA55*t*(1 + nuw))))/ \
            (2*t*(-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1413 = -1/2*(Pi**4*(-1 + 2*nuw)*(4*D11*t*Ew*(-1 + nuw) - 4*B11*h*t*Ew*(-1 + nuw) + A11*h**2*t*Ew*(-1 + nuw) - 12*B11**2*(-1 + nuw + 2*nuw**2) + 12*A11*D11*(-1 + nuw + 2*nuw**2)))/ \
            (t**2*(-1 + nuw)*(-4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) - A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) + 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) - \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1420 = -2 * nuw /(b - b * nuw)

        c1422 = (-6*(4*D11 + h*(-4*B11 + A11*h))*Ew)/ \
            (4*D11*(-6 + Pi**2)*t*Ew*(-1 + nuw) - 4*B11*h*(-6 + Pi**2)*t*Ew*(-1 + nuw) + A11*h**2*(-6 + Pi**2)*t*Ew*(-1 + nuw) - 12*B11**2*Pi**2*(-1 + nuw + 2*nuw**2) + \
            12*A11*D11*Pi**2*(-1 + nuw + 2*nuw**2))

        c1604 = (-18*Pi*(1 + nuw)*(-1 + 2*nuw)*(A11 + 2*kA55*(-1 + nuw) - 2*A11*nuw))/(b*(-1 + nuw)*((-6 + Pi**2)*t*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1608 = -1/4*(Pi*(12*A11*(-1 + nuw + 2*nuw**2)*(b**2 + 3*h*t*(-1 + 2*nuw)) - t*(-1 + nuw)*(b**2*Ew*(-7 + 12*nuw) + 144*kB55*(-1 + nuw + 2*nuw**2))))/ \
            (b*t*(-1 + nuw)*((-6 + Pi**2)*t*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))  

        c1611 = (-3*Pi*(-1 + 2*nuw)*(12*A11*t*(-1 + nuw + 2*nuw**2) - (-1 + nuw)*(b**2*Ew + 24*kA55*t*(1 + nuw))))/ \
            (2*b*t*(-1 + nuw)*((-6 + Pi**2)*t*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))


        c1615 = ((-1 + 2*nuw)*(t*(b**2*Pi**4 + 12*(-6 + Pi**2)*t**2)*Ew*(-1 + nuw) + 3*A11*Pi**2*(b**2*Pi**2 + 12*t**2)*(-1 + nuw + 2*nuw**2)))/ \
            (2*b**2*t**2*(-1 + nuw)*((-6 + Pi**2)*t*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2)))

        c1618 = -((3 - 6*nuw)/(b*(-1 + nuw)))

        c1624 =  (-6*Ew)/((-6 + Pi**2)*t*Ew*(-1 + nuw) + 3*A11*Pi**2*(-1 + nuw + 2*nuw**2))

        c1803 =(-6*Pi*Ew*(18*(h**2*kA55*Pi**2 - 4*h*kB55*Pi**2 + 4*kD55*Pi**2 + h*kA55*(-8 + Pi**2)*t - 2*kB55*(-8 + Pi**2)*t)*(1 + nuw) + b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))))/ \
            (t*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c1807 = -((18*Pi*(-1 + nuw + 2*nuw**2)*(-4*kD55*Pi**2*(3*h + 7*t)*Ew + 2*kB55*(6*h**2*Pi**2 + h*(-24 + 17*Pi**2)*t + 4*(-6 + Pi**2)*t**2)*Ew + 96*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h*(3*h**2*Pi**2 + 2*h*(-12 + 5*Pi**2)*t + 4*(-6 + Pi**2)*t**2)*Ew + 96*kD55*Pi**2*(1 + nuw))) - b**2*Pi*(144*kA55**2*Pi**2*(1 + nuw)**2*(-1 + 2*nuw) + \
            6*kA55*Ew*(1 + nuw)*((-24 + 11*Pi**2)*t*(-1 + 2*nuw) + 3*h*Pi**2*(-3 + 4*nuw)) + Ew*((-6 + Pi**2)*t*(3*h + 7*t)*Ew*(-1 + 2*nuw) - 72*kB55*Pi**2*(-1 + nuw**2))))/ \
            (t*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw))))))

        c1812 = (-9*b**2*(h*kA55 - 2*kB55)*Pi*Ew*(1 + nuw)*(8 + Pi**2*(-1 + 4*nuw)))/ \
            ((-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c1816 = -((-36*Pi**2*(-1 + nuw + 2*nuw**2)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) -kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw))) + \
            2*b**2*(36*kA55**2*Pi**4*(1 + nuw)**2*(-1 + 2*nuw) - 12*kA55*Pi**2*Ew*(1 + nuw)*(3*h - (-6 + Pi**2)*t*(-1 + 2*nuw)) + Ew*(72*kB55*Pi**2*(1 + nuw) + (-6 + Pi**2)**2*t**2*Ew*(-1 + 2*nuw))))/ \
            (b*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) -  \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw))))))

        c1817 = -((-(b**2*Pi**4*(t*Ew + 6*kA55*(1 + nuw))*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))) + 18*Pi**2*(1 + nuw)*(-4*kD55*Pi**4*t*Ew + 4*kB55*t*(h*Pi**4 + 12*t)*Ew + 24*kB55**2*Pi**4*(1 + nuw) - kA55*(h*t*(h*Pi**4 + 24*t)*Ew + 24*kD55*Pi**4*(1 + nuw))))/
            (t**2*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 -18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw))))))

        c1823 = (24*Ew*(18*(h**2*kA55 - 4*h*kB55 + 4*kD55)*Pi**2*t*(-1 + nuw + 2*nuw**2) + b**2*(-3*h*kA55*Pi**4*(-1 + nuw**2) + 6*kB55*Pi**4*(-1 + nuw**2) + t*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + \
            6*kA55*Pi**2*(1 + nuw)))))/ \
            (b*t*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2002 = (24*nuw)/(b*Pi - 2*b*Pi*nuw)

        c2005 = (-48*nuw)/(b*Pi*t - 2*b*Pi*t*nuw)

        c2010 = (12*h*nuw)/(b*Pi - 2*b*Pi*nuw)

        c2014 = (12*nuw)/(b - 2*b*nuw)

        c2019 =Pi**2/t**2 + (24*(-1 + nuw))/(b**2*(-1 + 2*nuw))

        c2202 = -((Pi*(24*kA55*(1 + nuw) + t*Ew*(1 + 12*nuw)))/(t*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))))

        c2205 = (-12*Pi*Ew*(-1 + nuw))/(t*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c2210 = -1/2*(Pi*(24*h*kA55*(1 + nuw) + h*t*Ew*(1 + 12*nuw) - 24*kA55*t*(-1 + nuw + 2*nuw**2)))/(t*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c2214 = (-12*Ew)/((-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c2219 = (-48*Ew*nuw)/(b*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c2221 = (2*Pi**4*(-1 + nuw)*(t*Ew + 6*kA55*(1 + nuw)))/(t**2*(-1 + 2*nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))

        c2403 = (-6*Pi*(b**2*(-6*kA55*(3*h*Pi**2 + (24 - 5*Pi**2)*t)*Ew*(1 + nuw) + 144*kA55**2*Pi**2*(1 + nuw)**2 + Ew*((-6 + Pi**2)*t**2*Ew + 36*kB55*Pi**2*(1 + nuw))) - \
            72*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))/ \
            (b*t*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2407 = (-3*Pi*(2*b**4*Ew*(-1 + nuw)*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))) - 72*(h + t)*(-1 + nuw + 2*nuw**2)*(h**2*kA55*(-6 + Pi**2)*t*Ew -\
            4*h*kB55*(-6 + Pi**2)*t*Ew + 4*kD55*(-6 + Pi**2)*t*Ew - 24*kB55**2*Pi**2*(1 + nuw) + 24*kA55*kD55*Pi**2*(1 + nuw)) - b**2*(-1 + 2*nuw)*(-18*h**2*kA55*Pi**2*Ew*(1 + nuw) + 12*t*(1 + nuw)* \
            (3*kB55*(8 + Pi**2)*Ew + 2*kA55*(-6 + Pi**2)*t*Ew + 12*kA55**2*Pi**2*(1 + nuw)) + h*(12*kA55*(-24 + Pi**2)*t*Ew*(1 + nuw) + 144*kA55**2*Pi**2*(1 + nuw)**2 + \
            Ew*((-6 + Pi**2)*t**2*Ew + 36*kB55*Pi**2*(1 + nuw)))))/ \
            (b*t*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))


        c2412 = -1/2*(72*b*Pi*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw))) - \
                    b**3*Pi*(144*kA55**2*Pi**2*(1 + nuw)**2 + (-6 + Pi**2)*t**2*Ew**2*(1 + 12*nuw) + 6*kA55*t*Ew*(1 + nuw)*(-24 + Pi**2*(5 + 12*nuw))))/(t*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + \
                    6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2416 = (-12*b**2*Ew*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)))/ \
            ((-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2417 = (-36*b*Ew*(3*h*kA55*Pi**4*(1 + nuw) - 6*kB55*Pi**4*(1 + nuw) + 2*t*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))))/ \
            (t*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw)))))

        c2423 = -((-(2*b**4*Pi**4*(-1 + nuw)*(t*Ew + 6*kA55*(1 + nuw))*((-6 + Pi**2)*t*Ew) + 6*kA55*Pi**2*(1 + nuw)) + 216*Pi**2*t**2*(-1 + nuw + 2*nuw**2)*(4*h*kB55*(-6 + Pi**2)*t*Ew - \
            4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw))) + 12*b**2*(12*kB55*Pi**2*t*Ew*(6*t*(1 - 2*nuw) + \
            h*Pi**2*(-6 + Pi**2)*(-1 + nuw))*(1 + nuw) + 72*kB55**2*Pi**6*(-1 + nuw)*(1 + nuw)**2 - 36*kA55**2*Pi**4*t**2*(1 + nuw)**2*(-1 + 2*nuw) - \
            (-6 + Pi**2)*t*Ew*((-6 + Pi**2)*t**3*Ew*(-1 + 2*nuw) + 12*kD55*Pi**4*(-1 + nuw**2)) - 3*kA55*Pi**2*(1 + nuw)*(24*kD55*Pi**4*(-1 + nuw**2) + t*Ew*(12*h*t*(1 - 2*nuw) + \
            h**2*Pi**2*(-6 + Pi**2)*(-1 + nuw) + 4*(-6 + Pi**2)*t**2*(-1 + 2*nuw)))))/ \
            (b**2*t**2*(-1 + 2*nuw)*(b**2*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))**2 - 18*Pi**2*(1 + nuw)*(4*h*kB55*(-6 + Pi**2)*t*Ew - 4*kD55*(-6 + Pi**2)*t*Ew + 24*kB55**2*Pi**2*(1 + nuw) - \
            kA55*(h**2*(-6 + Pi**2)*t*Ew + 24*kD55*Pi**2*(1 + nuw))))))

        SystemMatrixC = [[    0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [c0201,     0,     0,     0,     0, c0206,     0,     0, c0209,     0,     0,     0, c0213,     0,     0,     0,     0,     0,     0,     0,     0, c0222,     0,     0],
                         [    0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    0,     0, c0403,     0,     0,     0, c0407,     0,     0,     0,     0, c0412,     0,     0,     0, c0416, c0417,     0,     0,     0,     0,     0, c0423,     0],
                         [    0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    0, c0602,     0,     0, c0605,     0,     0,     0,     0, c0610,     0,     0,     0, c0614,     0,     0,     0,     0, c0619,     0, c0621,     0,     0,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    0,     0, c0803,     0,     0,     0, c0807,     0,     0,     0,     0, c0812,     0,     0,     0, c0816, c0817,     0,     0,     0,     0,     0, c0823,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [c1001,     0,     0,     0,     0, c1006,     0,     0, c1009,     0,     0,     0, c1013,     0,     0,     0,     0,     0,     0,     0,     0, c1022,     0,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    0,     0,     0, c1204,     0,     0,     0, c1208,     0,     0, c1211,     0,     0,     0, c1215,     0,     0,     0,     0,     0,     0,     0,     0, c1224],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
                         [c1401,     0,     0,     0,     0, c1406,     0,     0, c1409,     0,     0,     0, c1413,     0,     0,     0,     0,     0,     0, c1420,     0, c1422,     0,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0],
                         [    0,     0,     0, c1604,     0,     0,     0, c1608,     0,     0, c1611,     0,     0,     0, c1615,     0,     0, c1618,     0,     0,     0,     0,     0, c1624],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0],
                         [    0,     0, c1803,     0,     0,     0, c1807,     0,     0,     0,     0, c1812,     0,     0,     0, c1816, c1817,     0,     0,     0,     0,     0, c1823,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0],
                         [    0, c2002,     0,     0, c2005,     0,     0,     0,     0, c2010,     0,     0,     0, c2014,     0,     0,     0,     0, c2019,     0,     0,     0,     0,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0],
                         [    0, c2202,     0,     0, c2205,     0,     0,     0,     0, c2210,     0,     0,     0, c2214,     0,     0,     0,     0, c2219,     0, c2221,     0,     0,     0],
                         [    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1],
                         [    0,     0, c2403,     0,     0,     0, c2407,     0,     0,     0,     0, c2412,     0,     0,     0, c2416, c2417,     0,     0,     0,     0,     0, c2423,     0],]
        

        self.sysmat = np.array(SystemMatrixC)

    def calc_eigensystem(self):
        """
        Computes and classifies the eigenvalues and eigenvectors of the system matrix.
        
        Separates real and complex eigenvalues and their corresponding eigenvectors, storing them for later use. Sets numerical stability shifts for positive eigenvalues to enhance robustness.
        """
        # Calculate eigenvalues (ew) and eigenvectors (ev)
        ew, ev = np.linalg.eig(self.sysmat)
        # Classify real and complex eigenvalues
        real = (ew.imag == 0) & (ew.real != 0)  # real eigenvalues
        cmplx = ew.imag > 0                   # positive complex conjugates
        # Eigenvalues
        self.ewC = ew[cmplx]
        self.ewR = ew[real].real
        # Eigenvectors
        self.evC = np.around(ev[:, cmplx],15)
        self.evR = np.around(ev[:, real].real,15)
        # Prepare positive eigenvalue shifts for numerical robustness
        self.sR, self.sC = np.zeros(self.ewR.shape), np.zeros(self.ewC.shape)
        self.sR[self.ewR > 0], self.sC[self.ewC > 0] = -1, -1

    def calc_fundamental_system(self):
        """
        Updates the fundamental system by recalculating stiffness matrices, the system matrix, and the eigensystem.
        
        This method should be called after any change to slab or foundation properties to ensure all system matrices and eigenvalues are consistent with the current configuration.
        """
        self.calc_laminate_stiffness_matrix()
        self.calc_system_matrix()
        self.calc_eigensystem()

    def get_weight_load(self, phi, theta = 0):
        """
        Calculates the line load components from the slab's own weight, accounting for inclination and rotation.
        
        Args:
            phi: Slab inclination angle in degrees (counterclockwise positive).
            theta: Slab rotation angle in degrees (default is 0).
        
        Returns:
            qx: Axial component of the line load (N/mm).
            qy: Out-of-plane component of the line load (N/mm).
            qz: Normal component of the line load (N/mm).
        
        The load is decomposed into axial, out-of-plane, and normal directions at the slab's center of gravity.
        """
        # Convert units
        phi = np.deg2rad(phi)                   # Convert inclination to rad
        theta = np.deg2rad(theta)               # Convert rotation to rad
        rho = self.slab[:, 0]*1e-12             # Convert density to t/mm^3
        # Sum up layer weight loads
        q = sum(rho*self.b*self.g*self.slab[:, 1])     # Line load (N/mm)
        # Split into components
        qz = q*np.cos(phi)* np.cos(theta)       # normal direction 
        qx = -q*np.sin(phi)                     # axial  direction
        qy = q*np.sin(theta)                    # out-of-plane direction

        

        return qx, qy, qz

    def get_surface_load(self, phi, theta = 0):
        """
        Calculates the components of the applied surface line load in tangential, out-of-plane, and normal directions based on slab inclination and rotation.
        
        Args:
            phi: Slab inclination angle in degrees (counterclockwise positive).
            theta: Slab rotation angle in degrees (default is 0).
        
        Returns:
            Tuple of floats (pt, po, pn) representing the surface line load (N/mm) in the tangential, out-of-plane, and normal directions, respectively.
        """
        # Convert units
        phi = np.deg2rad(phi)                   # Convert inclination to rad
        theta = np.deg2rad(theta)
        # Split into components
        pz = self.p*np.cos(phi) *np.cos(theta)  # Normal direction
        px = -self.p*np.sin(phi)                # Tangential direction
        py = self.p * np.sin(theta)             # Out-of-plane direction
   
        return px, py, pz

    def get_skier_load(self, m, phi,theta = 0):
        """
        Calculates the components of a skier's point load on the slab, accounting for inclination and rotation.
        
        Args:
            m: Skier mass in kilograms.
            phi: Slab inclination angle in degrees (counterclockwise positive).
            theta: Slab rotation angle in degrees (default is 0).
        
        Returns:
            Fx: Tangential component of the skier load (N).
            Fy: Out-of-plane component of the skier load (N).
            Fz: Normal component of the skier load (N).
        """
        phi = np.deg2rad(phi)                   # Convert inclination to rad
        theta = np.deg2rad(theta)               # Convert rotation to rad
        F = 1e-3*np.array(m)*self.g/self.lski   # Total skier load (N)
        Fz = F*np.cos(phi)*np.cos(theta)        # Normal skier load (N)
        Fx = -F*np.sin(phi)                     # Tangential skier load (N)
        Fy = F*np.sin(theta)                    # Out-of-plane skier load  (N)

        return Fx, Fy, Fz


    def zh(self, x, l=0, bed=True):
        """
        Computes the complementary solution matrix at position x for either a bedded (with foundation) or free (unbedded) segment.
        
        For bedded segments, constructs the solution using eigenvalues and eigenvectors of the system matrix, combining real and complex parts. For free segments, returns a predefined matrix of polynomial and stiffness-based terms representing the complementary solution.
        
        Args:
            x: Horizontal coordinate (mm).
            l: Segment length (mm), used for phase shifting in bedded segments.
            bed: If True, computes the solution for a bedded segment; if False, for a free segment.
        
        Returns:
            The complementary solution matrix at position x. For bedded segments, the shape matches the system size (typically 24x24); for free segments, returns a 12x12 matrix.
        """

        A11 = self.A11
        B11 = self.B11
        D11 = self.D11
        kA55 = self.kA55
        kB55 = self.kB55
        b = np.longdouble(self.b)

        if bed:
            zh = np.around(np.concatenate([
                # Real
                self.evR*np.exp(self.ewR*(x + l*self.sR)),
                # Complex
                np.exp(self.ewC.real*(x + l*self.sC))*(
                       self.evC.real*np.cos(self.ewC.imag*x)
                       - self.evC.imag*np.sin(self.ewC.imag*x)),
                # Complex
                np.exp(self.ewC.real*(x + l*self.sC))*(
                       self.evC.imag*np.cos(self.ewC.imag*x)
                       + self.evC.real*np.sin(self.ewC.imag*x))], axis=1),15)
        else:
            # Abbreviations for the unbedded segemnts in accordance with Mathematica script
            H0101=1
            H0102=x
            H0104=(B11*kA55*x**2)/(2*B11**2 - 2*A11*D11)
            H0105=(B11*kA55*x**2)/(2*B11**2 - 2*A11*D11)

            H0202=1
            H0204=(B11*kA55*x)/(B11**2 - A11*D11)
            H0205=(B11*kA55*x)/(B11**2 - A11*D11)

            H0307=1
            H0308=x - (2*kA55*x**3)/(A11*b**2)
            H0310=(2*kB55*x**3)/(A11*b**2)
            H0311=(2*kA55*x**3)/(A11*b**2)
            H0312=x**2/2

            H0408=1 - (6*kA55*x**2)/(A11*b**2)
            H0410=(6*kB55*x**2)/(A11*b**2)
            H0411=(6*kA55*x**2)/(A11*b**2)
            H0412=x

            H0503=1
            H0504=(6*B11**2*x - 6*A11*D11*x + A11*kA55*x**3)/(6*B11**2 - 6*A11*D11)
            H0505=(A11*kA55*x**3)/(6*B11**2 - 6*A11*D11)
            H0506=-1/2*x**2

            H0604=(2*B11**2 - 2*A11*D11 + A11*kA55*x**2)/(2*B11**2 - 2*A11*D11)
            H0605=(A11*kA55*x**2)/(2*B11**2 - 2*A11*D11)
            H0606=-x

            H0709=1
            H0710=x

            H0810=1

            H0904=(A11*kA55*x**2)/(-2*B11**2 + 2*A11*D11)
            H0905=(-2*B11**2 + 2*A11*D11 + A11*kA55*x**2)/(-2*B11**2 + 2*A11*D11)
            H0906=x

            H1004=(A11*kA55*x)/(-B11**2 + A11*D11)
            H1005=(A11*kA55*x)/(-B11**2 + A11*D11)
            H1006=1

            H1108=(-6*kA55*x**2)/(A11*b**2)
            H1110=(6*kB55*x**2)/(A11*b**2)
            H1111=1 + (6*kA55*x**2)/(A11*b**2)
            H1112=x

            H1208=(-12*kA55*x)/(A11*b**2)
            H1210=(12*kB55*x)/(A11*b**2)
            H1211=(12*kA55*x)/(A11*b**2)
            H1212=1


            # Complementary solution matrix of free segments
            zh = np.array(
                [[H0101, H0102,     0, H0104, H0105,     0,     0,     0,     0,     0,     0,     0],
                 [    0, H0202,     0, H0204, H0205,     0,     0,     0,     0,     0,     0,     0],
                 [    0,     0,     0,     0,     0,     0, H0307, H0308,     0, H0310, H0311, H0312],
                 [    0,     0,     0,     0,     0,     0,     0, H0408,     0, H0410, H0411, H0412],
                 [    0,     0, H0503, H0504, H0505, H0506,     0,     0,     0,     0,     0,     0],
                 [    0,     0,     0, H0604, H0605, H0606,     0,     0,     0,     0,     0,     0],
                 [    0,     0,     0,     0,     0,     0,     0,     0, H0709, H0710,     0,     0],
                 [    0,     0,     0,     0,     0,     0,     0,     0,     0, H0810,     0,     0],
                 [    0,     0,     0, H0904, H0905, H0906,     0,     0,     0,     0,     0,     0],
                 [    0,     0,     0, H1004, H1005, H1006,     0,     0,     0,     0,     0,     0],
                 [    0,     0,     0,     0,     0,     0,     0, H1108,     0, H1110, H1111, H1112],
                 [    0,     0,     0,     0,     0,     0,     0, H1208,     0, H1210, H1211, H1212]])

        return zh

    def zp(self, x, phi,theta = 0, bed=True, load = False):
        """
        Computes the particular integral vector at position x for a layered beam system, accounting for slab inclination, rotation, and loading.
        
        Depending on the `bed` flag, returns the particular solution for either a bedded segment (with elastic foundation) or a free segment (without foundation). The vector incorporates the effects of slab self-weight, surface loads, geometric properties, and material parameters. If `load` is False, external loads are omitted from the calculation.
        
        Args:
            x: Horizontal coordinate (mm).
            phi: Slab inclination angle in degrees.
            theta: Slab rotation angle in degrees (default is 0).
            bed: If True, computes for a bedded segment; if False, for a free segment.
            load: If False, ignores external loads in the calculation.
        
        Returns:
            ndarray: Particular integral vector at position x (24x1 for bedded, 12x1 for free segments).
        """
        # Get weight and surface loads
        qx, qy, qz = self.get_weight_load(phi,theta)
        
        px, py, pz = self.get_surface_load(phi,theta)
        # Unpack laminate stiffnesses
        A11 = self.A11
        B11 = self.B11
        kA55 = self.kA55
        D11 = self.D11
        kB55 = self.kB55
        kD55 = self.kD55
        b = np.longdouble(self.b)
        h = self.h
        t = self.t
        # Unpack weak layer properties
        Ew = self.weak['E']
        nuw = self.weak['nu']
        rhow = self.weak['rho']

        # Unpack layering  information

        # Unpack general variables
        g = self.g
        Pi = np.pi


        # Unpack geometric properties
        h = self.h
        t = self.t
        zS = self.zs
        zA = self.zA
        yA = self.yA
        
        if not load:
            px = 0
            py = 0
            pz = 0
        my = -qx * zS - px * zA
        mz = -px * yA
        mx = -qy * zS + pz * yA - py * zA
        # Assemble particular integral vectors in accordance with Mathematica scripts

        if bed:
            zp = np.array([
                [(h*(h*(px + qx) - 2*(px*zA + qx*zS))*Ew + 8*kA55*(px + qx)*t*(1 + nuw) - 4*b*g*kA55*t**2*(1 + nuw)*rhow*np.sin(np.deg2rad(phi)))/(4*b*kA55*Ew)],
                [0],
                [np.longdouble((t*(1 + nuw)*(2*(np.longdouble(2*b**4*Pi**8*(py + qy))*(-1 + nuw)**2 + 6*Pi**2*(3*h**2*Pi**2*(-8 + Pi**2)*(py + qy) + 6*h*Pi**2*(-8 + Pi**2)*\
                    (pz*yA + py*(t - zA) + qy*(t - zS)) + 2*t*(2*(-24 - 6*Pi**2 + Pi**4)*py*t + 2*(-24 - 6*Pi**2 + Pi**4)*qy*t + 3*Pi**2*(-8 + Pi**2)*\
                    pz*yA - 3*Pi**2*(-8 + Pi**2)*py*zA - 3*Pi**2*(-8 + Pi**2)*qy*zS))*(t - 2*t*nuw)**2 + b**2*Pi**4*(3*h**2*Pi**4*(py + qy) + \
                    6*h*Pi**4*(pz*yA + py*(t - zA) + qy*(t - zS)) + 2*t*(2*(-24 + 3*Pi**2 + Pi**4)*py*t + 2*(-24 + 3*Pi**2 + Pi**4)*qy*t + 3*Pi**4*pz*yA - 3*Pi**4*py*zA -\
                    3*Pi**4*qy*zS))*(1 - 3*nuw + 2*nuw**2)) + b*g*t*(6*t**3*(3*h*(256 - 32*Pi**2 - 8*Pi**4 + Pi**6) + 4*(192 - 48*Pi**2 - 6*Pi**4 + Pi**6)*t)*(1 - 2*nuw)**2 + \
                    np.longdouble(2*b**4*Pi**8*(-1 + nuw)**2) + b**2*Pi**4*t*(3*h*(-32 + Pi**4) + 4*(-48 + 3*Pi**2 + Pi**4)*t)*(1 - 3*nuw + 2*nuw**2))*rhow*np.sin(np.deg2rad(theta))))/ \
                    (b*Ew*(6*(768 - 96*Pi**2 - 8*Pi**4 + Pi**6)*t**4*(1 - 2*nuw)**2 + np.longdouble(2*b**4*Pi**8*(-1 + nuw)**2 )+ b**2*Pi**4*(-192 + 12*Pi**2 + Pi**4)*t**2*(1 - 3*nuw + 2*nuw**2))))],
                [0],
                [(Pi**2*t*(-1 + nuw + 2*nuw**2)*(24*t**2*(-1 + nuw) + b**2*Pi**2*(-1 + 2*nuw))*(2*(pz + qz) + b*g*t*rhow*np.cos(np.deg2rad(theta))*np.cos(np.deg2rad(phi))))/ \
                 (2*b*Ew*(24*t**2*(Pi**2*(-1 + nuw)**2 - 8*nuw**2) + b**2*Pi**4*(1 - 3*nuw + 2*nuw**2)))],
                [0],
                [(6*t*(1 + nuw)*(-1 + 2*nuw)*(2*Pi**4*(h*(py + qy) + py*t + qy*t + 2*pz*yA - 2*py*zA - 2*qy*zS) + b*g*(-32 + Pi**4)*t**2*rhow*np.sin(np.deg2rad(theta))))/ \
                 (b*Ew*(2*b**2*Pi**4*(-1 + nuw) + (-96 + Pi**4)*t**2*(-1 + 2*nuw)))],
                [0],
                [(-(h*(px + qx)) + 2*(px*zA + qx*zS))/(2*b*kA55)],
                [0],
                [np.longdouble((-24*Pi**2*px*t*(b**2*Pi**2 + 12*t**2)*yA*(1 + nuw))/ \
                 np.longdouble(b**5*Pi**4*Ew + 48*b*t**3*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)) + 4*b**3*Pi**2*t*((3 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))))],
                [0],
                [(-8*g*t**2*(1 + nuw)*rhow*np.sin(np.deg2rad(phi)))/(Pi**3*Ew)],
                [0],
                [(-288*Pi*px*t**3*yA*(1 + nuw))/ \
                 np.longdouble((b**4*Pi**4*Ew + 48*t**3*((-6 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw)) + 4*b**2*Pi**2*t*((3 + Pi**2)*t*Ew + 6*kA55*Pi**2*(1 + nuw))))],
                [0],
                [(16*Pi*t**2*(1 + nuw)*(-3*(h*(py + qy) + py*t + qy*t + 2*pz*yA - 2*py*zA - 2*qy*zS)*(-1 + 2*nuw) + b*g*(t**2*(1 - 2*nuw) +\
                    b**2*(-1 + nuw))*rhow*np.sin(np.deg2rad(theta))))/(b*Ew*(2*b**2*Pi**4*(-1 + nuw) + (-96 + Pi**4)*t**2*(-1 + 2*nuw)))],
                [0],
                [(-24*Pi*t**2*nuw*(1 + nuw)*(-1 + 2*nuw)*(2*(pz + qz) + b*g*t*rhow*np.cos(np.deg2rad(theta))*np.cos(np.deg2rad(phi))))/ \
                 (Ew*(24*t**2*(Pi**2*(-1 + nuw)**2 - 8*nuw**2) + b**2*Pi**4*(1 - 3*nuw + 2*nuw**2)))],
                [0],
                [(4*g*t**2*(-1 + nuw + 2*nuw**2)*rhow*np.cos(np.deg2rad(theta))*np.cos(np.deg2rad(phi)))/(Pi**3*Ew*(-1 + nuw))],
                [0],
                [(12*Pi*t**2*(1 + nuw)*(-1 + 2*nuw)*(2*(py + qy) + b*g*t*rhow*np.sin(np.deg2rad(theta))))/(Ew*(b**2*Pi**4*(-1 + nuw) + 6*(-8 + Pi**2)*t**2*(-1 + 2*nuw)))],
                [0]],dtype = np.double)
            
        else:
            zp01=-1/6*(x**2*(3*B11*my - 3*D11*(qx+px) + B11*(qz+pz)*x))/(b*(B11**2 - A11*D11))
            zp02=-1/2*(x*(2*B11*my - 2*D11*(qx+px) + B11*(qz+pz)*x))/(b*(B11**2 - A11*D11))
            zp03=(x**2*(-((b**2*(12*kB55*mx + b**2*kA55*(qy+py) + 12*kD55*(qy+py)))/(b**2*kA55*kA55 - 12*kB55**2 + 12*kA55*kD55)) + (x*(-4*mz + (qy+py)*x))/A11))/(2*b**3)
            zp04=(x*(-((b**2*(12*kB55*mx + b**2*kA55*(qy+py) + 12*kD55*(py+qy)))/(b**2*kA55*kA55 - 12*kB55**2 + 12*kA55*kD55)) + (2*x*(-3*mz + (qy+py)*x))/A11))/b**3
            zp05=-1/24*(x**2*(12*B11**2*(qz+pz) - 12*A11*D11*(qz+pz) - 4*B11*kA55*(qx+px)*x + A11*kA55*x*(4*my + (qz+pz)*x)))/(b*(B11**2 - A11*D11)*kA55)
            zp06=-1/6*(x*(6*B11**2*(qz+pz) - 6*A11*D11*(qz+pz) - 3*B11*kA55*(qx+px)*x + A11*kA55*x*(3*my + (qz+pz)*x)))/(b*(B11**2 - A11*D11)*kA55)
            zp07=(-6*(kA55*mx + kB55*(qy+py))*x**2)/(b*(b**2*kA55*kA55 - 12*kB55**2 + 12*kA55*kD55))
            zp08=(-12*(kA55*mx + kB55*(qy+py))*x)/(b*(b**2*kA55*kA55 - 12*kB55**2 + 12*kA55*kD55))
            zp09=(x**2*(3*A11*my - 3*B11*(qx+px) + A11*(qz+pz)*x))/(6*b*(B11**2 - A11*D11))
            zp10=(2*A11*my*x - 2*B11*(qx+px)*x + A11*(qz+pz)*x**2)/(2*b*B11**2 - 2*A11*b*D11)
            zp11=(2*x**2*(-3*mz + (qy+py)*x))/(A11*b**3)
            zp12=(6*x*(-2*mz + (qy+py)*x))/(A11*b**3)

            zp = np.array([
                [zp01],
                [zp02],
                [zp03],
                [zp04],
                [zp05],
                [zp06],
                [zp07],
                [zp08],
                [zp09],
                [zp10],
                [zp11],
                [zp12]])
                
        return zp

    def z(self, x, C, l, phi, theta = 0, bed=True, load = True):
        """
        Computes the total solution vector at specified positions by combining the complementary and particular solutions.
        
        Args:
            x: Scalar or sequence of horizontal positions (mm) where the solution is evaluated.
            C: Array of constants for the complementary solution.
            l: Segment length (mm).
            phi: Slab inclination angle in degrees.
            theta: Slab rotation angle in degrees (default is 0).
            bed: If True, computes for a segment with foundation; if False, for a free segment.
            load: If True, includes external loads in the particular solution.
        
        Returns:
            The solution vector(s) at the specified position(s), with shape depending on the system (bedded or free segment).
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            z = np.concatenate([
                np.dot(self.zh(xi, l, bed), C)
                + self.zp(xi, phi,theta, bed,load) for xi in x], axis=1)
        else:
            z = np.dot(self.zh(x, l, bed), C) + self.zp(x, phi,theta, bed,load)

        return z
