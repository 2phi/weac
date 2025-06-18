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

    def __init__(self, system="pst-", touchdown=False):
        """
        Initialize eigensystem with user input.

        Parameters
        ----------
        system : {'pst-', '-pst', 'skier', 'skiers'}, optional
            Type of system to analyze:
            - 'pst-': PST cut from the right
            - '-pst': PST cut from the left
            - 'skier': One skier on infinite slab
            - 'skiers': Multiple skiers on infinite slab
            Default is 'pst-'.
        """
        # Assign global attributes
        self.g = 9810  # Gravitation (mm/s^2)
        self.lski = 1000  # Effective out-of-plane length of skis (mm)
        self.tol = 1e-3  # Relative Romberg integration tolerance
        self.system = system  # System type: 'pst-', '-pst', 'skier', 'skiers'

        # Initialize weak-layer attributes
        self.weak = False  # Weak-layer properties dictionary
        self.t = False  # Weak-layer thickness (mm)
        self.tc = False  # Weak-layer collapse height (mm)

        # Initialize slab attributes
        self.p = 0  # Surface line load (N/mm)
        self.slab = False  # Slab properties matrix
        self.k = False  # Slab shear correction factor
        self.h = False  # Total slab height (mm)
        self.b = False  # Total snowpack thickness (mm)
        self.zs = False  # Z-coordinate of slab center of gravity (mm)
        self.zA = False  # Z-coordinate of weights center of gravity (mm)
        self.yA = False  # Y-coordinate of weights center of gravity (mm)
        self.A11 = False  # Slab extensional stiffness (N/mm)
        self.B11 = False  # Slab bending-extension coupling stiffness (N)
        self.D11 = False  # Slab bending stiffness (Nmm)
        self.kA55 = False  # Slab shear stiffness (N/mm)
        self.kB55 = False  # Higher-order slab shear stiffness (N)
        self.kD55 = False  # Higher-order slab shear stiffness (Nmm)
        self.K0 = False  # Stiffness determinant (N^2)

        # Initialize eigensystem attributes
        self.ewC = False  # Complex eigenvalues
        self.ewR = False  # Real eigenvalues
        self.evC = False  # Complex eigenvectors
        self.evR = False  # Real eigenvectors
        self.sC = False  # Stability shift of complex eigenvalues (mm)
        self.sR = False  # Stability shift of real eigenvalues (mm)
        self.sysmat = False  # System matrix

        # Initialize touchdown attributes
        self.touchdown = touchdown
        self.lC = False  # Minimum length of substratum contact (mm)
        self.lS = (
            False  # Maximum length of span between bedded and touchdowned boundary (mm)
        )
        self.ratio = False  # Stiffness ratio of collapsed to uncollapsed weak-layer
        self.beta = False  # Ratio of slab to bedding stiffness

    def set_foundation_properties(
        self,
        t=30,
        E=0.25,
        nu=0.25,
        rhoweak=100,
        constitutive="plane strain",
        update=False,
    ):
        """
        Set material properties and geometry of foundation (weak layer).

        Parameters
        ----------
        t : float, optional
            Weak-layer thickness (mm). Default is 30.
        cf : float, optional
            Fraction by which the weak-layer thickness is reduced due to collapse.
            Default is 0.5.
        E : float, optional
            Weak-layer Young's modulus (MPa). Default is 0.25.
        nu : float, optional
            Weak-layer Poisson's ratio. Default is 0.25.
        rhoweak : float, optional
            Weak-layer density (kg/m^3). Default is 100.
        constitutive: string, optional
            Constitutive behavior of the weak layer in out-of-plane direction. Possible values are 'plane strain', 'plane stress' and 'uniaxial'
        update : bool, optional
            If True, recalculate the fundamental system after foundation properties
            have changed. Default is False.
        """
        # Geometry
        self.t = t  # Weak-layer thickness (mm)
        if constitutive == "plane strain":
            nuUpdate = nu
            EUpdate = E
        elif constitutive == "plane stress":
            nuUpdate = nu / (1 + nu)
            EUpdate = E * (1 + 2 * nu) / ((1 + nu) ** 2)
        elif constitutive == "uniaxial":
            nuUpdate = 0
            EUpdate = E

        # Material properties
        self.weak = {
            "nu": nuUpdate,  # Poisson's ratio (-)
            "E": EUpdate,  # Young's modulus (MPa)
            "rho": rhoweak * 1e-12,  # Density (t/mm^3)
        }

        # Recalculate the fundamental system after properties have changed
        if update:
            self.calc_fundamental_system()

    def set_beam_properties(
        self,
        layers,
        C0=6.5,
        C1=4.40,
        nu=0.25,
        b=290,
        k=5 / 6,
        update=False,
    ):
        """
        Set material and geometry properties of beam (slab).

        Parameters
        ----------
        layers : list or str
            2D list of top-to-bottom layer densities and thicknesses.
            Columns are density (kg/m^3) and thickness (mm).
            If entered as str, last split must be available in database.
        C0 : float, optional
            Multiplicative constant of Young's modulus parametrization
            according to Bergfeld et al. (2021). Default is 6.5.
        C1 : float, optional
            Exponent of Young's modulus parameterization according to
            Bergfeld et al. (2021). Default is 4.40.
        nu : float, optional
            Poisson's ratio. Default is 0.25.
        b : float, optional
            Total snowpack thickness (mm). Default is 290.
        k: float, optional
            Shear correction factor. Default is 5/6.
        update : bool, optional
            If True, recalculate the fundamental system after
            beam properties have changed. Default is False.
        """
        if isinstance(layers, str):
            # Read layering and Young's modulus from database
            layers, E = load_dummy_profile(layers.split()[-1])
        else:
            # Compute Young's modulus from density parametrization
            layers = np.array(layers)
            E = bergfeld(layers[:, 0], C0=C0, C1=C1)  # Young's modulus (MPa)

        # Derive other elastic properties
        nu = nu * np.ones(layers.shape[0])  # Global Poisson's ratio
        G = E / (2 * (1 + nu))  # Shear modulus (MPa)
        self.k = k  # Shear correction factor

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

    def set_surface_load(self, p, yA=0, zA=0):
        """
        Set surface line load.

        Define a distributed surface load p (N/mm) that acts
        in vertical (gravity) direction and the center of attack for optional additional weights.

        Arguments
        ---------
        p  : float
            Total weight (kg) of the additional loads
        yA : float, optional
            y-coordinate of the additional loads (mm)
        zA : float, optional
            z-coordinate of the additional loads (mm)

        """
        self.p = p  # Total line load from the additional weigths
        self.zA = zA  # z-coordiante of the additional weights
        self.yA = yA  # y-coordinate of the additional weigths

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
        return np.cumsum(t) - self.h / 2

    def calc_laminate_stiffness_matrix(self):
        """
        Provide ABD matrix.

        Return plane-strain laminate stiffness matrix (ABD matrix).
        """
        # Get ply coordinates (z-list is top to bottom, negative to positive)
        z = self.get_ply_coordinates()
        # Initialize stiffness components
        A11, B11, D11, kA55, kB55, kD55 = 0, 0, 0, 0, 0, 0
        # Add layerwise contributions
        for i in range(len(z) - 1):
            E, G, nu = self.slab[i, 2:5]
            A11 = A11 + E / (1 - nu**2) * (z[i + 1] - z[i])
            B11 = B11 + 1 / 2 * E / (1 - nu**2) * (z[i + 1] ** 2 - z[i] ** 2)
            D11 = D11 + 1 / 3 * E / (1 - nu**2) * (z[i + 1] ** 3 - z[i] ** 3)
            kA55 = kA55 + self.k * G * (z[i + 1] - z[i])
            kB55 = kB55 + 1 / 2 * self.k * G * (z[i + 1] ** 2 - z[i] ** 2)
            kD55 = kD55 + 1 / 3 * self.k * G * (z[i + 1] ** 3 - z[i] ** 3)
        self.A11 = A11
        self.B11 = B11
        self.D11 = D11
        self.kA55 = kA55
        self.kB55 = kB55
        self.kD55 = kD55
        self.K0 = B11**2 - A11 * D11

    def calc_system_matrix(self):
        """
        Assemble first-order ODE system matrix.

        Using the solution vector z = [u, u', w, w', psi, psi',phiU, phiU', phiW, phiW']
        the ODE system is written in the form Az' + Bz = d
        and rearranged to z' = -(A ^ -1) B z + (A ^ -1) d = E z + F
        """
        Ew = self.weak["E"]
        nuw = self.weak["nu"]

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

        c0201 = (-3 * (2 * D11 - B11 * h) * Pi**2 * Ew * (-1 + 2 * nuw)) / (
            t
            * (
                -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c0206 = -(
            (
                4 * h * kA55 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 6 * D11 * Ew * (-8 + Pi**2 * (-1 + 4 * nuw))
                + 3
                * B11
                * (
                    h * Ew * (8 + Pi**2 - 4 * Pi**2 * nuw)
                    + 8 * kA55 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                )
            )
            / (
                8 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - 8 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 2 * A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - 24 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                + 24 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c0209 = (
            -1
            / 2
            * (
                -3
                * B11
                * Pi**2
                * (-1 + 2 * nuw)
                * (h**2 * Ew + 8 * kA55 * t * (1 + nuw))
                + 2
                * h
                * Ew
                * (
                    -2 * kA55 * (-6 + Pi**2) * t**2 * (-1 + nuw)
                    + 3 * D11 * Pi**2 * (-1 + 2 * nuw)
                )
            )
            / (
                t
                * (
                    -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                    - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                )
            )
        )

        c0213 = (3 * (2 * D11 - B11 * h) * Pi**3 * Ew * (-1 + 2 * nuw)) / (
            t
            * (
                -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c0222 = (6 * (2 * D11 - B11 * h) * Pi * Ew) / (
            4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
            - 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
            + A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
            - 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            + 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
        )

        c0403 = (
            3
            * Ew
            * (
                36 * Pi**2 * (2 * kD55 * Pi**2 - kB55 * (-8 + Pi**2) * t) * (1 + nuw)
                + b**2 * (Pi**2 * (-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**4 * (1 + nuw))
                - 3
                * h
                * ((48 - 14 * Pi**2 + Pi**4) * t**2 * Ew + 12 * kB55 * Pi**4 * (1 + nuw))
            )
        ) / (
            t
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c0407 = (
            -3
            * Ew
            * (
                -(
                    (-1 + 2 * nuw)
                    * (
                        24
                        * Pi**2
                        * t
                        * (-3 * kD55 * (8 + Pi**2) + 2 * kB55 * (-6 + Pi**2) * t)
                        * (1 + nuw)
                        + 3
                        * h**2
                        * (
                            (48 - 14 * Pi**2 + Pi**4) * t**2 * Ew
                            + 12 * kB55 * Pi**4 * (1 + nuw)
                        )
                        + 4
                        * h
                        * (
                            (-6 + Pi**2) ** 2 * t**3 * Ew
                            - 18 * kD55 * Pi**4 * (1 + nuw)
                            + 18 * kB55 * Pi**4 * t * (1 + nuw)
                        )
                    )
                )
                + b**2
                * (
                    -24 * kB55 * Pi**4 * (-1 + nuw**2)
                    + (8 + Pi**2)
                    * t
                    * (-1 + 2 * nuw)
                    * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                    + h
                    * (
                        Pi**2 * (-6 + Pi**2) * t * Ew
                        + 6 * kA55 * Pi**4 * (-1 + nuw + 2 * nuw**2)
                    )
                )
            )
        ) / (
            2
            * t
            * (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c0412 = (
            -1
            / 4
            * (
                72
                * Pi**2
                * (-1 + nuw + 2 * nuw**2)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
                + b**2
                * (
                    -144 * kA55**2 * Pi**4 * (1 + nuw) ** 2 * (-1 + 2 * nuw)
                    - 48 * kA55 * Pi**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw + 2 * nuw**2)
                    + Ew
                    * (
                        36 * kB55 * Pi**2 * (1 + nuw) * (8 + Pi**2 * (-1 + 4 * nuw))
                        + (-6 + Pi**2)
                        * t
                        * Ew
                        * (
                            -4 * (-6 + Pi**2) * t * (-1 + 2 * nuw)
                            + 3 * h * (8 + Pi**2 * (-1 + 4 * nuw))
                        )
                    )
                )
            )
            / (
                (-1 + 2 * nuw)
                * (
                    b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                    - 18
                    * Pi**2
                    * (1 + nuw)
                    * (
                        4 * h * kB55 * (-6 + Pi**2) * t * Ew
                        - 4 * kD55 * (-6 + Pi**2) * t * Ew
                        + 24 * kB55**2 * Pi**2 * (1 + nuw)
                        - kA55
                        * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                    )
                )
            )
        )

        c0416 = (
            6 * b * Pi * Ew * (h * (-6 + Pi**2) * t * Ew + 12 * kB55 * Pi**2 * (1 + nuw))
        ) / (
            (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c0417 = (
            -3
            * Pi
            * Ew
            * (
                b**2 * (Pi**2 * (-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**4 * (1 + nuw))
                - 12
                * (
                    h * (-6 + Pi**2) * t**2 * Ew
                    + 3 * h * kB55 * Pi**4 * (1 + nuw)
                    - 6 * Pi**2 * (kD55 * Pi**2 - 2 * kB55 * t) * (1 + nuw)
                )
            )
        ) / (
            t
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c0423 = (
            -6
            * Pi
            * Ew
            * (
                -72 * (h * kB55 - 2 * kD55) * Pi**2 * t * (-1 + nuw + 2 * nuw**2)
                + b**2
                * (
                    12 * kB55 * Pi**4 * (-1 + nuw**2)
                    + 12 * kA55 * Pi**2 * t * (-1 + nuw + 2 * nuw**2)
                    + (-6 + Pi**2)
                    * t
                    * Ew
                    * (h * Pi**2 * (-1 + nuw) + 2 * t * (-1 + 2 * nuw))
                )
            )
        ) / (
            b
            * t
            * (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c0602 = (3 * Ew * (8 + Pi**2 * (-1 + 4 * nuw))) / (
            2 * (-1 + 2 * nuw) * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        )

        c0605 = (6 * Pi**2 * Ew * (-1 + nuw)) / (
            t * (-1 + 2 * nuw) * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        )

        c0610 = (
            -3
            * (
                h * Ew * (-8 + Pi**2 * (1 - 4 * nuw))
                + 8 * kA55 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        ) / (4 * (-1 + 2 * nuw) * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)))

        c0614 = (6 * Pi * Ew) / (
            (-1 + 2 * nuw) * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        )

        c0619 = (24 * Pi * Ew * nuw) / (
            b * (-1 + 2 * nuw) * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        )

        c0621 = (-6 * Pi**3 * Ew * (-1 + nuw)) / (
            t * (-1 + 2 * nuw) * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        )

        c0803 = (
            -18
            * Ew
            * (
                6 * h * kA55 * Pi**4 * (1 + nuw)
                - 12 * kB55 * Pi**4 * (1 + nuw)
                + (-8 + Pi**2)
                * t
                * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            )
        ) / (
            t
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c0807 = (
            3
            * Ew
            * (
                2
                * b**2
                * Pi**2
                * (-1 + nuw)
                * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                + (-1 + 2 * nuw)
                * (
                    18 * h**2 * kA55 * Pi**4 * (1 + nuw)
                    + 4
                    * t
                    * (
                        -9 * kB55 * Pi**2 * (8 + Pi**2) * (1 + nuw)
                        + (-6 + Pi**2)
                        * t
                        * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                    )
                    + h
                    * (
                        -36 * kB55 * Pi**4 * (1 + nuw)
                        + 3
                        * t
                        * (
                            (48 - 14 * Pi**2 + Pi**4) * t * Ew
                            + 12 * kA55 * Pi**4 * (1 + nuw)
                        )
                    )
                )
            )
        ) / (
            t
            * (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c0812 = (
            -3
            * b**2
            * Ew
            * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            * (8 + Pi**2 * (-1 + 4 * nuw))
        ) / (
            2
            * (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c0816 = (
            12 * b * Pi * Ew * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        ) / (
            (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c0817 = (
            36
            * Pi
            * Ew
            * (
                3 * h * kA55 * Pi**4 * (1 + nuw)
                - 6 * kB55 * Pi**4 * (1 + nuw)
                + 2 * t * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            )
        ) / (
            t
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c0823 = (
            -12
            * Pi**3
            * Ew
            * (
                -36 * (h * kA55 - 2 * kB55) * t * (-1 + nuw + 2 * nuw**2)
                + b**2
                * (-1 + nuw)
                * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            )
        ) / (
            b
            * t
            * (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c1001 = (3 * (2 * B11 - A11 * h) * Pi**2 * Ew * (-1 + 2 * nuw)) / (
            t
            * (
                -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1006 = -(
            (
                -8 * kA55 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 6 * B11 * Ew * (8 + Pi**2 - 4 * Pi**2 * nuw)
                - 3
                * A11
                * (
                    h * Ew * (8 + Pi**2 - 4 * Pi**2 * nuw)
                    + 8 * kA55 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                )
            )
            / (
                8 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - 8 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 2 * A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - 24 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                + 24 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1009 = (
            -1
            / 2
            * (
                6 * B11 * h * Pi**2 * Ew * (1 - 2 * nuw)
                + 8 * kA55 * (-6 + Pi**2) * t**2 * Ew * (-1 + nuw)
                + 3
                * A11
                * Pi**2
                * (-1 + 2 * nuw)
                * (h**2 * Ew + 8 * kA55 * t * (1 + nuw))
            )
            / (
                t
                * (
                    -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                    - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                )
            )
        )

        c1013 = (-3 * (2 * B11 - A11 * h) * Pi**3 * Ew * (-1 + 2 * nuw)) / (
            t
            * (
                -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1022 = (6 * (2 * B11 - A11 * h) * Pi * Ew) / (
            -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
            + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
            - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
            + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
        )

        c1204 = (
            -6 * (-1 + 2 * nuw) * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        ) / (
            b**2
            * (
                (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1208 = (
            3
            * (
                4 * h * (-6 + Pi**2) * t * Ew * (-1 + 2 * nuw)
                + 48 * kB55 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                + b**2 * Ew * (-8 + Pi**2 * (-1 + 4 * nuw))
            )
        ) / (
            4
            * b**2
            * (
                (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1211 = (
            3
            * (-1 + 2 * nuw)
            * (
                b**2 * Pi**2 * Ew
                + 4 * (-6 + Pi**2) * t**2 * Ew
                + 24 * kA55 * Pi**2 * t * (1 + nuw)
            )
        ) / (
            2
            * b**2
            * t
            * (
                (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1215 = (3 * Pi**3 * Ew * (-1 + 2 * nuw)) / (
            b
            * t
            * (
                (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1224 = (-6 * Pi * Ew) / (
            b
            * (
                (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1401 = (3 * (4 * D11 + h * (-4 * B11 + A11 * h)) * Pi * Ew * (-1 + 2 * nuw)) / (
            t
            * (
                -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1406 = (
            -1
            / 2
            * (
                Pi
                * (
                    48 * B11**2 * (-1 + nuw + 2 * nuw**2)
                    - 48 * A11 * D11 * (-1 + nuw + 2 * nuw**2)
                    + 4 * D11 * t * Ew * (7 - 19 * nuw + 12 * nuw**2)
                    + A11
                    * h
                    * t
                    * (-1 + nuw)
                    * (h * Ew * (-7 + 12 * nuw) - 24 * kA55 * (-1 + nuw + 2 * nuw**2))
                    + 4
                    * B11
                    * t
                    * (-1 + nuw)
                    * (h * Ew * (7 - 12 * nuw) + 12 * kA55 * (-1 + nuw + 2 * nuw**2))
                )
            )
            / (
                t
                * (-1 + nuw)
                * (
                    -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                    - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                )
            )
        )

        c1409 = (
            -3
            * Pi
            * (-1 + 2 * nuw)
            * (
                4 * B11 * (h**2 * Ew + 4 * kA55 * t * (1 + nuw))
                - h * (4 * D11 * Ew + A11 * h**2 * Ew + 8 * A11 * kA55 * t * (1 + nuw))
            )
        ) / (
            2
            * t
            * (
                -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1413 = (
            -1
            / 2
            * (
                Pi**4
                * (-1 + 2 * nuw)
                * (
                    4 * D11 * t * Ew * (-1 + nuw)
                    - 4 * B11 * h * t * Ew * (-1 + nuw)
                    + A11 * h**2 * t * Ew * (-1 + nuw)
                    - 12 * B11**2 * (-1 + nuw + 2 * nuw**2)
                    + 12 * A11 * D11 * (-1 + nuw + 2 * nuw**2)
                )
            )
            / (
                t**2
                * (-1 + nuw)
                * (
                    -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                    - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                )
            )
        )

        c1420 = -2 * nuw / (b - b * nuw)

        c1422 = (-6 * (4 * D11 + h * (-4 * B11 + A11 * h)) * Ew) / (
            4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
            - 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
            + A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
            - 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            + 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
        )

        c1604 = (
            -18
            * Pi
            * (1 + nuw)
            * (-1 + 2 * nuw)
            * (A11 + 2 * kA55 * (-1 + nuw) - 2 * A11 * nuw)
        ) / (
            b
            * (-1 + nuw)
            * (
                (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1608 = (
            -1
            / 4
            * (
                Pi
                * (
                    12
                    * A11
                    * (-1 + nuw + 2 * nuw**2)
                    * (b**2 + 3 * h * t * (-1 + 2 * nuw))
                    - t
                    * (-1 + nuw)
                    * (
                        b**2 * Ew * (-7 + 12 * nuw)
                        + 144 * kB55 * (-1 + nuw + 2 * nuw**2)
                    )
                )
            )
            / (
                b
                * t
                * (-1 + nuw)
                * (
                    (-6 + Pi**2) * t * Ew * (-1 + nuw)
                    + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                )
            )
        )

        c1611 = (
            -3
            * Pi
            * (-1 + 2 * nuw)
            * (
                12 * A11 * t * (-1 + nuw + 2 * nuw**2)
                - (-1 + nuw) * (b**2 * Ew + 24 * kA55 * t * (1 + nuw))
            )
        ) / (
            2
            * b
            * t
            * (-1 + nuw)
            * (
                (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1615 = (
            (-1 + 2 * nuw)
            * (
                t * (b**2 * Pi**4 + 12 * (-6 + Pi**2) * t**2) * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (b**2 * Pi**2 + 12 * t**2) * (-1 + nuw + 2 * nuw**2)
            )
        ) / (
            2
            * b**2
            * t**2
            * (-1 + nuw)
            * (
                (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        c1618 = -((3 - 6 * nuw) / (b * (-1 + nuw)))

        c1624 = (-6 * Ew) / (
            (-6 + Pi**2) * t * Ew * (-1 + nuw)
            + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
        )

        c1803 = (
            -6
            * Pi
            * Ew
            * (
                18
                * (
                    h**2 * kA55 * Pi**2
                    - 4 * h * kB55 * Pi**2
                    + 4 * kD55 * Pi**2
                    + h * kA55 * (-8 + Pi**2) * t
                    - 2 * kB55 * (-8 + Pi**2) * t
                )
                * (1 + nuw)
                + b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            )
        ) / (
            t
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c1807 = -(
            (
                18
                * Pi
                * (-1 + nuw + 2 * nuw**2)
                * (
                    -4 * kD55 * Pi**2 * (3 * h + 7 * t) * Ew
                    + 2
                    * kB55
                    * (
                        6 * h**2 * Pi**2
                        + h * (-24 + 17 * Pi**2) * t
                        + 4 * (-6 + Pi**2) * t**2
                    )
                    * Ew
                    + 96 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (
                        h
                        * (
                            3 * h**2 * Pi**2
                            + 2 * h * (-12 + 5 * Pi**2) * t
                            + 4 * (-6 + Pi**2) * t**2
                        )
                        * Ew
                        + 96 * kD55 * Pi**2 * (1 + nuw)
                    )
                )
                - b**2
                * Pi
                * (
                    144 * kA55**2 * Pi**2 * (1 + nuw) ** 2 * (-1 + 2 * nuw)
                    + 6
                    * kA55
                    * Ew
                    * (1 + nuw)
                    * (
                        (-24 + 11 * Pi**2) * t * (-1 + 2 * nuw)
                        + 3 * h * Pi**2 * (-3 + 4 * nuw)
                    )
                    + Ew
                    * (
                        (-6 + Pi**2) * t * (3 * h + 7 * t) * Ew * (-1 + 2 * nuw)
                        - 72 * kB55 * Pi**2 * (-1 + nuw**2)
                    )
                )
            )
            / (
                t
                * (-1 + 2 * nuw)
                * (
                    b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                    - 18
                    * Pi**2
                    * (1 + nuw)
                    * (
                        4 * h * kB55 * (-6 + Pi**2) * t * Ew
                        - 4 * kD55 * (-6 + Pi**2) * t * Ew
                        + 24 * kB55**2 * Pi**2 * (1 + nuw)
                        - kA55
                        * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                    )
                )
            )
        )

        c1812 = (
            -9
            * b**2
            * (h * kA55 - 2 * kB55)
            * Pi
            * Ew
            * (1 + nuw)
            * (8 + Pi**2 * (-1 + 4 * nuw))
        ) / (
            (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c1816 = -(
            (
                -36
                * Pi**2
                * (-1 + nuw + 2 * nuw**2)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
                + 2
                * b**2
                * (
                    36 * kA55**2 * Pi**4 * (1 + nuw) ** 2 * (-1 + 2 * nuw)
                    - 12
                    * kA55
                    * Pi**2
                    * Ew
                    * (1 + nuw)
                    * (3 * h - (-6 + Pi**2) * t * (-1 + 2 * nuw))
                    + Ew
                    * (
                        72 * kB55 * Pi**2 * (1 + nuw)
                        + (-6 + Pi**2) ** 2 * t**2 * Ew * (-1 + 2 * nuw)
                    )
                )
            )
            / (
                b
                * (-1 + 2 * nuw)
                * (
                    b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                    - 18
                    * Pi**2
                    * (1 + nuw)
                    * (
                        4 * h * kB55 * (-6 + Pi**2) * t * Ew
                        - 4 * kD55 * (-6 + Pi**2) * t * Ew
                        + 24 * kB55**2 * Pi**2 * (1 + nuw)
                        - kA55
                        * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                    )
                )
            )
        )

        c1817 = -(
            (
                -(
                    b**2
                    * Pi**4
                    * (t * Ew + 6 * kA55 * (1 + nuw))
                    * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                )
                + 18
                * Pi**2
                * (1 + nuw)
                * (
                    -4 * kD55 * Pi**4 * t * Ew
                    + 4 * kB55 * t * (h * Pi**4 + 12 * t) * Ew
                    + 24 * kB55**2 * Pi**4 * (1 + nuw)
                    - kA55
                    * (h * t * (h * Pi**4 + 24 * t) * Ew + 24 * kD55 * Pi**4 * (1 + nuw))
                )
            )
            / (
                t**2
                * (
                    b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                    - 18
                    * Pi**2
                    * (1 + nuw)
                    * (
                        4 * h * kB55 * (-6 + Pi**2) * t * Ew
                        - 4 * kD55 * (-6 + Pi**2) * t * Ew
                        + 24 * kB55**2 * Pi**2 * (1 + nuw)
                        - kA55
                        * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                    )
                )
            )
        )

        c1823 = (
            24
            * Ew
            * (
                18
                * (h**2 * kA55 - 4 * h * kB55 + 4 * kD55)
                * Pi**2
                * t
                * (-1 + nuw + 2 * nuw**2)
                + b**2
                * (
                    -3 * h * kA55 * Pi**4 * (-1 + nuw**2)
                    + 6 * kB55 * Pi**4 * (-1 + nuw**2)
                    + t
                    * (-1 + 2 * nuw)
                    * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                )
            )
        ) / (
            b
            * t
            * (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c2002 = (24 * nuw) / (b * Pi - 2 * b * Pi * nuw)

        c2005 = (-48 * nuw) / (b * Pi * t - 2 * b * Pi * t * nuw)

        c2010 = (12 * h * nuw) / (b * Pi - 2 * b * Pi * nuw)

        c2014 = (12 * nuw) / (b - 2 * b * nuw)

        c2019 = Pi**2 / t**2 + (24 * (-1 + nuw)) / (b**2 * (-1 + 2 * nuw))

        c2202 = -(
            (Pi * (24 * kA55 * (1 + nuw) + t * Ew * (1 + 12 * nuw)))
            / (
                t
                * (-1 + 2 * nuw)
                * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            )
        )

        c2205 = (-12 * Pi * Ew * (-1 + nuw)) / (
            t * (-1 + 2 * nuw) * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        )

        c2210 = (
            -1
            / 2
            * (
                Pi
                * (
                    24 * h * kA55 * (1 + nuw)
                    + h * t * Ew * (1 + 12 * nuw)
                    - 24 * kA55 * t * (-1 + nuw + 2 * nuw**2)
                )
            )
            / (
                t
                * (-1 + 2 * nuw)
                * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            )
        )

        c2214 = (-12 * Ew) / (
            (-1 + 2 * nuw) * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        )

        c2219 = (-48 * Ew * nuw) / (
            b * (-1 + 2 * nuw) * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        )

        c2221 = (2 * Pi**4 * (-1 + nuw) * (t * Ew + 6 * kA55 * (1 + nuw))) / (
            t**2
            * (-1 + 2 * nuw)
            * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        )

        c2403 = (
            -6
            * Pi
            * (
                b**2
                * (
                    -6 * kA55 * (3 * h * Pi**2 + (24 - 5 * Pi**2) * t) * Ew * (1 + nuw)
                    + 144 * kA55**2 * Pi**2 * (1 + nuw) ** 2
                    + Ew * ((-6 + Pi**2) * t**2 * Ew + 36 * kB55 * Pi**2 * (1 + nuw))
                )
                - 72
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        ) / (
            b
            * t
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c2407 = (
            -3
            * Pi
            * (
                2
                * b**4
                * Ew
                * (-1 + nuw)
                * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            )
            - 72
            * (h + t)
            * (-1 + nuw + 2 * nuw**2)
            * (
                h**2 * kA55 * (-6 + Pi**2) * t * Ew
                - 4 * h * kB55 * (-6 + Pi**2) * t * Ew
                + 4 * kD55 * (-6 + Pi**2) * t * Ew
                - 24 * kB55**2 * Pi**2 * (1 + nuw)
                + 24 * kA55 * kD55 * Pi**2 * (1 + nuw)
            )
            - b**2
            * (-1 + 2 * nuw)
            * (
                -18 * h**2 * kA55 * Pi**2 * Ew * (1 + nuw)
                + 12
                * t
                * (1 + nuw)
                * (
                    3 * kB55 * (8 + Pi**2) * Ew
                    + 2 * kA55 * (-6 + Pi**2) * t * Ew
                    + 12 * kA55**2 * Pi**2 * (1 + nuw)
                )
                + h
                * (
                    12 * kA55 * (-24 + Pi**2) * t * Ew * (1 + nuw)
                    + 144 * kA55**2 * Pi**2 * (1 + nuw) ** 2
                    + Ew * ((-6 + Pi**2) * t**2 * Ew + 36 * kB55 * Pi**2 * (1 + nuw))
                )
            )
        ) / (
            b
            * t
            * (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c2412 = (
            -1
            / 2
            * (
                72
                * b
                * Pi
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
                - b**3
                * Pi
                * (
                    144 * kA55**2 * Pi**2 * (1 + nuw) ** 2
                    + (-6 + Pi**2) * t**2 * Ew**2 * (1 + 12 * nuw)
                    + 6 * kA55 * t * Ew * (1 + nuw) * (-24 + Pi**2 * (5 + 12 * nuw))
                )
            )
            / (
                t
                * (-1 + 2 * nuw)
                * (
                    b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                    - 18
                    * Pi**2
                    * (1 + nuw)
                    * (
                        4 * h * kB55 * (-6 + Pi**2) * t * Ew
                        - 4 * kD55 * (-6 + Pi**2) * t * Ew
                        + 24 * kB55**2 * Pi**2 * (1 + nuw)
                        - kA55
                        * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                    )
                )
            )
        )

        c2416 = (
            -12 * b**2 * Ew * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
        ) / (
            (-1 + 2 * nuw)
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c2417 = (
            -36
            * b
            * Ew
            * (
                3 * h * kA55 * Pi**4 * (1 + nuw)
                - 6 * kB55 * Pi**4 * (1 + nuw)
                + 2 * t * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            )
        ) / (
            t
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        c2423 = -(
            (
                -(
                    2
                    * b**4
                    * Pi**4
                    * (-1 + nuw)
                    * (t * Ew + 6 * kA55 * (1 + nuw))
                    * ((-6 + Pi**2) * t * Ew)
                    + 6 * kA55 * Pi**2 * (1 + nuw)
                )
                + 216
                * Pi**2
                * t**2
                * (-1 + nuw + 2 * nuw**2)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
                + 12
                * b**2
                * (
                    12
                    * kB55
                    * Pi**2
                    * t
                    * Ew
                    * (6 * t * (1 - 2 * nuw) + h * Pi**2 * (-6 + Pi**2) * (-1 + nuw))
                    * (1 + nuw)
                    + 72 * kB55**2 * Pi**6 * (-1 + nuw) * (1 + nuw) ** 2
                    - 36 * kA55**2 * Pi**4 * t**2 * (1 + nuw) ** 2 * (-1 + 2 * nuw)
                    - (-6 + Pi**2)
                    * t
                    * Ew
                    * (
                        (-6 + Pi**2) * t**3 * Ew * (-1 + 2 * nuw)
                        + 12 * kD55 * Pi**4 * (-1 + nuw**2)
                    )
                    - 3
                    * kA55
                    * Pi**2
                    * (1 + nuw)
                    * (
                        24 * kD55 * Pi**4 * (-1 + nuw**2)
                        + t
                        * Ew
                        * (
                            12 * h * t * (1 - 2 * nuw)
                            + h**2 * Pi**2 * (-6 + Pi**2) * (-1 + nuw)
                            + 4 * (-6 + Pi**2) * t**2 * (-1 + 2 * nuw)
                        )
                    )
                )
            )
            / (
                b**2
                * t**2
                * (-1 + 2 * nuw)
                * (
                    b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                    - 18
                    * Pi**2
                    * (1 + nuw)
                    * (
                        4 * h * kB55 * (-6 + Pi**2) * t * Ew
                        - 4 * kD55 * (-6 + Pi**2) * t * Ew
                        + 24 * kB55**2 * Pi**2 * (1 + nuw)
                        - kA55
                        * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                    )
                )
            )
        )

        SystemMatrixC = [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                c0201,
                0,
                0,
                0,
                0,
                c0206,
                0,
                0,
                c0209,
                0,
                0,
                0,
                c0213,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                c0222,
                0,
                0,
            ],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0,
                0,
                c0403,
                0,
                0,
                0,
                c0407,
                0,
                0,
                0,
                0,
                c0412,
                0,
                0,
                0,
                c0416,
                c0417,
                0,
                0,
                0,
                0,
                0,
                c0423,
                0,
            ],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0,
                c0602,
                0,
                0,
                c0605,
                0,
                0,
                0,
                0,
                c0610,
                0,
                0,
                0,
                c0614,
                0,
                0,
                0,
                0,
                c0619,
                0,
                c0621,
                0,
                0,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0,
                0,
                c0803,
                0,
                0,
                0,
                c0807,
                0,
                0,
                0,
                0,
                c0812,
                0,
                0,
                0,
                c0816,
                c0817,
                0,
                0,
                0,
                0,
                0,
                c0823,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                c1001,
                0,
                0,
                0,
                0,
                c1006,
                0,
                0,
                c1009,
                0,
                0,
                0,
                c1013,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                c1022,
                0,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0,
                0,
                0,
                c1204,
                0,
                0,
                0,
                c1208,
                0,
                0,
                c1211,
                0,
                0,
                0,
                c1215,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                c1224,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                c1401,
                0,
                0,
                0,
                0,
                c1406,
                0,
                0,
                c1409,
                0,
                0,
                0,
                c1413,
                0,
                0,
                0,
                0,
                0,
                0,
                c1420,
                0,
                c1422,
                0,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0,
                0,
                0,
                c1604,
                0,
                0,
                0,
                c1608,
                0,
                0,
                c1611,
                0,
                0,
                0,
                c1615,
                0,
                0,
                c1618,
                0,
                0,
                0,
                0,
                0,
                c1624,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [
                0,
                0,
                c1803,
                0,
                0,
                0,
                c1807,
                0,
                0,
                0,
                0,
                c1812,
                0,
                0,
                0,
                c1816,
                c1817,
                0,
                0,
                0,
                0,
                0,
                c1823,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [
                0,
                c2002,
                0,
                0,
                c2005,
                0,
                0,
                0,
                0,
                c2010,
                0,
                0,
                0,
                c2014,
                0,
                0,
                0,
                0,
                c2019,
                0,
                0,
                0,
                0,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [
                0,
                c2202,
                0,
                0,
                c2205,
                0,
                0,
                0,
                0,
                c2210,
                0,
                0,
                0,
                c2214,
                0,
                0,
                0,
                0,
                c2219,
                0,
                c2221,
                0,
                0,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [
                0,
                0,
                c2403,
                0,
                0,
                0,
                c2407,
                0,
                0,
                0,
                0,
                c2412,
                0,
                0,
                0,
                c2416,
                c2417,
                0,
                0,
                0,
                0,
                0,
                c2423,
                0,
            ],
        ]

        self.sysmat = np.array(SystemMatrixC)
        return SystemMatrixC

    def calc_eigensystem(self):
        """Calculate eigenvalues and eigenvectors of the system matrix."""
        # Calculate eigenvalues (ew) and eigenvectors (ev)
        ew, ev = np.linalg.eig(self.sysmat)
        # Classify real and complex eigenvalues
        real = (ew.imag == 0) & (ew.real != 0)  # real eigenvalues
        cmplx = ew.imag > 0  # positive complex conjugates
        # Eigenvalues
        self.ewC = ew[cmplx]
        self.ewR = ew[real].real
        # Eigenvectors
        self.evC = np.around(ev[:, cmplx], 15)
        self.evR = np.around(ev[:, real].real, 15)
        # Prepare positive eigenvalue shifts for numerical robustness
        self.sR, self.sC = np.zeros(self.ewR.shape), np.zeros(self.ewC.shape)
        self.sR[self.ewR > 0], self.sC[self.ewC > 0] = -1, -1

    def calc_fundamental_system(self):
        """Calculate the fundamental system of the problem."""
        self.calc_laminate_stiffness_matrix()
        self.calc_system_matrix()
        self.calc_eigensystem()

    def get_weight_load(self, phi, theta=0):
        """
        Calculate line loads from slab mass.

        Arguments
        ---------
        phi : float
            Inclination (degrees). Counterclockwise positive.
        theta : float
            Rotation (degrees).

        Returns
        -------
        qx : float
            Line load (N/mm) at center of gravity in axial direction.
        qy : float
            Line load (N/mm) at center of gravity in out-of-plane direction.
        qz : float
            Line load (N/mm) at center of gravity in normal direction.
        """
        # Convert units
        phi = np.deg2rad(phi)  # Convert inclination to rad
        theta = np.deg2rad(theta)  # Convert rotation to rad
        rho = self.slab[:, 0] * 1e-12  # Convert density to t/mm^3
        # Sum up layer weight loads
        q = sum(rho * self.b * self.g * self.slab[:, 1])  # Line load (N/mm)
        # Split into components
        qz = q * np.cos(phi) * np.cos(theta)  # normal direction
        qx = -q * np.sin(phi)  # axial  direction
        qy = q * np.sin(theta)  # out-of-plane direction

        return qx, qy, qz

    def get_weak_layer_load(self, phi, theta=0):
        """
        Calculate surface line loads.

        Arguments
        ---------
        phi : float
            Inclination (degrees). Counterclockwise positive.
        theta : float
            Rotation (degrees).

        Returns
        -------
        fx : float
            Surface line load (N/mm³) in axial direction.
        fy : float
            Surface line load (N/mm³) in out-of-plane direction.
        fz : float
            Surface line load (N/mm³) in normal direction.
        """
        # Convert units
        phi = np.deg2rad(phi)  # Convert inclination to rad
        theta = np.deg2rad(theta)
        f = self.weak["rho"] * self.g
        # Split into components
        fz = f * np.cos(phi) * np.cos(theta)  # Normal direction
        fx = -f * np.sin(phi)  # Tangential direction
        fy = f * np.sin(theta)  # Out-of-plane direction

        return fx, fy, fz

    def get_surface_load(self, phi, theta=0):
        """
        Calculate surface line loads.

        Arguments
        ---------
        phi : float
            Inclination (degrees). Counterclockwise positive.
        theta : float
            Rotation (degrees).

        Returns
        -------
        px : float
            Surface line load (N/mm) in axial direction.
        py : float
            Surface line load (N/mm) in out-of-plane direction.
        pz : float
            Surface line load (N/mm) in normal direction.
        """
        # Convert units
        phi = np.deg2rad(phi)  # Convert inclination to rad
        theta = np.deg2rad(theta)
        # Split into components
        pz = self.p * np.cos(phi) * np.cos(theta)  # Normal direction
        px = -self.p * np.sin(phi)  # Tangential direction
        py = self.p * np.sin(theta)  # Out-of-plane direction

        return px, py, pz

    def get_skier_load(self, m, phi, theta=0):
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
        phi = np.deg2rad(phi)  # Convert inclination to rad
        theta = np.deg2rad(theta)  # Convert rotation to rad
        F = 1e-3 * np.array(m) * self.g / self.lski  # Total skier load (N)
        Fz = F * np.cos(phi) * np.cos(theta)  # Normal skier load (N)
        Fx = -F * np.sin(phi)  # Tangential skier load (N)
        Fy = F * np.sin(theta)  # Out-of-plane skier load  (N)

        return Fx, Fy, Fz

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

        A11 = self.A11
        B11 = self.B11
        D11 = self.D11
        kA55 = self.kA55
        kB55 = self.kB55
        b = np.longdouble(self.b)

        if bed:
            zh = np.around(
                np.concatenate(
                    [
                        # Real
                        self.evR * np.exp(self.ewR * (x + l * self.sR)),
                        # Complex
                        np.exp(self.ewC.real * (x + l * self.sC))
                        * (
                            self.evC.real * np.cos(self.ewC.imag * x)
                            - self.evC.imag * np.sin(self.ewC.imag * x)
                        ),
                        # Complex
                        np.exp(self.ewC.real * (x + l * self.sC))
                        * (
                            self.evC.imag * np.cos(self.ewC.imag * x)
                            + self.evC.real * np.sin(self.ewC.imag * x)
                        ),
                    ],
                    axis=1,
                ),
                15,
            )
        else:
            # Abbreviations for the unbedded segemnts in accordance with Mathematica script
            H0101 = 1
            H0102 = x
            H0104 = (B11 * kA55 * x**2) / (2 * B11**2 - 2 * A11 * D11)
            H0105 = (B11 * kA55 * x**2) / (2 * B11**2 - 2 * A11 * D11)

            H0202 = 1
            H0204 = (B11 * kA55 * x) / (B11**2 - A11 * D11)
            H0205 = (B11 * kA55 * x) / (B11**2 - A11 * D11)

            H0307 = 1
            H0308 = x - (2 * kA55 * x**3) / (A11 * b**2)
            H0310 = (2 * kB55 * x**3) / (A11 * b**2)
            H0311 = (2 * kA55 * x**3) / (A11 * b**2)
            H0312 = x**2 / 2

            H0408 = 1 - (6 * kA55 * x**2) / (A11 * b**2)
            H0410 = (6 * kB55 * x**2) / (A11 * b**2)
            H0411 = (6 * kA55 * x**2) / (A11 * b**2)
            H0412 = x

            H0503 = 1
            H0504 = (6 * B11**2 * x - 6 * A11 * D11 * x + A11 * kA55 * x**3) / (
                6 * B11**2 - 6 * A11 * D11
            )
            H0505 = (A11 * kA55 * x**3) / (6 * B11**2 - 6 * A11 * D11)
            H0506 = -1 / 2 * x**2

            H0604 = (2 * B11**2 - 2 * A11 * D11 + A11 * kA55 * x**2) / (
                2 * B11**2 - 2 * A11 * D11
            )
            H0605 = (A11 * kA55 * x**2) / (2 * B11**2 - 2 * A11 * D11)
            H0606 = -x

            H0709 = 1
            H0710 = x

            H0810 = 1

            H0904 = (A11 * kA55 * x**2) / (-2 * B11**2 + 2 * A11 * D11)
            H0905 = (-2 * B11**2 + 2 * A11 * D11 + A11 * kA55 * x**2) / (
                -2 * B11**2 + 2 * A11 * D11
            )
            H0906 = x

            H1004 = (A11 * kA55 * x) / (-(B11**2) + A11 * D11)
            H1005 = (A11 * kA55 * x) / (-(B11**2) + A11 * D11)
            H1006 = 1

            H1108 = (-6 * kA55 * x**2) / (A11 * b**2)
            H1110 = (6 * kB55 * x**2) / (A11 * b**2)
            H1111 = 1 + (6 * kA55 * x**2) / (A11 * b**2)
            H1112 = x

            H1208 = (-12 * kA55 * x) / (A11 * b**2)
            H1210 = (12 * kB55 * x) / (A11 * b**2)
            H1211 = (12 * kA55 * x) / (A11 * b**2)
            H1212 = 1

            # Complementary solution matrix of free segments
            zh = np.array(
                [
                    [H0101, H0102, 0, H0104, H0105, 0, 0, 0, 0, 0, 0, 0],
                    [0, H0202, 0, H0204, H0205, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, H0307, H0308, 0, H0310, H0311, H0312],
                    [0, 0, 0, 0, 0, 0, 0, H0408, 0, H0410, H0411, H0412],
                    [0, 0, H0503, H0504, H0505, H0506, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, H0604, H0605, H0606, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, H0709, H0710, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, H0810, 0, 0],
                    [0, 0, 0, H0904, H0905, H0906, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, H1004, H1005, H1006, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, H1108, 0, H1110, H1111, H1112],
                    [0, 0, 0, 0, 0, 0, 0, H1208, 0, H1210, H1211, H1212],
                ]
            )

        return zh

    def zp(self, x, phi, theta=0, bed=True, load=False):
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
            Particular integral vector (24x1) at position x.
        """
        # Get weight and surface loads
        qx, qy, qz = self.get_weight_load(phi, theta)

        px, py, pz = self.get_surface_load(phi, theta)

        fx, fy, fz = self.get_weak_layer_load(phi, theta)
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
        Ew = self.weak["E"]
        nuw = self.weak["nu"]

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
            zp01 = (
                h * (2 * my + h * (px + qx)) * Ew
                + 4 * kA55 * t * (2 * px + 2 * qx + b * fx * t) * (1 + nuw)
            ) / (4 * b * kA55 * Ew)

            zp02 = 0

            zp03 = (
                t
                * (
                    (
                        -3
                        * Pi**4
                        * (h + t)
                        * (4 * mx + b * fy * h * t)
                        * (1 + nuw)
                        * (-1 + 2 * nuw)
                    )
                    / b
                    - 96 * fy * t**2 * (h + t) * (-1 + nuw + 2 * nuw**2)
                    + (
                        Pi**2
                        * (2 * py + 2 * qy + b * fy * t)
                        * (1 + nuw)
                        * (
                            2 * b**4 * Pi**6 * (-1 + nuw) ** 2
                            + 6
                            * (
                                3 * h**2 * Pi**2 * (-8 + Pi**2)
                                + 6 * h * Pi**2 * (-8 + Pi**2) * t
                                + 4 * (-24 - 6 * Pi**2 + Pi**4) * t**2
                            )
                            * (t - 2 * t * nuw) ** 2
                            + b**2
                            * Pi**2
                            * (
                                3 * h**2 * Pi**4
                                + 6 * h * Pi**4 * t
                                + 4 * (-24 + 3 * Pi**2 + Pi**4) * t**2
                            )
                            * (1 - 3 * nuw + 2 * nuw**2)
                        )
                    )
                    / (
                        b**3 * Pi**4 * (-1 + nuw)
                        + 6 * b * (-8 + Pi**2) * t**2 * (-1 + 2 * nuw)
                    )
                )
            ) / (
                Ew
                * (2 * b**2 * Pi**4 * (-1 + nuw) + (-96 + Pi**4) * t**2 * (-1 + 2 * nuw))
            )

            zp04 = 0

            zp05 = (
                Pi**2
                * t
                * (2 * pz + 2 * qz + b * fz * t)
                * (1 + nuw)
                * (-1 + 2 * nuw)
                * (24 * t**2 * (-1 + nuw) + b**2 * Pi**2 * (-1 + 2 * nuw))
            ) / (
                2
                * b
                * Ew
                * (
                    24 * t**2 * (Pi**2 * (-1 + nuw) ** 2 - 8 * nuw**2)
                    + b**2 * Pi**4 * (1 - 3 * nuw + 2 * nuw**2)
                )
            )

            zp06 = 0

            zp07 = (
                6
                * t
                * (
                    -4 * mx * Pi**4
                    + 2 * h * Pi**4 * (py + qy)
                    + t * (-32 * b * fy * t + Pi**4 * (2 * py + 2 * qy + b * fy * t))
                )
                * (-1 + nuw + 2 * nuw**2)
            ) / (
                b
                * Ew
                * (2 * b**2 * Pi**4 * (-1 + nuw) + (-96 + Pi**4) * t**2 * (-1 + 2 * nuw))
            )

            zp08 = 0

            zp09 = -1 / 2 * (2 * my + h * (px + qx)) / (b * kA55)

            zp10 = 0

            zp11 = (-24 * mz * Pi**2 * t * (b**2 * Pi**2 + 12 * t**2) * (1 + nuw)) / (
                b**5 * Pi**4 * Ew
                + 48 * b * t**3 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                + 4
                * b**3
                * Pi**2
                * t
                * ((3 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            )

            zp12 = 0

            zp13 = (8 * fx * t**2 * (1 + nuw)) / (Pi**3 * Ew)

            zp14 = 0

            zp15 = (-288 * mz * Pi * t**3 * (1 + nuw)) / (
                b**4 * Pi**4 * Ew
                + 48 * t**3 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                + 4
                * b**2
                * Pi**2
                * t
                * ((3 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
            )

            zp16 = 0

            zp17 = (
                16
                * Pi
                * t**2
                * (1 + nuw)
                * (
                    b * fy * t**2 * (1 - 2 * nuw)
                    + b**3 * fy * (-1 + nuw)
                    + 3 * (2 * mx - (py + qy) * (h + t)) * (-1 + 2 * nuw)
                )
            ) / (
                b
                * Ew
                * (2 * b**2 * Pi**4 * (-1 + nuw) + (-96 + Pi**4) * t**2 * (-1 + 2 * nuw))
            )

            zp18 = 0

            zp19 = (
                -24
                * Pi
                * t**2
                * (2 * pz + 2 * qz + b * fz * t)
                * nuw
                * (1 + nuw)
                * (-1 + 2 * nuw)
            ) / (
                Ew
                * (
                    24 * t**2 * (Pi**2 * (-1 + nuw) ** 2 - 8 * nuw**2)
                    + b**2 * Pi**4 * (1 - 3 * nuw + 2 * nuw**2)
                )
            )

            zp20 = 0

            zp21 = (4 * fz * t**2 * (1 + nuw) * (-1 + 2 * nuw)) / (
                Pi**3 * Ew * (-1 + nuw)
            )

            zp22 = 0

            zp23 = (
                12
                * Pi
                * t**2
                * (2 * py + 2 * qy + b * fy * t)
                * (1 + nuw)
                * (-1 + 2 * nuw)
            ) / (
                Ew
                * (b**2 * Pi**4 * (-1 + nuw) + 6 * (-8 + Pi**2) * t**2 * (-1 + 2 * nuw))
            )

            zp24 = 0

            zp = np.array(
                [
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
                    [zp12],
                    [zp13],
                    [zp14],
                    [zp15],
                    [zp16],
                    [zp17],
                    [zp18],
                    [zp19],
                    [zp20],
                    [zp21],
                    [zp22],
                    [zp23],
                    [zp24],
                ],
                dtype=np.double,
            )

        else:
            zp01 = (
                -1
                / 6
                * (x**2 * (3 * B11 * my - 3 * D11 * (qx + px) + B11 * (qz + pz) * x))
                / (b * (B11**2 - A11 * D11))
            )
            zp02 = (
                -1
                / 2
                * (x * (2 * B11 * my - 2 * D11 * (qx + px) + B11 * (qz + pz) * x))
                / (b * (B11**2 - A11 * D11))
            )
            zp03 = (
                x**2
                * (
                    -(
                        (
                            b**2
                            * (
                                12 * kB55 * mx
                                + b**2 * kA55 * (qy + py)
                                + 12 * kD55 * (qy + py)
                            )
                        )
                        / (b**2 * kA55 * kA55 - 12 * kB55**2 + 12 * kA55 * kD55)
                    )
                    + (x * (-4 * mz + (qy + py) * x)) / A11
                )
            ) / (2 * b**3)
            zp04 = (
                x
                * (
                    -(
                        (
                            b**2
                            * (
                                12 * kB55 * mx
                                + b**2 * kA55 * (qy + py)
                                + 12 * kD55 * (py + qy)
                            )
                        )
                        / (b**2 * kA55 * kA55 - 12 * kB55**2 + 12 * kA55 * kD55)
                    )
                    + (2 * x * (-3 * mz + (qy + py) * x)) / A11
                )
            ) / b**3
            zp05 = (
                -1
                / 24
                * (
                    x**2
                    * (
                        12 * B11**2 * (qz + pz)
                        - 12 * A11 * D11 * (qz + pz)
                        - 4 * B11 * kA55 * (qx + px) * x
                        + A11 * kA55 * x * (4 * my + (qz + pz) * x)
                    )
                )
                / (b * (B11**2 - A11 * D11) * kA55)
            )
            zp06 = (
                -1
                / 6
                * (
                    x
                    * (
                        6 * B11**2 * (qz + pz)
                        - 6 * A11 * D11 * (qz + pz)
                        - 3 * B11 * kA55 * (qx + px) * x
                        + A11 * kA55 * x * (3 * my + (qz + pz) * x)
                    )
                )
                / (b * (B11**2 - A11 * D11) * kA55)
            )
            zp07 = (-6 * (kA55 * mx + kB55 * (qy + py)) * x**2) / (
                b * (b**2 * kA55 * kA55 - 12 * kB55**2 + 12 * kA55 * kD55)
            )
            zp08 = (-12 * (kA55 * mx + kB55 * (qy + py)) * x) / (
                b * (b**2 * kA55 * kA55 - 12 * kB55**2 + 12 * kA55 * kD55)
            )
            zp09 = (
                x**2 * (3 * A11 * my - 3 * B11 * (qx + px) + A11 * (qz + pz) * x)
            ) / (6 * b * (B11**2 - A11 * D11))
            zp10 = (
                2 * A11 * my * x - 2 * B11 * (qx + px) * x + A11 * (qz + pz) * x**2
            ) / (2 * b * B11**2 - 2 * A11 * b * D11)
            zp11 = (2 * x**2 * (-3 * mz + (qy + py) * x)) / (A11 * b**3)
            zp12 = (6 * x * (-2 * mz + (qy + py) * x)) / (A11 * b**3)

            zp = np.array(
                [
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
                    [zp12],
                ]
            )

        return zp

    def z(self, x, C, l, phi, theta=0, bed=True, load=True):
        """
        Assemble solution vector at positions x.

        Arguments
        ---------
        x : float or sequence
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
            z = np.concatenate(
                [
                    np.dot(self.zh(xi, l, bed), C) + self.zp(xi, phi, theta, bed, load)
                    for xi in x
                ],
                axis=1,
            )
        else:
            z = np.dot(self.zh(x, l, bed), C) + self.zp(x, phi, theta, bed, load)

        return z

    def get_load_vector(self, phi, theta):
        """
        Compute vector d for the identification of z' = Kz + d

        Arguments
        ---------
        phi : float
            Inclination (degrees).
        theta : float
            Rotation (degrees)

        Returns
        -------
        d : ndarray
            Particular integral vector (24x1) at position x.
        """
        # Get weight and surface loads
        qx, qy, qz = self.get_weight_load(phi, theta)

        px, py, pz = self.get_surface_load(phi, theta)

        fx, fy, fz = self.get_weak_layer_load(phi, theta)
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
        Ew = self.weak["E"]
        nuw = self.weak["nu"]

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

        my = -qx * zS - px * zA
        mz = -px * yA
        mx = -qy * zS + pz * yA - py * zA

        q01 = 0

        q02 = (
            h * (-6 + Pi**2) * (2 * my + h * (px + qx)) * t * Ew * (-1 + nuw)
            + 3
            * B11
            * (4 * my * Pi**2 - b * fx * h * (-8 + Pi**2) * t)
            * (-1 + nuw + 2 * nuw**2)
            + 6
            * D11
            * (-8 * b * fx * t + Pi**2 * (2 * px + 2 * qx + b * fx * t))
            * (-1 + nuw + 2 * nuw**2)
        ) / (
            b
            * (
                -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        q03 = 0

        q04 = (
            -3
            * (1 + nuw)
            * (
                72 * b * fy * kD55 * Pi**2 * (-8 + Pi**2) * t * (1 + nuw)
                - 36
                * kB55
                * Pi**2
                * (4 * mx * Pi**2 + b * fy * h * (-8 + Pi**2) * t)
                * (1 + nuw)
                + 2
                * b**2
                * Pi**2
                * (py + qy)
                * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                + b**3
                * fy
                * (-8 + Pi**2)
                * t
                * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                + 6
                * Pi**2
                * (
                    h * (-6 + Pi**2) * (-2 * mx + h * (py + qy)) * t * Ew
                    + 24 * kD55 * Pi**2 * (py + qy) * (1 + nuw)
                )
            )
        ) / (
            b**3 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
            - 18
            * b
            * Pi**2
            * (1 + nuw)
            * (
                4 * h * kB55 * (-6 + Pi**2) * t * Ew
                - 4 * kD55 * (-6 + Pi**2) * t * Ew
                + 24 * kB55**2 * Pi**2 * (1 + nuw)
                - kA55 * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
            )
        )

        q05 = 0

        q06 = (
            -3 * (-8 * b * fz * t + Pi**2 * (2 * pz + 2 * qz + b * fz * t)) * (1 + nuw)
        ) / (b * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)))

        q07 = 0

        q08 = (
            36
            * Pi**2
            * (1 + nuw)
            * (
                (-6 + Pi**2) * (2 * mx - h * (py + qy)) * t * Ew
                + 3 * kA55 * (4 * mx * Pi**2 + b * fy * h * (-8 + Pi**2) * t) * (1 + nuw)
                - 6
                * kB55
                * (-8 * b * fy * t + Pi**2 * (2 * py + 2 * qy + b * fy * t))
                * (1 + nuw)
            )
        ) / (
            b**3 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
            - 18
            * b
            * Pi**2
            * (1 + nuw)
            * (
                4 * h * kB55 * (-6 + Pi**2) * t * Ew
                - 4 * kD55 * (-6 + Pi**2) * t * Ew
                + 24 * kB55**2 * Pi**2 * (1 + nuw)
                - kA55 * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
            )
        )

        q09 = 0

        q10 = (
            -3
            * A11
            * (4 * my * Pi**2 - b * fx * h * (-8 + Pi**2) * t)
            * (-1 + nuw + 2 * nuw**2)
            - 2
            * (
                (-6 + Pi**2) * (2 * my + h * (px + qx)) * t * Ew * (-1 + nuw)
                + 3
                * B11
                * (-8 * b * fx * t + Pi**2 * (2 * px + 2 * qx + b * fx * t))
                * (-1 + nuw + 2 * nuw**2)
            )
        ) / (
            b
            * (
                -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        q11 = 0

        q12 = (36 * mz * Pi**2 * (1 + nuw) * (-1 + 2 * nuw)) / (
            b**3
            * (
                (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        q13 = 0

        q14 = (
            Pi
            * (1 + nuw)
            * (-1 + 2 * nuw)
            * (
                12
                * (
                    -2 * B11 * my
                    + A11 * h * my
                    - 2 * D11 * (px + qx)
                    + B11 * h * (px + qx)
                )
                * Ew
                * (-1 + nuw)
                + b
                * fx
                * (
                    4 * D11 * t * Ew * (-1 + nuw)
                    - 4 * B11 * h * t * Ew * (-1 + nuw)
                    + A11 * h**2 * t * Ew * (-1 + nuw)
                    - 48 * B11**2 * (-1 + nuw + 2 * nuw**2)
                    + 48 * A11 * D11 * (-1 + nuw + 2 * nuw**2)
                )
            )
        ) / (
            b
            * Ew
            * (-1 + nuw)
            * (
                -4 * D11 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 4 * B11 * h * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                - A11 * h**2 * (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 12 * B11**2 * Pi**2 * (-1 + nuw + 2 * nuw**2)
                - 12 * A11 * D11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        q15 = 0

        q16 = (36 * mz * Pi * (1 + nuw) * (-1 + 2 * nuw)) / (
            b**2
            * (
                (-6 + Pi**2) * t * Ew * (-1 + nuw)
                + 3 * A11 * Pi**2 * (-1 + nuw + 2 * nuw**2)
            )
        )

        q17 = 0

        q18 = (
            -2
            * Pi
            * (1 + nuw)
            * (
                216
                * Pi**2
                * (
                    -(h * kA55 * mx)
                    + 2 * kB55 * mx
                    + h * kB55 * (py + qy)
                    - 2 * kD55 * (py + qy)
                )
                * Ew
                * (1 + nuw)
                - 6
                * b**2
                * (py + qy)
                * Ew
                * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                + b**3
                * fy
                * (t * Ew + 24 * kA55 * (1 + nuw))
                * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw))
                - 18
                * b
                * fy
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * t * Ew
                    - 4 * kD55 * t * Ew
                    + 96 * kB55**2 * (1 + nuw)
                    - kA55 * (h**2 * t * Ew + 96 * kD55 * (1 + nuw))
                )
            )
        ) / (
            b
            * Ew
            * (
                b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
                - 18
                * Pi**2
                * (1 + nuw)
                * (
                    4 * h * kB55 * (-6 + Pi**2) * t * Ew
                    - 4 * kD55 * (-6 + Pi**2) * t * Ew
                    + 24 * kB55**2 * Pi**2 * (1 + nuw)
                    - kA55
                    * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
                )
            )
        )

        q19 = 0

        q20 = 0

        q21 = 0

        q22 = (
            -2
            * Pi
            * (1 + nuw)
            * (-6 * (pz + qz) * Ew + b * fz * (t * Ew + 24 * kA55 * (1 + nuw)))
        ) / (b * Ew * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)))

        q23 = 0

        q24 = (
            -36
            * Pi
            * (1 + nuw)
            * (
                (-6 + Pi**2) * (2 * mx - h * (py + qy)) * t * Ew
                + 3 * kA55 * (4 * mx * Pi**2 + b * fy * h * (-8 + Pi**2) * t) * (1 + nuw)
                - 6
                * kB55
                * (-8 * b * fy * t + Pi**2 * (2 * py + 2 * qy + b * fy * t))
                * (1 + nuw)
            )
        ) / (
            b**2 * ((-6 + Pi**2) * t * Ew + 6 * kA55 * Pi**2 * (1 + nuw)) ** 2
            - 18
            * Pi**2
            * (1 + nuw)
            * (
                4 * h * kB55 * (-6 + Pi**2) * t * Ew
                - 4 * kD55 * (-6 + Pi**2) * t * Ew
                + 24 * kB55**2 * Pi**2 * (1 + nuw)
                - kA55 * (h**2 * (-6 + Pi**2) * t * Ew + 24 * kD55 * Pi**2 * (1 + nuw))
            )
        )

        q = np.array(
            [
                [q01],
                [q02],
                [q03],
                [q04],
                [q05],
                [q06],
                [q07],
                [q08],
                [q09],
                [q10],
                [q11],
                [q12],
                [q13],
                [q14],
                [q15],
                [q16],
                [q17],
                [q18],
                [q19],
                [q20],
                [q21],
                [q22],
                [q23],
                [q24],
            ],
            dtype=np.double,
        )
        return q
