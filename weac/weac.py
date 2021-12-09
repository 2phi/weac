#!/usr/bin/env python

"""
Functions library (module) for the main script run.py.

Author: Philipp Rosendahl, mail@2phi.de
Date: 03/2020
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from scipy.optimize import root_scalar, basinhopping, brentq
from scipy.integrate import quad, trapz, romberg

# === HELPER FUNCTIONS =========================================================


def time():
    """
    Return current time in milliseconds.
    """
    return 1e3*timer()


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# === MODEL CLASS ==============================================================


class Layered(object):
    """
    Layered beam on elastic foundation model class.

    Provides geometry, material and loading attributes, and methods for the
    system assembly and solution.
    """

    def __init__(
            self, t=30, rho=240, mat=False, lski=1000,
            nvis=250, tol=1e-3, system='pst', thickness_scaling=1,
            E_base=0., t_base=0., opt=False):
        """Initialize model with user input."""

        # Terminal outputs
        np.set_printoptions(precision=3, linewidth=80, suppress=True)

        # Attributes
        self.g = 9810               # Gravitaiton (mm/s^2)
        # Effective out-of-plane length of skis (mm)
        self.lski = lski
        self.system = system        # Skier, pst
        self.tol = tol              # Relative romberg integration tolerance
        self.nvis = nvis            # Discretization for visualization
        self.thickness_scaling = float(thickness_scaling)
        # Scale thickness of profiles

        # Plot styles and material properties
        self.set_plotstyles()
        if isinstance(opt, (list, tuple, np.ndarray)):
            self.set_foundation_properties(mat=mat, t=t, E=opt[2])
            self.set_beam_properties(mat=mat, C0=opt[0], C1=opt[1])
        else:
            self.set_foundation_properties(mat=mat, t=t)
            self.set_beam_properties(mat=mat, rho=rho)

        # Model setup
        self.calc_foundation_stiffness(E_base, t_base)
        self.calc_laminate_stiffness_matrix()
        self.calc_system_matrix()
        self.calc_eigensystem()

    def set_plotstyles(self):
        """
        Define styles plot markers, labels and colors.
        """
        self.labelstyle = {         # Text style of plot labels
            'backgroundcolor': 'w',
            'horizontalalignment': 'center',
            'verticalalignment': 'center'}
        self.markerstyle = {        # Style of plot markers
            'marker': 'o',
            'markersize': 5,
            'markerfacecolor': 'w',
            'zorder': 3}
        self.colors = np.array([    # TUD color palette
            ['#DCDCDC', '#B5B5B5', '#898989', '#535353'],   # gray
            ['#5D85C3', '#005AA9', '#004E8A', '#243572'],   # blue
            ['#009CDA', '#0083CC', '#00689D', '#004E73'],   # ocean
            ['#50B695', '#009D81', '#008877', '#00715E'],   # teal
            ['#AFCC50', '#99C000', '#7FAB16', '#6A8B22'],   # green
            ['#DDDF48', '#C9D400', '#B1BD00', '#99A604'],   # lime
            ['#FFE05C', '#FDCA00', '#D7AC00', '#AE8E00'],   # yellow
            ['#F8BA3C', '#F5A300', '#D28700', '#BE6F00'],   # sand
            ['#EE7A34', '#EC6500', '#CC4C03', '#A94913'],   # orange
            ['#E9503E', '#E6001A', '#B90F22', '#961C26'],   # red
            ['#C9308E', '#A60084', '#951169', '#732054'],   # magenta
            ['#804597', '#721085', '#611C73', '#4C226A']])  # puple

    def set_foundation_properties(self, mat, t, E=0.25):
        """
        Set material properties of the foundation (weak layer).
        """
        self.weak = {
            'nu': 0.25,         # Poisson's ratio
            'E': E,             # Young's modulus (MPa)
            'Sc': 2.6,          # Compressive strength (kPa)
            'Tc': 0.4}          # Shear strength (kPa)
        self.t = t              # Weak layer thickness

    def set_beam_properties(self, mat, rho=240, C0=6e-10, C1=4.60):
        """
        Set material properties of the beam (slab).
        """
        # UNTERSCHEIDUNG ZWISCHEN .csv, .xlsx, .npy, string und False bei import!

        if isinstance(mat, np.ndarray):
            # Compute elastic properties
            # Global poisson's ratio
            nu = 0.25*np.ones(mat.shape[0])
            E = self.gerling(mat[:, 0], C0=C0, C1=C1)   # Young's modulus
            G = E/(2*(1 + nu))                          # Shear modulus
            # Assemble layering info matrix (bottom to top)
            self.slab = np.flipud(np.vstack([mat.T, E, G, nu]).T)
            # Further outputs
            self.h = sum(mat[:, 1])                 # Total slab thickness
            self.k = 5/6                            # Shear correction factor
            self.orthotropic = True                 # Flag
            self.profile_label = 'input'
            # Center of gravity
            self.zs = self.calc_zs(mat)
        elif not mat:  # Homogeneous default slab
            self.orthotropic = False
            self.h = 200
            self.zs = 0
            self.slab = {'E': 5.23, 'nu': 0.25}
            self.rho = rho
            self.ext = 'db'
        elif mat == 'L2':  # Homogeneous snow layers
            # Top to bottom density and thickness [rho (kg/m^3), t (mm)]
            mat = np.array(
                [[120, 120],
                 [180, 120],
                 [270, 120]])
            # Compute elastic properties
            nu = 0.25*np.ones(mat.shape[0])         # Global poisson's ratio
            E = self.scapozza(mat[:, 0])            # Young's modulus
            G = E/(2*(1 + nu))                      # Shear modulus
            # Assemble layering info matrix (bottom to top)
            self.slab = np.flipud(np.vstack([mat.T, E, G, nu]).T)
            # Further outputs
            self.h = sum(mat[:, 1])                 # Total slab thickness
            self.k = 5/6                            # Shear correction factor
            self.orthotropic = True                 # Flag
            self.ext = 'db'                         # Materials from internal DB
            # Center of gravity
            self.zs = self.calc_zs(mat)
        elif mat.split('.')[0] == 'profile':
            # Load profile
            self.profile_id = mat.split('.')[1].lower()
            self.profile_label = 'Profile ' + self.profile_id
            # Top to bottom density and thickness [rho (kg/m^3), t (mm)]
            mat, E = self.load_profile(self.profile_id)
            mat[:, 1] = mat[:, 1] * self.thickness_scaling  # Apply scaling
            # Compute elastic properties
            nu = 0.25*np.ones(mat.shape[0])         # Global poisson's ratio
            G = E/(2*(1 + nu))                      # Shear modulus
            # Assemble layering info matrix (bottom to top)
            # !! Order is different than layer order
            self.slab = np.flipud(np.vstack([mat.T, E, G, nu]).T)
            # Further outputs
            self.h = sum(mat[:, 1])                 # Total slab thickness
            self.k = 5/6                            # Shear correction factor
            self.orthotropic = True                 # Flag
            self.ext = 'db'                         # Materials from internal DB
            # Center of gravity
            self.zs = self.calc_zs(mat)
        else:
            path, self.ext = mat.split('.')
            if self.ext == 'csv':
                # Import csv
                self.profile = pd.read_csv(mat)
                # Get list of layers and densities (bottom to top)
                layers = self.profile['layer_top'].values
                rho = self.profile['density'].values
                # Calculate layer thicknesses (convert from cm to mm)
                t = 10*np.append(layers[0], layers[1:] - layers[:-1])
                # Provide mat (top to bottom)
                mat = np.fliplr([rho, t]).T
                # Compute elastic properties
                nu = 0.25*np.ones(mat.shape[0])     # Global poisson's ratio
                E = self.scapozza(mat[:, 0])        # Young's modulus
                G = E/(2*(1 + nu))                  # Shear modulus
                # Assemble layering info matrix (bottom to top)
                self.strat = np.flipud(np.vstack([mat.T, E, G, nu]).T)
                # Further outputs
                self.k = 5/6                        # Shear correction factor
                self.orthotropic = True             # Flag
                self.slab = np.empty([2, 6])         # Just initialize
                self.h = False                      # Just initialize

                ## Determine possible weak layers

                # Compute moving average of stiffness profile
                avg = pd.Series(self.strat[:, 2]).rolling(
                    window=5, center=True).mean()
                # Compute difference between actual stiffness and moving average
                diff = self.strat[:, 2] - avg
                # Set leading and trailing NaNs from moving avg calculation to zero
                np.nan_to_num(diff, copy=False, nan=0.0)
                # Normalize to maximum local stiffness decrease
                norm = diff/diff.min()
                # Get indices of stiffness drops of at least 30% of the maximum
                self.weakidx = self.profile.index.values[norm > 0.3]  # bot2top
                # Plot
                plt.plot(self.profile['layer_top'], self.strat[:, 2])
                plt.plot(self.profile['layer_top'], avg)
                norm[norm < 0.3] = 0
                plt.plot(self.profile['layer_top'], 30*norm)
                plt.show()

            elif self.ext == 'xlsx':
                'import excel'
            elif self.ext == 'npy':
                'import numpy'

    def calc_zs(self, mat):
        # Layering info for center of gravity calculation (bottom to top)
        n = mat.shape[0]                        # Number of layers
        rho = self.slab[:, 0]                   # Layer densities
        t = self.slab[:, 1]                     # Layer thicknesses
        # Layer center coordinates (bottom to top)
        zi = [self.h/2 - sum(t[0:j]) - t[j]/2 for j in range(n)]
        # Center of gravity
        return sum(zi*t*rho)/sum(t*rho)

    def load_profile(self, profile_id):
        # profile:
        # Top to bottom density, thickness, Young's modulus
        # [rho (kg/m^3), t (mm), E (N/mm^2)]
        soft = [120.,  120.,   0.3]
        medium = [180.,  120.,   1.5]
        hard = [270.,  120.,   7.5]
        if profile_id == 'a':
            profile = np.array([hard, medium, soft])
        elif profile_id == 'b':
            profile = np.array([soft, medium, hard])
        elif profile_id == 'c':
            profile = np.array([hard, soft, hard])
        elif profile_id == 'd':
            profile = np.array([soft, hard, soft])
        elif profile_id == 'e':
            profile = np.array([hard, soft, soft])
        elif profile_id == 'f':
            profile = np.array([soft, soft, hard])
        elif profile_id == 'hom_soft':
            profile = np.array([soft, soft, soft])
        elif profile_id == 'hom_medium':
            profile = np.array([medium, medium, medium])
        elif profile_id == 'hom_hard':
            profile = np.array([hard, hard, hard])
        elif profile_id == 'test':
            profile = np.array([hard, soft, hard, soft, hard, soft, hard, soft,
                                hard, soft, hard, soft, hard, soft, hard, soft, hard, soft, hard,
                                soft, hard, soft, hard])
        else:
            raise ValueError('This profile is not defined.')
        mat = profile[:, 0:2]
        E = profile[:, 2]
        return mat, E

    def scapozza(self, rho):
        """Compute Young's modulus in MPa from rho in kg/m^3."""
        rho = rho*1e-12                 # Convert to t/mm^3
        rho0 = 917e-12                  # Desity of ice in t/mm^3
        return 5.07e3*(rho/rho0)**5.13   # Young's modulus in MPa

    def gerling(self, rho, C0=6, C1=4.60):
        """Young's modulus (MPa) from rho (kg/m^3) (Gerling et al. 2017)."""
        return C0*1e-10*rho**C1

    def calc_foundation_stiffness(self, E_base, t_base):
        """
        Compute foundation normal and shear stiffness.
        """
        # Plaine strain shear and Young's modulus (MPa)
        G = self.weak['E']/(2*(1 + self.weak['nu']))
        E = self.weak['E']/(1 - self.weak['nu']**2)

        # Foundation stiffnesses
        self.kn = E/self.t   # Normal stiffness (N/mm^3)
        self.kt = G/self.t   # Shear stiffness (N/mm^3)

        if E_base == 0.:
            self.kn_foundation = self.kn
            self.kt_foundation = self.kt
        else:
            nu_base = 0.25
            G_base = E_base/(2.*(1. + nu_base))

            kn_base = E_base/t_base
            kt_base = G_base/t_base

            self.kn_foundation = ((self.kn + kn_base)/(self.kn * kn_base))**-1
            self.kt_foundation = ((self.kt + kt_base)/(self.kt * kt_base))**-1

    def calc_laminate_stiffness_matrix(self):
        """
        Provide ABD matrix and thermal loads.

        Return plane-strain laminate stiffness matrix (ABD) and thermal
        normal force and thermal bending moment.
        """
        if self.orthotropic == True:
            # Number of plies, ply thicknesses
            n = self.slab.shape[0]
            t = self.slab[:, 1]
            # Initialize ply coordinates (bottom to top in laminate CSYS)
            z = np.zeros(n + 1)
            for j in range(n + 1):
                z[j] = -self.h/2 + sum(t[0:j])
            # Initialize stiffness components
            A11, B11, D11, kA55 = 0, 0, 0, 0
            # Add layerwise contributions (sign of B11 to agree with beam CSYS)
            for i in range(n):
                # Add layerwise contributions
                E, G, nu = self.slab[i, 2:5]
                A11 = A11 + E/(1 - nu**2)*(z[i+1] - z[i])
                B11 = B11 - 1/2*E/(1 - nu**2)*(z[i+1]**2 - z[i]**2)
                D11 = D11 + 1/3*E/(1 - nu**2)*(z[i+1]**3 - z[i]**3)
                kA55 = kA55 + self.k*G*(z[i+1] - z[i])
        else:  # homogeneous isotropic
            A11 = self.slab['E']*self.h/(1 - self.slab['nu']**2)
            B11 = 0
            D11 = self.slab['E']*self.h**3/(12*(1 - self.slab['nu']**2))
            kA55 = 5/6*self.slab['E']*self.h/(2*(1 + self.slab['nu']))

        self.A11 = A11
        self.B11 = B11
        self.D11 = D11
        self.kA55 = kA55
        self.E0 = B11**2 - A11*D11

    def calc_system_matrix(self):
        """
        Assemble first-order ODE system matrix.

        Using the solution vector z = [u, u', w, w', psi, psi']
        the ODE system is written in the form Az' + Bz = d
        and rearranged to z' = -(A^-1)Bz + (A^-1)d = Ez + F
        """
        kn = self.kn_foundation
        kt = self.kt_foundation

        # Abbreviations (MIT t/2 im GGW, MIT w' in Kinematik)
        E21 = kt*(-2*self.D11 + self.B11*(self.h + self.t))/(2*self.E0)
        E24 = (2*self.D11*kt*self.t
               - self.B11*kt*self.t*(self.h + self.t)
               + 4*self.B11*self.kA55)/(4*self.E0)
        E25 = (-2*self.D11*self.h*kt
               + self.B11*self.h*kt*(self.h + self.t)
               + 4*self.B11*self.kA55)/(4*self.E0)
        E43 = kn/self.kA55
        E61 = kt*(2*self.B11 - self.A11*(self.h + self.t))/(2*self.E0)
        E64 = (-2*self.B11*kt*self.t
               + self.A11*kt*self.t*(self.h+self.t)
               - 4*self.A11*self.kA55)/(4*self.E0)
        E65 = (2*self.B11*self.h*kt
               - self.A11*self.h*kt*(self.h+self.t)
               - 4*self.A11*self.kA55)/(4*self.E0)

        # System matrix
        E = [[0,    1,    0,    0,    0,    0],
             [E21,    0,    0,  E24,  E25,    0],
             [0,    0,    0,    1,    0,    0],
             [0,    0,  E43,    0,    0,   -1],
             [0,    0,    0,    0,    0,    1],
             [E61,    0,    0,  E64,  E65,    0]]

        self.sysmat = np.array(E)

    def calc_eigensystem(self):
        """
        Run eigenvalue analysis of the system matrix.
        """
        # Calculate eigenvalues (ew) and eigenvectors (ev)
        ew, ev = np.linalg.eig(self.sysmat)
        # Classify real and complex eigenvalues
        real = (ew.imag == 0) & (ew.real != 0)  # real eigenvalues
        complex = ew.imag > 0                   # positive complex conjugates
        # Eigenvalues
        self.ewC = ew[complex]
        self.ewR = ew[real].real
        # Eigenvectors
        self.evC = ev[:, complex]
        self.evR = ev[:, real].real
        # Count eigenvalues
        self.nR = len(self.ewR)
        self.nC = len(self.ewC)
        # Prepare positive eigenvalue shifts for numerical robustness
        self.sR, self.sC = np.zeros(self.ewR.shape), np.zeros(self.ewC.shape)
        self.sR[self.ewR > 0], self.sC[self.ewC > 0] = -1, -1

    def get_weight_load(self, phi):
        """
        Calculate line loads.
        """
        phi = np.deg2rad(phi)           # Convert phi to rad

        # Sum up layer weight loads
        if self.orthotropic:
            q = sum(self.slab[:, 0]*1e-12*self.g*self.slab[:, 1])
        else:
            rho = self.rho*1e-12        # Convert rho to t/mm^3
            q = rho*self.g*self.h       # Line load (N/mm)

        qn = q*np.cos(phi)              # Line load in normal direction
        qt = q*np.sin(phi)              # Line load in tangential direction

        return qn, qt

    def get_skier_load(self, m, phi):
        """
        Calculate skier line load.
        """
        phi = np.deg2rad(phi)
        F = 1e-3*np.array(m)*self.g*1/self.lski     # Total skier load (N)
        Fn = F*np.cos(phi)                          # Normal skier load (N)
        Ft = F*np.sin(phi)                          # Tangential skier load (N)
        return Fn, Ft

    def zh(self, x, l=0, bed=True):
        """
        Compute complementary solution of bedded and free segments at position x.
        """
        if bed:
            return np.concatenate([
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
            H54 = -3*x**2 + 6*self.E0/(self.A11*self.kA55)
            # Complementary solution matrix of free segments
            return np.array(
                [[0,      0,      0,    H14,      1,      x],
                 [0,      0,      0,    H24,      0,      1],
                 [1,      x,   x**2,   x**3,      0,      0],
                 [0,      1,    2*x, 3*x**2,      0,      0],
                 [0,     -1,   -2*x,    H54,      0,      0],
                 [0,      0,     -2,   -6*x,      0,      0]])

    def zp(self, x, phi, bed=True):
        """
        Compute particular integral of bedded and free segments at position x.
        """
        # Get weight load
        qn, qt = self.get_weight_load(phi)

        # Set foundation stiffness variables
        kn = self.kn_foundation
        kt = self.kt_foundation

        # Assemble particular integral vectors
        if bed:
            return np.array([
                [qt/kt + self.h*qt*(self.h + self.t - 2
                                    * self.zs)/(4*self.kA55)],
                [0],
                [qn/kn],
                [0],
                [-qt*(self.h + self.t - 2*self.zs)/(2*self.kA55)],
                [0]])
        else:
            return np.array([
                [(-3*qt/self.A11 - self.B11*qn*x/self.E0)/6*x**2],
                [(-2*qt/self.A11 - self.B11*qn*x/self.E0)/2*x],
                [-self.A11*qn*x**4/(24*self.E0)],
                [-self.A11*qn*x**3/(6*self.E0)],
                [self.A11*qn*x**3/(6*self.E0) + (
                    (self.zs - self.B11/self.A11)*qt - qn*x)/self.kA55],
                [self.A11*qn*x**2/(2*self.E0) - qn/self.kA55]])

    def z(self, C, x, l, phi, bed=True):
        """
        Assemble solution vector at positions x.
        """
        if isinstance(x, (list, tuple, np.ndarray)):
            return np.concatenate([
                np.dot(self.zh(xi, l, bed), C)
                + self.zp(xi, phi, bed) for xi in x], axis=1)
        else:
            return np.dot(self.zh(x, l, bed), C) + self.zp(x, phi, bed)

    def w(self, Z):
        """
        Get centerline deflection w.
        """
        return Z[2, :]

    def wp(self, Z):
        """
        Get first derivative w' of the centerline deflection.
        """
        return Z[3, :]

    def psi(self, Z):
        """
        Get midplane rotation psi.
        """
        return Z[4, :]

    def psip(self, Z):
        """
        Get first derivative psi' of the midplane rotation.
        """
        return Z[5, :]

    def u(self, Z, z0):
        """
        Get horizontal displacement u = u0 + z0 psi.
        """
        return Z[0, :] + z0*self.psi(Z)

    def up(self, Z, z0):
        """
        Get first derivative u' = u0' + z0 psi' of the horizontal displacement.
        """
        return Z[1, :] + z0*self.psip(Z)

    def N(self, Z):
        """
        Get axial normal force N = A11 u' + B11 psi'.
        """
        return self.A11*Z[1, :] + self.B11*Z[5, :]

    def M(self, Z):
        """
        Get bending moment M = B11 u' + D11 psi'.
        """
        return self.B11*Z[1, :] + self.D11*Z[5, :]

    def V(self, Z):
        """
        Get vertical shear force V = kA55 (w' + psi).
        """
        return self.kA55*(Z[3, :] + Z[4, :])

    def sig(self, Z):
        """
        Get normal stress.
        """
        return -self.kn_foundation*self.w(Z)

    def tau(self, Z):
        """
        Get shear stress.
        """
        return self.kt_foundation*(self.wp(Z)*self.t/2 - self.u(Z, z0=self.h/2))

    def eps(self, Z):
        """
        Get normal strain.
        """
        return -self.w(Z)/self.t

    def gamma(self, Z):
        """
        Get shear strain.
        """
        return self.wp(Z)/2 - self.u(Z, z0=self.h/2)/self.t

    def maxp(self, Z):
        """
        Get maximum principal stress.
        """
        sig = self.sig(Z)
        tau = self.tau(Z)
        return np.amax([[sig + np.sqrt(sig**2 + 4*tau**2),
                         sig - np.sqrt(sig**2 + 4*tau**2)]], axis=1)[0]/2

    def Gi(self, Ztip):
        """
        Get mode I differential energy release rate at crack tip.
        """
        return self.sig(Ztip)**2/(2*self.kn)

    def Gii(self, Ztip):
        """
        Get mode II differential energy release rate at crack tip.
        """
        return self.tau(Ztip)**2/(2*self.kt)

    def bc(self, z):
        """
        Provide equations for free (pst) or infinite (skiers) ends.
        """
        if self.system in ['pst', 'pst left']:
            # Free ends
            return np.array([self.N(z), self.M(z), self.V(z)])
        elif self.system in ['skier', 'skiers']:
            # Infinite ends (vanishing complementary solution)
            return np.array([self.u(z, z0=0), self.w(z), self.psi(z)])
        else:
            raise ValueError(
                'Boundary conditions not defined for this system type.')

    def eqs(self, zl, zr, pos='mid'):
        """
        Provide boundary and transmission conditions for beam segments.
        """
        if pos == 'l' or pos == 'left':
            eqs = np.array([
                self.bc(zl)[0],             # Left boundary condition
                self.bc(zl)[1],             # Left boundary condition
                self.bc(zl)[2],             # Left boundary condition
                self.u(zr, z0=0),           # ui(xi = li)
                self.w(zr),                 # wi(xi = li)
                self.psi(zr),               # psii(xi = li)
                self.N(zr),                 # Ni(xi = li)
                self.M(zr),                 # Mi(xi = li)
                self.V(zr)])                # Vi(xi = li)
        elif pos == 'm' or pos == 'mid':
            eqs = np.array([
                -self.u(zl, z0=0),          # -ui(xi = 0)
                -self.w(zl),                # -wi(xi = 0)
                -self.psi(zl),              # -psii(xi = 0)
                -self.N(zl),                # -Ni(xi = 0)
                -self.M(zl),                # -Mi(xi = 0)
                -self.V(zl),                # -Vi(xi = 0)
                self.u(zr, z0=0),           # ui(xi = li)
                self.w(zr),                 # wi(xi = li)
                self.psi(zr),               # psii(xi = li)
                self.N(zr),                 # Ni(xi = li)
                self.M(zr),                 # Mi(xi = li)
                self.V(zr)])                # Vi(xi = li)
        elif pos == 'r' or pos == 'right':
            eqs = np.array([
                -self.u(zl, z0=0),          # -ui(xi = 0)
                -self.w(zl),                # -wi(xi = 0)
                -self.psi(zl),              # -psii(xi = 0)
                -self.N(zl),                # -Ni(xi = 0)
                -self.M(zl),                # -Mi(xi = 0)
                -self.V(zl),                # -Vi(xi = 0)
                self.bc(zr)[0],             # Right boundary condition
                self.bc(zr)[1],             # Right boundary condition
                self.bc(zr)[2]])            # Right boundary condition
        else:
            sys.exit('Input error: position argument ' + pos + ' given. '
                     'Valid segment positions are '
                     + 'l, m, and r, or left, mid and right.')

        return eqs

    def calc_segments(self, li=False, mi=False, ki=False, k0=False,
                      L=1e4, a=0, m=0, **kwargs):
        """
        Provide length (li), foundation (ki, k0), and skier weight (mi) lists.
        """
        if self.system == 'skiers':
            li = np.array(li)                           # Segment lengths
            mi = np.array(mi)                           # Skier weights
            ki = np.array(ki)                           # Crack
            k0 = np.array(k0)                           # No crack
        elif self.system == 'pst':
            li = np.array([L - a, a])                   # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([True, False])                # No crack
            k0 = np.array([True, True])                 # Crack
        elif self.system == 'pst left':
            li = np.array([a, L - a])                   # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([False, True])                # No crack
            k0 = np.array([True, True])                 # Crack
        elif self.system == 'skier':
            lb = (L - a)/2                              # Half bedded length
            lf = a/2                                    # Half free length
            li = np.array([lb, lf, lf, lb])             # Segment lengths
            mi = np.array([0, m, 0])                    # Skier weights
            ki = np.array([True, False, False, True])   # No crack
            k0 = np.array([True, True, True, True])     # Crack
        else:
            sys.exit('Input error: system not implemented.')

        # Fill dictionary
        segments = {
            'nocrack': {'li': li, 'mi': mi, 'ki': k0},
            'crack': {'li': li, 'mi': mi, 'ki': ki},
            'both': {'li': li, 'mi': mi, 'ki': ki, 'k0': k0}}

        return segments

    def assemble_and_solve(self, phi, li, mi, ki):
        """
        Compute free constants for arbitrary beam assembly.

        Assemble LHS from bedded and free segments in the form
        [  ]   [ zh1  0   0  ...  0   0   0  ][   ]   [    ]   [     ]  left
        [  ] = [ zh1 zh2  0  ...  0   0   0  ][   ] + [    ] = [     ]  mid
        [  ]   [  0  zh2 zh3 ...  0   0   0  ][   ]   [    ]   [     ]  mid
        [z0]   [ ... ... ... ... ... ... ... ][ C ]   [ zp ]   [ rhs ]  mid
        [  ]   [  0   0   0  ... zhL zhM  0  ][   ]   [    ]   [     ]  mid
        [  ]   [  0   0   0  ...  0  zhM zhN ][   ]   [    ]   [     ]  mid
        [  ]   [  0   0   0  ...  0   0  zhN ][   ]   [    ]   [     ]  right
        and solve for constants C.
        """
        # --- CATCH ERRORS -----------------------------------------------------

        if not any(ki):
            sys.exit('Input error: Provide at least one bedded segment.')

        if len(li) != len(ki) or len(li)-1 != len(mi):
            sys.exit('Input error: Make sure len(li)=N, len(ki)=N, '
                     + 'and len(mi)=N-1 for a system of N segments.')

        if self.system not in ['pst', 'pst left']:
            if not all([ki[0], ki[-1]]):
                sys.exit('Input error: Provide bedded boundary segments in '
                         + 'order to account for infinite extensions.')

            if li[0] < 5e3 or li[-1] < 5e3:
                print('WARNING: Boundary segments are short. Make sure the '
                      + 'complementary solution has decayed to the boundaries.')

        # --- PREPROCESSING ----------------------------------------------------

        # Determine size of linear system of equations
        nS = len(li)            # Number of beam segments
        nDOF = 6                # Number of free constants per segment

        # Add dummy segment if only one segment provided
        if nS == 1:
            li.append(0)
            ki.append(True)
            mi.append(0)
            nS = 2

        # Assemble position vector
        pi = np.full(nS, 'm')
        pi[0], pi[-1] = 'l', 'r'

        # Initialize matrices
        zh0 = np.zeros([nS*6, nS*nDOF])
        zp0 = np.zeros([nS*6, 1])
        rhs = np.zeros([nS*6, 1])

        # --- ASSEMBLE LINEAR SYSTEM OF EQUATIONS ------------------------------

        # Loop through segments to assemble left-hand side
        for i in range(nS):
            # Length, foundation and position of segment i
            l, k, pos = li[i], ki[i], pi[i]
            # Transmission conditions at left and right segment ends
            zhi = self.eqs(
                zl=self.zh(x=0, l=l, bed=k),
                zr=self.zh(x=l, l=l, bed=k),
                pos=pos)
            zpi = self.eqs(
                zl=self.zp(x=0, phi=phi, bed=k),
                zr=self.zp(x=l, phi=phi, bed=k),
                pos=pos)
            # Rows for left-hand side assembly
            start = 0 if i == 0 else 3
            stop = 6 if i == nS-1 else 9
            # Assemble left-hand side
            zh0[(6*i-start):(6*i+stop), i*nDOF:(i+1)*nDOF] = zhi
            zp0[(6*i-start):(6*i+stop)] += zpi

        # Loop through loads to assemble right-hand side
        for i, m in enumerate(mi, start=1):
            # Get skier loads
            Fn, Ft = self.get_skier_load(m, phi)
            # Right-hand side for transmission from segment i-1 to segment i
            rhs[6*i:6*i+3] = np.vstack([Ft, -Ft*self.h/2, Fn])

        # Set rhs so that complementary integral vanishes at boundaries
        if self.system not in ['pst', 'pst left']:
            rhs[:3] = self.bc(self.zp(x=0, phi=phi, bed=ki[0]))
            rhs[-3:] = self.bc(self.zp(x=li[-1], phi=phi, bed=ki[-1]))

        # --- SOLVE ------------------------------------------------------------

        # Solve z0 = zh0*C + zp0 = rhs for constants, i.e. zh0*C = rhs - zp0
        C = np.linalg.solve(zh0, rhs - zp0)

        # Sort (nDOF = 6) constants for each segment into columns of a matrix
        return C.reshape([-1, nDOF]).T

    def rasterize_solution(self, C, phi, li, ki, **kwargs):
        """
        Compute rasterized solution vector.
        """
        # Drop zero-length segments
        isnonzero = li > 0
        C, ki, li = C[:, isnonzero], ki[isnonzero], li[isnonzero]

        # Compute number of plot points per segment (+1 for last segment)
        nq = np.ceil(li/li.sum()*self.nvis).astype('int')
        nq[-1] += 1

        # Provide cumulated length and plot point lists
        lic = np.insert(np.cumsum(li), 0, 0)
        nqc = np.insert(np.cumsum(nq), 0, 0)

        # Initialize arrays
        isbedded = np.full(nq.sum(), True)
        xq = np.full(nq.sum(), np.nan)
        zq = np.full([6, xq.size], np.nan)

        # Loop through segments
        for i, l in enumerate(li):
            # Get local x-coordinates of segment i
            xi = np.linspace(0, l, num=nq[i], endpoint=(i == li.size - 1))
            # Compute start and end coordinates of segment i
            x0 = lic[i]
            # Assemble global coordinate vector
            xq[nqc[i]:nqc[i+1]] = x0 + xi
            # Mask coordinates not on foundation (excluding endpoints)
            if not ki[i]:
                isbedded[nqc[i]+1:nqc[i+1]] = False
            # Compute segment solution
            zi = self.z(C[:, [i]], xi, l, phi, ki[i])
            # Assemble global solution matrix
            zq[:, nqc[i]:nqc[i+1]] = zi

        # Add masking of consecutive unbedded segments
        isrepeated = [ki[j] or ki[j+1] for j, _ in enumerate(ki[:-1])]
        for i, truefalse in enumerate(isrepeated, start=1):
            isbedded[nqc[i]] = truefalse

        # Assemble vector of coordinates on foundation
        xb = np.full(nq.sum(), np.nan)
        xb[isbedded] = xq[isbedded]

        return xq, zq, xb

    def ginc(self, C0, C1, phi, li, ki, k0, **kwargs):
        """
        Compute incremental energy relase rate of of all crack increments.
        """
        # TODO is this correct with base?
        # Did we account for kn_foundation correctly?

        # Make sure inputs are np.arrays
        li, ki, k0 = np.array(li), np.array(ki), np.array(k0)

        # Reduce inputs to segments with crack advance
        iscrack = k0 & ~ki
        C0, C1, li = C0[:, iscrack], C1[:, iscrack], li[iscrack]

        # Compute total crack lenght and initialize outputs
        da = li.sum() if li.sum() > 0 else np.nan
        Ginc1, Ginc2 = 0, 0

        # Loop through segments with crack advance
        for j, l in enumerate(li):
            # Uncracked (0) and cracked (1) solutions at integration points xi
            def z0(x): return self.z(C0[:, [j]], x, l, phi, bed=True)
            def z1(x): return self.z(C1[:, [j]], x, l, phi, bed=False)
            # # Mode I (1) and II (2) integrands at integration points
            # int1 = lambda x: self.sig(z0(x))*self.w(z1(x))
            # int2 = lambda x: self.tau(z0(x))*self.u(z1(x), z0=self.h/2)
            #  # Segement contributions to total crack opening integral
            # Ginc1 += -romberg(int1, 0, l, rtol=self.tol, vec_func=True)/(2*da)
            # Ginc2 += -romberg(int2, 0, l, rtol=self.tol, vec_func=True)/(2*da)
            # Mode I (1) and II (2) integrands at integration points
            def int1(x): return self.sig(z0(x))*self.eps(z1(x))*self.t
            def int2(x): return self.tau(z0(x))*self.gamma(z1(x))*self.t
            # Segement contributions to total crack opening integral
            Ginc1 += romberg(int1, 0, l, rtol=self.tol, vec_func=True)/(2*da)
            Ginc2 += romberg(int2, 0, l, rtol=self.tol, vec_func=True)/(2*da)

        return np.array([Ginc1 + Ginc2, Ginc1, Ginc2]).flatten()

    def gdif(self, C, phi, li, ki, **kwargs):
        """
        Compute differential energy release rate of all crack tips.
        """
        # Get number and indices of segment transitions
        ntr = len(li) - 1
        itr = np.arange(ntr)

        # Identify bedded-free and free-bedded transitions as crack tips
        iscracktip = [ki[j] != ki[j+1] for j in range(ntr)]

        # Transition indices of crack tips and total number of crack tips
        ict = itr[iscracktip]
        nct = len(ict)

        # Initialize energy release rate array
        Gdif = np.zeros([3, nct])

        # Compute energy relase rate of all crack tips
        for j, idx in enumerate(ict):
            # Solution at crack tip
            z = self.z(C[:, [idx]], li[idx], li[idx], phi, bed=ki[idx])
            # Mode I and II differential energy release rates
            Gdif[1:, j] = self.Gi(z), self.Gii(z)

        # Sum mode I and II contributions
        Gdif[0, :] = Gdif[1, :] + Gdif[2, :]

        # Adjust contributions for center cracks
        avgmask = np.full(nct, True)        # Initialize mask
        avgmask[[0, -1]] = ki[[0, -1]]      # Do not weight edge cracks
        Gdif[:, avgmask] *= 0.5             # Weigth with half crack length

        # Return total differential energy release rate of all crack tips
        return Gdif.sum(axis=1)

    def stress_criterion(self, x, C, phi, li, ki, crit='quads', **kwargs):
        """
        Evaluate the stress criterion at locations x.
        """
        # Unpack strengths (given in kPa)
        Sc = self.weak['Sc']
        Tc = self.weak['Tc']

        # Make sure x is np.ndarray and determine its size
        x = np.asarray([x]) if np.isscalar(x) else np.asarray(x)
        n = x.size

        # Compute cumulate lengths list and segment index of all x
        lic = np.cumsum(li)
        nsegment = np.searchsorted(lic, x)

        # Calculate global x coordinate of left segment ends and init stresses
        x0 = np.insert(lic, 0, 0)
        sig = np.zeros(n)
        tau = np.zeros(n)

        # Compute stresses at x and convert to kPa
        for i, seg in enumerate(nsegment):
            z0 = self.z(C[:, [seg]], x[i] - x0[seg], li[seg], phi, bed=ki[seg])
            sig[i] = 1e3*max([0, self.sig(z0)])
            tau[i] = 1e3*self.tau(z0)

        # Evaluate stress criterion
        if crit == 'quads':
            return np.sqrt((sig/Sc)**2 + (tau/Tc)**2) - 1

    def find_roots(self, L, C, phi, nintervals=50, **kwargs):
        """
        Find all points where the stress criterion is satisfied identically.
        """
        # Subdivide domain into intervals
        intvls = np.linspace(0, L, num=nintervals+1)
        roots = []

        # See if we can find roots in given intervals
        for i in range(nintervals):
            try:
                roots.append(root_scalar(
                    self.stress_criterion, method='brentq',
                    args=(C, phi, kwargs['li'], kwargs['ki']),
                    bracket=intvls[i:i+2], xtol=1e-1).root)
            except ValueError:
                pass

        # Determine index to insert skier position
        iskier = np.searchsorted(roots, L/2)
        # Add domain ends and skier position to points of interest
        poi = np.unique(np.insert(roots, [0, iskier, len(roots)], [0, L/2, L]))
        # Compute new segment lengths
        li = poi[1:] - poi[:-1]
        # Compute new segment midpoints
        xmid = (poi[1:] + poi[:-1])/2
        # Compute new foundation order
        ki = self.stress_criterion(
            xmid, C, phi, kwargs['li'], kwargs['ki']) < 0
        k0 = np.full_like(ki, True)
        # Compute new position of skier weight
        lic = np.cumsum(li)[:-1]
        iskier = np.searchsorted(lic, L/2)
        mi = np.insert(np.zeros_like(lic), iskier, kwargs['mi'].sum())
        print(lic)
        print(mi)

        # Assemble update segmentation as output
        segments = {
            'nocrack': {'li': li, 'mi': mi, 'ki': k0},
            'crack': {'li': li, 'mi': mi, 'ki': ki},
            'both': {'li': li, 'mi': mi, 'ki': ki, 'k0': k0}}

        return segments

    def energy_criterion(self):
        """
        Evaluate the energy criterion for a crack da = [xstart, xstop].
        """
        pass

    def external_potential(self, C, phi, **segments):
        """
        Compute total external potential (PST or skier-on-slab setup).
        """
        # Rasterize solution
        xq, zq, xb = self.rasterize_solution(C=C, phi=phi, **segments)
        # Compute displacements where weight loads are applied
        w0 = self.w(zq)
        us = self.u(zq, z0=self.zs)
        # Get weight loads
        qn, qt = self.get_weight_load(phi)
        # Integrate external work
        Wext = trapz(qn*w0 + qt*us, xq)

        if self.system == 'skier':
            # Get skier weight and length of left free beam segment
            m, l2 = segments['mi'][1], segments['li'][1]
            # Compute solution at the skier's location
            z0 = self.z(C=C[:, [1]], x=l2, l=l2, phi=phi, bed=False)
            # Compute skier force
            Fn, Ft = self.get_skier_load(m, phi)
            # Add external work of the skier loading
            Wext += Fn*self.w(z0) + Ft*self.u(z0, z0=-self.h/2)
        elif self.system != 'pst':
            sys.exit('Input error: Only skier-on-slab and PST setups '
                     + 'implemented at the moment.')

        # Return potential of external forces Pext = - Wext
        return -Wext

    def internal_potential(self, C, phi, **segments):
        """
        Compute total internal potential (PST or skier-on-slab setup).
        """
        # Rasterize solution
        xq, zq, xb = self.rasterize_solution(C=C, phi=phi, **segments)
        # Compute section forces
        N, M, V = self.N(zq), self.M(zq), self.V(zq)
        # Compute stored energy of the slab (beam)
        Pint = trapz(N**2/self.A11 + M**2/self.D11 + V**2/self.kA55, xq)/2

        # Drop parts of the solution that are not a foundation
        zweak = zq[:, ~np.isnan(xb)]
        xweak = xb[~np.isnan(xb)]

        if self.system == 'pst':
            # Compute displacments of segment on foundation
            # w = self.w(zweak)
            # u = self.u(zweak, z0=self.h/2)
            eps = self.eps(zweak)
            gamma = self.gamma(zweak)
            sig = self.sig(zweak)
            tau = self.tau(zweak)
            # Compute stored energy of the weak layer (foundation)
            # Pint += trapz(self.kn*w**2 + self.kt*u**2, xweak)/2
            Pint += 1/2*trapz(sig*eps + tau*gamma, xweak)*self.t
        elif self.system == 'skier':
            # Split left and right bedded segments
            zl, zr = np.array_split(zweak, 2, axis=1)
            xl, xr = np.array_split(xweak, 2)
            # Compute displacements on left and right foundations
            wl, wr = self.w(zl), self.w(zr)
            ul, ur = self.u(zl, z0=self.h/2), self.u(zr, z0=self.h/2)
            # Compute stored energy of the weak layer (foundation)
            Pint += trapz(self.kn*wl**2 + self.kt*ul**2, xl)/2
            Pint += trapz(self.kn*wr**2 + self.kt*ur**2, xr)/2
        else:
            sys.exit('Input error: Only skier-on-slab and PST setups '
                     + 'implemented at the moment.')

        return Pint

    def plot_data(
            self, name, ax1data, ax1label,
            ax2data=None, ax2label=None,
            labelpos=None, vlines=True,
            li=False, mi=False, ki=False,
            xlabel=r'Horizontal position $x$ (cm)'):
        """
        Plot data.
        """
        # Clear figure
        plt.clf()

        # Font setup
        plt.rc('font', family='serif', size=12)
        plt.rc('text', usetex=True)

        # Create figure
        plt.axis()
        ax1 = plt.gca()

        # Axis limits
        ax1.autoscale(axis='x', tight=True)

        # Set axis labels
        ax1.set_xlabel(xlabel + r' $\longrightarrow$')
        ax1.set_ylabel(ax1label + r' $\longrightarrow$')

        # Plot x-axis
        ax1.axhline(0, linewidth=0.5, color='gray')

        # Plot vertical separators
        if vlines:
            ax1.axvline(0, linewidth=0.5, color='gray')
            for i, f in enumerate(ki):
                if not f:
                    ax1.axvspan(sum(li[:i])/10, sum(li[:i+1])/10,
                                facecolor='gray', alpha=0.05, zorder=100)
            for i, m in enumerate(mi, start=1):
                if m > 0:
                    ax1.axvline(sum(li[:i])/10, linewidth=0.5, color='gray')
        else:
            ax1.autoscale(axis='y', tight=True)

        # Calculate labelposition
        if not labelpos:
            x = ax1data[0][0]
            labelpos = int(0.9*len(x[~np.isnan(x)]))

        # Fill left y-axis
        i = -2
        for x, y, label in ax1data:
            i += 1
            if label == '' or 'FEA' in label:
                # line, = ax1.plot(x, y, 'k:', linewidth=1)
                prep, = ax1.plot(x, y, linewidth=3, color='white')
                line, = ax1.plot(x, y, ':', linewidth=1)  # , color='black'
                thislabelpos = -2
                x, y = x[~np.isnan(x)], y[~np.isnan(x)]
                xtx = (x[thislabelpos - 1] + x[thislabelpos])/2
                ytx = (y[thislabelpos - 1] + y[thislabelpos])/2
                ax1.text(xtx, ytx, label, color=line.get_color(),
                         **self.labelstyle)
            else:
                # Plot line
                prep, = ax1.plot(x, y, linewidth=3, color='white')
                line, = ax1.plot(x, y, linewidth=1)
                # Line label
                x, y = x[~np.isnan(x)], y[~np.isnan(x)]
                if len(x) > 0:
                    xtx = (x[labelpos - 1 + 10*i] + x[labelpos + 10*i])/2
                    ytx = (y[labelpos - 1 + 10*i] + y[labelpos + 10*i])/2
                    ax1.text(xtx, ytx, label, color=line.get_color(),
                             **self.labelstyle)

        # Fill right y-axis
        if ax2data:
            # Create right y-axis
            ax2 = ax1.twinx()
            # Set axis label
            ax2.set_ylabel(ax2label + r' $\longrightarrow$')
            # Fill
            for x, y, label in ax2data:
                # Plot line
                prep, = ax2.plot(x, y, linewidth=3, color='white')
                line, = ax2.plot(x, y, linewidth=1, color=self.colors[8, 0])
                # Line label
                x, y = x[~np.isnan(x)], y[~np.isnan(x)]
                xtx = (x[labelpos - 1] + x[labelpos])/2
                ytx = (y[labelpos - 1] + y[labelpos])/2
                ax2.text(xtx, ytx, label, color=line.get_color(),
                         **self.labelstyle)

        # Save figure
        filename = name + '.pdf'
        if isnotebook():
            plt.show()
        else:
            print('Rendering', filename, '...')
            plt.savefig('plots/' + filename, bbox_inches='tight')

    def return_displacements_for_plot(self, x, z):
        data = [[x/10, self.u(z, z0=0), r'$u_0$'],
                [x/10, -self.w(z), r'$w$'],
                [x/10, np.rad2deg(self.psi(z)), r'$\psi$']]
        return data

    def plot_displacements(self, x, z, **segments):
        """
        Wrapper for dispalcements plot.
        """
        data = self.return_displacements_for_plot(x, z)
        self.plot_data(ax1label=r'Displacements (mm)', ax1data=data,
                       name='disp', **segments)

    def plot_sectionforces(self, x, z, **segments):
        """
        Wrapper for section forces plot.
        """
        data = [[x/10, self.N(z), r'$N$'],
                [x/10, self.M(z), r'$M$'],
                [x/10, self.V(z), r'$V$']]
        self.plot_data(ax1label=r'Section forces', ax1data=data,
                       name='forc', **segments)

    def return_stresses_for_plot(self, x, z):
        data = [[x/10, 1e3*self.tau(z), r'$\tau$'],
                [x/10, 1e3*self.sig(z), r'$\sigma$']]
        return data

    def plot_stresses(self, x, z, **segments):
        """
        Wrapper for stress plot.
        """
        data = self.return_stresses_for_plot(x, z)
        self.plot_data(ax1label=r'Stress (kPa)', ax1data=data,
                       name='stress', **segments)

    def plot_criteria(self, x, stress, **segments):
        """
        Wrapper for plot of stress and energy criteria.
        """
        data = [[x/10, stress, r'$\sigma/\sigma_\mathrm{c}$']]
        # [x/10, 1e3*self.sig(z), r'$\sigma$']]
        self.plot_data(ax1label=r'Criteria', ax1data=data,
                       name='crit', **segments)

    def plot_err(self, da, Gdif, Ginc, mode=0):
        """
        Wrapper for energy release rate plot.
        """
        data = [
            [da/10, 1e3*Gdif[mode, :], r'$\mathcal{G}$'],
            [da/10, 1e3*Ginc[mode, :], r'$\bar{\mathcal{G}}$']]
        # [da/10, 1e3*Ginc['int'][mode, :], r'$\bar{\mathcal{G}}_\mathrm{coi}$']]
        # [da/10, 1e3*Gbar['pia'], r'$\bar{\mathcal{G}}_\mathrm{pia}$'],
        # [da/10, 1e3*Gbar['pii'], r'$\bar{\mathcal{G}}_\mathrm{pii}$']]
        self.plot_data(
            xlabel=r'Crack length $\Delta a$ (cm)',
            ax1label=r'Energy release rate (J/m$^2$)',
            ax1data=data, name='err', vlines=False)

    def plot_modes(self, da, G, type='inc'):
        """
        Wrapper for energy release rate plot.
        """
        label = r'$\bar{\mathcal{G}}$' if type == 'inc' else r'$\mathcal{G}$'
        data = [
            [da/10, 1e3*G[0, :], label + r'$_\mathrm{I+I\!I}$'],
            [da/10, 1e3*G[1, :], label + r'$_\mathrm{I}$'],
            [da/10, 1e3*G[2, :], label + r'$_\mathrm{I\!I}$']]
        self.plot_data(
            xlabel=r'Crack length $\Delta a$ (cm)',
            ax1label=r'Energy release rate (J/m$^2$)',
            ax1data=data, name='modes', vlines=False)

    def plot_fea_disp(self, xq, zq, fea):
        """
        Wrapper for dispalcements plot.
        """
        data = [[fea[:, 0]/10, -np.flipud(fea[:, 1]), r'FEA $u_0$'],
                [fea[:, 0]/10, np.flipud(fea[:, 2]), r'FEA $w_0$'],
                # [fea[:, 0]/10, -np.flipud(fea[:, 3]), r'FEA $u(z=-h/2)$'],
                # [fea[:, 0]/10, np.flipud(fea[:, 4]), r'FEA $w(z=-h/2)$'],
                [fea[:, 0]/10,
                    np.flipud(np.rad2deg(fea[:, 5])), r'FEA $\psi$'],
                [xq/10, self.u(zq, z0=0), r'$u_0$'],
                [xq/10, -self.w(zq), r'$w$'],
                [xq/10, np.rad2deg(self.psi(zq)), r'$\psi$']]
        self.plot_data(
            ax1label=r'Displacements (mm)', ax1data=data, name='fea_disp',
            labelpos=-50)

    def plot_fea_stress(self, xb, zb, fea):
        """
        Wrapper for stress plot.
        """
        data = [  # [fea[:, 0]/10, 1e3*np.flipud(fea[:, 1]), r'FEA $\sigma_1$'],
                [fea[:, 0]/10, 1e3*np.flipud(fea[:, 2]), r'FEA $\sigma_2$'],
                [fea[:, 0]/10, 1e3*np.flipud(fea[:, 3]), r'FEA $\tau_{12}$'],
                [xb/10, 1e3*self.tau(zb), r'$\tau$'],
                [xb/10, 1e3*self.sig(zb), r'$\sigma$']]
        self.plot_data(ax1label=r'Stress (kPa)', ax1data=data, name='fea_stress',
                       labelpos=-50)
